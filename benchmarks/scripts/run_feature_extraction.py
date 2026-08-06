#!/usr/bin/env python
"""Feature extraction pipeline for WSI datasets.

Extracts patch-level features from slides using pathology foundation models
via the TRIDENT library. Dataset-specific paths and settings are loaded from
a dataset configuration YAML.

Examples
--------
uv run --package autobench python benchmarks/scripts/run_feature_extraction.py --dataset ovarian --gpu 0
uv run --package autobench python benchmarks/scripts/run_feature_extraction.py --dataset clwd --all_gpus
uv run --package autobench python benchmarks/scripts/run_feature_extraction.py --dataset ovarian --models conch_v15 uni_v2
"""

import argparse
import gc
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
from dotenv import load_dotenv

from autobench.config import load_dataset_config


def parse_args():
    parser = argparse.ArgumentParser(description="Feature extraction for WSI datasets")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset config name (e.g., 'ovarian', 'clwd') or path to YAML")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index to use")
    parser.add_argument("--all_gpus", action="store_true",
                        help="Use all available GPUs — assigns one model per GPU concurrently.")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Subset of encoder keys to run. Default: all from dataset config.")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size for feature extraction (default: from dataset config)")
    parser.add_argument("--skip_seg", action="store_true",
                        help="Skip segmentation and patching (if already done)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit to first N slides (for testing)")
    return parser.parse_args()


def load_encoder(encoder_key: str):
    """Load an encoder by its registry key."""
    if encoder_key == "h0_mini":
        from autobench.encoders.h0_mini import H0MiniInferenceEncoder
        return H0MiniInferenceEncoder()
    else:
        from trident.patch_encoder_models import encoder_factory
        return encoder_factory(encoder_key)


def _segment_on_gpu(wsi_csv: str, output_dir: str, wsi_dir: str,
                    max_workers: int | None = None) -> str:
    """Worker function: run tissue segmentation on the GPU bound by gpu_init."""
    import gc
    import torch

    device = "cuda:0"
    gpu_label = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    print(f"[GPU {gpu_label}] Starting tissue segmentation")

    try:
        from trident import Processor
        from trident.segmentation_models import segmentation_model_factory

        processor = Processor(
            job_dir=output_dir,
            wsi_source=wsi_dir,
            custom_list_of_wsis=wsi_csv,
            skip_errors=True,
            max_workers=max_workers,
        )

        seg_model = segmentation_model_factory("hest")
        seg_model.to(device)
        processor.run_segmentation_job(segmentation_model=seg_model, device=device)

        del seg_model
        torch.cuda.empty_cache()
        gc.collect()
        print(f"[GPU {gpu_label}] Segmentation complete")
        return gpu_label
    except Exception as exc:
        print(f"[GPU {gpu_label}] Segmentation FAILED: {exc!r}")
        raise


def _extract_single_model(
    encoder_key: str,
    wsi_csv: str,
    coords_dir: str,
    batch_size: int,
    output_dir: str,
    wsi_dir: str,
    max_workers: int | None = None,
) -> tuple[str, str | None]:
    """Worker function: extract features for one model on the GPU bound by gpu_init."""
    import gc
    import torch

    device = "cuda:0"
    print(f"[GPU {os.environ.get('CUDA_VISIBLE_DEVICES')}] Starting {encoder_key}")

    try:
        from trident import Processor

        processor = Processor(
            job_dir=output_dir,
            wsi_source=wsi_dir,
            custom_list_of_wsis=wsi_csv,
            skip_errors=True,
            max_workers=max_workers,
        )

        encoder = load_encoder(encoder_key)
        encoder.eval()
        encoder.to(device)

        features_dir = processor.run_patch_feature_extraction_job(
            coords_dir=coords_dir,
            patch_encoder=encoder,
            device=device,
            batch_limit=batch_size,
        )
        print(f"[GPU {os.environ.get('CUDA_VISIBLE_DEVICES')}] Completed {encoder_key} -> {features_dir}")
        return (encoder_key, features_dir)
    except torch.cuda.OutOfMemoryError:
        print(f"[GPU {os.environ.get('CUDA_VISIBLE_DEVICES')}] OOM on {encoder_key}, skipping")
        return (encoder_key, None)
    finally:
        gc.collect()
        torch.cuda.empty_cache()


def main():
    args = parse_args()

    # Load environment variables
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    load_dotenv(env_path)

    # Load dataset configuration
    ds = load_dataset_config(args.dataset)
    print(f"Loaded dataset config: {ds.name}")

    # Dataset paths
    output_dir = ds.output_dir
    wsi_dir = ds.wsi_dir
    mapping_csv = ds.mapping_csv
    mag = ds.magnification
    patch_size = ds.patch_size
    batch_size = args.batch_size or ds.batch_size

    # HuggingFace login
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token, add_to_git_credential=False)
        print("Logged in to HuggingFace Hub.")
    else:
        print("Warning: HF_TOKEN not found. Gated model downloads may fail.")

    # Worker count — respect SLURM CPU allocation over raw os.cpu_count()
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    num_cpus = int(slurm_cpus) if slurm_cpus else (os.cpu_count() or 16)
    max_workers = max(1, num_cpus - 2)  # leave headroom for main + GPU threads
    print(f"DataLoader workers: {max_workers} (from {num_cpus} CPUs)")

    # GPU setup
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if args.all_gpus:
            gpu_ids = list(range(num_gpus))
            device = f"cuda:{gpu_ids[0]}" if gpu_ids else "cpu"
            print(f"Using all available GPUs: {gpu_ids}")
        else:
            if args.gpu >= num_gpus:
                print(f"Error: --gpu {args.gpu} invalid; detected GPUs: 0..{num_gpus - 1}")
                sys.exit(1)
            gpu_ids = [args.gpu]
            device = f"cuda:{args.gpu}"
            print(f"Using single GPU: {args.gpu}")
    else:
        gpu_ids = []
        device = "cpu"
        print("CUDA not available. Using CPU.")

    # Determine encoder keys
    all_encoder_keys = list(ds.encoder_models.values())
    if args.models:
        encoder_keys = [k for k in args.models if k in all_encoder_keys]
        invalid = [k for k in args.models if k not in all_encoder_keys]
        if invalid:
            print(f"Warning: unknown encoder keys ignored: {invalid}")
        if not encoder_keys:
            print("Error: no valid encoder keys specified.")
            sys.exit(1)
    else:
        encoder_keys = all_encoder_keys

    # Load and filter slides
    from autobench.data import load_all_slides, validate_slides, generate_wsi_list_csv

    print("Loading slides from mapping CSV...")
    filtered_df = load_all_slides(mapping_csv, ds)
    if args.limit:
        filtered_df = filtered_df.head(args.limit)
    print(f"Found {len(filtered_df)} slides to process.")

    print("Validating slides...")
    filtered_df, skipped = validate_slides(filtered_df, wsi_dir, ds)
    if skipped:
        print(f"Skipped {len(skipped)} corrupted slides: {skipped[:10]}")
    print(f"{len(filtered_df)} valid slides remaining.")

    # Generate WSI list CSV
    wsi_csv = os.path.join(output_dir, "slide_list.csv")
    generate_wsi_list_csv(filtered_df, wsi_csv, ds)
    print(f"WSI list written to {wsi_csv}")

    # Create TRIDENT Processor
    from trident import Processor

    processor = Processor(
        job_dir=output_dir,
        wsi_source=wsi_dir,
        custom_list_of_wsis=wsi_csv,
        skip_errors=True,
        max_workers=max_workers,
    )

    if not args.skip_seg:
        if args.all_gpus and len(gpu_ids) > 1:
            _run_multigpu_segmentation(gpu_ids, wsi_csv, output_dir, wsi_dir,
                                       max_workers=max_workers)
            processor = Processor(
                job_dir=output_dir,
                wsi_source=wsi_dir,
                custom_list_of_wsis=wsi_csv,
                skip_errors=True,
                max_workers=max_workers,
            )
        else:
            from trident.segmentation_models import segmentation_model_factory

            print("Running tissue segmentation...")
            seg_model = segmentation_model_factory("hest")
            seg_model.to(device)
            processor.run_segmentation_job(segmentation_model=seg_model, device=device)
            del seg_model
            torch.cuda.empty_cache()
            gc.collect()
            print("Segmentation complete.")

        print(f"Running patching at {mag}x, {patch_size}px...")
        coords_dir = processor.run_patching_job(
            target_magnification=mag,
            patch_size=patch_size,
        )
        print(f"Patching complete. Coords dir: {coords_dir}")
    else:
        coords_dir = os.path.join(output_dir, f"{mag}x_{patch_size}px_0px_overlap")
        print(f"Skipping segmentation/patching. Using coords dir: {coords_dir}")

    # Run feature extraction
    if args.all_gpus and len(gpu_ids) > 1:
        _run_multigpu(encoder_keys, gpu_ids, wsi_csv, coords_dir, batch_size,
                      output_dir, wsi_dir, max_workers=max_workers)
    else:
        _run_single_gpu(encoder_keys, processor, device, coords_dir, batch_size)

    print("\nAll feature extractions complete.")


def _run_single_gpu(encoder_keys, processor, device, coords_dir, batch_size):
    """Run models sequentially on a single GPU."""
    for encoder_key in encoder_keys:
        print(f"\n{'='*60}")
        print(f"Loading encoder: {encoder_key}")
        print(f"{'='*60}")

        encoder = load_encoder(encoder_key)
        encoder.eval()
        encoder.to(device)

        features_dir = processor.run_patch_feature_extraction_job(
            coords_dir=coords_dir,
            patch_encoder=encoder,
            device=device,
            batch_limit=batch_size,
        )
        print(f"Features saved to: {features_dir}")

        del encoder
        torch.cuda.empty_cache()
        gc.collect()


def _run_multigpu_segmentation(gpu_ids, wsi_csv, output_dir, wsi_dir, max_workers=None):
    """Run tissue segmentation across multiple GPUs."""
    from autobench.pipeline._gpu_worker import gpu_init

    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    num_cores = int(slurm_cpus) if slurm_cpus else (os.cpu_count() or 16)
    workers_per_gpu = max_workers or max(1, num_cores // (2 * len(gpu_ids)))
    print(f"\nMulti-GPU segmentation across {len(gpu_ids)} GPUs ({workers_per_gpu} workers each)")

    ctx = multiprocessing.get_context("spawn")
    pools = []
    futures = {}

    try:
        for gpu_id in gpu_ids:
            pool = ProcessPoolExecutor(
                max_workers=1, mp_context=ctx,
                initializer=gpu_init, initargs=(gpu_id,),
            )
            pools.append(pool)
            future = pool.submit(_segment_on_gpu, wsi_csv, output_dir, wsi_dir, workers_per_gpu)
            futures[future] = gpu_id

        for future in as_completed(futures):
            gpu_id = futures[future]
            try:
                future.result()
            except Exception as exc:
                print(f"GPU {gpu_id}: segmentation FAILED with {exc!r}")
    finally:
        for pool in pools:
            pool.shutdown(wait=True)


def _run_multigpu(encoder_keys, gpu_ids, wsi_csv, coords_dir, batch_size,
                  output_dir, wsi_dir, max_workers=None):
    """Run models concurrently, one model per GPU."""
    from autobench.pipeline._gpu_worker import gpu_init

    assignments = {g: [] for g in gpu_ids}
    for idx, key in enumerate(encoder_keys):
        assignments[gpu_ids[idx % len(gpu_ids)]].append(key)

    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    num_cores = int(slurm_cpus) if slurm_cpus else (os.cpu_count() or 16)
    workers_per_gpu = max_workers or max(1, num_cores // (2 * len(gpu_ids)))

    ctx = multiprocessing.get_context("spawn")
    pools = []
    futures = {}
    failed = []

    try:
        for gpu_id in gpu_ids:
            keys = assignments[gpu_id]
            if not keys:
                continue
            pool = ProcessPoolExecutor(
                max_workers=1, mp_context=ctx,
                initializer=gpu_init, initargs=(gpu_id,),
            )
            pools.append(pool)
            for encoder_key in keys:
                future = pool.submit(
                    _extract_single_model, encoder_key, wsi_csv, coords_dir,
                    batch_size, output_dir, wsi_dir, workers_per_gpu,
                )
                futures[future] = (gpu_id, encoder_key)

        for future in as_completed(futures):
            gpu_id, encoder_key = futures[future]
            try:
                key, features_dir = future.result()
                if features_dir:
                    print(f"GPU {gpu_id}: {key} -> {features_dir}")
                else:
                    failed.append(encoder_key)
            except Exception as exc:
                failed.append(encoder_key)
                print(f"GPU {gpu_id}: {encoder_key} FAILED with {exc!r}")
    finally:
        for pool in pools:
            pool.shutdown(wait=True)

    if failed:
        print(f"\nFailed models ({len(failed)}): {failed}")


if __name__ == "__main__":
    main()
