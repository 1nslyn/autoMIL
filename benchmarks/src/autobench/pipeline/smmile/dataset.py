"""SMMILe-compatible dataset that reads NIC .npy and superpixel .npy files.

Returns data matching SMMILe's collate_MIL output format:
  (features_nic_tensor, label_int, [coords_nic, mask, sp, adj, coords_nd], inst_label_list)

Key conventions matching lib/SMMILe/single/datasets/dataset_nic.py:
  - sp (superpixel map) is transposed: sp = sp_record['m_slic'].transpose(1,0)
  - mask is float64 (used with np.where)
  - features are float32 torch.Tensor
  - label is a raw int (training loop handles conversion)
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, SequentialSampler, RandomSampler


class SMMILeDataset(Dataset):
    """Dataset for SMMILe training from pre-converted NIC .npy files.

    Args:
        slide_data: DataFrame with 'slide_id' and 'label' (int-encoded).
        npy_dir: Directory with {slide_id}_0_{patch_size}.npy feature files.
        sp_dir: Directory with {slide_id}_0_{patch_size}.npy superpixel files.
        patch_size: Patch step size (for filename lookup).
    """

    def __init__(
        self,
        slide_data: pd.DataFrame,
        npy_dir: str,
        sp_dir: str,
        patch_size: int = 224,
    ):
        self.slide_data = slide_data.reset_index(drop=True)
        self.npy_dir = npy_dir
        self.sp_dir = sp_dir
        self.patch_size = patch_size
        self.num_classes = 1  # SMMILe_SINGLE: binary
        self.slide_cls_ids = self._build_cls_ids()

    def _build_cls_ids(self) -> list[np.ndarray]:
        """Build per-class index arrays (needed for weighted sampling)."""
        labels = self.slide_data["label"].values
        n_classes = len(np.unique(labels))
        return [np.where(labels == c)[0] for c in range(n_classes)]

    def _npy_path(self, slide_id: str) -> str:
        return os.path.join(self.npy_dir, f"{slide_id}_0_{self.patch_size}.npy")

    def _sp_path(self, slide_id: str) -> str:
        return os.path.join(self.sp_dir, f"{slide_id}_0_{self.patch_size}.npy")

    def __len__(self) -> int:
        return len(self.slide_data)

    def getlabel(self, idx: int) -> int:
        return int(self.slide_data["label"].iloc[idx])

    def __getitem__(self, idx: int):
        slide_id = self.slide_data["slide_id"].iloc[idx]
        label = int(self.slide_data["label"].iloc[idx])

        # Load NIC features
        record = np.load(self._npy_path(slide_id), allow_pickle=True)[()]
        features = record["feature"]   # (C, H, W)
        coords_nd = record["index"]    # (N, 2)
        mask = record["mask"]          # (H, W)

        # Load superpixels
        sp_record = np.load(self._sp_path(slide_id), allow_pickle=True)[()]
        sp = sp_record["m_slic"]       # (H, W)
        adj = sp_record["m_adj"]       # (n_sp, n_sp)

        # NOTE: Do NOT transpose sp. Our prepare.py stores m_slic in (H, W)
        # matching the mask orientation. The original SMMILe transposes because
        # their generator stored it in (W, H) form. Since we generate sp from
        # the same mask grid, sp is already aligned with features and mask.

        features_t = torch.from_numpy(features).float()

        # coords_nic placeholder (only used for heatmap generation)
        coords_nic = np.zeros_like(mask)

        # No instance annotations
        inst_label_nic = []

        return features_t, label, [coords_nic, mask, sp, adj, coords_nd], inst_label_nic


def _collate_smmile(batch):
    """Collate for batch_size=1, matching SMMILe's collate_MIL."""
    img = batch[0][0]
    label = torch.LongTensor([batch[0][1]])
    coords = batch[0][2]
    inst_label = batch[0][3]
    return [img, label, coords, inst_label]


def make_smmile_loader(
    dataset: SMMILeDataset,
    training: bool = False,
    weighted: bool = False,
) -> DataLoader:
    """Create DataLoader with collate_MIL-style batching."""
    kwargs = {"num_workers": 4} if torch.cuda.is_available() else {}

    if training and weighted:
        labels = dataset.slide_data["label"].values
        class_counts = np.bincount(labels.astype(int))
        weights = 1.0 / class_counts[labels.astype(int)]
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        return DataLoader(
            dataset, batch_size=1, sampler=sampler,
            collate_fn=_collate_smmile, **kwargs,
        )

    if training:
        return DataLoader(
            dataset, batch_size=1, sampler=RandomSampler(dataset),
            collate_fn=_collate_smmile, **kwargs,
        )

    return DataLoader(
        dataset, batch_size=1, sampler=SequentialSampler(dataset),
        collate_fn=_collate_smmile, **kwargs,
    )


def create_smmile_split(
    slide_data: pd.DataFrame,
    split_csv: str,
    npy_dir: str,
    sp_dir: str,
    patch_size: int = 224,
) -> tuple[SMMILeDataset, SMMILeDataset, SMMILeDataset]:
    """Create train/val/test SMMILeDataset splits from a split CSV."""
    splits = pd.read_csv(split_csv, dtype=str)

    datasets = []
    for col in ("train", "val", "test"):
        ids = splits[col].dropna().tolist()
        # Strip only a trailing ".0" (float-coercion for numeric ids); do NOT
        # split on "." — TCGA slide_ids carry a "." before the UUID suffix.
        ids = [s[:-2] if s.endswith(".0") and s[:-2].isdigit() else s
               for s in map(str, ids)]
        mask = slide_data["slide_id"].astype(str).isin(ids)
        df_split = slide_data[mask].reset_index(drop=True)

        # Filter to slides with existing NIC files
        has_file = df_split["slide_id"].apply(
            lambda sid: os.path.exists(
                os.path.join(npy_dir, f"{sid}_0_{patch_size}.npy")
            )
        )
        if (~has_file).any():
            missing = df_split.loc[~has_file, "slide_id"].tolist()
            print(f"  [WARNING] {col}: dropping {len(missing)} slides missing NIC files")
            df_split = df_split[has_file].reset_index(drop=True)

        datasets.append(SMMILeDataset(df_split, npy_dir, sp_dir, patch_size))

    return tuple(datasets)
