"""Technique-name normalisation map for the autobench (MIL pathology) consumer.

The framework ships ``ExperimentGraph.DEFAULT_TECHNIQUE_MAP = {}`` by
design (autoMIL is generic; consumer vocabulary stays in the consumer).
This module owns the autobench-side mapping that the previous bundled
map encoded — pass it to ``ExperimentGraph(...,
technique_map=AUTOBENCH_TECHNIQUE_MAP)`` from consumer paths that need
shorthand normalisation in node descriptions / techniques lists.
"""
from __future__ import annotations

AUTOBENCH_TECHNIQUE_MAP: dict[str, str] = {
    "no_inst": "no_inst_eval",
    "focal": "focal_g1",
    "gamma1": "focal_g1",
    "gc05": "grad_clip",
    "gc0.5": "grad_clip",
    "rdrop": "rdrop",
    "step_lr": "step_lr",
    "coord_pe": "coord_pe",
    "noise001": "noise_aug",
    "d0.1": "dropout_01",
    "big": "big_model",
    "bw0.5": "bag_weight_05",
    "trans_mil": "trans_mil",
    "dtfd": "dtfd_mil",
    "ilra": "ilra_mil",
    "vit": "vision_transformer",
    # ABMIL arm (Framework.ABMIL): attention-based MIL. Model keys are "abmil"
    # (non-gated, the default) and "abmil_gated" (gated). Matching is substring,
    # so "abmil_gated" run descriptions also carry this "abmil" tag — one
    # ABMIL-family bucket. Add an explicit "abmil_gated" entry to split them out.
    "abmil": "abmil",
    "clam_sb": "clam_sb",
    "uni_v2": "uni_v2",
    "hibou_l": "hibou_l",
    "psemix": "psemix",
    "aem": "aem",
    "variance_pool": "variance_pool",
    "topk": "topk_attn",
    "maxsoft": "maxsoft",
    "supcon": "supcon",
    "attn_temp": "attn_temp",
    "poly1": "poly_loss",
    "bilevel_lr": "bilevel_lr",
}
