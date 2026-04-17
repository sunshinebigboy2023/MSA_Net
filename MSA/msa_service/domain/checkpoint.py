from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict

import torch


@dataclass(frozen=True)
class FeatureDims:
    audio: int
    text: int
    video: int


@dataclass(frozen=True)
class RuntimeFlags:
    enable_av_cross: bool
    enable_label_prototype: bool
    enable_intensity_contrastive: bool
    disable_adaptive_cross_gate: bool
    disable_adaptive_va_gate: bool
    enable_text_stim: bool
    enable_fusion_gate: bool
    enable_conflict_head: bool
    enable_late_unimodal_calibration: bool
    enable_shared_private_decomp: bool


@dataclass(frozen=True)
class ModelRuntime:
    checkpoint_path: str
    dataset: str
    hidden: int
    n_classes: int
    depth: int
    num_heads: int
    feature_dims: FeatureDims
    flags: RuntimeFlags


def _load_state_dict(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError(f"Unsupported checkpoint payload: {type(checkpoint)!r}")


def _infer_dataset(checkpoint_path: str, n_classes: int) -> str:
    checkpoint_text = checkpoint_path.upper()
    basename = os.path.basename(checkpoint_path).upper()
    if "SIMS" in checkpoint_text:
        return "SIMS"
    if "CMUMOSI" in basename or "MOSI" in basename:
        return "CMUMOSI"
    if "CMUMOSEI" in basename or "MOSEI" in basename:
        return "CMUMOSEI"
    if "IEMOCAPFOUR" in basename:
        return "IEMOCAPFour"
    if "IEMOCAPSIX" in basename:
        return "IEMOCAPSix"
    if n_classes == 1:
        return "CMUMOSI"
    if n_classes == 4:
        return "IEMOCAPFour"
    if n_classes == 6:
        return "IEMOCAPSix"
    return "CMUMOSEI"


def infer_runtime_from_checkpoint(checkpoint_path: str) -> ModelRuntime:
    state = _load_state_dict(checkpoint_path)
    feature_dims = FeatureDims(
        audio=int(state["a_in_proj.0.weight"].shape[1]),
        text=int(state["t_in_proj.0.weight"].shape[1]),
        video=int(state["v_in_proj.0.weight"].shape[1]),
    )
    hidden = int(state["a_in_proj.0.weight"].shape[0])
    n_classes = int(state["nlp_head.weight"].shape[0])
    dataset = _infer_dataset(checkpoint_path, n_classes)
    flags = RuntimeFlags(
        enable_av_cross=any(key.startswith("cross_av.") for key in state),
        enable_label_prototype="sentiment_prototypes" in state,
        enable_intensity_contrastive=any(key.startswith("contrastive_proj.") for key in state),
        disable_adaptive_cross_gate=not any(key.startswith("cross_gate_ta.") for key in state),
        disable_adaptive_va_gate=not any(key.startswith("cross_gate_va.") for key in state),
        enable_text_stim=any(key.startswith("text_stim_gate_") for key in state),
        enable_fusion_gate=any(key.startswith("fusion_gate.") for key in state),
        enable_conflict_head=any(key.startswith("conflict_proj.") for key in state),
        enable_late_unimodal_calibration=any(key.startswith("late_calib_router.") for key in state),
        enable_shared_private_decomp=any(key.startswith("shared_proj_a.") for key in state),
    )
    return ModelRuntime(
        checkpoint_path=checkpoint_path,
        dataset=dataset,
        hidden=hidden,
        n_classes=n_classes,
        depth=4,
        num_heads=2,
        feature_dims=feature_dims,
        flags=flags,
    )
