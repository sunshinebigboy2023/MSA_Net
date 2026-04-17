from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Callable, Dict, Mapping

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
for model_dir in (ROOT_DIR / "DT-MSA", ROOT_DIR / "MoMKE"):
    if model_dir.exists() and str(model_dir) not in sys.path:
        sys.path.insert(0, str(model_dir))

from utils import build_model  # type: ignore

from msa_service.domain.checkpoint import ModelRuntime, infer_runtime_from_checkpoint
from msa_service.domain.schemas import PredictionResult


_CONDITION_PATTERN = re.compile(r"test-condition-([atv]+)\.pth$", re.IGNORECASE)
_MODALITY_TO_CODE = {"audio": "a", "text": "t", "video": "v"}
_DEFAULT_DATASET = "CMUMOSI"


def _used_modalities(audio_feature=None, text_feature=None, video_feature=None):
    used = []
    if audio_feature is not None:
        used.append("audio")
    if text_feature is not None:
        used.append("text")
    if video_feature is not None:
        used.append("video")
    return used


def condition_from_features(audio_feature=None, text_feature=None, video_feature=None) -> str:
    used = _used_modalities(audio_feature, text_feature, video_feature)
    return "".join(_MODALITY_TO_CODE[name] for name in used)


def _dataset_from_checkpoint_path(path: Path) -> str:
    text = str(path).upper()
    if "SIMS" in text:
        return "SIMS"
    if "CMUMOSEI" in text or "MOSEI" in text:
        return "CMUMOSEI"
    if "CMUMOSI" in text or "MOSI" in text:
        return "CMUMOSI"
    return _DEFAULT_DATASET


def discover_condition_checkpoints(source: str | os.PathLike | None = None) -> Dict[str, Dict[str, str]]:
    root = Path(source or Path.cwd() / "models")
    candidates = [root] if root.is_file() else [Path(path) for path in glob.glob(str(root / "**" / "*.pth"), recursive=True)]
    discovered: Dict[str, Dict[str, Path]] = {}
    for path in candidates:
        match = _CONDITION_PATTERN.search(path.name)
        if not match:
            continue
        dataset = _dataset_from_checkpoint_path(path)
        condition = match.group(1).lower()
        dataset_paths = discovered.setdefault(dataset, {})
        previous = dataset_paths.get(condition)
        if previous is None or path.stat().st_mtime >= previous.stat().st_mtime:
            dataset_paths[condition] = path
    return {
        dataset: {condition: str(path) for condition, path in sorted(paths.items())}
        for dataset, paths in sorted(discovered.items())
    }


def _to_feature_array(feature, expected_dim: int, name: str) -> np.ndarray:
    if feature is None:
        return np.zeros((expected_dim,), dtype=np.float32)
    array = np.asarray(feature, dtype=np.float32).reshape(-1)
    if array.shape[0] != expected_dim:
        raise ValueError(f"{name} feature dim mismatch: expected {expected_dim}, got {array.shape[0]}")
    return array


def _score_to_polarity(score: float) -> str:
    if score > 0:
        return "positive"
    if score < 0:
        return "negative"
    return "neutral"


def _score_to_confidence(score: float) -> float:
    return float(torch.sigmoid(torch.tensor(abs(score))).item())


class MoMKEPredictor:
    def __init__(self, runtime: ModelRuntime, model: torch.nn.Module):
        self.runtime = runtime
        self.model = model.eval()

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str) -> "MoMKEPredictor":
        runtime = infer_runtime_from_checkpoint(checkpoint_path)
        args = cls._build_args(runtime)
        model = build_model(
            args,
            runtime.feature_dims.audio,
            runtime.feature_dims.text,
            runtime.feature_dims.video,
        )
        state = torch.load(checkpoint_path, map_location="cpu")
        state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
        model.load_state_dict(state_dict, strict=True)
        return cls(runtime=runtime, model=model)

    @staticmethod
    def _build_args(runtime: ModelRuntime):
        flags = runtime.flags
        return argparse.Namespace(
            hidden=runtime.hidden,
            depth=runtime.depth,
            num_heads=runtime.num_heads,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            no_cuda=True,
            device=torch.device("cpu"),
            num_experts=2,
            enable_layerwise_sparse_router=False,
            num_shared_experts=1,
            router_topk=2,
            router_noise_std=0.0,
            audio_drop_rate=None,
            text_drop_rate=None,
            visual_drop_rate=None,
            use_cross_modal=True,
            cross_mix_weight=0.7,
            cross_mix_ta=None,
            cross_mix_tv=None,
            cross_mix_at=None,
            cross_mix_vt=None,
            enable_av_cross=flags.enable_av_cross,
            cross_mix_av=0.2,
            cross_mix_va=0.2,
            enable_text_stim=flags.enable_text_stim,
            text_stim_a_scale=0.0,
            text_stim_v_scale=0.1,
            disable_detach_text_stim_source=False,
            enable_fusion_gate=flags.enable_fusion_gate,
            fusion_gate_scale=0.1,
            disable_detach_fusion_text_source=False,
            enable_conflict_head=flags.enable_conflict_head,
            conflict_head_scale=0.1,
            enable_late_unimodal_calibration=flags.enable_late_unimodal_calibration,
            late_calib_scale=0.2,
            disable_clean_va_path=False,
            disable_detach_va_source=False,
            disable_adaptive_va_gate=flags.disable_adaptive_va_gate,
            va_gate_delta=0.03,
            disable_adaptive_cross_gate=flags.disable_adaptive_cross_gate,
            enable_shared_private_decomp=flags.enable_shared_private_decomp,
            enable_intensity_contrastive=flags.enable_intensity_contrastive,
            contrastive_dim=32,
            contrastive_label_sigma=0.75,
            enable_label_prototype=flags.enable_label_prototype,
            prototype_dim=32,
            prototype_bins=7,
            prototype_temperature=0.5,
            prototype_target_sigma=0.75,
            prototype_min=-3.0,
            prototype_max=3.0,
            enable_polarity_loss=False,
            n_classes=runtime.n_classes,
        )

    def predict_from_features(self, audio_feature=None, text_feature=None, video_feature=None, task_id: str | None = None):
        audio = _to_feature_array(audio_feature, self.runtime.feature_dims.audio, "audio")
        text = _to_feature_array(text_feature, self.runtime.feature_dims.text, "text")
        video = _to_feature_array(video_feature, self.runtime.feature_dims.video, "video")

        used_modalities = _used_modalities(audio_feature, text_feature, video_feature)
        missing_modalities = [name for name in ("audio", "text", "video") if name not in used_modalities]

        audio_host = torch.from_numpy(audio).view(1, 1, -1)
        text_host = torch.from_numpy(text).view(1, 1, -1)
        video_host = torch.from_numpy(video).view(1, 1, -1)
        umask = torch.ones((1, 1), dtype=torch.float32)
        modality_mask = torch.tensor(
            [
                [
                    [
                        1.0 if "audio" in used_modalities else 0.0,
                        1.0 if "text" in used_modalities else 0.0,
                        1.0 if "video" in used_modalities else 0.0,
                    ]
                ]
            ],
            dtype=torch.float32,
        )
        masked_audio = audio_host if "audio" in used_modalities else torch.zeros_like(audio_host)
        masked_text = text_host if "text" in used_modalities else torch.zeros_like(text_host)
        masked_video = video_host if "video" in used_modalities else torch.zeros_like(video_host)
        features = torch.cat([masked_audio, masked_text, masked_video], dim=2)

        with torch.no_grad():
            _, out, _, _, _, _ = self.model(features, modality_mask, umask, False)

        score = float(out.view(-1)[0].item())
        return PredictionResult(
            taskId=task_id or str(uuid.uuid4()),
            usedModalities=used_modalities,
            missingModalities=missing_modalities,
            emotionLabel=None,
            sentimentPolarity=_score_to_polarity(score),
            score=score,
            confidence=_score_to_confidence(score),
            message="success",
            error=None,
        )


class MoMKEPredictorRegistry:
    def __init__(
        self,
        checkpoint_paths: Mapping[str, str] | Mapping[str, Mapping[str, str]],
        predictor_factory: Callable[[str], MoMKEPredictor] | None = None,
    ):
        self.checkpoint_paths = self._normalize_checkpoint_paths(checkpoint_paths)
        if not self.checkpoint_paths:
            raise FileNotFoundError("No condition checkpoints found")
        self._predictor_factory = predictor_factory or MoMKEPredictor.from_checkpoint
        self._predictors: Dict[tuple[str, str], MoMKEPredictor] = {}

    @staticmethod
    def _normalize_checkpoint_paths(
        checkpoint_paths: Mapping[str, str] | Mapping[str, Mapping[str, str]],
    ) -> Dict[str, Dict[str, str]]:
        paths = dict(checkpoint_paths)
        if not paths:
            return {}
        if all(isinstance(value, str) for value in paths.values()):
            return {_DEFAULT_DATASET: {str(condition): str(path) for condition, path in paths.items()}}
        return {
            str(dataset): {str(condition): str(path) for condition, path in dict(conditions).items()}
            for dataset, conditions in paths.items()
        }

    @classmethod
    def from_checkpoint_source(cls, source: str | os.PathLike | None = None) -> "MoMKEPredictorRegistry":
        return cls(discover_condition_checkpoints(source))

    @property
    def supported_conditions(self):
        conditions = set()
        for dataset_paths in self.checkpoint_paths.values():
            conditions.update(dataset_paths)
        return tuple(sorted(conditions))

    @property
    def supported_datasets(self):
        return tuple(sorted(self.checkpoint_paths))

    def _default_dataset(self) -> str:
        if _DEFAULT_DATASET in self.checkpoint_paths:
            return _DEFAULT_DATASET
        return sorted(self.checkpoint_paths)[0]

    def _predictor_for(self, dataset: str, condition: str) -> MoMKEPredictor:
        if dataset not in self.checkpoint_paths:
            supported = ", ".join(self.supported_datasets)
            raise ValueError(f"No checkpoint available for dataset {dataset!r}. Supported datasets: {supported}")
        if condition not in self.checkpoint_paths[dataset]:
            supported = ", ".join(sorted(self.checkpoint_paths[dataset]))
            raise ValueError(
                f"No checkpoint available for dataset {dataset!r} and condition {condition!r}. "
                f"Supported conditions: {supported}"
            )
        key = (dataset, condition)
        if key not in self._predictors:
            self._predictors[key] = self._predictor_factory(self.checkpoint_paths[dataset][condition])
        return self._predictors[key]

    def predict_from_features(
        self,
        audio_feature=None,
        text_feature=None,
        video_feature=None,
        task_id: str | None = None,
        dataset: str | None = None,
    ):
        condition = condition_from_features(audio_feature, text_feature, video_feature)
        if not condition:
            raise ValueError("At least one modality feature is required")
        predictor = self._predictor_for(dataset or self._default_dataset(), condition)
        return predictor.predict_from_features(
            audio_feature=audio_feature,
            text_feature=text_feature,
            video_feature=video_feature,
            task_id=task_id,
        )
