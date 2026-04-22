from __future__ import annotations

import math
import sys
import types
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as AF


class FeatureExtractionUnavailable(RuntimeError):
    pass


class _ZeroPad1d(nn.Module):
    def __init__(self, pad_left: int, pad_right: int):
        super().__init__()
        self.pad_left = pad_left
        self.pad_right = pad_right

    def forward(self, x):
        return F.pad(x, (self.pad_left, self.pad_right))


def _norm_block(dim: int, affine: bool = True):
    return nn.GroupNorm(1, dim, affine=affine)


class _ConvFeatureExtractionModel(nn.Module):
    def __init__(self, conv_layers, log_compression: bool, non_affine_group_norm: bool, activation):
        super().__init__()
        in_dim = 1
        layers = []
        for out_dim, kernel, stride in conv_layers:
            layers.append(
                nn.Sequential(
                    nn.Conv1d(in_dim, out_dim, kernel, stride=stride, bias=False),
                    nn.Dropout(p=0.0),
                    _norm_block(out_dim, affine=not non_affine_group_norm),
                    activation,
                )
            )
            in_dim = out_dim
        self.conv_layers = nn.ModuleList(layers)
        self.log_compression = log_compression

    def forward(self, x):
        x = x.unsqueeze(1)
        for conv in self.conv_layers:
            x = conv(x)
        if self.log_compression:
            x = torch.log(torch.abs(x) + 1)
        return x


class _ConvAggregator(nn.Module):
    def __init__(
        self,
        conv_layers,
        embed: int,
        skip_connections: bool,
        residual_scale: float,
        non_affine_group_norm: bool,
        conv_bias: bool,
        zero_pad: bool,
        activation,
    ):
        super().__init__()
        in_dim = embed
        layers = []
        residual_proj = []
        for out_dim, kernel, stride in conv_layers:
            left = kernel // 2
            right = left - 1 if kernel % 2 == 0 else left
            pad = _ZeroPad1d(left + right, 0) if zero_pad else nn.ReplicationPad1d((left + right, 0))
            layers.append(
                nn.Sequential(
                    pad,
                    nn.Conv1d(in_dim, out_dim, kernel, stride=stride, bias=conv_bias),
                    nn.Dropout(p=0.0),
                    _norm_block(out_dim, affine=not non_affine_group_norm),
                    activation,
                )
            )
            residual_proj.append(nn.Conv1d(in_dim, out_dim, 1, bias=False) if in_dim != out_dim and skip_connections else None)
            in_dim = out_dim
        self.conv_layers = nn.Sequential(*layers)
        self.residual_proj = nn.ModuleList(residual_proj)
        self.skip_connections = skip_connections
        self.residual_scale = math.sqrt(residual_scale)

    def forward(self, x):
        for projection, conv in zip(self.residual_proj, self.conv_layers):
            residual = x
            x = conv(x)
            if self.skip_connections:
                if projection is not None:
                    residual = projection(residual)
                x = (x + residual) * self.residual_scale
        return x


class _Wav2VecFeatureModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        conv_feature_layers = _as_layers(args.conv_feature_layers)
        conv_aggregator_layers = _as_layers(args.conv_aggregator_layers)
        activation_name = getattr(args, "activation", None) or "relu"
        activation = nn.ReLU() if activation_name == "relu" else nn.GELU()
        embed = conv_feature_layers[-1][0]
        self.feature_extractor = _ConvFeatureExtractionModel(
            conv_layers=conv_feature_layers,
            log_compression=bool(getattr(args, "log_compression", False)),
            non_affine_group_norm=bool(getattr(args, "non_affine_group_norm", False)),
            activation=activation,
        )
        self.feature_aggregator = _ConvAggregator(
            conv_layers=conv_aggregator_layers,
            embed=embed,
            skip_connections=bool(getattr(args, "skip_connections_agg", False)),
            residual_scale=float(getattr(args, "residual_scale", 0.5)),
            non_affine_group_norm=bool(getattr(args, "non_affine_group_norm", False)),
            conv_bias=not bool(getattr(args, "no_conv_bias", False)),
            zero_pad=bool(getattr(args, "agg_zero_pad", False)),
            activation=activation,
        )


def _as_layers(value):
    if isinstance(value, str):
        return eval(value)
    return value


def _install_fairseq_checkpoint_stubs():
    if "fairseq.meters" in sys.modules:
        return

    class AverageMeter:
        pass

    class TimeMeter:
        pass

    class StopwatchMeter:
        pass

    fairseq = types.ModuleType("fairseq")
    meters = types.ModuleType("fairseq.meters")
    meters.AverageMeter = AverageMeter
    meters.TimeMeter = TimeMeter
    meters.StopwatchMeter = StopwatchMeter
    sys.modules.setdefault("fairseq", fairseq)
    sys.modules["fairseq.meters"] = meters


class Wav2VecAudioFeatureService:
    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        self.model_path = Path(model_path) if model_path else self._resolve_model_path()
        self.device = torch.device(device)
        self._model = None

    def extract(self, audio_path: str) -> np.ndarray:
        model = self._load_model()
        audio, sample_rate = sf.read(audio_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = np.asarray(audio, dtype=np.float32)
        if sample_rate != 16000:
            audio = AF.resample(torch.from_numpy(audio), sample_rate, 16000).numpy()

        tensor = torch.from_numpy(audio[np.newaxis, :]).to(self.device)
        with torch.no_grad():
            z = model.feature_extractor(tensor)
            c = model.feature_aggregator(z)
        feature = c.detach().squeeze().t().cpu().numpy()
        if feature.ndim != 1:
            feature = feature.mean(axis=0)
        return feature.astype(np.float32)

    def _load_model(self):
        if self._model is not None:
            return self._model

        _install_fairseq_checkpoint_stubs()
        checkpoint = torch.load(str(self.model_path), map_location=self.device, weights_only=False)
        model = _Wav2VecFeatureModel(checkpoint["args"])
        model.load_state_dict(checkpoint["model"], strict=False)
        self._model = model.to(self.device).eval()
        return self._model

    @staticmethod
    def _resolve_model_path() -> Path:
        path = Path(__file__).resolve().parents[2] / "tools" / "wav2vec" / "wav2vec_large.pt"
        if not path.exists():
            raise FileNotFoundError(f"wav2vec checkpoint not found: {path}")
        return path
