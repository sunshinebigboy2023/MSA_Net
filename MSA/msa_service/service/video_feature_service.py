from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from msa_service.service.audio_feature_service import FeatureExtractionUnavailable
from msa_service.service.openface_service import OpenFaceService


class ManetVideoFeatureService:
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = "cpu",
        batch_size: int = 16,
        openface_service: Optional[OpenFaceService] = None,
    ):
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else self._resolve_checkpoint_path()
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.openface_service = openface_service or OpenFaceService()
        self._model = None

    def extract(self, video_path: str) -> np.ndarray:
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        output_dir = Path("temp") / "openface" / Path(video_path).stem
        aligned_dir = self.openface_service.extract_aligned_faces(video_path, str(output_dir))
        return self.extract_from_aligned_dir(aligned_dir)

    def extract_from_aligned_dir(self, aligned_dir: str | Path) -> np.ndarray:
        face_dir = Path(aligned_dir)
        image_paths = sorted(
            [
                *face_dir.glob("*.bmp"),
                *face_dir.glob("*.jpg"),
                *face_dir.glob("*.png"),
            ]
        )
        if not image_paths:
            raise FeatureExtractionUnavailable(f"No aligned face images found under {face_dir}")

        model = self._load_model()
        embeddings = []
        for start in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[start : start + self.batch_size]
            batch = torch.stack([self._image_to_tensor(path) for path in batch_paths]).to(self.device)
            with torch.no_grad():
                embedding = model(batch, return_embedding=True)
            embeddings.append(embedding.detach().cpu().numpy())

        feature = np.concatenate(embeddings, axis=0).mean(axis=0)
        return feature.astype(np.float32)

    def _load_model(self):
        if self._model is not None:
            return self._model

        visual_root = Path(__file__).resolve().parents[2] / "GCNet" / "feature_extraction" / "visual"
        if str(visual_root) not in sys.path:
            sys.path.insert(0, str(visual_root))

        try:
            from manet.model.manet import manet
        except ModuleNotFoundError as exc:
            raise FeatureExtractionUnavailable("MANet source files are missing") from exc

        model = manet(num_classes=7)
        self._load_checkpoint(model)
        self._model = model.to(self.device).eval()
        return self._model

    def _load_checkpoint(self, model):
        class RecorderMeter:
            pass

        import __main__

        if not hasattr(__main__, "RecorderMeter"):
            setattr(__main__, "RecorderMeter", RecorderMeter)

        checkpoint = torch.load(str(self.checkpoint_path), map_location=self.device, weights_only=False)
        state_dict = {key.replace("module.", ""): value for key, value in checkpoint["state_dict"].items()}
        model.load_state_dict(state_dict, strict=True)

    @staticmethod
    def _image_to_tensor(image_path: Path) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB").resize((224, 224))
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = np.transpose(array, (2, 0, 1))
        return torch.from_numpy(array)

    @staticmethod
    def _resolve_checkpoint_path() -> Path:
        checkpoint = (
            Path(__file__).resolve().parents[2]
            / "tools"
            / "manet"
            / "[02-08]-[21-19]-model_best-acc88.33.pth"
        )
        if not checkpoint.exists():
            raise FileNotFoundError(f"MANet checkpoint not found: {checkpoint}")
        return checkpoint
