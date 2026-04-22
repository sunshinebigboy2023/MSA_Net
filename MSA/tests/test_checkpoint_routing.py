import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from msa_service.domain.checkpoint import FeatureDims, ModelRuntime, RuntimeFlags


class CheckpointRoutingTests(unittest.TestCase):
    def test_discovers_condition_checkpoints_from_filenames(self):
        from msa_service.service.predictor_service import discover_condition_checkpoints

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for condition in ("a", "t", "at", "atv"):
                (root / f"model_test-condition-{condition}.pth").write_bytes(b"fake")

            paths = discover_condition_checkpoints(root)

        self.assertEqual(set(paths), {"CMUMOSI"})
        self.assertEqual(set(paths["CMUMOSI"]), {"a", "t", "at", "atv"})
        self.assertTrue(paths["CMUMOSI"]["t"].endswith("model_test-condition-t.pth"))

    def test_registry_selects_exact_checkpoint_for_used_modalities(self):
        from msa_service.domain.schemas import PredictionResult
        from msa_service.service.predictor_service import MoMKEPredictorRegistry

        loaded = []

        class FakePredictor:
            def __init__(self, path):
                self.path = path

            def predict_from_features(self, audio_feature=None, text_feature=None, video_feature=None, task_id=None):
                loaded.append(self.path)
                return PredictionResult(
                    taskId=task_id or "task",
                    usedModalities=[],
                    missingModalities=[],
                    emotionLabel=None,
                    sentimentPolarity="positive",
                    score=1.0,
                    confidence=0.73,
                    message="success",
                    error=None,
                )

        registry = MoMKEPredictorRegistry(
            {"t": "text.pth", "at": "audio-text.pth"},
            predictor_factory=FakePredictor,
        )

        registry.predict_from_features(text_feature=np.ones((1024,), dtype=np.float32))
        registry.predict_from_features(
            audio_feature=np.ones((512,), dtype=np.float32),
            text_feature=np.ones((1024,), dtype=np.float32),
        )

        self.assertEqual(loaded, ["text.pth", "audio-text.pth"])

    def test_registry_rejects_unsupported_modality_combination(self):
        from msa_service.service.predictor_service import MoMKEPredictorRegistry

        registry = MoMKEPredictorRegistry({"t": "text.pth"}, predictor_factory=lambda path: None)

        with self.assertRaisesRegex(ValueError, "No checkpoint available"):
            registry.predict_from_features(video_feature=np.ones((1024,), dtype=np.float32))

    def test_registry_falls_back_to_best_supported_subset(self):
        from msa_service.domain.schemas import PredictionResult
        from msa_service.service.predictor_service import MoMKEPredictorRegistry

        loaded = []

        class FakePredictor:
            def __init__(self, path):
                self.path = path

            def predict_from_features(self, audio_feature=None, text_feature=None, video_feature=None, task_id=None):
                used = []
                if audio_feature is not None:
                    used.append("audio")
                if text_feature is not None:
                    used.append("text")
                if video_feature is not None:
                    used.append("video")
                loaded.append((self.path, used))
                return PredictionResult(
                    taskId=task_id or "task",
                    usedModalities=used,
                    missingModalities=[name for name in ("audio", "text", "video") if name not in used],
                    emotionLabel=None,
                    sentimentPolarity="positive",
                    score=1.0,
                    confidence=0.73,
                    message="success",
                    error=None,
                )

        registry = MoMKEPredictorRegistry(
            {"t": "text.pth", "at": "audio-text.pth"},
            predictor_factory=FakePredictor,
        )

        result = registry.predict_from_features(
            audio_feature=np.ones((512,), dtype=np.float32),
            text_feature=np.ones((1024,), dtype=np.float32),
            video_feature=np.ones((1024,), dtype=np.float32),
        )

        self.assertEqual(loaded, [("audio-text.pth", ["audio", "text"])])
        self.assertEqual(result.usedModalities, ["audio", "text"])


class PredictorMaskTests(unittest.TestCase):
    def _runtime(self):
        return ModelRuntime(
            checkpoint_path="fake.pth",
            dataset="CMUMOSI",
            hidden=4,
            n_classes=1,
            depth=1,
            num_heads=1,
            feature_dims=FeatureDims(audio=2, text=3, video=4),
            flags=RuntimeFlags(
                enable_av_cross=False,
                enable_label_prototype=False,
                enable_intensity_contrastive=False,
                disable_adaptive_cross_gate=True,
                disable_adaptive_va_gate=True,
                enable_text_stim=False,
                enable_fusion_gate=False,
                enable_conflict_head=False,
                enable_late_unimodal_calibration=False,
                enable_shared_private_decomp=False,
            ),
        )

    def test_predictor_passes_missing_modality_mask_to_model(self):
        from msa_service.service.predictor_service import MoMKEPredictor

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.seen_mask = None

            def forward(self, features, modality_mask, umask, first_stage):
                self.seen_mask = modality_mask.detach().cpu().numpy()
                out = torch.tensor([[[0.5]]], dtype=torch.float32)
                return None, out, out, out, out, np.array([])

        model = FakeModel()
        predictor = MoMKEPredictor(self._runtime(), model)

        predictor.predict_from_features(text_feature=np.ones((3,), dtype=np.float32))

        np.testing.assert_array_equal(model.seen_mask, np.array([[[0.0, 1.0, 0.0]]], dtype=np.float32))
