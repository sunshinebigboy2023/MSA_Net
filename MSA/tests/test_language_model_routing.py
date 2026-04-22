import tempfile
import unittest
from pathlib import Path

import numpy as np


class LanguageModelRoutingTests(unittest.TestCase):
    def test_detects_chinese_text_as_sims_and_english_text_as_cmumosi(self):
        from msa_service.service.language_service import dataset_for_text, detect_text_language

        self.assertEqual(detect_text_language("我很高兴"), "zh")
        self.assertEqual(dataset_for_text("我很高兴"), "SIMS")
        self.assertEqual(detect_text_language("I am happy"), "en")
        self.assertEqual(dataset_for_text("I am happy"), "CMUMOSI")
        self.assertEqual(detect_text_language("I am happy with 这个结果"), "en")

    def test_discovers_checkpoints_by_dataset_and_condition(self):
        from msa_service.service.predictor_service import discover_condition_checkpoints

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "CMUMOSI").mkdir()
            (root / "SIMS").mkdir()
            (root / "CMUMOSI" / "CMUMOSI_test-condition-t.pth").write_bytes(b"fake")
            (root / "SIMS" / "SIMS_test-condition-t.pth").write_bytes(b"fake")
            (root / "SIMS" / "SIMS_test-condition-atv.pth").write_bytes(b"fake")

            paths = discover_condition_checkpoints(root)

        self.assertEqual(set(paths), {"CMUMOSI", "SIMS"})
        self.assertTrue(paths["CMUMOSI"]["t"].endswith("CMUMOSI_test-condition-t.pth"))
        self.assertTrue(paths["SIMS"]["t"].endswith("SIMS_test-condition-t.pth"))
        self.assertTrue(paths["SIMS"]["atv"].endswith("SIMS_test-condition-atv.pth"))

    def test_registry_selects_dataset_and_modality_condition(self):
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
            {
                "CMUMOSI": {"t": "mosi-text.pth"},
                "SIMS": {"t": "sims-text.pth"},
            },
            predictor_factory=FakePredictor,
        )

        registry.predict_from_features(text_feature=np.ones((1024,), dtype=np.float32), dataset="SIMS")
        registry.predict_from_features(text_feature=np.ones((1024,), dtype=np.float32), dataset="CMUMOSI")

        self.assertEqual(loaded, ["sims-text.pth", "mosi-text.pth"])


class AnalysisServiceLanguageRoutingTests(unittest.TestCase):
    def test_chinese_text_routes_to_sims_and_english_text_routes_to_cmumosi(self):
        from tests.test_analysis_service_scenarios import _build_service

        service = _build_service()
        chinese_task = service.submit({"text": "我很高兴"})
        chinese_result = service.run_task(chinese_task.task_id)

        english_task = service.submit({"text": "I am happy"})
        english_result = service.run_task(english_task.task_id)

        self.assertEqual(chinese_result["language"], "zh")
        self.assertEqual(chinese_result["modelDataset"], "SIMS")
        self.assertEqual(english_result["language"], "en")
        self.assertEqual(english_result["modelDataset"], "CMUMOSI")
