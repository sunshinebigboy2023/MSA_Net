import glob
import math
import os
import unittest

import numpy as np


def _find_cmumosi_checkpoint():
    matches = glob.glob(os.path.join(os.getcwd(), "models", "**", "*test-condition-t.pth"), recursive=True)
    if not matches:
        matches = glob.glob(os.path.join(os.getcwd(), "models", "**", "*CMUMOSI*.pth"), recursive=True)
    if not matches:
        raise FileNotFoundError("No CMUMOSI checkpoint found")
    return matches[0]


class PredictorSmokeTests(unittest.TestCase):
    def test_text_only_feature_prediction_returns_structured_result(self):
        from msa_service.service.predictor_service import MoMKEPredictor

        predictor = MoMKEPredictor.from_checkpoint(_find_cmumosi_checkpoint())
        result = predictor.predict_from_features(
            text_feature=np.zeros((predictor.runtime.feature_dims.text,), dtype=np.float32),
        )

        self.assertEqual(result.usedModalities, ["text"])
        self.assertEqual(sorted(result.missingModalities), ["audio", "video"])
        self.assertIn(result.sentimentPolarity, {"negative", "neutral", "positive"})
        self.assertTrue(math.isfinite(result.score))
        self.assertTrue(math.isfinite(result.confidence))
