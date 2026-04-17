import glob
import os
import unittest


def _find_cmumosi_checkpoint():
    matches = glob.glob(os.path.join(os.getcwd(), "models", "**", "*test-condition-atv.pth"), recursive=True)
    if not matches:
        matches = glob.glob(os.path.join(os.getcwd(), "models", "**", "*CMUMOSI*.pth"), recursive=True)
    if not matches:
        raise FileNotFoundError("No CMUMOSI checkpoint found")
    return matches[0]


class CheckpointRuntimeTests(unittest.TestCase):
    def test_infers_runtime_from_checkpoint(self):
        from msa_service.domain.checkpoint import infer_runtime_from_checkpoint

        runtime = infer_runtime_from_checkpoint(_find_cmumosi_checkpoint())

        self.assertEqual(runtime.dataset, "CMUMOSI")
        self.assertEqual(runtime.feature_dims.audio, 512)
        self.assertEqual(runtime.feature_dims.text, 1024)
        self.assertEqual(runtime.feature_dims.video, 1024)
        self.assertFalse(runtime.flags.enable_label_prototype)
