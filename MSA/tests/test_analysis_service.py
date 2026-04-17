import glob
import os
import unittest


def _find_cmumosi_checkpoint():
    matches = glob.glob(os.path.join(os.getcwd(), "models", "**", "*test-condition-t.pth"), recursive=True)
    if not matches:
        matches = glob.glob(os.path.join(os.getcwd(), "models", "**", "*CMUMOSI*.pth"), recursive=True)
    if not matches:
        raise FileNotFoundError("No CMUMOSI checkpoint found")
    return matches[0]


class AnalysisServiceTests(unittest.TestCase):
    def test_analyze_text_payload_finishes_with_success(self):
        from msa_service.service import AnalysisService

        service = AnalysisService(_find_cmumosi_checkpoint())
        task = service.submit({"text": "this movie was unexpectedly good"})
        result = service.run_task(task.task_id)

        self.assertEqual(result["message"], "success")
        self.assertEqual(result["usedModalities"], ["text"])
        self.assertEqual(service.task_dao.get(task.task_id).status, "SUCCESS")
