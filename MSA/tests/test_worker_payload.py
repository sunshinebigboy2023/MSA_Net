import unittest


class WorkerPayloadTests(unittest.TestCase):
    def test_callback_payload_includes_task_status_and_result(self):
        from msa_service.worker import build_success_callback_payload

        result = {"confidence": 0.82, "usedModalities": ["text"]}
        payload = build_success_callback_payload("task-1", result, 123)

        self.assertEqual(payload["taskId"], "task-1")
        self.assertEqual(payload["status"], "SUCCESS")
        self.assertEqual(payload["result"]["confidence"], 0.82)
        self.assertEqual(payload["processingTimeMs"], 123)


if __name__ == "__main__":
    unittest.main()
