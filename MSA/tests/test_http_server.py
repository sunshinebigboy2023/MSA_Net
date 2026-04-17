import io
import json
import unittest

from dataclasses import dataclass


@dataclass
class _Task:
    task_id: str
    status: str = "PENDING"
    error: str | None = None
    result: dict | None = None


class _FakeService:
    def __init__(self):
        self.task = None

    def submit(self, payload):
        self.task = _Task(task_id="task-1")
        self.payload = payload
        return self.task

    def run_task(self, task_id):
        self.task.status = "SUCCESS"
        self.task.result = {
            "taskId": task_id,
            "usedModalities": ["text"],
            "missingModalities": ["audio", "video"],
            "sentimentPolarity": "positive",
            "score": 0.5,
            "confidence": 0.62,
            "message": "success",
            "error": None,
            "transcript": None,
            "featureStatus": {"audio": "missing", "text": "extracted", "video": "missing"},
            "rawInputs": {"text": "hello"},
            "processingTimeMs": 1,
        }
        return self.task.result

    def get_task(self, task_id):
        return self.task if self.task and self.task.task_id == task_id else None


def _call_app(app, method, path, payload=None):
    body = json.dumps(payload or {}).encode("utf-8")
    status_headers = {}

    def start_response(status, headers):
        status_headers["status"] = status
        status_headers["headers"] = headers

    environ = {
        "REQUEST_METHOD": method,
        "PATH_INFO": path,
        "CONTENT_LENGTH": str(len(body)),
        "wsgi.input": io.BytesIO(body),
    }
    response_body = b"".join(app(environ, start_response)).decode("utf-8")
    return status_headers["status"], json.loads(response_body)


class HttpServerTests(unittest.TestCase):
    def test_analyze_task_and_result_routes(self):
        from msa_service.controller.http_server import build_app

        service = _FakeService()
        app = build_app(service, run_async=False)

        status, payload = _call_app(app, "POST", "/analyze", {"text": "hello"})
        self.assertEqual(status, "202 Accepted")
        self.assertEqual(payload, {"taskId": "task-1", "status": "SUCCESS"})

        status, task_payload = _call_app(app, "GET", "/task/task-1")
        self.assertEqual(status, "200 OK")
        self.assertEqual(task_payload["status"], "SUCCESS")

        status, result_payload = _call_app(app, "GET", "/result/task-1")
        self.assertEqual(status, "200 OK")
        self.assertEqual(result_payload["featureStatus"]["text"], "extracted")
        self.assertEqual(result_payload["processingTimeMs"], 1)
