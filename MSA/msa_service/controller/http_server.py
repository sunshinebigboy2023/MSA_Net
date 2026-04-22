from __future__ import annotations

import argparse
import glob
import json
import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING
from wsgiref.simple_server import make_server

if TYPE_CHECKING:
    from msa_service.service.analysis_service import AnalysisService


def _default_checkpoint_source():
    models_dir = Path(os.getcwd()) / "models"
    candidates = glob.glob(str(models_dir / "**" / "*test-condition-*.pth"), recursive=True)
    if candidates:
        return str(models_dir)
    candidates = glob.glob(os.path.join(os.getcwd(), "*CMUMOSI*.pth"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError("No CMUMOSI checkpoint found. Please pass --checkpoint explicitly.")


def _json_response(start_response, status: str, payload):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = [
        ("Content-Type", "application/json; charset=utf-8"),
        ("Content-Length", str(len(body))),
    ]
    start_response(status, headers)
    return [body]


def build_app(service: "AnalysisService", run_async: bool = True):
    def app(environ, start_response):
        path = environ.get("PATH_INFO", "")
        method = environ.get("REQUEST_METHOD", "GET").upper()
        try:
            if method == "POST" and path == "/analyze":
                length = int(environ.get("CONTENT_LENGTH") or "0")
                payload = json.loads(environ["wsgi.input"].read(length).decode("utf-8") or "{}")
                task = service.submit(payload)

                def _run():
                    try:
                        service.run_task(task.task_id)
                    except Exception:
                        pass

                if run_async:
                    threading.Thread(target=_run, daemon=True).start()
                else:
                    _run()
                return _json_response(start_response, "202 Accepted", {"taskId": task.task_id, "status": task.status})

            parts = [part for part in path.split("/") if part]
            if method == "GET" and len(parts) == 2 and parts[0] == "task":
                task = service.get_task(parts[1])
                if task is None:
                    return _json_response(start_response, "404 Not Found", {"message": "task not found"})
                return _json_response(
                    start_response,
                    "200 OK",
                    {"taskId": task.task_id, "status": task.status, "error": task.error},
                )

            if method == "GET" and len(parts) == 2 and parts[0] == "result":
                task = service.get_task(parts[1])
                if task is None:
                    return _json_response(start_response, "404 Not Found", {"message": "task not found"})
                if task.result is None:
                    return _json_response(
                        start_response,
                        "202 Accepted",
                        {"taskId": task.task_id, "status": task.status, "error": task.error},
                    )
                return _json_response(start_response, "200 OK", task.result)

            return _json_response(start_response, "404 Not Found", {"message": "not found"})
        except Exception as exc:
            return _json_response(start_response, "500 Internal Server Error", {"message": "internal server error", "error": str(exc)})

    return app


def main():
    from msa_service.service.analysis_service import AnalysisService

    parser = argparse.ArgumentParser(description="Standalone MoMKE local HTTP server")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    service = AnalysisService(args.checkpoint or _default_checkpoint_source())
    app = build_app(service)
    print(f"Serving on http://{args.host}:{args.port}")
    with make_server(args.host, args.port, app) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
