from __future__ import annotations

import argparse
import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class MockAnalysisHandler(BaseHTTPRequestHandler):
    delay_ms = 12

    def do_POST(self):
        if self.path != "/api/analysis/analyze":
            self.send_response(404)
            self.end_headers()
            return

        length = int(self.headers.get("Content-Length") or 0)
        if length:
            self.rfile.read(length)

        time.sleep(self.delay_ms / 1000)
        body = json.dumps(
            {
                "code": 0,
                "data": {"taskId": str(uuid.uuid4()), "status": "QUEUED"},
                "message": "ok",
                "description": "",
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format, *args):
        return


def parse_args():
    parser = argparse.ArgumentParser(description="Mock /api/analysis/analyze for load-test smoke checks.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--delay-ms", type=int, default=12)
    return parser.parse_args()


def main():
    args = parse_args()
    MockAnalysisHandler.delay_ms = args.delay_ms
    server = ThreadingHTTPServer((args.host, args.port), MockAnalysisHandler)
    print(f"Mock analysis submit server on http://{args.host}:{args.port}/api/analysis/analyze")
    server.serve_forever()


if __name__ == "__main__":
    main()
