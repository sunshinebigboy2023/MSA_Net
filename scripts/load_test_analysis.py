from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class Sample:
    ok: bool
    status_code: int
    elapsed_ms: int
    task_id: Optional[str] = None
    error: Optional[str] = None


def submit_once(base_url: str, text: str, language: str, cookie: Optional[str]) -> Sample:
    started = time.perf_counter()
    data = urllib.parse.urlencode({"text": text, "language": language}).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + "/analysis/analyze",
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    if cookie:
        request.add_header("Cookie", cookie)
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            body = response.read().decode("utf-8")
            payload = json.loads(body)
            data = (payload or {}).get("data") or {}
            task_id = data.get("taskId")
            business_ok = payload.get("code") == 0 and bool(task_id)
            error = None if business_ok else (payload.get("message") or "missing taskId")
            return Sample(business_ok, response.status, int((time.perf_counter() - started) * 1000), task_id=task_id, error=error)
    except urllib.error.HTTPError as exc:
        return Sample(False, exc.code, int((time.perf_counter() - started) * 1000), error=exc.reason)
    except Exception as exc:
        return Sample(False, 0, int((time.perf_counter() - started) * 1000), error=str(exc))


def summarize(samples: List[Sample]) -> Dict[str, object]:
    elapsed = [sample.elapsed_ms for sample in samples]
    ok_count = sum(1 for sample in samples if sample.ok)
    status_counts: Dict[int, int] = {}
    error_counts: Dict[str, int] = {}
    for sample in samples:
        status_counts[sample.status_code] = status_counts.get(sample.status_code, 0) + 1
        if sample.error:
            error_counts[sample.error] = error_counts.get(sample.error, 0) + 1
    return {
        "total": len(samples),
        "ok": ok_count,
        "failed": len(samples) - ok_count,
        "successRate": round(ok_count / len(samples), 4) if samples else 0,
        "avgMs": round(statistics.mean(elapsed), 2) if elapsed else 0,
        "p95Ms": percentile(elapsed, 95),
        "maxMs": max(elapsed) if elapsed else 0,
        "statusCounts": status_counts,
        "errorCounts": error_counts,
    }


def percentile(values: List[int], p: int) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((p / 100) * (len(ordered) - 1))))
    return ordered[index]


def run_load(base_url: str, total: int, concurrency: int, text: str, language: str, cookie: Optional[str]) -> List[Sample]:
    samples: List[Sample] = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(submit_once, base_url, text, language, cookie) for _ in range(total)]
        for future in as_completed(futures):
            samples.append(future.result())
    return samples


def parse_args():
    parser = argparse.ArgumentParser(description="Load test the async MSA-Net analysis submit endpoint.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/api")
    parser.add_argument("--total", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--text", default="I am happy with this high concurrency inference service.")
    parser.add_argument("--language", default="en", choices=["en", "zh"])
    parser.add_argument("--cookie", default=None, help="Login cookie, for example: JSESSIONID=...")
    return parser.parse_args()


def main():
    args = parse_args()
    started = time.perf_counter()
    samples = run_load(args.base_url, args.total, args.concurrency, args.text, args.language, args.cookie)
    report = summarize(samples)
    report["wallMs"] = int((time.perf_counter() - started) * 1000)
    report["qps"] = round(args.total / max(report["wallMs"] / 1000, 0.001), 2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
