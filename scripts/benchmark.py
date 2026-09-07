"""Offline concurrency benchmark for the in-process diagnostic service."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from devassist.config import Settings  # noqa: E402
from devassist.evaluation.runner import load_cases  # noqa: E402
from devassist.models import DiagnoseRequest  # noqa: E402
from devassist.service import DevAssistService  # noqa: E402


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * quantile)))
    return round(ordered[index], 3)


def run_benchmark(*, requests: int, concurrency: int) -> dict[str, Any]:
    if requests <= 0 or concurrency <= 0:
        raise ValueError("requests and concurrency must be positive")
    settings = Settings.from_env(ROOT)
    service = DevAssistService(settings)
    cases = [case for case in load_cases(settings.evaluation_path) if case.answerable]
    payloads = [
        DiagnoseRequest(query=case.query, project=case.project, version=case.version)
        for case in cases
    ]

    # Warm vectorizer transforms and Python import caches before measurement.
    for payload in payloads[: min(5, len(payloads))]:
        service.diagnose(payload)

    def execute(index: int) -> tuple[float, bool]:
        payload = payloads[index % len(payloads)]
        started = time.perf_counter()
        response = service.diagnose(payload)
        return (time.perf_counter() - started) * 1_000.0, response.status == "answered"

    wall_started = time.perf_counter()
    results: list[tuple[float, bool]] = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(execute, index) for index in range(requests)]
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception:  # pragma: no cover - reported as a benchmark error
                results.append((0.0, False))
    wall_seconds = time.perf_counter() - wall_started
    latencies = [latency for latency, _ in results if latency > 0]
    successes = sum(answered for _, answered in results)
    return {
        "metadata": {
            "generated_at": datetime.now(UTC).isoformat(),
            "scope": "in-process offline benchmark; excludes HTTP/network/LLM latency",
            "python": platform.python_version(),
            "platform": platform.platform(),
            "documents": service.document_count,
            "requests": requests,
            "concurrency": concurrency,
        },
        "results": {
            "successes": successes,
            "errors": requests - successes,
            "error_rate": round((requests - successes) / requests, 6),
            "throughput_requests_per_second": round(requests / wall_seconds, 3),
            "mean_latency_ms": round(statistics.fmean(latencies), 3) if latencies else 0.0,
            "p50_latency_ms": percentile(latencies, 0.50),
            "p95_latency_ms": percentile(latencies, 0.95),
            "p99_latency_ms": percentile(latencies, 0.99),
            "wall_seconds": round(wall_seconds, 3),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "benchmark.json")
    args = parser.parse_args()
    report = run_benchmark(requests=args.requests, concurrency=args.concurrency)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return int(report["results"]["errors"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
