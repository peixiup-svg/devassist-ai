"""In-process operational metrics suitable for the local demo."""

from __future__ import annotations

import math
import threading
from collections import deque

from devassist.models import MetricsResponse


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = max(0, math.ceil(percentile * len(ordered)) - 1)
    return round(ordered[position], 3)


class AppMetrics:
    def __init__(self, history_size: int = 1_000) -> None:
        self._lock = threading.Lock()
        self._latencies: deque[float] = deque(maxlen=history_size)
        self._requests = 0
        self._answers = 0
        self._refusals = 0
        self._feedback = 0

    def record_diagnosis(self, *, answered: bool, latency_ms: float) -> None:
        with self._lock:
            self._requests += 1
            self._answers += int(answered)
            self._refusals += int(not answered)
            self._latencies.append(latency_ms)

    def record_feedback(self) -> None:
        with self._lock:
            self._feedback += 1

    def snapshot(self) -> MetricsResponse:
        with self._lock:
            latencies = list(self._latencies)
            return MetricsResponse(
                requests=self._requests,
                answers=self._answers,
                refusals=self._refusals,
                feedback_items=self._feedback,
                p50_latency_ms=_percentile(latencies, 0.50),
                p95_latency_ms=_percentile(latencies, 0.95),
            )
