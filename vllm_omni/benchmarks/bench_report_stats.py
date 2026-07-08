"""Percentile helpers for benchmark markdown reports."""
from __future__ import annotations

import statistics
from typing import Sequence


def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(float(v) for v in values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (k - f) * (s[c] - s[f])


def stats_ms(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p99": 0.0}
    return {
        "mean": float(statistics.mean(values)),
        "median": float(statistics.median(values)),
        "p99": float(percentile(values, 0.99)),
    }


POSITIVE_ONLY_STAGE_KEYS = frozenset({
    "serving_time_to_first_output_ms",
    "time_per_output_unit_ms",
    "inter_output_latencies_ms",
    "vllm_ttft_ms",
    "vllm_tpot_ms",
    "vllm_itls_ms",
})


def stats_ms_positive(values: Sequence[float]) -> dict[str, float]:
    positive = [float(v) for v in values if float(v) > 0]
    if not positive:
        return {}
    return stats_ms(positive)
