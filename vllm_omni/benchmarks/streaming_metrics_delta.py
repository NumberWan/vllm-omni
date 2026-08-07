"""Convert cumulative WS stage metrics into per-turn deltas for streaming bench."""
from __future__ import annotations

import copy
from typing import Any

# Counters / durations that the server reports cumulatively across turns in one session.
_CUMULATIVE_DELTA_KEYS = (
    "stage_gen_time_ms",
    "num_tokens_out",
    "num_tokens_in",
    "output_unit_count",
    "audio_duration_s",
    "audio_frames",
)

# Per-turn latency snapshots — keep the current turn value, do not subtract previous.
_PER_TURN_ABSOLUTE_KEYS = (
    "serving_time_to_first_output_ms",
    "time_per_output_unit_ms",
    "vllm_ttft_ms",
    "vllm_tpot_ms",
)

_LIST_DELTA_KEYS = ("inter_output_latencies_ms", "vllm_itls_ms")


def _stage_value(stage: dict[str, Any], key: str) -> float:
    value = stage.get(key)
    if value is None:
        return 0.0
    return float(value)


def _delta_stage_snapshot(current: dict[str, Any], previous: dict[str, Any]) -> dict[str, Any]:
    out = dict(current)
    for key in _CUMULATIVE_DELTA_KEYS:
        if key in current:
            out[key] = _stage_value(current, key) - _stage_value(previous, key)
    for key in _PER_TURN_ABSOLUTE_KEYS:
        if key in current:
            out[key] = _stage_value(current, key)
    for key in _LIST_DELTA_KEYS:
        cur_list = current.get(key)
        if not isinstance(cur_list, list):
            continue
        prev_list = previous.get(key)
        if isinstance(prev_list, list) and len(cur_list) >= len(prev_list):
            new_items = [float(x) for x in cur_list[len(prev_list) :]]
        else:
            new_items = [float(x) for x in cur_list]
        out[key] = new_items
        if key == "inter_output_latencies_ms" and new_items:
            out["inter_output_latency_ms"] = sum(new_items) / len(new_items)
    return out


def _lookup_stage(stage_metrics: dict[str, Any], sid: str | int) -> dict[str, Any]:
    stage = stage_metrics.get(str(sid))
    if isinstance(stage, dict):
        return stage
    stage = stage_metrics.get(sid)
    return stage if isinstance(stage, dict) else {}


def _delta_cumulative_text_done_metrics(
    current: dict[str, Any],
    previous: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not current:
        return None
    if previous is None:
        return copy.deepcopy(current)

    delta: dict[str, Any] = {}
    for key in ("e2el_ms", "num_tokens_out", "ttft_ms", "audio_duration_s", "audio_ttfp_ms", "audio_rtf"):
        if current.get(key) is not None:
            delta[key] = current[key]

    cur_sm = current.get("stage_metrics")
    prev_sm = previous.get("stage_metrics")
    if not isinstance(cur_sm, dict):
        return delta or copy.deepcopy(current)

    prev_sm = prev_sm if isinstance(prev_sm, dict) else {}
    delta_sm: dict[str, Any] = {}
    for sid, stage in cur_sm.items():
        if not isinstance(stage, dict):
            continue
        delta_sm[str(sid)] = _delta_stage_snapshot(stage, _lookup_stage(prev_sm, sid))
    delta["stage_metrics"] = delta_sm
    return delta


def _delta_cumulative_audio_done_metrics(
    current: dict[str, Any],
    previous: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not current:
        return None
    if previous is None:
        return copy.deepcopy(current)

    cur_sm = current.get("stage_metrics")
    if not isinstance(cur_sm, dict):
        return copy.deepcopy(current)

    prev_sm = previous.get("stage_metrics")
    prev_sm = prev_sm if isinstance(prev_sm, dict) else {}
    delta_sm: dict[str, Any] = {}
    for sid, stage in cur_sm.items():
        if str(sid) not in ("2", "3") or not isinstance(stage, dict):
            continue
        delta_sm[str(sid)] = _delta_stage_snapshot(stage, _lookup_stage(prev_sm, sid))

    if not delta_sm:
        return copy.deepcopy(current)

    delta: dict[str, Any] = {}
    for key in ("e2el_ms", "audio_duration_s", "audio_ttfp_ms", "audio_rtf"):
        if current.get(key) is not None:
            delta[key] = current[key]
    delta["stage_metrics"] = delta_sm
    return delta
