from __future__ import annotations

from vllm_omni.benchmarks.streaming_metrics_delta import (
    _delta_cumulative_audio_done_metrics,
    _delta_cumulative_text_done_metrics,
)


def test_delta_cumulative_audio_done_metrics_converts_stage23_to_per_turn() -> None:
    previous = {
        "stage_metrics": {
            "2": {
                "num_tokens_out": 120,
                "stage_gen_time_ms": 500.0,
                "output_unit_count": 120,
            },
            "3": {
                "audio_duration_s": 12.5,
                "audio_frames": 300000,
                "stage_gen_time_ms": 6000.0,
                "output_unit_count": 312,
                "inter_output_latencies_ms": [10.0, 12.0],
            },
        }
    }
    current = {
        "stage_metrics": {
            "2": {
                "num_tokens_out": 180,
                "stage_gen_time_ms": 740.0,
                "output_unit_count": 180,
            },
            "3": {
                "audio_duration_s": 18.0,
                "audio_frames": 432000,
                "stage_gen_time_ms": 8900.0,
                "output_unit_count": 450,
                "inter_output_latencies_ms": [10.0, 12.0, 14.0, 16.0],
            },
        }
    }

    delta = _delta_cumulative_audio_done_metrics(current, previous)

    assert delta is not None
    assert delta["stage_metrics"]["2"]["num_tokens_out"] == 60
    assert delta["stage_metrics"]["2"]["stage_gen_time_ms"] == 240.0
    assert delta["stage_metrics"]["3"]["audio_duration_s"] == 5.5
    assert delta["stage_metrics"]["3"]["audio_frames"] == 132000
    assert delta["stage_metrics"]["3"]["stage_gen_time_ms"] == 2900.0
    assert delta["stage_metrics"]["3"]["output_unit_count"] == 138
    assert delta["stage_metrics"]["3"]["inter_output_latencies_ms"] == [14.0, 16.0]
    assert delta["stage_metrics"]["3"]["inter_output_latency_ms"] == 15.0


def test_delta_cumulative_streaming_metrics_text_done_stage1_delta() -> None:
    previous = {
        "e2el_ms": 1200.0,
        "num_tokens_out": 40,
        "stage_metrics": {
            "1": {
                "stage_gen_time_ms": 900.0,
                "num_tokens_out": 40,
                "vllm_ttft_ms": 80.0,
            }
        },
    }
    current = {
        "e2el_ms": 2100.0,
        "num_tokens_out": 62,
        "stage_metrics": {
            "1": {
                "stage_gen_time_ms": 1500.0,
                "num_tokens_out": 62,
                "vllm_ttft_ms": 95.0,
            }
        },
    }

    delta = _delta_cumulative_text_done_metrics(current, previous)

    assert delta is not None
    assert delta["e2el_ms"] == 2100.0
    assert delta["num_tokens_out"] == 62
    assert delta["stage_metrics"]["1"]["stage_gen_time_ms"] == 600.0
    assert delta["stage_metrics"]["1"]["num_tokens_out"] == 22
    assert delta["stage_metrics"]["1"]["vllm_ttft_ms"] == 95.0


def test_delta_preserves_per_turn_ttft_when_current_is_lower() -> None:
    previous = {
        "stage_metrics": {
            "1": {"vllm_ttft_ms": 80.0, "num_tokens_out": 10, "stage_gen_time_ms": 100.0},
        }
    }
    current = {
        "stage_metrics": {
            "1": {"vllm_ttft_ms": 25.0, "num_tokens_out": 12, "stage_gen_time_ms": 130.0},
        }
    }
    delta = _delta_cumulative_text_done_metrics(current, previous)
    assert delta is not None
    assert delta["stage_metrics"]["1"]["vllm_ttft_ms"] == 25.0
    assert delta["stage_metrics"]["1"]["num_tokens_out"] == 2


def test_delta_cumulative_audio_done_metrics_keeps_first_audio_turn_unchanged() -> None:
    current = {
        "stage_metrics": {
            "3": {
                "audio_duration_s": 6.25,
                "audio_frames": 150000,
                "stage_gen_time_ms": 3200.0,
            }
        }
    }

    delta = _delta_cumulative_audio_done_metrics(current, None)

    assert delta == current
