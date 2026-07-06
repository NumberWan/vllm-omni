from __future__ import annotations

import json
from pathlib import Path

from vllm_omni.benchmarks.format_bench_report import (
    build_summary_from_result,
    format_bench_report,
    write_bench_reports,
)


def test_format_bench_report_from_fixture(tmp_path: Path) -> None:
    result = {
        "duration": 10.0,
        "completed": 1,
        "failed": 0,
        "total_input_tokens": 10,
        "total_output_tokens": 20,
        "request_throughput": 0.1,
        "output_throughput": 2.0,
        "total_token_throughput": 3.0,
        "max_concurrent_requests": 1.0,
        "mean_e2el_ms": 1000.0,
        "median_e2el_ms": 900.0,
        "p99_e2el_ms": 1200.0,
        "mean_ttft_ms": 100.0,
        "median_ttft_ms": 90.0,
        "p99_ttft_ms": 150.0,
        "mean_tpot_ms": 5.0,
        "median_tpot_ms": 4.0,
        "p99_tpot_ms": 8.0,
        "mean_itl_ms": 0.0,
        "median_itl_ms": 0.0,
        "p99_itl_ms": 0.0,
        "total_audio_duration_s": 1.5,
        "total_audio_frames": 36000,
        "audio_throughput": 0.15,
        "mean_audio_ttfp_ms": 200.0,
        "median_audio_ttfp_ms": 180.0,
        "p99_audio_ttfp_ms": 250.0,
        "mean_audio_rtf": 0.5,
        "median_audio_rtf": 0.4,
        "p99_audio_rtf": 0.8,
        "omniinteract_evaluated": 2,
        "omniinteract_ia_qtf1": 0.42,
        "omniinteract_ids": {"NOR": 1.0, "PAQ": 0.0, "CSM_SR": 0.0, "CSM_AS_seconds": 0.0},
        "omniinteract_nccs": 0.0,
        "per_requests": [
            {
                "video_id": "0002",
                "video_path": "/data/1q1a/videos/0002.mp4",
                "subset": "1q1a",
                "success": True,
                "latency_s": 10.0,
                "output_tokens": 20,
                "stage_metrics": {
                    "0": {"stage_gen_time_ms": 25.0, "serving_time_to_first_output_ms": 15.0},
                    "1": {
                        "stage_gen_time_ms": 120.0,
                        "num_tokens_out": 20,
                        "vllm_ttft_ms": 80.0,
                        "vllm_tpot_ms": 5.0,
                    },
                },
                "streaming_chunks": [
                    {
                        "timestamp": [1.0, 2.0],
                        "text": "好的",
                        "is_silent": False,
                        "e2el_ms": 120.0,
                        "metrics": {
                            "stage_metrics": {
                                "1": {"stage_gen_time_ms": 80.0, "num_tokens_out": 10, "vllm_ttft_ms": 40.0},
                            }
                        },
                    },
                    {
                        "timestamp": [3.0, 4.0],
                        "text": "<|silent|>",
                        "is_silent": True,
                        "e2el_ms": 25.0,
                        "metrics": {
                            "stage_metrics": {
                                "1": {"stage_gen_time_ms": 20.0, "num_tokens_out": 1, "vllm_ttft_ms": 15.0},
                            }
                        },
                    },
                ],
            }
        ],
    }
    summary = build_summary_from_result(result, label="1q1a (1 videos)", video_line=result["per_requests"][0]["video_path"])
    report = format_bench_report(summary)
    assert "# vllm-omni OmniInteract streaming" in report
    assert "============ Stage 1 (aura) ============" in report
    assert "============ OmniInteract QA metrics ============" in report
    assert "============ Spoken turns ============" in report
    assert "============ Silent turns ============" in report
    assert "spoken=1, silent=1" in report
    assert summary["successful_requests"] == 1
    assert summary["spoken_class"]["turn_count"] == 1
    assert summary["silent_class"]["turn_count"] == 1
    assert summary["video_sessions"] == 1


def test_spoken_class_audio_backfill_from_per_request(tmp_path: Path) -> None:
    """Per-turn chunks may only carry ASR/aura metrics; audio totals come from per_request."""
    result = {
        "duration": 100.0,
        "completed": 1,
        "failed": 0,
        "endpoint_type": "openai-video-stream",
        "per_requests": [
            {
                "video_id": "0002",
                "video_path": "/data/1q1a/videos/0002.mp4",
                "subset": "1q1a",
                "success": True,
                "audio_duration_s": 12.5,
                "audio_frames": 300000,
                "stage_metrics": {
                    "3": {
                        "stage_gen_time_ms": 40.0,
                        "audio_duration_s": 12.5,
                        "audio_frames": 300000,
                        "serving_time_to_first_output_ms": 850.0,
                        "audio_rtf": 0.42,
                    },
                },
                "streaming_chunks": [
                    {
                        "timestamp": [1.0, 2.0],
                        "text": "好的",
                        "is_silent": False,
                        "metrics": {
                            "stage_metrics": {
                                "1": {"stage_gen_time_ms": 80.0, "num_tokens_out": 10, "vllm_ttft_ms": 40.0},
                            }
                        },
                    },
                ],
            }
        ],
    }
    summary = build_summary_from_result(result, label="1q1a (1 videos)", video_line="0002")
    spoken = summary["spoken_class"]
    assert spoken["total_audio_duration_s"] == 12.5
    assert spoken["total_audio_frames"] == 300000
    assert spoken["audio_ttfp_ms"]["mean"] == 850.0
    assert spoken["audio_rtf"]["mean"] == 0.42


def test_silent_class_does_not_inherit_session_audio(tmp_path: Path) -> None:
    """Session-level audio totals must not leak into the silent turn-class summary."""
    result = {
        "duration": 100.0,
        "completed": 1,
        "failed": 0,
        "endpoint_type": "openai-video-stream",
        "per_requests": [
            {
                "video_id": "0002",
                "video_path": "/data/1q1a/videos/0002.mp4",
                "subset": "1q1a",
                "success": True,
                "audio_duration_s": 12.5,
                "audio_frames": 300000,
                "stage_metrics": {
                    "3": {
                        "audio_duration_s": 12.5,
                        "audio_frames": 300000,
                        "serving_time_to_first_output_ms": 850.0,
                    },
                },
                "streaming_chunks": [
                    {
                        "timestamp": [1.0, 2.0],
                        "text": "<|silent|>",
                        "is_silent": True,
                        "metrics": {
                            "stage_metrics": {
                                "1": {"stage_gen_time_ms": 80.0, "num_tokens_out": 4, "vllm_ttft_ms": 40.0},
                            }
                        },
                    },
                ],
            }
        ],
    }
    summary = build_summary_from_result(result, label="1q1a (1 videos)", video_line="0002")
    silent = summary["silent_class"]
    assert silent["turn_count"] == 1
    assert silent["total_audio_duration_s"] == 0.0
    assert silent["total_audio_frames"] == 0
    assert silent["audio_ttfp_ms"]["mean"] == 0.0

    json_path = tmp_path / "result.json"
    json_path.write_text(json.dumps(result), encoding="utf-8")
    report_path = write_bench_reports(result, result_dir=tmp_path, json_path=json_path)
    assert report_path.exists()
    assert (tmp_path / "videos" / "0002" / "bench_report.md").exists()
    assert (tmp_path / "videos" / "0002" / "streaming_chunks.json").exists()


def test_spoken_class_uses_top_level_num_tokens_out_when_stage1_zero() -> None:
    """WS streaming chunks often carry num_tokens_out only on metrics, not stage-1 snapshot."""
    result = {
        "duration": 100.0,
        "completed": 1,
        "failed": 0,
        "endpoint_type": "openai-video-stream",
        "per_requests": [
            {
                "video_id": "0002",
                "video_path": "/data/1q1a/videos/0002.mp4",
                "subset": "1q1a",
                "success": True,
                "streaming_chunks": [
                    {
                        "timestamp": [1.0, 2.0],
                        "text": "好的",
                        "is_silent": False,
                        "metrics": {
                            "num_tokens_out": 36,
                            "stage_metrics": {
                                "1": {
                                    "stage_gen_time_ms": 0.0,
                                    "num_tokens_out": 0,
                                    "vllm_ttft_ms": 96.0,
                                    "vllm_tpot_ms": 6.6,
                                },
                            },
                        },
                    },
                    {
                        "timestamp": [3.0, 4.0],
                        "text": "<|silent|>",
                        "is_silent": True,
                        "metrics": {
                            "num_tokens_out": 2,
                            "stage_metrics": {"1": {"num_tokens_out": 0}},
                        },
                    },
                ],
            }
        ],
    }
    summary = build_summary_from_result(result, label="1q1a (1 videos)", video_line="0002")
    spoken = summary["spoken_class"]
    assert spoken["total_generated_tokens"] == 36
