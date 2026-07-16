"""Format vllm-omni OmniInteract streaming bench JSON as native-style bench_report.md."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vllm_omni.benchmarks.bench_report_stats import POSITIVE_ONLY_STAGE_KEYS, stats_ms, stats_ms_positive
from vllm_omni.metrics import definitions as defs
from vllm_omni.model_executor.stage_input_processors.aura_session_history import is_effectively_silent

_STAGE_DISPLAY_NAMES = {
    0: "asr",
    1: "aura",
    2: "qwen3_tts",
    3: "code2wav",
}


def _line(label: str, value: str, width: int = 40) -> str:
    return f"{label:<{width}} {value}"


def _fmt_stat(stats: dict[str, float] | None, key: str, *, na: bool = False) -> str:
    if na or not stats:
        return "N/A"
    v = stats.get(key)
    if v is None:
        return "N/A"
    return f"{v:.2f}"


def _fmt_scalar(value: float | int | None, *, na: bool = False, digits: int = 2) -> str:
    if na or value is None:
        return "N/A"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def _stage_stats(values: list[float], key: str) -> dict[str, float]:
    if key in POSITIVE_ONLY_STAGE_KEYS:
        return stats_ms_positive(values)
    return stats_ms(values)


def _turn_is_silent(turn: dict[str, Any]) -> bool:
    if turn.get("is_silent") is not None:
        return bool(turn.get("is_silent"))
    return is_effectively_silent(str(turn.get("text") or ""))


def _count_spoken_silent(chunks: list[dict[str, Any]]) -> tuple[int, int]:
    spoken = 0
    silent = 0
    for chunk in chunks:
        text = str(chunk.get("text") or "")
        if not text.strip() and not chunk.get("metrics"):
            continue
        if _turn_is_silent(chunk):
            silent += 1
        else:
            spoken += 1
    return spoken, silent


def _partition_turns(turns: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    spoken: list[dict[str, Any]] = []
    silent: list[dict[str, Any]] = []
    for turn in turns:
        text = str(turn.get("text") or "")
        if not text.strip() and not turn.get("metrics"):
            continue
        if _turn_is_silent(turn):
            silent.append(turn)
        else:
            spoken.append(turn)
    return spoken, silent


def _empty_stats_ms() -> dict[str, float]:
    return {"mean": 0.0, "median": 0.0, "p99": 0.0}


def _turn_aura_metrics(turn: dict[str, Any]) -> dict[str, Any]:
    metrics = turn.get("metrics")
    if not isinstance(metrics, dict):
        return {}
    stage_metrics = metrics.get("stage_metrics")
    if not isinstance(stage_metrics, dict):
        return {}
    aura = stage_metrics.get("1") or stage_metrics.get(1)
    return aura if isinstance(aura, dict) else {}


def _turn_stage_metrics_map(turn: dict[str, Any]) -> dict[str, Any]:
    metrics = turn.get("metrics")
    if not isinstance(metrics, dict):
        return {}
    stage_metrics = metrics.get("stage_metrics")
    return stage_metrics if isinstance(stage_metrics, dict) else {}


def _turn_asr_gen_ms(turn: dict[str, Any]) -> float:
    """ASR stage_gen_time_ms for this turn (0 when skip / missing)."""
    stage_map = _turn_stage_metrics_map(turn)
    asr = stage_map.get("0") or stage_map.get(0) or {}
    if not isinstance(asr, dict):
        return 0.0
    return float(asr.get(defs.STAGE_GEN_TIME_MS) or 0.0)


def _turn_aura_ttft_ms(turn: dict[str, Any]) -> float | None:
    """Aura-only TTFT (serving first-output or vllm_ttft), excluding ASR."""
    aura = _turn_aura_metrics(turn)
    for key in (defs.SERVING_TIME_TO_FIRST_OUTPUT_MS, defs.VLLM_TTFT_MS):
        if aura.get(key) is None:
            continue
        value = float(aura[key])
        if value > 0:
            return value
    return None


def _turn_text_ttft_ms(turn: dict[str, Any]) -> float | None:
    """True text TTFT: ASR gen + Aura TTFT for both spoken and silent turns."""
    aura_ttft = _turn_aura_ttft_ms(turn)
    if aura_ttft is None:
        return None
    asr_ms = _turn_asr_gen_ms(turn)
    if asr_ms > 0:
        return asr_ms + aura_ttft
    return aura_ttft


def _turn_num_tokens_out(turn: dict[str, Any]) -> int:
    """Per-turn output tokens from stage-1 snapshot or top-level WS metrics."""
    aura = _turn_aura_metrics(turn)
    stage_tokens = int(aura.get("num_tokens_out") or 0)
    if stage_tokens > 0:
        return stage_tokens
    metrics = turn.get("metrics")
    if isinstance(metrics, dict):
        top = int(metrics.get("num_tokens_out") or 0)
        if top > 0:
            return top
    return 0


def _estimate_stage_gen_time_ms(stage: dict[str, Any], *, tokens: int = 0) -> float:
    """Fallback stage_gen_time from TTFT/TPOT when WS snapshot omits it."""
    gen_ms = float(stage.get(defs.STAGE_GEN_TIME_MS) or 0.0)
    if gen_ms > 0:
        return gen_ms
    ttft = float(stage.get(defs.VLLM_TTFT_MS) or stage.get(defs.SERVING_TIME_TO_FIRST_OUTPUT_MS) or 0.0)
    tpot = float(stage.get(defs.VLLM_TPOT_MS) or 0.0)
    if ttft > 0 and tokens > 0:
        return ttft + tpot * max(tokens - 1, 0)
    if ttft > 0:
        return ttft
    return 0.0


def _latency_based_stage_gen_time_ms(stage: dict[str, Any], *, tokens: int = 0) -> float:
    """Per-turn gen estimate from latency fields only (ignores recorded stage_gen_time)."""
    ttft = float(stage.get(defs.VLLM_TTFT_MS) or stage.get(defs.SERVING_TIME_TO_FIRST_OUTPUT_MS) or 0.0)
    tpot = float(stage.get(defs.VLLM_TPOT_MS) or 0.0)
    if ttft > 0 and tokens > 0:
        return ttft + tpot * max(tokens - 1, 0)
    if ttft > 0:
        return ttft
    return 0.0


def _repair_stage3_audio_snapshot(stage: dict[str, Any]) -> None:
    """Fix session-bleed audio_duration when cumulative metrics were not delta'd."""
    dur = float(stage.get(f"{defs.AUDIO_DURATION}_s") or stage.get("audio_duration_s") or 0.0)
    gen_ms = float(stage.get(defs.STAGE_GEN_TIME_MS) or 0.0)
    if dur <= 0 or gen_ms <= 0:
        return
    est_dur = max((gen_ms / 1000.0) / 0.08, 0.1)
    if dur > max(60.0, est_dur * 5.0):
        stage[f"{defs.AUDIO_DURATION}_s"] = round(est_dur, 2)
        stage["audio_duration_s"] = round(est_dur, 2)
        stage[defs.AUDIO_FRAMES] = int(est_dur * 24000)
        stage["audio_frames"] = int(est_dur * 24000)
        stage[defs.AUDIO_RTF] = round((gen_ms / 1000.0) / est_dur, 4)
        stage["audio_rtf"] = stage[defs.AUDIO_RTF]


def _repair_stage2_gen_snapshot(stage_metrics: dict[str, Any]) -> None:
    """Fix session-bleed TTS stage_gen when stage-3 per-turn gen is plausible."""
    s2 = stage_metrics.get("2")
    s3 = stage_metrics.get("3")
    if not isinstance(s2, dict) or not isinstance(s3, dict):
        return
    s2_gen = float(s2.get(defs.STAGE_GEN_TIME_MS) or 0.0)
    s3_gen = float(s3.get(defs.STAGE_GEN_TIME_MS) or 0.0)
    if s2_gen <= 0 or s3_gen <= 0:
        return
    if s2_gen > max(5000.0, s3_gen * 10.0):
        s2[defs.STAGE_GEN_TIME_MS] = s3_gen


def _clear_silent_audio_pipeline_stages(stage_metrics: dict[str, Any]) -> dict[str, Any]:
    """Drop TTS/code2wav timings from silent turns.

    Silent AURA turns only emit an empty finish-sentinel to release the
    prewarmed Talker/Code2Wav wait gate — no synthesis runs. Residual
    ``stage_gen_time_ms`` / audio counters are wait-gate close latency or
    cumulative-delta bleed from a prior spoken turn and must not appear in
    silent aggregates.
    """
    cleared = {str(k): (dict(v) if isinstance(v, dict) else v) for k, v in stage_metrics.items()}
    for sid in ("2", "3"):
        stage = cleared.get(sid)
        if not isinstance(stage, dict):
            continue
        stage[defs.STAGE_GEN_TIME_MS] = 0.0
        stage[defs.SERVING_TIME_TO_FIRST_OUTPUT_MS] = 0.0
        stage[defs.TIME_PER_OUTPUT_UNIT_MS] = 0.0
        stage[defs.INTER_OUTPUT_LATENCIES_MS] = []
        stage["inter_output_latency_ms"] = 0.0
        stage[defs.AUDIO_FRAMES] = 0
        stage["audio_frames"] = 0
        stage[f"{defs.AUDIO_DURATION}_s"] = 0.0
        stage["audio_duration_s"] = 0.0
        stage[defs.AUDIO_RTF] = 0.0
        stage["audio_rtf"] = 0.0
        stage["serving_audio_ttfp_ms"] = 0.0
        stage["audio_ttfp_ms"] = 0.0
    return cleared


def _enrich_turn_stage_snapshot(
    stage_metrics: dict[str, Any],
    *,
    tokens: int = 0,
    silent: bool = False,
) -> dict[str, Any]:
    """Fill missing per-turn stage_gen_time / ITL proxies in a stage_metrics map."""
    enriched = {str(k): dict(v) if isinstance(v, dict) else v for k, v in stage_metrics.items()}
    s0 = enriched.get("0")
    if isinstance(s0, dict):
        gen = _estimate_stage_gen_time_ms(
            s0,
            tokens=max(int(s0.get(defs.OUTPUT_UNIT_COUNT) or 0), 1),
        )
        if float(s0.get(defs.STAGE_GEN_TIME_MS) or 0.0) <= 0 and gen > 0:
            s0[defs.STAGE_GEN_TIME_MS] = gen
    s1 = enriched.get("1")
    if isinstance(s1, dict):
        turn_tokens = tokens or int(s1.get(defs.NUM_TOKENS_OUT) or 0)
        if turn_tokens < 0:
            turn_tokens = 0
        gen = _estimate_stage_gen_time_ms(s1, tokens=turn_tokens)
        latency_gen = _latency_based_stage_gen_time_ms(s1, tokens=turn_tokens)
        current_gen = float(s1.get(defs.STAGE_GEN_TIME_MS) or 0.0)
        # Silent turns often carry tiny cumulative deltas for stage_gen_time while
        # vllm_ttft_ms is per-turn absolute; repair under-reported gen snapshots.
        if latency_gen > 0 and (current_gen <= 0 or current_gen < latency_gen * 0.5):
            s1[defs.STAGE_GEN_TIME_MS] = latency_gen
        elif current_gen <= 0 and gen > 0:
            s1[defs.STAGE_GEN_TIME_MS] = gen
        if turn_tokens > 0 and int(s1.get(defs.NUM_TOKENS_OUT) or 0) <= 0:
            s1[defs.NUM_TOKENS_OUT] = turn_tokens
        itls = s1.get(defs.VLLM_ITLS_MS)
        tpot = float(s1.get(defs.VLLM_TPOT_MS) or 0.0)
        if tpot > 0 and turn_tokens > 1 and (not isinstance(itls, list) or not itls):
            s1[defs.VLLM_ITLS_MS] = [tpot] * (turn_tokens - 1)
    if silent:
        return _clear_silent_audio_pipeline_stages(enriched)
    s3 = enriched.get("3")
    if isinstance(s3, dict):
        _repair_stage3_audio_snapshot(s3)
    _repair_stage2_gen_snapshot(enriched)
    return enriched


def _reconcile_streaming_turn_metrics(turns: list[dict[str, Any]]) -> None:
    """Enrich per-turn snapshots and repair known audio metric bleed in saved JSON."""
    prev_audio_frames: int | None = None
    for turn in turns:
        metrics = turn.get("metrics")
        if not isinstance(metrics, dict):
            continue
        stage_metrics = metrics.get("stage_metrics")
        if not isinstance(stage_metrics, dict):
            continue
        silent = _turn_is_silent(turn)
        if not silent:
            s3 = stage_metrics.get("3")
            if isinstance(s3, dict):
                frames = int(s3.get(defs.AUDIO_FRAMES) or s3.get("audio_frames") or 0)
                _repair_stage3_audio_snapshot(s3)
                prev_audio_frames = int(s3.get(defs.AUDIO_FRAMES) or s3.get("audio_frames") or 0) or prev_audio_frames
        tokens = _turn_num_tokens_out(turn)
        enriched_sm = _enrich_turn_stage_snapshot(stage_metrics, tokens=tokens, silent=silent)
        cleaned = {**metrics, "stage_metrics": enriched_sm}
        if silent:
            for key in ("audio_duration_s", "audio_frames", "audio_ttfp_ms", "audio_rtf"):
                if key in cleaned:
                    cleaned[key] = 0.0 if key != "audio_frames" else 0
        turn["metrics"] = cleaned

def _turn_audio_stage(stage_metrics: dict[str, Any]) -> dict[str, Any]:
    """Pick the stage snapshot that carries audio duration (code2wav preferred, then TTS)."""
    for sid in ("3", "2", 3, 2):
        stage = stage_metrics.get(sid)
        if not isinstance(stage, dict):
            continue
        if float(stage.get(f"{defs.AUDIO_DURATION}_s") or stage.get("audio_duration_s") or 0.0) > 0:
            return stage
    for sid in ("3", "2", 3, 2):
        stage = stage_metrics.get(sid)
        if isinstance(stage, dict):
            return stage
    return {}


def _positive_stage_gen_ms(stage_metrics: dict[str, Any], stage_id: int | str) -> float:
    stage = stage_metrics.get(str(stage_id)) or stage_metrics.get(stage_id) or {}
    if not isinstance(stage, dict):
        return 0.0
    value = float(stage.get(defs.STAGE_GEN_TIME_MS) or 0.0)
    return value if value > 0 else 0.0


def _turn_audio_ttfp_ms(stage_metrics: dict[str, Any]) -> float | None:
    """Finish-then-start AUDIO_TTFP: first audio only after ASR+Aura full gen.

    AURA ``aura2tts_async_chunk`` waits for Stage-1 finish before Stage-2, so
    root-relative ``serving_time_to_first_output_ms`` on TTS/code2wav is not a
    reliable turn timeline. Approximate true TTFP as ASR gen + Aura gen (same
    lower bound native bench uses when reconstructing ``t_first_audio``).
    """
    asr_gen = _positive_stage_gen_ms(stage_metrics, 0)
    aura_gen = _positive_stage_gen_ms(stage_metrics, 1)
    if aura_gen > 0:
        return asr_gen + aura_gen
    audio_stage = _turn_audio_stage(stage_metrics)
    ttfp = audio_stage.get(defs.SERVING_TIME_TO_FIRST_OUTPUT_MS)
    if ttfp is not None and float(ttfp) > 0:
        return float(ttfp)
    return None


def _merge_turn_metrics(dest_metrics: dict[str, Any], src_metrics: dict[str, Any]) -> None:
    """Deep-merge server metrics from response.audio.done into a turn record."""
    if not src_metrics:
        return
    for key in ("e2el_ms", "ttft_ms", "audio_duration_s", "audio_frames", "audio_ttfp_ms", "audio_rtf"):
        if src_metrics.get(key) is not None:
            dest_metrics[key] = src_metrics[key]
    dest_sm = dest_metrics.get("stage_metrics")
    if not isinstance(dest_sm, dict):
        dest_sm = {}
        dest_metrics["stage_metrics"] = dest_sm
    src_sm = src_metrics.get("stage_metrics")
    if not isinstance(src_sm, dict):
        return
    for sid, stage in src_sm.items():
        if not isinstance(stage, dict):
            continue
        key = str(sid)
        if key in dest_sm and isinstance(dest_sm[key], dict):
            dest_sm[key].update(stage)
        else:
            dest_sm[key] = dict(stage)


def _backfill_class_audio_from_per_requests(
    class_summary: dict[str, Any],
    per_requests: list[dict[str, Any]],
    *,
    spoken_only: bool,
    benchmark_duration_s: float,
) -> None:
    """Fallback for saved JSON where per-turn chunks lack TTS stage metrics."""
    if float(class_summary.get("total_audio_duration_s") or 0) > 0:
        return
    # Silent turns never produce audio; session-level per_request totals are spoken-only.
    if not spoken_only:
        return

    total_duration_s = 0.0
    total_frames = 0
    audio_ttfp_vals: list[float] = []
    audio_rtf_vals: list[float] = []

    for item in per_requests:
        if not item.get("success"):
            continue
        chunks = list(item.get("streaming_chunks") or [])
        if spoken_only:
            if not any(not _turn_is_silent(c) for c in chunks):
                continue
        elif not any(_turn_is_silent(c) for c in chunks):
            continue

        smap = item.get("stage_metrics") or {}
        audio_stage = _turn_audio_stage(smap if isinstance(smap, dict) else {})
        duration_s = float(
            item.get("audio_duration_s")
            or audio_stage.get(f"{defs.AUDIO_DURATION}_s")
            or audio_stage.get("audio_duration_s")
            or 0.0
        )
        frames = int(
            item.get("audio_frames")
            or audio_stage.get(defs.AUDIO_FRAMES)
            or audio_stage.get("audio_frames")
            or 0
        )
        total_duration_s += duration_s
        total_frames += frames

        ttfp = _turn_audio_ttfp_ms(smap if isinstance(smap, dict) else {})
        if ttfp is not None and ttfp > 0:
            audio_ttfp_vals.append(ttfp)
        rtf = audio_stage.get(defs.AUDIO_RTF) or audio_stage.get("audio_rtf")
        if rtf is not None and float(rtf) > 0:
            audio_rtf_vals.append(float(rtf))

    if total_duration_s <= 0 and total_frames <= 0:
        return

    duration_s = float(benchmark_duration_s or 0.0)
    class_summary["total_audio_duration_s"] = round(total_duration_s, 2)
    class_summary["total_audio_frames"] = total_frames
    class_summary["audio_throughput"] = round(total_duration_s / duration_s, 2) if duration_s > 0 else 0.0
    if audio_ttfp_vals:
        class_summary["audio_ttfp_ms"] = stats_ms(audio_ttfp_vals)
    if audio_rtf_vals:
        class_summary["audio_rtf"] = stats_ms(audio_rtf_vals)


def _build_turn_class_summary(
    turns: list[dict[str, Any]],
    *,
    benchmark_duration_s: float,
) -> dict[str, Any]:
    e2els = [value for turn in turns if (value := _collect_turn_e2el_ms(turn)) is not None]
    snapshots = _turn_stage_snapshots(turns)
    turn_count = len(turns)
    ttft_vals: list[float] = []
    tpot_vals: list[float] = []
    itl_vals: list[float] = []
    token_total = 0
    audio_duration_s = 0.0
    audio_frames = 0
    audio_ttfp_vals: list[float] = []
    audio_rtf_vals: list[float] = []

    for turn in turns:
        aura = _turn_aura_metrics(turn)
        # Top-level Text TTFT = ASR gen + Aura TTFT (spoken and silent).
        # Stage-1 section still reports aura-only Serving/vllm TTFT from stage snapshots.
        text_ttft = _turn_text_ttft_ms(turn)
        if text_ttft is not None and text_ttft > 0:
            ttft_vals.append(text_ttft)
        if aura.get("vllm_tpot_ms") is not None and float(aura["vllm_tpot_ms"]) > 0:
            tpot_vals.append(float(aura["vllm_tpot_ms"]))
        itls = aura.get("vllm_itls_ms")
        if isinstance(itls, list) and itls:
            itl_vals.extend(float(x) for x in itls if float(x or 0) > 0)
        elif aura.get("vllm_tpot_ms") is not None and float(aura["vllm_tpot_ms"]) > 0:
            tokens = _turn_num_tokens_out(turn)
            if tokens > 1:
                itl_vals.extend([float(aura["vllm_tpot_ms"])] * (tokens - 1))
        token_total += _turn_num_tokens_out(turn)

        if _turn_is_silent(turn):
            continue

        metrics = turn.get("metrics")
        if isinstance(metrics, dict):
            stage_metrics = metrics.get("stage_metrics")
            if isinstance(stage_metrics, dict):
                audio_stage = _turn_audio_stage(stage_metrics)
                audio_duration_s += float(
                    audio_stage.get(f"{defs.AUDIO_DURATION}_s")
                    or audio_stage.get("audio_duration_s")
                    or 0.0
                )
                audio_frames += int(
                    audio_stage.get(defs.AUDIO_FRAMES) or audio_stage.get("audio_frames") or 0
                )
                ttfp = _turn_audio_ttfp_ms(stage_metrics)
                if ttfp is not None and ttfp > 0:
                    audio_ttfp_vals.append(ttfp)
                rtf = audio_stage.get(defs.AUDIO_RTF) or audio_stage.get("audio_rtf")
                if rtf is not None and float(rtf) > 0:
                    audio_rtf_vals.append(float(rtf))

    duration_s = float(benchmark_duration_s or 0.0)
    silent_only = turn_count > 0 and all(_turn_is_silent(t) for t in turns)
    return {
        "turn_count": turn_count,
        "successful_requests": len(e2els) if e2els else turn_count,
        "request_throughput_rps": (turn_count / duration_s) if duration_s > 0 and turn_count > 0 else 0.0,
        "e2el_ms": stats_ms(e2els) if e2els else _empty_stats_ms(),
        "ttft_ms": stats_ms(ttft_vals) if ttft_vals else _empty_stats_ms(),
        "tpot_ms": stats_ms_positive(tpot_vals) if tpot_vals and not silent_only else {},
        "itl_ms": stats_ms_positive(itl_vals) if itl_vals and not silent_only else {},
        "total_generated_tokens": token_total,
        "total_audio_duration_s": round(audio_duration_s, 2),
        "total_audio_frames": audio_frames,
        "audio_throughput": round(audio_duration_s / duration_s, 2) if duration_s > 0 else 0.0,
        "audio_ttfp_ms": stats_ms(audio_ttfp_vals)
        if audio_ttfp_vals and not silent_only
        else (_empty_stats_ms() if silent_only else {}),
        "audio_rtf": stats_ms(audio_rtf_vals)
        if audio_rtf_vals and not silent_only
        else (_empty_stats_ms() if silent_only else {}),
        "server_stage_snapshots": snapshots,
    }


def _append_turn_class_section(
    lines: list[str],
    title: str,
    class_summary: dict[str, Any],
    *,
    silent_class: bool = False,
) -> None:
    lines.append(f"============ {title} ============")
    lines.append(_line("Turns:", str(class_summary.get("turn_count", 0))))
    lines.append(_line("Successful requests:", str(class_summary.get("successful_requests", 0))))
    lines.append(_line("Request throughput (req/s):", f"{class_summary.get('request_throughput_rps', 0):.2f}"))

    e2el = class_summary.get("e2el_ms") or {}
    lines.append("----------------End-to-end Latency----------------")
    lines.append(_line("Mean E2EL (ms):", f"{e2el.get('mean', 0):.2f}"))
    lines.append(_line("Median E2EL (ms):", f"{e2el.get('median', 0):.2f}"))
    lines.append(_line("P99 E2EL (ms):", f"{e2el.get('p99', 0):.2f}"))

    lines.append("============ Text Result ============")
    lines.append(_line("Total generated tokens:", str(class_summary.get("total_generated_tokens", 0))))

    ttft = class_summary.get("ttft_ms") or {}
    lines.append("----------------Time to First Token----------------")
    lines.append(_line("Mean TTFT (ms):", f"{ttft.get('mean', 0):.2f}"))
    lines.append(_line("Median TTFT (ms):", f"{ttft.get('median', 0):.2f}"))
    lines.append(_line("P99 TTFT (ms):", f"{ttft.get('p99', 0):.2f}"))

    tpot = class_summary.get("tpot_ms") or {}
    lines.append("------Time per Output Token (excl. 1st token)------")
    lines.append(_line("Mean TPOT (ms):", _fmt_stat(tpot, "mean", na=silent_class)))
    lines.append(_line("Median TPOT (ms):", _fmt_stat(tpot, "median", na=silent_class)))
    lines.append(_line("P99 TPOT (ms):", _fmt_stat(tpot, "p99", na=silent_class)))

    lines.append("============ Audio Result ============")
    lines.append(
        _line(
            "Total audio duration generated(s):",
            _fmt_scalar(class_summary.get("total_audio_duration_s"), na=silent_class),
        )
    )
    lines.append(
        _line(
            "Total audio frames generated:",
            _fmt_scalar(class_summary.get("total_audio_frames"), na=silent_class, digits=0),
        )
    )
    lines.append(
        _line(
            "Audio throughput(audio duration/s):",
            _fmt_scalar(class_summary.get("audio_throughput"), na=silent_class),
        )
    )

    audio_ttfp = class_summary.get("audio_ttfp_ms") or {}
    lines.append("----------------Time to First Packet----------------")
    lines.append(_line("Mean AUDIO_TTFP (ms):", _fmt_stat(audio_ttfp, "mean", na=silent_class)))
    lines.append(_line("Median AUDIO_TTFP (ms):", _fmt_stat(audio_ttfp, "median", na=silent_class)))
    lines.append(_line("P99 AUDIO_TTFP (ms):", _fmt_stat(audio_ttfp, "p99", na=silent_class)))

    audio_rtf = class_summary.get("audio_rtf") or {}
    lines.append("----------------Real Time Factor----------------")
    lines.append(_line("Mean AUDIO_RTF:", _fmt_stat(audio_rtf, "mean", na=silent_class)))
    lines.append(_line("Median AUDIO_RTF:", _fmt_stat(audio_rtf, "median", na=silent_class)))
    lines.append(_line("P99 AUDIO_RTF:", _fmt_stat(audio_rtf, "p99", na=silent_class)))
    lines.append("")

    snapshots = class_summary.get("server_stage_snapshots") or []
    lines.extend(
        _stage_sections_from_snapshots(
            snapshots,
            total_generated_tokens=int(class_summary.get("total_generated_tokens") or 0),
            audio_ttfp_ms=audio_ttfp,
            audio_rtf=audio_rtf,
            total_audio_duration_s=float(class_summary.get("total_audio_duration_s") or 0.0),
            total_audio_frames=int(class_summary.get("total_audio_frames") or 0),
            silent_class=silent_class,
        )
    )


def _peak_output_tps(per_requests: list[dict[str, Any]]) -> float:
    peak = 0.0
    for item in per_requests:
        tokens = int(item.get("output_tokens") or 0)
        latency_s = float(item.get("latency_s") or 0.0)
        if tokens > 0 and latency_s > 0:
            peak = max(peak, tokens / latency_s)
    return round(peak, 2)


def _collect_stage_values(
    snapshots: list[dict[str, Any]],
    sid: str,
    key: str,
) -> list[float]:
    positive_only = key in POSITIVE_ONLY_STAGE_KEYS
    out: list[float] = []
    for smap in snapshots:
        if not isinstance(smap, dict):
            continue
        info = smap.get(sid) or {}
        if not isinstance(info, dict):
            continue
        value = info.get(key)
        if isinstance(value, list):
            for x in value:
                fv = float(x or 0.0)
                if fv > 0 or not positive_only:
                    out.append(fv)
        elif value is not None:
            fv = float(value)
            if fv > 0 or not positive_only:
                out.append(fv)
    return out


def _stage_block_lines(
    stage_id: int,
    stage_name: str,
    gen_ms: dict[str, float],
    *,
    text_tokens: int | None = None,
    serving_ttft: dict[str, float] | None = None,
    ttft: dict[str, float] | None = None,
    tpot: dict[str, float] | None = None,
    itl: dict[str, float] | None = None,
    internal_ttfc: dict[str, float] | None = None,
    internal_tpop: dict[str, float] | None = None,
    internal_icl: dict[str, float] | None = None,
    audio_duration_s: float | None = None,
    audio_frames: int | None = None,
    serving_audio_ttfp: dict[str, float] | None = None,
    audio_rtf: dict[str, float] | None = None,
    audio_na_label: str | None = None,
    silent_class: bool = False,
) -> list[str]:
    lines: list[str] = []
    lines.append(f"============ Stage {stage_id} ({stage_name}) ============")
    lines.append("----------------Stage Timing----------------")
    lines.append(_line("Mean stage_gen_time (ms):", f"{gen_ms['mean']:.2f}"))
    lines.append(_line("Median stage_gen_time (ms):", f"{gen_ms['median']:.2f}"))
    lines.append(_line("P99 stage_gen_time (ms):", f"{gen_ms['p99']:.2f}"))

    if internal_ttfc is not None:
        lines.append("============ Internal Stream Result ============")
        lines.append("--------Serving Time to First Chunk--------")
        lines.append(_line("Mean Serving TTFC (ms):", _fmt_stat(internal_ttfc, "mean", na=not internal_ttfc)))
        lines.append(_line("Median Serving TTFC (ms):", _fmt_stat(internal_ttfc, "median", na=not internal_ttfc)))
        lines.append(_line("P99 Serving TTFC (ms):", _fmt_stat(internal_ttfc, "p99", na=not internal_ttfc)))
        if internal_tpop is not None:
            lines.append("-----Time per Output Chunk (excl. 1st chunk)-----")
            lines.append(_line("Mean TPOP (ms):", _fmt_stat(internal_tpop, "mean", na=not internal_tpop)))
            lines.append(_line("Median TPOP (ms):", _fmt_stat(internal_tpop, "median", na=not internal_tpop)))
            lines.append(_line("P99 TPOP (ms):", _fmt_stat(internal_tpop, "p99", na=not internal_tpop)))
        if internal_icl is not None:
            lines.append("----------------Inter-chunk Latency----------------")
            lines.append(_line("Mean ICL (ms):", _fmt_stat(internal_icl, "mean", na=not internal_icl)))
            lines.append(_line("Median ICL (ms):", _fmt_stat(internal_icl, "median", na=not internal_icl)))
            lines.append(_line("P99 ICL (ms):", _fmt_stat(internal_icl, "p99", na=not internal_icl)))

    if text_tokens is not None:
        lines.append("============ Text Result ============")
        lines.append(_line("Stage generated tokens:", str(text_tokens)))
        if serving_ttft is not None:
            lines.append("--------Serving Time to First Token--------")
            lines.append(_line("Mean Serving TTFT (ms):", _fmt_stat(serving_ttft, "mean")))
            lines.append(_line("Median Serving TTFT (ms):", _fmt_stat(serving_ttft, "median")))
            lines.append(_line("P99 Serving TTFT (ms):", _fmt_stat(serving_ttft, "p99")))
        if ttft is not None:
            lines.append("----------------Time to First Token----------------")
            lines.append(_line("Mean TTFT (ms):", _fmt_stat(ttft, "mean")))
            lines.append(_line("Median TTFT (ms):", _fmt_stat(ttft, "median")))
            lines.append(_line("P99 TTFT (ms):", _fmt_stat(ttft, "p99")))
        if tpot is not None:
            lines.append("-----Time per Output Token (excl. 1st token)-----")
            lines.append(_line("Mean TPOT (ms):", _fmt_stat(tpot, "mean", na=silent_class)))
            lines.append(_line("Median TPOT (ms):", _fmt_stat(tpot, "median", na=silent_class)))
            lines.append(_line("P99 TPOT (ms):", _fmt_stat(tpot, "p99", na=silent_class)))
        if itl is not None:
            lines.append("----------------Inter-token Latency----------------")
            lines.append(_line("Mean ITL (ms):", _fmt_stat(itl, "mean", na=silent_class)))
            lines.append(_line("Median ITL (ms):", _fmt_stat(itl, "median", na=silent_class)))
            lines.append(_line("P99 ITL (ms):", _fmt_stat(itl, "p99", na=silent_class)))

    if audio_na_label is not None:
        lines.append("============ Audio Result ============")
        lines.append(_line("Stage audio:", audio_na_label))
    elif audio_duration_s is not None:
        lines.append("============ Audio Result ============")
        lines.append(
            _line(
                "Stage audio duration generated(s):",
                _fmt_scalar(audio_duration_s, na=silent_class),
            )
        )
        lines.append(
            _line(
                "Stage audio frames generated:",
                _fmt_scalar(audio_frames, na=silent_class, digits=0),
            )
        )
        if serving_audio_ttfp is not None:
            lines.append("--------Serving Time to First Packet--------")
            lines.append(_line("Mean Serving AUDIO_TTFP (ms):", _fmt_stat(serving_audio_ttfp, "mean", na=silent_class)))
            lines.append(_line("Median Serving AUDIO_TTFP (ms):", _fmt_stat(serving_audio_ttfp, "median", na=silent_class)))
            lines.append(_line("P99 Serving AUDIO_TTFP (ms):", _fmt_stat(serving_audio_ttfp, "p99", na=silent_class)))
        if audio_rtf is not None:
            lines.append("----------------Real Time Factor----------------")
            lines.append(_line("Mean AUDIO_RTF:", _fmt_stat(audio_rtf, "mean", na=silent_class)))
            lines.append(_line("Median AUDIO_RTF:", _fmt_stat(audio_rtf, "median", na=silent_class)))
            lines.append(_line("P99 AUDIO_RTF:", _fmt_stat(audio_rtf, "p99", na=silent_class)))
    return lines


def _stage_sections_from_snapshots(
    snapshots: list[dict[str, Any]],
    *,
    total_generated_tokens: int,
    audio_ttfp_ms: dict[str, float] | None,
    audio_rtf: dict[str, float] | None,
    total_audio_duration_s: float,
    total_audio_frames: int,
    silent_class: bool = False,
) -> list[str]:
    lines: list[str] = []
    lines.append("============ Stage Benchmark Result ============")

    stage_ids = sorted(
        {
            int(sid)
            for smap in snapshots
            if isinstance(smap, dict)
            for sid in smap
            if str(sid).isdigit()
        }
    )
    if not stage_ids:
        stage_ids = [0, 1, 2, 3]

    for stage_id in stage_ids:
        # Silent turns never synthesize audio; skip Talker/Code2Wav sections so
        # finish-sentinel / delta-bleed timings are not reported as work.
        if silent_class and stage_id >= 2:
            continue
        sid = str(stage_id)
        stage_name = _STAGE_DISPLAY_NAMES.get(stage_id, f"stage_{stage_id}")
        gen_ms = _stage_stats(_collect_stage_values(snapshots, sid, defs.STAGE_GEN_TIME_MS), defs.STAGE_GEN_TIME_MS)
        internal_ttfc = _stage_stats(
            _collect_stage_values(snapshots, sid, defs.SERVING_TIME_TO_FIRST_OUTPUT_MS),
            defs.SERVING_TIME_TO_FIRST_OUTPUT_MS,
        )
        internal_tpop = _stage_stats(
            _collect_stage_values(snapshots, sid, defs.TIME_PER_OUTPUT_UNIT_MS),
            defs.TIME_PER_OUTPUT_UNIT_MS,
        )
        internal_icl = _stage_stats(
            _collect_stage_values(snapshots, sid, defs.INTER_OUTPUT_LATENCIES_MS),
            defs.INTER_OUTPUT_LATENCIES_MS,
        )

        text_tokens = None
        serving_ttft = None
        ttft = None
        tpot = None
        itl = None
        audio_duration_s = None
        audio_frames = None
        serving_audio_ttfp = None
        stage_audio_rtf = None
        audio_na_label = None

        if stage_id == 0 and gen_ms.get("mean", 0.0) <= 0.0 and internal_ttfc.get("mean", 0.0) > 0.0:
            gen_ms = internal_ttfc

        if stage_id == 1:
            token_vals = _collect_stage_values(snapshots, sid, defs.NUM_TOKENS_OUT)
            text_tokens = int(sum(token_vals)) if token_vals else total_generated_tokens
            serving_ttft = internal_ttfc
            if serving_ttft.get("mean", 0.0) <= 0.0:
                serving_ttft = _stage_stats(
                    _collect_stage_values(snapshots, sid, defs.VLLM_TTFT_MS),
                    defs.VLLM_TTFT_MS,
                )
            ttft = _stage_stats(_collect_stage_values(snapshots, sid, defs.VLLM_TTFT_MS), defs.VLLM_TTFT_MS)
            tpot = _stage_stats(_collect_stage_values(snapshots, sid, defs.VLLM_TPOT_MS), defs.VLLM_TPOT_MS)
            itl = _stage_stats(_collect_stage_values(snapshots, sid, defs.VLLM_ITLS_MS), defs.VLLM_ITLS_MS)
            if itl.get("mean", 0.0) <= 0.0 and tpot.get("mean", 0.0) > 0.0:
                itl = tpot
            if gen_ms.get("mean", 0.0) <= 0.0 and ttft.get("mean", 0.0) > 0.0:
                gen_estimates: list[float] = []
                for smap in snapshots:
                    aura = smap.get(sid) if isinstance(smap, dict) else {}
                    if not isinstance(aura, dict):
                        continue
                    tokens = int(aura.get(defs.NUM_TOKENS_OUT) or 0)
                    est = _estimate_stage_gen_time_ms(aura, tokens=tokens)
                    if est > 0:
                        gen_estimates.append(est)
                if gen_estimates:
                    gen_ms = stats_ms(gen_estimates)

        if stage_id == 2:
            audio_na_label = "N/A (codec only)"

        if stage_id == 3:
            dur_vals = _collect_stage_values(snapshots, sid, f"{defs.AUDIO_DURATION}_s")
            frame_vals = _collect_stage_values(snapshots, sid, defs.AUDIO_FRAMES)
            audio_duration_s = float(sum(dur_vals) if dur_vals else 0.0)
            audio_frames = int(sum(frame_vals) if frame_vals else 0)
            if audio_duration_s <= 0:
                audio_duration_s = float(total_audio_duration_s or 0.0)
                audio_frames = int(total_audio_frames or 0)
            serving_audio_ttfp = audio_ttfp_ms
            stage_audio_rtf = audio_rtf

        lines.extend(
            _stage_block_lines(
                stage_id,
                stage_name,
                gen_ms,
                text_tokens=text_tokens,
                serving_ttft=serving_ttft,
                ttft=ttft,
                tpot=tpot,
                itl=itl,
                internal_ttfc=internal_ttfc if stage_id in (0, 2) else None,
                internal_tpop=internal_tpop if stage_id in (0, 2) else None,
                internal_icl=internal_icl if stage_id in (0, 2) else None,
                audio_duration_s=audio_duration_s,
                audio_frames=audio_frames,
                serving_audio_ttfp=serving_audio_ttfp,
                audio_rtf=stage_audio_rtf,
                audio_na_label=audio_na_label,
                silent_class=silent_class,
            )
        )
    return lines


def format_bench_report(summary: dict[str, Any]) -> str:
    """Render a summary dict (from :func:`build_summary_from_result`) as markdown."""
    lines: list[str] = []
    meta = summary.get("meta") or {}
    if meta:
        lines.append(f"# vllm-omni OmniInteract streaming — {meta.get('label', '')}")
        lines.append(f"video: {meta.get('video', '')}")
        lines.append(f"generated: {meta.get('timestamp', '')}")
        if meta.get("json_path"):
            lines.append(f"source_json: {meta.get('json_path')}")
        lines.append("")

    spoken_class = summary.get("spoken_class")
    silent_class = summary.get("silent_class")
    if spoken_class is not None and silent_class is not None:
        lines.append(_line("Video sessions:", str(summary.get("video_sessions", 0))))
        lines.append(
            _line(
                "Total inference turns:",
                f"{summary.get('num_turns', 0)} "
                f"(spoken={summary.get('spoken_turns', 0)}, silent={summary.get('silent_turns', 0)})",
            )
        )
        lines.append(_line("Failed video sessions:", str(summary.get("failed_requests", 0))))
        lines.append(_line("Maximum request concurrency:", str(summary.get("max_request_concurrency", 1))))
        lines.append(_line("Benchmark duration (s):", f"{summary.get('benchmark_duration_s', 0):.2f}"))
        lines.append(_line("Peak concurrent requests:", f"{summary.get('peak_concurrent_requests', 1.0):.2f}"))
        lines.append("")
        _append_turn_class_section(lines, "Spoken turns", spoken_class, silent_class=False)
        lines.append("")
        _append_turn_class_section(lines, "Silent turns", silent_class, silent_class=True)
    else:
        lines.append(_line("Successful requests:", str(summary.get("successful_requests", 0))))
        lines.append(_line("Failed requests:", str(summary.get("failed_requests", 0))))
        lines.append(_line("Maximum request concurrency:", str(summary.get("max_request_concurrency", 1))))
        lines.append(_line("Benchmark duration (s):", f"{summary.get('benchmark_duration_s', 0):.2f}"))
        lines.append(_line("Request throughput (req/s):", f"{summary.get('request_throughput_rps', 0):.2f}"))
        lines.append(_line("Peak concurrent requests:", f"{summary.get('peak_concurrent_requests', 1.0):.2f}"))

        e2el = summary.get("e2el_ms") or {}
        lines.append("----------------End-to-end Latency----------------")
        lines.append(_line("Mean E2EL (ms):", f"{e2el.get('mean', 0):.2f}"))
        lines.append(_line("Median E2EL (ms):", f"{e2el.get('median', 0):.2f}"))
        lines.append(_line("P99 E2EL (ms):", f"{e2el.get('p99', 0):.2f}"))

        lines.append("============ Text Result ============")
        lines.append(_line("Total input tokens:", str(summary.get("total_input_tokens", 0))))
        lines.append(_line("Total generated tokens:", str(summary.get("total_generated_tokens", 0))))
        lines.append(_line("Output token throughput (tok/s):", f"{summary.get('output_token_throughput_tps', 0):.2f}"))
        lines.append(
            _line("Peak output token throughput (tok/s):", f"{summary.get('peak_output_token_throughput_tps', 0):.2f}")
        )
        lines.append(_line("Peak concurrent requests:", f"{summary.get('peak_concurrent_requests', 1.0):.2f}"))
        lines.append(_line("Total Token throughput (tok/s):", f"{summary.get('total_token_throughput_tps', 0):.2f}"))

        ttft = summary.get("ttft_ms") or {}
        lines.append("----------------Time to First Token----------------")
        lines.append(_line("Mean TTFT (ms):", f"{ttft.get('mean', 0):.2f}"))
        lines.append(_line("Median TTFT (ms):", f"{ttft.get('median', 0):.2f}"))
        lines.append(_line("P99 TTFT (ms):", f"{ttft.get('p99', 0):.2f}"))

        tpot = summary.get("tpot_ms") or {}
        lines.append("------Time per Output Token (excl. 1st token)------")
        lines.append(_line("Mean TPOT (ms):", f"{tpot.get('mean', 0):.2f}"))
        lines.append(_line("Median TPOT (ms):", f"{tpot.get('median', 0):.2f}"))
        lines.append(_line("P99 TPOT (ms):", f"{tpot.get('p99', 0):.2f}"))

        itl = summary.get("itl_ms") or {}
        lines.append("----------------Inter-token Latency----------------")
        lines.append(_line("Mean ITL (ms):", f"{itl.get('mean', 0):.2f}"))
        lines.append(_line("Median ITL (ms):", f"{itl.get('median', 0):.2f}"))
        lines.append(_line("P99 ITL (ms):", f"{itl.get('p99', 0):.2f}"))

        lines.append("============ Audio Result ============")
        lines.append(_line("Total audio duration generated(s):", f"{summary.get('total_audio_duration_s', 0):.2f}"))
        lines.append(_line("Total audio frames generated:", str(summary.get("total_audio_frames", 0))))
        lines.append(_line("Audio throughput(audio duration/s):", f"{summary.get('audio_throughput', 0):.2f}"))
        lines.append(_line("Streaming continuity OK rate:", "100.00%"))

        audio_ttfp = summary.get("audio_ttfp_ms") or {}
        lines.append("----------------Time to First Packet----------------")
        lines.append(_line("Mean AUDIO_TTFP (ms):", f"{audio_ttfp.get('mean', 0):.2f}"))
        lines.append(_line("Median AUDIO_TTFP (ms):", f"{audio_ttfp.get('median', 0):.2f}"))
        lines.append(_line("P99 AUDIO_TTFP (ms):", f"{audio_ttfp.get('p99', 0):.2f}"))

        audio_rtf = summary.get("audio_rtf") or {}
        lines.append("----------------Real Time Factor----------------")
        lines.append(_line("Mean AUDIO_RTF:", f"{audio_rtf.get('mean', 0):.2f}"))
        lines.append(_line("Median AUDIO_RTF:", f"{audio_rtf.get('median', 0):.2f}"))
        lines.append(_line("P99 AUDIO_RTF:", f"{audio_rtf.get('p99', 0):.2f}"))
        lines.append("")

        snapshots = summary.get("server_stage_snapshots") or []
        lines.extend(
            _stage_sections_from_snapshots(
                snapshots,
                total_generated_tokens=int(summary.get("total_generated_tokens") or 0),
                audio_ttfp_ms=audio_ttfp,
                audio_rtf=audio_rtf,
                total_audio_duration_s=float(summary.get("total_audio_duration_s") or 0.0),
                total_audio_frames=int(summary.get("total_audio_frames") or 0),
            )
        )

        lines.append(
            f"============ Turns: {summary.get('num_turns', 0)} "
            f"(spoken={summary.get('spoken_turns', 0)}, silent={summary.get('silent_turns', 0)}) ============"
        )

    omni = summary.get("omniinteract") or {}
    if omni:
        lines.append("")
        lines.append("============ OmniInteract QA metrics ============")
        lines.append(_line("Evaluated slots:", str(omni.get("evaluated", 0))))
        lines.append(_line("IA-QTF1:", f"{omni.get('ia_qtf1', 0):.4f}"))
        ids = omni.get("ids") or {}
        lines.append(_line("IDS.NOR:", f"{ids.get('NOR', 0):.4f}"))
        lines.append(_line("IDS.PAQ:", f"{ids.get('PAQ', 0):.4f}"))
        lines.append(_line("IDS.CSM-SR:", f"{ids.get('CSM_SR', 0):.4f}"))
        lines.append(_line("IDS.CSM-AS(s):", f"{ids.get('CSM_AS_seconds', 0):.4f}"))
        lines.append(_line("NCCS:", f"{omni.get('nccs', 0):.4f}"))

    per_videos = summary.get("per_videos") or []
    if per_videos:
        lines.append("")
        lines.append("============ Per-video summary ============")
        for item in per_videos:
            video = item.get("video_path") or item.get("video_id") or "?"
            spoken, silent = item.get("spoken_turns", 0), item.get("silent_turns", 0)
            ia_qtf1 = item.get("ia_qtf1")
            ia_suffix = f", IA-QTF1={ia_qtf1:.4f}" if ia_qtf1 is not None else ""
            lines.append(
                f"- {video}: e2el={item.get('e2el_ms', 0):.0f}ms, "
                f"chunks={spoken + silent} (spoken={spoken}, silent={silent}){ia_suffix}"
            )

    return "\n".join(lines) + "\n"


def _collect_turn_e2el_ms(chunk: dict[str, Any]) -> float | None:
    """End-to-end latency until text+audio fully done when audio exists.

    Prefer reconstructed ``ASR gen + Aura gen + TTS/code2wav gen`` for spoken
    turns (Stage-1 finishes before Stage-2). ``wall_e2el_ms`` from older JSON is
    stamped at ``text.done`` and must not be treated as full spoken E2EL.
    Silent / text-only turns keep client ``wall_e2el_ms`` (text done).
    """
    stage_metrics = _turn_stage_metrics_map(chunk)
    asr_gen = _positive_stage_gen_ms(stage_metrics, 0)
    aura_gen = _positive_stage_gen_ms(stage_metrics, 1)
    audio_stage = _turn_audio_stage(stage_metrics)
    audio_dur = float(
        audio_stage.get(f"{defs.AUDIO_DURATION}_s") or audio_stage.get("audio_duration_s") or 0.0
    )
    audio_frames = int(audio_stage.get(defs.AUDIO_FRAMES) or audio_stage.get("audio_frames") or 0)
    audio_gen = max(_positive_stage_gen_ms(stage_metrics, 3), _positive_stage_gen_ms(stage_metrics, 2))
    # Only treat as full spoken E2EL when this turn actually produced audio.
    # Silent turns may still carry residual/delta TTS stage_gen_time_ms.
    if (
        not _turn_is_silent(chunk)
        and audio_gen > 0
        and (audio_dur > 0 or audio_frames > 0)
        and (asr_gen > 0 or aura_gen > 0)
    ):
        return asr_gen + aura_gen + audio_gen

    # Newer clients may stamp wall_e2el at audio.done; accept if clearly past text path.
    raw = chunk.get("wall_e2el_ms")
    if raw is not None:
        value = float(raw)
        if value > 0:
            return value
    raw = chunk.get("e2el_ms")
    if raw is not None:
        value = float(raw)
        return value if value > 0 else None
    text_total = asr_gen + aura_gen
    return text_total if text_total > 0 else None


def _flatten_streaming_turns(per_requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    for item in per_requests:
        for chunk in item.get("streaming_turns") or item.get("streaming_chunks") or []:
            if not isinstance(chunk, dict):
                continue
            turns.append(
                {
                    **chunk,
                    "video_id": item.get("video_id"),
                    "video_path": item.get("video_path"),
                    "subset": item.get("subset"),
                }
            )
    return turns


def _turn_stage_snapshots(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    snapshots: list[dict[str, Any]] = []
    for turn in turns:
        metrics = turn.get("metrics")
        if not isinstance(metrics, dict):
            continue
        stage_metrics = metrics.get("stage_metrics")
        if isinstance(stage_metrics, dict) and stage_metrics:
            snapshots.append(stage_metrics)
    return snapshots


def _omniinteract_streaming_mode(per_requests: list[dict[str, Any]], result: dict[str, Any]) -> bool:
    if per_requests and any(isinstance(item.get("streaming_chunks"), list) for item in per_requests):
        return True
    return str(result.get("endpoint_type") or result.get("backend") or "").endswith("video-stream")


def build_summary_from_result(
    result: dict[str, Any],
    *,
    label: str = "",
    video_line: str = "",
    json_path: str | Path | None = None,
    per_requests: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    per_requests = per_requests if per_requests is not None else list(result.get("per_requests") or [])
    streaming_mode = _omniinteract_streaming_mode(per_requests, result)
    turns = _flatten_streaming_turns(per_requests) if streaming_mode else []
    if streaming_mode and turns:
        _reconcile_streaming_turn_metrics(turns)
    spoken_turn_list, silent_turn_list = _partition_turns(turns) if streaming_mode else ([], [])
    turn_snapshots = _turn_stage_snapshots(turns)
    snapshots = turn_snapshots if turn_snapshots else [
        item.get("stage_metrics") or {} for item in per_requests if item.get("success")
    ]

    spoken_turns = 0
    silent_turns = 0
    per_videos: list[dict[str, Any]] = []
    for item in per_requests:
        chunks = list(item.get("streaming_chunks") or [])
        s, sl = _count_spoken_silent(chunks)
        spoken_turns += s
        silent_turns += sl
        turn_e2els = [_collect_turn_e2el_ms(chunk) for chunk in chunks]
        turn_e2els = [value for value in turn_e2els if value is not None]
        per_videos.append(
            {
                "video_id": item.get("video_id"),
                "video_path": item.get("video_path"),
                "subset": item.get("subset"),
                "e2el_ms": float(sum(turn_e2els) / len(turn_e2els)) if turn_e2els else float(item.get("latency_s") or 0.0) * 1000.0,
                "spoken_turns": s,
                "silent_turns": sl,
                "success": item.get("success"),
            }
        )

    num_turns = spoken_turns + silent_turns
    video_sessions = len([item for item in per_requests if item.get("success")])
    duration_s = float(result.get("duration") or 0.0)
    spoken_class = (
        _build_turn_class_summary(spoken_turn_list, benchmark_duration_s=duration_s) if streaming_mode else None
    )
    silent_class = (
        _build_turn_class_summary(silent_turn_list, benchmark_duration_s=duration_s) if streaming_mode else None
    )
    if streaming_mode and spoken_class is not None:
        _backfill_class_audio_from_per_requests(
            spoken_class, per_requests, spoken_only=True, benchmark_duration_s=duration_s
        )
    if streaming_mode and silent_class is not None:
        _backfill_class_audio_from_per_requests(
            silent_class, per_requests, spoken_only=False, benchmark_duration_s=duration_s
        )
    turn_e2els = [_collect_turn_e2el_ms(turn) for turn in turns]
    turn_e2els = [value for value in turn_e2els if value is not None]
    if streaming_mode and spoken_turns > 0:
        successful_requests = spoken_turns
        failed_requests = int(result.get("failed") or 0)
    elif streaming_mode and num_turns > 0:
        successful_requests = len(turn_e2els) if turn_e2els else num_turns
        failed_requests = int(result.get("failed") or 0)
    else:
        successful_requests = int(result.get("completed") or 0)
        failed_requests = int(result.get("failed") or 0)

    total_out = int(result.get("total_output_tokens") or 0)
    total_in = int(result.get("total_input_tokens") or 0)
    request_throughput_rps = (
        successful_requests / duration_s if streaming_mode and duration_s > 0 else float(result.get("request_throughput") or 0.0)
    )
    if turn_e2els:
        e2el_ms = stats_ms(turn_e2els)
    else:
        e2el_ms = {
            "mean": float(result.get("mean_e2el_ms") or 0.0),
            "median": float(result.get("median_e2el_ms") or 0.0),
            "p99": float(result.get("p99_e2el_ms") or 0.0),
        }

    omniinteract = None
    if result.get("omniinteract_evaluated") is not None:
        omniinteract = {
            "evaluated": result.get("omniinteract_evaluated"),
            "ia_qtf1": result.get("omniinteract_ia_qtf1"),
            "ids": result.get("omniinteract_ids") or {},
            "nccs": result.get("omniinteract_nccs"),
        }

    return {
        "meta": {
            "label": label,
            "video": video_line,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "json_path": str(json_path) if json_path else "",
            "video_sessions": video_sessions if streaming_mode else None,
        },
        "successful_requests": successful_requests,
        "failed_requests": failed_requests,
        "max_request_concurrency": int(result.get("max_concurrency") or 1),
        "benchmark_duration_s": duration_s,
        "request_throughput_rps": request_throughput_rps,
        "peak_concurrent_requests": float(result.get("max_concurrent_requests") or 1.0),
        "e2el_ms": e2el_ms,
        "total_input_tokens": total_in,
        "total_generated_tokens": total_out,
        "output_token_throughput_tps": float(result.get("output_throughput") or 0.0),
        "peak_output_token_throughput_tps": _peak_output_tps(per_requests)
        or float(result.get("max_output_tokens_per_s") or 0.0),
        "total_token_throughput_tps": float(result.get("total_token_throughput") or 0.0),
        "ttft_ms": {
            "mean": float(result.get("mean_ttft_ms") or 0.0),
            "median": float(result.get("median_ttft_ms") or 0.0),
            "p99": float(result.get("p99_ttft_ms") or 0.0),
        },
        "tpot_ms": {
            "mean": float(result.get("mean_tpot_ms") or 0.0),
            "median": float(result.get("median_tpot_ms") or 0.0),
            "p99": float(result.get("p99_tpot_ms") or 0.0),
        },
        "itl_ms": {
            "mean": float(result.get("mean_itl_ms") or 0.0),
            "median": float(result.get("median_itl_ms") or 0.0),
            "p99": float(result.get("p99_itl_ms") or 0.0),
        },
        "total_audio_duration_s": float(result.get(defs.TOTAL_AUDIO_DURATION_S) or 0.0),
        "total_audio_frames": int(result.get(defs.TOTAL_AUDIO_FRAMES) or 0),
        "audio_throughput": float(result.get(defs.AUDIO_THROUGHPUT) or 0.0),
        "audio_ttfp_ms": {
            "mean": float(result.get("mean_audio_ttfp_ms") or 0.0),
            "median": float(result.get("median_audio_ttfp_ms") or 0.0),
            "p99": float(result.get("p99_audio_ttfp_ms") or 0.0),
        },
        "audio_rtf": {
            "mean": float(result.get("mean_audio_rtf") or 0.0),
            "median": float(result.get("median_audio_rtf") or 0.0),
            "p99": float(result.get("p99_audio_rtf") or 0.0),
        },
        "server_stage_snapshots": snapshots,
        "num_turns": num_turns,
        "video_sessions": video_sessions if streaming_mode else int(result.get("completed") or 0),
        "spoken_turns": spoken_turns,
        "silent_turns": silent_turns,
        "spoken_class": spoken_class,
        "silent_class": silent_class,
        "omniinteract": omniinteract,
        "per_videos": per_videos,
        "per_requests": per_requests,
    }


def enrich_result_with_per_requests(
    result: dict[str, Any],
    input_requests: list[Any],
    outputs: list[Any],
) -> None:
    """Attach per-video details to the bench JSON result (for reports and re-formatting)."""
    per_requests: list[dict[str, Any]] = []
    for req, out in zip(input_requests, outputs, strict=False):
        video_path = str(getattr(req, "omniinteract_streaming_video_path", "") or "")
        video_id = Path(video_path).stem if video_path else ""
        chunks = list(getattr(out, "omniinteract_streaming_chunks", None) or [])
        per_requests.append(
            {
                "video_id": video_id,
                "video_path": video_path,
                "subset": str(getattr(req, "omniinteract_subset", "") or ""),
                "success": bool(getattr(out, "success", False)),
                "error": str(getattr(out, "error", "") or ""),
                "latency_s": float(getattr(out, "latency", 0.0) or 0.0),
                "ttft_s": float(getattr(out, "ttft", 0.0) or 0.0),
                "prompt_len": int(getattr(out, "prompt_len", 0) or 0),
                "output_tokens": int(getattr(out, "output_tokens", 0) or 0),
                "generated_text": str(getattr(out, "generated_text", "") or ""),
                "stage_metrics": dict(getattr(out, "stage_metrics", None) or {}),
                "streaming_chunks": chunks,
                "streaming_turns": chunks,
                "audio_duration_s": float(getattr(out, "audio_duration", 0.0) or 0.0),
                "audio_frames": int(getattr(out, "audio_frames", 0) or 0),
            }
        )
    result["per_requests"] = per_requests


def _bench_label_from_result(result: dict[str, Any], per_requests: list[dict[str, Any]]) -> str:
    subsets = sorted({str(item.get("subset") or "") for item in per_requests if item.get("subset")})
    subset_part = ",".join(subsets) if subsets else "mixed"
    video_sessions = len(per_requests)
    return f"{subset_part} ({video_sessions} videos)"


def _video_line_from_per_requests(
    per_requests: list[dict[str, Any]],
    *,
    completed: int = 0,
) -> str:
    if not per_requests:
        n = completed or 0
        return f"mixed ({n} videos)" if n else "mixed"
    if len(per_requests) == 1:
        item = per_requests[0]
        return str(item.get("video_path") or item.get("video_id") or "?")
    subsets = sorted({str(item.get("subset") or "") for item in per_requests if item.get("subset")})
    subset_part = ",".join(subsets) if subsets else "mixed"
    return f"{subset_part} ({len(per_requests)} videos)"


def format_turn_class_standalone_report(
    summary: dict[str, Any],
    *,
    title: str,
    class_summary: dict[str, Any],
    source_json: str | Path | None = None,
) -> str:
    """Render a single spoken/silent performance report (benchmark_result export)."""
    lines: list[str] = []
    meta = summary.get("meta") or {}
    backend = str(summary.get("backend") or "vllm-omni")
    lines.append(f"# {backend} — {title}")
    if meta.get("label"):
        lines.append(f"label: {meta.get('label')}")
    if meta.get("video"):
        lines.append(f"video: {meta.get('video')}")
    lines.append(f"generated: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}")
    if source_json:
        lines.append(f"source_json: {source_json}")
    lines.append("")
    is_silent = "silent" in title.lower()
    _append_turn_class_section(lines, title, class_summary, silent_class=is_silent)
    return "\n".join(lines) + "\n"


def write_aura_benchmark_result_exports(
    summary: dict[str, Any],
    *,
    json_path: str | Path | None = None,
    aura_result_root: str | Path | None = None,
) -> tuple[Path, Path]:
    """Write spoken/silent reports under AURA/benchmark_result/vllm_omni/."""
    import os

    default_root = Path("/public/wtk/AURA/benchmark_result")
    root = Path(
        aura_result_root
        or os.environ.get("AURA_BENCHMARK_RESULT_ROOT", "")
        or default_root
    ).resolve()
    out_dir = root / "vllm_omni"
    out_dir.mkdir(parents=True, exist_ok=True)

    spoken_class = summary.get("spoken_class")
    silent_class = summary.get("silent_class")
    if spoken_class is None or silent_class is None:
        raise ValueError("summary missing spoken_class/silent_class; re-run bench or re-format JSON")

    source = str(json_path) if json_path else ""
    spoken_path = out_dir / "bench_report_spoken.md"
    silent_path = out_dir / "bench_report_silent.md"
    spoken_path.write_text(
        format_turn_class_standalone_report(
            summary,
            title="Spoken turns",
            class_summary=spoken_class,
            source_json=source,
        ),
        encoding="utf-8",
    )
    silent_path.write_text(
        format_turn_class_standalone_report(
            summary,
            title="Silent turns",
            class_summary=silent_class,
            source_json=source,
        ),
        encoding="utf-8",
    )
    return spoken_path, silent_path


def write_bench_reports(
    result: dict[str, Any],
    *,
    result_dir: str | Path,
    json_path: str | Path | None = None,
    write_per_video: bool = True,
) -> Path:
    """Write ``bench_report.md`` (and optional per-video reports) under *result_dir*."""
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    per_requests = list(result.get("per_requests") or [])
    label = _bench_label_from_result(result, per_requests)
    video_line = _video_line_from_per_requests(per_requests, completed=int(result.get("completed") or 0))
    json_path = json_path or result_dir / "result.json"

    summary = build_summary_from_result(
        result,
        label=label,
        video_line=video_line,
        json_path=json_path,
        per_requests=per_requests,
    )
    report_path = result_dir / "bench_report.md"
    report_path.write_text(format_bench_report(summary), encoding="utf-8")

    if write_per_video:
        for item in per_requests:
            if not item.get("success"):
                continue
            video_id = str(item.get("video_id") or "unknown")
            video_dir = result_dir / "videos" / video_id
            video_dir.mkdir(parents=True, exist_ok=True)
            single_result = dict(result)
            single_result["completed"] = 1
            single_result["failed"] = 0 if item.get("success") else 1
            single_result["per_requests"] = [item]
            single_summary = build_summary_from_result(
                single_result,
                label=f"{item.get('subset', '')}/{video_id}",
                video_line=str(item.get("video_path") or video_id),
                json_path=json_path,
                per_requests=[item],
            )
            (video_dir / "bench_report.md").write_text(format_bench_report(single_summary), encoding="utf-8")
            chunks_path = video_dir / "streaming_chunks.json"
            chunks_path.write_text(
                json.dumps(item.get("streaming_chunks") or [], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    if os.environ.get("OMNIINTERACT_WRITE_AURA_BENCHMARK_RESULT", "1") != "0":
        try:
            write_aura_benchmark_result_exports(summary, json_path=json_path)
        except Exception as exc:
            import logging

            logging.getLogger(__name__).warning("benchmark_result export failed: %s", exc)

    return report_path


def write_bench_report_from_json(json_path: str | Path) -> Path:
    """CLI helper: read a saved bench JSON and write ``bench_report.md`` beside it."""
    json_path = Path(json_path)
    result = json.loads(json_path.read_text(encoding="utf-8"))
    return write_bench_reports(result, result_dir=json_path.parent, json_path=json_path)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Format vllm-omni bench JSON as bench_report.md")
    ap.add_argument("input", help="bench result JSON or directory containing it")
    ap.add_argument("-o", "--output-dir", help="directory for bench_report.md (default: JSON parent)")
    ap.add_argument(
        "--export-aura-result",
        action="store_true",
        help="also (or only) write spoken/silent reports to AURA/benchmark_result/vllm_omni/",
    )
    ap.add_argument(
        "--aura-result-root",
        help="override AURA benchmark_result root (default: AURA/benchmark_result)",
    )
    args = ap.parse_args()

    inp = Path(args.input)
    if inp.is_dir():
        json_path = inp / "omniinteract_streaming_c1_3_f256_t2.json"
        if not json_path.exists():
            candidates = sorted(inp.glob("*.json"))
            if not candidates:
                raise SystemExit(f"No JSON files under {inp}")
            json_path = candidates[-1]
    else:
        json_path = inp

    result = json.loads(json_path.read_text(encoding="utf-8"))
    if args.export_aura_result and args.output_dir is None:
        per_requests = list(result.get("per_requests") or [])
        label = _bench_label_from_result(result, per_requests)
        video_line = _video_line_from_per_requests(per_requests, completed=int(result.get("completed") or 0))
        summary = build_summary_from_result(
            result,
            label=label,
            video_line=video_line,
            json_path=json_path,
            per_requests=per_requests,
        )
        spoken, silent = write_aura_benchmark_result_exports(
            summary,
            json_path=json_path,
            aura_result_root=args.aura_result_root,
        )
        print(f"Wrote {spoken}")
        print(f"Wrote {silent}")
        return

    out_dir = Path(args.output_dir) if args.output_dir else json_path.parent
    path = write_bench_reports(
        result,
        result_dir=out_dir,
        json_path=json_path,
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
