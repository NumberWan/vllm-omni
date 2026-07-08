"""OmniInteract benchmark metrics (IA-QTF1 / IDS / NCCS) for vLLM bench."""

from __future__ import annotations

import math
import re
import string
from typing import Any

from vllm.benchmarks.lib.endpoint_request_func import RequestFuncOutput

from vllm_omni.benchmarks.data_modules.omniinteract_dataset import OmniInteractSampleRequest

_CJK_PUNCT = "，。！？；：（）【】《》“”‘’、"


def _normalize_text(text: str | None) -> str:
    t = (text or "").strip().lower()
    if not t:
        return ""
    table = str.maketrans("", "", string.punctuation + _CJK_PUNCT)
    t = t.translate(table)
    t = re.sub(r"\s+", "", t)
    return t


def _safe_ratio(num: int, den: int) -> float | None:
    return (num / den) if den else None


def _safe_div(num: float, den: float) -> float:
    return (num / den) if den else 0.0


def _safe_metric(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _f1(precision: float, recall: float) -> float:
    return _safe_div(2.0 * precision * recall, precision + recall)


def _metric_row(tp: float, fp: float, fn: float, num_slots: int) -> dict[str, Any]:
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    return {
        "num_slots": num_slots,
        "Global_TP": tp,
        "Global_FP": fp,
        "Global_FN": fn,
        "Precision": precision,
        "Recall": recall,
        "IA_QTF1": _f1(precision, recall),
    }


def _metric_sub(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    tp = float(a.get("Global_TP", 0.0)) - float(b.get("Global_TP", 0.0))
    fp = float(a.get("Global_FP", 0.0)) - float(b.get("Global_FP", 0.0))
    fn = float(a.get("Global_FN", 0.0)) - float(b.get("Global_FN", 0.0))
    slots = int(a.get("num_slots", 0) or 0) - int(b.get("num_slots", 0) or 0)
    return _metric_row(tp=max(0.0, tp), fp=max(0.0, fp), fn=max(0.0, fn), num_slots=max(0, slots))


def _calculate_decay(time_value: float, peak: float, end: float, alpha: float = 1.0, gamma: float = 1.0) -> float:
    if end <= peak:
        return 1.0 if time_value <= peak else 0.0
    if time_value <= peak:
        return 1.0
    if time_value >= end:
        return 0.0
    ratio = (time_value - peak) / (end - peak)
    return max(0.0, min(1.0, 1.0 - alpha * (ratio**gamma)))


def _chunk_text(chunks: list[dict[str, Any]]) -> str:
    return "".join(str(c.get("text", "") or "") for c in chunks).strip()


def _chunk_bounds(chunks: list[dict[str, Any]]) -> tuple[float | None, float | None]:
    if not chunks:
        return None, None
    ordered = sorted(chunks, key=lambda c: (float(c.get("start", 0.0)), float(c.get("end", 0.0))))
    return float(ordered[0].get("start", 0.0)), float(ordered[-1].get("end", 0.0))


def _normalize_streaming_output_chunks(out: RequestFuncOutput) -> list[dict[str, Any]]:
    raw = getattr(out, "omniinteract_streaming_chunks", None) or []
    chunks: list[dict[str, Any]] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        ts = item.get("timestamp")
        if not (isinstance(ts, list) and len(ts) == 2):
            continue
        start = _safe_metric(ts[0])
        end = _safe_metric(ts[1])
        if start is None or end is None:
            continue
        if end < start:
            start, end = end, start
        text = str(item.get("text", "") or "")
        chunks.append(
            {
                "chunk_id": item.get("chunk_id", idx),
                "start": float(start),
                "end": float(end),
                "text": text,
            }
        )
    chunks.sort(key=lambda c: (float(c["start"]), float(c["end"])))
    return chunks


def _streaming_match_quality(pred_raw: str, gold_raw: str) -> float:
    pred = _normalize_text(pred_raw)
    gold = _normalize_text(gold_raw)
    if not pred or not gold:
        return 0.0
    return 1.0 if pred == gold or pred in gold or gold in pred else 0.0


def _assign_chunks_to_slots(
    slots: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = [{**slot, "all_chunks": [], "early_chunks": [], "core_chunks": []} for slot in slots]
    unmatched: list[dict[str, Any]] = []
    for chunk in chunks:
        start = float(chunk.get("start", 0.0))
        candidates = [
            idx
            for idx, slot in enumerate(rows)
            if float(slot.get("start", 0.0)) <= start < float(slot.get("end", 0.0))
        ]
        if not candidates:
            unmatched.append(chunk)
            continue
        idx = max(candidates, key=lambda i: (float(rows[i].get("start", 0.0)), int(rows[i].get("slot_id", 0))))
        slot = rows[idx]
        boundary = float(slot.get("t_a", slot.get("start", 0.0)))
        tagged = {
            **chunk,
            "phase": "early" if start < boundary else "core",
            "spill": float(chunk.get("end", start)) > float(slot.get("end", boundary)),
        }
        slot["all_chunks"].append(tagged)
        slot["early_chunks" if start < boundary else "core_chunks"].append(tagged)
    return rows, unmatched


def _summarize_streaming_group(acc: dict[str, dict[str, float]]) -> dict[str, Any]:
    return {
        key: _metric_row(row["Global_TP"], row["Global_FP"], row["Global_FN"], int(row["num_slots"]))
        for key, row in sorted(acc.items())
    }


def _add_streaming_metric(
    acc: dict[str, dict[str, float]],
    key: str,
    tp: float,
    fp: float,
    fn: float,
) -> None:
    row = acc.setdefault(key or "unknown", {"num_slots": 0.0, "Global_TP": 0.0, "Global_FP": 0.0, "Global_FN": 0.0})
    row["num_slots"] += 1.0
    row["Global_TP"] += float(tp)
    row["Global_FP"] += float(fp)
    row["Global_FN"] += float(fn)


def _nested_pair_key(slot: dict[str, Any]) -> tuple[str, int]:
    return (str(slot.get("video", "") or ""), int(slot.get("nested_group_id") or slot.get("slot_id") or 0))


def _compute_streaming_omniinteract_metrics(
    input_requests: list[Any],
    outputs: list[RequestFuncOutput],
    *,
    include_per_item: bool,
) -> dict[str, Any]:
    global_tp = 0.0
    global_fp = 0.0
    global_fn = 0.0
    failed = 0
    evaluated_slots = 0
    unmatched_chunks = 0
    by_scene: dict[str, dict[str, float]] = {}
    by_qtype: dict[str, dict[str, float]] = {}
    nested_by_role: dict[str, dict[str, float]] = {}
    nested_pairs: dict[tuple[str, int], dict[str, float]] = {}
    nested_pair_times: dict[tuple[str, int], dict[str, float | None]] = {}
    interrupted_total = 0
    interrupted_no_output = 0
    interrupted_output_count = 0
    interrupted_quality_sum = 0.0
    interrupted_spill_positive = 0
    interrupted_spill_seconds = 0.0
    items: list[dict[str, Any]] = []

    for req, out in zip(input_requests, outputs, strict=True):
        assert isinstance(req, OmniInteractSampleRequest)
        slots = list(getattr(out, "omniinteract_streaming_slots", None) or req.omniinteract_streaming_slots or [])
        if not slots:
            continue
        if not out.success:
            failed += 1
            for slot in slots:
                if bool(slot.get("is_interrupted")):
                    interrupted_total += 1
                    interrupted_no_output += 1
                else:
                    global_fn += 1.0
                    evaluated_slots += 1
                    _add_streaming_metric(by_scene, str(slot.get("scene_type") or "unknown"), 0.0, 0.0, 1.0)
                    _add_streaming_metric(by_qtype, str(slot.get("question_type") or "unknown"), 0.0, 0.0, 1.0)
            continue

        chunks = _normalize_streaming_output_chunks(out)
        rows, unmatched = _assign_chunks_to_slots(slots, chunks)
        unmatched_chunks += len(unmatched)
        global_fp += float(len(unmatched))
        for slot in rows:
            evaluated_slots += 1
            scene = str(slot.get("scene_type") or "unknown")
            if scene.lower() == "1qna":
                scene = "1QnA"
            qtype = str(slot.get("question_type") or "unknown")
            is_interrupted = bool(slot.get("is_interrupted"))
            all_chunks = list(slot.get("all_chunks") or [])
            early_chunks = list(slot.get("early_chunks") or [])
            core_chunks = list(slot.get("core_chunks") or [])
            text_core = _chunk_text(core_chunks)
            text_all = _chunk_text(all_chunks)
            quality_core = _streaming_match_quality(text_core, str(slot.get("gt_answer", "") or ""))
            quality_all = _streaming_match_quality(text_all, str(slot.get("gt_answer", "") or ""))
            core_start, core_end = _chunk_bounds(core_chunks)
            early_start, _ = _chunk_bounds(early_chunks)
            t_a = float(slot.get("t_a", slot.get("start", 0.0)))
            end = float(slot.get("end", t_a))
            local_w_ack = 0.2
            if scene == "1QnA" and int(slot.get("step_index") or slot.get("slot_id") or 1) > 1:
                local_w_ack = 0.0
            local_w_core = 1.0 - local_w_ack
            tp_ack = 0.0
            if early_chunks and early_start is not None and not is_interrupted:
                tp_ack = _calculate_decay(early_start, float(slot.get("start", 0.0)), t_a) * local_w_ack
            tp_core = 0.0
            if core_chunks and core_start is not None and not is_interrupted and quality_core > 0.0:
                tp_core = quality_core * _calculate_decay(max(core_start, t_a), t_a, end) * local_w_core
            spill_seconds = 0.0
            if all_chunks:
                max_end = max(float(c.get("end", end)) for c in all_chunks)
                spill_seconds = max(0.0, max_end - end)
            fp = 1.0 if core_chunks and not is_interrupted and quality_core <= 0.0 else 0.0
            if spill_seconds > 0.0:
                fp += 1.0
            tp = max(0.0, min(1.0, tp_ack + tp_core))
            fn = 0.0 if is_interrupted else (0.0 if tp_core > 0.0 else 1.0)

            if is_interrupted:
                interrupted_total += 1
                if not all_chunks:
                    interrupted_no_output += 1
                else:
                    interrupted_output_count += 1
                    interrupted_quality_sum += quality_all
                    if spill_seconds > 0.0:
                        interrupted_spill_positive += 1
                    interrupted_spill_seconds += spill_seconds
            else:
                global_tp += tp
                global_fp += fp
                global_fn += fn
                _add_streaming_metric(by_scene, scene, tp, fp, fn)
                _add_streaming_metric(by_qtype, qtype, tp, fp, fn)
                role = str(slot.get("nested_role") or "").strip().lower()
                if scene == "nested" and role in {"inner", "outer"}:
                    _add_streaming_metric(nested_by_role, role, tp, fp, fn)
                    pair_key = _nested_pair_key({**slot, "video": req.omniinteract_video})
                    nested_pairs.setdefault(pair_key, {})[role] = tp_core
                    nested_pair_times.setdefault(pair_key, {})[role] = core_start

            if include_per_item:
                items.append(
                    {
                        "request_id": req.request_id,
                        "video": req.omniinteract_video,
                        "slot_id": slot.get("slot_id"),
                        "scene_type": scene,
                        "question_type": qtype,
                        "nested_group_id": slot.get("nested_group_id"),
                        "nested_role": slot.get("nested_role"),
                        "is_interrupted": is_interrupted,
                        "gt": slot.get("gt_answer", ""),
                        "predicted_core": text_core,
                        "predicted_all": text_all,
                        "Score_core": tp_core,
                        "TP_n": tp,
                        "FP_delta": fp,
                        "FN_delta": fn,
                        "spill_seconds": spill_seconds,
                        "num_chunks": len(all_chunks),
                    }
                )

    by_scene_metric = _summarize_streaming_group(by_scene)
    by_qtype_metric = _summarize_streaming_group(by_qtype)
    nested_by_role_out = _summarize_streaming_group(nested_by_role)
    realtime_exclusive = _metric_sub(by_qtype_metric.get("realtime", {}), nested_by_role_out.get("inner", {}))
    proactive_exclusive = _metric_sub(by_qtype_metric.get("proactive", {}), nested_by_role_out.get("outer", {}))
    nested_metric = by_scene_metric.get("nested", _metric_row(0.0, 0.0, 0.0, 0))
    one_qna_metric = by_scene_metric.get("1QnA", _metric_row(0.0, 0.0, 0.0, 0))
    one_q1a_tp = (
        float(realtime_exclusive.get("Global_TP", 0.0))
        + float(proactive_exclusive.get("Global_TP", 0.0))
        + float(nested_metric.get("Global_TP", 0.0))
    )
    one_q1a_fp = (
        float(realtime_exclusive.get("Global_FP", 0.0))
        + float(proactive_exclusive.get("Global_FP", 0.0))
        + float(nested_metric.get("Global_FP", 0.0))
    )
    one_q1a_fn = (
        float(realtime_exclusive.get("Global_FN", 0.0))
        + float(proactive_exclusive.get("Global_FN", 0.0))
        + float(nested_metric.get("Global_FN", 0.0))
    )
    one_q1a_slots = (
        int(realtime_exclusive.get("num_slots", 0) or 0)
        + int(proactive_exclusive.get("num_slots", 0) or 0)
        + int(nested_metric.get("num_slots", 0) or 0)
    )
    all_global = _metric_row(global_tp, global_fp, global_fn, evaluated_slots)

    num_pairs = 0
    success_pairs = 0
    missing_outer = 0
    missing_inner = 0
    order_error = 0
    ira_sum = 0.0
    for pair_key, pair in nested_pairs.items():
        num_pairs += 1
        q1 = float(pair.get("outer", 0.0))
        q2 = float(pair.get("inner", 0.0))
        if q1 <= 0:
            missing_outer += 1
        if q2 <= 0:
            missing_inner += 1
        q1_time = nested_pair_times.get(pair_key, {}).get("outer")
        q2_time = nested_pair_times.get(pair_key, {}).get("inner")
        order_ok = True
        if q1 > 0 and q2 > 0 and q1_time is not None and q2_time is not None:
            order_ok = float(q2_time) <= float(q1_time)
        if q1 > 0 and q2 > 0 and not order_ok:
            order_error += 1
        if q1 > 0 and q2 > 0 and order_ok:
            success_pairs += 1
            ira_sum += math.sqrt(q1 * q2)
    nccs = _safe_div(ira_sum, num_pairs)
    nor = _safe_div(interrupted_no_output, interrupted_total)
    paq = _safe_div(interrupted_quality_sum, interrupted_output_count)
    csm_sr = _safe_div(interrupted_spill_positive, interrupted_output_count)
    csm_as = _safe_div(interrupted_spill_seconds, interrupted_output_count)

    metrics: dict[str, Any] = {
        "omniinteract_evaluated": evaluated_slots,
        "omniinteract_request_failed": failed,
        "omniinteract_unmatched_chunks": unmatched_chunks,
        "omniinteract_paper_metrics": {
            "exp_f1": {
                "realtime": realtime_exclusive,
                "proactive": proactive_exclusive,
                "nested": nested_metric,
                "one_q1a_global": _metric_row(one_q1a_tp, one_q1a_fp, one_q1a_fn, one_q1a_slots),
                "one_qna": one_qna_metric,
                "all_global": all_global,
            },
            "exp_interruption": {
                "NOR": nor,
                "PAQ": paq,
                "CSM_SR": csm_sr,
                "CSM_AS_seconds": csm_as,
                "interrupted_slot_count": interrupted_total,
                "interrupted_with_output_count": interrupted_output_count,
            },
            "exp_nested": {
                "NCCS": nccs,
                "inner_IA_QTF1": float(nested_by_role_out.get("inner", {}).get("IA_QTF1", 0.0)),
                "outer_IA_QTF1": float(nested_by_role_out.get("outer", {}).get("IA_QTF1", 0.0)),
                "missed_outer": missing_outer,
                "missed_inner": missing_inner,
                "order_error": order_error,
                "num_pairs": num_pairs,
                "success_pairs": success_pairs,
            },
        },
        "omniinteract_ia_qtf1": float(all_global["IA_QTF1"]),
        "omniinteract_ids": {
            "NOR": nor,
            "PAQ": paq,
            "CSM_SR": csm_sr,
            "CSM_AS_seconds": csm_as,
        },
        "omniinteract_nccs": nccs,
        "omniinteract_metric_note": (
            "Streaming IA-QTF1/IDS/NCCS are computed from annotation slots and "
            "timestamped response chunks. Text quality uses the benchmark's local "
            "soft string match; plug in the official LLM judge output for semantic scoring."
        ),
    }
    if include_per_item:
        metrics["omniinteract_eval_items"] = items
    return metrics


def compute_omniinteract_metrics(
    input_requests: list[Any],
    outputs: list[RequestFuncOutput],
    *,
    include_per_item: bool = False,
) -> dict[str, Any] | None:
    if not input_requests or len(input_requests) != len(outputs):
        return None
    if not all(isinstance(r, OmniInteractSampleRequest) for r in input_requests):
        return None
    if any(getattr(r, "omniinteract_streaming_slots", None) for r in input_requests):
        return _compute_streaming_omniinteract_metrics(
            input_requests,
            outputs,
            include_per_item=include_per_item,
        )

    exact = 0
    soft = 0
    evaluated = 0
    failed = 0
    per_subset: dict[str, dict[str, int]] = {}
    per_qtype: dict[str, dict[str, int]] = {}
    global_tp = 0.0
    global_fp = 0.0
    global_fn = 0.0
    by_scene: dict[str, dict[str, float]] = {}
    by_qtype_metric: dict[str, dict[str, float]] = {}
    nested_by_role: dict[str, dict[str, float]] = {}
    nested_pairs: dict[tuple[str, int], dict[str, float]] = {}
    interrupted_total = 0
    interrupted_no_output = 0
    interrupted_output_quality_sum = 0.0
    interrupted_output_count = 0
    interrupted_spill_timed_count = 0
    interrupted_spill_positive_count = 0
    interrupted_spill_seconds = 0.0
    items: list[dict[str, Any]] = []

    def _accumulate(store: dict[str, dict[str, float]], key: str, tp: float, fp: float, fn: float) -> None:
        row = store.setdefault(key, {"num_slots": 0.0, "Global_TP": 0.0, "Global_FP": 0.0, "Global_FN": 0.0})
        row["num_slots"] += 1.0
        row["Global_TP"] += tp
        row["Global_FP"] += fp
        row["Global_FN"] += fn

    for req, out in zip(input_requests, outputs, strict=True):
        assert isinstance(req, OmniInteractSampleRequest)
        subset = (req.omniinteract_subset or "unknown").strip() or "unknown"
        qtype = (req.omniinteract_question_type or "unknown").strip() or "unknown"
        scene = (req.omniinteract_scene_type or "").strip().lower() or ("1qna" if subset == "1qna" else "multi_turn")
        scene = "1qna" if scene in {"1qna", "1qna_long"} else scene
        per_subset.setdefault(subset, {"exact": 0, "soft": 0, "total": 0})
        per_qtype.setdefault(qtype, {"exact": 0, "soft": 0, "total": 0})

        if not out.success:
            failed += 1
            if req.omniinteract_is_interrupted:
                interrupted_total += 1
                interrupted_no_output += 1
            elif qtype:
                global_fn += 1.0
                _accumulate(by_scene, scene, 0.0, 0.0, 1.0)
                _accumulate(by_qtype_metric, qtype, 0.0, 0.0, 1.0)
            if include_per_item:
                items.append(
                    {
                        "request_id": req.request_id,
                        "subset": subset,
                        "question_type": qtype,
                        "video": req.omniinteract_video,
                        "scene_type": scene,
                        "nested_group_id": req.omniinteract_nested_group_id,
                        "nested_role": req.omniinteract_nested_role,
                        "error": (out.error or "")[:500],
                        "correct_exact": False,
                        "correct_soft": False,
                        "quality_score": 0.0,
                        "has_output": False,
                    }
                )
            continue

        pred_raw = out.generated_text or ""
        gold_raw = req.omniinteract_gold_answer or ""
        pred = _normalize_text(pred_raw)
        gold = _normalize_text(gold_raw)
        if not gold:
            continue

        evaluated += 1
        per_subset[subset]["total"] += 1
        per_qtype[qtype]["total"] += 1
        is_exact = pred == gold
        is_soft = bool(pred and gold and (pred in gold or gold in pred))
        quality_score = 1.0 if is_soft else 0.0
        has_output = bool(pred_raw.strip())
        if is_exact:
            exact += 1
            per_subset[subset]["exact"] += 1
            per_qtype[qtype]["exact"] += 1
        if is_soft:
            soft += 1
            per_subset[subset]["soft"] += 1
            per_qtype[qtype]["soft"] += 1

        if req.omniinteract_is_interrupted:
            interrupted_total += 1
            if not has_output:
                interrupted_no_output += 1
            else:
                interrupted_output_count += 1
                interrupted_output_quality_sum += quality_score
                spill_seconds = _safe_metric(getattr(out, "omniinteract_spill_seconds", None))
                if spill_seconds is None:
                    spill_seconds = _safe_metric(getattr(out, "spill_seconds", None))
                if spill_seconds is not None:
                    interrupted_spill_timed_count += 1
                    interrupted_spill_seconds += max(0.0, spill_seconds)
                    if spill_seconds > 0:
                        interrupted_spill_positive_count += 1
        else:
            fp = 1.0 if quality_score <= 0 else 0.0
            fn = 1.0 if quality_score <= 0 else 0.0
            global_tp += quality_score
            global_fp += fp
            global_fn += fn
            _accumulate(by_scene, scene, quality_score, fp, fn)
            _accumulate(by_qtype_metric, qtype, quality_score, fp, fn)
            role = (req.omniinteract_nested_role or "").strip().lower()
            if scene == "nested" and role in {"outer", "inner"}:
                _accumulate(nested_by_role, role, quality_score, fp, fn)
                if req.omniinteract_nested_group_id is not None:
                    pair_key = (req.omniinteract_video or "", int(req.omniinteract_nested_group_id))
                    nested_pairs.setdefault(pair_key, {})[role] = quality_score

        if include_per_item:
            items.append(
                {
                    "request_id": req.request_id,
                    "subset": subset,
                    "scene_type": scene,
                    "question_type": qtype,
                    "video": req.omniinteract_video,
                    "nested_group_id": req.omniinteract_nested_group_id,
                    "nested_role": req.omniinteract_nested_role,
                    "question_time": req.omniinteract_question_time,
                    "answer_time": req.omniinteract_answer_time,
                    "gold": gold_raw,
                    "predicted": pred_raw,
                    "gold_normalized": gold,
                    "predicted_normalized": pred,
                    "correct_exact": is_exact,
                    "correct_soft": is_soft,
                    "quality_score": quality_score,
                    "has_output": has_output,
                }
            )

    by_scene_metric = {
        name: _metric_row(v["Global_TP"], v["Global_FP"], v["Global_FN"], int(v["num_slots"]))
        for name, v in by_scene.items()
    }
    by_qtype_metric_out = {
        name: _metric_row(v["Global_TP"], v["Global_FP"], v["Global_FN"], int(v["num_slots"]))
        for name, v in by_qtype_metric.items()
    }
    nested_by_role_out = {
        name: _metric_row(v["Global_TP"], v["Global_FP"], v["Global_FN"], int(v["num_slots"]))
        for name, v in nested_by_role.items()
    }

    realtime_exclusive = _metric_sub(by_qtype_metric_out.get("realtime", {}), nested_by_role_out.get("inner", {}))
    proactive_exclusive = _metric_sub(by_qtype_metric_out.get("proactive", {}), nested_by_role_out.get("outer", {}))
    nested_metric = by_scene_metric.get("nested", _metric_row(0.0, 0.0, 0.0, 0))
    one_qna_metric = by_scene_metric.get("1qna", _metric_row(0.0, 0.0, 0.0, 0))
    one_q1a_tp = (
        float(realtime_exclusive.get("Global_TP", 0.0))
        + float(proactive_exclusive.get("Global_TP", 0.0))
        + float(nested_metric.get("Global_TP", 0.0))
    )
    one_q1a_fp = (
        float(realtime_exclusive.get("Global_FP", 0.0))
        + float(proactive_exclusive.get("Global_FP", 0.0))
        + float(nested_metric.get("Global_FP", 0.0))
    )
    one_q1a_fn = (
        float(realtime_exclusive.get("Global_FN", 0.0))
        + float(proactive_exclusive.get("Global_FN", 0.0))
        + float(nested_metric.get("Global_FN", 0.0))
    )
    one_q1a_slots = (
        int(realtime_exclusive.get("num_slots", 0) or 0)
        + int(proactive_exclusive.get("num_slots", 0) or 0)
        + int(nested_metric.get("num_slots", 0) or 0)
    )
    all_global = _metric_row(global_tp, global_fp, global_fn, int(one_q1a_slots + one_qna_metric.get("num_slots", 0)))

    num_pairs = 0
    success_pairs = 0
    missing_outer = 0
    ira_sum = 0.0
    for pair in nested_pairs.values():
        num_pairs += 1
        q1 = float(pair.get("outer", 0.0))
        q2 = float(pair.get("inner", 0.0))
        if q1 <= 0:
            missing_outer += 1
        if q1 > 0 and q2 > 0:
            success_pairs += 1
            ira_sum += math.sqrt(q1 * q2)
    nccs = _safe_div(ira_sum, num_pairs)

    nor = _safe_div(interrupted_no_output, interrupted_total)
    paq = _safe_div(interrupted_output_quality_sum, interrupted_output_count)
    csm_sr = (
        _safe_div(interrupted_spill_positive_count, interrupted_spill_timed_count)
        if interrupted_spill_timed_count
        else None
    )
    csm_as = (
        _safe_div(interrupted_spill_seconds, interrupted_spill_timed_count) if interrupted_spill_timed_count else None
    )

    out: dict[str, Any] = {
        "omniinteract_evaluated": evaluated,
        "omniinteract_request_failed": failed,
        "omniinteract_exact_match": _safe_ratio(exact, evaluated),
        "omniinteract_soft_match": _safe_ratio(soft, evaluated),
        "omniinteract_exact_count": exact,
        "omniinteract_soft_count": soft,
        "omniinteract_per_subset": per_subset,
        "omniinteract_per_question_type": per_qtype,
        "omniinteract_paper_metrics": {
            "exp_f1": {
                "realtime": realtime_exclusive,
                "proactive": proactive_exclusive,
                "nested": nested_metric,
                "one_q1a_global": _metric_row(one_q1a_tp, one_q1a_fp, one_q1a_fn, one_q1a_slots),
                "one_qna": one_qna_metric,
                "all_global": all_global,
            },
            "exp_interruption": {
                "NOR": nor,
                "PAQ": paq,
                "CSM_SR": csm_sr,
                "CSM_AS_seconds": csm_as,
                "interrupted_slot_count": interrupted_total,
                "interrupted_with_output_count": interrupted_output_count,
                "interrupted_with_spill_timing_count": interrupted_spill_timed_count,
            },
            "exp_nested": {
                "NCCS": nccs,
                "inner_IA_QTF1": float(nested_by_role_out.get("inner", {}).get("IA_QTF1", 0.0)),
                "outer_IA_QTF1": float(nested_by_role_out.get("outer", {}).get("IA_QTF1", 0.0)),
                "missed_outer": missing_outer,
                "num_pairs": num_pairs,
                "success_pairs": success_pairs,
            },
        },
        "omniinteract_ia_qtf1": float(all_global["IA_QTF1"]),
        "omniinteract_ids": {
            "NOR": nor,
            "PAQ": paq,
            "CSM_SR": csm_sr,
            "CSM_AS_seconds": csm_as,
        },
        "omniinteract_nccs": nccs,
        "omniinteract_metric_note": (
            "IA-QTF1/IDS/NCCS are estimated from per-request benchmark outputs. "
            "IDS CSM_SR/CSM_AS require continuous-turn spill timing; they are N/A "
            "unless outputs provide omniinteract_spill_seconds/spill_seconds."
        ),
    }

    out["omniinteract_per_subset_exact"] = {
        name: _safe_ratio(vals["exact"], vals["total"]) for name, vals in per_subset.items()
    }
    out["omniinteract_per_question_type_exact"] = {
        name: _safe_ratio(vals["exact"], vals["total"]) for name, vals in per_qtype.items()
    }
    if include_per_item:
        out["omniinteract_eval_items"] = items
    return out


def print_omniinteract_summary(metrics: dict[str, Any]) -> None:
    if (
        int(metrics.get("omniinteract_evaluated", 0) or 0) == 0
        and int(metrics.get("omniinteract_request_failed", 0) or 0) == 0
    ):
        return
    print("{s:{c}^{n}}".format(s=" OmniInteract QA metrics ", n=50, c="="))
    print("{:<40} {:<10}".format("Evaluated:", metrics.get("omniinteract_evaluated", 0)))
    if metrics.get("omniinteract_ia_qtf1") is not None:
        print("{:<40} {:<10.4f}".format("IA-QTF1:", float(metrics.get("omniinteract_ia_qtf1"))))
    ids = metrics.get("omniinteract_ids") or {}
    if ids:
        print("{:<40} {:<10.4f}".format("IDS.NOR:", float(ids.get("NOR", 0.0))))
        print("{:<40} {:<10.4f}".format("IDS.PAQ:", float(ids.get("PAQ", 0.0))))
        csm_sr = ids.get("CSM_SR")
        csm_as = ids.get("CSM_AS_seconds")
        if csm_sr is None:
            print("{:<40} {:<10}".format("IDS.CSM-SR:", "N/A"))
        else:
            print("{:<40} {:<10.4f}".format("IDS.CSM-SR:", float(csm_sr)))
        if csm_as is None:
            print("{:<40} {:<10}".format("IDS.CSM-AS(s):", "N/A"))
        else:
            print("{:<40} {:<10.4f}".format("IDS.CSM-AS(s):", float(csm_as)))
    if metrics.get("omniinteract_nccs") is not None:
        print("{:<40} {:<10.4f}".format("NCCS:", float(metrics.get("omniinteract_nccs"))))
    print("=" * 50)
