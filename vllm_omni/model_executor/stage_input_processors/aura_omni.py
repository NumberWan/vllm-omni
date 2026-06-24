# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage processors for the AURA Omni pipeline."""

from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from vllm.logger import init_logger
from vllm.tokenizers import cached_tokenizer_from_config

from vllm_omni.engine.serialization import deserialize_additional_information
from vllm_omni.entrypoints.openai.aura_session_history import SessionHistory, is_effectively_silent
from vllm_omni.inputs.data import OmniTokensPrompt
from vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder import (
    PRECOMPUTED_TEXT_IDS_KEY,
)

logger = init_logger(__name__)

DEFAULT_AURA_SYSTEM_PROMPT = (
    "You are receiving a live video stream where the final frame is the present moment. "
    "Respond only when a response is needed based on the user's message or the visual context. "
    "Otherwise, output '<|silent|>' to signify silence. Respond in Chinese."
)

SILENT_TEXT = "<|silent|>"
QWEN_IM_START_ID = 151644
QWEN_IM_END_ID = 151645
QWEN_ASSISTANT_ID = 77091

logger = init_logger(__name__)

_TURN_TRANSCRIPTS_BY_REQUEST: dict[str, str] = {}

QWEN_NEWLINE_ID = 198
QWEN_ASSISTANT_PREFIX_IDS = [QWEN_IM_START_ID, QWEN_ASSISTANT_ID, QWEN_NEWLINE_ID]
QWEN_ASSISTANT_SUFFIX_IDS = [
    QWEN_IM_END_ID,
    QWEN_NEWLINE_ID,
    QWEN_IM_START_ID,
    QWEN_ASSISTANT_ID,
    QWEN_NEWLINE_ID,
]
AURA_SILENT_TOKEN_IDS = [151669]
QWEN_TEXT_SILENT_TOKEN_IDS = [27, 91, 68658, 91, 29]
QWEN_TEXT_SILENT_PREFIX_TOKEN_IDS = [
    [27],
    [27, 91],
    [27, 91, 34804],
    [27, 91, 34804, 91],
]
DEFAULT_QWEN3_TTS_REF_AUDIO = "vllm-omni/tests/assets/qwen3_tts/clone_2.wav"
DEFAULT_QWEN3_TTS_REF_TEXT = (
    "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."
)

_AURA_TTS_INFO_KEYS = (
    "tts_task_type",
    "tts_language",
    "tts_instruct",
    "tts_max_new_tokens",
    "tts_ref_audio",
    "tts_ref_text",
    "tts_x_vector_only_mode",
    "tts_speaker",
    "tts_non_streaming_mode",
    "tts_ref_code_length",
    "tts_pass_token_ids",
)


def default_qwen3_tts_ref_audio_path() -> str:
    """Return absolute path to the bundled ``clone_2.wav`` reference asset."""
    bundled = Path(__file__).resolve().parents[3] / "tests" / "assets" / "qwen3_tts" / "clone_2.wav"
    if bundled.is_file():
        return str(bundled)
    return DEFAULT_QWEN3_TTS_REF_AUDIO


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _as_prompt_dict(prompt_item: Any) -> dict[str, Any]:
    return prompt_item if isinstance(prompt_item, dict) else {}


def _first_value(value: Any, default: Any = None) -> Any:
    if isinstance(value, list):
        return value[0] if value else default
    return default if value is None else value


def _first_bool(value: Any, default: bool = False) -> bool:
    value = value[0] if isinstance(value, list) and value else value or default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _first_str(value: Any, default: str = "") -> str:
    value = _first_value(value, default)
    return value if isinstance(value, str) else default


def _normalize_qwen3_tts_speaker(speaker: Any) -> Any:
    if not isinstance(speaker, str):
        return speaker
    speaker = speaker.strip()
    if not speaker:
        return speaker
    if "_" in speaker:
        return speaker
    return speaker[0].upper() + speaker[1:].lower()


def _extract_output(source_output: Any) -> Any:
    outputs = getattr(source_output, "outputs", None)
    if isinstance(outputs, list) and outputs:
        return outputs[0]
    return source_output


def _is_finished(source_output: Any) -> bool:
    return bool(getattr(source_output, "finished", False))


def _extract_text(source_output: Any) -> str:
    output = _extract_output(source_output)
    cumulative_text = getattr(output, "cumulative_text", None)
    if isinstance(cumulative_text, str) and cumulative_text:
        return cumulative_text
    text = getattr(output, "text", None)
    if isinstance(text, str):
        return text
    mm = getattr(output, "multimodal_output", None)
    if isinstance(mm, dict):
        for key in ("text", "transcript", "asr_text"):
            value = mm.get(key)
            if isinstance(value, str):
                return value
            if isinstance(value, list) and value and isinstance(value[0], str):
                return value[0]
    return ""


def _clean_asr_transcript(text: str) -> str:
    """Strip Qwen3-ASR wrappers like ``language Chinese<asr_text>...``."""
    if not isinstance(text, str):
        return ""
    cleaned = text.strip()
    if "<asr_text>" in cleaned:
        cleaned = cleaned.split("<asr_text>", 1)[-1]
    cleaned = re.sub(r"^language\s+[\w-]+\s*", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _normalize_request_id_for_transcript(request_id: str) -> str:
    """Map AsyncOmni internal ids (``{external}-{uuid8}``) to the external id.

    ``AsyncOmni.generate()`` rewrites ``request_id`` to an internal orchestrator id
    while the streaming handler keeps the external id for ``on_turn_complete``.
    Transcript storage must use the external id so ``pop_turn_transcript`` works.
    """
    rid = str(request_id).strip()
    if not rid:
        return rid
    head, tail = rid.rsplit("-", 1)
    if len(tail) == 8 and all(ch in "0123456789abcdefABCDEF" for ch in tail):
        return head
    return rid


def record_turn_transcript(request_id: str, transcript: str) -> None:
    _TURN_TRANSCRIPTS_BY_REQUEST[_normalize_request_id_for_transcript(request_id)] = transcript


def pop_turn_transcript(request_id: str | None) -> str:
    if not request_id:
        return ""
    return _TURN_TRANSCRIPTS_BY_REQUEST.pop(_normalize_request_id_for_transcript(str(request_id)), "")


def _extract_token_ids(source_output: Any) -> list[int]:
    output = _extract_output(source_output)
    token_ids = getattr(output, "cumulative_token_ids", None)
    if isinstance(token_ids, list):
        return [int(token_id) for token_id in token_ids if isinstance(token_id, int)]
    return []


def _strip_boundary_chat_template_tokens(token_ids: list[int]) -> list[int]:
    seg = list(token_ids)
    while seg and seg[0] in {QWEN_IM_START_ID, QWEN_IM_END_ID, QWEN_NEWLINE_ID, QWEN_ASSISTANT_ID}:
        seg.pop(0)
    while seg and seg[-1] in {QWEN_IM_START_ID, QWEN_IM_END_ID, QWEN_NEWLINE_ID}:
        seg.pop()
    return seg


def _remove_inline_silent_runs(token_ids: list[int]) -> list[int]:
    if not token_ids:
        return []
    silent_sequences = [AURA_SILENT_TOKEN_IDS, QWEN_TEXT_SILENT_TOKEN_IDS]
    result: list[int] = []
    i = 0
    while i < len(token_ids):
        skipped = False
        for silent_seq in silent_sequences:
            seq_len = len(silent_seq)
            if seq_len and token_ids[i : i + seq_len] == silent_seq:
                i += seq_len
                skipped = True
                break
        if skipped:
            continue
        result.append(token_ids[i])
        i += 1
    return result


def _extract_aura_speakable_token_ids(token_ids: list[int]) -> list[int]:
    """Collect all non-silent AURA content across multi-turn chat-template output.

    AURA may emit multiple assistant segments in one generation, separated by
    ``im_end`` markers, before trailing ``<|silent|>`` tokens. Legacy trimming
    stopped at the first ``im_end`` and dropped later speakable paragraphs.
    """
    if not token_ids:
        return []

    ids = list(token_ids)
    if ids[: len(QWEN_ASSISTANT_PREFIX_IDS)] == QWEN_ASSISTANT_PREFIX_IDS:
        ids = ids[len(QWEN_ASSISTANT_PREFIX_IDS) :]

    segments: list[list[int]] = []
    start = 0
    for idx, token_id in enumerate(ids):
        if token_id == QWEN_IM_END_ID:
            if idx > start:
                segments.append(ids[start:idx])
            start = idx + 1
    if start < len(ids):
        segments.append(ids[start:])

    speakable: list[int] = []
    for segment in segments:
        cleaned = _remove_inline_silent_runs(_strip_boundary_chat_template_tokens(segment))
        if not cleaned or _is_silent_token_prefix(cleaned):
            continue
        speakable.extend(cleaned)
    return speakable


def _trim_aura_response_token_ids(token_ids: list[int]) -> list[int]:
    return _extract_aura_speakable_token_ids(token_ids)


def _qwen3_tts_assistant_token_ids_from_aura(source_output: Any) -> list[int]:
    content_ids = _trim_aura_response_token_ids(_extract_token_ids(source_output))
    if not content_ids:
        return []
    return QWEN_ASSISTANT_PREFIX_IDS + content_ids + QWEN_ASSISTANT_SUFFIX_IDS


def _ensure_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().reshape(-1).tolist()
    return [int(token_id) for token_id in list(value) if isinstance(token_id, int)]


def _is_silent_token_prefix(content_ids: list[int]) -> bool:
    if not content_ids:
        return False
    candidates = [
        AURA_SILENT_TOKEN_IDS,
        QWEN_TEXT_SILENT_TOKEN_IDS,
        *QWEN_TEXT_SILENT_PREFIX_TOKEN_IDS,
    ]
    return any(candidate[: len(content_ids)] == content_ids for candidate in candidates)


def _request_additional_info(request: Any) -> dict[str, Any]:
    def decode_info(raw_info: Any) -> dict[str, Any]:
        if isinstance(raw_info, dict):
            return raw_info
        info = deserialize_additional_information(raw_info)
        return info if isinstance(info, dict) else {}

    info = decode_info(getattr(request, "omni_stage_payload", None))
    current_info = decode_info(getattr(request, "additional_information", None))
    if current_info:
        info = {**info, **current_info}

    nested_info = info.get("additional_information") if isinstance(info, dict) else None
    if nested_info is not None:
        nested_info = decode_info(nested_info)
        if isinstance(nested_info, dict):
            # Connector payloads wrap the original OpenAI request metadata under
            # `additional_information`; expose those tts_* keys at the level
            # consumed by asr2aura/aura2tts while preserving explicit top-level
            # overrides.
            info = {**nested_info, **info}

    return info


def _request_output_text(request: Any) -> str:
    output_text = getattr(request, "output_text", None)
    if isinstance(output_text, str):
        return output_text
    if isinstance(output_text, list) and output_text and isinstance(output_text[0], str):
        return output_text[0]
    return ""


def _aura_request_log_ctx(request: Any, *, is_finished: bool | None = None) -> str:
    """Compact request context string for aura pipeline debug logs."""
    req_id = getattr(request, "request_id", None)
    ext_req_id = getattr(request, "external_req_id", None)
    output_token_ids = _ensure_int_list(getattr(request, "output_token_ids", []) or [])
    output_text = _request_output_text(request)
    request_finished = bool(getattr(request, "is_finished", lambda: False)())
    resumable = getattr(request, "resumable", None)
    num_computed = getattr(request, "num_computed_tokens", None)
    num_output_ph = getattr(request, "num_output_placeholders", None)
    parts = [
        f"req={req_id}",
        f"ext={ext_req_id}",
        f"resumable={resumable}",
        f"request_finished={request_finished}",
    ]
    if is_finished is not None:
        parts.append(f"is_finished_arg={is_finished}")
    parts.extend(
        [
            f"output_token_ids_len={len(output_token_ids)}",
            f"output_text_len={len(output_text)}",
            f"num_computed_tokens={num_computed}",
            f"num_output_placeholders={num_output_ph}",
        ]
    )
    if output_text:
        preview = output_text[:120] + ("..." if len(output_text) > 120 else "")
        parts.append(f"output_text_preview={preview!r}")
    if output_token_ids:
        parts.append(f"output_token_ids_tail={output_token_ids[-8:]}")
    return " ".join(parts)


def _clean_asr_text(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    cleaned = text
    for marker in (
        "<|im_start|>",
        "<|im_end|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|video_pad|>",
        "<|image_pad|>",
    ):
        cleaned = cleaned.replace(marker, "")
    return " ".join(cleaned.split()).strip()


def _clean_tts_text(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    return " ".join(text.split()).strip()


def _aura_system_prompt(additional_info: dict[str, Any]) -> str:
    return _first_str(additional_info.get("aura_system_prompt"), DEFAULT_AURA_SYSTEM_PROMPT) or DEFAULT_AURA_SYSTEM_PROMPT


def _merged_vision_multimodal_data(*sources: Any) -> dict[str, Any]:
    multi_modal_data: dict[str, Any] = {}
    for source in sources:
        if isinstance(source, dict):
            multi_modal_data.update(source)
    return _vision_multimodal_data(multi_modal_data)


def _tts_additional_info(additional_info: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in additional_info.items() if isinstance(key, str) and key.startswith("tts_")}


def _build_aura_input_payload(
    *,
    transcript: str,
    additional_info: dict[str, Any],
    multi_modal_data: dict[str, Any],
    requires_multimodal_data: bool,
    mm_processor_kwargs: Any = None,
    tokenizer: Any | None = None,
    include_empty_multimodal_data: bool = False,
) -> dict[str, Any]:
    prompt = _aura_prompt(_aura_system_prompt(additional_info), transcript, multi_modal_data)
    next_input: dict[str, Any] = {"prompt": prompt}
    if tokenizer is not None:
        prompt_token_ids = tokenizer.encode(prompt)
        next_input["prompt_token_ids"] = prompt_token_ids
        next_input["ids"] = {"prompt": prompt_token_ids}
    if requires_multimodal_data and (multi_modal_data or include_empty_multimodal_data):
        next_input["multi_modal_data"] = multi_modal_data
    if mm_processor_kwargs is not None:
        next_input["mm_processor_kwargs"] = mm_processor_kwargs
    tts_info = _tts_additional_info(additional_info)
    if tts_info:
        next_input["additional_information"] = tts_info
    return next_input


def _tts_info_and_prompt_len(
    additional_info: dict[str, Any],
    *,
    assistant_token_ids: list[int],
    text: str,
    prompt_len_token_ids: list[int] | None = None,
    default_language: str = "Chinese",
    allow_default_base_refs: bool = True,
) -> tuple[dict[str, Any], int]:
    task_type = _first_value(additional_info.get("tts_task_type"), "Base")
    language = _first_value(additional_info.get("tts_language"), default_language)
    instruct = _first_value(additional_info.get("tts_instruct"), "")
    max_new_tokens = _first_value(additional_info.get("tts_max_new_tokens"), 2048)
    tts_info: dict[str, Any] = {
        "task_type": [task_type],
        "language": [language],
        "instruct": [instruct],
        "max_new_tokens": [int(max_new_tokens)],
    }
    if assistant_token_ids:
        tts_info[PRECOMPUTED_TEXT_IDS_KEY] = [assistant_token_ids]
        prompt_len = _estimate_tts_prompt_len_from_token_ids(
            assistant_token_ids,
            task_type=str(task_type),
            language=str(language),
            instruct=str(instruct),
        )
    else:
        text = _clean_tts_text(text)
        tts_info["text"] = [text]
        text_token_count_proxy = max(1, len(text)) if isinstance(text, str) else 1
        token_ids_for_len = (
            prompt_len_token_ids
            if prompt_len_token_ids
            else [0] * (text_token_count_proxy + len(QWEN_ASSISTANT_PREFIX_IDS) + len(QWEN_ASSISTANT_SUFFIX_IDS))
        )
        prompt_len = _estimate_tts_prompt_len_from_token_ids(
            token_ids_for_len,
            task_type=str(task_type),
            language=str(language),
            instruct=str(instruct),
        )

    if task_type == "Base":
        ref_audio = _first_value(additional_info.get("tts_ref_audio"), None)
        ref_text = _first_value(additional_info.get("tts_ref_text"), None)
        if allow_default_base_refs:
            ref_audio = ref_audio or default_qwen3_tts_ref_audio_path()
            ref_text = ref_text or DEFAULT_QWEN3_TTS_REF_TEXT
        x_vector_only_mode = _first_bool(additional_info.get("tts_x_vector_only_mode"), False)
        ref_code_len_value = _first_value(additional_info.get("tts_ref_code_length"), None)
        ref_code_len = int(ref_code_len_value) if isinstance(ref_code_len_value, int) else None
        if not x_vector_only_mode and ref_code_len is None:
            ref_code_len = _estimate_ref_code_len_from_ref_audio(ref_audio)
        if ref_code_len is not None:
            tts_info["ref_code_length"] = [int(ref_code_len)]
            prompt_len = _estimate_tts_prompt_len_from_token_ids(
                assistant_token_ids if assistant_token_ids else [0] * max(1, len(text)),
                task_type="Base",
                language=str(language),
                instruct=str(instruct),
                x_vector_only_mode=x_vector_only_mode,
                ref_code_len=ref_code_len,
            )
        if ref_audio:
            tts_info["ref_audio"] = [ref_audio]
        if ref_text:
            tts_info["ref_text"] = [ref_text]
        tts_info["x_vector_only_mode"] = [x_vector_only_mode]

    elif task_type == "CustomVoice":
        speaker = _first_value(additional_info.get("tts_speaker"), "Vivian")
        tts_info["speaker"] = [_normalize_qwen3_tts_speaker(speaker)]

    return tts_info, prompt_len


def _source_prompt_by_request_id(source_outputs: list[Any], prompt: Any) -> dict[str, dict[str, Any]]:
    prompts = _as_list(prompt)
    return {
        str(getattr(source_output, "request_id", idx)): _as_prompt_dict(prompt_item)
        for idx, (source_output, prompt_item) in enumerate(zip(source_outputs, prompts))
    }


def _vision_placeholder(multi_modal_data: dict[str, Any]) -> str:
    if "video" in multi_modal_data:
        return "<|vision_start|><|video_pad|><|vision_end|>"
    if "image" in multi_modal_data:
        return "<|vision_start|><|image_pad|><|vision_end|>"
    return ""


def _vision_multimodal_data(multi_modal_data: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in multi_modal_data.items() if key in {"image", "video"}}


def _aura_prompt(system_prompt: str, transcript: str, multi_modal_data: dict[str, Any]) -> str:
    vision = _vision_placeholder(multi_modal_data)
    query = transcript.strip()
    user_body = f"{vision}{query}" if query else vision

    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|><|im_start|>user\n{user_body}<|im_end|><|im_start|>assistant\n"
    )


def asr2aura(
    source_outputs: list[Any],
    prompt: Any = None,
    requires_multimodal_data: bool = True,
) -> list[dict[str, Any]]:
    """Build AURA Qwen3-VL prompts from ASR transcripts and original video payloads."""
    prompt_by_request_id = _source_prompt_by_request_id(source_outputs, prompt)
    next_inputs: list[dict[str, Any]] = []
    for idx, source_output in enumerate(source_outputs):
        src_prompt = prompt_by_request_id.get(str(getattr(source_output, "request_id", idx)), {})
        additional_info = src_prompt.get("additional_information") or {}
        transcript = _clean_asr_transcript(_extract_text(source_output))
        multi_modal_data = _merged_vision_multimodal_data(
            src_prompt.get("multi_modal_data") or {},
            additional_info.get("deferred_multi_modal_data") or {},
        )

        next_inputs.append(
            _build_aura_input_payload(
                transcript=transcript,
                additional_info=additional_info,
                multi_modal_data=multi_modal_data,
                requires_multimodal_data=requires_multimodal_data,
                mm_processor_kwargs=src_prompt.get("mm_processor_kwargs"),
                include_empty_multimodal_data=True,
            )
        )
    return next_inputs


def asr2aura_async_chunk(
    transfer_manager: Any,
    multimodal_output: Any | None = None,
    request: Any | None = None,
    is_finished: bool = False,
    **_: Any,
) -> dict[str, Any] | None:
    """Accumulate ASR text chunks and emit one complete AURA input at ASR finish."""
    del multimodal_output
    if request is None:
        raise ValueError("asr2aura_async_chunk requires request.")

    request_id = getattr(request, "external_req_id", None) or getattr(request, "request_id", None)
    finished = bool(is_finished or request.is_finished())
    request_payload = getattr(transfer_manager, "request_payload", None)
    if request_payload is None:
        request_payload = {}
        transfer_manager.request_payload = request_payload
    state = request_payload.setdefault(str(request_id), {})
    if not isinstance(state, dict):
        state = {}
        request_payload[str(request_id)] = state

    output_text = _request_output_text(request)
    if output_text:
        previous_text = str(state.get("asr_text", ""))
        cleaned_output_text = _clean_asr_text(output_text)
        state["asr_text"] = (
            cleaned_output_text
            if cleaned_output_text.startswith(previous_text)
            else _clean_asr_text(previous_text + output_text)
        )

    if not finished:
        return None

    tokenizer = cached_tokenizer_from_config(transfer_manager.config)
    if not state.get("asr_text"):
        token_ids = _ensure_int_list(getattr(request, "output_token_ids", []) or [])
        if token_ids:
            state["asr_text"] = _clean_asr_text(tokenizer.decode(token_ids))

    additional_info = _request_additional_info(request)
    multi_modal_data = _merged_vision_multimodal_data(
        getattr(request, "multi_modal_data", None) or {},
        additional_info.get("deferred_multi_modal_data") or {},
    )
    mm_processor_kwargs = getattr(request, "mm_processor_kwargs", None)

    payload = _build_aura_input_payload(
        transcript=str(state.get("asr_text", "")),
        additional_info=additional_info,
        multi_modal_data=multi_modal_data,
        requires_multimodal_data=True,
        mm_processor_kwargs=mm_processor_kwargs,
        tokenizer=tokenizer,
    )
    return payload


def video_tuple_from_aura_turn_video(aura_turn_video: Any) -> tuple[np.ndarray, dict[str, Any]] | None:
    if not isinstance(aura_turn_video, dict):
        return None
    frames = aura_turn_video.get("frames")
    if frames is None:
        return None
    metadata = dict(aura_turn_video.get("metadata") or {})
    video_array = np.asarray(frames, dtype=np.uint8)
    if video_array.ndim != 4:
        return None
    if video_array.shape[0] < 2:
        video_array = np.concatenate([video_array, video_array], axis=0)[:2]
        metadata = dict(metadata)
        metadata["total_num_frames"] = 2
        metadata["duration"] = 2 / float(metadata.get("fps", 2.0))
    return video_array, metadata


def _copy_aura_tts_fields(additional_info: dict[str, Any]) -> dict[str, Any]:
    copied: dict[str, Any] = {}
    for key in _AURA_TTS_INFO_KEYS:
        if key in additional_info:
            copied[key] = additional_info[key]
    return copied


def asr2aura_session(
    source_outputs: list[Any],
    prompt: Any = None,
    requires_multimodal_data: bool = True,
) -> list[dict[str, Any]]:
    """Build AURA prompts from ASR transcripts and server-side or serialized SessionHistory."""
    from vllm_omni.entrypoints.openai.aura_session_store import get_session_history

    prompt_by_request_id = _source_prompt_by_request_id(source_outputs, prompt)
    next_inputs: list[dict[str, Any]] = []

    for idx, source_output in enumerate(source_outputs):
        src_prompt = prompt_by_request_id.get(str(getattr(source_output, "request_id", idx)), {})
        additional_info = src_prompt.get("additional_information") or {}
        session_id = additional_info.get("aura_session_id")
        history = get_session_history(str(session_id)) if session_id else None
        if session_id and history is None:
            logger.warning(
                "AURA session_id=%s not found in server-side store; "
                "falling back to aura_session_state or single-turn asr2aura",
                session_id,
            )
        if history is None and additional_info.get("aura_session_state") is None:
            [next_input] = asr2aura(
                [source_output],
                prompt=[src_prompt],
                requires_multimodal_data=requires_multimodal_data,
            )
            next_inputs.append(next_input)
            continue

        request_id = str(getattr(source_output, "request_id", idx))
        transcript = _clean_asr_transcript(_extract_text(source_output))
        record_turn_transcript(request_id, transcript)
        video_tuple = video_tuple_from_aura_turn_video(additional_info.get("aura_turn_video"))
        if history is None:
            history = SessionHistory.from_dict(additional_info["aura_session_state"])
            history.add_user_message(transcript, video_tuple=video_tuple)
            vllm_inputs = history.get_vllm_inputs()
        else:
            vllm_inputs = history.preview_vllm_inputs(transcript, video_tuple=video_tuple)

        next_input = {
            "prompt": vllm_inputs["prompt"],
            "additional_information": _copy_aura_tts_fields(additional_info),
        }
        system_prompt = _first_value(additional_info.get("aura_system_prompt"), DEFAULT_AURA_SYSTEM_PROMPT)
        next_input["additional_information"]["aura_system_prompt"] = [str(system_prompt)]

        if requires_multimodal_data:
            next_input["multi_modal_data"] = vllm_inputs.get("multi_modal_data", {})
        if src_prompt.get("mm_processor_kwargs") is not None:
            next_input["mm_processor_kwargs"] = src_prompt.get("mm_processor_kwargs")
        next_inputs.append(next_input)

    return next_inputs


def _estimate_ref_code_len_from_ref_audio(ref_audio: Any) -> int | None:
    """Estimate Qwen3-TTS ref_code length from a ref-audio payload.

    For Qwen3-TTS 12Hz models, code length is approximately:
        ceil(duration_seconds * 12.5)
    i.e. one codec frame per 1920 samples at 24kHz.
    """

    codec_frame_rate = 24000.0 / 1920.0

    # Unwrap common list wrappers.
    item = ref_audio
    while isinstance(item, list) and item:
        item = item[0]

    # Accept tuple/list like (wav, sr).
    if isinstance(item, (tuple, list)) and len(item) == 2 and isinstance(item[1], (int, float)):
        wav, sr = item
        sr_i = int(sr)
        if sr_i <= 0:
            return None
        if hasattr(wav, "__len__"):
            n_samples = len(wav)
        elif hasattr(wav, "shape"):
            shape = getattr(wav, "shape", None)
            if not shape:
                return None
            n_samples = shape[-1] if len(shape) > 1 else shape[0]
        else:
            return None
        if n_samples <= 0:
            return None
        return max(1, int(math.ceil((float(n_samples) / float(sr_i)) * codec_frame_rate)))

    # Accept file path (wav only).
    if isinstance(item, str) and item:
        audio_path = item
        if not os.path.isfile(audio_path) or not audio_path.lower().endswith(".wav"):
            return None
        try:
            info = sf.info(audio_path)
            n_frames = int(info.frames)
            sr = int(info.samplerate)
            if n_frames <= 0 or sr <= 0:
                return None
            return max(1, int(math.ceil((float(n_frames) / float(sr)) * codec_frame_rate)))
        except Exception:
            return None

    return None


def _estimate_tts_prompt_len_from_token_ids(
    token_ids: list[int],
    *,
    task_type: str = "Base",
    language: str = "Chinese",
    instruct: str = "",
    x_vector_only_mode: bool = False,
    non_streaming_mode: bool | None = None,
    ref_code_len: int | None = None,
) -> int:
    """Estimate Talker prefill length from prompt structure.

    This mirrors Qwen3-TTS prompt assembly at length level:
      prompt_len = instruct_len + role_len + codec_prefix_len + text/icl term
    """

    # Official defaults: Base -> streaming, others -> non-streaming.
    if non_streaming_mode is None:
        non_streaming_mode = task_type in ("CustomVoice", "VoiceDesign")

    # We do not have tokenizer here; use char length as a monotonic proxy.
    instruct_len = len(instruct.strip()) if isinstance(instruct, str) else 0
    assistant_len = max(0, len(token_ids))

    # role_len = 3; codec_prefix_len = (prefill_len + speaker_len + 2) - 1
    # prefill_len = 4 when language_id exists else 3. Use non-auto language as
    # the language-id-present proxy.
    has_language_id = isinstance(language, str) and language.strip().lower() != "auto"
    prefill_len = 4 if has_language_id else 3
    speaker_len = 1 if task_type in ("CustomVoice", "Base") else 0
    base_len = instruct_len + 3 + (prefill_len + speaker_len + 2 - 1)
    if task_type in ("CustomVoice", "VoiceDesign"):
        if non_streaming_mode:
            prompt_len = base_len + max(0, assistant_len - 6)
        else:
            prompt_len = base_len + 1
        return int(prompt_len)

    if task_type == "Base":
        in_context_mode = not bool(x_vector_only_mode)
        if in_context_mode and ref_code_len is not None:
            codec_lens = 1 + int(ref_code_len)
            if non_streaming_mode:
                # Exact non-streaming ICL needs ref_ids token length; unavailable
                # in this processor. Keep a conservative upper estimate.
                prompt_len = base_len + codec_lens + max(0, assistant_len - 8) + 1
            else:
                # Streaming ICL exact length term: 1 + ref_code_len
                prompt_len = base_len + codec_lens
        else:
            # Base x-vector-only (or missing ref_code length) follows CV shape.
            if non_streaming_mode:
                prompt_len = base_len + max(0, assistant_len - 6)
            else:
                prompt_len = base_len + 1
        return int(prompt_len)

    # Defensive fallback for unknown task types.
    return int(base_len + max(assistant_len, 1))


def aura2tts(
    source_outputs: list[Any],
    prompt: Any = None,
    requires_multimodal_data: bool = False,
    streaming_context: Any | None = None,
) -> list[OmniTokensPrompt]:
    """Convert AURA text output into Qwen3-TTS Talker requests."""
    del requires_multimodal_data
    del streaming_context
    prompt_by_request_id = _source_prompt_by_request_id(source_outputs, prompt)
    next_inputs: list[OmniTokensPrompt] = []
    for idx, source_output in enumerate(source_outputs):
        req_id = getattr(source_output, "request_id", idx)
        text = _extract_text(source_output).strip()
        finished = _is_finished(source_output)
        if not finished and text and SILENT_TEXT.startswith(text):
            # AURA may stream the special silent marker token-by-token. Hold
            # these prefixes until the marker is complete so TTS never speaks it.
            continue
        if is_effectively_silent(text):
            continue

        src_prompt = prompt_by_request_id.get(str(req_id), {})
        additional_info = src_prompt.get("additional_information") or {}
        assistant_token_ids_for_len = _qwen3_tts_assistant_token_ids_from_aura(source_output)
        pass_token_ids = _first_bool(additional_info.get("tts_pass_token_ids"), False)
        use_token_ids = pass_token_ids and bool(assistant_token_ids_for_len)
        tts_info, prompt_len = _tts_info_and_prompt_len(
            additional_info,
            assistant_token_ids=assistant_token_ids_for_len if use_token_ids else [],
            text=text,
            prompt_len_token_ids=assistant_token_ids_for_len if not use_token_ids else None,
            default_language="English",
            allow_default_base_refs=False,
        )

        task_type = _first_value(additional_info.get("tts_task_type"), "Base")
        if task_type == "Base":
            if not tts_info.get("ref_audio") or not tts_info.get("ref_text"):
                raise ValueError("AURA Base TTS requires tts_ref_audio and tts_ref_text.")
        next_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[0] * prompt_len,
                additional_information=tts_info,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )
    return next_inputs


def aura2tts_async_chunk(
    transfer_manager: Any,
    multimodal_output: Any | None = None,
    request: Any | None = None,
    is_finished: bool = False,
    **_: Any,
) -> dict[str, Any] | None:
    """Accumulate AURA output and emit one Qwen3-TTS Talker input at finish."""
    del multimodal_output
    if request is None:
        raise ValueError("aura2tts_async_chunk requires request.")

    request_id = getattr(request, "external_req_id", None) or getattr(request, "request_id", None)
    raw_content_ids = _ensure_int_list(getattr(request, "output_token_ids", []) or [])
    content_ids = _trim_aura_response_token_ids(raw_content_ids)
    finished = bool(is_finished or request.is_finished())
    if raw_content_ids and not content_ids:
        return None
    if content_ids and _is_silent_token_prefix(content_ids):
        return None

    request_payload = getattr(transfer_manager, "request_payload", None)
    if request_payload is None:
        request_payload = {}
        transfer_manager.request_payload = request_payload
    state = request_payload.setdefault(str(request_id), {})
    if not isinstance(state, dict):
        state = {}
        request_payload[str(request_id)] = state

    request_text = _clean_tts_text(_request_output_text(request))
    if content_ids:
        state["aura2tts_content_ids"] = content_ids
    if request_text:
        previous_text = str(state.get("aura2tts_text", ""))
        state["aura2tts_text"] = (
            request_text if request_text.startswith(previous_text) else _clean_tts_text(previous_text + request_text)
        )

    additional_info = _request_additional_info(request)
    tts_metadata = _tts_additional_info(additional_info)
    if tts_metadata:
        state["aura2tts_tts_metadata"] = dict(tts_metadata)

    if not finished:
        return None

    content_ids = list(state.get("aura2tts_content_ids", content_ids) or [])
    if not content_ids:
        return None
    if _is_silent_token_prefix(content_ids):
        return None

    cached_tts_metadata = state.get("aura2tts_tts_metadata")
    if isinstance(cached_tts_metadata, dict):
        additional_info = {**cached_tts_metadata, **additional_info}
    pass_token_ids = _first_bool(additional_info.get("tts_pass_token_ids"), False)
    request_text = _clean_tts_text(str(state.get("aura2tts_text", ""))) or request_text
    if not request_text and not pass_token_ids:
        try:
            tokenizer = cached_tokenizer_from_config(transfer_manager.config)
            request_text = _clean_tts_text(tokenizer.decode(content_ids))
        except Exception:
            logger.exception(
                "Failed to decode AURA token ids for req=%s; falling back to token ids",
                getattr(request, "request_id", None),
            )

    assistant_token_ids = QWEN_ASSISTANT_PREFIX_IDS + content_ids + QWEN_ASSISTANT_SUFFIX_IDS
    use_token_ids = pass_token_ids
    if not use_token_ids and not request_text:
        use_token_ids = True
    tts_info, prompt_len = _tts_info_and_prompt_len(
        additional_info,
        assistant_token_ids=assistant_token_ids if use_token_ids else [],
        text=request_text,
        prompt_len_token_ids=assistant_token_ids if not use_token_ids else None,
    )

    payload = {
        **tts_info,
        "prompt_token_ids": [0] * prompt_len,
    }
    return payload
