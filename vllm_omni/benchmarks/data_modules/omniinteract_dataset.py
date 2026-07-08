"""OmniInteract dataset loader for ``vllm bench serve --omni``.

OmniInteract is a streaming audio-visual QA benchmark:
https://huggingface.co/datasets/lucky-lance/OmniInteract

For non-streaming modes this loader flattens per-QA annotation entries into
independent benchmark requests. For ``1q1a`` / ``1q1a_math`` each request uses
the per-QA ``subvideos/{video}_{qa_idx}.mp4`` clip together with the matching
``audios/{video}_{qa_idx}.wav`` when present. The default ``video`` input mode
sends one ``video_url`` (native audio in the video stream) plus a text question.
The ``aura`` input mode sends only ``audio_url`` + ``video_url`` so ASR receives
the spoken question while AURA receives the subvideo clip.

``aura_streaming`` follows the original OmniInteract online protocol: one
request streams the full ``videos/*.mp4`` file and injects the per-QA audio
question at the annotation ``question_time``.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from vllm.benchmarks.datasets import BenchmarkDataset, SampleRequest
from vllm.tokenizers import TokenizerLike
from vllm.tokenizers.hf import get_cached_tokenizer
from vllm_omni.model_executor.stage_input_processors.aura_session_history import (
    DEFAULT_AURA_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)

OmniInteractSubset = Literal["1q1a", "1q1a_math", "1qna"]
OmniInteractInputMode = Literal["video", "aura", "aura_streaming"]

DEFAULT_AURA_SYSTEM_PROMPT_FOR_OMNIINTERACT = (
    "You are answering OmniInteract audio-visual QA tasks. Use the ASR transcript "
    "of the user's spoken question together with the video frames. Answer the "
    "question directly and concisely in the same language as the question. "
    "Do not output '<|silent|>'."
)


def _is_remote_or_data_url(value: str) -> bool:
    return value.startswith(("http://", "https://", "data:"))


@dataclass
class OmniInteractSampleRequest(SampleRequest):
    """``SampleRequest`` carrying OmniInteract labels and request fields."""

    omniinteract_gold_answer: str = ""
    omniinteract_subset: str = ""
    omniinteract_question_type: str = ""
    omniinteract_video: str = ""
    omniinteract_question_time: str = ""
    omniinteract_answer_time: str = ""
    omniinteract_is_interrupted: bool | None = None
    omniinteract_scene_type: str = ""
    omniinteract_nested_group_id: int | None = None
    omniinteract_nested_role: str = ""
    omni_extra_body: dict[str, Any] | None = None
    omni_chat_messages: list[dict[str, Any]] | None = None
    omniinteract_streaming_video_path: str = ""
    omniinteract_streaming_audio_path: str = ""
    omniinteract_streaming_audio_schedule: list[dict[str, Any]] | None = None
    omniinteract_streaming_audio_from_video: bool = False
    omniinteract_streaming_slots: list[dict[str, Any]] | None = None
    omniinteract_streaming_config: dict[str, Any] | None = None


@dataclass
class _OmniInteractEntry:
    subset: str
    video_rel: str
    video_path: Path
    question_text: str
    answer_text: str
    question_time: str
    answer_time: str
    question_type: str
    is_interrupted: bool | None
    audio_path: Path | None = None
    scene_type: str = "multi_turn"
    nested_group_id: int | None = None
    nested_role: str = ""


@dataclass
class _OmniInteractStreamingEntry:
    subset: str
    video_rel: str
    video_path: Path
    annotation_path: Path
    scene_type: str
    rows: list[dict[str, Any]]


def aura_sampling_params_list() -> list[dict[str, Any]]:
    return [
        {"temperature": 0.0, "top_p": 1.0, "top_k": -1, "max_tokens": 256, "seed": 42},
        {
            "temperature": 0.5,
            "top_p": 1.0,
            "top_k": -1,
            "max_tokens": 256,
            "seed": 42,
            "repetition_penalty": 1.0,
            "stop_token_ids": [151669, 151645],
        },
        {
            "temperature": 0.9,
            "top_k": 50,
            "max_tokens": 4096,
            "seed": 42,
            "detokenize": False,
            "repetition_penalty": 1.05,
            "stop_token_ids": [2150],
        },
        {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "max_tokens": 65536,
            "seed": 42,
            "repetition_penalty": 1.0,
        },
    ]


def aura_extra_body(
    *,
    tts_task_type: str,
    tts_language: str,
    tts_speaker: str | None = None,
    tts_ref_audio: str | None = None,
    tts_ref_text: str | None = None,
) -> dict[str, Any]:
    additional_information: dict[str, Any] = {
        "aura_system_prompt": DEFAULT_AURA_SYSTEM_PROMPT_FOR_OMNIINTERACT,
        "tts_task_type": tts_task_type,
        "tts_language": tts_language,
    }
    if tts_task_type == "Base":
        if not tts_ref_audio or not tts_ref_text:
            raise ValueError(
                "OmniInteract AURA Base TTS requires both --omniinteract-aura-tts-ref-audio "
                "and --omniinteract-aura-tts-ref-text."
            )
        additional_information["tts_ref_audio"] = tts_ref_audio
        additional_information["tts_ref_text"] = tts_ref_text
        # Match native AURA tts_service: full-sentence ICL synthesis per turn.
        additional_information["tts_non_streaming_mode"] = True
    elif tts_speaker:
        additional_information["tts_speaker"] = tts_speaker
    return {
        "modalities": ["text", "audio"],
        "mm_processor_kwargs": {"use_audio_in_video": False},
        "sampling_params_list": aura_sampling_params_list(),
        "additional_information": additional_information,
    }


def resolve_aura_streaming_system_prompt(
    *,
    mode: Literal["native", "omniinteract_qa"] = "native",
    override: str | None = None,
) -> str:
    """Pick system prompt for aura_streaming ``session.config``.

    ``native`` matches original AURA TCP server (allows ``<|silent|>``).
    ``omniinteract_qa`` is the legacy bench prompt (forbids silent; QA-style).
    """
    if override:
        return override
    if mode == "omniinteract_qa":
        return DEFAULT_AURA_SYSTEM_PROMPT_FOR_OMNIINTERACT
    return DEFAULT_AURA_SYSTEM_PROMPT


def aura_streaming_config(
    *,
    tts_task_type: str,
    tts_language: str,
    tts_speaker: str | None = None,
    tts_ref_audio: str | None = None,
    tts_ref_text: str | None = None,
    sample_fps: float = 2.0,
    max_frames: int = 0,
    max_frames_per_round: int = 16,
    auto_trigger_min_frames: int = 2,
    send_fps: float = 2.0,
    enable_frame_filter: bool = False,
    cross_turn_penalty: float = 0.0,
    cross_turn_lookback: int = 2,
    aura_system_prompt_mode: Literal["native", "omniinteract_qa"] = "native",
    aura_system_prompt: str | None = None,
) -> dict[str, Any]:
    """Build ``session.config`` fields for AURA WebSocket streaming benchmark."""

    config: dict[str, Any] = {
        "modalities": ["text", "audio"],
        "auto_trigger": True,
        "auto_trigger_min_frames": int(auto_trigger_min_frames),
        "max_frames": int(max_frames),
        "max_frames_per_round": int(max_frames_per_round),
        "video_fps": float(sample_fps),
        "send_fps": float(send_fps),
        "enable_frame_filter": bool(enable_frame_filter),
        "sampling_params_list": aura_sampling_params_list(),
        "aura_system_prompt": resolve_aura_streaming_system_prompt(
            mode=aura_system_prompt_mode,
            override=aura_system_prompt,
        ),
        "tts_task_type": tts_task_type,
        "tts_language": tts_language,
    }
    if tts_task_type == "Base":
        if not tts_ref_audio or not tts_ref_text:
            raise ValueError(
                "OmniInteract AURA streaming Base TTS requires both "
                "--omniinteract-aura-tts-ref-audio and --omniinteract-aura-tts-ref-text."
            )
        config["tts_ref_audio"] = tts_ref_audio
        config["tts_ref_text"] = tts_ref_text
        # Match native AURA tts_service: full-sentence ICL synthesis per turn.
        config["tts_non_streaming_mode"] = True
    elif tts_speaker:
        config["tts_speaker"] = tts_speaker
    if float(cross_turn_penalty) > 0:
        config["cross_turn_penalty"] = float(cross_turn_penalty)
        config["cross_turn_lookback"] = max(1, int(cross_turn_lookback))
    return config


def _parse_time_seconds(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    s = value.strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        pass
    parts = s.split(":")
    if len(parts) not in (2, 3):
        return None
    try:
        nums = [float(x) for x in parts]
    except ValueError:
        return None
    if len(nums) == 2:
        minutes, seconds = nums
        return minutes * 60.0 + seconds
    hours, minutes, seconds = nums
    return hours * 3600.0 + minutes * 60.0 + seconds


def _infer_nested_roles(ann: list[dict[str, Any]]) -> dict[int, tuple[int, str]]:
    rows: list[tuple[int, float, float]] = []
    for idx, qa in enumerate(ann):
        q_time = _parse_time_seconds(qa.get("question_time"))
        a_time = _parse_time_seconds(qa.get("answer_time"))
        if q_time is None or a_time is None:
            continue
        rows.append((idx, q_time, a_time))
    rows.sort(key=lambda r: (r[1], r[2], r[0]))

    nested_meta: dict[int, tuple[int, str]] = {}
    group_id = 1
    cursor = 0
    while cursor < len(rows):
        outer_idx, outer_q, outer_a = rows[cursor]
        inner_pos: int | None = None
        for cand_pos in range(cursor + 1, len(rows)):
            _, cand_q, cand_a = rows[cand_pos]
            if outer_q < cand_q < outer_a and cand_a <= outer_a:
                inner_pos = cand_pos
                break
        if inner_pos is None:
            cursor += 1
            continue
        inner_idx = rows[inner_pos][0]
        nested_meta[outer_idx] = (group_id, "outer")
        nested_meta[inner_idx] = (group_id, "inner")
        group_id += 1
        cursor = inner_pos + 1
    return nested_meta


def _slot_end_for_rows(rows: list[dict[str, Any]], idx: int, *, last_slot_tail_sec: float = 60.0) -> float:
    q_time = float(rows[idx]["q_time"])
    a_time = float(rows[idx]["a_time"])
    if idx + 1 < len(rows):
        return max(float(rows[idx + 1]["q_time"]), q_time)
    return max(a_time + float(last_slot_tail_sec), q_time)


def _streaming_slots_for_rows(rows: list[dict[str, Any]], scene_type: str) -> list[dict[str, Any]]:
    """Build enough official OmniInteract slot metadata for benchmark scoring."""

    if scene_type == "1QnA":
        slots: list[dict[str, Any]] = []
        q_start = float(rows[0]["q_time"]) if rows else 0.0
        question_text = str(rows[0].get("question_text") or "") if rows else ""
        for idx, row in enumerate(rows):
            t_a = float(row["a_time"])
            start = q_start if idx == 0 else t_a
            end = float(rows[idx + 1]["a_time"]) if idx + 1 < len(rows) else t_a + 60.0
            slots.append(
                {
                    "slot_id": idx + 1,
                    "start": start,
                    "t_a": t_a,
                    "end": max(end, start),
                    "boundary_type": "Soft",
                    "question_text": question_text,
                    "gt_answer": row["answer_text"],
                    "scene_type": "1QnA",
                    "step_index": idx + 1,
                    "turn_index": 1,
                    "question_type": row["question_type"],
                    "is_interrupted": bool(row["is_interrupted"]),
                    "label": row.get("label", ""),
                    "nested_group_id": None,
                    "nested_role": "",
                    "inferred_knowledge": row.get("inferred_knowledge", ""),
                }
            )
        return slots

    if scene_type == "nested":
        slots: list[dict[str, Any]] = []
        slot_id = 1
        group_id = 1
        idx = 0
        while idx < len(rows):
            outer = rows[idx]
            inner_idx = None
            for cand_idx in range(idx + 1, len(rows)):
                cand = rows[cand_idx]
                if outer["q_time"] < cand["q_time"] < outer["a_time"] and cand["a_time"] <= outer["a_time"]:
                    inner_idx = cand_idx
                    break
            if inner_idx is None:
                idx += 1
                continue
            inner = rows[inner_idx]
            next_outer_q = rows[inner_idx + 1]["q_time"] if inner_idx + 1 < len(rows) else outer["a_time"] + 60.0
            slots.append(
                {
                    "slot_id": slot_id,
                    "start": float(outer["q_time"]),
                    "t_a": float(outer["a_time"]),
                    "end": max(float(next_outer_q), float(outer["a_time"])),
                    "boundary_type": "Hard",
                    "question_text": outer["question_text"],
                    "gt_answer": outer["answer_text"],
                    "scene_type": "nested",
                    "turn_index": group_id,
                    "question_type": outer["question_type"],
                    "is_interrupted": bool(outer["is_interrupted"]),
                    "nested_group_id": group_id,
                    "nested_role": "outer",
                }
            )
            slot_id += 1
            slots.append(
                {
                    "slot_id": slot_id,
                    "start": float(inner["q_time"]),
                    "t_a": float(inner["a_time"]),
                    "end": float(outer["a_time"]),
                    "boundary_type": "Hard",
                    "question_text": inner["question_text"],
                    "gt_answer": inner["answer_text"],
                    "scene_type": "nested",
                    "turn_index": group_id,
                    "question_type": inner["question_type"],
                    "is_interrupted": bool(inner["is_interrupted"]),
                    "nested_group_id": group_id,
                    "nested_role": "inner",
                }
            )
            slot_id += 1
            group_id += 1
            idx = inner_idx + 1
        return slots

    return [
        {
            "slot_id": idx + 1,
            "start": float(row["q_time"]),
            "t_a": float(row["a_time"]),
            "end": _slot_end_for_rows(rows, idx),
            "boundary_type": "Hard",
            "question_text": row["question_text"],
            "gt_answer": row["answer_text"],
            "scene_type": scene_type,
            "turn_index": idx + 1,
            "question_type": row["question_type"],
            "is_interrupted": bool(row["is_interrupted"]),
            "nested_group_id": None,
            "nested_role": "",
        }
        for idx, row in enumerate(rows)
    ]


def _parse_1qna_rows(ann: Any) -> list[dict[str, Any]]:
    """Parse OmniInteract 1QnA annotations into answer-step rows."""

    if not isinstance(ann, dict):
        return []
    inferred = str(ann.get("inferred_knowledge", "") or "").strip()
    q_time = _parse_time_seconds(ann.get("question_time"))
    question_text = str(ann.get("question_text", "") or "").strip()
    rows: list[dict[str, Any]] = []

    conversations = ann.get("conversations")
    if isinstance(conversations, list):
        first_user = next(
            (
                item
                for item in conversations
                if isinstance(item, dict) and str(item.get("from", "")).strip().lower() == "user"
            ),
            None,
        )
        if first_user is not None:
            q_time = _parse_time_seconds(first_user.get("timestamp")) or q_time
            question_text = str(first_user.get("value", "") or question_text).strip()
        for idx, item in enumerate(conversations):
            if not isinstance(item, dict) or str(item.get("from", "")).strip().lower() != "assistant":
                continue
            a_time = _parse_time_seconds(item.get("timestamp"))
            answer = str(item.get("value", "") or "").strip()
            if a_time is None or not answer:
                continue
            rows.append(
                {
                    "qa_index": idx,
                    "q_time": float(q_time or 0.0),
                    "a_time": float(a_time),
                    "question_time": str(q_time or 0.0),
                    "answer_time": str(item.get("timestamp") or "").strip(),
                    "question_text": question_text,
                    "answer_text": answer,
                    "question_type": str(item.get("label", "") or "step").strip().lower() or "step",
                    "is_interrupted": bool(item.get("interrupted", item.get("is_interrupted", False))),
                    "label": str(item.get("label", "") or "").strip(),
                    "inferred_knowledge": inferred,
                }
            )
    elif isinstance(ann.get("answers"), list):
        for idx, item in enumerate(ann["answers"]):
            if not isinstance(item, dict):
                continue
            a_time = _parse_time_seconds(item.get("answer_time"))
            answer = str(item.get("answer_text", "") or "").strip()
            if a_time is None or not answer:
                continue
            rows.append(
                {
                    "qa_index": idx,
                    "q_time": float(q_time or 0.0),
                    "a_time": float(a_time),
                    "question_time": str(q_time or 0.0),
                    "answer_time": str(item.get("answer_time") or "").strip(),
                    "question_text": question_text,
                    "answer_text": answer,
                    "question_type": str(item.get("label", "") or "step").strip().lower() or "step",
                    "is_interrupted": bool(item.get("interrupted", item.get("is_interrupted", False))),
                    "label": str(item.get("label", "") or "").strip(),
                    "inferred_knowledge": inferred,
                }
            )
    rows.sort(key=lambda row: (float(row["a_time"]), int(row["qa_index"])))
    return rows


def _hf_cache_root() -> Path:
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")).expanduser().resolve()


def _tar_fingerprint(tar_path: Path) -> str:
    st = tar_path.stat()
    return f"v1:{st.st_size}:{int(st.st_mtime_ns)}"


def _resolve_data_dir_under(root: Path) -> Path:
    """Return the OmniInteract data root that contains 1q1a/1q1a_math/1qna."""
    probe = root / "1q1a"
    if probe.is_dir():
        return root
    probe = root / "data" / "1q1a"
    if probe.is_dir():
        return root / "data"
    raise FileNotFoundError(f"Could not locate OmniInteract data dir under: {root}")


def _extract_tar_archive(tar_path: Path, cache_root: Path) -> Path:
    """Extract ``data.tar.gz`` under ``cache_root`` and return the data root."""
    extracted_root = cache_root / "extracted"
    marker = cache_root / ".extracted"
    fp = _tar_fingerprint(tar_path)
    if marker.is_file() and extracted_root.is_dir():
        try:
            if marker.read_text(encoding="utf-8").strip() == fp:
                data_dir = _resolve_data_dir_under(extracted_root)
                logger.info("Reusing cached OmniInteract media at %s", data_dir)
                return data_dir
        except Exception:
            shutil.rmtree(extracted_root, ignore_errors=True)
            marker.unlink(missing_ok=True)

    cache_root.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(extracted_root, ignore_errors=True)
    extracted_root.mkdir(parents=True, exist_ok=True)
    logger.info("Extracting OmniInteract archive %s", tar_path)
    with tarfile.open(tar_path, "r:*") as tf:
        tf.extractall(path=extracted_root, filter="data")
    marker.write_text(fp, encoding="utf-8")
    return _resolve_data_dir_under(extracted_root)


def _ensure_extracted_data_dir(root: Path) -> Path:
    """Resolve a local directory to extracted OmniInteract data (auto-extract if needed)."""
    try:
        return _resolve_data_dir_under(root)
    except FileNotFoundError:
        pass

    for name in ("data.tar.gz", "data.tar"):
        tar_path = root / name
        if tar_path.is_file():
            cache_root = root / ".vllm_omni_omniinteract_extracted"
            return _extract_tar_archive(tar_path, cache_root)

    raise FileNotFoundError(
        f"Could not locate OmniInteract data under {root}. "
        "Expected extracted 1q1a/ (or data/1q1a/) or data.tar.gz in that directory."
    )


def resolve_omniinteract_root(
    dataset_path: str | None = None,
    *,
    explicit_root: str | Path | None = None,
) -> Path:
    """Return OmniInteract data root from a local path or HF dataset repo id.

    Resolution order:
    1. ``explicit_root`` (--omniinteract-root)
    2. Local directory ``dataset_path`` (--dataset-path)
    3. HF dataset repo id via ``dataset_path`` / ``--hf-name``
    """
    if explicit_root:
        root = Path(explicit_root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"--omniinteract-root is not a directory: {root}")
        return _ensure_extracted_data_dir(root)

    if not dataset_path:
        raise ValueError("OmniInteract requires --dataset-path (HF repo id or local directory) or --omniinteract-root.")

    p = Path(dataset_path).expanduser()
    if p.exists() and p.is_dir():
        return _ensure_extracted_data_dir(p.resolve())

    return ensure_omniinteract_data_dir(dataset_path.strip())


def ensure_omniinteract_data_dir(repo_id: str) -> Path:
    """Download/extract ``data.tar.gz`` from HF and return extracted data root."""
    rid = (repo_id or "").strip()
    if not rid:
        raise ValueError("repo_id is required for OmniInteract HF download")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError(
            "OmniInteract HF download requires huggingface_hub. "
            "Install it or pass --omniinteract-root with local extracted data."
        ) from e

    safe = rid.replace("/", "__").replace("\\", "_")
    cache_root = _hf_cache_root() / "vllm_omni" / "omniinteract_media" / safe

    tar_path: Path | None = None
    for name in ("data.tar.gz", "data.tar"):
        try:
            tar_path = Path(hf_hub_download(repo_id=rid, filename=name, repo_type="dataset"))
            break
        except Exception:
            continue
    if tar_path is None or not tar_path.is_file():
        raise FileNotFoundError(f"Could not download data.tar.gz from dataset repo {rid!r}.")

    return _extract_tar_archive(tar_path, cache_root)


class OmniInteractDataset(BenchmarkDataset):
    """OmniInteract audio+video QA dataset."""

    SUPPORTED_DATASET_PATHS: set[str] = {"lucky-lance/OmniInteract"}
    DEFAULT_HF_DATASET_ID = "lucky-lance/OmniInteract"
    DEFAULT_OUTPUT_LEN = 256
    IS_MULTIMODAL = True

    def __init__(
        self,
        dataset_path: str | None = None,
        random_seed: int = 0,
        data_root: str | None = None,
        subsets: list[OmniInteractSubset] | None = None,
        inline_local_video: bool = False,
        input_mode: OmniInteractInputMode = "video",
        aura_tts_task_type: str = "Base",
        aura_tts_language: str = "Chinese",
        aura_tts_speaker: str | None = None,
        aura_tts_ref_audio: str | None = None,
        aura_tts_ref_text: str | None = None,
        streaming_sample_fps: float = 2.0,
        streaming_send_fps: float = 2.0,
        streaming_max_frames: int = 0,
        streaming_auto_trigger_min_frames: int = 2,
        streaming_enable_frame_filter: bool = False,
        streaming_cross_turn_penalty: float = 0.0,
        streaming_cross_turn_lookback: int = 2,
        streaming_video_ids: list[str] | None = None,
        streaming_aura_system_prompt_mode: Literal["native", "omniinteract_qa"] = "native",
        **kwargs: Any,
    ) -> None:
        self.dataset_path = dataset_path or self.DEFAULT_HF_DATASET_ID
        self.data_root_input = Path(data_root).expanduser().resolve() if data_root else None
        self.subsets = list(subsets or ["1q1a", "1q1a_math", "1qna"])
        self.inline_local_video = inline_local_video
        self.input_mode = input_mode
        self.aura_tts_task_type = aura_tts_task_type
        self.aura_tts_language = aura_tts_language
        self.aura_tts_speaker = aura_tts_speaker
        self.aura_tts_ref_audio = self._normalize_aura_ref_audio(aura_tts_ref_audio)
        self.aura_tts_ref_text = aura_tts_ref_text
        self.streaming_sample_fps = streaming_sample_fps
        self.streaming_send_fps = streaming_send_fps
        self.streaming_max_frames = streaming_max_frames
        self.streaming_auto_trigger_min_frames = streaming_auto_trigger_min_frames
        self.streaming_enable_frame_filter = streaming_enable_frame_filter
        self.streaming_cross_turn_penalty = streaming_cross_turn_penalty
        self.streaming_cross_turn_lookback = streaming_cross_turn_lookback
        self.streaming_video_ids = [v.strip() for v in (streaming_video_ids or []) if v and v.strip()]
        self.streaming_aura_system_prompt_mode = streaming_aura_system_prompt_mode
        self._data_root: Path | None = None
        self._entries: list[_OmniInteractEntry] = []
        self._streaming_entries: list[_OmniInteractStreamingEntry] = []

        super().__init__(
            dataset_path=self.dataset_path,
            random_seed=random_seed,
            **kwargs,
        )
        self.load_data()

    def _resolve_data_root(self) -> Path:
        if self._data_root is not None:
            return self._data_root
        dataset_ref = None if self.data_root_input is not None else self.dataset_path
        self._data_root = resolve_omniinteract_root(
            dataset_ref,
            explicit_root=self.data_root_input,
        )
        return self._data_root

    @staticmethod
    def _read_json(path: Path) -> Any:
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def _iter_subset_entries(self, data_root: Path, subset: OmniInteractSubset) -> list[_OmniInteractEntry]:
        entries: list[_OmniInteractEntry] = []
        subset_root = data_root / subset
        if not subset_root.is_dir():
            logger.warning("Subset directory does not exist: %s", subset_root)
            return entries

        if subset in ("1q1a", "1q1a_math"):
            map_path = subset_root / "video_json_map.json"
            map_data = self._read_json(map_path) if map_path.is_file() else {"entries": []}
            for item in map_data.get("entries", []):
                video_rel = str(item.get("video") or "").strip()
                ann_rel = str(item.get("annotation") or "").strip()
                scene_type = str(item.get("scene_type") or "multi_turn").strip().lower() or "multi_turn"
                if not video_rel or not ann_rel:
                    continue
                ann_path = subset_root / ann_rel
                if not ann_path.is_file():
                    continue
                ann = self._read_json(ann_path)
                if not isinstance(ann, list):
                    continue
                nested_meta = _infer_nested_roles(ann) if scene_type == "nested" else {}
                video_stem = Path(video_rel).stem
                for qa_idx, qa in enumerate(ann):
                    q = str(qa.get("question_text") or "").strip()
                    a = str(qa.get("answer_text") or "").strip()
                    if not q or not a:
                        continue
                    subvideo_rel = f"subvideos/{video_stem}_{qa_idx}.mp4"
                    subvideo_path = subset_root / subvideo_rel
                    if not subvideo_path.is_file():
                        continue
                    audio_path = subset_root / "audios" / f"{video_stem}_{qa_idx}.wav"
                    nested_group_id, nested_role = nested_meta.get(qa_idx, (None, ""))
                    entries.append(
                        _OmniInteractEntry(
                            subset=subset,
                            video_rel=subvideo_rel,
                            video_path=subvideo_path,
                            question_text=q,
                            answer_text=a,
                            question_time=str(qa.get("question_time") or "").strip(),
                            answer_time=str(qa.get("answer_time") or "").strip(),
                            question_type=str(qa.get("question_type") or "").strip(),
                            is_interrupted=qa.get("is_interrupted"),
                            audio_path=audio_path,
                            scene_type=scene_type,
                            nested_group_id=nested_group_id,
                            nested_role=nested_role,
                        )
                    )
            return entries

        ann_root = subset_root / "annotations"
        video_root = subset_root / "videos_bench"
        if not ann_root.is_dir() or not video_root.is_dir():
            return entries
        for ann_path in sorted(ann_root.rglob("*.json")):
            rel = ann_path.relative_to(ann_root)
            video_path = (video_root / rel).with_suffix(".mp4")
            if not video_path.is_file():
                continue
            ann = self._read_json(ann_path)
            video_rel = str(video_path.relative_to(subset_root))
            rows = _parse_1qna_rows(ann)
            for row in rows:
                entries.append(
                    _OmniInteractEntry(
                        subset=subset,
                        video_rel=video_rel,
                        video_path=video_path,
                        question_text=str(row.get("question_text") or ""),
                        answer_text=str(row.get("answer_text") or ""),
                        question_time=str(row.get("question_time") or "").strip(),
                        answer_time=str(row.get("answer_time") or "").strip(),
                        question_type=str(row.get("question_type") or "").strip(),
                        is_interrupted=bool(row.get("is_interrupted")),
                        audio_path=None,
                        scene_type="1QnA",
                    )
                )
        return entries

    def _iter_subset_streaming_entries(
        self,
        data_root: Path,
        subset: OmniInteractSubset,
    ) -> list[_OmniInteractStreamingEntry]:
        entries: list[_OmniInteractStreamingEntry] = []
        subset_root = data_root / subset
        if not subset_root.is_dir():
            return entries
        if subset == "1qna":
            ann_root = subset_root / "annotations"
            video_root = subset_root / "videos_bench"
            if not ann_root.is_dir() or not video_root.is_dir():
                return entries
            for ann_path in sorted(ann_root.rglob("*.json")):
                rel = ann_path.relative_to(ann_root)
                video_path = (video_root / rel).with_suffix(".mp4")
                if not video_path.is_file():
                    continue
                rows = _parse_1qna_rows(self._read_json(ann_path))
                if not rows:
                    continue
                entries.append(
                    _OmniInteractStreamingEntry(
                        subset=subset,
                        video_rel=str(video_path.relative_to(subset_root)),
                        video_path=video_path,
                        annotation_path=ann_path,
                        scene_type="1QnA",
                        rows=rows,
                    )
                )
            return entries

        if subset not in ("1q1a", "1q1a_math"):
            return entries

        map_path = subset_root / "video_json_map.json"
        map_data = self._read_json(map_path) if map_path.is_file() else {"entries": []}
        for item in map_data.get("entries", []):
            video_rel = str(item.get("video") or "").strip()
            ann_rel = str(item.get("annotation") or "").strip()
            scene_type = str(item.get("scene_type") or "multi_turn").strip().lower() or "multi_turn"
            if not video_rel or not ann_rel:
                continue
            video_path = subset_root / video_rel
            ann_path = subset_root / ann_rel
            if not video_path.is_file() or not ann_path.is_file():
                continue
            ann = self._read_json(ann_path)
            if not isinstance(ann, list):
                continue
            rows: list[dict[str, Any]] = []
            video_stem = Path(video_rel).stem
            nested_meta = _infer_nested_roles(ann) if scene_type == "nested" else {}
            for qa_idx, qa in enumerate(ann):
                q_time = _parse_time_seconds(qa.get("question_time"))
                a_time = _parse_time_seconds(qa.get("answer_time"))
                q = str(qa.get("question_text") or "").strip()
                a = str(qa.get("answer_text") or "").strip()
                audio_path = subset_root / "audios" / f"{video_stem}_{qa_idx}.wav"
                if q_time is None or a_time is None or not q or not a or not audio_path.is_file():
                    continue
                nested_group_id, nested_role = nested_meta.get(qa_idx, (None, ""))
                rows.append(
                    {
                        "qa_index": qa_idx,
                        "q_time": float(q_time),
                        "a_time": float(a_time),
                        "question_time": str(qa.get("question_time") or "").strip(),
                        "answer_time": str(qa.get("answer_time") or "").strip(),
                        "question_text": q,
                        "answer_text": a,
                        "question_type": str(qa.get("question_type") or "").strip().lower() or "unknown",
                        "is_interrupted": bool(qa.get("is_interrupted")),
                        "audio_path": str(audio_path.expanduser().resolve()),
                        "nested_group_id": nested_group_id,
                        "nested_role": nested_role,
                    }
                )
            if not rows:
                continue
            rows.sort(key=lambda row: (float(row["q_time"]), float(row["a_time"]), int(row["qa_index"])))
            entries.append(
                _OmniInteractStreamingEntry(
                    subset=subset,
                    video_rel=video_rel,
                    video_path=video_path,
                    annotation_path=ann_path,
                    scene_type="nested" if scene_type == "nested" else "multi_turn",
                    rows=rows,
                )
            )
        return entries

    def load_data(self) -> None:
        root = self._resolve_data_root()
        all_entries: list[_OmniInteractEntry] = []
        all_streaming_entries: list[_OmniInteractStreamingEntry] = []
        for subset in self.subsets:
            if self.input_mode == "aura_streaming":
                all_streaming_entries.extend(self._iter_subset_streaming_entries(root, subset))
            else:
                all_entries.extend(self._iter_subset_entries(root, subset))
        if self.input_mode == "aura_streaming":
            if not all_streaming_entries:
                raise ValueError(f"No OmniInteract streaming videos found under {root} (subsets={self.subsets})")
        elif not all_entries:
            raise ValueError(f"No OmniInteract QA entries found under {root} (subsets={self.subsets})")
        if self.input_mode == "aura_streaming" and self.streaming_video_ids:
            wanted = set(self.streaming_video_ids)
            filtered = [
                entry
                for entry in all_streaming_entries
                if Path(entry.video_rel).stem in wanted
            ]
            order = {vid: idx for idx, vid in enumerate(self.streaming_video_ids)}
            filtered.sort(key=lambda entry: order.get(Path(entry.video_rel).stem, 10_000))
            missing = [vid for vid in self.streaming_video_ids if vid not in {Path(e.video_rel).stem for e in filtered}]
            if missing:
                raise ValueError(
                    f"OmniInteract streaming video id(s) not found under {root}: {', '.join(missing)}"
                )
            all_streaming_entries = filtered
        if not getattr(self, "disable_shuffle", False) and not self.streaming_video_ids:
            import random

            rng = random.Random(self.random_seed)
            rng.shuffle(all_streaming_entries if self.input_mode == "aura_streaming" else all_entries)
        self._entries = all_entries
        self._streaming_entries = all_streaming_entries
        self.data = self._streaming_entries if self.input_mode == "aura_streaming" else self._entries
        logger.info("Loaded OmniInteract: root=%s subsets=%s rows=%d", root, self.subsets, len(self.data))

    @staticmethod
    def _question_prompt(e: _OmniInteractEntry) -> str:
        return (
            "You are given a video with its original audio. "
            "Answer the question concisely based on the observed visual and spoken content.\n"
            f"Question time: {e.question_time or 'N/A'}\n"
            f"Expected answer time: {e.answer_time or 'N/A'}\n"
            f"Question: {e.question_text}\n"
            "Answer:"
        )

    def _video_payload(self, video_path: Path) -> dict[str, Any]:
        p = video_path.expanduser().resolve()
        if self.inline_local_video:
            raw = p.read_bytes()
            b64 = base64.b64encode(raw).decode("ascii")
            return {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{b64}"}}
        return {"type": "video_url", "video_url": {"url": p.as_uri()}}

    def _audio_payload(self, audio_path: Path) -> dict[str, Any]:
        p = audio_path.expanduser().resolve()
        if self.inline_local_video:
            raw = p.read_bytes()
            b64 = base64.b64encode(raw).decode("ascii")
            return {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{b64}"}}
        return {"type": "audio_url", "audio_url": {"url": p.as_uri()}}

    @staticmethod
    def _normalize_aura_ref_audio(ref_audio: str | None) -> str | None:
        if ref_audio is None:
            return None
        value = ref_audio.strip()
        if not value:
            return None
        if _is_remote_or_data_url(value):
            return value
        path = Path(value).expanduser()
        if not path.is_file():
            raise ValueError(f"--omniinteract-aura-tts-ref-audio does not exist: {value}")
        return str(path.resolve())

    def _build_messages(
        self,
        e: _OmniInteractEntry,
        video_payload: dict[str, Any],
        audio_payload: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if audio_payload is not None and self.input_mode == "aura":
            content: list[dict[str, Any]] = [audio_payload, video_payload]
        else:
            content = [video_payload, {"type": "text", "text": self._question_prompt(e)}]
            if audio_payload is not None:
                content = [audio_payload, *content]
        return [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a helpful multimodal assistant that understands video and audio.",
                    }
                ],
            },
            {
                "role": "user",
                "content": content,
            },
        ]

    def sample(
        self,
        tokenizer: TokenizerLike,
        num_requests: int,
        output_len: int | None = None,
        request_id_prefix: str = "",
        no_oversample: bool = False,
        **kwargs: Any,
    ) -> list[SampleRequest]:
        if output_len is None:
            output_len = self.DEFAULT_OUTPUT_LEN
        out: list[SampleRequest] = []
        tok = get_cached_tokenizer(tokenizer)

        if self.input_mode == "aura_streaming":
            for i, entry in enumerate(self._streaming_entries):
                if len(out) >= num_requests:
                    break
                slots = _streaming_slots_for_rows(entry.rows, entry.scene_type)
                if not slots:
                    continue
                prompt = "\n".join(row["question_text"] for row in entry.rows)
                config = aura_streaming_config(
                    tts_task_type=self.aura_tts_task_type,
                    tts_language=self.aura_tts_language,
                    tts_speaker=self.aura_tts_speaker,
                    tts_ref_audio=self.aura_tts_ref_audio,
                    tts_ref_text=self.aura_tts_ref_text,
                    sample_fps=self.streaming_sample_fps,
                    send_fps=self.streaming_send_fps,
                    max_frames=self.streaming_max_frames,
                    auto_trigger_min_frames=self.streaming_auto_trigger_min_frames,
                    enable_frame_filter=self.streaming_enable_frame_filter,
                    cross_turn_penalty=self.streaming_cross_turn_penalty,
                    cross_turn_lookback=self.streaming_cross_turn_lookback,
                    aura_system_prompt_mode=self.streaming_aura_system_prompt_mode,
                )
                out.append(
                    OmniInteractSampleRequest(
                        prompt=prompt,
                        prompt_len=len(tok.encode(prompt)),
                        expected_output_len=output_len,
                        multi_modal_data=None,
                        request_id=f"{request_id_prefix}{i}",
                        omniinteract_gold_answer="\n".join(row["answer_text"] for row in entry.rows),
                        omniinteract_subset=entry.subset,
                        omniinteract_question_type="streaming",
                        omniinteract_video=entry.video_rel,
                        omniinteract_scene_type=entry.scene_type,
                        omni_extra_body=None,
                        omni_chat_messages=None,
                        omniinteract_streaming_video_path=str(entry.video_path.expanduser().resolve()),
                        omniinteract_streaming_audio_schedule=[
                            {
                                "at_sec": float(row["q_time"]),
                                "audio_path": row["audio_path"],
                                "qa_index": int(row["qa_index"]),
                                "question_text": row["question_text"],
                            }
                            for row in entry.rows
                            if row.get("audio_path")
                        ],
                        omniinteract_streaming_audio_from_video=entry.subset == "1qna",
                        omniinteract_streaming_slots=slots,
                        omniinteract_streaming_config=config,
                    )
                )
            self.maybe_oversample_requests(out, num_requests, request_id_prefix, no_oversample)
            return out

        for i, entry in enumerate(self._entries):
            if len(out) >= num_requests:
                break
            if not entry.video_path.is_file():
                continue
            if self.input_mode in ("aura", "aura_streaming"):
                prompt = entry.question_text
            else:
                prompt = self._question_prompt(entry)
            payload = self._video_payload(entry.video_path)
            audio_payload = None
            extra_body = {"mm_processor_kwargs": {"use_audio_in_video": True}}
            streaming_config = None
            streaming_video_path = ""
            streaming_audio_path = ""
            if self.input_mode in ("aura", "aura_streaming"):
                if entry.audio_path is None or not entry.audio_path.is_file():
                    logger.warning(
                        "Skipping OmniInteract row without synthesized audio for AURA mode: subset=%s video=%s audio=%s",
                        entry.subset,
                        entry.video_rel,
                        entry.audio_path,
                    )
                    continue
                if self.input_mode == "aura_streaming":
                    extra_body = None
                    streaming_video_path = str(entry.video_path.expanduser().resolve())
                    streaming_audio_path = str(entry.audio_path.expanduser().resolve())
                    streaming_config = aura_streaming_config(
                        tts_task_type=self.aura_tts_task_type,
                        tts_language=self.aura_tts_language,
                        tts_speaker=self.aura_tts_speaker,
                        tts_ref_audio=self.aura_tts_ref_audio,
                        tts_ref_text=self.aura_tts_ref_text,
                        sample_fps=self.streaming_sample_fps,
                        send_fps=self.streaming_send_fps,
                        max_frames=self.streaming_max_frames,
                        auto_trigger_min_frames=self.streaming_auto_trigger_min_frames,
                        enable_frame_filter=self.streaming_enable_frame_filter,
                        cross_turn_penalty=self.streaming_cross_turn_penalty,
                        cross_turn_lookback=self.streaming_cross_turn_lookback,
                        aura_system_prompt_mode=self.streaming_aura_system_prompt_mode,
                    )
                else:
                    audio_payload = self._audio_payload(entry.audio_path)
                    extra_body = aura_extra_body(
                        tts_task_type=self.aura_tts_task_type,
                        tts_language=self.aura_tts_language,
                        tts_speaker=self.aura_tts_speaker,
                        tts_ref_audio=self.aura_tts_ref_audio,
                        tts_ref_text=self.aura_tts_ref_text,
                    )
            messages = self._build_messages(entry, payload, audio_payload)
            prompt_len = len(tok.encode(prompt))
            out.append(
                OmniInteractSampleRequest(
                    prompt=prompt,
                    prompt_len=prompt_len,
                    expected_output_len=output_len,
                    multi_modal_data=None,
                    request_id=f"{request_id_prefix}{i}",
                    omniinteract_gold_answer=entry.answer_text,
                    omniinteract_subset=entry.subset,
                    omniinteract_question_type=entry.question_type,
                    omniinteract_video=entry.video_rel,
                    omniinteract_question_time=entry.question_time,
                    omniinteract_answer_time=entry.answer_time,
                    omniinteract_is_interrupted=entry.is_interrupted,
                    omniinteract_scene_type=entry.scene_type,
                    omniinteract_nested_group_id=entry.nested_group_id,
                    omniinteract_nested_role=entry.nested_role,
                    omni_extra_body=extra_body,
                    omni_chat_messages=None if self.input_mode == "aura_streaming" else messages,
                    omniinteract_streaming_video_path=streaming_video_path,
                    omniinteract_streaming_audio_path=streaming_audio_path,
                    omniinteract_streaming_config=streaming_config,
                )
            )

        self.maybe_oversample_requests(out, num_requests, request_id_prefix, no_oversample)
        return out
