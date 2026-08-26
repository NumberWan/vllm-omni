# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Streaming video WebSocket handlers for multi-model pipelines.

Shared protocol (see :mod:`video_stream_base`):
    Client -> Server:
        {"type": "session.config", ...}         # Session config (sent once)
        {"type": "video.frame", "data": "..."}  # base64 JPEG/PNG frame
        {"type": "audio.chunk", "data": "..."}  # base64 PCM16 16kHz mono
        {"type": "audio.done"}                  # End of utterance (alias: audio.commit)
        {"type": "video.query", "text": "..."}  # Submit query about buffered frames
        {"type": "video.done"}                  # End of session

    Server -> Client:
        Response events include ``request_id`` so overlapping AURA turns can be
        attributed correctly.
        {"type": "response.start", "request_id": "..."}
        {"type": "user.transcript.done", "text": "...", "request_id": "..."}
        {"type": "response.text.delta", "delta": "...", "request_id": "..."}
        {"type": "response.text.done", "text": "...", "request_id": "..."}
        {"type": "response.audio.delta", "data": "...", "format": "wav", "request_id": "..."}
        {"type": "response.audio.done", "request_id": "..."}
        {"type": "response.tool.preamble.text", "text": "...", "request_id": "..."}
        {"type": "response.tool.preamble.audio.delta", "data": "...", "format": "wav", "request_id": "..."}
        {"type": "response.tool.preamble.audio.done", "request_id": "..."}
        {"type": "response.tool.started", "call_id": "...", "name": "...", "request_id": "..."}
        {"type": "response.tool.done", "call_id": "...", "name": "...", "status": "...", "content": "...", "request_id": "..."}
        {"type": "session.done"}
        {"type": "error", "message": "..."}

Model-specific handlers:
    :class:`QwenOmniStreamingVideoHandler` — Qwen3-Omni (thinker -> talker -> code2wav);
        turns start on ``video.query`` only.
    :class:`AuraStreamingVideoHandler` — AURA Omni (ASR -> AURA -> TTS -> code2wav);
        auto-trigger on buffered frames; ``audio.done`` opens a turn after a full
        utterance when unlocked; ``video.query`` is ignored.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image
from pydantic import Field
from vllm.logger import init_logger

from vllm_omni.entrypoints.openai.aura_tool_executor import (
    AuraToolCall,
    AuraToolResult,
    ParsedAuraToolTurn,
    aura_any_tool_intent,
    parse_aura_tool_output,
)
from vllm_omni.entrypoints.openai.video_stream_base import (
    _BAD_FRAME,
    _DEFAULT_CONFIG_TIMEOUT,
    _DEFAULT_IDLE_TIMEOUT,
    StreamingVideoSessionConfig,
    VideoStreamTurnTrigger,
    _decode_frame_bytes,
    _downscale_frame_max_edge,
)
from vllm_omni.entrypoints.openai.video_stream_base import (
    OmniStreamingVideoHandler as OmniStreamingVideoHandlerBase,
)
from vllm_omni.model_executor.stage_input_processors.aura_cross_turn_penalty import (
    CrossTurnPenalty,
    merge_penalty_sampling_params,
)
from vllm_omni.model_executor.stage_input_processors.aura_omni import (
    _clean_asr_transcript,
    _clean_tts_text,
    _extract_text,
    _trim_aura_response_token_ids,
    build_aura_streaming_turn_additional_information,
    frames_to_video_tuple,
)
from vllm_omni.model_executor.stage_input_processors.aura_session_history import (
    DEFAULT_AURA_SYSTEM_PROMPT,
    SILENT_TEXT,
    AuraSessionState,
    create_streaming_session,
    has_pending_turn,
    record_pending_turn,
    should_stop_aura_silent_generation,
    unregister_session,
)
from vllm_omni.model_executor.stage_input_processors.aura_tool_protocol import (
    aura_natural_content,
    extract_aura_tool_preamble,
)

__all__ = [
    "AuraSessionState",
    "AuraStreamingVideoHandler",
    "AuraStreamingVideoSessionConfig",
    "QwenOmniStreamingVideoHandler",
    "StreamingVideoSessionConfig",
    "create_streaming_video_handler",
]

logger = init_logger(__name__)

_AURA_PIPELINE_NAMES = frozenset({"aura_omni"})
_AURA_ADDITIONAL_INFO_KEY = "_aura_additional_information"


def decode_aura_content_ids_for_display(tokenizer: Any, token_ids: list[int] | None) -> str:
    """Full-sequence decode of trimmed Stage-1 ids (same source TTS uses)."""
    content_ids = _trim_aura_response_token_ids(list(token_ids or []))
    if not content_ids or tokenizer is None:
        return ""
    decode = getattr(tokenizer, "decode", None)
    if not callable(decode):
        return ""
    try:
        decoded = decode(content_ids)
    except Exception:
        return ""
    return aura_natural_content(_clean_tts_text(decoded)).strip()


def _cjk_char_count(text: str) -> int:
    return sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")


def _latin_alnum_count(text: str) -> int:
    return sum(1 for ch in text if ("a" <= ch <= "z") or ("A" <= ch <= "Z"))


def canonical_aura_response_text(*, streaming_text: str, decoded_text: str) -> str:
    """Prefer Stage-1 full decode when it refines streaming (e.g. 30→3000).

    Never replace good CJK streaming text with a wrong-vocab decode (ASR
    tokenizer on AURA ids yields Latin/CJK soup such as ``Trilogy-même…``).
    """
    if streaming_text == SILENT_TEXT:
        return SILENT_TEXT
    if not decoded_text:
        return streaming_text or ""
    if not streaming_text:
        return decoded_text
    if decoded_text == streaming_text:
        return decoded_text

    stream_cjk = _cjk_char_count(streaming_text)
    decoded_cjk = _cjk_char_count(decoded_text)
    decoded_latin = _latin_alnum_count(decoded_text)
    # Wrong tokenizer: streaming reads as Chinese speech, decode is Latin soup.
    if stream_cjk >= 2 and decoded_latin >= max(8, decoded_cjk):
        return streaming_text
    return decoded_text


def _resolve_deploy_pipeline(engine_client: Any) -> str | None:
    """Read ``pipeline:`` from the engine deploy YAML (entrypoints-only; no engine field)."""
    config_path = getattr(engine_client, "config_path", None)
    if config_path is None:
        return None

    from vllm_omni.config.stage_config import _DEPLOY_DIR, load_deploy_config

    path = Path(config_path)
    if not path.exists():
        if path.parent != Path("."):
            return None
        bare_name = path.name if path.name.endswith(".yaml") else f"{path.name}.yaml"
        candidate = _DEPLOY_DIR / bare_name
        if not candidate.exists():
            return None
        path = candidate
    pipeline = load_deploy_config(path).pipeline
    return str(pipeline) if pipeline else None


def _resolve_aura_stage1_model(engine_client: Any) -> str | None:
    """Resolve the server-owned Stage1 tokenizer path; never trust client input."""

    def _stage_model(stage: Any) -> str | None:
        if isinstance(stage, dict):
            model = stage.get("model")
            engine_values = stage.get("engine_extras") or stage.get("yaml_engine_args") or {}
        else:
            model = getattr(stage, "model", None)
            engine_values = (
                getattr(stage, "engine_extras", None)
                or getattr(stage, "yaml_engine_args", None)
                or {}
            )
        if not model and isinstance(engine_values, dict):
            model = engine_values.get("model")
        return str(model) if model else None

    stages = getattr(engine_client, "stage_configs", None)
    if stages:
        for stage in stages:
            stage_id = stage.get("stage_id") if isinstance(stage, dict) else getattr(stage, "stage_id", None)
            if stage_id == 1:
                model = _stage_model(stage)
                if model:
                    return model

    config_path = getattr(engine_client, "config_path", None)
    if config_path is None:
        return None
    from vllm_omni.config.stage_config import _DEPLOY_DIR, load_deploy_config

    path = Path(config_path)
    if not path.exists():
        bare_name = path.name if path.name.endswith(".yaml") else f"{path.name}.yaml"
        path = _DEPLOY_DIR / bare_name
    if not path.exists():
        return None
    deploy = load_deploy_config(path)
    for stage in deploy.stages:
        if stage.stage_id == 1:
            return _stage_model(stage)
    return None


class QwenOmniStreamingVideoHandler(OmniStreamingVideoHandlerBase):
    """Qwen-Omni pipeline: manual ``video.query`` trigger and image_pil prompts."""

    def should_trigger_turn(self, trigger: VideoStreamTurnTrigger) -> bool:
        return False

    def build_engine_prompt(
        self,
        config: StreamingVideoSessionConfig,
        frame_buffer: list[str],
        audio_buffer: bytearray,
        message_history: list[dict[str, Any]],
        query_text: str,
        prewarmed_frames: dict[str, tuple[Any, str]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        n_buf = len(frame_buffer)
        if n_buf <= config.num_frames:
            frames = list(frame_buffer)
        else:
            stride = max(1, n_buf // config.num_frames)
            idx = [i * stride for i in range(config.num_frames - 1)] + [n_buf - 1]
            frames = [frame_buffer[i] for i in idx]

        prewarmed = prewarmed_frames or {}
        user_content: list[dict] = []
        for frame_b64 in frames:
            cached = prewarmed.get(frame_b64)
            if cached is _BAD_FRAME:
                continue
            if cached is not None:
                pil, pil_uuid = cached
                user_content.append(
                    {
                        "type": "image_pil",
                        "image_pil": pil,
                        "uuid": pil_uuid,
                    }
                )
            else:
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"},
                    }
                )

        if len(audio_buffer) > 0:
            wav_b64 = self._pcm_to_wav_b64(bytes(audio_buffer))
            user_content.append(
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": wav_b64,
                        "format": "wav",
                    },
                }
            )

        if query_text:
            user_content.append({"type": "text", "text": query_text})

        user_message: dict[str, Any] = {"role": "user", "content": user_content}

        messages: list[dict[str, Any]] = []
        if config.system_prompt:
            messages.append({"role": "system", "content": config.system_prompt})

        recent_history = message_history[-2:] if len(message_history) > 2 else message_history
        for hist_msg in recent_history:
            messages.append(self._text_only_message(hist_msg))

        messages.append(user_message)

        return messages, user_message

    def on_turn_complete(
        self,
        message_history: list[dict[str, Any]],
        user_message: dict[str, Any],
        response_text: str,
        request_id: str | None = None,
    ) -> None:
        del request_id
        message_history.append(user_message)
        message_history.append({"role": "assistant", "content": response_text})

    _build_messages = build_engine_prompt


class AuraStreamingVideoSessionConfig(StreamingVideoSessionConfig):
    """Session config for AURA streaming video."""

    auto_trigger: bool = Field(default=True, description="Auto-start a turn after enough frames.")
    auto_trigger_min_frames: int = Field(default=2, ge=1, description="Minimum buffered frames to auto-trigger.")
    max_frames_per_round: int = Field(default=16, ge=2, description="Max frames per video_tuple.")
    max_frame_edge: int = Field(
        default=640,
        description=(
            "After JPEG decode, downscale so max(H, W) <= this edge before buffering "
            "(aligned with native AURA client max_frame_edge=640). "
            "Set <=0 to keep full decoded resolution."
        ),
    )
    pruning_enabled: bool = Field(default=True, description="Enable SessionHistory pruning.")
    max_rounds: int = Field(default=45, ge=1, description="Sliding-window round limit before pruning.")
    num_rounds_keep: int = Field(default=30, ge=1, description="Rounds to keep in sliding window after pruning.")
    max_context_qas: int = Field(default=10, ge=1, description="Max QAs in compressed context history.")
    max_1qna_rounds: int = Field(default=4, ge=1, description="Max rounds per 1QNA context-history QA.")
    aura_system_prompt: str | None = Field(default=None, description="Override AURA system prompt.")
    video_fps: float = Field(default=2.0, gt=0.0, description="FPS metadata for video_tuple.")
    cross_turn_penalty: float = Field(
        default=1.0,
        ge=0.0,
        description="Cross-turn repetition penalty strength (0=disabled, 2.0–3.0 recommended).",
    )
    cross_turn_lookback: int = Field(
        default=10,
        ge=1,
        description="Number of recent assistant responses for cross-turn penalty window.",
    )
    cross_turn_ngram_sizes: list[int] = Field(
        default_factory=lambda: [3, 4, 5],
        description="N-gram sizes for bad_words hard blocking in cross-turn penalty.",
    )
    stream_text_deltas: bool = Field(
        default=False,
        description=(
            "When false (default), accumulate assistant text server-side and only "
            "emit response.text.done to the client. Audio streaming is unaffected."
        ),
    )
    tts_task_type: str | None = Field(default=None, description="Qwen3-TTS task type override.")
    tts_language: str | None = Field(default=None, description="Qwen3-TTS language override.")
    tts_speaker: str | None = Field(default=None, description="CustomVoice speaker name.")
    tts_ref_audio: str | None = Field(default=None, description="Base TTS reference audio path.")
    tts_ref_text: str | None = Field(default=None, description="Base TTS reference transcript.")
    tts_instruct: str | None = Field(default=None, description="VoiceDesign / style instruct text.")
    tts_max_new_tokens: int | None = Field(
        default=None,
        ge=1,
        description="Max Qwen3-TTS codec tokens per spoken turn.",
    )
    tts_pass_token_ids: bool | None = Field(
        default=None,
        description="Pass AURA assistant token ids directly to Qwen3-TTS.",
    )
    tool_mode: Literal["none", "auto"] = Field(
        default="none",
        description="Server-side AURA_v2 tool loop; disabled unless the server has an allowlist executor.",
    )
    max_tool_depth: int = Field(
        default=3,
        ge=1,
        le=3,
        description="Maximum tool-call passes in one logical turn.",
    )
    tool_intent_gate: bool = Field(
        default=True,
        description=(
            "Start without a tool template and enable tools only when the user's "
            "transcript matches a tool domain. Once tools are enabled, selected "
            "tools execute like Native (no per-tool name re-check)."
        ),
    )

    def session_history_kwargs(self) -> dict[str, Any]:
        return {
            "max_rounds": self.max_rounds,
            "num_rounds_keep": self.num_rounds_keep,
            "pruning_enabled": self.pruning_enabled,
            "max_context_qas": self.max_context_qas,
            "max_1qna_rounds": self.max_1qna_rounds,
            "system_prompt": self.aura_system_prompt or DEFAULT_AURA_SYSTEM_PROMPT,
        }

    def tts_kwargs(self) -> dict[str, Any]:
        return {
            "tts_task_type": self.tts_task_type,
            "tts_language": self.tts_language,
            "tts_speaker": self.tts_speaker,
            "tts_ref_audio": self.tts_ref_audio,
            "tts_ref_text": self.tts_ref_text,
            "tts_instruct": self.tts_instruct,
            "tts_max_new_tokens": self.tts_max_new_tokens,
            "tts_pass_token_ids": self.tts_pass_token_ids,
        }

    def cross_turn_penalty_kwargs(self) -> dict[str, Any]:
        return {
            "window": self.cross_turn_lookback,
            "logit_penalty": self.cross_turn_penalty,
            "ngram_sizes": self.cross_turn_ngram_sizes,
        }


class AuraStreamingVideoHandler(OmniStreamingVideoHandlerBase):
    """AURA pipeline: frame auto-trigger + voice turn after complete ``audio.done``."""

    def __init__(
        self,
        *args: Any,
        tool_executor: Any | None = None,
        tool_tokenizer_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._tool_executor = tool_executor
        self._tool_tokenizer_path = tool_tokenizer_path
        self._aura_stage1_tokenizer: Any | None = None
        self._aura_stage1_tokenizer_loaded = False

    async def _get_aura_stage1_tokenizer(self) -> Any | None:
        """Tokenizer for Stage-1 AURA ids — not ``get_tokenizer()`` (often ASR)."""
        if self._aura_stage1_tokenizer_loaded:
            return self._aura_stage1_tokenizer
        self._aura_stage1_tokenizer_loaded = True
        engine = getattr(self._engine_client, "engine", None)
        processors = getattr(engine, "output_processors", None) if engine is not None else None
        if isinstance(processors, list) and len(processors) > 1:
            tok = getattr(processors[1], "tokenizer", None)
            if tok is not None:
                self._aura_stage1_tokenizer = tok
                return tok
        model = _resolve_aura_stage1_model(self._engine_client)
        if not model:
            return None
        try:
            from transformers import AutoTokenizer

            self._aura_stage1_tokenizer = AutoTokenizer.from_pretrained(
                model,
                trust_remote_code=True,
            )
        except Exception:
            logger.exception("Failed to load Stage-1 tokenizer from %s", model)
            self._aura_stage1_tokenizer = None
        return self._aura_stage1_tokenizer

    def supports_manual_query_turn(self) -> bool:
        return False

    def supports_query_interrupt(self) -> bool:
        return False

    def releases_turn_after_text_done(self) -> bool:
        # Allow the next vision turn after assistant text finishes while TTS may
        # still be draining. Combined with freeze_turn_video / commit_turn frame
        # retention, proactive "object appeared" frames during spoken TTS are kept.
        return True

    def should_trigger_on_audio_done(self, *, has_audio: bool, is_turn_locked: bool) -> bool:
        """Open a turn only after the full utterance is buffered (not per chunk)."""
        return bool(has_audio) and not is_turn_locked

    def ensure_frames_for_audio_turn(
        self,
        frame_buffer: list[str],
        message_history: Any,
        config: StreamingVideoSessionConfig,
    ) -> bool:
        """Seed turn frames from the most recent session buffer frame if needed."""
        if isinstance(message_history, AuraSessionState) and message_history.turn_frame_arrays:
            return True
        if not frame_buffer:
            return False
        last_b64 = frame_buffer[-1]
        try:
            raw_bytes = base64.b64decode(last_b64, validate=True)
        except Exception:
            return False
        self.on_frame_buffered(raw_bytes, last_b64, message_history, config)
        if isinstance(message_history, AuraSessionState):
            return bool(message_history.turn_frame_arrays)
        return True

    def create_message_history(self, config: StreamingVideoSessionConfig) -> AuraSessionState:
        aura_config = self._as_aura_config(config)
        return create_streaming_session(**aura_config.session_history_kwargs())

    def on_session_end(self, message_history: Any) -> None:
        if isinstance(message_history, AuraSessionState) and message_history.session_id:
            if self._tool_executor is not None:
                self._tool_executor.clear_session(message_history.session_id)
            unregister_session(message_history.session_id)

    def should_trigger_turn(self, trigger: VideoStreamTurnTrigger) -> bool:
        config = self._as_aura_config(trigger.config)
        if not config.auto_trigger:
            return False
        # When text.done releases the turn lock, ``is_turn_locked`` is the gate
        # (TTS may still be draining so ``is_generating`` stays true).
        if self.releases_turn_after_text_done():
            if trigger.is_turn_locked:
                return False
        elif trigger.is_generating:
            return False
        return trigger.frame_count >= config.auto_trigger_min_frames and not trigger.is_turn_locked

    def auto_trigger_frame_count(
        self,
        frame_buffer: list[str],
        message_history: Any,
    ) -> int:
        del frame_buffer
        if isinstance(message_history, AuraSessionState):
            return len(message_history.turn_frame_arrays)
        return 0

    def on_frame_buffered(
        self,
        raw_bytes: bytes,
        frame_b64: str,
        message_history: Any,
        config: StreamingVideoSessionConfig,
    ) -> None:
        del frame_b64
        if not isinstance(message_history, AuraSessionState):
            return
        frame = _decode_frame_bytes(raw_bytes)
        aura_config = self._as_aura_config(config)
        frame = _downscale_frame_max_edge(frame, aura_config.max_frame_edge)
        message_history.append_turn_frame(frame)

    def build_engine_prompt(
        self,
        config: StreamingVideoSessionConfig,
        frame_buffer: list[str],
        audio_buffer: bytearray,
        message_history: Any,
        query_text: str,
        prewarmed_frames: dict[str, tuple[Any, str]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        del frame_buffer, prewarmed_frames
        aura_config = self._as_aura_config(config)
        if not isinstance(message_history, AuraSessionState):
            raise TypeError("AURA streaming requires AuraSessionState message history")

        frames = list(message_history.turn_frame_arrays)
        video_array, metadata = frames_to_video_tuple(
            frames,
            fps=aura_config.video_fps,
            max_frames=aura_config.max_frames_per_round,
        )
        # Vision debug: dump the exact ndarray last frame that enters Stage-1.
        if os.environ.get("AURA_STAGE1_FRAME_DUMP", "1").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            try:
                dump_dir = Path(
                    os.environ.get(
                        "AURA_STAGE1_FRAME_DUMP_DIR",
                        "/tmp/aura_v2_native_demo/stage1_frames",
                    )
                )
                dump_dir.mkdir(parents=True, exist_ok=True)
                last = np.asarray(video_array[-1])
                if last.ndim == 3 and last.shape[-1] == 3:
                    stamp = time.strftime("%H%M%S")
                    out = dump_dir / (
                        f"stage1_{stamp}_n{len(video_array)}_"
                        f"{last.shape[1]}x{last.shape[0]}_"
                        f"{'voice' if len(audio_buffer) else 'silent'}.jpg"
                    )
                    Image.fromarray(last.astype("uint8", copy=False)).save(out, quality=90)
                    logger.info(
                        "AURA stage1 frame dump %s shape=%s voice=%s",
                        out,
                        tuple(video_array.shape),
                        bool(audio_buffer),
                    )
            except Exception as exc:
                logger.warning("AURA stage1 frame dump failed: %s", exc)

        user_content: list[dict[str, Any]] = []
        if len(audio_buffer) > 0:
            wav_b64 = self._pcm_to_wav_b64(bytes(audio_buffer))
            user_content.append(
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": wav_b64,
                        "format": "wav",
                    },
                }
            )
        if query_text:
            user_content.append({"type": "text", "text": query_text})

        user_message: dict[str, Any] = {"role": "user", "content": user_content}
        messages = [user_message]

        system_prompt = aura_config.aura_system_prompt or DEFAULT_AURA_SYSTEM_PROMPT
        additional_information = build_aura_streaming_turn_additional_information(
            session_id=message_history.session_id,
            video_array=video_array,
            video_metadata=metadata,
            system_prompt=system_prompt,
            skip_asr=len(audio_buffer) == 0,
            include_tts="audio" in aura_config.modalities,
            max_rounds=aura_config.max_rounds,
            num_rounds_keep=aura_config.num_rounds_keep,
            pruning_enabled=aura_config.pruning_enabled,
            max_context_qas=aura_config.max_context_qas,
            max_1qna_rounds=aura_config.max_1qna_rounds,
            tool_enabled=aura_config.tool_mode == "auto",
            tools=self._tool_executor.tool_schemas if aura_config.tool_mode == "auto" else None,
            tool_tokenizer_path=self._tool_tokenizer_path,
            **aura_config.tts_kwargs(),
        )
        user_message[_AURA_ADDITIONAL_INFO_KEY] = additional_information

        return messages, user_message

    def on_turn_complete(
        self,
        message_history: Any,
        user_message: dict[str, Any],
        response_text: str,
        request_id: str | None = None,
    ) -> None:
        del user_message
        if not isinstance(message_history, AuraSessionState):
            return
        message_history.commit_turn(
            response_text=response_text,
            request_id=request_id,
        )

    async def _ensure_cross_turn_penalty(
        self,
        config: AuraStreamingVideoSessionConfig,
        message_history: AuraSessionState,
    ) -> CrossTurnPenalty | None:
        if message_history.cross_turn_penalty is not None:
            return message_history.cross_turn_penalty
        if config.cross_turn_penalty <= 0 or self._engine_client is None:
            return None
        try:
            tokenizer = await self._engine_client.get_tokenizer()
        except Exception:
            return None
        message_history.cross_turn_penalty = CrossTurnPenalty(
            tokenizer,
            **config.cross_turn_penalty_kwargs(),
        )
        return message_history.cross_turn_penalty

    async def _receive_config(self, websocket) -> StreamingVideoSessionConfig | None:
        import asyncio
        import json

        from pydantic import ValidationError

        try:
            raw = await asyncio.wait_for(websocket.receive_text(), timeout=self._config_timeout)
        except asyncio.TimeoutError:
            await self._send_error(websocket, "Timeout waiting for session.config")
            return None

        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            await self._send_error(websocket, "Invalid JSON in session.config")
            return None

        if not isinstance(msg, dict) or msg.get("type") != "session.config":
            await self._send_error(
                websocket,
                f"Expected session.config, got: {msg.get('type') if isinstance(msg, dict) else type(msg).__name__}",
            )
            return None

        config_data = {k: v for k, v in msg.items() if k != "type"}
        alias_map = {
            "num_sample_frames": "num_frames",
            "evs_enabled": "enable_frame_filter",
            "evs_threshold": "frame_filter_threshold",
        }
        for old_key, new_key in alias_map.items():
            if old_key in config_data and new_key not in config_data:
                config_data[new_key] = config_data[old_key]

        try:
            config = AuraStreamingVideoSessionConfig(**config_data)
        except ValidationError as e:
            await self._send_error(websocket, f"Invalid session config: {e}")
            return None
        if config.tool_mode == "auto":
            parser_ready = bool(
                getattr(self._chat_service, "enable_auto_tools", False)
                and getattr(self._chat_service, "parser_cls", None) is not None
            )
            if self._tool_executor is None or not self._tool_tokenizer_path or not parser_ready:
                await self._send_error(
                    websocket,
                    "AURA tool mode is unavailable; server must enable the mock executor, "
                    "--enable-auto-tool-choice and --tool-call-parser qwen3_xml",
                )
                return None
        return config

    async def prepare_chat_request_kwargs(
        self,
        config: StreamingVideoSessionConfig,
        frame_buffer: list[str],
        audio_buffer: bytearray,
        message_history: Any,
        query_text: str,
        prewarmed_frames: dict[str, tuple[Any, str]],
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        messages, user_message = self.build_engine_prompt(
            config,
            frame_buffer,
            audio_buffer,
            message_history,
            query_text,
            prewarmed_frames,
        )
        additional_information = user_message.pop(_AURA_ADDITIONAL_INFO_KEY, None)

        if isinstance(message_history, AuraSessionState) and isinstance(additional_information, dict):
            deferred = additional_information.get("deferred_multi_modal_data")
            message_history.freeze_turn_video(deferred if isinstance(deferred, dict) else None)

        aura_config = self._as_aura_config(config)
        penalty_kwargs: dict[str, Any] = {}
        if isinstance(message_history, AuraSessionState):
            penalty = await self._ensure_cross_turn_penalty(aura_config, message_history)
            if penalty is not None:
                penalty_kwargs = penalty.build_sampling_kwargs()

        request_kwargs: dict[str, Any] = {
            "model": config.model or "default",
            "messages": messages,
            "stream": True,
            "modalities": config.modalities,
            "add_generation_prompt": True,
            "continue_final_message": False,
            "add_special_tokens": False,
        }
        if config.sampling_params_list or penalty_kwargs:
            request_kwargs["sampling_params_list"] = merge_penalty_sampling_params(
                config.sampling_params_list,
                penalty_kwargs,
            )

        extra_attrs: dict[str, Any] = {}
        if isinstance(additional_information, dict):
            extra_attrs["additional_information"] = additional_information
        return request_kwargs, user_message, extra_attrs

    async def _process_query_engine(
        self,
        websocket,
        config: StreamingVideoSessionConfig,
        frame_buffer: list[str],
        audio_buffer: bytearray,
        message_history: Any,
        query_text: str,
        request_id: str,
        interrupt_event,
        prewarmed_frames: dict[str, tuple[Any, str]],
        release_turn_lock=None,
    ) -> None:
        """Run a bounded pass1 -> execute -> resume transaction for tool mode."""

        aura_config = self._as_aura_config(config)
        if aura_config.tool_mode != "auto":
            return await super()._process_query_engine(
                websocket,
                config,
                frame_buffer,
                audio_buffer,
                message_history,
                query_text,
                request_id,
                interrupt_event,
                prewarmed_frames,
                release_turn_lock=release_turn_lock,
            )
        if self._tool_executor is None or not isinstance(message_history, AuraSessionState):
            await self._send_error(websocket, "AURA tool executor is unavailable")
            return

        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionRequest,
        )

        request_kwargs, user_message, extra_attrs = await self.prepare_chat_request_kwargs(
            config,
            frame_buffer,
            audio_buffer,
            message_history,
            query_text,
            prewarmed_frames,
        )
        base_additional = extra_attrs.get("additional_information")
        if not isinstance(base_additional, dict):
            await self._send_error(websocket, "AURA tool transaction metadata is missing")
            return

        try:
            tokenizer = await self._engine_client.get_tokenizer()
        except Exception:
            await self._send_error(websocket, "Unable to initialize AURA tool parser")
            return

        await websocket.send_json({"type": "response.start", "request_id": request_id})
        transient_messages: list[dict[str, Any]] = []
        transcript = ""
        transcript_done_sent = False
        pass_number = 1
        tool_depth = 1
        depth_error_returned = False
        intent_gate_retries = 0
        current_kwargs = request_kwargs
        routing_pass = aura_config.tool_intent_gate
        current_additional = (
            {**base_additional, "aura_tool_enabled": [False]}
            if routing_pass
            else base_additional
        )

        while True:
            internal_request_id = f"{request_id}-tool-p{pass_number}"
            try:
                chat_request = ChatCompletionRequest(**current_kwargs)
                setattr(chat_request, "additional_information", current_additional)
                engine_prompt = await self._preprocess_to_engine_prompt(chat_request)
            except Exception as exc:
                logger.exception("AURA tool pass preprocessing failed")
                await self._send_error(websocket, f"Tool pass preprocessing failed: {exc}")
                return

            parsed: ParsedAuraToolTurn | None = None
            tool_task: asyncio.Task[list[AuraToolResult]] | None = None
            tool_results: list[AuraToolResult] | None = None
            transaction_calls: list[AuraToolCall] = []

            async def start_tool_execution() -> asyncio.Task[list[AuraToolResult]]:
                for call in transaction_calls:
                    await websocket.send_json(
                        {
                            "type": "response.tool.started",
                            "request_id": request_id,
                            "call_id": call.id,
                            "name": call.name,
                            "status": "started",
                        }
                    )

                async def _execute_calls() -> list[AuraToolResult]:
                    results = list(
                        await asyncio.gather(
                            *[
                                self._tool_executor.execute(
                                    session_id=message_history.session_id,
                                    request_id=internal_request_id,
                                    call=call,
                                    depth=tool_depth,
                                )
                                for call in transaction_calls
                            ]
                        )
                    )
                    for result in results:
                        await websocket.send_json(
                            {
                                "type": "response.tool.done",
                                "request_id": request_id,
                                "call_id": result.call_id,
                                "name": result.name,
                                "status": result.status,
                                "content": result.content,
                            }
                        )
                    return results

                return asyncio.create_task(_execute_calls())

            async def on_stage1_text(raw_text: str) -> bool:
                nonlocal parsed, tool_task, tool_results, transaction_calls, depth_error_returned
                if parsed is not None:
                    return bool(transaction_calls)
                parsed = parse_aura_tool_output(
                    tokenizer,
                    raw_text,
                    request_id=internal_request_id,
                    tool_schemas=self._tool_executor.tool_schemas,
                )
                if not parsed.calls and parsed.error is None:
                    return False
                preamble = extract_aura_tool_preamble(raw_text)
                if preamble:
                    await websocket.send_json(
                        {
                            "type": "response.tool.preamble.text",
                            "request_id": request_id,
                            "text": preamble,
                        }
                    )
                transaction_calls = list(parsed.calls)
                if parsed.error:
                    transaction_calls = [
                        AuraToolCall(
                            id=f"{internal_request_id}-invalid",
                            name="invalid_tool_call",
                            arguments={},
                        )
                    ]
                if tool_depth > aura_config.max_tool_depth:
                    depth_error_returned = True
                    tool_results = [
                        self._tool_error_result(call, "tool_depth_exceeded")
                        for call in transaction_calls
                    ]
                    for result in tool_results:
                        await websocket.send_json(
                            {
                                "type": "response.tool.done",
                                "request_id": request_id,
                                "call_id": result.call_id,
                                "name": result.name,
                                "status": result.status,
                                "content": result.content,
                            }
                        )
                    return True
                if parsed.error:
                    tool_results = [
                        self._tool_error_result(call, parsed.error) for call in transaction_calls
                    ]
                    for result in tool_results:
                        await websocket.send_json(
                            {
                                "type": "response.tool.done",
                                "request_id": request_id,
                                "call_id": result.call_id,
                                "name": result.name,
                                "status": result.status,
                                "content": result.content,
                            }
                        )
                    return True

                # Routing / no-tool passes keep tools disabled. Defer execution
                # until the pass finishes so we can retry without executing.
                tools_enabled = bool(
                    (current_additional.get("aura_tool_enabled") or [False])[0]
                )
                if aura_config.tool_intent_gate and not tools_enabled:
                    return True
                tool_task = await start_tool_execution()
                return True

            async def on_preamble_audio(b64: str) -> None:
                await websocket.send_json(
                    {
                        "type": "response.tool.preamble.audio.delta",
                        "request_id": request_id,
                        "data": b64,
                        "format": "wav",
                    }
                )

            async def on_transcript(text: str) -> None:
                nonlocal transcript, transcript_done_sent
                transcript = text
                if transcript_done_sent:
                    return
                await websocket.send_json(
                    {
                        "type": "user.transcript.done",
                        "text": transcript,
                        "request_id": request_id,
                    }
                )
                transcript_done_sent = True

            collected = await self._collect_aura_tool_pass(
                config=aura_config,
                request_id=internal_request_id,
                interrupt_event=interrupt_event,
                engine_prompt=engine_prompt,
                on_text_ready=on_stage1_text,
                on_preamble_audio=on_preamble_audio,
                on_transcript=on_transcript,
            )
            if collected["interrupted"]:
                if tool_task is not None and not tool_task.done():
                    tool_task.cancel()
                return
            if collected["transcript"] and not transcript_done_sent:
                transcript = collected["transcript"]
                await websocket.send_json(
                    {
                        "type": "user.transcript.done",
                        "text": transcript,
                        "request_id": request_id,
                    }
                )
                transcript_done_sent = True

            if parsed is None:
                parsed = parse_aura_tool_output(
                    tokenizer,
                    collected["text"],
                    request_id=internal_request_id,
                    tool_schemas=self._tool_executor.tool_schemas,
                )
            if collected["preamble_audio_count"]:
                await websocket.send_json(
                    {
                        "type": "response.tool.preamble.audio.done",
                        "request_id": request_id,
                    }
                )
            if not parsed.calls and parsed.error is None:
                if routing_pass:
                    routing_pass = False
                    intent_text = transcript or query_text
                    if aura_any_tool_intent(self._tool_executor.tool_schemas, intent_text):
                        # The routing pass intentionally had no tool template.
                        # Reuse its ASR transcript and video in a tool-enabled
                        # pass only when the user's own words request a tool.
                        pass_number += 1
                        current_additional = {
                            **base_additional,
                            "aura_tool_enabled": [True],
                            "aura_tool_pass": [pass_number],
                            "aura_tool_resume": [{"transcript": transcript}],
                            "omni_skip_stages": [0],
                        }
                        current_kwargs = {
                            **request_kwargs,
                            "messages": [{"role": "user", "content": []}],
                        }
                        continue
                final_text = (parsed.content or "").strip()
                await self._emit_aura_tool_final(
                    websocket=websocket,
                    config=aura_config,
                    message_history=message_history,
                    user_message=user_message,
                    request_id=request_id,
                    response_text=final_text,
                    audio_deltas=collected["audio_deltas"],
                    release_turn_lock=release_turn_lock,
                    tool_chain=transient_messages or None,
                    transcript=transcript,
                )
                return

            calls = transaction_calls or list(parsed.calls)
            if parsed.error and not calls:
                calls = [AuraToolCall(id=f"{internal_request_id}-invalid", name="invalid_tool_call", arguments={})]
            tools_enabled = bool(
                (current_additional.get("aura_tool_enabled") or [False])[0]
            )
            # Native has no per-tool intent classifier. Omni only uses the gate
            # to decide whether tools are on; while they are off, never execute.
            if (
                aura_config.tool_intent_gate
                and not tools_enabled
                and calls
                and parsed.error is None
            ):
                if intent_gate_retries:
                    await self._emit_aura_tool_final(
                        websocket=websocket,
                        config=aura_config,
                        message_history=message_history,
                        user_message=user_message,
                        request_id=request_id,
                        response_text=(parsed.content or "").strip()
                        or "抱歉，我暫時無法直接回答這個問題。",
                        audio_deltas=[],
                        release_turn_lock=release_turn_lock,
                        transcript=transcript,
                    )
                    return
                logger.warning(
                    "AURA tool intent gate ignored tool calls while tools disabled "
                    "request_id=%s tools=%s transcript=%r",
                    internal_request_id,
                    [call.name for call in calls],
                    transcript or query_text,
                )
                # Re-run Stage-1 against the same video and transcript without a
                # tool template. Do not persist the hallucinated call or expose
                # it as a real tool event.
                intent_gate_retries += 1
                pass_number += 1
                current_additional = {
                    **base_additional,
                    "aura_tool_enabled": [False],
                    "aura_tool_pass": [pass_number],
                    "aura_tool_resume": [{"transcript": transcript}],
                    "omni_skip_stages": [0],
                }
                current_kwargs = {
                    **request_kwargs,
                    "messages": [{"role": "user", "content": []}],
                }
                continue
            if tool_task is None and calls and parsed.error is None:
                transaction_calls = calls
                tool_task = await start_tool_execution()
            if tool_task is not None:
                tool_results = await tool_task
            results = tool_results or []

            transient_messages.append(
                {
                    "role": "assistant",
                    "content": parsed.content or "",
                    "reasoning_content": parsed.reasoning or "",
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": call.arguments,
                            },
                        }
                        for call in calls
                    ],
                }
            )
            transient_messages.extend(
                {
                    "role": "tool",
                    "tool_call_id": result.call_id,
                    "content": result.content,
                }
                for result in results
            )
            force_final_answer = bool(
                getattr(self._tool_executor, "force_final_after_success", False)
                and results
                and all(result.status == "completed" for result in results)
            )

            if depth_error_returned:
                # Give the model one final error-response pass. A repeated tool
                # call cannot execute and is converted to a safe text fallback.
                if pass_number > aura_config.max_tool_depth + 1:
                    await self._emit_aura_tool_final(
                        websocket=websocket,
                        config=aura_config,
                        message_history=message_history,
                        user_message=user_message,
                        request_id=request_id,
                        response_text="抱歉，工具呼叫未能完成，請稍後再試。",
                        audio_deltas=[],
                        release_turn_lock=release_turn_lock,
                        tool_chain=transient_messages or None,
                        transcript=transcript,
                        rearm_pending=True,
                    )
                    return

            pass_number += 1
            tool_depth += 1
            current_additional = {
                **base_additional,
                "aura_tool_pass": [pass_number],
                "aura_tool_resume": [
                    {
                        "transcript": transcript,
                        "transient_messages": transient_messages,
                        "force_final_answer": force_final_answer,
                    }
                ],
                "omni_skip_stages": [0],
            }
            current_kwargs = {
                **request_kwargs,
                "messages": [{"role": "user", "content": []}],
            }

    async def _collect_aura_tool_pass(
        self,
        *,
        config: AuraStreamingVideoSessionConfig,
        request_id: str,
        interrupt_event,
        engine_prompt: Any,
        on_text_ready=None,
        on_preamble_audio=None,
        on_transcript=None,
    ) -> dict[str, Any]:
        """Drain one engine pass; start tools when Stage-1 text is complete."""

        from vllm_omni.entrypoints.openai import video_stream_envs

        text_parts: list[str] = []
        previous_text = ""
        last_token_ids: list[int] = []
        transcript = ""
        audio_deltas: list[str] = []
        audio_chunks_drained = 0
        audio_tail_tensors: list[Any] = []
        interrupted = False
        streaming = video_stream_envs.VLLM_VIDEO_ASYNC_CHUNK == "on"
        text_ready = False
        tool_transaction = False
        preamble_audio_count = 0
        tokenizer = None
        if self._engine_client is not None:
            try:
                tokenizer = await self._get_aura_stage1_tokenizer()
            except Exception:
                tokenizer = None

        def _pass_text() -> str:
            return canonical_aura_response_text(
                streaming_text="".join(text_parts),
                decoded_text=decode_aura_content_ids_for_display(tokenizer, last_token_ids),
            )

        async def notify_text_ready() -> None:
            nonlocal text_ready, tool_transaction
            if text_ready:
                return
            text_ready = True
            if on_text_ready is not None:
                tool_transaction = bool(await on_text_ready(_pass_text()))

        async def take_audio_b64(b64: str) -> None:
            nonlocal preamble_audio_count
            if not b64:
                return
            if tool_transaction and on_preamble_audio is not None:
                preamble_audio_count += 1
                await on_preamble_audio(b64)
            else:
                audio_deltas.append(b64)

        result_gen = self._engine_client.generate(
            prompt=engine_prompt,
            request_id=request_id,
            output_modalities=config.modalities,
        )
        async for output in result_gen:
            if interrupt_event.is_set():
                interrupted = True
            if interrupted:
                continue
            out_type = getattr(output, "final_output_type", "text")
            if out_type == "transcript":
                current = _extract_text(getattr(output, "request_output", None))
                if current:
                    transcript = _clean_asr_transcript(current)
                    if (
                        on_transcript is not None
                        and transcript
                        and getattr(output, "finished", False)
                    ):
                        await on_transcript(transcript)
                continue
            if out_type == "audio":
                await notify_text_ready()
                if streaming:
                    b64, audio_chunks_drained = self._extract_audio_delta_b64(
                        output,
                        audio_chunks_drained,
                    )
                    await take_audio_b64(b64)
                else:
                    audio_data = self._get_audio_data(output)
                    if audio_data is not None:
                        audio_tail_tensors = list(audio_data) if isinstance(audio_data, list) else [audio_data]
                continue
            token_ids = self._output_token_ids(output)
            if token_ids:
                last_token_ids = token_ids
            delta_text, previous_text = self._extract_text_delta(output, previous_text)
            if delta_text:
                text_parts.append(delta_text)
            if getattr(output, "finished", False):
                await notify_text_ready()

        if not streaming and audio_tail_tensors:
            import torch

            coalesced = (
                audio_tail_tensors[0]
                if len(audio_tail_tensors) == 1
                else torch.cat(audio_tail_tensors, dim=-1)
            )
            tail_np = self._tensor_to_1d_np(coalesced)
            b64, _ = self._encode_tail(
                tail_np,
                0,
                new_drained=len(audio_tail_tensors),
                is_first=True,
            )
            await notify_text_ready()
            await take_audio_b64(b64)
        await notify_text_ready()
        return {
            "text": _pass_text(),
            "transcript": transcript,
            "audio_deltas": audio_deltas,
            "preamble_audio_count": preamble_audio_count,
            "interrupted": interrupted,
        }

    async def _emit_aura_tool_final(
        self,
        *,
        websocket,
        config: AuraStreamingVideoSessionConfig,
        message_history: AuraSessionState,
        user_message: dict[str, Any],
        request_id: str,
        response_text: str,
        audio_deltas: list[str],
        release_turn_lock=None,
        tool_chain: list[dict[str, Any]] | None = None,
        transcript: str = "",
        rearm_pending: bool = False,
    ) -> None:
        """Emit only parser-classified natural content and its buffered audio."""

        if tool_chain:
            message_history.pending_commit_tool_chain = list(tool_chain)
            # Depth-error finals may run after Stage1 discarded pending on tool
            # XML. Do not re-arm after a successful pass2: Stage1 already
            # committed, and a new pending would double-write the turn.
            if (
                rearm_pending
                and message_history.session_id
                and not has_pending_turn(message_history.session_id)
            ):
                record_pending_turn(
                    message_history.session_id,
                    request_id=request_id,
                    transcript=transcript,
                    video_tuple=None,
                    deferred_mm=message_history.pending_turn_video,
                    tool_chain=tool_chain,
                    had_vision=bool(message_history.pending_turn_video),
                )

        if config.stream_text_deltas and response_text:
            await websocket.send_json(
                {
                    "type": "response.text.delta",
                    "request_id": request_id,
                    "delta": response_text,
                }
            )
        await websocket.send_json(
            {
                "type": "response.text.done",
                "request_id": request_id,
                "text": response_text,
            }
        )
        if release_turn_lock is not None:
            await release_turn_lock(
                message_history=message_history,
                user_message=user_message,
                response_text=response_text,
                request_id=request_id,
            )
        else:
            self.on_turn_complete(message_history, user_message, response_text, request_id)
        for data in audio_deltas:
            await websocket.send_json(
                {
                    "type": "response.audio.delta",
                    "request_id": request_id,
                    "data": data,
                    "format": "wav",
                }
            )
        if audio_deltas or "audio" in config.modalities:
            await websocket.send_json(
                {
                    "type": "response.audio.done",
                    "request_id": request_id,
                }
            )

    @staticmethod
    def _tool_error_result(call: AuraToolCall, code: str) -> AuraToolResult:
        content = json.dumps(
            {"ok": False, "error": {"code": code}},
            separators=(",", ":"),
        )
        return AuraToolResult(
            call_id=call.id,
            name=call.name,
            status="error",
            content=content,
            latency_ms=0.0,
            output_bytes=len(content.encode()),
        )

    async def _run_engine_generation(
        self,
        websocket,
        config: StreamingVideoSessionConfig,
        message_history: Any,
        user_message: dict[str, Any],
        request_id: str,
        interrupt_event,
        engine_prompt: Any,
        release_turn_lock=None,
    ) -> None:
        """Stream engine outputs; release turn lock after assistant text when configured."""
        from vllm_omni.entrypoints.openai import video_stream_envs
        from vllm_omni.outputs import OmniRequestOutput

        def _response_event(payload: dict[str, Any]) -> dict[str, Any]:
            payload["request_id"] = request_id
            return payload

        await websocket.send_json(_response_event({"type": "response.start"}))
        text_parts: list[str] = []
        last_token_ids: list[int] = []
        text_done_sent = False
        turn_lock_released = False
        audio_chunk_count = 0
        audio_chunks_drained = 0
        previous_text = ""
        previous_transcript = ""
        transcript_done_sent = False
        interrupted = False
        user_content = user_message.get("content", [])
        voice_turn = isinstance(user_content, list) and any(
            isinstance(item, dict) and item.get("type") == "input_audio" for item in user_content
        )

        async_chunk_mode = video_stream_envs.VLLM_VIDEO_ASYNC_CHUNK
        streaming = async_chunk_mode == "on"
        aura_config = self._as_aura_config(config)
        stream_text_deltas = aura_config.stream_text_deltas
        audio_tail_tensors: list[Any] = []
        last_text_metrics: dict[str, Any] | None = None
        last_audio_metrics: dict[str, Any] | None = None
        tokenizer = None
        if self._engine_client is not None:
            try:
                tokenizer = await self._get_aura_stage1_tokenizer()
            except Exception:
                tokenizer = None

        def _client_text() -> str:
            return canonical_aura_response_text(
                streaming_text="".join(text_parts),
                decoded_text=decode_aura_content_ids_for_display(tokenizer, last_token_ids),
            )

        def _event_metrics(output: OmniRequestOutput | None) -> dict[str, Any] | None:
            if not getattr(config, "return_stage_metrics", False) or output is None:
                return None
            metrics = getattr(output, "metrics", None)
            return metrics if isinstance(metrics, dict) else None

        def _with_metrics(payload: dict[str, Any], metrics: dict[str, Any] | None) -> dict[str, Any]:
            if metrics:
                payload["metrics"] = metrics
            return payload

        async def _try_release_turn_lock(full_text: str) -> None:
            nonlocal turn_lock_released
            if release_turn_lock is None or turn_lock_released:
                return
            turn_lock_released = True
            await release_turn_lock(
                message_history=message_history,
                user_message=user_message,
                response_text=full_text,
                request_id=request_id,
            )

        async def _finalize_silent_turn() -> None:
            """Mark the WebSocket response as silent; keep draining ``generate()``.

            Do not ``break`` out of the ``async for`` over ``generate()``: closing
            the async generator runs ``GeneratorExit`` in ``AsyncOmni.generate``,
            which aborts the orchestrator request and can poison stage-2/3 for the
            next spoken turn on the same session.  Do not call
            ``engine_client.abort()`` either for the same reason.
            """
            nonlocal previous_text, interrupted
            interrupted = True
            text_parts.clear()
            text_parts.append(SILENT_TEXT)
            previous_text = SILENT_TEXT

        async def _maybe_finalize_silent_turn() -> None:
            """Send silent ``text.done`` early, then drain engine outputs to completion."""
            nonlocal text_done_sent
            if interrupted:
                return
            await _finalize_silent_turn()
            if not text_done_sent:
                full_text = _client_text()
                await websocket.send_json(
                    _with_metrics(
                        _response_event({"type": "response.text.done", "text": full_text}),
                        last_text_metrics,
                    )
                )
                text_done_sent = True
                await _try_release_turn_lock(full_text)

        try:
            result_gen = self._engine_client.generate(
                prompt=engine_prompt,
                request_id=request_id,
                output_modalities=config.modalities,
            )

            async for output in result_gen:
                if interrupt_event.is_set():
                    if not interrupted:
                        interrupted = True
                if interrupted:
                    continue

                if not isinstance(output, OmniRequestOutput):
                    continue

                out_type = getattr(output, "final_output_type", "text")
                metrics = _event_metrics(output)

                if out_type == "transcript":
                    if not voice_turn or transcript_done_sent:
                        continue
                    current_transcript = _extract_text(output.request_output)
                    if current_transcript:
                        previous_transcript = current_transcript
                    if output.finished:
                        transcript = _clean_asr_transcript(previous_transcript)
                        if transcript:
                            await websocket.send_json(
                                _with_metrics(
                                    _response_event({"type": "user.transcript.done", "text": transcript}),
                                    metrics,
                                )
                            )
                        transcript_done_sent = True
                    continue

                if out_type == "audio":
                    if streaming and not text_done_sent:
                        full_text = _client_text()
                        await websocket.send_json(
                            _with_metrics(
                                _response_event({"type": "response.text.done", "text": full_text}),
                                last_text_metrics,
                            )
                        )
                        text_done_sent = True
                        await _try_release_turn_lock(full_text)

                    audio_chunk_count += 1
                    last_audio_metrics = metrics or last_audio_metrics
                    if streaming:
                        b64, audio_chunks_drained = self._extract_audio_delta_b64(
                            output,
                            audio_chunks_drained,
                        )
                        if b64:
                            logger.info(
                                "[video_stream response.audio.delta] req=%s chunk=%d drained=%d "
                                "b64_len=%d current_text_len=%d text_done_sent=%s",
                                request_id,
                                audio_chunk_count,
                                audio_chunks_drained,
                                len(b64),
                                len("".join(text_parts)),
                                text_done_sent,
                            )
                            await websocket.send_json(
                                _with_metrics(
                                    _response_event(
                                        {
                                            "type": "response.audio.delta",
                                            "data": b64,
                                            "format": "wav",
                                        }
                                    ),
                                    metrics,
                                )
                            )
                    else:
                        audio_data = self._get_audio_data(output)
                        if audio_data is not None:
                            audio_tail_tensors = list(audio_data) if isinstance(audio_data, list) else [audio_data]
                else:
                    last_text_metrics = metrics or last_text_metrics
                    token_ids = self._output_token_ids(output)
                    if token_ids:
                        last_token_ids = token_ids
                    content_ids = _trim_aura_response_token_ids(token_ids)
                    # Classify silent on trimmed content ids (not chat-template prefix).
                    if content_ids and should_stop_aura_silent_generation(token_ids=content_ids):
                        await _maybe_finalize_silent_turn()
                        continue
                    delta_text, previous_text = self._extract_text_delta(output, previous_text)
                    if delta_text:
                        text_parts.append(delta_text)
                        full_text = "".join(text_parts)
                        # Only judge text-silent after Stage1 finishes — partial
                        # whitespace/punctuation deltas (e.g. leading " ") must
                        # not drop a later Chinese spoken turn's audio.
                        if output.finished and should_stop_aura_silent_generation(text=full_text):
                            await _maybe_finalize_silent_turn()
                            continue
                        if streaming and stream_text_deltas:
                            await websocket.send_json(
                                _with_metrics(
                                    _response_event({"type": "response.text.delta", "delta": delta_text}),
                                    metrics,
                                )
                            )

            if not text_done_sent:
                full_text = _client_text()
                await websocket.send_json(
                    _with_metrics(
                        _response_event({"type": "response.text.done", "text": full_text}),
                        last_text_metrics,
                    )
                )
                text_done_sent = True
                await _try_release_turn_lock(full_text)

            if not streaming and audio_tail_tensors:
                import torch

                try:
                    coalesced = (
                        audio_tail_tensors[0] if len(audio_tail_tensors) == 1 else torch.cat(audio_tail_tensors, dim=-1)
                    )
                    tail_np = self._tensor_to_1d_np(coalesced)
                    b64, _ = self._encode_tail(tail_np, 0, new_drained=len(audio_tail_tensors), is_first=True)
                    if b64:
                        await websocket.send_json(
                            _with_metrics(
                                _response_event(
                                    {
                                        "type": "response.audio.delta",
                                        "data": b64,
                                        "format": "wav",
                                    }
                                ),
                                last_audio_metrics,
                            )
                        )
                except Exception:
                    pass

            if audio_chunk_count > 0:
                logger.info(
                    "[video_stream response.audio.done] req=%s audio_chunks=%d drained=%d text_len=%d",
                    request_id,
                    audio_chunk_count,
                    audio_chunks_drained,
                    len("".join(text_parts)),
                )
                await websocket.send_json(
                    _with_metrics(_response_event({"type": "response.audio.done"}), last_audio_metrics)
                )

            if release_turn_lock is None and not turn_lock_released:
                response_text = _client_text()
                self.on_turn_complete(message_history, user_message, response_text, request_id)

        except Exception:
            await self._send_error(websocket, "Query processing failed")

        if not text_done_sent:
            full_text = _client_text()
            await websocket.send_json(
                _with_metrics(
                    _response_event({"type": "response.text.done", "text": full_text}),
                    last_text_metrics,
                )
            )

    @staticmethod
    def _as_aura_config(config: StreamingVideoSessionConfig) -> AuraStreamingVideoSessionConfig:
        if isinstance(config, AuraStreamingVideoSessionConfig):
            return config
        return AuraStreamingVideoSessionConfig(**config.model_dump())

    @staticmethod
    def _output_token_ids(output: Any) -> list[int]:
        """First completion sequence token ids from an engine output chunk."""
        request_output = getattr(output, "request_output", None)
        if request_output is None:
            return []
        outputs = getattr(request_output, "outputs", None)
        if not isinstance(outputs, list) or not outputs:
            return []
        first = outputs[0]
        cumulative = getattr(first, "cumulative_token_ids", None)
        if cumulative:
            return list(cumulative)
        token_ids = getattr(first, "token_ids", None)
        return list(token_ids) if token_ids else []


def create_streaming_video_handler(
    chat_service: Any,
    idle_timeout: float = _DEFAULT_IDLE_TIMEOUT,
    config_timeout: float = _DEFAULT_CONFIG_TIMEOUT,
    engine_client: Any | None = None,
    tool_executor: Any | None = None,
) -> OmniStreamingVideoHandlerBase:
    """Create the handler for ``/v1/video/chat/stream``.

    Routes to :class:`AuraStreamingVideoHandler` when the deploy YAML
    ``pipeline`` is ``aura_omni``.
    """
    pipeline = _resolve_deploy_pipeline(engine_client) if engine_client is not None else None
    if pipeline in _AURA_PIPELINE_NAMES:
        return AuraStreamingVideoHandler(
            chat_service=chat_service,
            idle_timeout=idle_timeout,
            config_timeout=config_timeout,
            engine_client=engine_client,
            tool_executor=tool_executor,
            tool_tokenizer_path=_resolve_aura_stage1_model(engine_client),
        )

    return QwenOmniStreamingVideoHandler(
        chat_service=chat_service,
        idle_timeout=idle_timeout,
        config_timeout=config_timeout,
        engine_client=engine_client,
    )
