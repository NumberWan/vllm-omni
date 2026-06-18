# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for AuraStreamingVideoHandler."""

from __future__ import annotations

import base64
import io
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from vllm_omni.entrypoints.openai.aura_session_history import SessionHistory
from vllm_omni.entrypoints.openai.serving_video_stream import (
    AuraSessionState,
    AuraStreamingVideoHandler,
    AuraStreamingVideoSessionConfig,
)
from vllm_omni.entrypoints.openai.video_stream_base import VideoStreamTurnTrigger

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_jpeg(r: int = 128, g: int = 128, b: int = 128) -> bytes:
    img = Image.new("RGB", (16, 16), (r, g, b))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def _session_state() -> AuraSessionState:
    return AuraSessionState(history=SessionHistory(), turn_frame_arrays=[])


def test_aura_disables_manual_query_and_interrupt():
    handler = AuraStreamingVideoHandler(chat_service=object())
    assert handler.supports_manual_query_turn() is False
    assert handler.supports_query_interrupt() is False
    assert handler.releases_turn_after_text_done() is True


def test_should_trigger_turn_respects_auto_trigger_gate():
    handler = AuraStreamingVideoHandler(chat_service=object())
    config = AuraStreamingVideoSessionConfig(model="test", auto_trigger=True, auto_trigger_min_frames=2)

    assert handler.should_trigger_turn(
        VideoStreamTurnTrigger(frame_count=1, is_generating=False, is_turn_locked=False, config=config)
    ) is False
    assert handler.should_trigger_turn(
        VideoStreamTurnTrigger(frame_count=2, is_generating=False, is_turn_locked=False, config=config)
    ) is True
    assert handler.should_trigger_turn(
        VideoStreamTurnTrigger(frame_count=3, is_generating=True, is_turn_locked=True, config=config)
    ) is False
    # TTS tail still running: is_generating=True but conversation turn is released.
    assert handler.should_trigger_turn(
        VideoStreamTurnTrigger(frame_count=3, is_generating=True, is_turn_locked=False, config=config)
    ) is True

    disabled = AuraStreamingVideoSessionConfig(model="test", auto_trigger=False)
    assert handler.should_trigger_turn(
        VideoStreamTurnTrigger(frame_count=5, is_generating=False, is_turn_locked=False, config=disabled)
    ) is False


def test_build_engine_prompt_stores_audio_and_session_payload():
    handler = AuraStreamingVideoHandler(chat_service=object())
    config = AuraStreamingVideoSessionConfig(model="test", aura_system_prompt="system-a")
    state = _session_state()
    state.turn_frame_arrays = [
        np.zeros((8, 8, 3), dtype=np.uint8),
        np.ones((8, 8, 3), dtype=np.uint8),
    ]

    messages, user_message = handler.build_engine_prompt(
        config,
        [_b64(_make_jpeg())],
        bytearray(b"\x00\x01"),
        state,
        "",
        {},
    )

    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    content_types = [part["type"] for part in messages[0]["content"]]
    assert content_types == ["input_audio"]

    additional = user_message["_aura_additional_information"]
    assert "aura_session_state" in additional
    assert additional["aura_system_prompt"] == ["system-a"]
    assert additional["aura_turn_video"]["metadata"]["total_num_frames"] == 2
    assert len(additional["aura_turn_video"]["frames"]) == 2


@pytest.mark.asyncio
async def test_build_engine_prompt_via_preprocess_mock():
    captured_requests: list[Any] = []

    class CapturingChatService:
        chat_template = None
        chat_template_content_format = "string"

        class _Renderer:
            pass

        renderer = _Renderer()

        async def _preprocess_chat(self, request, messages, **kwargs):
            captured_requests.append(request)
            return messages, [{"prompt": "engine-prompt"}]

    handler = AuraStreamingVideoHandler(chat_service=CapturingChatService(), engine_client=MagicMock())
    config = AuraStreamingVideoSessionConfig(model="test")
    state = _session_state()
    state.turn_frame_arrays = [
        np.zeros((8, 8, 3), dtype=np.uint8),
        np.zeros((8, 8, 3), dtype=np.uint8),
    ]

    async def _noop_generation(*_args, **_kwargs):
        return None

    handler._run_engine_generation = _noop_generation  # type: ignore[method-assign]

    await handler._process_query_engine(
        websocket=MagicMock(),
        config=config,
        frame_buffer=[_b64(_make_jpeg())],
        audio_buffer=bytearray(b"\x00\x00"),
        message_history=state,
        query_text="",
        request_id="req-aura-1",
        interrupt_event=MagicMock(),
        prewarmed_frames={},
    )

    assert captured_requests
    request = captured_requests[0]
    additional = getattr(request, "additional_information")
    assert "aura_session_state" in additional
    assert "aura_turn_video" in additional
    assert request.messages[0]["content"][0]["type"] == "input_audio"


@pytest.mark.asyncio
async def test_process_query_merges_cross_turn_penalty_sampling_params():
    captured_requests: list[Any] = []

    class CapturingChatService:
        chat_template = None
        chat_template_content_format = "string"

        class _Renderer:
            pass

        renderer = _Renderer()

        async def _preprocess_chat(self, request, messages, **kwargs):
            captured_requests.append(request)
            return messages, [{"prompt": "engine-prompt"}]

    class _FakeTokenizer:
        all_special_ids = [0, 1]

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            return [ord(c) for c in text if c.isalnum() or ord(c) > 127]

        def decode(self, token_ids: list[int]) -> str:
            return "".join(chr(tid) for tid in token_ids)

    from vllm_omni.entrypoints.openai.aura_cross_turn_penalty import CrossTurnPenalty

    mock_engine = MagicMock()

    async def _get_tokenizer():
        return _FakeTokenizer()

    mock_engine.get_tokenizer = _get_tokenizer

    handler = AuraStreamingVideoHandler(chat_service=CapturingChatService(), engine_client=mock_engine)
    config = AuraStreamingVideoSessionConfig(model="test", cross_turn_penalty=2.0)
    state = _session_state()
    state.turn_frame_arrays = [
        np.zeros((8, 8, 3), dtype=np.uint8),
        np.zeros((8, 8, 3), dtype=np.uint8),
    ]
    penalty = CrossTurnPenalty(_FakeTokenizer(), window=2, logit_penalty=2.0)
    penalty.record("hello world")
    penalty.record("hello again")
    state.cross_turn_penalty = penalty

    async def _noop_generation(*_args, **_kwargs):
        return None

    handler._run_engine_generation = _noop_generation  # type: ignore[method-assign]

    await handler._process_query_engine(
        websocket=MagicMock(),
        config=config,
        frame_buffer=[_b64(_make_jpeg())],
        audio_buffer=bytearray(),
        message_history=state,
        query_text="",
        request_id="req-aura-penalty",
        interrupt_event=MagicMock(),
        prewarmed_frames={},
    )

    assert captured_requests
    sampling = getattr(captured_requests[0], "sampling_params_list", None)
    assert sampling is not None
    assert len(sampling) >= 2
    assert sampling[1].get("logit_bias") or sampling[1].get("bad_words")
