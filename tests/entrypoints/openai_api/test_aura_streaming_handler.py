# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for AuraStreamingVideoHandler."""

from __future__ import annotations

import asyncio
import base64
import io
import time
from dataclasses import replace
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from vllm_omni.entrypoints.openai.aura_tool_executor import (
    AuraToolCall,
    AuraToolExecutor,
    ParsedAuraToolTurn,
)
from vllm_omni.entrypoints.openai.serving_video_stream import (
    AuraStreamingVideoHandler,
    AuraStreamingVideoSessionConfig,
    canonical_aura_response_text,
    decode_aura_content_ids_for_display,
)
from vllm_omni.entrypoints.openai.video_stream_base import VideoStreamTurnTrigger
from vllm_omni.model_executor.stage_input_processors.aura_cross_turn_penalty import (
    merge_penalty_sampling_params,
)
from vllm_omni.model_executor.stage_input_processors.aura_omni import (
    frames_to_video_tuple,
    unpack_aura_video_ndarray,
)
from vllm_omni.model_executor.stage_input_processors.aura_session_history import (
    SILENT_TEXT,
    AuraSessionState,
    SessionHistory,
    clear_all_sessions,
    get_or_create_session_history,
    get_session_history,
    get_session_state,
    record_pending_turn,
    register_session,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture(autouse=True)
def _clear_aura_session_store():
    clear_all_sessions()
    yield
    clear_all_sessions()


def _make_jpeg(r: int = 128, g: int = 128, b: int = 128) -> bytes:
    img = Image.new("RGB", (16, 16), (r, g, b))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def _session_state() -> AuraSessionState:
    history = SessionHistory()
    session_id = "aura-test-session"
    register_session(session_id, history)
    return AuraSessionState(history=history, turn_frame_arrays=[], session_id=session_id)


def test_aura_enables_manual_query_without_interrupt():
    handler = AuraStreamingVideoHandler(chat_service=object())
    assert handler.supports_manual_query_turn() is True
    assert handler.supports_query_interrupt() is False
    assert handler.releases_turn_after_text_done() is True


def test_aura_streaming_session_config_native_aligned_defaults():
    config = AuraStreamingVideoSessionConfig(model="test")
    assert config.cross_turn_penalty == 1.0
    assert config.cross_turn_lookback == 10
    assert config.tool_intent_gate is True


def test_should_trigger_turn_respects_auto_trigger_gate():
    handler = AuraStreamingVideoHandler(chat_service=object())
    config = AuraStreamingVideoSessionConfig(model="test", auto_trigger=True, auto_trigger_min_frames=2)

    assert (
        handler.should_trigger_turn(
            VideoStreamTurnTrigger(frame_count=1, is_generating=False, is_turn_locked=False, config=config)
        )
        is False
    )
    assert (
        handler.should_trigger_turn(
            VideoStreamTurnTrigger(frame_count=2, is_generating=False, is_turn_locked=False, config=config)
        )
        is True
    )
    assert (
        handler.should_trigger_turn(
            VideoStreamTurnTrigger(frame_count=3, is_generating=True, is_turn_locked=True, config=config)
        )
        is False
    )
    # After text.done unlock, TTS may still be draining (is_generating=True).
    assert (
        handler.should_trigger_turn(
            VideoStreamTurnTrigger(frame_count=3, is_generating=True, is_turn_locked=False, config=config)
        )
        is True
    )

    disabled = AuraStreamingVideoSessionConfig(model="test", auto_trigger=False)
    assert (
        handler.should_trigger_turn(
            VideoStreamTurnTrigger(frame_count=5, is_generating=False, is_turn_locked=False, config=disabled)
        )
        is False
    )


def test_ensure_frames_for_audio_turn_uses_recent_buffer():
    handler = AuraStreamingVideoHandler(chat_service=object())
    config = AuraStreamingVideoSessionConfig(model="test")
    state = _session_state()
    assert state.turn_frame_arrays == []
    assert (
        handler.ensure_frames_for_audio_turn([], state, config) is False
    ), "empty session buffer cannot seed frames"

    raw = _make_jpeg(10, 20, 30)
    b64 = _b64(raw)
    assert handler.ensure_frames_for_audio_turn([b64], state, config) is True
    assert len(state.turn_frame_arrays) >= 1
    # Second call is a no-op when turn frames already exist.
    n = len(state.turn_frame_arrays)
    assert handler.ensure_frames_for_audio_turn([b64, _b64(_make_jpeg())], state, config) is True
    assert len(state.turn_frame_arrays) == n
    # Single seeded frame is enough for frames_to_video_tuple (duplicates to 2).
    video_array, meta = frames_to_video_tuple(
        list(state.turn_frame_arrays),
        fps=2.0,
        max_frames=16,
    )
    assert video_array.shape[0] >= 2
    assert int(meta["total_num_frames"]) >= 2


def test_auto_trigger_frame_count_uses_turn_frame_arrays():
    handler = AuraStreamingVideoHandler(chat_service=object())
    state = _session_state()
    session_buffer = [_b64(_make_jpeg()) for _ in range(5)]
    state.turn_frame_arrays = [
        np.zeros((8, 8, 3), dtype=np.uint8),
        np.zeros((8, 8, 3), dtype=np.uint8),
    ]
    assert handler.auto_trigger_frame_count(session_buffer, state) == 2


def test_per_turn_auto_trigger_not_cumulative_session_buffer():
    """13 frames @ min=2 should yield 6 turns, not 12 from uncleared frame_buffer."""
    handler = AuraStreamingVideoHandler(chat_service=object())
    config = AuraStreamingVideoSessionConfig(model="test", auto_trigger=True, auto_trigger_min_frames=2)
    state = _session_state()
    session_buffer: list[str] = []
    triggers = 0

    for i in range(13):
        raw = _make_jpeg(i, i, i)
        b64 = _b64(raw)
        session_buffer.append(b64)
        handler.on_frame_buffered(raw, b64, state, config)
        if handler.should_trigger_turn(
            VideoStreamTurnTrigger(
                frame_count=handler.auto_trigger_frame_count(session_buffer, state),
                is_generating=False,
                is_turn_locked=False,
                config=config,
            )
        ):
            triggers += 1
            state.turn_frame_arrays.clear()

    assert triggers == 6
    assert len(session_buffer) == 13


def test_on_frame_buffered_downscales_to_max_frame_edge():
    """API ingress should match native long-edge=640 before buffering pixels."""
    handler = AuraStreamingVideoHandler(chat_service=object())
    config = AuraStreamingVideoSessionConfig(model="test", max_frame_edge=640)
    state = _session_state()

    img = Image.new("RGB", (1080, 612), (10, 20, 30))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    raw = buf.getvalue()

    handler.on_frame_buffered(raw, _b64(raw), state, config)
    assert len(state.turn_frame_arrays) == 1
    frame = state.turn_frame_arrays[0]
    assert max(int(frame.shape[0]), int(frame.shape[1])) <= 640
    # 1080x612 -> scale 640/1080 => ~640x363
    assert frame.shape[1] == 640
    assert 350 <= frame.shape[0] <= 370


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
    assert additional["aura_session_id"] == state.session_id
    assert "aura_session_state" not in additional
    assert additional["aura_system_prompt"] == ["system-a"]
    deferred = additional["deferred_multi_modal_data"]
    assert deferred["video"][0][1]["total_num_frames"] == 2
    packed = deferred["video"][0][0]
    assert isinstance(packed, dict) and packed.get("__aura_video_ndarray__")
    assert unpack_aura_video_ndarray(packed).shape == (2, 8, 8, 3)
    assert additional["tts_ref_audio"]
    assert additional["tts_ref_text"]


def test_build_engine_prompt_omni_skip_stages():
    handler = AuraStreamingVideoHandler(chat_service=object())
    state = _session_state()
    state.turn_frame_arrays = [np.zeros((8, 8, 3), dtype=np.uint8)]

    _, text_only = handler.build_engine_prompt(
        AuraStreamingVideoSessionConfig(model="test", modalities=["text"]),
        [],
        bytearray(),
        state,
        "typed question",
        {},
    )
    assert text_only["_aura_additional_information"]["omni_skip_stages"] == [0]
    assert text_only["_aura_additional_information"]["omni_bypass_stage_text"] == ["typed question"]
    assert text_only["_aura_additional_information"]["aura_tts_enabled"] == [False]

    _, with_audio = handler.build_engine_prompt(
        AuraStreamingVideoSessionConfig(model="test"),
        [_b64(_make_jpeg())],
        bytearray(b"\x00\x01"),
        state,
        "",
        {},
    )
    assert with_audio["_aura_additional_information"]["omni_skip_stages"] == []
    assert with_audio["_aura_additional_information"]["aura_tts_enabled"] == [True]
    assert "tts_ref_audio" not in text_only["_aura_additional_information"]


def test_create_message_history_registers_server_side_store():
    handler = AuraStreamingVideoHandler(chat_service=object())
    config = AuraStreamingVideoSessionConfig(model="test")

    state = handler.create_message_history(config)

    assert state.session_id
    assert get_session_state(state.session_id) is state


def test_on_session_end_unregisters_server_side_store():
    handler = AuraStreamingVideoHandler(chat_service=object())
    state = _session_state()

    handler.on_session_end(state)

    assert get_session_state(state.session_id) is None


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

    from vllm_omni.model_executor.stage_input_processors.aura_cross_turn_penalty import CrossTurnPenalty

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

    merged = merge_penalty_sampling_params(
        [{"temperature": 0.7}, {"top_p": 0.9}],
        {"logit_bias": {42: -1.5}, "bad_words": ["foo"]},
    )
    assert merged[0] == {"temperature": 0.7}
    assert merged[1]["logit_bias"] == {42: -1.5}
    assert merged[1]["bad_words"] == ["foo"]


@pytest.mark.asyncio
async def test_tool_loop_drains_pass1_executes_once_and_resumes(monkeypatch):
    raw_tool = (
        "<think>use mock</think><tool_call><function=mock_echo>"
        "<parameter=text>hello</parameter></function></tool_call>"
    )
    generated_ids: list[str] = []
    drained_ids: list[str] = []
    preprocessed: list[Any] = []

    class FakeEngine:
        async def get_tokenizer(self):
            return object()

        async def generate(self, *, prompt, request_id, output_modalities):
            del prompt, output_modalities
            generated_ids.append(request_id)
            if request_id.endswith("p1"):
                yield SimpleNamespace(
                    final_output_type="transcript",
                    finished=True,
                    request_output=SimpleNamespace(
                        outputs=[SimpleNamespace(text="請回聲 hello", cumulative_text="請回聲 hello")]
                    ),
                )
            texts = (
                ["<think>use mock</think>", raw_tool]
                if request_id.endswith("p1")
                else ["<think>done</think>已完成 hello。"]
            )
            for text in texts:
                yield SimpleNamespace(
                    final_output_type="text",
                    request_output=SimpleNamespace(
                        outputs=[SimpleNamespace(text=text, cumulative_text=text)]
                    ),
                )
            drained_ids.append(request_id)

    class FakeWebSocket:
        def __init__(self):
            self.events: list[dict[str, Any]] = []

        async def send_json(self, event):
            self.events.append(event)

    def fake_parse(_tokenizer, raw, *, request_id, tool_schemas):
        del request_id, tool_schemas
        if "<tool_call>" in raw:
            return ParsedAuraToolTurn(
                reasoning="use mock",
                content=None,
                calls=[AuraToolCall(id="call-stable", name="mock_echo", arguments={"text": "hello"})],
            )
        return ParsedAuraToolTurn(reasoning="done", content="已完成 hello。", calls=[])

    monkeypatch.setattr(
        "vllm_omni.entrypoints.openai.serving_video_stream.parse_aura_tool_output",
        fake_parse,
    )
    engine = FakeEngine()
    executor = AuraToolExecutor()
    handler = AuraStreamingVideoHandler(
        chat_service=SimpleNamespace(enable_auto_tools=True, parser_cls=object()),
        engine_client=engine,
        tool_executor=executor,
        tool_tokenizer_path="/workspace/models/AURA_v2",
    )

    state = _session_state()
    get_or_create_session_history(state.session_id)

    async def fake_preprocess(chat_request):
        preprocessed.append(chat_request)
        info = getattr(chat_request, "additional_information")
        # Stage1 records pending on pass2; this mock has no aura2tts, so the
        # API commit_turn fallback is the commit authority.
        if isinstance(info, dict):
            resume_list = info.get("aura_tool_resume") or []
            resume = resume_list[0] if resume_list else None
            if isinstance(resume, dict):
                record_pending_turn(
                    state.session_id,
                    request_id="logical-request",
                    transcript=str(resume.get("transcript") or ""),
                    video_tuple=None,
                    tool_chain=resume.get("transient_messages"),
                    had_vision=True,
                )
        return info

    handler._preprocess_to_engine_prompt = fake_preprocess  # type: ignore[method-assign]
    state.turn_frame_arrays = [
        np.zeros((8, 8, 3), dtype=np.uint8),
        np.ones((8, 8, 3), dtype=np.uint8),
    ]
    websocket = FakeWebSocket()
    released: list[tuple[str, str]] = []

    async def release_turn_lock(**kwargs):
        released.append((kwargs["request_id"], kwargs["response_text"]))
        handler.on_turn_complete(
            kwargs["message_history"],
            kwargs["user_message"],
            kwargs["response_text"],
            kwargs["request_id"],
        )

    await handler._process_query_engine(
        websocket=websocket,
        config=AuraStreamingVideoSessionConfig(
            model="test",
            tool_mode="auto",
            tool_intent_gate=False,
            cross_turn_penalty=0,
        ),
        frame_buffer=[_b64(_make_jpeg())],
        audio_buffer=bytearray(),
        message_history=state,
        query_text="",
        request_id="logical-request",
        interrupt_event=SimpleNamespace(is_set=lambda: False),
        prewarmed_frames={},
        release_turn_lock=release_turn_lock,
    )

    assert generated_ids == [
        "logical-request-tool-p1",
        "logical-request-tool-p2",
    ]
    assert drained_ids == generated_ids
    assert len(preprocessed) == 2
    first_info = preprocessed[0].additional_information
    second_info = preprocessed[1].additional_information
    assert first_info["aura_session_id"] == second_info["aura_session_id"] == state.session_id
    assert second_info["aura_tool_pass"] == [2]
    assert second_info["aura_tool_resume"][0]["transient_messages"][-1]["role"] == "tool"
    assert first_info["deferred_multi_modal_data"] == second_info["deferred_multi_modal_data"]

    event_types = [event["type"] for event in websocket.events]
    assert event_types == [
        "response.start",
        "user.transcript.done",
        "response.tool.started",
        "response.tool.done",
        "response.text.done",
        "response.audio.done",
        "response.done",
    ]
    transcript_done = next(
        event for event in websocket.events if event["type"] == "user.transcript.done"
    )
    assert transcript_done["text"] == "請回聲 hello"
    tool_done = next(event for event in websocket.events if event["type"] == "response.tool.done")
    assert tool_done["content"] == '{"ok":true,"result":{"text":"hello"}}'
    assert all("<tool_call>" not in str(event) and "<think>" not in str(event) for event in websocket.events)
    assert released == [("logical-request", "已完成 hello。")]
    history = get_session_history(state.session_id)
    assert history is not None
    assert [message["role"] for message in history.history] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert history.history[-3]["tool_calls"][0]["id"] == "call-stable"
    assert history.history[-2]["tool_call_id"] == "call-stable"
    assert history.history[-1]["content"] == "已完成 hello。"


@pytest.mark.asyncio
async def test_tool_intent_gate_retries_visual_question_without_executing(monkeypatch):
    raw_tool = (
        "<tool_call><function=WebSearch><parameter=query>"
        "2024-10-01 人民幣匯率"
        "</parameter></function></tool_call>"
    )

    class FakeEngine:
        async def get_tokenizer(self):
            return object()

    class FakeWebSocket:
        def __init__(self):
            self.events = []

        async def send_json(self, event):
            self.events.append(event)

    def fake_parse(_tokenizer, raw, *, request_id, tool_schemas):
        del request_id, tool_schemas
        if "<tool_call>" in raw:
            return ParsedAuraToolTurn(
                reasoning=None,
                content="",
                calls=[
                    AuraToolCall(
                        id="hallucinated",
                        name="WebSearch",
                        arguments={"query": "2024-10-01 人民幣匯率"},
                    )
                ],
            )
        return ParsedAuraToolTurn(reasoning=None, content="我看到清楚的辦公室畫面。", calls=[])

    monkeypatch.setattr(
        "vllm_omni.entrypoints.openai.serving_video_stream.parse_aura_tool_output",
        fake_parse,
    )
    executor = AuraToolExecutor(mode="safe")

    async def forbidden_execute(**kwargs):
        raise AssertionError(f"intent-gated tool executed: {kwargs}")

    executor.execute = forbidden_execute  # type: ignore[method-assign]
    handler = AuraStreamingVideoHandler(
        chat_service=SimpleNamespace(enable_auto_tools=True, parser_cls=object()),
        engine_client=FakeEngine(),
        tool_executor=executor,
        tool_tokenizer_path="/workspace/models/AURA_v2",
    )
    preprocessed = []

    async def fake_preprocess(chat_request):
        preprocessed.append(getattr(chat_request, "additional_information"))
        return chat_request

    passes = 0

    async def fake_collect(*, on_text_ready, on_transcript, **kwargs):
        nonlocal passes
        del kwargs
        passes += 1
        if passes == 1:
            await on_transcript("你現在看到畫面嗎？")
            await on_text_ready(raw_tool)
            return {
                "interrupted": False,
                "transcript": "你現在看到畫面嗎？",
                "text": raw_tool,
                "preamble_audio_count": 0,
                "audio_deltas": [],
            }
        await on_text_ready("我看到清楚的辦公室畫面。")
        return {
            "interrupted": False,
            "transcript": "",
            "text": "我看到清楚的辦公室畫面。",
            "preamble_audio_count": 0,
            "audio_deltas": [],
        }

    final_texts = []

    async def fake_emit(**kwargs):
        final_texts.append(kwargs["response_text"])

    handler._preprocess_to_engine_prompt = fake_preprocess  # type: ignore[method-assign]
    handler._collect_aura_tool_pass = fake_collect  # type: ignore[method-assign]
    handler._emit_aura_tool_final = fake_emit  # type: ignore[method-assign]

    state = _session_state()
    state.turn_frame_arrays = [np.zeros((8, 8, 3), dtype=np.uint8)]
    get_or_create_session_history(state.session_id)
    websocket = FakeWebSocket()
    await handler._process_query_engine(
        websocket=websocket,
        config=AuraStreamingVideoSessionConfig(
            model="test",
            tool_mode="auto",
            cross_turn_penalty=0,
        ),
        frame_buffer=[_b64(_make_jpeg())],
        audio_buffer=bytearray(),
        message_history=state,
        query_text="",
        request_id="intent-gate",
        interrupt_event=SimpleNamespace(is_set=lambda: False),
        prewarmed_frames={},
    )

    assert passes == 2
    assert preprocessed[0]["aura_tool_enabled"] == [False]
    assert preprocessed[1]["aura_tool_enabled"] == [False]
    assert preprocessed[1]["aura_tool_resume"][0]["transcript"] == "你現在看到畫面嗎？"
    assert not any(event["type"].startswith("response.tool.") for event in websocket.events)
    assert final_texts == ["我看到清楚的辦公室畫面。"]


@pytest.mark.asyncio
async def test_tool_loop_interrupt_drains_without_execution(monkeypatch):
    drained = False

    class FakeEngine:
        async def get_tokenizer(self):
            return object()

        async def generate(self, **kwargs):
            nonlocal drained
            del kwargs
            yield SimpleNamespace(
                final_output_type="text",
                request_output=SimpleNamespace(
                    outputs=[SimpleNamespace(text="<tool_call>", cumulative_text="<tool_call>")]
                ),
            )
            drained = True

    handler = AuraStreamingVideoHandler(
        chat_service=SimpleNamespace(enable_auto_tools=True, parser_cls=object()),
        engine_client=FakeEngine(),
        tool_executor=AuraToolExecutor(),
        tool_tokenizer_path="/workspace/models/AURA_v2",
    )

    async def fake_preprocess(chat_request):
        return chat_request

    handler._preprocess_to_engine_prompt = fake_preprocess  # type: ignore[method-assign]
    state = _session_state()
    state.turn_frame_arrays = [np.zeros((8, 8, 3), dtype=np.uint8)]
    websocket = MagicMock()
    websocket.send_json = MagicMock()

    async def send_json(_event):
        return None

    websocket.send_json = send_json
    await handler._process_query_engine(
        websocket=websocket,
        config=AuraStreamingVideoSessionConfig(
            model="test",
            tool_mode="auto",
            tool_intent_gate=False,
            cross_turn_penalty=0,
        ),
        frame_buffer=[],
        audio_buffer=bytearray(),
        message_history=state,
        query_text="",
        request_id="interrupted",
        interrupt_event=SimpleNamespace(is_set=lambda: True),
        prewarmed_frames={},
    )

    assert drained is True
    assert handler._tool_executor._tasks == {}


@pytest.mark.asyncio
async def test_tool_loop_preamble_audio_overlaps_delayed_tool(monkeypatch):
    from vllm_omni.entrypoints.openai import video_stream_envs

    monkeypatch.setattr(video_stream_envs, "VLLM_VIDEO_ASYNC_CHUNK", "on")
    raw_tool = "我先查一下。<tool_call><function=mock_echo><parameter=text>hello</parameter></function></tool_call>"

    class FakeEngine:
        async def get_tokenizer(self):
            return object()

        async def generate(self, *, prompt, request_id, output_modalities):
            del prompt, output_modalities
            if request_id.endswith("p1"):
                yield SimpleNamespace(
                    final_output_type="transcript",
                    request_output=SimpleNamespace(
                        outputs=[SimpleNamespace(text="查一下", cumulative_text="查一下")]
                    ),
                )
                yield SimpleNamespace(
                    final_output_type="text",
                    finished=True,
                    request_output=SimpleNamespace(
                        outputs=[SimpleNamespace(text=raw_tool, cumulative_text=raw_tool)]
                    ),
                )
                await asyncio.sleep(0.03)
                yield SimpleNamespace(final_output_type="audio", request_output=None)
                return
            yield SimpleNamespace(
                final_output_type="text",
                request_output=SimpleNamespace(
                    outputs=[SimpleNamespace(text="已完成 hello。", cumulative_text="已完成 hello。")]
                ),
            )

    class FakeWebSocket:
        def __init__(self):
            self.events: list[dict[str, Any]] = []
            self.times: list[float] = []

        async def send_json(self, event):
            self.events.append(event)
            self.times.append(time.perf_counter())

    def fake_parse(_tokenizer, raw, *, request_id, tool_schemas):
        del request_id, tool_schemas
        if "<tool_call>" in raw:
            return ParsedAuraToolTurn(
                reasoning=None,
                content="我先查一下。",
                calls=[AuraToolCall(id="call-slow", name="mock_echo", arguments={"text": "hello"})],
            )
        return ParsedAuraToolTurn(reasoning=None, content="已完成 hello。", calls=[])

    monkeypatch.setattr(
        "vllm_omni.entrypoints.openai.serving_video_stream.parse_aura_tool_output",
        fake_parse,
    )
    executor = AuraToolExecutor()
    spec = executor._registry["mock_echo"]

    def slow(arguments):
        time.sleep(0.12)
        return {"text": arguments.text}

    executor._registry["mock_echo"] = replace(spec, handler=slow)
    handler = AuraStreamingVideoHandler(
        chat_service=SimpleNamespace(enable_auto_tools=True, parser_cls=object()),
        engine_client=FakeEngine(),
        tool_executor=executor,
        tool_tokenizer_path="/workspace/models/AURA_v2",
    )
    async def fake_preprocess(chat_request):
        return chat_request

    handler._extract_audio_delta_b64 = lambda _output, drained: ("QUFBQQ==", drained + 1)  # type: ignore[method-assign]
    handler._preprocess_to_engine_prompt = fake_preprocess  # type: ignore[method-assign]

    websocket = FakeWebSocket()
    state = _session_state()
    state.turn_frame_arrays = [np.zeros((8, 8, 3), dtype=np.uint8)]
    await handler._process_query_engine(
        websocket=websocket,
        config=AuraStreamingVideoSessionConfig(
            model="test",
            tool_mode="auto",
            tool_intent_gate=False,
            cross_turn_penalty=0,
        ),
        frame_buffer=[],
        audio_buffer=bytearray(),
        message_history=state,
        query_text="",
        request_id="overlap-request",
        interrupt_event=SimpleNamespace(is_set=lambda: False),
        prewarmed_frames={},
    )

    types = [event["type"] for event in websocket.events]
    assert "response.tool.preamble.text" in types
    assert "response.tool.preamble.audio.delta" in types
    assert "response.tool.preamble.audio.done" in types
    assert types.count("response.text.done") == 1
    preamble = next(event for event in websocket.events if event["type"] == "response.tool.preamble.text")
    assert preamble["text"] == "我先查一下。"
    assert all("<tool_call>" not in str(event) for event in websocket.events)
    audio_first = websocket.times[types.index("response.tool.preamble.audio.delta")]
    tool_done = websocket.times[types.index("response.tool.done")]
    assert audio_first < tool_done


@pytest.mark.asyncio
async def test_tool_loop_async_chunk_off_still_emits_preamble_without_overlap_claim(monkeypatch):
    from vllm_omni.entrypoints.openai import video_stream_envs

    monkeypatch.setattr(video_stream_envs, "VLLM_VIDEO_ASYNC_CHUNK", "off")
    raw_tool = "稍等。<tool_call><function=mock_echo><parameter=text>hello</parameter></function></tool_call>"

    class FakeEngine:
        async def get_tokenizer(self):
            return object()

        async def generate(self, *, prompt, request_id, output_modalities):
            del prompt, output_modalities
            if request_id.endswith("p1"):
                yield SimpleNamespace(
                    final_output_type="text",
                    request_output=SimpleNamespace(
                        outputs=[SimpleNamespace(text=raw_tool, cumulative_text=raw_tool)]
                    ),
                )
                return
            yield SimpleNamespace(
                final_output_type="text",
                request_output=SimpleNamespace(
                    outputs=[SimpleNamespace(text="好。", cumulative_text="好。")]
                ),
            )

    def fake_parse(_tokenizer, raw, *, request_id, tool_schemas):
        del request_id, tool_schemas
        if "<tool_call>" in raw:
            return ParsedAuraToolTurn(
                reasoning=None,
                content="稍等。",
                calls=[AuraToolCall(id="call-off", name="mock_echo", arguments={"text": "hello"})],
            )
        return ParsedAuraToolTurn(reasoning=None, content="好。", calls=[])

    monkeypatch.setattr(
        "vllm_omni.entrypoints.openai.serving_video_stream.parse_aura_tool_output",
        fake_parse,
    )
    websocket_events: list[str] = []

    class FakeWebSocket:
        async def send_json(self, event):
            websocket_events.append(event["type"])

    handler = AuraStreamingVideoHandler(
        chat_service=SimpleNamespace(enable_auto_tools=True, parser_cls=object()),
        engine_client=FakeEngine(),
        tool_executor=AuraToolExecutor(),
        tool_tokenizer_path="/workspace/models/AURA_v2",
    )
    async def fake_preprocess(chat_request):
        return chat_request

    handler._preprocess_to_engine_prompt = fake_preprocess  # type: ignore[method-assign]
    state = _session_state()
    state.turn_frame_arrays = [np.zeros((8, 8, 3), dtype=np.uint8)]
    await handler._process_query_engine(
        websocket=FakeWebSocket(),
        config=AuraStreamingVideoSessionConfig(
            model="test",
            tool_mode="auto",
            tool_intent_gate=False,
            cross_turn_penalty=0,
        ),
        frame_buffer=[],
        audio_buffer=bytearray(),
        message_history=state,
        query_text="",
        request_id="off-request",
        interrupt_event=SimpleNamespace(is_set=lambda: False),
        prewarmed_frames={},
    )
    assert "response.tool.preamble.text" in websocket_events
    assert "response.tool.started" in websocket_events
    assert websocket_events.count("response.text.done") == 1


@pytest.mark.asyncio
async def test_tool_final_without_audio_chunks_still_closes_audio_segment():
    events: list[dict[str, Any]] = []

    class FakeWebSocket:
        async def send_json(self, event):
            events.append(event)

    handler = AuraStreamingVideoHandler(chat_service=object())
    state = _session_state()
    await handler._emit_aura_tool_final(
        websocket=FakeWebSocket(),
        config=AuraStreamingVideoSessionConfig(
            model="test",
            modalities=["text", "audio"],
        ),
        message_history=state,
        user_message={"role": "user", "content": "test"},
        request_id="empty-audio-final",
        response_text="工具呼叫未能完成。",
        audio_deltas=[],
    )

    assert [event["type"] for event in events] == [
        "response.text.done",
        "response.audio.done",
        "response.done",
    ]


@pytest.mark.asyncio
async def test_tool_pass_only_publishes_finished_asr_transcript(monkeypatch):
    from vllm_omni.entrypoints.openai import video_stream_envs

    monkeypatch.setattr(video_stream_envs, "VLLM_VIDEO_ASYNC_CHUNK", "on")

    class FakeEngine:
        async def generate(self, **_kwargs):
            for text, finished in (
                ("language", False),
                ("language zh 請查上海天氣", True),
            ):
                yield SimpleNamespace(
                    final_output_type="transcript",
                    finished=finished,
                    request_output=SimpleNamespace(
                        outputs=[
                            SimpleNamespace(
                                text=text,
                                cumulative_text=text,
                            )
                        ]
                    ),
                )
            yield SimpleNamespace(
                final_output_type="text",
                finished=True,
                request_output=SimpleNamespace(
                    outputs=[
                        SimpleNamespace(
                            text="<tool_call></tool_call>",
                            cumulative_text="<tool_call></tool_call>",
                        )
                    ]
                ),
            )

    published: list[str] = []
    async def publish_transcript(text: str) -> None:
        published.append(text)

    handler = AuraStreamingVideoHandler(
        chat_service=object(),
        engine_client=FakeEngine(),
    )
    collected = await handler._collect_aura_tool_pass(
        config=AuraStreamingVideoSessionConfig(
            model="test",
            modalities=["text"],
        ),
        request_id="asr-finish",
        interrupt_event=SimpleNamespace(is_set=lambda: False),
        engine_prompt={},
        on_transcript=publish_transcript,
    )

    assert published == ["請查上海天氣"]
    assert collected["transcript"] == "請查上海天氣"


def test_canonical_aura_response_text_prefers_decoded_ids():
    streaming = "会被罚款30元。"
    decoded = "会被罚款3000元。"
    assert (
        canonical_aura_response_text(streaming_text=streaming, decoded_text=decoded) == decoded
    )


def test_canonical_aura_response_text_keeps_streaming_when_decode_empty():
    assert canonical_aura_response_text(streaming_text="你好。", decoded_text="") == "你好。"


def test_canonical_aura_response_text_keeps_silent_marker():
    assert (
        canonical_aura_response_text(streaming_text=SILENT_TEXT, decoded_text="会被罚款3000元。")
        == SILENT_TEXT
    )


def test_canonical_aura_response_text_rejects_wrong_vocab_latin_soup():
    streaming = "没看到，画面里就一个戴眼镜的男生在摸头发。"
    decoded = (
        "Trilogy-même getting拥有 democrat departamento widget Nay诗词/utility責任eworld conexao通过对_r"
    )
    assert canonical_aura_response_text(streaming_text=streaming, decoded_text=decoded) == streaming


def test_decode_aura_content_ids_strips_think_wrappers():
    class FakeTokenizer:
        def decode(self, token_ids: list[int]) -> str:
            del token_ids
            return "<think>内部推理</think>会被罚款3000元。"

    assert decode_aura_content_ids_for_display(FakeTokenizer(), [11, 22, 33]) == "会被罚款3000元。"


@pytest.mark.asyncio
async def test_text_done_uses_full_token_decode_when_incremental_detok_drops_digits():
    from vllm_omni.outputs import OmniRequestOutput

    streaming_wrong = "会被罚款30元。"
    decoded_right = "会被罚款3000元。"

    class FakeTokenizer:
        def decode(self, token_ids: list[int]) -> str:
            del token_ids
            return decoded_right

    class FakeEngine:
        def __init__(self) -> None:
            tok = FakeTokenizer()
            self.engine = SimpleNamespace(
                output_processors=[SimpleNamespace(tokenizer=None), SimpleNamespace(tokenizer=tok)]
            )

        async def get_tokenizer(self):
            raise AssertionError("must use Stage-1 tokenizer, not ASR get_tokenizer()")

        async def generate(self, *, prompt, request_id, output_modalities):
            del prompt, request_id, output_modalities
            output = SimpleNamespace(text=streaming_wrong, token_ids=[11, 22, 33])
            yield OmniRequestOutput(
                final_output_type="text",
                request_output=SimpleNamespace(outputs=[output]),
                finished=True,
            )

    events: list[dict[str, Any]] = []

    class FakeWebSocket:
        async def send_json(self, event):
            events.append(event)

    handler = AuraStreamingVideoHandler(
        chat_service=object(),
        engine_client=FakeEngine(),
    )
    await handler._run_engine_generation(
        websocket=FakeWebSocket(),
        config=AuraStreamingVideoSessionConfig(
            model="test",
            modalities=["text"],
            stream_text_deltas=False,
            cross_turn_penalty=0,
        ),
        message_history=_session_state(),
        user_message={"role": "user", "content": "呢张海报系咩嚟？"},
        request_id="poster-30-vs-3000",
        interrupt_event=SimpleNamespace(is_set=lambda: False),
        engine_prompt={"prompt": "x"},
    )
    done = next(event for event in events if event.get("type") == "response.text.done")
    assert done["text"] == decoded_right
