# SPDX-License-Identifier: Apache-2.0
"""Tests for the original-Native-frontend AURA_v2 bridge."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import wave
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from examples.online_serving.aura_omni.native_gateway_web_demo import warmup_aura
from examples.online_serving.aura_omni.native_gateway_web_demo.server import (
    NativeEventTranslator,
    _parse_video_frames,
)
from examples.online_serving.aura_omni.native_gateway_web_demo.server import (
    app as bridge_app,
)

DEMO_ROOT = (
    Path(__file__).resolve().parents[3]
    / "examples/online_serving/aura_omni/native_gateway_web_demo"
)
ORIGINAL_EVAL_SHA256 = "c7b33d9a6c4a9faa09b84c1697895f56a671ea224043a6466f9267e05b867afe"


def test_index_stops_playback_on_ptt() -> None:
    index = (DEMO_ROOT / "static/index.html").read_text(encoding="utf-8")
    assert "function stopPlayback()" in index
    assert "stopPlayback();" in index
    assert "function startPtt()" in index
    start = index.index("function startPtt()")
    assert "stopPlayback();" in index[start : start + 120]


def test_bridge_health_and_index() -> None:
    client = TestClient(bridge_app)
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["model"] == "/workspace/models/AURA_v2"
    assert health.json()["tool_mode"] == "auto"
    assert health.json()["auto_trigger"] is True
    assert health.json()["max_tool_depth"] == 3

    page = client.get("/")
    assert page.status_code == 200
    assert "function stopPlayback()" in page.text
    eval_page = DEMO_ROOT / "static/eval.html"
    assert hashlib.sha256(eval_page.read_bytes()).hexdigest() == ORIGINAL_EVAL_SHA256


def _wav_b64(samples: list[int], sample_rate: int = 24000) -> str:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"".join(sample.to_bytes(2, "little", signed=True) for sample in samples))
    return base64.b64encode(output.getvalue()).decode()


def _decode_envelope(raw: str) -> dict:
    return json.loads(raw)


@pytest.mark.asyncio
async def test_warmup_runs_browser_aligned_silent_turn_before_voice(
    monkeypatch, tmp_path
) -> None:
    wav_path = tmp_path / "input.wav"
    with wave.open(str(wav_path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(b"\0\0" * 1600)

    class FakeWebSocket:
        def __init__(self) -> None:
            self.sent = []
            self.received = [
                {"type": "session.ready"},
                {"type": "response.start", "request_id": "silent"},
                {
                    "type": "response.text.done",
                    "request_id": "silent",
                    "text": "<|silent|>",
                },
                {"type": "response.start", "request_id": "voice"},
                {
                    "type": "response.text.done",
                    "request_id": "voice",
                    "text": "沒問題。",
                },
                {
                    "type": "response.audio.delta",
                    "request_id": "voice",
                    "data": "d2F2",
                },
                {"type": "response.audio.done", "request_id": "voice"},
            ]

        async def send(self, payload: str) -> None:
            self.sent.append(json.loads(payload))

        async def recv(self) -> str:
            return json.dumps(self.received.pop(0))

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

    websocket = FakeWebSocket()
    monkeypatch.setattr(
        warmup_aura.websockets,
        "connect",
        lambda *_args, **_kwargs: websocket,
    )
    result = await warmup_aura._warmup(
        aura_ws="ws://test",
        model="model",
        wav=wav_path,
        tts_speaker="Vivian",
        tts_language="Chinese",
        tts_instruct="",
        tts_task_type="Base",
        timeout_s=5,
        frame_count=2,
        frame_width=640,
        frame_height=360,
        silent_first=True,
    )

    assert result["ok"] is True
    assert result["silent_text"] == "<|silent|>"
    assert result["text"] == "沒問題。"
    assert [message["type"] for message in websocket.sent[:3]] == [
        "session.config",
        "video.frame",
        "video.frame",
    ]
    assert websocket.sent[0]["auto_trigger"] is True


def test_stack_scripts_require_browser_aligned_warmup() -> None:
    for name in ("run_1gpu_stack.sh", "run_2gpu_stack.sh"):
        script = (DEMO_ROOT / name).read_text(encoding="utf-8")
        warmup_call = script[script.index('"$PYTHON_BIN" "$SCRIPT_DIR/warmup_aura.py"') :]
        assert '--frame-count "${WARMUP_FRAME_COUNT:-2}"' in warmup_call
        assert '--frame-width "${WARMUP_FRAME_WIDTH:-640}"' in warmup_call
        assert '--frame-height "${WARMUP_FRAME_HEIGHT:-360}"' in warmup_call
        assert "--silent-first" in warmup_call
        assert warmup_call.index("--silent-first") < warmup_call.index("nohup env")


def test_native_video_batch_splits_into_omni_frames() -> None:
    assert _parse_video_frames("data:video/jpeg;base64,one,two,three") == [
        "one",
        "two",
        "three",
    ]


def test_translator_buffers_audio_into_one_native_wav_and_one_turn_done() -> None:
    translator = NativeEventTranslator()
    session_id = "session"
    request_id = "request"

    text_events = translator.observe(
        {"type": "response.text.done", "request_id": request_id, "text": "你好。"},
        session_id=session_id,
    )
    assert [_decode_envelope(item)["type"] for item in text_events] == ["text"]

    for samples in ([1, 2], [3, 4, 5]):
        assert (
            translator.observe(
                {
                    "type": "response.audio.delta",
                    "request_id": request_id,
                    "data": _wav_b64(samples),
                },
                session_id=session_id,
            )
            == []
        )

    done_events = [
        _decode_envelope(item)
        for item in translator.observe(
            {"type": "response.audio.done", "request_id": request_id},
            session_id=session_id,
        )
    ]
    assert [event["type"] for event in done_events] == ["audio", "turn_done"]
    raw = base64.b64decode(done_events[0]["data"]["audio"])
    with wave.open(io.BytesIO(raw), "rb") as wav_file:
        assert wav_file.getframerate() == 24000
        assert wav_file.getnframes() == 5

    assert (
        translator.observe(
            {"type": "response.audio.done", "request_id": request_id},
            session_id=session_id,
        )
        == []
    )


def test_translator_silent_turn_finishes_once_without_audio() -> None:
    translator = NativeEventTranslator()
    event = {"type": "response.text.done", "request_id": "silent", "text": "<|silent|>"}
    first = [_decode_envelope(item) for item in translator.observe(event, session_id="s")]
    assert [item["type"] for item in first] == ["turn_done"]
    assert (
        translator.observe(
            {"type": "response.audio.done", "request_id": "silent"},
            session_id="s",
        )
        == []
    )


def test_translator_preamble_audio_does_not_send_turn_done() -> None:
    translator = NativeEventTranslator()
    session_id = "session"
    request_id = "request"

    preamble_text = [
        _decode_envelope(item)
        for item in translator.observe(
            {"type": "response.tool.preamble.text", "request_id": request_id, "text": "我先查一下。"},
            session_id=session_id,
        )
    ]
    assert [event["type"] for event in preamble_text] == ["text"]

    translator.observe(
        {
            "type": "response.tool.preamble.audio.delta",
            "request_id": request_id,
            "data": _wav_b64([1, 2, 3]),
        },
        session_id=session_id,
    )
    preamble_audio = [
        _decode_envelope(item)
        for item in translator.observe(
            {"type": "response.tool.preamble.audio.done", "request_id": request_id},
            session_id=session_id,
        )
    ]
    assert [event["type"] for event in preamble_audio] == ["audio"]

    translator.observe(
        {"type": "response.tool.started", "request_id": request_id, "call_id": "c", "name": "calculator"},
        session_id=session_id,
    )
    translator.observe(
        {
            "type": "response.text.done",
            "request_id": request_id,
            "text": "結果是 703。",
        },
        session_id=session_id,
    )
    translator.observe(
        {
            "type": "response.audio.delta",
            "request_id": request_id,
            "data": _wav_b64([4, 5]),
        },
        session_id=session_id,
    )
    final_events = [
        _decode_envelope(item)
        for item in translator.observe(
            {"type": "response.audio.done", "request_id": request_id},
            session_id=session_id,
        )
    ]
    assert [event["type"] for event in final_events] == ["audio", "turn_done"]
    assert (
        translator.observe(
            {"type": "response.audio.done", "request_id": request_id},
            session_id=session_id,
        )
        == []
    )


def test_translator_maps_safe_tool_result_to_original_tool_ui() -> None:
    translator = NativeEventTranslator()
    started = _decode_envelope(
        translator.observe(
            {
                "type": "response.tool.started",
                "request_id": "r",
                "call_id": "c",
                "name": "calculator",
            },
            session_id="s",
        )[0]
    )
    assert started["data"] == {
        "status": "started",
        "id": "c",
        "name": "calculator",
    }

    completed = _decode_envelope(
        translator.observe(
            {
                "type": "response.tool.done",
                "request_id": "r",
                "call_id": "c",
                "name": "calculator",
                "status": "completed",
                "content": '{"ok":true,"result":{"summary":"37 * 19 = 703"}}',
            },
            session_id="s",
        )[0]
    )
    assert completed["data"] == {
        "status": "success",
        "id": "c",
        "name": "calculator",
        "output": "37 * 19 = 703",
    }


def test_translator_barge_in_drops_pending_tts() -> None:
    translator = NativeEventTranslator()
    request_id = "old-turn"
    translator.observe(
        {"type": "response.text.done", "request_id": request_id, "text": "先聽這段。"},
        session_id="s",
    )
    translator.observe(
        {
            "type": "response.audio.delta",
            "request_id": request_id,
            "data": _wav_b64([1, 2, 3, 4]),
        },
        session_id="s",
    )
    translator.barge_in()
    leftover = translator.observe(
        {"type": "response.audio.done", "request_id": request_id},
        session_id="s",
    )
    types = [_decode_envelope(item)["type"] for item in leftover]
    assert types == ["turn_done"]
    assert "audio" not in types
