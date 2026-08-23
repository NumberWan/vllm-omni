# SPDX-License-Identifier: Apache-2.0
"""Tests for the original-Native-frontend AURA_v2 bridge."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import wave
from pathlib import Path

from fastapi.testclient import TestClient

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
