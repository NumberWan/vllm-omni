# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the AURA MiniCPM-style realtime web demo shell."""

from __future__ import annotations

import base64
import io
import json
import wave
from array import array
from pathlib import Path

from fastapi.testclient import TestClient

from examples.online_serving.aura_omni.minicpm_style_web_demo.server import (
    DEFAULT_WARMUP_AUDIO,
    STREAM_PATH,
    _TtsTurnDumper,
    build_app,
    join_ws_url,
)
from examples.online_serving.aura_omni.minicpm_style_web_demo.verify_e2e import (
    _make_warmup_frame,
    _wav_to_pcm16_16k,
)

APP_DIR = (
    Path(__file__).resolve().parents[3]
    / "examples/online_serving/aura_omni/minicpm_style_web_demo/app"
)


def test_join_ws_url_combines_path_and_query() -> None:
    assert join_ws_url("ws://127.0.0.1:8666", STREAM_PATH) == (
        "ws://127.0.0.1:8666/v1/video/chat/stream"
    )
    assert join_ws_url("ws://127.0.0.1:8666/", STREAM_PATH, "x=1") == (
        "ws://127.0.0.1:8666/v1/video/chat/stream?x=1"
    )


def test_required_static_assets_exist() -> None:
    for relative in (
        "index.html",
        "static/app.js",
        "static/styles.css",
        "static/pcm_worklet.js",
        "static/playback_worklet.js",
    ):
        assert (APP_DIR / relative).is_file(), relative


def test_bundled_warmup_assets_are_self_contained() -> None:
    assert DEFAULT_WARMUP_AUDIO.is_file()
    assert _make_warmup_frame().startswith(b"\xff\xd8")
    assert _wav_to_pcm16_16k(DEFAULT_WARMUP_AUDIO)


def test_tts_dump_uses_request_id_when_silent_turn_starts_during_audio(tmp_path: Path) -> None:
    pcm = array("h", [0, 1000, -1000, 0])
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)
        wav_file.writeframes(pcm.tobytes())
    audio_b64 = base64.b64encode(wav_buffer.getvalue()).decode()

    dumper = _TtsTurnDumper(tmp_path)

    def observe(event: dict[str, str]) -> None:
        dumper.observe(json.dumps(event))

    observe({"type": "response.start", "request_id": "spoken-1"})
    observe({"type": "response.text.done", "request_id": "spoken-1", "text": "你好。"})
    observe({"type": "response.start", "request_id": "silent-2"})
    observe({"type": "response.text.done", "request_id": "silent-2", "text": "<|silent|>"})
    observe(
        {
            "type": "response.audio.delta",
            "request_id": "spoken-1",
            "data": audio_b64,
        }
    )
    observe(
        {
            "type": "response.audio.delta",
            "request_id": "spoken-1",
            "data": audio_b64,
        }
    )
    observe({"type": "response.audio.done", "request_id": "spoken-1"})

    turn_dirs = list(tmp_path.glob("turn_*"))
    assert len(turn_dirs) == 1
    assert "spoken-1" in turn_dirs[0].name
    assert (turn_dirs[0] / "request_id.txt").read_text().strip() == "spoken-1"
    assert (turn_dirs[0] / "text.txt").read_text().strip() == "你好。"
    assert len(list(turn_dirs[0].glob("chunk_*.wav"))) == 2
    assert (turn_dirs[0] / "merged.wav").is_file()
    assert "silent-2" not in dumper.turns


def test_healthz_and_index_config_injection() -> None:
    app = build_app(
        ws_backend="ws://127.0.0.1:8666",
        model="aurateam/AURA",
        video_fps=2.0,
        tts_language="Chinese",
        tts_speaker="Vivian",
        tool_mode="auto",
        listen_port=7862,
    )
    assert app.state.listen_port == 7862
    client = TestClient(app)

    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.text == "ok"

    page = client.get("/")
    assert page.status_code == 200
    assert "AURA Streaming Voice" in page.text
    assert "__AURA_WEB_CONFIG__" not in page.text
    assert "aurateam/AURA" in page.text
    assert "v1/video/chat/stream" in page.text
    assert "AURA_WEB_CONFIG" in page.text
    assert '"toolMode": "auto"' in page.text

    app_js = client.get("/static/app.js")
    assert app_js.status_code == 200
    body = app_js.text
    assert "session.config" in body
    assert "tool_mode: config.toolMode === 'auto' ? 'auto' : 'none'" in body
    assert "max_tool_depth: 2" in body
    assert "auto_trigger: true" in body
    assert "audio.done" in body
    assert "video.frame" in body
    assert "user.transcript.done" in body
    assert "function clearPlayback()" in body
    assert "camera streaming on" in body
    assert "Do not clearPlayback here" in body
    assert "bindPlaybackRequest" in body
    assert "Hold to talk" in page.text
    assert "playback.ack" not in body
    assert "response.cancel" not in body
