#!/usr/bin/env python3
"""Serve the original AURA demo frontend against AURA_v2 Omni.

Serves ``gateway/static`` from AURA_demo and translates:
  frontend ``/ws``  (video / audio / text JSON)
    ↔
  AURA ``/v1/video/chat/stream`` (session.config / video.frame / audio.chunk)

The files under ``static/`` are vendored from AURA_demo with a small PTT
playback-stop patch so barge-in can mute leftover TTS.

Env:
  AURA_WS_URL, AURA_MODEL, BRIDGE_HOST, BRIDGE_PORT, STATIC_DIR
  TTS_SPEAKER (Vivian), TTS_INSTRUCT (empty), TTS_LANGUAGE (Chinese),
  TTS_TASK_TYPE (Base), TOOL_MODE (auto), MAX_TOOL_DEPTH (3),
  AURA_TTS_DUMP_DIR (/tmp/aura_v2_native_demo_tts)
"""

from __future__ import annotations

import asyncio
import audioop
import base64
import contextlib
import io
import json
import logging
import os
import time
import wave
from array import array
from dataclasses import dataclass, field
from pathlib import Path

import uvicorn
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

LOG = logging.getLogger("aura_v2_native_bridge")

ROOT = Path(__file__).resolve().parent
DEFAULT_STATIC = ROOT / "static"
STATIC_DIR = Path(os.environ.get("STATIC_DIR", str(DEFAULT_STATIC))).expanduser()
AURA_WS_URL = os.environ.get("AURA_WS_URL", "ws://127.0.0.1:8666/v1/video/chat/stream")
AURA_MODEL = os.environ.get("AURA_MODEL", "/workspace/models/AURA_v2")
BRIDGE_HOST = os.environ.get("BRIDGE_HOST", "0.0.0.0")
BRIDGE_PORT = int(os.environ.get("BRIDGE_PORT", "9999"))

TTS_SPEAKER = os.environ.get("TTS_SPEAKER", "Vivian")
TTS_INSTRUCT = os.environ.get(
    "TTS_INSTRUCT",
    "请用专业、清晰、自然的语气说话，语速稍快，情绪克制，避免夸张和过度热情。",
)
TTS_LANGUAGE = os.environ.get("TTS_LANGUAGE", "Chinese")
TTS_TASK_TYPE = os.environ.get("TTS_TASK_TYPE", "Base")
TOOL_MODE = os.environ.get("TOOL_MODE", "auto")
MAX_TOOL_DEPTH = int(os.environ.get("MAX_TOOL_DEPTH", "3"))
AUTO_TRIGGER = os.environ.get("AUTO_TRIGGER", "1").strip().lower() in {"1", "true", "yes", "on"}
TTS_DUMP_DIR = Path(os.environ.get("AURA_TTS_DUMP_DIR", "/tmp/aura_v2_native_demo_tts")).expanduser()
FRAME_DUMP_DIR = Path(
    os.environ.get("AURA_FRAME_DUMP_DIR", "/tmp/aura_v2_native_demo/frames")
).expanduser()
FRAME_DUMP_ENABLED = os.environ.get("AURA_FRAME_DUMP", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_frame_dump_idx = 0

app = FastAPI(title="AURA_v2 Omni ↔ original AURA demo bridge")


def _dump_incoming_frames(frames: list[str]) -> None:
    """Save a few recent client JPEGs for vision debugging (capped)."""
    global _frame_dump_idx
    if not FRAME_DUMP_ENABLED or not frames:
        return
    FRAME_DUMP_DIR.mkdir(parents=True, exist_ok=True)
    for fr in frames:
        _frame_dump_idx += 1
        path = FRAME_DUMP_DIR / f"frame_{_frame_dump_idx:06d}.jpg"
        try:
            raw = base64.b64decode(fr)
            path.write_bytes(raw)
        except Exception as exc:
            LOG.warning("frame dump failed: %s", exc)
            continue
        # Keep only the newest ~40 files.
        if _frame_dump_idx % 10 == 0:
            old = sorted(FRAME_DUMP_DIR.glob("frame_*.jpg"))
            for stale in old[:-40]:
                stale.unlink(missing_ok=True)


def _tts_config() -> dict:
    return {
        "tts_speaker": TTS_SPEAKER,
        "tts_instruct": TTS_INSTRUCT,
        "tts_language": TTS_LANGUAGE,
        "tts_task_type": TTS_TASK_TYPE,
        "sentence_tts_env": os.environ.get("VLLM_AURA_SENTENCE_TTS", "(server-side)"),
        "dump_dir": str(TTS_DUMP_DIR),
    }


@app.get("/health")
async def health():
    return {
        "ok": True,
        "aura_ws": AURA_WS_URL,
        "model": AURA_MODEL,
        "static": str(STATIC_DIR),
        "tool_mode": TOOL_MODE,
        "max_tool_depth": MAX_TOOL_DEPTH,
        "auto_trigger": AUTO_TRIGGER,
        **_tts_config(),
    }


_NO_CACHE = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
}


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html", headers=_NO_CACHE)


@app.get("/eval")
async def eval_page():
    path = STATIC_DIR / "eval.html"
    if path.exists():
        return FileResponse(path)
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/question_pages")
async def question_pages():
    path = STATIC_DIR / "question_pages.html"
    if path.exists():
        return FileResponse(path)
    return FileResponse(STATIC_DIR / "index.html")


if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _envelope(msg_type: str, data: dict, *, session_id: str | None = None) -> str:
    out: dict = {"type": msg_type, "data": data}
    if session_id:
        out["session_id"] = session_id
    return json.dumps(out, ensure_ascii=False)


def _parse_video_frames(video_url: str) -> list[str]:
    """Split ``data:video/jpeg;base64,<b64>,<b64>,...`` into per-frame b64."""
    raw = (video_url or "").strip()
    if not raw:
        return []
    marker = "base64,"
    idx = raw.find(marker)
    payload = raw[idx + len(marker) :] if idx >= 0 else raw
    return [p for p in payload.split(",") if p.strip()]


def _wav_b64_to_pcm16_16k(audio_b64: str) -> bytes:
    raw = base64.b64decode(audio_b64)
    with wave.open(io.BytesIO(raw), "rb") as wf:
        nch = wf.getnchannels()
        sw = wf.getsampwidth()
        sr = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    if sw != 2:
        raise ValueError(f"expected 16-bit wav, got sampwidth={sw}")
    pcm = frames
    if nch == 2:
        pcm = audioop.tomono(pcm, 2, 0.5, 0.5)
    if sr != 16000:
        pcm, _ = audioop.ratecv(pcm, 2, 1, sr, 16000, None)
    return pcm


def _wav_bytes_to_pcm16(raw: bytes) -> tuple[bytes, int]:
    try:
        with wave.open(io.BytesIO(raw), "rb") as wf:
            if wf.getsampwidth() != 2:
                return b"", 0
            nch = wf.getnchannels()
            rate = wf.getframerate()
            pcm = array("h")
            pcm.frombytes(wf.readframes(wf.getnframes()))
    except Exception:
        return b"", 0
    if nch == 2:
        pcm = array("h", [(pcm[i] + pcm[i + 1]) // 2 for i in range(0, len(pcm), 2)])
    return pcm.tobytes(), int(rate)


def _write_pcm16_wav(path: Path, pcm: bytes, sample_rate: int) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)


def _pcm16_wav_b64(pcm: bytes, sample_rate: int) -> str:
    output = io.BytesIO()
    with wave.open(output, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return base64.b64encode(output.getvalue()).decode()


@dataclass
class _TtsDumpTurn:
    request_id: str
    turn_dir: Path | None = None
    chunk_idx: int = 0
    text: str = ""
    transcript: str = ""
    pcm_chunks: list[bytes] = field(default_factory=list)
    sample_rate: int = 24000
    silent: bool = False


class _TtsTurnDumper:
    """Save spoken AURA audio by request ID (same idea as minicpm_style_web_demo)."""

    _LEGACY = "legacy"

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.turns: dict[str, _TtsDumpTurn] = {}

    def observe(self, event: dict) -> None:
        event_type = event.get("type")
        request_id = str(event.get("request_id") or self._LEGACY)
        if event_type == "response.start":
            self.turns[request_id] = _TtsDumpTurn(request_id=request_id)
            return

        turn = self.turns.setdefault(request_id, _TtsDumpTurn(request_id=request_id))
        if event_type == "user.transcript.done":
            turn.transcript = str(event.get("text") or "")
            if turn.turn_dir is not None and turn.transcript:
                (turn.turn_dir / "transcript.txt").write_text(turn.transcript + "\n", encoding="utf-8")
        elif event_type == "response.text.done":
            turn.text = str(event.get("text") or "")
            if "<|silent|>" in turn.text:
                turn.silent = True
                self._discard(turn)
                self.turns.pop(request_id, None)
            elif turn.turn_dir is not None:
                (turn.turn_dir / "text.txt").write_text(turn.text + "\n", encoding="utf-8")
        elif event_type == "response.audio.delta":
            if not turn.silent:
                self._save_audio_delta(turn, event)
        elif event_type == "response.audio.done":
            self.turns.pop(request_id, None)
            if not turn.silent:
                self._finish(turn)

    @staticmethod
    def _discard(turn: _TtsDumpTurn) -> None:
        if turn.turn_dir is not None and turn.turn_dir.exists():
            for path in turn.turn_dir.glob("*"):
                path.unlink(missing_ok=True)
            turn.turn_dir.rmdir()
        turn.turn_dir = None
        turn.chunk_idx = 0
        turn.pcm_chunks = []

    def _ensure_dir(self, turn: _TtsDumpTurn) -> Path:
        if turn.turn_dir is None:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in turn.request_id)
            turn.turn_dir = self.root / f"turn_{stamp}_{safe}"
            turn.turn_dir.mkdir(parents=True, exist_ok=True)
            (turn.turn_dir / "request_id.txt").write_text(turn.request_id + "\n", encoding="utf-8")
            LOG.info("TTS dump dir: %s", turn.turn_dir)
            if turn.text:
                (turn.turn_dir / "text.txt").write_text(turn.text + "\n", encoding="utf-8")
            if turn.transcript:
                (turn.turn_dir / "transcript.txt").write_text(turn.transcript + "\n", encoding="utf-8")
        return turn.turn_dir

    def _save_audio_delta(self, turn: _TtsDumpTurn, event: dict) -> None:
        data_b64 = event.get("data") or event.get("delta") or ""
        if not data_b64:
            return
        try:
            raw = base64.b64decode(data_b64, validate=True)
        except Exception:
            return
        turn_dir = self._ensure_dir(turn)
        (turn_dir / f"chunk_{turn.chunk_idx:02d}.wav").write_bytes(raw)
        turn.chunk_idx += 1
        pcm, rate = _wav_bytes_to_pcm16(raw)
        if pcm:
            turn.pcm_chunks.append(pcm)
            turn.sample_rate = rate or turn.sample_rate

    @staticmethod
    def _finish(turn: _TtsDumpTurn) -> None:
        if turn.turn_dir is None or not turn.pcm_chunks:
            return
        merged = turn.turn_dir / "merged.wav"
        _write_pcm16_wav(merged, b"".join(turn.pcm_chunks), turn.sample_rate)
        if turn.text:
            (turn.turn_dir / "text.txt").write_text(turn.text + "\n", encoding="utf-8")
        if turn.transcript:
            (turn.turn_dir / "transcript.txt").write_text(turn.transcript + "\n", encoding="utf-8")
        LOG.info(
            "TTS dump complete: %s (%d chunks, text=%r)",
            merged,
            turn.chunk_idx,
            turn.text[:60],
        )


@dataclass
class _NativeTurn:
    request_id: str
    text_buf: str = ""
    audio_chunks: list[bytes] = field(default_factory=list)
    sample_rate: int = 24000
    silent: bool = False
    done_sent: bool = False
    suppress_audio: bool = False


class NativeEventTranslator:
    """Translate AURA stream events into the original Native frontend protocol."""

    _LEGACY = "legacy"

    def __init__(self) -> None:
        self.turns: dict[str, _NativeTurn] = {}

    def barge_in(self) -> None:
        """Drop in-flight TTS so a new PTT is not followed by leftover audio."""
        for turn in self.turns.values():
            turn.audio_chunks.clear()
            turn.suppress_audio = True

    def _turn(self, event: dict) -> _NativeTurn:
        request_id = str(event.get("request_id") or self._LEGACY)
        return self.turns.setdefault(request_id, _NativeTurn(request_id=request_id))

    @staticmethod
    def _tool_output(content: object) -> str:
        if content is None:
            return ""
        raw = str(content)
        try:
            payload = json.loads(raw)
        except (TypeError, ValueError):
            return raw
        if not isinstance(payload, dict):
            return raw
        result = payload.get("result")
        if isinstance(result, dict) and result.get("summary"):
            return str(result["summary"])
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _done(turn: _NativeTurn, session_id: str) -> list[str]:
        if turn.done_sent:
            return []
        turn.done_sent = True
        return [_envelope("turn_done", {}, session_id=session_id)]

    def observe(self, event: dict, *, session_id: str) -> list[str]:
        event_type = str(event.get("type") or "")
        turn = self._turn(event)
        output: list[str] = []

        if event_type == "response.start":
            self.turns[turn.request_id] = _NativeTurn(request_id=turn.request_id)
        elif event_type == "user.transcript.done":
            text = str(event.get("text") or "").strip()
            if text:
                output.append(_envelope("asr_text", {"text": text}, session_id=session_id))
        elif event_type == "response.text.delta":
            turn.text_buf += str(event.get("delta") or "")
        elif event_type == "response.tool.preamble.text":
            text = str(event.get("text") or "").strip()
            if text:
                output.append(_envelope("text", {"text": text}, session_id=session_id))
        elif event_type == "response.text.done":
            text = str(event.get("text") or turn.text_buf).strip()
            turn.text_buf = ""
            if not text or "<|silent|>" in text:
                turn.silent = True
                output.extend(self._done(turn, session_id))
            else:
                output.append(_envelope("text", {"text": text}, session_id=session_id))
        elif event_type in {"response.audio.delta", "response.tool.preamble.audio.delta"}:
            if turn.suppress_audio:
                return output
            data = str(event.get("data") or event.get("delta") or "")
            if data:
                try:
                    pcm, sample_rate = _wav_bytes_to_pcm16(base64.b64decode(data, validate=True))
                except Exception:
                    pcm, sample_rate = b"", 0
                if pcm:
                    if turn.audio_chunks and sample_rate != turn.sample_rate:
                        LOG.warning(
                            "dropping mismatched TTS chunk request=%s expected=%d got=%d",
                            turn.request_id,
                            turn.sample_rate,
                            sample_rate,
                        )
                    else:
                        turn.sample_rate = sample_rate or turn.sample_rate
                        turn.audio_chunks.append(pcm)
        elif event_type == "response.tool.preamble.audio.done":
            if turn.suppress_audio:
                turn.audio_chunks.clear()
            elif turn.audio_chunks and not turn.silent:
                output.append(
                    _envelope(
                        "audio",
                        {
                            "audio": _pcm16_wav_b64(b"".join(turn.audio_chunks), turn.sample_rate),
                            "sample_rate": turn.sample_rate,
                        },
                        session_id=session_id,
                    )
                )
                turn.audio_chunks.clear()
        elif event_type == "response.audio.done":
            if turn.done_sent:
                return []
            if turn.suppress_audio:
                turn.audio_chunks.clear()
            elif turn.audio_chunks and not turn.silent:
                output.append(
                    _envelope(
                        "audio",
                        {
                            "audio": _pcm16_wav_b64(b"".join(turn.audio_chunks), turn.sample_rate),
                            "sample_rate": turn.sample_rate,
                        },
                        session_id=session_id,
                    )
                )
                turn.audio_chunks.clear()
            output.extend(self._done(turn, session_id))
        elif event_type == "response.tool.started":
            output.append(
                _envelope(
                    "toolcall",
                    {
                        "status": "started",
                        "id": str(event.get("call_id") or ""),
                        "name": str(event.get("name") or ""),
                    },
                    session_id=session_id,
                )
            )
        elif event_type == "response.tool.done":
            completed = event.get("status") == "completed"
            data = {
                "status": "success" if completed else "error",
                "id": str(event.get("call_id") or ""),
                "name": str(event.get("name") or ""),
            }
            content = self._tool_output(event.get("content"))
            if content:
                data["output" if completed else "error"] = content
            output.append(_envelope("toolcall", data, session_id=session_id))
        elif event_type == "error":
            output.append(
                _envelope(
                    "error",
                    {"message": str(event.get("message") or event.get("error") or event)},
                    session_id=session_id,
                )
            )
            output.extend(self._done(turn, session_id))
        return output


async def _send_client(ws: WebSocket, payload: str) -> None:
    await ws.send_text(payload)


async def _bridge_session(client: WebSocket) -> None:
    session_id = "unknown"
    dumper = _TtsTurnDumper(TTS_DUMP_DIR)
    translator = NativeEventTranslator()
    await client.accept()

    try:
        async with websockets.connect(AURA_WS_URL, max_size=32 * 1024 * 1024) as aura:
            await aura.send(
                json.dumps(
                    {
                        "type": "session.config",
                        "model": AURA_MODEL,
                        "modalities": ["text", "audio"],
                        "auto_trigger": AUTO_TRIGGER,
                        "auto_trigger_min_frames": 2,
                        "max_frames": 256,
                        "max_frames_per_round": 16,
                        "video_fps": 2.0,
                        "stream_text_deltas": False,
                        "enable_frame_filter": False,
                        "tts_task_type": TTS_TASK_TYPE,
                        "tts_language": TTS_LANGUAGE,
                        "tts_speaker": TTS_SPEAKER,
                        "tts_instruct": TTS_INSTRUCT,
                        "tool_mode": TOOL_MODE,
                        "max_tool_depth": MAX_TOOL_DEPTH,
                    }
                )
            )
            LOG.info(
                "connected to AURA %s speaker=%s instruct=%r lang=%s",
                AURA_WS_URL,
                TTS_SPEAKER,
                TTS_INSTRUCT,
                TTS_LANGUAGE,
            )

            async def client_to_aura() -> None:
                nonlocal session_id
                while True:
                    raw = await client.receive_text()
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    session_id = str(msg.get("session_id") or session_id)
                    mtype = str(msg.get("type") or "")
                    data = msg.get("data") if isinstance(msg.get("data"), dict) else {}

                    if mtype == "video":
                        frames = _parse_video_frames(str(data.get("video_url") or ""))
                        _dump_incoming_frames(frames)
                        for fr in frames:
                            await aura.send(json.dumps({"type": "video.frame", "data": fr}))
                        text = str(data.get("text") or "").strip()
                        if text:
                            LOG.info("ignoring video-attached text (len=%d)", len(text))

                    elif mtype == "audio":
                        translator.barge_in()
                        b64 = str(data.get("audio_b64") or "")
                        if not b64:
                            continue
                        try:
                            pcm = _wav_b64_to_pcm16_16k(b64)
                        except Exception as exc:
                            await _send_client(
                                client,
                                _envelope(
                                    "error",
                                    {"message": f"audio decode failed: {exc}"},
                                    session_id=session_id,
                                ),
                            )
                            continue
                        chunk = 16000 * 2  # 1s
                        for off in range(0, len(pcm), chunk):
                            await aura.send(
                                json.dumps(
                                    {
                                        "type": "audio.chunk",
                                        "data": base64.b64encode(pcm[off : off + chunk]).decode(),
                                    }
                                )
                            )
                        await aura.send(json.dumps({"type": "audio.done"}))

                    elif mtype == "text":
                        await _send_client(
                            client,
                            _envelope(
                                "error",
                                {"message": "AURA_v2 Omni typed text is unavailable; use hold-to-talk"},
                                session_id=session_id,
                            ),
                        )

            async def aura_to_client() -> None:
                async for raw in aura:
                    try:
                        ev = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(ev, dict):
                        dumper.observe(ev)
                    for payload in translator.observe(ev, session_id=session_id):
                        await _send_client(client, payload)

            done, pending = await asyncio.wait(
                [
                    asyncio.create_task(client_to_aura(), name="client_to_aura"),
                    asyncio.create_task(aura_to_client(), name="aura_to_client"),
                ],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending:
                t.cancel()
            for t in done:
                exc = t.exception()
                if exc and not isinstance(exc, (WebSocketDisconnect, asyncio.CancelledError)):
                    LOG.exception("bridge task failed: %s", exc)
                    try:
                        await _send_client(
                            client,
                            _envelope("error", {"message": str(exc)}, session_id=session_id),
                        )
                    except Exception:
                        pass
    except WebSocketDisconnect:
        LOG.info("client disconnected")
    except Exception as exc:
        LOG.exception("bridge session error: %s", exc)
        try:
            await _send_client(client, _envelope("error", {"message": str(exc)}))
        except Exception:
            pass
    finally:
        with contextlib.suppress(Exception):
            await client.close()


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await _bridge_session(websocket)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if not STATIC_DIR.is_dir():
        raise SystemExit(f"STATIC_DIR missing: {STATIC_DIR}")
    TTS_DUMP_DIR.mkdir(parents=True, exist_ok=True)
    LOG.info(
        "static=%s aura=%s listen=%s:%s speaker=%s dump=%s",
        STATIC_DIR,
        AURA_WS_URL,
        BRIDGE_HOST,
        BRIDGE_PORT,
        TTS_SPEAKER,
        TTS_DUMP_DIR,
    )
    uvicorn.run(app, host=BRIDGE_HOST, port=BRIDGE_PORT, log_level="info")


if __name__ == "__main__":
    main()
