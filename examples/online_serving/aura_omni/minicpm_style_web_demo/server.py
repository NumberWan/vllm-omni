# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Static host and same-origin WebSocket proxy for the AURA browser demo."""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import hashlib
import json
import logging
import os
import time
import wave
from array import array
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlencode

import uvicorn
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger(__name__)
APP_DIR = Path(__file__).parent / "app"
STATIC_DIR = APP_DIR / "static"
STREAM_PATH = "/v1/video/chat/stream"
REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_WARMUP_AUDIO = REPO_ROOT / "tests/assets/glm_tts/jiayan_zh.wav"
TTS_DUMP_DIR = Path(os.environ.get("AURA_TTS_DUMP_DIR", "/tmp/aura_web_demo_tts"))
DEFAULT_TTS_INSTRUCT = ""


def join_ws_url(base: str, path: str, query: str = "") -> str:
    """Join a WebSocket base URL with an absolute path and optional query."""
    return base.rstrip("/") + path + (("?" + query) if query else "")


def local_open_url(port: int) -> str:
    """URL to open from Cursor Remote SSH / local browser via port forward."""
    return f"http://127.0.0.1:{int(port)}/"


def announce_ready(port: int) -> None:
    """Print a terminal hyperlink that Cursor can Ctrl/Cmd-click."""
    url = local_open_url(port)
    down = "↓" * 52
    up = "↑" * 52
    # OSC-8 hyperlink (clickable in Cursor / VS Code terminal).
    print(f"\n{down}", flush=True)
    print(down, flush=True)
    print(f"\033]8;;{url}\033\\>>> Demo ready — click to open: {url}\033]8;;\033\\", flush=True)
    print(up, flush=True)
    print(up, flush=True)
    print(
        ">>> Tip: in Cursor Remote SSH, Ctrl/Cmd-click the link "
        "(or use Ports panel → Open in Browser).\n",
        flush=True,
    )


async def run_startup_warmup(
    *,
    ws_backend: str,
    model: str,
    warmup_audio: Path,
    tts_language: str,
    tts_speaker: str,
    tts_instruct: str,
) -> None:
    """Warm all four stages with bundled speech and a generated frame."""
    from .verify_e2e import _make_warmup_frame, _spoken_via_proxy, _wav_to_pcm16_16k

    if not warmup_audio.is_file():
        raise FileNotFoundError(f"warmup audio missing: {warmup_audio}")

    print("Warming AURA with bundled speech + generated frame...", flush=True)
    started = time.monotonic()
    frame, pcm = await asyncio.gather(
        asyncio.to_thread(_make_warmup_frame),
        asyncio.to_thread(_wav_to_pcm16_16k, warmup_audio),
    )
    warmup_task = asyncio.create_task(
        _spoken_via_proxy(
            demo_ws=join_ws_url(ws_backend, STREAM_PATH),
            model=model,
            frames=[frame],
            pcm=pcm,
            tts_language=tts_language,
            tts_speaker=tts_speaker,
            tts_instruct=tts_instruct,
        )
    )
    while not warmup_task.done():
        try:
            await asyncio.wait_for(asyncio.shield(warmup_task), timeout=5)
        except asyncio.TimeoutError:
            print(f"  warmup still running... {time.monotonic() - started:.0f}s", flush=True)
    result = await warmup_task
    if not result["ok"]:
        raise RuntimeError(
            f"warmup did not produce spoken audio: events={result['events']} errors={result['errors']}"
        )
    print(
        f"Warmup complete in {time.monotonic() - started:.1f}s "
        f"({result['audio_deltas']} audio chunks).",
        flush=True,
    )


async def _pump_client_to_backend(client: WebSocket, backend) -> None:
    try:
        while True:
            message = await client.receive()
            if message["type"] == "websocket.disconnect":
                await backend.close()
                return
            if message.get("text") is not None:
                await backend.send(message["text"])
            elif message.get("bytes") is not None:
                await backend.send(message["bytes"])
    except (WebSocketDisconnect, websockets.ConnectionClosed):
        with contextlib.suppress(Exception):
            await backend.close()


async def _pump_backend_to_client(client: WebSocket, backend) -> None:
    dumper = _TtsTurnDumper(TTS_DUMP_DIR)
    try:
        async for message in backend:
            if isinstance(message, bytes):
                await client.send_bytes(message)
            else:
                dumper.observe(message)
                await client.send_text(message)
    except (WebSocketDisconnect, websockets.ConnectionClosed):
        return
    except RuntimeError as exc:
        if "websocket.send" in str(exc) and "websocket.close" in str(exc):
            return
        raise


@dataclass
class _TtsDumpTurn:
    request_id: str
    turn_dir: Path | None = None
    chunk_idx: int = 0
    text: str = ""
    pcm_chunks: list[bytes] = field(default_factory=list)
    sample_rate: int = 24000
    silent: bool = False


class _TtsTurnDumper:
    """Save spoken AURA audio by request ID, even when turns overlap."""

    _LEGACY_REQUEST_ID = "legacy"

    def __init__(self, root: Path) -> None:
        self.root = root
        self.turns: dict[str, _TtsDumpTurn] = {}

    def observe(self, message: str) -> None:
        try:
            event = json.loads(message)
        except json.JSONDecodeError:
            return
        if not isinstance(event, dict):
            return

        event_type = event.get("type")
        request_id = str(event.get("request_id") or self._LEGACY_REQUEST_ID)
        if event_type == "response.start":
            self.turns[request_id] = _TtsDumpTurn(request_id=request_id)
            return

        turn = self.turns.setdefault(request_id, _TtsDumpTurn(request_id=request_id))
        if event_type == "response.text.done":
            turn.text = str(event.get("text") or "")
            if "<|silent|>" in turn.text:
                turn.silent = True
                self._discard_turn(turn)
                self.turns.pop(request_id, None)
            elif turn.turn_dir is not None:
                (turn.turn_dir / "text.txt").write_text(turn.text + "\n", encoding="utf-8")
        elif event_type == "response.audio.delta":
            if not turn.silent:
                self._save_audio_delta(turn, event)
        elif event_type == "response.audio.done":
            self.turns.pop(request_id, None)
            if not turn.silent:
                self._finish_turn(turn)

    @staticmethod
    def _discard_turn(turn: _TtsDumpTurn) -> None:
        if turn.turn_dir is not None and turn.turn_dir.exists():
            for path in turn.turn_dir.glob("*"):
                path.unlink(missing_ok=True)
            turn.turn_dir.rmdir()
        turn.turn_dir = None
        turn.chunk_idx = 0
        turn.pcm_chunks = []

    def _ensure_turn_dir(self, turn: _TtsDumpTurn) -> Path:
        if turn.turn_dir is None:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            safe_id = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in turn.request_id)
            turn.turn_dir = self.root / f"turn_{stamp}_{safe_id}"
            turn.turn_dir.mkdir(parents=True, exist_ok=True)
            (turn.turn_dir / "request_id.txt").write_text(turn.request_id + "\n", encoding="utf-8")
            logger.info("TTS dump dir: %s (request_id=%s)", turn.turn_dir, turn.request_id)
            if turn.text:
                (turn.turn_dir / "text.txt").write_text(turn.text + "\n", encoding="utf-8")
        return turn.turn_dir

    def _save_audio_delta(self, turn: _TtsDumpTurn, event: dict) -> None:
        data_b64 = event.get("data") or event.get("delta") or ""
        if not data_b64:
            return
        try:
            raw = base64.b64decode(data_b64, validate=True)
        except Exception:
            return
        turn_dir = self._ensure_turn_dir(turn)
        chunk_path = turn_dir / f"chunk_{turn.chunk_idx:02d}.wav"
        chunk_path.write_bytes(raw)
        turn.chunk_idx += 1
        pcm, rate = _wav_bytes_to_pcm16(raw)
        if pcm:
            turn.pcm_chunks.append(pcm)
            turn.sample_rate = rate or turn.sample_rate

    @staticmethod
    def _finish_turn(turn: _TtsDumpTurn) -> None:
        if turn.turn_dir is None or not turn.pcm_chunks:
            return
        merged = turn.turn_dir / "merged.wav"
        _write_pcm16_wav(merged, b"".join(turn.pcm_chunks), turn.sample_rate)
        if turn.text and not (turn.turn_dir / "text.txt").exists():
            (turn.turn_dir / "text.txt").write_text(turn.text + "\n", encoding="utf-8")
        logger.info(
            "TTS dump complete: %s (%d chunks, request_id=%s, text=%r)",
            merged,
            turn.chunk_idx,
            turn.request_id,
            turn.text,
        )


def _wav_bytes_to_pcm16(raw: bytes) -> tuple[bytes, int]:
    import io

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


def _expected_proxy_close(exc: BaseException) -> bool:
    if isinstance(exc, (WebSocketDisconnect, websockets.ConnectionClosed, asyncio.CancelledError)):
        return True
    return isinstance(exc, RuntimeError) and "websocket.send" in str(exc) and "websocket.close" in str(exc)


def build_app(
    *,
    ws_backend: str = "ws://127.0.0.1:8666",
    model: str = "aurateam/AURA",
    public_stream_url: str | None = None,
    video_fps: float = 2.0,
    tts_task_type: str = "CustomVoice",
    tts_language: str = "Chinese",
    tts_speaker: str = "Vivian",
    tts_instruct: str | None = None,
    tool_mode: str = "none",
    startup_warmup: bool = False,
    warmup_audio: Path = DEFAULT_WARMUP_AUDIO,
    listen_port: int = 7862,
) -> FastAPI:
    """Build the FastAPI app that serves the UI and proxies AURA streaming WS."""
    if tool_mode not in {"none", "auto"}:
        raise ValueError("tool_mode must be 'none' or 'auto'")
    resolved_tts_instruct = (tts_instruct or "").strip() or DEFAULT_TTS_INSTRUCT

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        if startup_warmup:
            try:
                await run_startup_warmup(
                    ws_backend=ws_backend,
                    model=model,
                    warmup_audio=warmup_audio,
                    tts_language=tts_language,
                    tts_speaker=tts_speaker,
                    tts_instruct=resolved_tts_instruct,
                )
            except Exception as exc:
                logger.warning("Startup warmup failed; continuing cold: %s", exc)
        announce_ready(int(getattr(_app.state, "listen_port", listen_port)))
        yield

    app = FastAPI(title="AURA Streaming Web Demo", lifespan=lifespan)
    app.state.listen_port = int(listen_port)
    index_path = APP_DIR / "index.html"
    app_version_hash = hashlib.sha256()
    for asset_path in (
        STATIC_DIR / "app.js",
        STATIC_DIR / "pcm_worklet.js",
        STATIC_DIR / "playback_worklet.js",
    ):
        app_version_hash.update(asset_path.read_bytes())
    app_version = app_version_hash.hexdigest()[:12]

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        config = json.dumps(
            {
                "model": model,
                "streamPath": public_stream_url or "v1/video/chat/stream",
                "videoFps": video_fps,
                "ttsTaskType": tts_task_type,
                "ttsLanguage": tts_language,
                "ttsSpeaker": tts_speaker,
                "ttsInstruct": resolved_tts_instruct,
                "toolMode": tool_mode,
                "appVersion": app_version,
            },
            ensure_ascii=True,
        )
        html = (
            index_path.read_text(encoding="utf-8")
            .replace("__AURA_WEB_CONFIG__", config)
            .replace("__AURA_WEB_APP_VERSION__", app_version)
        )
        return HTMLResponse(html, headers={"Cache-Control": "no-store"})

    @app.get("/healthz")
    def healthz() -> Response:
        return Response(content="ok", media_type="text/plain")

    @app.websocket(STREAM_PATH)
    async def stream_proxy(websocket: WebSocket) -> None:
        await websocket.accept()
        query = urlencode(websocket.query_params.multi_items())
        backend_url = join_ws_url(ws_backend, STREAM_PATH, query)
        logger.info("Proxying AURA streaming WebSocket to %s", backend_url)
        try:
            async with websockets.connect(
                backend_url,
                max_size=64 * 1024 * 1024,
            ) as backend:
                tasks = {
                    asyncio.create_task(_pump_client_to_backend(websocket, backend)),
                    asyncio.create_task(_pump_backend_to_client(websocket, backend)),
                }
                done, pending = await asyncio.wait(
                    tasks,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                for result in await asyncio.gather(*done, return_exceptions=True):
                    if isinstance(result, BaseException) and not _expected_proxy_close(result):
                        raise result
        except (WebSocketDisconnect, websockets.ConnectionClosed):
            return
        except Exception:
            logger.exception("AURA streaming WebSocket proxy failed")
            with contextlib.suppress(Exception):
                await websocket.close(code=1011)

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument("--ws-backend", default="ws://127.0.0.1:8666")
    parser.add_argument(
        "--public-stream-url",
        help="Browser-visible ws:// or wss:// stream URL; defaults to the same-origin proxy.",
    )
    parser.add_argument("--model", default="aurateam/AURA")
    parser.add_argument("--video-fps", type=float, default=2.0)
    parser.add_argument("--tts-task-type", default="CustomVoice", choices=["CustomVoice", "Base"])
    parser.add_argument("--tts-language", default="Chinese")
    parser.add_argument("--tts-speaker", default="Vivian")
    parser.add_argument("--tool-mode", default="none", choices=["none", "auto"])
    parser.add_argument(
        "--tts-instruct",
        default="",
        help="CustomVoice style instruction (default: empty, matches Native).",
    )
    parser.add_argument(
        "--warmup-audio",
        type=Path,
        default=DEFAULT_WARMUP_AUDIO,
        help="Speech WAV used for startup warmup (default: bundled repository test asset).",
    )
    parser.add_argument("--skip-warmup", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    app = build_app(
        ws_backend=args.ws_backend,
        model=args.model,
        public_stream_url=args.public_stream_url,
        video_fps=args.video_fps,
        tts_task_type=args.tts_task_type,
        tts_language=args.tts_language,
        tts_speaker=args.tts_speaker,
        tts_instruct=args.tts_instruct or None,
        tool_mode=args.tool_mode,
        startup_warmup=not args.skip_warmup,
        warmup_audio=args.warmup_audio,
        listen_port=args.port,
    )
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
