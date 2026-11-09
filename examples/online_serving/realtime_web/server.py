# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Static host and same-origin proxy for the shared realtime browser UI."""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlencode

import uvicorn
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger(__name__)
APP_DIR = Path(__file__).parent / "app"
STATIC_DIR = APP_DIR / "static"


def _join_ws_url(base: str, path: str, query: str) -> str:
    return base.rstrip("/") + path + (("?" + query) if query else "")


class ToolExecutor(Protocol):
    """Runs the client side of function calling on the browser's behalf."""

    @property
    def realtime_tools(self) -> list[dict[str, Any]]: ...

    async def run(self, name: str, arguments: str) -> str: ...


def _json_event(text: str) -> dict[str, Any] | None:
    try:
        event = json.loads(text)
    except ValueError:
        return None
    return event if isinstance(event, dict) else None


def _with_session_tools(text: str, tools: ToolExecutor) -> str:
    """Add the executor's tools to a ``session.update`` that has none."""
    event = _json_event(text)
    if event is None or event.get("type") != "session.update" or not isinstance(event.get("session"), dict):
        return text
    if "tools" in event["session"]:
        return text
    event["session"] = {**event["session"], "tools": tools.realtime_tools}
    return json.dumps(event)


def _finished_function_call(event: dict[str, Any]) -> dict[str, Any] | None:
    """The call in a ``response.output_item.done``; the arguments event carries no name."""
    item = event.get("item")
    if event.get("type") != "response.output_item.done" or not isinstance(item, dict):
        return None
    if item.get("type") != "function_call" or not isinstance(item.get("call_id"), str) or not item["call_id"]:
        return None
    return item


# A failed search result makes the model call again. Stop opening follow-ups
# after this many, or one question can spin for the life of the session.
_MAX_TOOL_FOLLOWUPS = 3


def _is_user_commit(text: str) -> bool:
    event = _json_event(text)
    return event is not None and event.get("type") == "input_audio_buffer.commit"


async def _answer_function_call(backend, tools: ToolExecutor, call: dict[str, Any], *, follow_up: bool = True) -> None:
    output = await tools.run(str(call.get("name") or ""), str(call.get("arguments") or ""))
    item = {"type": "function_call_output", "call_id": call["call_id"], "output": output}
    await backend.send(json.dumps({"type": "conversation.item.create", "item": item}))
    if follow_up:
        await backend.send(json.dumps({"type": "response.create"}))


async def _pump_client_to_backend(
    client: WebSocket,
    backend,
    tools: ToolExecutor | None = None,
    tool_followups: list[int] | None = None,
) -> None:
    try:
        while True:
            message = await client.receive()
            if message["type"] == "websocket.disconnect":
                await backend.close()
                return
            if message.get("text") is not None:
                text = message["text"]
                if tool_followups is not None and _is_user_commit(text):
                    tool_followups[0] = 0
                await backend.send(text if tools is None else _with_session_tools(text, tools))
            elif message.get("bytes") is not None:
                await backend.send(message["bytes"])
    except (WebSocketDisconnect, websockets.ConnectionClosed):
        with contextlib.suppress(Exception):
            await backend.close()


async def _pump_backend_to_client(
    client: WebSocket,
    backend,
    tools: ToolExecutor | None = None,
    tool_followups: list[int] | None = None,
) -> None:
    calls: set[asyncio.Task] = set()
    try:
        async for message in backend:
            if isinstance(message, bytes):
                await client.send_bytes(message)
                continue
            await client.send_text(message)
            if tools is None:
                continue
            event = _json_event(message)
            call = _finished_function_call(event) if event is not None else None
            if call is not None:
                count = tool_followups[0] if tool_followups is not None else 0
                follow_up = count < _MAX_TOOL_FOLLOWUPS
                if tool_followups is not None and follow_up:
                    tool_followups[0] = count + 1
                task = asyncio.create_task(_answer_function_call(backend, tools, call, follow_up=follow_up))
                calls.add(task)
                task.add_done_callback(calls.discard)
    except (WebSocketDisconnect, websockets.ConnectionClosed):
        return
    except RuntimeError as exc:
        if "websocket.send" in str(exc) and "websocket.close" in str(exc):
            return
        raise


def _expected_proxy_close(exc: BaseException) -> bool:
    if isinstance(exc, (WebSocketDisconnect, websockets.ConnectionClosed, asyncio.CancelledError)):
        return True
    return isinstance(exc, RuntimeError) and "websocket.send" in str(exc) and "websocket.close" in str(exc)


def build_app(
    *,
    ws_backend: str | None = None,
    model: str | None = None,
    public_realtime_url: str | None = None,
    ref_audio: str | None = None,
    profile: str = "minicpm-native",
    turn_mode: str = "manual",
    speech_input: bool = False,
    tool_executor: ToolExecutor | None = None,
) -> FastAPI:
    if profile not in {"minicpm-native", "qwen3-turn", "aura-ptt", "joyai-turn"}:
        raise ValueError(f"Unknown profile: {profile}")
    if turn_mode not in {"manual", "vad", "semantic_vad"}:
        raise ValueError(f"Unknown turn mode: {turn_mode}")
    if turn_mode == "semantic_vad" and profile != "qwen3-turn":
        raise ValueError("--semantic-vad requires --profile qwen3-turn")
    if profile == "qwen3-turn" and ref_audio:
        raise ValueError("Qwen3 profile does not accept --ref-audio")
    if profile == "aura-ptt" and ref_audio:
        raise ValueError("AURA profile does not accept --ref-audio")
    if profile == "joyai-turn" and ref_audio:
        raise ValueError("JoyAI profile does not accept --ref-audio")
    ws_backend = ws_backend or ("ws://127.0.0.1:8091" if profile == "qwen3-turn" else "ws://127.0.0.1:8099")
    model = (
        model
        or {
            "minicpm-native": "openbmb/MiniCPM-o-4_5",
            "aura-ptt": "aurateam/AURA",
            "joyai-turn": "jdopensource/JoyAI-VL-Interaction-Preview",
            "qwen3-turn": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        }[profile]
    )
    app = FastAPI(title="Omni Realtime Web UI")
    index_path = APP_DIR / "index.html"
    app_version_hash = hashlib.sha256()
    for asset_path in (
        STATIC_DIR / "app.js",
        STATIC_DIR / "pcm_worklet.js",
        STATIC_DIR / "playback_worklet.js",
        *sorted((STATIC_DIR / "profiles").glob("*.js")),
    ):
        app_version_hash.update(asset_path.read_bytes())
    app_version = app_version_hash.hexdigest()[:12]

    ref_audio_uri: str | None = None
    if ref_audio:
        ref_path = Path(ref_audio)
        if not ref_path.is_file():
            raise SystemExit(f"--ref-audio not found: {ref_audio}")
        encoded = base64.b64encode(ref_path.read_bytes()).decode("ascii")
        ref_audio_uri = f"data:audio/wav;base64,{encoded}"

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        config = json.dumps(
            {
                "model": model,
                "profile": profile,
                "turnMode": turn_mode,
                "speechInput": speech_input,
                "realtimePath": public_realtime_url or "v1/realtime",
                "refAudio": ref_audio_uri,
                "appVersion": app_version,
            },
            ensure_ascii=True,
        ).replace("<", "\\u003c")
        html = (
            index_path.read_text(encoding="utf-8")
            .replace("__FULL_DUPLEX_CONFIG__", config)
            .replace("__FULL_DUPLEX_APP_VERSION__", app_version)
        )
        return HTMLResponse(html, headers={"Cache-Control": "no-store"})

    @app.get("/healthz")
    def healthz() -> Response:
        return Response(content="ok", media_type="text/plain")

    @app.websocket("/v1/realtime")
    async def realtime_proxy(websocket: WebSocket) -> None:
        await websocket.accept()
        query = urlencode(websocket.query_params.multi_items())
        backend_url = _join_ws_url(ws_backend, "/v1/realtime", query)
        logger.info("Proxying Realtime WebSocket to %s", backend_url)
        try:
            async with websockets.connect(
                backend_url,
                max_size=64 * 1024 * 1024,
            ) as backend:
                tool_followups = [0]
                tasks = {
                    asyncio.create_task(_pump_client_to_backend(websocket, backend, tool_executor, tool_followups)),
                    asyncio.create_task(_pump_backend_to_client(websocket, backend, tool_executor, tool_followups)),
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
        except (OSError, TimeoutError) as exc:
            message = (
                f"Cannot connect to realtime backend {ws_backend}. "
                "Check that vllm serve is running and --backend matches its --port. "
                "For containers or remote hosts, use the backend address reachable from the UI server."
            )
            logger.warning("%s (%s)", message, exc)
            with contextlib.suppress(Exception):
                await websocket.send_json({"type": "error", "code": "backend_unavailable", "error": message})
                await websocket.close(code=1011)
        except Exception:
            logger.exception("Realtime WebSocket proxy failed")
            with contextlib.suppress(Exception):
                await websocket.close(code=1011)
        finally:
            with contextlib.suppress(Exception):
                await websocket.close()

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    return app


def main(default_profile: str = "minicpm-native", tool_executor: ToolExecutor | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument(
        "--profile", choices=("minicpm-native", "qwen3-turn", "aura-ptt", "joyai-turn"), default=default_profile
    )
    parser.add_argument("--ws-backend", "--backend", dest="ws_backend")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--manual", dest="turn_mode", action="store_const", const="manual")
    mode.add_argument("--vad", dest="turn_mode", action="store_const", const="vad")
    mode.add_argument(
        "--semantic-vad",
        dest="turn_mode",
        action="store_const",
        const="semantic_vad",
        help="Qwen3 semantic turn detection (requires the Smart Turn artifact).",
    )
    parser.set_defaults(turn_mode="manual")
    parser.add_argument("--speech-input", action="store_true", help="Enable JoyAI microphone input (ASR deployment).")
    parser.add_argument(
        "--public-realtime-url",
        help="Browser-visible ws:// or wss:// Realtime URL; defaults to the same-origin proxy.",
    )
    parser.add_argument("--model")
    parser.add_argument(
        "--ref-audio",
        help=(
            "Optional MiniCPM assistant voice override; when omitted, the server "
            "uses the model-bundled assets/HT_ref_audio.wav"
        ),
    )
    args = parser.parse_args()
    if args.profile != "minicpm-native" and args.ref_audio:
        parser.error("--ref-audio is only supported by minicpm-native")

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        build_app(
            ws_backend=args.ws_backend,
            model=args.model,
            public_realtime_url=args.public_realtime_url,
            ref_audio=args.ref_audio,
            profile=args.profile,
            turn_mode=args.turn_mode,
            speech_input=args.speech_input,
            tool_executor=tool_executor,
        ),
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
