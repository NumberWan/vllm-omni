# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""AURA Realtime browser UI + same-origin WebSocket proxy (push-to-talk).

Adapted from ``examples/online_serving/minicpmo/realtime_web`` but kept under
``aura_omni/`` so MiniCPM's demo stays untouched.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import logging
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
DEFAULT_INSTRUCTIONS = "你是 AURA，一個有幫助的語音助手。請用簡短清楚的中文回答使用者的問題。"


def _join_ws_url(base: str, path: str, query: str) -> str:
    return base.rstrip("/") + path + (("?" + query) if query else "")


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
    try:
        async for message in backend:
            if isinstance(message, bytes):
                await client.send_bytes(message)
            else:
                await client.send_text(message)
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
    ws_backend: str = "ws://127.0.0.1:8091",
    model: str = "aurateam/AURA",
    public_realtime_url: str | None = None,
    instructions: str = DEFAULT_INSTRUCTIONS,
) -> FastAPI:
    app = FastAPI(title="AURA Realtime PTT Demo")
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
                "realtimePath": public_realtime_url or "v1/realtime",
                "instructions": instructions,
                "appVersion": app_version,
            },
            ensure_ascii=True,
        )
        html = (
            index_path.read_text(encoding="utf-8")
            .replace("__AURA_REALTIME_CONFIG__", config)
            .replace("__AURA_REALTIME_APP_VERSION__", app_version)
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
            logger.exception("Realtime WebSocket proxy failed")
            with contextlib.suppress(Exception):
                await websocket.close(code=1011)

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument("--ws-backend", default="ws://127.0.0.1:8091")
    parser.add_argument(
        "--public-realtime-url",
        help="Browser-visible ws:// or wss:// Realtime URL; defaults to the same-origin proxy.",
    )
    parser.add_argument("--model", default="aurateam/AURA")
    parser.add_argument(
        "--instructions",
        default=DEFAULT_INSTRUCTIONS,
        help="AURA system prompt / session instructions",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        build_app(
            ws_backend=args.ws_backend,
            model=args.model,
            public_realtime_url=args.public_realtime_url,
            instructions=args.instructions,
        ),
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
