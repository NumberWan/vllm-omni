# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Gateway entry for the AURA duplex backend: main page, /models, /health, /ws.

Run from the parent directory of ``gateway/``:

    PYTHONPATH=<parent>:<vllm-omni tree> python -m gateway.duplex_app
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import quote

import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse

from gateway.duplex_actor import DuplexActor
from gateway.protocol import active_session_count, new_session_id, register_session, unregister_session

BACKEND = os.getenv("AURA_DUPLEX_BACKEND", "ws://127.0.0.1:8092").rstrip("/")
MODEL = os.getenv("AURA_DUPLEX_MODEL", "/workspace/models/AURA_v2")
BACKEND_URL = f"{BACKEND}/v1/realtime?duplex=1&model={quote(MODEL, safe='')}"
STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Aura Gateway (duplex)")
_tools = None
if os.getenv("AURA_GATEWAY_TOOLS", "1") != "0":
    from gateway.native_tools import NativeGatewayTools

    _tools = NativeGatewayTools()


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-store, max-age=0"})


@app.get("/static/playback_worklet.js")
async def playback_worklet():
    return FileResponse(
        STATIC_DIR / "playback_worklet.js",
        media_type="text/javascript",
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@app.get("/models")
async def list_models():
    return [{"name": Path(MODEL).name, "native": False, "id": ""}]


@app.get("/health")
async def health() -> dict[str, str | int]:
    return {"status": "ok", "active_sessions": active_session_count()}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    enable_toolcall = ws.query_params.get("enable_toolcall", "true").strip().lower() not in {"0", "false", "no", "off"}
    try:
        history = json.loads(ws.query_params.get("history", "") or "[]")
    except json.JSONDecodeError:
        history = []
    session_id = new_session_id()
    register_session(session_id)
    try:
        actor = DuplexActor(
            session_id=session_id,
            active_sessions=active_session_count,
            backend_url=BACKEND_URL,
            system_prompt=ws.query_params.get("system_prompt", ""),
            enable_toolcall=enable_toolcall,
            tools=_tools,
            history=history if isinstance(history, list) else [],
        )
        await actor.handle(ws)
    finally:
        unregister_session(session_id)
        try:
            await ws.close()
        except Exception:
            pass


def main() -> None:
    host = os.getenv("GATEWAY_HOST", "0.0.0.0")
    port = int(os.getenv("GATEWAY_PORT", "9999"))
    print(f"[gateway] http://{host}:{port}/  backend={BACKEND_URL}", flush=True)
    uvicorn.run(app, host=host, port=port, log_level=os.getenv("GATEWAY_LOG_LEVEL", "info"))


if __name__ == "__main__":
    main()
