# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""One browser /ws session bridged onto one AURA duplex realtime WebSocket.

Replaces Actor's ASR / chat / TTS HTTP clients: the duplex backend owns ASR,
history, the two-token decision, Talker and Code2Wav. This class only
translates the page's chunk protocol to realtime events and back.

Turn rules follow the vllm-omni ``realtime_web`` AURA page (``app.js``):
one response at a time. A speech commit or ``response.created`` locks
vision-follow; ``response.listen``, the assistant transcript ``.done`` or a
``response.done`` without a function call releases it. A vision commit in
the gap before a speech turn's response starts aborts that turn's ASR.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import re
import struct
import time
import traceback
import wave
from collections import deque
from typing import Any

import numpy as np
import websockets

from .protocol import (
    Chunk,
    DiscardChunk,
    ProtocolError,
    dump_downstream_event,
    parse_upstream_event,
)

V2_SYSTEM_PROMPT = (
    "You are a real-time multimodal AI companion in a live video conversation with the user. "
    "You are receiving a live video stream where the final frame is the present moment. "
    "Decide whether a response is needed based on the user's message or the visual context. "
    "If a response is needed, output `<|respond|>` followed immediately by the response content. "
    "If no response is needed, output exactly `<|silent|>` to signify silence."
)
# Page preset 1 / original Actor.SYSTEM_PROMPT: the v1 sentence. AURA_v2 goes
# silent far more often under it, so it is swapped for the v2 sentence.
V1_SYSTEM_PROMPT = (
    "You are receiving a live video stream where the final frame is the present moment. "
    "Respond only when a response is needed based on the user's message or the visual context. "
    "Otherwise, output `<|silent|>` to signify silence."
)

_ASR_RATE = 16000
_TTS_RATE = 24000
# Vision-follow turns carry no speech; the backend still wants an audio append.
_VISION_SILENCE = base64.b64encode(b"\x00\x00" * 160).decode("ascii")
# A typed turn rides a speech commit; aura_forced_user_text replaces whatever
# Stage0 hears in this pad.
_TYPED_SILENCE = base64.b64encode(b"\x00\x00" * 1600).decode("ascii")
_MAX_TOOL_FOLLOWUPS = 5
# Calls of one model turn arrive back to back; a batch is complete once no
# new call has arrived for this long.
_TOOL_BATCH_GRACE_S = 0.1
_CLOSE_WAIT_S = 5.0
_SESSION_WAIT_S = 30.0
# A lock whose release event never came (dropped response) must not stop
# vision-follow for the rest of the session.
_VISION_LOCK_MAX_S = 30.0
_EMPTY_DONE_GRACE_S = 1.5
_CLIENT_TOOL_RESULT = "CLIENT_TOOL_RESULT "
_NO_FRAMES = "还没有画面，请稍后再输入文字。"
_BOX_PATTERN = re.compile(r"<\|box_start\|>(.*?)<\|box_end\|>")
# A streaming transcript can stop inside a box or a special token.
_OPEN_BOX_TAIL = re.compile(r"<\|box_start\|>[^<]*$")
_SPECIAL_TAIL = re.compile(r"<\|?[a-z_]*\|?$")
_OBJECT_REF_TAGS = ("<|object_ref_start|>", "<|object_ref_end|>")
_OBJECT_REF_TAIL = re.compile(r"<\|object_ref(?:_start)?$")
_RESPOND = "<|respond|>"
_TRACE = os.getenv("AURA_GATEWAY_TRACE", "0") == "1"


def _frames_from_video_url(video_url: str) -> list[str]:
    prefix = "data:video/jpeg;base64,"
    body = video_url[len(prefix) :] if video_url.startswith(prefix) else video_url.split(",", 1)[-1]
    return [part for part in body.split(",") if part]


def _wav_to_pcm16_16k(wav_bytes: bytes) -> bytes:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        rate = wav.getframerate()
        raw = wav.readframes(wav.getnframes())
    if width != 2:
        raise ProtocolError(f"unsupported WAV sample width: {width}")
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    if rate != _ASR_RATE and samples.size:
        n_out = max(1, int(round(samples.size * _ASR_RATE / rate)))
        samples = np.interp(np.linspace(0, samples.size - 1, n_out), np.arange(samples.size), samples)
    return np.clip(np.round(samples), -32768, 32767).astype("<i2").tobytes()


def _wav_from_pcm16(pcm: bytes, rate: int = _TTS_RATE) -> bytes:
    header = b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVE"
    header += b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
    header += b"data" + struct.pack("<I", len(pcm))
    return header + pcm


def _parse_boxes(text: str) -> tuple[str, list[list[int]]]:
    boxes: list[list[int]] = []
    for match in _BOX_PATTERN.finditer(text):
        cleaned = match.group(1).replace("(", "").replace(")", "")
        try:
            parts = [int(float(c.strip())) for c in cleaned.split(",") if c.strip()]
        except ValueError:
            continue
        if len(parts) == 4:
            boxes.append([max(0, min(1000, v)) for v in parts])
    text = _BOX_PATTERN.sub("", text)
    for tag in _OBJECT_REF_TAGS:
        text = text.replace(tag, "")
    return _OBJECT_REF_TAIL.sub("", text).strip(), boxes


def _visible_partial(text: str) -> str:
    text = _OPEN_BOX_TAIL.sub("", text.replace(_RESPOND, ""))
    text = _SPECIAL_TAIL.sub("", text)
    return _parse_boxes(text)[0]


def _response_id(event: dict[str, Any]) -> str:
    rid = event.get("response_id")
    response = event.get("response")
    if not rid and isinstance(response, dict):
        rid = response.get("id")
        metadata = response.get("metadata")
        if not rid and isinstance(metadata, dict):
            rid = metadata.get("response_id")
    return str(rid or "")


def _done_has_function_call(event: dict[str, Any]) -> bool:
    response = event.get("response")
    output = response.get("output") if isinstance(response, dict) else None
    return isinstance(output, list) and any(isinstance(i, dict) and i.get("type") == "function_call" for i in output)


class _Commit:
    __slots__ = ("chunk", "sent_at", "audio_sent", "last_rid", "typed")

    def __init__(self, chunk: Chunk, *, typed: bool = False) -> None:
        self.chunk = chunk
        self.sent_at = time.time()
        self.audio_sent = False
        self.last_rid = ""
        self.typed = typed


class _Response:
    __slots__ = ("rid", "commit", "text_sent", "listened", "done_sent", "partial", "partial_sent", "text_at")

    def __init__(self, rid: str, commit: _Commit) -> None:
        self.rid = rid
        self.commit = commit
        self.text_sent = False
        self.listened = False
        self.done_sent = False
        self.partial = ""
        self.partial_sent = ""
        self.text_at: float | None = None


class DuplexActor:
    def __init__(
        self,
        *,
        session_id: str,
        active_sessions,
        backend_url: str,
        system_prompt: str | None,
        enable_toolcall: bool,
        tools,
        history: list | None,
    ) -> None:
        self._session_id = session_id
        self._active_sessions = active_sessions
        self._backend_url = backend_url
        prompt = (system_prompt or "").strip()
        self._instructions = V2_SYSTEM_PROMPT if not prompt or prompt == V1_SYSTEM_PROMPT else prompt
        self._tools = tools if enable_toolcall else None
        if history:
            self._log(f"ignoring page history ({len(history)} messages): duplex history lives on the server")
        self._closed = False
        self._backend = None
        # Commits sent but not yet acknowledged by input_audio_buffer.committed.
        self._unacked: deque[_Commit] = deque()
        self._items: dict[str, _Commit] = {}
        # Newest acknowledged commit: the input every new response answers.
        self._latest: _Commit | None = None
        self._responses: dict[str, _Response] = {}
        self._current_rid = ""
        self._vision_locked_at: float | None = None
        # Typed questions without their own frames see the latest camera view.
        self._last_frames: list[str] = []
        self._follow = {"count": 0, "gen": 0, "awaiting_done": 0}
        self._tool_tasks: set[asyncio.Task] = set()
        # One batch of calls runs one by one on a single worker.
        self._tool_queue: deque[tuple] = deque()
        self._tool_worker: asyncio.Task | None = None
        self._tool_last_at = 0.0
        self._tool_dones = asyncio.Event()
        self._tool_dones.set()

    def _log(self, message: str) -> None:
        print(f"[duplex:{self._session_id[:8]}] {message}", flush=True)

    async def _send(self, ws, type_: str, data: dict, chunk: Chunk | None = None) -> None:
        if self._closed:
            return
        try:
            await ws.send_text(
                dump_downstream_event(
                    type_,
                    data,
                    session_id=self._session_id,
                    chunk_id=chunk.chunk_id if chunk is not None else 0,
                    video_timestamp_ms=chunk.video_timestamp_ms if chunk is not None else 0,
                    wall_time_ms=chunk.wall_time_ms if chunk is not None else 0,
                )
            )
        except Exception:
            self._closed = True

    async def _backend_send(self, event: dict) -> None:
        await self._backend.send(json.dumps(event))

    # ---- lifecycle ----

    async def handle(self, ws) -> None:
        try:
            async with websockets.connect(self._backend_url, max_size=64 * 1024 * 1024) as backend:
                self._backend = backend
                if not await self._open_session(ws):
                    return
                await self._send(ws, "session", {"active_sessions": self._active_sessions()})
                up = asyncio.create_task(self._pump_page(ws))
                down = asyncio.create_task(self._pump_backend(ws))
                done, pending = await asyncio.wait({up, down}, return_when=asyncio.FIRST_COMPLETED)
                for task in (*pending, *self._tool_tasks):
                    task.cancel()
                for task in done:
                    if not task.cancelled() and task.exception() is not None:
                        self._log(f"session ended: {task.exception()!r}")
                await self._close_backend(down_finished=down in done)
        except (OSError, websockets.InvalidStatus, websockets.InvalidHandshake) as exc:
            self._log(f"backend connect failed: {exc}")
            await self._send(ws, "error", {"message": f"后端连接失败: {exc}"})
        except Exception as exc:
            traceback.print_exc()
            await self._send(ws, "error", {"message": str(exc)})

    async def _open_session(self, ws) -> bool:
        session: dict[str, Any] = {
            "modalities": ["audio", "text"],
            "instructions": self._instructions,
            "tools": self._tools.realtime_tools if self._tools is not None else [],
            "extra_body": {"auto_response": True},
        }
        await self._backend_send({"type": "session.update", "session": session})
        deadline = time.monotonic() + _SESSION_WAIT_S
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                await self._send(ws, "error", {"message": "后端 session.updated 超时（可能还在暖机）"})
                return False
            event = json.loads(await asyncio.wait_for(self._backend.recv(), timeout=remaining))
            kind = event.get("type")
            if kind == "session.updated":
                return True
            if kind == "error":
                err = event.get("error") or {}
                code = err.get("code") or ""
                message = err.get("message") or str(err)
                if code == "duplex_session_capacity_exhausted":
                    message = "后端只有一个会话位，正被占用（或上一个会话 30 秒内未释放），请稍后重试"
                await self._send(ws, "error", {"message": f"{message} [{code}]"})
                return False

    async def _close_backend(self, *, down_finished: bool) -> None:
        if down_finished:
            return
        try:
            await self._backend_send({"type": "session.close"})
            deadline = time.monotonic() + _CLOSE_WAIT_S
            while (remaining := deadline - time.monotonic()) > 0:
                event = json.loads(await asyncio.wait_for(self._backend.recv(), timeout=remaining))
                if event.get("type") == "session.closed":
                    return
        except Exception:
            return

    # ---- vision-follow lock ----

    def _vision_locked(self) -> bool:
        if self._vision_locked_at is None:
            return False
        if time.monotonic() - self._vision_locked_at > _VISION_LOCK_MAX_S:
            self._log("vision lock expired without a release event")
            self._vision_locked_at = None
            return False
        return True

    def _lock_vision(self) -> None:
        self._vision_locked_at = time.monotonic()

    def _release_vision(self, rid: str) -> None:
        if rid and self._current_rid and rid != self._current_rid:
            return
        self._vision_locked_at = None

    # ---- upstream ----

    async def _pump_page(self, ws) -> None:
        while True:
            raw = await ws.receive_text()
            try:
                chunk = parse_upstream_event(raw, expected_session_id=self._session_id)
            except DiscardChunk:
                continue
            except ProtocolError as exc:
                self._log(f"dropped invalid chunk: {exc}")
                continue
            if chunk.type == "multimodal":
                await self._speech_turn(ws, chunk)
            elif chunk.type == "video":
                text = (chunk.data.get("text") or "").strip()
                if text:
                    await self._typed_turn(ws, chunk, text)
                else:
                    await self._vision_turn(chunk)
            elif chunk.type == "text":
                text = (chunk.data.get("text") or "").strip()
                if text and not text.startswith(_CLIENT_TOOL_RESULT):
                    await self._typed_turn(ws, chunk, text)
            else:
                await self._send(ws, "error", {"message": f"不支援的输入类型: {chunk.type}"}, chunk)
                await self._send(ws, "turn_done", {}, chunk)

    async def _vision_turn(self, chunk: Chunk) -> None:
        frames = _frames_from_video_url(chunk.data.get("video_url") or "")
        if frames:
            self._last_frames = frames
        if self._vision_locked() or not frames:
            return
        self._unacked.append(_Commit(chunk))
        await self._backend_send(
            {
                "type": "input_audio_buffer.append",
                "audio": _VISION_SILENCE,
                "format": "pcm16",
                "sample_rate_hz": _ASR_RATE,
                "is_speech": False,
                "video_frames": frames,
            }
        )
        await self._backend_send({"type": "input_audio_buffer.commit", "create_response": True})
        if _TRACE:
            self._log(f"sent vision commit chunk={chunk.chunk_id}")

    async def _speech_turn(self, ws, chunk: Chunk) -> None:
        try:
            audio_b64 = chunk.data["audio_b64"]
            pcm = _wav_to_pcm16_16k(base64.b64decode(audio_b64 + "=" * (-len(audio_b64) % 4)))
        except (ValueError, wave.Error, ProtocolError) as exc:
            await self._send(ws, "error", {"message": f"音频解析失败: {exc}"}, chunk)
            await self._send(ws, "turn_done", {}, chunk)
            return
        frames = _frames_from_video_url(chunk.data.get("video_url") or "")
        if not pcm or not frames:
            await self._send(ws, "turn_done", {}, chunk)
            return
        self._last_frames = frames
        self._reset_tools()
        self._lock_vision()
        self._unacked.append(_Commit(chunk))
        # AURA rejects any append without video_frames, so the utterance goes
        # in one append that carries the PTT frames.
        await self._backend_send(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm).decode("ascii"),
                "format": "pcm16",
                "sample_rate_hz": _ASR_RATE,
                "is_speech": True,
                "video_frames": frames,
            }
        )
        await self._backend_send({"type": "input_audio_buffer.commit", "create_response": True})
        if _TRACE:
            self._log(f"sent speech commit chunk={chunk.chunk_id} samples={len(pcm) // 2}")

    async def _typed_turn(self, ws, chunk: Chunk, text: str) -> None:
        frames = _frames_from_video_url(chunk.data.get("video_url") or "") or self._last_frames
        if not frames:
            await self._send(ws, "error", {"message": _NO_FRAMES}, chunk)
            await self._send(ws, "turn_done", {}, chunk)
            return
        self._last_frames = frames
        self._reset_tools()
        self._lock_vision()
        self._unacked.append(_Commit(chunk, typed=True))
        await self._backend_send(
            {
                "type": "input_audio_buffer.append",
                "audio": _TYPED_SILENCE,
                "format": "pcm16",
                "sample_rate_hz": _ASR_RATE,
                "is_speech": True,
                "video_frames": frames,
                "aura_forced_user_text": text,
            }
        )
        await self._backend_send({"type": "input_audio_buffer.commit", "create_response": True})
        if _TRACE:
            self._log(f"sent typed commit chunk={chunk.chunk_id} chars={len(text)}")

    # ---- downstream ----

    async def _pump_backend(self, ws) -> None:
        async for raw in self._backend:
            if isinstance(raw, bytes):
                continue
            try:
                event = json.loads(raw)
            except ValueError:
                continue
            await self._on_event(ws, event)

    async def _turn_done(self, ws, resp: _Response | None) -> None:
        if resp is None or resp.done_sent:
            return
        resp.done_sent = True
        await self._send(ws, "turn_done", {}, resp.commit.chunk)

    async def _deferred_turn_done(self, ws, resp: _Response) -> None:
        await asyncio.sleep(_EMPTY_DONE_GRACE_S)
        if resp.commit.last_rid == resp.rid:
            await self._turn_done(ws, resp)

    async def _on_event(self, ws, event: dict[str, Any]) -> None:
        kind = event.get("type") or ""
        if _TRACE and kind not in ("response.output_audio.delta", "response.output_audio_transcript.delta"):
            brief = {k: v for k, v in event.items() if k not in ("delta", "audio", "session", "event")}
            self._log("backend " + json.dumps(brief, ensure_ascii=False)[:2000])
        rid = _response_id(event)
        resp = self._responses.get(rid) if rid else None

        if kind == "input_audio_buffer.committed":
            if self._unacked:
                commit = self._unacked.popleft()
                self._latest = commit
                if event.get("item_id"):
                    self._items[str(event["item_id"])] = commit
            return
        if kind == "conversation.item.input_audio_transcription.completed":
            commit = self._items.get(str(event.get("item_id") or ""))
            text = (event.get("transcript") or "").strip()
            # A typed turn's transcript is only the pad; the page already shows the text.
            if text and commit is not None and not commit.typed:
                asr_ms = int((time.time() - commit.sent_at) * 1000)
                await self._send(ws, "asr_text", {"text": text, "asr_ms": asr_ms}, commit.chunk)
            return
        if kind == "response.created":
            if rid and self._latest is not None:
                self._responses[rid] = _Response(rid, self._latest)
                self._latest.last_rid = rid
                self._current_rid = rid
                self._lock_vision()
            return
        if kind in ("response.output_audio_transcript.done", "response.output_text.done"):
            if resp is None or resp.text_sent:
                return
            resp.text_sent = True
            raw_text = (event.get("transcript") if kind.endswith("transcript.done") else event.get("text")) or ""
            text, boxes = _parse_boxes(raw_text.replace(_RESPOND, ""))
            if text:
                # Model time ends with the last text delta, not with playback.
                infer_ms = int(((resp.text_at or time.time()) - resp.commit.sent_at) * 1000)
                await self._send(ws, "text", {"text": text, "infer_ms": infer_ms}, resp.commit.chunk)
            if boxes:
                await self._show_boxes(ws, boxes, resp.commit.chunk)
            self._release_vision(rid)
            return
        if kind in ("response.output_audio_transcript.delta", "response.output_text.delta"):
            delta = event.get("delta") or ""
            if resp is None or resp.text_sent or not delta:
                return
            resp.partial += delta
            resp.text_at = time.time()
            visible = _visible_partial(resp.partial)
            if visible and visible != resp.partial_sent:
                resp.partial_sent = visible
                await self._send(ws, "text_delta", {"text": visible}, resp.commit.chunk)
            return
        if kind == "response.output_audio.delta":
            delta = event.get("delta") or ""
            if resp is None or not delta:
                return
            commit = resp.commit
            data: dict[str, Any] = {
                "audio": base64.b64encode(_wav_from_pcm16(base64.b64decode(delta))).decode("ascii"),
                "sample_rate": _TTS_RATE,
            }
            if not commit.audio_sent:
                commit.audio_sent = True
                data["tts_ms"] = int((time.time() - commit.sent_at) * 1000)
            await self._send(ws, "audio", data, commit.chunk)
            return
        if kind == "response.output_item.done":
            item = event.get("item")
            if isinstance(item, dict) and item.get("type") == "function_call" and item.get("call_id"):
                await self._on_function_call(ws, resp, item)
            return
        if kind == "response.listen":
            if resp is not None:
                resp.listened = True
            self._release_vision(rid)
            await self._turn_done(ws, resp)
            return
        if kind == "response.done":
            if _done_has_function_call(event):
                follow = self._follow
                follow["awaiting_done"] = max(0, follow["awaiting_done"] - 1)
                if not follow["awaiting_done"]:
                    self._tool_dones.set()
                return
            self._release_vision(rid)
            if resp is not None and not resp.text_sent and not resp.listened:
                # A speech turn that calls a tool can first end an empty
                # response; its function-call response arrives right after.
                task = asyncio.create_task(self._deferred_turn_done(ws, resp))
                self._tool_tasks.add(task)
                task.add_done_callback(self._tool_tasks.discard)
                return
            await self._turn_done(ws, resp)
            return
        if kind == "error":
            err = event.get("error") or {}
            code = err.get("code") or ""
            if code == "input_audio_buffer_empty":
                if self._unacked:
                    commit = self._unacked.popleft()
                    await self._send(ws, "turn_done", {}, commit.chunk)
                self._vision_locked_at = None
                return
            await self._send(ws, "error", {"message": f"{err.get('message') or code} [{code}]"})
            return
        if kind == "session.closed":
            await self._send(ws, "error", {"message": f"后端会话已关闭: {event.get('reason') or ''}"})
            raise ConnectionError("backend session closed")

    async def _show_boxes(self, ws, boxes: list[list[int]], chunk: Chunk) -> None:
        tool_id = f"box_{int(time.time() * 1000)}"
        started = {"status": "started", "id": tool_id, "name": "ShowToUser", "arguments": {"boxes": boxes}}
        await self._send(ws, "toolcall", started, chunk)
        await self._send(
            ws,
            "toolcall",
            {
                "status": "success",
                "id": tool_id,
                "name": "ShowToUser",
                "output": f"[BBOX Parser Result] ShowToUser toolcall dispatched to client with {len(boxes)} box(es).",
                "error": "",
            },
            chunk,
        )

    # ---- tools (native actor: at most 5 follow-up rounds) ----

    async def _on_function_call(self, ws, resp: _Response | None, item: dict[str, Any]) -> None:
        name = str(item.get("name") or "")
        arguments_raw = str(item.get("arguments") or "")
        try:
            arguments = json.loads(arguments_raw or "{}")
        except ValueError:
            arguments = {"raw": arguments_raw}
        chunk = resp.commit.chunk if resp is not None else (self._latest.chunk if self._latest else None)
        await self._send(
            ws, "toolcall", {"status": "started", "id": item["call_id"], "name": name, "arguments": arguments}, chunk
        )
        self._follow["awaiting_done"] += 1
        self._tool_dones.clear()
        self._tool_queue.append((resp, item, name, arguments_raw, chunk))
        self._tool_last_at = time.monotonic()
        if self._tool_worker is None or self._tool_worker.done():
            task = asyncio.create_task(self._run_tool_batch(ws, self._follow["gen"]))
            self._tool_worker = task
            self._tool_tasks.add(task)
            task.add_done_callback(self._tool_tasks.discard)

    def _reset_tools(self) -> None:
        """A new user turn: drop the previous turn's queued calls and round count."""
        self._follow.update(gen=self._follow["gen"] + 1, count=0, awaiting_done=0)
        self._tool_queue.clear()
        self._tool_dones.set()
        # The old worker exits on its own; the next call starts a fresh one.
        self._tool_worker = None

    async def _run_tool_batch(self, ws, generation: int) -> None:
        """Run one turn's calls one by one, then ask for a single follow-up.

        Native executes every call of a round and answers with all results at
        once; the backend splices each ``function_call_output`` into history
        and the one ``response.create`` reads them back together.
        """
        follow = self._follow
        resp = chunk = None
        while True:
            while self._tool_queue and follow["gen"] == generation:
                resp, item, name, arguments_raw, chunk = self._tool_queue.popleft()
                await self._run_tool(ws, item, name, arguments_raw, chunk)
            if follow["gen"] != generation:
                return
            wait = _TOOL_BATCH_GRACE_S - (time.monotonic() - self._tool_last_at)
            if wait > 0:
                await asyncio.sleep(wait)
                continue
            # response.create is dropped while a call's response is still open.
            await self._tool_dones.wait()
            if follow["gen"] != generation:
                return
            if not self._tool_queue:
                break
        if follow["count"] >= _MAX_TOOL_FOLLOWUPS:
            # No follow-up will open another response: end the turn here.
            self._vision_locked_at = None
            await self._send(ws, "text", {"text": "工具调用次数到达上限"}, chunk)
            await self._turn_done(ws, resp)
            return
        follow["count"] += 1
        await self._backend_send({"type": "response.create"})

    async def _run_tool(self, ws, item, name, arguments_raw, chunk) -> None:
        if self._tools is None:
            result = {"status": "error", "output": "", "error": f"tool not found: {name}"}
        else:
            result = await self._tools.run(name, arguments_raw, self._last_frames[-1] if self._last_frames else "")
        await self._send(
            ws,
            "toolcall",
            {
                "status": result["status"],
                "id": item["call_id"],
                "name": name,
                "output": result["output"],
                "error": result["error"],
            },
            chunk,
        )
        await self._backend_send(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": item["call_id"],
                    "output": result["output"] or result["error"],
                },
            }
        )
