# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import base64
import json
import os
import uuid
from collections.abc import Sequence
from dataclasses import dataclass

ACTIVE_SESSION_IDS: set[str] = set()

MODEL_CONFIGS_PATH = os.getenv("MODEL_CONFIGS_PATH", "model_configs.json")


def _load_all() -> dict:
    path = MODEL_CONFIGS_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"model configs file not found: {path}")
    with open(path) as f:
        return json.load(f)


def load_model_configs() -> Sequence[dict]:
    return _load_all().get("models", [])


def load_gateway_config() -> dict:
    return _load_all().get("gateway", {})


class ProtocolError(ValueError):
    pass


class DiscardChunk(ProtocolError):  # noqa: N818
    pass


@dataclass(frozen=True, slots=True)
class Chunk:
    direction: str
    type: str
    session_id: str
    chunk_id: int
    video_timestamp_ms: int
    wall_time_ms: int
    data: dict

    def to_dict(self) -> dict:
        return {
            "direction": self.direction,
            "type": self.type,
            "session_id": self.session_id,
            "chunk_id": self.chunk_id,
            "video_timestamp_ms": self.video_timestamp_ms,
            "wall_time_ms": self.wall_time_ms,
            "data": dict(self.data),
        }


def new_session_id() -> str:
    return uuid.uuid4().hex


def register_session(session_id: str) -> None:
    ACTIVE_SESSION_IDS.add(session_id)


def unregister_session(session_id: str) -> None:
    ACTIVE_SESSION_IDS.discard(session_id)


def active_session_count() -> int:
    return len(ACTIVE_SESSION_IDS)


def _required_str(msg: dict, field: str) -> str:
    value = msg.get(field)
    if not isinstance(value, str) or not value:
        raise ProtocolError(f"{field} is required")
    return value


def _required_int(msg: dict, field: str) -> int:
    try:
        return int(msg[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError(f"{field} is required") from exc


def _parse_chunk(raw: str | dict, *, expected_direction: str) -> Chunk:
    if isinstance(raw, str):
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ProtocolError("invalid chunk JSON") from exc
    else:
        msg = raw

    if not isinstance(msg, dict):
        raise ProtocolError("chunk must be an object")
    direction = _required_str(msg, "direction")
    if direction != expected_direction:
        raise ProtocolError("invalid chunk direction")
    data = msg.get("data")
    if not isinstance(data, dict):
        raise ProtocolError("data is required")
    return Chunk(
        direction=direction,
        type=_required_str(msg, "type"),
        session_id=_required_str(msg, "session_id"),
        chunk_id=_required_int(msg, "chunk_id"),
        video_timestamp_ms=_required_int(msg, "video_timestamp_ms"),
        wall_time_ms=_required_int(msg, "wall_time_ms"),
        data=data,
    )


def parse_upstream_event(raw: str | dict, *, expected_session_id: str | None = None) -> Chunk:
    chunk = _parse_chunk(raw, expected_direction="upstream")
    if chunk.session_id not in ACTIVE_SESSION_IDS:
        raise ProtocolError("unknown or inactive session_id")
    if expected_session_id is not None and chunk.session_id != expected_session_id:
        raise ProtocolError("chunk session_id does not match connection")
    if chunk.type not in {"audio", "text", "video", "image", "multimodal"}:
        raise ProtocolError("invalid upstream chunk type")

    if chunk.type in {"audio", "multimodal"}:
        audio_b64 = chunk.data.get("audio_b64")
        if not isinstance(audio_b64, str):
            raise ProtocolError(f"{chunk.type}.data.audio_b64 is required")
        if not audio_b64.strip():
            raise DiscardChunk("empty audio chunk")
        if chunk.type == "multimodal":
            video_url = chunk.data.get("video_url")
            if not isinstance(video_url, str) or not video_url:
                raise ProtocolError("multimodal.data.video_url is required")
            try:
                frame_count = int(chunk.data.get("frame_count", 0))
            except (TypeError, ValueError) as exc:
                raise ProtocolError("multimodal.data.frame_count must be an integer") from exc
            if frame_count < 2:
                raise ProtocolError("multimodal.data.frame_count must be at least 2")
    elif chunk.type == "text":
        if not isinstance(chunk.data.get("text"), str):
            raise ProtocolError("text.data.text is required")
    elif chunk.type == "video":
        text = chunk.data.get("text")
        if not isinstance(chunk.data.get("video_url"), str) or not chunk.data["video_url"]:
            raise ProtocolError("video.data.video_url is required")
        if text is not None and not isinstance(text, str):
            raise ProtocolError("video.data.text must be a string when present")
    elif chunk.type == "image":
        text = chunk.data.get("text")
        if not isinstance(chunk.data.get("image_url"), str) or not chunk.data["image_url"]:
            raise ProtocolError("image.data.image_url is required")
        if text is not None and not isinstance(text, str):
            raise ProtocolError("image.data.text must be a string when present")
    return chunk


def build_downstream_event(
    type_: str,
    data: dict,
    *,
    session_id: str,
    chunk_id: int,
    video_timestamp_ms: int = 0,
    wall_time_ms: int = 0,
) -> Chunk:
    if type_ not in {"session", "asr_text", "text", "text_delta", "audio", "toolcall", "turn_done", "error", "status"}:
        raise ProtocolError(f"invalid downstream event type: {type_}")
    if not isinstance(data, dict):
        raise ProtocolError("downstream data must be an object")
    return Chunk(
        direction="downstream",
        type=type_,
        session_id=session_id,
        chunk_id=int(chunk_id),
        video_timestamp_ms=int(video_timestamp_ms),
        wall_time_ms=int(wall_time_ms),
        data=dict(data),
    )


def dump_downstream_event(
    type_: str,
    data: dict,
    *,
    session_id: str,
    chunk_id: int,
    video_timestamp_ms: int = 0,
    wall_time_ms: int = 0,
) -> str:
    return json.dumps(
        build_downstream_event(
            type_,
            data,
            session_id=session_id,
            chunk_id=chunk_id,
            video_timestamp_ms=video_timestamp_ms,
            wall_time_ms=wall_time_ms,
        ).to_dict(),
        ensure_ascii=False,
    )


def build_toolcall_started(tool_call: dict) -> dict:
    return {
        "status": "started",
        "id": str(tool_call.get("id") or ""),
        "name": str(tool_call.get("name") or ""),
        "arguments": tool_call.get("arguments") or {},
    }


def build_toolcall_result(result: dict) -> dict:
    status = str(result.get("status") or "")
    if status not in {"success", "error", "timeout"}:
        raise ProtocolError(f"invalid toolcall result status: {status}")

    payload = {
        "status": status,
        "id": str(result.get("id") or ""),
        "name": str(result.get("name") or ""),
        "output": str(result.get("output") or ""),
        "error": str(result.get("error") or ""),
    }
    if "arguments" in result:
        payload["arguments"] = result.get("arguments") or {}
    return payload


def decode_audio_data(data: dict) -> bytes:
    audio_b64 = data.get("audio_b64")
    if not isinstance(audio_b64, str) or not audio_b64:
        raise ProtocolError("audio.data.audio_b64 is required")
    rem = len(audio_b64) % 4
    if rem:
        audio_b64 += "=" * (4 - rem)
    return base64.b64decode(audio_b64, validate=True)
