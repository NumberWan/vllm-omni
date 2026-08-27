"""Portable video evaluation helpers for the Native AURA demo bridge."""

from __future__ import annotations

import asyncio
import base64
import json
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

import cv2
import websockets


def _encode_frame(frame: Any) -> str:
    height, width = frame.shape[:2]
    if height > 360:
        scale = 360 / height
        frame = cv2.resize(frame, (max(2, int(width * scale) // 2 * 2), 360))
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if not ok:
        raise ValueError("failed to encode video frame")
    return base64.b64encode(encoded).decode()


def extract_frames(video_path: Path, fps: int) -> tuple[list[str], list[float], float]:
    """Sample a video at ``fps`` using presentation timestamps."""
    if fps < 1:
        raise ValueError("fps must be at least 1")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("video cannot be opened")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / source_fps if source_fps > 0 and frame_count > 0 else 0.0
    frames: list[str] = []
    timestamps: list[float] = []
    target = 1 / fps
    try:
        while duration <= 0 or target <= duration + 1e-6:
            capture.set(cv2.CAP_PROP_POS_MSEC, target * 1000)
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(_encode_frame(frame))
            timestamps.append(target)
            target += 1 / fps
    finally:
        capture.release()
    return frames, timestamps, duration


def build_chunks(
    frames: list[str],
    timestamps: list[float],
    frames_per_chunk: int,
    queries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Group sampled frames and attach each timestamped query exactly once."""
    if frames_per_chunk < 1:
        raise ValueError("num_frames_per_chunk must be at least 1")
    if len(frames) != len(timestamps):
        raise ValueError("frame/timestamp count mismatch")
    ordered = sorted(queries, key=lambda query: float(query.get("timestamp", 0)))
    query_index = 0
    chunks: list[dict[str, Any]] = []
    for start in range(0, len(frames), frames_per_chunk):
        end = min(start + frames_per_chunk, len(frames))
        chunk_frames = list(frames[start:end])
        if len(chunk_frames) == 1:
            chunk_frames.append(chunk_frames[0])
        timestamp = timestamps[end - 1]
        chunk_queries = []
        while query_index < len(ordered):
            query = ordered[query_index]
            if float(query.get("timestamp", 0)) > timestamp:
                break
            chunk_queries.append(query)
            query_index += 1
        chunks.append(
            {
                "frames": chunk_frames,
                "timestamp": timestamp,
                "queries": chunk_queries,
            }
        )
    return chunks


async def _collect_turn(ws: Any, timeout_s: float = 300) -> list[dict[str, Any]]:
    responses: list[dict[str, Any]] = []
    while True:
        raw = await asyncio.wait_for(ws.recv(), timeout=timeout_s)
        message = json.loads(raw)
        message_type = message.get("type")
        data = message.get("data") if isinstance(message.get("data"), dict) else {}
        if message_type == "turn_done":
            return responses
        if message_type == "error":
            raise RuntimeError(str(data.get("message") or "eval bridge error"))
        if message_type == "text":
            responses.append(
                {
                    "type": "text",
                    "content": str(data.get("text") or ""),
                    "infer_ms": int(data.get("infer_ms") or 0),
                }
            )
        elif message_type == "toolcall" and data.get("status") != "started":
            responses.append(
                {
                    "type": "toolcall",
                    "name": data.get("name"),
                    "status": data.get("status"),
                    "output": data.get("output"),
                    "error": data.get("error", ""),
                }
            )


def _format_turn(
    *,
    chunk_index: int,
    timestamp: float,
    query: dict[str, Any] | None,
    responses: list[dict[str, Any]],
    elapsed_ms: int,
) -> dict[str, Any]:
    text_responses = [response for response in responses if response.get("type") == "text"]
    return {
        "chunk_index": chunk_index,
        "timestamp_ms": int(timestamp * 1000),
        "elapsed_ms": elapsed_ms,
        "infer_ms": sum(int(response.get("infer_ms") or 0) for response in text_responses),
        "user": str(query.get("text") or "") if query else "",
        "model": "\n".join(
            str(response.get("content") or "")
            for response in text_responses
            if response.get("content")
        ),
        "responses": responses,
    }


async def evaluate_video(
    video_path: Path,
    *,
    ws_url: str,
    queries: list[dict[str, Any]],
    fps: int,
    frames_per_chunk: int,
) -> dict[str, Any]:
    """Evaluate one uploaded video through the Native bridge protocol."""
    frames, timestamps, duration = await asyncio.to_thread(extract_frames, video_path, fps)
    if not frames:
        raise ValueError("video contains no sampleable frames")
    chunks = build_chunks(frames, timestamps, frames_per_chunk, queries)
    turns: list[dict[str, Any]] = []
    started = time.monotonic()
    async with websockets.connect(
        ws_url,
        max_size=32 * 1024 * 1024,
        open_timeout=60,
        ping_interval=60,
        ping_timeout=300,
    ) as ws:
        for chunk_index, chunk in enumerate(chunks):
            chunk_queries = chunk["queries"]
            data: dict[str, Any] = {
                "video_url": "data:video/jpeg;base64," + ",".join(chunk["frames"])
            }
            if chunk_queries:
                data["text"] = str(chunk_queries[0].get("text") or "")
            turn_started = time.monotonic()
            await ws.send(json.dumps({"type": "video", "data": data}))
            responses = await _collect_turn(ws)
            if chunk_queries or responses:
                turns.append(
                    _format_turn(
                        chunk_index=chunk_index,
                        timestamp=chunk["timestamp"],
                        query=chunk_queries[0] if chunk_queries else None,
                        responses=responses,
                        elapsed_ms=int((time.monotonic() - turn_started) * 1000),
                    )
                )
            for query in chunk_queries[1:]:
                turn_started = time.monotonic()
                await ws.send(json.dumps({"type": "text", "data": {"text": query["text"]}}))
                responses = await _collect_turn(ws)
                turns.append(
                    _format_turn(
                        chunk_index=chunk_index,
                        timestamp=chunk["timestamp"],
                        query=query,
                        responses=responses,
                        elapsed_ms=int((time.monotonic() - turn_started) * 1000),
                    )
                )
    return {
        "video_path": str(video_path),
        "turns": turns,
        "fps": fps,
        "num_frames_per_chunk": frames_per_chunk,
        "video_duration_sec": duration,
        "expected_frames": int(duration * fps + 1e-6),
        "total_chunks": len(chunks),
        "total_frames": len(frames),
        "elapsed_ms": int((time.monotonic() - started) * 1000),
    }


def _vtt_time(seconds: float) -> str:
    milliseconds = max(0, int(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{millis:03d}"


def create_subtitle_assets(
    source: Path,
    result: dict[str, Any],
    original_name: str,
    output_dir: Path,
) -> dict[str, Any] | None:
    turns = [turn for turn in result.get("turns", []) if turn.get("user") or turn.get("model")]
    if not turns:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{Path(original_name).stem or 'video'}_{uuid.uuid4().hex[:10]}"
    suffix = Path(original_name).suffix or source.suffix or ".mp4"
    source_name = f"{stem}_source{suffix}"
    shutil.copyfile(source, output_dir / source_name)
    vtt_name = f"{stem}.vtt"
    ass_name = f"{stem}.ass"
    vtt_lines = ["WEBVTT", ""]
    ass_lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, Alignment",
        "Style: Default,Noto Sans CJK SC,26,&H00FFFFFF,2",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Text",
    ]
    for index, turn in enumerate(turns):
        start = float(turn.get("timestamp_ms") or 0) / 1000
        end = start + 4
        text = " | ".join(
            part
            for part in (
                f"User: {turn.get('user')}" if turn.get("user") else "",
                f"Model: {turn.get('model')}" if turn.get("model") else "",
            )
            if part
        )
        vtt_lines.extend([str(index + 1), f"{_vtt_time(start)} --> {_vtt_time(end)}", text, ""])
        ass_text = text.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")
        ass_lines.append(f"Dialogue: 0,{_vtt_time(start)[:-1]},{_vtt_time(end)[:-1]},Default,{ass_text}")
    (output_dir / vtt_name).write_text("\n".join(vtt_lines), encoding="utf-8")
    (output_dir / ass_name).write_text("\n".join(ass_lines) + "\n", encoding="utf-8")
    return {
        "source": {
            "filename": source_name,
            "download_url": f"/eval/videos/{source_name}",
        },
        "subtitles": {
            "vtt": {"filename": vtt_name, "download_url": f"/eval/videos/{vtt_name}"},
            "ass": {"filename": ass_name, "download_url": f"/eval/videos/{ass_name}"},
        },
    }


def burn_subtitles(source: Path, subtitles: Path, output: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(source),
            "-vf",
            f"ass={subtitles}",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
