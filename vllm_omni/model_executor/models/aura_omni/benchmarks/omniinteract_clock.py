# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""OmniInteract sender for video-required, turn-commit duplex sessions.

The shared harness still streams a 1 FPS soundtrack. This clock is the previous
AURA bench: frames are the clock, each append carries a frame, and each
annotation WAV is one append at ``question_time`` followed by ``commit``.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ANNOTATION_VIDEO_FPS = 2.0
_PCM16_SAMPLE_RATE = 16_000
_PCM16_BYTES_PER_SAMPLE = 2


def wants_annotation_video_clock(capabilities: object) -> bool:
    """True when every append must carry video and a commit is one turn.

    MiniCPM requires audio and is not turn-commit-only, so it stays on the
    shared soundtrack clock. The check is the session capability set, not the
    model name.
    """
    if not isinstance(capabilities, dict):
        return False
    required = capabilities.get("required_input_modalities") or ()
    if isinstance(required, str):
        required = (required,)
    try:
        required_set = {str(item) for item in required}
    except TypeError:
        return False
    return "video" in required_set and bool(capabilities.get("supports_turn_commit_only"))


def _parse_clock(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if not isinstance(value, str) or not value.strip():
        return None
    parts = value.strip().split(":")
    try:
        nums = [float(part) for part in parts]
    except ValueError:
        return None
    if len(nums) == 1:
        return nums[0]
    if len(nums) == 2:
        minutes, seconds = nums
        return minutes * 60.0 + seconds
    if len(nums) == 3:
        hours, minutes, seconds = nums
        return hours * 3600.0 + minutes * 60.0 + seconds
    return None


def _load_annotation_pcm(path: Path) -> bytes:
    command = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-f",
        "s16le",
        "-ac",
        "1",
        "-ar",
        str(_PCM16_SAMPLE_RATE),
        "pipe:1",
    ]
    try:
        result = subprocess.run(command, capture_output=True, timeout=60, check=False)
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"ffmpeg timed out reading {path}") from exc
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required to read OmniInteract annotation WAV") from exc
    if result.returncode:
        error = result.stderr.decode("utf-8", "ignore").strip()
        raise RuntimeError(f"ffmpeg failed for {path}: {error}")
    return result.stdout


def load_annotation_utterances(video_path: Path) -> list[dict[str, Any]]:
    """Annotation WAV files injected at ``question_time``.

    Layout is ``{subset}/annotations/{stem}.json`` and
    ``{subset}/audios/{stem}_{qa}.wav``. The video soundtrack is not used.
    """
    subset_root = video_path.parent.parent
    annotation_path = subset_root / "annotations" / f"{video_path.stem}.json"
    if not annotation_path.is_file():
        raise ValueError(f"Missing OmniInteract annotation for {video_path}: {annotation_path}")
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    if not isinstance(annotation, list):
        raise ValueError(f"OmniInteract annotation is not a list: {annotation_path}")
    items: list[dict[str, Any]] = []
    for index, row in enumerate(annotation):
        if not isinstance(row, dict):
            continue
        at_sec = _parse_clock(row.get("question_time"))
        audio_path = subset_root / "audios" / f"{video_path.stem}_{index}.wav"
        if at_sec is None or not audio_path.is_file():
            continue
        pcm = _load_annotation_pcm(audio_path)
        if not pcm:
            continue
        items.append({"at_sec": at_sec, "pcm": pcm, "sent": False, "qa_index": index})
    if not items:
        raise ValueError(
            f"No annotation WAV at question_time for {video_path.name}; "
            f"expected {subset_root / 'audios' / (video_path.stem + '_*.wav')}"
        )
    items.sort(key=lambda item: float(item["at_sec"]))
    return items


async def stream_annotation_clock(
    client: Any,
    frames: Sequence[str | None],
    playback: Any,
    *,
    video_path: Path,
    fps: float = ANNOTATION_VIDEO_FPS,
) -> tuple[int, int, float, float]:
    """Send one frame per tick. Commit only when an annotation WAV is due."""
    utterances = await asyncio.to_thread(load_annotation_utterances, video_path)
    frame_interval_s = 1.0 / fps
    started_at = time.monotonic()
    sent_frames = chunks = 0
    lags: list[float] = []
    held_frame: str | None = None
    for index, frame in enumerate(frames):
        if frame:
            held_frame = frame
        if not held_frame:
            continue
        source_time = index / fps
        due = [item for item in utterances if not item["sent"] and float(item["at_sec"]) <= source_time]
        due_at = time.monotonic()
        if due:
            for item in due:
                audio = item["pcm"]
                assert isinstance(audio, bytes)
                await client.send(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(audio).decode("ascii"),
                        "input_audio_format": "pcm16",
                        "sample_rate_hz": _PCM16_SAMPLE_RATE,
                        "duration_ms": len(audio) * 1000 // (_PCM16_SAMPLE_RATE * _PCM16_BYTES_PER_SAMPLE),
                        "is_speech": True,
                        "video_frames": [held_frame],
                    }
                )
                chunks += 1
                sent_frames += 1
                await playback.acknowledge(client)
                item["sent"] = True
                await client.commit()
        else:
            await client.send(
                {
                    "type": "input_audio_buffer.append",
                    "is_speech": False,
                    "sample_rate_hz": _PCM16_SAMPLE_RATE,
                    "video_frames": [held_frame],
                }
            )
            chunks += 1
            sent_frames += 1
            await playback.acknowledge(client)
        lags.append(max(0.0, time.monotonic() - due_at))
        await asyncio.sleep(max(0.0, started_at + (index + 1) * frame_interval_s - time.monotonic()))
    unsent = [item for item in utterances if not item["sent"]]
    if unsent:
        logger.warning(
            "Annotation WAV past the last frame was not injected: video=%s qa=%s",
            video_path,
            [item["qa_index"] for item in unsent],
        )
    return chunks, sent_frames, sum(lags) / len(lags) if lags else 0.0, max(lags, default=0.0)
