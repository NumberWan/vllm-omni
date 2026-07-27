#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify the MiniCPM-style AURA browser demo proxy end-to-end.

Requires:
  - AURA server reachable by the demo proxy
  - Demo proxy on --demo (default http://127.0.0.1:7862)
  - OmniInteract sample video/audio under --dataset-root
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import json
import urllib.request
import wave
from array import array
from pathlib import Path

import websockets


def _make_warmup_frame() -> bytes:
    """Create a small neutral JPEG without any external video asset."""
    import cv2
    import numpy as np

    frame = np.full((240, 320, 3), (44, 36, 28), dtype=np.uint8)
    cv2.putText(
        frame,
        "AURA warmup",
        (68, 128),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (220, 220, 220),
        2,
        cv2.LINE_AA,
    )
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise RuntimeError("warmup JPEG encode failed")
    return encoded.tobytes()


def _wav_to_pcm16_16k(path: Path) -> bytes:
    with wave.open(str(path), "rb") as wf:
        if wf.getsampwidth() != 2:
            raise ValueError(f"expected 16-bit wav: {path}")
        nch = wf.getnchannels()
        sr = wf.getframerate()
        pcm = array("h")
        pcm.frombytes(wf.readframes(wf.getnframes()))
    if nch == 2:
        pcm = array("h", [(pcm[i] + pcm[i + 1]) // 2 for i in range(0, len(pcm), 2)])
    if sr != 16000:
        ratio = 16000 / sr
        out = array("h")
        n = int(len(pcm) * ratio)
        for i in range(n):
            x = i / ratio
            j = int(x)
            f = x - j
            a = pcm[j]
            b = pcm[j + 1] if j + 1 < len(pcm) else a
            out.append(int(a * (1 - f) + b * f))
        pcm = out
    return pcm.tobytes()


def _load_video_frames(path: Path, count: int = 4) -> list[bytes]:
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    interval = max(1, int(fps / 2))
    frames: list[bytes] = []
    idx = 0
    while len(frames) < count:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % interval == 0:
            h, w = frame.shape[:2]
            scale = min(1.0, 640 / max(h, w))
            if scale < 1.0:
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok2:
                raise RuntimeError("jpeg encode failed")
            frames.append(buf.tobytes())
        idx += 1
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames from {path}")
    return frames


async def _spoken_via_proxy(
    *,
    demo_ws: str,
    model: str,
    frames: list[bytes],
    pcm: bytes,
    tts_language: str = "Chinese",
    tts_speaker: str = "Vivian",
    tts_instruct: str = "",
) -> dict:
    text = ""
    audio_n = 0
    audio_bytes = 0
    errors: list[str] = []
    events: list[str] = []
    async with websockets.connect(demo_ws, max_size=32 * 1024 * 1024) as ws:
        await ws.send(
            json.dumps(
                {
                    "type": "session.config",
                    "model": model,
                    "modalities": ["text", "audio"],
                    "auto_trigger": False,
                    "auto_trigger_min_frames": 2,
                    "max_frames": 64,
                    "max_frames_per_round": 16,
                    "stream_text_deltas": True,
                    "enable_frame_filter": False,
                    "video_fps": 2.0,
                    "tts_task_type": "CustomVoice",
                    "tts_language": tts_language,
                    "tts_speaker": tts_speaker,
                    "tts_instruct": tts_instruct,
                }
            )
        )
        # One frame only: avoids vision auto-trigger before audio.done.
        await ws.send(json.dumps({"type": "video.frame", "data": base64.b64encode(frames[0]).decode()}))
        chunk = 16000 * 2
        for off in range(0, len(pcm), chunk):
            await ws.send(
                json.dumps(
                    {
                        "type": "audio.chunk",
                        "data": base64.b64encode(pcm[off : off + chunk]).decode(),
                    }
                )
            )
        await ws.send(json.dumps({"type": "audio.done"}))

        silent = False
        got_text = False
        got_audio = False
        while True:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=120)
            except asyncio.TimeoutError:
                errors.append("timeout waiting for spoken response")
                break
            msg = json.loads(raw)
            t = str(msg.get("type") or "")
            events.append(t)
            if t == "response.text.delta":
                text += str(msg.get("delta") or "")
            elif t == "response.text.done":
                text = str(msg.get("text") or text)
                got_text = True
                silent = "<|silent|>" in text
            elif t == "response.audio.delta":
                audio_n += 1
                data = msg.get("data") or ""
                if data:
                    audio_bytes += len(base64.b64decode(data))
            elif t == "response.audio.done":
                got_audio = True
            elif t == "error":
                errors.append(str(msg.get("message") or msg))
                break
            elif t == "session.done":
                break

            if got_text and ((not silent and got_audio) or silent):
                with contextlib.suppress(Exception):
                    await ws.send(json.dumps({"type": "video.done"}))
                break

    return {
        "ok": bool(got_text and got_audio and not silent and not errors),
        "spoken": bool(got_text and not silent),
        "text": text,
        "audio_deltas": audio_n,
        "audio_payload_bytes": audio_bytes,
        "errors": errors,
        "events": events,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--demo", default="http://127.0.0.1:7862")
    ap.add_argument("--model", default="/workspace/models/AURA")
    ap.add_argument(
        "--dataset-root",
        default="/workspace/models/datasets/OmniInteract/data/1q1a",
    )
    ap.add_argument("--video-id", default="0002")
    args = ap.parse_args()

    health = urllib.request.urlopen(f"{args.demo.rstrip('/')}/healthz", timeout=5).read().decode()
    if health.strip() != "ok":
        raise SystemExit(f"demo healthz failed: {health!r}")
    page = urllib.request.urlopen(f"{args.demo.rstrip('/')}/", timeout=5).read().decode()
    for needle in (
        "AURA Streaming Voice",
        "Start session",
        "Hold to talk",
        "Camera",
        "AURA_WEB_CONFIG",
    ):
        if needle not in page:
            raise SystemExit(f"demo page missing {needle!r}")

    root = Path(args.dataset_root)
    video = root / "videos" / f"{args.video_id}.mp4"
    wav = root / "audios" / f"{args.video_id}_0.wav"
    frames = _load_video_frames(video, count=4)
    pcm = _wav_to_pcm16_16k(wav)

    demo_ws = args.demo.replace("https://", "wss://").replace("http://", "ws://").rstrip("/")
    demo_ws = f"{demo_ws}/v1/video/chat/stream"
    result = asyncio.run(_spoken_via_proxy(demo_ws=demo_ws, model=args.model, frames=frames, pcm=pcm))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["ok"]:
        raise SystemExit(1)
    print("VERIFY_OK")


if __name__ == "__main__":
    main()
