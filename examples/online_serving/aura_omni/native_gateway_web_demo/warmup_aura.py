#!/usr/bin/env python3
"""Spoken warmup against AURA /v1/video/chat/stream (old minicpm demo style).

Runs one real voice turn so Stage0–3 leave cold-start before users connect.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
import time
from pathlib import Path

import websockets

# Reuse OmniInteract / bundled test speech when available.
DEFAULT_WAVS = [
    Path("/home/wtk/vllm-omni-AURA_026/tests/assets/glm_tts/jiayan_zh.wav"),
    Path("/workspace/models/datasets/OmniInteract/data/1q1a/audios/0001_0.wav"),
    Path("/workspace/models/datasets/OmniInteract/data/1q1a/audios/0002_0.wav"),
]


def _make_frames(count: int = 2, width: int = 640, height: int = 360) -> list[bytes]:
    import io

    from PIL import Image, ImageDraw

    frames = []
    for index in range(count):
        img = Image.new("RGB", (width, height), (44 + index, 36, 28))
        d = ImageDraw.Draw(img)
        position = (max((width - 160) // 2, 0), max((height - 20) // 2, 0))
        d.text(position, "AURA warmup", fill=(220, 220, 220))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        frames.append(buf.getvalue())
    return frames


def _wav_to_pcm16_16k(path: Path) -> bytes:
    import audioop
    import wave

    with wave.open(str(path), "rb") as wf:
        nch, sw, sr = wf.getnchannels(), wf.getsampwidth(), wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    if sw != 2:
        raise ValueError(f"expected 16-bit wav: {path}")
    pcm = frames
    if nch == 2:
        pcm = audioop.tomono(pcm, 2, 0.5, 0.5)
    if sr != 16000:
        pcm, _ = audioop.ratecv(pcm, 2, 1, sr, 16000, None)
    return pcm


async def _warmup(
    *,
    aura_ws: str,
    model: str,
    wav: Path,
    tts_speaker: str,
    tts_language: str,
    tts_instruct: str,
    tts_task_type: str,
    timeout_s: float,
    frame_count: int,
    frame_width: int,
    frame_height: int,
    silent_first: bool,
) -> dict:
    frame_b64s = [
        base64.b64encode(frame).decode()
        for frame in _make_frames(frame_count, frame_width, frame_height)
    ]
    pcm = _wav_to_pcm16_16k(wav)
    text = ""
    audio_n = 0
    errors: list[str] = []
    events: list[str] = []
    t0 = time.time()

    async def recv_turn(ws, *, phase: str) -> tuple[str, int, float]:
        phase_text = ""
        phase_audio_n = 0
        phase_t0 = time.time()
        while time.time() - t0 < timeout_s:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=60)
            except asyncio.TimeoutError:
                errors.append(f"{phase}_timeout")
                break
            msg = json.loads(raw)
            event_type = str(msg.get("type") or "")
            events.append(f"{phase}:{event_type}")
            if event_type == "response.text.delta":
                phase_text += str(msg.get("delta") or "")
            elif event_type == "response.text.done":
                phase_text = str(msg.get("text") or phase_text)
                if "<|silent|>" in phase_text:
                    break
            elif event_type == "response.audio.delta":
                phase_audio_n += 1
            elif event_type == "response.audio.done":
                break
            elif event_type == "error":
                errors.append(f"{phase}:{msg.get('message') or msg}")
                break
        return phase_text, phase_audio_n, round(time.time() - phase_t0, 2)

    async with websockets.connect(aura_ws, max_size=32 * 1024 * 1024) as ws:
        await ws.send(
            json.dumps(
                {
                    "type": "session.config",
                    "model": model,
                    "modalities": ["text", "audio"],
                    "auto_trigger": silent_first,
                    "auto_trigger_min_frames": 2,
                    "max_frames": 64,
                    "max_frames_per_round": 16,
                    "stream_text_deltas": True,
                    "enable_frame_filter": False,
                    "video_fps": 2.0,
                    "tts_task_type": tts_task_type,
                    "tts_language": tts_language,
                    "tts_speaker": tts_speaker,
                    "tts_instruct": tts_instruct,
                }
            )
        )
        for frame_b64 in frame_b64s:
            await ws.send(json.dumps({"type": "video.frame", "data": frame_b64}))
        silent_text = ""
        silent_elapsed_s = 0.0
        if silent_first:
            silent_text, _, silent_elapsed_s = await recv_turn(ws, phase="silent")
            if errors:
                return {
                    "ok": False,
                    "elapsed_s": round(time.time() - t0, 2),
                    "silent_text": silent_text[:120],
                    "silent_elapsed_s": silent_elapsed_s,
                    "text": "",
                    "audio_n": 0,
                    "errors": errors,
                    "events": events[-40:],
                    "wav": str(wav),
                }
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
        text, audio_n, voice_elapsed_s = await recv_turn(ws, phase="voice")

    silent = "<|silent|>" in (text or "")
    ok = audio_n > 0 and not silent and not errors
    return {
        "ok": ok,
        "elapsed_s": round(time.time() - t0, 2),
        "silent_text": silent_text[:120],
        "silent_elapsed_s": silent_elapsed_s,
        "voice_elapsed_s": voice_elapsed_s,
        "text": text[:120],
        "audio_n": audio_n,
        "errors": errors,
        "events": events[-40:],
        "wav": str(wav),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--aura-ws", default="ws://127.0.0.1:8666/v1/video/chat/stream")
    ap.add_argument("--model", default="/workspace/models/AURA")
    ap.add_argument("--wav", type=Path, default=None)
    ap.add_argument("--tts-speaker", default="Vivian")
    ap.add_argument("--tts-language", default="Chinese")
    ap.add_argument("--tts-instruct", default="")
    ap.add_argument("--tts-task-type", default="Base")
    ap.add_argument("--timeout", type=float, default=180)
    ap.add_argument("--frame-count", type=int, default=2)
    ap.add_argument("--frame-width", type=int, default=640)
    ap.add_argument("--frame-height", type=int, default=360)
    ap.add_argument("--silent-first", action="store_true")
    args = ap.parse_args()

    wav = args.wav
    if wav is None:
        wav = next((p for p in DEFAULT_WAVS if p.is_file()), None)
    if wav is None or not wav.is_file():
        print("ERROR: no warmup wav found", file=sys.stderr)
        sys.exit(1)

    print(f"Warming AURA via {args.aura_ws} with {wav} ...", flush=True)
    result = asyncio.run(
        _warmup(
            aura_ws=args.aura_ws,
            model=args.model,
            wav=wav,
            tts_speaker=args.tts_speaker,
            tts_language=args.tts_language,
            tts_instruct=args.tts_instruct,
            tts_task_type=args.tts_task_type,
            timeout_s=args.timeout,
            frame_count=args.frame_count,
            frame_width=args.frame_width,
            frame_height=args.frame_height,
            silent_first=args.silent_first,
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if not result["ok"]:
        print("WARNING: warmup did not get spoken audio; continuing cold", flush=True)
        sys.exit(2)
    print(f"Warmup OK in {result['elapsed_s']}s ({result['audio_n']} chunks)", flush=True)


if __name__ == "__main__":
    main()
