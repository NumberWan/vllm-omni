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


def _make_frame() -> bytes:
    from PIL import Image, ImageDraw
    import io

    img = Image.new("RGB", (320, 240), (44, 36, 28))
    d = ImageDraw.Draw(img)
    d.text((80, 110), "AURA warmup", fill=(220, 220, 220))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return buf.getvalue()


def _wav_to_pcm16_16k(path: Path) -> bytes:
    import audioop
    import io
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
) -> dict:
    frame_b64 = base64.b64encode(_make_frame()).decode()
    pcm = _wav_to_pcm16_16k(wav)
    text = ""
    audio_n = 0
    errors: list[str] = []
    events: list[str] = []
    t0 = time.time()
    async with websockets.connect(aura_ws, max_size=32 * 1024 * 1024) as ws:
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
                    "tts_task_type": tts_task_type,
                    "tts_language": tts_language,
                    "tts_speaker": tts_speaker,
                    "tts_instruct": tts_instruct,
                }
            )
        )
        await ws.send(json.dumps({"type": "video.frame", "data": frame_b64}))
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

        got_audio = False
        while time.time() - t0 < timeout_s:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=60)
            except asyncio.TimeoutError:
                errors.append("timeout")
                break
            msg = json.loads(raw)
            et = str(msg.get("type") or "")
            events.append(et)
            if et == "response.text.delta":
                text += str(msg.get("delta") or "")
            elif et == "response.text.done":
                text = str(msg.get("text") or text)
            elif et == "response.audio.delta":
                audio_n += 1
                if msg.get("data"):
                    got_audio = True
            elif et == "response.audio.done":
                break
            elif et == "error":
                errors.append(str(msg.get("message") or msg))
                break

    silent = "<|silent|>" in (text or "")
    ok = got_audio and not silent and not errors
    return {
        "ok": ok,
        "elapsed_s": round(time.time() - t0, 2),
        "text": text[:120],
        "audio_n": audio_n,
        "errors": errors,
        "events": events[-20:],
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
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if not result["ok"]:
        print("WARNING: warmup did not get spoken audio; continuing cold", flush=True)
        sys.exit(2)
    print(f"Warmup OK in {result['elapsed_s']}s ({result['audio_n']} chunks)", flush=True)


if __name__ == "__main__":
    main()
