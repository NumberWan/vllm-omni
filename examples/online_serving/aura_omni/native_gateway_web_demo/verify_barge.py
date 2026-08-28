#!/usr/bin/env python3
"""Live barge-in checks against AURA /v1/video/chat/stream."""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import time
from pathlib import Path

import websockets
from PIL import Image

DEFAULT_WAVS = [
    Path("/workspace/models/datasets/OmniInteract/data/1q1a/audios/0002_0.wav"),
    Path("/home/wtk/vllm-omni-AURA_026/tests/assets/glm_tts/jiayan_zh.wav"),
    Path("/home/wtk/vllm-omni-AURA_barge/tests/assets/minicpmo_4_5/soft_interrupt_16k.wav"),
]


def _jpeg_b64(color: tuple[int, int, int]) -> str:
    image = Image.new("RGB", (640, 360), color)
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=80)
    return base64.b64encode(output.getvalue()).decode()


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


async def _send_pcm(ws, pcm: bytes) -> None:
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


async def _open_session(ws, *, model: str, auto_trigger: bool) -> None:
    await ws.send(
        json.dumps(
            {
                "type": "session.config",
                "model": model,
                "modalities": ["text", "audio"],
                "auto_trigger": auto_trigger,
                "auto_trigger_min_frames": 2,
                "max_frames": 64,
                "max_frames_per_round": 16,
                "stream_text_deltas": True,
                "enable_frame_filter": False,
                "video_fps": 2.0,
                "tts_task_type": "Base",
                "tts_language": "Chinese",
                "tts_speaker": "Vivian",
                "tts_instruct": "请用专业、清晰、自然的语气说话，语速稍快，情绪克制。",
            }
        )
    )
    await ws.send(json.dumps({"type": "video.frame", "data": _jpeg_b64((44, 36, 28))}))
    await ws.send(json.dumps({"type": "video.frame", "data": _jpeg_b64((48, 36, 28))}))


def _match_wait(event: dict, wait_for: str, *, audio_n: int) -> bool:
    event_type = str(event.get("type") or "")
    if wait_for == "text.delta":
        return event_type == "response.text.delta" and bool(event.get("delta"))
    if wait_for == "audio.delta":
        return event_type == "response.audio.delta"
    if wait_for == "text.done":
        return event_type == "response.text.done"
    if wait_for == "audio.mid":
        return event_type == "response.audio.delta" and audio_n >= 3
    return False


async def _barge_case(
    *,
    aura_ws: str,
    model: str,
    pcm: bytes,
    wait_for: str,
    timeout_s: float,
) -> dict:
    t0 = time.time()
    old_id = None
    new_ids: set[str] = set()
    old_audio_after = 0
    new_audio = 0
    barged = False
    events: list[str] = []
    audio_n = 0
    errors: list[str] = []

    async with websockets.connect(aura_ws, max_size=32 * 1024 * 1024) as ws:
        await _open_session(ws, model=model, auto_trigger=False)
        await ws.send(
            json.dumps({"type": "video.query", "text": "請用比較長的一段話介紹你自己。"})
        )
        while time.time() - t0 < timeout_s:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=30)
            except asyncio.TimeoutError:
                errors.append("recv_timeout")
                break
            event = json.loads(raw)
            event_type = str(event.get("type") or "")
            request_id = str(event.get("request_id") or "")
            events.append(event_type)
            if event_type == "error":
                errors.append(str(event.get("message") or event))
                break
            if event_type == "response.audio.delta":
                audio_n += 1
            if not barged:
                if request_id:
                    old_id = request_id
                if _match_wait(event, wait_for, audio_n=audio_n):
                    barged = True
                    await _send_pcm(ws, pcm)
                continue
            if request_id and old_id and request_id != old_id:
                new_ids.add(request_id)
                if event_type == "response.audio.delta":
                    new_audio += 1
            elif event_type == "response.audio.delta":
                old_audio_after += 1
            if new_audio > 0 or (new_ids and time.time() - t0 > 8):
                if new_audio > 0 or any(
                    item.startswith("response.") for item in events[-6:]
                ):
                    break
        await ws.send(json.dumps({"type": "video.done"}))

    ok = barged and old_audio_after <= 4 and bool(new_ids) and not errors
    return {
        "case": wait_for,
        "ok": ok,
        "barged": barged,
        "old_id": old_id,
        "new_ids": sorted(new_ids),
        "old_audio_after": old_audio_after,
        "new_audio": new_audio,
        "errors": errors,
        "elapsed_s": round(time.time() - t0, 2),
        "events_tail": events[-24:],
    }


async def _vision_no_abort(
    *,
    aura_ws: str,
    model: str,
    timeout_s: float,
) -> dict:
    t0 = time.time()
    old_id = None
    extra_frames = False
    old_audio_after_frames = 0
    aborted_ids: set[str] = set()
    events: list[str] = []
    errors: list[str] = []
    text_done = False

    async with websockets.connect(aura_ws, max_size=32 * 1024 * 1024) as ws:
        await _open_session(ws, model=model, auto_trigger=True)
        await ws.send(json.dumps({"type": "video.query", "text": "請慢慢介紹眼前畫面。"}))
        while time.time() - t0 < timeout_s:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=30)
            except asyncio.TimeoutError:
                errors.append("recv_timeout")
                break
            event = json.loads(raw)
            event_type = str(event.get("type") or "")
            request_id = str(event.get("request_id") or "")
            events.append(event_type)
            if event_type == "error":
                errors.append(str(event.get("message") or event))
                break
            if event_type == "response.start" and request_id:
                if old_id is None:
                    old_id = request_id
                elif request_id != old_id:
                    aborted_ids.add(request_id)
            if event_type == "response.text.done":
                text_done = True
            if text_done and not extra_frames:
                extra_frames = True
                await ws.send(
                    json.dumps({"type": "video.frame", "data": _jpeg_b64((90, 40, 28))})
                )
                await ws.send(
                    json.dumps({"type": "video.frame", "data": _jpeg_b64((120, 40, 28))})
                )
                continue
            if extra_frames and request_id == old_id and event_type == "response.audio.delta":
                old_audio_after_frames += 1
            if extra_frames and old_audio_after_frames >= 2:
                break
        await ws.send(json.dumps({"type": "video.done"}))

    ok = text_done and extra_frames and old_audio_after_frames >= 1 and not errors
    return {
        "case": "vision_no_abort",
        "ok": ok,
        "text_done": text_done,
        "old_id": old_id,
        "old_audio_after_frames": old_audio_after_frames,
        "later_starts": sorted(aborted_ids),
        "errors": errors,
        "elapsed_s": round(time.time() - t0, 2),
        "events_tail": events[-24:],
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aura-ws", default="ws://127.0.0.1:8666/v1/video/chat/stream")
    parser.add_argument("--model", default="/workspace/models/AURA_v2")
    parser.add_argument("--wav", type=Path, default=None)
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    wav = args.wav or next((path for path in DEFAULT_WAVS if path.is_file()), None)
    if wav is None or not wav.is_file():
        raise SystemExit("no barge wav found")
    pcm = _wav_to_pcm16_16k(wav)
    results = []
    for wait_for in ("text.delta", "audio.delta", "text.done", "audio.mid"):
        result = await _barge_case(
            aura_ws=args.aura_ws,
            model=args.model,
            pcm=pcm,
            wait_for=wait_for,
            timeout_s=args.timeout,
        )
        print(json.dumps(result, ensure_ascii=False), flush=True)
        results.append(result)
    vision = await _vision_no_abort(
        aura_ws=args.aura_ws,
        model=args.model,
        timeout_s=args.timeout,
    )
    print(json.dumps(vision, ensure_ascii=False), flush=True)
    results.append(vision)
    failed = [item["case"] for item in results if not item["ok"]]
    if failed:
        raise SystemExit(f"barge verify failed: {failed}")
    print("BARGE_VERIFY_OK", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
