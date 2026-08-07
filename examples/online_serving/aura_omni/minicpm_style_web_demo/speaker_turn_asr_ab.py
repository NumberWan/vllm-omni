#!/usr/bin/env python3
"""WebSocket speaker A/B: multi-turn spoken session → save TTS wav → ASR.

Measures first-turn vs later-turn intelligibility per CustomVoice speaker.

  PYTHONPATH=. .venv/bin/python examples/online_serving/aura_omni/\\
    minicpm_style_web_demo/speaker_turn_asr_ab.py \\
    --speakers Vivian,Dylan,Uncle_fu,Aiden,Ryan --turns 2
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import wave
from array import array
from pathlib import Path

import websockets


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


def _wav_b64_to_pcm16(data_b64: str) -> tuple[bytes, int]:
    raw = base64.b64decode(data_b64)
    with wave.open(io.BytesIO(raw), "rb") as wf:
        if wf.getsampwidth() != 2:
            return b"", 0
        nch = wf.getnchannels()
        sr = wf.getframerate()
        pcm = wf.readframes(wf.getnframes())
    if nch == 2:
        samples = array("h")
        samples.frombytes(pcm)
        mono = array("h", [(samples[i] + samples[i + 1]) // 2 for i in range(0, len(samples), 2)])
        pcm = mono.tobytes()
    return pcm, sr


def _write_pcm_wav(path: Path, pcm: bytes, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm)


async def _one_session(
    *,
    demo_ws: str,
    model: str,
    frames: list[bytes],
    pcm: bytes,
    speaker: str,
    turns: int,
    out_dir: Path,
) -> list[dict]:
    results: list[dict] = []
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
                    "tts_language": "Chinese",
                    "tts_speaker": speaker,
                    "tts_instruct": "",
                }
            )
        )

        for turn_i in range(1, turns + 1):
            await ws.send(
                json.dumps({"type": "video.frame", "data": base64.b64encode(frames[0]).decode()})
            )
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

            text = ""
            pcm_parts: list[bytes] = []
            sr = 24000
            audio_n = 0
            silent = False
            got_text = False
            got_audio = False
            errors: list[str] = []

            while True:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=180)
                except asyncio.TimeoutError:
                    errors.append("timeout")
                    break
                msg = json.loads(raw)
                t = str(msg.get("type") or "")
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
                        part, part_sr = _wav_b64_to_pcm16(data)
                        if part:
                            pcm_parts.append(part)
                            sr = part_sr or sr
                elif t == "response.audio.done":
                    got_audio = True
                elif t == "error":
                    errors.append(str(msg.get("message") or msg))
                    break

                if got_text and ((not silent and got_audio) or silent):
                    break

            merged = b"".join(pcm_parts)
            wav_path = out_dir / f"{speaker}_turn{turn_i}.wav"
            if merged:
                _write_pcm_wav(wav_path, merged, sr)
            results.append(
                {
                    "speaker": speaker,
                    "turn": turn_i,
                    "text": text.strip(),
                    "audio_deltas": audio_n,
                    "dur_s": (len(merged) / 2 / sr) if merged else 0.0,
                    "wav": str(wav_path) if merged else "",
                    "errors": errors,
                    "ok_audio": bool(merged) and audio_n > 0 and not silent,
                }
            )

        await ws.send(json.dumps({"type": "video.done"}))
    return results


def _asr_paths(paths: list[Path]) -> dict[str, str]:
    import soundfile as sf
    import torch

    try:
        from qwen_asr import Qwen3ASRModel
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "qwen_asr not in this venv; re-run ASR with "
            "/home/wtk/AURA/AURA/.venv/bin/python or pass --skip-asr"
        ) from exc

    asr = Qwen3ASRModel.from_pretrained(
        "/workspace/models/Qwen3-ASR-1.7B",
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    out: dict[str, str] = {}
    for p in paths:
        if not p.is_file():
            out[str(p)] = ""
            continue
        wav, sr = sf.read(str(p))
        hyp = asr.transcribe(audio=(wav, sr), language="Chinese")[0].text
        out[str(p)] = hyp
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--demo", default="http://127.0.0.1:7862")
    ap.add_argument("--model", default="/workspace/models/AURA")
    ap.add_argument(
        "--dataset-root",
        default="/workspace/models/datasets/OmniInteract/data/1q1a",
    )
    ap.add_argument("--video-id", default="0002")
    ap.add_argument(
        "--speakers",
        default="Vivian,Dylan,Uncle_fu,Aiden,Ryan",
        help="Comma-separated CustomVoice speakers",
    )
    ap.add_argument("--turns", type=int, default=2, help="Spoken turns per session")
    ap.add_argument("--out", default="/tmp/aura_speaker_turn_ab")
    ap.add_argument("--skip-asr", action="store_true")
    args = ap.parse_args()

    root = Path(args.dataset_root)
    frames = _load_video_frames(root / "videos" / f"{args.video_id}.mp4", count=2)
    pcm = _wav_to_pcm16_16k(root / "audios" / f"{args.video_id}_0.wav")
    demo_ws = args.demo.replace("https://", "wss://").replace("http://", "ws://").rstrip("/")
    demo_ws = f"{demo_ws}/v1/video/chat/stream"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    speakers = [s.strip() for s in args.speakers.split(",") if s.strip()]
    all_rows: list[dict] = []
    for speaker in speakers:
        print(f"=== session speaker={speaker} turns={args.turns} ===", flush=True)
        rows = asyncio.run(
            _one_session(
                demo_ws=demo_ws,
                model=args.model,
                frames=frames,
                pcm=pcm,
                speaker=speaker,
                turns=args.turns,
                out_dir=out_dir,
            )
        )
        for row in rows:
            print(
                f"  turn{row['turn']}: deltas={row['audio_deltas']} "
                f"dur={row['dur_s']:.2f}s text={row['text'][:40]!r}",
                flush=True,
            )
        all_rows.extend(rows)

    if not args.skip_asr:
        wavs = [Path(r["wav"]) for r in all_rows if r.get("wav")]
        print("=== ASR ===", flush=True)
        hyps = _asr_paths(wavs)
        for row in all_rows:
            hyp = hyps.get(row.get("wav") or "", "")
            row["asr"] = hyp
            print(
                f"{row['speaker']:10s} turn{row['turn']}: "
                f"ref≈{row['text'][:28]!r} | asr={hyp!r}",
                flush=True,
            )

    summary = out_dir / "summary.json"
    summary.write_text(json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {summary}", flush=True)


if __name__ == "__main__":
    main()
