#!/usr/bin/env python3
"""Verify AURA_v2 Native bridge I/O: ASR, reply, one merged audio, turn_done.

Usage (stack already up):
  /home/wtk/vllm-omni-AURA_026/.venv/bin/python verify_io.py
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import re
import sys
import time
import wave
from pathlib import Path

import websockets
from PIL import Image, ImageDraw

DEFAULT_WAV = Path("/workspace/models/datasets/OmniInteract/data/1q1a/audios/0002_0.wav")
DEFAULT_DUMP = Path("/tmp/aura_v2_native_demo/tts")


def _soft_norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"<\|silent\|>", "", s)
    s = re.sub(r"[\s\W_]+", "", s, flags=re.UNICODE)
    return s


def _soft_match(a: str, b: str) -> bool:
    na, nb = _soft_norm(a), _soft_norm(b)
    if not na or not nb:
        return False
    if na in nb or nb in na:
        return True
    # overlap of first 8 chars of longer Chinese chunk
    short, long = (na, nb) if len(na) <= len(nb) else (nb, na)
    if len(short) >= 4 and short[: min(8, len(short))] in long:
        return True
    return False


def _make_frames_b64(n: int = 2) -> list[str]:
    out = []
    for i in range(n):
        img = Image.new("RGB", (640, 360), (30 + i * 10, 40, 50))
        d = ImageDraw.Draw(img)
        d.rectangle([40, 40, 600, 320], outline=(200, 200, 200), width=3)
        d.ellipse([240, 100, 400, 260], fill=(160, 50, 50))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        out.append(base64.b64encode(buf.getvalue()).decode())
    return out


def _pick_asr_device() -> str:
    import os

    override = os.environ.get("BRIDGE_ASR_DEVICE", "").strip()
    if override:
        return override
    import torch

    if not torch.cuda.is_available():
        return "cpu"
    # Prefer a free card (AURA stack usually occupies 1,2).
    try:
        import subprocess

        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        free = []
        for line in out.strip().splitlines():
            idx_s, mem_s = [x.strip() for x in line.split(",")]
            if float(mem_s) < 2000:
                free.append(int(idx_s))
        if free:
            return f"cuda:{free[-1]}"
    except Exception:
        pass
    return "cuda:0"


def _asr_wav(path: Path, *, asr_python: str | None = None) -> str:
    """Transcribe dump WAV. Optionally run in another venv that has qwen_asr."""
    import os
    import subprocess

    asr_py = asr_python or os.environ.get("BRIDGE_ASR_PYTHON", "").strip()
    if asr_py:
        device = _pick_asr_device()
        env = os.environ.copy()
        # Isolate physical GPU so child torch sees it as cuda:0.
        if device.startswith("cuda:"):
            env["CUDA_VISIBLE_DEVICES"] = device.split(":", 1)[1]
            map_device = "cuda:0"
        else:
            map_device = device
        script = r"""
import sys
import soundfile as sf
import torch
from qwen_asr import Qwen3ASRModel
path, device = sys.argv[1], sys.argv[2]
asr = Qwen3ASRModel.from_pretrained(
    "/workspace/models/Qwen3-ASR-1.7B",
    device_map=device,
    dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
    attn_implementation="sdpa",
)
wav, sr = sf.read(path)
print(asr.transcribe(audio=(wav, sr), language="Chinese")[0].text)
"""
        print(f"ASR via {asr_py} physical={device} map={map_device}", flush=True)
        out = subprocess.check_output(
            [asr_py, "-c", script, str(path), map_device],
            text=True,
            env=env,
        )
        return out.strip().splitlines()[-1]

    import soundfile as sf
    import torch

    try:
        from qwen_asr import Qwen3ASRModel
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "qwen_asr missing; set BRIDGE_ASR_PYTHON=/home/wtk/AURA/AURA/.venv/bin/python"
        ) from exc

    device = _pick_asr_device()
    print(f"ASR device={device}", flush=True)
    asr = Qwen3ASRModel.from_pretrained(
        "/workspace/models/Qwen3-ASR-1.7B",
        device_map=device,
        dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        attn_implementation="sdpa",
    )
    wav, sr = sf.read(str(path))
    return asr.transcribe(audio=(wav, sr), language="Chinese")[0].text


def _latest_dump_with_merged(dump_root: Path, after_ts: float) -> Path | None:
    best = None
    best_mtime = after_ts - 1
    for p in dump_root.glob("turn_*/merged.wav"):
        m = p.stat().st_mtime
        if m >= after_ts - 2 and m >= best_mtime:
            best = p.parent
            best_mtime = m
    return best


async def _run_bridge_turn(*, bridge_ws: str, wav_path: Path, timeout_s: float) -> dict:
    frames = _make_frames_b64(2)
    wav_b64 = base64.b64encode(wav_path.read_bytes()).decode()
    video_url = "data:video/jpeg;base64," + ",".join(frames)

    asr_text = ""
    reply_text = ""
    audio_n = 0
    audio_b64_len = 0
    errors: list[str] = []
    events: list[str] = []

    t0 = time.time()
    async with websockets.connect(bridge_ws, max_size=32 * 1024 * 1024) as ws:
        await ws.send(
            json.dumps(
                {
                    "type": "video",
                    "session_id": "verify_io",
                    "chunk_id": "v1",
                    "wall_time_ms": int(time.time() * 1000),
                    "data": {"video_url": video_url},
                }
            )
        )
        await asyncio.sleep(0.3)
        await ws.send(
            json.dumps(
                {
                    "type": "audio",
                    "session_id": "verify_io",
                    "chunk_id": "a1",
                    "wall_time_ms": int(time.time() * 1000),
                    "data": {"audio_b64": wav_b64},
                }
            )
        )

        got_spoken = False
        while time.time() - t0 < timeout_s:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=60)
            except asyncio.TimeoutError:
                errors.append("timeout waiting for events")
                break
            msg = json.loads(raw)
            et = str(msg.get("type") or "")
            data = msg.get("data") if isinstance(msg.get("data"), dict) else {}
            events.append(et)
            if et == "asr_text":
                asr_text = str(data.get("text") or "")
            elif et == "text":
                reply_text = str(data.get("text") or "")
                if reply_text and "<|silent|>" not in reply_text:
                    got_spoken = True
            elif et == "audio":
                audio_n += 1
                audio_b64_len += len(data.get("audio") or "")
            elif et == "turn_done":
                if got_spoken and audio_n > 0:
                    break
            elif et == "error":
                errors.append(str(data.get("message") or msg))
                break

    return {
        "elapsed_s": round(time.time() - t0, 2),
        "asr_text": asr_text,
        "reply_text": reply_text,
        "audio_n": audio_n,
        "audio_b64_len": audio_b64_len,
        "errors": errors,
        "events": events,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bridge", default="http://127.0.0.1:9999")
    ap.add_argument("--wav", type=Path, default=DEFAULT_WAV)
    ap.add_argument("--dump-dir", type=Path, default=DEFAULT_DUMP)
    ap.add_argument("--timeout", type=float, default=150)
    ap.add_argument(
        "--with-output-asr",
        action="store_true",
        help="Also load a second ASR model to transcribe TTS output (off by default).",
    )
    ap.add_argument(
        "--asr-python",
        default="/home/wtk/AURA/AURA/.venv/bin/python",
        help="Python with qwen_asr (default: AURA native venv)",
    )
    ap.add_argument("--out", type=Path, default=Path("/tmp/aura_v2_native_demo/verify_bridge.json"))
    args = ap.parse_args()

    if not args.wav.is_file():
        raise SystemExit(f"wav missing: {args.wav}")

    bridge_ws = args.bridge.replace("https://", "wss://").replace("http://", "ws://").rstrip("/")
    bridge_ws = f"{bridge_ws}/ws"

    print(f"bridge={bridge_ws}", flush=True)
    print(f"wav={args.wav}", flush=True)
    t_before = time.time()
    result = asyncio.run(
        _run_bridge_turn(bridge_ws=bridge_ws, wav_path=args.wav, timeout_s=args.timeout)
    )
    print(json.dumps({k: result[k] for k in ("elapsed_s", "asr_text", "reply_text", "audio_n", "errors")}, ensure_ascii=False, indent=2), flush=True)

    dump_turn = _latest_dump_with_merged(args.dump_dir, t_before)
    merged = dump_turn / "merged.wav" if dump_turn else None
    text_file = (dump_turn / "text.txt").read_text(encoding="utf-8").strip() if dump_turn and (dump_turn / "text.txt").exists() else result["reply_text"]
    transcript_file = (
        (dump_turn / "transcript.txt").read_text(encoding="utf-8").strip()
        if dump_turn and (dump_turn / "transcript.txt").exists()
        else result["asr_text"]
    )

    checks = {
        "has_asr_text": bool(result["asr_text"].strip()),
        "has_reply_text": bool(result["reply_text"].strip()) and "<|silent|>" not in result["reply_text"],
        "has_audio_chunks": result["audio_n"] > 0,
        "has_merged_dump": bool(merged and merged.is_file()),
        "no_errors": not result["errors"],
    }

    dump_asr = ""
    text_vs_dump_asr = False
    if merged and merged.is_file() and args.with_output_asr:
        print(f"ASR dump {merged} ...", flush=True)
        dump_asr = _asr_wav(merged, asr_python=args.asr_python)
        text_vs_dump_asr = _soft_match(text_file, dump_asr)
        print(f"text={text_file!r}", flush=True)
        print(f"dump_asr={dump_asr!r}", flush=True)
        checks["text_vs_dump_asr"] = text_vs_dump_asr
    elif not args.with_output_asr:
        checks["text_vs_dump_asr"] = None
    else:
        checks["text_vs_dump_asr"] = False

    # duration sanity
    dur_ok = False
    if merged and merged.is_file():
        with wave.open(str(merged), "rb") as wf:
            dur = wf.getnframes() / float(wf.getframerate() or 1)
        # Chinese ~3-4 chars/sec rough; allow wide band
        nchars = max(len(_soft_norm(text_file)), 1)
        dur_ok = 0.4 <= dur <= max(12.0, nchars * 0.8)
        checks["dump_duration_ok"] = dur_ok
        checks["dump_duration_s"] = round(dur, 2)

    report = {
        "checks": checks,
        "bridge_result": result,
        "dump_dir": str(dump_turn) if dump_turn else None,
        "dump_text": text_file,
        "dump_transcript": transcript_file,
        "dump_asr": dump_asr,
        "pass": all(v for k, v in checks.items() if k != "dump_duration_s" and v is not None),
    }
    # dump_duration_ok required if we have dump
    if "dump_duration_ok" in checks:
        report["pass"] = report["pass"] and checks["dump_duration_ok"]
    if checks.get("text_vs_dump_asr") is False:
        report["pass"] = False

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.out}", flush=True)
    print("PASS" if report["pass"] else "FAIL", flush=True)
    print(json.dumps(checks, ensure_ascii=False, indent=2), flush=True)
    sys.exit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
