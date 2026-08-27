#!/usr/bin/env python3
"""Live API and browser acceptance for Native typed text and /eval."""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
from pathlib import Path

import cv2
import httpx
import numpy as np
import websockets
from PIL import Image
from playwright.async_api import async_playwright


def _jpeg_b64(color: tuple[int, int, int]) -> str:
    image = Image.new("RGB", (640, 360), color)
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=80)
    return base64.b64encode(output.getvalue()).decode()


def _create_video(path: Path) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10,
        (640, 360),
    )
    if not writer.isOpened():
        raise RuntimeError("failed to create eval fixture video")
    try:
        for index in range(20):
            frame = np.zeros((360, 640, 3), dtype=np.uint8)
            frame[:, :] = (40 + index, 80, 120)
            cv2.putText(
                frame,
                "AURA typed eval",
                (160, 190),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            writer.write(frame)
    finally:
        writer.release()


async def _verify_typed_ws(ws_url: str) -> dict:
    events = []
    reply = ""
    async with websockets.connect(f"{ws_url}?tts=0", max_size=32 * 1024 * 1024) as ws:
        await ws.send(
            json.dumps(
                {
                    "type": "video",
                    "session_id": "typed-acceptance",
                    "data": {
                        "video_url": (
                            "data:video/jpeg;base64,"
                            + ",".join(
                                [
                                    _jpeg_b64((44, 36, 28)),
                                    _jpeg_b64((45, 36, 28)),
                                ]
                            )
                        ),
                        "text": "請簡短確認你收到這條文字訊息。",
                    },
                }
            )
        )
        while True:
            message = json.loads(await asyncio.wait_for(ws.recv(), timeout=180))
            message_type = str(message.get("type") or "")
            events.append(message_type)
            data = message.get("data") if isinstance(message.get("data"), dict) else {}
            if message_type == "error":
                raise RuntimeError(str(data.get("message") or message))
            if message_type == "text":
                reply += str(data.get("text") or "")
            if message_type == "turn_done":
                break
    if not reply.strip():
        raise RuntimeError("typed WebSocket turn returned no model text")
    return {"events": events, "reply": reply}


async def _verify_browser(base_url: str, video_path: Path, screenshot: Path) -> dict:
    console_errors: list[str] = []
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        page = await browser.new_page()
        page.on(
            "console",
            lambda message: (
                console_errors.append(message.text) if message.type == "error" else None
            ),
        )
        await page.goto(f"{base_url}/eval", wait_until="networkidle")
        await page.set_input_files("#fileInput", str(video_path))
        await page.fill(
            "#queriesInput",
            json.dumps(
                [{"timestamp": 0.5, "text": "請簡短描述畫面。"}],
                ensure_ascii=False,
            ),
        )
        await page.fill("#fps", "1")
        await page.fill("#nFrames", "2")
        await page.click("#runBtn")
        await page.wait_for_function(
            "() => document.querySelector('#status')?.classList.contains('done')",
            timeout=300_000,
        )
        status = await page.locator("#status").inner_text()
        turns = await page.locator("#turns").inner_text()
        await page.screenshot(path=str(screenshot), full_page=True)
        await browser.close()
    if "評測完成" not in status and "评测完成" not in status:
        raise RuntimeError(f"browser eval did not complete: {status}")
    if "請簡短描述畫面" not in turns and "请简短描述画面" not in turns:
        raise RuntimeError(f"browser eval did not render the query: {turns}")
    if not turns.strip() or "无模型输出" in turns or "無模型輸出" in turns:
        raise RuntimeError(f"browser eval rendered no model output: {turns}")
    if console_errors:
        raise RuntimeError(f"browser console errors: {console_errors}")
    return {"status": status, "turns": turns, "console_errors": console_errors}


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge", default="http://127.0.0.1:19999")
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/aura_typed_eval_acceptance"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fixture = args.out_dir / "fixture.mp4"
    screenshot = args.out_dir / "eval.png"
    _create_video(fixture)
    ws_url = args.bridge.replace("https://", "wss://").replace("http://", "ws://") + "/ws"
    typed = await _verify_typed_ws(ws_url)
    async with httpx.AsyncClient(timeout=30) as client:
        invalid = await client.post(
            f"{args.bridge}/eval",
            files={"file": ("fixture.mp4", fixture.read_bytes(), "video/mp4")},
            data={"queries": "not-json"},
        )
    if invalid.status_code != 400:
        raise RuntimeError(f"invalid eval request returned {invalid.status_code}")
    browser = await _verify_browser(args.bridge, fixture, screenshot)
    report = {
        "pass": True,
        "typed": typed,
        "invalid_eval_status": invalid.status_code,
        "browser": browser,
        "fixture": str(fixture),
        "screenshot": str(screenshot),
    }
    (args.out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
