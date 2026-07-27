# AURA MiniCPM-style Browser Demo

Everything required by this demo is contained in this directory:

- `server.py`: FastAPI static host and same-origin WebSocket proxy
- `app/`: browser HTML, CSS, JavaScript, microphone capture, and audio playback
- `run_demo.sh`: launcher

The existing AURA server remains unchanged. The demo connects to:

```text
ws://127.0.0.1:8666/v1/video/chat/stream
```

## Run

**1-GPU demo stack (recommended here):** one command starts AURA on `:8666`
(if needed) then the browser UI on `:7862`:

```bash
bash examples/online_serving/aura_omni/minicpm_style_web_demo/run_1gpu_demo_stack.sh
```

Stop the matching 1-GPU AURA process (PID file under `/tmp/aura_omni_1gpu_demo`,
not the default `/tmp/aura_omni_serve`):

```bash
bash examples/online_serving/aura_omni/minicpm_style_web_demo/stop_1gpu_demo_stack.sh
```

Defaults match the single-GPU pack (`CUDA_VISIBLE_DEVICES=3`,
`aura_omni_1gpu_demo.yaml`, Vivian, empty TTS instruct). If `:8666` / `:7862` are busy, the
script picks the next free ports. It forces a writable `HF_HOME` under
`~/.cache/huggingface` so a parent-shell `HF_HOME=/workspace/model` does
not hit PermissionError. Override with env vars if needed
(`MODEL`, `CUDA_VISIBLE_DEVICES`, `TTS_SPEAKER`, `AURA_PORT`, `DEMO_PORT`,
`AURA_HF_HOME`, `VLLM_AURA_SENTENCE_TTS`, …).

By default the 1-GPU demo sets `VLLM_AURA_SENTENCE_TTS=0` so AURA finishes the
full reply before TTS (CustomVoice cannot yet chain mid-gen sentence payloads
on one request). Set `VLLM_AURA_SENTENCE_TTS=1` for lower TTFP (requires AURA
restart).

Open `http://127.0.0.1:7862/` after startup (the launcher prints a clickable
`>>> Demo ready — click to open: ...` line). In Cursor Remote SSH,
**Ctrl/Cmd-click** that link, or use the **Ports** panel → **Open in Browser**.
Before printing that ready link, the launcher sends one request using a
generated neutral frame and the speech WAV already stored at
`tests/assets/glm_tts/jiayan_zh.wav`, then drains its spoken reply to warm all
four stages. The browser demo therefore does **not** require OmniInteract.
Use `--warmup-audio /path/to/speech.wav` to replace the bundled warmup or
`--skip-warmup` to disable it.

Spoken replies are also dumped under `/tmp/aura_web_demo_tts/turn_*/` as
exact AURA `response.audio.delta` WAV chunks plus `merged.wav` / `text.txt`
and `request_id.txt`. Audio is grouped by the WS event `request_id`, even when
the next silent turn starts while spoken TTS is still draining. Only spoken
turns are saved; `<|silent|>` is skipped.

The page requests camera and microphone access on load. While a session is
active the camera continuously sends `video.frame` at `video_fps` with
`auto_trigger` enabled, so AURA can emit proactive replies or `<|silent|>`.
Hold **Hold to talk** while speaking and release to commit the buffered
utterance as `audio.chunk` / `audio.done` (plus one fresh present-moment
frame).

Run `run_demo.sh --help` for model, backend, port, FPS, TTS, and public
WebSocket options.

Automated proxy check (AURA server + demo already running):

```bash
python examples/online_serving/aura_omni/minicpm_style_web_demo/verify_e2e.py \
  --model /workspace/models/AURA
```

This E2E checker still uses an OmniInteract sample by default; it is optional
and is not part of normal interactive demo startup.

## Scope

This preserves the MiniCPM-style Start session / Camera interface, but
uses AURA's streaming protocol. It intentionally does not implement:

- `/v1/realtime?duplex=1` or model-owned listen/speak tokens
- server-side barge-in or `response.cancel`
- `playback.ack`

Releasing push-to-talk commits a voice turn; it does not cancel an in-flight
generation.

Browser microphone and camera access generally requires `localhost` or HTTPS.
When TLS is terminated by a proxy that does not forward WebSocket upgrades,
pass `--public-stream-url wss://.../v1/video/chat/stream`.
