# AURA_v2 Omni with the original Native frontend

This demo runs the four-stage AURA_v2 Omni pipeline on one GPU and serves the
Native frontend from `AURA_demo-main` on port 9999.

```text
browser (Native /ws) -> protocol bridge -> AURA_v2 Omni (:8666)
```

## Frontend provenance

`eval.html` and `question_pages.html` are byte-for-byte copies from
`/home/wtk/AURA_demo-main/gateway/static/` as of 2026-08-20.
`index.html` is that same page plus a small PTT `stopPlayback()` patch so a new
hold-to-talk mutes leftover TTS. Omni pipeline TTS is not cancelled.

| File | SHA-256 |
|---|---|
| `index.html` | `14ec901cb3deb89e9216f5ed7060dd60d789a49e064f4474ba995ef3a591902e` |
| `eval.html` | `c7b33d9a6c4a9faa09b84c1697895f56a671ea224043a6466f9267e05b867afe` |
| `question_pages.html` | `79ca4b00e095b4830f77b238c370aec1681b7fa6681b90b692bc199f9067f43c` |

## Start

The script selects the first GPU using less than 2 GiB. It never kills another
user's process. Select a specific free GPU with `AURA_GPU=N`.

```bash
bash examples/online_serving/aura_omni/native_gateway_web_demo/run_1gpu_stack.sh
```

Optional tool keys (DeepSeek / Serper): put them in
`~/.config/aura/tool_keys.env` (outside git; `chmod 600`). The start script
sources that file when present. You can also `export DEEPSEEK_API_KEY=...`
and `export SERPER_API_KEY=...` before starting.

Open:

- local: `http://127.0.0.1:9999/`
- LAN: `http://<host-ip>:9999/`

Defaults:

- AURA backend: `:8666`
- Native frontend/bridge: `:9999`
- AURA model: `/workspace/models/AURA_v2`
- one concurrent session
- `auto_trigger=true` with `auto_trigger_min_frames=2` (default; same as Native
  Gateway always running inference on each video batch). Silence depends on the
  model emitting `<|silent|>`. Set `AUTO_TRIGGER=0` for PTT-only turns.

Stop:

```bash
bash examples/online_serving/aura_omni/native_gateway_web_demo/stop_1gpu_stack.sh
```

## Verify

```bash
.venv/bin/python \
  examples/online_serving/aura_omni/native_gateway_web_demo/verify_bridge.py
```

The smoke test sends Native-format camera and PTT messages and requires ASR
text, assistant text, one merged playable WAV, and one `turn_done`.

The default deploy is `aura_omni_v2_1gpu_base.yaml`. Task type and checkpoint
must match: do not send `Base` input to a CustomVoice checkpoint (or vice
versa), because their speaker embeddings have different dimensions.

## Protocol behavior

- Native multi-frame JPEG batches become individual Omni `video.frame` events.
- Native WAV PTT audio becomes 16 kHz mono PCM `audio.chunk` + `audio.done`.
- Omni ASR/text/tool events are mapped to the original Native event names.
- Pre-tool prefix events become a Native `text` plus one merged `audio` segment
  and **do not** send `turn_done`.
- Final Omni TTS chunks are buffered into one WAV; the bridge synthesizes
  exactly one Native `turn_done` after final audio, silent text, or error.
- A new Native `audio` (PTT) barges in at the bridge: pending TTS is dropped.
  The page also stops the current `AudioBufferSource`. Server TTS may still
  finish on GPU.

Typed text and the Native HTTP eval flow are not part of the realtime
showcase. Realtime camera context, PTT, tool bubbles, and 2-frame auto-trigger
are supported.
