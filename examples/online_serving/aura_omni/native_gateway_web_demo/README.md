# AURA_v2 Omni with the original Native frontend

This demo runs the four-stage AURA_v2 Omni pipeline on one GPU and serves the
unmodified frontend from `AURA_demo-main` on port 9999.

```text
browser (Native /ws) -> protocol bridge -> AURA_v2 Omni (:8666)
```

## Frontend provenance

The files in `static/` are byte-for-byte copies from
`/home/wtk/AURA_demo-main/gateway/static/` as of 2026-08-20. The bridge adapts
the protocol; it does not modify the UI.

| File | SHA-256 |
|---|---|
| `index.html` | `81503f4a67f3088390f1a3b94cbe3cdbf5bc2a1170a1cd7799439b45a67a7e0a` |
| `eval.html` | `c7b33d9a6c4a9faa09b84c1697895f56a671ea224043a6466f9267e05b867afe` |
| `question_pages.html` | `79ca4b00e095b4830f77b238c370aec1681b7fa6681b90b692bc199f9067f43c` |

## Start

The script selects the first GPU using less than 2 GiB. It never kills another
user's process. Select a specific free GPU with `AURA_GPU=N`.

```bash
bash examples/online_serving/aura_omni/native_gateway_web_demo/run_1gpu_stack.sh
```

Open:

- local: `http://127.0.0.1:9999/`
- LAN: `http://<host-ip>:9999/`

Defaults:

- AURA backend: `:8666`
- Native frontend/bridge: `:9999`
- AURA model: `/workspace/models/AURA_v2`
- one concurrent session
- PTT-triggered turns (`auto_trigger=false`); camera frames provide visual context
- Qwen3-TTS 1.7B Base voice clone using bundled `clone_2.wav`
- Chinese; default style instruct asks for professional, clear, slightly faster,
  emotionally restrained delivery (override with `TTS_INSTRUCT`)
- safe tools enabled: calculator, datetime, weather, currency, DeepSeek mock,
  location, and Serper `WebSearch` (key from `SERPER_API_KEY` only)

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

Typed text, proactive video-only replies, and the Native HTTP eval flow are not
part of the realtime showcase. Disabling video-only auto-trigger avoids a race
where a silent camera turn can lock out PTT audio. Realtime camera context,
PTT, and tool bubbles are supported.
