# AURA Streaming Video API

vLLM-Omni exposes the same WebSocket endpoint as Qwen-Omni streaming video, but when the server is started with the `aura_omni` deploy profile it uses `AuraStreamingVideoHandler`:

- **ASR → AURA → TTS → Code2Wav** four-stage pipeline
- **Automatic turn trigger** after `auto_trigger_min_frames` buffered frames (default `2`)
- **SessionHistory** across turns via `asr2aura_session`
- **`modalities: ["text", "audio"]`** for TTS output via `response.audio.delta` / `response.audio.done`
- **Frame-only auto trigger** — `frames >= auto_trigger_min_frames` and **`not is_turn_locked`**
- **Early turn release** — after assistant text (`response.text.done`), SessionHistory updates and the next frame may trigger while TTS audio still streams
- **`video.query` is ignored** — no manual trigger, no interrupt

See also: [video_stream_api.md](video_stream_api.md) for shared protocol fields.

## Quick Start

### Start the Server

```bash
vllm serve aurateam/AURA \
    --deploy-config vllm_omni/deploy/aura_omni.yaml \
    --omni \
    --port 8000 \
    --trust-remote-code
```

### Run the Example Client

```bash
python examples/online_serving/aura_omni/streaming_video_client.py \
    --url ws://localhost:8000/v1/video/chat/stream \
    --synthetic-frames 8
```

With optional microphone audio (PCM16 16 kHz mono):

```bash
python examples/online_serving/aura_omni/streaming_video_client.py \
    --audio /path/to/audio.pcm \
    --synthetic-frames 8
```

## AURA-Specific `session.config` Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `auto_trigger` | bool | `true` | Auto-start a turn when enough frames arrive. |
| `auto_trigger_min_frames` | int | `2` | Minimum buffered frames before auto trigger. |
| `max_frames_per_round` | int | `16` | Max frames packed into each `video_tuple`. |
| `modalities` | list[string] | `["text", "audio"]` | Request text and/or TTS audio deltas. Use `["text"]` for text-only. |
| `cross_turn_penalty` | float | `0.0` | Cross-turn repetition penalty (0=disabled; 2.0–3.0 recommended). |
| `cross_turn_lookback` | int | `2` | Recent assistant responses in the penalty window. |
| `cross_turn_ngram_sizes` | list[int] | `[3, 4, 5]` | N-gram sizes for `bad_words` hard blocking. |
| `pruning_enabled` | bool | `true` | Enable SessionHistory sliding-window pruning. |
| `max_rounds` | int | `45` | Sliding-window round limit before pruning. |
| `num_rounds_keep` | int | `30` | Rounds kept in the sliding window after pruning. |
| `aura_system_prompt` | string | AURA default | Override the AURA system prompt. |
| `video_fps` | float | `2.0` | FPS metadata attached to each `video_tuple`. |

All standard fields from [video_stream_api.md](video_stream_api.md) (`max_frames`, EVS, `sampling_params_list`, etc.) still apply.

## Trigger Semantics

| Event | Behavior |
|-------|----------|
| `video.frame` with `frames >= auto_trigger_min_frames` and `not is_turn_locked` | Start a turn (ASR fills transcript from buffered audio) |
| `video.frame` while `is_turn_locked` | Frame buffered only (ASR→AURA text in flight) |
| `video.frame` during TTS tail only (`is_generating` but `not is_turn_locked`) | **May trigger** the next turn |
| `audio.chunk` | Appended to session buffer; snapshot at turn start (send full utterance before trigger for push-to-talk) |
| `video.query` | **Ignored** |

## Audio Output

When `modalities` includes `"audio"`, the server emits:

| Event | Payload |
|-------|---------|
| `response.audio.delta` | `data` (base64 WAV chunk), `format: "wav"` |
| `response.audio.done` | (no payload) |

Set `VLLM_VIDEO_ASYNC_CHUNK=on` for incremental audio deltas during generation (same as Qwen-Omni streaming).

The example client saves concatenated PCM to `--output-wav` (default `aura_stream_output.wav`). Pass `--text-only` to request text-only modalities.

## Handler Selection

`create_streaming_video_handler()` picks the handler from the deploy profile
``pipeline`` name (``engine_client.pipeline_name``):

| Deploy profile | `pipeline` | Handler |
|----------------|------------|---------|
| `aura_omni.yaml` | `aura_omni` | `AuraStreamingVideoHandler` |
| `qwen3_omni.yaml` (default omni video) | *(other)* | `QwenOmniStreamingVideoHandler` |
