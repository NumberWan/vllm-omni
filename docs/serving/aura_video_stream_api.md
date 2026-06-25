# AURA Streaming Video API

vLLM-Omni exposes the same WebSocket endpoint as Qwen-Omni streaming video, but when the server is started with the `aura_omni_streaming` deploy profile it uses `AuraStreamingVideoHandler`:

- **ASR → AURA → TTS → Code2Wav** four-stage pipeline
- **Automatic turn trigger** after `auto_trigger_min_frames` buffered frames (default `2`)
- **SessionHistory** across turns via `asr2aura_session` (selected by deploy `pipeline: aura_omni_streaming`)
- **`modalities: ["text", "audio"]`** for TTS output via `response.audio.delta` / `response.audio.done`
- **Frame-only auto trigger** — per-turn `turn_frame_arrays` count `>= auto_trigger_min_frames` and **`not is_turn_locked`** (not cumulative `frame_buffer`)
- **Early turn release** — after assistant text (`response.text.done`), SessionHistory updates and the next frame may trigger while TTS audio still streams
- **`video.query` is ignored** — no manual trigger, no interrupt

See also: [video_stream_api.md](video_stream_api.md) for shared protocol fields.

## Required: `pipeline: aura_omni_streaming` (multi-turn WebSocket)

The WebSocket handler always registers an `aura_session_id` and updates `SessionHistory` in-process. **Cross-turn prompts only work when stage-1 is wired to `asr2aura_session`**, which the registry pipeline `aura_omni_streaming` selects:

```yaml
# vllm_omni/deploy/aura_omni.yaml
pipeline: aura_omni_streaming
```

| Deploy `pipeline` | Stage-1 processor | WebSocket multi-turn |
|-------------------|-------------------|----------------------|
| `aura_omni_streaming` | `asr2aura_session` | SessionHistory across turns |
| `aura_omni` | `asr2aura` | Single-turn only — `aura_session_id` ignored for history |

If you use `pipeline: aura_omni` with the AURA streaming handler, stage-1 logs a **one-time WARNING** when it sees `aura_session_id`.

Single-turn `/chat/completions` or Gradio can use `pipeline: aura_omni`.

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
| `max_context_qas` | int | `10` | Max Q&A blocks in compressed context history after prune. |
| `aura_system_prompt` | string | AURA default | Override the AURA system prompt. |
| `video_fps` | float | `2.0` | FPS metadata attached to each `video_tuple`. |
| `stream_text_deltas` | bool | `false` | When `false`, the server buffers assistant text and only sends `response.text.done` (no per-token `response.text.delta`). Set `true` for incremental text streaming. |

All standard fields from [video_stream_api.md](video_stream_api.md) (`max_frames`, EVS, `sampling_params_list`, etc.) still apply.

## Text Output

By default AURA does **not** stream `response.text.delta` to clients. Assistant tokens are accumulated server-side; the client receives a single `response.text.done` with the full reply (sent early when TTS audio starts if `VLLM_VIDEO_ASYNC_CHUNK=on`, so the next turn can begin while audio still streams).

Set `stream_text_deltas: true` in `session.config` if you need incremental text events (e.g. for a live caption UI).

## Trigger Semantics

| Event | Behavior |
|-------|----------|
| `video.frame` with per-turn frames `>= auto_trigger_min_frames` and `not is_turn_locked` | Start a turn (ASR fills transcript from buffered audio) |
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

### Timed replay demo (`streaming_video_demo.py`)

`--audio-schedule SEC:PATH` injects WAV at stream wall-clock seconds (not clip length).

Pruning defaults (`max_rounds=45`) are **not** reached on short clips (~43 turns for `aura_test.mp4` @ 2 fps). To exercise prune:

```bash
python examples/online_serving/aura_omni/streaming_video_demo.py \
  ... --max-rounds 30 --num-rounds-keep 20
```

### Debug: log assembled AURA prompts

Set before `vllm serve` (logs appear on **Stage-1 worker**, not always the API process):

```bash
export VLLM_AURA_LOG_TURN_PROMPT=1
```

Each turn prints `AURA turn prompt request_id=...` with `prompt_text` and video metadata (no pixels).

## Handler Selection

`create_streaming_video_handler()` reads the deploy YAML ``pipeline`` from ``engine_client.config_path``:

| Deploy profile | `pipeline` | Handler |
|----------------|------------|---------|
| `aura_omni.yaml` | `aura_omni_streaming` | `AuraStreamingVideoHandler` |
| single-turn AURA | `aura_omni` | `AuraStreamingVideoHandler` |
| `qwen3_omni.yaml` (default omni video) | *(other)* | `QwenOmniStreamingVideoHandler` |
