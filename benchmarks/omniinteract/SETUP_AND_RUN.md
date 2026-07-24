# AURA OmniInteract Streaming Benchmark

This benchmark validates the production AURA WebSocket path. It streams each
full video at 2 FPS and injects annotation audio at the recorded question time.
Legacy single-request `video` and `aura` modes are intentionally not supported.

## Prerequisites

1. Install vLLM-Omni and matching FlashInfer packages.
2. Prepare the OmniInteract dataset locally or allow Hugging Face download.
3. For `1q1a` and `1q1a_math`, provide annotation WAV files at
   `audios/{video_id}_{qa_index}.wav`.
4. For Base voice TTS, provide a reference WAV and transcript.

See the [AURA serving guide](../../examples/online_serving/aura_omni/README.md)
for server requirements.

## Start AURA

```bash
CUDA_VISIBLE_DEVICES=0,1 bash benchmarks/omniinteract/run_aura_server.sh
```

The script delegates to the production launcher and waits for the health
endpoint. Use `MODEL`, `DEPLOY`, `PORT`, and `CUDA_VISIBLE_DEVICES` to override
the defaults.

## Run the native-aligned smoke benchmark

Always warm the same server process with video `0001`. The scored run starts at
`0002`; `0001` must not be included in accuracy or latency statistics.

```bash
DATASET_PATH=/path/to/OmniInteract \
OMNIINTERACT_AURA_TTS_REF_AUDIO=/path/to/reference.wav \
OMNIINTERACT_AURA_TTS_REF_TEXT='Reference transcript.' \
NATIVE_ALIGNED=1 \
bash benchmarks/omniinteract/run_streaming_bench.sh
```

`NATIVE_ALIGNED=1` runs:

- warmup: `0001` in a separate result JSON
- scored videos: `0002,0003,0004`
- sample/send FPS: `2`
- full videos (`max_frames=0`)
- native AURA system prompt

Results are written to `./omniinteract_bench/`.

## Run c=1 or c=8

Specify video IDs explicitly to keep runs reproducible:

```bash
# c=1
OMNIINTERACT_VIDEO_IDS=0002 \
MAX_CONCURRENCY=1 NUM_PROMPTS=1 \
DATASET_PATH=/path/to/OmniInteract \
OMNIINTERACT_AURA_TTS_TASK_TYPE=CustomVoice \
OMNIINTERACT_AURA_TTS_SPEAKER=Vivian \
bash benchmarks/omniinteract/run_streaming_bench.sh

# c=8
OMNIINTERACT_VIDEO_IDS=0002,0003,0004,0005,0006,0007,0008,0009 \
MAX_CONCURRENCY=8 NUM_PROMPTS=8 \
DATASET_PATH=/path/to/OmniInteract \
OMNIINTERACT_AURA_TTS_TASK_TYPE=CustomVoice \
OMNIINTERACT_AURA_TTS_SPEAKER=Vivian \
bash benchmarks/omniinteract/run_streaming_bench.sh
```

The script runs `0001` as a separate warmup request against the same server
before both examples. Set `OMNIINTERACT_SKIP_WARMUP=1` only when that exact
server process has already completed a `0001` streaming session.

## Important environment variables

| Variable | Default | Purpose |
|---|---:|---|
| `HOST` | `127.0.0.1` | Server host |
| `PORT` | `8666` | Server port |
| `DATASET_PATH` | `lucky-lance/OmniInteract` | Local dataset root or HF dataset reference |
| `MAX_CONCURRENCY` | `1` | Maximum concurrent sessions |
| `NUM_PROMPTS` | `32` | Number of videos |
| `OMNIINTERACT_VIDEO_IDS` | empty | Ordered comma-separated video IDs |
| `STREAMING_MAX_FRAMES` | `0` | Frame cap; `0` means full video |
| `STREAMING_SEND_FPS` | `2` | Wall-clock frame send rate |
| `OMNIINTERACT_AURA_TTS_TASK_TYPE` | `Base` | `Base` or `CustomVoice` |
| `OMNIINTERACT_AURA_TTS_REF_AUDIO` | empty | Required for Base voice |
| `OMNIINTERACT_AURA_TTS_REF_TEXT` | short Chinese text | Required for Base voice |

## Outputs

- `omniinteract_streaming_*.json`: aggregate metrics and per-request records
- `bench_report.md`: aggregate native-style report
- `videos/<id>/bench_report.md`: per-video report
- `videos/<id>/streaming_chunks.json`: per-turn timing and text

Set `OMNIINTERACT_SAVE_OUTPUT_WAV=1` only when a full-session output WAV is
needed for manual inspection.

## Stop and clean up

```bash
bash scripts/stop_aura_omni.sh
nvidia-smi
```

The stop script only stops the server recorded by the production launcher.
