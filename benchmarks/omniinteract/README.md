# OmniInteract for AURA Streaming

This directory contains the production regression benchmark for AURA's
WebSocket audio-video service.

Use the scripts:

```bash
bash benchmarks/omniinteract/run_aura_server.sh
bash benchmarks/omniinteract/run_streaming_bench.sh
```

The benchmark:

- streams full videos through `/v1/video/chat/stream`
- sends frames at the native 2 FPS rate
- injects annotation WAV questions at `question_time`
- records per-turn ASR, Stage-1, TTFT, audio TTFP, and evaluation metrics
- runs `0001` as a separate same-process warmup and excludes it from scored results

Dataset subsets:

- `1q1a`
- `1q1a_math`
- `1qna`

Expected layout for `1q1a` and `1q1a_math`:

```text
data/
  1q1a/
    video_json_map.json
    videos/0001.mp4
    annotations/0001.json
    audios/0001_0.wav
```

`1qna` uses `videos_bench/` and `annotations/`; speech is read from the source
video.

The loader intentionally supports only the AURA streaming protocol. Historical
single-request `video` and `aura` modes were removed because they did not
exercise session history, frame timing, turn locking, or the production
async-chunk path.

See [SETUP_AND_RUN.md](SETUP_AND_RUN.md) for setup, c=1/c=8 commands, outputs,
and cleanup.
