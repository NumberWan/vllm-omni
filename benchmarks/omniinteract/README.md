# OmniInteract Benchmark

> **Setup & run guide:** [SETUP_AND_RUN.md](./SETUP_AND_RUN.md)  
> Native AURA alignment notes: `/public/wtk/docs/aura/vllm_omni_benchmark_alignment_backlog.md`

This benchmark evaluates audio-visual interaction on the
[OmniInteract](https://github.com/Lucky-Lance/OmniInteract) dataset with
`vllm bench serve --omni`.

OmniInteract has three subsets:

- `1q1a`: Chinese daily-life QA, including realtime, proactive, and nested slots.
- `1q1a_math`: English math-reasoning QA.
- `1qna`: long-horizon monitoring where one spoken instruction leads to many timed answers.

## Dataset Layout

Download and extract `lucky-lance/OmniInteract` so the benchmark root contains:

```text
data/
├── 1q1a/
│   ├── videos/
│   ├── subvideos/
│   ├── audios/
│   ├── annotations/
│   └── video_json_map.json
├── 1q1a_math/
│   ├── videos/
│   ├── subvideos/
│   ├── audios/
│   ├── annotations/
│   └── video_json_map.json
└── 1qna/
    ├── videos_bench/
    └── annotations/
```

Pass the extracted `data/` parent or `data/` itself with `--dataset-path`, or use
`--omniinteract-root` if your run script exposes it.

## Input Modes

### `aura_streaming`

Use this mode for AURA's WebSocket endpoint:

- Backend: `openai-video-stream`
- Endpoint: `/v1/video/chat/stream`
- Video input:
  - `1q1a` / `1q1a_math`: streams sampled frames from `videos/*.mp4`.
  - `1qna`: streams sampled frames from `videos_bench/**/*.mp4`.
- Audio input:
  - `1q1a` / `1q1a_math`: sends `audios/{video}_{qa_idx}.wav` at each annotation `question_time`.
  - `1qna`: extracts the MP4 audio track with `ffmpeg` and sends it in sync with video frames, matching the original online benchmark protocol.
- Accuracy metrics are computed from timestamped response chunks and annotation slots.

The recommended default is `--omniinteract-streaming-max-frames 0` (full video
extract) with `send_fps=2`, matching native AURA OmniInteract bench. Use
`STREAMING_MAX_FRAMES=16` only for a legacy quick smoke (see `run_streaming_bench.sh`
comments). Server `session.config.max_frames` is still capped at 256 (rolling buffer).

### `video`

Use this mode for generic OpenAI-compatible chat/video backends. The benchmark
flattens annotations into independent requests and uses `subvideos/` for
`1q1a` / `1q1a_math`. For `1qna`, each answer step uses the full
`videos_bench/**/*.mp4` plus a text prompt containing the instruction and timing
metadata.

### `aura`

Use this mode for non-streaming AURA chat-style requests. It sends the per-QA
spoken question as `audio_url` and the per-QA `subvideos/` clip as `video_url`.
This mode is intended for `1q1a` / `1q1a_math`, where per-QA WAV files exist.
`1qna` is best evaluated with `aura_streaming`, because its instruction is stored
inside the full video audio track rather than as per-answer WAV files.

## Metrics

The runner reports standard serving metrics and optional OmniInteract accuracy
metrics when `--omniinteract-eval` is set.

Performance metrics:

- `ttft`: time to first text token.
- `tpot`: time per output text token.
- `itl`: inter-token latency.
- `e2el`: end-to-end request latency.
- `audio_ttfp`: time to first audio packet.
- `audio_rtf`: audio real-time factor.
- `ttfc`, `tpoc`, `icl`: internal streaming stage latency metrics when stage metrics are available.

OmniInteract accuracy metrics:

- `IA_QTF1`: interaction-aware quality-timeliness F1. It aggregates slot-level
  true-positive credit, false positives, and false negatives. Streaming mode uses
  response timestamps against annotation slots.
- `IDS.NOR`: no-output rate for interrupted slots.
- `IDS.PAQ`: partial-answer quality for interrupted slots with output.
- `IDS.CSM_SR`: conditional spillover rate after an interrupted slot's end.
- `IDS.CSM_AS_seconds`: average spillover seconds for interrupted slots with output.
- `NCCS`: nested chain completion score, computed over nested inner/outer pairs.
- `inner_IA_QTF1` / `outer_IA_QTF1`: nested local slot quality by role.

The built-in benchmark scorer uses local soft string matching for content quality.
For paper-exact semantic scoring, run the official OmniInteract LLM-judge pipeline
on timestamped model outputs.

## Examples

### Streaming AURA, Full OmniInteract

Start the AURA server first, then run:

```bash
OMNIINTERACT_EVAL=1 \
MODEL=/data/models/AURA \
DATASET_PATH=/data/models/datasets/OmniInteract \
HOST=127.0.0.1 \
PORT=8666 \
NUM_PROMPTS=250 \
MAX_CONCURRENCY=1 \
STREAMING_MAX_FRAMES=16 \
bash /data/yrr/rein_test/omniinteract_bench.sh
```

Equivalent direct command:

```bash
vllm bench serve --omni \
  --host 127.0.0.1 \
  --port 8666 \
  --trust-remote-code \
  --backend openai-video-stream \
  --endpoint /v1/video/chat/stream \
  --model /data/models/AURA \
  --dataset-name omniinteract \
  --dataset-path /data/models/datasets/OmniInteract \
  --no-oversample \
  --num-prompts 16 \
  --max-concurrency 1 \
  --omniinteract-subsets 1q1a,1q1a_math,1qna \
  --omniinteract-input-mode aura_streaming \
  --omniinteract-streaming-sample-fps 2 \
  --omniinteract-streaming-send-fps 2 \
  --omniinteract-streaming-max-frames 16 \
  --omniinteract-aura-tts-task-type CustomVoice \
  --omniinteract-aura-tts-language English \
  --omniinteract-aura-tts-speaker Vivian \
  --omniinteract-eval \
  --percentile-metrics ttft,tpot,itl,e2el,audio_ttfp,audio_rtf,audio_duration,ttfc,tpoc,icl \
  --save-result \
  --result-dir ./omniinteract_bench \
  --result-filename omniinteract_streaming.json
```

### Streaming Smoke Test

```bash
vllm bench serve --omni \
  --host 127.0.0.1 \
  --port 8666 \
  --backend openai-video-stream \
  --endpoint /v1/video/chat/stream \
  --model /data/models/AURA \
  --dataset-name omniinteract \
  --dataset-path /data/models/datasets/OmniInteract \
  --num-prompts 3 \
  --no-oversample \
  --omniinteract-subsets 1qna \
  --omniinteract-input-mode aura_streaming \
  --omniinteract-streaming-sample-fps 2 \
  --omniinteract-streaming-send-fps 2 \
  --omniinteract-streaming-max-frames 16 \
  --omniinteract-aura-tts-task-type CustomVoice \
  --omniinteract-aura-tts-language English \
  --omniinteract-aura-tts-speaker Vivian 
```

### Non-Streaming AURA on 1Q1A

```bash
vllm bench serve --omni \
  --host 127.0.0.1 \
  --port 8666 \
  --backend openai-chat-omni \
  --endpoint /v1/chat/completions \
  --model /data/models/AURA \
  --dataset-name omniinteract \
  --dataset-path /data/models/datasets/OmniInteract \
  --num-prompts 32 \
  --omniinteract-subsets 1q1a,1q1a_math \
  --omniinteract-input-mode aura \
  --omniinteract-aura-tts-task-type CustomVoice \
  --omniinteract-aura-tts-language English \
  --omniinteract-aura-tts-speaker Vivian \
  --omniinteract-eval
```

### Generic Non-Streaming Video Backend

```bash
vllm bench serve --omni \
  --host 127.0.0.1 \
  --port 8666 \
  --backend openai-chat-omni \
  --endpoint /v1/chat/completions \
  --model /data/models/AURA \
  --dataset-name omniinteract \
  --dataset-path /data/models/datasets/OmniInteract \
  --num-prompts 32 \
  --omniinteract-subsets 1q1a,1qna \
  --omniinteract-input-mode video \
  --omniinteract-eval
```
