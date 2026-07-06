# vllm-omni AURA Streaming — Server Setup & OmniInteract Benchmark

> **One-page ops guide (humans / AI agents)**  
> Branch: `aura_streaming_video_with_bench`  
> Location: `<vllm-omni>/benchmarks/omniinteract/`

---

## Quick start: 3-video smoke (native-aligned)

Videos `0002`, `0003`, `0004` — same subset as native AURA OmniInteract smoke.

**Terminal 1 — server**

```bash
cd /path/to/vllm-omni
git checkout aura_streaming_video_with_bench
source /path/to/.venv/bin/activate
pip install -e .

bash benchmarks/omniinteract/run_aura_server.sh
```

**Terminal 2 — benchmark** (after server is ready)

```bash
cd /path/to/vllm-omni
source /path/to/.venv/bin/activate

NATIVE_ALIGNED=1 OMNIINTERACT_EVAL=1 \
  bash benchmarks/omniinteract/run_streaming_bench.sh
```

Outputs:

- `omniinteract_bench/omniinteract_streaming_native_smoke3.json`
- `omniinteract_bench/bench_report.md`
- `omniinteract_bench/videos/{0002,0003,0004}/bench_report.md`

Runtime: ~8–10 minutes (3 full-length streaming sessions).

Optional export for side-by-side comparison with native AURA:

```bash
python -m vllm_omni.benchmarks.format_bench_report \
  omniinteract_bench/omniinteract_streaming_native_smoke3.json \
  --export-aura-result
```

---

## 1. What this stack does

| Component | Role |
|-----------|------|
| **AURA Server** | `vllm serve --omni` — ASR → AURA (VL) → TTS Talker → Code2Wav |
| **Benchmark client** | `vllm bench serve` over WebSocket `/v1/video/chat/stream` |
| **Artifacts** | `./omniinteract_bench/omniinteract_streaming_*.json`, `bench_report.md` |

This is **separate** from native AURA (TCP port `12346`). **Do not run both on the same GPUs.**

---

## 2. Prerequisites

### Code & branch

```bash
cd /path/to/vllm-omni
git checkout aura_streaming_video_with_bench
source /path/to/.venv/bin/activate
pip install -e .
```

After editing `vllm_omni/`, run `pip install -e .` again and **restart the server**.

### Weights & dataset (local defaults)

`vllm_omni/deploy/aura_omni.local.yaml` expects:

| Asset | Path |
|-------|------|
| AURA | `/models/AURA` |
| Qwen3-ASR | `/models/Qwen3-ASR-1.7B` |
| Qwen3-TTS CustomVoice | `/models/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-CustomVoice/snapshots/...` |
| OmniInteract | `/models/datasets/OmniInteract` |

Override via env vars (see below).

### GPUs

Default `CUDA_VISIBLE_DEVICES=0,1`:

- **GPU 0**: ASR + TTS (stages 0, 2, 3)
- **GPU 1**: AURA inference (stage 1)

Stop native AURA (`Qwen3_VL_online_streaming`, `tts_service`, etc.) before starting:

```bash
nvidia-smi
```

---

## 3. Start the server (Terminal 1)

### Recommended script

```bash
cd /path/to/vllm-omni
bash benchmarks/omniinteract/run_aura_server.sh
```

Use **tmux** / **screen** for a long-running session:

```bash
tmux new -s vllm_aura_server
bash benchmarks/omniinteract/run_aura_server.sh
# Ctrl+B D to detach
```

### Environment variables (`run_aura_server.sh`)

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL` | `/models/AURA` | `--model` and `--served-model-name` |
| `DEPLOY_CONFIG` | `vllm_omni/deploy/aura_omni.local.yaml` | Stage / GPU layout |
| `VLLM_BIN` | `/public/wtk/.venv/bin/vllm` | `vllm` executable |
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `8666` | HTTP / WebSocket port |
| `CUDA_VISIBLE_DEVICES` | `0,1` | Physical GPUs (see deploy yaml) |

### Equivalent manual command

```bash
cd /path/to/vllm-omni
export CUDA_VISIBLE_DEVICES=0,1

vllm serve /models/AURA \
  --omni \
  --deploy-config vllm_omni/deploy/aura_omni.local.yaml \
  --host 0.0.0.0 \
  --port 8666 \
  --served-model-name /models/AURA \
  --trust-remote-code
```

### Verify server is up

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8666/health   # expect 200
```

Logs should show `Application startup complete` and route `/v1/video/chat/stream`. Model load can take several minutes; the bench script waits up to 900s.

---

## 4. Run the benchmark (Terminal 2)

### Full dataset (all subsets, native-aligned streaming defaults)

```bash
OMNIINTERACT_EVAL=1 \
NUM_PROMPTS=250 \
NUM_WARMUPS=0 \
bash benchmarks/omniinteract/run_streaming_bench.sh
```

Defaults: `1q1a,1q1a_math,1qna`, native system prompt, `max_frames=0` (full extract), `send_fps=2`, EVS frame filter off. Set `NUM_PROMPTS` ≥ total video count (`--no-oversample` is always on).

### `NATIVE_ALIGNED=1` (3-video smoke only)

Pins videos `0002`, `0003`, `0004` and `NUM_PROMPTS=3`. Streaming settings are the same as the default above.

```bash
NATIVE_ALIGNED=1 OMNIINTERACT_EVAL=1 \
  bash benchmarks/omniinteract/run_streaming_bench.sh
```

```bash
STREAMING_MAX_FRAMES=16 STREAMING_SEND_FPS=0 \
  OMNIINTERACT_STREAMING_SYSTEM_PROMPT_MODE=omniinteract_qa \
  NUM_PROMPTS=3 \
  bash benchmarks/omniinteract/run_streaming_bench.sh
```

### Bench environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `127.0.0.1` | Server host |
| `PORT` | `8666` | Must match server |
| `MODEL` | `/models/AURA` | Must match `--served-model-name` |
| `DATASET_PATH` | `/models/datasets/OmniInteract` | Dataset root |
| `NATIVE_ALIGNED` | `0` | `1` = 3-video native-aligned smoke |
| `OMNIINTERACT_EVAL` | `1` | OmniInteract accuracy metrics |

---

## 5. Troubleshooting

| Symptom | Checks |
|---------|--------|
| Server OOM | `nvidia-smi`; lower `gpu_memory_utilization` in deploy yaml |
| Bench `Successful requests: 0` | Server on `:8666`; `pip install -e .` + restart; read `errors` in JSON |
| TTS `2048 vs 1024` | Ensure P0 TTS `session.config` fix is installed; restart server |
| Numbers ≠ native AURA | Use `NATIVE_ALIGNED=1`; see `docs/aura/vllm_omni_benchmark_alignment_backlog.md` |

---

## 6. Related files

```text
benchmarks/omniinteract/run_aura_server.sh
benchmarks/omniinteract/run_streaming_bench.sh
vllm_omni/deploy/aura_omni.local.yaml
vllm_omni/entrypoints/openai/serving_video_stream.py
vllm_omni/benchmarks/format_bench_report.py
benchmarks/omniinteract/README.md          # dataset layout & metrics
```
