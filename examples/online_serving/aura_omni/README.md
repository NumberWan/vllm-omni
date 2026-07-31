# Serve AURA with vLLM-Omni

AURA is served as a four-stage streaming pipeline:

```text
Qwen3-ASR -> AURA -> Qwen3-TTS Talker -> Code2Wav
```

The production profile uses two GPUs, FlashInfer, streaming session history,
silent-turn Stage-0 bypass, and sentence-level TTS handoff.

## Prerequisites

- Linux with two CUDA GPUs that can hold the four models
- Python 3.10–3.13
- A working vLLM-Omni installation
- Access to:
  - `Qwen/Qwen3-ASR-1.7B`
  - `aurateam/AURA`
  - `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`
- Matching `flashinfer-python` and `flashinfer-jit-cache` versions

From the repository root, run the AURA installer:

```bash
bash scripts/install_aura_omni.sh
```

It creates `.venv`, installs this checkout, detects the CUDA version used by
PyTorch, installs the matching `flashinfer-jit-cache`, and verifies that its
version matches `flashinfer-python`. Set `VENV_DIR=/path/to/venv` to use a
different environment.

## Start the server

### API-only quick start

Auto-loads the default deploy yaml (`vllm_omni/deploy/aura_omni.yaml`).
Stages pin visible devices `0`/`1`, so without `CUDA_VISIBLE_DEVICES` this
uses physical GPUs 0 and 1:

```bash
vllm serve aurateam/AURA --omni
```

Use a local checkpoint path instead of the Hub id when weights are already
on disk (`vllm serve /path/to/AURA --omni`). Add `--port 8666` if you need a
fixed port (vLLM defaults to `8000`). Set `CUDA_VISIBLE_DEVICES` only when
you need other cards (for example `2,3`).

### Helper script (background + checks)

From the repository root:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/start_aura_omni.sh
```

The script validates the deploy file, port, GPU count, and FlashInfer packages,
then starts the server in the background. Models are downloaded from
Hugging Face on first use. Startup can take several minutes.

Check health and logs:

```bash
curl http://127.0.0.1:8666/v1/models
tail -f /tmp/aura_omni_serve/server_*.log
```

The script never terminates an unrelated process occupying the configured port.
Use another port when necessary:

```bash
PORT=8667 CUDA_VISIBLE_DEVICES=2,3 bash scripts/start_aura_omni.sh
```

## Send a streaming request

The production endpoint is a WebSocket:

```text
ws://127.0.0.1:8666/v1/video/chat/stream
```

Run the included smoke client with generated frames:

```bash
python examples/online_serving/aura_omni/streaming_video_client.py \
  --url ws://127.0.0.1:8666/v1/video/chat/stream \
  --synthetic-frames 8
```

Use `--audio /path/to/question.pcm` for PCM16, 16 kHz, mono microphone audio.
Use `--text-only` when synthesized speech is not required.

For a single HTTP multimodal request:

```bash
python examples/online_serving/aura_omni/openai_chat_completion_client.py \
  --host 127.0.0.1 \
  --port 8666 \
  --model aurateam/AURA \
  --modalities text,audio
```

## MiniCPM-style browser demo

A browser UI modeled after the MiniCPM-o realtime demo is isolated under
`examples/online_serving/aura_omni/minicpm_style_web_demo/`. The single-GPU
demo profile can start AURA and the UI with one command:

```bash
bash examples/online_serving/aura_omni/minicpm_style_web_demo/run_1gpu_demo_stack.sh
```

It uses a bundled speech sample and a generated frame for startup warmup, so
normal browser use does not require the OmniInteract dataset. Smoke3 remains a
separate, optional regression benchmark for measured accuracy and latency.

See the demo's own
[`README.md`](minicpm_style_web_demo/README.md) for controls, options, and the
intentional non-full-duplex boundary.

## Stop the server

```bash
bash scripts/stop_aura_omni.sh
```

The stop script only signals the PID recorded by the start script and refuses
to terminate a process that is not `vllm serve`.

## Local or offline model weights

Copy the production deploy and replace its four `model:` values with local
paths:

```bash
cp vllm_omni/deploy/aura_omni_2gpu_best.yaml /tmp/aura_local.yaml
# Edit /tmp/aura_local.yaml, then:
DEPLOY=/tmp/aura_local.yaml \
MODEL=/models/AURA \
ALLOWED_LOCAL_MEDIA_PATH=/models \
CUDA_VISIBLE_DEVICES=0,1 \
bash scripts/start_aura_omni.sh
```

`devices: "0"` and `devices: "1"` in the deploy file are relative to
`CUDA_VISIBLE_DEVICES`.

## Common failures

### FlashInfer argument or ABI errors

Check the installed versions:

```bash
python - <<'PY'
import importlib.metadata as md
print("flashinfer-python:", md.version("flashinfer-python"))
print("flashinfer-jit-cache:", md.version("flashinfer-jit-cache"))
PY
```

Their base versions must match. Reinstall both from the wheel index for the
machine's CUDA version.

### Port already in use

The start script exits without killing the existing process. Set `PORT` to a
free value or stop the owner of that port yourself.

### Out of memory

Confirm that exactly two free GPUs are exposed. Do not share them with another
server. For a custom placement or memory budget, copy the deploy YAML and pass
it through `DEPLOY=`.

### Local media is rejected

Set `ALLOWED_LOCAL_MEDIA_PATH` to the directory containing request media. This
is not required by the WebSocket client because it sends frames and audio over
the connection.

## Advanced configuration

- API protocol: [`docs/serving/aura_video_stream_api.md`](../../../docs/serving/aura_video_stream_api.md)
- Performance and feature flags: [`docs/aura/AURA_OMNI_TUNABLES.md`](../../../docs/aura/AURA_OMNI_TUNABLES.md)
- OmniInteract regression benchmark: [`benchmarks/omniinteract/SETUP_AND_RUN.md`](../../../benchmarks/omniinteract/SETUP_AND_RUN.md)
