#!/usr/bin/env bash
# One command: 2-GPU AURA (aligned 2gpu_best) + two MiniCPM-style demos
# on the same backend (for multi-client / concurrent session checks).
#
#   bash examples/online_serving/aura_omni/minicpm_style_web_demo/run_2gpu_dual_demo_stack.sh
#
# Defaults: CUDA 0,1 · AURA :8666 · demos :7862 + :7863
# Env overrides: MODEL, CUDA_VISIBLE_DEVICES, AURA_PORT, DEMO_PORT, DEMO_PORT_B,
#   TTS_SPEAKER, VENV_DIR, DEMO_PYTHON_BIN, ASR_MODEL, TTS_MODEL, LOG_DIR.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

export MODEL="${MODEL:-/workspace/models/AURA}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export LOG_DIR="${LOG_DIR:-/tmp/aura_omni_2gpu_dual_demo}"
export DEPLOY_SRC="${DEPLOY_SRC:-$REPO_ROOT/vllm_omni/deploy/aura_omni_2gpu_best.yaml}"
export VLLM_AURA_SENTENCE_TTS="${VLLM_AURA_SENTENCE_TTS:-0}"
export VLLM_AURA_TTS_TOKENIZER="${VLLM_AURA_TTS_TOKENIZER:-/workspace/models/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-CustomVoice/snapshots/0c0e3051f131929182e2c023b9537f8b1c68adfe}"

export VENV_DIR="${VENV_DIR:-/home/wtk/test/.venv}"
export VLLM_BIN="${VLLM_BIN:-$VENV_DIR/bin/vllm}"
export PYTHON_BIN="${PYTHON_BIN:-$VENV_DIR/bin/python}"
DEMO_PYTHON_BIN="${DEMO_PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"

export HF_HOME="${AURA_HF_HOME:-$HOME/.cache/huggingface}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-$HF_HOME/modules}"
# Prefer writable personal hub cache to avoid PermissionError on shared /workspace/models/hub.
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
if [[ "${TRANSFORMERS_CACHE:-}" == /workspace/model* ]]; then
  unset TRANSFORMERS_CACHE
fi
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export ALLOWED_LOCAL_MEDIA_PATH="${ALLOWED_LOCAL_MEDIA_PATH:-/workspace/models}"
mkdir -p "$HF_HOME" "$HF_MODULES_CACHE" "$HUGGINGFACE_HUB_CACHE" "$LOG_DIR"

AURA_PORT="${AURA_PORT:-8666}"
DEMO_PORT="${DEMO_PORT:-7862}"
DEMO_PORT_B="${DEMO_PORT_B:-7863}"
TTS_SPEAKER="${TTS_SPEAKER:-Vivian}"
TTS_INSTRUCT="${TTS_INSTRUCT:-}"
ASR_MODEL="${ASR_MODEL:-/workspace/models/Qwen3-ASR-1.7B}"
TTS_MODEL="${TTS_MODEL:-/workspace/models/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-CustomVoice/snapshots/0c0e3051f131929182e2c023b9537f8b1c68adfe}"

aura_ready() {
  curl -sf --max-time 2 "http://127.0.0.1:${AURA_PORT}/v1/models" >/dev/null 2>&1
}

# Rewrite HF ids → local paths so startup does not need hub write locks.
DEPLOY_LOCAL="${LOG_DIR}/aura_omni_2gpu_best_local.yaml"
"$PYTHON_BIN" - "$DEPLOY_SRC" "$DEPLOY_LOCAL" "$ASR_MODEL" "$MODEL" "$TTS_MODEL" <<'PY'
import sys
from pathlib import Path
import yaml

src, dst, asr, aura, tts = sys.argv[1:6]
cfg = yaml.safe_load(Path(src).read_text(encoding="utf-8"))
paths = {0: asr, 1: aura, 2: tts, 3: tts}
for stage in cfg["stages"]:
    stage["model"] = paths[int(stage["stage_id"])]
Path(dst).write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
print(dst)
PY
export DEPLOY="$DEPLOY_LOCAL"
export PORT="$AURA_PORT"

echo "AURA port=$AURA_PORT  demos=$DEMO_PORT + $DEMO_PORT_B"
echo "DEPLOY=$DEPLOY  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

if aura_ready; then
  echo "AURA already ready on :${AURA_PORT} — skipping start"
else
  bash "$REPO_ROOT/scripts/start_aura_omni.sh"
fi

if ! aura_ready; then
  echo "ERROR: AURA not reachable at http://127.0.0.1:${AURA_PORT}/v1/models" >&2
  exit 1
fi

start_demo() {
  local port="$1"
  local log="$LOG_DIR/demo_${port}.log"
  local pid_file="$LOG_DIR/demo_${port}.pid"
  if curl -sf --max-time 1 "http://127.0.0.1:${port}/healthz" >/dev/null 2>&1; then
    echo "demo :${port} already healthy — skipping"
    return
  fi
  nohup env \
    PYTHONPATH="$REPO_ROOT" \
    "$DEMO_PYTHON_BIN" -m examples.online_serving.aura_omni.minicpm_style_web_demo \
    --host 0.0.0.0 \
    --port "$port" \
    --ws-backend "ws://127.0.0.1:${AURA_PORT}" \
    --model "$MODEL" \
    --video-fps "${VIDEO_FPS:-2.0}" \
    --tts-task-type CustomVoice \
    --tts-language Chinese \
    --tts-speaker "$TTS_SPEAKER" \
    ${TTS_INSTRUCT:+--tts-instruct "$TTS_INSTRUCT"} \
    >"$log" 2>&1 &
  echo $! >"$pid_file"
  echo "demo :${port} pid=$(cat "$pid_file") log=$log"
}

start_demo "$DEMO_PORT"
start_demo "$DEMO_PORT_B"

echo "waiting for demos…"
for _ in $(seq 1 60); do
  ok_a=0
  ok_b=0
  curl -sf --max-time 1 "http://127.0.0.1:${DEMO_PORT}/healthz" >/dev/null 2>&1 && ok_a=1
  curl -sf --max-time 1 "http://127.0.0.1:${DEMO_PORT_B}/healthz" >/dev/null 2>&1 && ok_b=1
  if (( ok_a && ok_b )); then
    break
  fi
  sleep 2
done

curl -sf --max-time 2 "http://127.0.0.1:${DEMO_PORT}/healthz" >/dev/null
curl -sf --max-time 2 "http://127.0.0.1:${DEMO_PORT_B}/healthz" >/dev/null

HOST_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
echo
echo "Demo ready"
echo "  local:  http://127.0.0.1:${DEMO_PORT}/"
echo "  local:  http://127.0.0.1:${DEMO_PORT_B}/"
if [[ -n "${HOST_IP:-}" ]]; then
  echo "  LAN:    http://${HOST_IP}:${DEMO_PORT}/"
  echo "  LAN:    http://${HOST_IP}:${DEMO_PORT_B}/"
fi
echo "  AURA:   http://127.0.0.1:${AURA_PORT}/v1/models"
echo "stop: bash $SCRIPT_DIR/stop_2gpu_dual_demo_stack.sh"
