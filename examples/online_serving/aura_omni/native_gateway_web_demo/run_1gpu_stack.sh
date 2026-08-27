#!/usr/bin/env bash
# Start AURA_v2 Omni on one free GPU and serve the original Native frontend.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"
PYTHON_BIN="${PYTHON_BIN:-$VENV_DIR/bin/python}"

# Local tool keys (outside git). Override by exporting before this script.
AURA_TOOL_KEYS_ENV="${AURA_TOOL_KEYS_ENV:-$HOME/.config/aura/tool_keys.env}"
if [[ -f "$AURA_TOOL_KEYS_ENV" ]]; then
  # shellcheck disable=SC1090
  set -a
  source "$AURA_TOOL_KEYS_ENV"
  set +a
fi

MODEL="${MODEL:-/workspace/models/AURA_v2}"
BASE_TTS_MODEL="/workspace/models/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/fd4b254389122332181a7c3db7f27e918eec64e3"
DEPLOY="${DEPLOY:-$SCRIPT_DIR/aura_omni_v2_1gpu_base.yaml}"
AURA_PORT="${AURA_PORT:-8666}"
BRIDGE_PORT="${BRIDGE_PORT:-9999}"
LOG_DIR="${LOG_DIR:-/tmp/aura_v2_native_demo}"
PID_FILE="${PID_FILE:-$LOG_DIR/server.pid}"
BRIDGE_PID_FILE="${BRIDGE_PID_FILE:-$LOG_DIR/bridge.pid}"
STATIC_DIR="${STATIC_DIR:-$SCRIPT_DIR/static}"
AURA_GPU="${AURA_GPU:-auto}"
DEFAULT_TTS_INSTRUCT="请用专业、清晰、自然的语气说话，语速稍快，情绪克制，避免夸张和过度热情。"

if [[ "$AURA_GPU" == "auto" ]]; then
  AURA_GPU="$(
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
      awk -F, '$2 + 0 < 2000 {gsub(/ /, "", $1); print $1; exit}'
  )"
  if [[ -z "$AURA_GPU" ]]; then
    echo "ERROR: no GPU with less than 2 GiB in use; refusing to stop other users' processes." >&2
    exit 1
  fi
fi

mkdir -p "$LOG_DIR"

if [[ "${CHECK_ONLY:-0}" == "1" ]]; then
  [[ -x "$PYTHON_BIN" ]] || { echo "ERROR: Python missing: $PYTHON_BIN" >&2; exit 1; }
  [[ -f "$DEPLOY" ]] || { echo "ERROR: deploy missing: $DEPLOY" >&2; exit 1; }
  [[ -f "$STATIC_DIR/index.html" ]] || { echo "ERROR: Native UI missing: $STATIC_DIR/index.html" >&2; exit 1; }
  echo "preflight passed: gpu=$AURA_GPU model=$MODEL deploy=$DEPLOY"
  exit 0
fi

port_open() {
  "$PYTHON_BIN" - "$1" <<'PY'
import socket
import sys

with socket.socket() as sock:
    sock.settimeout(0.3)
    raise SystemExit(0 if sock.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0 else 1)
PY
}

if port_open "$BRIDGE_PORT"; then
  echo "ERROR: bridge port $BRIDGE_PORT is already in use." >&2
  exit 1
fi

if models_json="$(curl -sf --max-time 2 "http://127.0.0.1:${AURA_PORT}/v1/models" 2>/dev/null)"; then
  if ! "$PYTHON_BIN" - "$MODEL" "$models_json" <<'PY'
import json
import sys

expected = sys.argv[1]
payload = json.loads(sys.argv[2])
served = {str(item.get("id")) for item in payload.get("data", []) if isinstance(item, dict)}
raise SystemExit(0 if expected in served else 1)
PY
  then
    echo "ERROR: :$AURA_PORT serves a different model; refusing to reuse it." >&2
    exit 1
  fi
  echo "Using existing AURA_v2 backend on :$AURA_PORT"
else
  if port_open "$AURA_PORT"; then
    echo "ERROR: AURA port $AURA_PORT is occupied by a non-AURA service." >&2
    exit 1
  fi
  echo "Starting AURA_v2 on physical GPU $AURA_GPU ..."
  (
    cd "$REPO_ROOT"
    env \
      CUDA_VISIBLE_DEVICES="$AURA_GPU" \
      ALLOW_ONE_GPU=1 \
      MODEL="$MODEL" \
      DEPLOY="$DEPLOY" \
      HOST=0.0.0.0 \
      PORT="$AURA_PORT" \
      LOG_DIR="$LOG_DIR" \
      PID_FILE="$PID_FILE" \
      VENV_DIR="$VENV_DIR" \
      VLLM_AURA_SILENT_TOKEN_ID=248070 \
      VLLM_AURA_IM_END_TOKEN_ID=248046 \
      VLLM_AURA_IM_START_TOKEN_ID=248045 \
      VLLM_AURA_ASSISTANT_TOKEN_ID=74455 \
      VLLM_AURA_SENTENCE_TTS=0 \
      VLLM_AURA_TOOL_EXECUTOR=safe \
      VLLM_AURA_TTS_TOKENIZER="${VLLM_AURA_TTS_TOKENIZER:-$BASE_TTS_MODEL}" \
      ALLOWED_LOCAL_MEDIA_PATH="${ALLOWED_LOCAL_MEDIA_PATH:-$REPO_ROOT/tests/assets/qwen3_tts}" \
      ${DEEPSEEK_API_KEY:+DEEPSEEK_API_KEY="$DEEPSEEK_API_KEY"} \
      ${SERPER_API_KEY:+SERPER_API_KEY="$SERPER_API_KEY"} \
      bash scripts/start_aura_omni.sh
  ) >"$LOG_DIR/aura_start.out" 2>&1 &
  echo $! >"$LOG_DIR/aura_wrapper.pid"

  for attempt in $(seq 1 120); do
    if curl -sf --max-time 2 "http://127.0.0.1:${AURA_PORT}/v1/models" >/dev/null 2>&1; then
      echo "AURA_v2 ready after about $((attempt * 10)) seconds."
      break
    fi
    if [[ -f "$PID_FILE" ]] && ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
      echo "ERROR: AURA_v2 exited; see $LOG_DIR/aura_start.out" >&2
      exit 1
    fi
    sleep 10
  done
  if ! curl -sf --max-time 2 "http://127.0.0.1:${AURA_PORT}/v1/models" >/dev/null; then
    echo "ERROR: timed out waiting for AURA_v2." >&2
    exit 1
  fi
fi

if [[ "${SKIP_WARMUP:-0}" != "1" ]]; then
  "$PYTHON_BIN" "$SCRIPT_DIR/warmup_aura.py" \
    --aura-ws "ws://127.0.0.1:${AURA_PORT}/v1/video/chat/stream" \
    --model "$MODEL" \
    --tts-speaker "${TTS_SPEAKER:-Vivian}" \
    --tts-language "${TTS_LANGUAGE:-Chinese}" \
    --tts-instruct "${TTS_INSTRUCT:-$DEFAULT_TTS_INSTRUCT}" \
    --tts-task-type "${TTS_TASK_TYPE:-Base}" \
    --frame-count "${WARMUP_FRAME_COUNT:-2}" \
    --frame-width "${WARMUP_FRAME_WIDTH:-640}" \
    --frame-height "${WARMUP_FRAME_HEIGHT:-360}" \
    --silent-first \
    | tee "$LOG_DIR/warmup.out"
fi

nohup env \
  AURA_WS_URL="ws://127.0.0.1:${AURA_PORT}/v1/video/chat/stream" \
  AURA_MODEL="$MODEL" \
  BRIDGE_HOST=0.0.0.0 \
  BRIDGE_PORT="$BRIDGE_PORT" \
  STATIC_DIR="$STATIC_DIR" \
  TTS_SPEAKER="${TTS_SPEAKER:-Vivian}" \
  TTS_LANGUAGE="${TTS_LANGUAGE:-Chinese}" \
  TTS_INSTRUCT="${TTS_INSTRUCT:-$DEFAULT_TTS_INSTRUCT}" \
  TTS_TASK_TYPE="${TTS_TASK_TYPE:-Base}" \
  TOOL_MODE=auto \
  MAX_TOOL_DEPTH=3 \
  AUTO_TRIGGER="${AUTO_TRIGGER:-1}" \
  AURA_TTS_DUMP_DIR="${AURA_TTS_DUMP_DIR:-$LOG_DIR/tts}" \
  "$PYTHON_BIN" "$SCRIPT_DIR/server.py" \
  >"$LOG_DIR/bridge.out" 2>&1 &
echo $! >"$BRIDGE_PID_FILE"

for _ in $(seq 1 20); do
  curl -sf "http://127.0.0.1:${BRIDGE_PORT}/health" >/dev/null && break
  sleep 0.5
done
if ! curl -sf "http://127.0.0.1:${BRIDGE_PORT}/health" >/dev/null; then
  echo "ERROR: bridge failed; see $LOG_DIR/bridge.out" >&2
  exit 1
fi

LAN_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
echo "AURA_v2 backend: http://127.0.0.1:${AURA_PORT}"
echo "Native frontend: http://127.0.0.1:${BRIDGE_PORT}/"
[[ -n "$LAN_IP" ]] && echo "LAN frontend: http://${LAN_IP}:${BRIDGE_PORT}/"
echo "Stop: LOG_DIR=$LOG_DIR bash $SCRIPT_DIR/stop_1gpu_stack.sh"
