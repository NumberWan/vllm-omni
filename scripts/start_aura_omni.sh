#!/usr/bin/env bash
# Start AURA Omni with the recommended 2-GPU best config.
# Defaults are already optimal — only override when you need a different trade-off.
#
#   bash scripts/start_aura_omni.sh
#
# See docs/aura/AURA_OMNI_TUNABLES.md
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLLM_BIN="${VLLM_BIN:-$ROOT/.venv/bin/vllm}"
if [[ ! -x "$VLLM_BIN" ]]; then
  VLLM_BIN="$(command -v vllm || true)"
fi
DEPLOY="${DEPLOY:-$ROOT/vllm_omni/deploy/aura_omni_2gpu_best.yaml}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8666}"
MODEL="${MODEL:-/models/AURA}"
LOG_DIR="${LOG_DIR:-/tmp/aura_omni_serve}"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="${LOG:-$LOG_DIR/server_${STAMP}.log}"
PID_FILE="${PID_FILE:-$LOG_DIR/server.pid}"

SERVER_ENV=("CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES")
for key in \
  VLLM_AURA_STAGE0_BYPASS \
  VLLM_AURA_SILENT_STOP_AT_STAGE1 \
  VLLM_AURA_TTS_GATE_ON_VOICE_ASR \
  VLLM_AURA_SENTENCE_TTS
do
  if [[ -n "${!key:-}" ]]; then
    SERVER_ENV+=("${key}=${!key}")
  fi
done

echo "=== AURA Omni start (2-GPU best defaults) ==="
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "DEPLOY=$DEPLOY"
echo "PORT=$PORT  MODEL=$MODEL"
echo "LOG=$LOG"
echo "tunables: $ROOT/docs/aura/AURA_OMNI_TUNABLES.md"

python3 - <<PY
import os, glob
port = int("$PORT")
target = f"{port:04X}"
for path in ["/proc/net/tcp", "/proc/net/tcp6"]:
    if not os.path.exists(path):
        continue
    with open(path) as f:
        next(f)
        for line in f:
            parts = line.split()
            if len(parts) < 10:
                continue
            if parts[1].split(":")[-1].upper() != target:
                continue
            inode = parts[9]
            for fd in glob.glob("/proc/[0-9]*/fd/*"):
                try:
                    if os.readlink(fd) == f"socket:[{inode}]":
                        os.kill(int(fd.split("/")[2]), 15)
                except Exception:
                    pass
PY
sleep 2

cd "$ROOT"
if [[ ! -x "$VLLM_BIN" ]]; then
  echo "vllm binary not found; set VLLM_BIN="; exit 1
fi
nohup env "${SERVER_ENV[@]}" \
  "$VLLM_BIN" serve "$MODEL" --omni \
    --deploy-config "$DEPLOY" \
    --host "$HOST" --port "$PORT" \
    --served-model-name "$MODEL" \
    --trust-remote-code --init-timeout 1200 \
    --allowed-local-media-path /models \
  >"$LOG" 2>&1 &
echo $! >"$PID_FILE"
echo "pid=$(cat "$PID_FILE")"

deadline=$((SECONDS + 1200))
until curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; do
  if ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "server died"; tail -80 "$LOG"; exit 1
  fi
  if (( SECONDS >= deadline )); then
    echo "timeout waiting for :$PORT"; tail -80 "$LOG"; exit 1
  fi
  sleep 5
done
echo "ready http://127.0.0.1:${PORT}/v1/models"
echo "tail -f $LOG"
