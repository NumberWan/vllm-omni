#!/usr/bin/env bash
# Start vllm-omni AURA server with local /models/* weights.
#
# Full guide: benchmarks/omniinteract/SETUP_AND_RUN.md
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs

MODEL="${MODEL:-/models/AURA}"
DEPLOY_CONFIG="${DEPLOY_CONFIG:-vllm_omni/deploy/aura_omni.yaml}"
VLLM_BIN="${VLLM_BIN:-/public/wtk/.venv/bin/vllm}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8666}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"

echo "MODEL=$MODEL"
echo "DEPLOY_CONFIG=$DEPLOY_CONFIG"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Listen ${HOST}:${PORT}"

exec env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  "$VLLM_BIN" serve "$MODEL" \
  --omni \
  --deploy-config "$DEPLOY_CONFIG" \
  --host "$HOST" \
  --port "$PORT" \
  --served-model-name "$MODEL" \
  --trust-remote-code \
  "$@"
