#!/usr/bin/env bash
# Start vllm-omni AURA server with local /models/* weights.
#
# Prefer scripts/start_aura_omni.sh for the recommended 2-GPU best profile.
# Full guide: benchmarks/omniinteract/SETUP_AND_RUN.md
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs

if [[ -x "$ROOT/scripts/start_aura_omni.sh" ]] && [[ "${USE_START_AURA_OMNI:-1}" == "1" ]]; then
  exec bash "$ROOT/scripts/start_aura_omni.sh" "$@"
fi

MODEL="${MODEL:-/models/AURA}"
DEPLOY_CONFIG="${DEPLOY_CONFIG:-vllm_omni/deploy/aura_omni.yaml}"
VLLM_BIN="${VLLM_BIN:-$ROOT/.venv/bin/vllm}"
if [[ ! -x "$VLLM_BIN" ]]; then
  VLLM_BIN="$(command -v vllm || true)"
fi
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8666}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"

if [[ ! -x "$VLLM_BIN" ]]; then
  echo "vllm binary not found; set VLLM_BIN="; exit 1
fi

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
