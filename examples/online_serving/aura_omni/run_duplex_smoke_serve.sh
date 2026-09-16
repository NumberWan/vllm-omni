#!/usr/bin/env bash
# Minimal AURA duplex Realtime server for smoke.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

PORT="${PORT:-8099}"
MODEL="${MODEL:-/workspace/models/AURA_v2new}"
DEPLOY_CONFIG="${DEPLOY_CONFIG:-$REPO_ROOT/examples/online_serving/aura_omni/aura_omni_v2_duplex_smoke.yaml}"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"

export VLLM_AURA_SILENT_TOKEN_ID="${VLLM_AURA_SILENT_TOKEN_ID:-248070}"
export VLLM_AURA_IM_END_TOKEN_ID="${VLLM_AURA_IM_END_TOKEN_ID:-248046}"
export VLLM_AURA_IM_START_TOKEN_ID="${VLLM_AURA_IM_START_TOKEN_ID:-248045}"
export VLLM_AURA_ASSISTANT_TOKEN_ID="${VLLM_AURA_ASSISTANT_TOKEN_ID:-74455}"
# Do NOT set TORCHDYNAMO_DISABLE here: it breaks Stage0/1 AOT when those
# stages are not enforce_eager. Stage2/3 use enforce_eager; code_predictor
# honors that and skips its own torch.compile.

exec "$REPO_ROOT/.venv/bin/vllm-omni" serve "$MODEL" \
  --omni \
  --deploy-config "$DEPLOY_CONFIG" \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port "$PORT"
