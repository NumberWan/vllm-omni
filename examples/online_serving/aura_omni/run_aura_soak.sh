#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Start AURA streaming soak in background (12h, 10 consecutive passes).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

mkdir -p logs

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"

PYTHON="${AURA_SOAK_PYTHON:-/public/wtk/.venv/bin/python}"
TARGET="${AURA_SOAK_TARGET:-10}"
MAX_HOURS="${AURA_SOAK_MAX_HOURS:-12}"

echo "Starting AURA soak: target=${TARGET} consecutive, max_hours=${MAX_HOURS}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Logs: ${REPO_ROOT}/logs/aura_soak_<timestamp>/ and logs/aura_soak_launcher.log"

nohup "$PYTHON" examples/online_serving/aura_omni/aura_streaming_soak.py \
  --target-consecutive "$TARGET" \
  --max-hours "$MAX_HOURS" \
  > logs/aura_soak_launcher.log 2>&1 &

echo "PID=$!"
echo "Tail launcher: tail -f logs/aura_soak_launcher.log"
