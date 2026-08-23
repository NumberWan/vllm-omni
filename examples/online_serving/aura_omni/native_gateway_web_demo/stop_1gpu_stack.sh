#!/usr/bin/env bash
# Stop only processes recorded by the AURA_v2 Native demo stack.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
LOG_DIR="${LOG_DIR:-/tmp/aura_v2_native_demo}"
BRIDGE_PID_FILE="${BRIDGE_PID_FILE:-$LOG_DIR/bridge.pid}"

if [[ -f "$BRIDGE_PID_FILE" ]]; then
  bridge_pid="$(cat "$BRIDGE_PID_FILE" 2>/dev/null || true)"
  if [[ -n "$bridge_pid" ]] && kill -0 "$bridge_pid" 2>/dev/null; then
    kill -TERM "$bridge_pid"
    for _ in $(seq 1 50); do
      kill -0 "$bridge_pid" 2>/dev/null || break
      sleep 0.1
    done
    echo "Stopped Native frontend bridge pid=$bridge_pid."
  fi
  rm -f "$BRIDGE_PID_FILE"
fi

LOG_DIR="$LOG_DIR" \
PID_FILE="${PID_FILE:-$LOG_DIR/server.pid}" \
  bash "$REPO_ROOT/scripts/stop_aura_omni.sh"
