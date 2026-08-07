#!/usr/bin/env bash
# Stop the AURA process started by run_1gpu_demo_stack.sh / 1-GPU demo LOG_DIR.
# Default stop_aura_omni.sh looks at /tmp/aura_omni_serve — that is the wrong path here.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
export LOG_DIR="${LOG_DIR:-/tmp/aura_omni_1gpu_demo}"
export PID_FILE="${PID_FILE:-$LOG_DIR/server.pid}"
export STOP_TIMEOUT="${STOP_TIMEOUT:-30}"

bash "$REPO_ROOT/scripts/stop_aura_omni.sh"

# If the recorded serve process ignored TERM (common when a stage is wedged),
# escalate once so the next stack start is not blocked by a stale PID file.
if [[ -f "$PID_FILE" ]]; then
  pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "${pid:-}" ]] && [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
    cmdline="$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null || true)"
    if [[ "$cmdline" == *"vllm"* ]] && [[ "$cmdline" == *"serve"* ]]; then
      echo "pid=$pid still alive after TERM; sending KILL"
      kill -KILL "$pid" 2>/dev/null || true
      sleep 1
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      rm -f "$PID_FILE"
      echo "stopped pid=$pid"
    else
      echo "ERROR: pid=$pid still alive after KILL" >&2
      exit 1
    fi
  fi
fi
