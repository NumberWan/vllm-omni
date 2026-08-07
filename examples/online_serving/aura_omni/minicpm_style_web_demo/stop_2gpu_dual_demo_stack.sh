#!/usr/bin/env bash
# Stop AURA + demos started by run_2gpu_dual_demo_stack.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
export LOG_DIR="${LOG_DIR:-/tmp/aura_omni_2gpu_dual_demo}"
export PID_FILE="${PID_FILE:-$LOG_DIR/server.pid}"
export STOP_TIMEOUT="${STOP_TIMEOUT:-30}"

DEMO_PORT="${DEMO_PORT:-7862}"
DEMO_PORT_B="${DEMO_PORT_B:-7863}"

for port in "$DEMO_PORT" "$DEMO_PORT_B"; do
  pid_file="$LOG_DIR/demo_${port}.pid"
  if [[ -f "$pid_file" ]]; then
    pid="$(cat "$pid_file" 2>/dev/null || true)"
    if [[ -n "${pid:-}" ]] && [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
      echo "stopping demo :${port} pid=$pid"
      kill -TERM "$pid" 2>/dev/null || true
      sleep 1
      kill -KILL "$pid" 2>/dev/null || true
    fi
    rm -f "$pid_file"
  fi
  fuser -k "${port}/tcp" 2>/dev/null || true
done

bash "$REPO_ROOT/scripts/stop_aura_omni.sh"

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

echo "stopped 2-GPU dual-demo stack (LOG_DIR=$LOG_DIR)"
