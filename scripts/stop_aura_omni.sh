#!/usr/bin/env bash
# Stop only the AURA server recorded by scripts/start_aura_omni.sh.
set -euo pipefail

LOG_DIR="${LOG_DIR:-/tmp/aura_omni_serve}"
PID_FILE="${PID_FILE:-$LOG_DIR/server.pid}"
STOP_TIMEOUT="${STOP_TIMEOUT:-30}"

if [[ ! -f "$PID_FILE" ]]; then
  echo "AURA server is not running (no PID file at $PID_FILE)."
  exit 0
fi

pid="$(cat "$PID_FILE" 2>/dev/null || true)"
if [[ -z "$pid" ]] || ! [[ "$pid" =~ ^[0-9]+$ ]]; then
  echo "ERROR: invalid PID file: $PID_FILE" >&2
  exit 1
fi

if ! kill -0 "$pid" 2>/dev/null; then
  rm -f "$PID_FILE"
  echo "Removed stale AURA PID file (pid=$pid is not running)."
  exit 0
fi

cmdline="$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null || true)"
if [[ "$cmdline" != *"vllm"* ]] || [[ "$cmdline" != *"serve"* ]]; then
  echo "ERROR: refusing to stop pid=$pid because it is not a vllm serve process: $cmdline" >&2
  exit 1
fi

kill -TERM "$pid"
deadline=$((SECONDS + STOP_TIMEOUT))
while kill -0 "$pid" 2>/dev/null; do
  if (( SECONDS >= deadline )); then
    echo "ERROR: pid=$pid did not stop within ${STOP_TIMEOUT}s; inspect it manually." >&2
    exit 1
  fi
  sleep 1
done

rm -f "$PID_FILE"
echo "Stopped AURA server pid=$pid."
