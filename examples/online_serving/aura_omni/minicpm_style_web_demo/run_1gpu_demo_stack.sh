#!/usr/bin/env bash
# One command: start AURA 1-GPU demo server (if needed) + MiniCPM-style browser UI.
#
#   bash examples/online_serving/aura_omni/minicpm_style_web_demo/run_1gpu_demo_stack.sh
#
# Env overrides (optional): MODEL, CUDA_VISIBLE_DEVICES, AURA_PORT, DEMO_PORT,
#   TTS_SPEAKER, VLLM_AURA_SENTENCE_TTS, VENV_DIR / VLLM_BIN / PYTHON_BIN (AURA),
#   DEMO_PYTHON_BIN (UI), AURA_HF_HOME (writable HF cache; default ~/.cache/huggingface).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

export MODEL="${MODEL:-/workspace/models/AURA}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export ALLOW_ONE_GPU="${ALLOW_ONE_GPU:-1}"
export LOG_DIR="${LOG_DIR:-/tmp/aura_omni_1gpu_demo}"
export DEPLOY="${DEPLOY:-$SCRIPT_DIR/aura_omni_1gpu_demo.yaml}"
# Wait for full AURA text before TTS (CustomVoice one-shot cannot yet chain
# mid-gen sentence payloads on the same request_id — later clauses get dropped).
# Set VLLM_AURA_SENTENCE_TTS=1 to try per-sentence handoff / lower TTFP.
export VLLM_AURA_SENTENCE_TTS="${VLLM_AURA_SENTENCE_TTS:-0}"

export VENV_DIR="${VENV_DIR:-/home/wtk/test/.venv}"
export VLLM_BIN="${VLLM_BIN:-$VENV_DIR/bin/vllm}"
export PYTHON_BIN="${PYTHON_BIN:-$VENV_DIR/bin/python}"
DEMO_PYTHON_BIN="${DEMO_PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"

# Parent shells often export HF_HOME=/workspace/model (root-owned → PermissionError).
# Always use a writable cache for dynamic modules; model weights stay at MODEL.
export HF_HOME="${AURA_HF_HOME:-$HOME/.cache/huggingface}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-$HF_HOME/modules}"
# Prefer shared hub if present; otherwise hub under HF_HOME.
if [[ -d /workspace/models/hub ]]; then
  export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/workspace/models/hub}"
else
  export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
fi
# Do not inherit TRANSFORMERS_CACHE=/workspace/model* from the parent shell.
if [[ "${TRANSFORMERS_CACHE:-}" == /workspace/model* ]]; then
  unset TRANSFORMERS_CACHE
fi
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
mkdir -p "$HF_HOME" "$HF_MODULES_CACHE" "$HUGGINGFACE_HUB_CACHE"

AURA_PORT_PREF="${AURA_PORT:-8666}"
DEMO_PORT_PREF="${DEMO_PORT:-7862}"
TTS_SPEAKER="${TTS_SPEAKER:-Vivian}"
PORT_SCAN_SPAN="${PORT_SCAN_SPAN:-40}"

mkdir -p "$LOG_DIR"
PID_FILE="${PID_FILE:-$LOG_DIR/server.pid}"

port_listening() {
  local port="$1"
  "$PYTHON_BIN" - "$port" <<'PY'
import socket, sys
port = int(sys.argv[1])
with socket.socket() as s:
    s.settimeout(0.3)
    raise SystemExit(0 if s.connect_ex(("127.0.0.1", port)) == 0 else 1)
PY
}

aura_ready_on() {
  local port="$1"
  curl -sf --max-time 2 "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1
}

# First free TCP port in [start, start+span).
pick_free_port() {
  local start="$1"
  "$PYTHON_BIN" - "$start" "$PORT_SCAN_SPAN" <<'PY'
import socket, sys
start, span = int(sys.argv[1]), int(sys.argv[2])
for port in range(start, start + span):
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            continue
        print(port)
        raise SystemExit(0)
raise SystemExit(f"ERROR: no free port in {start}..{start + span - 1}")
PY
}

# Prefer an already-ready AURA on the preferred port; else first free port.
resolve_aura_port() {
  local pref="$1"
  if aura_ready_on "$pref"; then
    echo "$pref"
    return
  fi
  if ! port_listening "$pref"; then
    echo "$pref"
    return
  fi
  echo "Preferred AURA port :${pref} is busy (not our ready server); scanning…" >&2
  pick_free_port "$((pref + 1))"
}

resolve_demo_port() {
  local pref="$1"
  if ! port_listening "$pref"; then
    echo "$pref"
    return
  fi
  echo "Preferred demo port :${pref} is busy; scanning…" >&2
  pick_free_port "$((pref + 1))"
}

export PORT="$(resolve_aura_port "$AURA_PORT_PREF")"
DEMO_PORT="$(resolve_demo_port "$DEMO_PORT_PREF")"
WS_BACKEND="${WS_BACKEND:-ws://127.0.0.1:${PORT}}"

echo "HF_HOME=$HF_HOME  (modules → $HF_MODULES_CACHE)"
echo "AURA port=$PORT  demo port=$DEMO_PORT"


if aura_ready_on "$PORT"; then
  echo "AURA already ready on :${PORT} — skipping start"
else
  if [[ -f "$PID_FILE" ]]; then
    old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [[ -n "$old_pid" ]] && ! kill -0 "$old_pid" 2>/dev/null; then
      rm -f "$PID_FILE"
    elif [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
      echo "ERROR: AURA pid=$old_pid is recorded at $PID_FILE but :${PORT} is not ready." >&2
      echo "Stop it with:" >&2
      echo "  bash $SCRIPT_DIR/stop_1gpu_demo_stack.sh" >&2
      exit 1
    fi
  fi
  echo "=== Starting AURA 1-GPU demo stack ==="
  bash "$REPO_ROOT/scripts/start_aura_omni.sh"
fi

if ! aura_ready_on "$PORT"; then
  echo "ERROR: AURA not reachable at http://127.0.0.1:${PORT}/v1/models" >&2
  exit 1
fi

echo "=== Starting browser demo on :${DEMO_PORT} ==="
exec env \
  PYTHON_BIN="$DEMO_PYTHON_BIN" \
  MODEL="$MODEL" \
  WS_BACKEND="$WS_BACKEND" \
  TTS_SPEAKER="$TTS_SPEAKER" \
  PORT="$DEMO_PORT" \
  bash "$SCRIPT_DIR/run_demo.sh"
