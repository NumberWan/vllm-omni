#!/usr/bin/env bash
# Start AURA Omni with the recommended 2-GPU best config.
# Defaults are already optimal — only override when you need a different trade-off.
#
#   bash scripts/start_aura_omni.sh
#
# See docs/aura/AURA_OMNI_TUNABLES.md
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT/.venv}"
VLLM_BIN="${VLLM_BIN:-$VENV_DIR/bin/vllm}"
if [[ ! -x "$VLLM_BIN" ]]; then
  VLLM_BIN="$(command -v vllm || true)"
fi
DEPLOY="${DEPLOY:-$ROOT/vllm_omni/deploy/aura_omni_2gpu_best.yaml}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8666}"
MODEL="${MODEL:-aurateam/AURA}"
LOG_DIR="${LOG_DIR:-/tmp/aura_omni_serve}"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="${LOG:-$LOG_DIR/server_${STAMP}.log}"
PID_FILE="${PID_FILE:-$LOG_DIR/server.pid}"
PYTHON_BIN="${PYTHON_BIN:-$VENV_DIR/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3 || true)"
fi
ALLOWED_LOCAL_MEDIA_PATH="${ALLOWED_LOCAL_MEDIA_PATH:-}"

# AURA_v2 closed-think prefix / silent trim read these env ids, not YAML
# stop_token_ids. If Stage1 is AURA_v2 and the ids are unset, workers fall
# back to v1 assistant prefix and the model can drift into English/LaTeX/code.
if [[ "$MODEL" == *AURA_v2* ]] \
  || grep -Eq 'AURA_v2|248070' "$DEPLOY" 2>/dev/null; then
  : "${VLLM_AURA_SILENT_TOKEN_ID:=248070}"
  : "${VLLM_AURA_IM_END_TOKEN_ID:=248046}"
  : "${VLLM_AURA_IM_START_TOKEN_ID:=248045}"
  : "${VLLM_AURA_ASSISTANT_TOKEN_ID:=74455}"
  export VLLM_AURA_SILENT_TOKEN_ID VLLM_AURA_IM_END_TOKEN_ID \
    VLLM_AURA_IM_START_TOKEN_ID VLLM_AURA_ASSISTANT_TOKEN_ID
fi

SERVER_ENV=("CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES")
# Avoid leaking a parent-shell PYTHONPATH (e.g. another checkout's site-packages)
# into the AURA process, which would defeat an isolated VENV_DIR install.
SERVER_ENV+=("PYTHONPATH=")
for key in \
  VLLM_AURA_STAGE0_BYPASS \
  VLLM_AURA_TTS_GATE_ON_VOICE_ASR \
  VLLM_AURA_SENTENCE_TTS \
  VLLM_AURA_TTS_TOKENIZER \
  VLLM_AURA_SILENT_TOKEN_ID \
  VLLM_AURA_IM_END_TOKEN_ID \
  VLLM_AURA_IM_START_TOKEN_ID \
  VLLM_AURA_ASSISTANT_TOKEN_ID \
  VLLM_AURA_TOOL_EXECUTOR \
  VLLM_AURA_TOOL_BRAVE \
  VLLM_AURA_TOOL_DDG \
  VLLM_AURA_TOOL_WEBFETCH \
  DEEPSEEK_API_KEY \
  DEEPSEEK_BASE_URL \
  DEEPSEEK_MODEL \
  SERPER_API_KEY \
  SERPER_SEARCH_ENDPOINT \
  BRAVE_SEARCH_API_KEY \
  BRAVE_SEARCH_ENDPOINT \
  VLLM_VIDEO_ASYNC_CHUNK \
  VLLM_LOGGING_LEVEL
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

cd "$ROOT"
if [[ -z "$PYTHON_BIN" ]] || [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Python not found. Run: bash scripts/install_aura_omni.sh" >&2
  exit 1
fi
if [[ ! -x "$VLLM_BIN" ]]; then
  echo "ERROR: vllm binary not found; install vllm-omni or set VLLM_BIN." >&2
  exit 1
fi
if [[ ! -f "$DEPLOY" ]]; then
  echo "ERROR: deploy config not found: $DEPLOY" >&2
  exit 1
fi
if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "ERROR: AURA server already running (pid=$old_pid). Run scripts/stop_aura_omni.sh first." >&2
    exit 1
  fi
  rm -f "$PID_FILE"
fi

"$PYTHON_BIN" - "$HOST" "$PORT" <<'PY'
import socket
import sys

host, port = sys.argv[1], int(sys.argv[2])
probe_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
with socket.socket() as sock:
    sock.settimeout(0.5)
    if sock.connect_ex((probe_host, port)) == 0:
        raise SystemExit(f"ERROR: {probe_host}:{port} is already in use; refusing to terminate another process.")
PY

"$PYTHON_BIN" - <<'PY'
import importlib.metadata as md

try:
    import flashinfer  # noqa: F401
except Exception as exc:
    raise SystemExit(
        "ERROR: FlashInfer is required by the production deploy. "
        "Run: bash scripts/install_aura_omni.sh\n"
        f"Original error: {exc}"
    ) from exc

try:
    runtime = md.version("flashinfer-python").split("+", 1)[0]
    jit = md.version("flashinfer-jit-cache").split("+", 1)[0]
except md.PackageNotFoundError as exc:
    raise SystemExit(
        "ERROR: FlashInfer runtime or JIT cache is missing. "
        "Run: bash scripts/install_aura_omni.sh"
    ) from exc
else:
    if runtime != jit:
        raise SystemExit(
            f"ERROR: FlashInfer package mismatch: flashinfer-python={runtime}, "
            f"flashinfer-jit-cache={jit}."
        )
PY

IFS=',' read -r -a gpu_ids <<<"$CUDA_VISIBLE_DEVICES"
if (( ${#gpu_ids[@]} < 2 )) && [[ "${ALLOW_ONE_GPU:-0}" != "1" ]]; then
  echo "ERROR: the production profile requires two visible GPUs; got CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2
  echo "hint: for temporary single-GPU demo packs set ALLOW_ONE_GPU=1" >&2
  exit 1
fi

MEDIA_ARGS=()
if [[ -n "$ALLOWED_LOCAL_MEDIA_PATH" ]]; then
  if [[ ! -d "$ALLOWED_LOCAL_MEDIA_PATH" ]]; then
    echo "ERROR: ALLOWED_LOCAL_MEDIA_PATH is not a directory: $ALLOWED_LOCAL_MEDIA_PATH" >&2
    exit 1
  fi
  MEDIA_ARGS+=(--allowed-local-media-path "$ALLOWED_LOCAL_MEDIA_PATH")
fi

TOOL_ARGS=()
tool_executor="${VLLM_AURA_TOOL_EXECUTOR:-}"
tool_executor="${tool_executor,,}"
if [[ "$tool_executor" == "mock" || "$tool_executor" == "safe" ]]; then
  has_auto_tool_choice=0
  has_tool_parser=0
  for arg in "$@"; do
    [[ "$arg" == "--enable-auto-tool-choice" ]] && has_auto_tool_choice=1
    [[ "$arg" == "--tool-call-parser" || "$arg" == --tool-call-parser=* ]] && has_tool_parser=1
  done
  (( has_auto_tool_choice )) || TOOL_ARGS+=(--enable-auto-tool-choice)
  (( has_tool_parser )) || TOOL_ARGS+=(--tool-call-parser qwen3_xml)
fi

if [[ "${CHECK_ONLY:-0}" == "1" ]]; then
  echo "preflight checks passed"
  exit 0
fi

nohup env "${SERVER_ENV[@]}" \
  "$VLLM_BIN" serve "$MODEL" --omni \
    --deploy-config "$DEPLOY" \
    --host "$HOST" --port "$PORT" \
    --served-model-name "$MODEL" \
    --trust-remote-code --init-timeout 1200 \
    "${MEDIA_ARGS[@]}" "${TOOL_ARGS[@]}" "$@" \
  >"$LOG" 2>&1 &
echo $! >"$PID_FILE"
echo "pid=$(cat "$PID_FILE")"
echo "waiting for model load on :$PORT (can take several minutes for multi-stage AURA)..."
echo "  live log: tail -f $LOG"

STAGE_LABELS=("ASR" "AURA" "TTS" "Code2Wav")
DASHBOARD_LINES=$((1 + ${#STAGE_LABELS[@]}))
DASHBOARD_DRAWN=0
# Interactive TTY: rewrite in place + tick every 1s.
# Piped/non-TTY: avoid ANSI spam; print a fresh block every 15s.
if [[ -t 1 ]]; then
  DASHBOARD_INPLACE=1
  DASHBOARD_INTERVAL=1
else
  DASHBOARD_INPLACE=0
  DASHBOARD_INTERVAL=15
fi

stage_status() {
  local stage="$1"
  if grep -Fq "Stage $stage initialized" "$LOG" 2>/dev/null; then
    printf '✓ ready'
  elif grep -Eiq "stage${stage}.*Starting to load model" "$LOG" 2>/dev/null; then
    printf '↓ loading'
  elif grep -Fq "[stage_init] Stage-$stage" "$LOG" 2>/dev/null; then
    printf '↻ starting'
  else
    printf '· waiting'
  fi
}

show_startup_dashboard() {
  local elapsed="$1"
  local stage
  if (( DASHBOARD_INPLACE && DASHBOARD_DRAWN )); then
    printf '\033[%dA' "$DASHBOARD_LINES"
  elif (( DASHBOARD_INPLACE == 0 )); then
    printf '\n'
  fi
  # \033[K clears to end of line so shorter statuses don't leave leftovers.
  printf 'loading AURA Omni  %ss\033[K\n' "$elapsed"
  for stage in "${!STAGE_LABELS[@]}"; do
    printf '  stage%d %-8s %s\033[K\n' \
      "$stage" "${STAGE_LABELS[$stage]}" "$(stage_status "$stage")"
  done
  DASHBOARD_DRAWN=1
}

deadline=$((SECONDS + 1200))
started=$SECONDS
next_heartbeat=$SECONDS
until curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; do
  if ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    if (( DASHBOARD_DRAWN )); then printf '\n'; fi
    echo "ERROR: server exited during startup" >&2
    rm -f "$PID_FILE"
    tail -80 "$LOG"
    exit 1
  fi
  if (( SECONDS >= deadline )); then
    if (( DASHBOARD_DRAWN )); then printf '\n'; fi
    echo "ERROR: timeout waiting for :$PORT; stopping pid=$(cat "$PID_FILE")" >&2
    kill -TERM "$(cat "$PID_FILE")" 2>/dev/null || true
    rm -f "$PID_FILE"
    tail -80 "$LOG"
    exit 1
  fi
  if (( SECONDS >= next_heartbeat )); then
    show_startup_dashboard "$((SECONDS - started))"
    next_heartbeat=$((SECONDS + DASHBOARD_INTERVAL))
  fi
  sleep 1
done
show_startup_dashboard "$((SECONDS - started))"
echo "ready http://127.0.0.1:${PORT}/v1/models  (loaded in $((SECONDS - started))s)"
echo "tail -f $LOG"
echo "stop: bash scripts/stop_aura_omni.sh"
