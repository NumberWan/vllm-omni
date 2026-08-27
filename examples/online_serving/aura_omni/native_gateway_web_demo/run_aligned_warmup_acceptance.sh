#!/usr/bin/env bash
# Validate three fresh browser-aligned warmups on isolated 2-GPU stacks.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-/home/wtk/vllm-omni-AURA_026/.venv}"
PYTHON_BIN="$VENV_DIR/bin/python"
OUT_ROOT="${OUT_ROOT:-/tmp/aura_aligned_warmup_acceptance}"
REPS="${REPS:-3}"
AURA_PORT="${AURA_PORT:-8670}"
BRIDGE_PORT="${BRIDGE_PORT:-19999}"
RESULTS="$OUT_ROOT/results.tsv"

mkdir -p "$OUT_ROOT"
printf 'rep\twarmup_s\twarm_voice_s\tverify_s\tjit_before\tjit_after\tnew_jit\tpass\tlog_dir\n' >"$RESULTS"

port_open() {
  "$PYTHON_BIN" - "$1" <<'PY'
import socket
import sys
with socket.socket() as sock:
    sock.settimeout(0.3)
    raise SystemExit(0 if sock.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0 else 1)
PY
}

jit_count() {
  local log="$1"
  local count
  count="$(rg -c 'Triton kernel JIT compilation during inference' "$log" 2>/dev/null || true)"
  printf '%s' "${count:-0}"
}

launcher_pid=""
log_dir=""
cleanup() {
  if [[ -n "$log_dir" ]]; then
    LOG_DIR="$log_dir" bash "$SCRIPT_DIR/stop_1gpu_stack.sh" >/dev/null 2>&1 || true
  fi
  if [[ -n "$launcher_pid" ]]; then
    wait "$launcher_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

for rep in $(seq 1 "$REPS"); do
  log_dir="$OUT_ROOT/aligned_${rep}"
  mkdir -p "$log_dir"
  if port_open "$AURA_PORT" || port_open "$BRIDGE_PORT"; then
    echo "ERROR: isolated test ports are occupied before repetition $rep" >&2
    exit 1
  fi

  env \
    AURA_PORT="$AURA_PORT" \
    BRIDGE_PORT="$BRIDGE_PORT" \
    LOG_DIR="$log_dir" \
    VENV_DIR="$VENV_DIR" \
    VLLM_AURA_SENTENCE_TTS=0 \
    TOOL_MODE=none \
    GPU_TIMEOUT=2h \
    GPU_NOTE="AURA aligned warmup acceptance ${rep}/${REPS}" \
    bash "$SCRIPT_DIR/run_2gpu_stack.sh" >"$log_dir/stack.out" 2>&1 &
  launcher_pid=$!

  ready=0
  for _ in $(seq 1 180); do
    if curl -sf --max-time 2 "http://127.0.0.1:${BRIDGE_PORT}/health" >/dev/null; then
      ready=1
      break
    fi
    if ! kill -0 "$launcher_pid" 2>/dev/null; then
      break
    fi
    sleep 5
  done
  if [[ "$ready" != "1" ]]; then
    echo "ERROR: repetition $rep failed readiness; see $log_dir/stack.out" >&2
    exit 1
  fi

  server_logs=("$log_dir"/server_*.log)
  if [[ ! -f "${server_logs[0]}" ]]; then
    echo "ERROR: repetition $rep has no server log" >&2
    exit 1
  fi
  server_log="${server_logs[0]}"
  before="$(jit_count "$server_log")"
  set +e
  "$PYTHON_BIN" "$SCRIPT_DIR/verify_bridge.py" \
    --bridge "http://127.0.0.1:${BRIDGE_PORT}" \
    --dump-dir "$log_dir/tts" \
    --out "$log_dir/verify.json" \
    --timeout 180 >"$log_dir/verify.out" 2>&1
  verify_rc=$?
  set -e
  after="$(jit_count "$server_log")"

  metrics="$(
    "$PYTHON_BIN" - "$log_dir" "$verify_rc" "$before" "$after" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
verify_rc = int(sys.argv[2])
before, after = int(sys.argv[3]), int(sys.argv[4])
text = (root / "warmup.out").read_text(errors="replace")
start, end = text.find("{"), text.rfind("}")
warmup = json.loads(text[start : end + 1]) if start >= 0 and end >= start else {}
verify = json.loads((root / "verify.json").read_text()) if (root / "verify.json").exists() else {}
warmup_s = float(warmup.get("elapsed_s") or 0)
warm_voice_s = float(warmup.get("voice_elapsed_s") or 0)
verify_s = float(verify.get("bridge_result", {}).get("elapsed_s") or 0)
passed = (
    warmup.get("ok") is True
    and "<|silent|>" in str(warmup.get("silent_text") or "")
    and 0 < warm_voice_s <= 8
    and verify_rc == 0
    and verify.get("pass") is True
    and 0 < verify_s <= 8
    and after - before == 0
)
print(
    "\t".join(
        [
            str(warmup_s),
            str(warm_voice_s),
            str(verify_s),
            str(after - before),
            "1" if passed else "0",
        ]
    )
)
PY
  )"
  IFS=$'\t' read -r warmup_s warm_voice_s verify_s new_jit pass <<<"$metrics"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$rep" "$warmup_s" "$warm_voice_s" "$verify_s" \
    "$before" "$after" "$new_jit" "$pass" "$log_dir" >>"$RESULTS"

  cleanup
  launcher_pid=""
  log_dir=""
  if [[ "$pass" != "1" ]]; then
    echo "ERROR: repetition $rep failed acceptance; see $RESULTS" >&2
    exit 1
  fi
  sleep 3
done

trap - EXIT
echo "Aligned warmup acceptance passed: $RESULTS"
