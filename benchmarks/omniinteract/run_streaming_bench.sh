#!/usr/bin/env bash
# vllm-omni OmniInteract aura_streaming benchmark (multi-turn defaults).
# Default GPUs: 2,3 (native AURA stack uses 0,1).
#
# Full guide: benchmarks/omniinteract/SETUP_AND_RUN.md
#
# Prerequisites:
#   git checkout aura_streaming_video_with_bench && pip install -e .
#   bash benchmarks/omniinteract/run_aura_server.sh
#   # or: vllm serve /models/AURA --omni --deploy-config vllm_omni/deploy/aura_omni.yaml ...
#
# Native-aligned smoke (same 3 videos / prompt / realtime send as AURA original):
#   NATIVE_ALIGNED=1 bash benchmarks/omniinteract/run_streaming_bench.sh
#
# Legacy quick smoke (16-frame cap, fast send, QA system prompt):
#   STREAMING_MAX_FRAMES=16 STREAMING_SEND_FPS=0 \
#   OMNIINTERACT_STREAMING_SYSTEM_PROMPT_MODE=omniinteract_qa \
#   NUM_PROMPTS=3 bash benchmarks/omniinteract/run_streaming_bench.sh
#
# Default streaming client settings match native AURA bench: native system prompt,
# full video extract (max_frames=0), wall-clock send_fps=2, frame_filter off.
# NATIVE_ALIGNED=1 only pins the 3-video smoke subset (0002,0003,0004).
#
# Outputs (under ./omniinteract_bench by default):
#   omniinteract_streaming_*.json   — full metrics + per_requests
#   bench_report.md                 — native AURA-style summary
#   videos/<id>/bench_report.md     — per-video report + streaming_chunks.json
set -euo pipefail

MODEL="${MODEL:-/models/AURA}"
DATASET_PATH="${DATASET_PATH:-/models/datasets/OmniInteract}"
VLLM_BIN="${VLLM_BIN:-/public/wtk/.venv/bin/vllm}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8666}"
NUM_PROMPTS="${NUM_PROMPTS:-32}"
NUM_WARMUPS="${NUM_WARMUPS:-2}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"
NATIVE_ALIGNED="${NATIVE_ALIGNED:-0}"

# TTS: match native AURA stack (run_omniinteract_bench.sh → tts_service.py)
AURA_NATIVE_ROOT="${AURA_NATIVE_ROOT:-/public/wtk/AURA/AURA}"
OMNIINTERACT_AURA_TTS_TASK_TYPE="${OMNIINTERACT_AURA_TTS_TASK_TYPE:-Base}"
OMNIINTERACT_AURA_TTS_LANGUAGE="${OMNIINTERACT_AURA_TTS_LANGUAGE:-Chinese}"
OMNIINTERACT_AURA_TTS_REF_AUDIO="${OMNIINTERACT_AURA_TTS_REF_AUDIO:-${AURA_NATIVE_ROOT}/shuhan.mp3}"
OMNIINTERACT_AURA_TTS_REF_TEXT="${OMNIINTERACT_AURA_TTS_REF_TEXT:-读书指通过阅读书籍获取知识、交流思想的行为。}"

# Native-aligned streaming defaults (shared by full dataset and smoke3).
STREAMING_MAX_FRAMES="${STREAMING_MAX_FRAMES:-0}"
STREAMING_SEND_FPS="${STREAMING_SEND_FPS:-2}"
STREAMING_AUTO_TRIGGER_MIN_FRAMES="${STREAMING_AUTO_TRIGGER_MIN_FRAMES:-2}"
OMNIINTERACT_STREAMING_SYSTEM_PROMPT_MODE="${OMNIINTERACT_STREAMING_SYSTEM_PROMPT_MODE:-native}"
STREAMING_ENABLE_FRAME_FILTER="${STREAMING_ENABLE_FRAME_FILTER:-0}"

if [[ "${NATIVE_ALIGNED}" == "1" || "${NATIVE_ALIGNED}" == "true" ]]; then
  OMNIINTERACT_SUBSETS="${OMNIINTERACT_SUBSETS:-1q1a}"
  OMNIINTERACT_VIDEO_IDS="${OMNIINTERACT_VIDEO_IDS:-0002,0003,0004}"
  NUM_PROMPTS=3
  NUM_WARMUPS=0
  OMNIINTERACT_WARMUP_VIDEO_ID="${OMNIINTERACT_WARMUP_VIDEO_ID:-0001}"
  DISABLE_SHUFFLE_FLAG=(--disable-shuffle)
  RESULT_TAG="native_smoke3"
else
  OMNIINTERACT_SUBSETS="${OMNIINTERACT_SUBSETS:-1q1a,1q1a_math,1qna}"
  OMNIINTERACT_VIDEO_IDS="${OMNIINTERACT_VIDEO_IDS:-}"
  DISABLE_SHUFFLE_FLAG=()
  if [[ -n "${OMNIINTERACT_VIDEO_IDS}" ]]; then
    DISABLE_SHUFFLE_FLAG=(--disable-shuffle)
    RESULT_TAG="ids_${OMNIINTERACT_VIDEO_IDS//,/_}"
  else
    RESULT_TAG="c${MAX_CONCURRENCY}_${NUM_PROMPTS}_f${STREAMING_MAX_FRAMES}_t${STREAMING_AUTO_TRIGGER_MIN_FRAMES}"
  fi
fi

CROSS_TURN_PENALTY="${CROSS_TURN_PENALTY:-1}"
CROSS_TURN_LOOKBACK="${CROSS_TURN_LOOKBACK:-10}"
READY_TIMEOUT_SEC="${READY_TIMEOUT_SEC:-900}"
OMNIINTERACT_EVAL_FLAG=()
if [[ "${OMNIINTERACT_EVAL:-1}" == "1" || "${OMNIINTERACT_EVAL:-}" == "true" ]]; then
  OMNIINTERACT_EVAL_FLAG+=(--omniinteract-eval)
fi

VIDEO_IDS_FLAG=()
if [[ -n "${OMNIINTERACT_VIDEO_IDS}" ]]; then
  VIDEO_IDS_FLAG=(--omniinteract-video-ids "${OMNIINTERACT_VIDEO_IDS}")
fi

FRAME_FILTER_FLAG=()
if [[ "${STREAMING_ENABLE_FRAME_FILTER}" == "1" || "${STREAMING_ENABLE_FRAME_FILTER}" == "true" ]]; then
  FRAME_FILTER_FLAG=(--omniinteract-streaming-enable-frame-filter)
fi

echo "Waiting for AURA server at ${HOST}:${PORT} (timeout: ${READY_TIMEOUT_SEC}s)..."
deadline=$((SECONDS + READY_TIMEOUT_SEC))
until ${PYTHON:-python3} - "${HOST}" "${PORT}" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
sock = socket.socket()
sock.settimeout(2)
try:
    sock.connect((host, port))
except OSError:
    raise SystemExit(1)
finally:
    sock.close()
PY
do
  if (( SECONDS >= deadline )); then
    echo "AURA server is not ready at ${HOST}:${PORT} after ${READY_TIMEOUT_SEC}s." >&2
    exit 1
  fi
  sleep 5
done
echo "AURA server is accepting connections."

echo "OmniInteract streaming bench: subsets=${OMNIINTERACT_SUBSETS} video_ids=${OMNIINTERACT_VIDEO_IDS:-all} native_aligned=${NATIVE_ALIGNED}"
echo "TTS (native-aligned): task=${OMNIINTERACT_AURA_TTS_TASK_TYPE} lang=${OMNIINTERACT_AURA_TTS_LANGUAGE} ref=${OMNIINTERACT_AURA_TTS_REF_AUDIO}"

_run_streaming_bench() {
  local _num_prompts="$1"
  local _result_filename="$2"
  local _video_ids="${3:-}"
  local -a _video_ids_flag=()
  if [[ -n "${_video_ids}" ]]; then
    _video_ids_flag=(--omniinteract-video-ids "${_video_ids}")
  elif [[ -n "${OMNIINTERACT_VIDEO_IDS}" ]]; then
    _video_ids_flag=(--omniinteract-video-ids "${OMNIINTERACT_VIDEO_IDS}")
  fi

  "${VLLM_BIN}" bench serve --omni \
    --host "${HOST}" \
    --port "${PORT}" \
    --trust-remote-code \
    --backend openai-video-stream \
    --endpoint /v1/video/chat/stream \
    --model "${MODEL}" \
    --no-oversample \
    --dataset-name omniinteract \
    --dataset-path "${DATASET_PATH}" \
    --max-concurrency "${MAX_CONCURRENCY}" \
    --num-warmups 0 \
    --num-prompts "${_num_prompts}" \
    --omniinteract-subsets "${OMNIINTERACT_SUBSETS}" \
    "${_video_ids_flag[@]}" \
    "${DISABLE_SHUFFLE_FLAG[@]}" \
    --omniinteract-input-mode aura_streaming \
    --omniinteract-streaming-sample-fps 2 \
    --omniinteract-streaming-send-fps "${STREAMING_SEND_FPS}" \
    --omniinteract-streaming-max-frames "${STREAMING_MAX_FRAMES}" \
    --omniinteract-streaming-auto-trigger-min-frames "${STREAMING_AUTO_TRIGGER_MIN_FRAMES}" \
    --omniinteract-streaming-system-prompt-mode "${OMNIINTERACT_STREAMING_SYSTEM_PROMPT_MODE}" \
    "${FRAME_FILTER_FLAG[@]}" \
    --omniinteract-cross-turn-penalty "${CROSS_TURN_PENALTY}" \
    --omniinteract-cross-turn-lookback "${CROSS_TURN_LOOKBACK}" \
    --omniinteract-aura-tts-task-type "${OMNIINTERACT_AURA_TTS_TASK_TYPE}" \
    --omniinteract-aura-tts-language "${OMNIINTERACT_AURA_TTS_LANGUAGE}" \
    --omniinteract-aura-tts-ref-audio "${OMNIINTERACT_AURA_TTS_REF_AUDIO}" \
    --omniinteract-aura-tts-ref-text "${OMNIINTERACT_AURA_TTS_REF_TEXT}" \
    --percentile-metrics ttft,tpot,itl,e2el,audio_ttfp,audio_rtf,audio_duration,ttfc,tpoc,icl \
    --save-result \
    --save-detailed \
    --print-stage \
    --result-dir ./omniinteract_bench \
    --result-filename "${_result_filename}" \
    "${OMNIINTERACT_EVAL_FLAG[@]}"
}

if [[ "${NATIVE_ALIGNED}" == "1" || "${NATIVE_ALIGNED}" == "true" ]] \
  && [[ "${OMNIINTERACT_SKIP_WARMUP:-0}" != "1" ]] \
  && [[ -n "${OMNIINTERACT_WARMUP_VIDEO_ID:-}" ]]; then
  echo ""
  echo "=== Warmup: video ${OMNIINTERACT_WARMUP_VIDEO_ID} (excluded from scored JSON) ==="
  _run_streaming_bench 1 "omniinteract_streaming_${RESULT_TAG}_warmup.json" "${OMNIINTERACT_WARMUP_VIDEO_ID}"
fi

echo ""
echo "=== Scored run: videos ${OMNIINTERACT_VIDEO_IDS:-all} ==="
_run_streaming_bench "${NUM_PROMPTS}" "omniinteract_streaming_${RESULT_TAG}.json"
