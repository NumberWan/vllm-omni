#!/usr/bin/env bash
# Launch the MiniCPM-style AURA browser demo (UI + same-origin WS proxy).
# The AURA server must already be running (e.g. scripts/start_aura_omni.sh).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

MODEL="${MODEL:-aurateam/AURA}"
WS_BACKEND="${WS_BACKEND:-ws://127.0.0.1:8666}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-7862}"
VIDEO_FPS="${VIDEO_FPS:-2.0}"
TTS_TASK_TYPE="${TTS_TASK_TYPE:-CustomVoice}"
TTS_LANGUAGE="${TTS_LANGUAGE:-Chinese}"
TTS_SPEAKER="${TTS_SPEAKER:-Dylan}"
TTS_INSTRUCT="${TTS_INSTRUCT:-}"
TOOL_MODE="${TOOL_MODE:-none}"
PUBLIC_STREAM_URL="${PUBLIC_STREAM_URL:-}"
WARMUP_AUDIO="${WARMUP_AUDIO:-}"
SKIP_WARMUP="${SKIP_WARMUP:-0}"
PYTHON_BIN="${PYTHON_BIN:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --ws-backend) WS_BACKEND="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --video-fps) VIDEO_FPS="$2"; shift 2 ;;
    --tts-task-type) TTS_TASK_TYPE="$2"; shift 2 ;;
    --tts-language) TTS_LANGUAGE="$2"; shift 2 ;;
    --tts-speaker) TTS_SPEAKER="$2"; shift 2 ;;
    --tts-instruct) TTS_INSTRUCT="$2"; shift 2 ;;
    --tool-mode) TOOL_MODE="$2"; shift 2 ;;
    --public-stream-url) PUBLIC_STREAM_URL="$2"; shift 2 ;;
    --warmup-audio) WARMUP_AUDIO="$2"; shift 2 ;;
    --skip-warmup) SKIP_WARMUP=1; shift ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --help)
      cat <<EOF
Usage: $0 [options]

  --model NAME                 Served model name (default: aurateam/AURA)
  --ws-backend URL             Backend AURA WebSocket base (default: ws://127.0.0.1:8666)
  --host ADDR                  Demo bind host (default: 0.0.0.0)
  --port PORT                  Demo HTTP port (default: 7862)
  --video-fps FPS              Camera send FPS / session.config video_fps (default: 2.0)
  --tts-task-type TYPE         CustomVoice|Base (default: CustomVoice)
  --tts-language LANG          CustomVoice language (default: Chinese)
  --tts-speaker NAME           CustomVoice speaker (default: Dylan)
  --tts-instruct TEXT          Style instruction (default: clean/brisk, no coughs)
  --tool-mode MODE             none|auto (default: none)
  --public-stream-url URL      Browser-visible WS URL when reverse-proxying WebSockets
  --warmup-audio PATH          Speech WAV for startup warmup (default: bundled asset)
  --skip-warmup                Start UI without warming the AURA pipeline
  --python PATH                Python interpreter (default: repo .venv or python3)

Start the AURA server first:
  CUDA_VISIBLE_DEVICES=0,1 bash scripts/start_aura_omni.sh
EOF
      exit 0
      ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
  else
    PYTHON_BIN="$(command -v python3)"
  fi
fi

cd "$REPO_ROOT"
CMD=(
  "$PYTHON_BIN" -m examples.online_serving.aura_omni.minicpm_style_web_demo
  --host "$HOST"
  --port "$PORT"
  --ws-backend "$WS_BACKEND"
  --model "$MODEL"
  --video-fps "$VIDEO_FPS"
  --tts-task-type "$TTS_TASK_TYPE"
  --tts-language "$TTS_LANGUAGE"
  --tts-speaker "$TTS_SPEAKER"
  --tool-mode "$TOOL_MODE"
)
if [[ -n "$TTS_INSTRUCT" ]]; then
  CMD+=(--tts-instruct "$TTS_INSTRUCT")
fi
if [[ -n "$PUBLIC_STREAM_URL" ]]; then
  CMD+=(--public-stream-url "$PUBLIC_STREAM_URL")
fi
if [[ -n "$WARMUP_AUDIO" ]]; then
  CMD+=(--warmup-audio "$WARMUP_AUDIO")
fi
if [[ "$SKIP_WARMUP" == "1" ]]; then
  CMD+=(--skip-warmup)
fi

echo "AURA MiniCPM-style web demo"
echo "  backend: ${WS_BACKEND}/v1/video/chat/stream"
echo "  speaker: ${TTS_SPEAKER} (${TTS_LANGUAGE})"
echo "  note:    Hold to talk; release submits voice"
echo
exec "${CMD[@]}"
