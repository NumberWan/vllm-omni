#!/usr/bin/env bash
# Install vLLM, this vLLM-Omni checkout, and the CUDA-matched FlashInfer
# JIT cache for AURA. Official order is preserved: vLLM first, then omni.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BOOTSTRAP_PYTHON="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$ROOT/.venv}"
CHECK_ONLY="${CHECK_ONLY:-0}"
VLLM_VERSION="${VLLM_VERSION:-0.23.0}"
# Parent shells (IDE, prior checkouts) may export PYTHONPATH into another
# site-packages tree; that falsely satisfies FlashInfer checks and defeats
# an isolated VENV_DIR install. Always install/verify against the target venv.
export PYTHONPATH=

if ! command -v "$BOOTSTRAP_PYTHON" >/dev/null 2>&1; then
  echo "ERROR: Python was not found: $BOOTSTRAP_PYTHON" >&2
  exit 1
fi

"$BOOTSTRAP_PYTHON" - <<'PY'
import sys

if not ((3, 10) <= sys.version_info[:2] <= (3, 13)):
    raise SystemExit(
        f"ERROR: AURA requires Python 3.10-3.13; found "
        f"{sys.version_info.major}.{sys.version_info.minor}."
    )
PY

if [[ "$CHECK_ONLY" != "1" ]]; then
  if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo "Creating virtual environment: $VENV_DIR"
    "$BOOTSTRAP_PYTHON" -m venv "$VENV_DIR"
  fi

  echo "Installing vLLM-Omni from: $ROOT"
  "$VENV_DIR/bin/python" -m pip install --upgrade pip
  # Official order: install vLLM first, then vLLM-Omni.
  # vllm is not declared in omni requirements, so ``pip install -e .`` alone
  # will not pull it in.
  echo "Installing vLLM ${VLLM_VERSION}"
  if "$VENV_DIR/bin/python" - <<PY 2>/dev/null
import importlib.metadata as md
raise SystemExit(0 if md.version("vllm") == "${VLLM_VERSION}" else 1)
PY
  then
    echo "vLLM ${VLLM_VERSION} already installed; skipping"
  elif command -v uv >/dev/null 2>&1; then
    uv pip install --python "$VENV_DIR/bin/python" \
      "vllm==${VLLM_VERSION}" --torch-backend=auto
  else
    "$VENV_DIR/bin/python" -m pip install "vllm==${VLLM_VERSION}"
  fi
  "$VENV_DIR/bin/python" -m pip install -e "$ROOT"
elif [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "ERROR: CHECK_ONLY=1 requires an existing environment at $VENV_DIR" >&2
  exit 1
fi

PYTHON="$VENV_DIR/bin/python"

cuda_tag="$(
  "$PYTHON" - <<'PY'
import torch

cuda = torch.version.cuda
supported = {
    "12.6": "cu126",
    "12.8": "cu128",
    "12.9": "cu129",
    "13.0": "cu130",
    "13.1": "cu131",
}
if not cuda:
    raise SystemExit("ERROR: the installed PyTorch build has no CUDA support.")
try:
    print(supported[cuda])
except KeyError:
    choices = ", ".join(supported)
    raise SystemExit(
        f"ERROR: FlashInfer JIT-cache installation does not support PyTorch CUDA "
        f"{cuda}; supported versions: {choices}."
    ) from None
PY
)"

flashinfer_version="$(
  "$PYTHON" - <<'PY'
import importlib.metadata as md

try:
    print(md.version("flashinfer-python").split("+", 1)[0])
except md.PackageNotFoundError:
    raise SystemExit(
        "ERROR: flashinfer-python was not installed with vLLM. "
        "Check the pip output above."
    ) from None
PY
)"

jit_version="$(
  "$PYTHON" - <<'PY'
import importlib.metadata as md

try:
    print(md.version("flashinfer-jit-cache"))
except md.PackageNotFoundError:
    print("")
PY
)"
jit_base="${jit_version%%+*}"

if [[ "$CHECK_ONLY" != "1" ]] && [[ "$jit_base" != "$flashinfer_version" ]]; then
  echo "Installing FlashInfer JIT cache ${flashinfer_version}+${cuda_tag}"
  "$PYTHON" -m pip install --upgrade \
    "flashinfer-jit-cache==${flashinfer_version}+${cuda_tag}" \
    --index-url "https://flashinfer.ai/whl/${cuda_tag}"
elif [[ -n "$jit_version" ]]; then
  echo "Using installed FlashInfer JIT cache: $jit_version"
fi

"$PYTHON" - <<'PY'
import importlib.metadata as md

import flashinfer  # noqa: F401

runtime = md.version("flashinfer-python").split("+", 1)[0]
try:
    jit_full = md.version("flashinfer-jit-cache")
except md.PackageNotFoundError:
    raise SystemExit(
        "ERROR: flashinfer-jit-cache is missing. Re-run this installer with network access."
    ) from None
jit_base = jit_full.split("+", 1)[0]
if runtime != jit_base:
    raise SystemExit(
        f"ERROR: FlashInfer mismatch: flashinfer-python={runtime}, "
        f"flashinfer-jit-cache={jit_full}."
    )
print(f"FlashInfer ready: runtime={runtime}, jit-cache={jit_full}")
PY

echo
echo "AURA installation is ready."
echo "Start with:"
echo "  CUDA_VISIBLE_DEVICES=0,1 bash scripts/start_aura_omni.sh"
