# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Step-by-step parity logging for Qwen-Image (debugging e.g. GitHub #3256).

#3256 reproduces via **online** serving::

    python -m vllm_omni.entrypoints.cli.main serve Qwen/Qwen-Image-2512 ...

This module writes **one plain-text trace per generation request** when enabled.
Use the **same default path and format** on old vs new vllm-omni branches so you
can ``diff`` two runs after sending one API request on each side.

Environment variables
---------------------

Required to enable::

    export VLLM_OMNI_QWEN_PARITY_LOG=1

Optional::

    export VLLM_OMNI_QWEN_PARITY_LOG_FILE=/path/to/qwen_image_parity.log
        If unset, default is **always** (same on every machine)::

            ~/.vllm_omni/parity/qwen_image_parity.log

    export VLLM_OMNI_QWEN_PARITY_MAX_STEPS=N
        Only log first N denoise steps (default: all).

    export VLLM_OMNI_QWEN_PARITY_APPEND=1
        Append to the log file instead of overwriting at each new request
        (default: overwrite on each new ``prepare_generation_context``).

Tensor lines are stable ``format=v1`` text for line-by-line comparison.
"""

from __future__ import annotations

import os
import sys
import time

import torch

_ENV_ENABLE = "VLLM_OMNI_QWEN_PARITY_LOG"
_ENV_PATH = "VLLM_OMNI_QWEN_PARITY_LOG_FILE"
_ENV_MAX_STEPS = "VLLM_OMNI_QWEN_PARITY_MAX_STEPS"
_ENV_APPEND = "VLLM_OMNI_QWEN_PARITY_APPEND"

# Fixed relative location under $HOME for cross-branch / cross-run diff.
_DEFAULT_REL_PATH = (".vllm_omni", "parity", "qwen_image_parity.log")


def _append_mode_requested() -> bool:
    return os.environ.get(_ENV_APPEND, "").strip().lower() in ("1", "true", "yes", "on")


_FORMAT_TAG = "format=v1 issue_3256_compat"

_first_header_in_session = True
# First write: truncate unless APPEND=1 (then always append to file).
_next_open_truncates = not _append_mode_requested()

# Aligns with one diffuse iteration: predict_noise_maybe_with_cfg (peek N) then scheduler_step (still N), bump after scheduler.
_parity_cfg_denoise_step = 0


def parity_enabled() -> bool:
    return os.environ.get(_ENV_ENABLE, "").strip().lower() in ("1", "true", "yes", "on")


def parity_default_log_path() -> str:
    """Stable default path (override with VLLM_OMNI_QWEN_PARITY_LOG_FILE)."""
    return os.path.join(os.path.expanduser("~"), *_DEFAULT_REL_PATH)


def parity_log_path() -> str:
    p = os.environ.get(_ENV_PATH, "").strip()
    if p:
        return os.path.abspath(os.path.expanduser(p))
    return os.path.abspath(parity_default_log_path())


def _max_steps_limit() -> int | None:
    raw = os.environ.get(_ENV_MAX_STEPS, "").strip()
    if not raw:
        return None
    try:
        n = int(raw)
    except ValueError:
        return None
    return None if n <= 0 else n


def parity_should_log_denoise_step(step_idx: int) -> bool:
    lim = _max_steps_limit()
    if lim is None:
        return True
    return step_idx < lim


def parity_cfg_peek_denoise_step() -> int:
    """Current denoise index for CFGParallelMixin tracing (matches diffuse loop step)."""
    return _parity_cfg_denoise_step


def parity_cfg_bump_denoise_step_after_scheduler() -> None:
    global _parity_cfg_denoise_step
    _parity_cfg_denoise_step += 1


def parity_cfg_reset_denoise_step() -> None:
    global _parity_cfg_denoise_step
    _parity_cfg_denoise_step = 0


def parity_reset_session() -> None:
    """Start a new trace file for the next generation (overwrite unless APPEND=1)."""
    global _first_header_in_session, _next_open_truncates
    if not parity_enabled():
        return
    parity_cfg_reset_denoise_step()
    _first_header_in_session = True
    if not _append_mode_requested():
        _next_open_truncates = True


def _summarize_tensor(name: str, t: torch.Tensor | None) -> str:
    if t is None:
        return f"{name}=None"
    x = t.detach().float().cpu().reshape(-1)
    if x.numel() == 0:
        return f"{name} empty shape={tuple(t.shape)} dtype={t.dtype}"
    n = x.numel()
    head_n = min(8, n)
    head = x[:head_n].tolist()
    return (
        f"{name} shape={tuple(t.shape)} dtype={t.dtype} "
        f"mean={float(x.mean()):.8g} std={float(x.std()):.8g} "
        f"min={float(x.min()):.8g} max={float(x.max()):.8g} "
        f"absmax={float(x.abs().max()):.8g} head{head_n}={head}"
    )


def parity_write(line: str) -> None:
    global _first_header_in_session, _next_open_truncates
    if not parity_enabled():
        return
    path = parity_log_path()
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    mode = "w" if _next_open_truncates else "a"
    _next_open_truncates = False

    with open(path, mode, encoding="utf-8", newline="\n") as f:
        if _first_header_in_session:
            f.write(f"# {_FORMAT_TAG}\n")
            f.write(
                f"# pid={os.getpid()} ts_utc={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
                f"python={sys.version.split()[0]}\n"
            )
            f.write(
                f"# torch={getattr(torch, '__version__', '?')} "
                f"cuda_available={torch.cuda.is_available()}\n"
            )
            _first_header_in_session = False
        f.write(line.rstrip() + "\n")


def parity_msg(msg: str) -> None:
    if not parity_enabled():
        return
    parity_write(msg)


def parity_tensor(stage: str, tensor: torch.Tensor | None) -> None:
    if not parity_enabled():
        return
    parity_write(_summarize_tensor(stage, tensor))


def parity_section(title: str) -> None:
    if not parity_enabled():
        return
    parity_write(f"### {title} ###")


def parity_runtime_snapshot(tag: str = "runtime") -> None:
    """Record runtime/kernel-related flags for step0 divergence triage."""
    if not parity_enabled():
        return
    parity_section(tag)
    parity_msg(
        "tf32 "
        f"cuda_matmul={getattr(torch.backends.cuda.matmul, 'allow_tf32', None)} "
        f"cudnn={getattr(torch.backends.cudnn, 'allow_tf32', None)} "
        f"deterministic_algorithms={torch.are_deterministic_algorithms_enabled()}"
    )
    if torch.cuda.is_available():
        try:
            dev_idx = torch.cuda.current_device()
            dev_name = torch.cuda.get_device_name(dev_idx)
            parity_msg(f"cuda_device index={dev_idx} name={dev_name}")
        except Exception as e:
            parity_msg(f"cuda_device unknown error={e}")
    env_keys = (
        "CUDA_VISIBLE_DEVICES",
        "CUBLAS_WORKSPACE_CONFIG",
        "CUDA_LAUNCH_BLOCKING",
        "NVIDIA_TF32_OVERRIDE",
        "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
        "PYTORCH_CUDA_ALLOC_CONF",
        "HF_HOME",
        "TRANSFORMERS_CACHE",
    )
    for k in env_keys:
        parity_msg(f"env {k}={os.environ.get(k)}")
