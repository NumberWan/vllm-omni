# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AURA video-only turn helpers (kept import-light to avoid cycles)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

AURA_SKIP_ASR_KEY = "aura_skip_asr"


def _first_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, list):
        value = value[0] if value else default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if value is None:
        return default
    return bool(value)


def should_skip_aura_asr(prompt: Any) -> bool:
    """Return True when the AURA streaming handler marked a video-only turn."""
    if not isinstance(prompt, dict):
        return False
    additional_info = prompt.get("additional_information")
    if not isinstance(additional_info, dict):
        return False
    return _first_bool(additional_info.get(AURA_SKIP_ASR_KEY), default=False)


def make_mock_asr_source_output(request_id: str, text: str = "", *, finished: bool = True) -> Any:
    """Synthetic stage-0 output for video-only turns that bypass ASR."""
    output = SimpleNamespace(
        text=text,
        cumulative_text=text,
        cumulative_token_ids=[],
        multimodal_output={},
        finished=finished,
    )
    return SimpleNamespace(
        request_id=request_id,
        outputs=[output],
        finished=finished,
    )
