# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared AURA text helpers (punctuation-only detection, etc.)."""

from __future__ import annotations

# Same set used by CrossTurnPenalty to skip non-content tokens; extended with
# common CJK filler glyphs (e.g. ﹑) that models emit instead of <|silent|>.
AURA_PUNCT_CHARS = frozenset(
    ".,!?;:，。！？；：、'\"()[]{}""''…—–\n\t\r /-_@#$%^&*+=<>~`|\\（）【】《》﹑·"
)

__all__ = ["AURA_PUNCT_CHARS", "is_punctuation_only_text"]


def is_punctuation_only_text(text: str) -> bool:
    """True when stripped text is empty or only whitespace / AURA punctuation."""
    if not isinstance(text, str):
        return False
    stripped = text.strip()
    if not stripped:
        return True
    return all(ch.isspace() or ch in AURA_PUNCT_CHARS for ch in stripped)
