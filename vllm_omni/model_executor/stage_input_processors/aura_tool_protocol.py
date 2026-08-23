# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure AURA_v2 tool-protocol helpers shared by API and Stage-1 workers."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

AURA_TOOL_CALL_START = "<tool_call>"
AURA_TOOL_CALL_END = "</tool_call>"
AURA_TOOL_CALL_START_ID = 248058
AURA_TOOL_CALL_END_ID = 248059

# qwen3_5 / AURA_v2 silent id (v1 default remains 151669).
AURA_V2_SILENT_TOKEN_ID = 248070

# Match /workspace/models/AURA_v2/chat_template.jinja generation_prompt.
AURA_ASSISTANT_PREFIX_V1 = "<|im_start|>assistant"
AURA_ASSISTANT_PREFIX_THINK_OPEN = "<|im_start|>assistant\n<think>\n"
AURA_ASSISTANT_PREFIX_THINK_CLOSED = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
AURA_CHINESE_RESPONSE_INSTRUCTION = (
    "\n\n请只用简体中文回答；除数学变量、品牌名及不可翻译的专有名词外，"
    "不要输出英文句子、JSON、代码、LaTeX、论文引用或与问题无关的内容。"
)


def _env_flag_on(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def is_aura_v2_runtime() -> bool:
    """True when Stage-1 is configured for AURA_v2 (qwen3_5) silent token."""

    raw = (os.environ.get("VLLM_AURA_SILENT_TOKEN_ID") or "").strip()
    if not raw:
        return False
    try:
        return int(raw) == AURA_V2_SILENT_TOKEN_ID
    except ValueError:
        return False


def aura_enable_thinking() -> bool:
    """Whether AURA_v2 prompts should open a thinking block.

    Default **False** (align Native Gateway ``enable_thinking=False``).
    Set ``VLLM_AURA_ENABLE_THINKING=1`` to restore open-think prefixes.
    """

    return _env_flag_on("VLLM_AURA_ENABLE_THINKING")


def aura_response_instruction() -> str:
    """Return the optional per-turn response constraint for AURA_v2.

    Native Gateway does not inject a Chinese-only user-turn instruction.
    Keep the opt-in path for experiments; default off to preserve silent
    vision-only turns. When enabled, keep the constraint out of persisted
    history and append it only to the turn being generated.
    """

    if not is_aura_v2_runtime():
        return ""
    if not _env_flag_on("VLLM_AURA_CHINESE_ONLY"):
        return ""
    return AURA_CHINESE_RESPONSE_INSTRUCTION


def with_aura_response_instruction(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy messages and append the v2 response constraint to the last user."""

    instruction = aura_response_instruction()
    if not instruction:
        return messages
    rendered = list(messages)
    for index in range(len(rendered) - 1, -1, -1):
        message = rendered[index]
        if message.get("role") != "user":
            continue
        updated = dict(message)
        content = updated.get("content")
        if isinstance(content, str):
            updated["content"] = content + instruction
        elif isinstance(content, list):
            updated["content"] = [*content, {"type": "text", "text": instruction}]
        else:
            updated["content"] = instruction.lstrip()
        rendered[index] = updated
        return rendered
    return rendered


def aura_assistant_generation_prefix(*, enable_thinking: bool | None = None) -> str:
    """Assistant turn prefix for Stage-1 generation.

    - AURA v1 (default silent id): bare ``<|im_start|>assistant``.
    - AURA_v2 + thinking off: empty closed ``<think></think>`` (Native-aligned).
    - AURA_v2 + thinking on: open ``<think>\\n`` for model continuation.
    """

    if not is_aura_v2_runtime():
        return AURA_ASSISTANT_PREFIX_V1
    thinking = aura_enable_thinking() if enable_thinking is None else bool(enable_thinking)
    if thinking:
        return AURA_ASSISTANT_PREFIX_THINK_OPEN
    return AURA_ASSISTANT_PREFIX_THINK_CLOSED


def has_aura_tool_call_marker(
    text: str | None = None,
    token_ids: Sequence[int] | None = None,
) -> bool:
    """Return whether output must be withheld from TTS as a tool transaction.

    This is deliberately fail-closed: parsing/validation happens in the API
    process, while the Stage-1 worker only needs an unambiguous marker to keep
    raw XML out of Talker.
    """

    if token_ids and AURA_TOOL_CALL_START_ID in token_ids:
        return True
    return isinstance(text, str) and AURA_TOOL_CALL_START in text


def aura_natural_content(text: str) -> str:
    """Remove Qwen3.5 reasoning wrappers before user-visible/TTS handling."""

    if "</think>" in text:
        return text.rsplit("</think>", 1)[-1].lstrip()
    if "<think>" in text:
        # An unclosed reasoning block is not natural assistant content.
        return ""
    return text


def extract_aura_tool_preamble(text: str) -> str:
    """Return spoken prefix that appears before the first tool marker.

    Empty when the turn is a pure tool call, an unclosed think block, or the
    prefix is only XML/whitespace. Never includes ``<tool_call>`` payload.
    """

    if not isinstance(text, str) or not text:
        return ""
    prefix = text.split(AURA_TOOL_CALL_START, 1)[0] if AURA_TOOL_CALL_START in text else text
    return aura_natural_content(prefix).strip()


def render_aura_tool_prompt(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    *,
    enable_thinking: bool | None = None,
) -> str:
    """Render the checkpoint-owned Qwen3.5 tool template.

    Multimodal tensors and UUIDs are intentionally handled by
    ``SessionHistory``; the tokenizer owns only the textual protocol.
    Default ``enable_thinking`` follows ``aura_enable_thinking()`` (off).
    """

    if not tools:
        raise ValueError("AURA tool prompt requires a non-empty tools list")
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_chat_template):
        raise TypeError("AURA tool prompt requires a tokenizer with apply_chat_template")
    thinking = aura_enable_thinking() if enable_thinking is None else bool(enable_thinking)
    rendered = apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=thinking,
    )
    if not isinstance(rendered, str) or not rendered:
        raise ValueError("AURA tool chat template returned an empty prompt")
    return rendered
