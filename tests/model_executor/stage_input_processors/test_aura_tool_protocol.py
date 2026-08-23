# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

import pytest
from transformers import AutoTokenizer
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser.qwen3 import Qwen3Parser

from vllm_omni.entrypoints.openai.aura_tool_executor import parse_aura_tool_output
from vllm_omni.model_executor.stage_input_processors.aura_session_history import (
    SessionHistory,
)
from vllm_omni.model_executor.stage_input_processors.aura_tool_protocol import (
    AURA_ASSISTANT_PREFIX_THINK_CLOSED,
    AURA_ASSISTANT_PREFIX_THINK_OPEN,
    AURA_ASSISTANT_PREFIX_V1,
    AURA_CHINESE_RESPONSE_INSTRUCTION,
    AURA_TOOL_CALL_END_ID,
    AURA_TOOL_CALL_START_ID,
    aura_assistant_generation_prefix,
    aura_natural_content,
    aura_response_instruction,
    extract_aura_tool_preamble,
    has_aura_tool_call_marker,
    is_aura_v2_runtime,
    with_aura_response_instruction,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

MODEL_PATH = Path("/workspace/models/AURA_v2")
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "mock_echo",
            "description": "Return the supplied text.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": False,
            },
        },
    }
]


@pytest.fixture(scope="module")
def aura_v2_tokenizer():
    if not MODEL_PATH.is_dir():
        pytest.skip("AURA_v2 checkpoint is not available")
    return AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)


def test_aura_v2_tool_template_golden(aura_v2_tokenizer):
    history = SessionHistory(system_prompt="You are precise.")
    rendered = history.preview_vllm_inputs(
        "echo hello",
        tokenizer=aura_v2_tokenizer,
        tools=TOOLS,
    )["prompt"]

    assert rendered.startswith("<|im_start|>system\n# Tools\n")
    assert '"name": "mock_echo"' in rendered
    assert "<tools>\n" in rendered and "\n</tools>" in rendered
    assert "<|im_start|>user\necho hello<|im_end|>\n" in rendered
    assert rendered.endswith(AURA_ASSISTANT_PREFIX_THINK_CLOSED)


def test_aura_v2_tool_response_and_multi_call_template(aura_v2_tokenizer):
    history = SessionHistory(system_prompt="You are precise.")
    transient = [
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "two calls",
            "tool_calls": [
                {"id": "call-1", "function": {"name": "mock_echo", "arguments": {"text": "one"}}},
                {"id": "call-2", "function": {"name": "mock_echo", "arguments": {"text": "two"}}},
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": '{"text":"one"}'},
        {"role": "tool", "tool_call_id": "call-2", "content": '{"text":"two"}'},
    ]
    rendered = history.preview_vllm_inputs(
        "echo twice",
        tokenizer=aura_v2_tokenizer,
        tools=TOOLS,
        transient_messages=transient,
    )["prompt"]

    assert rendered.count("<tool_call>") == 4  # two protocol mentions plus two calls
    assert rendered.count("<tool_response>") == 2
    assert "<parameter=text>\none\n</parameter>" in rendered
    assert rendered.endswith(AURA_ASSISTANT_PREFIX_THINK_CLOSED)


def test_aura_v2_tool_template_open_think_when_env_enabled(aura_v2_tokenizer, monkeypatch):
    monkeypatch.setenv("VLLM_AURA_ENABLE_THINKING", "1")
    history = SessionHistory(system_prompt="You are precise.")
    rendered = history.preview_vllm_inputs(
        "echo hello",
        tokenizer=aura_v2_tokenizer,
        tools=TOOLS,
    )["prompt"]
    assert rendered.endswith(AURA_ASSISTANT_PREFIX_THINK_OPEN)


def test_qwen3_parser_round_trip_and_marker_ids(aura_v2_tokenizer):
    request = ChatCompletionRequest(
        model="AURA_v2",
        messages=[{"role": "user", "content": "echo hello"}],
        tools=TOOLS,
    )
    raw = (
        "<think>use echo</think>\n"
        "<tool_call>\n<function=mock_echo>\n"
        "<parameter=text>\nhello\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    parser = Qwen3Parser(aura_v2_tokenizer, request.tools)
    reasoning, content, calls = parser.parse(
        raw,
        request,
        enable_auto_tools=True,
        model_output_token_ids=aura_v2_tokenizer.encode(raw, add_special_tokens=False),
    )

    assert reasoning == "use echo"
    assert content is None
    assert calls and calls[0].name == "mock_echo"
    assert calls[0].arguments == '{"text": "hello"}'
    assert aura_v2_tokenizer.convert_tokens_to_ids("<tool_call>") == AURA_TOOL_CALL_START_ID
    assert aura_v2_tokenizer.convert_tokens_to_ids("</tool_call>") == AURA_TOOL_CALL_END_ID
    assert has_aura_tool_call_marker(token_ids=[1, AURA_TOOL_CALL_START_ID])
    assert has_aura_tool_call_marker(text="<tool_call>\nmalformed")


def test_qwen3_parser_separates_natural_content_and_rejects_malformed_xml(aura_v2_tokenizer):
    natural = parse_aura_tool_output(
        aura_v2_tokenizer,
        "<think>private reasoning</think>\n公開回答",
        request_id="natural",
        tool_schemas=TOOLS,
    )
    malformed = parse_aura_tool_output(
        aura_v2_tokenizer,
        "<think>reasoning</think><tool_call><function=mock_echo>",
        request_id="malformed",
        tool_schemas=TOOLS,
    )

    assert natural.reasoning == "private reasoning"
    assert natural.content == "公開回答"
    assert natural.calls == []
    assert malformed.error == "malformed_tool_xml"
    assert aura_natural_content("<think>hidden</think>\nspoken") == "spoken"
    assert aura_natural_content("<think>unfinished") == ""
    assert extract_aura_tool_preamble("我先查一下。<tool_call><function=x></function></tool_call>") == "我先查一下。"
    assert extract_aura_tool_preamble("<think>private</think>\n稍等。<tool_call></tool_call>") == "稍等。"
    assert extract_aura_tool_preamble("<tool_call><function=x></function></tool_call>") == ""
    assert extract_aura_tool_preamble("<think>unfinished<tool_call></tool_call>") == ""


def test_v2_generation_prefix_requires_silent_token_env(monkeypatch):
    monkeypatch.delenv("VLLM_AURA_SILENT_TOKEN_ID", raising=False)
    assert is_aura_v2_runtime() is False
    assert aura_assistant_generation_prefix() == AURA_ASSISTANT_PREFIX_V1
    monkeypatch.setenv("VLLM_AURA_SILENT_TOKEN_ID", "248070")
    assert is_aura_v2_runtime() is True
    assert aura_assistant_generation_prefix() == AURA_ASSISTANT_PREFIX_THINK_CLOSED


def test_v2_response_instruction_is_user_turn_local(monkeypatch):
    original = [
        {"role": "system", "content": "system"},
        {
            "role": "user",
            "content": [
                {"type": "video", "video": "pixels"},
                {"type": "text", "text": "Solve this question."},
            ],
        },
    ]
    monkeypatch.setenv("VLLM_AURA_SILENT_TOKEN_ID", "248070")

    rendered = with_aura_response_instruction(original)

    assert aura_response_instruction() == AURA_CHINESE_RESPONSE_INSTRUCTION
    assert rendered is not original
    assert rendered[-1]["content"][-1]["text"] == AURA_CHINESE_RESPONSE_INSTRUCTION
    assert original[-1]["content"][-1]["text"] == "Solve this question."

    monkeypatch.setenv("VLLM_AURA_CHINESE_ONLY", "0")
    assert with_aura_response_instruction(original) is original
