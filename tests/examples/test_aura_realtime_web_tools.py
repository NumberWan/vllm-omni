# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
import json

import pytest

from examples.online_serving.aura_omni.realtime_web import tools
from examples.online_serving.realtime_web import server

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _run(executor: tools.AuraToolExecutor, name: str, arguments: str) -> dict:
    return json.loads(asyncio.run(executor.run(name, arguments)))


def test_realtime_tools_use_the_flat_session_shape() -> None:
    names = [tool["name"] for tool in tools.AuraToolExecutor().realtime_tools]
    assert names == [
        "calculator",
        "get_current_datetime",
        "get_city_weather",
        "convert_currency",
        "DeepSeek",
        "get_current_location",
        "WebSearch",
    ]
    tool = tools.AuraToolExecutor().realtime_tools[0]
    assert tool["type"] == "function"
    assert tool["parameters"]["required"] == ["expression"]


def test_calculator_returns_the_result_and_rejects_code() -> None:
    executor = tools.AuraToolExecutor()
    assert _run(executor, "calculator", '{"expression": "17 * 23"}') == {
        "ok": True,
        "result": {"expression": "17 * 23", "result": 391, "summary": "17 * 23 = 391"},
    }
    assert _run(executor, "calculator", '{"expression": "__import__(\'os\')"}')["error"]["code"] == (
        "calculator_unsupported_function"
    )


def test_bad_calls_become_error_outputs() -> None:
    executor = tools.AuraToolExecutor()
    assert _run(executor, "rm_rf", "{}")["error"]["code"] == "unknown_tool"
    assert _run(executor, "calculator", "not json")["error"]["code"] == "invalid_tool_arguments"
    assert _run(executor, "calculator", '{"expr": "1"}')["error"]["code"] == "invalid_tool_arguments"


def test_weather_aliases_map_to_the_registered_tool() -> None:
    assert tools._canonical_tool_name("get_weather") == "get_city_weather"
    assert tools._canonical_tool_name("Get City Weather") == "get_city_weather"
    assert tools._canonical_tool_name("get_current_location") == "get_current_location"


def test_optional_tools_need_their_env_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_AURA_TOOL_DDG", "1")
    names = [tool["name"] for tool in tools.AuraToolExecutor().realtime_tools]
    assert "duckduckgo_search" in names
    assert "WebFetch" not in names


def test_webfetch_blocks_private_hosts() -> None:
    with pytest.raises(ValueError, match="webfetch_ssrf_blocked"):
        tools._check_ssrf("http://127.0.0.1:8091/")


class _Tools:
    realtime_tools = [{"type": "function", "name": "calculator"}]

    async def run(self, name: str, arguments: str) -> str:
        return json.dumps({"name": name, "arguments": arguments})


def test_session_update_gets_tools_only_when_it_has_none() -> None:
    update = json.dumps({"type": "session.update", "session": {"instructions": "x"}})
    assert json.loads(server._with_session_tools(update, _Tools()))["session"] == {
        "instructions": "x",
        "tools": _Tools.realtime_tools,
    }
    own = json.dumps({"type": "session.update", "session": {"tools": []}})
    assert server._with_session_tools(own, _Tools()) == own
    append = json.dumps({"type": "input_audio_buffer.append", "audio": "AA"})
    assert server._with_session_tools(append, _Tools()) == append


def test_function_call_is_answered_then_followed_up() -> None:
    sent: list[dict] = []

    class _Backend:
        async def send(self, message: str) -> None:
            sent.append(json.loads(message))

    event = {
        "type": "response.output_item.done",
        "item": {
            "type": "function_call",
            "call_id": "call_1",
            "name": "calculator",
            "arguments": '{"expression": "1+1"}',
        },
    }
    call = server._finished_function_call(event)
    assert call is not None
    assert server._finished_function_call({"type": "response.function_call_arguments.done", "call_id": "x"}) is None
    assert server._finished_function_call({"type": "response.output_item.done", "item": {"type": "message"}}) is None
    asyncio.run(server._answer_function_call(_Backend(), _Tools(), call))
    assert [message["type"] for message in sent] == ["conversation.item.create", "response.create"]
    sent.clear()
    asyncio.run(server._answer_function_call(_Backend(), _Tools(), call, follow_up=False))
    assert [message["type"] for message in sent] == ["conversation.item.create"]
    assert server._is_user_commit('{"type":"input_audio_buffer.commit"}')


def test_follow_up_waits_until_the_tool_response_is_done() -> None:
    sent: list[dict] = []

    class _Backend:
        async def send(self, message: str) -> None:
            sent.append(json.loads(message))

    call = {
        "type": "function_call",
        "call_id": "call_1",
        "name": "calculator",
        "arguments": '{"expression": "1+1"}',
    }
    state = {"count": 0, "gen": 0, "pending_gen": 0, "awaiting_done": False, "ready": False}
    server._hold_follow_up_for_response_done(state)
    asyncio.run(server._answer_function_call(_Backend(), _Tools(), call, follow_state=state))
    assert [message["type"] for message in sent] == ["conversation.item.create"]
    assert state["ready"] is True
    assert server._take_deferred_follow_up(state) is True
    assert server._take_deferred_follow_up(state) is False

    state["ready"] = True
    state["pending_gen"] = 0
    state["gen"] = 1
    assert server._take_deferred_follow_up(state) is False

    class _CommitDuringTool(_Tools):
        async def run(self, name: str, arguments: str) -> str:
            state["gen"] += 1
            state["awaiting_done"] = False
            state["ready"] = False
            return await super().run(name, arguments)

    state.update({"awaiting_done": True, "ready": False, "gen": 0, "pending_gen": 0})
    sent.clear()
    asyncio.run(server._answer_function_call(_Backend(), _CommitDuringTool(), call, follow_state=state))
    assert [message["type"] for message in sent] == ["conversation.item.create"]
    assert state["ready"] is False

    state.update({"awaiting_done": False, "ready": False, "gen": 2, "pending_gen": 0})
    sent.clear()
    asyncio.run(server._answer_function_call(_Backend(), _Tools(), call, follow_state=state))
    assert [message["type"] for message in sent] == ["conversation.item.create", "response.create"]
    assert not server._is_user_commit('{"type":"input_audio_buffer.append"}')
    item = sent[0]["item"]
    assert item["type"] == "function_call_output"
    assert item["call_id"] == "call_1"
    assert json.loads(item["output"]) == {"name": "calculator", "arguments": '{"expression": "1+1"}'}
