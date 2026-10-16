# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import json
from types import SimpleNamespace

import numpy as np
import pytest

from vllm_omni.engine.duplex.config import DuplexSessionConfig
from vllm_omni.engine.duplex.contracts import DuplexFence, duplex_resource_request_id
from vllm_omni.engine.duplex.plugin import DuplexRuntimeConfigError
from vllm_omni.engine.duplex.realtime_events import prepared_function_call_output
from vllm_omni.engine.duplex.session.model_channel import (
    _INVALID_FUNCTION_CALL,
    _coerce_function_call,
    _recognized_function_call,
)
from vllm_omni.model_executor.models.qwen3_omni.duplex.plugin import Qwen3OmniDuplexPlugin, QwenDataPlane
from vllm_omni.model_executor.models.qwen3_omni.duplex.tools_utils import ThinkerToolCursor

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_CALL = '<tool_call>{"name": "get_current_time", "arguments": {}}</tool_call>'


def test_partial_tool_text_is_not_a_call_or_speech():
    cursor = ThinkerToolCursor()
    speech, call = cursor.absorb('<tool_call>{"name": "get_current')
    assert speech == ""
    assert call is None
    assert cursor.holding_open_call()


def test_complete_call_is_recognized_once():
    cursor = ThinkerToolCursor()
    speech, call = cursor.absorb(f"The time is {_CALL}")
    assert speech == "The time is "
    assert call is not None
    assert call["name"] == "get_current_time"
    assert json.loads(call["arguments"]) == {}
    # A longer snapshot of the same span is not a second call.
    again_speech, again_call = cursor.absorb(cursor.text + " thanks")
    assert again_call is None
    assert again_speech == " thanks"
    assert "<tool_call>" not in again_speech
    # A later identical tag is a new span, so it is a second call.
    second_speech, second_call = cursor.absorb(_CALL)
    assert second_speech == ""
    assert second_call is not None
    assert second_call["name"] == "get_current_time"
    trailing_speech, trailing_call = cursor.absorb(" thanks")
    assert trailing_speech == " thanks"
    assert trailing_call is None


def test_unclosed_call_is_dropped_at_end_of_turn():
    cursor = ThinkerToolCursor()
    cursor.absorb("Hello <tool_call>")
    assert cursor.finish() == ""
    assert cursor.tool_turn


def test_delta_chunks_are_appended_even_when_they_share_a_prefix():
    cursor = ThinkerToolCursor()
    for chunk in [" very", " very", " good"]:
        speech, call = cursor.absorb(chunk)
        assert call is None
        assert speech == chunk
    assert cursor.text == " very very good"
    grown = ThinkerToolCursor()
    grown.absorb(" very")
    grown.absorb(" very good")
    assert grown.text == " very good"


def _output(text: str, *, request_id: str = "req", stage_id: int = 0, finished: bool = False, audio: bool = False):
    completion = SimpleNamespace(text=text, multimodal_output={})
    mm = {"audio": np.ones(16, dtype=np.float32), "sr": 16000} if audio else {}
    return {
        "data_plane_outputs": [
            SimpleNamespace(
                request_id=request_id,
                stage_id=stage_id,
                finished=finished,
                outputs=(completion,),
                multimodal_output=mm,
            )
        ]
    }


def test_data_plane_holds_markup_and_emits_one_call():
    plane = QwenDataPlane(lambda *args: "AAAA")
    context = {"modalities": ("text", "audio"), "response_format": "pcm16", "turn_id": 1}
    first = list(plane.project(_output(f"Hi {_CALL}", audio=True), context=context))
    assert [event.get("text") for event in first] == ["Hi ", ""]
    assert first[0]["audio_data"] == ""
    assert first[1]["qwen_function_call"]["name"] == "get_current_time"
    assert "qwen_function_call" not in first[0]
    # Later deltas after the call are not a second call.
    second = list(plane.project(_output(" ok", audio=True), context=context))
    assert all("qwen_function_call" not in event for event in second)
    # Preface text existed, so later-stage audio is kept for that spoken preface.
    audio = list(plane.project(_output("", stage_id=2, audio=True), context=context))
    assert audio and audio[0]["audio_data"] == "AAAA"


def test_stage2_keeps_audio_after_a_stage0_tool_call():
    plane = QwenDataPlane(lambda *args: "AAAA")
    context = {"modalities": ("text", "audio"), "response_format": "pcm16", "turn_id": 1}
    list(plane.project(_output(_CALL, audio=True), context=context))
    audio = list(plane.project(_output("", stage_id=2, audio=True), context=context))
    assert audio and audio[0]["audio_data"] == "AAAA"


def test_plain_text_still_passes_through():
    plane = QwenDataPlane(lambda *args: "AAAA")
    events = list(
        plane.project(
            _output("hello", audio=True, finished=True),
            context={"modalities": ("text", "audio"), "turn_id": 0},
        )
    )
    assert len(events) == 1
    assert events[0]["text"] == "hello"
    assert events[0]["audio_data"] == "AAAA"
    assert events[0]["end_of_turn"] is True
    assert "qwen_function_call" not in events[0]


def test_tools_reach_the_chat_template_and_results_join_the_next_prompt():
    seen: dict[str, object] = {}

    def apply_chat_template(messages, **kwargs):
        seen["messages"] = messages
        seen["kwargs"] = kwargs
        return "prompt"

    plugin = Qwen3OmniDuplexPlugin(lambda *args: None)
    plugin.processor = SimpleNamespace(apply_chat_template=apply_chat_template)
    plugin.validate_client_extra_body({"realtime_tools": [{"type": "function", "name": "get_current_time"}]})
    runtime = plugin.runtime_config_for_update(
        DuplexSessionConfig(
            model="qwen",
            extra_body={"realtime_tools": [{"type": "function", "name": "get_current_time", "description": "Now"}]},
        ),
        {},
    )
    call = plugin.parse_function_call(
        {
            "qwen_function_call": {
                "call_id": "call_1",
                "name": "get_current_time",
                "arguments": "{}",
            }
        }
    )
    assert call == {"call_id": "call_1", "name": "get_current_time", "arguments": "{}"}
    updated = plugin.runtime_config_for_function_output(
        DuplexSessionConfig(model="qwen"),
        runtime,
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "name": "get_current_time",
            "arguments": "{}",
            "output": "2026-10-05T00:00:00+00:00",
        },
        history_len=1,
    )
    turns = updated["qwen_tool_turns"]
    assert isinstance(turns, list)
    assert turns[0]["history_len"] == 1
    plugin.plan_append(
        request_id="follow-up",
        fence=None,
        session_config={
            "qwen_messages": [{"role": "user", "content": "what time is it", "_src": 0}],
            "qwen_tool_turns": turns,
        },
        runtime_config=updated,
        seq=1,
        turn_seq=1,
        payload={"type": "conversation"},
        final=True,
        sampling_params=None,
    )
    messages = seen["messages"]
    kwargs = seen["kwargs"]
    assert isinstance(messages, list)
    assert isinstance(kwargs, dict)
    assert messages[-2]["role"] == "assistant"
    assert messages[-2]["tool_calls"][0]["function"]["name"] == "get_current_time"
    assert messages[-1] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "2026-10-05T00:00:00+00:00",
    }
    assert kwargs["tools"] == [{"type": "function", "function": {"name": "get_current_time", "description": "Now"}}]


def test_qwen_preparation_uses_the_supplied_call_and_can_be_repeated():
    plugin = Qwen3OmniDuplexPlugin(lambda *args: None)
    parsed = plugin.parse_function_call(
        {"qwen_function_call": {"call_id": "call_1", "name": "get_weather", "arguments": '{"city":"Taipei"}'}}
    )
    assert parsed == {"call_id": "call_1", "name": "get_weather", "arguments": '{"city":"Taipei"}'}
    item = {
        "type": "function_call_output",
        "call_id": "call_1",
        "name": "get_weather",
        "arguments": '{"city":"Taipei"}',
        "output": "25C",
    }
    first = plugin.runtime_config_for_function_output(DuplexSessionConfig(model="qwen"), {}, item)
    second = plugin.runtime_config_for_function_output(
        DuplexSessionConfig(model="qwen"),
        first,
        item,
        history_len=1,
    )
    assert len(first["qwen_tool_turns"]) == 1
    assert len(second["qwen_tool_turns"]) == 2
    assert first["qwen_tool_turns"][0]["messages"][0]["tool_calls"][0]["function"] == {
        "name": "get_weather",
        "arguments": '{"city":"Taipei"}',
    }
    with pytest.raises(DuplexRuntimeConfigError) as refused:
        plugin.runtime_config_for_function_output(
            DuplexSessionConfig(model="qwen"),
            {},
            {"type": "function_call_output", "call_id": "call_1", "name": "forged", "output": "25C"},
        )
    assert refused.value.code == "invalid_function_call_output"


def test_malformed_tool_call_is_not_speech():
    cursor = ThinkerToolCursor()
    speech, call = cursor.absorb("Sure <tool_call>not json</tool_call> thanks")
    assert speech == "Sure "
    assert call == {"call_id": "", "name": "", "invalid": "1"}
    rest, later = cursor.absorb("")
    assert rest == " thanks"
    assert later is None
    assert "<tool_call>" not in speech + rest
    plugin = Qwen3OmniDuplexPlugin(lambda *args: None)
    recognized = _recognized_function_call(plugin, {"qwen_function_call": call})
    assert recognized is _INVALID_FUNCTION_CALL


def test_client_history_len_does_not_steer_the_splice():
    plugin = Qwen3OmniDuplexPlugin(lambda *args: None)
    published = plugin.note_published_function_call({}, call_id="call_paris", history_len=1)
    updated = plugin.runtime_config_for_function_output(
        DuplexSessionConfig(model="qwen"),
        published,
        {
            "type": "function_call_output",
            "call_id": "call_paris",
            "name": "get_weather",
            "arguments": "{}",
            "output": "18C",
            "_history_len": 99,
        },
        history_len=2,
    )
    assert updated["qwen_tool_turns"][0]["history_len"] == 1
    assert "call_paris" not in updated.get("qwen_call_anchors", {})


def test_cancel_puts_a_submitted_tool_pair_back():
    plugin = Qwen3OmniDuplexPlugin(lambda *args: None)
    pair = {"history_len": 1, "messages": [{"role": "tool", "content": "18C"}]}
    held = plugin.runtime_config_after_response(
        {"qwen_tool_turns": [], "qwen_tool_turns_inflight": [pair]},
        committed=False,
    )
    assert held["qwen_tool_turns"] == [pair]
    assert "qwen_tool_turns_inflight" not in held
    dropped = plugin.runtime_config_after_response(
        {"qwen_tool_turns": [], "qwen_tool_turns_inflight": [pair]},
        committed=True,
    )
    assert dropped["qwen_tool_turns"] == []
    assert "qwen_tool_turns_inflight" not in dropped


def test_playback_keeps_the_pair_until_that_response_stores_text():
    plugin = Qwen3OmniDuplexPlugin(lambda *args: None)
    pair = {"history_len": 1, "messages": [{"role": "tool", "content": "18C"}]}
    runtime = {
        "qwen_tool_turns": [],
        "qwen_tool_turns_inflight": [pair],
        "qwen_tool_response_id": "resp-a",
    }
    assert (
        plugin.runtime_config_after_response(runtime, committed=False, awaiting_playback=True, response_id="resp-a")
        is None
    )
    assert plugin.runtime_config_after_response(runtime, committed=False, response_id="resp-b") is None
    dropped = plugin.runtime_config_after_response(runtime, committed=True, response_id="resp-a")
    assert dropped["qwen_tool_turns"] == []
    assert "qwen_tool_turns_inflight" not in dropped
    assert "qwen_tool_response_id" not in dropped


def test_stream_and_session_close_drop_thinker_cursors():
    plane = QwenDataPlane(lambda *args: None)
    mine = duplex_resource_request_id(DuplexFence(session_id="sess-a", epoch=1, turn_id=1), "stage0-turn1")
    other = duplex_resource_request_id(DuplexFence(session_id="sess-b", epoch=1, turn_id=1), "stage0-turn1")

    def project(request_id: str) -> None:
        output = SimpleNamespace(
            request_id=request_id,
            stage_id=0,
            finished=False,
            outputs=[SimpleNamespace(text="hello", multimodal_output={})],
            multimodal_output={"sr": 24000},
        )
        list(plane.project({"data_plane_outputs": [output]}, context={"modalities": ["text"]}))

    project(mine)
    assert mine in plane._tool_cursors
    plane.close_stream(mine)
    assert mine not in plane._tool_cursors
    project(mine)
    project(other)
    kept = plane._tool_cursors[other]
    plane.close_session("sess-a")
    assert mine not in plane._tool_cursors
    assert plane._tool_cursors[other] is kept


def test_call_anchors_older_than_the_prompt_window_are_dropped():
    plugin = Qwen3OmniDuplexPlugin(lambda *args: None)
    plugin.processor = SimpleNamespace(apply_chat_template=lambda messages, **kwargs: "prompt")
    published = plugin.note_published_function_call({}, call_id="call_old", history_len=1)
    assert published is not None
    later = plugin.note_published_function_call(published, call_id="call_new", history_len=20)
    assert later is not None
    assert "call_old" not in later["qwen_call_anchors"]
    assert later["qwen_call_anchors"]["call_new"] == 20
    conversation = [{"role": "user", "content": f"q{i}"} for i in range(20)]
    plan = plugin.plan_append(
        request_id="prune-anchors",
        fence=None,
        session_config={
            "qwen_messages": [{"role": "user", "content": "q19", "_src": 19}],
            "conversation": conversation,
        },
        runtime_config={"qwen_call_anchors": {"call_old": 1, "call_new": 20}},
        seq=1,
        turn_seq=1,
        payload={"type": "conversation"},
        final=True,
        sampling_params=None,
    )
    assert plan.prompt["_qwen_call_anchors"] == {"call_new": 20}
    patch = plugin.runtime_patch_for_submitted_prompt(plan.prompt)
    assert patch is not None
    assert patch["qwen_call_anchors"] == {"call_new": 20}
    assert "_qwen_call_anchors" not in plan.prompt


def test_published_anchor_beats_the_length_when_the_result_arrives():
    plugin = Qwen3OmniDuplexPlugin(lambda *args: None)
    published = plugin.note_published_function_call({}, call_id="call_paris", history_len=1)
    assert published is not None
    kept = plugin.runtime_config_for_update(
        DuplexSessionConfig(model="qwen"),
        published,
    )
    updated = plugin.runtime_config_for_function_output(
        DuplexSessionConfig(model="qwen"),
        kept,
        {
            "type": "function_call_output",
            "call_id": "call_paris",
            "name": "get_weather",
            "arguments": "{}",
            "output": "18C",
        },
        history_len=2,
    )
    assert updated["qwen_tool_turns"][0]["history_len"] == 1
    assert plugin.has_pending_tool_followup(updated)
    assert plugin.extra_prompt_config(updated)["qwen_tool_turns"][0]["history_len"] == 1
    prompt = {"prompt": "x", "_qwen_tool_turns_remaining": []}
    patch = plugin.runtime_patch_for_submitted_prompt(prompt)
    assert "_qwen_tool_turns_remaining" not in prompt
    assert patch == {"qwen_tool_turns": []}
    assert not plugin.has_pending_tool_followup({**updated, **patch})


def test_prompt_without_tools_does_not_pass_a_tools_argument():
    seen: dict[str, object] = {}

    def apply_chat_template(messages, **kwargs):
        seen["kwargs"] = kwargs
        return repr(messages)

    plugin = Qwen3OmniDuplexPlugin(lambda *args: None)
    plugin.processor = SimpleNamespace(apply_chat_template=apply_chat_template)
    plan = plugin.plan_append(
        request_id="plain",
        fence=None,
        session_config={"qwen_messages": [{"role": "user", "content": "hello"}]},
        runtime_config={},
        seq=1,
        turn_seq=1,
        payload={"type": "conversation"},
        final=True,
        sampling_params=None,
    )
    kwargs = seen["kwargs"]
    assert isinstance(kwargs, dict)
    assert "tools" not in kwargs
    assert plan.prompt["prompt"] == repr([{"role": "user", "content": [{"type": "text", "text": "hello"}]}])


def test_one_thinker_chunk_emits_every_completed_call():
    plane = QwenDataPlane(lambda *args: "AAAA")
    second = '<tool_call>{"name": "get_weather", "arguments": {"city": "Taipei"}}</tool_call>'
    events = list(
        plane.project(
            _output(f"{_CALL}{second}", finished=True),
            context={"modalities": ("text", "audio"), "turn_id": 1},
        )
    )
    names = [event["qwen_function_call"]["name"] for event in events if "qwen_function_call" in event]
    assert names == ["get_current_time", "get_weather"]
    assert events[-1]["end_of_turn"] is True


def test_identical_calls_in_one_chunk_both_emit():
    plane = QwenDataPlane(lambda *args: "AAAA")
    events = list(
        plane.project(
            _output(f"{_CALL}{_CALL}", finished=True),
            context={"modalities": ("text", "audio"), "turn_id": 1},
        )
    )
    calls = [event["qwen_function_call"] for event in events if "qwen_function_call" in event]
    assert [call["name"] for call in calls] == ["get_current_time", "get_current_time"]
    assert calls[0]["call_id"] != calls[1]["call_id"]


def _weather_turn(call_id: str, city: str, output: str, history_len: int) -> dict[str, object]:
    return {
        "history_len": history_len,
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": json.dumps({"city": city})},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": call_id, "content": output},
        ],
    }


def test_history_tool_pairs_keep_each_call_with_its_question():
    seen: dict[str, object] = {}

    def apply_chat_template(messages, **kwargs):
        del kwargs
        seen["messages"] = messages
        return "prompt"

    plugin = Qwen3OmniDuplexPlugin(lambda *args: None)
    plugin.processor = SimpleNamespace(apply_chat_template=apply_chat_template)
    state = plugin.create_session_state()
    turns = [_weather_turn("call_1", "Taipei", "25C", 1), _weather_turn("call_2", "Tokyo", "29C", 2)]
    config = plugin.prepare_prompt_config(
        {
            "conversation": [
                {"role": "user", "content": "taipei"},
                {"role": "user", "content": "tokyo"},
            ],
            "qwen_tool_turns": turns,
        },
        state=state,
        payload={},
    )
    plugin.plan_append(
        request_id="second-follow-up",
        fence=None,
        session_config=config,
        runtime_config={},
        seq=1,
        turn_seq=1,
        payload={"type": "conversation"},
        final=True,
        sampling_params=None,
    )
    prompt_messages = seen["messages"]
    assert isinstance(prompt_messages, list)
    roles = [message["role"] for message in prompt_messages]
    assert roles == ["user", "assistant", "tool", "user", "assistant", "tool"]
    assert prompt_messages[2]["content"] == "25C"
    assert prompt_messages[5]["content"] == "29C"


def test_new_user_audio_does_not_rebind_the_previous_question():
    plugin = Qwen3OmniDuplexPlugin(lambda *args: None)
    state = plugin.create_session_state()
    paris = {"role": "user", "content": [{"type": "audio_url"}]}
    plugin.prepare_prompt_config({"conversation": [paris]}, state=state, payload={"audio": "paris"})
    tokyo = {"role": "user", "content": [{"type": "audio_url"}]}
    config = plugin.prepare_prompt_config(
        {
            "conversation": [paris, tokyo],
            "qwen_tool_turns": [_weather_turn("call_p", "Paris", "18C", 1)],
        },
        state=state,
        payload={"audio": "tokyo"},
    )
    payloads = [message["audio_payload"]["audio"] for message in config["qwen_messages"] if "audio_payload" in message]
    assert payloads == ["paris", "tokyo"]


def test_projected_function_call_supplies_name_and_arguments():
    prepared = prepared_function_call_output(
        {"type": "function_call_output", "call_id": "call_1", "output": "25C", "name": "forged"},
        {
            "item_1": {
                "type": "function_call",
                "call_id": "call_1",
                "name": "get_weather",
                "arguments": '{"city":"Taipei"}',
            }
        },
    )
    assert prepared["name"] == "get_weather"
    assert prepared["arguments"] == '{"city":"Taipei"}'
    assert prepared["output"] == "25C"


def test_missing_parse_function_call_is_not_a_call():
    assert _recognized_function_call(SimpleNamespace(), {"text": "hello"}) is None
    call = _recognized_function_call(
        SimpleNamespace(),
        {"function_call": True, "call_id": "call_1", "name": "get_weather", "arguments": {"city": "Taipei"}},
    )
    assert call == {"call_id": "call_1", "name": "get_weather", "arguments": '{"city": "Taipei"}'}


def test_parse_function_call_json_dumps_dict_arguments():
    plugin = Qwen3OmniDuplexPlugin(lambda *args: None)
    parsed = plugin.parse_function_call(
        {"qwen_function_call": {"call_id": "call_1", "name": "get_weather", "arguments": {"city": "Taipei"}}}
    )
    assert parsed == {"call_id": "call_1", "name": "get_weather", "arguments": '{"city": "Taipei"}'}
    assert _coerce_function_call({"call_id": "c", "name": "n", "arguments": {"x": 1}}) == {
        "call_id": "c",
        "name": "n",
        "arguments": '{"x": 1}',
    }
