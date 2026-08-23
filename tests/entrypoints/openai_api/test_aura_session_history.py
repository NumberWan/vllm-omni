# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for AURA SessionHistory."""

from __future__ import annotations

import numpy as np
import pytest

from vllm_omni.model_executor.stage_input_processors.aura_session_history import (
    AURA_IM_END_TOKEN_ID,
    AURA_SILENT_TOKEN_ID,
    SessionHistory,
    aura_silent_stop_token_ids,
    is_effectively_silent,
    normalize_assistant_text,
    should_stop_aura_silent_generation,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _video_tuple(num_frames: int = 2) -> tuple[np.ndarray, dict]:
    frames = np.zeros((num_frames, 8, 8, 3), dtype=np.uint8)
    metadata = {
        "fps": 2.0,
        "duration": num_frames / 2.0,
        "total_num_frames": num_frames,
        "frames_indices": list(range(num_frames)),
        "video_backend": "opencv",
        "do_sample_frames": False,
    }
    return frames, metadata


def test_get_vllm_inputs_includes_video_and_text():
    history = SessionHistory()
    history.add_user_message("What is happening?", video_tuple=_video_tuple())

    vllm_inputs = history.get_vllm_inputs()

    assert (
        "<|im_start|>user\n"
        "<|vision_start|><|video_pad|><|vision_end|>"
        "What is happening?<|im_end|>\n"
    ) in vllm_inputs["prompt"]
    assert vllm_inputs["prompt"].endswith("<|im_start|>assistant\n")
    assert "video" in vllm_inputs["multi_modal_data"]
    assert len(vllm_inputs["multi_modal_data"]["video"]) == 1


def test_non_tool_v2_prompt_uses_closed_think_prefix(monkeypatch):
    monkeypatch.setenv("VLLM_AURA_SILENT_TOKEN_ID", "248070")
    monkeypatch.delenv("VLLM_AURA_ENABLE_THINKING", raising=False)
    history = SessionHistory()
    history.add_user_message("What is happening?", video_tuple=_video_tuple())
    from vllm_omni.model_executor.stage_input_processors.aura_tool_protocol import (
        AURA_ASSISTANT_PREFIX_THINK_CLOSED,
    )

    assert history.get_vllm_inputs()["prompt"].endswith(AURA_ASSISTANT_PREFIX_THINK_CLOSED)


def test_preview_vllm_inputs_matches_committed_turn():
    history = SessionHistory()
    history.add_user_message("first", video_tuple=_video_tuple())
    history.add_assistant_message("reply")

    preview = history.preview_vllm_inputs("second", video_tuple=_video_tuple(3))
    history.add_user_message("second", video_tuple=_video_tuple(3), mm_uuid=history._pending_mm_uuid)
    committed = history.get_vllm_inputs()

    assert preview["prompt"] == committed["prompt"]
    assert len(preview["multi_modal_data"]["video"]) == len(committed["multi_modal_data"]["video"])


def test_preview_attaches_stable_uuids_and_marks_warm_history():
    history = SessionHistory()
    uid0 = history.new_mm_uuid()
    history.add_user_message("first", video_tuple=_video_tuple(), mm_uuid=uid0)
    history.add_assistant_message("reply")
    history.mark_mm_uuids_warm([uid0])

    preview = history.preview_vllm_inputs("second", video_tuple=_video_tuple(3))
    videos = preview["multi_modal_data"]["video"]
    uuids = preview["multi_modal_uuids"]["video"]

    # Pixels always present (HF path cannot take None); warm count is diagnostic.
    assert videos[0] is not None
    assert videos[1] is not None
    assert uuids[0] == uid0
    assert uuids[1] == history._pending_mm_uuid
    assert preview["mm_uuid_only_videos"] == 1
    assert preview["mm_pixel_videos"] == 1


def test_tool_resume_preview_keeps_chain_transient_until_commit():
    rendered_messages = []

    class RecordingTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["tools"]
            rendered_messages.extend(messages)
            return "tool-rendered-prompt"

    history = SessionHistory()
    old_uuid = history.new_mm_uuid()
    history.add_user_message("first", video_tuple=_video_tuple(), mm_uuid=old_uuid)
    history.add_assistant_message("first answer")
    history.mark_mm_uuids_warm([old_uuid])
    transient = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "function": {
                        "name": "mock_echo",
                        "arguments": {"text": "hello"},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": '{"ok":true}',
        },
    ]

    preview = history.preview_vllm_inputs(
        "second",
        video_tuple=_video_tuple(3),
        tokenizer=RecordingTokenizer(),
        tools=[{"type": "function", "function": {"name": "mock_echo"}}],
        transient_messages=transient,
    )

    assert [message["role"] for message in rendered_messages] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "tool",
    ]
    assert preview["multi_modal_uuids"]["video"] == [
        old_uuid,
        history._pending_mm_uuid,
    ]
    assert len(preview["multi_modal_data"]["video"]) == 2
    # Before commit, tool XML/result remain transient-only.
    assert [message["role"] for message in history.history] == [
        "system",
        "user",
        "assistant",
    ]


def test_commit_session_turn_persists_full_tool_chain():
    from vllm_omni.model_executor.stage_input_processors.aura_session_history import (
        clear_all_sessions,
        commit_session_turn,
        get_or_create_session_history,
        record_pending_turn,
    )

    clear_all_sessions()
    history = get_or_create_session_history("tool-hist-commit")
    tool_chain = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "mock_echo", "arguments": {"text": "hello"}},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": '{"text":"hello"}'},
    ]
    record_pending_turn(
        "tool-hist-commit",
        request_id="req-tool",
        transcript="請回聲",
        video_tuple=_video_tuple(),
        tool_chain=tool_chain,
    )
    commit_session_turn("tool-hist-commit", "已為你回聲 hello。")

    roles = [message["role"] for message in history.history]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    assert history.history[-3]["tool_calls"][0]["id"] == "call-1"
    assert history.history[-2]["tool_call_id"] == "call-1"
    assert history.history[-1]["content"] == "已為你回聲 hello。"

    rendered: list[dict] = []

    class RecordingTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            rendered.extend(messages)
            return "next-turn"

    history.preview_vllm_inputs(
        "follow-up",
        tokenizer=RecordingTokenizer(),
        tools=[{"type": "function", "function": {"name": "mock_echo"}}],
    )
    assert [m["role"] for m in rendered] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "user",
    ]
    clear_all_sessions()


def test_pruning_keeps_tool_chain_in_context_history():
    history = SessionHistory(
        max_rounds=2,
        num_rounds_keep=1,
        pruning_enabled=True,
        max_context_qas=5,
        max_1qna_rounds=4,
    )
    history.add_user_message("use tool", video_tuple=_video_tuple())
    history.add_assistant_tool_message(
        "",
        [{"id": "c1", "type": "function", "function": {"name": "mock_echo", "arguments": {}}}],
    )
    history.add_tool_message("c1", '{"ok":true}')
    history.add_assistant_message("done with tool")
    history.add_user_message("q1", video_tuple=_video_tuple())
    history.add_assistant_message("a1")
    history.add_user_message("q2", video_tuple=_video_tuple())
    history.add_assistant_message("a2")

    flat = [msg for qa in history._context_history for msg in qa]
    assert any(msg.get("role") == "tool" for msg in flat)
    assert any(msg.get("tool_calls") for msg in flat if msg.get("role") == "assistant")


def test_to_dict_roundtrip_preserves_tool_fields():
    history = SessionHistory(pruning_enabled=False)
    history.add_user_message("tool please", video_tuple=_video_tuple())
    history.add_assistant_tool_message(
        "calling",
        [{"id": "c9", "type": "function", "function": {"name": "mock_echo", "arguments": {"x": 1}}}],
        reasoning_content="",
    )
    history.add_tool_message("c9", '{"x":1}')
    history.add_assistant_message("ok")

    restored = SessionHistory.from_dict(history.to_dict())
    roles = [m["role"] for m in restored.history]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    assert restored.history[-3]["tool_calls"][0]["id"] == "c9"
    assert restored.history[-2]["tool_call_id"] == "c9"


def test_to_dict_roundtrip_preserves_history():
    history = SessionHistory(max_rounds=4, num_rounds_keep=2, pruning_enabled=False)
    history.add_user_message("", video_tuple=_video_tuple())
    history.add_assistant_message("I see movement.")
    history.add_user_message("Tell me more.", video_tuple=_video_tuple(3))
    history.add_assistant_message("<|silent|>")

    restored = SessionHistory.from_dict(history.to_dict())

    assert restored.system_prompt == history.system_prompt
    assert restored.current_rounds == history.current_rounds
    assert len(restored.history) == len(history.history)
    restored_inputs = restored.get_vllm_inputs()
    original_inputs = history.get_vllm_inputs()
    assert restored_inputs["prompt"] == original_inputs["prompt"]
    assert len(restored_inputs["multi_modal_data"].get("video", [])) == len(
        original_inputs["multi_modal_data"].get("video", [])
    )


def test_is_effectively_silent_treats_punctuation_filler_as_silent():
    assert is_effectively_silent("")
    assert is_effectively_silent("  ")
    assert is_effectively_silent("<|silent|>")
    assert is_effectively_silent(" ﹑")
    assert is_effectively_silent("，。")
    assert not is_effectively_silent("好的")
    assert not is_effectively_silent(" 好的，")


def test_is_effectively_silent_treats_empty_think_as_silent():
    assert is_effectively_silent("<think></think>")
    assert is_effectively_silent("<think>\n\n</think>")
    assert is_effectively_silent("<think>\n\n</think>\n\n")
    assert is_effectively_silent("<think>unfinished")
    assert not is_effectively_silent("<think>note</think>\n好的，我会留意。")


def test_normalize_assistant_text_strips_think_and_maps_empty_to_silent():
    assert normalize_assistant_text("<think></think>") == "<|silent|>"
    assert normalize_assistant_text("<think>\n\n</think>\n\n") == "<|silent|>"
    assert normalize_assistant_text("<think>hidden</think>\n好的，我会留意。") == "好的，我会留意。"


def test_add_assistant_message_normalizes_punctuation_filler_to_silent():
    history = SessionHistory()
    history.add_assistant_message(" ﹑")
    assert history.history[-1]["content"] == "<|silent|>"
    assert normalize_assistant_text(" ﹑") == "<|silent|>"


def test_empty_think_assistant_is_pruned_like_silent():
    history = SessionHistory(
        max_rounds=2,
        num_rounds_keep=1,
        pruning_enabled=True,
        max_context_qas=5,
        max_1qna_rounds=4,
    )
    history.add_user_message("vision", video_tuple=_video_tuple())
    history.add_assistant_message("<think>\n\n</think>")
    history.add_user_message("question 1", video_tuple=_video_tuple())
    history.add_assistant_message("answer 1")
    history.add_user_message("question 2", video_tuple=_video_tuple())
    history.add_assistant_message("answer 2")

    # Empty-think round must not land in compressed context as spoken text.
    for qa in history._context_history:
        for msg in qa:
            assert "<think>" not in str(msg.get("content", ""))
            assert msg.get("content") != "<think>\n\n</think>"
    assert history.history[-1]["content"] == "answer 2"


def test_pruning_moves_old_rounds_to_context_history():
    history = SessionHistory(
        max_rounds=2,
        num_rounds_keep=1,
        pruning_enabled=True,
        max_context_qas=5,
        max_1qna_rounds=4,
    )

    for round_idx in range(3):
        history.add_user_message(f"question {round_idx}", video_tuple=_video_tuple())
        history.add_assistant_message(f"answer {round_idx}")

    assert history._sw_round_count() <= history.max_rounds
    assert len(history._context_history) >= 1
    assert any("question 0" in msg.get("content", "") for qa in history._context_history for msg in qa)
    for qa in history._context_history:
        for msg in qa:
            content = msg.get("content", "")
            assert isinstance(content, str), "context_history stores text-only user/assistant content"
            assert "<|video_pad|>" not in content
            assert content != "<|silent|>"


def test_aura_session_state_commit_turn_clears_frames_without_api_history():
    from vllm_omni.model_executor.stage_input_processors.aura_session_history import (
        clear_all_sessions,
        create_streaming_session,
        get_or_create_session_history,
        get_session_history,
        record_pending_turn,
    )

    clear_all_sessions()
    state = create_streaming_session(pruning_enabled=False)
    state.turn_frame_arrays = [np.zeros((4, 4, 3), dtype=np.uint8), np.ones((4, 4, 3), dtype=np.uint8)]
    state.freeze_turn_video(
        {
            "video": [
                (
                    np.zeros((2, 4, 4, 3), dtype=np.uint8),
                    {"fps": 2.0, "total_num_frames": 2},
                )
            ]
        }
    )
    get_or_create_session_history(state.session_id, system_prompt="sys")
    record_pending_turn(
        state.session_id,
        request_id="req-1",
        transcript="hello",
        video_tuple=(np.zeros((2, 4, 4, 3), dtype=np.uint8), {"fps": 2.0, "total_num_frames": 2}),
    )

    state.commit_turn(response_text="reply", request_id="req-1")

    assert state.turn_frame_arrays == []
    assert state.pending_turn_video is None
    assert state.history.current_rounds == 0
    stage_hist = get_session_history(state.session_id)
    assert stage_hist is not None
    prompt = stage_hist.get_vllm_inputs()["prompt"]
    assert "hello" in prompt
    assert "reply" in prompt
    clear_all_sessions()


def test_should_stop_aura_silent_generation_on_first_token():
    assert aura_silent_stop_token_ids() == (AURA_SILENT_TOKEN_ID, AURA_IM_END_TOKEN_ID)
    assert should_stop_aura_silent_generation(token_ids=[AURA_SILENT_TOKEN_ID])
    assert should_stop_aura_silent_generation(token_ids=[AURA_IM_END_TOKEN_ID])
    assert not should_stop_aura_silent_generation(token_ids=[42])
    assert should_stop_aura_silent_generation(text="<|silent|>")
    assert should_stop_aura_silent_generation(text=" ﹑")
    assert not should_stop_aura_silent_generation(text="好的")
