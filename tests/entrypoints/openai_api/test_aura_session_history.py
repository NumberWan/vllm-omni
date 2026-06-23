# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for AURA SessionHistory."""

from __future__ import annotations

import numpy as np
import pytest

from vllm_omni.entrypoints.openai.aura_session_history import (
    SessionHistory,
    is_effectively_silent,
    normalize_assistant_text,
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

    assert "<|video_pad|>" in vllm_inputs["prompt"]
    assert "What is happening?" in vllm_inputs["prompt"]
    assert vllm_inputs["prompt"].endswith("<|im_start|>assistant\n")
    assert "video" in vllm_inputs["multi_modal_data"]
    assert len(vllm_inputs["multi_modal_data"]["video"]) == 1


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


def test_add_assistant_message_normalizes_punctuation_filler_to_silent():
    history = SessionHistory()
    history.add_assistant_message(" ﹑")
    assert history.history[-1]["content"] == "<|silent|>"
    assert normalize_assistant_text(" ﹑") == "<|silent|>"


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
