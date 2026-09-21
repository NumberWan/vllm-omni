# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from vllm_omni.engine.duplex.realtime_events import (
    RealtimeProjectionState,
    project_internal_event,
)


def test_input_transcribed_is_a_user_transcript_event() -> None:
    events = project_internal_event(
        RealtimeProjectionState(session_id="s"),
        {"type": "input.transcribed", "transcript": "  你好  "},
    )
    assert len(events) == 1
    assert events[0].wire_type == "conversation.item.input_audio_transcription.completed"
    assert events[0].transcript == "你好"


def test_blank_input_transcribed_emits_nothing() -> None:
    events = project_internal_event(
        RealtimeProjectionState(session_id="s"),
        {"type": "input.transcribed", "transcript": "  "},
    )
    assert events == []
