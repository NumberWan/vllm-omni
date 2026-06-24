# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm_omni.entrypoints.openai.aura.session_history import SessionHistory, clear_all_sessions, register_session
from vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder import (
    PRECOMPUTED_TEXT_IDS_KEY,
)
from vllm_omni.model_executor.stage_input_processors.aura_omni import (
    SILENT_TEXT,
    _clean_asr_transcript,
    asr2aura,
    asr2aura_session,
    aura2tts,
    pop_turn_transcript,
    record_turn_transcript,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _source_output(text: str, request_id: str = "req-1", token_ids: list[int] | None = None):
    output = SimpleNamespace(text=text, cumulative_token_ids=token_ids or [1, 2, 3], multimodal_output={})
    return SimpleNamespace(request_id=request_id, outputs=[output])


def _source_delta_final_output(cumulative_text: str, request_id: str = "req-1"):
    output = SimpleNamespace(
        text="",
        cumulative_text=cumulative_text,
        cumulative_token_ids=[1, 2, 3],
        multimodal_output={},
    )
    return SimpleNamespace(request_id=request_id, outputs=[output])


def test_asr2aura_carries_video_and_strips_audio_from_vl_input():
    prompt = {
        "multi_modal_data": {
            "audio": ("wave", 16000),
            "video": ["frame-0", "frame-1"],
        },
        "additional_information": {"aura_system_prompt": ["system"]},
    }

    [next_input] = asr2aura([_source_output("What is happening now?")], prompt=[prompt])

    assert next_input["multi_modal_data"] == {"video": ["frame-0", "frame-1"]}
    assert "<|video_pad|>" in next_input["prompt"]
    assert "What is happening now?" in next_input["prompt"]
    assert next_input["prompt"].startswith("<|im_start|>system\nsystem")


def test_clean_asr_transcript_strips_qwen3_asr_wrapper():
    assert _clean_asr_transcript("language Chinese<asr_text>画面有什么？") == "画面有什么？"
    assert _clean_asr_transcript("language Chinese<asr_text>画面有什么") == "画面有什么"
    assert _clean_asr_transcript("  hello  ") == "hello"


def test_transcript_storage_normalizes_internal_request_id():
    external = "video-a234643a36e3"
    internal = f"{external}-b52ad29b"
    record_turn_transcript(internal, "画面有什么")
    assert pop_turn_transcript(external) == "画面有什么"
    assert pop_turn_transcript(internal) == ""


def test_asr2aura_session_restores_history_and_turn_video():
    history = SessionHistory(pruning_enabled=False)
    prompt = {
        "additional_information": {
            "aura_session_state": history.to_dict(),
            "aura_turn_video": {
                "frames": [
                    [[[1, 0, 0], [0, 1, 0]], [[0, 0, 1], [1, 1, 0]]],
                    [[[2, 0, 0], [0, 2, 0]], [[0, 0, 2], [2, 2, 0]]],
                ],
                "metadata": {
                    "fps": 2.0,
                    "duration": 1.0,
                    "total_num_frames": 2,
                    "frames_indices": [0, 1],
                    "video_backend": "opencv",
                    "do_sample_frames": False,
                },
            },
            "aura_system_prompt": ["custom system"],
        }
    }

    [next_input] = asr2aura_session(
        [_source_output("language Chinese<asr_text>Hello there.", request_id="video-testreq01-abcd1234")],
        prompt=[prompt],
    )

    assert "<|video_pad|>" in next_input["prompt"]
    assert "Hello there." in next_input["prompt"]
    assert "language Chinese" not in next_input["prompt"]
    assert "<asr_text>" not in next_input["prompt"]
    assert next_input["multi_modal_data"]["video"]
    assert pop_turn_transcript("video-testreq01") == "Hello there."


def test_asr2aura_session_uses_server_side_store():
    clear_all_sessions()
    history = SessionHistory(pruning_enabled=False)
    session_id = "aura-store-test"
    register_session(session_id, history)
    history.add_user_message(
        "prior round",
        video_tuple=(
            [
                [[[1, 0, 0], [0, 1, 0]], [[0, 0, 1], [1, 1, 0]]],
                [[[2, 0, 0], [0, 2, 0]], [[0, 0, 2], [2, 2, 0]]],
            ],
            {
                "fps": 2.0,
                "duration": 1.0,
                "total_num_frames": 2,
                "frames_indices": [0, 1],
                "video_backend": "opencv",
                "do_sample_frames": False,
            },
        ),
    )
    history.add_assistant_message("ack")

    prompt = {
        "additional_information": {
            "aura_session_id": session_id,
            "aura_turn_video": {
                "frames": [
                    [[[3, 0, 0], [0, 3, 0]], [[0, 0, 3], [3, 3, 0]]],
                    [[[4, 0, 0], [0, 4, 0]], [[0, 0, 4], [4, 4, 0]]],
                ],
                "metadata": {
                    "fps": 2.0,
                    "duration": 1.0,
                    "total_num_frames": 2,
                    "frames_indices": [0, 1],
                    "video_backend": "opencv",
                    "do_sample_frames": False,
                },
            },
            "aura_system_prompt": ["custom system"],
        }
    }

    [next_input] = asr2aura_session(
        [_source_output("language Chinese<asr_text>Hello there.", request_id="video-testreq02-abcd1234")],
        prompt=[prompt],
    )

    assert "prior round" in next_input["prompt"]
    assert "Hello there." in next_input["prompt"]
    assert len(next_input["multi_modal_data"]["video"]) == 2
    assert len(history.get_vllm_inputs()["multi_modal_data"]["video"]) == 1
    clear_all_sessions()


def test_asr2aura_supports_video_only_observation():
    prompt = {"multi_modal_data": {"video": ["frame-0", "frame-1"]}}

    [next_input] = asr2aura([_source_output("")], prompt=[prompt])

    assert "<|video_pad|>" in next_input["prompt"]
    assert "<|im_start|>assistant" in next_input["prompt"]


@pytest.mark.parametrize(
    ("additional_information", "source", "expected"),
    [
        pytest.param(
            {
                "tts_language": ["Chinese"],
                "tts_instruct": ["Calm voice."],
                "tts_ref_audio": ["ref.wav"],
                "tts_ref_text": ["Reference transcript sample."],
            },
            _source_output("Hello."),
            {
                "task_type": ["Base"],
                "language": ["Chinese"],
                "text": ["Hello."],
                "ref_audio": ["ref.wav"],
                "ref_text": ["Reference transcript sample."],
                "x_vector_only_mode": [False],
                "instruct": ["Calm voice."],
            },
            id="base",
        ),
        pytest.param(
            {
                "tts_task_type": ["CustomVoice"],
                "tts_speaker": ["vivian"],
            },
            _source_output("Hello."),
            {
                "task_type": ["CustomVoice"],
                "speaker": ["Vivian"],
                "text": ["Hello."],
            },
            id="custom_voice",
        ),
        pytest.param(
            {
                "tts_task_type": ["Base"],
                "tts_x_vector_only_mode": [True],
                "tts_ref_audio": ["ref.wav"],
                "tts_ref_text": ["Reference transcript sample."],
            },
            _source_output("Hello."),
            {
                "task_type": ["Base"],
                "x_vector_only_mode": [True],
                "text": ["Hello."],
            },
            id="x_vector_only",
        ),
        pytest.param(
            {
                "tts_ref_audio": ["ref.wav"],
                "tts_ref_text": ["Reference transcript sample."],
                "tts_pass_token_ids": [True],
            },
            _source_output("Hello.", token_ids=[151644, 77091, 198, 108386, 1773, 151645, 198]),
            {
                PRECOMPUTED_TEXT_IDS_KEY: [[151644, 77091, 198, 108386, 1773, 151645, 198, 151644, 77091, 198]],
            },
            id="token_ids",
        ),
    ],
)
def test_aura2tts_modes(additional_information, source, expected):
    prompt = {"additional_information": additional_information}

    [tts_input] = aura2tts([source], prompt=[prompt])
    info = tts_input["additional_information"]

    for key, value in expected.items():
        assert info[key] == value
    if PRECOMPUTED_TEXT_IDS_KEY in expected:
        assert "text" not in info
    else:
        assert PRECOMPUTED_TEXT_IDS_KEY not in info
        assert len(tts_input["prompt_token_ids"]) >= 32


def test_aura2tts_prefers_streaming_cumulative_text():
    prompt = {
        "additional_information": {
            "tts_ref_audio": ["ref.wav"],
            "tts_ref_text": ["Reference transcript sample."],
        }
    }

    [tts_input] = aura2tts(
        [_source_delta_final_output("The complete AURA reply.")],
        prompt=[prompt],
    )

    assert tts_input["additional_information"]["text"] == ["The complete AURA reply."]


@pytest.mark.parametrize(
    "response_text",
    [SILENT_TEXT, " ﹑"],
    ids=["silent", "punctuation_only"],
)
def test_aura2tts_drops_non_spoken_response(response_text):
    assert aura2tts([_source_output(response_text)]) == []
