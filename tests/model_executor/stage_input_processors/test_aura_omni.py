# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest

from vllm_omni.model_executor.stage_input_processors.aura_session_history import (
    SessionHistory,
    clear_all_sessions,
    register_session,
)
from vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder import (
    PRECOMPUTED_TEXT_IDS_KEY,
)
from vllm_omni.model_executor.stage_input_processors.aura_omni import (
    QWEN_IM_END_ID,
    QWEN_IM_START_ID,
    QWEN_TEXT_SILENT_TOKEN_IDS,
    SILENT_TEXT,
    _clean_asr_transcript,
    _trim_aura_response_token_ids,
    asr2aura,
    asr2aura_async_chunk,
    aura2tts,
    aura2tts_async_chunk,
    pop_turn_transcript,
    video_tuple_from_additional_info,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _source_output(text: str, request_id: str = "req-1", token_ids: list[int] | None = None):
    output = SimpleNamespace(text=text, cumulative_token_ids=token_ids or [1, 2, 3], multimodal_output={})
    return SimpleNamespace(request_id=request_id, outputs=[output])


def _partial_source_output(text: str, request_id: str = "req-1", token_ids: list[int] | None = None):
    output = SimpleNamespace(text=text, cumulative_token_ids=token_ids or [1, 2, 3], multimodal_output={})
    return SimpleNamespace(request_id=request_id, outputs=[output], finished=False)


def _source_delta_final_output(cumulative_text: str, request_id: str = "req-1", token_ids: list[int] | None = None):
    output = SimpleNamespace(
        text="partial",
        cumulative_text=cumulative_text,
        cumulative_token_ids=token_ids or [1, 2, 3],
        multimodal_output={},
    )
    return SimpleNamespace(request_id=request_id, outputs=[output], finished=True)


def _transfer_manager():
    return SimpleNamespace(config=SimpleNamespace(), request_payload={})


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


def test_asr2aura_forwards_tts_options_to_aura_worker():
    prompt = {
        "multi_modal_data": {"video": ["frame-0"]},
        "additional_information": {
            "tts_task_type": ["Base"],
            "tts_ref_audio": ["voice.wav"],
            "tts_ref_text": ["hello"],
            "aura_system_prompt": ["system"],
        },
    }

    [next_input] = asr2aura([_source_output("看看视频")], prompt=[prompt])

    assert next_input["additional_information"] == {
        "tts_task_type": ["Base"],
        "tts_ref_audio": ["voice.wav"],
        "tts_ref_text": ["hello"],
    }


def test_asr2aura_drops_audio_before_qwen3_vl_stage():
    prompt = {
        "multi_modal_data": {
            "audio": ("wave", 16000),
            "video": ["frame-0", "frame-1"],
        },
    }

    [next_input] = asr2aura([_source_output("Check the video")], prompt=[prompt])

    assert next_input["multi_modal_data"] == {"video": ["frame-0", "frame-1"]}
    assert "<|video_pad|>" in next_input["prompt"]


def test_asr2aura_reads_video_stashed_for_downstream_stage():
    prompt = {
        "multi_modal_data": {"audio": ("wave", 16000)},
        "additional_information": {
            "deferred_multi_modal_data": {"video": ["frame-0", "frame-1"]},
        },
    }

    [next_input] = asr2aura([_source_output("Check the video")], prompt=[prompt])

    assert next_input["multi_modal_data"] == {"video": ["frame-0", "frame-1"]}
    assert "<|video_pad|>" in next_input["prompt"]


def test_video_tuple_from_additional_info_legacy_aura_turn_video():
    frames = [
        [[[1, 0, 0], [0, 1, 0]], [[0, 0, 1], [1, 1, 0]]],
        [[[2, 0, 0], [0, 2, 0]], [[0, 0, 2], [2, 2, 0]]],
    ]
    video_tuple = video_tuple_from_additional_info(
        {
            "aura_turn_video": {
                "frames": frames,
                "metadata": {"fps": 2.0},
            }
        }
    )
    assert video_tuple is not None
    arr, meta = video_tuple
    assert arr.shape[0] == 2
    assert meta["fps"] == 2.0


def test_asr2aura_uses_server_side_store():
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
            "deferred_multi_modal_data": {
                "video": [
                    (
                        np.array(
                            [
                                [[[3, 0, 0], [0, 3, 0]], [[0, 0, 3], [3, 3, 0]]],
                                [[[4, 0, 0], [0, 4, 0]], [[0, 0, 4], [4, 4, 0]]],
                            ],
                            dtype=np.uint8,
                        ),
                        {
                            "fps": 2.0,
                            "duration": 1.0,
                            "total_num_frames": 2,
                            "frames_indices": [0, 1],
                            "video_backend": "opencv",
                            "do_sample_frames": False,
                        },
                    )
                ],
            },
            "aura_system_prompt": ["custom system"],
        }
    }

    [next_input] = asr2aura(
        [_source_output("language Chinese<asr_text>Hello there.", request_id="video-testreq02-abcd1234")],
        prompt=[prompt],
    )

    assert "prior round" in next_input["prompt"]
    assert "Hello there." in next_input["prompt"]
    assert "language Chinese" not in next_input["prompt"]
    assert "<asr_text>" not in next_input["prompt"]
    assert pop_turn_transcript("video-testreq02") == "Hello there."
    assert len(next_input["multi_modal_data"]["video"]) == 2
    assert len(history.get_vllm_inputs()["multi_modal_data"]["video"]) == 1
    clear_all_sessions()


def test_asr2aura_supports_video_only_observation():
    prompt = {"multi_modal_data": {"video": ["frame-0", "frame-1"]}}

    [next_input] = asr2aura([_source_output("")], prompt=[prompt])

    assert "<|video_pad|>" in next_input["prompt"]
    assert "<|im_start|>assistant" in next_input["prompt"]


def test_asr2aura_async_chunk_waits_until_asr_finished(monkeypatch):
    class FakeTokenizer:
        def encode(self, text):
            return [ord(ch) for ch in text]

    monkeypatch.setattr(
        "vllm_omni.model_executor.stage_input_processors.aura_omni.cached_tokenizer_from_config",
        lambda _config: FakeTokenizer(),
    )
    transfer_manager = SimpleNamespace(config=SimpleNamespace())
    request = SimpleNamespace(
        request_id="req-1",
        external_req_id="req-1",
        output_text="看看视频<|im_end|>\n",
        additional_information={
            "aura_system_prompt": ["system"],
            "deferred_multi_modal_data": {"video": ["frame-0"]},
            "tts_ref_audio": ["voice.wav"],
        },
        is_finished=lambda: False,
    )

    assert asr2aura_async_chunk(transfer_manager, None, request, is_finished=False) is None
    request.output_text = "看看视频里有什么<|im_end|>\n"

    payload = asr2aura_async_chunk(transfer_manager, None, request, is_finished=True)

    assert "看看视频里有什么" in payload["prompt"]
    assert "<|im_end|><|im_end|>" not in payload["prompt"]
    assert payload["ids"]["prompt"] == payload["prompt_token_ids"]
    assert payload["multi_modal_data"] == {"video": ["frame-0"]}
    assert payload["additional_information"] == {"tts_ref_audio": ["voice.wav"]}


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


def test_aura2tts_supports_base_ref_audio_override():
    prompt = {
        "additional_information": {
            "tts_ref_audio": ["custom.wav"],
            "tts_ref_text": ["custom transcript"],
        }
    }

    [tts_input] = aura2tts([_source_output("Hello.")], prompt=[prompt])

    assert tts_input["additional_information"]["task_type"] == ["Base"]
    assert tts_input["additional_information"]["ref_audio"] == ["custom.wav"]
    assert tts_input["additional_information"]["ref_text"] == ["custom transcript"]
    assert tts_input["additional_information"]["x_vector_only_mode"] == [False]


def test_aura2tts_supports_x_vector_only_mode_for_base():
    prompt = {
        "additional_information": {
            "tts_task_type": ["Base"],
            "tts_x_vector_only_mode": [True],
            "tts_ref_audio": ["ref.wav"],
            "tts_ref_text": ["Reference transcript sample."],
        }
    }

    [tts_input] = aura2tts([_source_output("Hello.")], prompt=[prompt])

    assert tts_input["additional_information"]["x_vector_only_mode"] == [True]


def test_aura2tts_supports_custom_voice_mode():
    prompt = {
        "additional_information": {
            "tts_task_type": ["CustomVoice"],
            "tts_speaker": ["vivian"],
        }
    }

    [tts_input] = aura2tts([_source_output("Hello.")], prompt=[prompt])

    assert tts_input["additional_information"]["task_type"] == ["CustomVoice"]
    assert tts_input["additional_information"]["speaker"] == ["Vivian"]
    assert "ref_audio" not in tts_input["additional_information"]
    assert len(tts_input["prompt_token_ids"]) == 14


def test_aura2tts_passes_token_ids_to_qwen3_tts_when_enabled():
    prompt = {
        "additional_information": {
            "tts_ref_audio": ["ref.wav"],
            "tts_ref_text": ["Reference transcript sample."],
            "tts_pass_token_ids": [True],
        }
    }

    [tts_input] = aura2tts(
        [
            _source_output(
                "Hello.",
                token_ids=[151644, 77091, 198, 108386, 1773, 151645, 198],
            )
        ],
        prompt=[prompt],
    )

    text_ids = tts_input["additional_information"][PRECOMPUTED_TEXT_IDS_KEY][0]
    assert 108386 in text_ids
    assert 1773 in text_ids
    assert "text" not in tts_input["additional_information"]


def test_aura2tts_async_chunk_waits_until_aura_finished():
    transfer_manager = _transfer_manager()
    request = SimpleNamespace(
        request_id="req-1",
        external_req_id="req-1",
        output_token_ids=[108386, 1773],
        additional_information={
            "tts_ref_audio": ["custom.wav"],
            "tts_ref_text": ["custom transcript"],
        },
        is_finished=lambda: False,
    )

    assert aura2tts_async_chunk(transfer_manager, None, request) is None
    payload = aura2tts_async_chunk(transfer_manager, None, request, is_finished=True)

    text_ids = payload[PRECOMPUTED_TEXT_IDS_KEY][0]
    assert 108386 in text_ids
    assert 1773 in text_ids
    assert payload["ref_audio"] == ["custom.wav"]
    assert payload["ref_text"] == ["custom transcript"]
    assert payload["task_type"] == ["Base"]
    assert payload["prompt_token_ids"]
    assert "next_stage_prompt_len" not in payload


def test_aura2tts_async_chunk_reads_nested_request_additional_information():
    transfer_manager = _transfer_manager()
    request = SimpleNamespace(
        request_id="req-1",
        external_req_id="req-1",
        output_token_ids=[108386, 1773],
        additional_information={
            "additional_information": {
                "tts_task_type": ["CustomVoice"],
                "tts_speaker": ["vivian"],
                "tts_language": ["Chinese"],
            }
        },
        is_finished=lambda: False,
    )

    assert aura2tts_async_chunk(transfer_manager, None, request) is None
    payload = aura2tts_async_chunk(transfer_manager, None, request, is_finished=True)

    assert payload["task_type"] == ["CustomVoice"]
    assert payload["speaker"] == ["Vivian"]
    assert payload["language"] == ["Chinese"]
    assert payload["prompt_token_ids"]
    assert "ref_audio" not in payload
    assert "ref_text" not in payload


def test_aura2tts_async_chunk_keeps_tts_metadata_when_request_info_is_cleared():
    transfer_manager = _transfer_manager()
    request = SimpleNamespace(
        request_id="req-1",
        external_req_id="req-1",
        output_token_ids=[108386, 1773],
        additional_information={
            "additional_information": {
                "tts_task_type": ["CustomVoice"],
                "tts_speaker": ["vivian"],
                "tts_language": ["Chinese"],
            }
        },
        is_finished=lambda: False,
    )

    assert aura2tts_async_chunk(transfer_manager, None, request, is_finished=False) is None
    request.additional_information = {}
    payload = aura2tts_async_chunk(transfer_manager, None, request, is_finished=True)

    assert payload["task_type"] == ["CustomVoice"]
    assert payload["speaker"] == ["Vivian"]
    assert payload["language"] == ["Chinese"]
    assert "ref_audio" not in payload


def test_aura2tts_async_chunk_reads_tts_metadata_from_stage_payload_when_request_info_is_cleared():
    transfer_manager = _transfer_manager()
    request = SimpleNamespace(
        request_id="req-1",
        external_req_id="req-1",
        output_token_ids=[108386, 1773],
        additional_information={},
        omni_stage_payload={
            "prompt": "aura prompt",
            "additional_information": {
                "tts_task_type": ["CustomVoice"],
                "tts_speaker": ["vivian"],
                "tts_language": ["Chinese"],
            },
        },
        is_finished=lambda: False,
    )

    payload = aura2tts_async_chunk(transfer_manager, None, request, is_finished=True)

    assert payload["task_type"] == ["CustomVoice"]
    assert payload["speaker"] == ["Vivian"]
    assert payload["language"] == ["Chinese"]
    assert "ref_audio" not in payload


def test_aura2tts_async_chunk_accumulates_and_sends_full_text_once_finished():
    transfer_manager = _transfer_manager()
    request = SimpleNamespace(
        request_id="req-1",
        external_req_id="req-1",
        output_token_ids=[108386, 1773],
        output_text="你好",
        additional_information={
            "tts_ref_audio": ["custom.wav"],
            "tts_ref_text": ["custom transcript"],
        },
        is_finished=lambda: False,
    )

    assert aura2tts_async_chunk(transfer_manager, None, request) is None
    request.output_token_ids = [108386, 1773, 104139]
    request.output_text = "你好，世界"
    payload = aura2tts_async_chunk(transfer_manager, None, request, is_finished=True)

    assert payload["text"] == ["你好，世界"]
    assert payload["task_type"] == ["Base"]
    assert payload["ref_audio"] == ["custom.wav"]
    assert payload["prompt_token_ids"]


def test_aura2tts_async_chunk_passes_token_ids_only_when_enabled():
    transfer_manager = _transfer_manager()
    request = SimpleNamespace(
        request_id="req-1",
        external_req_id="req-1",
        output_token_ids=[108386, 1773],
        output_text="你好",
        additional_information={
            "tts_ref_audio": ["custom.wav"],
            "tts_ref_text": ["custom transcript"],
            "tts_pass_token_ids": [True],
        },
        is_finished=lambda: False,
    )

    assert aura2tts_async_chunk(transfer_manager, None, request) is None
    request.output_token_ids = [108386, 1773, 104139]
    payload = aura2tts_async_chunk(transfer_manager, None, request, is_finished=True)

    assert PRECOMPUTED_TEXT_IDS_KEY in payload
    assert "text" not in payload
    assert payload["task_type"] == ["Base"]
    assert 104139 in payload[PRECOMPUTED_TEXT_IDS_KEY][0]


def test_aura2tts_async_chunk_decodes_text_instead_of_passing_source_token_ids(monkeypatch):
    class FakeTokenizer:
        def decode(self, token_ids):
            assert token_ids == [101, 102, 103, 104]
            return "第一句\n\n第二句"

    monkeypatch.setattr(
        "vllm_omni.model_executor.stage_input_processors.aura_omni.cached_tokenizer_from_config",
        lambda config: FakeTokenizer(),
    )
    transfer_manager = _transfer_manager()
    transfer_manager.config = SimpleNamespace()
    request = SimpleNamespace(
        request_id="req-1",
        external_req_id="req-1",
        output_token_ids=[101, 102, 103, 104],
        additional_information={
            "tts_task_type": ["CustomVoice"],
            "tts_speaker": ["Vivian"],
            "tts_language": ["Chinese"],
        },
        is_finished=lambda: False,
    )

    payload = aura2tts_async_chunk(transfer_manager, None, request, is_finished=True)

    assert payload["text"] == ["第一句 第二句"]
    assert payload["task_type"] == ["CustomVoice"]
    assert PRECOMPUTED_TEXT_IDS_KEY not in payload
    assert payload["prompt_token_ids"]


def test_aura2tts_async_chunk_holds_silent_token_prefix():
    request = SimpleNamespace(
        request_id="req-1",
        output_token_ids=[151669],
        additional_information={},
        is_finished=lambda: False,
    )

    assert aura2tts_async_chunk(None, None, request) is None


def test_trim_aura_response_token_ids_keeps_all_speakable_segments():
    first_sentence = [99692, 3837, 105351, 108519, 3837]
    second_sentence = [106040, 18493, 108141, 71817, 105465]
    token_ids = [
        *first_sentence,
        QWEN_IM_END_ID,
        QWEN_IM_START_ID,
        *second_sentence,
        *QWEN_TEXT_SILENT_TOKEN_IDS,
        QWEN_IM_END_ID,
    ]

    speakable = _trim_aura_response_token_ids(token_ids)

    assert speakable[: len(first_sentence)] == first_sentence
    assert speakable[len(first_sentence) :] == second_sentence
    assert QWEN_IM_END_ID not in speakable
    assert QWEN_IM_START_ID not in speakable
    assert all(token_id not in speakable for token_id in QWEN_TEXT_SILENT_TOKEN_IDS)


def test_aura2tts_drops_silent_response():
    assert aura2tts([_source_output(SILENT_TEXT)]) == []


def test_aura2tts_holds_partial_silent_prefix():
    assert aura2tts([_partial_source_output("<|sil")]) == []


def test_aura2tts_streaming_partial_content_enters_tts():
    prompt = {
        "additional_information": {
            "tts_ref_audio": ["ref.wav"],
            "tts_ref_text": ["Reference transcript sample."],
        }
    }

    [tts_input] = aura2tts(
        [_partial_source_output("你好", token_ids=[151644, 77091, 198, 108386])],
        prompt=[prompt],
    )

    assert tts_input["additional_information"]["text"] == ["你好"]
    assert PRECOMPUTED_TEXT_IDS_KEY not in tts_input["additional_information"]


def test_aura2tts_drops_punctuation_only_filler():
    assert aura2tts([_source_output(" ﹑")]) == []
