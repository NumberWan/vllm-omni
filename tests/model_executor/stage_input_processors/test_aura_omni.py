# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder import (
    PRECOMPUTED_TEXT_IDS_KEY,
)
from vllm_omni.model_executor.stage_input_processors.aura_omni import (
    SILENT_TEXT,
    asr2aura,
    asr2aura_async_chunk,
    aura2tts,
    aura2tts_async_chunk,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _source_output(text: str, request_id: str = "req-1", token_ids: list[int] | None = None):
    output = SimpleNamespace(text=text, cumulative_token_ids=token_ids or [1, 2, 3], multimodal_output={})
    return SimpleNamespace(request_id=request_id, outputs=[output])


def _partial_source_output(text: str, request_id: str = "req-1", token_ids: list[int] | None = None):
    output = SimpleNamespace(text=text, cumulative_token_ids=token_ids or [1, 2, 3], multimodal_output={})
    return SimpleNamespace(request_id=request_id, outputs=[output], finished=False)


def test_asr2aura_carries_video_payload_and_transcript():
    prompt = {
        "multi_modal_data": {"video": ["frame-0", "frame-1"]},
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

    assert payload["prompt"]
    assert "看看视频里有什么" in payload["prompt"]
    assert "<|im_end|><|im_end|>" not in payload["prompt"]
    assert payload["prompt_token_ids"]
    assert payload["ids"]["prompt"] == payload["prompt_token_ids"]
    assert payload["multi_modal_data"] == {"video": ["frame-0"]}
    assert payload["additional_information"] == {"tts_ref_audio": ["voice.wav"]}


def test_aura2tts_builds_qwen3_tts_prompt_information():
    prompt = {
        "additional_information": {
            "tts_language": ["Chinese"],
            "tts_instruct": ["Calm voice."],
            "tts_ref_audio": ["ref.wav"],
            "tts_ref_text": ["Reference transcript sample."],
        }
    }

    [tts_input] = aura2tts([_source_output("Hello.")], prompt=[prompt])

    assert len(tts_input["prompt_token_ids"]) > 0
    assert tts_input["additional_information"]["text"] == ["Hello."]
    assert PRECOMPUTED_TEXT_IDS_KEY not in tts_input["additional_information"]
    assert tts_input["additional_information"]["task_type"] == ["Base"]
    assert tts_input["additional_information"]["language"] == ["Chinese"]
    assert tts_input["additional_information"]["ref_audio"] == ["ref.wav"]
    assert tts_input["additional_information"]["ref_text"] == ["Reference transcript sample."]
    assert tts_input["additional_information"]["x_vector_only_mode"] == [False]
    assert tts_input["additional_information"]["instruct"] == ["Calm voice."]


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

    assert tts_input["additional_information"][PRECOMPUTED_TEXT_IDS_KEY] == [
        [151644, 77091, 198, 108386, 1773, 151645, 198, 151644, 77091, 198]
    ]
    assert "text" not in tts_input["additional_information"]


def test_aura2tts_async_chunk_packs_tts_conditioning_in_connector_payload():
    transfer_manager = SimpleNamespace()
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

    payload = aura2tts_async_chunk(transfer_manager, None, request)

    assert payload[PRECOMPUTED_TEXT_IDS_KEY] == [
        [151644, 77091, 198, 108386, 1773, 151645, 198, 151644, 77091, 198]
    ]
    assert payload["ref_audio"] == ["custom.wav"]
    assert payload["ref_text"] == ["custom transcript"]
    assert payload["task_type"] == ["Base"]
    assert payload["prompt_token_ids"]
    assert "next_stage_prompt_len" not in payload


def test_aura2tts_async_chunk_accepts_multimodal_output_keyword():
    transfer_manager = SimpleNamespace()
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

    payload = aura2tts_async_chunk(
        transfer_manager=transfer_manager,
        multimodal_output={},
        request=request,
    )

    assert payload["text"] == ["你好"]
    assert payload["task_type"] == ["Base"]


def test_aura2tts_async_chunk_emits_updated_text_ids_for_non_final_chunks():
    transfer_manager = SimpleNamespace()
    request = SimpleNamespace(
        request_id="req-1",
        external_req_id="req-1",
        output_token_ids=[108386, 1773],
        output_text="第一句",
        additional_information={"tts_ref_audio": ["custom.wav"], "tts_ref_text": ["custom transcript"]},
        is_finished=lambda: False,
    )

    first_payload = aura2tts_async_chunk(transfer_manager, None, request)
    request.output_token_ids = [108386, 1773, 104139]
    request.output_text = "第一句，第二句"
    second_payload = aura2tts_async_chunk(transfer_manager, None, request)

    assert first_payload["text"] == ["第一句"]
    assert first_payload["task_type"] == ["Base"]
    assert first_payload["ref_audio"] == ["custom.wav"]
    assert first_payload["prompt_token_ids"]
    assert "_reuse_tts_info" not in first_payload
    assert second_payload["text"] == ["，第二句"]
    assert second_payload["_reuse_tts_info"] is True
    assert "task_type" not in second_payload
    assert "ref_audio" not in second_payload
    assert "next_stage_prompt_len" not in second_payload


def test_aura2tts_async_chunk_passes_token_ids_only_when_enabled():
    transfer_manager = SimpleNamespace()
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

    first_payload = aura2tts_async_chunk(transfer_manager, None, request)
    request.output_token_ids = [108386, 1773, 104139]
    second_payload = aura2tts_async_chunk(transfer_manager, None, request)

    assert PRECOMPUTED_TEXT_IDS_KEY in first_payload
    assert "text" not in first_payload
    assert first_payload["task_type"] == ["Base"]
    assert second_payload[PRECOMPUTED_TEXT_IDS_KEY] == [
        [151644, 77091, 198, 104139, 151645, 198, 151644, 77091, 198]
    ]
    assert second_payload["_reuse_tts_info"] is True
    assert "task_type" not in second_payload


def test_aura2tts_async_chunk_holds_silent_token_prefix():
    request = SimpleNamespace(
        request_id="req-1",
        output_token_ids=[151669],
        additional_information={},
        is_finished=lambda: False,
    )

    assert aura2tts_async_chunk(None, None, request) is None


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
