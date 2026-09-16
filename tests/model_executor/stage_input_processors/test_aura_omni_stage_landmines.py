# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Per-stage landmine matrix for AURA Omni streaming (CPU-only).

Pipeline: Stage0 ASR → Stage1 AURA → Stage2 Talker → Stage3 Code2Wav.

Complements ``test_aura_omni_internal_stages.py`` and
``test_chunk_transfer_adapter.py`` with an explicit Stage0→3 checklist of
contracts that historically crashed Smoke3 before / without GPU work.
"""

from __future__ import annotations

import threading
from collections import defaultdict, deque
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from pytest_mock import MockerFixture
from vllm.v1.request import RequestStatus

from vllm_omni.data_entry_keys import MetaStruct
from vllm_omni.distributed.omni_connectors.adapter import (
    _coerce_payload_meta,
    construct_next_stage_streaming_input_prompt,
)
from vllm_omni.distributed.omni_connectors.transfer_adapter.base import OmniTransferAdapterBase
from vllm_omni.distributed.omni_connectors.transfer_adapter.chunk_transfer_adapter import (
    OmniChunkTransferAdapter,
    _apply_max_new_tokens_from_payload,
    _meta_to_dict,
    _payload_has_talker_conditioning,
)
from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_code2wav import Qwen3TTSCode2Wav
from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_talker import (
    _has_tts_text_conditioning,
)
from vllm_omni.model_executor.stage_input_processors.aura_omni import (
    AURA_VIDEO_WIRE_MARKER,
    asr2aura_async_chunk,
    aura2tts_async_chunk,
    resolve_aura_async_chunk_stage_payload,
    unpack_aura_video_ndarray,
    _aura2tts_empty_finished_payload,
)
from vllm_omni.model_executor.stage_input_processors.aura_session_history import (
    clear_all_sessions,
)
from vllm_omni.model_executor.stage_input_processors.qwen3_tts import (
    talker2code2wav_async_chunk,
    _qwen3_tts_degenerate_finished_payload,
)
from vllm_omni.model_executor.stage_input_processors.stage_bypass import (
    build_empty_asr_aura_chunk_payload,
)
from vllm_omni.worker.gpu_generation_model_runner import (
    ExecuteModelState,
    GPUGenerationModelRunner,
)
from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_TTS_INFO = {
    "tts_task_type": ["CustomVoice"],
    "tts_speaker": ["Vivian"],
    "tts_language": ["Chinese"],
}


def _asr_tm():
    return SimpleNamespace(config=SimpleNamespace(), request_payload={})


def _aura_request(*, rid: str, text: str, finished: bool, token_ids: list[int] | None = None, **extra):
    return SimpleNamespace(
        request_id=rid,
        external_req_id=rid,
        output_token_ids=token_ids or [1, 2, 3],
        output_text=text,
        additional_information={**_TTS_INFO, **extra},
        is_finished=lambda: finished,
    )


@pytest.fixture
def build_adapter(monkeypatch, mocker: MockerFixture):
    def _build(*, stage_id: int = 2, model_mode: str = "ar"):
        connector = mocker.MagicMock()
        connector.stage_id = stage_id
        connector.config = {"extra": {}}
        connector.get.return_value = None
        connector.put.return_value = (True, 1, {})

        def _fake_base_init(self, config):
            self.config = config
            self._pending_load_reqs = deque()
            self._finished_load_reqs = set()
            self._cancelled_load_reqs = set()
            self._pending_save_reqs = deque()
            self._finished_save_reqs = set()
            self.stop_event = threading.Event()
            self._recv_cond = threading.Condition()
            self._save_cond = threading.Condition()

        monkeypatch.setattr(OmniTransferAdapterBase, "__init__", _fake_base_init)
        monkeypatch.setattr(
            OmniChunkTransferAdapter,
            "create_connector",
            classmethod(lambda cls, _model_config: connector),
        )
        model_config = SimpleNamespace(
            worker_type=model_mode,
            max_num_seqs=1,
            active_stream_window=0,
            stage_connector_config={"name": "SharedMemoryConnector", "extra": {}},
        )
        adapter = OmniChunkTransferAdapter(
            SimpleNamespace(
                model_config=model_config,
                scheduler_config=SimpleNamespace(max_num_seqs=1),
            )
        )
        return adapter, connector

    return _build


def _talker_req(
    rid: str,
    status: RequestStatus,
    *,
    computed: int,
    prompt_len: int,
    max_tokens: int,
    resumable: bool = True,
):
    generated = max(0, computed - prompt_len)
    return SimpleNamespace(
        request_id=rid,
        external_req_id=rid,
        status=status,
        resumable=resumable,
        prompt_token_ids=[0] * prompt_len,
        num_prompt_tokens=prompt_len,
        num_computed_tokens=computed,
        num_output_placeholders=0,
        prefill_stats=None,
        max_tokens=max_tokens,
        sampling_params=SimpleNamespace(max_tokens=max_tokens),
        _output_token_ids=[1] * generated,
        _all_token_ids=[0] * prompt_len + [1] * generated,
        additional_information=None,
        update_block_hashes=lambda: None,
        is_finished=lambda: False,
    )


# =============================================================================
# Stage 0 — ASR
# =============================================================================


def test_s0_landmine_unfinished_holds_emit(monkeypatch):
    monkeypatch.setattr(
        "vllm_omni.model_executor.stage_input_processors.aura_omni.cached_tokenizer_from_config",
        lambda _config: SimpleNamespace(encode=lambda text: [1], decode=lambda ids: ""),
    )
    req = SimpleNamespace(
        request_id="s0-hold",
        external_req_id="s0-hold",
        output_text="partial",
        additional_information={"aura_system_prompt": ["sys"]},
        is_finished=lambda: False,
    )
    assert asr2aura_async_chunk(_asr_tm(), None, req, is_finished=False) is None


def test_s0_landmine_finished_carries_transcript(monkeypatch):
    monkeypatch.setattr(
        "vllm_omni.model_executor.stage_input_processors.aura_omni.cached_tokenizer_from_config",
        lambda _config: SimpleNamespace(encode=lambda text: [1], decode=lambda ids: ""),
    )
    req = SimpleNamespace(
        request_id="s0-ok",
        external_req_id="s0-ok",
        output_text="简单介绍一下",
        additional_information={"aura_system_prompt": ["sys"]},
        is_finished=lambda: True,
    )
    payload = asr2aura_async_chunk(_asr_tm(), None, req, is_finished=True)
    assert payload["aura_asr_transcript"] == "简单介绍一下"
    assert payload["additional_information"]["aura_system_prompt"] == ["sys"]


def test_s0_landmine_video_wire_is_bytes_not_nested_ints(monkeypatch):
    """tolist() of uint8 frames used to inflate Stage0→1 IPC ~3×."""
    monkeypatch.setattr(
        "vllm_omni.model_executor.stage_input_processors.aura_omni.cached_tokenizer_from_config",
        lambda _config: SimpleNamespace(encode=lambda text: [1], decode=lambda ids: ""),
    )
    frames = np.arange(2 * 4 * 4 * 3, dtype=np.uint8).reshape(2, 4, 4, 3)
    req = SimpleNamespace(
        request_id="s0-vid",
        external_req_id="s0-vid",
        output_text="看看视频",
        additional_information={
            "aura_system_prompt": ["sys"],
            "deferred_multi_modal_data": {
                "video": [(frames, {"fps": 2.0, "total_num_frames": 2})]
            },
        },
        is_finished=lambda: True,
        multi_modal_data=None,
        mm_processor_kwargs=None,
    )
    payload = asr2aura_async_chunk(_asr_tm(), None, req, is_finished=True)
    wire = payload["aura_turn_video"]["frames"]
    assert wire[AURA_VIDEO_WIRE_MARKER] is True
    assert isinstance(wire["data"], (bytes, bytearray))
    restored = unpack_aura_video_ndarray(wire)
    assert restored is not None
    assert restored.shape == (2, 4, 4, 3)


def test_s0_landmine_empty_vad_bypass_is_not_talker_conditioning():
    payload = build_empty_asr_aura_chunk_payload({"aura_system_prompt": ["sys"]})
    assert payload["aura_asr_transcript"] == ""
    assert bool(payload["meta"]["finished"].item()) is True
    assert _payload_has_talker_conditioning(payload) is False


def test_s0_landmine_stage1_resolve_builds_prompt_from_asr():
    clear_all_sessions()
    payload = {
        "aura_asr_transcript": "简单介绍一下桌上的东西",
        "additional_information": {
            "aura_session_id": "s0-resolve",
            "aura_system_prompt": ["system"],
        },
    }
    request = SimpleNamespace(
        request_id="s0-resolve",
        external_req_id="s0-resolve",
        additional_information=None,
        omni_stage_payload=None,
    )

    class _Tok:
        def encode(self, prompt: str) -> list[int]:
            assert "简单介绍一下桌上的东西" in prompt
            return [10, 20, 30]

    import vllm_omni.model_executor.stage_input_processors.aura_omni as aura_mod

    original = aura_mod.cached_tokenizer_from_config
    aura_mod.cached_tokenizer_from_config = lambda _cfg: _Tok()
    try:
        resolve_aura_async_chunk_stage_payload(payload, request, SimpleNamespace())
    finally:
        aura_mod.cached_tokenizer_from_config = original

    assert payload["prompt_token_ids"] == [10, 20, 30]


# =============================================================================
# Stage 1 — AURA → Talker
# =============================================================================


def test_s1_landmine_sentence_emit_is_talker_ready(monkeypatch):
    monkeypatch.setenv("VLLM_AURA_SENTENCE_TTS", "1")
    monkeypatch.setenv("VLLM_AURA_SENTENCE_TTS_MIN_CHARS", "4")
    clear_all_sessions()
    req = _aura_request(rid="s1-sent", text="今天天气很好。后面还有", finished=False)
    payload = aura2tts_async_chunk(_asr_tm(), None, req, is_finished=False)
    assert payload is not None
    assert _payload_has_talker_conditioning(payload) is True
    assert payload["text"] == ["今天天气很好。"]
    assert payload["prompt_token_ids"]
    assert bool(payload["meta"]["finished"].item()) is False


def test_s1_landmine_empty_finish_sentinel_has_no_conditioning():
    """Silent / EOS-with-nothing — crashed Stage2 if applied as new_segment."""
    payload = _aura2tts_empty_finished_payload()
    assert payload["prompt_token_ids"] == []
    assert bool(payload["meta"]["finished"].item()) is True
    assert _payload_has_talker_conditioning(payload) is False
    assert _has_tts_text_conditioning(payload) is False


def test_s1_landmine_v2_silent_token_emits_empty_finish(monkeypatch):
    monkeypatch.setattr(
        "vllm_omni.model_executor.stage_input_processors.aura_omni.AURA_SILENT_TOKEN_IDS",
        [248070],
    )
    clear_all_sessions()
    req = _aura_request(rid="s1-silent", text="", finished=True, token_ids=[248070])
    payload = aura2tts_async_chunk(_asr_tm(), None, req, is_finished=True)
    assert payload is not None
    assert payload["prompt_token_ids"] == []
    assert bool(payload["meta"]["finished"].item()) is True
    assert _payload_has_talker_conditioning(payload) is False


def test_s1_landmine_finished_leftover_flushes_final_sentence(monkeypatch):
    monkeypatch.setenv("VLLM_AURA_SENTENCE_TTS", "1")
    monkeypatch.setenv("VLLM_AURA_SENTENCE_TTS_MIN_CHARS", "4")
    clear_all_sessions()
    req = _aura_request(rid="s1-final", text="最后一句", finished=True)
    payload = aura2tts_async_chunk(_asr_tm(), None, req, is_finished=True)
    assert payload is not None
    assert bool(payload["meta"]["finished"].item()) is True
    # Either a speakable remnant or empty sentinel — both legal; never partial hold.
    if _payload_has_talker_conditioning(payload):
        assert "最后一句" in (payload.get("text") or [""])[0]
    else:
        assert payload["prompt_token_ids"] == []


# =============================================================================
# Stage 2 — Talker
# =============================================================================


def test_s2_landmine_empty_finish_while_waiting_skips_new_segment(build_adapter):
    adapter, _ = build_adapter(stage_id=2, model_mode="ar")
    req = _talker_req("s2-ef", RequestStatus.WAITING, computed=0, prompt_len=22, max_tokens=4096)
    last_hidden = torch.ones(4)
    req.additional_information = {
        "text": ["第一句。"],
        "hidden_states": {"last": last_hidden},
    }
    adapter._pending_upstream_payloads["s2-ef"] = deque(
        [
            {
                "payload": _aura2tts_empty_finished_payload(),
                "finished": True,
                "segment_finished": False,
                "chunk_id": 2,
            }
        ]
    )
    assert adapter.try_apply_pending_upstream_payload(req) is True
    assert req.additional_information["text"] == ["第一句。"]
    assert req.additional_information["hidden_states"]["last"] is last_hidden
    assert "s2-ef" in adapter.upstream_exhausted_requests
    assert req.resumable is False


def test_s2_landmine_late_payload_queues_while_running_then_drains(build_adapter):
    adapter, connector = build_adapter(stage_id=2, model_mode="ar")
    req = _talker_req(
        "s2-q",
        RequestStatus.RUNNING,
        computed=22 + 50,
        prompt_len=22,
        max_tokens=143,
    )
    last_hidden = torch.ones(4)
    req.additional_information = {
        "text": ["第一句。"],
        "hidden_states": {"last": last_hidden},
    }
    late = {
        "prompt_token_ids": [0] * 23,
        "max_new_tokens": 73,
        "text": ["下一句。"],
        "meta": {"finished": torch.tensor(False, dtype=torch.bool)},
    }
    connector.get.return_value = (late, 16)
    assert adapter._poll_single_request(req) is False
    assert req.additional_information["text"] == ["第一句。"]
    assert req.additional_information["hidden_states"]["last"] is last_hidden
    assert req.max_tokens == 143
    assert "s2-q" in adapter._pending_upstream_payloads

    req.status = RequestStatus.WAITING
    connector.get.return_value = None
    assert adapter._poll_single_request(req) is True
    assert req.additional_information["text"] == ["下一句。"]
    assert req.num_computed_tokens == 0


def test_s2_landmine_mid_decode_max_tokens_clamp_raises():
    req = _talker_req(
        "s2-clamp",
        RequestStatus.RUNNING,
        computed=22 + 54,
        prompt_len=22,
        max_tokens=143,
    )
    with pytest.raises(RuntimeError, match="Refuse max_new_tokens"):
        _apply_max_new_tokens_from_payload(req, {"max_new_tokens": 40})


def test_s2_landmine_metastruct_replace_flag_coerces():
    meta = MetaStruct(
        finished=None,
        replace_streaming_prompt=True,
        next_stage_prompt_len=23,
    )
    as_dict = _meta_to_dict(meta)
    assert as_dict["replace_streaming_prompt"] is True
    assert as_dict["next_stage_prompt_len"] == 23
    coerced = _coerce_payload_meta(meta)
    assert coerced["replace_streaming_prompt"] is True


def test_s2_landmine_construct_replace_with_metastruct(mocker: MockerFixture):
    request = SimpleNamespace(
        request_id="s2-meta",
        _all_token_ids=[0] * 22 + [1] * 50,
        _output_token_ids=[1] * 50,
        prompt_token_ids=[0] * 22,
        num_computed_tokens=72,
        num_prompt_tokens=22,
        update_block_hashes=mocker.Mock(),
    )
    payload = {
        "ids": {"prompt": [0] * 23},
        "meta": MetaStruct(
            finished=None,
            replace_streaming_prompt=True,
            next_stage_prompt_len=23,
        ),
    }
    construct_next_stage_streaming_input_prompt(payload, request)
    assert request.num_computed_tokens == 0
    assert request.num_prompt_tokens == 23
    request.update_block_hashes.assert_called_once_with()


def test_s2_landmine_gpu_buffer_wipes_only_on_replace():
    class _FakeRunner:
        def __init__(self):
            self.requests = {"req": object()}
            self.model_intermediate_buffer = {
                "req": {"hidden_states": {"last": torch.ones(2, 4)}, "text": ["旧句。"]}
            }

        def _is_fresh_chunk_payload(self, payload_info):
            return OmniGPUModelRunner._is_fresh_chunk_payload(payload_info)

        def _update_intermediate_buffer(self, req_id, payload_info):
            buf = self.model_intermediate_buffer.setdefault(req_id, {})
            buf.update(payload_info)

        def _set_or_update_intermediate_buffer(self, req_id, payload_info):
            return OmniGPUModelRunner._set_or_update_intermediate_buffer(
                self, req_id, payload_info
            )

    keep = _FakeRunner()
    keep._set_or_update_intermediate_buffer(
        "req",
        {"prompt_token_ids": [0] * 23, "text": ["下一句。"], "meta": {"finished": True}},
    )
    assert "last" in keep.model_intermediate_buffer["req"]["hidden_states"]

    wipe = _FakeRunner()
    wipe._set_or_update_intermediate_buffer(
        "req",
        {
            "prompt_token_ids": [0] * 23,
            "text": ["下一句。"],
            "meta": {"replace_streaming_prompt": True, "next_stage_prompt_len": 23},
        },
    )
    assert "last" not in wipe.model_intermediate_buffer["req"].get("hidden_states", {})


# =============================================================================
# Stage 3 — Code2Wav
# =============================================================================


def test_s3_landmine_talker_emits_ic8_then_finish_codes():
    extra = {
        "codec_chunk_frames": 20,
        "codec_left_context_frames": 72,
        "initial_codec_chunk_frames": 8,
    }
    tm = SimpleNamespace(
        code_prompt_token_ids=defaultdict(list),
        scheduler_max_num_seqs=1,
        put_req_chunk=defaultdict(int),
        ramp_chunk_count=defaultdict(int),
        request_payload={},
        connector=SimpleNamespace(config={"extra": extra}),
    )
    rid = "s3-ic8"
    frame = [1, 2, 3, 4]
    req = SimpleNamespace(
        external_req_id=rid,
        is_finished=lambda: False,
        additional_information=None,
    )
    tm.code_prompt_token_ids[rid] = [frame[:] for _ in range(8)]
    first = talker2code2wav_async_chunk(
        transfer_manager=tm,
        multimodal_output={"codes": {"audio": torch.zeros((0,))}},
        request=req,
        is_finished=False,
    )
    assert first is not None
    assert bool(first.meta.finished.item()) is False

    tm.code_prompt_token_ids[rid] = [frame[:] for _ in range(12)]
    req.is_finished = lambda: True
    fin = talker2code2wav_async_chunk(
        transfer_manager=tm,
        multimodal_output={"codes": {"audio": torch.zeros((0,))}},
        request=req,
        is_finished=True,
    )
    assert fin is not None
    assert bool(fin.meta.finished.item()) is True


def test_s3_landmine_degenerate_finish_has_no_conditioning():
    payload = _qwen3_tts_degenerate_finished_payload()
    assert _payload_has_talker_conditioning(payload) is False
    assert bool(payload["meta"]["finished"].item()) is True


def test_s3_landmine_empty_silent_batch_one_wav_slot_per_request():
    from unittest.mock import patch

    dec_config = SimpleNamespace(num_quantizers=2, sliding_window=0)
    tok_config = SimpleNamespace(decoder_config=dec_config, output_sample_rate=24000)

    class _FakeDecoder(torch.nn.Module):
        total_upsample = 4

        def to(self, *args, **kwargs):
            return self

        def chunked_decode(self, codes, **kwargs):
            raise AssertionError("silent empty batch must not decode codes")

    with (
        patch(
            "vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_code2wav.Qwen3TTSTokenizerV2Config.from_pretrained",
            return_value=tok_config,
        ),
        patch(
            "vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_code2wav.Qwen3TTSTokenizerV2Decoder._from_config",
            return_value=_FakeDecoder(),
        ),
    ):
        model = Qwen3TTSCode2Wav(
            vllm_config=SimpleNamespace(
                load_config=SimpleNamespace(),
                model_config=SimpleNamespace(
                    model="unused",
                    revision=None,
                    stage_connector_config=None,
                    async_chunk=True,
                ),
                device_config=SimpleNamespace(device=torch.device("cpu")),
            )
        )

    out = model.forward(
        input_ids=None,
        runtime_additional_information=[
            {"meta": {"finished": torch.tensor(True)}} for _ in range(4)
        ],
    )
    assert len(out.multimodal_outputs["model_outputs"]) == 4
    assert all(audio.numel() == 0 for audio in out.multimodal_outputs["model_outputs"])


def test_s3_landmine_runner_broadcasts_empty_outputs_to_batch():
    runner = object.__new__(GPUGenerationModelRunner)
    runner.execute_model_state = ExecuteModelState(
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        {"model_outputs": [torch.zeros(0)], "sr": [torch.tensor(24000)]},
        None,
    )
    runner.kv_connector_output = None
    runner.input_batch = SimpleNamespace(
        req_ids=[f"s3-{i}" for i in range(4)],
        req_id_to_index={f"s3-{i}": i for i in range(4)},
        num_reqs=4,
        vocab_size=10,
    )
    runner.use_async_scheduling = False
    runner.device = torch.device("cpu")
    runner.supports_mm_inputs = False
    runner.speculative_config = None
    runner.routed_experts_initialized = False
    runner._async_chunk = False

    output = GPUGenerationModelRunner.sample_tokens(runner)
    assert output is not None
