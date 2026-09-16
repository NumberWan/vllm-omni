# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU contracts for AURA-internal stages only (not Native full-duplex).

Pipeline: Stage0 ASR → Stage1 AURA → Stage2 Talker → Stage3 Code2Wav.
These tests do not load weights. They catch payload / wait-gate / batch-shape
landmines that used to crash Smoke3 before GPU work starts.
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

from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler
from vllm_omni.distributed.omni_connectors.transfer_adapter.base import OmniTransferAdapterBase
from vllm_omni.distributed.omni_connectors.transfer_adapter.chunk_transfer_adapter import (
    OmniChunkTransferAdapter,
    _apply_max_new_tokens_from_payload,
    _payload_has_talker_conditioning,
)
from vllm_omni.model_executor.stage_input_processors.aura_omni import (
    AURA_VIDEO_WIRE_MARKER,
    asr2aura_async_chunk,
    aura2tts_async_chunk,
    resolve_aura_async_chunk_stage_payload,
    unpack_aura_video_ndarray,
)
from vllm_omni.model_executor.stage_input_processors.aura_session_history import clear_all_sessions
from vllm_omni.model_executor.stage_input_processors.qwen3_tts import talker2code2wav_async_chunk
from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_code2wav import Qwen3TTSCode2Wav
from vllm_omni.worker.gpu_generation_model_runner import ExecuteModelState, GPUGenerationModelRunner

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


def _talker_req(rid: str, status: RequestStatus, *, computed: int, prompt_len: int, max_tokens: int):
    generated = max(0, computed - prompt_len)
    return SimpleNamespace(
        request_id=rid,
        external_req_id=rid,
        status=status,
        resumable=True,
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


# --- Stage 0 ASR -----------------------------------------------------------


def test_stage0_asr_holds_until_finished(monkeypatch):
    monkeypatch.setattr(
        "vllm_omni.model_executor.stage_input_processors.aura_omni.cached_tokenizer_from_config",
        lambda _config: SimpleNamespace(encode=lambda text: [1], decode=lambda ids: ""),
    )
    tm = _asr_tm()
    request = SimpleNamespace(
        request_id="video-asr",
        external_req_id="video-asr",
        output_text="简单介绍一下",
        additional_information={"aura_system_prompt": ["system"]},
        is_finished=lambda: False,
    )
    assert asr2aura_async_chunk(tm, None, request, is_finished=False) is None
    payload = asr2aura_async_chunk(tm, None, request, is_finished=True)
    assert payload["aura_asr_transcript"] == "简单介绍一下"
    assert payload["additional_information"]["aura_system_prompt"] == ["system"]


def test_stage0_asr_packs_video_bytes_not_nested_ints(monkeypatch):
    """tolist() of uint8 frames used to inflate Stage0→1 IPC ~3×."""
    monkeypatch.setattr(
        "vllm_omni.model_executor.stage_input_processors.aura_omni.cached_tokenizer_from_config",
        lambda _config: SimpleNamespace(encode=lambda text: [1], decode=lambda ids: ""),
    )
    frames = np.arange(2 * 4 * 4 * 3, dtype=np.uint8).reshape(2, 4, 4, 3)
    tm = _asr_tm()
    request = SimpleNamespace(
        request_id="video-pack",
        external_req_id="video-pack",
        output_text="看看视频",
        additional_information={
            "aura_system_prompt": ["system"],
            "deferred_multi_modal_data": {"video": [(frames, {"fps": 2.0, "total_num_frames": 2})]},
        },
        is_finished=lambda: True,
        multi_modal_data=None,
        mm_processor_kwargs=None,
    )
    payload = asr2aura_async_chunk(tm, None, request, is_finished=True)
    wire = payload["aura_turn_video"]["frames"]
    assert wire[AURA_VIDEO_WIRE_MARKER] is True
    assert isinstance(wire["data"], (bytes, bytearray))
    restored = unpack_aura_video_ndarray(wire)
    assert restored is not None
    assert restored.shape == (2, 4, 4, 3)


# --- Stage 1 AURA ----------------------------------------------------------


def test_stage1_resolves_asr_passthrough_into_aura_prompt():
    clear_all_sessions()
    payload = {
        "aura_asr_transcript": "简单介绍一下桌上的东西",
        "additional_information": {
            "aura_session_id": "aura-stage1-resolve",
            "aura_system_prompt": ["system"],
        },
    }
    request = SimpleNamespace(
        request_id="video-s1",
        external_req_id="video-s1",
        additional_information=None,
        omni_stage_payload=None,
    )

    class _Tok:
        def encode(self, prompt: str) -> list[int]:
            assert "<|im_start|>system" in prompt
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
    assert "prompt" not in payload


def test_stage1_two_sentence_tts_payloads_are_talker_ready(monkeypatch):
    monkeypatch.setenv("VLLM_AURA_SENTENCE_TTS", "1")
    monkeypatch.setenv("VLLM_AURA_SENTENCE_TTS_MIN_CHARS", "4")
    tm = _asr_tm()
    request = _aura_request(
        rid="video-s1-tts",
        text="今天天气很好。后面还有",
        finished=False,
    )
    first = aura2tts_async_chunk(tm, None, request, is_finished=False)
    assert first is not None
    assert _payload_has_talker_conditioning(first)
    assert first["text"] == ["今天天气很好。"]
    assert first["task_type"] == ["CustomVoice"]
    assert first["prompt_token_ids"]
    assert int(first["max_new_tokens"][0] if isinstance(first["max_new_tokens"], list) else first["max_new_tokens"]) > 0
    assert bool(first["meta"]["finished"].item()) is False

    request.output_text = "今天天气很好。后面还有内容。"
    request.is_finished = lambda: True
    second = aura2tts_async_chunk(tm, None, request, is_finished=True)
    assert second is not None
    assert bool(second["meta"]["finished"].item()) is True
    if _payload_has_talker_conditioning(second):
        assert "后面还有内容" in (second.get("text") or [""])[0]
        cap2 = second["max_new_tokens"][0] if isinstance(second["max_new_tokens"], list) else second["max_new_tokens"]
        cap1 = first["max_new_tokens"][0] if isinstance(first["max_new_tokens"], list) else first["max_new_tokens"]
        assert int(cap2) != int(cap1) or second["text"] != first["text"]
    else:
        # Remnant already flushed as first sentence; finish is empty sentinel.
        assert second.get("prompt_token_ids") == []


def test_stage1_v2_silent_token_emits_empty_finish_not_tts(monkeypatch):
    monkeypatch.setattr(
        "vllm_omni.model_executor.stage_input_processors.aura_omni.AURA_SILENT_TOKEN_IDS",
        [248070],
    )
    tm = _asr_tm()
    request = _aura_request(
        rid="video-silent",
        text="",
        finished=True,
        token_ids=[248070],
    )
    payload = aura2tts_async_chunk(tm, None, request, is_finished=True)
    assert payload is not None
    assert payload["prompt_token_ids"] == []
    assert bool(payload["meta"]["finished"].item()) is True
    assert _payload_has_talker_conditioning(payload) is False


# --- Stage 2 Talker --------------------------------------------------------


def test_stage2_applies_first_sentence_then_queues_second(monkeypatch, build_adapter):
    monkeypatch.setenv("VLLM_AURA_SENTENCE_TTS", "1")
    monkeypatch.setenv("VLLM_AURA_SENTENCE_TTS_MIN_CHARS", "4")
    tm = _asr_tm()
    request = _aura_request(rid="video-s2", text="今天天气很好。后面还有", finished=False)
    sentence1 = aura2tts_async_chunk(tm, None, request, is_finished=False)
    request.output_text = "今天天气很好。后面还有内容。"
    request.is_finished = lambda: True
    sentence2 = aura2tts_async_chunk(tm, None, request, is_finished=True)
    assert sentence1 is not None and sentence2 is not None
    assert _payload_has_talker_conditioning(sentence1)

    adapter, connector = build_adapter()
    talker = _talker_req("video-s2", RequestStatus.WAITING, computed=0, prompt_len=0, max_tokens=4096)
    connector.get.return_value = (sentence1, 16)
    assert adapter._poll_single_request(talker) is True
    assert talker.max_tokens < 4096
    assert talker.additional_information["text"] == ["今天天气很好。"]

    live_last = torch.ones(4)
    talker.status = RequestStatus.RUNNING
    talker.num_prompt_tokens = len(talker.prompt_token_ids) or talker.num_prompt_tokens
    if talker.num_prompt_tokens <= 0:
        talker.prompt_token_ids = [0] * 22
        talker.num_prompt_tokens = 22
    talker.num_computed_tokens = talker.num_prompt_tokens + 54
    talker._output_token_ids = [1] * 54
    talker._all_token_ids = talker.prompt_token_ids + [1] * 54
    talker.additional_information["hidden_states"] = {"last": live_last}

    connector.get.return_value = (sentence2, 16)
    assert adapter._poll_single_request(talker) is False
    assert talker.additional_information["text"] == ["今天天气很好。"]
    assert talker.additional_information["hidden_states"]["last"] is live_last
    assert talker.max_tokens < 4096

    if _payload_has_talker_conditioning(sentence2):
        with pytest.raises(RuntimeError, match="Refuse max_new_tokens"):
            _apply_max_new_tokens_from_payload(talker, sentence2)

    talker.status = RequestStatus.WAITING_FOR_CHUNK
    assert adapter.try_apply_pending_upstream_payload(talker) is False
    assert talker.additional_information["text"] == ["今天天气很好。"]

    talker.status = RequestStatus.WAITING
    connector.get.return_value = None
    drained = adapter._poll_single_request(talker) or adapter.try_apply_pending_upstream_payload(talker)
    assert drained is True
    if _payload_has_talker_conditioning(sentence2):
        assert talker.num_computed_tokens == 0
        assert "后面还有内容" in (talker.additional_information.get("text") or [""])[0]
    else:
        assert talker.additional_information["text"] == ["今天天气很好。"]
        assert "video-s2" in adapter.upstream_exhausted_requests


def test_stage2_scheduler_rearms_waiting_when_next_sentence_is_queued():
    sched = OmniARScheduler.__new__(OmniARScheduler)
    sched._new_prompt_len_snapshot = {}
    sched.vllm_config = SimpleNamespace(model_config=SimpleNamespace(stage_id=2))
    sched.num_waiting_for_streaming_input = 0
    sched.log_stats = False
    sched.skipped_waiting = set()
    enqueued: list = []
    sched._enqueue_waiting_request = enqueued.append
    sched.chunk_transfer_adapter = SimpleNamespace(
        receives_chunks=True,
        upstream_exhausted_requests=set(),
        _pending_upstream_payloads={"video-s2": object()},
    )
    session = SimpleNamespace(
        request_id="video-s2",
        resumable=True,
        status=RequestStatus.FINISHED_STOPPED,
        streaming_queue=None,
    )
    assert OmniARScheduler._handle_stopped_request(sched, session) is False
    assert session.status == RequestStatus.WAITING
    assert enqueued == [session]


# --- Stage 3 Code2Wav ------------------------------------------------------


def test_stage3_talker_emits_ic8_then_finish_codes():
    """Match aura_omni_v2_1gpu.yaml: initial_codec_chunk_frames=8, chunk=20."""
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
    rid = "video-s3"
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


def test_stage3_empty_silent_batch_emits_one_wav_slot_per_request():
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
        runtime_additional_information=[{"meta": {"finished": torch.tensor(True)}} for _ in range(4)],
    )
    assert len(out.multimodal_outputs["model_outputs"]) == 4
    assert all(audio.numel() == 0 for audio in out.multimodal_outputs["model_outputs"])


def test_stage3_runner_broadcasts_empty_model_outputs_to_batch():
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
        req_ids=[f"video-{i}" for i in range(4)],
        req_id_to_index={f"video-{i}": i for i in range(4)},
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
    assert len(output.multimodal_outputs) == 4
    assert all(payload["model_outputs"].numel() == 0 for payload in output.multimodal_outputs)
