"""Unit tests for Omni AR streaming-session async placeholder handling."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

# Imports must run in this order: vllm_omni applies patches to vllm.v1.request before
# Request / StreamingUpdate are bound in this module. Ruff isort would reorder them.
# isort: off
import vllm_omni  # noqa: F401 - import for side effects (patch vLLM)
from vllm.sampling_params import SamplingParams
from vllm.v1.request import Request, RequestStatus, StreamingUpdate
from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler

# isort: on

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Queue(list):
    def add_request(self, request: Request) -> None:
        self.append(request)

    def remove_requests(self, requests) -> None:
        for request in requests:
            if request in self:
                self.remove(request)


def _make_scheduler(*, stage_id: int = 0) -> OmniARScheduler:
    sched = OmniARScheduler.__new__(OmniARScheduler)
    sched._new_prompt_len_snapshot = {}
    sched.vllm_config = SimpleNamespace(model_config=SimpleNamespace(stage_id=stage_id))
    sched.num_waiting_for_streaming_input = 0
    sched.log_stats = False
    sched.chunk_transfer_adapter = None
    return sched


def _make_request() -> Request:
    return Request(
        request_id="req-ar-streaming-test",
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=8),
        pooling_params=None,
        arrival_time=100.0,
        block_hasher=None,
    )


def _make_update(prompt_token_ids: list[int] | None = None) -> StreamingUpdate:
    return StreamingUpdate(
        mm_features=None,
        prompt_token_ids=[10, 20] if prompt_token_ids is None else prompt_token_ids,
        max_tokens=32,
        arrival_time=200.0,
        sampling_params=SamplingParams(max_tokens=16),
    )


def test_stage0_streaming_update_discards_outstanding_async_placeholder_token() -> None:
    sched = _make_scheduler(stage_id=0)
    session = _make_request()
    session.status = RequestStatus.WAITING_FOR_STREAMING_REQ
    session.append_output_token_ids([7, 8, 9])
    session.num_computed_tokens = 6
    session.num_output_placeholders = 1
    session.spec_token_ids = [-1]

    sched._update_request_as_session(session, _make_update([10, 20]))

    assert session.async_tokens_to_discard == 1
    assert session.num_output_placeholders == 0
    assert session.spec_token_ids == []
    # The async placeholder makes token 9 unconfirmed, so only 7 and 8 are
    # carried into the next streaming prompt before the new chunk tokens.
    assert session.prompt_token_ids == [1, 2, 3, 7, 8, 10, 20]
    assert list(session._all_token_ids) == [1, 2, 3, 7, 8, 10, 20]
    assert session._output_token_ids == []
    assert session.num_prompt_tokens == 7
    assert sched._new_prompt_len_snapshot[session.request_id] == 2


def test_stage0_streaming_update_keeps_all_computed_tokens_without_placeholder() -> None:
    sched = _make_scheduler(stage_id=0)
    session = _make_request()
    session.status = RequestStatus.WAITING_FOR_STREAMING_REQ
    session.append_output_token_ids([7, 8, 9])
    session.num_computed_tokens = 6
    session.num_output_placeholders = 0

    sched._update_request_as_session(session, _make_update([10, 20]))

    assert getattr(session, "async_tokens_to_discard", 0) == 0
    assert session.num_output_placeholders == 0
    assert session.prompt_token_ids == [1, 2, 3, 7, 8, 9, 10, 20]
    assert list(session._all_token_ids) == [1, 2, 3, 7, 8, 9, 10, 20]
    assert session._output_token_ids == []
    assert session.num_prompt_tokens == 8
    assert sched._new_prompt_len_snapshot[session.request_id] == 2


def test_downstream_chunk_stop_resets_request_for_next_connector_chunk() -> None:
    sched = _make_scheduler(stage_id=2)
    sched.waiting = _Queue()
    sched.skipped_waiting = _Queue()
    freed_kv = []
    freed_encoder = []
    sched.kv_cache_manager = SimpleNamespace(free=lambda request: freed_kv.append(request.request_id))
    sched.encoder_cache_manager = SimpleNamespace(free=lambda request: freed_encoder.append(request.request_id))
    sched.chunk_transfer_adapter = SimpleNamespace(is_done_receiving_chunks=lambda request_id: False)

    request = _make_request()
    request.status = RequestStatus.FINISHED_STOPPED
    request.num_computed_tokens = 9
    request.num_output_placeholders = 1
    request.spec_token_ids = [99]
    request.append_output_token_ids([7, 8])

    assert sched._should_wait_for_next_chunk(request) is True

    sched._reset_request_for_next_chunk(request)

    assert freed_kv == [request.request_id]
    assert freed_encoder == [request.request_id]
    assert request.status == RequestStatus.WAITING
    assert request.num_computed_tokens == 0
    assert request.num_output_placeholders == 0
    assert request.spec_token_ids == []
    assert request._output_token_ids == []
    assert list(request._all_token_ids) == request.prompt_token_ids
    assert request.num_prompt_tokens == len(request.prompt_token_ids)
    assert sched.waiting == [request]
