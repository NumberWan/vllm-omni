# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace

import janus
import pytest
from vllm.sampling_params import SamplingParams

from vllm_omni.engine.orchestrator import (
    Orchestrator,
    OrchestratorRequestState,
    _should_skip_stage_submission,
)
from vllm_omni.engine.stage_pool import StagePool
from vllm_omni.model_executor.stage_input_processors.stage_bypass import OMNI_SKIP_STAGES_KEY

from .test_orchestrator import (
    FakeOutputProcessor,
    FakeStageClient,
    _build_harness,
    _enqueue_add_request,
    _sampling_params,
    _shutdown_orchestrator,
    _wait_for,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_should_skip_stage_submission_reads_omni_skip_stages() -> None:
    original_prompt = {
        "prompt": "video turn",
        "additional_information": {OMNI_SKIP_STAGES_KEY: [0]},
    }
    prompt = SimpleNamespace(request_id="req-skip", prompt_token_ids=[1, 2, 3])
    assert _should_skip_stage_submission(prompt, original_prompt, stage_id=0) is True
    assert _should_skip_stage_submission(prompt, original_prompt, stage_id=1) is False


@pytest.mark.asyncio
async def test_forward_bypassed_stage_zero_skips_asr_submit() -> None:
    stage0 = FakeStageClient(stage_type="llm", final_output=False)
    stage1 = FakeStageClient(
        stage_type="llm",
        final_output=True,
        next_inputs=[{"prompt_token_ids": [7, 8, 9]}],
    )
    processors = [
        FakeOutputProcessor(request_outputs=[]),
        FakeOutputProcessor(request_outputs=[]),
    ]
    stage_pools = [
        StagePool(
            0,
            [stage0],
            output_processor=processors[0],
            stage_vllm_config=SimpleNamespace(model_config=SimpleNamespace(max_model_len=64)),
        ),
        StagePool(
            1,
            [stage1],
            output_processor=processors[1],
            stage_vllm_config=SimpleNamespace(model_config=SimpleNamespace(max_model_len=64)),
        ),
    ]
    request_q = janus.Queue()
    output_q = janus.Queue()
    rpc_q = janus.Queue()
    orchestrator = Orchestrator(
        request_async_queue=request_q.async_q,
        output_async_queue=output_q.async_q,
        rpc_async_queue=rpc_q,
        stage_pools=stage_pools,
        async_chunk=False,
    )

    original_prompt = {
        "prompt": "video-only",
        "additional_information": {OMNI_SKIP_STAGES_KEY: [0]},
    }
    req_state = OrchestratorRequestState(
        request_id="req-bypass",
        prompt=original_prompt,
        sampling_params_list=[_sampling_params(), _sampling_params()],
        final_stage_id=1,
    )

    await orchestrator._forward_bypassed_stage_zero("req-bypass", req_state)

    assert len(stage0.add_request_calls) == 0
    assert len(stage1.add_request_calls) == 1
    assert stage1.add_request_calls[0][0].request_id == "req-bypass"


@pytest.mark.asyncio
async def test_add_request_bypasses_stage_zero_end_to_end() -> None:
    stage0 = FakeStageClient(stage_type="llm", final_output=False)
    stage1 = FakeStageClient(
        stage_type="llm",
        final_output=True,
        next_inputs=[{"prompt_token_ids": [7, 8, 9]}],
    )
    processors = [
        FakeOutputProcessor(request_outputs=[]),
        FakeOutputProcessor(request_outputs=[]),
    ]
    fixture = _build_harness([stage0, stage1], output_processors=processors)
    request = SimpleNamespace(request_id="req-skip-asr", prompt_token_ids=[1, 2, 3])

    try:
        await _enqueue_add_request(
            fixture,
            request_id="req-skip-asr",
            prompt=request,
            original_prompt={
                "prompt": "video-only",
                "additional_information": {OMNI_SKIP_STAGES_KEY: [0]},
            },
            sampling_params_list=[_sampling_params(), _sampling_params()],
            final_stage_id=1,
        )

        await _wait_for(lambda: len(stage1.add_request_calls) == 1, timeout=3.0)
        assert len(stage0.add_request_calls) == 0
        assert stage1.add_request_calls[0][0].request_id == "req-skip-asr"
    finally:
        await _shutdown_orchestrator(fixture)
