# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Public duplex hook: resumable defaults True; ephemeral ids when False."""

from __future__ import annotations

from vllm.sampling_params import SamplingParams

from vllm_omni.engine.duplex.config import DuplexCapabilities
from vllm_omni.engine.duplex.contracts import (
    DuplexFence,
    DuplexStageRequestContext,
    DuplexStageSubmission,
    duplex_ephemeral_stage_request_id,
)
from vllm_omni.engine.duplex.session_manager import DuplexSessionManager


def test_submission_resumable_default_preserves_minicpm() -> None:
    ctx = DuplexStageRequestContext(
        request_id="r",
        session_id="s",
        fence=DuplexFence("s"),
        stage_id=0,
        final_stage_id=2,
        config_generation=0,
        sampling_params=(SamplingParams(max_tokens=1),),
    )
    submission = DuplexStageSubmission(
        context=ctx,
        prompt={"prompt_token_ids": [1, 2]},
        already_submitted=False,
    )
    assert submission.resumable is True


def test_stage_request_id_respects_resumable_flag() -> None:
    fence = DuplexFence("sess", epoch=3, turn_id=9)
    resumable_id = DuplexSessionManager.stage_request_id(fence, stage_id=0, resumable=True)
    ephemeral_id = DuplexSessionManager.stage_request_id(fence, stage_id=0, resumable=False)
    assert ephemeral_id == duplex_ephemeral_stage_request_id(fence, stage_id=0)
    assert resumable_id != ephemeral_id
    assert "t9" in ephemeral_id


def test_capabilities_expose_overlapped_input_default_false() -> None:
    caps = DuplexCapabilities()
    assert caps.supports_overlapped_input is False
    assert caps.supports_core_resumable_request is False
    payload = caps.as_dict()
    assert payload["supports_overlapped_input"] is False
