# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm_omni.model_executor.stage_input_processors.stage_bypass import (
    OMNI_SKIP_STAGES_KEY,
    should_skip_stage,
    should_skip_stage_from_info,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_should_skip_stage_reads_additional_information_flag():
    assert should_skip_stage({}, stage_id=0) is False
    assert should_skip_stage({"additional_information": {OMNI_SKIP_STAGES_KEY: [0]}}, stage_id=0) is True
    assert should_skip_stage({"additional_information": {OMNI_SKIP_STAGES_KEY: [0]}}, stage_id=1) is False
    assert should_skip_stage({"additional_information": {OMNI_SKIP_STAGES_KEY: [0, 1]}}, stage_id=1) is True
    assert should_skip_stage({"additional_information": {OMNI_SKIP_STAGES_KEY: []}}, stage_id=0) is False
