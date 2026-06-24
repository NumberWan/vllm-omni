# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm_omni.model_executor.stage_input_processors.aura_omni_skip import (
    AURA_SKIP_ASR_KEY,
    make_mock_asr_source_output,
    should_skip_aura_asr,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_should_skip_aura_asr_reads_additional_information_flag():
    assert should_skip_aura_asr({}) is False
    assert should_skip_aura_asr({"additional_information": {AURA_SKIP_ASR_KEY: True}}) is True
    assert should_skip_aura_asr({"additional_information": {AURA_SKIP_ASR_KEY: [True]}}) is True
    assert should_skip_aura_asr({"additional_information": {AURA_SKIP_ASR_KEY: False}}) is False

    mock = make_mock_asr_source_output("req-skip")
    assert mock.request_id == "req-skip"
    assert mock.outputs[0].cumulative_text == ""
