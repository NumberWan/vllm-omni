# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm_omni.model_executor.models.aura_omni.qwen3_5 import (
    AuraQwen3_5ForConditionalGeneration,
    AuraQwen3_5ProcessingInfo,
    Qwen3VLProcessor,
)
from vllm_omni.model_executor.models.registry import OmniModelRegistry

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeCtx:
    def __init__(self, config):
        self.config = config
        self.processor_type = None

    def get_hf_config(self, typ=None):
        if typ is not None:
            raise TypeError("wrong config class")
        return self.config

    def get_hf_processor(self, typ=None, **kwargs):
        self.processor_type = typ
        return SimpleNamespace(kwargs=kwargs)


def test_aura_qwen3_5_processing_accepts_qwen3_5_config():
    info = object.__new__(AuraQwen3_5ProcessingInfo)
    info.ctx = _FakeCtx(
        SimpleNamespace(
            model_type="qwen3_5",
            vision_config=SimpleNamespace(spatial_merge_size=2),
            text_config=SimpleNamespace(hidden_size=4096),
        )
    )

    assert info.get_hf_config().model_type == "qwen3_5"


def test_aura_qwen3_5_processing_rejects_unrelated_remote_config():
    info = object.__new__(AuraQwen3_5ProcessingInfo)
    info.ctx = _FakeCtx(SimpleNamespace(model_type="not_qwen3_5"))

    with pytest.raises(TypeError, match="wrong config class"):
        info.get_hf_config()


def test_aura_qwen3_5_model_arch_registered():
    model_cls = OmniModelRegistry._try_load_model_cls("AuraQwen3_5ForConditionalGeneration")

    assert model_cls is AuraQwen3_5ForConditionalGeneration


def test_aura_qwen3_5_opts_out_of_per_step_hidden_d2h():
    assert AuraQwen3_5ForConditionalGeneration.requires_full_prefix_cached_hidden_states is False
    assert AuraQwen3_5ForConditionalGeneration.omni_pooler_payload_include_hidden is False


def test_aura_qwen3_5_processing_forces_upstream_processor_class():
    info = object.__new__(AuraQwen3_5ProcessingInfo)
    ctx = _FakeCtx(SimpleNamespace(model_type="qwen3_5"))
    info.ctx = ctx

    processor = info.get_hf_processor(extra="value")

    assert ctx.processor_type is Qwen3VLProcessor
    assert processor.kwargs["use_fast"] is True
    assert processor.kwargs["extra"] == "value"
