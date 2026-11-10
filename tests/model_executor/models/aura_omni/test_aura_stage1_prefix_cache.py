# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""AURA Stage1 reuses native KV and aligned recurrent state across turns.

Qwen3.5 is hybrid, so the Omni stage-output cache refuses it. Stage1 opts into
native-only caching the way Ming3 does, and the deploy files turn the flag on.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import yaml

from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.core.prefix_cache import ModelCachePolicy, OmniPrefixCacheUnmatchError, stage_prefix_cache_config
from vllm_omni.model_executor.models.aura_omni.qwen3_5 import AuraQwen3_5ForConditionalGeneration

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_DEPLOYS = ["aura_omni.yaml", "aura_omni_1gpu.yaml"]


def _gate(policy: ModelCachePolicy):
    hybrid = SimpleNamespace(kv_cache_groups=[SimpleNamespace(), SimpleNamespace()], num_blocks=128)
    return stage_prefix_cache_config(
        kv_cache_config=hybrid,
        cache_config=SimpleNamespace(enable_prefix_caching=True, block_size=16),
        kv_transfer_config=None,
        scheduler_config=SimpleNamespace(),
        model_config=SimpleNamespace(),
        is_pooling_model=False,
        model_cache_policy=policy,
    )


def test_aura_stage1_is_native_kv_only():
    policy = ModelCachePolicy.from_model(AuraQwen3_5ForConditionalGeneration)

    assert policy.native_kv_cache_only is True
    assert policy.needs_full_hidden_states is False
    assert not policy.deferred_keys


def test_hybrid_kv_groups_only_pass_the_omni_gate_when_native_only():
    assert _gate(ModelCachePolicy.from_model(AuraQwen3_5ForConditionalGeneration)) is None
    with pytest.raises(OmniPrefixCacheUnmatchError, match="single full-attention"):
        _gate(ModelCachePolicy(needs_full_hidden_states=False))


@pytest.mark.parametrize("name", _DEPLOYS)
def test_deploy_enables_prefix_cache_on_asr_and_stage1_only(name):
    with open(get_deploy_config_path(name), encoding="utf-8") as f:
        stages = {stage["stage_id"]: stage for stage in yaml.safe_load(f)["stages"]}

    assert [stages[i]["enable_prefix_caching"] for i in range(4)] == [True, True, False, False]
    assert "mamba_cache_mode" not in stages[1]
