# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AURA-compatible Qwen3.5-VL wrapper (AURA_v2 / qwen3_5 checkpoints).

Mirrors ``aura_omni/qwen3_vl.py``: keep stock vLLM Qwen3.5 multimodal logic while
accepting structurally compatible remote configs and opting out of per-step
hidden D2H for Stage1 → TTS text handoff.
"""

from __future__ import annotations

from typing import Any

from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    Qwen3_5ProcessingInfo,
)
from vllm.model_executor.models.qwen3_vl import (
    Qwen3VLDummyInputsBuilder,
    Qwen3VLMultiModalProcessor,
    Qwen3VLProcessor,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.transformers_utils.configs.qwen3_5 import Qwen3_5Config


class AuraQwen3_5ProcessingInfo(Qwen3_5ProcessingInfo):
    def get_hf_config(self) -> Any:
        try:
            return self.ctx.get_hf_config(Qwen3_5Config)
        except TypeError:
            config = self.ctx.get_hf_config()
            if (
                getattr(config, "model_type", None) == "qwen3_5"
                and hasattr(config, "vision_config")
                and hasattr(config, "text_config")
            ):
                return config
            raise

    def get_hf_processor(self, **kwargs: object) -> Any:
        # Qwen3.5 multimodal still uses the Qwen3-VL processor class in vLLM.
        return self.ctx.get_hf_processor(
            Qwen3VLProcessor,
            use_fast=kwargs.pop("use_fast", True),
            **kwargs,
        )


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=AuraQwen3_5ProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class AuraQwen3_5ForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    # Stage1 AURA → TTS only needs finished text / token ids (aura2tts_async_chunk).
    requires_full_prefix_cached_hidden_states = False
    omni_pooler_payload_include_hidden = False
