# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""AURA-compatible Qwen3.5-VL wrapper (AURA_v2 / qwen3_5 checkpoints).

Mirrors ``aura_omni/qwen3_vl.py``: keep stock vLLM Qwen3.5 multimodal logic while
accepting structurally compatible remote configs. Stage1 → TTS only needs finished
text / token ids, so full-prefix hidden D2H is left off.
"""

from __future__ import annotations

import threading
from typing import Any

from transformers import Qwen3_5Config
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

# Running video clock for the prompt being processed. Thread-local so
# concurrent sessions processed on different threads cannot share one clock.
_VIDEO_CLOCK = threading.local()


class AuraQwen3_5ProcessingInfo(Qwen3_5ProcessingInfo):
    def _get_video_second_idx(self, metadata: dict[str, Any], *args: Any, **kwargs: Any) -> list[float]:
        """Run video timestamps on across the clips of one prompt.

        AURA_v2 sees its retained history as consecutive clips. The native
        gateway served it with a patched vLLM ``qwen3_vl.py`` in which clip k
        starts at the summed duration of clips 0..k-1. Stock vLLM restarts
        every clip at 0 s, so the whole history claims to be the first second.
        """
        timestamps = super()._get_video_second_idx(metadata, *args, **kwargs)
        base = getattr(_VIDEO_CLOCK, "seconds", 0.0)
        _VIDEO_CLOCK.seconds = base + float(metadata["duration"])
        if not base:
            return timestamps
        return [timestamp + base for timestamp in timestamps]

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


class AuraQwen3VLMultiModalProcessor(Qwen3VLMultiModalProcessor):
    def _apply_hf_processor_main(self, *args: Any, **kwargs: Any) -> Any:
        # One call processes every clip of one prompt in order: the clock
        # starts at 0 s here, exactly where the native patch reset it.
        _VIDEO_CLOCK.seconds = 0.0
        try:
            return super()._apply_hf_processor_main(*args, **kwargs)
        finally:
            _VIDEO_CLOCK.seconds = 0.0


@MULTIMODAL_REGISTRY.register_processor(
    AuraQwen3VLMultiModalProcessor,
    info=AuraQwen3_5ProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class AuraQwen3_5ForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    # Stage1 AURA → TTS only needs finished text / token ids.
    requires_full_prefix_cached_hidden_states = False
    # Qwen3.5 is hybrid (full attention + GDN), so the Omni stage-output cache
    # refuses it. Native KV + aligned recurrent-state caching still serves the
    # shared history prefix across turns.
    uses_native_kv_cache_only = True
