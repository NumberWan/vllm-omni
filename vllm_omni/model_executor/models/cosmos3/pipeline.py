# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cosmos3 opt-in topologies for policy (OpenPI) and T2I deploy YAML.

Both topologies declare neither ``hf_architectures`` nor
``diffusers_class_name``. Every Cosmos3 checkpoint (T2I, video, policy)
shares HF metadata (``model_type=cosmos3_omni``, ``model_index.json``
``_class_name=Cosmos3OmniDiffusersPipeline``). Auto-registering any of them
under that metadata would capture the others. Select explicitly via a deploy
yaml ``pipeline:`` key.

Policy (``cosmos3_policy``)::

    vllm serve nvidia/Cosmos3-Nano-Policy-DROID --omni \\
      --deploy-config .../vllm_omni/deploy/cosmos3_policy_droid.yaml

T2I (``cosmos3_omni_t2i``) — makes ``--deploy-config`` apply stage /
``model_config`` (e.g. ``guardrails: false``) instead of being silently
dropped (#6874)::

    vllm serve nvidia/Cosmos3-Super-Text2Image --omni \\
      --deploy-config .../vllm_omni/deploy/cosmos3_super_t2i.yaml

Without ``--deploy-config``, T2I/video keep the default single-stage
diffusion CLI fallback. ``--no-guardrails`` remains the CLI-only path.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

COSMOS3_POLICY_PIPELINE = PipelineConfig(
    model_type="cosmos3_policy",
    model_arch="Cosmos3OmniDiffusersPipeline",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="diffusion",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(),
            final_output=True,
            final_output_type="action",
            model_arch="Cosmos3OmniDiffusersPipeline",
        ),
    ),
)

COSMOS3_OMNI_T2I_PIPELINE = PipelineConfig(
    model_type="cosmos3_omni_t2i",
    model_arch="Cosmos3OmniDiffusersPipeline",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="diffusion",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(),
            final_output=True,
            final_output_type="image",
            model_arch="Cosmos3OmniDiffusersPipeline",
        ),
    ),
)
