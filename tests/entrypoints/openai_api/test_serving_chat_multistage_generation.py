# SPDX-License-Identifier: Apache-2.0
"""Regression tests for multistage diffusion generation input construction."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from PIL import Image
from vllm.sampling_params import SamplingParams

from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def serving_chat():
    from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

    return object.__new__(OmniOpenAIServingChat)


def test_build_multistage_generation_inputs_applies_stage_specific_overrides(serving_chat):
    from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

    engine = SimpleNamespace(
        stage_configs=[
            SimpleNamespace(stage_type="llm", is_comprehension=True),
            SimpleNamespace(stage_type="diffusion", is_comprehension=False),
            SimpleNamespace(stage_type="diffusion", is_comprehension=False),
        ],
        default_sampling_params_list=[
            SamplingParams(temperature=0.2, seed=11),
            OmniDiffusionSamplingParams(),
            OmniDiffusionSamplingParams(),
        ],
    )
    reference_image = Image.new("RGB", (24, 24), color="green")
    extra_body = {
        "negative_prompt": "blurry",
        "num_inference_steps": 28,
        "guidance_scale": 7.5,
        "true_cfg_scale": 5.0,
        "guidance_scale_2": 1.25,
        "layers": 6,
        "resolution": 1024,
        "lora": {"name": "adapter-a", "path": "/tmp/adapter-a", "scale": 0.6},
    }
    gen_params = OmniDiffusionSamplingParams(height=768, width=1024, seed=0, num_outputs_per_prompt=2)

    engine_prompt, sampling_params_list = OmniOpenAIServingChat._build_multistage_generation_inputs(
        serving_chat,
        engine=engine,
        prompt="draw a robot",
        extra_body=extra_body,
        reference_images=[reference_image],
        gen_params=gen_params,
    )

    assert engine_prompt["prompt"] == "draw a robot"
    assert engine_prompt["modalities"] == ["image"]
    assert engine_prompt["negative_prompt"] == "blurry"
    assert engine_prompt["mm_processor_kwargs"] == {"target_h": 768, "target_w": 1024}
    assert engine_prompt["multi_modal_data"]["image"].size == (24, 24)

    assert len(sampling_params_list) == 3
    assert sampling_params_list[0].temperature == 0.2
    assert sampling_params_list[0].seed == 0
    assert sampling_params_list[0].extra_args == {"target_h": 768, "target_w": 1024}
    assert sampling_params_list[1] is not gen_params
    assert sampling_params_list[2] is not gen_params
    assert sampling_params_list[1] is not sampling_params_list[2]
    assert sampling_params_list[1].height == 768
    assert sampling_params_list[1].width == 1024
    assert sampling_params_list[1].seed == 0
    assert sampling_params_list[1].num_inference_steps == 28
    assert sampling_params_list[1].guidance_scale == 7.5
    assert sampling_params_list[1].num_outputs_per_prompt == 2
    assert sampling_params_list[1].true_cfg_scale == 5.0
    assert sampling_params_list[1].lora_request.name == "adapter-a"
    assert sampling_params_list[1].lora_scale == 0.6
    assert sampling_params_list[2].height == 768
    assert sampling_params_list[2].width == 1024
    assert sampling_params_list[2].seed == 0
    assert sampling_params_list[2].num_inference_steps == 28
    assert sampling_params_list[2].lora_request.name == "adapter-a"
    assert sampling_params_list[2].lora_scale == 0.6
    assert gen_params.lora_request is None
    assert engine.default_sampling_params_list[1].height is None
    assert engine.default_sampling_params_list[1].lora_request is None
    assert engine.default_sampling_params_list[2].resolution == 640
    assert engine.default_sampling_params_list[2].lora_request is None


def test_engine_first_llm_hunyuan_detects_model_arch_under_engine_args():
    """Regression: resolved stages expose ``model_arch`` under ``engine_args``, not top-level."""
    from vllm_omni.config.stage_config import StageType
    from vllm_omni.entrypoints.openai.serving_chat import _engine_first_llm_is_hunyuan_image3

    eng = SimpleNamespace(
        stage_configs=[
            SimpleNamespace(
                stage_type=StageType.LLM,
                engine_args={"model_arch": "HunyuanImage3ForCausalMM"},
            ),
            SimpleNamespace(stage_type=StageType.DIFFUSION, engine_args={}),
        ]
    )
    assert _engine_first_llm_is_hunyuan_image3(eng) is True


def test_engine_first_llm_hunyuan_detects_enum_stage_type_not_str_eq_llm():
    """``StageType.LLM`` must not be rejected by ``str(enum).lower() != 'llm'`` style checks."""
    from vllm_omni.config.stage_config import StageType
    from vllm_omni.entrypoints.openai.serving_chat import _engine_first_llm_is_hunyuan_image3

    eng = SimpleNamespace(
        stage_configs=[
            SimpleNamespace(
                stage_type=StageType.LLM,
                engine_args=SimpleNamespace(model_arch="HunyuanImage3ForCausalMM"),
            ),
        ]
    )
    assert _engine_first_llm_is_hunyuan_image3(eng) is True


def test_build_multistage_hunyuan_tokenizer_sets_prompt_with_img_placeholder(serving_chat):
    """When using ``prompt_token_ids``, plain ``prompt`` must still contain ``<img>`` for MM apply."""
    from unittest.mock import MagicMock

    from vllm_omni.config.stage_config import StageType
    from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

    tok = MagicMock()
    tok.convert_tokens_to_ids = MagicMock(side_effect=lambda t: {"<|startoftext|>": 1, "<img>": 2}.get(t, 0))
    tok.encode = MagicMock(return_value=[3, 4])

    engine = SimpleNamespace(
        stage_configs=[
            SimpleNamespace(
                stage_type=StageType.LLM,
                engine_args={"model_arch": "HunyuanImage3ForCausalMM"},
                is_comprehension=False,
            ),
            SimpleNamespace(stage_type=StageType.DIFFUSION, is_comprehension=False),
        ],
        default_sampling_params_list=None,
    )

    ref = Image.new("RGB", (8, 8), color="blue")
    gen_params = OmniDiffusionSamplingParams(height=512, width=512, seed=1, num_outputs_per_prompt=1)
    engine_prompt, _ = OmniOpenAIServingChat._build_multistage_generation_inputs(
        serving_chat,
        engine=engine,
        prompt="edit this",
        extra_body={},
        reference_images=[ref],
        gen_params=gen_params,
        tokenizer=tok,
    )
    assert "<img>" in engine_prompt["prompt"]
    assert engine_prompt.get("prompt_token_ids") is not None
