# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import base64
import io
import os
from pathlib import Path

import pytest
import requests
import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from PIL import Image

from tests.e2e.accuracy.helpers import assert_images_pixel_close, assert_similarity, model_output_dir
from tests.helpers.clean import cleanup_test_environment
from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServer
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.full_model, pytest.mark.diffusion]


MODEL_ID = "Qwen/Qwen-Image"
MODEL_ENV_VAR = "QWEN_IMAGE_MODEL"
PROMPT = "A photo of a cat sitting on a laptop keyboard, digital art style."
NEGATIVE_PROMPT = "blurry, low quality"
WIDTH = 512
HEIGHT = 512
NUM_INFERENCE_STEPS = 20
TRUE_CFG_SCALE = 4.0
SEED = 42
# Compiled TORCH_SDPA is bistable, so these cases run --enforce-eager.
# The gates below are the previous compile-mode scores and are not the eager
# baseline. Replace them after the eager remeasure.
# H100 nightly #17701, compiled TORCH_SDPA vs Diffusers native:
# SSIM 0.982257 / PSNR 32.072731.
# A3 nightly #9423: SSIM 0.955357 / PSNR 27.170992.
# B200 has no measurement on this backend and uses the H100 bounds.
_CUDA_SSIM_THRESHOLD = 0.98
_CUDA_PSNR_THRESHOLD = 32.0
_NPU_SSIM_THRESHOLD = 0.95
_NPU_PSNR_THRESHOLD = 27.0
# Previous H100 #17701 pixel gates were for Qwen-Image-2512. Keep the same
# numeric room until the eager Qwen-Image pixel run is measured.
_CUDA_MEAN_ABS_DIFF_THRESHOLD = 3e-2
_CUDA_P99_ABS_DIFF_THRESHOLD = 4e-1
_NPU_MEAN_ABS_DIFF_THRESHOLD = 1.0
_NPU_P99_ABS_DIFF_THRESHOLD = 1.0


def _similarity_thresholds() -> tuple[float, float]:
    if current_omni_platform.is_npu():
        return _NPU_SSIM_THRESHOLD, _NPU_PSNR_THRESHOLD
    return _CUDA_SSIM_THRESHOLD, _CUDA_PSNR_THRESHOLD


def _pixel_thresholds() -> tuple[float, float]:
    if current_omni_platform.is_npu():
        return _NPU_MEAN_ABS_DIFF_THRESHOLD, _NPU_P99_ABS_DIFF_THRESHOLD
    return _CUDA_MEAN_ABS_DIFF_THRESHOLD, _CUDA_P99_ABS_DIFF_THRESHOLD


def _model_name() -> str:
    return os.environ.get(MODEL_ENV_VAR, MODEL_ID)


def _local_files_only(model: str) -> bool:
    return Path(model).exists()


# Same Omni / Diffusers pair on CUDA (H100, B200) and NPU A3.
# FLASH_ATTN_3_HUB is Hopper-only and fails on B200/A100 (#8087). Auto-select
# can still land on that kernel. TORCH_SDPA matches Diffusers ``native`` and
# has a binding on both sides. --enforce-eager skips regional compile, which
# otherwise emits one of two bit-stable images.
_OMNI_CI_ATTN_BACKEND = "TORCH_SDPA"
_DIFFUSERS_CI_ATTN_BACKEND = "native"
_OMNI_CI_ATTN_ENV = {"DIFFUSION_ATTENTION_BACKEND": _OMNI_CI_ATTN_BACKEND}


def _run_vllm_omni_qwen_image(*, model: str, output_path: Path) -> Image.Image:
    server_args = [
        "--num-gpus",
        "1",
        "--stage-init-timeout",
        "300",
        "--init-timeout",
        "900",
        "--diffusion-attention-backend",
        _OMNI_CI_ATTN_BACKEND,
        "--enforce-eager",
    ]
    with OmniServer(model, server_args, use_omni=True, env_dict=_OMNI_CI_ATTN_ENV) as omni_server:
        response = requests.post(
            f"http://{omni_server.host}:{omni_server.port}/v1/images/generations",
            json={
                "model": omni_server.model,
                "prompt": PROMPT,
                "size": f"{WIDTH}x{HEIGHT}",
                "n": 1,
                "response_format": "b64_json",
                "negative_prompt": NEGATIVE_PROMPT,
                "num_inference_steps": NUM_INFERENCE_STEPS,
                "true_cfg_scale": TRUE_CFG_SCALE,
                "seed": SEED,
            },
            timeout=600,
        )
        response.raise_for_status()
        payload = response.json()
        assert len(payload["data"]) == 1
        image_bytes = base64.b64decode(payload["data"][0]["b64_json"])
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image.load()
        image.save(output_path)
        return image


def _run_diffusers_qwen_image(*, model: str, output_path: Path) -> Image.Image:
    cleanup_test_environment()
    pipe: DiffusionPipeline | None = None
    try:
        pipe = DiffusionPipeline.from_pretrained(
            model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            local_files_only=_local_files_only(model),
        ).to(current_omni_platform.device_type)
        pipe.transformer.set_attention_backend(_DIFFUSERS_CI_ATTN_BACKEND)
        generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(SEED)
        result = pipe(  # pyright: ignore[reportCallIssue]
            prompt=PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            width=WIDTH,
            height=HEIGHT,
            num_inference_steps=NUM_INFERENCE_STEPS,
            true_cfg_scale=TRUE_CFG_SCALE,
            generator=generator,
        )
        output_image = result.images[0].convert("RGB")
        output_image.save(output_path)
        return output_image
    finally:
        if pipe is not None and hasattr(pipe, "maybe_free_model_hooks"):
            pipe.maybe_free_model_hooks()
        del pipe
        cleanup_test_environment()


@pytest.mark.benchmark
@hardware_test(res={"cuda": ["H100", "B200"], "npu": "A3"}, num_cards=1)
def test_qwen_image_matches_diffusers(accuracy_artifact_root: Path) -> None:
    model = _model_name()
    output_dir = model_output_dir(accuracy_artifact_root, MODEL_ID)

    vllm_output = _run_vllm_omni_qwen_image(model=model, output_path=output_dir / "vllm_omni.png")
    diffusers_output = _run_diffusers_qwen_image(model=model, output_path=output_dir / "diffusers.png")

    ssim_threshold, psnr_threshold = _similarity_thresholds()
    assert_similarity(
        model_name=MODEL_ID,
        vllm_image=vllm_output,
        diffusers_image=diffusers_output,
        width=WIDTH,
        height=HEIGHT,
        ssim_threshold=ssim_threshold,
        psnr_threshold=psnr_threshold,
    )


@pytest.mark.benchmark
@hardware_test(res={"cuda": ["H100", "B200"], "npu": "A3"}, num_cards=1)
def test_qwen_image_matches_diffusers_pixelwise(accuracy_artifact_root: Path) -> None:
    model = _model_name()
    output_dir = model_output_dir(accuracy_artifact_root, f"{MODEL_ID}-pixelwise")
    vllm_output_path = output_dir / "vllm_omni.png"
    diffusers_output_path = output_dir / "diffusers.png"

    vllm_output = _run_vllm_omni_qwen_image(model=model, output_path=vllm_output_path)
    diffusers_output = _run_diffusers_qwen_image(model=model, output_path=diffusers_output_path)

    print(f"{MODEL_ID} pixelwise generated images:")
    print(f"  vllm_omni: {vllm_output_path}")
    print(f"  diffusers: {diffusers_output_path}")

    mean_threshold, p99_threshold = _pixel_thresholds()
    assert_images_pixel_close(
        model_name=MODEL_ID,
        vllm_image=vllm_output,
        diffusers_image=diffusers_output,
        mean_threshold=mean_threshold,
        p99_threshold=p99_threshold,
    )
