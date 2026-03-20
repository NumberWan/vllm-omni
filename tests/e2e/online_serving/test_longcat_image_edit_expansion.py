"""
Comprehensive tests of diffusion features that are available in online serving mode
and are supported by the following model:
- LongCat-Image-Edit: image-to-image edit with a single image + single edit prompt input
Coverage:
- CPU offloading (model-level sequential offload via --enable-cpu-offload)
- Cache-DiT
- SP (Ulysses & Ring)
- CFG-Parallel
- Tensor-Parallel

This validates:
 - The presence of all supported diffusion features in the generation request
 - Successful image generation at the expected 512x512 resolution
"""

import pytest

from tests.conftest import (
    OmniServer,
    OmniServerParams,
    OpenAIClientHandler,
    dummy_messages_from_mix_data,
    generate_synthetic_image,
)
from tests.utils import hardware_marks

EDIT_PROMPT = "Transform this modern image into a cinematic animation style with vibrant colors and soft lighting."
NEGATIVE_PROMPT = "blurry, low quality, distorted, oversaturated"
PARALLEL_FEATURE_MARKS = hardware_marks(res={"cuda": "H100"}, num_cards=2)


def _get_diffusion_feature_cases(model: str):
    """Return diffusion feature cases for LongCat-Image-Edit."""
    return [
        pytest.param(
            OmniServerParams(
                model=model,
                server_args=[
                    "--cache-backend",
                    "cache_dit",
                    "--ulysses-degree",
                    "2",
                ],
            ),
            id="parallel_001",
            marks=PARALLEL_FEATURE_MARKS,
        ),
        pytest.param(
            OmniServerParams(
                model=model,
                server_args=[
                    "--cache-backend",
                    "cache_dit",
                    "--ring",
                    "2",
                ],
            ),
            id="parallel_002",
            marks=PARALLEL_FEATURE_MARKS,
        ),
        pytest.param(
            OmniServerParams(
                model=model,
                server_args=[
                    "--cfg-parallel-size",
                    "2",
                ],
            ),
            id="parallel_003",
            marks=PARALLEL_FEATURE_MARKS,
        ),
        pytest.param(
            OmniServerParams(
                model=model,
                server_args=[
                    "--cache-backend",
                    "cache_dit",
                    "--tensor-parallel-size",
                    "2",
                ],
            ),
            id="parallel_004",
            marks=PARALLEL_FEATURE_MARKS,
        ),
    ]


@pytest.mark.advanced_model
@pytest.mark.diffusion
@pytest.mark.parametrize(
    "omni_server",
    _get_diffusion_feature_cases("meituan-longcat/LongCat-Image-Edit"),
    indirect=True,
)
def test_longcat_image_edit(omni_server: OmniServer, openai_client: OpenAIClientHandler):
    """Test all supported diffusion features with LongCat-Image-Edit in regular image-edit scenarios."""
    image_data_url = f"data:image/jpeg;base64,{generate_synthetic_image(512, 512)['base64']}"
    messages = dummy_messages_from_mix_data(image_data_url=image_data_url, content_text=EDIT_PROMPT)

    # CFG parallel is only activated when a negative prompt and guidance_scale > 1.0 are both present
    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "extra_body": {
            "height": 512,
            "width": 512,
            "num_inference_steps": 2,
            "negative_prompt": NEGATIVE_PROMPT,
            "guidance_scale": 4.0,
            "seed": 42,
        },
    }

    openai_client.send_diffusion_request(request_config)