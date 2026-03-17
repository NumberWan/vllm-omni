"""L4 diffusion feature expansion tests for Bagel.

This file follows the structure of:
- tests/e2e/online_serving/test_qwen_image_edit_expansion.py
- RFC #1832 (L4 diffusion e2e tests)

Covered Bagel features (from RFC #1217):
- TeaCache
- Cache-DiT
- CFG-Parallel
- Tensor-Parallel
"""

import pytest

from tests.conftest import (
    OmniServer,
    OmniServerParams,
    OpenAIClientHandler,
    dummy_messages_from_mix_data,
)
from tests.utils import hardware_marks

BAGEL_MODEL = "ByteDance-Seed/BAGEL-7B-MoT"

PROMPT = "A futuristic city skyline at twilight, cyberpunk style, ultra-detailed, high resolution."
NEGATIVE_PROMPT = "low quality, blurry, distorted, deformed, watermark"

SINGLE_CARD_FEATURE_MARKS = hardware_marks(res={"cuda": "H100"})
PARALLEL_FEATURE_MARKS = hardware_marks(res={"cuda": "H100"}, num_cards=2)


def _get_diffusion_feature_cases_for_bagel():
    """Return L4 diffusion feature cases for Bagel.

    Each case enables at least one of the Bagel-supported diffusion
    acceleration features listed in RFC #1217:
    TeaCache, Cache-DiT, CFG-Parallel, Tensor-Parallel.
    """

    return [
        # TeaCache (single-card)
        pytest.param(
            OmniServerParams(
                model=BAGEL_MODEL,
                server_args=[
                    "--cache-backend",
                    "tea_cache",
                ],
            ),
            id="single_card_teacache",
            marks=SINGLE_CARD_FEATURE_MARKS,
        ),
        # Cache-DiT (single-card)
        pytest.param(
            OmniServerParams(
                model=BAGEL_MODEL,
                server_args=[
                    "--cache-backend",
                    "cache_dit",
                ],
            ),
            id="single_card_cache_dit",
            marks=SINGLE_CARD_FEATURE_MARKS,
        ),
        # CFG-Parallel size 2 (2 GPUs, TeaCache backend)
        pytest.param(
            OmniServerParams(
                model=BAGEL_MODEL,
                server_args=[
                    "--cache-backend",
                    "tea_cache",
                    "--cfg-parallel-size",
                    "2",
                ],
            ),
            id="parallel_cfg_2",
            marks=PARALLEL_FEATURE_MARKS,
        ),
        # Tensor-Parallel size 2 (2 GPUs, Cache-DiT backend)
        pytest.param(
            OmniServerParams(
                model=BAGEL_MODEL,
                server_args=[
                    "--cache-backend",
                    "cache_dit",
                    "--tensor-parallel-size",
                    "2",
                ],
            ),
            id="parallel_tp_2",
            marks=PARALLEL_FEATURE_MARKS,
        ),
    ]


@pytest.mark.advanced_model
@pytest.mark.diffusion
@pytest.mark.parametrize(
    "omni_server",
    _get_diffusion_feature_cases_for_bagel(),
    indirect=True,
)
def test_bagel_diffusion_features(
    omni_server: OmniServer,
    openai_client: OpenAIClientHandler,
):
    """L4 diffusion feature coverage for Bagel on H100.

    This test exercises:
    - TeaCache
    - Cache-DiT
    - CFG-Parallel (size=2)
    - Tensor-Parallel (size=2)

    Validation is delegated to assert_diffusion_response in tests.conftest,
    which checks output dimensions and basic correctness.
    """

    messages = dummy_messages_from_mix_data(content_text=PROMPT)

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "extra_body": {
            "height": 512,
            "width": 512,
            "num_inference_steps": 2,
            # Enable CFG for models that use classifier-free guidance
            "negative_prompt": NEGATIVE_PROMPT,
            "true_cfg_scale": 4.0,
            "seed": 42,
        },
    }

    openai_client.send_diffusion_request(request_config)

