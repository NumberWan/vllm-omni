"""Forbid deriving ``txt_seq_lens`` from ``mask.sum()`` in Qwen-Image pipelines.

RoPE text length must follow padded embed width, not valid-token count.
This only bans the ``mask.sum()`` assignment pattern; it does not require a
specific helper. Feel free to modify or remove if pipeline wiring changes.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

_REPO_ROOT = Path(__file__).resolve().parents[4]
QWEN_IMAGE_DIR = _REPO_ROOT / "vllm_omni" / "diffusion" / "models" / "qwen_image"

_MASK_SUM_TXT_SEQ_LENS_RE = re.compile(
    r"(?:negative_)?txt_seq_lens\s*=\s*[^\n]*mask\.sum\s*\(",
    re.MULTILINE,
)


@pytest.mark.parametrize(
    "pipeline_path",
    sorted(QWEN_IMAGE_DIR.glob("pipeline_qwen_image*.py")),
    ids=lambda path: path.name,
)
def test_qwen_image_pipeline_does_not_set_txt_seq_lens_from_mask_sum(pipeline_path: Path):
    source = pipeline_path.read_text(encoding="utf-8")
    match = _MASK_SUM_TXT_SEQ_LENS_RE.search(source)
    assert match is None, (
        f"{pipeline_path.name} must not derive txt_seq_lens from mask.sum(); found: {match.group(0)!r}"
    )
