# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Video timestamps run on across the clips of one prompt (native AURA patch)."""

from __future__ import annotations

import pytest
from vllm.model_executor.models.qwen3_5 import Qwen3_5ProcessingInfo
from vllm.model_executor.models.qwen3_vl import Qwen3VLMultiModalProcessor

from vllm_omni.model_executor.models.aura_omni import qwen3_5 as aura_qwen3_5

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _clip(duration: float) -> dict:
    return {"duration": duration, "fps": 2.0, "frames_indices": [0, 1], "total_num_frames": 2}


def test_clips_of_one_prompt_share_one_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    # Stock vLLM restarts every clip at 0 s.
    monkeypatch.setattr(Qwen3_5ProcessingInfo, "_get_video_second_idx", lambda self, metadata, *a, **k: [0.25])
    calls: list[list[float]] = []

    def fake_main(self, *args, **kwargs):
        info = aura_qwen3_5.AuraQwen3_5ProcessingInfo.__new__(aura_qwen3_5.AuraQwen3_5ProcessingInfo)
        for clip in (_clip(1.0), _clip(1.0), _clip(0.5), _clip(1.0)):
            calls.append(info._get_video_second_idx(clip))
        return "processed"

    monkeypatch.setattr(Qwen3VLMultiModalProcessor, "_apply_hf_processor_main", fake_main)
    processor = aura_qwen3_5.AuraQwen3VLMultiModalProcessor.__new__(aura_qwen3_5.AuraQwen3VLMultiModalProcessor)

    assert processor._apply_hf_processor_main() == "processed"
    assert calls == [[0.25], [1.25], [2.25], [2.75]]

    # The next prompt starts from 0 s again.
    calls.clear()
    processor._apply_hf_processor_main()
    assert calls[0] == [0.25]


def test_aura_registers_the_running_clock_processor() -> None:
    assert issubclass(aura_qwen3_5.AuraQwen3VLMultiModalProcessor, Qwen3VLMultiModalProcessor)
