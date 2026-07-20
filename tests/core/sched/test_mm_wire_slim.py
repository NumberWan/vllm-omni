"""Unit tests for AURA mm_features Engine→Worker wire slim."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
import torch

from vllm.multimodal.inputs import MultiModalFieldElem, MultiModalKwargsItem, MultiModalSharedField
from vllm_omni.core.sched.omni_ar_scheduler import slim_mm_features_for_worker_ipc


@dataclass
class _FakeFeat:
    data: Any
    modality: str = "video"
    identifier: str = "x"
    mm_position: Any = None
    mm_hash: str | None = None


def _item_with_pixels(pixel_bytes: int = 1024) -> MultiModalKwargsItem:
    field = MultiModalSharedField(batch_size=1)
    return MultiModalKwargsItem(
        {
            "pixel_values_videos": MultiModalFieldElem(
                data=torch.zeros(pixel_bytes, dtype=torch.uint8),
                field=field,
            ),
            "video_grid_thw": MultiModalFieldElem(
                data=torch.tensor([[1, 2, 2]]),
                field=field,
            ),
            "second_per_grid_ts": MultiModalFieldElem(
                data=torch.tensor([1.0]),
                field=field,
            ),
        }
    )


def test_slim_disabled_by_default_preserves_pixels() -> None:
    """Default must keep pixels so chunked encode / cache eviction can re-run."""
    feats = [_FakeFeat(data=_item_with_pixels()) for _ in range(3)]
    out = slim_mm_features_for_worker_ipc(feats, encode_idxs={2})
    assert out is feats
    assert all("pixel_values_videos" in f.data for f in out)


def test_slim_keeps_encode_idx_full(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_OMNI_SLIM_MM_IPC", "1")
    feats = [_FakeFeat(data=_item_with_pixels()) for _ in range(3)]
    out = slim_mm_features_for_worker_ipc(feats, encode_idxs={2})
    assert out is not None
    assert "pixel_values_videos" in out[2].data
    assert "pixel_values_videos" not in out[0].data
    assert "video_grid_thw" in out[0].data
    assert "second_per_grid_ts" in out[0].data


def test_slim_noop_when_all_encoded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_OMNI_SLIM_MM_IPC", "1")
    feats = [_FakeFeat(data=_item_with_pixels()) for _ in range(2)]
    out = slim_mm_features_for_worker_ipc(feats, encode_idxs={0, 1})
    assert out is not None
    assert all("pixel_values_videos" in f.data for f in out)
