# SPDX-License-Identifier: Apache-2.0
"""Ephemeral Stage0 bind uses r.stage0_t{N}, not resumable r.stage0."""

from __future__ import annotations

from types import SimpleNamespace

from vllm_omni.engine.duplex.contracts import (
    duplex_ephemeral_stage_request_id,
    duplex_resource_request_id,
)
from vllm_omni.engine.duplex.session_runner import DuplexSessionRunner


def test_stage0_request_id_ephemeral_when_not_resumable() -> None:
    runner = DuplexSessionRunner.__new__(DuplexSessionRunner)
    runner.session = SimpleNamespace(
        session_id="duplex-test",
        turn_id=3,
        capabilities=SimpleNamespace(supports_core_resumable_request=False),
    )
    rid = DuplexSessionRunner._stage0_request_id(runner, epoch=0, turn_id=3)
    from vllm_omni.engine.duplex.contracts import DuplexFence

    expected = duplex_ephemeral_stage_request_id(DuplexFence("duplex-test", epoch=0, turn_id=3), stage_id=0)
    assert rid == expected
    assert rid.endswith(".r.stage0_t3")
    assert not rid.endswith(".r.stage0")


def test_stage0_request_id_resumable_keeps_stable_role() -> None:
    runner = DuplexSessionRunner.__new__(DuplexSessionRunner)
    runner.session = SimpleNamespace(
        session_id="duplex-test",
        turn_id=3,
        capabilities=SimpleNamespace(supports_core_resumable_request=True),
    )
    rid = DuplexSessionRunner._stage0_request_id(runner, epoch=1, turn_id=3)
    from vllm_omni.engine.duplex.contracts import DuplexFence

    expected = duplex_resource_request_id(DuplexFence("duplex-test", epoch=1, turn_id=3), "stage0")
    assert rid == expected
    assert rid.endswith(".r.stage0")
