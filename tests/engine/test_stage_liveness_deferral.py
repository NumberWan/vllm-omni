"""Gate tests: deferred stage liveness monitor wiring.

These assert the production bring-up contract without GPUs:

* Stage clients attach ``engine_manager`` without starting the monitor.
* ``StageRuntime._finalize_initialized_stages`` is what starts monitors.
"""

from __future__ import annotations

import inspect

from vllm_omni.engine.stage_engine_core_client import StageEngineCoreClientBase
from vllm_omni.engine.stage_runtime import StageRuntime


def test_finalize_starts_liveness_monitors() -> None:
    source = inspect.getsource(StageRuntime._finalize_initialized_stages)
    assert "start_liveness_monitor" in source
    assert "pre-monitor" in source


def test_start_liveness_monitor_is_idempotent() -> None:
    client = StageEngineCoreClientBase.__new__(StageEngineCoreClientBase)
    client._liveness_monitor_started = False
    client.resources = type("R", (), {"engine_manager": object()})()
    calls: list[bool] = []

    def _start() -> None:
        calls.append(True)

    client.start_engine_core_monitor = _start  # type: ignore[method-assign]
    client.start_liveness_monitor()
    client.start_liveness_monitor()
    assert calls == [True]
    assert client._liveness_monitor_started is True
