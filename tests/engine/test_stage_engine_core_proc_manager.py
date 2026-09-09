"""Regression: multi-stage spawn must not false-kill earlier engines.

Before merging multi-stage bring-up changes into the production tree, these
CPU-only checks must pass. They catch the G1 failure mode where:

1. Stage-0 attaches and the inherited sentinel monitor starts.
2. Stage-1/2/3 spawn in the same parent.
3. Earlier multiprocessing sentinels become readable; ``is_alive()``/``poll()``
   latch a false exit; ``MPClient`` SIGTERMs a healthy engine.
4. Bring-up then aborts with ``Engine core initialization failed``.

Protocol / warmup / Stage3-silent tests do not cover this path.
"""

from __future__ import annotations

import inspect
import os
import threading
import time
from types import SimpleNamespace

from vllm.utils.system_utils import get_mp_context

from vllm_omni.engine.stage_engine_core_client import StageEngineCoreClientBase
from vllm_omni.engine.stage_engine_core_proc_manager import (
    StageEngineCoreProcManager,
    _os_child_is_running,
)


def test_os_child_running_for_live_process() -> None:
    ctx = get_mp_context()
    proc = ctx.Process(target=time.sleep, args=(30,), name="live-child")
    proc.start()
    try:
        assert proc.pid is not None
        os.kill(proc.pid, 0)
        assert _os_child_is_running(proc) is True
    finally:
        proc.terminate()
        proc.join(timeout=5)


def test_monitor_does_not_trip_when_sibling_process_spawns() -> None:
    """Sibling spawn must not make the stage monitor call shutdown()."""
    ctx = get_mp_context()
    first = ctx.Process(target=time.sleep, args=(30,), name="fake-stage0")
    sibling = ctx.Process(target=time.sleep, args=(30,), name="fake-stage1")
    first.start()
    mgr = StageEngineCoreProcManager.__new__(StageEngineCoreProcManager)
    mgr.processes = [first]
    mgr.manager_stopped = threading.Event()
    mgr.failed_proc_name = None
    shutdowns: list[bool] = []

    def _shutdown(timeout: float | None = None) -> None:
        shutdowns.append(True)
        mgr.manager_stopped.set()

    mgr.shutdown = _shutdown  # type: ignore[method-assign]
    monitor = threading.Thread(target=mgr.monitor_engine_liveness, name="test-monitor")
    monitor.start()
    try:
        time.sleep(0.2)
        sibling.start()
        time.sleep(2.5)
        assert shutdowns == [], "sibling spawn must not trip stage liveness monitor"
        assert first.pid is not None
        os.kill(first.pid, 0)
        assert _os_child_is_running(first) is True
    finally:
        mgr.manager_stopped.set()
        monitor.join(timeout=5)
        for proc in (first, sibling):
            if proc.pid is not None:
                proc.terminate()
                proc.join(timeout=5)


def test_os_child_running_for_missing_pid() -> None:
    assert _os_child_is_running(SimpleNamespace(pid=None)) is False
    assert _os_child_is_running(SimpleNamespace(pid=999_999_999)) is False


def test_stage_manager_forces_spawn_context() -> None:
    """Omni multi-stage bring-up must force spawn, not inherit fork."""
    source = inspect.getsource(StageEngineCoreProcManager.__init__)
    assert 'get_context("spawn")' in source or "get_context('spawn')" in source


def test_liveness_monitor_starts_only_via_explicit_api() -> None:
    """Client must not auto-start the monitor in __init__; runtime starts it later."""
    source = inspect.getsource(StageEngineCoreClientBase.__init__)
    assert "start_engine_core_monitor()" not in source
    assert "_liveness_monitor_started" in source
    start_src = inspect.getsource(StageEngineCoreClientBase.start_liveness_monitor)
    assert "start_engine_core_monitor()" in start_src
