# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AURA-only patch: ignore spurious Process.sentinel during multi-stage init.

Upstream ``CoreEngineProcManager.monitor_engine_liveness`` treats a ready
``Process.sentinel`` as child death and always SIGTERMs the stage engines.
During colocated AURA Stage0–3 bring-up an earlier stage's sentinel can become
permanently readable while that OS process is still alive; the false positive
kills healthy engines ~20–30s into a later stage's init (Stage2/3 EngineDead).

Shared ``stage_engine_core_proc_manager.py`` is left unchanged. This module
monkey-patches the class method when enabled via
``VLLM_AURA_STAGE_LIVENESS_PATCH=1`` (set by AURA duplex smoke serve).
"""

from __future__ import annotations

import logging
import os
from multiprocessing.process import BaseProcess
from typing import cast

logger = logging.getLogger("vllm_omni.aura_omni.stage_liveness_patch")

_PATCH_ATTR = "_aura_spurious_sentinel_patch"


def _monitor_engine_liveness(self) -> None:
    """Same contract as upstream, but verify ``is_alive()`` on sentinel readiness."""
    import multiprocessing.connection as connection

    sentinel_to_proc = {proc.sentinel: proc for proc in self.processes}
    sentinels = set(sentinel_to_proc.keys())
    poll_procs: set[BaseProcess] = set()

    while not self.manager_stopped.is_set():
        if sentinels:
            died_sentinels = connection.wait(list(sentinels), timeout=1.0)
        else:
            self.manager_stopped.wait(timeout=1.0)
            died_sentinels = []

        confirmed_dead = False
        for sentinel in died_sentinels:
            proc = sentinel_to_proc.get(cast(int, sentinel))
            if proc is None:
                sentinels.discard(sentinel)
                continue
            proc.join(timeout=0)
            if proc.is_alive():
                logger.warning(
                    "[AURA] spurious sentinel for still-alive %s (pid=%s); "
                    "switching to is_alive polling",
                    proc.name,
                    proc.pid,
                )
                sentinels.discard(sentinel)
                poll_procs.add(proc)
                continue
            sentinel_to_proc.pop(cast(int, sentinel), None)
            sentinels.discard(sentinel)
            poll_procs.discard(proc)
            if proc.exitcode not in (None, 0) and not self.manager_stopped.is_set():
                self.failed_proc_name = proc.name
            confirmed_dead = True

        still_polling: set[BaseProcess] = set()
        for proc in poll_procs:
            if proc.is_alive():
                still_polling.add(proc)
                continue
            if proc.exitcode not in (None, 0) and not self.manager_stopped.is_set():
                self.failed_proc_name = proc.name
            confirmed_dead = True
        poll_procs = still_polling

        if confirmed_dead:
            break
        if not sentinels and not poll_procs:
            break

    if self.manager_stopped.is_set() or any(not p.is_alive() for p in self.processes):
        self.shutdown()
    else:
        logger.error(
            "[AURA] monitor ended without dead procs; not shutting down. procs=%s",
            [(p.name, p.pid, p.is_alive(), p.exitcode) for p in self.processes],
        )


def install_aura_stage_liveness_patch() -> bool:
    """Patch shared manager method in-process. Idempotent. Returns True if applied."""
    from vllm_omni.engine.stage_engine_core_proc_manager import StageEngineCoreProcManager

    current = StageEngineCoreProcManager.monitor_engine_liveness
    if getattr(current, _PATCH_ATTR, False):
        return False

    patched = _monitor_engine_liveness
    setattr(patched, _PATCH_ATTR, True)
    StageEngineCoreProcManager.monitor_engine_liveness = patched  # type: ignore[method-assign]
    logger.info(
        "[AURA] installed StageEngineCoreProcManager.monitor_engine_liveness "
        "spurious-sentinel patch"
    )
    return True


def maybe_install_aura_stage_liveness_patch() -> bool:
    """Install only when ``VLLM_AURA_STAGE_LIVENESS_PATCH`` is truthy (default off)."""
    flag = os.environ.get("VLLM_AURA_STAGE_LIVENESS_PATCH", "").strip().lower()
    if flag not in ("1", "true", "yes", "on"):
        return False
    return install_aura_stage_liveness_patch()
