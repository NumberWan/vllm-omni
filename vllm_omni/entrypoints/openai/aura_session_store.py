# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""In-process registry for AURA streaming SessionHistory objects."""

from __future__ import annotations

import threading
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.aura_session_history import SessionHistory

__all__ = [
    "create_session_id",
    "register_session",
    "get_session_history",
    "unregister_session",
    "clear_all_sessions",
]

_LOCK = threading.Lock()
_SESSIONS: dict[str, SessionHistory] = {}


def create_session_id() -> str:
    return f"aura-{uuid.uuid4().hex}"


def register_session(session_id: str, history: SessionHistory) -> None:
    with _LOCK:
        _SESSIONS[session_id] = history


def get_session_history(session_id: str) -> SessionHistory | None:
    with _LOCK:
        return _SESSIONS.get(session_id)


def unregister_session(session_id: str) -> None:
    with _LOCK:
        _SESSIONS.pop(session_id, None)


def clear_all_sessions() -> None:
    """Clear all registered sessions (for tests)."""
    with _LOCK:
        _SESSIONS.clear()
