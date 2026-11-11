# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Gateway (9999 page) duplex actor: playback acks and per-vision-turn tool budget."""

from __future__ import annotations

import asyncio
import base64
from types import SimpleNamespace

import pytest

from examples.online_serving.aura_v2_gateway.gateway import duplex_actor

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _actor() -> tuple[duplex_actor.DuplexActor, list[dict]]:
    actor = duplex_actor.DuplexActor(
        session_id="s",
        active_sessions=lambda: 1,
        backend_url="ws://unused",
        system_prompt="",
        enable_toolcall=False,
        tools=None,
        history=None,
    )
    sent: list[dict] = []

    async def capture(event: dict) -> None:
        sent.append(event)

    actor._backend_send = capture
    return actor, sent


def test_whole_reply_audio_is_acked_once_sent() -> None:
    """Unacked sent audio keeps the reply "in progress" on the backend, which then
    refuses the next tool follow-up with response_already_active."""
    actor, sent = _actor()
    pcm_100ms = base64.b64encode(b"\x00\x00" * 2400).decode()  # 24 kHz PCM16
    delta = {"type": "response.output_audio.delta", "response_id": "r1", "item_id": "item_r1", "delta": pcm_100ms}

    async def run() -> None:
        for _ in range(3):
            await actor._ack_playback(delta, delta["type"], "r1")
        assert sent == []
        done = {"type": "response.output_audio.done", "response_id": "r1"}
        await actor._ack_playback(done, done["type"], "r1")
        # response.done afterwards must not ack the same reply twice.
        await actor._ack_playback({"type": "response.done"}, "response.done", "r1")

    asyncio.run(run())
    assert sent == [
        {"type": "playback.ack", "response_id": "r1", "played_ms": 301, "committed_ms": 301, "item_id": "item_r1"}
    ]


def test_silent_reply_sends_no_ack() -> None:
    actor, sent = _actor()
    asyncio.run(actor._ack_playback({"type": "response.done"}, "response.done", "r2"))
    assert sent == []


def test_each_vision_turn_gets_its_own_tool_budget() -> None:
    """Native runs MAX_TOOL_ROUNDS per video chunk. Without a reset, the round cap
    stopped every later vision turn's follow-up and the model looped on tools."""
    actor, sent = _actor()
    actor._follow.update(count=duplex_actor._MAX_TOOL_FOLLOWUPS, gen=7)
    chunk = SimpleNamespace(chunk_id=1, data={"video_url": "data:video/jpeg;base64,AAAA,BBBB"})
    asyncio.run(actor._vision_turn(chunk))
    assert actor._follow["count"] == 0
    assert actor._follow["gen"] == 8
    assert [event["type"] for event in sent] == ["input_audio_buffer.append", "input_audio_buffer.commit"]
