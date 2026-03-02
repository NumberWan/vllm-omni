# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class StageStatus(str, Enum):
    """Enumeration for stage instance status."""

    UP = "up"  # Instance is ready and available
    DOWN = "down"  # Instance is shutdown gracefully
    ERROR = "error"  # Instance encountered an error or timeout


@dataclass
class InstanceEvent:
    """Wire payload from OmniCoordClientForStage to OmniCoordinator.

    Schema for Stage → Coordinator events over ZMQ:
    zmq_addr, stage_id, status, queue_length, event_type.
    """

    zmq_addr: str
    stage_id: int
    event_type: str  # "update" | "heartbeat"
    status: StageStatus
    queue_length: int


@dataclass
class InstanceInfo:
    """Metadata for a single stage instance.

    This type is stored in OmniCoordinator's internal registry and is also
    published to hubs via :class:`InstanceList`.
    """

    zmq_addr: str  # StageCoreProc instance ZMQ address (e.g., "tcp://host:port")
    stage_id: int  # Stage ID of this instance
    status: StageStatus  # Current status of the instance
    queue_length: int  # Current queue length of this instance
    last_heartbeat: float  # Timestamp of the last heartbeat received (seconds)
    registered_at: float  # Timestamp when the instance was registered (seconds)


@dataclass
class InstanceList:
    """Container for instance list updates.

    OmniCoordinator publishes an :class:`InstanceList` whenever its view of
    active instances changes. OmniCoordClientForHub caches the latest value
    and exposes it to AsyncOmni and the load balancer.
    """

    instances: list[InstanceInfo]
    timestamp: float  # Time when the list was last updated (seconds)
