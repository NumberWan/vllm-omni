# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from time import time

from vllm_omni.distributed.omni_coordinator import (
    InstanceInfo,
    LeastQueueLengthBalancer,
    RandomBalancer,
    StageStatus,
    RoundRobinBalancer,
)


def test_load_balancer_select_returns_valid_index():
    """Verify RandomBalancer.select() returns a valid index for instances."""
    # Task structure mirrors async_omni; RandomBalancer ignores task contents.
    task: dict = {
        "request_id": "test",
        "engine_inputs": None,
        "sampling_params": None,
    }

    now = time()
    instances = [
        InstanceInfo(
            input_addr="tcp://host:10001",
            output_addr="tcp://host:10001-out",
            stage_id=0,
            status=StageStatus.UP,
            queue_length=0,
            last_heartbeat=now,
            registered_at=now,
        ),
        InstanceInfo(
            input_addr="tcp://host:10002",
            output_addr="tcp://host:10002-out",
            stage_id=0,
            status=StageStatus.UP,
            queue_length=1,
            last_heartbeat=now,
            registered_at=now,
        ),
        InstanceInfo(
            input_addr="tcp://host:10003",
            output_addr="tcp://host:10003-out",
            stage_id=1,
            status=StageStatus.UP,
            queue_length=2,
            last_heartbeat=now,
            registered_at=now,
        ),
    ]

    balancer = RandomBalancer()

    index = balancer.select(task, instances)

    assert isinstance(index, int)
    assert 0 <= index < len(instances)


def test_round_robin_balancer_cycles_instances():
    now = time()
    instances = [
        InstanceInfo(
            input_addr="tcp://host:10001",
            output_addr="tcp://host:10001-out",
            stage_id=0,
            status=StageStatus.UP,
            queue_length=2,
            last_heartbeat=now,
            registered_at=now,
        ),
        InstanceInfo(
            input_addr="tcp://host:10002",
            output_addr="tcp://host:10002-out",
            stage_id=0,
            status=StageStatus.UP,
            queue_length=1,
            last_heartbeat=now,
            registered_at=now,
        ),
        InstanceInfo(
            input_addr="tcp://host:10003",
            output_addr="tcp://host:10003-out",
            stage_id=1,
            status=StageStatus.UP,
            queue_length=0,
            last_heartbeat=now,
            registered_at=now,
        ),
    ]

    balancer = RoundRobinBalancer()
    results = [balancer.select({}, instances) for _ in range(5)]

    # Default start_index=0 => 0,1,2,0,1
    assert results == [0, 1, 2, 0, 1]


def test_least_queue_length_balancer_picks_min_queue():
    now = time()
    instances = [
        InstanceInfo(
            input_addr="tcp://host:10001",
            output_addr="tcp://host:10001-out",
            stage_id=0,
            status=StageStatus.UP,
            queue_length=2,
            last_heartbeat=now,
            registered_at=now,
        ),
        InstanceInfo(
            input_addr="tcp://host:10002",
            output_addr="tcp://host:10002-out",
            stage_id=0,
            status=StageStatus.UP,
            queue_length=0,
            last_heartbeat=now,
            registered_at=now,
        ),
        InstanceInfo(
            input_addr="tcp://host:10003",
            output_addr="tcp://host:10003-out",
            stage_id=1,
            status=StageStatus.UP,
            queue_length=5,
            last_heartbeat=now,
            registered_at=now,
        ),
    ]

    balancer = LeastQueueLengthBalancer()
    index = balancer.select({}, instances)
    assert index == 1
