# OmniCoordinator

---

## Table of Contents

- [Overview](#overview)
- [Motivation](#motivation)
- [Features](#features)
- [Accuracy, Reliability, Performance](#accuracy-reliability-performance)
- [Architecture](#architecture)
- [Use Cases](#use-cases)
- [API Design](#api-design)
- [Test Cases](#test-cases)
- [References](#references)

---

## Overview

### Data Parallelism Routing for vLLM‑Omni

OmniCoordinator provides **data parallel routing** for vLLM‑Omni multi‑stage
pipelines. It is a **singleton process** that collects status of all
instances of all stages and publishes instance lists to AsyncOmni and API
servers.

---

## Motivation

In enterprise deployments, it is often a must to:

- Deploy **multiple replicas or instances** of each stage.
- Dispatch incoming user requests to instances according to load balance
  policies.
- Dynamically add or drop instances without restarting the whole service.

OmniCoordinator addresses these requirements by acting as a central
coordination service for **instance discovery**, **routing**, and **retry**.

---

## Features

- **Support multiple API Servers**
- **Support multiple instances among each stage**
- **Automatic instance discovery**
  - Instances can be added or dropped dynamically.
  - API servers can discover instances in real time.
- **Task routing via LoadBalancer**
  - Tasks are dispatched according to a load balance policy
    (Random for now).

---

## Accuracy, Reliability, Performance

### Accuracy

- There is **no difference in end‑to‑end output** of the same request between:
  - DP enabled (DP > 1) and
  - DP disabled (DP = 1) modes.

### Reliability

- Multiple API servers and stage instances to avoid single points of failure.
- Stage instance **heartbeat mechanism**.
- Request / task **retry mechanism** by selecting another instance on routing
  failure.

### Performance

- The goodput (TPS) of any stage should be roughly **proportional to the
  number of instances** of that stage, assuming other resources are sufficient.

---

## Architecture

Multiple API servers, multiple stages, multiple instances. The overall design
takes reference from vLLM.

- **API Server**
  - OpenAI‑compatible HTTP API.
  - Supports multiple deployments to prevent single point of failure.
  - Clients may send requests to API servers by random choice.
- **AsyncOmni**
  - Python API.
  - Request / task lifecycle management with retry mechanism.
- **Instance Discovery**
  - Communicates with OmniCoordinator.
  - Collects status of all instances of all stages.
- **LoadBalancer**
  - Dispatches tasks according to load balance policy (Random).
- **OmniCoordinator**
  - Singleton process that collects status of all instances and publishes
    instance lists to all AsyncOmni / API servers.
  - Not the upstream vLLM OmniCoordinator; extra info is needed such as
    `stage_id` and ZMQ addresses of instances.
- **StageCoreProc**
  - Stage instance top‑level controller.
  - Receives tasks and sends events to OmniCoordinator.

---

## Use Cases

### 1. Single node: all stages with DP

- **Scenario**: A user just wants to quickly serve a model with data
  parallelism.
- **Configuration**:
  - In CLI, omit the stage‑related arguments (`--stage-id`) and coordinator
    related arguments (`--omni-dp-address`, `--omni-dp-rpc-port`).
- **Benefits**:
  - Simple to use.

### 2. Multiple nodes: stages separated across nodes with DP

- **Scenario**: A user wants to boost goodput and fine‑tune the performance of
  each stage.
- **Configuration**:
  - Provide `--stage-id` and all other `--omni-dp-*` arguments.
  - Add `--headless` for non‑head runtimes.
- **Benefits**:
  - Flexible: stages and their replicas can be placed across nodes.

### OmniCoordinator process lifecycle

- Started and managed by the **head** (`without --headless`) runtime:

```bash
vllm serve <model> --omni
```

- No separate startup command.

Additional DP arguments:

- `--omni-dp-size-local` – data parallelism size of this runtime.
- `--omni-dp-address` – IP address of OmniCoordinator.
- `--omni-dp-rpc-port` – port number of OmniCoordinator’s ROUTER socket.

Original vLLM data parallel arguments are **not applicable** to vLLM‑Omni DP:

- `--data-parallel-size`
- `--data-parallel-size-local`
- `--data-parallel-address`
- `--data-parallel-rpc-port`

Headless runtimes can be started (or stopped) anytime after the head runtime,
on the same or different nodes, to provide additional instances for any stage
(including stage 0).

---

## API Design

### Modules

| Module                      | Description                                                                                                | New?  |
| --------------------------- | ---------------------------------------------------------------------------------------------------------- | ----- |
| **API Server**              | OpenAI‑compatible HTTP API interface, supporting multiple deployments                                     | No    |
| **AsyncOmni**               | Python API interface, request / task lifecycle management with retry mechanism                             | No    |
| **LoadBalancer**            | Base class of routing tasks (subclass like `RandomBalancer`)                                              | Yes   |
| **OmniCoordinator**         | Singleton process aggregating instance status and publishing instance list                                 | Yes   |
| **OmniCoordClientForStage** | Used in stage instance side for sending events to OmniCoordinator                                         | Yes   |
| **OmniCoordClientForHub**   | Used on the AsyncOmni side for receiving stage instance list and their status (Instance Discovery)        | Yes   |
| **StageCoreProc**           | Stage instance top‑level controller; receives tasks and sends events to OmniCoordinator                    | No    |

### Message Protocol (control plane)

**Task** (AsyncOmni → StageCoreProc, simplified example):

```json
{
  "session_id": "uuid-string",
  "request_id": "uuid-string",
  "sampling_params": { ... },
  "retry_count": 0,
  "user_inputs": { ... }
}
```

**Instance Event**  
`OmniCoordClientForStage` (StageCoreProc) → OmniCoordinator:

```json
{
  "zmq_addr": "tcp://host:port",
  "stage_id": 0,
  "status": "up | down | error",
  "queue_length": 5,
  "event_type": "update | heartbeat"
}
```

**Instance List**  
OmniCoordinator → `OmniCoordClientForHub` (AsyncOmni):

```json
{
  "instances": [
    {
      "zmq_addr": "tcp://host:port",
      "stage_id": 0,
      "status": "up | down | error",
      "queue_length": 5,
      "last_heartbeat": 12578.1,
      "registered_at": 2354.4
    }
  ],
  "timestamp": 12345.6
}
```

### Major API

Package: `vllm_omni.distributed.omni_coordinator`

- **InstanceStatus**
  - `UP`: instance is ready and available.
  - `DOWN`: instance is shut down gracefully.
  - `ERROR`: instance encountered an error or timeout.
- **OmniCoordinator**
  - Initializes with `router_zmq_addr`, `pub_zmq_addr`, `heartbeat_timeout`.
  - Listens for instance events, handles heartbeat timeout, and publishes
    instance lists.
  - `close()` cleans up ZMQ sockets and background threads.
- **OmniCoordClientForStage**
  - Used in stage instances to send events to OmniCoordinator.
  - Automatically registers on `__init__`.
  - `update_info(status, queue_length)` sends status / load updates.
  - `close()` sends a final `DOWN` event and closes the socket.
- **OmniCoordClientForHub**
  - Used on AsyncOmni side to receive instance lists.
  - Subscribes to OmniCoordinator via PUB/SUB.
  - `get_instance_list()` returns current cached list.
  - `get_instances_for_stage(stage_id)` filters by stage id.
  - `close()` closes the SUB socket and stops background thread.
- **LoadBalancer**
  - Abstract base class with:
    - `select(task, instances) -> int`
  - `RandomBalancer` is a simple implementation that returns a random index.

---

## Test Cases

Implementation is covered by unit tests under
`tests/distributed/omni_coordinator/`. When changing OmniCoordinator or
related components, contributors should at least run these tests:

- `test_omni_coordinator.py`
  - Registration broadcast, heartbeat timeout handling, instance shutdown handling.
- `test_load_balancer.py`
  - Basic sanity checks for `LoadBalancer.select()`.
- `test_omni_coord_client_for_stage.py`
  - Auto‑registration on init, status / queue length updates, graceful close.
- `test_omni_coord_client_for_hub.py`
  - Caching of instance lists, filtering by `stage_id`, close behavior.

For end‑to‑end verification, you can also:

- Start an **omni** serving deployment with DP>1 (single node or multi‑node).
- Scale the number of stage instances and confirm:
  - Requests are distributed across instances.
  - Failed instances are removed from routing after heartbeat timeout.
  - Goodput (TPS) increases when adding more healthy instances, up to
    hardware limits.

---

## References

- **Core implementation**:
  - `vllm_omni/distributed/omni_coordinator/omni_coordinator.py`
  - `vllm_omni/distributed/omni_coordinator/omni_coord_client_for_stage.py`
  - `vllm_omni/distributed/omni_coordinator/omni_coord_client_for_hub.py`
  - `vllm_omni/distributed/omni_coordinator/messages.py`
- **Tests**:
  - `tests/distributed/omni_coordinator/test_omni_coordinator.py`
  - `tests/distributed/omni_coordinator/test_omni_coord_client_for_stage.py`
  - `tests/distributed/omni_coordinator/test_omni_coord_client_for_hub.py`
  - `tests/distributed/omni_coordinator/test_load_balancer.py`

