# OmniCoordinator

---

## Table of Contents

- [Overview](#overview)
- [Motivation and Goals](#motivation-and-goals)
- [Architecture](#architecture)
- [Use Cases](#use-cases)
- [References](#references)

---
## Implementation Status (v0.18.0)

### Implemented
- **OmniCoordinator**
- **LoadBalancer** 
- **OmniCoordClientForStage**
- **OmniCoordClientForHub**
### Not completed yet
- End-to-end DP router integration in `vllm serve` path
- Full request-path routing/retry orchestration across all deployment modes
- Planned CLI flags wiring for `--omni-dp-*`



## Overview

### What is OmniCoordinator?

OmniCoordinator is a **singleton control-plane process** that provides **data-parallel routing** for vLLM‑Omni multi‑stage pipelines. It aggregates liveness and load signals (e.g., status, queue length, heartbeats) from all stage instances, then publishes an up‑to‑date instance list to AsyncOmni and API servers for **instance discovery**, **routing**, and **retry**.

---

## Motivation and Goals

In enterprise deployments, it is often a must to:

- Deploy **multiple replicas or instances** of each stage. 
- Dispatch incoming user requests to instances according to load balance policies.
- Dynamically add or drop instances without restarting the whole service.

OmniCoordinator addresses these requirements by acting as a central coordination service for **instance discovery**, **routing**, and **retry**.

### Features

- **Support multiple API Servers** [Planned]
- **Support multiple instances among each stage** [Planned]
- **Automatic instance discovery**  [Implemented]
  - Instances can be added or dropped dynamically.
  - API servers can discover instances in real time.
- **Task routing via LoadBalancer** [Implemented]
  - Selects target instances for tasks according to a load balance policy.

### Accuracy, Reliability, Performance

### Accuracy

- There is **no difference in end‑to‑end output** of the same request between:
  - DP enabled (DP > 1) and [Planned]
  - DP disabled (DP = 1) modes. [Planned]

### Reliability

- Multiple API servers and stage instances to avoid single points of failure. [Planned]
- Stage instance **heartbeat mechanism**. [Implemented]
- Request / task **retry mechanism** by selecting another instance on routing failure. [Planned]

### Performance

- The goodput (TPS) of any stage should be roughly **proportional to the number of instances** of that stage, assuming other resources are sufficient.

---

## Architecture

Multiple API servers, multiple stages, multiple instances. The overall design takes reference from vLLM.



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
  - Select target instances for tasks according to load balance policy.
- **OmniCoordinator**
  - Singleton process that collects status of all instances and publishes instance lists to all AsyncOmni / API servers.
  - Not the upstream vLLM OmniCoordinator; extra info is needed such as
  `stage_id` and ZMQ addresses of instances.
- **StageCoreProc**
  - Stage instance top‑level controller.
  - Receives tasks and sends events to OmniCoordinator.

---

## Use Cases

### 1. Single node: all stages with DP [Planned]

- **Scenario**: A user just wants to quickly serve a model with data parallelism.
- **Configuration**:
  - In CLI, run `vllm serve <model> --omni` on the head runtime.
  - Note: the DP-specific `--omni-dp-`* flags described in this doc are planned but not yet supported by the current `vllm serve` entrypoint.
- **Benefits**:
  - Simple to use.

### 2. Multiple nodes: stages separated across nodes with DP [Planned]

- **Scenario**: A user wants to boost goodput and fine‑tune the performance of each stage.
- **Configuration**:
  - Provide `--stage-id` and the currently supported master address flags (`--omni-master-address`, `--omni-master-port`).
  - Note: additional `--omni-dp-`* flags described in this doc are planned but not yet supported by the current `vllm serve` entrypoint.
  - Add `--headless` for non‑head runtimes.
- **Benefits**:
  - Flexible: stages and their replicas can be placed across nodes.

### OmniCoordinator process lifecycle [Planned]

- **Status**: The end-to-end CLI workflow in this section is **work in progress**. Some flags and flows described below are **not yet supported** by the current `vllm serve` entrypoint.
- Started and managed by the **head** (`without --headless`) runtime (planned).
- No separate startup command (planned).

Currently supported flags (as of this repo version):

- `--omni-master-address` / `--omni-master-port` – address of the Omni orchestrator (master) used for multi-stage coordination.
- `--stage-id` – launch a single stage; requires `--omni-master-address` and `--omni-master-port`.

Planned DP flags (not yet supported in `vllm serve`, subject to change):

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

### Modules


| Module                      | Description                                                                                        | New? |
| --------------------------- | -------------------------------------------------------------------------------------------------- | ---- |
| **API Server**              | OpenAI‑compatible HTTP API interface, supporting multiple deployments                              | No   |
| **AsyncOmni**               | Python API interface, request / task lifecycle management with retry mechanism                     | No   |
| **LoadBalancer**            | Base class of routing tasks (subclass like `RandomBalancer`)                                       | Yes  |
| **OmniCoordinator**         | Singleton process aggregating instance status and publishing instance list                         | Yes  |
| **OmniCoordClientForStage** | Used in stage instance side for sending events to OmniCoordinator                                  | Yes  |
| **OmniCoordClientForHub**   | Used on the AsyncOmni side for receiving stage instance list and their status (Instance Discovery) | Yes  |
| **StageCoreProc**           | Stage instance top‑level controller; receives tasks and sends events to OmniCoordinator            | No   |


## References

- **Core implementation**:
  - `vllm_omni/distributed/omni_coordinator/omni_coordinator.py`
  - `vllm_omni/distributed/omni_coordinator/omni_coord_client_for_stage.py`
  - `vllm_omni/distributed/omni_coordinator/omni_coord_client_for_hub.py`
  - `vllm_omni/distributed/omni_coordinator/load_balancer.py`
- **Tests**:
  - `tests/distributed/omni_coordinator/test_omni_coordinator.py`
  - `tests/distributed/omni_coordinator/test_omni_coord_client_for_stage.py`
  - `tests/distributed/omni_coordinator/test_omni_coord_client_for_hub.py`
  - `tests/distributed/omni_coordinator/test_load_balancer.py`

