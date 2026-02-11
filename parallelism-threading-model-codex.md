# Aerial Parallelism and Threading Model -codex (v1.0)

> Document Version: v1.0-codex
> Last Updated: 2026-02-11
> Scope: End-to-end parallel execution model across control-plane and driver runtime.

## Executive Summary
Aerial runtime is parallelized at four layers:

1. Message-level concurrency (IPC/FAPI ingest, epoll loops, tick thread).
2. Slot-level concurrency (multiple in-flight slot maps and per-slot task counters).
3. Channel-level concurrency (PDSCH/PDCCH/PBCH/CSI-RS on DL and PUSCH/PUCCH/PRACH/SRS on UL).
4. GPU-level concurrency (multi-stream kernels, event chaining, optional graph mode per channel).

The central architectural primitive is the slot map (`SlotMapDl`, `SlotMapUl`), which records task lifecycle, dependencies, and timing telemetry across the slot.

## Source Basis
Primary evidence files:
- Worker implementation:
  - `cuPHY-CP/cuphydriver/include/worker.hpp`
  - `cuPHY-CP/cuphydriver/src/common/worker.cpp`
- UL scheduling and dependencies:
  - `cuPHY-CP/cuphydriver/src/uplink/task_function_ul_aggr.cpp`
  - `cuPHY-CP/cuphydriver/include/slot_map_ul.hpp`
- DL scheduling and dependencies:
  - `cuPHY-CP/cuphydriver/src/downlink/task_function_dl_aggr.cpp`
  - `cuPHY-CP/cuphydriver/src/downlink/slot_map_dl.cpp`
- Control-plane thread loops:
  - `cuPHY-CP/cuphyl2adapter/lib/nvPHY/nv_phy_module.cpp`
  - `cuPHY-CP/cuphyl2adapter/lib/nvPHY/nv_tick_generator.cpp`
  - `cuPHY-CP/cuphyl2adapter/lib/nvPHY/nv_phy_epoll_context.cpp`

## Thread Domains
## Domain A: L2 adapter message processing
`PHY_module` creates `msg_processing` thread and configures:
- thread name (`pthread_setname_np`)
- SCHED_FIFO priority
- CPU affinity

It can run with epoll-driven fd events or transport wait loops depending on compile-time/runtime path.

Evidence: `nv_phy_module.cpp`.

## Domain B: Tick generation thread
`nv_tick_generator.cpp` creates a dedicated timer thread with mode-selectable implementation:
- poll method
- sleep method
- timer_fd + epoll method

Thread priority and CPU affinity are explicitly set, and thread lifecycle is refcount-governed.

## Domain C: Driver worker pools
`Worker` objects encapsulate driver task runners.

Key mechanics in `worker.cpp`:
- `Worker::run()` spawns worker thread
- sets thread name, affinity, policy
- enters worker routine loop and executes task objects

## Domain D: Auxiliary threads
Examples include:
- OAM/cell update threads in `nv_phy_module.cpp`
- optional capture/logging/metrics helper threads from config

## Slot-Map-Centric Concurrency
## UL slot map (`SlotMapUl`)
`slot_map_ul.hpp` exposes lifecycle gates:
- `waitULCTasksComplete(...)`
- `waitChannelStartTask()`
- `addChannelEndTask()`
- `addSlotEndTask()`
- `abortTasks()`

UL task workers repeatedly use these guards to coordinate channel setup/run/completion.

## DL slot map (`SlotMapDl`)
`slot_map_dl.cpp` manages:
- channel completion counters
- compression and GPU comm completion waits
- slot-end synchronization
- per-slot timing probes

Representative wait gates:
- `waitDLCDone(...)`
- `waitDlCompEnd()`
- `waitDlGpuCommEnd()`
- `waitSlotEndTask(...)`

## Task Graph Structure
## UL task graph
`task_function_ul_aggr.cpp` shows layered tasks such as:
- U-plane receive/order-kernel phases
- channel run tasks for PUSCH/PUCCH/PRACH/SRS
- completion/callback tasks
- early-UCI and timeout-aware branches

Frequent guard pattern:
- wait for prerequisites
- perform setup
- run channel phase
- register timing
- mark channel/slot completion
- abort on fatal condition

## DL task graph
`task_function_dl_aggr.cpp` similarly stages:
- channel tasks (PDSCH/PDCCH/PBCH/CSI-RS)
- compression task
- U-plane packet prep and send tasks
- callback and cleanup phases

DL introduces strict sequencing between channel completion, compression, and packet send via explicit events and slot-map state.

## Synchronization Primitives and Patterns
## CPU-side
- Atomics/counters in slot-map objects.
- Mutexes for serialized regions (for example compression task lock).
- Busy-wait with threshold logging in several wait functions.

## Event-loop side
- epoll fd muxing via `phy_epoll_context`.
- callback dispatch per ready fd.

## GPU-side
- CUDA events between channel kernels, compression, and packet send.
- Stream-level waits to maintain deterministic dependency order.

## Scheduling Policy and Affinity
Thread policy is intentionally explicit in critical paths:
- Workers and key control threads use `pthread_setschedparam(..., SCHED_FIFO, ...)` where configured.
- CPU pinning is set via `pthread_setaffinity_np`.

This design aims to reduce jitter and cross-core migration overhead for timing-sensitive slots.

## Timing and Telemetry Model
Both UL and DL slot maps expose rich timing fields (`timings`) recorded at:
- setup start/end
- run start/end
- callback start/end
- compression and GPU-comm phases
- order-kernel and fronthaul-related milestones

These timings support bottleneck localization and production regression checks.

## Parallelism Topology by Layer
## Control-plane layer
- Message ingest and tick generation are parallel but synchronized by shared slot state.

## Driver layer
- Multiple workers process tasks concurrently.
- Per-slot counters/barriers ensure correctness despite concurrency.

## Channel layer
- Multiple channels can run in same slot depending on configuration and task graph.
- Some phases serialize intentionally for correctness (for example compression ordering constraints).

## GPU layer
- Streams and events provide overlap where safe.
- Full overlap is constrained by slot deadlines and event dependencies.

## Common Bottlenecks
## 1) Wait loops under load
`wait*` functions can become latency amplifiers when prerequisites are delayed.

## 2) Over-serialization
Locks or conservative event dependencies can reduce intended overlap.

## 3) Affinity mismatch
Thread-core mappings that conflict with system contention can increase jitter.

## 4) Channel imbalance
UL/DL channel mixes with skewed compute cost cause uneven worker utilization.

## 5) Timeout cascades
Single delayed dependency can trigger multiple follow-on aborts and slot-level failures.

## Failure Patterns and Diagnostics
## Pattern A: ULC prerequisite timeout
Observed via warnings like timeout waiting for ULC tasks in UL task functions.

## Pattern B: channel setup/run failures
Any channel run failure often causes abort propagation in that slot map.

## Pattern C: compression or GPU comm delay
DL wait functions report prolonged waits for compression or GPU comm completion.

## Pattern D: order-kernel induced back-pressure
UL order path can stall downstream channel completion paths when delayed.

## Pattern E: thread policy/affinity setup failures
APIs emit logs if thread policy or affinity calls fail; this can silently degrade determinism if ignored.

## Tuning and Optimization Recommendations
1. Validate core affinity map against hardware NUMA and NIC/GPU locality.
2. Benchmark each tick mode under production-like load before choosing default.
3. Track slot-map timing percentiles and alarm on drift by phase.
4. Revisit serialized sections only after confirming correctness constraints.
5. Keep timeout thresholds aligned with measured platform jitter, not lab ideal values.

## Validation Checklist
1. Thread launch integrity:
- verify all critical threads set expected name/policy/affinity.

2. Slot-map lifecycle:
- run stress test and ensure channel-end and slot-end counters always converge.

3. Timeout behavior:
- inject controlled delay at specific phases and verify bounded error propagation.

4. Parallel efficiency:
- compare measured overlap against expected overlap in UL and DL paths.

5. Stability under mixed traffic:
- run mixed PDSCH/PUSCH/PRACH/SRS load and verify no deadlocks or starvation.

## Refactor Coverage
This report consolidates and expands threading/parallel content from:
- `MULTITHREAD_PARALLEL_ARCHITECTURE-claude.md`
- `multi_task_parallel_architecture.md`
- `multi_task_parallel_architecture-v0.md`
- `UPLINK_MULTITHREAD_ARCHITECTURE.md`
- threading sections of `UPLINK_PUSCH_PROCESSING_ARCHITECTURE.md`

Legacy snapshots are preserved:
- `archive/aerial-reports-v1.1-2026-02-11/`

## Related Consolidated Reports
- `architecture-overview-codex.md`
- `control-plane-runtime-codex.md`
- `downlink-pdsch-pipeline-codex.md`
- `uplink-pusch-prach-pipeline-codex.md`
