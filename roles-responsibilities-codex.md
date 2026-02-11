# Aerial Component Roles and Responsibilities -codex (v1.0)

> Document Version: v1.0-codex
> Last Updated: 2026-02-11
> Scope: Canonical ownership model for core Aerial runtime components.

## Executive Summary
This document defines who owns what across the Aerial stack. The most important architectural boundary is between:

- control-plane orchestration (`cuphycontroller`, adapters, FAPI handling), and
- data-plane execution (`cuphydriver` + `cuPHY` channel kernels + fronthaul).

Clear role separation is essential because the same slot touches parser logic, task scheduling, CUDA compute, compression, and transport.

## Source Basis
Key ownership evidence files:
- Controller and parser:
  - `cuPHY-CP/cuphycontroller/src/cuphydriver.cpp`
  - `cuPHY-CP/cuphycontroller/src/yamlparser.cpp`
- Driver core:
  - `cuPHY-CP/cuphydriver/src/common/cuphydriver_api.cpp`
  - `cuPHY-CP/cuphydriver/src/common/context.cpp`
  - `cuPHY-CP/cuphydriver/src/common/cell.cpp`
- Adapter runtime:
  - `cuPHY-CP/cuphyl2adapter/lib/nvPHY/nv_phy_module.cpp`
  - `cuPHY-CP/scfl2adapter/lib/scf_5g_fapi/scf_5g_fapi_phy.cpp`
- Channel and compute:
  - `cuPHY/src/cuphy_channels/*.cpp`
  - `cuPHY/src/cuphy/*.cu`
- Fronthaul:
  - `cuPHY-CP/aerial-fh-driver/*`

## Ownership Matrix (Primary)
## `cuphycontroller`
### Owns
- Application startup orchestration.
- YAML parse + validation and conversion to runtime config objects.
- Initialization wrappers that call driver API (`pc_init_phydriver`).

### Does not own
- Per-slot channel execution logic.
- CUDA kernel invocation details.

## `cuphyl2adapter` + `scfl2adapter`
### Owns
- Transport message ingest.
- FAPI message parse/validate and slot command assembly.
- Tick synchronization, timing gates, and dispatch timing policy.
- Indication and response publication to L2.

### Does not own
- Channel math/PHY algorithms.
- Compression internals.

## `cuphydriver`
### Owns
- L1 API implementation (`l1_init`, `l1_enqueue_phy_work`, lifecycle APIs).
- Worker pools, task graphs, slot-map lifecycle, and execution ordering.
- Channel aggregator orchestration for DL/UL pipelines.
- Compression invocation and fronthaul prep/send integration.

### Does not own
- External MAC scheduling policy.
- Python API design.

## `cuPHY`
### Owns
- Channel algorithm implementations and CUDA kernels.
- Channel setup/run/destroy APIs (for example PDSCH, PUSCH, PRACH).
- Tensor descriptors, channel-specific memory and compute logic.

### Does not own
- FAPI message parsing.
- Slot-level scheduling policy.

## `aerial-fh-driver`
### Owns
- O-RAN packet structures and section formatting.
- User-plane/C-plane packet prep and send pathways.
- FH transport backend interactions.

### Does not own
- Channel-level signal generation/decoding.

## `cuMAC-CP` + `cuMAC`
### Owns
- MAC side message production and scheduler decisions.
- Contract-level message types sent to L1 adapter.

### Does not own
- Driver-level task execution internals.

## `pyaerial`
### Owns
- Python-facing API ergonomics and pybind interfaces.
- Conversion from Python types/config to cuPHY-compatible structs.

### Does not own
- Runtime slot scheduling in production L1 path.

## Detailed Role Profiles
## 1) `cuphycontroller`
Representative responsibilities from code:
- YAML key extraction for worker topology, timing, compression, and per-cell mappings.
- Validation of backend addresses and parser-level invariants.
- Passing parsed values into driver context config and worker descriptors.

Representative anchors:
- `YamlParser::parse_cuphydriver_configs`
- `YamlParser::parse_cell_configs`
- `YamlParser::parse_single_cell`
- `pc_init_phydriver(...)`

## 2) `PHY_module` (adapter runtime core)
Representative responsibilities:
- receive and process control/data messages from transport.
- maintain slot-state readiness and dispatch conditions.
- enforce timing threshold checks before enqueue.
- route slot command to driver proxy.

Representative anchors:
- `PHY_module::msg_processing`
- `PHY_module::process_phy_commands`
- `PHY_module::check_time_threshold`
- `PHYDriverProxy::l1_enqueue_phy_work`

## 3) `cuphydriver` context and worker subsystem
Representative responsibilities:
- instantiate per-cell and per-worker runtime state.
- bind worker threads to cores with scheduling policy.
- execute slot task lists with synchronization and timeout handling.

Representative anchors:
- `Worker::run` and worker loop in `worker.cpp`
- `l1_enqueue_phy_work` in `cuphydriver_api.cpp`
- slot map classes and task functions (`slot_map_dl.cpp`, `task_function_*`)

## 4) Channel aggregators (DL/UL)
Representative responsibilities:
- map slot command parameters into cuPHY dynamic structs.
- call cuPHY setup/run APIs and collect outputs.
- perform callback invocation and channel-specific error handling.

Representative anchors:
- `phypdsch_aggr.cpp`
- `phypusch_aggr.cpp`
- `phyprach_aggr.cpp`

## 5) Fronthaul integration
Representative responsibilities:
- prepare O-RAN section payloads.
- synchronize compression completion and packet send.
- invoke GPU communication send path.

Representative anchors:
- `task_function_dl_aggr.cpp` (`prepareUPlanePackets`, `UserPlaneSendPacketsGpuComm`)
- `aerial-fh-driver` `oran.hpp` and `gpu_comm.cpp`

## Responsibility Boundaries That Commonly Cause Confusion
## Boundary A: Adapter vs driver timing policy
- Adapter decides when a slot command is ready to send.
- Driver decides how tasks execute once command is accepted.

## Boundary B: Driver vs cuPHY error ownership
- Driver handles orchestration and retries/abort decisions.
- cuPHY reports compute/setup API status and channel-level errors.

## Boundary C: Compression ownership
- Scaling constants may originate in driver config/state.
- Actual compression/payload packing logic lives in compression/FH components.

## Boundary D: Python API vs production L1 pipeline
- `pyaerial` is primarily API/bindings and test/dev pipeline tooling.
- production real-time slot orchestration is in CP/driver components.

## RACI-Style Summary
## Config and startup
- Responsible: `cuphycontroller`
- Accountable: `cuphycontroller`
- Consulted: `cuphydriver`
- Informed: adapter/FH components

## Slot dispatch gating
- Responsible: adapter (`PHY_module`, `scf_5g_fapi_phy`)
- Accountable: adapter runtime
- Consulted: driver
- Informed: L2/MAC

## Slot execution and channel orchestration
- Responsible: `cuphydriver`
- Accountable: `cuphydriver`
- Consulted: `cuPHY`
- Informed: adapter callback path

## PHY channel algorithms
- Responsible: `cuPHY`
- Accountable: `cuPHY`
- Consulted: driver
- Informed: adapter/FH through outputs

## Fronthaul packetization and send
- Responsible: `aerial-fh-driver` + driver integration path
- Accountable: driver/FH integration ownership
- Consulted: adapter for slot/cell context
- Informed: L2 via callback and stats

## Risk Register by Ownership Boundary
## RB1: parser and runtime config divergence
Owner boundary: controller <-> driver
Risk: runtime behavior mismatch from incomplete key propagation.

## RB2: slot completeness and dispatch race
Owner boundary: adapter <-> driver
Risk: incomplete or stale slot commands reaching execution layer.

## RB3: channel setup semantics drift
Owner boundary: driver <-> cuPHY
Risk: wrong assumptions about dynamic struct contents and optional flags.

## RB4: compression contract mismatch
Owner boundary: driver <-> compression/FH
Risk: invalid payload size/scale interpretation across RU path.

## RB5: indication mapping drift
Owner boundary: driver/adapter <-> L2
Risk: inconsistent message semantics for CRC/UCI/RACH under edge cases.

## Validation Checklist by Component
1. Controller:
- parser unit checks for key configuration surfaces.

2. Adapter:
- message-order and partial-command handling tests.

3. Driver:
- slot-map lifecycle and timeout-path stress tests.

4. cuPHY:
- channel API setup/run correctness and vector validation.

5. FH:
- packet format and decompression interoperability checks.

6. End-to-end:
- multi-cell mixed DL/UL traffic with indication accounting.

## Refactor Coverage
This report consolidates and normalizes role descriptions from:
- `CUPHYCONTROLLER_ROLES.md`
- `CUPHYDRIVER_ROLES.md`
- role fragments across architecture and threading reports

Legacy snapshots are preserved:
- `archive/aerial-reports-v1.1-2026-02-11/`

## Related Consolidated Reports
- `architecture-overview-codex.md`
- `control-plane-runtime-codex.md`
- `parallelism-threading-model-codex.md`
