# Aerial cuBB Architecture Overview -codex (v1.0)

> Document Version: v1.0-codex
> Last Updated: 2026-02-11
> Source Baseline: cuBB-25-2-Codex workspace
> Purpose: Consolidated, self-contained architecture report replacing fragmented overview docs.

## Executive Summary
This report describes the runtime architecture spanning `cuPHY-CP`, `cuPHY`, `cuMAC-CP`, `cuMAC`, and `pyaerial`. The central architecture pattern is:

1. Control-plane ingestion and validation in `cuPHY-CP` (`cuphycontroller` + `cuphyl2adapter`/`scfl2adapter`).
2. Slot-level command materialization into `slot_command` objects.
3. Driver-side task graph dispatch in `cuphydriver` using per-slot maps and worker pools.
4. CUDA channel aggregators in `cuPHY` (DL and UL) with asynchronous stream orchestration.
5. Fronthaul packaging/compression and callback/indication egress back to MAC/L2.

The codebase is engineered around deterministic slot cadence, strict state transitions, pre-allocation, and multi-level parallelism (message-level, slot-level, channel-level, and kernel-level).

## Scope and Method
This report is based on repository code and in-tree documentation only. It intentionally excludes assumptions from external deployment playbooks.

Primary evidence anchors:
- Driver API/lifecycle: `cuPHY-CP/cuphydriver/src/common/cuphydriver_api.cpp`
- Driver context/cell/resource state: `cuPHY-CP/cuphydriver/src/common/context.cpp`, `cuPHY-CP/cuphydriver/src/common/cell.cpp`
- Controller startup and YAML parsing: `cuPHY-CP/cuphycontroller/src/cuphydriver.cpp`, `cuPHY-CP/cuphycontroller/src/yamlparser.cpp`
- L2 adapter timing and slot gating: `cuPHY-CP/cuphyl2adapter/lib/nvPHY/nv_phy_module.cpp`, `cuPHY-CP/cuphyl2adapter/lib/nvPHY/nv_tick_generator.cpp`
- FAPI message conversion and indications: `cuPHY-CP/scfl2adapter/lib/scf_5g_fapi/scf_5g_fapi_phy.cpp`
- Channel pipelines: `cuPHY/src/cuphy_channels/pdsch_tx.cpp`, `cuPHY/src/cuphy_channels/prach_rx.cpp`, `cuPHY/src/cuphy/cuphy_api.h`
- Python surface: `pyaerial/src/aerial/phy5g/pdsch/pdsch_tx.py`, `pyaerial/pybind11/pycuphy_pdsch_params.cpp`

## Repository Decomposition
### L1 control + orchestration (`cuPHY-CP`)
- `cuphycontroller`: startup, YAML parse/validation, driver init wrappers.
- `cuphyl2adapter`/`scfl2adapter`: ingress from IPC/FAPI, slot assembly, tick synchronization, message and indication handling.
- `cuphydriver`: core task scheduler, worker execution, slot map lifecycle, channel aggregators, fronthaul handoff.
- `compression_decompression` + `aerial-fh-driver`: O-RAN data path (compression, packet prep, transport).

### L1 compute (`cuPHY`)
- Channel entry points and APIs in `cuPHY/src/cuphy_channels/*`.
- Core kernels and compute blocks in `cuPHY/src/cuphy/*` (LDPC, modulation, equalization, DMRS, PRACH, etc.).

### L2/MAC integration (`cuMAC-CP`, `cuMAC`)
- Message/API definitions and request generation in `cuMAC-CP/lib/cumac_msg.h`, `cuMAC-CP/lib/cumac_msg.c`.
- Scheduler and associated compute functions in `cuMAC`.

### Python and notebooks (`pyaerial`)
- High-level API wrappers and orchestration classes.
- pybind layer mapping Python config objects to cuPHY structs.

## End-to-End Runtime Lifecycle
## 1) Boot and config ingestion
- `YamlParser::parse_cuphydriver_configs`, `parse_cell_configs`, `parse_single_cell` in `cuPHY-CP/cuphycontroller/src/yamlparser.cpp`.
- Startup wrapper `pc_init_phydriver` in `cuPHY-CP/cuphycontroller/src/cuphydriver.cpp` calls into driver API.

## 2) Driver initialization
- Driver creation and state setup via `l1_init(...)` and resource init in `cuPHY-CP/cuphydriver/src/common/cuphydriver_api.cpp`.
- Cell registration/start flows and worker launch are coordinated from the same API layer plus `context.cpp`/`worker.cpp`.

## 3) Message ingress and slot command build
- `PHY_module::msg_processing` ingests transport events in `cuPHY-CP/cuphyl2adapter/lib/nvPHY/nv_phy_module.cpp`.
- Message types are normalized to slot commands through `scf_5g_fapi_phy.cpp` and slot-command helpers.
- Timing gate uses `PHY_module::check_time_threshold(...)` and `process_phy_commands(...)`.

## 4) Slot execution dispatch
- L2 adapter calls proxy `PHYDriverProxy::l1_enqueue_phy_work(...)` in `nv_phy_driver_proxy.cpp`.
- Driver entry `l1_enqueue_phy_work(...)` in `cuphydriver_api.cpp` partitions channel work into task graphs and slot maps.

## 5) CUDA channel run + FH path
- DL aggregators (for example `phypdsch_aggr.cpp`) and UL aggregators (for example `phypusch_aggr.cpp`, `phyprach_aggr.cpp`) call cuPHY setup/run APIs.
- Compression and O-RAN payload path use driver common components and fronthaul driver integration.

## 6) Indication/callback egress
- CRC/UCI/RACH/SRS/DL response callbacks are built and emitted in `scf_5g_fapi_phy.cpp`.
- L2 sees standardized FAPI-like indication events.

## Plane Separation and Contracts
## Control-plane responsibilities
- Config correctness and topology checks.
- Slot command assembly and message order handling.
- State machine transitions and timeout policy.

## Data-plane responsibilities
- Compute kernels and channel data transforms.
- Tensor/buffer movement, compression, and fronthaul egress.
- Deterministic completion signaling back to control-plane.

## Key shared contract: `slot_command`
The `slot_command` API is the semantic boundary. It carries DL/UL channel parameters, callbacks, slot metadata, and pointers required by driver and channel aggregators.

Evidence: `cuPHY-CP/gt_common_libs/slot_command/include/slot_command/slot_command.hpp`.

## Concurrency and Parallelism Model
## Thread domains
- Message processing thread (`msg_processing`) in `nv_phy_module.cpp`.
- Tick generator thread and mode-specific event loop in `nv_tick_generator.cpp`.
- Driver worker pools (UL/DL plus optional specialized workers) in `cuphydriver`.

## Scheduling layers
1. Message-level concurrency at ingress/IPC level.
2. Slot-level parallelism across in-flight slot maps.
3. Channel-level parallelism across DL/UL aggregators.
4. Kernel-level parallelism within cuPHY CUDA kernels.

## Synchronization patterns
- Atomic state fields and bounded queues.
- Slot map counters and channel completion bitmaps.
- CUDA events for inter-kernel and inter-stream ordering.
- epoll-driven I/O loops (`nv_phy_epoll_context.cpp`).

## Memory and Buffering Architecture
- Heavy use of pre-allocation for slot objects, channel descriptors, and tensor buffers.
- Pinned host buffers for transfer-sensitive paths.
- Device buffers retained across slots where possible to reduce alloc/free churn.
- Circular slot maps in UL and DL task stacks.

Evidence anchors:
- `cuPHY-CP/cuphydriver/src/downlink/slot_map_dl.cpp`
- `cuPHY-CP/cuphydriver/src/uplink/task_function_ul_aggr.cpp`
- `cuPHY/src/cuphy_channels/pdsch_tx.cpp` (data-in/data-out and fallback buffer handling)

## Timing Model and Guardrails
- Slot cadence is driven by tick-generation modes (`poll+sleep`, `sleep`, `timer_fd`) in `nv_tick_generator.cpp`.
- `process_phy_commands(...)` enforces slot readiness and drop/latency guards.
- Reorder/partial-command behavior is explicit in FAPI processing code (`scf_5g_fapi_phy.cpp`).

## Configuration Surfaces That Change Behavior
Most architecture changes in runtime behavior are config-driven, not code-path changes:
- Worker affinity and scheduling priority (`workers_ul`, `workers_dl`, `workers_sched_priority`).
- Tick mode and timing window.
- Compression bit-width and `fix_beta_dl`.
- Optional helper threads (`h2d_copy_thread`, pcap threads, prometheus thread).

Evidence: `cuPHY-CP/cuphycontroller/include/yamlparser.hpp`, `cuPHY-CP/cuphycontroller/src/yamlparser.cpp`.

## Integration with cuMAC and pyAerial
## cuMAC/cuMAC-CP
- Message IDs and request taxonomy align with L1 adapter path (`CONFIG`, `DL_TTI`, `UL_TTI`, `TX_DATA`).
- Runtime exchange is primarily contract-based, not direct object coupling.

Evidence:
- `cuMAC-CP/lib/cumac_msg.h`
- `cuMAC-CP/lib/cumac_msg.c`

## pyAerial
- Python-side objects are converted to cuPHY structures via pybind bridges.
- PDSCH, DMRS, and precoding flows are implemented through `pycuphy` wrappers and call into channel pipelines.

Evidence:
- `pyaerial/src/aerial/phy5g/pdsch/pdsch_tx.py`
- `pyaerial/pybind11/pycuphy_pdsch_params.cpp`
- `pyaerial/pybind11/pycuphy_pybind.cpp`

## Architectural Risks and Failure Modes
## 1) Slot command completeness race
If DL/UL/TX_DATA arrive out-of-order or partially for a slot, incorrect command issuance can cause drops or stale command release.

## 2) Timing threshold sensitivity
Overly strict or mis-tuned thresholds can convert recoverable latency into dropped work and recurrent error indications.

## 3) Cross-thread coordination bugs
UL/DL and helper-thread interactions rely on counters, atomics, and event sequencing; any contract mismatch can cause deadlocks or timeout cascades.

## 4) Compression/scaling drift
Misalignment between configured bit-width/scaling and expected RU decode behavior can produce silent quality loss.

## 5) Config drift across environments
Many paths are runtime-configured. Undocumented configuration differences can appear as "random" behavior differences.

## Validation Checklist
1. Boot sequence:
- Verify YAML parse and `pc_init_phydriver` complete with expected worker topology.
2. Steady-state slot path:
- Confirm `process_phy_commands` emits expected slot command cadence.
- Confirm `l1_enqueue_phy_work` success rate and latency histograms.
3. Channel execution:
- Validate DL/UL completion counters and callback completeness.
4. Timing robustness:
- Inject controlled latency to validate threshold behavior and recovery.
5. Compression/FH:
- Cross-check payload size/format and round-trip decode quality.
6. Indications:
- Verify CRC/UCI/RACH/SRS counts versus known traffic model.

## Refactor Notes for Report Consumers
This report supersedes and consolidates architecture material previously split across:
- `architecture_analysis.md`
- `MAC_ARCHITECTURE.md`
- `MULTITHREAD_PARALLEL_ARCHITECTURE-claude.md`
- `multi_task_parallel_architecture.md`
- `multi_task_parallel_architecture-v0.md`

Legacy reports are preserved in archive:
- `archive/aerial-reports-v1.1-2026-02-11/`

## Related Consolidated Reports
- `control-plane-runtime-codex.md`
- `parallelism-threading-model-codex.md`
- `downlink-pdsch-pipeline-codex.md`
- `uplink-pusch-prach-pipeline-codex.md`
- `roles-responsibilities-codex.md`
- `migration-checklist-25-3-codex.md`
