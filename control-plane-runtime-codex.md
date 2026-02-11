# Aerial cuBB Control-Plane Runtime Deep Dive -codex (v1.0)

> Document Version: v1.0-codex
> Last Updated: 2026-02-11
> Scope: `cuphycontroller`, `cuphyl2adapter`, `scfl2adapter`, and driver control interfaces.

## Executive Summary
Control-plane runtime in Aerial is built around a strict pipeline:

1. Parse and validate deployment YAML into driver/runtime config structs.
2. Initialize driver context and cells.
3. Ingest FAPI-like messages into per-slot command state.
4. Gate dispatch by timing and completeness rules.
5. Enqueue slot commands to L1 driver.
6. Publish results and indications back to MAC/L2.

The largest operational risks are configuration drift, slot completeness races (DL_TTI/UL_TTI/TX_DATA ordering), and threshold-based late-slot handling.

## Source Basis and Core Entry Points
## Startup/initialization
- `cuPHY-CP/cuphycontroller/examples/cuphycontroller_scf.cpp`
- `cuPHY-CP/cuphycontroller/src/cuphydriver.cpp` (`pc_init_phydriver`)
- `cuPHY-CP/cuphydriver/src/common/cuphydriver_api.cpp` (`l1_init`, `l1_finalize`, `l1_enqueue_phy_work`)

## YAML parser and config validation
- `cuPHY-CP/cuphycontroller/include/yamlparser.hpp`
- `cuPHY-CP/cuphycontroller/src/yamlparser.cpp`

## L2 adapter and slot dispatch
- `cuPHY-CP/cuphyl2adapter/lib/nvPHY/nv_phy_module.cpp`
- `cuPHY-CP/cuphyl2adapter/lib/nvPHY/nv_phy_module.hpp`
- `cuPHY-CP/cuphyl2adapter/lib/nvPHY/nv_phy_driver_proxy.cpp`
- `cuPHY-CP/scfl2adapter/lib/scf_5g_fapi/scf_5g_fapi_phy.cpp`

## Tick and event loop infrastructure
- `cuPHY-CP/cuphyl2adapter/lib/nvPHY/nv_tick_generator.cpp`
- `cuPHY-CP/cuphyl2adapter/lib/nvPHY/nv_phy_epoll_context.cpp`

## Startup Sequence (Detailed)
## Step 1: YAML parse and config materialization
`YamlParser` loads runtime config and transforms YAML fields into strongly typed config objects consumed by controller and driver.

High-impact parser functions:
- `parse_cuphydriver_configs(...)`
- `parse_cell_configs(...)`
- `parse_single_cell(...)`
- `parse_eAxC_to_beam_map(...)`

Evidence: `cuPHY-CP/cuphycontroller/src/yamlparser.cpp`.

## Step 2: Driver context initialization
`pc_init_phydriver(...)` in `cuphycontroller/src/cuphydriver.cpp` calls the driver API to:
- create driver context
- register cells
- start worker pools and support threads
- make L1 API handle available to adapter path

## Step 3: Adapter/transport initialization
`PHY_module` starts message processing and registers transport fds into epoll context.

Key internals:
- `PHY_module::msg_processing()`
- `phy_epoll_context::start_event_loop()`
- thread naming/affinity/scheduling setup in `nv_phy_module.cpp`

## Step 4: Tick generator activation
Tick modes are selected from config (`tick_generator_mode`):
- mode 0: poll + sleep
- mode 1: sleep
- mode 2: timer_fd + epoll

Evidence: `nv_tick_generator.cpp`.

## Runtime Message and Slot Pipeline
## Message ingestion
`scf_5g_fapi_phy.cpp` parses/validates incoming messages:
- `SCF_FAPI_CONFIG_REQUEST`
- `SCF_FAPI_DL_TTI_REQUEST`
- `SCF_FAPI_UL_TTI_REQUEST`
- `SCF_FAPI_TX_DATA_REQUEST`

Message IDs and body contracts are defined in `scf_5g_fapi.h`.

## Slot assembly
The adapter builds or updates per-slot command state using DL_TTI/UL_TTI/TX_DATA fragments and enforces consistency checks (for example PDU index and RNTI matching behavior).

## Dispatch gate
`PHY_module::process_phy_commands(bool slot_end_rcvd)` checks readiness and timing. Dispatch occurs only when slot command conditions are satisfied.

Core gate condition components:
- time threshold (`check_time_threshold(...)`)
- number of assembled per-cell commands
- partial command flags

## Driver enqueue
On successful gate checks, adapter invokes:
- `PHYDriverProxy::l1_enqueue_phy_work(...)`
- driver `l1_enqueue_phy_work(...)`

Failure path logs and cleanup actions are explicitly implemented in both module and FAPI layers.

## YAML Configuration Surfaces and Their Architectural Impact
## Thread topology and scheduling
Relevant keys in parser header (`yamlparser.hpp`):
- `workers_ul`, `workers_dl`, `workers_dl_validation`
- `workers_sched_priority`
- `prometheus_thread`, `dpdk_thread`, `pdump_client_thread`
- `h2d_copy_thread_*`
- `ul_pcap_capture_thread_*`, `pcap_logger_thread_*`

Impact:
- Changes CPU topology, worker parallelism, contention behavior, and runtime determinism.

## Radio and per-cell channel mapping
`parse_single_cell` and related helpers process:
- cell-level IDs and numerology-aligned parameters
- eAxC-to-channel mappings for DL/UL channels
- beam/channel map association

Impact:
- Misconfiguration here propagates directly to fronthaul mapping correctness.

## Compression/scaling behavior
`fix_beta_dl` is parsed in controller and consumed downstream by driver scaling logic.

Evidence:
- parser: `yamlparser.cpp`
- driver use: `cuPHY-CP/cuphydriver/src/common/cell.cpp`

## Time, State, and Fault Management
## State machine boundaries
`scf_5g_fapi_phy.cpp` enforces state checks before accepting requests (for example handling CONFIG while RUNNING/CONFIGURED).

## Threshold and drop logic
`check_time_threshold` logic prevents dispatching commands considered stale.

## Duplicate and mismatch handling
FAPI handlers explicitly log and guard against duplicate DL_TTI/UL_TTI/TX_DATA and mismatch conditions.

## Indications and response paths
Control-plane composes and sends:
- `CONFIG.response`
- `DL_TTI.response`
- `CRC.indication`
- `UCI.indication`
- `RACH.indication`
- `SRS.indication`

## Interface Contracts
## FAPI contracts
Core definitions and PDU enums:
- `cuPHY-CP/scfl2adapter/lib/scf_5g_fapi/scf_5g_fapi.h`

## Driver API contracts
- `l1_init`, `l1_enqueue_phy_work`, cell lifecycle APIs in `cuphydriver_api.*`
- `PHYDriverProxy` bridging in `nv_phy_driver_proxy.*`

## Slot command contracts
- `slot_command` structures and callbacks in `gt_common_libs/slot_command/include/slot_command/slot_command.hpp`

## Control-Plane Performance Considerations
## Hot paths
- `PHY_module::msg_processing`
- `process_phy_commands`
- FAPI validation and slot command assembly in `scf_5g_fapi_phy.cpp`

## Dominant overhead contributors
- Message parsing + validation for heavy multi-cell/multi-UE slots
- Synchronization around slot completeness and timeout checks
- Thread contention or suboptimal affinity under high traffic density

## Tuning levers
1. Worker/core topology from YAML.
2. Tick mode selection (`tick_generator_mode`).
3. Threshold tuning for late-slot behavior.
4. Optional helper-thread enablement (H2D copy, PCAP capture, metrics).

## Risk Register
## R1: Partial slot command release mismatch
Symptoms:
- stale TX_DATA records
- repeated "not released" logs
- duplicate message handling under burst conditions

Evidence anchors:
- `scf_5g_fapi_phy.cpp` duplicate and release checks

## R2: Overly strict time threshold
Symptoms:
- sudden increase in dropped slot commands
- frequent timing-related error indications

Evidence anchors:
- `nv_phy_module.cpp` `check_time_threshold` and dispatch guard

## R3: Config/affinity drift across deployments
Symptoms:
- unstable latency and throughput
- hard-to-reproduce regressions between environments

Evidence anchors:
- `yamlparser.hpp/.cpp` extensive thread and core-affinity parameters

## R4: Invalid eAxC/channel map assumptions
Symptoms:
- wrong channel-to-transport mapping
- fronthaul payload corruption or misrouting

Evidence anchors:
- `parse_eAxC_to_beam_map` in `yamlparser.cpp`

## Control-Plane Validation Matrix
1. Boot validation:
- parse YAML and assert expected values for workers, threads, `fix_beta_dl`.
- ensure `pc_init_phydriver` succeeds and workers are running.

2. Message-order validation:
- test nominal order and out-of-order arrival for DL_TTI/UL_TTI/TX_DATA.
- verify no leaked/duplicate per-slot state.

3. Timing validation:
- apply controlled delay and verify threshold behavior.
- track resulting drop/error rates.

4. Multi-cell stress:
- high-cell-count mixed DL/UL loads.
- validate per-cell indication completeness.

5. Recovery behavior:
- inject malformed PDUs.
- verify robust error indication without process instability.

## Refactor Coverage
This report consolidates control-plane content formerly spread across:
- `CUPHYCONTROLLER_ANALYSIS.md`
- `CUPHYCONTROLLER_ROLES.md`
- control-plane sections of `architecture_analysis.md`
- parts of `UPLINK_PUSCH_PROCESSING_ARCHITECTURE.md` and `PDSCH_PIPELINE.md`

Legacy snapshots are preserved at:
- `archive/aerial-reports-v1.1-2026-02-11/`

## Related Consolidated Reports
- `architecture-overview-codex.md`
- `roles-responsibilities-codex.md`
- `parallelism-threading-model-codex.md`
- `downlink-pdsch-pipeline-codex.md`
- `uplink-pusch-prach-pipeline-codex.md`
