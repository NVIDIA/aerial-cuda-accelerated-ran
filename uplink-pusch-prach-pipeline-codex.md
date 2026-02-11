# Aerial Uplink PUSCH/PRACH Pipeline Deep Analysis -codex (v1.0)

> Document Version: v1.0-codex
> Last Updated: 2026-02-11
> Scope: Uplink PUSCH and PRACH processing from UL_TTI ingress to UL indications.

## Executive Summary
Uplink runtime uses a multi-stage slot pipeline driven by UL slot commands and synchronized task phases. PUSCH and PRACH each have dedicated aggregator objects and task functions, but share the same slot-map orchestration and callback framework.

High-level chain:
1. UL_TTI ingestion and validation.
2. Slot-command creation with channel-specific dynamic parameters.
3. UL task execution (`task_function_ul_aggr.cpp`) and channel setup/run.
4. cuPHY channel pipelines (`PuschRx`, `PrachRx`).
5. CRC/UCI/RACH indication generation back to MAC/L2.

## Source Basis
Primary files used:
- UL ingest and FAPI handling:
  - `cuPHY-CP/scfl2adapter/lib/scf_5g_fapi/scf_5g_fapi_phy.cpp`
  - `cuPHY-CP/scfl2adapter/lib/scf_5g_fapi/scf_5g_fapi_ul_validate.cpp`
- UL slot task orchestration:
  - `cuPHY-CP/cuphydriver/src/uplink/task_function_ul_aggr.cpp`
- PUSCH aggregator:
  - `cuPHY-CP/cuphydriver/src/uplink/phypusch_aggr.cpp`
- PRACH aggregator:
  - `cuPHY-CP/cuphydriver/src/uplink/phyprach_aggr.cpp` (referenced by task code and UL report set)
- cuPHY channel APIs:
  - `cuPHY/src/cuphy/cuphy_api.h`
  - `cuPHY/src/cuphy_channels/prach_rx.cpp`
  - `cuPHY/src/cuphy_channels/prach_rx.hpp`

## UL Ingress and Slot Command Formation
## 1) UL_TTI request handling
`scf_5g_fapi_phy.cpp` parses UL_TTI PDUs and dispatches by PDU type:
- `UL_TTI_PDU_TYPE_PUSCH`
- `UL_TTI_PDU_TYPE_PRACH`
- `UL_TTI_PDU_TYPE_PUCCH`
- `UL_TTI_PDU_TYPE_SRS`

Enum contract is defined in `scf_5g_fapi.h`.

## 2) Validation phase
`scf_5g_fapi_ul_validate.cpp` enforces bounds and format checks across UL PDU variants.

## 3) Slot command population
UL channel-specific parameters are written into slot-command structures and consumed by driver aggregators.

Evidence:
- `slot_command` UL parameter structures in `gt_common_libs/slot_command/include/slot_command/slot_command.hpp`

## UL Driver Task Graph and Scheduling
`task_function_ul_aggr.cpp` implements per-channel task workers, including:
- `task_work_function_ul_aggr_1_pucch_pusch`
- `task_work_function_ul_aggr_1_pusch`
- `task_work_function_ul_aggr_1_prach`
- `task_work_function_ul_aggr_1_srs`

Each worker follows this pattern:
1. wait for prerequisite ULC tasks
2. setup channel objects
3. optional waits on order/completion events
4. run channel kernels/phases
5. capture timing and invoke callback path

The same file contains timeout/abort handling and channel-end/slot-end bookkeeping.

## PUSCH Processing Deep Dive
## Aggregator creation and setup
`phypusch_aggr.cpp` creates and initializes PUSCH pipeline state, including static and dynamic parameter vectors.

Key API calls:
- `cuphyCreatePuschRx(...)`
- `cuphySetupPuschRx(...)`
- `cuphyRunPuschRx(...)`
- `cuphyDestroyPuschRx(...)`

Evidence anchors:
- `phypusch_aggr.cpp` setup/run regions and API invocations
- API definitions in `cuphy_api.h`

## Run phases and early HARQ behavior
UL task logic chooses run phase modes based on slot conditions and configuration:
- `PUSCH_RUN_SUB_SLOT_PROC`
- `PUSCH_RUN_FULL_SLOT_PROC`
- `PUSCH_RUN_ALL_PHASES`
- `PUSCH_RUN_FULL_SLOT_COPY`

The task layer tracks early HARQ/front-loaded DMRS flags and adjusts sequencing accordingly.

## Result handling and callbacks
PUSCH aggregator validates outputs, composes callback payloads, and triggers:
- CRC and RX data indication flows
- optional UCI-on-PUSCH extraction

Additional operational hooks:
- H5 debug dump and data-lake notification hooks
- timeout diagnostics and cell timeout lists

## PRACH Processing Deep Dive
## Aggregator path
UL task code invokes PRACH setup/run through `task_work_function_ul_aggr_1_prach` with slot-map synchronization similar to other UL channels.

## cuPHY PRACH channel APIs
In `cuPHY/src/cuphy_channels/prach_rx.cpp`:
- `cuphyCreatePrachRx(...)`
- `cuphySetupPrachRx(...)`
- `cuphyRunPrachRx(...)`
- `cuphyDestroyPrachRx(...)`

The PRACH implementation includes:
- static parameter derivation from config index / FR / duplex / root sequence / zero-corr config
- occasion-wise dynamic parameter expansion
- graph-based execution option (`PRACH_PROC_MODE_WITH_GRAPH`)
- receiver workspace sizing and FFT plan allocation

## Detection and output path
PRACH results are propagated to control-plane callback logic, which emits `RACH.indication` message structures on completion.

## Shared UL Synchronization Model
UL channel workers coordinate through:
- slot-map task counters and wait functions
- channel-start and order-launch barriers
- CUDA events for inter-stream ordering
- timeout/abort safety paths

Evidence anchors:
- `task_function_ul_aggr.cpp` wait and abort paths
- `nv_phy_module.cpp` and `scf_5g_fapi_phy.cpp` for upstream gating and dispatch cadence

## Data Structures and Interface Contracts
## UL slot-command structures
- PUSCH/PRACH params are represented in `slot_command.hpp` under channel-specific structs.

## cuPHY API contracts
- `cuphyPusch*` and `cuphyPrach*` families in `cuphy_api.h` define setup/run/output buffers.

## FAPI indication contracts
- CRC/UCI/RACH indication message structures in `scf_5g_fapi.h`.

## Performance and Scalability Considerations
## CPU-side
- UL_TTI validation and command assembly cost scales with number of UL PDUs and UEs.
- UL worker synchronization overhead increases with mixed channel compositions in dense slots.

## GPU-side
- PUSCH channel-equation/soft-demap kernels are heavy and sensitive to antenna/layer dimensions.
- PRACH workload scales by number of active occasions, preamble format, and graph mode.

## Pipeline overlap
- Parallel channel execution is possible but constrained by explicit waits and order-entity dependencies.
- Poorly tuned sequencing can convert intended overlap into serial execution.

## Failure Modes
## FM1: ULC prerequisite timeout
UL workers may time out while waiting for prerequisite tasks, causing channel aborts for slot.

Evidence:
- timeout warning paths in `task_function_ul_aggr.cpp`

## FM2: setup/run API errors
`cuphySetupPuschRx`/`cuphyRunPuschRx`/`cuphySetupPrachRx`/`cuphyRunPrachRx` failures propagate to channel error handling and indication paths.

## FM3: Early HARQ path instability
Incorrect flag handling or sequencing in early HARQ/front-loaded DMRS modes can trigger timing errors or invalid output assumptions.

## FM4: Slot-map coordination faults
Incorrect counter/barrier behavior can produce deadlocks, premature abort, or missing callbacks.

## FM5: PRACH occasion config mismatch
Invalid PRACH configuration index/root sequence/zcz combination can produce detection failures despite healthy transport.

## Validation Checklist
1. UL message validity:
- fuzz UL_TTI PDU fields and verify validator catches malformed combinations.

2. PUSCH phase coverage:
- run with and without early HARQ/front-loaded DMRS and compare callback completeness.

3. PRACH coverage:
- sweep PRACH configuration index and root sequence settings across supported FR/duplex modes.

4. Timeout robustness:
- inject deterministic delays before ULC completion and verify expected timeout/error behavior.

5. Multi-cell load:
- mixed PUSCH/PRACH/PUCCH/SRS traffic with per-cell indication accounting.

6. Callback correctness:
- verify CRC/UCI/RACH indication content and ordering against expected slot timeline.

## Refactor Coverage
This report consolidates and expands content formerly in:
- `UPLINK_PUSCH_PROCESSING_ARCHITECTURE.md`
- `PRACH_PROCESSING_CHAIN.md`
- `UPLINK_MULTITHREAD_ARCHITECTURE.md` (UL-specific sections)

Legacy snapshots are preserved:
- `archive/aerial-reports-v1.1-2026-02-11/`

## Related Consolidated Reports
- `architecture-overview-codex.md`
- `control-plane-runtime-codex.md`
- `parallelism-threading-model-codex.md`
- `roles-responsibilities-codex.md`
