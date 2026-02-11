# Aerial Downlink PDSCH Pipeline Deep Analysis -codex (v1.0)

> Document Version: v1.0-codex
> Last Updated: 2026-02-11
> Scope: PDSCH data path from L2 request ingestion to O-RAN U-plane packet send.

## Executive Summary
The PDSCH downlink path is a staged pipeline with strict separation of responsibilities:

1. FAPI/L2 ingest and slot command assembly (`scfl2adapter`/`cuphyl2adapter`).
2. Driver aggregator setup/run (`PhyPdschAggr`) and DL task orchestration.
3. cuPHY PDSCH TX compute (`cuphySetupPdschTx`, `cuphyRunPdschTx`).
4. Compression and fronthaul packet preparation.
5. GPU-comm-based packet transmission.

Critical correctness domains:
- UE-level dynamic params (including `beta_qam`, TB pointers, symbol masks)
- optional precoding matrix handling
- compression scaling (`beta_dl`) and bit-width consistency
- strict slot and event sequencing around compression and U-plane send

## Source Basis
Core files used for this report:
- Slot/FAPI ingress and command build:
  - `cuPHY-CP/scfl2adapter/lib/scf_5g_fapi/scf_5g_fapi_phy.cpp`
  - `cuPHY-CP/cuphyl2adapter/lib/nvPHY/nv_phy_module.cpp`
- Driver DL execution:
  - `cuPHY-CP/cuphydriver/src/downlink/task_function_dl_aggr.cpp`
  - `cuPHY-CP/cuphydriver/src/downlink/phypdsch_aggr.cpp`
- cuPHY PDSCH compute:
  - `cuPHY/src/cuphy_channels/pdsch_tx.cpp`
  - `cuPHY/src/cuphy/modulation_mapper/modulation_mapper.cu`
  - `cuPHY/src/cuphy/pdsch_dmrs/pdsch_dmrs.cu`
- Compression and payload packing:
  - `cuPHY-CP/cuphydriver/src/common/comp_kernel.cuh`
  - `cuPHY-CP/compression_decompression/comp_decomp_lib/include/gpu_blockFP.h`
  - `cuPHY-CP/compression_decompression/comp_decomp_lib/include/gpu_packing.h`
  - `cuPHY-CP/aerial-fh-driver/include/aerial-fh-driver/oran.hpp`
- Python-to-cuPHY mapping:
  - `pyaerial/src/aerial/phy5g/pdsch/pdsch_tx.py`
  - `pyaerial/pybind11/pycuphy_pdsch_params.cpp`

## End-to-End Processing Chain
## 1) DL request ingest and slot command completion
`scf_5g_fapi_phy.cpp` receives and validates DL_TTI + TX_DATA and merges them into slot command state.

Key behaviors:
- Duplicate and mismatch checks for DL_TTI/TX_DATA combinations.
- PDU matching, including TB pointer/offset handling paths.
- Slot dispatch only when command completeness and timing checks pass.

## 2) Driver enqueue and DL task graph launch
Adapter calls `l1_enqueue_phy_work(...)`, then DL tasks in `task_function_dl_aggr.cpp` run channel setup/run, compression, and send phases.

Important staging points in `task_function_dl_aggr.cpp`:
- channel tasks (including PDSCH) setup and run
- compression phase (`task_work_function_dl_aggr_1_compression`)
- FH packet prep (`prepareUPlanePackets`)
- packet send (`UserPlaneSendPacketsGpuComm`)

## 3) PDSCH aggregator setup/run
`PhyPdschAggr::setup(...)` prepares per-cell/per-UE dynamic descriptors and calls:
- `cuphySetupPdschTx(...)`

`PhyPdschAggr::run()` triggers execution:
- `cuphyRunPdschTx(...)`

Evidence:
- `cuPHY-CP/cuphydriver/src/downlink/phypdsch_aggr.cpp`

## 4) cuPHY PDSCH compute internals
Entry points in `cuPHY/src/cuphy_channels/pdsch_tx.cpp`:
- `cuphySetupPdschTx` (line region around 546)
- `cuphyRunPdschTx` (line region around 659)

Pipeline internals include:
- TB/CB handling and rate matching orchestration
- DMRS insertion and modulation
- optional precoding path specialization

Precoding-specific behavior evidence:
- `nPrecodingMatrices` gates logic in `pdsch_tx.cpp`
- explicit message notes reference-check bypass when precoding exists:
  - "No comparison because at least one UE has precoding enabled."

## 5) Modulation and `beta_qam`
`modulation_mapper.cu` scales constellation values by `beta_qam`.

Evidence anchors:
- `rev_qam_256` table and multiplication by `params[...].beta_qam`
- QPSK path uses `0.707106... * beta_qam`
- symbols written into modulation output tensor

DMRS path also carries `beta_qam`:
- `pdsch_dmrs.cu` assigns `h_dmrs_params[TB_id].beta_qam = ue->beta_qam`

Python/pybind mapping of `beta_qam`:
- `pyaerial/pybind11/pycuphy_pdsch_params.cpp` sets `m_cellGrpDynPrms.pUePrms[ueIdx].beta_qam`

## 6) Precoding path (Python -> pybind -> cuPHY)
## Python API level
`pyaerial/src/aerial/phy5g/pdsch/pdsch_tx.py` accepts `precoding_matrices` and forms dynamic params.

## pybind translation
`pycuphy_pdsch_params.cpp`:
- reads matrix list
- sets `nPrecodingMatrices`
- writes matrix coefficients into cuPHY-compatible storage

## cuPHY usage
`pdsch_tx.cpp` checks `dynamic_params->pCellGrpDynPrm->nPrecodingMatrices` and selects precoding-specialized behavior.

## 7) Compression path (post-channel)
Compression is orchestrated in DL task functions and executed with GPU kernels.

### Driver kernel wrapper
`comp_kernel.cuh` reads per-cell scaling `beta` and calls `scale_compress_blockFP(...)`.

### Block floating-point algorithm
`gpu_blockFP.h`:
- scales IQ values
- determines per-PRB `shift`
- calls `packOutput<3>(...)`

### 9-bit packing
`gpu_packing.h` contains specialized packing path and `warpWrite<9,...>` behavior.

### O-RAN payload sizing
`oran.hpp` selects PRB size by compression width:
- 9-bit path -> `PRB_SIZE_9F`

## 8) `beta_dl` scaling model and runtime effect
Driver computes DL scaling in `cell.cpp`.

Evidence anchors:
- `sqrt_fs0 = 2^(dl_bit_width-1) * 2^(2^exponent_dl - 1)`
- `fs = sqrt_fs0^2 * 2^(-fs_offset_dl)`
- `beta_dl = sqrt((fs * 10^(ref_dl/10)) / (12 * nPrbDlBwp))`
- optional forced constants when `fixBetaDl()` is set:
  - 9-bit -> `65536`
  - 14-bit -> `2097152`

Implication:
`beta_qam` and `beta_dl` control different stages:
- `beta_qam`: symbol-domain scaling in modulation stage
- `beta_dl`: compression-domain scaling before BFP quantization

## Synchronization and Ordering Guarantees
DL orchestration uses explicit events and waiting points:
- wait for all channels done before compression
- run compression
- wait for compression stop event before packet send on GPU comm stream

Evidence anchors:
- `task_function_dl_aggr.cpp` (`waitCompressionStart`, `waitCompressionStop`, `getCompressionStopEvt`)
- `aerial-fh-driver/lib/gpu_comm.cpp` waits on `compression_stop_evt`

This design prevents U-plane send from racing ahead of compression completion.

## Data Contracts and Structures
## At control/slot boundary
- `slot_command` carries PDSCH params and callbacks.

## At cuPHY boundary
- `cuphyPdschDynPrms_t` and associated cell/UE group params.

## At FH boundary
- PRB info structures and compression metadata consumed by `prepareUPlanePackets`/`UserPlaneSendPacketsGpuComm`.

## Performance Characteristics and Bottlenecks
## CPU-side
- FAPI decode/validate and DL slot assembly overhead grows with UE and PDU density.
- DL task graph synchronization costs increase with cell count and mixed-channel slots.

## GPU-side
- PDSCH compute path is bandwidth and occupancy sensitive.
- Compression and packet preparation can become bottlenecks if stream/event overlap is mis-tuned.

## Cross-domain
- The channel-to-compression gap and compression execution duration are explicitly tracked in slot timings (`task_function_dl_aggr.cpp`).

## Failure Modes and Observability
## FM1: setup/run API failure in PDSCH
`PhyPdschAggr` logs setup/run errors and short-circuits execution on failure.

## FM2: DL/TX_DATA mismatch
L2 adapter reports mismatches and may release stale data requests.

## FM3: compression/output pointer mismatch
`task_function_dl_aggr.cpp` has defensive checks for null compression buffer scenarios.

## FM4: packet send failure
`UserPlaneSendPacketsGpuComm` fatal return path is explicitly logged.

## FM5: scaling/bitwidth mismatch
Incorrect `fix_beta_dl`, `dl_bit_width`, or RU decode assumptions can silently degrade EVM even when pipeline stays "healthy".

## Validation Checklist
1. Functional correctness:
- Compare generated grid/tensor outputs for non-precoded path.
- Validate expected behavior for precoded path where reference checks are intentionally constrained.

2. Scaling correctness:
- Sweep `beta_qam`, `dl_bit_width`, `fix_beta_dl`, `fs_offset_dl`, `exponent_dl` and verify expected IQ magnitude and BFP exponent trends.

3. Compression/FH correctness:
- Verify PRB payload size matches selected bit-width (`PRB_SIZE_9F` for 9-bit).
- Validate decompressed IQ against expected tolerances.

4. Timing and overlap:
- Measure channel->compression gap and compression durations.
- Ensure no send-before-compression events appear.

5. Error-path validation:
- Inject malformed DL_TTI/TX_DATA combinations.
- Validate robust logs and no stale state leakage across slots.

## Refactor Coverage
This report consolidates and expands content from:
- `PDSCH_PIPELINE.md`
- `pdsch_processing_report.md`
- `pdsch_precoding_analysis.md`
- `pdsch_iq_bfp_report.md`
- relevant PDSCH sections in `UPLINK_PUSCH_PROCESSING_ARCHITECTURE.md` and architecture reports

Legacy snapshots are preserved:
- `archive/aerial-reports-v1.1-2026-02-11/`

## Related Consolidated Reports
- `architecture-overview-codex.md`
- `control-plane-runtime-codex.md`
- `parallelism-threading-model-codex.md`
- `uplink-pusch-prach-pipeline-codex.md`
