# Aerial 25.2 -> 25.3 Migration Playbook -codex (v1.0)

> Document Version: v1.0-codex
> Last Updated: 2026-02-11
> Scope: Practical migration framework for reconciling this workspace with 25.3 baseline while preserving local intent.

## Executive Summary
Migration from local 25.2-derived state to 25.3 is high risk because drift is concentrated in `cuPHY-CP` and `cuPHY`, where control-plane timing, compression, and channel APIs intersect.

This playbook provides:
- deterministic migration sequence,
- rollback-safe workflow,
- verification gates by subsystem,
- explicit ownership for each migration stage.

## Baseline Snapshot
Workspace root:
- `/Users/yschae87/Library/Mobile Documents/com~apple~CloudDocs/Nvidia/Aerial/cuBB-25-2-Codex`

Local marker:
- `aerial-sdk-version` indicates local lineage at `25-2-cubb`.

Reference baseline staged in repo:
- `/Users/yschae87/Library/Mobile Documents/com~apple~CloudDocs/Nvidia/Aerial/cuBB-25-2-Codex/upgrade-25-3-workspace`

Legacy migration report archived source:
- `UPGRADE_25_3_MIGRATION_CHECKLIST.md`

## Migration Objectives
1. Align runtime-critical paths with 25.3 behavior.
2. Preserve intentional local customizations.
3. Eliminate accidental drift and unknown differences.
4. Keep each merge step buildable and testable.

## Migration Constraints
- Avoid large all-at-once merges.
- Never mix architecture changes and formatting-only changes in one commit.
- Keep rollback path simple (small commit batches).
- Treat timing-sensitive and compression-sensitive code as high-risk subsystems.

## Drift Hotspots and Priority Order
## Track 1: `cuPHY-CP` (highest risk)
Why first:
- owns orchestration, slot dispatch, FH send, compression coupling, and timeout policy.

High-impact subareas:
- `cuphydriver/src/common/*`
- `cuphydriver/src/downlink/*`
- `cuphydriver/src/uplink/*`
- `cuphyl2adapter/lib/nvPHY/*`
- `scfl2adapter/lib/scf_5g_fapi/*`
- `aerial-fh-driver/*`

## Track 2: `cuPHY`
Why second:
- channel kernels and data path correctness depend on caller contract from Track 1.

High-impact subareas:
- `cuphy_channels/pdsch_tx.cpp`, `prach_rx.cpp`
- `cuphy/pdsch_dmrs/*`, `modulation_mapper/*`, `dl_rate_matching/*`
- `cuphy/cuphy_api.h` ABI-sensitive structs and enums

## Track 3: `cuMAC`, `cuMAC-CP`, `pyaerial`
Why third:
- wrapper/API layers should adapt after L1 contract stabilizes.

High-impact subareas:
- message contract definitions in `cuMAC-CP/lib/*`
- pybind parameter mapping in `pyaerial/pybind11/*`
- high-level pipeline wrappers in `pyaerial/src/aerial/phy5g/*`

## Workstream Model
## Phase A: Inventory and classify drift
Classification buckets:
- B1: behavior-critical (runtime semantics, timing, memory layout)
- B2: API-contract (message structures, enum values, function signatures)
- B3: observability-only (logging/metrics)
- B4: build/config-only
- B5: docs/tests

Do not merge B1+B4+B5 together.

## Phase B: Sync behavior-critical core first
For each subsystem:
1. Diff local vs 25.3 file set.
2. annotate intent for local deviations.
3. cherry-pick or manual-port smallest coherent unit.
4. build + targeted runtime test gate.

## Phase C: Reconcile API and wrappers
- align `slot_command`, FAPI handlers, and pybind structs with stabilized core.
- update tests and adapters only after API boundary is known-good.

## Phase D: Final cleanup
- reconcile non-critical drift
- run static checks and broader regressions
- produce migration report with accepted vs rejected upstream changes

## Command Cookbook
## 1) Drift count per module
```bash
LOCAL="/Users/yschae87/Library/Mobile Documents/com~apple~CloudDocs/Nvidia/Aerial/cuBB-25-2-Codex"
UP="$LOCAL/upgrade-25-3-workspace"
for d in cmake cuPHY cuPHY-CP cuMAC cuMAC-CP pyaerial; do
  c=$(git diff --no-index --name-only "$LOCAL/$d" "$UP/$d" 2>/dev/null | wc -l | tr -d ' ')
  printf "%s %s\n" "$d" "$c"
done
```

## 2) Generate per-track file list
```bash
LOCAL="/Users/yschae87/Library/Mobile Documents/com~apple~CloudDocs/Nvidia/Aerial/cuBB-25-2-Codex"
UP="$LOCAL/upgrade-25-3-workspace"
git diff --no-index --name-only "$LOCAL/cuPHY-CP" "$UP/cuPHY-CP" > /tmp/track1_cuphycp_files.txt
git diff --no-index --name-only "$LOCAL/cuPHY" "$UP/cuPHY" > /tmp/track2_cuphy_files.txt
```

## 3) Inspect a single file patch
```bash
LOCAL="/Users/yschae87/Library/Mobile Documents/com~apple~CloudDocs/Nvidia/Aerial/cuBB-25-2-Codex"
UP="$LOCAL/upgrade-25-3-workspace"
git diff --no-index "$LOCAL/cuPHY/src/cuphy_channels/pdsch_tx.cpp" "$UP/cuPHY/src/cuphy_channels/pdsch_tx.cpp"
```

## 4) Safe staged copy of one file
```bash
LOCAL="/Users/yschae87/Library/Mobile Documents/com~apple~CloudDocs/Nvidia/Aerial/cuBB-25-2-Codex"
UP="$LOCAL/upgrade-25-3-workspace"
cp "$UP/cuPHY/src/cuphy/pdsch_dmrs/pdsch_dmrs.cu" "$LOCAL/cuPHY/src/cuphy/pdsch_dmrs/pdsch_dmrs.cu"
```

## 5) Build gate
```bash
cd "/Users/yschae87/Library/Mobile Documents/com~apple~CloudDocs/Nvidia/Aerial/cuBB-25-2-Codex"
cmake -Bbuild -GNinja -DCMAKE_TOOLCHAIN_FILE=cuPHY/cmake/toolchains/native -DNVIPC_FMTLOG_ENABLE=OFF -DASIM_CUPHY_SRS_OUTPUT_FP32=ON
cmake --build build -t _pycuphy pycuphycpp
```

## 6) Python package and checks gate
```bash
cd "/Users/yschae87/Library/Mobile Documents/com~apple~CloudDocs/Nvidia/Aerial/cuBB-25-2-Codex"
pyaerial/scripts/install_dev_pkg.sh
pyaerial/scripts/run_static_tests.sh
```

## 7) Unit test gate (vector path required)
```bash
cd "/Users/yschae87/Library/Mobile Documents/com~apple~CloudDocs/Nvidia/Aerial/cuBB-25-2-Codex"
TEST_VECTOR_DIR=/mnt/cicd_tvs/develop/GPU_test_input/ pyaerial/scripts/run_unit_tests.sh
```

## Acceptance Gates by Track
## Track 1 (`cuPHY-CP`) gate
- boot and config load pass
- message ingest and slot enqueue pass
- DL compression + U-plane send pass
- UL PUSCH/PRACH callback paths pass

## Track 2 (`cuPHY`) gate
- PDSCH setup/run pass for non-precoded and precoded configurations
- PUSCH and PRACH setup/run pass for representative vectors
- no ABI mismatch vs driver headers

## Track 3 (`cuMAC`/`pyaerial`) gate
- message ID/type alignment verification
- pybind type conversion tests pass
- representative Python PDSCH pipeline smoke tests pass

## Risk Register and Mitigation
## R1: Silent API drift
Mitigation:
- diff public headers first (`cuphy_api.h`, message headers, slot-command headers).
- fail fast on ABI-impacting diffs.

## R2: Timing regressions after merge
Mitigation:
- keep slot-map timing telemetry comparisons before/after each batch.
- use same hardware profile for A/B checks.

## R3: Compression path incompatibility
Mitigation:
- validate bit-width, scaling (`beta_dl`), and payload size through RU-compatible decode checks.

## R4: Out-of-order slot command behavior changes
Mitigation:
- replay controlled DL_TTI/UL_TTI/TX_DATA ordering cases.

## R5: Wrapper mismatch in pyaerial
Mitigation:
- lock down pybind struct/field mapping tests before final integration.

## Recommended Commit Granularity
Use commit scopes like:
- `cuPHY-CP: sync nv_phy_module timing gate with 25.3 baseline`
- `cuPHY: port pdsch_dmrs updates from 25.3.2`
- `pyaerial: align pdsch params mapping with updated cuphy structs`

Each commit should include:
1. one coherent behavior or API delta
2. build result
3. focused runtime/test evidence

## Suggested Execution Sequence
1. Track 1 core enqueue/timing path (`nv_phy_module`, `cuphydriver_api`, slot maps).
2. Track 1 DL compression/FH path.
3. Track 1 UL PUSCH/PRACH path.
4. Track 2 PDSCH + DMRS + modulation + API structures.
5. Track 2 UL channel API compatibility checks.
6. Track 3 wrappers and MAC message contract alignment.
7. full validation + cleanup.

## Deliverables for a Completed Migration
1. Migration log with accepted/rejected upstream hunks and rationale.
2. Per-track validation evidence (build + tests + runtime checks).
3. Final compatibility matrix across driver, cuPHY, and pyAerial.
4. Post-migration operations notes for config knobs that changed behavior.

## Refactor Coverage
This report supersedes and expands:
- `UPGRADE_25_3_MIGRATION_CHECKLIST.md`

Legacy snapshot preserved:
- `archive/aerial-reports-v1.1-2026-02-11/UPGRADE_25_3_MIGRATION_CHECKLIST.md`

## Related Consolidated Reports
- `architecture-overview-codex.md`
- `control-plane-runtime-codex.md`
- `downlink-pdsch-pipeline-codex.md`
- `uplink-pusch-prach-pipeline-codex.md`
