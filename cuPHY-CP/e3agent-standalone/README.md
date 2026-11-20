# E3 Agent Standalone

A self contained environment that runs the production E3 Agent without the full RAN. It lets you develop, debug, and test dApps against realistic E3AP telemetry without cuPHY, FAPI, live radio, or a GPU pipeline.

<div align="center">
  <img src="docs/e3agent-standalone-overview.png" alt="E3 Agent Standalone overview" width="1000"><br>
  <em>One telemetry trace, many sources; the dApp sees the same E3 interface as live.</em>
</div>

## Overview

e3agent-standalone compiles the real `e3_agent.cpp` from the sibling `data_lake/` unmodified and feeds its shared memory buffers from a local driver instead of from a live L1. A dApp connecting to it gets the exact same subscriptions, indication schema, and shared memory layout it would see against a real cuBB L1, so logic developed here runs unchanged in production.

It feeds the agent in two ways:

* **synth**: a config driven generator produces synthetic PUSCH, SRS, and FH content on a slot clock.
* **replay**: a recorded binary trace is played back into the buffers, paced by the captured TAI timing.

## How It Works

In a live deployment, cuPHY and FAPI fill the Data Lake from real radio traffic, the E3 Agent serves that data over E3AP, and dApps subscribe to it. e3agent-standalone keeps the E3 Agent and the dApp side identical and replaces only the data source:

* **Data Lake shim**: a minimal stand in for the real Data Lake, owning the same double buffered info structs the agent reads.
* **E3 Agent**: production code, untouched. Same ZMQ ports, same NVIDIA KPM service model, same indication JSON.
* **Feeder / Replay**: fills the shared memory buffers (synth generator or trace) where cuPHY would in a live system.
* **dApp framework**: unchanged. Same client, same subscription flow, same inference logic.

The trace is a single self describing binary (`replay_format.hpp`, magic `E3RT`), so any source can produce it. We provide `scripts/clickhouse_to_trace.py` as a ready-to-use generator that exports a Data Lake capture to a trace; the same format can equally be emitted by other sources such as math models, Sionna, or AODT. See [`docs/trace_format.md`](docs/trace_format.md) for the file structure and pacing you need to write your own.

## Requirements

The build stages the production `e3_agent.cpp`, `e3_agent.hpp`, and `data_lake.hpp` from the sibling `../data_lake/`, so no extra setup is needed. To build against sources elsewhere, point the `AERIAL_DATA_LAKE_DIR` environment variable at another `cuPHY-CP/data_lake/`.

## Quick Start

All commands run from `cuPHY-CP/e3agent-standalone/`.

### 1. Build and run

```bash
./restart_e3agent_standalone.sh              # start (reuses the image; builds it the first time)
./restart_e3agent_standalone.sh -b           # rebuild after C++ or staged data_lake changes
./restart_e3agent_standalone.sh -c my.yaml   # start with a specific config
./restart_e3agent_standalone.sh -t ~/traces/run1.trace  # replay any host trace
```

This starts the `e3agent-standalone` container with shared memory and E3AP ports. `config/` and `data/` are mounted, so config- or trace-only edits require no rebuild. Press Ctrl+C to stop. The config file sets the mode and settings, and command line overrides take precedence over it; see [Configuration](#configuration) for every key and flag, or run with `-h`. For `replay` mode, generate a trace beforehand (see [Data generation](#data-generation)).

### 2. Attach a dApp

Point the dApp `ipc_mode` to `container:e3agent-standalone` in its `e3_config.json`, start the dApp, then subscribe exactly as you would in production. Example with prb-power-python dApp:

```bash
docker exec -it dapp-prb-power-python \
    /opt/src/common/client/e3_e2e_test.py -a NVIDIA_L1 -t 1,4,5,6,78 -d 10
```

The dApp receives indications and runs inference as if connected to a live L1.

## Configuration

Single file, `config/e3agent-standalone.example.yaml`. `mode` picks the active block (`synth` or `replay`); unknown keys are rejected. Flags go to `restart_e3agent_standalone.sh` and take precedence over the file.

| Key | Default | Flag | Meaning |
|---|---|---|---|
| `mode` | required | `--synth`, `--replay` | Data source |
| `log_level` | `INF` | `--log-level LVL` | `VRB`, `DBG`, `INF`, `WRN`, `ERR`, or `FAT` |
| `agent.rep_port`, `pub_port`, `sub_port` | 5555, 5556, 5557 | | E3AP ZMQ ports (REP/PUB/SUB), mirroring cuphycontroller |
| `rows.fh`, `pusch`, `hest`, `srs_iq`, `srs`, `srs_hest` | 120, 180, 140, 40, 70, 90 | | Shared memory ring depth per buffer |
| `cpu.feeder_core` | -1 | `--feeder-core N` | Core for the feeder threads; an isolated core keeps slot timing steady, -1 = no pinning |
| `synth.slot_tick_us` | 500 | | Slot clock |
| `synth.tdd_pattern` | `DDDSU` | | Uplink slots are `U`; `D` and `S` are skipped |
| `synth.srs_periodicity_ms` | 40 | | SRS cadence; SRS rides a `U` slot |
| `synth.duration_slots` | 0 | `--duration-slots N` | Run length; 0 = forever |
| `synth.n_cells` | 1 | | Cells per slot, 1 to 40; every ring needs at least `n_cells` rows |
| `replay.path` | required | `-t`, `--trace PATH` | Trace file; the flag takes any host path, mounts it read only, and implies `--replay` |
| `replay.loops` | 0 | `--loops N` | Passes over the trace; 0 = forever |
| `replay.wait_for_subscriber` | false | `--wait-subscriber` | Start each pass on the first E3 subscription; restart from the first slot when the last one is released |

Replay paces slots by the recorded `ts_tai` (gaps clamped to 1 s). With `wait_for_subscriber`, an abandoned pass does not count toward `loops`, and a dApp that crashes without releasing keeps its subscription, so the replay keeps running.

## Data generation

Two sources feed the agent, selected by `mode`; more may be added in future releases.

* **synth**: built-in, needs no database or capture. On each uplink slot it fills the PUSCH, SRS, and FH buffers with deterministic metrics and simple patterned channel estimates and IQ, and it fills `ts_tai`, so the dApp sees the same timing as replay. Useful for exercising the full subscription, indication, and inference path; the IQ itself is not physically meaningful.
* **replay**: plays back a trace file. We provide `clickhouse_to_trace.py` to export a Data Lake capture to a trace, for example:

```bash
python3 scripts/clickhouse_to_trace.py -o data/session.trace --limit 600
```

Useful flags:

* `--streams`/`-s`: data types to include (`fh,pusch,hest,srs_iq,srs,srs_hest`, default `all`).
* `--cells`/`-c`: CellId subset to include (default `all`, e.g. `51,52`); all cells of a slot are kept together.
* `--limit`/`-l`: cap the trace to the first N slot timestamps across the selected PUSCH/SRS streams.
* `--since`/`--until`: window by wall-clock time, `'YYYY-MM-DD HH:MM:SS'` in the ClickHouse server timezone (typically UTC), e.g. `--since '2026-06-23 14:00:00' --until '2026-06-23 14:05:00'`.

plus the usual ClickHouse connection flags. Then set `mode: replay` and point `replay.path` at the result (`/data` is mounted read only into the container), or pass it with `-t`. To write a trace from another source, see [`docs/trace_format.md`](docs/trace_format.md).

## Notes

* **Drift guard**: `check_drift.py` runs on every build and fails if the vendored `e3_types.hpp` diverges from the canonical `aerial_sdk` structs. Disable with `-DE3SA_CHECK_DRIFT=OFF` for fast local iteration.
* **Numerology**: fixed mu=1 (500 us slot, 20 slots/frame).
* **Multi cell**: synth emits `synth.n_cells` cells per slot, and replay regroups the per-cell trace records that share a `ts_tai` into one multi-cell indication. More cells widen each slot, so size `rows` accordingly.
