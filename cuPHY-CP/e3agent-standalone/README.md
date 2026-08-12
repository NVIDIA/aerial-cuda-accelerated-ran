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
./restart_e3agent_standalone.sh
```

This stages the aerial sources, builds the Docker image, and starts the `e3agent-standalone` container with shared memory and the E3AP ports. Press Ctrl+C to stop. Edit `config/e3agent-standalone.example.yaml` first (see [Configuration](#configuration)) to pick the mode and settings; for `replay` mode, generate a trace beforehand (see [Data generation](#data-generation)).

### 2. Attach a dApp

Point the dApp `ipc_mode` to `container:e3agent-standalone` in its `e3_config.json`, start the dApp, then subscribe exactly as you would in production. Example with prb-power-python dApp:

```bash
docker exec -it dapp-prb-power-python \
    /opt/src/common/client/e3_e2e_test.py -a NVIDIA_L1 -t 1,4,5,6,78 -d 10
```

The dApp receives indications and runs inference as if connected to a live L1.

## Configuration

Single file, `config/e3agent-standalone.example.yaml`. `mode` picks the active block; unknown keys are rejected.

* **agent**: E3AP ZMQ ports `rep_port`/`pub_port`/`sub_port` (REP/PUB/SUB sockets), mirroring cuphycontroller.
* **rows**: shared memory ring depths per buffer.
* **cpu.feeder_core**: pins the feeder threads; use an isolated core for jitter free synth, or `-1` for none.
* **synth**: a built-in generator that needs no database or capture. It ticks a slot clock (`slot_tick_us`) and, on each uplink slot, fills the PUSCH, SRS, and FH buffers with deterministic metrics and simple patterned channel estimates and IQ. `tdd_pattern` sets which slots are uplink, `srs_periodicity_ms` how often SRS appears, and `duration_slots` how long to run (`0` = forever). It also fills `ts_tai`, so the dApp sees the same timing as replay. Useful for exercising the full subscription, indication, and inference path; the IQ itself is not physically meaningful.
* **replay**: `path`, `loops` (0 loops forever; pacing follows recorded `ts_tai`).

## Data generation

Two sources feed the agent, selected by `mode`; more may be added in future releases.

* **synth**: built-in, needs no data (see the `synth` keys above).
* **replay**: plays back a trace file. We provide `clickhouse_to_trace.py` to export a Data Lake capture to a trace, for example:

```bash
python3 scripts/clickhouse_to_trace.py -o data/session.trace --limit 600
```

Useful flags:

* `--streams`/`-s`: data types to include (`fh,pusch,hest,srs_iq,srs,srs_hest`, default `all`).
* `--limit`/`-l`: cap the trace to the first N slot timestamps across the selected PUSCH/SRS streams.
* `--since`/`--until`: window by wall-clock time, `'YYYY-MM-DD HH:MM:SS'` in the ClickHouse server timezone (typically UTC), e.g. `--since '2026-06-23 14:00:00' --until '2026-06-23 14:05:00'`.

plus the usual ClickHouse connection flags. Then set `mode: replay` and point `replay.path` at the result (`/data` is mounted read only into the container). To write a trace from another source, see [`docs/trace_format.md`](docs/trace_format.md).

## Notes

* **Drift guard**: `check_drift.py` runs on every build and fails if the vendored `e3_types.hpp` diverges from the canonical `aerial_sdk` structs. Disable with `-DE3SA_CHECK_DRIFT=OFF` for fast local iteration.
* **Numerology**: fixed mu=1 (500 us slot, 20 slots/frame), single cell.
