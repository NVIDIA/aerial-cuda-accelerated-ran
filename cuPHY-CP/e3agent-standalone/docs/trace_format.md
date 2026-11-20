# Trace Format

The telemetry trace is a single self-describing binary that `src/replay.cpp` reads back into the agent's shared memory. Any generator can produce it; `scripts/clickhouse_to_trace.py` is the example we provide. This document describes the file layout and pacing, for reference and for building new generators.

The byte-level source of truth is `include/replay_format.hpp`: field names, types, and struct sizes all come from there. This document explains the framing and semantics around those structs. It does not restate the tensor layout of the data blobs; for that, see [`data_representation.md`](../../../docs/data_representation.md) and [`application_development_guide.md`](../../../docs/application_development_guide.md).

## File layout

A trace is a fixed header followed by a stream of records, one per cell-slot, in ascending `ts_tai` order. A multi-cell slot is several records sharing one `ts_tai`; replay regroups them into a single indication (see [Ordering](#ordering)):

```text
+------------------------------------------+  offset 0
|  FileHeader (16 B)                       |  magic, version, shm_layout_version
+------------------------------------------+
|  PUSCH  ts_tai=T0  cell 0                |  \
+------------------------------------------+   |  slot T0: cells grouped by ts_tai,
|  PUSCH  ts_tai=T0  cell 1                |   |  ascending cell_id, PUSCH before SRS
+------------------------------------------+   |
|  SRS    ts_tai=T0  cell 0                |   |
+------------------------------------------+   |
|  SRS    ts_tai=T0  cell 1                |  /
+------------------------------------------+
|  PUSCH  ts_tai=T1  cell 0                |  \  next slot (ascending ts_tai)
+------------------------------------------+   \
|  ...                                     |   /
+------------------------------------------+  /
```

The header carries the `E3RT` magic, the trace `version`, and the `shm_layout_version` (`e3::SHM_LAYOUT_VERSION` at generation time). A reader rejects any file whose magic does not match or whose `version` it cannot parse, so a stale trace fails fast rather than feeding garbage into the buffers; an SHM layout mismatch is logged as a warning. The reader accepts v1.2.0 and v1.1.0. v1.1.0 does not record `n_prb_ul_bwp` (its `PuschSlotHeader` is 32 B; its `SrsSlotHeader` has a reserved field there), so replay assumes 273 PRB and logs a warning. That is correct only for 100 MHz captures; recapture narrower carriers as v1.2.0. The v1.1.0 `SrsUeMetrics` is 65 B and ends before `sequence_id`, so replay publishes SRS pilot streams 81-86 as 0.

## Records

Each record is a small header plus a body. The header gives the record type and the body length, so a reader can skip records it does not care about:

```text
+------------------------------------------+
|  RecordHeader (8 B)                      |  tag, payload_len
+------------------------------------------+
|  body  (payload_len bytes)               |
+------------------------------------------+
```

`tag` is either `TAG_PUSCH (1)` or `TAG_SRS (2)`, and the body layout depends on it.

A **PUSCH** body is the slot header, the per-UE metrics, then the fronthaul and channel-estimate blobs:

```text
+------------------------------------------+
|  PuschSlotHeader (34 B)                  |
+------------------------------------------+
|  PuschUeMetrics x n_ue                   |
+------------------------------------------+
|  blob: fh     (whole row)                |
+------------------------------------------+
|  blob: hest   (whole slot row)           |
+------------------------------------------+
```

An **SRS** body is the slot header, the per-UE metrics, the cell-level IQ blob, then the per-UE channel-estimate and per-RB-SNR blobs:

```text
+------------------------------------------+
|  SrsSlotHeader (32 B)                    |
+------------------------------------------+
|  SrsUeMetrics x n_srs_ue                 |
+------------------------------------------+
|  blob: iq     (per cell)                 |
+------------------------------------------+
|  per UE, in order:                       |
|    blob: srs_hest                        |
|    blob: srs_rb_snr                      |
+------------------------------------------+
```

The slot and UE headers mirror the agent's `E3*BufferInfo` and `E3*Metrics` structs, minus the SHM bookkeeping (buffer/write indices and row offsets) that replay recomputes on the fly. Per-UE slicing into the blobs uses the offsets and sizes carried in the UE metrics. Each header also carries its own `cell_id`; `n_cells` is advisory (the count the capture saw), since replay frames slots on `ts_tai` alone.

## Blobs

Each stream is stored as a blob: a 4-byte length followed by that many raw bytes.

```text
+------------------------------------------+
|  BlobHeader { len }                      |
+------------------------------------------+
|  len bytes (raw, little-endian)          |   len == 0 => stream absent
+------------------------------------------+
```

A length of zero means the stream is absent, and replay zero-fills it. Payloads are the raw on-wire bytes for the same architecture: `hest` is float32, `fh`/`iq`/`srs_hest` are int16, and `srs_rb_snr` is float32. The tensor axis order of each blob (`fh`, `hest`, `iq`, `srs_hest`, `srs_rb_snr`) is defined in [`data_representation.md`](../../../docs/data_representation.md); for example, `hest` is `[n_dmrs, n_subcarrier, n_bs_ants, n_layers]` (row-major) and `srs_hest` is `[n_ant_ports, n_rx_ant_srs, n_prg]` (column-major).

Keep each UE's `h_offset`/`h_size` consistent with the `hest` blob: set them to the UE's slice when the blob is present, and to 0 when it is absent, so a consumer reads no estimate rather than stale bytes. (SRS is automatic: `srs_hest_size` is just the per-UE blob length.)

## Ordering

Records are ordered by `ts_tai` (the TAI capture timestamp). Within one `ts_tai`, a slot's cells ascend by `cell_id`, and a PUSCH record comes before an SRS record on ties. Replay accumulates the consecutive records that share a `ts_tai` and publishes them as one multi-cell indication when the timestamp advances; a single-cell trace has one record per `ts_tai` and plays back as a one-cell slot.

## Pacing

Replay is driven entirely by `timestamp_tai_ns`. The delay between two records is the difference of their TAI timestamps, with large gaps clamped to one second so a pause during capture does not stall playback. `replay.loops` repeats the whole file (`0` loops forever). Synth mode does not read a file; it paces on `slot_tick_us` and synthesizes `timestamp_tai_ns` itself, so the dApp sees the same timing either way.

A generator must therefore set `timestamp_tai_ns` monotonically on every record; that is the clock the reader paces on, not the software timestamp.

## Writing a generator

1. Write the `FileHeader` with the matching `magic`, `version`, and `shm_layout_version`.
2. Emit records in ascending `ts_tai` order; within a slot order cells by `cell_id`, PUSCH before SRS on ties.
3. Fill the slot and UE headers per `replay_format.hpp`, and set `timestamp_tai_ns` for pacing.
4. Append each stream as a blob, or write `len == 0` to omit it.
