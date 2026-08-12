# Trace Format

The telemetry trace is a single self-describing binary that `src/replay.cpp` reads back into the agent's shared memory. Any generator can produce it; `scripts/clickhouse_to_trace.py` is the example we provide. This document describes the file layout and pacing, for reference and for building new generators.

The byte-level source of truth is `include/replay_format.hpp`: field names, types, and struct sizes all come from there. This document explains the framing and semantics around those structs. It does not restate the tensor layout of the data blobs; for that, see [`data_representation.md`](../../../docs/data_representation.md) and [`application_development_guide.md`](../../../docs/application_development_guide.md).

## File layout

A trace is a fixed header followed by a stream of records, one per E3 indication, in ascending `ts_tai` order:

```text
+------------------------------------------+  offset 0
|  FileHeader (16 B)                       |  magic, version, shm_layout_version
+------------------------------------------+
|  Record 0                                |  \
+------------------------------------------+   \  ascending ts_tai
|  Record 1                                |   /  (PUSCH before SRS on ties)
+------------------------------------------+  /
|  ...                                     |
+------------------------------------------+
```

The header carries the `E3RT` magic, the trace `version`, and the `shm_layout_version` (`e3::SHM_LAYOUT_VERSION` at generation time). A reader rejects any file whose magic or SHM layout version does not match its own build, so a stale trace fails fast rather than feeding garbage into the buffers.

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
|  PuschSlotHeader (32 B)                  |
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

The slot and UE headers mirror the agent's `E3*BufferInfo` and `E3*Metrics` structs, minus the SHM bookkeeping (buffer/write indices and row offsets) that replay recomputes on the fly. Per-UE slicing into the blobs uses the offsets and sizes carried in the UE metrics.

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

Records are ordered by `ts_tai` (the TAI capture timestamp), and when a PUSCH and an SRS record share a timestamp the PUSCH one comes first.

## Pacing

Replay is driven entirely by `timestamp_tai_ns`. The delay between two records is the difference of their TAI timestamps, with large gaps clamped to one second so a pause during capture does not stall playback. `replay.loops` repeats the whole file (`0` loops forever). Synth mode does not read a file; it paces on `slot_tick_us` and synthesizes `timestamp_tai_ns` itself, so the dApp sees the same timing either way.

A generator must therefore set `timestamp_tai_ns` monotonically on every record; that is the clock the reader paces on, not the software timestamp.

## Writing a generator

1. Write the `FileHeader` with the matching `magic`, `version`, and `shm_layout_version`.
2. Emit records in ascending `ts_tai` order, PUSCH before SRS on ties.
3. Fill the slot and UE headers per `replay_format.hpp`, and set `timestamp_tai_ns` for pacing.
4. Append each stream as a blob, or write `len == 0` to omit it.
