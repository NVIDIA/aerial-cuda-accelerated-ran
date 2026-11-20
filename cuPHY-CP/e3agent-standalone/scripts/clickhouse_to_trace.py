#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Offline ClickHouse -> E3 replay-trace generator.

Emits the self-describing binary trace consumed by e3agent-standalone replay
mode (see include/replay_format.hpp, the byte-layout source of truth). One record
per (E3 indication, cell), ordered by TsTaiNs with PUSCH before SRS on ties and a
slot's cells consecutive; replay regroups cells sharing a TsTaiNs into one
multi-cell indication. Any blob may be absent => len 0 (zero-filled on replay).
Blobs are the raw row bytes: hest float32, fh/iq/srs_hest int16, srs rb_snr
float32; little-endian, same-arch as the replay host.
"""

from __future__ import annotations

import argparse
import re
import struct
import sys
import time
from collections import defaultdict
from collections.abc import Iterator
from typing import Any, BinaryIO

import numpy as np
import clickhouse_connect

# --- replay_format.hpp mirror (packed, little-endian) -----------------------
MAGIC = struct.unpack('<I', b'E3RT')[0]
VERSION = 0x010200
SHM_LAYOUT_VERSION = 0x010100
TAG_PUSCH, TAG_SRS = 1, 2

# Selectable streams, named as cuphycontroller datalake_data_types.
STREAMS = ('fh', 'pusch', 'hest', 'srs_iq', 'srs', 'srs_hest')

# --since/--until must be a bare datetime literal before it reaches SQL.
TS_RE = re.compile(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(\.\d+)?$')

FILE_HDR = struct.Struct('<IIII')          # magic, version, shm_layout, reserved
REC_HDR = struct.Struct('<HHI')            # tag, reserved, payload_len
BLOB_HDR = struct.Struct('<I')             # len

PUSCH_SLOT = struct.Struct('<QQHHHHHHBBHH')
PUSCH_UE = struct.Struct('<HBIfffHfBBBHHBBIIHBBHHIIHBHffBB')
SRS_SLOT = struct.Struct('<QQHHHHHBBHH')
SRS_UE = struct.Struct('<HffffBfffBBBBBBBBHBBHHIHHHIIHBBBHH')
assert (PUSCH_SLOT.size, PUSCH_UE.size, SRS_SLOT.size, SRS_UE.size) == (34, 73, 32, 74)


def blob(b: bytes) -> bytes:
    return BLOB_HDR.pack(len(b)) + b if b else BLOB_HDR.pack(0)


def to_bytes(arr: object, dtype: Any) -> bytes:
    return np.asarray(arr, dtype=dtype).tobytes() if arr is not None and len(arr) else b''


class BlobStream:
    """Server-side streamed (tai, CellId, blob) rows ordered by (TsTaiNs, CellId)."""

    def __init__(self, client: Any, sql: str, dtype: Any) -> None:
        self.dtype = dtype
        self._gen = self._rows(client, sql)
        self._cur = next(self._gen, None)

    @staticmethod
    def _rows(client: Any, sql: str) -> Iterator:
        with client.query_row_block_stream(sql) as stream:
            for block in stream:
                yield from block

    def fetch(self, tai: int, cell: int) -> bytes | None:
        """Row bytes for exactly (tai, cell), advancing past anything older; else None."""
        while self._cur is not None and (self._cur[0], self._cur[1]) < (tai, cell):
            self._cur = next(self._gen, None)
        if self._cur is not None and self._cur[0] == tai and self._cur[1] == cell:
            b = to_bytes(self._cur[2], self.dtype)
            self._cur = next(self._gen, None)
            return b
        return None


class GroupedBlobStream:
    """Per-UE streamed (tai, CellId, rnti, blob) rows ordered by (tai, CellId, rnti);
    fetch(tai, cell) returns {rnti: bytes} for that cell (srs_hest has N rows per cell)."""

    def __init__(self, client: Any, sql: str, dtype: Any) -> None:
        self.dtype = dtype
        self._gen = BlobStream._rows(client, sql)
        self._cur = next(self._gen, None)

    def fetch(self, tai: int, cell: int) -> dict:
        while self._cur is not None and (self._cur[0], self._cur[1]) < (tai, cell):
            self._cur = next(self._gen, None)
        group = {}
        while self._cur is not None and self._cur[0] == tai and self._cur[1] == cell:
            group[self._cur[2]] = to_bytes(self._cur[3], self.dtype)
            self._cur = next(self._gen, None)
        return group


def rows(client: Any, sql: str) -> tuple[list, list]:
    r = client.query(sql)
    return r.column_names, r.result_rows


def pack_pusch(slot: dict, ue_rows: list, hest: bytes, fh: bytes) -> bytes:
    hdr = PUSCH_SLOT.pack(
        slot['sw'], slot['tai'], slot['SFN'], slot['Slot'], slot['CellId'],
        slot['nRxAnt'], slot['nRxAntSrs'], slot['nCells'], slot['nBsAnts'], 0, len(ue_rows),
        slot['nPrbUlBwp'])
    has_hest = bool(hest)  # no hest bytes => zero the per-UE indices so consumers read nothing
    body = [hdr]
    for u in ue_rows:
        body.append(PUSCH_UE.pack(
            u['rnti'], u['tbCrcFail'], u['cbErrors'], u['rsrp'], u['noiseVar'], u['sinr'],
            u['numCb'], u['rssi'], u['qamModOrder'], u['mcsIndex'], u['mcsTable'],
            u['rbStart'], u['rbSize'], u['StartSymbolIndex'], u['NrOfSymbols'], u['TBSize'],
            u['pduLen'], u['targetCodeRate'], u['newDataIndicator'], u['nrOfLayers'],
            u['layerOffset'], u['ueGrpIdx'],
            u['hOffset'] if has_hest else 0, u['hSize'] if has_hest else 0, u['nSubcarriers'],
            u['nDmrsEstimates'], u['dmrsSymbPos'], u['timingAdvance'], u['cfoHz'],
            u['harqProcessID'], u['rvIndex']))
    body.append(blob(fh))
    body.append(blob(hest))
    return b''.join(body)


def pack_srs(slot: dict, ue_rows: list, iq: bytes, ue_blobs: list) -> bytes:
    hdr = SRS_SLOT.pack(
        slot['sw'], slot['tai'], slot['SFN'], slot['Slot'], slot['CellId'], slot['nCells'],
        slot['nRxAntSrs'], slot['srsCellStartSym'], slot['srsCellNSrsSym'], len(ue_rows),
        slot['nPrbUlBwp'])
    body = [hdr]
    for u, (hb, rb) in zip(ue_rows, ue_blobs):
        body.append(SRS_UE.pack(
            u['rnti'], u['widebandSnr'], u['signalEnergy'], u['noiseEnergy'], u['toaUs'],
            u['hdAntFlag'], u['scCorrRe'], u['scCorrIm'], u['csCorrRatioDb'], u['nAntPorts'],
            u['nSyms'], u['nRepetitions'], u['combSize'], u['combOffset'], u['startSym'],
            u['cyclicShift'], u['frequencyPosition'], u['frequencyShift'], u['frequencyHopping'],
            u['resourceType'], u['tSrs'], u['tOffset'], u['usage'], u['nValidPrg'], u['prgSize'],
            u['nPrbGrps'], len(hb), len(rb), u['sequenceId'], u['configIdx'], u['bandwidthIdx'],
            u['groupOrSequenceHopping'], u['srsStartPrg'], u['startValidPrg']))
    body.append(blob(iq))
    for hb, rb in ue_blobs:
        body.append(blob(hb))
        body.append(blob(rb))
    return b''.join(body)


def write_record(f: BinaryIO, tag: int, body: bytes) -> None:
    f.write(REC_HDR.pack(tag, 0, len(body)))
    f.write(body)


def emit_srs(f: BinaryIO, s: dict, iq_st: BlobStream | None, hest_st: GroupedBlobStream | None) -> None:
    iq = iq_st.fetch(s['tai'], s['CellId']) if iq_st else None
    grp = hest_st.fetch(s['tai'], s['CellId']) if hest_st else {}
    ueb = [(grp.get(u['rnti'], b''), u['_rb']) for u in s['ue']]
    write_record(f, TAG_SRS, pack_srs(s, s['ue'], iq or b'', ueb))


def load_pusch(client: Any, where: str) -> dict:
    cols = ("toUnixTimestamp64Nano(TsTaiNs) AS tai, toUnixTimestamp64Nano(TsSwNs) AS sw, "
            "SFN, Slot, CellId, nCells, nBsAnts, rnti, tbCrcFail, cbErrors, rsrp, noiseVar, "
            "sinr, numCb, rssi, qamModOrder, mcsIndex, mcsTable, rbStart, rbSize, "
            "StartSymbolIndex, NrOfSymbols, TBSize, pduLen, targetCodeRate, newDataIndicator, "
            "nrOfLayers, layerOffset, ueGrpIdx, hOffset, hSize, nSubcarriers, nDmrsEstimates, "
            "dmrsSymbPos, timingAdvance, cfoHz, harqProcessID, rvIndex")
    names, data = rows(client, f"SELECT {cols} FROM fapi {where} ORDER BY tai, CellId, ueGrpIdx")
    idx = {n: i for i, n in enumerate(names)}
    slots = {}
    for r in data:
        key = (r[idx['tai']], r[idx['CellId']])            # one record per (timestamp, cell)
        s = slots.get(key)
        if s is None:
            s = slots[key] = {'tai': r[idx['tai']], 'sw': r[idx['sw']], 'SFN': r[idx['SFN']],
                              'Slot': r[idx['Slot']], 'CellId': r[idx['CellId']],
                              'nCells': r[idx['nCells']], 'nBsAnts': r[idx['nBsAnts']], 'ue': []}
        s['ue'].append({n: r[idx[n]] for n in names})
    return slots


def load_srs(client: Any, where: str, cellmap: dict) -> dict:
    cols = ("toUnixTimestamp64Nano(TsTaiNs) AS tai, toUnixTimestamp64Nano(TsSwNs) AS sw, "
            "SFN, Slot, CellId, nCells, srsCellStartSym, srsCellNSrsSym, rnti, widebandSnr, "
            "signalEnergy, noiseEnergy, toaUs, hdAntFlag, scCorrRe, scCorrIm, csCorrRatioDb, "
            "nAntPorts, nSyms, nRepetitions, combSize, combOffset, startSym, cyclicShift, "
            "frequencyPosition, frequencyShift, frequencyHopping, resourceType, tSrs, tOffset, "
            "usage, nValidPrg, prgSize, nPrbGrps, rbSnrData, sequenceId, configIdx, bandwidthIdx, "
            "groupOrSequenceHopping, srsStartPrg, startValidPrg")
    names, data = rows(client, f"SELECT {cols} FROM srs {where} ORDER BY tai, CellId, rnti")
    idx = {n: i for i, n in enumerate(names)}
    slots = {}
    for r in data:
        key = (r[idx['tai']], r[idx['CellId']])            # one record per (timestamp, cell)
        s = slots.get(key)
        if s is None:
            s = slots[key] = {
                'tai': r[idx['tai']], 'sw': r[idx['sw']], 'SFN': r[idx['SFN']], 'Slot': r[idx['Slot']],
                'CellId': r[idx['CellId']], 'nCells': r[idx['nCells']],
                'srsCellStartSym': r[idx['srsCellStartSym']],
                'srsCellNSrsSym': r[idx['srsCellNSrsSym']],
                'nRxAntSrs': cellmap.get(r[idx['CellId']], (0, 0, 0))[1],
                'nPrbUlBwp': cellmap.get(r[idx['CellId']], (0, 0, 0))[2], 'ue': []}
        u = {n: r[idx[n]] for n in names}
        u['_rb'] = to_bytes(r[idx['rbSnrData']], np.float32)
        s['ue'].append(u)
    return slots


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('-o', '--output', required=True, help='trace output path')
    ap.add_argument('--host', default='localhost')
    ap.add_argument('--port', type=int, default=8123)
    ap.add_argument('--database', default='default')
    ap.add_argument('--user', default='default')
    ap.add_argument('--password', default='')
    ap.add_argument('-l', '--limit', type=int, default=0,
                    help='cap to first N slot timestamps across selected pusch/srs streams; 0 = all')
    ap.add_argument('-s', '--streams', default='all',
                    help="comma-separated subset to include, default 'all': " + ', '.join(STREAMS))
    ap.add_argument('-c', '--cells', default='all',
                    help="comma-separated CellId subset to include, default 'all' (e.g. '1,2')")
    ap.add_argument('--since', help="window start on TsSwNs (wall clock), "
                    "'YYYY-MM-DD HH:MM:SS[.fff]' in the ClickHouse server timezone (typically UTC)")
    ap.add_argument('--until', help="window end on TsSwNs, same format as --since")
    args = ap.parse_args()

    for name, v in (('--since', args.since), ('--until', args.until)):
        if v is not None and not TS_RE.match(v):
            ap.error(f"{name} must be 'YYYY-MM-DD HH:MM:SS[.fff]'")

    sel = set(STREAMS) if args.streams.strip() == 'all' \
        else {s.strip() for s in args.streams.split(',') if s.strip()}
    bad = sel - set(STREAMS)
    if bad:
        print(f"unknown stream(s): {', '.join(sorted(bad))}; valid: {', '.join(STREAMS)}", file=sys.stderr)
        return 2
    if (sel & {'fh', 'hest'}) and 'pusch' not in sel:
        print("note: fh/hest ride PUSCH records; ignored without 'pusch'", file=sys.stderr)
    if (sel & {'srs_iq', 'srs_hest'}) and 'srs' not in sel:
        print("note: srs_iq/srs_hest ride SRS records; ignored without 'srs'", file=sys.stderr)

    cells = None
    if args.cells.strip() != 'all':
        try:
            cells = sorted({int(c) for c in args.cells.split(',') if c.strip()})
        except ValueError:
            ap.error("--cells must be comma-separated CellId integers, e.g. '1,2'")
        if not cells or cells[0] < 0:
            ap.error("--cells must be non-negative CellId integers")

    t0 = time.time()
    print(f"clickhouse_to_trace -> {args.output} "
          f"[{args.since or 'begin'} .. {args.until or 'end'}] "
          f"streams={','.join(s for s in STREAMS if s in sel)} "
          f"cells={'all' if cells is None else ','.join(map(str, cells))}", file=sys.stderr)

    client = clickhouse_connect.get_client(host=args.host, port=args.port,
                                           database=args.database, username=args.user,
                                           password=args.password,
                                           autogenerate_session_id=False)

    win = []
    if args.since:
        win.append(f"TsSwNs >= toDateTime64('{args.since}', 9)")
    if args.until:
        win.append(f"TsSwNs <= toDateTime64('{args.until}', 9)")
    if cells is not None:
        win.append(f"CellId IN ({','.join(map(str, cells))})")

    def build_where(*extra: str) -> str:
        preds = win + [p for p in extra if p]
        return ('WHERE ' + ' AND '.join(preds)) if preds else ''

    where = build_where()
    if args.limit > 0:
        slot_tbls = [t for t, s in (('fapi', 'pusch'), ('srs', 'srs')) if s in sel]
        if not slot_tbls:
            print("--limit needs 'pusch' and/or 'srs' selected", file=sys.stderr)
            return 2
        union = ' UNION DISTINCT '.join(f"SELECT DISTINCT TsTaiNs AS t FROM {t} {where}"
                                        for t in slot_tbls)
        cut = rows(client, "SELECT toUnixTimestamp64Nano(t) FROM "
                           f"(SELECT t FROM ({union}) ORDER BY t LIMIT {args.limit})")[1]
        if not cut:
            print('no slots found in the selected window', file=sys.stderr)
            return 1
        where = build_where(f"toUnixTimestamp64Nano(TsTaiNs) <= {cut[-1][0]}")

    # Cell parameters are static per cell; two values means the window spans runs.
    cell_cols = ('nRxAnt', 'nRxAntSrs', 'nPrbUlBwp')

    def set_cell(cid: int, i: int, v: int) -> None:
        e = cellmap.setdefault(cid, [0, 0, 0])
        if e[i] and e[i] != v:
            sys.exit(f"cell {cid}: conflicting {cell_cols[i]} ({e[i]} vs {v}); "
                     "window spans multiple runs, narrow --since/--until or --cells")
        e[i] = v

    cellmap: dict = {}
    for cid, *vals in rows(client, f"SELECT DISTINCT CellId, {', '.join(cell_cols)} FROM fh {where}")[1]:
        for i, v in enumerate(vals):
            set_cell(cid, i, v)
    if 'srs' in sel:                                       # SRS-only windows have no fh rows
        for cid, nrxs, nprb in rows(client, "SELECT DISTINCT CellId, nRxAntSrs, nPrbUlBwp "
                                            f"FROM srs_iq {where}")[1]:
            set_cell(cid, 1, nrxs)
            set_cell(cid, 2, nprb)

    pusch = load_pusch(client, where) if 'pusch' in sel else {}
    for s in pusch.values():
        s['nRxAnt'], s['nRxAntSrs'], s['nPrbUlBwp'] = cellmap.get(s['CellId'], (0, 0, 0))
    srs = load_srs(client, where, cellmap) if 'srs' in sel else {}

    # Group per-cell records by timestamp; within a slot cells ascend by CellId
    # and PUSCH precedes SRS (replay regroups them into one multi-cell indication).
    pusch_by_tai: dict = defaultdict(list)
    for s in pusch.values():
        pusch_by_tai[s['tai']].append(s)
    srs_by_tai: dict = defaultdict(list)
    for s in srs.values():
        srs_by_tai[s['tai']].append(s)
    for lst in pusch_by_tai.values():
        lst.sort(key=lambda s: s['CellId'])
    for lst in srs_by_tai.values():
        lst.sort(key=lambda s: s['CellId'])

    hest_st = (BlobStream(client, f"SELECT toUnixTimestamp64Nano(TsTaiNs) AS tai, CellId, hestData "
                                  f"FROM hest {where} ORDER BY tai, CellId", np.float32)
               if 'hest' in sel else None)
    fh_st = (BlobStream(client, f"SELECT toUnixTimestamp64Nano(TsTaiNs) AS tai, CellId, fhData "
                                f"FROM fh {where} ORDER BY tai, CellId", np.int16)
             if 'fh' in sel else None)
    srs_iq_st = (BlobStream(client, f"SELECT toUnixTimestamp64Nano(TsTaiNs) AS tai, CellId, iqData "
                                    f"FROM srs_iq {where} ORDER BY tai, CellId", np.int16)
                 if 'srs_iq' in sel else None)
    srs_hest_st = (GroupedBlobStream(client, f"SELECT toUnixTimestamp64Nano(TsTaiNs) AS tai, CellId, rnti, "
                                             f"hestData FROM srs_hest {where} ORDER BY tai, CellId, rnti", np.int16)
                   if 'srs_hest' in sel else None)

    n_pusch = n_srs = 0
    total = len(pusch) + len(srs)

    def tick() -> None:
        if not total:
            return
        d = n_pusch + n_srs
        if d % 1000 == 0 or d == total:
            sys.stderr.write(f"\r  {d}/{total} records ({100.0 * d / total:5.1f}%) "
                             f"{time.time() - t0:6.1f}s")
            sys.stderr.flush()

    with open(args.output, 'wb') as f:
        f.write(FILE_HDR.pack(MAGIC, VERSION, SHM_LAYOUT_VERSION, 0))
        for tai in sorted(set(pusch_by_tai) | set(srs_by_tai)):
            for s in pusch_by_tai.get(tai, ()):
                hb = hest_st.fetch(tai, s['CellId']) if hest_st else b''  # absent => empty, indices zeroed
                fb = fh_st.fetch(tai, s['CellId']) if fh_st else b''
                write_record(f, TAG_PUSCH, pack_pusch(s, s['ue'], hb or b'', fb or b''))
                n_pusch += 1
                tick()
            for s in srs_by_tai.get(tai, ()):
                emit_srs(f, s, srs_iq_st, srs_hest_st)
                n_srs += 1
                tick()
    if total:
        sys.stderr.write("\n")

    print(f"wrote {args.output}: {n_pusch} PUSCH, {n_srs} SRS records "
          f"in {time.time() - t0:.1f}s [streams: {', '.join(s for s in STREAMS if s in sel)}]")
    return 0


if __name__ == '__main__':
    sys.exit(main())
