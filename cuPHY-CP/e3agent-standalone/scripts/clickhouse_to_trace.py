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
per E3 indication, ordered by TsTaiNs with PUSCH before SRS on ties. Any blob may
be absent => len 0 (zero-filled on replay); when hest is absent the per-UE
h_offset/h_size are zeroed so consumers read no estimate. Blobs are the raw row
bytes: hest float32, fh/iq/srs_hest int16, srs rb_snr float32; little-endian,
same-arch as the replay host.
"""

from __future__ import annotations

import argparse
import re
import struct
import sys
from collections.abc import Iterator
from typing import Any, BinaryIO

import numpy as np
import clickhouse_connect

# --- replay_format.hpp mirror (packed, little-endian) -----------------------
MAGIC = struct.unpack('<I', b'E3RT')[0]
VERSION = 0x010100
SHM_LAYOUT_VERSION = 0x010100
TAG_PUSCH, TAG_SRS = 1, 2

# Selectable streams, named as cuphycontroller datalake_data_types.
STREAMS = ('fh', 'pusch', 'hest', 'srs_iq', 'srs', 'srs_hest')

# --since/--until must be a bare datetime literal before it reaches SQL.
TS_RE = re.compile(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(\.\d+)?$')

FILE_HDR = struct.Struct('<IIII')          # magic, version, shm_layout, reserved
REC_HDR = struct.Struct('<HHI')            # tag, reserved, payload_len
BLOB_HDR = struct.Struct('<I')             # len

PUSCH_SLOT = struct.Struct('<QQHHHHHHBBH')
PUSCH_UE = struct.Struct('<HBIfffHfBBBHHBBIIHBBHHIIHBHffBB')
SRS_SLOT = struct.Struct('<QQHHHHHBBHH')
SRS_UE = struct.Struct('<HffffBfffBBBBBBBBHBBHHIHHHII')
assert (PUSCH_SLOT.size, PUSCH_UE.size, SRS_SLOT.size, SRS_UE.size) == (32, 73, 32, 65)


def blob(b: bytes) -> bytes:
    return BLOB_HDR.pack(len(b)) + b if b else BLOB_HDR.pack(0)


def to_bytes(arr: object, dtype: Any) -> bytes:
    return np.asarray(arr, dtype=dtype).tobytes() if arr is not None and len(arr) else b''


class BlobStream:
    """Server-side streamed (tai_ns, blob) rows ordered by TsTaiNs."""

    def __init__(self, client: Any, sql: str, dtype: Any) -> None:
        self.dtype = dtype
        self._gen = self._rows(client, sql)
        self._cur = next(self._gen, None)

    @staticmethod
    def _rows(client: Any, sql: str) -> Iterator:
        with client.query_row_block_stream(sql) as stream:
            for block in stream:
                yield from block

    def fetch(self, tai: int) -> bytes | None:
        """Row bytes for exactly tai, advancing past anything older; else None."""
        while self._cur is not None and self._cur[0] < tai:
            self._cur = next(self._gen, None)
        if self._cur is not None and self._cur[0] == tai:
            b = to_bytes(self._cur[1], self.dtype)
            self._cur = next(self._gen, None)
            return b
        return None


class GroupedBlobStream:
    """Per-UE streamed (tai, rnti, blob) rows ordered by (tai, rnti); fetch(tai)
    returns {rnti: bytes} for the slot (srs_hest has N rows per slot)."""

    def __init__(self, client: Any, sql: str, dtype: Any) -> None:
        self.dtype = dtype
        self._gen = BlobStream._rows(client, sql)
        self._cur = next(self._gen, None)

    def fetch(self, tai: int) -> dict:
        while self._cur is not None and self._cur[0] < tai:
            self._cur = next(self._gen, None)
        group = {}
        while self._cur is not None and self._cur[0] == tai:
            group[self._cur[1]] = to_bytes(self._cur[2], self.dtype)
            self._cur = next(self._gen, None)
        return group


def rows(client: Any, sql: str) -> tuple[list, list]:
    r = client.query(sql)
    return r.column_names, r.result_rows


def pack_pusch(slot: dict, ue_rows: list, hest: bytes, fh: bytes) -> bytes:
    hdr = PUSCH_SLOT.pack(
        slot['sw'], slot['tai'], slot['SFN'], slot['Slot'], slot['CellId'],
        slot['nRxAnt'], slot['nRxAntSrs'], slot['nCells'], slot['nBsAnts'], 0, len(ue_rows))
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
        slot['nRxAntSrs'], slot['srsCellStartSym'], slot['srsCellNSrsSym'], len(ue_rows), 0)
    body = [hdr]
    for u, (hb, rb) in zip(ue_rows, ue_blobs):
        body.append(SRS_UE.pack(
            u['rnti'], u['widebandSnr'], u['signalEnergy'], u['noiseEnergy'], u['toaUs'],
            u['hdAntFlag'], u['scCorrRe'], u['scCorrIm'], u['csCorrRatioDb'], u['nAntPorts'],
            u['nSyms'], u['nRepetitions'], u['combSize'], u['combOffset'], u['startSym'],
            u['cyclicShift'], u['frequencyPosition'], u['frequencyShift'], u['frequencyHopping'],
            u['resourceType'], u['tSrs'], u['tOffset'], u['usage'], u['nValidPrg'], u['prgSize'],
            u['nPrbGrps'], len(hb), len(rb)))
    body.append(blob(iq))
    for hb, rb in ue_blobs:
        body.append(blob(hb))
        body.append(blob(rb))
    return b''.join(body)


def write_record(f: BinaryIO, tag: int, body: bytes) -> None:
    f.write(REC_HDR.pack(tag, 0, len(body)))
    f.write(body)


def emit_srs(f: BinaryIO, s: dict, iq_st: BlobStream | None, hest_st: GroupedBlobStream | None) -> None:
    iq = iq_st.fetch(s['tai']) if iq_st else None
    grp = hest_st.fetch(s['tai']) if hest_st else {}
    ueb = [(grp.get(u['rnti'], b''), u['_rb']) for u in s['ue']]
    write_record(f, TAG_SRS, pack_srs(s, s['ue'], iq or b'', ueb))


def load_pusch(client: Any, where: str) -> dict:
    cols = ("toUnixTimestamp64Nano(TsTaiNs) AS tai, toUnixTimestamp64Nano(TsSwNs) AS sw, "
            "SFN, Slot, CellId, nCells, nBsAnts, rnti, tbCrcFail, cbErrors, rsrp, noiseVar, "
            "sinr, numCb, rssi, qamModOrder, mcsIndex, mcsTable, rbStart, rbSize, "
            "StartSymbolIndex, NrOfSymbols, TBSize, pduLen, targetCodeRate, newDataIndicator, "
            "nrOfLayers, layerOffset, ueGrpIdx, hOffset, hSize, nSubcarriers, nDmrsEstimates, "
            "dmrsSymbPos, timingAdvance, cfoHz, harqProcessID, rvIndex")
    names, data = rows(client, f"SELECT {cols} FROM fapi {where} ORDER BY tai, ueGrpIdx")
    idx = {n: i for i, n in enumerate(names)}
    slots = {}
    for r in data:
        tai = r[idx['tai']]
        s = slots.get(tai)
        if s is None:
            s = slots[tai] = {'tai': tai, 'sw': r[idx['sw']], 'SFN': r[idx['SFN']],
                              'Slot': r[idx['Slot']], 'CellId': r[idx['CellId']],
                              'nCells': r[idx['nCells']], 'nBsAnts': r[idx['nBsAnts']], 'ue': []}
        elif r[idx['CellId']] != s['CellId']:              # single-cell only; multi-cell TBD
            raise SystemExit(f"multi-cell PUSCH at tai={tai} (CellId {s['CellId']} and "
                             f"{r[idx['CellId']]}); not supported")
        s['ue'].append({n: r[idx[n]] for n in names})
    return slots


def load_srs(client: Any, where: str, cellmap: dict) -> dict:
    cols = ("toUnixTimestamp64Nano(TsTaiNs) AS tai, toUnixTimestamp64Nano(TsSwNs) AS sw, "
            "SFN, Slot, CellId, nCells, srsCellStartSym, srsCellNSrsSym, rnti, widebandSnr, "
            "signalEnergy, noiseEnergy, toaUs, hdAntFlag, scCorrRe, scCorrIm, csCorrRatioDb, "
            "nAntPorts, nSyms, nRepetitions, combSize, combOffset, startSym, cyclicShift, "
            "frequencyPosition, frequencyShift, frequencyHopping, resourceType, tSrs, tOffset, "
            "usage, nValidPrg, prgSize, nPrbGrps, rbSnrData")
    names, data = rows(client, f"SELECT {cols} FROM srs {where} ORDER BY tai, rnti")
    idx = {n: i for i, n in enumerate(names)}
    slots = {}
    for r in data:
        tai = r[idx['tai']]
        s = slots.get(tai)
        if s is None:
            s = slots[tai] = {
                'tai': tai, 'sw': r[idx['sw']], 'SFN': r[idx['SFN']], 'Slot': r[idx['Slot']],
                'CellId': r[idx['CellId']], 'nCells': r[idx['nCells']],
                'srsCellStartSym': r[idx['srsCellStartSym']],
                'srsCellNSrsSym': r[idx['srsCellNSrsSym']],
                'nRxAntSrs': cellmap.get(r[idx['CellId']], (0, 0))[1], 'ue': []}
        elif r[idx['CellId']] != s['CellId']:              # single-cell only; multi-cell TBD
            raise SystemExit(f"multi-cell SRS at tai={tai} (CellId {s['CellId']} and "
                             f"{r[idx['CellId']]}); not supported")
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

    client = clickhouse_connect.get_client(host=args.host, port=args.port,
                                           database=args.database, username=args.user,
                                           password=args.password,
                                           autogenerate_session_id=False)

    win = []
    if args.since:
        win.append(f"TsSwNs >= toDateTime64('{args.since}', 9)")
    if args.until:
        win.append(f"TsSwNs <= toDateTime64('{args.until}', 9)")

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

    cellmap = {r[0]: (r[1], r[2]) for r in
               rows(client, "SELECT DISTINCT CellId, nRxAnt, nRxAntSrs FROM fh")[1]}

    pusch = {}
    if 'pusch' in sel:
        pusch = load_pusch(client, where)
        for s in pusch.values():
            s['nRxAnt'], s['nRxAntSrs'] = cellmap.get(s['CellId'], (0, 0))
    srs = sorted(load_srs(client, where, cellmap).values(),
                 key=lambda s: s['tai']) if 'srs' in sel else []

    hest_st = (BlobStream(client, f"SELECT toUnixTimestamp64Nano(TsTaiNs) AS tai, hestData "
                                  f"FROM hest {where} ORDER BY tai", np.float32)
               if 'hest' in sel else None)
    fh_st = (BlobStream(client, f"SELECT toUnixTimestamp64Nano(TsTaiNs) AS tai, fhData "
                                f"FROM fh {where} ORDER BY tai", np.int16)
             if 'fh' in sel else None)
    srs_iq_st = (BlobStream(client, f"SELECT toUnixTimestamp64Nano(TsTaiNs) AS tai, iqData "
                                    f"FROM srs_iq {where} ORDER BY tai", np.int16)
                 if 'srs_iq' in sel else None)
    srs_hest_st = (GroupedBlobStream(client, f"SELECT toUnixTimestamp64Nano(TsTaiNs) AS tai, rnti, "
                                             f"hestData FROM srs_hest {where} ORDER BY tai, rnti", np.int16)
                   if 'srs_hest' in sel else None)

    n_pusch = n_srs = j = 0
    with open(args.output, 'wb') as f:
        f.write(FILE_HDR.pack(MAGIC, VERSION, SHM_LAYOUT_VERSION, 0))
        for tai in sorted(pusch):
            while j < len(srs) and srs[j]['tai'] < tai:        # SRS strictly older first
                emit_srs(f, srs[j], srs_iq_st, srs_hest_st)
                n_srs += 1
                j += 1
            hb = hest_st.fetch(tai) if hest_st else b''    # absent => empty blob, h_offset/h_size zeroed
            fb = fh_st.fetch(tai) if fh_st else b''
            write_record(f, TAG_PUSCH, pack_pusch(pusch[tai], pusch[tai]['ue'], hb or b'', fb or b''))
            n_pusch += 1
        while j < len(srs):                                    # trailing SRS
            emit_srs(f, srs[j], srs_iq_st, srs_hest_st)
            n_srs += 1
            j += 1

    print(f"wrote {args.output}: {n_pusch} PUSCH, {n_srs} SRS records "
          f"[streams: {', '.join(s for s in STREAMS if s in sel)}]")
    return 0


if __name__ == '__main__':
    sys.exit(main())
