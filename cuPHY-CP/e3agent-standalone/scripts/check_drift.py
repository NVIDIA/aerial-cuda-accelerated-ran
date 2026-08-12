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

"""Guard the vendored e3_types.hpp against drift from its canonical source.

The standalone harness vendors an E3-only subset of aerial_sdk's data_lake.hpp.
That duplication is intentional (keeps cuPHY/CUDA/FAPI/ClickHouse out of the
build) but rots silently: a struct field added/reordered/retyped upstream would
make the harness write SHM at wrong offsets with no error. This compares the two
and fails on any layout-affecting divergence.

Checks:
  - 9 layout structs: normalized field tokens must match.
  - hestDataType: canonical must stay 'typedef cuFloatComplex hestDataType;'
    (vendored deliberately substitutes a binary-compatible struct).
  - *_INFO_MEMBER_COUNT constants must match.
  - 6 per-row SHM sizing constants: evaluated values must match (vendored
    e3::shm::* vs data_lake.hpp DataLake members).
  - SHM version: vendored SHM_LAYOUT_VERSION == e3_agent.cpp 'header->version'.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

STRUCTS = [
    "E3UeMetrics", "E3BufferInfo", "fhInfo_t", "hestInfo_t", "srsIqInfo_t",
    "srsHestInfo_t", "srsInfo_t", "E3SrsUeMetrics", "E3SrsBufferInfo",
]
MEMBER_COUNTS = [
    "FH_INFO_MEMBER_COUNT", "HEST_INFO_MEMBER_COUNT", "SRS_IQ_INFO_MEMBER_COUNT",
    "SRS_HEST_INFO_MEMBER_COUNT", "SRS_INFO_MEMBER_COUNT",
]
# Per-row SHM sizing constants, compared by evaluated value (nPrbs resolved first).
SIZING = [
    "numFhSamples", "maxPuschPduSize", "maxHestSamplesPerRow",
    "maxSrsIqSamplesPerRow", "maxSrsHestBytesPerRow", "maxSrsRbSnrBytesPerRow",
]
SIZEOF = {"float": 4, "double": 8, "int16_t": 2, "uint32_t": 4, "hestDataType": 8}
HEST_TYPEDEF = "typedef cuFloatComplex hestDataType;"


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//[^\n]*", "", text)
    return text


def struct_body(text: str, name: str) -> str | None:
    """Return the brace-balanced body of `struct [alignas(N)] name { ... }`.

    Brace-aware so default-init braces ('uint16_t rnti{};') don't terminate early.
    """
    m = re.search(r"struct\s+(?:alignas\(\d+\)\s+)?" + re.escape(name) + r"\b", text)
    if not m:
        return None
    i = text.find("{", m.end())
    if i < 0:
        return None
    depth, j = 0, i
    while j < len(text):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j]
        j += 1
    return None


def normalize(body: str) -> str:
    return re.sub(r"\s+", " ", body).strip()


def find_count(text: str, name: str) -> str | None:
    m = re.search(r"\b" + re.escape(name) + r"\s*=\s*(\d+)\s*;", text)
    return m.group(1) if m else None


def find_hex(text: str, pattern: str) -> str | None:
    m = re.search(pattern, text)
    return m.group(1).lower() if m else None


def find_rhs(text: str, name: str) -> str | None:
    """Return the initializer expression of `... name = <expr>;` (name as LHS)."""
    m = re.search(r"\b" + re.escape(name) + r"\s*=\s*([^;{]+);", text)
    return m.group(1).strip() if m else None


def eval_const(text: str, name: str, env: dict[str, int]) -> int | None:
    """Evaluate a C++ integer const expression for `name` found in `text`.

    Substitutes sizeof(T) and known vars (env), then evaluates the remaining
    integer arithmetic. Returns int, or None if not found/unsafe.
    """
    expr = find_rhs(text, name)
    if expr is None:
        return None
    expr = re.sub(r"sizeof\(\s*(\w+)\s*\)",
                  lambda m: str(SIZEOF.get(m.group(1), "?")), expr)
    for var, val in env.items():
        expr = re.sub(r"\b" + re.escape(var) + r"\b", f"({val})", expr)
    if not re.fullmatch(r"[0-9+\-*/() ]+", expr):
        return None
    try:
        return int(eval(expr, {"__builtins__": {}}, {}))
    except Exception:
        return None


def main() -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-lake-dir", type=Path, required=True,
                    help="aerial_sdk/cuPHY-CP/data_lake/ (canonical source)")
    ap.add_argument("--vendored", type=Path,
                    default=here.parent / "include" / "e3_types.hpp",
                    help="path to vendored e3_types.hpp")
    args = ap.parse_args()

    canon_hpp = args.data_lake_dir / "data_lake.hpp"
    canon_agent = args.data_lake_dir / "e3_agent.cpp"
    for p in (canon_hpp, canon_agent, args.vendored):
        if not p.is_file():
            print(f"check_drift: missing required file: {p}", file=sys.stderr)
            return 2

    canon = strip_comments(canon_hpp.read_text())
    vend = strip_comments(args.vendored.read_text())
    errors = []

    for name in STRUCTS:
        cb, vb = struct_body(canon, name), struct_body(vend, name)
        if cb is None:
            errors.append(f"struct {name}: not found in canonical data_lake.hpp")
        elif vb is None:
            errors.append(f"struct {name}: not found in vendored e3_types.hpp")
        elif normalize(cb) != normalize(vb):
            errors.append(
                f"struct {name}: field layout differs\n"
                f"  canonical: {normalize(cb)}\n"
                f"  vendored : {normalize(vb)}")

    if HEST_TYPEDEF not in re.sub(r"\s+", " ", canon):
        errors.append(
            f"hestDataType: canonical no longer declares '{HEST_TYPEDEF}'. "
            "The vendored binary-compatible struct may be invalid; re-verify layout.")

    for name in MEMBER_COUNTS:
        cv, vv = find_count(canon, name), find_count(vend, name)
        if cv is None:
            errors.append(f"{name}: not found in canonical")
        elif vv is None:
            errors.append(f"{name}: not found in vendored")
        elif cv != vv:
            errors.append(f"{name}: {cv} (canonical) != {vv} (vendored)")

    canon_nprbs = eval_const(canon, "nPrbs", {})
    vend_nprbs = eval_const(vend, "nPrbs", {})
    for name in SIZING:
        cv = eval_const(canon, name, {"nPrbs": canon_nprbs} if canon_nprbs else {})
        vv = eval_const(vend, name, {"nPrbs": vend_nprbs} if vend_nprbs else {})
        if cv is None:
            errors.append(f"sizing {name}: not found/unevaluable in canonical")
        elif vv is None:
            errors.append(f"sizing {name}: not found/unevaluable in vendored")
        elif cv != vv:
            errors.append(f"sizing {name}: {cv} (canonical) != {vv} (vendored)")

    canon_ver = find_hex(canon_agent.read_text(), r"header->version\s*=\s*(0x[0-9a-fA-F]+)")
    vend_ver = find_hex(vend, r"SHM_LAYOUT_VERSION\s*=\s*(0x[0-9a-fA-F]+)")
    if canon_ver is None:
        errors.append("SHM version: 'header->version = 0x...' not found in e3_agent.cpp")
    elif vend_ver is None:
        errors.append("SHM version: 'SHM_LAYOUT_VERSION = 0x...' not found in vendored")
    elif canon_ver != vend_ver:
        errors.append(f"SHM version: {canon_ver} (e3_agent.cpp) != {vend_ver} (vendored)")

    if errors:
        print("check_drift: vendored e3_types.hpp has DRIFTED from "
              f"{canon_hpp}\n", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        print("\nResolve by syncing include/e3_types.hpp with the canonical "
              "source (or bump SHM_LAYOUT_VERSION on an intentional change).",
              file=sys.stderr)
        return 1

    print(f"check_drift: OK — {len(STRUCTS)} structs, {len(MEMBER_COUNTS)} "
          f"counts, {len(SIZING)} sizing consts, version {vend_ver} match canonical.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
