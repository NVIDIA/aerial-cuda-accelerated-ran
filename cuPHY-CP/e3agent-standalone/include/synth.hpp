/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Synthetic telemetry source: fills E3 buffer scalars for the feeder.

#ifndef E3SA_SYNTH_HPP
#define E3SA_SYNTH_HPP

#include "e3_types.hpp"

namespace e3sa::synth {

// Fill placeholder PUSCH scalars (cell + per-UE metrics). Structural fields
// (sfn/slot/timestamp/indices) are set by the feeder.
void fillPusch(E3BufferInfo& bi);

// Fill placeholder SRS scalars (per-UE metrics).
void fillSrs(E3SrsBufferInfo& si);

// Write one FH IQ row (numFhSamples int16, fp16 bit-patterns) as a per-PRB
// amplitude ramp (ant, sym, prb, sc, IQ). scale (0..1) modulates the level.
void fillPuschIq(int16_t* dst, float scale);

// PUSCH H-est element count per row: [dmrs, prb, sc, ant, layer] complex64.
constexpr uint32_t kHestSamples = 3u * 273u * 12u * 4u * 2u;  // 78624

// Write one PUSCH H-est row (kHestSamples complex64) as a per-PRB amplitude
// ramp in [dmrs, prb, sc, ant, layer] order. scale (0..1) modulates the level.
void fillPuschHest(hestDataType* dst, float scale);

// SRS dimensions, fixed to a captured 1-UE / 4Rx setup.
constexpr uint16_t kSrsRxAnt    = 4;
constexpr uint16_t kSrsAntPorts = 2;
constexpr uint16_t kSrsPrbGrps  = 272;  // = nValidPrg at prgSize=1, full band
constexpr uint16_t kSrsSyms     = 6;    // max SRS symbols per slot
constexpr uint32_t kSrsIqSamples    = 273u * 12u * kSrsSyms * kSrsRxAnt * 2u; // 157248 int16
constexpr uint32_t kSrsHestSamples  = uint32_t(kSrsRxAnt) * kSrsPrbGrps * kSrsAntPorts * 2u;  // 4352 int16 (short2)
constexpr uint32_t kSrsRbSnrSamples = kSrsPrbGrps;                            // 272 float32

// SRS blob rows, scale (0..1) modulates the level (mirrors the PUSCH path).
void fillSrsIq(int16_t* dst, float scale);    // fp16 bit-patterns, [rx, sym, prb, sc, IQ]
void fillSrsHest(int16_t* dst, float scale);  // short2 fixed-point, [rx, prg, port] (I,Q)
void fillSrsRbSnr(float* dst, float scale);   // per-PRG SNR (dB)

} // namespace e3sa::synth

#endif // E3SA_SYNTH_HPP
