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

#include "synth.hpp"

#include <cstdint>
#include <cstring>

namespace e3sa::synth {

namespace {
constexpr int kAnt = 4, kSym = 14, kPrb = 273, kSc = 12, kDmrs = 3, kLayer = 2;
static_assert(kAnt * kSym * kPrb * kSc * 2 == e3::shm::numFhSamples, "FH IQ layout mismatch");
static_assert(uint32_t(kDmrs * kPrb * kSc * kAnt * kLayer) == kHestSamples, "H-est layout mismatch");
static_assert(kHestSamples <= e3::shm::maxHestSamplesPerRow, "H-est row exceeds SHM capacity");
static_assert(kSrsIqSamples <= e3::shm::maxSrsIqSamplesPerRow, "SRS IQ row exceeds SHM capacity");
static_assert(kSrsHestSamples * sizeof(int16_t) <= e3::shm::maxSrsHestBytesPerRow, "SRS H-est row exceeds SHM capacity");
static_assert(kSrsRbSnrSamples * sizeof(float) <= e3::shm::maxSrsRbSnrBytesPerRow, "SRS RbSNR row exceeds SHM capacity");

// float32 -> IEEE-754 half (binary16) bit-pattern, round-to-nearest-even
uint16_t f32_to_f16_bits(float value) {
	uint32_t f;
	std::memcpy(&f, &value, sizeof(f));
	const uint32_t sign = (f >> 16) & 0x8000u;
	f &= 0x7fffffffu;  // abs
	if (f >= 0x47800000u) {  // Inf/NaN/overflow
		return uint16_t(sign | (f > 0x7f800000u ? 0x7e00u : 0x7c00u));
	}
	if (f < 0x38800000u) {  // subnormal/zero half: align via magic add
		constexpr uint32_t magic = 0x3f000000u;
		float vf, mf;
		std::memcpy(&vf, &f, sizeof(vf));
		std::memcpy(&mf, &magic, sizeof(mf));
		vf += mf;
		std::memcpy(&f, &vf, sizeof(f));
		return uint16_t(sign | (f - magic));
	}
	const uint32_t mant_odd = (f >> 13) & 1u;  // round-to-even
	f += 0xc8000fffu + mant_odd;               // rebias exponent + rounding bias
	return uint16_t(sign | (f >> 13));
}
}  // namespace

void fillPusch(E3BufferInfo& bi) {
	bi.n_cells = 1;
	bi.cell_id = 1;
	bi.n_rx_ant = 4;
	bi.n_rx_ant_srs = 4;
	bi.n_bs_ants = 4;
	bi.n_ue = 1;

	E3UeMetrics m{};
	m.rnti = 45936;
	m.rsrp = -90.0f;
	m.sinr = 20.0f;
	m.rssi = -70.0f;
	m.qam_mod_order = 8;
	m.mcs_index = 21;
	m.rb_start = 0;
	m.rb_size = 273;
	m.start_symbol_index = 0;
	m.nr_of_symbols = 13;
	m.n_layers = kLayer;
	m.n_subcarriers = kPrb * kSc;
	m.n_dmrs_estimates = kDmrs;
	m.h_offset = 0;
	m.h_size = kHestSamples;
	m.tb_size = 52247;
	m.pdu_len = m.tb_size;
	bi.ue_metrics.push_back(m);
}

void fillSrs(E3SrsBufferInfo& si) {
	si.n_srs_ue = 1;
	si.n_rx_ant_srs = kSrsRxAnt;
	si.n_cells = 1;
	si.cell_id = 1;

	E3SrsUeMetrics m{};
	m.rnti = 45936;
	m.wideband_snr = 20.0f;
	m.n_ant_ports = kSrsAntPorts;
	m.n_syms = 1;
	m.n_repetitions = 1;
	m.comb_size = 2;
	m.comb_offset = 0;
	m.start_sym = 1;
	m.resource_type = 2;
	m.usage = 2;
	m.t_srs = 80;
	m.t_offset = 2;
	m.n_valid_prg = kSrsPrbGrps;
	m.prg_size = 1;
	m.n_prb_grps = kSrsPrbGrps;
	si.ue_metrics.push_back(m);  // srs_hest/rb_snr offsets+sizes set by the feeder
}

void fillPuschIq(int16_t* dst, float scale) {
	int16_t prb_bits[kPrb];
	for (int prb = 0; prb < kPrb; ++prb) {
		const uint16_t h = f32_to_f16_bits((0.1f + 0.9f * float(prb) / float(kPrb - 1)) * scale);
		std::memcpy(&prb_bits[prb], &h, sizeof(int16_t));
	}

	size_t k = 0;
	for (int ant = 0; ant < kAnt; ++ant) {
		for (int sym = 0; sym < kSym; ++sym) {
			for (int prb = 0; prb < kPrb; ++prb) {
				const int16_t b = prb_bits[prb];
				for (int sc = 0; sc < kSc; ++sc) { dst[k++] = b; dst[k++] = b; }
			}
		}
	}
}

void fillPuschHest(hestDataType* dst, float scale) {
	size_t k = 0;
	for (int d = 0; d < kDmrs; ++d) {
		for (int prb = 0; prb < kPrb; ++prb) {
			const float amp = (0.1f + 0.9f * float(prb) / float(kPrb - 1)) * scale;
			for (int sc = 0; sc < kSc; ++sc) {
				for (int ant = 0; ant < kAnt; ++ant) {
					for (int lay = 0; lay < kLayer; ++lay) { dst[k].x = amp; dst[k].y = amp; ++k; }
				}
			}
		}
	}
}

void fillSrsIq(int16_t* dst, float scale) {
	int16_t prb_bits[kPrb];
	for (int prb = 0; prb < kPrb; ++prb) {
		const uint16_t h = f32_to_f16_bits((0.1f + 0.9f * float(prb) / float(kPrb - 1)) * scale);
		std::memcpy(&prb_bits[prb], &h, sizeof(int16_t));
	}

	size_t k = 0;
	for (int rx = 0; rx < kSrsRxAnt; ++rx) {
		for (int sym = 0; sym < kSrsSyms; ++sym) {
			for (int prb = 0; prb < kPrb; ++prb) {
				const int16_t b = prb_bits[prb];
				for (int sc = 0; sc < kSc; ++sc) { dst[k++] = b; dst[k++] = b; }
			}
		}
	}
}

void fillSrsHest(int16_t* dst, float scale) {
	size_t k = 0;
	for (int rx = 0; rx < kSrsRxAnt; ++rx) {
		for (int prg = 0; prg < kSrsPrbGrps; ++prg) {
			const int16_t v = static_cast<int16_t>((0.1f + 0.9f * float(prg) / float(kSrsPrbGrps - 1)) * scale * 4096.0f);
			for (int port = 0; port < kSrsAntPorts; ++port) { dst[k++] = v; dst[k++] = v; }
		}
	}
}

void fillSrsRbSnr(float* dst, float scale) {
	for (int prg = 0; prg < kSrsPrbGrps; ++prg) {
		dst[prg] = (5.0f + 15.0f * float(prg) / float(kSrsPrbGrps - 1)) * scale;
	}
}

} // namespace e3sa::synth
