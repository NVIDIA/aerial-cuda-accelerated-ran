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

// E3-facing types + SHM offset helpers. Vendored mirror of the E3-bound subset
// of cuPHY-CP/data_lake/data_lake.hpp; lets e3agent-standalone compile
// aerial_sdk's e3_agent.cpp without pulling cuPHY/CUDA/FAPI/ClickHouse.
//   Drift check:        scripts/check_drift.py
//   Upstream candidate: cuPHY-CP/data_lake/e3_types.hpp

#ifndef E3_TYPES_HPP
#define E3_TYPES_HPP

#include <chrono>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

// ============================================================================
// [MIRRORED] from aerial_sdk/cuPHY-CP/data_lake/data_lake.hpp
// ============================================================================

// Per-UE PUSCH metrics. (per-group) fields are inherited from the UE group and
// duplicated across UEs sharing it.
struct E3UeMetrics {
	uint16_t rnti{};
	uint8_t tb_crc_fail{};
	uint32_t cb_errors{};
	float rsrp{};
	float noise_var{};
	float sinr{};
	uint16_t cb_count{};
	float rssi{};                   // per-group
	uint8_t qam_mod_order{};
	uint8_t mcs_index{};
	uint8_t mcs_table_index{};
	uint16_t rb_start{};            // per-group
	uint16_t rb_size{};             // per-group
	uint8_t start_symbol_index{};   // per-group
	uint8_t nr_of_symbols{};        // per-group
	uint32_t tb_size{};
	uint32_t pdu_len{};             // tb_size on CRC pass, 0 on CRC fail
	uint16_t target_code_rate{};
	uint8_t new_data_indicator{};
	uint8_t n_layers{};
	uint16_t layer_offset{};
	uint16_t ue_grp_idx{};
	uint32_t h_offset{};            // per-group
	uint32_t h_size{};              // per-group
	uint16_t n_subcarriers{};       // per-group
	uint8_t n_dmrs_estimates{};     // per-group
	uint16_t dmrs_symb_pos{};       // per-group
	float timing_advance{};
	float cfo_hz{};
	uint8_t harq_process_id{};
	uint8_t rv_index{};
};

struct E3BufferInfo {
	uint8_t current_fh_buffer{};
	uint8_t current_pusch_buffer{};
	uint8_t current_hest_buffer{};
	uint32_t fh_write_index{};
	uint32_t pusch_write_index{};
	uint32_t hest_write_index{};
	uint32_t hest_row_byte_offset{};

	uint16_t sfn{};
	uint16_t slot{};
	uint64_t timestamp_ns{};
	uint64_t timestamp_tai_ns{};

	uint16_t cell_id{};
	uint16_t n_rx_ant{};
	uint16_t n_rx_ant_srs{};
	uint16_t n_cells{};

	uint8_t n_bs_ants{};

	uint16_t n_ue{};
	std::vector<E3UeMetrics> ue_metrics;
};

struct fhInfo_t {
	std::string bufferName;
	std::chrono::high_resolution_clock::time_point collectStartTime;
	std::chrono::high_resolution_clock::time_point collectFullTime;
	std::vector<uint16_t> cellId;
	std::vector<uint64_t> tsSwNs;
	std::vector<uint64_t> tsTaiNs;
	std::vector<uint16_t> sfn;
	std::vector<uint16_t> slot;
	std::vector<uint16_t> nRxAnt;
	std::vector<uint16_t> nRxAntSrs;
	std::vector<uint16_t> nUes;
	int16_t* pDataAlloc;
	std::vector<int16_t*> fhData;
};
const uint32_t FH_INFO_MEMBER_COUNT = 9;

template<typename F>
void forEachFhInfoMember(fhInfo_t* info, F&& func) {
	func(info->cellId);
	func(info->tsSwNs);
	func(info->tsTaiNs);
	func(info->sfn);
	func(info->slot);
	func(info->nRxAnt);
	func(info->nRxAntSrs);
	func(info->nUes);
}

inline void clearFhInfo(fhInfo_t* info) {
	forEachFhInfoMember(info, [](auto& vec) { vec.clear(); });
}

// Binary-compatible with CUDA's cuFloatComplex (float2 __align__(8)).
struct alignas(8) hestDataType {
	float x;
	float y;
};

// PUSCH H-estimate buffer — one row per slot, all UE-groups concatenated.
// Per-UE slicing uses fapi.hOffset / fapi.hSize via JOIN on (sfn, slot).
struct hestInfo_t {
	std::string bufferName;
	std::chrono::high_resolution_clock::time_point collectStartTime;
	std::chrono::high_resolution_clock::time_point collectFullTime;
	std::vector<uint16_t> cellId;
	std::vector<uint64_t> tsSwNs;
	std::vector<uint64_t> tsTaiNs;
	std::vector<uint16_t> sfn;
	std::vector<uint16_t> slot;
	std::vector<uint32_t> hestSize;
	std::vector<hestDataType*> hestData;
	hestDataType* pDataAlloc;
	size_t writeOffsetBytes{};
};
const uint32_t HEST_INFO_MEMBER_COUNT = 7;

template<typename F>
void forEachHestInfoMember(hestInfo_t* info, F&& func) {
	func(info->cellId);
	func(info->tsSwNs);
	func(info->tsTaiNs);
	func(info->sfn);
	func(info->slot);
	func(info->hestSize);
}

inline void clearHestInfo(hestInfo_t* info) {
	forEachHestInfoMember(info, [](auto& vec) { vec.clear(); });
}

struct srsIqInfo_t {
	std::string bufferName;
	std::chrono::high_resolution_clock::time_point collectStartTime;
	std::chrono::high_resolution_clock::time_point collectFullTime;
	std::vector<uint16_t> cellId;
	std::vector<uint64_t> tsSwNs;
	std::vector<uint64_t> tsTaiNs;
	std::vector<uint16_t> sfn;
	std::vector<uint16_t> slot;
	std::vector<uint16_t> nRxAntSrs;
	std::vector<uint16_t> nSrsUes;
	int16_t* pDataAlloc;
	std::vector<int16_t*> iqData;
	size_t writeOffsetBytes{};
};
const uint32_t SRS_IQ_INFO_MEMBER_COUNT = 8;

template<typename F>
void forEachSrsIqInfoMember(srsIqInfo_t* info, F&& func) {
	func(info->cellId);
	func(info->tsSwNs);
	func(info->tsTaiNs);
	func(info->sfn);
	func(info->slot);
	func(info->nRxAntSrs);
	func(info->nSrsUes);
}

inline void clearSrsIqInfo(srsIqInfo_t* info) {
	forEachSrsIqInfoMember(info, [](auto& vec) { vec.clear(); });
}

// SRS H-estimate — per-UE (each UE has its own SRS resource).
struct srsHestInfo_t {
	std::string bufferName;
	std::chrono::high_resolution_clock::time_point collectStartTime;
	std::chrono::high_resolution_clock::time_point collectFullTime;
	std::vector<uint16_t> cellId;
	std::vector<uint64_t> tsSwNs;
	std::vector<uint64_t> tsTaiNs;
	std::vector<uint16_t> sfn;
	std::vector<uint16_t> slot;
	std::vector<uint16_t> rnti;
	std::vector<uint32_t> hestSize;
	int16_t* pDataAlloc;
	std::vector<int16_t*> hestData;
	size_t writeOffsetBytes{};
};
const uint32_t SRS_HEST_INFO_MEMBER_COUNT = 8;

template<typename F>
void forEachSrsHestInfoMember(srsHestInfo_t* info, F&& func) {
	func(info->cellId);
	func(info->tsSwNs);
	func(info->tsTaiNs);
	func(info->sfn);
	func(info->slot);
	func(info->rnti);
	func(info->hestSize);
}

inline void clearSrsHestInfo(srsHestInfo_t* info) {
	forEachSrsHestInfoMember(info, [](auto& vec) { vec.clear(); });
}

struct srsInfo_t {
	std::string bufferName;
	std::chrono::high_resolution_clock::time_point collectStartTime;
	std::chrono::high_resolution_clock::time_point collectFullTime;
	std::vector<uint16_t> cellId;
	std::vector<uint64_t> tsSwNs;
	std::vector<uint64_t> tsTaiNs;
	std::vector<uint16_t> sfn;
	std::vector<uint16_t> slot;
	std::vector<uint16_t> rnti;
	// Measurements (cuphySrsReport_t)
	std::vector<float> widebandSnr;
	std::vector<float> signalEnergy;
	std::vector<float> noiseEnergy;
	std::vector<float> toaUs;
	std::vector<uint8_t> hdAntFlag;
	std::vector<float> scCorrRe;
	std::vector<float> scCorrIm;
	std::vector<float> csCorrRatioDb;
	// Config (cuphyUeSrsPrm_t, E3-exposed subset)
	std::vector<uint8_t> nAntPorts;
	std::vector<uint8_t> nSyms;
	std::vector<uint8_t> nRepetitions;
	std::vector<uint8_t> combSize;
	std::vector<uint8_t> combOffset;
	std::vector<uint8_t> startSym;
	std::vector<uint8_t> cyclicShift;
	std::vector<uint8_t> frequencyPosition;
	std::vector<uint16_t> frequencyShift;
	std::vector<uint8_t> frequencyHopping;
	std::vector<uint8_t> resourceType;
	std::vector<uint16_t> tSrs;
	std::vector<uint16_t> tOffset;
	std::vector<uint32_t> usage;
	std::vector<uint16_t> nValidPrg;
	std::vector<uint16_t> prgSize;
	// Replay-only (Data Lake, not E3)
	std::vector<uint16_t> sequenceId;
	std::vector<uint8_t> configIdx;
	std::vector<uint8_t> bandwidthIdx;
	std::vector<uint8_t> groupOrSequenceHopping;
	// Grid dims (decode srs_hest blob via JOIN on sfn+slot+rnti)
	std::vector<uint16_t> nPrbGrps;
	std::vector<uint16_t> prbGrpSize;
	// Cell-level SRS config
	std::vector<uint16_t> nCells;
	std::vector<uint8_t> srsCellStartSym;
	std::vector<uint8_t> srsCellNSrsSym;
	// RbSNR SHM ring (pointer + per-row sizes)
	float* pRbSnrDataAlloc;
	std::vector<float*> rbSnrData;
	std::vector<uint32_t> rbSnrSize;
	size_t writeOffsetBytes{};
};
const uint32_t SRS_INFO_MEMBER_COUNT = 41; // 40 scalars + rbSnrData

template<typename F>
void forEachSrsInfoMember(srsInfo_t* info, F&& func) {
	func(info->cellId);
	func(info->tsSwNs);
	func(info->tsTaiNs);
	func(info->sfn);
	func(info->slot);
	func(info->rnti);
	func(info->widebandSnr);
	func(info->signalEnergy);
	func(info->noiseEnergy);
	func(info->toaUs);
	func(info->hdAntFlag);
	func(info->scCorrRe);
	func(info->scCorrIm);
	func(info->csCorrRatioDb);
	func(info->nAntPorts);
	func(info->nSyms);
	func(info->nRepetitions);
	func(info->combSize);
	func(info->combOffset);
	func(info->startSym);
	func(info->cyclicShift);
	func(info->frequencyPosition);
	func(info->frequencyShift);
	func(info->frequencyHopping);
	func(info->resourceType);
	func(info->tSrs);
	func(info->tOffset);
	func(info->usage);
	func(info->nValidPrg);
	func(info->prgSize);
	func(info->sequenceId);
	func(info->configIdx);
	func(info->bandwidthIdx);
	func(info->groupOrSequenceHopping);
	func(info->nPrbGrps);
	func(info->prbGrpSize);
	func(info->rbSnrSize);
	func(info->nCells);
	func(info->srsCellStartSym);
	func(info->srsCellNSrsSym);
}

inline void clearSrsInfo(srsInfo_t* info) {
	forEachSrsInfoMember(info, [](auto& vec) { vec.clear(); });
}

struct E3SrsUeMetrics {
	uint16_t rnti{};
	float wideband_snr{};
	float signal_energy{};
	float noise_energy{};
	float toa_us{};
	uint8_t hd_ant_flag{};
	float sc_corr_re{};
	float sc_corr_im{};
	float cs_corr_ratio_db{};
	uint8_t n_ant_ports{};
	uint8_t n_syms{};
	uint8_t n_repetitions{};
	uint8_t comb_size{};
	uint8_t comb_offset{};
	uint8_t start_sym{};
	uint8_t cyclic_shift{};
	uint8_t frequency_position{};
	uint16_t frequency_shift{};
	uint8_t frequency_hopping{};
	uint8_t resource_type{};
	uint16_t t_srs{};
	uint16_t t_offset{};
	uint32_t usage{};
	uint16_t n_valid_prg{};
	uint16_t prg_size{};
	uint16_t n_prb_grps{};
	uint32_t srs_hest_offset{};
	uint32_t srs_hest_size{};
	uint32_t srs_rb_snr_offset{};
	uint32_t srs_rb_snr_size{};
};

struct E3SrsBufferInfo {
	uint8_t current_srs_iq_buffer{};
	uint8_t current_srs_hest_buffer{};
	uint8_t current_srs_rb_snr_buffer{};
	uint32_t srs_iq_write_index{};
	uint32_t srs_hest_write_index{};
	uint32_t srs_rb_snr_write_index{};
	uint32_t srs_iq_row_byte_offset{};

	uint16_t sfn{};
	uint16_t slot{};
	uint64_t timestamp_ns{};
	uint64_t timestamp_tai_ns{};
	uint16_t cell_id{};
	uint16_t n_cells{};
	uint16_t n_rx_ant_srs{};
	uint8_t srs_cell_start_sym{};
	uint8_t srs_cell_n_srs_sym{};
	uint16_t n_srs_ue{};

	std::vector<E3SrsUeMetrics> ue_metrics;
};

// ============================================================================
// [STANDALONE] Additions, not yet upstreamed.
// ============================================================================

// Bumped on any breaking change to SHM layout or [MIRRORED] structs above.
// Mirrors the literal in aerial_sdk/.../e3_agent.cpp ('header->version = ...').
namespace e3 {
constexpr uint32_t SHM_LAYOUT_VERSION = 0x010100; // v1.1.0

// [MIRRORED] fixed per-row SHM strides (row counts are config-driven).
namespace shm {
constexpr uint32_t nPrbs                  = 273 * 12 * 14 * 4;
constexpr uint32_t numFhSamples           = nPrbs * 2;            // 2 for I & Q
constexpr uint32_t maxPuschPduSize        = 160000;
constexpr uint32_t maxHestSamplesPerRow   = 273 * 12 * 4 * 4 * 4;
constexpr uint32_t maxSrsIqSamplesPerRow  = 273 * 12 * 6 * 4 * 2;
constexpr uint32_t maxSrsHestBytesPerRow  = 273 * 4 * 4 * 4;
constexpr uint32_t maxSrsRbSnrBytesPerRow = 273 * sizeof(float);
} // namespace shm
} // namespace e3

// SHM row-pointer + offset advance. Single source of truth for the
// `pData[idx] = pBase + writeOffset/sizeof(T); memcpy; offset += bytes` pattern.
// Pointer is always set; copy + advance gated by (src && bytes > 0).
inline void hest_advance(hestInfo_t* h, size_t idx, const void* src, uint32_t bytes) {
	h->hestData[idx] = h->pDataAlloc + (h->writeOffsetBytes / sizeof(hestDataType));
	if (src && bytes > 0) {
		std::memcpy(h->hestData[idx], src, bytes);
		h->writeOffsetBytes += bytes;
	}
}

inline void srs_iq_advance(srsIqInfo_t* s, size_t idx, const void* src, uint32_t bytes) {
	s->iqData[idx] = s->pDataAlloc + (s->writeOffsetBytes / sizeof(int16_t));
	if (src && bytes > 0) {
		std::memcpy(s->iqData[idx], src, bytes);
		s->writeOffsetBytes += bytes;
	}
}

inline void srs_hest_advance(srsHestInfo_t* s, size_t idx, const void* src, uint32_t bytes) {
	s->hestData[idx] = s->pDataAlloc + (s->writeOffsetBytes / sizeof(int16_t));
	if (src && bytes > 0) {
		std::memcpy(s->hestData[idx], src, bytes);
		s->writeOffsetBytes += bytes;
	}
}

inline void srs_rb_snr_advance(srsInfo_t* s, size_t idx, const void* src, uint32_t bytes) {
	s->rbSnrData[idx] = s->pRbSnrDataAlloc + (s->writeOffsetBytes / sizeof(float));
	if (src && bytes > 0) {
		std::memcpy(s->rbSnrData[idx], src, bytes);
		s->writeOffsetBytes += bytes;
	}
}

// Per-buffer ring cursor: accumulates rows, ping-pongs at capacity (Data Lake model).
struct Accum {
	uint8_t half = 0;
	uint32_t row = 0;
	void advance(uint32_t cap) { if (++row >= cap) { half ^= 1; row = 0; } }
};

#endif // E3_TYPES_HPP
