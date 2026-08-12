/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DATA_LAKE_H
#define DATA_LAKE_H

#include <iostream>
#include <string>
#include <signal.h>
#include <pthread.h>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#include <condition_variable>
#include <atomic>
#include <functional>
#include <algorithm>
#include <clickhouse/client.h>
#include "scf_5g_fapi.h"
#include "oran.hpp"
#include "slot_command/slot_command.hpp"

#include "cuphy.h"
#include "cuphy_api.h"
#include "memtrace.h"
#include "nvlog.hpp"
#define TAG_DATALAKE (NVLOG_TAG_BASE_CUPHY_CONTROLLER + 6) // "CTL.DATA_LAKE"

#include "e3_agent.hpp"

// Per-UE PUSCH metrics, one entry per UE per slot across all UE groups.
// Fields marked (per-group) are inherited from the UE group and duplicated
// across UEs sharing the same group.
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

// Cell-level buffer info and per-UE metrics for E3 indications
struct E3BufferInfo {
	// SHM buffer tracking
	uint8_t current_fh_buffer{};
	uint8_t current_pusch_buffer{};
	uint8_t current_hest_buffer{};
	uint32_t fh_write_index{};
	uint32_t pusch_write_index{};
	uint32_t hest_write_index{};
	uint32_t hest_row_byte_offset{};

	// Slot-level timing
	uint16_t sfn{};
	uint16_t slot{};
	uint64_t timestamp_ns{};
	uint64_t timestamp_tai_ns{};

	// IQ metadata
	uint16_t cell_id{};
	uint16_t n_rx_ant{};
	uint16_t n_rx_ant_srs{};
	uint16_t n_cells{};

	// H estimates dimension (cell-level)
	uint8_t n_bs_ants{};

	// Per-UE metrics
	uint16_t n_ue{};
	std::vector<E3UeMetrics> ue_metrics;
};

namespace ch = clickhouse;

#define DL_LOG_ELAPSED_TIME
#ifdef DL_LOG_ELAPSED_TIME
	#define GET_ELAPSED_US(START) std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - START).count()
	#define GET_ELAPSED_MS(START) std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - START).count()
#else
	#define GET_ELAPSED_US(START) -1
	#define GET_ELAPSED_MS(START) -1
#endif

// Typedef for PUSCH information vectors
struct puschInfo_t {
	std::string bufferName;
	std::chrono::high_resolution_clock::time_point collectStartTime;
	std::chrono::high_resolution_clock::time_point collectFullTime;
	std::vector<uint64_t> tsSwNs;
	std::vector<uint64_t> tsTaiNs;
	std::vector<uint16_t> sfn;
	std::vector<uint16_t> slot;
	std::vector<uint16_t> nUes;
	std::vector<uint16_t> cellId;
	std::vector<uint16_t> rnti;
	std::vector<uint8_t> nBsAnts;
	std::vector<uint16_t> nCells;
	std::vector<uint8_t> mcsIndex;
	std::vector<float> rssi;
	std::vector<uint32_t> pduLen;
	std::vector<uint16_t> pduBitmap;
	std::vector<int16_t> bwpSize;
	std::vector<int16_t> bwpStart;
	std::vector<uint8_t> subcarrierSpacing;
	std::vector<uint8_t> cyclicPrefix;
	std::vector<uint16_t> targetCodeRate;
	std::vector<uint8_t> qamModOrder;
	std::vector<uint8_t> mcsTable;
	std::vector<uint8_t> transformPrecoding;
	std::vector<uint16_t> dataScramblingId;
	std::vector<uint8_t> nrOfLayers;
	std::vector<uint16_t> ulDmrsSymbPos;
	std::vector<uint8_t> dmrsConfigType;
	std::vector<uint16_t> ulDmrsScramblingId;
	std::vector<uint16_t> puschIdentity;
	std::vector<uint8_t> scid;
	std::vector<uint8_t> numDmrsCdmGrpsNoData;
	std::vector<uint16_t> dmrsPorts;
	std::vector<uint8_t> resourceAlloc;
	std::vector<uint16_t> rbStart;
	std::vector<uint16_t> rbSize;
	std::vector<int8_t> vrbToPrbMapping;
	std::vector<int8_t> frequencyHopping;
	std::vector<int16_t> txDirectCurrentLocation;
	std::vector<int8_t> uplinkFrequencyShift7p5khz;
	std::vector<uint8_t> startSymbolIndex;
	std::vector<uint8_t> nrOfSymbols;
	std::vector<uint8_t> rvIndex;
	std::vector<uint8_t> harqProcessId;
	std::vector<uint8_t> newDataIndicator;
	std::vector<uint32_t> tbSize;
	std::vector<uint16_t> numCb;
	std::vector<float> sinr;
	std::vector<float> noiseVar;
	std::vector<uint8_t> tbCrcFail;
	std::vector<float> timingAdvance;
	std::vector<float> cfoHz;
	std::vector<uint8_t> cbErrors;
	std::vector<float> rsrp;
	std::vector<uint16_t> layerOffset;
	std::vector<uint16_t> ueGrpIdx;
	std::vector<uint32_t> hOffset;
	std::vector<uint32_t> hSize;
	std::vector<uint16_t> nSubcarriers;
	std::vector<uint8_t> nDmrsEstimates;
	std::vector<uint16_t> dmrsSymbPos;
	uint8_t* pDataAlloc;
	std::vector<uint8_t*> pPduData;
	std::shared_ptr<ch::ColumnUInt64> pduOffsetsColumn;
};
const uint32_t PUSCH_INFO_MEMBER_COUNT = 59; // tsSwNs, tsTaiNs, sfn, slot, nUes, phyCellId, rnti, nBsAnts, nCells, mcsIndex, rssi, pduLen, pduBitmap,
								    // bwpSize, bwpStart, subcarrierSpacing, cyclicPrefix, targetCodeRate, qamModOrder,
								    // mcsTable, transformPrecoding, dataScramblingId, nrOfLayers, ulDmrsSymbPos,
								    // dmrsConfigType, ulDmrsScramblingId, puschIdentity, scid, numDmrsCdmGrpsNoData,
								    // dmrsPorts, resourceAlloc, rbStart, rbSize, vrbToPrbMapping, frequencyHopping,
								    // txDirectCurrentLocation, uplinkFrequencyShift7p5khz, startSymbolIndex, nrOfSymbols,
								    // rvIndex, harqProcessId, newDataIndicator, tbSize, numCb, sinr, noiseVar,
								    // tbCrcFail, timingAdvance, cfoHz, cbErrors, rsrp, layerOffset, ueGrpIdx,
								    // hOffset, hSize, nSubcarriers, nDmrsEstimates, dmrsSymbPos, pduData

// Macro to iterate over all members of puschInfo_t and call a member function, except for pduData and pduOffsetsColumn
template<typename F>
void forEachPuschInfoMember(puschInfo_t* info, F&& func) {
	func(info->tsSwNs);
	func(info->tsTaiNs);
	func(info->sfn);
	func(info->slot);
	func(info->nUes);
	func(info->cellId);
	func(info->rnti);
	func(info->nBsAnts);
	func(info->nCells);
	func(info->mcsIndex);
	func(info->rssi);
	func(info->pduLen);
	func(info->pduBitmap);
	func(info->bwpSize);
	func(info->bwpStart);
	func(info->subcarrierSpacing);
	func(info->cyclicPrefix);
	func(info->targetCodeRate);
	func(info->qamModOrder);
	func(info->mcsTable);
	func(info->transformPrecoding);
	func(info->dataScramblingId);
	func(info->nrOfLayers);
	func(info->ulDmrsSymbPos);
	func(info->dmrsConfigType);
	func(info->ulDmrsScramblingId);
	func(info->puschIdentity);
	func(info->scid);
	func(info->numDmrsCdmGrpsNoData);
	func(info->dmrsPorts);
	func(info->resourceAlloc);
	func(info->rbStart);
	func(info->rbSize);
	func(info->vrbToPrbMapping);
	func(info->frequencyHopping);
	func(info->txDirectCurrentLocation);
	func(info->uplinkFrequencyShift7p5khz);
	func(info->startSymbolIndex);
	func(info->nrOfSymbols);
	func(info->rvIndex);
	func(info->harqProcessId);
	func(info->newDataIndicator);
	func(info->tbSize);
	func(info->numCb);
	func(info->sinr);
	func(info->noiseVar);
	func(info->tbCrcFail);
	func(info->timingAdvance);
	func(info->cfoHz);
	func(info->cbErrors);
	func(info->rsrp);
	func(info->layerOffset);
	func(info->ueGrpIdx);
	func(info->hOffset);
	func(info->hSize);
	func(info->nSubcarriers);
	func(info->nDmrsEstimates);
	func(info->dmrsSymbPos);
	func(info->pPduData);
}


inline void clearPuschInfo(puschInfo_t* info) {
	forEachPuschInfoMember(info, [](auto& vec) { vec.clear(); });

	// Cleared above, but needs to be initialized for the next loop
	info->pPduData.push_back(info->pDataAlloc);
	info->pduOffsetsColumn->Clear();
}


// Typedef for fronthaul information vectors
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
const uint32_t FH_INFO_MEMBER_COUNT = 9; // cellId, tsSwNs, tsTaiNs, sfn, slot, nRxAnt, nRxAntSrs, nUes, fhData

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
	// Don't do anything with pDataAlloc or fhData:
	// pDataAlloc will be overwritten, and fhData are constant
}

inline void clearFhInfo(fhInfo_t* info) {
	forEachFhInfoMember(info, [](auto& vec) { vec.clear(); });
}

// Typedef for H matrix estimates data type (complex float)
typedef cuFloatComplex hestDataType;

// PUSCH H-estimate buffer — one row per slot, all UE-groups concatenated.
// Intentionally NOT per-UE: cuPHY produces one blob per UE-group (shared PRBs/DMRS),
// so per-UE rows would duplicate the group blob for MU-MIMO UEs. Use fapi.hOffset /
// fapi.hSize to extract per-UE slices via JOIN.
// Note: SRS H-estimates (srs_hest) are per-UE since each UE has its own SRS resource.
struct hestInfo_t {
	std::string bufferName;
	std::chrono::high_resolution_clock::time_point collectStartTime;
	std::chrono::high_resolution_clock::time_point collectFullTime;
	std::vector<uint16_t> cellId;
	std::vector<uint64_t> tsSwNs;
	std::vector<uint64_t> tsTaiNs;
	std::vector<uint16_t> sfn;
	std::vector<uint16_t> slot;
	std::vector<uint32_t> hestSize;        // Size of H matrix for first UE group only
	std::vector<hestDataType*> hestData;   // Pointers to H estimates data
	hestDataType* pDataAlloc;              // Allocated memory for H estimates
	size_t writeOffsetBytes{};             // Running byte cursor into pDataAlloc
};
const uint32_t HEST_INFO_MEMBER_COUNT = 7; // cellId, tsSwNs, tsTaiNs, sfn, slot, hestSize, hestData

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

// --- SRS data structures ---

// SRS IQ ping-pong buffer — one row per cell per SRS slot.
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
	size_t writeOffsetBytes{};             // Running byte cursor into pDataAlloc
};
const uint32_t SRS_IQ_INFO_MEMBER_COUNT = 8; // cellId, tsSwNs, tsTaiNs, sfn, slot, nRxAntSrs, nSrsUes, iqData

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

// SRS H-estimate buffer — per-UE (each UE has its own SRS resource, no UE-groups).
// Unlike PUSCH hestInfo_t which is per-slot with concatenated group blobs,
// SRS channel estimates are naturally per-UE so no duplication occurs.
// Grid dimensions (nPrbGrps, prbGrpSize) live in the srs table; JOIN on sfn+slot+rnti.
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
	std::vector<uint32_t> hestSize;     // Bytes of ChEst data for this UE
	int16_t* pDataAlloc;
	std::vector<int16_t*> hestData;    // Per-row pointers into pDataAlloc (short2 = complex int16)
	size_t writeOffsetBytes{};         // Running byte cursor into pDataAlloc
};
const uint32_t SRS_HEST_INFO_MEMBER_COUNT = 8; // cellId, tsSwNs, tsTaiNs, sfn, slot, rnti, hestSize, hestData

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

// SRS scalar + RbSNR buffer — per-UE, all SRS metrics + config + rb_snr array.
// Stored in ClickHouse 'srs' table; rb_snr stored as Array(Float32).
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
	// Measurements from cuphySrsReport_t
	std::vector<float> widebandSnr;
	std::vector<float> signalEnergy;
	std::vector<float> noiseEnergy;
	std::vector<float> toaUs;
	std::vector<uint8_t> hdAntFlag;
	std::vector<float> scCorrRe;
	std::vector<float> scCorrIm;
	std::vector<float> csCorrRatioDb;
	// Config from cuphyUeSrsPrm_t (E3-exposed subset)
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
	// Replay config (Data Lake only, not E3)
	std::vector<uint16_t> sequenceId;
	std::vector<uint8_t> configIdx;
	std::vector<uint8_t> bandwidthIdx;
	std::vector<uint8_t> groupOrSequenceHopping;
	// H-estimate grid dimensions (needed to decode srs_hest blob via JOIN on sfn+slot+rnti)
	std::vector<uint16_t> nPrbGrps;
	std::vector<uint16_t> prbGrpSize;
	// Cell-level SRS config
	std::vector<uint16_t> nCells;
	std::vector<uint8_t> srsCellStartSym;
	std::vector<uint8_t> srsCellNSrsSym;
	// RbSNR raw pointer + offsets (for SHM ping-pong)
	float* pRbSnrDataAlloc;
	std::vector<float*> rbSnrData;
	std::vector<uint32_t> rbSnrSize;       // Bytes of RbSNR for this UE (nValidPrg * sizeof(float))
	size_t writeOffsetBytes{};             // Running byte cursor into pRbSnrDataAlloc
};

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
	func(info->nCells);
	func(info->srsCellStartSym);
	func(info->srsCellNSrsSym);
	func(info->rbSnrSize);
}

inline void clearSrsInfo(srsInfo_t* info) {
	forEachSrsInfoMember(info, [](auto& vec) { vec.clear(); });
}
const uint32_t SRS_INFO_MEMBER_COUNT = 41; // 40 scalar columns + rbSnrData array

// Per-UE SRS metrics for E3 indications
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
	// SHM cross-references
	uint32_t srs_hest_offset{};
	uint32_t srs_hest_size{};
	uint32_t srs_rb_snr_offset{};
	uint32_t srs_rb_snr_size{};
};

// Cell-level SRS buffer info for E3 indications
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

class DataLake {
	public:
	DataLake(
		const bool enableDbInsert = true,
		const int numSamples = 1000,
		const std::string dbAddress = "localhost",
		const std::string dbEngine = "Memory",
		const std::vector<std::string> datalakeDataTypes = {"fh", "pusch", "hest", "srs_iq", "srs", "srs_hest"},
		const bool storeFailedPdu = false,
		const int numRowsToInsertFh = 120, // Use 2*60 for the same reason as the Babylonians
		const int numRowsToInsertPusch = 400, // Try to not have these have a common multiple
		const int numRowsToInsertHest = 200, // H estimates buffer size
		const int numRowsToInsertSrsIq = 40,    // SRS IQ ring buffer rows
		const int numRowsToInsertSrs = 70,      // SRS scalar+RbSNR ring buffer rows
		const int numRowsToInsertSrsHest = 90,  // SRS Hest ring buffer rows
		const bool e3AgentEnabled = false, // E3 Agent runtime enable flag
		const uint16_t e3RepPort = 5555, // E3 reply port
		const uint16_t e3PubPort = 5556, // E3 publisher port
		const uint16_t e3SubPort = 5557, // E3 subscriber port
		const bool dropTables = false
    ):
		numSamples_(numSamples),
		numRowsToInsertHest(numRowsToInsertHest),
		numRowsToInsertSrsIq(numRowsToInsertSrsIq),
		numRowsToInsertSrs(numRowsToInsertSrs),
		numRowsToInsertSrsHest(numRowsToInsertSrsHest),
		e3AgentEnabled(e3AgentEnabled),
		e3RepPort(e3RepPort),
		e3PubPort(e3PubPort),
		e3SubPort(e3SubPort),
		dbAddress(dbAddress),
		dbEngine(dbEngine),
		numRowsToInsertFh(numRowsToInsertFh),
		numRowsToInsertPusch(numRowsToInsertPusch),
		dropTables(dropTables),
		enableDbInsert(enableDbInsert),
		storeFailedPdu(storeFailedPdu),
		datalakeDataTypes(datalakeDataTypes),
		totalFhBytes(numFhSamples*sizeof(int16_t)*numRowsToInsertFh)
	{
		totalHestBytes = maxHestSamplesPerRow * sizeof(hestDataType) * numRowsToInsertHest;
		totalSrsIqBytes = maxSrsIqSamplesPerRow * sizeof(int16_t) * numRowsToInsertSrsIq;
		totalSrsRbSnrBytes = maxSrsRbSnrBytesPerRow * numRowsToInsertSrs;
		totalSrsHestBytes = maxSrsHestBytesPerRow * numRowsToInsertSrsHest;
		
		// Validate data types
		static const std::set<std::string> VALID_TYPES = {"fh", "pusch", "hest", "srs_iq", "srs", "srs_hest"};
		for (const auto& type : datalakeDataTypes) {
			if (VALID_TYPES.find(type) == VALID_TYPES.end()) {
				NVLOGE_FMT(TAG_DATALAKE, AERIAL_CONFIG_EVENT, 
					"Invalid datalake_data_type: '{}' (valid: fh, pusch, hest, srs_iq, srs, srs_hest)", type);
				throw std::invalid_argument("Invalid datalake_data_type configuration");
			}
		}
		
		// Pre-compute enablement flags for DB insertions
		fhDbEnabled = enableDbInsert && std::find(datalakeDataTypes.begin(), datalakeDataTypes.end(), "fh") != datalakeDataTypes.end();
		puschDbEnabled = enableDbInsert && std::find(datalakeDataTypes.begin(), datalakeDataTypes.end(), "pusch") != datalakeDataTypes.end();
		hestDbEnabled = enableDbInsert && std::find(datalakeDataTypes.begin(), datalakeDataTypes.end(), "hest") != datalakeDataTypes.end();
		srsIqDbEnabled = enableDbInsert && std::find(datalakeDataTypes.begin(), datalakeDataTypes.end(), "srs_iq") != datalakeDataTypes.end();
		srsDbEnabled = enableDbInsert && std::find(datalakeDataTypes.begin(), datalakeDataTypes.end(), "srs") != datalakeDataTypes.end();
		srsHestDbEnabled = enableDbInsert && std::find(datalakeDataTypes.begin(), datalakeDataTypes.end(), "srs_hest") != datalakeDataTypes.end();
		
		if (enableDbInsert) {
			NVLOGC_FMT(TAG_DATALAKE, "Database insertion enabled - fh:{} pusch:{} hest:{} srs_iq:{} srs:{} srs_hest:{}", 
				fhDbEnabled, puschDbEnabled, hestDbEnabled, srsIqDbEnabled, srsDbEnabled, srsHestDbEnabled);
		} else {
			NVLOGC_FMT(TAG_DATALAKE, "Database insertion disabled - Data collection only for E3 Agent");
		}
		
		initMem();
		initThreads(numThreads);
		if (enableDbInsert) {
			try {
				dbInit(dbAddress,dbEngine,dropTables);
			} catch (const std::exception& e) {
				NVLOGF_FMT(TAG_DATALAKE, AERIAL_CONFIG_EVENT,
					"ClickHouse connection failed at '{}': {}. "
					"Start the ClickHouse DB container or disable Data Lake by setting datalake_db_write_enable: 0.",
					dbAddress, e.what());
			}
		}
	}
	~DataLake(void);
	
	void initMem (void);
	void initThreads(uint8_t numThreads);
	void dbInit (std::string host, std::string engine, bool dropTables);
	void notify(uint32_t nCrc, 
		const slot_command_api::slot_indication* slot,
		const slot_command_api::pusch_params* params,
		::cuphyPuschDataOut_t const* out, ::cuphyPuschStatPrms_t const* puschStatPrms);
	void notifySrs(
		const slot_command_api::slot_indication* slot,
		const slot_command_api::srs_params* params,
		::cuphySrsDataOut_t const* out, ::cuphySrsStatPrms_t const* srsStatPrms);

	void insertPusch(puschInfo_t* puschInfo);
	void insertFh(fhInfo_t* fhInfo);
	void insertHest(hestInfo_t* hestInfo);
	void insertSrsIq(srsIqInfo_t* info);
	void insertSrs(srsInfo_t* info);
	void insertSrsHest(srsHestInfo_t* info);
	
	void collectSlot(void);
	void collectSrs(void);
	void doInsertsPusch(void);
	void doInsertsSrs(void);
	void submitTask(std::function<void()> task);
	void logThreadPoolStats() const;

	// Thread pool status getters
	size_t getFreeThreadCount() const;
	size_t getActiveThreadCount() const;
	size_t getPeakActiveThreadCount() const;
	size_t getQueuedTaskCount() const;
	double getAverageTaskSubmissionTimeMs() const;
	double getAverageTaskExecutionTimeMs() const;


	protected:
		ch::Client *fhClient = nullptr;
		ch::Client *dbClient = nullptr;
		ch::Client *hestClient = nullptr;
		ch::Client *srsIqClient = nullptr;
		ch::Client *srsClient = nullptr;
		ch::Client *srsHestClient = nullptr;
		friend class E3Agent;
		
	private:
		int numSamples_;
		bool debug = false;
		bool flushColumns = false;
		const bool dropTables = false;
		const int numThreads = 7;
		const bool enableDbInsert = true;
		const bool storeFailedPdu = false;
		const std::vector<std::string> datalakeDataTypes;
		bool fhDbEnabled{};
		bool puschDbEnabled{};
		bool hestDbEnabled{};
		bool srsIqDbEnabled{};
		bool srsDbEnabled{};
		bool srsHestDbEnabled{};
		const int numRowsToInsertFh;
		const int numRowsToInsertPusch;
		const int numRowsToInsertHest;
		const int numRowsToInsertSrsIq;
		const int numRowsToInsertSrs;
		const int numRowsToInsertSrsHest;

		const uint32_t nPrbs = 273*12*14*4;
		const uint32_t numFhSamples = nPrbs*2; // 2 for I & Q
		const uint32_t totalFhBytes;

		// --- Max per-row sizing constants ---

		// PUSCH PDU worst-case transport block size: MCS table 1, MCS 27, 14 sym, 273 PRBs = 159,749 bytes.
		static constexpr uint32_t maxPuschPduSize = 160000;

		// PUSCH Hest (all UE groups concatenated; FDM-partitioned, total bounded by cell bandwidth)
		// Physical memory layout per group: [NH_DMRS][N_SUBCARRIERS][N_BS_ANTS][N_LAYERS] (row-major)
		// NH = dmrsAddlnPos + 1 (1-4), NF = nPrb*12, N_BS_ANTS = N_RX_ANT (typ. 4), N_LAYERS = per-group layer count
		// Worst case: NH=4, NF=273*12=3276, N_ANT=4, N_LAYERS=4 -> 4*3276*4*4 = 209,664 complex samples (~1.6 MB/row)
		const uint32_t maxHestSamplesPerRow = 273 * 12 * 4 * 4 * 4;

		// SRS worst-case sizing at 4T4R (nRxAntSrs=4, nAntPorts=4).
		// SRS IQ stride is per-cell: 6 SRS sym * 3276 SC * 4 ant * 2(I+Q) = 157,248 int16 samples (~307 KB/row).
		static constexpr uint32_t maxSrsIqSamplesPerRow = 273 * 12 * 6 * 4 * 2;
		// SRS Hest stride is per-UE (one row per UE per slot, see collectSrs).
		// Worst case at 4T4R: nPrbGrps(273) * nRxAntSrs(4) * nAntPorts(4) * sizeof(short2)(4) = 17,472 B/row.
		// Revisit if nRxAntSrs grows (mMIMO).
		static constexpr uint32_t maxSrsHestBytesPerRow = 273 * 4 * 4 * 4;
		// Same in int16 samples (short2 = 2 x int16 per complex element)
		static constexpr uint32_t maxSrsHestSamplesPerRow = maxSrsHestBytesPerRow / sizeof(int16_t);
		// SRS RbSNR stride is per-UE (one row per UE per slot): max nValidPrg(273) * sizeof(float) = 1,092 B/row.
		static constexpr uint32_t maxSrsRbSnrBytesPerRow = 273 * sizeof(float);
		// Same in float samples
		static constexpr uint32_t maxSrsRbSnrSamplesPerRow = maxSrsRbSnrBytesPerRow / sizeof(float);

		// --- Total pre-allocated bytes (max_per_row * num_rows) ---
		uint32_t totalHestBytes{};
		uint32_t totalSrsIqBytes{};
		uint32_t totalSrsRbSnrBytes{};
		uint32_t totalSrsHestBytes{};

		// --- PUSCH path: notify state + ClickHouse columns ---
		std::chrono::high_resolution_clock::time_point notifyTime;
		uint32_t nCrc_;
		const slot_command_api::slot_indication* slot_;
		const slot_command_api::pusch_params* params_;
		const ::cuphyPuschDataOut_t * out_;
		const ::cuphyPuschStatPrms_t * puschStatPrms_;
		std::vector<std::tuple<uint16_t,uint32_t>> ueSampCnt;

		// ClickHouse columns for FH IQ
		static std::shared_ptr<ch::ColumnInt16> fh_data_column;
		static std::shared_ptr<ch::ColumnUInt64> fh_offsets_column;
		// ClickHouse columns for PUSCH (scalars + PDU)
		static std::shared_ptr<ch::ColumnUInt8> pdu_data_column;
		static std::shared_ptr<ch::ColumnUInt64> pdu_offsets_column;
		// ClickHouse columns for PUSCH Hest
		static std::shared_ptr<ch::ColumnFloat32> hest_data_column;
		static std::shared_ptr<ch::ColumnUInt64> hest_offsets_column;
		static std::atomic<bool> insertFhWorking;
		static std::atomic<bool> insertPuschWorking;

		// --- SRS path: notify state + ClickHouse columns ---
		std::chrono::high_resolution_clock::time_point srsNotifyTime;
		const slot_command_api::slot_indication* srsSlot_{};
		const slot_command_api::srs_params* srsParams_{};
		const ::cuphySrsDataOut_t* srsOut_{};
		const ::cuphySrsStatPrms_t* srsStatPrms_{};

		// ClickHouse columns for SRS IQ
		static std::shared_ptr<ch::ColumnInt16> srs_iq_data_column;
		static std::shared_ptr<ch::ColumnUInt64> srs_iq_offsets_column;

		// ClickHouse columns for SRS (scalars + RbSNR)
		static std::shared_ptr<ch::ColumnFloat32> srs_rb_snr_data_column;
		static std::shared_ptr<ch::ColumnUInt64> srs_rb_snr_offsets_column;

		// ClickHouse columns for SRS Hest
		static std::shared_ptr<ch::ColumnInt16> srs_hest_data_column;
		static std::shared_ptr<ch::ColumnUInt64> srs_hest_offsets_column;

		static std::atomic<bool> insertSrsIqWorking;
		static std::atomic<bool> insertSrsWorking;
		static std::atomic<bool> insertSrsHestWorking;

		// --- DB infrastructure ---
		std::string dbAddress;
		std::string dbEngine;

		// Thread pool for database writes (7 = 3 PUSCH + 3 SRS + 1 spare)
		std::vector<std::thread> db_write_thread_pool;
		std::queue<std::function<void()>> task_queue;
		mutable std::mutex task_queue_mutex;
		std::condition_variable task_queue_cv;
		std::atomic<bool> stop_thread_pool{false};

		// Thread pool profiling
		std::atomic<size_t> active_threads{0};
		std::atomic<size_t> peak_active_threads{0};
		std::atomic<size_t> total_tasks_submitted{0};
		std::atomic<size_t> total_tasks_completed{0};
		std::atomic<uint64_t> total_task_submission_time_ns{0};
		std::atomic<uint64_t> total_task_execution_time_ns{0};

		// E3 Agent configuration
		uint8_t e3AgentEnabled;
		uint16_t e3RepPort;
		uint16_t e3PubPort;
		uint16_t e3SubPort;

		// E3 buffer tracking — PUSCH path
		E3BufferInfo e3_buffer_info;
		std::mutex e3_buffer_mutex;

		// E3 buffer tracking — SRS path
		E3SrsBufferInfo e3_srs_buffer_info;
		std::mutex e3_srs_buffer_mutex;

		// E3 Agent instance (nullptr when disabled)
		std::unique_ptr<E3Agent> e3_agent;

};


void* waitForLakeData(DataLake* dl);
void* waitForLakeDataSrs(DataLake* dl);
#endif
