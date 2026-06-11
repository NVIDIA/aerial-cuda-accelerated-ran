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

// Buffer information for E3
struct E3BufferInfo {
	uint8_t current_fh_buffer{};
	uint8_t current_pusch_buffer{};
	uint8_t current_hest_buffer{};
	uint32_t fh_write_index{};
	uint32_t pusch_write_index{};
	uint32_t hest_write_index{};
	uint32_t hest_data_size{};  // Actual size of written H estimates data (in elements)
	uint16_t sfn{};
	uint16_t slot{};
	uint64_t timestamp_ns{};

	// IQ metadata
	uint16_t cell_id{};
	uint16_t n_rx_ant{};
	uint16_t n_rx_ant_srs{};
	uint16_t n_cells{};
	uint8_t n_ue{};  // Number of UEs with data in this buffer (0 = no UE data, 1 = first UE only)

	// H estimates metadata
	uint8_t n_bs_ants{};
	uint8_t n_layers{};
	uint16_t n_subcarriers{};
	uint8_t n_dmrs_estimates{};
	uint16_t rb_size{};
	uint16_t dmrs_symb_pos{};

	// PUSCH metadata
	uint64_t rnti{};
	uint8_t tb_crc_fail{};
	uint32_t cb_errors{};
	float rsrp{};
	float cqi{};
	uint16_t cb_count{};
	float rssi{};
	uint8_t qam_mod_order{};
	uint8_t mcs_index{};
	uint8_t mcs_table_index{};
	uint16_t rb_start{};
	uint8_t start_symbol_index{};
	uint8_t nr_of_symbols{};
};

namespace ch = clickhouse;

typedef int16_t fhDataType;
typedef ch::ColumnInt16 chFhDataType;

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
	std::vector<float> cqi;
	std::vector<uint8_t> tbCrcFail;
	std::vector<float> timingAdvance;
	std::vector<uint8_t> cbErrors;
	std::vector<float> rsrp;
	uint8_t* pDataAlloc;
	std::vector<uint8_t*> pPduData;
	std::shared_ptr<ch::ColumnUInt64> pduOffsetsColumn;
};
#define PUSCH_INFO_MEMBER_COUNT 43  // tsSwNs, tsTaiNs, sfn, slot, nUes, phyCellId, rnti, mcsIndex, rssi, pduLen, pduBitmap,
								    // bwpSize, bwpStart, subcarrierSpacing, cyclicPrefix, targetCodeRate, qamModOrder,
								    // mcsTable, transformPrecoding, dataScramblingId, nrOfLayers, ulDmrsSymbPos,
								    // dmrsConfigType, ulDmrsScramblingId, puschIdentity, scid, numDmrsCdmGrpsNoData,
								    // dmrsPorts, resourceAlloc, rbStart, rbSize, vrbToPrbMapping, frequencyHopping,
								    // txDirectCurrentLocation, uplinkFrequencyShift7p5khz, startSymbolIndex, nrOfSymbols,
								    // rvIndex, harqProcessId, newDataIndicator, tbSize, numCb, cqi, tbCrcFail,
								    // timingAdvance, cbErrors, rsrp, pduData

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
	func(info->cqi);
	func(info->tbCrcFail);
	func(info->timingAdvance);
	func(info->cbErrors);
	func(info->rsrp);
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
	fhDataType* pDataAlloc;
	std::vector<fhDataType*> fhData;
};
#define FH_INFO_MEMBER_COUNT 8  // cellId, tsSwNs, tsTaiNs, sfn, slot, nRxAnt, nRxAntSrs, nUes, fhData

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

// Typedef for H matrix estimates information vectors  
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
};
const uint32_t HEST_INFO_MEMBER_COUNT = 6; // cellId, tsSwNs, tsTaiNs, sfn, slot, hestSize, hestData

template<typename F>
void forEachHestInfoMember(hestInfo_t* info, F&& func) {
	func(info->cellId);
	func(info->tsSwNs);
	func(info->tsTaiNs);
	func(info->sfn);
	func(info->slot);
	func(info->hestSize);
	func(info->hestData);
}

inline void clearHestInfo(hestInfo_t* info) {
	forEachHestInfoMember(info, [](auto& vec) { vec.clear(); });
}

class DataLake {
	public:
	DataLake(
		const bool enableDbInsert = true,
		const int numSamples = 1000,
		const std::string dbAddress = "localhost",
		const std::string dbEngine = "Memory",
		const std::vector<std::string> datalakeDataTypes = {"fh", "pusch", "hest"},
		const bool storeFailedPdu = false,
		const int numRowsToInsertFh = 120, // Use 2*60 for the same reason as the Babylonians
		const int numRowsToInsertPusch = 400, // Try to not have these have a common multiple
		const int numRowsToInsertHest = 200, // H estimates buffer size
		const bool e3AgentEnabled = false, // E3 Agent runtime enable flag
		const uint16_t e3RepPort = 5555, // E3 reply port
		const uint16_t e3PubPort = 5556, // E3 publisher port
		const uint16_t e3SubPort = 5557, // E3 subscriber port
		const bool dropTables = false
    ):
		numSamples_(numSamples),
		numRowsToInsertHest(numRowsToInsertHest),
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
		totalFhBytes(numFhSamples*sizeof(fhDataType)*numRowsToInsertFh)
	{
		totalHestBytes = maxHestSamplesPerRow * sizeof(hestDataType) * numRowsToInsertHest;
		
		// Validate data types
		static const std::set<std::string> VALID_TYPES = {"fh", "pusch", "hest"};
		for (const auto& type : datalakeDataTypes) {
			if (VALID_TYPES.find(type) == VALID_TYPES.end()) {
				NVLOGE_FMT(TAG_DATALAKE, AERIAL_CONFIG_EVENT, 
					"Invalid datalake_data_type: '{}' (valid: fh, pusch, hest)", type);
				throw std::invalid_argument("Invalid datalake_data_type configuration");
			}
		}
		
		// Pre-compute enablement flags for DB insertions
		fhDbEnabled = enableDbInsert && std::find(datalakeDataTypes.begin(), datalakeDataTypes.end(), "fh") != datalakeDataTypes.end();
		puschDbEnabled = enableDbInsert && std::find(datalakeDataTypes.begin(), datalakeDataTypes.end(), "pusch") != datalakeDataTypes.end();
		hestDbEnabled = enableDbInsert && std::find(datalakeDataTypes.begin(), datalakeDataTypes.end(), "hest") != datalakeDataTypes.end();
		
		if (enableDbInsert) {
			NVLOGC_FMT(TAG_DATALAKE, "Database insertion enabled - fh:{} pusch:{} hest:{}", 
				fhDbEnabled, puschDbEnabled, hestDbEnabled);
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

	// Dump SRS-derived dynamic DL beamforming weights (per UE / RNTI) into the
	// dyn_bfw_weights table. bfp = per-bundle BFP block [exponent byte + packed
	// 2*bfwIqWidth-bit I/Q mantissas], nPrg bundles back-to-back; decompressed to
	// Float32 (Q15) off the RT path by a worker thread.
	void notifyBfw(uint16_t cellId, uint16_t rnti, uint16_t sfn, uint16_t slot,
		uint16_t beamId, uint8_t nGnbAnt, uint16_t nPrg, uint16_t prgSize,
		uint8_t bfwIqWidth, uint8_t nLayers, const uint8_t* bfp, uint32_t bfpLen);

	void initMem (void);
	void initThreads(uint8_t numThreads);
	void dbInit (std::string host, std::string engine, bool dropTables);
	void notify(uint32_t nCrc, 
		const slot_command_api::slot_indication* slot,
		const slot_command_api::pusch_params* params,
		::cuphyPuschDataOut_t const* out, ::cuphyPuschStatPrms_t const* puschStatPrms);

	void insertPusch(puschInfo_t* puschInfo);
	void insertFh(fhInfo_t* fhInfo);
	void insertHest(hestInfo_t* hestInfo);


	void collectSlot(void);
	void doInserts(void);
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
		ch::Client *bfwClient = nullptr;      // dyn_bfw_weights inserts
		std::mutex  bfwClientMutex;           // ch::Client is not concurrency-safe
		friend class E3Agent;
		
	private:
		int numSamples_;
		bool debug = false;
		bool flushColumns = false;
		const bool dropTables = false;
		const int numThreads = 4;
		const bool enableDbInsert = true;
		const bool storeFailedPdu = false;
		const std::vector<std::string> datalakeDataTypes;
		bool fhDbEnabled = true;
		bool puschDbEnabled = true;
		bool hestDbEnabled = true;
		const int numRowsToInsertFh;
		const int numRowsToInsertPusch;
		const int numRowsToInsertHest;

		const uint32_t nPrbs = 273*12*14*4;
		const uint32_t numFhSamples = nPrbs*2; // 2 for I & Q
		const uint32_t totalFhBytes;

		// Max H estimates samples per row (worst case for ONE UE group: max PRBs, max antennas, max layers, max DMRS)
		// Physical memory layout: [NH_DMRS_ESTIMATES][N_SUBCARRIERS][N_BS_ANTS][N_LAYERS] (row-major)
		// Dimensions: NH=4, NF=273*12=3276, N_ANT=4, N_LAYER=1 → 4*3276*4*1 = 52,416 complex samples (~410 KB/row)
		const uint32_t maxHestSamplesPerRow = 273 * 12 * 4 * 4 * 1;
		uint32_t totalHestBytes;

		std::string dbAddress;
		std::string dbEngine;
		std::chrono::high_resolution_clock::time_point notifyTime;
		uint32_t nCrc_;
		const slot_command_api::slot_indication* slot_;
		const slot_command_api::pusch_params* params_;
		const ::cuphyPuschDataOut_t * out_;
		const ::cuphyPuschStatPrms_t * puschStatPrms_;
		std::vector<std::tuple<uint16_t,uint32_t>> ueSampCnt;

		static std::shared_ptr<chFhDataType> fh_data_column;
		static std::shared_ptr<ch::ColumnUInt64> fh_offsets_column;

		static std::shared_ptr<ch::ColumnUInt8> pdu_data_column;
		static std::shared_ptr<ch::ColumnUInt64> pdu_offsets_column;

		static std::shared_ptr<ch::ColumnFloat32> hest_data_column;
		static std::shared_ptr<ch::ColumnUInt64> hest_offsets_column;
		static std::atomic<bool> insertFhWorking;
		static std::atomic<bool> insertPuschWorking;

		// Thread pool for database writes
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
		
		// E3 buffer tracking (accessed by DataLake::collect and E3Agent)
		E3BufferInfo e3_buffer_info;
		std::mutex e3_buffer_mutex;
		
		// E3 Agent instance (nullptr when disabled)
		std::unique_ptr<E3Agent> e3_agent;

};


void* waitForLakeData(DataLake* dl);
static bool dataLakeWorkReady=false;
static bool dataLakeWorking=false;
#endif
