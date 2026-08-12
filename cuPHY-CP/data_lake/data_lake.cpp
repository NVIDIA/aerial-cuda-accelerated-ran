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

#include <regex>
#include <cstring>
#include "data_lake.hpp"
#include "e3_agent.hpp"

#include "cuphy.h"
#include "cuphy_api.h"
#include "memtrace.h"

static fhInfo_t fhInfo[2];
static fhInfo_t *pFh, *pInsertFh;
static puschInfo_t puschInfo[2];
static puschInfo_t *p, *pInsertPusch;
static hestInfo_t hestInfo[2];
static hestInfo_t *pHest, *pInsertHest;

static srsIqInfo_t srsIqInfo[2];
static srsIqInfo_t *pSrsIq, *pInsertSrsIq;
static srsInfo_t srsScalarInfo[2];
static srsInfo_t *pSrs, *pInsertSrs;
static srsHestInfo_t srsHestInfoBuf[2];
static srsHestInfo_t *pSrsHest, *pInsertSrsHest;

static bool dataLakeWorkReady = false;
static bool dataLakeWorking = false;
static bool dataLakeSrsWorkReady = false;
static bool dataLakeSrsWorking = false;

std::shared_ptr<ch::ColumnInt16> DataLake::fh_data_column = nullptr;
std::shared_ptr<ch::ColumnUInt64> DataLake::fh_offsets_column = nullptr;
std::shared_ptr<ch::ColumnUInt8> DataLake::pdu_data_column = nullptr;
std::shared_ptr<ch::ColumnFloat32> DataLake::hest_data_column = nullptr;
std::shared_ptr<ch::ColumnUInt64> DataLake::hest_offsets_column = nullptr;

std::shared_ptr<ch::ColumnInt16> DataLake::srs_iq_data_column = nullptr;
std::shared_ptr<ch::ColumnUInt64> DataLake::srs_iq_offsets_column = nullptr;
std::shared_ptr<ch::ColumnFloat32> DataLake::srs_rb_snr_data_column = nullptr;
std::shared_ptr<ch::ColumnUInt64> DataLake::srs_rb_snr_offsets_column = nullptr;
std::shared_ptr<ch::ColumnInt16> DataLake::srs_hest_data_column = nullptr;
std::shared_ptr<ch::ColumnUInt64> DataLake::srs_hest_offsets_column = nullptr;

std::atomic<bool> DataLake::insertSrsIqWorking{false};
std::atomic<bool> DataLake::insertSrsWorking{false};
std::atomic<bool> DataLake::insertSrsHestWorking{false};

void DataLake::initThreads(uint8_t numThreads) {
	if (numThreads == 0) {
		NVLOGF_FMT(TAG_DATALAKE, AERIAL_CONFIG_EVENT, "Invalid thread pool size: 0");
	}

	// Initialize thread pool for database writes
	db_write_thread_pool.reserve(numThreads);

	try {
		for (size_t i = 0; i < numThreads; ++i) {
			db_write_thread_pool.emplace_back([this, i]() {
				// Set thread name for debugging
				std::string thread_name = "datalake_task" + std::to_string(i);
				if( pthread_setname_np(pthread_self(), thread_name.c_str()) ) {
					NVLOGE_FMT(TAG_DATALAKE, AERIAL_CONFIG_EVENT, "Failed to set name for thread: {}", thread_name);
				}

				while (true) {
					std::function<void()> task;
					{
						std::unique_lock<std::mutex> lock(task_queue_mutex);
						task_queue_cv.wait(lock, [this] { return stop_thread_pool.load() || !task_queue.empty(); });

						if (stop_thread_pool.load() && task_queue.empty()) {
							return;
						}

						task = std::move(task_queue.front());
						task_queue.pop();
					}

					// Track active threads and update peak
					const size_t current_active = active_threads.fetch_add(1) + 1;

					// Update peak active threads using compare-and-swap loop
					size_t current_peak = peak_active_threads.load();
					while (current_active > current_peak &&
						!peak_active_threads.compare_exchange_weak(current_peak, current_active)) {
						// Loop until we successfully update the peak or find it's already higher
					}

					// Execute task and measure execution time
					auto task_start = std::chrono::high_resolution_clock::now();
					task();
					auto task_end = std::chrono::high_resolution_clock::now();

					// Update profiling metrics
					auto execution_time = std::chrono::duration_cast<std::chrono::nanoseconds>(task_end - task_start).count();
					total_task_execution_time_ns.fetch_add(execution_time);
					total_tasks_completed.fetch_add(1);
					active_threads.fetch_sub(1);
				}
		});
		}
	} catch (const std::system_error& e) {
		stop_thread_pool.store(true);
		task_queue_cv.notify_all();
		// Clean up any successfully created threads
		for (auto& thread : db_write_thread_pool) {
			if (thread.joinable()) {
				thread.join();
			}
		}
		db_write_thread_pool.clear();

		NVLOGF_FMT(TAG_DATALAKE, AERIAL_CONFIG_EVENT, "Failed to create thread pool: {}", e.what());
	}

	NVLOGI_FMT(TAG_DATALAKE,"Initialized thread pool with {} threads for database writes", numThreads);
}

template <typename InfoT>
static void initPingPong(InfoT (&arr)[2], InfoT*& pPing, InfoT*& pPong) {
	pPing = &arr[0];
	pPing->bufferName = "ping";
	pPong = &arr[1];
	pPong->bufferName = "pong";
}

void DataLake::initMem(void) {
	initPingPong(fhInfo,         pFh,      pInsertFh);
	initPingPong(puschInfo,      p,        pInsertPusch);
	initPingPong(hestInfo,       pHest,    pInsertHest);
	initPingPong(srsIqInfo,      pSrsIq,   pInsertSrsIq);
	initPingPong(srsScalarInfo,  pSrs,     pInsertSrs);
	initPingPong(srsHestInfoBuf, pSrsHest, pInsertSrsHest);

	// Not sure this actually helps.
	forEachPuschInfoMember(p, [this](auto& vec) { vec.reserve(numRowsToInsertPusch); });
	forEachPuschInfoMember(pInsertPusch, [this](auto& vec) { vec.reserve(numRowsToInsertPusch); });

	forEachHestInfoMember(pHest, [this](auto& vec) { vec.reserve(numRowsToInsertHest); });
	forEachHestInfoMember(pInsertHest, [this](auto& vec) { vec.reserve(numRowsToInsertHest); });

	forEachSrsIqInfoMember(pSrsIq, [this](auto& vec) { vec.reserve(numRowsToInsertSrsIq); });
	forEachSrsIqInfoMember(pInsertSrsIq, [this](auto& vec) { vec.reserve(numRowsToInsertSrsIq); });

	forEachSrsInfoMember(pSrs, [this](auto& vec) { vec.reserve(numRowsToInsertSrs); });
	forEachSrsInfoMember(pInsertSrs, [this](auto& vec) { vec.reserve(numRowsToInsertSrs); });

	forEachSrsHestInfoMember(pSrsHest, [this](auto& vec) { vec.reserve(numRowsToInsertSrsHest); });
	forEachSrsHestInfoMember(pInsertSrsHest, [this](auto& vec) { vec.reserve(numRowsToInsertSrsHest); });

	if (e3AgentEnabled) {
		// E3 MODE: Create E3Agent instance and use shared memory
		e3_agent = std::make_unique<E3Agent>(
			this,
			e3RepPort,
			e3PubPort,
			e3SubPort,
			numRowsToInsertFh,
			numRowsToInsertPusch,
			numRowsToInsertHest,
			numFhSamples,
			maxPuschPduSize,
			maxHestSamplesPerRow,
			numRowsToInsertSrsIq,
			numRowsToInsertSrs,
			numRowsToInsertSrsHest,
			maxSrsIqSamplesPerRow,
			maxSrsHestBytesPerRow,
			maxSrsRbSnrBytesPerRow
		);

		// Create shared memory buffers through E3Agent
		if (!e3_agent->createSharedMemoryBuffers(&pFh, &pInsertFh, &p, &pInsertPusch, &pHest, &pInsertHest,
				&pSrsIq, &pInsertSrsIq, &pSrs, &pInsertSrs, &pSrsHest, &pInsertSrsHest)) {
			NVLOGF_FMT(TAG_DATALAKE, AERIAL_CONFIG_EVENT, "Failed to create E3 Agent shared memory buffers");
		}

		// Initialize E3Agent (starts threads)
		if (!e3_agent->init()) {
			NVLOGF_FMT(TAG_DATALAKE, AERIAL_CONFIG_EVENT, "Failed to initialize E3 Agent - check ports {}/{}/{} availability and permissions",
				e3RepPort, e3PubPort, e3SubPort);
		}

		NVLOGC_FMT(TAG_DATALAKE, "E3 Agent initialized successfully on ports {}/{}/{}", e3RepPort, e3PubPort, e3SubPort);
	} else {
		// REGULAR MODE: Use heap allocation for buffers
		e3_agent = nullptr;

		pFh->pDataAlloc = new int16_t[numFhSamples*numRowsToInsertFh];
		pInsertFh->pDataAlloc = new int16_t[numFhSamples*numRowsToInsertFh];
		p->pDataAlloc = new uint8_t[maxPuschPduSize*numRowsToInsertPusch];
		pInsertPusch->pDataAlloc = new uint8_t[maxPuschPduSize*numRowsToInsertPusch];
		pHest->pDataAlloc = new hestDataType[maxHestSamplesPerRow*numRowsToInsertHest];
		pInsertHest->pDataAlloc = new hestDataType[maxHestSamplesPerRow*numRowsToInsertHest];

		pSrsIq->pDataAlloc = new int16_t[maxSrsIqSamplesPerRow*numRowsToInsertSrsIq];
		pInsertSrsIq->pDataAlloc = new int16_t[maxSrsIqSamplesPerRow*numRowsToInsertSrsIq];
		pSrs->pRbSnrDataAlloc = new float[maxSrsRbSnrSamplesPerRow*numRowsToInsertSrs];
		pInsertSrs->pRbSnrDataAlloc = new float[maxSrsRbSnrSamplesPerRow*numRowsToInsertSrs];
		pSrsHest->pDataAlloc = new int16_t[maxSrsHestSamplesPerRow*numRowsToInsertSrsHest];
		pInsertSrsHest->pDataAlloc = new int16_t[maxSrsHestSamplesPerRow*numRowsToInsertSrsHest];

		NVLOGC_FMT(TAG_DATALAKE, "DataLake initialized with regular heap allocation");
	}

	// Resize data pointers. The collect path sets each pointer via writeOffsetBytes.
	pHest->hestData.resize(numRowsToInsertHest);
	pInsertHest->hestData.resize(numRowsToInsertHest);
	pSrsIq->iqData.resize(numRowsToInsertSrsIq);
	pInsertSrsIq->iqData.resize(numRowsToInsertSrsIq);
	pSrs->rbSnrData.resize(numRowsToInsertSrs);
	pInsertSrs->rbSnrData.resize(numRowsToInsertSrs);
	pSrsHest->hestData.resize(numRowsToInsertSrsHest);
	pInsertSrsHest->hestData.resize(numRowsToInsertSrsHest);

	if (pHest->pDataAlloc && pInsertHest->pDataAlloc) {
		NVLOGI_FMT(TAG_DATALAKE,"Allocated memory for H estimates: 0x{:x}, 0x{:x}",
			(uintptr_t)pHest->pDataAlloc, (uintptr_t)pInsertHest->pDataAlloc);
	} else {
		NVLOGC_FMT(TAG_DATALAKE,"Failed to allocate memory for H estimates");
		return;
	}

	if (pFh->pDataAlloc && pInsertFh->pDataAlloc) {
		NVLOGI_FMT(TAG_DATALAKE,"Allocated memory for fhData: 0x{:x}, 0x{:x} of size {} bytes",
			(uintptr_t)pFh->pDataAlloc, (uintptr_t)pInsertFh->pDataAlloc, numFhSamples*numRowsToInsertFh*sizeof(int16_t));
	} else {
		NVLOGC_FMT(TAG_DATALAKE,"Failed to allocate memory for fhData");
		return;
	}

	// Initialize fhData vectors with the correct size
	pFh->fhData.resize(numRowsToInsertFh);
	pInsertFh->fhData.resize(numRowsToInsertFh);
	fh_offsets_column = std::make_shared<ch::ColumnUInt64>();

	for (size_t i = 0; i < numRowsToInsertFh; i++) {
		fh_offsets_column->Append((i+1) * numFhSamples);
		pFh->fhData[i] = pFh->pDataAlloc + i * numFhSamples;
		pInsertFh->fhData[i] = pInsertFh->pDataAlloc + i * numFhSamples;
	}

	// Initialize data column with proper size
	fh_data_column = std::make_shared<ch::ColumnInt16>();
	fh_data_column->Reserve(numFhSamples*numRowsToInsertFh);
	auto& local_data_vector = fh_data_column->GetWritableData();
	for (size_t i = 0; i < numFhSamples * numRowsToInsertFh; ++i) {
		fh_data_column->Append(0);
	}

	hest_offsets_column = std::make_shared<ch::ColumnUInt64>();
	// Initialize H estimates data column with proper size
	hest_data_column = std::make_shared<ch::ColumnFloat32>();
	hest_data_column->Reserve(maxHestSamplesPerRow * 2 * numRowsToInsertHest); // *2 for complex (real,imag)
	for (size_t i = 0; i < maxHestSamplesPerRow * 2 * numRowsToInsertHest; ++i) {
		hest_data_column->Append(0.0f);
	}

	srs_iq_offsets_column = std::make_shared<ch::ColumnUInt64>();
	srs_iq_data_column = std::make_shared<ch::ColumnInt16>();
	srs_iq_data_column->Reserve(maxSrsIqSamplesPerRow * numRowsToInsertSrsIq);
	for (size_t i = 0; i < maxSrsIqSamplesPerRow * static_cast<size_t>(numRowsToInsertSrsIq); ++i) {
		srs_iq_data_column->Append(0);
	}

	srs_hest_offsets_column = std::make_shared<ch::ColumnUInt64>();
	srs_hest_data_column = std::make_shared<ch::ColumnInt16>();
	srs_hest_data_column->Reserve(maxSrsHestSamplesPerRow * numRowsToInsertSrsHest);
	for (size_t i = 0; i < maxSrsHestSamplesPerRow * static_cast<size_t>(numRowsToInsertSrsHest); ++i) {
		srs_hest_data_column->Append(0);
	}

	srs_rb_snr_offsets_column = std::make_shared<ch::ColumnUInt64>();
	srs_rb_snr_data_column = std::make_shared<ch::ColumnFloat32>();
	srs_rb_snr_data_column->Reserve(maxSrsRbSnrSamplesPerRow * numRowsToInsertSrs);
	for (size_t i = 0; i < maxSrsRbSnrSamplesPerRow * static_cast<size_t>(numRowsToInsertSrs); ++i) {
		srs_rb_snr_data_column->Append(0.0f);
	}

	p->pduOffsetsColumn = std::make_shared<ch::ColumnUInt64>();
	pInsertPusch->pduOffsetsColumn = std::make_shared<ch::ColumnUInt64>();

	if (p->pDataAlloc && pInsertPusch->pDataAlloc) {
		NVLOGI_FMT(TAG_DATALAKE,"Allocated memory for pduData: 0x{:x}, 0x{:x} of size {} bytes",
			(uintptr_t)p->pDataAlloc, (uintptr_t)pInsertPusch->pDataAlloc, maxPuschPduSize*numRowsToInsertPusch*sizeof(uint8_t));
	} else {
		NVLOGC_FMT(TAG_DATALAKE,"Failed to allocate memory for pduData");
		return;
	}

	// Give this an address, we'll do the appending as the PDUs come in
	p->pPduData.push_back(p->pDataAlloc);
	NVLOGD_FMT(TAG_DATALAKE,"p->pDataAlloc: {} 0x{:x}, {} 0x{:x}", p->bufferName, (uintptr_t)p->pDataAlloc, p->pPduData.size(), (uintptr_t)p->pPduData.back());

	pInsertPusch->pPduData.push_back(pInsertPusch->pDataAlloc);
	NVLOGD_FMT(TAG_DATALAKE,"pInsertPusch->pDataAlloc: {} 0x{:x}, {} 0x{:x}", pInsertPusch->bufferName, (uintptr_t)pInsertPusch->pDataAlloc, pInsertPusch->pPduData.size(), (uintptr_t)pInsertPusch->pPduData.back());

	pdu_data_column = std::make_shared<ch::ColumnUInt8>();
	pdu_data_column->Reserve(maxPuschPduSize*numRowsToInsertPusch);
	NVLOGD_FMT(TAG_DATALAKE,"pdu_data_column: {} of {}", pdu_data_column->Size(), pdu_data_column->Capacity());

	NVLOGD_FMT(TAG_DATALAKE,"initMem done. Will save {} samples per UE to database, {} rows per FH insert, {} per PUSCH, {} per Hest, {} per SRS IQ, {} per SRS, {} per SRS Hest",
		numSamples_,numRowsToInsertFh,numRowsToInsertPusch,numRowsToInsertHest,numRowsToInsertSrsIq,numRowsToInsertSrs,numRowsToInsertSrsHest);
}

void DataLake::dbInit (std::string host, std::string engine, bool dropTables) {
	NVLOGD_FMT(TAG_DATALAKE,"{} connecting to database at {}",__func__,host);
	static bool initDone = false;
	if(false == initDone) {
		dbClient = new ch::Client (ch::ClientOptions().SetHost(host));
		fhClient = new ch::Client (ch::ClientOptions().SetHost(host));
		hestClient = new ch::Client (ch::ClientOptions().SetHost(host));
		srsIqClient = new ch::Client(ch::ClientOptions().SetHost(host));
		srsClient = new ch::Client(ch::ClientOptions().SetHost(host));
		srsHestClient = new ch::Client(ch::ClientOptions().SetHost(host));
		if (dropTables) {
			NVLOGC_FMT(TAG_DATALAKE,"Dropping tables per datalake_drop_tables");
			dbClient->Execute("DROP TABLE IF EXISTS fapi");
			fhClient->Execute("DROP TABLE IF EXISTS fh");
			hestClient->Execute("DROP TABLE IF EXISTS hest");
			srsIqClient->Execute("DROP TABLE IF EXISTS srs_iq");
			srsClient->Execute("DROP TABLE IF EXISTS srs");
			srsHestClient->Execute("DROP TABLE IF EXISTS srs_hest");
		}
		if (engine != "Memory") {
			NVLOGC_FMT(TAG_DATALAKE,"Creating tables using datalake_engine: {}",engine);
			NVLOGC_FMT(TAG_DATALAKE,"If you have changed engine, you may need to drop tables for this to take effect.");
		}

		// Create FAPI Table
		std::string createTableFapi = "CREATE TABLE IF NOT EXISTS fapi ( \
			TsTaiNs						DateTime64(9)	NOT NULL, \
			TsSwNs						DateTime64(9)	NOT NULL, \
			SFN 						UInt16	NOT NULL, \
			Slot						UInt16	NOT NULL, \
			nUEs						UInt16	NOT NULL, \
			CellId						UInt16	NOT NULL, \
			pduBitmap					UInt16	NOT NULL, \
			rnti						UInt16	NOT NULL, \
			BWPSize						Int16	NOT NULL, \
			BWPStart					Int16	NOT NULL, \
			SubcarrierSpacing			UInt8	NOT NULL, \
			CyclicPrefix				UInt8	NOT NULL, \
			targetCodeRate				UInt16	NOT NULL, \
			qamModOrder					UInt8	NOT NULL, \
			mcsIndex					UInt8	NOT NULL, \
			mcsTable					UInt8	NOT NULL, \
			TransformPrecoding			UInt8	NOT NULL, \
			dataScramblingId			UInt16	NOT NULL, \
			nrOfLayers					UInt8	NOT NULL, \
			ulDmrsSymbPos				UInt16	NOT NULL, \
			dmrsConfigType				UInt8	NOT NULL, \
			ulDmrsScramblingId			UInt16	NOT NULL, \
			puschIdentity				UInt16	NOT NULL, \
			SCID						UInt8	NOT NULL, \
			numDmrsCdmGrpsNoData		UInt8	NOT NULL, \
			dmrsPorts					UInt16	NOT NULL, \
			resourceAlloc				UInt8	NOT NULL, \
			rbBitmap					Array(UInt8) 	NOT NULL, \
			rbStart						UInt16	NOT NULL, \
			rbSize						UInt16	NOT NULL, \
			VRBtoPRBMapping				Int8	NOT NULL, \
			FrequencyHopping			Int8	NOT NULL, \
			txDirectCurrentLocation		Int16	NOT NULL, \
			uplinkFrequencyShift7p5khz	Int8	NOT NULL, \
			StartSymbolIndex			UInt8	NOT NULL, \
			NrOfSymbols					UInt8	NOT NULL, \
			rvIndex						UInt8	NOT NULL, \
			harqProcessID				UInt8	NOT NULL, \
			newDataIndicator			UInt8	NOT NULL, \
			TBSize						UInt32	NOT NULL, \
			numCb						UInt16	NOT NULL, \
			numPRGs						UInt16, \
			prgSize						UInt16, \
			digBFInterface				UInt8, \
	 		tbCrcFail					UInt8, \
			sinr						Float32, \
			noiseVar					Float32, \
			timingAdvance 				Float32, \
			rssi						Float32, \
			pduLen						UInt32, \
			pduData						Array(UInt8), \
			cbErrors					UInt8, \
			rsrp						Float32, \
			layerOffset					UInt16, \
			ueGrpIdx					UInt16, \
			hOffset						UInt32, \
			hSize						UInt32, \
			nSubcarriers				UInt16, \
			nDmrsEstimates				UInt8, \
			dmrsSymbPos					UInt16, \
			cfoHz						Float32, \
			nBsAnts						UInt8	NOT NULL, \
			nCells						UInt16	NOT NULL \
			) \
			ENGINE = " + engine + ";";

		// Create the fronthaul data table
		std::string createTableFh = "CREATE TABLE IF NOT EXISTS fh ( \
			CellId 			UInt16	NOT NULL, \
			TsTaiNs			DateTime64(9)	NOT NULL, \
			TsSwNs 			DateTime64(9)	NOT NULL, \
			SFN				UInt16	NOT NULL, \
			Slot			UInt16	NOT NULL, \
			nRxAnt 			UInt16	NOT NULL, \
			nRxAntSrs		UInt16	NOT NULL, \
			nUEs			UInt16	NOT NULL, \
			fhData 			Array(Int16) \
			) \
			ENGINE = " + engine + ";";

		// Create the H estimates data table
		std::string createTableHest = "CREATE TABLE IF NOT EXISTS hest ( \
		CellId 			UInt16	NOT NULL, \
		TsTaiNs			DateTime64(9)	NOT NULL, \
		TsSwNs 			DateTime64(9)	NOT NULL, \
		SFN				UInt16	NOT NULL, \
		Slot			UInt16	NOT NULL, \
		hestSize		UInt32	NOT NULL, \
		hestData 		Array(Float32) \
		) \
		ENGINE = " + engine + ";";

		std::string createTableSrsIq = "CREATE TABLE IF NOT EXISTS srs_iq ( \
		CellId			UInt16	NOT NULL, \
		TsTaiNs			DateTime64(9)	NOT NULL, \
		TsSwNs			DateTime64(9)	NOT NULL, \
		SFN				UInt16	NOT NULL, \
		Slot			UInt16	NOT NULL, \
		nRxAntSrs		UInt16	NOT NULL, \
		nSrsUes			UInt16	NOT NULL, \
		iqData			Array(Int16) \
		) \
		ENGINE = " + engine + ";";

		std::string createTableSrs = "CREATE TABLE IF NOT EXISTS srs ( \
		CellId			UInt16	NOT NULL, \
		TsTaiNs			DateTime64(9)	NOT NULL, \
		TsSwNs			DateTime64(9)	NOT NULL, \
		SFN				UInt16	NOT NULL, \
		Slot			UInt16	NOT NULL, \
		rnti			UInt16	NOT NULL, \
		widebandSnr		Float32, \
		signalEnergy	Float32, \
		noiseEnergy		Float32, \
		toaUs			Float32, \
		hdAntFlag		UInt8, \
		scCorrRe		Float32, \
		scCorrIm		Float32, \
		csCorrRatioDb	Float32, \
		nAntPorts		UInt8, \
		nSyms			UInt8, \
		nRepetitions	UInt8, \
		combSize		UInt8, \
		combOffset		UInt8, \
		startSym		UInt8, \
		cyclicShift		UInt8, \
		frequencyPosition	UInt8, \
		frequencyShift	UInt16, \
		frequencyHopping	UInt8, \
		resourceType	UInt8, \
		tSrs			UInt16, \
		tOffset			UInt16, \
		usage			UInt32, \
		nValidPrg		UInt16, \
		prgSize			UInt16, \
		sequenceId		UInt16, \
		configIdx		UInt8, \
		bandwidthIdx	UInt8, \
		groupOrSequenceHopping	UInt8, \
		nPrbGrps		UInt16, \
		prbGrpSize		UInt16, \
		rbSnrSize		UInt32, \
		rbSnrData		Array(Float32), \
		nCells			UInt16	NOT NULL, \
		srsCellStartSym	UInt8	NOT NULL, \
		srsCellNSrsSym	UInt8	NOT NULL \
		) \
		ENGINE = " + engine + ";";

		std::string createTableSrsHest = "CREATE TABLE IF NOT EXISTS srs_hest ( \
		CellId			UInt16	NOT NULL, \
		TsTaiNs			DateTime64(9)	NOT NULL, \
		TsSwNs			DateTime64(9)	NOT NULL, \
		SFN				UInt16	NOT NULL, \
		Slot			UInt16	NOT NULL, \
		rnti			UInt16	NOT NULL, \
		hestSize		UInt32	NOT NULL, \
		hestData		Array(Int16) \
		) \
		ENGINE = " + engine + ";";

		// Otherwise the log is terrible
		std::regex tabRegex("\t+");
		NVLOGD_FMT(TAG_DATALAKE,"Creating table fapi: {}", std::regex_replace(createTableFapi, tabRegex, " "));
		NVLOGD_FMT(TAG_DATALAKE,"Creating table fh: {}", std::regex_replace(createTableFh, tabRegex, " "));
		NVLOGD_FMT(TAG_DATALAKE,"Creating table hest: {}", std::regex_replace(createTableHest, tabRegex, " "));
		NVLOGD_FMT(TAG_DATALAKE,"Creating table srs_iq: {}", std::regex_replace(createTableSrsIq, tabRegex, " "));
		NVLOGD_FMT(TAG_DATALAKE,"Creating table srs: {}", std::regex_replace(createTableSrs, tabRegex, " "));
		NVLOGD_FMT(TAG_DATALAKE,"Creating table srs_hest: {}", std::regex_replace(createTableSrsHest, tabRegex, " "));

		dbClient->Execute(createTableFapi);
		fhClient->Execute(createTableFh);
		hestClient->Execute(createTableHest);

		srsIqClient->Execute(createTableSrsIq);
		srsClient->Execute(createTableSrs);
		srsHestClient->Execute(createTableSrsHest);

		initDone = true;
	}
	notifyTime = std::chrono::high_resolution_clock::now();
}

inline uint64_t sfn_to_tai(int sfn, int slot, uint64_t approx_tai_time_ns, int64_t gps_alpha, int64_t gps_beta, int mu)
{
	static const uint64_t TAI_TO_GPS_OFFSET_NS = (315964800ULL + 19ULL) * 1000000000ULL;
	int64_t gps_offset = ((gps_beta * 1000000000LL) / 100LL) + ((gps_alpha * 10000ULL) / 12288ULL);
	static const uint64_t FRAME_PERIOD_NS = 10000000;
	static const int SFN_MAX_PLUS1 = 1024;
	static const int slot_period_ns[] = {1000000, 500000, 250000, 125000, 62500};

	// First, figure out the base SFN
	uint64_t approx_gps_time_ns = approx_tai_time_ns - TAI_TO_GPS_OFFSET_NS;
	int64_t full_wrap_period_ns = FRAME_PERIOD_NS * SFN_MAX_PLUS1;
	int64_t half_wrap_period_adjust_ns = full_wrap_period_ns / 2 - sfn * FRAME_PERIOD_NS - slot * slot_period_ns[mu];

	uint64_t base_gps_time_ns = (approx_gps_time_ns - gps_offset + half_wrap_period_adjust_ns) / full_wrap_period_ns;
	base_gps_time_ns *= full_wrap_period_ns;
	base_gps_time_ns += gps_offset%full_wrap_period_ns;
	uint64_t base_tai_time_ns = base_gps_time_ns + TAI_TO_GPS_OFFSET_NS;

	return base_tai_time_ns + sfn * FRAME_PERIOD_NS + slot * slot_period_ns[mu];
}

void DataLake::notify(uint32_t nCrc,
	const slot_command_api::slot_indication* slot,
	const slot_command_api::pusch_params* params,
	::cuphyPuschDataOut_t const* out, ::cuphyPuschStatPrms_t const* puschStatPrms)
{
	NVLOGD_FMT(TAG_DATALAKE, "TIMESTAMP_LOG: DataLake notify (Op #2) for {:4}.{:02} entry at {}", slot->sfn_, slot->slot_, std::chrono::high_resolution_clock::now().time_since_epoch().count());
	// When we have one cell this will always be true, when we have two cells using one as a dummy this will be
	// true only when the "real" cell has PUSCH in it.
	// TODO should make this behavior configurable
	if(params->cell_grp_info.nCells == puschStatPrms->nMaxCells) {
		if (__atomic_load_n(&dataLakeWorking, __ATOMIC_RELAXED)) {
			NVLOGI_FMT(TAG_DATALAKE,"{:4}.{:02} Notify not called for collectSlot busy",slot->sfn_,slot->slot_);
			return;
		}
		nCrc_ = nCrc;
		slot_ = slot;
		params_ = params;
		out_ = out;
		puschStatPrms_ = puschStatPrms;
		notifyTime = std::chrono::high_resolution_clock::now();

		NVLOGD_FMT(TAG_DATALAKE,"{:4}.{:02} Notify called",slot->sfn_,slot->slot_);
		__atomic_store_n(&dataLakeWorkReady, true, __ATOMIC_RELAXED);
	} else {
		NVLOGI_FMT(TAG_DATALAKE,"{:4}.{:02} Notify skipped",slot->sfn_,slot->slot_);
	}
}

void DataLake::notifySrs(
	const slot_command_api::slot_indication* slot,
	const slot_command_api::srs_params* params,
	::cuphySrsDataOut_t const* out, ::cuphySrsStatPrms_t const* srsStatPrms)
{
	NVLOGD_FMT(TAG_DATALAKE, "TIMESTAMP_LOG: DataLake notifySrs for {:4}.{:02} entry at {}", slot->sfn_, slot->slot_, std::chrono::high_resolution_clock::now().time_since_epoch().count());
	// Only collect when all configured cells are present (mirrors PUSCH notify guard)
	// TODO should make this behavior configurable
	if(params->cell_grp_info.nCells == srsStatPrms->nMaxCells) {
		if (__atomic_load_n(&dataLakeSrsWorking, __ATOMIC_RELAXED)) {
			NVLOGI_FMT(TAG_DATALAKE,"{:4}.{:02} notifySrs not called for collectSrs busy",slot->sfn_,slot->slot_);
			return;
		}
		srsSlot_ = slot;
		srsParams_ = params;
		srsOut_ = out;
		srsStatPrms_ = srsStatPrms;
		srsNotifyTime = std::chrono::high_resolution_clock::now();

		NVLOGD_FMT(TAG_DATALAKE,"{:4}.{:02} notifySrs called",slot->sfn_,slot->slot_);
		__atomic_store_n(&dataLakeSrsWorkReady, true, __ATOMIC_RELAXED);
	} else {
		NVLOGI_FMT(TAG_DATALAKE,"{:4}.{:02} notifySrs skipped",slot->sfn_,slot->slot_);
	}
}

// PUSCH/FH worker thread: collectSlot + DB inserts for FH, PUSCH, Hest
void* waitForLakeData(DataLake* dl)
{
	while (1) {
		if (__atomic_load_n(&dataLakeWorkReady, __ATOMIC_RELAXED)) {
			__atomic_store_n(&dataLakeWorkReady, false, __ATOMIC_RELAXED);
			__atomic_store_n(&dataLakeWorking, true, __ATOMIC_RELAXED);
			dl->collectSlot();
			__atomic_store_n(&dataLakeWorking, false, __ATOMIC_RELAXED);
		}
		dl->doInsertsPusch();
		std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
	}
}

// SRS worker thread: collectSrs + DB inserts for SRS IQ, SRS, SRS Hest
void* waitForLakeDataSrs(DataLake* dl)
{
	while (1) {
		if (__atomic_load_n(&dataLakeSrsWorkReady, __ATOMIC_RELAXED)) {
			__atomic_store_n(&dataLakeSrsWorkReady, false, __ATOMIC_RELAXED);
			__atomic_store_n(&dataLakeSrsWorking, true, __ATOMIC_RELAXED);
			dl->collectSrs();
			__atomic_store_n(&dataLakeSrsWorking, false, __ATOMIC_RELAXED);
		}
		dl->doInsertsSrs();
		std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
	}
}

void DataLake::collectSlot(void)
{
	NVLOGD_FMT(TAG_DATALAKE, "TIMESTAMP_LOG: DataLake collectSlot (Op #3) entry at {}", std::chrono::high_resolution_clock::now().time_since_epoch().count());

	auto collectStart = std::chrono::high_resolution_clock::now();
	auto elapsedNotify = GET_ELAPSED_US(notifyTime);
	if(elapsedNotify > 270) { // If isn't called early enough, the CRCs will all be wrong
		NVLOGI_FMT(TAG_DATALAKE,"{:4}.{:02} {} us: collectSlot slow start, skip slot",slot_->sfn_,slot_->slot_,elapsedNotify);
		return;
	} else {
		NVLOGD_FMT(TAG_DATALAKE,"{:4}.{:02} {} us: collectSlot start",slot_->sfn_,slot_->slot_,elapsedNotify);
	}

	bool saveTtiInfo = false;
	int insertionUe = 0;

	uint16_t nUes = params_->cell_grp_info.nUes;
	for(uint16_t ueIdx = 0; ueIdx < nUes; ++ueIdx ) {
		auto ue = &params_->ue_info[ueIdx];
		auto ueGrp = ue->pUeGrpPrm;
		uint16_t ueRnti = params_->ue_info[ueIdx].rnti;

		auto it = std::find_if(ueSampCnt.begin(), ueSampCnt.end(),
			[&](const std::tuple<uint16_t,uint32_t>& ue ) {return std::get<0>(ue) == ueRnti;}
		);

		if(it != ueSampCnt.end()) {
			// If DB inserts are disabled or any UE has fewer than numSamples entry in the DB then every UE in the TTI will be added
			if(!enableDbInsert || std::get<1>(*it) < numSamples_) {
				saveTtiInfo = true;
				flushColumns = false;
			} else {
				if(std::get<1>(*it) == numSamples_) {
					flushColumns = true;
					NVLOGC_FMT(TAG_DATALAKE, "Stopping capture for rnti {} after reaching configured number of samples ({}).",std::get<0>(*it),numSamples_);
				}
			}
			std::get<1>(*it)++;
			insertionUe = std::distance(std::begin(ueSampCnt), it);
		} else {
			std::tuple<uint16_t,uint32_t> ue(ueRnti,1);
			ueSampCnt.emplace_back(ue);
			saveTtiInfo = true;
			insertionUe = ueSampCnt.size() -1;
		}
	}

	if(pFh->tsSwNs.size() == numRowsToInsertFh) {
		NVLOGW_FMT(TAG_DATALAKE,"{:4}.{:02} {} us: Skipping slot because {} buffer full, size: {}. Filled in {} ms",
			slot_->sfn_,slot_->slot_,GET_ELAPSED_US(notifyTime),pFh->bufferName,pFh->tsTaiNs.size(),
			std::chrono::duration_cast<std::chrono::milliseconds>(pFh->collectFullTime - pFh->collectStartTime).count());
		return;
	}

	if(pHest->tsSwNs.size() == numRowsToInsertHest) {
		NVLOGW_FMT(TAG_DATALAKE,"{:4}.{:02} {} us: Skipping slot because {} H estimates buffer full, size: {}. Filled in {} ms",
			slot_->sfn_,slot_->slot_,GET_ELAPSED_US(notifyTime),pHest->bufferName,pHest->tsTaiNs.size(),
			std::chrono::duration_cast<std::chrono::milliseconds>(pHest->collectFullTime - pHest->collectStartTime).count());
		return;
	}

	if (saveTtiInfo) {
		struct timespec ts;
		std::timespec_get(&ts, TIME_UTC);
		uint64_t ts_ns = ts.tv_sec * UINT64_C(1000000000) + ts.tv_nsec;
		uint64_t ts_tai_ns = sfn_to_tai(slot_->sfn_, slot_->slot_, ts_ns, 0, 0, 1);

		// Store current buffer info for E3 (only if E3 Agent is enabled)
		if (e3_agent) {
			std::lock_guard<std::mutex> lock(e3_buffer_mutex);
			e3_buffer_info.current_fh_buffer = (pFh == &fhInfo[0]) ? 0 : 1;
			e3_buffer_info.current_pusch_buffer = (p == &puschInfo[0]) ? 0 : 1;
			e3_buffer_info.current_hest_buffer = (pHest == &hestInfo[0]) ? 0 : 1;
			e3_buffer_info.fh_write_index = pFh->tsTaiNs.size();
			e3_buffer_info.pusch_write_index = p->tsTaiNs.size();
			e3_buffer_info.hest_write_index = pHest->tsTaiNs.size();
			e3_buffer_info.hest_row_byte_offset = static_cast<uint32_t>(pHest->writeOffsetBytes);
			e3_buffer_info.sfn = slot_->sfn_;
			e3_buffer_info.slot = slot_->slot_;
			e3_buffer_info.timestamp_ns = ts_ns;
			e3_buffer_info.timestamp_tai_ns = ts_tai_ns;

			// IQ metadata
			if (params_->cell_grp_info.nCells > 0) {
				e3_buffer_info.cell_id = puschStatPrms_->pCellStatPrms[0].phyCellId;
				e3_buffer_info.n_rx_ant = puschStatPrms_->pCellStatPrms[0].nRxAnt;
				e3_buffer_info.n_rx_ant_srs = puschStatPrms_->pCellStatPrms[0].nRxAntSrs;
			} else {
				e3_buffer_info.cell_id = 0;
				e3_buffer_info.n_rx_ant = 0;
				e3_buffer_info.n_rx_ant_srs = 0;
			}
			e3_buffer_info.n_cells = params_->cell_grp_info.nCells;

			// H estimates dimension (cell-level)
			if (params_->cell_grp_info.nUeGrps > 0) {
				e3_buffer_info.n_bs_ants = puschStatPrms_->pCellStatPrms[0].nRxAnt;
			}

			// Per-UE metrics for all UEs across all groups.
			uint16_t numUes = params_->cell_grp_info.nUes;
			e3_buffer_info.n_ue = numUes;
			e3_buffer_info.ue_metrics.clear();
			e3_buffer_info.ue_metrics.reserve(numUes);

			uint32_t hOffset = 0;
			for (uint16_t grpIdx = 0; grpIdx < params_->cell_grp_info.nUeGrps; ++grpIdx) {
				auto* grp = &params_->cell_grp_info.pUeGrpPrms[grpIdx];
				uint16_t layerOffset = 0;

				uint32_t grpHSize = (out_->pChannelEstSizes) ? out_->pChannelEstSizes[grpIdx] : 0;

				for (uint16_t i = 0; i < grp->nUes; ++i) {
					uint16_t ueIdx = grp->pUePrmIdxs[i];
					auto ue = &params_->ue_info[ueIdx];
					E3UeMetrics m{};

					m.rnti = ue->rnti;
					m.qam_mod_order = ue->qamModOrder;
					m.mcs_index = ue->mcsIndex;
					m.mcs_table_index = ue->mcsTableIndex;
					m.rb_start = grp->startPrb;
					m.rb_size = grp->nPrb;
					m.start_symbol_index = grp->puschStartSym;
					m.nr_of_symbols = grp->nPuschSym;
					m.n_layers = ue->nUeLayers;
					m.layer_offset = layerOffset;
					m.ue_grp_idx = grpIdx;
					m.tb_size = params_->ue_tb_size[ueIdx];
					m.target_code_rate = ue->targetCodeRate;
					m.new_data_indicator = ue->ndi;

					m.h_offset = hOffset;
					m.h_size = grpHSize;
					m.n_subcarriers = grp->nPrb * 12;
					m.n_dmrs_estimates = (grp->pDmrsDynPrm != nullptr) ? grp->pDmrsDynPrm->dmrsAddlnPos + 1 : 0;
					m.dmrs_symb_pos = grp->dmrsSymLocBmsk;

					m.tb_crc_fail = (out_->pTbCrcs && out_->pStartOffsetsTbCrc)	? ((out_->pTbCrcs[out_->pStartOffsetsTbCrc[ueIdx]] != 0) ? 1 : 0) : 1;
					m.pdu_len = (m.tb_crc_fail == 0) ? m.tb_size : 0;

					m.cb_count = 0;
					m.cb_errors = 0;
					if (out_->pStartOffsetsCbCrc != nullptr) {
						uint32_t cbStart = out_->pStartOffsetsCbCrc[ueIdx];
						uint32_t cbEnd = (ueIdx < numUes - 1) ?
							out_->pStartOffsetsCbCrc[ueIdx + 1] : out_->totNumCbs;
						m.cb_count = cbEnd - cbStart;
						if (out_->pCbCrcs != nullptr) {
							for (uint32_t cb = cbStart; cb < cbEnd; cb++) {
								if (out_->pCbCrcs[cb] != 0) m.cb_errors++;
							}
						}
					}

					m.rsrp = (out_->pRsrp != nullptr) ? out_->pRsrp[ueIdx] : -std::numeric_limits<float>::max();
					// Active (pre xor post) per enable_pusch_sinr; the driver NULLs the inactive domain.
					m.noise_var = (out_->pNoiseVarPreEq != nullptr) ? out_->pNoiseVarPreEq[ueIdx]
					            : (out_->pNoiseVarPostEq != nullptr) ? out_->pNoiseVarPostEq[ueIdx]
					            : -std::numeric_limits<float>::max();
					m.sinr = (out_->pSinrPreEq != nullptr) ? out_->pSinrPreEq[ueIdx]
					       : (out_->pSinrPostEq != nullptr) ? out_->pSinrPostEq[ueIdx]
					       : -std::numeric_limits<float>::max();
					m.rssi = (out_->pRssi != nullptr) ? out_->pRssi[grpIdx] : -std::numeric_limits<float>::max();
					m.timing_advance = (out_->pTaEsts != nullptr) ? out_->pTaEsts[ueIdx] : 0.0f;
					m.cfo_hz = (out_->pCfoHz != nullptr) ? out_->pCfoHz[ueIdx] : 0.0f;
					m.harq_process_id = ue->harqProcessId;
					m.rv_index = ue->rv;

					layerOffset += ue->nUeLayers;
					e3_buffer_info.ue_metrics.push_back(m);
				}
				hOffset += grpHSize;
			}
		}

		// Send E3 notification after data is collected
		if (e3_agent) {
			e3_agent->notifyDataReady();
		}

		if (p->tsTaiNs.size() == 0) {
			p->collectStartTime = std::chrono::high_resolution_clock::now();
		}
		if (pFh->tsTaiNs.size() == 0) {
			pFh->collectStartTime = std::chrono::high_resolution_clock::now();
		}

		// Pre-compute per-UE group metadata by iterating groups
		uint16_t ueLayerOffsets[MAX_N_TBS_PER_CELL_GROUP_SUPPORTED] = {};
		uint32_t ueHOffsets[MAX_N_TBS_PER_CELL_GROUP_SUPPORTED] = {};
		uint32_t ueHSizes[MAX_N_TBS_PER_CELL_GROUP_SUPPORTED] = {};
		uint16_t ueNSubcarriers[MAX_N_TBS_PER_CELL_GROUP_SUPPORTED] = {};
		uint8_t  ueNDmrsEstimates[MAX_N_TBS_PER_CELL_GROUP_SUPPORTED] = {};
		uint16_t ueDmrsSymbPos[MAX_N_TBS_PER_CELL_GROUP_SUPPORTED] = {};
		uint32_t hOff = 0;
		for (uint16_t g = 0; g < params_->cell_grp_info.nUeGrps; ++g) {
			auto* grp = &params_->cell_grp_info.pUeGrpPrms[g];
			uint32_t grpHSize = (out_->pChannelEstSizes) ? out_->pChannelEstSizes[g] : 0;
			uint16_t layOff = 0;
			for (uint16_t i = 0; i < grp->nUes; ++i) {
				uint16_t idx = grp->pUePrmIdxs[i];
				ueLayerOffsets[idx] = layOff;
				ueHOffsets[idx] = hOff;
				ueHSizes[idx] = grpHSize;
				ueNSubcarriers[idx] = grp->nPrb * 12;
				ueNDmrsEstimates[idx] = (grp->pDmrsDynPrm != nullptr) ? grp->pDmrsDynPrm->dmrsAddlnPos + 1 : 0;
				ueDmrsSymbPos[idx] = grp->dmrsSymLocBmsk;
				layOff += params_->ue_info[idx].nUeLayers;
			}
			hOff += grpHSize;
		}

		for(uint16_t ueIdx = 0; ueIdx < nUes; ++ueIdx) {
			if(ueIdx > 0) {
				std::timespec_get(&ts, TIME_UTC);
				ts_ns = ts.tv_sec * UINT64_C(1000000000) + ts.tv_nsec;
			}

			p->tsSwNs.push_back(ts_ns);
			p->tsTaiNs.push_back(ts_tai_ns);

			p->sfn.push_back(slot_->sfn_);
			p->slot.push_back(slot_->slot_);
			p->nUes.push_back(nUes);

			auto ue = &params_->ue_info[ueIdx];
			auto ueGrp = ue->pUeGrpPrm;
			uint16_t ueRnti = params_->ue_info[ueIdx].rnti;

			uint8_t * tb_start = out_->pTbPayloads+out_->pStartOffsetsTbPayload[ueIdx];
			uint32_t tb_size = params_->ue_tb_size[ueIdx];
			std::vector<uint8_t> data_buf (tb_start,tb_start+tb_size);

			auto cellIdx = ue->pUeGrpPrm->pCellPrm->cellPrmStatIdx;
			uint16_t cellId = puschStatPrms_->pCellStatPrms[cellIdx].phyCellId;

			// Store all PUSCH parameters
			p->cellId.push_back(cellId);
			p->rnti.push_back(ueRnti);
			// Cell-level config
			p->nBsAnts.push_back(puschStatPrms_->pCellStatPrms[0].nRxAnt);
			p->nCells.push_back(params_->cell_grp_info.nCells);
			p->pduBitmap.push_back(ue->pduBitmap);

			// BWP information
			p->bwpSize.push_back(-1);
			p->bwpStart.push_back(-1);
			p->subcarrierSpacing.push_back(1);
			p->cyclicPrefix.push_back(0);

			// Codeword information
			p->targetCodeRate.push_back(ue->targetCodeRate);
			p->qamModOrder.push_back(ue->qamModOrder);
			p->mcsIndex.push_back(ue->mcsIndex);
			p->mcsTable.push_back(ue->mcsTableIndex);
			p->transformPrecoding.push_back(ue->enableTfPrcd);
			p->dataScramblingId.push_back(ue->dataScramId);
			p->nrOfLayers.push_back(ue->nUeLayers);

			// DMRS [TS38.211 sec 6.4.1.1]
			p->ulDmrsSymbPos.push_back(ueGrp->dmrsSymLocBmsk);
			p->dmrsConfigType.push_back(0); // FIXME not stored in shared memory because it's not used
			p->ulDmrsScramblingId.push_back(ueGrp->pDmrsDynPrm->dmrsScrmId);
			p->puschIdentity.push_back(ue->puschIdentity);
			p->scid.push_back(ue->scid);
			p->numDmrsCdmGrpsNoData.push_back(ueGrp->pDmrsDynPrm->nDmrsCdmGrpsNoData);
			p->dmrsPorts.push_back(ue->dmrsPortBmsk);

			// Pusch Allocation in frequency domain [TS38.214, sec 6.1.2.2]
			p->resourceAlloc.push_back(0); // FIXME ueGrp->resourceAlloc);
			p->rbStart.push_back(ueGrp->startPrb);
			p->rbSize.push_back(ueGrp->nPrb);

			/* Note that the following variables aren't handled by L1 so rather than
			being unsigned and inserting 0 as in the spec make them signed and insert -1. */
			p->vrbToPrbMapping.push_back(-1);
			p->frequencyHopping.push_back(-1);
			p->txDirectCurrentLocation.push_back(-1);
			p->uplinkFrequencyShift7p5khz.push_back(-1);

			// Resource Allocation in time domain [TS38.214, sec 5.1.2.1]
			p->startSymbolIndex.push_back(ueGrp->puschStartSym);
			p->nrOfSymbols.push_back(ueGrp->nPuschSym);

			p->rvIndex.push_back(ue->rv);
			p->harqProcessId.push_back(ue->harqProcessId);
			p->newDataIndicator.push_back(ue->ndi);

			p->tbSize.push_back(tb_size);

			// Calculate actual number of CBs per UE from CB CRC data
			uint32_t numCbsForUe = 0;
			uint32_t cbStartOffset = 0;
			if (out_->pStartOffsetsCbCrc != nullptr) {
				cbStartOffset = out_->pStartOffsetsCbCrc[ueIdx];
				uint32_t cbEndOffset = (ueIdx < nUes - 1) ?
					out_->pStartOffsetsCbCrc[ueIdx + 1] : out_->totNumCbs;
				numCbsForUe = cbEndOffset - cbStartOffset;
			}
			p->numCb.push_back(numCbsForUe);

			// Zero means success but fapi wants 1 to mean failure
			uint8_t crcFail = (out_->pTbCrcs && out_->pStartOffsetsTbCrc) ? (out_->pTbCrcs[out_->pStartOffsetsTbCrc[ueIdx]] != 0) : 1;
			p->tbCrcFail.push_back(crcFail);
			bool savePdu = (crcFail == 0 || storeFailedPdu);

			// Count CB errors for this UE
			uint32_t cbErrorCount = 0;
			if (out_->pCbCrcs != nullptr && numCbsForUe > 0) {
				for (uint32_t cbIdx = cbStartOffset; cbIdx < cbStartOffset + numCbsForUe; cbIdx++) {
					if (out_->pCbCrcs[cbIdx] != 0) {
						cbErrorCount++;
					}
				}
			}
			p->cbErrors.push_back(cbErrorCount);

			// Add RSRP data per-UE
			if (out_->pRsrp != nullptr) {
				p->rsrp.push_back(out_->pRsrp[ueIdx]);
			} else {
				p->rsrp.push_back(-std::numeric_limits<float>::max());
			}

			if(savePdu) {
				p->pduLen.push_back(tb_size);
				NVLOGV_FMT(TAG_DATALAKE,"{:4}.{:02} {} us: ueIdx:{} copy {} bytes to: {}",slot_->sfn_,slot_->slot_,GET_ELAPSED_US(notifyTime),ueIdx,tb_size,(void*)p->pPduData.back());

				std::memcpy(p->pPduData.back(), data_buf.data(), tb_size * sizeof(uint8_t));
				p->pPduData.push_back(p->pPduData.back() + tb_size); // Update the pointer for next time
				p->pduOffsetsColumn->Append(p->pPduData.back() - p->pDataAlloc); // Current cumulative size of array
				//NVLOGD_FMT(TAG_DATALAKE,"{:4}.{:02} {} us: pduOffsetsColumn: {} = {}",
				//	slot_->sfn_,slot_->slot_,GET_ELAPSED_US(notifyTime),p->pduOffsetsColumn->Size(),p->pduOffsetsColumn->At(p->pduOffsetsColumn->Size()-1));
			} else {
				p->pduLen.push_back(0);
				NVLOGI_FMT(TAG_DATALAKE,"{:4}.{:02} {} us: ueIdx:{} crc fail",slot_->sfn_,slot_->slot_,GET_ELAPSED_US(notifyTime),ueIdx);

				p->pPduData.push_back(p->pPduData.back());
				p->pduOffsetsColumn->Append(p->pPduData.back() - p->pDataAlloc); // Cumulative size of array
				//NVLOGD_FMT(TAG_DATALAKE,"{:4}.{:02} {} us: pduOffsetsColumn: {} = {}",
				//	slot_->sfn_,slot_->slot_,GET_ELAPSED_US(notifyTime),p->pduOffsetsColumn->Size(),p->pduOffsetsColumn->At(p->pduOffsetsColumn->Size()-1));
			}

			p->sinr.push_back((out_->pSinrPreEq != nullptr) ? out_->pSinrPreEq[ueIdx]
			                : (out_->pSinrPostEq != nullptr) ? out_->pSinrPostEq[ueIdx]
			                : -std::numeric_limits<float>::max());
			p->noiseVar.push_back((out_->pNoiseVarPreEq != nullptr) ? out_->pNoiseVarPreEq[ueIdx]
			                    : (out_->pNoiseVarPostEq != nullptr) ? out_->pNoiseVarPostEq[ueIdx]
			                    : -std::numeric_limits<float>::max());
			p->timingAdvance.push_back((out_->pTaEsts != nullptr) ? out_->pTaEsts[ueIdx] : 0.0f);
			p->cfoHz.push_back((out_->pCfoHz != nullptr) ? out_->pCfoHz[ueIdx] : 0.0f);
			p->rssi.push_back((out_->pRssi != nullptr) ? out_->pRssi[params_->ue_info[ueIdx].ueGrpIdx] : -std::numeric_limits<float>::max());
			p->layerOffset.push_back(ueLayerOffsets[ueIdx]);
			p->ueGrpIdx.push_back(params_->ue_info[ueIdx].ueGrpIdx);
			p->hOffset.push_back(ueHOffsets[ueIdx]);
			p->hSize.push_back(ueHSizes[ueIdx]);
			p->nSubcarriers.push_back(ueNSubcarriers[ueIdx]);
			p->nDmrsEstimates.push_back(ueNDmrsEstimates[ueIdx]);
			p->dmrsSymbPos.push_back(ueDmrsSymbPos[ueIdx]);
		}

		NVLOGD_FMT(TAG_DATALAKE,"{:4}.{:02} {} us: collectSlot done {} UEs in pusch.{} buffer, size: {}",slot_->sfn_,slot_->slot_,GET_ELAPSED_US(notifyTime),nUes,p->bufferName,p->tsTaiNs.size());

		// Process FH data
		uint16_t nCells = params_->cell_grp_info.nCells;
		for (int cell = 0; cell < nCells; cell++) {
			auto cellInfo = params_->cell_dyn_info[cell];
			auto cellIdx = cellInfo.cellPrmStatIdx;
			auto pCell = puschStatPrms_->pCellStatPrms[cellIdx];
			auto cellId = pCell.phyCellId;
			auto nrxant = pCell.nRxAnt;
			auto nrxantsrs = pCell.nRxAntSrs;
			pFh->tsSwNs.push_back(ts_ns);
			pFh->tsTaiNs.push_back(ts_tai_ns);

			pFh->sfn.push_back(slot_->sfn_);
			pFh->slot.push_back(slot_->slot_);
			pFh->nUes.push_back(nUes); // Useful for referencing slots of interest

			pFh->cellId.push_back(cellId);
			pFh->nRxAnt.push_back(nrxant);
			pFh->nRxAntSrs.push_back(nrxantsrs);


			uint32_t offset = 273*12*14*16; // Max 16 antenna ports

			auto type_conversion_start_time = std::chrono::high_resolution_clock::now();
			size_t dataIndex = (pFh->tsTaiNs.size() - 1);
			auto copyStart = std::chrono::high_resolution_clock::now();
			// Can't do all of the cells at once because the memory isn't contiguous from GPU
			int16_t* prbs_int = reinterpret_cast<int16_t*>(&out_->pDataRx[0]+offset*cell);
			//NVLOGV_FMT(TAG_DATALAKE,"{:4}.{:02} {} us: Copying {} elements from {:p} to fhData[{}] at address {:p}", slot_->sfn_, slot_->slot_, GET_ELAPSED_US(notifyTime), nPrbs*2, (void*)prbs_int, dataIndex, (void*)pFh->fhData[dataIndex]);
			std::memcpy(pFh->fhData[dataIndex], prbs_int, nPrbs*2 * sizeof(int16_t));
			NVLOGV_FMT(TAG_DATALAKE,"{:4}.{:02} {} us: Done copying cell {} in {} us", slot_->sfn_, slot_->slot_, GET_ELAPSED_US(notifyTime), cell, GET_ELAPSED_US(copyStart));
		}
		if (pFh->tsTaiNs.size() == numRowsToInsertFh) {
			pFh->collectFullTime = std::chrono::high_resolution_clock::now();
		}
		if (p->tsTaiNs.size() == numRowsToInsertPusch) {
			p->collectFullTime = std::chrono::high_resolution_clock::now();
		}
		NVLOGD_FMT(TAG_DATALAKE,"{:4}.{:02} {} us: collectSlot done {} cells in fh.{} buffer, size: {}",
			slot_->sfn_,slot_->slot_,GET_ELAPSED_US(notifyTime),nCells,pFh->bufferName,pFh->tsTaiNs.size());

		// Process H estimates data -- all UE groups concatenated
		if (out_->pChannelEsts && out_->pChannelEstSizes && pHest->tsTaiNs.size() <= numRowsToInsertHest && params_->cell_grp_info.nUeGrps > 0) {
			if (pHest->tsTaiNs.size() == 0) {
				pHest->collectStartTime = std::chrono::high_resolution_clock::now();
			}

			// Get the first cell ID
			uint16_t cellId = 0;
			if (params_->cell_grp_info.nCells > 0) {
				cellId = puschStatPrms_->pCellStatPrms[0].phyCellId;
			}

			pHest->tsSwNs.push_back(ts_ns);
			pHest->tsTaiNs.push_back(ts_tai_ns);
			pHest->sfn.push_back(slot_->sfn_);
			pHest->slot.push_back(slot_->slot_);
			pHest->cellId.push_back(cellId);

			// Total H-est size across all groups (elements are contiguous after pusch_rx copy)
			uint32_t hestSize = 0;
			for (uint16_t g = 0; g < params_->cell_grp_info.nUeGrps; ++g) {
				hestSize += out_->pChannelEstSizes[g];
			}
			pHest->hestSize.push_back(hestSize);

			size_t dataIndex = pHest->tsTaiNs.size() - 1;
			pHest->hestData[dataIndex] = pHest->pDataAlloc + (pHest->writeOffsetBytes / sizeof(hestDataType));
			if (hestSize > 0 && hestSize <= maxHestSamplesPerRow) {
				std::memcpy(pHest->hestData[dataIndex], out_->pChannelEsts,
					hestSize * sizeof(hestDataType));
				pHest->writeOffsetBytes += hestSize * sizeof(hestDataType);
				NVLOGV_FMT(TAG_DATALAKE,"{:4}.{:02} {} us: Copied {} H estimate samples to hest[{}] offset {}",
					slot_->sfn_, slot_->slot_, GET_ELAPSED_US(notifyTime), hestSize, dataIndex, pHest->writeOffsetBytes);
			}

			if (pHest->tsTaiNs.size() == numRowsToInsertHest) {
				pHest->collectFullTime = std::chrono::high_resolution_clock::now();
			}

			NVLOGD_FMT(TAG_DATALAKE,"{:4}.{:02} {} us: collectSlot done H estimates in hest.{} buffer, size: {}",
				slot_->sfn_,slot_->slot_,GET_ELAPSED_US(notifyTime),pHest->bufferName,pHest->tsTaiNs.size());
		}
	}

	elapsedNotify = GET_ELAPSED_US(notifyTime);
	auto elapsedCollect = GET_ELAPSED_US(collectStart);
	if(elapsedNotify > 1000) { // 1 ms
		NVLOGW_FMT(TAG_DATALAKE,"{:4}.{:02} {} us: collectSlot took {} us, data may be incorrect",slot_->sfn_,slot_->slot_,elapsedNotify,elapsedCollect);
		// Could try popping the data to not keep it?
	} else {
		NVLOGD_FMT(TAG_DATALAKE,"{:4}.{:02} {} us: collectSlot done in {} us",slot_->sfn_,slot_->slot_,elapsedNotify,elapsedCollect);
	}
}

void DataLake::collectSrs(void)
{
	auto collectStart = std::chrono::high_resolution_clock::now();
	auto elapsedNotify = GET_ELAPSED_US(srsNotifyTime);
	if(elapsedNotify > 270) { // Host-pinned SRS buffers are reused by the next slot's GPU DMA; copy must start before overwrite
		NVLOGI_FMT(TAG_DATALAKE,"{:4}.{:02} {} us: collectSrs slow start, skip slot",srsSlot_->sfn_,srsSlot_->slot_,elapsedNotify);
		return;
	} else {
		NVLOGD_FMT(TAG_DATALAKE,"{:4}.{:02} {} us: collectSrs start",srsSlot_->sfn_,srsSlot_->slot_,elapsedNotify);
	}

	uint16_t nSrsUes = srsParams_->cell_grp_info.nSrsUes;
	uint16_t nCells  = srsParams_->cell_grp_info.nCells;
	if (nSrsUes == 0) {
		NVLOGD_FMT(TAG_DATALAKE,"{:4}.{:02} collectSrs: 0 SRS UEs, skip",srsSlot_->sfn_,srsSlot_->slot_);
		return;
	}

	if (pSrsIq->tsTaiNs.size() + nCells > static_cast<size_t>(numRowsToInsertSrsIq)) {
		NVLOGW_FMT(TAG_DATALAKE,"{:4}.{:02} {} us: Skipping SRS slot because {} SRS IQ buffer full, size: {}. Filled in {} ms",
			srsSlot_->sfn_,srsSlot_->slot_,GET_ELAPSED_US(srsNotifyTime),pSrsIq->bufferName,pSrsIq->tsTaiNs.size(),
			std::chrono::duration_cast<std::chrono::milliseconds>(pSrsIq->collectFullTime - pSrsIq->collectStartTime).count());
		return;
	}

	if (pSrs->tsTaiNs.size() + nSrsUes > static_cast<size_t>(numRowsToInsertSrs)) {
		NVLOGW_FMT(TAG_DATALAKE,"{:4}.{:02} {} us: Skipping SRS slot because {} SRS scalar buffer full, size: {}. Filled in {} ms",
			srsSlot_->sfn_,srsSlot_->slot_,GET_ELAPSED_US(srsNotifyTime),pSrs->bufferName,pSrs->tsTaiNs.size(),
			std::chrono::duration_cast<std::chrono::milliseconds>(pSrs->collectFullTime - pSrs->collectStartTime).count());
		return;
	}

	if (pSrsHest->tsTaiNs.size() + nSrsUes > static_cast<size_t>(numRowsToInsertSrsHest)) {
		NVLOGW_FMT(TAG_DATALAKE,"{:4}.{:02} {} us: Skipping SRS slot because {} SRS Hest buffer full, size: {}. Filled in {} ms",
			srsSlot_->sfn_,srsSlot_->slot_,GET_ELAPSED_US(srsNotifyTime),pSrsHest->bufferName,pSrsHest->tsTaiNs.size(),
			std::chrono::duration_cast<std::chrono::milliseconds>(pSrsHest->collectFullTime - pSrsHest->collectStartTime).count());
		return;
	}

	struct timespec ts;
	std::timespec_get(&ts, TIME_UTC);
	uint64_t ts_ns = ts.tv_sec * UINT64_C(1000000000) + ts.tv_nsec;
	uint64_t ts_tai_ns = sfn_to_tai(srsSlot_->sfn_, srsSlot_->slot_, ts_ns, 0, 0, 1);

	// --- SRS IQ (one row per cell) ---
	if (pSrsIq->tsTaiNs.size() == 0) {
		pSrsIq->collectStartTime = std::chrono::high_resolution_clock::now();
	}
	// Snapshot offset of this slot's first cell row before writing
	const size_t srsIqSlotStartOffsetBytes = pSrsIq->writeOffsetBytes;
	for (uint16_t cellIdx = 0; cellIdx < nCells; ++cellIdx) {
		auto& cellDyn = srsParams_->cell_dyn_info[cellIdx];
		uint16_t cellStatIdx = cellDyn.cellPrmStatIdx;
		uint16_t cellId = srsStatPrms_->pCellStatPrms[cellStatIdx].phyCellId;
		uint16_t nRxAntSrs = srsStatPrms_->pCellStatPrms[cellStatIdx].nRxAntSrs;

		pSrsIq->tsSwNs.push_back(ts_ns);
		pSrsIq->tsTaiNs.push_back(ts_tai_ns);
		pSrsIq->sfn.push_back(srsSlot_->sfn_);
		pSrsIq->slot.push_back(srsSlot_->slot_);
		pSrsIq->cellId.push_back(cellId);
		pSrsIq->nRxAntSrs.push_back(nRxAntSrs);
		pSrsIq->nSrsUes.push_back(nSrsUes);

		if (srsOut_->pDataRxSrs) {
			// 273 PRB * 12 SC * 6 SRS sym * nRxAntSrs * 2(I+Q) = per-cell sample count
			size_t perCellSamples = static_cast<size_t>(273) * 12 * 6 * nRxAntSrs * 2;
			size_t dataIndex = pSrsIq->tsTaiNs.size() - 1;
			size_t copySamples = std::min(perCellSamples, static_cast<size_t>(maxSrsIqSamplesPerRow)); // To-Do: revisit for mMIMO (>4 RX ant)
			size_t copyBytes = copySamples * sizeof(int16_t);
			pSrsIq->iqData[dataIndex] = pSrsIq->pDataAlloc + (pSrsIq->writeOffsetBytes / sizeof(int16_t));
			const int16_t* src = reinterpret_cast<const int16_t*>(srsOut_->pDataRxSrs) + perCellSamples * cellIdx;
			std::memcpy(pSrsIq->iqData[dataIndex], src, copyBytes);
			pSrsIq->writeOffsetBytes += copyBytes;
			NVLOGV_FMT(TAG_DATALAKE,"{:4}.{:02} {} us: SRS IQ cell {} copied {} bytes offset {}",
				srsSlot_->sfn_,srsSlot_->slot_,GET_ELAPSED_US(srsNotifyTime),cellIdx,copyBytes,pSrsIq->writeOffsetBytes);
		}
	}
	if (pSrsIq->tsTaiNs.size() >= static_cast<size_t>(numRowsToInsertSrsIq)) {
		pSrsIq->collectFullTime = std::chrono::high_resolution_clock::now();
	}

	// --- Per-UE: SRS scalars/RbSNR + SRS Hest ---
	auto* ueArr = srsParams_->cell_grp_info.pUeSrsPrms;

	if (pSrs->tsTaiNs.size() == 0) {
		pSrs->collectStartTime = std::chrono::high_resolution_clock::now();
	}
	if (pSrsHest->tsTaiNs.size() == 0) {
		pSrsHest->collectStartTime = std::chrono::high_resolution_clock::now();
	}

	for (uint16_t ueIdx = 0; ueIdx < nSrsUes; ++ueIdx) {
		auto& ue = ueArr[ueIdx];
		uint16_t cellStatIdx = ue.cellIdx;
		uint16_t cellId = srsStatPrms_->pCellStatPrms[cellStatIdx].phyCellId;

		// --- SRS scalar table (per-UE) ---
		pSrs->tsSwNs.push_back(ts_ns);
		pSrs->tsTaiNs.push_back(ts_tai_ns);
		pSrs->sfn.push_back(srsSlot_->sfn_);
		pSrs->slot.push_back(srsSlot_->slot_);
		pSrs->cellId.push_back(cellId);
		pSrs->rnti.push_back(ue.rnti);
		// Cell-level config
		pSrs->nCells.push_back(nCells);
		pSrs->srsCellStartSym.push_back(srsParams_->cell_dyn_info[0].srsStartSym);
		pSrs->srsCellNSrsSym.push_back(srsParams_->cell_dyn_info[0].nSrsSym);

		// Measurements from cuphySrsReport_t
		auto& rpt = srsOut_->pSrsReports[ueIdx];
		pSrs->widebandSnr.push_back(rpt.widebandSnr);
		pSrs->signalEnergy.push_back(rpt.widebandSignalEnergy);
		pSrs->noiseEnergy.push_back(rpt.widebandNoiseEnergy);
		pSrs->toaUs.push_back(rpt.toEstMicroSec);
		pSrs->hdAntFlag.push_back(rpt.highDensityAntPortFlag);
		__half2 sc = rpt.widebandScCorr;
		pSrs->scCorrRe.push_back(__half2float(sc.x));
		pSrs->scCorrIm.push_back(__half2float(sc.y));
		pSrs->csCorrRatioDb.push_back(rpt.widebandCsCorrRatioDb);

		// Config from cuphyUeSrsPrm_t
		pSrs->nAntPorts.push_back(ue.nAntPorts);
		pSrs->nSyms.push_back(ue.nSyms);
		pSrs->nRepetitions.push_back(ue.nRepetitions);
		pSrs->combSize.push_back(ue.combSize);
		pSrs->combOffset.push_back(ue.combOffset);
		pSrs->startSym.push_back(ue.startSym);
		pSrs->cyclicShift.push_back(ue.cyclicShift);
		pSrs->frequencyPosition.push_back(ue.frequencyPosition);
		pSrs->frequencyShift.push_back(ue.frequencyShift);
		pSrs->frequencyHopping.push_back(ue.frequencyHopping);
		pSrs->resourceType.push_back(ue.resourceType);
		pSrs->tSrs.push_back(ue.Tsrs);
		pSrs->tOffset.push_back(ue.Toffset);
		pSrs->usage.push_back(ue.usage);
		pSrs->nValidPrg.push_back(ue.nValidPrg);
		pSrs->prgSize.push_back(ue.prgSize);

		// Replay config
		pSrs->sequenceId.push_back(ue.sequenceId);
		pSrs->configIdx.push_back(ue.configIdx);
		pSrs->bandwidthIdx.push_back(ue.bandwidthIdx);
		pSrs->groupOrSequenceHopping.push_back(ue.groupOrSequenceHopping);

		// H-estimate grid dimensions (for JOIN with srs_hest)
		auto& chEst = srsOut_->pSrsChEstToL2[ueIdx];
		pSrs->nPrbGrps.push_back(chEst.nPrbGrps);
		pSrs->prbGrpSize.push_back(chEst.prbGrpSize);

		// RbSNR per-UE: stride is 273 floats per UE, valid data is nValidPrg floats
		uint32_t rbSnrStart = srsOut_->pRbSnrBuffOffsets[ueIdx];
		uint32_t rbSnrBytes = ue.nValidPrg * sizeof(float);

		size_t srsDataIdx = pSrs->tsTaiNs.size() - 1;
		pSrs->rbSnrSize.push_back(rbSnrBytes);
		pSrs->rbSnrData[srsDataIdx] = pSrs->pRbSnrDataAlloc + (pSrs->writeOffsetBytes / sizeof(float));
		if (rbSnrBytes > 0 && rbSnrBytes <= maxSrsRbSnrBytesPerRow) {
			std::memcpy(pSrs->rbSnrData[srsDataIdx],
				&srsOut_->pRbSnrBuffer[rbSnrStart],
				rbSnrBytes);
			pSrs->writeOffsetBytes += rbSnrBytes;
		}

		// --- SRS H-estimate table (per-UE) ---
		uint16_t nRxAntSrs = srsStatPrms_->pCellStatPrms[cellStatIdx].nRxAntSrs;
		// ChEst blob: nRxAntSrs * nPrbGrps * nAntPorts * sizeof(short2)
		// Matches srs_rx.cpp::copyOutputToCPU which uses this exact formula
		uint32_t hestBytes = static_cast<uint32_t>(nRxAntSrs) * chEst.nPrbGrps * ue.nAntPorts * sizeof(int32_t);
		if (hestBytes > maxSrsHestBytesPerRow) hestBytes = maxSrsHestBytesPerRow; // To-Do: revisit maxSrsHestBytesPerRow for mMIMO (>4 RX ant)

		pSrsHest->tsSwNs.push_back(ts_ns);
		pSrsHest->tsTaiNs.push_back(ts_tai_ns);
		pSrsHest->sfn.push_back(srsSlot_->sfn_);
		pSrsHest->slot.push_back(srsSlot_->slot_);
		pSrsHest->cellId.push_back(cellId);
		pSrsHest->rnti.push_back(ue.rnti);
		pSrsHest->hestSize.push_back(hestBytes);

		size_t hestDataIdx = pSrsHest->tsTaiNs.size() - 1;
		pSrsHest->hestData[hestDataIdx] = pSrsHest->pDataAlloc + (pSrsHest->writeOffsetBytes / sizeof(int16_t));
		if (chEst.pChEstCpuBuff && hestBytes > 0) {
			std::memcpy(pSrsHest->hestData[hestDataIdx], chEst.pChEstCpuBuff, hestBytes);
			pSrsHest->writeOffsetBytes += hestBytes;
			NVLOGV_FMT(TAG_DATALAKE,"{:4}.{:02} {} us: SRS Hest UE {} (rnti={}) copied {} bytes offset {}",
				srsSlot_->sfn_,srsSlot_->slot_,GET_ELAPSED_US(srsNotifyTime),ueIdx,ue.rnti,hestBytes,pSrsHest->writeOffsetBytes);
		}
	}

	if (pSrs->tsTaiNs.size() >= static_cast<size_t>(numRowsToInsertSrs)) {
		pSrs->collectFullTime = std::chrono::high_resolution_clock::now();
	}
	if (pSrsHest->tsTaiNs.size() >= static_cast<size_t>(numRowsToInsertSrsHest)) {
		pSrsHest->collectFullTime = std::chrono::high_resolution_clock::now();
	}

	// Store SRS buffer info for E3
	if (e3_agent) {
		std::lock_guard<std::mutex> lock(e3_srs_buffer_mutex);
		e3_srs_buffer_info.current_srs_iq_buffer = (pSrsIq == &srsIqInfo[0]) ? 0 : 1;
		e3_srs_buffer_info.current_srs_hest_buffer = (pSrsHest == &srsHestInfoBuf[0]) ? 0 : 1;
		e3_srs_buffer_info.current_srs_rb_snr_buffer = (pSrs == &srsScalarInfo[0]) ? 0 : 1;
		e3_srs_buffer_info.srs_iq_write_index = pSrsIq->tsTaiNs.size();
		e3_srs_buffer_info.srs_iq_row_byte_offset = static_cast<uint32_t>(srsIqSlotStartOffsetBytes);
		e3_srs_buffer_info.srs_hest_write_index = pSrsHest->tsTaiNs.size();
		e3_srs_buffer_info.srs_rb_snr_write_index = pSrs->tsTaiNs.size();
		e3_srs_buffer_info.sfn = srsSlot_->sfn_;
		e3_srs_buffer_info.slot = srsSlot_->slot_;
		e3_srs_buffer_info.timestamp_ns = ts_ns;
		e3_srs_buffer_info.timestamp_tai_ns = ts_tai_ns;

		if (nCells > 0) {
			uint16_t cellStatIdx = srsParams_->cell_dyn_info[0].cellPrmStatIdx;
			e3_srs_buffer_info.cell_id = srsStatPrms_->pCellStatPrms[cellStatIdx].phyCellId;
			e3_srs_buffer_info.n_cells = nCells;
			e3_srs_buffer_info.n_rx_ant_srs = srsStatPrms_->pCellStatPrms[cellStatIdx].nRxAntSrs;
			e3_srs_buffer_info.srs_cell_start_sym = srsParams_->cell_dyn_info[0].srsStartSym;
			e3_srs_buffer_info.srs_cell_n_srs_sym = srsParams_->cell_dyn_info[0].nSrsSym;
		}
		e3_srs_buffer_info.n_srs_ue = nSrsUes;

		e3_srs_buffer_info.ue_metrics.clear();
		e3_srs_buffer_info.ue_metrics.reserve(nSrsUes);
		for (uint16_t ueIdx = 0; ueIdx < nSrsUes; ++ueIdx) {
			auto& ue = ueArr[ueIdx];
			auto& rpt = srsOut_->pSrsReports[ueIdx];
			auto& chEst = srsOut_->pSrsChEstToL2[ueIdx];
			E3SrsUeMetrics m{};

			m.rnti = ue.rnti;
			m.wideband_snr = rpt.widebandSnr;
			m.signal_energy = rpt.widebandSignalEnergy;
			m.noise_energy = rpt.widebandNoiseEnergy;
			m.toa_us = rpt.toEstMicroSec;
			m.hd_ant_flag = rpt.highDensityAntPortFlag;
			__half2 sc = rpt.widebandScCorr;
			m.sc_corr_re = __half2float(sc.x);
			m.sc_corr_im = __half2float(sc.y);
			m.cs_corr_ratio_db = rpt.widebandCsCorrRatioDb;

			m.n_ant_ports = ue.nAntPorts;
			m.n_syms = ue.nSyms;
			m.n_repetitions = ue.nRepetitions;
			m.comb_size = ue.combSize;
			m.comb_offset = ue.combOffset;
			m.start_sym = ue.startSym;
			m.cyclic_shift = ue.cyclicShift;
			m.frequency_position = ue.frequencyPosition;
			m.frequency_shift = ue.frequencyShift;
			m.frequency_hopping = ue.frequencyHopping;
			m.resource_type = ue.resourceType;
			m.t_srs = ue.Tsrs;
			m.t_offset = ue.Toffset;
			m.usage = ue.usage;
			m.n_valid_prg = ue.nValidPrg;
			m.prg_size = ue.prgSize;
			m.n_prb_grps = chEst.nPrbGrps;

			// SHM cross-references for dApps to locate blobs in ping-pong buffers
			size_t rbSnrRow = pSrs->tsTaiNs.size() - nSrsUes + ueIdx;
			m.srs_rb_snr_offset = static_cast<uint32_t>(
				reinterpret_cast<uint8_t*>(pSrs->rbSnrData[rbSnrRow]) -
				reinterpret_cast<uint8_t*>(pSrs->pRbSnrDataAlloc));
			m.srs_rb_snr_size = ue.nValidPrg * sizeof(float);

			uint16_t nRxAntSrs = srsStatPrms_->pCellStatPrms[ue.cellIdx].nRxAntSrs;
			size_t hestRow = pSrsHest->tsTaiNs.size() - nSrsUes + ueIdx;
			m.srs_hest_offset = static_cast<uint32_t>(
				reinterpret_cast<uint8_t*>(pSrsHest->hestData[hestRow]) -
				reinterpret_cast<uint8_t*>(pSrsHest->pDataAlloc));
			m.srs_hest_size = static_cast<uint32_t>(nRxAntSrs) * chEst.nPrbGrps * ue.nAntPorts * sizeof(int32_t);
			if (m.srs_hest_size > maxSrsHestBytesPerRow) m.srs_hest_size = maxSrsHestBytesPerRow;

			e3_srs_buffer_info.ue_metrics.push_back(m);
		}
	}

	// Send E3 SRS notification after data is collected.
	if (e3_agent) {
		e3_agent->notifySrsDataReady();
	}

	NVLOGD_FMT(TAG_DATALAKE,"{:4}.{:02} {} us: collectSrs done {} UEs in {} us",
		srsSlot_->sfn_,srsSlot_->slot_,GET_ELAPSED_US(srsNotifyTime),nSrsUes,GET_ELAPSED_US(collectStart));
}

void DataLake::doInsertsPusch() {
	bool inserted = false;
	if(pInsertFh->tsTaiNs.size() == 0 && pFh->tsTaiNs.size() >= numRowsToInsertFh) {
		NVLOGD_FMT(TAG_DATALAKE,"{:4}.{:02} {} pFH: {}, {} us since notify",
			slot_->sfn_,slot_->slot_,__FUNCTION__,pFh->bufferName,GET_ELAPSED_US(notifyTime));
		std::swap(pFh, pInsertFh);

		if (fhDbEnabled) {
			inserted = true;
			submitTask([=,this]() { insertFh(pInsertFh); });
		} else {
			clearFhInfo(pInsertFh);
		}
	}

	if(pInsertPusch->tsTaiNs.size() == 0 && ((p->tsTaiNs.size() >= numRowsToInsertPusch) || (p->tsTaiNs.size() > 0 && flushColumns) )) {
		NVLOGD_FMT(TAG_DATALAKE,"{:4}.{:02} {} pPusch: {}, {} us since notify",
			slot_->sfn_,slot_->slot_,__FUNCTION__,p->bufferName,GET_ELAPSED_US(notifyTime));
		std::swap(p, pInsertPusch);

		if (puschDbEnabled) {
			inserted = true;
			submitTask([=,this]() { insertPusch(pInsertPusch); });
		} else {
			clearPuschInfo(pInsertPusch);
		}
	}

	if(pInsertHest->tsTaiNs.size() == 0 && pHest->tsTaiNs.size() >= numRowsToInsertHest) {
		NVLOGD_FMT(TAG_DATALAKE,"{:4}.{:02} {} pHest: {}, {} us since notify",
			slot_->sfn_,slot_->slot_,__FUNCTION__,pHest->bufferName,GET_ELAPSED_US(notifyTime));
		std::swap(pHest, pInsertHest);
		pHest->writeOffsetBytes = 0;

		if (hestDbEnabled) {
			inserted = true;
			submitTask([=,this]() { insertHest(pInsertHest); });
		} else {
			clearHestInfo(pInsertHest);
		}
	}

	if (inserted) {
		size_t completed = total_tasks_completed.load();
		if (completed > 0 && completed % 100 == 0) {
			logThreadPoolStats();
		}
	}
}

void DataLake::doInsertsSrs() {
	bool inserted = false;

	if (pInsertSrsIq->tsTaiNs.size() == 0 && pSrsIq->tsTaiNs.size() >= static_cast<size_t>(numRowsToInsertSrsIq)) {
		NVLOGD_FMT(TAG_DATALAKE,"{:4}.{:02} {} pSrsIq: {}, {} us since notify",
			srsSlot_->sfn_,srsSlot_->slot_,__FUNCTION__,pSrsIq->bufferName,GET_ELAPSED_US(srsNotifyTime));
		std::swap(pSrsIq, pInsertSrsIq);
		pSrsIq->writeOffsetBytes = 0;

		if (srsIqDbEnabled) {
			inserted = true;
			submitTask([=,this]() { insertSrsIq(pInsertSrsIq); });
		} else {
			clearSrsIqInfo(pInsertSrsIq);
		}
	}

	if (pInsertSrs->tsTaiNs.size() == 0 && pSrs->tsTaiNs.size() >= static_cast<size_t>(numRowsToInsertSrs)) {
		NVLOGD_FMT(TAG_DATALAKE,"{:4}.{:02} {} pSrs: {}, {} us since notify",
			srsSlot_->sfn_,srsSlot_->slot_,__FUNCTION__,pSrs->bufferName,GET_ELAPSED_US(srsNotifyTime));
		std::swap(pSrs, pInsertSrs);
		pSrs->writeOffsetBytes = 0;

		if (srsDbEnabled) {
			inserted = true;
			submitTask([=,this]() { insertSrs(pInsertSrs); });
		} else {
			clearSrsInfo(pInsertSrs);
		}
	}

	if (pInsertSrsHest->tsTaiNs.size() == 0 && pSrsHest->tsTaiNs.size() >= static_cast<size_t>(numRowsToInsertSrsHest)) {
		NVLOGD_FMT(TAG_DATALAKE,"{:4}.{:02} {} pSrsHest: {}, {} us since notify",
			srsSlot_->sfn_,srsSlot_->slot_,__FUNCTION__,pSrsHest->bufferName,GET_ELAPSED_US(srsNotifyTime));
		std::swap(pSrsHest, pInsertSrsHest);
		pSrsHest->writeOffsetBytes = 0;

		if (srsHestDbEnabled) {
			inserted = true;
			submitTask([=,this]() { insertSrsHest(pInsertSrsHest); });
		} else {
			clearSrsHestInfo(pInsertSrsHest);
		}
	}

	if (inserted) {
		size_t completed = total_tasks_completed.load();
		if (completed > 0 && completed % 100 == 0) {
			logThreadPoolStats();
		}
	}
}

void DataLake::submitTask(std::function<void()> task) {
	auto submission_start = std::chrono::high_resolution_clock::now();

	{
		std::lock_guard<std::mutex> lock(task_queue_mutex);
		task_queue.push(std::move(task));
	}
	task_queue_cv.notify_one();

	auto submission_end = std::chrono::high_resolution_clock::now();
	auto submission_time = std::chrono::duration_cast<std::chrono::nanoseconds>(submission_end - submission_start).count();

	// Update profiling metrics
	total_task_submission_time_ns.fetch_add(submission_time);
	total_tasks_submitted.fetch_add(1);
}

void DataLake::logThreadPoolStats() const {
	const size_t pool_size = db_write_thread_pool.size();
	const size_t active = active_threads.load();
	const size_t free_threads = pool_size - active;
	const size_t queued_tasks = task_queue.size();
	const size_t submitted = total_tasks_submitted.load();
	const size_t completed = total_tasks_completed.load();

	// Calculate average times
	uint64_t avg_submission_time_us = 0;
	uint64_t avg_execution_time_us = 0;

	if (submitted > 0) {
		avg_submission_time_us = total_task_submission_time_ns.load() / submitted / 1000;
	}
	if (completed > 0) {
		avg_execution_time_us = total_task_execution_time_ns.load() / completed / 1000;
	}

	NVLOGI_FMT(TAG_DATALAKE, "Thread Pool Stats - Total: {}, Active: {}, Peak: {}, Free: {}, Queued: {}, Submitted: {}, Completed: {}, Avg Submission: {} us, Avg Execution: {} us",
		pool_size, active, peak_active_threads.load(), free_threads, queued_tasks, submitted, completed, avg_submission_time_us, avg_execution_time_us);
}

size_t DataLake::getFreeThreadCount() const {
	return db_write_thread_pool.size() - active_threads.load();
}

size_t DataLake::getActiveThreadCount() const {
	return active_threads.load();
}

size_t DataLake::getPeakActiveThreadCount() const {
	return peak_active_threads.load();
}

size_t DataLake::getQueuedTaskCount() const {
	std::lock_guard<std::mutex> lock(task_queue_mutex);
	return task_queue.size();
}

double DataLake::getAverageTaskSubmissionTimeMs() const {
	const size_t submitted = total_tasks_submitted.load();
	if (submitted == 0) return 0.0;

	const uint64_t total_ns = total_task_submission_time_ns.load();
	return static_cast<double>(total_ns) / 1'000'000.0 / submitted;
}

double DataLake::getAverageTaskExecutionTimeMs() const {
	const size_t completed = total_tasks_completed.load();
	if (completed == 0) return 0.0;

	const uint64_t total_ns = total_task_execution_time_ns.load();
	return static_cast<double>(total_ns) / 1'000'000.0 / completed;
}


// Helper function to create a ClickHouse block from vectors
void DataLake::insertPusch(puschInfo_t* puschInfo) {
	auto start_time = std::chrono::high_resolution_clock::now();
	NVLOGD_FMT(TAG_DATALAKE,"insertPusch start {} us since notify, buffer capture duration: {} ms", GET_ELAPSED_US(notifyTime),
		std::chrono::duration_cast<std::chrono::milliseconds>(puschInfo->collectFullTime - puschInfo->collectStartTime).count());

	ch::Block block(PUSCH_INFO_MEMBER_COUNT, puschInfo->tsTaiNs.size());

	// Create columns from vectors
	auto tsSwNsCol = std::make_shared<ch::ColumnDateTime64>(9);
	auto tsTaiNsCol = std::make_shared<ch::ColumnDateTime64>(9);
	auto sfnCol = std::make_shared<ch::ColumnUInt16>();
	auto slotCol = std::make_shared<ch::ColumnUInt16>();
	auto nUesCol = std::make_shared<ch::ColumnUInt16>();
	auto cellIdCol = std::make_shared<ch::ColumnUInt16>();
	auto rntiCol = std::make_shared<ch::ColumnUInt16>();
	auto mcsIndexCol = std::make_shared<ch::ColumnUInt8>();
	auto rssiCol = std::make_shared<ch::ColumnFloat32>();
	auto pduLenCol = std::make_shared<ch::ColumnUInt32>();
	auto pduBitmapCol = std::make_shared<ch::ColumnUInt16>();
	auto bwpSizeCol = std::make_shared<ch::ColumnInt16>();
	auto bwpStartCol = std::make_shared<ch::ColumnInt16>();
	auto subcarrierSpacingCol = std::make_shared<ch::ColumnUInt8>();
	auto cyclicPrefixCol = std::make_shared<ch::ColumnUInt8>();
	auto targetCodeRateCol = std::make_shared<ch::ColumnUInt16>();
	auto qamModOrderCol = std::make_shared<ch::ColumnUInt8>();
	auto mcsTableCol = std::make_shared<ch::ColumnUInt8>();
	auto transformPrecodingCol = std::make_shared<ch::ColumnUInt8>();
	auto dataScramblingIdCol = std::make_shared<ch::ColumnUInt16>();
	auto nrOfLayersCol = std::make_shared<ch::ColumnUInt8>();
	auto ulDmrsSymbPosCol = std::make_shared<ch::ColumnUInt16>();
	auto dmrsConfigTypeCol = std::make_shared<ch::ColumnUInt8>();
	auto ulDmrsScramblingIdCol = std::make_shared<ch::ColumnUInt16>();
	auto puschIdentityCol = std::make_shared<ch::ColumnUInt16>();
	auto scidCol = std::make_shared<ch::ColumnUInt8>();
	auto numDmrsCdmGrpsNoDataCol = std::make_shared<ch::ColumnUInt8>();
	auto dmrsPortsCol = std::make_shared<ch::ColumnUInt16>();
	auto resourceAllocCol = std::make_shared<ch::ColumnUInt8>();
	auto rbStartCol = std::make_shared<ch::ColumnUInt16>();
	auto rbSizeCol = std::make_shared<ch::ColumnUInt16>();
	auto vrbToPrbMappingCol = std::make_shared<ch::ColumnInt8>();
	auto frequencyHoppingCol = std::make_shared<ch::ColumnInt8>();
	auto txDirectCurrentLocationCol = std::make_shared<ch::ColumnInt16>();
	auto uplinkFrequencyShift7p5khzCol = std::make_shared<ch::ColumnInt8>();
	auto startSymbolIndexCol = std::make_shared<ch::ColumnUInt8>();
	auto nrOfSymbolsCol = std::make_shared<ch::ColumnUInt8>();
	auto rvIndexCol = std::make_shared<ch::ColumnUInt8>();
	auto harqProcessIdCol = std::make_shared<ch::ColumnUInt8>();
	auto newDataIndicatorCol = std::make_shared<ch::ColumnUInt8>();
	auto tbSizeCol = std::make_shared<ch::ColumnUInt32>();
	auto numCbCol = std::make_shared<ch::ColumnUInt16>();
	auto sinrCol = std::make_shared<ch::ColumnFloat32>();
	auto noiseVarCol = std::make_shared<ch::ColumnFloat32>();
	auto tbCrcFailCol = std::make_shared<ch::ColumnUInt8>();
	auto timingAdvanceCol = std::make_shared<ch::ColumnFloat32>();
	auto cbErrorsCol = std::make_shared<ch::ColumnUInt8>();
	auto rsrpCol = std::make_shared<ch::ColumnFloat32>();
	auto layerOffsetCol = std::make_shared<ch::ColumnUInt16>();
	auto ueGrpIdxCol = std::make_shared<ch::ColumnUInt16>();
	auto hOffsetCol = std::make_shared<ch::ColumnUInt32>();
	auto hSizeCol = std::make_shared<ch::ColumnUInt32>();
	auto nSubcarriersCol = std::make_shared<ch::ColumnUInt16>();
	auto nDmrsEstimatesCol = std::make_shared<ch::ColumnUInt8>();
	auto dmrsSymbPosCol = std::make_shared<ch::ColumnUInt16>();
	auto cfoHzCol = std::make_shared<ch::ColumnFloat32>();
	auto nBsAntsCol = std::make_shared<ch::ColumnUInt8>();
	auto nCellsCol = std::make_shared<ch::ColumnUInt16>();

	// Fill columns from vectors
	for (size_t i = 0; i < puschInfo->tsTaiNs.size(); ++i) {
		tsSwNsCol->Append(puschInfo->tsSwNs[i]);
		tsTaiNsCol->Append(puschInfo->tsTaiNs[i]);
		sfnCol->Append(puschInfo->sfn[i]);
		slotCol->Append(puschInfo->slot[i]);
		nUesCol->Append(puschInfo->nUes[i]);
		cellIdCol->Append(puschInfo->cellId[i]);
		rntiCol->Append(puschInfo->rnti[i]);
		mcsIndexCol->Append(puschInfo->mcsIndex[i]);
		rssiCol->Append(puschInfo->rssi[i]);
		pduLenCol->Append(puschInfo->pduLen[i]);
		pduBitmapCol->Append(puschInfo->pduBitmap[i]);
		bwpSizeCol->Append(puschInfo->bwpSize[i]);
		bwpStartCol->Append(puschInfo->bwpStart[i]);
		subcarrierSpacingCol->Append(puschInfo->subcarrierSpacing[i]);
		cyclicPrefixCol->Append(puschInfo->cyclicPrefix[i]);
		targetCodeRateCol->Append(puschInfo->targetCodeRate[i]);
		qamModOrderCol->Append(puschInfo->qamModOrder[i]);
		mcsTableCol->Append(puschInfo->mcsTable[i]);
		transformPrecodingCol->Append(puschInfo->transformPrecoding[i]);
		dataScramblingIdCol->Append(puschInfo->dataScramblingId[i]);
		nrOfLayersCol->Append(puschInfo->nrOfLayers[i]);
		ulDmrsSymbPosCol->Append(puschInfo->ulDmrsSymbPos[i]);
		dmrsConfigTypeCol->Append(puschInfo->dmrsConfigType[i]);
		ulDmrsScramblingIdCol->Append(puschInfo->ulDmrsScramblingId[i]);
		puschIdentityCol->Append(puschInfo->puschIdentity[i]);
		scidCol->Append(puschInfo->scid[i]);
		numDmrsCdmGrpsNoDataCol->Append(puschInfo->numDmrsCdmGrpsNoData[i]);
		dmrsPortsCol->Append(puschInfo->dmrsPorts[i]);
		resourceAllocCol->Append(puschInfo->resourceAlloc[i]);
		rbStartCol->Append(puschInfo->rbStart[i]);
		rbSizeCol->Append(puschInfo->rbSize[i]);
		vrbToPrbMappingCol->Append(puschInfo->vrbToPrbMapping[i]);
		frequencyHoppingCol->Append(puschInfo->frequencyHopping[i]);
		txDirectCurrentLocationCol->Append(puschInfo->txDirectCurrentLocation[i]);
		uplinkFrequencyShift7p5khzCol->Append(puschInfo->uplinkFrequencyShift7p5khz[i]);
		startSymbolIndexCol->Append(puschInfo->startSymbolIndex[i]);
		nrOfSymbolsCol->Append(puschInfo->nrOfSymbols[i]);
		rvIndexCol->Append(puschInfo->rvIndex[i]);
		harqProcessIdCol->Append(puschInfo->harqProcessId[i]);
		newDataIndicatorCol->Append(puschInfo->newDataIndicator[i]);
		tbSizeCol->Append(puschInfo->tbSize[i]);
		numCbCol->Append(puschInfo->numCb[i]);
		sinrCol->Append(puschInfo->sinr[i]);
		noiseVarCol->Append(puschInfo->noiseVar[i]);
		tbCrcFailCol->Append(puschInfo->tbCrcFail[i]);
		timingAdvanceCol->Append(puschInfo->timingAdvance[i]);
		cbErrorsCol->Append(puschInfo->cbErrors[i]);
		rsrpCol->Append(puschInfo->rsrp[i]);
		layerOffsetCol->Append(puschInfo->layerOffset[i]);
		ueGrpIdxCol->Append(puschInfo->ueGrpIdx[i]);
		hOffsetCol->Append(puschInfo->hOffset[i]);
		hSizeCol->Append(puschInfo->hSize[i]);
		nSubcarriersCol->Append(puschInfo->nSubcarriers[i]);
		nDmrsEstimatesCol->Append(puschInfo->nDmrsEstimates[i]);
		dmrsSymbPosCol->Append(puschInfo->dmrsSymbPos[i]);
		cfoHzCol->Append(puschInfo->cfoHz[i]);
		nBsAntsCol->Append(puschInfo->nBsAnts[i]);
		nCellsCol->Append(puschInfo->nCells[i]);
		//NVLOGD_FMT(TAG_DATALAKE,"insertPusch pduData(offset)[{}]: {} crc: {} len: {}",
		//	i,puschInfo->pduOffsetsColumn->At(i),puschInfo->tbCrcFail[i],puschInfo->pduLen[i]);
	}

	auto pduColTime = std::chrono::high_resolution_clock::now();

	uint32_t copySize = puschInfo->pPduData.back() - puschInfo->pPduData.front();
	NVLOGD_FMT(TAG_DATALAKE,"insertPusch copy {} bytes {} rows",copySize, puschInfo->pduOffsetsColumn->Size());

	// Copy and create the array column with the preallocated columns
	auto& local_data_vector = pdu_data_column->GetWritableData();

	// Resize for clickhouse-cpp, then create the column
	local_data_vector.resize(copySize);
	std::memcpy(local_data_vector.data(), puschInfo->pDataAlloc, copySize);
	auto pduDataCol = std::make_shared<ch::ColumnArrayT<ch::ColumnUInt8>>(pdu_data_column, puschInfo->pduOffsetsColumn);

	auto pduColEnd = std::chrono::high_resolution_clock::now();
	NVLOGD_FMT(TAG_DATALAKE,"insertPusch create pdu column time: {} us",
		std::chrono::duration_cast<std::chrono::microseconds>(pduColEnd - pduColTime).count());

	// Append columns to block
	block.AppendColumn("TsSwNs", tsSwNsCol);
	block.AppendColumn("TsTaiNs", tsTaiNsCol);
	block.AppendColumn("SFN", sfnCol);
	block.AppendColumn("Slot", slotCol);
	block.AppendColumn("nUEs", nUesCol);
	block.AppendColumn("CellId", cellIdCol);
	block.AppendColumn("rnti", rntiCol);
	block.AppendColumn("mcsIndex", mcsIndexCol);
	block.AppendColumn("rssi", rssiCol);
	block.AppendColumn("pduBitmap", pduBitmapCol);
	block.AppendColumn("BWPSize", bwpSizeCol);
	block.AppendColumn("BWPStart", bwpStartCol);
	block.AppendColumn("SubcarrierSpacing", subcarrierSpacingCol);
	block.AppendColumn("CyclicPrefix", cyclicPrefixCol);
	block.AppendColumn("targetCodeRate", targetCodeRateCol);
	block.AppendColumn("qamModOrder", qamModOrderCol);
	block.AppendColumn("mcsTable", mcsTableCol);
	block.AppendColumn("TransformPrecoding", transformPrecodingCol);
	block.AppendColumn("dataScramblingId", dataScramblingIdCol);
	block.AppendColumn("nrOfLayers", nrOfLayersCol);
	block.AppendColumn("ulDmrsSymbPos", ulDmrsSymbPosCol);
	block.AppendColumn("dmrsConfigType", dmrsConfigTypeCol);
	block.AppendColumn("ulDmrsScramblingId", ulDmrsScramblingIdCol);
	block.AppendColumn("puschIdentity", puschIdentityCol);
	block.AppendColumn("SCID", scidCol);
	block.AppendColumn("numDmrsCdmGrpsNoData", numDmrsCdmGrpsNoDataCol);
	block.AppendColumn("dmrsPorts", dmrsPortsCol);
	block.AppendColumn("resourceAlloc", resourceAllocCol);
	block.AppendColumn("rbStart", rbStartCol);
	block.AppendColumn("rbSize", rbSizeCol);
	block.AppendColumn("VRBtoPRBMapping", vrbToPrbMappingCol);
	block.AppendColumn("FrequencyHopping", frequencyHoppingCol);
	block.AppendColumn("txDirectCurrentLocation", txDirectCurrentLocationCol);
	block.AppendColumn("uplinkFrequencyShift7p5khz", uplinkFrequencyShift7p5khzCol);
	block.AppendColumn("StartSymbolIndex", startSymbolIndexCol);
	block.AppendColumn("NrOfSymbols", nrOfSymbolsCol);
	block.AppendColumn("rvIndex", rvIndexCol);
	block.AppendColumn("harqProcessID", harqProcessIdCol);
	block.AppendColumn("newDataIndicator", newDataIndicatorCol);
	block.AppendColumn("TBSize", tbSizeCol);
	block.AppendColumn("numCb", numCbCol);
	block.AppendColumn("sinr", sinrCol);
	block.AppendColumn("noiseVar", noiseVarCol);
	block.AppendColumn("tbCrcFail", tbCrcFailCol);
	block.AppendColumn("timingAdvance", timingAdvanceCol);
	block.AppendColumn("pduLen", pduLenCol);
	block.AppendColumn("pduData", pduDataCol);
	block.AppendColumn("cbErrors", cbErrorsCol);
	block.AppendColumn("rsrp", rsrpCol);
	block.AppendColumn("layerOffset", layerOffsetCol);
	block.AppendColumn("ueGrpIdx", ueGrpIdxCol);
	block.AppendColumn("hOffset", hOffsetCol);
	block.AppendColumn("hSize", hSizeCol);
	block.AppendColumn("nSubcarriers", nSubcarriersCol);
	block.AppendColumn("nDmrsEstimates", nDmrsEstimatesCol);
	block.AppendColumn("dmrsSymbPos", dmrsSymbPosCol);
	block.AppendColumn("cfoHz", cfoHzCol);
	block.AppendColumn("nBsAnts", nBsAntsCol);
	block.AppendColumn("nCells", nCellsCol);

	auto insertStart = std::chrono::high_resolution_clock::now();
	dbClient->Insert("fapi", block);
	NVLOGD_FMT(TAG_DATALAKE,"{} {} rows {} insert time: {} ms",__FUNCTION__,
		puschInfo->tsTaiNs.size(), puschInfo->bufferName, GET_ELAPSED_MS(insertStart));

	clearPuschInfo(puschInfo);

	NVLOGI_FMT(TAG_DATALAKE,"insertPusch {} buffer took: {} ms", puschInfo->bufferName, GET_ELAPSED_MS(start_time));
}

// Helper function to insert fronthaul data
void DataLake::insertFh(fhInfo_t* fhInfo) {
	auto start_time = std::chrono::high_resolution_clock::now();
	NVLOGD_FMT(TAG_DATALAKE,"insertFh start {} us since notify, buffer capture duration: {} ms", GET_ELAPSED_US(notifyTime),
		std::chrono::duration_cast<std::chrono::milliseconds>(fhInfo->collectFullTime - fhInfo->collectStartTime).count());

	ch::Block block(FH_INFO_MEMBER_COUNT, fhInfo->tsTaiNs.size());

	auto cellIdCol = std::make_shared<ch::ColumnUInt16>();
	auto tsSwNsCol = std::make_shared<ch::ColumnDateTime64>(9);
	auto tsTaiNsCol = std::make_shared<ch::ColumnDateTime64>(9);
	auto sfnCol = std::make_shared<ch::ColumnUInt16>();
	auto slotCol = std::make_shared<ch::ColumnUInt16>();
	auto nRxAntCol = std::make_shared<ch::ColumnUInt16>();
	auto nRxAntSrsCol = std::make_shared<ch::ColumnUInt16>();
	auto nUesCol = std::make_shared<ch::ColumnUInt16>();

	// Fill columns from vectors
	auto appendHestStart = std::chrono::high_resolution_clock::now();
	for (size_t i = 0; i < fhInfo->tsSwNs.size(); ++i) {
		cellIdCol->Append(fhInfo->cellId[i]);
		tsSwNsCol->Append(fhInfo->tsSwNs[i]);
		tsTaiNsCol->Append(fhInfo->tsTaiNs[i]);
		sfnCol->Append(fhInfo->sfn[i]);
		slotCol->Append(fhInfo->slot[i]);
		nRxAntCol->Append(fhInfo->nRxAnt[i]);
		nRxAntSrsCol->Append(fhInfo->nRxAntSrs[i]);
		nUesCol->Append(fhInfo->nUes[i]);
	}
	auto appendHestEnd = std::chrono::high_resolution_clock::now();

	auto dataCopyStart = std::chrono::high_resolution_clock::now();

	// Use the preallocated static member variables
	auto& local_data_vector = fh_data_column->GetWritableData();

	// Copy all of the rows to the preallocated column memory
	std::memcpy(local_data_vector.data(), fhInfo->pDataAlloc, totalFhBytes);

	// Create the array column with the preallocated columns
	auto fhDataCol = std::make_shared<ch::ColumnArrayT<ch::ColumnInt16>>(fh_data_column, fh_offsets_column);

	auto dataCopyEnd = std::chrono::high_resolution_clock::now();
	NVLOGD_FMT(TAG_DATALAKE,"insertFh create info columns in: {} us, iq columns in: {} us",
		std::chrono::duration_cast<std::chrono::microseconds>(appendHestEnd - appendHestStart).count(),
		std::chrono::duration_cast<std::chrono::microseconds>(dataCopyEnd - dataCopyStart).count());

	// Append columns to block
	block.AppendColumn("CellId", cellIdCol);
	block.AppendColumn("TsSwNs", tsSwNsCol);
	block.AppendColumn("TsTaiNs", tsTaiNsCol);
	block.AppendColumn("SFN", sfnCol);
	block.AppendColumn("Slot", slotCol);
	block.AppendColumn("nRxAnt", nRxAntCol);
	block.AppendColumn("nRxAntSrs", nRxAntSrsCol);
	block.AppendColumn("nUEs", nUesCol);
	block.AppendColumn("fhData", fhDataCol);

	auto insertStart = std::chrono::high_resolution_clock::now();
	fhClient->Insert("fh", block);
	NVLOGD_FMT(TAG_DATALAKE,"{} {} rows {} insert time: {} ms",__FUNCTION__,
		fhInfo->tsTaiNs.size(), fhInfo->bufferName, GET_ELAPSED_MS(insertStart));

	clearFhInfo(fhInfo);

	NVLOGI_FMT(TAG_DATALAKE,"insertFh {} buffer took: {} ms", fhInfo->bufferName, GET_ELAPSED_MS(start_time));
}

// Helper function to insert H estimates data
void DataLake::insertHest(hestInfo_t* hestInfo) {
	auto start_time = std::chrono::high_resolution_clock::now();
	NVLOGD_FMT(TAG_DATALAKE,"insertHest start {} us since notify, buffer capture duration: {} ms", GET_ELAPSED_US(notifyTime),
		std::chrono::duration_cast<std::chrono::milliseconds>(hestInfo->collectFullTime - hestInfo->collectStartTime).count());

	ch::Block block(HEST_INFO_MEMBER_COUNT, hestInfo->tsTaiNs.size());

	auto cellIdCol = std::make_shared<ch::ColumnUInt16>();
	auto tsSwNsCol = std::make_shared<ch::ColumnDateTime64>(9);
	auto tsTaiNsCol = std::make_shared<ch::ColumnDateTime64>(9);
	auto sfnCol = std::make_shared<ch::ColumnUInt16>();
	auto slotCol = std::make_shared<ch::ColumnUInt16>();
	auto hestSizeCol = std::make_shared<ch::ColumnUInt32>();

	// Fill columns from vectors
	auto appendHestStart = std::chrono::high_resolution_clock::now();
	for (size_t i = 0; i < hestInfo->tsSwNs.size(); ++i) {
		cellIdCol->Append(hestInfo->cellId[i]);
		tsSwNsCol->Append(hestInfo->tsSwNs[i]);
		tsTaiNsCol->Append(hestInfo->tsTaiNs[i]);
		sfnCol->Append(hestInfo->sfn[i]);
		slotCol->Append(hestInfo->slot[i]);
		hestSizeCol->Append(hestInfo->hestSize[i]);
	}
	auto appendHestEnd = std::chrono::high_resolution_clock::now();

	auto dataCopyStart = std::chrono::high_resolution_clock::now();

	auto& local_data_vector = hest_data_column->GetWritableData();

	// cuFloatComplex is {float x, y} — contiguous, identical layout to float[2].
	hest_offsets_column->Clear();
	size_t dest_offset = 0;
	for (size_t row = 0; row < hestInfo->tsTaiNs.size(); ++row) {
		const uint32_t hestSize = hestInfo->hestSize[row];
		const hestDataType* src_data = hestInfo->hestData[row];

		std::memcpy(&local_data_vector[dest_offset], src_data, hestSize * sizeof(hestDataType));
		dest_offset += hestSize * 2; // *2: cuFloatComplex → 2 floats
		hest_offsets_column->Append(dest_offset);
	}
	local_data_vector.resize(dest_offset);

	auto hestDataCol = std::make_shared<ch::ColumnArrayT<ch::ColumnFloat32>>(hest_data_column, hest_offsets_column);

	auto dataCopyEnd = std::chrono::high_resolution_clock::now();
	NVLOGD_FMT(TAG_DATALAKE,"insertHest create info columns in: {} us, data columns in: {} us",
		std::chrono::duration_cast<std::chrono::microseconds>(appendHestEnd - appendHestStart).count(),
		std::chrono::duration_cast<std::chrono::microseconds>(dataCopyEnd - dataCopyStart).count());

	block.AppendColumn("CellId", cellIdCol);
	block.AppendColumn("TsSwNs", tsSwNsCol);
	block.AppendColumn("TsTaiNs", tsTaiNsCol);
	block.AppendColumn("SFN", sfnCol);
	block.AppendColumn("Slot", slotCol);
	block.AppendColumn("hestSize", hestSizeCol);
	block.AppendColumn("hestData", hestDataCol);

	auto insertStart = std::chrono::high_resolution_clock::now();
	hestClient->Insert("hest", block);
	NVLOGD_FMT(TAG_DATALAKE,"{} {} rows {} insert time: {} ms",__FUNCTION__,
		hestInfo->tsTaiNs.size(), hestInfo->bufferName, GET_ELAPSED_MS(insertStart));

	local_data_vector.resize(maxHestSamplesPerRow * 2 * numRowsToInsertHest);
	clearHestInfo(hestInfo);

	NVLOGI_FMT(TAG_DATALAKE,"insertHest {} buffer took: {} ms", hestInfo->bufferName, GET_ELAPSED_MS(start_time));
}

DataLake::~DataLake() {
	// Stop thread pool
	{
		std::lock_guard<std::mutex> lock(task_queue_mutex);
		stop_thread_pool.store(true);
	}
	task_queue_cv.notify_all();

	// Wait for all threads to finish
	for (auto& thread : db_write_thread_pool) {
		if (thread.joinable()) {
			thread.join();
		}
	}

	if (e3_agent) {
		// E3 MODE: E3Agent cleanup handled automatically by unique_ptr destructor
		// This will munmap shared memory and shm_unlink
		e3_agent.reset();
	} else {
		// REGULAR MODE: Clean up heap-allocated memory

		// Clean up fhDataAlloc memory
		if (pFh->pDataAlloc) {
			delete[] pFh->pDataAlloc;
			pFh->pDataAlloc = nullptr;
		}
		if (pInsertFh->pDataAlloc) {
			delete[] pInsertFh->pDataAlloc;
			pInsertFh->pDataAlloc = nullptr;
		}

		// Clean up pduDataAlloc memory
		if (p->pDataAlloc) {
			delete[] p->pDataAlloc;
			p->pDataAlloc = nullptr;
		}
		if (pInsertPusch->pDataAlloc) {
			delete[] pInsertPusch->pDataAlloc;
			pInsertPusch->pDataAlloc = nullptr;
		}

		// Clean up hestDataAlloc memory
		if (pHest->pDataAlloc) {
			delete[] pHest->pDataAlloc;
			pHest->pDataAlloc = nullptr;
		}
		if (pInsertHest->pDataAlloc) {
			delete[] pInsertHest->pDataAlloc;
			pInsertHest->pDataAlloc = nullptr;
		}

		// Clean up SRS buffers
		if (pSrsIq->pDataAlloc) { delete[] pSrsIq->pDataAlloc; pSrsIq->pDataAlloc = nullptr; }
		if (pInsertSrsIq->pDataAlloc) { delete[] pInsertSrsIq->pDataAlloc; pInsertSrsIq->pDataAlloc = nullptr; }
		if (pSrs->pRbSnrDataAlloc) { delete[] pSrs->pRbSnrDataAlloc; pSrs->pRbSnrDataAlloc = nullptr; }
		if (pInsertSrs->pRbSnrDataAlloc) { delete[] pInsertSrs->pRbSnrDataAlloc; pInsertSrs->pRbSnrDataAlloc = nullptr; }
		if (pSrsHest->pDataAlloc) { delete[] pSrsHest->pDataAlloc; pSrsHest->pDataAlloc = nullptr; }
		if (pInsertSrsHest->pDataAlloc) { delete[] pInsertSrsHest->pDataAlloc; pInsertSrsHest->pDataAlloc = nullptr; }
	}

	// Clean up database clients if needed
	if (dbClient) {
		delete dbClient;
		dbClient = nullptr;
	}
	if (fhClient) {
		delete fhClient;
		fhClient = nullptr;
	}
	if (hestClient) {
		delete hestClient;
		hestClient = nullptr;
	}
	if (srsIqClient) {
		delete srsIqClient;
		srsIqClient = nullptr;
	}
	if (srsClient) {
		delete srsClient;
		srsClient = nullptr;
	}
	if (srsHestClient) {
		delete srsHestClient;
		srsHestClient = nullptr;
	}
}

void DataLake::insertSrsIq(srsIqInfo_t* info) {
	auto start_time = std::chrono::high_resolution_clock::now();
	NVLOGD_FMT(TAG_DATALAKE,"insertSrsIq start {} us since notify, buffer capture duration: {} ms", GET_ELAPSED_US(srsNotifyTime),
		std::chrono::duration_cast<std::chrono::milliseconds>(info->collectFullTime - info->collectStartTime).count());

	ch::Block block(SRS_IQ_INFO_MEMBER_COUNT, info->tsTaiNs.size());

	auto cellIdCol = std::make_shared<ch::ColumnUInt16>();
	auto tsTaiNsCol = std::make_shared<ch::ColumnDateTime64>(9);
	auto tsSwNsCol = std::make_shared<ch::ColumnDateTime64>(9);
	auto sfnCol = std::make_shared<ch::ColumnUInt16>();
	auto slotCol = std::make_shared<ch::ColumnUInt16>();
	auto nRxAntSrsCol = std::make_shared<ch::ColumnUInt16>();
	auto nSrsUesCol = std::make_shared<ch::ColumnUInt16>();

	auto appendStart = std::chrono::high_resolution_clock::now();
	for (size_t i = 0; i < info->tsTaiNs.size(); ++i) {
		cellIdCol->Append(info->cellId[i]);
		tsTaiNsCol->Append(info->tsTaiNs[i]);
		tsSwNsCol->Append(info->tsSwNs[i]);
		sfnCol->Append(info->sfn[i]);
		slotCol->Append(info->slot[i]);
		nRxAntSrsCol->Append(info->nRxAntSrs[i]);
		nSrsUesCol->Append(info->nSrsUes[i]);
	}
	auto appendEnd = std::chrono::high_resolution_clock::now();

	auto dataCopyStart = std::chrono::high_resolution_clock::now();
	auto& local_data_vector = srs_iq_data_column->GetWritableData();

	srs_iq_offsets_column->Clear();
	size_t dest_offset_iq = 0;
	for (size_t row = 0; row < info->tsTaiNs.size(); ++row) {
		const size_t perCellSamples = static_cast<size_t>(273) * 12 * 6 * info->nRxAntSrs[row] * 2;
		const size_t validSamples = std::min(perCellSamples, static_cast<size_t>(maxSrsIqSamplesPerRow));
		std::memcpy(&local_data_vector[dest_offset_iq], info->iqData[row], validSamples * sizeof(int16_t));
		dest_offset_iq += validSamples;
		srs_iq_offsets_column->Append(dest_offset_iq);
	}
	local_data_vector.resize(dest_offset_iq);
	auto iqDataCol = std::make_shared<ch::ColumnArrayT<ch::ColumnInt16>>(srs_iq_data_column, srs_iq_offsets_column);
	auto dataCopyEnd = std::chrono::high_resolution_clock::now();
	NVLOGD_FMT(TAG_DATALAKE,"insertSrsIq create info columns in: {} us, iq columns in: {} us",
		std::chrono::duration_cast<std::chrono::microseconds>(appendEnd - appendStart).count(),
		std::chrono::duration_cast<std::chrono::microseconds>(dataCopyEnd - dataCopyStart).count());

	block.AppendColumn("CellId", cellIdCol);
	block.AppendColumn("TsTaiNs", tsTaiNsCol);
	block.AppendColumn("TsSwNs", tsSwNsCol);
	block.AppendColumn("SFN", sfnCol);
	block.AppendColumn("Slot", slotCol);
	block.AppendColumn("nRxAntSrs", nRxAntSrsCol);
	block.AppendColumn("nSrsUes", nSrsUesCol);
	block.AppendColumn("iqData", iqDataCol);

	auto insertStart = std::chrono::high_resolution_clock::now();
	srsIqClient->Insert("srs_iq", block);
	NVLOGD_FMT(TAG_DATALAKE,"{} {} rows {} insert time: {} ms",__FUNCTION__,
		info->tsTaiNs.size(), info->bufferName, GET_ELAPSED_MS(insertStart));

	local_data_vector.resize(maxSrsIqSamplesPerRow * numRowsToInsertSrsIq);
	clearSrsIqInfo(info);
	NVLOGI_FMT(TAG_DATALAKE,"insertSrsIq {} buffer took: {} ms", info->bufferName, GET_ELAPSED_MS(start_time));
}

void DataLake::insertSrs(srsInfo_t* info) {
	auto start_time = std::chrono::high_resolution_clock::now();
	NVLOGD_FMT(TAG_DATALAKE,"insertSrs start {} us since notify, buffer capture duration: {} ms", GET_ELAPSED_US(srsNotifyTime),
		std::chrono::duration_cast<std::chrono::milliseconds>(info->collectFullTime - info->collectStartTime).count());

	size_t nRows = info->tsTaiNs.size();
	ch::Block block(SRS_INFO_MEMBER_COUNT, nRows);

	auto cellIdCol = std::make_shared<ch::ColumnUInt16>();
	auto tsTaiNsCol = std::make_shared<ch::ColumnDateTime64>(9);
	auto tsSwNsCol = std::make_shared<ch::ColumnDateTime64>(9);
	auto sfnCol = std::make_shared<ch::ColumnUInt16>();
	auto slotCol = std::make_shared<ch::ColumnUInt16>();
	auto rntiCol = std::make_shared<ch::ColumnUInt16>();
	auto widebandSnrCol = std::make_shared<ch::ColumnFloat32>();
	auto signalEnergyCol = std::make_shared<ch::ColumnFloat32>();
	auto noiseEnergyCol = std::make_shared<ch::ColumnFloat32>();
	auto toaUsCol = std::make_shared<ch::ColumnFloat32>();
	auto hdAntFlagCol = std::make_shared<ch::ColumnUInt8>();
	auto scCorrReCol = std::make_shared<ch::ColumnFloat32>();
	auto scCorrImCol = std::make_shared<ch::ColumnFloat32>();
	auto csCorrRatioDbCol = std::make_shared<ch::ColumnFloat32>();
	auto nAntPortsCol = std::make_shared<ch::ColumnUInt8>();
	auto nSymsCol = std::make_shared<ch::ColumnUInt8>();
	auto nRepetitionsCol = std::make_shared<ch::ColumnUInt8>();
	auto combSizeCol = std::make_shared<ch::ColumnUInt8>();
	auto combOffsetCol = std::make_shared<ch::ColumnUInt8>();
	auto startSymCol = std::make_shared<ch::ColumnUInt8>();
	auto cyclicShiftCol = std::make_shared<ch::ColumnUInt8>();
	auto frequencyPositionCol = std::make_shared<ch::ColumnUInt8>();
	auto frequencyShiftCol = std::make_shared<ch::ColumnUInt16>();
	auto frequencyHoppingCol = std::make_shared<ch::ColumnUInt8>();
	auto resourceTypeCol = std::make_shared<ch::ColumnUInt8>();
	auto tSrsCol = std::make_shared<ch::ColumnUInt16>();
	auto tOffsetCol = std::make_shared<ch::ColumnUInt16>();
	auto usageCol = std::make_shared<ch::ColumnUInt32>();
	auto nValidPrgCol = std::make_shared<ch::ColumnUInt16>();
	auto prgSizeCol = std::make_shared<ch::ColumnUInt16>();
	auto sequenceIdCol = std::make_shared<ch::ColumnUInt16>();
	auto configIdxCol = std::make_shared<ch::ColumnUInt8>();
	auto bandwidthIdxCol = std::make_shared<ch::ColumnUInt8>();
	auto groupOrSequenceHoppingCol = std::make_shared<ch::ColumnUInt8>();
	auto nPrbGrpsCol = std::make_shared<ch::ColumnUInt16>();
	auto prbGrpSizeCol = std::make_shared<ch::ColumnUInt16>();
	auto rbSnrSizeCol = std::make_shared<ch::ColumnUInt32>();
	auto nCellsCol = std::make_shared<ch::ColumnUInt16>();
	auto srsCellStartSymCol = std::make_shared<ch::ColumnUInt8>();
	auto srsCellNSrsSymCol = std::make_shared<ch::ColumnUInt8>();

	auto appendStart = std::chrono::high_resolution_clock::now();
	for (size_t i = 0; i < nRows; ++i) {
		cellIdCol->Append(info->cellId[i]);
		tsTaiNsCol->Append(info->tsTaiNs[i]);
		tsSwNsCol->Append(info->tsSwNs[i]);
		sfnCol->Append(info->sfn[i]);
		slotCol->Append(info->slot[i]);
		rntiCol->Append(info->rnti[i]);
		widebandSnrCol->Append(info->widebandSnr[i]);
		signalEnergyCol->Append(info->signalEnergy[i]);
		noiseEnergyCol->Append(info->noiseEnergy[i]);
		toaUsCol->Append(info->toaUs[i]);
		hdAntFlagCol->Append(info->hdAntFlag[i]);
		scCorrReCol->Append(info->scCorrRe[i]);
		scCorrImCol->Append(info->scCorrIm[i]);
		csCorrRatioDbCol->Append(info->csCorrRatioDb[i]);
		nAntPortsCol->Append(info->nAntPorts[i]);
		nSymsCol->Append(info->nSyms[i]);
		nRepetitionsCol->Append(info->nRepetitions[i]);
		combSizeCol->Append(info->combSize[i]);
		combOffsetCol->Append(info->combOffset[i]);
		startSymCol->Append(info->startSym[i]);
		cyclicShiftCol->Append(info->cyclicShift[i]);
		frequencyPositionCol->Append(info->frequencyPosition[i]);
		frequencyShiftCol->Append(info->frequencyShift[i]);
		frequencyHoppingCol->Append(info->frequencyHopping[i]);
		resourceTypeCol->Append(info->resourceType[i]);
		tSrsCol->Append(info->tSrs[i]);
		tOffsetCol->Append(info->tOffset[i]);
		usageCol->Append(info->usage[i]);
		nValidPrgCol->Append(info->nValidPrg[i]);
		prgSizeCol->Append(info->prgSize[i]);
		sequenceIdCol->Append(info->sequenceId[i]);
		configIdxCol->Append(info->configIdx[i]);
		bandwidthIdxCol->Append(info->bandwidthIdx[i]);
		groupOrSequenceHoppingCol->Append(info->groupOrSequenceHopping[i]);
		nPrbGrpsCol->Append(info->nPrbGrps[i]);
		prbGrpSizeCol->Append(info->prbGrpSize[i]);
		rbSnrSizeCol->Append(info->rbSnrSize[i]);
		nCellsCol->Append(info->nCells[i]);
		srsCellStartSymCol->Append(info->srsCellStartSym[i]);
		srsCellNSrsSymCol->Append(info->srsCellNSrsSym[i]);
	}
	auto appendEnd = std::chrono::high_resolution_clock::now();

	auto dataCopyStart = std::chrono::high_resolution_clock::now();
	auto& rbSnr_local = srs_rb_snr_data_column->GetWritableData();

	srs_rb_snr_offsets_column->Clear();
	size_t dest_offset_snr = 0;
	for (size_t row = 0; row < nRows; ++row) {
		const uint32_t validFloats = info->rbSnrSize[row] / sizeof(float);
		std::memcpy(&rbSnr_local[dest_offset_snr], info->rbSnrData[row], validFloats * sizeof(float));
		dest_offset_snr += validFloats;
		srs_rb_snr_offsets_column->Append(dest_offset_snr);
	}
	rbSnr_local.resize(dest_offset_snr);
	auto rbSnrArrayCol = std::make_shared<ch::ColumnArrayT<ch::ColumnFloat32>>(srs_rb_snr_data_column, srs_rb_snr_offsets_column);
	auto dataCopyEnd = std::chrono::high_resolution_clock::now();
	NVLOGD_FMT(TAG_DATALAKE,"insertSrs create info columns in: {} us, rbSnr columns in: {} us",
		std::chrono::duration_cast<std::chrono::microseconds>(appendEnd - appendStart).count(),
		std::chrono::duration_cast<std::chrono::microseconds>(dataCopyEnd - dataCopyStart).count());
	block.AppendColumn("CellId", cellIdCol);
	block.AppendColumn("TsTaiNs", tsTaiNsCol);
	block.AppendColumn("TsSwNs", tsSwNsCol);
	block.AppendColumn("SFN", sfnCol);
	block.AppendColumn("Slot", slotCol);
	block.AppendColumn("rnti", rntiCol);
	block.AppendColumn("widebandSnr", widebandSnrCol);
	block.AppendColumn("signalEnergy", signalEnergyCol);
	block.AppendColumn("noiseEnergy", noiseEnergyCol);
	block.AppendColumn("toaUs", toaUsCol);
	block.AppendColumn("hdAntFlag", hdAntFlagCol);
	block.AppendColumn("scCorrRe", scCorrReCol);
	block.AppendColumn("scCorrIm", scCorrImCol);
	block.AppendColumn("csCorrRatioDb", csCorrRatioDbCol);
	block.AppendColumn("nAntPorts", nAntPortsCol);
	block.AppendColumn("nSyms", nSymsCol);
	block.AppendColumn("nRepetitions", nRepetitionsCol);
	block.AppendColumn("combSize", combSizeCol);
	block.AppendColumn("combOffset", combOffsetCol);
	block.AppendColumn("startSym", startSymCol);
	block.AppendColumn("cyclicShift", cyclicShiftCol);
	block.AppendColumn("frequencyPosition", frequencyPositionCol);
	block.AppendColumn("frequencyShift", frequencyShiftCol);
	block.AppendColumn("frequencyHopping", frequencyHoppingCol);
	block.AppendColumn("resourceType", resourceTypeCol);
	block.AppendColumn("tSrs", tSrsCol);
	block.AppendColumn("tOffset", tOffsetCol);
	block.AppendColumn("usage", usageCol);
	block.AppendColumn("nValidPrg", nValidPrgCol);
	block.AppendColumn("prgSize", prgSizeCol);
	block.AppendColumn("sequenceId", sequenceIdCol);
	block.AppendColumn("configIdx", configIdxCol);
	block.AppendColumn("bandwidthIdx", bandwidthIdxCol);
	block.AppendColumn("groupOrSequenceHopping", groupOrSequenceHoppingCol);
	block.AppendColumn("nPrbGrps", nPrbGrpsCol);
	block.AppendColumn("prbGrpSize", prbGrpSizeCol);
	block.AppendColumn("rbSnrSize", rbSnrSizeCol);
	block.AppendColumn("rbSnrData", rbSnrArrayCol);
	block.AppendColumn("nCells", nCellsCol);
	block.AppendColumn("srsCellStartSym", srsCellStartSymCol);
	block.AppendColumn("srsCellNSrsSym", srsCellNSrsSymCol);

	auto insertStart = std::chrono::high_resolution_clock::now();
	srsClient->Insert("srs", block);
	NVLOGD_FMT(TAG_DATALAKE,"{} {} rows {} insert time: {} ms",__FUNCTION__,
		nRows, info->bufferName, GET_ELAPSED_MS(insertStart));

	rbSnr_local.resize(maxSrsRbSnrSamplesPerRow * numRowsToInsertSrs);
	clearSrsInfo(info);
	NVLOGI_FMT(TAG_DATALAKE,"insertSrs {} buffer took: {} ms", info->bufferName, GET_ELAPSED_MS(start_time));
}

void DataLake::insertSrsHest(srsHestInfo_t* info) {
	auto start_time = std::chrono::high_resolution_clock::now();
	NVLOGD_FMT(TAG_DATALAKE,"insertSrsHest start {} us since notify, buffer capture duration: {} ms", GET_ELAPSED_US(srsNotifyTime),
		std::chrono::duration_cast<std::chrono::milliseconds>(info->collectFullTime - info->collectStartTime).count());

	ch::Block block(SRS_HEST_INFO_MEMBER_COUNT, info->tsTaiNs.size());

	auto cellIdCol = std::make_shared<ch::ColumnUInt16>();
	auto tsTaiNsCol = std::make_shared<ch::ColumnDateTime64>(9);
	auto tsSwNsCol = std::make_shared<ch::ColumnDateTime64>(9);
	auto sfnCol = std::make_shared<ch::ColumnUInt16>();
	auto slotCol = std::make_shared<ch::ColumnUInt16>();
	auto rntiCol = std::make_shared<ch::ColumnUInt16>();
	auto hestSizeCol = std::make_shared<ch::ColumnUInt32>();

	auto appendStart = std::chrono::high_resolution_clock::now();
	for (size_t i = 0; i < info->tsTaiNs.size(); ++i) {
		cellIdCol->Append(info->cellId[i]);
		tsTaiNsCol->Append(info->tsTaiNs[i]);
		tsSwNsCol->Append(info->tsSwNs[i]);
		sfnCol->Append(info->sfn[i]);
		slotCol->Append(info->slot[i]);
		rntiCol->Append(info->rnti[i]);
		hestSizeCol->Append(info->hestSize[i]);
	}
	auto appendEnd = std::chrono::high_resolution_clock::now();

	auto dataCopyStart = std::chrono::high_resolution_clock::now();

	// hestSize is in bytes; each int16 sample is 2 bytes.
	auto& local_data_vector = srs_hest_data_column->GetWritableData();

	srs_hest_offsets_column->Clear();
	size_t dest_offset_hest = 0;
	for (size_t row = 0; row < info->tsTaiNs.size(); ++row) {
		const uint32_t validSamples = info->hestSize[row] / sizeof(int16_t);
		std::memcpy(&local_data_vector[dest_offset_hest], info->hestData[row], validSamples * sizeof(int16_t));
		dest_offset_hest += validSamples;
		srs_hest_offsets_column->Append(dest_offset_hest);
	}
	local_data_vector.resize(dest_offset_hest);
	auto hestDataCol = std::make_shared<ch::ColumnArrayT<ch::ColumnInt16>>(srs_hest_data_column, srs_hest_offsets_column);

	auto dataCopyEnd = std::chrono::high_resolution_clock::now();
	NVLOGD_FMT(TAG_DATALAKE,"insertSrsHest create info columns in: {} us, data columns in: {} us",
		std::chrono::duration_cast<std::chrono::microseconds>(appendEnd - appendStart).count(),
		std::chrono::duration_cast<std::chrono::microseconds>(dataCopyEnd - dataCopyStart).count());

	block.AppendColumn("CellId", cellIdCol);
	block.AppendColumn("TsTaiNs", tsTaiNsCol);
	block.AppendColumn("TsSwNs", tsSwNsCol);
	block.AppendColumn("SFN", sfnCol);
	block.AppendColumn("Slot", slotCol);
	block.AppendColumn("rnti", rntiCol);
	block.AppendColumn("hestSize", hestSizeCol);
	block.AppendColumn("hestData", hestDataCol);

	auto insertStart = std::chrono::high_resolution_clock::now();
	srsHestClient->Insert("srs_hest", block);
	NVLOGD_FMT(TAG_DATALAKE,"{} {} rows {} insert time: {} ms",__FUNCTION__,
		info->tsTaiNs.size(), info->bufferName, GET_ELAPSED_MS(insertStart));

	local_data_vector.resize(maxSrsHestSamplesPerRow * numRowsToInsertSrsHest);
	clearSrsHestInfo(info);
	NVLOGI_FMT(TAG_DATALAKE,"insertSrsHest {} buffer took: {} ms", info->bufferName, GET_ELAPSED_MS(start_time));
}
