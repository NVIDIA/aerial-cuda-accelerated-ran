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

std::shared_ptr<chFhDataType> DataLake::fh_data_column = nullptr;
std::shared_ptr<ch::ColumnUInt64> DataLake::fh_offsets_column = nullptr;

std::shared_ptr<ch::ColumnUInt8> DataLake::pdu_data_column = nullptr;

std::shared_ptr<ch::ColumnFloat32> DataLake::hest_data_column = nullptr;
std::shared_ptr<ch::ColumnUInt64> DataLake::hest_offsets_column = nullptr;

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

void DataLake::initMem(void) {
	pFh = &fhInfo[0];
	pFh->bufferName = "ping";
	pInsertFh = &fhInfo[1];
	pInsertFh->bufferName = "pong";
	p = &puschInfo[0];
	p->bufferName = "ping";
	pInsertPusch = &puschInfo[1];
	pInsertPusch->bufferName = "pong";
	pHest = &hestInfo[0];
	pHest->bufferName = "ping";
	pInsertHest = &hestInfo[1];
	pInsertHest->bufferName = "pong";

	// Not sure this actually helps.
	forEachPuschInfoMember(p, [this](auto& vec) { vec.reserve(numRowsToInsertPusch); });
	forEachPuschInfoMember(pInsertPusch, [this](auto& vec) { vec.reserve(numRowsToInsertPusch); });

	forEachHestInfoMember(pHest, [this](auto& vec) { vec.reserve(numRowsToInsertHest); });
	forEachHestInfoMember(pInsertHest, [this](auto& vec) { vec.reserve(numRowsToInsertHest); });

	// Define average PDU size for buffer allocation
	constexpr uint32_t averagePduSize = 80000; // Largest PDU MCS table 1 MCS 27 14 symbols 273 PRBs is 159749 bytes

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
			maxHestSamplesPerRow
		);

		// Create shared memory buffers through E3Agent
		if (!e3_agent->createSharedMemoryBuffers(&pFh, &pInsertFh, &p, &pInsertPusch, &pHest, &pInsertHest)) {
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

		pFh->pDataAlloc = new fhDataType[numFhSamples*numRowsToInsertFh];
		pInsertFh->pDataAlloc = new fhDataType[numFhSamples*numRowsToInsertFh];
		p->pDataAlloc = new uint8_t[averagePduSize*numRowsToInsertPusch];
		pInsertPusch->pDataAlloc = new uint8_t[averagePduSize*numRowsToInsertPusch];
		pHest->pDataAlloc = new hestDataType[maxHestSamplesPerRow*numRowsToInsertHest];
		pInsertHest->pDataAlloc = new hestDataType[maxHestSamplesPerRow*numRowsToInsertHest];

		NVLOGC_FMT(TAG_DATALAKE, "DataLake initialized with regular heap allocation");
	}

	// Initialize data pointers
	pHest->hestData.resize(numRowsToInsertHest);
	pInsertHest->hestData.resize(numRowsToInsertHest);
	for (size_t i = 0; i < numRowsToInsertHest; i++) {
		pHest->hestData[i] = pHest->pDataAlloc + i * maxHestSamplesPerRow;
		pInsertHest->hestData[i] = pInsertHest->pDataAlloc + i * maxHestSamplesPerRow;
	}

	if (pHest->pDataAlloc && pInsertHest->pDataAlloc) {
		NVLOGI_FMT(TAG_DATALAKE,"Allocated memory for H estimates: 0x{:x}, 0x{:x}",
			(uintptr_t)pHest->pDataAlloc, (uintptr_t)pInsertHest->pDataAlloc);
	} else {
		NVLOGC_FMT(TAG_DATALAKE,"Failed to allocate memory for H estimates");
		return;
	}

	if (pFh->pDataAlloc && pInsertFh->pDataAlloc) {
		NVLOGI_FMT(TAG_DATALAKE,"Allocated memory for fhData: 0x{:x}, 0x{:x} of size {} bytes",
			(uintptr_t)pFh->pDataAlloc, (uintptr_t)pInsertFh->pDataAlloc, numFhSamples*numRowsToInsertFh*sizeof(fhDataType));
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
	fh_data_column = std::make_shared<chFhDataType>();
	fh_data_column->Reserve(numFhSamples*numRowsToInsertFh);
	auto& local_data_vector = fh_data_column->GetWritableData();
	for (size_t i = 0; i < numFhSamples * numRowsToInsertFh; ++i) {
		fh_data_column->Append(0);
	}

	// Initialize H estimates offsets column
	hest_offsets_column = std::make_shared<ch::ColumnUInt64>();
	for (size_t i = 0; i < numRowsToInsertHest; i++) {
		// Each H estimate sample is complex (2 floats), so multiply by 2
		hest_offsets_column->Append((i+1) * maxHestSamplesPerRow * 2);
	}

	// Initialize H estimates data column with proper size
	hest_data_column = std::make_shared<ch::ColumnFloat32>();
	hest_data_column->Reserve(maxHestSamplesPerRow * 2 * numRowsToInsertHest); // *2 for complex (real,imag)
	for (size_t i = 0; i < maxHestSamplesPerRow * 2 * numRowsToInsertHest; ++i) {
		hest_data_column->Append(0.0f);
	}

	p->pduOffsetsColumn = std::make_shared<ch::ColumnUInt64>();
	pInsertPusch->pduOffsetsColumn = std::make_shared<ch::ColumnUInt64>();

	if (p->pDataAlloc && pInsertPusch->pDataAlloc) {
		NVLOGI_FMT(TAG_DATALAKE,"Allocated memory for pduData: 0x{:x}, 0x{:x} of size {} bytes",
			(uintptr_t)p->pDataAlloc, (uintptr_t)pInsertPusch->pDataAlloc, averagePduSize*numRowsToInsertPusch*sizeof(uint8_t));
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
	pdu_data_column->Reserve(averagePduSize*numRowsToInsertPusch);
	NVLOGD_FMT(TAG_DATALAKE,"pdu_data_column: {} of {}", pdu_data_column->Size(), pdu_data_column->Capacity());

	NVLOGD_FMT(TAG_DATALAKE,"initMem done. Will save {} samples per UE to database, {} samples per FH insert and {} samples per PUSCH insert",
		numSamples_,numRowsToInsertFh,numRowsToInsertPusch);
}

void DataLake::dbInit (std::string host, std::string engine, bool dropTables) {
	NVLOGD_FMT(TAG_DATALAKE,"{} connecting to database at {}",__func__,host);
	static bool initDone = false;
	if(false == initDone) {
		dbClient = new ch::Client (ch::ClientOptions().SetHost(host));
		fhClient = new ch::Client (ch::ClientOptions().SetHost(host));
		hestClient = new ch::Client (ch::ClientOptions().SetHost(host));
		bfwClient = new ch::Client (ch::ClientOptions().SetHost(host));
		if (dropTables) {
			NVLOGC_FMT(TAG_DATALAKE,"Dropping tables per datalake_drop_tables");
			dbClient->Execute("DROP TABLE IF EXISTS fapi");
			fhClient->Execute("DROP TABLE IF EXISTS fh");
			hestClient->Execute("DROP TABLE IF EXISTS hest");
			bfwClient->Execute("DROP TABLE IF EXISTS dyn_bfw_weights");
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
			CQI							Float32, \
			timingAdvance 				Float32, \
			rssi						Float32, \
			pduLen						UInt32, \
			pduData						Array(UInt8), \
			cbErrors					UInt8, \
			rsrp						Float32 \
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

		// SRS-derived dynamic DL beamforming weights, per UE (rnti). bfwIQ is the
		// decompressed Q15 weight vector, I/Q interleaved, ordered [layer][prg][antenna].
		std::string createTableBfw = "CREATE TABLE IF NOT EXISTS dyn_bfw_weights ( \
			TsTaiNs			DateTime64(9)	NOT NULL, \
			SFN				UInt16	NOT NULL, \
			Slot			UInt16	NOT NULL, \
			CellId			UInt16	NOT NULL, \
			rnti			UInt16	NOT NULL, \
			beamId			UInt16	NOT NULL, \
			nGnbAnt			UInt8	NOT NULL, \
			nPrg			UInt16	NOT NULL, \
			prgSize			UInt16	NOT NULL, \
			bfwIqWidth		UInt8	NOT NULL, \
			nLayers			UInt8	NOT NULL, \
			bfwIQ			Array(Float32)	NOT NULL \
			) \
			ENGINE = " + engine + ";";

		// Otherwise the log is terrible
		std::regex tabRegex("\t+");
		NVLOGD_FMT(TAG_DATALAKE,"Creating table fapi: {}", std::regex_replace(createTableFapi, tabRegex, " "));
		NVLOGD_FMT(TAG_DATALAKE,"Creating table fh: {}", std::regex_replace(createTableFh, tabRegex, " "));
		NVLOGD_FMT(TAG_DATALAKE,"Creating table hest: {}", std::regex_replace(createTableHest, tabRegex, " "));

		dbClient->Execute(createTableFapi);
		fhClient->Execute(createTableFh);
		hestClient->Execute(createTableHest);


		bfwClient->Execute(createTableBfw);
		initDone = true;
	}
	notifyTime = std::chrono::high_resolution_clock::now();
}

// Dump SRS-derived dynamic DL BFW weights for one UE into dyn_bfw_weights.
// Called from the RT FAPI/slot-command path: it only copies the raw BFP block
// here, then defers BFP->Float32 decompression + ClickHouse insert to the worker
// thread pool so the RT path stays cheap.
void DataLake::notifyBfw(uint16_t cellId, uint16_t rnti, uint16_t sfn, uint16_t slot,
	uint16_t beamId, uint8_t nGnbAnt, uint16_t nPrg, uint16_t prgSize,
	uint8_t bfwIqWidth, uint8_t nLayers, const uint8_t* bfp, uint32_t bfpLen)
{
	if (bfwClient == nullptr || bfp == nullptr || bfpLen == 0 || nGnbAnt == 0 || bfwIqWidth == 0)
		return;

	// Per-bundle layout: 1 exponent byte + (2*nGnbAnt*bfwIqWidth/8) packed mantissa bytes.
	const uint32_t mantBytes  = (uint32_t)(2 * nGnbAnt * bfwIqWidth) / 8;
	const uint32_t bundleBytes = mantBytes + 1;
	const uint32_t nBundles   = (uint32_t)nPrg * (uint32_t)(nLayers ? nLayers : 1);
	if (bundleBytes == 0 || bfpLen < nBundles * bundleBytes)
		return;

	// Copy the raw BFP off the RT path; decode + insert on a worker thread.
	std::vector<uint8_t> raw(bfp, bfp + nBundles * bundleBytes);
	uint64_t tsTaiNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
		std::chrono::system_clock::now().time_since_epoch()).count();

	submitTask([=, this, raw = std::move(raw)]() {
		// MSB-first bit reader over the packed mantissa stream.
		auto getbits = [](const uint8_t* b, int pos, int n) -> int32_t {
			int32_t v = 0;
			for (int k = 0; k < n; ++k) { int p = pos + k; v = (v << 1) | ((b[p >> 3] >> (7 - (p & 7))) & 1); }
			return v;
		};
		const int iqw = bfwIqWidth;
		auto fp32 = std::make_shared<ch::ColumnFloat32>();
		auto offsets = std::make_shared<ch::ColumnUInt64>();
		uint64_t nVals = 0;
		for (uint32_t bnd = 0; bnd < nBundles; ++bnd) {
			const uint8_t* base = raw.data() + (size_t)bnd * bundleBytes;
			int32_t shift = base[0] & 0xFF;          // per-bundle BFP exponent
			const uint8_t* mant = base + 1;
			for (uint32_t a = 0; a < nGnbAnt; ++a) {
				int ioff = (int)a * 2 * iqw, qoff = ioff + iqw;
				// raw compbits -> sign-extend -> apply exponent (matches bfw_decompress_blockFP)
				int32_t ri = (getbits(mant, ioff, iqw) << (32 - iqw)) >> (32 - iqw - shift);
				int32_t rq = (getbits(mant, qoff, iqw) << (32 - iqw)) >> (32 - iqw - shift);
				fp32->Append((float)ri / 32768.0f);  // Q15 fraction (matches on-wire bfwI/bfwQ)
				fp32->Append((float)rq / 32768.0f);
				nVals += 2;
			}
		}
		offsets->Append(nVals);  // single row: Array offset = total element count

		auto col_u64 = [](uint64_t v){ auto c = std::make_shared<ch::ColumnUInt64>(); c->Append(v); return c; };
		auto col_u16 = [](uint16_t v){ auto c = std::make_shared<ch::ColumnUInt16>(); c->Append(v); return c; };
		auto col_u8  = [](uint8_t v){ auto c = std::make_shared<ch::ColumnUInt8>(); c->Append(v); return c; };

		auto tsCol = std::make_shared<ch::ColumnDateTime64>(9);
		tsCol->Append((int64_t)tsTaiNs);

		ch::Block block;
		block.AppendColumn("TsTaiNs",    tsCol);
		block.AppendColumn("SFN",        col_u16(sfn));
		block.AppendColumn("Slot",       col_u16(slot));
		block.AppendColumn("CellId",     col_u16(cellId));
		block.AppendColumn("rnti",       col_u16(rnti));
		block.AppendColumn("beamId",     col_u16(beamId));
		block.AppendColumn("nGnbAnt",    col_u8(nGnbAnt));
		block.AppendColumn("nPrg",       col_u16(nPrg));
		block.AppendColumn("prgSize",    col_u16(prgSize));
		block.AppendColumn("bfwIqWidth", col_u8(bfwIqWidth));
		block.AppendColumn("nLayers",    col_u8(nLayers ? nLayers : 1));
		block.AppendColumn("bfwIQ",      std::make_shared<ch::ColumnArrayT<ch::ColumnFloat32>>(fp32, offsets));
		try {
			std::lock_guard<std::mutex> lk(bfwClientMutex);
			bfwClient->Insert("dyn_bfw_weights", block);
		} catch (const std::exception& e) {
			NVLOGE_FMT(TAG_DATALAKE, AERIAL_CONFIG_EVENT, "notifyBfw insert failed: {}", e.what());
		}
	});
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

void* waitForLakeData(DataLake* dl)
{
	while (1) {
		if (__atomic_load_n(&dataLakeWorkReady, __ATOMIC_RELAXED)) {
			__atomic_store_n(&dataLakeWorkReady, false, __ATOMIC_RELAXED);
			__atomic_store_n(&dataLakeWorking, true, __ATOMIC_RELAXED);
			dl->collectSlot();
			__atomic_store_n(&dataLakeWorking, false, __ATOMIC_RELAXED);
		}
		dl->doInserts();
		std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
	}
}

void DataLake::collectSlot(void)
{
	NVLOGD_FMT(TAG_DATALAKE, "TIMESTAMP_LOG: DataLake collectSlot (Op #3) entry at {}", std::chrono::high_resolution_clock::now().time_since_epoch().count());
	static int insert_cnt_fh = 0;
	static int insert_cnt_fapi = 0;

	auto collectStart = std::chrono::high_resolution_clock::now();
	auto elapsedNotify = GET_ELAPSED_US(notifyTime);
	if(elapsedNotify > 270) { // If isn't called early enough, the CRCs will all be wrong
		NVLOGI_FMT(TAG_DATALAKE,"{:4}.{:02} {} us: collectSlot slow start, skip slot",slot_->sfn_,slot_->slot_,elapsedNotify);
		return;
	} else {
		NVLOGD_FMT(TAG_DATALAKE,"{:4}.{:02} {} us: collectSlot start",slot_->sfn_,slot_->slot_,elapsedNotify);
	}

	// Currently only handling one ue group
	int ueGrpIdx = 0;

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
			e3_buffer_info.sfn = slot_->sfn_;
			e3_buffer_info.slot = slot_->slot_;
			e3_buffer_info.timestamp_ns = ts_ns;

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

			// H estimates and PUSCH metadata - from first UE group
			if (params_->cell_grp_info.nUeGrps > 0 && params_->cell_grp_info.nUes > 0) {
				// TODO: Extend to capture all UEs using array/vector structure for per-UE data
				//       Currently only capturing first UE from params_->cell_grp_info.nUes total
				e3_buffer_info.n_ue = 1;  // Only first UE captured

				auto ue = &params_->ue_info[0];  // First UE
				auto ueGrp = ue->pUeGrpPrm;

				// H estimates metadata
				e3_buffer_info.n_bs_ants = puschStatPrms_->pCellStatPrms[0].nRxAnt;  // Using RX antennas as BS antennas
				e3_buffer_info.n_layers = ue->nUeLayers;
				e3_buffer_info.rb_size = ueGrp->nPrb;
				e3_buffer_info.n_subcarriers = ueGrp->nPrb * 12;  // 12 subcarriers per PRB
				e3_buffer_info.dmrs_symb_pos = ueGrp->dmrsSymLocBmsk;

				// Calculate number of DMRS estimates from dmrsAddlnPos
				uint8_t dmrsAddlnPos = ueGrp->pDmrsDynPrm->dmrsAddlnPos;
				e3_buffer_info.n_dmrs_estimates = dmrsAddlnPos + 1;

				// Get actual H estimates data size
				e3_buffer_info.hest_data_size = 0;
				if (out_->pChannelEstSizes) {
					e3_buffer_info.hest_data_size = out_->pChannelEstSizes[0];  // Number of __half2 elements
				}

				// PUSCH FAPI fields
				e3_buffer_info.rnti = ue->rnti;
				e3_buffer_info.qam_mod_order = ue->qamModOrder;
				e3_buffer_info.mcs_index = ue->mcsIndex;
				e3_buffer_info.mcs_table_index = ue->mcsTableIndex;
				e3_buffer_info.rb_start = ueGrp->startPrb;
				e3_buffer_info.start_symbol_index = ueGrp->puschStartSym;
				e3_buffer_info.nr_of_symbols = ueGrp->nPuschSym;

				// CB errors and RSRP for first UE (UE index 0)
				uint32_t cbStartOffset = out_->pStartOffsetsCbCrc[0];
				uint32_t cbEndOffset = (params_->cell_grp_info.nUes > 1) ?
					out_->pStartOffsetsCbCrc[1] : out_->totNumCbs;
				uint32_t cbErrorCount = 0;
				if (out_->pCbCrcs != nullptr) {
					for (uint32_t cbIdx = cbStartOffset; cbIdx < cbEndOffset; cbIdx++) {
						if (out_->pCbCrcs[cbIdx] != 0) cbErrorCount++;
					}
				}
				e3_buffer_info.cb_errors = cbErrorCount;

				// RSRP for first UE
				if (out_->pRsrp != nullptr) {
					e3_buffer_info.rsrp = out_->pRsrp[0];
				} else {
					e3_buffer_info.rsrp = -std::numeric_limits<float>::max();
				}

				// CQI/SINR for first UE
				if (out_->pSinrPostEq != nullptr) {
					e3_buffer_info.cqi = out_->pSinrPostEq[0];
				} else if (out_->pSinrPreEq != nullptr) {
					e3_buffer_info.cqi = out_->pSinrPreEq[0];
				} else {
					e3_buffer_info.cqi = -std::numeric_limits<float>::max();
				}

				// CB count for first UE
				uint32_t numCbsForFirstUe = cbEndOffset - cbStartOffset;
				e3_buffer_info.cb_count = numCbsForFirstUe;

				// RSSI for first UE's group (per-UE-group data)
				if (out_->pRssi != nullptr) {
					e3_buffer_info.rssi = out_->pRssi[ue->ueGrpIdx];
				} else {
					e3_buffer_info.rssi = -std::numeric_limits<float>::max();
				}
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

		// Process UE data
		// Initialize CRC failure tracking for E3
		if (e3_agent) {
			std::lock_guard<std::mutex> lock(e3_buffer_mutex);
			e3_buffer_info.tb_crc_fail = 0;  // Default to no CRC failure
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
			uint32_t cbStartOffset = out_->pStartOffsetsCbCrc[ueIdx];
			uint32_t cbEndOffset = (ueIdx < nUes - 1) ?
				out_->pStartOffsetsCbCrc[ueIdx + 1] : out_->totNumCbs;
			uint32_t numCbsForUe = cbEndOffset - cbStartOffset;
			p->numCb.push_back(numCbsForUe);

			// Zero means sucess but fapi wants 1 to mean failure
			uint8_t crcFail = (0 != out_->pTbCrcs[out_->pStartOffsetsTbCrc[ueIdx]]);
			p->tbCrcFail.push_back(crcFail);
			bool savePdu = (crcFail == 0 || storeFailedPdu);

			// Count CB errors for this UE
			uint32_t cbErrorCount = 0;
			if (out_->pCbCrcs != nullptr && numCbsForUe > 0) {
				for (uint32_t cbIdx = cbStartOffset; cbIdx < cbEndOffset; cbIdx++) {
					if (out_->pCbCrcs[cbIdx] != 0) {  // Non-zero indicates CRC error
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

			// Update aggregated CRC failure status for E3
			if (e3_agent && crcFail != 0) {
				std::lock_guard<std::mutex> lock(e3_buffer_mutex);
				e3_buffer_info.tb_crc_fail = 1;  // Set to 1 if any UE has CRC failure
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

			if(out_->pSinrPostEq != nullptr) {
				p->cqi.push_back(out_->pSinrPostEq[ueIdx]);
			} else if (out_->pSinrPreEq != nullptr) {
				p->cqi.push_back(out_->pSinrPreEq[ueIdx]);
			} else {
				p->cqi.push_back(-std::numeric_limits<float>::max());
			}

			p->timingAdvance.push_back(out_->pTaEsts[ueIdx]);

			p->rssi.push_back(out_->pRssi[params_->ue_info[ueIdx].ueGrpIdx]);
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
			fhDataType* prbs_int = reinterpret_cast<fhDataType*>(&out_->pDataRx[0]+offset*cell);
			//NVLOGV_FMT(TAG_DATALAKE,"{:4}.{:02} {} us: Copying {} elements from {:p} to fhData[{}] at address {:p}", slot_->sfn_, slot_->slot_, GET_ELAPSED_US(notifyTime), nPrbs*2, (void*)prbs_int, dataIndex, (void*)pFh->fhData[dataIndex]);
			std::memcpy(pFh->fhData[dataIndex], prbs_int, nPrbs*2 * sizeof(fhDataType));
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

		// Process H estimates data - Currently only handling first UE group (like IQ samples)
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

			// Get size for first UE group only (in elements, not bytes)
			uint32_t hestSize = 0;
			if (out_->pChannelEstSizes) {
				hestSize = out_->pChannelEstSizes[0];  // Number of __half2 elements
			}
			pHest->hestSize.push_back(hestSize);

			// Copy H estimates data for first UE group only
			size_t dataIndex = pHest->tsTaiNs.size() - 1;
			if (hestSize > 0 && hestSize <= maxHestSamplesPerRow) {
				std::memcpy(pHest->hestData[dataIndex], out_->pChannelEsts,
					hestSize * sizeof(hestDataType));
				NVLOGV_FMT(TAG_DATALAKE,"{:4}.{:02} {} us: Copied {} H estimate samples to hest[{}]",
					slot_->sfn_, slot_->slot_, GET_ELAPSED_US(notifyTime), hestSize, dataIndex);
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

void DataLake::doInserts() {
	bool inserted = false;
	// Do this check before the insertFh check because this is a fixed sized buffer
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
	auto cqiCol = std::make_shared<ch::ColumnFloat32>();
	auto tbCrcFailCol = std::make_shared<ch::ColumnUInt8>();
	auto timingAdvanceCol = std::make_shared<ch::ColumnFloat32>();
	auto cbErrorsCol = std::make_shared<ch::ColumnUInt8>();
	auto rsrpCol = std::make_shared<ch::ColumnFloat32>();

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
		cqiCol->Append(puschInfo->cqi[i]);
		tbCrcFailCol->Append(puschInfo->tbCrcFail[i]);
		timingAdvanceCol->Append(puschInfo->timingAdvance[i]);
		cbErrorsCol->Append(puschInfo->cbErrors[i]);
		rsrpCol->Append(puschInfo->rsrp[i]);
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
	block.AppendColumn("CQI", cqiCol);
	block.AppendColumn("tbCrcFail", tbCrcFailCol);
	block.AppendColumn("timingAdvance", timingAdvanceCol);
	block.AppendColumn("pduLen", pduLenCol);
	block.AppendColumn("pduData", pduDataCol);
	block.AppendColumn("cbErrors", cbErrorsCol);
	block.AppendColumn("rsrp", rsrpCol);

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
	auto fhDataCol = std::make_shared<ch::ColumnArrayT<chFhDataType>>(fh_data_column, fh_offsets_column);

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

	// Use the preallocated static member variables
	auto& local_data_vector = hest_data_column->GetWritableData();

	// Copy H estimates data (interleaved complex: real1, imag1, real2, imag2, ...)
	// Physical memory layout from cuPHY: [dmrs][subcarrier][antenna][layer] (row-major)
	size_t dest_offset = 0;
	for (size_t row = 0; row < hestInfo->tsTaiNs.size(); ++row) {
		const uint32_t hestSize = hestInfo->hestSize[row];
		const hestDataType* src_data = hestInfo->hestData[row];

		// Copy interleaved complex data: cuFloatComplex -> [real, imag, real, imag, ...]
		for (uint32_t i = 0; i < hestSize; ++i) {
			local_data_vector[dest_offset++] = src_data[i].x; // real part
			local_data_vector[dest_offset++] = src_data[i].y; // imaginary part
		}

		// Fill remaining slots with zeros if needed
		const uint32_t remainingSlots = maxHestSamplesPerRow - hestSize;
		for (uint32_t i = 0; i < remainingSlots * 2; ++i) {
			local_data_vector[dest_offset++] = 0.0f;
		}
	}

	// Create the array column with the preallocated columns
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
}


