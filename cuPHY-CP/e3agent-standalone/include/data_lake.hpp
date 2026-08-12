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

// Drop-in replacement for aerial_sdk's data_lake.hpp, shadowing it on the
// harness include path. Provides only what e3_agent.cpp/.hpp reference, with
// none of the cuPHY/CUDA/FAPI/ClickHouse dependencies. Same include guard as
// the real header so it wins if both are ever reachable.
//
// NOTE: e3_agent.cpp must be compiled from a copied location (not its original
// aerial_sdk dir), otherwise its `#include "data_lake.hpp"` resolves to the
// adjacent real header before this one. See e3agent-standalone CMake staging.

#ifndef DATA_LAKE_H
#define DATA_LAKE_H

#include <cstdint>
#include <mutex>
#include <vector>

#include "config.hpp"
#include "e3_types.hpp"

// Minimal stand-in for the constant e3_agent.cpp uses as a reserve() hint.
namespace slot_command_api {
inline constexpr int MAX_PUSCH_UE_PER_TTI = 132;
}

// PUSCH SHM blob buffer. The real struct also stages ClickHouse columns; the
// harness needs only the raw PDU allocation (set by createSharedMemoryBuffers)
// and the per-PDU pointer list (written by the synth/replay path).
struct puschInfo_t {
	uint8_t* pDataAlloc{};
	std::vector<uint8_t*> pPduData;
};

// E3-facing subset of DataLake. Members are public (no friend needed): the
// agent reads/writes these directly, and the harness populates them.
class DataLake {
public:
	std::mutex e3_buffer_mutex;
	E3BufferInfo e3_buffer_info;
	std::mutex e3_srs_buffer_mutex;
	E3SrsBufferInfo e3_srs_buffer_info;

	// Double-buffered ring storage. createSharedMemoryBuffers() points the
	// pData* members into SHM; the feeder fills rows (blobs deferred).
	fhInfo_t fhInfo[2];
	puschInfo_t puschInfo[2];
	hestInfo_t hestInfo[2];
	srsIqInfo_t srsIqInfo[2];
	srsInfo_t srsInfo[2];
	srsHestInfo_t srsHestInfo[2];

	fhInfo_t *pFh{}, *pInsertFh{};
	puschInfo_t *p{}, *pInsertPusch{};
	hestInfo_t *pHest{}, *pInsertHest{};
	srsIqInfo_t *pSrsIq{}, *pInsertSrsIq{};
	srsInfo_t *pSrs{}, *pInsertSrs{};
	srsHestInfo_t *pSrsHest{}, *pInsertSrsHest{};

	void initBuffers() {
		pFh = &fhInfo[0];           pInsertFh = &fhInfo[1];
		p = &puschInfo[0];          pInsertPusch = &puschInfo[1];
		pHest = &hestInfo[0];       pInsertHest = &hestInfo[1];
		pSrsIq = &srsIqInfo[0];     pInsertSrsIq = &srsIqInfo[1];
		pSrs = &srsInfo[0];         pInsertSrs = &srsInfo[1];
		pSrsHest = &srsHestInfo[0]; pInsertSrsHest = &srsHestInfo[1];
	}

	// Resize blob-pointer vectors into SHM. Call after createSharedMemoryBuffers().
	void initRowPointers(const e3sa::RowsCfg& rows) {
		for (int h = 0; h < 2; ++h) {
			fhInfo[h].fhData.resize(rows.fh);
			for (uint32_t i = 0; i < rows.fh; ++i) {
				fhInfo[h].fhData[i] = fhInfo[h].pDataAlloc + i * e3::shm::numFhSamples;
			}

			hestInfo[h].hestData.resize(rows.hest);
			srsIqInfo[h].iqData.resize(rows.srs_iq);
			srsInfo[h].rbSnrData.resize(rows.srs);
			srsHestInfo[h].hestData.resize(rows.srs_hest);

			puschInfo[h].pPduData.assign(1, puschInfo[h].pDataAlloc);
		}
	}
};

#endif  // DATA_LAKE_H
