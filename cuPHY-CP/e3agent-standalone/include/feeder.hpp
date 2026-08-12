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

// Slot-timed driver that fills DataLake buffers and notifies the agent.

#ifndef E3SA_FEEDER_HPP
#define E3SA_FEEDER_HPP

#include "config.hpp"
#include "e3_types.hpp"

#include <atomic>
#include <ctime>
#include <vector>

class DataLake;
class E3Agent;

namespace e3sa {

// Two slot clocks sharing one monotonic epoch (so sfn/slot stay aligned):
// PUSCH on every U slot, SRS on U slots at the configured periodicity.
class Feeder {
public:
	Feeder(DataLake& dl, E3Agent& agent, const SynthCfg& synth, const CpuCfg& cpu, const RowsCfg& rows);
	void run(std::atomic<bool>& stop);

private:
	void buildTables();  // precompute per-phase FH/H-est rows (gen once, memcpy per slot)
	void puschLoop(std::atomic<bool>& stop, timespec next);
	void srsLoop(std::atomic<bool>& stop, timespec next);

	DataLake& dl_;
	E3Agent& agent_;
	SynthCfg synth_;
	CpuCfg cpu_;
	RowsCfg rows_;

	uint64_t tai_base_ns_{};  // anchor for slot-aligned synthetic TAI (base + abs_slot*tick)

	std::vector<int16_t> fh_table_;         // N rows x numFhSamples
	std::vector<hestDataType> hest_table_;  // N rows x kHestSamples
	std::vector<int16_t> srs_iq_table_;     // N rows x kSrsIqSamples
	std::vector<int16_t> srs_hest_table_;   // N rows x kSrsHestSamples
	std::vector<float> srs_rbsnr_table_;    // N rows x kSrsRbSnrSamples
};

} // namespace e3sa

#endif // E3SA_FEEDER_HPP
