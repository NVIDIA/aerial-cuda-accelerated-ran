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

// Harness configuration parsed from YAML.

#ifndef E3SA_CONFIG_HPP
#define E3SA_CONFIG_HPP

#include <cstdint>
#include <string>

namespace e3sa {

enum class Mode { Synth, Replay };

struct AgentCfg {
	uint16_t rep_port = 5555;
	uint16_t pub_port = 5556;
	uint16_t sub_port = 5557;
};

// SHM ring depths (num_rows_*).
struct RowsCfg {
	uint32_t fh = 120;
	uint32_t pusch = 180;
	uint32_t hest = 140;
	uint32_t srs_iq = 40;
	uint32_t srs = 70;
	uint32_t srs_hest = 90;
};

struct SynthCfg {
	uint32_t slot_tick_us = 500;          // mu=1 (30 kHz SCS)
	std::string tdd_pattern = "DDDSU";    // D=downlink, S=special, U=uplink(PUSCH)
	uint32_t srs_periodicity_ms = 40;     // SRS rides a U slot
	uint64_t duration_slots = 0;          // 0 = forever
};

struct ReplayCfg {
	std::string path;
	uint64_t loops = 0;  // 0 = forever
};

struct CpuCfg {
	int feeder_core = -1;  // pins both feeder threads; -1 = no pinning
};

struct Config {
	Mode mode = Mode::Synth;
	std::string log_level = "INF";  // VRB | DBG | INF | WRN | ERR | FAT
	AgentCfg agent;
	RowsCfg rows;
	CpuCfg cpu;
	SynthCfg synth;
	ReplayCfg replay;

	// Throws std::runtime_error on any invalid or unknown field.
	static Config load(const std::string& path);
};

} // namespace e3sa

#endif // E3SA_CONFIG_HPP
