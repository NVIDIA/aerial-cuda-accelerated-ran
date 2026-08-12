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

// Entry point: parse config, bring up E3Agent + SHM, run the slot feeder.

#include "config.hpp"
#include "data_lake.hpp"
#include "e3_agent.hpp"
#include "feeder.hpp"
#include "replay.hpp"

#include <sys/mman.h>

#include "nvlog.hpp"

#include <atomic>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace {
constexpr char SHM_KEY[] = "/e3_ran_buffers";  // fixed E3 key (matches E3Agent)
std::atomic<bool> g_stop{false};
void onSignal(int) { g_stop.store(true); }

e3shim::Level toLevel(const std::string& s) {
	if (s == "VRB") return e3shim::VRB;
	if (s == "DBG") return e3shim::DBG;
	if (s == "WRN") return e3shim::WRN;
	if (s == "ERR") return e3shim::ERR;
	if (s == "FAT") return e3shim::FAT;
	return e3shim::INF;
}
}

int main(int argc, char** argv) {
	const char* cfg_path = nullptr;
	for (int i = 1; i < argc; ++i) {
		if (std::strcmp(argv[i], "--config") == 0 && i + 1 < argc) cfg_path = argv[++i];
	}
	if (!cfg_path) {
		std::fprintf(stderr, "usage: %s --config <path>\n", argv[0]);
		return 2;
	}

	e3sa::Config cfg;
	try {
		cfg = e3sa::Config::load(cfg_path);
	} catch (const std::exception& e) {
		std::fprintf(stderr, "%s\n", e.what());
		return 1;
	}

	e3shim::setThreshold(toLevel(cfg.log_level));
	e3shim::setLogFile("/logs/e3agent_standalone.log");

	std::signal(SIGTERM, onSignal);
	std::signal(SIGINT, onSignal);

	shm_unlink(SHM_KEY);  // clear stale SHM from a prior run

	DataLake dl;
	dl.initBuffers();

	E3Agent agent(&dl,
		cfg.agent.rep_port, cfg.agent.pub_port, cfg.agent.sub_port,
		static_cast<int>(cfg.rows.fh), static_cast<int>(cfg.rows.pusch), static_cast<int>(cfg.rows.hest),
		e3::shm::numFhSamples, e3::shm::maxPuschPduSize, e3::shm::maxHestSamplesPerRow,
		static_cast<int>(cfg.rows.srs_iq), static_cast<int>(cfg.rows.srs), static_cast<int>(cfg.rows.srs_hest),
		e3::shm::maxSrsIqSamplesPerRow, e3::shm::maxSrsHestBytesPerRow, e3::shm::maxSrsRbSnrBytesPerRow);

	if (!agent.createSharedMemoryBuffers(&dl.pFh, &dl.pInsertFh, &dl.p, &dl.pInsertPusch,
			&dl.pHest, &dl.pInsertHest, &dl.pSrsIq, &dl.pInsertSrsIq,
			&dl.pSrs, &dl.pInsertSrs, &dl.pSrsHest, &dl.pInsertSrsHest)) {
		std::fprintf(stderr, "failed to create SHM buffers\n");
		shm_unlink(SHM_KEY);
		return 1;
	}

	dl.initRowPointers(cfg.rows);

	if (!agent.init()) {
		std::fprintf(stderr, "E3Agent init failed (ports %u/%u/%u busy?)\n",
			cfg.agent.rep_port, cfg.agent.pub_port, cfg.agent.sub_port);
		agent.shutdown();
		shm_unlink(SHM_KEY);
		return 1;
	}

	if (cfg.mode == e3sa::Mode::Replay) {
		std::printf("e3agent-standalone: replay on ports %u/%u/%u, path=%s\n",
			cfg.agent.rep_port, cfg.agent.pub_port, cfg.agent.sub_port, cfg.replay.path.c_str());
		e3sa::Replay replay(dl, agent, cfg.replay, cfg.cpu, cfg.rows);
		replay.run(g_stop);
	} else {
		std::printf("e3agent-standalone: synth on ports %u/%u/%u, tdd=%s\n",
			cfg.agent.rep_port, cfg.agent.pub_port, cfg.agent.sub_port, cfg.synth.tdd_pattern.c_str());
		e3sa::Feeder feeder(dl, agent, cfg.synth, cfg.cpu, cfg.rows);
		feeder.run(g_stop);
	}

	agent.shutdown();
	shm_unlink(SHM_KEY);
	return 0;
}
