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

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "feeder.hpp"

#include "data_lake.hpp"
#include "e3_agent.hpp"
#include "synth.hpp"

#include <cerrno>
#include <pthread.h>
#include <sched.h>
#include <time.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <thread>
#include <utility>

namespace e3sa {
namespace {

constexpr uint16_t SLOTS_PER_FRAME = 20;  // numerology mu=1
constexpr uint64_t MOD_SLOTS = 100;       // amplitude oscillation period / table depth

// Triangle 0.3..1.0 over MOD_SLOTS so the IQ/H-est level rises and falls.
float scaleForPhase(uint64_t slot) {
	const float ph = float(slot % MOD_SLOTS) / float(MOD_SLOTS);
	return 0.3f + 0.7f * (1.0f - std::fabs(2.0f * ph - 1.0f));
}

uint64_t toNs(const timespec& t) { return uint64_t(t.tv_sec) * 1000000000ull + t.tv_nsec; }

uint64_t nowNs(clockid_t clk) {
	timespec ts;
	clock_gettime(clk, &ts);
	return toNs(ts);
}

void pinThread(int core) {
	if (core < 0) return;
	cpu_set_t set;
	CPU_ZERO(&set);
	CPU_SET(core, &set);
	int rc = pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
	if (rc != 0) {
		std::fprintf(stderr, "feeder: failed to pin thread to core %d: %s\n", core, std::strerror(rc));
	}
}

// Sleep to the next absolute slot deadline; warn if the prior slot overran.
void waitSlot(timespec& next, uint64_t tick_ns, const char* who, uint64_t slot) {
	next.tv_nsec += tick_ns;
	next.tv_sec += next.tv_nsec / 1000000000ull;
	next.tv_nsec %= 1000000000ull;
	if (nowNs(CLOCK_MONOTONIC) > toNs(next)) {  // prior slot's work blew this deadline
		std::fprintf(stderr, "feeder[%s]: slot %llu overran tick budget\n", who, static_cast<unsigned long long>(slot));
	}
	// Resume across EINTR so a stray signal can't cut a slot short.
	while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next, nullptr) == EINTR) {}
}

} // namespace

Feeder::Feeder(DataLake& dl, E3Agent& agent, const SynthCfg& synth, const CpuCfg& cpu, const RowsCfg& rows)
	: dl_(dl), agent_(agent), synth_(synth), cpu_(cpu), rows_(rows) {}

void Feeder::buildTables() {
	fh_table_.resize(MOD_SLOTS * e3::shm::numFhSamples);
	hest_table_.resize(MOD_SLOTS * synth::kHestSamples);
	srs_iq_table_.resize(MOD_SLOTS * synth::kSrsIqSamples);
	srs_hest_table_.resize(MOD_SLOTS * synth::kSrsHestSamples);
	srs_rbsnr_table_.resize(MOD_SLOTS * synth::kSrsRbSnrSamples);
	for (uint64_t i = 0; i < MOD_SLOTS; ++i) {
		const float scale = scaleForPhase(i);
		synth::fillPuschIq(&fh_table_[i * e3::shm::numFhSamples], scale);
		synth::fillPuschHest(&hest_table_[i * synth::kHestSamples], scale);
		synth::fillSrsIq(&srs_iq_table_[i * synth::kSrsIqSamples], scale);
		synth::fillSrsHest(&srs_hest_table_[i * synth::kSrsHestSamples], scale);
		synth::fillSrsRbSnr(&srs_rbsnr_table_[i * synth::kSrsRbSnrSamples], scale);
	}
}

void Feeder::run(std::atomic<bool>& stop) {
	buildTables();  // generate once; the hot loops only memcpy
	timespec epoch;
	clock_gettime(CLOCK_MONOTONIC, &epoch);  // shared so both clocks agree on slot N
	tai_base_ns_ = nowNs(CLOCK_REALTIME);    // TAI = base + abs_slot*tick
	std::thread srs(&Feeder::srsLoop, this, std::ref(stop), epoch);
	puschLoop(stop, epoch);
	srs.join();
}

void Feeder::puschLoop(std::atomic<bool>& stop, timespec next) {
	pinThread(cpu_.feeder_core);
	const uint64_t tick_ns = uint64_t(synth_.slot_tick_us) * 1000;
	const size_t plen = synth_.tdd_pattern.size();
	uint64_t abs_slot = 0;
	Accum fh, pusch, hest;

	while (!stop.load(std::memory_order_relaxed)) {
		waitSlot(next, tick_ns, "pusch", abs_slot);

		if (synth_.tdd_pattern[abs_slot % plen] == 'U') {
			const uint16_t slot = abs_slot % SLOTS_PER_FRAME;
			const uint16_t sfn = (abs_slot / SLOTS_PER_FRAME) % 1024;
			const uint64_t ts = nowNs(CLOCK_REALTIME);
			const uint64_t ph = abs_slot % MOD_SLOTS;  // precomputed-row index

			E3BufferInfo bi;
			bi.sfn = sfn;
			bi.slot = slot;
			bi.timestamp_ns = ts;
			bi.timestamp_tai_ns = tai_base_ns_ + abs_slot * tick_ns;
			synth::fillPusch(bi, synth_.n_cells);

			// PUSCH rows count UE records; all cells of the slot share one base row.
			uint32_t slot_ues = 0;
			for (const auto& cell : bi.cells) slot_ues += cell.ues.size();
			pusch.reserve(slot_ues, rows_.pusch);
			const uint32_t pusch_row = pusch.row;

			// One FH row + one H-est row per cell. Copy the blobs into SHM (lock-free vs
			// dApp reader), then publish all cell indices at once under the lock.
			for (uint16_t c = 0; c < synth_.n_cells; ++c) {
				E3CellInfo& cell = bi.cells[c];

				std::memcpy(dl_.fhInfo[fh.half].fhData[fh.row], &fh_table_[ph * e3::shm::numFhSamples],
					e3::shm::numFhSamples * sizeof(int16_t));
				cell.current_fh_buffer = fh.half;
				cell.fh_write_index = fh.row;
				fh.advance(rows_.fh);

				// H-est packs rows into the active half via a byte cursor (reset on flip).
				hestInfo_t& hi = dl_.hestInfo[hest.half];
				if (hest.row == 0) hi.writeOffsetBytes = 0;
				const uint32_t hest_off = hi.writeOffsetBytes;
				hi.hestData[hest.row] = hi.pDataAlloc + hest_off / sizeof(hestDataType);
				std::memcpy(hi.hestData[hest.row], &hest_table_[ph * synth::kHestSamples],
					synth::kHestSamples * sizeof(hestDataType));
				hi.writeOffsetBytes += synth::kHestSamples * sizeof(hestDataType);
				cell.current_hest_buffer = hest.half;
				cell.hest_write_index = hest.row;
				cell.hest_row_byte_offset = hest_off;
				hest.advance(rows_.hest);

				cell.current_pusch_buffer = pusch.half;
				cell.pusch_write_index = pusch_row;
				for (auto& m : cell.ues) pusch.placePdu(m);
			}

			{
				std::lock_guard<std::mutex> lk(dl_.e3_buffer_mutex);
				dl_.e3_buffer_info = std::move(bi);
			}
			agent_.notifyDataReady();
		}

		++abs_slot;
		if (synth_.duration_slots && abs_slot >= synth_.duration_slots) break;
	}
}

void Feeder::srsLoop(std::atomic<bool>& stop, timespec next) {
	const uint64_t period = uint64_t(synth_.srs_periodicity_ms) * 1000 / synth_.slot_tick_us;
	if (!period) return;  // SRS disabled
	pinThread(cpu_.feeder_core);
	const uint64_t tick_ns = uint64_t(synth_.slot_tick_us) * 1000;
	const size_t plen = synth_.tdd_pattern.size();
	uint64_t abs_slot = 0;
	bool srs_pending = false;
	Accum iq, hest, rb;  // IQ per-cell, H-est + RbSNR per-UE; each accumulates/flips independently

	while (!stop.load(std::memory_order_relaxed)) {
		waitSlot(next, tick_ns, "srs", abs_slot);

		if (abs_slot % period == 0) srs_pending = true;
		if (srs_pending && synth_.tdd_pattern[abs_slot % plen] == 'U') {
			const uint16_t slot = abs_slot % SLOTS_PER_FRAME;
			const uint16_t sfn = (abs_slot / SLOTS_PER_FRAME) % 1024;
			const uint64_t ts = nowNs(CLOCK_REALTIME);
			const uint64_t ph = abs_slot % MOD_SLOTS;

			E3SrsBufferInfo si;
			si.sfn = sfn;
			si.slot = slot;
			si.timestamp_ns = ts;
			si.timestamp_tai_ns = tai_base_ns_ + abs_slot * tick_ns;
			synth::fillSrs(si, synth_.n_cells);

			// H-est and RbSNR rows of all cells share one half; the slot's first row is
			// the write index of every cell.
			hest.reserve(synth_.n_cells, rows_.srs_hest);
			rb.reserve(synth_.n_cells, rows_.srs);
			const uint32_t hest_row = hest.row, rb_row = rb.row;

			// Per cell: one SRS-IQ row + per-UE H-est/RbSNR rows. Copy the blobs into SHM
			// (lock-free vs dApp reader), then publish all cell indices at once under the lock.
			for (uint16_t c = 0; c < synth_.n_cells; ++c) {
				E3SrsCellInfo& cell = si.cells[c];

				srsIqInfo_t& qi = dl_.srsIqInfo[iq.half];
				if (iq.row == 0) qi.writeOffsetBytes = 0;
				const uint32_t iq_off = qi.writeOffsetBytes;
				srs_iq_advance(&qi, iq.row, &srs_iq_table_[ph * synth::kSrsIqSamples],
					synth::kSrsIqSamples * sizeof(int16_t));
				cell.current_srs_iq_buffer = iq.half;
				cell.srs_iq_write_index = iq.row;
				cell.srs_iq_row_byte_offset = iq_off;
				iq.advance(rows_.srs_iq);

				srsHestInfo_t& hi = dl_.srsHestInfo[hest.half];
				if (hest.row == 0) hi.writeOffsetBytes = 0;
				const uint32_t hest_off = hi.writeOffsetBytes;
				srs_hest_advance(&hi, hest.row, &srs_hest_table_[ph * synth::kSrsHestSamples],
					synth::kSrsHestSamples * sizeof(int16_t));
				cell.current_srs_hest_buffer = hest.half;
				cell.srs_hest_write_index = hest_row;
				hest.advance(rows_.srs_hest);

				srsInfo_t& ri = dl_.srsInfo[rb.half];
				if (rb.row == 0) ri.writeOffsetBytes = 0;
				const uint32_t rb_off = ri.writeOffsetBytes;
				srs_rb_snr_advance(&ri, rb.row, &srs_rbsnr_table_[ph * synth::kSrsRbSnrSamples],
					synth::kSrsRbSnrSamples * sizeof(float));
				cell.current_srs_rb_snr_buffer = rb.half;
				cell.srs_rb_snr_write_index = rb_row;
				rb.advance(rows_.srs);

				E3SrsUeMetrics& m = cell.ues[0];
				m.srs_hest_offset = hest_off;
				m.srs_hest_size = synth::kSrsHestSamples * sizeof(int16_t);
				m.srs_rb_snr_offset = rb_off;
				m.srs_rb_snr_size = synth::kSrsRbSnrSamples * sizeof(float);
			}

			{
				std::lock_guard<std::mutex> lk(dl_.e3_srs_buffer_mutex);
				dl_.e3_srs_buffer_info = std::move(si);
			}
			agent_.notifySrsDataReady();
			srs_pending = false;
		}

		++abs_slot;
		if (synth_.duration_slots && abs_slot >= synth_.duration_slots) break;
	}
}

} // namespace e3sa
