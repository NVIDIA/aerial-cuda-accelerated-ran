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
#include "replay.hpp"

#include "data_lake.hpp"
#include "e3_agent.hpp"
#include "nvlog.hpp"
#include "replay_format.hpp"

#include <pthread.h>
#include <sched.h>
#include <time.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace e3sa {
namespace rt = trace;
namespace {

constexpr int64_t MAX_PACE_NS = 1000000000;  // clamp ts_tai delta to ~1 s

// Worst-case record body
constexpr uint64_t kMaxUeRows = 0xFFFF;  // uint16 n_ue / n_srs_ue ceiling
constexpr uint64_t kMaxPuschBody =
	sizeof(rt::PuschSlotHeader)
	+ kMaxUeRows * sizeof(rt::PuschUeMetrics)
	+ 2 * sizeof(rt::BlobHeader)
	+ uint64_t(e3::shm::numFhSamples) * sizeof(int16_t)
	+ uint64_t(e3::shm::maxHestSamplesPerRow) * sizeof(hestDataType);
constexpr uint64_t kMaxSrsBody =
	sizeof(rt::SrsSlotHeader)
	+ kMaxUeRows * (sizeof(rt::SrsUeMetrics) + 2 * sizeof(rt::BlobHeader))
	+ sizeof(rt::BlobHeader)
	+ uint64_t(e3::shm::maxSrsIqSamplesPerRow) * sizeof(int16_t)
	+ e3::shm::maxSrsHestBytesPerRow + e3::shm::maxSrsRbSnrBytesPerRow;
constexpr uint32_t MAX_RECORD_BYTES =
	uint32_t(kMaxPuschBody > kMaxSrsBody ? kMaxPuschBody : kMaxSrsBody);

void pinThread(int core) {
	if (core < 0) return;
	cpu_set_t set;
	CPU_ZERO(&set);
	CPU_SET(core, &set);
	int rc = pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
	if (rc != 0) {
		std::fprintf(stderr, "replay: failed to pin thread to core %d: %s\n", core, std::strerror(rc));
	}
}

void addNs(timespec& t, uint64_t ns) {
	const uint64_t nsec = uint64_t(t.tv_nsec) + ns;
	t.tv_sec += time_t(nsec / 1000000000ull);
	t.tv_nsec = long(nsec % 1000000000ull);
}

// Bounds-checked walk over a record body.
struct Cursor {
	const uint8_t* p;
	const uint8_t* end;
	template<class T> bool get(T& v) {
		if (size_t(end - p) < sizeof(T)) return false;
		std::memcpy(&v, p, sizeof(T));
		p += sizeof(T);
		return true;
	}
	const uint8_t* take(uint32_t n) {
		if (size_t(end - p) < n) return nullptr;
		const uint8_t* r = p;
		p += n;
		return r;
	}
};

struct PuschRings { Accum fh, pusch, hest; };
struct SrsRings   { Accum iq, hest, rb; };

// Sequential reader over the trace: validates the file header once, then
// yields one (tag + body) per next(). Returns false at clean EOF or truncation.
class TraceReader {
public:
	TraceReader() = default;
	TraceReader(const TraceReader&) = delete;
	TraceReader& operator=(const TraceReader&) = delete;

	bool open(const std::string& path) {
		f_ = std::fopen(path.c_str(), "rb");
		if (!f_) return false;
		rt::FileHeader h{};
		if (std::fread(&h, sizeof(h), 1, f_) != 1 || h.magic != rt::MAGIC) {
			close();
			return false;
		}
		const uint32_t cv = h.version, sv = h.shm_layout_version;  // copy out of packed struct for fmt
		if (cv != rt::VERSION) {
			e3shim::log(e3shim::WRN, "WRN", 0,
				fmt::format("replay: trace version 0x{:06X} != expected 0x{:06X}", cv, rt::VERSION));
		}
		if (sv != e3::SHM_LAYOUT_VERSION) {
			e3shim::log(e3shim::WRN, "WRN", 0,
				fmt::format("replay: SHM layout 0x{:06X} != expected 0x{:06X}",
					sv, uint32_t(e3::SHM_LAYOUT_VERSION)));
		}
		return true;
	}
	bool next(uint16_t& tag, std::vector<uint8_t>& buf) {
		if (!f_) return false;
		rt::RecordHeader rh{};
		if (std::fread(&rh, sizeof(rh), 1, f_) != 1) return false;
		if (rh.payload_len > MAX_RECORD_BYTES) {
			const uint32_t pl = rh.payload_len;
			e3shim::log(e3shim::WRN, "WRN", 0,
				fmt::format("replay: record payload_len {} exceeds cap; stopping", pl));
			return false;
		}
		buf.resize(rh.payload_len);
		if (rh.payload_len && std::fread(buf.data(), 1, rh.payload_len, f_) != rh.payload_len) return false;
		tag = rh.tag;
		return true;
	}
	void close() { if (f_) { std::fclose(f_); f_ = nullptr; } }
	~TraceReader() { close(); }
private:
	FILE* f_ = nullptr;
};

bool writePusch(DataLake& dl, E3Agent& agent, const RowsCfg& rows, PuschRings& r,
                const uint8_t* body, size_t len) {
	Cursor c{body, body + len};
	rt::PuschSlotHeader hdr{};
	if (!c.get(hdr)) return false;
	std::vector<rt::PuschUeMetrics> ue(hdr.n_ue);
	for (auto& u : ue) if (!c.get(u)) return false;
	rt::BlobHeader hb{}, fb{};
	if (!c.get(fb)) return false;
	const uint8_t* fd = c.take(fb.len);
	if (fb.len && !fd) return false;
	if (!c.get(hb)) return false;
	const uint8_t* hd = c.take(hb.len);
	if (hb.len && !hd) return false;

	const uint32_t helems = hb.len / sizeof(hestDataType);
	if (helems > e3::shm::maxHestSamplesPerRow) {
		const uint16_t sfn = hdr.sfn, slot = hdr.slot;  // copy out of packed struct for fmt
		e3shim::log(e3shim::WRN, "WRN", 0,
			fmt::format("replay: PUSCH sfn={} slot={} H-est row exceeds cap; skipped", sfn, slot));
		return false;
	}

	// FH IQ row (whole row); absent or short => zero the rest so stale data never leaks.
	const size_t fhbytes = size_t(e3::shm::numFhSamples) * sizeof(int16_t);
	int16_t* fhrow = dl.fhInfo[r.fh.half].fhData[r.fh.row];
	if (fd && fb.len) {
		const size_t n = std::min<size_t>(fb.len, fhbytes);
		std::memcpy(fhrow, fd, n);
		if (n < fhbytes) std::memset(reinterpret_cast<uint8_t*>(fhrow) + n, 0, fhbytes - n);
	} else {
		std::memset(fhrow, 0, fhbytes);
	}

	// H-est row packed via byte cursor (reset on flip); UEs index it by h_offset.
	hestInfo_t& hi = dl.hestInfo[r.hest.half];
	if (r.hest.row == 0) hi.writeOffsetBytes = 0;
	const uint32_t rowbase = hi.writeOffsetBytes;
	hi.hestData[r.hest.row] = hi.pDataAlloc + rowbase / sizeof(hestDataType);
	if (hd && hb.len) std::memcpy(hi.hestData[r.hest.row], hd, hb.len);
	hi.writeOffsetBytes += hb.len;

	{
		std::lock_guard<std::mutex> lk(dl.e3_buffer_mutex);
		E3BufferInfo& bi = dl.e3_buffer_info;
		bi = {};
		bi.current_fh_buffer = r.fh.half;       bi.fh_write_index = r.fh.row;
		bi.current_pusch_buffer = r.pusch.half; bi.pusch_write_index = r.pusch.row;
		bi.current_hest_buffer = r.hest.half;   bi.hest_write_index = r.hest.row;
		bi.hest_row_byte_offset = rowbase;
		bi.sfn = hdr.sfn;
		bi.slot = hdr.slot;
		bi.timestamp_ns = hdr.timestamp_ns;
		bi.timestamp_tai_ns = hdr.timestamp_tai_ns;
		bi.cell_id = hdr.cell_id;
		bi.n_rx_ant = hdr.n_rx_ant;
		bi.n_rx_ant_srs = hdr.n_rx_ant_srs;
		bi.n_cells = hdr.n_cells;
		bi.n_bs_ants = hdr.n_bs_ants;
		bi.n_ue = hdr.n_ue;
		bi.ue_metrics.resize(hdr.n_ue);
		for (uint16_t i = 0; i < hdr.n_ue; ++i) {
			const rt::PuschUeMetrics& s = ue[i];
			E3UeMetrics& m = bi.ue_metrics[i];
			m.rnti = s.rnti; m.tb_crc_fail = s.tb_crc_fail; m.cb_errors = s.cb_errors;
			m.rsrp = s.rsrp; m.noise_var = s.noise_var; m.sinr = s.sinr; m.cb_count = s.cb_count;
			m.rssi = s.rssi; m.qam_mod_order = s.qam_mod_order; m.mcs_index = s.mcs_index;
			m.mcs_table_index = s.mcs_table_index; m.rb_start = s.rb_start; m.rb_size = s.rb_size;
			m.start_symbol_index = s.start_symbol_index; m.nr_of_symbols = s.nr_of_symbols;
			m.tb_size = s.tb_size; m.pdu_len = s.pdu_len; m.target_code_rate = s.target_code_rate;
			m.new_data_indicator = s.new_data_indicator; m.n_layers = s.n_layers;
			m.layer_offset = s.layer_offset; m.ue_grp_idx = s.ue_grp_idx;
			m.h_offset = s.h_offset; m.h_size = s.h_size; m.n_subcarriers = s.n_subcarriers;
			m.n_dmrs_estimates = s.n_dmrs_estimates; m.dmrs_symb_pos = s.dmrs_symb_pos;
			m.timing_advance = s.timing_advance; m.cfo_hz = s.cfo_hz;
			m.harq_process_id = s.harq_process_id; m.rv_index = s.rv_index;
		}
	}
	agent.notifyDataReady();
	r.fh.advance(rows.fh);
	r.pusch.advance(rows.pusch);
	r.hest.advance(rows.hest);
	return true;
}

bool writeSrs(DataLake& dl, E3Agent& agent, const RowsCfg& rows, SrsRings& r,
              const uint8_t* body, size_t len) {
	Cursor c{body, body + len};
	rt::SrsSlotHeader hdr{};
	if (!c.get(hdr)) return false;
	std::vector<rt::SrsUeMetrics> ue(hdr.n_srs_ue);
	for (auto& u : ue) if (!c.get(u)) return false;

	rt::BlobHeader iqb{};
	if (!c.get(iqb)) return false;
	const uint8_t* iqd = c.take(iqb.len);
	if (iqb.len && !iqd) return false;

	// Per-UE (hest, rb_snr) blob pairs in UE order.
	struct UeBlob { rt::BlobHeader hb, rb; const uint8_t* hd; const uint8_t* rd; };
	std::vector<UeBlob> ub(hdr.n_srs_ue);
	size_t hsum = 0, rsum = 0;
	for (auto& b : ub) {
		if (!c.get(b.hb)) return false;
		b.hd = c.take(b.hb.len);
		if (b.hb.len && !b.hd) return false;
		if (!c.get(b.rb)) return false;
		b.rd = c.take(b.rb.len);
		if (b.rb.len && !b.rd) return false;
		hsum += b.hb.len;
		rsum += b.rb.len;
	}
	if (iqb.len > size_t(e3::shm::maxSrsIqSamplesPerRow) * sizeof(int16_t)
	    || hsum > e3::shm::maxSrsHestBytesPerRow || rsum > e3::shm::maxSrsRbSnrBytesPerRow) {
		const uint16_t sfn = hdr.sfn, slot = hdr.slot;  // copy out of packed struct for fmt
		e3shim::log(e3shim::WRN, "WRN", 0,
			fmt::format("replay: SRS sfn={} slot={} row exceeds cap; skipped", sfn, slot));
		return false;
	}

	// IQ: one per-cell blob.
	srsIqInfo_t& qi = dl.srsIqInfo[r.iq.half];
	if (r.iq.row == 0) qi.writeOffsetBytes = 0;
	const uint32_t iq_off = qi.writeOffsetBytes;
	srs_iq_advance(&qi, r.iq.row, iqd, iqb.len);

	// H-est + RbSNR: per-UE blobs concatenated into one row; capture byte offsets.
	srsHestInfo_t& shi = dl.srsHestInfo[r.hest.half];
	if (r.hest.row == 0) shi.writeOffsetBytes = 0;
	srsInfo_t& ri = dl.srsInfo[r.rb.half];
	if (r.rb.row == 0) ri.writeOffsetBytes = 0;
	std::vector<uint32_t> hest_off(hdr.n_srs_ue), rb_off(hdr.n_srs_ue);
	for (uint16_t i = 0; i < hdr.n_srs_ue; ++i) {
		hest_off[i] = shi.writeOffsetBytes;
		srs_hest_advance(&shi, r.hest.row, ub[i].hd, ub[i].hb.len);
		rb_off[i] = ri.writeOffsetBytes;
		srs_rb_snr_advance(&ri, r.rb.row, ub[i].rd, ub[i].rb.len);
	}

	{
		std::lock_guard<std::mutex> lk(dl.e3_srs_buffer_mutex);
		E3SrsBufferInfo& si = dl.e3_srs_buffer_info;
		si = {};
		si.current_srs_iq_buffer = r.iq.half;       si.srs_iq_write_index = r.iq.row + 1;
		si.current_srs_hest_buffer = r.hest.half;   si.srs_hest_write_index = r.hest.row + 1;
		si.current_srs_rb_snr_buffer = r.rb.half;   si.srs_rb_snr_write_index = r.rb.row + 1;
		si.srs_iq_row_byte_offset = iq_off;
		si.sfn = hdr.sfn;
		si.slot = hdr.slot;
		si.timestamp_ns = hdr.timestamp_ns;
		si.timestamp_tai_ns = hdr.timestamp_tai_ns;
		si.cell_id = hdr.cell_id;
		si.n_cells = hdr.n_cells;
		si.n_rx_ant_srs = hdr.n_rx_ant_srs;
		si.srs_cell_start_sym = hdr.srs_cell_start_sym;
		si.srs_cell_n_srs_sym = hdr.srs_cell_n_srs_sym;
		si.n_srs_ue = hdr.n_srs_ue;
		si.ue_metrics.resize(hdr.n_srs_ue);
		for (uint16_t i = 0; i < hdr.n_srs_ue; ++i) {
			const rt::SrsUeMetrics& s = ue[i];
			E3SrsUeMetrics& m = si.ue_metrics[i];
			m.rnti = s.rnti; m.wideband_snr = s.wideband_snr; m.signal_energy = s.signal_energy;
			m.noise_energy = s.noise_energy; m.toa_us = s.toa_us; m.hd_ant_flag = s.hd_ant_flag;
			m.sc_corr_re = s.sc_corr_re; m.sc_corr_im = s.sc_corr_im; m.cs_corr_ratio_db = s.cs_corr_ratio_db;
			m.n_ant_ports = s.n_ant_ports; m.n_syms = s.n_syms; m.n_repetitions = s.n_repetitions;
			m.comb_size = s.comb_size; m.comb_offset = s.comb_offset; m.start_sym = s.start_sym;
			m.cyclic_shift = s.cyclic_shift; m.frequency_position = s.frequency_position;
			m.frequency_shift = s.frequency_shift; m.frequency_hopping = s.frequency_hopping;
			m.resource_type = s.resource_type; m.t_srs = s.t_srs; m.t_offset = s.t_offset;
			m.usage = s.usage; m.n_valid_prg = s.n_valid_prg; m.prg_size = s.prg_size;
			m.n_prb_grps = s.n_prb_grps;
			m.srs_hest_offset = hest_off[i];   m.srs_hest_size = ub[i].hb.len;
			m.srs_rb_snr_offset = rb_off[i];   m.srs_rb_snr_size = ub[i].rb.len;
		}
	}
	agent.notifySrsDataReady();
	r.iq.advance(rows.srs_iq);
	r.hest.advance(rows.srs_hest);
	r.rb.advance(rows.srs);
	return true;
}

} // namespace

Replay::Replay(DataLake& dl, E3Agent& agent, const ReplayCfg& cfg, const CpuCfg& cpu,
               const RowsCfg& rows)
	: dl_(dl), agent_(agent), cfg_(cfg), cpu_(cpu), rows_(rows) {}

void Replay::run(std::atomic<bool>& stop) {
	pinThread(cpu_.feeder_core);
	// Rings persist across loops, mirroring a continuous live stream
	PuschRings pr;
	SrsRings sr;
	for (uint64_t pass = 0; (cfg_.loops == 0 || pass < cfg_.loops) && !stop.load(); ++pass) {
		TraceReader rd;
		if (!rd.open(cfg_.path)) {
			e3shim::log(e3shim::ERR, "ERR", 0, fmt::format("replay: no readable trace {}", cfg_.path));
			return;
		}

		timespec next;
		clock_gettime(CLOCK_MONOTONIC, &next);
		uint64_t prev_tai = 0, emitted = 0;
		bool have_prev = false;

		uint16_t tag = 0;
		std::vector<uint8_t> buf;
		while (rd.next(tag, buf) && !stop.load(std::memory_order_relaxed)) {
			if (buf.size() < sizeof(uint64_t) * 2) continue;  // need timestamp fields
			uint64_t tai;
			std::memcpy(&tai, buf.data() + sizeof(uint64_t), sizeof(tai));  // timestamp_tai_ns
			if (have_prev) {
				int64_t d = int64_t(tai - prev_tai);
				if (d < 0) d = 0;
				if (d > MAX_PACE_NS) d = MAX_PACE_NS;
				if (d > 0) addNs(next, uint64_t(d));
			}
			have_prev = true;
			prev_tai = tai;
			clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next, nullptr);

			bool ok = false;
			if (tag == rt::TAG_PUSCH) ok = writePusch(dl_, agent_, rows_, pr, buf.data(), buf.size());
			else if (tag == rt::TAG_SRS) ok = writeSrs(dl_, agent_, rows_, sr, buf.data(), buf.size());
			if (ok) ++emitted;
		}
		rd.close();
		e3shim::log(e3shim::INF, "INF", 0,
			fmt::format("replay: pass {} emitted {} records", pass, emitted));
		if (emitted == 0) return;  // empty/garbage trace: don't spin forever
	}
}

} // namespace e3sa
