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
#include <cstddef>
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

// Returns false if stopped before any dApp subscribed.
bool waitForSubscriber(E3Agent& agent, const std::atomic<bool>& stop) {
	if (agent.activeSubscriptionCount() > 0) return true;
	e3shim::log(e3shim::INF, "INF", 0, "replay: waiting for an E3 subscription");
	const timespec poll{0, 20000000};
	while (!stop.load(std::memory_order_relaxed)) {
		if (agent.activeSubscriptionCount() > 0) return true;
		nanosleep(&poll, nullptr);
	}
	return false;
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
	template<class T> bool get(T& v, size_t n = sizeof(T)) {
		if (size_t(end - p) < n) return false;
		std::memcpy(&v, p, n);
		p += n;
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

// A slot assembled from cells sharing one ts_tai; hdr_n_cells is advisory.
// *_row is the slot's first row in a ring whose rows all cells of the slot share.
struct PendingPusch { E3BufferInfo bi; bool active = false; uint16_t hdr_n_cells = 0; uint32_t pusch_row = 0; };
struct PendingSrs   { E3SrsBufferInfo si; bool active = false; uint16_t hdr_n_cells = 0; uint32_t hest_row = 0, rb_row = 0; };

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
		if (cv != rt::VERSION && cv != rt::VERSION_1_1) {
			e3shim::log(e3shim::ERR, "ERR", 0,
				fmt::format("replay: trace version 0x{:06X} unsupported, expected 0x{:06X} or 0x{:06X}",
					cv, rt::VERSION, rt::VERSION_1_1));
			close();
			return false;
		}
		v1_1_ = cv == rt::VERSION_1_1;
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
	bool v1_1() const { return v1_1_; }
	~TraceReader() { close(); }
private:
	FILE* f_ = nullptr;
	bool v1_1_ = false;
};

// A slot needing more rows than the shallowest ring would wrap and clobber earlier
// cells still pending; fail fast so rows.* is sized to the capture's max rows per
// slot (cells for per-cell rings, UEs for the PUSCH, SRS H-est and RbSNR rings).
bool ringOverflow(size_t need, uint32_t cap, const char* kind, std::atomic<bool>& stop) {
	if (need <= cap) return false;
	e3shim::log(e3shim::ERR, "ERR", 0, fmt::format(
		"replay: {} slot exceeds ring depth {}; increase rows.* to the capture's "
		"max rows per slot", kind, cap));
	stop.store(true);
	return true;
}

// Parse one PUSCH record, stage its blobs in SHM, and append its cell to the slot.
bool accumulatePusch(DataLake& dl, const RowsCfg& rows, PuschRings& r,
                     PendingPusch& pend, const uint8_t* body, size_t len, bool v1_1, std::atomic<bool>& stop) {
	Cursor c{body, body + len};
	rt::PuschSlotHeader hdr{};
	if (!c.get(hdr, v1_1 ? offsetof(rt::PuschSlotHeader, n_prb_ul_bwp) : sizeof(hdr))) return false;
	if (v1_1) hdr.n_prb_ul_bwp = rt::V1_1_N_PRB_UL_BWP;
	std::vector<rt::PuschUeMetrics> ue(hdr.n_ue);
	for (auto& u : ue) if (!c.get(u)) return false;
	if (pend.active) {
		size_t ue_rows = hdr.n_ue;
		for (const auto& cl : pend.bi.cells) ue_rows += cl.ues.size();
		if (ringOverflow(pend.bi.cells.size() + 1, std::min(rows.fh, rows.hest), "PUSCH", stop)
		    || ringOverflow(ue_rows, rows.pusch, "PUSCH", stop))
			return false;
	}
	rt::BlobHeader hb{}, fb{};
	if (!c.get(fb)) return false;
	const uint8_t* fd = c.take(fb.len);
	if (fb.len && !fd) return false;
	if (!c.get(hb)) return false;
	const uint8_t* hd = c.take(hb.len);
	if (hb.len && !hd) return false;

	const uint32_t helems = hb.len / sizeof(hestDataType);
	if (helems > e3::shm::maxHestSamplesPerRow || hdr.n_ue > rows.pusch) {
		const uint16_t sfn = hdr.sfn, slot = hdr.slot;  // copy out of packed struct for fmt
		e3shim::log(e3shim::WRN, "WRN", 0,
			fmt::format("replay: PUSCH sfn={} slot={} row exceeds cap; skipped", sfn, slot));
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

	// PUSCH rows count UE records; a cell whose UEs do not fit starts the next half.
	const uint8_t pusch_half = r.pusch.half;
	r.pusch.reserve(hdr.n_ue, rows.pusch);
	if (!pend.active || r.pusch.half != pusch_half) pend.pusch_row = r.pusch.row;

	if (!pend.active) {  // first cell fixes the slot-level fields
		pend.bi = {};
		pend.bi.sfn = hdr.sfn;
		pend.bi.slot = hdr.slot;
		pend.bi.timestamp_ns = hdr.timestamp_ns;
		pend.bi.timestamp_tai_ns = hdr.timestamp_tai_ns;
		pend.hdr_n_cells = hdr.n_cells;
		pend.active = true;
	}

	pend.bi.cells.emplace_back();
	E3CellInfo& cell = pend.bi.cells.back();
	cell.current_fh_buffer = r.fh.half;       cell.fh_write_index = r.fh.row;
	cell.current_pusch_buffer = r.pusch.half; cell.pusch_write_index = pend.pusch_row;
	cell.current_hest_buffer = r.hest.half;   cell.hest_write_index = r.hest.row;
	cell.hest_row_byte_offset = rowbase;
	cell.cell_id = hdr.cell_id;
	cell.n_rx_ant = hdr.n_rx_ant;
	cell.n_rx_ant_srs = hdr.n_rx_ant_srs;
	cell.n_prb_ul_bwp = hdr.n_prb_ul_bwp;
	cell.n_bs_ants = hdr.n_bs_ants;
	cell.n_ue = hdr.n_ue;
	cell.ues.resize(hdr.n_ue);
	for (uint16_t i = 0; i < hdr.n_ue; ++i) {
		const rt::PuschUeMetrics& s = ue[i];
		E3UeMetrics& m = cell.ues[i];
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
		r.pusch.placePdu(m);
	}

	r.fh.advance(rows.fh);
	r.hest.advance(rows.hest);
	return true;
}

// Publish the accumulated PUSCH slot and reset the pending buffer.
void publishPusch(DataLake& dl, E3Agent& agent, PendingPusch& pend, bool& warn) {
	pend.bi.n_cells = uint16_t(pend.bi.cells.size());
	if (warn && pend.hdr_n_cells && pend.hdr_n_cells != pend.bi.n_cells) {
		e3shim::log(e3shim::WRN, "WRN", 0, fmt::format(
			"replay: PUSCH slot n_cells hdr={} != {} accumulated", pend.hdr_n_cells, pend.bi.n_cells));
		warn = false;
	}
	{
		std::lock_guard<std::mutex> lk(dl.e3_buffer_mutex);
		dl.e3_buffer_info = std::move(pend.bi);
	}
	agent.notifyDataReady();
	pend.bi = {};
	pend.active = false;
}

// Parse one SRS record, stage its blobs in SHM, and append its cell to the slot.
bool accumulateSrs(DataLake& dl, const RowsCfg& rows, SrsRings& r,
                   PendingSrs& pend, const uint8_t* body, size_t len, bool v1_1, std::atomic<bool>& stop) {
	Cursor c{body, body + len};
	rt::SrsSlotHeader hdr{};
	if (!c.get(hdr)) return false;
	if (v1_1) hdr.n_prb_ul_bwp = rt::V1_1_N_PRB_UL_BWP;
	std::vector<rt::SrsUeMetrics> ue(hdr.n_srs_ue);
	for (auto& u : ue) if (!c.get(u, v1_1 ? offsetof(rt::SrsUeMetrics, sequence_id) : sizeof(u))) return false;
	if (pend.active) {
		size_t ue_rows = hdr.n_srs_ue;
		for (const auto& cl : pend.si.cells) ue_rows += cl.ues.size();
		if (ringOverflow(pend.si.cells.size() + 1, rows.srs_iq, "SRS", stop)
		    || ringOverflow(ue_rows, std::min(rows.srs_hest, rows.srs), "SRS", stop))
			return false;
	}

	rt::BlobHeader iqb{};
	if (!c.get(iqb)) return false;
	const uint8_t* iqd = c.take(iqb.len);
	if (iqb.len && !iqd) return false;

	// Per-UE (hest, rb_snr) blob pairs in UE order.
	struct UeBlob { rt::BlobHeader hb, rb; const uint8_t* hd; const uint8_t* rd; };
	std::vector<UeBlob> ub(hdr.n_srs_ue);
	uint32_t hmax = 0, rmax = 0;
	for (auto& b : ub) {
		if (!c.get(b.hb)) return false;
		b.hd = c.take(b.hb.len);
		if (b.hb.len && !b.hd) return false;
		if (!c.get(b.rb)) return false;
		b.rd = c.take(b.rb.len);
		if (b.rb.len && !b.rd) return false;
		hmax = std::max(hmax, b.hb.len);
		rmax = std::max(rmax, b.rb.len);
	}
	if (iqb.len > size_t(e3::shm::maxSrsIqSamplesPerRow) * sizeof(int16_t)
	    || hmax > e3::shm::maxSrsHestBytesPerRow || rmax > e3::shm::maxSrsRbSnrBytesPerRow
	    || hdr.n_srs_ue > std::min(rows.srs_hest, rows.srs)) {
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

	// H-est + RbSNR: one row per UE, as in the live Data Lake. A cell's UE rows
	// must share one buffer half, so switch halves first if they do not fit.
	const uint8_t hest_half = r.hest.half, rb_half = r.rb.half;
	r.hest.reserve(hdr.n_srs_ue, rows.srs_hest);
	r.rb.reserve(hdr.n_srs_ue, rows.srs);
	if (!pend.active || r.hest.half != hest_half) pend.hest_row = r.hest.row;
	if (!pend.active || r.rb.half != rb_half) pend.rb_row = r.rb.row;
	srsHestInfo_t& shi = dl.srsHestInfo[r.hest.half];
	if (r.hest.row == 0) shi.writeOffsetBytes = 0;
	srsInfo_t& ri = dl.srsInfo[r.rb.half];
	if (r.rb.row == 0) ri.writeOffsetBytes = 0;
	std::vector<uint32_t> hest_off(hdr.n_srs_ue), rb_off(hdr.n_srs_ue);
	for (uint16_t i = 0; i < hdr.n_srs_ue; ++i) {
		hest_off[i] = shi.writeOffsetBytes;
		srs_hest_advance(&shi, r.hest.row++, ub[i].hd, ub[i].hb.len);
		rb_off[i] = ri.writeOffsetBytes;
		srs_rb_snr_advance(&ri, r.rb.row++, ub[i].rd, ub[i].rb.len);
	}

	if (!pend.active) {  // first cell fixes the slot-level fields
		pend.si = {};
		pend.si.sfn = hdr.sfn;
		pend.si.slot = hdr.slot;
		pend.si.timestamp_ns = hdr.timestamp_ns;
		pend.si.timestamp_tai_ns = hdr.timestamp_tai_ns;
		pend.hdr_n_cells = hdr.n_cells;
		pend.active = true;
	}

	pend.si.cells.emplace_back();
	E3SrsCellInfo& cell = pend.si.cells.back();
	cell.current_srs_iq_buffer = r.iq.half;       cell.srs_iq_write_index = r.iq.row;
	cell.current_srs_hest_buffer = r.hest.half;   cell.srs_hest_write_index = pend.hest_row;
	cell.current_srs_rb_snr_buffer = r.rb.half;   cell.srs_rb_snr_write_index = pend.rb_row;
	cell.srs_iq_row_byte_offset = iq_off;
	cell.cell_id = hdr.cell_id;
	cell.n_rx_ant_srs = hdr.n_rx_ant_srs;
	cell.n_prb_ul_bwp = hdr.n_prb_ul_bwp;
	cell.srs_cell_start_sym = hdr.srs_cell_start_sym;
	cell.srs_cell_n_srs_sym = hdr.srs_cell_n_srs_sym;
	cell.n_srs_ue = hdr.n_srs_ue;
	cell.ues.resize(hdr.n_srs_ue);
	for (uint16_t i = 0; i < hdr.n_srs_ue; ++i) {
		const rt::SrsUeMetrics& s = ue[i];
		E3SrsUeMetrics& m = cell.ues[i];
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
		m.sequence_id = s.sequence_id; m.config_idx = s.config_idx; m.bandwidth_idx = s.bandwidth_idx;
		m.group_seq_hopping = s.group_seq_hopping; m.start_prg = s.start_prg; m.start_valid_prg = s.start_valid_prg;
		m.srs_hest_offset = hest_off[i];   m.srs_hest_size = ub[i].hb.len;
		m.srs_rb_snr_offset = rb_off[i];   m.srs_rb_snr_size = ub[i].rb.len;
	}

	r.iq.advance(rows.srs_iq);
	return true;
}

// Publish the accumulated SRS slot and reset the pending buffer.
void publishSrs(DataLake& dl, E3Agent& agent, PendingSrs& pend, bool& warn) {
	pend.si.n_cells = uint16_t(pend.si.cells.size());
	if (warn && pend.hdr_n_cells && pend.hdr_n_cells != pend.si.n_cells) {
		e3shim::log(e3shim::WRN, "WRN", 0, fmt::format(
			"replay: SRS slot n_cells hdr={} != {} accumulated", pend.hdr_n_cells, pend.si.n_cells));
		warn = false;
	}
	{
		std::lock_guard<std::mutex> lk(dl.e3_srs_buffer_mutex);
		dl.e3_srs_buffer_info = std::move(pend.si);
	}
	agent.notifySrsDataReady();
	pend.si = {};
	pend.active = false;
}

} // namespace

Replay::Replay(DataLake& dl, E3Agent& agent, const ReplayCfg& cfg, const CpuCfg& cpu,
               const RowsCfg& rows)
	: dl_(dl), agent_(agent), cfg_(cfg), cpu_(cpu), rows_(rows) {}

bool Replay::run(std::atomic<bool>& stop) {
	pinThread(cpu_.feeder_core);
	// Rings persist across loops, mirroring a continuous live stream
	PuschRings pr;
	SrsRings sr;
	for (uint64_t pass = 0; (cfg_.loops == 0 || pass < cfg_.loops) && !stop.load();) {
		if (cfg_.wait_for_subscriber && !waitForSubscriber(agent_, stop)) break;
		TraceReader rd;
		if (!rd.open(cfg_.path)) {
			e3shim::log(e3shim::ERR, "ERR", 0, fmt::format("replay: invalid or unreadable trace {}", cfg_.path));
			return false;
		}
		if (pass == 0 && rd.v1_1()) {
			e3shim::log(e3shim::WRN, "WRN", 0,
				"replay: v1.1.0 trace has no n_prb_ul_bwp, assuming 273 PRB (correct only for 100 MHz captures; recapture narrower carriers as v1.2.0)");
		}

		timespec next;
		clock_gettime(CLOCK_MONOTONIC, &next);
		uint64_t last_tai = 0, emitted = 0;
		bool have_last = false, warn_pusch = true, warn_srs = true, abandoned = false;
		PendingPusch pp;
		PendingSrs sp;

		uint16_t tag = 0;
		std::vector<uint8_t> buf;
		while (rd.next(tag, buf) && !stop.load(std::memory_order_relaxed)) {
			if (buf.size() < sizeof(uint64_t) * 2) continue;  // need timestamp fields
			uint64_t tai;
			std::memcpy(&tai, buf.data() + sizeof(uint64_t), sizeof(tai));  // timestamp_tai_ns

			// ts_tai advanced: the prior slot is complete. Flush before pacing
			// so each slot publishes at its own capture time.
			if (have_last && tai != last_tai) {
				if (pp.active) { publishPusch(dl_, agent_, pp, warn_pusch); ++emitted; }
				if (sp.active) { publishSrs(dl_, agent_, sp, warn_srs); ++emitted; }
				if (cfg_.wait_for_subscriber && agent_.activeSubscriptionCount() == 0) {
					abandoned = true;
					break;
				}
			}
			if (have_last) {
				int64_t d = int64_t(tai - last_tai);
				if (d < 0) d = 0;
				if (d > MAX_PACE_NS) d = MAX_PACE_NS;
				if (d > 0) addNs(next, uint64_t(d));
			}
			have_last = true;
			last_tai = tai;
			clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next, nullptr);

			if (tag == rt::TAG_PUSCH) accumulatePusch(dl_, rows_, pr, pp, buf.data(), buf.size(), rd.v1_1(), stop);
			else if (tag == rt::TAG_SRS) accumulateSrs(dl_, rows_, sr, sp, buf.data(), buf.size(), rd.v1_1(), stop);
		}
		if (cfg_.wait_for_subscriber && !stop.load(std::memory_order_relaxed) && agent_.activeSubscriptionCount() == 0)
			abandoned = true;
		// Flush the trailing slot (EOF or end of this pass). On stop, skip the partial.
		if (!stop.load(std::memory_order_relaxed) && !abandoned) {
			if (pp.active) { publishPusch(dl_, agent_, pp, warn_pusch); ++emitted; }
			if (sp.active) { publishSrs(dl_, agent_, sp, warn_srs); ++emitted; }
		}
		rd.close();
		if (abandoned) {
			e3shim::log(e3shim::INF, "INF", 0,
				fmt::format("replay: last subscription released after {} slots; pass {} restarts on the next one", emitted, pass));
			continue;
		}
		e3shim::log(e3shim::INF, "INF", 0,
			fmt::format("replay: pass {} emitted {} slots", pass, emitted));
		if (emitted == 0) return true;  // empty/garbage trace: don't spin forever
		++pass;
	}
	return true;
}

} // namespace e3sa
