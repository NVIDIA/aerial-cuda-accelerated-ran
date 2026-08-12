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

// Replay trace format: single self-describing binary, source of truth shared by
// the C++ reader (src/replay.cpp) and the offline ClickHouse generator.
//
// Layout: FileHeader, then a stream of records {RecordHeader, body}.
//   PUSCH body: PuschSlotHeader, PuschUeMetrics[n_ue], blob fh (whole row),
//               blob hest (whole row).
//   SRS body:   SrsSlotHeader, SrsUeMetrics[n_srs_ue], blob iq (per cell),
//               then per UE in order: blob srs_hest, blob srs_rb_snr.
// A blob is BlobHeader{len} + len bytes; len==0 means absent (zero-fill on replay).
// Scalars mirror E3*BufferInfo/E3*Metrics minus SHM bookkeeping (buffer/write
// indices and row offsets), which replay recomputes. Little-endian, same-arch.

#ifndef E3SA_REPLAY_FORMAT_HPP
#define E3SA_REPLAY_FORMAT_HPP

#include <cstdint>

namespace e3sa {
namespace trace {

constexpr uint32_t fourcc(char a, char b, char c, char d) {
	return uint32_t(uint8_t(a)) | (uint32_t(uint8_t(b)) << 8)
	     | (uint32_t(uint8_t(c)) << 16) | (uint32_t(uint8_t(d)) << 24);
}

constexpr uint32_t MAGIC   = fourcc('E', '3', 'R', 'T'); // E3 Replay Trace
constexpr uint32_t VERSION = 0x010100;                   // v1.1.0 (dApps sample apps)

enum Tag : uint16_t {
	TAG_PUSCH = 1,
	TAG_SRS   = 2,
};

struct FileHeader {
	uint32_t magic;
	uint32_t version;            // container layout (VERSION)
	uint32_t shm_layout_version; // e3::SHM_LAYOUT_VERSION at generation time
	uint32_t reserved;
} __attribute__((packed));

// One record per E3 indication; payload_len covers the body after this header.
struct RecordHeader {
	uint16_t tag;
	uint16_t reserved;
	uint32_t payload_len;
} __attribute__((packed));

struct BlobHeader {
	uint32_t len;
} __attribute__((packed));

// Mirrors E3BufferInfo (consumer-facing subset).
struct PuschSlotHeader {
	uint64_t timestamp_ns;
	uint64_t timestamp_tai_ns;
	uint16_t sfn;
	uint16_t slot;
	uint16_t cell_id;
	uint16_t n_rx_ant;
	uint16_t n_rx_ant_srs;
	uint16_t n_cells;
	uint8_t  n_bs_ants;
	uint8_t  reserved;
	uint16_t n_ue;
} __attribute__((packed));

// Mirrors E3UeMetrics; h_offset/h_size index into the slot hest blob.
struct PuschUeMetrics {
	uint16_t rnti;
	uint8_t  tb_crc_fail;
	uint32_t cb_errors;
	float    rsrp;
	float    noise_var;
	float    sinr;
	uint16_t cb_count;
	float    rssi;
	uint8_t  qam_mod_order;
	uint8_t  mcs_index;
	uint8_t  mcs_table_index;
	uint16_t rb_start;
	uint16_t rb_size;
	uint8_t  start_symbol_index;
	uint8_t  nr_of_symbols;
	uint32_t tb_size;
	uint32_t pdu_len;
	uint16_t target_code_rate;
	uint8_t  new_data_indicator;
	uint8_t  n_layers;
	uint16_t layer_offset;
	uint16_t ue_grp_idx;
	uint32_t h_offset;
	uint32_t h_size;
	uint16_t n_subcarriers;
	uint8_t  n_dmrs_estimates;
	uint16_t dmrs_symb_pos;
	float    timing_advance;
	float    cfo_hz;
	uint8_t  harq_process_id;
	uint8_t  rv_index;
} __attribute__((packed));

// Mirrors E3SrsBufferInfo (consumer-facing subset).
struct SrsSlotHeader {
	uint64_t timestamp_ns;
	uint64_t timestamp_tai_ns;
	uint16_t sfn;
	uint16_t slot;
	uint16_t cell_id;
	uint16_t n_cells;
	uint16_t n_rx_ant_srs;
	uint8_t  srs_cell_start_sym;
	uint8_t  srs_cell_n_srs_sym;
	uint16_t n_srs_ue;
	uint16_t reserved;
} __attribute__((packed));

// Mirrors E3SrsUeMetrics minus SHM row offsets (recomputed on packing);
// blob sizes are intrinsic and retained for the consumer.
struct SrsUeMetrics {
	uint16_t rnti;
	float    wideband_snr;
	float    signal_energy;
	float    noise_energy;
	float    toa_us;
	uint8_t  hd_ant_flag;
	float    sc_corr_re;
	float    sc_corr_im;
	float    cs_corr_ratio_db;
	uint8_t  n_ant_ports;
	uint8_t  n_syms;
	uint8_t  n_repetitions;
	uint8_t  comb_size;
	uint8_t  comb_offset;
	uint8_t  start_sym;
	uint8_t  cyclic_shift;
	uint8_t  frequency_position;
	uint16_t frequency_shift;
	uint8_t  frequency_hopping;
	uint8_t  resource_type;
	uint16_t t_srs;
	uint16_t t_offset;
	uint32_t usage;
	uint16_t n_valid_prg;
	uint16_t prg_size;
	uint16_t n_prb_grps;
	uint32_t srs_hest_size;
	uint32_t srs_rb_snr_size;
} __attribute__((packed));

static_assert(sizeof(FileHeader) == 16, "FileHeader layout");
static_assert(sizeof(RecordHeader) == 8, "RecordHeader layout");
static_assert(sizeof(BlobHeader) == 4, "BlobHeader layout");
static_assert(sizeof(PuschSlotHeader) == 32, "PuschSlotHeader layout");
static_assert(sizeof(PuschUeMetrics) == 73, "PuschUeMetrics layout");
static_assert(sizeof(SrsSlotHeader) == 32, "SrsSlotHeader layout");
static_assert(sizeof(SrsUeMetrics) == 65, "SrsUeMetrics layout");

} // namespace trace
} // namespace e3sa

#endif // E3SA_REPLAY_FORMAT_HPP
