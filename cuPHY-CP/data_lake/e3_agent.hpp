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

#ifndef E3_AGENT_HPP
#define E3_AGENT_HPP

#include <string_view>
#include <string>
#include <thread>
#include <atomic>
#include <mutex>
#include <map>
#include <unordered_map>
#include <vector>
#include <chrono>
#include <cstdint>

#include <zmq.hpp>
#include <nlohmann/json.hpp>

#include "nvlog.hpp"

#define TAG_E3 (NVLOG_TAG_BASE_CUPHY_CONTROLLER + 7) // "CTL.E3"

// Forward declarations
class DataLake;
struct fhInfo_t;
struct puschInfo_t;
struct hestInfo_t;
struct srsIqInfo_t;
struct srsInfo_t;
struct srsHestInfo_t;
using json = nlohmann::json;

// E3 Protocol definitions
namespace e3 {

/**
 * E3AP Telemetry stream types as bit flags for efficient internal processing.
 *
 * Telemetry ID = bit_position + 1 (e.g. IQ_SAMPLES = bit 0 -> telemetry ID 1).
 *
 * DO NOT reorder or remove. IDs are stable wire protocol values
 * used in E3-SubscriptionRequest telemetryIdentifierList and E3-RanFunctionDefinition.
 * New entries go at the end only. Capped at 128 streams (__uint128_t); beyond that,
 * switch to another method like two-word struct or std::bitset.
 */
enum class StreamType : __uint128_t {
    NONE                  = 0,
    IQ_SAMPLES            = __uint128_t(1) << 0,
    PDU_DATA              = __uint128_t(1) << 1,
    H_ESTIMATES           = __uint128_t(1) << 2,
    TIMESTAMP             = __uint128_t(1) << 3,
    SFN                   = __uint128_t(1) << 4,
    SLOT                  = __uint128_t(1) << 5,
    CELL_ID               = __uint128_t(1) << 6,
    N_RX_ANT              = __uint128_t(1) << 7,
    N_RX_ANT_SRS          = __uint128_t(1) << 8,
    N_CELLS               = __uint128_t(1) << 9,
    N_BS_ANTS             = __uint128_t(1) << 10,
    N_LAYERS              = __uint128_t(1) << 11,
    N_SUBCARRIERS         = __uint128_t(1) << 12,
    N_DMRS_ESTIMATES      = __uint128_t(1) << 13,
    DMRS_SYMB_POS         = __uint128_t(1) << 14,
    TB_CRC_FAIL           = __uint128_t(1) << 15,
    CB_ERRORS             = __uint128_t(1) << 16,
    RSRP                  = __uint128_t(1) << 17,
    NOISE_VAR             = __uint128_t(1) << 18,
    CB_COUNT              = __uint128_t(1) << 19,
    RSSI                  = __uint128_t(1) << 20,
    QAM_MOD_ORDER         = __uint128_t(1) << 21,
    MCS_INDEX             = __uint128_t(1) << 22,
    MCS_TABLE_INDEX       = __uint128_t(1) << 23,
    RB_START              = __uint128_t(1) << 24,
    RB_SIZE               = __uint128_t(1) << 25,
    START_SYMBOL_INDEX    = __uint128_t(1) << 26,
    NR_OF_SYMBOLS         = __uint128_t(1) << 27,
    TB_SIZE               = __uint128_t(1) << 28,
    PDU_LEN               = __uint128_t(1) << 29,
    TARGET_CODE_RATE      = __uint128_t(1) << 30,
    NEW_DATA_INDICATOR    = __uint128_t(1) << 31,
    RNTI                  = __uint128_t(1) << 32,
    N_UE                  = __uint128_t(1) << 33,
    LAYER_OFFSET          = __uint128_t(1) << 34,
    UE_GRP_IDX            = __uint128_t(1) << 35,
    H_OFFSET              = __uint128_t(1) << 36,
    H_SIZE                = __uint128_t(1) << 37,
    SINR                  = __uint128_t(1) << 38,
    TIMING_ADVANCE        = __uint128_t(1) << 39,
    HARQ_PROCESS_ID       = __uint128_t(1) << 40,
    RV_INDEX              = __uint128_t(1) << 41,
    CFO_HZ                = __uint128_t(1) << 42,

    // SRS streams (IDs 44-77). SHM refs grouped with their per-UE decoders.
    SRS_IQ_SAMPLES        = __uint128_t(1) << 43,  // SHM: raw SRS IQ grid (cell-level)
    SRS_HEST              = __uint128_t(1) << 44,  // SHM: SRS channel estimates (per-UE concatenated)
    SRS_HEST_N_PRB_GRPS   = __uint128_t(1) << 45,
    SRS_HEST_OFFSET       = __uint128_t(1) << 46,
    SRS_HEST_SIZE         = __uint128_t(1) << 47,
    SRS_RB_SNR            = __uint128_t(1) << 48,  // SHM: per-RB SNR (per-UE concatenated)
    SRS_RB_SNR_OFFSET     = __uint128_t(1) << 49,
    SRS_RB_SNR_SIZE       = __uint128_t(1) << 50,
    SRS_CELL_START_SYM    = __uint128_t(1) << 51,
    SRS_CELL_N_SRS_SYM    = __uint128_t(1) << 52,
    N_SRS_UE              = __uint128_t(1) << 53,
    SRS_WIDEBAND_SNR      = __uint128_t(1) << 54,
    SRS_SIGNAL_ENERGY     = __uint128_t(1) << 55,
    SRS_NOISE_ENERGY      = __uint128_t(1) << 56,
    SRS_TOA               = __uint128_t(1) << 57,
    SRS_HD_ANT_FLAG       = __uint128_t(1) << 58,
    SRS_SC_CORR           = __uint128_t(1) << 59,  // JSON array [re, im] (float32)
    SRS_CS_CORR_RATIO_DB  = __uint128_t(1) << 60,
    SRS_ANT_PORTS         = __uint128_t(1) << 61,
    SRS_N_SYMS            = __uint128_t(1) << 62,
    SRS_N_REPETITIONS     = __uint128_t(1) << 63,
    SRS_COMB_SIZE         = __uint128_t(1) << 64,
    SRS_COMB_OFFSET       = __uint128_t(1) << 65,
    SRS_START_SYM         = __uint128_t(1) << 66,
    SRS_CYCLIC_SHIFT      = __uint128_t(1) << 67,
    SRS_FREQ_POSITION     = __uint128_t(1) << 68,
    SRS_FREQ_SHIFT        = __uint128_t(1) << 69,
    SRS_FREQ_HOPPING      = __uint128_t(1) << 70,
    SRS_RESOURCE_TYPE     = __uint128_t(1) << 71,
    SRS_T_SRS             = __uint128_t(1) << 72,
    SRS_T_OFFSET          = __uint128_t(1) << 73,
    SRS_USAGE             = __uint128_t(1) << 74,
    SRS_N_VALID_PRG       = __uint128_t(1) << 75,
    SRS_PRG_SIZE          = __uint128_t(1) << 76,
    TIMESTAMP_TAI         = __uint128_t(1) << 77
};

constexpr uint32_t STREAM_TYPE_COUNT = 78;

/** E3AP protocol version supported by this agent implementation */
constexpr std::string_view E3AP_PROTOCOL_VERSION = "1.0.0";
/** RAN identifier used in E3 Setup messages */
constexpr std::string_view RAN_IDENTIFIER = "NVIDIA_L1";
/** RAN function ID for NVIDIA KPM (Key Performance Monitoring) */
constexpr uint32_t RAN_FUNCTION_ID_NVIDIA_KPM = 2;

/**
 * Convert telemetry ID (1-based) to StreamType.
 *
 * @param[in] id Telemetry identifier (1-based, valid range 1 to STREAM_TYPE_COUNT).
 *               IDs of 0 or greater than STREAM_TYPE_COUNT are treated as invalid.
 * @return Corresponding StreamType bitfield value, or StreamType::NONE for invalid IDs.
 */
constexpr StreamType telemetryIdToStreamType(uint32_t id) noexcept
{
    if (id == 0 || id > STREAM_TYPE_COUNT) return StreamType::NONE;
    return static_cast<StreamType>(__uint128_t(1) << (id - 1));
}

/**
 * Converts string stream name to StreamType enum
 * 
 * @param[in] stream_name The stream name as string
 * @return Corresponding StreamType enum value
 */
constexpr StreamType streamNameToType(const std::string_view stream_name) noexcept
{
    if (stream_name == "iq_samples") return StreamType::IQ_SAMPLES;
    if (stream_name == "pdu_data") return StreamType::PDU_DATA;
    if (stream_name == "h_estimates") return StreamType::H_ESTIMATES;
    if (stream_name == "timestamp") return StreamType::TIMESTAMP;
    if (stream_name == "timestamp_tai") return StreamType::TIMESTAMP_TAI;
    if (stream_name == "sfn") return StreamType::SFN;
    if (stream_name == "slot") return StreamType::SLOT;
    if (stream_name == "cell_id") return StreamType::CELL_ID;
    if (stream_name == "n_rx_ant") return StreamType::N_RX_ANT;
    if (stream_name == "n_rx_ant_srs") return StreamType::N_RX_ANT_SRS;
    if (stream_name == "n_cells") return StreamType::N_CELLS;
    if (stream_name == "n_bs_ants") return StreamType::N_BS_ANTS;
    if (stream_name == "n_layers") return StreamType::N_LAYERS;
    if (stream_name == "n_subcarriers") return StreamType::N_SUBCARRIERS;
    if (stream_name == "n_dmrs_estimates") return StreamType::N_DMRS_ESTIMATES;
    if (stream_name == "dmrs_symb_pos") return StreamType::DMRS_SYMB_POS;
    if (stream_name == "tb_crc_fail") return StreamType::TB_CRC_FAIL;
    if (stream_name == "cb_errors") return StreamType::CB_ERRORS;
    if (stream_name == "rsrp") return StreamType::RSRP;
    if (stream_name == "noise_var") return StreamType::NOISE_VAR;
    if (stream_name == "cb_count") return StreamType::CB_COUNT;
    if (stream_name == "rssi") return StreamType::RSSI;
    if (stream_name == "qam_mod_order") return StreamType::QAM_MOD_ORDER;
    if (stream_name == "mcs_index") return StreamType::MCS_INDEX;
    if (stream_name == "mcs_table_index") return StreamType::MCS_TABLE_INDEX;
    if (stream_name == "rb_start") return StreamType::RB_START;
    if (stream_name == "rb_size") return StreamType::RB_SIZE;
    if (stream_name == "start_symbol_index") return StreamType::START_SYMBOL_INDEX;
    if (stream_name == "nr_of_symbols") return StreamType::NR_OF_SYMBOLS;
    if (stream_name == "tb_size") return StreamType::TB_SIZE;
    if (stream_name == "pdu_len") return StreamType::PDU_LEN;
    if (stream_name == "target_code_rate") return StreamType::TARGET_CODE_RATE;
    if (stream_name == "new_data_indicator") return StreamType::NEW_DATA_INDICATOR;
    if (stream_name == "rnti") return StreamType::RNTI;
    if (stream_name == "n_ue") return StreamType::N_UE;
    if (stream_name == "layer_offset") return StreamType::LAYER_OFFSET;
    if (stream_name == "ue_grp_idx") return StreamType::UE_GRP_IDX;
    if (stream_name == "h_offset") return StreamType::H_OFFSET;
    if (stream_name == "h_size") return StreamType::H_SIZE;
    if (stream_name == "sinr") return StreamType::SINR;
    if (stream_name == "timing_advance") return StreamType::TIMING_ADVANCE;
    if (stream_name == "harq_process_id") return StreamType::HARQ_PROCESS_ID;
    if (stream_name == "rv_index") return StreamType::RV_INDEX;
    if (stream_name == "cfo_hz") return StreamType::CFO_HZ;
    // SRS streams
    if (stream_name == "srs_iq_samples") return StreamType::SRS_IQ_SAMPLES;
    if (stream_name == "srs_hest") return StreamType::SRS_HEST;
    if (stream_name == "srs_hest_n_prb_grps") return StreamType::SRS_HEST_N_PRB_GRPS;
    if (stream_name == "srs_hest_offset") return StreamType::SRS_HEST_OFFSET;
    if (stream_name == "srs_hest_size") return StreamType::SRS_HEST_SIZE;
    if (stream_name == "srs_rb_snr") return StreamType::SRS_RB_SNR;
    if (stream_name == "srs_rb_snr_offset") return StreamType::SRS_RB_SNR_OFFSET;
    if (stream_name == "srs_rb_snr_size") return StreamType::SRS_RB_SNR_SIZE;
    if (stream_name == "srs_cell_start_sym") return StreamType::SRS_CELL_START_SYM;
    if (stream_name == "srs_cell_n_srs_sym") return StreamType::SRS_CELL_N_SRS_SYM;
    if (stream_name == "n_srs_ue") return StreamType::N_SRS_UE;
    if (stream_name == "srs_wideband_snr") return StreamType::SRS_WIDEBAND_SNR;
    if (stream_name == "srs_signal_energy") return StreamType::SRS_SIGNAL_ENERGY;
    if (stream_name == "srs_noise_energy") return StreamType::SRS_NOISE_ENERGY;
    if (stream_name == "srs_toa") return StreamType::SRS_TOA;
    if (stream_name == "srs_hd_ant_flag") return StreamType::SRS_HD_ANT_FLAG;
    if (stream_name == "srs_sc_corr") return StreamType::SRS_SC_CORR;
    if (stream_name == "srs_cs_corr_ratio_db") return StreamType::SRS_CS_CORR_RATIO_DB;
    if (stream_name == "srs_ant_ports") return StreamType::SRS_ANT_PORTS;
    if (stream_name == "srs_n_syms") return StreamType::SRS_N_SYMS;
    if (stream_name == "srs_n_repetitions") return StreamType::SRS_N_REPETITIONS;
    if (stream_name == "srs_comb_size") return StreamType::SRS_COMB_SIZE;
    if (stream_name == "srs_comb_offset") return StreamType::SRS_COMB_OFFSET;
    if (stream_name == "srs_start_sym") return StreamType::SRS_START_SYM;
    if (stream_name == "srs_cyclic_shift") return StreamType::SRS_CYCLIC_SHIFT;
    if (stream_name == "srs_freq_position") return StreamType::SRS_FREQ_POSITION;
    if (stream_name == "srs_freq_shift") return StreamType::SRS_FREQ_SHIFT;
    if (stream_name == "srs_freq_hopping") return StreamType::SRS_FREQ_HOPPING;
    if (stream_name == "srs_resource_type") return StreamType::SRS_RESOURCE_TYPE;
    if (stream_name == "srs_t_srs") return StreamType::SRS_T_SRS;
    if (stream_name == "srs_t_offset") return StreamType::SRS_T_OFFSET;
    if (stream_name == "srs_usage") return StreamType::SRS_USAGE;
    if (stream_name == "srs_n_valid_prg") return StreamType::SRS_N_VALID_PRG;
    if (stream_name == "srs_prg_size") return StreamType::SRS_PRG_SIZE;
    return StreamType::NONE;
}

/**
 * Bitwise OR operator for StreamType flags
 */
constexpr StreamType operator|(const StreamType lhs, const StreamType rhs) noexcept
{
    return static_cast<StreamType>(static_cast<__uint128_t>(lhs) | static_cast<__uint128_t>(rhs));
}

/**
 * Bitwise OR assignment operator for StreamType flags
 */
constexpr StreamType& operator|=(StreamType& lhs, const StreamType rhs) noexcept
{
    lhs = lhs | rhs;
    return lhs;
}

/**
 * Bitwise AND operator for StreamType flags
 */
constexpr StreamType operator&(const StreamType lhs, const StreamType rhs) noexcept
{
    return static_cast<StreamType>(static_cast<__uint128_t>(lhs) & static_cast<__uint128_t>(rhs));
}

/**
 * Bitwise NOT operator for StreamType flags
 */
constexpr StreamType operator~(const StreamType val) noexcept
{
    return static_cast<StreamType>(~static_cast<__uint128_t>(val));
}

// Streams emitted inside ue_metrics[] (per-UE); everything else is cell-level
constexpr StreamType PER_UE_STREAMS =
    // PUSCH per-UE
    StreamType::RNTI | StreamType::TB_CRC_FAIL | StreamType::CB_ERRORS |
    StreamType::RSRP | StreamType::NOISE_VAR | StreamType::SINR |
    StreamType::CB_COUNT | StreamType::RSSI |
    StreamType::QAM_MOD_ORDER | StreamType::MCS_INDEX |
    StreamType::MCS_TABLE_INDEX | StreamType::RB_START | StreamType::RB_SIZE |
    StreamType::START_SYMBOL_INDEX | StreamType::NR_OF_SYMBOLS |
    StreamType::N_LAYERS | StreamType::TB_SIZE | StreamType::PDU_LEN |
    StreamType::TARGET_CODE_RATE | StreamType::NEW_DATA_INDICATOR |
    StreamType::LAYER_OFFSET | StreamType::UE_GRP_IDX |
    StreamType::N_SUBCARRIERS | StreamType::N_DMRS_ESTIMATES |
    StreamType::DMRS_SYMB_POS | StreamType::H_OFFSET | StreamType::H_SIZE |
    StreamType::TIMING_ADVANCE | StreamType::HARQ_PROCESS_ID | StreamType::RV_INDEX |
    StreamType::CFO_HZ |
    // SRS per-UE
    StreamType::SRS_HEST_N_PRB_GRPS | StreamType::SRS_HEST_OFFSET |
    StreamType::SRS_HEST_SIZE | StreamType::SRS_RB_SNR_OFFSET |
    StreamType::SRS_RB_SNR_SIZE | StreamType::SRS_WIDEBAND_SNR |
    StreamType::SRS_SIGNAL_ENERGY | StreamType::SRS_NOISE_ENERGY |
    StreamType::SRS_TOA | StreamType::SRS_HD_ANT_FLAG |
    StreamType::SRS_SC_CORR | StreamType::SRS_CS_CORR_RATIO_DB |
    StreamType::SRS_ANT_PORTS | StreamType::SRS_N_SYMS |
    StreamType::SRS_N_REPETITIONS | StreamType::SRS_COMB_SIZE |
    StreamType::SRS_COMB_OFFSET | StreamType::SRS_START_SYM |
    StreamType::SRS_CYCLIC_SHIFT | StreamType::SRS_FREQ_POSITION |
    StreamType::SRS_FREQ_SHIFT | StreamType::SRS_FREQ_HOPPING |
    StreamType::SRS_RESOURCE_TYPE | StreamType::SRS_T_SRS |
    StreamType::SRS_T_OFFSET | StreamType::SRS_USAGE |
    StreamType::SRS_N_VALID_PRG | StreamType::SRS_PRG_SIZE;

// MAINTENANCE: When adding a new StreamType, update exactly one of the two
// PROVIDABLE masks below and the corresponding notify*() switch cases.
// Shared streams (TIMESTAMP, TIMESTAMP_TAI, SFN, SLOT, CELL_ID, N_RX_ANT_SRS,
// N_CELLS, RNTI) must appear in both masks. Also update PER_UE_STREAMS if per-UE.
// The SHARED / *_ONLY masks below are derived automatically — no manual update needed.

// All streams that the PUSCH indication path can provide: PUSCH-specific + shared.
// Subscriptions with at least one PUSCH_PROVIDABLE bit fire from notifyDataReady().
constexpr StreamType PUSCH_PROVIDABLE_STREAMS =
    // PUSCH-only cell-level
    StreamType::IQ_SAMPLES | StreamType::PDU_DATA | StreamType::H_ESTIMATES |
    StreamType::N_RX_ANT | StreamType::N_BS_ANTS | StreamType::N_UE |
    // Shared cell-level (also provided by SRS path)
    StreamType::TIMESTAMP | StreamType::TIMESTAMP_TAI | StreamType::SFN | StreamType::SLOT |
    StreamType::CELL_ID | StreamType::N_RX_ANT_SRS | StreamType::N_CELLS |
    // PUSCH-only per-UE
    StreamType::RNTI | StreamType::TB_CRC_FAIL | StreamType::CB_ERRORS |
    StreamType::RSRP | StreamType::NOISE_VAR | StreamType::SINR |
    StreamType::CB_COUNT | StreamType::RSSI |
    StreamType::QAM_MOD_ORDER | StreamType::MCS_INDEX |
    StreamType::MCS_TABLE_INDEX | StreamType::RB_START | StreamType::RB_SIZE |
    StreamType::START_SYMBOL_INDEX | StreamType::NR_OF_SYMBOLS |
    StreamType::N_LAYERS | StreamType::TB_SIZE | StreamType::PDU_LEN |
    StreamType::TARGET_CODE_RATE | StreamType::NEW_DATA_INDICATOR |
    StreamType::LAYER_OFFSET | StreamType::UE_GRP_IDX |
    StreamType::N_SUBCARRIERS | StreamType::N_DMRS_ESTIMATES |
    StreamType::DMRS_SYMB_POS | StreamType::H_OFFSET | StreamType::H_SIZE |
    StreamType::TIMING_ADVANCE | StreamType::HARQ_PROCESS_ID |
    StreamType::RV_INDEX | StreamType::CFO_HZ;

// All streams that the SRS indication path can provide: SRS-specific + shared.
// Subscriptions with at least one SRS_PROVIDABLE bit fire from notifySrsDataReady().
constexpr StreamType SRS_PROVIDABLE_STREAMS =
    // Shared cell-level (also provided by PUSCH path)
    StreamType::TIMESTAMP | StreamType::TIMESTAMP_TAI | StreamType::SFN | StreamType::SLOT |
    StreamType::CELL_ID | StreamType::N_RX_ANT_SRS | StreamType::N_CELLS |
    // Shared per-UE
    StreamType::RNTI |
    // SRS-only cell-level
    StreamType::SRS_IQ_SAMPLES | StreamType::SRS_HEST |
    StreamType::SRS_RB_SNR |
    StreamType::SRS_CELL_START_SYM | StreamType::SRS_CELL_N_SRS_SYM |
    StreamType::N_SRS_UE |
    // SRS-only per-UE
    StreamType::SRS_HEST_N_PRB_GRPS | StreamType::SRS_HEST_OFFSET |
    StreamType::SRS_HEST_SIZE |
    StreamType::SRS_RB_SNR_OFFSET | StreamType::SRS_RB_SNR_SIZE |
    StreamType::SRS_WIDEBAND_SNR |
    StreamType::SRS_SIGNAL_ENERGY | StreamType::SRS_NOISE_ENERGY |
    StreamType::SRS_TOA | StreamType::SRS_HD_ANT_FLAG |
    StreamType::SRS_SC_CORR | StreamType::SRS_CS_CORR_RATIO_DB |
    StreamType::SRS_ANT_PORTS | StreamType::SRS_N_SYMS |
    StreamType::SRS_N_REPETITIONS | StreamType::SRS_COMB_SIZE |
    StreamType::SRS_COMB_OFFSET | StreamType::SRS_START_SYM |
    StreamType::SRS_CYCLIC_SHIFT | StreamType::SRS_FREQ_POSITION |
    StreamType::SRS_FREQ_SHIFT | StreamType::SRS_FREQ_HOPPING |
    StreamType::SRS_RESOURCE_TYPE | StreamType::SRS_T_SRS |
    StreamType::SRS_T_OFFSET | StreamType::SRS_USAGE |
    StreamType::SRS_N_VALID_PRG | StreamType::SRS_PRG_SIZE;

// Derived masks for indication gating: prevent a path from firing when the
// subscription only contains streams exclusive to the *other* path.
//   fire_pusch = (sub & PUSCH_ONLY) != 0 || ((sub & SRS_ONLY) == 0 && (sub & SHARED) != 0)
//   fire_srs   = (sub & SRS_ONLY)  != 0 || ((sub & PUSCH_ONLY) == 0 && (sub & SHARED) != 0)
constexpr StreamType SHARED_PROVIDABLE_STREAMS =
    PUSCH_PROVIDABLE_STREAMS & SRS_PROVIDABLE_STREAMS;
constexpr StreamType PUSCH_ONLY_PROVIDABLE_STREAMS =
    PUSCH_PROVIDABLE_STREAMS & ~SHARED_PROVIDABLE_STREAMS;
constexpr StreamType SRS_ONLY_PROVIDABLE_STREAMS =
    SRS_PROVIDABLE_STREAMS & ~SHARED_PROVIDABLE_STREAMS;

} // namespace e3

// Shared memory header structure. Append-only: new fields go before reserved[].
struct SharedMemoryHeader {
    uint32_t version;                    // 0x010100 = v1.1.0
    // FH / PUSCH / Hest
    uint32_t fh_buffer_size;
    uint32_t pusch_buffer_size;
    uint32_t hest_buffer_size;
    uint32_t num_fh_samples;
    uint32_t num_fh_rows;
    uint32_t num_pusch_rows;
    uint32_t num_hest_rows;
    uint32_t max_hest_samples_per_row;
    // SRS IQ / SRS Hest / SRS RbSNR
    uint32_t srs_iq_buffer_size;
    uint32_t num_srs_iq_samples;
    uint32_t num_srs_iq_rows;
    uint32_t srs_hest_buffer_size;
    uint32_t max_srs_hest_bytes_per_row;
    uint32_t num_srs_hest_rows;
    uint32_t srs_rb_snr_buffer_size;
    uint32_t max_srs_rb_snr_bytes_per_row;
    uint32_t num_srs_rb_snr_rows;
    uint32_t reserved[32];
};

class E3Agent {
public:
    E3Agent(
        DataLake* dataLake,
        const uint16_t e3RepPort,
        const uint16_t e3PubPort,
        const uint16_t e3SubPort,
        const int numRowsToInsertFh,
        const int numRowsToInsertPusch,
        const int numRowsToInsertHest,
        const uint32_t numFhSamples,
        const uint32_t maxPuschPduSize,
        const uint32_t maxHestSamplesPerRow,
        const int numRowsToInsertSrsIq,
        const int numRowsToInsertSrs,
        const int numRowsToInsertSrsHest,
        const uint32_t maxSrsIqSamplesPerRow,
        const uint32_t maxSrsHestBytesPerRow,
        const uint32_t maxSrsRbSnrBytesPerRow
    );
    ~E3Agent();

    bool init();
    void shutdown();
    
    // Shared memory management
    bool createSharedMemoryBuffers(
        fhInfo_t** pFh,
        fhInfo_t** pInsertFh,
        puschInfo_t** p,
        puschInfo_t** pInsertPusch,
        hestInfo_t** pHest,
        hestInfo_t** pInsertHest,
        srsIqInfo_t** pSrsIq,
        srsIqInfo_t** pInsertSrsIq,
        srsInfo_t** pSrs,
        srsInfo_t** pInsertSrs,
        srsHestInfo_t** pSrsHest,
        srsHestInfo_t** pInsertSrsHest
    );

    void notifyDataReady();
    void notifySrsDataReady();

private:
    DataLake* dataLake;

    // E3 Agent configuration
    static constexpr std::string_view E3_SHARED_MEMORY_KEY = "/e3_ran_buffers";
    uint16_t e3RepPort;
    uint16_t e3PubPort;
    uint16_t e3SubPort;
    int numRowsToInsertFh;
    int numRowsToInsertPusch;
    int numRowsToInsertHest;
    uint32_t numFhSamples;
    uint32_t maxPuschPduSize;
    uint32_t maxHestSamplesPerRow;
    int numRowsToInsertSrsIq;
    int numRowsToInsertSrs;
    int numRowsToInsertSrsHest;
    uint32_t maxSrsIqSamplesPerRow;
    uint32_t maxSrsHestBytesPerRow;
    uint32_t maxSrsRbSnrBytesPerRow;

    // ZMQ components
    zmq::context_t zmq_context;
    zmq::socket_t e3_rep_socket;  // Manager → Agent (REQ-REP)
    zmq::socket_t e3_pub_socket;  // Agent → Manager (indications)
    std::mutex e3_pub_socket_mutex_;
    zmq::socket_t e3_sub_socket;  // Manager → Agent (PUB-SUB commands)

    // Thread management
    std::thread e3_data_thread;
    std::thread e3_reaper_thread;
    std::thread e3_sub_thread;
    std::atomic<bool> e3_running{false};
    std::atomic<bool> e3_reaper_running{false};
    std::atomic<bool> e3_sub_running{false};

    // Active subscriptions
    struct E3Subscription {
        uint32_t subscription_id;
        uint32_t dapp_id;
        uint32_t ran_function_id;
        std::vector<uint32_t> telemetry_ids;      // Granted telemetry IDs (wire protocol values)
        e3::StreamType stream_bitfield;           // Internal bitfield for indication processing
        uint32_t periodicity_us;
        // Per-path timestamps to avoid cross-path periodicity starvation
        std::chrono::steady_clock::time_point last_update_pusch;
        std::chrono::steady_clock::time_point last_update_srs;
        std::chrono::steady_clock::time_point expiry_time;  // time_point::max() = indefinite
    };
    std::unordered_map<uint32_t, E3Subscription> e3_subscriptions;
    std::mutex e3_subscriptions_mutex;

    // Connected dApp managers
    struct DAppConnectionInfo {
        std::chrono::steady_clock::time_point last_activity_time;
    };
    std::map<uint32_t, DAppConnectionInfo> e3_connected_dapps;
    std::mutex e3_dapps_mutex;

    // Shared memory buffers for data exchange
    int shm_data_fd{-1};
    void* shm_data_ptr{nullptr};
    size_t shm_data_size{0};

    // Thread functions
    void dataServerThread();
    void reaperThread();
    void reapTimedOutDapps();
    void managerSubscriptionThread();

    /** Handle E3 Setup request (REQ-REP)
     *
     * @param[in] request JSON request containing dApp setup parameters
     * @param[out] response JSON response with dAppIdentifier and available streams
     */
    void handleSetupRequest(const json& request, std::string& response);

    /** Handle subscription request from dApp (PUB-SUB)
     *
     * @param[in] request JSON request with telemetryIdentifierList and ranFunctionIdentifier
     * @param[out] response JSON response with responseCode (positive/negative)
     */
    void handleSubscriptionRequest(const json& request, std::string& response);

    /** Handle subscription deletion request from dApp (PUB-SUB)
     *
     * @param[in] request JSON request containing subscriptionId
     * @param[out] response JSON response with responseCode (positive/negative)
     */
    void handleSubscriptionDelete(const json& request, std::string& response);

    /** Handle control action from dApp (PUB-SUB) - placeholder, no control dispatch yet
     *
     * @param[in] request JSON request containing dAppControlAction fields
     * @param[out] response JSON ack with responseCode
     */
    void handleControlMessage(const json& request, std::string& response);

    /** Dispatch incoming PUB-SUB messages by type
     *
     * @param[in] message Parsed JSON message from the dApp PUB socket
     */
    void handleManagerMessage(const json& message);

    /** Release a dApp connection and clean up associated subscriptions
     *
     * @param[in] dapp_id Identifier of the dApp to release
     */
    void releaseDapp(uint32_t dapp_id);

    /** Update last activity time for a connected dApp
     *
     * @param[in] dapp_id Identifier of the dApp
     * @return true if dApp exists and was updated, false if not found
     */
    bool updateDappActivity(uint32_t dapp_id);

    /** Broadcast E3 release message for the specified dApp
     *
     * @param[in] dapp_id Identifier of the dApp being released
     * @return true if message was sent successfully, false otherwise
     */
    bool sendRelease(uint32_t dapp_id);

    /** Generate a unique message identifier for E3AP messages
     *
     * @return Unique uint32_t message identifier
     */
    uint32_t generateMessageId();

    /** Generate a unique dApp identifier for new connections
     *
     * @return Unique uint32_t dApp identifier
     */
    uint32_t generateDappId();

    /** Generate a unique subscription identifier
     *
     * @return Unique uint32_t subscription identifier
     */
    uint32_t generateSubscriptionId();
    
    // Stream creation helpers
    json createIndicationPayloadDelivery(const std::string& stream_id) const;
    json createIndicationPayloadStream(
        const std::string& stream_id,
        const std::string& data_type,
        const std::string& description
    ) const;
    json createSharedMemoryStream(
        const std::string& stream_id,
        const std::string& data_type,
        const std::string& description,
        const size_t memory_size_bytes,
        const uint32_t max_elements,
        const json& additional_shm_info = json::object(),
        const json& data_schema = json::object()
    ) const;
};

#endif // E3_AGENT_HPP

