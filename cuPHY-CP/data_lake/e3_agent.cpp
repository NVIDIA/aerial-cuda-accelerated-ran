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

#include "e3_agent.hpp"
#include "data_lake.hpp"
#include "fmt/format.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>

// Constructor
E3Agent::E3Agent(
    DataLake* dl,
    const uint16_t repPort,
    const uint16_t pubPort,
    const uint16_t subPort,
    const int rowsFh,
    const int rowsPusch,
    const int rowsHest,
    const uint32_t fhSamples,
    const uint32_t maxPuschPdu,
    const uint32_t maxHestSamples,
    const int rowsSrsIq,
    const int rowsSrs,
    const int rowsSrsHest,
    const uint32_t maxSrsIqSamples,
    const uint32_t maxSrsHestBytes,
    const uint32_t maxSrsRbSnrBytes
) :
    dataLake(dl),
    e3RepPort(repPort),
    e3PubPort(pubPort),
    e3SubPort(subPort),
    numRowsToInsertFh(rowsFh),
    numRowsToInsertPusch(rowsPusch),
    numRowsToInsertHest(rowsHest),
    numFhSamples(fhSamples),
    maxPuschPduSize(maxPuschPdu),
    maxHestSamplesPerRow(maxHestSamples),
    numRowsToInsertSrsIq(rowsSrsIq),
    numRowsToInsertSrs(rowsSrs),
    numRowsToInsertSrsHest(rowsSrsHest),
    maxSrsIqSamplesPerRow(maxSrsIqSamples),
    maxSrsHestBytesPerRow(maxSrsHestBytes),
    maxSrsRbSnrBytesPerRow(maxSrsRbSnrBytes),
    zmq_context(1),
    e3_rep_socket(zmq_context, ZMQ_REP),
    e3_pub_socket(zmq_context, ZMQ_PUB),
    e3_sub_socket(zmq_context, ZMQ_SUB)
{
}

// Destructor
E3Agent::~E3Agent()
{
    shutdown();

    if (shm_data_ptr != nullptr && shm_data_ptr != MAP_FAILED) {
        munmap(shm_data_ptr, shm_data_size);
    }
    if (shm_data_fd != -1) {
        close(shm_data_fd);
        shm_unlink(E3_SHARED_MEMORY_KEY.data());
    }
}

// Initialize E3 agent - bind sockets and start threads
bool E3Agent::init()
{
    if (e3_running.load()) {
        return true;
    }

    NVLOGC_FMT(TAG_E3, "Initializing E3 Agent...");

    try {
        e3_rep_socket.bind("tcp://*:" + std::to_string(e3RepPort));
        e3_pub_socket.bind("tcp://*:" + std::to_string(e3PubPort));
        e3_sub_socket.bind("tcp://*:" + std::to_string(e3SubPort));

        e3_rep_socket.set(zmq::sockopt::tcp_keepalive, 1);
        e3_rep_socket.set(zmq::sockopt::tcp_keepalive_idle, 5);
        e3_rep_socket.set(zmq::sockopt::tcp_keepalive_intvl, 2);
        e3_rep_socket.set(zmq::sockopt::tcp_keepalive_cnt, 3);

        e3_sub_socket.set(zmq::sockopt::subscribe, "");
        e3_sub_socket.set(zmq::sockopt::linger, 1000);  // 1 second linger to allow graceful shutdown

        NVLOGC_FMT(TAG_E3, "E3 sockets initialized - REP: {}, PUB: {}, SUB: {}", e3RepPort, e3PubPort, e3SubPort);
    } catch (const zmq::error_t& e) {
        NVLOGC_FMT(TAG_E3, "Failed to initialize E3 sockets: {}", e.what());
        return false;
    }

    e3_running = true;
    e3_data_thread = std::thread(&E3Agent::dataServerThread, this);

    e3_reaper_running = true;
    e3_reaper_thread = std::thread(&E3Agent::reaperThread, this);

    e3_sub_running = true;
    e3_sub_thread = std::thread(&E3Agent::managerSubscriptionThread, this);

    {
        std::lock_guard<std::mutex> lock(dataLake->e3_buffer_mutex);
        dataLake->e3_buffer_info = {};
        dataLake->e3_buffer_info.ue_metrics.reserve(slot_command_api::MAX_PUSCH_UE_PER_TTI);
    }
    {
        std::lock_guard<std::mutex> lock(dataLake->e3_srs_buffer_mutex);
        dataLake->e3_srs_buffer_info = {};
    }

    NVLOGC_FMT(TAG_E3, "E3 Agent initialized successfully.");
    return true;
}

// Shutdown E3 agent - stop threads and close sockets
void E3Agent::shutdown()
{
    if (e3_running) {
        e3_running = false;
        if (e3_data_thread.joinable()) {
            e3_data_thread.join();
        }
        NVLOGC_FMT(TAG_E3, "E3 data server thread shutdown");
    }

    if (e3_sub_running) {
        e3_sub_running = false;
        if (e3_sub_thread.joinable()) {
            e3_sub_thread.join();
        }
        NVLOGC_FMT(TAG_E3, "E3 subscription thread shutdown");
    }

    if (e3_reaper_running) {
        e3_reaper_running = false;
        if (e3_reaper_thread.joinable()) {
            e3_reaper_thread.join();
        }
        NVLOGC_FMT(TAG_E3, "E3 reaper thread shutdown");
    }

    try {
        e3_pub_socket.close();
        e3_rep_socket.close();
        e3_sub_socket.close();
        NVLOGC_FMT(TAG_E3, "E3 sockets closed successfully");
    } catch (const zmq::error_t& e) {
        NVLOGC_FMT(TAG_E3, "Error closing E3 sockets: {}", e.what());
    }
}

// Create shared memory buffers for data exchange
bool E3Agent::createSharedMemoryBuffers(
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
)
{
    NVLOGC_FMT(TAG_E3, "Creating shared memory buffers for E3");

    // FH / PUSCH / Hest buffer sizes
    const size_t fh_buffer_size = numFhSamples * numRowsToInsertFh * sizeof(int16_t);
    const size_t pusch_buffer_size = maxPuschPduSize * numRowsToInsertPusch;
    const size_t hest_buffer_size = maxHestSamplesPerRow * numRowsToInsertHest * sizeof(hestDataType);

    // SRS path buffer sizes
    const size_t srs_iq_buffer_size = maxSrsIqSamplesPerRow * numRowsToInsertSrsIq * sizeof(int16_t);
    const size_t srs_rb_snr_buffer_size = maxSrsRbSnrBytesPerRow * numRowsToInsertSrs;
    const size_t srs_hest_buffer_size = maxSrsHestBytesPerRow * numRowsToInsertSrsHest;

    const size_t total_size = sizeof(SharedMemoryHeader) +
                             (2 * fh_buffer_size) +
                             (2 * pusch_buffer_size) +
                             (2 * hest_buffer_size) +
                             (2 * srs_iq_buffer_size) +
                             (2 * srs_rb_snr_buffer_size) +
                             (2 * srs_hest_buffer_size);

    shm_data_fd = shm_open(E3_SHARED_MEMORY_KEY.data(), O_CREAT | O_RDWR, 0666);
    if (shm_data_fd == -1) {
        NVLOGE_FMT(TAG_E3, AERIAL_SYSTEM_API_EVENT, "Failed to create shared memory, errno: {}", errno);
        return false;
    }

    if (ftruncate(shm_data_fd, total_size) == -1) {
        NVLOGE_FMT(TAG_E3, AERIAL_SYSTEM_API_EVENT, "Failed to set shared memory size, errno: {}", errno);
        close(shm_data_fd);
        shm_unlink(E3_SHARED_MEMORY_KEY.data());
        return false;
    }

    shm_data_ptr = mmap(nullptr, total_size, PROT_READ | PROT_WRITE,
                        MAP_SHARED, shm_data_fd, 0);
    if (shm_data_ptr == MAP_FAILED) {
        NVLOGE_FMT(TAG_E3, AERIAL_SYSTEM_API_EVENT, "Failed to map shared memory, errno: {}", errno);
        close(shm_data_fd);
        shm_unlink(E3_SHARED_MEMORY_KEY.data());
        return false;
    }

    shm_data_size = total_size;

    SharedMemoryHeader* header = static_cast<SharedMemoryHeader*>(shm_data_ptr);
    memset(header, 0, sizeof(SharedMemoryHeader));
    header->version = 0x010100;  // v1.1.0
    // FH / PUSCH / Hest
    header->fh_buffer_size = fh_buffer_size;
    header->pusch_buffer_size = pusch_buffer_size;
    header->hest_buffer_size = hest_buffer_size;
    header->num_fh_samples = numFhSamples;
    header->num_fh_rows = numRowsToInsertFh;
    header->num_pusch_rows = numRowsToInsertPusch;
    header->num_hest_rows = numRowsToInsertHest;
    header->max_hest_samples_per_row = maxHestSamplesPerRow;
    // SRS IQ / SRS RbSNR / SRS Hest
    header->srs_iq_buffer_size = srs_iq_buffer_size;
    header->num_srs_iq_samples = maxSrsIqSamplesPerRow;
    header->num_srs_iq_rows = numRowsToInsertSrsIq;
    header->srs_rb_snr_buffer_size = srs_rb_snr_buffer_size;
    header->max_srs_rb_snr_bytes_per_row = maxSrsRbSnrBytesPerRow;
    header->num_srs_rb_snr_rows = numRowsToInsertSrs;
    header->srs_hest_buffer_size = srs_hest_buffer_size;
    header->max_srs_hest_bytes_per_row = maxSrsHestBytesPerRow;
    header->num_srs_hest_rows = numRowsToInsertSrsHest;

    // Lay out ping-pong buffers contiguously after header
    uint8_t* base_ptr = reinterpret_cast<uint8_t*>(header + 1);
    size_t offset = 0;

    // PUSCH path: FH, PUSCH PDU, Hest
    (*pFh)->pDataAlloc = reinterpret_cast<int16_t*>(base_ptr + offset);
    offset += fh_buffer_size;
    (*pInsertFh)->pDataAlloc = reinterpret_cast<int16_t*>(base_ptr + offset);
    offset += fh_buffer_size;

    (*p)->pDataAlloc = reinterpret_cast<uint8_t*>(base_ptr + offset);
    offset += pusch_buffer_size;
    (*pInsertPusch)->pDataAlloc = reinterpret_cast<uint8_t*>(base_ptr + offset);
    offset += pusch_buffer_size;

    (*pHest)->pDataAlloc = reinterpret_cast<hestDataType*>(base_ptr + offset);
    offset += hest_buffer_size;
    (*pInsertHest)->pDataAlloc = reinterpret_cast<hestDataType*>(base_ptr + offset);
    offset += hest_buffer_size;

    // SRS path: SRS IQ, SRS RbSNR, SRS Hest
    (*pSrsIq)->pDataAlloc = reinterpret_cast<int16_t*>(base_ptr + offset);
    offset += srs_iq_buffer_size;
    (*pInsertSrsIq)->pDataAlloc = reinterpret_cast<int16_t*>(base_ptr + offset);
    offset += srs_iq_buffer_size;

    (*pSrs)->pRbSnrDataAlloc = reinterpret_cast<float*>(base_ptr + offset);
    offset += srs_rb_snr_buffer_size;
    (*pInsertSrs)->pRbSnrDataAlloc = reinterpret_cast<float*>(base_ptr + offset);
    offset += srs_rb_snr_buffer_size;

    (*pSrsHest)->pDataAlloc = reinterpret_cast<int16_t*>(base_ptr + offset);
    offset += srs_hest_buffer_size;
    (*pInsertSrsHest)->pDataAlloc = reinterpret_cast<int16_t*>(base_ptr + offset);
    offset += srs_hest_buffer_size;

    NVLOGC_FMT(TAG_E3, "Shared memory buffers created successfully");
    NVLOGC_FMT(TAG_E3, "  Total size: {} bytes ({:.1f} MB)", total_size, total_size / (1024.0 * 1024.0));
    NVLOGC_FMT(TAG_E3, "  FH buffers: {} bytes each", fh_buffer_size);
    NVLOGC_FMT(TAG_E3, "  PUSCH buffers: {} bytes each", pusch_buffer_size);
    NVLOGC_FMT(TAG_E3, "  PUSCH Hest buffers: {} bytes each", hest_buffer_size);
    NVLOGC_FMT(TAG_E3, "  SRS IQ buffers: {} bytes each", srs_iq_buffer_size);
    NVLOGC_FMT(TAG_E3, "  SRS RbSNR buffers: {} bytes each", srs_rb_snr_buffer_size);
    NVLOGC_FMT(TAG_E3, "  SRS Hest buffers: {} bytes each", srs_hest_buffer_size);

    return true;
}


// Notify subscribers that PUSCH data is ready
// Flow: gate → periodicity → bit-walks (cell + per-UE) → empty check → send → update timestamp.
void E3Agent::notifyDataReady()
{
    if (!e3_running) {
        return;
    }

    NVLOGD_FMT(TAG_E3, "TIMESTAMP_LOG: e3NotifyDataReady entry at {}", std::chrono::high_resolution_clock::now().time_since_epoch().count());

    E3BufferInfo buffer_info;
    {
        std::lock_guard<std::mutex> lock(dataLake->e3_buffer_mutex);
        buffer_info = dataLake->e3_buffer_info;
    }

    std::lock_guard<std::mutex> lock(e3_subscriptions_mutex);
    for (auto& [sub_id, sub] : e3_subscriptions) {
        // Fire PUSCH path only when sub has PUSCH-exclusive streams, or has
        // shared-only streams (no SRS-exclusive).
        {
            const bool has_pusch_only = (sub.stream_bitfield & e3::PUSCH_ONLY_PROVIDABLE_STREAMS) != static_cast<e3::StreamType>(0);
            const bool has_srs_only   = (sub.stream_bitfield & e3::SRS_ONLY_PROVIDABLE_STREAMS)  != static_cast<e3::StreamType>(0);
            const bool has_shared     = (sub.stream_bitfield & e3::SHARED_PROVIDABLE_STREAMS)    != static_cast<e3::StreamType>(0);
            if (!has_pusch_only && !(!has_srs_only && has_shared)) {
                continue;
            }
        }

        // Respect per-subscription periodicity on the PUSCH path
        const auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::microseconds>(now - sub.last_update_pusch).count() < sub.periodicity_us) {
            continue;
        }

        json notif_json;
        notif_json["type"] = "indicationMessage";
        notif_json["id"] = generateMessageId();
        notif_json["dAppIdentifier"] = sub.dapp_id;
        notif_json["ranFunctionIdentifier"] = sub.ran_function_id;
        notif_json["subscriptionId"] = sub.subscription_id;

        json protocolData;

        // Cell-level streams: shared + PUSCH-only
        const e3::StreamType cell_streams = sub.stream_bitfield & e3::PUSCH_PROVIDABLE_STREAMS & ~e3::PER_UE_STREAMS;
        __uint128_t remaining = static_cast<__uint128_t>(cell_streams);
        while (remaining != 0) {
            const __uint128_t lowest_bit = remaining & (~remaining + 1);
            switch (static_cast<e3::StreamType>(lowest_bit)) {
                case e3::StreamType::IQ_SAMPLES: {
                    json iq_shm_data;
                    iq_shm_data["shm_name"] = E3_SHARED_MEMORY_KEY;
                    iq_shm_data["fh_buffer_index"] = static_cast<int>(buffer_info.current_fh_buffer);
                    iq_shm_data["fh_write_index"] = buffer_info.fh_write_index;
                    protocolData["iq_samples"] = iq_shm_data;
                    break;
                }
                case e3::StreamType::PDU_DATA: {
                    json pdu_shm_data;
                    pdu_shm_data["shm_name"] = E3_SHARED_MEMORY_KEY;
                    pdu_shm_data["pusch_buffer_index"] = static_cast<int>(buffer_info.current_pusch_buffer);
                    pdu_shm_data["pusch_write_index"] = buffer_info.pusch_write_index;
                    protocolData["pdu_data"] = pdu_shm_data;
                    break;
                }
                case e3::StreamType::H_ESTIMATES: {
                    json hest_shm_data;
                    hest_shm_data["shm_name"] = E3_SHARED_MEMORY_KEY;
                    hest_shm_data["hest_buffer_index"] = static_cast<int>(buffer_info.current_hest_buffer);
                    hest_shm_data["hest_write_index"] = buffer_info.hest_write_index;
                    hest_shm_data["hest_row_byte_offset"] = buffer_info.hest_row_byte_offset;
                    protocolData["h_estimates"] = hest_shm_data;
                    break;
                }
                case e3::StreamType::TIMESTAMP:
                    protocolData["timestamp"] = buffer_info.timestamp_ns;
                    break;
                case e3::StreamType::TIMESTAMP_TAI:
                    protocolData["timestamp_tai"] = buffer_info.timestamp_tai_ns;
                    break;
                case e3::StreamType::SFN:
                    protocolData["sfn"] = buffer_info.sfn;
                    break;
                case e3::StreamType::SLOT:
                    protocolData["slot"] = buffer_info.slot;
                    break;
                case e3::StreamType::CELL_ID:
                    protocolData["cell_id"] = buffer_info.cell_id;
                    break;
                case e3::StreamType::N_RX_ANT:
                    protocolData["n_rx_ant"] = buffer_info.n_rx_ant;
                    break;
                case e3::StreamType::N_RX_ANT_SRS:
                    protocolData["n_rx_ant_srs"] = buffer_info.n_rx_ant_srs;
                    break;
                case e3::StreamType::N_CELLS:
                    protocolData["n_cells"] = buffer_info.n_cells;
                    break;
                case e3::StreamType::N_BS_ANTS:
                    protocolData["n_bs_ants"] = buffer_info.n_bs_ants;
                    break;
                case e3::StreamType::N_UE:
                    protocolData["n_ue"] = buffer_info.n_ue;
                    break;
                default:
                    break;
            }
            remaining &= ~lowest_bit;
        }

        // Per-UE streams: shared (RNTI) + PUSCH-only
        const e3::StreamType ue_streams = sub.stream_bitfield & e3::PUSCH_PROVIDABLE_STREAMS & e3::PER_UE_STREAMS;
        if (static_cast<__uint128_t>(ue_streams) != 0 && !buffer_info.ue_metrics.empty()) {
            json ue_metrics_arr = json::array();
            for (const auto& ue : buffer_info.ue_metrics) {
                json ue_obj;
                __uint128_t ue_remaining = static_cast<__uint128_t>(ue_streams);
                while (ue_remaining != 0) {
                    const __uint128_t lowest_bit = ue_remaining & (~ue_remaining + 1);
                    switch (static_cast<e3::StreamType>(lowest_bit)) {
                        case e3::StreamType::RNTI:
                            ue_obj["rnti"] = ue.rnti;
                            break;
                        case e3::StreamType::TB_CRC_FAIL:
                            ue_obj["tb_crc_fail"] = ue.tb_crc_fail;
                            break;
                        case e3::StreamType::CB_ERRORS:
                            ue_obj["cb_errors"] = ue.cb_errors;
                            break;
                        case e3::StreamType::RSRP:
                            ue_obj["rsrp"] = ue.rsrp;
                            break;
                        case e3::StreamType::NOISE_VAR:
                            ue_obj["noise_var"] = ue.noise_var;
                            break;
                        case e3::StreamType::CB_COUNT:
                            ue_obj["cb_count"] = ue.cb_count;
                            break;
                        case e3::StreamType::RSSI:
                            ue_obj["rssi"] = ue.rssi;
                            break;
                        case e3::StreamType::QAM_MOD_ORDER:
                            ue_obj["qam_mod_order"] = ue.qam_mod_order;
                            break;
                        case e3::StreamType::MCS_INDEX:
                            ue_obj["mcs_index"] = ue.mcs_index;
                            break;
                        case e3::StreamType::MCS_TABLE_INDEX:
                            ue_obj["mcs_table_index"] = ue.mcs_table_index;
                            break;
                        case e3::StreamType::RB_START:
                            ue_obj["rb_start"] = ue.rb_start;
                            break;
                        case e3::StreamType::RB_SIZE:
                            ue_obj["rb_size"] = ue.rb_size;
                            break;
                        case e3::StreamType::START_SYMBOL_INDEX:
                            ue_obj["start_symbol_index"] = ue.start_symbol_index;
                            break;
                        case e3::StreamType::NR_OF_SYMBOLS:
                            ue_obj["nr_of_symbols"] = ue.nr_of_symbols;
                            break;
                        case e3::StreamType::N_LAYERS:
                            ue_obj["n_layers"] = ue.n_layers;
                            break;
                        case e3::StreamType::TB_SIZE:
                            ue_obj["tb_size"] = ue.tb_size;
                            break;
                        case e3::StreamType::PDU_LEN:
                            ue_obj["pdu_len"] = ue.pdu_len;
                            break;
                        case e3::StreamType::TARGET_CODE_RATE:
                            ue_obj["target_code_rate"] = ue.target_code_rate;
                            break;
                        case e3::StreamType::NEW_DATA_INDICATOR:
                            ue_obj["new_data_indicator"] = ue.new_data_indicator;
                            break;
                        case e3::StreamType::LAYER_OFFSET:
                            ue_obj["layer_offset"] = ue.layer_offset;
                            break;
                        case e3::StreamType::UE_GRP_IDX:
                            ue_obj["ue_grp_idx"] = ue.ue_grp_idx;
                            break;
                        case e3::StreamType::H_OFFSET:
                            ue_obj["h_offset"] = ue.h_offset;
                            break;
                        case e3::StreamType::H_SIZE:
                            ue_obj["h_size"] = ue.h_size;
                            break;
                        case e3::StreamType::N_SUBCARRIERS:
                            ue_obj["n_subcarriers"] = ue.n_subcarriers;
                            break;
                        case e3::StreamType::N_DMRS_ESTIMATES:
                            ue_obj["n_dmrs_estimates"] = ue.n_dmrs_estimates;
                            break;
                        case e3::StreamType::DMRS_SYMB_POS:
                            ue_obj["dmrs_symb_pos"] = ue.dmrs_symb_pos;
                            break;
                        case e3::StreamType::SINR:
                            ue_obj["sinr"] = ue.sinr;
                            break;
                        case e3::StreamType::TIMING_ADVANCE:
                            ue_obj["timing_advance"] = ue.timing_advance;
                            break;
                        case e3::StreamType::HARQ_PROCESS_ID:
                            ue_obj["harq_process_id"] = ue.harq_process_id;
                            break;
                        case e3::StreamType::RV_INDEX:
                            ue_obj["rv_index"] = ue.rv_index;
                            break;
                        case e3::StreamType::CFO_HZ:
                            ue_obj["cfo_hz"] = ue.cfo_hz;
                            break;
                        default:
                            break;
                    }
                    ue_remaining &= ~lowest_bit;
                }
                if (!ue_obj.empty()) {
                    ue_metrics_arr.push_back(std::move(ue_obj));
                }
            }
            protocolData["ue_metrics"] = std::move(ue_metrics_arr);
        }

        if (protocolData.empty()) {
            continue;
        }
        notif_json["protocolData"] = protocolData;

        try {
            NVLOGD_FMT(TAG_E3, "TIMESTAMP_LOG: Before PUSCH ZMQ send at {}", std::chrono::high_resolution_clock::now().time_since_epoch().count());
            const std::string message = notif_json.dump();
            {
                std::lock_guard<std::mutex> lock(e3_pub_socket_mutex_);
                e3_pub_socket.send(zmq::buffer(message), zmq::send_flags::dontwait);
            }
            NVLOGD_FMT(TAG_E3, "TIMESTAMP_LOG: After PUSCH ZMQ send at {}", std::chrono::high_resolution_clock::now().time_since_epoch().count());
            NVLOGD_FMT(TAG_E3, "Sent PUSCH E3 indication to dApp {} for subscription {}", sub.dapp_id, sub.subscription_id);
        } catch (const zmq::error_t& e) {
            NVLOGD_FMT(TAG_E3, "PUSCH E3 indication send failed: {}", e.what());
        }
        sub.last_update_pusch = now;
    }
}

// Notify subscribers that SRS data is ready
// Flow: gate → periodicity → bit-walks (cell + per-UE) → empty check → send → update timestamp.
void E3Agent::notifySrsDataReady()
{
    if (!e3_running) {
        return;
    }

    NVLOGD_FMT(TAG_E3, "TIMESTAMP_LOG: e3NotifySrsDataReady entry at {}", std::chrono::high_resolution_clock::now().time_since_epoch().count());

    E3SrsBufferInfo srs_info;
    {
        std::lock_guard<std::mutex> lock(dataLake->e3_srs_buffer_mutex);
        srs_info = dataLake->e3_srs_buffer_info;
    }

    std::lock_guard<std::mutex> lock(e3_subscriptions_mutex);
    for (auto& [sub_id, sub] : e3_subscriptions) {
        // Fire SRS path only when sub has SRS-exclusive streams, or has
        // shared-only streams (no PUSCH-exclusive).
        {
            const bool has_srs_only   = (sub.stream_bitfield & e3::SRS_ONLY_PROVIDABLE_STREAMS)  != static_cast<e3::StreamType>(0);
            const bool has_pusch_only = (sub.stream_bitfield & e3::PUSCH_ONLY_PROVIDABLE_STREAMS) != static_cast<e3::StreamType>(0);
            const bool has_shared     = (sub.stream_bitfield & e3::SHARED_PROVIDABLE_STREAMS)    != static_cast<e3::StreamType>(0);
            if (!has_srs_only && !(!has_pusch_only && has_shared)) {
                continue;
            }
        }

        // Respect per-subscription periodicity on the SRS path
        const auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::microseconds>(now - sub.last_update_srs).count() < sub.periodicity_us) {
            continue;
        }

        json notif_json;
        notif_json["type"] = "indicationMessage";
        notif_json["id"] = generateMessageId();
        notif_json["dAppIdentifier"] = sub.dapp_id;
        notif_json["ranFunctionIdentifier"] = sub.ran_function_id;
        notif_json["subscriptionId"] = sub.subscription_id;

        json protocolData;

        // Cell-level streams: shared + SRS-only
        const e3::StreamType cell_streams = sub.stream_bitfield & e3::SRS_PROVIDABLE_STREAMS & ~e3::PER_UE_STREAMS;
        __uint128_t remaining = static_cast<__uint128_t>(cell_streams);
        while (remaining != 0) {
            const __uint128_t lowest_bit = remaining & (~remaining + 1);
            switch (static_cast<e3::StreamType>(lowest_bit)) {
                // Shared cell-level
                case e3::StreamType::TIMESTAMP:
                    protocolData["timestamp"] = srs_info.timestamp_ns;
                    break;
                case e3::StreamType::TIMESTAMP_TAI:
                    protocolData["timestamp_tai"] = srs_info.timestamp_tai_ns;
                    break;
                case e3::StreamType::SFN:
                    protocolData["sfn"] = srs_info.sfn;
                    break;
                case e3::StreamType::SLOT:
                    protocolData["slot"] = srs_info.slot;
                    break;
                case e3::StreamType::CELL_ID:
                    protocolData["cell_id"] = srs_info.cell_id;
                    break;
                case e3::StreamType::N_RX_ANT_SRS:
                    protocolData["n_rx_ant_srs"] = srs_info.n_rx_ant_srs;
                    break;
                case e3::StreamType::N_CELLS:
                    protocolData["n_cells"] = srs_info.n_cells;
                    break;
                // SRS-only cell-level
                case e3::StreamType::SRS_IQ_SAMPLES: {
                    json srs_iq_shm;
                    srs_iq_shm["shm_name"] = E3_SHARED_MEMORY_KEY;
                    srs_iq_shm["srs_iq_buffer_index"] = static_cast<int>(srs_info.current_srs_iq_buffer);
                    srs_iq_shm["srs_iq_write_index"] = srs_info.srs_iq_write_index;
                    srs_iq_shm["srs_iq_row_byte_offset"] = srs_info.srs_iq_row_byte_offset;
                    protocolData["srs_iq_samples"] = srs_iq_shm;
                    break;
                }
                case e3::StreamType::SRS_HEST: {
                    json srs_hest_shm;
                    srs_hest_shm["shm_name"] = E3_SHARED_MEMORY_KEY;
                    srs_hest_shm["srs_hest_buffer_index"] = static_cast<int>(srs_info.current_srs_hest_buffer);
                    srs_hest_shm["srs_hest_write_index"] = srs_info.srs_hest_write_index;
                    protocolData["srs_hest"] = srs_hest_shm;
                    break;
                }
                case e3::StreamType::SRS_RB_SNR: {
                    json srs_rbsnr_shm;
                    srs_rbsnr_shm["shm_name"] = E3_SHARED_MEMORY_KEY;
                    srs_rbsnr_shm["srs_rb_snr_buffer_index"] = static_cast<int>(srs_info.current_srs_rb_snr_buffer);
                    srs_rbsnr_shm["srs_rb_snr_write_index"] = srs_info.srs_rb_snr_write_index;
                    protocolData["srs_rb_snr"] = srs_rbsnr_shm;
                    break;
                }
                case e3::StreamType::SRS_CELL_START_SYM:
                    protocolData["srs_cell_start_sym"] = srs_info.srs_cell_start_sym;
                    break;
                case e3::StreamType::SRS_CELL_N_SRS_SYM:
                    protocolData["srs_cell_n_srs_sym"] = srs_info.srs_cell_n_srs_sym;
                    break;
                case e3::StreamType::N_SRS_UE:
                    protocolData["n_srs_ue"] = srs_info.n_srs_ue;
                    break;
                default:
                    break;
            }
            remaining &= ~lowest_bit;
        }

        // Per-UE streams: shared (RNTI) + SRS-only
        const e3::StreamType ue_streams = sub.stream_bitfield & e3::SRS_PROVIDABLE_STREAMS & e3::PER_UE_STREAMS;
        if (static_cast<__uint128_t>(ue_streams) != 0 && !srs_info.ue_metrics.empty()) {
            json ue_metrics_arr = json::array();
            for (const auto& ue : srs_info.ue_metrics) {
                json ue_obj;
                __uint128_t ue_remaining = static_cast<__uint128_t>(ue_streams);
                while (ue_remaining != 0) {
                    const __uint128_t lowest_bit = ue_remaining & (~ue_remaining + 1);
                    switch (static_cast<e3::StreamType>(lowest_bit)) {
                        case e3::StreamType::RNTI:
                            ue_obj["rnti"] = ue.rnti;
                            break;
                        case e3::StreamType::SRS_WIDEBAND_SNR:
                            ue_obj["srs_wideband_snr"] = ue.wideband_snr;
                            break;
                        case e3::StreamType::SRS_SIGNAL_ENERGY:
                            ue_obj["srs_signal_energy"] = ue.signal_energy;
                            break;
                        case e3::StreamType::SRS_NOISE_ENERGY:
                            ue_obj["srs_noise_energy"] = ue.noise_energy;
                            break;
                        case e3::StreamType::SRS_TOA:
                            ue_obj["srs_toa"] = ue.toa_us;
                            break;
                        case e3::StreamType::SRS_HD_ANT_FLAG:
                            ue_obj["srs_hd_ant_flag"] = ue.hd_ant_flag;
                            break;
                        case e3::StreamType::SRS_SC_CORR:
                            ue_obj["srs_sc_corr"] = {ue.sc_corr_re, ue.sc_corr_im};
                            break;
                        case e3::StreamType::SRS_CS_CORR_RATIO_DB:
                            ue_obj["srs_cs_corr_ratio_db"] = ue.cs_corr_ratio_db;
                            break;
                        case e3::StreamType::SRS_ANT_PORTS:
                            ue_obj["srs_ant_ports"] = ue.n_ant_ports;
                            break;
                        case e3::StreamType::SRS_N_SYMS:
                            ue_obj["srs_n_syms"] = ue.n_syms;
                            break;
                        case e3::StreamType::SRS_N_REPETITIONS:
                            ue_obj["srs_n_repetitions"] = ue.n_repetitions;
                            break;
                        case e3::StreamType::SRS_COMB_SIZE:
                            ue_obj["srs_comb_size"] = ue.comb_size;
                            break;
                        case e3::StreamType::SRS_COMB_OFFSET:
                            ue_obj["srs_comb_offset"] = ue.comb_offset;
                            break;
                        case e3::StreamType::SRS_START_SYM:
                            ue_obj["srs_start_sym"] = ue.start_sym;
                            break;
                        case e3::StreamType::SRS_CYCLIC_SHIFT:
                            ue_obj["srs_cyclic_shift"] = ue.cyclic_shift;
                            break;
                        case e3::StreamType::SRS_FREQ_POSITION:
                            ue_obj["srs_freq_position"] = ue.frequency_position;
                            break;
                        case e3::StreamType::SRS_FREQ_SHIFT:
                            ue_obj["srs_freq_shift"] = ue.frequency_shift;
                            break;
                        case e3::StreamType::SRS_FREQ_HOPPING:
                            ue_obj["srs_freq_hopping"] = ue.frequency_hopping;
                            break;
                        case e3::StreamType::SRS_RESOURCE_TYPE:
                            ue_obj["srs_resource_type"] = ue.resource_type;
                            break;
                        case e3::StreamType::SRS_T_SRS:
                            ue_obj["srs_t_srs"] = ue.t_srs;
                            break;
                        case e3::StreamType::SRS_T_OFFSET:
                            ue_obj["srs_t_offset"] = ue.t_offset;
                            break;
                        case e3::StreamType::SRS_USAGE:
                            ue_obj["srs_usage"] = ue.usage;
                            break;
                        case e3::StreamType::SRS_N_VALID_PRG:
                            ue_obj["srs_n_valid_prg"] = ue.n_valid_prg;
                            break;
                        case e3::StreamType::SRS_PRG_SIZE:
                            ue_obj["srs_prg_size"] = ue.prg_size;
                            break;
                        case e3::StreamType::SRS_HEST_N_PRB_GRPS:
                            ue_obj["srs_hest_n_prb_grps"] = ue.n_prb_grps;
                            break;
                        case e3::StreamType::SRS_HEST_OFFSET:
                            ue_obj["srs_hest_offset"] = ue.srs_hest_offset;
                            break;
                        case e3::StreamType::SRS_HEST_SIZE:
                            ue_obj["srs_hest_size"] = ue.srs_hest_size;
                            break;
                        case e3::StreamType::SRS_RB_SNR_OFFSET:
                            ue_obj["srs_rb_snr_offset"] = ue.srs_rb_snr_offset;
                            break;
                        case e3::StreamType::SRS_RB_SNR_SIZE:
                            ue_obj["srs_rb_snr_size"] = ue.srs_rb_snr_size;
                            break;
                        default:
                            break;
                    }
                    ue_remaining &= ~lowest_bit;
                }
                if (!ue_obj.empty()) {
                    ue_metrics_arr.push_back(std::move(ue_obj));
                }
            }
            protocolData["ue_metrics"] = std::move(ue_metrics_arr);
        }

        if (protocolData.empty()) {
            continue;
        }
        notif_json["protocolData"] = protocolData;

        try {
            NVLOGD_FMT(TAG_E3, "TIMESTAMP_LOG: Before SRS ZMQ send at {}", std::chrono::high_resolution_clock::now().time_since_epoch().count());
            const std::string message = notif_json.dump();
            {
                std::lock_guard<std::mutex> lock(e3_pub_socket_mutex_);
                e3_pub_socket.send(zmq::buffer(message), zmq::send_flags::dontwait);
            }
            NVLOGD_FMT(TAG_E3, "TIMESTAMP_LOG: After SRS ZMQ send at {}", std::chrono::high_resolution_clock::now().time_since_epoch().count());
            NVLOGD_FMT(TAG_E3, "Sent SRS E3 indication to dApp {} for subscription {}", sub.dapp_id, sub.subscription_id);
        } catch (const zmq::error_t& e) {
            NVLOGD_FMT(TAG_E3, "SRS E3 indication send failed: {}", e.what());
        }
        sub.last_update_srs = now;
    }
}

// Thread functions

// E3 data server thread - handles ZMQ request/reply
void E3Agent::dataServerThread()
{
    e3_rep_socket.set(zmq::sockopt::rcvtimeo, 1000);

    NVLOGC_FMT(TAG_E3, "E3 data server thread started");

    while (e3_running) {
        zmq::message_t request;
        if (e3_rep_socket.recv(request, zmq::recv_flags::none)) {
            std::string response;
            try {
                const json req_json = json::parse(std::string(static_cast<char*>(request.data()), request.size()));
                const std::string type = req_json.value("type", "");

                NVLOGC_FMT(TAG_E3, "Received E3 request: {}", req_json.dump());

                if (type == "setupRequest") {
                    handleSetupRequest(req_json, response);
                } else {
                    json error_resp;
                    error_resp["type"] = type;
                    error_resp["id"] = generateMessageId();
                    error_resp["requestId"] = req_json.value("id", 0u);
                    error_resp["responseCode"] = "negative";
                    error_resp["message"] = "unknown request type";
                    response = error_resp.dump();
                }
            } catch (const json::parse_error& e) {
                json error_resp;
                error_resp["responseCode"] = "negative";
                error_resp["message"] = "invalid JSON format";
                response = error_resp.dump();
                NVLOGC_FMT(TAG_E3, "Failed to parse request: {}", e.what());
            }
            e3_rep_socket.send(zmq::buffer(response));
        }
    }
    NVLOGC_FMT(TAG_E3, "E3 data server thread stopped");
}

// E3 reaper thread - cleanup disconnected dApps
void E3Agent::reaperThread()
{
    NVLOGC_FMT(TAG_E3, "E3 reaper thread started");

    while (e3_reaper_running) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        reapTimedOutDapps();
    }
    NVLOGC_FMT(TAG_E3, "E3 reaper thread stopped");
}

// Remove timed-out dApps
// NOTE: dApps with active subscriptions are kept alive even if inactive, since indications
// are fire-and-forget (no ACK). A crashed dApp with subscriptions won't be reaped until
// an explicit release message is sent.
void E3Agent::reapTimedOutDapps()
{
    constexpr auto ACTIVITY_TIMEOUT_SECONDS = 1800;

    const auto now = std::chrono::steady_clock::now();

    // Expire time-bounded subscriptions
    {
        std::lock_guard<std::mutex> lock(e3_subscriptions_mutex);
        for (auto it = e3_subscriptions.begin(); it != e3_subscriptions.end(); ) {
            if (now >= it->second.expiry_time) {
                NVLOGC_FMT(TAG_E3, "Subscription {} expired for dApp {}", it->first, it->second.dapp_id);
                it = e3_subscriptions.erase(it);
            } else {
                ++it;
            }
        }
    }

    std::vector<uint32_t> timed_out_dapps;

    {
        std::lock_guard<std::mutex> lock(e3_dapps_mutex);
        for (auto const& [dapp_id, conn_info] : e3_connected_dapps) {
            if (std::chrono::duration_cast<std::chrono::seconds>(now - conn_info.last_activity_time).count() > ACTIVITY_TIMEOUT_SECONDS) {
                bool has_active_subscriptions = false;
                {
                    std::lock_guard<std::mutex> subs_lock(e3_subscriptions_mutex);
                    for (const auto& [sub_id, sub] : e3_subscriptions) {
                        if (sub.dapp_id == dapp_id) {
                            has_active_subscriptions = true;
                            break;
                        }
                    }
                }

                if (!has_active_subscriptions) {
                    timed_out_dapps.push_back(dapp_id);
                }
            }
        }
    }

    for (const uint32_t dapp_id : timed_out_dapps) {
        NVLOGC_FMT(TAG_E3, "dApp {} timed out after {} seconds of inactivity. Releasing.", dapp_id, ACTIVITY_TIMEOUT_SECONDS);
        sendRelease(dapp_id);
    }
}

// Manager subscription thread - receives commands from E3 Manager
void E3Agent::managerSubscriptionThread()
{
    NVLOGC_FMT(TAG_E3, "E3 Manager subscription thread started");

    while (e3_sub_running) {
        try {
            zmq::message_t msg;
            const auto result = e3_sub_socket.recv(msg, zmq::recv_flags::dontwait);

            if (result) {
                try {
                    const json msg_json = json::parse(msg.to_string());
                    handleManagerMessage(msg_json);
                } catch (const json::exception& e) {
                    NVLOGC_FMT(TAG_E3, "Failed to parse dApp message: {}", e.what());
                }
            }
        } catch (const zmq::error_t& e) {
            if (e.num() == ETERM) {
                // Context terminated during shutdown - exit gracefully
                break;
            } else if (e.num() != EAGAIN) {
                NVLOGC_FMT(TAG_E3, "Error receiving from Manager: {}", e.what());
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    NVLOGC_FMT(TAG_E3, "E3 Manager subscription thread stopped");
}

// Handle messages asynchronously received from E3 Manager via PUB-SUB
void E3Agent::handleManagerMessage(const json& message)
{
    NVLOGD_FMT(TAG_E3, "Handling Manager message: {}", message.dump());

    const std::string type = message.value("type", "");

    if (type == "releaseMessage") {
        const uint32_t dapp_id = message.value("dAppIdentifier", 0u);
        if (dapp_id == 0) {
            NVLOGC_FMT(TAG_E3, "Received e3_release with invalid dAppIdentifier");
            return;
        }
        NVLOGC_FMT(TAG_E3, "Received e3_release from dApp {}", dapp_id);
        releaseDapp(dapp_id);

    } else if (type == "subscriptionRequest" || type == "subscriptionDelete") {
        // Subscription Request/Delete via PUB-SUB: process and publish response on Agent PUB
        const uint32_t dapp_id = message.value("dAppIdentifier", 0u);

        // Silently ignore requests for dApps we don't own (multi-agent correctness)
        {
            std::lock_guard<std::mutex> lock(e3_dapps_mutex);
            if (e3_connected_dapps.find(dapp_id) == e3_connected_dapps.end()) {
                return;
            }
        }

        std::string response;
        if (type == "subscriptionRequest") {
            handleSubscriptionRequest(message, response);
        } else if (type == "subscriptionDelete") {
            handleSubscriptionDelete(message, response);
        }

        // Publish response on Agent PUB socket
        try {
            {
                std::lock_guard<std::mutex> lock(e3_pub_socket_mutex_);
                e3_pub_socket.send(zmq::buffer(response), zmq::send_flags::dontwait);
            }
            NVLOGC_FMT(TAG_E3, "Published subscription response to dApp {}", dapp_id);
        } catch (const zmq::error_t& e) {
            NVLOGC_FMT(TAG_E3, "Failed to publish subscription response to dApp {}: {}", dapp_id, e.what());
        }

    } else if (type == "dAppControlAction") {
        const uint32_t dapp_id = message.value("dAppIdentifier", 0u);

        {
            std::lock_guard<std::mutex> lock(e3_dapps_mutex);
            if (e3_connected_dapps.find(dapp_id) == e3_connected_dapps.end()) {
                return;
            }
        }

        std::string response;
        handleControlMessage(message, response);

        // Optional ack to control message
        try {
            {
                std::lock_guard<std::mutex> lock(e3_pub_socket_mutex_);
                e3_pub_socket.send(zmq::buffer(response), zmq::send_flags::dontwait);
            }
        } catch (const zmq::error_t& e) {
            NVLOGC_FMT(TAG_E3, "Failed to publish control ack to dApp {}: {}", dapp_id, e.what());
        }
    }
}

bool E3Agent::updateDappActivity(uint32_t dapp_id)
{
    std::lock_guard<std::mutex> lock(e3_dapps_mutex);
    auto it = e3_connected_dapps.find(dapp_id);
    if (it == e3_connected_dapps.end()) return false;
    it->second.last_activity_time = std::chrono::steady_clock::now();
    return true;
}

// Release a dApp: remove all subscriptions and connection state
void E3Agent::releaseDapp(uint32_t dapp_id)
{
    std::lock_guard<std::mutex> dapps_lock(e3_dapps_mutex);
    std::lock_guard<std::mutex> subs_lock(e3_subscriptions_mutex);

    auto it = e3_connected_dapps.find(dapp_id);
    if (it == e3_connected_dapps.end()) {
        NVLOGC_FMT(TAG_E3, "Release: dApp {} not found, ignoring", dapp_id);
        return;
    }

    e3_connected_dapps.erase(it);

    for (auto sub_it = e3_subscriptions.begin(); sub_it != e3_subscriptions.end(); ) {
        if (sub_it->second.dapp_id == dapp_id) {
            NVLOGC_FMT(TAG_E3, "Release: removing subscription {} for dApp {}", sub_it->first, dapp_id);
            sub_it = e3_subscriptions.erase(sub_it);
        } else {
            ++sub_it;
        }
    }

    NVLOGC_FMT(TAG_E3, "dApp {} released successfully", dapp_id);
}

// Send e3_release to a dApp via PUB socket
bool E3Agent::sendRelease(uint32_t dapp_id)
{
    // Verify dApp exists before publishing
    {
        std::lock_guard<std::mutex> lock(e3_dapps_mutex);
        if (e3_connected_dapps.find(dapp_id) == e3_connected_dapps.end()) {
            NVLOGC_FMT(TAG_E3, "sendRelease: dApp {} not found", dapp_id);
            return false;
        }
    }

    json release_msg;
    release_msg["type"] = "releaseMessage";
    release_msg["id"] = generateMessageId();
    release_msg["dAppIdentifier"] = dapp_id;

    const std::string message = release_msg.dump();

    try {
        {
            std::lock_guard<std::mutex> lock(e3_pub_socket_mutex_);
            e3_pub_socket.send(zmq::buffer(message), zmq::send_flags::dontwait);
        }
        NVLOGC_FMT(TAG_E3, "Sent e3_release to dApp {}", dapp_id);
    } catch (const zmq::error_t& e) {
        NVLOGC_FMT(TAG_E3, "Failed to send e3_release to dApp {}: {}", dapp_id, e.what());
        return false;
    }

    releaseDapp(dapp_id);
    return true;
}

// E3AP Message helpers

uint32_t E3Agent::generateMessageId()
{
    static std::atomic<uint32_t> message_counter{1};
    return message_counter.fetch_add(1);
}

uint32_t E3Agent::generateDappId()
{
    static std::atomic<uint32_t> dapp_counter{1};
    return dapp_counter.fetch_add(1);
}

uint32_t E3Agent::generateSubscriptionId()
{
    static std::atomic<uint32_t> sub_counter{1};
    return sub_counter.fetch_add(1);
}

// Stream creation helpers

json E3Agent::createIndicationPayloadDelivery(const std::string& stream_id) const
{
    json delivery;
    delivery["transport_type"] = "protocolData";
    delivery["keyword"] = stream_id;
    delivery["encoding"] = "json";
    return delivery;
}

json E3Agent::createIndicationPayloadStream(
    const std::string& stream_id,
    const std::string& data_type,
    const std::string& description
) const
{
    json stream;
    __uint128_t val = static_cast<__uint128_t>(e3::streamNameToType(stream_id));
    if (val == 0) {
        NVLOGC_FMT(TAG_E3, "createIndicationPayloadStream: unknown stream_id '{}', telemetryIdentifier will be 0", stream_id);
        stream["telemetryIdentifier"] = 0;
    } else {
        uint32_t pos = 0;
        __uint128_t tmp = val;
        while (tmp >>= 1) ++pos;
        stream["telemetryIdentifier"] = pos + 1;
    }
    stream["stream_id"] = stream_id;
    stream["data_type"] = data_type;
    stream["description"] = description;
    stream["status"] = "available";
    stream["delivery_method"] = createIndicationPayloadDelivery(stream_id);
    return stream;
}

json E3Agent::createSharedMemoryStream(
    const std::string& stream_id,
    const std::string& data_type,
    const std::string& description,
    const size_t memory_size_bytes,
    const uint32_t max_elements,
    const json& additional_shm_info,
    const json& data_schema
) const
{
    json stream;
    __uint128_t val = static_cast<__uint128_t>(e3::streamNameToType(stream_id));
    if (val == 0) {
        NVLOGC_FMT(TAG_E3, "createSharedMemoryStream: unknown stream_id '{}', telemetryIdentifier will be 0", stream_id);
        stream["telemetryIdentifier"] = 0;
    } else {
        uint32_t pos = 0;
        __uint128_t tmp = val;
        while (tmp >>= 1) ++pos;
        stream["telemetryIdentifier"] = pos + 1;
    }
    stream["stream_id"] = stream_id;
    stream["data_type"] = data_type;
    stream["description"] = description;
    stream["status"] = "available";

    json delivery;
    delivery["transport_type"] = "shared_memory";

    json shm_info;
    shm_info["memory_key"] = E3_SHARED_MEMORY_KEY;
    shm_info["memory_size_bytes"] = memory_size_bytes;
    shm_info["access_pattern"] = "double_buffer";
    shm_info["max_elements"] = max_elements;

    if (!additional_shm_info.empty()) {
        shm_info.update(additional_shm_info);
    }

    delivery["shared_memory_info"] = shm_info;
    stream["delivery_method"] = delivery;

    if (!data_schema.empty()) {
        stream["data_schema"] = data_schema;
    }

    return stream;
}

// Request handlers

void E3Agent::handleSetupRequest(const json& request, std::string& response) {
	json response_json;
	json e3_setup_response;
	uint32_t request_id = 0;
	
	try {
		const json& e3_setup_req = request;
		
		request_id = e3_setup_req.value("id", 0u);
		std::string protocol_version = e3_setup_req.value("e3apProtocolVersion", "");
		std::string dapp_name = e3_setup_req.value("dAppName", "unknown");
		std::string dapp_version = e3_setup_req.value("dAppVersion", "unknown");
		std::string vendor = e3_setup_req.value("vendor", "unknown");
		
		NVLOGC_FMT(TAG_E3, "E3 Setup Request from dApp '{}' v{} by {} (E3AP v{})",
				   dapp_name, dapp_version, vendor, protocol_version);
		
		if (protocol_version != e3::E3AP_PROTOCOL_VERSION) {
			NVLOGC_FMT(TAG_E3, "E3 Setup rejected: protocol version mismatch (received '{}', expected '{}')",
					   protocol_version, e3::E3AP_PROTOCOL_VERSION);
			json error_resp;
			error_resp["type"] = "setupResponse";
			error_resp["id"] = generateMessageId();
			error_resp["requestId"] = request_id;
			error_resp["responseCode"] = "negative";
			error_resp["message"] = "protocol version mismatch";
			error_resp["e3apProtocolVersion"] = e3::E3AP_PROTOCOL_VERSION;
			response = error_resp.dump();
			return;
		}
		
		// Generate dApp ID during setup phase
		const uint32_t dapp_id = generateDappId();
		{
			std::lock_guard<std::mutex> lock(e3_dapps_mutex);
			e3_connected_dapps[dapp_id] = {std::chrono::steady_clock::now()};
		}
		
		// Create E3AP Setup Response
		e3_setup_response["type"] = "setupResponse";
		e3_setup_response["id"] = generateMessageId();
		e3_setup_response["requestId"] = request_id;
		e3_setup_response["responseCode"] = "positive";
		e3_setup_response["e3apProtocolVersion"] = e3::E3AP_PROTOCOL_VERSION;
		e3_setup_response["dAppIdentifier"] = dapp_id;
		e3_setup_response["ranIdentifier"] = e3::RAN_IDENTIFIER;
		
		// Available data streams
		json available_data_streams = json::array();
		
		// IQ Samples stream (FH Data)
		available_data_streams.push_back(createSharedMemoryStream(
			"iq_samples",
			"array(int16)",
			"Raw IQ samples (Fronthaul data)",
			numFhSamples * numRowsToInsertFh * sizeof(int16_t),
			numRowsToInsertFh
		));

		// PDU Data stream (PUSCH Data)
		available_data_streams.push_back(createSharedMemoryStream(
			"pdu_data",
			"array(uint8)",
			"PUSCH PDU data",
			maxPuschPduSize * numRowsToInsertPusch,
			numRowsToInsertPusch
		));
		
		// H Estimates stream
		json hest_shm_info;
		hest_shm_info["max_samples_per_row"] = maxHestSamplesPerRow;
		
		json hest_schema;
		hest_schema["dimensions"] = "Variable per group: (N_DMRS_ESTIMATES, N_SUBCARRIERS, N_BS_ANTS, N_LAYERS)";
		hest_schema["N_BS_ANTS"] = "Number of base station antennas (limited to 4)";
		hest_schema["N_LAYERS"] = "Total spatial layers in the group";
		hest_schema["N_SUBCARRIERS"] = "Number of subcarriers (PRBs * 12)";
		hest_schema["N_DMRS_ESTIMATES"] = "Number of DMRS estimates";
		
		available_data_streams.push_back(createSharedMemoryStream(
			"h_estimates",
			"array(complex64)",
			"PUSCH H matrix estimates, all UE groups concatenated",
			maxHestSamplesPerRow * numRowsToInsertHest * sizeof(hestDataType),
			numRowsToInsertHest,
			hest_shm_info,
			hest_schema
		));
		
		// Timing streams
		available_data_streams.push_back(createIndicationPayloadStream("timestamp", "uint64", "Agent-side software timestamp (ns)"));
		available_data_streams.push_back(createIndicationPayloadStream("timestamp_tai", "uint64", "TAI timestamp (ns) aligned from SFN/slot via grandmaster clock"));
		available_data_streams.push_back(createIndicationPayloadStream("sfn", "uint16", "Network frame timing information"));
		available_data_streams.push_back(createIndicationPayloadStream("slot", "uint16", "Network slot timing information"));
		available_data_streams.push_back(createIndicationPayloadStream("cell_id", "uint16", "Physical Cell ID"));
		
		// Antenna and cell configuration streams
		available_data_streams.push_back(createIndicationPayloadStream("n_rx_ant", "uint16", "Number of receive antennas"));
		available_data_streams.push_back(createIndicationPayloadStream("n_rx_ant_srs", "uint16", "Number of SRS receive antennas"));
		available_data_streams.push_back(createIndicationPayloadStream("n_cells", "uint16", "Number of cells"));
		
		// H Estimates metadata (cell-level)
		available_data_streams.push_back(createIndicationPayloadStream("n_bs_ants", "uint8", "Number of base station antennas in H estimates"));
		
		// Cell-level UE count
		available_data_streams.push_back(createIndicationPayloadStream("n_ue", "uint16", "Number of UEs scheduled in this slot"));
		
		// Per-UE streams (delivered inside ue_metrics[] array in protocolData)
		available_data_streams.push_back(createIndicationPayloadStream("rnti", "uint16", "UE Radio Network Temporary Identifier [per-UE]"));
		available_data_streams.push_back(createIndicationPayloadStream("tb_crc_fail", "uint8", "Transport Block CRC failure indicator (0=pass, 1=fail) [per-UE]"));
		available_data_streams.push_back(createIndicationPayloadStream("cb_errors", "uint32", "Code Block CRC error count [per-UE]"));
		available_data_streams.push_back(createIndicationPayloadStream("rsrp", "float32", "Reference Signal Received Power in dB [per-UE]"));
		available_data_streams.push_back(createIndicationPayloadStream("noise_var", "float32", "Noise+interference variance in dB; pre- or post-eq per enable_pusch_sinr (pre by default) [per-UE]"));
		available_data_streams.push_back(createIndicationPayloadStream("cb_count", "uint16", "Number of Code Blocks in transport block [per-UE]"));
		available_data_streams.push_back(createIndicationPayloadStream("rssi", "float32", "Received Signal Strength Indicator in dB [per-UE, per-group]"));
		available_data_streams.push_back(createIndicationPayloadStream("qam_mod_order", "uint8", "QAM modulation order [per-UE]"));
		available_data_streams.push_back(createIndicationPayloadStream("mcs_index", "uint8", "MCS index (range 0-31) [per-UE]"));
		available_data_streams.push_back(createIndicationPayloadStream("mcs_table_index", "uint8", "MCS table index [per-UE]"));
		available_data_streams.push_back(createIndicationPayloadStream("rb_start", "uint16", "Starting resource block [per-UE, per-group]"));
		available_data_streams.push_back(createIndicationPayloadStream("rb_size", "uint16", "Number of resource blocks [per-UE, per-group]"));
		available_data_streams.push_back(createIndicationPayloadStream("start_symbol_index", "uint8", "Start symbol index (range 0-13) [per-UE, per-group]"));
		available_data_streams.push_back(createIndicationPayloadStream("nr_of_symbols", "uint8", "PUSCH duration in symbols (range 1-14) [per-UE, per-group]"));
		available_data_streams.push_back(createIndicationPayloadStream("n_layers", "uint8", "Number of spatial layers [per-UE]"));
		available_data_streams.push_back(createIndicationPayloadStream("tb_size", "uint32", "Transport block size in bytes [per-UE]"));
		available_data_streams.push_back(createIndicationPayloadStream("pdu_len", "uint32", "PDU length in bytes (tb_size on CRC pass, 0 on fail) [per-UE]"));
		available_data_streams.push_back(createIndicationPayloadStream("target_code_rate", "uint16", "Target code rate (x10, e.g. 3080 = R=0.308) [per-UE]"));
		available_data_streams.push_back(createIndicationPayloadStream("new_data_indicator", "uint8", "New data indicator (0=retx, 1=new) [per-UE]"));
		available_data_streams.push_back(createIndicationPayloadStream("layer_offset", "uint16", "Start index in H matrix layer dimension within group [per-UE]"));
		available_data_streams.push_back(createIndicationPayloadStream("ue_grp_idx", "uint16", "cuPHY UE group index [per-UE]"));
		available_data_streams.push_back(createIndicationPayloadStream("h_offset", "uint32", "Element offset of group H data in SHM H-est row [per-UE, per-group]"));
		available_data_streams.push_back(createIndicationPayloadStream("h_size", "uint32", "Element count of group H blob in SHM H-est row [per-UE, per-group]"));
		available_data_streams.push_back(createIndicationPayloadStream("n_subcarriers", "uint16", "Number of subcarriers (PRBs * 12) [per-UE, per-group]"));
		available_data_streams.push_back(createIndicationPayloadStream("n_dmrs_estimates", "uint8", "Number of DMRS estimates [per-UE, per-group]"));
		available_data_streams.push_back(createIndicationPayloadStream("dmrs_symb_pos", "uint16", "DMRS symbol positions bitmap [per-UE, per-group]"));
		available_data_streams.push_back(createIndicationPayloadStream("sinr", "float32", "SINR in dB; pre- or post-eq per enable_pusch_sinr (pre by default) [per-UE]"));
		available_data_streams.push_back(createIndicationPayloadStream("timing_advance", "float32", "Timing advance in microseconds [per-UE]"));
		available_data_streams.push_back(createIndicationPayloadStream("harq_process_id", "uint8", "HARQ process ID [per-UE]"));
		available_data_streams.push_back(createIndicationPayloadStream("rv_index", "uint8", "Redundancy version index (0-3) [per-UE]"));
		available_data_streams.push_back(createIndicationPayloadStream("cfo_hz", "float32", "Carrier Frequency Offset in Hz (requires enable_pusch_cfo=1) [per-UE]"));

		// SRS SHM streams (cell-level)
		available_data_streams.push_back(createSharedMemoryStream(
			"srs_iq_samples",
			"array(int16)",
			"Raw SRS IQ samples per slot (cell-level SHM ping-pong buffer; underlying __half2 fp16 complex pairs stored as int16 bit-pattern)",
			maxSrsIqSamplesPerRow * sizeof(int16_t) * numRowsToInsertSrsIq,
			numRowsToInsertSrsIq
		));
		available_data_streams.push_back(createSharedMemoryStream(
			"srs_hest",
			"array(int16)",
			"SRS channel estimates per slot, all UEs concatenated (SHM ping-pong buffer, decode via srs_hest_offset/srs_hest_size per UE)",
			maxSrsHestBytesPerRow * numRowsToInsertSrsHest,
			numRowsToInsertSrsHest
		));
		available_data_streams.push_back(createSharedMemoryStream(
			"srs_rb_snr",
			"array(float32)",
			"SRS per-RB SNR per slot, all UEs concatenated (SHM ping-pong buffer, decode via srs_rb_snr_offset/srs_rb_snr_size per UE)",
			maxSrsRbSnrBytesPerRow * numRowsToInsertSrs,
			numRowsToInsertSrs
		));

		// SRS cell-level scalar streams
		available_data_streams.push_back(createIndicationPayloadStream("srs_cell_start_sym", "uint8", "SRS starting OFDM symbol index (cell-level)"));
		available_data_streams.push_back(createIndicationPayloadStream("srs_cell_n_srs_sym", "uint8", "Number of SRS OFDM symbols (cell-level)"));
		available_data_streams.push_back(createIndicationPayloadStream("n_srs_ue", "uint8", "Number of SRS UEs in this slot (cell-level)"));

		// SRS per-UE SHM decoder streams
		available_data_streams.push_back(createIndicationPayloadStream("srs_hest_n_prb_grps", "uint16", "Number of PRB groups in SRS H-estimate grid [per-UE]"));
		available_data_streams.push_back(createIndicationPayloadStream("srs_hest_offset", "uint32", "Byte offset of UE H-estimate blob in SHM SRS Hest row [per-UE]"));
		available_data_streams.push_back(createIndicationPayloadStream("srs_hest_size", "uint32", "Byte count of UE H-estimate blob in SHM SRS Hest row [per-UE]"));
		available_data_streams.push_back(createIndicationPayloadStream("srs_rb_snr_offset", "uint32", "Byte offset of UE RbSNR blob in SHM SRS RbSNR row [per-UE]"));
		available_data_streams.push_back(createIndicationPayloadStream("srs_rb_snr_size", "uint32", "Byte count of UE RbSNR blob in SHM SRS RbSNR row [per-UE]"));

		// SRS per-UE measurement streams (from cuphySrsReport_t)
		available_data_streams.push_back(createIndicationPayloadStream("srs_wideband_snr", "float32", "SRS wideband SNR in dB [per-UE]"));
		available_data_streams.push_back(createIndicationPayloadStream("srs_signal_energy", "float32", "SRS signal energy [per-UE]"));
		available_data_streams.push_back(createIndicationPayloadStream("srs_noise_energy", "float32", "SRS noise energy [per-UE]"));
		available_data_streams.push_back(createIndicationPayloadStream("srs_toa", "float32", "SRS time of arrival in microseconds [per-UE]"));
		available_data_streams.push_back(createIndicationPayloadStream("srs_hd_ant_flag", "uint8", "SRS half-duplex antenna flag [per-UE]"));
		available_data_streams.push_back(createIndicationPayloadStream("srs_sc_corr", "array(float32)", "SRS wideband spatial correlation [re, im] [per-UE]"));
		available_data_streams.push_back(createIndicationPayloadStream("srs_cs_corr_ratio_db", "float32", "SRS cross-section correlation ratio in dB [per-UE]"));

		// SRS per-UE config streams (from cuphyUeSrsPrm_t)
		available_data_streams.push_back(createIndicationPayloadStream("srs_ant_ports", "uint8", "SRS number of antenna ports [per-UE]"));
		available_data_streams.push_back(createIndicationPayloadStream("srs_n_syms", "uint8", "SRS number of symbols [per-UE]"));
		available_data_streams.push_back(createIndicationPayloadStream("srs_n_repetitions", "uint8", "SRS number of repetitions [per-UE]"));
		available_data_streams.push_back(createIndicationPayloadStream("srs_comb_size", "uint8", "SRS comb size [per-UE]"));
		available_data_streams.push_back(createIndicationPayloadStream("srs_comb_offset", "uint8", "SRS comb offset [per-UE]"));
		available_data_streams.push_back(createIndicationPayloadStream("srs_start_sym", "uint8", "SRS start symbol [per-UE]"));
		available_data_streams.push_back(createIndicationPayloadStream("srs_cyclic_shift", "uint8", "SRS cyclic shift [per-UE]"));
		available_data_streams.push_back(createIndicationPayloadStream("srs_freq_position", "uint8", "SRS frequency position [per-UE]"));
		available_data_streams.push_back(createIndicationPayloadStream("srs_freq_shift", "uint16", "SRS frequency shift [per-UE]"));
		available_data_streams.push_back(createIndicationPayloadStream("srs_freq_hopping", "uint8", "SRS frequency hopping [per-UE]"));
		available_data_streams.push_back(createIndicationPayloadStream("srs_resource_type", "uint8", "SRS resource type [per-UE]"));
		available_data_streams.push_back(createIndicationPayloadStream("srs_t_srs", "uint16", "SRS periodicity T_SRS [per-UE]"));
		available_data_streams.push_back(createIndicationPayloadStream("srs_t_offset", "uint16", "SRS slot offset T_offset [per-UE]"));
		available_data_streams.push_back(createIndicationPayloadStream("srs_usage", "uint32", "SRS usage bitmask [per-UE]"));
		available_data_streams.push_back(createIndicationPayloadStream("srs_n_valid_prg", "uint16", "Number of valid PRB groups in SRS [per-UE]"));
		available_data_streams.push_back(createIndicationPayloadStream("srs_prg_size", "uint16", "SRS PRB group size [per-UE]"));

		// E3-RanFunctionDefinition: telemetry IDs 1..STREAM_TYPE_COUNT
		json telemetry_id_list = json::array();
		for (uint32_t id = 1; id <= e3::STREAM_TYPE_COUNT; ++id) {
			telemetry_id_list.push_back(id);
		}

		json ran_function;
		ran_function["ranFunctionIdentifier"] = e3::RAN_FUNCTION_ID_NVIDIA_KPM;
		ran_function["telemetryIdentifierList"] = telemetry_id_list;
		ran_function["controlIdentifierList"] = json::array();
		ran_function["ranFunctionData"] = available_data_streams;

		e3_setup_response["ranFunctionList"] = json::array({ran_function});
		
		response_json = e3_setup_response;
		
		NVLOGC_FMT(TAG_E3, "E3 Setup successful for dApp '{}' assigned ID: {}", dapp_name, dapp_id);
		
	} catch (const json::exception& e) {
		NVLOGC_FMT(TAG_E3, "Error processing E3 Setup Request: {}", e.what());
		json error_resp;
		error_resp["type"] = "setupResponse";
		error_resp["id"] = generateMessageId();
		error_resp["requestId"] = request_id;
		error_resp["responseCode"] = "negative";
		error_resp["message"] = "invalid setup request format";
		response = error_resp.dump();
		return;
	}
	
	response = response_json.dump();
}

void E3Agent::handleSubscriptionRequest(const json& request, std::string& response)
{
    json response_json;
    json e3_sub_response;
    uint32_t dapp_id = 0;
    uint32_t request_id = 0;

    try {
        const json& e3_sub_req = request;
        dapp_id = e3_sub_req.at("dAppIdentifier").get<uint32_t>();
        request_id = e3_sub_req.value("id", 0u);

        if (!updateDappActivity(dapp_id)) {
            NVLOGC_FMT(TAG_E3, "Subscription rejected for non-connected dApp {}", dapp_id);
            e3_sub_response["responseCode"] = "negative";
            e3_sub_response["message"] = "dApp not connected or timed out";
        } else {
            uint32_t ran_func_id = e3_sub_req.at("ranFunctionIdentifier").get<uint32_t>();
            if (ran_func_id != e3::RAN_FUNCTION_ID_NVIDIA_KPM) {
                NVLOGC_FMT(TAG_E3, "Subscription rejected: unsupported ranFunctionIdentifier {} (expected {})",
                           ran_func_id, e3::RAN_FUNCTION_ID_NVIDIA_KPM);
                e3_sub_response["responseCode"] = "negative";
                e3_sub_response["message"] = "unsupported ranFunctionIdentifier";
            } else {
                auto telemetry_ids = e3_sub_req.value("telemetryIdentifierList", std::vector<uint32_t>{});
                auto control_ids = e3_sub_req.value("controlIdentifierList", std::vector<uint32_t>{});
                uint32_t periodicity_us = e3_sub_req.value("periodicity", 100000u);
                uint32_t subscription_time_s = e3_sub_req.value("subscriptionTime", 0u);

                e3::StreamType stream_bitfield = e3::StreamType::NONE;
                bool valid = true;

                // NVIDIA KPM: telemetry-only, no control dispatch yet.
                // Relax to (telemetry_ids.empty() && control_ids.empty()) when controls are implemented.
                if (telemetry_ids.empty()) {
                    NVLOGC_FMT(TAG_E3, "Subscription rejected: empty telemetryIdentifierList");
                    e3_sub_response["responseCode"] = "negative";
                    e3_sub_response["message"] = "telemetryIdentifierList must not be empty";
                    valid = false;
                }

                // Validate all telemetry IDs
                for (uint32_t tid : telemetry_ids) {
                    e3::StreamType st = e3::telemetryIdToStreamType(tid);
                    if (st == e3::StreamType::NONE) {
                        NVLOGC_FMT(TAG_E3, "Subscription rejected: invalid telemetry ID {}", tid);
                        e3_sub_response["responseCode"] = "negative";
                        e3_sub_response["message"] = "invalid telemetry identifier";
                        valid = false;
                        break;
                    }
                    stream_bitfield |= st;
                }

                if (valid) {
                    uint32_t sub_id = generateSubscriptionId();
                    auto now = std::chrono::steady_clock::now();
                    auto expiry = (subscription_time_s > 0)
                        ? now + std::chrono::seconds(subscription_time_s)
                        : std::chrono::steady_clock::time_point::max();

                    {
                        std::lock_guard<std::mutex> lock(e3_subscriptions_mutex);
                        e3_subscriptions[sub_id] = {sub_id, dapp_id, ran_func_id, telemetry_ids, stream_bitfield, periodicity_us, now, now, expiry};
                    }

                    std::string ids_str = fmt::format("[{}]", fmt::join(telemetry_ids, ","));
                    NVLOGC_FMT(TAG_E3, "E3 Subscription {} created for dApp {} (ranFunction={}, telemetryIds={})",
                               sub_id, dapp_id, ran_func_id, ids_str);

                    e3_sub_response["responseCode"] = "positive";
                    e3_sub_response["subscriptionId"] = sub_id;
                    e3_sub_response["ranFunctionIdentifier"] = ran_func_id;
                    e3_sub_response["telemetryGrantedList"] = telemetry_ids;
                    e3_sub_response["controlGrantedList"] = json::array();
                    e3_sub_response["periodicity"] = periodicity_us;
                }
            }
        }

        e3_sub_response["type"] = "subscriptionResponse";
        e3_sub_response["id"] = generateMessageId();
        e3_sub_response["requestId"] = request_id;
        e3_sub_response["dAppIdentifier"] = dapp_id;
        response_json = e3_sub_response;

    } catch (const json::exception& e) {
        NVLOGC_FMT(TAG_E3, "Error processing E3 Subscription Request: {}", e.what());
        json e3_err_resp;
        e3_err_resp["type"] = "subscriptionResponse";
        e3_err_resp["id"] = generateMessageId();
        e3_err_resp["requestId"] = request_id;
        e3_err_resp["responseCode"] = "negative";
        e3_err_resp["message"] = "missing or invalid parameters in subscription request";
        e3_err_resp["dAppIdentifier"] = dapp_id;
        response_json = e3_err_resp;
    }
    response = response_json.dump();
}

void E3Agent::handleSubscriptionDelete(const json& request, std::string& response)
{
    json response_json;
    json e3_unsub_response;
    uint32_t dapp_id = 0;
    uint32_t sub_id = 0;
    uint32_t request_id = 0;

    try {
        const json& e3_unsub_req = request;
        dapp_id = e3_unsub_req.at("dAppIdentifier").get<uint32_t>();
        sub_id = e3_unsub_req.at("subscriptionId").get<uint32_t>();
        request_id = e3_unsub_req.value("id", 0u);

        bool found = false;
        {
            std::lock_guard<std::mutex> lock(e3_subscriptions_mutex);
            auto it = e3_subscriptions.find(sub_id);
            if (it != e3_subscriptions.end() && it->second.dapp_id == dapp_id) {
                e3_subscriptions.erase(it);
                found = true;
            }
        }

        updateDappActivity(dapp_id);

        if (found) {
            NVLOGC_FMT(TAG_E3, "E3 Subscription Delete successful for subscription {}", sub_id);
            e3_unsub_response["responseCode"] = "positive";
        } else {
            NVLOGC_FMT(TAG_E3, "E3 Subscription Delete failed for sub_id {}, dApp_id {}", sub_id, dapp_id);
            e3_unsub_response["responseCode"] = "negative";
            e3_unsub_response["message"] = "subscription not found or dApp ID mismatch";
        }

        e3_unsub_response["subscriptionId"] = sub_id;
        e3_unsub_response["type"] = "subscriptionResponse";
        e3_unsub_response["id"] = generateMessageId();
        e3_unsub_response["requestId"] = request_id;
        e3_unsub_response["dAppIdentifier"] = dapp_id;
        response_json = e3_unsub_response;

    } catch (const json::exception& e) {
        NVLOGC_FMT(TAG_E3, "Error processing E3 Subscription Delete Request: {}", e.what());
        json e3_err_resp;
        e3_err_resp["type"] = "subscriptionResponse";
        e3_err_resp["id"] = generateMessageId();
        e3_err_resp["requestId"] = request_id;
        e3_err_resp["responseCode"] = "negative";
        e3_err_resp["message"] = "missing or invalid parameters in subscription delete request";
        e3_err_resp["dAppIdentifier"] = dapp_id;
        e3_err_resp["subscriptionId"] = sub_id;
        response_json = e3_err_resp;
    }
    response = response_json.dump();
}


void E3Agent::handleControlMessage(const json& request, std::string& response)
{
    json ack;
    uint32_t dapp_id = 0;
    uint32_t request_id = 0;

    try {
        dapp_id = request.at("dAppIdentifier").get<uint32_t>();
        request_id = request.value("id", 0u);

        updateDappActivity(dapp_id);

        // Control actions not implemented
        ack["responseCode"] = "negative";
        ack["message"] = "control actions not implemented";

    } catch (const json::exception& e) {
        ack["responseCode"] = "negative";
        ack["message"] = "invalid control message format";
        NVLOGC_FMT(TAG_E3, "Error processing E3 Control Message: {}. Request: {}", e.what(), request.dump());
    }

    ack["type"] = "messageAck";
    ack["id"] = generateMessageId();
    ack["requestId"] = request_id;
    ack["dAppIdentifier"] = dapp_id;

    response = ack.dump();
}
