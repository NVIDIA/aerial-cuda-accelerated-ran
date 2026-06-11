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

#include "peer.hpp"

#include "dpdk.hpp"
#include "flow.hpp"
#include "fronthaul.hpp"
#include "nic.hpp"
#include "pdump_client.hpp"
#include "time.hpp"
#include "utils.hpp"
#include "memreg.hpp"
#include "gpu_comm.hpp"
#include "aerial-fh-driver/pcap_logger.hpp"

// TODO FIXME remove when DOCA warnings are fixed
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

#define TAG "FH.PEER"

namespace aerial_fh
{
Peer::Peer(Nic* nic, PeerInfo const* info,std::vector<uint16_t>& eAxC_list_ul,std::vector<uint16_t>& eAxC_list_srs,std::vector<uint16_t>& eAxC_list_dl) :
    nic_{nic},
    metrics_{this, info->id},
    info_{*info},
    cpu_regular_size(0),
    cpu_pinned_size(0),
    gpu_regular_size(0),
    gpu_pinned_size(0)
{
    NVLOGI_FMT(TAG, "Adding Peer {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x} with id {} to NIC {}",
        info_.dst_mac_addr.bytes[0],info_.dst_mac_addr.bytes[1],info_.dst_mac_addr.bytes[2],
        info_.dst_mac_addr.bytes[3],info_.dst_mac_addr.bytes[4],info_.dst_mac_addr.bytes[5],
        info_.id, nic_->get_name());

    validate_input();

    prb_size_upl_     = get_prb_size(info_.ud_comp_info.iq_sample_size, info_.ud_comp_info.method);
    prbs_per_pkt_upl_ = (nic->get_mtu() - ORAN_IQ_HDR_SZ) / prb_size_upl_;
    // Set these two fields to 0 for modulation compression, info_.ud_comp_info.method == MODULATION_COMPRESSION
    if(info_.ud_comp_info.method == UserDataCompressionMethod::MODULATION_COMPRESSION)
    {
        prb_size_upl_     = 0;
        prbs_per_pkt_upl_ = 0;
    }
    adjust_src_mac_addr();

    rxq_ = nullptr;
    if(get_fronthaul()->rmax_enabled() && info_.rx_mode != RxApiMode::TXONLY)
        THROW_FH(EINVAL, StringBuilder() << "With Rivermax enabled only RxApiMode::TXONLY is allowed.");

    request_nic_resources();

    if((!(get_fronthaul()->get_info().cuda_device_ids.empty())) && (info_.rx_mode != RxApiMode::UEMODE && info_.rx_mode != RxApiMode::TXONLY))
        doca_gpu_sem_create();

    create_rx_rules(eAxC_list_ul,eAxC_list_srs,eAxC_list_dl);
    gpu_comm_create_up_slot_list();
    create_cplane_sections_cache();
}

Peer::~Peer()
{
    NVLOGI_FMT(TAG, "Removing Peer {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x} with id {} from NIC {}",
        info_.dst_mac_addr.bytes[0],info_.dst_mac_addr.bytes[1],info_.dst_mac_addr.bytes[2],
        info_.dst_mac_addr.bytes[3],info_.dst_mac_addr.bytes[4],info_.dst_mac_addr.bytes[5],
        info_.id, nic_->get_name());
    free_nic_resources();
    //cudaFreeHost(up_symbol_info_gpu_); //FIXME error checking?
    free(cplane_sections_info_list_);
    cudaFreeHost(h_up_slot_info_);
    cudaFree(d_up_slot_info_);
    cudaFree(d_flow_pkt_hdr_index_);
    cudaFree(d_flow_sym_pkt_hdr_index_);
    cudaFree(d_block_count_);
    cudaFree(d_ecpri_seq_id_);
    cudaFree(d_hdr_template_);
    if(info_.ud_comp_info.method == aerial_fh::UserDataCompressionMethod::MODULATION_COMPRESSION)
    {
        for(int up_slot_idx = 0; up_slot_idx < kPeerSlotsInfo; ++up_slot_idx)
        {
            auto& message_info = partial_up_slot_info_[up_slot_idx].message_info;
            for(int msg_idx = 0; msg_idx < kPeerSymbolsInfo; ++msg_idx)
            {
                ASSERT_CUDA_FH(cudaFreeHost(message_info[msg_idx].mod_comp_params));
            }
        }
    }
    ASSERT_CUDA_FH(cudaFreeHost(partial_up_slot_info_));
    ASSERT_CUDA_FH(cudaFreeHost(h_flow_hdr_size_info_));
}

void Peer::update_max_num_prbs_per_symbol(uint16_t max_num_prbs_per_symbol)
{
    info_.max_num_prbs_per_symbol = max_num_prbs_per_symbol;
}

void Peer::update(UserDataCompressionMethod dl_comp_meth, uint8_t dl_bit_width)
{
    info_.ud_comp_info.method         = dl_comp_meth;
    info_.ud_comp_info.iq_sample_size = dl_bit_width;
    prb_size_upl_                     = get_prb_size(info_.ud_comp_info.iq_sample_size, info_.ud_comp_info.method);
    prbs_per_pkt_upl_                 = (nic_->get_mtu() - ORAN_IQ_HDR_SZ) / prb_size_upl_;
    // Set these two fields to 0 for modulation compression, info_.ud_comp_info.method == MODULATION_COMPRESSION
    if(info_.ud_comp_info.method == UserDataCompressionMethod::MODULATION_COMPRESSION)
    {
        prb_size_upl_     = 0;
        prbs_per_pkt_upl_ = 0;
    }
}

void Peer::update(MacAddr dst_mac_addr,std::vector<uint16_t>& eAxC_list_ul,std::vector<uint16_t>& eAxC_list_srs)
{
    NVLOGI_FMT(TAG, "Updating Peer {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x} dst MAC address to {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}",
        info_.dst_mac_addr.bytes[0],info_.dst_mac_addr.bytes[1],info_.dst_mac_addr.bytes[2],
        info_.dst_mac_addr.bytes[3],info_.dst_mac_addr.bytes[4],info_.dst_mac_addr.bytes[5],
        dst_mac_addr.bytes[0],dst_mac_addr.bytes[1],dst_mac_addr.bytes[2],dst_mac_addr.bytes[3],
        dst_mac_addr.bytes[4],dst_mac_addr.bytes[5]);
    info_.dst_mac_addr = dst_mac_addr;
    rx_flow_rules_.clear();
    std::vector<uint16_t> eAxC_list_dl_dummy; //only used for UE Mode
    create_rx_rules(eAxC_list_ul,eAxC_list_srs,eAxC_list_dl_dummy);
}

void Peer::update(MacAddr dst_mac_addr, uint16_t vlan_tci,std::vector<uint16_t>& eAxC_list_ul,std::vector<uint16_t>& eAxC_list_srs)
{
    NVLOGI_FMT(TAG, "Updating Peer {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x} dst MAC address to {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x} and vlan tci to {}",
        info_.dst_mac_addr.bytes[0],info_.dst_mac_addr.bytes[1],info_.dst_mac_addr.bytes[2],
        info_.dst_mac_addr.bytes[3],info_.dst_mac_addr.bytes[4],info_.dst_mac_addr.bytes[5],
        dst_mac_addr.bytes[0],dst_mac_addr.bytes[1],dst_mac_addr.bytes[2],dst_mac_addr.bytes[3],
        dst_mac_addr.bytes[4],dst_mac_addr.bytes[5],vlan_tci);

    info_.dst_mac_addr = dst_mac_addr;
    info_.vlan.tci = vlan_tci;
    rx_flow_rules_.clear();
    std::vector<uint16_t> eAxC_list_dl_dummy; //only used for UE Mode
    create_rx_rules(eAxC_list_ul,eAxC_list_srs,eAxC_list_dl_dummy);
}

void Peer::validate_input()
{
    if(info_.txq_count_uplane == 0 && info_.txq_count_uplane_gpu == 0)
    {
        THROW_FH(EINVAL, StringBuilder() << "No U-plane NIC TXQs were specified for Peer " << info_.dst_mac_addr);
    }
}

void Peer::request_nic_resources()
{
    auto& queue_manager = nic_->get_queue_manager();
    auto  nic_name      = nic_->get_name();

    try
    {
        NVLOGI_FMT(TAG,"Requesting {} TXQ for CPU U-plane",info_.txq_count_uplane);
        NVLOGI_FMT(TAG,"Requesting {} TXQ for GPU U-plane",info_.txq_count_uplane_gpu);
        if(info_.share_txqs)
        {
            NVLOGI_FMT(TAG,"Peer {} Requesting {} Shared TXQ for CPU U-plane", info_.id, info_.txq_count_uplane);
            queue_manager.assign_shared_txqs(txqs_uplane_, info_.txq_count_uplane, false);
            for(size_t txq_idx = 0; txq_idx < info_.txq_count_uplane; txq_idx++)
            {
                NVLOGI_FMT(TAG, "Peer SRC: {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x} DST: {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x} is using NIC {} TXQ #{} for U-plane",
                    info_.src_mac_addr.bytes[0],info_.src_mac_addr.bytes[1],info_.src_mac_addr.bytes[2],
                    info_.src_mac_addr.bytes[3],info_.src_mac_addr.bytes[4],info_.src_mac_addr.bytes[5],
                    info_.dst_mac_addr.bytes[0],info_.dst_mac_addr.bytes[1],info_.dst_mac_addr.bytes[2],
                    info_.dst_mac_addr.bytes[3],info_.dst_mac_addr.bytes[4],info_.dst_mac_addr.bytes[5],
                    nic_name, txqs_uplane_.items()[txq_idx]->get_id());
            }
        }
        else
        {
            for(size_t txq_idx = 0; txq_idx < info_.txq_count_uplane; txq_idx++)
            {
                auto txq = queue_manager.assign_txq(false);
                txqs_uplane_.add(txq);
                NVLOGI_FMT(TAG, "Peer {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x} is using NIC {} TXQ #{} for U-plane",
                    info_.dst_mac_addr.bytes[0],info_.dst_mac_addr.bytes[1],info_.dst_mac_addr.bytes[2],
                    info_.dst_mac_addr.bytes[3],info_.dst_mac_addr.bytes[4],info_.dst_mac_addr.bytes[5],
                    nic_name, txq->get_id());
            }

            for(size_t txq_idx = 0; txq_idx < info_.txq_count_uplane_gpu; txq_idx++)
            {
                auto txq = queue_manager.assign_txq(true);
                txqs_uplane_gpu_.add(txq);
                NVLOGI_FMT(TAG, "Peer {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x} is using NIC {} GPU TXQ #{} for U-plane",
                    info_.dst_mac_addr.bytes[0],info_.dst_mac_addr.bytes[1],info_.dst_mac_addr.bytes[2],
                    info_.dst_mac_addr.bytes[3],info_.dst_mac_addr.bytes[4],info_.dst_mac_addr.bytes[5],
                    nic_name, txq->get_id());
            }
            if(info_.txq_cplane)
            {
                txq_dl_cplane_ = queue_manager.assign_txq(false);
                NVLOGI_FMT(TAG, "Peer {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x} is using NIC {} TXQ #{} for DL C-plane",
                    info_.dst_mac_addr.bytes[0],info_.dst_mac_addr.bytes[1],info_.dst_mac_addr.bytes[2],
                    info_.dst_mac_addr.bytes[3],info_.dst_mac_addr.bytes[4],info_.dst_mac_addr.bytes[5],
                    nic_name, txq_dl_cplane_->get_id());
                txq_ul_cplane_ = queue_manager.assign_txq(false);
                NVLOGI_FMT(TAG, "Peer {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x} is using NIC {} TXQ #{} for UL C-plane",
                    info_.dst_mac_addr.bytes[0],info_.dst_mac_addr.bytes[1],info_.dst_mac_addr.bytes[2],
                    info_.dst_mac_addr.bytes[3],info_.dst_mac_addr.bytes[4],info_.dst_mac_addr.bytes[5],
                    nic_name, txq_ul_cplane_->get_id());
                if(info_.txq_bfw_cplane)
                {
                    // When YAML instructs to alloc & transmit DLC BFW in a separate queue or if ULC BFW C-Plane is to be spread across the window,
                    // allocate a separate queue.
                    if((info_.bfw_cplane_info.dlc_bfw_enable_divide_per_cell) || (info_.bfw_cplane_info.dlc_alloc_cplane_bfw_txq))
                    {
                        txq_dl_bfw_cplane_ = queue_manager.assign_txq(false);
                        NVLOGI_FMT(TAG, "Peer {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x} is using NIC {} TXQ #{} for DL BFW C-plane",
                        info_.dst_mac_addr.bytes[0],info_.dst_mac_addr.bytes[1],info_.dst_mac_addr.bytes[2],
                            info_.dst_mac_addr.bytes[3],info_.dst_mac_addr.bytes[4],info_.dst_mac_addr.bytes[5],
                            nic_name, txq_dl_bfw_cplane_->get_id());
                    }
                    else
                    {
                        txq_dl_bfw_cplane_ = txq_dl_cplane_;
                    }

                    // When YAML instructs to alloc & transmit ULC BFW in a separate queue or if ULC BFW C-Plane is to be spread across the window,
                    // allocate a separate queue.
                    if((info_.bfw_cplane_info.ulc_bfw_enable_divide_per_cell) || (info_.bfw_cplane_info.ulc_alloc_cplane_bfw_txq))
                    {
                        txq_ul_bfw_cplane_ = queue_manager.assign_txq(false);
                        NVLOGI_FMT(TAG, "Peer {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x} is using NIC {} TXQ #{} for UL BFW C-plane",
                            info_.dst_mac_addr.bytes[0],info_.dst_mac_addr.bytes[1],info_.dst_mac_addr.bytes[2],
                            info_.dst_mac_addr.bytes[3],info_.dst_mac_addr.bytes[4],info_.dst_mac_addr.bytes[5],
                            nic_name, txq_ul_bfw_cplane_->get_id());
                    }
                    else
                    {
                        txq_ul_bfw_cplane_ = txq_ul_cplane_;
                    }
                }
            }
        }
        if(info_.rx_mode == RxApiMode::PEER)
        {
            rxq_ = queue_manager.assign_rxq();
            NVLOGD_FMT(TAG, "Peer {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x} is using NIC {} RXQ #{}",
                info_.dst_mac_addr.bytes[0],info_.dst_mac_addr.bytes[1],info_.dst_mac_addr.bytes[2],
                info_.dst_mac_addr.bytes[3],info_.dst_mac_addr.bytes[4],info_.dst_mac_addr.bytes[5],
                nic_name, rxq_->get_id());
            if(info_.enable_srs)
            {
                if(!(get_fronthaul()->get_info().cuda_device_ids.empty()))
                {
                    rxqSrs_ = queue_manager.assign_rxq();
                    NVLOGD_FMT(TAG, "Peer {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x} is using NIC {} RXQ #{} for SRS",
                        info_.dst_mac_addr.bytes[0],info_.dst_mac_addr.bytes[1],info_.dst_mac_addr.bytes[2],
                        info_.dst_mac_addr.bytes[3],info_.dst_mac_addr.bytes[4],info_.dst_mac_addr.bytes[5],
                        nic_name, rxqSrs_->get_id());
                }
            }
        }
        else if(info_.rx_mode == RxApiMode::HYBRID)
        {
            rxq_ = queue_manager.assign_rxq();
            NVLOGD_FMT(TAG, "Peer {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x} is using NIC {} RXQ #{}",
                info_.dst_mac_addr.bytes[0],info_.dst_mac_addr.bytes[1],info_.dst_mac_addr.bytes[2],
                info_.dst_mac_addr.bytes[3],info_.dst_mac_addr.bytes[4],info_.dst_mac_addr.bytes[5],
                nic_name, rxq_->get_id());
            if(info_.enable_srs)
            {
                rxqSrs_ = queue_manager.assign_rxq();
                NVLOGD_FMT(TAG, "Peer {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x} is using NIC {} RXQ #{} for SRS",
                info_.dst_mac_addr.bytes[0],info_.dst_mac_addr.bytes[1],info_.dst_mac_addr.bytes[2],
                    info_.dst_mac_addr.bytes[3],info_.dst_mac_addr.bytes[4],info_.dst_mac_addr.bytes[5],
                    nic_name, rxqSrs_->get_id());
            }
        }
        else if(info_.rx_mode == RxApiMode::UEMODE)
        {
            // CPU RXQs
            rxq_ = queue_manager.assign_rxq();
            NVLOGD_FMT(TAG, "Peer {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x} is using NIC {} RXQ #{}",
                info_.dst_mac_addr.bytes[0],info_.dst_mac_addr.bytes[1],info_.dst_mac_addr.bytes[2],
                info_.dst_mac_addr.bytes[3],info_.dst_mac_addr.bytes[4],info_.dst_mac_addr.bytes[5],
                nic_name, rxq_->get_id());
        }
    }
    catch(...)
    {
        free_nic_resources();
        throw;
    }
}

void Peer::free_nic_resources()
{
    auto& queue_manager = nic_->get_queue_manager();
    auto  nic_name      = nic_->get_name();

    if(!info_.share_txqs)
    {

        for(auto txq : txqs_uplane_.items())
        {
            NVLOGD_FMT(TAG, "Peer {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x} is returning U-plane TXQ #{} to NIC {}",
                    info_.dst_mac_addr.bytes[0],info_.dst_mac_addr.bytes[1],info_.dst_mac_addr.bytes[2],
                    info_.dst_mac_addr.bytes[3],info_.dst_mac_addr.bytes[4],info_.dst_mac_addr.bytes[5],
                    txq->get_id(), nic_name);
            queue_manager.reclaim(txq);
        }

        for(auto txq : txqs_uplane_gpu_.items())
        {
            NVLOGD_FMT(TAG, "Peer {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x} is returning U-plane TXQ GPU #{} to NIC {}",
                info_.dst_mac_addr.bytes[0],info_.dst_mac_addr.bytes[1],info_.dst_mac_addr.bytes[2],
                info_.dst_mac_addr.bytes[3],info_.dst_mac_addr.bytes[4],info_.dst_mac_addr.bytes[5],
                txq->get_id(), nic_name);
            queue_manager.reclaim(txq);
        }

        if(txq_dl_cplane_)
        {
            NVLOGD_FMT(TAG, "Peer {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x} is returning DL C-plane TXQ #{} to NIC {}",
                info_.dst_mac_addr.bytes[0],info_.dst_mac_addr.bytes[1],info_.dst_mac_addr.bytes[2],
                info_.dst_mac_addr.bytes[3],info_.dst_mac_addr.bytes[4],info_.dst_mac_addr.bytes[5],
                txq_dl_cplane_->get_id(), nic_name);
            queue_manager.reclaim(txq_dl_cplane_);
        }

        if(txq_ul_cplane_)
        {
            NVLOGD_FMT(TAG, "Peer {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x} is returning UL C-plane TXQ #{} to NIC {}",
                info_.dst_mac_addr.bytes[0],info_.dst_mac_addr.bytes[1],info_.dst_mac_addr.bytes[2],
                info_.dst_mac_addr.bytes[3],info_.dst_mac_addr.bytes[4],info_.dst_mac_addr.bytes[5],
                txq_ul_cplane_->get_id(), nic_name);
            queue_manager.reclaim(txq_ul_cplane_);
        }
    }

    if(rxq_)
    {
        NVLOGD_FMT(TAG, "Peer {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x} is returning RXQ #{} to NIC {}" ,
            info_.dst_mac_addr.bytes[0],info_.dst_mac_addr.bytes[1],info_.dst_mac_addr.bytes[2],
            info_.dst_mac_addr.bytes[3],info_.dst_mac_addr.bytes[4],info_.dst_mac_addr.bytes[5],
            rxq_->get_id(), nic_name);
        queue_manager.reclaim(rxq_);
    }
    if(rxqSrs_)
    {
        NVLOGD_FMT(TAG, "Peer {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x} is returning RXQ #{} to NIC {}" ,
            info_.dst_mac_addr.bytes[0],info_.dst_mac_addr.bytes[1],info_.dst_mac_addr.bytes[2],
            info_.dst_mac_addr.bytes[3],info_.dst_mac_addr.bytes[4],info_.dst_mac_addr.bytes[5],
            rxqSrs_->get_id(), nic_name);
        queue_manager.reclaim(rxqSrs_);
    }
}

void Peer::create_cplane_sections_cache()
{
    cplane_sections_info_list_     = (slot_command_api::cplane_sections_info_t*)malloc(kPrbSplitInfo * sizeof(slot_command_api::cplane_sections_info_t));
    cplane_sections_info_list_cnt_ = 0;
}

void Peer::gpu_comm_create_up_slot_list()
{
    int ret = 0;

    if(nic_->get_info().cuda_device < 0)
        return;
    Gpu* gpu_ = nic_->get_fronthaul()->gpus()[nic_->get_info().cuda_device].get();

    size_t gpu_mem_size = 0;

    // Allocate u-plane slot info array on the GPU
    ASSERT_CUDA_FH(cudaMalloc((void**)&d_up_slot_info_, sizeof(UplaneSlotInfo_t) * kPeerSlotsInfo));
    ASSERT_CUDA_FH(cudaMemset(d_up_slot_info_, 0, sizeof(UplaneSlotInfo_t) * kPeerSlotsInfo));
    ASSERT_CUDA_FH(cudaMallocHost((void**)&h_up_slot_info_, sizeof(UplaneSlotInfoHost_t) * kPeerSlotsInfo));
    gpu_mem_size += sizeof(UplaneSlotInfo_t) * kPeerSlotsInfo;

   // Allocate large buffer of kPeerSlotsInof * MAX_DL_EAXCIDS elements initially memset to 0 to keep track of gpu_index per unique flow
    ASSERT_CUDA_FH(cudaMalloc((void**)&d_flow_pkt_hdr_index_, sizeof(uint32_t) * kPeerSlotsInfo * MAX_DL_EAXCIDS));
    ASSERT_CUDA_FH(cudaMemset(d_flow_pkt_hdr_index_, 0, sizeof(uint32_t) * kPeerSlotsInfo * MAX_DL_EAXCIDS));
    gpu_mem_size += sizeof(uint32_t) * kPeerSlotsInfo * MAX_DL_EAXCIDS;

    ASSERT_CUDA_FH(cudaMalloc((void**)&d_flow_sym_pkt_hdr_index_, sizeof(uint32_t) * 14 * MAX_DL_EAXCIDS));
    ASSERT_CUDA_FH(cudaMemset(d_flow_sym_pkt_hdr_index_, 0, sizeof(uint32_t) * 14 * MAX_DL_EAXCIDS));
    gpu_mem_size += sizeof(uint32_t) * 14 * MAX_DL_EAXCIDS;

    ASSERT_CUDA_FH(cudaMalloc((void**)&d_block_count_, sizeof(uint32_t)));
    ASSERT_CUDA_FH(cudaMemset(d_block_count_, 0, sizeof(uint32_t)));
    gpu_mem_size += sizeof(uint32_t);


    ASSERT_CUDA_FH(cudaMallocHost((void**)&h_flow_hdr_size_info_, sizeof(FlowPtrInfo) * kPeerSlotsInfo * MAX_DL_EAXCIDS));
    cpu_pinned_size += sizeof(FlowPtrInfo) * kPeerSlotsInfo * MAX_DL_EAXCIDS;
    ASSERT_CUDA_FH(cudaMalloc((void**)&d_ecpri_seq_id_, sizeof(uint32_t) * MAX_DL_EAXCIDS));
    gpu_mem_size += sizeof(uint32_t) * MAX_DL_EAXCIDS;
    ASSERT_CUDA_FH(cudaMemset(d_ecpri_seq_id_, 0, sizeof(uint32_t) * MAX_DL_EAXCIDS));
    ASSERT_CUDA_FH(cudaMalloc((void**)&d_hdr_template_, sizeof(uint32_t) * 8 * kPeerSlotsInfo * MAX_DL_EAXCIDS));
    gpu_mem_size += sizeof(uint32_t) * 8 * kPeerSlotsInfo * MAX_DL_EAXCIDS;
    ASSERT_CUDA_FH(cudaMallocHost((void**)&partial_up_slot_info_, sizeof(PartialUplaneSlotInfo_t) * kPeerSlotsInfo));
    cpu_pinned_size += sizeof(PartialUplaneSlotInfo_t) * kPeerSlotsInfo;

    if(info_.ud_comp_info.method == aerial_fh::UserDataCompressionMethod::MODULATION_COMPRESSION)
    {
        for(int up_slot_idx = 0; up_slot_idx < kPeerSlotsInfo; ++up_slot_idx)
        {
            auto& message_info = partial_up_slot_info_[up_slot_idx].message_info;
            for(int msg_idx = 0; msg_idx < kPeerSymbolsInfo; ++msg_idx)
            {
                ASSERT_CUDA_FH(cudaMallocHost((void**)&(message_info[msg_idx].mod_comp_params), sizeof(ModCompPartialSectionInfoPerMessagePerSymbol_t)));
                cpu_pinned_size += sizeof(ModCompPartialSectionInfoPerMessagePerSymbol_t);
            }
        }
    }
    up_slot_info_cnt_ = 0;

    up_tx_request_ = (TxRequestUplaneGpuComm*) allocate_memory(sizeof(TxRequestUplaneGpuComm) * kPeerSlotsInfo, pageSizeAlign);

    gpu_regular_size += gpu_mem_size;
    memfoot_add_gpu_size(MF_TAG_FH_PEER, gpu_mem_size);
}

size_t Peer::getCpuRegularSize() const {
    return cpu_regular_size;
}

size_t Peer::getCpuPinnedSize() const {
    return cpu_pinned_size;
}

size_t Peer::getGpuPinnedSize() const {
    return gpu_pinned_size;
}

size_t Peer::getGpuRegularSize() const {
    return gpu_regular_size;
}

uint32_t Peer::get_next_cplane_sections_info_list_idx()
{

    uint32_t cnt = cplane_sections_info_list_cnt_.load();
    uint32_t next;
    do {
        next = (cnt + 1) % kPrbSplitInfo;
    } while (!cplane_sections_info_list_cnt_.compare_exchange_weak(cnt, next));

    return cnt;
}

uint32_t Peer::gpu_comm_get_next_up_slot_idx()
{
    uint32_t cnt = 0;
    const std::lock_guard<aerial_fh::FHMutex> lock(mtx_);

    cnt = up_slot_info_cnt_;
    up_slot_info_cnt_ = (up_slot_info_cnt_+1) % kPeerSlotsInfo;

    return cnt;
}

void Peer::create_rx_rules(const std::vector<uint16_t>& eAxC_list_ul, const std::vector<uint16_t>& eAxC_list_srs, const std::vector<uint16_t>& eAxC_list_dl)
{
    if(info_.rx_mode == RxApiMode::PEER)
    {
        if(nic_->pdump_enabled())
        {
            create_rx_rule_with_cpu_mirroring();
        }
        else
        {
            create_rx_rule(eAxC_list_ul,eAxC_list_srs);
        }
    }
    else if(info_.rx_mode == RxApiMode::HYBRID)
    {
        create_rx_rule_for_cplane(eAxC_list_ul, eAxC_list_srs, eAxC_list_dl);
    }
    else if(info_.rx_mode == RxApiMode::UEMODE)
    {
        create_rx_rule_for_dl_uplane(eAxC_list_dl);
    }
}

void Peer::doca_gpu_sem_create()
{
    doca_error_t ret;
    Fronthaul* fh=nic_->get_fronthaul();

    ret = doca_create_semaphore(rxq_->get_doca_rx_items(), rxq_->get_doca_rx_items()->gpu_dev, MAX_SEM_ITEMS, DOCA_GPU_MEM_TYPE_GPU, DOCA_GPU_MEM_TYPE_CPU_GPU, sizeof(struct doca_order_sem_info));
    if (ret != DOCA_SUCCESS)
        THROW_FH(ret, StringBuilder() << "doca_gpu_semaphore_create error");

    NVLOGI_FMT(TAG,"Doca sem created!: sem_in gpu_addr {} cpu_addr {}",
        reinterpret_cast<void*>(rxq_->get_doca_rx_items()->sem_gpu),
        reinterpret_cast<void*>(rxq_->get_doca_rx_items()->sem_cpu));

    if (info_.enable_srs) {
        ret = doca_create_semaphore(rxqSrs_->get_doca_rx_items(), rxq_->get_doca_rx_items()->gpu_dev, MAX_SEM_ITEMS, DOCA_GPU_MEM_TYPE_GPU, DOCA_GPU_MEM_TYPE_CPU_GPU, sizeof(struct doca_order_sem_info));
        if (ret != DOCA_SUCCESS)
            THROW_FH(ret, StringBuilder() << "doca_gpu_semaphore_create error");

        NVLOGI_FMT(TAG,"Doca sem created!: sem_in gpu_addr %p cpu_addr %p",
                reinterpret_cast<void*>(rxqSrs_->get_doca_rx_items()->sem_gpu),
                reinterpret_cast<void*>(rxqSrs_->get_doca_rx_items()->sem_cpu));

        if (ret != DOCA_SUCCESS)
            THROW_FH(ret, StringBuilder() << "doca_gpu_semaphore_create error");

        NVLOGC_FMT(TAG,"Doca sem created!: sem_in gpu_addr {} cpu_addr {}",
            reinterpret_cast<void*>(rxqSrs_->get_doca_rx_items()->sem_gpu), reinterpret_cast<void*>(rxqSrs_->get_doca_rx_items()->sem_cpu));
    }
}

void Peer::create_rx_rule(const std::vector<uint16_t>& eAxC_list_ul,const std::vector<uint16_t>& eAxC_list_srs)
{
    auto port_id = nic_->get_port_id();
    auto name    = nic_->get_name();
    Fronthaul* fh=nic_->get_fronthaul();

    rte_flow_error    err;
    rte_flow_item_eth eth_spec, eth_mask;

    memset(&eth_spec, 0, sizeof(eth_spec));
    memset(&eth_mask, 0, sizeof(eth_mask));
    memcpy(eth_spec.dst.addr_bytes, info_.src_mac_addr.bytes, RTE_ETHER_ADDR_LEN);
    // dMIMO: true wildcard source MAC -- leave eth_spec.src zeroed AND mask.src=0 so
    // this primary UL RX rule matches UL/SRS U-plane from ANY RU/PE into the primary
    // queue. eAxC IDs isolate the flows. NB: on 26-1 mlx5, a non-zero spec.src with a
    // zero mask is mishandled (drops all UL), so spec.src must be left zero here.
    memset(&eth_mask.dst.addr_bytes, 0xFF, sizeof(eth_mask.dst.addr_bytes));
    memset(&eth_mask.src.addr_bytes, 0x00, sizeof(eth_mask.src.addr_bytes));
    eth_spec.type = rte_cpu_to_be_16(RTE_ETHER_TYPE_VLAN);
    eth_mask.type = 0xFFFF;
    eth_spec.has_vlan = 1;
    eth_mask.has_vlan = 1;

    rte_flow_item_vlan vlan_spec, vlan_mask;

    memset(&vlan_spec, 0, sizeof(vlan_spec));
    memset(&vlan_mask, 0, sizeof(vlan_mask));

    vlan_spec.inner_type = 0;
    vlan_mask.inner_type = 0;
    vlan_spec.tci = rte_cpu_to_be_16(info_.vlan.tci);
    vlan_mask.tci = rte_cpu_to_be_16(0xFFFF);

    NVLOGI_FMT(TAG, "Setting up RX flow rules for Peer #{} of NIC {}: "\
            "[Destination MAC: {:02X}:{:02X}:{:02X}:{:02X}:{:02X}:{:02X}|{:02X}:{:02X}:{:02X}:{:02X}:{:02X}:{:02X}] "\
            "[Source MAC: {:02X}:{:02X}:{:02X}:{:02X}:{:02X}:{:02X}|{:02X}:{:02X}:{:02X}:{:02X}:{:02X}:{:02X}] "\
            "[VLAN inner type: 0x{:04x}|0x{:04x}] [VLAN TCI: 0x{:04x}|0x{:04x}] "\
            "[Outer Ethertype: 0x{:04x}|0x{:04x}] [Inner Ethertype: 0x{:04x}|0x{:04x}]\n",
                info_.id, name,
                eth_spec.dst.addr_bytes[0],eth_spec.dst.addr_bytes[1],eth_spec.dst.addr_bytes[2],
                eth_spec.dst.addr_bytes[3],eth_spec.dst.addr_bytes[4],eth_spec.dst.addr_bytes[5],
                eth_mask.dst.addr_bytes[0],eth_mask.dst.addr_bytes[1],eth_mask.dst.addr_bytes[2],
                eth_mask.dst.addr_bytes[3],eth_mask.dst.addr_bytes[4],eth_mask.dst.addr_bytes[5],
                eth_spec.src.addr_bytes[0],eth_spec.src.addr_bytes[1],eth_spec.src.addr_bytes[2],
                eth_spec.src.addr_bytes[3],eth_spec.src.addr_bytes[4],eth_spec.src.addr_bytes[5],
                eth_mask.src.addr_bytes[0],eth_mask.src.addr_bytes[1],eth_mask.src.addr_bytes[2],
                eth_mask.src.addr_bytes[3],eth_mask.src.addr_bytes[4],eth_mask.src.addr_bytes[5],
                rte_be_to_cpu_16(vlan_spec.inner_type), rte_be_to_cpu_16(vlan_mask.inner_type),
                rte_be_to_cpu_16(vlan_spec.tci), rte_be_to_cpu_16(vlan_mask.tci),
                rte_be_to_cpu_16(eth_spec.type), rte_be_to_cpu_16(eth_mask.type),
                rte_be_to_cpu_16(vlan_spec.inner_type), rte_be_to_cpu_16(vlan_mask.inner_type));

    rte_flow_attr         attr{.group = 0, .ingress = 1};
    rte_flow_action_queue queue;
    if(!(fh->get_info().cuda_device_ids.empty()))
    {
        rte_flow_item_ecpri ecpri_spec, ecpri_mask;
        {
            queue.index=rxq_->get_doca_rx_items()->dpdk_queue_idx;
            rte_flow_action       actions[]{
                {.type = RTE_FLOW_ACTION_TYPE_QUEUE, .conf = &queue},
                {.type = RTE_FLOW_ACTION_TYPE_END}};
                for(auto eAxC:eAxC_list_ul)
                {
                    memset(&ecpri_spec, 0, sizeof(ecpri_spec));
                    memset(&ecpri_mask, 0, sizeof(ecpri_mask));
                    ecpri_spec.hdr.common.type = RTE_ECPRI_MSG_TYPE_IQ_DATA;
                    ecpri_mask.hdr.common.type = 0xFF;
                    ecpri_spec.hdr.type0.pc_id = rte_cpu_to_be_16((uint16_t)eAxC);
                    ecpri_mask.hdr.type0.pc_id = 0xFFFF;
                    ecpri_spec.hdr.common.u32 = rte_cpu_to_be_32(ecpri_spec.hdr.common.u32);
                    ecpri_mask.hdr.common.u32 = rte_cpu_to_be_32(ecpri_mask.hdr.common.u32);
                    rte_flow_item patterns[]{
                        {.type = RTE_FLOW_ITEM_TYPE_ETH, .spec = &eth_spec, .last = nullptr, .mask = &eth_mask},
                        {.type = RTE_FLOW_ITEM_TYPE_VLAN, .spec = &vlan_spec, .last = nullptr, .mask = &vlan_mask},
                        {.type = RTE_FLOW_ITEM_TYPE_ECPRI, .spec = &ecpri_spec, .last = nullptr, .mask = &ecpri_mask},
                        {.type = RTE_FLOW_ITEM_TYPE_END}};
                    auto ret = rte_flow_validate(port_id, &attr, patterns, actions, &err);
                    if(ret)
                    {
                        THROW_FH(ret, StringBuilder() << "Failed to validate flow rule for Peer #" << info_.id << " on NIC " << name << ": " << err.message);
                    }

                    auto flow = rte_flow_create(port_id, &attr, patterns, actions, &err);
                    if(!flow && std::string(err.message).find("eCPRI") != std::string::npos)
                    {
                        NVLOGW_FMT(TAG, "eCPRI parser not supported on NIC {}, retrying without eCPRI", name.c_str());
                        patterns[2] = {.type = RTE_FLOW_ITEM_TYPE_END};
                        flow = rte_flow_create(port_id, &attr, patterns, actions, &err);
                    }
                    if(!flow)
                    {
                        THROW_FH(EINVAL, StringBuilder() << "Failed to create flow rule for Peer #" << info_.id << " on NIC " << name << ": " << err.message);
                    }

                    rx_flow_rules_.emplace_back(new RxFlowRule(port_id, flow));
                }
        }
        if(info_.enable_srs)
        {
            queue.index=rxqSrs_->get_doca_rx_items()->dpdk_queue_idx;
            rte_flow_action       actions[]{
                {.type = RTE_FLOW_ACTION_TYPE_QUEUE, .conf = &queue},
                {.type = RTE_FLOW_ACTION_TYPE_END}};
            for(auto eAxC:eAxC_list_srs)
            {
                memset(&ecpri_spec, 0, sizeof(ecpri_spec));
                memset(&ecpri_mask, 0, sizeof(ecpri_mask));
                ecpri_spec.hdr.common.type = RTE_ECPRI_MSG_TYPE_IQ_DATA;
                ecpri_mask.hdr.common.type = 0xFF;
                ecpri_spec.hdr.type0.pc_id = rte_cpu_to_be_16((uint16_t)eAxC);
                ecpri_mask.hdr.type0.pc_id = 0xFFFF;
                ecpri_spec.hdr.common.u32 = rte_cpu_to_be_32(ecpri_spec.hdr.common.u32);
                ecpri_mask.hdr.common.u32 = rte_cpu_to_be_32(ecpri_mask.hdr.common.u32);
                rte_flow_item patterns[]{
                    {.type = RTE_FLOW_ITEM_TYPE_ETH, .spec = &eth_spec, .last = nullptr, .mask = &eth_mask},
                    {.type = RTE_FLOW_ITEM_TYPE_VLAN, .spec = &vlan_spec, .last = nullptr, .mask = &vlan_mask},
                    {.type = RTE_FLOW_ITEM_TYPE_ECPRI, .spec = &ecpri_spec, .last = nullptr, .mask = &ecpri_mask},
                    {.type = RTE_FLOW_ITEM_TYPE_END}};
                auto ret = rte_flow_validate(port_id, &attr, patterns, actions, &err);
                if(ret)
                {
                    THROW_FH(ret, StringBuilder() << "Failed to validate flow rule for Peer #" << info_.id << " on NIC " << name << ": " << err.message);
                }

                auto flow = rte_flow_create(port_id, &attr, patterns, actions, &err);
                if(!flow && std::string(err.message).find("eCPRI") != std::string::npos)
                {
                    NVLOGW_FMT(TAG, "eCPRI parser not supported on NIC {}, retrying without eCPRI", name.c_str());
                    patterns[2] = {.type = RTE_FLOW_ITEM_TYPE_END};
                    flow = rte_flow_create(port_id, &attr, patterns, actions, &err);
                }
                if(!flow)
                {
                    THROW_FH(EINVAL, StringBuilder() << "Failed to create flow rule for Peer #" << info_.id << " on NIC " << name << ": " << err.message);
                }

                rx_flow_rules_.emplace_back(new RxFlowRule(port_id, flow));
            }
        }
    }
    else
    {
        rte_flow_item patterns[]{
            {.type = RTE_FLOW_ITEM_TYPE_ETH, .spec = &eth_spec, .last = nullptr, .mask = &eth_mask},
            {.type = RTE_FLOW_ITEM_TYPE_VLAN, .spec = &vlan_spec, .last = nullptr, .mask = &vlan_mask},
            {.type = RTE_FLOW_ITEM_TYPE_END}};

        NVLOGI_FMT(TAG, "RxQ address from Peer(2) :{}",(void*)rxq_);
        queue.index=rxq_->get_id();
        rte_flow_action       actions[]{
            {.type = RTE_FLOW_ACTION_TYPE_QUEUE, .conf = &queue},
            {.type = RTE_FLOW_ACTION_TYPE_END}};

        auto ret = rte_flow_validate(port_id, &attr, patterns, actions, &err);
        if(ret)
        {
            THROW_FH(ret, StringBuilder() << "Failed to validate flow rule for Peer #" << info_.id << " on NIC " << name << ": " << err.message);
        }

        auto flow = rte_flow_create(port_id, &attr, patterns, actions, &err);
        if(flow == nullptr)
        {
            THROW_FH(EINVAL, StringBuilder() << "Failed to create flow rule for Peer #" << info_.id << " on NIC " << name << ": " << err.message);
        }

        rx_flow_rules_.emplace_back(new RxFlowRule(port_id, flow));
    }
}

void Peer::create_rx_rule_for_cplane(const std::vector<uint16_t>& eAxC_list_ul,const std::vector<uint16_t>& eAxC_list_srs, const std::vector<uint16_t>& eAxC_list_dl)
{
    auto port_id = nic_->get_port_id();
    auto name    = nic_->get_name();

    rte_flow_error    err;
    rte_flow_item_eth eth_spec, eth_mask;

    memset(&eth_spec, 0, sizeof(eth_spec));
    memset(&eth_mask, 0, sizeof(eth_mask));
    memcpy(eth_spec.dst.addr_bytes, info_.src_mac_addr.bytes, RTE_ETHER_ADDR_LEN);
    memcpy(eth_spec.src.addr_bytes, info_.dst_mac_addr.bytes, RTE_ETHER_ADDR_LEN);
    memset(&eth_mask.dst.addr_bytes, 0xFF, sizeof(eth_mask.dst.addr_bytes));
    memset(&eth_mask.src.addr_bytes, 0xFF, sizeof(eth_mask.src.addr_bytes));
    eth_spec.type = rte_cpu_to_be_16(RTE_ETHER_TYPE_VLAN);
    eth_mask.type = 0xFFFF;

    eth_spec.has_vlan = 1;
    eth_mask.has_vlan = 1;

    rte_flow_item_vlan vlan_spec, vlan_mask;

    memset(&vlan_spec, 0, sizeof(vlan_spec));
    memset(&vlan_mask, 0, sizeof(vlan_mask));
    vlan_spec.tci        = rte_cpu_to_be_16(info_.vlan.tci);
    vlan_mask.tci        = rte_cpu_to_be_16(0xFFFF);
    vlan_spec.inner_type = 0;
    vlan_mask.inner_type = 0;

    rte_flow_item_ecpri ecpri_spec, ecpri_mask;

    if(!info_.enable_srs)
    {
        memset(&ecpri_spec, 0, sizeof(ecpri_spec));
        memset(&ecpri_mask, 0, sizeof(ecpri_mask));
        ecpri_spec.hdr.common.type = RTE_ECPRI_MSG_TYPE_RTC_CTRL;
        ecpri_mask.hdr.common.type = 0xFF;
        ecpri_spec.hdr.common.u32 = rte_cpu_to_be_32(ecpri_spec.hdr.common.u32);
        ecpri_mask.hdr.common.u32 = rte_cpu_to_be_32(ecpri_mask.hdr.common.u32);

        rte_flow_item patterns[]{
            {.type = RTE_FLOW_ITEM_TYPE_ETH, .spec = &eth_spec, .last = nullptr, .mask = &eth_mask},
            {.type = RTE_FLOW_ITEM_TYPE_VLAN, .spec = &vlan_spec, .last = nullptr, .mask = &vlan_mask},
            {.type = RTE_FLOW_ITEM_TYPE_ECPRI, .spec = &ecpri_spec, .last = nullptr, .mask = &ecpri_mask},
            {.type = RTE_FLOW_ITEM_TYPE_END}};

        rte_flow_attr         attr{.group = 0, .ingress = 1};
        rte_flow_action_queue queue{.index = rxq_->get_id()};
        rte_flow_action       actions[]{
            {.type = RTE_FLOW_ACTION_TYPE_QUEUE, .conf = &queue},
            {.type = RTE_FLOW_ACTION_TYPE_END}};

        auto ret = rte_flow_validate(port_id, &attr, patterns, actions, &err);
        if(ret)
        {
            THROW_FH(ret, StringBuilder() << "Failed to validate C-plane flow rule for Peer " << info_.dst_mac_addr << " on NIC " << name << ": " << err.message);
        }

        auto flow = rte_flow_create(port_id, &attr, patterns, actions, &err);
        if(!flow && std::string(err.message).find("eCPRI") != std::string::npos)
        {
            // Fallback: retry without eCPRI parser (for NICs without flex parser support)
            NVLOGW_FMT(TAG, "eCPRI parser not supported on NIC {}, retrying without eCPRI", name.c_str());
            patterns[2] = {.type = RTE_FLOW_ITEM_TYPE_END};
            flow = rte_flow_create(port_id, &attr, patterns, actions, &err);
        }
        if(!flow)
        {
            THROW_FH(EINVAL, StringBuilder() << "Failed to create C-plane flow rule for Peer " << info_.dst_mac_addr << " on NIC " << name << ": " << err.message);
        }
    }
    else
    {
        for(auto eAxC:eAxC_list_ul)
        {
            memset(&ecpri_spec, 0, sizeof(ecpri_spec));
            memset(&ecpri_mask, 0, sizeof(ecpri_mask));
            ecpri_spec.hdr.common.type = RTE_ECPRI_MSG_TYPE_RTC_CTRL;
            ecpri_mask.hdr.common.type = 0xFF;
            ecpri_spec.hdr.type0.pc_id = rte_cpu_to_be_16((uint16_t)eAxC);
            ecpri_mask.hdr.type0.pc_id = 0xFFFF;
            ecpri_spec.hdr.common.u32 = rte_cpu_to_be_32(ecpri_spec.hdr.common.u32);
            ecpri_mask.hdr.common.u32 = rte_cpu_to_be_32(ecpri_mask.hdr.common.u32);

            rte_flow_item patterns[]{
                {.type = RTE_FLOW_ITEM_TYPE_ETH, .spec = &eth_spec, .last = nullptr, .mask = &eth_mask},
                {.type = RTE_FLOW_ITEM_TYPE_VLAN, .spec = &vlan_spec, .last = nullptr, .mask = &vlan_mask},
                {.type = RTE_FLOW_ITEM_TYPE_ECPRI, .spec = &ecpri_spec, .last = nullptr, .mask = &ecpri_mask},
                {.type = RTE_FLOW_ITEM_TYPE_END}};

            rte_flow_attr         attr{.group = 0, .ingress = 1};
            rte_flow_action_queue queue{.index = rxq_->get_id()};
            rte_flow_action       actions[]{
                {.type = RTE_FLOW_ACTION_TYPE_QUEUE, .conf = &queue},
                {.type = RTE_FLOW_ACTION_TYPE_END}};

            auto ret = rte_flow_validate(port_id, &attr, patterns, actions, &err);
            if(ret)
            {
                THROW_FH(ret, StringBuilder() << "Failed to validate C-plane flow rule for Peer " << info_.dst_mac_addr << " on NIC " << name << ": " << err.message);
            }

            auto flow = rte_flow_create(port_id, &attr, patterns, actions, &err);
            if(!flow && std::string(err.message).find("eCPRI") != std::string::npos)
            {
                NVLOGW_FMT(TAG, "eCPRI parser not supported on NIC {}, retrying without eCPRI", name.c_str());
                patterns[2] = {.type = RTE_FLOW_ITEM_TYPE_END};
                flow = rte_flow_create(port_id, &attr, patterns, actions, &err);
            }
            if(!flow)
            {
                THROW_FH(EINVAL, StringBuilder() << "Failed to create C-plane flow rule for Peer " << info_.dst_mac_addr << " on NIC " << name << ": " << err.message);
            }
        }
        for(auto eAxC:eAxC_list_srs)
        {
            memset(&ecpri_spec, 0, sizeof(ecpri_spec));
            memset(&ecpri_mask, 0, sizeof(ecpri_mask));
            ecpri_spec.hdr.common.type = RTE_ECPRI_MSG_TYPE_RTC_CTRL;
            ecpri_mask.hdr.common.type = 0xFF;
            ecpri_spec.hdr.type0.pc_id = rte_cpu_to_be_16((uint16_t)eAxC);
            ecpri_mask.hdr.type0.pc_id = 0xFFFF;
            ecpri_spec.hdr.common.u32 = rte_cpu_to_be_32(ecpri_spec.hdr.common.u32);
            ecpri_mask.hdr.common.u32 = rte_cpu_to_be_32(ecpri_mask.hdr.common.u32);

            rte_flow_item patterns[]{
                {.type = RTE_FLOW_ITEM_TYPE_ETH, .spec = &eth_spec, .last = nullptr, .mask = &eth_mask},
                {.type = RTE_FLOW_ITEM_TYPE_VLAN, .spec = &vlan_spec, .last = nullptr, .mask = &vlan_mask},
                {.type = RTE_FLOW_ITEM_TYPE_ECPRI, .spec = &ecpri_spec, .last = nullptr, .mask = &ecpri_mask},
                {.type = RTE_FLOW_ITEM_TYPE_END}};

            rte_flow_attr         attr{.group = 0, .ingress = 1};
            rte_flow_action_queue queue{.index = rxqSrs_->get_id()};
            rte_flow_action       actions[]{
                {.type = RTE_FLOW_ACTION_TYPE_QUEUE, .conf = &queue},
                {.type = RTE_FLOW_ACTION_TYPE_END}};

            auto ret = rte_flow_validate(port_id, &attr, patterns, actions, &err);
            if(ret)
            {
                THROW_FH(ret, StringBuilder() << "Failed to validate C-plane flow rule for Peer " << info_.dst_mac_addr << " on NIC " << name << ": " << err.message);
            }

            auto flow = rte_flow_create(port_id, &attr, patterns, actions, &err);
            if(!flow && std::string(err.message).find("eCPRI") != std::string::npos)
            {
                NVLOGW_FMT(TAG, "eCPRI parser not supported on NIC {}, retrying without eCPRI", name.c_str());
                patterns[2] = {.type = RTE_FLOW_ITEM_TYPE_END};
                flow = rte_flow_create(port_id, &attr, patterns, actions, &err);
            }
            if(!flow)
            {
                THROW_FH(EINVAL, StringBuilder() << "Failed to create C-plane flow rule for Peer " << info_.dst_mac_addr << " on NIC " << name << ": " << err.message);
            }
        }
        for(auto eAxC:eAxC_list_dl)
        {
            memset(&ecpri_spec, 0, sizeof(ecpri_spec));
            memset(&ecpri_mask, 0, sizeof(ecpri_mask));
            ecpri_spec.hdr.common.type = RTE_ECPRI_MSG_TYPE_RTC_CTRL;
            ecpri_mask.hdr.common.type = 0xFF;
            ecpri_spec.hdr.type0.pc_id = rte_cpu_to_be_16((uint16_t)eAxC);
            ecpri_mask.hdr.type0.pc_id = 0xFFFF;
            ecpri_spec.hdr.common.u32 = rte_cpu_to_be_32(ecpri_spec.hdr.common.u32);
            ecpri_mask.hdr.common.u32 = rte_cpu_to_be_32(ecpri_mask.hdr.common.u32);

            rte_flow_item patterns[]{
                {.type = RTE_FLOW_ITEM_TYPE_ETH, .spec = &eth_spec, .last = nullptr, .mask = &eth_mask},
                {.type = RTE_FLOW_ITEM_TYPE_VLAN, .spec = &vlan_spec, .last = nullptr, .mask = &vlan_mask},
                {.type = RTE_FLOW_ITEM_TYPE_ECPRI, .spec = &ecpri_spec, .last = nullptr, .mask = &ecpri_mask},
                {.type = RTE_FLOW_ITEM_TYPE_END}};

            rte_flow_attr         attr{.group = 0, .ingress = 1};
            rte_flow_action_queue queue{.index = rxq_->get_id()};
            rte_flow_action       actions[]{
                {.type = RTE_FLOW_ACTION_TYPE_QUEUE, .conf = &queue},
                {.type = RTE_FLOW_ACTION_TYPE_END}};

            auto ret = rte_flow_validate(port_id, &attr, patterns, actions, &err);
            if(ret)
            {
                THROW_FH(ret, StringBuilder() << "Failed to validate C-plane flow rule for Peer " << info_.dst_mac_addr << " on NIC " << name << ": " << err.message);
            }

            auto flow = rte_flow_create(port_id, &attr, patterns, actions, &err);
            if(!flow && std::string(err.message).find("eCPRI") != std::string::npos)
            {
                NVLOGW_FMT(TAG, "eCPRI parser not supported on NIC {}, retrying without eCPRI", name.c_str());
                patterns[2] = {.type = RTE_FLOW_ITEM_TYPE_END};
                flow = rte_flow_create(port_id, &attr, patterns, actions, &err);
            }
            if(!flow)
            {
                THROW_FH(EINVAL, StringBuilder() << "Failed to create C-plane flow rule for Peer " << info_.dst_mac_addr << " on NIC " << name << ": " << err.message);
            }
        }
    }
}

void Peer::create_rx_rule_for_dl_uplane(const std::vector<uint16_t>& eAxC_list_dl)
{
    auto port_id = nic_->get_port_id();
    auto name    = nic_->get_name();

    rte_flow_error    err;
    rte_flow_item_eth eth_spec, eth_mask;

    memset(&eth_spec, 0, sizeof(eth_spec));
    memset(&eth_mask, 0, sizeof(eth_mask));
    memcpy(eth_spec.dst.addr_bytes, info_.src_mac_addr.bytes, RTE_ETHER_ADDR_LEN);
    memcpy(eth_spec.src.addr_bytes, info_.dst_mac_addr.bytes, RTE_ETHER_ADDR_LEN);
    memset(&eth_mask.dst.addr_bytes, 0xFF, sizeof(eth_mask.dst.addr_bytes));
    memset(&eth_mask.src.addr_bytes, 0xFF, sizeof(eth_mask.src.addr_bytes));
    eth_spec.type = rte_cpu_to_be_16(RTE_ETHER_TYPE_VLAN);
    eth_mask.type = 0xFFFF;

    eth_spec.has_vlan = 1;
    eth_mask.has_vlan = 1;

    rte_flow_item_vlan vlan_spec, vlan_mask;

    memset(&vlan_spec, 0, sizeof(vlan_spec));
    memset(&vlan_mask, 0, sizeof(vlan_mask));
    vlan_spec.tci        = rte_cpu_to_be_16(info_.vlan.tci);
    vlan_mask.tci        = rte_cpu_to_be_16(0xFFFF);
    vlan_spec.inner_type = 0;
    vlan_mask.inner_type = 0;

    // Commented out to allow for both U-Plane and C-Plane to arrive on the RXQ per cell
    rte_flow_item_ecpri ecpri_spec, ecpri_mask;

    // memset(&ecpri_spec, 0, sizeof(ecpri_spec));
    // memset(&ecpri_mask, 0, sizeof(ecpri_mask));
    // ecpri_spec.hdr.common.type = RTE_ECPRI_MSG_TYPE_IQ_DATA;
    // ecpri_mask.hdr.common.type = 0xFF;

    NVLOGI_FMT(TAG, "Setting up U-plane RX flow rules for Peer #{} of NIC {}: "\
            "[Destination MAC: {:02X}:{:02X}:{:02X}:{:02X}:{:02X}:{:02X}|{:02X}:{:02X}:{:02X}:{:02X}:{:02X}:{:02X}] "\
            "[Source MAC: {:02X}:{:02X}:{:02X}:{:02X}:{:02X}:{:02X}|{:02X}:{:02X}:{:02X}:{:02X}:{:02X}:{:02X}] "\
            "[VLAN inner type: 0x{:04x}|0x{:04x}] [VLAN TCI: 0x{:04x}|0x{:04x}] [Outer Ethertype: 0x{:04x}|0x{:04x}] "\
            "[Inner Ethertype: 0x{:04x}|0x{:04x}]",
                info_.id, name,
                eth_spec.dst.addr_bytes[0], eth_spec.dst.addr_bytes[1], eth_spec.dst.addr_bytes[2], eth_spec.dst.addr_bytes[3], eth_spec.dst.addr_bytes[4], eth_spec.dst.addr_bytes[5],
                eth_mask.dst.addr_bytes[0], eth_mask.dst.addr_bytes[1], eth_mask.dst.addr_bytes[2], eth_mask.dst.addr_bytes[3], eth_mask.dst.addr_bytes[4], eth_mask.dst.addr_bytes[5],
                eth_spec.src.addr_bytes[0], eth_spec.src.addr_bytes[1], eth_spec.src.addr_bytes[2], eth_spec.src.addr_bytes[3], eth_spec.src.addr_bytes[4], eth_spec.src.addr_bytes[5],
                eth_mask.src.addr_bytes[0], eth_mask.src.addr_bytes[1], eth_mask.src.addr_bytes[2], eth_mask.src.addr_bytes[3], eth_mask.src.addr_bytes[4], eth_mask.src.addr_bytes[5],
                rte_be_to_cpu_16(vlan_spec.inner_type),rte_be_to_cpu_16(vlan_mask.inner_type),
                rte_be_to_cpu_16(vlan_spec.tci), rte_be_to_cpu_16(vlan_mask.tci),
                rte_be_to_cpu_16(eth_spec.type), rte_be_to_cpu_16(eth_mask.type),
                rte_be_to_cpu_16(vlan_spec.inner_type), rte_be_to_cpu_16(vlan_mask.inner_type)
                // ,static_cast<int>(ecpri_spec.hdr.common.type), static_cast<int>(ecpri_mask.hdr.common.type)
                );

    ecpri_spec.hdr.common.u32 = rte_cpu_to_be_32(ecpri_spec.hdr.common.u32);
    ecpri_mask.hdr.common.u32 = rte_cpu_to_be_32(ecpri_mask.hdr.common.u32);

    rte_flow_item patterns[]{
        {.type = RTE_FLOW_ITEM_TYPE_ETH, .spec = &eth_spec, .last = nullptr, .mask = &eth_mask},
        {.type = RTE_FLOW_ITEM_TYPE_VLAN, .spec = &vlan_spec, .last = nullptr, .mask = &vlan_mask},
        // {.type = RTE_FLOW_ITEM_TYPE_ECPRI, .spec = &ecpri_spec, .last = nullptr, .mask = &ecpri_mask},
        {.type = RTE_FLOW_ITEM_TYPE_END}};

    rte_flow_attr         attr{.group = 0, .ingress = 1};
    rte_flow_action_queue queue{.index = rxq_->get_id()};
    rte_flow_action       actions[]{
        {.type = RTE_FLOW_ACTION_TYPE_QUEUE, .conf = &queue},
        {.type = RTE_FLOW_ACTION_TYPE_END}};

    auto ret = rte_flow_validate(port_id, &attr, patterns, actions, &err);
    if(ret)
    {
        THROW_FH(ret, StringBuilder() << "Failed to validate C-plane flow rule for Peer " << info_.dst_mac_addr << " on NIC " << name << ": " << err.message);
    }

    if(!rte_flow_create(port_id, &attr, patterns, actions, &err))
    {
        THROW_FH(EINVAL, StringBuilder() << "Failed to create C-plane flow rule for Peer " << info_.dst_mac_addr << " on NIC " << name << ": " << err.message);
    }
}

void Peer::create_rx_rule_with_cpu_mirroring()
{
    auto port_id  = nic_->get_port_id();
    auto name     = nic_->get_name();
    auto rxq_pcap = nic_->get_pcap_rxq();
    if(unlikely(rxq_pcap == nullptr || rxq_ == nullptr))
    {
        THROW_FH(EINVAL, StringBuilder() << "Failed to get PCAP RXQ or RXQ for Peer " << info_.dst_mac_addr << " on NIC " << name);
    }

    rte_flow_error    err;
    rte_flow_item_eth eth_spec, eth_mask;

    memset(&eth_spec, 0, sizeof(eth_spec));
    memset(&eth_mask, 0, sizeof(eth_mask));
    memcpy(eth_spec.dst.addr_bytes, info_.src_mac_addr.bytes, RTE_ETHER_ADDR_LEN);
    memcpy(eth_spec.src.addr_bytes, info_.dst_mac_addr.bytes, RTE_ETHER_ADDR_LEN);
    memset(&eth_mask.dst.addr_bytes, 0xFF, sizeof(eth_mask.dst.addr_bytes));
    memset(&eth_mask.src.addr_bytes, 0xFF, sizeof(eth_mask.src.addr_bytes));
    eth_spec.type = rte_cpu_to_be_16(RTE_ETHER_TYPE_VLAN);
    eth_mask.type = 0xFFFF;

    rte_flow_item_vlan vlan_spec, vlan_mask;

    memset(&vlan_spec, 0, sizeof(vlan_spec));
    memset(&vlan_mask, 0, sizeof(vlan_mask));
    vlan_spec.inner_type = rte_cpu_to_be_16(RTE_ETHER_TYPE_ECPRI);
    vlan_mask.inner_type = 0xFFFF;

    NVLOGD_FMT(TAG, "Setting up RX flow rules with pdump support for Peer #{} of NIC {}: "\
            "[Destination MAC: {:02X}:{:02X}:{:02X}:{:02X}:{:02X}:{:02X}|{:02X}:{:02X}:{:02X}:{:02X}:{:02X}:{:02X}] "\
            "[Source MAC: {:02X}:{:02X}:{:02X}:{:02X}:{:02X}:{:02X}|{:02X}:{:02X}:{:02X}:{:02X}:{:02X}:{:02X}] "\
            "[Outer Ethertype: 0x{:04x}|0x{:04x}] "\
            "[Inner Ethertype: 0x{:04x}|0x{:04x}]\n",
                info_.id, name,
                eth_spec.dst.addr_bytes[0], eth_spec.dst.addr_bytes[1], eth_spec.dst.addr_bytes[2], eth_spec.dst.addr_bytes[3], eth_spec.dst.addr_bytes[4], eth_spec.dst.addr_bytes[5],
                eth_mask.dst.addr_bytes[0], eth_mask.dst.addr_bytes[1], eth_mask.dst.addr_bytes[2], eth_mask.dst.addr_bytes[3], eth_mask.dst.addr_bytes[4], eth_mask.dst.addr_bytes[5],
                eth_spec.src.addr_bytes[0], eth_spec.src.addr_bytes[1], eth_spec.src.addr_bytes[2], eth_spec.src.addr_bytes[3], eth_spec.src.addr_bytes[4], eth_spec.src.addr_bytes[5],
                eth_mask.src.addr_bytes[0], eth_mask.src.addr_bytes[1], eth_mask.src.addr_bytes[2], eth_mask.src.addr_bytes[3], eth_mask.src.addr_bytes[4], eth_mask.src.addr_bytes[5],
                rte_be_to_cpu_16(eth_spec.type), rte_be_to_cpu_16(eth_mask.type),
                rte_be_to_cpu_16(vlan_spec.inner_type), rte_be_to_cpu_16(vlan_mask.inner_type));

    rte_flow_item patterns[]{
        {.type = RTE_FLOW_ITEM_TYPE_ETH, .spec = &eth_spec, .last = nullptr, .mask = &eth_mask},
        {.type = RTE_FLOW_ITEM_TYPE_VLAN, .spec = &vlan_spec, .last = nullptr, .mask = &vlan_mask},
        {.type = RTE_FLOW_ITEM_TYPE_END}};

    // Group 1 flow rule: Mirror incoming packets
    {
        rte_flow_attr         attr{.group = 1, .ingress = 1};
        rte_flow_action_queue pcap_queue{.index = rxq_pcap->get_id()};
        rte_flow_action       mirror_actions[]{
            {.type = RTE_FLOW_ACTION_TYPE_QUEUE, .conf = &pcap_queue},
            {.type = RTE_FLOW_ACTION_TYPE_END}};

        rte_flow_action_queue  queue{rxq_->get_id()};
        rte_flow_action_sample sample_action{.ratio = 1, .actions = &mirror_actions[0]};
        rte_flow_action        actions[]{
            {.type = RTE_FLOW_ACTION_TYPE_QUEUE, .conf = &queue},
            {.type = RTE_FLOW_ACTION_TYPE_SAMPLE, .conf = &sample_action},
            {.type = RTE_FLOW_ACTION_TYPE_END}};

        auto ret = rte_flow_validate(port_id, &attr, patterns, actions, &err);
        if(ret)
        {
            THROW_FH(ret, StringBuilder() << "Failed to validate group " << attr.group << " RX flow rule with mirroring for Peer " << info_.dst_mac_addr << " on NIC " << name << ": " << err.message);
        }

        auto flow = rte_flow_create(port_id, &attr, patterns, actions, &err);
        if(flow == nullptr)
        {
            THROW_FH(EINVAL, StringBuilder() << "Failed to create group " << attr.group << " RX flow rule with mirroring for Peer " << info_.dst_mac_addr << " on NIC " << name << ": " << err.message);
        }

        rx_flow_rules_.emplace_back(new RxFlowRule(port_id, flow));
    }

    // Group 0 flow rule: Jump to group 0 rule
    {
        rte_flow_attr        attr{.group = 0, .ingress = 1};
        rte_flow_action_jump jump{.group = 1};

        rte_flow_action actions[]{
            {.type = RTE_FLOW_ACTION_TYPE_JUMP, .conf = &jump},
            {.type = RTE_FLOW_ACTION_TYPE_END}};

        auto ret = rte_flow_validate(port_id, &attr, patterns, actions, &err);
        if(ret)
        {
            THROW_FH(ret, StringBuilder() << "Failed to validate group " << attr.group << " RX flow rule with mirroring for Peer " << info_.dst_mac_addr << " on NIC " << name << ": " << err.message);
        }

        auto flow = rte_flow_create(port_id, &attr, patterns, actions, &err);
        if(flow == nullptr)
        {
            THROW_FH(EINVAL, StringBuilder() << "Failed to create group " << attr.group << " RX flow rule with mirroring for Peer " << info_.dst_mac_addr << " on NIC " << name << ": " << err.message);
        }

        rx_flow_rules_.emplace_back(new RxFlowRule(port_id, flow));
    }
}

void Peer::get_uplane_txqs(Txq** txqs, size_t* num_txqs) const
{
    auto&  txqs_local = txqs_uplane_.items();
    size_t txq_idx{};

    for(txq_idx = 0; (txq_idx < txqs_local.size()) && (txq_idx < *num_txqs); txq_idx++)
    {
        txqs[txq_idx] = txqs_local[txq_idx];
    }

    *num_txqs = txq_idx;
}

void Peer::get_uplane_txqs_gpu(Txq** txqs, size_t* num_txqs) const
{
    auto&  txqs_local = txqs_uplane_gpu_.items();
    size_t txq_idx{};

    for(txq_idx = 0; (txq_idx < txqs_local.size()) && (txq_idx < *num_txqs); txq_idx++)
    {
        txqs[txq_idx] = txqs_local[txq_idx];
    }

    *num_txqs = txq_idx;
}

void Peer::adjust_src_mac_addr()
{
    // If the src eth addr provided is all 0s, fetch the eth addr of the NIC device used by this peer
    auto src_addr = reinterpret_cast<rte_ether_addr*>(&info_.src_mac_addr.bytes);
    if(rte_is_zero_ether_addr(src_addr) && rte_eth_macaddr_get(nic_->get_port_id(), src_addr))
    {
        THROW_FH(ENODEV, StringBuilder() << "Could not get NIC " << nic_->get_name() << " MAC address");
    }
}
Nic* Peer::get_nic() const
{
    return nic_;
}

Fronthaul* Peer::get_fronthaul() const
{
    return nic_->get_fronthaul();
}

Txq* Peer::get_next_uplane_txq()
{
    return txqs_uplane_.next();
}

Rxq* Peer::get_rxq()
{
    return rxq_;
}

Rxq* Peer::get_rxq_srs()
{
    return rxqSrs_;
}

Txq* Peer::get_next_uplane_txq_gpu()
{
    return txqs_uplane_gpu_.next();
}

PeerInfo const& Peer::get_info() const
{
    return info_;
}

std::unordered_map<u_int16_t, u_int16_t>& Peer::get_eaxcid_idx_mp()
{
    return eaxcid_idx_mp;
}

std::unordered_map<u_int16_t, u_int16_t>& Peer::get_dlu_eaxcid_idx_mp()
{
    return dlu_eaxcid_idx_mp;
}

PeerId Peer::get_id() const
{
    return info_.id;
}

void Peer::poll_tx_complete()
{
    auto& txqs = txqs_uplane_.items();
    for(auto& txq : txqs)
    {
        txq->poll_complete();
    }
}

inline void init_packet(const CPlaneMsgSendInfo& info, rte_mbuf** mbufs, uint16_t& packet_num, rte_mbuf*& m, Flow* flow, PacketHeaderTemplate*& data, uint8_t*& common_hdr_ptr, uint8_t*& section_ptr, uint16_t common_hdr_size, uint16_t pkt_section_info_room, uint16_t& pkt_remaining_capacity, uint16_t& total_section_info_size, uint8_t& sections_generated) noexcept
{
    m           = mbufs[packet_num++];

    m->ol_flags = 0;
    data        = rte_pktmbuf_mtod(m, PacketHeaderTemplate*);
    memcpy(data, &flow->get_packet_header_template(), sizeof(PacketHeaderTemplate));

    common_hdr_ptr = rte_pktmbuf_mtod_offset(m, uint8_t*, sizeof(PacketHeaderTemplate));
    memcpy(common_hdr_ptr, &info.section_common_hdr.sect_1_common_hdr, common_hdr_size);

    section_ptr = common_hdr_ptr + common_hdr_size;

    total_section_info_size = 0;
    sections_generated      = 0;
    pkt_remaining_capacity  = pkt_section_info_room;
}

inline void add_packet_to_mbuf_array(rte_mbuf*& m, MbufArray& mbufs)
{
    mbufs.push_back(m);
}

inline void populate_section_hdr(uint8_t* section_ptr, CPlaneSectionInfo& section_info, uint16_t section_hdr_size, uint8_t& sections_generated)
{
    memcpy(section_ptr, &section_info, section_hdr_size);
    sections_generated++;
}

// Helper function to finalize packet with common settings
inline void finalize_packet_common(rte_mbuf*& m, rte_mbuf** mbufs, uint16_t mtu, uint16_t& pkt_remaining_capacity,
                                    PacketHeaderTemplate*& data, SequenceIdGenerator& sequence_id_generator,
                                    uint8_t*& common_hdr_ptr, uint8_t sections_generated,
                                    const CPlaneMsgSendInfo& info, sendCPlaneTxqInfo& txq_info,
                                    Fronthaul* fhi) {
    if(unlikely(m == nullptr))
    {
        THROW_FH(EINVAL, StringBuilder() << "mbuf is nullptr in finalize_packet_common");
    }

    // Common packet finalization
    if(m->next == nullptr)
    {
        m->data_len = mtu - pkt_remaining_capacity;
        m->pkt_len  = m->data_len;
        data->ecpri.ecpriSeqid = sequence_id_generator.next();
        data->ecpri.ecpriPayload = rte_cpu_to_be_16(m->data_len - sizeof(PacketHeaderTemplate) + 4);
    }
    else // if packet has a chained mbuf for BFW
    {
        // Fix if there are section extension 4/5 + 11
        // if chaining is enabled, data_len holds se4+se5 extlen
        m->data_len = mtu - pkt_remaining_capacity;
        data->ecpri.ecpriSeqid = sequence_id_generator.next();
        data->ecpri.ecpriPayload = rte_cpu_to_be_16(m->data_len - sizeof(PacketHeaderTemplate) + 4);
        // With chaining, we need to subtract the data_len of each chained mbuf from the total data_len
        rte_mbuf* current_mbuf = m;
        while(current_mbuf != nullptr)
        {
            if(current_mbuf != m)
            {
                m->data_len -= current_mbuf->data_len;
            }
            current_mbuf = current_mbuf->next;
        }
        m->pkt_len  = m->data_len;
    }

    *(common_hdr_ptr + offsetof(oran_cmsg_radio_app_hdr, numberOfSections)) = sections_generated;

    // Handle tx window timestamp
    auto tx_window_start = txq_info.is_bfw_send_req ? info.tx_window.tx_window_bfw_start : info.tx_window.tx_window_start;
    if (tx_window_start > *(txq_info.last_cplane_tx_ts_)) {
        mbufs[0]->ol_flags = fhi->get_timestamp_mask_();
        *RTE_MBUF_DYNFIELD(mbufs[0], fhi->get_timestamp_offset(), uint64_t*) = tx_window_start;
        *(txq_info.last_cplane_tx_ts_) = tx_window_start;
    }

    // Add packet to appropriate mbuf array
    add_packet_to_mbuf_array(m, *(txq_info.mbufs_));
}

// coverity[exn_spec_violation]
inline void add_new_packet(Fronthaul* fhi, const CPlaneMsgSendInfo& info, rte_mbuf** mbufs, uint16_t& packet_num, rte_mbuf*& m, Flow* flow, SequenceIdGenerator& sequence_id_generator, PacketHeaderTemplate*& data, uint8_t*& common_hdr_ptr, uint8_t*& section_ptr, uint16_t common_hdr_size, uint16_t mtu, uint16_t pkt_section_info_room, uint16_t& pkt_remaining_capacity, uint16_t& total_section_info_size, uint8_t& sections_generated, sendCPlaneTxqInfo& txq_info) noexcept
{
    // End current packet
    finalize_packet_common(m, mbufs, mtu, pkt_remaining_capacity, data, sequence_id_generator,
                         common_hdr_ptr, sections_generated, info, txq_info, fhi);

    // Start new packet for remaining content
    init_packet(info, mbufs, packet_num, m, flow, data, common_hdr_ptr, section_ptr,
               common_hdr_size, pkt_section_info_room, pkt_remaining_capacity,
               total_section_info_size, sections_generated);
}

// coverity[exn_spec_violation]
inline void populate_last_packet(Fronthaul* fhi, const CPlaneMsgSendInfo& info, rte_mbuf** mbufs, rte_mbuf*& m, SequenceIdGenerator& sequence_id_generator, PacketHeaderTemplate*& data, uint8_t*& common_hdr_ptr, uint16_t mtu, uint16_t& pkt_remaining_capacity, uint8_t& sections_generated, sendCPlaneTxqInfo& txq_info) noexcept
{
    finalize_packet_common(m, mbufs, mtu, pkt_remaining_capacity, data, sequence_id_generator,
                         common_hdr_ptr, sections_generated, info, txq_info, fhi);
}

inline uint16_t populate_se4(uint8_t* section_ptr, CPlaneSectionExtInfo* ext4_ptr)
{
    auto ext4_common_hdr = static_cast<oran_cmsg_ext_hdr*>(&ext4_ptr->sect_ext_common_hdr);
    memcpy(section_ptr, ext4_common_hdr, sizeof(oran_cmsg_ext_hdr));
    section_ptr += sizeof(oran_cmsg_ext_hdr);

    oran_cmsg_sect_ext_type_4* ext4_hdr = static_cast<oran_cmsg_sect_ext_type_4*>(&ext4_ptr->ext_4.ext_hdr);
    memcpy(section_ptr, ext4_hdr, sizeof(oran_cmsg_sect_ext_type_4));
    section_ptr += sizeof(oran_cmsg_sect_ext_type_4);
    return sizeof(oran_cmsg_ext_hdr) + sizeof(oran_cmsg_sect_ext_type_4);
}

inline uint16_t populate_se5(uint8_t* section_ptr, CPlaneSectionExtInfo* ext5_ptr)
{
    auto ext5_hdr = static_cast<oran_cmsg_ext_hdr*>(&ext5_ptr->sect_ext_common_hdr);
    memcpy(section_ptr, ext5_hdr, sizeof(oran_cmsg_ext_hdr));
    section_ptr += sizeof(oran_cmsg_ext_hdr);

    uint64_t* ext5_2sets = reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(&ext5_ptr->ext_5.ext_hdr) + 1);
    *ext5_2sets          = rte_cpu_to_be_64(*ext5_2sets);

    oran_cmsg_sect_ext_type_5* ext5_common_hdr = static_cast<oran_cmsg_sect_ext_type_5*>(&ext5_ptr->ext_5.ext_hdr);
    memcpy(section_ptr, ext5_common_hdr, sizeof(oran_cmsg_sect_ext_type_5));
    return sizeof(oran_cmsg_ext_hdr) + sizeof(oran_cmsg_sect_ext_type_5);
}

inline void chain_bfw_cplane_mmimo(bfw_cplane_chain_info* bfw_cplane_chain_info)
{
    rte_mbuf* bfw_mbuf = nullptr;
    rte_mbuf* padding_mbuf = nullptr;

    // TODO: count needed mbufs in count_cplane_packets_mmimo
    // if(bfw_cplane_chain_info->bfw_padding_size > 0) {
    //     constexpr uint8_t num_chain_mbufs = 2; // 1 bfw + 1 padding
    //     rte_mbuf* mbufs[num_chain_mbufs];
    //     if(unlikely(0 != rte_mempool_get_bulk(bfw_cplane_chain_info->cpu_mbuf_pool,reinterpret_cast<void**>(mbufs), num_chain_mbufs))) {
    //         THROW_FH(ENOMEM, StringBuilder() << "Failed to allocate " << num_chain_mbufs << " mbufs for C-plane message");
    //     }
    //     bfw_mbuf = mbufs[0];
    //     padding_mbuf = mbufs[1];
    // } else {
    //     bfw_mbuf = rte_pktmbuf_alloc(bfw_cplane_chain_info->cpu_mbuf_pool);
    //     if(unlikely(bfw_mbuf == nullptr)) {
    //         THROW_FH(ENOMEM, StringBuilder() << "Failed to allocate bfw mbuf for C-plane message");
    //     }
    // }

    bfw_mbuf = bfw_cplane_chain_info->chain_mbufs[bfw_cplane_chain_info->chained_mbufs++];
    if(bfw_cplane_chain_info->bfw_padding_size > 0) {
        padding_mbuf = bfw_cplane_chain_info->chain_mbufs[bfw_cplane_chain_info->chained_mbufs++];
    }

    // Chain IQ samples
    auto& bfw_buffer_addr = bfw_cplane_chain_info->bfw_buffer;
    auto bfw_buffer_size = bfw_cplane_chain_info->bfw_buffer_size;
    attach_extbuf(bfw_mbuf, bfw_buffer_addr, bfw_buffer_size);

    bfw_mbuf->data_len = bfw_buffer_size;
    bfw_mbuf->pkt_len = bfw_mbuf->data_len;

    // Header mbuf size is added in finalize_packet_common, data_len is used to hold ext4/5 extLen
    auto& header_mbuf = bfw_cplane_chain_info->header_mbuf;
    if(unlikely(rte_pktmbuf_chain(header_mbuf, bfw_mbuf))) {
        THROW_FH(EINVAL, StringBuilder() << "Failed to chain header_mbuf->bfw_mbuf");
    }

    if(bfw_cplane_chain_info->bfw_padding_size == 0) {
        return;
    } else if(bfw_cplane_chain_info->bfw_padding_size > 0) {
        padding_mbuf->data_len = bfw_cplane_chain_info->bfw_padding_size;
        padding_mbuf->pkt_len = padding_mbuf->data_len;
        memset(rte_pktmbuf_mtod(padding_mbuf, uint8_t *), 0, bfw_cplane_chain_info->bfw_padding_size); // Only zero out the padding size needed
        if(unlikely(rte_pktmbuf_chain(header_mbuf, padding_mbuf))) {
            THROW_FH(EINVAL, StringBuilder() << "Failed to chain header_mbuf->padding_mbuf");
        }
    }
}

std::tuple<uint16_t, uint16_t> populate_se11_header(uint8_t*& section_ptr, CPlaneSectionExtInfo* ext11_ptr, uint16_t num_prbc, uint16_t start_bundle, uint16_t num_bundles)
{
    // Get disable BFWs flag
    auto disableBFWs = oran_cmsg_get_ext_11_disableBFWs(&ext11_ptr->ext_11.ext_hdr);

    // Calculate total extension length
    uint16_t ext11_len = sizeof(oran_cmsg_ext_hdr) + sizeof(oran_cmsg_sect_ext_type_11);
    ext11_len += disableBFWs ? 0 : sizeof(oran_cmsg_sect_ext_type_11_disableBFWs_0_bfwCompHdr);

    // Get bundle sizes
    auto bundle_hdr_size = ext11_ptr->ext_11.bundle_hdr_size;
    auto bfwIQ_size = ext11_ptr->ext_11.bfwIQ_size;
    auto bundle_size = bundle_hdr_size + (disableBFWs ? 0 : bfwIQ_size);

    // Copy common header
    oran_cmsg_ext_hdr* ext11_common_hdr = static_cast<oran_cmsg_ext_hdr*>(&ext11_ptr->sect_ext_common_hdr);
    memcpy(section_ptr, ext11_common_hdr, sizeof(oran_cmsg_ext_hdr));
    section_ptr += sizeof(oran_cmsg_ext_hdr);

    // Copy extension type 11 header
    oran_cmsg_sect_ext_type_11* ext11_hdr = static_cast<oran_cmsg_sect_ext_type_11*>(&ext11_ptr->ext_11.ext_hdr);
    memcpy(section_ptr, ext11_hdr, sizeof(oran_cmsg_sect_ext_type_11));
    oran_cmsg_sect_ext_type_11* current_ext11_ptr = reinterpret_cast<oran_cmsg_sect_ext_type_11*>(section_ptr);

    // Verify no further extensions after SE11
    if(unlikely(oran_cmsg_get_ext_ef(ext11_common_hdr))) {
        THROW_FH(EINVAL, StringBuilder() << "No further extensions after SE11 !");
    }

    // Validate static BFW if enabled
    if (ext11_ptr->ext_11.static_bfw) {
        if (unlikely(ext11_ptr->ext_11.numBundPrb < num_prbc)) {
            THROW_FH(EINVAL, StringBuilder() << "numBundPrb less than expected");
        }
    }

    section_ptr += sizeof(oran_cmsg_sect_ext_type_11);

    // Copy compression header if BFWs are enabled
    if (!disableBFWs) {
        auto ext_comp_ptr = &ext11_ptr->ext_11.ext_comp_hdr;
        memcpy(section_ptr, ext_comp_ptr, sizeof(oran_cmsg_sect_ext_type_11_disableBFWs_0_bfwCompHdr));
        section_ptr += sizeof(oran_cmsg_sect_ext_type_11_disableBFWs_0_bfwCompHdr);
    }

    ext11_len += bundle_size * num_bundles;

    // Zero pad to 4-byte boundary per Table 7.7.11.1-1: Section Extension 11 Data Format (when disableBFWs = 0)
    auto padding = oran_cmsg_se11_disableBFWs_0_padding_bytes(ext11_len);
    ext11_len += padding;
    current_ext11_ptr->extLen = rte_cpu_to_be_16(ext11_len >> 2);
    return {ext11_len, padding};
}

inline void populate_se11_disablebfw_1_bundles(
    uint8_t*& section_ptr,
    CPlaneSectionExtInfo* ext11_ptr,
    int start_bundle,
    int num_bundles,
    size_t padding)
{
    for(int bundle_idx = start_bundle; bundle_idx < start_bundle + num_bundles; ++bundle_idx)
    {
        auto& bundle_info = ext11_ptr->ext_11.bundles[bundle_idx];
        auto bundle_ptr = reinterpret_cast<oran_cmsg_sect_ext_type_11_disableBFWs_1_bundle*>(section_ptr);
        memcpy(bundle_ptr, &bundle_info.disableBFWs_1, sizeof(oran_cmsg_sect_ext_type_11_disableBFWs_1_bundle));
        bundle_ptr->beamId = bundle_info.disableBFWs_1.beamId.get();
        bundle_ptr->reserved = 0;  // Initialize reserved field
        section_ptr += sizeof(oran_cmsg_sect_ext_type_11_disableBFWs_1_bundle);
    }
    memset(section_ptr, 0, padding);
    section_ptr += padding;
}

// --- dMIMO SE11 PE1->64 antenna-port scatter ------------------------------
// The cuPHY coeff buffer holds the 32 computed PE1 BFW weights in chain-major
// (eAxC_id_srs) order: [RFSoC_A: ports 1,2,3,4,9,10,11,12][RFSoC_B ...][C][D].
// The Amplitech RU lays its 64 DL-TRX elements out antenna-port-major (port
// 1..16, RFSoC A..D within each port); PE1 owns antenna ports 1-4 & 9-12, which
// map to wire slots {0-15, 32-47}, while PE2 (ports 5-8 & 13-16) owns {16-31,
// 48-63}. So PE1 is NOT a contiguous 0..31 block: each computed weight must be
// placed at its antenna-port slot and the PE2 slots zeroed. se11_pe1_scatter[i]
// = wire antenna index for chain-major source weight i.
static const uint8_t se11_pe1_scatter[32] = {
     0,  4,  8, 12, 32, 36, 40, 44,   // RFSoC_A: ports 1-4, 9-12
     1,  5,  9, 13, 33, 37, 41, 45,   // RFSoC_B: ports 1-4, 9-12
     2,  6, 10, 14, 34, 38, 42, 46,   // RFSoC_C: ports 1-4, 9-12
     3,  7, 11, 15, 35, 39, 43, 47    // RFSoC_D: ports 1-4, 9-12
};
// MSB-first bit get/set over the O-RAN BFW octet stream (bit 0 = MSB of byte 0).
static inline uint32_t se11_get_bits(const uint8_t* b, int pos, int n) {
    uint32_t v = 0;
    for(int k = 0; k < n; ++k) { int p = pos + k; v = (v << 1) | ((b[p >> 3] >> (7 - (p & 7))) & 1u); }
    return v;
}
static inline void se11_set_bits(uint8_t* b, int pos, int n, uint32_t v) {
    for(int k = 0; k < n; ++k) {
        int p = pos + k, by = p >> 3, sh = 7 - (p & 7);
        if((v >> (n - 1 - k)) & 1u) b[by] |= (1u << sh); else b[by] &= ~(1u << sh);
    }
}
// Scatter the BFP-packed PE1 weights (real_sz bytes, 32 antennas) from src into
// the 64-antenna wire layout (wire_sz bytes) in dst, zeroing PE2 slots. iqw =
// bits per I/Q component (so 2*iqw bits per antenna). Falls back to a contiguous
// copy if this is not the 32->64 PE1 case (forward-safe).
static inline void se11_scatter_bfw_pe1(uint8_t* dst, const uint8_t* src, int iqw,
                                        int real_sz, int wire_sz) {
    const int unit  = 2 * iqw;                // bits per antenna weight (I+Q)
    const int n_src = (real_sz * 8) / unit;   // computed antennas (expect 32)
    const int n_dst = (wire_sz * 8) / unit;   // wire antennas (expect 64)
    memset(dst, 0, wire_sz);
    if(n_src != 32 || n_dst != 64 || unit > 32) { memcpy(dst, src, real_sz); return; }
    for(int i = 0; i < 32; ++i)
        se11_set_bits(dst, se11_pe1_scatter[i] * unit, unit, se11_get_bits(src, i * unit, unit));
}

inline void populate_se11_disablebfw_0_bundles(
    uint8_t*& section_ptr,
    CPlaneSectionExtInfo* ext11_ptr,
    int start_bundle,
    int num_bundles,
    uint16_t bfwIQ_size,
    size_t padding)
{
    for(int bundle_idx = start_bundle; bundle_idx < start_bundle + num_bundles; ++bundle_idx)
    {
        auto& bundle_info = ext11_ptr->ext_11.bundles[bundle_idx];
        auto comp_meth = static_cast<UserDataCompressionMethod>(ext11_ptr->ext_11.ext_comp_hdr.bfwCompMeth.get());
        if(comp_meth == UserDataCompressionMethod::NO_COMPRESSION)
        {
            auto bundle_ptr = reinterpret_cast<oran_cmsg_sect_ext_type_11_disableBFWs_0_bundle_uncompressed*>(section_ptr);
            memcpy(bundle_ptr, &bundle_info.disableBFWs_0_uncompressed, sizeof(oran_cmsg_sect_ext_type_11_disableBFWs_0_bundle_uncompressed));
            bundle_ptr->beamId = bundle_info.disableBFWs_0_uncompressed.beamId.get();
            bundle_ptr->reserved = 0;  // Initialize reserved field
            auto bundle_iq_ptr = reinterpret_cast<uint8_t*>(bundle_ptr->bfw);
            uint16_t real_sz = ext11_ptr->ext_11.real_bfwIQ_size;
            if(real_sz == 0 || real_sz > bfwIQ_size) real_sz = bfwIQ_size; // no padding (static/PDSCH/other)
            if(bfwIQ_size > real_sz) // dMIMO: scatter PE1 weights to antenna-port slots, zero PE2
                se11_scatter_bfw_pe1(bundle_iq_ptr, bundle_info.bfwIQ, ext11_ptr->ext_11.ext_comp_hdr.bfwIqWidth.get(), real_sz, bfwIQ_size);
            else
                memcpy(bundle_iq_ptr, bundle_info.bfwIQ, real_sz);
            section_ptr = bundle_iq_ptr + bfwIQ_size;
        }
        else if(comp_meth == UserDataCompressionMethod::BLOCK_FLOATING_POINT)
        {
            auto bundle_ptr = reinterpret_cast<oran_cmsg_sect_ext_type_11_disableBFWs_0_bfp_compressed_bundle_hdr*>(section_ptr);
            memcpy(bundle_ptr, &bundle_info.disableBFWs_0_compressed, sizeof(oran_cmsg_sect_ext_type_11_disableBFWs_0_bfp_compressed_bundle_hdr));
            bundle_ptr->beamId                = bundle_info.disableBFWs_0_compressed.beamId.get();
            bundle_ptr->bfwCompParam.exponent = bundle_info.disableBFWs_0_compressed.bfwCompParam.exponent.get();
            bundle_ptr->bfwCompParam.reserved = 0;  // Initialize reserved field

            auto bundle_iq_ptr                = reinterpret_cast<uint8_t*>(bundle_ptr->bfw);
            uint16_t real_sz = ext11_ptr->ext_11.real_bfwIQ_size;
            if(real_sz == 0 || real_sz > bfwIQ_size) real_sz = bfwIQ_size; // no padding (static/PDSCH/other)
            if(bfwIQ_size > real_sz) // dMIMO: scatter PE1 weights to antenna-port slots, zero PE2
                se11_scatter_bfw_pe1(bundle_iq_ptr, bundle_info.bfwIQ, ext11_ptr->ext_11.ext_comp_hdr.bfwIqWidth.get(), real_sz, bfwIQ_size);
            else
                memcpy(bundle_iq_ptr, bundle_info.bfwIQ, real_sz);
            section_ptr = bundle_iq_ptr + bfwIQ_size;
        }
    }
    memset(section_ptr, 0, padding);
    section_ptr += padding;
}

inline void populate_se11_disablebfw_0_bundles_with_chaining(
    CPlaneSectionExtInfo* ext11_ptr,
    int start_bundle,
    int num_bundles,
    size_t bundle_size,
    size_t padding,
    bfw_cplane_chain_info* bfw_cplane_chain_info)
{
    size_t offset = (start_bundle + ext11_ptr->ext_11.start_bundle_offset_in_bfw_buffer) * bundle_size;
    bfw_cplane_chain_info->bfw_buffer_size = bundle_size * num_bundles;
    bfw_cplane_chain_info->bfw_padding_size = padding;
    bfw_cplane_chain_info->bfw_buffer = &(ext11_ptr->ext_11.bfwIQ[offset]);
    chain_bfw_cplane_mmimo(bfw_cplane_chain_info);
}

inline uint16_t populate_se11(uint8_t* section_ptr, CPlaneSectionExtInfo* ext11_ptr, uint16_t num_prbc, int start_bundle, int num_bundles, bfw_cplane_chain_info* bfw_cplane_chain_info)
{
    auto [ext11_len, padding] = populate_se11_header(section_ptr, ext11_ptr, num_prbc, start_bundle, num_bundles);
    auto disableBFWs = oran_cmsg_get_ext_11_disableBFWs(&ext11_ptr->ext_11.ext_hdr);
    auto bundle_hdr_size = ext11_ptr->ext_11.bundle_hdr_size;
    auto bfwIQ_size = ext11_ptr->ext_11.bfwIQ_size;
    auto bundle_size = bundle_hdr_size + (disableBFWs ? 0 : bfwIQ_size);

    bool chaining_enabled = (bfw_cplane_chain_info != nullptr) && (bfw_cplane_chain_info->bfw_chain_mode == BfwCplaneChainingMode::CPU_CHAINING || bfw_cplane_chain_info->bfw_chain_mode == BfwCplaneChainingMode::GPU_CHAINING);
    // Chaining is only used for DL dynamic BFW, disableBFWs == 1
    if(disableBFWs)
    {
        populate_se11_disablebfw_1_bundles(section_ptr, ext11_ptr, start_bundle, num_bundles, padding);
    }
    else if(ext11_ptr->ext_11.static_bfw)
    {
        populate_se11_disablebfw_0_bundles(section_ptr, ext11_ptr, start_bundle, num_bundles, bfwIQ_size, padding);
    }
    else if(chaining_enabled)
    {
        populate_se11_disablebfw_0_bundles_with_chaining(ext11_ptr, start_bundle, num_bundles, bundle_size, padding, bfw_cplane_chain_info);
    }
    else
    {
        populate_se11_disablebfw_0_bundles(section_ptr, ext11_ptr, start_bundle, num_bundles, bfwIQ_size, padding);
    }

    return ext11_len;
}
inline void update_cplane_combined_reMask(CPlaneSectionInfo& section_info, int ap_idx, int sym_id, uint16_t reMask)
{
    auto& cplane_sections_info = section_info.prb_info->cplane_sections_info;
    cplane_sections_info[sym_id]->combined_reMask[ap_idx] |= reMask;
}

/**
 * Handle section ID assignment based on lookback index
 * 
 * Assigns section ID based on section_id_lookback_index for CSI-RS compact signaling.
 * 
 * Case 1: lookback_index > 0 (Reference existing section)
 *   The current section reuses the section ID from a previous section in the array.
 *   This enables CSI-RS compact signaling where multiple antenna ports share the same section ID.
 *   
 *   Example: If section_num=3 and lookback_index=2:
 *     section_id = sections[3-2].sectionId = sections[1].sectionId
 *
 * Case 2: lookback_index == 0 (Create new section ID)
 *   The current section is a reference section that gets a new unique section ID.
 *   This section ID is saved so future sections can reference it via lookback.
 *   
 *   Note: section_num - lookback_index = section_num - 0 = section_num (saves to current section)
 * 
 * @param[in,out] sections Array of C-plane section info
 * @param[in] section_num Current section number
 * @param[in] ap_idx Antenna port index
 * @param[in] sym_id Symbol ID
 * @param[in,out] nxt_section_id Array of atomic next section IDs per antenna port
 * @return Section ID to use for this section
 */
inline uint16_t handle_section_id_assignment(CPlaneSectionInfo* sections, const uint8_t section_num, const int ap_idx, const int sym_id, uint16_t* nxt_section_id)
{
    uint16_t section_id = nxt_section_id[ap_idx];
    
    if(sections[section_num].section_id_lookback_index > 0)
    {
        // Reuse section ID from referenced section (lookback_index positions back)
        if (sections[section_num].section_id_lookback_index > section_num) {
            THROW_FH(EINVAL, StringBuilder() << "Invalid lookback index: " << sections[section_num].section_id_lookback_index << " exceeds section number: " << section_num);
        }
        auto& ref_section = sections[section_num - sections[section_num].section_id_lookback_index];
        section_id = ref_section.sect_1.sectionId;
        //The refrence section should also carry the reMask of all the section that refrence its 
        //section id.
        update_cplane_combined_reMask(ref_section, ap_idx, sym_id, sections[section_num].prb_info->common.reMask);
    }
    else
    {
        nxt_section_id[ap_idx]++;
        // Store the new section ID in current section (section_num - 0 = section_num)
        // This makes the current section the reference point for future sections
        sections[section_num].sect_1.sectionId = section_id;
    }
    
    return section_id;
}

inline void store_cplane_section_info(CPlaneSectionInfo& section_info, int ap_idx, int sym_id, uint16_t start_prbc, uint16_t num_prbc, uint16_t section_id)
{
    auto& cplane_sections_info                                           = section_info.prb_info->cplane_sections_info;
    auto& cplane_section_idx                                             = cplane_sections_info[sym_id]->cplane_sections_count[ap_idx];
    cplane_sections_info[sym_id]->startPrbc[ap_idx][cplane_section_idx]  = start_prbc;
    cplane_sections_info[sym_id]->numPrbc[ap_idx][cplane_section_idx]    = num_prbc;
    cplane_sections_info[sym_id]->section_id[ap_idx][cplane_section_idx] = section_id;
    cplane_section_idx++;
}

inline void populate_section_info(uint8_t*& section_ptr, CPlaneSectionInfo& section_info, uint16_t& total_section_info_size, uint16_t& pkt_remaining_capacity, uint16_t section_hdr_size, uint16_t max_prb_x_sym, uint16_t start_prbc, uint16_t num_prbc, uint16_t section_id, int start_bundle, int num_bundles, uint8_t& sections_generated, bfw_cplane_chain_info* bfw_cplane_chain_info)
{
    if(unlikely(num_prbc == max_prb_x_sym && start_prbc))
    {
        THROW_FH(EINVAL, StringBuilder() << "Wrong start_prbc or num_prbc!");
    }
    populate_section_hdr(section_ptr, section_info, section_hdr_size, sections_generated);
    uint16_t total_ext_sz = 0;
    if(oran_cmsg_get_section_1_ef(&section_info.sect_1))
    {
        auto ext4_ptr  = static_cast<CPlaneSectionExtInfo*>(section_info.ext4);
        auto ext5_ptr  = static_cast<CPlaneSectionExtInfo*>(section_info.ext5);
        auto ext11_ptr = static_cast<CPlaneSectionExtInfo*>(section_info.ext11);

        if(ext4_ptr != nullptr)
        {
            total_ext_sz += populate_se4(section_ptr + section_hdr_size, ext4_ptr);
        }
        else if(ext5_ptr != nullptr)
        {
            total_ext_sz += populate_se5(section_ptr + section_hdr_size, ext5_ptr);
        }

        if(ext11_ptr != nullptr)
        {
            total_ext_sz += populate_se11(section_ptr + section_hdr_size + total_ext_sz, ext11_ptr, num_prbc, start_bundle, num_bundles, bfw_cplane_chain_info);
        }
    }
    auto sect_info_ptr              = reinterpret_cast<CPlaneSectionInfo*>(section_ptr);
    sect_info_ptr->sect_1.startPrbc = start_prbc;
    sect_info_ptr->sect_1.numPrbc   = (num_prbc == max_prb_x_sym ? 0 : num_prbc);
    sect_info_ptr->sect_1.sectionId = section_id;

    total_section_info_size += section_hdr_size + total_ext_sz;
    pkt_remaining_capacity -= section_hdr_size + total_ext_sz;
    section_ptr += section_hdr_size + total_ext_sz;
}

inline uint16_t Peer::prepare_cplane_message_mmimo_no_se(const CPlaneMsgSendInfo& info, rte_mbuf** mbufs, MbufArray* mbufs_regular, MbufArray* mbufs_bfw)
{
    auto fhi = get_fronthaul();
    auto flow = static_cast<Flow*>(info.flow);
    auto ap_idx = info.ap_idx;
    auto nxt_section_id = info.nxt_section_id;
    auto&    radio_app_hdr      = info.section_common_hdr.sect_1_common_hdr.radioAppHdr;
    uint8_t  section_type       = radio_app_hdr.sectionType;
    uint8_t  number_of_sections = radio_app_hdr.numberOfSections;
    uint8_t  sym_id             = radio_app_hdr.startSymbolId.get();
    uint16_t common_hdr_size    = get_cmsg_common_hdr_size(section_type);
    uint16_t section_hdr_size   = get_cmsg_section_size(section_type);

    uint16_t mtu                   = nic_->get_mtu();
    uint16_t pkt_section_info_room = mtu - ORAN_CMSG_HDR_OFFSET - common_hdr_size;

    uint8_t               section_num             = 0;
    uint16_t              packet_num              = 0;
    rte_mbuf*             m                       = nullptr;
    PacketHeaderTemplate* data                    = nullptr;
    uint8_t*              common_hdr_ptr          = nullptr;
    uint8_t*              section_ptr             = nullptr;
    uint16_t              pkt_remaining_capacity  = 0;
    uint16_t              total_section_info_size = 0;
    uint8_t               sections_generated      = 0;

    sendCPlaneTxqInfo txq_info{};
    txq_info.last_cplane_tx_ts_ = info.data_direction == DIRECTION_UPLINK ? &last_ul_cplane_tx_ts_ : &last_dl_cplane_tx_ts_;
    txq_info.mbufs_ = mbufs_regular;
    txq_info.is_bfw_send_req = false;

    init_packet(info, mbufs, packet_num, m, flow, data, common_hdr_ptr, section_ptr, common_hdr_size, pkt_section_info_room, pkt_remaining_capacity, total_section_info_size, sections_generated);

    auto  direction             = info.section_common_hdr.sect_1_common_hdr.radioAppHdr.dataDirection;
    auto& sequence_id_generator = (direction == DIRECTION_UPLINK) ? flow->get_sequence_id_generator_uplink() : flow->get_sequence_id_generator_downlink();
    while(section_num < number_of_sections) {
        {
            auto& cur_section_info = info.sections[section_num];
            if((cur_section_info.prb_info->cplane_sections_info_sym_map & (1 << sym_id)) == 0) {
                auto cplane_sections_info_list_idx = this->get_next_cplane_sections_info_list_idx();  // Added this->
                cur_section_info.prb_info->cplane_sections_info[sym_id] = &this->cplane_sections_info_list_[cplane_sections_info_list_idx];  // Added this->

                for(auto i = 0; i < slot_command_api::MAX_AP_PER_SLOT_SRS; i++) {
                    cur_section_info.prb_info->cplane_sections_info[sym_id]->cplane_sections_count[i] = 0;
                    cur_section_info.prb_info->cplane_sections_info[sym_id]->combined_reMask[i] = 0;
                }
                cur_section_info.prb_info->cplane_sections_info_sym_map |= (1 << sym_id);
            }
        }

        if(section_hdr_size > pkt_remaining_capacity) {
            add_new_packet(fhi, info, mbufs, packet_num, m, flow, sequence_id_generator,
                         data, common_hdr_ptr, section_ptr, common_hdr_size, mtu,
                         pkt_section_info_room, pkt_remaining_capacity, total_section_info_size,
                         sections_generated, txq_info);
        }

        populate_section_hdr(section_ptr, info.sections[section_num], section_hdr_size, sections_generated);
        pkt_remaining_capacity -= section_hdr_size;

        auto sect_info_ptr = reinterpret_cast<CPlaneSectionInfo*>(section_ptr);
        if(info.sections[section_num].csirs_of_multiplex_pdsch_csirs) //csirs part of pdsch+csirs
        {
            //Get section ID for all CSI-RS sections from the PDSCH section.
            auto ref_section_num = section_num - info.sections[section_num].section_id_lookback_index;
            auto cplane_section_idx         = info.sections[ref_section_num].prb_info->cplane_sections_info[sym_id]->cplane_sections_count[ap_idx];
            sect_info_ptr->sect_1.sectionId = info.sections[ref_section_num].prb_info->cplane_sections_info[sym_id]->section_id[ap_idx][cplane_section_idx - 1]; // we did cplane_section_idx - 1 because gets incremented after saving section id for PDSCH section.
        }
        else
        {
            uint16_t section_id = handle_section_id_assignment(info.sections, section_num, ap_idx, sym_id, nxt_section_id);
            sect_info_ptr->sect_1.sectionId = section_id;
            store_cplane_section_info(info.sections[section_num], ap_idx, sym_id, info.sections[section_num].sect_1.startPrbc.get(), info.sections[section_num].sect_1.numPrbc.get(), sect_info_ptr->sect_1.sectionId.get());
            if(info.sections[section_num].section_id_lookback_index == 0)
            {
                update_cplane_combined_reMask(info.sections[section_num], ap_idx, sym_id, info.sections[section_num].prb_info->common.reMask);
            }
        }

        section_ptr += section_hdr_size;
        section_num++;
    }

    populate_last_packet(fhi, info, mbufs, m, sequence_id_generator, data,
                        common_hdr_ptr, mtu, pkt_remaining_capacity,
                        sections_generated, txq_info);
    return packet_num;
}

void Peer::prepare_cplane_message_mmimo(const CPlaneMsgSendInfo& info, rte_mbuf** mbufs, rte_mbuf** chain_mbufs, MbufArray* mbufs_regular, MbufArray* mbufs_bfw, cplanePrepareInfo& cplane_prepare_info)
{
    if(info.data_direction == DIRECTION_DOWNLINK) {
        prepare_cplane_message_mmimo_dl(info, mbufs, chain_mbufs, mbufs_regular, mbufs_bfw, cplane_prepare_info);
    } else {
        cplane_prepare_info.created_pkts += prepare_cplane_message_mmimo_ul(info, mbufs, mbufs_regular, mbufs_bfw);
    }
}

uint16_t Peer::prepare_cplane_message_mmimo_ul(const CPlaneMsgSendInfo& info, rte_mbuf** mbufs, MbufArray* mbufs_regular, MbufArray* mbufs_bfw)
{
    if(!info.hasSectionExt)
    {
        return prepare_cplane_message_mmimo_no_se(info, mbufs, mbufs_regular, mbufs_bfw);
    }
    else
    {
        auto fhi            = get_fronthaul();
        auto flow           = static_cast<Flow*>(info.flow);
        auto ap_idx         = info.ap_idx;
        auto nxt_section_id = info.nxt_section_id;

        auto&    radio_app_hdr      = info.section_common_hdr.sect_1_common_hdr.radioAppHdr;
        uint8_t  section_type       = radio_app_hdr.sectionType;
        uint8_t  number_of_sections = radio_app_hdr.numberOfSections;
        uint8_t  sym_id             = radio_app_hdr.startSymbolId.get();
        uint16_t common_hdr_size    = get_cmsg_common_hdr_size(section_type);
        uint16_t section_hdr_size   = get_cmsg_section_size(section_type);

        uint16_t mtu                   = nic_->get_mtu();
        uint16_t pkt_section_info_room = mtu - ORAN_CMSG_HDR_OFFSET - common_hdr_size;

        uint8_t               section_num             = 0;
        uint16_t              packet_num              = 0;
        rte_mbuf*             m                       = nullptr;
        PacketHeaderTemplate* data                    = nullptr;
        uint8_t*              common_hdr_ptr          = nullptr;
        uint8_t*              section_ptr             = nullptr;
        uint16_t              pkt_remaining_capacity  = 0;
        uint16_t              total_section_info_size = 0;
        uint8_t               sections_generated      = 0;

        sendCPlaneTxqInfo txq_info{};
        sendCPlaneTxqInfo bfw_txq_info{};
        txq_info.last_cplane_tx_ts_ = &last_ul_cplane_tx_ts_;
        bfw_txq_info.last_cplane_tx_ts_ = &last_ul_bfw_cplane_tx_ts_;

        txq_info.mbufs_ = mbufs_regular;
        bfw_txq_info.mbufs_ = mbufs_bfw;
        txq_info.is_bfw_send_req = false;
        bfw_txq_info.is_bfw_send_req = true;

        bfw_cplane_chain_info chain_info = {
            .ap_idx = ap_idx,
            .bfw_chain_mode = BfwCplaneChainingMode::NO_CHAINING
        };

        init_packet(info, mbufs, packet_num, m, flow, data, common_hdr_ptr, section_ptr, common_hdr_size, pkt_section_info_room, pkt_remaining_capacity, total_section_info_size, sections_generated);

        auto  direction             = info.section_common_hdr.sect_1_common_hdr.radioAppHdr.dataDirection;
        auto& sequence_id_generator = (direction == DIRECTION_UPLINK) ? flow->get_sequence_id_generator_uplink() : flow->get_sequence_id_generator_downlink();

        uint8_t*                    current_ptr             = nullptr;
        oran_cmsg_sect_ext_type_11* current_ext11_ptr       = nullptr;
        uint16_t                    current_ext11_len       = 0;
        uint16_t                    curr_start_prbc         = 0;
        uint16_t                    section_start_prbc      = 0;
        uint16_t                    section_num_prbc        = 0;
        uint16_t                    section_max_prbc        = 0;
        bool last_packet_disableBFWs = true;
        for(section_num = 0; section_num < number_of_sections; section_num++)
        {
            {
                auto& cur_section_info = info.sections[section_num];
                if((cur_section_info.prb_info->cplane_sections_info_sym_map & (1 << sym_id)) == 0)
                {
                    auto cplane_sections_info_list_idx                      = get_next_cplane_sections_info_list_idx();
                    cur_section_info.prb_info->cplane_sections_info[sym_id] = &cplane_sections_info_list_[cplane_sections_info_list_idx];
                    //memset(cur_section_info.prb_info->cplane_sections_info[sym_id]->split_count, 0, sizeof(uint16_t) * slot_command_api::MAX_AP_PER_SLOT_SRS);
                    for(auto i = 0; i < slot_command_api::MAX_AP_PER_SLOT_SRS; i++)
                    {
                        cur_section_info.prb_info->cplane_sections_info[sym_id]->cplane_sections_count[i] = 0;
                        cur_section_info.prb_info->cplane_sections_info[sym_id]->combined_reMask[i] = 0;
                    }
                    cur_section_info.prb_info->cplane_sections_info_sym_map |= (1 << sym_id);
                }
            }

            CPlaneSectionExtInfo* ext11_ptr     = static_cast<CPlaneSectionExtInfo*>(info.sections[section_num].ext11);
            if(ext11_ptr != nullptr)
            {
                oran_cmsg_ext_hdr* ext11_common_hdr = nullptr;
                auto               disableBFWs      = oran_cmsg_get_ext_11_disableBFWs(&ext11_ptr->ext_11.ext_hdr);
                last_packet_disableBFWs = disableBFWs;
                uint16_t           ext_len          = ext11_ptr->ext_11.ext_hdr.extLen << 2;
                ext11_common_hdr                    = static_cast<oran_cmsg_ext_hdr*>(static_cast<void*>(&ext11_ptr->sect_ext_common_hdr));

                //bfwIQBitWidth == 9
                auto ext11_hdr_size = sizeof(oran_cmsg_ext_hdr) + sizeof(oran_cmsg_sect_ext_type_11);
                ext11_hdr_size += disableBFWs ? 0 : sizeof(oran_cmsg_sect_ext_type_11_disableBFWs_0_bfwCompHdr);
                auto bundle_hdr_size = ext11_ptr->ext_11.bundle_hdr_size;
                auto bfwIQ_size      = ext11_ptr->ext_11.bfwIQ_size;
                auto bundle_size     = bundle_hdr_size + (disableBFWs ? 0 : bfwIQ_size);
                int  num_bundles     = ext11_ptr->ext_11.numPrbBundles;
                auto num_bundle_prb  = ext11_ptr->ext_11.numBundPrb;

                int start_bundle = 0;
                int start_prbc   = info.sections[section_num].sect_1.startPrbc;
                int max_num_prbc = (info.sections[section_num].sect_1.numPrbc == 0) ? info_.max_num_prbs_per_symbol : info.sections[section_num].sect_1.numPrbc;

                bool one_bundle  = num_bundle_prb >= max_num_prbc;
                int  split_count = 0;
                while(max_num_prbc)
                {
                    if(unlikely(++split_count > kMaxSplitCount))
                    {
                        THROW_FH(EINVAL, StringBuilder() << "Error: section split_count should not be more than " << kMaxSplitCount);
                    }
                    auto min_sect_sz = section_hdr_size + ext11_hdr_size + bundle_size;
                    // Packet fragmentation
                    if(total_section_info_size + min_sect_sz > pkt_section_info_room)
                    {
                        add_new_packet(fhi, info, mbufs, packet_num, m, flow, sequence_id_generator, data, common_hdr_ptr, section_ptr, common_hdr_size, mtu, pkt_section_info_room, pkt_remaining_capacity, total_section_info_size, sections_generated,
                            last_packet_disableBFWs == 0 ? bfw_txq_info : txq_info);
                    }

                    // Start new packet always for BFW weights if chaining enabled
                    // if(total_section_info_size != 0 && disableBFWs == 0)
                    // {
                    //     add_new_packet(fhi, info, mbufs, packet_num, m, flow, sequence_id_generator, data, common_hdr_ptr, section_ptr, common_hdr_size, mtu, pkt_section_info_room, pkt_remaining_capacity, total_section_info_size, sections_generated,
                    //         disableBFWs == 0 ? bfw_txq_info : txq_info);
                    // }

                    int bundle_room         = 0;
                    int current_num_bundles = 0;
                    int num_prbc            = max_num_prbc;
                    int current_num_prbc    = max_num_prbc;

                    if(one_bundle)
                    {
                        start_bundle                 = 0;
                        current_num_bundles          = 1;
                        current_num_prbc             = std::min(max_num_prbc, ORAN_MAX_PRB_X_SECTION);
                        ext11_ptr->ext_11.numBundPrb = current_num_prbc;
                    }
                    else
                    {
                        bundle_room         = pkt_remaining_capacity - section_hdr_size - ext11_hdr_size;
                        current_num_bundles = std::min(num_bundles, static_cast<int>(bundle_room / bundle_size));
                        num_prbc            = current_num_bundles * num_bundle_prb;
                        current_num_prbc    = std::min(num_prbc, max_num_prbc);
                        if(ext11_ptr->ext_11.static_bfw)
                        {
                            current_num_prbc             = std::min(current_num_prbc, ORAN_MAX_PRB_X_SECTION);
                            ext11_ptr->ext_11.numBundPrb = current_num_prbc;
                            start_bundle                 = 0;
                        }
                    }

                    if(one_bundle && ext11_ptr != nullptr)
                    {
                        ext11_ptr->ext_11.ext_hdr.numBundPrb = current_num_prbc;
                    }

                    uint16_t section_id = nxt_section_id[ap_idx]++;
                    chain_info.header_mbuf = m;
                    populate_section_info(section_ptr, info.sections[section_num], total_section_info_size, pkt_remaining_capacity, section_hdr_size, info_.max_num_prbs_per_symbol, start_prbc, current_num_prbc, section_id, start_bundle, current_num_bundles, sections_generated, &chain_info);
                    store_cplane_section_info(info.sections[section_num], ap_idx, sym_id, start_prbc, current_num_prbc, section_id);

                    start_bundle += current_num_bundles;
                    num_bundles -= current_num_bundles;
                    start_prbc += current_num_prbc;
                    max_num_prbc -= current_num_prbc;
                }
            }
            else
            {
                auto&    cur_section_info = info.sections[section_num];
                uint16_t ext_len          = 0;
                auto     ext4_ptr         = static_cast<CPlaneSectionExtInfo*>(cur_section_info.ext4);
                auto     ext5_ptr         = static_cast<CPlaneSectionExtInfo*>(cur_section_info.ext5);
                if(ext4_ptr != nullptr)
                {
                    ext_len = sizeof(oran_cmsg_ext_hdr) + sizeof(oran_cmsg_sect_ext_type_4);
                }
                else if(ext5_ptr != nullptr)
                {
                    ext_len = sizeof(oran_cmsg_ext_hdr) + sizeof(oran_cmsg_sect_ext_type_5);
                }

                if(total_section_info_size + section_hdr_size + ext_len > pkt_section_info_room)
                {
                    add_new_packet(fhi, info, mbufs, packet_num, m, flow, sequence_id_generator, data, common_hdr_ptr, section_ptr, common_hdr_size, mtu, pkt_section_info_room, pkt_remaining_capacity, total_section_info_size, sections_generated, txq_info);
                }
                populate_section_hdr(section_ptr, info.sections[section_num], section_hdr_size, sections_generated);

                if(ext4_ptr != nullptr)
                {
                    populate_se4(section_ptr + section_hdr_size, ext4_ptr);
                }
                else if(ext5_ptr != nullptr)
                {
                    populate_se5(section_ptr + section_hdr_size, ext5_ptr);
                }

                auto  sect_info_ptr        = reinterpret_cast<CPlaneSectionInfo*>(section_ptr);
                if(info.sections[section_num].csirs_of_multiplex_pdsch_csirs) //csirs part of pdsch+csirs
                {
                    auto cplane_section_idx         = info.sections[section_num].prb_info->cplane_sections_info[sym_id]->cplane_sections_count[ap_idx];
                    sect_info_ptr->sect_1.sectionId = info.sections[section_num - 1].prb_info->cplane_sections_info[sym_id]->section_id[ap_idx][cplane_section_idx - 1];
                }
                else
                {
                    sect_info_ptr->sect_1.sectionId                                      = nxt_section_id[ap_idx]++;
                    store_cplane_section_info(info.sections[section_num], ap_idx, sym_id, info.sections[section_num].sect_1.startPrbc.get(), info.sections[section_num].sect_1.numPrbc.get(), sect_info_ptr->sect_1.sectionId.get());
                }
                pkt_remaining_capacity -= section_hdr_size + ext_len;
                total_section_info_size += section_hdr_size + ext_len;
                section_ptr += section_hdr_size + ext_len;
            }
        }

        populate_last_packet(fhi, info, mbufs, m, sequence_id_generator, data, common_hdr_ptr, mtu, pkt_remaining_capacity, sections_generated,
            last_packet_disableBFWs == 0 ? bfw_txq_info : txq_info);
        return packet_num;
    }
}

void Peer::prepare_cplane_message_mmimo_dl(const CPlaneMsgSendInfo& info, rte_mbuf** mbufs, rte_mbuf** chain_mbufs, MbufArray* mbufs_regular, MbufArray* mbufs_bfw, cplanePrepareInfo& cplane_prepare_info)
{
    if(!info.hasSectionExt)
    {
        cplane_prepare_info.created_pkts += prepare_cplane_message_mmimo_no_se(info, mbufs, mbufs_regular, mbufs_bfw);
    }
    else
    {
        auto fhi            = get_fronthaul();
        auto flow           = static_cast<Flow*>(info.flow);
        auto ap_idx         = info.ap_idx;
        auto nxt_section_id = info.nxt_section_id;

        auto&    radio_app_hdr      = info.section_common_hdr.sect_1_common_hdr.radioAppHdr;
        uint8_t  section_type       = radio_app_hdr.sectionType;
        uint8_t  number_of_sections = radio_app_hdr.numberOfSections;
        uint8_t  sym_id             = radio_app_hdr.startSymbolId.get();
        uint16_t common_hdr_size    = get_cmsg_common_hdr_size(section_type);
        uint16_t section_hdr_size   = get_cmsg_section_size(section_type);

        uint16_t mtu                   = nic_->get_mtu();
        uint16_t pkt_section_info_room = mtu - ORAN_CMSG_HDR_OFFSET - common_hdr_size;

        uint8_t               section_num             = 0;
        uint16_t              packet_num              = 0;
        rte_mbuf*             m                       = nullptr;
        PacketHeaderTemplate* data                    = nullptr;
        uint8_t*              common_hdr_ptr          = nullptr;
        uint8_t*              section_ptr             = nullptr;
        uint16_t              pkt_remaining_capacity  = 0;
        uint16_t              total_section_info_size = 0;
        uint8_t               sections_generated      = 0;

        sendCPlaneTxqInfo txq_info{};
        sendCPlaneTxqInfo bfw_txq_info{};
        txq_info.last_cplane_tx_ts_ = &last_dl_cplane_tx_ts_;
        bfw_txq_info.last_cplane_tx_ts_ = &last_dl_bfw_cplane_tx_ts_;

        txq_info.mbufs_ = mbufs_regular;
        bfw_txq_info.mbufs_ = mbufs_bfw;
        txq_info.is_bfw_send_req = false;
        bfw_txq_info.is_bfw_send_req = true;

        bfw_cplane_chain_info chain_info = {
            .chain_mbufs = chain_mbufs,
            .chained_mbufs = 0,
            .ap_idx = ap_idx,
            .bfw_chain_mode = info_.bfw_cplane_info.bfw_chain_mode,
        };

        init_packet(info, mbufs, packet_num, m, flow, data, common_hdr_ptr, section_ptr, common_hdr_size, pkt_section_info_room, pkt_remaining_capacity, total_section_info_size, sections_generated);

        auto  direction             = info.section_common_hdr.sect_1_common_hdr.radioAppHdr.dataDirection;
        auto& sequence_id_generator = (direction == DIRECTION_UPLINK) ? flow->get_sequence_id_generator_uplink() : flow->get_sequence_id_generator_downlink();
    #if 0
        for(section_num = 0; section_num < number_of_sections; section_num++)
        {
            auto& cur_section_info = info.sections[section_num];
            if(cur_section_info.prb_info == nullptr) {
                THROW_FH(EINVAL, StringBuilder() << "Error: prb_info == nullptr");
            }
            // if(cur_section_info.prb_info->cplane_sections_info[sym_id] == nullptr)
            // {
            //     auto cplane_section_idx                        = get_next_cplane_sections_info_list_idx();
            //     cur_section_info.prb_info->cplane_sections_info[sym_id] = &cplane_sections_info_list_[cplane_section_idx];
            //     //memset(cur_section_info.prb_info->cplane_sections_info[sym_id]->split_count, 0, sizeof(uint16_t) * slot_command_api::MAX_AP_PER_SLOT_SRS);
            //     for(auto i = 0; i < slot_command_api::MAX_AP_PER_SLOT_SRS; i++)
            //     {
            //         cur_section_info.prb_info->cplane_sections_info[sym_id]->cplane_sections_count[i] = 0;
            //     }
            // }
        }
    #endif

        uint8_t*                    current_ptr             = nullptr;
        oran_cmsg_sect_ext_type_11* current_ext11_ptr       = nullptr;
        uint16_t                    current_ext11_len       = 0;
        uint16_t                    curr_start_prbc         = 0;
        uint16_t                    section_start_prbc      = 0;
        uint16_t                    section_num_prbc        = 0;
        uint16_t                    section_max_prbc        = 0;
        bool last_packet_disableBFWs = true;
        bool prev_section_dyn_bfw = false;
        for(section_num = 0; section_num < number_of_sections; section_num++)
        {
            {
                auto& cur_section_info = info.sections[section_num];
                if((cur_section_info.prb_info->cplane_sections_info_sym_map & (1 << sym_id)) == 0)
                {
                    auto cplane_sections_info_list_idx                      = get_next_cplane_sections_info_list_idx();
                    cur_section_info.prb_info->cplane_sections_info[sym_id] = &cplane_sections_info_list_[cplane_sections_info_list_idx];
                    //memset(cur_section_info.prb_info->cplane_sections_info[sym_id]->split_count, 0, sizeof(uint16_t) * slot_command_api::MAX_AP_PER_SLOT_SRS);
                    for(auto i = 0; i < slot_command_api::MAX_AP_PER_SLOT_SRS; i++)
                    {
                        cur_section_info.prb_info->cplane_sections_info[sym_id]->cplane_sections_count[i] = 0;
                        cur_section_info.prb_info->cplane_sections_info[sym_id]->combined_reMask[i] = 0;
                    }
                    cur_section_info.prb_info->cplane_sections_info_sym_map |= (1 << sym_id);
                }
            }

            CPlaneSectionExtInfo* ext11_ptr     = static_cast<CPlaneSectionExtInfo*>(info.sections[section_num].ext11);
            CPlaneSectionExtInfo* ext4_ptr     = static_cast<CPlaneSectionExtInfo*>(info.sections[section_num].ext4);
            CPlaneSectionExtInfo* ext5_ptr     = static_cast<CPlaneSectionExtInfo*>(info.sections[section_num].ext5);
            CPlaneSectionExtInfo* nxt_ext11_ptr = nullptr;
            CPlaneSectionExtInfo* nxt_ext4_ptr = nullptr;
            CPlaneSectionExtInfo* nxt_ext5_ptr = nullptr;
            if(section_num + 1 < number_of_sections)
            {
                nxt_ext11_ptr = static_cast<CPlaneSectionExtInfo*>(info.sections[section_num + 1].ext11);
                nxt_ext4_ptr = static_cast<CPlaneSectionExtInfo*>(info.sections[section_num + 1].ext4);
                nxt_ext5_ptr = static_cast<CPlaneSectionExtInfo*>(info.sections[section_num + 1].ext5);
            }

            //Handle section split with pdsch+csi_rs
            // This if block handles the PDSCH+CSIRS over lap case. We expect here that PDSCH section will always be followed by CSIRS section for the overlap.
            // ext11_ptr is the current PDSCH section ext11 pointer.
            // nxt_ext11_ptr is the next CSIRS section ext11 pointer.
            if(section_num + 1 < number_of_sections && info.sections[section_num + 1].csirs_of_multiplex_pdsch_csirs && (ext11_ptr != nullptr || nxt_ext11_ptr != nullptr) &&
            //Added below check because in case of csi-rs compact signalling, 
            //multiple CSIRS sections can be present for the same PDSCH section.
            //In such cases all csirs section reference their section id from the PDSCH section.
            info.sections[section_num].section_id_lookback_index == 0) 
            {
                int pdsch_sect_sz         = section_hdr_size;
                int pdsch_ext11_hdr_size  = 0;
                int pdsch_ext4_hdr_size  = 0;
                int pdsch_ext5_hdr_size  = 0;
                int pdsch_bundle_hdr_size = 0;
                int pdsch_bfwIQ_size      = 0;
                int pdsch_bundle_size     = 0;
                int pdsch_num_bundles     = 0;
                int pdsch_num_bundle_prb  = 0;
                int csirs_sect_sz         = section_hdr_size;
                int csirs_num_bundles     = 0;
                int csirs_num_bundle_prb  = 0;

                bool                  pdsch_bfw       = false;
                CPlaneSectionExtInfo* pdsch_ext11_ptr = nullptr;
                if(ext11_ptr != nullptr)
                {
                    pdsch_ext11_ptr                     = static_cast<CPlaneSectionExtInfo*>(info.sections[section_num].ext11);
                    oran_cmsg_ext_hdr* ext11_common_hdr = nullptr;
                    pdsch_sect_sz += pdsch_ext11_ptr->ext_11.ext_hdr.extLen << 2;
                    pdsch_num_bundles     = pdsch_ext11_ptr->ext_11.numPrbBundles;
                    bool pdschDisableBFWs = oran_cmsg_get_ext_11_disableBFWs(&pdsch_ext11_ptr->ext_11.ext_hdr);
                    last_packet_disableBFWs = pdschDisableBFWs;
                    pdsch_num_bundle_prb  = pdsch_ext11_ptr->ext_11.numBundPrb;

                    // Assuming bfwIQBitWidth == 9
                    pdsch_ext11_hdr_size = sizeof(oran_cmsg_ext_hdr) + sizeof(oran_cmsg_sect_ext_type_11);
                    pdsch_ext11_hdr_size += pdschDisableBFWs ? 0 : sizeof(oran_cmsg_sect_ext_type_11_disableBFWs_0_bfwCompHdr);
                    pdsch_bundle_hdr_size = pdsch_ext11_ptr->ext_11.bundle_hdr_size;
                    pdsch_bfwIQ_size      = pdsch_ext11_ptr->ext_11.bfwIQ_size;
                    pdsch_bundle_size     = pdsch_bundle_hdr_size + (pdschDisableBFWs ? 0 : pdsch_bfwIQ_size);
                    pdsch_bfw             = true;
                }
                if(ext4_ptr != nullptr)
                {
                    pdsch_ext4_hdr_size = sizeof(oran_cmsg_ext_hdr) + sizeof(oran_cmsg_sect_ext_type_4);
                }
                if(ext5_ptr != nullptr)
                {
                    pdsch_ext5_hdr_size = sizeof(oran_cmsg_ext_hdr) + sizeof(oran_cmsg_sect_ext_type_5);
                }
                int pdsch_start_bundle = 0;
                int pdsch_start_prbc   = info.sections[section_num].sect_1.startPrbc;
                int pdsch_max_num_prbc = (info.sections[section_num].sect_1.numPrbc == 0) ? info_.max_num_prbs_per_symbol : info.sections[section_num].sect_1.numPrbc;

                bool pdsch_one_bundle = pdsch_num_bundle_prb >= pdsch_max_num_prbc;
                int  split_count      = 0;
                bool                  csirs_bfw       = false;
                auto csirs_section_index = section_num + 1;
                CPlaneSectionExtInfo* csirs_ext11_ptr = nullptr;
                while(info.sections[csirs_section_index].csirs_of_multiplex_pdsch_csirs && info.sections[csirs_section_index].section_id_lookback_index > 0)
                {
                    if(nxt_ext11_ptr != nullptr)
                    {
                        csirs_ext11_ptr                     = static_cast<CPlaneSectionExtInfo*>(info.sections[csirs_section_index].ext11);
                        oran_cmsg_ext_hdr* ext11_common_hdr = nullptr;
                        csirs_sect_sz += csirs_ext11_ptr->ext_11.ext_hdr.extLen << 2;
                        csirs_bfw            = csirs_ext11_ptr->ext_11.static_bfw;
                        csirs_num_bundles    = csirs_ext11_ptr->ext_11.numPrbBundles;
                        csirs_num_bundle_prb = csirs_ext11_ptr->ext_11.numBundPrb;

                    }
                    if(nxt_ext4_ptr != nullptr)
                    {
                        csirs_sect_sz += sizeof(oran_cmsg_ext_hdr) + sizeof(oran_cmsg_sect_ext_type_4);
                    }
                    if(nxt_ext5_ptr != nullptr)
                    {
                        csirs_sect_sz += sizeof(oran_cmsg_ext_hdr) + sizeof(oran_cmsg_sect_ext_type_5);
                    }
                    if(++csirs_section_index < number_of_sections)
                    {
                        nxt_ext11_ptr = static_cast<CPlaneSectionExtInfo*>(info.sections[csirs_section_index].ext11);
                        nxt_ext4_ptr = static_cast<CPlaneSectionExtInfo*>(info.sections[csirs_section_index].ext4);
                        nxt_ext5_ptr = static_cast<CPlaneSectionExtInfo*>(info.sections[csirs_section_index].ext5);
                    }
                    else
                    {
                        break;
                    }
                }                
                bool csirs_one_bundle = csirs_num_bundle_prb >= pdsch_max_num_prbc;
                while(pdsch_max_num_prbc)
                {
                    if(unlikely(++split_count > kMaxSplitCount))
                    {
                        THROW_FH(EINVAL, StringBuilder() << "Error: pdsch + csirs split_count should not be more than " << kMaxSplitCount);
                    }
                    auto min_pdsch_sect_sz = section_hdr_size + pdsch_ext4_hdr_size + pdsch_ext5_hdr_size + pdsch_ext11_hdr_size + pdsch_bundle_size;

                    // Start new packet always for BFW weights if chaining enabled
                    bool start_new_dyn_bfw_packet = prev_section_dyn_bfw || (last_packet_disableBFWs == 0 && pdsch_ext11_ptr->ext_11.static_bfw == false);
                    start_new_dyn_bfw_packet = start_new_dyn_bfw_packet && (info_.bfw_cplane_info.bfw_chain_mode != BfwCplaneChainingMode::NO_CHAINING);

                    if(total_section_info_size != 0 && start_new_dyn_bfw_packet)
                    {
                        add_new_packet(fhi, info, mbufs, packet_num, m, flow, sequence_id_generator, data, common_hdr_ptr, section_ptr, common_hdr_size, mtu, pkt_section_info_room, pkt_remaining_capacity, total_section_info_size, sections_generated,
                            last_packet_disableBFWs == 0 ? bfw_txq_info : txq_info);
                    }
                    else if(total_section_info_size + min_pdsch_sect_sz + csirs_sect_sz > pkt_section_info_room)
                    {
                        add_new_packet(fhi, info, mbufs, packet_num, m, flow, sequence_id_generator, data, common_hdr_ptr, section_ptr, common_hdr_size, mtu, pkt_section_info_room, pkt_remaining_capacity, total_section_info_size, sections_generated,
                            last_packet_disableBFWs == 0 ? bfw_txq_info : txq_info);
                    }
                    prev_section_dyn_bfw = (ext11_ptr != nullptr && last_packet_disableBFWs == 0 && pdsch_ext11_ptr->ext_11.static_bfw == false);
                    int pdsch_bundle_room         = 0;
                    int current_pdsch_num_bundles = 0;
                    int pdsch_num_prbc            = pdsch_max_num_prbc;
                    int current_pdsch_num_prbc    = pdsch_max_num_prbc;

                    if(pdsch_one_bundle)
                    {
                        pdsch_start_bundle        = 0;
                        current_pdsch_num_bundles = 1;
                        current_pdsch_num_prbc    = std::min(current_pdsch_num_prbc, ORAN_MAX_PRB_X_SECTION);
                    }
                    else if(pdsch_bfw)
                    {
                        pdsch_bundle_room = pkt_remaining_capacity - section_hdr_size - pdsch_ext11_hdr_size - csirs_sect_sz;

                        current_pdsch_num_bundles = std::min(pdsch_num_bundles, static_cast<int>(pdsch_bundle_room / pdsch_bundle_size));
                        pdsch_num_prbc            = current_pdsch_num_bundles * pdsch_num_bundle_prb;
                        current_pdsch_num_prbc    = std::min(pdsch_num_prbc, pdsch_max_num_prbc);
                        if(csirs_bfw && current_pdsch_num_prbc > ORAN_MAX_PRB_X_SECTION)
                        {
                            current_pdsch_num_prbc    = std::min(current_pdsch_num_prbc, ORAN_MAX_PRB_X_SECTION);
                            current_pdsch_num_bundles = current_pdsch_num_prbc / pdsch_num_bundle_prb;
                            current_pdsch_num_prbc    = current_pdsch_num_bundles * pdsch_num_bundle_prb;
                        }
                    }
                    else if(csirs_bfw)
                    {
                        current_pdsch_num_prbc = std::min(current_pdsch_num_prbc, ORAN_MAX_PRB_X_SECTION);
                    }

                    if(pdsch_one_bundle && pdsch_ext11_ptr != nullptr)
                    {
                        pdsch_ext11_ptr->ext_11.ext_hdr.numBundPrb = current_pdsch_num_prbc;
                    }


                    uint16_t section_id = nxt_section_id[ap_idx]++;

                    //csirs section
                    //FIXME PDSCH with dyn BFW needs to start new packet for CSIRS
                    auto csirs_section_index = section_num + 1;
                    while(info.sections[csirs_section_index].csirs_of_multiplex_pdsch_csirs && info.sections[csirs_section_index].section_id_lookback_index > 0)
                    {
                        auto csirs_ext11_ptr = static_cast<CPlaneSectionExtInfo*>(info.sections[csirs_section_index].ext11);
                        if(csirs_one_bundle && csirs_ext11_ptr != nullptr)
                        {
                            csirs_ext11_ptr->ext_11.ext_hdr.numBundPrb = current_pdsch_num_prbc;
                        }
                        populate_section_info(section_ptr, info.sections[csirs_section_index], total_section_info_size, pkt_remaining_capacity, section_hdr_size, info_.max_num_prbs_per_symbol, pdsch_start_prbc, current_pdsch_num_prbc, section_id, 0, csirs_num_bundles, sections_generated, &chain_info);
                        if(++csirs_section_index >= number_of_sections)
                        {
                            break;
                        }
                    }
                    //pdsch section
                    chain_info.header_mbuf = m;
                    populate_section_info(section_ptr, info.sections[section_num], total_section_info_size, pkt_remaining_capacity, section_hdr_size, info_.max_num_prbs_per_symbol, pdsch_start_prbc, current_pdsch_num_prbc, section_id, pdsch_start_bundle, current_pdsch_num_bundles, sections_generated, &chain_info);

                    store_cplane_section_info(info.sections[section_num], ap_idx, sym_id, pdsch_start_prbc, current_pdsch_num_prbc, section_id);

                    pdsch_start_bundle += current_pdsch_num_bundles;
                    pdsch_num_bundles -= current_pdsch_num_bundles;
                    pdsch_start_prbc += current_pdsch_num_prbc;
                    pdsch_max_num_prbc -= current_pdsch_num_prbc;
                }
                section_num = (csirs_section_index - 1);
            }
            else if(ext11_ptr != nullptr)
            {
                oran_cmsg_ext_hdr* ext11_common_hdr = nullptr;
                auto               disableBFWs      = oran_cmsg_get_ext_11_disableBFWs(&ext11_ptr->ext_11.ext_hdr);
                last_packet_disableBFWs = disableBFWs;
                uint16_t           ext_len          = ext11_ptr->ext_11.ext_hdr.extLen << 2;
                ext11_common_hdr                    = static_cast<oran_cmsg_ext_hdr*>(static_cast<void*>(&ext11_ptr->sect_ext_common_hdr));

                //bfwIQBitWidth == 9
                auto ext11_hdr_size = sizeof(oran_cmsg_ext_hdr) + sizeof(oran_cmsg_sect_ext_type_11);
                ext11_hdr_size += disableBFWs ? 0 : sizeof(oran_cmsg_sect_ext_type_11_disableBFWs_0_bfwCompHdr);
                auto bundle_hdr_size = ext11_ptr->ext_11.bundle_hdr_size;
                auto bfwIQ_size      = ext11_ptr->ext_11.bfwIQ_size;
                auto bundle_size     = bundle_hdr_size + (disableBFWs ? 0 : bfwIQ_size);
                int  num_bundles     = ext11_ptr->ext_11.numPrbBundles;
                auto num_bundle_prb  = ext11_ptr->ext_11.numBundPrb;

                int start_bundle = 0;
                int start_prbc   = info.sections[section_num].sect_1.startPrbc;
                int max_num_prbc = (info.sections[section_num].sect_1.numPrbc == 0) ? info_.max_num_prbs_per_symbol : info.sections[section_num].sect_1.numPrbc;

                bool one_bundle  = num_bundle_prb >= max_num_prbc;
                int  split_count = 0;
                while(max_num_prbc)
                {
                    if(unlikely(++split_count > kMaxSplitCount))
                    {
                        THROW_FH(EINVAL, StringBuilder() << "Error: section split_count should not be more than " << kMaxSplitCount);
                    }
                    auto min_sect_sz = section_hdr_size + ext11_hdr_size + bundle_size;

                    // Start new packet always for BFW weights if chaining enabled
                    bool start_new_dyn_bfw_packet = prev_section_dyn_bfw || (ext11_ptr != nullptr && disableBFWs == 0 && ext11_ptr->ext_11.static_bfw == false);
                    start_new_dyn_bfw_packet = start_new_dyn_bfw_packet && (info_.bfw_cplane_info.bfw_chain_mode != BfwCplaneChainingMode::NO_CHAINING);

                    if(total_section_info_size != 0 && start_new_dyn_bfw_packet)
                    {
                        add_new_packet(fhi, info, mbufs, packet_num, m, flow, sequence_id_generator, data, common_hdr_ptr, section_ptr, common_hdr_size, mtu, pkt_section_info_room, pkt_remaining_capacity, total_section_info_size, sections_generated,
                            last_packet_disableBFWs == 0 ? bfw_txq_info : txq_info);
                    }
                    // Packet fragmentation
                    else if(total_section_info_size + min_sect_sz > pkt_section_info_room)
                    {
                        add_new_packet(fhi, info, mbufs, packet_num, m, flow, sequence_id_generator, data, common_hdr_ptr, section_ptr, common_hdr_size, mtu, pkt_section_info_room, pkt_remaining_capacity, total_section_info_size, sections_generated,
                            last_packet_disableBFWs == 0 ? bfw_txq_info : txq_info);
                    }
                    prev_section_dyn_bfw = (ext11_ptr != nullptr && disableBFWs == 0 && ext11_ptr->ext_11.static_bfw == false);

                    int bundle_room         = 0;
                    int current_num_bundles = 0;
                    int num_prbc            = max_num_prbc;
                    int current_num_prbc    = max_num_prbc;

                    if(one_bundle)
                    {
                        start_bundle                 = 0;
                        current_num_bundles          = 1;
                        current_num_prbc             = std::min(max_num_prbc, ORAN_MAX_PRB_X_SECTION);
                        ext11_ptr->ext_11.numBundPrb = current_num_prbc;
                    }
                    else
                    {
                        bundle_room         = pkt_remaining_capacity - section_hdr_size - ext11_hdr_size;
                        current_num_bundles = std::min(num_bundles, static_cast<int>(bundle_room / bundle_size));
                        num_prbc            = current_num_bundles * num_bundle_prb;
                        current_num_prbc    = std::min(num_prbc, max_num_prbc);
                        if(ext11_ptr->ext_11.static_bfw)
                        {
                            current_num_prbc             = std::min(current_num_prbc, ORAN_MAX_PRB_X_SECTION);
                            ext11_ptr->ext_11.numBundPrb = current_num_prbc;
                            start_bundle                 = 0;
                        }
                    }

                    if(one_bundle && ext11_ptr != nullptr)
                    {
                        ext11_ptr->ext_11.ext_hdr.numBundPrb = current_num_prbc;
                    }

                    uint16_t section_id = handle_section_id_assignment(info.sections, section_num, ap_idx, sym_id, nxt_section_id);
                    chain_info.header_mbuf = m;
                    populate_section_info(section_ptr, info.sections[section_num], total_section_info_size, pkt_remaining_capacity, section_hdr_size, info_.max_num_prbs_per_symbol, start_prbc, current_num_prbc, section_id, start_bundle, current_num_bundles, sections_generated, &chain_info);
                    store_cplane_section_info(info.sections[section_num], ap_idx, sym_id, start_prbc, current_num_prbc, section_id);
                    if(info.sections[section_num].section_id_lookback_index == 0)
                    {
                        update_cplane_combined_reMask(info.sections[section_num], ap_idx, sym_id, info.sections[section_num].prb_info->common.reMask);
                    }
                    start_bundle += current_num_bundles;
                    num_bundles -= current_num_bundles;
                    start_prbc += current_num_prbc;
                    max_num_prbc -= current_num_prbc;
                }
            }
            else
            {
                auto&    cur_section_info = info.sections[section_num];
                uint16_t ext_len          = 0;
                auto     ext4_ptr         = static_cast<CPlaneSectionExtInfo*>(cur_section_info.ext4);
                auto     ext5_ptr         = static_cast<CPlaneSectionExtInfo*>(cur_section_info.ext5);
                if(ext4_ptr != nullptr)
                {
                    ext_len = sizeof(oran_cmsg_ext_hdr) + sizeof(oran_cmsg_sect_ext_type_4);
                }
                else if(ext5_ptr != nullptr)
                {
                    ext_len = sizeof(oran_cmsg_ext_hdr) + sizeof(oran_cmsg_sect_ext_type_5);
                }

                if(total_section_info_size + section_hdr_size + ext_len > pkt_section_info_room)
                {
                    add_new_packet(fhi, info, mbufs, packet_num, m, flow, sequence_id_generator, data, common_hdr_ptr, section_ptr, common_hdr_size, mtu, pkt_section_info_room, pkt_remaining_capacity, total_section_info_size, sections_generated, txq_info);
                }
                populate_section_hdr(section_ptr, info.sections[section_num], section_hdr_size, sections_generated);

                if(ext4_ptr != nullptr)
                {
                    populate_se4(section_ptr + section_hdr_size, ext4_ptr);
                }
                else if(ext5_ptr != nullptr)
                {
                    populate_se5(section_ptr + section_hdr_size, ext5_ptr);
                }

                auto  sect_info_ptr        = reinterpret_cast<CPlaneSectionInfo*>(section_ptr);
                if(info.sections[section_num].csirs_of_multiplex_pdsch_csirs) //csirs part of pdsch+csirs
                {
                    //Get section ID for all CSI-RS sections from the PDSCH section.
                    auto ref_section_num = section_num - info.sections[section_num].section_id_lookback_index;
                    auto cplane_section_idx         = info.sections[ref_section_num].prb_info->cplane_sections_info[sym_id]->cplane_sections_count[ap_idx];
                    sect_info_ptr->sect_1.sectionId = info.sections[ref_section_num].prb_info->cplane_sections_info[sym_id]->section_id[ap_idx][cplane_section_idx - 1]; // we did cplane_section_idx - 1 because gets incremented after saving section id for PDSCH section.
                }
                else
                {
                    uint16_t section_id = handle_section_id_assignment(info.sections, section_num, ap_idx, sym_id, nxt_section_id);
                    sect_info_ptr->sect_1.sectionId                                      = section_id;
                    store_cplane_section_info(info.sections[section_num], ap_idx, sym_id, info.sections[section_num].sect_1.startPrbc.get(), info.sections[section_num].sect_1.numPrbc.get(), sect_info_ptr->sect_1.sectionId.get());
                    //Todo: Need to figure out how to run this code only for CSI-RS. It will not to do any functional harm but is unnecessary overhead.
                    //for other channels.
                    if(info.sections[section_num].section_id_lookback_index == 0)
                    {
                        update_cplane_combined_reMask(info.sections[section_num], ap_idx, sym_id, info.sections[section_num].prb_info->common.reMask);
                    }
                }
                pkt_remaining_capacity -= section_hdr_size + ext_len;
                total_section_info_size += section_hdr_size + ext_len;
                section_ptr += section_hdr_size + ext_len;
            }
        }

        populate_last_packet(fhi, info, mbufs, m, sequence_id_generator, data, common_hdr_ptr, mtu, pkt_remaining_capacity, sections_generated,
            last_packet_disableBFWs == 0 ? bfw_txq_info : txq_info);
        cplane_prepare_info.created_pkts += packet_num;
        cplane_prepare_info.chained_mbufs += chain_info.chained_mbufs;
    }
}

inline void validate_section_type(uint8_t section_type) {
    if(unlikely(section_type > ORAN_CMSG_SECTION_TYPE_5)) {
        THROW_FH(EINVAL, StringBuilder() << "Section type " << (int)section_type << " is not supported");
    }
}

inline void validate_common_header_size(uint16_t common_hdr_size) {
    if(unlikely(common_hdr_size == 0)) {
        THROW_FH(EINVAL, StringBuilder() << "Section type is not supported");
    }
}

inline void validate_section_size(uint16_t section_size) {
    if(unlikely(section_size == 0)) {
        THROW_FH(EINVAL, StringBuilder() << "Section size is not supported");
    }
}

inline size_t count_cplane_packets_mmimo_no_section_ext(const CPlaneMsgSendInfo& info, Nic* nic) {
    auto& radio_app_hdr = info.section_common_hdr.sect_1_common_hdr.radioAppHdr;
    auto  section_type  = radio_app_hdr.sectionType;

    validate_section_type(section_type);

    auto number_of_sections = radio_app_hdr.numberOfSections;
    auto common_hdr_size    = get_cmsg_common_hdr_size(section_type);
    validate_common_header_size(common_hdr_size);

    auto total_section_info_size = get_cmsg_section_size(section_type) * number_of_sections;
    auto pkt_section_info_room   = nic->get_mtu() - ORAN_CMSG_HDR_OFFSET - common_hdr_size;

    return std::max(1UL, (total_section_info_size + pkt_section_info_room - 1) / pkt_section_info_room);
}

inline void count_cplane_packets_mmimo_ext4_ext5(const CPlaneSectionInfo& cur_section_info, uint16_t section_hdr_size, uint16_t pkt_section_info_room,
                                         uint16_t& total_section_info_size, size_t& section_num_packets) {
    uint16_t ext_len = 0;
    auto ext4_ptr = static_cast<CPlaneSectionExtInfo*>(cur_section_info.ext4);
    auto ext5_ptr = static_cast<CPlaneSectionExtInfo*>(cur_section_info.ext5);
    if(ext4_ptr != nullptr)
    {
        ext_len = sizeof(oran_cmsg_ext_hdr) + sizeof(oran_cmsg_sect_ext_type_4);
    }
    else if(ext5_ptr != nullptr)
    {
        ext_len = sizeof(oran_cmsg_ext_hdr) + sizeof(oran_cmsg_sect_ext_type_5);
    }

    if(total_section_info_size + section_hdr_size + ext_len > pkt_section_info_room)
    { //do the split
        ++section_num_packets;
        total_section_info_size = 0;
    }
    total_section_info_size += section_hdr_size + ext_len;
}

void Peer::count_cplane_packets_mmimo_ext11(
    uint16_t section_hdr_size,
    uint16_t pkt_section_info_room,
    uint16_t& total_section_info_size,
    size_t& section_num_packets,
    size_t& section_bfw_mbufs,
    size_t& section_padding_mbufs,
    int& start_prbc,
    int& max_num_prbc,
    int& num_bundles,
    CPlaneSectionExtInfo* ext11_ptr,
    const uint8_t direction,
    bool& prev_section_dyn_bfw
) {
    auto disableBFWs = oran_cmsg_get_ext_11_disableBFWs(&ext11_ptr->ext_11.ext_hdr);
    auto ext11_hdr_size = sizeof(oran_cmsg_ext_hdr) + sizeof(oran_cmsg_sect_ext_type_11);
    ext11_hdr_size += disableBFWs ? 0 : sizeof(oran_cmsg_sect_ext_type_11_disableBFWs_0_bfwCompHdr);
    auto bundle_hdr_size = ext11_ptr->ext_11.bundle_hdr_size;
    auto bfwIQ_size = ext11_ptr->ext_11.bfwIQ_size;
    auto bundle_size = bundle_hdr_size + (disableBFWs ? 0 : bfwIQ_size);
    auto num_bundle_prb = ext11_ptr->ext_11.numBundPrb;

    bool one_bundle = num_bundle_prb >= max_num_prbc;
    int split_count = 0;

    while(max_num_prbc) {
        if(unlikely(++split_count > kMaxSplitCount)) {
            THROW_FH(EINVAL, StringBuilder() << "Error: section split_count should not be more than " << kMaxSplitCount);
        }

        auto min_sect_sz = section_hdr_size + ext11_hdr_size + bundle_size;
        bool start_new_dyn_bfw_packet = prev_section_dyn_bfw || (disableBFWs == 0 && ext11_ptr->ext_11.static_bfw == false);
        start_new_dyn_bfw_packet = start_new_dyn_bfw_packet && (info_.bfw_cplane_info.bfw_chain_mode != BfwCplaneChainingMode::NO_CHAINING);
        if(direction == DIRECTION_DOWNLINK && (total_section_info_size != 0 && start_new_dyn_bfw_packet)) {
            // Start new packet always if last section has dyn BFW weights if chaining enabled with DL
            // TODO enable UL chaining
            ++section_num_packets;
            ++section_bfw_mbufs;
            total_section_info_size = 0;
        }
        else if(total_section_info_size + min_sect_sz > pkt_section_info_room) {
            ++section_num_packets;
            total_section_info_size = 0;
        }
        prev_section_dyn_bfw = (disableBFWs == 0 && ext11_ptr->ext_11.static_bfw == false);

        int bundle_room = 0;
        int current_num_bundles = 0;
        int num_prbc = max_num_prbc;
        int current_num_prbc = max_num_prbc;

        if(one_bundle) {
            current_num_bundles = 1;
            current_num_prbc = std::min(current_num_prbc, ORAN_MAX_PRB_X_SECTION);
            if(disableBFWs == 0 && ext11_ptr->ext_11.static_bfw == 0)
            {
                auto padding = oran_cmsg_se11_disableBFWs_0_padding_bytes(ext11_hdr_size + bundle_size * current_num_bundles);
                if(padding > 0)
                {
                    ++section_padding_mbufs;
                }
            }
        } else {
            bundle_room = pkt_section_info_room - total_section_info_size - section_hdr_size - ext11_hdr_size;
            current_num_bundles = ext11_ptr->ext_11.static_bfw ? 1 : std::min(num_bundles, static_cast<int>(bundle_room / bundle_size));
            num_prbc = current_num_bundles * num_bundle_prb;
            current_num_prbc = std::min(num_prbc, max_num_prbc);
            if(ext11_ptr->ext_11.static_bfw) {
                current_num_prbc = std::min(current_num_prbc, ORAN_MAX_PRB_X_SECTION);
            }

            if(disableBFWs == 0 && ext11_ptr->ext_11.static_bfw == 0)
            {
                auto padding = oran_cmsg_se11_disableBFWs_0_padding_bytes(ext11_hdr_size + bundle_size * current_num_bundles);
                if(padding > 0)
                {
                    ++section_padding_mbufs;
                }
            }
        }

        total_section_info_size += section_hdr_size + ext11_hdr_size + bundle_size * current_num_bundles;

        num_bundles -= current_num_bundles;
        start_prbc += current_num_prbc;
        max_num_prbc -= current_num_prbc;
    }
}

cplaneCountInfo Peer::count_cplane_packets_mmimo(CPlaneMsgSendInfo const* infos, size_t num_msgs, size_t max_num_packets)
{
    cplaneCountInfo cplane_count_info{.num_packets = 0, .num_bfw_mbufs = 0, .num_bfw_padding_mbufs = 0};

    if(num_msgs == 0)
    {
        return cplane_count_info;
    }

    // All messages are of the same direction by design
    if(infos[0].data_direction == DIRECTION_DOWNLINK)
    {
        cplane_count_info = count_cplane_packets_mmimo_dl(infos, num_msgs);
    }
    else
    {
        cplane_count_info.num_packets = count_cplane_packets_mmimo_ul(infos, num_msgs);
    }

    if(unlikely(cplane_count_info.num_packets > max_num_packets))
    {
        THROW_FH(EINVAL, StringBuilder() << "Too many C-plane packets to send: " << cplane_count_info.num_packets << ". Please raise kTxPktBurstCplane " << kTxPktBurstCplane << " !");
    }

    return cplane_count_info;
}

size_t Peer::count_cplane_packets_mmimo_ul(CPlaneMsgSendInfo const* infos, size_t num_msgs) {
    size_t num_packets = 0;
    size_t section_bfw_mbufs = 0;
    size_t section_padding_mbufs = 0;
    bool prev_section_dyn_bfw = false;
    for(size_t i = 0; i < num_msgs; ++i)
    {
        auto const& info = infos[i];

        if(!info.hasSectionExt)
        {
            num_packets += count_cplane_packets_mmimo_no_section_ext(info, nic_);
        }
        else
        {
            size_t section_num_packets = 1;
            auto&  radio_app_hdr       = info.section_common_hdr.sect_1_common_hdr.radioAppHdr;
            auto   section_type        = radio_app_hdr.sectionType;
            validate_section_type(section_type);
            auto number_of_sections = radio_app_hdr.numberOfSections;
            auto common_hdr_size    = get_cmsg_common_hdr_size(section_type);
            validate_common_header_size(common_hdr_size);
            auto section_hdr_size   = get_cmsg_section_size(section_type);
            validate_section_size(section_hdr_size);

            uint16_t total_section_info_size = 0;
            auto     pkt_section_info_room   = nic_->get_mtu() - ORAN_CMSG_HDR_OFFSET - common_hdr_size;

            if(unlikely(pkt_section_info_room < section_hdr_size))
            {
                THROW_FH(EINVAL, StringBuilder() << "MTU " << nic_->get_mtu() << " is too small for " << section_type << " section header size " << section_hdr_size);
            }

            for(uint8_t section_num = 0; section_num < number_of_sections; ++section_num)
            {
                if(info.sections[section_num].ext11 != nullptr)
                {
                    auto ext11_ptr = static_cast<CPlaneSectionExtInfo*>(info.sections[section_num].ext11);
                    int start_prbc = info.sections[section_num].sect_1.startPrbc;
                    int max_num_prbc = (info.sections[section_num].sect_1.numPrbc == 0) ? info_.max_num_prbs_per_symbol : info.sections[section_num].sect_1.numPrbc;
                    int num_bundles = ext11_ptr->ext_11.numPrbBundles;

                    count_cplane_packets_mmimo_ext11(
                        section_hdr_size,
                        pkt_section_info_room,
                        total_section_info_size,
                        section_num_packets,
                        section_bfw_mbufs,
                        section_padding_mbufs,
                        start_prbc,
                        max_num_prbc,
                        num_bundles,
                        ext11_ptr,
                        DIRECTION_UPLINK,
                        prev_section_dyn_bfw
                    );
                }
            }
            num_packets += std::max(1UL, section_num_packets);
        }
    }
    return num_packets;
}

cplaneCountInfo Peer::count_cplane_packets_mmimo_dl(CPlaneMsgSendInfo const* infos, size_t num_msgs)
{
    cplaneCountInfo cplane_count_info{.num_packets = 0, .num_bfw_mbufs = 0, .num_bfw_padding_mbufs = 0};
    bool prev_section_disableBFW0 = false;
    for(size_t i = 0; i < num_msgs; ++i)
    {
        auto const& info = infos[i];

        if(!info.hasSectionExt)
        {
            cplane_count_info.num_packets += count_cplane_packets_mmimo_no_section_ext(info, nic_);
        }
        else
        {
            size_t section_num_packets = 1;
            size_t section_bfw_mbufs = 0;
            size_t section_padding_mbufs = 0;
            bool prev_section_dyn_bfw = false;
            if(info.sections[0].ext11 != nullptr)
            {
                if(info.sections[0].ext11->ext_11.static_bfw == 0 && oran_cmsg_get_ext_11_disableBFWs(&info.sections[0].ext11->ext_11.ext_hdr) == 0)
                {
                    section_bfw_mbufs = 1;
                }
            }
            auto&  radio_app_hdr       = info.section_common_hdr.sect_1_common_hdr.radioAppHdr;
            auto   section_type        = radio_app_hdr.sectionType;
            validate_section_type(section_type);
            auto number_of_sections = radio_app_hdr.numberOfSections;
            auto common_hdr_size    = get_cmsg_common_hdr_size(section_type);
            validate_common_header_size(common_hdr_size);
            auto section_hdr_size   = get_cmsg_section_size(section_type);
            validate_section_size(section_hdr_size);

            uint16_t total_section_info_size = 0;
            auto     pkt_section_info_room   = nic_->get_mtu() - ORAN_CMSG_HDR_OFFSET - common_hdr_size;

            if(unlikely(pkt_section_info_room < section_hdr_size))
            {
                THROW_FH(EINVAL, StringBuilder() << "MTU " << nic_->get_mtu() << " is too small for " << section_type << " section header size " << section_hdr_size);
            }

            for(uint8_t section_num = 0; section_num < number_of_sections; section_num++)
            {
                CPlaneSectionExtInfo* ext11_ptr     = static_cast<CPlaneSectionExtInfo*>(info.sections[section_num].ext11);
                CPlaneSectionExtInfo* ext4_ptr     = static_cast<CPlaneSectionExtInfo*>(info.sections[section_num].ext4);
                CPlaneSectionExtInfo* ext5_ptr     = static_cast<CPlaneSectionExtInfo*>(info.sections[section_num].ext5);
                CPlaneSectionExtInfo* nxt_ext11_ptr = nullptr;
                CPlaneSectionExtInfo* nxt_ext4_ptr = nullptr;
                CPlaneSectionExtInfo* nxt_ext5_ptr = nullptr;
                if(section_num + 1 < number_of_sections)
                {
                    nxt_ext11_ptr = static_cast<CPlaneSectionExtInfo*>(info.sections[section_num + 1].ext11);
                    nxt_ext4_ptr = static_cast<CPlaneSectionExtInfo*>(info.sections[section_num + 1].ext4);
                    nxt_ext5_ptr = static_cast<CPlaneSectionExtInfo*>(info.sections[section_num + 1].ext5);
                }
                auto csirs_section_index = section_num + 1;
                //Handle section split with pdsch+csi_rs
                if(section_num + 1 < number_of_sections && info.sections[section_num + 1].csirs_of_multiplex_pdsch_csirs && (info.sections[section_num].section_id_lookback_index== 0) && (ext11_ptr != nullptr || nxt_ext11_ptr != nullptr))
                {
                    int pdsch_sect_sz         = section_hdr_size;
                    int pdsch_ext11_hdr_size  = 0;
                    int pdsch_ext4_hdr_size  = 0;
                    int pdsch_ext5_hdr_size  = 0;
                    int pdsch_bundle_hdr_size = 0;
                    int pdsch_bfwIQ_size      = 0;
                    int pdsch_bundle_size     = 0;
                    int pdsch_num_bundles     = 0;
                    int pdsch_num_bundle_prb  = 0;
                    int csirs_sect_sz         = section_hdr_size;
                    int csirs_num_bundles     = 0;

                    bool pdsch_bfw = false;
                    bool pdschDisableBFWs = false;
                    bool pdsch_static_bfw = true;

                    if(ext11_ptr != nullptr)
                    {
                        auto               pdsch_ext11_ptr  = static_cast<CPlaneSectionExtInfo*>(info.sections[section_num].ext11);
                        oran_cmsg_ext_hdr* ext11_common_hdr = nullptr;
                        pdsch_sect_sz += pdsch_ext11_ptr->ext_11.ext_hdr.extLen << 2;
                        pdsch_num_bundles     = pdsch_ext11_ptr->ext_11.numPrbBundles;
                        pdschDisableBFWs = oran_cmsg_get_ext_11_disableBFWs(&pdsch_ext11_ptr->ext_11.ext_hdr);
                        pdsch_num_bundle_prb  = pdsch_ext11_ptr->ext_11.numBundPrb;

                        // Assuming bfwIQBitWidth == 9
                        pdsch_ext11_hdr_size = sizeof(oran_cmsg_ext_hdr) + sizeof(oran_cmsg_sect_ext_type_11);
                        pdsch_ext11_hdr_size += pdschDisableBFWs ? 0 : sizeof(oran_cmsg_sect_ext_type_11_disableBFWs_0_bfwCompHdr);
                        pdsch_bundle_hdr_size = pdsch_ext11_ptr->ext_11.bundle_hdr_size;
                        pdsch_bfwIQ_size      = pdsch_ext11_ptr->ext_11.bfwIQ_size;
                        pdsch_bundle_size     = pdsch_bundle_hdr_size + (pdschDisableBFWs ? 0 : pdsch_bfwIQ_size);
                        pdsch_bfw             = true;
                        pdsch_static_bfw      = !(!pdsch_ext11_ptr->ext_11.static_bfw);
                    }

                    if(ext4_ptr != nullptr)
                    {
                        pdsch_ext4_hdr_size = sizeof(oran_cmsg_ext_hdr) + sizeof(oran_cmsg_sect_ext_type_4);
                    }
                    if(ext5_ptr != nullptr)
                    {
                        pdsch_ext5_hdr_size = sizeof(oran_cmsg_ext_hdr) + sizeof(oran_cmsg_sect_ext_type_5);
                    }

                    bool csirs_bfw = false;
                    csirs_section_index = section_num + 1;
                    CPlaneSectionExtInfo* csirs_ext11_ptr = nullptr;
                    while(info.sections[csirs_section_index].csirs_of_multiplex_pdsch_csirs && info.sections[csirs_section_index].section_id_lookback_index > 0)
                    {
                        if(nxt_ext11_ptr != nullptr)
                        {
                            csirs_ext11_ptr                     = static_cast<CPlaneSectionExtInfo*>(info.sections[csirs_section_index].ext11);
                            oran_cmsg_ext_hdr* ext11_common_hdr = nullptr;
                            csirs_sect_sz += csirs_ext11_ptr->ext_11.ext_hdr.extLen << 2;
                            csirs_bfw         = csirs_ext11_ptr->ext_11.static_bfw;
                            csirs_num_bundles = csirs_ext11_ptr->ext_11.numPrbBundles;
                        }
                        if(nxt_ext4_ptr != nullptr)
                        {
                            csirs_sect_sz += sizeof(oran_cmsg_ext_hdr) + sizeof(oran_cmsg_sect_ext_type_4);
                        }
                        if(nxt_ext5_ptr != nullptr)
                        {
                            csirs_sect_sz += sizeof(oran_cmsg_ext_hdr) + sizeof(oran_cmsg_sect_ext_type_5);
                        }
                        if(++csirs_section_index < number_of_sections)
                        {
                            nxt_ext11_ptr = static_cast<CPlaneSectionExtInfo*>(info.sections[csirs_section_index].ext11);
                            nxt_ext4_ptr = static_cast<CPlaneSectionExtInfo*>(info.sections[csirs_section_index].ext4);
                            nxt_ext5_ptr = static_cast<CPlaneSectionExtInfo*>(info.sections[csirs_section_index].ext5);
                        }
                        else
                        {
                            break;
                        }
                    }

                    int start_prbc = info.sections[section_num].sect_1.startPrbc.get();
                    int  pdsch_max_num_prbc = (info.sections[section_num].sect_1.numPrbc == 0) ? info_.max_num_prbs_per_symbol : info.sections[section_num].sect_1.numPrbc;
                    bool pdsch_one_bundle   = pdsch_num_bundle_prb >= pdsch_max_num_prbc;
                    int  split_count        = 0;

                    while(pdsch_max_num_prbc)
                    {
                        if(unlikely(++split_count > kMaxSplitCount))
                        {
                            THROW_FH(EINVAL, StringBuilder() << "Error: pdsch + csirs split_count should not be more than " << kMaxSplitCount);
                        }

                        bool start_new_dyn_bfw_packet = prev_section_dyn_bfw || (ext11_ptr != nullptr && pdschDisableBFWs == 0 && pdsch_static_bfw == false);
                        start_new_dyn_bfw_packet = start_new_dyn_bfw_packet && (info_.bfw_cplane_info.bfw_chain_mode != BfwCplaneChainingMode::NO_CHAINING);
                        if(total_section_info_size != 0 && start_new_dyn_bfw_packet)
                        {
                            ++section_num_packets;
                            ++section_bfw_mbufs;
                            total_section_info_size = 0;
                        }

                        auto min_pdsch_sect_sz = section_hdr_size + pdsch_ext4_hdr_size + pdsch_ext5_hdr_size + pdsch_ext11_hdr_size + pdsch_bundle_size;
                        if(total_section_info_size + min_pdsch_sect_sz + csirs_sect_sz > pkt_section_info_room)
                        {
                            ++section_num_packets;
                            total_section_info_size = 0;
                        }

                        prev_section_dyn_bfw = (ext11_ptr != nullptr && pdschDisableBFWs == 0 && pdsch_static_bfw == false);

                        int pdsch_bundle_room         = 0;
                        int current_pdsch_num_bundles = 0;
                        int pdsch_num_prbc            = pdsch_max_num_prbc;
                        int current_pdsch_num_prbc    = pdsch_max_num_prbc;

                        if(pdsch_one_bundle)
                        {
                            current_pdsch_num_bundles = 1;
                            current_pdsch_num_prbc    = std::min(current_pdsch_num_prbc, ORAN_MAX_PRB_X_SECTION);
                            if(pdschDisableBFWs == 0 && pdsch_static_bfw == 0)
                            {
                                auto padding = oran_cmsg_se11_disableBFWs_0_padding_bytes(pdsch_ext11_hdr_size + pdsch_bundle_size * current_pdsch_num_bundles);
                                if(padding > 0)
                                {
                                    ++section_padding_mbufs;
                                }
                            }
                        }
                        else if(pdsch_bfw)
                        {
                            pdsch_bundle_room = pkt_section_info_room - total_section_info_size - section_hdr_size - pdsch_ext11_hdr_size - csirs_sect_sz;

                            current_pdsch_num_bundles = std::min(pdsch_num_bundles, static_cast<int>(pdsch_bundle_room / pdsch_bundle_size));
                            pdsch_num_prbc            = current_pdsch_num_bundles * pdsch_num_bundle_prb;
                            current_pdsch_num_prbc    = std::min(pdsch_num_prbc, pdsch_max_num_prbc);
                            if(csirs_bfw && current_pdsch_num_prbc > ORAN_MAX_PRB_X_SECTION)
                            {
                                current_pdsch_num_prbc    = std::min(current_pdsch_num_prbc, ORAN_MAX_PRB_X_SECTION);
                                current_pdsch_num_bundles = current_pdsch_num_prbc / pdsch_num_bundle_prb;
                                current_pdsch_num_prbc    = current_pdsch_num_bundles * pdsch_num_bundle_prb;
                            }
                            if(pdschDisableBFWs == 0 && pdsch_static_bfw == 0)
                            {
                                auto padding = oran_cmsg_se11_disableBFWs_0_padding_bytes(pdsch_ext11_hdr_size + pdsch_bundle_size * current_pdsch_num_bundles);
                                if(padding > 0)
                                {
                                    ++section_padding_mbufs;
                                }
                            }
                        }
                        else if(csirs_bfw)
                        {
                            current_pdsch_num_prbc = std::min(current_pdsch_num_prbc, ORAN_MAX_PRB_X_SECTION);
                        }

                        //FIXME PDSCH with dyn BFW needs to start new packet for CSIRS
                        total_section_info_size += section_hdr_size + pdsch_ext11_hdr_size + pdsch_bundle_size * current_pdsch_num_bundles + csirs_sect_sz;

                        pdsch_num_bundles -= current_pdsch_num_bundles;
                        pdsch_max_num_prbc -= current_pdsch_num_prbc;
                    }
                    section_num = (csirs_section_index - 1);
                }
                else if(ext11_ptr != nullptr)
                {
                    int start_prbc = info.sections[section_num].sect_1.startPrbc;
                    int max_num_prbc = (info.sections[section_num].sect_1.numPrbc == 0) ? info_.max_num_prbs_per_symbol : info.sections[section_num].sect_1.numPrbc;
                    int num_bundles = ext11_ptr->ext_11.numPrbBundles;

                    count_cplane_packets_mmimo_ext11(
                        section_hdr_size,
                        pkt_section_info_room,
                        total_section_info_size,
                        section_num_packets,
                        section_bfw_mbufs,
                        section_padding_mbufs,
                        start_prbc,
                        max_num_prbc,
                        num_bundles,
                        ext11_ptr,
                        DIRECTION_DOWNLINK,
                        prev_section_dyn_bfw
                    );
                }
                else
                {
                    auto& cur_section_info = info.sections[section_num];
                    count_cplane_packets_mmimo_ext4_ext5(cur_section_info, section_hdr_size, pkt_section_info_room,
                                                 total_section_info_size, section_num_packets);
                }
            }
            cplane_count_info.num_packets += std::max(1UL, section_num_packets);
            if(info_.bfw_cplane_info.bfw_chain_mode != BfwCplaneChainingMode::NO_CHAINING)
            {
                cplane_count_info.num_bfw_mbufs += section_bfw_mbufs;
                cplane_count_info.num_bfw_padding_mbufs += section_padding_mbufs;
            }
        }
    }
    return cplane_count_info;
}

uint16_t Peer::prepare_cplane_message(const CPlaneMsgSendInfo& info, rte_mbuf** mbufs)
{
    auto fhi  = get_fronthaul();
    auto flow = static_cast<Flow*>(info.flow);

    auto&    radio_app_hdr      = info.section_common_hdr.sect_1_common_hdr.radioAppHdr;
    uint8_t  section_type       = radio_app_hdr.sectionType;
    uint8_t  number_of_sections = radio_app_hdr.numberOfSections;
    uint16_t common_hdr_size    = get_cmsg_common_hdr_size(section_type);
    uint16_t section_size       = get_cmsg_section_size(section_type);

    uint16_t mtu                   = nic_->get_mtu();
    uint16_t pkt_section_info_room = mtu - ORAN_CMSG_HDR_OFFSET - common_hdr_size;

    uint8_t  section_num = 0;
    uint16_t packet_num  = 0;

    rte_mbuf*             m    = mbufs[packet_num++];
    m->ol_flags = 0;
    PacketHeaderTemplate* data = rte_pktmbuf_mtod(m, PacketHeaderTemplate*);
    memcpy(data, &flow->get_packet_header_template(), sizeof(PacketHeaderTemplate));

    auto common_hdr_ptr = rte_pktmbuf_mtod_offset(m, uint8_t*, sizeof(PacketHeaderTemplate));
    memcpy(common_hdr_ptr, &info.section_common_hdr, common_hdr_size);
    auto  direction             = info.section_common_hdr.sect_1_common_hdr.radioAppHdr.dataDirection;
    auto& sequence_id_generator = (direction == DIRECTION_UPLINK) ? flow->get_sequence_id_generator_uplink() : flow->get_sequence_id_generator_downlink();

    uint8_t* section_ptr            = common_hdr_ptr + common_hdr_size;
    uint16_t pkt_remaining_capacity = pkt_section_info_room;
    uint8_t  sections_generated     = 0;
    auto& last_packet_ts = (info.data_direction == DIRECTION_DOWNLINK) ? last_dl_cplane_tx_ts_ : last_ul_cplane_tx_ts_;

    if(!info.hasSectionExt)
    {
        while(section_num < number_of_sections)
        {
            size_t extension_size = 0;
            size_t section_size_with_extension = section_size + extension_size;

            if(section_size_with_extension > pkt_remaining_capacity)
            {
                m->data_len                                                             = mtu - pkt_remaining_capacity;
                m->pkt_len                                                              = m->data_len;
                data->ecpri.ecpriSeqid                                                  = sequence_id_generator.next();
                data->ecpri.ecpriPayload                                                = rte_cpu_to_be_16(m->data_len - sizeof(PacketHeaderTemplate) + 4);
                *(common_hdr_ptr + offsetof(oran_cmsg_radio_app_hdr, numberOfSections)) = section_num - sections_generated;
                sections_generated                                                      = section_num;

                m    = mbufs[packet_num++];
                m->ol_flags = 0;
                data = rte_pktmbuf_mtod(m, PacketHeaderTemplate*);
                memcpy(data, &flow->get_packet_header_template(), sizeof(PacketHeaderTemplate));

                common_hdr_ptr = rte_pktmbuf_mtod_offset(m, uint8_t*, sizeof(PacketHeaderTemplate));
                memcpy(common_hdr_ptr, &info.section_common_hdr.sect_1_common_hdr, common_hdr_size);

                section_ptr            = common_hdr_ptr + common_hdr_size;
                pkt_remaining_capacity = pkt_section_info_room - section_size_with_extension;
            }
            else
            {
                pkt_remaining_capacity -= section_size_with_extension;
            }

            memcpy(section_ptr, &info.sections[section_num], section_size);
            section_ptr += section_size;
            section_num++;
        }

        m->data_len                                                             = mtu - pkt_remaining_capacity;
        m->pkt_len                                                              = m->data_len;
        data->ecpri.ecpriSeqid                                                  = sequence_id_generator.next();
        data->ecpri.ecpriPayload                                                = rte_cpu_to_be_16(m->data_len - sizeof(PacketHeaderTemplate) + 4);
        *(common_hdr_ptr + offsetof(oran_cmsg_radio_app_hdr, numberOfSections)) = section_num - sections_generated;

        if(info.tx_window.tx_window_start > last_packet_ts)
        {
            mbufs[0]->ol_flags                                                   = fhi->get_timestamp_mask_();
            *RTE_MBUF_DYNFIELD(mbufs[0], fhi->get_timestamp_offset(), uint64_t*) = info.tx_window.tx_window_start;
            last_packet_ts                                                       = info.tx_window.tx_window_start;
        }

        return packet_num;
    }
    else
    {
        uint8_t* current_ptr = nullptr;
        uint16_t total_section_info_size = 0;
        oran_cmsg_sect_ext_type_11* current_ext11_ptr = nullptr;
        uint16_t current_ext4_len = 0;
        uint16_t current_ext5_len = 0;
        uint16_t current_ext11_len = 0;
        uint16_t curr_start_prbc = 0;
        uint16_t section_start_prbc = 0;
        uint16_t section_num_prbc = 0;
        uint16_t section_max_prbc = 0;
        auto ext4_hdr_size   = sizeof(oran_cmsg_ext_hdr) + sizeof(oran_cmsg_sect_ext_type_4);
        auto ext5_hdr_size   = sizeof(oran_cmsg_ext_hdr) + sizeof(oran_cmsg_sect_ext_type_5);

        for(section_num = 0; section_num < number_of_sections; section_num++)
        {
            if(total_section_info_size + section_size > pkt_section_info_room)
            {
                m->data_len                                                             = mtu - pkt_remaining_capacity;
                m->pkt_len                                                              = m->data_len;
                data->ecpri.ecpriSeqid                                                  = sequence_id_generator.next();
                data->ecpri.ecpriPayload                                                = rte_cpu_to_be_16(m->data_len - sizeof(PacketHeaderTemplate) + 4);
                *(common_hdr_ptr + offsetof(oran_cmsg_radio_app_hdr, numberOfSections)) = section_num - sections_generated;
                sections_generated                                                      = section_num;

                m    = mbufs[packet_num++];
                m->ol_flags = 0;
                data = rte_pktmbuf_mtod(m, PacketHeaderTemplate*);
                memcpy(data, &flow->get_packet_header_template(), sizeof(PacketHeaderTemplate));

                common_hdr_ptr = rte_pktmbuf_mtod_offset(m, uint8_t*, sizeof(PacketHeaderTemplate));
                memcpy(common_hdr_ptr, &info.section_common_hdr.sect_1_common_hdr, common_hdr_size);

                section_ptr            = common_hdr_ptr + common_hdr_size;

                total_section_info_size = 0;
                pkt_remaining_capacity = pkt_section_info_room;
            }
            total_section_info_size += section_size;
            pkt_remaining_capacity -= section_size;
            memcpy(section_ptr, &info.sections[section_num], section_size);
            curr_start_prbc = info.sections[section_num].sect_1.startPrbc;
            section_start_prbc = info.sections[section_num].sect_1.startPrbc;
            section_num_prbc = (info.sections[section_num].sect_1.numPrbc == 0) ? info_.max_num_prbs_per_symbol : info.sections[section_num].sect_1.numPrbc;
            section_max_prbc = section_start_prbc + section_num_prbc;
            current_ptr = section_ptr + section_size;
            current_ext11_len = 0;
            if(section_type == ORAN_CMSG_SECTION_TYPE_1)
            {
                if(oran_cmsg_get_section_1_ef(&info.sections[section_num].sect_1))
                {
                    auto               ext4_ptr = static_cast<CPlaneSectionExtInfo*>(info.sections[section_num].ext4);
                    auto               ext5_ptr = static_cast<CPlaneSectionExtInfo*>(info.sections[section_num].ext5);
                    auto               ext11_ptr = static_cast<CPlaneSectionExtInfo*>(info.sections[section_num].ext11);
                    oran_cmsg_ext_hdr* ext4_hdr = nullptr;
                    oran_cmsg_ext_hdr* ext5_hdr = nullptr;
                    oran_cmsg_ext_hdr* ext11_hdr = nullptr;

                    if(ext5_ptr != nullptr)
                    {
                        uint64_t* ext5_2sets = reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(&ext5_ptr->ext_5.ext_hdr) + 1);
                        *ext5_2sets          = rte_cpu_to_be_64(*ext5_2sets);
                    }

                    if(info.sections[section_num].ext11 != nullptr)
                    {
                        auto disableBFWs = oran_cmsg_get_ext_11_disableBFWs(&ext11_ptr->ext_11.ext_hdr);

                        // Assuming bfwIQBitWidth == 9
                        auto ext11_hdr_size = sizeof(oran_cmsg_ext_hdr) + sizeof(oran_cmsg_sect_ext_type_11);
                        ext11_hdr_size += disableBFWs ? 0 : sizeof(oran_cmsg_sect_ext_type_11_disableBFWs_0_bfwCompHdr);
                        auto bundle_hdr_size = ext11_ptr->ext_11.bundle_hdr_size;
                        auto bfwIQ_size = ext11_ptr->ext_11.bfwIQ_size;
                        auto bundle_size = bundle_hdr_size + (disableBFWs ? 0 : bfwIQ_size);
                        auto total_ext_size  = ((ext4_ptr == nullptr) ? 0 : ext4_hdr_size) + ((ext5_ptr == nullptr) ? 0 : ext5_hdr_size) + ext11_hdr_size + bundle_size;

                        if(unlikely(pkt_section_info_room < section_size + total_ext_size))
                        {
                            THROW_FH(EINVAL, StringBuilder() << "MTU " << nic_->get_mtu() << " is too small to hold a section with a single extType 11 bundle, please increase it!");
                        }

                        if(total_section_info_size + total_ext_size > pkt_section_info_room)
                        {
                            m->data_len                                                             = mtu - pkt_remaining_capacity;
                            m->pkt_len                                                              = m->data_len;
                            data->ecpri.ecpriSeqid                                                  = sequence_id_generator.next();
                            data->ecpri.ecpriPayload                                                = rte_cpu_to_be_16(m->data_len - sizeof(PacketHeaderTemplate) + 4);
                            *(common_hdr_ptr + offsetof(oran_cmsg_radio_app_hdr, numberOfSections)) = section_num + 1 - sections_generated; //Still working on current section
                            sections_generated                                                      = section_num;

                            m    = mbufs[packet_num++];
                            m->ol_flags = 0;
                            data = rte_pktmbuf_mtod(m, PacketHeaderTemplate*);
                            memcpy(data, &flow->get_packet_header_template(), sizeof(PacketHeaderTemplate));

                            common_hdr_ptr = rte_pktmbuf_mtod_offset(m, uint8_t*, sizeof(PacketHeaderTemplate));
                            memcpy(common_hdr_ptr, &info.section_common_hdr.sect_1_common_hdr, common_hdr_size);

                            section_ptr            = common_hdr_ptr + common_hdr_size;

                            total_section_info_size = 0;
                            pkt_remaining_capacity = pkt_section_info_room;

                            memcpy(section_ptr, &info.sections[section_num], section_size);

                            current_ptr = section_ptr + section_size;
                            total_section_info_size += section_size;
                            pkt_remaining_capacity -= section_size;
                            current_ext4_len  = 0;
                            current_ext5_len  = 0;
                            current_ext11_len = 0;
                        }

                        if(ext4_ptr != nullptr)
                        {
                            total_section_info_size += ext4_hdr_size;
                            pkt_remaining_capacity -= ext4_hdr_size;
                            current_ext4_len                     = ext4_hdr_size;
                            oran_cmsg_sect_ext_type_4* ext_4_hdr = nullptr;
                            ext4_hdr                      = static_cast<oran_cmsg_ext_hdr*>(&ext4_ptr->sect_ext_common_hdr);
                            memcpy(current_ptr, ext4_hdr, sizeof(oran_cmsg_ext_hdr));
                            current_ptr += sizeof(oran_cmsg_ext_hdr);

                            ext_4_hdr = static_cast<oran_cmsg_sect_ext_type_4*>(&ext4_ptr->ext_4.ext_hdr);
                            memcpy(current_ptr, ext_4_hdr, sizeof(oran_cmsg_sect_ext_type_4));
                            current_ptr += sizeof(oran_cmsg_sect_ext_type_4);
                        }
                        else if(ext5_ptr != nullptr)
                        {
                            total_section_info_size += ext5_hdr_size;
                            pkt_remaining_capacity -= ext5_hdr_size;
                            current_ext5_len                     = ext5_hdr_size;
                            oran_cmsg_sect_ext_type_5* ext_5_hdr = nullptr;
                            ext5_hdr                             = static_cast<oran_cmsg_ext_hdr*>(&ext5_ptr->sect_ext_common_hdr);
                            memcpy(current_ptr, ext5_hdr, sizeof(oran_cmsg_ext_hdr));
                            current_ptr += sizeof(oran_cmsg_ext_hdr);

                            ext_5_hdr = static_cast<oran_cmsg_sect_ext_type_5*>(&ext5_ptr->ext_5.ext_hdr);
                            memcpy(current_ptr, ext_5_hdr, sizeof(oran_cmsg_sect_ext_type_5));
                            current_ptr += sizeof(oran_cmsg_sect_ext_type_5);
                        }

                        total_section_info_size += ext11_hdr_size;
                        pkt_remaining_capacity -= ext11_hdr_size;
                        current_ext11_len = ext11_hdr_size;
                        oran_cmsg_sect_ext_type_11* ext_11_hdr = nullptr;
                        ext11_hdr          = static_cast<oran_cmsg_ext_hdr*>(&ext11_ptr->sect_ext_common_hdr);
                        memcpy(current_ptr, ext11_hdr, sizeof(oran_cmsg_ext_hdr));
                        current_ptr += sizeof(oran_cmsg_ext_hdr);

                        ext_11_hdr       = static_cast<oran_cmsg_sect_ext_type_11*>(&ext11_ptr->ext_11.ext_hdr);
                        memcpy(current_ptr, ext_11_hdr, sizeof(oran_cmsg_sect_ext_type_11));
                        current_ext11_ptr       = reinterpret_cast<oran_cmsg_sect_ext_type_11*>(current_ptr);
                        current_ptr += sizeof(oran_cmsg_sect_ext_type_11);

                        if(!disableBFWs)
                        {
                            auto ext_comp_ptr = &ext11_ptr->ext_11.ext_comp_hdr;
                            memcpy(current_ptr, ext_comp_ptr, sizeof(oran_cmsg_sect_ext_type_11_disableBFWs_0_bfwCompHdr));
                            current_ptr += sizeof(oran_cmsg_sect_ext_type_11_disableBFWs_0_bfwCompHdr);
                        }

                        auto& num_bundles = ext11_ptr->ext_11.numPrbBundles;
                        int bundles_included = 0;
                        for (int bundle_idx = 0; bundle_idx < num_bundles; ++bundle_idx)
                        {
                            auto padding = oran_cmsg_se11_disableBFWs_0_padding_bytes(current_ext11_len);
                            if(total_section_info_size + bundle_size + padding > pkt_section_info_room)
                            {
                                memset(current_ptr, 0, padding);
                                current_ext11_ptr->extLen = rte_cpu_to_be_16((current_ext11_len + padding) >> 2);
                                auto sect_info_ptr = reinterpret_cast<CPlaneSectionInfo*>(section_ptr);
                                auto numPrbc = bundles_included * ext11_ptr->ext_11.ext_hdr.numBundPrb;
                                numPrbc = (curr_start_prbc + numPrbc > section_max_prbc) ? (section_max_prbc - curr_start_prbc) : numPrbc;
                                sect_info_ptr->sect_1.startPrbc = curr_start_prbc;
                                sect_info_ptr->sect_1.numPrbc = numPrbc;
                                curr_start_prbc += numPrbc;
                                bundles_included = 0;
                                m->data_len                                                             = mtu - pkt_remaining_capacity;
                                m->pkt_len                                                              = m->data_len;
                                data->ecpri.ecpriSeqid                                                  = sequence_id_generator.next();
                                data->ecpri.ecpriPayload                                                = rte_cpu_to_be_16(m->data_len - sizeof(PacketHeaderTemplate) + 4);
                                *(common_hdr_ptr + offsetof(oran_cmsg_radio_app_hdr, numberOfSections)) = section_num + 1 - sections_generated; //Still working on current section
                                sections_generated                                                      = section_num;

                                m    = mbufs[packet_num++];
                                m->ol_flags = 0;
                                data = rte_pktmbuf_mtod(m, PacketHeaderTemplate*);
                                memcpy(data, &flow->get_packet_header_template(), sizeof(PacketHeaderTemplate));

                                common_hdr_ptr = rte_pktmbuf_mtod_offset(m, uint8_t*, sizeof(PacketHeaderTemplate));
                                memcpy(common_hdr_ptr, &info.section_common_hdr.sect_1_common_hdr, common_hdr_size);

                                section_ptr            = common_hdr_ptr + common_hdr_size;

                                total_section_info_size = 0;
                                current_ext11_len = 0;
                                pkt_remaining_capacity = pkt_section_info_room;

                                memcpy(section_ptr, &info.sections[section_num], section_size);

                                current_ptr = section_ptr + section_size;

                                total_section_info_size += section_size;
                                pkt_remaining_capacity -= section_size;

                                if(ext4_ptr != nullptr)
                                {
                                    total_section_info_size += ext4_hdr_size;
                                    pkt_remaining_capacity -= ext4_hdr_size;
                                    current_ext4_len                     = ext4_hdr_size;
                                    oran_cmsg_sect_ext_type_4* ext_4_hdr = nullptr;
                                    ext4_hdr                             = static_cast<oran_cmsg_ext_hdr*>(&ext4_ptr->sect_ext_common_hdr);
                                    memcpy(current_ptr, ext4_hdr, sizeof(oran_cmsg_ext_hdr));
                                    current_ptr += sizeof(oran_cmsg_ext_hdr);

                                    ext_4_hdr = static_cast<oran_cmsg_sect_ext_type_4*>(&ext4_ptr->ext_4.ext_hdr);
                                    memcpy(current_ptr, ext_4_hdr, sizeof(oran_cmsg_sect_ext_type_4));
                                    current_ptr += sizeof(oran_cmsg_sect_ext_type_4);
                                }
                                else if(ext5_ptr != nullptr)
                                {
                                    total_section_info_size += ext5_hdr_size;
                                    pkt_remaining_capacity -= ext5_hdr_size;
                                    current_ext5_len                     = ext5_hdr_size;
                                    oran_cmsg_sect_ext_type_5* ext_5_hdr = nullptr;
                                    ext5_hdr                             = static_cast<oran_cmsg_ext_hdr*>(&ext5_ptr->sect_ext_common_hdr);
                                    memcpy(current_ptr, ext5_hdr, sizeof(oran_cmsg_ext_hdr));
                                    current_ptr += sizeof(oran_cmsg_ext_hdr);

                                    ext_5_hdr = static_cast<oran_cmsg_sect_ext_type_5*>(&ext5_ptr->ext_5.ext_hdr);
                                    memcpy(current_ptr, ext_5_hdr, sizeof(oran_cmsg_sect_ext_type_5));
                                    current_ptr += sizeof(oran_cmsg_sect_ext_type_5);
                                }

                                ext11_hdr          = static_cast<oran_cmsg_ext_hdr*>(&ext11_ptr->sect_ext_common_hdr);
                                memcpy(current_ptr, ext11_hdr, sizeof(oran_cmsg_ext_hdr));
                                current_ptr += sizeof(oran_cmsg_ext_hdr);

                                ext_11_hdr       = static_cast<oran_cmsg_sect_ext_type_11*>(&ext11_ptr->ext_11.ext_hdr);
                                memcpy(current_ptr, ext_11_hdr, sizeof(oran_cmsg_sect_ext_type_11));
                                current_ext11_ptr       = reinterpret_cast<oran_cmsg_sect_ext_type_11*>(current_ptr);
                                current_ptr += sizeof(oran_cmsg_sect_ext_type_11);
                                if(!disableBFWs)
                                {
                                    auto ext_comp_ptr = &ext11_ptr->ext_11.ext_comp_hdr;
                                    memcpy(current_ptr, ext_comp_ptr, sizeof(oran_cmsg_sect_ext_type_11_disableBFWs_0_bfwCompHdr));
                                    current_ptr += sizeof(oran_cmsg_sect_ext_type_11_disableBFWs_0_bfwCompHdr);
                                }
                                total_section_info_size += ext11_hdr_size;
                                pkt_remaining_capacity -= ext11_hdr_size;
                                current_ext11_len = ext11_hdr_size;
                            }

                            auto& bundle_info = ext11_ptr->ext_11.bundles[bundle_idx];
                            if(disableBFWs)
                            {
                                auto bundle_ptr = reinterpret_cast<oran_cmsg_sect_ext_type_11_disableBFWs_1_bundle*>(current_ptr);
                                memcpy(bundle_ptr, &bundle_info.disableBFWs_1, sizeof(oran_cmsg_sect_ext_type_11_disableBFWs_1_bundle));
                                bundle_ptr->beamId = bundle_info.disableBFWs_1.beamId.get();
                                bundle_ptr->reserved = 0;  // Initialize reserved field
                                current_ptr += sizeof(oran_cmsg_sect_ext_type_11_disableBFWs_1_bundle);
                            }
                            else
                            {
                                auto comp_meth = static_cast<UserDataCompressionMethod>(ext11_ptr->ext_11.ext_comp_hdr.bfwCompMeth.get());
                                if(comp_meth == UserDataCompressionMethod::NO_COMPRESSION)
                                {
                                    auto bundle_ptr = reinterpret_cast<oran_cmsg_sect_ext_type_11_disableBFWs_0_bundle_uncompressed*>(current_ptr);
                                    memcpy(bundle_ptr, &bundle_info.disableBFWs_0_uncompressed, sizeof(oran_cmsg_sect_ext_type_11_disableBFWs_0_bundle_uncompressed));
                                    bundle_ptr->beamId = bundle_info.disableBFWs_0_uncompressed.beamId.get();
                                    bundle_ptr->reserved = 0;  // Initialize reserved field
                                    auto bundle_iq_ptr = reinterpret_cast<uint8_t*>(bundle_ptr->bfw);
                                    uint16_t real_sz = ext11_ptr->ext_11.real_bfwIQ_size;
                                    if(real_sz == 0 || real_sz > bfwIQ_size) real_sz = bfwIQ_size; // no padding
                                    memcpy(bundle_iq_ptr, bundle_info.bfwIQ, real_sz);
                                    if(bfwIQ_size > real_sz) memset(bundle_iq_ptr + real_sz, 0, bfwIQ_size - real_sz); // dMIMO: zero-fill padded antennas
                                    current_ptr = bundle_iq_ptr + bfwIQ_size;
                                }
                                else if(comp_meth == UserDataCompressionMethod::BLOCK_FLOATING_POINT)
                                {
                                    auto bundle_ptr = reinterpret_cast<oran_cmsg_sect_ext_type_11_disableBFWs_0_bfp_compressed_bundle_hdr*>(current_ptr);
                                    memcpy(bundle_ptr, &bundle_info.disableBFWs_0_compressed, sizeof(oran_cmsg_sect_ext_type_11_disableBFWs_0_bfp_compressed_bundle_hdr));
                                    bundle_ptr->beamId                = bundle_info.disableBFWs_0_compressed.beamId.get();
                                    bundle_ptr->bfwCompParam.exponent = bundle_info.disableBFWs_0_compressed.bfwCompParam.exponent.get();
                                    bundle_ptr->bfwCompParam.reserved = 0;  // Initialize reserved field
                                    auto bundle_iq_ptr                = reinterpret_cast<uint8_t*>(bundle_ptr->bfw);
                                    uint16_t real_sz = ext11_ptr->ext_11.real_bfwIQ_size;
                                    if(real_sz == 0 || real_sz > bfwIQ_size) real_sz = bfwIQ_size; // no padding
                                    memcpy(bundle_iq_ptr, bundle_info.bfwIQ, real_sz);
                                    if(bfwIQ_size > real_sz) memset(bundle_iq_ptr + real_sz, 0, bfwIQ_size - real_sz); // dMIMO: zero-fill padded antennas
                                    current_ptr = bundle_iq_ptr + bfwIQ_size;
                                }
                            }

                            ++bundles_included;
                            total_section_info_size += bundle_size;
                            pkt_remaining_capacity -= bundle_size;
                            current_ext11_len += bundle_size;
                        }
                        // TODO multiple extensions not implemented
                        if(unlikely(oran_cmsg_get_ext_ef(ext11_hdr)))
                        {
                            THROW_FH(EINVAL, StringBuilder() << "Multiple section extensions in a single section is not yet supported!");
                        }
                        auto sect_info_ptr = reinterpret_cast<CPlaneSectionInfo*>(section_ptr);
                        auto numPrbc = bundles_included * ext11_ptr->ext_11.ext_hdr.numBundPrb;
                        numPrbc = (curr_start_prbc + numPrbc > section_max_prbc) ? (section_max_prbc - curr_start_prbc) : numPrbc;
                        sect_info_ptr->sect_1.startPrbc = curr_start_prbc;
                        sect_info_ptr->sect_1.numPrbc = numPrbc;
                        curr_start_prbc += numPrbc;
                        // zero pad to 4-byte boundary per Table 7.7.11.1-1: Section Extension 11 Data Format (when disableBFWs = 0)
                        auto padding = oran_cmsg_se11_disableBFWs_0_padding_bytes(current_ext11_len);
                        memset(current_ptr, 0, padding);
                        total_section_info_size += padding;
                        pkt_remaining_capacity -= padding;
                        current_ext11_len += padding;
                        current_ext11_ptr->extLen = rte_cpu_to_be_16(current_ext11_len >> 2);
                    }
                    else if(info.sections[section_num].ext4 != nullptr)
                    {
                        auto               ext4_ptr        = info.sections[section_num].ext4;
                        oran_cmsg_ext_hdr* ext4_hdr = nullptr;
                        if(total_section_info_size + ext4_hdr_size > pkt_section_info_room)
                        {
                            m->data_len                                                             = mtu - pkt_remaining_capacity;
                            m->pkt_len                                                              = m->data_len;
                            data->ecpri.ecpriSeqid                                                  = sequence_id_generator.next();
                            data->ecpri.ecpriPayload                                                = rte_cpu_to_be_16(m->data_len - sizeof(PacketHeaderTemplate) + 4);
                            *(common_hdr_ptr + offsetof(oran_cmsg_radio_app_hdr, numberOfSections)) = section_num + 1 - sections_generated; //Still working on current section
                            sections_generated                                                      = section_num;

                            m           = mbufs[packet_num++];
                            m->ol_flags = 0;
                            data        = rte_pktmbuf_mtod(m, PacketHeaderTemplate*);
                            memcpy(data, &flow->get_packet_header_template(), sizeof(PacketHeaderTemplate));

                            common_hdr_ptr = rte_pktmbuf_mtod_offset(m, uint8_t*, sizeof(PacketHeaderTemplate));
                            memcpy(common_hdr_ptr, &info.section_common_hdr.sect_0_common_hdr, common_hdr_size);

                            section_ptr = common_hdr_ptr + common_hdr_size;

                            total_section_info_size = 0;
                            pkt_remaining_capacity  = pkt_section_info_room;

                            memcpy(section_ptr, &info.sections[section_num], section_size);

                            current_ptr = section_ptr + section_size;
                            total_section_info_size += section_size;
                            pkt_remaining_capacity -= section_size;
                            current_ext4_len = 0;
                        }

                        total_section_info_size += ext4_hdr_size;
                        pkt_remaining_capacity -= ext4_hdr_size;
                        current_ext4_len                     = ext4_hdr_size;
                        oran_cmsg_sect_ext_type_4* ext_4_hdr = nullptr;
                        ext4_hdr                             = static_cast<oran_cmsg_ext_hdr*>(&ext4_ptr->sect_ext_common_hdr);
                        memcpy(current_ptr, ext4_hdr, sizeof(oran_cmsg_ext_hdr));
                        current_ptr += sizeof(oran_cmsg_ext_hdr);

                        ext_4_hdr = static_cast<oran_cmsg_sect_ext_type_4*>(&ext4_ptr->ext_4.ext_hdr);
                        memcpy(current_ptr, ext_4_hdr, sizeof(oran_cmsg_sect_ext_type_4));
                        current_ptr += sizeof(oran_cmsg_sect_ext_type_4);
                    }
                    else if(info.sections[section_num].ext5 != nullptr)
                    {
                        auto               ext5_ptr        = info.sections[section_num].ext5;
                        oran_cmsg_ext_hdr* ext5_hdr = nullptr;
                        if(total_section_info_size + ext5_hdr_size > pkt_section_info_room)
                        {
                            m->data_len                                                             = mtu - pkt_remaining_capacity;
                            m->pkt_len                                                              = m->data_len;
                            data->ecpri.ecpriSeqid                                                  = sequence_id_generator.next();
                            data->ecpri.ecpriPayload                                                = rte_cpu_to_be_16(m->data_len - sizeof(PacketHeaderTemplate) + 4);
                            *(common_hdr_ptr + offsetof(oran_cmsg_radio_app_hdr, numberOfSections)) = section_num + 1 - sections_generated; //Still working on current section
                            sections_generated                                                      = section_num;

                            m           = mbufs[packet_num++];
                            m->ol_flags = 0;
                            data        = rte_pktmbuf_mtod(m, PacketHeaderTemplate*);
                            memcpy(data, &flow->get_packet_header_template(), sizeof(PacketHeaderTemplate));

                            common_hdr_ptr = rte_pktmbuf_mtod_offset(m, uint8_t*, sizeof(PacketHeaderTemplate));
                            memcpy(common_hdr_ptr, &info.section_common_hdr.sect_0_common_hdr, common_hdr_size);

                            section_ptr = common_hdr_ptr + common_hdr_size;

                            total_section_info_size = 0;
                            pkt_remaining_capacity  = pkt_section_info_room;

                            memcpy(section_ptr, &info.sections[section_num], section_size);

                            current_ptr = section_ptr + section_size;
                            total_section_info_size += section_size;
                            pkt_remaining_capacity -= section_size;
                            current_ext5_len = 0;
                        }

                        total_section_info_size += ext5_hdr_size;
                        pkt_remaining_capacity -= ext5_hdr_size;
                        current_ext5_len                     = ext5_hdr_size;
                        oran_cmsg_sect_ext_type_5* ext_5_hdr = nullptr;
                        ext5_hdr                             = static_cast<oran_cmsg_ext_hdr*>(&ext5_ptr->sect_ext_common_hdr);
                        memcpy(current_ptr, ext5_hdr, sizeof(oran_cmsg_ext_hdr));
                        current_ptr += sizeof(oran_cmsg_ext_hdr);

                        ext_5_hdr = static_cast<oran_cmsg_sect_ext_type_5*>(&ext5_ptr->ext_5.ext_hdr);
                        memcpy(current_ptr, ext_5_hdr, sizeof(oran_cmsg_sect_ext_type_5));
                        current_ptr += sizeof(oran_cmsg_sect_ext_type_5);
                    }
                }
            }
            section_ptr += section_size + current_ext4_len + current_ext5_len + current_ext11_len;
            current_ext4_len = 0;
            current_ext5_len = 0;
        }

        m->data_len                                                             = mtu - pkt_remaining_capacity;
        m->pkt_len                                                              = m->data_len;
        data->ecpri.ecpriSeqid                                                  = sequence_id_generator.next();
        data->ecpri.ecpriPayload                                                = rte_cpu_to_be_16(m->data_len - sizeof(PacketHeaderTemplate) + 4);
        *(common_hdr_ptr + offsetof(oran_cmsg_radio_app_hdr, numberOfSections)) = section_num - sections_generated;

        if(info.tx_window.tx_window_start > last_packet_ts)
        {
            mbufs[0]->ol_flags                                                   = fhi->get_timestamp_mask_();
            *RTE_MBUF_DYNFIELD(mbufs[0], fhi->get_timestamp_offset(), uint64_t*) = info.tx_window.tx_window_start;
            last_packet_ts                                                   = info.tx_window.tx_window_start;
        }

        return packet_num;
    }
}

size_t Peer::count_cplane_packets(CPlaneMsgSendInfo const* infos, size_t num_msgs)
{
    size_t num_packets = 0;

    for(size_t i = 0; i < num_msgs; ++i)
    {
        auto const& info = infos[i];

        if(!info.hasSectionExt)
        {
            auto& radio_app_hdr = info.section_common_hdr.sect_1_common_hdr.radioAppHdr;
            auto  section_type  = radio_app_hdr.sectionType;

            if(unlikely(section_type > ORAN_CMSG_SECTION_TYPE_5))
            {
                THROW_FH(EINVAL, StringBuilder() << "Section type " << (int)section_type << " is not supported");
            }

            auto number_of_sections = radio_app_hdr.numberOfSections;
            auto common_hdr_size    = get_cmsg_common_hdr_size(section_type);

            if(unlikely(common_hdr_size == 0))
            {
                THROW_FH(EINVAL, StringBuilder() << "Section type " << (int)section_type << " is not supported");
            }

            auto total_section_info_size = get_cmsg_section_size(section_type) * number_of_sections;
            auto pkt_section_info_room   = nic_->get_mtu() - ORAN_CMSG_HDR_OFFSET - common_hdr_size;

            num_packets += std::max(1UL, (total_section_info_size + pkt_section_info_room - 1) / pkt_section_info_room);
        }
        else
        {
            size_t section_num_packets = 1;
            auto& radio_app_hdr = info.section_common_hdr.sect_1_common_hdr.radioAppHdr;
            auto  section_type  = radio_app_hdr.sectionType;

            if(unlikely(section_type > ORAN_CMSG_SECTION_TYPE_5))
            {
                THROW_FH(EINVAL, StringBuilder() << "Section type " << (int)section_type << " is not supported");
            }

            auto number_of_sections = radio_app_hdr.numberOfSections;
            auto common_hdr_size    = get_cmsg_common_hdr_size(section_type);
            auto section_size       = get_cmsg_section_size(section_type);

            if(unlikely(common_hdr_size == 0))
            {
                THROW_FH(EINVAL, StringBuilder() << "Section type " << (int)section_type << " is not supported");
            }

            uint16_t total_section_info_size = 0;
            auto pkt_section_info_room   = nic_->get_mtu() - ORAN_CMSG_HDR_OFFSET - common_hdr_size;

            if(unlikely(pkt_section_info_room < section_size))
            {
                THROW_FH(EINVAL, StringBuilder() << "MTU " << nic_->get_mtu() << " is too small for " << section_type << " section header size " << section_size);
            }

            auto ext4_hdr_size = sizeof(oran_cmsg_ext_hdr) + sizeof(oran_cmsg_sect_ext_type_4);
            auto ext5_hdr_size = sizeof(oran_cmsg_ext_hdr) + sizeof(oran_cmsg_sect_ext_type_5);
            for(uint8_t section_num = 0; section_num < number_of_sections; section_num++)
            {
                if(section_type == ORAN_CMSG_SECTION_TYPE_1)
                {
                    total_section_info_size += section_size;

                    if(total_section_info_size > pkt_section_info_room)
                    {
                        ++section_num_packets;
                        total_section_info_size = section_size;
                    }
                    if(oran_cmsg_get_section_1_ef(&info.sections[section_num].sect_1))
                    {
                        auto               ext4_ptr = static_cast<CPlaneSectionExtInfo*>(info.sections[section_num].ext4);
                        oran_cmsg_ext_hdr* ext4_hdr = nullptr;
                        auto               ext5_ptr = static_cast<CPlaneSectionExtInfo*>(info.sections[section_num].ext5);
                        oran_cmsg_ext_hdr* ext5_hdr = nullptr;
                        auto               ext11_ptr = static_cast<CPlaneSectionExtInfo*>(info.sections[section_num].ext11);
                        oran_cmsg_ext_hdr* ext11_hdr = nullptr;

                        if(ext11_ptr != nullptr)
                        {
                            auto disableBFWs = oran_cmsg_get_ext_11_disableBFWs(&ext11_ptr->ext_11.ext_hdr);
                            uint16_t ext_len = ext11_ptr->ext_11.ext_hdr.extLen << 2;
                            ext11_hdr          = static_cast<oran_cmsg_ext_hdr*>(static_cast<void*>(&ext11_ptr->sect_ext_common_hdr));

                            //bfwIQBitWidth == 9
                            auto ext11_hdr_size = sizeof(oran_cmsg_ext_hdr) + sizeof(oran_cmsg_sect_ext_type_11);
                            ext11_hdr_size += disableBFWs ? 0 : sizeof(oran_cmsg_sect_ext_type_11_disableBFWs_0_bfwCompHdr);
                            auto bundle_hdr_size = ext11_ptr->ext_11.bundle_hdr_size;
                            auto bfwIQ_size      = ext11_ptr->ext_11.bfwIQ_size;
                            auto bundle_size     = bundle_hdr_size + (disableBFWs ? 0 : bfwIQ_size);
                            auto total_ext_size  = ((ext4_ptr == nullptr) ? 0 : ext4_hdr_size) + ((ext5_ptr == nullptr) ? 0 : ext5_hdr_size) + ext11_hdr_size + bundle_size;
                            if(unlikely(pkt_section_info_room < section_size + total_ext_size))
                            {
                                THROW_FH(EINVAL, StringBuilder() << "MTU " << nic_->get_mtu() << " is too small to hold a section with a single extType 11 bundle, please increase it!");
                            }

                            if(total_section_info_size + total_ext_size > pkt_section_info_room)
                            {
                                ++section_num_packets;
                                total_section_info_size = section_size;
                            }

                            total_section_info_size += ((ext4_ptr == nullptr) ? 0 : ext4_hdr_size) + ((ext5_ptr == nullptr) ? 0 : ext5_hdr_size);
                            total_section_info_size += ext11_hdr_size;
                            auto fragmented_ext_len = ext11_hdr_size;
                            auto& num_bundles = ext11_ptr->ext_11.numPrbBundles;
                            for (int bundle_idx = 0; bundle_idx < num_bundles; ++bundle_idx)
                            {
                                auto padding = oran_cmsg_se11_disableBFWs_0_padding_bytes(fragmented_ext_len);
                                if(total_section_info_size + bundle_size + padding > pkt_section_info_room)
                                {
                                    ++section_num_packets;
                                    total_section_info_size = section_size + ((ext4_ptr == nullptr) ? 0 : ext4_hdr_size) + ((ext5_ptr == nullptr) ? 0 : ext5_hdr_size) + ext11_hdr_size;
                                    fragmented_ext_len = ext11_hdr_size;
                                }
                                total_section_info_size += bundle_size;
                                fragmented_ext_len += bundle_size;
                            }
                        }
                        else if(ext4_ptr != nullptr)
                        {
                            if(total_section_info_size + ext4_hdr_size > pkt_section_info_room)
                            {
                                ++section_num_packets;
                                total_section_info_size = section_size;
                            }
                            total_section_info_size += ext4_hdr_size;
                        }
                        else if(ext5_ptr != nullptr)
                        {
                            if(total_section_info_size + ext5_hdr_size > pkt_section_info_room)
                            {
                                ++section_num_packets;
                                total_section_info_size = section_size;
                            }
                            total_section_info_size += ext5_hdr_size;
                        }
                    }
                }
            }
            num_packets += std::max(1UL, section_num_packets);
        }
    }

    return num_packets;
}

void Peer::prepare_cplane(CPlaneMsgSendInfo const* info, size_t num_msgs, TxRequestCplane** tx_request)
{
    TxRequestCplane* tx_request_local = nullptr;

    auto num_packets = count_cplane_packets(info, num_msgs);
    if(unlikely(num_packets > kTxPktBurstCplane))
    {
        THROW_FH(EINVAL, StringBuilder() << "Too many C-plane packets to send: " << num_packets << ". Please raise kTxPktBurstCplane " << kTxPktBurstCplane << " !");
    }

    auto tx_request_mempool = nic_->get_tx_request_cplane_pool();
    if(unlikely(0 != rte_mempool_get(tx_request_mempool, reinterpret_cast<void**>(&tx_request_local))))
    {
        THROW_FH(ENOMEM, "Ran out of TxRequestCplane descriptors");
    }

    auto      cpu_mbuf_pool = nic_->get_cpu_mbuf_pool();
    if(unlikely(0 != rte_mempool_get_bulk(cpu_mbuf_pool, reinterpret_cast<void**>(tx_request_local->mbufs), num_packets)))
    {
        THROW_FH(ENOMEM, StringBuilder() << "Failed to allocate " << num_packets << " mbufs for " << num_msgs << " C-plane messages");
    }

    uint16_t created_pkts = 0;
    for(size_t i = 0; i < num_msgs; ++i)
    {
        created_pkts += this->prepare_cplane_message(info[i], &tx_request_local->mbufs[created_pkts]);
    }

    tx_request_local->peer       = this;
    tx_request_local->mbuf_count = created_pkts;
    *tx_request                  = tx_request_local;
}

void Peer::enqueue_cplane_tx_request_descriptor(TxRequestCplane* tx_request)
{
    rte_mempool_put(nic_->get_tx_request_cplane_pool(), tx_request);
}

void Peer::send_cplane_enqueue_nic(Txq *txq, rte_mbuf* mbufs[], size_t num_packets)
{
    if(!(get_fronthaul()->get_info().cuda_device_ids.empty()))
    {
        txq->send(&mbufs[0], num_packets);
    }
    else
    {
        txq->send_lock(&mbufs[0], num_packets);
    }
}


size_t Peer::send_cplane(CPlaneMsgSendInfo const* info, size_t num_msgs)
{
    Txq* txq = (info->data_direction == DIRECTION_DOWNLINK) ? txq_dl_cplane_ : txq_ul_cplane_;
    if(unlikely(txq == nullptr))
    {
        THROW_FH(EINVAL, StringBuilder() << "Calling send_cplane() but Peer " << info_.dst_mac_addr << " was configured with txq_cplane = false");
    }

    size_t num_packets = count_cplane_packets(info, num_msgs);

    if(unlikely(num_packets > kTxPktBurstCplane))
    {
        THROW_FH(EINVAL, StringBuilder() << "Too many C-plane packets to send: " << num_packets << ". Please raise kTxPktBurstCplane " << kTxPktBurstCplane << " !");
    }

    rte_mbuf* mbufs[kTxPktBurstCplane];
    auto      cpu_mbuf_pool = nic_->get_cpu_mbuf_pool();

    if(unlikely(0 != rte_mempool_get_bulk(cpu_mbuf_pool, reinterpret_cast<void**>(mbufs), num_packets)))
    {
        THROW_FH(ENOMEM, StringBuilder() << "Failed to allocate " << num_packets << " mbufs for " << num_msgs << " C-plane messages");
    }

    uint16_t created_pkts = 0;
    for(size_t i = 0; i < num_msgs; ++i)
    {
        created_pkts += this->prepare_cplane_message(info[i], &mbufs[created_pkts]);

    }

    // Any interception to dump the mbufs to PCAP can be done now. The logger clones the provided mbufs and frees them
    // after logging into the file.
    if (unlikely(PcapLogger::instance().isDlCplaneLoggingEnabled() && (info->data_direction == DIRECTION_DOWNLINK))) {
        for (uint16_t i = 0; i < created_pkts; ++i) {
            // Clone the mbuf. The consumer is responsible to free it after logging
            rte_mbuf *cmbuf = rte_pktmbuf_clone(mbufs[i], cpu_mbuf_pool);
            if (!PcapLogger::instance().enqueue(
                cmbuf,
                PcapLoggerType::DL_CPLANE)) {
                rte_pktmbuf_free(cmbuf);
                printf ("PcapLogger ring buffer is full, so packet dropped from logging\n");
            }
        }
    }

    if (unlikely(PcapLogger::instance().isUlCplaneLoggingEnabled() && (info->data_direction != DIRECTION_DOWNLINK))) {
        for (uint16_t i = 0; i < created_pkts; ++i) {
            // Clone the mbuf. The consumer is responsible to free it after logging
            rte_mbuf *cmbuf = rte_pktmbuf_clone(mbufs[i], cpu_mbuf_pool);
            if (!PcapLogger::instance().enqueue(
                cmbuf,
                PcapLoggerType::UL_CPLANE)) {
                rte_pktmbuf_free(cmbuf);
                printf ("PcapLogger ring buffer is full, so packet dropped from logging\n");
            }
        }
    }

    send_cplane_enqueue_nic(txq, mbufs, num_packets); 

#ifdef AERIAL_METRICS
    size_t tx_bytes = sum_up_tx_bytes(&mbufs[0], num_packets);
    metrics_.update(PeerMetric::kCPlaneTxPackets, num_packets);
    metrics_.update(PeerMetric::kCPlaneTxBytes, tx_bytes);
#endif

    return num_packets;
}

void Peer::send_cplane_packets_dl(MbufArray& mbufs_bfw, MbufArray& mbufs_regular, rte_mbuf* mbufs[], size_t num_packets, size_t created_pkts) {
    if(unlikely(num_packets != created_pkts))
    {
        NVLOGC_FMT(TAG, "num_packets: {} created_pkts: {} please check count_cplane_packets_mmimo_dl", num_packets, created_pkts);
    }
    send_cplane_packets(mbufs_bfw, mbufs_regular, mbufs, num_packets, created_pkts, txq_dl_cplane_, txq_dl_bfw_cplane_, (info_.bfw_cplane_info.dlc_bfw_enable_divide_per_cell || info_.bfw_cplane_info.dlc_alloc_cplane_bfw_txq));
}

void Peer::send_cplane_packets_ul(MbufArray& mbufs_bfw, MbufArray& mbufs_regular, rte_mbuf* mbufs[], size_t num_packets, size_t created_pkts) {
    if(unlikely(num_packets != created_pkts))
    {
        NVLOGC_FMT(TAG, "num_packets: {} created_pkts: {} please check count_cplane_packets_mmimo_ul", num_packets, created_pkts);
    }
    send_cplane_packets(mbufs_bfw, mbufs_regular, mbufs, num_packets, created_pkts, txq_ul_cplane_, txq_ul_bfw_cplane_, (info_.bfw_cplane_info.ulc_bfw_enable_divide_per_cell || info_.bfw_cplane_info.ulc_alloc_cplane_bfw_txq));
}

// Helper method to handle C-plane packet sending for both UL and DL
void Peer::send_cplane_packets(MbufArray& mbufs_bfw, MbufArray& mbufs_regular, rte_mbuf* mbufs[], size_t num_packets, size_t created_pkts, Txq* txq, Txq* bfw_txq, bool use_bfw_q) {
    const char* direction = (txq == txq_dl_cplane_) ? "dl" : "ul";

    if(unlikely(txq == nullptr))
    {
        THROW_FH(EINVAL, StringBuilder() << "Calling send_cplane() but Peer " << info_.dst_mac_addr << " was configured with txq_cplane = false");
    }

    if (use_bfw_q) {
        if (mbufs_bfw.size() > 0) {
            if (unlikely(bfw_txq == nullptr)) {
                THROW_FH(EINVAL, StringBuilder() << "Calling send_cplane() but Peer " << info_.dst_mac_addr
                        << " was configured with txq_" << direction << "_bfw_cplane = false");
            }
            bfw_txq->send(&mbufs_bfw.data()[0], mbufs_bfw.size());
        }
        if (mbufs_regular.size() > 0) {
            txq->send_lock(&mbufs_regular.data()[0], mbufs_regular.size());
        }
        if (unlikely(mbufs_bfw.size() + mbufs_regular.size() != num_packets)) {
            THROW_FH(EINVAL, StringBuilder() << "Potential race condition detected: mbufs_bfw.size() " << mbufs_bfw.size() << " + mbufs_regular.size() " << mbufs_regular.size() << " != num_packets " << num_packets);
        }
    } else {
        if(!(get_fronthaul()->get_info().cuda_device_ids.empty()))
        {
            txq->send(&mbufs[0], num_packets);
        }
        else
        {
            txq->send_lock(&mbufs[0], num_packets);
        }
    }
}

size_t Peer::send_cplane_mmimo(CPlaneMsgSendInfo const* info, size_t num_msgs)
{
    cplaneCountInfo cplane_count_info = count_cplane_packets_mmimo(info, num_msgs, kTxPktBurstCplane);
    size_t num_packets = cplane_count_info.num_packets;
    size_t bfw_mbufs = cplane_count_info.num_bfw_mbufs;
    size_t bfw_padding_mbufs = cplane_count_info.num_bfw_padding_mbufs;

    // NVLOGC_FMT(TAG, "F{}S{}S{} num_packets: {} bfw_mbufs: {} bfw_padding_mbufs: {}", info->section_common_hdr.sect_1_common_hdr.radioAppHdr.frameId, info->section_common_hdr.sect_1_common_hdr.radioAppHdr.subframeId.get(), info->section_common_hdr.sect_1_common_hdr.radioAppHdr.slotId.get(), num_packets, bfw_mbufs, bfw_padding_mbufs);
    //TODO: Optimize get_bulk call to allocate bfw and padding mbufs in one call

    rte_mbuf* mbufs[kTxPktBurstCplane];
    MbufArray mbufs_regular;
    MbufArray mbufs_bfw;

    auto      cpu_mbuf_pool = nic_->get_cpu_mbuf_pool();

    if(unlikely(0 != rte_mempool_get_bulk(cpu_mbuf_pool, reinterpret_cast<void**>(mbufs), num_packets + bfw_mbufs + bfw_padding_mbufs)))
    {
        THROW_FH(ENOMEM, StringBuilder() << "Failed to allocate " << num_packets + bfw_mbufs + bfw_padding_mbufs << " mbufs for " << num_msgs << " C-plane messages");
    }

    rte_mbuf** chain_mbufs = &mbufs[num_packets];
    cplanePrepareInfo cplane_prepare_info = {.created_pkts = 0, .chained_mbufs = 0};
    for(size_t i = 0; i < num_msgs; ++i)
    {
        prepare_cplane_message_mmimo(info[i], &mbufs[cplane_prepare_info.created_pkts], &chain_mbufs[cplane_prepare_info.chained_mbufs], &mbufs_regular, &mbufs_bfw, cplane_prepare_info);
        if(unlikely(cplane_prepare_info.created_pkts > num_packets))
        {
            THROW_FH(EINVAL, StringBuilder() << "Creating more C-plane packets than allocated: " << cplane_prepare_info.created_pkts << " > " << num_packets);
        }
        if(unlikely(cplane_prepare_info.chained_mbufs > bfw_mbufs + bfw_padding_mbufs))
        {
            THROW_FH(EINVAL, StringBuilder() << "Creating more C-plane chained mbufs than allocated: " << cplane_prepare_info.chained_mbufs << " > " << bfw_mbufs << " + " << bfw_padding_mbufs);
        }
    }

    auto& created_pkts = cplane_prepare_info.created_pkts;

    auto& chained_mbufs = cplane_prepare_info.chained_mbufs;

    // Check for unused allocated mbufs
    const std::size_t total_allocated = num_packets + bfw_mbufs + bfw_padding_mbufs;
    const std::size_t total_used = created_pkts + chained_mbufs;
    if(unlikely(total_used < total_allocated))
    {
        const std::size_t leaked = total_allocated - total_used;
        NVLOGI_FMT(TAG, "Peer {} MBUF OVERALLOCATION DETECTED: allocated={} (num_packets={} bfw+pad={}) used={} (pkts={} chains={}) LEAKED={}",
                   info_.id, total_allocated, num_packets, (bfw_mbufs + bfw_padding_mbufs), total_used, created_pkts, chained_mbufs, leaked);

        // Free unused packet mbufs
        if(created_pkts < num_packets)
        {
            const std::size_t unused_packets = num_packets - created_pkts;
            NVLOGI_FMT(TAG, "Peer {} freeing {} unused packet mbufs", info_.id, unused_packets);
            rte_pktmbuf_free_bulk(&mbufs[created_pkts], unused_packets);
        }

        // Free unused chain mbufs
        if(chained_mbufs < bfw_mbufs + bfw_padding_mbufs)
        {
            const std::size_t unused_chains = (bfw_mbufs + bfw_padding_mbufs) - chained_mbufs;
            NVLOGI_FMT(TAG, "Peer {} freeing {} unused chain mbufs", info_.id, unused_chains);
            rte_pktmbuf_free_bulk(&chain_mbufs[chained_mbufs], unused_chains);
        }
    }

    // Any interception to dump the mbufs to PCAP can be done now. The logger clones the provided mbufs and frees them
    // after logging into the file.
    if (unlikely(PcapLogger::instance().isDlCplaneLoggingEnabled() && (info->data_direction == DIRECTION_DOWNLINK))) {
        for (uint16_t i = 0; i < created_pkts; ++i) {
            // Clone the mbuf. The consumer is responsible to free it after logging
            rte_mbuf *cmbuf = rte_pktmbuf_clone(mbufs[i], cpu_mbuf_pool);
            if (!PcapLogger::instance().enqueue(
                cmbuf,
                PcapLoggerType::DL_CPLANE)) {
                rte_pktmbuf_free(cmbuf);
                printf ("PcapLogger ring buffer is full, so packet dropped from logging\n");
            }
        }
    }

    if (unlikely(PcapLogger::instance().isUlCplaneLoggingEnabled() && (info->data_direction != DIRECTION_DOWNLINK))) {
        for (uint16_t i = 0; i < created_pkts; ++i) {
            // Clone the mbuf. The consumer is responsible to free it after logging
            rte_mbuf *cmbuf = rte_pktmbuf_clone(mbufs[i], cpu_mbuf_pool);
            if (!PcapLogger::instance().enqueue(
                cmbuf,
                PcapLoggerType::UL_CPLANE)) {
                rte_pktmbuf_free(cmbuf);
                printf ("PcapLogger ring buffer is full, so packet dropped from logging\n");
            }
        }
    }

    if(info->data_direction == DIRECTION_DOWNLINK)
    {
        send_cplane_packets_dl(mbufs_bfw, mbufs_regular, mbufs, num_packets, created_pkts);
    }
    else
    {
        send_cplane_packets_ul(mbufs_bfw, mbufs_regular, mbufs, num_packets, created_pkts);
    }

#ifdef AERIAL_METRICS
    size_t tx_bytes = sum_up_tx_bytes(&mbufs[0], num_packets);
    metrics_.update(PeerMetric::kCPlaneTxPackets, num_packets);
    metrics_.update(PeerMetric::kCPlaneTxBytes, tx_bytes);
#endif
    return num_packets;
}

static inline void update_ecpri_header(PacketHeaderTemplate* data, const EcpriHdrConfig* ecpri_hdr_cfg)
{
    if(ecpri_hdr_cfg != nullptr)
    {
        if(ecpri_hdr_cfg->ecpriVersion.enable)
        {
            data->ecpri.ecpriVersion = ecpri_hdr_cfg->ecpriVersion.value;
        }
        if(ecpri_hdr_cfg->ecpriReserved.enable)
        {
            data->ecpri.ecpriReserved = ecpri_hdr_cfg->ecpriReserved.value;
        }
        if(ecpri_hdr_cfg->ecpriConcatenation.enable)
        {
            data->ecpri.ecpriConcatenation = ecpri_hdr_cfg->ecpriConcatenation.value;
        }
        if(ecpri_hdr_cfg->ecpriMessage.enable)
        {
            data->ecpri.ecpriMessage = ecpri_hdr_cfg->ecpriMessage.value;
        }
        if(ecpri_hdr_cfg->ecpriPayload.enable)
        {
            data->ecpri.ecpriPayload = ecpri_hdr_cfg->ecpriPayload.value;
        }
        if(ecpri_hdr_cfg->ecpriRtcid.enable)
        {
            data->ecpri.ecpriRtcid = ecpri_hdr_cfg->ecpriRtcid.value;
        }
        if(ecpri_hdr_cfg->ecpriPcid.enable)
        {
            data->ecpri.ecpriPcid = ecpri_hdr_cfg->ecpriPcid.value;
        }
        if(ecpri_hdr_cfg->ecpriSeqid.enable)
        {
            data->ecpri.ecpriSeqid = ecpri_hdr_cfg->ecpriSeqid.value;
        }
        if(ecpri_hdr_cfg->ecpriEbit.enable)
        {
            data->ecpri.ecpriEbit = ecpri_hdr_cfg->ecpriEbit.value;
        }
        if(ecpri_hdr_cfg->ecpriSubSeqid.enable)
        {
            data->ecpri.ecpriSubSeqid = ecpri_hdr_cfg->ecpriSubSeqid.value;
        }
    }
}
size_t Peer::prepare_uplane_message_v3(const UPlaneMsgMultiSectionSendInfo& info,
                                       rte_mbuf**                           header_mbufs,
                                       int                                  txq_index)
{
    auto flow = static_cast<Flow*>(info.flow);
    auto fhi  = get_fronthaul();

    size_t mtu_sz             = nic_->get_mtu();
    size_t pkt_sz             = 0;
    size_t remaining_num_prbu = 0;
    size_t section_idx        = 0;
    size_t packet_idx         = 0;
    size_t max_prbs_for_packet = 0;
    bool first_pkt = true;
    size_t num_prbs           = 0;
    size_t prbu_buf_len       = 0;
    size_t section_buf_len    = 0;
    // Offset buffer by the starting prb
    uint8_t* prbu_buf_ptr;

    rte_mbuf* header_mbuf;
    rte_mbuf* section_hdr_mbuf;
    rte_mbuf* iq_data_mbuf;

    size_t ecpri_payload_overhead = 4 + sizeof(oran_umsg_iq_hdr);

    uint16_t              ecpri_payload = 0;
    PacketHeaderTemplate* data;
    if(info.section_num == 0)
    {
        return 0;
    }
    // initial packet
    pkt_sz = ORAN_IQ_STATIC_OVERHEAD;
    header_mbuf           = header_mbufs[packet_idx++];
    header_mbuf->ol_flags = 0;
    data                  = rte_pktmbuf_mtod(header_mbuf, PacketHeaderTemplate*);
    memcpy(data, &flow->get_packet_header_template(), sizeof(PacketHeaderTemplate));

    auto umsg_iq_hdr = rte_pktmbuf_mtod_offset(header_mbuf, oran_umsg_iq_hdr*, sizeof(PacketHeaderTemplate));
    memcpy(umsg_iq_hdr, &info.radio_app_hdr, sizeof(oran_umsg_iq_hdr));

    header_mbuf->data_len  = ORAN_IQ_STATIC_OVERHEAD;
    header_mbuf->pkt_len   = header_mbuf->data_len;
    data->ecpri.ecpriSeqid = flow->get_sequence_id_generator_downlink().next();
    ecpri_payload          = ecpri_payload_overhead;

    for(size_t section_idx = 0; section_idx < info.section_num; ++section_idx)
    {
        remaining_num_prbu = info.section_infos[section_idx].num_prbu;
        prbu_buf_ptr       = reinterpret_cast<uint8_t*>(info.section_infos[section_idx].iq_data_buffer);

        while(remaining_num_prbu > 0)
        {
            max_prbs_for_packet = (mtu_sz - pkt_sz - ORAN_IQ_UNCOMPRESSED_SECTION_OVERHEAD) / prb_size_upl_;
            if(max_prbs_for_packet == 0)
            {
                pkt_sz = ORAN_IQ_STATIC_OVERHEAD;
                header_mbuf           = header_mbufs[packet_idx++];
                header_mbuf->ol_flags = 0;
                data                  = rte_pktmbuf_mtod(header_mbuf, PacketHeaderTemplate*);
                memcpy(data, &flow->get_packet_header_template(), sizeof(PacketHeaderTemplate));

                auto umsg_iq_hdr = rte_pktmbuf_mtod_offset(header_mbuf, oran_umsg_iq_hdr*, sizeof(PacketHeaderTemplate));
                memcpy(umsg_iq_hdr, &info.radio_app_hdr, sizeof(oran_umsg_iq_hdr));

                header_mbuf->data_len  = ORAN_IQ_STATIC_OVERHEAD;
                header_mbuf->pkt_len   = header_mbuf->data_len;
                data->ecpri.ecpriSeqid = flow->get_sequence_id_generator_downlink().next();
                ecpri_payload          = ecpri_payload_overhead;
            }
            max_prbs_for_packet = (mtu_sz - pkt_sz - ORAN_IQ_UNCOMPRESSED_SECTION_OVERHEAD) / prb_size_upl_;
            num_prbs     = std::min(remaining_num_prbu, max_prbs_for_packet);
            if(num_prbs > ORAN_MAX_PRB_X_SECTION && num_prbs < info_.max_num_prbs_per_symbol)
            {
                num_prbs = ORAN_MAX_PRB_X_SECTION;
            }
            prbu_buf_len = num_prbs * prb_size_upl_;
            remaining_num_prbu -= num_prbs;

            section_buf_len = ORAN_IQ_UNCOMPRESSED_SECTION_OVERHEAD + prbu_buf_len;

            pkt_sz += section_buf_len;
            oran_u_section_uncompressed section_template{};
            section_template.sectionId = info.section_infos[section_idx].section_id;
            section_template.rb        = info.section_infos[section_idx].rb;
            section_template.symInc    = info.section_infos[section_idx].sym_inc;

            auto prb_cnt               = info.section_infos[section_idx].num_prbu - remaining_num_prbu - num_prbs;
            section_template.startPrbu = info.section_infos[section_idx].start_prbu + prb_cnt;
            section_template.numPrbu   = (num_prbs > ORAN_MAX_PRB_X_SECTION && num_prbs == info_.max_num_prbs_per_symbol) ? 0 : num_prbs;

            memcpy(rte_pktmbuf_mtod_offset(header_mbuf, uint8_t *, header_mbuf->data_len), &section_template, sizeof(oran_u_section_uncompressed));
            //rte_pktmbuf_adj(mbuf, buf_len);
            header_mbuf->data_len  += sizeof(oran_u_section_uncompressed);
            header_mbuf->pkt_len   = header_mbuf->data_len;

            memcpy(rte_pktmbuf_mtod_offset(header_mbuf, uint8_t *, header_mbuf->data_len), prbu_buf_ptr, prbu_buf_len);
            //rte_pktmbuf_adj(mbuf, buf_len);
            header_mbuf->data_len  += prbu_buf_len;
            header_mbuf->pkt_len   = header_mbuf->data_len;

            prbu_buf_ptr += prbu_buf_len;

            ecpri_payload += section_buf_len;
            data->ecpri.ecpriPayload = rte_cpu_to_be_16(ecpri_payload);
            update_ecpri_header(data, info.ecpri_hdr_cfg);

            if(remaining_num_prbu != 0)
            {
                pkt_sz = ORAN_IQ_STATIC_OVERHEAD;
                header_mbuf           = header_mbufs[packet_idx++];
                header_mbuf->ol_flags = 0;
                data                  = rte_pktmbuf_mtod(header_mbuf, PacketHeaderTemplate*);
                memcpy(data, &flow->get_packet_header_template(), sizeof(PacketHeaderTemplate));

                auto umsg_iq_hdr = rte_pktmbuf_mtod_offset(header_mbuf, oran_umsg_iq_hdr*, sizeof(PacketHeaderTemplate));
                memcpy(umsg_iq_hdr, &info.radio_app_hdr, sizeof(oran_umsg_iq_hdr));

                header_mbuf->data_len  = ORAN_IQ_STATIC_OVERHEAD;
                header_mbuf->pkt_len   = header_mbuf->data_len;
                data->ecpri.ecpriSeqid = flow->get_sequence_id_generator_downlink().next();
                ecpri_payload          = ecpri_payload_overhead;
            }
        }
    }

    if(txq_index >= kMaxTxqUplanePerPeer)
    {
        THROW_FH(EFAULT, StringBuilder() << "txq_index " << txq_index << " >= kMaxTxqUplanePerPeer " << kMaxTxqUplanePerPeer);
    }

#if 0
    if((info.tx_window.tx_window_start > last_uplane_tx_ts_on_txq_[txq_index]))
    {
        header_mbufs[0]->ol_flags |= fhi->get_timestamp_mask_();
        *RTE_MBUF_DYNFIELD(header_mbufs[0], fhi->get_timestamp_offset(), uint64_t*) = info.tx_window.tx_window_start;
        last_uplane_tx_ts_on_txq_[txq_index]                                        = info.tx_window.tx_window_start;
    }
#endif
    return packet_idx;
}



size_t Peer::prepare_uplane_message(const UPlaneMsgSendInfo& info,
                                    rte_mbuf**               header_mbufs,
                                    rte_mbuf**               iq_data_mbufs,
                                    int txq_index)
{
    auto flow = static_cast<Flow*>(info.flow);
    auto fhi  = get_fronthaul();

    oran_u_section_uncompressed section_template{};
    section_template.sectionId = info.section_info.section_id;
    section_template.rb        = info.section_info.rb;
    section_template.symInc    = info.section_info.sym_inc;

    // Offset buffer by the starting prb
    auto   prbu_buf_ptr = reinterpret_cast<uint8_t*>(info.section_info.iq_data_buffer);
    size_t prb_cnt      = 0;
    size_t num_packets  = std::max(1UL, (info.section_info.num_prbu + prbs_per_pkt_upl_ - 1) / prbs_per_pkt_upl_);

    for(size_t packet_idx = 0; packet_idx < num_packets; packet_idx++)
    {
        size_t num_prbs     = std::min(info.section_info.num_prbu - prb_cnt, prbs_per_pkt_upl_);
        if(num_prbs > ORAN_MAX_PRB_X_SECTION && num_prbs < info_.max_num_prbs_per_symbol)
        {
            num_prbs = ORAN_MAX_PRB_X_SECTION;
        }
        size_t prbu_buf_len = num_prbs * prb_size_upl_;
        auto   header_mbuf  = header_mbufs[packet_idx];
        auto   iq_data_mbuf = iq_data_mbufs[packet_idx];

        header_mbuf->ol_flags = 0;
        attach_extbuf(iq_data_mbuf, prbu_buf_ptr, prbu_buf_len);

        auto data = rte_pktmbuf_mtod(header_mbuf, PacketHeaderTemplate*);
        memcpy(data, &flow->get_packet_header_template(), sizeof(PacketHeaderTemplate));

        // 4 for ecpriPcid, ecprSeqid, ecpriEbit, ecpriSubSeqid
        constexpr size_t ecpri_payload_overhead = 4 + sizeof(oran_umsg_iq_hdr) + sizeof(oran_u_section_uncompressed);
        header_mbuf->data_len                   = ORAN_IQ_HDR_SZ;
        data->ecpri.ecpriSeqid                  = flow->get_sequence_id_generator_downlink().next();
        data->ecpri.ecpriPayload                = rte_cpu_to_be_16(ecpri_payload_overhead + prbu_buf_len);
        auto umsg_iq_hdr = rte_pktmbuf_mtod_offset(header_mbuf, oran_umsg_iq_hdr*, sizeof(PacketHeaderTemplate));
        memcpy(umsg_iq_hdr, &info.radio_app_hdr, sizeof(oran_umsg_iq_hdr));

        section_template.startPrbu = info.section_info.start_prbu + prb_cnt;
        section_template.numPrbu   = (num_prbs > ORAN_MAX_PRB_X_SECTION && num_prbs == info_.max_num_prbs_per_symbol) ? 0 : num_prbs;
        auto sect                  = static_cast<oran_u_section_uncompressed*>(RTE_PTR_ADD(umsg_iq_hdr, sizeof(*umsg_iq_hdr)));
        memcpy(sect, &section_template, sizeof(oran_u_section_uncompressed));

        header_mbuf->pkt_len   = header_mbuf->data_len;
        iq_data_mbuf->data_len = prbu_buf_len;
        iq_data_mbuf->pkt_len  = iq_data_mbuf->data_len;

        if(unlikely(rte_pktmbuf_chain(header_mbuf, iq_data_mbuf)))
        {
            THROW_FH(EINVAL, "rte_pktmbuf_chain failed");
        }

        //rte_mbuf_sanity_check(hdr, 1);

        prbu_buf_ptr += prbu_buf_len;
        prb_cnt += num_prbs;
    }

#if 0
    if((info.tx_window.tx_window_start > last_uplane_tx_ts_on_txq_[txq_index]))
    {
        header_mbufs[0]->ol_flags |= fhi->get_timestamp_mask_();
        *RTE_MBUF_DYNFIELD(header_mbufs[0], fhi->get_timestamp_offset(), uint64_t*) = info.tx_window.tx_window_start;
        last_uplane_tx_ts_on_txq_[txq_index]                                        = info.tx_window.tx_window_start;
    }
#endif

    return num_packets;
}

size_t Peer::count_uplane_packets(UPlaneMsgSendInfo const* info, size_t num_msgs)
{
    size_t num_packets = 0;

    for(size_t i = 0; i < num_msgs; ++i)
    {
        num_packets += std::max(1UL, (info[i].section_info.num_prbu + prbs_per_pkt_upl_ - 1) / prbs_per_pkt_upl_);
    }

    return num_packets;
}

std::pair<size_t, size_t> Peer::count_uplane_packets(UPlaneMsgMultiSectionSendInfo const* info, bool chained_mbuf)
{
    if(info->section_num == 0)
    {
        return {0, 0};
    }

    size_t mtu_sz             = nic_->get_mtu();
    size_t pkt_sz             = 0;
    size_t remaining_num_prbu = 0;
    size_t section_idx        = 0;
    size_t num_packets        = 0;
    size_t num_iq_buf         = 0;
    size_t max_prbs_for_packet = 0;
    bool first_pkt = true;
    size_t prbs;
    if(info->section_num == 0)
    {
        return {0, 0};
    }

    // initial packet
    pkt_sz = ORAN_IQ_STATIC_OVERHEAD;
    ++num_packets;

    for(section_idx = 0; section_idx < info->section_num; ++section_idx)
    {
        remaining_num_prbu = info->section_infos[section_idx].num_prbu;

        while(remaining_num_prbu > 0)
        {
            max_prbs_for_packet = (mtu_sz - pkt_sz - ORAN_IQ_UNCOMPRESSED_SECTION_OVERHEAD) / prb_size_upl_;
            if(max_prbs_for_packet == 0)
            {
                pkt_sz = ORAN_IQ_STATIC_OVERHEAD;
                ++num_packets;
            }
            max_prbs_for_packet = (mtu_sz - pkt_sz - ORAN_IQ_UNCOMPRESSED_SECTION_OVERHEAD) / prb_size_upl_;
            prbs = std::min(remaining_num_prbu, max_prbs_for_packet);
            remaining_num_prbu -= prbs;
            num_iq_buf++;
            pkt_sz += ORAN_IQ_UNCOMPRESSED_SECTION_OVERHEAD + prbs * prb_size_upl_;
            if(remaining_num_prbu != 0 || (chained_mbuf && num_iq_buf % kMaxULUPSections == 0))
            {
                pkt_sz = ORAN_IQ_STATIC_OVERHEAD;
                ++num_packets;
            }
        }
    }
    return {num_packets, num_iq_buf};
}

void Peer::preallocate_mbufs(TxRequestUplane** tx_request, int num_mbufs)
{
    TxRequestUplane* tx_request_local = *tx_request;
    
    // Only reallocate if needed:
    // - indices are not at start (some mbufs were used), OR
    // - nothing allocated yet, OR
    // - wrong number of mbufs allocated
    if(tx_request_local->prepare_index != 0 || 
       tx_request_local->preallocated_mbuf_count == 0 ||
       tx_request_local->preallocated_mbuf_count != static_cast<size_t>(num_mbufs))
    {
        // Free any existing allocations
        free_preallocated_mbufs(tx_request);

        // Allocate new mbufs
        auto cpu_mbuf_pool = nic_->get_cpu_tx_mbuf_pool();
        if(unlikely(0 != rte_mempool_get_bulk(cpu_mbuf_pool, reinterpret_cast<void**>(tx_request_local->mbufs), num_mbufs)))
        {
            THROW_FH(ENOMEM, StringBuilder() << "Failed to allocate " << num_mbufs << " mbufs for U-plane message");
        }
        tx_request_local->preallocated_mbuf_count = num_mbufs;
        tx_request_local->prepare_index = 0;
        tx_request_local->send_index = 0;
    }
}

void Peer::free_preallocated_mbufs(TxRequestUplane** tx_request)
{
    TxRequestUplane* tx_request_local = *tx_request;
    const size_t num_unused_mbufs = tx_request_local->preallocated_mbuf_count - tx_request_local->prepare_index;
    if(tx_request_local->preallocated_mbuf_count > 0 && num_unused_mbufs > 0)
    {
        rte_pktmbuf_free_bulk(&tx_request_local->mbufs[tx_request_local->prepare_index], num_unused_mbufs);
    }
    tx_request_local->preallocated_mbuf_count = 0;
    tx_request_local->prepare_index = 0;
    tx_request_local->send_index = 0;
}


void Peer::prepare_uplane_with_preallocated_tx_request(UPlaneMsgMultiSectionSendInfo const* info, UPlaneTxCompleteNotification notification, TxRequestUplane** tx_request, int txq_index)
{
    TxRequestUplane* tx_request_local = *tx_request;

    auto [num_packets, _] = count_uplane_packets(info, false);

    if(unlikely(num_packets > kTxPktBurstUplane))
    {
        THROW_FH(EINVAL, StringBuilder() << "Too many U-plane packets to send: " << num_packets << ". Please raise kTxPktBurstUplane!");
    }

    if(likely(num_packets != 0))
    {
        const size_t available_space = tx_request_local->preallocated_mbuf_count - tx_request_local->prepare_index;
        if(available_space < num_packets)
        {
            preallocate_mbufs(tx_request, num_packets);
        }

        size_t created_pkts = 0;
        created_pkts = this->prepare_uplane_message_v3(*info, &tx_request_local->mbufs[tx_request_local->prepare_index], txq_index);

        tx_request_local->mbufs[tx_request_local->prepare_index + created_pkts - 1]->shinfo->free_cb    = notification.callback;
        tx_request_local->mbufs[tx_request_local->prepare_index + created_pkts - 1]->shinfo->fcb_opaque = notification.callback_arg;
        
        // Advance prepare index for next batch
        tx_request_local->prepare_index += num_packets;
    }

    tx_request_local->tx_window_start = info->tx_window.tx_window_start;
}

size_t Peer::prepare_uplane_count_packets(UPlaneMsgMultiSectionSendInfo const* info)
{
    auto [num_packets, _] = count_uplane_packets(info, false);
    return num_packets;
}

size_t Peer::prepare_cplane_count_packets(CPlaneMsgSendInfo const* info, size_t num_msgs)
{
    auto num_packets = count_cplane_packets(info, num_msgs);
    return num_packets;
}



void Peer::prepare_uplane(UPlaneMsgSendInfo const* info, size_t num_msgs, UPlaneTxCompleteNotification notification, TxRequestUplane** tx_request, int txq_index)
{
    TxRequestUplane* tx_request_local = nullptr;

    auto num_packets = count_uplane_packets(info, num_msgs);

    if(unlikely(num_packets > kTxPktBurstUplane))
    {
        THROW_FH(EINVAL, StringBuilder() << "Too many U-plane packets to send: " << num_packets << ". Please raise kTxPktBurstUplane!");
    }

    auto tx_request_mempool = nic_->get_tx_request_pool();
    if(unlikely(0 != rte_mempool_get(tx_request_mempool, reinterpret_cast<void**>(&tx_request_local))))
    {
        THROW_FH(ENOMEM, "Ran out of TxRequestUplane descriptors");
    }
    
    // Initialize from pool (may contain garbage from previous usage)
    tx_request_local->preallocated_mbuf_count = 0;
    tx_request_local->prepare_index = 0;
    tx_request_local->send_index = 0;

    if(likely(num_packets != 0))
    {
        // Allocate fresh mbufs (preallocate_mbufs will see count==0 and allocate)
        preallocate_mbufs(tx_request, num_packets);
        
        auto cpu_mbuf_pool = nic_->get_cpu_tx_mbuf_pool();
        rte_mbuf* iq_data_mbufs[kTxPktBurstUplane];
        if(unlikely(0 != rte_mempool_get_bulk(cpu_mbuf_pool, reinterpret_cast<void**>(iq_data_mbufs), num_packets)))
        {
            // Free the header mbufs we just allocated
            free_preallocated_mbufs(tx_request);
            rte_mempool_put(tx_request_mempool, tx_request_local);
            THROW_FH(ENOMEM, StringBuilder() << "Failed to allocate " << num_packets << " extbuf mbufs for " << num_msgs << " U-plane messages");
        }

        size_t created_pkts = 0;
        for(size_t i = 0; i < num_msgs; ++i)
        {
            created_pkts += this->prepare_uplane_message(info[i], &tx_request_local->mbufs[created_pkts], &iq_data_mbufs[created_pkts], txq_index);
        }

        iq_data_mbufs[num_packets - 1]->shinfo->free_cb    = notification.callback;
        iq_data_mbufs[num_packets - 1]->shinfo->fcb_opaque = notification.callback_arg;
        
        // Update indices: send from 0, prepare advanced by num_packets
        tx_request_local->prepare_index = num_packets;
    }

    tx_request_local->peer       = this;
    *tx_request                  = tx_request_local;
}

void Peer::gpu_comm_update_tx_metrics(TxRequestUplaneGpuComm* tx_request)
{
    PartialUplaneSlotInfo_t* partial_up_slot_info = tx_request->partial_slot_info;
    uint32_t tx_packets = 0, tx_bytes = 0;
    for(uint16_t symbol_id = 0; symbol_id < kPeerSymbolsInfo; symbol_id++)
    {
        int current_messages = partial_up_slot_info->section_info[symbol_id].num_messages;
        for(int i = 0; i < current_messages; i++)
        {
            tx_packets += partial_up_slot_info->message_info[symbol_id].num_packets[i];
            tx_bytes += partial_up_slot_info->message_info[symbol_id].num_bytes[i];
        }
    }
    update_tx_metrics(tx_packets, tx_bytes);
}

void Peer::gpu_comm_prepare_uplane(UPlaneMsgSendInfo const* info, size_t num_msgs,
                                    TxRequestUplaneGpuComm** tx_request,
                                    std::chrono::nanoseconds cell_start_time, std::chrono::nanoseconds symbol_duration,bool commViaCpu)
{

    int up_slot_idx = gpu_comm_get_next_up_slot_idx();
//    NVLOGC_FMT(TAG, "gpu_comm_prepare_uplane start for slot {} with num_msgs {}", up_slot_idx, (int)num_msgs);
    UplaneSlotInfo_t* d_slot_info = &(d_up_slot_info_[up_slot_idx]);
    PartialUplaneSlotInfo_t* partial_up_slot_info = &(partial_up_slot_info_[up_slot_idx]);

    auto port_id = nic_->get_port_id();
    int buffer_slot_offset = up_slot_idx * MAX_DL_EAXCIDS;
    uint32_t index;
    uint32_t syms_with_packets_mask = 0;
    uint16_t tmp_packets = 0;
    (*tx_request) = &(up_tx_request_[up_slot_idx]);
    (*tx_request)->txq = get_next_uplane_txq_gpu();
    (*tx_request)->prb_size_upl = prb_size_upl_;
    (*tx_request)->prbs_per_pkt_upl = prbs_per_pkt_upl_;
    (*tx_request)->frame_id = info[0].radio_app_hdr.frameId;
    (*tx_request)->subframe_id = info[0].radio_app_hdr.subframeId.get();
    (*tx_request)->slot_id = info[0].radio_app_hdr.slotId.get();
    (*tx_request)->flow_d_info = d_flow_pkt_hdr_index_; //FIXME  + buffer_slot_offset; Confirm
    (*tx_request)->flow_sym_d_info = d_flow_sym_pkt_hdr_index_; //FIXME  + buffer_slot_offset; Confirm
    (*tx_request)->block_count = d_block_count_; //FIXME  + buffer_slot_offset; Confirm
    (*tx_request)->flow_hdr_size_info = h_flow_hdr_size_info_ + buffer_slot_offset;
    (*tx_request)->flow_d_ecpri_seq_id = d_ecpri_seq_id_;//Remove '+ buffer_slot_offset' as the same sequence generator (buffer) has to be used for all slots for a given eaxcid
    (*tx_request)->flow_d_hdr_template_info = d_hdr_template_ + 8*buffer_slot_offset;
    partial_up_slot_info->last_sym_with_packets = 0;
    partial_up_slot_info->syms_with_packets = 0;

    {
        MemtraceDisableScope mds; // something in DOCA is allocating memory.
        //Confirmed that calling rte_pmd_mldx5* does not have any problematic side-effects if there are no messages for that symbol
        for (uint16_t symbol_id = 0; symbol_id < kPeerSymbolsInfo; symbol_id++)
        {
            partial_up_slot_info->section_info[symbol_id].ptp_ts=cell_start_time.count() + symbol_id * symbol_duration.count();
            doca_eth_txq_calculate_timestamp((*tx_request)->txq->get_doca_tx_items()->eth_txq_cpu, cell_start_time.count() + symbol_id * symbol_duration.count(),
                                        &(partial_up_slot_info->section_info[symbol_id].ts));

            partial_up_slot_info->section_info[symbol_id].num_messages = 0; // reset for later
            partial_up_slot_info->section_info[symbol_id].num_packets = 0;
        }
        if(commViaCpu)
        {
            memset(&partial_up_slot_info->flowInfo_slot.cumulative_sym_flow_packet_count, 0, sizeof(partial_up_slot_info->flowInfo_slot.cumulative_sym_flow_packet_count));
            memset(&partial_up_slot_info->flowInfo_slot.sym_flow_packet_count, 0, sizeof(partial_up_slot_info->flowInfo_slot.sym_flow_packet_count));
        }
    }

   // Update the frame/subframe/slot information from radio_app_hdr
   // Currently same info replicated across all cells in a slot even though identical
   partial_up_slot_info->frame_8b_subframe_4b_slot_6b = info[0].radio_app_hdr.frameId << 10; // 10 is 4 for subframe and 6 bits for slot
   partial_up_slot_info->frame_8b_subframe_4b_slot_6b |= (info[0].radio_app_hdr.subframeId.get() << 6);
   partial_up_slot_info->frame_8b_subframe_4b_slot_6b |= info[0].radio_app_hdr.slotId.get();

    if(commViaCpu)
    {
        //Reset total packets count and other arrays before update in the loop
        partial_up_slot_info->ttl_pkts=0;
        partial_up_slot_info->flowInfo_slot.flow_eaxcid.fill(-1);
        partial_up_slot_info->flowInfo_slot.flow_packet_count.fill(0);
        partial_up_slot_info->flowInfo_slot.num_flows=0;
    }

   // section information needed per message
   for(size_t i = 0; i < num_msgs; ++i) //num_msgs is messages across all symbols for this cell
   {
       uint16_t msgi_num_prbu = info[i].section_info.num_prbu;
       uint16_t msgi_start_prbu = info[i].section_info.start_prbu;
       uint16_t msgi_section_id = info[i].section_info.section_id;
       uint16_t msgi_section_prb_size = info[i].section_info.prb_size;
       uint8_t msgi_rb = info[i].section_info.rb;

       uint16_t symbol_id = (uint16_t)info[i].radio_app_hdr.symbolId.get();
       if(symbol_id >= kPeerSymbolsInfo)
       {
           THROW_FH(ENOMEM, StringBuilder() << "Symbol ID in radio app hdr set greater than or equal to " << kPeerSymbolsInfo);
       }

       partial_up_slot_info->last_sym_with_packets = std::max(partial_up_slot_info->last_sym_with_packets, symbol_id);
       syms_with_packets_mask |= (1U << symbol_id);

       int current_messages                                                       = partial_up_slot_info->section_info[symbol_id].num_messages;
       partial_up_slot_info->message_info[symbol_id].num_prbu[current_messages]   = msgi_num_prbu;
       partial_up_slot_info->message_info[symbol_id].start_prbu[current_messages] = msgi_start_prbu;
       partial_up_slot_info->message_info[symbol_id].section_id[current_messages] = msgi_section_id;
       partial_up_slot_info->message_info[symbol_id].rb[current_messages]         = msgi_rb;
       if(info[i].section_info.mod_comp_enable)
       {
           partial_up_slot_info->message_info[symbol_id].mod_comp_params->prb_size_upl[current_messages]     = msgi_section_prb_size;
           partial_up_slot_info->message_info[symbol_id].mod_comp_params->mod_comp_enabled[current_messages] = 1;
       }
       else
       {
           partial_up_slot_info->message_info[symbol_id].mod_comp_params = nullptr;
       }
       if(unlikely(dlu_eaxcid_idx_mp[info[i].eaxcid] >= MAX_DL_EAXCIDS))
       {
           THROW_FH(ENOMEM, StringBuilder() << "eaxcid idx " << dlu_eaxcid_idx_mp[info[i].eaxcid] << " out of bounds, max " << MAX_DL_EAXCIDS - 1);
       }
       partial_up_slot_info->message_info[symbol_id].flow_index_info[current_messages] = dlu_eaxcid_idx_mp[info[i].eaxcid];
       partial_up_slot_info->section_info[symbol_id].num_messages += 1;


       if(info[i].section_info.mod_comp_enable)
       {
           uint16_t prb_per_pkt_upl                                                    = (nic_->get_mtu() - ORAN_IQ_HDR_SZ) / msgi_section_prb_size;
           tmp_packets                                                                 = (msgi_num_prbu + prb_per_pkt_upl - 1) / prb_per_pkt_upl;
           partial_up_slot_info->message_info[symbol_id].num_packets[current_messages] = tmp_packets;
           partial_up_slot_info->message_info[symbol_id].num_bytes[current_messages]   = static_cast<uint32_t>(tmp_packets) * ORAN_IQ_HDR_SZ + static_cast<uint32_t>(msgi_num_prbu) * msgi_section_prb_size;
       }
       else
       {
           tmp_packets                                                                 = (msgi_num_prbu + prbs_per_pkt_upl_ - 1) / prbs_per_pkt_upl_;
           partial_up_slot_info->message_info[symbol_id].num_packets[current_messages] = tmp_packets;
           partial_up_slot_info->message_info[symbol_id].num_bytes[current_messages]   = static_cast<uint32_t>(tmp_packets) * ORAN_IQ_HDR_SZ + static_cast<uint32_t>(msgi_num_prbu) * prb_size_upl_;
       }

       if(commViaCpu)
       {
            partial_up_slot_info->ttl_pkts += tmp_packets;
            partial_up_slot_info->section_info[symbol_id].num_packets += tmp_packets;

            auto it=std::find(partial_up_slot_info->flowInfo_slot.flow_eaxcid.begin(),partial_up_slot_info->flowInfo_slot.flow_eaxcid.end(),info[i].eaxcid);
            if(it!=partial_up_slot_info->flowInfo_slot.flow_eaxcid.end())
            {
                index=std::distance(partial_up_slot_info->flowInfo_slot.flow_eaxcid.begin(),it);
                partial_up_slot_info->flowInfo_slot.flow_packet_count[index]+=tmp_packets;
                partial_up_slot_info->flowInfo_slot.sym_flow_packet_count[dlu_eaxcid_idx_mp[info[i].eaxcid]][symbol_id] += tmp_packets;
            }
            else
            {
                partial_up_slot_info->flowInfo_slot.flow_eaxcid[partial_up_slot_info->flowInfo_slot.num_flows]=info[i].eaxcid;
                partial_up_slot_info->flowInfo_slot.flow_packet_count[partial_up_slot_info->flowInfo_slot.num_flows]+=tmp_packets;
                partial_up_slot_info->flowInfo_slot.sym_flow_packet_count[dlu_eaxcid_idx_mp[info[i].eaxcid]][symbol_id] += tmp_packets;
                partial_up_slot_info->flowInfo_slot.num_flows++;
            }
       }
#if 0
if (tmp_packets != 1)
        printf("msg %d, symbol %d, packets %d\n", i, symbol_id, std::max(1UL, (info[i].section_info.num_prbu + prbs_per_pkt_upl_ - 1) / prbs_per_pkt_upl_));
#endif
   }

    if(commViaCpu)
    {
        partial_up_slot_info->syms_with_packets = __builtin_popcount(syms_with_packets_mask);
        partial_up_slot_info->total_num_flows=getTotalNumFlows();
        NVLOGI_FMT(TAG,"F{}S{}S{} Total packets={} Total num flows={} partial_up_slot_info->syms_with_packets={}",
            (*tx_request)->frame_id,(*tx_request)->subframe_id,(*tx_request)->slot_id,partial_up_slot_info->ttl_pkts,partial_up_slot_info->total_num_flows, partial_up_slot_info->syms_with_packets);
        for(int index=0;index<partial_up_slot_info->flowInfo_slot.num_flows;index++)
        {
            //printf("index %d\n", index);
            for (int sym = 1; sym < 14; sym++) {
                partial_up_slot_info->flowInfo_slot.cumulative_sym_flow_packet_count[index][sym] = partial_up_slot_info->flowInfo_slot.cumulative_sym_flow_packet_count[index][sym - 1] + partial_up_slot_info->flowInfo_slot.sym_flow_packet_count[index][sym-1];
                NVLOGI_FMT(TAG,"Partial packets {} {} {}", index, sym, partial_up_slot_info->flowInfo_slot.cumulative_sym_flow_packet_count[index][sym]);
            }
            NVLOGI_FMT(TAG,"F{}S{}S{} eaxcid={} packets_per_flow={}",(*tx_request)->frame_id,(*tx_request)->subframe_id,(*tx_request)->slot_id,partial_up_slot_info->flowInfo_slot.flow_eaxcid[index],partial_up_slot_info->flowInfo_slot.flow_packet_count[index]);
        }
    }
    // this is how info is communicated
    (*tx_request)->partial_slot_info = &(partial_up_slot_info_[up_slot_idx]);
    (*tx_request)->d_slot_info = &(d_up_slot_info_[up_slot_idx]);
    (*tx_request)->h_up_slot_info_ = &(h_up_slot_info_[up_slot_idx]);

}

size_t Peer::send_uplane(TxRequestUplane* tx_request, Txq* txq)
{
    const size_t batch_count = tx_request->prepare_index - tx_request->send_index;
    auto txq_to_use = txq ? txq : this->get_next_uplane_txq();
    if(!(get_fronthaul()->get_info().cuda_device_ids.empty()))
    {
        txq_to_use->send(&tx_request->mbufs[tx_request->send_index], batch_count);
    }
    else
    {
        txq_to_use->send_lock(&tx_request->mbufs[tx_request->send_index], batch_count);
    }
    rte_mempool_put(nic_->get_tx_request_pool(), tx_request);

#ifdef AERIAL_METRICS
    size_t tx_bytes = sum_up_tx_bytes(&tx_request->mbufs[tx_request->send_index], batch_count);
    update_tx_metrics(batch_count, tx_bytes);
#endif

    return batch_count;
}

size_t Peer::send_uplane_without_freeing_tx_request(TxRequestUplane* tx_request, Txq* txq, TxqSendTiming* timing)
{
    const size_t batch_count = tx_request->prepare_index - tx_request->send_index;
    auto txq_to_use = txq ? txq : this->get_next_uplane_txq();
    if(!(get_fronthaul()->get_info().cuda_device_ids.empty()))
    {
        txq_to_use->send(&tx_request->mbufs[tx_request->send_index], batch_count, timing);
    }
    else
    {
        //txq_to_use->send_lock(&tx_request->mbufs[tx_request->send_index], batch_count);
        txq_to_use->send_lock(&tx_request->mbufs[tx_request->send_index], batch_count, tx_request->tx_window_start, timing);
    }
    // After send, update send_index to point to where prepare_index currently is
    tx_request->send_index = tx_request->prepare_index;

    return batch_count;
}

void Peer::alloc_tx_request(TxRequestUplane** tx_request)
{
    TxRequestUplane* tx_request_local = nullptr;
    auto tx_request_mempool = nic_->get_tx_request_pool();
    if(unlikely(0 != rte_mempool_get(tx_request_mempool, reinterpret_cast<void**>(&tx_request_local))))
    {
        THROW_FH(ENOMEM, "Ran out of TxRequestUplane descriptors");
    }
    tx_request_local->preallocated_mbuf_count = 0;
    tx_request_local->prepare_index = 0;
    tx_request_local->send_index = 0;
    tx_request_local->peer       = this;
    *tx_request = tx_request_local;
}

void Peer::free_tx_request(TxRequestUplane* tx_request)
{
    rte_mempool_put(nic_->get_tx_request_pool(), tx_request);
}

void Peer::receive(MsgReceiveInfo* info, size_t* num_msgs, bool srs)
{
    if(srs)
    {
        if(unlikely(rxqSrs_ == nullptr))
        {
            THROW_FH(EINVAL, StringBuilder() << "Calling receive() but no RXQ assigned to Peer " << info_.dst_mac_addr);
        }

        size_t rx_bytes = rxqSrs_->receive(info, num_msgs);
        update_rx_metrics(*num_msgs, rx_bytes);
    }
    else
    {
        if(unlikely(rxq_ == nullptr))
        {
            THROW_FH(EINVAL, StringBuilder() << "Calling receive() but no RXQ assigned to Peer " << info_.dst_mac_addr);
        }

        size_t rx_bytes = rxq_->receive(info, num_msgs);
        update_rx_metrics(*num_msgs, rx_bytes);
    }

}

void Peer::receive_until(MsgReceiveInfo* info, size_t* num_msgs, Ns timeout)
{
    if(unlikely(rxq_ == nullptr))
    {
        THROW_FH(EINVAL, StringBuilder() << "Calling receive_until() but no RXQ assigned to Peer " << info_.dst_mac_addr);
    }

    size_t rx_bytes = rxq_->receive_until(info, num_msgs, timeout);
    update_rx_metrics(*num_msgs, rx_bytes);
}

void Peer::update_rx_metrics(size_t rx_packets, size_t rx_bytes)
{
    metrics_.update(PeerMetric::kUPlaneRxPackets, rx_packets);
    metrics_.update(PeerMetric::kUPlaneRxBytes, rx_bytes);
}

void Peer::update_tx_metrics(size_t tx_packets, size_t tx_bytes)
{
    metrics_.update(PeerMetric::kUPlaneTxPackets, tx_packets);
    metrics_.update(PeerMetric::kUPlaneTxBytes, tx_bytes);
}

std::string Peer::get_mac_address()
{
    std::string mac_address;
    char        eth_addr[RTE_ETHER_ADDR_LEN * 2];
    int         i = 0, j = 0;

    for(i = 0, j = 0; i < RTE_ETHER_ADDR_LEN * 2 && j < RTE_ETHER_ADDR_LEN; i += 2, j++)
        sprintf(&(eth_addr[i]), "%02x", info_.dst_mac_addr.bytes[j]);
    mac_address = eth_addr;

    return mac_address;
}

FlowPtrInfo* Peer::get_flow_ptr_info()
{
    return h_flow_hdr_size_info_;
}

uint32_t* Peer::get_hdr_template_info()
{
    return d_hdr_template_;
}

} // namespace aerial_fh
