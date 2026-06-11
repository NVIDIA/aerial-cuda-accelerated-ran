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

#include "flow.hpp"

#include "fronthaul.hpp"
#include "nic.hpp"
#include "peer.hpp"
#include "utils.hpp"
#include "memreg.hpp"
#include "ti_generic.hpp"

#define TAG "FH.FLOW"
#define TAG_STARTUP_TIMES (NVLOG_TAG_BASE_CUPHY_CONTROLLER + 5) // "CTL.STARTUP_TIMES"

namespace aerial_fh
{
Flow::Flow(Peer* peer, FlowInfo* info) :
    peer_{peer},
    info_{*info},
    flow_type_{info->type == FlowType::CPLANE ? "C-plane" : "U-plane"}
{
    TI_GENERIC_INIT("Flow::Flow",15);
    TI_GENERIC_ADD("Start Task");

    TI_GENERIC_ADD("peer get_info");
    auto peer_addr = peer_->get_info().dst_mac_addr;
    NVLOGI_FMT(TAG, "Adding {} Flow with eAxC={} and vlan_id={} to Peer {:02X}:{:02X}:{:02X}:{:02X}:{:02X}:{:02X}",
        flow_type_.c_str(), info_.eAxC, static_cast<unsigned short>(info_.vlan_tag.vid), peer_addr.bytes[0], peer_addr.bytes[1],
        peer_addr.bytes[2], peer_addr.bytes[3], peer_addr.bytes[4], peer_addr.bytes[5]);

    if(info->type == FlowType::UPLANE && info_.direction == FlowDir::DL)
    {
        auto flow_num=peer_->getTotalNumFlows();
        flow_num+=1;
        peer_->setTotalNumFlows(flow_num);
    }

    TI_GENERIC_ADD("FH settings check");
    rxq_ = nullptr;
    if(get_fronthaul()->rmax_enabled() && peer_->get_info().rx_mode != RxApiMode::TXONLY)
        THROW_FH(EINVAL, StringBuilder() << "With Rivermax enabled only RxApiMode::TXONLY is allowed.");

    TI_GENERIC_ADD("FH get_info");
    GPU_device_count = 0;
    if(!(get_fronthaul()->get_info().cuda_device_ids.empty()))
    {
        cudaError_t error = cudaGetDeviceCount(&GPU_device_count);
        if (error != cudaSuccess) {
            NVLOGI_FMT(TAG, "cudaGetDeviceCount returned {}", +error);
        }
    }
    flow_number = -1;
    TI_GENERIC_ADD("request_nic_resources");
    if(!(get_fronthaul()->get_info().cuda_device_ids.empty()))
    {
        request_nic_resources();
    }
    else
    {
        if(info->request_new_rxq==true)
        {
            request_nic_resources();
            info->rxq=(void*)rxq_;
        }
        else
        {
            rxq_=(Rxq*)info->rxq;
        }
    }
    TI_GENERIC_ADD("create_rx_rules");
    create_rx_rules();
    TI_GENERIC_ADD("setup_flow_index");
    setup_flow_index();
    TI_GENERIC_ADD("setup_packet_header_template");
    setup_packet_header_template();
    TI_GENERIC_ADD("setup_packet_header_gpu");
    if(info_.direction == FlowDir::DL)
    {
        setup_packet_header_gpu();
    }

    TI_GENERIC_ADD("End Task");
    TI_GENERIC_ALL_NVLOGI(TAG_STARTUP_TIMES);
}

Flow::~Flow()
{
    auto peer_addr = peer_->get_info().dst_mac_addr;
    Gpu* gpu_ = get_nic()->get_fronthaul()->gpus()[get_nic()->get_info().cuda_device].get();

    NVLOGI_FMT(TAG, "Removing {} Flow with eAxC={} and vlan_id={} from Peer {:02X}:{:02X}:{:02X}:{:02X}:{:02X}:{:02X}",
            flow_type_.c_str(), info_.eAxC, static_cast<unsigned short>(info_.vlan_tag.vid), peer_addr.bytes[0], peer_addr.bytes[1],
            peer_addr.bytes[2], peer_addr.bytes[3], peer_addr.bytes[4], peer_addr.bytes[5]);

    // unregister_memory(pkt_hdr_gpu_mr_);
    // rte_gpu_mem_free(gpu_dpdk_id, pkt_hdr_gpu_);
    free_nic_resources();
}

void Flow::update(FlowInfo const* info)
{
    NVLOGI_FMT(TAG, "Updating {} Flow with eAxC={}", flow_type_.c_str(), info_.eAxC);
    // Assume direction of flow does not change w.r.t. direction
    auto tmp = info_.direction;
    info_ = *info;
    info_.direction = tmp;
    rx_flow_rules_.clear();
    create_rx_rules();
    setup_packet_header_template();
    if(info_.direction == FlowDir::DL)
    {
        setup_packet_header_gpu();
    }
}

void Flow::request_nic_resources()
{
    auto  nic           = peer_->get_nic();
    auto  nic_name      = nic->get_name();
    auto& queue_manager = nic->get_queue_manager();
    auto& info          = peer_->get_info();

    try
    {
        if(info.rx_mode == RxApiMode::FLOW || ((info.rx_mode == RxApiMode::HYBRID) && (info_.type == FlowType::UPLANE && info_.flow_rx_mode == FlowRxApiMode::TXANDRX)))
        {
            rxq_ = queue_manager.assign_rxq();
            NVLOGD_FMT(TAG, "{} Flow with eAxC={} is using NIC {} RXQ #{}", flow_type_.c_str(), info_.eAxC, nic_name.c_str(), rxq_->get_id());
        }
    }
    catch(...)
    {
        free_nic_resources();
        throw;
    }
}

void Flow::free_nic_resources()
{
    auto  nic           = peer_->get_nic();
    auto  nic_name      = nic->get_name();
    auto& queue_manager = nic->get_queue_manager();

    if(flow_number != -1)
    {
        NVLOGD_FMT(TAG, "Flow with eAxC={} is returning flow idx #{} to NIC ", info_.eAxC, flow_number);
        nic->free_flow_idx(flow_number);
    }

    if(rxq_)
    {
        NVLOGD_FMT(TAG,"{} Flow with eAxC={} is returning RXQ #{} to NIC ",
            flow_type_.c_str(), info_.eAxC, rxq_->get_id(), nic_name.c_str());
        queue_manager.reclaim(rxq_);
    }
}

Peer* Flow::get_peer() const
{
    return peer_;
}

Nic* Flow::get_nic() const
{
    return this->get_peer()->get_nic();
}

Fronthaul* Flow::get_fronthaul() const
{
    return peer_->get_fronthaul();
}

FlowInfo const& Flow::get_info() const
{
    return info_;
}

void Flow::setup_flow_index()
{
    auto& eaxcid_idx_mp = peer_->get_eaxcid_idx_mp();
    if(eaxcid_idx_mp.find(info_.eAxC) == eaxcid_idx_mp.end())
    {
        auto idx = eaxcid_idx_mp.size();
        eaxcid_idx_mp[info_.eAxC] = idx;
        NVLOGI_FMT(TAG, "Mapping Flow with eAxC={} to idx {}", info_.eAxC, idx);
    }

    if(info_.type == FlowType::UPLANE && info_.direction == FlowDir::DL)
    {
        auto& dlu_eaxcid_idx_mp = peer_->get_dlu_eaxcid_idx_mp();
        if(dlu_eaxcid_idx_mp.find(info_.eAxC) == dlu_eaxcid_idx_mp.end())
        {
            auto idx = dlu_eaxcid_idx_mp.size();
            dlu_eaxcid_idx_mp[info_.eAxC] = idx;
            NVLOGI_FMT(TAG, "Mapping DLU Flow with eAxC={} to idx {}", info_.eAxC, idx);
        }
    }
}

void Flow::setup_packet_header_template()
{
    auto peer_info = peer_->get_info();
    auto nic       = get_nic();
    auto fhi       = get_fronthaul();

    rte_ether_addr src_addr;

    if(rte_eth_macaddr_get(nic->get_port_id(), &src_addr))
    {
        THROW_FH(ENODEV, StringBuilder() << "Could not get NIC ( " << nic->get_name() << ") MAC address");
    }

    memset(&pkt_hdr_template_, 0, sizeof(pkt_hdr_template_));
    memcpy(pkt_hdr_template_.eth.src_addr.addr_bytes, peer_info.src_mac_addr.bytes, RTE_ETHER_ADDR_LEN);
    // dMIMO: a flow may carry a per-flow dst MAC override (e.g. SRS sounded by a
    // secondary PE) so its C-plane reaches that PE over this (primary) peer's TX queue.
    if(info_.has_dst_mac_override)
        memcpy(pkt_hdr_template_.eth.dst_addr.addr_bytes, info_.dst_mac_override, RTE_ETHER_ADDR_LEN);
    else
        memcpy(pkt_hdr_template_.eth.dst_addr.addr_bytes, peer_info.dst_mac_addr.bytes, RTE_ETHER_ADDR_LEN);

    pkt_hdr_template_.eth.ether_type = rte_cpu_to_be_16(RTE_ETHER_TYPE_VLAN);
    pkt_hdr_template_.vlan.vlan_tci  = rte_cpu_to_be_16(info_.vlan_tag.tci);
    pkt_hdr_template_.vlan.eth_proto = rte_cpu_to_be_16(RTE_ETHER_TYPE_ECPRI);

    pkt_hdr_template_.ecpri.ecpriVersion       = RTE_ECPRI_REV_UP_TO_20;
    pkt_hdr_template_.ecpri.ecpriReserved      = 0;
    pkt_hdr_template_.ecpri.ecpriConcatenation = 0;
    pkt_hdr_template_.ecpri.ecpriMessage       = (info_.type == FlowType::CPLANE) ?
                                                     RTE_ECPRI_MSG_TYPE_RTC_CTRL :
                                                     RTE_ECPRI_MSG_TYPE_IQ_DATA;
    pkt_hdr_template_.ecpri.ecpriPcid          = rte_cpu_to_be_16(info_.eAxC);

    pkt_hdr_template_.ecpri.ecpriSubSeqid = 0;
    pkt_hdr_template_.ecpri.ecpriEbit     = 1;

    NVLOGI_FMT(TAG, "Setting up header template for {} Flow {}: "\
                "[Destination MAC: {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}] "\
                "[Source MAC: {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}] "\
                "[outer EtherType: 0x{:04x}] [VLAN TCI: 0x{:04x}] [inner EtherType: 0x{:04x}] "\
                "[eCPRI Protocol Revision: {}] [eCPRI Concatenation Indicator: {}] "\
                "[eCPRI Message Type: 0x{:02x}] [eCPRI eAxC ID: 0x{:04x}] "\
                "[eCPRI Subsequence ID: {}] [eCPRI E-bit: {}]\n",
                flow_type_.c_str(), info_.eAxC,

                static_cast<unsigned char>(pkt_hdr_template_.eth.dst_addr.addr_bytes[0]),
                static_cast<unsigned char>(pkt_hdr_template_.eth.dst_addr.addr_bytes[1]),
                static_cast<unsigned char>(pkt_hdr_template_.eth.dst_addr.addr_bytes[2]),
                static_cast<unsigned char>(pkt_hdr_template_.eth.dst_addr.addr_bytes[3]),
                static_cast<unsigned char>(pkt_hdr_template_.eth.dst_addr.addr_bytes[4]),
                static_cast<unsigned char>(pkt_hdr_template_.eth.dst_addr.addr_bytes[5]),
                static_cast<unsigned char>(pkt_hdr_template_.eth.src_addr.addr_bytes[0]),
                static_cast<unsigned char>(pkt_hdr_template_.eth.src_addr.addr_bytes[1]),
                static_cast<unsigned char>(pkt_hdr_template_.eth.src_addr.addr_bytes[2]),
                static_cast<unsigned char>(pkt_hdr_template_.eth.src_addr.addr_bytes[3]),
                static_cast<unsigned char>(pkt_hdr_template_.eth.src_addr.addr_bytes[4]),
                static_cast<unsigned char>(pkt_hdr_template_.eth.src_addr.addr_bytes[5]),

                rte_be_to_cpu_16(pkt_hdr_template_.eth.ether_type),
                rte_be_to_cpu_16(pkt_hdr_template_.vlan.vlan_tci),
                rte_be_to_cpu_16(pkt_hdr_template_.vlan.eth_proto),
                RTE_ECPRI_REV_UP_TO_20,
                0,
                pkt_hdr_template_.ecpri.ecpriMessage,
                rte_be_to_cpu_16(pkt_hdr_template_.ecpri.ecpriPcid),
                0,
                1);

    if ((GPU_device_count > 0) && (info_.type == FlowType::UPLANE) && info_.direction == FlowDir::DL)  { // Only update the following for DL U-plane
       int eaxcid = info_.eAxC;
       auto& dlu_eaxcid_idx_mp = peer_->get_dlu_eaxcid_idx_mp();
       int eaxcid_idx = dlu_eaxcid_idx_mp[info_.eAxC];
       if(eaxcid_idx >= MAX_DL_EAXCIDS)
       {
           NVLOGE_FMT(TAG, AERIAL_ORAN_FH_EVENT, "Exceeding max excid number {}", eaxcid_idx, MAX_DL_EAXCIDS);
       }
       for (int i = 0; i < kPeerSlotsInfo; i++) {
           uint32_t* d_hdr_template_info = peer_->get_hdr_template_info() + (i * MAX_DL_EAXCIDS + eaxcid_idx)*8;
           ASSERT_CUDA_FH(cudaMemcpy(d_hdr_template_info, (char*)get_header_template() + 4, 32, cudaMemcpyHostToDevice)); // other synchronous calls elsewhere too at this time
           /*printf("peer %s, eaxcid %d, peer's hdr_info %p, d_hdr_template_info %p\n", peer_->get_mac_address().c_str(), eaxcid, peer_->get_hdr_template_info(), d_hdr_template_info);
           if (i == 0) {
               for (int j = 0; j < 32; j++)
               printf("peer %s, eaxcid %d, header[%d] == %x\n", peer_->get_mac_address().c_str(), eaxcid, j, (*(((char*)get_header_template()) + 4 + j) & 0xFF));
           }*/
       }
    }

}

PacketHeaderTemplate& Flow::get_packet_header_template()
{
    return pkt_hdr_template_;
}

SequenceIdGenerator& Flow::get_sequence_id_generator_uplink()
{
    return ecpriSeqId_generator_uplink_;
}

SequenceIdGenerator& Flow::get_sequence_id_generator_downlink()
{
    return ecpriSeqId_generator_downlink_;
}

void Flow::setup_packet_header_gpu()
{
    TI_GENERIC_INIT("setup_packet_header_gpu",15);
    TI_GENERIC_ADD("Start Task");
    constexpr int total_packets_across_all_flows = kGpuCommSendPeers*kMaxFlows*kMaxPktsFlow;

    TI_GENERIC_ADD("nic get_info");
    if(get_nic()->get_info().cuda_device < 0)
        return;

    if(info_.type == FlowType::CPLANE)
        return;

    TI_GENERIC_ADD("packet size calculations");
    auto mtu = get_nic()->get_mtu();
    packet_size_rnd = ((mtu + pageSizeAlign - 1) / pageSizeAlign) * pageSizeAlign;
    size_t packet_size_rnd_local = packet_size_rnd;
    if (rte_is_power_of_2(packet_size_rnd_local) == 0)
        packet_size_rnd_local = rte_align32pow2(packet_size_rnd_local);

    uint8_t *aggr_pkt_hdr_gpu_ = nullptr;
    pkt_hdr_gpu_lkey_ = 0;
    pkt_header_idx_   = 0;
    pkt_hdr_gpu_      = nullptr;
    TI_GENERIC_ADD("dpdk calculations");
    aggr_pkt_hdr_gpu_ = (uint8_t*)(get_nic()->get_flow_comm_buf()->gpu_pkt_addr);

    TI_GENERIC_ADD("allocate_memory");
    uint8_t * pkt_hdr_cpu = (uint8_t *) allocate_memory(kMaxPktsFlow * packet_size_rnd_local, pageSizeAlign);
    TI_GENERIC_ADD("get_packet_header_template");
    PacketHeaderTemplate pkt_hdr_template = get_packet_header_template();
    TI_GENERIC_ADD("packet set loop");
    for (int idx = 0; idx < kMaxPktsFlow; idx++) {
        ((struct oran_umsg_hdrs*)(&pkt_hdr_cpu[idx * packet_size_rnd_local]))->ethvlan.eth_hdr.src_addr.addr_bytes[0] = pkt_hdr_template.eth.src_addr.addr_bytes[0];
        ((struct oran_umsg_hdrs*)(&pkt_hdr_cpu[idx * packet_size_rnd_local]))->ethvlan.eth_hdr.src_addr.addr_bytes[1] = pkt_hdr_template.eth.src_addr.addr_bytes[1];
        ((struct oran_umsg_hdrs*)(&pkt_hdr_cpu[idx * packet_size_rnd_local]))->ethvlan.eth_hdr.src_addr.addr_bytes[2] = pkt_hdr_template.eth.src_addr.addr_bytes[2];
        ((struct oran_umsg_hdrs*)(&pkt_hdr_cpu[idx * packet_size_rnd_local]))->ethvlan.eth_hdr.src_addr.addr_bytes[3] = pkt_hdr_template.eth.src_addr.addr_bytes[3];
        ((struct oran_umsg_hdrs*)(&pkt_hdr_cpu[idx * packet_size_rnd_local]))->ethvlan.eth_hdr.src_addr.addr_bytes[4] = pkt_hdr_template.eth.src_addr.addr_bytes[4];
        ((struct oran_umsg_hdrs*)(&pkt_hdr_cpu[idx * packet_size_rnd_local]))->ethvlan.eth_hdr.src_addr.addr_bytes[5] = pkt_hdr_template.eth.src_addr.addr_bytes[5];

        ((struct oran_umsg_hdrs*)(&pkt_hdr_cpu[idx * packet_size_rnd_local]))->ethvlan.eth_hdr.dst_addr.addr_bytes[0] = pkt_hdr_template.eth.dst_addr.addr_bytes[0];
        ((struct oran_umsg_hdrs*)(&pkt_hdr_cpu[idx * packet_size_rnd_local]))->ethvlan.eth_hdr.dst_addr.addr_bytes[1] = pkt_hdr_template.eth.dst_addr.addr_bytes[1];
        ((struct oran_umsg_hdrs*)(&pkt_hdr_cpu[idx * packet_size_rnd_local]))->ethvlan.eth_hdr.dst_addr.addr_bytes[2] = pkt_hdr_template.eth.dst_addr.addr_bytes[2];
        ((struct oran_umsg_hdrs*)(&pkt_hdr_cpu[idx * packet_size_rnd_local]))->ethvlan.eth_hdr.dst_addr.addr_bytes[3] = pkt_hdr_template.eth.dst_addr.addr_bytes[3];
        ((struct oran_umsg_hdrs*)(&pkt_hdr_cpu[idx * packet_size_rnd_local]))->ethvlan.eth_hdr.dst_addr.addr_bytes[4] = pkt_hdr_template.eth.dst_addr.addr_bytes[4];
        ((struct oran_umsg_hdrs*)(&pkt_hdr_cpu[idx * packet_size_rnd_local]))->ethvlan.eth_hdr.dst_addr.addr_bytes[5] = pkt_hdr_template.eth.dst_addr.addr_bytes[5];

        ((struct oran_umsg_hdrs*)(&pkt_hdr_cpu[idx * packet_size_rnd_local]))->ethvlan.eth_hdr.ether_type = pkt_hdr_template.eth.ether_type;
        ((struct oran_umsg_hdrs*)(&pkt_hdr_cpu[idx * packet_size_rnd_local]))->ethvlan.eth_hdr.ether_type = pkt_hdr_template.eth.ether_type;
        ((struct oran_umsg_hdrs*)(&pkt_hdr_cpu[idx * packet_size_rnd_local]))->ethvlan.vlan_hdr.vlan_tci = pkt_hdr_template.vlan.vlan_tci;
        ((struct oran_umsg_hdrs*)(&pkt_hdr_cpu[idx * packet_size_rnd_local]))->ethvlan.vlan_hdr.eth_proto = pkt_hdr_template.vlan.eth_proto;

        ((struct oran_umsg_hdrs*)(&pkt_hdr_cpu[idx * packet_size_rnd_local]))->ecpri = pkt_hdr_template.ecpri;

        ((struct oran_umsg_hdrs*)(&pkt_hdr_cpu[idx * packet_size_rnd_local]))->iq_hdr.dataDirection = DIRECTION_DOWNLINK;
        ((struct oran_umsg_hdrs*)(&pkt_hdr_cpu[idx * packet_size_rnd_local]))->iq_hdr.payloadVersion = ORAN_DEF_PAYLOAD_VERSION;
        ((struct oran_umsg_hdrs*)(&pkt_hdr_cpu[idx * packet_size_rnd_local]))->iq_hdr.filterIndex = 0;

        ((struct oran_umsg_hdrs*)(&pkt_hdr_cpu[idx * packet_size_rnd_local]))->sec_hdr.rb = 0;
        ((struct oran_umsg_hdrs*)(&pkt_hdr_cpu[idx * packet_size_rnd_local]))->sec_hdr.symInc = 0;
    }

    if (flow_number == -1) {
        flow_number = get_nic()->get_nxt_flow_idx();
    }
    pkt_hdr_gpu_ = aggr_pkt_hdr_gpu_ + flow_number * kMaxPktsFlow * packet_size_rnd_local;
    // NVLOGC_FMT(TAG, "Cell {} flow number {} [Destination MAC: {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}] [Source MAC: {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}] Offset {} pkt_hdr_gpu_ {}",
    //             peer_->get_id(), flow_number,
    //             static_cast<unsigned char>(pkt_hdr_template_.eth.dst_addr.addr_bytes[0]),
    //             static_cast<unsigned char>(pkt_hdr_template_.eth.dst_addr.addr_bytes[1]),
    //             static_cast<unsigned char>(pkt_hdr_template_.eth.dst_addr.addr_bytes[2]),
    //             static_cast<unsigned char>(pkt_hdr_template_.eth.dst_addr.addr_bytes[3]),
    //             static_cast<unsigned char>(pkt_hdr_template_.eth.dst_addr.addr_bytes[4]),
    //             static_cast<unsigned char>(pkt_hdr_template_.eth.dst_addr.addr_bytes[5]),
    //             static_cast<unsigned char>(pkt_hdr_template_.eth.src_addr.addr_bytes[0]),
    //             static_cast<unsigned char>(pkt_hdr_template_.eth.src_addr.addr_bytes[1]),
    //             static_cast<unsigned char>(pkt_hdr_template_.eth.src_addr.addr_bytes[2]),
    //             static_cast<unsigned char>(pkt_hdr_template_.eth.src_addr.addr_bytes[3]),
    //             static_cast<unsigned char>(pkt_hdr_template_.eth.src_addr.addr_bytes[4]),
    //             static_cast<unsigned char>(pkt_hdr_template_.eth.src_addr.addr_bytes[5]),
    //             flow_number * kMaxPktsFlow * packet_size_rnd_local, (void*)pkt_hdr_gpu_
    //             );

    TI_GENERIC_ADD("cudaMemcpy");
    if (GPU_device_count > 0) {
        ASSERT_CUDA_FH(cudaMemcpy(pkt_hdr_gpu_, pkt_hdr_cpu, kMaxPktsFlow * packet_size_rnd_local, cudaMemcpyDefault));
    }
    TI_GENERIC_ADD("free_memory");
    free_memory(pkt_hdr_cpu);

    TI_GENERIC_ADD("log print");
    NVLOGI_FMT(TAG, "Registering total packets {} packet_size_rnd {} packet_size_rnd_local {} pkts_per_flow {} page size {} flow num {} aggr_ptr {} flow_ptr {}",
               total_packets_across_all_flows, packet_size_rnd, packet_size_rnd_local, kMaxPktsFlow, pageSizeAlign, flow_number, reinterpret_cast<void*>(aggr_pkt_hdr_gpu_), reinterpret_cast<void*>(pkt_hdr_gpu_));

    TI_GENERIC_ADD("eaxcid mapping");
    if (info_.type == FlowType::UPLANE && info_.direction == FlowDir::DL)  { // Only update the following for U-plane
       // Update pkt_hdr_gpu_  and lkey values in the peer's h_flow_hdr_size_info_ peer slots * eaxcids buffer
       int eaxcid = info_.eAxC;
       auto& dlu_eaxcid_idx_mp = peer_->get_dlu_eaxcid_idx_mp();
       int eaxcid_idx = dlu_eaxcid_idx_mp[info_.eAxC];
       for (int i = 0; i < kPeerSlotsInfo; i++) {
            FlowPtrInfo* flow_ptr_info = peer_->get_flow_ptr_info() + i * MAX_DL_EAXCIDS + eaxcid_idx; // host pinned memory
            flow_ptr_info->hdr_stride = flow_number*kMaxPktsFlow;
            flow_ptr_info->pkt_buff_mkey = get_nic()->get_flow_comm_buf()->pkt_buff_mkey;
            flow_ptr_info->cpu_comms_pkt_buff_mkey = get_nic()->get_flow_comm_buf()->cpu_comms_pkt_buff_mkey;
            flow_ptr_info->max_pkt_sz = get_nic()->get_flow_comm_buf()->max_pkt_sz;
            flow_ptr_info->gpu_pkt_addr = get_nic()->get_flow_comm_buf()->gpu_pkt_addr;
            flow_ptr_info->cpu_pkt_addr = get_nic()->get_flow_comm_buf()->cpu_comms_cpu_pkt_addr;            
           //printf("Flow %p: eaxcid %d, peerslot %d, flow_ptr_info %p, pkt_hdr_gpu %p, rnd %d lkey %u\n", this, eaxcid, i, flow_ptr_info, pkt_hdr_gpu_, packet_size_rnd, pkt_hdr_gpu_lkey_);
        }
     }

     TI_GENERIC_ADD("End Task");
     TI_GENERIC_ALL_NVLOGI(TAG_STARTUP_TIMES);
}

uint32_t Flow::get_next_header_stride_gpu(uint32_t num_pkts)
{
    uint32_t tmp = 0;

    const std::lock_guard<aerial_fh::FHMutex> lock(mtx_);
    // Restart from 0 if not enough consecutive headers are available
    if(pkt_header_idx_ + num_pkts > kMaxPktsFlow)
        pkt_header_idx_ = 0;

    tmp = pkt_header_idx_;
    pkt_header_idx_ = (pkt_header_idx_ + num_pkts) % kMaxPktsFlow;

    return tmp;
}

uint8_t* Flow::get_packet_header_gpu()
{
    return pkt_hdr_gpu_;
}

uint8_t* Flow::get_next_header_gpu(uint32_t num_pkts)
{
    uint8_t * tmp = nullptr;

    if(pkt_hdr_gpu_ == nullptr)
        return nullptr;

    const std::lock_guard<aerial_fh::FHMutex> lock(mtx_);
    // Restart from 0 if not enough consecutive headers are available
    if(pkt_header_idx_ + num_pkts > kMaxPktsFlow)
        pkt_header_idx_ = 0;
    tmp = (uint8_t*) (pkt_hdr_gpu_ + (pkt_header_idx_ * packet_size_rnd));
    pkt_header_idx_ = (pkt_header_idx_ + num_pkts) % kMaxPktsFlow;

    return tmp;
}

uint32_t Flow::get_packet_header_gpu_lkey()
{
    return pkt_hdr_gpu_lkey_;
}

void Flow::receive(MsgReceiveInfo* info, size_t* num_msgs)
{
    if(unlikely(rxq_ == nullptr))
    {
        THROW_FH(EINVAL, StringBuilder() << "Calling receive_flow() but no RXQ assigned to Flow with eAxC=" << info_.eAxC);
    }

    size_t rx_bytes = rxq_->receive(info, num_msgs);
    peer_->update_rx_metrics(*num_msgs, rx_bytes);
}

void Flow::receive_until(MsgReceiveInfo* info, size_t* num_msgs, Ns timeout)
{
    if(unlikely(rxq_ == nullptr))
    {
        THROW_FH(EINVAL, StringBuilder() << "Calling receive_flow_until() but no RXQ assigned to Flow with eAxC=" << info_.eAxC);
    }

    size_t rx_bytes = rxq_->receive_until(info, num_msgs, timeout);
    peer_->update_rx_metrics(*num_msgs, rx_bytes);
}

void Flow::create_rx_rules()
{
    auto rx_mode = peer_->get_info().rx_mode;
    if(rx_mode == RxApiMode::FLOW)
    {
        if(peer_->get_nic()->pdump_enabled())
        {
            create_rx_rule_with_cpu_mirroring();
        }
        else
        {
            create_rx_rule();
        }
    }
    else if((rx_mode == RxApiMode::HYBRID) && (info_.type == FlowType::UPLANE && info_.flow_rx_mode == FlowRxApiMode::TXANDRX))
    {
        create_rx_rule_for_uplane();
    }
}

void Flow::create_rx_rule()
{
    auto  nic       = peer_->get_nic();
    auto  port_id   = nic->get_port_id();
    auto  name      = nic->get_name();
    auto& peer_info = peer_->get_info();

    if(unlikely(rxq_ == nullptr))
    {
        THROW_FH(EINVAL, StringBuilder() << "Calling create_rx_rule() but no RXQ assigned to Flow with eAxC=" << info_.eAxC);
    }

    rte_flow_error    err;
    rte_flow_item_eth eth_spec, eth_mask;

    memset(&eth_spec, 0, sizeof(eth_spec));
    memset(&eth_mask, 0, sizeof(eth_mask));
    memcpy(eth_spec.dst.addr_bytes, peer_info.src_mac_addr.bytes, RTE_ETHER_ADDR_LEN);
    memcpy(eth_spec.src.addr_bytes, peer_info.dst_mac_addr.bytes, RTE_ETHER_ADDR_LEN);
    memset(&eth_mask.dst.addr_bytes, 0xFF, sizeof(eth_mask.dst.addr_bytes));
    memset(&eth_mask.src.addr_bytes, 0xFF, sizeof(eth_mask.src.addr_bytes));
    eth_spec.type = rte_cpu_to_be_16(RTE_ETHER_TYPE_VLAN);
    eth_mask.type = 0xFFFF;
    eth_spec.has_vlan = 1;
    eth_mask.has_vlan = 1;

    rte_flow_item_vlan vlan_spec, vlan_mask;

    memset(&vlan_spec, 0, sizeof(vlan_spec));
    memset(&vlan_mask, 0, sizeof(vlan_mask));
    vlan_spec.tci        = rte_cpu_to_be_16(info_.vlan_tag.tci);
    vlan_mask.tci        = rte_cpu_to_be_16(0xFFFF);
    vlan_spec.inner_type = 0;
    vlan_mask.inner_type = 0;

    rte_flow_item_ecpri ecpri_spec, ecpri_mask;

    memset(&ecpri_spec, 0, sizeof(ecpri_spec));
    memset(&ecpri_mask, 0, sizeof(ecpri_mask));
    ecpri_spec.hdr.common.type = info_.type == FlowType::UPLANE ? RTE_ECPRI_MSG_TYPE_IQ_DATA : RTE_ECPRI_MSG_TYPE_RTC_CTRL;
    ecpri_mask.hdr.common.type = 0xFF;
    ecpri_spec.hdr.type0.pc_id = rte_cpu_to_be_16(info_.eAxC);
    ecpri_mask.hdr.type0.pc_id = 0xFFFF;

    NVLOGI_FMT(TAG, "Setting up flow rules for {} Flow {} of NIC {}: "\
                "[Destination MAC: {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}|{:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}] "\
                "[Source MAC: {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}|{:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}] "\
                "[Outer Ethertype: 0x{:04x}|0x{:04x}] [VLAN Tci: 0x{:04x}|0x{:04x}] [Inner Ethertype: 0x{:04x}|0x{:04x}] "\
                "[eCPRI Message Type: 0x{:02x}|0x{:02x}] [eCPRI eAxC ID: 0x{:04x}|0x{:04x}]\n",
                flow_type_.c_str(), info_.eAxC, name,
                eth_spec.dst.addr_bytes[0],eth_spec.dst.addr_bytes[1],eth_spec.dst.addr_bytes[2],
                eth_spec.dst.addr_bytes[3],eth_spec.dst.addr_bytes[4],eth_spec.dst.addr_bytes[5],
                eth_mask.dst.addr_bytes[0],eth_mask.dst.addr_bytes[1],eth_mask.dst.addr_bytes[2],
                eth_mask.dst.addr_bytes[3],eth_mask.dst.addr_bytes[4],eth_mask.dst.addr_bytes[5],
                eth_spec.src.addr_bytes[0],eth_spec.src.addr_bytes[1],eth_spec.src.addr_bytes[2],
                eth_spec.src.addr_bytes[3],eth_spec.src.addr_bytes[4],eth_spec.src.addr_bytes[5],
                eth_mask.src.addr_bytes[0],eth_mask.src.addr_bytes[1],eth_mask.src.addr_bytes[2],
                eth_mask.src.addr_bytes[3],eth_mask.src.addr_bytes[4],eth_mask.src.addr_bytes[5],
                rte_be_to_cpu_16(eth_spec.type), rte_be_to_cpu_16(eth_mask.type),
                rte_be_to_cpu_16(vlan_spec.tci), rte_be_to_cpu_16(vlan_mask.tci),
                rte_be_to_cpu_16(vlan_spec.inner_type), rte_be_to_cpu_16(vlan_mask.inner_type),
                static_cast<int>(ecpri_spec.hdr.common.type), static_cast<int>(ecpri_mask.hdr.common.type),
                rte_be_to_cpu_16(ecpri_spec.hdr.type0.pc_id), rte_be_to_cpu_16(ecpri_mask.hdr.type0.pc_id));

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
        THROW_FH(ret, StringBuilder() << "Failed to validate flow rule for " << flow_type_ << " Flow " << info_.eAxC << " on NIC " << name << ": " << err.message);
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
        THROW_FH(EINVAL, StringBuilder() << "Failed to create flow rule for " << flow_type_ << " Flow " << info_.eAxC << " on NIC " << name << ": " << err.message);
    }
}

void Flow::create_rx_rule_for_uplane()
{
    auto  nic       = peer_->get_nic();
    auto  port_id   = nic->get_port_id();
    auto  name      = nic->get_name();
    auto& peer_info = peer_->get_info();

    rte_flow_error    err;
    rte_flow_item_eth eth_spec, eth_mask;

    memset(&eth_spec, 0, sizeof(eth_spec));
    memset(&eth_mask, 0, sizeof(eth_mask));
    memcpy(eth_spec.dst.addr_bytes, peer_info.src_mac_addr.bytes, RTE_ETHER_ADDR_LEN);
    memcpy(eth_spec.src.addr_bytes, peer_info.dst_mac_addr.bytes, RTE_ETHER_ADDR_LEN);
    memset(&eth_mask.dst.addr_bytes, 0xFF, sizeof(eth_mask.dst.addr_bytes));
    memset(&eth_mask.src.addr_bytes, 0xFF, sizeof(eth_mask.src.addr_bytes));
    eth_spec.type = rte_cpu_to_be_16(RTE_ETHER_TYPE_VLAN);
    eth_mask.type = 0xFFFF;
    eth_spec.has_vlan = 1;
    eth_mask.has_vlan = 1;

    rte_flow_item_vlan vlan_spec, vlan_mask;

    memset(&vlan_spec, 0, sizeof(vlan_spec));
    memset(&vlan_mask, 0, sizeof(vlan_mask));
    vlan_spec.tci        = rte_cpu_to_be_16(info_.vlan_tag.tci);
    vlan_mask.tci        = rte_cpu_to_be_16(0xFFFF);
    vlan_spec.inner_type = 0;
    vlan_mask.inner_type = 0;

    rte_flow_item_ecpri ecpri_spec, ecpri_mask;

    memset(&ecpri_spec, 0, sizeof(ecpri_spec));
    memset(&ecpri_mask, 0, sizeof(ecpri_mask));
    ecpri_spec.hdr.common.type = RTE_ECPRI_MSG_TYPE_IQ_DATA;
    ecpri_mask.hdr.common.type = 0xFF;
    ecpri_spec.hdr.type0.pc_id = rte_cpu_to_be_16(info_.eAxC);
    ecpri_mask.hdr.type0.pc_id = 0xFFFF;

    NVLOGI_FMT(TAG, "Setting up U-plane RX flow rules for {} Flow {} of NIC {}: "\
                "[Destination MAC: {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}|{:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}] "\
                "[Source MAC:  {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}|{:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}] "\
                "[Outer Ethertype: 0x{:04x}|0x{:04x}] "\
                "[VLAN Tci: 0x{:04x}|0x{:04x}] "\
                "[Inner Ethertype: 0x{:04x}|0x{:04x}] "\
                "[eCPRI Message Type: 0x{:02x}|0x{:02x}]\n",
                flow_type_.c_str(), info_.eAxC, name,
                eth_spec.dst.addr_bytes[0],eth_spec.dst.addr_bytes[1],eth_spec.dst.addr_bytes[2],
                eth_spec.dst.addr_bytes[3],eth_spec.dst.addr_bytes[4],eth_spec.dst.addr_bytes[5],
                eth_mask.dst.addr_bytes[0],eth_mask.dst.addr_bytes[1],eth_mask.dst.addr_bytes[2],
                eth_mask.dst.addr_bytes[3],eth_mask.dst.addr_bytes[4],eth_mask.dst.addr_bytes[5],
                eth_spec.src.addr_bytes[0],eth_spec.src.addr_bytes[1],eth_spec.src.addr_bytes[2],
                eth_spec.src.addr_bytes[3],eth_spec.src.addr_bytes[4],eth_spec.src.addr_bytes[5],
                eth_mask.src.addr_bytes[0],eth_mask.src.addr_bytes[1],eth_mask.src.addr_bytes[2],
                eth_mask.src.addr_bytes[3],eth_mask.src.addr_bytes[4],eth_mask.src.addr_bytes[5],
                rte_be_to_cpu_16(eth_spec.type), rte_be_to_cpu_16(eth_mask.type),
                rte_be_to_cpu_16(vlan_spec.tci), rte_be_to_cpu_16(vlan_mask.tci),
                rte_be_to_cpu_16(vlan_spec.inner_type), rte_be_to_cpu_16(vlan_mask.inner_type),
                static_cast<int>(ecpri_spec.hdr.common.type), static_cast<int>(ecpri_mask.hdr.common.type));

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
        THROW_FH(ret, StringBuilder() << "Failed to validate U-plane flow rule for " << flow_type_ << " Flow " << info_.eAxC << " on NIC " << name << ": " << err.message);
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
        THROW_FH(EINVAL, StringBuilder() << "Failed to create U-plane flow rule for " << flow_type_ << " Flow " << info_.eAxC << " on NIC " << name << ": " << err.message);
    }

    rx_flow_rules_.emplace_back(new RxFlowRule(port_id, flow));
}

void Flow::create_rx_rule_with_cpu_mirroring()
{
    auto  nic       = peer_->get_nic();
    auto  rxq_pcap  = nic->get_pcap_rxq();
    auto  port_id   = nic->get_port_id();
    auto  name      = nic->get_name();
    auto& peer_info = peer_->get_info();

    rte_flow_error    err;
    rte_flow_item_eth eth_spec, eth_mask;

    memset(&eth_spec, 0, sizeof(eth_spec));
    memset(&eth_mask, 0, sizeof(eth_mask));
    memcpy(eth_spec.dst.addr_bytes, peer_info.src_mac_addr.bytes, RTE_ETHER_ADDR_LEN);
    memcpy(eth_spec.src.addr_bytes, peer_info.dst_mac_addr.bytes, RTE_ETHER_ADDR_LEN);
    memset(&eth_mask.dst.addr_bytes, 0xFF, sizeof(eth_mask.dst.addr_bytes));
    memset(&eth_mask.src.addr_bytes, 0xFF, sizeof(eth_mask.src.addr_bytes));
    eth_spec.type = rte_cpu_to_be_16(RTE_ETHER_TYPE_VLAN);
    eth_mask.type = 0xFFFF;
    eth_spec.has_vlan = 1;
    eth_mask.has_vlan = 1;

    rte_flow_item_vlan vlan_spec, vlan_mask;

    memset(&vlan_spec, 0, sizeof(vlan_spec));
    memset(&vlan_mask, 0, sizeof(vlan_mask));
    vlan_spec.tci        = rte_cpu_to_be_16(info_.vlan_tag.tci);
    vlan_mask.tci        = rte_cpu_to_be_16(0xFFFF);
    vlan_spec.inner_type = 0;
    vlan_mask.inner_type = 0;

    rte_flow_item_ecpri ecpri_spec, ecpri_mask;

    memset(&ecpri_spec, 0, sizeof(ecpri_spec));
    memset(&ecpri_mask, 0, sizeof(ecpri_mask));
    uint8_t ecpri_message_type = info_.type == FlowType::UPLANE ? RTE_ECPRI_MSG_TYPE_IQ_DATA : RTE_ECPRI_MSG_TYPE_RTC_CTRL;
    ecpri_spec.hdr.common.type = ecpri_message_type;
    ecpri_mask.hdr.common.type = 0xFF;
    ecpri_spec.hdr.type0.pc_id = rte_cpu_to_be_16(info_.eAxC);
    ecpri_mask.hdr.type0.pc_id = 0xFFFF;

    NVLOGD_FMT(TAG, "Setting up RX flow rules with pdump support for {} Flow {} of NIC {}: "\
            "[Destination MAC: {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}|{:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}] "\
            "[Source MAC: {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}|Source MAC: {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}] "\
            "[Outer Ethertype: 0x{:04x}|0x{:04x}] [VLAN Tci: 0x{:04x}|0x{:04x}] "\
            "[Inner Ethertype: 0x{:04x}|0x{:04x}] [eCPRI Message Type: 0x{:02x}|0xffff] [eCPRI eAxC ID: 0x{:04x}|0x{:04x}]\n",
                flow_type_.c_str(), info_.eAxC, name,
                eth_spec.dst.addr_bytes[0], eth_spec.dst.addr_bytes[1], eth_spec.dst.addr_bytes[2],
                eth_spec.dst.addr_bytes[3], eth_spec.dst.addr_bytes[4], eth_spec.dst.addr_bytes[5],
                eth_mask.dst.addr_bytes[0],eth_mask.dst.addr_bytes[1],eth_mask.dst.addr_bytes[2],
                eth_mask.dst.addr_bytes[3],eth_mask.dst.addr_bytes[4],eth_mask.dst.addr_bytes[5],
                eth_spec.src.addr_bytes[0],eth_spec.src.addr_bytes[1],eth_spec.src.addr_bytes[2],
                eth_spec.src.addr_bytes[3],eth_spec.src.addr_bytes[4],eth_spec.src.addr_bytes[5],
                eth_mask.src.addr_bytes[0],eth_mask.src.addr_bytes[1],eth_mask.src.addr_bytes[2],
                eth_mask.src.addr_bytes[3],eth_mask.src.addr_bytes[4],eth_mask.src.addr_bytes[5],
                rte_be_to_cpu_16(eth_spec.type), rte_be_to_cpu_16(eth_mask.type),
                rte_be_to_cpu_16(vlan_spec.tci), rte_be_to_cpu_16(vlan_mask.tci),
                rte_be_to_cpu_16(vlan_spec.inner_type),  rte_be_to_cpu_16(vlan_mask.inner_type),
                ecpri_message_type,
                rte_be_to_cpu_16(ecpri_spec.hdr.type0.pc_id), rte_be_to_cpu_16(ecpri_mask.hdr.type0.pc_id));

    ecpri_spec.hdr.common.u32 = rte_cpu_to_be_32(ecpri_spec.hdr.common.u32);
    ecpri_mask.hdr.common.u32 = rte_cpu_to_be_32(ecpri_mask.hdr.common.u32);

    rte_flow_item patterns[]{
        {.type = RTE_FLOW_ITEM_TYPE_ETH, .spec = &eth_spec, .last = nullptr, .mask = &eth_mask},
        {.type = RTE_FLOW_ITEM_TYPE_VLAN, .spec = &vlan_spec, .last = nullptr, .mask = &vlan_mask},
        {.type = RTE_FLOW_ITEM_TYPE_ECPRI, .spec = &ecpri_spec, .last = nullptr, .mask = &ecpri_mask},
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
            THROW_FH(ret, StringBuilder() << "Failed to validate group " << attr.group << " RX flow rule with mirroring for " << flow_type_ << " Flow " << info_.eAxC << " on NIC " << name << ": " << err.message);
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
            THROW_FH(EINVAL, StringBuilder() << "Failed to create group " << attr.group << " RX flow rule with mirroring for " << flow_type_ << " Flow " << info_.eAxC << " on NIC " << name << ": " << err.message);
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
            THROW_FH(ret, StringBuilder() << "Failed to validate group " << attr.group << " RX flow rule with mirroring for " << flow_type_ << " Flow " << info_.eAxC << " on NIC " << name << ": " << err.message);
        }

        auto flow = rte_flow_create(port_id, &attr, patterns, actions, &err);
        if(flow == nullptr)
        {
            THROW_FH(EINVAL, StringBuilder() << "Failed to create group " << attr.group << " RX flow rule with mirroring for " << flow_type_ << " Flow " << info_.eAxC << " on NIC " << name << ": " << err.message);
        }

        rx_flow_rules_.emplace_back(new RxFlowRule(port_id, flow));
    }
}

uint8_t SequenceIdGenerator::next()
{
    return ecpriSeqId_++;
}

uint8_t SequenceIdGenerator::next(uint8_t num)
{
    return ecpriSeqId_ += num;
}

} // namespace aerial_fh
