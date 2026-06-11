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

#define TAG (NVLOG_TAG_BASE_CUPHY_DRIVER + 5) // "DRV.FH"
#define TAG_STARTUP_TIMES (NVLOG_TAG_BASE_CUPHY_CONTROLLER + 5) // "CTL.STARTUP_TIMES"

#include "app_config.hpp"
#include "fh.hpp"
#include "exceptions.hpp"
#include "context.hpp"
#include <unistd.h>
#include "ti_generic.hpp"
#include <limits>

#define likely(x) __builtin_expect((x), 1)
#define unlikely(x) __builtin_expect((x), 0)

#define __fh_always_inline inline __attribute__((always_inline))

#define ENABLE_MOD_COMP 0

using namespace std;
using namespace aerial_fh;
static constexpr uint8_t MAX_FLOWS = 32;
static constexpr uint16_t MAX_PRB_INFO_PRB_SYMBOL = 273;
static constexpr uint16_t MAX_COMP_INFO_PER_PRB = 4;

static constexpr int NUM_BUFFER_PER_DIR = 2; 
    
// The buffers are per direction (to avoid ULC & DLC stomping) and max_cells_per_slot (to enable cell level parallelism)
static std::array<std::array<cplane_buffer_t,MAX_CELLS_PER_SLOT>,NUM_BUFFER_PER_DIR> cplane_prepare_buffer; 

FhProxy::FhProxy(
    phydriver_handle _pdh, const context_config& ctx_cfg) : pdh(_pdh)
{
    FronthaulInfo fh_info{
        .dpdk_thread = ctx_cfg.fh_cpu_core,
        .accu_tx_sched_res_ns = ctx_cfg.accu_tx_sched_res_ns,
        .pdump_client_thread = ctx_cfg.pdump_client_thread,
        .accu_tx_sched_disable = ctx_cfg.accu_tx_sched_disable,
        .dpdk_verbose_logs = ctx_cfg.dpdk_verbose_logs,
        .dpdk_file_prefix = ctx_cfg.dpdk_file_prefix,
        .cuda_device_ids = {ctx_cfg.gpu_id},
        .cuda_device_ids_for_compute = {ctx_cfg.gpu_id},
        .rivermax = false,
        .fh_stats_dump_cpu_core = ctx_cfg.fh_stats_dump_cpu_core,
        .cpu_rx_only = false,
        .enable_gpu_comm_via_cpu = ctx_cfg.enable_gpu_comm_via_cpu
    };

    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    if(pdctx->getUeMode())
    {
        fh_info.cpu_rx_only = true;
    }

    if(pdctx->cpuCommEnabled()) //Clear the cuda devices vector(non compute) if CPU Comms is enabled
    {
        fh_info.cuda_device_ids.clear();
    }
    
    if(aerial_fh::open(&fh_info, &fhi))
    {
        PHYDRIVER_THROW_EXCEPTIONS(-1, "aerial_fh::open returned error");
    }

    mf.init(_pdh, std::string("FhProxy"), sizeof(FhProxy));

    static_beam_id_start = ctx_cfg.static_beam_id_start;
    static_beam_id_end = ctx_cfg.static_beam_id_end;
    dynamic_beam_id_start = ctx_cfg.dynamic_beam_id_start;
    dynamic_beam_id_offset = ctx_cfg.dynamic_beam_id_start;
    dynamic_beam_id_end = ctx_cfg.dynamic_beam_id_end;

    dynamic_beam_ids_per_slot     = CUPHY_BFW_N_MAX_PRB_GRPS * CUPHY_BFW_COEF_COMP_N_MAX_LAYERS_PER_USER_GRP;
    dynamic_beam_id_covered_slots = (dynamic_beam_id_end - dynamic_beam_id_start + 1) / dynamic_beam_ids_per_slot;

    NVLOGC_FMT(TAG, "Dynamic beam id start: {}, end: {}, num of beam ids per slots: {}, number of consecutive slots with unique of beam ids: {} ", dynamic_beam_id_start, dynamic_beam_id_end, dynamic_beam_ids_per_slot, dynamic_beam_id_covered_slots);

    bfw_c_plane_chaining_mode = static_cast<aerial_fh::BfwCplaneChainingMode>(ctx_cfg.bfw_c_plane_chaining_mode);
    bfw_coeff_size = (4 * sizeof(uint8_t)) + (slot_command_api::MAX_BFW_COFF_STORE_INDEX * slot_command_api::MAX_DL_UL_BF_UE_GROUPS *
        slot_command_api::MAX_MU_MIMO_LAYERS * slot_command_api::MAX_NUM_PRGS_DBF *
        slot_command_api::NUM_GNB_TX_RX_ANT_PORTS * slot_command_api::IQ_REPR_FP32_COMPLEX * sizeof(uint32_t));

    dlc_bfw_enable_divide_per_cell = !(!ctx_cfg.dlc_bfw_enable_divide_per_cell);
    ulc_bfw_enable_divide_per_cell = !(!ctx_cfg.ulc_bfw_enable_divide_per_cell);
    
    dlc_alloc_cplane_bfw_txq = (ctx_cfg.dlc_alloc_cplane_bfw_txq == 1) ? true : false;
    ulc_alloc_cplane_bfw_txq = (ctx_cfg.ulc_alloc_cplane_bfw_txq == 1) ? true : false;

}

FhProxy::~FhProxy()
{
    //FIXME: cleanup peers
    // delete sync_ready_list_gdr; //FIXME: causes cudaFree error
    // sync_buffer.reset();
    // last_ordered_h.reset();

    for (auto nic : nic_map)
    {
        aerial_fh::print_stats(nic.second, true);
        aerial_fh::remove_nic(nic.second);
    }


    if (aerial_fh::close(fhi))
    {
        NVLOGE_FMT(TAG, AERIAL_ORAN_FH_EVENT, "Failed to close Fronthaul");
    }
}

phydriver_handle FhProxy::getPhyDriverHandler(void) const
{
    return pdh;
}

FronthaulHandle FhProxy::getFhInstance(void) const
{
    return fhi;
}

struct fh_peer_t * FhProxy::getPeerFromId(peer_id_t peer_id) {
    auto it = peer_map.find(peer_id);
    if(it == peer_map.end())
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Peer {} not registered\n", peer_id);
        return nullptr;
    }

    return it->second.get();
}

struct fh_peer_t * FhProxy::getPeerFromAbsoluteId(int index) {
    if(index >= peer_id_vector.size())
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Peer index {} not registered\n", index);
        return nullptr;
    }
    peer_id_t peer_id = peer_id_vector[index];
    auto it = peer_map.find(peer_id);
    if(it == peer_map.end())
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Peer {} not registered\n", peer_id);
        return nullptr;
    }
    return it->second.get();
}

int FhProxy::flushMemory(struct fh_peer_t * peer_ptr) {
    uint32_t* tmp = (uint32_t*)peer_ptr->rx_order_items.flush_gmem->addrh();
    return (int)tmp[0]; //this should always be 0
}

bool FhProxy::checkIfNicExists(std::string nic_name)
{
    return nic_map.find(nic_name) != nic_map.end();
}

std::vector<std::string> FhProxy::getNicList()
{
    std::vector<std::string> res;
    for(auto [nic, _] : nic_map) res.push_back(nic);
    return res;
}

int FhProxy::registerNic(struct nic_cfg cfg, int gpu_id)
{
    uint16_t txq_gpu = 0;
    uint16_t txq_cpu = 0;
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();

    if(pdctx->gpuCommDlEnabled()) {
        txq_gpu = cfg.txq_count_uplane;
        txq_cpu = cfg.txq_count_cplane;
    }
    if(pdctx->cpuCommEnabled()){
        txq_gpu = 0;
        txq_cpu = cfg.txq_count_uplane + cfg.txq_count_cplane;
    }

    aerial_fh::NicInfo ninfo{
        cfg.nic_bus_addr,
        cfg.nic_mtu,
        false,
        cfg.cpu_mbuf_num,
        0,
        0,
        0,
        cfg.tx_req_num,
        txq_cpu,
        txq_gpu,
        cfg.rxq_count,
        cfg.txq_size, cfg.rxq_size, gpu_id, false};


    aerial_fh::NicHandle nic;

    if(aerial_fh::add_nic(fhi, &ninfo, &nic))
    {
        NVLOGE_FMT(TAG, AERIAL_ORAN_FH_EVENT, "Failed to add NIC {}", cfg.nic_bus_addr.c_str());
        return -1;
    }

    nic_map[cfg.nic_bus_addr] = nic;

    return 0;
}

void FhProxy::updatePeerMap(peer_id_t peer_id, std::unique_ptr<struct fh_peer_t> p_fh_peer) {
    peer_map.insert({peer_id, std::move(p_fh_peer)});
}

int FhProxy::registerPeer(
    uint16_t                       cell_id,
    peer_id_t&                     peer_id,
    std::array<uint8_t, 6>         src_eth_addr,
    std::array<uint8_t, 6>         dst_eth_addr,
    uint16_t                       vlan_tci,
    uint8_t                        txq_count_uplane,
    enum UserDataCompressionMethod dl_comp_meth,
    uint8_t                        dl_bit_width,
    int                            gpu_id,
    std::string                    nic_name,
    struct doca_rx_items*          doca_rxq_info,
    struct doca_rx_items*          doca_rxq_info_srs,
    std::vector<uint16_t>&         eAxC_list_ul,
    std::vector<uint16_t>&         eAxC_list_srs,
    std::vector<uint16_t>&         eAxC_list_dl,
    uint16_t                       max_num_prbs_per_symbol)
{
    PeerHandle peer;
    PeerInfo peer_info;
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    struct fh_peer_t * peer_ptr;
    struct doca_rx_items d_rxq_info;

    peer_info.id                          = cell_id;
    peer_info.src_mac_addr.bytes[0]       = src_eth_addr[0];
    peer_info.src_mac_addr.bytes[1]       = src_eth_addr[1];
    peer_info.src_mac_addr.bytes[2]       = src_eth_addr[2];
    peer_info.src_mac_addr.bytes[3]       = src_eth_addr[3];
    peer_info.src_mac_addr.bytes[4]       = src_eth_addr[4];
    peer_info.src_mac_addr.bytes[5]       = src_eth_addr[5];
    peer_info.dst_mac_addr.bytes[0]       = dst_eth_addr[0];
    peer_info.dst_mac_addr.bytes[1]       = dst_eth_addr[1];
    peer_info.dst_mac_addr.bytes[2]       = dst_eth_addr[2];
    peer_info.dst_mac_addr.bytes[3]       = dst_eth_addr[3];
    peer_info.dst_mac_addr.bytes[4]       = dst_eth_addr[4];
    peer_info.dst_mac_addr.bytes[5]       = dst_eth_addr[5];
    peer_info.vlan.tci                    = vlan_tci;
    peer_info.ud_comp_info.iq_sample_size = dl_bit_width; //USER_DATA_IQ_BIT_WIDTH;     // Ignored if FLOW_TYPE_CPLANE
    peer_info.ud_comp_info.method = dl_comp_meth;

    //FIXME?
    if(dl_comp_meth == UserDataCompressionMethod::BLOCK_FLOATING_POINT && dl_bit_width == BFP_NO_COMPRESSION)
    {
        peer_info.ud_comp_info.method = UserDataCompressionMethod::NO_COMPRESSION;
    }

    peer_info.max_num_prbs_per_symbol = max_num_prbs_per_symbol;

    if(pdctx->gpuCommDlEnabled() && !pdctx->cpuCommEnabled()) {
        peer_info.txq_count_uplane = 0;
        peer_info.txq_count_uplane_gpu = txq_count_uplane;
    } else {
        peer_info.txq_count_uplane = txq_count_uplane;
        peer_info.txq_count_uplane_gpu = 0;
    }

    GpuDevice * gDev = pdctx->getGpuById(gpu_id);
    if(gDev == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "GPU {} not in the context", gpu_id);
        return EINVAL;
    }

    peer_info.rx_mode          = RxApiMode::PEER;
    if(pdctx->getUeMode())
    {
        peer_info.rx_mode          = RxApiMode::UEMODE;
    }
    peer_info.txq_cplane       = true;
    peer_info.mMIMO_enable     = pdctx->getmMIMO_enable();
    if(pdctx->getmMIMO_enable())
    {
        peer_info.txq_bfw_cplane = true;
        peer_info.bfw_cplane_info.bfw_chain_mode = bfw_c_plane_chaining_mode;
        peer_info.bfw_cplane_info.bfw_cplane_buffer_size = bfw_coeff_size;
        peer_info.bfw_cplane_info.dlc_bfw_enable_divide_per_cell = dlc_bfw_enable_divide_per_cell;
        peer_info.bfw_cplane_info.ulc_bfw_enable_divide_per_cell = ulc_bfw_enable_divide_per_cell;
        peer_info.bfw_cplane_info.dlc_alloc_cplane_bfw_txq = dlc_alloc_cplane_bfw_txq;
        peer_info.bfw_cplane_info.ulc_alloc_cplane_bfw_txq = ulc_alloc_cplane_bfw_txq;
    }
    else
    {
        peer_info.txq_bfw_cplane = false;
    }
    peer_info.enable_srs       = pdctx->get_enable_srs();

    if(aerial_fh::add_peer(nic_map[nic_name], &peer_info, &peer,eAxC_list_ul,eAxC_list_srs, eAxC_list_dl))
    {
        NVLOGE_FMT(TAG, AERIAL_ORAN_FH_EVENT, "aerial_fh::add_peer error");
        return -1;
    }
    // Update allocations related to Peer creation
    // FH Peer's GPU memory is tracked in aerial-fh-driver, do not duplicate tracking here
    // mf.addGpuRegularSize(aerial_fh::get_gpu_regular_size(peer));
    // mf.addGpuPinnedSize(aerial_fh::get_gpu_pinned_size(peer));
    mf.addCpuRegularSize(aerial_fh::get_cpu_regular_size(peer));
    mf.addCpuPinnedSize(aerial_fh::get_cpu_pinned_size(peer));

    aerial_fh::get_doca_rxq_items(peer,&d_rxq_info);
    doca_rxq_info->gpu_dev=d_rxq_info.gpu_dev;
    doca_rxq_info->ddev=d_rxq_info.ddev;
    doca_rxq_info->eth_rxq_ctx=d_rxq_info.eth_rxq_ctx;
    doca_rxq_info->eth_rxq_cpu=d_rxq_info.eth_rxq_cpu;
    doca_rxq_info->eth_rxq_gpu=d_rxq_info.eth_rxq_gpu;
    doca_rxq_info->pkt_buff_mmap=d_rxq_info.pkt_buff_mmap;
    doca_rxq_info->gpu_pkt_addr=d_rxq_info.gpu_pkt_addr;
    doca_rxq_info->dpdk_queue_idx=d_rxq_info.dpdk_queue_idx;
    doca_rxq_info->sem_cpu=d_rxq_info.sem_cpu;
    doca_rxq_info->sem_gpu=d_rxq_info.sem_gpu;
    doca_rxq_info->nitems=d_rxq_info.nitems;
    doca_rxq_info->sem_gpu_aerial_fh=d_rxq_info.sem_gpu_aerial_fh;

    if(pdctx->get_enable_srs())
    {
        aerial_fh::get_doca_rxq_items_srs(peer,&d_rxq_info);
        doca_rxq_info_srs->gpu_dev=d_rxq_info.gpu_dev;
        doca_rxq_info_srs->ddev=d_rxq_info.ddev;
        doca_rxq_info_srs->eth_rxq_ctx=d_rxq_info.eth_rxq_ctx;
        doca_rxq_info_srs->eth_rxq_cpu=d_rxq_info.eth_rxq_cpu;
        doca_rxq_info_srs->eth_rxq_gpu=d_rxq_info.eth_rxq_gpu;
        doca_rxq_info_srs->pkt_buff_mmap=d_rxq_info.pkt_buff_mmap;
        doca_rxq_info_srs->gpu_pkt_addr=d_rxq_info.gpu_pkt_addr;
        doca_rxq_info_srs->dpdk_queue_idx=d_rxq_info.dpdk_queue_idx;
        doca_rxq_info_srs->sem_cpu=d_rxq_info.sem_cpu;
        doca_rxq_info_srs->sem_gpu=d_rxq_info.sem_gpu;
        doca_rxq_info_srs->nitems=d_rxq_info.nitems;
        doca_rxq_info_srs->sem_gpu_aerial_fh=d_rxq_info.sem_gpu_aerial_fh;
    }

    peer_id = Time::nowNs().count();
    peer_id_vector.push_back(peer_id);
    if(peer_map.insert(std::pair<peer_id_t, std::unique_ptr<fh_peer_t>>(
                                                                    peer_id,
                                                                    std::move(std::unique_ptr<fh_peer_t>(new fh_peer_t(peer_id, peer, peer_info, dl_comp_meth, dl_bit_width)))
                                                                    )
                    ).second == false)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "FH peer %" PRIu64" insert error\n", peer_id );
        return -1;
    }

    peer_ptr = getPeerFromId(peer_id);
    if(!peer_ptr)
        return -1;

    mf.addCpuRegularSize(sizeof(fh_peer_t));

    ////////////////////////////////////////////////////////////////////////////////////////
    /// U-plane packets to Order Kernel
    ////////////////////////////////////////////////////////////////////////////////////////
    peer_ptr->rx_order_items.sync_buffer.reset(std::move(new host_buf(RX_QUEUE_SYNC_LIST_ITEMS * sizeof(struct rx_queue_sync), nullptr)));
    peer_ptr->rx_order_items.sync_buffer->clear();
    mf.addCpuPinnedSize(RX_QUEUE_SYNC_LIST_ITEMS * sizeof(struct rx_queue_sync));

    peer_ptr->rx_order_items.sync_list             = (struct rx_queue_sync*)peer_ptr->rx_order_items.sync_buffer->addr();

    peer_ptr->rx_order_items.sync_ready_list_gdr   = gDev->newGDRbuf(RX_QUEUE_SYNC_LIST_ITEMS * sizeof(uint32_t));
    peer_ptr->rx_order_items.sync_item             = 0;
    mf.addGpuPinnedSize(peer_ptr->rx_order_items.sync_ready_list_gdr->size_alloc);

    peer_ptr->rx_order_items.last_ordered_h.reset(std::move(new host_buf(1 * sizeof(int), nullptr)));
    peer_ptr->rx_order_items.last_ordered_h->clear();
    peer_ptr->rx_order_items.last_ufree                     = 0;
    mf.addCpuPinnedSize(sizeof(int));

    for(int x = 0; x < RX_QUEUE_SYNC_LIST_ITEMS; x++)
    {
        peer_ptr->rx_order_items.umsg_rx_list[x].umsg_info = (fhproxy_umsg_rx*)calloc(CK_ORDER_PKTS_BUFFERING, sizeof(fhproxy_umsg_rx));
        if(peer_ptr->rx_order_items.umsg_rx_list[x].umsg_info == NULL)
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Couldn't allocate {} fhproxy_umsg_rx", RX_QUEUE_SYNC_LIST_ITEMS);
        }

        mf.addCpuRegularSize(CK_ORDER_PKTS_BUFFERING * sizeof(fhproxy_umsg_rx));
        peer_ptr->rx_order_items.umsg_rx_list[x].num = 0;
    }

    peer_ptr->rx_order_items.umsg_rx_index = 0;

    peer_ptr->rx_order_items.flush_gmem.reset(gDev->newGDRbuf(sizeof(uint32_t)));
    ((uint32_t*)peer_ptr->rx_order_items.flush_gmem->addrh())[0] = 0;
    mf.addGpuPinnedSize(peer_ptr->rx_order_items.flush_gmem->size_alloc);

    return 0;
}

int FhProxy::updatePeer(
    peer_id_t               peer_id,
    enum UserDataCompressionMethod   dl_comp_meth,
    uint8_t dl_bit_width
)
{
    auto peer_ptr = getPeerFromId(peer_id);
    if(!peer_ptr)
        return -1;

    peer_ptr->dl_comp_meth = dl_comp_meth;
    peer_ptr->dl_bit_width = dl_bit_width;
    if(aerial_fh::update_peer(peer_ptr->peer, dl_comp_meth, dl_bit_width))
    {
        NVLOGE_FMT(TAG, AERIAL_ORAN_FH_EVENT, "Failed to update peer info");
        return -1;
    }

    return 0;
}

int FhProxy::removePeer(peer_id_t peer_id)
{
    auto peer_ptr = getPeerFromId(peer_id);
    if(!peer_ptr)
        return -1;

    for (auto &flow : peer_ptr->eAxC_ids_unique_cplane)
    {
        if(aerial_fh::remove_flow(flow.second))
        {
            NVLOGE_FMT(TAG, AERIAL_ORAN_FH_EVENT, "Failed to remove C-plane flow: {}", flow.first);
            return -1;
        }
    }

    for (auto &flow : peer_ptr->eAxC_ids_unique_uplane)
    {
        if(aerial_fh::remove_flow(flow.second))
        {
            NVLOGE_FMT(TAG, AERIAL_ORAN_FH_EVENT, "Failed to remove U-plane flow: {}", flow.first);
            return -1;
        }
    }

    if(aerial_fh::remove_peer(peer_ptr->peer))
    {
        NVLOGE_FMT(TAG, AERIAL_ORAN_FH_EVENT, "Failed to remove peer");
        return -1;
    }
    return 0;
}

int FhProxy::update_peer_rx_metrics(peer_id_t peer_id, size_t rx_packets, size_t rx_bytes)
{
    auto peer_ptr = getPeerFromId(peer_id);
    if(!peer_ptr)
        return -1;
    if(aerial_fh::update_rx_metrics(peer_ptr->peer, rx_packets, rx_bytes))
    {
        NVLOGE_FMT(TAG, AERIAL_ORAN_FH_EVENT, "Failed to update peer rx metrics");
        return -1;
    }
    return 0;
}

int FhProxy::update_peer_tx_metrics(peer_id_t peer_id, size_t tx_packets, size_t tx_bytes)
{
    auto peer_ptr = getPeerFromId(peer_id);
    if(!peer_ptr)
        return -1;
    if(aerial_fh::update_tx_metrics(peer_ptr->peer, tx_packets, tx_bytes))
    {
        NVLOGE_FMT(TAG, AERIAL_ORAN_FH_EVENT, "Failed to update peer tx metrics");
        return -1;
    }
    return 0;
}

int FhProxy::updatePeer(
    peer_id_t               peer_id,
    std::array<uint8_t, 6>  dst_eth_addr,
    uint16_t                vlan_tci,
    std::vector<uint16_t>& eAxC_list_ul,std::vector<uint16_t>& eAxC_list_srs
)
{
    auto peer_ptr = getPeerFromId(peer_id);
    if(!peer_ptr)
        return -1;

    aerial_fh::MacAddr dst_mac_addr;
    dst_mac_addr.bytes[0] = dst_eth_addr[0];
    dst_mac_addr.bytes[1] = dst_eth_addr[1];
    dst_mac_addr.bytes[2] = dst_eth_addr[2];
    dst_mac_addr.bytes[3] = dst_eth_addr[3];
    dst_mac_addr.bytes[4] = dst_eth_addr[4];
    dst_mac_addr.bytes[5] = dst_eth_addr[5];

    if(aerial_fh::update_peer(peer_ptr->peer, dst_mac_addr, vlan_tci,eAxC_list_ul,eAxC_list_srs))
    {
        NVLOGE_FMT(TAG, AERIAL_ORAN_FH_EVENT, "Failed to update peer info");
        return -1;
    }

    for (auto &flow : peer_ptr->eAxC_ids_unique_cplane)
    {
        aerial_fh::FlowInfo info{flow.first, aerial_fh::FlowType::CPLANE, vlan_tci};

        if(aerial_fh::update_flow(flow.second, &info))
        {
            NVLOGE_FMT(TAG, AERIAL_ORAN_FH_EVENT, "Failed to update C-plane flow info");
            return -1;
        }
    }

    for (auto &flow : peer_ptr->eAxC_ids_unique_uplane)
    {
        aerial_fh::FlowInfo info{flow.first, aerial_fh::FlowType::UPLANE, vlan_tci};

        if(aerial_fh::update_flow(flow.second, &info))
        {
            NVLOGE_FMT(TAG, AERIAL_ORAN_FH_EVENT, "Failed to update U-plane flow info");
            return -1;
        }
    }

    return 0;
}

int FhProxy::update_peer_max_num_prbs_per_symbol(
    peer_id_t               peer_id,
    uint16_t max_num_prbs_per_symbol
)
{
    auto peer_ptr = getPeerFromId(peer_id);
    if(!peer_ptr)
        return -1;

    if(aerial_fh::update_peer_max_num_prbs_per_symbol(peer_ptr->peer,max_num_prbs_per_symbol))
    {
        NVLOGE_FMT(TAG, AERIAL_ORAN_FH_EVENT, "Failed to update max_num_prbs_per_symbol peer info");
        return -1;
    }

    return 0;
}

struct rx_order_t * FhProxy::getRxOrderItemsPeer(peer_id_t peer_id) {
    struct fh_peer_t * peer_ptr = getPeerFromId(peer_id);
    if(!peer_ptr)
        return nullptr;

    return &(peer_ptr->rx_order_items);
}

int FhProxy::registerFlow(peer_id_t peer_id, uint16_t eAxC_id, uint16_t vlan_tci, slot_command_api::channel_type channel, const std::array<uint8_t, 6>* dst_mac_override)
{
    TI_GENERIC_INIT("FhProxy::registerFlow",15);
    TI_GENERIC_ADD("Start Task");

    TI_GENERIC_ADD("getPeerFromId");
    struct fh_peer_t * peer_ptr = getPeerFromId(peer_id);
    if(!peer_ptr)
        return -1;

    TI_GENERIC_ADD("Init And Error Checking");
    auto &cplane_flows = peer_ptr->cplane_flows[channel];
    auto &uplane_flows = peer_ptr->uplane_flows[channel];
    auto &eAxC_ids_unique_cplane =  peer_ptr->eAxC_ids_unique_cplane;
    auto &eAxC_ids_unique_uplane =  peer_ptr->eAxC_ids_unique_uplane;

    if(channel == slot_command_api::channel_type::SRS)
    {
        if(cplane_flows.size() > MAX_AP_PER_SLOT_SRS)
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Can't register more than {} SRS flows x peer (current {}) channel {}", MAX_AP_PER_SLOT_SRS , cplane_flows.size(), +channel) ;
            return -1;
        }
    }
    else if(channel == slot_command_api::channel_type::CSI_RS)
    {
        if(cplane_flows.size() > MAX_AP_PER_SLOT_CSI_RS)
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Can't register more than {} CSI-RS flows x peer (current {}) channel {}", MAX_AP_PER_SLOT_CSI_RS , cplane_flows.size(), +channel) ;
            return -1;
        }
    }
    else if(cplane_flows.size() > MAX_AP_PER_SLOT)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Can't register more than {} flows x peer (current {}) channel {}", MAX_AP_PER_SLOT , cplane_flows.size(), +channel) ;
        return -1;
    }
    //NVLOGD_FMT(TAG, "Register flow for peer {}, eaxc_id {}, vlan_tci {}, channel_type {}" ,  (int) peer_id , (int)eAxC_id , vlan_tci, channel);

    // C-plane flow
    if (eAxC_ids_unique_cplane.find(eAxC_id) == eAxC_ids_unique_cplane.end())
    {
        TI_GENERIC_ADD("Create Flow Objects");
        FlowInfo flow_info{eAxC_id, FlowType::CPLANE, vlan_tci};
        // dMIMO: per-flow dst MAC override (applies to both the C-plane and U-plane flow below)
        if(dst_mac_override)
        {
            memcpy(flow_info.dst_mac_override, dst_mac_override->data(), 6);
            flow_info.has_dst_mac_override = true;
        }
        FlowHandle flow_handle;
        if(channel >= slot_command_api::channel_type::PDSCH_CSIRS && channel <= slot_command_api::channel_type::PDCCH_DMRS)
        {
            flow_info.direction = FlowDir::DL;
        }
        else
        {
            flow_info.direction = FlowDir::UL;
        }


        TI_GENERIC_ADD("add_flow cplane");
        if(aerial_fh::add_flow(peer_ptr->peer, &flow_info, &flow_handle))
        {
            NVLOGE_FMT(TAG, AERIAL_ORAN_FH_EVENT, "aerial_fh::add_flow error");
            return -1;
        }

        TI_GENERIC_ADD("flow push_back 1");
        cplane_flows.push_back({flow_handle, eAxC_id});
        TI_GENERIC_ADD("mf.addCpuRegularSize");
        mf.addCpuRegularSize(sizeof(flow_info));
        eAxC_ids_unique_cplane[eAxC_id] = flow_handle;

        // related U-plane flow
        TI_GENERIC_ADD("add_flow uplane");
        flow_info.type = FlowType::UPLANE;
        if(aerial_fh::add_flow(peer_ptr->peer, &flow_info, &flow_handle))
        {
            NVLOGE_FMT(TAG, AERIAL_ORAN_FH_EVENT, "aerial_fh::add_flow error");
            return -1;
        }

        TI_GENERIC_ADD("flow push_back 2");
        uplane_flows.push_back({flow_handle, eAxC_id});
        TI_GENERIC_ADD("add_");
        mf.addCpuRegularSize(sizeof(flow_info));

        eAxC_ids_unique_uplane[eAxC_id] = flow_handle;
    }
    else
    {
        TI_GENERIC_ADD("Shortcut push_back");
        cplane_flows.push_back({eAxC_ids_unique_cplane[eAxC_id], eAxC_id});
        uplane_flows.push_back({eAxC_ids_unique_uplane[eAxC_id], eAxC_id});
    }

    TI_GENERIC_ADD("End Task");
    TI_GENERIC_ALL_NVLOGI(TAG_STARTUP_TIMES);

    return 0;
}
inline int16_t convert_static_beam_weight_endian(int16_t value) {
    #if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
        return static_cast<int16_t>(__builtin_bswap16(static_cast<uint16_t>(value))); // Swap on little endian
    #else
        return value;  // No swap needed on big endian
    #endif
}

[[nodiscard]] int FhProxy::storeDBTPdu(uint16_t cell_id, void* data_buf)
{
    int retVal = 0;
    scf_fapi_dbt_pdu_t& dbt_pdu = *reinterpret_cast<scf_fapi_dbt_pdu_t*>(data_buf);

    auto numDigBeams = dbt_pdu.numDigBeams;
    auto numBBPorts = dbt_pdu.numTXRUs;
    NVLOGD_FMT(TAG,"cell_id = {} numDigBeams={}, numBBPorts={}", cell_id, numDigBeams, numBBPorts);
    scf_fapi_digBeam_t *digBeamStart = dbt_pdu.digBeam;
    for ( uint16_t digBeamIdx = 0 ; digBeamIdx < numDigBeams ; digBeamIdx++)
    {
        //const auto& digBeam = dbt_pdu.digBeam[digBeamIdx];
        auto beamId         = digBeamStart->beamIdx;
        auto digBeamIQStart = digBeamStart->digBeamWeightPerTxRU;
        if((beamId < static_beam_id_start) || (beamId > static_beam_id_end))
        {
            retVal = -1;
            NVLOGE_FMT(TAG, AERIAL_ORAN_FH_EVENT, "cell_id {} beamId={} is out of range", cell_id, beamId);
            break;
        }

        //Flatten IQ weighets into BeamInfoArray
        int flattenedBuffOffset = 0;
        // Disabling dynamic memory allocation check as this is happening duing Cell Bring-up.
        MemtraceDisableScope md;
        auto beamWeights = std::make_unique<uint8_t[]>(numBBPorts * 2 * sizeof(int16_t));
        NVLOGD_FMT(TAG,"beamId={} beamWeight={}", beamId, static_cast<void *>(beamWeights.get()));
        for ( uint16_t bbPortIdx = 0 ; bbPortIdx < numBBPorts ; bbPortIdx++)
        {
            // Original value in Q15 format
            // Note: The actual numerical value changes after the byte swap, but that's expected as we're just reordering the bytes for transmission or protocol requirements.
            // The byte swap is done to ensure the correct byte order for the transmission or protocol requirements.
            int16_t real      = convert_static_beam_weight_endian(digBeamIQStart->digBeamWeightRe);
            int16_t imaginary = convert_static_beam_weight_endian(digBeamIQStart->digBeamWeightIm);

            NVLOGD_FMT(TAG,"real={} imaginary={} digBeamWeightRe={} digBeamWeightIm={}", real, imaginary,static_cast<int16_t>(digBeamIQStart->digBeamWeightRe), static_cast<int16_t>(digBeamIQStart->digBeamWeightIm));
            std::memcpy(&beamWeights[flattenedBuffOffset],
                &real,
                sizeof(int16_t));
            flattenedBuffOffset += sizeof(int16_t);
            std::memcpy(&beamWeights[flattenedBuffOffset],
                &imaginary,
                sizeof(int16_t));
            flattenedBuffOffset += sizeof(int16_t);
            digBeamIQStart++;
        }
        // Insert data
        fhStaticBfwStorage[cell_id][beamId] = std::make_tuple(false, std::move(beamWeights));
        digBeamStart = reinterpret_cast<scf_fapi_digBeam_t*>(digBeamIQStart);
    }
    return retVal;
}

int FhProxy::resetDBTStorage(uint16_t cell_id)
{
    // Check if the cell_id exists in the storage
    auto cellIt = fhStaticBfwStorage.find(cell_id);
    if (cellIt == fhStaticBfwStorage.end())
    {
        NVLOGE_FMT(TAG, AERIAL_ORAN_FH_EVENT, "{} cell_id={} not found in fhStaticBfwStorage", __func__, cell_id);
        return -1; // Indicate failure
    }

    // Access the StaticBfwMap for the given cell_id
    auto& staticBfwMap = cellIt->second;

    // Iterate through all beamId entries and reset the bool to false
    for (auto& [beamId, beamInfo] : staticBfwMap)
    {
        std::get<0>(beamInfo) = false; // Set the bool parameter to false
        NVLOGD_FMT(TAG, "Reset beamId={} for cell_id={}", beamId, cell_id);
    }

    return 0; // Indicate success
}

uint8_t* FhProxy::getStaticBFWWeights(uint16_t cell_id, uint16_t beamIdx)
{
    // Check if the cell_id exists in the storage
    auto cellIt = fhStaticBfwStorage.find(cell_id);
    if (cellIt == fhStaticBfwStorage.end())
    {
        NVLOGE_FMT(TAG, AERIAL_ORAN_FH_EVENT, "{} cell_id={} not found in fhStaticBfwStorage", __func__, cell_id);
        return nullptr; // Indicate failure
    }

    // Access the StaticBfwMap for the given cell_id
    auto& staticBfwMap = cellIt->second;

    // Check if the beamIdx exists in the map
    auto beamIt = staticBfwMap.find(beamIdx);
    if (beamIt == staticBfwMap.end())
    {
        NVLOGI_FMT(TAG, "{} beamIdx={} not found for cell_id={}", __func__, beamIdx, cell_id);
        return nullptr; // indicate that the given beamIdx is predefined beam
    }

    auto& [beamWeightSent, beamWeights] = beamIt->second;

    NVLOGD_FMT(TAG, "getStaticBFWWeights::Returning BeamInfoArray for cell_id={}, beamIdx={}, beamWeight={}", cell_id, beamIdx, static_cast<void *>(beamWeights.get()));
    return beamWeights.get();
}

// Return val -1: Error, 0: false, 1: true
int FhProxy::getBeamWeightsSentFlag(uint16_t cell_id, uint16_t beamIdx)
{
    // Check if the cell_id exists in the storage
    auto cellIt = fhStaticBfwStorage.find(cell_id);
    if (cellIt == fhStaticBfwStorage.end())
    {
        NVLOGE_FMT(TAG, AERIAL_ORAN_FH_EVENT, "{} cell_id={} not found in fhStaticBfwStorage", __func__, cell_id);
        return -1; // Indicate failure
    }

    // Access the StaticBfwMap for the given cell_id
    auto& staticBfwMap = cellIt->second;

    // Check if the beamIdx exists in the map
    auto beamIt = staticBfwMap.find(beamIdx);
    if (beamIt == staticBfwMap.end())
    {
        NVLOGI_FMT(TAG, "{} beamIdx={} not found for cell_id={}", __func__, beamIdx, cell_id);
        return -1; // Indicate failure
    }

    auto& [beamWeightSent, beamWeights] = beamIt->second;

    NVLOGD_FMT(TAG, "getBeamWeightsSentFlag::Returning for cell_id={}, beamIdx={}, beamWeightSent={}", cell_id, beamIdx, beamWeightSent);
    return beamWeightSent;
}

// retur '0' => success
int FhProxy::setBeamWeightsSentFlag(uint16_t cell_id, uint16_t beamIdx)
{
    // Check if the cell_id exists in the storage
    auto cellIt = fhStaticBfwStorage.find(cell_id);
    if (cellIt == fhStaticBfwStorage.end())
    {
        NVLOGE_FMT(TAG, AERIAL_ORAN_FH_EVENT, "{} cell_id={} not found in fhStaticBfwStorage", __func__, cell_id);
        return -1; // Indicate failure
    }

    // Access the StaticBfwMap for the given cell_id
    auto& staticBfwMap = cellIt->second;

    // Check if the beamIdx exists in the map
    auto beamIt = staticBfwMap.find(beamIdx);
    if (beamIt == staticBfwMap.end())
    {
        NVLOGE_FMT(TAG, AERIAL_ORAN_FH_EVENT, "{} beamIdx={} not found for cell_id={}", __func__, beamIdx, cell_id);
        return -1; // Indicate failure
    }

    auto& [beamWeightSent, beamWeights] = beamIt->second;

    beamWeightSent = true;

    NVLOGD_FMT(TAG, "setBeamWeightsSentFlag::setting beamWeightSent true for cell_id={}, beamIdx={}", cell_id, beamIdx);

    return 0; // Success
}

//return: 0: static BFW not configured, 1: static BFW configured
int FhProxy::staticBFWConfigured(uint16_t cell_id)
{
    // Check if the cell_id exists in the storage
    auto cellIt = fhStaticBfwStorage.find(cell_id);
    if (cellIt == fhStaticBfwStorage.end())
    {
        return 0; // Indicate failure
    }
    return 1;
}

int FhProxy::registerMem(MemRegInfo const* memreg_info, MemRegHandle* memreg)
{
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    int           ret   = 0;

    if(aerial_fh::register_memory(fhi, memreg_info, memreg))
    {
        NVLOGE_FMT(TAG, AERIAL_ORAN_FH_EVENT, "aerial_fh::register_memory error");
        ret = 1;
    }

    return ret;
}

static __fh_always_inline uint8_t adjustPrbCount(uint16_t numPrbc)
{
    if(numPrbc == ORAN_MAX_PRB_X_SLOT)
    {
        return 0;
    }
    else if (numPrbc > ORAN_MAX_PRB_X_SECTION)
    {
        return ORAN_MAX_PRB_X_SECTION;
    }
    else
    {
        return static_cast<uint8_t>(numPrbc);
    }
}

static __fh_always_inline uint8_t adjustPrbCountMuMimo(uint16_t numPrbc, uint16_t prg_size)
{
    if(numPrbc == ORAN_MAX_PRB_X_SLOT)
    {
        return 0;
    }
    else if (numPrbc > ORAN_MAX_PRB_X_SECTION)
    {
        if ((prg_size) && (prg_size <= ORAN_MAX_PRB_X_SECTION))
        {
            numPrbc = (ORAN_MAX_PRB_X_SECTION/prg_size) * prg_size;
        }
        else
        {
            numPrbc = ORAN_MAX_PRB_X_SECTION;
        }
        return numPrbc;
    }
    else
    {
        return static_cast<uint8_t>(numPrbc);
    }
}

static __fh_always_inline uint16_t adjustPrbuCount(uint16_t numPrbu)
{
    if(numPrbu == ORAN_MAX_PRB_X_SLOT)
    {
        return numPrbu;
    }
    else if (numPrbu > ORAN_MAX_PRB_X_SECTION)
    {
        return ORAN_MAX_PRB_X_SECTION;
    }
    else
    {
        return numPrbu;
    }
}

inline void parsePortMask(uint64_t portMask,size_t &num_ap_indices,std::array<std::size_t,64> &ap_index_list)
 {
    num_ap_indices = __builtin_popcountll(portMask);
    uint8_t i =0;
    uint8_t j =0;
    while(j < num_ap_indices)
    {
        if(portMask & (1<< i)) //test for bit
        {
            ap_index_list[j++]=i;
        }
        i++;
    }
}

int FhProxy::sendCPlane_timingCheck(t_ns start_tx_time,t_ns start_ch_task_time,int direction)
{
    int ret=0;
    t_ns time_now = Time::nowNs();
    PhyDriverCtx*           pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    t_ns c_plane_timing_error_th(pdctx->getSendCPlane_timing_error_th_ns());
    if((start_tx_time.count()-time_now.count())<c_plane_timing_error_th.count()) //Timing error check
    {
        if(DIRECTION_DOWNLINK==(oran_pkt_dir)direction)
        {
            NVLOGW_FMT(TAG,"{} : sendCPlane Timing error for DLC start_tx_time {} current_time {} threshold: {}",__func__,start_tx_time.count(),time_now.count(), c_plane_timing_error_th.count());
            ret=SEND_DL_CPLANE_TIMING_ERROR;
        }
        else
        {
            ret=SEND_UL_CPLANE_TIMING_ERROR;
            NVLOGW_FMT(TAG,"{} : sendCPlane Timing error for ULC start_tx_time {} current_time {}",__func__,start_tx_time.count(),time_now.count());
        }
    }
    return ret;
}

int increment_section_count(uint16_t& section_count, int max, int flow)
{
    if(section_count + 1 >= max)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Too many sections for flow {}!", flow);
        return -1;
    }
    ++section_count;
    return 0;
}

void fill_section_ext4_ext5(slot_command_api::mod_comp_info_t& comp_info, fhproxy_cmsg_section& section_info, std::array<fhproxy_cmsg_section_ext,MAX_CPLANE_SECTIONS_EXT_PER_SLOT> &section_ext_infos, size_t& section_ext_index)
{
    auto& section_ext_info                       = section_ext_infos[section_ext_index++];
    section_ext_info.sect_ext_common_hdr.ef      = (section_info.ext11 == nullptr) ? 0 : 1;
    if(comp_info.common.nSections == 1)
    {
        section_ext_info.sect_ext_common_hdr.extType = ORAN_CMSG_SECTION_EXT_TYPE_4;
        section_ext_info.ext_4.ext_hdr.extLen        = 1;
        section_ext_info.ext_4.ext_hdr.csf           = comp_info.sections[0].csf;
 
        section_ext_info.ext_4.ext_hdr.modCompScalor = comp_info.sections[0].mcScaleOffset;
        section_info.ext4                            = &section_ext_info;
#if 0
        NVLOGC_FMT(TAG, "mcScaleOffset {} val {} mcScaleOffsetAsInt {} ", __half2float(mcScaleOffset), val, mcScaleOffsetAsInt);
        NVLOGC_FMT(TAG, "File {} Line {} prb_info udIqWidth: 0x{:x} , ext4:  mcScaleReMask=0x{:x}  mcScaleOffset=0x{:x}, csf=0x{:x}, modCompScalor=0x{:x} ", __FILE__, __LINE__, +comp_info.common.udIqWidth.get(), comp_info.sections[0].mcScaleReMask.get(), (uint16_t)comp_info.sections[0].mcScaleOffset.get(), comp_info.sections[0].csf.get(), (uint16_t)val);
#endif

    }
    else if(comp_info.common.nSections == 2)
    {
        section_ext_info.sect_ext_common_hdr.extType      = ORAN_CMSG_SECTION_EXT_TYPE_5;
        section_ext_info.ext_5.ext_hdr.extLen             = 3;
        section_ext_info.ext_5.ext_hdr.csf_1              = comp_info.sections[0].csf;

        section_ext_info.ext_5.ext_hdr.mcScaleOffset_1    = comp_info.sections[0].mcScaleOffset;
        section_ext_info.ext_5.ext_hdr.mcScaleReMask_1    = comp_info.sections[0].mcScaleReMask;
        section_ext_info.ext_5.ext_hdr.csf_2              = comp_info.sections[1].csf;

        section_ext_info.ext_5.ext_hdr.mcScaleOffset_2    = comp_info.sections[1].mcScaleOffset;
        section_ext_info.ext_5.ext_hdr.mcScaleReMask_2    = comp_info.sections[1].mcScaleReMask;
        section_ext_info.ext_5.ext_hdr.zero_padding       = 0;
        section_ext_info.ext_5.ext_hdr.extra_zero_padding = 0;
        section_info.ext5                                 = &section_ext_info;

#if 0
        NVLOGD_FMT(TAG, "mcScaleOffset_1 {} val1 {} mcScaleOffset_1AsInt {} mcScaleOffset_2 {} val2 {} mcScaleOffset_2AsInt {}", __half2float(mcScaleOffset_1), val1, mcScaleOffset_1AsInt, __half2float(mcScaleOffset_2), val2, mcScaleOffset_2AsInt);
        NVLOGC_FMT(TAG, "File {} Line {} prb_info udIqWidth: 0x{:x}, ext5_set1:  mcScaleReMask=0x{:x}  mcScaleOffset=0x{:x}, csf=0x{:x}, modCompScalor=0x{:x}, ext5_set2:  mcScaleReMask=0x{:x}  mcScaleOffset=0x{:x}, csf=0x{:x}, modCompScalor=0x{:x}", __FILE__, __LINE__,  +comp_info.common.udIqWidth.get(), comp_info.sections[0].mcScaleReMask.get(), comp_info.sections[0].mcScaleOffset.get(), comp_info.sections[0].csf.get(), val1 , comp_info.sections[1].mcScaleReMask.get(), comp_info.sections[1].mcScaleOffset.get(), comp_info.sections[1].csf.get(),
        val2);
#endif

    }
    else
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Modulation compression nSections error {}!", comp_info.common.nSections);
    }
    section_info.sect_1.ef = 1;
}

void FhProxy::fill_dynamic_section_ext11(const slot_command_api::oran_slot_ind& slot_indication, DynamicSectionExt11Params& params)
{
    auto& bfwCoeff_buf_info = params.prb_info.bfwCoeff_buf_info;
    int bundle_size = 0;
    if(bfwCoeff_buf_info.p_buf_bfwCoef_h == nullptr)
    {
        return;
    }
    params.section_info.sect_1.ef = 1;
    auto& section_ext_info = params.section_ext_infos[params.section_ext_index++];
    if(params.section_ext_index == MAX_CPLANE_SECTIONS_EXT_PER_SLOT)
    {
        NVLOGC_FMT(TAG, "MAX section ext reached");
    }

    section_ext_info.ext_11.static_bfw    = false;
    section_ext_info.ext_11.numPrbBundles = (params.numPrbc + bfwCoeff_buf_info.prg_size - 1) / bfwCoeff_buf_info.prg_size;
    section_ext_info.ext_11.numBundPrb    = bfwCoeff_buf_info.prg_size;
    section_ext_info.sect_ext_common_hdr.extType = ORAN_CMSG_SECTION_EXT_TYPE_11;
    // Fix: When prg_size covers entire bandwidth (>= numPrbc), set numBundPrb to numPrbc
    // since the section will be split and numBundPrb should reflect actual PRBs in the bundle
    if (bfwCoeff_buf_info.prg_size >= static_cast<uint32_t>(params.numPrbc)) {
        section_ext_info.ext_11.ext_hdr.numBundPrb = static_cast<uint8_t>(std::min(params.numPrbc, 255));
    } else {
        section_ext_info.ext_11.ext_hdr.numBundPrb = static_cast<uint8_t>(std::min(static_cast<int>(bfwCoeff_buf_info.prg_size), 255));
    }
    section_ext_info.ext_11.ext_hdr.disableBFWs = params.disableBFWs;
    section_ext_info.ext_11.ext_hdr.RAD = params.RAD;
    section_ext_info.ext_11.ext_hdr.reserved = 0;
    section_ext_info.ext_11.ext_comp_hdr.bfwCompMeth = static_cast<uint8_t>(UserDataBFWCompressionMethod::BLOCK_FLOATING_POINT);
    section_ext_info.ext_11.ext_comp_hdr.bfwIqWidth = params.bfwIQBitwidth;
    params.section_info.sect_1.beamId = 0x7FFF;
    section_ext_info.sect_ext_common_hdr.ef = 0;
    params.L_TRX = bfwCoeff_buf_info.nGnbAnt;

    auto& numPrbBundles = section_ext_info.ext_11.numPrbBundles;

    int extLenBytes = sizeof(oran_cmsg_ext_hdr) + sizeof(oran_cmsg_sect_ext_type_11);

    if(params.disableBFWs == 0)
    {
        extLenBytes += sizeof(oran_cmsg_sect_ext_type_11_disableBFWs_0_bfwCompHdr);

        section_ext_info.ext_11.bundle_hdr_size = oran_cmsg_get_bfw_bundle_hdr_size(static_cast<UserDataBFWCompressionMethod>(section_ext_info.ext_11.ext_comp_hdr.bfwCompMeth.get()));
        // dMIMO: pad the SE11 weights up to the full antenna array on the wire. Only params.L_TRX
        // (= nGnbAnt, e.g. 32 for PE1) weights are actually computed; the remaining antennas are
        // zero-filled in populate_se11_disablebfw_0_bundles. real_bfwIQ_size keeps the coeff-buffer
        // stride at the computed width; bfwIQ_size is the (possibly wider) on-the-wire width.
        constexpr int BFW_WIRE_NUM_ANT = 64; // full dual-PE array; un-sounded antennas zero-filled
        int wire_L_TRX = std::max<int>(params.L_TRX, BFW_WIRE_NUM_ANT);
        section_ext_info.ext_11.real_bfwIQ_size = (params.L_TRX * params.bfwIQBitwidth * 2) / 8;
        section_ext_info.ext_11.bfwIQ_size = (wire_L_TRX * params.bfwIQBitwidth * 2) / 8;
        section_ext_info.ext_11.bundle_size = section_ext_info.ext_11.bfwIQ_size + section_ext_info.ext_11.bundle_hdr_size;
    }
    else
    {
        section_ext_info.ext_11.bundle_hdr_size = sizeof(oran_cmsg_sect_ext_type_11_disableBFWs_1_bundle);
        section_ext_info.ext_11.bfwIQ_size = 0;
        section_ext_info.ext_11.real_bfwIQ_size = 0;
        section_ext_info.ext_11.bundle_size = section_ext_info.ext_11.bundle_hdr_size;
    }
    int bundleLenBytes = section_ext_info.ext_11.bundle_hdr_size + section_ext_info.ext_11.bfwIQ_size;
    extLenBytes += numPrbBundles * bundleLenBytes;
    section_ext_info.ext_11.ext_hdr.extLen = ((extLenBytes + 3) / 4); //pad to 4 byte boundary
    int section_bundle_start_index = params.section_ext_11_bundle_index;

    auto cur_prb = params.section_info.sect_1.startPrbc.get();

    if(bfw_c_plane_chaining_mode == aerial_fh::BfwCplaneChainingMode::NO_CHAINING || params.prb_info.common.direction == slot_command_api::FH_DIR_UL)
    {
        // Assumption 1. PDSCH will always start at a multiple of prgsize
        // Assumption 2. when PDSCH and CSIRS overlap, it starts at a multiple of prgsize
        // cuPHY fills buffer from index 0 regardless of startPrbc, but account for the section split due to 255 numPrbc limit
        // params.startPrbc is the adjusted startPrbc to account for above comment.
        int start_bundle_offset = params.startPrbc / bfwCoeff_buf_info.prg_size;
        for(int bundle_index = start_bundle_offset; bundle_index < start_bundle_offset + numPrbBundles; ++bundle_index)
        {
            if(unlikely(params.section_ext_11_bundle_index == MAX_CPLANE_EXT_11_BUNDLES_PER_SLOT))
            {
                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "MAX section ext bundles reached");
            }
            auto& bundle_info = params.section_ext_11_bundle_infos[params.section_ext_11_bundle_index++];

            if(params.disableBFWs == 0)
            {
                params.bfw_beam_id[cur_prb] = params.dyn_beam_id++;
                if(unlikely(params.dyn_beam_id > params.slot_dyn_beam_id_offset + dynamic_beam_ids_per_slot))
                {
                    NVLOGW_FMT(TAG, "MAX number of dynamic beam ids reached");
                }
                
                memset(&bundle_info.disableBFWs_0_compressed, 0, sizeof(bundle_info.disableBFWs_0_compressed)); 

                params.bfw_seen[cur_prb] = true;
                bundle_info.disableBFWs_0_compressed.beamId = params.bfw_beam_id[cur_prb];
                // ru_type::MULTI_SECT_MODE does not send the TX precoding and beamforming PDU, we can set the inner beamId to a dummy value.
                auto buffer_ptr = reinterpret_cast<uint8_t*>(bfwCoeff_buf_info.p_buf_bfwCoef_h);
                // Note1: in the representation: nGnbAnt x nPrbGrpBfw x nLayers, nGnbAnt is the innermost i.e. fastest changing dimension
                // and nLayers is the outermost dimension i.e. slowest changing dimension
                // Note2: for compressed beamforming coefficients, the block floating point exponent is prefixed to every nGnbAnt coefficients
                // (see field bfwCompParam in Table 7.7.11.1-1 in O-RAN.WG4.CUS.0-v10.00)
                // nLayers dimension

                // Stride on the REAL (computed) coeff width, not the padded wire width: the cuPHY
                // coeff buffer holds exponent + real_bfwIQ_size coefficients per bundle.
                int buffer_index = params.active_ap_idx * bfwCoeff_buf_info.num_prgs * (section_ext_info.ext_11.real_bfwIQ_size + 1); // +1 for exponent
                buffer_index += bundle_index * (section_ext_info.ext_11.real_bfwIQ_size + 1); // +1 for exponent
                bundle_info.disableBFWs_0_compressed.bfwCompParam.exponent = buffer_ptr[buffer_index++];
                bundle_info.bfwIQ = &buffer_ptr[buffer_index];
                *params.bfw_header = bfwCoeff_buf_info.header;
            }
            else
            {
                memset(&bundle_info.disableBFWs_1, 0, sizeof(bundle_info.disableBFWs_1)); 
                bundle_info.disableBFWs_1.beamId = params.bfw_beam_id[cur_prb];
            }
            cur_prb += bfwCoeff_buf_info.prg_size;
        }
    }
    else
    {        
        // Use absolute startPrbc to calculate beamId with start_bundle_offset

        int start_bundle_offset = params.startPrbc / bfwCoeff_buf_info.prg_size;
        int start_beamid_offset = (params.section_info.sect_1.startPrbc.get() - params.startPrbc) / CUPHY_BFW_MIN_PRB_GRP_SIZE;
        // NVLOGC_FMT(TAG, "start_bundle_offset {}, start_beamid_offset {}", start_bundle_offset, start_beamid_offset);

        if(params.disableBFWs == 0)
        {
            *params.bfw_header = bfwCoeff_buf_info.header;
            uint8_t* bfw_chaining_buffer = (bfw_c_plane_chaining_mode == aerial_fh::BfwCplaneChainingMode::GPU_CHAINING)
                                            ? bfwCoeff_buf_info.p_buf_bfwCoef_d
                                            : bfwCoeff_buf_info.p_buf_bfwCoef_h;
            size_t offset = static_cast<size_t>(params.active_ap_idx) *
                            static_cast<size_t>(bfwCoeff_buf_info.num_prgs) *
                            static_cast<size_t>(section_ext_info.ext_11.bundle_size);

            // Sanity-check buffer size before dereference
            size_t buf_size = getBfwCoeffSize();
            if (unlikely(offset + section_ext_info.ext_11.bundle_size > buf_size))
            {
                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT,
                           "BFW chaining buffer overflow: offset {} > size {}", offset, buf_size);
                return; // or throw / early-exit
            }

            section_ext_info.ext_11.bfwIQ = &bfw_chaining_buffer[offset];
            
            for(int bundle_index = start_bundle_offset; bundle_index < start_bundle_offset + numPrbBundles; ++bundle_index)
            {
                // IQ AND Beam IDs filled out by kernel, use the same logic as kernel
                params.bfw_seen[cur_prb] = true;
                cur_prb += bfwCoeff_buf_info.prg_size;
            }

        } else {

            for(int bundle_index = start_bundle_offset; bundle_index < start_bundle_offset + numPrbBundles; ++bundle_index)
            {
                if(unlikely(params.section_ext_11_bundle_index == MAX_CPLANE_EXT_11_BUNDLES_PER_SLOT))
                {
                    NVLOGC_FMT(TAG, "MAX section ext bundles reached");
                }
                auto& bundle_info = params.section_ext_11_bundle_infos[params.section_ext_11_bundle_index++];
                memset(&bundle_info.disableBFWs_1, 0, sizeof(bundle_info.disableBFWs_1)); 

                bundle_info.disableBFWs_1.beamId = params.slot_dyn_beam_id_offset + params.active_ap_idx * CUPHY_BFW_N_MAX_PRB_GRPS + bundle_index + start_beamid_offset;
            }
        }
    }
    section_ext_info.ext_11.bundles = &params.section_ext_11_bundle_infos[section_bundle_start_index];
    params.section_info.ext11 = &section_ext_info;

    params.section_info.ext11->ext_11.start_bundle_offset_in_bfw_buffer = params.start_bundle_offset_in_bfw_buffer;
}


void fill_static_section_ext(uint8_t* static_bfw_ptr, uint32_t cell_id, slot_command_api::prb_info_t& prb_info, fhproxy_cmsg_section &section_info, std::array<fhproxy_cmsg_section_ext,MAX_CPLANE_SECTIONS_EXT_PER_SLOT> &section_ext_infos, std::array<fhproxy_cmsg_section_ext_11_bundle_info,MAX_CPLANE_EXT_11_BUNDLES_PER_SLOT> &section_ext_11_bundle_infos,
    size_t& section_ext_index, size_t& section_ext_11_bundle_index, int bfwIQBitwidth, int disableBFWs, int RAD, int active_ap_idx, int actual_ap_idx, int startPrbc, int numPrbc, uint16_t static_beam_id)
{
    auto& bfwCoeff_buf_info = prb_info.static_bfwCoeff_buf_info;
    if(static_bfw_ptr == nullptr)
    {
        // No static BFW available, clear extension flag and set beamId
        section_info.sect_1.ef = 0;
        section_info.sect_1.beamId = static_beam_id;
        return;
    }
    section_info.sect_1.ef = 1;
    auto& section_ext_info = section_ext_infos[section_ext_index++];
    int extLenBytes = sizeof(oran_cmsg_ext_hdr) + sizeof(oran_cmsg_sect_ext_type_11) + sizeof(oran_cmsg_sect_ext_type_11_disableBFWs_0_bfwCompHdr);
    section_ext_info.sect_ext_common_hdr.ef = 0;
    section_ext_info.ext_11.bundle_hdr_size = sizeof(oran_cmsg_sect_ext_type_11_disableBFWs_0_bundle_uncompressed);
    int L_TRX = bfwCoeff_buf_info.nGnbAnt;
    section_ext_info.ext_11.bfwIQ_size = (L_TRX * bfwIQBitwidth * 2) / 8;
    section_ext_info.ext_11.real_bfwIQ_size = section_ext_info.ext_11.bfwIQ_size; // static BFW: no zero-padding
    int bundleLenBytes = section_ext_info.ext_11.bundle_hdr_size + section_ext_info.ext_11.bfwIQ_size;
    section_ext_info.ext_11.numPrbBundles = (numPrbc + bfwCoeff_buf_info.prg_size - 1) / bfwCoeff_buf_info.prg_size;
    section_ext_info.ext_11.static_bfw = true;
    section_ext_info.ext_11.numBundPrb = bfwCoeff_buf_info.prg_size;
    auto& numPrbBundles = section_ext_info.ext_11.numPrbBundles;
    extLenBytes += numPrbBundles * bundleLenBytes;
    section_ext_info.ext_11.ext_hdr.extLen = ((extLenBytes + 3) / 4); //pad to 4 byte boundary

    if(bfwCoeff_buf_info.num_prgs != 1)
    {
        NVLOGC_FMT(TAG, " fill_static_section_ext startPrbc {} numPrbc {} extType {} isStaticBfwEncoded {}", (int)prb_info.common.startPrbc, (int)prb_info.common.numPrbc, (int)prb_info.common.extType, (int)prb_info.common.isStaticBfwEncoded);
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "fill_static_section_ext11@fh.cpp. Wrong prg num: {}", bfwCoeff_buf_info.num_prgs);
    }
    disableBFWs = 0;

    section_ext_info.sect_ext_common_hdr.extType = ORAN_CMSG_SECTION_EXT_TYPE_11;
    // Fix: When prg_size covers entire bandwidth (>= numPrbc), set numBundPrb to numPrbc
    // since the section will be split and numBundPrb should reflect actual PRBs in the bundle
    if (bfwCoeff_buf_info.prg_size >= static_cast<uint32_t>(numPrbc)) {
        section_ext_info.ext_11.ext_hdr.numBundPrb = static_cast<uint8_t>(std::min(numPrbc, 255));
    } else {
        section_ext_info.ext_11.ext_hdr.numBundPrb = static_cast<uint8_t>(std::min(static_cast<int>(bfwCoeff_buf_info.prg_size), 255));
    }
    section_ext_info.ext_11.ext_hdr.disableBFWs = disableBFWs;
    section_ext_info.ext_11.ext_hdr.RAD = RAD;
    section_ext_info.ext_11.ext_hdr.reserved = 0;
    section_ext_info.ext_11.ext_comp_hdr.bfwCompMeth = static_cast<uint8_t>(UserDataCompressionMethod::NO_COMPRESSION);
    section_ext_info.ext_11.ext_comp_hdr.bfwIqWidth = bfwIQBitwidth;
    section_info.sect_1.beamId = 0x7FFF;
    int section_bundle_start_index = section_ext_11_bundle_index;
    // Assumption 1. PDSCH will always start at a multiple of prgsize
    // Assumption 2. when PDSCH and CSIRS overlap, it starts at a multiple of prgsize
    // cuPHY fills buffer from index 0 regardless of startPrbc, but account for the section split due to 255 numPrbc limit
    int start_bundle_offset = startPrbc / bfwCoeff_buf_info.prg_size;
    for(int bundle_index = start_bundle_offset; bundle_index < start_bundle_offset + numPrbBundles; ++bundle_index)
    // cuPHY fills buffer from index 0 regardless of startPrbc
    // for(int bundle_index = 0; bundle_index < numPrbBundles; ++bundle_index)
    {
        if(section_ext_11_bundle_index == MAX_CPLANE_EXT_11_BUNDLES_PER_SLOT)
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "MAX section ext reached");
        }
        auto& bundle_info = section_ext_11_bundle_infos[section_ext_11_bundle_index++];
        memset(&bundle_info.disableBFWs_0_uncompressed, 0, sizeof(bundle_info.disableBFWs_0_uncompressed)); 

        // ru_type::MULTI_SECT_MODE does not send the TX precoding and beamforming PDU, we can set the inner beamId to a dummy value.
        bundle_info.disableBFWs_0_uncompressed.beamId = static_beam_id;
        bundle_info.bfwIQ = static_bfw_ptr;
    }
    section_ext_info.ext_11.bundles = &section_ext_11_bundle_infos[section_bundle_start_index];
    section_info.ext11 = &section_ext_info;
}

int FhProxy::sendCPlaneMMIMO(
    bool is_bfw,
    uint32_t cell_id,
    peer_id_t peer_id,
    oran_pkt_dir direction,
    ti_subtask_info &ti_info)
{

    int                  buf_idx        = (direction == DIRECTION_DOWNLINK) ? 0 : 1;
    auto                 &message_index = cplane_prepare_buffer.at(buf_idx).at(cell_id).message_index_; 
    auto                 &message_infos = cplane_prepare_buffer.at(buf_idx).at(cell_id).message_infos_;
    struct fh_peer_t     *peer_ptr      = getPeerFromId(peer_id);

    if (unlikely(!peer_ptr))
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Peer {} is not registered!", peer_id); 
        return SEND_CPLANE_FUNC_ERROR; 
    }

    if (is_bfw) 
    {
        if (message_index[fh_msg_info_type::BFW_MSG_INFO] > 0) 
        {
            char sbuf[MAX_SUBTASK_CHARS];
            // sym-id is encoded to be 0 to be forward looking to allow for future updates 
            // to task organization on a per-symbol basis
            snprintf(sbuf, sizeof(sbuf), "bfw_prepare cell_%d sym_%d", cell_id, 0); 
            TI_SUBTASK_INFO_ADD(ti_info,sbuf);
            if(0 == send_cplane_mmimo(peer_ptr->peer, const_cast<const fhproxy_cmsg*>(&message_infos[fh_msg_info_type::BFW_MSG_INFO][0]), message_index[fh_msg_info_type::BFW_MSG_INFO]))
            {
                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "aerial_fh::send_cplane (bfw) error cell {}", cell_id);
                return SEND_CPLANE_FUNC_ERROR;
            }
        }
    } else 
    {
        if (message_index[fh_msg_info_type::NON_BFW_MSG_INFO] > 0) 
        {
            char sbuf[MAX_SUBTASK_CHARS];
            // sym-id is encoded to be 0 to be forward looking to allow for future updates 
            // to task organization on a per-symbol basis
            snprintf(sbuf, sizeof(sbuf), "nonbfw_prepare cell_%d sym_%d", cell_id, 0); 
            TI_SUBTASK_INFO_ADD(ti_info,sbuf);
            if(0 == send_cplane_mmimo(peer_ptr->peer, const_cast<const fhproxy_cmsg*>(&message_infos[fh_msg_info_type::NON_BFW_MSG_INFO][0]), message_index[fh_msg_info_type::NON_BFW_MSG_INFO]))
            {
                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "aerial_fh::send_cplane (non-bfw) error cell {}", cell_id);
                return SEND_CPLANE_FUNC_ERROR;
            }
        }
    }
    return SEND_CPLANE_NO_ERROR;
}

int FhProxy::prepareCPlaneInfo(
    uint32_t cell_id,
    ru_type ru,
    peer_id_t peer_id,
    uint16_t dl_comp_meth,
    t_ns start_tx_time,
    uint64_t tx_cell_start_ofs_ns,
    oran_pkt_dir direction,
    const slot_command_api::oran_slot_ind &slot_indication,
    slot_command_api::slot_info_t &slot_info,
    uint16_t time_offset,
    int16_t dyn_beam_id_offset,
    uint8_t frame_structure,
    uint16_t cp_length,
    uint8_t** bfw_header,
    t_ns start_ch_task_time,
    int  prevSlotBfwCompStatus,
    ti_subtask_info &ti_info
    )
{
    //NVLOGC_FMT(TAG, "sendCPlane: oframe_id_ {} osfid_ {} oslotid_ {}  dyn_beam_id_offset {}", slot_indication.oframe_id_, slot_indication.osfid_, slot_indication.oslotid_, dyn_beam_id_offset);
    int ret;
    if((ret = sendCPlane_timingCheck(start_tx_time, start_ch_task_time, direction)) != SEND_CPLANE_NO_ERROR)
    {
        NVLOGW_FMT(TAG, "sendCPlane_timingCheck: Error at oframe_id_ {} osfid_ {} oslotid_ {}", slot_indication.oframe_id_, slot_indication.osfid_, slot_indication.oslotid_);
        slot_info.section_id_ready.store(true);
        return ret;
    }
    struct fh_peer_t *peer_ptr = getPeerFromId(peer_id);
    if(!peer_ptr)
    {
        slot_info.section_id_ready.store(true);
        return SEND_CPLANE_FUNC_ERROR;
    }
    start_tx_time += t_ns(AppConfig::getInstance().getTaiOffset());

    fhproxy_cmsg_section section_infos[MAX_CPLANE_SECTIONS_PER_SLOT];
    
    // Ping-pong per direction
    int buf_idx = (direction == DIRECTION_DOWNLINK) ? 0 : 1;
    auto &section_ext_infos = cplane_prepare_buffer.at(buf_idx).at(cell_id).section_ext_infos_; 
    auto &section_ext_11_bundle_infos = cplane_prepare_buffer.at(buf_idx).at(cell_id).section_ext_11_bundle_infos_; 

    int                 disableBFWs = 0;
    int                 RAD = 0;
    int                 numBundPrb = 0;
    int                 startPrb = 0;
    int                 numPrb = 0;
    int                 bfwIQBitwidth = 9; // FIXME use configurable
    int                 staticBFWIQBitwidth = 16; // FIXME use configurable
    int                 L_TRX = slot_command_api::NUM_GNB_TX_RX_ANT_PORTS;//nRxAnts, getRxAntennas

    PhyDriverCtx*           pdctx = StaticConversion<PhyDriverCtx>(pdh).get();

    uint16_t            start_section_id_srs = pdctx->getStartSectionIdSrs();
    uint16_t            start_section_id_prach  = pdctx->getStartSectionIdPrach();

    size_t               section_index = 0;
    size_t               section_ext_index = 0;
    size_t               section_ext_11_bundle_index = 0;
    size_t               start_section_index = 0;
    size_t               next_start_section_index = 0;
    t_ns                 tx_time     = t_ns(SYMBOL_DURATION_NS);
    t_ns                 end_tx_time = start_tx_time + tx_time;
    uint16_t             ud_comp_hdr = 0;
    uint16_t             section_id = 0;
    uint16_t             section_id_srs = start_section_id_srs;
    uint16_t             section_id_prach = start_section_id_prach;

    // TODO generalize. Support other compression methods
    // Based on nvbug 4189837 we will set udCompHdr to 0x0 for DL and UL C plane
    // if (peer_ptr->dl_bit_width == BFP_COMPRESSION_9_BITS || peer_ptr->dl_bit_width == BFP_COMPRESSION_14_BITS)
    // {
    //     ud_comp_hdr = ((peer_ptr->dl_bit_width & 0xF) << 4) | 0x1;
    // }

    int start_channel_type;
    int end_channel_type;
    fhproxy_cmsg message_template;
    fhproxy_cmsg_section section_template={};
    message_template.data_direction = direction;
    message_template.hasSectionExt = false;
    uint8_t start_symbol = 0;
    if (direction == DIRECTION_DOWNLINK)
    {
        start_channel_type = slot_command_api::channel_type::PDSCH_CSIRS;
        end_channel_type = slot_command_api::channel_type::PDCCH_DMRS;
        start_symbol = (ru == SINGLE_SECT_MODE ? slot_info.start_symbol_dl: 0);
    }
    else
    {
        start_channel_type = slot_command_api::channel_type::PUSCH;
        end_channel_type = (ru == SINGLE_SECT_MODE) ? slot_command_api::channel_type::PRACH : slot_command_api::channel_type::SRS; // Don't send SRS for SINGLE_SECT_MODE
        start_symbol = 0; //(ru == SINGLE_SECT_MODE ? slot_info.start_symbol_ul: 0);
    }

    // Populate the fields shared between all type C-plane messages, regardless of the section type
    auto& radio_app_hdr         = message_template.section_common_hdr.sect_1_common_hdr.radioAppHdr;
    radio_app_hdr.payloadVersion = ORAN_DEF_PAYLOAD_VERSION;
    radio_app_hdr.filterIndex    = ORAN_DEF_FILTER_INDEX;
    
    auto& frame_id = slot_indication.oframe_id_;
    auto& subframe_id = slot_indication.osfid_;
    auto& slot_id = slot_indication.oslotid_;
    radio_app_hdr.frameId        = slot_indication.oframe_id_;
    radio_app_hdr.subframeId     = slot_indication.osfid_;
    radio_app_hdr.slotId         = slot_indication.oslotid_;

    auto& section_info    = section_template.sect_1;
    section_info.rb       = false;
    section_info.symInc   = false;
    section_info.ef       = 0;
    bool mod_comp_enabled = (direction == DIRECTION_DOWNLINK && static_cast<UserDataCompressionMethod>(dl_comp_meth) == UserDataCompressionMethod::MODULATION_COMPRESSION);
    
    if(pdctx->getmMIMO_enable())
    {
        
        std::array<std::array<uint16_t, MAX_AP_PER_SLOT_SRS>,fh_msg_info_type::NUM_MSG_INFO_TYPES> section_count;
        std::array<std::array<uint16_t, MAX_AP_PER_SLOT_SRS>,fh_msg_info_type::NUM_MSG_INFO_TYPES> start_section_count;
        
        // Choose the right buffers & counter arrays for current slot & cell. 
        auto                 &message_index = cplane_prepare_buffer.at(buf_idx).at(cell_id).message_index_; 
        auto                 &message_infos = cplane_prepare_buffer.at(buf_idx).at(cell_id).message_infos_;
        auto                 &section_infos_per_ant_per_msgtype = cplane_prepare_buffer.at(buf_idx).at(cell_id).section_infos_;
        auto                 &section_count_per_ant_per_msgtype = section_count;
        auto                 &start_section_count_per_ant_per_msgtype = start_section_count;

        // NOTE: These are non-atomic! For any concurrency between BFW and non-BFW tasks, these ought to be atomic.
        auto                 &section_id_per_ant = cplane_prepare_buffer.at(buf_idx).at(cell_id).section_id_per_ant_;
        auto                 &start_section_id_srs_per_ant = cplane_prepare_buffer.at(buf_idx).at(cell_id).start_section_id_srs_per_ant_;
        auto                 &start_section_id_prach_per_ant = cplane_prepare_buffer.at(buf_idx).at(cell_id).start_section_id_prach_per_ant_;

        uint64_t             sent_ext_11_port_mask[ORAN_ALL_SYMBOLS][ORAN_MAX_PRB_X_SLOT]{};
        std::array<csirs_section_id_info_t, MAX_FLOWS> csirs_section_id_map{};
        bool                 sent_ext_11[ORAN_ALL_SYMBOLS][ORAN_MAX_PRB_X_SLOT]{};

        message_index[fh_msg_info_type::BFW_MSG_INFO] = 0; 
        message_index[fh_msg_info_type::NON_BFW_MSG_INFO] = 0; 

        for(int i = 0; i < MAX_AP_PER_SLOT_SRS; ++i)
        {
            //start_section_count_per_ant[i] = 0;
            section_count_per_ant_per_msgtype[fh_msg_info_type::NON_BFW_MSG_INFO][i] = 0;
            section_count_per_ant_per_msgtype[fh_msg_info_type::BFW_MSG_INFO][i] = 0;
            section_id_per_ant[i] = 0;
            start_section_id_srs_per_ant[i] = start_section_id_srs;
            start_section_id_prach_per_ant[i] = start_section_id_prach;
        }

        uint16_t dyn_bfw_beam_id[MAX_AP_PER_SLOT][ORAN_MAX_PRB]{};
        bool     dyn_bfw_seen[MAX_AP_PER_SLOT][ORAN_MAX_PRB]{};
        //for each symbol
        auto cur_dyn_bfw_beam_id = (uint16_t)dyn_beam_id_offset;//dynamic_beam_id_start;
        for(int symbol_id = start_symbol; symbol_id < slot_info.symbols.size(); symbol_id++)
        {
            for(int ap_idx = 0; ap_idx < MAX_AP_PER_SLOT_SRS; ++ap_idx)
            {
                start_section_count_per_ant_per_msgtype[fh_msg_info_type::NON_BFW_MSG_INFO][ap_idx] = 0;
                start_section_count_per_ant_per_msgtype[fh_msg_info_type::BFW_MSG_INFO][ap_idx] = 0;
            }
            message_template.tx_window.tx_window_start = start_tx_time.count();
            // tx_cell_start_ofs_ns is 0 if cell-based time division is disabled
            message_template.tx_window.tx_window_bfw_start = tx_cell_start_ofs_ns + start_tx_time.count();
            message_template.tx_window.tx_window_end   = end_tx_time.count();
            radio_app_hdr.startSymbolId                = symbol_id;

            // for each channel
            for (int channel_type = start_channel_type; channel_type <= end_channel_type; channel_type++)
            {

                auto &flows               = peer_ptr->cplane_flows[channel_type];

                // TODO: if it is PDSCH+CSIRS

                // Use the correct prbs
                bool use_alt_csirs_list = false;
                const std::vector<std::size_t> * prb_index_list = nullptr;
                slot_command_api::prb_info_list_t prbs = nullptr;
                auto csirs_list_size = 0;
                // PDSCH_CSIRS has the overlapping portions of PDSCH and CSI_RS
                bool is_pdsch_csirs = (channel_type == slot_command_api::channel_type::PDSCH_CSIRS) && (slot_info.symbols[symbol_id][channel_type].size() > 0);
                if(is_pdsch_csirs)
                {
                    csirs_list_size = slot_info.symbols[symbol_id][slot_command_api::channel_type::CSI_RS].size();
                }
                // If it is CSI_RS and the overlapping PRBs CSI_RS and PDSCH is not 0
                if((channel_type == slot_command_api::channel_type::CSI_RS) &&
                    (slot_info.symbols[symbol_id][slot_command_api::channel_type::PDSCH_CSIRS].size() != 0))
                {
                    // Alt CSI_RS prb info index list is for the non-overlapping CSI_RS REs
                    if(slot_info.alt_csirs_prb_info_idx_list[symbol_id].size())
                    {
                        prb_index_list = &slot_info.alt_csirs_prb_info_idx_list[symbol_id];
                        prbs = const_cast<slot_command_api::prb_info_list_t>(slot_info.alt_csirs_prb_info_list);
                        use_alt_csirs_list = true;
                    }
                    else
                        continue;
                }
                else
                {
                    prb_index_list = &slot_info.symbols[symbol_id][channel_type];
                    prbs = const_cast<slot_command_api::prb_info_list_t>(slot_info.prbs);
                }

                for(int ap_idx = 0; ap_idx < flows.size(); ++ap_idx)
                {
                    start_section_count_per_ant_per_msgtype[fh_msg_info_type::NON_BFW_MSG_INFO][ap_idx] = section_count_per_ant_per_msgtype[fh_msg_info_type::NON_BFW_MSG_INFO][ap_idx];
                    start_section_count_per_ant_per_msgtype[fh_msg_info_type::BFW_MSG_INFO][ap_idx]     = section_count_per_ant_per_msgtype[fh_msg_info_type::BFW_MSG_INFO][ap_idx];
                }

                // for each prb_info
                for(auto prb_info_idx : *prb_index_list)
                {
                    auto &prb_info = prbs[prb_info_idx];
                    // Initialize beam_idx based on the first set bit in portMask.
                    // This is needed when prb_info is divided into multiple entries in the prb_info_list.
                    // (e.g., DMRS 4-port allocation, resulting in split into two entries in the prb_info_list with 
                    // portMask=0x3 for layers 0,1 and portMask=0xc for layers 2,3).
                    // The beam_idx should start at the layer offset to index correctly into beams_array.
                    int beam_idx = (prb_info.common.portMask != 0) ? __builtin_ctzll(prb_info.common.portMask) : 0;
                    auto &overlap_csirs_port_info = slot_info.overlap_csirs_port_info[prb_info_idx];
                    int csirs_beam_idx = 0;
                    int active_ap_idx = 0;
                    int overlaping_csirs_section_start_idx = 0;
                    int overlaping_csirs_section_end_idx = 0;
                    int csirs_beam_list_idx = 0;
                    slot_command_api::prb_info_t* csi_rs_prb_info   = nullptr; 
                    slot_command_api::beamid_array_t * beams_csirs_array   = nullptr;
                    size_t beams_csirs_array_size = 0;
                    uint16_t overlapping_csi_rs_remask = 0;
                    uint16_t re_mask = prb_info.common.reMask;
                    //For PDSCH_CSIRS, the PDSCH sections have same reMask for all ports
                    // but for overlapping CSI_RS sections, the reMask is different for different ports
                    // so we need to use the complement of the reMask for the CSI_RS sections
                    if(mod_comp_enabled) {
                        if(channel_type == slot_command_api::channel_type::PDSCH_CSIRS)
                        {
                            re_mask = ~prb_info.comp_info.sections[1].mcScaleReMask;
                        }
                    }
                    // If it is pdsch + sect 11, only send for the first symbol
                    // need to know what PRB ranges have been sent, start PRB and end PRB match, and startSym+numSym is covered
                    // if it is a new PRB range, then it means it is a new UEG

                    if(prb_info.common.extType == ORAN_CMSG_SECTION_EXT_TYPE_11 || channel_type == slot_command_api::channel_type::PDSCH || channel_type == slot_command_api::channel_type::PDSCH_CSIRS || channel_type == slot_command_api::channel_type::PBCH)
                    {
                        //if the remask of previously sent PRBs does not have the current re_mask then prb section
                        //needs to be set on a different port. 
                        if(mod_comp_enabled) {
                            if((sent_ext_11_port_mask[symbol_id][prb_info.common.startPrbc] & prb_info.common.portMask) == prb_info.common.portMask) {
                                // NVLOGC_FMT(TAG, "{} ##### sent_ext_11_port_mask={:x}, symbol_id={}, startPrbc={}, prb_portMask = {:x} and_operation = {:x}",__func__, sent_ext_11_port_mask[symbol_id][prb_info.common.startPrbc] , symbol_id, +prb_info.common.startPrbc, static_cast<uint64_t>(prb_info.common.portMask), (sent_ext_11_port_mask[symbol_id][prb_info.common.startPrbc] & prb_info.common.portMask));
                                continue;
                            }
                        } else {
                            if(sent_ext_11[symbol_id][prb_info.common.startPrbc])
                            {
                                continue;
                            }
                        }
                        auto tempStartSym = symbol_id;
                        auto tempEndSym   = symbol_id + prb_info.common.numSymbols;
                        for(int temp_sym_id = tempStartSym; temp_sym_id < tempEndSym; ++temp_sym_id)
                        {
                            auto tempStartPrb = prb_info.common.startPrbc;
                            auto tempEndPrb   = prb_info.common.startPrbc + prb_info.common.numPrbc;
                            for(int temp_prb_idx = tempStartPrb; temp_prb_idx < tempEndPrb; temp_prb_idx++)
                            {
                                if(mod_comp_enabled) {
                                    sent_ext_11_port_mask[temp_sym_id][temp_prb_idx] |= prb_info.common.portMask;
                                } else {
                                    sent_ext_11[temp_sym_id][temp_prb_idx] = true;
                                }
                            }
                        }
                    }

                    if(channel_type == slot_command_api::channel_type::SRS)
                    {
                        prb_info.common.portMask = (flows.size()==L_TRX ?(0xFFFFFFFFFFFFFFFF):((1ULL << flows.size()) - 1));
                        //NVLOGC_FMT(TAG, "##### flows.size={}, portMask = {}", flows.size(), static_cast<uint64_t>(prb_info.common.portMask));
                    }
#if 0
                    if(channel_type == slot_command_api::channel_type::PDSCH_CSIRS)
                    {
                        NVLOGC_FMT(TAG, "{} ##### prb_info={}, channel_type={}, flows.size={}, portMask = {}",__func__, static_cast<void *>(&prb_info), channel_type, flows.size(), static_cast<uint64_t>(prb_info.common.portMask));
                        NVLOGC_FMT(TAG, "{} F{}S{}S{} PRBINFO symbol {} Channel {} startPrbc {} numPrbc {} extType {}",__func__, slot_indication.oframe_id_, slot_indication.osfid_, slot_indication.oslotid_, symbol_id, channel_type, (int)prb_info.common.startPrbc, (int)prb_info.common.numPrbc, (int)prb_info.common.extType);
                    }
#endif
                    // NVLOGC_FMT(TAG, "F{}S{}S{} PRBINFO symbol {} Channel {} startPrbc {} numPrbc {} extType {}", slot_indication.oframe_id_, slot_indication.osfid_, slot_indication.oslotid_, symbol_id, channel_type, (int)prb_info.common.startPrbc, (int)prb_info.common.numPrbc, (int)prb_info.common.extType);
                    // for each antenna
                    auto num_ports = flows.size();
                    auto pdsch_comp_info = prb_info.comp_info;
                    auto csi_rs_comp_info = prb_info.comp_info;
                    if(mod_comp_enabled) {
                        if(is_pdsch_csirs)
                        {
                            //Using PDSCH data to fill section extension 11 of CSI_RS
                            csi_rs_prb_info = &prb_info;
                            beams_csirs_array = &prb_info.beams_array2;
                            beams_csirs_array_size = prb_info.beams_array_size2;
                            prb_info.common.reMask = prb_info.comp_info.sections[0].mcScaleReMask;
                            overlapping_csi_rs_remask = prb_info.comp_info.sections[1].mcScaleReMask;

                            pdsch_comp_info.common.nSections = 1;
                            csi_rs_comp_info.common.nSections =1;
                            csi_rs_comp_info.sections[0] = prb_info.comp_info.sections[1];
                        }
                        
                        if(channel_type == slot_command_api::channel_type::CSI_RS)
                        {
                            uint64_t portMask = prb_info.common.portMask;
                            int lastSetBit = -1;
                            if (portMask) {
                                lastSetBit = 63 - __builtin_clzll(portMask);
                                if(lastSetBit + 1> flows.size())
                                {
                                    num_ports = lastSetBit + 1;
                                }
                            }
                            
                        }
                    } else {
                        if(is_pdsch_csirs)
                        {
                            while((overlaping_csirs_section_start_idx < slot_info.symbols[symbol_id][slot_command_api::channel_type::CSI_RS].size()) &&
                                    ((prb_info.common.startPrbc + prb_info.common.numPrbc) >
                                    (prbs[slot_info.symbols[symbol_id][slot_command_api::channel_type::CSI_RS][overlaping_csirs_section_start_idx]].common.startPrbc +
                                    prbs[slot_info.symbols[symbol_id][slot_command_api::channel_type::CSI_RS][overlaping_csirs_section_start_idx]].common.numPrbc)))
                            {
                                overlaping_csirs_section_start_idx++;
                            }
                            csirs_beam_list_idx = overlaping_csirs_section_start_idx;

                            if(csirs_beam_list_idx < csirs_list_size)
                            {
                                    csi_rs_prb_info = &(prbs[slot_info.symbols[symbol_id][slot_command_api::channel_type::CSI_RS][csirs_beam_list_idx]]);
                                    beams_csirs_array      = &(csi_rs_prb_info->beams_array);
                                    beams_csirs_array_size = csi_rs_prb_info->beams_array_size;
                                    overlapping_csi_rs_remask = ~prb_info.common.reMask;
                            }
                        }
                    }
                    for(int ap_idx = 0; ap_idx < num_ports; ++ap_idx)
                    {
                        //Both UL and DL we will use the portMask to determine what antennas to send the C-plane
                        if(!(prb_info.common.portMask & (1 << ap_idx)))
                        {
                            //NVLOGC_FMT(TAG, "if condition ##### symbol_id={} numSymbols={} ap_idx={}, prb_info={}, channel_type={}, flows.size={}, portMask = {}", symbol_id, static_cast<uint8_t>(prb_info.common.numSymbols), ap_idx, static_cast<void *>(&prb_info), channel_type, flows.size(), static_cast<uint64_t>(prb_info.common.portMask));
                            continue;
                        }
                        else
                        {
                            //NVLOGC_FMT(TAG, "Else ##### symbol_id={}, numSymbols={}, ap_idx={}, prb_info={}, channel_type={}, flows.size={}, portMask = {}", symbol_id, static_cast<uint8_t>(prb_info.common.numSymbols), ap_idx, static_cast<void *>(&prb_info), channel_type, flows.size(), static_cast<uint64_t>(prb_info.common.portMask));
                        }
                        
                        // This single-ton copy is pre-filled until the point when it is determined whether this applies to a packet w/ or w/o BFW.
                        auto&                                   beams_array            = prb_info.beams_array;
                        size_t                                  beams_array_size       = prb_info.beams_array_size;
                        auto                                    beam_id                = 0;
                        auto                                    csirs_beam_id          = 0;
                        auto                                    num_prbc               = prb_info.common.numPrbc;
                        auto                                    start_prbc             = prb_info.common.startPrbc;
                        auto                                    num_prbc_split         = ORAN_MAX_PRB_X_SECTION;
                        auto                                    flow_idx               = ap_idx % flows.size();
                        csirs_section_id_info_t& csirs_section_id_info = csirs_section_id_map[flow_idx];
                        radio_app_hdr.dataDirection   = prb_info.common.direction;
                        
                        // A singleton local instance that gets populated until BFW vs non-BFW decision is known
                        fhproxy_cmsg_section section_infos_per_ant_singleton{};
                        auto& section_info = section_infos_per_ant_singleton;
                        section_info.sect_1.rb        = false;
                        section_info.sect_1.symInc    = false;
                        section_info.sect_1.ef        = 0;
                        section_info.sect_1.startPrbc = start_prbc;
                        section_info.ext4 = nullptr;
                        section_info.ext5 = nullptr;
                        section_info.ext11 = nullptr;

                        bool csirs_section_only        = (channel_type == slot_command_api::channel_type::PDSCH_CSIRS && !(prb_info.common.pdschPortMask & (1 << ap_idx)) && csi_rs_prb_info != nullptr && (csi_rs_prb_info->common.portMask & (1 << ap_idx)));
                        bool pdsch_section_only        = (channel_type == slot_command_api::channel_type::PDSCH_CSIRS && (prb_info.common.pdschPortMask & (1 << ap_idx)) && csi_rs_prb_info != nullptr && !(csi_rs_prb_info->common.portMask & (1 << ap_idx)));
                        bool both_pdsch_csirs_included = channel_type == slot_command_api::channel_type::PDSCH_CSIRS && (prb_info.common.pdschPortMask & (1 << ap_idx)) && csi_rs_prb_info != nullptr && (csi_rs_prb_info->common.portMask & (1 << ap_idx));

                        if(both_pdsch_csirs_included || csirs_section_only)
                        {
                            if(beams_csirs_array_size > 0)
                            {
                                beams_csirs_array_size = std::min(beams_csirs_array_size, overlap_csirs_port_info.num_ports != 0 ? overlap_csirs_port_info.num_ports : num_ports); //TODO: This needs to be changed to num_ports
                                if(overlap_csirs_port_info.num_overlap_ports > 0)
                                {
                                    csirs_beam_idx = overlap_csirs_port_info.reMask_ap_idx_pairs[0].second % beams_csirs_array_size;
                                    csirs_beam_id          = (*beams_csirs_array)[csirs_beam_idx];
                                }
                                else
                                {
                                    csirs_beam_id          = (*beams_csirs_array)[csirs_beam_idx % beams_csirs_array_size];
                                    csirs_beam_idx         = (csirs_beam_idx + 1) % beams_csirs_array_size; //TODO: Need to check if the beam_idx is correct
                                }
                            } 
                        }

                        section_info.sect_1.reMask = (csirs_section_only ? ~prb_info.common.reMask : (pdsch_section_only ? 0xfff : prb_info.common.reMask));

                        section_info.sect_1.numSymbol = prb_info.common.numSymbols;
                        section_info.sect_1.rb        = prb_info.common.useAltPrb ? 1 : 0;
                        radio_app_hdr.filterIndex     = prb_info.common.filterIndex;

                        section_info.prb_info                       = &prb_info;
                        section_info.csirs_of_multiplex_pdsch_csirs = false;

                        if (channel_type == slot_command_api::channel_type::PRACH)
                        {
                            radio_app_hdr.sectionType = ORAN_CMSG_SECTION_TYPE_3;

                            auto& sect_3_common_hdr          = message_template.section_common_hdr.sect_3_common_hdr;
                            sect_3_common_hdr.timeOffset     = time_offset;
                            sect_3_common_hdr.frameStructure = frame_structure;
                            sect_3_common_hdr.cpLength       = cp_length;
                            sect_3_common_hdr.udCompHdr      = ud_comp_hdr;

                            auto &section_3_info      = section_info.sect_3;
                            //section_3_info.sectionId  = start_section_id_prach_per_ant[ap_idx]++;
                            section_3_info.freqOffset = prb_info.common.freqOffset;
                            section_3_info.reserved   = 0;
                        }
                        else
                        {
                            radio_app_hdr.sectionType     = ORAN_CMSG_SECTION_TYPE_1;
                            if (channel_type == slot_command_api::channel_type::CSI_RS) {
                                // Manage section ID lookback index for CSI-RS compact signaling with section splits
                                //
                                // csirs_section_id_info[flow_idx].section_id_lookback_index tracks the lookback distance
                                // for NEW sections to reference the correct section ID when CSI-RS sections are split.
                                //
                                // Key behaviors:
                                // 1. After a split: Both resulting sections have lookback_index=0 (independent section IDs)
                                // 2. When split occurs: Increment csirs_section_id_info[flow_idx].section_id_lookback_index
                                // 3. For new sections: Increment csirs_section_id_info[flow_idx].section_id_lookback_index 
                                //    before assignment to correctly reference existing sections
                                //
                                // Example 1 - With section split:
                                //
                                //   Initial: csirs_section_id_info[flow_idx].section_id_lookback_index = 0
                                //   Section[0]: lookback=0, section_id=100 (reference section)
                                //
                                //   Section[0] splits due to PRB constraints:
                                //     csirs_section_id_info[flow_idx].section_id_lookback_index → 1
                                //     Section[0]: PRB 0-255,   lookback=0, section_id=100 (first half, independent)
                                //     Section[1]: PRB 255-end, lookback=0, section_id=101 (second half, independent)
                                //
                                //   New section arrives (same PRB set):
                                //     csirs_section_id_info[flow_idx].section_id_lookback_index: 1 → 2
                                //     Section[2]: lookback=2, section_id=100 (references Section[2-2]=Section[0])
                                //     
                                //   Section[2] splits:
                                //     Section[2]: PRB 0-255,   lookback=2, section_id=100 (keep lookback)
                                //     Section[3]: PRB 255-end, lookback=2, section_id=101 (references Section[3-2]=Section[1])
                                //
                                // Example 2 - Without section split:
                                //
                                //   Initial: csirs_section_id_info[flow_idx].section_id_lookback_index = 0
                                //   Section[0]: lookback=0, section_id=100 (reference section)
                                //
                                //   New section arrives (same PRB set, no split needed):
                                //     csirs_section_id_info[flow_idx].section_id_lookback_index: 0 → 1
                                //     Section[1]: lookback=1, section_id=100 (references Section[1-1]=Section[0])
                                //
                                // The lookback_index ensures new sections correctly pair with their corresponding
                                // reference sections (matching antenna port) to obtain the same section ID.
                                if(csirs_section_id_info.start_prb == start_prbc && csirs_section_id_info.num_prb == num_prbc && csirs_section_id_info.symbol == symbol_id)
                                {
                                    section_info.sect_1.sectionId = csirs_section_id_info.csirs_section_id;
                                    section_info.section_id_lookback_index = ++csirs_section_id_info.section_id_lookback_index;
                                }
                                else
                                {
                                    csirs_section_id_info.start_prb = start_prbc;
                                    csirs_section_id_info.num_prb = num_prbc;
                                    csirs_section_id_info.symbol = symbol_id;
                                    csirs_section_id_info.csirs_section_id = section_id_per_ant[flow_idx]++;
                                    section_info.sect_1.sectionId = csirs_section_id_info.csirs_section_id;
                                    section_info.section_id_lookback_index = 0;
                                    csirs_section_id_info.section_id_lookback_index = 0;
                                }
                                beam_idx = ap_idx; // This is done because for CSI-RS one PRB will carry data for only one ap_idx. So beam_idx doesn't get updated and remains 0.
                            } else {
                                section_info.sect_1.sectionId = channel_type == slot_command_api::channel_type::SRS ? start_section_id_srs_per_ant[flow_idx]++ : section_id_per_ant[flow_idx]++;
                            }
                            auto& sect_1_common_hdr       = message_template.section_common_hdr.sect_1_common_hdr;
                            sect_1_common_hdr.udCompHdr   = ud_comp_hdr;
                            sect_1_common_hdr.reserved    = 0;
                        }

                        if(beams_array_size > 0)
                        {
                            beams_array_size = std::min(beams_array_size, num_ports);
                            beam_id = beams_array[beam_idx % beams_array_size];
                            beam_idx = (beam_idx + 1) % beams_array_size; //TODO: Need to check if the beam_idx is correct
                        }

                        bool encode_ext_11      = (prb_info.common.extType == ORAN_CMSG_SECTION_EXT_TYPE_11 || (csi_rs_prb_info != nullptr && csi_rs_prb_info->common.extType == ORAN_CMSG_SECTION_EXT_TYPE_11));
                        bool encode_dyn_ext_11  = false;
                        bool encode_stat_ext_11 = false;
                        disableBFWs             = 0;
                        uint16_t prg_size       = 0;
                        if(encode_ext_11)
                        {
                            if(!is_pdsch_csirs) // Everything not PDSCH+CSIRS (includes PDSCH only and CSIRS only)
                            {
                                if(prb_info.common.isStaticBfwEncoded)
                                {
                                    prg_size = prb_info.static_bfwCoeff_buf_info.prg_size;
                                    encode_stat_ext_11 = true;
                                    encode_dyn_ext_11 = false;
                                }
                                else if(!prb_info.common.isStaticBfwEncoded) //Why not just use else?
                                {
                                    prg_size = prb_info.bfwCoeff_buf_info.prg_size;
                                    encode_stat_ext_11 = false;
                                    encode_dyn_ext_11 = true;
                                }
                            }
                            else // For PDSCH+CSI_RS
                            {
                                if(!csirs_section_only)
                                {
                                    //PDSCH CSIRS section 1 (PDSCH)
                                    if(prb_info.common.isStaticBfwEncoded)
                                    {
                                        prg_size = prb_info.static_bfwCoeff_buf_info.prg_size;
                                        encode_stat_ext_11 = true;
                                        encode_dyn_ext_11  = false;
                                    }
                                    else if(prb_info.bfwCoeff_buf_info.p_buf_bfwCoef_h != nullptr && !prb_info.common.isStaticBfwEncoded)
                                    {
                                        prg_size = prb_info.bfwCoeff_buf_info.prg_size;
                                        encode_stat_ext_11 = false;
                                        encode_dyn_ext_11  = true;
                                    }
                                }

                                //PDSCH CSIRS section 2 (CSIRS)
                                if(csi_rs_prb_info != nullptr && csi_rs_prb_info->common.extType == ORAN_CMSG_SECTION_EXT_TYPE_11 && csi_rs_prb_info->common.isStaticBfwEncoded)
                                {
                                    encode_stat_ext_11 = true;
                                }
                            }
                        }
                        else
                        {
                            encode_stat_ext_11 = false;
                            encode_dyn_ext_11 = false;
                            section_info.sect_1.beamId = beam_id;
                        }

                        section_info.sect_1.numPrbc = adjustPrbCountMuMimo(num_prbc, prg_size);
#if 0
                        if(channel_type != slot_command_api::channel_type::SRS)
                        {
                            NVLOGC_FMT(TAG, "##### ap_idx={}, prb_info={}, channel_type={}, flows.size={}, portMask = {}", ap_idx, static_cast<void *>(&prb_info), channel_type, flows.size(), static_cast<uint64_t>(prb_info.common.portMask));
                            NVLOGC_FMT(TAG, "F{}S{}S{} PRBINFO symbol {} Channel {} startPrbc {} numPrbc {} extType {} isStaticBfwEncoded {}", slot_indication.oframe_id_, slot_indication.osfid_, slot_indication.oslotid_, symbol_id, channel_type, (int)prb_info.common.startPrbc, (int)prb_info.common.numPrbc, (int)prb_info.common.extType, (int)prb_info.common.isStaticBfwEncoded);
                            NVLOGC_FMT(TAG, "encode_stat_ext_11={} encode_dyn_ext_11={} symbol_id={}, numSymbols={}", encode_stat_ext_11, encode_dyn_ext_11,  symbol_id, (int)prb_info.common.numSymbols); 
                            if (csi_rs_prb_info != nullptr)
                            {
                                NVLOGC_FMT(TAG, "##### csi_rs_prb_info extType={} isStaticBfwEncoded={} startPrbc={} numPrbc={}", (int)csi_rs_prb_info->common.extType, (int)csi_rs_prb_info->common.isStaticBfwEncoded, (int)csi_rs_prb_info->common.startPrbc, (int)csi_rs_prb_info->common.numPrbc);

                            }
                        }
#endif
                        /* If DL BFW is not complete do not encode the ext 11*/
                        if(!prevSlotBfwCompStatus)
                        {
                            // Avoid including ext11 as the DL BFW weights are not ready
                            encode_dyn_ext_11 = false;
                        }
                        if(encode_dyn_ext_11)
                        {
                            disableBFWs = dyn_bfw_seen[ap_idx][prb_info.common.startPrbc];
                            if(bfwIQBitwidth == 9)
                            {
                                message_template.hasSectionExt = true;
                                // cuPHY buffer starts from 0 regardless of startPrbc
                                int startPrbc = prb_info.common.isPdschSplitAcrossPrbInfo ? prb_info.common.startPrbc : 0;
                                int start_bundle_offset_in_bfw_buffer = prb_info.common.isPdschSplitAcrossPrbInfo ? prb_info.common.startPrbc / prb_info.bfwCoeff_buf_info.prg_size : 0;
                                int numPrbc   = prb_info.common.numPrbc;
                                if ((numPrbc > ORAN_MAX_PRB_X_SECTION) && (numPrbc < ORAN_MAX_PRB_X_SLOT))
                                {
                                    prg_size = prb_info.bfwCoeff_buf_info.prg_size;
                                    numPrbc = adjustPrbCountMuMimo(numPrbc, prg_size);
                                }
                                num_prbc_split = numPrbc;
                                //NVLOGD_FMT(TAG, "fh {}: common.active_eaxc_ids[{}]= {}",__LINE__, ap_idx, static_cast<int>(prb_info.common.active_eaxc_ids[ap_idx]));

                                DynamicSectionExt11Params params{
                                    .prb_info = prb_info,
                                    .section_info = section_info,
                                    .section_ext_infos = section_ext_infos,
                                    .section_ext_11_bundle_infos = section_ext_11_bundle_infos,
                                    .section_ext_index = section_ext_index,
                                    .section_ext_11_bundle_index = section_ext_11_bundle_index,
                                    .L_TRX = L_TRX,
                                    .bfwIQBitwidth = bfwIQBitwidth,
                                    .disableBFWs = disableBFWs,
                                    .RAD = RAD,
                                    .bfw_header = bfw_header,
                                    .active_ap_idx = prb_info.common.active_eaxc_ids[ap_idx],
                                    .actual_ap_idx = ap_idx,
                                    .symbol = symbol_id,
                                    .startPrbc = startPrbc,
                                    .numPrbc = numPrbc,
                                    .dyn_beam_id = cur_dyn_bfw_beam_id,
                                    .bfw_beam_id = dyn_bfw_beam_id[ap_idx],
                                    .bfw_seen = dyn_bfw_seen[ap_idx],
                                    .slot_dyn_beam_id_offset = (uint16_t)dyn_beam_id_offset,
                                    .start_bundle_offset_in_bfw_buffer = start_bundle_offset_in_bfw_buffer
                                };
                                fill_dynamic_section_ext11(slot_indication, params);
                            }
                            else
                            {
                                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "The configuration of bfwIQBitwidth = {} is not supported for dynamic BFW, omitting section extension!", bfwIQBitwidth);
                            }
                        }
                        else if(encode_stat_ext_11)
                        {
                            message_template.hasSectionExt = true;
                            int startPrbc                  = 0;
                            int numPrbc                    = prb_info.common.numPrbc;
                            if((numPrbc > ORAN_MAX_PRB_X_SECTION) && (numPrbc < ORAN_MAX_PRB_X_SLOT))
                            {
                                uint16_t prg_size = prb_info.static_bfwCoeff_buf_info.prg_size;
                                prg_size          = prb_info.bfwCoeff_buf_info.prg_size;
                                numPrbc           = adjustPrbCountMuMimo(numPrbc, prg_size);
                            }
                            num_prbc_split = numPrbc;

                            if(csirs_section_only)
                            {
                                uint8_t* static_bfw_ptr = getStaticBFWWeights(cell_id, csirs_beam_id);
                                fill_static_section_ext(static_bfw_ptr, cell_id, *csi_rs_prb_info, section_info, section_ext_infos, section_ext_11_bundle_infos, section_ext_index, section_ext_11_bundle_index, staticBFWIQBitwidth, disableBFWs, RAD, active_ap_idx, ap_idx, startPrbc, numPrbc, csirs_beam_id);
                            }
                            else
                            {
                                if (prb_info.common.isStaticBfwEncoded)
                                {
                                    uint8_t* static_bfw_ptr = getStaticBFWWeights(cell_id, beam_id);
                                    fill_static_section_ext(static_bfw_ptr, cell_id, prb_info, section_info, section_ext_infos, section_ext_11_bundle_infos, section_ext_index, section_ext_11_bundle_index, staticBFWIQBitwidth, disableBFWs, RAD, active_ap_idx, ap_idx, startPrbc, numPrbc, beam_id);
                                }
                                else
                                {
                                    section_info.sect_1.ef     = 0;
                                    section_info.sect_1.beamId = beam_id;
                                }
                            }
                        }

                        if(mod_comp_enabled && section_info.ext4 == nullptr)
                        {
                            auto& comp =
                                (is_pdsch_csirs ? pdsch_comp_info : prb_info.comp_info);
                            fill_section_ext4_ext5(comp, section_info, section_ext_infos, section_ext_index);

                            message_template.hasSectionExt = true;
                        }

                        // At this point, the decision to disable BFW is known, so populate the singleton info in the appropriate set
                        auto &section_count_per_ant = ((encode_dyn_ext_11 || encode_stat_ext_11) && disableBFWs == 0) ? section_count_per_ant_per_msgtype[fh_msg_info_type::BFW_MSG_INFO] : section_count_per_ant_per_msgtype[fh_msg_info_type::NON_BFW_MSG_INFO]; 
                        auto &section_infos_per_ant = ((encode_dyn_ext_11 || encode_stat_ext_11) && disableBFWs == 0) ? section_infos_per_ant_per_msgtype[fh_msg_info_type::BFW_MSG_INFO] : section_infos_per_ant_per_msgtype[fh_msg_info_type::NON_BFW_MSG_INFO];
                        
                        memcpy(&section_infos_per_ant[flow_idx].at(section_count_per_ant[flow_idx]), &section_info, sizeof(fhproxy_cmsg_section)); 
                        increment_section_count(section_count_per_ant[flow_idx], MAX_CPLANE_SECTIONS_PER_SLOT_PER_AP, flow_idx);

                        if(both_pdsch_csirs_included)
                        { //Followed by csirs
                            auto pdsch_section_index = section_count_per_ant[flow_idx] - 1;
                            uint8_t num_overlap_ports = overlap_csirs_port_info.num_overlap_ports ? overlap_csirs_port_info.num_overlap_ports : 1;
                            for(uint8_t i = 0; i < num_overlap_ports; i++)
                            {
                                auto csirs_section_index = section_count_per_ant[flow_idx];
                                memcpy(&section_infos_per_ant[flow_idx][csirs_section_index], &section_infos_per_ant[flow_idx][pdsch_section_index], sizeof(section_template));
                                section_infos_per_ant[flow_idx][csirs_section_index].sect_1.reMask = overlap_csirs_port_info.num_overlap_ports > 0 ? overlap_csirs_port_info.reMask_ap_idx_pairs[i].first : overlapping_csi_rs_remask;
                                section_infos_per_ant[flow_idx][csirs_section_index].csirs_of_multiplex_pdsch_csirs = true;
                                section_infos_per_ant[flow_idx][csirs_section_index].ext4 = nullptr;
                                section_infos_per_ant[flow_idx][csirs_section_index].ext5 = nullptr;
                                section_infos_per_ant[flow_idx][csirs_section_index].ext11 = nullptr;
                                if(overlap_csirs_port_info.num_overlap_ports > 0)
                                {
                                    csirs_beam_idx = overlap_csirs_port_info.reMask_ap_idx_pairs[i].second % beams_csirs_array_size;
                                }
                                csirs_beam_id          = (*beams_csirs_array)[csirs_beam_idx];
                                section_infos_per_ant[flow_idx][csirs_section_index].section_id_lookback_index = i + 1;
                                
                                if(encode_stat_ext_11)
                                {
                                    message_template.hasSectionExt = true;
                                    int startPrbc                  = 0;
                                    int numPrbc                    = prb_info.common.numPrbc;
                                    if((numPrbc > ORAN_MAX_PRB_X_SECTION) && (numPrbc < ORAN_MAX_PRB_X_SLOT))
                                    {
                                        prg_size = prb_info.bfwCoeff_buf_info.prg_size;
                                        numPrbc  = adjustPrbCountMuMimo(numPrbc, prg_size);
                                    }
                                    num_prbc_split = numPrbc;
                                    uint8_t* static_bfw_ptr = getStaticBFWWeights(cell_id, csirs_beam_id);
                                    fill_static_section_ext(static_bfw_ptr, cell_id, *csi_rs_prb_info, section_infos_per_ant[flow_idx][csirs_section_index], section_ext_infos, section_ext_11_bundle_infos, section_ext_index, section_ext_11_bundle_index, staticBFWIQBitwidth, disableBFWs, RAD, active_ap_idx, ap_idx, startPrbc, numPrbc, csirs_beam_id);
                                    //static_buff_count++;
                                }
                                else
                                {
                                    section_infos_per_ant[flow_idx][csirs_section_index].sect_1.ef     = 0;
                                    section_infos_per_ant[flow_idx][csirs_section_index].sect_1.beamId = csirs_beam_id;
                                }
                                increment_section_count(section_count_per_ant[flow_idx], MAX_CPLANE_SECTIONS_PER_SLOT_PER_AP, flow_idx);
                                if(mod_comp_enabled)
                                {
                                    fill_section_ext4_ext5(csi_rs_comp_info, section_infos_per_ant[flow_idx][csirs_section_index], section_ext_infos, section_ext_index);
                                }
                            }
                        }

                        if ((num_prbc > ORAN_MAX_PRB_X_SECTION) && (num_prbc < ORAN_MAX_PRB_X_SLOT))
                        {
                            uint8_t num_overlap_ports = both_pdsch_csirs_included? overlap_csirs_port_info.num_overlap_ports > 0 ? overlap_csirs_port_info.num_overlap_ports : 1 : 0;
                            auto prev_section_index = section_count_per_ant[flow_idx] - (num_overlap_ports + 1);
                            auto curr_section_index = section_count_per_ant[flow_idx];
                            // For CSI-RS: Increment lookback index to account for the section split
                            // This ensures subsequent sections reference the correct split section for section ID
                            if(channel_type == slot_command_api::channel_type::CSI_RS) 
                            {
                                csirs_section_id_info.section_id_lookback_index++;
                            }
                            memcpy(&section_infos_per_ant[flow_idx][curr_section_index], &section_infos_per_ant[flow_idx][prev_section_index], sizeof(section_template));
                            
                            //Incase alternate PRB is being used in a section. if the section started from
                            //even PRB new section should start from even PRB. Same goes for section starting
                            // from odd PRB. This is applicable in case we move beyond 100 MHZ bandwidth
                            section_infos_per_ant[flow_idx][section_count_per_ant[flow_idx]].sect_1.startPrbc = start_prbc + num_prbc_split;
                            section_infos_per_ant[flow_idx][section_count_per_ant[flow_idx]].sect_1.numPrbc   = num_prbc - num_prbc_split;
                            section_infos_per_ant[flow_idx][section_count_per_ant[flow_idx]].ext4             = nullptr;
                            section_infos_per_ant[flow_idx][section_count_per_ant[flow_idx]].ext5             = nullptr;
                            section_infos_per_ant[flow_idx][section_count_per_ant[flow_idx]].ext11            = nullptr;
                            // cuPHY buffer starts from 0 regardless of startPrbc, so adjust to num_prbc_split/255
                            int   startPrbc    = num_prbc_split;
                            int   numPrbc      = prb_info.common.numPrbc - num_prbc_split;
                            auto& section_info = section_infos_per_ant[flow_idx][curr_section_index];
                            if(encode_dyn_ext_11)
                            {
                                message_template.hasSectionExt = true;
                                //NVLOGD_FMT(TAG, "fh {}: common.active_eaxc_ids[{}]= {}",__LINE__, ap_idx, static_cast<int>(prb_info.common.active_eaxc_ids[ap_idx]));
                                DynamicSectionExt11Params params{
                                    .prb_info = prb_info,
                                    .section_info = section_info,
                                    .section_ext_infos = section_ext_infos,
                                    .section_ext_11_bundle_infos = section_ext_11_bundle_infos,
                                    .section_ext_index = section_ext_index,
                                    .section_ext_11_bundle_index = section_ext_11_bundle_index,
                                    .L_TRX = L_TRX,
                                    .bfwIQBitwidth = bfwIQBitwidth,
                                    .disableBFWs = disableBFWs,
                                    .RAD = RAD,
                                    .bfw_header = bfw_header,
                                    .active_ap_idx = prb_info.common.active_eaxc_ids[ap_idx],
                                    .actual_ap_idx = ap_idx,
                                    .symbol = symbol_id,
                                    .startPrbc = startPrbc,
                                    .numPrbc = numPrbc,
                                    .dyn_beam_id = cur_dyn_bfw_beam_id,
                                    .bfw_beam_id = dyn_bfw_beam_id[ap_idx],
                                    .bfw_seen = dyn_bfw_seen[ap_idx],
                                    .slot_dyn_beam_id_offset = (uint16_t)dyn_beam_id_offset,
                                    .start_bundle_offset_in_bfw_buffer = num_prbc_split / prb_info.bfwCoeff_buf_info.prg_size
                                };
                                fill_dynamic_section_ext11(slot_indication, params);
                            }
                            else if(encode_stat_ext_11)
                            {
                                message_template.hasSectionExt = true;
                                if(csirs_section_only)
                                {
                                    uint8_t* static_bfw_ptr = getStaticBFWWeights(cell_id, csirs_beam_id);
                                    fill_static_section_ext(static_bfw_ptr, cell_id, *csi_rs_prb_info, section_info, section_ext_infos, section_ext_11_bundle_infos, section_ext_index, section_ext_11_bundle_index, staticBFWIQBitwidth, disableBFWs, RAD, active_ap_idx, ap_idx, startPrbc, numPrbc, csirs_beam_id);
                                }
                                else
                                {
                                    uint8_t* static_bfw_ptr = getStaticBFWWeights(cell_id, beam_id);
                                    fill_static_section_ext(static_bfw_ptr, cell_id, prb_info, section_info, section_ext_infos, section_ext_11_bundle_infos, section_ext_index, section_ext_11_bundle_index, staticBFWIQBitwidth, disableBFWs, RAD, active_ap_idx, ap_idx, startPrbc, numPrbc, beam_id);
                                }
                            }
                            increment_section_count(section_count_per_ant[flow_idx], MAX_CPLANE_SECTIONS_PER_SLOT_PER_AP, flow_idx);

                            if(mod_comp_enabled)
                            {
                                auto& comp =
                                    (is_pdsch_csirs ? pdsch_comp_info : prb_info.comp_info);
                                fill_section_ext4_ext5(comp, section_info, section_ext_infos, section_ext_index);
                                message_template.hasSectionExt = true;
                            }

                            if(both_pdsch_csirs_included)
                            {//Followed by csirs
                                for(uint8_t i = 0; i < num_overlap_ports; i++){
                                    auto pdsch_section_index = section_count_per_ant[flow_idx] - (i + 1);
                                    auto csirs_section_index = section_count_per_ant[flow_idx];
                                    memcpy(&section_infos_per_ant[flow_idx][csirs_section_index], &section_infos_per_ant[flow_idx][pdsch_section_index], sizeof(section_template));
                                    section_infos_per_ant[flow_idx][csirs_section_index].sect_1.reMask                  = overlap_csirs_port_info.num_overlap_ports > 0 ? overlap_csirs_port_info.reMask_ap_idx_pairs[i].first : overlapping_csi_rs_remask;;
                                    section_infos_per_ant[flow_idx][csirs_section_index].csirs_of_multiplex_pdsch_csirs = true;
                                    section_infos_per_ant[flow_idx][csirs_section_index].ext4                           = nullptr;
                                    section_infos_per_ant[flow_idx][csirs_section_index].ext5                           = nullptr;
                                    section_infos_per_ant[flow_idx][csirs_section_index].ext11                          = nullptr;
                                    if(overlap_csirs_port_info.num_overlap_ports > 0){
                                        csirs_beam_idx = overlap_csirs_port_info.reMask_ap_idx_pairs[i].second % beams_csirs_array_size;
                                    }
                                    csirs_beam_id          = (*beams_csirs_array)[csirs_beam_idx];
                                    section_infos_per_ant[flow_idx][csirs_section_index].section_id_lookback_index = i + 1;

                                    if(encode_stat_ext_11)
                                    {
                                        message_template.hasSectionExt = true;
                                        int startPrbc                  = num_prbc_split;
                                        int numPrbc                    = prb_info.common.numPrbc - num_prbc_split;
                                        uint8_t* static_bfw_ptr = getStaticBFWWeights(cell_id, csirs_beam_id);
                                        fill_static_section_ext(static_bfw_ptr, cell_id, *csi_rs_prb_info, section_infos_per_ant[flow_idx][csirs_section_index], section_ext_infos, section_ext_11_bundle_infos, section_ext_index, section_ext_11_bundle_index, staticBFWIQBitwidth, disableBFWs, RAD, active_ap_idx, ap_idx, startPrbc, numPrbc, csirs_beam_id);
                                        //static_buff_count++;
                                    }
                                    else
                                    {
                                        section_infos_per_ant[flow_idx][csirs_section_index].sect_1.ef     = 0;
                                        section_infos_per_ant[flow_idx][csirs_section_index].sect_1.beamId = csirs_beam_id;
                                    }
                                    increment_section_count(section_count_per_ant[flow_idx], MAX_CPLANE_SECTIONS_PER_SLOT_PER_AP, flow_idx);
                                    if(mod_comp_enabled)
                                    {
                                        fill_section_ext4_ext5(csi_rs_comp_info, section_infos_per_ant[flow_idx][csirs_section_index], section_ext_infos, section_ext_index);
                                    }
                                }
                            }
                        }
                        active_ap_idx++;
                    } // for(int ap_idx = 0; ap_idx < flows.size(); ++ap_idx)
                } // for(auto prb_info_idx : *prb_index_list)
                for (int bfw_type = 0; bfw_type < fh_msg_info_type::NUM_MSG_INFO_TYPES; ++bfw_type) 
                {
                    for(int ap_idx = 0; ap_idx < flows.size(); ++ap_idx)
                    {
                        int start_section_idx = start_section_count_per_ant_per_msgtype[bfw_type][ap_idx]; 
                        radio_app_hdr.numberOfSections = section_count_per_ant_per_msgtype[bfw_type][ap_idx] - start_section_idx;
                        if(radio_app_hdr.numberOfSections == 0)
                        {
                            continue;
                        }
                        message_template.sections  = &section_infos_per_ant_per_msgtype[bfw_type][ap_idx][start_section_idx];
                        message_template.flow      = flows[ap_idx].handle;
                        message_template.ap_idx    = ap_idx;
                        if(channel_type == slot_command_api::channel_type::PRACH)
                        {
                            message_template.nxt_section_id = &start_section_id_prach_per_ant[0];
                        }
                        else if(channel_type == slot_command_api::channel_type::SRS)
                        {
                            message_template.nxt_section_id = &start_section_id_srs_per_ant[0];
                        }
                        else
                        {
                            message_template.nxt_section_id = &section_id_per_ant[0];
                        }
                        memcpy(&message_infos[bfw_type][message_index[bfw_type]++], &message_template, sizeof(message_template));
                    }
                }
            } // for (int channel_type = start_channel_type; channel_type <= end_channel_type; channel_type++)
            start_tx_time = end_tx_time;
            end_tx_time = start_tx_time + tx_time;
        } // for(int symbol_id = start_symbol; symbol_id < slot_info.symbols.size(); symbol_id++)

        /*
         * NOTE: For MMIMO DL - the task ordering is setup to process the BFW
         * packets of all the cells followed by non-BFW packets. This enables
         * meeting the stricter BFW packet deadlines. See
         * task_dl_function_dl_aggr.cpp::task_work_function_cplane() for the
         * ordering of sendCPlaneMMIMO calls.
         *
         * For MMIMO UL - the task ordering is setup to process BFW of a cell followed by its non-BFW packets.
         * See task_function_ul_aggr.cpp::task_work_function_ul_aggr_1_cplane() for the ordering of 
         * sendCPlaneMMIMO calls. 
         *
         * Note: The data structures for section_id (start_section_id_prach_per_ant/start_section_id_srs_per_ant/section_id_per_ant) 
         * need to be made atomic if BFW and non-BFW will execute concurrently. 
         *
         * */

    }
    else
    {
        fhproxy_cmsg         message_infos[MAX_CPLANE_MSGS_PER_SLOT];
        size_t               message_index{};
        // Disable Modcomp for now for 4T4R
// #ifndef ENABLE_MODCOMP
        for(int symbol_id = start_symbol; symbol_id < slot_info.symbols.size(); symbol_id++)
        {
            message_template.tx_window.tx_window_start = start_tx_time.count();
            message_template.tx_window.tx_window_end   = end_tx_time.count();
            radio_app_hdr.startSymbolId                = symbol_id;

            for (int channel_type = start_channel_type; channel_type <= end_channel_type; channel_type++)
            {
                auto &flows               = peer_ptr->cplane_flows[channel_type];

                start_section_index       = section_index;
                std::array<std::size_t, slot_command_api::MAX_BEAMS>  beam_list_idx_v;
                size_t beam_list_idx_v_size = 0;
                uint16_t beam_list_idx    = 0;
                bool is_pdsch_csirs = (channel_type == slot_command_api::channel_type::PDSCH_CSIRS) && (slot_info.symbols[symbol_id][channel_type].size() > 0);
                bool use_alt_csirs_list = false;
                const std::vector<std::size_t> * prb_index_list = nullptr;
                slot_command_api::prb_info_list_t prbs = nullptr;
                if((channel_type == slot_command_api::channel_type::CSI_RS) &&
                    (slot_info.symbols[symbol_id][slot_command_api::channel_type::PDSCH_CSIRS].size() != 0))
                {
                    if(slot_info.alt_csirs_prb_info_idx_list[symbol_id].size())
                    {
                        prb_index_list = &slot_info.alt_csirs_prb_info_idx_list[symbol_id];
                        prbs = const_cast<slot_command_api::prb_info_list_t>(slot_info.alt_csirs_prb_info_list);
                        use_alt_csirs_list = true;
                    }
                    else
                        continue;
                }
                else
                {
                    prb_index_list = &slot_info.symbols[symbol_id][channel_type];
                    prbs = const_cast<slot_command_api::prb_info_list_t>(slot_info.prbs);
                }

                for (auto prb_info_idx : *prb_index_list)
                {
                    //NVLOGC_FMT(TAG, "channel type = {}, Start section ID = {}", channel_type, start_section_index);
                    auto &prb_info              = prbs[prb_info_idx];
                    auto num_prbc               = prb_info.common.numPrbc;
                    auto start_prbc             = prb_info.common.startPrbc;

                    radio_app_hdr.dataDirection = prb_info.common.direction;
                    section_info.startPrbc      = start_prbc;
                    section_info.numPrbc        = adjustPrbCount(num_prbc);
                    section_info.reMask         = prb_info.common.reMask;
                    section_info.numSymbol      = prb_info.common.numSymbols;
                    section_info.rb             = prb_info.common.useAltPrb ? 1 : 0;
                    radio_app_hdr.filterIndex   = prb_info.common.filterIndex;
                    
                    if (channel_type == slot_command_api::channel_type::PRACH)
                    {
                        radio_app_hdr.sectionType = ORAN_CMSG_SECTION_TYPE_3;

                        auto &sect_3_common_hdr          = message_template.section_common_hdr.sect_3_common_hdr;
                        sect_3_common_hdr.timeOffset     = time_offset;
                        sect_3_common_hdr.frameStructure = frame_structure;
                        sect_3_common_hdr.cpLength       = cp_length;
                        sect_3_common_hdr.udCompHdr      = ud_comp_hdr;

                        auto &section_3_info      = section_template.sect_3;
                        section_3_info.sectionId  = section_id_prach++;
                        section_3_info.freqOffset = prb_info.common.freqOffset;
                        section_3_info.reserved   = 0;
                    }
                    else
                    {
                        radio_app_hdr.sectionType = ORAN_CMSG_SECTION_TYPE_1;
                        section_info.sectionId    = channel_type == slot_command_api::channel_type::SRS ? section_id_srs++ : section_id++;
                        //if(channel_type == slot_command_api::channel_type::CSI_RS)
                        //NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "C-Plane channel Prb_info_index = {} type = {}, Start section ID = {} start_prbu ={} num_prbu = {}", prb_info_idx ,channel_type, section_id, prb_info.common.startPrbc, prb_info.common.numPrbc);
                        auto &sect_1_common_hdr     = message_template.section_common_hdr.sect_1_common_hdr;
                        sect_1_common_hdr.udCompHdr = ud_comp_hdr;
                        sect_1_common_hdr.reserved  = 0;
                    }

#if 0
                    if(prb_info.common.extType == ORAN_CMSG_SECTION_EXT_TYPE_11)
                    {
                        if(disableBFWs == 0 && bfwIQBitwidth == 9)
                        {
                            message_template.hasSectionExt = true;
                            section_info.ef = 1;
                        }
                        else
                        {
                            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "The configuration of disableBFWs = {} and bfwIQBitwidth = {} is not supported, omitting section extension!", disableBFWs, bfwIQBitwidth);
                        }
                    } 
                    else
                    {
                        //message_template.hasSectionExt = false;
                        section_info.ef = 0;
                    }
#endif

                    memcpy(&section_infos[section_index++], &section_template, sizeof(section_template));
                    beam_list_idx_v[beam_list_idx_v_size++] = beam_list_idx;

                    if ((num_prbc > ORAN_MAX_PRB_X_SECTION) && (num_prbc < ORAN_MAX_PRB_X_SLOT))
                    {
                        memcpy(&section_infos[section_index], &section_template, sizeof(section_template));
                        section_infos[section_index].sect_1.sectionId = section_id++;
                        //Incase alternate PRB is being used in a section. if the section started from 
                        //even PRB new section should start from even PRB. Same goes for section starting
                        // from odd PRB. This is applicable in case we move beyond 100 MHZ bandwidth
                        //section_infos[section_index].sect_1.startPrbc = (section_infos[section_index].sect_1.rb == 1) ? (start_prbc + 256) : (start_prbc + 255);
                        section_infos[section_index].sect_1.startPrbc = start_prbc + 255;
                        section_infos[section_index].sect_1.numPrbc = num_prbc - 255;
                        section_index++;
                        beam_list_idx_v[beam_list_idx_v_size++] = beam_list_idx;
                    }

                    if (channel_type == slot_command_api::channel_type::PDSCH_CSIRS)
                    {
                        if ((num_prbc > ORAN_MAX_PRB_X_SECTION) && (num_prbc < ORAN_MAX_PRB_X_SLOT))
                        {
                            memcpy(&section_infos[section_index], &section_infos[section_index-2], sizeof(section_template));
                            section_infos[section_index].sect_1.reMask = ~section_infos[section_index-2].sect_1.reMask;
                            section_index++;
                            beam_list_idx_v[beam_list_idx_v_size++] = beam_list_idx;
                            memcpy(&section_infos[section_index], &section_infos[section_index-2], sizeof(section_template));
                            section_infos[section_index].sect_1.reMask = ~section_infos[section_index-2].sect_1.reMask;
                            section_index++;
                            beam_list_idx_v[beam_list_idx_v_size++] = beam_list_idx;
                        }
                        else
                        {
                            memcpy(&section_infos[section_index], &section_infos[section_index-1], sizeof(section_template));
                            section_infos[section_index].sect_1.reMask = ~section_infos[section_index-1].sect_1.reMask;
                            section_index++;
                            beam_list_idx_v[beam_list_idx_v_size++] = beam_list_idx;
                        }
                    }
                    beam_list_idx++;
                }

                auto number_of_sections = section_index - start_section_index;

                if (number_of_sections == 0)
                {
                    continue;
                }

                radio_app_hdr.numberOfSections = number_of_sections;

                size_t num_ap_indices=0;;
                std::array<std::size_t,64> ap_index_list;
                auto &prb_info = prbs[(*prb_index_list)[0]];
                bool read_ap_index_list = false;

                if(ru == SINGLE_SECT_MODE)
                {
                    num_ap_indices = flows.size();
                }
                else if((channel_type == slot_command_api::channel_type::PDSCH ||
                    channel_type == slot_command_api::channel_type::PDSCH_CSIRS))
                {
                    parsePortMask(prb_info.common.portMask,num_ap_indices,ap_index_list);
                    read_ap_index_list = true;
                }
                else
                {
                    num_ap_indices = (direction == DIRECTION_DOWNLINK)? prb_info.common.numApIndices :  flows.size();
                    // TODO change slot_command so that numApIndices  dosen't change within a symbol (for a given channel)
                    
                }

                if(num_ap_indices > flows.size())
                {
                    NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Invalid AP info");
                    return SEND_CPLANE_FUNC_ERROR;
                }

                
                auto j = start_section_index;
                for(auto ap_index = 0; ap_index < num_ap_indices; ap_index++)
                {
                    auto flow_index = (read_ap_index_list)? ap_index_list[ap_index] : ap_index;
                    int pdsch_csirs_beam_id_alternate = 0;
                    int csirs_beam_list_idx = 0;
                    for (auto i = start_section_index; i < section_index; i++)
                    {
                        int beam_list_idx = beam_list_idx_v[i-start_section_index];
                        int beam_id_ch = channel_type;
                        const slot_command_api::beamid_array_t *beams_csirs_array = nullptr;
                        size_t beams_csirs_array_size = 0;

                        if(is_pdsch_csirs)
                        {
                            // Find the section of CSI-RS overlapping PDSCH
                            while((csirs_beam_list_idx < slot_info.symbols[symbol_id][slot_command_api::channel_type::CSI_RS].size()) &&
                                ((prbs[slot_info.symbols[symbol_id][beam_id_ch][beam_list_idx]].common.startPrbc +
                                prbs[slot_info.symbols[symbol_id][beam_id_ch][beam_list_idx]].common.numPrbc) >
                                (prbs[slot_info.symbols[symbol_id][slot_command_api::channel_type::CSI_RS][csirs_beam_list_idx]].common.startPrbc +
                                prbs[slot_info.symbols[symbol_id][slot_command_api::channel_type::CSI_RS][csirs_beam_list_idx]].common.numPrbc)))
                            {
                                csirs_beam_list_idx++;
                            }
                            if(csirs_beam_list_idx < slot_info.symbols[symbol_id][slot_command_api::channel_type::CSI_RS].size())
                            {
                                beams_csirs_array = &(prbs[slot_info.symbols[symbol_id][slot_command_api::channel_type::CSI_RS][csirs_beam_list_idx]].beams_array);
                                beams_csirs_array_size = prbs[slot_info.symbols[symbol_id][slot_command_api::channel_type::CSI_RS][csirs_beam_list_idx]].beams_array_size;
                            }
                        }
                        
                        slot_command_api::prb_info_t *prb_info = nullptr;
                        use_alt_csirs_list? prb_info = &prbs[slot_info.alt_csirs_prb_info_idx_list[symbol_id][beam_list_idx]] : 
                                            prb_info = &prbs[slot_info.symbols[symbol_id][beam_id_ch][beam_list_idx]];
                        //auto &prb_info = prbs[(*prb_index_list)[beam_list_idx]];
                        auto &beams_array = (*prb_info).beams_array;
                        size_t beams_array_size = (*prb_info).beams_array_size;

                        uint16_t beam_id = 0;
                        if(beams_array_size > 0)
                        {
                            beams_array_size = std::min(beams_array_size, num_ap_indices);
                            auto beam_repeat_interval = num_ap_indices/beams_array_size;
                            beam_id = beams_array[ap_index/beam_repeat_interval];
                        }
                        uint16_t beam_id_csirs = 0;
                        uint16_t cur_beam_id = beam_id;
                        if(is_pdsch_csirs && (beams_csirs_array_size > 0))
                        {
                            beams_csirs_array_size = std::min(beams_csirs_array_size, num_ap_indices);
                            auto beam_repeat_interval = num_ap_indices/beams_csirs_array_size;
                            beam_id_csirs = (*beams_csirs_array)[ap_index/beam_repeat_interval];
                        }

                        if(is_pdsch_csirs && i > start_section_index && section_infos[i].sect_1.reMask != section_infos[i-1].sect_1.reMask)
                        {
                            pdsch_csirs_beam_id_alternate = (++pdsch_csirs_beam_id_alternate) % 2;
                            cur_beam_id = pdsch_csirs_beam_id_alternate == 1 ? beam_id_csirs : beam_id;
                        }

                        memcpy(&section_infos[j], &section_infos[i], sizeof(section_template));
                        if(prb_info->common.extType == ORAN_CMSG_SECTION_EXT_TYPE_11)
                        {
                            if (!pdsch_csirs_beam_id_alternate)
                            {
                                if(disableBFWs == 0 && bfwIQBitwidth == 9)
                                {
                                    auto& section_ext_info = section_ext_infos[section_ext_index++];
                                    int extLenBytes = sizeof(oran_cmsg_ext_hdr) + sizeof(oran_cmsg_sect_ext_type_11) + sizeof(oran_cmsg_sect_ext_type_11_disableBFWs_0_bfwCompHdr);
                                    section_ext_info.sect_ext_common_hdr.ef = 0;
                                    section_ext_info.ext_11.bundle_hdr_size = sizeof(oran_cmsg_sect_ext_type_11_disableBFWs_0_bfp_compressed_bundle_hdr);
                                    L_TRX = prb_info->bfwCoeff_buf_info.nGnbAnt;
                                    section_ext_info.ext_11.bfwIQ_size = (L_TRX * bfwIQBitwidth * 2) / 8;
                                    section_ext_info.ext_11.real_bfwIQ_size = section_ext_info.ext_11.bfwIQ_size; // PDSCH BFW: no zero-padding
                                    int bundleLenBytes = section_ext_info.ext_11.bundle_hdr_size + section_ext_info.ext_11.bfwIQ_size;
                                    section_ext_info.ext_11.numPrbBundles = (prb_info->common.numPrbc + prb_info->bfwCoeff_buf_info.prg_size - 1) / prb_info->bfwCoeff_buf_info.prg_size;
                                    auto& numPrbBundles = section_ext_info.ext_11.numPrbBundles;
                                    extLenBytes += numPrbBundles * bundleLenBytes;
                                    section_ext_info.ext_11.ext_hdr.extLen = ((extLenBytes + 3) / 4); //pad to 4 byte boundary

                                    section_ext_info.sect_ext_common_hdr.extType = ORAN_CMSG_SECTION_EXT_TYPE_11;
                                    section_ext_info.ext_11.ext_hdr.numBundPrb = prb_info->bfwCoeff_buf_info.prg_size;
                                    section_ext_info.ext_11.ext_hdr.disableBFWs = disableBFWs;
                                    section_ext_info.ext_11.ext_hdr.RAD = RAD;
                                    section_ext_info.ext_11.ext_hdr.reserved = 0;
                                    section_ext_info.ext_11.ext_comp_hdr.bfwCompMeth = static_cast<uint8_t>(UserDataCompressionMethod::BLOCK_FLOATING_POINT);
                                    section_ext_info.ext_11.ext_comp_hdr.bfwIqWidth = bfwIQBitwidth;
                                    section_infos[j].sect_1.beamId = 0x7FFF;
                                    int section_bundle_start_index = section_ext_11_bundle_index;
                                    //for(int bundle_index = 0; bundle_index < numPrbBundles; ++bundle_index)
                                    // Assumption 1. PDSCH will always start at a multiple of prgsize
                                    // Assumption 2. when PDSCH and CSIRS overlap, it starts at a multiple of prgsize
                                    int start_bundle_offset = prb_info->common.startPrbc / prb_info->bfwCoeff_buf_info.prg_size;
                                    auto buffer_ptr = reinterpret_cast<uint8_t*>((*prb_info).bfwCoeff_buf_info.p_buf_bfwCoef_h);
                                    for(int bundle_index = start_bundle_offset; bundle_index < start_bundle_offset + numPrbBundles; ++bundle_index)
                                    {
                                        auto& bundle_info = section_ext_11_bundle_infos[section_ext_11_bundle_index++];
                                        // ru_type::MULTI_SECT_MODE does not send the TX precoding and beamforming PDU, we can set the inner beamId to a dummy value.
                                        bundle_info.disableBFWs_0_compressed.beamId = 0;//(*prb_info).beams_array[bundle_index * (*prb_info).bfwCoeff_buf_info.dig_bf_interfaces + ap_index];
                                        // Note1: in the representation: nGnbAnt x nPrbGrpBfw x nLayers, nGnbAnt is the innermost i.e. fastest changing dimension
                                        // and nLayers is the outermost dimension i.e. slowest changing dimension
                                        // Note2: for compressed beamforming coefficients, the block floating point exponent is prefixed to every nGnbAnt coefficients
                                        // (see field bfwCompParam in Table 7.7.11.1-1 in O-RAN.WG4.CUS.0-v10.00)
                                        
                                        // nLayers dimension
                                        int buffer_index = ap_index * prb_info->bfwCoeff_buf_info.num_prgs * (section_ext_info.ext_11.bfwIQ_size + 1);  // +1 for exponent
                                        buffer_index += bundle_index * (section_ext_info.ext_11.bfwIQ_size + 1); // +1 for exponent
                                        bundle_info.disableBFWs_0_compressed.bfwCompParam.exponent = buffer_ptr[buffer_index++];
                                        bundle_info.bfwIQ = &buffer_ptr[buffer_index];
                                        *bfw_header = prb_info->bfwCoeff_buf_info.header;
                                    }
                                    section_ext_info.ext_11.bundles = &section_ext_11_bundle_infos[section_bundle_start_index];
                                    section_infos[j].ext11 = &section_ext_info;
                                    message_template.hasSectionExt = true;
                                    section_infos[j].sect_1.ef = 1;
                                }
                            }
                            else
                            {
                                section_infos[j].sect_1.ef = 0;
                            }
                        }
                        else
                        {
                            section_infos[j].sect_1.beamId = cur_beam_id;
                        }

                        if(mod_comp_enabled && section_infos[j].ext4 == nullptr && section_infos[j].ext5 == nullptr)
                        {
                            fill_section_ext4_ext5(prb_info->comp_info, section_infos[j], section_ext_infos, section_ext_index);
                            message_template.hasSectionExt = true;
                        }
                        ++j;
                         //NVLOGI_FMT(TAG, "C-Plane  channel type = {}, Start section number = {} ap_index = {} flow_index = {} frameId {}, subframeId {}, slotid {}", channel_type, j-1, ap_index, flow_index, slot_indication.oframe_id_, slot_indication.osfid_, slot_indication.oslotid_);
                    }

                    message_template.sections = &section_infos[j - number_of_sections];
                    message_template.flow = flows[flow_index].handle;
                    memcpy(&message_infos[message_index++], &message_template, sizeof(message_template));
                }

                section_index = j;
    #if 0
                if ((channel_type == slot_command_api::channel_type::PDSCH_CSIRS) && (slot_info.symbols[symbol_id][slot_command_api::channel_type::PDSCH_CSIRS].size() > 0))
                {
                    channel_type += 1;
                }
    #endif
            }

            start_tx_time = end_tx_time;
            end_tx_time = start_tx_time + tx_time;

            if(ru == SINGLE_SECT_MODE && (direction == DIRECTION_DOWNLINK))
            {
                break;
            }
        }

        if (section_index > MAX_CPLANE_SECTIONS_PER_SLOT)
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Too many C-plane sections to send. Please increase MAX_CPLANE_SECTIONS_PER_SLOT");
            return SEND_CPLANE_FUNC_ERROR;
        }

        if (direction == DIRECTION_UPLINK && ((section_id >= start_section_id_prach) || (section_id_prach >= start_section_id_srs) || (section_id_srs < start_section_id_srs)))
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "At least two sections have the same SectionId value");
            return SEND_CPLANE_FUNC_ERROR;
        }

        if(0 == send_cplane(peer_ptr->peer, const_cast<const fhproxy_cmsg*>(&message_infos[0]), message_index))
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "aerial_fh::send_cplane error");
            return SEND_CPLANE_FUNC_ERROR;
        }


        // Uncomment the following code when Modcomp is to be enabled for 4T4R
// #else
//         std::array<std::size_t, MAX_FLOWS> num_section_per_flow = {};
//         size_t max_section_per_flow = 0;
//         std::array<std::size_t,MAX_PRB_INFO_PRB_SYMBOL> num_ap_indices  = {};
//         std::array<std::array<std::size_t,64>,MAX_PRB_INFO_PRB_SYMBOL> ap_index_list = {};
//         std::array<std::array<uint8_t,MAX_PRB_INFO_PRB_SYMBOL>, MAX_FLOWS>overlaping_csirs_section_idx ={}; 
//         std::array<slot_command_api::mod_comp_info_t,MAX_COMP_INFO_PER_PRB> comp_info = {};

//         for(int symbol_id = start_symbol; symbol_id < slot_info.symbols.size(); ++symbol_id)
//         {
//             message_template.tx_window.tx_window_start = start_tx_time.count();
//             message_template.tx_window.tx_window_end   = end_tx_time.count();
//             radio_app_hdr.startSymbolId                = symbol_id;
//             auto csirs_prb_list_size = slot_info.symbols[symbol_id][slot_command_api::channel_type::CSI_RS].size();
//             for (int channel_type = start_channel_type; channel_type <= end_channel_type; ++channel_type)
//             {
//                 auto &flows               = peer_ptr->cplane_flows[channel_type];
//                 auto prb_loop_index = 0;
//                 size_t start_section_index = next_start_section_index;
//                 size_t section_index = start_section_index;
//                 bool read_ap_index_list     = false;
                
//                 bool is_pdsch_csirs = (channel_type == slot_command_api::channel_type::PDSCH_CSIRS);
//                 size_t b = is_pdsch_csirs? 1 : 0;
//                 const std::vector<std::size_t> * prb_index_list = nullptr;
//                 slot_command_api::prb_info_list_t prbs = nullptr;
//                 if((channel_type == slot_command_api::channel_type::CSI_RS) &&
//                     (slot_info.symbols[symbol_id][slot_command_api::channel_type::PDSCH_CSIRS].size() != 0))
//                 {
//                     if(slot_info.alt_csirs_prb_info_idx_list[symbol_id].size())
//                     {
//                         prb_index_list = &slot_info.alt_csirs_prb_info_idx_list[symbol_id];
//                         prbs = const_cast<slot_command_api::prb_info_list_t>(slot_info.alt_csirs_prb_info_list);
//                     }
//                     else
//                         continue;
//                 }
//                 else
//                 {
//                     prb_index_list = &slot_info.symbols[symbol_id][channel_type];
//                     prbs = const_cast<slot_command_api::prb_info_list_t>(slot_info.prbs);
//                 }

//                 if((channel_type == slot_command_api::channel_type::PDSCH ||
//                         channel_type == slot_command_api::channel_type::PDSCH_CSIRS)||
//                         channel_type == slot_command_api::channel_type::CSI_RS)
//                 {
//                     if(ru != SINGLE_SECT_MODE)
//                         read_ap_index_list = true;
//                 }
//                 for (auto prb_info_idx : *prb_index_list)
//                 {
//                     auto &prb_info              = prbs[prb_info_idx];
//                     uint16_t overlaping_csirs_section_start_idx = 0;
//                     uint16_t overlaping_csirs_section_end_idx = 0;
//                     auto num_prbc = prb_info.common.numPrbc;
//                     size_t a = (num_prbc > ORAN_MAX_PRB_X_SECTION) && (num_prbc < ORAN_MAX_PRB_X_SLOT)? 1 : 0;
//                     if(ru == SINGLE_SECT_MODE || (direction == DIRECTION_UPLINK))
//                     {
//                         num_ap_indices[prb_loop_index] = flows.size();
//                     }
//                     else if(read_ap_index_list)
//                     {
//                         parsePortMask(prb_info.common.portMask, num_ap_indices[prb_loop_index],ap_index_list[prb_loop_index]);
//                     }
//                     else
//                     {
//                         num_ap_indices[prb_loop_index] =  prb_info.common.numApIndices;
//                     }

//                     if( num_ap_indices[prb_loop_index] > flows.size())
//                     {
//                         NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Invalid AP info");
//                         return SEND_CPLANE_FUNC_ERROR;
//                     }
//                     // if only num_prbc > 255 increment by 2
//                     // if only channel_type is PDSCH_CSIRS increment by 2
//                     // if both num_prbc > 255 and channel_type is PDSCH_CSIRS increment by 4
//                     // if none of the above is true increment by 1.
//                     auto increment = 1+((a << b)| b); 

//                     if (read_ap_index_list) {
//                         for (auto ap_index = 0; ap_index < num_ap_indices[prb_loop_index]; ++ap_index) {
//                             auto flow_index = ap_index_list[prb_loop_index][ap_index];
//                             num_section_per_flow[flow_index] += increment;
//                         }
//                     } else {
//                         for (auto flow_index = 0; flow_index < num_ap_indices[prb_loop_index]; ++flow_index) {
//                             num_section_per_flow[flow_index] += increment;
//                         }
//                     }
//                     ++prb_loop_index;
//                 }
//                 max_section_per_flow = *(std::max_element(num_section_per_flow.begin(), num_section_per_flow.end()));

//                 prb_loop_index = 0;
//                 //Set a bit for each flow 
//                 uint64_t first_section_per_flow = 0XFFFF;
//                 for (auto prb_info_idx : *prb_index_list)
//                 {
//                     auto &prb_info              = prbs[prb_info_idx];
//                     auto num_prbc               = prb_info.common.numPrbc;
//                     auto start_prbc             = prb_info.common.startPrbc;
//                     auto flow_index              = (read_ap_index_list)? ap_index_list[prb_loop_index][0] : 0;
//                     slot_command_api::prb_info_t* csirs_prb_info_ptr    = nullptr;
//                     radio_app_hdr.dataDirection = prb_info.common.direction;
//                     section_info.startPrbc      = start_prbc;
//                     section_info.numPrbc        = adjustPrbCount(num_prbc);
//                     section_info.reMask         = prb_info.common.reMask;
//                     section_info.numSymbol      = prb_info.common.numSymbols;
//                     section_info.rb             = prb_info.common.useAltPrb ? 1 : 0;
//                     radio_app_hdr.filterIndex   = prb_info.common.filterIndex;
//                     auto prb_segment_start_section_index = section_index;
//                     auto i                      = prb_segment_start_section_index;
//                     auto j                      = prb_segment_start_section_index;
//                     uint16_t beam_id = 0;
//                     auto &beams_array = prb_info.beams_array;
//                     size_t beams_array_size = prb_info.beams_array_size;
//                     uint16_t beam_id_csirs = 0;
//                     auto beam_repeat_interval = 0;
//                     if(beams_array_size > 0)
//                     {
//                         beams_array_size = std::min(beams_array_size, num_ap_indices[prb_loop_index]);
//                         beam_repeat_interval = num_ap_indices[prb_loop_index]/beams_array_size;
//                     }
//                     if (channel_type == slot_command_api::channel_type::PRACH)
//                     {
//                         radio_app_hdr.sectionType = ORAN_CMSG_SECTION_TYPE_3;

//                         auto &sect_3_common_hdr          = message_template.section_common_hdr.sect_3_common_hdr;
//                         sect_3_common_hdr.timeOffset     = time_offset;
//                         sect_3_common_hdr.frameStructure = frame_structure;
//                         sect_3_common_hdr.cpLength       = cp_length;
//                         sect_3_common_hdr.udCompHdr      = ud_comp_hdr;

//                         auto &section_3_info      = section_template.sect_3;
//                         section_3_info.sectionId  = section_id_prach++;
//                         section_3_info.freqOffset = prb_info.common.freqOffset;
//                         section_3_info.reserved   = 0;
//                     }
//                     else
//                     {
//                         radio_app_hdr.sectionType = ORAN_CMSG_SECTION_TYPE_1;
//                         section_info.sectionId    = channel_type == slot_command_api::channel_type::SRS ? section_id_srs++ : section_id++;
//                         //if(channel_type == slot_command_api::channel_type::CSI_RS)
//                             //NVLOGC_FMT(TAG, "C-Plane channel Prb_info_index = {} type = {}, Start section ID = {} start_prbu ={} num_prbu = {}", prb_info_idx ,channel_type, section_id, static_cast<uint16_t>(prb_info.common.startPrbc), static_cast<uint16_t>(prb_info.common.numPrbc));
//                         auto &sect_1_common_hdr     = message_template.section_common_hdr.sect_1_common_hdr;
//                         sect_1_common_hdr.udCompHdr = ud_comp_hdr;
//                         sect_1_common_hdr.reserved  = 0;
//                     }

//                     memcpy(&section_infos[section_index++], &section_template, sizeof(section_template));
//                     if(mod_comp_enabled)
//                     {
//                         comp_info[0] = prb_info.comp_info;
//                     }
//                     bool is_large_prb = (num_prbc > ORAN_MAX_PRB_X_SECTION) && (num_prbc < ORAN_MAX_PRB_X_SLOT);
//                     if (is_large_prb)
//                     {
//                         memcpy(&section_infos[section_index], &section_template, sizeof(section_template));
//                         section_infos[section_index].sect_1.sectionId = section_id++;
//                         //Incase alternate PRB is being used in a section. if the section started from 
//                         //even PRB new section should start from even PRB. Same goes for section starting
//                         // from odd PRB. This is applicable in case we move beyond 100 MHZ bandwidth
//                         //section_infos[section_index].sect_1.startPrbc = (section_infos[section_index].sect_1.rb == 1) ? (start_prbc + 256) : (start_prbc + 255);
//                         section_infos[section_index].sect_1.startPrbc = start_prbc + 255;
//                         section_infos[section_index++].sect_1.numPrbc = num_prbc - 255;
//                         if(mod_comp_enabled)
//                         {
//                             comp_info[1] = prb_info.comp_info;
//                         }
//                     }

//                     if (is_pdsch_csirs)
//                     {
//                         uint16_t csirs_remask = ~prb_info.common.reMask;
//                         if(mod_comp_enabled)
//                         {
//                             prb_info.common.reMask = prb_info.comp_info.sections[0].mcScaleReMask.get();
//                             csirs_remask           = prb_info.comp_info.sections[1].mcScaleReMask.get();
//                         }
//                         if (is_large_prb)
//                         {
//                             memcpy(&section_infos[section_index], &section_infos[section_index-2], 2 * sizeof(section_template));
//                             section_infos[section_index++].sect_1.reMask = csirs_remask;
//                             section_infos[section_index++].sect_1.reMask = csirs_remask;
//                             if(mod_comp_enabled)
//                             {
//                                 comp_info[2]                  = prb_info.comp_info;
//                                 comp_info[3]                  = prb_info.comp_info;
//                                 comp_info[0].common.nSections = 1;
//                                 comp_info[1].common.nSections = 1;
//                                 comp_info[2].common.nSections = 1;
//                                 comp_info[3].common.nSections = 1;
//                                 comp_info[2].sections[0]      = comp_info[0].sections[1];
//                                 comp_info[3].sections[0]      = comp_info[1].sections[1];
//                             }
//                         }
//                         else
//                         {
//                             memcpy(&section_infos[section_index], &section_infos[section_index-1], sizeof(section_template));
//                             section_infos[section_index++].sect_1.reMask = csirs_remask;
//                             if(mod_comp_enabled)
//                             {
//                                 comp_info[1]             = prb_info.comp_info;
//                                 comp_info[1].sections[0] = comp_info[0].sections[1];
//                             }
//                         }
//                     }

//                     auto number_of_sections = section_index - prb_segment_start_section_index;

//                     if (number_of_sections == 0)
//                     {
//                         continue;
//                     }
//                     if(prb_loop_index == 0)
//                     {
//                         radio_app_hdr.numberOfSections = max_section_per_flow;
//                     }
//                     size_t beams_csirs_array_size = 0;
//                     slot_command_api::beamid_array_t * beams_csirs_array = nullptr;
//                     uint16_t csirs_beam_repeat_interval = 0;
//                     if(is_pdsch_csirs && csirs_prb_list_size > 0)
//                     {
//                          beams_csirs_array_size = prb_info.beams_array_size2;
//                         beams_csirs_array = &prb_info.beams_array2;
//                         if(beams_csirs_array_size > 0)
//                         {
//                             beams_csirs_array_size = std::min(beams_csirs_array_size, num_ap_indices[prb_loop_index]);
//                             csirs_beam_repeat_interval = num_ap_indices[prb_loop_index]/beams_csirs_array_size;
//                         }
//                     }
//                     for(auto ap_index = 0; ap_index < num_ap_indices[prb_loop_index]; ++ap_index)
//                     {
//                         flow_index = (read_ap_index_list)? ap_index_list[prb_loop_index][ap_index] : ap_index;
//                         uint8_t pdsch_csirs_beam_id_alternate = 0;
//                         uint16_t csirs_remask = 0;
//                         if(beams_array_size > 0)
//                         {
//                             beam_id = beams_array[ap_index/beam_repeat_interval];
//                         }
//                         beam_id =  beams_csirs_array_size > 0 ? beams_array[ap_index/csirs_beam_repeat_interval] : 0;
//                         beam_id_csirs =  beams_csirs_array_size > 0 ? (*beams_csirs_array)[ap_index/csirs_beam_repeat_interval] : 0;

//                         if(ap_index > 0)
//                             memcpy(&section_infos[j], &section_infos[i], number_of_sections*sizeof(section_template));
//                         for (auto k = 0; k < number_of_sections; ++k)
//                         {
//                             // A single prb_info section can create upto 4 section
//                             // depending upon channel type and when number of PRBs
//                             // in a section are more than 255 or not.
//                             // i marks the start index of the set of upto 4 sections
//                             // replicating same prb_info.  Also these set of sections
//                             // will again get replicated if the prb_info is mapped 
//                             // on more than 1 antenna ports. j marks start index 
//                             // of the copy of set of upto 4 sections created for current
//                             // antenna port.
//                             // k marks the current index in the set of upto four sections 
//                             auto true_section_idx = i + k;
//                             auto copy_section_idx = j + k;

//                             if(is_pdsch_csirs)
//                             {
//                                 if( k > 0 && section_infos[true_section_idx].sect_1.reMask != section_infos[true_section_idx-1].sect_1.reMask)
//                                 {
//                                     //Toggle pdsch_csirs_beam_id_alternate
//                                     pdsch_csirs_beam_id_alternate ^=  1;
//                                 }
//                                 //If section getting copied is intended for CSI-RS. 
//                                 if(pdsch_csirs_beam_id_alternate == 1)
//                                 {
//                                     beam_id = beam_id_csirs;
//                                     if(csirs_prb_info_ptr != nullptr)
//                                     {
//                                         section_infos[copy_section_idx].sect_1.reMask = csirs_prb_info_ptr->common.reMask;
//                                     }
//                                 }
//                             }

//                             section_infos[copy_section_idx].sect_1.beamId = beam_id;


//                             if(mod_comp_enabled && section_infos[copy_section_idx].ext4 == nullptr && section_infos[copy_section_idx].ext5 == nullptr)
//                             {
//                                 fill_section_ext4_ext5(comp_info[k], section_infos[copy_section_idx], section_ext_infos, section_ext_index);
//                                 message_template.hasSectionExt = true;
//                             }

//                             //NVLOGC_FMT(TAG, "C-Plane  channel type = {}, Start section number = {} ap_index = {} flow_index = {} frameId {}, subframeId {}, slotid {}", channel_type, j, ap_index, flow_index, slot_indication.oframe_id_, slot_indication.osfid_, slot_indication.oslotid_);

//                         }
//                         //Either this should be the first prb segment for this flow
//                         if(first_section_per_flow & (1 << flow_index))
//                         {
//                             message_template.sections = &section_infos[j];
//                             message_template.flow = flows[flow_index].handle;
//                             memcpy(&message_infos[message_index++], &message_template, sizeof(message_template));
//                             first_section_per_flow &= ~(1 << flow_index);
//                             num_section_per_flow[flow_index] = 0;
//                         }
//                         j += max_section_per_flow;
//                     }

//                     if (prb_loop_index == 0)
//                     {
//                         next_start_section_index = j;
//                     }
                    
//                     ++prb_loop_index;
//                 }
//             }
            
//             start_tx_time = end_tx_time;
//             end_tx_time = start_tx_time + tx_time;

//             if(ru == SINGLE_SECT_MODE)
//             {
//                 break;
//             }
//         } // for(int symbol_id = start_symbol; symbol_id < slot_info.symbols.size(); ++symbol_id)

//         if (next_start_section_index > MAX_CPLANE_SECTIONS_PER_SLOT)
//         {
//             NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Too many C-plane sections to send. Please increase MAX_CPLANE_SECTIONS_PER_SLOT");
//             return SEND_CPLANE_FUNC_ERROR;
//         }

//         if (direction == DIRECTION_UPLINK && ((section_id >= start_section_id_prach) || (section_id_prach >= start_section_id_srs) || (section_id_srs < start_section_id_srs)))
//         {
//             NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "At least two sections have the same SectionId value");
//             return SEND_CPLANE_FUNC_ERROR;
//         }

//         if(0 == send_cplane(peer_ptr->peer, const_cast<const fhproxy_cmsg*>(&message_infos[0]), message_index))
//         {
//             NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "aerial_fh::send_cplane error");
//             return SEND_CPLANE_FUNC_ERROR;
//         }
// #endif
    }

    return SEND_CPLANE_NO_ERROR;
}

void release_dlbuffer_cb(void* addr, void* opaque)
{
    DLOutputBuffer* dlbuffer = static_cast<DLOutputBuffer*>(opaque);
    PUSH_RANGE_PHYDRV("DLBUF RELEASE", 4);
    dlbuffer->release();
    POP_RANGE_PHYDRV
}

void FhProxy::UpdateTxMetricsGpuComm(
    peer_id_t              peer_id,
    struct umsg_fh_tx_msg& umsg_tx_list)
{
    struct fh_peer_t* peer_ptr = getPeerFromId(peer_id);
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();

    if(pdctx->gpuCommDlEnabled())
    {
        gpu_comm_update_tx_metrics(peer_ptr->peer, umsg_tx_list.txrq_gpu);
    }
}

int FhProxy::prepareUPlanePackets(
    ru_type ru,
    peer_id_t peer_id, cudaStream_t dl_stream, t_ns start_tx_time,
    const slot_command_api::oran_slot_ind &slot_indication,
    const slot_command_api::slot_info_t &slot_info,
    struct umsg_fh_tx_msg& umsg_tx_list,
    size_t size, mod_compression_params* mod_comp_prm, mod_compression_params* mod_comp_config_temp, void * cb_obj, t_ns symbol_duration,
    cuphyBatchedMemcpyHelper& batchedMemcpyHelper)
{

    start_tx_time += t_ns(AppConfig::getInstance().getTaiOffset());
    PhyDriverCtx*           pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    uint8_t*                payload_addr = nullptr;
    size_t                  message_index = 0;
    t_ns                    end_tx_time = start_tx_time + symbol_duration;
    uint16_t                section_id = 0;
    t_ns cell_start_time = start_tx_time;
    int                  section_id_per_ant[MAX_AP_PER_SLOT];

    struct fh_peer_t *peer_ptr = getPeerFromId(peer_id);
    if(!peer_ptr)
    {
        return -1;
    }
    auto& dl_comp_method = peer_ptr->dl_comp_meth;
    fhproxy_umsg_tx message_template={};

    auto& radio_app_hdr          = message_template.radio_app_hdr;

    if(pdctx->getUeMode())
    {
        radio_app_hdr.dataDirection  = DIRECTION_UPLINK;
    }
    else
    {
        radio_app_hdr.dataDirection  = DIRECTION_DOWNLINK;
    }

    radio_app_hdr.payloadVersion = ORAN_DEF_PAYLOAD_VERSION;
    radio_app_hdr.filterIndex    = 0; //Should this change?
    radio_app_hdr.frameId        = slot_indication.oframe_id_;
    radio_app_hdr.subframeId     = slot_indication.osfid_;
    radio_app_hdr.slotId         = slot_indication.oslotid_;


    if(pdctx->getmMIMO_enable())
    {
        auto& section_info        = message_template.section_info;
        section_info.rb           = false;
        section_info.sym_inc      = false;
        bool cplane_section_info_exist = true;

        //Assuming the compression method will be the same for a given cell 
        if(dl_comp_method == aerial_fh::UserDataCompressionMethod::MODULATION_COMPRESSION)
            { 
                section_info.mod_comp_enable = true;
                std::memset(mod_comp_config_temp->num_messages_per_list, 0, sizeof(mod_comp_config_temp->num_messages_per_list));
            }
        else
        {
            section_info.mod_comp_enable = false;
        }
        for(int symbol_id = 0; symbol_id < slot_info.symbols.size(); symbol_id++)
        {
            if(cplane_section_info_exist == false)
            {
                break;
            }
            if(ru == SINGLE_SECT_MODE)
            {
                section_id = 0;
            }

            /* The window start information is needed in the gpu_comm_prepare_uplane*() code.
            This information is per symbol per slot so we could provide to aerial-fh this way rather
            than looping all the messages there.
            */
            message_template.tx_window.tx_window_start = start_tx_time.count();
            message_template.tx_window.tx_window_end   = end_tx_time.count();
            radio_app_hdr.symbolId                     = symbol_id;

            //NVLOGC_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "symbol_id {}, start_tx_time.count() {} and symbol duration {}", symbol_id, start_tx_time.count(), symbol_duration.count());


            for (int channel_type = slot_command_api::channel_type::PDSCH_CSIRS; channel_type <= slot_command_api::channel_type::PDCCH_DMRS; channel_type++)
            {
                if(cplane_section_info_exist == false)
                {
                    break;
                }
                //Different channels can have the same flow for the same cell
                auto &flows = peer_ptr->uplane_flows[channel_type];

                const std::vector<std::size_t> * prb_index_list = nullptr;
                slot_command_api::prb_info_list_t prbs = nullptr;
                if((channel_type == slot_command_api::channel_type::CSI_RS) &&
                    (slot_info.symbols[symbol_id][slot_command_api::channel_type::PDSCH_CSIRS].size() != 0))
                {
                    if(slot_info.alt_csirs_prb_info_idx_list[symbol_id].size())
                    {
                        prb_index_list = &slot_info.alt_csirs_prb_info_idx_list[symbol_id];
                        prbs = const_cast<slot_command_api::prb_info_list_t>(slot_info.alt_csirs_prb_info_list);
                    }
                    else
                        continue;
                }
                else
                {
                    prb_index_list = &slot_info.symbols[symbol_id][channel_type];
                    prbs = const_cast<slot_command_api::prb_info_list_t>(slot_info.prbs);
                }
                bool parse_portmask = false;
                if((channel_type == slot_command_api::channel_type::PDSCH || channel_type == slot_command_api::channel_type::PDSCH_CSIRS) && (ru != SINGLE_SECT_MODE))
                    parse_portmask = true;

                bool is_csirs_remask_set = false;
                for (auto prb_info_idx : *prb_index_list)
                {
                    //NVLOGC_FMT(TAG, "channel type = {}, Start section ID = {}", channel_type, start_section_index);
                    auto& prb_info          = prbs[prb_info_idx];
                    auto  num_prbu          = prb_info.common.numPrbc;
                    auto  start_prbu        = prb_info.common.startPrbc;
                    section_info.rb         = prb_info.common.useAltPrb ? 1 : 0;
                    section_info.num_prbu   = adjustPrbuCount(num_prbu);
                    section_info.start_prbu = start_prbu;
                    if(section_info.mod_comp_enable)
                    {
                        section_info.prb_size          = static_cast<uint32_t>(prb_info.comp_info.common.udIqWidth) * 3;
                    }
                    //if(channel_type == slot_command_api::channel_type::CSI_RS)
                    //NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "U-Plane Prb_info_index = {} channel type = {}, Start section ID = {} start_prbu ={} num_prbu = {}",prb_info_idx , channel_type, section_info.section_id, section_info.start_prbu, section_info.num_prbu);
                    radio_app_hdr.filterIndex = prb_info.common.filterIndex;
#if 0
                    auto start_tx_time = Time::nowNs();
                    while(slot_info.section_id_ready.load() == false)
                    {
                        auto diff = Time::getDifferenceNowToNs(start_tx_time).count();
                        if(diff > 500000) NVLOGC_FMT(TAG, "section_id_ready diff {} \n", diff);
                    }
#else
                    while(slot_info.section_id_ready.load() == false);
#endif

                    if(prb_info.cplane_sections_info_sym_map == 0)
                    {
                        cplane_section_info_exist = false;
                        NVLOGC_FMT(TAG, "prepareUPlanePackets: prb C-Plane sections info not initialized, likely due to sendCPlane Timing error");
                        // NVLOGC_FMT(TAG, "{} ##### prb_info={}, channel_type={}, flows.size={}, portMask = {}",__func__, static_cast<void*>(&prb_info), channel_type, flows.size(), static_cast<uint64_t>(prb_info.common.portMask));
                        // NVLOGC_FMT(TAG, "{} F{}S{}S{} PRBINFO symbol {} Channel {} startPrbc {} numPrbc {} extType {}",__func__, slot_indication.oframe_id_, slot_indication.osfid_, slot_indication.oslotid_, symbol_id, channel_type, (int)prb_info.common.startPrbc, (int)prb_info.common.numPrbc, (int)prb_info.common.extType);
                        break;
                    }

                    if((prb_info.cplane_sections_info_sym_map & (1 << symbol_id)) == 0)
                    {
                        bool cplane_sections_found = false;
                        if(prb_info.common.numSymbols > 0)
                        {
                            int tmp_sym = 0;
                            for(; tmp_sym < slot_info.symbols.size(); tmp_sym++)
                            {
                                if((prb_info.cplane_sections_info_sym_map & (1 << tmp_sym)) != 0) break;
                            }
                            if(tmp_sym < symbol_id)
                            {
                                prb_info.cplane_sections_info[symbol_id] = prb_info.cplane_sections_info[tmp_sym];
                                cplane_sections_found = true;
                            }
                        }

                        if(!cplane_sections_found)
                        {
#if 0
                            NVLOGC_FMT(TAG, "File {} line {} ##### prb_info={}, channel_type={}, flows.size={}, portMask = {}", __FILE__, __LINE__, static_cast<void *>(&prb_info), channel_type, flows.size(), static_cast<uint64_t>(prb_info.common.portMask));
                            NVLOGC_FMT(TAG, "File {} line {} F{}S{}S{} PRBINFO symbol {} Channel {} startPrbc {} numPrbc {} extType {}", __FILE__, __LINE__, slot_indication.oframe_id_, slot_indication.osfid_, slot_indication.oslotid_, symbol_id, channel_type, (int)prb_info.common.startPrbc, (int)prb_info.common.numPrbc, (int)prb_info.common.extType);
#endif
                            continue;
                        }
                    }
                    auto num_ports = flows.size();
                    if(channel_type == slot_command_api::channel_type::CSI_RS)
                    {
                        num_ports = 64 - __builtin_clzll(prb_info.common.portMask);
                    }
                    
                    for(int ap_idx = 0; ap_idx < num_ports; ++ap_idx)
                    {
                        if(!(prb_info.common.portMask & (1 << ap_idx)))
                        {
                            continue;
                        }
                        auto flow_idx = ap_idx % flows.size();
                        message_template.flow   = flows[flow_idx].handle;
                        message_template.eaxcid = flows[flow_idx].eAxC_id;
                        bool split_prb = false;
                        if((channel_type == slot_command_api::channel_type::CSI_RS) && (prb_info.cplane_sections_info[symbol_id]->combined_reMask[flow_idx] == 0))
                        {
                            continue;
                        }
                        for(auto cplane_section_idx = 0; cplane_section_idx < prb_info.cplane_sections_info[symbol_id]->cplane_sections_count[flow_idx]; cplane_section_idx++)
                        {
                            auto num_prbu   = prb_info.cplane_sections_info[symbol_id]->numPrbc[flow_idx][cplane_section_idx];
                            auto start_prbu = prb_info.cplane_sections_info[symbol_id]->startPrbc[flow_idx][cplane_section_idx];
                            auto section_id = prb_info.cplane_sections_info[symbol_id]->section_id[flow_idx][cplane_section_idx];
                            //if(prb_info.cplane_sections_info[symbol_id]->cplane_sections_count[flow_idx] > 1) NVLOGC_FMT(TAG, "section_info.start_prbu {} section_info.num_prbu {} section_info.section_id {}", start_prbu, num_prbu, section_id);
                            if(num_prbu == 0) num_prbu = ORAN_MAX_PRB_X_SLOT;
                            section_info.num_prbu   = adjustPrbuCount(num_prbu);
                            section_info.start_prbu = start_prbu;
                            section_info.section_id = prb_info.cplane_sections_info[symbol_id]->section_id[flow_idx][cplane_section_idx];
                            if(num_prbu >= ORAN_MAX_PRB_X_SECTION)
                            {
                                split_prb = true;
                            }
                            if(section_info.mod_comp_enable)
                            {
                                //populate the prb size for GPU init comm function
                                section_info.prb_size   = static_cast<uint32_t>(prb_info.comp_info.common.udIqWidth) * 3;
                                
                                uint32_t mod_comp_msg_index = mod_comp_config_temp->num_messages_per_list[flow_idx][symbol_id];

#ifdef ENABLE_32DL
				// Bounds check to prevent buffer overflow in mod_compression_params arrays
                                if (mod_comp_msg_index >= MAX_SECTIONS_PER_UPLANE_SYMBOL) {
                                    NVLOGE_FMT(TAG, AERIAL_ORAN_FH_EVENT, 
                                               "ModComp overflow: flow_idx={} symbol_id={} msg_index={} exceeds MAX_SECTIONS_PER_UPLANE_SYMBOL={}. Skipping section.",
                                               flow_idx, symbol_id, mod_comp_msg_index, MAX_SECTIONS_PER_UPLANE_SYMBOL);
                                    continue; // Skip this section to prevent overflow
                                }
#endif

                                //start prb and num of prbs for a given prb allocation (common if there are two channels on the same prb)
                                mod_comp_config_temp->nprbs_per_list[flow_idx][symbol_id][mod_comp_msg_index]        = section_info.num_prbu;
                                mod_comp_config_temp->prb_start_per_list[flow_idx][symbol_id][mod_comp_msg_index]    = section_info.start_prbu;

                                //assuming there could be max of 2 channels multiplexed on the same prb allocation
                                float2 scale_val = {0, 0};
                                scale_val.x = prb_info.comp_info.modCompScalingValue[0];
                                

                                auto    udIqwidth = prb_info.comp_info.common.udIqWidth;
                                auto    csf_1     = prb_info.comp_info.sections[0].csf;
                                uint8_t csf_2     = 0;
                                uint16_t reMask_1 = 0;
                                uint16_t reMask_2 = 0;
                                if(channel_type == slot_command_api::channel_type::CSI_RS)
                                {
                                    reMask_1 = prb_info.cplane_sections_info[symbol_id]->combined_reMask[flow_idx];
                                }
                                else
                                {
                                    reMask_1 = prb_info.comp_info.sections[0].mcScaleReMask;
                                }

                                if(prb_info.comp_info.common.nSections == 2)
                                {
                                    scale_val.y = prb_info.comp_info.modCompScalingValue[1];
                                    csf_2       = prb_info.comp_info.sections[1].csf;
                                    reMask_2    = prb_info.comp_info.sections[1].mcScaleReMask;
                                }

                                mod_comp_config_temp->scaling[flow_idx][symbol_id][mod_comp_msg_index] = scale_val;
                                mod_comp_config_temp->params_per_list[flow_idx][symbol_id][mod_comp_msg_index].set(static_cast<QamListParam::qamwidth>(udIqwidth), csf_1, csf_2);
                                mod_comp_config_temp->prb_params_per_list[flow_idx][symbol_id][mod_comp_msg_index].set(reMask_1, reMask_2);

                                mod_comp_config_temp->num_messages_per_list[flow_idx][symbol_id]++;
#if 0
                                NVLOGC_FMT(TAG, "flw_idx: {}, sym_id: {}, msg_index: {}, udIqwith={}, mcScale_1={}, mcScale_2={}, start_prbu={}, num_prb={}, csf_1={}, csf_2={}, re_1={}, re_2={}", 
                                    flow_idx, symbol_id, mod_comp_msg_index, udIqwidth, scale_val.x, scale_val.y, section_info.start_prbu, section_info.num_prbu, csf_1, csf_2, reMask_1, reMask_2);
#endif
                            }
                            memcpy(&umsg_tx_list.umsg_info_symbol_antenna[message_index++], &message_template, sizeof(message_template));

                            if((num_prbu > ORAN_MAX_PRB_X_SECTION) && (num_prbu < ORAN_MAX_PRB_X_SLOT))
                            {
                                section_info.start_prbu = start_prbu + 255;
                                section_info.num_prbu   = num_prbu - 255;
                                section_info.section_id = section_id;
                                if(section_info.mod_comp_enable) // MODCOMP TODO Update in cuphydriver
                                {
                                    //populate the prb size for GPU init comm function
                                    section_info.prb_size = static_cast<uint32_t>(prb_info.comp_info.common.udIqWidth) * 3;

                                    uint32_t mod_comp_msg_index = mod_comp_config_temp->num_messages_per_list[flow_idx][symbol_id];

                                    //start prb and num of prbs for a given prb allocation (common if there are two channels on the same prb)
                                    mod_comp_config_temp->nprbs_per_list[flow_idx][symbol_id][mod_comp_msg_index]     = section_info.num_prbu;
                                    mod_comp_config_temp->prb_start_per_list[flow_idx][symbol_id][mod_comp_msg_index] = section_info.start_prbu;

                                    //assuming there could be max of 2 channels multiplexed on the same prb allocation
                                    float2 scale_val = {0, 0};
                                    scale_val.x      = prb_info.comp_info.modCompScalingValue[0];

                                    uint8_t udIqwidth = prb_info.comp_info.common.udIqWidth;
                                    uint8_t csf_1     = prb_info.comp_info.sections[0].csf;
                                    uint8_t csf_2     = 0;
                                    uint16_t reMask_2 = 0;
                                    uint16_t reMask_1 = 0;
                                    if(channel_type == slot_command_api::channel_type::CSI_RS)
                                    {
                                        reMask_1 = prb_info.cplane_sections_info[symbol_id]->combined_reMask[flow_idx];
                                    }
                                    else
                                    {
                                        reMask_1 = prb_info.comp_info.sections[0].mcScaleReMask;
                                    }

                                    if(prb_info.comp_info.common.nSections == 2)
                                    {
                                        scale_val.y = prb_info.comp_info.modCompScalingValue[1];
                                        csf_2       = prb_info.comp_info.sections[1].csf;
                                        reMask_2    = prb_info.comp_info.sections[1].mcScaleReMask;
                                    }
                                    mod_comp_config_temp->scaling[flow_idx][symbol_id][mod_comp_msg_index] = scale_val;
                                    mod_comp_config_temp->params_per_list[flow_idx][symbol_id][mod_comp_msg_index].set(static_cast<QamListParam::qamwidth>(udIqwidth), csf_1, csf_2);
                                    mod_comp_config_temp->prb_params_per_list[flow_idx][symbol_id][mod_comp_msg_index].set(reMask_1, reMask_2);

                                    mod_comp_config_temp->num_messages_per_list[flow_idx][symbol_id]++;
#if 0
                                        NVLOGC_FMT(TAG, "flw_idx: {}, sym_id: {}, msg_index: {}, udIqwith={}, mcScale_1={}, mcScale_2={}, start_prbu={}, num_prb={}, csf_1={}, csf_2={}, re_1={}, re_2={}", 
                                                flow_idx, symbol_id, mod_comp_msg_index, udIqwidth, scale_val.x, scale_val.y, section_info.start_prbu, section_info.num_prbu, csf_1, csf_2, reMask_1, reMask_2);
#endif
                                        memcpy(&umsg_tx_list.umsg_info_symbol_antenna[message_index++], &message_template, sizeof(message_template));
                                }
                            }
                        }
#if 0
                        if ((symbol_id == 4)&& (radio_app_hdr.slotId == 0)) {
                            NVLOGC_FMT(TAG, "channel {} flow_index {}, message_index {}, section_info.section_id {}, start_prbu {}, num_prbu {}",
                        channel_type & 0xF, flow_index, message_index, section_info.section_id, start_prbu, num_prbu);
                        }
#endif
                    }
                }
            }

            start_tx_time = end_tx_time;
            end_tx_time   = start_tx_time + symbol_duration; //symbol time
        }

        //copy the template contents to device memory    
        if(dl_comp_method == aerial_fh::UserDataCompressionMethod::MODULATION_COMPRESSION)
        { 
            batchedMemcpyHelper.updateMemcpy(mod_comp_prm, mod_comp_config_temp, sizeof(mod_compression_params), 
                    cudaMemcpyHostToDevice, dl_stream);
        }
        
        umsg_tx_list.num = message_index;

        if(pdctx->gpuCommDlEnabled()) {
            if(prepare_uplane_gpu_comm(peer_ptr->peer, const_cast<const fhproxy_umsg_tx*>(umsg_tx_list.umsg_info_symbol_antenna), umsg_tx_list.num,
                                        &(umsg_tx_list.txrq_gpu), cell_start_time, symbol_duration,pdctx->gpuCommEnabledViaCpu()))
            {
                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "aerial_fh::prepare_uplane error");
                return -1;
            }
        } else {
            UPlaneTxCompleteNotification notification{release_dlbuffer_cb, cb_obj};
            if(prepare_uplane(peer_ptr->peer, const_cast<const fhproxy_umsg_tx*>(umsg_tx_list.umsg_info_symbol_antenna), umsg_tx_list.num, notification, &(umsg_tx_list.txrq)))
            {
                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "aerial_fh::prepare_uplane error");
                return -1;
            }
        }
    }
    else
    {
        auto& section_info      = message_template.section_info;
        section_info.rb         = false;
        section_info.sym_inc    = false;
        // MODCOMP TODO
        section_info.mod_comp_enable = false;
        //Assuming the compression method will be the same for a given cell 
        if(dl_comp_method == aerial_fh::UserDataCompressionMethod::MODULATION_COMPRESSION)
            {
                section_info.mod_comp_enable = true;
                std::memset(mod_comp_config_temp->num_messages_per_list, 0, sizeof(mod_comp_config_temp->num_messages_per_list));
            }
        else
            section_info.mod_comp_enable = false;
        for(int symbol_id = 0; symbol_id < slot_info.symbols.size(); symbol_id++)
        {
            if(ru == SINGLE_SECT_MODE)
            {
                section_id = 0;
            }

            /* The window start information is needed in the gpu_comm_prepare_uplane*() code.
            This information is per symbol per slot so we could provide to aerial-fh this way rather
            than looping all the messages there.
            */
            message_template.tx_window.tx_window_start = start_tx_time.count();
            message_template.tx_window.tx_window_end   = end_tx_time.count();
            radio_app_hdr.symbolId                     = symbol_id;

            //NVLOGC_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "symbol_id {}, start_tx_time.count() {} and symbol duration {}", symbol_id, start_tx_time.count(), symbol_duration.count());

            for (int channel_type = slot_command_api::channel_type::PDSCH_CSIRS; channel_type <= slot_command_api::channel_type::PDCCH_DMRS; channel_type++)
            {
                //Different channels can have the same flow for the same cell
                auto &flows = peer_ptr->uplane_flows[channel_type];

                const std::vector<std::size_t> * prb_index_list = nullptr;
                slot_command_api::prb_info_list_t prbs = nullptr;
                if((channel_type == slot_command_api::channel_type::CSI_RS) &&
                    (slot_info.symbols[symbol_id][slot_command_api::channel_type::PDSCH_CSIRS].size() != 0))
                {
                    if(slot_info.alt_csirs_prb_info_idx_list[symbol_id].size())
                    {
                        prb_index_list = &slot_info.alt_csirs_prb_info_idx_list[symbol_id];
                        prbs = const_cast<slot_command_api::prb_info_list_t>(slot_info.alt_csirs_prb_info_list);
                    }
                    else
                        continue;
                }
                else
                {
                    prb_index_list = &slot_info.symbols[symbol_id][channel_type];
                    prbs = const_cast<slot_command_api::prb_info_list_t>(slot_info.prbs);
                }
                bool parse_portmask = false;
                if((channel_type == slot_command_api::channel_type::PDSCH || channel_type == slot_command_api::channel_type::PDSCH_CSIRS) && (ru != SINGLE_SECT_MODE))
                    parse_portmask = true;

                for (auto prb_info_idx : *prb_index_list)
                {
                    //NVLOGC_FMT(TAG, "channel type = {}, Start section ID = {}", channel_type, start_section_index);
                    auto &prb_info              = prbs[prb_info_idx];
                    auto num_prbu   = prb_info.common.numPrbc;
                    auto start_prbu = prb_info.common.startPrbc;
                    section_info.rb         = prb_info.common.useAltPrb ? 1 : 0;
                    section_info.section_id = section_id++;
                    section_info.num_prbu   = adjustPrbuCount(num_prbu);
                    section_info.start_prbu = start_prbu;
                    if(section_info.mod_comp_enable)
                    {
                         section_info.prb_size   = static_cast<uint32_t>(prb_info.comp_info.common.udIqWidth) * 3;
                    }
                    //if(channel_type == slot_command_api::channel_type::CSI_RS)
                    //NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "U-Plane Prb_info_index = {} channel type = {}, Start section ID = {} start_prbu ={} num_prbu = {}",prb_info_idx , channel_type, section_info.section_id, section_info.start_prbu, section_info.num_prbu);
                    radio_app_hdr.filterIndex = prb_info.common.filterIndex;

            
                    size_t num_ap_indices;
                    std::array<std::size_t,64> ap_index_list;

                    if(ru == SINGLE_SECT_MODE)
                    {
                        num_ap_indices = flows.size();
                    }
                    else if(parse_portmask)
                    {
                        if(pdctx->getUeMode())
                            parsePortMask(0b1111,num_ap_indices,ap_index_list);
                        else
                            parsePortMask(prb_info.common.portMask,num_ap_indices,ap_index_list);
                    }
                    else
                    {
                        num_ap_indices    = prb_info.common.numApIndices; 
                    }

                    if(num_ap_indices> flows.size())
                    {
                        num_ap_indices = flows.size();
                    }

                    for(auto ap_index = 0; ap_index < (num_ap_indices) && (message_index < MAX_UPLANE_MSGS_PER_SLOT); ap_index++)
                    {
                        auto flow_index         = parse_portmask ? ap_index_list[ap_index] : ap_index;
                        message_template.flow   = flows[flow_index].handle;
                        message_template.eaxcid = flows[flow_index].eAxC_id;
                        memcpy(&umsg_tx_list.umsg_info_symbol_antenna[message_index++], &message_template, sizeof(message_template));

                        //populate mod compression config params
                        //TODO: need to write a function for the following
                        if(section_info.mod_comp_enable)
                        {
                            uint32_t mod_comp_msg_index = mod_comp_config_temp->num_messages_per_list[flow_index][symbol_id];

                            mod_comp_config_temp->nprbs_per_list[flow_index][symbol_id][mod_comp_msg_index]     = section_info.num_prbu;
                            mod_comp_config_temp->prb_start_per_list[flow_index][symbol_id][mod_comp_msg_index] = section_info.start_prbu;

                            //assuming there could be max of 2 channels multiplexed on the same prb allocation
                            float2 scale_val = {0, 0};
                            scale_val.x      = prb_info.comp_info.modCompScalingValue[0];

                            auto udIqwidth    = prb_info.comp_info.common.udIqWidth;
                            auto csf_1        = prb_info.comp_info.sections[0].csf;
                            uint8_t csf_2     = 0;

                            uint16_t reMask_1 = static_cast<uint16_t>(prb_info.comp_info.sections[0].mcScaleReMask);
                            uint16_t reMask_2 = 0;

                            if(prb_info.comp_info.common.nSections == 2)
                            {
                                scale_val.y = prb_info.comp_info.modCompScalingValue[1];
                                csf_2       = prb_info.comp_info.sections[1].csf;
                                reMask_2    = prb_info.comp_info.sections[1].mcScaleReMask;
                            }

                            mod_comp_config_temp->scaling[flow_index][symbol_id][mod_comp_msg_index] = scale_val;
                            mod_comp_config_temp->params_per_list[flow_index][symbol_id][mod_comp_msg_index].set(static_cast<QamListParam::qamwidth>(udIqwidth), csf_1, csf_2);
                            mod_comp_config_temp->prb_params_per_list[flow_index][symbol_id][mod_comp_msg_index].set(reMask_1, reMask_2);

                            mod_comp_config_temp->num_messages_per_list[flow_index][symbol_id]++;

                            //NVLOGC_FMT(TAG, "u_plane_mod_comp, udIqwith={}, mcScale_1={}, mcScale_2={}, num_prb={}, symbol_id={}, csf_1={}, csf_2={}", 
                            //    udIqwidth, scale_val.x, scale_val.y, section_info.num_prbu, symbol_id, csf_1, csf_2);
                        }

#if 0
                        if ((symbol_id == 4)&& (radio_app_hdr.slotId == 0)) {
                            NVLOGC_FMT(TAG, "channel {} flow_index {}, message_index {}, section_info.section_id {}, start_prbu {}, num_prbu {}",
                        channel_type & 0xF, flow_index, message_index, section_info.section_id, start_prbu, num_prbu);
    }
    #endif
                    }

                    if ((num_prbu > ORAN_MAX_PRB_X_SECTION) && (num_prbu < ORAN_MAX_PRB_X_SLOT))
                    {
                        message_template.section_info.section_id = section_id++;
                        message_template.section_info.start_prbu = start_prbu + 255;
                        message_template.section_info.num_prbu = num_prbu - 255;
                        if(section_info.mod_comp_enable)
                        {
                            message_template.section_info.prb_size   = static_cast<uint32_t>(prb_info.comp_info.common.udIqWidth) * 3;
                        }
                        for(int ap_index = 0; (ap_index < num_ap_indices) && (message_index < MAX_UPLANE_MSGS_PER_SLOT); ap_index++)
                        {
                            auto flow_index = parse_portmask ? ap_index_list[ap_index]: ap_index;
                            message_template.flow       = flows[flow_index].handle;
                            message_template.eaxcid = flows[flow_index].eAxC_id;
                            memcpy(&umsg_tx_list.umsg_info_symbol_antenna[message_index++], &message_template, sizeof(message_template));
                            if(section_info.mod_comp_enable)
                            {
                                uint32_t mod_comp_msg_index = mod_comp_config_temp->num_messages_per_list[flow_index][symbol_id];

                                mod_comp_config_temp->nprbs_per_list[flow_index][symbol_id][mod_comp_msg_index]     = section_info.num_prbu;
                                mod_comp_config_temp->prb_start_per_list[flow_index][symbol_id][mod_comp_msg_index] = section_info.start_prbu;

                                //assuming there could be max of 2 channels multiplexed on the same prb allocation
                                float2 scale_val = {0, 0};
                                scale_val.x      = prb_info.comp_info.modCompScalingValue[0];

                                uint8_t udIqwidth = static_cast<uint8_t>(prb_info.comp_info.common.udIqWidth);
                                uint8_t csf_1     = static_cast<uint8_t>(prb_info.comp_info.sections[0].csf);
                                uint8_t csf_2     = 0;

                                uint16_t reMask_1 = static_cast<uint16_t>(prb_info.comp_info.sections[0].mcScaleReMask);
                                uint16_t reMask_2 = 0;

                                if(prb_info.comp_info.common.nSections == 2)
                                {
                                    scale_val.y = prb_info.comp_info.modCompScalingValue[1];
                                    csf_2       = static_cast<uint8_t>(prb_info.comp_info.sections[1].csf);
                                    reMask_2    = static_cast<uint16_t>(prb_info.comp_info.sections[1].mcScaleReMask);
                                }

                                mod_comp_config_temp->scaling[flow_index][symbol_id][mod_comp_msg_index] = scale_val;
                                mod_comp_config_temp->params_per_list[flow_index][symbol_id][mod_comp_msg_index].set(static_cast<QamListParam::qamwidth>(udIqwidth), csf_1, csf_2);
                                mod_comp_config_temp->prb_params_per_list[flow_index][symbol_id][mod_comp_msg_index].set(reMask_1, reMask_2);

                                mod_comp_config_temp->num_messages_per_list[flow_index][symbol_id]++;
                                // NVLOGC_FMT(TAG, "File {} Line {} prb_info udIqwith: 0x{:x} ReMask: 0x{:x}, ext5_set1:  mcScaleReMask=0x{:x}  mcScaleOffset=0x{:x}, csf=0x{:x}, ext5_set2:  mcScaleReMask=0x{:x}  mcScaleOffset=0x{:x}, csf=0x{:x} section_info.prb_size {}", 
                                //     __FILE__, __LINE__, +udIqwidth, +prb_info.common.reMask, reMask_1, scale_val.x, +csf_1, reMask_2, scale_val.y +csf_2, section_info.prb_size);
                            }
                        }
                    }
                }
    #if 0
                if ((channel_type == slot_command_api::channel_type::PDSCH_CSIRS) && (slot_info.symbols[symbol_id][slot_command_api::channel_type::PDSCH_CSIRS].size() > 0))
                {
                    channel_type +=1;
                }
    #endif
            }

            start_tx_time = end_tx_time;
            end_tx_time   = start_tx_time + symbol_duration; //symbol time
        }

        //copy the template contents to device memory    
        if(dl_comp_method == aerial_fh::UserDataCompressionMethod::MODULATION_COMPRESSION)
        { 
            batchedMemcpyHelper.updateMemcpy(mod_comp_prm, mod_comp_config_temp, sizeof(mod_compression_params), 
                    cudaMemcpyHostToDevice, dl_stream);
        }
        
        
        umsg_tx_list.num = message_index;

        if(pdctx->gpuCommDlEnabled()) {
            if(prepare_uplane_gpu_comm(peer_ptr->peer, const_cast<const fhproxy_umsg_tx*>(umsg_tx_list.umsg_info_symbol_antenna), umsg_tx_list.num,
                                        &(umsg_tx_list.txrq_gpu), cell_start_time, symbol_duration,pdctx->gpuCommEnabledViaCpu()))
            {
                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "aerial_fh::prepare_uplane error");
                return -1;
            }
        } else {
            UPlaneTxCompleteNotification notification{release_dlbuffer_cb, cb_obj};
            if(prepare_uplane(peer_ptr->peer, const_cast<const fhproxy_umsg_tx*>(umsg_tx_list.umsg_info_symbol_antenna), umsg_tx_list.num, notification, &(umsg_tx_list.txrq)))
            {
                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "aerial_fh::prepare_uplane error");
                return -1;
            }
        }

    }
    // NVSLOGD(TAG) << "txUplanePrepareSymbol symbol " << symbol_id << " TX in " << Time::getDifferenceNsToNow(start_tx_time).count() << " ns umsg_slot_list size " << umsg_slot_list.size();

    return 0;
}

int FhProxy::UserPlaneSendPackets(peer_id_t peer_id, struct umsg_fh_tx_msg& umsg_tx_list)
{
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();

    // NVSLOGD(TAG) << "Sending from peer " << peer_id;

    if(0 == aerial_fh::send_uplane(umsg_tx_list.txrq))
    {
        NVLOGE_FMT(TAG, AERIAL_ORAN_FH_EVENT, "aerial_fh::send_uplane error");
        return -1;
    }

    return 0;
}


int FhProxy::RingCPUDoorbell(TxRequestGpuPercell *pTxRequestGpuPercell, PreparePRBInfo &prb_info, PacketTimingInfo &packet_timing_info) {
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();

    if(!pdctx->gpuCommDlEnabled()) {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "GPU-init comm not enabled");
        return -ENOTSUP;
    }

    if(0 != aerial_fh::ring_cpu_doorbell(nic_map[pTxRequestGpuPercell->nic_name], pTxRequestGpuPercell, prb_info, packet_timing_info))
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "aerial_fh::ring_cpu_doorbell error");
        return -1;
    }

    return 0;    
}

int FhProxy::UserPlaneSendPacketsGpuComm(TxRequestGpuPercell *pTxRequestGpuPercell,PreparePRBInfo &prb_info)
{
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();

    if(!pdctx->gpuCommDlEnabled()) {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "GPU-init comm not enabled");
        return -ENOTSUP;
    }
    // NVSLOGE(TAG) << "Sending with GPUcomm TXQ requests # " << tx_v.size();

    //FIXME: How to manage multiple nics?
    // auto it = nic_map.begin();

    if(0 != aerial_fh::send_uplane_gpu_comm(nic_map[pTxRequestGpuPercell->nic_name], pTxRequestGpuPercell, prb_info))
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "aerial_fh::send_uplane_gpu_comm error");
        return -1;
    }

    return 0;
}

int FhProxy::triggerCqeTracerCb(std::string& nic_name,TxRequestGpuPercell *pTxRequestGpuPercell)
{
    if(0 != aerial_fh::trigger_cqe_tracer_cb(nic_map[nic_name],pTxRequestGpuPercell))
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "aerial_fh::triggerCqeTracerCb error");
        return -1;
    }

    return 0;    
}

int FhProxy::setTriggerTsGpuComm(std::string& nic_name,uint32_t slot_idx,uint64_t trigger_ts)
{
    if(0 != aerial_fh::set_TriggerTs_GpuComm(nic_map[nic_name], slot_idx, trigger_ts))
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "aerial_fh::set_TriggerTs_GpuComm error");
        return -1;
    }

    return 0;
}


int FhProxy::print_max_delays(std::string nic_name)
{
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();

    if(0 != aerial_fh::print_max_delays(nic_map[nic_name]))
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "aerial_fh::print_max_delays error");
        return -1;
    }

    return 0;
}


int FhProxy::receiveUPlane(struct fh_peer_t * peer_ptr, struct umsg_fh_rx& umsg_item)
{
    size_t rx_msgs_req = static_cast<size_t>(CK_ORDER_PKTS_BUFFERING);
    size_t rx_msgs_recv = rx_msgs_req;

    t_ns wait_time = t_ns(CK_ORDER_PKTS_BUFFERING_FLOW_NS * (peer_ptr->uplane_flows[slot_command_api::channel_type::PUSCH].size() + peer_ptr->uplane_flows[slot_command_api::channel_type::PRACH].size()));
    t_ns timeout = Time::nowNs() + wait_time;

    if(receive_until(peer_ptr->peer, &(umsg_item.umsg_info[0]), &rx_msgs_recv, timeout.count()))
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "receive_until error");
        return -1;
    }

    if (rx_msgs_recv < rx_msgs_req)
    {
        NVLOGI_FMT(TAG, "Receive U-plane expired after {} us. {}/{} packets received", (Time::NsToUs(wait_time)).count(), rx_msgs_recv, rx_msgs_req);
    }
    else
    {
        NVLOGI_FMT(TAG, "Receive U-plane received all {} packets requested", rx_msgs_recv);
    }

    umsg_item.num = static_cast<int>(rx_msgs_recv);
    return 0;
}

int FhProxy::UserPlaneReceivePackets(peer_id_t peer_id)
{
    PhyDriverCtx*           pdctx = StaticConversion<PhyDriverCtx>(this->getPhyDriverHandler()).get();
    int                     ret = 0, rx_pkts = 0, x = 0, totMsgCnt = 0;
    struct rx_queue_sync *  sync_list;
    struct fh_peer_t *      peer_ptr;

    peer_ptr = getPeerFromId(peer_id);
    if(!peer_ptr)
        return EINVAL;


    ret = receiveUPlane(peer_ptr, peer_ptr->rx_order_items.umsg_rx_list[peer_ptr->rx_order_items.umsg_rx_index]);
    if(ret < 0)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "recvUplane PUSCH error");
        return -1; //goto error_next;
    }

    if(peer_ptr->rx_order_items.umsg_rx_list[peer_ptr->rx_order_items.umsg_rx_index].num > 0)
    {
        // NVSLOGE(TAG, AERIAL_CUPHYDRV_API_EVENT)  << "Receiving cell " << cell_ptr->getPhyId() << " umsg item " << peer_ptr->rx_order_items.umsg_rx_index << " num: " << peer_ptr->rx_order_items.umsg_rx_list[peer_ptr->rx_order_items.umsg_rx_index].num;
        for(x = 0; x < peer_ptr->rx_order_items.umsg_rx_list[peer_ptr->rx_order_items.umsg_rx_index].num; x++)
        {
            peer_ptr->rx_order_items.sync_list[peer_ptr->rx_order_items.sync_item].addr[x] = reinterpret_cast<uintptr_t>(peer_ptr->rx_order_items.umsg_rx_list[peer_ptr->rx_order_items.umsg_rx_index].umsg_info[x].buffer);
            peer_ptr->rx_order_items.sync_list[peer_ptr->rx_order_items.sync_item].rx_timestamp[x] = peer_ptr->rx_order_items.umsg_rx_list[peer_ptr->rx_order_items.umsg_rx_index].umsg_info[x].rx_timestamp;

            if(peer_ptr->rx_order_items.sync_list[peer_ptr->rx_order_items.sync_item].addr[x] == 0)
            {
                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Error: sync_list[ {} ].addr[ {}] == 0", peer_ptr->rx_order_items.sync_item, x );
                EXIT_L1(EXIT_FAILURE);
            }
            else
            {
                    NVLOGI_FMT(TAG,"{}:peer_ptr->rx_order_items.sync_item {} x {} addr {}",__func__,peer_ptr->rx_order_items.sync_item,x,(void*)peer_ptr->rx_order_items.sync_list[peer_ptr->rx_order_items.sync_item].addr[x]);
            }
            totMsgCnt++;
        }

        for(; x < CK_ORDER_PKTS_BUFFERING; x++)
            peer_ptr->rx_order_items.sync_list[peer_ptr->rx_order_items.sync_item].addr[x] = 0;

        peer_ptr->rx_order_items.sync_list[peer_ptr->rx_order_items.sync_item].umsg_info = peer_ptr->rx_order_items.umsg_rx_list[peer_ptr->rx_order_items.umsg_rx_index].umsg_info;
        peer_ptr->rx_order_items.sync_list[peer_ptr->rx_order_items.sync_item].umsg_num  = peer_ptr->rx_order_items.umsg_rx_list[peer_ptr->rx_order_items.umsg_rx_index].num;

        flushMemory(peer_ptr);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        ((uint32_t*)(peer_ptr->rx_order_items.sync_ready_list_gdr->addrh()))[peer_ptr->rx_order_items.sync_item] = (int)SYNC_PACKET_STATUS_READY; // Order kernel unblock
        ACCESS_ONCE(peer_ptr->rx_order_items.sync_list[peer_ptr->rx_order_items.sync_item].status)              = (int)SYNC_PACKET_STATUS_READY; // CPU thread tracking
        std::atomic_thread_fence(std::memory_order_seq_cst);
        flushMemory(peer_ptr);

        rx_pkts += peer_ptr->rx_order_items.umsg_rx_list[peer_ptr->rx_order_items.umsg_rx_index].num;
        ret = peer_ptr->rx_order_items.umsg_rx_list[peer_ptr->rx_order_items.umsg_rx_index].num;

        // NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Cell {} RX pkts {} set rsync item {} with {}/{} msgs umsg item {} at {}",
        //                 cell_ptr->getPhyId(), rx_pkts, peer_ptr->rx_order_items.sync_item,
        //                 umsg_rx_list[peer_ptr->rx_order_items.umsg_rx_index].num, CK_ORDER_PKTS_BUFFERING, peer_ptr->rx_order_items.umsg_rx_index, Time::nowNs()
        //         );

        // NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "CELL {} TOT RX pkts {}", cell_ptr->getPhyId(), rx_pkts);

        peer_ptr->rx_order_items.umsg_rx_index = (peer_ptr->rx_order_items.umsg_rx_index + 1) % RX_QUEUE_SYNC_LIST_ITEMS;
        peer_ptr->rx_order_items.umsg_rx_list[peer_ptr->rx_order_items.umsg_rx_index].num = 0;
        peer_ptr->rx_order_items.sync_item = (peer_ptr->rx_order_items.sync_item + 1) % RX_QUEUE_SYNC_LIST_ITEMS;

        // NVSLOGE(TAG, AERIAL_CUPHYDRV_API_EVENT) << "Cell " << cell_ptr->getPhyId() << " received num_msgs " << umsg_item.num << " flow " << f.first << " Order item update time " << end_t_rx.count() << "-" << start_t_item.count() << " = " << Time::NsToUs(end_t_rx - start_t_item).count();
    }

    return ret;
}

int FhProxy::UserPlaneReceivePacketsCPU(int index, MsgReceiveInfo* info, size_t& num_msgs)
{
    struct fh_peer_t *      peer_ptr;
    int ret = 0;
    peer_ptr = getPeerFromAbsoluteId(index);
    if(!peer_ptr)
    {
        num_msgs = 0;
        return EINVAL;
    }
    ret = aerial_fh::receive(peer_ptr->peer, &info[0], &num_msgs);
    if (ret != 0){
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "UserPlaneReceivePacketsCPU error");
        return EINVAL;
    }
    return 0;
}

int FhProxy::UserPlaneFreePacketsCPU(MsgReceiveInfo* info, size_t num_msgs)
{
    int ret = aerial_fh::free_rx_messages(&info[0],num_msgs);
    if (ret != 0)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "UserPlaneFreePacketsCPU error");
        return EINVAL;
    }
    return 0;
}


int FhProxy::UserPlaneFreeMsg(peer_id_t peer_id) {
    struct fh_peer_t *      peer_ptr;

    peer_ptr = getPeerFromId(peer_id);
    if(!peer_ptr)
        return EINVAL;

    /*
     * Here we assume that there are enough items to be processed per slot, no collisions.
     */
    while(ACCESS_ONCE(peer_ptr->rx_order_items.sync_list[peer_ptr->rx_order_items.last_ufree].status) == (int)SYNC_PACKET_STATUS_DONE)
    {
        // NVSLOGD(TAG) << "freeUPlaneMsgs cell " << getId() << " item " << peer_ptr->rx_order_items.last_ufree << " mbufs num " << umsg_rx_list[peer_ptr->rx_order_items.last_ufree].num << " address umsg " << umsg_rx_list[peer_ptr->rx_order_items.last_ufree].umsg_info;

        if(free_rx_messages(reinterpret_cast<const fhproxy_umsg_rx*>(peer_ptr->rx_order_items.sync_list[peer_ptr->rx_order_items.last_ufree].umsg_info),
                                peer_ptr->rx_order_items.sync_list[peer_ptr->rx_order_items.last_ufree].umsg_num
                            )
        )
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "free_rx_messages error");
            return -1;
        }

        ACCESS_ONCE(peer_ptr->rx_order_items.sync_list[peer_ptr->rx_order_items.last_ufree].status) = (int)SYNC_PACKET_STATUS_FREE;

        peer_ptr->rx_order_items.last_ufree = (peer_ptr->rx_order_items.last_ufree + 1) % RX_QUEUE_SYNC_LIST_ITEMS;
        // count++;
        // if(count >= RX_QUEUE_SYNC_LIST_ITEMS)
        //     break;
    }

    return 0;
}

int FhProxy::UserPlaneCheckMsg(peer_id_t peer_id) {
    struct fh_peer_t *  peer_ptr;
    int                 idx;

    peer_ptr = getPeerFromId(peer_id);
    if(!peer_ptr)
        return EINVAL;

    // idx = (peer_ptr->rx_order_items.last_ufree + 1) % RX_QUEUE_SYNC_LIST_ITEMS;
    idx = peer_ptr->rx_order_items.last_ufree;

    if(ACCESS_ONCE(peer_ptr->rx_order_items.sync_list[idx].status) == (int)SYNC_PACKET_STATUS_READY)
        return 1;

    return 0;
}


int FhProxy::updateMetrics()
{
    if(update_metrics(fhi))
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "updateMetrics failed");
        return -1;
    }

    return 0;
}

uint16_t FhProxy::countPuschPucchPrbs(slot_command_api::slot_info_t& slot_info, size_t pusch_eAxC_id_count, size_t pucch_eAxC_id_count, uint32_t* pusch_prb_symbol_map, uint32_t* pucch_prb_symbol_map,uint32_t* num_order_cells_sym_mask_arr,int cell_idx, uint8_t& pusch_prb_non_zero, void* param) const
{
    SlotMapUl*                                                                   slot_map = (SlotMapUl*)param;
    PhyDriverCtx*                                                                pdctx    = StaticConversion<PhyDriverCtx>(slot_map->getPhyDriverHandler()).get();    
    uint16_t pusch_prb_count = 0;
    

    for(int symbol_id = 0; symbol_id < slot_info.symbols.size(); symbol_id++)
    {
        for (auto prb_info_idx : slot_info.symbols[symbol_id][slot_command_api::channel_type::PUSCH])
        {
            auto &prb_info = slot_info.prbs[prb_info_idx].common;
            if(pdctx->getmMIMO_enable())
            {
                pusch_prb_count += (prb_info.numPrbc*__builtin_popcount(prb_info.portMask)) * prb_info.numSymbols;
            }
            else
            {
                pusch_prb_count += prb_info.numPrbc * prb_info.numSymbols;
            }
            for(int symbol_id_inner=symbol_id;symbol_id_inner<(symbol_id+prb_info.numSymbols);symbol_id_inner++)
            {         
                if(pdctx->getmMIMO_enable())
                {
                    pusch_prb_symbol_map[symbol_id_inner]+=(prb_info.numPrbc*__builtin_popcount(prb_info.portMask));
                }
                else
                {
                    pusch_prb_symbol_map[symbol_id_inner]+=prb_info.numPrbc;            
                }       
            }
            NVLOGD_FMT(TAG,"{}:prb_info_idx={} pusch_prb_count={} pusch_prb_symbol_map[{}]={} prb_info.numSymbols={} slot_info.symbols.size()={} symbol_id={}",__func__,prb_info_idx,pusch_prb_count,symbol_id,pusch_prb_symbol_map[symbol_id],prb_info.numSymbols,slot_info.symbols.size(),symbol_id);
        }
        if(!pdctx->getmMIMO_enable())
        {
            pusch_prb_symbol_map[symbol_id]*=pusch_eAxC_id_count;
        }
        if((pusch_prb_symbol_map[symbol_id]>0) && (pusch_prb_non_zero==0)){
            pusch_prb_non_zero=1;
        }
    }

    uint16_t pucch_prb_count = 0;

    for(int symbol_id = 0; symbol_id < slot_info.symbols.size(); symbol_id++)
    {
        for (auto prb_info_idx : slot_info.symbols[symbol_id][slot_command_api::channel_type::PUCCH])
        {
            auto &prb_info = slot_info.prbs[prb_info_idx].common;
            if(pdctx->getmMIMO_enable())
            {
                pucch_prb_count += (prb_info.numPrbc*__builtin_popcount(prb_info.portMask)) * prb_info.numSymbols;

            }
            else
            {
                pucch_prb_count += prb_info.numPrbc * prb_info.numSymbols;
            }
            for(int symbol_id_inner=symbol_id;symbol_id_inner<(symbol_id+prb_info.numSymbols);symbol_id_inner++)
            {  
                if(pdctx->getmMIMO_enable())
                {
                    pucch_prb_symbol_map[symbol_id_inner]+=(prb_info.numPrbc*__builtin_popcount(prb_info.portMask)); //Include PUCCh PRBs as well since we do not differentiate between PUSCH/PUCCH PRBs in FH
                }
                else
                {
                    pucch_prb_symbol_map[symbol_id_inner]+=prb_info.numPrbc; //Include PUCCh PRBs as well since we do not differentiate between PUSCH/PUCCH PRBs in FH
                }              
            }            
        }
        if(!pdctx->getmMIMO_enable())
        {
            pucch_prb_symbol_map[symbol_id]*=pucch_eAxC_id_count;
        }
        pusch_prb_symbol_map[symbol_id]+=pucch_prb_symbol_map[symbol_id];
        if(pusch_prb_symbol_map[symbol_id]>0){
            num_order_cells_sym_mask_arr[symbol_id]|=(0x1<<cell_idx);
        }

    }
    if(pdctx->getmMIMO_enable())
    {
        NVLOGD_FMT(TAG,"{} MMIMO:Total PRB count={},pusch_prb_count={},pucch_prb_count={}",
                __func__,((pusch_prb_count) + (pucch_prb_count)),pusch_prb_count,pucch_prb_count);
        return (pusch_prb_count  + pucch_prb_count );
    }
    else
    {
        NVLOGD_FMT(TAG,"{}:Total PRB count={},pusch_prb_count={},pusch_eAxC_id_count={},pucch_prb_count={},pucch_eAxC_id_count={}",
                __func__,((pusch_prb_count * pusch_eAxC_id_count) + (pucch_prb_count * pucch_eAxC_id_count)),pusch_prb_count,pusch_eAxC_id_count,pucch_prb_count,pucch_eAxC_id_count);
        return (pusch_prb_count * pusch_eAxC_id_count) + (pucch_prb_count * pucch_eAxC_id_count);
    }
}

uint16_t FhProxy::countPrachPrbs(slot_command_api::slot_info_t& slot_info, size_t prach_eAxC_id_count, void* param) const
{
    SlotMapUl* slot_map = (SlotMapUl*)param;
    PhyDriverCtx* pdctx    = StaticConversion<PhyDriverCtx>(slot_map->getPhyDriverHandler()).get(); 
    uint16_t prach_prb_count = 0;

    for(int symbol_id = 0; symbol_id < slot_info.symbols.size(); symbol_id++)
    {
        for (auto prb_info_idx : slot_info.symbols[symbol_id][slot_command_api::channel_type::PRACH])
        {
            auto &prb_info = slot_info.prbs[prb_info_idx].common;
            if(pdctx->getmMIMO_enable())
            {
                prach_prb_count += (prb_info.numPrbc*__builtin_popcount(prb_info.portMask)) * prb_info.numSymbols;
            }
            else
            {
                prach_prb_count += prb_info.numPrbc * prb_info.numSymbols;
            }
        }
    }
    if(!pdctx->getmMIMO_enable())
    {
        return prach_prb_count * prach_eAxC_id_count;
    }
    else
    {
        return prach_prb_count;
    }
}

uint32_t FhProxy::countSrsPrbs(slot_command_api::slot_info_t& slot_info, size_t srs_eAxC_id_count) const
{
    uint32_t srs_prb_count = 0;
    for(int symbol_id = 0; symbol_id < slot_info.symbols.size(); symbol_id++)
    {
        for (auto prb_info_idx : slot_info.symbols[symbol_id][slot_command_api::channel_type::SRS])
        {
            auto &prb_info = slot_info.prbs[prb_info_idx].common;
            srs_prb_count += prb_info.numPrbc * prb_info.numSymbols;
        }
    }
    return srs_prb_count * srs_eAxC_id_count;
}
