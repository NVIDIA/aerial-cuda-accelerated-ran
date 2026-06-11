/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#define TAG (NVLOG_TAG_BASE_CUPHY_DRIVER + 30) // "DRV.CELL"
#define TAG_STARTUP_TIMES (NVLOG_TAG_BASE_CUPHY_CONTROLLER + 5) // "CTL.STARTUP_TIMES"

#include "cell.hpp"
#include "context.hpp"
#include "constant.hpp"
#include <cmath>
#include "ti_generic.hpp"

#define PRINT_MEMORY_FOOTPRINT_IN_CTOR 1

Cell::Cell(
    phydriver_handle            _pdh,
    cell_id_t                   _cell_id,
    const cell_mplane_info&    _mplane,
    FhProxy*                    _fh_proxy,
    GpuDevice*                  _gDev,
    uint32_t                    _idx)
    :
    pdh(_pdh),
    cell_id(_cell_id),
    fh_proxy(_fh_proxy),
    gDev(_gDev),
    active(CELL_INACTIVE),
    active_srs(CELL_INACTIVE),
    metrics(_mplane.mplane_id),
    idx(_idx)
{
    TI_GENERIC_INIT("Cell::Cell",15);
    TI_GENERIC_ADD("Start Task");

    TI_GENERIC_ADD("PhyDriverCtx get");
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    TI_GENERIC_ADD("geteAxCIdsUl");
    std::vector<uint16_t>& eAxC_list_uplink=geteAxCIdsUl();
    
    mplane_id              = _mplane.mplane_id;
    ru                     = _mplane.ru;
    src_eth_addr           = _mplane.src_eth_addr;
    dst_eth_addr           = _mplane.dst_eth_addr;
    vlan_tci               = _mplane.vlan_tci;
    txq_count_uplane       = _mplane.nic_cfg.txq_count_uplane;
    nic_name               = _mplane.nic_name;
    nic_index              = _mplane.nic_index;
    dlc_core_index         = _mplane.dlc_core_index;
    tv_pusch_h5            = _mplane.tv_pusch_h5;
    tv_srs_h5              = _mplane.tv_srs_h5;
    dl_comp_meth           = _mplane.dl_comp_meth;
    ul_comp_meth           = _mplane.ul_comp_meth;
    dl_bit_width           = _mplane.dl_bit_width;
    ul_bit_width           = _mplane.ul_bit_width;
    fs_offset_dl           = _mplane.fs_offset_dl;
    exponent_dl            = _mplane.exponent_dl;
    ref_dl                 = _mplane.ref_dl;
    fs_offset_ul           = _mplane.fs_offset_ul;
    exponent_ul            = _mplane.exponent_ul;
    max_amp_ul             = _mplane.max_amp_ul;
    section_3_time_offset  = _mplane.section_3_time_offset;
    beta_dl                = 1;
    beta_ul                = 1;
    oam_linear_gain        = 1;
    t1a_max_up_ns          = _mplane.t1a_max_up_ns;
    t1a_max_cp_ul_ns       = _mplane.t1a_max_cp_ul_ns;
    t1a_min_cp_ul_ns       = _mplane.t1a_min_cp_ul_ns;
    ta4_min_ns             = _mplane.ta4_min_ns;
    ta4_max_ns             = _mplane.ta4_max_ns;
    ta4_min_ns_srs         = _mplane.ta4_min_ns_srs;
    ta4_max_ns_srs         = _mplane.ta4_max_ns_srs;
    t1a_min_cp_dl_ns       = _mplane.t1a_min_cp_dl_ns;
    t1a_max_cp_dl_ns       = _mplane.t1a_max_cp_dl_ns;
    ul_u_plane_tx_offset_ns = _mplane.ul_u_plane_tx_offset_ns;
    tcp_adv_dl_ns          = _mplane.tcp_adv_dl_ns;
    eAxC_ids               = _mplane.eAxC_ids;
    mplane_destinations_   = _mplane.destinations;
    fh_len_range           = _mplane.fh_len_range;
    nMaxRxAnt              = _mplane.nMaxRxAnt;
    srs_cell_dyn_index   = -1;
    num_consecutive_ok_timeout   = 0;
    num_consecutive_unhealthy_slots = 0;
    TI_GENERIC_ADD("L1 Sanity Check");
    if(pdctx->enableL1ParamSanityCheck())
    {
        pusch_prb_stride       = ((_mplane.pusch_prb_stride == 0 || _mplane.pusch_prb_stride > ORAN_MAX_PRB)  ? ORAN_PUSCH_PRBS_X_PORT_X_SYMBOL : _mplane.pusch_prb_stride);
        prach_prb_stride       = ((_mplane.prach_prb_stride == 0 || _mplane.prach_prb_stride > ORAN_PRACH_PRB) ? ORAN_PRACH_B4_PRBS_X_PORT_X_SYMBOL : _mplane.prach_prb_stride);
        srs_prb_stride         = (_mplane.srs_prb_stride == 0 ? ORAN_SRS_PRBS_X_PORT_X_SYMBOL : _mplane.srs_prb_stride);    
    }
    else
    {
        pusch_prb_stride       = (_mplane.pusch_prb_stride == 0 ? ORAN_PUSCH_PRBS_X_PORT_X_SYMBOL : _mplane.pusch_prb_stride);
        prach_prb_stride       = (_mplane.prach_prb_stride == 0 ? ORAN_PRACH_B4_PRBS_X_PORT_X_SYMBOL : _mplane.prach_prb_stride);
        srs_prb_stride         = (_mplane.srs_prb_stride == 0 ? ORAN_SRS_PRBS_X_PORT_X_SYMBOL : _mplane.srs_prb_stride);    
    }

    pusch_ldpc_max_num_itr_algo_type = static_cast<uint8_t>(_mplane.pusch_ldpc_max_num_itr_algo_type);
    pusch_fixed_max_num_ldpc_itrs = static_cast<uint8_t>(_mplane.pusch_fixed_max_num_ldpc_itrs);
    pusch_ldpc_early_termination = static_cast<uint8_t>(_mplane.pusch_ldpc_early_termination);
    pusch_ldpc_algo_index        = static_cast<uint8_t>(_mplane.pusch_ldpc_algo_index);
    pusch_ldpc_flags             = static_cast<uint8_t>(_mplane.pusch_ldpc_flags);
    pusch_ldpc_use_half          = static_cast<uint8_t>(_mplane.pusch_ldpc_use_half);
    pusch_nMaxPrb                = static_cast<uint16_t>(_mplane.pusch_nMaxPrb);
    pusch_nMaxRx                = static_cast<uint16_t>(_mplane.pusch_nMaxRx);

    dl_grid_size = ORAN_MAX_PRB_X_SLOT;
    ul_grid_size = ORAN_MAX_PRB_X_SLOT;

    phy_stat.phyCellId  = DEFAULT_PHY_CELL_ID;

    TI_GENERIC_ADD("mf.init");
    mf.init(_pdh, std::string("Cell"), sizeof(Cell));

    TI_GENERIC_ADD("Create eaxc list");
    for (int channel = slot_command_api::channel_type::PUSCH; channel < slot_command_api::channel_type::SRS; channel++)
    {
        for (auto eAxC : eAxC_ids[channel])
        {
            auto it = std::find(std::begin(eAxC_list_uplink), std::end(eAxC_list_uplink), (uint16_t) eAxC);
            if(it!=std::end(eAxC_list_uplink))
                continue;
            eAxC_list_uplink.push_back(eAxC);
        }
    }

    TI_GENERIC_ADD("registerPeer");
    for(auto& nic : fh_proxy->getNicList())
    {
        if(fh_proxy->registerPeer(mplane_id, nic2peer_map[nic], src_eth_addr, dst_eth_addr, vlan_tci, txq_count_uplane, dl_comp_meth, dl_bit_width, gDev->getId(), nic, &nic2doca_rxq_info_map[nic], &nic2doca_rxq_info_srs_map[nic], eAxC_list_uplink, (std::vector<uint16_t>&)geteAxCIdsSrs(), (std::vector<uint16_t>&)geteAxCIdsPdsch()))
            PHYDRIVER_THROW_EXCEPTIONS(-1, "Add Peer for cell returned error");

        TI_GENERIC_ADD("registerFlow");
        for(int channel = slot_command_api::channel_type::PDSCH_CSIRS; channel < slot_command_api::channel_type::CHANNEL_MAX; channel++)
        {
            // dMIMO: SRS may be sounded by multiple PEs (destinations). Register each
            // destination's SRS eAxC with that destination's dst MAC so the SRS C-plane
            // reaches the right PE over this single (primary) peer's TX queue. The SRS
            // U-plane is caught back into the primary SRS RX queue by the wildcard src
            // MAC RX rule, so no extra peer/RX queue is needed.
            if(!mplane_destinations_.empty() && channel == slot_command_api::channel_type::SRS)
            {
                for(const auto& dest : mplane_destinations_)
                {
                    for(auto eAxC : dest.eAxC_ids[slot_command_api::channel_type::SRS])
                    {
                        if(fh_proxy->registerFlow(nic2peer_map[nic], eAxC, vlan_tci, slot_command_api::channel_type::SRS, &dest.dst_eth_addr))
                            PHYDRIVER_THROW_EXCEPTIONS(-1, "Add flow returned error");
                    }
                }
                continue;
            }
            for(auto eAxC : eAxC_ids[channel])
            {
                if(fh_proxy->registerFlow(nic2peer_map[nic], eAxC, vlan_tci, static_cast<slot_command_api::channel_type>(channel)))
                    PHYDRIVER_THROW_EXCEPTIONS(-1, "Add flowreturned error");
            }
        }
    }

    TI_GENERIC_ADD("Init UL Lists");
    int section_type_1_ant = std::max(std::max(geteAxCNumPusch(), geteAxCNumPucch()), (size_t)nMaxRxAnt);
    int section_type_3_ant = std::max(geteAxCNumPrach(), (size_t)nMaxRxAnt);

    uint8_t ul_input_buffer_num_per_cell = pdctx->getUlInputBufferPerCell();
    uint8_t ul_input_buffer_num_per_cell_srs = pdctx->getUlInputBufferPerCellSrs();
    for(int i = 0; i < ul_input_buffer_num_per_cell; i++)
    {
        ulbuf_st1_list.push_back(std::unique_ptr<ULInputBuffer>(new ULInputBuffer(pdh, gDev, cell_id, UL_ST1_AP_BUF_SIZE * section_type_1_ant)));
    }

    for(int i = 0; i < ul_input_buffer_num_per_cell; i++)
    {
        ulbuf_st3_list.push_back(std::unique_ptr<ULInputBuffer>(new ULInputBuffer(pdh, gDev, cell_id, UL_ST3_AP_BUF_SIZE * section_type_3_ant)));
    }

    for(int i = 0; i < ul_input_buffer_num_per_cell_srs; i++)
    {
        ulbuf_st2_list.push_back(std::unique_ptr<ULInputBuffer>(new ULInputBuffer(pdh, gDev, cell_id, UL_ST2_AP_BUF_SIZE * geteAxCNumSrs())));
    }
    ulbuf_st1_index     = 0;
    ulbuf_st2_index     = 0;
    ulbuf_st3_index     = 0;
    dlbuf_index         = 0;

    if(pdctx->get_ul_pcap_capture_enable() == 1)
    {
        auto ul_pcap_capture_mtu = pdctx->get_ul_pcap_capture_mtu();
        ul_pcap_capture_buffer = std::unique_ptr<ULInputBuffer>(new ULInputBuffer(pdh, gDev, cell_id, MAX_PKTS_PER_PCAP_BUFFER * (ul_pcap_capture_mtu)));
        ul_pcap_capture_rxtimestamp_buffer = std::unique_ptr<ULInputBuffer>(new ULInputBuffer(pdh, gDev, cell_id, MAX_PKTS_PER_PCAP_BUFFER * sizeof(uint64_t)));
    }

    switch(pdctx->getFhProxy()->getBfwCPlaneChainingMode())
    {
        case BfwCplaneChainingMode::NO_CHAINING:
        {
            bfw_coeff_buffer_pinned = std::move(cuphy::buffer<uint8_t, cuphy::pinned_alloc>(pdctx->getFhProxy()->getBfwCoeffSize()));

            bfw_coeff_buffer_info.header = &bfw_coeff_buffer_header;
            bfw_coeff_buffer_info.dataH = bfw_coeff_buffer_pinned.addr();
            bfw_coeff_buffer_info.dataD = nullptr;
            break;
        }
        case BfwCplaneChainingMode::CPU_CHAINING:
        {
            bfw_coeff_buffer_pinned = std::move(cuphy::buffer<uint8_t, cuphy::pinned_alloc>(pdctx->getFhProxy()->getBfwCoeffSize()));
            pdctx->registerBufferToFh(bfw_coeff_buffer_pinned.addr(), pdctx->getFhProxy()->getBfwCoeffSize());
            bfw_coeff_buffer_info.header = &bfw_coeff_buffer_header;
            bfw_coeff_buffer_info.dataH = bfw_coeff_buffer_pinned.addr();
            bfw_coeff_buffer_info.dataD = nullptr;
            break;
        }
        case BfwCplaneChainingMode::GPU_CHAINING: // UL still uses CPU buffer need to keep for both
        {

            bfw_coeff_buffer_dev = std::move(cuphy::buffer<uint8_t, cuphy::device_alloc>(pdctx->getFhProxy()->getBfwCoeffSize()));
            mf.addGpuRegularSize(sizeof(uint8_t)*pdctx->getFhProxy()->getBfwCoeffSize());
            bfw_coeff_buffer_pinned = std::move(cuphy::buffer<uint8_t, cuphy::pinned_alloc>(pdctx->getFhProxy()->getBfwCoeffSize()));
            pdctx->registerBufferToFh(bfw_coeff_buffer_pinned.addr(), pdctx->getFhProxy()->getBfwCoeffSize());
            pdctx->registerBufferToFh(bfw_coeff_buffer_dev.addr(), pdctx->getFhProxy()->getBfwCoeffSize());
            bfw_coeff_buffer_info.header = &bfw_coeff_buffer_header;
            bfw_coeff_buffer_info.dataH = bfw_coeff_buffer_pinned.addr();
            bfw_coeff_buffer_info.dataD = bfw_coeff_buffer_dev.addr();
            break;
        }
    }


    TI_GENERIC_ADD("cudaMemsets");
    last_sem_idx_rx_h.reset(new dev_buf(1 * sizeof(uint32_t), gDev));
    mf.addGpuRegularSize(last_sem_idx_rx_h->size_alloc);
    CUDA_CHECK(cudaMemset((uint32_t*)last_sem_idx_rx_h->addr(), 0, sizeof(uint32_t)));
    last_sem_idx_order_h.reset(new dev_buf(1 * sizeof(uint32_t), gDev));
    mf.addGpuRegularSize(last_sem_idx_order_h->size_alloc);
    CUDA_CHECK(cudaMemset((uint32_t*)last_sem_idx_order_h->addr(), 0, sizeof(uint32_t)));

    ul_pcap_capture_buffer_index.reset(new dev_buf(1 * sizeof(uint32_t), gDev));
    mf.addGpuRegularSize(ul_pcap_capture_buffer_index->size_alloc);
    CUDA_CHECK(cudaMemset((uint32_t*)ul_pcap_capture_buffer_index->addr(), 0, sizeof(uint32_t)));

    last_sem_idx_srs_rx_h.reset(new dev_buf(1 * sizeof(uint32_t), gDev));
    mf.addGpuRegularSize(last_sem_idx_srs_rx_h->size_alloc);
    CUDA_CHECK(cudaMemset((uint32_t*)last_sem_idx_srs_rx_h->addr(), 0, sizeof(uint32_t)));
    last_sem_idx_srs_order_h.reset(new dev_buf(1 * sizeof(uint32_t), gDev));
    mf.addGpuRegularSize(last_sem_idx_srs_order_h->size_alloc);
    CUDA_CHECK(cudaMemset((uint32_t*)last_sem_idx_srs_order_h->addr(), 0, sizeof(uint32_t)));
    order_kernel_last_timeout_error_time.reset(new dev_buf(1 * sizeof(uint64_t), gDev));
    mf.addGpuRegularSize(order_kernel_last_timeout_error_time->size_alloc);
    CUDA_CHECK(cudaMemset((uint32_t*)order_kernel_last_timeout_error_time->addr(), 0, sizeof(uint64_t)));
    order_kernel_srs_last_timeout_error_time.reset(new dev_buf(1 * sizeof(uint64_t), gDev));
    mf.addGpuRegularSize(order_kernel_srs_last_timeout_error_time->size_alloc);
    CUDA_CHECK(cudaMemset((uint32_t*)order_kernel_srs_last_timeout_error_time->addr(), 0, sizeof(uint64_t)));
    
    next_slot_on_time_rx_packets.reset(new dev_buf(1 * sizeof(uint32_t), gDev));
    mf.addGpuRegularSize(next_slot_on_time_rx_packets->size_alloc);
    CUDA_CHECK(cudaMemset((uint32_t*)next_slot_on_time_rx_packets->addr(), 0, sizeof(uint32_t)));
    next_slot_early_rx_packets.reset(new dev_buf(1 * sizeof(uint32_t), gDev));
    mf.addGpuRegularSize(next_slot_early_rx_packets->size_alloc);
    CUDA_CHECK(cudaMemset((uint32_t*)next_slot_early_rx_packets->addr(), 0, sizeof(uint32_t)));
    next_slot_late_rx_packets.reset(new dev_buf(1 * sizeof(uint32_t), gDev));
    mf.addGpuRegularSize(next_slot_late_rx_packets->size_alloc);
    CUDA_CHECK(cudaMemset((uint32_t*)next_slot_late_rx_packets->addr(), 0, sizeof(uint32_t)));

    next_slot_on_time_rx_packets_srs.reset(new dev_buf(1 * sizeof(uint32_t), gDev));
    mf.addGpuRegularSize(next_slot_on_time_rx_packets_srs->size_alloc);
    CUDA_CHECK(cudaMemset((uint32_t*)next_slot_on_time_rx_packets_srs->addr(), 0, sizeof(uint32_t)));
    next_slot_early_rx_packets_srs.reset(new dev_buf(1 * sizeof(uint32_t), gDev));
    mf.addGpuRegularSize(next_slot_early_rx_packets_srs->size_alloc);
    CUDA_CHECK(cudaMemset((uint32_t*)next_slot_early_rx_packets_srs->addr(), 0, sizeof(uint32_t)));
    next_slot_late_rx_packets_srs.reset(new dev_buf(1 * sizeof(uint32_t), gDev));
    mf.addGpuRegularSize(next_slot_late_rx_packets_srs->size_alloc);
    CUDA_CHECK(cudaMemset((uint32_t*)next_slot_late_rx_packets_srs->addr(), 0, sizeof(uint32_t)));

    next_slot_rx_packets_count_srs.reset(new dev_buf(1 * sizeof(uint32_t), gDev));
    mf.addGpuRegularSize(next_slot_rx_packets_count_srs->size_alloc);
    CUDA_CHECK(cudaMemset((uint32_t*)next_slot_rx_packets_count_srs->addr(), 0, sizeof(uint32_t)));
    next_slot_rx_bytes_count_srs.reset(new dev_buf(1 * sizeof(uint32_t), gDev));
    mf.addGpuRegularSize(next_slot_rx_bytes_count_srs->size_alloc);
    CUDA_CHECK(cudaMemset((uint32_t*)next_slot_rx_bytes_count_srs->addr(), 0, sizeof(uint32_t)));

    next_slot_rx_packets_ts_srs.reset(new dev_buf(ORDER_KERNEL_MAX_PKTS_PER_OFDM_SYM*ORAN_SRS_SYMBOLS_X_SLOT*sizeof(uint64_t), gDev));
    mf.addGpuRegularSize(next_slot_rx_packets_ts_srs->size_alloc);
    CUDA_CHECK(cudaMemset((uint64_t*)next_slot_rx_packets_ts_srs->addr(), 0, ORDER_KERNEL_MAX_PKTS_PER_OFDM_SYM*ORAN_SRS_SYMBOLS_X_SLOT*sizeof(uint64_t)));
    
    next_slot_rx_packets_count_per_sym_srs.reset(new dev_buf(ORAN_SRS_SYMBOLS_X_SLOT*sizeof(uint32_t), gDev));
    mf.addGpuRegularSize(next_slot_rx_packets_count_per_sym_srs->size_alloc);
    CUDA_CHECK(cudaMemset((uint32_t*)next_slot_rx_packets_count_per_sym_srs->addr(), 0, ORAN_SRS_SYMBOLS_X_SLOT*sizeof(uint32_t)));

    next_slot_rx_packets_ts.reset(new dev_buf(ORDER_KERNEL_MAX_PKTS_PER_OFDM_SYM*ORAN_PUSCH_SYMBOLS_X_SLOT*sizeof(uint64_t), gDev));
    mf.addGpuRegularSize(next_slot_rx_packets_ts->size_alloc);
    CUDA_CHECK(cudaMemset((uint64_t*)next_slot_rx_packets_ts->addr(), 0, ORDER_KERNEL_MAX_PKTS_PER_OFDM_SYM*ORAN_PUSCH_SYMBOLS_X_SLOT*sizeof(uint64_t)));
    
    next_slot_rx_packets_count.reset(new dev_buf(ORAN_PUSCH_SYMBOLS_X_SLOT*sizeof(uint32_t), gDev));
    mf.addGpuRegularSize(next_slot_rx_packets_count->size_alloc);
    CUDA_CHECK(cudaMemset((uint32_t*)next_slot_rx_packets_count->addr(), 0, ORAN_PUSCH_SYMBOLS_X_SLOT*sizeof(uint32_t)));

    next_slot_rx_bytes_count.reset(new dev_buf(ORAN_PUSCH_SYMBOLS_X_SLOT*sizeof(uint32_t), gDev));
    mf.addGpuRegularSize(next_slot_rx_bytes_count->size_alloc);
    CUDA_CHECK(cudaMemset((uint32_t*)next_slot_rx_bytes_count->addr(), 0, ORAN_PUSCH_SYMBOLS_X_SLOT*sizeof(uint32_t)));

    next_slot_num_prb_ch1.reset(new dev_buf(1 * sizeof(uint32_t), gDev));
    mf.addGpuRegularSize(next_slot_num_prb_ch1->size_alloc);
    CUDA_CHECK(cudaMemset((uint32_t*)next_slot_num_prb_ch1->addr(), 0, sizeof(uint32_t)));

    next_slot_num_prb_ch2.reset(new dev_buf(1 * sizeof(uint32_t), gDev));
    mf.addGpuRegularSize(next_slot_num_prb_ch2->size_alloc);
    CUDA_CHECK(cudaMemset((uint32_t*)next_slot_num_prb_ch2->addr(), 0, sizeof(uint32_t)));

    next_slot_num_prb_ch3.reset(new dev_buf(1 * sizeof(uint32_t), gDev));
    mf.addGpuRegularSize(next_slot_num_prb_ch3->size_alloc);
    CUDA_CHECK(cudaMemset((uint32_t*)next_slot_num_prb_ch3->addr(), 0, sizeof(uint32_t)));

    uint32_t max_K_per_CB = CUPHY_LDPC_BG1_INFO_NODES * CUPHY_LDPC_MAX_LIFTING_SIZE;
    uint32_t tb_bytes = PDSCH_MAX_UES_PER_CELL* MAX_N_CBS_PER_TB_SUPPORTED * div_round_up<uint32_t>(max_K_per_CB, 8);
    
    TI_GENERIC_ADD("cudaMallocs");
    pdctx->getPdschMpsCtx()->setCtx();
    for(int i = 0; i < PDSCH_MAX_GPU_BUFFS ; i++)
    {
       CUDA_CHECK_PHYDRIVER(cudaMalloc(&pdsch_tb_buffer[i], tb_bytes));   
       mf.addGpuRegularSize(tb_bytes);
    }

    TI_GENERIC_ADD("Print memory footprint");
#if PRINT_MEMORY_FOOTPRINT_IN_CTOR
    MemFoot mf_acc;
    mf_acc.init((phydriver_handle)pdh, std::string("AccumulatorCellPhy"), sizeof(MemFoot));
    mf_acc.reset();
    //FIXME ulbuf_st2 was missing from here and destructor; unclear why. Added it
    // dl_buf_list.size() not available here, only UL buffer are; for DL added code to Cell:setIOBuf()

    if(ulbuf_st1_list.size() > 0)
    {
        ulbuf_st1_list[0]->mf.printMemoryFootprint();
        for(int i = 0; i < ulbuf_st1_list.size(); i++)
        {
            mf_acc.addMF(ulbuf_st1_list[i]->mf);
        }
        mf_acc.printMemoryFootprint();
        pdctx->wip_accum_mf.addMF(mf_acc);
        //pdctx->wip_accum_mf.printMemoryFootprint();
        mf_acc.reset();
    }

    if(ulbuf_st3_list.size() > 0)
    {
        ulbuf_st3_list[0]->mf.printMemoryFootprint();
        for(int i = 0; i < ulbuf_st3_list.size(); i++)
        {
            mf_acc.addMF(ulbuf_st3_list[i]->mf);
        }
        mf_acc.printMemoryFootprint();
        pdctx->wip_accum_mf.addMF(mf_acc);
        //pdctx->wip_accum_mf.printMemoryFootprint();
        mf_acc.reset();
    }

    if(ulbuf_st2_list.size() > 0)
    {
        ulbuf_st2_list[0]->mf.printMemoryFootprint();
        for(int i = 0; i < ulbuf_st2_list.size(); i++)
        {
            mf_acc.addMF(ulbuf_st2_list[i]->mf);
        }
        mf_acc.printMemoryFootprint();
        pdctx->wip_accum_mf.addMF(mf_acc);
        //pdctx->wip_accum_mf.printMemoryFootprint();
        mf_acc.reset();
    }

#endif

    TI_GENERIC_ADD("End Task");
    TI_GENERIC_ALL_NVLOGI(TAG_STARTUP_TIMES);

}

Cell::~Cell()
{
    clearIOBuffers();
}

void Cell::clearIOBuffers()
{
    MemFoot mf_acc;
    mf_acc.init((phydriver_handle)pdh, std::string("AccumulatorCellPhy"), sizeof(MemFoot));
    mf_acc.reset();

    ////////////////////////////////////////////////////////////////////////////////////////
    //// I/O Buffers
    ////////////////////////////////////////////////////////////////////////////////////////
    #if !PRINT_MEMORY_FOOTPRINT_IN_CTOR
    if(dlbuf_list.size() > 0)
    {
        dlbuf_list[0]->mf.printMemoryFootprint();
        for(int i = 0; i < dlbuf_list.size(); i++)
        {
            mf_acc.addMF(dlbuf_list[i]->mf);
        }
        mf_acc.printMemoryFootprint();
        mf_acc.reset();
    }
    #endif
    dlbuf_list.clear();
    // NVSLOGD(TAG) << "DL Output buffers destroyed";

    #if !PRINT_MEMORY_FOOTPRINT_IN_CTOR
    if(ulbuf_st1_list.size() > 0)
    {
        ulbuf_st1_list[0]->mf.printMemoryFootprint();
        for(int i = 0; i < ulbuf_st1_list.size(); i++)
        {
            mf_acc.addMF(ulbuf_st1_list[i]->mf);
        }
        mf_acc.printMemoryFootprint();
        mf_acc.reset();
    }
    #endif
    ulbuf_st1_list.clear();
    // NVSLOGD(TAG) << "UL Input buffers ST1 destroyed";

    #if !PRINT_MEMORY_FOOTPRINT_IN_CTOR
    if(ulbuf_st3_list.size() > 0)
    {
        ulbuf_st3_list[0]->mf.printMemoryFootprint();
        for(int i = 0; i < ulbuf_st3_list.size(); i++)
        {
            mf_acc.addMF(ulbuf_st3_list[i]->mf);
        }
        mf_acc.printMemoryFootprint();
        mf_acc.reset();
    }
    #endif
    ulbuf_st3_list.clear();
    // NVSLOGD(TAG) << "UL Input buffers ST3 destroyed";

    //added missing ulbuf_st2
    #if !PRINT_MEMORY_FOOTPRINT_IN_CTOR
    if(ulbuf_st2_list.size() > 0)
    {
        ulbuf_st2_list[0]->mf.printMemoryFootprint();
        for(int i = 0; i < ulbuf_st2_list.size(); i++)
        {
            mf_acc.addMF(ulbuf_st2_list[i]->mf);
        }
        mf_acc.printMemoryFootprint();
        mf_acc.reset();
    }
    #endif
    ulbuf_st2_list.clear();
    // NVSLOGD(TAG) << "UL Input buffers ST2 destroyed";
}

void Cell::clearULBuffers()
{
    ulbuf_st1_list.clear();
    // NVSLOGD(TAG) << "UL Input buffers ST1 destroyed";
    ulbuf_st2_list.clear();
    // NVSLOGD(TAG) << "UL Input buffers ST2 destroyed";
    ulbuf_st3_list.clear();
    // NVSLOGD(TAG) << "UL Input buffers ST3 destroyed";
}

void Cell::resetSemIndices()
{
    cudaMemset((uint32_t*)last_sem_idx_rx_h->addr(), 0, sizeof(uint32_t));
    cudaMemset((uint32_t*)last_sem_idx_order_h->addr(), 0, sizeof(uint32_t));
}

void Cell::resetSrsSemIndices()
{
    cudaMemset((uint32_t*)last_sem_idx_srs_rx_h->addr(), 0, sizeof(uint32_t));
    cudaMemset((uint32_t*)last_sem_idx_srs_order_h->addr(), 0, sizeof(uint32_t));
}

void Cell::printCelleAxCIds()
{
        NVLOGC_FMT(TAG, "\tFlow list SSB/PBCH: ");
        for(auto eAxC_id : eAxC_ids[slot_command_api::channel_type::PBCH])
            NVLOGC_FMT(TAG, "\t\t{}", eAxC_id);
        NVLOGC_FMT(TAG, "\tFlow list PDCCH_DL: ");
        for(auto eAxC_id : eAxC_ids[slot_command_api::channel_type::PDCCH_DL])
            NVLOGC_FMT(TAG, "\t\t{}", eAxC_id);
        NVLOGC_FMT(TAG, "\tFlow list PDCCH_UL: ");
        for(auto eAxC_id : eAxC_ids[slot_command_api::channel_type::PDCCH_UL])
            NVLOGC_FMT(TAG, "\t\t{}", eAxC_id);
        NVLOGC_FMT(TAG, "\tFlow list PDSCH: ");
        for(auto eAxC_id : eAxC_ids[slot_command_api::channel_type::PDSCH])
            NVLOGC_FMT(TAG, "\t\t{}", eAxC_id);
        NVLOGC_FMT(TAG, "\tFlow list PDSCH_CSIRS: ");
        for(auto eAxC_id : eAxC_ids[slot_command_api::channel_type::PDSCH_CSIRS])
            NVLOGC_FMT(TAG, "\t\t{}", eAxC_id);
        NVLOGC_FMT(TAG, "\tFlow list CSIRS: ");
        for(auto eAxC_id : eAxC_ids[slot_command_api::channel_type::CSI_RS])
            NVLOGC_FMT(TAG, "\t\t{}", eAxC_id);
        NVLOGC_FMT(TAG, "\tFlow list PUSCH: ");
        for(auto eAxC_id : eAxC_ids[slot_command_api::channel_type::PUSCH])
            NVLOGC_FMT(TAG, "\t\t{}", eAxC_id);
        NVLOGC_FMT(TAG, "\tFlow list PUCCH: ");
        for(auto eAxC_id : eAxC_ids[slot_command_api::channel_type::PUCCH])
            NVLOGC_FMT(TAG, "\t\t{}", eAxC_id);
        NVLOGC_FMT(TAG, "\tFlow list SRS: ");
        for(auto eAxC_id : eAxC_ids[slot_command_api::channel_type::SRS])
            NVLOGC_FMT(TAG, "\t\t{}", eAxC_id);
        NVLOGC_FMT(TAG, "\tFlow list PRACH: ");
        for(auto eAxC_id : eAxC_ids[slot_command_api::channel_type::PRACH])
            NVLOGC_FMT(TAG, "\t\t{}", eAxC_id);
}

int Cell::updateeAxCIds(std::unordered_map<int, std::vector<uint16_t>>& eaxcids_ch_map)
{
    for(auto& [ch, eaxcids] : eaxcids_ch_map)
    {
        eAxC_ids[static_cast<slot_command_api::channel_type>(ch)] = eaxcids;
    }
    NVLOGC_FMT(TAG, "\t Cell {} eAxCIds updated: ", mplane_id);
    printCelleAxCIds();

    for(auto& nic : fh_proxy->getNicList())
    {
        fh_proxy->removePeer(nic2peer_map[nic]);
    }
    clearULBuffers();

    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    std::vector<uint16_t>& eAxC_list_uplink=geteAxCIdsUl();
    eAxC_list_uplink.clear();

    for (int channel = slot_command_api::channel_type::PUSCH; channel < slot_command_api::channel_type::SRS; channel++)
    {
        for (auto eAxC : eAxC_ids[channel])
        {
            auto it = std::find(std::begin(eAxC_list_uplink), std::end(eAxC_list_uplink), (uint16_t) eAxC);
            if(it!=std::end(eAxC_list_uplink))
                continue;
            eAxC_list_uplink.push_back(eAxC);
        }
    }

    for(auto& nic : fh_proxy->getNicList())
    {
        if(fh_proxy->registerPeer(mplane_id, nic2peer_map[nic], src_eth_addr, dst_eth_addr, vlan_tci, txq_count_uplane, dl_comp_meth, dl_bit_width, gDev->getId(), nic, &nic2doca_rxq_info_map[nic], &nic2doca_rxq_info_srs_map[nic], eAxC_list_uplink, (std::vector<uint16_t>&)geteAxCIdsSrs(), (std::vector<uint16_t>&)geteAxCIdsPdsch()))
            PHYDRIVER_THROW_EXCEPTIONS(-1, "Add Peer for cell returned error");

        for(int channel = slot_command_api::channel_type::PDSCH_CSIRS; channel < slot_command_api::channel_type::CHANNEL_MAX; channel++)
        {
            // dMIMO: SRS may be sounded by multiple PEs (destinations). Register each
            // destination's SRS eAxC with that destination's dst MAC so the SRS C-plane
            // reaches the right PE over this single (primary) peer's TX queue. The SRS
            // U-plane is caught back into the primary SRS RX queue by the wildcard src
            // MAC RX rule, so no extra peer/RX queue is needed.
            if(!mplane_destinations_.empty() && channel == slot_command_api::channel_type::SRS)
            {
                for(const auto& dest : mplane_destinations_)
                {
                    for(auto eAxC : dest.eAxC_ids[slot_command_api::channel_type::SRS])
                    {
                        if(fh_proxy->registerFlow(nic2peer_map[nic], eAxC, vlan_tci, slot_command_api::channel_type::SRS, &dest.dst_eth_addr))
                            PHYDRIVER_THROW_EXCEPTIONS(-1, "Add flow returned error");
                    }
                }
                continue;
            }
            for(auto eAxC : eAxC_ids[channel])
            {
                if(fh_proxy->registerFlow(nic2peer_map[nic], eAxC, vlan_tci, static_cast<slot_command_api::channel_type>(channel)))
                    PHYDRIVER_THROW_EXCEPTIONS(-1, "Add flowreturned error");
            }
        }
    }

    int section_type_1_ant = std::max(std::max(geteAxCNumPusch(), geteAxCNumPucch()), (size_t)nMaxRxAnt);
    int section_type_3_ant = std::max(geteAxCNumPrach(), (size_t)nMaxRxAnt);

    uint8_t ul_input_buffer_num_per_cell = pdctx->getUlInputBufferPerCell();
    uint8_t ul_input_buffer_num_per_cell_srs = pdctx->getUlInputBufferPerCellSrs();
    for(int i = 0; i < ul_input_buffer_num_per_cell; i++)
    {
        ulbuf_st1_list.push_back(std::unique_ptr<ULInputBuffer>(new ULInputBuffer(pdh, gDev, cell_id, UL_ST1_AP_BUF_SIZE * section_type_1_ant)));
    }

    for(int i = 0; i < ul_input_buffer_num_per_cell; i++)
    {
        ulbuf_st3_list.push_back(std::unique_ptr<ULInputBuffer>(new ULInputBuffer(pdh, gDev, cell_id, UL_ST3_AP_BUF_SIZE * section_type_3_ant)));
    }

    for(int i = 0; i < ul_input_buffer_num_per_cell_srs; i++)
    {
        ulbuf_st2_list.push_back(std::unique_ptr<ULInputBuffer>(new ULInputBuffer(pdh, gDev, cell_id, UL_ST2_AP_BUF_SIZE * geteAxCNumSrs())));
    }
    ulbuf_st1_index     = 0;
    ulbuf_st2_index     = 0;
    ulbuf_st3_index     = 0;

    resetSemIndices();
    resetSrsSemIndices();

#if PRINT_MEMORY_FOOTPRINT_IN_CTOR
    MemFoot mf_acc;
    mf_acc.init((phydriver_handle)pdh, std::string("AccumulatorCellPhy"), sizeof(MemFoot));
    mf_acc.reset();
    //FIXME ulbuf_st2 was missing from here and destructor; unclear why. Added it
    // dl_buf_list.size() not available here, only UL buffer are; for DL added code to Cell:setIOBuf()

    if(ulbuf_st1_list.size() > 0)
    {
        ulbuf_st1_list[0]->mf.printMemoryFootprint();
        for(int i = 0; i < ulbuf_st1_list.size(); i++)
        {
            mf_acc.addMF(ulbuf_st1_list[i]->mf);
        }
        mf_acc.printMemoryFootprint();
        pdctx->wip_accum_mf.addMF(mf_acc);
        //pdctx->wip_accum_mf.printMemoryFootprint();
        mf_acc.reset();
    }

    if(ulbuf_st3_list.size() > 0)
    {
        ulbuf_st3_list[0]->mf.printMemoryFootprint();
        for(int i = 0; i < ulbuf_st3_list.size(); i++)
        {
            mf_acc.addMF(ulbuf_st3_list[i]->mf);
        }
        mf_acc.printMemoryFootprint();
        pdctx->wip_accum_mf.addMF(mf_acc);
        //pdctx->wip_accum_mf.printMemoryFootprint();
        mf_acc.reset();
    }

    if(ulbuf_st2_list.size() > 0)
    {
        ulbuf_st2_list[0]->mf.printMemoryFootprint();
        for(int i = 0; i < ulbuf_st2_list.size(); i++)
        {
            mf_acc.addMF(ulbuf_st2_list[i]->mf);
        }
        mf_acc.printMemoryFootprint();
        pdctx->wip_accum_mf.addMF(mf_acc);
        //pdctx->wip_accum_mf.printMemoryFootprint();
        mf_acc.reset();
    }
#endif
    return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////////
///// PHY
//////////////////////////////////////////////////////////////////////////////////////////////
int Cell::setPhyStatic(struct cell_phy_info& c_phy)
{
    float numerator, denominator;
    float sqrt_fs0, fs;
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(this->getPhyDriverHandler()).get();

    if(c_phy.mplane_id != mplane_id)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Cell mplane {} doesn't correspond to input mplane_id {}", mplane_id, c_phy.mplane_id);
        return -1;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    //// cuPHY Static Info
    ///////////////////////////////////////////////////////////////////////////////////////////////
    name           = std::move(c_phy.name);
    tti            = c_phy.tti;
    slot_ahead     = c_phy.slot_ahead;
    if(pdctx->get_slot_advance()!=slot_ahead) //Update slot advance param under Phy driver context if required
    {
        pdctx->set_slot_advance(slot_ahead);
    }
    phy_stat       = c_phy.phy_stat;
    dl_grid_size   = phy_stat.nPrbDlBwp;
    ul_grid_size   = phy_stat.nPrbUlBwp;
    //prach_config_list.reserve(1);
    //prach_config_list = c_phy.prach_configs;
    prachCellStatParams = c_phy.prachStatParams;
    prachOccaStatParamList.reserve(PRACH_MAX_OCCASIONS);
    prachOccaStatParamList = c_phy.prach_configs;

    fh_proxy->update_peer_max_num_prbs_per_symbol(getPeerId(),std::max(phy_stat.nPrbDlBwp,phy_stat.nPrbUlBwp));

    switch(dl_comp_meth)
    {
        case aerial_fh::UserDataCompressionMethod::NO_COMPRESSION:
            beta_dl = float(1UL << (dl_bit_width - 5)); // Set beta to have 3 integer and 1 sign bit
            break;
        case aerial_fh::UserDataCompressionMethod::BLOCK_FLOATING_POINT:
        default:
            sqrt_fs0       = pow(2., dl_bit_width - 1) * pow(2., pow(2., exponent_dl) - 1);
            fs             = sqrt_fs0 * sqrt_fs0 * pow(2., -1 * fs_offset_dl);
            numerator      = fs * pow(10., ref_dl / 10.);
            denominator    = 12 * phy_stat.nPrbDlBwp;
            beta_dl        = sqrt(numerator / denominator);

            if(pdctx->fixBetaDl())
            {
                if(dl_bit_width == BFP_COMPRESSION_9_BITS)
                {
                    beta_dl = 65536;
                }
                else if(dl_bit_width == BFP_COMPRESSION_14_BITS)
                {
                    beta_dl = 2097152;
                }
            }
    }

    sqrt_fs0       = pow(2., ul_bit_width - 1) * pow(2., pow(2., exponent_ul) - 1);
    fs             = sqrt_fs0 * sqrt_fs0 * pow(2., -1 * fs_offset_ul);
    beta_ul        = max_amp_ul / sqrt(fs);
    if(ul_comp_meth == aerial_fh::UserDataCompressionMethod::NO_COMPRESSION)
    {
        beta_ul = 1.0/float(1UL << (ul_bit_width - 5)); // Set beta to have 3 integer and 1 sign bit
    }

    return 0;
}

int Cell::setGpuItems()
{
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(this->getPhyDriverHandler()).get();

    if(pdctx->getUlCtx()->getGpuDevice()->getId() != gDev->getId())
        PHYDRIVER_THROW_EXCEPTIONS(errno, "MPS device is different from Cell device. Multi-GPU is not supported yet");
    pdctx->setUlCtx();
    CUDA_CHECK(cudaStreamCreateWithPriority(&stream_ul, cudaStreamNonBlocking, -3));
    CUDA_CHECK(cudaStreamCreateWithPriority(&stream_order, cudaStreamNonBlocking, -5));

    launch_kernel_warmup(stream_order);
    launch_kernel_order(stream_order, 1, 0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, 0, 0, 0, 0, nullptr, 0, 0, 0, 0, 0);
    launch_kernel_order(stream_order, 1, 0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, 0, 0, 0, 0, nullptr, 0, 0, 0, 0, 0);
    gDev->synchronizeStream(stream_order);

    if(pdctx->getDlCtx()->getGpuDevice()->getId() != gDev->getId())
        PHYDRIVER_THROW_EXCEPTIONS(errno, "MPS device is different from Cell device. Multi-GPU is not supported yet");
    pdctx->setDlCtx();
    CUDA_CHECK(cudaStreamCreateWithPriority(&stream_dl, cudaStreamNonBlocking, -4));
    launch_kernel_warmup(stream_dl);
    
    cudaDeviceSynchronize();

    return 0;
}

int Cell::setPhyObj()
{
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(this->getPhyDriverHandler()).get();

    return 0;
}

int Cell::setIOBuf()
{
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(this->getPhyDriverHandler()).get();
    pdctx->setDlCtx();

    for(int i = 0; i < DL_OUTPUT_BUFFER_NUM_PER_CELL; i++)
    {
        dlbuf_list.push_back(std::unique_ptr<DLOutputBuffer>(new DLOutputBuffer(pdh, gDev, cell_id)));
    }

    //start();
#if PRINT_MEMORY_FOOTPRINT_IN_CTOR
    MemFoot mf_acc;
    mf_acc.init((phydriver_handle)pdh, std::string("AccumulatorCellPhy"), sizeof(MemFoot));
    mf_acc.reset();
    // DL I/O Buffers
   if(dlbuf_list.size() > 0)
    {
        dlbuf_list[0]->mf.printMemoryFootprint();
        for(int i = 0; i < dlbuf_list.size(); i++)
        {
            mf_acc.addMF(dlbuf_list[i]->mf);
        }
        mf_acc.printMemoryFootprint();
        pdctx->wip_accum_mf.addMF(mf_acc);
        //pdctx->wip_accum_mf.printMemoryFootprint();
        mf_acc.reset();
    }
#endif

    return 0;
}

DLOutputBuffer* Cell::getNextDlBuffer()
{
    DLOutputBuffer* dlbuf = nullptr;
    int             cnt   = 0;

    mlock_dlbuf.lock();
    while(dlbuf_list[dlbuf_index]->reserve() != 0 && cnt < DL_OUTPUT_BUFFER_NUM_PER_CELL)
    {
        dlbuf_index = (dlbuf_index + 1) % DL_OUTPUT_BUFFER_NUM_PER_CELL;
        cnt++;
    }

    if(cnt < DL_OUTPUT_BUFFER_NUM_PER_CELL)
    {
        dlbuf = dlbuf_list[dlbuf_index].get();
        //Alreday set to the next one
        dlbuf_index = (dlbuf_index + 1) % DL_OUTPUT_BUFFER_NUM_PER_CELL;
    }

    mlock_dlbuf.unlock();

    return dlbuf;
}

ULInputBuffer* Cell::getNextUlBufferST1()
{
    ULInputBuffer* ulbuf = nullptr;
    int             cnt   = 0;

    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(this->getPhyDriverHandler()).get();
    uint8_t ul_input_buffer_num_per_cell = pdctx->getUlInputBufferPerCell();
    mlock_ulbuf_st1.lock();
    while(ulbuf_st1_list[ulbuf_st1_index]->reserve() != 0 && cnt < ul_input_buffer_num_per_cell)
    {
        ulbuf_st1_index = (ulbuf_st1_index + 1) % ul_input_buffer_num_per_cell;
        cnt++;
    }

    if(cnt < ul_input_buffer_num_per_cell)
    {
        ulbuf = ulbuf_st1_list[ulbuf_st1_index].get();
        //Alreday set to the next one
        ulbuf_st1_index = (ulbuf_st1_index + 1) % ul_input_buffer_num_per_cell;
    }

    mlock_ulbuf_st1.unlock();

    return ulbuf;
}

ULInputBuffer* Cell::getNextUlBufferST2()
{
    ULInputBuffer* ulbuf = nullptr;
    int             cnt   = 0;

    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(this->getPhyDriverHandler()).get();
    uint8_t ul_input_buffer_num_per_cell_srs = pdctx->getUlInputBufferPerCellSrs();

    mlock_ulbuf_st2.lock();
    while(ulbuf_st2_list[ulbuf_st2_index]->reserve() != 0 && cnt < ul_input_buffer_num_per_cell_srs)
    {
        ulbuf_st2_index = (ulbuf_st2_index + 1) % ul_input_buffer_num_per_cell_srs;
        cnt++;
    }

    if(cnt < ul_input_buffer_num_per_cell_srs)
    {
        ulbuf = ulbuf_st2_list[ulbuf_st2_index].get();
        //Alreday set to the next one
        ulbuf_st2_index = (ulbuf_st2_index + 1) % ul_input_buffer_num_per_cell_srs;
    }

    mlock_ulbuf_st2.unlock();

    return ulbuf;
}

ULInputBuffer* Cell::getNextUlBufferST3()
{
    ULInputBuffer* ulbuf = nullptr;
    int             cnt   = 0;

    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(this->getPhyDriverHandler()).get();
    uint8_t ul_input_buffer_num_per_cell = pdctx->getUlInputBufferPerCell();
    mlock_ulbuf_st3.lock();

    while(ulbuf_st3_list[ulbuf_st3_index]->reserve() != 0 && cnt < ul_input_buffer_num_per_cell)
    {
        ulbuf_st3_index = (ulbuf_st3_index + 1) % ul_input_buffer_num_per_cell;
        cnt++;
    }

    if(cnt < ul_input_buffer_num_per_cell)
    {
        ulbuf = ulbuf_st3_list[ulbuf_st3_index].get();
        //Alreday set to the next one
        ulbuf_st3_index = (ulbuf_st3_index + 1) % ul_input_buffer_num_per_cell;
    }

    mlock_ulbuf_st3.unlock();

    return ulbuf;
}

ULInputBuffer* Cell::getUlBufferPcap()
{
    return ul_pcap_capture_buffer.get();
}

ULInputBuffer* Cell::getUlBufferPcapTs()
{
    return ul_pcap_capture_rxtimestamp_buffer.get();
}

cuphyPrachCellStatPrms_t* Cell::getPrachCellStatConfig() {
    return &prachCellStatParams;
}

cuphyPrachOccaStatPrms_t* Cell::getPrachOccaStatConfig() {
    return &prachOccaStatParamList[0];
}

std::vector<cuphyPrachOccaStatPrms_t>* Cell::getPrachOccaStatVec(){
    return &prachOccaStatParamList;
}

std::size_t Cell::getPrachOccaSize() const {
    return prachOccaStatParamList.size();
}

void Cell::setPrachOccaPrmStatIdx(uint16_t index)
{
    prachOccaPrmStatIdx = index;
}

uint16_t Cell::getPrachOccaPrmStatIdx()
{
    return prachOccaPrmStatIdx;
}

const char* Cell::getTvPuschH5File(void)
{
    if(tv_pusch_h5.empty()) return nullptr;
    else return tv_pusch_h5.c_str();
}

const char* Cell::getTvSrsH5File(void)
{
    if(tv_srs_h5.empty()) return nullptr;
    else return tv_srs_h5.c_str();
}

cuphyCellStatPrm_t* Cell::getPhyStatic(void)
{
    return &phy_stat;
}
//////////////////////////////////////////////////////////////////////////////////////////////
///// Network
//////////////////////////////////////////////////////////////////////////////////////////////

/*
 * TX QUEUE
 */
void Cell::lockTxQueue()
{
    tx_queue_lock.lock();
    // tx_queue_locked = true;
}

void Cell::unlockTxQueue()
{
    // tx_queue_locked = false;
    tx_queue_lock.unlock();
}

//////////////////////////////////////////////////////////////////////////////////////////////
///// Cell info
//////////////////////////////////////////////////////////////////////////////////////////////

phydriver_handle Cell::getPhyDriverHandler(void) const
{
    return pdh;
}

cell_id_t Cell::getId() const
{
    return cell_id;
}

uint16_t Cell::getPhyId() const
{
    return phy_stat.phyCellId;
}

uint32_t Cell::getIdx() const
{
    return idx;
}

uint16_t Cell::getMplaneId() const
{
    return mplane_id;
}


uint16_t Cell::getRxAnt() const
{
    return phy_stat.nRxAnt;
}

uint16_t Cell::getRxAntSrs() const
{
    return phy_stat.nRxAntSrs;
}

uint16_t Cell::getTxAnt() const
{
    return phy_stat.nTxAnt;
}

uint16_t Cell::getPrbUlBwp() const
{
    return phy_stat.nPrbUlBwp;
}

uint16_t Cell::getPrbDlBwp() const
{
    return phy_stat.nPrbDlBwp;
}

uint8_t Cell::getMu() const
{
    return phy_stat.mu;
}

GpuDevice* Cell::getGpuDevice() const
{
    return gDev;
}

cudaStream_t Cell::getUlChannelStream()
{
    return stream_ul;
}

cudaStream_t Cell::getUlOrderStream()
{
    return stream_order;
}

cudaStream_t Cell::getDlStream()
{
    return stream_dl;
}

std::array<uint8_t, 6> Cell::getSrcEthAddr() const
{
    return src_eth_addr;
}

std::array<uint8_t, 6> Cell::getDstEthAddr() const
{
    return dst_eth_addr;
}

uint16_t Cell::getVlanTci() const
{
    return vlan_tci;
}

uint8_t Cell::getUplaneTxqCount() const
{
    return txq_size_uplane;
}

peer_id_t Cell::getPeerId()
{
    return nic2peer_map[nic_name];;
}

void Cell::updateMetric(CellMetric metric, uint64_t value)
{
    metrics.update(metric, value);
}

const std::vector<uint16_t>& Cell::geteAxCIdsPdsch() const
{
    return eAxC_ids[slot_command_api::channel_type::PDSCH];
}

const std::vector<uint16_t>& Cell::geteAxCIdsPusch() const
{
    return eAxC_ids[slot_command_api::channel_type::PUSCH];
}

const std::vector<uint16_t>& Cell::geteAxCIdsPucch() const
{
    return eAxC_ids[slot_command_api::channel_type::PUCCH];
}

const std::vector<uint16_t>& Cell::geteAxCIdsPrach() const
{
    return eAxC_ids[slot_command_api::channel_type::PRACH];
}

const std::vector<uint16_t>& Cell::geteAxCIdsSrs() const
{
    return eAxC_ids[slot_command_api::channel_type::SRS];
}

std::vector<uint16_t>& Cell::geteAxCIdsUl()
{
    return eAxC_list_ul;
}

size_t Cell::geteAxCNumPdsch() const
{
    return geteAxCIdsPdsch().size();
}

size_t Cell::geteAxCNumPusch() const
{
    return geteAxCIdsPusch().size();
}

size_t Cell::geteAxCNumPucch() const
{
    return geteAxCIdsPucch().size();
}

size_t Cell::geteAxCNumSrs() const
{
    return geteAxCIdsSrs().size();
}

size_t Cell::geteAxCNumPrach() const
{
    return geteAxCIdsPrach().size();
}

int Cell::getSlotAhead() const
{
    return slot_ahead;
}

uint32_t Cell::getTcpAdvDlNs() const
{
    return tcp_adv_dl_ns;
}

uint32_t Cell::getT1aMaxUpNs() const
{
    return t1a_max_up_ns;
}

uint32_t Cell::getMaxFhLen() const
{
    return fh_len_range;
}

uint32_t Cell::getT1aMaxCpUlNs() const
{
    return t1a_max_cp_ul_ns;
}

uint32_t Cell::getT1aMinCpUlNs() const
{
    return t1a_min_cp_ul_ns;
}

uint32_t Cell::getTa4MinNs() const
{
    return ta4_min_ns;
}

uint32_t Cell::getTa4MaxNs() const
{
    return ta4_max_ns;
}

uint32_t Cell::getTa4MinNsSrs() const
{
    return ta4_min_ns_srs;
}

uint32_t Cell::getTa4MaxNsSrs() const
{
    return ta4_max_ns_srs;
}

uint32_t Cell::getT1aMinCpDlNs() const
{
    return t1a_min_cp_dl_ns;
}

uint32_t Cell::getT1aMaxCpDlNs() const
{
    return t1a_max_cp_dl_ns;
}

uint32_t Cell::getUlUplaneTxOffsetNs() const
{
    return ul_u_plane_tx_offset_ns;
}

uint32_t Cell::getPuschPrbStride() const
{
    return pusch_prb_stride;
}

void Cell::setPuschPrbStride(uint32_t pusch_prb_stride)
{
    this->pusch_prb_stride = pusch_prb_stride;
}

uint32_t Cell::getPrachPrbStride() const
{
    return prach_prb_stride;
}

void Cell::setPrachPrbStride(uint32_t prach_prb_stride)
{
    this->prach_prb_stride = prach_prb_stride;
}

int Cell::getSrsDynPrmIndex()
{
    return srs_cell_dyn_index;
}

void Cell::setSrsDynPrmIndex(int index)
{
    srs_cell_dyn_index=index;
}

uint16_t Cell::getDLGridSize() const
{
    return dl_grid_size;
}

void Cell::setDLGridSize(uint16_t dl_grid_size)
{
    this->dl_grid_size = dl_grid_size;
}

uint16_t Cell::getULGridSize() const
{
    return ul_grid_size;
}

void Cell::setULGridSize(uint16_t ul_grid_size)
{
    this->ul_grid_size = ul_grid_size;
}

uint32_t Cell::getSrsPrbStride() const
{
    return srs_prb_stride;
}

uint16_t Cell::getSection3TimeOffset() const
{
    return section_3_time_offset;
}

void Cell::setSection3TimeOffset(uint16_t section_3_time_offset)
{
    this->section_3_time_offset = section_3_time_offset;
}

uint8_t Cell::getPuschLdpcMaxNumItrAlgoType() const
{
    return pusch_ldpc_max_num_itr_algo_type;
}

uint8_t Cell::getFixedMaxNumLdpcItrs() const
{
    return pusch_fixed_max_num_ldpc_itrs;
}

uint8_t Cell::getPuschLdpcEarlyTermination() const
{
    return pusch_ldpc_early_termination;
}

uint8_t Cell::getPuschLdpcAlgoIndex() const
{
    return pusch_ldpc_algo_index;
}

uint8_t Cell::getPuschLdpcFlags() const
{
    return pusch_ldpc_flags;
}

uint8_t Cell::getPuschLdpcUseHalf() const
{
    return pusch_ldpc_use_half;
}

uint16_t Cell::getPuschnMaxPrb() const
{
    return pusch_nMaxPrb;
}

uint16_t Cell::getPuschnMaxRx() const
{
    return pusch_nMaxRx;
}

ru_type Cell::getRUType() const
{
    return ru;
}

void Cell::setRUType(ru_type ru)
{
    this->ru = ru;
    //TODO: propagte the change
}

uint16_t Cell::getDLCompMeth() const
{
    return static_cast<uint16_t>(dl_comp_meth);
}

uint8_t Cell::getDLBitWidth() const
{
    return dl_bit_width;
}

void Cell::setDLIQDataFmt(UserDataCompressionMethod comp_meth, uint8_t bit_width)
{
    if(dl_comp_meth == comp_meth && dl_bit_width == bit_width)
    {
        NVLOGC_FMT(TAG, "DL IQ data format not changed, return..", __func__, nic_name);
        return;
    }
    dl_comp_meth = comp_meth;
    dl_bit_width = bit_width;
    updateCellConfig(comp_meth, bit_width);
}

uint16_t Cell::getCompressionPrbSize() const
{
    switch(dl_bit_width)
    {
        case BFP_NO_COMPRESSION:
            return static_cast<uint16_t>(BFP_NO_COMPRESSION_BITWIDTH);
        case BFP_COMPRESSION_9_BITS:
            return static_cast<uint16_t>(BITWIDTH_9_BITS);
        case BFP_COMPRESSION_14_BITS:
            return static_cast<uint16_t>(BITWIDTH_14_BITS);
        default:
            return 0;
    };

    return 0;
}

uint16_t Cell::getULCompMeth() const
{
    return static_cast<uint16_t>(ul_comp_meth);
}

uint8_t Cell::getULBitWidth() const
{
    return ul_bit_width;
}

void Cell::setULIQDataFmt(UserDataCompressionMethod comp_meth, uint8_t bit_width)
{
    if(ul_comp_meth == comp_meth && ul_bit_width == bit_width)
    {
        NVLOGC_FMT(TAG, "UL IQ data format not changed, return..", __func__, nic_name);
        return;
    }
    ul_comp_meth = comp_meth;
    ul_bit_width = bit_width;
}

uint16_t Cell::getDecompressionPrbSize() const
{
    switch(ul_bit_width)
    {
    case BFP_NO_COMPRESSION:
        return static_cast<uint16_t>(BFP_NO_COMPRESSION_BITWIDTH);
    case BFP_COMPRESSION_9_BITS:
        return static_cast<uint16_t>(BITWIDTH_9_BITS);
    case BFP_COMPRESSION_14_BITS:
        return static_cast<uint16_t>(BITWIDTH_14_BITS);
    default:
        return 0;
    };

    return 0;
}

int Cell::getFsDlOffset() const
{
    return fs_offset_dl;
}

int Cell::getFsUlOffset() const
{
    return fs_offset_ul;
}

int Cell::getDlExponent() const
{
    return exponent_dl;
}

void Cell::setDlExponent(int exp_dl)
{
    exponent_dl = exp_dl;
}

void Cell::setUlExponent(int exp_ul)
{
    exponent_ul = exp_ul;
}

void Cell::setRefDl(int ref_dl)
{
    this->ref_dl = ref_dl;
}

int Cell::getUlMaxAmp() const
{
    return max_amp_ul;
}

void Cell::setUlMaxAmp(int max_amp_ul)
{
    this->max_amp_ul = max_amp_ul;
}

void Cell::setAttenuation_dB(float attenuation_dB)
{
    oam_linear_gain = pow(10.0,-attenuation_dB/20.0);
}

float Cell::getBetaUlPowerScaling() const
{
    return beta_ul * oam_linear_gain;
}

float Cell::getBetaDlPowerScaling() const
{
    return beta_dl * oam_linear_gain;
}

std::string Cell::getNicName() const
{
    return nic_name;
}

int Cell::setNicName(std::string nic_name)
{
    int ret = 0;
    if(this->nic_name == nic_name)
    {
        NVLOGC_FMT(TAG, "{}: current nic is already {}", __func__, nic_name);
        return 0;
    }
    if(fh_proxy->checkIfNicExists(nic_name))
    {
        this->nic_name    = nic_name;
    }
    else
    {
        NVLOGC_FMT(TAG, "{}: nic {} not supported", __func__, nic_name);
        ret = -1;
    }
    return ret;
}

uint32_t  Cell::getNicIndex() const
{
    return nic_index;
}

uint8_t Cell::getDlcCoreIndex() const
{
    return dlc_core_index;
}

uint32_t* Cell::getLastRxItem() const
{
    return (uint32_t *)last_sem_idx_rx_h->addr();
}

uint32_t* Cell::getLastOrderedItem() const
{
    return (uint32_t *)last_sem_idx_order_h->addr();
}

uint32_t* Cell::getLastRxSrsItem() const
{
    return (uint32_t *)last_sem_idx_srs_rx_h->addr();
}

uint32_t* Cell::getLastOrderedSrsItem() const
{
    return (uint32_t *)last_sem_idx_srs_order_h->addr();
}

uint64_t* Cell::getOrderkernelLastTimeoutErrorTimeItem() const
{
    return (uint64_t *)order_kernel_last_timeout_error_time->addr();
}

uint64_t* Cell::getOrderkernelSrsLastTimeoutErrorTimeItem() const
{
    return (uint64_t *)order_kernel_srs_last_timeout_error_time->addr();
}


uint32_t* Cell::getNextSlotEarlyRxPackets() const
{
    return (uint32_t *)next_slot_early_rx_packets->addr();
}

uint32_t* Cell::getNextSlotLateRxPackets() const
{
    return (uint32_t *)next_slot_late_rx_packets->addr();
}

uint32_t* Cell::getNextSlotOnTimeRxPackets() const
{
    return (uint32_t *)next_slot_on_time_rx_packets->addr();
}

uint32_t* Cell::getNextSlotEarlyRxPacketsSRS() const
{
    return (uint32_t *)next_slot_early_rx_packets_srs->addr();
}

uint32_t* Cell::getNextSlotLateRxPacketsSRS() const
{
    return (uint32_t *)next_slot_late_rx_packets_srs->addr();
}

uint32_t* Cell::getNextSlotOnTimeRxPacketsSRS() const
{
    return (uint32_t *)next_slot_on_time_rx_packets_srs->addr();
}

uint32_t* Cell::getNextSlotRxPacketCountSRS() const
{
    return (uint32_t *)next_slot_rx_packets_count_srs->addr();
}

uint32_t* Cell::getNextSlotRxByteCountSRS() const
{
    return (uint32_t *)next_slot_rx_bytes_count_srs->addr();
}


uint32_t* Cell::getNextSlotRxPacketCountPerSymSRS() const
{
    return (uint32_t *)next_slot_rx_packets_count_per_sym_srs->addr();
}

uint64_t* Cell::getNextSlotRxPacketTsSRS() const
{
    return (uint64_t *)next_slot_rx_packets_ts_srs->addr();
}

uint32_t* Cell::getNextSlotRxPacketCount() const
{
    return (uint32_t *)next_slot_rx_packets_count->addr();
}

uint32_t* Cell::getNextSlotRxByteCount() const
{
    return (uint32_t *)next_slot_rx_bytes_count->addr();
}

uint64_t* Cell::getNextSlotRxPacketTs() const
{
    return (uint64_t *)next_slot_rx_packets_ts->addr();
}

uint32_t* Cell::getNextSlotNumPrbCh1() const
{
    return (uint32_t *)next_slot_num_prb_ch1->addr();
}

uint32_t* Cell::getNextSlotNumPrbCh2() const
{
    return (uint32_t *)next_slot_num_prb_ch2->addr();
}

uint32_t* Cell::getNextSlotNumPrbCh3() const
{
    return (uint32_t *)next_slot_num_prb_ch3->addr();
}

uint32_t* Cell::getULPcapCaptureBufferIndex() const
{
    return (uint32_t *)ul_pcap_capture_buffer_index->addr();
}

//////////////////////////////////////////////////////////////////////////////////////
//// Active
//////////////////////////////////////////////////////////////////////////////////////

bool Cell::isActive()
{
    cell_status c = active.load();
    if(c == CELL_INACTIVE)
        return false;
    else
        return true;
}

bool Cell::isActiveSrs()
{
    cell_status c = active_srs.load();
    if(c == CELL_INACTIVE)
        return false;
    else
        return true;
}

void Cell::stop()
{
    active = CELL_INACTIVE;
    active_srs = CELL_INACTIVE;
}

void Cell::start()
{
    active = CELL_ACTIVE;
    active_srs = CELL_ACTIVE;
}

bool Cell::isHealthy()
{
    cell_status c = active.load();
    if(c == CELL_ACTIVE)
        return true;
    else
        return false;
}

bool Cell::isHealthySrs()
{
    cell_status c = active_srs.load();
    if(c == CELL_ACTIVE)
        return true;
    else
        return false;
}

void Cell::setUnhealthy()
{
    active = CELL_UNHEALTHY;
}

void Cell::setUnhealthySrs()
{
    active_srs = CELL_UNHEALTHY;
}

void Cell::setHealthy()
{
    active = CELL_ACTIVE;
}

void Cell::setHealthySrs()
{
    active_srs = CELL_ACTIVE;
}

void Cell::lockRxQueue()
{
    rx_queue_lock.lock();
    // rx_queue_locked = true;
}

void Cell::unlockRxQueue()
{
    // rx_queue_locked = false;
    rx_queue_lock.unlock();
}

static void parse_eth_addr(const std::string &eth_addr_str, std::array<uint8_t, 6> &eth_addr_bytes)
{
    char * pch = nullptr;
    int i=0, tmp=0;

    pch = strtok (const_cast<char*>(eth_addr_str.c_str()),":");
    while (pch != nullptr)
    {
        std::stringstream str;
        std::string stok;
        stok.assign(pch);
        str << stok;
        str >> std::hex >> tmp;
        eth_addr_bytes[i] = static_cast<uint8_t>(tmp);
        pch = strtok (nullptr,":");
        i++;
    }
}

int Cell::update_peer_rx_metrics(size_t rx_packets, size_t rx_bytes)
{
    return fh_proxy->update_peer_rx_metrics(nic2peer_map[nic_name], rx_packets, rx_bytes);
}

int Cell::update_peer_tx_metrics(size_t tx_packets, size_t tx_bytes)
{
    return fh_proxy->update_peer_tx_metrics(nic2peer_map[nic_name], tx_packets, tx_bytes);
}

int Cell::updateCellConfig(
    enum UserDataCompressionMethod dl_comp_meth,
    uint8_t dl_bit_width
)
{
    NVLOGI_FMT(TAG, "{}: dl_comp_meth={} dl_bit_width={}", __func__, (int)dl_comp_meth, dl_bit_width);
    return fh_proxy->updatePeer(nic2peer_map[nic_name], dl_comp_meth, dl_bit_width);
}

int Cell::updateCellConfig(
    std::string dst_eth_addr,
    uint16_t vlan_tci
)
{
    NVLOGI_FMT(TAG, "{}: dst_eth_addr={} vlan_tci=0x{:X}", __func__, dst_eth_addr.c_str(), vlan_tci);
    std::array<uint8_t, 6>  dst_eth_addr_bytes;
    parse_eth_addr(dst_eth_addr, dst_eth_addr_bytes);
    return fh_proxy->updatePeer(nic2peer_map[nic_name], dst_eth_addr_bytes, vlan_tci,geteAxCIdsUl(),(std::vector<uint16_t>&)geteAxCIdsSrs());
}

int Cell::updateFhLenConfig(
    uint16_t fh_len_range)
{
    if(this->fh_len_range == fh_len_range)
    {
        NVLOGC_FMT(TAG, "{}: fh_len_range is already {}, (0: 0~30km, 1: 20~50km), no need to adjust delay params", __func__, fh_len_range);
        return -1;
    }
    NVLOGC_FMT(TAG, "{}: fh_len_range change to {} from {}, (0: 0~30km, 1: 20~50km)", __func__, fh_len_range, this->fh_len_range);

    t1a_max_up_ns += fh_len_range ? FH_EXTENSION_DELAY_ADJUSTMENT : -FH_EXTENSION_DELAY_ADJUSTMENT;
    t1a_max_cp_ul_ns += fh_len_range ? FH_EXTENSION_DELAY_ADJUSTMENT : -FH_EXTENSION_DELAY_ADJUSTMENT;
    ta4_min_ns += fh_len_range ? FH_EXTENSION_DELAY_ADJUSTMENT : -FH_EXTENSION_DELAY_ADJUSTMENT;
    ta4_max_ns += fh_len_range ? FH_EXTENSION_DELAY_ADJUSTMENT : -FH_EXTENSION_DELAY_ADJUSTMENT;

    NVLOGC_FMT(TAG, "{}: t1a_max_up_ns adjusted to {}", __func__, t1a_max_up_ns);
    NVLOGC_FMT(TAG, "{}: t1a_max_cp_ul_ns adjusted to {}", __func__, t1a_max_cp_ul_ns);
    NVLOGC_FMT(TAG, "{}: ta4_min_ns adjusted to {}", __func__, ta4_min_ns);
    NVLOGC_FMT(TAG, "{}: ta4_max_ns adjusted to {}", __func__, ta4_max_ns);

    this->fh_len_range = fh_len_range;
    return 0;
}

struct doca_rx_items* Cell::docaGetRxqInfo()
{
    return &nic2doca_rxq_info_map[nic_name];
}

uint16_t Cell::docaGetSemNum()
{
    return nic2doca_rxq_info_map[nic_name].nitems;
}

struct doca_rx_items* Cell::docaGetRxqInfoSrs()
{
    return &nic2doca_rxq_info_srs_map[nic_name];
}

uint16_t Cell::docaGetSemNumSrs()
{
    return nic2doca_rxq_info_srs_map[nic_name].nitems;
}

bfw_buffer_info* Cell::getBfwCoeffBuffer()
{
    return &bfw_coeff_buffer_info;
}
