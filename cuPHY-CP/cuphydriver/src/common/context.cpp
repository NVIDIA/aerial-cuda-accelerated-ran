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

#define TAG (NVLOG_TAG_BASE_CUPHY_DRIVER + 3) // "DRV.CTX"
#define TAG_STARTUP_TIMES (NVLOG_TAG_BASE_CUPHY_CONTROLLER + 5) // "CTL.STARTUP_TIMES"
#define TAG_METRICS_FAPI_SRS_TAG (NVLOG_TAG_BASE_CUPHY_DRIVER + 44) // "DRV.SRS_FAPI_PACKET_SUMMARY"
#define TAG_PERF_METRICS (NVLOG_TAG_BASE_CUPHY_DRIVER + 48) // "DRV.PERF_METRICS"

#include "context.hpp"
#include "nvlog.hpp"
#include "time.hpp"
#include "cuphydriver_api.hpp"
#include "aerial-fh-driver/oran.hpp"
#include "aerial-fh-driver/pcap_logger.hpp"
#include "ti_generic.hpp"
#include <fstream>
#include "cuphyoam.hpp"

void default_err_hndl(const char* msg)
{
    fputs(msg, stderr);
}

void default_inf_hndl(const char* msg)
{
    fputs(msg, stdout);
}

void default_dbg_hndl(const char* msg)
{
    fputs(msg, stdout);
}

// This is a minimal constructor initialized for test purposes.
PhyDriverCtx::PhyDriverCtx(const context_config & ctx_cfg, bool minimal) : 
    log_err_fn_(default_err_hndl),
    log_inf_fn_(default_inf_hndl),
    log_dbg_fn_(default_dbg_hndl),
    log_lvl(L1_LOG_LVL_ERROR),
    use_batched_memcpy(ctx_cfg.use_batched_memcpy),
    m_batchedMemcpyHelper(DL_MAX_CELLS_PER_SLOT, batchedMemcpySrcHint::srcIsDevice, batchedMemcpyDstHint::dstIsHost, (use_batched_memcpy == 1) && (CUPHYDRIVER_PDSCH_USE_BATCHED_COPY == 1)) 
{
    standalone = ctx_cfg.standalone;
    validation = ctx_cfg.validation;  // Enable validation mode for safer testing
    prometheus_cpu_core = ctx_cfg.prometheus_cpu_core;
    data_core = ctx_cfg.data_core; 
    enable_cpu_init_comms     = ctx_cfg.enable_cpu_init_comms;

    enable_ul_cuphy_graphs  = ctx_cfg.enable_ul_cuphy_graphs;
    enable_dl_cuphy_graphs  = ctx_cfg.enable_dl_cuphy_graphs;

    // Initialize additional fields that FhProxy constructor may use
    enable_gpu_comm_via_cpu  = ctx_cfg.enable_gpu_comm_via_cpu;  // CPU-only mode
    send_static_bfw_wt_all_cplane  = ctx_cfg.send_static_bfw_wt_all_cplane;
    enable_gpu_comm_dl     = ctx_cfg.enable_gpu_comm_dl;
    mMIMO_enable           = ctx_cfg.mMIMO_enable;
    pusch_nMaxTbPerNode    = ctx_cfg.pusch_nMaxTbPerNode;
    sendCPlane_timing_error_th_ns = ctx_cfg.sendCPlane_timing_error_th_ns;
    start_section_id_prach = ctx_cfg.start_section_id_prach;
    start_section_id_srs   = ctx_cfg.start_section_id_srs;

    fh_proxy = std::make_unique<FhProxy>((phydriver_handle)this,ctx_cfg);

    for (auto nic_cfg : ctx_cfg.nic_configs) {
        if (fh_proxy->registerNic(nic_cfg, ctx_cfg.gpu_id))
            PHYDRIVER_THROW_EXCEPTIONS(EINVAL, "NIC registration error");
    }
    minimal_phydriver = minimal; 
}


PhyDriverCtx::PhyDriverCtx(const context_config& ctx_cfg) :
    log_err_fn_(default_err_hndl),
    log_inf_fn_(default_inf_hndl),
    log_dbg_fn_(default_dbg_hndl),
    log_lvl(L1_LOG_LVL_ERROR),
    use_batched_memcpy(ctx_cfg.use_batched_memcpy),
    m_batchedMemcpyHelper(DL_MAX_CELLS_PER_SLOT, batchedMemcpySrcHint::srcIsDevice, batchedMemcpyDstHint::dstIsHost, (use_batched_memcpy == 1) && (CUPHYDRIVER_PDSCH_USE_BATCHED_COPY == 1))
{
    ctx_tot_cpu_regular_memory = 0;
    ctx_tot_cpu_pinned_memory = 0;
    ctx_tot_gpu_regular_memory = 0;
    ctx_tot_gpu_pinned_memory = 0;

    mf.init(this, std::string("Context"), sizeof(PhyDriverCtx));
    cuphyChannelsAccumMf.init(this, std::string("cuphyChannelsAccumMf"), 0);
    wip_accum_mf.init(this, std::string("wipAccumAccumMf"), 0);

    for(int i = 0; i < SLOT_MAP_NUM; i++)
    {
        slot_map_ul_array[i] = std::move(new SlotMapUl(static_cast<phydriver_handle>(this), i)); //.reset(new SlotMap(static_cast<phydriver_handle>(this), i));
        slot_map_dl_array[i] = std::move(new SlotMapDl(static_cast<phydriver_handle>(this), i, 0)); //Force disable batched memcpy inside DL slotmap usage for now (TODO: Re-enable back based on perf analysis data). NOTE : Enabling batched memcpy presently could result in erroneous behavior owing to the "DL Task GPU Comms Prepare" task fanned out and cuphyBatchedMemcpyHelper
                                                                                                    //helper methods not called in a thread-safe manner.
    }

    sc_aggr_array = (struct slot_params_aggr*) calloc(SLOT_CMD_NUM, sizeof(struct slot_params_aggr));
    for(int i = 0; i < SLOT_CMD_NUM; i++)
        sc_aggr_array[i].cleanup();

    // slot_map_list.push_back(std::unique_ptr<SlotMap>(new SlotMap(static_cast<phydriver_handle>(this), i)));

    for(int i = 0; i < TASK_ITEM_NUM; i++)
        task_item_array[i] = std::move(new Task(static_cast<phydriver_handle>(this), i)); //.reset(new Task(static_cast<phydriver_handle>(this), i));
    // task_item_list.push_back(std::unique_ptr<Task>(new Task(static_cast<phydriver_handle>(this), i)));

    sc_aggr_index = 0;
    slot_map_ul_index   = 0;
    slot_map_dl_index   = 0;
    task_item_index     = 0;
    oentity_index       = 0;
    aggr_error_info_dl.nonAvailCount = 0;
    aggr_error_info_dl.prevSlotNonAvail = false;
    aggr_error_info_dl.l1RecoverySlots = 0;
    aggr_error_info_dl.availCount = 0;
    aggr_error_info_ul.nonAvailCount = 0;
    aggr_error_info_ul.prevSlotNonAvail = false; 
    aggr_error_info_ul.l1RecoverySlots = 0;
    aggr_error_info_ul.availCount = 0;
    slot_advance = TICK_SLOT_ADVANCE_INIT_VAL;
    standalone          = ctx_cfg.standalone;
    validation          = ctx_cfg.validation;
    cplane_disable      = ctx_cfg.cplane_disable;
    prometheus_cpu_core = ctx_cfg.prometheus_cpu_core;
    start_section_id_prach = ctx_cfg.start_section_id_prach;
    start_section_id_srs   = ctx_cfg.start_section_id_srs;
    workers_ul_cores    = ctx_cfg.ul_cores;
    workers_dl_cores    = ctx_cfg.dl_cores;
    workers_dl_validation_cores = ctx_cfg.dl_validation_cores;
    use_green_contexts  = ctx_cfg.use_green_contexts;
    use_gc_workqueues   = ctx_cfg.use_gc_workqueues;

    mps_sm_pusch        = ctx_cfg.mps_sm_pusch;
    mps_sm_pucch        = ctx_cfg.mps_sm_pucch;
    mps_sm_prach        = ctx_cfg.mps_sm_prach;
    mps_sm_ul_order     = ctx_cfg.mps_sm_ul_order;
    mps_sm_srs          = ctx_cfg.mps_sm_srs;
    mps_sm_pdsch        = ctx_cfg.mps_sm_pdsch;
    mps_sm_pdcch        = ctx_cfg.mps_sm_pdcch;
    mps_sm_pbch         = ctx_cfg.mps_sm_pdcch;
    mps_sm_dl_ctrl      = mps_sm_pdcch + mps_sm_pbch;
    mps_sm_gpu_comms    = ctx_cfg.mps_sm_gpu_comms;

    pdsch_fallback      = ctx_cfg.pdsch_fallback;
    cell_group_num      = ctx_cfg.cell_group_num;
    pusch_workCancelMode = ctx_cfg.pusch_workCancelMode;
    enable_pusch_tdi    = ctx_cfg.enable_pusch_tdi;
    enable_pusch_cfo    = ctx_cfg.enable_pusch_cfo;
    enable_pusch_dftsofdm    = ctx_cfg.enable_pusch_dftsofdm;
    enable_pusch_tbsizecheck = ctx_cfg.enable_pusch_tbsizecheck;
#ifdef SCF_FAPI_10_04
    pusch_earlyHarqEn   = 1;
#else
    pusch_earlyHarqEn   = 0;
#endif
    pusch_deviceGraphLaunchEn        = ctx_cfg.pusch_deviceGraphLaunchEn;
    pusch_waitTimeOutPreEarlyHarqUs  = ctx_cfg.pusch_waitTimeOutPreEarlyHarqUs;
    pusch_waitTimeOutPostEarlyHarqUs = ctx_cfg.pusch_waitTimeOutPostEarlyHarqUs;
    select_pusch_eqcoeffalgo = ctx_cfg.select_pusch_eqcoeffalgo;
    select_pusch_chestalgo   = ctx_cfg.select_pusch_chestalgo;
    enable_pusch_perprgchest = ctx_cfg.enable_pusch_perprgchest;
    enable_pusch_to     = ctx_cfg.enable_pusch_to;
    enable_pusch_rssi   = ctx_cfg.enable_pusch_rssi;
    enable_pusch_sinr   = ctx_cfg.enable_pusch_sinr;
    enable_weighted_average_cfo = ctx_cfg.enable_weighted_average_cfo;
    mPuxchPolarDcdrListSz = ctx_cfg.mPuxchPolarDcdrListSz;
    mPuschrxChestFactorySettingsFilename = ctx_cfg.mPuschrxChestFactorySettingsFilename;
    notify_ul_harq_buffer_release = ctx_cfg.notify_ul_harq_buffer_release;

    fix_beta_dl         = ctx_cfg.fix_beta_dl;
    disable_empw = ctx_cfg.disable_empw;
    enable_cpu_task_tracing = ctx_cfg.enable_cpu_task_tracing;
    enable_prepare_tracing = ctx_cfg.enable_prepare_tracing;
    cupti_enable_tracing = ctx_cfg.cupti_enable_tracing;
    cupti_buffer_size = ctx_cfg.cupti_buffer_size;
    cupti_num_buffers = ctx_cfg.cupti_num_buffers;
    enable_dl_cqe_tracing = ctx_cfg.enable_dl_cqe_tracing;
    cqe_trace_cell_mask = ctx_cfg.cqe_trace_cell_mask;
    cqe_trace_slot_mask = ctx_cfg.cqe_trace_slot_mask;
    enable_ok_tb = ctx_cfg.enable_ok_tb;
    num_ok_tb_slot = ctx_cfg.num_ok_tb_slot;    
    ul_rx_pkt_tracing_level = ctx_cfg.ul_rx_pkt_tracing_level;
    ul_rx_pkt_tracing_level_srs = ctx_cfg.ul_rx_pkt_tracing_level_srs;
    ul_warmup_frame_count = ctx_cfg.ul_warmup_frame_count;
    pmu_metrics = ctx_cfg.pmu_metrics;
    h2d_copy_thread_enable = ctx_cfg.h2d_cpy_th_cfg.enable_h2d_copy_thread;
    enable_l1_param_sanity_check = ctx_cfg.enable_l1_param_sanity_check;

    mMIMO_enable           = ctx_cfg.mMIMO_enable;
    enable_srs             = ctx_cfg.enable_srs;
    enable_dl_core_affinity = ctx_cfg.enable_dl_core_affinity;
    dlc_core_packing_scheme = ctx_cfg.dlc_core_packing_scheme;
    
    // Validate DL C-plane core packing scheme
    if (dlc_core_packing_scheme == 2) {
        NVLOGE_FMT(TAG, AERIAL_CONFIG_EVENT, "dlc_core_packing_scheme=2 (dynamic workload-based) is not yet supported");
        PHYDRIVER_THROW_EXCEPTIONS(EINVAL, "dlc_core_packing_scheme=2 is not yet supported");
    }
    if (dlc_core_packing_scheme > 2) {
        NVLOGE_FMT(TAG, AERIAL_CONFIG_EVENT, "Invalid dlc_core_packing_scheme value: {}. Must be 0, 1, or 2", dlc_core_packing_scheme);
        PHYDRIVER_THROW_EXCEPTIONS(EINVAL, "Invalid dlc_core_packing_scheme value");
    }
    // Note: scheme 1 (fixed per-cell) requires enable_dl_core_affinity=1
    if (dlc_core_packing_scheme == 1 && enable_dl_core_affinity == 0) {
        NVLOGE_FMT(TAG, AERIAL_CONFIG_EVENT, "dlc_core_packing_scheme=1 requires enable_dl_core_affinity=1");
        PHYDRIVER_THROW_EXCEPTIONS(EINVAL, "dlc_core_packing_scheme=1 requires enable_dl_core_affinity=1");
    }
    
    aggr_obj_non_avail_th  = ctx_cfg.aggr_obj_non_avail_th;
    split_ul_cuda_streams  = ctx_cfg.split_ul_cuda_streams;
    serialize_pucch_pusch  = ctx_cfg.serialize_pucch_pusch;
    sendCPlane_timing_error_th_ns = ctx_cfg.sendCPlane_timing_error_th_ns;
    sendCPlane_ulbfw_backoff_th_ns = ctx_cfg.sendCPlane_ulbfw_backoff_th_ns;
    sendCPlane_dlbfw_backoff_th_ns = ctx_cfg.sendCPlane_dlbfw_backoff_th_ns;
    forcedNumCsi2Bits         = ctx_cfg.forcedNumCsi2Bits;
    pusch_nMaxLdpcHetConfigs  = ctx_cfg.pusch_nMaxLdpcHetConfigs;
    pusch_nMaxTbPerNode       = ctx_cfg.pusch_nMaxTbPerNode;
    mCh_segment_proc_enable   = ctx_cfg.mCh_segment_proc_enable;
    max_ru_unhealthy_ul_slots = ctx_cfg.max_ru_unhealthy_ul_slots;
    ul_pcap_capture_enable = ctx_cfg.ul_pcap_capture_enable;
    ul_pcap_capture_mtu = 0;
    for(const auto& nic: ctx_cfg.nic_configs)
    {
        ul_pcap_capture_mtu = (ul_pcap_capture_mtu > nic.nic_mtu) ? ul_pcap_capture_mtu : nic.nic_mtu;
        if(enable_ok_tb)
            setConfigOkTbMaxPacketSize(nic.nic_mtu);
    }
    ulbfw_aggr_per_ctx = (enable_ok_tb) ? (PHY_ULBFW_AGGR_X_CTX<<1) : PHY_ULBFW_AGGR_X_CTX; //Double the number of aggr objects if OKTB is enabled since the ULBFW aggr objects are seen to be exhausted during the pre Phase4 run for the test bench

    // Round up to next 16 byte multiple to be able to do vectorized copies
    ul_pcap_capture_mtu = (ul_pcap_capture_mtu + 15) & ~15;

    ul_pcap_capture_thread_cpu_affinity = ctx_cfg.ul_pcap_capture_thread_cpu_affinity;
    ul_pcap_capture_thread_sched_priority = ctx_cfg.ul_pcap_capture_thread_sched_priority;
    srs_chest_algo_type                    = (cuphySrsChEstAlgoType_t)ctx_cfg.srs_chest_algo_type;
    srs_chest_tol2_normalization_algo_type = ctx_cfg.srs_chest_tol2_normalization_algo_type;
    srs_chest_tol2_constant_scaler         = ctx_cfg.srs_chest_tol2_constant_scaler;
    bfw_power_normalization_alg_selector   = ctx_cfg.bfw_power_normalization_alg_selector;
    bfw_beta_prescaler                     = ctx_cfg.bfw_beta_prescaler;
    total_num_srs_chest_buffers            = ctx_cfg.total_num_srs_chest_buffers;
    send_static_bfw_wt_all_cplane          = ctx_cfg.send_static_bfw_wt_all_cplane;

    task_list_ul = std::unique_ptr<TaskList>(new TaskList((phydriver_handle)this, 0, TASK_LIST_SIZE));
    task_list_dl = std::unique_ptr<TaskList>(new TaskList((phydriver_handle)this, 1, TASK_LIST_SIZE));
    task_list_dl_validation = std::unique_ptr<TaskList>(new TaskList((phydriver_handle)this, 2, TASK_LIST_SIZE));
    task_list_debug = std::unique_ptr<TaskList>(new TaskList((phydriver_handle)this, 3, TASK_LIST_SIZE));

    debug_worker  = ctx_cfg.debug_worker;
    data_core  = ctx_cfg.data_core;

    enable_ul_cuphy_graphs = ctx_cfg.enable_ul_cuphy_graphs;
    enable_dl_cuphy_graphs = ctx_cfg.enable_dl_cuphy_graphs;
    //FIXME: from controller
    enable_gpu_comm_dl     = ctx_cfg.enable_gpu_comm_dl;
    enable_gpu_comm_via_cpu     = ctx_cfg.enable_gpu_comm_via_cpu;
    enable_cpu_init_comms     = ctx_cfg.enable_cpu_init_comms;

    ul_order_timeout_cpu_ns = (ctx_cfg.ul_order_timeout_cpu_ns == 0 ? (ORDER_KERNEL_ENABLE_THRESHOLD * NS_X_MS) : ctx_cfg.ul_order_timeout_cpu_ns);
    ul_order_timeout_gpu_ns = (ctx_cfg.ul_order_timeout_gpu_ns == 0 ? (ORDER_KERNEL_WAIT_TIMEOUT_MS * NS_X_MS) : ctx_cfg.ul_order_timeout_gpu_ns);
    ul_order_timeout_gpu_srs_ns = (ctx_cfg.ul_order_timeout_gpu_srs_ns == 0 ? (ORDER_KERNEL_WAIT_TIMEOUT_MS * NS_X_MS) : ctx_cfg.ul_order_timeout_gpu_srs_ns);
    ul_order_timeout_gpu_log_enable = ctx_cfg.ul_order_timeout_gpu_log_enable;
    ul_order_kernel_mode = ctx_cfg.ul_order_kernel_mode;
    ue_mode = ctx_cfg.ue_mode;
    ul_srs_aggr3_task_launch_offset_ns = ctx_cfg.ul_srs_aggr3_task_launch_offset_ns;

    pusch_aggr_per_ctx = ctx_cfg.pusch_aggr_per_ctx;
    pucch_aggr_per_ctx = ctx_cfg.pucch_aggr_per_ctx;
    srs_aggr_per_ctx = ctx_cfg.srs_aggr_per_ctx;
    prach_aggr_per_ctx = ctx_cfg.prach_aggr_per_ctx;
    max_harq_pools  = ctx_cfg.max_harq_pools;
    max_harq_tx_count_bundled = ctx_cfg.max_harq_tx_count_bundled;
    max_harq_tx_count_non_bundled = ctx_cfg.max_harq_tx_count_non_bundled;
    ul_input_buffer_per_cell = ctx_cfg.ul_input_buffer_per_cell;
    ul_input_buffer_per_cell_srs = ctx_cfg.ul_input_buffer_per_cell_srs;
    ul_order_max_rx_pkts    = (ctx_cfg.ul_order_max_rx_pkts == 0 ? (ORDER_KERNEL_MAX_RX_PKTS) : ctx_cfg.ul_order_max_rx_pkts);
    ul_order_rx_pkts_timeout_ns    = (ctx_cfg.ul_order_rx_pkts_timeout_ns == 0 ? (ORDER_KERNEL_RX_PKTS_TIMEOUT_NS) : ctx_cfg.ul_order_rx_pkts_timeout_ns);
    ul_order_timeout_first_pkt_gpu_ns = (uint32_t)(ul_order_timeout_gpu_ns/2);
    ul_order_timeout_first_pkt_gpu_srs_ns = (uint32_t)(ul_order_timeout_gpu_srs_ns/2);
    ul_order_timeout_log_interval_ns = ctx_cfg.ul_order_timeout_log_interval_ns;
    enable_tx_notification = ctx_cfg.enable_tx_notification;

    //Set DL wait threshold values
    if(ctx_cfg.dl_wait_th_list.size()==0){
        //Set default values
        h2d_copy_wait_th = 500000; //0.5 ms
        cuphy_dl_channel_wait_th = 4000000; //4 ms
    }
    else if(ctx_cfg.dl_wait_th_list.size()==1){ //Only h2d wait th provided in Yaml
        h2d_copy_wait_th = ctx_cfg.dl_wait_th_list[0];
        cuphy_dl_channel_wait_th = 4000000; //4 ms
    }
    else
    {
        h2d_copy_wait_th = ctx_cfg.dl_wait_th_list[0];
        cuphy_dl_channel_wait_th = ctx_cfg.dl_wait_th_list[1];
    }

    for(int ulc = 0; ulc < workers_ul_cores.size(); ulc++)
    {
        char name[16];
        snprintf(name, 16, "UlPhyDriver%02d", workers_ul_cores[ulc]);
        worker_id wid = create_worker_id();
        std::unique_ptr<Worker> w = std::unique_ptr<Worker>(new Worker((phydriver_handle)this, wid, WORKER_UL, name, workers_ul_cores[ulc],
                                                                       ctx_cfg.workers_sched_priority, ctx_cfg.pmu_metrics, worker_default, nullptr));
        auto ret = worker_ul_map.insert(std::pair<worker_id, std::unique_ptr<Worker>>(wid, std::move(w)));
        if(ret.second == false)
            PHYDRIVER_THROW_EXCEPTIONS(errno, "Worker UL creation error");

        worker_ul_ordering.push_back(wid);
    }

    for(int dlc = 0; dlc < workers_dl_cores.size(); dlc++)
    {
        char name[16];
        snprintf(name, 16, "DlPhyDriver%02d", workers_dl_cores[dlc]);
        worker_id wid = create_worker_id();
        std::unique_ptr<Worker> w = std::unique_ptr<Worker>(new Worker((phydriver_handle)this, wid, WORKER_DL, name, workers_dl_cores[dlc],
                                                                       ctx_cfg.workers_sched_priority, ctx_cfg.pmu_metrics, worker_default, nullptr));
        auto ret = worker_dl_map.insert(std::pair<worker_id, std::unique_ptr<Worker>>(wid, std::move(w)));
        if(ret.second == false)
            PHYDRIVER_THROW_EXCEPTIONS(errno, "Worker DL creation error");

        worker_dl_ordering.push_back(wid);
    }

    // If we have the debug core, start that worker too
    if (debug_worker_enabled()) {
        char name[20];
        snprintf(name, 20, "DebugWorker%02d", debug_worker);
        worker_id wid = create_worker_id();
        std::unique_ptr<Worker> w = std::unique_ptr<Worker>(new Worker((phydriver_handle)this, wid, WORKER_GENERIC, name, debug_worker,
                                                                       ctx_cfg.workers_sched_priority, ctx_cfg.pmu_metrics, worker_default, nullptr));
        addGenericWorker(std::move(w));
    }

    // If  data collection is enabled, create the object and start it
    if (datalake_enabled()) {
        dataLake.reset( new DataLake(
            ctx_cfg.datalake_db_write_enable,
            ctx_cfg.datalake_samples,
            ctx_cfg.datalake_address,
            ctx_cfg.datalake_engine,
            ctx_cfg.datalake_data_types,
            ctx_cfg.datalake_store_failed_pdu,
            ctx_cfg.num_rows_fh,
            ctx_cfg.num_rows_pusch,
            ctx_cfg.num_rows_hest,
            ctx_cfg.num_rows_srs_iq,
            ctx_cfg.num_rows_srs,
            ctx_cfg.num_rows_srs_hest,
            ctx_cfg.e3_agent_enabled,
            ctx_cfg.e3_rep_port,
            ctx_cfg.e3_pub_port,
            ctx_cfg.e3_sub_port,
            ctx_cfg.datalake_drop_tables));

        // PUSCH/FH worker thread
        std::thread t = std::thread(waitForLakeData, dataLake.get());
        datalake_pusch_thread.swap(t);

        int name_st = pthread_setname_np(datalake_pusch_thread.native_handle(), "datalake_pusch");

        if (name_st != 0 )
        {
            NVLOGE_FMT(TAG, AERIAL_THREAD_API_EVENT ,"datalake_pusch Thread pthread_setname_np failed with status: {}",std::strerror(name_st));
        }

        sched_param sch;
        int         policy;
        int         status = 0;
        //
        // Set thread CPU affinity
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(data_core, &cpuset);
        status = pthread_setaffinity_np(datalake_pusch_thread.native_handle(), sizeof(cpu_set_t), &cpuset);
        if(status)
        {
            NVLOGE_FMT(TAG, AERIAL_THREAD_API_EVENT, "datalake_pusch setaffinity_np failed with status : {}" , std::strerror(status));
        }

        // SRS worker thread — same data_core, parallel to PUSCH worker
        std::thread t_srs = std::thread(waitForLakeDataSrs, dataLake.get());
        datalake_srs_thread.swap(t_srs);

        name_st = pthread_setname_np(datalake_srs_thread.native_handle(), "datalake_srs");
        if (name_st != 0)
        {
            NVLOGE_FMT(TAG, AERIAL_THREAD_API_EVENT, "datalake_srs Thread pthread_setname_np failed with status: {}", std::strerror(name_st));
        }

        cpu_set_t cpuset_srs;
        CPU_ZERO(&cpuset_srs);
        CPU_SET(data_core, &cpuset_srs);
        status = pthread_setaffinity_np(datalake_srs_thread.native_handle(), sizeof(cpu_set_t), &cpuset_srs);
        if(status)
        {
            NVLOGE_FMT(TAG, AERIAL_THREAD_API_EVENT, "datalake_srs setaffinity_np failed with status : {}", std::strerror(status));
        }
    }

    for(int dl = 0; dl < workers_dl_validation_cores.size(); dl++)
    {
        // NVLOGC_FMT(TAG, "workers_dl_validation_cores[{}] =  {}", dl, workers_dl_validation_cores[dl]);
        char name[16];
        snprintf(name, 16, "DlValPhyDrv%02d", workers_dl_validation_cores[dl]);
        worker_id wid = create_worker_id();
        std::unique_ptr<Worker> w = std::unique_ptr<Worker>(new Worker((phydriver_handle)this, wid, WORKER_DL_VALIDATION, name, workers_dl_validation_cores[dl],
                                                                       ctx_cfg.workers_sched_priority, ctx_cfg.pmu_metrics, worker_default, nullptr));
        auto ret = worker_dl_validation_map.insert(std::pair<worker_id, std::unique_ptr<Worker>>(wid, std::move(w)));
        if(ret.second == false)
            PHYDRIVER_THROW_EXCEPTIONS(errno, "Worker DL Validation creation error");
    }

    std::unique_ptr<GpuDevice> g = std::unique_ptr<GpuDevice>(new GpuDevice((phydriver_handle)this, ctx_cfg.gpu_id, true));
    auto ret = gpu_map.insert(std::pair<int, std::unique_ptr<GpuDevice>>(ctx_cfg.gpu_id, std::move(g)));
    if(ret.second == false)
        PHYDRIVER_THROW_EXCEPTIONS(errno, "GPU can't be inserted");

    // Create MPS UL and DL context
    GpuDevice* gpu_device = getFirstGpu();

    // Log value of CUDA_DEVICE_MAX_CONNECTIONS env. variable. Useful for debugging, esp. for green contexts
    const char* dev_max_connections_env_var = getenv("CUDA_DEVICE_MAX_CONNECTIONS");
    NVLOGC_FMT(TAG, "CUDA_DEVICE_MAX_CONNECTIONS {}", dev_max_connections_env_var ? dev_max_connections_env_var : "not set");
    //FIXME Shall we manually hardcode the value to 12 or 32 for green contexts instead of relying on the user doing it beforehand?
    // Could also throw an error if this is green contexts mode and the value is below some threshold.

    // Log CUDA runtime version
    auto runtime_version = 0;
    CUDA_CHECK_PHYDRIVER(cudaRuntimeGetVersion(&runtime_version));
    NVLOGC_FMT(TAG, "CUDA runtime version from cudaRuntimeGetVersion(): {}", runtime_version);
    // Log latest CUDA version supported by the driver
    auto driver_version = 0;
    CUDA_CHECK_PHYDRIVER(cudaDriverGetVersion(&driver_version));
    NVLOGC_FMT(TAG, "CUDA driver version from cudaDriverGetVersion(): {}", driver_version);

    {
#if CUDA_VERSION < 12040
        if(use_green_contexts != 0)
        {
            // Exit, since green contexts are not supported.
            NVLOGF_FMT(TAG,AERIAL_INTERNAL_EVENT, "Green contexts are not supported before CUDA 12.4. Set use_green_contexts to 0 in the cuphycontroller yaml file and rerun.");
        }
#endif

        bool use_workqueues = (use_gc_workqueues != 0);
        NVLOGC_FMT(TAG, "use_green_contexts {}", use_green_contexts);
        NVLOGC_FMT(TAG, "use_gc_workqueues {}", use_gc_workqueues);
        if (use_green_contexts != 0) {

                const bool print_resources = true;

                unsigned int wq_concurrency_limit_of_2 = 2;
                unsigned int wq_concurrency_limit_of_1 = 1;
                // For specific GCs (PUSCH, PDSCH, Order kernel), a concurrency limit add-on of 1 is needed in the mMIMO case, as more
                // streams exist for those cases
                unsigned int wq_concurrency_limit_mmimo_add_on = (this->mMIMO_enable) ? 1 : 0;

		CUdevice device;
		CU_CHECK_PHYDRIVER(cuDeviceGet(&device, gpu_device->getId()));
		int gpuId = gpu_device->getId();
#if CUDA_VERSION >= 12040
                // Note that setting the CUDA_DEVICE_MAX_CONNECTIONS variable here via setenv will not have the desired
                // effect as the driver has already been initialized.

		// check if MPS service is running
		int mpsEnabled = 0;
		CU_CHECK_PHYDRIVER(cuDeviceGetAttribute(&mpsEnabled, CU_DEVICE_ATTRIBUTE_MPS_ENABLED, device));
		if (mpsEnabled == 1)
		{
			NVLOGE_FMT(TAG, AERIAL_CUPHY_EVENT,  "MPS is enabled. Heads-up that currently using green contexts with MPS enabled can have unintended side effects. Will run regardless.");
		}

		unsigned int use_flags = 0;
		//unsigned int use_flags = CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING;

		// Hard code SM granularity for green context splits
		int major_cc = 0;
		int minor_cc = 0;
		CU_CHECK_PHYDRIVER(cuDeviceGetAttribute(&major_cc, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
		int SM_granularity = 0;
                int min_count = 0;
		if(major_cc == 8)
		{
		    SM_granularity = 2;
                    min_count =  4;
		}
		else if (major_cc == 9)
		{
                    // 2 is in case the CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING_FLAG_IS_USED
                    SM_granularity = (use_flags == 0) ? 8 : 2;
                    min_count =  (use_flags == 0) ? 8 : 2;
		}
		else
		{
		    NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT,  "Running in untested compute capability ({}.{}). Granularity of SM splits for green contexts unknown.", major_cc, minor_cc);
		}
		int32_t gpuMaxSmCount = 0;
                CU_CHECK_PHYDRIVER(cuDeviceGetAttribute(&gpuMaxSmCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device));


                // On every cuDevSmResourceSplitByCount() API call, we create two CUdevResource(s): the resulting one and the remaining one.
                // Even though the remaining one may not be always used, and thus one could pass a nullptr to that API call, we keep the 2x notation for convenience;
                // Theoretically, every devResource could be used to create a green context.
		devResources.resize(2*CURRENT_MAX_GREEN_CTXS);
                // Both actual_split_groups and min_sm_counts vectors only exercise their event indices. The 2x CURRENT_MAX_GREEN_CTXS notation is again kept for convenience
		actual_split_groups.resize(2*CURRENT_MAX_GREEN_CTXS);
		min_sm_counts.resize(2*CURRENT_MAX_GREEN_CTXS);

		initial_device_GPU_resources = {};
		CU_CHECK_PHYDRIVER(cuDeviceGetDevResource(device, &initial_device_GPU_resources, default_resource_type));
		NVLOGC_FMT(TAG, "Initial GPU resources retrieved via cuDeviceGetDevResource() have type {} and SM count {}.",  +initial_device_GPU_resources.type, initial_device_GPU_resources.sm.smCount);

                // Current SM split strategy is to take into consideration the execution duration, timing requirements and scheduling pattern of different channels and establish,
                // as much as possible, min. SM overlap when different channels contend for the same SMs due to GPU oversubscription.
                // For example, if PDSCH gets 102 SMs and PUSCH 66, the goal is to have as little overlap between them as possible given these SM counts and also have some SMs only be used
                // by PDSCH and some only by PUSCH. Similarly, because DL (downlink channels) have tighter timing requirements than UL, their green contexts' SM allocations should have as
                // as little overhead as possible with those from UL channels.

                // Assume you have all available SMs of a GPU visualized as [0..max_SMs-1]. Current splits could potentially be visualized as shown below.
                // The numbers correspond to splits (i.e., cuDevSmResourceSplitByCount API calls). Every split, splits its resources into a resulting and a remaining split.
                // A [---] line is used when a resource isn't involved in that specific split; this happens when a previous resource is re-split.
                //
                // (1) [     ][      Y SMs      ]  PUSCH runs on a green context with Y SMs.
                // (2) [        ][     A + B SMs]  Split necessary to enable subsequent split, which will resplit the A+B resource for PUCCH and PRACH.
                // (3) [--------][ A SMs][ B SMs]  PUCCH's green context can use A SMs; PRACH's GC B SMs. Both PUCCH and PRACH will contend with some PUSCH SMs, but not with each other.
                // (4) [       N SMs       ][   ]  PDSCH can use N SMs. Partial overlap with PUSCH SMs depending on values of N, Y and max_SMs of the GPU.
                // (5) [X  ][                   ]  DL control channels (PDCCH, SSB and CSI-RS) can use X SMs. They will contend with PDSCH's SMs.
                // (6) [ ][Z ][---------------- ]  Order kernel runs on a green context with Z SMs. It does not overlap with the Y SMs of PUSCH or any of the SMs of the UL control channels.
                //                                 The resource for this split comes from a resplit of the resulting resource from split (1), i.e., the SMs not used by PUSCH.
                // (7) [W][                     ]  The gpuComm green context uses W SMs. These overlap with DL SMs.
                // Please note that these splits are all subject to change and the schematic is not up to scale. The actual values of Y, A, B, N, X, Z, W and the GPU's max_SM count also matter.

		// PUSCH
		unsigned int resource_index_pusch_split = 0;  //index into devResources
		actual_split_groups[resource_index_pusch_split] = 1;
		min_sm_counts[resource_index_pusch_split] = gpuMaxSmCount - getMpsSmPusch();
                // Since the resource split will round up every split but the remaining one to SM_granularity, update first split accordingly to ensure we have at least the
                // user specified SM count for PUSCH, as long as there are at least SM_granularity SMs left.
                int first_split_modulo_SM_granularity = (min_sm_counts[resource_index_pusch_split] % SM_granularity);
                if ((min_sm_counts[resource_index_pusch_split] > SM_granularity) && (first_split_modulo_SM_granularity != 0)) {
                    min_sm_counts[resource_index_pusch_split] -= first_split_modulo_SM_granularity;
                }
		CU_CHECK_PHYDRIVER(cuDevSmResourceSplitByCount(&devResources[resource_index_pusch_split], &actual_split_groups[resource_index_pusch_split], &initial_device_GPU_resources, &devResources[resource_index_pusch_split+1], use_flags, min_sm_counts[resource_index_pusch_split]));
		//greenContexts[0].create(gpuId, &devResources[resource_index_pusch_split+1]);

                // Increasing wq concurrency limit, if applicable, in case of mMIMO case by 1, given UL-BFW also runs under PUSCH ctx in this case
		puschMpsCtx = new MpsCtx((phydriver_handle)this, gpu_device, &devResources[resource_index_pusch_split+1], "PUSCH", print_resources, use_workqueues, wq_concurrency_limit_of_2 + wq_concurrency_limit_mmimo_add_on);
		mpsCtxList.push_back(puschMpsCtx);
		NVLOGC_FMT(TAG, "PUSCH green context with SM count of {}.", devResources[resource_index_pusch_split+1].sm.smCount);

                // Do another split of the initial GPU resources; will then resplit the remaining split for PUCCH and PRACH
                unsigned int resource_index_prep_for_ul_ctrl_split = 2;  //index into devResources
                actual_split_groups[resource_index_prep_for_ul_ctrl_split] = 1;
                int rounded_up_PUCCH = round_up_to_next(getMpsSmPucch(), SM_granularity);
                int rounded_up_PRACH = getMpsSmPrach() + 6; //FIXME temporarily increase from 2 to 8 SMs to see perf. impact if any
                min_sm_counts[resource_index_prep_for_ul_ctrl_split] = gpuMaxSmCount - rounded_up_PUCCH - rounded_up_PRACH;
                int second_split_modulo_SM_granularity = min_sm_counts[resource_index_prep_for_ul_ctrl_split] % SM_granularity;
                if ((min_sm_counts[resource_index_prep_for_ul_ctrl_split] > SM_granularity) && (second_split_modulo_SM_granularity != 0)) {
                    min_sm_counts[resource_index_prep_for_ul_ctrl_split] -= second_split_modulo_SM_granularity;
                }
                CU_CHECK_PHYDRIVER(cuDevSmResourceSplitByCount(&devResources[resource_index_prep_for_ul_ctrl_split], &actual_split_groups[resource_index_prep_for_ul_ctrl_split], &initial_device_GPU_resources, &devResources[resource_index_prep_for_ul_ctrl_split+1], use_flags, min_sm_counts[resource_index_prep_for_ul_ctrl_split]));
                tmpGreenContextsForResplit[0].create(gpuId, &devResources[resource_index_prep_for_ul_ctrl_split+1], print_resources);
                // Note tmpGreenContexts used only for resplit are created with defaults in cuPHY/examples/common/cuphy.hpp etc.

                // Now resplit the remaining split. It'll be {PUCCH, PRACH}

		// PUCCH
		unsigned int resource_index_pucch_split = 4;  //index into devResources
		actual_split_groups[resource_index_pucch_split] = 1;
		min_sm_counts[resource_index_pucch_split] = rounded_up_PUCCH;
                CUdevResource resource_to_split = {};
                tmpGreenContextsForResplit[0].getResources(&resource_to_split);
                CU_CHECK_PHYDRIVER(cuDevSmResourceSplitByCount(&devResources[resource_index_pucch_split], &actual_split_groups[resource_index_pucch_split], &resource_to_split, &devResources[resource_index_pucch_split+1], use_flags, min_sm_counts[resource_index_pucch_split]));

		pucchMpsCtx = new MpsCtx((phydriver_handle)this, gpu_device, &devResources[resource_index_pucch_split], "PUCCH", print_resources, use_workqueues, wq_concurrency_limit_of_1);
		mpsCtxList.push_back(pucchMpsCtx);
		NVLOGC_FMT(TAG, "PUCCH green context with SM count of {}.", devResources[resource_index_pucch_split].sm.smCount);

		prachMpsCtx = new MpsCtx((phydriver_handle)this, gpu_device, &devResources[resource_index_pucch_split + 1], "PRACH", print_resources, use_workqueues, wq_concurrency_limit_of_1);
		mpsCtxList.push_back(prachMpsCtx);
		NVLOGC_FMT(TAG, "PRACH green context with SM count of {}.", devResources[resource_index_pucch_split+1].sm.smCount);

		// PDSCH
		unsigned int resource_index_pdsch_split = 6;  //index into devResources
		actual_split_groups[resource_index_pdsch_split] = 1;
		min_sm_counts[resource_index_pdsch_split] = getMpsSmPdsch();
                int pdsch_split_modulo_SM_granularity = min_sm_counts[resource_index_pdsch_split] % SM_granularity;

                // Round up (default behavior of API) unless this would cause PDSCH to take all SMs
                if (pdsch_split_modulo_SM_granularity != 0) {
                    min_sm_counts[resource_index_pdsch_split] += (SM_granularity - pdsch_split_modulo_SM_granularity);
                    if (min_sm_counts[resource_index_pdsch_split] >= gpuMaxSmCount) {
                        min_sm_counts[resource_index_pdsch_split] -= SM_granularity;
                    }
                }


		CU_CHECK_PHYDRIVER(cuDevSmResourceSplitByCount(&devResources[resource_index_pdsch_split], &actual_split_groups[resource_index_pdsch_split], &initial_device_GPU_resources, &devResources[resource_index_pdsch_split+1], use_flags, min_sm_counts[resource_index_pdsch_split]));
		pdschMpsCtx = new MpsCtx((phydriver_handle)this, gpu_device, &devResources[resource_index_pdsch_split], "PDSCH", print_resources, use_workqueues, wq_concurrency_limit_of_2 + wq_concurrency_limit_mmimo_add_on);
		mpsCtxList.push_back(pdschMpsCtx);
		dlMpsCtx = pdschMpsCtx;
		NVLOGC_FMT(TAG, "PDSCH green context with SM count of {}.", devResources[resource_index_pdsch_split].sm.smCount);

                // Do another split of the initial GPU resources to get Dl Ctrl resource
		// Do getMpsSmDlCtrl() instead of separate for PDCCH and SSB and CSI-RS?
		unsigned int resource_index_dl_ctrl_split = 8;  //index into devResources
		actual_split_groups[resource_index_dl_ctrl_split] = 1;
		min_sm_counts[resource_index_dl_ctrl_split] = getMpsSmDlCtrl();
		CU_CHECK_PHYDRIVER(cuDevSmResourceSplitByCount(&devResources[resource_index_dl_ctrl_split], &actual_split_groups[resource_index_dl_ctrl_split], &initial_device_GPU_resources, &devResources[resource_index_dl_ctrl_split+1], use_flags, min_sm_counts[resource_index_dl_ctrl_split]));
		dlCtrlMpsCtx = new MpsCtx((phydriver_handle)this, gpu_device, &devResources[resource_index_dl_ctrl_split], "DL Ctrl", print_resources, use_workqueues, wq_concurrency_limit_of_2);
		mpsCtxList.push_back(dlCtrlMpsCtx);
		csiRsMpsCtx = dlCtrlMpsCtx;
		pbchMpsCtx  = dlCtrlMpsCtx;
		pdcchMpsCtx = dlCtrlMpsCtx;

		NVLOGC_FMT(TAG, "DL Ctrl green context with SM count of {}.", devResources[resource_index_dl_ctrl_split].sm.smCount);

		if(this->mMIMO_enable)
		{
			dlBfwMpsCtx = pdschMpsCtx;
			ulBfwMpsCtx = puschMpsCtx;
		}

		// UL order kernel overlap
		unsigned int resource_index_ul_order_split = 10;
		actual_split_groups[resource_index_ul_order_split] = 1;

#if 0
                // Limit overlap of order kernel SMs with UL control channels. Due to small number of SMs assigned on each channel (e.g., PUCCH, PRACH) if order kernel SMs almost fully overlap with
                // PUCCH's SM, then PUCCH for slot 4 could be delayed significantly.
                // FIXME this fix resolves the increased PUCCH time for Slot 4. However PUSCH GPU run for slot 4 still suffers.
		min_sm_counts[resource_index_ul_order_split] = getMpsSmUlOrder();
                CUdevResource pusch_split = {};
                puschMpsCtx->getResources(&pusch_split); // we need to get the resources from the green context we had created and resplit
		CU_CHECK_PHYDRIVER(cuDevSmResourceSplitByCount(&devResources[resource_index_ul_order_split], &actual_split_groups[resource_index_ul_order_split], &pusch_split, &devResources[resource_index_ul_order_split+1], use_flags, min_sm_counts[resource_index_ul_order_split]));
                ulMpsCtx = new MpsCtx((phydriver_handle)this, gpu_device, &devResources[resource_index_ul_order_split]);
		mpsCtxList.push_back(ulMpsCtx);
		NVLOGC_FMT(TAG, "UL order green context with SM count of {}.", devResources[resource_index_ul_order_split].sm.smCount);
#else
                // Based on the observation from #if 0 clause, try to not have order kernel overlap with any of PUSCH or UL control channel SMs
                tmpGreenContextsForResplit[2].create(gpuId, &devResources[resource_index_pusch_split], print_resources);
                CUdevResource complement_of_pusch_split = {};
                tmpGreenContextsForResplit[2].getResources(&complement_of_pusch_split); // we need to get the resources from the green context we had created and resplit
                // Add extra check in case PUSCH got almost all SMs which would leave 0 for that split. Could consider adding a temp. workaround in that case or simply ignoring yaml's SM alloc
		if (complement_of_pusch_split.sm.smCount <= getMpsSmUlOrder()){ // Exception!
			NVLOGC_FMT(TAG, "Doing an exceptional GC split for order kernel. You are advised to not assign that many SMs to PUSCH!");
			min_sm_counts[resource_index_ul_order_split] = getMpsSmUlOrder();
                        CU_CHECK_PHYDRIVER(cuDevSmResourceSplitByCount(&devResources[resource_index_ul_order_split], &actual_split_groups[resource_index_ul_order_split], &initial_device_GPU_resources, &devResources[resource_index_ul_order_split+1], use_flags, min_sm_counts[resource_index_ul_order_split]));
			ulMpsCtx = new MpsCtx((phydriver_handle)this, gpu_device, &devResources[resource_index_ul_order_split], "UL order", print_resources, use_workqueues, wq_concurrency_limit_of_1 + wq_concurrency_limit_mmimo_add_on);
			mpsCtxList.push_back(ulMpsCtx);
			NVLOGC_FMT(TAG, "UL order green context with SM count of {}.", devResources[resource_index_ul_order_split].sm.smCount);
                }
                else
		{
                    // Common case
			min_sm_counts[resource_index_ul_order_split] = complement_of_pusch_split.sm.smCount - getMpsSmUlOrder();
			//lower size to ensure we get at least needed SMs in remaining split
			int complement_split_modulo_SM_granularity = (min_sm_counts[resource_index_ul_order_split] % SM_granularity);
			if ((min_sm_counts[resource_index_ul_order_split] > SM_granularity) && (complement_split_modulo_SM_granularity != 0)) {
				min_sm_counts[resource_index_ul_order_split] -= complement_split_modulo_SM_granularity;
			}

			CU_CHECK_PHYDRIVER(cuDevSmResourceSplitByCount(&devResources[resource_index_ul_order_split], &actual_split_groups[resource_index_ul_order_split], &complement_of_pusch_split, &devResources[resource_index_ul_order_split+1], use_flags, min_sm_counts[resource_index_ul_order_split]));
			ulMpsCtx = new MpsCtx((phydriver_handle)this, gpu_device, &devResources[resource_index_ul_order_split+1], "UL order", print_resources, use_workqueues, wq_concurrency_limit_of_1 + wq_concurrency_limit_mmimo_add_on);
			mpsCtxList.push_back(ulMpsCtx);
			NVLOGC_FMT(TAG, "UL order green context with SM count of {}.", devResources[resource_index_ul_order_split+1].sm.smCount);
                }
#endif

		// GPU-comms
		if(gpuCommDlEnabled()) {
			unsigned int resource_index_gpu_comm_split = 12;
			actual_split_groups[resource_index_gpu_comm_split] = 1;
#if 1
			min_sm_counts[resource_index_gpu_comm_split] = getMpsSmGpuComms();
			CU_CHECK_PHYDRIVER(cuDevSmResourceSplitByCount(&devResources[resource_index_gpu_comm_split], &actual_split_groups[resource_index_gpu_comm_split], &initial_device_GPU_resources, &devResources[resource_index_gpu_comm_split+1], use_flags, min_sm_counts[resource_index_gpu_comm_split]));
			gpuCommsMpsCtx = new MpsCtx((phydriver_handle)this, gpu_device, &devResources[resource_index_gpu_comm_split], "GPU comm", print_resources, use_workqueues, wq_concurrency_limit_of_1);
			mpsCtxList.push_back(gpuCommsMpsCtx);

		        NVLOGC_FMT(TAG, "GPU comm green context with SM count of {}.", devResources[resource_index_gpu_comm_split].sm.smCount);
#else
                        // Force 0 overlap of DL SMs with gpu-comm SMs.. 
			min_sm_counts[resource_index_gpu_comm_split] = gpuMaxSmCount - getMpsSmGpuComms();
                        //FIXME check and round up?
			CU_CHECK_PHYDRIVER(cuDevSmResourceSplitByCount(&devResources[resource_index_gpu_comm_split], &actual_split_groups[resource_index_gpu_comm_split], &initial_device_GPU_resources, &devResources[resource_index_gpu_comm_split+1], use_flags, min_sm_counts[resource_index_gpu_comm_split]));
			gpuCommsMpsCtx = new MpsCtx((phydriver_handle)this, gpu_device, &devResources[resource_index_gpu_comm_split+1]);
			mpsCtxList.push_back(gpuCommsMpsCtx);

		        NVLOGC_FMT(TAG, "GPU comm green context with SM count of {}.", devResources[resource_index_gpu_comm_split+1].sm.smCount);
#endif
		}

		if(this->enable_srs) //Move SRS MPS context creation under enable srs flag for memory savings
		{

                    CUdevResource resource_to_split_for_srs = {};
                    puschMpsCtx->getResources(&resource_to_split_for_srs);

                    unsigned int resource_index_srs_split = 14;
		    actual_split_groups[resource_index_srs_split] = 1;
#if 1
		    min_sm_counts[resource_index_srs_split] = getMpsSmSrs();
		    CU_CHECK_PHYDRIVER(cuDevSmResourceSplitByCount(&devResources[resource_index_srs_split], &actual_split_groups[resource_index_srs_split], &resource_to_split_for_srs, &devResources[resource_index_srs_split+1], use_flags, min_sm_counts[resource_index_srs_split]));
			srsMpsCtx = new MpsCtx((phydriver_handle)this, gpu_device, &devResources[resource_index_srs_split], "SRS", print_resources, use_workqueues, wq_concurrency_limit_of_1);
			mpsCtxList.push_back(srsMpsCtx);

		        NVLOGC_FMT(TAG, "SRS green context with SM count of {}.", devResources[resource_index_srs_split].sm.smCount);
#else
                   // Experiment with the reverse split; this will cause SRS SM's to not overlap with PDSCH.
		   min_sm_counts[resource_index_srs_split] = devResources[resource_index_pusch_split+1].sm.smCount - getMpsSmSrs();
                   //lower size to ensure we get at least needed SMs in remaining split
		   int complement_split_modulo_SM_granularity = (min_sm_counts[resource_index_srs_split] % SM_granularity);
		   if ((min_sm_counts[resource_index_srs_split] > SM_granularity) && (complement_split_modulo_SM_granularity != 0)) {
			min_sm_counts[resource_index_srs_split] -= complement_split_modulo_SM_granularity;
	           }
		   CU_CHECK_PHYDRIVER(cuDevSmResourceSplitByCount(&devResources[resource_index_srs_split], &actual_split_groups[resource_index_srs_split], &resource_to_split_for_srs, &devResources[resource_index_srs_split+1], use_flags, min_sm_counts[resource_index_srs_split]));
		   srsMpsCtx = new MpsCtx((phydriver_handle)this, gpu_device, &devResources[resource_index_srs_split+1]);
		   mpsCtxList.push_back(srsMpsCtx);

		   NVLOGC_FMT(TAG, "SRS green context with SM count of {}.", devResources[resource_index_srs_split+1].sm.smCount);
#endif
                }
// Get a warning if the total of requested WQs exceeds CUDA_DEVICE_MAX_CONNECTIONS env. variable.
#if CUDA_VERSION >= 13010
                if(use_workqueues) {

                    CUdevResource initial_WQ_config_resources = {};
                    CU_CHECK_PHYDRIVER(cuDeviceGetDevResource(device,
                                                              &initial_WQ_config_resources,
                                                              CU_DEV_RESOURCE_TYPE_WORKQUEUE_CONFIG));
                    unsigned int initial_device_wqs = initial_WQ_config_resources.wqConfig.wqConcurrencyLimit; // should match CUDA_DEVICE_MAX_CONNECTIONS

                    // Sum up the requested wqConcurrencyLimits across all GCs with CU_WORKQUEUE_SCOPE_GREEN_CTX_BALANCED sharing scope,
                    // Also include tmp GCs used for resplits for completeness.
                    unsigned int requested_wq_count = 0;
                    for (auto gc : mpsCtxList) {
                        CUdevResource wq_config_resource{};
                        CU_CHECK_PHYDRIVER(cuGreenCtxGetDevResource(gc->getGreenCtx(), &wq_config_resource, CU_DEV_RESOURCE_TYPE_WORKQUEUE_CONFIG));
                        if(wq_config_resource.wqConfig.sharingScope == CU_WORKQUEUE_SCOPE_GREEN_CTX_BALANCED)
                        {
                            requested_wq_count += wq_config_resource.wqConfig.wqConcurrencyLimit;
                        }
                    }
                    for (int i = 0; i <= 2; i += 2)
                    {
                        CUdevResource wq_config_resource{};
                        tmpGreenContextsForResplit[i].getResources(&wq_config_resource, CU_DEV_RESOURCE_TYPE_WORKQUEUE_CONFIG);
                        if(wq_config_resource.wqConfig.sharingScope == CU_WORKQUEUE_SCOPE_GREEN_CTX_BALANCED)
                        {
                            requested_wq_count += wq_config_resource.wqConfig.wqConcurrencyLimit;
                        }
                    }
                    if(requested_wq_count > initial_device_wqs)
                    {
                        NVLOGW_FMT(TAG, "Total WQ count requested is {} but GPU's initial WQ concurrency limit is {} (CUDA_DEVICE_MAX_CONNECTIONS {}). There is a risk of aliasing depending on how these GCs are used!",
                                         requested_wq_count, initial_device_wqs, dev_max_connections_env_var ? dev_max_connections_env_var : "not set");
                    }
                }
#endif
#endif
        } else {

		puschMpsCtx = new MpsCtx((phydriver_handle)this, gpu_device, getMpsSmPusch());
		NVLOGC_FMT(TAG, "PUSCH MPS context with max. SM count of {}.", getMpsSmPusch());
		mpsCtxList.push_back(puschMpsCtx);
		pucchMpsCtx = new MpsCtx((phydriver_handle)this, gpu_device, getMpsSmPucch());
		NVLOGC_FMT(TAG, "PUCCH MPS context with max. SM count of {}.", getMpsSmPucch());
		mpsCtxList.push_back(pucchMpsCtx);
		prachMpsCtx = new MpsCtx((phydriver_handle)this, gpu_device, getMpsSmPrach());
		NVLOGC_FMT(TAG, "PRACH MPS context with max. SM count of {}.", getMpsSmPrach());
		mpsCtxList.push_back(prachMpsCtx);
		pdschMpsCtx = new MpsCtx((phydriver_handle)this, gpu_device, getMpsSmPdsch());
		NVLOGC_FMT(TAG, "PDSCH MPS context with max. SM count of {}.", getMpsSmPdsch());
		mpsCtxList.push_back(pdschMpsCtx);
		//pdcchMpsCtx = new MpsCtx((phydriver_handle)this, gpu_device, getMpsSmPdcch());
		//pbchMpsCtx = new MpsCtx((phydriver_handle)this, gpu_device, getMpsSmPbch());
		dlCtrlMpsCtx = new MpsCtx((phydriver_handle)this, gpu_device, getMpsSmDlCtrl());
		NVLOGC_FMT(TAG, "DL Ctrl MPS context with max. SM count of {}.", getMpsSmDlCtrl());
		mpsCtxList.push_back(dlCtrlMpsCtx);
		csiRsMpsCtx = dlCtrlMpsCtx;
		pbchMpsCtx  = dlCtrlMpsCtx;
		pdcchMpsCtx = dlCtrlMpsCtx;
		// If the cuphycontroller yaml config file does not contain the mps_sm_ul_order key, a default value of 16 SMs is used.
		ulMpsCtx = new MpsCtx((phydriver_handle)this, gpu_device, getMpsSmUlOrder());
		NVLOGC_FMT(TAG, "UL Order MPS context with max. SM count of {}.", getMpsSmUlOrder());
		mpsCtxList.push_back(ulMpsCtx);

		dlMpsCtx = pdschMpsCtx;
		if(this->enable_srs) //Move SRS MPS context creation under enable srs flag for memory savings
		{
			srsMpsCtx   = new MpsCtx((phydriver_handle)this, gpu_device, getMpsSmSrs());
		        NVLOGC_FMT(TAG, "SRS MPS context with max. SM count of {}.", getMpsSmSrs());
			mpsCtxList.push_back(srsMpsCtx);
		}
		if(this->mMIMO_enable)
		{
			dlBfwMpsCtx = pdschMpsCtx;
			ulBfwMpsCtx = puschMpsCtx;
		}

		if(gpuCommDlEnabled()) {
			// If the cuphycontroller yaml config file does not contain the mps_sm_gpu_comms key, the old default value of 8 SMs is used.
			gpuCommsMpsCtx = new MpsCtx((phydriver_handle)this, gpu_device, getMpsSmGpuComms());
		        NVLOGC_FMT(TAG, "GPU comm MPS context with max. SM count of {}.", getMpsSmGpuComms());
			mpsCtxList.push_back(gpuCommsMpsCtx);
		}
        }

        // Force loading of kernels for CUDA 13.0 (temp. workaround to avoid dyn. memory allocation during first kernel launch)
        // Not handling cases of templated kernels; for these mem. tracing is explicitly disabled
        const char* memtrace_env = std::getenv("AERIAL_MEMTRACE");
        if((memtrace_env != nullptr) && std::atoi(memtrace_env) == 1)
        {
            force_loading_generic_cuda_kernels();
            force_loading_order_kernels();
        }

        if(gpuCommDlEnabled())
        {
	    gpuCommsMpsCtx->setCtx();
	    CUDA_CHECK_PHYDRIVER(cudaEventCreateWithFlags(&gpu_comm_prepare_done, cudaEventDisableTiming));
        }

        ulMpsCtx->setCtx();
        CUDA_CHECK(cudaStreamCreateWithPriority(&stream_order_srs_pd, cudaStreamNonBlocking, -5));
        CUDA_CHECK(cudaStreamCreateWithPriority(&stream_order_pd, cudaStreamNonBlocking, -5));

        if(enable_ok_tb)
        {
            for (int cell_idx=0;cell_idx<UL_MAX_CELLS_PER_SLOT;cell_idx++)
            {
                CUDA_CHECK_PHYDRIVER(cudaMallocHost((void**)&fh_buf_ok_tb[cell_idx],MAX_PKTS_PER_SLOT_OK_TB*getConfigOkTbMaxPacketSize()*MAX_UL_SLOTS_OK_TB));
                CUDA_CHECK_PHYDRIVER(cudaMallocHost((void**)&config_ok_tb[cell_idx],sizeof(ok_tb_config_info_t)));
                CUDA_CHECK_PHYDRIVER(cudaMallocHost((void**)&fh_buf_ok_tb_srs[cell_idx],MAX_PKTS_PER_SLOT_OK_TB*getConfigOkTbMaxPacketSize()*MAX_UL_SLOTS_OK_TB));
                CUDA_CHECK_PHYDRIVER(cudaMallocHost((void**)&config_ok_tb_srs[cell_idx],sizeof(ok_tb_config_srs_info_t)));
            }
            setConfigOkTbNumSlots(0);
            setConfigOkTbSrsNumSlots(0);
        }

        puschMpsCtx->setCtx();
        CUDA_CHECK_PHYDRIVER(cudaStreamCreateWithPriority(&aggr_stream_pusch[PHASE1_SPLIT_STREAM1], cudaStreamNonBlocking, -2));
        warmupStream(aggr_stream_pusch[PHASE1_SPLIT_STREAM1]);
        CUDA_CHECK_PHYDRIVER(cudaStreamCreateWithPriority(&aggr_stream_pusch[PHASE2_SPLIT_STREAM1], cudaStreamNonBlocking, -2));
        warmupStream(aggr_stream_pusch[PHASE2_SPLIT_STREAM1]);
        if(splitUlCudaStreamsEnabled())
        {
            CUDA_CHECK_PHYDRIVER(cudaStreamCreateWithPriority(&aggr_stream_pusch[PHASE1_SPLIT_STREAM2], cudaStreamNonBlocking, 0));
            warmupStream(aggr_stream_pusch[PHASE1_SPLIT_STREAM2]);
            CUDA_CHECK_PHYDRIVER(cudaStreamCreateWithPriority(&aggr_stream_pusch[PHASE2_SPLIT_STREAM2], cudaStreamNonBlocking, 0));
            warmupStream(aggr_stream_pusch[PHASE2_SPLIT_STREAM2]);
        }
        aggr_last_pusch = 0;

        for(int i = 0; i < getPuschAggrPerCtx(); i++)
            aggr_pusch_items.push_back(std::unique_ptr<PhyPuschAggr>(new PhyPuschAggr((phydriver_handle)this, gpu_device, aggr_stream_pusch, puschMpsCtx)));


        if(this->mMIMO_enable)
        {
            ulBfwMpsCtx->setCtx();
            CUDA_CHECK_PHYDRIVER(cudaStreamCreateWithPriority(&aggr_stream_ulbfw, cudaStreamNonBlocking, -2));
            aggr_last_ulbfw = 0;
            for(int i = 0; i < getUlbfwAggrPerCtx(); i++)
                aggr_ulbfw_items.push_back(std::unique_ptr<PhyUlBfwAggr>(new PhyUlBfwAggr((phydriver_handle)this, gpu_device, aggr_stream_ulbfw, ulBfwMpsCtx)));

            for(int i=0;i<SLOTS_PER_FRAME;i++)
            {
                CUDA_CHECK(cudaEventCreateWithFlags(&ulbfw_run_completion_event[i], cudaEventDisableTiming));
            }
        }

        pucchMpsCtx->setCtx();
        CUDA_CHECK_PHYDRIVER(cudaStreamCreateWithPriority(&aggr_stream_pucch[0], cudaStreamNonBlocking, -3));
        warmupStream(aggr_stream_pucch[0]);

        if(splitUlCudaStreamsEnabled())
        {
            CUDA_CHECK_PHYDRIVER(cudaStreamCreateWithPriority(&aggr_stream_pucch[1], cudaStreamNonBlocking, -1));
            warmupStream(aggr_stream_pucch[1]);
        }
        aggr_last_pucch = 0;
        for(int i = 0; i < getPucchAggrPerCtx(); i++)
            aggr_pucch_items.push_back(std::unique_ptr<PhyPucchAggr>(new PhyPucchAggr((phydriver_handle)this, gpu_device, aggr_stream_pucch, pucchMpsCtx)));

        prachMpsCtx->setCtx();
        CUDA_CHECK_PHYDRIVER(cudaStreamCreateWithPriority(&aggr_stream_prach[0], cudaStreamNonBlocking, -3));
        warmupStream(aggr_stream_prach[0]);

        if(splitUlCudaStreamsEnabled())
        {
            CUDA_CHECK_PHYDRIVER(cudaStreamCreateWithPriority(&aggr_stream_prach[1], cudaStreamNonBlocking, -1));
            warmupStream(aggr_stream_prach[1]);
        }

        aggr_last_prach = 0;
        for(int i = 0; i < getPrachAggrPerCtx(); i++)
            aggr_prach_items.push_back(std::unique_ptr<PhyPrachAggr>(new PhyPrachAggr((phydriver_handle)this, gpu_device, aggr_stream_prach, prachMpsCtx)));

        if(this->enable_srs) //Move SRS Phy aggregate object creation under enable srs flag for memory savings
        {
            srsMpsCtx->setCtx();
            CUDA_CHECK_PHYDRIVER(cudaStreamCreateWithPriority(&aggr_stream_srs, cudaStreamNonBlocking, -2));
            aggr_last_srs = 0;
            for(int i = 0; i < getSrsAggrPerCtx(); i++)
                aggr_srs_items.push_back(std::unique_ptr<PhySrsAggr>(new PhySrsAggr((phydriver_handle)this, gpu_device, aggr_stream_srs, srsMpsCtx)));

        }

        pdschMpsCtx->setCtx();
        CUDA_CHECK_PHYDRIVER(cudaStreamCreateWithPriority(&aggr_stream_pdsch, cudaStreamNonBlocking, -4));
        CUDA_CHECK_PHYDRIVER(cudaStreamCreateWithPriority(&H2D_TB_CPY_stream, cudaStreamNonBlocking, -4));
        aggr_last_pdsch = 0;
        for(int i = 0; i < PHY_PDSCH_AGGR_X_CTX; i++)
            aggr_pdsch_items.push_back(std::unique_ptr<PhyPdschAggr>(new PhyPdschAggr((phydriver_handle)this, gpu_device, aggr_stream_pdsch, pdschMpsCtx)));

        // Allocate helper buffers used to memset DL cells' output buffers
        h_dl_buffers_addr = cuphy::make_unique_pinned<CleanupDlBufInfo>(PDSCH_MAX_CELLS_PER_CELL_GROUP * DL_HELPER_MEMSET_BUFFERS_PER_CTX);
        mf.addCpuPinnedSize(sizeof(CleanupDlBufInfo) * PDSCH_MAX_CELLS_PER_CELL_GROUP * DL_HELPER_MEMSET_BUFFERS_PER_CTX);
        d_dl_buffers_addr = cuphy::make_unique_device<CleanupDlBufInfo>(PDSCH_MAX_CELLS_PER_CELL_GROUP * DL_HELPER_MEMSET_BUFFERS_PER_CTX);
        mf.addGpuRegularSize(sizeof(CleanupDlBufInfo) * PDSCH_MAX_CELLS_PER_CELL_GROUP * DL_HELPER_MEMSET_BUFFERS_PER_CTX);

        for(int i=0;i<MAX_PDSCH_TB_CPY_CUDA_EVENTS;i++)
        {
            CUDA_CHECK(cudaEventCreate(&pdsch_tb_cpy_start[i]));
            CUDA_CHECK(cudaEventCreate(&pdsch_tb_cpy_complete[i]));
        }

        if(this->mMIMO_enable)
        {
            // Set the DL BFW stream priority lower (-3) than  the PDSCH stream (priority -4),
            // to prioritize PDSCH whenever possible. Both streams are under the same MPS/Green context
            CUDA_CHECK_PHYDRIVER(cudaStreamCreateWithPriority(&aggr_stream_dlbfw, cudaStreamNonBlocking, -3));
            aggr_last_dlbfw = 0;
            for(int i = 0; i < PHY_DLBFW_AGGR_X_CTX; i++)
                aggr_dlbfw_items.push_back(std::unique_ptr<PhyDlBfwAggr>(new PhyDlBfwAggr((phydriver_handle)this, gpu_device, aggr_stream_dlbfw, dlBfwMpsCtx)));

            for(int i=0;i<SLOTS_PER_FRAME;i++)
            {
                CUDA_CHECK(cudaEventCreateWithFlags(&dlbfw_run_completion_event[i], cudaEventDisableTiming));
            }
        }

        pdcchMpsCtx->setCtx();
        CUDA_CHECK_PHYDRIVER(cudaStreamCreateWithPriority(&aggr_stream_pdcch, cudaStreamNonBlocking, -4));

        aggr_last_pdcch_dl = 0;
        for(int i = 0; i < PHY_PDCCH_DL_AGGR_X_CTX; i++)
            aggr_pdcch_dl_items.push_back(std::unique_ptr<PhyPdcchAggr>(new PhyPdcchAggr((phydriver_handle)this, gpu_device, aggr_stream_pdcch, pdcchMpsCtx, slot_command_api::channel_type::PDCCH_DL)));
        aggr_last_pdcch_ul = 0;
        for(int i = 0; i < PHY_PDCCH_UL_AGGR_X_CTX; i++)
            aggr_pdcch_ul_items.push_back(std::unique_ptr<PhyPdcchAggr>(new PhyPdcchAggr((phydriver_handle)this, gpu_device, aggr_stream_pdcch, pdcchMpsCtx, slot_command_api::channel_type::PDCCH_UL)));
        pbchMpsCtx->setCtx();
        CUDA_CHECK_PHYDRIVER(cudaStreamCreateWithPriority(&aggr_stream_pbch, cudaStreamNonBlocking, -4));

        aggr_last_pbch = 0;
        for(int i = 0; i < PHY_PBCH_AGGR_X_CTX; i++)
            aggr_pbch_items.push_back(std::unique_ptr<PhyPbchAggr>(new PhyPbchAggr((phydriver_handle)this, gpu_device, aggr_stream_pbch, pbchMpsCtx)));
        //Using same context as PDSCH
        csiRsMpsCtx->setCtx();
        CUDA_CHECK_PHYDRIVER(cudaStreamCreateWithPriority(&aggr_stream_csirs, cudaStreamNonBlocking, -4));
        aggr_last_csirs = 0;
        for(int i = 0; i < PHY_CSIRS_AGGR_X_CTX; i++)
            aggr_csirs_items.push_back(std::unique_ptr<PhyCsiRsAggr>(new PhyCsiRsAggr((phydriver_handle)this, gpu_device, aggr_stream_csirs, csiRsMpsCtx)));


        num_new_prach_handles = 0;

    }

    dlMpsCtx->setCtx();
    CUDA_CHECK_PHYDRIVER(cudaStreamCreateWithPriority(&stream_timing_dl, cudaStreamNonBlocking, -1));
    ulMpsCtx->setCtx();
    CUDA_CHECK_PHYDRIVER(cudaStreamCreateWithPriority(&stream_timing_ul, cudaStreamNonBlocking, -1));

    /* Internal GPU-init comm will be created in default DL ctx */
    if(gpuCommDlEnabled())
        setGpuCommsCtx();

    fh_proxy = std::make_unique<FhProxy>((phydriver_handle)this,ctx_cfg);

    for (auto nic_cfg : ctx_cfg.nic_configs) {
        if (fh_proxy->registerNic(nic_cfg, ctx_cfg.gpu_id))
            PHYDRIVER_THROW_EXCEPTIONS(EINVAL, "NIC registration error");
    }
    setDlCtx();
    updateCellConfigCellId = -1;
    active = false;
    num_pdsch_buff_copy = 0;
    enable_prepone_h2d_cpy = false;
    h2d_write_idx=0;
    h2d_read_idx=0;

    std::fill(h2d_copy_done_cur_slot_idx.begin(), h2d_copy_done_cur_slot_idx.end(), -1);
    std::fill(h2d_copy_cuda_event_rec_done.begin(), h2d_copy_cuda_event_rec_done.end(), false);

    h2d_copy_done_cur_slot_read_idx = 0;
    h2d_copy_done_cur_slot_write_idx = 0;
    reset_h2d_copy_prepone_info();

    if(ctx_cfg.h2d_cpy_th_cfg.enable_h2d_copy_thread)
    {
        //Create h2d copy prepone thread
        std::thread t;
        t = std::thread(&l1_copy_TB_to_gpu_buf_thread_func, (void*)this);
        h2d_cpy_thread.swap(t);

        int name_st = pthread_setname_np(h2d_cpy_thread.native_handle(), "h2dcpy_thread");

        if (name_st != 0 )
        {
            NVLOGE_FMT(TAG, AERIAL_THREAD_API_EVENT ,"h2d_prepone_cpy_thread Thread pthread_setname_np failed with status: {}",std::strerror(name_st));
        }

        sched_param sch;
        int         policy;
        int         status = 0;
        //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -

        //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
        // Set thread CPU affinity
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(ctx_cfg.h2d_cpy_th_cfg.h2d_copy_thread_cpu_affinity, &cpuset);
        status = pthread_setaffinity_np(h2d_cpy_thread.native_handle(), sizeof(cpu_set_t), &cpuset);
        if(status)
        {
            NVLOGE_FMT(TAG, AERIAL_THREAD_API_EVENT, "h2d_prepone_cpy_thread setaffinity_np  failed with status : {}" , std::strerror(status));
        }

        if(ctx_cfg.h2d_cpy_th_cfg.h2d_copy_thread_sched_priority>0)
        {
            // Set thread priority
            status = pthread_getschedparam(h2d_cpy_thread.native_handle(), &policy, &sch);
            if(status != 0)
            {
                NVLOGE_FMT(TAG, AERIAL_THREAD_API_EVENT, "h2d_prepone_cpy_thread pthread_getschedparam failed with status : {}", std::strerror(status));
            }
            sch.sched_priority = ctx_cfg.h2d_cpy_th_cfg.h2d_copy_thread_sched_priority;

#ifdef ENABLE_SCHED_FIFO_ALL_RT
            status = pthread_setschedparam(h2d_cpy_thread.native_handle(), SCHED_FIFO, &sch);
            if(status != 0)
            {
                NVLOGE_FMT(TAG, AERIAL_THREAD_API_EVENT, "h2d_prepone_cpy_thread setschedparam failed with status : {}" , std::strerror(status));
            }
#endif
        }
    }

    if(ctx_cfg.ul_pcap_capture_enable)
    {
        CuphyOAM *oam = CuphyOAM::getInstance();
        oam->ul_pcap_enabled.store(true);
        launch_ul_capture_thread();
    }

    // Pcap thread initialization.
    PcapLoggerCfg cfg;
    cfg.enableDlUplane  = false; 
    cfg.enableDlCplane  = ctx_cfg.pcap_logger_dl_cplane_enable;
    cfg.enableUlCplane  = ctx_cfg.pcap_logger_ul_cplane_enable;
    cfg.output_path     = ctx_cfg.pcap_logger_file_save_dir;
    cfg.threadAffinity  = ctx_cfg.pcap_logger_thread_cpu_affinity;
    cfg.threadPriority  = ctx_cfg.pcap_logger_thread_sched_prio;

    PcapLogger::instance().init(cfg);
    PcapLogger::instance().start();

    task_list_dl->initListWithReserveSize(TASK_LIST_RESERVE_LENGTH, TASK_LIST_NUM_QUEUES);
    task_list_dl_validation->initListWithReserveSize(TASK_LIST_RESERVE_LENGTH, TASK_LIST_NUM_QUEUES);
    task_list_ul->initListWithReserveSize(TASK_LIST_RESERVE_LENGTH, TASK_LIST_NUM_QUEUES);
    task_list_debug->initListWithReserveSize(TASK_LIST_RESERVE_LENGTH, TASK_LIST_NUM_QUEUES);

    //If UE Mode is set, start up DL Validation persistent tasks 
    if (workers_dl_validation_cores.size() > 0) {
        exit_dl_validation.store(false);
        int cells_per_core = cell_group_num / workers_dl_validation_cores.size();
        int remainder = cell_group_num - cells_per_core * workers_dl_validation_cores.size();
        int start_cell = 0;
        for(int i = 0; i < workers_dl_validation_cores.size(); i++)
        {
            int num_cells = cells_per_core;
            if(i < remainder)
                num_cells++;
            workers_dl_validation_params.emplace_back(this, start_cell, num_cells);
            start_cell += num_cells;
        }
    }

    // Size of workers_dl_validation_params will be 0 when UE mode is set to 0, it will only be enabled for UE mode.
    for(int i = 0; i < workers_dl_validation_params.size(); i++)
    {
        if(workers_dl_validation_params[i].getNumCells() > 0)
        {
            t_ns start_ts(std::chrono::system_clock::now().time_since_epoch().count());
            auto dl_task = getNextTask();
            // NVLOGC_FMT(TAG, "workers_dl_validation_params[i] {}, pdctx {}", (void*)(&workers_dl_validation_params[i]), (void*)(workers_dl_validation_params[i].getPhyDriverHandler()));
            for(int cell_index = workers_dl_validation_params[i].getStartCell(); cell_index < workers_dl_validation_params[i].getStartCell() + workers_dl_validation_params[i].getNumCells(); ++cell_index)
            {
                workers_dl_validation_params[i].cell_T1a_max_cp_ul_ns[cell_index] = ctx_cfg.cell_mplane_list[cell_index].t1a_max_cp_ul_ns;
                workers_dl_validation_params[i].cell_T1a_max_up_ns[cell_index] = ctx_cfg.cell_mplane_list[cell_index].t1a_max_up_ns;
                workers_dl_validation_params[i].cell_Tcp_adv_dl_ns[cell_index] = ctx_cfg.cell_mplane_list[cell_index].tcp_adv_dl_ns;
            }
            dl_task->init(start_ts, "DLVal", task_work_function_dl_validation, (void*)(&workers_dl_validation_params[i]), i, 0, 0, 0);
            auto dl = getTaskListDlVal();
            dl->lock();
            dl->push(dl_task);
            dl->unlock();
        }
    }
}

PhyDriverCtx::~PhyDriverCtx()
{
    try {
        printf("PhyDriverCtx destructor starting\n");
        NVLOGC_FMT(TAG, "PhyDriverCtx destructor starting");
        if (minimal_phydriver) {
            try {
                fh_proxy.reset();
            } catch(const std::exception& e) {
                printf("EXCEPTION in fh_proxy.reset(): %s\n", e.what());
                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "EXCEPTION in fh_proxy.reset(): {}", e.what());
            }
            return;
        }

        exit_dl_validation.store(true);
        
        // Log Order Kernel timing statistics per slot
        NVLOGC_FMT(TAG_PERF_METRICS, "Order Kernel Completion Statistics (per slot):");
        char logLabel[32];
        snprintf(logLabel, sizeof(logLabel), "Order Kernel Completion");
        for (int slotIdx = 0; slotIdx < 80; slotIdx++) {
            
            
            int64_t count = order_kernel_timing_tracker.getTotalCount(slotIdx);
            if (count > 0) {
                order_kernel_timing_tracker.logStats<TAG_PERF_METRICS>(1, logLabel, slotIdx);
            }
        }
        // Log combined statistics across all slots
        order_kernel_timing_tracker.logStats<TAG_PERF_METRICS>(1, logLabel, -1);

        if(ul_stats.active())
        {
            ul_stats.flush_counters(cell_group_num);
        }
        // Keep file because CICD uses it even if it is empty
        ul_stats.flush_counters_file(cell_group_num, std::string("/tmp/ul_packet_times.txt"));

        if(srs_stats.active())
        {
            srs_stats.flush_counters(cell_group_num,true);
        }
        srs_stats.flush_counters_file(cell_group_num, std::string("/tmp/srs_packet_times.txt"));

        for(int type = 0; type < Packet_Statistics::MAX_DL_PACKET_TYPES; ++type)
        {
            if(dl_stats[type].active())
            {
                if(type == Packet_Statistics::DLC)
                {
                    NVLOGC_FMT(TAG,"DL C Stats");
                }
                else if(type == Packet_Statistics::DLU)
                {
                    NVLOGC_FMT(TAG,"DL U Stats");
                }
                else
                {
                    NVLOGC_FMT(TAG,"UL C Stats");
                }
                dl_stats[type].flush_counters(cell_group_num);
            }
        }

        printf("flushing counters completed\n");
        NVLOGC_FMT(TAG, "flushing counters completed");

        t_ns timeout_close(FINALIZE_TIMEOUT_NS);
        MemFoot mf_acc;
        mf_acc.init((phydriver_handle)this, std::string("Accumulator"), sizeof(MemFoot));
        mf_acc.reset();

        if(h2d_copy_thread_enable) // h2d_copy_thread_enable is reset later in the destructor too
        {
            try {
                h2d_cpy_thread.detach();
            } catch(const std::exception& e) {
                printf("EXCEPTION detaching h2d_cpy_thread: %s\n", e.what());
                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "EXCEPTION detaching h2d_cpy_thread: {}", e.what());
            }
        }

        if(ul_pcap_capture_enable)
        {
            stop_ul_pcap_thread.store(true);
            try {
                ul_pcap_capture_thread.detach();
            } catch(const std::exception& e) {
                printf("EXCEPTION detaching ul_pcap_capture_thread: %s\n", e.what());
                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "EXCEPTION detaching ul_pcap_capture_thread: {}", e.what());
            }
        }

        // Stop (any) PCAP logging thread instance.
        try {
            PcapLogger::instance().stop();
        } catch(const std::exception& e) {
            printf("EXCEPTION in PcapLogger::stop(): %s\n", e.what());
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "EXCEPTION in PcapLogger::stop(): {}", e.what());
        }

        /*
        * Everytime in removeCell the cell pointed by it is removed
        */
        for(auto& it : cell_map)
        {
            mf_acc.addMF(it.second->mf);
            it.second->stop();
        }
        wip_accum_mf.addMF(mf_acc);
        mf_acc.reset();
        Time::waitDurationNs(timeout_close);

        if(enableOKTb())
        {
            NVLOGI_FMT(TAG,"[Phy driver context destructor] OK TB enabled");
            for (int cell_idx=0;cell_idx<getCellNum();cell_idx++)
            {
                NVLOGI_FMT(TAG,"[Phy driver context destructor]Dumping binary file for cell index {}",cell_idx);
                if(getConfigOkTbNumSlots() > 0)
                {
                    std::string filename = "/tmp/fh_buf_ok_tb_cell_" + std::to_string(cell_idx) + ".bin";
                    std::ofstream outFile(filename, std::ios::binary);
                    outFile.write(reinterpret_cast<const char*>(fh_buf_ok_tb[cell_idx]), MAX_PKTS_PER_SLOT_OK_TB*getConfigOkTbMaxPacketSize()*MAX_UL_SLOTS_OK_TB);
                    outFile.close();
                    cudaFreeHost(fh_buf_ok_tb[cell_idx]);
                }
                if(getConfigOkTbSrsNumSlots() > 0)
                {
                    std::string filename = "/tmp/fh_buf_ok_tb_srs_cell_" + std::to_string(cell_idx) + ".bin";
                    std::ofstream outFile(filename, std::ios::binary);
                    outFile.write(reinterpret_cast<const char*>(fh_buf_ok_tb_srs[cell_idx]), MAX_PKTS_PER_SLOT_OK_TB*getConfigOkTbMaxPacketSize()*MAX_UL_SLOTS_OK_TB);
                    outFile.close();
                    cudaFreeHost(fh_buf_ok_tb_srs[cell_idx]);
                }
            }
            if(getConfigOkTbNumSlots() > 0)
            {
                std::ofstream configFile("/tmp/ok_tb_config_params.txt");
                configFile<<"num_valid_slots : "<<std::to_string(getConfigOkTbNumSlots())<<"\n";
                configFile<<"max_packet_size : "<<std::to_string(getConfigOkTbMaxPacketSize())<<"\n";
                for(int slot_idx=0;slot_idx<MAX_UL_SLOTS_OK_TB;slot_idx++)
                {
                    std::string slot_name = "slot_"+std::to_string(slot_idx);
                    configFile<<slot_name<<":\n";
                    configFile<<"   frame : "<<std::to_string(config_ok_tb[0]->frameId[slot_idx])<<"\n";
                    configFile<<"   subframe : "<<std::to_string(config_ok_tb[0]->subframeId[slot_idx])<<"\n";
                    configFile<<"   slot : "<<std::to_string(config_ok_tb[0]->slotId[slot_idx])<<"\n";
                    configFile<<"   num_order_cells_sym_mask : ";
                    for(int tmp=0;tmp<ORAN_PUSCH_SYMBOLS_X_SLOT;tmp++)
                    {
                        configFile<<config_ok_tb[0]->num_order_cells_sym_mask[slot_idx][tmp]<<" ";
                    }
                    configFile<<"\n";
                    for(int cell_idx=0;cell_idx<getCellNum();cell_idx++)
                    {
                        std::string cell_name = "cell_"+std::to_string(cell_idx);
                        configFile<<"   "<<cell_name<<":\n";
                        configFile<<"       cell_id : "<<config_ok_tb[cell_idx]->cell_id<<"\n";
                        configFile<<"       num_rx_packets : "<<config_ok_tb[cell_idx]->num_rx_packets[slot_idx]<<"\n";
                        configFile<<"       num_pusch_prbs : "<<config_ok_tb[cell_idx]->num_pusch_prbs[slot_idx]<<"\n";
                        configFile<<"       num_prach_prbs : "<<config_ok_tb[cell_idx]->num_prach_prbs[slot_idx]<<"\n";
                        configFile<<"       pusch_eAxC_num : "<<config_ok_tb[cell_idx]->pusch_eAxC_num<<"\n";
                        configFile<<"       pusch_eAxC_map : ";
                        for(int tmp=0;tmp<config_ok_tb[cell_idx]->pusch_eAxC_num;tmp++)
                        {
                            configFile<<config_ok_tb[cell_idx]->pusch_eAxC_map[tmp]<<" ";
                        }
                        configFile<<"\n";
                        configFile<<"       prach_eAxC_num : "<<config_ok_tb[cell_idx]->prach_eAxC_num<<"\n";
                        configFile<<"       prach_eAxC_map : ";
                        for(int tmp=0;tmp<config_ok_tb[cell_idx]->prach_eAxC_num;tmp++)
                        {
                            configFile<<config_ok_tb[cell_idx]->prach_eAxC_map[tmp]<<" ";
                        }
                        configFile<<"\n";
                        configFile<<"       pusch_prb_symbol_map : ";
                        for(int tmp=0;tmp<ORAN_PUSCH_SYMBOLS_X_SLOT;tmp++)
                        {
                            configFile<<config_ok_tb[cell_idx]->pusch_prb_symbol_map[slot_idx][tmp]<<" ";
                        }
                        configFile<<"\n";
                    }
                }
                configFile.close();
            }
            if(getConfigOkTbSrsNumSlots() > 0)
            {
                std::ofstream configFile("/tmp/ok_tb_config_params_srs.txt");
                configFile<<"num_valid_slots : "<<std::to_string(getConfigOkTbSrsNumSlots())<<"\n";
                configFile<<"max_packet_size : "<<std::to_string(getConfigOkTbMaxPacketSize())<<"\n";
                for(int slot_idx=0;slot_idx<MAX_UL_SLOTS_OK_TB;slot_idx++)
                {
                    std::string slot_name = "slot_"+std::to_string(slot_idx);
                    configFile<<slot_name<<":\n";
                    configFile<<"   frame : "<<std::to_string(config_ok_tb_srs[0]->frameId[slot_idx])<<"\n";
                    configFile<<"   subframe : "<<std::to_string(config_ok_tb_srs[0]->subframeId[slot_idx])<<"\n";
                    configFile<<"   slot : "<<std::to_string(config_ok_tb_srs[0]->slotId[slot_idx])<<"\n";
                    for(int cell_idx=0;cell_idx<getCellNum();cell_idx++)
                    {
                        std::string cell_name = "cell_"+std::to_string(cell_idx);
                        configFile<<"   "<<cell_name<<":\n";
                        configFile<<"       cell_id : "<<config_ok_tb_srs[cell_idx]->cell_id<<"\n";
                        configFile<<"       num_rx_packets : "<<config_ok_tb_srs[cell_idx]->num_rx_packets[slot_idx]<<"\n";
                        configFile<<"       num_srs_prbs : "<<config_ok_tb_srs[cell_idx]->num_srs_prbs[slot_idx]<<"\n";
                        configFile<<"       srs_eAxC_num : "<<config_ok_tb_srs[cell_idx]->srs_eAxC_num<<"\n";
                        configFile<<"       srs_eAxC_map : ";
                        for(int tmp=0;tmp<config_ok_tb_srs[cell_idx]->srs_eAxC_num;tmp++)
                        {
                            configFile<<config_ok_tb_srs[cell_idx]->srs_eAxC_map[tmp]<<" ";
                        }
                        configFile<<"\n";
                    }
                }
                configFile.close();
            }
        }

        NVLOGI_FMT(TAG, "Cells timeout");

        try {
            cell_map.clear();
        } catch(const std::exception& e) {
            printf("EXCEPTION in cell_map.clear(): %s\n", e.what());
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "EXCEPTION in cell_map.clear(): {}", e.what());
        }

        NVLOGI_FMT(TAG, "Cells destroyed");

        for (int i = 0; i < aggr_ulbfw_items.size(); i++) {
        wip_accum_mf.addMF(aggr_ulbfw_items[i]->mf);
        }
        aggr_ulbfw_items.clear();
        for (int i = 0; i < aggr_pusch_items.size(); i++) {
        wip_accum_mf.addMF(aggr_pusch_items[i]->mf);
        }
        aggr_pusch_items.clear();
        for (int i = 0; i < aggr_pucch_items.size(); i++) {
        wip_accum_mf.addMF(aggr_pucch_items[i]->mf);
        }
        aggr_pucch_items.clear();
        for (int i = 0; i < aggr_prach_items.size(); i++) {
        wip_accum_mf.addMF(aggr_prach_items[i]->mf);
        }
        aggr_prach_items.clear();
        for (int i = 0; i < aggr_srs_items.size(); i++) {
        wip_accum_mf.addMF(aggr_srs_items[i]->mf);
        }
        aggr_srs_items.clear();
        for (int i = 0; i < aggr_dlbfw_items.size(); i++) {
        wip_accum_mf.addMF(aggr_dlbfw_items[i]->mf);
        }
        aggr_dlbfw_items.clear();
        for (int i = 0; i < aggr_pdsch_items.size(); i++) {
        wip_accum_mf.addMF(aggr_pdsch_items[i]->mf);
        }
        aggr_pdsch_items.clear();
        for (int i = 0; i < aggr_pdcch_dl_items.size(); i++) {
        wip_accum_mf.addMF(aggr_pdcch_dl_items[i]->mf);
        }
        aggr_pdcch_dl_items.clear();
        for (int i = 0; i < aggr_pdcch_ul_items.size(); i++) {
        wip_accum_mf.addMF(aggr_pdcch_ul_items[i]->mf);
        }
        aggr_pdcch_ul_items.clear();
        for (int i = 0; i < aggr_pbch_items.size(); i++) {
        wip_accum_mf.addMF(aggr_pbch_items[i]->mf);
        }
        aggr_pbch_items.clear();
        for (int i = 0; i < aggr_csirs_items.size(); i++) {
        wip_accum_mf.addMF(aggr_csirs_items[i]->mf);
        }
        aggr_csirs_items.clear();

        NVLOGI_FMT(TAG, "Aggr items destroyed");

        try {
            cudaStreamDestroy(stream_order_pd);
            if(this->enable_srs)
            {
                cudaStreamDestroy(aggr_stream_srs);
            }
            if(this->mMIMO_enable)
            {
                cudaStreamDestroy(aggr_stream_ulbfw);
                cudaStreamDestroy(aggr_stream_dlbfw);
            }
            cudaStreamDestroy(stream_order_srs_pd);
            cudaStreamDestroy(aggr_stream_pusch[PHASE1_SPLIT_STREAM1]);
            cudaStreamDestroy(aggr_stream_pusch[PHASE2_SPLIT_STREAM1]);
            cudaStreamDestroy(aggr_stream_pucch[0]);
            cudaStreamDestroy(aggr_stream_prach[0]);

            if(splitUlCudaStreamsEnabled())
            {
                cudaStreamDestroy(aggr_stream_pusch[PHASE1_SPLIT_STREAM2]);
                cudaStreamDestroy(aggr_stream_pusch[PHASE2_SPLIT_STREAM2]);
                cudaStreamDestroy(aggr_stream_pucch[1]);
                cudaStreamDestroy(aggr_stream_prach[1]);
            }

            cudaStreamDestroy(aggr_stream_pdsch);
            cudaStreamDestroy(aggr_stream_pdcch);
            cudaStreamDestroy(aggr_stream_pbch);
            cudaStreamDestroy(aggr_stream_csirs);
            cudaStreamDestroy(H2D_TB_CPY_stream);

            cudaStreamDestroy(stream_timing_ul);
            cudaStreamDestroy(stream_timing_dl);

            NVLOGI_FMT(TAG, "Aggr streams destroyed");
        } catch(const std::exception& e) {
            printf("EXCEPTION in cudaStreamDestroy: %s\n", e.what());
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "EXCEPTION in cudaStreamDestroy: {}", e.what());
        }

        try {
            for(int i = 0; i < MAX_PDSCH_TB_CPY_CUDA_EVENTS; i++) {
                CUDA_CHECK_PHYDRIVER(cudaEventDestroy(pdsch_tb_cpy_start[i]));
                CUDA_CHECK_PHYDRIVER(cudaEventDestroy(pdsch_tb_cpy_complete[i]));
            }
            NVLOGI_FMT(TAG, "PDSCH TB copy events destroyed");
        } catch(const std::exception& e) {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "EXCEPTION in cudaEventDestroy for pdsch_tb_cpy events: {}", e.what());
        }

        /*
        * Ensure FH termination
        */
        fh_proxy->mf.printMemoryFootprint();
        wip_accum_mf.addMF(fh_proxy->mf);
        try {
            fh_proxy.reset();
        } catch(const std::exception& e) {
            printf("EXCEPTION in fh_proxy cleanup: %s\n", e.what());
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "EXCEPTION in fh_proxy cleanup: {}", e.what());
        }
        NVLOGD_FMT(TAG, "FH destroyed");

        if(worker_ul_map.size() > 0)
        {
            worker_ul_map.begin()->second->mf.printMemoryFootprint();
            for (auto it=worker_ul_map.begin(); it!=worker_ul_map.end(); ++it)
                mf_acc.addMF(it->second->mf);
            mf_acc.printMemoryFootprint();
            wip_accum_mf.addMF(mf_acc);
            mf_acc.reset();
        }

        if(worker_dl_map.size() > 0)
        {
            worker_dl_map.begin()->second->mf.printMemoryFootprint();
            for (auto it=worker_dl_map.begin(); it!=worker_dl_map.end(); ++it)
                mf_acc.addMF(it->second->mf);
            mf_acc.printMemoryFootprint();
            wip_accum_mf.addMF(mf_acc);
            mf_acc.reset();
        }

        if(worker_dl_validation_map.size() > 0)
        {
            worker_dl_validation_map.begin()->second->mf.printMemoryFootprint();
            for (auto it=worker_dl_validation_map.begin(); it!=worker_dl_validation_map.end(); ++it)
                mf_acc.addMF(it->second->mf);
            mf_acc.printMemoryFootprint();
            wip_accum_mf.addMF(mf_acc);
            mf_acc.reset();
        }


        if(worker_generic_map.size() > 0)
        {
            worker_generic_map.begin()->second->mf.printMemoryFootprint();
            for (auto it=worker_generic_map.begin(); it!=worker_generic_map.end(); ++it)
                mf_acc.addMF(it->second->mf);
            mf_acc.printMemoryFootprint();
            wip_accum_mf.addMF(mf_acc);
            mf_acc.reset();
        }

        worker_ul_map.clear();
        worker_ul_ordering.clear();
        worker_dl_map.clear();
        worker_dl_ordering.clear();
        worker_dl_validation_map.clear();
        worker_generic_map.clear();
        NVLOGD_FMT(TAG, "Workers destroyed");

        slot_map_ul_array[0]->mf.printMemoryFootprint();
        for(int i = 0; i < SLOT_MAP_NUM; i++)
        {
            mf_acc.addMF(slot_map_ul_array[i]->mf);
            delete slot_map_ul_array[i];
        }
        mf_acc.printMemoryFootprint();
        wip_accum_mf.addMF(mf_acc);
        mf_acc.reset();
        NVLOGD_FMT(TAG, "SlotMap UL destroyed");

        slot_map_dl_array[0]->mf.printMemoryFootprint();
        for(int i = 0; i < SLOT_MAP_NUM; i++)
        {
            mf_acc.addMF(slot_map_dl_array[i]->mf);
            delete slot_map_dl_array[i];
        }
        mf_acc.printMemoryFootprint();
        wip_accum_mf.addMF(mf_acc);
        mf_acc.reset();
        NVLOGD_FMT(TAG, "SlotMap DL destroyed");

        free(sc_aggr_array);

        task_item_array[0]->mf.printMemoryFootprint();
        for(int i = 0; i < TASK_ITEM_NUM; i++)
        {
            mf_acc.addMF(task_item_array[i]->mf);
            delete task_item_array[i];
        }
        mf_acc.printMemoryFootprint();
        wip_accum_mf.addMF(mf_acc);
        mf_acc.reset();
        NVLOGD_FMT(TAG, "Task item list destroyed");

        order_entity_list[0]->mf.printMemoryFootprint();
        for(int i = 0; i < ORDER_ENTITY_NUM; i++)
        {
            mf_acc.addMF(order_entity_list[i]->mf);
            delete order_entity_list[i];
        }
        mf_acc.printMemoryFootprint();
        wip_accum_mf.addMF(mf_acc);
        mf_acc.reset();
        // order_entity_list.erase(order_entity_list.begin(), order_entity_list.end());
        NVLOGD_FMT(TAG, "Order Kernels destroyed");

        task_list_ul->mf.printMemoryFootprint();
        wip_accum_mf.addMF(task_list_ul->mf);
        task_list_ul.reset();
        task_list_dl->mf.printMemoryFootprint();
        wip_accum_mf.addMF(task_list_dl->mf);
        task_list_dl.reset();
        task_list_dl_validation->mf.printMemoryFootprint();
        wip_accum_mf.addMF(task_list_dl_validation->mf);
        task_list_dl_validation.reset();

        task_list_debug->mf.printMemoryFootprint();
        wip_accum_mf.addMF(task_list_debug->mf);
        task_list_debug.reset();

        std::fill(h2d_copy_cuda_event_rec_done.begin(), h2d_copy_cuda_event_rec_done.end(), false);
        h2d_write_idx=0;
        h2d_read_idx=0;
        h2d_copy_thread_enable=0;

        if(this->mMIMO_enable)
        {
    #if 1
            mf_acc.addMF(cv_srs_chest_memory_bank->mf);
            //wip_accum_mf.addMf(cv_srs_chest_memory_bank->mf);
    #else
            mf_acc.addMF(cv_memory_bank->mf);
            //wip_accum_mf.addMf(cv_memory_bank->mf);
    #endif
            mf_acc.printMemoryFootprint();
            wip_accum_mf.addMF(mf_acc);
            mf_acc.reset();
        }

        if(active == true)
        {
            hq_manager->mf.printMemoryFootprint();
            wip_accum_mf.addMF(hq_manager->mf);
            hq_manager.reset();
        }

        // Destroy MPS ctx
        try {
            delete puschMpsCtx;
            delete pucchMpsCtx;
            delete prachMpsCtx;
            if(this->enable_srs)
            {
                delete srsMpsCtx;
            }
        } catch(const std::exception& e) {
            printf("EXCEPTION deleting UL MPS contexts: %s\n", e.what());
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "EXCEPTION deleting UL MPS contexts: {}", e.what());
        }

        try {
            auto del_h_dl_buffers_addr = h_dl_buffers_addr.get_deleter();
            auto* released_h_dl_buffers_addr = h_dl_buffers_addr.release();
            del_h_dl_buffers_addr(released_h_dl_buffers_addr);

            auto del_d_dl_buffers_addr = d_dl_buffers_addr.get_deleter();
            auto* released_d_dl_buffers_addr = d_dl_buffers_addr.release();
            del_d_dl_buffers_addr(released_d_dl_buffers_addr);
        } catch(const std::exception& e) {
            printf("EXCEPTION in DL buffer cleanup: %s\n", e.what());
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "EXCEPTION in DL buffer cleanup: {}", e.what());
        }

        try {
            delete pdschMpsCtx;
            //delete pdcchMpsCtx;
            //delete pbchMpsCtx;
            delete ulMpsCtx;
            //delete dlMpsCtx;
            delete dlCtrlMpsCtx;
            if(gpuCommDlEnabled())
            {
                delete gpuCommsMpsCtx;
            }
        } catch(const std::exception& e) {
            printf("EXCEPTION deleting DL MPS contexts: %s\n", e.what());
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "EXCEPTION deleting DL MPS contexts: {}", e.what());
        }

        // Cleanup GPU resources
        if(gpu_map.size() > 0)
        {
            gpu_map.begin()->second->mf.printMemoryFootprint();
            for (auto it=gpu_map.begin(); it!=gpu_map.end(); ++it)
                mf_acc.addMF(it->second->mf);
            //mf_acc.printMemoryFootprint();
            wip_accum_mf.addMF(mf_acc);
            mf_acc.reset();
        }
        gpu_map.clear();
        NVLOGD_FMT(TAG, "GPU destroyed");

        wip_accum_mf.addMF(mf);
        mf.printMemoryFootprint();

#if 0
        NVLOGI_FMT(TAG, "Total CPU regular memory {} B / {} KB / {} MB", ctx_tot_cpu_regular_memory, (ctx_tot_cpu_regular_memory/1024), ((ctx_tot_cpu_regular_memory/1024)/1024));
        NVLOGI_FMT(TAG, "Total CPU pinned memory {} B / {} KB / {} MB", ctx_tot_cpu_pinned_memory , (ctx_tot_cpu_pinned_memory/1024) , ((ctx_tot_cpu_pinned_memory/1024)/1024));
        NVLOGI_FMT(TAG, "Total GPU regular memory {} B / {} KB / {} MB", ctx_tot_gpu_regular_memory, (ctx_tot_gpu_regular_memory/1024),  ((ctx_tot_gpu_regular_memory/1024)/1024));
        NVLOGI_FMT(TAG, "Total GPU pinned memory {} B / {} KB / {} MB", ctx_tot_gpu_pinned_memory , (ctx_tot_gpu_pinned_memory/1024) , ((ctx_tot_gpu_pinned_memory/1024)/1024));
#else
        // Currently the ctx_to_cpu_regular_memory can differ from the wip_accum_mf regular cpu memory field (be greater than it), because the wip_accum_mf does not
        // include the memory allocation of all the MemFoot objects.
        NVLOGC_FMT(TAG, "Total CPU regular memory {} B / {} KiB / {} MiB", ctx_tot_cpu_regular_memory, (ctx_tot_cpu_regular_memory/1024), ((ctx_tot_cpu_regular_memory/1024)/1024));
        NVLOGC_FMT(TAG, "Total CPU pinned memory {} B / {} KiB / {} MiB", ctx_tot_cpu_pinned_memory , (ctx_tot_cpu_pinned_memory/1024) , ((ctx_tot_cpu_pinned_memory/1024)/1024));
        NVLOGC_FMT(TAG, "Total GPU regular memory {} B / {} KiB / {} MiB", ctx_tot_gpu_regular_memory, (ctx_tot_gpu_regular_memory/1024),  ((ctx_tot_gpu_regular_memory/1024)/1024));
        NVLOGC_FMT(TAG, "Total GPU pinned memory {} B / {} KiB / {} MiB", ctx_tot_gpu_pinned_memory , (ctx_tot_gpu_pinned_memory/1024) , ((ctx_tot_gpu_pinned_memory/1024)/1024));
#endif

        // FIXME: cudaStreamDestroy()
        wip_accum_mf.printMemoryFootprint();

        printf("PhyDriverCtx destructor completed successfully\n");
        NVLOGC_FMT(TAG, "PhyDriverCtx destructor completed successfully");
    } catch(const std::exception& e) {
        printf("EXCEPTION in caught in PhyDriverCtx destructor: %s\n", e.what());
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "EXCEPTION in caught in PhyDriverCtx destructor: {}", e.what());
    } catch(...) {
        printf("UNKNOWN EXCEPTION caught in PhyDriverCtx destructor\n");
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "UNKNOWN EXCEPTION caught in PhyDriverCtx destructor");
    }
}

bool PhyDriverCtx::isValidation()
{
    return validation;
}

/////////////////////////////////////////////////////////////////////
//// L2 callbacks
/////////////////////////////////////////////////////////////////////

int PhyDriverCtx::setUlCb(slot_command_api::ul_slot_callbacks& _ul_cb)
{
    ul_cb = _ul_cb;

    return 0;
}

int PhyDriverCtx::setDlCb(slot_command_api::dl_slot_callbacks& _dl_cb)
{
    dl_cb = _dl_cb;

    return 0;
}

bool PhyDriverCtx::getUlCb(slot_command_api::ul_slot_callbacks& cb)
{
    if(ul_cb.alloc_fn && ul_cb.callback_fn)
    {
        cb = ul_cb;
        return true;
    }

    return false;
}

bool PhyDriverCtx::getDlCb(slot_command_api::dl_slot_callbacks& cb)
{
    if(dl_cb.callback_fn)
    {
        cb = dl_cb;
        return true;
    }

    return false;
}
int PhyDriverCtx::setCellUpdateCb(::CellUpdateCallBackFn& callback) {
    if (callback) {
        cell_update_cb = callback;
    }
    return 0;
}



bool PhyDriverCtx::cellUpdateCbExists()
{
  if (cell_update_cb) {
    return true;
  }
   return false;
}
/////////////////////////////////////////////////////////////////////
//// SlotMap management
/////////////////////////////////////////////////////////////////////

SlotMapDl* PhyDriverCtx::getNextSlotMapDl()
{
    SlotMapDl* sm  = nullptr;
    int      cnt = 0;

    mlock_slot_map_dl.lock();

    while(slot_map_dl_array[slot_map_dl_index]->reserve() != 0 && cnt < SLOT_MAP_NUM)
    {
        slot_map_dl_index   = (slot_map_dl_index + 1) % SLOT_MAP_NUM;
        cnt++;
    }

    if(cnt < SLOT_MAP_NUM)
    {
        sm                  = slot_map_dl_array[slot_map_dl_index];      //.get();
        slot_map_dl_index   = (slot_map_dl_index + 1) % SLOT_MAP_NUM; //Alreday set to the next one
    }

    mlock_slot_map_dl.unlock();

    return sm;
}

SlotMapUl* PhyDriverCtx::getNextSlotMapUl()
{
    SlotMapUl* sm  = nullptr;
    int      cnt = 0;

    mlock_slot_map_ul.lock();

    while(slot_map_ul_array[slot_map_ul_index]->reserve() != 0 && cnt < SLOT_MAP_NUM)
    {
        slot_map_ul_index   = (slot_map_ul_index + 1) % SLOT_MAP_NUM;
        cnt++;
    }

    if(cnt < SLOT_MAP_NUM)
    {
        sm                  = slot_map_ul_array[slot_map_ul_index];      //.get();
        slot_map_ul_index   = (slot_map_ul_index + 1) % SLOT_MAP_NUM; //Alreday set to the next one
    }

    mlock_slot_map_ul.unlock();

    return sm;
}

struct slot_params_aggr* PhyDriverCtx::getNextSlotCmd()
{
    struct slot_params_aggr* sc  = nullptr;
    // int      cnt = 0;

    sc_aggr_lock.lock();

    //ToDo: Reserve sc structure
    // while(sc_aggr_array[sc_aggr_index].reserve() != 0 && cnt < SLOT_CMD_NUM)
    // {
    //     sc_aggr_index = (sc_aggr_index + 1) % SLOT_CMD_NUM;
    //     cnt++;
    // }

    sc                  = &sc_aggr_array[sc_aggr_index];
    sc_aggr_index = (sc_aggr_index + 1) % SLOT_CMD_NUM;

    sc_aggr_lock.unlock();

    return sc;
}

/////////////////////////////////////////////////////////////////////
//// TaskList management
/////////////////////////////////////////////////////////////////////

TaskList* PhyDriverCtx::getTaskListUl()
{
    return task_list_ul.get();
}

TaskList* PhyDriverCtx::getTaskListDl()
{
    return task_list_dl.get();
}

TaskList* PhyDriverCtx::getTaskListDebug()
{
    return task_list_debug.get();
}

TaskList* PhyDriverCtx::getTaskListDlVal()
{
    return task_list_dl_validation.get();
}


Task* PhyDriverCtx::getNextTask()
{
    Task* ts = nullptr;

    ts              = task_item_array[task_item_index]; //.get();
    task_item_index = (task_item_index + 1) % TASK_ITEM_NUM;

    return ts;
}

/////////////////////////////////////////////////////////////////////
//// Cell management
/////////////////////////////////////////////////////////////////////
int PhyDriverCtx::addNewCell(const cell_mplane_info& m_plane_info,uint32_t idx)
{
    TI_GENERIC_INIT("PhyDriverCtx::addNewCell",8);

    TI_GENERIC_ADD("Start Task");

    if(getGpuNum() <= 0)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "ERROR: No GPU associated to this context. Can't create a cell");
        return -1;
    }

    //FIXME: generic class to generate IDs?
    TI_GENERIC_ADD("Cell Create");
    cell_id_t cell_id = Time::nowNs().count();
    std::unique_ptr<Cell> c = std::unique_ptr<Cell>(new Cell((phydriver_handle)this, cell_id, m_plane_info, getFhProxy(), getFirstGpu(),idx));

    TI_GENERIC_ADD("Insert Map");
    if(cell_map.insert(std::pair<cell_id_t, std::unique_ptr<Cell>>(cell_id, std::move(c))).second == false)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Cell {} insert error", cell_id);
        return -1;
    }

    TI_GENERIC_ADD("Cell Pairing");
    Cell* cell_ptr = NULL;
    for(auto it = cell_map.begin(); it != cell_map.end(); it++)
    {
        if(it->second->getId() == cell_id)
            cell_ptr = it->second.get();
    }


    /* CAUTION NOTE: setIOBuf allocates memory and take > 100ms for each cell. As such, for 4 cells it was taking more than 500 ms.
     * ISV's have complained that this is too long to happen during cell lifecycle management from L2 (cell create, cell start etc).
     * So this function was moved from l1_cell_create to l1_init so that the memory allocation can happen during L1 Init and before
     * L1 is ready to communicate with L2. THIS SHOULD NEVER BE MOVED FROM HERE */
    TI_GENERIC_ADD("setIOBuf");
    int ret = cell_ptr->setIOBuf();
    if(ret)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "setIOBuf error {}",  ret);
        return ret;
    }

    setBfwCoeffBuffer(idx, cell_ptr->getBfwCoeffBuffer());

    TI_GENERIC_ADD("End Task");
    TI_GENERIC_ALL_NVLOGI(TAG_STARTUP_TIMES);

    return 0;
}

void PhyDriverCtx::setBfwCoeffBuffer(uint8_t cell_idx, bfw_buffer_info* buffer_info) noexcept
{
    bfw_coeff_buffer[cell_idx] = buffer_info;
}

bfw_buffer_info* PhyDriverCtx::getBfwCoeffBuffer(uint8_t cell_idx) const
{
    return bfw_coeff_buffer[cell_idx];
}

int PhyDriverCtx::setCellPhyByMplane(struct cell_phy_info& cell_pinfo)
{
    Cell*    cell_ptr   = nullptr;
    int      ret        = 0;

    for(auto it = cell_map.begin(); it != cell_map.end(); it++)
    {
        if(it->second->getMplaneId() == cell_pinfo.mplane_id)
            cell_ptr = it->second.get();
    }

    if(cell_ptr == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Can't find cell with M-plane ID {}", cell_pinfo.mplane_id);
        return -1;
    }

    ret = cell_ptr->setPhyStatic(cell_pinfo);
    if(ret)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "setPhyStaticInfo error {}", ret);
        return ret;
    }

    if(cell_index_map.insert(std::pair<uint16_t, cell_id_t>(cell_ptr->getPhyId(), cell_ptr->getId())).second == false)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "phy_cell_id {} already exists in cell_index_map", cell_ptr->getPhyId());
        return -1;
    }

    ret = cell_ptr->setGpuItems();
    if(ret)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "setGpuItems error {}", ret);
        return ret;
    }
    // Cannot add in the loops below wip_accum_mf.addMF(aggr_<channel>_items[i]->mf) because that will add number of aggr objects * number of cells rather than only aggr object times

    for(int i = 0; i < getPuschAggrPerCtx(); i++)
    {
        aggr_pusch_items[i]->createPhyObj();
        aggr_pusch_items[i]->updateMemoryTracker(); // update total GPU
        aggr_pusch_items[i]->printGpuMemoryFootprint(); // When there are multiple cells, non zero information will be printed only when the last cell of the group is added
        cuphyChannelsAccumMf.addMF(aggr_pusch_items[i]->cuphyMf);//getGpuMemoryFootprint());

    }

    for(int i = 0; i < getPucchAggrPerCtx(); i++)
    {
        aggr_pucch_items[i]->createPhyObj();
        aggr_pucch_items[i]->updateMemoryTracker();
        aggr_pucch_items[i]->printGpuMemoryFootprint();
        cuphyChannelsAccumMf.addMF(aggr_pucch_items[i]->cuphyMf);
    }

    for(int i = 0; i < getPrachAggrPerCtx(); i++)
    {
        aggr_prach_items[i]->createPhyObj();
        aggr_prach_items[i]->updateMemoryTracker();
        aggr_prach_items[i]->printGpuMemoryFootprint();
        cuphyChannelsAccumMf.addMF(aggr_prach_items[i]->cuphyMf);
    }

    if(this->enable_srs) //Move SRS object creation under mMIMO enable flag for memory savings
    {
        for(int i = 0; i < getSrsAggrPerCtx(); i++)
        {
            aggr_srs_items[i]->createPhyObj();
            aggr_srs_items[i]->updateMemoryTracker();
            aggr_srs_items[i]->printGpuMemoryFootprint();
            cuphyChannelsAccumMf.addMF(aggr_srs_items[i]->cuphyMf);
        }
    }
    if(this->mMIMO_enable)
    {
        for(int i = 0; i < getUlbfwAggrPerCtx(); i++)
        {
            aggr_ulbfw_items[i]->createPhyObj();
            aggr_ulbfw_items[i]->updateMemoryTracker();
            aggr_ulbfw_items[i]->printGpuMemoryFootprint();
            cuphyChannelsAccumMf.addMF(aggr_ulbfw_items[i]->cuphyMf);
        }

        for(int i = 0; i < PHY_DLBFW_AGGR_X_CTX; i++)
        {
            aggr_dlbfw_items[i]->createPhyObj();
            aggr_dlbfw_items[i]->updateMemoryTracker();
            aggr_dlbfw_items[i]->printGpuMemoryFootprint();
            cuphyChannelsAccumMf.addMF(aggr_dlbfw_items[i]->cuphyMf);
        }
    }

    for(int i = 0; i < PHY_PDSCH_AGGR_X_CTX; i++)
    {
        aggr_pdsch_items[i]->createPhyObj();
        aggr_pdsch_items[i]->updateMemoryTracker();
        aggr_pdsch_items[i]->printGpuMemoryFootprint();
        cuphyChannelsAccumMf.addMF(aggr_pdsch_items[i]->cuphyMf);
    }

    for(int i = 0; i < PHY_PBCH_AGGR_X_CTX; i++)
    {
        aggr_pbch_items[i]->createPhyObj();
        aggr_pbch_items[i]->updateMemoryTracker();
        aggr_pbch_items[i]->printGpuMemoryFootprint();
        cuphyChannelsAccumMf.addMF(aggr_pbch_items[i]->cuphyMf);
    }

    for(int i = 0; i < PHY_PDCCH_DL_AGGR_X_CTX; i++)
    {
        aggr_pdcch_dl_items[i]->createPhyObj();
        aggr_pdcch_dl_items[i]->updateMemoryTracker();
        aggr_pdcch_dl_items[i]->printGpuMemoryFootprint();
        cuphyChannelsAccumMf.addMF(aggr_pdcch_dl_items[i]->cuphyMf);
    }

    for(int i = 0; i < PHY_PDCCH_UL_AGGR_X_CTX; i++)
    {
        aggr_pdcch_ul_items[i]->createPhyObj();
        aggr_pdcch_ul_items[i]->updateMemoryTracker();
        aggr_pdcch_ul_items[i]->printGpuMemoryFootprint();
        cuphyChannelsAccumMf.addMF(aggr_pdcch_ul_items[i]->cuphyMf);
    }

    for(int i = 0; i < PHY_CSIRS_AGGR_X_CTX; i++)
    {
        aggr_csirs_items[i]->createPhyObj();
        aggr_csirs_items[i]->updateMemoryTracker();
        aggr_csirs_items[i]->printGpuMemoryFootprint();
        cuphyChannelsAccumMf.addMF(aggr_csirs_items[i]->cuphyMf);
    }
    //Diplay accumulated GPU memory footprint for allocation within cuPHY channel objects
    cuphyChannelsAccumMf.printGpuMemoryFootprint();
    cuphyChannelsAccumMf.printMemoryFootprint();
    wip_accum_mf.addMF(cuphyChannelsAccumMf);
    wip_accum_mf.printGpuMemoryFootprint();
    wip_accum_mf.printMemoryFootprint();

    return 0;
}

Cell* PhyDriverCtx::getCellById(cell_id_t c_id)
{
    auto it = cell_map.find(c_id);
    if(it == cell_map.end())
        return nullptr;
    return it->second.get();
}

Cell* PhyDriverCtx::getCellByPhyId(uint16_t c_phy_id)
{
    auto it = cell_index_map.find(c_phy_id);
    if(it == cell_index_map.end())
        return nullptr;

    return getCellById(it->second);
}

//Change the phyCellId of an existing entry in cell_index_map
int PhyDriverCtx::setCellPhyId(uint16_t c_phy_id_old, uint16_t c_phy_id_new, cell_id_t cell_id)
{
    auto it = cell_index_map.find(c_phy_id_old);
    if(it == cell_index_map.end())
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Cell {} not found in cell_index_map", c_phy_id_old);
        return -1;
    }

    auto cell_id_old = it->second;
    auto ret = cell_index_map.insert(std::pair<uint16_t, cell_id_t>(c_phy_id_new, cell_id));
    if(ret.second == false)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Cell {} insert into cell_index_map failed", c_phy_id_new);
        return -1;
    }

    it = cell_index_map.erase(it);
    // it = cell_index_map.find(cell_id_old);
    if(it == cell_index_map.end())
    {
        NVLOGI_FMT(TAG, "Cell {} erased from cell_index_map",c_phy_id_old);
    }

    /*it = cell_index_map.find(c_phy_id_new);
    if(it == cell_index_map.end())
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Cell {} not found in cell_index_map", c_phy_id_new);
        return -1;
    }*/
    else
        NVLOGI_FMT(TAG, "Cell {} added into cell_index_map",c_phy_id_new);

    //PDSCH, PUSCH and PUCCH aggregated object use phyCellId in static params. Update them
    for(uint32_t i=0; i < aggr_pusch_items.size(); i++)
        aggr_pusch_items[i]->updatePhyCellId(c_phy_id_old,c_phy_id_new);
    for(uint32_t i=0; i < aggr_pucch_items.size(); i++)
        aggr_pucch_items[i]->updatePhyCellId(c_phy_id_old,c_phy_id_new);
    /*Need to check if this is needed for SRS*/
    for(uint32_t i=0; i < aggr_pdsch_items.size(); i++)
        aggr_pdsch_items[i]->updatePhyCellId(c_phy_id_old,c_phy_id_new);
    for(uint32_t i=0; i < aggr_srs_items.size(); i++)
        aggr_srs_items[i]->updatePhyCellId(c_phy_id_old,c_phy_id_new);

    return 0;
}

Cell* PhyDriverCtx::getCellByMplaneId(uint16_t mplane_id)
{
    for (auto it = cell_map.begin(); it != cell_map.end(); it++)
    {
        Cell* c = it->second.get();
        if (c->getMplaneId() == mplane_id)
        {
            return c;
        }
    }
    return nullptr;
}

int PhyDriverCtx::removeCell(uint16_t cid)
{
    auto it_cell = cell_map.find(cid);
    if(it_cell == cell_map.end())
        return -1;

    NVLOGD_FMT(TAG, "Stopping cell",  it_cell->first);
    it_cell->second->stop();

#if 0
    for(int i = 0; i < phy_cell_list.size(); i++)
    {
        if(phy_cell_list[i].first == cid)
        {
            /*
             * TBD set a max tentative number
             */
            while(phy_cell_list[i].second->isActive() == true)
            {
                NVLOGD_FMT(TAG, "Cell {} can't be set to unavailable. Retry in {} us", cid, Time::NsToUs(waitns).count());
                Time::waitDurationNs(waitns);

                /*
                 * Avoid getting stuck on errors:
                 * If there are no workers in the system (simulation, benchmarks, errors)
                 * the cell and the PUSCH obj will never be freed
                 */
                if(worker_ul_map.size() == 0)
                {
                    NVLOGD_FMT(TAG, "No active workers, cleaning up");
                    phy_cell_list[i].second->release();
                    break;
                }
            }
            phy_cell_list.erase(phy_cell_list.begin() + i);
            break;
        }
    }
#endif

    // cell_map.erase(it_cell); //cell_map.find(cid) );

    return 0;
}

int PhyDriverCtx::getCellNum()
{
    return static_cast<int>(cell_map.size());
}

int PhyDriverCtx::getCellList(Cell **clist, uint32_t *pcellCount)
{
    for(auto &c : cell_map)
    {
        clist[*pcellCount] = c.second.get();
        (*pcellCount)++;
    }
    return 0;
}

int PhyDriverCtx::getCellIdxList(std::array<uint32_t,MAX_CELLS_PER_SLOT>& cell_idx_list)
{
    int cellCount{};
    Cell* cPtr{};
    for(auto &c : cell_map)
    {
        cPtr = c.second.get();
        cell_idx_list[cellCount] = cPtr->getIdx();
        ++cellCount;
    }
    return cellCount;
}

/////////////////////////////////////////////////////////////////////
//// Worker management
/////////////////////////////////////////////////////////////////////
int PhyDriverCtx::addGenericWorker(std::unique_ptr<Worker> w)
{
    enum worker_default_type type = w->getType();
    uint64_t                 wid  = w->getId();

    auto ret = worker_generic_map.insert(std::pair<worker_id, std::unique_ptr<Worker>>(wid, std::move(w)));
    if(ret.second == false)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Worker {} already exists", wid );
        return -1;
    }

    return 0;
}

Worker* PhyDriverCtx::getWorkerById(worker_id id)
{
    auto it_ul = worker_ul_map.find(id);
    if(it_ul != worker_ul_map.end())
        return it_ul->second.get();

    auto it_dl = worker_dl_map.find(id);
    if(it_dl != worker_dl_map.end())
        return it_dl->second.get();

    auto it_g = worker_generic_map.find(id);
    if(it_g != worker_generic_map.end())
        return it_g->second.get();

    return nullptr;
}

worker_id PhyDriverCtx::getULWorkerID(int worker_index) {
    if(worker_index < 0 || worker_index >= worker_ul_ordering.size()) {
        NVLOGW_FMT(TAG ,"getULWorkerID requested for an index that is out of bounds: {} {}",worker_index,worker_ul_ordering.size());
        return (worker_id) 0;
    } else {
        return worker_ul_ordering[worker_index];
    }
}

worker_id PhyDriverCtx::getDLWorkerID(int worker_index) {
    if(worker_index < 0 || worker_index >= worker_dl_ordering.size()) {
        NVLOGW_FMT(TAG ,"getDLWorkerID requested for an index that is out of bounds: {} {}",worker_index,worker_dl_ordering.size());
        return (worker_id) 0;
    } else {
        return worker_dl_ordering[worker_index];
    }
}



int PhyDriverCtx::getNumULWorkers() {
    return worker_ul_map.size();
}

int PhyDriverCtx::getNumDLWorkers() {
    return worker_dl_map.size();
}

int PhyDriverCtx::removeWorker(worker_id id)
{
    auto it_ul = worker_ul_map.find(id);
    if(it_ul != worker_ul_map.end())
    {
        worker_ul_map.erase(it_ul);

        //Also remove this from ordering
        for(int ii=0; ii<worker_ul_ordering.size(); ii++) {
            worker_id wid = worker_ul_ordering[ii];
            if(wid == id) {
                worker_ul_ordering.erase(worker_ul_ordering.begin()+ii);
                break;
            }
        }

        return 0;
    }

    auto it_dl = worker_dl_map.find(id);
    if(it_dl != worker_dl_map.end())
    {
        worker_dl_map.erase(it_dl);

        //Also remove this from ordering
        for(int ii=0; ii<worker_dl_ordering.size(); ii++) {
            worker_id wid = worker_dl_ordering[ii];
            if(wid == id) {
                worker_ul_ordering.erase(worker_dl_ordering.begin()+ii);
                break;
            }
        }

        return 0;
    }

    auto it_dl_v = worker_dl_validation_map.find(id);
    if(it_dl_v != worker_dl_validation_map.end())
    {
        worker_dl_validation_map.erase(it_dl_v);
        return 0;
    }

    auto it_g = worker_generic_map.find(id);
    if(it_g != worker_generic_map.end())
    {
        worker_generic_map.erase(it_g);
        return 0;
    }

    NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Worker {} doesn't exist", id );
    return -1;
}

/////////////////////////////////////////////////////////////////////
//// GPU management
/////////////////////////////////////////////////////////////////////
GpuDevice* PhyDriverCtx::getGpuById(int id)
{
    auto it = gpu_map.find(id);
    if(it == gpu_map.end())
        return nullptr;
    return it->second.get();
}

GpuDevice* PhyDriverCtx::getFirstGpu()
{
    auto tmp = gpu_map.begin();
    return tmp->second.get();
}

int PhyDriverCtx::getGpuNum()
{
    return static_cast<int>(gpu_map.size());
}

cudaStream_t PhyDriverCtx::getUlOrderStreamPd()
{
    return stream_order_pd;
}

cudaStream_t PhyDriverCtx::getUlOrderStreamSrsPd()
{
    return stream_order_srs_pd;
}

cudaStream_t* PhyDriverCtx::getUlOrderStreamsPusch()
{
    return aggr_stream_pusch;
}

cudaStream_t* PhyDriverCtx::getUlOrderStreamsPucch()
{
    return aggr_stream_pucch;
}

cudaStream_t* PhyDriverCtx::getUlOrderStreamsPrach()
{
    return aggr_stream_prach;
}

void PhyDriverCtx::warmupStream(cudaStream_t stream)
{
    GpuDevice* gpu_device = getFirstGpu();
    launch_kernel_warmup(stream);
    launch_kernel_order(stream, 1, 0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, 0, 0, 0, 0, nullptr, 0, 0, 0, 0, 0);
    launch_kernel_order(stream, 1, 0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, 0, 0, 0, 0, nullptr, 0, 0, 0, 0, 0);
    gpu_device->synchronizeStream(stream);
    return;
}

/////////////////////////////////////////////////////////////////////
//// PhyChannel
/////////////////////////////////////////////////////////////////////
OrderEntity* PhyDriverCtx::getNextOrderEntity(int32_t* cell_idx_list,uint8_t cell_idx_list_size ,OrderEntity* oe,bool new_order_entity)
{
    OrderEntity* oentity = nullptr;
    int             cnt   = 0;

    mlock_oentity.lock();
    if(new_order_entity)
    {
        while(order_entity_list[oentity_index]->reserve(cell_idx_list,cell_idx_list_size,new_order_entity) != 0 && cnt < ORDER_ENTITY_NUM)
        {
            oentity_index = (oentity_index + 1) % ORDER_ENTITY_NUM;
            cnt++;
        }

        if(cnt < ORDER_ENTITY_NUM)
        {
            oentity = order_entity_list[oentity_index]; //.get();
            //Alreday set to the next one
            oentity_index = (oentity_index + 1) % ORDER_ENTITY_NUM;
        }
    }
    else
    {
        if(oe==nullptr)
            NVLOGF_FMT(TAG,AERIAL_INTERNAL_EVENT, "Order entity nullptr error!");
        oe->reserve(cell_idx_list,cell_idx_list_size,new_order_entity);
        oentity=oe;
    }
    mlock_oentity.unlock();

    return oentity;
}

ok_tb_config_info_t* PhyDriverCtx::getOkTbConfig(uint8_t cell_idx) const
{
    return config_ok_tb[cell_idx];
}

uint32_t PhyDriverCtx::getConfigOkTbNumSlots() const
{
    return config_ok_tb_num_slots;
}

void PhyDriverCtx::setConfigOkTbNumSlots(uint32_t num_slots)
{
    config_ok_tb_num_slots = num_slots;
}

ok_tb_config_srs_info_t* PhyDriverCtx::getOkTbConfigSrs(uint8_t cell_idx) const
{
    return config_ok_tb_srs[cell_idx];
}

uint32_t PhyDriverCtx::getConfigOkTbSrsNumSlots() const
{
    return config_ok_tb_srs_num_slots;
}

void PhyDriverCtx::setConfigOkTbSrsNumSlots(uint32_t num_slots)
{
    config_ok_tb_srs_num_slots = num_slots;
}

uint8_t* PhyDriverCtx::getFhBufOkTbSrs(uint8_t cell_idx) const
{
    return fh_buf_ok_tb_srs[cell_idx];
}

uint32_t PhyDriverCtx::getConfigOkTbMaxPacketSize() const
{
    return config_ok_tb_max_packet_size;
}

void PhyDriverCtx::setConfigOkTbMaxPacketSize(uint32_t max_packet_size)
{
    // Round up to the nearest 16-byte multiple
    config_ok_tb_max_packet_size = ((max_packet_size + 15) / 16) * 16;
}

PhyUlBfwAggr* PhyDriverCtx::getNextUlBfwAggr(slot_params_aggr* aggr_slot_params)
{
    PhyUlBfwAggr* tmp = nullptr;
    int       cnt = 0;

    if(aggr_ulbfw_items.size() < getUlbfwAggrPerCtx())
        return nullptr;

    aggr_lock_cell_phy_ulbfw.lock();

    while(
        aggr_ulbfw_items[aggr_last_ulbfw]->reserveCellGroup() != 0 &&
        cnt < getUlbfwAggrPerCtx())
    {
        aggr_last_ulbfw = (aggr_last_ulbfw + 1) % getUlbfwAggrPerCtx();
        cnt++;
    }

    if(cnt < getUlbfwAggrPerCtx())
    {
        tmp = aggr_ulbfw_items[aggr_last_ulbfw].get();
        //Alreday set to the next one
        aggr_last_ulbfw = (aggr_last_ulbfw + 1) % getUlbfwAggrPerCtx();
    }

    aggr_lock_cell_phy_ulbfw.unlock();

    if(tmp)
    {
        tmp->setDynAggrParams(aggr_slot_params);
        if(tmp->getDynParams() == nullptr)
            tmp->release();
    }

    return tmp;
}


PhyPuschAggr* PhyDriverCtx::getNextPuschAggr(slot_params_aggr* aggr_slot_params)
{
    PhyPuschAggr* tmp = nullptr;
    int       cnt = 0;

    uint8_t pusch_aggr_x_ctx = getPuschAggrPerCtx();

    if(aggr_pusch_items.size() < pusch_aggr_x_ctx)
        return nullptr;

    aggr_lock_cell_phy_pusch.lock();

    while(
        aggr_pusch_items[aggr_last_pusch]->reserveCellGroup() != 0 &&
        cnt < pusch_aggr_x_ctx)
    {
        aggr_last_pusch = (aggr_last_pusch + 1) % pusch_aggr_x_ctx;
        cnt++;
    }
    if(cnt < pusch_aggr_x_ctx)
    {
        tmp = aggr_pusch_items[aggr_last_pusch].get();
        //Alreday set to the next one
        aggr_last_pusch = (aggr_last_pusch + 1) % pusch_aggr_x_ctx;
    }

    aggr_lock_cell_phy_pusch.unlock();

    if(tmp)
    {
        tmp->setDynAggrParams(aggr_slot_params);
        if(tmp->getDynParams() == nullptr)
            tmp->release();
    }

    return tmp;
}

PhyPucchAggr* PhyDriverCtx::getNextPucchAggr(slot_params_aggr* aggr_slot_params)
{
    PhyPucchAggr* tmp = nullptr;
    int       cnt = 0;

    uint8_t pucch_aggr_x_ctx = getPucchAggrPerCtx();

    if(aggr_pucch_items.size() < pucch_aggr_x_ctx)
        return nullptr;

    aggr_lock_cell_phy_pucch.lock();

    while(
        aggr_pucch_items[aggr_last_pucch]->reserveCellGroup() != 0 &&
        cnt < pucch_aggr_x_ctx)
    {
        aggr_last_pucch = (aggr_last_pucch + 1) % pucch_aggr_x_ctx;
        cnt++;
    }

    if(cnt < pucch_aggr_x_ctx)
    {
        tmp = aggr_pucch_items[aggr_last_pucch].get();
        //Alreday set to the next one
        aggr_last_pucch = (aggr_last_pucch + 1) % pucch_aggr_x_ctx;
    }

    aggr_lock_cell_phy_pucch.unlock();

    if(tmp)
    {
        tmp->setDynAggrParams(aggr_slot_params);
        if(tmp->getDynParams() == nullptr)
            tmp->release();
    }

    return tmp;
}

void* delete_prach_obj_func(void *arg)
{
    NVLOGI_FMT(TAG, "delete_prach_obj_func");
    if(pthread_setname_np(pthread_self(), "deletePrachObj") != 0)
    {
        NVLOGW_FMT(TAG, "{}: set thread name failed", __func__);
    }
    PhyDriverCtx* pdctx =  reinterpret_cast<PhyDriverCtx*>(arg);
    pdctx->deletePrachObjects();
    if(pdctx->cellUpdateCbExists())
    {
        NVLOGC_FMT(TAG, "{}: calling cell_update_cb for cell_id={} with error_code=0", __func__, pdctx->updateCellConfigCellId);
        pdctx->cell_update_cb(pdctx->updateCellConfigCellId,0);
    }

    pdctx->updateCellConfigMutex.unlock();

    NVLOGI_FMT(TAG, "Delete prach objects thread exit");
    return nullptr;
}

PhyPrachAggr* PhyDriverCtx::getNextPrachAggr(slot_params_aggr* aggr_slot_params)
{
    PhyPrachAggr* tmp = nullptr;
    int       cnt = 0;
    bool delete_objs = false;

    uint8_t prach_aggr_x_ctx = getPrachAggrPerCtx();

    if(aggr_prach_items.size() < prach_aggr_x_ctx)
        return nullptr;

    aggr_lock_cell_phy_prach.lock();

    while(
        aggr_prach_items[aggr_last_prach]->reserveCellGroup() != 0 &&
        cnt < prach_aggr_x_ctx)
    {
        aggr_last_prach = (aggr_last_prach + 1) % prach_aggr_x_ctx;
        cnt++;
    }

    if(cnt < prach_aggr_x_ctx)
    {
        tmp = aggr_prach_items[aggr_last_prach].get();
        //Alreday set to the next one
        aggr_last_prach = (aggr_last_prach + 1) % prach_aggr_x_ctx;
    }

    if(tmp && num_new_prach_handles)
    {
        NVLOGI_FMT(TAG, "{} new prach objects available. Use new object",num_new_prach_handles);
        tmp->changePhyObj();
        --num_new_prach_handles;
        if(num_new_prach_handles == 0)
            delete_objs = true;
    }

    aggr_lock_cell_phy_prach.unlock();

    if(tmp)
    {
        tmp->setDynAggrParams(aggr_slot_params);
        if(tmp->getDynParams() == nullptr)
            tmp->release();
    }

    if(delete_objs)
    {
        NVLOGI_FMT(TAG, "launch a new thread delete_prach_obj");
        pthread_t thread_id;
        pthread_create(&thread_id, NULL, delete_prach_obj_func, this);
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(0, &cpuset);
        pthread_setaffinity_np(thread_id, sizeof(cpu_set_t), &cpuset);
    }

    return tmp;
}

PhySrsAggr* PhyDriverCtx::getNextSrsAggr(slot_params_aggr* aggr_slot_params)
{
    PhySrsAggr* tmp = nullptr;
    int       cnt = 0;

    uint8_t srs_aggr_x_ctx = getSrsAggrPerCtx();

    if(aggr_srs_items.size() < srs_aggr_x_ctx)
        return nullptr;

    aggr_lock_cell_phy_srs.lock();

    while(
        aggr_srs_items[aggr_last_srs]->reserveCellGroup() != 0 &&
        cnt < srs_aggr_x_ctx)
    {
        aggr_last_srs = (aggr_last_srs + 1) % srs_aggr_x_ctx;
        cnt++;
    }

    if(cnt < srs_aggr_x_ctx)
    {
        tmp = aggr_srs_items[aggr_last_srs].get();
        //Alreday set to the next one
        aggr_last_srs = (aggr_last_srs + 1) % srs_aggr_x_ctx;
    }

    aggr_lock_cell_phy_srs.unlock();

    if(tmp)
    {
        tmp->setDynAggrParams(aggr_slot_params);
        if(tmp->getDynParams() == nullptr)
            tmp->release();
    }

    return tmp;
}


PhyDlBfwAggr* PhyDriverCtx::getNextDlBfwAggr(slot_params_aggr* aggr_slot_params)
{
    PhyDlBfwAggr* tmp = nullptr;
    int       cnt = 0;

    if(aggr_dlbfw_items.size() < PHY_DLBFW_AGGR_X_CTX)
        return nullptr;

    aggr_lock_cell_phy_dlbfw.lock();

    while(
        aggr_dlbfw_items[aggr_last_dlbfw]->reserveCellGroup() != 0 &&
        cnt < PHY_DLBFW_AGGR_X_CTX)
    {
        aggr_last_dlbfw = (aggr_last_dlbfw + 1) % PHY_DLBFW_AGGR_X_CTX;
        cnt++;
    }

    if(cnt < PHY_DLBFW_AGGR_X_CTX)
    {
        tmp = aggr_dlbfw_items[aggr_last_dlbfw].get();
        //Alreday set to the next one
        aggr_last_dlbfw = (aggr_last_dlbfw + 1) % PHY_DLBFW_AGGR_X_CTX;
    }

    aggr_lock_cell_phy_dlbfw.unlock();

    if(tmp)
    {
        tmp->setDynAggrParams(aggr_slot_params);
        if(tmp->getDynParams() == nullptr)
            tmp->release();
    }

    return tmp;
}

//DL BFW
void PhyDriverCtx::recordDlBFWCompletion(int slot)
{
    CUDA_CHECK_PHYDRIVER(cudaEventRecord(dlbfw_run_completion_event[slot], aggr_stream_dlbfw));
}

int PhyDriverCtx::queryDlBFWCompletion(int slot)
{
    cudaError_t cudaStatus = cudaEventQuery(dlbfw_run_completion_event[slot]);
    return (cudaStatus == cudaSuccess);
}

cudaError_t PhyDriverCtx::queryDlBFWCompletion_v2(int slot)
{
    return cudaEventQuery(dlbfw_run_completion_event[slot]);
}

//UL BFW
void PhyDriverCtx::recordUlBFWCompletion(int slot)
{
    CUDA_CHECK_PHYDRIVER(cudaEventRecord(ulbfw_run_completion_event[slot], aggr_stream_ulbfw));
}

int PhyDriverCtx::queryUlBFWCompletion(int slot)
{
    cudaError_t cudaStatus = cudaEventQuery(ulbfw_run_completion_event[slot]);
    return (cudaStatus == cudaSuccess);
}

PhyPdschAggr* PhyDriverCtx::getNextPdschAggr(slot_params_aggr* aggr_slot_params)
{
    PhyPdschAggr* tmp = nullptr;
    int       cnt = 0;

    if(aggr_pdsch_items.size() < PHY_PDSCH_AGGR_X_CTX)
        return nullptr;

    aggr_lock_cell_phy_pdsch.lock();

    while(
        aggr_pdsch_items[aggr_last_pdsch]->reserveCellGroup() != 0 &&
        cnt < PHY_PDSCH_AGGR_X_CTX)
    {
        aggr_last_pdsch = (aggr_last_pdsch + 1) % PHY_PDSCH_AGGR_X_CTX;
        cnt++;
    }

    if(cnt < PHY_PDSCH_AGGR_X_CTX)
    {
        tmp = aggr_pdsch_items[aggr_last_pdsch].get();
        //Alreday set to the next one
        aggr_last_pdsch = (aggr_last_pdsch + 1) % PHY_PDSCH_AGGR_X_CTX;
    }

    aggr_lock_cell_phy_pdsch.unlock();

    if(tmp)
    {
        tmp->setDynAggrParams(aggr_slot_params);
        if(tmp->getDynParams() == nullptr)
            tmp->release();
    }

    return tmp;
}

PhyPdcchAggr* PhyDriverCtx::getNextPdcchDlAggr(slot_params_aggr* aggr_slot_params)
{
    PhyPdcchAggr* tmp = nullptr;
    int       cnt = 0;

    if(aggr_pdcch_dl_items.size() < PHY_PDCCH_DL_AGGR_X_CTX)
        return nullptr;

    aggr_lock_cell_phy_pdcch_dl.lock();

    while(
        aggr_pdcch_dl_items[aggr_last_pdcch_dl]->reserveCellGroup() != 0 &&
        cnt < PHY_PDCCH_DL_AGGR_X_CTX)
    {
        aggr_last_pdcch_dl = (aggr_last_pdcch_dl + 1) % PHY_PDCCH_DL_AGGR_X_CTX;
        cnt++;
    }

    if(cnt < PHY_PDCCH_DL_AGGR_X_CTX)
    {
        tmp = aggr_pdcch_dl_items[aggr_last_pdcch_dl].get();
        //Alreday set to the next one
        aggr_last_pdcch_dl = (aggr_last_pdcch_dl + 1) % PHY_PDCCH_DL_AGGR_X_CTX;
    }

    aggr_lock_cell_phy_pdcch_dl.unlock();

    if(tmp)
    {
        tmp->setDynAggrParams(aggr_slot_params);
        if(tmp->getDynParams() == nullptr)
            tmp->release();
    }

    return tmp;
}

PhyPdcchAggr* PhyDriverCtx::getNextPdcchUlAggr(slot_params_aggr* aggr_slot_params)
{
    PhyPdcchAggr* tmp = nullptr;
    int       cnt = 0;

    if(aggr_pdcch_ul_items.size() < PHY_PDCCH_UL_AGGR_X_CTX)
        return nullptr;

    aggr_lock_cell_phy_pdcch_ul.lock();

    while(
        aggr_pdcch_ul_items[aggr_last_pdcch_ul]->reserveCellGroup() != 0 &&
        cnt < PHY_PDCCH_UL_AGGR_X_CTX)
    {
        if constexpr (PHY_PDCCH_UL_AGGR_X_CTX > 0)
            aggr_last_pdcch_ul = (aggr_last_pdcch_ul + 1) % PHY_PDCCH_UL_AGGR_X_CTX;
        cnt++;
    }

    if(cnt < PHY_PDCCH_UL_AGGR_X_CTX)
    {
        tmp = aggr_pdcch_ul_items[aggr_last_pdcch_ul].get();
        //Alreday set to the next one
        if constexpr (PHY_PDCCH_UL_AGGR_X_CTX > 0)
            aggr_last_pdcch_ul = (aggr_last_pdcch_ul + 1) % PHY_PDCCH_UL_AGGR_X_CTX;
    }

    aggr_lock_cell_phy_pdcch_ul.unlock();

    if(tmp)
    {
        tmp->setDynAggrParams(aggr_slot_params);
        if(tmp->getDynParams() == nullptr)
            tmp->release();
    }

    return tmp;
}

PhyPbchAggr* PhyDriverCtx::getNextPbchAggr(slot_params_aggr* aggr_slot_params)
{
    PhyPbchAggr* tmp = nullptr;
    int       cnt = 0;

    if(aggr_pbch_items.size() < PHY_PBCH_AGGR_X_CTX)
        return nullptr;

    aggr_lock_cell_phy_pbch.lock();

    while(
        aggr_pbch_items[aggr_last_pbch]->reserveCellGroup() != 0 &&
        cnt < PHY_PBCH_AGGR_X_CTX)
    {
        aggr_last_pbch = (aggr_last_pbch + 1) % PHY_PBCH_AGGR_X_CTX;
        cnt++;
    }

    if(cnt < PHY_PBCH_AGGR_X_CTX)
    {
        tmp = aggr_pbch_items[aggr_last_pbch].get();
        //Alreday set to the next one
        aggr_last_pbch = (aggr_last_pbch + 1) % PHY_PBCH_AGGR_X_CTX;
    }

    aggr_lock_cell_phy_pbch.unlock();

    if(tmp)
    {
        tmp->setDynAggrParams(aggr_slot_params);
        if(tmp->getDynParams() == nullptr)
            tmp->release();
    }

    return tmp;
}

PhyCsiRsAggr* PhyDriverCtx::getNextCsiRsAggr(slot_params_aggr* aggr_slot_params)
{
    PhyCsiRsAggr* tmp = nullptr;
    int       cnt = 0;

    if(aggr_csirs_items.size() < PHY_CSIRS_AGGR_X_CTX)
        return nullptr;

    aggr_lock_cell_phy_csirs.lock();

    while(
        aggr_csirs_items[aggr_last_csirs]->reserveCellGroup() != 0 &&
        cnt < PHY_CSIRS_AGGR_X_CTX)
    {
        aggr_last_csirs = (aggr_last_csirs + 1) % PHY_CSIRS_AGGR_X_CTX;
        cnt++;
    }

    if(cnt < PHY_CSIRS_AGGR_X_CTX)
    {
        tmp = aggr_csirs_items[aggr_last_csirs].get();
        //Alreday set to the next one
        aggr_last_csirs = (aggr_last_csirs + 1) % PHY_CSIRS_AGGR_X_CTX;
    }

    aggr_lock_cell_phy_csirs.unlock();

    if(tmp)
    {
        tmp->setDynAggrParams(aggr_slot_params);
        if(tmp->getDynParams() == nullptr)
            tmp->release();
    }

    return tmp;
}

int PhyDriverCtx::updateCellConfig(cell_id_t cell_id, cell_phy_info& cell_pinfo)
{
    for(int i = 0; i < getPrachAggrPerCtx(); i++)
    {
        PhyPrachAggr* ptr = aggr_prach_items[i].get();
        int ret = ptr->updateConfig(cell_id, cell_pinfo);
        if(ret == -1)
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Prach Aggr updateConfig failed");
            return ret;
        }
    }

    if(aggr_prach_items[0]->getCellStatVecSize() == getCellGroupNum())
    {
        //cuPhyObjects are already created.
        // 1 indicates operation needs to be completed in a separate thread context
        return 1;
    }

    //0 to indicate SUCCESS and operation completed
    return 0;
}

int PhyDriverCtx::createPrachObjects()
{
    int32_t val = 0;
    for(uint32_t i = 0; i < aggr_prach_items.size(); i++)
        val += aggr_prach_items[i]->createNewPhyObj();
    if(0 == val)
    {
        aggr_lock_cell_phy_prach.lock();
        num_new_prach_handles = aggr_prach_items.size();
        aggr_lock_cell_phy_prach.unlock();
        return 0;
    }
    else
    {
        deletePrachObjects();
        num_new_prach_handles = 0;
        return -1;
    }
}

void PhyDriverCtx::set_ru_type_for_srs_proc(ru_type ru_type)
{
    ru_type_for_srs_proc = ru_type;
}

ru_type PhyDriverCtx::get_ru_type_for_srs_proc() const
{
    return ru_type_for_srs_proc;
}

int PhyDriverCtx::deletePrachObjects()
{
    NVLOGI_FMT(TAG, "Delete prach objects start");
    for(uint32_t i = 0; i < aggr_prach_items.size(); i++)
        aggr_prach_items[i]->deleteTempPhyObj();
    NVLOGI_FMT(TAG, "Delete prach objects end");
    return 0;
}

int PhyDriverCtx::replacePrachObjects()
{
    for(uint32_t i = 0; i < aggr_prach_items.size(); i++)
        aggr_prach_items[i]->changePhyObj();
    num_new_prach_handles = 0;
    NVLOGI_FMT(TAG, "launch a new thread delete_prach_obj");
    pthread_t thread_id;
    pthread_create(&thread_id, NULL, delete_prach_obj_func, this);
    return 0;
}


/////////////////////////////////////////////////////////////////////
//// FhProxy
/////////////////////////////////////////////////////////////////////
FhProxy* PhyDriverCtx::getFhProxy()
{
    return fh_proxy.get();
}

/////////////////////////////////////////////////////////////////////
//// Logger
/////////////////////////////////////////////////////////////////////
log_handler_fn_t PhyDriverCtx::get_error_logger() const
{
    return log_err_fn_;
}
log_handler_fn_t PhyDriverCtx::get_info_logger() const
{
    return log_inf_fn_;
}
log_handler_fn_t PhyDriverCtx::get_debug_logger() const
{
    return log_dbg_fn_;
}

bool PhyDriverCtx::error_log_enabled() const
{
    return (log_lvl >= L1_LOG_LVL_ERROR) && (log_err_fn_ != nullptr);
}
bool PhyDriverCtx::info_log_enabled() const
{
    return (log_lvl >= L1_LOG_LVL_INFO) && (log_inf_fn_ != nullptr);
}
bool PhyDriverCtx::debug_log_enabled() const
{
    return (log_lvl >= L1_LOG_LVL_DBG) && (log_dbg_fn_ != nullptr);
}

void PhyDriverCtx::set_error_logger(log_handler_fn_t fn)
{
    log_err_fn_ = fn;
}
void PhyDriverCtx::set_info_logger(log_handler_fn_t fn)
{
    log_inf_fn_ = fn;
}
void PhyDriverCtx::set_debug_logger(log_handler_fn_t fn)
{
    log_dbg_fn_ = fn;
}

void PhyDriverCtx::set_level_logger(l1_log_level _log_lvl)
{
    log_lvl = _log_lvl;
}

int PhyDriverCtx::start()
{
    GpuDevice* gpu_device = getFirstGpu();
    if(gpu_device == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "No availble GPUs in this context");
        return -1;
    }

    // Create order kernel entity
    for(int i = 0; i < ORDER_ENTITY_NUM; i++)
        order_entity_list[i] = std::move(new OrderEntity(static_cast<phydriver_handle>(this), gpu_device));
        // order_entity_list.push_back(std::unique_ptr<OrderEntity>(new OrderEntity(static_cast<phydriver_handle>(this), gpu_device)));

    gpu_device->setDevice();

    //HARQ manager: need to create it here because eal is called within fh->start
    hq_manager = std::unique_ptr<HarqPoolManager>(new HarqPoolManager((phydriver_handle)this, gpu_device));

    //Weighted Average CFO Pool Manager: create if enabled
    if(getEnableWeightedAverageCfo())
    {
        FhProxy * fhproxy = getFhProxy();
        wavgcfo_manager = std::make_unique<WAvgCfoPoolManager>(static_cast<phydriver_handle>(this), gpu_device, fhproxy->getFhInstance());
    }

    if(this->enable_srs) //Moving the CV Memory bank allocation code under enable_srs flag for Memory savings
    {
        cv_srs_chest_memory_bank = std::make_unique<CvSrsChestMemoryBank>((phydriver_handle)this, gpu_device, total_num_srs_chest_buffers);
    }

    if(prometheus_cpu_core >= 0)
    {
        metrics = std::unique_ptr<Metrics>(new Metrics((phydriver_handle)(this), prometheus_cpu_core));
        metrics->start();
    }

    for(auto it = worker_ul_map.begin(); it != worker_ul_map.end(); it++)
    {
        if(it->second->run())
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Worker UL didn't start {}: {}", errno, std::strerror(errno));
            PHYDRIVER_THROW_EXCEPTIONS(errno, "Worker didn't start");
        }
    }

    for(auto it = worker_dl_map.begin(); it != worker_dl_map.end(); it++)
    {
        if(it->second->run())
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Worker DL didn't start {}: {}", errno , std::strerror(errno));
            PHYDRIVER_THROW_EXCEPTIONS(errno, "Worker didn't start");
        }
    }

    for(auto it = worker_dl_validation_map.begin(); it != worker_dl_validation_map.end(); it++)
    {
        if(it->second->run())
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Worker DL didn't start {}: {}", errno , std::strerror(errno));
            PHYDRIVER_THROW_EXCEPTIONS(errno, "Worker didn't start");
        }
    }

    for(auto it = worker_generic_map.begin(); it != worker_generic_map.end(); it++)
    {
        if(it->second->run())
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Generic worker didn't start {}: {}", errno , std::strerror(errno));
            PHYDRIVER_THROW_EXCEPTIONS(errno, "Worker didn't start");
        }
    }


    active = true;

    return 0;
}

int PhyDriverCtx::registerBufferToFh(void* buffer_ptr, size_t buffer_size)
{
    struct fh_memreg_buf mr;


    if(buffer_ptr == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "registerBufferToFh EINVAL");
        return EINVAL;
    }

    mr.memreg_info.len  = buffer_size; //4194304
    mr.memreg_info.addr = buffer_ptr;
    mr.memreg_info.page_sz = GPU_PAGE_SIZE;

    NVLOGI_FMT(TAG, "Calling register_mem on buffer {} size {}", buffer_ptr, buffer_size);

    if(fh_proxy->registerMem(&mr.memreg_info, &mr.memreg))
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "registerMem error");
        return -1;
    }

    mr.buffer_ptr = buffer_ptr;
    mr.buffer_size = buffer_size;

    return 0;
}

uint16_t PhyDriverCtx::getStartSectionIdSrs() const {
    return start_section_id_srs;
}

uint16_t PhyDriverCtx::getStartSectionIdPrach() const {
    return start_section_id_prach;
}

uint8_t PhyDriverCtx::getEnableUlCuphyGraphs() const {
    return enable_ul_cuphy_graphs;
}

uint8_t PhyDriverCtx::getEnableDlCuphyGraphs() const {
    return enable_dl_cuphy_graphs;
}

uint32_t PhyDriverCtx::getUlOrderTimeoutCPU() const {
    return ul_order_timeout_cpu_ns;
}

uint32_t PhyDriverCtx::getUlOrderTimeoutGPU() const {
    return ul_order_timeout_gpu_ns;
}

uint32_t PhyDriverCtx::getUlOrderTimeoutGPUSrs() const {
    return ul_order_timeout_gpu_srs_ns;
}

uint32_t PhyDriverCtx::getUlSrsAggr3TaskLaunchOffsetNs() const {
    return ul_srs_aggr3_task_launch_offset_ns;
}

uint32_t PhyDriverCtx::getUlOrderTimeoutLogInterval() const {
    return ul_order_timeout_log_interval_ns;
}

uint8_t PhyDriverCtx::getUlOrderKernelMode() const {
    return ul_order_kernel_mode;
}

uint8_t PhyDriverCtx::getUlOrderTimeoutGPULogEnable() const {
    return ul_order_timeout_gpu_log_enable;
}

uint32_t PhyDriverCtx::getUlOrderMaxRxPkts() const {
    return ul_order_max_rx_pkts;
}

uint32_t PhyDriverCtx::getUlOrderRxPktsTimeout() const {
    return ul_order_rx_pkts_timeout_ns;
}

bool PhyDriverCtx::isCPlaneDisabled() const {
    return cplane_disable;
}

uint32_t PhyDriverCtx::getUlOrderTimeoutFirstPktGPU() const {
    return ul_order_timeout_first_pkt_gpu_ns;
}

uint32_t PhyDriverCtx::getUlOrderTimeoutFirstPktGPUSrs() const {
    return ul_order_timeout_first_pkt_gpu_srs_ns;
}

HarqPoolManager * PhyDriverCtx::getHarPoolManager() const {
    if(active == true)
        return hq_manager.get();
    return nullptr;
}

WAvgCfoPoolManager * PhyDriverCtx::getWAvgCfoPoolManager() const {
    if(active == true)
        return wavgcfo_manager.get();
    return nullptr;
}

CvSrsChestMemoryBank* PhyDriverCtx::getCvSrsChestMemoryBank() const {
    return cv_srs_chest_memory_bank.get();
}

uint8_t PhyDriverCtx::getUseGreenContexts() const {
    return use_green_contexts;
}

uint8_t PhyDriverCtx::getUseGCWorkqueues() const {
    return use_gc_workqueues;
}

// Returns the requested configuration (yaml). However use of batched memcpy
// may not be possible if an insufficient CUDA_VERSION is used
uint8_t PhyDriverCtx::getUseBatchedMemcpy() const {
    return use_batched_memcpy;
}

int PhyDriverCtx::getMpsSmPusch() const {
    return mps_sm_pusch;
}

int PhyDriverCtx::getMpsSmPucch() const {
    return mps_sm_pucch;
}

int PhyDriverCtx::getMpsSmSrs() const {
    return mps_sm_srs;
}

int PhyDriverCtx::getMpsSmPrach() const {
    return mps_sm_prach;
}

int PhyDriverCtx::getMpsSmUlOrder() const {
    return mps_sm_ul_order;
}

int PhyDriverCtx::getMpsSmPdsch() const {
    return mps_sm_pdsch;
}

int PhyDriverCtx::getMpsSmPdcch() const {
    return mps_sm_pdcch;
}

int PhyDriverCtx::getMpsSmPbch() const {
    return mps_sm_pbch;
}

int PhyDriverCtx::getMpsSmCsiRs() const {
    return mps_sm_csirs;
}

int PhyDriverCtx::getMpsSmDlCtrl() const {
    return mps_sm_dl_ctrl;
}

int PhyDriverCtx::getMpsSmGpuComms() const {
    return mps_sm_gpu_comms;
}

uint8_t PhyDriverCtx::getPdschFallback() const {
    return pdsch_fallback;
}

MpsCtx* PhyDriverCtx::getUlCtx()
{
    return ulMpsCtx;
}

void PhyDriverCtx::setUlCtx()
{
    ulMpsCtx->setCtx();
}

MpsCtx* PhyDriverCtx::getDlCtx()
{
    return dlMpsCtx;
}

void PhyDriverCtx::setDlCtx()
{
    dlMpsCtx->setCtx();
}

MpsCtx* PhyDriverCtx::getGpuCommsCtx()
{
    return gpuCommsMpsCtx;
}

void PhyDriverCtx::setGpuCommsCtx()
{
    gpuCommsMpsCtx->setCtx();
}

uint8_t PhyDriverCtx::getCellGroupNum() const {
    return cell_group_num;
}

uint8_t PhyDriverCtx::getPuschWorkCancelMode(void) const {
    return pusch_workCancelMode;
}

uint8_t PhyDriverCtx::getPuschTdi(void) const {
    return enable_pusch_tdi;
}

uint8_t PhyDriverCtx::getPuschCfo(void) const {
    return enable_pusch_cfo;
}

uint8_t PhyDriverCtx::getPuschDftSOfdm(void) const {
    return enable_pusch_dftsofdm;
}

uint8_t PhyDriverCtx::getPuschTbSizeCheck(void) const {
    return enable_pusch_tbsizecheck;
}

uint8_t PhyDriverCtx::getPuschEarlyHarqEn(void) const {
    return pusch_earlyHarqEn;
}

uint8_t PhyDriverCtx::getPuschDeviceGraphLaunchEn(void) const {
    return pusch_deviceGraphLaunchEn;
}

void PhyDriverCtx::setPuschEarlyHarqEn(bool is_early_harq_detection_enabled)
{
    pusch_earlyHarqEn = (uint8_t)is_early_harq_detection_enabled;
}

uint16_t PhyDriverCtx::getPuschWaitTimeOutPreEarlyHarqUs(void) const {
    return pusch_waitTimeOutPreEarlyHarqUs;
}

uint16_t PhyDriverCtx::getPuschWaitTimeOutPostEarlyHarqUs(void) const {
    return pusch_waitTimeOutPostEarlyHarqUs;
}

uint8_t PhyDriverCtx::getPuschEqCoeffAlgo(void) const {
    return select_pusch_eqcoeffalgo;
}

uint8_t PhyDriverCtx::getPuschChEstAlgo(void) const {
    return select_pusch_chestalgo;
}

uint8_t PhyDriverCtx::getPuschEnablePerPrgChEst(void) const {
    return enable_pusch_perprgchest;
}

uint8_t PhyDriverCtx::getPuschTo(void) const {
    return enable_pusch_to;
}

uint8_t PhyDriverCtx::getPuschRssi(void) const {
    return enable_pusch_rssi;
}

uint8_t PhyDriverCtx::getPuschSinr(void) const {
    return enable_pusch_sinr;
}

uint8_t PhyDriverCtx::getPuxchPolarDcdrListSz(void) const {
    return mPuxchPolarDcdrListSz;
}

const std::string& PhyDriverCtx::getPuschrxChestFactorySettingsFilename() const noexcept {
    return mPuschrxChestFactorySettingsFilename;
}

uint8_t PhyDriverCtx::getNotifyUlHarqBufferRelease(void) const {
    return notify_ul_harq_buffer_release;
}

bool PhyDriverCtx::gpuCommDlEnabled(void) const {
    if(enable_gpu_comm_dl == 1)
        return true;
    return false;
}

bool PhyDriverCtx::gpuCommEnabledViaCpu(void) const {
    if(enable_gpu_comm_via_cpu == 1)
        return true;
    return false;
}

bool PhyDriverCtx::cpuCommEnabled(void) const {
    if(enable_cpu_init_comms == 1)
        return true;
    return false;
}

bool PhyDriverCtx::fixBetaDl(void) const {
    return (fix_beta_dl == 1);
}

uint8_t PhyDriverCtx::enableCPUTaskTracing(void) const {
    return enable_cpu_task_tracing;
}

bool PhyDriverCtx::enableL1ParamSanityCheck(void) const {
    return (enable_l1_param_sanity_check == 1);
}

bool PhyDriverCtx::enablePrepareTracing(void) const {
    return (enable_prepare_tracing == 1);
}

bool PhyDriverCtx::cuptiTracingEnabled(void) const {
    return (cupti_enable_tracing == 1);
}

uint64_t PhyDriverCtx::cuptiBufferSize(void) const {
    return cupti_buffer_size;
}

uint16_t PhyDriverCtx::cuptiNumBuffers(void) const {
    return cupti_num_buffers;
}

bool PhyDriverCtx::disableEmpw(void) const {
    return (disable_empw == 1);
}

bool PhyDriverCtx::enableDlCqeTracing(void) const {
    return (enable_dl_cqe_tracing == 1);
}

uint64_t PhyDriverCtx::get_cqe_trace_cell_mask(void) const {
    return cqe_trace_cell_mask;
}

uint32_t PhyDriverCtx::get_cqe_trace_slot_mask(void) const {
    return cqe_trace_slot_mask;
}

bool PhyDriverCtx::enableOKTb(void) const {
    return (enable_ok_tb == 1);
}

uint32_t PhyDriverCtx::get_num_ok_tb_slot(void) const {
    return num_ok_tb_slot;
}

uint8_t PhyDriverCtx::getUlRxPktTracingLevel(void) const {
    return ul_rx_pkt_tracing_level;
}

uint8_t PhyDriverCtx::getUlRxPktTracingLevelSrs(void) const {
    return ul_rx_pkt_tracing_level_srs;
}

uint32_t PhyDriverCtx::getUlWarmupFrameCount(void) const {
    return ul_warmup_frame_count;
}

uint8_t PhyDriverCtx::getPMUMetrics(void) const {
    return pmu_metrics;
}

uint8_t PhyDriverCtx::getmMIMO_enable(void) const {
    return mMIMO_enable;
}

uint8_t PhyDriverCtx::get_enable_srs(void) const {
    return enable_srs;
}

uint8_t PhyDriverCtx::get_enable_dl_core_affinity(void) const {
    return enable_dl_core_affinity;
}

uint8_t PhyDriverCtx::get_dlc_core_packing_scheme(void) const {
    return dlc_core_packing_scheme;
}

uint8_t PhyDriverCtx::getUeMode() const {
    return ue_mode;
}

uint8_t PhyDriverCtx::get_ch_segment_proc_enable() const {
    return mCh_segment_proc_enable;
}

uint32_t PhyDriverCtx::getAggr_obj_non_avail_th(void) const {
    return aggr_obj_non_avail_th;
}

uint32_t PhyDriverCtx::geth2d_copy_wait_th(void) const {
    return h2d_copy_wait_th;
}

void PhyDriverCtx::reset_h2d_copy_prepone_info()
{
    std::fill(h2d_cpy_info.begin(), h2d_cpy_info.end(), h2d_copy_prepone_info_t{});
}

uint32_t PhyDriverCtx::getcuphy_dl_channel_wait_th(void) const {
    return cuphy_dl_channel_wait_th;
}

uint32_t PhyDriverCtx::getSendCPlane_timing_error_th_ns(void) const {
    return sendCPlane_timing_error_th_ns;
}

uint32_t PhyDriverCtx::getSendCPlane_ulbfw_backoff_th_ns(void) const {
    return sendCPlane_ulbfw_backoff_th_ns;
}

uint32_t PhyDriverCtx::getSendCPlane_dlbfw_backoff_th_ns(void) const {
    return sendCPlane_dlbfw_backoff_th_ns;
}

bool PhyDriverCtx::ru_health_check_enabled(void) const {
    if(max_ru_unhealthy_ul_slots)
        return true;
    else
        return false;
}

uint32_t PhyDriverCtx::get_max_ru_unhealthy_slots(void) const {
    return max_ru_unhealthy_ul_slots;
}

cuphySrsChEstAlgoType_t PhyDriverCtx::get_srs_chest_algo_type()const {
    return srs_chest_algo_type;
}

uint8_t PhyDriverCtx::get_srs_chest_tol2_normalization_algo_type()const {
    return srs_chest_tol2_normalization_algo_type;
}

float PhyDriverCtx::get_srs_chest_tol2_constant_scaler()const {
    return srs_chest_tol2_constant_scaler;
}

uint8_t PhyDriverCtx::get_bfw_power_normalization_alg_selector()const {
    return bfw_power_normalization_alg_selector;
}

float PhyDriverCtx::get_bfw_beta_prescaler()const {
    return bfw_beta_prescaler;
}

uint32_t PhyDriverCtx::get_total_num_srs_chest_buffers()const {
    return total_num_srs_chest_buffers;
}

uint8_t PhyDriverCtx::get_ul_pcap_capture_enable()const {
    return ul_pcap_capture_enable;
}

uint16_t PhyDriverCtx::get_ul_pcap_capture_mtu()const {
    return ul_pcap_capture_mtu;
}

uint8_t PhyDriverCtx::get_send_static_bfw_wt_all_cplane()const {
    return send_static_bfw_wt_all_cplane;
}

void* PhyDriverCtx::getDlHBuffersAddr(int index)
{
    //return (CleanupDlBufInfo*)h_dl_buffers_addr.get() + (index % DL_HELPER_MEMSET_BUFFERS_PER_CTX)*PDSCH_MAX_CELLS_PER_CELL_GROUP;
    return (CleanupDlBufInfo*)h_dl_buffers_addr.get() + (index & (DL_HELPER_MEMSET_BUFFERS_PER_CTX - 1))*PDSCH_MAX_CELLS_PER_CELL_GROUP;
};

void* PhyDriverCtx::getDlDBuffersAddr(int index)
{
    //return (CleanupDlBufInfo*)d_dl_buffers_addr.get() + (index % DL_HELPER_MEMSET_BUFFERS_PER_CTX)*PDSCH_MAX_CELLS_PER_CELL_GROUP;
    return (CleanupDlBufInfo*)d_dl_buffers_addr.get() + (index & (DL_HELPER_MEMSET_BUFFERS_PER_CTX - 1))*PDSCH_MAX_CELLS_PER_CELL_GROUP;
}

void PhyDriverCtx::updateBatchedMemcpyInfo(void* dst_addr, void* src_addr, size_t count)
{
#if 0
    // If condition should never evaluate to true, since resetBatchedMemcpyBatches() is called when an L2 slot is dropped from cuphyl2adapter
    if (m_batchedMemcpyHelper.useBatchedMemcpy() &&  m_batchedMemcpyHelper.getMaxMemcopiesCount() <= batched_copies)
    {
        NVLOGE_FMT(TAG, AERIAL_THREAD_API_EVENT, "updateBatchedMemcpyInfo will error out because max count {} <= requested count of {}; reset memcpy count beforehand", m_batchedMemcpyHelper.getMaxMemcopiesCount(), batched_copies); // Behavior observed when a slot was dropped due to profiling overhead (when running through nsys). Resetting the count is a possible workaround.
        resetBatchedMemcpyBatches();
    }
#endif
    // batched_copies is a member variable updated on every such call. Needs to be reset via  resetBatchedMemcpyBatches().
    m_batchedMemcpyHelper.updateMemcpy(dst_addr, src_addr, count, cudaMemcpyHostToDevice, H2D_TB_CPY_stream);
}

cuphyStatus_t PhyDriverCtx::performBatchedMemcpy()
{
    return m_batchedMemcpyHelper.launchBatchedMemcpy(H2D_TB_CPY_stream);
}

void PhyDriverCtx::resetBatchedMemcpyBatches()
{
    m_batchedMemcpyHelper.reset();
}

bool PhyDriverCtx::getEnableTxNotification() const
{
    return !!enable_tx_notification;
}

aggr_obj_error_info_t* PhyDriverCtx::getAggrObjErrInfo(bool isDl)
{
    if(isDl)
        return &aggr_error_info_dl;
    else
        return &aggr_error_info_ul;
}

uint16_t PhyDriverCtx::getForcedNumCsi2Bits(void) const {
    return forcedNumCsi2Bits;
}

uint32_t PhyDriverCtx::getPuschMaxNumLdpcHetConfigs(void) const {
    return pusch_nMaxLdpcHetConfigs;
}

uint8_t PhyDriverCtx::getPuschMaxNumTbPerNode(void) const {
    return pusch_nMaxTbPerNode;
}

 DataLake* PhyDriverCtx::getDataLake(void) {
    return dataLake.get();
 }

 uint8_t* PhyDriverCtx::getFhBufOkTb(uint8_t cell_idx) const{
    return fh_buf_ok_tb[cell_idx];
 }

 uint8_t PhyDriverCtx:: getPuschAggrPerCtx() const {
    return pusch_aggr_per_ctx;
}

uint8_t PhyDriverCtx:: getPucchAggrPerCtx() const {
    return pucch_aggr_per_ctx;
}

uint8_t PhyDriverCtx:: getPrachAggrPerCtx() const {
    return prach_aggr_per_ctx;
}

uint8_t PhyDriverCtx:: getSrsAggrPerCtx() const {
    return srs_aggr_per_ctx;
}

uint8_t PhyDriverCtx:: getUlbfwAggrPerCtx() const {
    return ulbfw_aggr_per_ctx;
}

uint16_t PhyDriverCtx:: getMaxHarqPools() const {
    return max_harq_pools;
} 

uint8_t PhyDriverCtx:: getUlInputBufferPerCell() const {
    return ul_input_buffer_per_cell;
}

uint8_t PhyDriverCtx:: getUlInputBufferPerCellSrs() const {
    return ul_input_buffer_per_cell_srs;
}

bool PhyDriverCtx::getAggrObjFreeStatus() {

    for(int i = 0; i < getPuschAggrPerCtx(); i++)
    {
        if(aggr_pusch_items[i]->isActive()) {
            NVLOGI_FMT(TAG, "L1 Recovery : PUSCH[{}] not free", i);
            return false;
        }
    }

    for(int i = 0; i < getPucchAggrPerCtx(); i++)
    {
        if(aggr_pucch_items[i]->isActive()) {
            NVLOGI_FMT(TAG, "L1 Recovery : PUCCH[{}] not free", i);
            return false;}
    }

    for(int i = 0; i < getPrachAggrPerCtx(); i++)
    {
        if(aggr_prach_items[i]->isActive()) {
            NVLOGI_FMT(TAG, "L1 Recovery : PRACH[{}] not free", i);
            return false;}
    }

    if(this->mMIMO_enable)
    {
        for(int i = 0; i < getUlbfwAggrPerCtx(); i++)
        {
            if(aggr_ulbfw_items[i]->isActive()) {
                NVLOGI_FMT(TAG, "L1 Recovery : UL BFW[{}] not free", i);
                return false;
            }
        }
        
        for(int i = 0; i < PHY_DLBFW_AGGR_X_CTX; i++)
        {
            if(aggr_dlbfw_items[i]->isActive()) {
                NVLOGI_FMT(TAG, "L1 Recovery : DL BFW[{}] not free", i);
                return false;
            }
        }
    }

    if(this->enable_srs) 
    {
        for(int i = 0; i < getSrsAggrPerCtx(); i++)
        {
            if(aggr_srs_items[i]->isActive()) {
                NVLOGI_FMT(TAG, "L1 Recovery : SRS[{}] not free", i);
                return false;
            }
        }
    }

    for(int i = 0; i < PHY_PDSCH_AGGR_X_CTX; i++)
    {
        if(aggr_pdsch_items[i]->isActive()){
            NVLOGI_FMT(TAG, "L1 Recovery : PDSCH[{}] not free", i);
            return false;}
    }

    for(int i = 0; i < PHY_PDCCH_DL_AGGR_X_CTX; i++)
    {
        if(aggr_pdcch_dl_items[i]->isActive()) {
            NVLOGI_FMT(TAG, "L1 Recovery : PDCCH DL[{}] not free", i);
            return false;}
    }

    /*for(int i = 0; i < PHY_PDCCH_UL_AGGR_X_CTX; i++)
    {
        if(aggr_pdcch_ul_items[i]->isActive()) {
            NVLOGD_FMT(TAG, "L1 Recovery : PDCCH not free");
            return false;}
    }*/

    for(int i = 0; i < PHY_PBCH_AGGR_X_CTX; i++)
    {
        if(aggr_pbch_items[i]->isActive()) {
            NVLOGI_FMT(TAG, "L1 Recovery : PBCH[{}] not free", i);
            return false;}
    }

    for(int i = 0; i < PHY_CSIRS_AGGR_X_CTX; i++)
    {
        if(aggr_csirs_items[i]->isActive()){
            NVLOGI_FMT(TAG, "L1 Recovery : CSIRS[{}] not free", i);
            return false;}
    }

    return true;
}

bool PhyDriverCtx::incrL1RecoverySlots() {
    aggr_obj_error_info_t* errorInfoDl = getAggrObjErrInfo(true);
    if(++(errorInfoDl->l1RecoverySlots) >= default_mib_cycle)
    {
        NVLOGE_FMT(TAG, AERIAL_THREAD_API_EVENT, "L1 not responsive");
        return false;
    }
    else
    {
        NVLOGW_FMT(TAG, "L1 in recovery for {} slots",errorInfoDl->l1RecoverySlots);
        return true;
    }
}

bool PhyDriverCtx::incrAllObjFreeSlots()
{
    aggr_obj_error_info_t* errorInfoDl = getAggrObjErrInfo(true);
    if(++(errorInfoDl->availCount) >= getAggr_obj_non_avail_th()*2)
    {
        NVLOGW_FMT(TAG, "All cuPhy objects available for {} slots",errorInfoDl->availCount);
        return true;
    }
    else
        return false;
}

void PhyDriverCtx::resetAllObjFreeSlots()
{
    aggr_obj_error_info_t* errorInfoDl = getAggrObjErrInfo(true);
    errorInfoDl->availCount = 0;
}

void PhyDriverCtx::resetL1RecoverySlots()
{
    aggr_obj_error_info_t* errorInfoDl = getAggrObjErrInfo(true);
    errorInfoDl->l1RecoverySlots = 0;
}

uint8_t PhyDriverCtx::getEnableWeightedAverageCfo(void) const {
    return  getPuschSinr() == 2 && enable_weighted_average_cfo;
}

uint8_t PhyDriverCtx::getPuschAggrFactor() const {
    return pusch_aggr_factor;
}


void PhyDriverCtx::setPuschAggrFactor(const uint8_t pusch_aggr_factor_) {
    pusch_aggr_factor = pusch_aggr_factor_;
}

uint16_t PhyDriverCtx::getMaxHarqTxCountBundled() const {
    return max_harq_tx_count_bundled;
}

uint16_t PhyDriverCtx::getMaxHarqTxCountNonBundled() const {
    return max_harq_tx_count_non_bundled;
}

