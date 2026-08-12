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

#include "app_config.hpp"
#include "app_utils.hpp"
#include "yamlparser.hpp"
#include "cuphydriver.hpp"
#include "nv_phy_group.hpp"
#include "scf_5g_fapi.hpp"
#include "nv_phy_driver_proxy.hpp"
#include "cuphyoam.hpp"
#include <cuda_profiler_api.h>
#include <signal.h>

#include "nv_utils.h"
#include "nvlog_fmt.hpp"
#include "cuphy_pti.hpp"
#include "cupti_helper.hpp"
#include "ti_generic.hpp"
#include "exit_handler.hpp"

#include "ptp_service_status_checking.hpp"
#include <curl/curl.h>

#define TAG (NVLOG_TAG_BASE_CUPHY_CONTROLLER + 1) // "CTL.SCF"
#define TAG_STARTUP_TIMES (NVLOG_TAG_BASE_CUPHY_CONTROLLER + 5) // "CTL.STARTUP_TIMES"

phydriver_handle pdh; //!< cuPHYDriver handle pointer

static void l1_setPhydriverHandle();
static void l1_setFmtLogThreadId(pthread_t id);

/**
 * The nvlog exit handler for L1 to cleanup and exit
 */
static void nvlog_exit_handler(void)
{
    char thread_name[16];
    pthread_getname_np(pthread_self(), thread_name, 16);
    NVLOGC_FMT(TAG, "{}: nvlog exit handler called in thread [{}]", __func__, thread_name);

    // Call L1 exit handler
    l1_exit_handler();
}

/**
 * The signal handler to trigger L1 cleanup and exit
 * @param signum Signal number
 */
static void signal_handler(int signum)
{
    if (signum == SIGINT || signum == SIGTERM) {
        // Change exit_handler_flag to L1_EXIT to let msg_processing thread stop sending SLOT.indication
        exit_handler::getInstance().set_exit_handler_flag(exit_handler::L1_EXIT);
    }

    // Note: It's not async-signal-safe to print log in signal handler, but it's really necessary to add log to avoid silent exiting.
    NVLOGC_FMT(TAG, "[signal_handler] received signal {} - {} - {}", signum, sigabbrev_np(signum), sigdescr_np(signum));
}

int main(int argc,char* argv[]) {

    TI_GENERIC_INIT("cuphycontroller main",15);

    TI_GENERIC_ADD("Start Main");

    // curl_global_init should be called before any curl operations, RHOCP PTP events monitoring uses curl to communicate with the PTP producer.
    CURLcode curl_init_res = curl_global_init(CURL_GLOBAL_DEFAULT);
    if (curl_init_res != CURLE_OK) {
            NVLOGE_FMT(TAG, AERIAL_PTP_ERROR_EVENT, "curl_global_init failed with error: {}", curl_easy_strerror(curl_init_res));
            exit(EXIT_FAILURE);
    }

    // Debug starting CPU core issue
    printf("Started cuphycontroller on CPU core %d\n", sched_getcpu());

    pthread_setname_np(pthread_self(), "phy_init");
    nvlog_fmtlog_thread_init("phy_init");

    TI_GENERIC_ADD("Parse Cuphycontroller YAML");
    std::string config_file = "cuphycontroller";
    if(argc == 1)
    {
       config_file.append("_").append("F08");
    }
    else
    {
        for(int i = 1; i < argc; ++i)
        {
            if(argv[i] != NULL)
            {
                config_file.append("_").append(argv[i]);
            }
        }
    }

    config_file.append(".yaml");
    char config_full_path[MAX_PATH_LEN];
    get_full_path_file(config_full_path, CONFIG_YAML_FILE_PATH, config_file.c_str(), CONFIG_CUBB_ROOT_DIR_RELATIVE_NUM);
    config_file = std::string(config_full_path);
    NVLOGC_FMT(TAG, "Config file: {}", config_file.c_str());

    yaml::file_parser cfg_fp(config_full_path);
    yaml::document cfg_doc = cfg_fp.next_document();
    yaml::node cfg_node = cfg_doc.root();
    // Bind low-priority threads to configured core
    int low_priority_core = -1;
    if (cfg_node.has_key("low_priority_core")) {
        low_priority_core = cfg_node["low_priority_core"].as<int>();
    }
    if (low_priority_core >= 0) {
        nv_assign_thread_cpu_core(low_priority_core);
    }
    NVLOGC_FMT(TAG, "low_priority_core={}", low_priority_core);

    auto& appConfig = AppConfig::getInstance();
    appConfig.setLowPriorityCore(max(low_priority_core, 0));

    auto nic_tput_alert_threshold = cfg_node["nic_tput_alert_threshold_mbps"].as<int>();
    appConfig.setNicTputAlertThreshold(nic_tput_alert_threshold);

    auto enable_ptp_svc_monitoring = (uint8_t) static_cast<uint16_t>(cfg_node[YAML_PARAM_ENABLE_PTP_SVC_MONITORING]);
    if (enable_ptp_svc_monitoring)
    {
        auto ptp_rms_threshold = static_cast<int>(cfg_node[YAML_PARAM_PTP_RMS_THRESHOLD]);
        if(ptp_rms_threshold <= 0 || ptp_rms_threshold > 100)
        {
            NVLOGE_FMT(TAG, AERIAL_PTP_ERROR_EVENT, "Ptp rms threshold({}) error, currently we suggest set it in range [1~100]. Ideally rms should be a single digit value", ptp_rms_threshold);
            exit(EXIT_FAILURE);
        }
        NVLOGC_FMT(TAG, "Ptp service ptp_rms_threshold: {}", ptp_rms_threshold);
        std::string syslogPath = "/host/var/log/syslog"; // Bind-mounted path
        if (AppUtils::checkPtpServiceStatus(syslogPath, ptp_rms_threshold, ptp_rms_threshold))
        {
            NVLOGE_FMT(TAG, AERIAL_PTP_ERROR_EVENT, "Ptp service status error, quit...");
            exit(EXIT_FAILURE);
        }
        appConfig.setPtpRmsThreshold(ptp_rms_threshold);
    }
    appConfig.enablePtpSvcMonitoring(enable_ptp_svc_monitoring);

    auto enable_rhocp_ptp_events_monitoring = (cfg_node.has_key(YAML_PARAM_ENABLE_RHOCP_PTP_EVENTS_MONITORING) ?
        (uint8_t) static_cast<uint16_t>(cfg_node[YAML_PARAM_ENABLE_RHOCP_PTP_EVENTS_MONITORING]) : 0);
    if (enable_rhocp_ptp_events_monitoring && enable_ptp_svc_monitoring) {
        NVLOGE_FMT(TAG, AERIAL_PTP_ERROR_EVENT, "Both Rhocp PTP events monitoring and PTP service monitoring enabled, disable one of them");
        exit(EXIT_FAILURE);
    }
    if (enable_rhocp_ptp_events_monitoring)
    {   
        if(!cfg_node.has_key(YAML_PARAM_RHOCP_PTP_PUBLISHER) ||
           !cfg_node.has_key(YAML_PARAM_RHOCP_PTP_NODE_NAME) ||
           !cfg_node.has_key(YAML_PARAM_RHOCP_PTP_CONSUMER))
        {
            NVLOGE_FMT(TAG, AERIAL_PTP_ERROR_EVENT, "Rhocp PTP events monitoring enabled but missing required parameters in the config file");
            exit(EXIT_FAILURE);
        }
        std::string rhocp_ptp_publisher = cfg_node[YAML_PARAM_RHOCP_PTP_PUBLISHER].as<std::string>();
        std::string rhocp_ptp_node_name = cfg_node[YAML_PARAM_RHOCP_PTP_NODE_NAME].as<std::string>();
        std::string rhocp_ptp_consumer = cfg_node[YAML_PARAM_RHOCP_PTP_CONSUMER].as<std::string>();
        NVLOGC_FMT(TAG, "Rhocp PTP events monitoring enabled with publisher: {}, node name: {}, consumer: {}", rhocp_ptp_publisher, rhocp_ptp_node_name, rhocp_ptp_consumer);
        // Because no subscription is created yet, is it ok to not check rhocp events during the initialization phase? 
        appConfig.setRhocpPtpPublisher(rhocp_ptp_publisher);
        appConfig.setRhocpPtpNodeName(rhocp_ptp_node_name);
        appConfig.setRhocpPtpConsumer(rhocp_ptp_consumer);
    }
    
    appConfig.enableRhocpPtpEventsMonitoring(enable_rhocp_ptp_events_monitoring);
    TI_GENERIC_ADD("Init nvlog");

    // Relative path of this process is $cuBB_SDK/build/cuPHY-CP/cuphycontroller/examples/
    char root[1024];
    get_root_path(root, CONFIG_CUBB_ROOT_DIR_RELATIVE_NUM);
    std::string yaml_path = std::string(root).append(NVLOG_DEFAULT_CONFIG_FILE);
    pthread_t bg_thread_id = nvlog_fmtlog_init(yaml_path.c_str(), "phy.log",&nvlog_exit_handler);
    l1_setFmtLogThreadId(bg_thread_id);

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    YamlParser parser_cfg, parser_l2pattern;
    std::vector<phydriverwrk_handle> workers_descr(16);
    int ret = 0;
    std::vector<struct slot_command_api::slot_command *> scl;
    std::unique_ptr<nv::PHY_group> grp;

    yaml::file_parser fp(yaml_path.c_str());
    yaml::document    doc        = fp.next_document();
    yaml::node        root_node  = doc.root();

    if(parser_cfg.parse_file(config_file.c_str()) != 0)
    {
        exit(EXIT_FAILURE);
    }

    appConfig.setCellGroupNum(parser_cfg.get_cuphydriver_cell_group_num());
    parser_cfg.print_configs();

    //If we are using green contexts, set CUDA_DEVICE_MAX_CONNECTIONS to 32 before the first CUDA call which will initialize the driver (e.g., cudaSetDevice below)
    // Note that if we set this to 32, we will not be able to switch to MPS later, e.g., in cuphydriver context.cpp
    if (parser_cfg.get_cuphydriver_use_green_contexts() != 0)
    {
        int env_ret_val = setenv("CUDA_DEVICE_MAX_CONNECTIONS", "32", 1);
        if (env_ret_val != 0) {
            NVLOGC_FMT(TAG, "Failed to properly set CUDA_DEVICE_MAX_CONNECTIONS env. variable for green contexts.");
        }
        const char* updated_dev_max_connections_env_var = getenv("CUDA_DEVICE_MAX_CONNECTIONS");
        NVLOGC_FMT(TAG, "Updated CUDA_DEVICE_MAX_CONNECTIONS to {} for green contexts", updated_dev_max_connections_env_var);
    }


    TI_GENERIC_ADD("Cuda Set Device");
    cudaSetDevice(parser_cfg.get_cuphydriver_gpus()[0]);
    TI_GENERIC_ADD("Cuphy PTI Init");
    cuphy_pti_init(parser_cfg.get_cuphydriver_nics()[0].address.c_str());

    if (parser_cfg.get_cuphydriver_cupti_enable_tracing()) {
        cuphy_cupti_helper_init(
            parser_cfg.get_cuphydriver_cupti_buffer_size(),
            parser_cfg.get_cuphydriver_cupti_num_buffers());
    }

    try
    {
        // std::string nic_name = "aerial00";
        auto pcie_address = parser_cfg.get_cuphydriver_nics().at(0).address;
        std::string nic_name = AppUtils::get_nic_name(pcie_address);
        NVLOGC_FMT(TAG, "Network interface for PCIe address {} : {}", pcie_address, nic_name);
        AppUtils::get_phc_clock(nic_name.c_str());
    }
    catch (const std::exception &e)
    {
        NVLOGC_FMT(TAG, "get_phc_clock error: {}", e.what());
    }
    AppUtils::clock_sanity_check();

    TI_GENERIC_ADD("Init PHYDriver");
    pthread_setname_np(pthread_self(), "phy_drv_init");

    // Create and populate the context_config struct
    context_config ctx_cfg{};
    // Set GPU ID
    if (!parser_cfg.get_cuphydriver_gpus().empty()) {
        ctx_cfg.gpu_id = static_cast<int>(parser_cfg.get_cuphydriver_gpus()[0]);
    } else {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "No GPU ID provided");
        return EINVAL;
    }

    // Set all configuration parameters
    ctx_cfg.standalone = parser_cfg.get_cuphydriver_standalone();
    ctx_cfg.validation = parser_cfg.get_cuphydriver_validation();
    ctx_cfg.prometheus_cpu_core = parser_cfg.get_cuphydriver_prometheusthread();
    ctx_cfg.fh_cpu_core = parser_cfg.get_cuphydriver_dpdk_thread();
    ctx_cfg.dpdk_verbose_logs = parser_cfg.get_cuphydriver_dpdk_verbose_logs();
    ctx_cfg.accu_tx_sched_res_ns = parser_cfg.get_cuphydriver_accu_tx_sched_res_ns();
    ctx_cfg.accu_tx_sched_disable = parser_cfg.get_cuphydriver_accu_tx_sched_disable();
    ctx_cfg.fh_stats_dump_cpu_core = parser_cfg.get_cuphydriver_fh_stats_dump_cpu_core();
    ctx_cfg.pdump_client_thread = parser_cfg.get_cuphydriver_pdump_client_thread();
    ctx_cfg.dpdk_file_prefix = parser_cfg.get_cuphydriver_dpdk_file_prefix();
    ctx_cfg.workers_sched_priority = parser_cfg.get_cuphydriver_workers_sched_priority();
    ctx_cfg.start_section_id_srs = parser_cfg.get_cuphydriver_start_section_id_srs();
    ctx_cfg.start_section_id_prach = parser_cfg.get_cuphydriver_start_section_id_prach();
    ctx_cfg.enable_ul_cuphy_graphs = parser_cfg.get_cuphydriver_ul_cuphy_graphs();
    ctx_cfg.enable_dl_cuphy_graphs = parser_cfg.get_cuphydriver_dl_cuphy_graphs();
    ctx_cfg.ul_order_kernel_mode = parser_cfg.get_cuphydriver_ul_order_kernel_mode();
    ctx_cfg.ul_order_timeout_cpu_ns = parser_cfg.get_cuphydriver_timeout_cpu();
    ctx_cfg.ul_order_timeout_gpu_ns = parser_cfg.get_cuphydriver_timeout_gpu();
    ctx_cfg.ul_order_timeout_gpu_srs_ns = parser_cfg.get_cuphydriver_timeout_gpu_srs();
    ctx_cfg.ul_srs_aggr3_task_launch_offset_ns = parser_cfg.get_cuphydriver_ul_srs_aggr3_task_launch_offset_ns();
    ctx_cfg.ul_order_timeout_log_interval_ns = parser_cfg.get_cuphydriver_timeout_log_interval();
    ctx_cfg.ul_order_timeout_gpu_log_enable = parser_cfg.get_cuphydriver_timeout_gpu_log_enable();
    ctx_cfg.ul_order_max_rx_pkts = parser_cfg.get_cuphydriver_order_kernel_max_rx_pkts();
    ctx_cfg.ul_order_rx_pkts_timeout_ns = parser_cfg.get_cuphydriver_order_kernel_rx_pkts_timeout();
    ctx_cfg.cplane_disable = parser_cfg.get_cplane_disable();
    ctx_cfg.cell_mplane_list = parser_cfg.get_mplane_configs();
    ctx_cfg.ul_cores = parser_cfg.get_cuphydriver_workers_ul();
    ctx_cfg.dl_cores = parser_cfg.get_cuphydriver_workers_dl();
    ctx_cfg.dl_validation_cores = parser_cfg.get_cuphydriver_workers_dl_validation();
    ctx_cfg.debug_worker = parser_cfg.get_cuphydriver_debug_worker();
    ctx_cfg.data_core = parser_cfg.get_cuphydriver_data_core();
    ctx_cfg.datalake_db_write_enable = parser_cfg.get_cuphydriver_datalake_db_write_enable();
    ctx_cfg.datalake_samples = parser_cfg.get_cuphydriver_datalake_samples();
    ctx_cfg.datalake_address = parser_cfg.get_cuphydriver_datalake_address();
    ctx_cfg.datalake_engine = parser_cfg.get_cuphydriver_datalake_engine();
    ctx_cfg.datalake_data_types = parser_cfg.get_cuphydriver_datalake_data_types();
    ctx_cfg.datalake_store_failed_pdu = parser_cfg.get_cuphydriver_datalake_store_failed_pdu();
    ctx_cfg.num_rows_fh = parser_cfg.get_cuphydriver_num_rows_fh();
    ctx_cfg.num_rows_pusch = parser_cfg.get_cuphydriver_num_rows_pusch();
    ctx_cfg.num_rows_hest = parser_cfg.get_cuphydriver_num_rows_hest();
    ctx_cfg.num_rows_srs_iq = parser_cfg.get_cuphydriver_num_rows_srs_iq();
    ctx_cfg.num_rows_srs = parser_cfg.get_cuphydriver_num_rows_srs();
    ctx_cfg.num_rows_srs_hest = parser_cfg.get_cuphydriver_num_rows_srs_hest();
    ctx_cfg.e3_agent_enabled = parser_cfg.get_cuphydriver_e3_agent_enabled();
    ctx_cfg.e3_rep_port = parser_cfg.get_cuphydriver_e3_rep_port();
    ctx_cfg.e3_pub_port = parser_cfg.get_cuphydriver_e3_pub_port();
    ctx_cfg.e3_sub_port = parser_cfg.get_cuphydriver_e3_sub_port();
    ctx_cfg.datalake_drop_tables = parser_cfg.get_cuphydriver_datalake_drop_tables();
    ctx_cfg.use_green_contexts = parser_cfg.get_cuphydriver_use_green_contexts();
    ctx_cfg.use_gc_workqueues  = parser_cfg.get_cuphydriver_use_gc_workqueues();
    ctx_cfg.use_batched_memcpy = parser_cfg.get_cuphydriver_use_batched_memcpy();
    ctx_cfg.mps_sm_pusch = parser_cfg.get_cuphydriver_mps_sm_pusch();
    ctx_cfg.mps_sm_pucch = parser_cfg.get_cuphydriver_mps_sm_pucch();
    ctx_cfg.mps_sm_prach = parser_cfg.get_cuphydriver_mps_sm_prach();
    ctx_cfg.mps_sm_ul_order = parser_cfg.get_cuphydriver_mps_sm_ul_order();
    ctx_cfg.mps_sm_srs = parser_cfg.get_cuphydriver_mps_sm_srs();
    ctx_cfg.mps_sm_pdsch = parser_cfg.get_cuphydriver_mps_sm_pdsch();
    ctx_cfg.mps_sm_pdcch = parser_cfg.get_cuphydriver_mps_sm_pdcch();
    ctx_cfg.mps_sm_pbch = parser_cfg.get_cuphydriver_mps_sm_pbch();
    ctx_cfg.mps_sm_gpu_comms = parser_cfg.get_cuphydriver_mps_sm_gpu_comms();
    ctx_cfg.pdsch_fallback = parser_cfg.get_cuphydriver_pdsch_fallback();
    ctx_cfg.enable_gpu_comm_dl = parser_cfg.get_cuphydriver_gpu_init_comms_dl();
    ctx_cfg.enable_gpu_comm_via_cpu = parser_cfg.get_cuphydriver_gpu_init_comms_via_cpu();
    ctx_cfg.enable_cpu_init_comms = parser_cfg.get_cuphydriver_cpu_init_comms();
    ctx_cfg.cell_group = parser_cfg.get_cuphydriver_cell_group();
    ctx_cfg.cell_group_num = parser_cfg.get_cuphydriver_cell_group_num();
    ctx_cfg.pusch_workCancelMode = parser_cfg.get_cuphydriver_pusch_workCancelMode();
    ctx_cfg.enable_pusch_tdi = parser_cfg.get_cuphydriver_pusch_tdi();
    ctx_cfg.enable_pusch_cfo = parser_cfg.get_cuphydriver_pusch_cfo();
    ctx_cfg.enable_pusch_dftsofdm = parser_cfg.get_cuphydriver_pusch_dftsofdm();
    ctx_cfg.enable_pusch_tbsizecheck = parser_cfg.get_cuphydriver_pusch_tbsizecheck();
    ctx_cfg.pusch_deviceGraphLaunchEn = parser_cfg.get_cuphydriver_pusch_deviceGraphLaunchEn();
    ctx_cfg.pusch_waitTimeOutPreEarlyHarqUs = parser_cfg.get_cuphydriver_pusch_waitTimeOutPreEarlyHarqUs();
    ctx_cfg.pusch_waitTimeOutPostEarlyHarqUs = parser_cfg.get_cuphydriver_pusch_waitTimeOutPostEarlyHarqUs();
    ctx_cfg.enable_pusch_to = parser_cfg.get_cuphydriver_pusch_to();
    ctx_cfg.select_pusch_eqcoeffalgo = parser_cfg.get_cuphydriver_pusch_select_eqcoeffalgo();
    ctx_cfg.select_pusch_chestalgo = parser_cfg.get_cuphydriver_pusch_select_chestalgo();
    ctx_cfg.enable_pusch_perprgchest = parser_cfg.get_cuphydriver_pusch_enable_perprgchest();
    ctx_cfg.mPuxchPolarDcdrListSz = parser_cfg.get_cuphydriver_puxchPolarDcdrListSz();
    ctx_cfg.mPuschrxChestFactorySettingsFilename = parser_cfg.get_cuphydriver_puschrx_chest_factory_settings_filename();
    ctx_cfg.enable_pusch_rssi = parser_cfg.get_cuphydriver_pusch_rssi();
    ctx_cfg.enable_pusch_sinr = parser_cfg.get_cuphydriver_pusch_sinr();
    ctx_cfg.enable_weighted_average_cfo = parser_cfg.get_cuphydriver_pusch_weighted_average_cfo();
    ctx_cfg.fix_beta_dl = parser_cfg.get_cuphydriver_fix_beta_dl();
    ctx_cfg.disable_empw = parser_cfg.get_cuphydriver_disable_empw();
    ctx_cfg.enable_cpu_task_tracing = parser_cfg.get_cuphydriver_enable_cpu_task_tracing();
    ctx_cfg.enable_l1_param_sanity_check = parser_cfg.get_cuphydriver_enable_l1_param_sanity_check();
    ctx_cfg.enable_prepare_tracing = parser_cfg.get_cuphydriver_enable_prepare_tracing();
    ctx_cfg.cupti_enable_tracing = parser_cfg.get_cuphydriver_cupti_enable_tracing();
    ctx_cfg.cupti_buffer_size = parser_cfg.get_cuphydriver_cupti_buffer_size();
    ctx_cfg.cupti_num_buffers = parser_cfg.get_cuphydriver_cupti_num_buffers();
    ctx_cfg.enable_dl_cqe_tracing = parser_cfg.get_cuphydriver_enable_dl_cqe_tracing();
    ctx_cfg.cqe_trace_cell_mask = parser_cfg.get_cuphydriver_cqe_trace_cell_mask();
    ctx_cfg.cqe_trace_slot_mask = parser_cfg.get_cuphydriver_cqe_trace_slot_mask();
    ctx_cfg.enable_ok_tb = parser_cfg.get_cuphydriver_enable_ok_tb();
    ctx_cfg.num_ok_tb_slot = parser_cfg.get_cuphydriver_num_ok_tb_slot();
    ctx_cfg.ul_rx_pkt_tracing_level = parser_cfg.get_cuphydriver_ul_rx_pkt_tracing_level();
    ctx_cfg.ul_rx_pkt_tracing_level_srs = parser_cfg.get_cuphydriver_ul_rx_pkt_tracing_level_srs();
    ctx_cfg.ul_warmup_frame_count = parser_cfg.get_cuphydriver_ul_warmup_frame_count();
    ctx_cfg.pmu_metrics = parser_cfg.get_cuphydriver_pmu_metrics();

    // Set h2d copy thread config
    auto h2d_cfg = parser_cfg.get_cuphydriver_h2d_cpy_th_cfg();
    ctx_cfg.h2d_cpy_th_cfg.enable_h2d_copy_thread = h2d_cfg.enable_h2d_copy_thread;
    ctx_cfg.h2d_cpy_th_cfg.h2d_copy_thread_cpu_affinity = h2d_cfg.h2d_copy_thread_cpu_affinity;
    ctx_cfg.h2d_cpy_th_cfg.h2d_copy_thread_sched_priority = h2d_cfg.h2d_copy_thread_sched_priority;

    ctx_cfg.split_ul_cuda_streams = parser_cfg.get_cuphydriver_split_ul_cuda_streams();
    ctx_cfg.serialize_pucch_pusch = parser_cfg.get_cuphydriver_serialize_pucch_pusch();
    ctx_cfg.mMIMO_enable = parser_cfg.get_cuphydriver_mMIMO_enable();
    ctx_cfg.aggr_obj_non_avail_th = parser_cfg.get_cuphydriver_aggr_obj_non_avail_th();
    ctx_cfg.dl_wait_th_list = parser_cfg.get_cuphydriver_dl_wait_th();
    ctx_cfg.sendCPlane_timing_error_th_ns = parser_cfg.get_cuphydriver_sendCPlane_timing_error_th_ns();
    ctx_cfg.sendCPlane_ulbfw_backoff_th_ns = parser_cfg.get_cuphydriver_sendCPlane_ulbfw_backoff_th_ns();
    ctx_cfg.sendCPlane_dlbfw_backoff_th_ns = parser_cfg.get_cuphydriver_sendCPlane_dlbfw_backoff_th_ns();
    ctx_cfg.forcedNumCsi2Bits = parser_cfg.get_cuphydriver_forcedNumCsi2Bits();
    ctx_cfg.pusch_nMaxLdpcHetConfigs = parser_cfg.get_cuphydriver_pusch_nMaxLdpcHetConfigs();
    ctx_cfg.pusch_nMaxTbPerNode = parser_cfg.get_cuphydriver_pusch_nMaxTbPerNode();
    ctx_cfg.enable_srs = parser_cfg.get_cuphydriver_enable_srs();
    ctx_cfg.enable_dl_core_affinity = parser_cfg.get_cuphydriver_enable_dl_core_affinity();
    ctx_cfg.dlc_core_packing_scheme = parser_cfg.get_cuphydriver_dlc_core_packing_scheme();
    ctx_cfg.ue_mode = parser_cfg.get_cuphydriver_ue_mode();
    ctx_cfg.mCh_segment_proc_enable = parser_cfg.get_cuphydriver_ch_segment_proc_enable();
    ctx_cfg.pusch_aggr_per_ctx = parser_cfg.get_pusch_aggr_per_ctx();
    ctx_cfg.prach_aggr_per_ctx = parser_cfg.get_prach_aggr_per_ctx();
    ctx_cfg.max_harq_pools = parser_cfg.get_max_harq_pools();
    ctx_cfg.max_harq_tx_count_bundled = parser_cfg.get_max_harq_tx_count_bundled();
    ctx_cfg.max_harq_tx_count_non_bundled = parser_cfg.get_max_harq_tx_count_non_bundled();
    ctx_cfg.ul_input_buffer_per_cell = parser_cfg.get_ul_input_buffer_per_cell();
    ctx_cfg.pucch_aggr_per_ctx = parser_cfg.get_pucch_aggr_per_ctx();
    ctx_cfg.srs_aggr_per_ctx = parser_cfg.get_srs_aggr_per_ctx();
    ctx_cfg.ul_input_buffer_per_cell_srs = parser_cfg.get_ul_input_buffer_per_cell_srs();
    ctx_cfg.max_ru_unhealthy_ul_slots = parser_cfg.get_max_ru_unhealthy_ul_slots();
    ctx_cfg.ul_pcap_capture_enable = parser_cfg.get_ul_pcap_capture_enable();
    ctx_cfg.ul_pcap_capture_thread_cpu_affinity = parser_cfg.get_ul_pcap_capture_thread_cpu_affinity();
    ctx_cfg.ul_pcap_capture_thread_sched_priority = parser_cfg.get_ul_pcap_capture_thread_sched_priority();
    ctx_cfg.srs_chest_algo_type = parser_cfg.get_srs_chest_algo_type();
    ctx_cfg.srs_chest_tol2_normalization_algo_type = parser_cfg.get_srs_chest_tol2_normalization_algo_type();
    ctx_cfg.srs_chest_tol2_constant_scaler = parser_cfg.get_srs_chest_tol2_constant_scaler();
    ctx_cfg.bfw_power_normalization_alg_selector = parser_cfg.get_bfw_power_normalization_alg_selector();
    ctx_cfg.bfw_beta_prescaler = parser_cfg.get_bfw_beta_prescaler();
    ctx_cfg.total_num_srs_chest_buffers = parser_cfg.get_total_num_srs_chest_buffers();
    ctx_cfg.send_static_bfw_wt_all_cplane = parser_cfg.get_send_static_bfw_wt_all_cplane();
    ctx_cfg.pcap_logger_ul_cplane_enable = parser_cfg.get_pcap_logger_ul_cplane_enable();
    ctx_cfg.pcap_logger_dl_cplane_enable = parser_cfg.get_pcap_logger_dl_cplane_enable();
    ctx_cfg.pcap_logger_thread_cpu_affinity = parser_cfg.get_pcap_logger_thread_cpu_affinity();
    ctx_cfg.pcap_logger_thread_sched_prio = parser_cfg.get_pcap_logger_thread_sched_prio(); 
    ctx_cfg.pcap_logger_file_save_dir = parser_cfg.get_pcap_logger_file_save_dir();
    ctx_cfg.dlc_bfw_enable_divide_per_cell = parser_cfg.get_dlc_bfw_enable_divide_per_cell();
    ctx_cfg.ulc_bfw_enable_divide_per_cell = parser_cfg.get_ulc_bfw_enable_divide_per_cell();
    ctx_cfg.dlc_alloc_cplane_bfw_txq = parser_cfg.get_dlc_alloc_cplane_bfw_txq();
    ctx_cfg.ulc_alloc_cplane_bfw_txq = parser_cfg.get_ulc_alloc_cplane_bfw_txq();
    ctx_cfg.enable_tx_notification = parser_cfg.get_enable_tx_notification();
    ctx_cfg.notify_ul_harq_buffer_release = parser_cfg.get_notify_ul_harq_buffer_release();
    // Add NIC configurations
    ctx_cfg.nic_configs.reserve(parser_cfg.get_cuphydriver_nics().size());
    auto nics = parser_cfg.get_cuphydriver_nics();
    for (const auto& nic : nics) {
        auto txq_count_uplane = ctx_cfg.cell_group_num; // 1 UL U-Plane per cell
        auto txq_count_cplane = ctx_cfg.cell_group_num; // 1 DL C-Plane per cell
        txq_count_cplane += ctx_cfg.cell_group_num; // 1 UL C-Plane per cell

        if (ctx_cfg.mMIMO_enable)
        {
            if (ctx_cfg.dlc_alloc_cplane_bfw_txq || ctx_cfg.dlc_bfw_enable_divide_per_cell)
            {
                txq_count_cplane += ctx_cfg.cell_group_num; // 1 DL BFW C-Plane per cell
            }
            if (ctx_cfg.ulc_alloc_cplane_bfw_txq || ctx_cfg.ulc_bfw_enable_divide_per_cell)
            {
                txq_count_cplane += ctx_cfg.cell_group_num; // 1 UL BFW C-Plane per cell
            }
        }

        auto rxq_count = ctx_cfg.cell_group_num; // 1 non-SRS U-Plane RXQ per cell
        if (ctx_cfg.enable_srs)
        {
            rxq_count += ctx_cfg.cell_group_num; // 1 SRS U-Plane RXQ per cell
        }

        nic_cfg nic_config{
            .nic_bus_addr = nic.address,
            .nic_mtu      = nic.mtu,
            .cpu_mbuf_num = nic.cpu_mbuf_num,
            .tx_req_num   = nic.tx_req_num,
            .txq_count_uplane    = txq_count_uplane,
            .txq_count_cplane    = txq_count_cplane,
            .rxq_count    = rxq_count,
            .txq_size     = nic.txq_size,
            .rxq_size     = nic.rxq_size,
        };

        ctx_cfg.nic_configs.push_back(nic_config);
    }

    ctx_cfg.static_beam_id_start = parser_cfg.get_static_beam_id_start();
    ctx_cfg.static_beam_id_end = parser_cfg.get_static_beam_id_end();
    ctx_cfg.dynamic_beam_id_start = parser_cfg.get_dynamic_beam_id_start();
    ctx_cfg.dynamic_beam_id_end = parser_cfg.get_dynamic_beam_id_end();
    ctx_cfg.bfw_c_plane_chaining_mode = parser_cfg.get_bfw_c_plane_chaining_mode();
    ret = pc_init_phydriver(&pdh, ctx_cfg, workers_descr);

    l1_setPhydriverHandle();

    if(ret)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "pc_init_phydriver error {}", ret);
        EXIT_L1(EXIT_FAILURE);
    }

    NVLOGC_FMT(TAG, "====> PhyDriver initialized!");

    pthread_setname_np(pthread_self(), "phy_drv_proxy");

    TI_GENERIC_ADD("Init cuphydriver");
    if(parser_cfg.get_cuphydriver_standalone())
    {
        size_t slot_size = 0;
        if(parser_cfg.parse_standalone_config_file(config_file.c_str())) {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Standalone mode selected without specifying standalone config file");
            goto quit;
        }

        if(parser_l2pattern.parse_launch_pattern_file(parser_cfg.get_config_filename().c_str()))
        {
            NVLOGE_FMT(TAG, AERIAL_YAML_PARSER_EVENT, "Error parsing standalone config file {}", parser_cfg.get_config_filename().c_str());
            EXIT_L1(EXIT_FAILURE);
        }

        slot_size = parser_l2pattern.get_cuphydriver_standalone_slot_command_size();
        NVLOGC_FMT(TAG, "Slots are {}", slot_size);

        for(int i = 0; i < slot_size; i++)
        {
            scl.push_back(parser_l2pattern.get_cuphydriver_standalone_slot_command(i));
            if(scl[i]) {
                NVLOGC_FMT(TAG, "Slot {} cells {}", i, scl[i]->cells.size());
            }
        }

        if(pc_standalone_create_cells(pdh, parser_cfg.get_cell_configs()))
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Error creating cells");
            goto quit;
        }

        if(pc_start_l1(pdh))
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Error starting L1");
            goto quit;
        }

        if(pc_standalone_simulate_l2(pdh, 500, parser_cfg.get_cuphydriver_slots(), scl, 7, parser_cfg.get_cuphydriver_workers_sched_priority())) //Assuming MU always 1 (500us TTI)
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Error simulating L2 Adapter");
            goto quit;
        }
    }
    else
    {
        TI_GENERIC_ADD("Make PHYDriverProxy");
        nv::PHYDriverProxy::make(pdh, parser_cfg.get_mplane_configs());
        TI_GENERIC_ADD("Init SCF FAPI");
        scf_5g_fapi::init();
        TI_GENERIC_ADD("Create PHY_group");

        try
        {
            grp = std::make_unique<nv::PHY_group>(parser_cfg.get_l2adapter_filename().c_str());
        }
        catch (const std::exception &e)
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Failed to create PHY_group: {}", e.what());
            goto quit;
        }

        TI_GENERIC_ADD("Start PHY_group");
        grp->start();
        NVLOGC_FMT(TAG, "cuPHYController configured for {} cells", parser_cfg.get_cuphydriver_cell_group_num());
        NVLOGC_FMT(TAG, "====> cuPHYController initialized, L1 is ready!");

        TI_GENERIC_ADD("End Main");
        TI_GENERIC_ALL_NVLOGC(TAG_STARTUP_TIMES);

        ////////////////////////////////////////////////////////////////////////////////////////////////
        //// When profiling, just wait for a number of seconds and then quit                        ////
        ////////////////////////////////////////////////////////////////////////////////////////////////
        if(parser_cfg.get_cuphydriver_profiler_sec() > 0)
        {
            cudaProfilerStart();
            sleep(parser_cfg.get_cuphydriver_profiler_sec());
            cudaProfilerStop();
        }
        else
        {
            pthread_setname_np(pthread_self(), "phy_main");
            nvlog_fmtlog_thread_init("phy_main");
            if (grp)
            {
                NVLOGC_FMT(TAG, "Joining PHY_group");
                grp->join();
                NVLOGC_FMT(TAG, "Joined PHY_group");
            }
        }
        NVLOGC_FMT(TAG, "Shutting down CuphyOAM");
        CuphyOAM *oam = CuphyOAM::getInstance();
        oam->shutdown();
        oam->wait_shutdown();
    }

quit:
    // Stop CUPTI tracing (safe to call even if CUPTI wasn't initialized)
    NVLOGC_FMT(TAG, "Stopping CUPTI tracing");
    cuphy_cupti_helper_stop();

    NVLOGC_FMT(TAG, "Finalizing PHYDriver");
    ret = pc_finalize_phydriver(pdh);
    if(ret)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "pc_finalize_phydriver error {}", ret);
    }
    NVLOGC_FMT(TAG, "====> PhyDriver finalized! Exit from main function ...");
    nvlog_fmtlog_close(bg_thread_id);

    curl_global_cleanup();

    if (ret == EXIT_SUCCESS)
    {
        printf("EXIT successfully from main function\n");
    }
    else
    {
        printf("EXIT from main function with error_code=%d\n", ret);
    }

    return ret;
}

static void l1_setPhydriverHandle()
{
    l1_pdh=pdh;
}

static void l1_setFmtLogThreadId(pthread_t id)
{
    gBg_thread_id=id;
}

