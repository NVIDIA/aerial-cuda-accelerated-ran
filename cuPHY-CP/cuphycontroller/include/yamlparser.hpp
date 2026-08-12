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

#ifndef YAML_PARSER_H
#define YAML_PARSER_H

#include "yaml.hpp"
#include "cuphydriver_api.hpp"
#include <vector>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <libgen.h>
#include <unistd.h>

#define YAML_PARAM_CONFIG_FILENAME "config_filename"
#define YAML_PARAM_L2ADAPTER_FILENAME "l2adapter_filename"
#define YAML_PARAM_AERIAL_METRICS_BACKEND_ADDRESS "aerial_metrics_backend_address"
#define YAML_PARAM_STANDALONE_FILENAME "standalone_filename"

#define YAML_PARAM_ENABLE_PTP_SVC_MONITORING "enable_ptp_svc_monitoring"
#define YAML_PARAM_PTP_RMS_THRESHOLD "ptp_rms_threshold"
#define YAML_PARAM_ENABLE_RHOCP_PTP_EVENTS_MONITORING "enable_rhocp_ptp_events_monitoring"
#define YAML_PARAM_RHOCP_PTP_PUBLISHER "rhocp_ptp_publisher"
#define YAML_PARAM_RHOCP_PTP_NODE_NAME "rhocp_ptp_node_name"
#define YAML_PARAM_RHOCP_PTP_CONSUMER "rhocp_ptp_consumer"

/*
 * cuPHYDriver
 */
#define YAML_PARAM_CUPHYDRIVER_CONFIG "cuphydriver_config"
#define YAML_PARAM_VALIDATION "validation"
#define YAML_PARAM_STANDALONE "standalone"
#define YAML_PARAM_NSLOTS "num_slots"
#define YAML_PARAM_PROFILER_SEC "profiler_sec"
#define YAML_PARAM_WORKERS_UL "workers_ul"
#define YAML_PARAM_WORKERS_DL "workers_dl"
#define YAML_PARAM_DEBUG_WORKER "debug_worker"
#define YAML_PARAM_WORKERS_SCHED_PRIORITY "workers_sched_priority"
#define YAML_PARAM_PROMETHEUS_THREAD "prometheus_thread"
#define YAML_PARAM_START_SECTION_ID_SRS "start_section_id_srs"
#define YAML_PARAM_START_SECTION_ID_PRACH "start_section_id_prach"
#define YAML_PARAM_ENABLE_UL_CUPHY_GRAPHS "enable_ul_cuphy_graphs"
#define YAML_PARAM_ENABLE_DL_CUPHY_GRAPHS "enable_dl_cuphy_graphs"
#define YAML_PARAM_UL_ORDER_TIMEOUT_CPU_NS "ul_order_timeout_cpu_ns"
#define YAML_PARAM_UL_ORDER_KERNEL_MODE "ul_order_kernel_mode"
#define YAML_PARAM_UL_ORDER_TIMEOUT_GPU_NS "ul_order_timeout_gpu_ns"
#define YAML_PARAM_UL_ORDER_TIMEOUT_GPU_SRS_NS "ul_order_timeout_gpu_srs_ns"
#define YAML_PARAM_UL_ORDER_TIMEOUT_LOG_INTERVAL_NS "ul_order_timeout_log_interval_ns"
#define YAML_PARAM_UL_ORDER_TIMEOUT_GPU_LOG_ENABLE "ul_order_timeout_gpu_log_enable"
#define YAML_PARAM_UL_SRS_AGGR3_TASK_LAUNCH_OFFSET_NS "ul_srs_aggr3_task_launch_offset_ns"
#define YAML_PARAM_UL_ORDER_MAX_RX_PKTS    "ul_order_max_rx_pkts"
#define YAML_PARAM_UL_ORDER_RX_PKTS_TIMEOUT_NS    "ul_order_rx_pkts_timeout_ns"
#define YAML_PARAM_CPLANE_DISABLE "cplane_disable"
#define YAML_PARAM_DPDK_THREAD "dpdk_thread"
#define YAML_PARAM_DPDK_VERBOSE_LOGS "dpdk_verbose_logs"
#define YAML_PARAM_ACCU_TX_SCHED_RES_NS "accu_tx_sched_res_ns"
#define YAML_PARAM_ACCU_TX_SCHED_DISABLE "accu_tx_sched_disable"
#define YAML_PARAM_FH_STATS_DUMP_CPU_CORE "fh_stats_dump_cpu_core"
#define YAML_PARAM_PDUMP_CLIENT_THREAD "pdump_client_thread"
#define YAML_PARAM_DPDK_FILE_PREFIX "dpdk_file_prefix"
#define YAML_PARAM_LOGLVL "log_level"

#define YAML_PARAM_USE_GREEN_CONTEXTS "use_green_contexts"
#define YAML_PARAM_USE_GC_WORKQUEUES "use_gc_workqueues"
#define YAML_PARAM_USE_BATCHED_MEMCPY "use_batched_memcpy"
#define YAML_PARAM_MPS_SM_PUSCH "mps_sm_pusch"
#define YAML_PARAM_MPS_SM_PUCCH "mps_sm_pucch"
#define YAML_PARAM_MPS_SM_PRACH "mps_sm_prach"
#define YAML_PARAM_MPS_SM_UL_ORDER "mps_sm_ul_order"
#define YAML_PARAM_MPS_SM_SRS "mps_sm_srs"
#define YAML_PARAM_MPS_SM_PDSCH "mps_sm_pdsch"
#define YAML_PARAM_MPS_SM_PDCCH "mps_sm_pdcch"
#define YAML_PARAM_MPS_SM_PBCH "mps_sm_pbch"
#define YAML_PARAM_MPS_SM_GPU_COMMS "mps_sm_gpu_comms"

#define YAML_PARAM_PDSCH_FACLLBACK "pdsch_fallback"

#define YAML_PARAM_GPU_INIT_COMMS_DL "gpu_init_comms_dl"
#define YAML_PARAM_GPU_INIT_COMMS_VIA_CPU "gpu_init_comms_via_cpu"
#define YAML_PARAM_CPU_INIT_COMMS    "cpu_init_comms"


#define YAML_PARAM_CELL_GROUP "cell_group"
#define YAML_PARAM_CELL_GROUP_NUM "cell_group_num"

#define YAML_PARAM_PUSCH_WORKCANCELMODE "pusch_workCancelMode"
#define YAML_PARAM_PUSCH_TDI "pusch_tdi"
#define YAML_PARAM_PUSCH_CFO "pusch_cfo"
#define YAML_PARAM_PUSCH_DFTSOFDM "pusch_dftsofdm"
#define YAML_PARAM_PUSCH_TBSIZECHECK "pusch_tbsizecheck"
#define YAML_PARAM_PUSCH_DEVICEGRAPHLAUNCHEN "pusch_deviceGraphLaunchEn"
#define YAML_PARAM_PUSCH_WAIT_TIMEOUT_PRE_EARLY_HARQ_US "pusch_waitTimeOutPreEarlyHarqUs"
#define YAML_PARAM_PUSCH_WAIT_TIMEOUT_POST_EARLY_HARQ_US "pusch_waitTimeOutPostEarlyHarqUs"
#define YAML_PARAM_PUSCH_TO  "pusch_to"
#define YAML_PARAM_PUSCH_SELECT_EQCOEFFALGO "pusch_select_eqcoeffalgo"
#define YAML_PARAM_PUSCH_SELECT_CHESTALGO "pusch_select_chestalgo"
#define YAML_PARAM_PUSCH_ENABLE_PERPRGCHEST "pusch_enable_perprgchest"

#define YAML_PARAM_PUSCH_RSSI "pusch_rssi"
#define YAML_PARAM_PUSCH_SINR "pusch_sinr"
#define YAML_PARAM_PUSCH_WEIGHTED_AVERAGE_CFO "pusch_weighted_average_cfo"
#define YAML_PARAM_PUXCH_POLAR_DCDR_LIST_SZ "puxch_polarDcdrListSz"
#define YAML_PARAM_PUSCHRX_CHEST_FACTORY_SETTINGS_FILENAME "puschrx_chest_factory_settings_filename"

#define YAML_PARAM_ENABLE_L1_PARAM_SANITY_CHECK "enable_l1_param_sanity_check"
#define YAML_PARAM_ENABLE_CPU_TASK_TRACING "enable_cpu_task_tracing"
#define YAML_PARAM_ENABLE_PREPARE_TRACING "enable_prepare_tracing"
#define YAML_PARAM_CUPTI_ENABLE_TRACING "cupti_enable_tracing"
#define YAML_PARAM_CUPTI_BUFFER_SIZE "cupti_buffer_size"
#define YAML_PARAM_CUPTI_NUM_BUFFERS "cupti_num_buffers"
#define YAML_PARAM_CQE_TRACER_CONFIG "cqe_tracer_config"
#define YAML_PARAM_OK_TESTBENCH_CONFIG "ok_testbench_config"
#define YAML_PARAM_DATA_CONFIG "data_config"
#define YAML_PARAM_DISABLE_EMPW      "disable_empw"
#define YAML_PARAM_UL_RX_PKT_TRACING_LEVEL "ul_rx_pkt_tracing_level"
#define YAML_PARAM_UL_RX_PKT_TRACING_LEVEL_SRS "ul_rx_pkt_tracing_level_srs"
#define YAML_PARAM_UL_WARMUP_FRAME_COUNT "ul_warmup_frame_count"
#define YAML_PARAM_PMU_METRICS "pmu_metrics"

#define YAML_PARAM_ENABLE_H2D_COPY_THREAD "enable_h2d_copy_thread"
#define YAML_PARAM_H2D_COPY_THREAD_CPU_AFFINITY "h2d_copy_thread_cpu_affinity"
#define YAML_PARAM_H2D_COPY_THREAD_SCHED_PRIORITY "h2d_copy_thread_sched_priority"

#define YAML_PARAM_MMIMO_ENABLE "mMIMO_enable" 
#define YAML_PARAM_AGGR_OBJ_NON_AVAIL_TH "aggr_obj_non_avail_th"
#define YAML_PARAM_SPLIT_UL_CUDA_STREAMS "split_ul_cuda_streams"
#define YAML_PARAM_SERIALIZE_PUCCH_PUSCH "serialize_pucch_pusch"
#define YAML_PARAM_DL_WAIT_TH_NS "dl_wait_th_ns"
#define YAML_PARAM_SENDCPLANE_TIMING_ERROR_TH_NS "sendCPlane_timing_error_th_ns"
#define YAML_PARAM_SENDCPLANE_ULBFW_BACKOFF_TH_NS "sendCPlane_ulbfw_backoff_th_ns"
#define YAML_PARAM_SENDCPLANE_DLBFW_BACKOFF_TH_NS "sendCPlane_dlbfw_backoff_th_ns"
#define YAML_PARAM_ENABLE_SRS "enable_srs" 
#define YAML_PARAM_ENABLE_DL_CORE_AFFINITY "enable_dl_core_affinity"
#define YAML_PARAM_DLC_CORE_PACKING_SCHEME "dlc_core_packing_scheme"
#define YAML_PARAM_DLC_CORE_INDEX "dlc_core_index"
#define YAML_PARAM_UE_MODE "ue_mode"
#define YAML_PARAM_NOTIFY_UL_HARQ_BUFFER_RELEASE "notify_ul_harq_buffer_release"
#define YAML_PARAM_DL_VALIDATION_WORKERS "workers_dl_validation"

#define YAML_PARAM_GPUS "gpus"
#define YAML_PARAM_NICS "nics"
#define YAML_PARAM_NICS_NIC "nic"
#define YAML_PARAM_NICS_MTU "mtu"
#define YAML_PARAM_NICS_CPU_MBUFS "cpu_mbufs"
#define YAML_PARAM_NICS_UPLANE_TX_HANDLES "uplane_tx_handles"
#define YAML_PARAM_NICS_TXQ_SIZE "txq_size"
#define YAML_PARAM_NICS_RXQ_SIZE "rxq_size"
#define YAML_PARAM_NICS_GPU "gpu"
#define YAML_PARAM_MCH_SEGMENT_PROC_ENABLE "mCh_segment_proc_enable" 

#define YAML_PARAM_CUS_PORT_FAILOVER "cus_port_failover"

#define YAML_PARAM_PUSCH_AGGR_PER_CTX "pusch_aggr_per_ctx"
#define YAML_PARAM_PRACH_AGGR_PER_CTX "prach_aggr_per_ctx"
#define YAML_PARAM_PUCCH_AGGR_PER_CTX "pucch_aggr_per_ctx"
#define YAML_PARAM_SRS_AGGR_PER_CTX "srs_aggr_per_ctx"
#define YAML_PARAM_MAX_HARQ_POOLS "max_harq_pools"
#define YAML_PARAM_MAX_HARQ_TX_COUNT_BUNDLED "max_harq_tx_count_bundled"
#define YAML_PARAM_MAX_HARQ_TX_COUNT_NON_BUNDLED "max_harq_tx_count_non_bundled"
#define YAML_PARAM_UL_INPUT_BUFFER_NUM_PER_CELL "ul_input_buffer_per_cell"
#define YAML_PARAM_UL_INPUT_BUFFER_NUM_PER_CELL_SRS "ul_input_buffer_per_cell_srs"
#define YAML_PARAM_MAX_RU_UNHEALTHY_UL_SLOTS "max_ru_unhealthy_ul_slots"
#define YAML_PARAM_SRS_CHEST_ALGO_TYPE "srs_chest_algo_type"
#define YAML_PARAM_SRS_CHEST_TOL2_NORMALIZATION_ALGO_TYPE "srs_chest_tol2_normalization_algo_type"
#define YAML_PARAM_SRS_CHEST_TOL2_CONSTANT_SCALER "srs_chest_tol2_constant_scaler"
#define YAML_PARAM_BFW_POWER_NORMALIZATION_ALG_SELECTOR "bfw_power_normalization_alg_selector"
#define YAML_PARAM_BFW_BETA_PRESCALER "bfw_beta_prescaler"
#define YAML_PARAM_TOTAL_NUM_SRS_CHEST_BUFFERS "total_num_srs_chest_buffers"
#define YAML_PARAM_SEND_STATIC_BFW_WT_ALL_CPLANE "send_static_bfw_wt_all_cplane"
#define YAML_PARAM_UL_PCAP_CAPTURE_ENABLE "ul_pcap_capture_enable"
#define YAML_PARAM_UL_PCAP_CAPTURE_THREAD_CPU_AFFINITY "ul_pcap_capture_thread_cpu_affinity"
#define YAML_PARAM_UL_PCAP_CAPTURE_THREAD_SCHED_PRIORITY "ul_pcap_capture_thread_sched_priority"

// PCAP logger YAML config parameters
#define YAML_PARAM_PCAP_LOGGER__ENABLE_UL_CPLANE "pcap_logger_ul_cplane_enable"
#define YAML_PARAM_PCAP_LOGGER__ENABLE_DL_CPLANE "pcap_logger_dl_cplane_enable"
#define YAML_PARAM_PCAP_LOGGER__THREAD_CPU_AFFINITY "pcap_logger_thread_cpu_affinity"
#define YAML_PARAM_PCAP_LOGGER__THREAD_SCHED_PRIO "pcap_logger_thread_sched_prio"
#define YAML_PARAM_PCAP_LOGGER__FILE_SAVE_DIR "pcap_logger_file_save_dir"
#define YAML_PARAM_DLC_BFW_TX_WINDOW_SIZE_NS "dlc_bfw_tx_window_size_ns"
#define YAML_PARAM_DLC_BFW_TX_WINDOW_BUFFER_NS "dlc_bfw_tx_window_buffer_ns"
#define YAML_PARAM_DLC_BFW_ENABLE_DIVIDE_PER_CELL "dlc_bfw_enable_divide_per_cell"
#define YAML_PARAM_DLC_ALLOC_CPLANE_BFW_TXQ "dlc_alloc_cplane_bfw_txq"

#define YAML_PARAM_ULC_BFW_TX_WINDOW_SIZE_NS "ulc_bfw_tx_window_size_ns"
#define YAML_PARAM_ULC_BFW_TX_WINDOW_BUFFER_NS "ulc_bfw_tx_window_buffer_ns"
#define YAML_PARAM_ULC_BFW_ENABLE_DIVIDE_PER_CELL "ulc_bfw_enable_divide_per_cell"
#define YAML_PARAM_ULC_ALLOC_CPLANE_BFW_TXQ "ulc_alloc_cplane_bfw_txq"

#define YAML_PARAM_STATIC_BEAM_ID_START "static_beam_id_start"
#define YAML_PARAM_STATIC_BEAM_ID_END "static_beam_id_end"
#define YAML_PARAM_DYNAMIC_BEAM_ID_START "dynamic_beam_id_start"
#define YAML_PARAM_DYNAMIC_BEAM_ID_END "dynamic_beam_id_end"
#define YAML_PARAM_BFW_C_PLANE_CHAINING_MODE "bfw_c_plane_chaining_mode"
#define YAML_PARAM_ENABLE_TX_NOTIFICATION "enable_tx_notification"
/*
 * Cells
 */
#define YAML_PARAM_CELLS "cells"
#define YAML_PARAM_CELL_ID "cell_id"
#define YAML_PARAM_CELL_NAME "name"
#define YAML_PARAM_CELL_RU_TYPE "ru_type"
#define YAML_PARAM_CELL_SRC_MAC_ADDR "src_mac_addr"
#define YAML_PARAM_CELL_DST_MAC_ADDR "dst_mac_addr"
#define YAML_PARAM_CELL_TXQ_COUNT_UPLANE "txq_count_uplane"
#define YAML_PARAM_CELL_VLAN "vlan"
#define YAML_PARAM_CELL_PCP "pcp"
#define YAML_PARAM_CELL_UPLANE_TXQS "uplane_txqs"
#define YAML_PARAM_CELL_NIC "nic"

#define YAML_PARAM_CELL_PUSCH_PRB_STRIDE "pusch_prb_stride"
#define YAML_PARAM_CELL_PRACH_PRB_STRIDE "prach_prb_stride"
#define YAML_PARAM_CELL_SRS_PRB_STRIDE "srs_prb_stride"
#define YAML_PARAM_CELL_PUSCH_LDPC_MAX_NUM_ITR_ALGO_TYPE "pusch_ldpc_max_num_itr_algo_type"
#define YAML_PARAM_CELL_PUSCH_FIXED_MAX_NUM_LDPC_ITRS "pusch_fixed_max_num_ldpc_itrs"
#define YAML_PARAM_CELL_PUSCH_LDPC_EARLY_TERMINATION "pusch_ldpc_early_termination"
#define YAML_PARAM_CELL_PUSCH_LDPC_ALGO_INDEX "pusch_ldpc_algo_index"
#define YAML_PARAM_CELL_PUSCH_LDPC_FLAGS "pusch_ldpc_flags"
#define YAML_PARAM_CELL_PUSCH_LDPC_USE_HALF "pusch_ldpc_use_half"
#define YAML_PARAM_CELL_PUSCH_NMAXPRB "pusch_nMaxPrb"
#define YAML_PARAM_CELL_PUSCH_NMAXRX "pusch_nMaxRx"

#define YAML_PARAM_CELL_DL_IQ_DATA_FMT "dl_iq_data_fmt"
#define YAML_PARAM_CELL_UL_IQ_DATA_FMT "ul_iq_data_fmt"
#define YAML_PARAM_CELL_COMP_METH "comp_meth"
#define YAML_PARAM_CELL_BIT_WIDTH "bit_width"
#define YAML_PARAM_CELL_FS_OFFSET_DL "fs_offset_dl"
#define YAML_PARAM_CELL_EXPONENT_DL "exponent_dl"
#define YAML_PARAM_CELL_REF_DL "ref_dl"
#define YAML_PARAM_CELL_FS_OFFSET_UL "fs_offset_ul"
#define YAML_PARAM_CELL_EXPONENT_UL "exponent_ul"
#define YAML_PARAM_CELL_SECTION3_TIME_OFFSET "section_3_time_offset"
#define YAML_PARAM_CELL_MAX_AMP_UL "max_amp_ul"
#define YAML_PARAM_CELL_MU "mu"

#define YAML_PARAM_CELL_T1A_MAX_UP_NS "T1a_max_up_ns"
#define YAML_PARAM_CELL_T1A_MAX_CP_UL_NS "T1a_max_cp_ul_ns"
#define YAML_PARAM_CELL_T1A_MIN_CP_UL_NS "T1a_min_cp_ul_ns"
#define YAML_PARAM_CELL_TA4_MIN_NS "Ta4_min_ns"
#define YAML_PARAM_CELL_TA4_MAX_NS "Ta4_max_ns"
#define YAML_PARAM_CELL_TA4_MIN_NS_SRS "Ta4_min_ns_srs"
#define YAML_PARAM_CELL_TA4_MAX_NS_SRS "Ta4_max_ns_srs"
#define YAML_PARAM_CELL_TCP_ADV_DL_NS "Tcp_adv_dl_ns"
#define YAML_PARAM_CELL_T1A_UP_NS "T1a_up_ns"
#define YAML_PARAM_CELL_T1A_CP_UL_NS "T1a_cp_ul_ns"
#define YAML_PARAM_CELL_T1A_MIN_CP_DL_NS "T1a_min_cp_dl_ns"
#define YAML_PARAM_CELL_T1A_MAX_CP_DL_NS "T1a_max_cp_dl_ns"
#define YAML_PARAM_CELL_UL_U_PLANE_TX_OFFSET_NS "ul_u_plane_tx_offset_ns"
#define YAML_PARAM_CELL_MAX_FH_LEN "fh_len_range"
#define YAML_PARAM_CELL_EAXC_ID_SSB_PBCH "eAxC_id_ssb_pbch"
#define YAML_PARAM_CELL_EAXC_ID_PDCCH "eAxC_id_pdcch"
#define YAML_PARAM_CELL_EAXC_ID_PDSCH "eAxC_id_pdsch"
#define YAML_PARAM_CELL_EAXC_ID_CSIRS "eAxC_id_csirs"
#define YAML_PARAM_CELL_EAXC_ID_PUSCH "eAxC_id_pusch"
#define YAML_PARAM_CELL_EAXC_ID_PUCCH "eAxC_id_pucch"
#define YAML_PARAM_CELL_EAXC_ID_SRS "eAxC_id_srs"
#define YAML_PARAM_CELL_EAXC_ID_PRACH "eAxC_id_prach"
#define YAML_PARAM_CELL_TV_PUSCH "tv_pusch"
#define YAML_PARAM_CELL_TV_SRS "tv_pusch"
#define YAML_PARAM_CELL_UL_GAIN_CALIBRATION "ul_gain_calibration"
#define YAML_PARAM_CELL_LOWER_GUARD_BW "lower_guard_bw"
#define YAML_PARAM_PUSCH_FORCE_NUM_CSI2_BITS "pusch_forcedNumCsi2Bits"
#define YAML_PARAM_PUSCH_N_MAX_LDPC_HET_CONFIGS "pusch_nMaxLdpcHetConfigs"
#define YAML_PARAM_PUSCH_N_MAX_TB_PER_NODE "pusch_nMaxTbPerNode"

// Launch pattern keys
#define YAML_LP_SLOTS "slots"
#define YAML_LP_CELLS "cells"
#define YAML_LP_PUSCH "pusch"
#define YAML_LP_PDSCH "pdsch"
#define YAML_LP_PDCCH_DL "pdcch_dl"
#define YAML_LP_PDCCH_UL "pdcch_ul"
#define YAML_LP_PBCH "pbch"
#define YAML_LP_PRACH "prach"

#define DEFAULT_MAX_4T4R_RXANT 4

#define MAX_PATH_LEN 1024

#define CONFIG_CUBB_ROOT_DIR_RELATIVE_NUM 4 // Set CUBB_HOME to N level parent directory of this process. Example: 4 means "../../../../"
#define CONFIG_YAML_FILE_PATH "cuPHY-CP/cuphycontroller/config/"
#define CONFIG_LAUNCH_PATTERN_PATH "testVectors/multi-cell/"
#define CONFIG_TEST_VECTOR_PATH "testVectors/"

struct nic_config {
    std::string address;
    uint16_t mtu;
    uint32_t cpu_mbuf_num;
    uint32_t tx_req_num;
    uint16_t txq_size;
    uint16_t rxq_size;
    int gpu;
};

struct cuphydriver_config
{
    uint16_t validation;
    uint16_t standalone;
    int prometheus_thread;
    uint32_t workers_sched_priority;
    uint16_t start_section_id_srs;
    uint16_t start_section_id_prach;
    uint16_t num_slots;
    uint8_t enable_ul_cuphy_graphs;
    uint8_t enable_dl_cuphy_graphs;
    uint32_t ul_order_timeout_cpu_ns;
    uint32_t ul_order_timeout_gpu_ns;
    uint32_t ul_order_timeout_gpu_srs_ns;
    uint32_t ul_order_timeout_log_interval_ns;
    uint32_t ul_srs_aggr3_task_launch_offset_ns;
    uint8_t ul_order_kernel_mode;
    uint8_t ul_order_timeout_gpu_log_enable;
    uint32_t ul_order_max_rx_pkts;
    uint32_t ul_order_rx_pkts_timeout_ns;
    uint8_t cplane_disable;
    int profiler_sec;
    l1_log_level log_level;
    uint32_t dpdk_thread;
    uint8_t dpdk_verbose_logs;
    uint32_t accu_tx_sched_res_ns;
    uint8_t accu_tx_sched_disable;
    int fh_stats_dump_cpu_core;
    int pdump_client_thread;
    std::string dpdk_file_prefix;
    uint8_t  use_green_contexts;
    uint8_t  use_gc_workqueues;
    uint8_t  use_batched_memcpy;
    uint32_t mps_sm_pusch;
    uint32_t mps_sm_pucch;
    uint32_t mps_sm_prach;
    uint32_t mps_sm_ul_order;
    uint32_t mps_sm_srs;
    uint32_t mps_sm_pdsch;
    uint32_t mps_sm_pdcch;
    uint32_t mps_sm_pbch;
    uint32_t mps_sm_gpu_comms;
    uint8_t pdsch_fallback;
    uint8_t gpu_init_comms_dl;
    uint8_t gpu_init_comms_via_cpu;
    uint8_t cpu_init_comms;
    uint8_t cell_group;
    uint8_t cell_group_num;
    std::vector<uint8_t> workers_list_ul;
    std::vector<uint8_t> workers_list_dl;
    int16_t debug_worker;
    int16_t data_core;
    uint8_t datalake_db_write_enable;
    uint32_t datalake_samples;
    std::string datalake_address;
    std::string datalake_engine;
    std::vector<std::string> datalake_data_types;
    uint8_t datalake_store_failed_pdu;
    uint32_t num_rows_fh;
    uint32_t num_rows_pusch;
    uint32_t num_rows_hest;
    uint32_t num_rows_srs_iq;
    uint32_t num_rows_srs;
    uint32_t num_rows_srs_hest;
    uint8_t e3_agent_enabled;
    uint16_t e3_rep_port;
    uint16_t e3_pub_port;
    uint16_t e3_sub_port;
    uint8_t datalake_drop_tables;
    std::vector<uint16_t> gpus_list;
    std::vector<struct nic_config> nics_list;
    uint8_t pusch_workCancelMode;
    uint8_t puschTdi;
    uint8_t puschCfo;
    uint8_t puschDftSOfdm;
    uint8_t puschTbSizeCheck;
    uint8_t pusch_deviceGraphLaunchEn;
    uint16_t pusch_waitTimeOutPreEarlyHarqUs;
    uint16_t pusch_waitTimeOutPostEarlyHarqUs;
    uint8_t puschTo;
    uint8_t puschSelectEqCoeffAlgo;
    uint8_t puschSelectChEstAlgo;
    uint8_t puschEnablePerPrgChEst;
    uint8_t puschRssi;
    uint8_t puschSinr;
    uint8_t puschWeightedAverageCfo;
    std::string puschrxChestFactorySettingsFilename;
    uint8_t puxchPolarDcdrListSz;
    int fix_beta_dl;
    uint8_t enable_l1_param_sanity_check;
    uint8_t enable_cpu_task_tracing;
    uint8_t enable_prepare_tracing;
    uint8_t cupti_enable_tracing;
    uint64_t cupti_buffer_size;
    uint16_t cupti_num_buffers;
    uint8_t disable_empw;
    uint8_t enable_dl_cqe_tracing;
    uint64_t cqe_trace_cell_mask;
    uint32_t cqe_trace_slot_mask;
    uint8_t enable_ok_tb;
    uint32_t num_ok_tb_slot;
    uint8_t ul_rx_pkt_tracing_level;
    uint8_t ul_rx_pkt_tracing_level_srs;
    uint32_t ul_warmup_frame_count;
    uint8_t pmu_metrics;
    struct h2d_copy_thread_config h2d_cpy_th_cfg;
    uint8_t mMIMO_enable;
    uint32_t aggr_obj_non_avail_th;
    int split_ul_cuda_streams;
    int serialize_pucch_pusch;
    std::vector<uint32_t> dl_wait_th_list;
    uint32_t sendCPlane_timing_error_th_ns;
    uint32_t sendCPlane_ulbfw_backoff_th_ns;
    uint32_t sendCPlane_dlbfw_backoff_th_ns;
    uint16_t forcedNumCsi2Bits;
    uint32_t pusch_nMaxLdpcHetConfigs;
    uint8_t pusch_nMaxTbPerNode;
    uint8_t enable_srs;
    uint8_t enable_dl_core_affinity;
    uint8_t dlc_core_packing_scheme;
    uint8_t ue_mode;
    std::vector<uint8_t> workers_dl_validation;
    uint8_t mCh_segment_proc_enable;
    uint8_t pusch_aggr_per_ctx;
    uint8_t prach_aggr_per_ctx;
    uint8_t pucch_aggr_per_ctx;
    uint8_t srs_aggr_per_ctx;
    uint16_t max_harq_pools;
    uint16_t max_harq_tx_count_bundled;
    uint16_t max_harq_tx_count_non_bundled;
    uint8_t ul_input_buffer_per_cell;
    uint8_t ul_input_buffer_per_cell_srs;
    uint32_t max_ru_unhealthy_ul_slots;
    uint8_t ul_pcap_capture_enable;
    uint8_t ul_pcap_capture_thread_cpu_affinity;
    uint8_t ul_pcap_capture_thread_sched_priority;
    // PCAP logger configurations. 
    uint8_t pcap_logger_ul_cplane_enable;
    uint8_t pcap_logger_dl_cplane_enable;  
    uint8_t pcap_logger_thread_cpu_affinity;
    uint8_t pcap_logger_thread_sched_prio;   
    std::string pcap_logger_file_save_dir;
    
    uint8_t srs_chest_algo_type;
    uint8_t srs_chest_tol2_normalization_algo_type;
    float   srs_chest_tol2_constant_scaler;
    uint8_t bfw_power_normalization_alg_selector;
    float   bfw_beta_prescaler;
    uint32_t total_num_srs_chest_buffers;
    uint8_t send_static_bfw_wt_all_cplane;
    uint8_t dlc_bfw_enable_divide_per_cell;
    uint8_t ulc_bfw_enable_divide_per_cell;
    uint8_t dlc_alloc_cplane_bfw_txq;
    uint8_t ulc_alloc_cplane_bfw_txq;
    
    uint16_t static_beam_id_start;
    uint16_t static_beam_id_end;
    uint16_t dynamic_beam_id_start;
    uint16_t dynamic_beam_id_end;
    uint8_t bfw_c_plane_chaining_mode;
    uint8_t enable_tx_notification;
    uint8_t notify_ul_harq_buffer_release;
};

class YamlParser
{
public:
    int parse_file(const char* filename);
    int parse_launch_pattern_file(const char* filename);
    int parse_standalone_config_file(const char* filename);
    void print_configs() const;
    std::vector<struct cell_phy_info>& get_cell_configs() { return cell_configs; };
    std::vector<::cell_mplane_info>& get_mplane_configs() { return mplane_configs; };

    std::string get_config_filename() const { return cuphycontroller_config_filename; };
    std::string get_l2adapter_filename() const { return l2adapter_config_filename; };
    std::string get_standalone_filename() const { return standalone_config_filename; };
    std::vector<uint8_t>& get_cuphydriver_workers_ul();
    std::vector<uint8_t>& get_cuphydriver_workers_dl();
    std::vector<uint8_t>& get_cuphydriver_workers_dl_validation();
    std::vector<uint16_t>& get_cuphydriver_gpus();
    std::vector<struct nic_config> get_cuphydriver_nics();
    l1_log_level& get_cuphydriver_loglevel();
    uint16_t& get_cuphydriver_validation();
    uint16_t& get_cuphydriver_start_section_id_srs();
    uint16_t& get_cuphydriver_start_section_id_prach();
    uint16_t& get_cuphydriver_standalone();
    int& get_cuphydriver_profiler_sec();
    int& get_cuphydriver_prometheusthread();
    uint32_t& get_cuphydriver_dpdk_thread();
    uint8_t& get_cuphydriver_dpdk_verbose_logs();
    uint32_t& get_cuphydriver_accu_tx_sched_res_ns();
    uint8_t& get_cuphydriver_accu_tx_sched_disable();
    int& get_cuphydriver_pdump_client_thread();
    int& get_cuphydriver_fh_stats_dump_cpu_core();
    std::string& get_cuphydriver_dpdk_file_prefix();
    uint32_t& get_cuphydriver_workers_sched_priority();
    int16_t get_cuphydriver_debug_worker();
    int16_t get_cuphydriver_data_core();
    uint32_t get_cuphydriver_datalake_samples();
    std::string& get_cuphydriver_datalake_address();
    std::string& get_cuphydriver_datalake_engine();
    uint32_t get_cuphydriver_num_rows_fh();
    uint32_t get_cuphydriver_num_rows_pusch();
    uint32_t get_cuphydriver_num_rows_hest();
    uint32_t get_cuphydriver_num_rows_srs_iq();
    uint32_t get_cuphydriver_num_rows_srs();
    uint32_t get_cuphydriver_num_rows_srs_hest();
    uint8_t get_cuphydriver_e3_agent_enabled();
    uint16_t get_cuphydriver_e3_rep_port();
    uint16_t get_cuphydriver_e3_pub_port();
    uint16_t get_cuphydriver_e3_sub_port();
    uint8_t get_cuphydriver_datalake_drop_tables();
    uint8_t get_cuphydriver_datalake_store_failed_pdu();
    uint8_t get_cuphydriver_datalake_db_write_enable();
    std::vector<std::string>& get_cuphydriver_datalake_data_types();
    uint16_t& get_cuphydriver_slots();
    uint16_t& get_cuphydriver_section3_time_offset();
    int& get_cuphydriver_section3_freq_offset();
    uint8_t& get_cuphydriver_ul_cuphy_graphs();
    uint8_t& get_cuphydriver_dl_cuphy_graphs();
    uint32_t& get_cuphydriver_timeout_cpu();
    uint32_t& get_cuphydriver_timeout_gpu();
    uint32_t& get_cuphydriver_timeout_gpu_srs();
    uint32_t& get_cuphydriver_timeout_log_interval();
    uint32_t& get_cuphydriver_ul_srs_aggr3_task_launch_offset_ns();
    uint8_t& get_cuphydriver_ul_order_kernel_mode();
    uint8_t& get_cuphydriver_timeout_gpu_log_enable();
    uint8_t& get_cuphydriver_ue_mode();
    uint32_t& get_cuphydriver_order_kernel_max_rx_pkts();
    uint32_t& get_cuphydriver_order_kernel_rx_pkts_timeout();
    uint8_t& get_cplane_disable();
    uint8_t&  get_cuphydriver_use_green_contexts();
    uint8_t&  get_cuphydriver_use_gc_workqueues();
    uint8_t&  get_cuphydriver_use_batched_memcpy();
    uint32_t& get_cuphydriver_mps_sm_pusch();
    uint32_t& get_cuphydriver_mps_sm_pucch();
    uint32_t& get_cuphydriver_mps_sm_prach();
    uint32_t& get_cuphydriver_mps_sm_ul_order();
    uint32_t& get_cuphydriver_mps_sm_srs();
    uint32_t& get_cuphydriver_mps_sm_pdsch();
    uint32_t& get_cuphydriver_mps_sm_pdcch();
    uint32_t& get_cuphydriver_mps_sm_pbch();
    uint32_t& get_cuphydriver_mps_sm_gpu_comms();
    uint8_t& get_cuphydriver_pdsch_fallback();
    size_t get_cuphydriver_standalone_slot_command_size();
    uint8_t& get_cuphydriver_gpu_init_comms_dl();
    uint8_t& get_cuphydriver_gpu_init_comms_via_cpu();
    uint8_t& get_cuphydriver_cpu_init_comms();
    uint8_t& get_cuphydriver_cell_group();
    uint8_t& get_cuphydriver_cell_group_num();
    struct slot_command_api::slot_command * get_cuphydriver_standalone_slot_command(int slot_num);
    uint8_t get_cuphydriver_pusch_workCancelMode() const;
    uint8_t get_cuphydriver_pusch_tdi() const;
    uint8_t get_cuphydriver_pusch_cfo() const;
    uint8_t get_cuphydriver_pusch_dftsofdm() const;
    uint8_t get_cuphydriver_pusch_tbsizecheck() const;
    uint8_t get_cuphydriver_pusch_deviceGraphLaunchEn()const;  
    uint8_t get_cuphydriver_pusch_earlyHarqEn()const;  
    uint16_t get_cuphydriver_pusch_waitTimeOutPreEarlyHarqUs()const;   
    uint16_t get_cuphydriver_pusch_waitTimeOutPostEarlyHarqUs()const;     
    uint8_t get_cuphydriver_pusch_to() const;
    uint8_t get_cuphydriver_pusch_select_eqcoeffalgo() const;
    uint8_t get_cuphydriver_pusch_select_chestalgo() const;
    uint8_t get_cuphydriver_pusch_enable_perprgchest() const;
    uint8_t get_cuphydriver_pusch_rssi() const;
    uint8_t get_cuphydriver_pusch_sinr() const;
    uint8_t get_cuphydriver_pusch_weighted_average_cfo() const;
    [[nodiscard]] const std::string& get_cuphydriver_puschrx_chest_factory_settings_filename() const noexcept;
    uint8_t get_cuphydriver_puxchPolarDcdrListSz() const;
    uint8_t get_cuphydriver_fix_beta_dl() const;
    uint8_t get_cuphydriver_enable_l1_param_sanity_check()const;    
    uint8_t get_cuphydriver_enable_cpu_task_tracing() const;
    uint8_t get_cuphydriver_enable_prepare_tracing() const;
    uint8_t get_cuphydriver_cupti_enable_tracing() const;
    uint64_t get_cuphydriver_cupti_buffer_size() const;
    uint16_t get_cuphydriver_cupti_num_buffers() const;
    uint8_t get_cuphydriver_disable_empw()const;
    uint8_t get_cuphydriver_enable_dl_cqe_tracing() const;
    uint64_t get_cuphydriver_cqe_trace_cell_mask()const;
    uint32_t get_cuphydriver_cqe_trace_slot_mask()const;
    uint8_t get_cuphydriver_enable_ok_tb() const;
    uint32_t get_cuphydriver_num_ok_tb_slot()const;    
    uint8_t get_cuphydriver_ul_rx_pkt_tracing_level() const;
    uint8_t get_cuphydriver_ul_rx_pkt_tracing_level_srs() const;
    uint32_t get_cuphydriver_ul_warmup_frame_count() const;
    uint8_t get_cuphydriver_pmu_metrics() const;
    struct h2d_copy_thread_config get_cuphydriver_h2d_cpy_th_cfg() const;
    uint8_t& get_cuphydriver_mMIMO_enable();
    uint32_t& get_cuphydriver_aggr_obj_non_avail_th();
    uint32_t& get_cuphydriver_sendCPlane_timing_error_th_ns();
    uint32_t& get_cuphydriver_sendCPlane_ulbfw_backoff_th_ns();
    uint32_t& get_cuphydriver_sendCPlane_dlbfw_backoff_th_ns();
    uint8_t get_cuphydriver_split_ul_cuda_streams();
    uint8_t get_cuphydriver_serialize_pucch_pusch();
    std::vector<uint32_t>& get_cuphydriver_dl_wait_th();
    uint16_t get_cuphydriver_forcedNumCsi2Bits();
    uint32_t get_cuphydriver_pusch_nMaxLdpcHetConfigs();
    uint8_t get_cuphydriver_pusch_nMaxTbPerNode();
    uint8_t& get_cuphydriver_enable_srs();
    uint8_t& get_cuphydriver_enable_dl_core_affinity();
    uint8_t& get_cuphydriver_dlc_core_packing_scheme();
    uint8_t& get_cuphydriver_ch_segment_proc_enable();
    uint8_t& get_pusch_aggr_per_ctx();
    uint8_t& get_prach_aggr_per_ctx();
    uint8_t& get_pucch_aggr_per_ctx();
    uint8_t& get_srs_aggr_per_ctx();
    uint16_t& get_max_harq_pools();
    uint16_t& get_max_harq_tx_count_bundled();
    uint16_t& get_max_harq_tx_count_non_bundled();
    uint8_t& get_ul_input_buffer_per_cell();
    uint8_t& get_ul_input_buffer_per_cell_srs();
    uint32_t& get_max_ru_unhealthy_ul_slots();
    uint8_t& get_ul_pcap_capture_enable();
    uint8_t& get_ul_pcap_capture_thread_cpu_affinity();
    uint8_t& get_ul_pcap_capture_thread_sched_priority();
    uint8_t& get_pcap_logger_ul_cplane_enable(); 
    uint8_t& get_pcap_logger_dl_cplane_enable(); 
    uint8_t& get_pcap_logger_thread_cpu_affinity(); 
    uint8_t& get_pcap_logger_thread_sched_prio();
    std::string& get_pcap_logger_file_save_dir(); 
    uint16_t get_static_beam_id_start();
    uint16_t get_static_beam_id_end();
    uint16_t get_dynamic_beam_id_start();
    uint16_t get_dynamic_beam_id_end();
    uint8_t get_bfw_c_plane_chaining_mode();
    uint8_t& get_srs_chest_algo_type();
    uint8_t& get_srs_chest_tol2_normalization_algo_type();
    float&   get_srs_chest_tol2_constant_scaler();
    uint8_t& get_bfw_power_normalization_alg_selector();
    float&   get_bfw_beta_prescaler();
    uint32_t& get_total_num_srs_chest_buffers();
    uint8_t& get_send_static_bfw_wt_all_cplane();
    uint8_t get_dlc_bfw_enable_divide_per_cell();
    uint8_t get_ulc_bfw_enable_divide_per_cell();
    [[nodiscard]] uint8_t get_dlc_alloc_cplane_bfw_txq();
    [[nodiscard]] uint8_t get_ulc_alloc_cplane_bfw_txq();
    [[nodiscard]] uint8_t get_enable_tx_notification();
    [[nodiscard]] uint8_t get_notify_ul_harq_buffer_release() const;
private:
    int parse_cuphydriver_configs(yaml::node root);
    int parse_cell_configs(yaml::node root);
    int parse_single_cell(yaml::node cell,std::string *p_unique_nic_info,size_t length);
    int parse_eAxC_to_beam_map(yaml::node node, const std::vector<slot_command_api::channel_type>& channels, cell_mplane_info& mplane_cfg);

    std::string l2adapter_config_filename;
    std::string cuphycontroller_config_filename;
    std::string standalone_config_filename;
    std::vector<struct cell_phy_info> cell_configs;
    std::vector<::cell_mplane_info> mplane_configs;
    struct cuphydriver_config phydriver_config;
    std::vector<std::unique_ptr<struct slot_command_api::slot_command>> slot_command_list;
    std::unordered_set<std::string> dst_mac_set;
    std::unordered_set<uint16_t> cell_id_set;
};



#endif
