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

/** @file phydriver_api.hpp
 *  cuPHYDriver library header file
 *
 *  Header file for the cuPHYDriver L1 API
 */

#ifndef PHYDRIVER_API_H
#define PHYDRIVER_API_H

#include <cstdint>
#include <vector>
#include <array>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <iomanip>
#include <slot_command/slot_command.hpp>
#include "aerial-fh-driver/api.hpp"
#include "constant.hpp"
#include <QAM_param.cuh>

/**
 * @defgroup Handlers Exposed Opaque handlers
 *
 * This section lists Opaque Handlers and predefined types puclicly availabla to interact
 * with PhyDriver
 *
 * @{
 */

/******************************************************************/ /**
 * @brief Pointer to a cuPHYDriver context
 *
 */
typedef void* phydriver_handle;

/******************************************************************/ /**
 * @brief Pointer to a cuPHYDriver worker handler
 *
 */
typedef void* phydriverwrk_handle;

/******************************************************************/ /**
 * @brief Worker ID
 *
 */
typedef uint64_t worker_id;

/******************************************************************/ /**
 * @brief Format of a worker routine
 *
 */
typedef int (*worker_routine)(phydriverwrk_handle, void*);

/** @} */ /* END OPAQUE HANDLERS */

/**
 * @defgroup Context cuPHYDriver context management
 *
 * This section describes the error handling functions of the cuPHY 
 * application programming interface.
 *
 * @{
 */

/******************************************************************/ /**
 * @brief Hold cell cuPHY info
 *
 */
struct cell_phy_info
{
    std::string name;                                           ///< Cell name identifier

    cuphyCellStatPrm_t phy_stat;                                ///< cuPHY static cell parameters (bandwidth, numerology, etc.)

    int tti;                                                    ///< Transmission Time Interval in nanoseconds
    int slot_ahead;                                             ///< Number of slots to process ahead of air interface transmission

    /*
     * This is required to connect cell PHY info to cell M-plane info
     */
    uint16_t mplane_id;                                         ///< M-plane identifier to connect PHY and management plane configurations
    ::cuphyPrachCellStatPrms_t prachStatParams;                 ///< PRACH cell-level static parameters
    std::vector<::cuphyPrachOccaStatPrms_t> prach_configs;      ///< PRACH occasion configurations (multiple occasions per cell)
    bool is_early_harq_detection_enabled;                       ///< Enable early HARQ-ACK detection before full decoding completes
    cuphySrsChEstAlgoType_t srs_chest_algo_type;                ///< SRS channel estimation algorithm type selector
    uint8_t srs_chest_tol2_normalization_algo_type;             ///< SRS to L2 normalization algorithm type (0=disabled, 1=constant scaler, 2=auto)
    float srs_chest_tol2_constant_scaler;                       ///< Constant scaling factor for SRS to L2 normalization (when type=1)
    uint8_t bfw_power_normalization_alg_selector;               ///< Beamforming weights power normalization algorithm selector
    uint8_t pusch_aggr_factor;                                  ///< Number of TTI slots aggregated for PUSCH bundling
};

extern phydriver_handle l1_pdh;                                 ///< Global cuPHYDriver handle for signal handlers and exit routines
extern pthread_t gBg_thread_id;                                 ///< Background thread ID for formatted logging (nvlog_fmtlog)

/******************************************************************/ /**
 * @brief Per-cell NIC resource configuration
 *
 */
struct nic_resource_config
{
    uint8_t txq_count_uplane;                                   ///< Number of transmit queues for U-plane traffic
};

/**
 * eAxC (extended Antenna-Carrier) ID list per channel type
 * Array indexed by channel type (PUSCH, PRACH, SRS, etc.) containing vectors of eAxC IDs
 */
using eAxC_list = std::array<std::vector<uint16_t>, slot_command_api::channel_type::CHANNEL_MAX>;

/******************************************************************/ /**
 * @brief Per-destination MAC + eAxC mapping for multi-MAC (dMIMO) routing
 */
struct cell_mplane_destination
{
    std::array<uint8_t, 6> dst_eth_addr;
    eAxC_list              eAxC_ids;
};

/******************************************************************/ /**
 * @brief Simulate M-plane info per cell when creating the L1 context
 *
 */
struct cell_mplane_info
{
    uint16_t mplane_id;                                         ///< M-plane identifier for this cell
    enum ru_type ru;                                            ///< Radio Unit type (SINGLE_SECT_MODE, MULTI_SECT_MODE, OTHER_MODE)

    std::array<uint8_t, 6> src_eth_addr;                        ///< Source MAC address for ORAN fronthaul
    std::array<uint8_t, 6> dst_eth_addr;                        ///< Destination MAC address for ORAN fronthaul
    std::vector<cell_mplane_destination> destinations;          ///< dMIMO: per-RU-MAC eAxC routing (empty => single dst_eth_addr)
    std::string            nic_name;                            ///< Network interface card name
    uint32_t               nic_index;                           ///< Network interface card index
    uint16_t               vlan_tci;                            ///< VLAN Tag Control Information (TCI)
    nic_resource_config    nic_cfg;                             ///< NIC resource configuration (queue counts)
    eAxC_list              eAxC_ids;                            ///< Extended Antenna-Carrier IDs per channel type

    uint64_t                t1a_max_up_ns;                      ///< T1a max: Maximum advance time for DL U-plane to arrive before transmission (ns)
    uint64_t                t1a_max_cp_ul_ns;                   ///< T1a max: Maximum advance time for UL C-plane to arrive (ns)
    uint64_t                t1a_min_cp_ul_ns;                   ///< T1a min: Minimum advance time for UL C-plane to arrive (ns)
    uint64_t                ta4_min_ns;                         ///< Ta4 min: Minimum delay from UL slot boundary to RU U-plane transmission (ns)
    uint64_t                ta4_max_ns;                         ///< Ta4 max: Maximum delay from UL slot boundary to RU U-plane transmission (ns)
    uint64_t                ta4_min_ns_srs;                     ///< Ta4 min for SRS: Minimum delay for SRS transmission (ns)
    uint64_t                ta4_max_ns_srs;                     ///< Ta4 max for SRS: Maximum delay for SRS transmission (ns)
    uint64_t                tcp_adv_dl_ns;                      ///< Tcp advance: Time to send DL C-plane before DL U-plane (ns)
    uint64_t                t1a_min_cp_dl_ns;                   ///< T1a min: Minimum advance time for DL C-plane to arrive (ns)
    uint64_t                t1a_max_cp_dl_ns;                   ///< T1a max: Maximum advance time for DL C-plane to arrive (ns)
    uint64_t                ul_u_plane_tx_offset_ns;            ///< UL U-plane transmission offset from slot boundary (ns, UE mode)
    uint32_t                pusch_prb_stride;                   ///< PUSCH PRB stride for memory allocation
    uint32_t                prach_prb_stride;                   ///< PRACH PRB stride for memory allocation
    uint32_t                srs_prb_stride;                     ///< SRS PRB stride for memory allocation
    uint16_t                pusch_nMaxPrb;                      ///< Maximum number of PRBs for PUSCH
    uint16_t                pusch_nMaxRx;                       ///< Maximum number of receive antennas for PUSCH
    uint16_t                section_3_time_offset;              ///< Section type 3 time offset for ORAN C-plane
    uint8_t                 pusch_ldpc_max_num_itr_algo_type;   ///< LDPC max iteration algorithm type (0=static, 1=dynamic based on SNR)
    uint8_t                 dlc_core_index;                     ///< DL C-plane core index for fixed packing scheme (scheme=1)
    uint8_t                 pusch_fixed_max_num_ldpc_itrs;      ///< Fixed maximum number of LDPC iterations (when algo_type=0)
    uint8_t                 pusch_ldpc_n_iterations;            ///< Target number of LDPC iterations
    uint8_t                 pusch_ldpc_early_termination;       ///< Enable LDPC early termination (0=disabled, 1=enabled)
    uint8_t                 pusch_ldpc_algo_index;              ///< LDPC algorithm variant (0=auto, GPU-specific optimizations)
    uint8_t                 pusch_ldpc_flags;                   ///< LDPC decoder flags (reserved for future use)
    uint8_t                 pusch_ldpc_use_half;                ///< Use FP16 for LDPC (0=FP32, 1=FP16)
    uint8_t                 fh_len_range;                       ///< Fronthaul distance range category for timing adjustments
    float                   ul_gain_calibration;                ///< UL gain calibration factor
    uint32_t                lower_guard_bw;                     ///< Lower guard bandwidth in Hz

    uint16_t                nMaxRxAnt;                          ///< Maximum number of receive antennas for the cell

    std::string tv_pusch_h5;                                    ///< Test vector file path for PUSCH (HDF5 format)
    std::string tv_srs_h5;                                      ///< Test vector file path for SRS (HDF5 format)

    /*
     * Currently we support static compression only
     */

    enum aerial_fh::UserDataCompressionMethod dl_comp_meth;     ///< DL user data compression method (BFP, block scaling, etc.)
    enum aerial_fh::UserDataCompressionMethod ul_comp_meth;     ///< UL user data compression method (BFP, block scaling, etc.)
    uint8_t dl_bit_width;                                       ///< DL compressed IQ sample bit width
    uint8_t ul_bit_width;                                       ///< UL compressed IQ sample bit width

    // Power scaling
    int fs_offset_dl;                                           ///< DL full-scale offset for power scaling
    int exponent_dl;                                            ///< DL block floating point exponent
    int ref_dl;                                                 ///< DL reference power level

    int fs_offset_ul;                                           ///< UL full-scale offset for power scaling
    int exponent_ul;                                            ///< UL block floating point exponent
    int max_amp_ul;                                             ///< UL maximum amplitude for scaling
};

static constexpr char CELL_PARAM_RU_TYPE[] = "ru_type";                                        ///< Cell parameter key: Radio Unit type
static constexpr char CELL_PARAM_DST_MAC_ADDR[] = "dst_mac_addr";                              ///< Cell parameter key: Destination MAC address
static constexpr char CELL_PARAM_VLAN_ID[] = "vlan_id";                                        ///< Cell parameter key: VLAN identifier
static constexpr char CELL_PARAM_PCP[] = "pcp";                                                ///< Cell parameter key: Priority Code Point (VLAN priority)
static constexpr char CELL_PARAM_COMPRESSION_BITS[] = "compression_bits";                      ///< Cell parameter key: Compression bit width
static constexpr char CELL_PARAM_DECOMPRESSION_BITS[] = "decompression_bits";                  ///< Cell parameter key: Decompression bit width

static constexpr char CELL_PARAM_DL_COMP_METH[] = "dl_comp_meth";                              ///< Cell parameter key: Downlink compression method
static constexpr char CELL_PARAM_UL_COMP_METH[] = "ul_comp_meth";                              ///< Cell parameter key: Uplink compression method
static constexpr char CELL_PARAM_DL_BIT_WIDTH[] = "dl_bit_width";                              ///< Cell parameter key: Downlink bit width
static constexpr char CELL_PARAM_UL_BIT_WIDTH[] = "ul_bit_width";                              ///< Cell parameter key: Uplink bit width

static constexpr char CELL_PARAM_EXPONENT_DL[] = "exponent_dl";                                ///< Cell parameter key: Downlink BFP exponent
static constexpr char CELL_PARAM_EXPONENT_UL[] = "exponent_ul";                                ///< Cell parameter key: Uplink BFP exponent
static constexpr char CELL_PARAM_MAX_AMP_UL[] = "max_amp_ul";                                  ///< Cell parameter key: Uplink maximum amplitude
static constexpr char CELL_PARAM_PUSCH_PRB_STRIDE[] = "pusch_prb_stride";                      ///< Cell parameter key: PUSCH PRB stride
static constexpr char CELL_PARAM_PRACH_PRB_STRIDE[] = "prach_prb_stride";                      ///< Cell parameter key: PRACH PRB stride
static constexpr char CELL_PARAM_SECTION_3_TIME_OFFSET[] = "section_3_time_offset";            ///< Cell parameter key: ORAN section 3 time offset
static constexpr char CELL_PARAM_FH_DISTANCE_RANGE[] = "fh_distance_range";                    ///< Cell parameter key: Fronthaul distance range
static constexpr char CELL_PARAM_UL_GAIN_CALIBRATION[] = "ul_gain_calibration";                ///< Cell parameter key: Uplink gain calibration factor
static constexpr char CELL_PARAM_LOWER_GUARD_BW[] = "lower_guard_bw";                          ///< Cell parameter key: Lower guard bandwidth
static constexpr char CELL_PARAM_REF_DL[] = "ref_dl";                                          ///< Cell parameter key: Downlink reference power
static constexpr char CELL_PARAM_NIC[] = "nic";                                                ///< Cell parameter key: Network interface card name


struct nic_cfg
{
    std::string nic_bus_addr;                                   ///< NIC PCIe bus address (e.g., "0000:51:00.0")
    uint16_t    nic_mtu;                                        ///< Maximum Transmission Unit size in bytes
    uint32_t    cpu_mbuf_num;                                   ///< Number of CPU memory buffers for packet processing
    uint32_t    tx_req_num;                                     ///< Number of transmit request buffers
    uint16_t    txq_count_uplane;                               ///< Number of transmit queues for U-plane traffic
    uint16_t    txq_count_cplane;                               ///< Number of transmit queues for C-plane traffic
    uint16_t    rxq_count;                                      ///< Number of receive queues
    uint16_t    txq_size;                                       ///< Transmit queue size (number of descriptors)
    uint16_t    rxq_size;                                       ///< Receive queue size (number of descriptors)
};

struct h2d_copy_prepone_info{
    phydriver_handle pdh;                                       ///< cuPHYDriver handle
    uint16_t phy_cell_id;                                       ///< Physical cell ID
    uint8_t * tb_buff;                                          ///< Transport block buffer on host (CPU memory)
    uint8_t ** gpu_buff_ref;                                    ///< Reference to GPU buffer pointer
    uint32_t tb_len;                                            ///< Transport block length in bytes
    uint8_t slot_index;                                         ///< Slot index within circular buffer
    uint16_t sfn;                                               ///< System Frame Number, used to detect batch boundaries
};

struct h2d_copy_thread_config{
    uint8_t enable_h2d_copy_thread;                             ///< Enable dedicated thread for host-to-device copies (0=disabled, 1=enabled)
    uint16_t h2d_copy_thread_cpu_affinity;                      ///< CPU core affinity for H2D copy thread
    uint8_t h2d_copy_thread_sched_priority;                     ///< Scheduling priority for H2D copy thread
};

/**
 * Type alias for host-to-device copy prepone information structure
 */
typedef struct h2d_copy_prepone_info h2d_copy_prepone_info_t;

static constexpr uint32_t MAX_NUM_CELLS_PER_DEVICE = DL_MAX_CELLS_PER_SLOT;  ///< Maximum number of cells per GPU device

struct mod_compression_params{
    float2  scaling[API_MAX_ANTENNAS][ORAN_ALL_SYMBOLS][MAX_SECTIONS_PER_UPLANE_SYMBOL];         ///< Scaling factors per antenna/symbol/section
    uint16_t nprbs_per_list[API_MAX_ANTENNAS][ORAN_ALL_SYMBOLS][MAX_SECTIONS_PER_UPLANE_SYMBOL]; ///< Number of PRBs per section (range 0-273)
    uint16_t prb_start_per_list[API_MAX_ANTENNAS][ORAN_ALL_SYMBOLS][MAX_SECTIONS_PER_UPLANE_SYMBOL]; ///< Starting PRB index per section (range 0-273)
    uint8_t num_messages_per_list[API_MAX_ANTENNAS][ORAN_ALL_SYMBOLS];                           ///< Number of sections per antenna/symbol (range 0-MAX_SECTIONS_PER_UPLANE_SYMBOL)
    QamListParam params_per_list[API_MAX_ANTENNAS][ORAN_ALL_SYMBOLS][MAX_SECTIONS_PER_UPLANE_SYMBOL];  ///< QAM modulation parameters per section (IQ width, CSF, etc.)
    QamPrbParam prb_params_per_list[API_MAX_ANTENNAS][ORAN_ALL_SYMBOLS][MAX_SECTIONS_PER_UPLANE_SYMBOL]; ///< RE mask of PRBs per section
};

/// Number of entries in the aerial_fh::UserDataCompressionMethod enum
static constexpr std::size_t NUM_USER_DATA_COMPRESSION_METHODS = 7;

struct compression_params{
    uint8_t comp_meth[MAX_NUM_CELLS_PER_DEVICE];               ///< Compression method per cell
    uint8_t bit_width[MAX_NUM_CELLS_PER_DEVICE];               ///< Compressed bit width per cell
    uint8_t  *input_ptrs[MAX_NUM_CELLS_PER_DEVICE];            ///< Input buffer pointers per cell
    uint8_t **prb_ptrs[MAX_NUM_CELLS_PER_DEVICE];              ///< Per-PRB buffer pointers per cell
    int     num_prbs[MAX_NUM_CELLS_PER_DEVICE];                ///< Number of PRBs per cell
    float   beta[MAX_NUM_CELLS_PER_DEVICE];                    ///< Beta scaling factor per cell
    uint16_t max_num_prb_per_symbol[MAX_NUM_CELLS_PER_DEVICE]; ///< Maximum PRBs per symbol per cell
    uint8_t num_antennas[MAX_NUM_CELLS_PER_DEVICE];            ///< Number of antennas per cell
    uint8_t num_cells;                                          ///< Total number of active cells
    bool    gpu_comms;                                          ///< GPU direct communication enabled (true=enabled, false=via CPU)

    //The following is needed if mod compression is enabled    
    mod_compression_params *mod_compression_config[MAX_NUM_CELLS_PER_DEVICE];  ///< Modulation compression config per cell (null if disabled)
};

/******************************************************************/ /**
 * @brief Configure a new instance of cuPHYDriver
 *
 */
struct context_config
{
    int gpu_id;                                                 ///< GPU device ID to use for cuPHY processing
    std::vector<nic_cfg> nic_configs;                           ///< Network interface card configurations

    bool standalone;                                            ///< Standalone mode (no external L2 integration)
    bool validation;                                            ///< Enable validation mode for testing
    bool cplane_disable;                                        ///< Disable C-plane processing (U-plane only mode)

    std::vector<struct cell_mplane_info> cell_mplane_list;      ///< List of cell management plane configurations

    uint8_t fh_cpu_core;                                        ///< CPU core for fronthaul thread
    int fh_stats_dump_cpu_core;                                 ///< CPU core for fronthaul statistics dumping thread (-1=disabled)
    bool dpdk_verbose_logs;                                     ///< Enable verbose DPDK logging
    int prometheus_cpu_core;                                    ///< CPU core for Prometheus metrics thread (-1=disabled)

    uint32_t accu_tx_sched_res_ns;                              ///< Accurate TX scheduling resolution in nanoseconds
    bool accu_tx_sched_disable;                                 ///< Disable accurate TX scheduling
    int pdump_client_thread;                                    ///< CPU core for packet dump client thread (-1=disabled)
    std::string dpdk_file_prefix;                               ///< DPDK file prefix for shared memory objects

    uint32_t workers_sched_priority;                            ///< Scheduling priority for worker threads

    // Section Id per channel. ORAN SectionId 12bits
    uint16_t start_section_id_prach;                            ///< Starting ORAN section ID for PRACH (12-bit)
    uint16_t start_section_id_srs;                              ///< Starting ORAN section ID for SRS (12-bit)

    uint8_t enable_ul_cuphy_graphs;                             ///< Enable CUDA graphs for UL processing (0=disabled, 1=enabled)
    uint8_t enable_dl_cuphy_graphs;                             ///< Enable CUDA graphs for DL processing (0=disabled, 1=enabled)

    uint32_t ul_order_timeout_gpu_ns;                           ///< UL packet ordering GPU timeout in nanoseconds
    uint32_t ul_srs_aggr3_task_launch_offset_ns;                ///< SRS aggregation task 3 launch offset from T0 (ns)
    uint32_t ul_order_timeout_gpu_srs_ns;                       ///< UL packet ordering GPU timeout for SRS in nanoseconds
    uint32_t ul_order_timeout_cpu_ns;                           ///< UL packet ordering CPU timeout in nanoseconds
    uint32_t ul_order_timeout_log_interval_ns;                  ///< Logging interval for UL ordering timeouts (ns)
    uint8_t ul_order_timeout_gpu_log_enable;                    ///< Enable logging for UL ordering GPU timeouts (0=disabled, 1=enabled)
    uint8_t ul_order_kernel_mode;                               ///< UL order kernel mode selector (0: Ping-Pong[default], 1: dual CTA)
    uint32_t ul_order_max_rx_pkts;                              ///< Maximum number of packets to receive for a single call of DOCA rx call in the order kernel.
    uint32_t ul_order_rx_pkts_timeout_ns;                       ///< Timeout for receiving UL packets for a single call of DOCA rx call in the order kernel (ns).

    std::vector<uint8_t> ul_cores;                              ///< CPU cores for UL worker threads
    std::vector<uint8_t> dl_cores;                              ///< CPU cores for DL worker threads
    int16_t debug_worker;                                       ///< CPU core for debug worker thread (-1=disabled)
    int16_t data_core;                                          ///< CPU core for datalake thread (-1=disabled)
    uint8_t datalake_db_write_enable;                           ///< Enable database write operations for datalake (0=disabled, 1=enabled)
    uint32_t datalake_samples;                                  ///< Number of samples to capture for datalake
    std::string datalake_address;                               ///< Datalake server address
    std::string datalake_engine;                                ///< Datalake engine type (e.g., "ClickHouse")
    std::vector<std::string> datalake_data_types;               ///< List of data types to be stored in datalake
    uint8_t datalake_store_failed_pdu;                          ///< Store failed PDU data in datalake (0=disabled, 1=enabled)
    uint32_t num_rows_fh;                                       ///< Number of rows per batch for fronthaul datalake ingestion
    uint32_t num_rows_pusch;                                    ///< Number of rows per batch for PUSCH datalake ingestion
    uint32_t num_rows_hest;                                     ///< Number of rows per batch for channel estimation datalake ingestion
    uint8_t e3_agent_enabled;                                   ///< Enable E3 agent for RIC integration (0=disabled, 1=enabled)
    uint16_t e3_rep_port;                                       ///< E3 agent reply port number
    uint16_t e3_pub_port;                                       ///< E3 agent publish port number
    uint16_t e3_sub_port;                                       ///< E3 agent subscribe port number
    uint8_t datalake_drop_tables;                               ///< Drop existing datalake tables on startup (0=keep, 1=drop)
                       
    uint8_t  use_green_contexts;                                ///< Use CUDA green contexts (0=disabled, 1=enabled)
    uint8_t  use_gc_workqueues;                                 ///< Use green contexts' workqueue feature (0=disabled, 1=enabled), if green contexts enabled  
    uint8_t  use_batched_memcpy;                                ///< Use batched memory copy operations for efficiency (0=disabled, 1=enabled)
    uint32_t mps_sm_pusch;                                      ///< MPS (Multi-Process Service) SM percentage for PUSCH (0-100)
    uint32_t mps_sm_pucch;                                      ///< Maximum number of SMs allocated for PUCCH processing
    uint32_t mps_sm_prach;                                      ///< Maximumu number of SMs allocated for PRACH processing
    uint32_t mps_sm_ul_order;                                   ///< Maximum number of SMs allocated for UL packet ordering
    uint32_t mps_sm_srs;                                        ///< Maximum number of SMs allocated for SRS processing
    uint32_t mps_sm_pdsch;                                      ///< Maximum number of SMs allocated for PDSCH processing
    uint32_t mps_sm_pdcch;                                      ///< Maximum number of SMs allocated for PDCCH processing
    uint32_t mps_sm_pbch;                                       ///< Maximum number of SMs allocated for PBCH/PSS/SSS processing
    uint32_t mps_sm_gpu_comms;                                  ///< Maximum number of SMs allocated for GPU direct communications in DL

    uint8_t pdsch_fallback;                                     ///< Enable PDSCH fallback mode (0=disabled, 1=enabled). Used for testing/debugging only.

    uint8_t cell_group;                                         ///< Cell group ID for this context
    uint8_t cell_group_num;                                     ///< Total number of cell groups

    uint8_t pusch_workCancelMode;                               ///< PUSCH work cancellation mode (0=no cancel, 1=cancel on late arrival)
    uint8_t enable_pusch_tdi;                                   ///< Enable PUSCH time domain interpolation (0=disabled, 1=enabled)
    uint8_t enable_pusch_cfo;                                   ///< Enable PUSCH carrier frequency offset estimation (0=disabled, 1=enabled)
    uint8_t select_pusch_eqcoeffalgo;                           ///< PUSCH equalization coefficient algorithm selector
    uint8_t select_pusch_chestalgo;                             ///< PUSCH channel estimation algorithm selector
    uint8_t enable_pusch_perprgchest;                           ///< Enable per-PRB-group channel estimation for PUSCH (0=disabled, 1=enabled)
    uint8_t enable_pusch_to;                                    ///< Enable PUSCH timing offset estimation (0=disabled, 1=enabled)
    uint8_t enable_pusch_rssi;                                  ///< Enable PUSCH RSSI (Received Signal Strength Indicator) calculation (0=disabled, 1=enabled)
    uint8_t enable_pusch_sinr;                                  ///< Enable PUSCH SINR (Signal-to-Interference-plus-Noise Ratio) calculation (0=disabled, 1=enabled)
    uint8_t enable_weighted_average_cfo;                        ///< Enable weighted average for CFO estimation (0=disabled, 1=enabled)
    uint8_t enable_pusch_dftsofdm;                              ///< Enable DFT-s-OFDM processing for PUSCH (0=OFDM only, 1=DFT-s-OFDM enabled)
    uint8_t enable_pusch_tbsizecheck;                           ///< Enable PUSCH transport block size validation (0=disabled, 1=enabled)
    uint8_t pusch_deviceGraphLaunchEn;                          ///< Enable CUDA device graph launch for PUSCH (0=disabled, 1=enabled)
    uint16_t pusch_waitTimeOutPreEarlyHarqUs;                   ///< PUSCH wait timeout before early HARQ detection (microseconds)
    uint16_t pusch_waitTimeOutPostEarlyHarqUs;                  ///< PUSCH wait timeout after early HARQ detection (microseconds)
    uint8_t enable_gpu_comm_dl;                                 ///< Enable GPU direct communication for DL (0=disabled, 1=enabled)
    uint8_t enable_gpu_comm_via_cpu;                            ///< Enable GPU communication via CPU fallback (0=disabled, 1=enabled)
    uint8_t enable_cpu_init_comms;                              ///< Enable CPU initiated communication buffers (0=disabled, 1=enabled)

    uint8_t mPuxchPolarDcdrListSz;                              ///< PUCCH Polar decoder list size (number of decoders)
    std::string mPuschrxChestFactorySettingsFilename;           ///< PUSCH receiver channel estimation factory settings filename

    uint8_t fix_beta_dl;                                        ///< Use fixed beta value for DL scaling (0=dynamic, 1=fixed)

    uint8_t enable_cpu_task_tracing;                            ///< Enable CPU task tracing (0=disabled, 1=enabled)
    uint8_t enable_prepare_tracing;                             ///< Enable preparation phase tracing (0=disabled, 1=enabled)
    uint8_t cupti_enable_tracing;                               ///< Enable CUPTI tracing (0=disabled, 1=enabled)
    uint64_t cupti_buffer_size;                                 ///< CUPTI buffer size in bytes (default: 2GB)
    uint16_t cupti_num_buffers;                                 ///< Number of CUPTI buffers (default: 2)
    uint8_t disable_empw;                                       ///< Disable Enhanced Multi-packet write WQE feature (0=enabled, 1=disabled)
    uint8_t enable_dl_cqe_tracing;                              ///< Enable DL completion queue entry tracing (0=disabled, 1=enabled)
    uint64_t cqe_trace_cell_mask;                               ///< Cell mask for CQE tracing (bit per cell)
    uint32_t cqe_trace_slot_mask;                               ///< Slot mask for CQE tracing (bit per slot within frame)
    uint8_t enable_ok_tb;                                       ///< Enable order kernel testbench (0=disabled, 1=enabled)
    uint32_t num_ok_tb_slot;                                    ///< Number of slots to be used in order kernel testbench
    uint8_t ul_rx_pkt_tracing_level;                            ///< UL receive packet tracing level (0=off, higher=more verbose)
    uint8_t ul_rx_pkt_tracing_level_srs;                        ///< UL SRS receive packet tracing level (0=off, higher=more verbose)
    uint32_t ul_warmup_frame_count;                             ///< Number of UL warmup frames before full processing begins
    uint8_t pmu_metrics;                                        ///< Enable PMU (Performance Monitoring Unit) metrics collection (0=disabled, 1=enabled)
    uint8_t enable_l1_param_sanity_check;                       ///< Enable L1 parameter sanity checking (0=disabled, 1=enabled)

    struct h2d_copy_thread_config h2d_cpy_th_cfg;               ///< Host-to-device copy thread configuration

    uint8_t mMIMO_enable;                                       ///< Enable massive MIMO (mMIMO) processing (0=disabled, 1=enabled)
    uint8_t enable_srs;                                         ///< Enable SRS (Sounding Reference Signal) processing (0=disabled, 1=enabled)
    uint8_t enable_dl_core_affinity;                            ///< Enable DL core affinity for task distribution (0=disabled, 1=enabled)
    uint8_t dlc_core_packing_scheme;                            ///< DL C-plane core packing scheme (0=default, 1=fixed per-cell, 2=dynamic workload-based)
    uint8_t ue_mode;                                            ///< Enable User Equipment emulation mode (0=gNodeB, 1=UE mode)
    std::vector<uint8_t> dl_validation_cores;                   ///< CPU cores for DL validation worker threads
    uint32_t aggr_obj_non_avail_th;                             ///< Aggregation object non-availability threshold (ns)
    uint8_t split_ul_cuda_streams;                              ///< Split UL processing across multiple CUDA streams (0=single stream, 1=split)
    uint8_t serialize_pucch_pusch;                              ///< Serialize PUCCH and PUSCH processing (0=parallel, 1=serialized)
    std::vector<uint32_t> dl_wait_th_list;                      ///< List of DL wait thresholds per stage (ns)
    uint32_t sendCPlane_timing_error_th_ns;                     ///< C-plane send timing error threshold (ns)
    uint32_t sendCPlane_ulbfw_backoff_th_ns;                    ///< C-plane UL beamforming backoff threshold from deadline (ns)
    uint32_t sendCPlane_dlbfw_backoff_th_ns;                    ///< C-plane DL beamforming backoff threshold from deadline (ns)
    uint16_t forcedNumCsi2Bits;                                 ///< Force CSI part 2 to specific number of bits (0=use actual)
    uint32_t pusch_nMaxLdpcHetConfigs;                          ///< Maximum number of heterogeneous LDPC configurations for PUSCH
    uint8_t pusch_nMaxTbPerNode;                                ///< Maximum number of transport blocks per node for PUSCH
    uint8_t mCh_segment_proc_enable;                            ///< Enable tracking and validating of the processing timeline of PHY channels (0=disabled, 1=enabled)
    uint8_t pusch_aggr_per_ctx;                                 ///< Number of PUSCH aggregation objects per context
    uint8_t prach_aggr_per_ctx;                                 ///< Number of PRACH aggregation objects per context
    uint8_t pucch_aggr_per_ctx;                                 ///< Number of PUCCH aggregation objects per context
    uint8_t srs_aggr_per_ctx;                                   ///< Number of SRS aggregation objects per context
    uint16_t max_harq_pools;                                    ///< Maximum number of HARQ process buffer pools
    uint16_t max_harq_tx_count_bundled;                         ///< Maximum HARQ transmissions for bundled PDUs
    uint16_t max_harq_tx_count_non_bundled;                     ///< Maximum HARQ transmissions for non-bundled PDUs
    uint8_t ul_input_buffer_per_cell;                           ///< Number of UL input buffers per cell (circular buffer depth)
    uint8_t ul_input_buffer_per_cell_srs;                       ///< Number of UL input buffers per cell for SRS (circular buffer depth)
    uint32_t max_ru_unhealthy_ul_slots;                         ///< Maximum consecutive unhealthy UL slots before RU declared failed
    uint8_t ul_pcap_capture_enable;                             ///< Enable UL PCAP packet capture (0=disabled, 1=enabled)
    uint8_t ul_pcap_capture_thread_cpu_affinity;                ///< CPU core affinity for UL PCAP capture thread
    uint8_t ul_pcap_capture_thread_sched_priority;              ///< Scheduling priority for UL PCAP capture thread
    uint8_t pcap_logger_ul_cplane_enable;                       ///< Enable PCAP logging for UL C-plane (0=disabled, 1=enabled)
    uint8_t pcap_logger_dl_cplane_enable;                       ///< Enable PCAP logging for DL C-plane (0=disabled, 1=enabled)
    uint8_t pcap_logger_thread_cpu_affinity;                    ///< CPU core affinity for PCAP logger thread
    uint8_t pcap_logger_thread_sched_prio;                      ///< Scheduling priority for PCAP logger thread
    std::string pcap_logger_file_save_dir;                      ///< Directory path for saving PCAP files

    uint8_t srs_chest_algo_type;                                ///< SRS channel estimation algorithm type selector
    uint8_t srs_chest_tol2_normalization_algo_type;             ///< SRS to L2 normalization algorithm type (0=disabled, 1=constant scaler, 2=auto)
    float   srs_chest_tol2_constant_scaler;                     ///< Constant scaling factor for SRS to L2 normalization (when type=1)
    uint8_t bfw_power_normalization_alg_selector;               ///< Beamforming weights power normalization algorithm selector
    float   bfw_beta_prescaler;                                 ///< Beamforming beta prescaling factor
    uint32_t total_num_srs_chest_buffers;                       ///< Total number of SRS channel estimate buffers in global pool
    uint8_t send_static_bfw_wt_all_cplane;                      ///< Send static beamforming weights with all C-plane messages (0=once, 1=always)
    uint8_t dlc_bfw_enable_divide_per_cell;                     ///< Enable per-cell division for DL C-plane beamforming (0=shared, 1=per-cell)
    uint8_t ulc_bfw_enable_divide_per_cell;                     ///< Enable per-cell division for UL C-plane beamforming (0=shared, 1=per-cell)
    uint8_t dlc_alloc_cplane_bfw_txq;                           ///< Allocate dedicated C-plane TX queue for DL beamforming (0=shared, 1=dedicated)
    uint8_t ulc_alloc_cplane_bfw_txq;                           ///< Allocate dedicated C-plane TX queue for UL beamforming (0=shared, 1=dedicated)
    uint16_t static_beam_id_start;                              ///< Starting beam ID for static beamforming weights
    uint16_t static_beam_id_end;                                ///< Ending beam ID for static beamforming weights
    uint16_t dynamic_beam_id_start;                             ///< Starting beam ID for dynamic beamforming weights
    uint16_t dynamic_beam_id_end;                               ///< Ending beam ID for dynamic beamforming weights
    uint8_t bfw_c_plane_chaining_mode;                          ///< Beamforming C-plane message chaining mode
    bool enable_tx_notification;                                ///< Enable TX completion notifications (false=disabled, true=enabled)
    uint8_t notify_ul_harq_buffer_release;                      ///< Enable UL HARQ buffer release notifications (0=disabled, 1=enabled)
};

struct ReleasedHarqBuffer {
    uint32_t rnti{};                                            ///< Radio Network Temporary Identifier (UE ID)
    uint32_t harq_pid{};                                        ///< HARQ process ID (0-15)
    uint64_t cell_id{};                                         ///< Cell identifier
    uint16_t sfn{};                                             ///< System Frame Number when released (0-1023)
    uint16_t slot{};                                            ///< Slot number within frame when released
};

struct ReleasedHarqBufferInfo {
    uint32_t num_released_harq_buffers;                         ///< Number of HARQ buffers released in this notification
    std::vector<ReleasedHarqBuffer> released_harq_buffer_list;  ///< List of released HARQ buffer descriptors
    // Reset function to clear buffer list and counter
    void reset() {
        released_harq_buffer_list.clear();
        num_released_harq_buffers = 0;
    }
};
/******************************************************************/ /**
 * @brief Create a new cuPHYDriver instance
 *
 * Each instance of cuPHYDriver is completely independent.
 * An instance of cuPHYDriver is required to access all the APIs.
 *
 * @param[out] pdh Pointer to memory area useful to store cuPHYDriver handler will be stored
 * @param[in] ctx_cfg Set of parameters to configure a new istance of cuPHYDriver
 *
 * @return
 * \p 0 on success, Linux error code otherwise
 *
 * @sa ::l1_finalize
 */
int l1_init(phydriver_handle* pdh, const context_config& ctx_cfg);

/******************************************************************/ /**
 * @brief Destroy an instance of cuPHYDriver.
 *
 * @param[in] pdh Pointer to memory area with the cuPHYDriver context that must be destroyed
 *
 * @return
 * \p 0 on success, Linux error code otherwise
 *
 * @sa ::l1_init
 */
int l1_finalize(phydriver_handle pdh);

/** @} */ /* END CONTEXT */

/**
 * @defgroup Workers cuPHYDriver workers management
 *
 * API dedicated to cuPHYDriver Workers (pthreads)
 *
 * @{
 */

/******************************************************************/ /**
 * @brief Type of default workers
 *
 */
enum worker_default_type
{
    WORKER_UL = 0,                                              ///< Uplink worker thread type
    WORKER_DL,                                                  ///< Downlink worker thread type
    WORKER_DL_VALIDATION,                                       ///< Downlink validation worker thread type
    WORKER_GENERIC                                              ///< Generic worker thread type for user-defined routines
};

/******************************************************************/ /**
 * @brief Creates a new worker to execute an user-defined routine
 *
 * cuPHYDriver spawns a new worker (thread) to execute the user defined routine.
 * Worker is registered into the cuPHYDriver context
 *
 * @param[in] pdh cuPHYDriver handler.
 * @param[out] wh where to store the new worker handler.
 * @param[in] name Worker name.
 * @param[in] affinity_cores affinity to CPU core for this new worker.
 * @param[in] sched_priority scheduling priority for this worker.
 * @param[in] wr routine the new worker has to execute
 * @param[in] args routine args
 * 
 * @return
 * \p 0 on success, Linux error code otherwise
 *
 * @sa ::l1_worker_start_default,::l1_worker_stop
 */
int l1_worker_start_generic(phydriver_handle pdh, phydriverwrk_handle* wh, const char* name, uint8_t affinity_cores, uint32_t sched_priority, worker_routine wr, void* args);

/******************************************************************/ /**
 * @brief Return the Worker ID assigned by PhyDriver
 *
 * @param[in] wh Worker handler.
 *
 * @return
 * \p worker id on success, Linux error code otherwise
 *
 * @sa ::l1_worker_start_default,::l1_worker_start_generic
 */
worker_id l1_worker_get_id(phydriverwrk_handle wh);

/******************************************************************/ /**
 * @brief Return cuPHYDriver handler from Worker handler.
 *
 * @param[in] wh Worker handler.
 *
 * @return
 * \p cuPHYDriver handler
 *
 * @sa ::l1_worker_start_default,::l1_worker_start_generic
 */
phydriver_handle l1_worker_get_phydriver_handler(phydriverwrk_handle wh);

/******************************************************************/ /**
 * @brief Check Worker's exit condition
 *
 * This function must be called within a generic worker routing to verify if the worker has to be terminated
 *
 * @param[in] wh Worker handler.
 * 
 * @return
 * \p true if worker has to exit, false otherwise
 *
 * @sa ::l1_worker_start_default,::l1_worker_start_generic
 */
bool l1_worker_check_exit(phydriverwrk_handle wh);

/******************************************************************/ /**
 * @brief Set Woker's exit condition to true
 *
 * This function must be called within a generic worker routing to verify if the worker has to be terminated
 *
 * @param[in] wh Worker handler.
 * 
 * @return
 * \p 0 on success, -1 otherwise
 *
 * @sa ::l1_worker_start_default,::l1_worker_stop
 */
int l1_worker_stop(phydriverwrk_handle wh);

/** @} */ /* END WORKERS */

/**
 * @defgroup Cells cuPHYDriver cells management
 *
 * API dedicated to cuPHYDriver Cells management
 *
 * @{
 */

/******************************************************************/ /**
 * @brief Create a new cell
 *
 * Create a new cell and cuPHY related objects
 * By deafult, a new cell is not active and requires l1_cell_start() to be activated
 *
 * Returns ::0 on success, -1 otherwise
 * 
 * @param[in] pdh cuPHYDriver handler.
 * @param[in] cell_pinfo Static info required to initialize a new cell
 *
 * @return
 * \p 0 on success, -1 otherwise
 *
 * @sa ::l1_cell_start,::l1_cell_stop
 */
int l1_cell_create(phydriver_handle pdh, struct cell_phy_info& cell_pinfo);

/******************************************************************/ /**
 * @brief Destroy a new cell
 *
 * Allocates a cuPHY library context and returns a handle in the
 * address provided by the caller.
 *
 * Returns ::0 on success, -1 otherwise
 * 
 * @param[in] pdh opaque handler to cuPHYDriver context
 * @param[in] cell_id cell to be destroyed.
 *
 * @return
 * ::0
 *
 * @sa ::l1_cell_create,::l1_cell_start,::l1_cell_stop
 */
int l1_cell_destroy(phydriver_handle pdh, uint16_t cell_id);

/******************************************************************/ /**
 * @brief Activate a cell
 *
 * Activate an already created cell
 *
 * Returns ::0 on success, -1 otherwise
 * 
 * @param[in] pdh cuPHYDriver handler.
 * @param[in] cell_id cell to be started.
 *
 * @return
 * \p 0 on success, -1 otherwise
 *
 * @sa ::l1_cell_create,::l1_cell_stop
 */
int l1_cell_start(phydriver_handle pdh, uint16_t cell_id);

/******************************************************************/ /**
 * @brief Deactivate a cell
 *
 * Activate an already created cell
 *
 * Returns ::0 on success, -1 otherwise
 * 
 * @param[in] pdh cuPHYDriver handler.
 * @param[in] cell_id cell to be stopped.
 *
 * @return
 * \p 0 on success, -1 otherwise
 *
 * @sa ::l1_cell_create,::l1_cell_stop
 */
int l1_cell_stop(phydriver_handle pdh, uint16_t cell_id);

/** @} */ /* END CELLS */

/**
 * @defgroup TaskCalculation Task count calculation
 *
 * API for calculating number of tasks for UL and DL processing
 *
 * @{
 */

/******************************************************************/ /**
 * @brief Get number of ULC (Uplink Control) tasks per slot
 *
 * Calculates the number of uplink control plane tasks that must be processed
 * for each downlink slot based on the number of workers.
 *
 * @param[in] num_workers Number of worker threads available
 *
 * @return
 * \p Number of ULC tasks required per DL slot
 */
int get_num_ulc_tasks(int num_workers);

/******************************************************************/ /**
 * @brief Get number of DLC (Downlink Control) tasks per slot
 *
 * Calculates the number of downlink control plane tasks that must be processed
 * for each downlink slot based on number of workers, communication mode, and MIMO configuration.
 *
 * @param[in] num_workers Number of worker threads available
 * @param[in] commViaCpu Communication via CPU flag (true=via CPU, false=GPU direct)
 * @param[in] mMIMO_enable Massive MIMO enable flag (0=disabled, 1=enabled)
 *
 * @return
 * \p Number of DLC tasks required per DL slot
 */
int get_num_dlc_tasks(int num_workers,bool commViaCpu, uint8_t mMIMO_enable);

/** @} */ /* END TASK CALCULATION */

/******************************************************************/ /**
 * @brief Create a new Uplink or Downlink L1 pipeline for the current slot
 *
 * This should be used by L2 Adapter to enqueue new UL/DL tasks for cells foreach slot.
 *
 * Returns ::0 on success, -1 otherwise
 * 
 * @param[in] pdh cuPHYDriver handler.
 * @param[in] sc List of phy channels to be executed per cell during the current slot
 *
 * @return
 * \p 0 on success, -1 otherwise
 *
 */
int l1_enqueue_phy_work(phydriver_handle pdh, struct slot_command_api::slot_command* sc);

/******************************************************************/ /**
 * @brief Set the output callback for UL and DL
 *
 * This should be used by L2 Adapter to enqueue new UL/DL tasks for cells foreach slot.
 *
 * Returns ::0 on success, -1 otherwise
 * 
 * @param[in] pdh cuPHYDriver handler.
 * @param[in] cb Structure holding two callbacks, one for UL and one for DL
 *
 * @return
 * \p 0 on success, -1 otherwise
 *
 */
int l1_set_output_callback(phydriver_handle pdh, struct slot_command_api::callbacks& cb);

/** @} */ /* END WORK */

/**
 * @defgroup Logger Set cuPHYDriver log function
 *
 * Different functions can be used for error, info and debug log levels
 *
 * @{
 */

/**
 * Log handler function pointer type
 * Takes a NULL-terminated string containing the formatted log message
 */
typedef void (*log_handler_fn_t)(const char*);

/******************************************************************/ /**
 * @brief cuPHYDriver log levels
 *
 */
enum l1_log_level
{
    L1_LOG_LVL_ERROR = 1 << 0,                                  ///< Error log level
    L1_LOG_LVL_INFO  = 1 << 1,                                  ///< Info log level
    L1_LOG_LVL_DBG   = 1 << 2                                   ///< Debug log level
};

/******************************************************************/ /**
 * @brief Configures the function to be used for printing errors
 *
 * By default, cuPHYDriver driver prints errors to stderr. Those error messages can be
 * silenced by providing NULL pointer as the function handler. Otherwise,
 * a custom function can be provided for printing error messages.
 *
 * The messages are formatted internally in cuPHYDriver, a single NULL-terminated 
 * const char * is provided to the handler function
 * 
 * @param[in] pdh cuPHYDriver handler.
 * @param[in] log_handler_function Function to be used for error log
 *
 * @return
 * \p 0 on success, -1 otherwise
 */
int l1_set_log_error_handler(phydriver_handle pdh, log_handler_fn_t log_handler_function);

/******************************************************************/ /**
 * @brief Configures the function to be used for printing logs ad info level
 *
 * By default, cuPHYDriver driver prints info messages to stdout. Those messages can be
 * silenced by providing NULL pointer as the function handler. Otherwise,
 * a custom function can be provided for printing messages.
 *
 * The messages are formatted internally in cuPHYDriver, a single NULL-terminated 
 * const char * is provided to the handler function
 * 
 * @param[in] pdh cuPHYDriver handler.
 * @param[in] log_handler_function Function to be used for info log
 *
 * @return
 * \p 0 on success, -1 otherwise
 */
int l1_set_log_info_handler(phydriver_handle pdh, log_handler_fn_t log_handler_function);

/******************************************************************/ /**
 * @brief Configures the function to be used for printing debug and info level
 *
 * By default, cuPHYDriver driver prints debug messages to stdout. Those messages can be
 * silenced by providing NULL pointer as the function handler. Otherwise,
 * a custom function can be provided for printing messages.
 *
 * The messages are formatted internally in cuPHYDriver, a single NULL-terminated 
 * const char * is provided to the handler function
 * 
 * @param[in] pdh cuPHYDriver handler.
 * @param[in] log_handler_function Function to be used for debug log
 *
 * @return
 * \p 0 on success, -1 otherwise
 */
int l1_set_log_debug_handler(phydriver_handle pdh, log_handler_fn_t log_handler_function);

/******************************************************************/ /**
 * @brief Set cuPHYDriver log level
 *
 * By default, cuPHYDriver has error level log. With this function it's possible to have
 * more messages setting L1_LOG_LVL_INFO or L1_LOG_LVL_DBG message level
 * 
 * @param[in] pdh cuPHYDriver handler.
 * @param[in] log_lvl Log level
 *
 * @return
 * \p 0 on success, -1 otherwise
 */
int l1_set_log_level(phydriver_handle pdh, l1_log_level log_lvl);

/** @} */ /* END LOGGER */

/**
 * @defgroup CellUpdate cuPHYDriver cell configuration update
 *
 * API dedicated to updating cell configuration dynamically
 *
 * @{
 */

/******************************************************************/ /**
 * @brief Update cell configuration with resource grid size
 *
 * Updates the resource grid size (number of PRBs) for a cell's downlink or uplink.
 * Cell must not be active when calling this function.
 *
 * @param[in] pdh cuPHYDriver handler
 * @param[in] mplane_id M-plane identifier of the cell to update
 * @param[in] grid_sz Resource grid size (number of PRBs)
 * @param[in] dl Direction flag (true=downlink, false=uplink)
 *
 * @return
 * \p 0 on success, -1 otherwise
 */
int l1_cell_update_cell_config(phydriver_handle pdh, uint16_t mplane_id, uint16_t grid_sz, bool dl);

/******************************************************************/ /**
 * @brief Update cell configuration with MAC address and VLAN
 *
 * Updates the destination MAC address and VLAN TCI for a cell's fronthaul interface.
 * Cell must not be active when calling this function.
 *
 * @param[in] pdh cuPHYDriver handler
 * @param[in] mplane_id M-plane identifier of the cell to update
 * @param[in] dst_mac Destination MAC address string (e.g., "aa:bb:cc:dd:ee:ff")
 * @param[in] vlan_tci VLAN Tag Control Information (TCI) value
 *
 * @return
 * \p 0 on success, -1 otherwise
 */
int l1_cell_update_cell_config(phydriver_handle pdh, uint16_t mplane_id, std::string dst_mac, uint16_t vlan_tci);

/******************************************************************/ /**
 * @brief Update cell configuration with eAxC ID mappings
 *
 * Updates the extended Antenna-Carrier (eAxC) ID mappings for different channel types.
 * Cell must not be active when calling this function.
 *
 * @param[in] pdh cuPHYDriver handler
 * @param[in] mplane_id M-plane identifier of the cell to update
 * @param[in] eAxCIDs_ch_map Map of channel type to vector of eAxC IDs
 *
 * @return
 * \p 0 on success, -1 otherwise
 */
int l1_cell_update_cell_config(phydriver_handle pdh, uint16_t mplane_id, std::unordered_map<int, std::vector<uint16_t>>& eAxCIDs_ch_map);

/******************************************************************/ /**
 * @brief Update cell configuration with generic attributes
 *
 * Updates multiple cell configuration attributes using key-value pairs.
 * Some parameters can be updated while cell is active (e.g., NIC settings),
 * others require cell to be stopped. Results map indicates success/failure per attribute.
 *
 * @param[in] pdh cuPHYDriver handler
 * @param[in] mplane_id M-plane identifier of the cell to update
 * @param[in] attrs Map of attribute names to values (see CELL_PARAM_* constants)
 * @param[out] res Output map of attribute names to result codes (0=success, -1=failed)
 *
 * @return
 * \p 0 on success, -1 otherwise
 */
int l1_cell_update_cell_config(phydriver_handle pdh, uint16_t mplane_id, std::unordered_map<std::string, double>& attrs, std::unordered_map<std::string, int>& res);

/******************************************************************/ /**
 * @brief Update cell attenuation
 *
 * Updates the signal attenuation value for a cell, used for power scaling adjustments.
 *
 * @param[in] pdh cuPHYDriver handler
 * @param[in] mplane_id M-plane identifier of the cell to update
 * @param[in] attenuation_dB Attenuation value in decibels
 *
 * @return
 * \p 0 on success, -1 otherwise
 */
int l1_cell_update_attenuation(phydriver_handle pdh, uint16_t mplane_id, float attenuation_dB);

/******************************************************************/ /**
 * @brief Update GPS timing parameters
 *
 * Updates the GPS alpha and beta parameters used for timing synchronization.
 * These parameters are typically derived from PTP/GPS time alignment.
 *
 * @param[in] pdh cuPHYDriver handler
 * @param[in] alpha GPS alpha timing parameter
 * @param[in] beta GPS beta timing parameter (signed offset)
 *
 * @return
 * \p 0 on success, -1 otherwise
 */
int l1_update_gps_alpha_beta(phydriver_handle pdh,uint64_t alpha,int64_t beta);

/**
 * Cell update callback function type
 * Parameters: (result_code, status)
 */
using CellUpdateCallBackFn = std::function<void(int32_t, uint8_t)>;

/******************************************************************/ /**
 * @brief Update cell configuration with PHY info and callback
 *
 * Updates comprehensive cell PHY configuration including static parameters,
 * PRACH configurations, and SRS settings. Callback is invoked upon completion.
 * Allows updating Physical Cell ID (PCI) and other cell parameters.
 *
 * @param[in] pdh cuPHYDriver handler
 * @param[in] cell_pinfo Cell PHY information structure with new configuration
 * @param[in] callback Callback function invoked on completion (params: result code, status)
 *
 * @return
 * \p 0 on success, -1 otherwise
 */
int l1_cell_update_cell_config(phydriver_handle pdh, struct cell_phy_info& cell_pinfo, ::CellUpdateCallBackFn& callback);

/******************************************************************/ /**
 * @brief Get PRACH starting RO (Resource Occasion) index
 *
 * Retrieves the starting Random Access Resource Occasion index for a cell.
 * Used for PRACH resource allocation and configuration.
 *
 * @param[in] pdh cuPHYDriver handler
 * @param[in] phyCellId Physical cell ID
 *
 * @return
 * \p Starting RO index for PRACH
 */
uint8_t l1_get_prach_start_ro_index(phydriver_handle pdh, uint16_t phyCellId);

/******************************************************************/ /**
 * @brief Allocate SRS channel estimate buffer pool for a cell
 *
 * Allocates a pool of SRS channel estimate buffers from the global memory bank
 * for exclusive use by a specific cell. Must be called before SRS processing begins.
 *
 * @param[in] pdh cuPHYDriver handler
 * @param[in] requestedBy FAPI message type requesting allocation (SCF_FAPI_CONFIG_REQUEST=0x02, CV_MEM_BANK_CONFIG_REQUEST=0x92, or SCF_FAPI_START_REQUEST=0x04)
 * @param[in] phyCellId Physical cell ID to allocate buffers for
 * @param[in] poolSize Number of buffers to allocate from global pool
 *
 * @return
 * \p true on success, false if insufficient buffers available or invalid request source
 *
 * @sa ::l1_deAllocSrsChesBuffPool
 */
bool l1_allocSrsChesBuffPool(phydriver_handle pdh, uint32_t requestedBy, uint16_t phyCellId, uint32_t poolSize);

/******************************************************************/ /**
 * @brief Deallocate SRS channel estimate buffer pool for a cell
 *
 * Returns a cell's SRS channel estimate buffers back to the global memory bank.
 * Should be called when cell is stopped or before reconfiguration.
 *
 * @param[in] pdh cuPHYDriver handler
 * @param[in] phyCellId Physical cell ID to deallocate buffers for
 *
 * @return
 * \p true on success, false on failure
 *
 * @sa ::l1_allocSrsChesBuffPool
 */
bool l1_deAllocSrsChesBuffPool(phydriver_handle pdh, uint16_t phyCellId);

/******************************************************************/ /**
 * @brief Lock cell configuration update mutex
 *
 * Acquires a lock on the cell configuration update mutex to ensure thread-safe
 * updates to cell parameters. Must be paired with ::l1_unlock_update_cell_config_mutex.
 *
 * @param[in] pdh cuPHYDriver handler
 *
 * @return
 * \p 0 on success, -1 otherwise
 *
 * @sa ::l1_unlock_update_cell_config_mutex
 */
int l1_lock_update_cell_config_mutex(phydriver_handle pdh);

/******************************************************************/ /**
 * @brief Unlock cell configuration update mutex
 *
 * Releases the cell configuration update mutex acquired by ::l1_lock_update_cell_config_mutex.
 *
 * @param[in] pdh cuPHYDriver handler
 *
 * @return
 * \p 0 on success, -1 otherwise
 *
 * @sa ::l1_lock_update_cell_config_mutex
 */
int l1_unlock_update_cell_config_mutex(phydriver_handle pdh);

/** @} */ /* END CELL UPDATE */

/**
 * @defgroup TransportBlock Transport block host-to-device copy
 *
 * API for copying transport blocks from host to GPU device memory
 *
 * @{
 */

/******************************************************************/ /**
 * @brief Copy transport block to GPU buffer synchronously
 *
 * Copies a downlink transport block from host memory to GPU device memory.
 * This is a synchronous operation that uses cudaMemcpyAsync on the H2D stream.
 *
 * @param[in] pdh cuPHYDriver handler
 * @param[in] phy_cell_id Physical cell ID
 * @param[in] tb_buff Pointer to transport block data in host memory
 * @param[in,out] gpu_buff_ref Reference to GPU buffer pointer (will be updated with destination address)
 * @param[in] tb_len Transport block length in bytes
 * @param[in] slot_index Slot index within circular buffer (0-19)
 *
 * @sa ::l1_copy_TB_to_gpu_buf_thread_offload
 */
void l1_copy_TB_to_gpu_buf(phydriver_handle pdh, uint16_t phy_cell_id, uint8_t * tb_buff, uint8_t ** gpu_buff_ref, uint32_t tb_len, uint8_t slot_index);

/******************************************************************/ /**
 * @brief Offload transport block copy to dedicated thread
 *
 * Queues a transport block copy request to a dedicated H2D copy thread for asynchronous processing.
 * If H2D copy thread is disabled, falls back to synchronous copy.
 *
 * @param[in] pdh cuPHYDriver handler
 * @param[in] phy_cell_id Physical cell ID
 * @param[in] tb_buff Pointer to transport block data in host memory
 * @param[in,out] gpu_buff_ref Reference to GPU buffer pointer (will be updated with destination address)
 * @param[in] tb_len Transport block length in bytes
 * @param[in] slot_index Slot index within circular buffer (0-19)
 *
 * @sa ::l1_copy_TB_to_gpu_buf, ::l1_copy_TB_to_gpu_buf_thread_func
 */
void l1_copy_TB_to_gpu_buf_thread_offload(phydriver_handle pdh, uint16_t phy_cell_id, uint8_t * tb_buff, uint8_t ** gpu_buff_ref, uint32_t tb_len, uint8_t slot_index, uint16_t sfn = 0);

/******************************************************************/ /**
 * @brief H2D copy thread function
 *
 * Thread function that processes transport block copy requests from the queue.
 * Runs continuously until context finalization, copying transport blocks from
 * host to device memory and recording completion events.
 *
 * @param[in] arg cuPHYDriver handler (phydriver_handle) passed as void*
 *
 * @return
 * \p nullptr on thread exit
 *
 * @sa ::l1_copy_TB_to_gpu_buf_thread_offload
 */
void* l1_copy_TB_to_gpu_buf_thread_func(void* arg);

/******************************************************************/ /**
 * @brief Set H2D copy completion flag for current slot
 *
 * Marks that all H2D transport block copies for a given slot have been completed.
 * Used for synchronization between copy thread and PDSCH processing tasks.
 *
 * @param[in] pdh cuPHYDriver handler
 * @param[in] slot_idx Slot index to mark as complete (0-19)
 */
void l1_set_h2d_copy_done_cur_slot_flag(phydriver_handle pdh,int slot_idx);

/** @} */ /* END TRANSPORT BLOCK */

/**
 * @defgroup SRSMemBank SRS channel estimate memory bank
 *
 * API for managing SRS channel estimate buffers in the global memory bank
 *
 * @{
 */

/******************************************************************/ /**
 * @brief Update SRS channel estimate buffer with new data
 *
 * Updates an SRS channel estimate buffer in the memory bank with new channel
 * estimate data from cuPHY SRS processing. Allocates the buffer if needed and
 * copies the channel estimates to GPU device memory.
 *
 * @param[in] pdh cuPHYDriver handler
 * @param[in] cell_id Cell identifier
 * @param[in] rnti Radio Network Temporary Identifier (UE ID)
 * @param[in] buffer_idx Buffer index within cell's allocated pool
 * @param[in] reportType SRS report type (periodic/aperiodic/etc.)
 * @param[in] startPrbGrp Starting PRB group index for SRS allocation
 * @param[in] srsPrbGrpSize Size of each PRB group
 * @param[in] numPrgs Total number of PRB groups
 * @param[in] nGnbAnt Number of gNodeB receive antennas
 * @param[in] nUeAnt Number of UE transmit antenna ports
 * @param[in] offset Byte offset within buffer for copy
 * @param[in] srsChEsts Pointer to SRS channel estimate data (GPU device memory)
 * @param[in] startValidPrg First valid PRB group index
 * @param[in] nValidPrg Number of valid PRB groups
 *
 * @return
 * \p 0 on success, negative on error
 *
 * @sa ::l1_cv_mem_bank_retrieve_buffer
 */
int l1_cv_mem_bank_update(phydriver_handle pdh,uint32_t cell_id,uint16_t rnti,uint16_t buffer_idx,uint16_t reportType,uint16_t startPrbGrp,uint32_t srsPrbGrpSize,uint16_t numPrgs,uint8_t nGnbAnt,
    uint8_t nUeAnt,uint32_t offset, uint8_t* srsChEsts, uint16_t startValidPrg, uint16_t nValidPrg);

/******************************************************************/ /**
 * @brief Retrieve SRS channel estimate buffer
 *
 * Retrieves an existing SRS channel estimate buffer from the memory bank.
 * Returns buffer information including PRB allocation details, tensor descriptor,
 * and pointer to GPU device memory.
 *
 * @param[in] pdh cuPHYDriver handler
 * @param[in] cell_id Cell identifier
 * @param[in] rnti Radio Network Temporary Identifier (UE ID)
 * @param[in] buffer_idx Buffer index within cell's allocated pool
 * @param[in] reportType SRS report type (periodic/aperiodic/etc.)
 * @param[out] pSrsPrgSize Output: PRB group size
 * @param[out] pSrsStartPrg Output: Starting PRB group index
 * @param[out] pSrsStartValidPrg Output: First valid PRB group index
 * @param[out] pSrsNValidPrg Output: Number of valid PRB groups
 * @param[out] descr Output: cuPHY tensor descriptor handle
 * @param[out] ptr Output: Pointer to buffer data in GPU device memory
 *
 * @return
 * \p 0 on success, negative on error
 *
 * @sa ::l1_cv_mem_bank_update
 */
int l1_cv_mem_bank_retrieve_buffer(phydriver_handle pdh, uint32_t cell_id, uint16_t rnti, uint16_t buffer_idx, uint16_t reportType, uint8_t *pSrsPrgSize, uint16_t* pSrsStartPrg, uint16_t* pSrsStartValidPrg, uint16_t* pSrsNValidPrg, cuphyTensorDescriptor_t* descr, uint8_t** ptr);

/******************************************************************/ /**
 * @brief Update SRS channel estimate buffer state
 *
 * Updates the state of an SRS channel estimate buffer (INIT, REQUESTED, READY, NONE).
 * Used for tracking buffer lifecycle and synchronization.
 *
 * @param[in] pdh cuPHYDriver handler
 * @param[in] cell_id Cell identifier
 * @param[in] buffer_idx Buffer index within cell's allocated pool
 * @param[in] srs_chest_buff_state New buffer state
 *
 * @return
 * \p 0 on success, negative on error
 *
 * @sa ::l1_cv_mem_bank_get_buffer_state
 */
int l1_cv_mem_bank_update_buffer_state(phydriver_handle pdh, uint32_t cell_id, uint16_t buffer_idx, slot_command_api::srsChestBuffState srs_chest_buff_state);

/******************************************************************/ /**
 * @brief Get SRS channel estimate buffer state
 *
 * Retrieves the current state of an SRS channel estimate buffer.
 *
 * @param[in] pdh cuPHYDriver handler
 * @param[in] cell_id Cell identifier
 * @param[in] buffer_idx Buffer index within cell's allocated pool
 * @param[out] srs_chest_buff_state Output: Current buffer state
 *
 * @return
 * \p 0 on success, negative on error
 *
 * @sa ::l1_cv_mem_bank_update_buffer_state
 */
int l1_cv_mem_bank_get_buffer_state(phydriver_handle pdh, uint32_t cell_id, uint16_t buffer_idx, slot_command_api::srsChestBuffState *srs_chest_buff_state);

/******************************************************************/ /**
 * @brief Update SRS channel estimate buffer usage counter
 *
 * Updates the reference/usage count for an SRS channel estimate buffer.
 * Used for buffer lifetime management and garbage collection.
 *
 * @param[in] pdh cuPHYDriver handler
 * @param[in] cell_id Cell identifier
 * @param[in] rnti Radio Network Temporary Identifier (UE ID)
 * @param[in] buffer_idx Buffer index within cell's allocated pool
 * @param[in] usage New usage count
 *
 * @return
 * \p 0 on success, negative on error
 *
 * @sa ::l1_cv_mem_bank_get_buffer_usage
 */
int l1_cv_mem_bank_update_buffer_usage(phydriver_handle pdh,uint32_t cell_id, uint16_t rnti, uint16_t buffer_idx, uint32_t usage);

/******************************************************************/ /**
 * @brief Get SRS channel estimate buffer usage counter
 *
 * Retrieves the current reference/usage count for an SRS channel estimate buffer.
 *
 * @param[in] pdh cuPHYDriver handler
 * @param[in] cell_id Cell identifier
 * @param[in] rnti Radio Network Temporary Identifier (UE ID)
 * @param[in] buffer_idx Buffer index within cell's allocated pool
 * @param[out] usage Output: Current usage count
 *
 * @return
 * \p 0 on success, negative on error
 *
 * @sa ::l1_cv_mem_bank_update_buffer_usage
 */
int l1_cv_mem_bank_get_buffer_usage(phydriver_handle pdh,uint32_t cell_id, uint16_t rnti, uint16_t buffer_idx, uint32_t* usage);

/** @} */ /* END SRS MEM BANK */

/**
 * @defgroup ContextInfo cuPHYDriver context information queries
 *
 * API for querying context-level configuration and state
 *
 * @{
 */

/******************************************************************/ /**
 * @brief Get MU-MIMO enable status
 *
 * Retrieves whether multi-user MIMO (MU-MIMO) processing is enabled for the context.
 *
 * @param[in] pdh cuPHYDriver handler
 * @param[out] pMuMIMO_enable Output: MU-MIMO enable flag (0=disabled, 1=enabled)
 *
 * @return
 * \p 0 on success
 */
int l1_mMIMO_enable_info(phydriver_handle pdh, uint8_t *pMuMIMO_enable);

/******************************************************************/ /**
 * @brief Get SRS enable status
 *
 * Retrieves whether SRS (Sounding Reference Signal) processing is enabled for the context.
 *
 * @param[in] pdh cuPHYDriver handler
 * @param[out] pEnable_srs Output: SRS enable flag (0=disabled, 1=enabled)
 *
 * @return
 * \p 0 on success
 */
int l1_enable_srs_info(phydriver_handle pdh, uint8_t *pEnable_srs);

/******************************************************************/ /**
 * @brief Get cell group number
 *
 * Retrieves the total number of cell groups configured in the context.
 *
 * @param[in] pdh cuPHYDriver handler
 * @param[out] cell_group_num Output: Total number of cell groups
 *
 * @return
 * \p 0 on success
 */
int l1_get_cell_group_num(phydriver_handle pdh, uint8_t *cell_group_num);

/******************************************************************/ /**
 * @brief Get channel segmentation processing enable status
 *
 * Retrieves whether channel segmentation processing is enabled for the context.
 *
 * @param[in] pdh cuPHYDriver handler
 * @param[out] ch_seg_proc_enable Output: Channel segmentation enable flag (0=disabled, 1=enabled)
 *
 * @return
 * \p 0 on success
 */
int l1_get_ch_segment_proc_enable_info(phydriver_handle pdh, uint8_t* ch_seg_proc_enable);

/******************************************************************/ /**
 * @brief Get weighted average CFO enable status
 *
 * Retrieves whether weighted average carrier frequency offset (CFO) estimation is enabled.
 *
 * @param[in] pdh cuPHYDriver handler
 *
 * @return
 * \p Weighted average CFO enable flag (0=disabled, 1=enabled)
 */
uint8_t l1_get_enable_weighted_average_cfo(phydriver_handle pdh);

/******************************************************************/ /**
 * @brief Get DL TX notification enable status
 *
 * Retrieves whether downlink transmission completion notifications are enabled.
 *
 * @param[in] pdh cuPHYDriver handler
 *
 * @return
 * \p true if TX notifications enabled, false otherwise
 */
bool l1_get_dl_tx_notification(phydriver_handle pdh);

/******************************************************************/ /**
 * @brief Get static BFW send with all C-plane flag
 *
 * Retrieves whether static beamforming weights should be sent with all C-plane messages
 * or only once during initialization.
 *
 * @param[in] pdh cuPHYDriver handler
 *
 * @return
 * \p 0=send once, 1=send with all C-plane messages
 */
int l1_get_send_static_bfw_wt_all_cplane(phydriver_handle pdh);

/** @} */ /* END CONTEXT INFO */

/**
 * @defgroup Beamforming Beamforming weights management
 *
 * API for managing beamforming weights in fronthaul
 *
 * @{
 */

/******************************************************************/ /**
 * @brief Store DBT (Digital BeamForming Table) PDU in fronthaul
 *
 * Stores a Digital Beamforming Table PDU in the fronthaul driver for a specific cell.
 * Used for offline beamforming weight configuration.
 *
 * @param[in] pdh cuPHYDriver handler
 * @param[in] cell_id Cell identifier
 * @param[in] data_buf Pointer to DBT PDU data buffer
 *
 * @return
 * \p 0 on success, -1 on error
 */
int l1_storeDBTPduInFH(phydriver_handle pdh, uint16_t cell_id, void* data_buf);

/******************************************************************/ /**
 * @brief Reset DBT storage in fronthaul
 *
 * Clears all stored Digital Beamforming Table PDUs for a specific cell in the fronthaul driver.
 *
 * @param[in] pdh cuPHYDriver handler
 * @param[in] cell_id Cell identifier
 *
 * @return
 * \p 0 on success, -1 on error
 */
int l1_resetDBTStorageInFH(phydriver_handle pdh, uint16_t cell_id);

/******************************************************************/ /**
 * @brief Get beam weights sent flag from fronthaul
 *
 * Checks whether beamforming weights for a specific beam index have been sent
 * to the Radio Unit via the fronthaul interface.
 *
 * @param[in] pdh cuPHYDriver handler
 * @param[in] cell_id Cell identifier
 * @param[in] beamIdx Beam index
 *
 * @return
 * \p 1 if weights have been sent, 0 if not sent, -1 on error
 */
int l1_getBeamWeightsSentFlagInFH(phydriver_handle pdh, uint16_t cell_id, uint16_t beamIdx);

/******************************************************************/ /**
 * @brief Set beam weights sent flag in fronthaul
 *
 * Marks beamforming weights for a specific beam index as having been sent
 * to the Radio Unit via the fronthaul interface.
 *
 * @param[in] pdh cuPHYDriver handler
 * @param[in] cell_id Cell identifier
 * @param[in] beamIdx Beam index
 *
 * @return
 * \p 0 on success, -1 on error
 */
int l1_setBeamWeightsSentFlagInFH(phydriver_handle pdh, uint16_t cell_id, uint16_t beamIdx);

/******************************************************************/ /**
 * @brief Check if static BFW is configured in fronthaul
 *
 * Checks whether static beamforming weights have been configured for a cell
 * in the fronthaul driver.
 *
 * @param[in] pdh cuPHYDriver handler
 * @param[in] cell_id Cell identifier
 *
 * @return
 * \p 1 if static BFW configured, 0 if not configured, -1 on error
 */
int l1_staticBFWConfiguredInFH(phydriver_handle pdh, uint16_t cell_id);

/******************************************************************/ /**
 * @brief Get dynamic beam ID offset from fronthaul
 *
 * Retrieves the offset value for dynamic beam IDs configured in the fronthaul driver.
 * Dynamic beams use IDs starting from (static_beam_id_end + 1 + offset).
 *
 * @param[in] pdh cuPHYDriver handler
 * @param[in] cell_id Cell identifier
 *
 * @return
 * \p Dynamic beam ID offset, or -1 on error
 */
int16_t l1_getDynamicBeamIdOffset(phydriver_handle pdh, uint16_t cell_id);

/** @} */ /* END BEAMFORMING */

/**
 * @defgroup SystemControl System control and recovery
 *
 * API for system-level control, recovery, and diagnostics
 *
 * @{
 */

/******************************************************************/ /**
 * @brief Clear task lists
 *
 * Clears all pending tasks from the uplink and downlink task lists.
 * Used during system shutdown or error recovery.
 *
 * @param[in] pdh cuPHYDriver handler
 *
 * @return
 * \p 0 on success
 */
int l1_clear_task_list(phydriver_handle pdh);

/******************************************************************/ /**
 * @brief L1 exit handler
 *
 * Global exit handler for cuPHYDriver. Called on fatal errors or system shutdown.
 * Performs cleanup: clears task lists, synchronizes CUDA context, closes logs,
 * and triggers system abort if needed. Should not be called directly by user code.
 */
void l1_exit_handler();

/******************************************************************/ /**
 * @brief Get global cuPHYDriver handle
 *
 * Retrieves the global cuPHYDriver context handle. Used internally for signal
 * handlers and exit routines.
 *
 * @return
 * \p Global phydriver_handle, or nullptr if not initialized
 */
phydriver_handle l1_getPhydriverHandle();

/******************************************************************/ /**
 * @brief Get formatted logging thread ID
 *
 * Retrieves the pthread ID of the formatted logging background thread.
 * Used for logging system control.
 *
 * @return
 * \p pthread_t ID of formatted logging thread
 */
pthread_t l1_getFmtLogThreadId();

/******************************************************************/ /**
 * @brief Check cuPHY objects free status
 *
 * Checks whether all cuPHY aggregation objects (PUSCH, PUCCH, PRACH, SRS decoders/encoders)
 * are in the free state. Used for system health monitoring and recovery decisions.
 *
 * @param[in] pdh cuPHYDriver handler
 *
 * @return
 * \p true if all objects are free, false otherwise
 */
bool l1_check_cuphy_objects_status(phydriver_handle pdh);

/******************************************************************/ /**
 * @brief Increment L1 recovery slots counter
 *
 * Increments the counter tracking consecutive slots in recovery mode.
 * Returns true when recovery threshold is reached, indicating system should
 * attempt full recovery or restart.
 *
 * @param[in] pdh cuPHYDriver handler
 *
 * @return
 * \p true if recovery threshold reached, false otherwise
 *
 * @sa ::l1_reset_recovery_slots
 */
bool l1_incr_recovery_slots(phydriver_handle pdh);

/******************************************************************/ /**
 * @brief Increment all objects free slots counter
 *
 * Increments the counter tracking consecutive slots where all cuPHY objects
 * were free. Returns true when threshold is reached, indicating system has
 * successfully recovered.
 *
 * @param[in] pdh cuPHYDriver handler
 *
 * @return
 * \p true if all-objects-free threshold reached, false otherwise
 *
 * @sa ::l1_reset_all_obj_free_slots
 */
bool l1_incr_all_obj_free_slots(phydriver_handle pdh);

/******************************************************************/ /**
 * @brief Reset all objects free slots counter
 *
 * Resets the counter for consecutive slots with all cuPHY objects free.
 * Called when an object becomes busy again.
 *
 * @param[in] pdh cuPHYDriver handler
 *
 * @sa ::l1_incr_all_obj_free_slots
 */
void l1_reset_all_obj_free_slots(phydriver_handle pdh);

/******************************************************************/ /**
 * @brief Reset L1 recovery slots counter
 *
 * Resets the recovery mode slot counter. Called when system exits recovery mode.
 *
 * @param[in] pdh cuPHYDriver handler
 *
 * @sa ::l1_incr_recovery_slots
 */
void l1_reset_recovery_slots(phydriver_handle pdh);

/******************************************************************/ /**
 * @brief Increment FAPI SRS statistics counters
 *
 * Updates SRS packet arrival statistics for a cell, tracking early, on-time,
 * and late packet arrivals. Used for monitoring SRS timing performance.
 *
 * @param[in] pdh cuPHYDriver handler
 * @param[in] cell_id Cell identifier
 * @param[in] sfn System Frame Number (0-1023)
 * @param[in] slot Slot number within frame
 * @param[in] early Number of early SRS packet arrivals
 * @param[in] ontime Number of on-time SRS packet arrivals
 * @param[in] late Number of late SRS packet arrivals
 *
 * @return
 * \p 0 on success, negative on error
 */
int l1_increment_fapi_srs_stats(phydriver_handle pdh, uint32_t cell_id, int sfn, int slot, uint32_t early, uint32_t ontime, uint32_t late);

/******************************************************************/ /**
 * @brief Reset batched memcpy batches
 *
 * Resets all batched memory copy batch counters and state. Used when restarting
 * batched copy operations or during error recovery.
 *
 * @param[in] pdh cuPHYDriver handler
 */
void l1_resetBatchedMemcpyBatches(phydriver_handle pdh);

/** @} */ /* END SYSTEM CONTROL */

/**
 * @defgroup BFWBuffer Beamforming coefficient buffer management
 *
 * API for managing beamforming coefficient buffers
 *
 * @{
 */

/******************************************************************/ /**
 * @brief BFW coefficient buffer state header
 *
 */
struct bfw_buffer_info_header
{
    uint8_t state[slot_command_api::MAX_BFW_COFF_STORE_INDEX]{slot_command_api::BFW_COFF_MEM_FREE};  ///< State of each beamforming coefficient buffer slot (FREE/IN_USE/READY)
};

/******************************************************************/ /**
 * @brief BFW coefficient buffer information
 *
 */
struct bfw_buffer_info
{
    bfw_buffer_info_header* header{};                           ///< Pointer to buffer state header
    uint8_t* dataH{};                                           ///< Pointer to beamforming coefficient data on host (CPU memory)
    uint8_t* dataD{};                                           ///< Pointer to beamforming coefficient data on device (GPU memory)
};

/******************************************************************/ /**
 * @brief Retrieve beamforming coefficient buffer for a cell
 *
 * Retrieves the beamforming coefficient buffer information for a given cell,
 * including buffer state, host memory pointer, and device memory pointer.
 * Used for accessing pre-computed beamforming weights.
 *
 * @param[in] pdh cuPHYDriver handler
 * @param[in] cell_id Cell identifier
 * @param[out] bfw_buffer_info Output: BFW coefficient buffer information structure
 *
 * @return
 * \p 0 on success, -1 on error
 */
int l1_bfw_coeff_retrieve_buffer(phydriver_handle pdh, uint32_t cell_id, struct bfw_buffer_info* bfw_buffer_info);

/** @} */ /* END BFW BUFFER */

/******************************************************************/ /**
 * @brief Get split UL CUDA streams configuration
 *
 * Retrieves whether uplink processing is configured to use split CUDA streams
 * for parallel processing of different UL channels.
 *
 * @param[in] pdh cuPHYDriver handler
 *
 * @return
 * \p true if split UL CUDA streams enabled, false otherwise
 */
bool l1_get_split_ul_cuda_streams(phydriver_handle pdh);
#endif
