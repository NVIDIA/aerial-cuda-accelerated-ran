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

#ifndef AERIAL_FH_DRIVER_API__
#define AERIAL_FH_DRIVER_API__

#include "aerial-fh-driver/oran.hpp"
#include "aerial-fh-driver/doca_structs.hpp"
#include "slot_command/slot_command.hpp"

#include <cstdint>
#include <cstddef>

#include <string>
#include <vector>
#include <array>
#include <chrono>
#include <cuda/std/array>
#include <doca_gpunetio.h>

//#define ENABLE_DPDK_TX_PKT_TRACING

#define MAX_PKT_DEBUG 0xffff               //!< Maximum packet debug entries
#define MAX_DL_SLOTS_TRIGGER_TIME 512      //!< Maximum downlink slot trigger times

namespace aerial_fh
{
//!< Opaque handle for Fronthaul driver instance
using FronthaulHandle        = void*;
//!< Opaque handle for NIC instance
using NicHandle              = void*;
//!< Opaque handle for memory region
using MemRegHandle           = void*;
//!< Opaque handle for peer instance
using PeerHandle             = void*;
//!< Opaque handle for flow rule
using FlowHandle             = void*;
//!< Opaque handle for TX request
using TxRequestHandle        = void*;
//!< Opaque handle for GPU comm TX request
using TxRequestGpuCommHandle = void*;
//!< Opaque handle for ring buffer
using RingBufferHandle       = void*;
//!< Opaque handle for TX queue
using TxqHandle              = void*;
//!< Opaque handle for stream RX
using StreamRxHandle         = void*;

constexpr uint32_t kMaxSectionNum = 273;  //!< Maximum number of sections per U-plane message

/**
 * \defgroup Setup Setup
 *
 * @{
 */

using GpuId = int;
/******************************************************************/ /**
 * \brief Fronthaul driver information
 *
 */
struct FronthaulInfo
{
    uint32_t           dpdk_thread;            //!< CPU core to use for DPDK main process
    uint32_t           accu_tx_sched_res_ns;   //!< accurate send scheduling granularity in ns. Must be between 500 and 1 million
    int32_t            pdump_client_thread;    //!< CPU core to use for pdump (PCAP capture) client. -1 to disable
    bool               accu_tx_sched_disable;  //!< Disable TX on timestamp
    bool               dpdk_verbose_logs;      //!< Enable max log level in DPDK
    std::string        dpdk_file_prefix;       //!< DPDK process shared data file prefix
    std::vector<GpuId> cuda_device_ids;        //!< List of CUDA devices to use
    std::vector<GpuId> cuda_device_ids_for_compute; //!< List of CUDA devices to use for compute
    bool               rivermax;               //!< Enable Rivermax on the RX side
    int32_t            fh_stats_dump_cpu_core; //!< CPU core to use for FH stats dump. -1 to disable
    bool               cpu_rx_only;            //!< Override to only use CPU memory to RX for UE Mode
    uint8_t            enable_gpu_comm_via_cpu; //!< GPU Comms via CPU memory (typically enabled on platforms which do not support P2P)
    uint8_t            bfw_chaining_mode{0};       //!< 0: No chaining 1: CPU chaining 2: GPU chaining
};

/**
 * Packet start debug information
 *
 * Contains timing and identification info for packet debugging
 */
struct PacketStartDebugInfo {
    uint64_t time;           //!< System time (nanoseconds)
    uint64_t ptp_time;       //!< PTP time (nanoseconds)
    uint32_t packet_size;    //!< Packet size in bytes
    uint8_t frame_id;        //!< Frame ID
    uint8_t subframe_id;     //!< Subframe ID
    uint8_t slot_id;         //!< Slot ID
    uint8_t sym;             //!< Symbol number
};

/******************************************************************/ /**
 * \brief Create new instance of the fronthaul driver.
 *
 * \param info - Fronthaul driver info
 * \param fronthaul - Output fronthaul driver handle
 *
 * \return \p 0 on success, Linux error code otherwise
 *
 * \note Each instance of the fronthaul driver is completely independent.
 */
int open(FronthaulInfo* info, FronthaulHandle* fronthaul);

/******************************************************************/ /**
 * \brief Destroy the instance of the fronthaul driver.
 *
 * \param fronthaul - Fronthaul driver handle
 *
 * \return \p 0 on success, Linux error code otherwise
 *
 * \note This function frees all the allocated resources (e.g. memreg, ring, nic etc.)
 */
int close(FronthaulHandle fronthaul);

using GpuId = int;

/******************************************************************/ /**
 * \brief NIC port information
 *
 */
struct NicInfo
{
    std::string name;           //!< NIC device name
    uint16_t    mtu;            //!< maximum packet size that will be sent/received
    bool        per_rxq_mempool;//!< Create mempool buffer per receive queue
    uint32_t    cpu_mbuf_num;   //!< CPU mbuf pool size (for DU)
    uint32_t    cpu_mbuf_tx_num;   //!< CPU mbuf pool size for Tx (for RU)
    uint32_t    cpu_mbuf_rx_num;   //!< CPU mbuf pool size for Rx (for RU)
    uint32_t    cpu_mbuf_rx_num_per_rxq;   //!< CPU mbuf pool size for Rx per RXQ (for RU)
    uint32_t    tx_request_num; //!< U-plane TX request handle mempool size
    uint16_t    txq_count;      //!< Transmit queue count
    uint16_t    txq_count_gpu;  //!< Transmit queue count with GPU-init comm
    uint16_t    rxq_count;      //!< Receive queue count
    uint16_t    txq_size;       //!< Transmit queue size
    uint16_t    rxq_size;       //!< Receive queue size
    GpuId       cuda_device;    //!< CUDA device to RX packets
    bool        rx_ts_enable;   //!< Enable DPDK DEV_RX_OFFLOAD_TIMESTAMP
    bool        split_cpu_mp = false;   //!< Split CPU mempool for RX and TX
};

/******************************************************************/ /**
 * \brief Add a NIC to Fronthaul instance
 *
 * \param fronthaul - Fronthaul driver handle
 * \param info - NIC information
 * \param nic - Output NIC handle
 *
 * \return \p 0 on success, Linux error code otherwise
 */
int add_nic(FronthaulHandle fronthaul, NicInfo const* info, NicHandle* nic);

/******************************************************************/ /**
 * \brief Remove a NIC
 *
 * \param nic - NIC handle
 *
 * \return \p 0 on success, Linux error code otherwise
 */
int remove_nic(NicHandle nic);

/******************************************************************/ /**
 * \brief Print NIC statistics
 *
 * \param nic - NIC handle
 * \param extended - print extended counter values
 *
 * \return \p 0 on success, Linux error code otherwise
 */
int print_stats(NicHandle nic, bool extended = false);

/******************************************************************/ /**
 * \brief Check for CX-6 NIC
 *
 * \param nic - NIC handle
 * \param cx6 - true or false set based on CX-6 NIC device
 *
 * \return \p 0 on success, Linux error code otherwise
 */
int is_cx6_nic(NicHandle handle,bool& cx6);


/******************************************************************/ /**
 * \brief Re-set NIC statistics
 *
 * \param nic - NIC handle
 *
 * \return \p 0 on success, Linux error code otherwise
 */
int reset_stats(NicHandle nic);

/******************************************************************/ /**
 * \brief A struct to hold the bytes of a MAC address
 */
struct MacAddr
{
    uint8_t bytes[6];
} __attribute__((__aligned__(2)));

/******************************************************************/ /**
 * \brief VLAN tag
 */
typedef union
{
    uint16_t tci;
    struct
    {
#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
        uint16_t pcp : 3;
        uint16_t dei : 1;
        uint16_t vid : 12;
#else
        uint16_t vid : 12;
        uint16_t dei : 1;
        uint16_t pcp : 3;
#endif
    };
} __attribute__((packed)) VlanTci;

/******************************************************************/ /**
 * \brief User data compression methods
 */
enum class UserDataCompressionMethod : uint8_t
{
    NO_COMPRESSION                 = 0b0000,
    BLOCK_FLOATING_POINT           = 0b0001,
    BLOCK_SCALING                  = 0b0010,
    U_LAW                          = 0b0011,
    MODULATION_COMPRESSION         = 0b0100,
    BFP_SELECTIVE_RE_SENDING       = 0b0101,
    MOD_COMPR_SELECTIVE_RE_SENDING = 0b0110,
    RESERVED                       = 0b0111,
};

// Configuration-dependent cell limits
#ifdef ENABLE_64C
#define API_MAX_NUM_CELLS 2  //!< Maximum cells (64-cell config, power of 2 for DOCA)
#elif defined(ENABLE_20C)
#define API_MAX_NUM_CELLS 2  //!< Maximum cells (20-cell config, power of 2 for DOCA)
#else
#define API_MAX_NUM_CELLS 2  //!< Maximum cells (default config)
#endif

#ifdef ENABLE_32DL
#define API_MAX_ANTENNAS 32  //!< Maximum antenna ports per cell (32-layer Downlink config)
#else
#define API_MAX_ANTENNAS 16  //!< Maximum antenna ports per cell (default config)
#endif

/**
 * Packet timing information for GPU-accelerated transmission
 *
 * Captures timestamps at various stages of packet preparation and transmission
 * for performance analysis and debugging.
 */
struct PacketTimingInfo
{
    uint8_t     frame_id;        //!< Frame ID
    uint16_t    subframe_id;     //!< Subframe ID
    uint16_t    slot_id;         //!< Slot ID
    std::array<uint64_t, ORAN_ALL_SYMBOLS> pkt_copy_launch_timestamp;  //!< Packet copy launch timestamps per symbol
    std::array<uint64_t, ORAN_ALL_SYMBOLS> pkt_copy_done_timestamp;    //!< Packet copy completion timestamps per symbol
    std::array<uint64_t, ORAN_ALL_SYMBOLS> trigger_done_timestamp;     //!< TX trigger completion timestamps per symbol
    std::array<uint32_t, ORAN_ALL_SYMBOLS> num_packets_per_symbol;     //!< Packet count per symbol
    uint64_t cpu_send_start_timestamp;  //!< CPU send start timestamp
};

/**
 * Timing breakdown for TXQ send operations
 */
struct TxqSendTiming
{
    uint64_t lock_wait_ns{};      //!< Time spent waiting to acquire the lock
    uint64_t tx_burst_loop_ns{};  //!< Time spent in rte_eth_tx_burst loop
};

/**
 * PRB preparation information for GPU-accelerated DL transmission
 *
 * Contains pointers to PRB data, antenna mappings, and CUDA events
 * for synchronized GPU-accelerated packet preparation.
 */
struct PreparePRBInfo {
    uint8_t **prb_ptrs[API_MAX_NUM_CELLS];                        //!< PRB data pointers per cell
    uint8_t eAxCMap[API_MAX_NUM_CELLS][API_MAX_ANTENNAS];         //!< eAxC ID mapping per cell and antenna
    uint8_t num_antennas[API_MAX_NUM_CELLS];                      //!< Number of antennas per cell
    uint16_t max_num_prb_per_symbol[API_MAX_NUM_CELLS];           //!< Maximum PRBs per symbol per cell
    cudaEvent_t compression_stop_evt;                             //!< CUDA event for compression completion
    cudaEvent_t comm_start_evt;                                   //!< CUDA event for communication start
    cudaEvent_t comm_copy_evt;                                    //!< CUDA event for memory copy
    cuda::std::array<cudaEvent_t, ORAN_ALL_SYMBOLS> pkt_copy_evt; //!< Per-symbol packet copy events
    float* p_packet_mem_copy_per_symbol_dur_us;                   //!< Packet memory copy duration per symbol (microseconds)
    cudaEvent_t comm_preprep_stop_evt;                            //!< CUDA event for pre-preparation completion
    cudaEvent_t comm_stop_evt;                                    //!< CUDA event for communication completion
    cudaEvent_t trigger_end_evt;                                  //!< CUDA event for trigger end
    bool disable_empw;                                            //!< Disable enhanced multi-packet write
    bool enable_prepare_tracing;                                  //!< Enable packet preparation tracing
    bool enable_dl_cqe_tracing;                                   //!< Enable downlink CQE tracing
    bool use_copy_kernel_for_d2h;                                 //!< Use CUDA kernel for device-to-host copy
    uint64_t cqe_trace_cell_mask;                                 //!< Cell mask for CQE tracing
    uint32_t cqe_trace_slot_mask;                                 //!< Slot mask for CQE tracing
    uint32_t *ready_flag;                                         //!< Ready flag pointer for synchronization
    uint32_t wait_val;                                            //!< Wait value for synchronization
};


/******************************************************************/ /**
 * \brief User data compression information
 */
struct UserDataCompressionInfo
{
    size_t                    iq_sample_size; //!< I and Q are each 'iq_sample_size' bits wide
    UserDataCompressionMethod method;         //!< user data compression method
};

using PeerId = uint16_t;

/******************************************************************/ /**
 * \brief FH driver receive API mode for peer
 */
enum class RxApiMode
{
    FLOW,   //!< Receive data on each flow separately
    PEER,   //!< Simultaneously receive data on all flows associated with a cell
    HYBRID, //!< Receive C-plane per peer and U-plane per flow
    TXONLY, //!< Don't allocate any RXQs
    UEMODE, //!< Allocate CPU RXQs for RX, DOCA TXQs for TX
};

/******************************************************************/ /**
 * \brief FH driver receive API mode for flow
 */
enum class FlowRxApiMode
{
    TXANDRX, //!< Need to allocate RXQs
    TXONLY,  //!< Don't allocate any RXQs
};

/******************************************************************/ /**
 * \brief Cleanup buffer information for GPU memset kernel
 *
 * Used by memset kernel in both FH driver and application
 */
struct CleanupDlBufInfo {
    uint4* d_buf_addr;   //!< Device buffer address for cleanup
    size_t  buf_size;    //!< Buffer size in bytes
    // Note: Compressed buffer fields commented out - can be extended if needed:
    // uint4* d_comp_buf_addr;  // Compressed buffer address
    // size_t comp_buf_size;     // Compressed buffer size in bytes
};

/**
 * Beamforming weights C-plane chaining mode
 *
 * Determines how BFW C-plane packets are assembled and sent for massive MIMO
 */
enum class BfwCplaneChainingMode
{
    NO_CHAINING,   //!< No chaining - single packet per message
    CPU_CHAINING,  //!< CPU-based mbuf chaining for large BFW messages
    GPU_CHAINING   //!< GPU-based chaining for BFW messages
};

/**
 * Beamforming weights C-plane configuration
 *
 * Configuration for BFW C-plane packet handling in massive MIMO scenarios
 */
struct FhBfwCplaneInfo final
{
    bool dlc_bfw_enable_divide_per_cell{};         //!< Enable per-cell division for DL C-plane BFW
    bool ulc_bfw_enable_divide_per_cell{};         //!< Enable per-cell division for UL C-plane BFW
    BfwCplaneChainingMode bfw_chain_mode = BfwCplaneChainingMode::NO_CHAINING;  //!< BFW chaining mode
    size_t bfw_cplane_buffer_size{};               //!< BFW C-plane buffer size in bytes
    bool dlc_alloc_cplane_bfw_txq{};               //!< Allocate separate TXQ for DL C-plane BFW packets
    bool ulc_alloc_cplane_bfw_txq{};               //!< Allocate separate TXQ for UL C-plane BFW packets
};

/******************************************************************/ /**
 * \brief Peer (e.g. DU or RU) information
 */
struct PeerInfo
{
    PeerId                  id;                     //!< unique peer identifier
    MacAddr                 src_mac_addr;           //!< source MAC address. Set 00:00:00:00:00:00 to use the one of the NIC port
    MacAddr                 dst_mac_addr;           //!< destinaton (peer's) MAC address
    VlanTci                 vlan;                   //!< Vlan ID
    UserDataCompressionInfo ud_comp_info;           //!< static user data compression information
    uint8_t                 txq_count_uplane;       //!< U-plane transmit queue count
    uint8_t                 txq_count_uplane_gpu;   //!< U-plane transmit queue count with GPU-init communications
    RxApiMode               rx_mode;                //!< Receive API mode
    bool                    txq_cplane;             //!< Set to 'true' if you want to send C-plane
    bool                    txq_bfw_cplane;         //!< Set to 'true' if you want to send BFW C-plane
    FhBfwCplaneInfo         bfw_cplane_info;        //!< BFW C-plane information
    uint8_t                 mMIMO_enable;           //!< Set to 1 if 32T32R feature is enabled
    uint8_t                 enable_srs;             //!< Set to 1 if SRS channel is enabled
    uint16_t                max_num_prbs_per_symbol;//!< Cell BW
    bool                    share_txqs = false;     //!< Set to true if you want to share txqs among all peers
};

/******************************************************************/ /**
 * \brief Add a new peer
 *
 * \param nic - NIC handle
 * \param info - Peer information
 * \param peer - Output peer handle
 * \param eAxC_list_ul - UL eAxC ID list
 * \param eAxC_list_srs - SRS eAxC ID list (Only used for mMIMO)
 * \param eAxC_list_dl - DL eAxC ID list (Only used for UE mode)
 *
 * \return \p 0 on success, Linux error code otherwise
 */
int add_peer(NicHandle nic, PeerInfo const* info, PeerHandle* peer,std::vector<uint16_t>& eAxC_list_ul,std::vector<uint16_t>& eAxC_list_srs, std::vector<uint16_t>& eAxC_list_dl);

/******************************************************************/ /**
 * \brief Get GPU regular memory size for peer
 * \param handle - Peer handle
 * \return GPU regular memory size in bytes
 */
size_t get_gpu_regular_size(PeerHandle handle);

/******************************************************************/ /**
 * \brief Get GPU pinned memory size for peer
 * \param handle - Peer handle
 * \return GPU pinned memory size in bytes
 */
size_t get_gpu_pinned_size(PeerHandle handle);

/******************************************************************/ /**
 * \brief Get CPU regular memory size for peer
 * \param handle - Peer handle
 * \return CPU regular memory size in bytes
 */
size_t get_cpu_regular_size(PeerHandle handle);

/******************************************************************/ /**
 * \brief Get CPU pinned memory size for peer
 * \param handle - Peer handle
 * \return CPU pinned memory size in bytes
 */
size_t get_cpu_pinned_size(PeerHandle handle);

/******************************************************************/ /**
 * \brief Remove a peer
 *
 * \param peer - Peer handle
 *
 * \return \p 0 on success, Linux error code otherwise
 */
int remove_peer(PeerHandle peer);

/******************************************************************/ /**
 * \brief Update peer's compression_bitwidth
 *
 * \param peer - Peer handle
 * \param dl_comp_meth - New dl_comp_meth
 * \param dl_bit_width - New dl_bit_width
 *
 * \return \p 0 on success, Linux error code otherwise
 */
int update_peer(PeerHandle handle, UserDataCompressionMethod dl_comp_meth, uint8_t dl_bit_width);

/******************************************************************/ /**
 * \brief Update peer's compression_bitwidth
 *
 * \param peer - Peer handle
 * \param rx_packets - number of rx packets
 * \param rx_bytes - number of rx bytes
 *
 * \return \p 0 on success, Linux error code otherwise
 */
int update_rx_metrics(PeerHandle handle, size_t rx_packets, size_t rx_bytes);

/******************************************************************/ /**
 * \brief Update peer's compression_bitwidth
 *
 * \param peer - Peer handle
 * \param tx_packets - number of tx packets
 * \param tx_bytes - number of tx bytes
 *
 * \return \p 0 on success, Linux error code otherwise
 */
int update_tx_metrics(PeerHandle handle, size_t tx_packets, size_t tx_bytes);

/******************************************************************/ /**
 * \brief Update peer's destination MAC address
 *
 * \param peer - Peer handle
 * \param dst_mac_addr - New destination MAC address
 *
 * \return \p 0 on success, Linux error code otherwise
 */
int update_peer(PeerHandle peer, MacAddr dst_mac_addr,std::vector<uint16_t>& eAxC_list_ul,std::vector<uint16_t>& eAxC_list_srs);

/******************************************************************/ /**
 * \brief Update peer's destination MAC address
 *
 * \param peer - Peer handle
 * \param dst_mac_addr - New destination MAC address
 * \param vlan_tci - New vlan tci
 *
 * \return \p 0 on success, Linux error code otherwise
 */
int update_peer(PeerHandle peer, MacAddr dst_mac_addr, uint16_t vlan_tci,std::vector<uint16_t>& eAxC_list_ul,std::vector<uint16_t>& eAxC_list_srs);

/******************************************************************/ /**
 * \brief Update peer's MAX PRB's per symbol configuration
 *
 * \param peer - Peer handle
 * \param max_num_prbs_per_symbol - MAX PRBs per symbol
 *
 * \return \p 0 on success, Linux error code otherwise
 */
int update_peer_max_num_prbs_per_symbol(PeerHandle handle,uint16_t max_num_prbs_per_symbol);

/******************************************************************/ /**
 * \brief Get U-plane TXQ handles assigned to peer
 *
 * \param peer - Peer handle
 * \param txqs - List of TXQs peer can use for sending U-plane
 * \param num_txqs - Size of txqs list
 *
 * \return \p 0 on success, Linux error code otherwise
 */
int get_uplane_txqs(PeerHandle peer, TxqHandle* txqs, size_t* num_txqs);

/******************************************************************/ /**
 * \brief Get DOCA Rxq items assigned to peer
 *
 * \param handle - Peer handle
 * \param doca_rxq_items - DOCA RxQ items obtained from DOCA RxQ creation
 *
 * \return \p 0 on success, Linux error code otherwise
 */

/**
 * DOCA GPU inline configuration
 *
 * Configuration for DOCA GPU-accelerated inline packet processing
 */
struct doca_gpu_inline_cfg {
	char gpu_pcie_addr[32];  //!< GPU PCIe address (e.g., "0000:17:00.0")
	uint32_t nic_port;       //!< NIC port number
	uint8_t inference_type;  //!< Inference type (0=none, 1=TensorRT, etc.)
	uint8_t rxq_num;         //!< Number of RX queues
	uint8_t workload_mode;   //!< Workload mode
	uint8_t receive_mode;    //!< Receive mode
};

int get_doca_rxq_items(PeerHandle handle,void* rxq_items);
int get_doca_rxq_items_srs(PeerHandle handle,void* rxq_items);

/**
 * Flow type classification
 */
enum class FlowType
{
    CPLANE,  //!< Control plane flow
    UPLANE   //!< User plane flow
};

/**
 * Flow direction
 */
enum class FlowDir
{
    DL,  //!< Downlink (gNB to UE direction)
    UL   //!< Uplink (UE to gNB direction)
};

//!< Flow identifier type
using FlowId = uint16_t;

/******************************************************************/ /**
 * \brief TX eCPRI flow information
 */
struct FlowInfo
{
    FlowId        eAxC;            //!< eAxC ID for this flow
    FlowType      type;            //!< Flow type (C-plane or U-plane)
    VlanTci       vlan_tag;        //!< VLAN tag
    FlowRxApiMode flow_rx_mode;    //!< RX API mode for this flow
    FlowDir       direction;       //!< Flow direction (DL or UL)
    uint8_t       channel;         //!< Channel number (deprecated, to be removed)
    bool          request_new_rxq; //!< Request new RX queue for this flow
    void*         rxq;             //!< RX queue pointer
};

/******************************************************************/ /**
 * \brief RX Stream eCPRI stream information
 */
struct StreamRxInfo
{
    VlanTci             vlan_tag;        //!< VLAN tag
    std::vector<FlowId> eAxCs;           //!< List of eAxC IDs in this stream
    uint16_t            section_id_max;  //!< Maximum section ID
    int                 prbs_x_eaxc;     //!< PRBs per eAxC
};

/******************************************************************/ /**
 * \brief RX Stream eCPRI stream slot buffers information
 */
struct StreamRxSlotInfo
{
    void*    addr;         //!< Buffer address
    size_t   flow_stride;  //!< Stride between flows in bytes
    uint8_t  eAxC_num;     //!< Number of eAxCs
    uint64_t id;           //!< Slot identifier
};

/******************************************************************/ /**
 * \brief Add a new flow
 *
 * \param peer - Peer handle
 * \param info - Flow information
 * \param flow - Output flow handle
 *
 * \return \p 0 on success, Linux error code otherwise
 */
int add_flow(PeerHandle peer, FlowInfo* info, FlowHandle* flow);


/******************************************************************/ /**
 * \brief Remove a flow
 *
 * \param flow - Flow handle
 *
 * \return \p 0 on success, Linux error code otherwise
 */
int remove_flow(FlowHandle flow);

/******************************************************************/ /**
 * \brief Update flow info
 *
 * \param flow - Flow handle
 * \param info - Flow information
 *
 * \return \p 0 on success, Linux error code otherwise
 */
int update_flow(FlowHandle flow, FlowInfo const* info);


/** @} */ /* END SETUP */

/**
 * \defgroup Memory Memory
 *
 * @{
 */

/******************************************************************/ /**
 * \brief Allocate CPU memory from the huge-page area of memory
 *
 * \param size - Size (in bytes) to be allocated.
 * \param align - If 0, the return is a pointer that is suitably aligned for any kind of variable (in the same manner as malloc()).
 *                Otherwise, the return is a pointer that is a multiple of align. In this case, it must be a power of two.
 *                (Minimum alignment is the cacheline size, i.e. 64-bytes)
 *
 * \return \p NULL on error. Otherwise, the pointer to the allocated object
 *
 * \note The alloc'ed memory can be freed using free()
 */
void* allocate_memory(size_t size, unsigned align);

/******************************************************************/ /**
 * \brief Free CPU memory from huge-page
 *
 * \param ptr - Pointer to memory to free
 *
 * \return \p 0 on success, Linux error code otherwise
 */
int free_memory(void* ptr);

/******************************************************************/ /**
 * \brief Memory registration information
 */
struct MemRegInfo
{
    void*  addr;    //!< Pointer to the start of the buffer to register. Must be page_sz aligned
    size_t len;     //!< Length of the buffer register
    size_t page_sz; //!< Page size of the underlying memory
};

/******************************************************************/ /**
 * \brief Register memory region with Fronthaul instance.
 *
 * \param fronthaul - Fronthaul driver handle
 * \param info - Memory zone information
 * \param memreg - Output handle used for de-registering
 *
 * \return \p 0 on success, Linux error code otherwise
 */
int register_memory(FronthaulHandle fronthaul, MemRegInfo const* info, MemRegHandle* memreg);

/******************************************************************/ /**
 * \brief De-register memory zone
 *
 * \param memreg - Memory region handle to deregister
 *
 * \return \p 0 on success, Linux error code otherwise
 */
int unregister_memory(MemRegHandle memreg);

/******************************************************************/ /**
 * \brief Allocate DOCA CPU-GPU memory
 *
 * \param cuda_device - CUDA Device ID
 * \param size - Size of memory to allocate
 * \param gpu_ptr - GPU pointer
 * \param cpu_ptr - CPU pointer
 *
 * \return \p 0 on success, Linux error code otherwise
 */
// int allocate_doca_memory(int cuda_device, size_t size, void** gpu_ptr, void** cpu_ptr);

/** @} */ /* END MEMORY */

/**
 * \defgroup Tx Tx
 *
 * @{
 */

/******************************************************************/ /**
 * \brief Over The Air time: nanoseconds in PTP/atomic epoch
 *
 * \note Will rollover in year 2554
 */
using Ns = uint64_t;

/******************************************************************/ /**
 * \brief Message transmission window
 */
struct MsgSendWindow
{
    Ns tx_window_start;
    Ns tx_window_bfw_start;
    Ns tx_window_end;
};

/******************************************************************/ /**
 * \brief C-plane Section Common Header fields
 */
union CPlaneSectionCommonHdr
{
    oran_cmsg_sect0_common_hdr sect_0_common_hdr;
    oran_cmsg_sect1_common_hdr sect_1_common_hdr;
    oran_cmsg_sect3_common_hdr sect_3_common_hdr;
    oran_cmsg_sect5_common_hdr sect_5_common_hdr;
    // TODO implement Section Type 6 & 7
};

/******************************************************************/ /**
 * \brief C-plane Section fields
 */
struct CPlaneSectionExt11BundlesInfo
{
    union
    {
        oran_cmsg_sect_ext_type_11_disableBFWs_0_bfp_compressed_bundle_hdr disableBFWs_0_compressed;
        oran_cmsg_sect_ext_type_11_disableBFWs_0_bundle_uncompressed disableBFWs_0_uncompressed;
        oran_cmsg_sect_ext_type_11_disableBFWs_1_bundle disableBFWs_1;
    };

    uint8_t* bfwIQ;
};

static_assert(std::is_trivially_copyable_v<CPlaneSectionExt11BundlesInfo>);
static_assert(std::is_trivially_copyable_v<oran_cmsg_sect_ext_type_11_disableBFWs_0_bfp_compressed_bundle_hdr>);
static_assert(std::is_trivially_copyable_v<oran_cmsg_sect_ext_type_11_disableBFWs_0_bundle_uncompressed>);
static_assert(std::is_trivially_copyable_v<oran_cmsg_sect_ext_type_11_disableBFWs_1_bundle>);

/******************************************************************/ /**
 * \brief C-plane Section fields
 */
struct CPlaneSectionExt11Info
{
    oran_cmsg_sect_ext_type_11                          ext_hdr;
    oran_cmsg_sect_ext_type_11_disableBFWs_0_bfwCompHdr ext_comp_hdr;
    CPlaneSectionExt11BundlesInfo*                      bundles;
    uint16_t                                            numPrbBundles;
    uint16_t                                            numBundPrb;
    uint8_t                                             bundle_hdr_size;
    uint16_t                                            bfwIQ_size;
    uint8_t                                             bundle_size;
    bool                                                static_bfw;
    uint8_t*                                            bfwIQ; // Stores the pointer to the bfwIQ buffer with offset applied for this eAxC
    uint8_t                                             start_bundle_offset_in_bfw_buffer; // Due to numPrb fragmentation
};

struct CPlaneSectionExt4Info
{
    oran_cmsg_sect_ext_type_4 ext_hdr;
};

struct CPlaneSectionExt5Info
{
    oran_cmsg_sect_ext_type_5 ext_hdr;
};


/******************************************************************/ /**
 * \brief C-plane Section fields
 */
struct CPlaneSectionExtInfo
{
    oran_cmsg_ext_hdr sect_ext_common_hdr;
    union
    {
        CPlaneSectionExt4Info  ext_4;
        CPlaneSectionExt5Info  ext_5;
        CPlaneSectionExt11Info ext_11;
        // TODO implement other Section ext types
    };
};

static_assert(std::is_trivially_copyable_v<CPlaneSectionExtInfo>);
static_assert(std::is_trivially_copyable_v<oran_cmsg_ext_hdr>);
static_assert(std::is_trivially_copyable_v<CPlaneSectionExt4Info>);
static_assert(std::is_trivially_copyable_v<CPlaneSectionExt5Info>);
static_assert(std::is_trivially_copyable_v<CPlaneSectionExt11Info>);

/******************************************************************/ /**
 * \brief C-plane Section fields
 */
struct CPlaneSectionInfo
{
    union
    {
        oran_cmsg_sect0 sect_0;  //!< Section Type 0 (Unused RBs)
        oran_cmsg_sect1 sect_1;  //!< Section Type 1 (Most used)
        oran_cmsg_sect3 sect_3;  //!< Section Type 3 (PRACH and mixed numerology)
        oran_cmsg_sect5 sect_5;  //!< Section Type 5 (UE scheduling info)
        // TODO implement Section Type 6 & 7
    };

    bool                          csirs_of_multiplex_pdsch_csirs;  //!< CSI-RS multiplexed with PDSCH flag
    slot_command_api::prb_info_t* prb_info;                        //!< PRB allocation info

    CPlaneSectionExtInfo* ext4;   //!< Section Extension Type 4 (modulation compression params)
    CPlaneSectionExtInfo* ext5;   //!< Section Extension Type 5 (modulation compression additional params)
    CPlaneSectionExtInfo* ext11;  //!< Section Extension Type 11 (beamforming weights)
    // Section ID lookback index for CSI-RS compact signaling.
    //
    // Specifies the number of sections to look back in the section array to find
    // the reference section whose section ID should be reused for the current section.
    //
    // In CSI-RS transmissions with compact signaling, multiple sections corresponding
    // to different logical antenna ports but mapped to the same fronthaul flow must
    // share the same section ID. This field enables that ID sharing by indicating
    // which previous section's ID to reuse.
    //
    // Value semantics:
    //   0: Current section is assigned a new section ID (acts as reference section).
    //   N > 0: Current section reuses the section ID from the section that is N
    //          positions backward in the same flow's section array.
    //
    // NOTE: This is critical for O-RAN compliant CSI-RS compact signaling where section
    //       ID grouping reduces overhead while section extensions remain unique.
    uint8_t section_id_lookback_index; //!< Number of index lookback for section id 
};

/******************************************************************/ /**
 * \brief C-plane message descriptor
 */
struct CPlaneMsgSendInfo
{
    FlowHandle                     flow;                //!< eCPRI flow to send on
    uint16_t*                           nxt_section_id;      //!< Next section ID 
    //! Section-type specific fields
    //!
    //! \note all sections must be of the same type
    CPlaneSectionInfo* sections;  //!< Array of C-plane sections
    MsgSendWindow                  tx_window;           //!< TX timing window
    CPlaneSectionCommonHdr         section_common_hdr;  //!< Common section header fields
    uint8_t                        ap_idx;              //!< Antenna port index
    oran_pkt_dir                   data_direction;      //!< Data direction (DL/UL)
    bool                           hasSectionExt;       //!< true if sections have extensions
};

/******************************************************************/ /**
 * \brief For RU Emulator Standalone: Prepare C-plane messages for later use
 *
 * \param peer - Destination peer
 * \param info - Message descriptors
 * \param num_msgs - Number of messages to send (before fragmentation)
 * \param tx_request - Output TX request handle
 *
 * \return \p 0 on success, Linux error code otherwise
 *
 * \warning all C-plane messages must be of the same section type!
 * \warning C-plane messages must be in chronological order!
 */
int prepare_cplane(PeerHandle peer, CPlaneMsgSendInfo const* info, size_t num_msgs, TxRequestHandle* tx_request);

/******************************************************************/ /**
 * \brief Send C-plane messages.
 *
 * \param peer - Destination peer
 * \param info - Message descriptors
 * \param num_msgs - Number of messages to send (before fragmentation)
 *
 * \return Number of packets sent
 *
 * \warning all C-plane messages must be of the same section type!
 * \warning C-plane messages must be in chronological order!
 */
size_t send_cplane(PeerHandle peer, CPlaneMsgSendInfo const* info, size_t num_msgs);

/******************************************************************/ /**
 * \brief Send C-plane messages.
 *
 * \param peer - Destination peer
 * \param info - Message descriptors
 * \param num_msgs - Number of messages to send (before fragmentation)
 *
 * \return Number of packets sent
 *
 * \warning all C-plane messages must be of the same section type!
 * \warning C-plane messages must be in chronological order!
 */
size_t send_cplane_mmimo(PeerHandle peer, CPlaneMsgSendInfo const* info, size_t num_msgs);

/******************************************************************/ /**
 * \brief For FH Gen: Count number of C-Plane packets to be sent
 *
 * \param peer - Destination peer
 * \param info - Message descriptors
 * \param num_msgs - Number of messages to send (before fragmentation)
 *
 * \return \p 0 on success, Linux error code otherwise
 *
 */
size_t prepare_cplane_count_packets(PeerHandle peer, CPlaneMsgSendInfo const* info, size_t num_msgs);

/******************************************************************/ /**
 * \brief U-plane Section fields
 */
struct UPlaneSectionInfo
{
    uint16_t section_id;      //!< Section ID
    bool     rb;              //!< Resource block indicator (all RBs or subset)
    bool     sym_inc;         //!< Symbol number increment command
    uint16_t start_prbu;      //!< Starting PRB of user plane section
    uint16_t num_prbu;        //!< Number of PRBs (use 273 for all PRBs)
    void*    iq_data_buffer;  //!< User IQ data buffer
    uint16_t prb_size;        //!< PRB size in bytes
    bool     mod_comp_enable; //!< Enable modulation compression
};

/******************************************************************/ /**
 * \brief Callback function for U-plane TX completion
 */
using UPlaneTxCompleteCallback = void (*)(void* addr, void* opaque);

/******************************************************************/ /**
 * \brief U-plane Tx complete notification
 */
struct UPlaneTxCompleteNotification
{
    UPlaneTxCompleteCallback callback;     //!< Callback to invoke once U-plane message has been sent
    void*                    callback_arg; //!< Callback argument (opaque user data)
};

/******************************************************************/ /**
 * \brief U-plane message descriptor
 */
struct UPlaneMsgSendInfo
{
    FlowHandle        flow;            //!< eCPRI flow to send on
    MsgSendWindow     tx_window;       //!< TX timing window
    oran_umsg_iq_hdr  radio_app_hdr;   //!< ORAN radio application header
    UPlaneSectionInfo section_info;    //!< Section information
    uint16_t          eaxcid;          //!< eAxC ID
};

/******************************************************************/ /**
 * \brief Used to simulate malformatted eCPRI header
 */
struct EcpriHdrConfig
{
    struct EcpriHdrField
    {
        uint8_t enable;
        uint8_t value;
    };

    struct EcpriHdrField16
    {
        uint8_t  enable;
        uint16_t value;
    };

    EcpriHdrField   ecpriVersion;       // 4-bit field
    EcpriHdrField   ecpriReserved;      // 3-bit field
    EcpriHdrField   ecpriConcatenation; // 1-bit field
    EcpriHdrField   ecpriMessage;       // 8-bit field
    EcpriHdrField16 ecpriPayload;       // 16-bit field
    EcpriHdrField16 ecpriRtcid;         // 16-bit field (union with ecpriPcid)
    EcpriHdrField16 ecpriPcid;          // 16-bit field (union with ecpriRtcid)
    EcpriHdrField   ecpriSeqid;         // 8-bit field
    EcpriHdrField   ecpriEbit;          // 1-bit field
    EcpriHdrField   ecpriSubSeqid;      // 7-bit field

    // Ensure all fields are disabled and zero-initialised by default
    constexpr EcpriHdrConfig() noexcept
        : ecpriVersion{0, 0},
          ecpriReserved{0, 0},
          ecpriConcatenation{0, 0},
          ecpriMessage{0, 0},
          ecpriPayload{0, 0},
          ecpriRtcid{0, 0},
          ecpriPcid{0, 0},
          ecpriSeqid{0, 0},
          ecpriEbit{0, 0},
          ecpriSubSeqid{0, 0}
    {}

};

/******************************************************************/ /**
 * \brief U-plane multi-section message descriptor
 */
/**
 * Multi-section U-plane message descriptor
 *
 * Supports sending multiple sections in a single U-plane message,
 * used for advanced scheduling scenarios.
 */
struct UPlaneMsgMultiSectionSendInfo
{
    FlowHandle                                    flow;            //!< eCPRI flow to send on
    MsgSendWindow                                 tx_window;       //!< TX timing window
    oran_umsg_iq_hdr                              radio_app_hdr;   //!< ORAN radio application header
    UPlaneSectionInfo                             section_info;    //!< Primary section info (backward compatibility)
    std::array<UPlaneSectionInfo, kMaxSectionNum> section_infos;   //!< Array of section infos (multi-section support)
    uint16_t                                      section_num;     //!< Number of sections in this message
    uint16_t                                      eaxcid;          //!< eAxC ID
    const EcpriHdrConfig*                         ecpri_hdr_cfg{nullptr}; //!< Optional malformed eCPRI header config (for testing)
};

/******************************************************************/ /**
 * \brief Prepare U-plane messages to send later
 *
 * \param peer - Destination peer
 * \param info - Message descriptors
 * \param num_msgs - Number of messages to prepare (before fragmentation)
 * \param notification - Notification to generate once all messages have been sent
 * \param tx_request - Output TX request handle
 * \param txq_index - txq_index within the peer, optional, defaults to 0
 *
 * \return \p 0 on success, Linux error code otherwise
 *
 * \warning all U-plane messages must be of the same section type!
 * \warning U-plane messages must be in chronological order!
 */
int prepare_uplane(PeerHandle peer, UPlaneMsgSendInfo const* info, size_t num_msgs, UPlaneTxCompleteNotification notification, TxRequestHandle* tx_request, int txq_index = 0);


/******************************************************************/ /**
 * \brief Prepare U-plane messages to send later
 *
 * \param peer - Destination peer
 * \param info - Message descriptors
 * \param notification - Notification to generate once all messages have been sent
 * \param tx_request - Output TX request handle which is already pre-allocated
 * \param txq_index - txq_index within the peer, optional, defaults to 0
 *
 * \return \p 0 on success, Linux error code otherwise
 *
 * \warning all U-plane messages must be of the same section type!
 * \warning U-plane messages must be in chronological order!
 */
int prepare_uplane_with_preallocated_tx_request(PeerHandle peer, UPlaneMsgMultiSectionSendInfo const* info, UPlaneTxCompleteNotification notification, TxRequestHandle* tx_request, int txq_index = 0);

/******************************************************************/ /**
 * \brief For FH Gen: Count number of U-Plane packets to be sent
 *
 * \param peer - Destination peer
 * \param info - Message descriptors
 *
 * \return \p 0 on success, Linux error code otherwise
 *
 */
size_t prepare_uplane_count_packets(PeerHandle handle, UPlaneMsgMultiSectionSendInfo const* info);

/******************************************************************/ /**
 * \brief Preallocate mbufs for U-plane messages to send later
 *
 * \param peer - Destination peer
 * \param tx_request - Output TX request handle which is already pre-allocated
 * \param num_mbufs - Number of mbufs to allocate
 *
 * \return \p 0 on success, Linux error code otherwise
 *
 * \warning all U-plane messages must be of the same section type!
 * \warning U-plane messages must be in chronological order!
 */
int preallocate_mbufs(PeerHandle handle, TxRequestHandle* tx_request, int num_mbufs);

/******************************************************************/ /**
 * \brief Preallocate mbufs for U-plane messages to send later
 *
 * \param peer - Destination peer
 * \param tx_request - Output TX request handle which is already pre-allocated
 *
 * \return \p 0 on success, Linux error code otherwise
 *
 * \warning all U-plane messages must be of the same section type!
 * \warning U-plane messages must be in chronological order!
 */
int free_preallocated_mbufs(PeerHandle handle, TxRequestHandle* tx_request);

/******************************************************************/ /**
 * \brief Allocate tx_request
 *
 * \param peer - Destination peer
 * \param tx_request - Output TX request handle
 *
 * \return \p 0 on success, Linux error code otherwise
 */
int alloc_tx_request(PeerHandle peer, TxRequestHandle* output_handle);

/******************************************************************/ /**
 * \brief Free tx_request for reuse
 *
 * \param tx_request - TX request handle
 *
 * \return \p 0 on success, Linux error code otherwise
 */
int free_tx_request(TxRequestHandle tx_req_handle);

/******************************************************************/ /**
 * \brief Prepare and send U-plane messages
 *
 * \param peer - Destination peer
 * \param info - Message descriptors
 * \param num_msgs - Number of messages to send (before fragmentation)
 * \param notification - Notification to generate once all messages have been sent
 * \param txq - TXQ to use. If nullptr, the FH driver will choose one automatically
 * \param txq_index - txq_index within the peer, optional, defaults to 0
 *
 * \return Number of packets sent
 *
 * \warning all U-plane messages must be of the same section type!
 * \warning U-plane messages must be in chronological order!
 */
size_t prepare_and_send_uplane(PeerHandle peer, UPlaneMsgSendInfo const* info, size_t num_msgs, UPlaneTxCompleteNotification notification, TxqHandle txq = nullptr, int txq_index = 0);

/******************************************************************/ /**
 * \brief Send U-plane messages
 *
 * \param tx_request - TX request handle
 * \param txq - TXQ to use. If nullptr, the FH driver will choose one automatically
 *
 * \return Number of packets sent
 *
 * \note All resources allocated by \sa prepare_uplane() are implicitly freed
 *
 * \pre U-plane messages must have been prepared previously, \sa prepare_uplane()
 */
size_t send_uplane(TxRequestHandle tx_request, TxqHandle txq = nullptr);

/******************************************************************/ /**
 * \brief Send U-plane messages
 *
 * \param tx_request - TX request handle
 * \param txq - TXQ to use. If nullptr, the FH driver will choose one automatically
 * \param timing - Optional timing breakdown. If provided, returns complete send operation timing
 *
 * \return Number of packets sent
 *
 * \note All resources allocated by \sa prepare_uplane() are implicitly freed except tx_request
 *
 * \pre U-plane messages must have been prepared previously, \sa prepare_uplane()
 */
size_t send_uplane_without_freeing_tx_request(TxRequestHandle tx_request, TxqHandle txq = nullptr, TxqSendTiming* timing = nullptr);

/******************************************************************/ /**
 * \brief Poll for U-plane TX completions
 *
 * \param peer - peer to have its TX status checked
 *
 * \return \p 0 on success, Linux error code otherwise
 */
int poll_tx_complete(PeerHandle peer);

/** @} */ /* END TX API */

/**
 * \defgroup Rx Rx
 *
 * @{
 */

/******************************************************************/ /**
 * \brief Received message descriptor
 */
struct MsgReceiveInfo
{
    void*  buffer;         //!< Pointer to received message buffer
    size_t buffer_length;  //!< Length of received message in bytes
    void*  opaque;         //!< Opaque user data (mbuf pointer for DPDK)
    Ns     rx_timestamp;   //!< RX timestamp (nanoseconds)
};

/******************************************************************/ /**
 * \brief Order kernel helper struct for GPU RX synchronization
 *
 * Used by GPU kernels to synchronize and order received packets
 * from multiple flows before processing.
 */
struct rx_queue_sync
{
    uint32_t               status;                                //!< Sync status flag
    uint32_t               umsg_index;                            //!< U-plane message index
    uintptr_t              addr[RX_QUEUE_SYNC_LIST_ITEMS];        //!< Buffer addresses per item
    uint16_t               flow[RX_QUEUE_SYNC_LIST_ITEMS];        //!< Flow IDs per item
    uint64_t               rx_timestamp[RX_QUEUE_SYNC_LIST_ITEMS]; //!< RX timestamps per item
    MsgReceiveInfo*        umsg_info;                             //!< Pointer to message info array
    int                    umsg_num;                              //!< Number of messages
};

/******************************************************************/ /**
 * \brief Receive messages from a peer
 *
 * \param peer - Peer to receive from
 * \param info - Descriptor for each RX message
 * \param num_msgs - in-out parameter, on input: size of info array, on return: number of messages received
 *
 * \return \p 0 on success, Linux error code otherwise
 */
int receive(PeerHandle peer, MsgReceiveInfo* info, size_t* num_msgs, bool srs = false);

/******************************************************************/ /**
 * \brief Receive messages on a flow
 *
 * \param flow - Flow to receive on
 * \param info - Descriptor for each RX message
 * \param num_msgs - in-out parameter, on input: size of info array, on return: number of messages received
 *
 * \return \p 0 on success, Linux error code otherwise
 */
int receive_flow(FlowHandle flow, MsgReceiveInfo* info, size_t* num_msgs);

/******************************************************************/ /**
 * \brief Block until a desired number of messages has been received from a peer or timeout has been reached.
 *
 * \param peer - Peer to receive from
 * \param info - Descriptor for each RX message
 * \param num_msgs - In-out parameter, on input: size of info array, on return: number of messages received
 * \param timeout - Time by which this function must return
 *
 * \return \p 0 on success, Linux error code otherwise
 */
int receive_until(PeerHandle peer, MsgReceiveInfo* info, size_t* num_msgs, Ns timeout);

/******************************************************************/ /**
 * \brief Block until a desired number of messages has been received on a flow or timeout has been reached.
 *
 * \param flow - Flow to receive on
 * \param info - Descriptor for each RX message
 * \param num_msgs - In-out parameter, on input: size of info array, on return: number of messages received
 * \param timeout - Time by which this function must return
 *
 * \return \p 0 on success, Linux error code otherwise
 */
int receive_flow_until(FlowHandle flow, MsgReceiveInfo* info, size_t* num_msgs, Ns timeout);

/******************************************************************/ /**
 * \brief Free buffers after consuming RX messages
 *
 * \param info - Messages to free
 * \param num_msgs - Message count
 *
 * \return \p 0 on success, Linux error code otherwise
 */
int free_rx_messages(MsgReceiveInfo const* info, size_t num_msgs);

/** @} */ /* END RX API */

#define MAX_NUM_TX_REQ_UPLANE_GPU_COMM_PER_NIC API_MAX_NUM_CELLS  //!< Maximum TX requests per NIC for GPU comm

/**
 * TX request handle array for GPU communication (per cell)
 *
 * Manages an array of GPU comm TX request handles, one per cell,
 * for a specific NIC.
 */
typedef struct _TxRequestGpuPercell
{
    uint32_t               size;                                                 //!< Number of valid TX request handles
    std::string            nic_name;                                              //!< NIC name this belongs to
    TxRequestGpuCommHandle tx_v_per_nic[MAX_NUM_TX_REQ_UPLANE_GPU_COMM_PER_NIC]; //!< Array of TX request handles per cell
}TxRequestGpuPercell;


/**
 * \defgroup Buffer Thread-safe ring buffer
 *
 * API to manipulate a multi-producer, multi-consumer, fixed-size FIFO
 *
 * @{
 */

/******************************************************************/ /**
 * \brief Ring buffer information
 */
#define AERIAL_SOCKET_ID_ANY (-1)
struct RingBufferInfo
{
    const char* name;           //!< Name of the buffer
    size_t      count;          //!< Size of the buffer (will be aligned to a power of 2).
    int32_t     socket_id;      //!< socket identifier in case of NUMA. Use -1 if there is no NUMA constraint
    bool        multi_producer; //!< Multi-producer safe
    bool        multi_consumer; //!< Multi-consumer safe
};

/******************************************************************/ /**
 * \brief Create a ring buffer of given size
 *
 * \param fronthaul - Fronthaul driver handle
 * \param info - Ring buffer information
 * \param ring - Output ring buffer handle
 *
 * \return \p 0 on success, Linux error code otherwise
 */
int ring_create(FronthaulHandle fronthaul, RingBufferInfo const* info, RingBufferHandle* ring);

/******************************************************************/ /**
 * \brief Destroy a ring buffer
 *
 * \param ring - ring buffer handle
 *
 * \return \p 0 on success, Linux error code otherwise
 */
int ring_destroy(RingBufferHandle ring);

/******************************************************************/ /**
 * \brief Enqueue one object on a ring buffer
 *
 * \param ring - Ring buffer handle
 * \param obj - Object to enqueue
 *
 * \return \p 0 on success, Linux error code otherwise
 */
int ring_enqueue(RingBufferHandle ring, void* obj);

/******************************************************************/ /**
 * \brief For RU Emulator Standalone: Enqueue several tx_request cplane mbufs into a ring buffer, and free the tx_request
 *
 * \param ring - Ring buffer handle
 * \param txrequest - txrequest containing the mbufs
 * \param peer - peer handle used to access nic and call mempool_put for the tx_request
 * \param count - Number of objects to enqueue
 *
 * \return \p 0 on success, Linux error code otherwise
 */
int ring_enqueue_bulk_tx_request_cplane_mbufs(RingBufferHandle ring, TxRequestHandle txrequest, PeerHandle peer, size_t count);

/******************************************************************/ /**
 * \brief Enqueue several objects on a ring buffer
 *
 * \param ring - Ring buffer handle
 * \param objs - Objects to enqueue
 * \param count - Number of objects to enqueue
 *
 * \return The number of items enqueued, either 0 or \sa count
 */
size_t ring_enqueue_bulk(RingBufferHandle ring, void* const* objs, size_t count);

/******************************************************************/ /**
 * \brief Enqueue several objects on a ring buffer up to a maximum number
 *
 * \param ring - Ring buffer handle
 * \param objs - Objects to enqueue
 * \param count - Number of objects to enqueue
 *
 * \return The number of items enqueued
 */
size_t ring_enqueue_burst(RingBufferHandle ring, void* const* objs, size_t count);

/******************************************************************/ /**
 * \brief Dequeue one object from a ring buffer
 *
 * \param ring - Ring buffer handle
 * \param obj - Pointer to an object that will be filled.
 *
 * \return \p 0 on success, Linux error code otherwise
 */
int ring_dequeue(RingBufferHandle ring, void** obj);

/******************************************************************/ /**
 * \brief Dequeue several objects from a ring buffer
 *
 * \param ring - Ring buffer handle
 * \param objs - Pointer to a table of objects that will be filled.
 * \param count - Number of objects to dequeue
 *
 * \return The number of items dequeued, either 0 or \sa count
 */
size_t ring_dequeue_bulk(RingBufferHandle ring, void** objs, size_t count);

/******************************************************************/ /**
 * \brief For RU Emulator Standalone: dequeue several objects from a ring buffer up to a maximum number and properly set the mbufs as the MsgReceiveInfo types
 *
 * \param ring - Ring buffer handle
 * \param objs - Pointer to a table of objects that will be filled that will be temporarily filled by dequeue
 * \param info - Pointer to MsgReceiveInfo table, the size of the table should be the same as objs
 * \param count - Maximum number of objects to dequeue
 *
 * \return The number of items dequeued, either 0 or \sa count
 */
size_t ring_dequeue_burst_mbufs_payload_offset(RingBufferHandle handle, void** objs, MsgReceiveInfo* info, size_t count);

/******************************************************************/ /**
 * \brief Dequeue several objects from a ring buffer up to a maximum number
 *
 * \param ring - Ring buffer handle
 * \param objs - Pointer to a table of objects that will be filled.
 * \param count - Maximum number of objects to dequeue
 *
 * \return The number of items dequeued
 */
size_t ring_dequeue_burst(RingBufferHandle ring, void** objs, size_t count);

/******************************************************************/ /**
 * \brief Return the number of free entries in a ring buffer.
 *
 * \param ring - Ring buffer handle
 *
 * \return The number of free entries
 */
size_t ring_free_count(RingBufferHandle ring);

/******************************************************************/ /**
 * \brief Test if a ring buffer is full.
 *
 * \param ring - Ring buffer handle
 *
 * \return \p true if ring buffer is full, \p false otherwise
 */
bool ring_full(RingBufferHandle ring);

/******************************************************************/ /**
 * \brief Test if a ring buffer is empty.
 *
 * \param ring - Ring buffer handle
 *
 * \return \p true if ring buffer is empty, \p false otherwise
 */
bool ring_empty(RingBufferHandle ring);

/** @} */ /* END Thread-safe ring buffer */

/**
 * \defgroup Metrics Metrics
 *
 * API to generate metrics in fronthaul driver
 *
 * @{
 */

/******************************************************************/ /**
 * \brief Update all metrics exported by the FH driver
 *
 * \param fronthaul - Fronthaul driver handle
 *
 * \return \p 0 on success, Linux error code otherwise
 *
 * \warning This function should be used outside datapath
 */
int update_metrics(FronthaulHandle fronthaul);

/** @} */ /* END METRICS */

/**
 * \defgroup GPU-initiated communications GpuComm
 *
 * API to prepare and send packets with
 * GPU-initiated communications
 *
 * @{
 */

/*********************************************************************************/ /**
 * \brief Prepare U-plane messages for GPU-initiated communications
 *
 * \param peer - Destination peer
 * \param info - Message descriptors
 * \param num_msgs - Number of messages to prepare (before fragmentation)
 * \param wait_flag - Memory area to wait before triggering the send
 * \param wait_value - Wait flag value to wait for
 * \param output_handle - Output TX request handle
 * \param cell_start_time - start time for a given cell
 * \param symbol_duration - symbol time duration
 *
 * \return \p 0 on success, Linux error code otherwise
 *
 * \warning all U-plane messages must be of the same section type!
 * \warning U-plane messages must be in chronological order!
 */
int prepare_uplane_gpu_comm(PeerHandle handle, UPlaneMsgSendInfo const* info, size_t num_msgs, TxRequestGpuCommHandle* output_handle,
 std::chrono::nanoseconds cell_start_time,  std::chrono::nanoseconds symbol_duration,bool commViaCpu);

/*********************************************************************************/ /**
 * \brief Prepare U-plane messages for GPU-initiated communications
 *
 * \param handle - Handle to destination peer
 * \param info - Message descriptors
 * \param num_msgs - Number of messages to prepare (before fragmentation)
 * \param output_handle - Output TX request handle
 * \param cell_start_time - start time for a given cell
 * \param symbol_duration - symbol time duration
 *
 * \return \p 0 on success, Linux error code otherwise
 *
 * \warning all U-plane messages must be of the same section type!
 * \warning U-plane messages must be in chronological order!
 */
int prepare_uplane_gpu_comm_v2(PeerHandle handle, UPlaneMsgSendInfo const* info, size_t num_msgs, TxRequestGpuCommHandle* output_handle,
 std::chrono::nanoseconds cell_start_time,  std::chrono::nanoseconds symbol_duration);

/*********************************************************************************/ /**
 * \brief Send U-plane messages with GPU-initiated communications
 *
 * \param handle - NIC used to send packets
 * \param txreq_v - List of TX requests to handle
 *
 * \return \p 0 on success, Linux error code otherwise
 *
 * \warning all U-plane messages must be of the same section type!
 * \warning U-plane messages must be in chronological order!
 */
int send_uplane_gpu_comm(NicHandle handle, TxRequestGpuPercell *pTxRequestGpuPercell, PreparePRBInfo &prb_info);

/*********************************************************************************/ /**
 * \brief Update tx metrics for GPU-initiated communications
 *
 * \param handle - Handle to destination peer
 * \param tx_request_handle - Handle to TxRequest
 *
 * \return \p 0 on success, Linux error code otherwise
 */
int gpu_comm_update_tx_metrics(PeerHandle handle, TxRequestGpuCommHandle tx_request_handle);

int ring_cpu_doorbell(NicHandle handle, TxRequestGpuPercell *pTxRequestGpuPercell, PreparePRBInfo &prb_info, PacketTimingInfo &packet_timing_info);

int set_TriggerTs_GpuComm(NicHandle handle,uint32_t slot_idx,uint64_t trigger_ts);

int trigger_cqe_tracer_cb(NicHandle handle, TxRequestGpuPercell *pTxRequestGpuPercell);

int print_max_delays(NicHandle handle);

}; // namespace aerial_fh

#endif //ifndef AERIAL_FH_DRIVER_API__
