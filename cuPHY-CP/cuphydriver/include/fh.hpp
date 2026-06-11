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

#ifndef FH_CLASS_H
#define FH_CLASS_H

#include <atomic>
#include <unordered_map>
#include "cuphydriver_api.hpp"
#include "constant.hpp"
#include "aerial-fh-driver/api.hpp"
#include "nvlog.hpp"
#include "time.hpp"
#include "gpudevice.hpp"
#include "memfoot.hpp"
#include "cuphy.hpp"
#include "task_instrumentation_nested.hpp"

using namespace aerial_fh;

typedef uint64_t             peer_id_t;                                ///< Unique identifier for fronthaul peer (RU/O-RU)
typedef PeerHandle           fhproxy_peer;                             ///< Handle to fronthaul peer connection
typedef FlowHandle           fhproxy_flow;                             ///< Handle to fronthaul flow (per eAxC or channel)
typedef CPlaneMsgSendInfo    fhproxy_cmsg;                             ///< C-plane message send information structure
typedef CPlaneSectionInfo    fhproxy_cmsg_section;                     ///< C-plane section information (PRB allocation)
typedef CPlaneSectionExtInfo fhproxy_cmsg_section_ext;                 ///< C-plane section extension information
typedef CPlaneSectionExt11BundlesInfo fhproxy_cmsg_section_ext_11_bundle_info; ///< C-plane section extension type 11 bundle information (for beamforming weights)

typedef MsgReceiveInfo       fhproxy_umsg_rx;                          ///< U-plane message receive information structure
typedef UPlaneMsgSendInfo    fhproxy_umsg_tx;                          ///< U-plane message send information structure

/**
 * @brief PRB size in bytes for different BFP compression bit widths
 *
 * Calculation: prb_size = ceil((bit_width * num_subcarriers_prb * 2 + num_bits_exponent) / 8)
 * - bit_width: {9, 14, or 16 bits}
 * - num_subcarriers_prb: 12 subcarriers per PRB
 * - Factor of 2: I and Q samples
 * - num_bits_exponent: 4 bits for BFP compression (not present in uncompressed mode)
 */
enum fh_peer_compression_prb_size
{
    BITWIDTH_9_BITS         = 28,                                      ///< PRB size for 9-bit BFP compression (28 bytes)
    BITWIDTH_14_BITS        = 43,                                      ///< PRB size for 14-bit BFP compression (43 bytes)
    BFP_NO_COMPRESSION_BITWIDTH = 48                                   ///< PRB size for uncompressed 16-bit IQ data (48 bytes)
};

/**
 * @brief Message information type flags for beamforming weight handling
 */
enum fh_msg_info_type
{
    NON_BFW_MSG_INFO = 0,                                              ///< Regular message without beamforming weights
    BFW_MSG_INFO = 1,                                                  ///< Message contains beamforming weights
    NUM_MSG_INFO_TYPES                                                 ///< Total number of message info types
}; 

/**
 * @brief C-plane send error type codes
 *
 * Error codes returned by C-plane transmission functions to indicate
 * timing violations or functional errors.
 */
typedef enum fh_send_cplane_error_type
{
    SEND_CPLANE_NO_ERROR=0,                                            ///< No error, C-plane sent successfully
    SEND_DL_CPLANE_TIMING_ERROR,                                       ///< Downlink C-plane timing constraint violated
    SEND_UL_CPLANE_TIMING_ERROR,                                       ///< Uplink C-plane timing constraint violated
    SEND_CPLANE_FUNC_ERROR                                             ///< Functional error during C-plane send
}fh_send_cplane_error_type_t;

/**
 * @brief U-plane receive message container
 *
 * Holds array of received U-plane message information structures
 * for uplink data from RU/O-RU.
 */
struct umsg_fh_rx
{
    fhproxy_umsg_rx* umsg_info;                                        ///< Array of U-plane receive message info structures
    int              num;                                              ///< Number of messages in the array
};

/**
 * @brief U-plane transmit message container
 *
 * Holds array of U-plane message send information structures for
 * downlink data transmission to RU/O-RU, including both CPU and GPU
 * transmission request handles.
 */
struct umsg_fh_tx_msg
{
    fhproxy_umsg_tx* umsg_info_symbol_antenna;                         ///< Array of U-plane TX message info (per symbol per antenna)
    int              num;                                              ///< Number of messages in the array
    TxRequestHandle  txrq;                                             ///< CPU-based transmission request handle
    TxRequestGpuCommHandle  txrq_gpu;                                  ///< GPU-direct transmission request handle (for GPU communication)
};

/**
 * @brief Uplink receive packet ordering state
 *
 * Manages synchronization and ordering of received uplink U-plane packets
 * with GPU packet ordering kernel. Tracks ordering completion status and
 * maintains message buffers for received data.
 */
struct rx_order_t {
    std::unique_ptr<host_buf> sync_buffer;                             ///< Host buffer for synchronization with GPU ordering kernel
    struct rx_queue_sync*     sync_list;                               ///< Array of sync structures for packet ordering
    int                       sync_item;                               ///< Current sync item index
    struct gpinned_buffer*    sync_ready_list_gdr;                     ///< GPU Direct RDMA buffer for ready flag synchronization
    std::unique_ptr<host_buf> last_ordered_h;                          ///< Host buffer tracking last ordered packet index
    int                       last_ufree;                              ///< Last freed U-plane message index
    std::array<struct umsg_fh_rx, RX_QUEUE_SYNC_LIST_ITEMS> umsg_rx_list; ///< Array of received U-plane message containers
    int umsg_rx_index;                                                 ///< Current U-plane RX message array index
    std::unique_ptr<struct gpinned_buffer> flush_gmem;                 ///< Pinned GPU memory for cache flushing operations
};

/**
 * @brief Fronthaul flow configuration
 *
 * Associates a flow handle with its corresponding eAxC (extended Antenna-Carrier) ID.
 * Each flow represents a logical stream for a specific antenna-carrier combination.
 */
struct fh_flow
{
    FlowHandle handle;                                                 ///< Fronthaul library flow handle
    uint16_t eAxC_id;                                                  ///< Extended Antenna-Carrier identifier (eAxC) for this flow
};

/**
 * @brief PRB allocation information
 *
 * Describes the Physical Resource Block allocation in frequency (PRBs)
 * and time (OFDM symbols) domains for a specific channel section.
 */
struct fh_prb_info
{
    int startPrb;                                                      ///< Starting PRB index (frequency domain)
    int numPrb;                                                        ///< Number of allocated PRBs
    int startSym;                                                      ///< Starting OFDM symbol index (time domain)
    int numSym;                                                        ///< Number of OFDM symbols
    int sectionId;                                                     ///< Section identifier in C-plane message
};

using flows_per_channel_t = std::array<std::vector<fh_flow>, slot_command_api::channel_type::CHANNEL_MAX>; ///< Array of flow vectors per channel type (PDSCH, PUSCH, PRACH, etc.)

/**
 * @brief Fronthaul peer (RU/O-RU) connection state
 *
 * Manages states of a single fronthaul peer such as network configuration,
 * compression settings, flow mappings, and receive packet ordering state.
 */
struct fh_peer_t
{
    public:
        /**
         * @brief Construct a fronthaul peer
         *
         * @param _peer_id       - Unique peer identifier
         * @param _peer          - Peer handle from fronthaul library
         * @param _peer_info     - Peer configuration information
         * @param _dl_comp_meth  - Downlink compression method
         * @param _dl_bit_width  - Downlink compression bit width
         */
        fh_peer_t(peer_id_t _peer_id, PeerHandle _peer, PeerInfo _peer_info, UserDataCompressionMethod _dl_comp_meth, uint8_t _dl_bit_width)
                :
                peer_id(_peer_id), peer(_peer), peer_info(_peer_info), dl_comp_meth(_dl_comp_meth), dl_bit_width(_dl_bit_width)
        {
        }

        ~fh_peer_t() {}

        /**
         * @brief Get PRB size in bytes for current compression settings
         *
         * @return PRB size in bytes based on configured bit width
         */
        uint16_t getCompressionPrbSize() const
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

        peer_id_t peer_id;                                             ///< Unique peer identifier
        PeerInfo peer_info;                                            ///< Peer configuration (Ethernet addresses, VLAN, etc.)
        UserDataCompressionMethod dl_comp_meth;                        ///< Downlink compression method (BFP, modulation, none)
        uint8_t dl_bit_width;                                          ///< Downlink compression bit width (9, 14, or 16 bits)
        PeerHandle peer;                                               ///< Fronthaul library peer handle
        flows_per_channel_t cplane_flows;                              ///< C-plane flows organized by channel type
        flows_per_channel_t uplane_flows;                              ///< U-plane flows organized by channel type
        std::unordered_map<uint16_t, aerial_fh::FlowHandle> eAxC_ids_unique_cplane; ///< Map of unique C-plane eAxC IDs to flow handles
        std::unordered_map<uint16_t, aerial_fh::FlowHandle> eAxC_ids_unique_uplane; ///< Map of unique U-plane eAxC IDs to flow handles
        struct rx_order_t rx_order_items;                              ///< Uplink receive packet ordering state
};

/**
 * @brief Beamforming weight information tuple
 *
 * Stores static beamforming weights and a flag indicating whether they have been
 * sent to the C-plane/O-RU. The flag starts as false and is set to true after
 * the first transmission to avoid redundant sends of the same static weights.
 */
using BeamInfoTuple = std::tuple<bool, std::unique_ptr<uint8_t[]>>;       ///< Tuple: (sent_flag, beamforming_weight_data)
using StaticBfwMap = std::unordered_map<uint16_t, BeamInfoTuple>;         ///< Map from beam ID to beam weight information
using FhStaticBfwStorage = std::unordered_map<uint16_t, StaticBfwMap>;    ///< Map from cell ID to static beamforming weight storage

/**
 * @brief Parameters for dynamic section extension type 11 (beamforming weights)
 *
 * Contains all configuration and state needed to construct C-plane section extension
 * type 11, which carries beamforming weights for specific PRBs and antenna elements.
 * This structure supports both static and dynamic beamforming weight configurations.
 */
struct DynamicSectionExt11Params final {
    slot_command_api::prb_info_t& prb_info;                            ///< Reference to PRB allocation information (input/output)
    fhproxy_cmsg_section& section_info;                                ///< Reference to C-plane section info (output)
    std::array<fhproxy_cmsg_section_ext,MAX_CPLANE_SECTIONS_EXT_PER_SLOT> &section_ext_infos;              ///< Array of section extension structures (output)
    std::array<fhproxy_cmsg_section_ext_11_bundle_info,MAX_CPLANE_EXT_11_BUNDLES_PER_SLOT> &section_ext_11_bundle_infos; ///< Array of section ext 11 bundle info (output)
    size_t& section_ext_index;                                         ///< Current section extension array index (input/output)
    size_t& section_ext_11_bundle_index;                               ///< Current section ext 11 bundle array index (input/output)

    int L_TRX{0};                                                      ///< Number of TRX antenna elements
    int bfwIQBitwidth{0};                                              ///< Beamforming weight IQ bit width
    int disableBFWs{0};                                                ///< Flag to disable beamforming weights (1=disabled, 0=enabled)
    int RAD{0};                                                        ///< Resource Allocation Dimension (frequency or time grouping)
    uint8_t** bfw_header{nullptr};                                     ///< Array of beamforming weight header pointers

    int active_ap_idx{0};                                              ///< Active antenna port index (0-based)
    int actual_ap_idx{0};                                              ///< Actual antenna port index (physical port)

    int symbol{0};                                                     ///< OFDM symbol index
    int startPrbc{0};                                                  ///< Starting PRB index in compressed format
    int numPrbc{0};                                                    ///< Number of PRBs in compressed format

    uint16_t& dyn_beam_id;                                             ///< Dynamic beam ID for this section (input/output)
    uint16_t* bfw_beam_id{nullptr};                                    ///< Array of beam IDs per PRB
    bool* bfw_seen{nullptr};                                           ///< Array tracking which beams have been seen/processed
    uint16_t slot_dyn_beam_id_offset{0};                               ///< Dynamic beam ID offset for current slot

    int start_bundle_offset_in_bfw_buffer{0};                          ///< Starting offset for bundle split in beamforming weight buffer

    DynamicSectionExt11Params& operator=(const DynamicSectionExt11Params&) = delete; ///< Deleted copy assignment (struct contains reference members)
};

/**
 * @brief CSI-RS section identifier information
 *
 * Maps CSI-RS section IDs to their corresponding PRB and symbol allocations.
 * Used for tracking CSI-RS resource configuration in C-plane messages.
 */
struct csirs_section_id_info_t {
    int csirs_section_id;                                              ///< CSI-RS section identifier
    uint16_t start_prb;                                                ///< Starting PRB index for this CSI-RS resource
    uint16_t num_prb;                                                  ///< Number of PRBs allocated for this CSI-RS resource
    uint16_t symbol;                                                   ///< OFDM symbol index for this CSI-RS resource
    uint16_t section_id_lookback_index;                                ///< Section ID lookback index for CSI-RS compact signaling
};

struct cplane_buffer_t {
    
    // fhproxy_cmsg message_infos_[fh_msg_info_type::NUM_MSG_INFO_TYPES][MAX_CPLANE_MSGS_PER_SLOT]
    std::array<std::array<fhproxy_cmsg, MAX_CPLANE_MSGS_PER_SLOT>, fh_msg_info_type::NUM_MSG_INFO_TYPES> message_infos_{}; 

    // fhproxy_cmsg_section section_infos_[fh_msg_info_type::NUM_MSG_INFO_TYPES][MAX_AP_PER_SLOT_SRS][MAX_CPLANE_SECTIONS_PER_SLOT_PER_AP];
    std::array<std::array<std::array<fhproxy_cmsg_section, MAX_CPLANE_SECTIONS_PER_SLOT_PER_AP>, MAX_AP_PER_SLOT_SRS>,fh_msg_info_type::NUM_MSG_INFO_TYPES> section_infos_{};

    // uint16_t section_id_per_ant_[MAX_AP_PER_SLOT_SRS];
    std::array<uint16_t,MAX_AP_PER_SLOT_SRS> section_id_per_ant_{};

    // uint16_t start_section_id_srs_per_ant_[MAX_AP_PER_SLOT_SRS];
    std::array<uint16_t,MAX_AP_PER_SLOT_SRS> start_section_id_srs_per_ant_{};

    // uint16_t start_section_id_prach_per_ant_[MAX_AP_PER_SLOT_SRS];
    std::array<uint16_t,MAX_AP_PER_SLOT_SRS> start_section_id_prach_per_ant_{};

    // uint16_t message_index_[fh_msg_info_type::NUM_MSG_INFO_TYPES]; Indexes into message_infos_[][] array
    std::array<uint16_t,fh_msg_info_type::NUM_MSG_INFO_TYPES> message_index_{};
    static_assert(MAX_CPLANE_MSGS_PER_SLOT <= std::numeric_limits<decltype(message_index_)::value_type>::max()); 

    // fhproxy_cmsg_section_ext section_ext_infos_[MAX_CPLANE_SECTIONS_EXT_PER_SLOT]; 
    std::array<fhproxy_cmsg_section_ext, MAX_CPLANE_SECTIONS_EXT_PER_SLOT> section_ext_infos_{};

    // fhproxy_cmsg_section_ext_11_bundle_info section_ext_11_bundle_infos_[MAX_CPLANE_EXT_11_BUNDLES_PER_SLOT];
    std::array<fhproxy_cmsg_section_ext_11_bundle_info, MAX_CPLANE_EXT_11_BUNDLES_PER_SLOT> section_ext_11_bundle_infos_{};
}; 


/**
 * @brief Fronthaul proxy manager class
 *
 * Manages all fronthaul operations including:
 * - NIC (Network Interface Card) registration and management
 * - Peer (RU/O-RU) registration and configuration
 * - Flow registration for C-plane and U-plane
 * - C-plane message construction and transmission
 * - U-plane packet preparation and transmission
 * - Uplink packet reception and ordering
 * - Beamforming weight management (static and dynamic)
 * - Compression configuration and PRB calculations
 * - Fronthaul metrics tracking
 */
class FhProxy {
public:
    /**
     * @brief Construct fronthaul proxy manager
     *
     * @param _pdh     - Physical layer driver handle
     * @param ctx_cfg  - Context configuration including beamforming and NIC settings
     */
    FhProxy(phydriver_handle _pdh,const context_config& ctx_cfg);
    ~FhProxy();

    phydriver_handle getPhyDriverHandler(void) const;                  ///< Get physical layer driver handle
    FronthaulHandle  getFhInstance(void) const;                        ///< Get fronthaul library instance handle

    /**
     * @brief Register a Network Interface Card for fronthaul communication
     *
     * @param config - NIC configuration including device name and parameters
     * @param gpu_id - GPU device ID to associate with this NIC
     * @return 0 on success, negative error code on failure
     */
    int                      registerNic(struct nic_cfg config, int gpu_id);
    
    /**
     * @brief Check if a NIC is already registered
     *
     * @param nic_name - NIC device name to check
     * @return true if NIC exists, false otherwise
     */
    bool                     checkIfNicExists(std::string nic_name);
    
    std::vector<std::string> getNicList();                             ///< Get list of all registered NIC names
    
    /**
     * @brief Register a fronthaul peer (RU/O-RU)
     *
     * @param cell_id               - Cell identifier
     * @param peer_id               - Output peer identifier (assigned by function)
     * @param src_eth_addr          - Source Ethernet MAC address (DU)
     * @param dst_eth_addr          - Destination Ethernet MAC address (RU)
     * @param vlan_tci              - VLAN Tag Control Information
     * @param txq_count_uplane      - Number of U-plane TX queues
     * @param dl_comp_meth          - Downlink compression method
     * @param dl_bit_width          - Downlink compression bit width
     * @param gpu_id                - GPU device ID for packet processing
     * @param nic_name              - NIC device name to use
     * @param doca_rxq_info         - DOCA RX queue information for uplink
     * @param doca_rxq_info_srs     - DOCA RX queue information for SRS
     * @param eAxC_list_ul          - List of uplink eAxC IDs
     * @param eAxC_list_srs         - List of SRS eAxC IDs
     * @param eAxC_list_dl          - List of downlink eAxC IDs
     * @param max_num_prbs_per_symbol - Maximum PRBs per symbol (default: ORAN_MAX_PRB_X_SLOT)
     * @return 0 on success, negative error code on failure
     */
    int registerPeer(
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
        uint16_t                       max_num_prbs_per_symbol = ORAN_MAX_PRB_X_SLOT);
    
    /**
     * @brief Remove a registered fronthaul peer
     *
     * @param peer_id - Peer identifier to remove
     * @return 0 on success, negative error code on failure
     */
    int                 removePeer(peer_id_t peer_id);
    
    /**
     * @brief Register a flow for a specific eAxC ID and channel type
     *
     * @param peer_id  - Peer identifier
     * @param eAxC_id  - Extended Antenna-Carrier identifier (eAxC)
     * @param vlan_tci - VLAN Tag Control Information
     * @param channel  - Channel type (PDSCH, PUSCH, PRACH, etc.)
     * @return 0 on success, negative error code on failure
     */
    int                 registerFlow(peer_id_t peer_id, uint16_t eAxC_id, uint16_t vlan_tci, slot_command_api::channel_type channel, const std::array<uint8_t, 6>* dst_mac_override = nullptr);
    
    /**
     * @brief Store DBT (Digital Beamforming Table) PDU
     *
     * @param phy_cell_id - Physical cell identifier
     * @param data_buf    - Buffer containing DBT PDU data
     * @return 0 on success, negative error code on failure
     */
    [[nodiscard]] int                 storeDBTPdu(uint16_t phy_cell_id, void* data_buf);
    
    /**
     * @brief Get flag indicating if static beamforming weights have been sent
     *
     * @param cell_id - Cell identifier
     * @param beamIdx - Beam index
     * @return 1 if weights sent, 0 if not sent, negative on error
     */
    int                 getBeamWeightsSentFlag(uint16_t cell_id, uint16_t beamIdx);
    
    /**
     * @brief Mark static beamforming weights as sent for a beam
     *
     * @param cell_id - Cell identifier
     * @param beamIdx - Beam index
     * @return 0 on success, negative error code on failure
     */
    int                 setBeamWeightsSentFlag(uint16_t cell_id, uint16_t beamIdx);
    
    /**
     * @brief Check if static beamforming weights are configured for a cell
     *
     * @param cell_id - Cell identifier
     * @return 1 if configured, 0 if not configured
     */
    int                 staticBFWConfigured(uint16_t cell_id);
    
    /**
     * @brief Reset DBT storage for a cell (clear all stored beamforming weights)
     *
     * @param cell_id - Cell identifier
     * @return 0 on success, negative error code on failure
     */
    int                 resetDBTStorage(uint16_t cell_id);
    
    /**
     * @brief Update peer RX metrics (received packets/bytes)
     *
     * @param peer_id    - Peer identifier
     * @param rx_packets - Number of received packets
     * @param rx_bytes   - Number of received bytes
     * @return 0 on success, negative error code on failure
     */
    int                 update_peer_rx_metrics(peer_id_t peer_id, size_t rx_packets, size_t rx_bytes);
    
    /**
     * @brief Update peer TX metrics (transmitted packets/bytes)
     *
     * @param peer_id    - Peer identifier
     * @param tx_packets - Number of transmitted packets
     * @param tx_bytes   - Number of transmitted bytes
     * @return 0 on success, negative error code on failure
     */
    int                 update_peer_tx_metrics(peer_id_t peer_id, size_t tx_packets, size_t tx_bytes);
    
    /**
     * @brief Update peer compression configuration
     *
     * @param peer_id      - Peer identifier
     * @param dl_comp_meth - New downlink compression method
     * @param dl_bit_width - New downlink compression bit width
     * @return 0 on success, negative error code on failure
     */
    int                 updatePeer(peer_id_t peer_id, enum UserDataCompressionMethod dl_comp_meth, uint8_t dl_bit_width);
    
    /**
     * @brief Update peer network configuration
     *
     * @param peer_id       - Peer identifier
     * @param dst_eth_addr  - New destination Ethernet MAC address
     * @param vlan_tci      - New VLAN Tag Control Information
     * @param eAxC_list_ul  - New uplink eAxC ID list
     * @param eAxC_list_srs - New SRS eAxC ID list
     * @return 0 on success, negative error code on failure
     */
    int                 updatePeer(peer_id_t peer_id, std::array<uint8_t, 6> dst_eth_addr, uint16_t vlan_tci,std::vector<uint16_t>& eAxC_list_ul,std::vector<uint16_t>& eAxC_list_srs);
    
    /**
     * @brief Update maximum PRBs per symbol for a peer
     *
     * @param peer_id                - Peer identifier
     * @param max_num_prbs_per_symbol - New maximum PRBs per symbol
     * @return 0 on success, negative error code on failure
     */
    int                 update_peer_max_num_prbs_per_symbol(peer_id_t               peer_id,uint16_t max_num_prbs_per_symbol);    

    /**
     * @brief Check C-plane timing constraints before transmission
     * The timing threshold is controlled by the sendCPlane_timing_error_th_ns parameter in the
     * yaml configuration file. The default value of 0 disables the timing check.
     *
     * @param start_tx_time      - Scheduled transmission time
     * @param start_ch_task_time - Channel task start time
     * @param direction          - Direction (uplink or downlink)
     * @return 0 if timing OK, error code if timing violation detected
     */
    int sendCPlane_timingCheck(t_ns start_tx_time, t_ns start_ch_task_time, int direction);
    
    /**
     * @brief Fill dynamic section extension type 11 parameters (beamforming weights)
     *
     * @param slot_indication - ORAN slot indication with timing information
     * @param params          - Section extension 11 parameters to populate
     */
    void fill_dynamic_section_ext11(const slot_command_api::oran_slot_ind& slot_indication, DynamicSectionExt11Params& params);

    /**
     * @brief Send C-plane message for a slot
     *
     * Constructs and transmits ORAN C-plane message containing resource allocation,
     * beamforming weights, and timing information for the specified slot.
     *
     * @param cell_id              - Cell identifier
     * @param ru                   - RU type
     * @param peer_id              - Peer identifier
     * @param dl_comp_meth         - Downlink compression method
     * @param start_tx_time        - Scheduled transmission time
     * @param tx_cell_start_ofs_ns - Cell-specific TX time offset to spread C-plane transmission to reduce the load on the NIC (nanoseconds)
     * @param direction            - Packet direction (uplink or downlink)
     * @param slot_indication      - ORAN slot indication with timing
     * @param slot_info            - Slot configuration (PRB allocations, channels)
     * @param time_offset          - Time offset for C-plane timing (ORAN specific)
     * @param dyn_beam_id_offset   - Dynamic beam ID offset for this slot
     * @param frame_structure      - Frame structure (FDD/TDD)
     * @param cp_length            - Cyclic prefix length
     * @param bfw_header           - Array of beamforming weight headers
     * @param start_ch_task_time   - Channel task start time (for timing validation)
     * @param prevSlotBfwCompStatus - Previous slot beamforming weight completion status
     * @param ti_info              - Task instrumentation info for profiling
     * @return 0 on success, negative error code on failure
     */
    int prepareCPlaneInfo(
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
        ti_subtask_info &ti_info);
    
    /**
     * @brief Prepare MMIMO MBUF C-Plane packets & queue into the NIC
     *
     * Constructs and transmits ORAN C-plane MBUF containing resource allocation,
     * beamforming weights, and timing information for the specified slot.
     *
     * @param is_bfw - Boolean flag denoting whether to invoke BFW packet processing or non-BFW packet processing
     * @param cell_id              - Cell identifier
     * @param peer_id              - Peer identifier
     * @param direction            - Packet direction (uplink or downlink)
     * @param ti_info              - Task instrumentation info for profiling
     * @return 0 on success, negative error code on failure
     */
    int sendCPlaneMMIMO(
        bool is_bfw,
        uint32_t cell_id,
        peer_id_t peer_id,
        oran_pkt_dir direction,
        ti_subtask_info &ti_info);
    
    /**
     * @brief Prepare U-plane packets for downlink transmission
     *
     * Prepares ORAN U-plane packets from DL buffer data, including compression
     * and packet assembly on GPU or CPU.
     *
     * @param ru                    - RU type
     * @param peer_id               - Peer identifier
     * @param dl_stream             - CUDA stream for GPU operations
     * @param start_tx_time         - Scheduled transmission time
     * @param slot_indication       - ORAN slot indication with timing
     * @param slot_info             - Slot configuration (channels, PRB allocations)
     * @param umsg_tx_list          - Output U-plane TX message container
     * @param size                  - Buffer size
     * @param mod_comp_prm          - Modulation compression parameters
     * @param mod_comp_config_temp  - Temporary compression configuration
     * @param cb_obj                - Callback object pointer
     * @param symbol_duration       - OFDM symbol duration
     * @param batchedMemcpyHelper   - Helper for batched memory copy operations
     * @return 0 on success, negative error code on failure
     */
    int prepareUPlanePackets(
            ru_type ru,
            peer_id_t peer_id, cudaStream_t dl_stream, t_ns start_tx_time,
            const slot_command_api::oran_slot_ind &slot_indication,
            const slot_command_api::slot_info_t &slot_info,
            struct umsg_fh_tx_msg& umsg_tx_list,
            size_t size, mod_compression_params* mod_comp_prm, mod_compression_params* mod_comp_config_temp, void * cb_obj,  t_ns symbol_duration,
            cuphyBatchedMemcpyHelper& batchedMemcpyHelper);
    
    /**
     * @brief Update TX metrics for GPU direct communication
     *
     * @param peer_id      - Peer identifier
     * @param umsg_tx_list - U-plane TX message container with packet info
     */
    void                UpdateTxMetricsGpuComm(peer_id_t peer_id, struct umsg_fh_tx_msg& umsg_tx_list);
    
    /**
     * @brief Send U-plane packets (CPU path)
     *
     * @param peer_id      - Peer identifier
     * @param umsg_tx_list - U-plane TX message container
     * @return 0 on success, negative error code on failure
     */
    int                 UserPlaneSendPackets(peer_id_t peer_id, struct umsg_fh_tx_msg& umsg_tx_list);
    
    /**
     * @brief Receive U-plane packets for uplink
     *
     * @param peer_id - Peer identifier
     * @return Number of packets received, negative on error
     */
    int                 UserPlaneReceivePackets(peer_id_t peer_id);
    
    /**
     * @brief Free received U-plane message buffers
     *
     * @param peer_id - Peer identifier
     * @return 0 on success, negative error code on failure
     */
    int                 UserPlaneFreeMsg(peer_id_t peer_id);
    
    /**
     * @brief Check for received U-plane messages
     *
     * @param peer_id - Peer identifier
     * @return Number of messages available, negative on error
     */
    int                 UserPlaneCheckMsg(peer_id_t peer_id);
    
    /**
     * @brief Receive U-plane packets (CPU path)
     *
     * @param index    - Peer index
     * @param info     - Output array of received message info
     * @param num_msgs - Output number of received messages
     * @return 0 on success, negative error code on failure
     */
    int                 UserPlaneReceivePacketsCPU(int index, MsgReceiveInfo* info, size_t& num_msgs);
    
    /**
     * @brief Free U-plane packets (CPU path)
     *
     * @param info     - Array of message info to free
     * @param num_msgs - Number of messages to free
     * @return 0 on success, negative error code on failure
     */
    int                 UserPlaneFreePacketsCPU(MsgReceiveInfo* info, size_t num_msgs);

    /**
     * @brief Send U-plane packets (GPU direct communication path)
     *
     * @param pTxRequestGpuPercell - Per-cell GPU TX request structure
     * @param prb_info             - PRB information for packet assembly
     * @return 0 on success, negative error code on failure
     */
    int                 UserPlaneSendPacketsGpuComm(TxRequestGpuPercell *pTxRequestGpuPercell, PreparePRBInfo &prb_info);
    
    /**
     * @brief Ring CPU doorbell to trigger GPU direct transmission
     *
     * @param pTxRequestGpuPercell - Per-cell GPU TX request structure
     * @param prb_info             - PRB information
     * @param packet_timing_info   - Packet timing information
     * @return 0 on success, negative error code on failure
     */
    int                 RingCPUDoorbell(TxRequestGpuPercell *pTxRequestGpuPercell, PreparePRBInfo &prb_info, PacketTimingInfo &packet_timing_info);
    
    /**
     * @brief Set trigger timestamp for GPU direct communication
     *
     * @param nic_name   - NIC device name
     * @param slot_idx   - Slot index
     * @param trigger_ts - Trigger timestamp
     * @return 0 on success, negative error code on failure
     */
    int                 setTriggerTsGpuComm(std::string& nic_name,uint32_t slot_idx,uint64_t trigger_ts);
    
    /**
     * @brief Trigger CQE (Completion Queue Entry) tracer callback
     *
     * @param nic_name             - NIC device name
     * @param pTxRequestGpuPercell - Per-cell GPU TX request structure
     * @return 0 on success, negative error code on failure
     */
    int                 triggerCqeTracerCb(std::string& nic_name,TxRequestGpuPercell *pTxRequestGpuPercell);

    /**
     * @brief Print maximum TX/RX delays for a NIC
     *
     * @param nic_name - NIC device name
     * @return 0 on success, negative error code on failure
     */
    int print_max_delays(std::string nic_name);

    /**
     * @brief Get RX packet ordering state for a peer
     *
     * @param peer_id - Peer identifier
     * @return Pointer to RX ordering structure, nullptr on error
     */
    struct rx_order_t * getRxOrderItemsPeer(peer_id_t peer_id);

    /**
     * @brief Register memory region with fronthaul library
     *
     * @param memreg_info - Memory region information
     * @param memreg      - Output memory region handle
     * @return 0 on success, negative error code on failure
     */
    int registerMem(MemRegInfo const* memreg_info, MemRegHandle* memreg);
    
    /**
     * @brief Flush memory for a peer (cache synchronization)
     *
     * @param peer_ptr - Pointer to peer structure
     * @return 0 on success, negative error code on failure
     */
    int flushMemory(struct fh_peer_t * peer_ptr);
    
    /**
     * @brief Count total PRBs for PUSCH and PUCCH in a slot
     *
     * @param slot_info                  - Slot configuration
     * @param pusch_eAxC_id_count        - Number of PUSCH eAxC IDs
     * @param pucch_eAxC_id_count        - Number of PUCCH eAxC IDs
     * @param pusch_prb_symbol_map       - PUSCH PRB-symbol mapping array
     * @param pucch_prb_symbol_map       - PUCCH PRB-symbol mapping array
     * @param num_order_cells_sym_mask_arr - Ordering cell symbol mask array
     * @param cell_idx                   - Cell index
     * @param pusch_prb_non_zero         - Output flag indicating non-zero PUSCH PRBs
     * @param param                      - Additional parameters
     * @return Total PRB count
     */
    uint16_t countPuschPucchPrbs(slot_command_api::slot_info_t& slot_info, size_t pusch_eAxC_id_count, size_t pucch_eAxC_id_count,uint32_t* pusch_prb_symbol_map,uint32_t* pucch_prb_symbol_map,uint32_t* num_order_cells_sym_mask_arr,int cell_idx,uint8_t& pusch_prb_non_zero, void* param) const;
    
    /**
     * @brief Count total PRBs for PRACH in a slot
     *
     * @param slot_info          - Slot configuration
     * @param prach_eAxC_id_count - Number of PRACH eAxC IDs
     * @param param              - Additional parameters
     * @return Total PRB count
     */
    uint16_t countPrachPrbs(slot_command_api::slot_info_t& slot_info, size_t prach_eAxC_id_count, void* param) const;
    
    /**
     * @brief Count total PRBs for SRS in a slot
     *
     * @param slot_info        - Slot configuration
     * @param srs_eAxC_id_count - Number of SRS eAxC IDs
     * @return Total PRB count
     */
    uint32_t countSrsPrbs(slot_command_api::slot_info_t& slot_info, size_t srs_eAxC_id_count) const;
    
    /**
     * @brief Update fronthaul metrics (TX/RX packets, bytes, errors)
     *
     * @return 0 on success, negative error code on failure
     */
    int updateMetrics();

    MemFoot          mf;                                                   ///< Memory footprint tracker for fronthaul resources
    
    [[nodiscard]] uint16_t getDynamicBeamIdStart() const { return dynamic_beam_id_start; } ///< Get dynamic beam ID range start
    [[nodiscard]] uint16_t getDynamicBeamIdEnd() const { return dynamic_beam_id_end; }     ///< Get dynamic beam ID range end
    [[nodiscard]] uint16_t getDynamicBeamIdOffset() const { return dynamic_beam_id_offset; } ///< Get current dynamic beam ID offset
    
    /**
     * @brief Get dynamic beam ID offset of previous slot
     *
     * Calculates the beam ID offset used in the previous slot, wrapping
     * around the dynamic beam ID range as needed.
     *
     * @return Dynamic beam ID offset for previous slot
     */
    [[nodiscard]] uint16_t getDynamicBeamIdOffsetOfPrevSlot()
    {
        auto prev_slot_dyn_beam_id_offset = dynamic_beam_id_offset;
        if(prev_slot_dyn_beam_id_offset == dynamic_beam_id_start)
        {
            prev_slot_dyn_beam_id_offset = dynamic_beam_id_start + (dynamic_beam_id_covered_slots - 1) * dynamic_beam_ids_per_slot;
        }
        else
        {
            prev_slot_dyn_beam_id_offset -= dynamic_beam_ids_per_slot;
        }
        return prev_slot_dyn_beam_id_offset;
    }
    
    /**
     * @brief Update dynamic beam ID offset for next slot
     *
     * Advances the beam ID offset, wrapping back to start when end is reached.
     */
    void                   updateDynamicBeamIdOffset()
    {
        dynamic_beam_id_offset += dynamic_beam_ids_per_slot;
        if(dynamic_beam_id_offset + dynamic_beam_ids_per_slot > dynamic_beam_id_end) dynamic_beam_id_offset = dynamic_beam_id_start;
    }
    
    [[nodiscard]] uint16_t getStaticBeamIdStart() const { return static_beam_id_start; }   ///< Get static beam ID range start
    [[nodiscard]] uint16_t getStaticBeamIdEnd() const { return static_beam_id_end; }       ///< Get static beam ID range end
    [[nodiscard]] aerial_fh::BfwCplaneChainingMode getBfwCPlaneChainingMode() const { return bfw_c_plane_chaining_mode; } ///< Get beamforming weight C-plane chaining mode
    [[nodiscard]] size_t getBfwCoeffSize() const { return bfw_coeff_size; }                ///< Get beamforming weight coefficient size in bytes
    [[nodiscard]] bool getDlcBfwEnableDividePerCell() const { return dlc_bfw_enable_divide_per_cell; } ///< Check if DL C-plane beamforming weights are divided per cell
    [[nodiscard]] bool getUlcBfwEnableDividePerCell() const { return ulc_bfw_enable_divide_per_cell; } ///< Check if UL C-plane beamforming weights are divided per cell
    
    /**
     * @brief Update peer map with new peer
     *
     * @param peer_id    - Peer identifier
     * @param p_fh_peer  - Unique pointer to peer structure (ownership transferred)
     */
    void updatePeerMap(peer_id_t peer_id, std::unique_ptr<struct fh_peer_t> p_fh_peer); 

    /**
     * @brief Get NIC handle by bus address
     *
     * @param nic_bus_addr - NIC PCI bus address
     * @return NIC handle
     */
    aerial_fh::NicHandle getNic(std::string &nic_bus_addr) {
        return nic_map[nic_bus_addr]; 
    }

private:
    /**
     * @brief Receive U-plane packets (internal implementation)
     *
     * @param peer_ptr  - Pointer to peer structure
     * @param umsg_item - Output U-plane RX message container
     * @return Number of packets received, negative on error
     */
    int receiveUPlane(struct fh_peer_t * peer_ptr, struct umsg_fh_rx& umsg_item);
    
    /**
     * @brief Get static beamforming weights for a beam
     *
     * @param cell_id - Cell identifier
     * @param beamIdx - Beam index
     * @return Pointer to beamforming weight data, nullptr if not found
     */
    uint8_t*            getStaticBFWWeights(uint16_t cell_id, uint16_t beamIdx);
    
    /**
     * @brief Get peer structure by peer ID
     *
     * @param peer_id - Peer identifier
     * @return Pointer to peer structure, nullptr if not found
     */
    struct fh_peer_t *      getPeerFromId(peer_id_t peer_id);
    
    /**
     * @brief Get peer structure by absolute index
     *
     * @param index - Absolute peer index in vector
     * @return Pointer to peer structure, nullptr if not found
     */
    struct fh_peer_t *      getPeerFromAbsoluteId(int index);
    
    phydriver_handle        pdh{};                                     ///< Physical layer driver handle
    FronthaulHandle             fhi;                                   ///< Fronthaul library instance handle
    std::unordered_map<std::string, aerial_fh::NicHandle> nic_map;    ///< Map from NIC name to NIC handle
    std::unordered_map<peer_id_t, std::unique_ptr<struct fh_peer_t>> peer_map; ///< Map from peer ID to peer structure
    std::vector<peer_id_t> peer_id_vector;                             ///< Vector of registered peer IDs for iteration
    
    uint16_t static_beam_id_start;                                     ///< Static beam ID range start (inclusive)
    uint16_t static_beam_id_end;                                       ///< Static beam ID range end (exclusive)
    uint16_t dynamic_beam_id_start;                                    ///< Dynamic beam ID range start (inclusive)
    uint16_t dynamic_beam_id_end;                                      ///< Dynamic beam ID range end (exclusive)
    uint16_t dynamic_beam_id_offset;                                   ///< Current dynamic beam ID offset (rotates per slot)
    uint16_t dynamic_beam_id_covered_slots;                            ///< Number of slots covered by dynamic beam ID range
    uint16_t dynamic_beam_ids_per_slot;                                ///< Number of dynamic beam IDs per slot
    aerial_fh::BfwCplaneChainingMode bfw_c_plane_chaining_mode;        ///< Beamforming weight C-plane chaining mode (bundle or chain)
    size_t bfw_coeff_size;                                             ///< Beamforming weight coefficient size in bytes
    FhStaticBfwStorage fhStaticBfwStorage;                             ///< Storage for static beamforming weights (per cell, per beam)
    bool dlc_bfw_enable_divide_per_cell;                               ///< Divide/spread DL C-plane BFW transmission timing across multiple cells in a slot
    bool ulc_bfw_enable_divide_per_cell;                               ///< Divide/spread UL C-plane BFW transmission timing across multiple cells in a slot
    bool dlc_alloc_cplane_bfw_txq{};                                   ///< Allocate dedicated C-plane BFW TX queue for downlink
    bool ulc_alloc_cplane_bfw_txq{};                                   ///< Allocate dedicated C-plane BFW TX queue for uplink
};

#endif
