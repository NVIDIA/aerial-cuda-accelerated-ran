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

#ifndef CELL_CLASS_H
#define CELL_CLASS_H

#include <memory>
#include <vector>
#include <array>
#include <string>
#include <tuple>
#include "locks.hpp"
#include "cuphydriver_api.hpp"
#include "fh.hpp"
#include "constant.hpp"
#include "mps.hpp"
#include "memfoot.hpp"
#include "metrics.hpp"
#include "dlbuffer.hpp"
#include "ulbuffer.hpp"

/// Cell operational states - tracks whether a cell is processing traffic
enum cell_status
{
    CELL_ACTIVE=0,    ///< Cell is operational and processing data
    CELL_INACTIVE,    ///< Cell is configured but not processing traffic
    CELL_UNHEALTHY    ///< Cell has encountered errors and needs recovery
};

/**
 * Represents a single 5G/6G cell in the physical layer
 * 
 * This class encapsulates all configuration, buffers, and state for processing
 * uplink and downlink traffic for one cell. It manages:
 * - PHY layer processing via cuPHY
 * - Fronthaul (FH) connectivity via ORAN
 * - GPU resources and CUDA streams
 * - I/O buffers for UL/DL data paths
 * - Timing parameters and compression settings
 */
class Cell {
public:
    /**
     * Constructs a new Cell instance
     * 
     * @param[in] _pdh          PHY driver handle for this cell
     * @param[in] _cell_id      Unique cell identifier for this cell
     * @param[in] _mplane       M-plane (management plane) configuration
     * @param[in] _fh_proxy     Fronthaul proxy for communication with the NIC driver
     * @param[in] _gDev         GPU device for processing
     * @param[in] _idx          Cell index within the system. Used to index into system-wide arrays and buffers
     */
    Cell(
        phydriver_handle            _pdh,
        cell_id_t                   _cell_id,
        const cell_mplane_info&    _mplane,
        FhProxy*                    _fh_proxy,
        GpuDevice*                  _gDev,
        uint32_t                    _idx);

    ~Cell();

    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    //// Core Cell Management
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    phydriver_handle       getPhyDriverHandler(void) const;
    void                   start();                           ///< Activate cell to begin processing traffic
    void                   stop();                            ///< Deactivate cell and stop processing
    cell_id_t              getId() const;                     ///< Get internal cell ID (timestamp-based unique identifier for driver lookups)
    uint16_t               getPhyId() const;                  ///< Get 3GPP Physical Cell ID (PCI: 0-1007, or 0xFFFF if inactive)
    uint32_t               getIdx() const;                    ///< Get cell ordinal index (0, 1, 2...) for positioning in system arrays
    uint16_t               getMplaneId() const;               ///< Get management plane ID
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    //// Radio Configuration - antenna and bandwidth settings
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    bool                   getEarlyHarqDetEnabled() const;    ///< Check if early HARQ detection is enabled
    ru_type                getRUType() const;                 ///< Get O-RU type (FXN, FJT, OTHER)
    void                   setRUType(enum ru_type ru);        ///< Set O-RU type
    uint16_t               getRxAnt() const;                  ///< Get number of receive antennas
    uint16_t               getRxAntSrs() const;               ///< Get number of receive antennas configured for SRS procesing
    uint16_t               getTxAnt() const;                  ///< Get number of transmit antennas
    uint16_t               getPrbUlBwp() const;               ///< Get UL bandwidth part size in PRBs (Physical Resource Blocks)
    uint16_t               getPrbDlBwp() const;               ///< Get DL bandwidth part size in PRBs
    uint8_t                getMu() const;                     ///< Get numerology (μ) - determines subcarrier spacing: 15kHz * 2^μ. Note: only 30 kHz is supported.
    int                    getSlotAhead() const;              ///< Get advance slot timing for preparation of DL data before OTA transmission and
                                                              ///< for UL C-plane transmission.
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    //// ORAN Timing Parameters
    //// These define valid time windows for packet arrival per ORAN spec (in nanoseconds)
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    uint32_t               getT1aMaxUpNs() const;             ///< T1a max: Max timing advance, a DL U-plane data can arrive before transmission
    uint32_t               getT1aMaxCpUlNs() const;           ///< T1a max CP UL: Max timing advance for UL C-plane packets for transmission
    uint32_t               getT1aMinCpUlNs() const;           ///< T1a min CP UL: Min timing advance for UL C-plane packets for transmission
    uint32_t               getTa4MinNs() const;               ///< Ta4 min: Earliest UL U-plane data can arrive (PUSCH/PUCCH)
    uint32_t               getTa4MaxNs() const;               ///< Ta4 max: Latest UL U-plane data can arrive (PUSCH/PUCCH)
    uint32_t               getTa4MinNsSrs() const;            ///< Ta4 min SRS: Earliest SRS U-plane data can arrive
    uint32_t               getTa4MaxNsSrs() const;            ///< Ta4 max SRS: Latest SRS U-plane data can arrive
    uint32_t               getT1aMinCpDlNs() const;           ///< T1a min CP DL: Min timing advance for DL C-plane packets for transmission
    uint32_t               getT1aMaxCpDlNs() const;           ///< T1a max CP DL: Max timing advance for DL C-plane packets for transmission
    uint32_t               getUlUplaneTxOffsetNs() const;     ///< UE mode only - adjusts the timing for when the emulated UE should transmit its uplink data to match the expected arrival time at the base station.
    uint32_t               getTcpAdvDlNs() const;             ///< TCP advance for DL: Time to send C-plane before U-plane
    uint32_t               getMaxFhLen() const;               ///< Returns the maximum fronthaul distance (0: 0~30km, 1: 20~50km) which is used to adjust the timing parameters.
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    //// Channel-specific PRB (Physical Resource Block) stride configuration
    //// Stride determines memory layout for PRB data
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    uint32_t               getPuschPrbStride() const;         ///< PUSCH PRB stride: memory offset between PRBs
    void                   setPuschPrbStride(uint32_t pusch_prb_stride);
    uint32_t               getPrachPrbStride() const;         ///< PRACH PRB stride
    void                   setPrachPrbStride(uint32_t prach_prb_stride);
    uint32_t               getSrsPrbStride() const;           ///< SRS PRB stride
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    //// PUSCH and LDPC Decoder Configuration
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    uint8_t                getPuschLdpcMaxNumItrAlgoType() const; ///< Algorithm type for max iterations. Fixed number of iterations or variable number of iterations.
    uint8_t                getFixedMaxNumLdpcItrs() const;        ///< Number of maximum LDPC iterations if the algorithm type is fixed.
    uint8_t                getPuschLdpcEarlyTermination() const;  ///< Enable early termination when decoding converges
    uint8_t                getPuschLdpcAlgoIndex() const;         ///< LDPC algorithm variant index. Keep it at the default value of 0. cuPHY chooses the optimal algorithm based on the GPU.
    uint8_t                getPuschLdpcFlags() const;             ///< LDPC decoder configuration flags
    uint8_t                getPuschLdpcUseHalf() const;           ///< FP configuration flag - 0: FP32, 1: FP16
    uint16_t               getPuschnMaxPrb() const;               ///< Maximum PRBs for PUSCH - not LDPC specific.
    uint16_t               getPuschnMaxRx() const;                ///< Maximum receive antennas for PUSCH - not LDPC specific.
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    //// FH IQ Data Compression/Decompression - For fronthaul bandwidth reduction
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    uint16_t               getDLCompMeth() const;                 ///< DL compression method (none, BFP, etc.)
    uint8_t                getDLBitWidth() const;                 ///< DL compressed bit width per I/Q sample
    void                   setDLIQDataFmt(UserDataCompressionMethod comp_meth, uint8_t bit_width);
    uint16_t               getCompressionPrbSize() const;         ///< PRB size for compression blocks
    uint16_t               getULCompMeth() const;                 ///< UL compression method
    uint8_t                getULBitWidth() const;                 ///< UL compressed bit width per I/Q sample
    void                   setULIQDataFmt(UserDataCompressionMethod comp_meth, uint8_t bit_width);
    uint16_t               getDecompressionPrbSize() const;       ///< PRB size for decompression blocks
    uint16_t               getSection3TimeOffset() const;         ///< ORAN section type 3 timing offset
    void                   setSection3TimeOffset(uint16_t section_3_time_offset);
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    //// Signal Scaling and Power Control
    //// Manage signal amplitude to prevent clipping and optimize dynamic range
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    int                    getFsDlOffset() const;                 ///< DL full-scale shift offset (power-of-2 scaling for dynamic range)
    int                    getFsUlOffset() const;                 ///< UL full-scale shift offset (power-of-2 scaling for dynamic range)
    void                   setAttenuation_dB(float attenuation_dB); ///< Set signal attenuation in dB
    float                  getBetaUlPowerScaling() const;         ///< UL power scaling factor (beta)
    float                  getBetaDlPowerScaling() const;         ///< DL power scaling factor (beta)
    int                    getDlExponent() const;                 ///< DL exponent for block floating point
    void                   setDlExponent(int exp_dl);             ///< Set DL exponent
    void                   setUlExponent(int exp_ul);             ///< Set UL exponent
    void                   setRefDl(int ref_dl);                  ///< Set DL reference value
    int                    getUlMaxAmp() const;                   ///< Get UL maximum amplitude
    void                   setUlMaxAmp(int max_amp_ul);           ///< Set UL maximum amplitude to prevent saturation
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    //// Cell Health and State Management
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    bool                   isActive();                            ///< Check if cell is actively processing PUSCH/PUCCH/PRACH
    bool                   isActiveSrs();                         ///< Check if cell is actively processing SRS
    /**
     * Set cell active/inactive state
     * 
     * @param[in] _active  True to activate cell, false to deactivate
     */
    void                   setActive(bool _active);
    bool                   isHealthy();                           ///< Check if FH packet receive status for a cell is healthy (PUSCH/PUCCH/PRACH)
    bool                   isHealthySrs();                        ///< Check if FH packet receive status is healthy for SRS
    void                   setUnhealthy();                        ///< Mark cell as unhealthy due to errors in FH packet receive status
    void                   setUnhealthySrs();                     ///< Mark SRS processing as unhealthy due to errors in FH packet receive status
    void                   setHealthy();                          ///< Restore cell to healthy state
    void                   setHealthySrs();                       ///< Restore SRS to healthy state
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    //// Network Interface Configuration
    //// Ethernet and VLAN settings for fronthaul communication
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    std::array<uint8_t, 6> getSrcEthAddr() const;                 ///< Get source MAC address for this cell
    std::array<uint8_t, 6> getDstEthAddr() const;                 ///< Get destination MAC address (O-RU)
    uint16_t               getVlanTci() const;                    ///< Get VLAN TCI (Tag Control Information)
    uint8_t                getUplaneTxqCount() const;             ///< Get number of U-plane transmit queues
    fhproxy_peer           getPeer() const;                       ///< Get fronthaul peer information
    std::string            getNicName() const;                    ///< Get network interface device name (e.g., "eth0", "enp1s0f0")
    /**
     * Set network interface card for this cell
     * 
     * @param[in] nic_name  NIC device name (e.g., "eth0", "enp1s0f0")
     * @return 0 on success, negative error code on failure
     */
    int                    setNicName(std::string nic_name);
    uint32_t               getNicIndex() const;                   ///< Get NIC interface index
    uint8_t                getDlcCoreIndex() const;               ///< Get DL C-plane core index for fixed packing scheme
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    //// Buffer and Resource Management
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    void                   clearIOBuffers();                      ///< Clear all I/O buffers (DL and UL)
    void                   clearULBuffers();                      ///< Clear only UL buffers
    /**
     * Update performance and operational metrics for this cell that are exposed via Prometheus monitoring.
     * 
     * @param[in] metric  Metric type to update
     * @param[in] value   New metric value
     */
    void                   updateMetric(CellMetric metric, uint64_t value);
    void                   printCelleAxCIds();                    ///< Debug print all eAxC IDs configured for this cell
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    //// Runtime Configuration Updates
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * Update eAxCID mappings for each channel type (PUSCH, PUCCH, PRACH, SRS)
     * 
     * @param[in,out] eaxcids_ch_map  Map of channel types to eAxC ID vectors
     * @return 0 on success, negative error code on failure
     */
    int                    updateeAxCIds(std::unordered_map<int, std::vector<uint16_t>>& eaxcids_ch_map);
    /**
     * Update peer RX statistics
     * 
     * @param[in] rx_packets  Number of packets received
     * @param[in] rx_bytes    Number of bytes received
     * @return 0 on success, negative error code on failure
     */
    int                    update_peer_rx_metrics(size_t rx_packets, size_t rx_bytes);
    /**
     * Update peer TX statistics
     * 
     * @param[in] tx_packets  Number of packets transmitted
     * @param[in] tx_bytes    Number of bytes transmitted
     * @return 0 on success, negative error code on failure
     */
    int                    update_peer_tx_metrics(size_t tx_packets, size_t tx_bytes);
    /**
     * Update cell configuration for DL compression
     * 
     * @param[in] dl_comp_meth  Downlink compression method
     * @param[in] dl_bit_width  Downlink bit width after compression
     * @return 0 on success, negative error code on failure
     */
    int                    updateCellConfig(enum UserDataCompressionMethod dl_comp_meth, uint8_t dl_bit_width);
    /**
     * Update cell configuration for network addressing
     * 
     * @param[in] dst_eth_addr  Destination MAC address as string
     * @param[in] vlan_tci      VLAN tag control information
     * @return 0 on success, negative error code on failure
     */
    int                    updateCellConfig(std::string dst_eth_addr, uint16_t vlan_tci);
    /**
     * Update fronthaul length configuration
     * 
     * @param[in] fh_len_range  Fronthaul packet length range
     * @return 0 on success, negative error code on failure
     */
    int                    updateFhLenConfig(uint16_t fh_len_range);
    uint16_t               getDLGridSize() const;                 ///< Get DL resource grid size
    void                   setDLGridSize(uint16_t dl_grid_size);  ///< Set DL resource grid size
    uint16_t               getULGridSize() const;                 ///< Get UL resource grid size
    void                   setULGridSize(uint16_t dl_grid_size);  ///< Set UL resource grid size
    bfw_buffer_info*       getBfwCoeffBuffer();                   ///< Get beamforming weights coefficient buffer
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    //// cuPHY - Physical Layer Processing Interface
    //// Integration with NVIDIA cuPHY library for GPU-accelerated PHY processing
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * Configure static PHY parameters for this cell
     * 
     * @param[in] c_phy  Cell PHY information structure
     * @return 0 on success, negative error code on failure
     */
    int                    setPhyStatic(struct cell_phy_info& c_phy);
    /**
     * Initialize PHY objects and allocate resources
     * 
     * @return 0 on success, negative error code on failure
     */
    int                    setPhyObj();
    /**
     * Allocate and configure I/O buffers for PHY processing
     * 
     * @return 0 on success, negative error code on failure
     */
    int                    setIOBuf();
    DLOutputBuffer *       getNextDlBuffer();                     ///< Get next available DL output buffer (round-robin)
    ULInputBuffer *        getNextUlBufferST1();                  ///< Get next UL buffer for stage 1 processing
    ULInputBuffer *        getNextUlBufferST2();                  ///< Get next UL buffer for stage 2 processing
    ULInputBuffer *        getNextUlBufferST3();                  ///< Get next UL buffer for stage 3 processing
    ULInputBuffer *        getUlBufferPcap();                     ///< Get UL buffer for packet capture
    ULInputBuffer *        getUlBufferPcapTs();                   ///< Get UL buffer for packet capture with timestamps

    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    //// PRACH (Physical Random Access Channel) Configuration
    //// PRACH is used for initial UE access and timing advance estimation
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    cuphyPrachCellStatPrms_t* getPrachCellStatConfig();           ///< Get PRACH cell-level static configuration
    cuphyPrachOccaStatPrms_t* getPrachOccaStatConfig();           ///< Get PRACH occasion static configuration
    std::size_t               getPrachOccaSize() const;           ///< Get number of PRACH occasions
    std::vector<cuphyPrachOccaStatPrms_t>* getPrachOccaStatVec(); ///< Get vector of all PRACH occasion configs
    void                      setPrachOccaPrmStatIdx(uint16_t index); ///< Set PRACH occasion parameter index
    uint16_t                  getPrachOccaPrmStatIdx();           ///< Get current PRACH occasion parameter index

    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    //// Test Vector Files - HDF5 files for validation and testing
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    const char*            getTvPuschH5File(void);                ///< Get PUSCH test vector file path
    const char*            getTvPdschH5File(void);                ///< Get PDSCH test vector file path
    const char*            getTvPucchH5File(void);                ///< Get PUCCH test vector file path
    const char*            getTvPrachH5File(void);                ///< Get PRACH test vector file path
    const char*            getTvSrsH5File(void);                  ///< Get SRS test vector file path
    cuphyCellStatPrm_t*    getPhyStatic(void);                    ///< Get pointer to static PHY parameters

    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    //// FH - Fronthaul (ORAN) Interface
    //// Manage eAxC (enhanced Antenna-Carrier) IDs and flow control for ORAN fronthaul
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    peer_id_t                    getPeerId();                     ///< Get fronthaul peer identifier
    
    /**
     * eAxC IDs identify specific antenna-carrier streams in ORAN fronthaul.
     * Each physical channel (PDSCH, PUSCH, etc.) has its own set of eAxC IDs.
     * Format typically: DU_Port_ID + BandSector_ID + CC_ID + RU_Port_ID
     */
    const std::vector<uint16_t>& geteAxCIdsPdsch() const;         ///< Get eAxC IDs for PDSCH (DL data channel)
    const std::vector<uint16_t>& geteAxCIdsPusch() const;         ///< Get eAxC IDs for PUSCH (UL data channel)
    const std::vector<uint16_t>& geteAxCIdsPucch() const;         ///< Get eAxC IDs for PUCCH (UL control channel)
    const std::vector<uint16_t>& geteAxCIdsPrach() const;         ///< Get eAxC IDs for PRACH (random access)
    const std::vector<uint16_t>& geteAxCIdsSrs() const;           ///< Get eAxC IDs for SRS (sounding reference signals)
    std::vector<uint16_t>&       geteAxCIdsUl();                  ///< Get aggregated UL eAxC IDs (PUSCH+PUCCH+PRACH)
    size_t                       geteAxCNumPdsch() const;         ///< Get count of PDSCH eAxC IDs
    size_t                       geteAxCNumPusch() const;         ///< Get count of PUSCH eAxC IDs
    size_t                       geteAxCNumPucch() const;         ///< Get count of PUCCH eAxC IDs
    size_t                       geteAxCNumPrach() const;         ///< Get count of PRACH eAxC IDs
    size_t                       geteAxCNumSrs() const;           ///< Get count of SRS eAxC IDs
    /**
     * Check if a flow ID is valid for UL (PUSCH/PUCCH)
     * 
     * @param[in] flow_id  Flow identifier to check
     * @return true if flow_id is valid for this cell's UL channels
     */
    bool                         checkUlFlow(uint16_t flow_id);
    /**
     * Check if a flow ID is valid for PRACH
     * 
     * @param[in] flow_id  Flow identifier to check
     * @return true if flow_id is valid for this cell's PRACH
     */
    bool                         checkUlFlowPrach(uint16_t flow_id);
    void                         lockRxQueue();                   ///< Acquire RX queue lock (thread-safe access)
    void                         unlockRxQueue();                 ///< Release RX queue lock
    void                         resetSemIndices();               ///< Reset semaphore indices for UL packet ordering
    void                         resetSrsSemIndices();            ///< Reset semaphore indices for SRS packet ordering

    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    //// Packet Ordering and Timing Statistics
    //// Track semaphore indices and packet arrival statistics for monitoring and debugging
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    uint32_t*                    getLastRxItem() const;           ///< Get last received packet index (UL)
    uint32_t*                    getLastOrderedItem() const;      ///< Get last ordered packet index (UL)

    uint32_t*                    getLastRxSrsItem() const;        ///< Get last received packet index (SRS)
    uint32_t*                    getLastOrderedSrsItem() const;   ///< Get last ordered packet index (SRS)
    uint64_t*                    getOrderkernelLastTimeoutErrorTimeItem() const;    ///< Get timestamp of last order kernel timeout (UL)
    uint64_t*                    getOrderkernelSrsLastTimeoutErrorTimeItem() const; ///< Get timestamp of last order kernel timeout (SRS)

    /**
     * Packet timing statistics track whether packets arrive early, on-time, or late
     * relative to the expected timing window. These are critical for monitoring
     * fronthaul network performance and detecting timing issues.
     */
    uint32_t*                    getNextSlotEarlyRxPackets() const;       ///< Get count of early UL packets for next slot
    uint32_t*                    getNextSlotLateRxPackets() const;        ///< Get count of late UL packets for next slot
    uint32_t*                    getNextSlotOnTimeRxPackets() const;      ///< Get count of on-time UL packets for next slot
    uint32_t*                    getNextSlotEarlyRxPacketsSRS() const;    ///< Get count of early SRS packets for next slot
    uint32_t*                    getNextSlotLateRxPacketsSRS() const;     ///< Get count of late SRS packets for next slot
    uint32_t*                    getNextSlotOnTimeRxPacketsSRS() const;   ///< Get count of on-time SRS packets for next slot
    uint32_t*                    getNextSlotRxByteCountSRS() const;       ///< Get byte count for SRS packets in next slot
    uint32_t*                    getNextSlotRxPacketCountSRS() const;     ///< Get packet count for SRS in next slot
    uint32_t*                    getNextSlotRxPacketCountPerSymSRS() const; ///< Get per-symbol SRS packet count for next slot
    uint64_t*                    getNextSlotRxPacketTsSRS() const;        ///< Get SRS packet timestamp for next slot
    uint32_t*                    getNextSlotRxByteCount() const;          ///< Get byte count for UL packets in next slot
    uint32_t*                    getNextSlotRxPacketCount() const;        ///< Get packet count for UL in next slot
    uint64_t*                    getNextSlotRxPacketTs() const;           ///< Get UL packet timestamp for next slot
    uint32_t*                    getNextSlotNumPrbCh1() const;            ///< Get PRB count for channel 1 in next slot
    uint32_t*                    getNextSlotNumPrbCh2() const;            ///< Get PRB count for channel 2 in next slot
    uint32_t*                    getNextSlotNumPrbCh3() const;            ///< Get PRB count for channel 3 in next slot
    uint32_t*                    getULPcapCaptureBufferIndex() const;     ///< Get UL packet capture buffer index
    
    /**
     * Set PUSCH dynamic parameter index for a specific slot
     * 
     * @param[in] slot   Slot number (0-19 for a 10ms frame)
     * @param[in] index  Dynamic parameter index to use
     */
    void                         setPuschDynPrmIndex(uint16_t slot, int index) {pusch_cell_dyn_index[slot]=index;}
    /**
     * Get PUSCH dynamic parameter index for a specific slot
     * 
     * @param[in] slot  Slot number (0-19 for a 10ms frame)
     * @return Dynamic parameter index for the slot
     */
    int                          getPuschDynPrmIndex(uint16_t slot) {return pusch_cell_dyn_index[slot];}


    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    //// TX - Transmit Path
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    void                   lockTxQueue();                         ///< Acquire TX queue lock for thread-safe transmission
    void                   unlockTxQueue();                       ///< Release TX queue lock

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    //// CUDA + MPS - GPU Resource Management
    //// CUDA streams provide concurrent execution of GPU operations
    //// MPS (Multi-Process Service) enables GPU sharing between processes
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    GpuDevice *            getGpuDevice() const;                  ///< Get GPU device assigned to this cell
    cudaStream_t           getUlChannelStream();                  ///< Get CUDA stream for UL channel processing
    cudaStream_t           getUlOrderStream();                    ///< Get CUDA stream for UL packet ordering
    cudaStream_t           getDlStream();                         ///< Get CUDA stream for DL processing
    /**
     * Allocate and initialize GPU-side items (buffers, semaphores, etc.)
     * 
     * @return 0 on success, negative error code on failure
     */
    int                    setGpuItems();
    /**
     * Get PDSCH transport block buffer for a specific slot
     * 
     * @param[in] slot  Slot number
     * @return Pointer to transport block buffer for the slot
     */
    void *                 get_pdsch_tb_buffer(uint8_t slot)
    {
        return pdsch_tb_buffer[slot % PDSCH_MAX_GPU_BUFFS];
    };

    int                     getSrsDynPrmIndex();                  ///< Get SRS dynamic parameter index
    /**
     * Set SRS dynamic parameter index
     * 
     * @param[in] index  Dynamic parameter index to use for SRS
     */
    void                    setSrsDynPrmIndex(int index);
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    //// DOCA - NVIDIA DOCA (Data Center Infrastructure On a Chip Architecture)
    //// DOCA provides high-performance networking and DMA offload capabilities
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    struct doca_rx_items * docaGetRxqInfo();                      ///< Get DOCA RX queue info for UL
    uint16_t docaGetSemNum();                                     ///< Get number of semaphores for UL
    struct doca_rx_items * docaGetRxqInfoSrs();                   ///< Get DOCA RX queue info for SRS
    uint16_t docaGetSemNumSrs();                                  ///< Get number of semaphores for SRS
    
    /**
     * Calculate TTI (Transmission Time Interval) duration in nanoseconds from numerology
     * 
     * 5G NR uses different slot durations based on numerology (μ):
     * - μ=0 (15 kHz): 1 slot = 1.0 ms
     * - μ=1 (30 kHz): 1 slot = 0.5 ms
     * - μ=2 (60 kHz): 1 slot = 0.25 ms
     * - μ=3 (120 kHz): 1 slot = 0.125 ms (mmWave)
     * - μ=4 (240 kHz): 1 slot = 0.0625 ms (mmWave)
     * 
     * @param[in] mu  Numerology value (0-4)
     * @return TTI duration in nanoseconds, or 0 if invalid
     */
    static int getTtiNsFromMu(int mu)
    {
        switch(mu)
        {
        case 0: return 1000 * 1000;  // 1 ms
        case 1: return 500 * 1000;   // 0.5 ms
        case 2: return 250 * 1000;   // 0.25 ms
        case 3: return 125 * 1000;   // 0.125 ms
        case 4: return 625 * 100;    // 0.0625 ms
        }

        return 0;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    //// Public Health Monitoring Counters
    //// These atomic counters track error conditions to trigger health state changes
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    MemFoot          mf;                                          ///< Memory footprint tracker
    std::atomic<uint32_t> num_consecutive_ok_timeout;             ///< Consecutive UL order kernel timeouts (health indicator)
    std::atomic<uint32_t> num_consecutive_ok_timeout_srs;         ///< Consecutive SRS order kernel timeouts (health indicator)
    std::atomic<uint32_t> num_consecutive_unhealthy_slots;        ///< Consecutive unhealthy UL slots counter
    std::atomic<uint32_t> num_consecutive_unhealthy_slots_srs;    ///< Consecutive unhealthy SRS slots counter


private:
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    //// Core Cell Identity and Configuration
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    phydriver_handle   pdh;                                       ///< PHY driver handle
    std::atomic<cell_status>  active;                             ///< Cell operational status (PUSCH/PUCCH/PRACH)
    std::atomic<cell_status>  active_srs;                         ///< SRS operational status (separate from main UL)
    cell_id_t          cell_id;                                   ///< Unique cell identifier
    uint32_t           idx;                                       ///< Cell index within the system
    enum ru_type       ru;                                        ///< Radio Unit type
    uint16_t           mplane_id;                                 ///< Management plane identifier
    std::string        name;                                      ///< Cell name (for logging/debugging)
    std::string        nic_name;                                  ///< Network interface card name (e.g., "eth0")
    uint32_t           nic_index;                                 ///< NIC interface index
    uint8_t            dlc_core_index;                            ///< DL C-plane core index for fixed packing scheme
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    //// PHY Layer Configuration
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    cuphyCellStatPrm_t phy_stat;                                  ///< Static PHY parameters for this cell
    cuphyPrachCellStatPrms_t prachCellStatParams;                 ///< PRACH cell-level static parameters
    std::vector<cuphyPrachOccaStatPrms_t> prachOccaStatParamList; ///< List of PRACH occasion configurations
    uint16_t           prachOccaPrmStatIdx;                       ///< This cell's index into PrachStatPrms.pOccaPrms
    CellMetrics        metrics;                                   ///< Performance metrics for this cell

    std::unordered_map<std::string, peer_id_t> nic2peer_map;     ///< Map NIC names to fronthaul peer IDs

    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    //// Timing Configuration
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    int tti;                                                      ///< Transmission Time Interval (slot duration in ns)
    int slot_ahead;                                               ///< Number of slots to prepare DL data in advance

    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    //// Fronthaul and GPU Device Handles
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    FhProxy *     fh_proxy;                                       ///< Fronthaul proxy for ORAN communication
    fhproxy_peer  fpeer;                                          ///< Fronthaul peer configuration
    GpuDevice *   gDev;                                           ///< GPU device for PHY processing

    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    //// Network Layer Configuration (Ethernet/VLAN)
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    std::array<uint8_t, 6> src_eth_addr;                          ///< Source MAC address for this cell
    std::array<uint8_t, 6> dst_eth_addr;                          ///< Destination MAC address (O-RU)
    uint16_t vlan_tci;                                            ///< VLAN Tag Control Information (Priority + VID)
    uint8_t txq_count_uplane;                                     ///< Number of U-plane transmit queues
    uint16_t txq_size_cplane;                                     ///< C-plane transmit queue size
    uint16_t txq_size_uplane;                                     ///< U-plane transmit queue size
    uint16_t rxq_size_uplane;                                     ///< U-plane receive queue size

    uint32_t fh_len_range;                                        ///< Fronthaul packet length range
    uint16_t nMaxRxAnt;                                           ///< Maximum number of receive antennas

    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    //// eAxC ID Configuration (ORAN Antenna-Carrier Identifiers)
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    eAxC_list eAxC_ids;                                           ///< All eAxC IDs organized by channel type
    std::vector<cell_mplane_destination> mplane_destinations_;    ///< dMIMO: per-PE destinations (dst MAC + per-channel eAxC); empty for single-PE
    std::vector<uint16_t> eAxC_list_ul;                           ///< Unique list of eAxC IDs aggregated over PUSCH/PUCCH/PRACH channels

    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    //// CUDA Streams for Asynchronous Processing
    //// Using separate streams allows concurrent execution of different processing stages
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    cudaStream_t stream_ul;                                       ///< CUDA stream for UL channel processing
    cudaStream_t stream_order;                                    ///< CUDA stream for packet ordering
    cudaStream_t stream_dl;                                       ///< CUDA stream for DL processing

    ////////////////////////////////////////////////////////////////////////
    //// UL RX U-PLANE - Uplink Receive Timing and Configuration
    //// These parameters control when UL packets are expected to arrive
    ////////////////////////////////////////////////////////////////////////
    uint32_t t1a_max_cp_ul_ns;                                    ///< T1a max: Latest UL C-plane can arrive (ns)
    uint32_t t1a_min_cp_ul_ns;                                    ///< T1a min: Earliest UL C-plane can arrive (ns)
    uint32_t ta4_min_ns;                                          ///< Ta4 min: Earliest UL U-plane data (PUSCH/PUCCH) arrival (ns)
    uint32_t ta4_max_ns;                                          ///< Ta4 max: Latest UL U-plane data (PUSCH/PUCCH) arrival (ns)
    uint32_t ta4_min_ns_srs;                                      ///< Ta4 min: Earliest SRS U-plane data arrival (ns)
    uint32_t ta4_max_ns_srs;                                      ///< Ta4 max: Latest SRS U-plane data arrival (ns)
    uint32_t t1a_min_cp_dl_ns;                                    ///< T1a min: Earliest DL C-plane can arrive (ns)
    uint32_t t1a_max_cp_dl_ns;                                    ///< T1a max: Latest DL C-plane can arrive (ns)
    uint32_t ul_u_plane_tx_offset_ns;                             ///< UL U-plane transmission offset (ns)
    uint32_t pusch_prb_stride;                                    ///< PUSCH PRB stride (ORAN_PUSCH_PRBS_X_PORT_X_SYMBOL)
    uint32_t prach_prb_stride;                                    ///< PRACH PRB stride
    uint32_t srs_prb_stride;                                      ///< SRS PRB stride
    uint16_t pusch_nMaxPrb;                                       ///< Maximum PRBs for PUSCH
    uint16_t pusch_nMaxRx;                                        ///< Maximum receive antennas for PUSCH
    uint8_t pusch_ldpc_max_num_itr_algo_type;                     ///< LDPC max iteration algorithm type
    uint8_t pusch_fixed_max_num_ldpc_itrs;                        ///< Fixed max LDPC iterations
    uint8_t pusch_ldpc_early_termination;                         ///< Enable LDPC early termination
    uint8_t pusch_ldpc_algo_index;                                ///< LDPC algorithm index
    uint8_t pusch_ldpc_flags;                                     ///< LDPC configuration flags
    uint8_t pusch_ldpc_use_half;                                  ///< Use FP16 for LDPC decoding

    uint16_t section_3_time_offset;                               ///< ORAN section type 3 time offset
    uint16_t dl_grid_size;                                        ///< DL resource grid size
    uint16_t ul_grid_size;                                        ///< UL resource grid size

    ////////////////////////////////////////////////////////////////////////
    //// DL TX U-PLANE - Downlink Transmit Configuration
    ////////////////////////////////////////////////////////////////////////
    Mutex tx_queue_lock;                                          ///< Mutex for TX queue thread safety
    bool  tx_queue_locked;                                        ///< TX queue lock status
    uint32_t t1a_max_up_ns;                                       ///< T1a max: Latest DL U-plane can be sent (ns)
    uint32_t tcp_adv_dl_ns;                                       ///< TCP advance for DL: C-plane lead time before U-plane (ns)

    ////////////////////////////////////////////////////////////////////////
    //// TV files - Test Vector Files for Validation
    ////////////////////////////////////////////////////////////////////////
    std::string tv_pusch_h5;                                      ///< PUSCH test vector HDF5 file path
    std::string tv_srs_h5;                                        ///< SRS test vector HDF5 file path

    ////////////////////////////////////////////////////////////////////////
    //// COMPRESSION - IQ Data Compression Parameters
    //// Compression reduces fronthaul bandwidth requirements
    ////////////////////////////////////////////////////////////////////////
    enum UserDataCompressionMethod dl_comp_meth;                  ///< DL compression method (BFP, modulation, etc.)
    enum UserDataCompressionMethod ul_comp_meth;                  ///< UL compression method
    uint8_t dl_bit_width;                                         ///< DL bit width after compression
    uint8_t ul_bit_width;                                         ///< UL bit width after compression
    int                   fs_offset_dl;                           ///< DL full-scale shift offset (for BFP compression)
    int                   exponent_dl;                            ///< DL block floating point exponent
    int                   ref_dl;                                 ///< DL reference value for compression
    int                   fs_offset_ul;                           ///< UL full-scale shift offset (for BFP compression)
    int                   exponent_ul;                            ///< UL block floating point exponent
    int                   max_amp_ul;                             ///< UL maximum amplitude (prevents saturation)
    float                 beta_dl;                                ///< DL power scaling factor
    float                 beta_ul;                                ///< UL power scaling factor
    std::atomic<float>    oam_linear_gain;                        ///< OAM (Operations, Administration, Maintenance) linear gain

    //////////////////////////////////////////////////////////////////
    /// I/O Buffers - Double/Triple Buffering for Data Flow
    /// Multiple buffers enable pipelined processing without stalls
    //////////////////////////////////////////////////////////////////
    // DL Output Buffers (commented array shows old implementation)
    // std::array<DLOutputBuffer*, DL_OUTPUT_BUFFER_NUM_PER_CELL> dlbuf_list;
    std::vector<std::unique_ptr<DLOutputBuffer>> dlbuf_list;     ///< Pool of DL output buffers
    Mutex mlock_dlbuf;                                            ///< Mutex protecting DL buffer access
    int   dlbuf_index;                                            ///< Current DL buffer index (round-robin)

    // UL Input Buffers
    // std::array<ULInputBuffer*, UL_INPUT_BUFFER_NUM_PER_CELL> ulbuf_st1_list;
    std::vector<std::unique_ptr<ULInputBuffer>> ulbuf_st1_list;  ///< UL buffers for PUSCH/PUCCH
    std::vector<std::unique_ptr<ULInputBuffer>> ulbuf_st2_list;  ///< UL buffers for SRS
    std::vector<std::unique_ptr<ULInputBuffer>> ulbuf_st3_list;  ///< UL buffers for PRACH
    Mutex mlock_ulbuf_st1;                                        ///< Mutex for st1 buffer access
    Mutex mlock_ulbuf_st2;                                        ///< Mutex for st2 buffer access
    Mutex mlock_ulbuf_st3;                                        ///< Mutex for st3 buffer access
    int   ulbuf_st1_index;                                        ///< Current st1 buffer index
    int   ulbuf_st2_index;                                        ///< Current st2 buffer index
    int   ulbuf_st3_index;                                        ///< Current st3 buffer index

    // Packet Capture Buffers (for debugging/analysis)
    std::unique_ptr<ULInputBuffer>  ul_pcap_capture_buffer;      ///< Buffer for capturing UL packets
    std::unique_ptr<ULInputBuffer>  ul_pcap_capture_rxtimestamp_buffer; ///< Buffer for packet RX timestamps
    std::unique_ptr<dev_buf>        ul_pcap_capture_buffer_index; ///< Index into packet capture buffer

    int srs_cell_dyn_index;                                       ///< SRS dynamic parameter index
    Mutex rx_queue_lock;                                          ///< Mutex for RX queue access - prevents concurrent RX operations.
                                                                  ///< Protects state transitions during the UL_INIT → UL_SETUP → UL_START → UL_ORDERED state machine

    ////////////////////////////////////////////////////////////////////////
    //// DOCA RX Queue Mappings
    //// Map NICs to DOCA receive queue information for packet processing
    ////////////////////////////////////////////////////////////////////////
    std::unordered_map<std::string, struct doca_rx_items> nic2doca_rxq_info_map;     ///< NIC to DOCA RX queue map (UL)
    std::unordered_map<std::string, struct doca_rx_items> nic2doca_rxq_info_srs_map; ///< NIC to DOCA RX queue map (SRS)

    ////////////////////////////////////////////////////////////////////////
    //// Semaphore Indices for Packet Ordering
    //// Track last processed packet indices for GPU-based packet ordering
    ////////////////////////////////////////////////////////////////////////
    std::unique_ptr<dev_buf>              last_sem_idx_rx_h;       ///< Last RX semaphore index (UL)
    std::unique_ptr<dev_buf>              last_sem_idx_order_h;    ///< Last ordered semaphore index (UL)

    std::unique_ptr<dev_buf>              last_sem_idx_srs_rx_h;   ///< Last RX semaphore index (SRS)
    std::unique_ptr<dev_buf>              last_sem_idx_srs_order_h; ///< Last ordered semaphore index (SRS)
    
    std::unique_ptr<dev_buf>               order_kernel_last_timeout_error_time;     ///< Timestamp of last UL order kernel timeout
    std::unique_ptr<dev_buf>               order_kernel_srs_last_timeout_error_time; ///< Timestamp of last SRS order kernel timeout

    ////////////////////////////////////////////////////////////////////////
    //// Packet Timing Statistics - Device Buffers
    //// These GPU buffers track packet arrival timing for performance monitoring
    ////////////////////////////////////////////////////////////////////////
    // Packet timing stats for the next Uplink slot (overflow from current slot)
    std::unique_ptr<dev_buf>              next_slot_on_time_rx_packets;   ///< UL packets arriving on-time
    std::unique_ptr<dev_buf>              next_slot_early_rx_packets;     ///< UL packets arriving early
    std::unique_ptr<dev_buf>              next_slot_late_rx_packets;      ///< UL packets arriving late
    // SRS Packet timing stats for the next Uplink slot (overflow from current slot)
    std::unique_ptr<dev_buf>              next_slot_on_time_rx_packets_srs;  ///< SRS packets arriving on-time
    std::unique_ptr<dev_buf>              next_slot_early_rx_packets_srs;    ///< SRS packets arriving early
    std::unique_ptr<dev_buf>              next_slot_late_rx_packets_srs;     ///< SRS packets arriving late
    std::unique_ptr<dev_buf>              next_slot_rx_packets_count_srs;    ///< SRS packet count for next slot
    std::unique_ptr<dev_buf>              next_slot_rx_bytes_count_srs;      ///< SRS byte count for next slot
    std::unique_ptr<dev_buf>              next_slot_rx_packets_count_per_sym_srs; ///< SRS per-symbol packet count
    std::unique_ptr<dev_buf>              next_slot_rx_packets_ts_srs;       ///< SRS packet timestamps

    std::unique_ptr<dev_buf>              next_slot_rx_packets_ts;     ///< UL packet timestamps for next slot
    std::unique_ptr<dev_buf>              next_slot_rx_packets_count;  ///< UL packet count for next slot
    std::unique_ptr<dev_buf>              next_slot_rx_bytes_count;    ///< UL byte count for next slot

    std::unique_ptr<dev_buf>              next_slot_num_prb_ch1;       ///< PRB count for PUSCH/PUCCH on the next slot
    std::unique_ptr<dev_buf>              next_slot_num_prb_ch2;       ///< PRB count for PRACH on the next slot
    std::unique_ptr<dev_buf>              next_slot_num_prb_ch3;       ///< PRB count for SRS on the next slot
    
    ////////////////////////////////////////////////////////////////////////
    //// Per-Slot Dynamic Parameters
    ////////////////////////////////////////////////////////////////////////
    std::array<int, SLOTS_PER_FRAME>      pusch_cell_dyn_index = {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1}; ///< PUSCH dynamic parameter indices per slot (20 slots per frame)
                                                                                                                                ///< -1: indicates no PUSCH/PUCCH is scheduled for that slot
                                                                                                                                ///< 0 to N: index into cuPHY's dynamic parameter arrays for this slot's PUSCH data
    ////////////////////////////////////////////////////////////////////////
    //// PDSCH Transport Block Buffers
    ////////////////////////////////////////////////////////////////////////
    void *                                 pdsch_tb_buffer[PDSCH_MAX_GPU_BUFFS]; ///< PDSCH transport block buffers on GPU. 
                                                                                 ///<It is sized to hold PDSCH TB data for all UEs in a cell.
    //cudaEvent_t                            pdsch_tb_cpy_complete; // (commented out - CUDA event for copy completion)

    ////////////////////////////////////////////////////////////////////////
    //// BFW (Beamforming Weights) Parameters
    //// Buffers for storing and transferring beamforming coefficients
    ////////////////////////////////////////////////////////////////////////
    bfw_buffer_info_header                      bfw_coeff_buffer_header;  ///< BFW buffer header information
    cuphy::buffer<uint8_t, cuphy::pinned_alloc> bfw_coeff_buffer_pinned;  ///< Pinned host memory for BFW coefficients
    cuphy::buffer<uint8_t, cuphy::device_alloc> bfw_coeff_buffer_dev;     ///< GPU device memory for BFW coefficients
    bfw_buffer_info                             bfw_coeff_buffer_info;    ///< BFW buffer information structure
};

#endif
