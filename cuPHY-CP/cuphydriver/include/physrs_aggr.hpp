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

#ifndef PHY_SRS_AGGR_CHANNEL_H
#define PHY_SRS_AGGR_CHANNEL_H

#include "phychannel.hpp"
#include "cell.hpp"
#include "cv_memory_bank_srs_chest.hpp"
//static constexpr uint32_t NUM_SRS_CH_EST_BUFFS = UL_SRS_MAX_CELLS_PER_SLOT * MAX_SRS_CHEST_BUFFERS_PER_CELL * CV_REPORT_TYPE;
static constexpr uint32_t NUM_SRS_REPORT = UL_SRS_MAX_CELLS_PER_SLOT * slot_command_api::MAX_SRS_CHEST_BUFFERS_PER_CELL;  ///< Total number of SRS report buffers across all cells
static constexpr uint32_t NUM_SRS_SNR_BUF = UL_SRS_MAX_CELLS_PER_SLOT * ORAN_MAX_PRB * slot_command_api::MAX_SRS_CHEST_BUFFERS_PER_CELL;  ///< Total number of per-PRB SNR values across all cells and buffers

/**
 * @brief Sounding Reference Signal (SRS) aggregated processing channel
 *
 * Handles uplink SRS signal processing including channel estimation, SNR calculation,
 * and channel state information extraction. SRS is used for uplink channel quality
 * measurement and beamforming weight computation. Supports aggregated multi-cell processing
 * with GPU-accelerated kernels and RKHS (Reproducing Kernel Hilbert Space) channel estimation.
 */
class PhySrsAggr : public PhyChannel {
public:

    /**
     * @brief Construct SRS channel processing object
     *
     * Initializes output buffers for SRS reports, channel estimation, and SNR data.
     * Allocates tensor descriptors for input data and configures cuPHY SRS RX parameters.
     *
     * @param[in] _pdh        - Cuphydriver handle
     * @param[in] _gDev       - GPU device struct handle
     * @param[in] _s_channel  - CUDA stream for channel processing
     * @param[in] _mpsCtx     - MPS/green context for GPU resource partitioning
     */
    PhySrsAggr(
        phydriver_handle _pdh,
        GpuDevice*       _gDev,
        cudaStream_t     _s_channel,
        MpsCtx *        _mpsCtx
    );

    /**
     * @brief Destructor - destroys cuPHY SRS RX handle and frees resources
     */
    ~PhySrsAggr();

    ////////////////////////////////////////////////////////////
    /// Module generic
    ////////////////////////////////////////////////////////////
    /**
     * @brief Setup SRS processing for current slot
     *
     * Configures dynamic parameters for SRS including cell configuration, UE parameters,
     * input buffer pointers from uplink ordering stage, and output buffer descriptors.
     *
     * @param[in] aggr_cell_list   - List of cells being processed
     * @param[in] aggr_ulbuf_st2   - Uplink input buffers for SRS
     * @return 0 on success, -1 on error
     */
    int setup(const std::vector<Cell *>& aggr_cell_list, const std::vector<ULInputBuffer *>& aggr_ulbuf_st2);
    
    /**
     * @brief Execute SRS processing on GPU
     *
     * Launches cuPHY SRS RX kernel to process SRS signals, compute channel estimates,
     * calculate per-PRB SNR values, and generate channel state information.
     * Supports both graph and non-graph execution modes.
     *
     * @return 0 on success, -1 on error
     */
    int run();
    
    /**
     * @brief Cleanup SRS processing resources for current slot
     *
     * @return 0 on success
     */
    int cleanup();
    
    /**
     * @brief Validate SRS output and generate debug dump if enabled
     *
     * When SRS_H5DUMP is defined, writes debug buffers to HDF5 file for analysis.
     *
     * @return 0 on success
     */
    int validate();
    
    /**
     * @brief Callback after SRS processing completion
     *
     * Invokes registered uplink callback function with SRS reports, channel estimation data,
     * SNR values, and timeout status for each cell. Provides results to upper layers (e.g., L2 scheduler).
     *
     * @param[in] srs_order_cell_timeout_list - Per-cell timeout status from packet ordering stage
     * @return 0 on success
     */
    int callback(const std::array<bool,UL_MAX_CELLS_PER_SLOT>& srs_order_cell_timeout_list);
    
    /**
     * @brief Update physical cell ID mapping
     *
     * @param[in] old_id - Old physical cell ID
     * @param[in] new_id - New physical cell ID
     */
    void updatePhyCellId(uint16_t,uint16_t);
    
    /**
     * @brief Print static SRS parameters for debugging
     *
     * @param[in] pStaticPrms - Pointer to static SRS parameters
     */
    void printStaticApiPrms(cuphySrsStatPrms_t const* pStaticPrms);

    ////////////////////////////////////////////////////////////
    /// Module specific
    ////////////////////////////////////////////////////////////
    // slot_command_api::pusch_params* getDynParams();
    /**
     * @brief Validate SRS output data
     *
     * Currently stubbed out, returns true.
     *
     * @return true (always)
     */
    bool                            validateOutput();
    
    /**
     * @brief Create cuPHY SRS RX object
     *
     * Initializes cuPHY library handle for SRS processing and configures static parameters
     * including antenna count, PRB configuration, RKHS parameters, and debug settings.
     *
     * @return 0 on success, error code otherwise
     */
    int                             createPhyObj();
    
    /**
     * @brief Get pointer to SRS parameters from slot command
     *
     * Retrieves SRS parameters from aggregated slot command.
     *
     * @return Pointer to SRS parameters
     */
    slot_command_api::srs_params*   getDynParams();
    
    /**
     * @brief Get pointer to cuPHY SRS dynamic parameters
     *
     * Returns internal cuPHY dynamic parameter structure.
     *
     * @return Pointer to cuPHY SRS dynamic parameters
     */
    cuphySrsDynPrms_t *             getSrsDynParams();
    
    /**
     * @brief Load test vector static parameters from HDF5 file
     *
     * @param[in] tv_h5     - Path to test vector HDF5 file
     * @param[in] cell_idx  - Cell index
     */
    void                            tvStatPrms(const char* tv_h5, int cell_idx);

protected:
    cuphy::tensor_device tDataRxInput[UL_SRS_MAX_CELLS_PER_SLOT];                  ///< Input tensor descriptors for received SRS data (per-cell)
    cuphy::buffer<__half2, cuphy::pinned_alloc> bDataRxSrs;                         ///< Pinned host buffer for raw SRS IQ (GPU->host, gated on DataLake)
    cuphy::buffer<cuphySrsReport_t, cuphy::pinned_alloc> srsReport;                 ///< Pinned host buffer for SRS processing reports (timing, SNR, status)
    cuphy::buffer<cuphySrsChEstToL2_t, cuphy::pinned_alloc> srsChEstToL2;          ///< Pinned host buffer for channel estimation metadata passed to L2
    cuphy::buffer<float, cuphy::pinned_alloc> rbSnrBuffer;                          ///< Pinned host buffer for per-PRB SNR values
    std::unique_ptr<hdf5hpp::hdf5_file> debugFileH;                                 ///< HDF5 file handle for debug output (when SRS_H5DUMP enabled)
    cuphy::buffer<uint32_t, cuphy::pinned_alloc> rbSnrBuffOffsets;                 ///< Pinned host buffer for SNR buffer offsets per UE
    ////////////////////////////////////////////
    //// cuPHY aggregate
    ////////////////////////////////////////////
    std::vector<cell_id_t> cell_id_list;                                            ///< List of PHY cell IDs being processed
    cuphySrsStatPrms_t static_params;                                               ///< Static parameters passed to cuPHY (setup-time configuration)
    std::vector<cuphyCellStatPrm_t> static_params_cell;                             ///< Per-cell static configuration parameters
    cuphySrsDynPrms_t dyn_params;                                                   ///< Dynamic parameters passed to cuPHY (per-slot configuration)
    cuphy::tensor_device srsChEstBuffInfo[slot_command_api::MAX_SRS_CHEST_BUFFERS]; ///< GPU tensor descriptors for channel estimation output buffers
    ////////////////////////////////////////////
    //// cuPHY specific
    ////////////////////////////////////////////
    cuphySrsRxHndl_t                 SrsRxHndl;                                     ///< cuPHY library handle for SRS RX operations
    cuphySrsDataIn_t                 DataIn;                                        ///< Input data structure for SRS processing (received signal tensors)
    cuphySrsDataOut_t                DataOut;                                       ///< Output data structure for SRS processing (channel estimates, reports, SNR)
    cuphySrsCellGrpDynPrm_t          cellGrpDynPrm;                                 ///< Cell group dynamic parameters (per-cell and per-UE SRS configuration)
    cuphyTensorPrm_t                 tDataRx;                                       ///< Tensor parameter for received data (unused/legacy)
    bool                             read_tv;                                       ///< Flag to enable reading test vectors from HDF5
    
    cuphySrsStatusOut_t              statusOut;                                     ///< Status output from cuPHY SRS processing

    cuphy::tensor_device             tPrmFocc_table;                                ///< FOCC lookup table
    cuphy::tensor_device             tPrmFocc_comb2_table;                          ///< FOCC table for comb-2 SRS configuration
    cuphy::tensor_device             tPrmFocc_comb4_table;                          ///< FOCC table for comb-4 SRS configuration
    cuphy::tensor_device             tPrmW_comb2_nPorts1_wide;                      ///< Wideband precoding matrix for comb-2, 1-port SRS
    cuphy::tensor_device             tPrmW_comb2_nPorts2_wide;                      ///< Wideband precoding matrix for comb-2, 2-port SRS
    cuphy::tensor_device             tPrmW_comb2_nPorts4_wide;                      ///< Wideband precoding matrix for comb-2, 4-port SRS
    cuphy::tensor_device             tPrmW_comb2_nPorts8_wide;                      ///< Wideband precoding matrix for comb-2, 8-port SRS
    cuphy::tensor_device             tPrmW_comb4_nPorts1_wide;                      ///< Wideband precoding matrix for comb-4, 1-port SRS
    cuphy::tensor_device             tPrmW_comb4_nPorts2_wide;                      ///< Wideband precoding matrix for comb-4, 2-port SRS
    cuphy::tensor_device             tPrmW_comb4_nPorts4_wide;                      ///< Wideband precoding matrix for comb-4, 4-port SRS
    cuphy::tensor_device             tPrmW_comb4_nPorts6_wide;                      ///< Wideband precoding matrix for comb-4, 6-port SRS
    cuphy::tensor_device             tPrmW_comb4_nPorts12_wide;                     ///< Wideband precoding matrix for comb-4, 12-port SRS
    cuphy::tensor_device             tPrmW_comb2_nPorts1_narrow;                    ///< Narrowband precoding matrix for comb-2, 1-port SRS
    cuphy::tensor_device             tPrmW_comb2_nPorts2_narrow;                    ///< Narrowband precoding matrix for comb-2, 2-port SRS
    cuphy::tensor_device             tPrmW_comb2_nPorts4_narrow;                    ///< Narrowband precoding matrix for comb-2, 4-port SRS
    cuphy::tensor_device             tPrmW_comb2_nPorts8_narrow;                    ///< Narrowband precoding matrix for comb-2, 8-port SRS
    cuphy::tensor_device             tPrmW_comb4_nPorts1_narrow;                    ///< Narrowband precoding matrix for comb-4, 1-port SRS
    cuphy::tensor_device             tPrmW_comb4_nPorts2_narrow;                    ///< Narrowband precoding matrix for comb-4, 2-port SRS
    cuphy::tensor_device             tPrmW_comb4_nPorts4_narrow;                    ///< Narrowband precoding matrix for comb-4, 4-port SRS
    cuphy::tensor_device             tPrmW_comb4_nPorts6_narrow;                    ///< Narrowband precoding matrix for comb-4, 6-port SRS
    cuphy::tensor_device             tPrmW_comb4_nPorts12_narrow;                   ///< Narrowband precoding matrix for comb-4, 12-port SRS
    cuphy::tensor_device             tPrmSrsRkhs_eigValues_grid0;                   ///< RKHS eigenvalues for grid 0
    cuphy::tensor_device             tPrmSrsRkhs_eigValues_grid1;                   ///< RKHS eigenvalues for grid 1
    cuphy::tensor_device             tPrmSrsRkhs_eigValues_grid2;                   ///< RKHS eigenvalues for grid 2
    cuphy::tensor_device             tPrmSrsRkhs_eigenCorr_grid0;                   ///< RKHS eigen-correlation matrix for grid 0
    cuphy::tensor_device             tPrmSrsRkhs_eigenCorr_grid1;                   ///< RKHS eigen-correlation matrix for grid 1
    cuphy::tensor_device             tPrmSrsRkhs_eigenCorr_grid2;                   ///< RKHS eigen-correlation matrix for grid 2
    cuphy::tensor_device             tPrmSrsRkhs_eigenVecs_grid0;                   ///< RKHS eigenvectors for grid 0
    cuphy::tensor_device             tPrmSrsRkhs_eigenVecs_grid1;                   ///< RKHS eigenvectors for grid 1
    cuphy::tensor_device             tPrmSrsRkhs_eigenVecs_grid2;                   ///< RKHS eigenvectors for grid 2
    cuphy::tensor_device             tPrmSrsRkhs_secondStageFourierPerm_grid0;      ///< RKHS second-stage Fourier permutation for grid 0
    cuphy::tensor_device             tPrmSrsRkhs_secondStageFourierPerm_grid1;      ///< RKHS second-stage Fourier permutation for grid 1
    cuphy::tensor_device             tPrmSrsRkhs_secondStageFourierPerm_grid2;      ///< RKHS second-stage Fourier permutation for grid 2
    cuphy::tensor_device             tPrmSrsRkhs_secondStageTwiddleFactors_grid0;   ///< RKHS second-stage twiddle factors for grid 0
    cuphy::tensor_device             tPrmSrsRkhs_secondStageTwiddleFactors_grid1;   ///< RKHS second-stage twiddle factors for grid 1
    cuphy::tensor_device             tPrmSrsRkhs_secondStageTwiddleFactors_grid2;   ///< RKHS second-stage twiddle factors for grid 2
    uint64_t                         procModeBmsk;                                  ///< Processing mode bitmask (graph vs. non-graph execution)
    CvSrsChestMemoryBank*            CvSrsChestMemBank;                             ///< Pointer to SRS channel estimate memory bank (manages buffer pool)
    cuphySrsBatchPrmHndl_t           batchPrmHndl;                                  ///< cuPHY batch parameter handle (for batched processing)
    cuphySrsRxHndl_t               srsRxHndl;                                       ///< cuPHY SRS RX handle (duplicate of SrsRxHndl)
    cuphySrsDbgPrms_t              srsDbgPrms;                                      ///< Debug parameters for runtime output (HDF5 dump control) // TODO: DELETE.
    cuphySrsStatDbgPrms_t          srsStatDbgPrms;                                  ///< Static debug parameters (setup-time HDF5 dump control)
    cuphySrsDynDbgPrms_t            srsDynDbgPrms;                                  ///< Dynamic debug parameters (per-slot HDF5 API logging control)
    cuphySrsRkhsPrms_t              srsRkhsPrms;                                    ///< RKHS algorithm parameters (grid sizes, eigenvalue settings)
};

#endif
