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

#define TAG (NVLOG_TAG_BASE_CUPHY_DRIVER + 39) // "DRV.SRS"

#include "physrs_aggr.hpp"
#include "cuphydriver_api.hpp"
#include "context.hpp"
#include "nvlog.hpp"
#include "exceptions.hpp"
#include "cuphyoam.hpp"
#include "cuphy.h"
#include "scf_5g_fapi.h"


// #define PUSCH_INPUT_BUFFER_DEBUG
//#define SRS_H5DUMP

void printParametersAggr(PhyDriverCtx* pdctx, const cuphySrsCellGrpDynPrm_t* l2);

PhySrsAggr::PhySrsAggr(
    phydriver_handle _pdh,
    GpuDevice*       _gDev,
    cudaStream_t     _s_channel,
    MpsCtx *        _mpsCtx) :
    PhyChannel(_pdh, _gDev, 0, _s_channel, _mpsCtx)
{
    PhyDriverCtx * pdctx = StaticConversion<PhyDriverCtx>(_pdh).get();

    mf.init(_pdh, std::string("PhySrsAggr"), sizeof(PhySrsAggr));
    cuphyMf.init(_pdh, std::string("cuphySrsRx"), 0);

    channel_type = slot_command_api::channel_type::SRS;
    channel_name.assign("SRS");


    NVLOGI_FMT(TAG, "PhySrsAggr{}: construct", this_id);

    //-------------------------------------------------------------
    // Check for configuration information in the input file. Newer
    // input files will have configuration values in the file, so
    // that they don't need to be specified on the command line.
    cuphy::disable_hdf5_error_print(); // Temporarily disable HDF5 stderr printing

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////// Init SRS
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    try
    {

        // initialize output buffers
        srsReport     = std::move(cuphy::buffer<cuphySrsReport_t, cuphy::pinned_alloc>(NUM_SRS_REPORT));
        rbSnrBuffer   = std::move(cuphy::buffer<float, cuphy::pinned_alloc>(NUM_SRS_SNR_BUF));
        rbSnrBuffOffsets = std::move(cuphy::buffer<uint32_t, cuphy::pinned_alloc>(NUM_SRS_REPORT));
        srsChEstToL2 = std::move(cuphy::buffer<cuphySrsChEstToL2_t, cuphy::pinned_alloc>(NUM_SRS_REPORT));
        
        memset(srsReport.addr(), 0, sizeof(cuphySrsReport_t) * NUM_SRS_REPORT);
        memset(rbSnrBuffer.addr(), 0, sizeof(float) * NUM_SRS_SNR_BUF);
        memset(rbSnrBuffOffsets.addr(), 0, sizeof(uint32_t) * NUM_SRS_REPORT);
        DataOut.pChEstBuffInfo = (cuphySrsChEstBuffInfo_t*) calloc(slot_command_api::MAX_SRS_CHEST_BUFFERS, sizeof(cuphySrsChEstBuffInfo_t));
        srsRkhsPrms.pRkhsGridPrms = ( cuphyRkhsGridPrms_t *) calloc(NUM_RKHS_GRIDS, sizeof(cuphyRkhsGridPrms_t)); 
        srsRkhsPrms.nGridSizes = NUM_RKHS_GRIDS;
        for(int i = 0; i < slot_command_api::MAX_SRS_CHEST_BUFFERS; i++)
        {

            #if 0
            srsChEstBuffInfo[i] = std::move(cuphy::tensor_device(nullptr, CUPHY_C_32F, CV_NUM_PRBG, 
                                                            CV_NUM_GNB_ANT,
                                                            CV_NUM_UE_LAYER, 
                                                            cuphy::tensor_flags::align_tight));
            DataOut.pChEstBuffInfo[i].tChEstBuffer.desc = srsChEstBuffInfo[i].desc().handle();
            #endif
            DataOut.pChEstBuffInfo[i].tChEstBuffer.pAddr = nullptr;
        }
        DataOut.pSrsReports            = srsReport.addr();
        DataOut.pSrsChEstToL2          = srsChEstToL2.addr();
        DataOut.pRbSnrBuffer           = rbSnrBuffer.addr();
        DataOut.pRbSnrBuffOffsets      = rbSnrBuffOffsets.addr();
        gDev->synchronizeStream(s_channel);
    }
        PHYDRIVER_CATCH_THROW_EXCEPTIONS();

        cellGrpDynPrm.nCells = 0;
        cellGrpDynPrm.pCellPrms = (cuphySrsCellDynPrm_t*) calloc(UL_SRS_MAX_CELLS_PER_SLOT, sizeof(cuphySrsCellDynPrm_t));
        cellGrpDynPrm.nSrsUes = 0;
        cellGrpDynPrm.pUeSrsPrms = (cuphyUeSrsPrm_t*) calloc(slot_command_api::MAX_SRS_CHEST_BUFFERS, sizeof(cuphyUeSrsPrm_t));
        dyn_params.pCellGrpDynPrm = &cellGrpDynPrm;

        DataIn.pTDataRx = (cuphyTensorPrm_t*) calloc(UL_SRS_MAX_CELLS_PER_SLOT, sizeof(cuphyTensorPrm_t));

        // Data IN
        for(int idx = 0; idx < UL_SRS_MAX_CELLS_PER_SLOT; idx++)
        {
            tDataRxInput[idx] = std::move(cuphy::tensor_device(CUPHY_C_16F, ORAN_MAX_PRB * CUPHY_N_TONES_PER_PRB,
                                                                ORAN_MAX_SRS_SYMBOLS, MAX_AP_PER_SLOT_SRS, /*Change this  to actual number of SRS eAxC IDs used in the test if SRS_H5DUMP is enabled*/
                                                                cuphy::tensor_flags::align_tight));
            mf.addGpuRegularSize(tDataRxInput[idx].desc().get_size_in_bytes());
            DataIn.pTDataRx[idx].desc = tDataRxInput[idx].desc().handle();
            DataIn.pTDataRx[idx].pAddr = nullptr;
        }
        dyn_params.pDataIn = &DataIn;

        // SRS IQ host buffer: copies buf_st_2 GPU->host (mirrors PUSCH bDataRx pattern)
        int totSrsIqSize = (ORAN_MAX_PRB * CUPHY_N_TONES_PER_PRB) * ORAN_MAX_SRS_SYMBOLS * MAX_AP_PER_SLOT * UL_SRS_MAX_CELLS_PER_SLOT;
        if(pdctx->datalake_enabled()) {
            bDataRxSrs  = std::move(cuphy::buffer<__half2, cuphy::pinned_alloc>(totSrsIqSize));
            DataOut.pDataRxSrs = bDataRxSrs.addr();
        } else {
            bDataRxSrs  = std::move(cuphy::buffer<__half2, cuphy::pinned_alloc>(0));
            DataOut.pDataRxSrs = nullptr;
        }

        // Data OUT
        dyn_params.pDataOut = &DataOut;
        
        statusOut = {cuphySrsStatusType_t::CUPHY_SRS_STATUS_SUCCESS_OR_UNTRACKED_ISSUE, MAX_UINT16, MAX_UINT16};
        dyn_params.pStatusOut = &statusOut;
          
    cuphy::enable_hdf5_error_print(); // Re-enable HDF5 stderr printing
    static_params_cell.clear();
    cell_id_list.clear();

    launch_kernel_warmup(s_channel);
    launch_kernel_order(s_channel, 1, 0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, 0, 0, 0, 0, nullptr, 0, 0, 0, 0, 0);
    launch_kernel_order(s_channel, 1, 0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, 0, 0, 0, 0, nullptr, 0, 0, 0, 0, 0);
    gDev->synchronizeStream(s_channel);

    procModeBmsk = SRS_PROC_MODE_FULL_SLOT;

    read_tv = false;
    srsRxHndl = nullptr;
};

PhySrsAggr::~PhySrsAggr()
{
    if(srsRxHndl)
        cuphyDestroySrsRx(srsRxHndl);

    //cudaFreeHost(DataInOut.pHarqBuffersInOut);

    //CUDA_CHECK_PHYDRIVER(cudaEventDestroy(start_setup));
    //CUDA_CHECK_PHYDRIVER(cudaEventDestroy(end_setup));

    free(cellGrpDynPrm.pCellPrms);
    free(cellGrpDynPrm.pUeSrsPrms);
    free(DataIn.pTDataRx);
};

void PhySrsAggr::tvStatPrms(const char* tv_h5, int cell_idx) 
{
    PhyDriverCtx * pdctx = StaticConversion<PhyDriverCtx>(pdh).get();

    fInput = hdf5hpp::hdf5_file::open(tv_h5);
    
    cudaStreamSynchronize(s_channel);
    if(read_tv == true)
        return;
    // load filters (currently hardcoded to FP32). Load true TB data
    tPrmFocc_table  = cuphy::tensor_from_dataset(fInput.open_dataset("focc_table"), CUPHY_C_16F, cuphy::tensor_flags::align_tight, s_channel);
    mf.addGpuRegularSize(tPrmFocc_table.desc().get_size_in_bytes());
    tPrmFocc_comb2_table = cuphy::tensor_from_dataset(fInput.open_dataset("focc_table_comb2"), CUPHY_C_16F, cuphy::tensor_flags::align_tight, s_channel);
    mf.addGpuRegularSize(tPrmFocc_comb2_table.desc().get_size_in_bytes());
    tPrmFocc_comb4_table = cuphy::tensor_from_dataset(fInput.open_dataset("focc_table_comb4"), CUPHY_C_16F, cuphy::tensor_flags::align_tight, s_channel);
    mf.addGpuRegularSize(tPrmFocc_comb4_table.desc().get_size_in_bytes());
    tPrmW_comb2_nPorts1_wide = cuphy::tensor_from_dataset(fInput.open_dataset("W_comb2_nPorts1_wide"), CUPHY_C_16F, cuphy::tensor_flags::align_tight, s_channel);
    mf.addGpuRegularSize(tPrmW_comb2_nPorts1_wide.desc().get_size_in_bytes());
    tPrmW_comb2_nPorts2_wide = cuphy::tensor_from_dataset(fInput.open_dataset("W_comb2_nPorts2_wide"), CUPHY_C_16F, cuphy::tensor_flags::align_tight, s_channel);
    mf.addGpuRegularSize(tPrmW_comb2_nPorts2_wide.desc().get_size_in_bytes());
    tPrmW_comb2_nPorts4_wide = cuphy::tensor_from_dataset(fInput.open_dataset("W_comb2_nPorts4_wide"), CUPHY_C_16F, cuphy::tensor_flags::align_tight, s_channel);
    mf.addGpuRegularSize(tPrmW_comb2_nPorts4_wide.desc().get_size_in_bytes());
    tPrmW_comb2_nPorts8_wide = cuphy::tensor_from_dataset(fInput.open_dataset("W_comb2_nPorts8_wide"), CUPHY_C_16F, cuphy::tensor_flags::align_tight, s_channel);
    mf.addGpuRegularSize(tPrmW_comb2_nPorts8_wide.desc().get_size_in_bytes());
    tPrmW_comb4_nPorts1_wide = cuphy::tensor_from_dataset(fInput.open_dataset("W_comb4_nPorts1_wide"), CUPHY_C_16F, cuphy::tensor_flags::align_tight, s_channel);
    mf.addGpuRegularSize(tPrmW_comb4_nPorts1_wide.desc().get_size_in_bytes());
    tPrmW_comb4_nPorts2_wide = cuphy::tensor_from_dataset(fInput.open_dataset("W_comb4_nPorts2_wide"), CUPHY_C_16F, cuphy::tensor_flags::align_tight, s_channel);
    mf.addGpuRegularSize(tPrmW_comb4_nPorts2_wide.desc().get_size_in_bytes());
    tPrmW_comb4_nPorts4_wide = cuphy::tensor_from_dataset(fInput.open_dataset("W_comb4_nPorts4_wide"), CUPHY_C_16F, cuphy::tensor_flags::align_tight, s_channel);
    mf.addGpuRegularSize(tPrmW_comb4_nPorts4_wide.desc().get_size_in_bytes());
    tPrmW_comb4_nPorts6_wide = cuphy::tensor_from_dataset(fInput.open_dataset("W_comb4_nPorts6_wide"), CUPHY_C_16F, cuphy::tensor_flags::align_tight, s_channel);
    mf.addGpuRegularSize(tPrmW_comb4_nPorts6_wide.desc().get_size_in_bytes());
    tPrmW_comb4_nPorts12_wide = cuphy::tensor_from_dataset(fInput.open_dataset("W_comb4_nPorts12_wide"), CUPHY_C_16F, cuphy::tensor_flags::align_tight, s_channel);
    mf.addGpuRegularSize(tPrmW_comb4_nPorts12_wide.desc().get_size_in_bytes());
    tPrmW_comb2_nPorts1_narrow = cuphy::tensor_from_dataset(fInput.open_dataset("W_comb2_nPorts1_narrow"), CUPHY_C_16F, cuphy::tensor_flags::align_tight, s_channel);
    mf.addGpuRegularSize(tPrmW_comb2_nPorts1_narrow.desc().get_size_in_bytes());
    tPrmW_comb2_nPorts2_narrow = cuphy::tensor_from_dataset(fInput.open_dataset("W_comb2_nPorts2_narrow"), CUPHY_C_16F, cuphy::tensor_flags::align_tight, s_channel);
    mf.addGpuRegularSize(tPrmW_comb2_nPorts2_narrow.desc().get_size_in_bytes());
    tPrmW_comb2_nPorts4_narrow = cuphy::tensor_from_dataset(fInput.open_dataset("W_comb2_nPorts4_narrow"), CUPHY_C_16F, cuphy::tensor_flags::align_tight, s_channel);
    mf.addGpuRegularSize(tPrmW_comb2_nPorts4_narrow.desc().get_size_in_bytes());
    tPrmW_comb2_nPorts8_narrow = cuphy::tensor_from_dataset(fInput.open_dataset("W_comb2_nPorts8_narrow"), CUPHY_C_16F, cuphy::tensor_flags::align_tight, s_channel);
    mf.addGpuRegularSize(tPrmW_comb2_nPorts8_narrow.desc().get_size_in_bytes());
    tPrmW_comb4_nPorts1_narrow = cuphy::tensor_from_dataset(fInput.open_dataset("W_comb4_nPorts1_narrow"), CUPHY_C_16F, cuphy::tensor_flags::align_tight, s_channel);
    mf.addGpuRegularSize(tPrmW_comb4_nPorts1_narrow.desc().get_size_in_bytes());
    tPrmW_comb4_nPorts2_narrow = cuphy::tensor_from_dataset(fInput.open_dataset("W_comb4_nPorts2_narrow"), CUPHY_C_16F, cuphy::tensor_flags::align_tight, s_channel);
    mf.addGpuRegularSize(tPrmW_comb4_nPorts2_narrow.desc().get_size_in_bytes());
    tPrmW_comb4_nPorts4_narrow = cuphy::tensor_from_dataset(fInput.open_dataset("W_comb4_nPorts4_narrow"), CUPHY_C_16F, cuphy::tensor_flags::align_tight, s_channel);    
    mf.addGpuRegularSize(tPrmW_comb4_nPorts4_narrow.desc().get_size_in_bytes());
    tPrmW_comb4_nPorts6_narrow = cuphy::tensor_from_dataset(fInput.open_dataset("W_comb4_nPorts6_narrow"), CUPHY_C_16F, cuphy::tensor_flags::align_tight, s_channel);    
    mf.addGpuRegularSize(tPrmW_comb4_nPorts6_narrow.desc().get_size_in_bytes());
    tPrmW_comb4_nPorts12_narrow= cuphy::tensor_from_dataset(fInput.open_dataset("W_comb4_nPorts12_narrow"), CUPHY_C_16F, cuphy::tensor_flags::align_tight, s_channel);    
    mf.addGpuRegularSize(tPrmW_comb4_nPorts12_narrow.desc().get_size_in_bytes());
    tPrmSrsRkhs_eigValues_grid0 = cuphy::tensor_from_dataset(fInput.open_dataset("srsRkhs_eigValues_grid0"), CUPHY_R_32F, cuphy::tensor_flags::align_tight, s_channel);
    mf.addGpuRegularSize(tPrmSrsRkhs_eigValues_grid0.desc().get_size_in_bytes());
    tPrmSrsRkhs_eigValues_grid1 = cuphy::tensor_from_dataset(fInput.open_dataset("srsRkhs_eigValues_grid1"), CUPHY_R_32F, cuphy::tensor_flags::align_tight, s_channel);
    mf.addGpuRegularSize(tPrmSrsRkhs_eigValues_grid1.desc().get_size_in_bytes());
    tPrmSrsRkhs_eigValues_grid2 = cuphy::tensor_from_dataset(fInput.open_dataset("srsRkhs_eigValues_grid2"), CUPHY_R_32F, cuphy::tensor_flags::align_tight, s_channel);
    mf.addGpuRegularSize(tPrmSrsRkhs_eigValues_grid2.desc().get_size_in_bytes());
    tPrmSrsRkhs_eigenCorr_grid0 = cuphy::tensor_from_dataset(fInput.open_dataset("srsRkhs_eigenCorr_grid0"), CUPHY_C_16F, cuphy::tensor_flags::align_tight, s_channel);
    mf.addGpuRegularSize(tPrmSrsRkhs_eigenCorr_grid0.desc().get_size_in_bytes());
    tPrmSrsRkhs_eigenCorr_grid1 = cuphy::tensor_from_dataset(fInput.open_dataset("srsRkhs_eigenCorr_grid1"), CUPHY_C_16F, cuphy::tensor_flags::align_tight, s_channel);
    mf.addGpuRegularSize(tPrmSrsRkhs_eigenCorr_grid1.desc().get_size_in_bytes());
    tPrmSrsRkhs_eigenCorr_grid2 = cuphy::tensor_from_dataset(fInput.open_dataset("srsRkhs_eigenCorr_grid2"), CUPHY_C_16F, cuphy::tensor_flags::align_tight, s_channel);
    mf.addGpuRegularSize(tPrmSrsRkhs_eigenCorr_grid2.desc().get_size_in_bytes());
    tPrmSrsRkhs_eigenVecs_grid0 = cuphy::tensor_from_dataset(fInput.open_dataset("srsRkhs_eigenVecs_grid0"), CUPHY_R_32F, cuphy::tensor_flags::align_tight, s_channel);
    mf.addGpuRegularSize(tPrmSrsRkhs_eigenVecs_grid0.desc().get_size_in_bytes());
    tPrmSrsRkhs_eigenVecs_grid1 = cuphy::tensor_from_dataset(fInput.open_dataset("srsRkhs_eigenVecs_grid1"), CUPHY_R_32F, cuphy::tensor_flags::align_tight, s_channel);
    mf.addGpuRegularSize(tPrmSrsRkhs_eigenVecs_grid1.desc().get_size_in_bytes());
    tPrmSrsRkhs_eigenVecs_grid2 = cuphy::tensor_from_dataset(fInput.open_dataset("srsRkhs_eigenVecs_grid2"), CUPHY_R_32F, cuphy::tensor_flags::align_tight, s_channel);
    mf.addGpuRegularSize(tPrmSrsRkhs_eigenVecs_grid2.desc().get_size_in_bytes());
    //tPrmSrsRkhs_secondStageFourierPerm_grid0 = cuphy::tensor_from_dataset(fInput.open_dataset("srsRkhs_secondStageFourierPerm_grid0"), CUPHY_R_8U, cuphy::tensor_flags::align_tight, s_channel);
    //mf.addGpuRegularSize(tPrmSrsRkhs_secondStageFourierPerm_grid0.desc().get_size_in_bytes());
    //tPrmSrsRkhs_secondStageFourierPerm_grid1 = cuphy::tensor_from_dataset(fInput.open_dataset("srsRkhs_secondStageFourierPerm_grid1"), CUPHY_R_8U, cuphy::tensor_flags::align_tight, s_channel);
    //mf.addGpuRegularSize(tPrmSrsRkhs_secondStageFourierPerm_grid1.desc().get_size_in_bytes());
    tPrmSrsRkhs_secondStageFourierPerm_grid2 = cuphy::tensor_from_dataset(fInput.open_dataset("srsRkhs_secondStageFourierPerm_grid2"), CUPHY_R_8U, cuphy::tensor_flags::align_tight, s_channel);
    mf.addGpuRegularSize(tPrmSrsRkhs_secondStageFourierPerm_grid2.desc().get_size_in_bytes());
    //tPrmSrsRkhs_secondStageTwiddleFactors_grid0 = cuphy::tensor_from_dataset(fInput.open_dataset("srsRkhs_secondStageTwiddleFactors_grid0"), CUPHY_C_32F, cuphy::tensor_flags::align_tight, s_channel);
    //mf.addGpuRegularSize(tPrmSrsRkhs_secondStageTwiddleFactors_grid0.desc().get_size_in_bytes());
    //tPrmSrsRkhs_secondStageTwiddleFactors_grid1 = cuphy::tensor_from_dataset(fInput.open_dataset("srsRkhs_secondStageTwiddleFactors_grid1"), CUPHY_C_32F, cuphy::tensor_flags::align_tight, s_channel);
    //mf.addGpuRegularSize(tPrmSrsRkhs_secondStageTwiddleFactors_grid1.desc().get_size_in_bytes());
    tPrmSrsRkhs_secondStageTwiddleFactors_grid2 = cuphy::tensor_from_dataset(fInput.open_dataset("srsRkhs_secondStageTwiddleFactors_grid2"), CUPHY_C_32F, cuphy::tensor_flags::align_tight, s_channel);
    mf.addGpuRegularSize(tPrmSrsRkhs_secondStageTwiddleFactors_grid2.desc().get_size_in_bytes());


    cudaStreamSynchronize(s_channel);

    std::memset(&static_params, 0, sizeof(static_params));
    static_params.srsFilterPrms.tPrmFocc_table.desc  = tPrmFocc_table.desc().handle();
    static_params.srsFilterPrms.tPrmFocc_table.pAddr = tPrmFocc_table.addr();
    static_params.srsFilterPrms.tPrmFocc_comb2_table.desc = tPrmFocc_comb2_table.desc().handle();
    static_params.srsFilterPrms.tPrmFocc_comb2_table.pAddr = tPrmFocc_comb2_table.addr();
    static_params.srsFilterPrms.tPrmFocc_comb4_table.desc = tPrmFocc_comb4_table.desc().handle();
    static_params.srsFilterPrms.tPrmFocc_comb4_table.pAddr = tPrmFocc_comb4_table.addr(); 
    static_params.srsFilterPrms.tPrmW_comb2_nPorts1_wide.desc = tPrmW_comb2_nPorts1_wide.desc().handle();
    static_params.srsFilterPrms.tPrmW_comb2_nPorts1_wide.pAddr = tPrmW_comb2_nPorts1_wide.addr(); 
    static_params.srsFilterPrms.tPrmW_comb2_nPorts2_wide.desc = tPrmW_comb2_nPorts2_wide.desc().handle();
    static_params.srsFilterPrms.tPrmW_comb2_nPorts2_wide.pAddr = tPrmW_comb2_nPorts2_wide.addr();
    static_params.srsFilterPrms.tPrmW_comb2_nPorts4_wide.desc = tPrmW_comb2_nPorts4_wide.desc().handle();
    static_params.srsFilterPrms.tPrmW_comb2_nPorts4_wide.pAddr = tPrmW_comb2_nPorts4_wide.addr();
    static_params.srsFilterPrms.tPrmW_comb2_nPorts8_wide.desc = tPrmW_comb2_nPorts8_wide.desc().handle();
    static_params.srsFilterPrms.tPrmW_comb2_nPorts8_wide.pAddr = tPrmW_comb2_nPorts8_wide.addr();
    static_params.srsFilterPrms.tPrmW_comb4_nPorts1_wide.desc = tPrmW_comb4_nPorts1_wide.desc().handle();
    static_params.srsFilterPrms.tPrmW_comb4_nPorts1_wide.pAddr = tPrmW_comb4_nPorts1_wide.addr();
    static_params.srsFilterPrms.tPrmW_comb4_nPorts2_wide.desc = tPrmW_comb4_nPorts2_wide.desc().handle();
    static_params.srsFilterPrms.tPrmW_comb4_nPorts2_wide.pAddr = tPrmW_comb4_nPorts2_wide.addr();
    static_params.srsFilterPrms.tPrmW_comb4_nPorts4_wide.desc = tPrmW_comb4_nPorts4_wide.desc().handle();
    static_params.srsFilterPrms.tPrmW_comb4_nPorts4_wide.pAddr = tPrmW_comb4_nPorts4_wide.addr();
    static_params.srsFilterPrms.tPrmW_comb4_nPorts6_wide.desc = tPrmW_comb4_nPorts6_wide.desc().handle();
    static_params.srsFilterPrms.tPrmW_comb4_nPorts6_wide.pAddr = tPrmW_comb4_nPorts6_wide.addr();
    static_params.srsFilterPrms.tPrmW_comb4_nPorts12_wide.desc = tPrmW_comb4_nPorts12_wide.desc().handle();
    static_params.srsFilterPrms.tPrmW_comb4_nPorts12_wide.pAddr = tPrmW_comb4_nPorts12_wide.addr();
    static_params.srsFilterPrms.tPrmW_comb2_nPorts1_narrow.desc = tPrmW_comb2_nPorts1_narrow.desc().handle();
    static_params.srsFilterPrms.tPrmW_comb2_nPorts1_narrow.pAddr = tPrmW_comb2_nPorts1_narrow.addr();
    static_params.srsFilterPrms.tPrmW_comb2_nPorts2_narrow.desc = tPrmW_comb2_nPorts2_narrow.desc().handle();
    static_params.srsFilterPrms.tPrmW_comb2_nPorts2_narrow.pAddr = tPrmW_comb2_nPorts2_narrow.addr();    
    static_params.srsFilterPrms.tPrmW_comb2_nPorts4_narrow.desc = tPrmW_comb2_nPorts4_narrow.desc().handle();
    static_params.srsFilterPrms.tPrmW_comb2_nPorts4_narrow.pAddr = tPrmW_comb2_nPorts4_narrow.addr();
    static_params.srsFilterPrms.tPrmW_comb2_nPorts8_narrow.desc = tPrmW_comb2_nPorts8_narrow.desc().handle();
    static_params.srsFilterPrms.tPrmW_comb2_nPorts8_narrow.pAddr = tPrmW_comb2_nPorts8_narrow.addr();
    static_params.srsFilterPrms.tPrmW_comb4_nPorts1_narrow.desc = tPrmW_comb4_nPorts1_narrow.desc().handle();
    static_params.srsFilterPrms.tPrmW_comb4_nPorts1_narrow.pAddr = tPrmW_comb4_nPorts1_narrow.addr();
    static_params.srsFilterPrms.tPrmW_comb4_nPorts2_narrow.desc = tPrmW_comb4_nPorts2_narrow.desc().handle();
    static_params.srsFilterPrms.tPrmW_comb4_nPorts2_narrow.pAddr = tPrmW_comb4_nPorts2_narrow.addr();    
    static_params.srsFilterPrms.tPrmW_comb4_nPorts4_narrow.desc = tPrmW_comb4_nPorts4_narrow.desc().handle();
    static_params.srsFilterPrms.tPrmW_comb4_nPorts4_narrow.pAddr = tPrmW_comb4_nPorts4_narrow.addr();
    static_params.srsFilterPrms.tPrmW_comb4_nPorts6_narrow.desc = tPrmW_comb4_nPorts6_narrow.desc().handle();
    static_params.srsFilterPrms.tPrmW_comb4_nPorts6_narrow.pAddr = tPrmW_comb4_nPorts6_narrow.addr();
    static_params.srsFilterPrms.tPrmW_comb4_nPorts12_narrow.desc = tPrmW_comb4_nPorts12_narrow.desc().handle();
    static_params.srsFilterPrms.tPrmW_comb4_nPorts12_narrow.pAddr = tPrmW_comb4_nPorts12_narrow.addr();
    static_params.srsFilterPrms.noisEstDebias_comb2_nPorts1 = fInput.open_dataset("srsNoiseEstDebiasPrms")[0]["noisEstDebias_comb2_nPorts1"].as<float>();
    static_params.srsFilterPrms.noisEstDebias_comb2_nPorts2 = fInput.open_dataset("srsNoiseEstDebiasPrms")[0]["noisEstDebias_comb2_nPorts2"].as<float>();
    static_params.srsFilterPrms.noisEstDebias_comb2_nPorts4 = fInput.open_dataset("srsNoiseEstDebiasPrms")[0]["noisEstDebias_comb2_nPorts4"].as<float>();
    static_params.srsFilterPrms.noisEstDebias_comb2_nPorts8 = fInput.open_dataset("srsNoiseEstDebiasPrms")[0]["noisEstDebias_comb2_nPorts8"].as<float>();
    static_params.srsFilterPrms.noisEstDebias_comb4_nPorts1 = fInput.open_dataset("srsNoiseEstDebiasPrms")[0]["noisEstDebias_comb4_nPorts1"].as<float>();
    static_params.srsFilterPrms.noisEstDebias_comb4_nPorts2 = fInput.open_dataset("srsNoiseEstDebiasPrms")[0]["noisEstDebias_comb4_nPorts2"].as<float>();
    static_params.srsFilterPrms.noisEstDebias_comb4_nPorts4 = fInput.open_dataset("srsNoiseEstDebiasPrms")[0]["noisEstDebias_comb4_nPorts4"].as<float>();
    static_params.srsFilterPrms.noisEstDebias_comb4_nPorts6 = fInput.open_dataset("srsNoiseEstDebiasPrms")[0]["noisEstDebias_comb4_nPorts6"].as<float>();
    static_params.srsFilterPrms.noisEstDebias_comb4_nPorts12 = fInput.open_dataset("srsNoiseEstDebiasPrms")[0]["noisEstDebias_comb4_nPorts12"].as<float>();

    for(int i = 0; i < srsRkhsPrms.nGridSizes; i++)
    {
        srsRkhsPrms.pRkhsGridPrms[i].nEigs = fInput.open_dataset("rkhsGridPrms")[i]["nEigs"].as<uint8_t>();
        srsRkhsPrms.pRkhsGridPrms[i].gridSize = fInput.open_dataset("rkhsGridPrms")[i]["gridSize"].as<uint16_t>();
        srsRkhsPrms.pRkhsGridPrms[i].zpGridSize = fInput.open_dataset("rkhsGridPrms")[i]["zpGridSize"].as<uint16_t>();
    }
    srsRkhsPrms.pRkhsGridPrms[0].eigenValues.desc = tPrmSrsRkhs_eigValues_grid0.desc().handle();
    srsRkhsPrms.pRkhsGridPrms[0].eigenValues.pAddr = tPrmSrsRkhs_eigValues_grid0.addr();
    srsRkhsPrms.pRkhsGridPrms[1].eigenValues.desc = tPrmSrsRkhs_eigValues_grid1.desc().handle();
    srsRkhsPrms.pRkhsGridPrms[1].eigenValues.pAddr = tPrmSrsRkhs_eigValues_grid1.addr();
    srsRkhsPrms.pRkhsGridPrms[2].eigenValues.desc = tPrmSrsRkhs_eigValues_grid2.desc().handle();
    srsRkhsPrms.pRkhsGridPrms[2].eigenValues.pAddr = tPrmSrsRkhs_eigValues_grid2.addr();

    srsRkhsPrms.pRkhsGridPrms[0].eigenCorr.desc = tPrmSrsRkhs_eigenCorr_grid0.desc().handle();
    srsRkhsPrms.pRkhsGridPrms[0].eigenCorr.pAddr = tPrmSrsRkhs_eigenCorr_grid0.addr();
    srsRkhsPrms.pRkhsGridPrms[1].eigenCorr.desc = tPrmSrsRkhs_eigenCorr_grid1.desc().handle();
    srsRkhsPrms.pRkhsGridPrms[1].eigenCorr.pAddr = tPrmSrsRkhs_eigenCorr_grid1.addr();
    srsRkhsPrms.pRkhsGridPrms[2].eigenCorr.desc = tPrmSrsRkhs_eigenCorr_grid2.desc().handle();
    srsRkhsPrms.pRkhsGridPrms[2].eigenCorr.pAddr = tPrmSrsRkhs_eigenCorr_grid2.addr();

    srsRkhsPrms.pRkhsGridPrms[0].eigenVecs.desc = tPrmSrsRkhs_eigenVecs_grid0.desc().handle();
    srsRkhsPrms.pRkhsGridPrms[0].eigenVecs.pAddr = tPrmSrsRkhs_eigenVecs_grid0.addr();
    srsRkhsPrms.pRkhsGridPrms[1].eigenVecs.desc = tPrmSrsRkhs_eigenVecs_grid1.desc().handle();
    srsRkhsPrms.pRkhsGridPrms[1].eigenVecs.pAddr = tPrmSrsRkhs_eigenVecs_grid1.addr();
    srsRkhsPrms.pRkhsGridPrms[2].eigenVecs.desc = tPrmSrsRkhs_eigenVecs_grid2.desc().handle();
    srsRkhsPrms.pRkhsGridPrms[2].eigenVecs.pAddr = tPrmSrsRkhs_eigenVecs_grid2.addr();
    srsRkhsPrms.pRkhsGridPrms[0].secondStageTwiddleFactors.desc = tPrmSrsRkhs_secondStageTwiddleFactors_grid0.desc().handle();
    srsRkhsPrms.pRkhsGridPrms[0].secondStageTwiddleFactors.pAddr = tPrmSrsRkhs_secondStageTwiddleFactors_grid0.addr();
    srsRkhsPrms.pRkhsGridPrms[1].secondStageTwiddleFactors.desc = tPrmSrsRkhs_secondStageTwiddleFactors_grid1.desc().handle();
    srsRkhsPrms.pRkhsGridPrms[1].secondStageTwiddleFactors.pAddr = tPrmSrsRkhs_secondStageTwiddleFactors_grid1.addr();
    srsRkhsPrms.pRkhsGridPrms[2].secondStageTwiddleFactors.desc = tPrmSrsRkhs_secondStageTwiddleFactors_grid2.desc().handle();
    srsRkhsPrms.pRkhsGridPrms[2].secondStageTwiddleFactors.pAddr = tPrmSrsRkhs_secondStageTwiddleFactors_grid2.addr();
    
    srsRkhsPrms.pRkhsGridPrms[0].secondStageFourierPerm.desc = tPrmSrsRkhs_secondStageFourierPerm_grid0.desc().handle();
    srsRkhsPrms.pRkhsGridPrms[0].secondStageFourierPerm.pAddr = tPrmSrsRkhs_secondStageFourierPerm_grid0.addr();
    srsRkhsPrms.pRkhsGridPrms[1].secondStageFourierPerm.desc = tPrmSrsRkhs_secondStageFourierPerm_grid1.desc().handle();
    srsRkhsPrms.pRkhsGridPrms[1].secondStageFourierPerm.pAddr = tPrmSrsRkhs_secondStageFourierPerm_grid1.addr();
    srsRkhsPrms.pRkhsGridPrms[2].secondStageFourierPerm.desc = tPrmSrsRkhs_secondStageFourierPerm_grid2.desc().handle();
    srsRkhsPrms.pRkhsGridPrms[2].secondStageFourierPerm.pAddr = tPrmSrsRkhs_secondStageFourierPerm_grid2.addr();
    read_tv = true;
}

int PhySrsAggr::createPhyObj() 
{
    PhyDriverCtx * pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    Cell* cell_list[MAX_CELLS_PER_SLOT];
    uint32_t cellCount = 0;
    NVLOGI_FMT(TAG, "PhySrsAggr{}: createPhyObj", this_id);

    setCtx();
    /* Cleanup previous obj and re-create the cuPHY obj */
    CvSrsChestMemBank = pdctx->getCvSrsChestMemoryBank();
    if(CvSrsChestMemBank == nullptr)
        PHYDRIVER_THROW_EXCEPTIONS(-1, "CvSrsChestMemBank is null");

    pdctx->getCellList(cell_list,&cellCount);
    if(cellCount == 0)
        return EINVAL;

    for(uint32_t i = 0; i < cellCount; i++)
    {
        auto& cell_ptr = cell_list[i];
        int tmp_cell_id = cell_ptr->getPhyId();

        // Add only active cells here
        if(tmp_cell_id == DEFAULT_PHY_CELL_ID)
            continue;

        if(static_params_cell.size() > 0)
        {
            auto it = std::find_if(
                static_params_cell.begin(), static_params_cell.end(),
                [&tmp_cell_id](cuphyCellStatPrm_t _cellStatPrm) { return (_cellStatPrm.phyCellId == tmp_cell_id); }
            );

            if(it != static_params_cell.end())
                continue;
        }

        cell_id_list.push_back(tmp_cell_id);

        // set static cell paramaters
        cellStatPrm.phyCellId = cell_ptr->getPhyId();
        cellStatPrm.nRxAnt    = cell_ptr->getRxAnt();
        cellStatPrm.nRxAntSrs = cell_ptr->getRxAntSrs();
        cellStatPrm.nTxAnt    = cell_ptr->getTxAnt();
        cellStatPrm.nPrbUlBwp = cell_ptr->getPrbUlBwp();
        cellStatPrm.nPrbDlBwp = cell_ptr->getPrbDlBwp();
        cellStatPrm.mu        = cell_ptr->getMu();

        static_params_cell.push_back(cellStatPrm);

        tvStatPrms(cell_ptr->getTvSrsH5File(), static_params_cell.size() - 1);
    }

    static_params.nMaxCells            = static_params_cell.size();
    static_params.nMaxCellsPerSlot     = static_params_cell.size();
    static_params.pCellStatPrms        = static_cast<cuphyCellStatPrm_t*>(static_params_cell.data());
    /* 
     * Create cuPHY object only if the desider number of cells
     * has been activated into cuphydriver.
     */
    if((int)static_params_cell.size() == (int)pdctx->getCellGroupNum())
    {
        static_params.pOutInfo = &cuphy_tracker;
        static_params.pDbg = &srsDbgPrms;
        static_params.pStatDbg = &srsStatDbgPrms;
        
#ifdef SRS_H5DUMP
        std::string dbg_output_file = std::string("/tmp/SRS_Debug") + std::to_string(id) + std::string(".h5");
        debugFileH.reset(new hdf5hpp::hdf5_file(hdf5hpp::hdf5_file::create(dbg_output_file.c_str())));
        static_params.pDbg->pOutFileName             = dbg_output_file.c_str();
        std::string stat_dbg_output_file = std::string("/tmp/SRS_Stat_Debug") + std::to_string(id) + std::string(".h5");
        debugFileH.reset(new hdf5hpp::hdf5_file(hdf5hpp::hdf5_file::create(stat_dbg_output_file.c_str())));
        static_params.pStatDbg->pOutFileName             = stat_dbg_output_file.c_str();
        static_params.pStatDbg->enableApiLogging = 0;
#else
        static_params.pDbg->pOutFileName = nullptr;
        static_params.pStatDbg->pOutFileName = nullptr;
        static_params.pStatDbg->enableApiLogging = 0;
#endif
        static_params.pSrsRkhsPrms = &srsRkhsPrms;
        static_params.chEstAlgo = pdctx->get_srs_chest_algo_type();
        static_params.enableDelayOffsetCorrection = 1;
        static_params.chEstToL2NormalizationAlgo = pdctx->get_srs_chest_tol2_normalization_algo_type();
        static_params.chEstToL2ConstantScaler = pdctx->get_srs_chest_tol2_constant_scaler();
        static_params.enableBatchedMemcpy = pdctx->getUseBatchedMemcpy();
        int cuda_strm_prio = 0;
        CUDA_CHECK_PHYDRIVER(cudaStreamGetPriority(s_channel, &cuda_strm_prio));

        cuphyStatus_t createStatus = cuphyCreateSrsRx(&srsRxHndl, &static_params, s_channel);
        std::string cuphy_ch_create_name = "cuphyCreateSrsRx";            
        checkPhyChannelObjCreationError(createStatus,cuphy_ch_create_name);

        //pCuphyTracker = (const cuphyMemoryFootprint*)cuphyGetMemoryFootprintTrackerPrachRx(handle);
        pCuphyTracker = reinterpret_cast<const cuphyMemoryFootprint*>(static_params.pOutInfo->pMemoryFootprint);
        //pCuphyTracker->printMemoryFootprint();

        gDev->synchronizeStream(s_channel);
    }
    else if(static_params_cell.size() > pdctx->getCellGroupNum())
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, " Adding more cells then expected ({})", pdctx->getCellGroupNum());
        return -1;
    }

    //printStaticApiPrms(&static_params);

    return 0;
}

bool PhySrsAggr::validateOutput()
{
    int errorCount = 0;

#if 0
    for(int x = 0; x < buf_sz; x++)
    {
        if(buf_h[x] != originalInputData_h[x])
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Byte {}) new: {:02X} orig: {:02X}", x,(buf_h[x], originalInputData_h[x]));
            errorCount++;
        }
    }

    if(errorCount > 0)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Error Count is {}", errorCount);
        return false;
    }

    NVLOGI_FMT(TAG, "NO ERRORS in ordered input");
#endif
    return true;
}

slot_command_api::srs_params* PhySrsAggr::getDynParams()
{
    return aggr_slot_params->cgcmd->srs.get();
}


int PhySrsAggr::setup(
    const std::vector<Cell *>& aggr_cell_list,
    const std::vector<ULInputBuffer *>& aggr_ulbuf_st2
)
{
    PhyDriverCtx * pdctx    = StaticConversion<PhyDriverCtx>(this->getPhyDriverHandler()).get();
    #if 1
        CVSrsChestBuff * cv_ptr = nullptr;
    #else
        CVBuffer * cv_ptr = nullptr;
    #endif
    t_ns          t1        = Time::nowNs();
    //TBD: consider a list of buffers, one per cell. Here support is limited to a single cell
    cuphyStatus_t setupStatus;

    slot_command_api::srs_params* pparms = getDynParams();

    // Log Dyn API parameters default to 0
    dyn_params.pCellGrpDynPrm = &pparms->cell_grp_info;
    //printParametersAggr(pdctx, &pparms->cell_grp_info);

    if(aggr_cell_list.size() == 0)
        return -1;

    if(aggr_ulbuf_st2.size() == 0)
        return -1;

    for(int idx = 0; idx < aggr_ulbuf_st2.size(); idx++)
    {
        int count = -1;
        if(aggr_ulbuf_st2[idx] != nullptr)
        {
            auto phyCellId = aggr_cell_list[idx]->getPhyId();
            for(uint32_t dyn_idx=0; dyn_idx < dyn_params.pCellGrpDynPrm->nCells; dyn_idx++)
            {
                if(static_params_cell[dyn_params.pCellGrpDynPrm->pCellPrms[dyn_idx].cellPrmStatIdx].phyCellId == phyCellId)
                {
                    count = dyn_params.pCellGrpDynPrm->pCellPrms[dyn_idx].cellPrmDynIdx;
                    break;
                }
            }
            if(count != -1)
            {
                DataIn.pTDataRx[count].pAddr = aggr_ulbuf_st2[idx]->getBufD();
                aggr_cell_list[idx]->setSrsDynPrmIndex(count);
            }
            //else
                //NVLOGI_FMT(TAG, "PhyPuschAggr::setup - Cell {} has no PUSCH",phyCellId);
        }
    }

    dyn_params.cpuCopyOn  = 1;
    dyn_params.cuStream   = s_channel;

    t_ns t2 = Time::nowNs();

    setCtx();

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////// PUSCH Setup PHASE 1: get HARQ buffers sizes
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct slot_command_api::slot_indication* si = aggr_slot_params->si;
    procModeBmsk = pdctx->getEnableUlCuphyGraphs() ? SRS_PROC_MODE_FULL_SLOT_GRAPHS : SRS_PROC_MODE_FULL_SLOT;
    dyn_params.procModeBmsk = procModeBmsk;
    dyn_params.pDynDbg = &srsDynDbgPrms;
    dyn_params.pDynDbg->enableApiLogging=0;
    uint8_t countCell = 0;
    uint8_t countSrsUes = 1;
    for (int k=0; k < dyn_params.pCellGrpDynPrm->nSrsUes; k++)
    {
        uint32_t cell_idx = pparms->srs_ue_per_cell[countCell].cell_idx;
        if (pparms->srs_ue_per_cell[countCell].num_srs_ues == countSrsUes)
        {
            countCell++;
            countSrsUes = 1;
        }
        else
        {
            countSrsUes++;
        }
        uint32_t rnti = dyn_params.pCellGrpDynPrm->pUeSrsPrms[k].rnti;
        uint32_t bufferIdx = dyn_params.pCellGrpDynPrm->pUeSrsPrms[k].srsChestBufferIndexL2;
        uint32_t usage = dyn_params.pCellGrpDynPrm->pUeSrsPrms[k].usage;
        cv_ptr = nullptr;

        if(CvSrsChestMemBank->preAllocateBuffer(cell_idx,
                                        rnti,
                                        bufferIdx,
                                        usage,
                                        (CVSrsChestBuff**)&cv_ptr))
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "bucketAllocateBuffer returned error ");
            return -1;
        }
        else
        {
            uint16_t prgSize = dyn_params.pCellGrpDynPrm->pUeSrsPrms[k].prgSize;
            if(prgSize > 4)
            {
                prgSize = 2; 
            }
            cv_ptr->configSrsInfo(pparms->dl_ul_bwp_max_prg[cell_idx],
                                  pparms->nGnbAnt, 
                                  CV_NUM_UE_LAYER, 
                                  prgSize, 
                                  dyn_params.pCellGrpDynPrm->pUeSrsPrms[k].srsStartPrg,
                                  dyn_params.pCellGrpDynPrm->pUeSrsPrms[k].startValidPrg,
                                  dyn_params.pCellGrpDynPrm->pUeSrsPrms[k].nValidPrg);

            DataOut.pChEstBuffInfo[k].tChEstBuffer.desc  = cv_ptr->getSrsDescr();
            DataOut.pChEstBuffInfo[k].tChEstBuffer.pAddr = cv_ptr->getAddr();
            dyn_params.pCellGrpDynPrm->pUeSrsPrms[k].chEstBuffIdx = k; 


#if 0
            cuphyDataType_t dtype;
            int rank;
            vec<int, CUPHY_DIM_MAX> dimensions;
            vec<int, CUPHY_DIM_MAX> strides;
            cuphyStatus_t s = cuphyGetTensorDescriptor(DataOut.pChEstBuffInfo[k].tChEstBuffer.desc,
                                                    CUPHY_DIM_MAX,
                                                    &dtype,
                                                    &rank,
                                                    dimensions.begin(),
                                                    strides.begin());

            NVLOGD_FMT(TAG, "{}:dimensions[0]={}, dimensions[1]={}, dimensions[2]={}, strides[0]={}, strides[1]={}, strides[2]={}",
                                __func__, dimensions[0], dimensions[1], dimensions[2], strides[0], strides[1], strides[2]);
#endif
            NVLOGD_FMT(TAG, "PhySRSAggr{}: SFN {}.{} allocated CV buffer {} {} {} rnti {} usage {} srsPrgSize {} srsStartPrg {} cell_id {} bufferIdx {}",
                this_id, si->sfn_, si->slot_,
                k, reinterpret_cast<void*>(cv_ptr), dyn_params.pDataOut->pChEstBuffInfo[k].tChEstBuffer.pAddr,
                rnti,
                usage,
                dyn_params.pCellGrpDynPrm->pUeSrsPrms[k].prgSize,
                dyn_params.pCellGrpDynPrm->pUeSrsPrms[k].srsStartPrg,
                cell_idx,
                bufferIdx
            );
        }
        dyn_params.pDataOut->pSrsChEstToL2[k].pChEstCpuBuff = pparms->srs_chest_buffer[k];
        //NVLOGD_FMT(TAG, "PhySrsAggr pSrsChEstToL2[{}]={}",k,dyn_params.pDataOut->pSrsChEstToL2[k].pChEstCpuBuff); 
    }
    CUDA_CHECK_PHYDRIVER(cudaEventRecord(start_setup, s_channel));
    setupStatus = cuphySetupSrsRx(srsRxHndl, &(dyn_params), batchPrmHndl);
    if(setupStatus != CUPHY_STATUS_SUCCESS)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHY_API_EVENT, "cuphySetupSrshRx returned error {}", static_cast<int>(setupStatus));
        CUDA_CHECK_PHYDRIVER(cudaEventRecord(end_setup, s_channel));
        return -1;
    }
    CUDA_CHECK_PHYDRIVER(cudaEventRecord(end_setup, s_channel));

    NVLOGD_FMT(TAG, "PhySrsAggr{} SFN {}.{} setup", this_id, si->sfn_, si->slot_);
    //printStaticApiPrms(&static_params);
    return 0;
}

int PhySrsAggr::run()
{
    PhyDriverCtx * pdctx    = StaticConversion<PhyDriverCtx>(this->getPhyDriverHandler()).get();
    int ret=0;
    
    setCtx();

    #ifdef PUSCH_INPUT_BUFFER_DEBUG
        CUDA_CHECK_PHYDRIVER(cudaMemcpyAsync(buf_h, buf_d, buf_sz, cudaMemcpyDefault, s_channel));
    #endif

    CUDA_CHECK_PHYDRIVER(cudaEventRecord(start_run, s_channel));
    if((getSetupStatus() == CH_SETUP_DONE_NO_ERROR))
    {
        cuphyStatus_t runStatus = cuphyRunSrsRx(srsRxHndl, procModeBmsk);
        if(runStatus != CUPHY_STATUS_SUCCESS)
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHY_API_EVENT, "cuphyRunSRSRx returned error {}", static_cast<int>(runStatus));
            ret=-1;
        }    
    }
    {
        MemtraceDisableScope md;
        CUDA_CHECK_PHYDRIVER(cudaEventRecord(end_run, s_channel));
    }

    t_ns t3 = Time::nowNs();
    NVLOGD_FMT(TAG, "PhySrsAggr{} ru", this_id);

    return ret;
}


int PhySrsAggr::validate()
{
    PhyDriverCtx* pdctx    = StaticConversion<PhyDriverCtx>(this->getPhyDriverHandler()).get();
#if 0
    uint32_t* pCbCrcs    = static_cast<uint32_t*>(bCbCrcs.addr());
    uint32_t* pTbCrcs    = static_cast<uint32_t*>(bTbCrcs.addr());
    uint8_t*  pEstBytes  = static_cast<uint8_t*>(bTbPayloads.addr());
    uint8_t*  pTrueBytes = static_cast<uint8_t*>(tTbBytes.addr());

    for(uint32_t i = 0; i < totNumCbCrc; ++i)
    {
        if(pCbCrcs[i] != 0)
        {
            nCbCrcErrors += 1;
            NVLOGD_FMT(TAG, "SFN {}.{} PUSCH obj pCbCrcs[{}] = {}", aggr_slot_params->si->sfn_, aggr_slot_params->si->slot_, i, pCbCrcs[i]);
        }
    }

    for(uint32_t i = 0; i < totNumTbCrc; ++i)
    {
        if(pTbCrcs[i] != 0)
        {
            nTbCrcErrors += 1;
            NVLOGD_FMT(TAG, "SFN {}.{} PUSCH obj pTbCrcs[{}] = {}", aggr_slot_params->si->sfn_, aggr_slot_params->si->slot_, i, pTbCrcs[i]);
        }
    }

    // Byte error is not useful
    // for(uint32_t i = 0; i < totNumTbByte && i < tTbBytes.desc().get_size_in_bytes()-4; ++i)
    // {
    //     if(pEstBytes[i] != pTrueBytes[i])
    //         nTbByteErrors += 1;
    // }

    // NVLOGI_FMT(TAG, "SFN {}.{} Cell {} PUSCH obj TB CRC err: {}, CB CRC err: {}, byte err: {}",
    //     aggr_slot_params->si->sfn_, aggr_slot_params->si->slot_, cell_ptr->getPhyId(),
    //     nTbCrcErrors, nCbCrcErrors, nTbByteErrors);

    NVLOGI_FMT(TAG, "SFN {}.{} PUSCH obj TB CRC err: {}, CB CRC err: {}",
            aggr_slot_params->si->sfn_, aggr_slot_params->si->slot_,
            nTbCrcErrors, nCbCrcErrors);
#endif
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////// PUSCH Debug output 1 slot only
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifdef SRS_H5DUMP
    CuphyOAM *oam = CuphyOAM::getInstance();
    bool triggerH5Dump = true;
/*    
    if((nTbCrcErrors != 0 || nCbCrcErrors != 0) && h5dumped == false)
    {
        if(oam->puschH5DumpMutex.try_lock())
        {
            if(oam->puschH5dumpNextCrc.load() && !oam->puschH5dumpInProgress.load() && h5dumped == false)
            {
                oam->puschH5dumpInProgress.store(true);
                NVLOGC_FMT(TAG,"CRC error encountered SFN {}.{} Stored puschH5dumpInProgress true",  aggr_slot_params->si->sfn_, aggr_slot_params->si->slot_);
                triggerH5Dump = true;
                h5dumped = true;
            }
            oam->puschH5DumpMutex.unlock();
        }
    }
*/
    if(triggerH5Dump)
    {
        NVLOGC_FMT(TAG, "SFN {}.{} Generating H5 Debug SRS file {}", aggr_slot_params->si->sfn_, aggr_slot_params->si->slot_, std::to_string(id).c_str());
        auto& stream = s_channel;
        cudaStreamSynchronize(stream);
        cuphyStatus_t debugStatus = cuphyWriteDbgBufSynchSrs(srsRxHndl, stream);
        if(debugStatus != CUPHY_STATUS_SUCCESS)
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHY_API_EVENT, "cuphyWriteDbgBufSynchSrs returned error {}", debugStatus);
            return -1;
        }
        cudaStreamSynchronize(stream);
        debugFileH.get()->close();
        debugFileH.reset();
/*        
        NVLOGC_FMT(TAG, "SFN {}.{} Done Generating H5 Debug PUSCH {} file, please refer to the largest h5dump file created.", aggr_slot_params->si->sfn_, aggr_slot_params->si->slot_, std::to_string(id).c_str());
        oam->puschH5DumpMutex.lock();
        // oam->puschH5dumpNextCrc.store(false);
        // oam->puschH5dumpInProgress.store(false);
        oam->puschH5DumpMutex.unlock();
        NVLOGC_FMT(TAG, "SFN {}.{} Release H5Dump Lock, exitting forcefully, please expect a lot error messages!", aggr_slot_params->si->sfn_, aggr_slot_params->si->slot_);
*/        
        //EXIT_L1(EXIT_FAILURE);
    }
#endif
    return 0;
}



int PhySrsAggr::callback(const std::array<bool,UL_MAX_CELLS_PER_SLOT>& srs_order_cell_timeout_list)
{
    PhyDriverCtx*                       pdctx    = StaticConversion<PhyDriverCtx>(this->getPhyDriverHandler()).get();

    slot_command_api::ul_slot_callbacks ul_cb;
    struct slot_command_api::slot_indication* si = aggr_slot_params->si;
    NVLOGD_FMT(TAG, "PhySrsAggr{} SFN {}.{}: callback cnt_used={} ",
            this_id, si->sfn_, si->slot_, cnt_used);

    if(pdctx->getUlCb(ul_cb))
    {
        NVLOGD_FMT(TAG, "Calling UL Aggr callback");

        struct slot_command_api::ul_output_msg_buffer msg;
        msg.data_buf    = nullptr;
        msg.total_bytes = 0;
        msg.numTB       = 0;

        auto srs = getDynParams();

        // Notify DataLake that SRS data is ready. This only copies pointers and signals the worker thread.
        if(pdctx->getDataLake() != nullptr) {
            pdctx->getDataLake()->notifySrs(aggr_slot_params->si, srs, &DataOut, &static_params);
        }

        // Only for compilation to run pass actual cuphySrsDataOut_t* struct
        ul_cb.srs_cb_fn(ul_cb.srs_cb_context, msg, *(aggr_slot_params->si), *srs, &DataOut, &static_params, srs_order_cell_timeout_list);
    }

    return 0;
}

cuphySrsDynPrms_t *PhySrsAggr::getSrsDynParams()
{
    return &dyn_params;
}



void printParametersAggr(PhyDriverCtx* pdctx, cuphySrsCellGrpDynPrm_t* l2, cuphySrsCellGrpDynPrm_t* tv)
{
#if 0
    if (l2 == nullptr || tv == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Either L2 params and or TV params is null");
        return;
    }
    NVLOGI_FMT(TAG, "{}  L2: nCells:{} other: nCells {}", __FUNCTION__, l2->nCells, tv->ncells);
    for (uint16_t i = 0 ; i < l2->nCells; i++)
    {
        cuphySrsCellDynPrm_t* dyn = &l2->pCellPrms[i];
        cuphySrsCellDynPrm_t* dyn_tv = &tv->pCellPrms[i];

        NVLOGI_FMT(TAG, "{} L2: cellPrmStatIdx:{} other: cellPrmStatIdx:{}", __FUNCTION__, dyn->cellPrmStatIdx, dyn_tv->cellPrmStatIdx);
        NVLOGI_FMT(TAG, "{} L2: cellPrmDynIdx:{} other: cellPrmDynIdx:{}", __FUNCTION__, dyn->cellPrmDynIdx, dyn_tv->cellPrmDynIdx);
        NVLOGI_FMT(TAG, "{} L2: slotNum:{} other: slotNum:{}", __FUNCTION__, dyn->slotNum, dyn_tv->slotNum);
        NVLOGI_FMT(TAG, "{} L2: frameNum:{} other: frameNum:{}", __FUNCTION__, dyn->frameNum, dyn_tv->frameNum);
        NVLOGI_FMT(TAG, "{} L2: srsStartSym:{} other: srsStartSym:{}", __FUNCTION__, dyn->srsStartSym, dyn_tv->srsStartSym);
        NVLOGI_FMT(TAG, "{} L2: nSrsSym:{} other: nSrsSym:{}", __FUNCTION__, dyn->nSrsSym, dyn_tv->nSrsSym);
    }
    NVLOGI_FMT(TAG, "{}  L2: nSrsUes: {} other: nSrsUes:{}", __FUNCTION__, l2->nSrsUes, tv->nSrsUes);
    for (uint16_t i=0; i< l2->nSrsUes; i++)
    {
        cuphyUeSrsPrm_t* dyn = &l2->pUeSrsPrms[i];
        cuphyUeSrsPrm_t* dyn_tv = &tv->pUeSrsPrms[i];
        NVLOGI_FMT(TAG, "{} L2: cellIdx:{} other: cellIdx:{}", __FUNCTION__, dyn->cellIdx,dyn_tv->cellIdx );
        NVLOGI_FMT(TAG, "{} L2: nAntPorts:{} other: nAntPorts:{}", __FUNCTION__, dyn->nAntPorts, dyn_tv->nAntPorts);
        NVLOGI_FMT(TAG, "{} L2: nSyms:{} other: nSyms:{}", __FUNCTION__, dyn->nSyms, dyn_tv->nSyms);
        NVLOGI_FMT(TAG, "{} L2: nRepetitions:{} other: nRepetitions:{}", __FUNCTION__, dyn->nRepetitions, dyn_tv->nRepetitions);
        NVLOGI_FMT(TAG, "{} L2: combSize:{} other: combSize:{}", __FUNCTION__, dyn->combSize, dyn_tv->combSize);
        NVLOGI_FMT(TAG, "{} L2: startSym:{} other: startSym:{}", __FUNCTION__, dyn->startSym, dyn->startSym);
        NVLOGI_FMT(TAG, "{} L2: sequenceId:{} other: sequenceId:{}", __FUNCTION__, dyn->sequenceId, dyn_tv->sequenceId);
        NVLOGI_FMT(TAG, "{} L2: configIdx:{} other: configIdx:{}", __FUNCTION__, dyn->configIdx, dyn_tv->configIdx);
        NVLOGI_FMT(TAG, "{} L2: bandwidthIdx:{} other: bandwidthIdx:{}", __FUNCTION__, dyn->bandwidthIdx, dyn_tv->bandwidthIdx);
        NVLOGI_FMT(TAG, "{} L2: combOffset:{} other: combOffset:{}", __FUNCTION__, dyn->combOffset,dyn_tv->combOffset);
        NVLOGI_FMT(TAG, "{} L2: cyclicShift:{} other: cyclicShift:{}", __FUNCTION__, dyn->cyclicShift, dyn_tv->cyclicShift);
        NVLOGI_FMT(TAG, "{} L2: frequencyPosition:{} other: frequencyPosition:{}", __FUNCTION__, dyn->frequencyPosition, dyn_tv->frequencyPosition);
        NVLOGI_FMT(TAG, "{} L2: frequencyShift:{} other: frequencyShift:{}", __FUNCTION__, dyn->frequencyShift, dyn_tv->frequencyShift);
        NVLOGI_FMT(TAG, "{} L2: frequencyHopping:{} other: frequencyHopping:{}", __FUNCTION__, dyn->frequencyHopping, dyn_tv->frequencyHopping);
        NVLOGI_FMT(TAG, "{} L2: resourceType:{} other: resourceType:{}", __FUNCTION__, dyn->resourceType, dyn_tv->resourceType);
        NVLOGI_FMT(TAG, "{} L2: Tsrs:{} other: Tsrs:{}", __FUNCTION__, dyn->Tsrs, dyn_tvs->Tsrs);
        NVLOGI_FMT(TAG, "{} L2: Toffset:{} other: Toffset:{}", __FUNCTION__, dyn->Toffset, dyn_tv->Toffset);
        NVLOGI_FMT(TAG, "{} L2: groupOrSequenceHopping:{} other groupOrSequenceHopping:{}", __FUNCTION__, dyn->groupOrSequenceHopping, dyn_tv->groupOrSequenceHopping);
        NVLOGI_FMT(TAG, "{} L2: chEstBuffIdx:{}  other chEstBuffIdx:{}", __FUNCTION__, dyn->chEstBuffIdx, dyn_tv->chEstBuffIdx);
        NVLOGI_FMT(TAG, "{} L2: srsAntPortToUeAntMap[0]:{} other srsAntPortToUeAntMap[0]:{}", __FUNCTION__, dyn->srsAntPortToUeAntMap[0], dyn_tv->srsAntPortToUeAntMap[0]);
        NVLOGI_FMT(TAG, "{} L2: srsAntPortToUeAntMap[1]:{} other srsAntPortToUeAntMap[1]:{}", __FUNCTION__, dyn->srsAntPortToUeAntMap[1], dyn_tv->srsAntPortToUeAntMap[1]);
        NVLOGI_FMT(TAG, "{} L2: srsAntPortToUeAntMap[2]:{} other srsAntPortToUeAntMap[2]:{}", __FUNCTION__, dyn->srsAntPortToUeAntMap[2], dyn_tv->srsAntPortToUeAntMap[2]);
        NVLOGI_FMT(TAG, "{} L2: srsAntPortToUeAntMap[3]:{} other srsAntPortToUeAntMap[2]:{}", __FUNCTION__, dyn->srsAntPortToUeAntMap[3], dyn_tv->srsAntPortToUeAntMap[3]);
        NVLOGI_FMT(TAG, "{} L2: rnti:{} other rnti:{}", __FUNCTION__, dyn->rnti, dyn_tv->rnti);
        NVLOGI_FMT(TAG, "{} L2: handle: {} other handle{}", __FUNCTION__, dyn->handle, dyn_tv->handle);
        NVLOGI_FMT(TAG, "{} L2: usage:{} other usage {}", __FUNCTION__, dyn->usage, dyn_tv->handle);
    }

    NVLOGI_FMT(TAG, "===============================================");
#endif
return;
}

void PhySrsAggr::printStaticApiPrms(cuphySrsStatPrms_t const* pStaticPrms)
{
    printf("===============================================\n");
    printf("============print srsStaticApiPrms=============\n");
    printf("nMaxCells: %d\n", pStaticPrms->nMaxCells);
    printf("nMaxCellsPerSlot: %d\n", pStaticPrms->nMaxCellsPerSlot);
    printf("enableBatchedMemcpy: %d\n", pStaticPrms->enableBatchedMemcpy);
    const cuphyCellStatPrm_t* pCellStatPrms = nullptr;
    printf("===============================================\n");
    printf("cuphyCellStatPrm_t:\n");
    printf("===============================================\n");

    for(uint16_t i=0; i < pStaticPrms->nMaxCells; i++)
    {
        pCellStatPrms = &pStaticPrms->pCellStatPrms[i];
        printf("phyCellId: %d\n", pCellStatPrms->phyCellId);
        printf("nRxAnt: %d\n", pCellStatPrms->nRxAnt);
        printf("nRxAntSrs: %d\n", pCellStatPrms->nRxAntSrs);
        printf("nTxAnt: %d\n", pCellStatPrms->nTxAnt);
        printf("nPrbUlBwp: %d\n", pCellStatPrms->nPrbUlBwp);
        printf("nPrbDlBwp: %d\n", pCellStatPrms->nPrbDlBwp);
        printf("mu: %d\n", pCellStatPrms->mu);
    }
    printf("===============================================\n");
}

void printParametersAggr(PhyDriverCtx* pdctx, const cuphySrsCellGrpDynPrm_t* l2)
{
#if 1
    if (l2 == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "L2 params is null");
        return;
    }
    NVLOGI_FMT(TAG, "===============================================");
    NVLOGI_FMT(TAG, "{}  L2: nCells: {}", __FUNCTION__, l2->nCells);
    for (uint16_t i = 0 ; i < l2->nCells; i++)
    {
        cuphySrsCellDynPrm_t* dyn = &l2->pCellPrms[i];

        NVLOGI_FMT(TAG, "{} L2: cellPrmStatIdx:{}", __FUNCTION__, dyn->cellPrmStatIdx);
        NVLOGI_FMT(TAG, "{} L2: cellPrmDynIdx:{}", __FUNCTION__, dyn->cellPrmDynIdx);
        NVLOGI_FMT(TAG, "{} L2: slotNum:{}", __FUNCTION__, dyn->slotNum);
        NVLOGI_FMT(TAG, "{} L2: frameNum:{}", __FUNCTION__, dyn->frameNum);
        NVLOGI_FMT(TAG, "{} L2: srsStartSym:{}", __FUNCTION__, dyn->srsStartSym);
        NVLOGI_FMT(TAG, "{} L2: nSrsSym:{}", __FUNCTION__, dyn->nSrsSym);
    }
    NVLOGI_FMT(TAG, "{}  L2: nSrsUes: {}", __FUNCTION__, l2->nSrsUes);
    for (uint16_t i=0; i< l2->nSrsUes; i++)
    {
        cuphyUeSrsPrm_t* dyn = &l2->pUeSrsPrms[i];
        NVLOGI_FMT(TAG, "{} L2: cellIdx:{}", __FUNCTION__, dyn->cellIdx);
        NVLOGI_FMT(TAG, "{} L2: nAntPorts:{}", __FUNCTION__, dyn->nAntPorts);
        NVLOGI_FMT(TAG, "{} L2: nRepetitions:{}", __FUNCTION__, dyn->nRepetitions);
        NVLOGI_FMT(TAG, "{} L2: nSyms:{}", __FUNCTION__, dyn->nSyms);
        NVLOGI_FMT(TAG, "{} L2: combSize:{}", __FUNCTION__, dyn->combSize);
        NVLOGI_FMT(TAG, "{} L2: startSym:{}", __FUNCTION__, dyn->startSym);
        NVLOGI_FMT(TAG, "{} L2: sequenceId:{}", __FUNCTION__, dyn->sequenceId);
        NVLOGI_FMT(TAG, "{} L2: configIdx:{}", __FUNCTION__, dyn->configIdx);
        NVLOGI_FMT(TAG, "{} L2: bandwidthIdx:{}", __FUNCTION__, dyn->bandwidthIdx);
        NVLOGI_FMT(TAG, "{} L2: combOffset:{}", __FUNCTION__, dyn->combOffset);
        NVLOGI_FMT(TAG, "{} L2: cyclicShift:{}", __FUNCTION__, dyn->cyclicShift);
        NVLOGI_FMT(TAG, "{} L2: frequencyPosition:{}", __FUNCTION__, dyn->frequencyPosition);
        NVLOGI_FMT(TAG, "{} L2: frequencyShift:{}", __FUNCTION__, dyn->frequencyShift);
        NVLOGI_FMT(TAG, "{} L2: frequencyHopping:{}", __FUNCTION__, dyn->frequencyHopping);
        NVLOGI_FMT(TAG, "{} L2: resourceType:{}", __FUNCTION__, dyn->resourceType);
        NVLOGI_FMT(TAG, "{} L2: Tsrs:{}", __FUNCTION__, dyn->Tsrs);
        NVLOGI_FMT(TAG, "{} L2: Toffset:{}", __FUNCTION__, dyn->Toffset);
        NVLOGI_FMT(TAG, "{} L2: groupOrSequenceHopping:{}", __FUNCTION__, dyn->groupOrSequenceHopping);
        NVLOGI_FMT(TAG, "{} L2: chEstBuffIdx:{}", __FUNCTION__, dyn->chEstBuffIdx);
        NVLOGI_FMT(TAG, "{} L2: groupOrSequenceHopping:{}", __FUNCTION__, dyn->groupOrSequenceHopping);
        NVLOGI_FMT(TAG, "{} L2: chEstBuffIdx:{}", __FUNCTION__, dyn->chEstBuffIdx);
        NVLOGI_FMT(TAG, "{} L2: srsAntPortToUeAntMap[0]:{}", __FUNCTION__, dyn->srsAntPortToUeAntMap[0]);
        NVLOGI_FMT(TAG, "{} L2: srsAntPortToUeAntMap[1]:{}", __FUNCTION__, dyn->srsAntPortToUeAntMap[1]);
        NVLOGI_FMT(TAG, "{} L2: srsAntPortToUeAntMap[2]:{}", __FUNCTION__, dyn->srsAntPortToUeAntMap[2]);
        NVLOGI_FMT(TAG, "{} L2: srsAntPortToUeAntMap[3]:{}", __FUNCTION__, dyn->srsAntPortToUeAntMap[3]);
        NVLOGI_FMT(TAG, "{} L2: rnti:{}", __FUNCTION__, dyn->rnti);
        NVLOGI_FMT(TAG, "{} L2: handle:{}", __FUNCTION__, dyn->handle);
        NVLOGI_FMT(TAG, "{} L2: usage:{}", __FUNCTION__, dyn->usage);
    }
    NVLOGI_FMT(TAG, "===============================================");
#endif
}

void PhySrsAggr::updatePhyCellId(uint16_t phyCellId_old,uint16_t phyCellId_new)
{
    for(uint32_t i=0; i < static_params_cell.size(); i++)
    {
        if(static_params_cell[i].phyCellId == phyCellId_old)
        {
            static_params_cell[i].phyCellId = phyCellId_new;
            break;
        }
    }
}

int PhySrsAggr::cleanup()
{
    PhyChannel::cleanup();
    return 0;
}
