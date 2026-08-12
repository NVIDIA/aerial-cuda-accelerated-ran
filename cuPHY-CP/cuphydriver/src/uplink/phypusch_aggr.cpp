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

#define TAG (NVLOG_TAG_BASE_CUPHY_DRIVER + 23) // "DRV.PUSCH"

#include "phypusch_aggr.hpp"
#include "cuphydriver_api.hpp"
#include "context.hpp"
#include "nvlog.hpp"
#include "exceptions.hpp"
#include "cuda_events.hpp"
#include "cuphyoam.hpp"
#include "cuphy.h"

// #define PUSCH_INPUT_BUFFER_DEBUG
// #define PUSCH_H5DUMP

void printParametersAggr(PhyDriverCtx* pdctx, const cuphyPuschCellGrpDynPrm_t* l2);

PhyPuschAggr::PhyPuschAggr(
    phydriver_handle _pdh,
    GpuDevice*       _gDev,
    cudaStream_t*     _s_channels,
    MpsCtx *        _mpsCtx) :
    PhyChannel(_pdh, _gDev, 0, _s_channels[0], _mpsCtx)
{
    PhyDriverCtx * pdctx = StaticConversion<PhyDriverCtx>(_pdh).get();

    mf.init(_pdh, std::string("PhyPuschAggr"), sizeof(PhyPuschAggr));
    cuphyMf.init(_pdh, std::string("cuphyPuschRx"), 0);

    channel_type = slot_command_api::channel_type::PUSCH;
    channel_name.assign("PUSCH");


    NVLOGI_FMT(TAG, "PhyPuschAggr{}: construct", this_id);

    //-------------------------------------------------------------
    // Check for configuration information in the input file. Newer
    // input files will have configuration values in the file, so
    // that they don't need to be specified on the command line.
    cuphy::disable_hdf5_error_print(); // Temporarily disable HDF5 stderr printing

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////// Init PUSCH
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    try
    {
        // read output bits for testing validation
        static_params_dbg.descrmOn     = 1;
        static_params_dbg.enableApiLogging = 0;
        static_params_dbg.forcedNumCsi2Bits = pdctx->getForcedNumCsi2Bits();

        // initialize output buffers
        bStartOffsetsCbCrc     = std::move(cuphy::buffer<uint32_t, cuphy::pinned_alloc>(MAX_N_TBS_PER_CELL_GROUP_SUPPORTED));
        bStartOffsetsTbCrc     = std::move(cuphy::buffer<uint32_t, cuphy::pinned_alloc>(MAX_N_TBS_PER_CELL_GROUP_SUPPORTED));
        bStartOffsetsTbPayload = std::move(cuphy::buffer<uint32_t, cuphy::pinned_alloc>(MAX_N_TBS_PER_CELL_GROUP_SUPPORTED));

        nUciPayloadBytes = MAX_N_PRBS_SUPPORTED * MAX_N_BBU_LAYERS_PUSCH_SUPPORTED * OFDM_SYMBOLS_PER_SLOT * CUPHY_N_TONES_PER_PRB * CUPHY_QAM_256;
        nUciUes = CUPHY_MAX_N_UCI_ON_PUSCH;
        nUciSegs = nUciUes * 3;
        nCsi2Bits = CUPHY_MAX_N_CSI2_WORDS*32;
        //UCI on PUSCH --> FIXME sizes!
        bUciPayloads          = std::move(cuphy::buffer<uint8_t, cuphy::pinned_alloc>(nUciPayloadBytes));
        bUciCrcFlags          = std::move(cuphy::buffer<uint8_t, cuphy::pinned_alloc>(nUciSegs));
        //bNumCsi2Bits          = std::move(cuphy::buffer<uint16_t, cuphy::pinned_alloc>(nCsi2Bits));
        bNumCsi2Bits          = std::move(cuphy::buffer<uint16_t, cuphy::pinned_alloc>(CUPHY_MAX_N_PUSCH_CSI2));
        bUciOnPuschOutOffsets = std::move(cuphy::buffer<cuphyUciOnPuschOutOffsets_t, cuphy::pinned_alloc>(nUciUes));

        //FIXME: MAX NUM UES?
        totNumTbCrc = PHY_PUSCH_MAX_MIMO * PHY_PUSCH_MAX_FREQ_MULTI; // 16 x 1
        totNumCbCrc = PHY_PUSCH_MAX_CB_PER_TB * totNumTbCrc; // 148 x totNumTbCrc
        totNumTbByte = PHY_PUSCH_MAX_BYTES_PER_TB * totNumTbCrc; // 311386 x totNumTbCrc

        //Overalloate max UEs x max CELLs
        totNumCbCrc *= UL_MAX_CELLS_PER_SLOT;
        totNumTbCrc *= UL_MAX_CELLS_PER_SLOT;
        totNumTbByte *= UL_MAX_CELLS_PER_SLOT;

        int totFhDataSize = (ORAN_MAX_PRB * CUPHY_N_TONES_PER_PRB) * OFDM_SYMBOLS_PER_SLOT * MAX_AP_PER_SLOT * UL_MAX_CELLS_PER_SLOT;
        // Don't allocate the memory if we're not going to use it
        if(pdctx->datalake_enabled()) {
            bDataRx     = std::move(cuphy::buffer<__half2, cuphy::pinned_alloc>(totFhDataSize));
        }
        else {
            bDataRx     = std::move(cuphy::buffer<__half2, cuphy::pinned_alloc>(0));
        }

        // Channel estimates buffer: all UE groups concatenated.
        // bChannelEsts: worst-case for full cell bandwidth (groups FDM-partition PRBs, total is constant).
        // bChannelEstSizes: one uint32_t per group.
        int totHestDataSize = MAX_AP_PER_SLOT * MAX_N_BBU_LAYERS_PUSCH_SUPPORTED * (ORAN_MAX_PRB * CUPHY_N_TONES_PER_PRB) * OFDM_SYMBOLS_PER_SLOT;
        if(pdctx->datalake_enabled()) {
            bChannelEsts = std::move(cuphy::buffer<float2, cuphy::pinned_alloc>(totHestDataSize));
            bChannelEstSizes = std::move(cuphy::buffer<uint32_t, cuphy::pinned_alloc>(MAX_N_USER_GROUPS_SUPPORTED));
            std::memset(bChannelEstSizes.addr(), 0, MAX_N_USER_GROUPS_SUPPORTED * sizeof(uint32_t));
        }
        else {
            bChannelEsts = std::move(cuphy::buffer<float2, cuphy::pinned_alloc>(0));
            bChannelEstSizes = std::move(cuphy::buffer<uint32_t, cuphy::pinned_alloc>(0));
        }
        bCbCrcs     = std::move(cuphy::buffer<uint32_t, cuphy::pinned_alloc>(totNumCbCrc));
        bTbCrcs     = std::move(cuphy::buffer<uint32_t, cuphy::pinned_alloc>(totNumTbCrc));
        bTbPayloads = std::move(cuphy::buffer<uint8_t, cuphy::pinned_alloc>(totNumTbByte));
        bTaEst      = std::move(cuphy::buffer<float, cuphy::pinned_alloc>(totNumTbCrc));
        bRsrp       = std::move(cuphy::buffer<float, cuphy::pinned_alloc>(MAX_N_TBS_SUPPORTED));
        bRssi       = std::move(cuphy::buffer<float, cuphy::pinned_alloc>(MAX_N_TBS_SUPPORTED));
        bSinr       = std::move(cuphy::buffer<float, cuphy::pinned_alloc>(MAX_N_TBS_SUPPORTED));
        bCfo        = std::move(cuphy::buffer<float, cuphy::pinned_alloc>(MAX_N_TBS_SUPPORTED));
        bNoiseIntfVar = std::move(cuphy::buffer<float, cuphy::pinned_alloc>(MAX_N_TBS_SUPPORTED));
        bHarqDetectionStatus = std::move(cuphy::buffer<uint8_t, cuphy::pinned_alloc>(MAX_N_TBS_SUPPORTED));
        bCsiP1DetectionStatus = std::move(cuphy::buffer<uint8_t, cuphy::pinned_alloc>(MAX_N_TBS_SUPPORTED));
        bCsiP2DetectionStatus = std::move(cuphy::buffer<uint8_t, cuphy::pinned_alloc>(MAX_N_TBS_SUPPORTED));

        clearUciFlags(false);

        nCbCrcErrors  = 0;
        nTbCrcErrors  = 0;
        nTbByteErrors = 0;
        okTimeout = false;
        memset(bCbCrcs.addr(), 0, sizeof(uint32_t) * totNumCbCrc);
        memset(bTbCrcs.addr(), 1, sizeof(uint32_t) * totNumTbCrc);

        // PuschRx::copyOutputToCPU() copies out of memory if this address is valid
        if(pdctx->datalake_enabled()) {
            DataOut.pDataRx            = bDataRx.addr();
        } else {
            DataOut.pDataRx            = nullptr;
        }

        // Channel estimates API pointers
        if(pdctx->datalake_enabled()) {
            DataOut.pChannelEsts       = bChannelEsts.addr();
            DataOut.pChannelEstSizes   = bChannelEstSizes.addr();
        } else {
            DataOut.pChannelEsts       = nullptr;
            DataOut.pChannelEstSizes   = nullptr;
        }
        DataOut.pCbCrcs                = bCbCrcs.addr();
        DataOut.pTbCrcs                = bTbCrcs.addr();
        DataOut.pTbPayloads            = bTbPayloads.addr();
        DataOut.pStartOffsetsCbCrc     = bStartOffsetsCbCrc.addr();
        DataOut.pStartOffsetsTbCrc     = bStartOffsetsTbCrc.addr();
        DataOut.pStartOffsetsTbPayload = bStartOffsetsTbPayload.addr();
        DataOut.totNumCbs              = totNumCbCrc;
        DataOut.totNumTbs              = totNumTbCrc;
        DataOut.totNumPayloadBytes     = totNumTbByte;
        DataOut.pTaEsts                = bTaEst.addr();
        DataOut.pRsrp                  = bRsrp.addr();
        DataOut.pRssi                  = bRssi.addr();
        if(pdctx->getPuschSinr() == 1)
        {
            DataOut.pSinrPreEq             = NULL;
            DataOut.pSinrPostEq            = bSinr.addr();
            DataOut.pNoiseVarPreEq         = NULL;
            DataOut.pNoiseVarPostEq        = bNoiseIntfVar.addr();
            NVLOGI_FMT(TAG, "pSinrPostEq={} pNoiseVarPostEq={}",reinterpret_cast<void*>(DataOut.pSinrPostEq),reinterpret_cast<void*>(DataOut.pNoiseVarPostEq));
        }
        else if(pdctx->getPuschSinr() == 2)
        {
            DataOut.pSinrPreEq             = bSinr.addr();
            DataOut.pSinrPostEq            = NULL;
            DataOut.pNoiseVarPreEq         = bNoiseIntfVar.addr();
            DataOut.pNoiseVarPostEq        = NULL;
            NVLOGI_FMT(TAG, "pSinrPreEq={} pNoiseVarPreEq={}",reinterpret_cast<void*>(DataOut.pSinrPreEq),reinterpret_cast<void*>(DataOut.pNoiseVarPreEq));
        }
    
        DataOut.pCfoHz                 = bCfo.addr();
        DataOut.HarqDetectionStatus    = bHarqDetectionStatus.addr();
        DataOut.CsiP1DetectionStatus   = bCsiP1DetectionStatus.addr();
        DataOut.CsiP2DetectionStatus   = bCsiP2DetectionStatus.addr();

        DataOut.pUciPayloads           = bUciPayloads.addr();
        DataOut.pUciCrcFlags           = bUciCrcFlags.addr();
        DataOut.pNumCsi2Bits           = bNumCsi2Bits.addr();
        DataOut.pUciOnPuschOutOffsets  = bUciOnPuschOutOffsets .addr();

        // NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "totNumTbs={}", DataOut.totNumTbs);
        bHarqBufferSizeInBytes.resize(DataOut.totNumTbs);
        DataOut.h_harqBufferSizeInBytes = bHarqBufferSizeInBytes.data();
        cudaError_t status = cudaHostAlloc(&DataInOut.pHarqBuffersInOut, sizeof(uint8_t*)*DataOut.totNumTbs, cudaHostAllocPortable | cudaHostAllocMapped);
        if (status != cudaSuccess)
        {
            NVLOGC_FMT(TAG, "Failure with cudaHostAlloc {} for pHarqBuffersInOut", +status);
            EXIT_L1(EXIT_FAILURE);
        }
        status = cudaHostAlloc(&DataInOut.pFoCompensationBuffersInOut, sizeof(float*)*DataOut.totNumTbs, cudaHostAllocPortable | cudaHostAllocMapped);
        if (status != cudaSuccess)
        {
            NVLOGC_FMT(TAG, "Failure with cudaHostAlloc {} for pFoCompensationBuffersInOut", +status);
            EXIT_L1(EXIT_FAILURE);
        }
        
        CUDA_CHECK(cudaHostAlloc((void **)&(pPreEarlyHarqWaitKernelStatus), sizeof(uint8_t), cudaHostAllocPortable | cudaHostAllocMapped));
        CUDA_CHECK(cudaHostAlloc((void **)&(pPostEarlyHarqWaitKernelStatus), sizeof(uint8_t), cudaHostAllocPortable | cudaHostAllocMapped));
        
        CUDA_CHECK(cudaHostGetDevicePointer((void **)&(DataOut.pPreEarlyHarqWaitKernelStatusGpu), (void *)(pPreEarlyHarqWaitKernelStatus), 0));
        CUDA_CHECK(cudaHostGetDevicePointer((void **)&(DataOut.pPostEarlyHarqWaitKernelStatusGpu), (void *)(pPostEarlyHarqWaitKernelStatus), 0));

        // Host-pinned memory buffer holding early exit information. Initially  memset to 0.
        // TODO contents will be reset to 0 via PhyPuschAggr::cleanup() on UL slot map release.
        bWorkCancelInfo = std::move(cuphy::buffer<uint8_t, cuphy::pinned_alloc>(1));
        memset(bWorkCancelInfo.addr(), 0, sizeof(uint8_t));

        gDev->synchronizeStream(s_channel);
    }
    PHYDRIVER_CATCH_THROW_EXCEPTIONS();

    cellGrpDynPrm.nCells = 0;
    cellGrpDynPrm.pCellPrms = (cuphyPuschCellDynPrm_t*) calloc(UL_MAX_CELLS_PER_SLOT, sizeof(cuphyPuschCellDynPrm_t));
    cellGrpDynPrm.nUeGrps = 0;
    cellGrpDynPrm.pUeGrpPrms = (cuphyPuschUeGrpPrm_t*) calloc(UL_MAX_CELLS_PER_SLOT, sizeof(cuphyPuschUeGrpPrm_t));
    cellGrpDynPrm.nUes = 0;
    cellGrpDynPrm.pUePrms = (cuphyPuschUePrm_t*) calloc(UL_MAX_CELLS_PER_SLOT, sizeof(cuphyPuschUePrm_t));
    dyn_params.pCellGrpDynPrm = &cellGrpDynPrm;

    DataIn.pTDataRx = (cuphyTensorPrm_t*) calloc(UL_MAX_CELLS_PER_SLOT, sizeof(cuphyTensorPrm_t));

    // Data IN
    for(int idx = 0; idx < UL_MAX_CELLS_PER_SLOT; idx++)
    {
/*
        tDataRxInput[idx] = std::move(cuphy::tensor_device(CUPHY_C_16F, ORAN_MAX_PRB * CUPHY_N_TONES_PER_PRB,
                                                            OFDM_SYMBOLS_PER_SLOT, MAX_AP_PER_SLOT,
                                                            cuphy::tensor_flags::align_tight));
*/
        pusch_data_rx_desc[idx] = {CUPHY_C_16F, static_cast<int>(ORAN_MAX_PRB * CUPHY_N_TONES_PER_PRB), static_cast<int>(OFDM_SYMBOLS_PER_SLOT), static_cast<int>(MAX_AP_PER_SLOT), cuphy::tensor_flags::align_tight};

        DataIn.pTDataRx[idx].desc = pusch_data_rx_desc[idx].handle();
        DataIn.pTDataRx[idx].pAddr = nullptr;
    }
    dyn_params.pDataIn = &DataIn;

    // Data OUT
    dyn_params.pDataOut = &DataOut;  // (cuphyPuschDataOut_t *) calloc(UL_MAX_CELLS_PER_SLOT, sizeof(cuphyPuschDataOut_t));
    dyn_params.pDataInOut = &DataInOut;

    statusOut = {cuphyPuschStatusType_t::CUPHY_PUSCH_STATUS_SUCCESS_OR_UNTRACKED_ISSUE, MAX_UINT16, MAX_UINT16};
    dyn_params.pStatusOut = &statusOut;

    cuphy::enable_hdf5_error_print(); // Re-enable HDF5 stderr printing

    //Sub-slot Processing specific    
    sym_ord_done_sig_arr.reset(gDev->newGDRbuf(ORAN_PUSCH_SYMBOLS_X_SLOT * sizeof(uint32_t)));
    mf.addGpuPinnedSize(sym_ord_done_sig_arr->size_alloc);

    CUDA_CHECK_PHYDRIVER(cudaEventCreate(&start_setup_ph1));
    CUDA_CHECK_PHYDRIVER(cudaEventCreate(&end_setup_ph1));
    CUDA_CHECK_PHYDRIVER(cudaEventCreate(&start_setup_ph2));
    CUDA_CHECK_PHYDRIVER(cudaEventCreate(&end_setup_ph2));
    CUDA_CHECK_PHYDRIVER(cudaEventCreate(&start_crc));
    CUDA_CHECK_PHYDRIVER(cudaEventCreate(&end_crc));
    CUDA_CHECK_PHYDRIVER(cudaEventCreate(&subSlotCompletedEvent));
    CUDA_CHECK_PHYDRIVER(cudaEventCreate(&waitCompletedSubSlotEvent));
    CUDA_CHECK_PHYDRIVER(cudaEventCreate(&waitCompletedFullSlotEvent));
    CUDA_CHECK_PHYDRIVER(cudaEventCreate(&start_run_ph1));
    CUDA_CHECK_PHYDRIVER(cudaEventCreate(&end_run_ph1));
    CUDA_CHECK_PHYDRIVER(cudaEventCreate(&start_run_ph2));
    CUDA_CHECK_PHYDRIVER(cudaEventCreate(&end_run_ph2));

    static_params_cell.clear();
    cell_id_list.clear();

    launch_kernel_warmup(s_channel);
    launch_kernel_order(s_channel, 1, 0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, 0, 0, 0, 0, nullptr, 0, 0, 0, 0, 0);
    launch_kernel_order(s_channel, 1, 0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, 0, 0, 0, 0, nullptr, 0, 0, 0, 0, 0);
    gDev->synchronizeStream(s_channel);

    procModeBmsk = PUSCH_PROC_MODE_FULL_SLOT;
#if 0
    if(pdctx->getEnableUlCuphyGraphs())
    {
        procModeBmsk = PUSCH_PROC_MODE_FULL_SLOT_GRAPHS;

        //////////////////////////////////////////////////////////////////////////
        /////// Temporary WAR for PUSCH with cuPHY Graphs
        //////////////////////////////////////////////////////////////////////////
        cuphyPuschDynPrms_t dyn_prms;
        dyn_prms.procModeBmsk = procModeBmsk;
        dyn_prms.pCellGrpDynPrm = &cellGrpDynPrm;
        tDataRx.desc    = tDataRxInput[0].desc().handle();
        tDataRx.pAddr   = buf_d;
        DataIn.pTDataRx   = &tDataRx;
        dyn_prms.pDataIn   = &DataIn;
        dyn_prms.pDataOut  = &DataOut;
        dyn_prms.cpuCopyOn = 1;
        dyn_prms.phase1Stream = s_channel;
        cuphyStatus_t setupStatus = cuphySetupPuschRx(puschRxHndl, &(dyn_prms), batchPrmHndl);
        if(setupStatus != CUPHY_STATUS_SUCCESS)
            NVLOGE_FMT(TAG, AERIAL_CUPHY_API_EVENT, "cuphySetupPuschRx returned error {}", setupStatus);
        cudaStreamSynchronize(s_channel);
        //////////////////////////////////////////////////////////////////////////
    }
#endif

    read_tv = false;
    puschRxHndl = nullptr;
    hq_buffer_counter = 0;
    released_harq_buffer_info.num_released_harq_buffers = 0;
    released_harq_buffer_info.released_harq_buffer_list.reserve(pdctx->getMaxHarqPools());
};

PhyPuschAggr::~PhyPuschAggr()
{
    if(puschRxHndl)
        cuphyDestroyPuschRx(puschRxHndl);

    cudaFreeHost(DataInOut.pHarqBuffersInOut);
    cudaFreeHost(DataInOut.pFoCompensationBuffersInOut);
    cudaFreeHost(pPreEarlyHarqWaitKernelStatus);
    cudaFreeHost(pPostEarlyHarqWaitKernelStatus);

    CUDA_CHECK_PHYDRIVER(cudaEventDestroy(start_setup_ph1));
    CUDA_CHECK_PHYDRIVER(cudaEventDestroy(end_setup_ph1));
    CUDA_CHECK_PHYDRIVER(cudaEventDestroy(start_setup_ph2));
    CUDA_CHECK_PHYDRIVER(cudaEventDestroy(end_setup_ph2));
    CUDA_CHECK_PHYDRIVER(cudaEventDestroy(start_crc));
    CUDA_CHECK_PHYDRIVER(cudaEventDestroy(end_crc));
    CUDA_CHECK_PHYDRIVER(cudaEventDestroy(subSlotCompletedEvent));
    CUDA_CHECK_PHYDRIVER(cudaEventDestroy(waitCompletedSubSlotEvent));
    CUDA_CHECK_PHYDRIVER(cudaEventDestroy(waitCompletedFullSlotEvent));
    CUDA_CHECK_PHYDRIVER(cudaEventDestroy(start_run_ph1));
    CUDA_CHECK_PHYDRIVER(cudaEventDestroy(end_run_ph1));
    CUDA_CHECK_PHYDRIVER(cudaEventDestroy(start_run_ph2));
    CUDA_CHECK_PHYDRIVER(cudaEventDestroy(end_run_ph2));

    free(cellGrpDynPrm.pCellPrms);
    free(cellGrpDynPrm.pUeGrpPrms);
    free(cellGrpDynPrm.pUePrms);
    free(DataIn.pTDataRx);
    hq_buffer_counter = 0;
    released_harq_buffer_info.num_released_harq_buffers = 0;
    released_harq_buffer_info.released_harq_buffer_list.clear();
};

void PhyPuschAggr::tvStatPrms(const char* tv_h5, int cell_idx)
{
    PhyDriverCtx * pdctx = StaticConversion<PhyDriverCtx>(pdh).get();

    fInput = hdf5hpp::hdf5_file::open(tv_h5);

    cudaStreamSynchronize(s_channel);

    if(read_tv == true)
        return;

    // load filters (currently hardcoded to FP32). Load true TB data
    tWFreq      = cuphy::tensor_from_dataset(fInput.open_dataset("WFreq"), CUPHY_R_32F, cuphy::tensor_flags::align_tight, s_channel);
    mf.addGpuRegularSize(tWFreq.desc().get_size_in_bytes());
    tShiftSeq   = cuphy::tensor_from_dataset(fInput.open_dataset("ShiftSeq"), CUPHY_C_16F, cuphy::tensor_flags::align_tight, s_channel);
    mf.addGpuRegularSize(tShiftSeq.desc().get_size_in_bytes());
    tUnShiftSeq = cuphy::tensor_from_dataset(fInput.open_dataset("UnShiftSeq"), CUPHY_C_16F, cuphy::tensor_flags::align_tight, s_channel);
    mf.addGpuRegularSize(tUnShiftSeq.desc().get_size_in_bytes());
    if (fInput.is_valid_dataset("WFreq4")) {
        tWFreq4 = cuphy::tensor_from_dataset(fInput.open_dataset("WFreq4"), CUPHY_R_32F, cuphy::tensor_flags::align_tight, s_channel);
        mf.addGpuRegularSize(tWFreq4.desc().get_size_in_bytes());
    }
    else
        tWFreq4 = tWFreq;

    if (fInput.is_valid_dataset("WFreqSmall")) {
        tWFreqSmall = cuphy::tensor_from_dataset(fInput.open_dataset("WFreqSmall"), CUPHY_R_32F, cuphy::tensor_flags::align_tight, s_channel);
        mf.addGpuRegularSize(tWFreqSmall.desc().get_size_in_bytes());
    }
    else
        tWFreqSmall = tWFreq;

    if (fInput.is_valid_dataset("ShiftSeq4")) {
        tShiftSeq4 = cuphy::tensor_from_dataset(fInput.open_dataset("ShiftSeq4"), CUPHY_C_16F, cuphy::tensor_flags::align_tight, s_channel);
        mf.addGpuRegularSize(tShiftSeq4.desc().get_size_in_bytes());
    }
    else
        tShiftSeq4 = tShiftSeq;

    if (fInput.is_valid_dataset("UnShiftSeq4")) {
        tUnShiftSeq4 = cuphy::tensor_from_dataset(fInput.open_dataset("UnShiftSeq4"), CUPHY_C_16F, cuphy::tensor_flags::align_tight, s_channel);
        mf.addGpuRegularSize(tUnShiftSeq4.desc().get_size_in_bytes());
    }
    else
        tUnShiftSeq4 = tUnShiftSeq;

    cudaStreamSynchronize(s_channel);

    std::memset(&static_params, 0, sizeof(static_params));
    static_params.pWFreq        = &tPrmWFreq;
    static_params.pWFreq->desc  = tWFreq.desc().handle();
    static_params.pWFreq->pAddr = tWFreq.addr();

    static_params.pWFreq4        = &tPrmWFreq4;
    static_params.pWFreq4->desc  = tWFreq4.desc().handle();
    static_params.pWFreq4->pAddr = tWFreq4.addr();

    static_params.pWFreqSmall   = &tPrmWFreqSmall;
    static_params.pWFreqSmall->desc = tWFreqSmall.desc().handle();
    static_params.pWFreqSmall->pAddr = tWFreqSmall.addr();

    static_params.pShiftSeq        = &tPrmShiftSeq;
    static_params.pShiftSeq->desc  = tShiftSeq.desc().handle();
    static_params.pShiftSeq->pAddr = tShiftSeq.addr();

    static_params.pUnShiftSeq        = &tPrmUnShiftSeq;
    static_params.pUnShiftSeq->desc  = tUnShiftSeq.desc().handle();
    static_params.pUnShiftSeq->pAddr = tUnShiftSeq.addr();

    static_params.pShiftSeq4 = &tPrmShiftSeq4;
    static_params.pShiftSeq4->desc = tShiftSeq4.desc().handle();
    static_params.pShiftSeq4->pAddr = tShiftSeq4.addr();

    static_params.pUnShiftSeq4 = &tPrmUnShiftSeq4;
    static_params.pUnShiftSeq4->desc = tUnShiftSeq4.desc().handle();
    static_params.pUnShiftSeq4->pAddr = tUnShiftSeq4.addr();

    static_params.pDbg                 = &static_params_dbg;

    static_params.enableCfoCorrection     = pdctx->getPuschCfo();
    static_params.enableWeightedAverageCfo = pdctx->getEnableWeightedAverageCfo();
    static_params.enableToEstimation      = pdctx->getPuschTo();
    static_params.enableRssiMeasurement   = pdctx->getPuschRssi();
    static_params.enablePuschTdi          = pdctx->getPuschTdi();
    static_params.enableDftSOfdm          = pdctx->getPuschDftSOfdm();
    static_params.enableTbSizeCheck       = pdctx->getPuschTbSizeCheck();
    static_params.enableMassiveMIMO       = pdctx->getmMIMO_enable();
    static_params.enableEarlyHarq         = pdctx->getPuschEarlyHarqEn();
    static_params.enableDeviceGraphLaunch = pdctx->getPuschDeviceGraphLaunchEn();
    static_params.enableBatchedMemcpy     = pdctx->getUseBatchedMemcpy();
#ifdef SCF_FAPI_10_04
    static_params.enableCsiP2Fapiv3       = 1;
#else
    static_params.enableCsiP2Fapiv3       = 0;
#endif

    if(pdctx->getPuschSinr())
        static_params.enableSinrMeasurement = 1;
    else
        static_params.enableSinrMeasurement = 0;

    static_params.eqCoeffAlgo         = static_cast<cuphyPuschEqCoefAlgoType_t>(pdctx->getPuschEqCoeffAlgo());
    static_params.chEstAlgo           = static_cast<cuphyPuschChEstAlgoType_t>(pdctx->getPuschChEstAlgo());
    static_params.enablePerPrgChEst   = static_cast<uint8_t>(pdctx->getPuschEnablePerPrgChEst());
    static_params.pOutInfo = &cuphy_tracker;
    
    static_params.nMaxLdpcHetConfigs  = pdctx->getPuschMaxNumLdpcHetConfigs();
    static_params.nMaxTbPerNode       = pdctx->getPuschMaxNumTbPerNode();
    NVLOGC_FMT(TAG, "{}: PUSCH enableDeviceGraphLaunch={} enableCsiP2Fapiv3 = {} nMaxLdpcHetConfigs = {}", __func__, static_params.enableDeviceGraphLaunch, static_params.enableCsiP2Fapiv3, static_params.nMaxLdpcHetConfigs);
    /********************************************************/
    //For the first scheduler kernel (pre-early-HARQ) set timeout to 1100us which is >= 400us before T0 (which is PUSCH UL worker start) + 640us after T0 (~440us for symbol-3 arrival + 200us for reorder latency).
    //For the second scheduler kernel (post-early-HARQ) set timeout to 1500us which is >= 400us before T0 + 1100us after T0 (~800us for symbol-13 arrival + 200us reorder latency)
    NVLOGC_FMT(TAG, "Timeout values for wait kernels are {}us and {}us", pdctx->getPuschWaitTimeOutPreEarlyHarqUs(), pdctx->getPuschWaitTimeOutPostEarlyHarqUs());
    /********************************************************/
    
    read_tv = true;
}

int PhyPuschAggr::createPhyObj()
{
    PhyDriverCtx * pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    Cell* cell_list[MAX_CELLS_PER_SLOT];
    uint32_t cellCount = 0;
    NVLOGI_FMT(TAG, "PhyPuschAggr{}: createPhyObj", this_id);

    setCtx();
    /* Cleanup previous obj and re-create the cuPHY obj */
    hb_pool_m = pdctx->getHarPoolManager();
    if(hb_pool_m == nullptr)
        PHYDRIVER_THROW_EXCEPTIONS(-1, "HarqPoolManager is null");

    if (pdctx->getEnableWeightedAverageCfo()) {
        wavgcfo_pool_m = pdctx->getWAvgCfoPoolManager();
    // wavgcfo_pool_m can be null if weighted average CFO is not enabled
    } else {
        wavgcfo_pool_m = nullptr;
    }

    pdctx->getCellList(cell_list,&cellCount);
    if(cellCount == 0)
        return EINVAL;


    puschCellStatPrmsVec.reserve(cellCount);
    #ifdef SCF_FAPI_10_04
        csi2MapGpuBuffer = cuphy::make_unique_device<uint16_t>(cellCount * CUPHY_CSI2_SIZE_MAP_BUFFER_SIZE_PER_CELL);
        csi2MapParamsGpuBuffer = cuphy::make_unique_device<cuphyCsi2MapPrm_t> (cellCount * CUPHY_MAX_NUM_CSI2_SIZE_MAPS_PER_CELL);
        mf.addGpuRegularSize(sizeof(uint16_t) * cellCount * CUPHY_CSI2_SIZE_MAP_BUFFER_SIZE_PER_CELL);
        mf.addGpuRegularSize(sizeof(cuphyCsi2MapPrm_t) * cellCount * CUPHY_MAX_NUM_CSI2_SIZE_MAPS_PER_CELL);
    #endif 

    size_t mapOffset = 0, mapParamsOffset = 0;

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

        puschCellStatPrmsVec.push_back(*(cell_ptr->getPhyStatic()->pPuschCellStatPrms));

         auto& lastCell = puschCellStatPrmsVec.back();
#ifdef SCF_FAPI_10_04
           
        auto* mapBuf =  cell_ptr->getPhyStatic()->pPuschCellStatPrms->pCsi2MapBuffer;
        auto* mapParamBuf =  cell_ptr->getPhyStatic()->pPuschCellStatPrms->pCsi2MapPrm;
        auto lastVal = mapBuf[mapParamBuf[lastCell.nCsi2Maps - 1].csi2MapStartIdx + mapParamBuf[lastCell.nCsi2Maps - 1].csi2MapSize - 1];
        auto firstVal = mapBuf[mapParamBuf[lastCell.nCsi2Maps - 1].csi2MapStartIdx];

        NVLOGD_FMT(TAG , "firstVal 0x{:04X} lastVal 0x{:04X} csi2MapStartIdx 0x{:02X} csi2MapSize 0x{:02X} mapBuf[0] = 0x{:04X}  mapBuf[1] = 0x{:04X}", firstVal, lastVal, mapParamBuf[lastCell.nCsi2Maps - 1].csi2MapStartIdx, mapParamBuf[lastCell.nCsi2Maps - 1].csi2MapSize, mapBuf[0], mapBuf[1]);

        CUDA_CHECK_PHYDRIVER(cudaMemcpyAsync(
            lastCell.pCsi2MapBuffer + mapOffset, mapBuf,
            mapParamBuf[lastCell.nCsi2Maps - 1].csi2MapSize, cudaMemcpyHostToDevice, s_channel));

        CUDA_CHECK_PHYDRIVER(cudaMemcpyAsync(
            lastCell.pCsi2MapPrm + mapParamsOffset, mapParamBuf,
            lastCell.nCsi2Maps * sizeof(cuphyCsi2MapPrm_t) , cudaMemcpyHostToDevice, s_channel));
    
        mapOffset += CUPHY_CSI2_SIZE_MAP_BUFFER_SIZE_PER_CELL;
        mapParamsOffset += CUPHY_MAX_NUM_CSI2_SIZE_MAPS_PER_CELL;
#else 
        lastCell.pCsi2MapBuffer = nullptr;
        lastCell.pCsi2MapPrm = nullptr;
#endif

        cellStatPrm.pPuschCellStatPrms = &(puschCellStatPrmsVec[puschCellStatPrmsVec.size() -1]);

        static_params_cell.push_back(cellStatPrm);

        tvStatPrms(cell_ptr->getTvPuschH5File(), static_params_cell.size() - 1);

        static_params.nMaxPrb = cell_ptr->getPuschnMaxPrb();
        static_params.nMaxRx = cell_ptr->getPuschnMaxRx();
        
        NVLOGC_FMT(TAG, "static_params.nMaxPrb: {}", static_params.nMaxPrb);
        NVLOGC_FMT(TAG, "static_params.nMaxRx: {}", static_params.nMaxRx);

        if(i == 0)
        {
            //Assuming it's the same for all the cells
            uint8_t pusch_ldpc_max_num_itr_algo_type = cell_ptr->getPuschLdpcMaxNumItrAlgoType();
            static_params.ldpcMaxNumItrAlgo    = LDPC_MAX_NUM_ITR_ALGO_TYPE_LUT;
            if(pusch_ldpc_max_num_itr_algo_type==0)
            {
                static_params.ldpcMaxNumItrAlgo = LDPC_MAX_NUM_ITR_ALGO_TYPE_FIXED;
            }
            else if(pusch_ldpc_max_num_itr_algo_type==2)
            {
                static_params.ldpcMaxNumItrAlgo = LDPC_MAX_NUM_ITR_ALGO_TYPE_PER_UE;
            }
            static_params.fixedMaxNumLdpcItrs  = cell_ptr->getFixedMaxNumLdpcItrs();
            static_params.ldpcEarlyTermination = cell_ptr->getPuschLdpcEarlyTermination();
            static_params.ldpcAlgoIndex        = cell_ptr->getPuschLdpcAlgoIndex();
            static_params.ldpcFlags            = cell_ptr->getPuschLdpcFlags();
            static_params.ldpcUseHalf          = cell_ptr->getPuschLdpcUseHalf();
            static_params.ldpcClampValue       = 32.0f;
        }
    }

    static_params.nMaxCells            = static_params_cell.size();
    static_params.nMaxCellsPerSlot     = static_params_cell.size();
    static_params.pCellStatPrms        = static_cast<cuphyCellStatPrm_t*>(static_params_cell.data());
    static_params.polarDcdrListSz      = pdctx->getPuxchPolarDcdrListSz();
    static_params.subSlotCompletedEvent = subSlotCompletedEvent;
    static_params.waitCompletedSubSlotEvent = waitCompletedSubSlotEvent;
    static_params.waitCompletedFullSlotEvent = waitCompletedFullSlotEvent;
    static_params.pSymRxStatus        = (uint32_t*)sym_ord_done_sig_arr->addrd();
    static_params.puschrxChestFactorySettingsFilename = pdctx->getPuschrxChestFactorySettingsFilename().c_str();

    /*
     * Create cuPHY object only if the desired number of cells
     * has been activated into cuphydriver.
     */
    if((int)static_params_cell.size() == (int)pdctx->getCellGroupNum())
    {
#ifdef PUSCH_H5DUMP
        std::string dbg_output_file = std::string("PUSCH_Debug") + std::to_string(id) + std::string(".h5");
        debugFileH.reset(new hdf5hpp::hdf5_file(hdf5hpp::hdf5_file::create(dbg_output_file.c_str())));
        static_params.pDbg->pOutFileName             = dbg_output_file.c_str();
#else
        static_params_dbg.pOutFileName = nullptr;
#endif

        int cuda_strm_prio = 0;
        CUDA_CHECK_PHYDRIVER(cudaStreamGetPriority(s_channel, &cuda_strm_prio));
        static_params.stream_priority = cuda_strm_prio;
        static_params.ldpcKernelLaunch = PUSCH_RX_ENABLE_DRIVER_LDPC_LAUNCH;
        static_params.pWorkCancelInfo   = bWorkCancelInfo.addr();
        static_params.workCancelMode    = static_cast<cuphyPuschWorkCancelMode_t>(pdctx->getPuschWorkCancelMode()); // there was a check in yamlparser.cpp
        NVLOGI_FMT(TAG, "PUSCH workCancelMode set to {}", +static_params.workCancelMode);

        cuphyStatus_t createStatus = cuphyCreatePuschRx(&puschRxHndl, &static_params, s_channel);
        std::string cuphy_ch_create_name = "cuphyCreatePuschRx";            
        checkPhyChannelObjCreationError(createStatus,cuphy_ch_create_name);
        
        //pCuphyTracker = (const cuphyMemoryFootprint*)cuphyGetMemoryFootprintTrackerPuschRx(puschRxHndl);
        pCuphyTracker = reinterpret_cast<const cuphyMemoryFootprint*>(static_params.pOutInfo->pMemoryFootprint);
        //pCuphyTracker->printMemoryFootprint();

        gDev->synchronizeStream(s_channel);
    }
    else if(static_params_cell.size() > pdctx->getCellGroupNum())
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, " Adding more cells then expected ({})", pdctx->getCellGroupNum());
        return -1;
    }

    return 0;
}

bool PhyPuschAggr::validateOutput()
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
#endif
    if(errorCount > 0)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Error Count is {}", errorCount);
        return false;
    }

    NVLOGI_FMT(TAG, "NO ERRORS in ordered input");

    return true;
}

slot_command_api::pusch_params* PhyPuschAggr::getDynParams()
{
    return aggr_slot_params->cgcmd->pusch.get();
}


int PhyPuschAggr::setup(
    const std::vector<Cell *>& aggr_cell_list,
    const std::vector<ULInputBuffer *>& aggr_ulbuf_st1,
    cudaStream_t phase1_stream,
    cudaStream_t phase2_stream
)
{
    PhyDriverCtx * pdctx    = StaticConversion<PhyDriverCtx>(this->getPhyDriverHandler()).get();
    HarqBuffer * hb_ptr     = nullptr;
    t_ns          t1        = Time::nowNs();
    //TBD: consider a list of buffers, one per cell. Here support is limited to a single cell
    cuphyStatus_t setupStatus;

    slot_command_api::pusch_params* pparms = getDynParams();

    // Log Dyn API parameters default to 0
    dyn_params_dbg.enableApiLogging = 0;
    dyn_params.pDbg = &dyn_params_dbg;
    dyn_params.pCellGrpDynPrm = &pparms->cell_grp_info;

    if(aggr_cell_list.size() == 0)
        return -1;

    if(aggr_ulbuf_st1.size() == 0)
        return -1;

    for(int idx = 0; idx < aggr_ulbuf_st1.size(); idx++)
    {
        int count = -1;
        if(aggr_ulbuf_st1[idx] != nullptr)
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
                DataIn.pTDataRx[count].pAddr = aggr_ulbuf_st1[idx]->getBufD();
                aggr_cell_list[idx]->setPuschDynPrmIndex(aggr_slot_params->si->slot_, count);
                //NVLOGI_FMT(TAG, "PhyPuschAggr::setup - Cell {} cellPrmDynIdx {} ULBuffer {} at index {}",
                    //phyCellId,count,aggr_ulbuf_st1[idx]->getId(),idx);
            }
            //else
                //NVLOGI_FMT(TAG, "PhyPuschAggr::setup - Cell {} has no PUSCH",phyCellId);
        }
    }

    dyn_params.cpuCopyOn  = 1;

    dyn_params.phase1Stream = phase1_stream;
    dyn_params.phase2Stream = phase2_stream;

    t_ns t2 = Time::nowNs();

    setCtx();

    struct slot_command_api::slot_indication* si = aggr_slot_params->si;
    NVLOGD_FMT(TAG, "PhyPuschAggr{} SFN {}.{} setup", this_id, si->sfn_, si->slot_);
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////// PUSCH Setup PHASE 1: get HARQ buffers sizes
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    dyn_params.setupPhase = cuphyPuschSetupPhase_t::PUSCH_SETUP_PHASE_1;

    procModeBmsk = (pdctx->getEnableUlCuphyGraphs() ? PUSCH_PROC_MODE_FULL_SLOT_GRAPHS : PUSCH_PROC_MODE_FULL_SLOT) | PUSCH_PROC_MODE_SUB_SLOT;
    dyn_params.procModeBmsk = procModeBmsk;
    dyn_params.waitTimeOutPreEarlyHarqUs = pdctx->getPuschWaitTimeOutPreEarlyHarqUs(); 
    dyn_params.waitTimeOutPostEarlyHarqUs = pdctx->getPuschWaitTimeOutPostEarlyHarqUs(); 

    {
        MemtraceDisableScope md;
        CUDA_CHECK_PHYDRIVER(cudaEventRecord(start_setup_ph1, dyn_params.phase1Stream));
    }
    if(pdctx->getUseGreenContexts() == 0)
    {
        setupStatus = cuphySetupPuschRx(puschRxHndl, &(dyn_params), batchPrmHndl);
    }
    else
    {
        MemtraceDisableScope md;
        setupStatus = cuphySetupPuschRx(puschRxHndl, &(dyn_params), batchPrmHndl);
    }
    if(setupStatus != CUPHY_STATUS_SUCCESS)
    {
        if (dyn_params.pStatusOut->status == cuphyPuschStatusType_t::CUPHY_PUSCH_STATUS_UNSUPPORTED_MAX_ER_PER_CB)
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHY_API_EVENT, "SFN {}, slot {}: CUPHY_PUSCH_STATUS_UNSUPPORTED_MAX_ER_PER_CB Error in PUSCH_SETUP_PHASE_1 of cuphySetupPuschRx(): {}. Will not call cuphyRunPuschRx(). May be L2 misconfiguration. Triggered by TB {} in cell group and cellPrmStatIdx {}.", si->sfn_, si->slot_, cuphyGetErrorString(setupStatus), dyn_params.pStatusOut->ueIdx, dyn_params.pStatusOut->cellPrmStatIdx);
        }
        else if (dyn_params.pStatusOut->status == cuphyPuschStatusType_t::CUPHY_PUSCH_STATUS_TBSIZE_MISMATCH)
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHY_API_EVENT, "SFN {}, slot {}: CUPHY_PUSCH_STATUS_TBSIZE_MISMATCH Error in PUSCH_SETUP_PHASE_1 of cuphySetupPuschRx(): {}. Will not call cuphyRunPuschRx(). May be L2 misconfiguration. Triggered by TB {} in cell group and cellPrmStatIdx {}.", si->sfn_, si->slot_, cuphyGetErrorString(setupStatus), dyn_params.pStatusOut->ueIdx, dyn_params.pStatusOut->cellPrmStatIdx);
        }
        else
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHY_API_EVENT, "SFN {}, slot {}: Error in PUSCH_SETUP_PHASE_1 of cuphySetupPuschRx(): {}. Will not call cuphyRunPuschRx(). May be L2 misconfiguration.", si->sfn_, si->slot_, cuphyGetErrorString(setupStatus));
        }
        ////////TODO//////////////////////
        {
            MemtraceDisableScope md;
            CUDA_CHECK_PHYDRIVER(cudaEventRecord(end_setup_ph1, dyn_params.phase1Stream));
            CUDA_CHECK_PHYDRIVER(cudaEventRecord(start_setup_ph2, dyn_params.phase1Stream));
            CUDA_CHECK_PHYDRIVER(cudaEventRecord(end_setup_ph2, dyn_params.phase1Stream));
        }
        /////////////////////////////////
        return -1;
    }
#ifdef EARLY_UCI_CUBB_CALLFLOW_TEST
    NVLOGI_FMT(TAG,"Setting isEarlyHarqPresent to 1 for SFN{} , slot {}",si->sfn_,si->slot_);
    dyn_params.pDataOut->isEarlyHarqPresent=1;
#endif
    {
        MemtraceDisableScope md;
        CUDA_CHECK_PHYDRIVER(cudaEventRecord(end_setup_ph1, dyn_params.phase1Stream));
    }

    //struct slot_command_api::slot_indication* si = aggr_slot_params->si;
    //NVLOGD_FMT(TAG, "PhyPuschAggr{} SFN {}.{} setup", this_id, si->sfn_, si->slot_);

    // Allocate HARQ buffers based on the calculated requirements from setupPhase 1
    for (int k=0; k < dyn_params.pCellGrpDynPrm->nUes; k++)
    {
        cell_id_t cell_idx = dyn_params.pCellGrpDynPrm->pUePrms[k].pUeGrpPrm->pCellPrm->cellPrmStatIdx;
        if(dyn_params.pCellGrpDynPrm->pUePrms[k].pduBitmap & 0x01)
        {
            NVLOGI_FMT(TAG,"{}:Harq buffer allocation for UE {} Cell Stat Idx {} Cell Dyn Idx {} at SFN {} slot {}",
                       __func__,k,dyn_params.pCellGrpDynPrm->pUePrms[k].pUeGrpPrm->pCellPrm->cellPrmStatIdx,
                       dyn_params.pCellGrpDynPrm->pUePrms[k].pUeGrpPrm->pCellPrm->cellPrmDynIdx,si->sfn_,si->slot_);
            const uint8_t pusch_aggr_factor = pdctx->getPuschAggrFactor();
            bool is_bundled_pdu = (((dyn_params.pCellGrpDynPrm->pUePrms[k].pduBitmap & 0x10) != 0) && (pusch_aggr_factor > 1));
            //Allocate HARQ buffer only if PUSCH data exists
            if(hb_pool_m->bucketAllocateBuffer(
                                        (HarqBuffer**)&hb_ptr,
                                        dyn_params.pDataOut->h_harqBufferSizeInBytes[k],
                                        dyn_params.pCellGrpDynPrm->pUePrms[k].rnti,
                                        dyn_params.pCellGrpDynPrm->pUePrms[k].harqProcessId,
                                        &dyn_params.pCellGrpDynPrm->pUePrms[k].ndi,
                                        cell_idx,
                                        is_bundled_pdu,pusch_aggr_factor,si->sfn_,si->slot_))
            {
                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "bucketAllocateBuffer returned error ");
                return -1;
            }
            dyn_params.pDataInOut->pHarqBuffersInOut[k] = hb_ptr->getAddr();
            hb_ptr->setCellDynIdx((int)dyn_params.pCellGrpDynPrm->pUePrms[k].pUeGrpPrm->pCellPrm->cellPrmDynIdx);

            // Allocate WAvgCfo buffer if enabled
            if(wavgcfo_pool_m != nullptr)
            {
                WAvgCfoBuffer* cfo_ptr = wavgcfo_pool_m->allocate(
                    dyn_params.pCellGrpDynPrm->pUePrms[k].rnti,
                    static_cast<uint16_t>(cell_idx)
                );
                
                if(cfo_ptr != nullptr)
                {
                    dyn_params.pDataInOut->pFoCompensationBuffersInOut[k] = cfo_ptr->getAddr();   
                    cfo_ptr->setSfnSlot(si->sfn_, si->slot_);
                    cfo_ptr->setCellDynIdx(static_cast<int>(dyn_params.pCellGrpDynPrm->pUePrms[k].pUeGrpPrm->pCellPrm->cellPrmDynIdx));
                    cfo_ptr->setTimestampLastUsed();
                    
                    NVLOGD_FMT(TAG, "PhyPuschAggr{}: SFN {}.{} allocated WAvgCfo buffer {} for rnti {} cell_id {}",
                        this_id, si->sfn_, si->slot_,
                        reinterpret_cast<void*>(cfo_ptr),
                        dyn_params.pCellGrpDynPrm->pUePrms[k].rnti,
                        cell_idx
                    );
                }
                else
                {
                    NVLOGW_FMT(TAG, "PhyPuschAggr{}: SFN {}.{} failed to allocate WAvgCfo buffer for rnti {} cell_id {}",
                        this_id, si->sfn_, si->slot_,
                        dyn_params.pCellGrpDynPrm->pUePrms[k].rnti,
                        cell_idx
                    );
                    return -1;
                }
            }

            //This debug fmt log statement appears to cause large latency in phase 2 PUSCH setup.  Root cause has not been determined.  Commenting this out for now.
            // NVLOGD_FMT(TAG, "PhyPuschAggr{}: SFN {}.{} allocated buffer {} {} size {} rnti {} hPID {} ndi {} cell_id {} hq_buffer_counter {}",
            //        this_id, si->sfn_, si->slot_,
            //        k, reinterpret_cast<void*>(hb_ptr), dyn_params.pDataOut->h_harqBufferSizeInBytes[k],
            //        dyn_params.pCellGrpDynPrm->pUePrms[k].rnti,
            //        dyn_params.pCellGrpDynPrm->pUePrms[k].harqProcessId,
            //        dyn_params.pCellGrpDynPrm->pUePrms[k].ndi,
            //        cell_idx,
            //        hq_buffer_counter
            //    );

            if(hq_buffer_counter < MAX_N_TBS_PER_CELL_GROUP_SUPPORTED)
            {
                hb_slot[hq_buffer_counter++] = hb_ptr;
            }
            else
            {
                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "PhyPuschAggr{}: SFN {}.{} allocated buffer {} {} size {} rnti {} hPID {} ndi {} cell_id {} hq_buffer_counter {} hb_slot is full",
                   this_id, si->sfn_, si->slot_,
                   k, reinterpret_cast<void*>(hb_ptr), dyn_params.pDataOut->h_harqBufferSizeInBytes[k],
                   dyn_params.pCellGrpDynPrm->pUePrms[k].rnti,
                   dyn_params.pCellGrpDynPrm->pUePrms[k].harqProcessId,
                   dyn_params.pCellGrpDynPrm->pUePrms[k].ndi,
                   cell_idx,
                   hq_buffer_counter
               );
            }
        }
        else
        {
            //no HARQ buffer is allocated
            dyn_params.pDataInOut->pHarqBuffersInOut[k] = NULL;

            NVLOGD_FMT(TAG, "PhyPuschAggr{}: SFN {}.{} no HARQ buffer for ue-prm-idx {} rnti {} cell_id {}",
                   this_id, si->sfn_, si->slot_,k,dyn_params.pCellGrpDynPrm->pUePrms[k].rnti,cell_idx);
        }
    }
    printParametersAggr(pdctx, &pparms->cell_grp_info);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////// PUSCH Setup PHASE 2
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    dyn_params.setupPhase = cuphyPuschSetupPhase_t::PUSCH_SETUP_PHASE_2;
    {
        MemtraceDisableScope md;
        CUDA_CHECK_PHYDRIVER(cudaEventRecord(start_setup_ph2, dyn_params.phase1Stream));
    }
    // NVLOGE_FMT(TAG, AERIAL_CUDA_API_EVENT, "Calling Setup Phase 2");
    if(pdctx->getUseGreenContexts() == 0)
    {
        setupStatus = cuphySetupPuschRx(puschRxHndl, &(dyn_params), batchPrmHndl);
    }
    else
    {
        MemtraceDisableScope md;
        setupStatus = cuphySetupPuschRx(puschRxHndl, &(dyn_params), batchPrmHndl);
    }

    if(setupStatus != CUPHY_STATUS_SUCCESS)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHY_API_EVENT, "cuphySetupPuschRx PUSCH_SETUP_PHASE_2 returned error {}", setupStatus);
        return -1;
    }

    {
        MemtraceDisableScope md;
        CUDA_CHECK_PHYDRIVER(cudaEventRecord(end_setup_ph2, dyn_params.phase1Stream));
    }

    return 0;
}

int PhyPuschAggr::run(cuphyPuschRunPhase_t runPhase)
{
    PhyDriverCtx * pdctx    = StaticConversion<PhyDriverCtx>(this->getPhyDriverHandler()).get();
    int ret=0;
    
    setCtx();

    #ifdef PUSCH_INPUT_BUFFER_DEBUG
        CUDA_CHECK_PHYDRIVER(cudaMemcpyAsync(buf_h, buf_d, buf_sz, cudaMemcpyDefault, s_channel));
    #endif

    if(runPhase == cuphyPuschRunPhase_t::PUSCH_RUN_ALL_PHASES) {
        MemtraceDisableScope md;
        CUDA_CHECK_PHYDRIVER(cudaEventRecord(start_run, dyn_params.phase1Stream));
    }
    if(runPhase == cuphyPuschRunPhase_t::PUSCH_RUN_SUB_SLOT_PROC) {
        MemtraceDisableScope md;
        CUDA_CHECK_PHYDRIVER(cudaEventRecord(start_run, dyn_params.phase1Stream));
        CUDA_CHECK_PHYDRIVER(cudaEventRecord(start_run_ph1, dyn_params.phase1Stream));
    }
    if(runPhase == cuphyPuschRunPhase_t::PUSCH_RUN_FULL_SLOT_COPY) {
        //Second stream waits for PUSCH_RUN_EARLY_HARQ_PROC+PUSCH_RUN_FULL_SLOT_PROC to complete
        MemtraceDisableScope md;
        CUDA_CHECK_PHYDRIVER(cudaStreamWaitEvent(dyn_params.phase2Stream,end_run_ph1,0));
        CUDA_CHECK_PHYDRIVER(cudaEventRecord(start_run_ph2, dyn_params.phase2Stream));
    }

    if((getSetupStatus() == CH_SETUP_DONE_NO_ERROR))
    {
        //std::this_thread::sleep_for(std::chrono::microseconds(150));
        cuphyStatus_t runStatus = cuphyRunPuschRx(puschRxHndl, runPhase);
        if(runStatus != CUPHY_STATUS_SUCCESS)
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHY_API_EVENT, "cuphyRunPuschRx returned error {}", runStatus);
            ret=-1;
        }
    }

    if(runPhase == cuphyPuschRunPhase_t::PUSCH_RUN_ALL_PHASES) {
        MemtraceDisableScope md;
        CUDA_CHECK_PHYDRIVER(cudaEventRecord(end_run, dyn_params.phase2Stream));
    }

    if(runPhase == cuphyPuschRunPhase_t::PUSCH_RUN_FULL_SLOT_PROC) {
        MemtraceDisableScope md;
        CUDA_CHECK_PHYDRIVER(cudaEventRecord(end_run_ph1, dyn_params.phase1Stream));
    }

    if(runPhase == cuphyPuschRunPhase_t::PUSCH_RUN_FULL_SLOT_COPY){
        MemtraceDisableScope md;
        CUDA_CHECK_PHYDRIVER(cudaEventRecord(end_run_ph2, dyn_params.phase2Stream));
        CUDA_CHECK_PHYDRIVER(cudaEventRecord(end_run, dyn_params.phase2Stream));
    }

    t_ns t3 = Time::nowNs();

    return ret;
}

float PhyPuschAggr::getGPURunSubSlotTime() {
    return 1000.0f * GetCudaEventElapsedTime(waitCompletedSubSlotEvent, subSlotCompletedEvent, __func__, getId());
}

float PhyPuschAggr::getGPURunPostSubSlotTime() {
    return 1000.0f * GetCudaEventElapsedTime(waitCompletedFullSlotEvent, end_run_ph1, __func__, getId());
}

float PhyPuschAggr::getGPURunGapTime() {
    return 1000.0f * GetCudaEventElapsedTime(end_run_ph1, start_run_ph2, __func__, getId());
}

float PhyPuschAggr::getGPUPhaseRunTime(cuphyPuschRunPhase_t runPhase) {
    float ms = 0;

    if(runPhase == cuphyPuschRunPhase_t::PUSCH_RUN_SUB_SLOT_PROC) {
        return 1000.0f * GetCudaEventElapsedTime(start_run_ph1, end_run_ph1, __func__, getId());
    } else if (runPhase == cuphyPuschRunPhase_t::PUSCH_RUN_FULL_SLOT_COPY) {
        return 1000.0f * GetCudaEventElapsedTime(start_run_ph2, end_run_ph2, __func__, getId());
    } else if (runPhase == cuphyPuschRunPhase_t::PUSCH_RUN_ALL_PHASES) {
        return 1000.0f * GetCudaEventElapsedTime(start_run, end_run, __func__, getId());
    }

    return ms*1000;
}

#if 0
int PhyPuschAggr::wait(int wait_ns)
{
    t_ns          threshold_t(wait_ns), start_t = Time::nowNs();
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();

    if(!isActive())
        return -1;

    while(ACCESS_ONCE(*((uint32_t*)pusch_completed_h->addr())) == 0)
    // while(cudaEventQuery(end_crc) != cudaSuccess)
    {
        if(Time::nowNs() - start_t > threshold_t)
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "ERROR: PUSCH Object {} Cell {} waiting for GPU task more than {} ns",
                        getId(), cell_id, wait_ns);
            return -1;
        }
    }
#ifdef PUSCH_INPUT_BUFFER_DEBUG
    bool non_zero = false;
    for(int b=0; b < static_cast<int>(buf_sz); b++) {
        if(buf_h[b] != 0)
        {
            non_zero = true;
        }
    }
    if(non_zero == true)
    {
        NVLOGC_FMT(TAG, "SFN {} SLOT {} NON-ZERO PUSCH Payload", aggr_slot_params->si->sfn_, aggr_slot_params->si->slot_);
    }
    else
    {
        NVLOGC_FMT(TAG, "SFN {} SLOT {} ALL-ZERO PUSCH Payload", aggr_slot_params->si->sfn_, aggr_slot_params->si->slot_);
    }
#endif

    return 0;
}
#endif

int PhyPuschAggr::validate(std::array<uint8_t,UL_MAX_CELLS_PER_SLOT>& cell_timeout_list,bool gpu_early_harq_timeout, const std::vector<ULInputBuffer *>& aggr_ulbuf_pcap_capture, const std::vector<ULInputBuffer *>& aggr_ulbuf_pcap_capture_ts)
{
    PhyDriverCtx* pdctx    = StaticConversion<PhyDriverCtx>(this->getPhyDriverHandler()).get();

    uint32_t* pCbCrcs    = static_cast<uint32_t*>(bCbCrcs.addr());
    uint32_t* pTbCrcs    = static_cast<uint32_t*>(bTbCrcs.addr());
    uint8_t*  pEstBytes  = static_cast<uint8_t*>(bTbPayloads.addr());
    uint8_t*  pTrueBytes = static_cast<uint8_t*>(tTbBytes.addr());

    uint32_t* pStartOffsetsTbCrc  = static_cast<uint32_t*>(bStartOffsetsTbCrc.addr());
    uint64_t crc_error_cell_bitmask = 0;
    for(uint32_t i = 0; i < totNumCbCrc; ++i)
    {
        if(pCbCrcs[i] != 0)
        {
            nCbCrcErrors += 1;
            NVLOGD_FMT(TAG, "SFN {}.{} PUSCH obj pCbCrcs[{}] = {}", aggr_slot_params->si->sfn_, aggr_slot_params->si->slot_, i, pCbCrcs[i]);
        }
    }

    uint32_t hb_slot_index=0;
    totNumTbCrc = dyn_params.pCellGrpDynPrm->nUes;
    for(uint32_t i = 0; i < totNumTbCrc; ++i)
    {
        if(dyn_params.pCellGrpDynPrm->pUePrms[i].pduBitmap & 0x01)
        {
            uint32_t offsetsTbCrc = pStartOffsetsTbCrc[i];
            if(hb_slot_index < hq_buffer_counter && hb_slot[hb_slot_index]->getCellDynIdx()==-1){
                NVLOGE_FMT(TAG,AERIAL_INPUT_OUTPUT_EVENT,"{} : SFN {}.{} Cell dynamic index not being set for harq buffer slot index {}",__func__,aggr_slot_params->si->sfn_, aggr_slot_params->si->slot_,i);
                hb_slot_index++;
                continue;
            }
            if((hb_slot_index < hq_buffer_counter) && ((cell_timeout_list[hb_slot[hb_slot_index]->getCellDynIdx()]!=ORDER_KERNEL_EXIT_PRB) || gpu_early_harq_timeout))
            {
                pTbCrcs[offsetsTbCrc]=1;
                nTbCrcErrors += 1;
                NVLOGI_FMT(TAG, "SFN {}.{} PUSCH obj Forcing CRC error pTbCrcs[{}] = {} for Tb[{}] hb_slot[i]->getCellDynIdx() {} cell_timeout {}", 
                aggr_slot_params->si->sfn_, aggr_slot_params->si->slot_, offsetsTbCrc, pTbCrcs[offsetsTbCrc], i,hb_slot[hb_slot_index]->getCellDynIdx(),cell_timeout_list[hb_slot[hb_slot_index]->getCellDynIdx()]);
                //Flag okTimeout sets to true if any cell(s) has OK timeout. In callback, if okTimeout is true and a UE (of any cell) has NDI=1, free the HARQ buffer.
                //Reason: If OK timesout even for 1 cell, PUSCH pipeline does not run. Which means that HARQ buffer contents are not reset and contain residual data.
                //For NDI=1, free such a HARQ buffer to prevent recombining with residual data of another UE.
                if(cell_timeout_list[hb_slot[hb_slot_index]->getCellDynIdx()]!=ORDER_KERNEL_EXIT_PRB)
                {
                    okTimeout = true;
                }
            }
            else
            {
                if(pTbCrcs[offsetsTbCrc] != 0)
                {
                    nTbCrcErrors += 1;
                    NVLOGD_FMT(TAG, "SFN {}.{} PUSCH obj pTbCrcs[{}] = {} for Tb[{}]", aggr_slot_params->si->sfn_, aggr_slot_params->si->slot_, offsetsTbCrc, pTbCrcs[offsetsTbCrc], i);
                }
            }
            hb_slot_index++;

#ifdef AERIAL_METRICS
            ////////////////////////////////////////////
            //// Cell Metric
            ////////////////////////////////////////////
            auto uePrms = dyn_params.pCellGrpDynPrm->pUePrms[i];
            uint16_t cellIdx = uePrms.pUeGrpPrm->pCellPrm->cellPrmStatIdx;
            cell_metrics_info[cellIdx].nTBs++;
            cell_metrics_info[cellIdx].tbSize += uePrms.TBSize; // (in bytes)
            if(pTbCrcs[offsetsTbCrc] != 0)
            {
                cell_metrics_info[cellIdx].nTbCrc++;
                if(pdctx->get_ul_pcap_capture_enable())
                {
                    crc_error_cell_bitmask |= (1ULL << cellIdx);
                }
            }
#else
            if(pdctx->get_ul_pcap_capture_enable())
            {
                auto uePrms = dyn_params.pCellGrpDynPrm->pUePrms[i];
                uint16_t cellIdx = uePrms.pUeGrpPrm->pCellPrm->cellPrmStatIdx;
                if(pTbCrcs[offsetsTbCrc] != 0)
                {
                    crc_error_cell_bitmask |= (1ULL << cellIdx);
                }
            }
#endif
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

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////// PUSCH UL capture debug to capture on CRC error
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    if(pdctx->get_ul_pcap_capture_enable())
    {
        CuphyOAM *oam = CuphyOAM::getInstance();
        auto pcap_capture_cell_bitmask = oam->ul_pcap_arm_cell_bitmask.load();
        uint64_t capture_cells = crc_error_cell_bitmask & pcap_capture_cell_bitmask;
        for (int i = 0; i < UL_MAX_CELLS_PER_SLOT; ++i)
        {
            uint64_t cell_mask = 1ULL << i;
            if (capture_cells & cell_mask)
            {
                auto& capture_info = pdctx->ul_pcap_capture_context_info.ul_pcap_capture_info[pdctx->ul_pcap_capture_context_info.ul_pcap_capture_write_idx];
                capture_info.mtu = pdctx->get_ul_pcap_capture_mtu();
                capture_info.buffer_pointer = static_cast<uint8_t*>(aggr_ulbuf_pcap_capture[i]->getBufH());
                capture_info.buffer_pointer_ts = static_cast<uint8_t*>(aggr_ulbuf_pcap_capture_ts[i]->getBufH());
                capture_info.cell_id = i;
                capture_info.sfn = aggr_slot_params->si->sfn_;
                capture_info.slot = aggr_slot_params->si->slot_;
                pdctx->ul_pcap_capture_context_info.ul_pcap_capture_write_idx = (pdctx->ul_pcap_capture_context_info.ul_pcap_capture_write_idx + 1) % UL_MAX_CELLS_PER_SLOT;
                oam->ul_pcap_arm_cell_bitmask.fetch_and(~cell_mask, std::memory_order_release);
                NVLOGC_FMT(TAG, "Cell {} Trigger Pcap capture for SFN {} Slot {}", i, aggr_slot_params->si->sfn_, aggr_slot_params->si->slot_);
            }
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////// PUSCH Debug output 1 slot only
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifdef PUSCH_H5DUMP

    uint8_t* pHarqDetectionStatus                      = bHarqDetectionStatus.addr();
    uint8_t* pCsiP1DetectionStatus                     = bCsiP1DetectionStatus.addr();
    uint8_t* pCsiP2DetectionStatus                     = bCsiP2DetectionStatus.addr();

    uint16_t* pNumCsi2Bits                             = bNumCsi2Bits.addr();

    cuphyUciOnPuschOutOffsets_t* pUciOnPuschOutOffsets = bUciOnPuschOutOffsets.addr();
    uint16_t HarqDetectionStatusOffset, CsiP1DetectionStatusOffset, CsiP2DetectionStatusOffset;
    bool nUciErrorFlag = false;
    uint32_t firstUciErrorUe = 0;

    for(uint32_t i = 0; i < totNumTbCrc; ++i)
    {
        if(dyn_params.pCellGrpDynPrm->pUePrms[i].pduBitmap & 2)
        {
            HarqDetectionStatusOffset = pUciOnPuschOutOffsets[i].HarqDetectionStatusOffset;
            if(((dyn_params.pCellGrpDynPrm->pUePrms[i].pUciPrms)->nBitsHarq > 0) && ((pHarqDetectionStatus[HarqDetectionStatusOffset] != 1) && (pHarqDetectionStatus[HarqDetectionStatusOffset] != 4)))
            {
                nUciErrorFlag = true;
                firstUciErrorUe = i;
                break;
            }

            CsiP1DetectionStatusOffset = pUciOnPuschOutOffsets[i].CsiP1DetectionStatusOffset;
            if(((dyn_params.pCellGrpDynPrm->pUePrms[i].pUciPrms)->nBitsCsi1 > 0) && ((pCsiP1DetectionStatus[CsiP1DetectionStatusOffset] != 1) && (pCsiP1DetectionStatus[CsiP1DetectionStatusOffset] != 4)))
            {
                nUciErrorFlag = true;
                firstUciErrorUe = i;
                break;
            }
        }

        if(dyn_params.pCellGrpDynPrm->pUePrms[i].pduBitmap & 32)
        {
            CsiP2DetectionStatusOffset = pUciOnPuschOutOffsets[i].CsiP2DetectionStatusOffset;
            if((pNumCsi2Bits[CsiP2DetectionStatusOffset] > 0) && ((pCsiP2DetectionStatus[CsiP2DetectionStatusOffset] != 1) && (pCsiP2DetectionStatus[CsiP2DetectionStatusOffset] != 4)))
            {
                nUciErrorFlag = true;
                firstUciErrorUe = i;
                break;
            }

        }
    }
    if(nUciErrorFlag)
    {
        if((dyn_params.pCellGrpDynPrm->pUePrms[firstUciErrorUe].pUciPrms)->nBitsHarq > 0)
        {
            NVLOGE_FMT(TAG, AERIAL_INPUT_OUTPUT_EVENT, "UCI-on-PUSCH Error UE {}: nBitsHarq {} HarqDetStatus {}", firstUciErrorUe, (dyn_params.pCellGrpDynPrm->pUePrms[firstUciErrorUe].pUciPrms)->nBitsHarq, pHarqDetectionStatus[pUciOnPuschOutOffsets[firstUciErrorUe].HarqDetectionStatusOffset]);
        }
        if((dyn_params.pCellGrpDynPrm->pUePrms[firstUciErrorUe].pUciPrms)->nBitsCsi1 > 0)
        {
            NVLOGE_FMT(TAG, AERIAL_INPUT_OUTPUT_EVENT, "UCI-on-PUSCH Error UE {}: nBitsCsi1 {}, CsiP1DetStatus {}", firstUciErrorUe, (dyn_params.pCellGrpDynPrm->pUePrms[firstUciErrorUe].pUciPrms)->nBitsCsi1, pCsiP1DetectionStatus[pUciOnPuschOutOffsets[firstUciErrorUe].CsiP1DetectionStatusOffset]);
        }
        if(pNumCsi2Bits[pUciOnPuschOutOffsets[firstUciErrorUe].CsiP2DetectionStatusOffset] > 0)
        {
             NVLOGE_FMT(TAG, AERIAL_INPUT_OUTPUT_EVENT, "UCI-on-PUSCH Error UE {}: nBitsCsi2 {}, CsiP2DetStatus {}", firstUciErrorUe, pNumCsi2Bits[pUciOnPuschOutOffsets[firstUciErrorUe].CsiP2DetectionStatusOffset], pCsiP2DetectionStatus[pUciOnPuschOutOffsets[firstUciErrorUe].CsiP2DetectionStatusOffset]);
        }
    }

    CuphyOAM *oam = CuphyOAM::getInstance();
    bool triggerH5Dump = false;
    if((nTbCrcErrors != 0 || nCbCrcErrors != 0 || nUciErrorFlag) && h5dumped == false)
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

    if(triggerH5Dump)
    {
        NVLOGC_FMT(TAG, "CRC error encountered SFN {}.{} Generating H5 Debug PUSCH file {}", aggr_slot_params->si->sfn_, aggr_slot_params->si->slot_, std::to_string(id).c_str());
        auto& stream = s_channel;
        cudaStreamSynchronize(stream);
        cuphyStatus_t debugStatus = cuphyWriteDbgBufSynch(puschRxHndl, stream);
        if(debugStatus != CUPHY_STATUS_SUCCESS)
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHY_API_EVENT, "cuphyWriteDbgBufSynch returned error {}", debugStatus);
            return -1;
        }
        cudaStreamSynchronize(stream);
        debugFileH.get()->close();
        debugFileH.reset();
        NVLOGC_FMT(TAG, "SFN {}.{} Done Generating H5 Debug PUSCH {} file, please refer to the largest h5dump file created.", aggr_slot_params->si->sfn_, aggr_slot_params->si->slot_, std::to_string(id).c_str());
        oam->puschH5DumpMutex.lock();
        // oam->puschH5dumpNextCrc.store(false);
        // oam->puschH5dumpInProgress.store(false);
        oam->puschH5DumpMutex.unlock();
        NVLOGC_FMT(TAG, "SFN {}.{} Release H5Dump Lock, exitting forcefully, please expect a lot error messages!", aggr_slot_params->si->sfn_, aggr_slot_params->si->slot_);
        EXIT_L1(EXIT_FAILURE);
    }
#endif
    return 0;
}

void PhyPuschAggr::setWorkCancelFlag(bool flag_value)
{
    memset(bWorkCancelInfo.addr(), (flag_value) ? 1 : 0, sizeof(uint8_t));
}

int PhyPuschAggr::cleanup()
{
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    PhyChannel::cleanup();

    nCbCrcErrors  = 0;
    nTbCrcErrors  = 0;
    nTbByteErrors = 0;
    hq_buffer_counter = 0;
    okTimeout = false;
    memset(bCbCrcs.addr(), 0, sizeof(uint32_t) * totNumCbCrc);
    memset(bTbCrcs.addr(), 1, sizeof(uint32_t) * totNumTbCrc);
    for(int sym_idx=0;sym_idx<ORAN_PUSCH_SYMBOLS_X_SLOT;sym_idx++){
        ACCESS_ONCE(*((uint32_t*)sym_ord_done_sig_arr->addrh()+sym_idx))=(uint32_t)SYM_RX_NOT_DONE;
    }

#if 1
    memset(bWorkCancelInfo.addr(), 0, sizeof(uint8_t));
#else
    memset(bWorkCancelInfo.addr(), 1, sizeof(uint8_t));  // Enable for some basic test if PUSCH DGL or cond. IF nodes conditions are always false.
#endif

    return 0;
}

int PhyPuschAggr::clearUciFlags(bool harq_status_only)
{
    /*
    if(aggr_slot_params){
        struct slot_command_api::slot_indication* si = aggr_slot_params->si;
        NVLOGD_FMT(TAG,"PhyPuschAggr::{} SFN {}.{} harq_status_only={}",__func__,si->sfn_,si->slot_,harq_status_only);        
    }
    */
    memset(bHarqDetectionStatus.addr(), 2, sizeof(uint8_t) * MAX_N_TBS_SUPPORTED);
    if(!harq_status_only)
    {
        memset(bCsiP1DetectionStatus.addr(), 2, sizeof(uint8_t) * MAX_N_TBS_SUPPORTED);
        memset(bCsiP2DetectionStatus.addr(), 2, sizeof(uint8_t) * MAX_N_TBS_SUPPORTED);
        memset(bUciCrcFlags.addr(), 1, sizeof(uint8_t) * nUciSegs);    
    }
    return 1;
}

uint8_t PhyPuschAggr::getPreEarlyHarqWaitKernelStatus()
{
    return *pPreEarlyHarqWaitKernelStatus;
}

uint8_t PhyPuschAggr::getPostEarlyHarqWaitKernelStatus()
{
    return *pPostEarlyHarqWaitKernelStatus;
}


int PhyPuschAggr::callback(std::array<uint8_t,UL_MAX_CELLS_PER_SLOT>& cell_timeout_list,bool gpu_early_harq_timeout)
{
    PhyDriverCtx*                       pdctx    = StaticConversion<PhyDriverCtx>(this->getPhyDriverHandler()).get();

    slot_command_api::ul_slot_callbacks ul_cb;
    auto pusch = getDynParams();

    struct slot_command_api::slot_indication* si = aggr_slot_params->si;
    NVLOGD_FMT(TAG, "PhyPuschAggr{} SFN {}.{}: callback cnt_used={} hb_slot.size={}",
            this_id, si->sfn_, si->slot_, cnt_used, hq_buffer_counter);

    if(pdctx->getUlCb(ul_cb))
    {
        NVLOGD_FMT(TAG, "Calling UL Aggr callback");

        struct slot_command_api::ul_output_msg_buffer msg;
        msg.data_buf    = nullptr;
        msg.total_bytes = 0;
        msg.numTB       = 0;


        // Tell datalakes there is work to do. This only copies addresses and notifies the worker thread
        if(pdctx->getDataLake() != nullptr) {
           pdctx->getDataLake()->notify(nTbCrcErrors, aggr_slot_params->si, pusch, &DataOut, &static_params);
        }

        // Only for compilation to run pass actual cuphyPuschDataOut_t* struct
        // nCRC calculated on the GPU with cuPHYTools kernel
        ul_cb.callback_fn(ul_cb.callback_fn_context, nTbCrcErrors, msg, *(aggr_slot_params->si), *pusch, &DataOut, &static_params);
    }

    // Free HARQ buffer for CRC == 0
    if(nTbCrcErrors == 0)
    {
        for (int k=0; k < hq_buffer_counter; k++)
        {
            hb_slot[k]->refSub();
            NVLOGD_FMT(TAG, "SFN {}.{} Freeing HARQ buffer {} {} (dev_buf {}) for CRC errors == 0 rnti {} hpid {} cellid {} ref_count {}",
                    si->sfn_, si->slot_, k, reinterpret_cast<void*>(hb_slot[k]), reinterpret_cast<void*>(hb_slot[k]->getAddr()),
                    hb_slot[k]->getRnti(), hb_slot[k]->getHarqPid(), hb_slot[k]->getCellId(), hb_slot[k]->getRefCount()
            );

            if(hb_slot[k]->getRefCount() <= 0)
            {
                hb_pool_m->bucketReleaseBuffer(hb_slot[k]);
            }
        }
    }
    else
    {
        uint32_t count = 0;
        uint32_t hb_slot_index = 0;
        t_ns now = Time::nowNs();
        if(hq_buffer_counter>0) //Check for release only if non-zero size
        {
            //Sometime due to Harq pool depletion some UE might not have a HARQ buffer allocated.
            for (int k=0; k < pusch->cell_grp_info.nUes && hb_slot_index < hq_buffer_counter; k++)
            {
                // HARQ buffer wasn't allocated if !(pduBitmap&0x01)
                if(pusch->cell_grp_info.pUePrms[k].pduBitmap & 0x01)
                {
                    uint32_t offsetsTbCrc = DataOut.pStartOffsetsTbCrc[k];
                    if(hb_slot[hb_slot_index]->getCellDynIdx()==-1){
                        NVLOGE_FMT(TAG,AERIAL_INPUT_OUTPUT_EVENT,"{} : SFN {}.{} harq buffer for ue {} already released. hb_slot_index {}",__func__,aggr_slot_params->si->sfn_, aggr_slot_params->si->slot_,k,hb_slot_index);
                        hb_slot_index++;
                        continue;
                    }
                    hb_slot[hb_slot_index]->refSub(); //Got a callback for this HARQ buffer so we can reduce the number of activer references to this HARQ buffer
                    if(hb_slot[hb_slot_index]->getRefCount() <= 0)
                    {
                        hb_pool_m->unsetInUse(hb_slot[hb_slot_index]);
                        //Reset the number of TTIs that this HARQ buffer is bundled with. This to start fresh count in the next bundling window.
                        hb_slot[hb_slot_index]->resetNumTtiBundled();
                    }
                    //If OK timesout even for 1 cell, PUSCH pipeline does not run. Which means that HARQ buffer contents are not reset and contain residual data.
                    //if okTimeout == true and NDI==1, free such a HARQ buffer to prevent recombining with residual data of another UE.
                    if((DataOut.pTbCrcs[offsetsTbCrc] == 0) || (gpu_early_harq_timeout==true) || (cell_timeout_list[hb_slot[hb_slot_index]->getCellDynIdx()]!=ORDER_KERNEL_EXIT_PRB) || (okTimeout && pusch->cell_grp_info.pUePrms[k].ndi==1))
                    {
                        NVLOGI_FMT(TAG, "SFN {}.{} Freeing HARQ buffer ue={} ndi={} offsetsTbCrc={} DataOut.pTbCrcs[offsetsTbCrc]={} gpu_early_harq_timeout={} hb_slot_index={} hb_slot[hb_slot_index]->getCellDynIdx()={} cell_timeout={} okTimeout={} ref_count={}",
                            si->sfn_, si->slot_,k,pusch->cell_grp_info.pUePrms[k].ndi,offsetsTbCrc,DataOut.pTbCrcs[offsetsTbCrc],gpu_early_harq_timeout,hb_slot_index,hb_slot[hb_slot_index]->getCellDynIdx(),cell_timeout_list[hb_slot[hb_slot_index]->getCellDynIdx()],okTimeout, hb_slot[hb_slot_index]->getRefCount());
                        if(hb_slot[hb_slot_index]->getRefCount() <= 0)
                        {
                            if(!hb_pool_m->bucketReleaseBuffer(hb_slot[hb_slot_index])) {
                                count++;
                            }
                            else {
                                NVLOGI_FMT(TAG, "SFN {}.{} HARQ buffer already released (possibly due to HARQ pool depletion)",si->sfn_, si->slot_);
                            }
                        }
                    }
                    hb_slot_index++;
                }
            }        
        }
        NVLOGD_FMT(TAG, "SFN {}.{} Due to CRC errors {} can't free HARQ {} buffers", si->sfn_, si->slot_, nTbCrcErrors, hq_buffer_counter-count);
    }
    
    hb_pool_m->checkPoolDepletion(released_harq_buffer_info);
    if((released_harq_buffer_info.num_released_harq_buffers > 0) && (pdctx->getNotifyUlHarqBufferRelease()))
    {
        pdctx->getUlCb(ul_cb);
        ul_cb.ul_free_harq_buffer_fn(ul_cb.ul_free_harq_buffer_fn_context, released_harq_buffer_info, pusch, si->sfn_, si->slot_);
        released_harq_buffer_info.reset();
    }

    // Check WAvgCfo pool depletion if enabled
    if(wavgcfo_pool_m != nullptr)
    {
        wavgcfo_pool_m->checkPoolDepletion();
    }

    hq_buffer_counter = 0;

#ifdef AERIAL_METRICS
    Cell*    cell_list[MAX_CELLS_PER_SLOT];
    uint32_t cellCount = 0;
    pdctx->getCellList(cell_list, &cellCount);
    if(cellCount == 0)
        return EINVAL;

    for(uint32_t cellIdx = 0; cellIdx < cellCount; cellIdx++)
    {
        auto& cell_ptr = cell_list[cellIdx];
        cell_ptr->updateMetric(CellMetric::kPuschRxTbBytesTotal, cell_metrics_info[cellIdx].tbSize);
        cell_ptr->updateMetric(CellMetric::kPuschRxTbTotal, cell_metrics_info[cellIdx].nTBs);
        cell_ptr->updateMetric(CellMetric::kPuschRxTbCrcErrorTotal, cell_metrics_info[cellIdx].nTbCrc);
        cell_metrics_info[cellIdx].tbSize = 0;
        cell_metrics_info[cellIdx].nTBs = 0;
        cell_metrics_info[cellIdx].nTbCrc = 0;
    }
#endif

#if 0
    // cell_ptr->updateMetric(CellMetric::kPuschRxTbBytesTotal, DataOut.totNumPayloadBytes);
    // cell_ptr->updateMetric(CellMetric::kPuschRxTbTotal, DataOut.totNumTbs);

    uint32_t* pTbCrcs = static_cast<uint32_t*>(bTbCrcs.addr());
    uint32_t  nTbCrcErrors  = 0;
    for(uint32_t i = 0; i < totNumTbCrc; ++i)
    {
        if(pTbCrcs[i] != 0)
            nTbCrcErrors += 1;
    }

    // cell_ptr->updateMetric(CellMetric::kPuschRxTbCrcErrorTotal, nTbCrcErrors);
    // cell_ptr->updateMetric(CellMetric::kPuschNrOfUesPerSlot, cellGrpDynPrm.nUes);
    // cell_ptr->updateMetric(CellMetric::kPuschProcessingTime, this->getGPURunTime());
#endif
    //Avoid overflow
    cnt_used = (cnt_used + 1) % 65536;

    return 0;
}

float PhyPuschAggr::getGPUSetupPh1Time() {
    return 1000.0f * GetCudaEventElapsedTime(start_setup_ph1, end_setup_ph1, __func__, getId());
}

float PhyPuschAggr::getGPUSetupPh2Time() {
    return 1000.0f * GetCudaEventElapsedTime(start_setup_ph2, end_setup_ph2, __func__, getId());
}

float PhyPuschAggr::getGPUCrcTime() {
    return 1000.0f * GetCudaEventElapsedTime(start_crc, end_crc, __func__, getId());
}

HarqPoolManager* PhyPuschAggr::getHarqPoolManager()
{
    return hb_pool_m;
}

cuphyPuschDynPrms_t *PhyPuschAggr::getPuschDynParams()
{
    return &dyn_params;
}

cuphyPuschStatPrms_t *PhyPuschAggr::getPuschStatParams()
{
    return &static_params;
}

void PhyPuschAggr::updatePhyCellId(uint16_t phyCellId_old,uint16_t phyCellId_new)
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

uint32_t* PhyPuschAggr::getSymOrderSigDoneGpuFlag(int sym_idx)
{
    return ((uint32_t*)((uint32_t*)sym_ord_done_sig_arr->addrd()+sym_idx));
}

uint32_t* PhyPuschAggr::getSymOrderSigDoneCpuFlag(int sym_idx)
{
    return ((uint32_t*)((uint32_t*)sym_ord_done_sig_arr->addrh()+sym_idx));
}

void printParametersAggr(PhyDriverCtx* pdctx, cuphyPuschCellGrpDynPrm_t* l2, cuphyPuschCellGrpDynPrm_t* tv)
{
#if 0
    if (l2 == nullptr || tv == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Either L2 params and or TV params is null");
        return;
    }

    // NVSLOGE(TAG, AERIAL_CUPHYDRV_API_EVENT) <<__FUNCTION__ << " L2: nCells: " << l2->nCells << " TV: nCells: " <<tv->nCells;
    NVLOGI_FMT(TAG, "\n===============================================");

    NVLOGI_FMT(TAG, "{}  L2: nCells: {} TV: nCells: {}", __FUNCTION__, l2->nCells, tv->nCells);
    for (uint16_t i = 0 ; i < l2->nCells; i++)
    {
        cuphyPuschCellDynPrm_t* dyn = &l2->pCellPrms[i];
        cuphyPuschCellDynPrm_t* other = &tv->pCellPrms[i];

        NVLOGI_FMT(TAG, "{} L2: cellPrmStatIdx:{} TV: cellPrmStatIdx:{}", __FUNCTION__, dyn->cellPrmStatIdx, other->cellPrmStatIdx);
        NVLOGI_FMT(TAG, "{} L2: cellPrmDynIdx:{} TV: cellPrmDynIdx:{}", __FUNCTION__, dyn->cellPrmDynIdx, other->cellPrmDynIdx);
        NVLOGI_FMT(TAG, "{} L2: slotNum:{} TV: slotNum:{}", __FUNCTION__, dyn->slotNum, other->slotNum);

        // NVSLOGE(TAG, AERIAL_CUPHYDRV_API_EVENT) <<__FUNCTION__ << " L2: cellPrmStatIdx: " << dyn->cellPrmStatIdx << " TV: cellPrmStatIdx: " <<other->cellPrmStatIdx;
        // NVSLOGE(TAG, AERIAL_CUPHYDRV_API_EVENT) <<__FUNCTION__ << " L2: cellPrmDynIdx: " << dyn->cellPrmDynIdx << " TV: cellPrmDynIdx: " <<other->cellPrmDynIdx;
        // NVSLOGE(TAG, AERIAL_CUPHYDRV_API_EVENT) <<__FUNCTION__ << " L2: slotNum: " << dyn->slotNum << " TV: slotNum: " <<other->slotNum;
        // NVSLOGE(TAG, AERIAL_CUPHYDRV_API_EVENT) <<__FUNCTION__ << " L2: puschStartSym: " << dyn->puschStartSym << " TV: puschStartSym: " <<other->puschStartSym;
        // NVSLOGE(TAG, AERIAL_CUPHYDRV_API_EVENT) <<__FUNCTION__ << " L2: nPuschSym: " << dyn->nPuschSym << " TV: nPuschSym: " <<other->nPuschSym;
        // NVSLOGE(TAG, AERIAL_CUPHYDRV_API_EVENT) <<__FUNCTION__ << " L2: dmrsSymLocBmsk: " << dyn->dmrsSymLocBmsk << " TV: dmrsSymLocBmsk: " <<other->dmrsSymLocBmsk;
    }

    NVLOGI_FMT(TAG, "{}  L2: nUeGrps: {} TV: nUeGrps: {}", __FUNCTION__, l2->nUeGrps, tv->nUeGrps);

    // NVSLOGE(TAG, AERIAL_CUPHYDRV_API_EVENT) <<__FUNCTION__ << " L2: nUeGrps: " << l2->nUeGrps << " TV: nCells: " <<tv->nUeGrps;

    for (uint16_t i=0; i< l2->nUeGrps; i++)
    {
        cuphyPuschUeGrpPrm_t* dyn = &l2->pUeGrpPrms[i];
        cuphyPuschUeGrpPrm_t* other = &tv->pUeGrpPrms[i];
        NVLOGI_FMT(TAG, "{} L2: startPrb:{} TV: startPrb:{}", __FUNCTION__, dyn->startPrb, other->startPrb);
        NVLOGI_FMT(TAG, "{} L2: nPrb:{} TV: nPrb:{}", __FUNCTION__, dyn->nPrb, other->nPrb);
        NVLOGI_FMT(TAG, "{} L2: nUes:{} TV: nUes:{}", __FUNCTION__, dyn->nUes, other->nUes);

        // NVSLOGE(TAG, AERIAL_CUPHYDRV_API_EVENT) <<__FUNCTION__ << " L2: startPrb: " << dyn->startPrb << " TV: startPrb: " <<other->startPrb;
        // NVSLOGE(TAG, AERIAL_CUPHYDRV_API_EVENT) <<__FUNCTION__ << " L2: nPrb: " << dyn->nPrb << " TV: nPrb: " <<other->nPrb;
        // NVSLOGE(TAG, AERIAL_CUPHYDRV_API_EVENT) <<__FUNCTION__ << " L2: nUes: " << dyn->nUes << " TV: nUes: " <<other->nUes;
        for (uint16_t j = 0; j < dyn->nUes; j++)
        {
            //uint16_t* dynIdx = &dyn->pUePrmIdxs[i];
            //uint16_t* otherIdx = &other->pUePrmIdxs[i];
            NVLOGI_FMT(TAG, "{} L2: pUePrmIdxs:{} TV: pUePrmIdxs:{}", __FUNCTION__, dyn->pUePrmIdxs[j], other->pUePrmIdxs[j]);
            // NVSLOGD(TAG) <<__FUNCTION__ << " L2: pUePrmIdxs: " << dyn->pUePrmIdxs[j] << " TV: pUePrmIdxs: " <<other->pUePrmIdxs[j];

        }
        cuphyPuschDmrsPrm_t* dmrs = dyn->pDmrsDynPrm;
        cuphyPuschDmrsPrm_t* dmrsOther = other->pDmrsDynPrm;

        NVLOGI_FMT(TAG, "{} L2: dmrsAddlnPos:{} TV: dmrsAddlnPos:{}", __FUNCTION__, dmrs->dmrsAddlnPos, dmrsOther->dmrsAddlnPos);
        NVLOGI_FMT(TAG, "{} L2: dmrsMaxLen:{} TV: dmrsMaxLen:{}", __FUNCTION__, dmrs->dmrsMaxLen, dmrsOther->dmrsMaxLen);
        NVLOGI_FMT(TAG, "{} L2: nDmrsCdmGrpsNoData:{} TV: nDmrsCdmGrpsNoData:{}", __FUNCTION__, dmrs->nDmrsCdmGrpsNoData, dmrsOther->nDmrsCdmGrpsNoData);
        NVLOGI_FMT(TAG, "{} L2: dmrsScrmId:{} TV: dmrsScrmId:{}", __FUNCTION__, dmrs->dmrsScrmId, dmrsOther->dmrsScrmId);

        // NVSLOGE(TAG, AERIAL_CUPHYDRV_API_EVENT) <<__FUNCTION__ << " L2: dmrsAddlnPos: " << dmrs->dmrsAddlnPos << " TV: dmrsAddlnPos: " <<dmrsOther->dmrsAddlnPos;
        // NVSLOGE(TAG, AERIAL_CUPHYDRV_API_EVENT) <<__FUNCTION__ << " L2: dmrsMaxLen: " << dmrs->dmrsMaxLen << " TV: dmrsMaxLen: " <<dmrsOther->dmrsMaxLen;
        // NVSLOGE(TAG, AERIAL_CUPHYDRV_API_EVENT) <<__FUNCTION__ << " L2: nDmrsCdmGrpsNoData: " << dmrs->nDmrsCdmGrpsNoData << " TV: nDmrsCdmGrpsNoData: " <<dmrsOther->nDmrsCdmGrpsNoData;
        // NVSLOGE(TAG, AERIAL_CUPHYDRV_API_EVENT) <<__FUNCTION__ << " L2: dmrsScrmId: " << dmrs->dmrsScrmId << " TV: dmrsScrmId: " <<dmrsOther->dmrsScrmId;

    }

    NVLOGI_FMT(TAG, "{}  L2: nUes: {} TV: nUes: {}", __FUNCTION__, l2->nUes, tv->nUes);

    // NVSLOGE(TAG, AERIAL_CUPHYDRV_API_EVENT) <<__FUNCTION__ << " L2: nUes: " << l2->nUes << " TV: nUes: " <<tv->nUes;

    for (uint16_t i = 0; i < l2->nUes; i++)
    {
        cuphyPuschUePrm_t* dyn = &l2->pUePrms[i];
        cuphyPuschUePrm_t* other = &tv->pUePrms[i];
        NVLOGI_FMT(TAG, "{} L2: ueGrpIdx:{} TV: ueGrpIdx:{}", __FUNCTION__, dyn->ueGrpIdx, other->ueGrpIdx);
        NVLOGI_FMT(TAG, "{} L2: scid:{} TV: scid:{}", __FUNCTION__, dyn->scid, other->scid);
        NVLOGI_FMT(TAG, "{} L2: dmrsPortBmsk:{} TV: dmrsPortBmsk:{}", __FUNCTION__, dyn->dmrsPortBmsk, other->dmrsPortBmsk);
        NVLOGI_FMT(TAG, "{} L2: mcsTableIndex:{} TV: mcsTableIndex:{}", __FUNCTION__, dyn->mcsTableIndex, other->mcsTableIndex);
        NVLOGI_FMT(TAG, "{} L2: mcsIndex:{} TV: mcsIndex:{}", __FUNCTION__, dyn->mcsIndex, other->mcsIndex);
        NVLOGI_FMT(TAG, "{} L2: rv:{} TV: rv:{}", __FUNCTION__, dyn->rv, other->rv);
        NVLOGI_FMT(TAG, "{} L2: rnti:{} TV: rnti:{}", __FUNCTION__, dyn->rnti, other->rnti);
        NVLOGI_FMT(TAG, "{} L2: dataScramId:{} TV: dataScramId:{}", __FUNCTION__, dyn->dataScramId, other->dataScramId);
        NVLOGI_FMT(TAG, "{} L2: nUeLayers:{} TV: nUeLayers:{}", __FUNCTION__, dyn->nUeLayers, other->nUeLayers);
        NVLOGI_FMT(TAG, "{} L2: layerMap:%" PRIu64 " TV: layerMap:%"PRIu64"\n", __FUNCTION__, dyn->layerMap, other->layerMap);

        // NVSLOGE(TAG, AERIAL_CUPHYDRV_API_EVENT) <<__FUNCTION__ << " L2: ueGrpIdx: " << dyn->ueGrpIdx << " TV: ueGrpIdx: " <<other->ueGrpIdx;
        // NVSLOGE(TAG, AERIAL_CUPHYDRV_API_EVENT) <<__FUNCTION__ << " L2: scid: " << dyn->scid << " TV: scid: " <<other->scid;
        // NVSLOGE(TAG, AERIAL_CUPHYDRV_API_EVENT) <<__FUNCTION__ << " L2: dmrsPortBmsk: " << dyn->dmrsPortBmsk << " TV: dmrsPortBmsk: " <<other->dmrsPortBmsk;
        // NVSLOGE(TAG, AERIAL_CUPHYDRV_API_EVENT) <<__FUNCTION__ << " L2: mcsTableIndex: " << dyn->mcsTableIndex << " TV: mcsTableIndex: " <<other->mcsTableIndex;
        // NVSLOGE(TAG, AERIAL_CUPHYDRV_API_EVENT) <<__FUNCTION__ << " L2: mcsTableIndex: " << dyn->mcsIndex << " TV: mcsIndex: " <<other->mcsIndex;
        // NVSLOGE(TAG, AERIAL_CUPHYDRV_API_EVENT) <<__FUNCTION__ << " L2: rv: " << dyn->rv << " TV: rv: " <<other->rv;
        // NVSLOGE(TAG, AERIAL_CUPHYDRV_API_EVENT) <<__FUNCTION__ << " L2: rnti: " << dyn->rnti << " TV: rnti: " <<other->rnti;
        // NVSLOGE(TAG, AERIAL_CUPHYDRV_API_EVENT) <<__FUNCTION__ << " L2: dataScramId: " << dyn->dataScramId << " TV: dataScramId: " <<other->dataScramId;
        // NVSLOGE(TAG, AERIAL_CUPHYDRV_API_EVENT) <<__FUNCTION__ << " L2: nUeLayers: " << dyn->nUeLayers << " TV: nUeLayers: " <<other->nUeLayers;
        // NVSLOGE(TAG, AERIAL_CUPHYDRV_API_EVENT) <<__FUNCTION__ << " L2: cellPrmStatIdx: " << dyn->layerMap << " TV: cellPrmStatIdx: " <<other->layerMap;

    }
    NVLOGI_FMT(TAG, "===============================================");
#endif
}

void printParametersAggr(PhyDriverCtx* pdctx, const cuphyPuschCellGrpDynPrm_t* l2)
{
#if 0
    if (l2 == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "L2 params is null");
        return;
    }
    NVLOGI_FMT(TAG, "===============================================");
    NVLOGI_FMT(TAG, "{}  L2: nCells: {}", __FUNCTION__, l2->nCells);
    for (uint16_t i = 0 ; i < l2->nCells; i++)
    {
        cuphyPuschCellDynPrm_t* dyn = &l2->pCellPrms[i];

        NVLOGI_FMT(TAG, "{} L2: cellPrmStatIdx:{}", __FUNCTION__, dyn->cellPrmStatIdx);
        NVLOGI_FMT(TAG, "{} L2: cellPrmDynIdx:{}", __FUNCTION__, dyn->cellPrmDynIdx);
        NVLOGI_FMT(TAG, "{} L2: slotNum:{}", __FUNCTION__, dyn->slotNum);
    }
    NVLOGI_FMT(TAG, "{}  L2: nUeGrps: {}", __FUNCTION__, l2->nUeGrps);
    for (uint16_t i=0; i< l2->nUeGrps; i++)
    {
        cuphyPuschUeGrpPrm_t* dyn = &l2->pUeGrpPrms[i];
        NVLOGI_FMT(TAG, "{} L2: startPrb:{}", __FUNCTION__, dyn->startPrb);
        NVLOGI_FMT(TAG, "{} L2: nPrb:{}", __FUNCTION__, dyn->nPrb);
        NVLOGI_FMT(TAG, "{} L2: nUes:{}", __FUNCTION__, dyn->nUes);
        NVLOGI_FMT(TAG, "{} L2: puschStartSym:{}", __FUNCTION__, dyn->puschStartSym);
        NVLOGI_FMT(TAG, "{} L2: nPuschSym:{}", __FUNCTION__, dyn->nPuschSym);
        NVLOGI_FMT(TAG, "{} L2: dmrsSymLocBmsk:{}", __FUNCTION__, dyn->dmrsSymLocBmsk);
        NVLOGI_FMT(TAG, "{} L2: rssiSymLocBmsk:{}", __FUNCTION__, dyn->rssiSymLocBmsk);

        for (uint16_t j = 0; j < dyn->nUes; j++)
        {
            NVLOGI_FMT(TAG, "{} L2: pUePrmIdxs:{}", __FUNCTION__, dyn->pUePrmIdxs[j]);

        }
        cuphyPuschDmrsPrm_t* dmrs = dyn->pDmrsDynPrm;

        NVLOGI_FMT(TAG, "{} L2: dmrsAddlnPos:{}", __FUNCTION__, dmrs->dmrsAddlnPos);
        NVLOGI_FMT(TAG, "{} L2: dmrsMaxLen:{}", __FUNCTION__, dmrs->dmrsMaxLen);
        NVLOGI_FMT(TAG, "{} L2: nDmrsCdmGrpsNoData:{}", __FUNCTION__, dmrs->nDmrsCdmGrpsNoData);
        NVLOGI_FMT(TAG, "{} L2: dmrsScrmId:{}", __FUNCTION__, dmrs->dmrsScrmId);
    }
    NVLOGI_FMT(TAG, "{}  L2: nUes: {}", __FUNCTION__, l2->nUes);
    for (uint16_t i = 0; i < l2->nUes; i++)
    {
        cuphyPuschUePrm_t* dyn = &l2->pUePrms[i];
        NVLOGI_FMT(TAG, "{} L2: ueGrpIdx:{}", __FUNCTION__, dyn->ueGrpIdx);
        NVLOGI_FMT(TAG, "{} L2: scid:{}", __FUNCTION__, dyn->scid);
        NVLOGI_FMT(TAG, "{} L2: dmrsPortBmsk:{}", __FUNCTION__, dyn->dmrsPortBmsk);
        NVLOGI_FMT(TAG, "{} L2: mcsTableIndex:{}", __FUNCTION__, dyn->mcsTableIndex);
        NVLOGI_FMT(TAG, "{} L2: mcsIndex:{}", __FUNCTION__, dyn->mcsIndex);
        NVLOGI_FMT(TAG, "{} L2: targetCodeRate:{}", __FUNCTION__, dyn->targetCodeRate);
        NVLOGI_FMT(TAG, "{} L2: qamModOrder:{}", __FUNCTION__, dyn->qamModOrder);
        NVLOGI_FMT(TAG, "{} L2: TBSize:{}", __FUNCTION__, dyn->TBSize);
        NVLOGI_FMT(TAG, "{} L2: rv:{}", __FUNCTION__, dyn->rv);
        NVLOGI_FMT(TAG, "{} L2: rnti:{}", __FUNCTION__, dyn->rnti);
        NVLOGI_FMT(TAG, "{} L2: dataScramId:{}", __FUNCTION__, dyn->dataScramId);
        NVLOGI_FMT(TAG, "{} L2: nUeLayers:{}", __FUNCTION__, dyn->nUeLayers);
        NVLOGI_FMT(TAG, "{} L2: ndi:{}", __FUNCTION__, dyn->ndi);
        NVLOGI_FMT(TAG, "{} L2: harqProcessId:{}", __FUNCTION__, dyn->harqProcessId);
        NVLOGI_FMT(TAG, "{} L2: i_lbrm:{}", __FUNCTION__, dyn->i_lbrm);
        NVLOGI_FMT(TAG, "{} L2: maxLayers:{}", __FUNCTION__, dyn->maxLayers);
        NVLOGI_FMT(TAG, "{} L2: maxQm:{}", __FUNCTION__, dyn->maxQm);
        NVLOGI_FMT(TAG, "{} L2: n_PRB_LBRM:{}", __FUNCTION__, dyn->n_PRB_LBRM);
        NVLOGI_FMT(TAG, "{} L2: pduBitmap:{}", __FUNCTION__, dyn->pduBitmap);
    }
    NVLOGI_FMT(TAG, "===============================================");
#endif
}
