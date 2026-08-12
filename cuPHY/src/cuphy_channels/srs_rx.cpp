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

#include <cstddef>
#include <string>
#include "srs_rx.hpp"
#include "cuphy.hpp"
#include "cuphy_api.h"
#include "convert_tensor.cuh"

#include "cuphy_hdf5.hpp"
#include "hdf5hpp.hpp"

constexpr uint16_t SRS_BW_TABLE[64][8] =
{{4,1,4,1,4,1,4,1},
{8,1,4,2,4,1,4,1},
{12,1,4,3,4,1,4,1},
{16,1,4,4,4,1,4,1},
{16,1,8,2,4,2,4,1},
{20,1,4,5,4,1,4,1},
{24,1,4,6,4,1,4,1},
{24,1,12,2,4,3,4,1},
{28,1,4,7,4,1,4,1},
{32,1,16,2,8,2,4,2},
{36,1,12,3,4,3,4,1},
{40,1,20,2,4,5,4,1},
{48,1,16,3,8,2,4,2},
{48,1,24,2,12,2,4,3},
{52,1,4,13,4,1,4,1},
{56,1,28,2,4,7,4,1},
{60,1,20,3,4,5,4,1},
{64,1,32,2,16,2,4,4},
{72,1,24,3,12,2,4,3},
{72,1,36,2,12,3,4,3},
{76,1,4,19,4,1,4,1},
{80,1,40,2,20,2,4,5},
{88,1,44,2,4,11,4,1},
{96,1,32,3,16,2,4,4},
{96,1,48,2,24,2,4,6},
{104,1,52,2,4,13,4,1},
{112,1,56,2,28,2,4,7},
{120,1,60,2,20,3,4,5},
{120,1,40,3,8,5,4,2},
{120,1,24,5,12,2,4,3},
{128,1,64,2,32,2,4,8},
{128,1,64,2,16,4,4,4},
{128,1,16,8,8,2,4,2},
{132,1,44,3,4,11,4,1},
{136,1,68,2,4,17,4,1},
{144,1,72,2,36,2,4,9},
{144,1,48,3,24,2,12,2},
{144,1,48,3,16,3,4,4},
{144,1,16,9,8,2,4,2},
{152,1,76,2,4,19,4,1},
{160,1,80,2,40,2,4,10},
{160,1,80,2,20,4,4,5},
{160,1,32,5,16,2,4,4},
{168,1,84,2,28,3,4,7},
{176,1,88,2,44,2,4,11},
{184,1,92,2,4,23,4,1},
{192,1,96,2,48,2,4,12},
{192,1,96,2,24,4,4,6},
{192,1,64,3,16,4,4,4},
{192,1,24,8,8,3,4,2},
{208,1,104,2,52,2,4,13},
{216,1,108,2,36,3,4,9},
{224,1,112,2,56,2,4,14},
{240,1,120,2,60,2,4,15},
{240,1,80,3,20,4,4,5},
{240,1,48,5,16,3,8,2},
{240,1,24,10,12,2,4,3},
{256,1,128,2,64,2,4,16},
{256,1,128,2,32,4,4,8},
{256,1,16,16,8,2,4,2},
{264,1,132,2,44,3,4,11},
{272,1,136,2,68,2,4,17},
{272,1,68,4,4,17,4,1},
{272,1,16,17,8,2,4,2}};

  size_t SrsRx::getBufferSize(cuphySrsStatPrms_t const* pStatPrms)
 {

     constexpr int32_t EXTRA_PADDING = CUPHY_SRS_MAX_N_USERS * 128; // Upper bound for extra memory required per allocation due to 128 alignment
                                                                    // ToDo FIXME could change this upper limit to be less conservative
     size_t nBytesBuffer = 0;

    // parameters:
    uint16_t nMaxCells                = pStatPrms->nMaxCells;
    cuphyCellStatPrm_t* pCellStatPrms = pStatPrms->pCellStatPrms;
    uint16_t nMaxCellsPerSlot         = pStatPrms->nMaxCellsPerSlot;
    m_nPrbs                           = 273; // pCellStatPrms->nPrbUlBwp; Use max 273 instead of actual nPRBs

    // Memory to store perRb SNR
    nBytesBuffer += CUPHY_SRS_MAX_N_USERS * m_nPrbs * sizeof(float) + EXTRA_PADDING;

    // Memory to store Srs reports
    nBytesBuffer += CUPHY_SRS_MAX_N_USERS * sizeof(cuphySrsReport_t) + EXTRA_PADDING;

    // Memory to store ChEstToL2:
    constexpr uint16_t maxCellsInCellGrp = 100;
    std::array<size_t, maxCellsInCellGrp> maxSrsChEstToL2MemPerCell{};

    for(int cellIdx = 0; cellIdx < nMaxCells; cellIdx++){
#ifdef ASIM_CUPHY_SRS_OUTPUT_FP32
        maxSrsChEstToL2MemPerCell[cellIdx] = m_nPrbs * pCellStatPrms[cellIdx].nRxAntSrs * sizeof(float2) * CUPHY_SRS_MAX_FULL_BAND_SRS_ANT_PORTS_SLOT_PER_CELL * CUPHY_SRS_MAX_FULL_BAND_CHEST_PER_TTI * 2; 
#else
        maxSrsChEstToL2MemPerCell[cellIdx] = m_nPrbs * pCellStatPrms[cellIdx].nRxAntSrs * sizeof(short2) * CUPHY_SRS_MAX_FULL_BAND_SRS_ANT_PORTS_SLOT_PER_CELL * CUPHY_SRS_MAX_FULL_BAND_CHEST_PER_TTI * 2;
#endif
    }
    std::sort(maxSrsChEstToL2MemPerCell.begin(), maxSrsChEstToL2MemPerCell.end(), std::greater<>());

    size_t maxBytesSrsChEstToL2 = 0;
    for(int cellIdx = 0; cellIdx < nMaxCellsPerSlot; ++cellIdx){
        maxBytesSrsChEstToL2 += maxSrsChEstToL2MemPerCell[cellIdx];
    }
    nBytesBuffer += maxBytesSrsChEstToL2 + EXTRA_PADDING;

    // RKHS SRS workspace:
    if(pStatPrms->chEstAlgo == SRS_CH_EST_ALGO_TYPE_RKHS)
    {
        size_t nBytesRkhsWorkspace  = nMaxCells * CUPHY_SRS_RKHS_WORKSPACE_SIZE_PER_CELL;
        nBytesBuffer               += nBytesRkhsWorkspace + EXTRA_PADDING;
    }

     return nBytesBuffer;
 }

 void SrsRx::allocateDescr()
{
    // zero-initialize
    std::array<size_t, N_SRS_COMPONENTS> statDescrSizeBytes{};
    std::array<size_t, N_SRS_COMPONENTS> statDescrAlignBytes{};
    std::array<size_t, N_SRS_COMPONENTS> dynDescrSizeBytes{};
    std::array<size_t, N_SRS_COMPONENTS> dynDescrAlignBytes{};

    size_t* pStatDescrSizeBytes  = statDescrSizeBytes.data();
    size_t* pStatDescrAlignBytes = statDescrAlignBytes.data();
    size_t* pDynDescrSizeBytes   = dynDescrSizeBytes.data();
    size_t* pDynDescrAlignBytes  = dynDescrAlignBytes.data();

    cuphyStatus_t status = cuphySrsChEstGetDescrInfo(&pStatDescrSizeBytes[SRS_CHEST],
                                                      &pStatDescrAlignBytes[SRS_CHEST],
                                                      &pDynDescrSizeBytes[SRS_CHEST],
                                                      &pDynDescrAlignBytes[SRS_CHEST]);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        throw cuphy::cuphy_fn_exception(status, "cuphySrsChEstGetDescrInfo()");
    }

    m_kernelStatDescr.alloc(statDescrSizeBytes, statDescrAlignBytes, &m_memoryFootprint);
    m_kernelDynDescr.alloc(dynDescrSizeBytes, dynDescrAlignBytes, &m_memoryFootprint);
}


void SrsRx::createComponents(cuphySrsFilterPrms_t* pSrsFilterPrms, cudaStream_t cuStream)
{
    auto statCpuDescrStartAddrs      = m_kernelStatDescr.getCpuStartAddrs();
    auto statGpuDescrStartAddrs      = m_kernelStatDescr.getGpuStartAddrs();
    bool enableCpuToGpuDescrAsyncCpy = true;

    cuphyStatus_t statusCreate = cuphyCreateSrsChEst(&m_srsChEstHndl, 
                                                      pSrsFilterPrms,
                                                      m_pRkhsPrms,
                                                      m_chEstAlgo,
                                                      m_chEstToL2NormalizationAlgo,
                                                      m_chEstToL2ConstantScaler,
                                                      m_enableDelayOffsetCorrection,
                                                      static_cast<uint8_t>(enableCpuToGpuDescrAsyncCpy),
                                                      reinterpret_cast<void*>(statCpuDescrStartAddrs[SRS_CHEST]),
                                                      reinterpret_cast<void*>(statGpuDescrStartAddrs[SRS_CHEST]),
                                                      cuStream);
     if(CUPHY_STATUS_SUCCESS != statusCreate)
     {
         throw cuphy::cuphy_fn_exception(statusCreate, "cuphyCreateSrsChEst()");
     }
}

 void SrsRx::allocateDeviceMemory()
 {
    m_outputPrms.d_rbSnrBuffer   = static_cast<float*>(m_LinearAlloc.alloc(m_nSrsUes * m_nPrbs * sizeof(float)));
    m_outputPrms.d_srsReports    = static_cast<cuphySrsReport_t*>(m_LinearAlloc.alloc(m_nSrsUes * sizeof(cuphySrsReport_t)));
    size_t nBytesRkhsWorkspace   = 0;

    for(int ueIdx = 0; ueIdx < m_nSrsUes; ++ueIdx){
        uint16_t cellIdx     = m_hUeSrsPrm[ueIdx].cellIdx;
        uint8_t prgSize      = m_hUeSrsPrm[ueIdx].prgSize;

        uint16_t nRxAntSrs      = m_srsCellPrmsVec[cellIdx].nRxAntSrs;
        uint16_t nPrbGrpsPerHop = SRS_BW_TABLE[m_hUeSrsPrm[ueIdx].configIdx][2*m_hUeSrsPrm[ueIdx].bandwidthIdx] / prgSize;
        uint16_t nHops          = m_hUeSrsPrm[ueIdx].nSyms / m_hUeSrsPrm[ueIdx].nRepetitions;
        uint16_t nAntPorts      = m_hUeSrsPrm[ueIdx].nAntPorts;
#ifdef ASIM_CUPHY_SRS_OUTPUT_FP32
        size_t maxChEstSize           = nPrbGrpsPerHop * nRxAntSrs * nHops * nAntPorts * sizeof(float2);
#else
        size_t maxChEstSize           = nPrbGrpsPerHop * nRxAntSrs * nHops * nAntPorts * sizeof(short2);
#endif
        m_gpuAddrsChEstToL2Vec[ueIdx] = m_LinearAlloc.alloc(maxChEstSize);
        
        size_t maxChEstInnerSize = maxChEstSize;
        if(prgSize > 4)
        {
            prgSize = 2;
            nPrbGrpsPerHop = SRS_BW_TABLE[m_hUeSrsPrm[ueIdx].configIdx][2*m_hUeSrsPrm[ueIdx].bandwidthIdx] / prgSize;
#ifdef ASIM_CUPHY_SRS_OUTPUT_FP32
            maxChEstInnerSize = nPrbGrpsPerHop * nRxAntSrs * nHops * nAntPorts * sizeof(float2);
#else
            maxChEstInnerSize = nPrbGrpsPerHop * nRxAntSrs * nHops * nAntPorts * sizeof(short2);
#endif
        }

        m_gpuAddrsChEstToL2InnerVec[ueIdx] = m_LinearAlloc.alloc(maxChEstInnerSize);

        nBytesRkhsWorkspace += CUPHY_SRS_RKHS_WORKSPACE_SIZE_PER_UE; 
    }

    if(m_chEstAlgo == SRS_CH_EST_ALGO_TYPE_RKHS)
    {
        m_gpuAddrRkhsWorskpace = m_LinearAlloc.alloc(nBytesRkhsWorkspace);
    }
 }


 SrsRx::SrsRx(cuphySrsStatPrms_t const* pStatPrms, cudaStream_t cuStream) :
    m_LinearAlloc(getBufferSize(pStatPrms), &m_memoryFootprint),
    m_gpuAddrsChEstToL2InnerVec(CUPHY_SRS_MAX_N_USERS),
    m_gpuAddrsChEstToL2Vec(CUPHY_SRS_MAX_N_USERS),
    m_srsCellPrmsVec(pStatPrms->nMaxCellsPerSlot),
    m_hCellStatPrms(pStatPrms->pCellStatPrms),
    m_chEstAlgo(pStatPrms->chEstAlgo),
    m_chEstToL2NormalizationAlgo(pStatPrms->chEstToL2NormalizationAlgo),
    m_chEstToL2ConstantScaler(pStatPrms->chEstToL2ConstantScaler),
    m_enableDelayOffsetCorrection(pStatPrms->enableDelayOffsetCorrection),
    m_pRkhsPrms(pStatPrms->pSrsRkhsPrms),
    m_kernelStatDescr("SrsStatDescr"),
    m_kernelDynDescr("SrsDynDescr"),
    m_batchedMemcpyHelper(CUPHY_SRS_MAX_N_USERS + 2, // +2 for the SRS reports and rbSnrBuffer
                          batchedMemcpySrcHint::srcIsDevice, 
                          batchedMemcpyDstHint::dstIsHost, 
                          (SRS_USE_BATCHED_MEMCPY == 1) && (pStatPrms->enableBatchedMemcpy == 1))
 {
    // update static parameter field that points to the cuphyMemoryFootprintTracker object for this channel
    pStatPrms->pOutInfo->pMemoryFootprint = &m_memoryFootprint;

     // ToDo the initialization to zero is to suppress compute-sanitizer initcheck errors.
     // In SRS, output is written sparsely due to frequency hops and in copying back the output to host,
     // a single copy call is used. This triggers initcheck errors in compute-sanitizer. Since none-initialized copied data
     // is not functionally important, the current memset is used as a workaround to suppress such errors. This memset() could
     // potentially be removed in the future, once compute-sanitizer allows to suppress errors/warnings.
     m_LinearAlloc.memset(0, cuStream);
     CUDA_CHECK(cudaStreamSynchronize(cuStream));

    // Allocate descriptors for pipeline usage
    allocateDescr();

    cuphySrsFilterPrms_t srsFilterPrms = pStatPrms->srsFilterPrms;
    createComponents(&srsFilterPrms ,cuStream);

    createGraph();
#if CUDA_VERSION >= 12000    
    CU_CHECK_EXCEPTION(cuGraphInstantiate(&m_graphExec, m_graph, 0));
#else            
    CU_CHECK_EXCEPTION(cuGraphInstantiate(&m_graphExec, m_graph, 0, 0, 0));
#endif

    // Debug parameters
    m_outputPrms.debugOutputFlag = false;
    if(nullptr != pStatPrms->pStatDbg)  // TODO: enable after CP adds debug API parameters
    {
        if(pStatPrms->pStatDbg->pOutFileName != nullptr)
        {
            m_outputPrms.debugOutputFlag = true;
            m_outputPrms.outHdf5File     = hdf5hpp::hdf5_file::open(pStatPrms->pStatDbg->pOutFileName);
        }
    }

    // print memory info
    if (PRINT_GPU_MEMORY_CUPHY_CHANNEL == 1) 
    {
        m_memoryFootprint.printMemoryFootprint(this, "SRS");
    }
 }

 void SrsRx::createGraph()
 {
     // add node(s), initially start with some kernel parameters, at setup, do the updating
     CU_CHECK_EXCEPTION(cuGraphCreate(&m_graph, 0));
     CUPHY_CHECK(cuphySetEmptyKernelNodeParams(&m_emptyNode0ParamsDriver));
     CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&m_srsKernelNode, m_graph, nullptr, 0, &m_emptyNode0ParamsDriver));
     CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&m_srsNormalizationKernelNode, m_graph, &m_srsKernelNode, 1, &m_emptyNode0ParamsDriver));
     // add dependencies
     // as there is only one kernel node, for now dependencies not used

 }

 void SrsRx::updateGraph() const
 {
     MemtraceDisableScope md;
     CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_srsKernelNode, &(m_srsChEstLaunchCfg.kernelNodeParamsDriver)));
     CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_srsNormalizationKernelNode, &(m_srsChEstNormalizationLaunchCfg.kernelNodeParamsDriver)));
 }

 cuphyStatus_t SrsRx::setupComponents(bool enableCpuToGpuDescrAsyncCpy, const cuphySrsDynPrms_t *pDynPrm)
 {
    auto dynCpuDescrStartAddrs = m_kernelDynDescr.getCpuStartAddrs();
    auto dynGpuDescrStartAddrs = m_kernelDynDescr.getGpuStartAddrs();

    cuphyStatus_t srsChEstSetupStatus = cuphySetupSrsChEst(m_srsChEstHndl,
                                                            m_nSrsUes,
                                                            m_hUeSrsPrm,
                                                            m_nCells,
                                                            m_hPrmDataRx,
                                                            m_srsCellPrmsVec.data(), 
                                                            m_outputPrms.d_rbSnrBuffer,
                                                            m_outputPrms.h_rbSnrBuffOffsets,
                                                            m_outputPrms.d_srsReports,
                                                            m_outputPrms.h_chEstBuffInfo,
                                                            m_gpuAddrsChEstToL2InnerVec.data(),
                                                            m_gpuAddrsChEstToL2Vec.data(),
                                                            m_outputPrms.h_srsChEstToL2,
                                                            m_gpuAddrRkhsWorskpace,
                                                            static_cast<uint8_t>(enableCpuToGpuDescrAsyncCpy),
                                                            dynCpuDescrStartAddrs[SRS_CHEST],                     
                                                            dynGpuDescrStartAddrs[SRS_CHEST], 
                                                            &m_srsChEstLaunchCfg,
                                                            &m_srsChEstNormalizationLaunchCfg,
                                                            m_cuStream);


    if(CUPHY_STATUS_SUCCESS != srsChEstSetupStatus)
    {
        pDynPrm->pStatusOut->status = cuphySrsStatusType_t::CUPHY_SRS_STATUS_CHEST_SETUP_ERROR;
        pDynPrm->pStatusOut->ueIdx = MAX_UINT16;
        pDynPrm->pStatusOut->cellPrmStatIdx = MAX_UINT16; 
        return CUPHY_STATUS_INTERNAL_ERROR;
        //throw cuphy::cuphy_fn_exception(srsChEstSetupStatus, "srsChEstSetupStatus()");
    }
    
    return CUPHY_STATUS_SUCCESS;
 }

void SrsRx::setupCmn(const cuphySrsDynPrms_t *pDynPrm)
{
    // reset linear allocation:
    m_LinearAlloc.reset();

    // extract stream:
    m_cuStream   = pDynPrm->cuStream;
    m_nSrsUes    = pDynPrm->pCellGrpDynPrm->nSrsUes;
    m_nCells     = pDynPrm->pCellGrpDynPrm->nCells;
    m_hUeSrsPrm  = pDynPrm->pCellGrpDynPrm->pUeSrsPrms;
    m_hPrmDataRx = pDynPrm->pDataIn->pTDataRx;

    // srs cell parameters
    cuphySrsCellDynPrm_t* pCellPrms = pDynPrm->pCellGrpDynPrm->pCellPrms;
    for(int dynCellIdx = 0; dynCellIdx < m_nCells; ++dynCellIdx)
    {
        uint16_t statCellIdx = pCellPrms[dynCellIdx].cellPrmStatIdx;

        m_srsCellPrmsVec[dynCellIdx].slotNum      = pCellPrms[dynCellIdx].slotNum;
        m_srsCellPrmsVec[dynCellIdx].frameNum     = pCellPrms[dynCellIdx].frameNum;
        m_srsCellPrmsVec[dynCellIdx].srsStartSym  = pCellPrms[dynCellIdx].srsStartSym;
        m_srsCellPrmsVec[dynCellIdx].nSrsSym      = pCellPrms[dynCellIdx].nSrsSym;
        m_srsCellPrmsVec[dynCellIdx].nRxAntSrs    = m_hCellStatPrms[statCellIdx].nRxAntSrs;
        m_srsCellPrmsVec[dynCellIdx].mu           = m_hCellStatPrms[statCellIdx].mu;
    }

    // Output parameters:
    m_outputPrms.cpuCopyOn           = pDynPrm->cpuCopyOn;
    m_outputPrms.pDataRxSrsHost      = pDynPrm->pDataOut->pDataRxSrs;
    m_outputPrms.h_chEstBuffInfo     = pDynPrm->pDataOut->pChEstBuffInfo;
    m_outputPrms.h_srsReports        = pDynPrm->pDataOut->pSrsReports;
    m_outputPrms.h_rbSnrBuffer       = pDynPrm->pDataOut->pRbSnrBuffer;
    m_outputPrms.h_rbSnrBuffOffsets  = pDynPrm->pDataOut->pRbSnrBuffOffsets;
    m_outputPrms.h_srsChEstToL2      = pDynPrm->pDataOut->pSrsChEstToL2;
    for(int ueIdx = 0; ueIdx < m_nSrsUes; ++ueIdx)
    {
        m_outputPrms.h_rbSnrBuffOffsets[ueIdx] = m_nPrbs * ueIdx;
    }

    // allocate GPU buffers:
    allocateDeviceMemory();
}

 cuphyStatus_t SrsRx::setup(const cuphySrsDynPrms_t *pDynPrm)
 {
    // common setup (shared by both stream and graph modes)
    setupCmn(pDynPrm);

    // stream setup. TODO: add graph setup.
    bool enableCpuToGpuDescrAsyncCpy = false;
    cuphyStatus_t status = setupComponents(enableCpuToGpuDescrAsyncCpy, pDynPrm);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        return status;
    }

    // cuda graph setup
    m_cudaGraphModeEnabled = (pDynPrm->procModeBmsk & SRS_PROC_MODE_FULL_SLOT_GRAPHS) ? true : false;
    if (m_cudaGraphModeEnabled) {
        updateGraph();
    }
    
    // copy descriptors to GPU
    if(!enableCpuToGpuDescrAsyncCpy)
     {
         m_kernelDynDescr.asyncCpuToGpuCpy(m_cuStream);
     }
    return CUPHY_STATUS_SUCCESS;
 }

 cuphyStatus_t SrsRx::copyOutputToCPU(cudaStream_t cuStream) {
    if(m_nSrsUes > 0) {
        m_batchedMemcpyHelper.reset(); // reset for upcoming batch of updateMemcpy calls

        // Add raw SRS IQ copy (buf_st_2 GPU->host) for DataLake. The input descriptor is
        // sized for the worst-case antenna ports, but buf_st_2 only holds nRxAntSrs ports,
        // so copy per cell using the active ports to avoid over-reading the source.
        if(m_outputPrms.pDataRxSrsHost) {
            int dataRxDims[3] = {0, 0, 0};
            cuphyGetTensorDescriptor(m_hPrmDataRx[0].desc, 3, nullptr, nullptr, dataRxDims, nullptr);
            size_t perAntElems = static_cast<size_t>(dataRxDims[0]) * dataRxDims[1];
            size_t dstOffsetElems = 0;
            for(int32_t cellIdx = 0; cellIdx < m_nCells; ++cellIdx) {
                size_t cellElems = perAntElems * m_srsCellPrmsVec[cellIdx].nRxAntSrs;
                m_batchedMemcpyHelper.updateMemcpy(
                    m_outputPrms.pDataRxSrsHost + dstOffsetElems,
                    m_hPrmDataRx[cellIdx].pAddr,
                    cellElems * sizeof(__half2),
                    cudaMemcpyDeviceToHost,
                    cuStream
                );
                dstOffsetElems += cellElems;
            }
        }

        // Add SRS reports copy
        m_batchedMemcpyHelper.updateMemcpy(
            m_outputPrms.h_srsReports,
            m_outputPrms.d_srsReports,
            sizeof(cuphySrsReport_t) * m_nSrsUes,
            cudaMemcpyDeviceToHost,
            cuStream
        );

        // Add RB SNR buffer copy
        m_batchedMemcpyHelper.updateMemcpy(
            m_outputPrms.h_rbSnrBuffer,
            m_outputPrms.d_rbSnrBuffer,
            sizeof(float) * m_nSrsUes * m_nPrbs,
            cudaMemcpyDeviceToHost,
            cuStream
        );

        // Add channel estimates copies for each UE
        for(int ueIdx = 0; ueIdx < m_nSrsUes; ++ueIdx) {
            uint16_t nRxAntSrs = m_srsCellPrmsVec[m_hUeSrsPrm[ueIdx].cellIdx].nRxAntSrs;
            uint16_t nPrbGrps = m_outputPrms.h_srsChEstToL2[ueIdx].nPrbGrps;
            uint8_t nAntPorts = m_hUeSrsPrm[ueIdx].nAntPorts;

#ifdef ASIM_CUPHY_SRS_OUTPUT_FP32
            size_t chEstToL2MemSize = nRxAntSrs * nPrbGrps * nAntPorts * sizeof(float2);
#else
            size_t chEstToL2MemSize = nRxAntSrs * nPrbGrps * nAntPorts * sizeof(short2);
#endif

            m_batchedMemcpyHelper.updateMemcpy(
                m_outputPrms.h_srsChEstToL2[ueIdx].pChEstCpuBuff,
                m_gpuAddrsChEstToL2Vec[ueIdx],
                chEstToL2MemSize,
                cudaMemcpyDeviceToHost,
                cuStream
            );
        }

        // Launch all batched copies
        cuphyStatus_t status = m_batchedMemcpyHelper.launchBatchedMemcpy(cuStream);
        if(status != CUPHY_STATUS_SUCCESS)
        {
            NVLOGE_FMT(NVLOG_SRS, AERIAL_CUPHY_EVENT, "Launching batched memcpy for SRS returned an error");
            return status;
        }
    }
    return CUPHY_STATUS_SUCCESS;
 }

 cuphyStatus_t SrsRx::run()
 {
        cuphyStatus_t status = CUPHY_STATUS_SUCCESS;
        if (m_cudaGraphModeEnabled)
        {
            MemtraceDisableScope md;
            CU_CHECK_EXCEPTION(cuGraphLaunch(m_graphExec, m_cuStream));
        } else
        {
            CUresult srsChEstRunStatus = launch_kernel(m_srsChEstLaunchCfg.kernelNodeParamsDriver, m_cuStream);
            if(CUDA_SUCCESS != srsChEstRunStatus) throw cuphy::cuphy_exception(CUPHY_STATUS_INTERNAL_ERROR);
            
            CUresult srsChEstNormalizationRunStatus = launch_kernel(m_srsChEstNormalizationLaunchCfg.kernelNodeParamsDriver, m_cuStream);
            if(CUDA_SUCCESS != srsChEstNormalizationRunStatus) throw cuphy::cuphy_exception(CUPHY_STATUS_INTERNAL_ERROR);
        }

        if(m_outputPrms.cpuCopyOn)
        {
            status = copyOutputToCPU(m_cuStream); // OK not to return since currently last
        }
        return status;
 }

 void SrsRx::destroyComponents() const
{
    cuphyStatus_t statusDestroy = cuphyDestroySrsChEst(m_srsChEstHndl);
    if(CUPHY_STATUS_SUCCESS != statusDestroy)
    {
        printf("cuphyDestroySrsChEst() error %d\n", statusDestroy);
    }
}

const void* SrsRx::getMemoryTracker() const
{
   return &m_memoryFootprint;
}

SrsRx::~SrsRx()
{
    CUDA_CHECK_NO_THROW(cudaGraphDestroy(m_graph));
    CUDA_CHECK_NO_THROW(cudaGraphExecDestroy(m_graphExec));
    destroyComponents();
}

void SrsRx::printStaticApiPrms(cuphySrsStatPrms_t const* pStaticPrms)
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

void SrsRx::printDynApiPrms(const cuphySrsDynPrms_t* pDynPrm)
{
    printf("===============================================\n");
    printf("============print SRS DynApiPrms===============\n");
    printf("procModeBmsk: %lx\n", pDynPrm->procModeBmsk);

    const cuphySrsCellGrpDynPrm_t* pCellGrpDynPrm = pDynPrm->pCellGrpDynPrm;
    printf("nCells: %d\n", pCellGrpDynPrm->nCells);
    printf("===============================================\n");
    for (uint16_t i = 0 ; i < pCellGrpDynPrm->nCells; i++)
    {
        printf("-->Cell[%d]\n", i);
        cuphySrsCellDynPrm_t* pCellDynPrm = &pCellGrpDynPrm->pCellPrms[i];
        printf("pCellPrms: %p\n", pCellDynPrm);
        printf("cellPrmStatIdx: %d\n", pCellDynPrm->cellPrmStatIdx);
        printf("cellPrmDynIdx: %d\n", pCellDynPrm->cellPrmDynIdx);
        printf("slotNum: %d\n", pCellDynPrm->slotNum);
        printf("frameNum: %d\n", pCellDynPrm->frameNum);
        printf("srsStartSym: %d\n", pCellDynPrm->srsStartSym);
        printf("nSrsSym: %d\n", pCellDynPrm->nSrsSym);
    }

    printf("===============================================\n");
    printf("nUes: %d\n",  pCellGrpDynPrm->nSrsUes);
    printf("===============================================\n");
    printf("===============================================\n");
    for (uint16_t i = 0; i < pCellGrpDynPrm->nSrsUes; i++)
    {
        printf("-->UE[%d]\n", i);
        const cuphyUeSrsPrm_t* pUeSrsPrms = &pCellGrpDynPrm->pUeSrsPrms[i];
        printf("pUePrms: %p\n",  pUeSrsPrms);
        printf("cellIdx: %d\n",  pUeSrsPrms->cellIdx);
        printf("nAntPorts: %d\n",  pUeSrsPrms->nAntPorts);
        printf("nSyms: %d\n",  pUeSrsPrms->nSyms);
        printf("nRepetitions: %d\n",  pUeSrsPrms->nRepetitions);
        printf("combSize: %d\n",  pUeSrsPrms->combSize);
        printf("startSym: %d\n",  pUeSrsPrms->startSym);
        printf("sequenceId: %d\n",  pUeSrsPrms->sequenceId);
        printf("configIdx: %d\n",  pUeSrsPrms->configIdx);
        printf("bandwidthIdx: %d\n",  pUeSrsPrms->bandwidthIdx);
        printf("combOffset: %d\n",  pUeSrsPrms->combOffset);
        printf("cyclicShift: %d\n",  pUeSrsPrms->cyclicShift);
        printf("frequencyPosition: %d\n",  pUeSrsPrms->frequencyPosition);
        printf("frequencyShift: %d\n",  pUeSrsPrms->frequencyShift);
        printf("frequencyHopping: %d\n",  pUeSrsPrms->frequencyHopping);
        printf("resourceType: %d\n",  pUeSrsPrms->resourceType);
        printf("Tsrs: %d\n",  pUeSrsPrms->Tsrs);
        printf("Toffset: %d\n",  pUeSrsPrms->Toffset);
        printf("groupOrSequenceHopping: %d\n",  pUeSrsPrms->groupOrSequenceHopping);
        printf("chEstBuffIdx: %d\n",  pUeSrsPrms->chEstBuffIdx);
        printf("srsAntPortToUeAntMap[0]: %d\n",  pUeSrsPrms->srsAntPortToUeAntMap[0]);
        printf("srsAntPortToUeAntMap[1]: %d\n",  pUeSrsPrms->srsAntPortToUeAntMap[1]);
        printf("srsAntPortToUeAntMap[2]: %d\n",  pUeSrsPrms->srsAntPortToUeAntMap[2]);
        printf("srsAntPortToUeAntMap[3]: %d\n",  pUeSrsPrms->srsAntPortToUeAntMap[3]);
        printf("usage: %d\n",  pUeSrsPrms->usage);
        printf("srsChestBufferIndexL2: %d\n",  pUeSrsPrms->srsChestBufferIndexL2);
        printf("srsStartPrg: %d\n", pUeSrsPrms->srsStartPrg);
        printf("prgSize: %d\n", pUeSrsPrms->prgSize);
    }

    printf("===============================================\n");
}


void SrsRx::writeDbgBufSynch(cudaStream_t cuStrm)
{
        if(m_outputPrms.debugOutputFlag)
    {
        using cpuCuPhyBufU32_t = cuphy::buffer<uint32_t, cuphy::pinned_alloc>;
        using cpuCuPhyBufF32_t = cuphy::buffer<float, cuphy::pinned_alloc>;

        // init user parameters:
        cpuCuPhyBufU32_t cellIdx = std::move(cuphy::buffer<uint32_t, cuphy::pinned_alloc>(m_nSrsUes));
        cpuCuPhyBufU32_t  nAntPorts = std::move(cuphy::buffer<uint32_t, cuphy::pinned_alloc>(m_nSrsUes));
        cpuCuPhyBufU32_t  nSyms = std::move(cuphy::buffer<uint32_t, cuphy::pinned_alloc>(m_nSrsUes));
        cpuCuPhyBufU32_t  nRepetitions = std::move(cuphy::buffer<uint32_t, cuphy::pinned_alloc>(m_nSrsUes));
        cpuCuPhyBufU32_t  combSize = std::move(cuphy::buffer<uint32_t, cuphy::pinned_alloc>(m_nSrsUes));
        cpuCuPhyBufU32_t  startSym = std::move(cuphy::buffer<uint32_t, cuphy::pinned_alloc>(m_nSrsUes));
        cpuCuPhyBufU32_t  sequenceId = std::move(cuphy::buffer<uint32_t, cuphy::pinned_alloc>(m_nSrsUes));
        cpuCuPhyBufU32_t  configIdx = std::move(cuphy::buffer<uint32_t, cuphy::pinned_alloc>(m_nSrsUes));
        cpuCuPhyBufU32_t  bandwidthIdx = std::move(cuphy::buffer<uint32_t, cuphy::pinned_alloc>(m_nSrsUes));
        cpuCuPhyBufU32_t  combOffset = std::move(cuphy::buffer<uint32_t, cuphy::pinned_alloc>(m_nSrsUes));
        cpuCuPhyBufU32_t  cyclicShift = std::move(cuphy::buffer<uint32_t, cuphy::pinned_alloc>(m_nSrsUes));
        cpuCuPhyBufU32_t  frequencyPosition = std::move(cuphy::buffer<uint32_t, cuphy::pinned_alloc>(m_nSrsUes));
        cpuCuPhyBufU32_t  frequencyShift = std::move(cuphy::buffer<uint32_t, cuphy::pinned_alloc>(m_nSrsUes));
        cpuCuPhyBufU32_t  frequencyHopping = std::move(cuphy::buffer<uint32_t, cuphy::pinned_alloc>(m_nSrsUes));
        cpuCuPhyBufU32_t  resourceType = std::move(cuphy::buffer<uint32_t, cuphy::pinned_alloc>(m_nSrsUes));
        cpuCuPhyBufU32_t  Tsrs = std::move(cuphy::buffer<uint32_t, cuphy::pinned_alloc>(m_nSrsUes));
        cpuCuPhyBufU32_t  Toffset = std::move(cuphy::buffer<uint32_t, cuphy::pinned_alloc>(m_nSrsUes));
        cpuCuPhyBufU32_t  groupOrSequenceHopping = std::move(cuphy::buffer<uint32_t, cuphy::pinned_alloc>(m_nSrsUes));
        cpuCuPhyBufU32_t  chEstBuffIdx = std::move(cuphy::buffer<uint32_t, cuphy::pinned_alloc>(m_nSrsUes));
        cpuCuPhyBufU32_t  srsAntPortToUeAntMap0 = std::move(cuphy::buffer<uint32_t, cuphy::pinned_alloc>(m_nSrsUes));
        cpuCuPhyBufU32_t  srsAntPortToUeAntMap1 = std::move(cuphy::buffer<uint32_t, cuphy::pinned_alloc>(m_nSrsUes));  
        cpuCuPhyBufU32_t  srsAntPortToUeAntMap2 = std::move(cuphy::buffer<uint32_t, cuphy::pinned_alloc>(m_nSrsUes));  
        cpuCuPhyBufU32_t  srsAntPortToUeAntMap3 = std::move(cuphy::buffer<uint32_t, cuphy::pinned_alloc>(m_nSrsUes));
        cpuCuPhyBufU32_t prgSize = std::move(cuphy::buffer<uint32_t, cuphy::pinned_alloc>(m_nSrsUes));
        cpuCuPhyBufU32_t  usage = std::move(cuphy::buffer<uint32_t, cuphy::pinned_alloc>(m_nSrsUes));
        cpuCuPhyBufU32_t  srsChestBufferIndexL2 = std::move(cuphy::buffer<uint32_t, cuphy::pinned_alloc>(m_nSrsUes));

        // init cell parameters
        cpuCuPhyBufU32_t slotNum = std::move(cuphy::buffer<uint32_t, cuphy::pinned_alloc>(m_nSrsUes));
        cpuCuPhyBufU32_t frameNum = std::move(cuphy::buffer<uint32_t, cuphy::pinned_alloc>(m_nSrsUes));
        cpuCuPhyBufU32_t srsStartSym = std::move(cuphy::buffer<uint32_t, cuphy::pinned_alloc>(m_nSrsUes));
        cpuCuPhyBufU32_t nSrsSym = std::move(cuphy::buffer<uint32_t, cuphy::pinned_alloc>(m_nSrsUes));
        cpuCuPhyBufU32_t nRxAntSrs = std::move(cuphy::buffer<uint32_t, cuphy::pinned_alloc>(m_nSrsUes)); 
        cpuCuPhyBufU32_t mu = std::move(cuphy::buffer<uint32_t, cuphy::pinned_alloc>(m_nSrsUes)); 

        // init SRS report parameters:
        cpuCuPhyBufF32_t toEstMicroSec = std::move(cuphy::buffer<float, cuphy::pinned_alloc>(m_nSrsUes)); 
        cpuCuPhyBufF32_t widebandSnr = std::move(cuphy::buffer<float, cuphy::pinned_alloc>(m_nSrsUes));
        cpuCuPhyBufF32_t widebandNoiseEnergy = std::move(cuphy::buffer<float, cuphy::pinned_alloc>(m_nSrsUes)); 
        cpuCuPhyBufF32_t widebandSignalEnergy = std::move(cuphy::buffer<float, cuphy::pinned_alloc>(m_nSrsUes)); 
        cpuCuPhyBufF32_t widebandCsCorrRatioDb = std::move(cuphy::buffer<float, cuphy::pinned_alloc>(m_nSrsUes));
        cpuCuPhyBufF32_t widebandCsCorrUse = std::move(cuphy::buffer<float, cuphy::pinned_alloc>(m_nSrsUes)); 
        cpuCuPhyBufF32_t widebandCsCorrNotUse = std::move(cuphy::buffer<float, cuphy::pinned_alloc>(m_nSrsUes)); 

        // copy user parameters:
        for(int ueIdx = 0; ueIdx < m_nSrsUes; ++ueIdx)
        {
            cellIdx[ueIdx] = m_hUeSrsPrm[ueIdx].cellIdx;
            nAntPorts[ueIdx] = m_hUeSrsPrm[ueIdx].nAntPorts;
            nSyms[ueIdx] = m_hUeSrsPrm[ueIdx].nSyms;
            nRepetitions[ueIdx] = m_hUeSrsPrm[ueIdx].nRepetitions;
            combSize[ueIdx] = m_hUeSrsPrm[ueIdx].combSize;
            startSym[ueIdx] = m_hUeSrsPrm[ueIdx].startSym;
            sequenceId[ueIdx] = m_hUeSrsPrm[ueIdx].sequenceId;
            configIdx[ueIdx] = m_hUeSrsPrm[ueIdx].configIdx;
            bandwidthIdx[ueIdx] = m_hUeSrsPrm[ueIdx].bandwidthIdx;
            combOffset[ueIdx] = m_hUeSrsPrm[ueIdx].combOffset;
            cyclicShift[ueIdx] = m_hUeSrsPrm[ueIdx].cyclicShift;
            frequencyPosition[ueIdx] = m_hUeSrsPrm[ueIdx].frequencyPosition;
            frequencyShift[ueIdx] = m_hUeSrsPrm[ueIdx].frequencyShift;
            frequencyHopping[ueIdx] = m_hUeSrsPrm[ueIdx].frequencyHopping;
            resourceType[ueIdx] = m_hUeSrsPrm[ueIdx].resourceType;
            Tsrs[ueIdx] = m_hUeSrsPrm[ueIdx].Tsrs;
            Toffset[ueIdx] = m_hUeSrsPrm[ueIdx].Toffset;
            groupOrSequenceHopping[ueIdx] = m_hUeSrsPrm[ueIdx].groupOrSequenceHopping;
            chEstBuffIdx[ueIdx] = m_hUeSrsPrm[ueIdx].chEstBuffIdx;
            srsAntPortToUeAntMap0[ueIdx] = m_hUeSrsPrm[ueIdx].srsAntPortToUeAntMap[0];
            srsAntPortToUeAntMap1[ueIdx] = m_hUeSrsPrm[ueIdx].srsAntPortToUeAntMap[1];
            srsAntPortToUeAntMap2[ueIdx] = m_hUeSrsPrm[ueIdx].srsAntPortToUeAntMap[2];
            srsAntPortToUeAntMap3[ueIdx] = m_hUeSrsPrm[ueIdx].srsAntPortToUeAntMap[3];
            prgSize[ueIdx]               = m_hUeSrsPrm[ueIdx].prgSize;
            usage[ueIdx] = m_hUeSrsPrm[ueIdx].usage;
            srsChestBufferIndexL2[ueIdx] =  m_hUeSrsPrm[ueIdx].srsChestBufferIndexL2;
        }

        // copy cell parameters:
        for(int cellIdx = 0; cellIdx < m_nCells; ++cellIdx)
        {
            slotNum[cellIdx] = m_srsCellPrmsVec[cellIdx].slotNum;
            frameNum[cellIdx] = m_srsCellPrmsVec[cellIdx].frameNum;
            srsStartSym[cellIdx] = m_srsCellPrmsVec[cellIdx].srsStartSym;
            nSrsSym[cellIdx] = m_srsCellPrmsVec[cellIdx].nSrsSym;
            nRxAntSrs[cellIdx] = m_srsCellPrmsVec[cellIdx].nRxAntSrs;
            mu[cellIdx] = m_srsCellPrmsVec[cellIdx].mu;
        }

        // Copy srs report:
        for(int ueIdx = 0; ueIdx < m_nSrsUes; ++ueIdx)
        {
            toEstMicroSec[ueIdx]          = m_outputPrms.h_srsReports[ueIdx].toEstMicroSec; 
            widebandSnr[ueIdx]            = m_outputPrms.h_srsReports[ueIdx].widebandSnr;
            widebandNoiseEnergy[ueIdx]    = m_outputPrms.h_srsReports[ueIdx].widebandNoiseEnergy;
            widebandSignalEnergy[ueIdx]   = m_outputPrms.h_srsReports[ueIdx].widebandSignalEnergy;
            widebandCsCorrRatioDb[ueIdx]  = m_outputPrms.h_srsReports[ueIdx].widebandCsCorrRatioDb;
            widebandCsCorrUse[ueIdx]      = m_outputPrms.h_srsReports[ueIdx].widebandCsCorrUse;
            widebandCsCorrNotUse[ueIdx]   = m_outputPrms.h_srsReports[ueIdx].widebandCsCorrNotUse;            
        }

        // write user parameters to H5:
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, "cellIdx", CUPHY_R_32U, m_nSrsUes, cellIdx.addr());
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, "nAntPorts", CUPHY_R_32U, m_nSrsUes, nAntPorts.addr());
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, "nSyms", CUPHY_R_32U, m_nSrsUes, nSyms.addr());
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, "nRepetitions", CUPHY_R_32U, m_nSrsUes, nRepetitions.addr());
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, "combSize", CUPHY_R_32U, m_nSrsUes, combSize.addr());
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, "startSym", CUPHY_R_32U, m_nSrsUes, startSym.addr());
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, "sequenceId", CUPHY_R_32U, m_nSrsUes, sequenceId.addr());
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, "configIdx", CUPHY_R_32U, m_nSrsUes, configIdx.addr());
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, "bandwidthIdx", CUPHY_R_32U, m_nSrsUes, bandwidthIdx.addr());
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, "combOffset", CUPHY_R_32U, m_nSrsUes, combOffset.addr());
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, "cyclicShift", CUPHY_R_32U, m_nSrsUes, cyclicShift.addr());
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, "frequencyPosition", CUPHY_R_32U, m_nSrsUes, frequencyPosition.addr());
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, "frequencyShift", CUPHY_R_32U, m_nSrsUes, frequencyShift.addr());
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, "frequencyHopping", CUPHY_R_32U, m_nSrsUes, frequencyHopping.addr());
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, "resourceType", CUPHY_R_32U, m_nSrsUes, resourceType.addr());
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, "Tsrs", CUPHY_R_32U, m_nSrsUes, Tsrs.addr());
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, "Toffset", CUPHY_R_32U, m_nSrsUes, Toffset.addr());
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, "groupOrSequenceHopping", CUPHY_R_32U, m_nSrsUes, groupOrSequenceHopping.addr());
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, "chEstBuffIdx", CUPHY_R_32U, m_nSrsUes, chEstBuffIdx.addr());
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, "srsAntPortToUeAntMap0", CUPHY_R_32U, m_nSrsUes, srsAntPortToUeAntMap0.addr());
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, "srsAntPortToUeAntMap1", CUPHY_R_32U, m_nSrsUes, srsAntPortToUeAntMap1.addr());
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, "srsAntPortToUeAntMap2", CUPHY_R_32U, m_nSrsUes, srsAntPortToUeAntMap2.addr());
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, "srsAntPortToUeAntMap3", CUPHY_R_32U, m_nSrsUes, srsAntPortToUeAntMap3.addr());
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, "prgSize", CUPHY_R_32U, m_nSrsUes, prgSize.addr());
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, "usage", CUPHY_R_32U, m_nSrsUes, usage.addr());

        // write cell parameters to H5:
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, "slotNum", CUPHY_R_32U, m_nCells, slotNum.addr());
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, "frameNum", CUPHY_R_32U, m_nCells, frameNum.addr());
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, "srsStartSym", CUPHY_R_32U, m_nCells, srsStartSym.addr());
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, "nSrsSym", CUPHY_R_32U, m_nCells, nSrsSym.addr());
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, "nRxAntSrs", CUPHY_R_32U, m_nCells, nRxAntSrs.addr());
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, "mu", CUPHY_R_32U, m_nCells, mu.addr());

        // write SRS reports to H5:
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, "toEstMicroSec", CUPHY_R_32F, m_nSrsUes, toEstMicroSec.addr());
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, "widebandSnr", CUPHY_R_32F, m_nSrsUes, widebandSnr.addr());
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, "widebandNoiseEnergy", CUPHY_R_32F, m_nSrsUes, widebandNoiseEnergy.addr());
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, "widebandSignalEnergy", CUPHY_R_32F, m_nSrsUes, widebandSignalEnergy.addr());
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, "widebandCsCorrRatioDb", CUPHY_R_32F, m_nSrsUes, widebandCsCorrRatioDb.addr());
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, "widebandCsCorrUse", CUPHY_R_32F, m_nSrsUes, widebandCsCorrUse.addr());
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, "widebandCsCorrNotUse", CUPHY_R_32F, m_nSrsUes, widebandCsCorrNotUse.addr());

        // Copy rx data to H5:
        for(int cellIdx = 0; cellIdx < m_nCells; ++cellIdx)
        {
            cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, m_hPrmDataRx[cellIdx], std::string("dataRx" + std::to_string(cellIdx)).c_str(), cuStrm);
        }

        // Copy ChEsts to H5:
        cuphySrsChEstBuffInfo_t*  h_chEstBuffInfo = m_outputPrms.h_chEstBuffInfo;
        for(int ueIdx = 0; ueIdx < m_nSrsUes; ++ueIdx)
        {
            uint16_t chEstBuffIdx      = m_hUeSrsPrm[ueIdx].chEstBuffIdx;
            cuphyTensorPrm_t tPrmChEst = h_chEstBuffInfo[chEstBuffIdx].tChEstBuffer;
            cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, tPrmChEst, std::string("chEst" + std::to_string(ueIdx)).c_str(), cuStrm); 
        }

        // Copy ChEstToL2 to H5:
        for(int ueIdx = 0; ueIdx < m_nSrsUes; ++ueIdx)
        {
            uint16_t nPrbGrpsPerHop = SRS_BW_TABLE[m_hUeSrsPrm[ueIdx].configIdx][2*m_hUeSrsPrm[ueIdx].bandwidthIdx] / prgSize[ueIdx];
            uint16_t nHops          = m_hUeSrsPrm[ueIdx].nSyms / m_hUeSrsPrm[ueIdx].nRepetitions;

            int nRxAntSrs = m_srsCellPrmsVec[m_hUeSrsPrm[ueIdx].cellIdx].nRxAntSrs;
            int nPrbGrps  = nPrbGrpsPerHop * nHops;
            int nAntPorts = m_hUeSrsPrm[ueIdx].nAntPorts;

            cuphy::tensor_ref tRefChEstToL2;
            tRefChEstToL2.desc().set(CUPHY_C_16I, nPrbGrps, nRxAntSrs, nAntPorts, cuphy::tensor_flags::align_tight);
            tRefChEstToL2.set_addr(m_gpuAddrsChEstToL2Vec[ueIdx]);

            cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, tRefChEstToL2, tRefChEstToL2.desc(), std::string("chEstNormToL2" + std::to_string(ueIdx)).c_str(), cuStrm); 
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// cuphySetupSrsRx()

cuphyStatus_t CUPHYWINAPI cuphySetupSrsRx(cuphySrsRxHndl_t srsRxHndl, const cuphySrsDynPrms_t* pDynPrms, [[maybe_unused]] cuphySrsBatchPrmHndl_t const batchPrmHndl)
 {
     if(!srsRxHndl || !pDynPrms)
     {
         return CUPHY_STATUS_INVALID_ARGUMENT;
     }
     
     return cuphy::tryCallableAndCatch([&]
     {
         if(pDynPrms->pDynDbg->enableApiLogging) 
         {
            SrsRx::printDynApiPrms(pDynPrms);
         }
         auto* p = static_cast<SrsRx*>(srsRxHndl);
         return p->setup(pDynPrms);
     });
 }


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// cuphyCreateSrsRx()

cuphyStatus_t CUPHYWINAPI cuphyCreateSrsRx(cuphySrsRxHndl_t* pSrsRxHndl, cuphySrsStatPrms_t const* pStatPrms, cudaStream_t cuStream)
{
    if(!pSrsRxHndl || !pStatPrms || !cuStream)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    *pSrsRxHndl = nullptr;
    return cuphy::tryCallableAndCatch([&]
    {
        if (pStatPrms->pStatDbg->enableApiLogging) { // TODO: uncomment once cuPHY-CP populates API logging parameter
            SrsRx::printStaticApiPrms(pStatPrms);
        }
        auto* p    = new SrsRx(pStatPrms, cuStream);
        *pSrsRxHndl = static_cast<cuphySrsRxHndl_t>(p);
    });
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
 // cuphyRunSrsRx()

 cuphyStatus_t CUPHYWINAPI cuphyRunSrsRx(cuphySrsRxHndl_t srsRxHndl, uint64_t procModeBmsk)
 {
     if(!srsRxHndl)
     {
         return CUPHY_STATUS_INVALID_ARGUMENT;
     }
     
     return cuphy::tryCallableAndCatch([&]
     {
         auto* p = static_cast<SrsRx*>(srsRxHndl);
         return p->run();
     });
 }

 //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// cuphyWriteDbgBufSynch()

cuphyStatus_t CUPHYWINAPI cuphyWriteDbgBufSynchSrs(cuphySrsRxHndl_t srsRxHndl, cudaStream_t cuStream)
{
    if(!srsRxHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    return cuphy::tryCallableAndCatch([&]
    {
        auto* p = static_cast<SrsRx*>(srsRxHndl);
        cuphyStatus_t status = p->copyOutputToCPU(cuStream);
        if(status == CUPHY_STATUS_SUCCESS) // assuming one should not continue if copyOutputToCPU returned an error
        {
            p->writeDbgBufSynch(cuStream);
            //p->printInfo();
        }
        return status;
    });
}


 //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
 // cuphyDestroyPucchRx()

 cuphyStatus_t CUPHYWINAPI cuphyDestroySrsRx(cuphySrsRxHndl_t srsRxHndl)
 {
     if(!srsRxHndl)
     {
         return CUPHY_STATUS_INVALID_ARGUMENT;
     }
     auto* p = static_cast<SrsRx*>(srsRxHndl);
     delete p;
     return CUPHY_STATUS_SUCCESS;
 }

