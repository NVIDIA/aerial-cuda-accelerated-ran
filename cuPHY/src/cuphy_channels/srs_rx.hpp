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

#if !defined(SRS_RX_HPP_INCLUDED_)
#define SRS_RX_HPP_INCLUDED_
 
#include "cuphy.h"
#include <vector>
#include "cuphy_hdf5.hpp"
#include "cuphy.hpp"

// enables batched async memcpy in combination with static parameter field
static constexpr bool SRS_USE_BATCHED_MEMCPY = true;

struct cuphySrsRx
{

};

class SrsRx final : public cuphySrsRx {
public:
    enum Component
    {
        SRS_CHEST                  = 0,
        N_SRS_COMPONENTS           = 1
    };
    struct OutputParams
    {
        // flag to copy outputs to CPU after run
        bool cpuCopyOn;

        // device (GPU) output addresses
        cuphySrsReport_t*          d_srsReports;        // array containing SRS reports of all users
        float*                     d_rbSnrBuffer;       // buffer containing RB SNRs of all users

        // host (CPU) output addresses
        __half2*                  pDataRxSrsHost;      // host-pinned buffer for raw SRS IQ from buf_st_2
        cuphySrsChEstBuffInfo_t*  h_chEstBuffInfo;
        cuphySrsReport_t*         h_srsReports;        // array containing SRS reports of all users
        float*                    h_rbSnrBuffer;       // buffer containing RB SNRs of all users
        uint32_t*                 h_rbSnrBuffOffsets;  // buffer containing user offsets into pRbSnrBuffer
        cuphySrsChEstToL2_t*      h_srsChEstToL2;      // buffer containing SRS ChEst to L2

        // debug parameters
        bool               debugOutputFlag;
        hdf5hpp::hdf5_file outHdf5File;
    };
    SrsRx(cuphySrsStatPrms_t const* pStatPrms, cudaStream_t cuStream);
    SrsRx(SrsRx const&) = delete;
    SrsRx& operator=(SrsRx const&) = delete;
    ~SrsRx();

    [[nodiscard]] cuphyStatus_t setup(const cuphySrsDynPrms_t *pDynPrm);
    [[nodiscard]] cuphyStatus_t run();
    [[nodiscard]] cuphyStatus_t copyOutputToCPU(cudaStream_t cuStream);
    [[nodiscard]] const void* getMemoryTracker() const;

    // debug functions:
    void writeDbgBufSynch(cudaStream_t cuStream);
    static void printStaticApiPrms(cuphySrsStatPrms_t const* pStaticPrms);
    static void printDynApiPrms(const cuphySrsDynPrms_t* pDynPrm);


private:
    // creation functions
    size_t getBufferSize(cuphySrsStatPrms_t const* pStatPrms);
    void   allocateDescr();
    void   createComponents(cuphySrsFilterPrms_t* pSrsFilterPrms, cudaStream_t cuStream);

    // setup functions
    void setupCmn(const cuphySrsDynPrms_t *pDynPrm);
    void allocateDeviceMemory();
    cuphyStatus_t setupComponents(bool enableCpuToGpuDescrAsyncCpy, const cuphySrsDynPrms_t *pDynPrm);

    // graph functions
    void createGraph();
    void updateGraph() const;

    // destroy functions
    void destroyComponents() const;

    // count GPU memory usage
    cuphyMemoryFootprint m_memoryFootprint;

    // number of PRBs must be defined before m_LinearAlloc which is using it as part of getBufferSize function
    uint16_t m_nPrbs{};
    // input/intermediate/output buffers
    cuphy::linear_alloc<128, cuphy::device_alloc> m_LinearAlloc;
    cuphyTensorPrm_t*                             m_hPrmDataRx{};
    OutputParams                                  m_outputPrms{};
    std::vector<void*>                            m_gpuAddrsChEstToL2InnerVec;
    std::vector<void*>                            m_gpuAddrsChEstToL2Vec;

    // stream worker:
    cudaStream_t  m_cuStream{};

    // pipeline parameters:
    std::vector<cuphySrsCellPrms_t> m_srsCellPrmsVec;
    uint16_t                        m_nSrsUes{};
    uint16_t                        m_nCells{};
    cuphyCellStatPrm_t*             m_hCellStatPrms{};
    cuphyUeSrsPrm_t*                m_hUeSrsPrm{};

    // chEst parameters:
    cuphySrsChEstAlgoType_t m_chEstAlgo{};
    uint8_t m_chEstToL2NormalizationAlgo{};
    float m_chEstToL2ConstantScaler{};
    uint8_t m_enableDelayOffsetCorrection{};
    cuphySrsRkhsPrms_t*     m_pRkhsPrms{};
    void*                   m_gpuAddrRkhsWorskpace{};
  
    // kernel descriptors
    cuphy::kernelDescrs<N_SRS_COMPONENTS>   m_kernelStatDescr;
    cuphy::kernelDescrs<N_SRS_COMPONENTS>   m_kernelDynDescr;

    // Component handles 
    cuphySrsChEstHndl_t  m_srsChEstHndl{};

    // kernel launch configurations
    cuphySrsChEstLaunchCfg_t  m_srsChEstLaunchCfg{};
    cuphySrsChEstNormalizationLaunchCfg_t m_srsChEstNormalizationLaunchCfg{};

    // graph parameters
    bool        m_cudaGraphModeEnabled{};
    CUgraph     m_graph{};
    CUgraphExec m_graphExec{};
    CUgraphNode m_srsKernelNode{};
    CUgraphNode m_srsNormalizationKernelNode{};

    CUDA_KERNEL_NODE_PARAMS m_emptyNode0ParamsDriver{};

    // Batched memcpy helper object
    cuphyBatchedMemcpyHelper m_batchedMemcpyHelper;
};

#endif // !defined(SRS_RX_HPP_INCLUDED_)
