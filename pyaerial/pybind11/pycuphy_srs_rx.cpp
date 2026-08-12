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

#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "cuphy.h"
#include "cuphy_api.h"
#include "pycuphy_util.hpp"
#include "pycuphy_srs_util.hpp"
#include "pycuphy_srs_rx.hpp"

namespace py = pybind11;


namespace pycuphy {

SrsRx::SrsRx(const cudaStream_t cuStream, cuphySrsStatPrms_t* statPrms):
m_cuStream(cuStream) {
    if(cuphyStatus_t status = cuphyCreateSrsRx(&m_srsRxHndl, statPrms, m_cuStream); status != CUPHY_STATUS_SUCCESS) {
        NVLOGE_FMT(NVLOG_PYAERIAL, AERIAL_PYAERIAL_EVENT, "cuphyCreateSrsRx error {}", status);
        throw std::runtime_error("SrsRx::SrsRx: cuphyCreateSrsRx error!");
    }
}

void SrsRx::run(cuphySrsDynPrms_t* dynPrms) {
    if(cuphyStatus_t status = cuphySetupSrsRx(m_srsRxHndl, dynPrms, m_batchPrmHndl); status != CUPHY_STATUS_SUCCESS) {
        NVLOGE_FMT(NVLOG_PYAERIAL, AERIAL_PYAERIAL_EVENT, "cuphySetupSrsRx error {}", status);
        throw std::runtime_error("SrsRx::run: cuphySetupSrsRx error!");
    }

    constexpr uint64_t procModeBmsk = SRS_PROC_MODE_FULL_SLOT;
    if(cuphyStatus_t status = cuphyRunSrsRx(m_srsRxHndl, procModeBmsk); status != CUPHY_STATUS_SUCCESS) {
        NVLOGE_FMT(NVLOG_PYAERIAL, AERIAL_PYAERIAL_EVENT, "cuphyRunSrsRx error {}", status);
        throw std::runtime_error("SrsRx::run: cuphyRunSrsRx error!");
    }
}


PySrsRx::PySrsRx(const uint16_t nCells,
                 const std::vector<uint16_t>& nRxAnt,
                 const uint8_t chEstAlgoIdx,
                 const uint8_t enableDelayOffsetCorrection,
                 const py::dict& chEstParams,
                 const uint16_t nMaxSrsUes,
                 const uint64_t cuStream):
m_cuStream(reinterpret_cast<cudaStream_t>(cuStream)),
m_nMaxSrsUes(nMaxSrsUes),
m_tensorPrms(chEstParams, reinterpret_cast<cudaStream_t>(cuStream)) {

    // Fill needed parameters in the static parameters struct.

    m_cellStatPrms.resize(nCells);
    if(nRxAnt.size() != nCells) {
        NVLOGE_FMT(NVLOG_PYAERIAL,
                   AERIAL_PYAERIAL_EVENT,
                   "Number of receive antennas given for wrong number of cells!");
        throw std::runtime_error("PySrsRx::PySrsRx: Configuration error!");
    }
    for(int cellIdx = 0; auto& cellStatPrm : m_cellStatPrms) {
        cellStatPrm.phyCellId = 0;  // Not used.
        cellStatPrm.nRxAnt = 0;  // Not used.
        cellStatPrm.nRxAntSrs = nRxAnt[cellIdx++];
        cellStatPrm.nTxAnt = 0;  // Not used.
        cellStatPrm.nPrbUlBwp = 0;  // Not used.
        cellStatPrm.nPrbDlBwp = 0;  // Not used.
        cellStatPrm.mu = 1;
        cellStatPrm.pPuschCellStatPrms = nullptr;
        cellStatPrm.pPucchCellStatPrms = nullptr;
    }

    m_statPrms.srsFilterPrms = m_tensorPrms.getSrsFilterPrms();
    m_statPrms.nMaxCells = nCells;
    m_statPrms.pCellStatPrms = m_cellStatPrms.data();
    m_statPrms.nMaxCellsPerSlot = nCells;
    m_statPrms.pDbg = nullptr;
    m_srsStatDbgPrms.pOutFileName = nullptr;
    m_srsStatDbgPrms.enableApiLogging = 0;
    m_statPrms.pStatDbg = &m_srsStatDbgPrms;
    m_statPrms.chEstAlgo = static_cast<cuphySrsChEstAlgoType_t>(chEstAlgoIdx);
    m_statPrms.enableDelayOffsetCorrection = enableDelayOffsetCorrection;
    m_statPrms.pSrsRkhsPrms = const_cast<cuphySrsRkhsPrms_t*>(&m_tensorPrms.getSrsRkhsPrms());
    m_tracker.pMemoryFootprint = nullptr;
    m_statPrms.pOutInfo = &m_tracker;
    m_statPrms.enableBatchedMemcpy = 1;

    m_srsRx = std::make_unique<SrsRx>(m_cuStream, &m_statPrms);

    // Pre-allocate device memory for Rx data in 16-bit float format (needs conversion).
    m_dRxData = cuphy::make_unique_device<__half2>(m_nRxDataElemsPerCell * nCells);

    // Pre-allocate device/host  memory for channel estimates.
    m_dChEsts = cuphy::make_unique_device<std::complex<float>>(m_nChEstElemsPerUe * m_nMaxSrsUes);
    m_hChEsts = cuphy::make_unique_pinned<std::complex<float>>(m_nChEstElemsPerUe * m_nMaxSrsUes);
}

const std::vector<py::array_t<std::complex<float>>>& PySrsRx::run(const std::vector<cuda_array_t<std::complex<float>>>& rxData,
                                                                  const std::vector<py::object>& pySrsRxUeConfigs,
                                                                  const std::vector<py::object>& pySrsRxCellConfigs) {

    const uint16_t nCells = rxData.size();
    m_nSrsUes = pySrsRxUeConfigs.size();
    if(pySrsRxCellConfigs.size() != nCells) {
        NVLOGE_FMT(NVLOG_PYAERIAL,
                   AERIAL_PYAERIAL_EVENT,
                   "Number of cell configurations does not match with the number of Rx data tensors!");
        throw std::runtime_error("PySrsRx::run: Configuration error!");
    }

    m_tSrsChEsts.clear();
    m_chEsts.clear();
    m_chEstsToL2.clear();
    m_rbSnrs.resize(m_nSrsUes * MAX_N_PRBS_SUPPORTED);
    m_rbSnrBuffOffsets.resize(m_nSrsUes);
    m_srsChEstBuffInfo.resize(m_nSrsUes);
    m_srsReports.resize(m_nSrsUes);
    m_srsChEstToL2.resize(m_nSrsUes);
    m_chEstCpuBuff.resize(m_nSrsUes);

    // Create the SRS dynamic parameters struct.
    std::vector<int> numPrgs;
    readSrsCellDynParams(m_srsCellDynPrms, pySrsRxCellConfigs);
    readUeSrsRxParams(m_ueSrsPrms, numPrgs, pySrsRxUeConfigs);

    m_srsCellGrpDynPrms.nCells = nCells;
    m_srsCellGrpDynPrms.pCellPrms = m_srsCellDynPrms.data();
    m_srsCellGrpDynPrms.nSrsUes = m_nSrsUes;
    m_srsCellGrpDynPrms.pUeSrsPrms = m_ueSrsPrms.data();

    m_tDataRx.resize(nCells);
    m_dRxDataTensor.resize(nCells);
    for(int cellIdx = 0; const auto& rxDataCell : rxData) {
        m_dRxDataTensor[cellIdx] = deviceFromCudaArray<std::complex<float>>(
            rxDataCell,
            m_dRxData.get() + m_nRxDataElemsPerCell * cellIdx,
            CUPHY_C_32F,
            CUPHY_C_16F,
            cuphy::tensor_flags::align_tight,
            m_cuStream);
        m_tDataRx[cellIdx].desc = m_dRxDataTensor[cellIdx].desc().handle();
        m_tDataRx[cellIdx].pAddr = m_dRxDataTensor[cellIdx].addr();
        cellIdx++;
    }
    m_dataIn.pTDataRx = m_tDataRx.data();

    m_tSrsChEsts.reserve(m_nSrsUes);
    for(int ueIdx = 0; ueIdx < m_nSrsUes; ueIdx++) {
        const uint16_t cellIdx = m_ueSrsPrms[ueIdx].cellIdx;
        const uint16_t nRxAnt = m_cellStatPrms[cellIdx].nRxAntSrs;
        const uint16_t nAntPorts = m_ueSrsPrms[ueIdx].nAntPorts;

        m_tSrsChEsts.emplace_back(m_dChEsts.get() + ueIdx * m_nChEstElemsPerUe,
                                  CUPHY_C_32F,
                                  numPrgs[ueIdx],
                                  nRxAnt,
                                  nAntPorts,
                                  cuphy::tensor_flags::align_tight);
        m_srsChEstBuffInfo[ueIdx].tChEstBuffer.desc = m_tSrsChEsts[ueIdx].desc().handle();
        m_srsChEstBuffInfo[ueIdx].tChEstBuffer.pAddr = m_tSrsChEsts[ueIdx].addr();
        m_srsChEstBuffInfo[ueIdx].startPrbGrp = m_ueSrsPrms[ueIdx].srsStartPrg;

        m_rbSnrBuffOffsets[ueIdx] = ueIdx * MAX_N_PRBS_SUPPORTED;
        const size_t chEstSize = numPrgs[ueIdx] * nRxAnt * nAntPorts * sizeof(std::complex<float>);
        m_chEstCpuBuff[ueIdx] = std::move(cuphy::buffer<uint8_t, cuphy::pinned_alloc>(chEstSize));
        m_srsChEstToL2[ueIdx].pChEstCpuBuff = m_chEstCpuBuff[ueIdx].addr();
    }
    CUDA_CHECK(cudaMemsetAsync(m_dChEsts.get(), 0, m_nChEstElemsPerUe * m_nSrsUes * sizeof(std::complex<float>), m_cuStream));

    m_dataOut.pChEstBuffInfo = m_srsChEstBuffInfo.data();
    m_dataOut.pSrsReports = m_srsReports.data();
    m_dataOut.pSrsChEstToL2 = m_srsChEstToL2.data();
    m_dataOut.pRbSnrBuffer = m_rbSnrs.data();
    m_dataOut.pRbSnrBuffOffsets = m_rbSnrBuffOffsets.data();
    m_dataOut.pDataRxSrs = nullptr;

    m_srsDynDbgPrms.enableApiLogging = 0;

    m_srsStatusOut = {cuphySrsStatusType_t::CUPHY_SRS_STATUS_SUCCESS_OR_UNTRACKED_ISSUE, MAX_UINT16, MAX_UINT16};

    m_dynPrms.cuStream = m_cuStream;
    m_dynPrms.procModeBmsk = SRS_PROC_MODE_FULL_SLOT;
    m_dynPrms.pCellGrpDynPrm = &m_srsCellGrpDynPrms;
    m_dynPrms.pDataIn = &m_dataIn;
    m_dynPrms.pDataOut = &m_dataOut;
    m_dynPrms.cpuCopyOn = true;  // GPU side outputs not exposed by SrsRx API??
    m_dynPrms.pStatusOut = &m_srsStatusOut;
    m_dynPrms.pDynDbg = &m_srsDynDbgPrms;

    // Run the SRS Rx.
    m_srsRx->run(&m_dynPrms);

    // Move to host and Python.
    CUDA_CHECK(cudaMemcpyAsync(m_hChEsts.get(),
                               m_dChEsts.get(),
                               m_nChEstElemsPerUe * m_nSrsUes * sizeof(std::complex<float>),
                               cudaMemcpyDeviceToHost,
                               m_cuStream));
    CUDA_CHECK(cudaStreamSynchronize(m_cuStream));

    m_chEsts.reserve(m_nSrsUes);
    for(int ueIdx=0; ueIdx < m_nSrsUes; ueIdx++) {
        const size_t cellIdx = m_ueSrsPrms[ueIdx].cellIdx;
        const size_t nRxAnt = m_cellStatPrms[cellIdx].nRxAntSrs;
        const size_t nAntPorts = m_ueSrsPrms[ueIdx].nAntPorts;

        const std::vector shape = {static_cast<size_t>(numPrgs[ueIdx]), nRxAnt, nAntPorts};
        const std::vector<size_t> strides{};

        m_chEsts.push_back(hostToNumpy<std::complex<float>>(m_hChEsts.get() + ueIdx * m_nChEstElemsPerUe, shape, strides));
    }

    return m_chEsts;
}


const std::vector<py::array_t<std::complex<float>>>& PySrsRx::getChEstToL2() {
    m_chEstsToL2.reserve(m_nSrsUes);
    for(int ueIdx=0; ueIdx < m_nSrsUes; ueIdx++) {
        const size_t cellIdx = m_ueSrsPrms[ueIdx].cellIdx;
        const size_t nRxAnt = m_cellStatPrms[cellIdx].nRxAntSrs;
        const size_t nAntPorts = m_ueSrsPrms[ueIdx].nAntPorts;

        const std::vector shape = {static_cast<size_t>(m_srsChEstToL2[ueIdx].nPrbGrps), nRxAnt, nAntPorts};
        const std::vector<size_t> strides{};

        m_chEstsToL2.push_back(hostToNumpy<std::complex<float>>(reinterpret_cast<std::complex<float>*>(m_srsChEstToL2[ueIdx].pChEstCpuBuff),
                                                                shape,
                                                                strides));
    }
    return m_chEstsToL2;
}


const pybind11::array_t<float>& PySrsRx::getRbSnrBuffer()  {
    m_rbSnrArray = hostToNumpy<float>((float*)m_rbSnrs.data(), MAX_N_PRBS_SUPPORTED, m_nSrsUes);
    return m_rbSnrArray;
}


}
