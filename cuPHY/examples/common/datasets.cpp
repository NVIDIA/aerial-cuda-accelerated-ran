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

#include "datasets.hpp"
#include <string>
#include "cuphy_internal.h"
#include "utils.cuh"
#include <unordered_map>
#include "fmt/core.h"
#include <arpa/inet.h>
#define PUCCH_F2_LLR_THRESH 0.10
#define PUCCH_F3_LLR_THRESH 0.10

#define CUPHY_PUCCH_F2_MAX_E (512)
#define CUPHY_PUCCH_F3_MAX_E (4608)

//using namespace std;
using namespace cuphy;
//---------------------------------------------------------------------------------------------------
// Dataset holds dynamic api parameters/data

// Construct dataset from h5 file
DynApiDataset::DynApiDataset(const std::vector<std::string>& inputFileNameVec, cudaStream_t cuStrm, uint64_t procMode, bool cpuCopyOn, uint32_t fp16Mode, int apiTVflag, int drmDebug)
{
    cellDynPrmVec.resize(inputFileNameVec.size());
    dbgPrm.enableApiLogging = 0;

    CUDA_CHECK(cudaHostAlloc((void **)&(pPreEarlyHarqWaitKernelStatus), sizeof(uint8_t), cudaHostAllocPortable | cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc((void **)&(pPostEarlyHarqWaitKernelStatus), sizeof(uint8_t), cudaHostAllocPortable | cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer((void **)&(DataOut.pPreEarlyHarqWaitKernelStatusGpu), (void *)(pPreEarlyHarqWaitKernelStatus), 0));
    CUDA_CHECK(cudaHostGetDevicePointer((void **)&(DataOut.pPostEarlyHarqWaitKernelStatusGpu), (void *)(pPostEarlyHarqWaitKernelStatus), 0));

    if(apiTVflag == 0) // legacy test-vector
    {
        /*----------------------------- cellDynPrm parameters across all input file --------------------------*/
        cellGrpDynPrm.nUeGrps = cellGrpDynPrm.nUes = 0;
        for(uint32_t i = 0; i < inputFileNameVec.size(); i++)
        {
            // load 5h file
            hdf5hpp::hdf5_file      fInput    = hdf5hpp::hdf5_file::open(inputFileNameVec[i].c_str());
            cuphy::cuphyHDF5_struct gnbConfig = cuphy::get_HDF5_struct(fInput, "gnb_pars");
            cellGrpDynPrm.nUeGrps += gnbConfig.get_value_as<uint16_t>("nUserGroups");
            cellGrpDynPrm.nUes += gnbConfig.get_value_as<uint16_t>("numTb");
            // cell parameters
            cellDynPrmVec[i].cellPrmStatIdx = i;
            cellDynPrmVec[i].cellPrmDynIdx  = i;
            cellDynPrmVec[i].slotNum        = gnbConfig.get_value_as<uint16_t>("slotNumber");
        }

        if(cellGrpDynPrm.nUes > MAX_N_TBS_PER_CELL_GROUP_SUPPORTED)
        {
            throw std::runtime_error("nTBs in TV > MAX_N_TBS_PER_CELL_GROUP_SUPPORTED");
        }

        ueGrpPrmsVec.resize(cellGrpDynPrm.nUeGrps);
        dmrsPrmVec.resize(cellGrpDynPrm.nUeGrps);
        uePrmsVec.resize(cellGrpDynPrm.nUes);
        uciPrmsVec.resize(cellGrpDynPrm.nUes);
        calcCsi2PrmsVec.resize(cellGrpDynPrm.nUes * CUPHY_MAX_N_CSI2_REPORTS_PER_UE);
        ueGrpToUeIdxs.resize(cellGrpDynPrm.nUeGrps);
        cuphyCalcCsi2SizePrm_t* pCalcCsi2SizePrms = calcCsi2PrmsVec.data();

        tDataRxVec.resize(inputFileNameVec.size());
        tPrmDataRxVec.resize(inputFileNameVec.size());
        uint32_t              globalUeGrpIdx = 0;
        uint32_t              globalUeIdx    = 0;
        std::vector<uint32_t> numUesInUeGrp;
        numUesInUeGrp.resize(cellGrpDynPrm.nUeGrps);

        for(uint32_t i = 0; i < inputFileNameVec.size(); i++)
        {
            // Load 5h file
            hdf5hpp::hdf5_file      fInput    = hdf5hpp::hdf5_file::open(inputFileNameVec[i].c_str());
            cuphy::cuphyHDF5_struct gnbConfig = cuphy::get_HDF5_struct(fInput, "gnb_pars");
            hdf5hpp::hdf5_dataset   tbpDset   = fInput.open_dataset("tb_pars");
            hdf5hpp::hdf5_dataset   ueGrpDset = fInput.open_dataset("ueGrp_pars");
            cuphy::cuphyHDF5_struct tbConfig  = cuphy::get_HDF5_struct_index(tbpDset, 0);

            /*----------------------------- ueGrpPrmsVec parameters in each input file ----------------------------------------*/
            uint16_t              nUeGrps = gnbConfig.get_value_as<uint16_t>("nUserGroups");
            std::vector<uint32_t> localToGlobalUeGrpIdx;
            localToGlobalUeGrpIdx.resize(nUeGrps);

            for(int ueGrpIdx = 0; ueGrpIdx < nUeGrps; ++ueGrpIdx, globalUeGrpIdx++)
            {
                cuphy::cuphyHDF5_struct ueGrpCfg = cuphy::get_HDF5_struct_index(ueGrpDset, ueGrpIdx);

                ueGrpPrmsVec[globalUeGrpIdx].pCellPrm       = &cellDynPrmVec[i];
                ueGrpPrmsVec[globalUeGrpIdx].startPrb       = ueGrpCfg.get_value_as<uint16_t>("startPrb");
                ueGrpPrmsVec[globalUeGrpIdx].nPrb           = ueGrpCfg.get_value_as<uint16_t>("nPrb");
                ueGrpPrmsVec[globalUeGrpIdx].nUes           = ueGrpCfg.get_value_as<uint16_t>("nUes");
                ueGrpPrmsVec[globalUeGrpIdx].nUplinkStreams = ueGrpCfg.get_value_as<uint16_t>("nUplinkStreams");
                ueGrpPrmsVec[globalUeGrpIdx].puschStartSym  = ueGrpCfg.get_value_as<uint8_t>("StartSymbolIndex");
                ueGrpPrmsVec[globalUeGrpIdx].nPuschSym      = ueGrpCfg.get_value_as<uint8_t>("NrOfSymbols");
                ueGrpPrmsVec[globalUeGrpIdx].dmrsSymLocBmsk = ueGrpCfg.get_value_as<uint16_t>("dmrsSymLocBmsk");
                ueGrpPrmsVec[globalUeGrpIdx].rssiSymLocBmsk = ueGrpCfg.get_value_as<uint16_t>("rssiSymLocBmsk");
                ueGrpPrmsVec[globalUeGrpIdx].prgSize        = ueGrpCfg.get_value_as<uint16_t>("prgSize");
                ueGrpPrmsVec[globalUeGrpIdx].enablePerPrgChEstPerUeg = ueGrpCfg.get_value_as<uint8_t>("enablePerPrgChEst");
                ueGrpToUeIdxs[globalUeGrpIdx].resize(ueGrpPrmsVec[globalUeGrpIdx].nUes);
                ueGrpPrmsVec[globalUeGrpIdx].pUePrmIdxs = ueGrpToUeIdxs[globalUeGrpIdx].data();
                localToGlobalUeGrpIdx[ueGrpIdx]             = globalUeGrpIdx;
            }

            /*----------------------------- uePrmsVec parameters in each input file ----------------------------------------*/
            //uint16_t nUciUes = 0;
            uint16_t nSchUes = 0;
            uint16_t nUes    = gnbConfig.get_value_as<uint16_t>("numTb");
            //uciPrmsVec.resize(nUes);
            for(uint16_t ueIdx = 0; ueIdx < nUes; ++ueIdx, globalUeIdx++)
            {
                tbConfig = cuphy::get_HDF5_struct_index(tbpDset, ueIdx);

                uint16_t localUeGrpIdx      = tbConfig.get_value_as<uint16_t>("userGroupIndex");
                uint16_t tempGlobalUeGrpIdx = localToGlobalUeGrpIdx[localUeGrpIdx];

                // dynamic DMRS parameters
                dmrsPrmVec[tempGlobalUeGrpIdx].dmrsAddlnPos       = tbConfig.get_value_as<uint8_t>("dmrsAddlPosition");
                dmrsPrmVec[tempGlobalUeGrpIdx].dmrsMaxLen         = tbConfig.get_value_as<uint8_t>("dmrsMaxLength");
                dmrsPrmVec[tempGlobalUeGrpIdx].nDmrsCdmGrpsNoData = tbConfig.get_value_as<uint8_t>("numDmrsCdmGrpsNoData");
                dmrsPrmVec[tempGlobalUeGrpIdx].dmrsScrmId         = tbConfig.get_value_as<uint8_t>("dmrsScramId");
                ueGrpPrmsVec[tempGlobalUeGrpIdx].pDmrsDynPrm      = &dmrsPrmVec[tempGlobalUeGrpIdx];

                uePrmsVec[globalUeIdx].pUeGrpPrm                  = &ueGrpPrmsVec[tempGlobalUeGrpIdx];
                uePrmsVec[globalUeIdx].ueGrpIdx                   = tempGlobalUeGrpIdx;

                uePrmsVec[globalUeIdx].enableTfPrcd               = tbConfig.get_value_as<uint8_t>("enableTfPrcd");
                if(uePrmsVec[globalUeIdx].enableTfPrcd==1)
                {
                    uePrmsVec[globalUeIdx].puschIdentity              = tbConfig.get_value_as<uint32_t>("puschIdentity");
                    uePrmsVec[globalUeIdx].groupOrSequenceHopping     = tbConfig.get_value_as<uint8_t>("groupOrSequenceHopping");
                    uePrmsVec[globalUeIdx].N_symb_slot                = tbConfig.get_value_as<uint8_t>("N_symb_slot");
                    uePrmsVec[globalUeIdx].N_slot_frame               = tbConfig.get_value_as<uint8_t>("N_slot_frame");
                    uePrmsVec[globalUeIdx].lowPaprGroupNumber         = tbConfig.get_value_as<uint8_t>("lowPaprGroupNumber");
                    uePrmsVec[globalUeIdx].lowPaprSequenceNumber      = tbConfig.get_value_as<uint8_t>("lowPaprSequenceNumber");
                }
                uePrmsVec[globalUeIdx].scid                       = tbConfig.get_value_as<uint8_t>("nSCID");
                uePrmsVec[globalUeIdx].dmrsPortBmsk               = tbConfig.get_value_as<uint16_t>("dmrsPortBmsk");
                uePrmsVec[globalUeIdx].mcsTableIndex              = tbConfig.get_value_as<uint16_t>("mcsTableIndex");
                uePrmsVec[globalUeIdx].mcsIndex                   = tbConfig.get_value_as<uint16_t>("mcsIndex");
                uePrmsVec[globalUeIdx].rv                         = tbConfig.get_value_as<uint8_t>("rv");
                uePrmsVec[globalUeIdx].targetCodeRate             = tbConfig.get_value_as<uint16_t>("targetCodeRate");
                uePrmsVec[globalUeIdx].qamModOrder                = tbConfig.get_value_as<uint8_t>("qamModOrder");
                uePrmsVec[globalUeIdx].TBSize                     = tbConfig.get_value_as<uint32_t>("nTbByte");
                uePrmsVec[globalUeIdx].foForgetCoeff              = tbConfig.get_value_as<float>("foForgetCoeff");
                uePrmsVec[globalUeIdx].ldpcEarlyTerminationPerUe  = tbConfig.get_value_as<uint8_t>("ldpcEarlyTerminationPerUe");
                uePrmsVec[globalUeIdx].ldpcMaxNumItrPerUe         = tbConfig.get_value_as<uint8_t>("ldpcMaxNumItrPerUe");
                try
                {
                    uePrmsVec[globalUeIdx].ndi = tbConfig.get_value_as<uint8_t>("ndi");
                }
                catch(const cuphy::cuphyHDF5_exception& e)
                {
                    NVLOGW_FMT(NVLOG_PUSCH, "TV is missing ndi field for ueIdx {}.  Setting to 1.", ueIdx);
                    uePrmsVec[globalUeIdx].ndi = 1;
                }
                NVLOGD_FMT(NVLOG_PUSCH, "mcsTableIndex {} mcsIndex {} rv {} ndi {}", uePrmsVec[globalUeIdx].mcsTableIndex, uePrmsVec[globalUeIdx].mcsIndex, uePrmsVec[globalUeIdx].rv, uePrmsVec[globalUeIdx].ndi);

                uePrmsVec[globalUeIdx].rnti        = tbConfig.get_value_as<uint16_t>("nRnti");
                uePrmsVec[globalUeIdx].dataScramId = tbConfig.get_value_as<uint16_t>("dataScramId");
                uePrmsVec[globalUeIdx].nUeLayers   = tbConfig.get_value_as<uint8_t>("numLayers");

                try
                {
                    uePrmsVec[globalUeIdx].i_lbrm     = tbConfig.get_value_as<uint8_t>("I_LBRM");
                    uePrmsVec[globalUeIdx].n_PRB_LBRM = tbConfig.get_value_as<uint16_t>("n_PRB_LBRM");
                    uePrmsVec[globalUeIdx].maxLayers  = tbConfig.get_value_as<uint8_t>("maxLayers");
                    uePrmsVec[globalUeIdx].maxQm      = tbConfig.get_value_as<uint8_t>("maxQm");
                }
                catch(const cuphy::cuphyHDF5_exception& e)
                {
                    NVLOGW_FMT(NVLOG_PUSCH, "TV is missing i_lbrm or related LBRM fields.  Disabling LBRM");
                    uePrmsVec[globalUeIdx].i_lbrm = 0;
                }

                try
                {
                    uePrmsVec[globalUeIdx].pduBitmap = tbConfig.get_value_as<uint16_t>("pduBitmap");
                }
                catch(const cuphy::cuphyHDF5_exception& e)
                {
                    NVLOGW_FMT(NVLOG_PUSCH, "TV is missing pduBitmap field for ueIdx {}.  Setting to SCH data only transmission (pduBitmap = 8).", ueIdx);
                    uePrmsVec[globalUeIdx].pduBitmap = 8;
                }

                if(uePrmsVec[globalUeIdx].pduBitmap & 2) // check 1st bit for uci transmission
                {
                    uciPrmsVec[globalUeIdx].nBitsHarq         = tbConfig.get_value_as<uint16_t>("nBitsHarq");
                    uciPrmsVec[globalUeIdx].nBitsCsi1         = tbConfig.get_value_as<uint16_t>("nBitsCsi1");
                    uciPrmsVec[globalUeIdx].alphaScaling      = tbConfig.get_value_as<uint8_t>("alphaScaling");
                    uciPrmsVec[globalUeIdx].betaOffsetHarqAck = tbConfig.get_value_as<uint8_t>("betaOffsetHarqAck");
                    uciPrmsVec[globalUeIdx].betaOffsetCsi1    = tbConfig.get_value_as<uint8_t>("betaOffsetCsi1");
                    uciPrmsVec[globalUeIdx].betaOffsetCsi2    = tbConfig.get_value_as<uint8_t>("betaOffsetCsi2");
                    uciPrmsVec[globalUeIdx].rankBitOffset     = tbConfig.get_value_as<uint8_t>("rankBitOffset");
                    uciPrmsVec[globalUeIdx].nRanksBits        = tbConfig.get_value_as<uint8_t>("rankBitSize");
                    uciPrmsVec[globalUeIdx].DTXthreshold      = tbConfig.get_value_as<uint8_t>("DTXthreshold");
                    uciPrmsVec[globalUeIdx].nCsiReports       = 1;


                    // FAPIv3 CSI-P2 size calculation paramaters:
                     uciPrmsVec[globalUeIdx].nCsi2Reports = tbConfig.get_value_as<uint16_t>("nCsi2Reports");
                    if(uciPrmsVec[globalUeIdx].nCsi2Reports > 0)
                    {
                        uciPrmsVec[globalUeIdx].pCalcCsi2SizePrms = pCalcCsi2SizePrms;

                        std::vector<uint8_t>  nPart1PrmsVec     = tbpDset[ueIdx]["nPart1Prms"].as<std::vector<uint8_t>>();
                        std::vector<uint8_t>  prmSizesVec       = tbpDset[ueIdx]["prmSizes"].as<std::vector<uint8_t>>();
                        std::vector<uint16_t> prmOffsetsVec     = tbpDset[ueIdx]["prmOffsets"].as<std::vector<uint16_t>>();
                        std::vector<uint16_t> csi2sizeMapIdxVec = tbpDset[ueIdx]["csi2sizeMapIdx"].as<std::vector<uint16_t>>();

                        for(int csi2ReportIdx = 0; csi2ReportIdx < uciPrmsVec[globalUeIdx].nCsi2Reports; ++csi2ReportIdx)
                        {
                            pCalcCsi2SizePrms->nPart1Prms     = nPart1PrmsVec[csi2ReportIdx];
                            pCalcCsi2SizePrms->csi2sizeMapIdx = csi2sizeMapIdxVec[csi2ReportIdx];

                            for(int csi1PrmIdx = 0; csi1PrmIdx < CUPHY_MAX_N_CSI1_PRMS; ++csi1PrmIdx)
                            {
                                pCalcCsi2SizePrms->prmSizes[csi1PrmIdx]   = prmSizesVec[csi2ReportIdx * CUPHY_MAX_N_CSI1_PRMS + csi1PrmIdx];
                                pCalcCsi2SizePrms->prmOffsets[csi1PrmIdx] = prmOffsetsVec[csi2ReportIdx * CUPHY_MAX_N_CSI1_PRMS + csi1PrmIdx];
                            }
                            pCalcCsi2SizePrms += 1;
                        }
                    }else
                    {
                        uciPrmsVec[globalUeIdx].pCalcCsi2SizePrms = NULL;
                    }
                    uePrmsVec[globalUeIdx].pUciPrms = &uciPrmsVec[globalUeIdx];
                }
                else
                {
                    uePrmsVec[globalUeIdx].pUciPrms = NULL;
                }
                ueGrpPrmsVec[tempGlobalUeGrpIdx].pUePrmIdxs[numUesInUeGrp[tempGlobalUeGrpIdx]] = globalUeIdx;
                numUesInUeGrp[tempGlobalUeGrpIdx]++;

                if(uePrmsVec[globalUeIdx].pduBitmap & 1) // check 0th bit for data transmission
                {
                    if(drmDebug)
                    {
                        std::string dataset = "reference_derateCbsIndicesSizes";
                        if(fInput.is_valid_dataset(dataset.c_str()))
                        {
                            cuphy::tensor_pinned tSizes         = cuphy::tensor_from_dataset(fInput.open_dataset(dataset.c_str()), CUPHY_R_32U, cuphy::tensor_flags::align_tight, 0);
                            uint32_t*            pSizes         = static_cast<uint32_t*>(tSizes.addr());
                            uint32_t             nCbs           = tbConfig.get_value_as<uint32_t>("nCb");
                            uint32_t             indices_per_ue = 0;

                            for(uint16_t cb = 0; cb < nCbs; cb++)
                                indices_per_ue += *pSizes++;

                            NVLOGC_FMT(NVLOG_PUSCH, "Allocating {} de-rate-match indices for UE {}", indices_per_ue, globalUeIdx);
                            CUDA_CHECK(cudaHostAlloc(&uePrmsVec[globalUeIdx].debug_d_derateCbsIndices, sizeof(uint32_t*) * indices_per_ue, cudaHostAllocPortable | cudaHostAllocMapped));
                        }
                        else
                        {
                            NVLOGF_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "Old TV is missing {} field.  Can't check de-rate-match index results.",dataset);
                        }
                    }
                    else
                    {
                        uePrmsVec[globalUeIdx].debug_d_derateCbsIndices = nullptr;
                    }
                    nSchUes += 1;
                }
            }

            /*----------------------------- DataRx in each input file ----------------------------------------*/
            cuphyDataType_t cplxTypeDataRx = fp16Mode ? CUPHY_C_16F : CUPHY_C_32F;
            tDataRxVec[i]                    = cuphy::tensor_from_dataset(fInput.open_dataset("DataRx"), cplxTypeDataRx, cuphy::tensor_flags::align_tight, cuStrm);
            cudaStreamSynchronize(cuStrm);

            tPrmDataRxVec[i].desc  = tDataRxVec[i].desc().handle();
            tPrmDataRxVec[i].pAddr = tDataRxVec[i].addr();
        } // for inputfilename

        //Tying everything together now - set the cellGrpDynPrm and DataIn pointers
        cellGrpDynPrm.nCells     = inputFileNameVec.size();
        cellGrpDynPrm.pCellPrms  = &cellDynPrmVec[0];
        cellGrpDynPrm.pUeGrpPrms = &ueGrpPrmsVec[0];
        cellGrpDynPrm.pUePrms    = &uePrmsVec[0];
        DataIn.pTDataRx          = &tPrmDataRxVec[0];

        /*----------------------------- DataOut parameters ----------------------------------------*/
        globalUeIdx      = 0;
        uint32_t nUciUes = 0;
        uint32_t nCbs = 0, nTbs = 0, nBytes = 0, nBytesPerCell = 0;
        nUciPayloadBytes = 0;
        nUciSegs = 0;
        uint32_t nUeGrps = 0;
        bStartOffsetsTbPayloadDatasetsVec.resize(cellGrpDynPrm.nUes);
        // output parameters
        bStartOffsetsCbCrc     = std::move(cuphy::buffer<uint32_t, cuphy::pinned_alloc>(cellGrpDynPrm.nUes));
        bStartOffsetsTbCrc     = std::move(cuphy::buffer<uint32_t, cuphy::pinned_alloc>(cellGrpDynPrm.nUes));
        bStartOffsetsTbPayload = std::move(cuphy::buffer<uint32_t, cuphy::pinned_alloc>(cellGrpDynPrm.nUes));
        for(uint32_t i = 0; i < inputFileNameVec.size(); i++)
        {
            // load 5h file
            hdf5hpp::hdf5_file      fInput    = hdf5hpp::hdf5_file::open(inputFileNameVec[i].c_str());
            cuphy::cuphyHDF5_struct gnbConfig = cuphy::get_HDF5_struct(fInput, "gnb_pars");
            hdf5hpp::hdf5_dataset   tbpDset   = fInput.open_dataset("tb_pars");
            hdf5hpp::hdf5_dataset   ueGrpDset = fInput.open_dataset("ueGrp_pars");
            cuphy::cuphyHDF5_struct tbConfig  = cuphy::get_HDF5_struct_index(tbpDset, 0);

            nUeGrps += gnbConfig.get_value_as<uint16_t>("nUserGroups");
            uint16_t nUes = gnbConfig.get_value_as<uint16_t>("numTb");
            uint8_t  enableCsiP2Fapiv3 = gnbConfig.get_value_as<uint8_t>("enableCsiP2Fapiv3");
            nBytesPerCell = 0;
            for(uint16_t ueIdx = 0; ueIdx < nUes; ++ueIdx, ++globalUeIdx)
            {
                tbConfig = cuphy::get_HDF5_struct_index(tbpDset, ueIdx);

                if(uePrmsVec[globalUeIdx].pduBitmap & 1) // check 0th bit for data transmission
                {
                    bStartOffsetsTbPayloadDatasetsVec[globalUeIdx] = nBytesPerCell;

                    uint32_t nCbsThisTb                    = tbConfig.get_value_as<uint32_t>("nCb");
                    uint32_t nTbByte                       = tbConfig.get_value_as<uint32_t>("nTbByte");
                    uint8_t  crcByteSize                   = (nTbByte * 8) > 3824 ? 3 : 2; // 38.212, section 7.2.1
                    uint32_t nBytesThisTbPayload           = nTbByte + crcByteSize;        // in cuPHY each TB includes TB payload + TB CRC
                    uint32_t tbPayloadWordAlignBytePadding = (sizeof(uint32_t) - (nBytesThisTbPayload % sizeof(uint32_t))) % sizeof(uint32_t);

                    nCbs += nCbsThisTb;
                    nTbs += 1;
                    nBytes += nBytesThisTbPayload + tbPayloadWordAlignBytePadding;
                    nBytesPerCell += nBytesThisTbPayload + tbPayloadWordAlignBytePadding;

                    NVLOGD_FMT(NVLOG_PUSCH, "globalUeIdx {} nCbs {} nTbs {} nBytes {} nBytesPerCell {}", globalUeIdx, nCbs, nTbs, nBytes, nBytesPerCell);
                    NVLOGD_FMT(NVLOG_PUSCH, "globalUeIdx {} nCbsThisTb {} nBytesThisTbPayload {} tbPayloadWordAlignBytePadding {} nCbs {} nTbs {} nBytes {} nBytesPerCell {}", globalUeIdx, nCbsThisTb, nBytesThisTbPayload, tbPayloadWordAlignBytePadding, nCbs, nTbs, nBytes, nBytesPerCell);
                }

                if(uePrmsVec[globalUeIdx].pduBitmap & 2) // check 1st bit for uci transmission
                {
                    nUciUes += 1;
                    uint16_t nBitsHarq = uePrmsVec[globalUeIdx].pUciPrms->nBitsHarq;
                    if(nBitsHarq > 0)
                    {
                        nUciSegs += 1;
                        nUciPayloadBytes += sizeof(uint32_t) * div_round_up(nBitsHarq, static_cast<uint16_t>(32));
                    }

                    uint16_t nBitsCsi1 = uePrmsVec[globalUeIdx].pUciPrms->nBitsCsi1;
                    if(nBitsCsi1 > 0)
                    {
                        nUciSegs += 1;
                        nUciPayloadBytes += sizeof(uint32_t) * div_round_up(nBitsCsi1, static_cast<uint16_t>(32));
                    }

                    bool isCsi2Present = false;
                    if(enableCsiP2Fapiv3)
                    {
                        uint8_t nCsi2Reports = uePrmsVec[globalUeIdx].pUciPrms->nCsi2Reports;
                        if(nCsi2Reports > 0)
                        {
                            isCsi2Present = true;
                        }
                    }else
                    {
                        if((uePrmsVec[globalUeIdx].pduBitmap >> 5) & 1)
                        {
                            isCsi2Present = true;
                        }
                    }
                    if(isCsi2Present)
                    {
                        nUciSegs += 1;
                        nUciPayloadBytes += sizeof(uint32_t) * CUPHY_MAX_N_CSI2_WORDS;
                    }
                }
                NVLOGD_FMT(NVLOG_PUSCH, "ueIdx {} nAccumCbs {} nAccumTbs {} nAccumBytes {} nBytes {}", ueIdx, nCbs, nTbs, nBytes, tbConfig.get_value_as<uint32_t>("nTbByte"));
            }
        }

        totNumUes = globalUeIdx + 1;
        bHarqBufferSizeInBytes.resize(totNumUes);
        bHarqBufferPtrs             = std::move(cuphy::buffer<uint8_t*, cuphy::pinned_alloc>(totNumUes));
        DataInOut.pHarqBuffersInOut = bHarqBufferPtrs.addr();
        
        bFoCompensationBufferPtrs             = std::move(cuphy::buffer<float*, cuphy::pinned_alloc>(totNumUes));
        DataInOut.pFoCompensationBuffersInOut = bFoCompensationBufferPtrs.addr();

        bHarqDetectionStatus  = std::move(cuphy::buffer<uint8_t, cuphy::pinned_alloc>(totNumUes));
        bCsiP1DetectionStatus = std::move(cuphy::buffer<uint8_t, cuphy::pinned_alloc>(totNumUes));
        bCsiP2DetectionStatus = std::move(cuphy::buffer<uint8_t, cuphy::pinned_alloc>(totNumUes));
        
        memset(bHarqDetectionStatus.addr(), 0, sizeof(uint8_t) * totNumUes);
        memset(bCsiP1DetectionStatus.addr(), 0, sizeof(uint8_t) * totNumUes);
        memset(bCsiP2DetectionStatus.addr(), 0, sizeof(uint8_t) * totNumUes);

        if(nTbs > 0)
        {
            bCbCrcs     = std::move(cuphy::buffer<uint32_t, cuphy::pinned_alloc>(nCbs));
            bTbCrcs     = std::move(cuphy::buffer<uint32_t, cuphy::pinned_alloc>(nTbs));
            bTbPayloads = std::move(cuphy::buffer<uint8_t, cuphy::pinned_alloc>(nBytes));
            //outTaEsts = std::move(cuphy::buffer<float, cuphy::pinned_alloc>(nTbs));
            
            memset(bCbCrcs.addr(), 1, sizeof(uint32_t) * nCbs);
            memset(bTbCrcs.addr(), 1, sizeof(uint32_t) * nTbs);
        }

        outRssi   = std::move(cuphy::buffer<float, cuphy::pinned_alloc>(nUeGrps));
        outRsrp   = std::move(cuphy::buffer<float, cuphy::pinned_alloc>(cellGrpDynPrm.nUes));
#if USE_PUSCH_PER_UE_PREQ_NOISE_VAR
        outNoiseVarPreEq  = std::move(cuphy::buffer<float, cuphy::pinned_alloc>(cellGrpDynPrm.nUes));
#else
        outNoiseVarPreEq  = std::move(cuphy::buffer<float, cuphy::pinned_alloc>(nUeGrps));
#endif
        outNoiseVarPostEq = std::move(cuphy::buffer<float, cuphy::pinned_alloc>(cellGrpDynPrm.nUes));
        outSinrPreEq  = std::move(cuphy::buffer<float, cuphy::pinned_alloc>(cellGrpDynPrm.nUes));
        outSinrPostEq = std::move(cuphy::buffer<float, cuphy::pinned_alloc>(cellGrpDynPrm.nUes));
        outTaEsts = std::move(cuphy::buffer<float, cuphy::pinned_alloc>(cellGrpDynPrm.nUes));
        outCfoHz = std::move(cuphy::buffer<float, cuphy::pinned_alloc>(cellGrpDynPrm.nUes));


        if(nUciSegs > 0)
        {
            bUciPayloads          = std::move(cuphy::buffer<uint8_t, cuphy::pinned_alloc>(nUciPayloadBytes));
            bUciCrcFlags          = std::move(cuphy::buffer<uint8_t, cuphy::pinned_alloc>(nUciSegs));
            bUciOnPuschOutOffsets = std::move(cuphy::buffer<cuphyUciOnPuschOutOffsets_t, cuphy::pinned_alloc>(totNumUes));
            bNumCsi2Bits          = std::move(cuphy::buffer<uint16_t, cuphy::pinned_alloc>(nUciSegs));
            
            memset(bUciCrcFlags.addr(), 1, sizeof(uint8_t) * nUciSegs);
        }

        DataOut.pStartOffsetsCbCrc      = bStartOffsetsCbCrc.addr();
        DataOut.pStartOffsetsTbCrc      = bStartOffsetsTbCrc.addr();
        DataOut.pStartOffsetsTbPayload  = bStartOffsetsTbPayload.addr();
        DataOut.pDataRx                 = nullptr;
        DataOut.pChannelEsts            = nullptr;
        DataOut.pChannelEstSizes        = nullptr;
        DataOut.pCbCrcs                 = bCbCrcs.addr();
        DataOut.pTbCrcs                 = bTbCrcs.addr();
        DataOut.pTbPayloads             = bTbPayloads.addr();
        DataOut.HarqDetectionStatus     = bHarqDetectionStatus.addr();
        DataOut.CsiP1DetectionStatus    = bCsiP1DetectionStatus.addr();
        DataOut.CsiP2DetectionStatus    = bCsiP2DetectionStatus.addr();
        DataOut.pTaEsts                 = outTaEsts.addr();
        DataOut.pRssi                   = outRssi.addr();
        DataOut.pRsrp                   = outRsrp.addr();
        DataOut.pNoiseVarPreEq          = outNoiseVarPreEq.addr();
        DataOut.pSinrPreEq              = outSinrPreEq.addr();
        DataOut.pNoiseVarPostEq         = outNoiseVarPostEq.addr();
        DataOut.pSinrPostEq             = outSinrPostEq.addr();
        DataOut.pCfoHz                  = outCfoHz.addr();
        DataOut.h_harqBufferSizeInBytes = bHarqBufferSizeInBytes.data();
        DataOut.pUciPayloads            = bUciPayloads.addr();
        DataOut.pUciCrcFlags            = bUciCrcFlags.addr();
        DataOut.pUciOnPuschOutOffsets   = bUciOnPuschOutOffsets.addr();
        DataOut.pNumCsi2Bits            = bNumCsi2Bits.addr();

        if((procMode & PUSCH_PROC_MODE_SUB_SLOT) && (totNumUes > 0) && (nUciSegs > 0))
        {
            bEvalHarqDetectionStatus  = std::move(cuphy::buffer<uint8_t, cuphy::pinned_alloc>(totNumUes));
            bEvalUciPayloads          = std::move(cuphy::buffer<uint8_t, cuphy::pinned_alloc>(nUciPayloadBytes));
            bEvalUciCrcFlags          = std::move(cuphy::buffer<uint8_t, cuphy::pinned_alloc>(nUciSegs));
            bEvalPreEqSinr            = std::move(cuphy::buffer<float, cuphy::pinned_alloc>(totNumUes));
            bEvalRsrp                 = std::move(cuphy::buffer<float, cuphy::pinned_alloc>(totNumUes));
            bEvalRssi                 = std::move(cuphy::buffer<float, cuphy::pinned_alloc>(nUeGrps));

            evalHarqDetectionStatus   = bEvalHarqDetectionStatus.addr(); 
            evalUciPayloads           = bEvalUciPayloads.addr(); 
            evalUciCrcFlags           = bEvalUciCrcFlags.addr();
            evalPreEqSinrEhq          = bEvalPreEqSinr.addr();
            evalRsrpEhq               = bEvalRsrp.addr();
            evalRssiEhq               = bEvalRssi.addr();
        }

        /*----------------------------- puschDynPrm parameters ----------------------------------------*/
        puschDynPrm.phase1Stream   = cuStrm;
        puschDynPrm.phase2Stream   = cuStrm;
        puschDynPrm.procModeBmsk   = procMode;
        puschDynPrm.pDataIn        = &DataIn;
        puschDynPrm.pDataInOut     = &DataInOut;
        puschDynPrm.pDataOut       = &DataOut;
        puschDynPrm.pCellGrpDynPrm = &cellGrpDynPrm;
        puschDynPrm.cpuCopyOn      = static_cast<uint8_t>(cpuCopyOn);
        puschDynPrm.pDbg           = &dbgPrm;

        StatusOutput = {cuphyPuschStatusType_t::CUPHY_PUSCH_STATUS_SUCCESS_OR_UNTRACKED_ISSUE, MAX_UINT16, MAX_UINT16};
        puschDynPrm.pStatusOut = &StatusOutput;

        /********************************************************/
        //For the first scheduler kernel (pre-early-HARQ) set timeout to 1100us which is >= 400us before T0 (which is PUSCH UL worker start) + 640us after T0 (~440us for symbol-3 arrival + 200us for reorder latency).
        //For the second scheduler kernel (post-early-HARQ) set timeout to 1500us which is >= 400us before T0 + 1100us after T0 (~800us for symbol-13 arrival + 200us reorder latency)
        puschDynPrm.waitTimeOutPreEarlyHarqUs = 1100; //in us
        puschDynPrm.waitTimeOutPostEarlyHarqUs = 1500; //in us
        /********************************************************/
    }
    else // api test-vector
    {
        for(uint32_t i = 0; i < inputFileNameVec.size(); i++)
        {
            // load 5h file
            hdf5hpp::hdf5_file      fInput          = hdf5hpp::hdf5_file::open(inputFileNameVec[i].c_str());
            cuphy::cuphyHDF5_struct cellGrpDyn_pars = cuphy::get_HDF5_struct(fInput, "cellGrpDyn_pars");
            cuphy::cuphyHDF5_struct cellDyn_pars    = cuphy::get_HDF5_struct(fInput, "cellDyn_pars");
            hdf5hpp::hdf5_dataset   ueGrp_pars      = fInput.open_dataset("ueGrp_pars");
            hdf5hpp::hdf5_dataset   ue_pars         = fInput.open_dataset("ue_pars");
            cuphy::cuphyHDF5_struct dmrs_pars       = cuphy::get_HDF5_struct(fInput, "dmrs_pars");
            hdf5hpp::hdf5_dataset   output_pars     = fInput.open_dataset("output_pars");

            // cell parameters
            cellDynPrmVec[i].cellPrmStatIdx = i;
            cellDynPrmVec[i].cellPrmDynIdx  = i;
            cellDynPrmVec[i].slotNum        = cellDyn_pars.get_value_as<uint16_t>("slotNum");
#if 0
        // dynamic DMRS parameters
        dmrsPrm.dmrsAddlnPos          = dmrs_pars.get_value_as<uint8_t>("dmrsAddlnPos");
        dmrsPrm.dmrsMaxLen            = dmrs_pars.get_value_as<uint8_t>("dmrsMaxLen");
        dmrsPrm.nDmrsCdmGrpsNoData    = 1;
        dmrsPrm.dmrsScrmId            = dmrs_pars.get_value_as<uint8_t>("dmrsScramId");
#endif

            // ue group parameters
            uint16_t nUeGrps = cellGrpDyn_pars.get_value_as<uint16_t>("nUeGrps");
            ueGrpPrmsVec.resize(nUeGrps);
            ueGrpToUeIdxs.resize(nUeGrps);
            dmrsPrmVec.resize(nUeGrps);

            for(int ueGrpIdx = 0; ueGrpIdx < nUeGrps; ++ueGrpIdx)
            {
                cuphy::cuphyHDF5_struct ueGrpCfg = cuphy::get_HDF5_struct_index(ueGrp_pars, ueGrpIdx);

                std::vector<uint16_t> ueIdxVec = (ueGrp_pars[ueGrpIdx]["UePrmIdxs"].as<std::vector<uint16_t>>());
                ueGrpToUeIdxs[ueGrpIdx]        = ueIdxVec;

                ueGrpPrmsVec[ueGrpIdx].pCellPrm = &cellDynPrmVec[0];
                //ueGrpPrmsVec[ueGrpIdx].pDmrsDynPrm    = &dmrsPrmVec;
                ueGrpPrmsVec[ueGrpIdx].startPrb   = ueGrpCfg.get_value_as<uint16_t>("startPrb");
                ueGrpPrmsVec[ueGrpIdx].nPrb       = ueGrpCfg.get_value_as<uint16_t>("nPrb");
                ueGrpPrmsVec[ueGrpIdx].nUes       = ueGrpCfg.get_value_as<uint16_t>("nUes");
                ueGrpPrmsVec[ueGrpIdx].nUplinkStreams = ueGrpCfg.get_value_as<uint16_t>("nUplinkStreams");
                ueGrpPrmsVec[ueGrpIdx].pUePrmIdxs = ueGrpToUeIdxs[ueGrpIdx].data();
            }

            // ue parameters
            uint16_t nUes = cellGrpDyn_pars.get_value_as<uint16_t>("nUes");
            uePrmsVec.resize(nUes);
            for(uint16_t ueIdx = 0; ueIdx < nUes; ++ueIdx)
            {
                cuphy::cuphyHDF5_struct ueCfg = cuphy::get_HDF5_struct_index(ue_pars, ueIdx);

                uint16_t ueGrpIdx                       = ueCfg.get_value_as<uint16_t>("ueGrpIdx");
                uePrmsVec[ueIdx].pUeGrpPrm              = &ueGrpPrmsVec[ueGrpIdx];
                uePrmsVec[ueIdx].ueGrpIdx               = ueGrpIdx;
                dmrsPrmVec[ueGrpIdx].dmrsAddlnPos       = dmrs_pars.get_value_as<uint8_t>("dmrsAddlnPos");
                dmrsPrmVec[ueGrpIdx].dmrsMaxLen         = dmrs_pars.get_value_as<uint8_t>("dmrsMaxLen");
                dmrsPrmVec[ueGrpIdx].nDmrsCdmGrpsNoData = 1;
                dmrsPrmVec[ueGrpIdx].dmrsScrmId         = dmrs_pars.get_value_as<uint8_t>("dmrsScramId");
                ueGrpPrmsVec[ueGrpIdx].pDmrsDynPrm      = &dmrsPrmVec[ueGrpIdx];

                uePrmsVec[ueIdx].enableTfPrcd           = ueCfg.get_value_as<uint8_t>("enableTfPrcd");
                if(uePrmsVec[ueIdx].enableTfPrcd==1)
                {
                  uePrmsVec[ueIdx].puschIdentity          = ueCfg.get_value_as<uint32_t>("puschIdentity");
                  uePrmsVec[ueIdx].groupOrSequenceHopping = ueCfg.get_value_as<uint8_t>("groupOrSequenceHopping");
                  uePrmsVec[ueIdx].N_symb_slot            = ueCfg.get_value_as<uint8_t>("N_symb_slot");
                  uePrmsVec[ueIdx].N_slot_frame           = ueCfg.get_value_as<uint8_t>("N_slot_frame");
                  uePrmsVec[ueIdx].lowPaprGroupNumber     = ueCfg.get_value_as<uint8_t>("lowPaprGroupNumber");
                  uePrmsVec[ueIdx].lowPaprSequenceNumber  = ueCfg.get_value_as<uint8_t>("lowPaprSequenceNumber");
                }
                uePrmsVec[ueIdx].scid                   = ueCfg.get_value_as<uint8_t>("scid");
                uePrmsVec[ueIdx].dmrsPortBmsk           = ueCfg.get_value_as<uint16_t>("dmrsPortBmsk");
                uePrmsVec[ueIdx].mcsTableIndex          = 0; // mcsTableIndex is a deprecated API parameter, to be removed after 22-1 ED1
                uePrmsVec[ueIdx].mcsIndex               = 0; // mcsIndex is a deprecated API parameter, to be removed after 22-1 ED1
                uePrmsVec[ueIdx].rv                     = ueCfg.get_value_as<uint8_t>("rv");
                uePrmsVec[ueIdx].rnti                   = ueCfg.get_value_as<uint16_t>("rnti");
                uePrmsVec[ueIdx].dataScramId            = ueCfg.get_value_as<uint16_t>("dataScramId");
                uePrmsVec[ueIdx].nUeLayers              = ueCfg.get_value_as<uint8_t>("nUeLayers");
                uePrmsVec[ueIdx].targetCodeRate         = ueCfg.get_value_as<uint16_t>("targetCodeRate");
                uePrmsVec[ueIdx].qamModOrder            = ueCfg.get_value_as<uint8_t>("qamModOrder");
            }

            // cell group parameters
            cellGrpDynPrm.nCells     = i + 1;
            cellGrpDynPrm.pCellPrms  = &cellDynPrmVec[0];
            cellGrpDynPrm.nUeGrps    = nUeGrps;
            cellGrpDynPrm.pUeGrpPrms = ueGrpPrmsVec.data();
            cellGrpDynPrm.nUes       = nUes;
            cellGrpDynPrm.pUePrms    = uePrmsVec.data();

            // output parameters
            bStartOffsetsCbCrc     = std::move(cuphy::buffer<uint32_t, cuphy::pinned_alloc>(nUes));
            bStartOffsetsTbCrc     = std::move(cuphy::buffer<uint32_t, cuphy::pinned_alloc>(nUes));
            bStartOffsetsTbPayload = std::move(cuphy::buffer<uint32_t, cuphy::pinned_alloc>(nUes));

            std::vector<uint32_t> offsetCbCrc   = (output_pars[0]["offsetCbCrc"].as<std::vector<uint32_t>>());
            std::vector<uint32_t> offsetTbCrc   = (output_pars[0]["offsetTbCrc"].as<std::vector<uint32_t>>());
            std::vector<uint32_t> offsetPayload = (output_pars[0]["offsetPayload"].as<std::vector<uint32_t>>());

            for(uint16_t ueIdx = 0; ueIdx < nUes; ++ueIdx)
            {
                bStartOffsetsCbCrc[ueIdx]     = offsetCbCrc[ueIdx];
                bStartOffsetsTbCrc[ueIdx]     = offsetTbCrc[ueIdx];
                bStartOffsetsTbPayload[ueIdx] = offsetPayload[ueIdx];
            }

            DataOut.totNumCbs          = output_pars[0]["totNumCbs"].as<uint32_t>();
            DataOut.totNumTbs          = output_pars[0]["totNumTbs"].as<uint32_t>();
            DataOut.totNumPayloadBytes = output_pars[0]["totNumPayloadBytes"].as<uint32_t>();

            bCbCrcs     = std::move(cuphy::buffer<uint32_t, cuphy::pinned_alloc>(DataOut.totNumCbs));
            bTbCrcs     = std::move(cuphy::buffer<uint32_t, cuphy::pinned_alloc>(DataOut.totNumTbs));
            bTbPayloads = std::move(cuphy::buffer<uint8_t, cuphy::pinned_alloc>(DataOut.totNumPayloadBytes));
            bHarqDetectionStatus  = std::move(cuphy::buffer<uint8_t, cuphy::pinned_alloc>(DataOut.totNumTbs));
            bCsiP1DetectionStatus = std::move(cuphy::buffer<uint8_t, cuphy::pinned_alloc>(DataOut.totNumTbs));
            bCsiP2DetectionStatus = std::move(cuphy::buffer<uint8_t, cuphy::pinned_alloc>(DataOut.totNumTbs));
            DataOut.pStartOffsetsCbCrc     = bStartOffsetsCbCrc.addr();
            DataOut.pStartOffsetsTbCrc     = bStartOffsetsTbCrc.addr();
            DataOut.pStartOffsetsTbPayload = bStartOffsetsTbPayload.addr();
            DataOut.pDataRx                = nullptr;
            DataOut.pChannelEsts           = nullptr;
            DataOut.pChannelEstSizes       = nullptr;
            DataOut.pCbCrcs                = bCbCrcs.addr();
            DataOut.pTbCrcs                = bTbCrcs.addr();
            DataOut.pTbPayloads            = bTbPayloads.addr();
            DataOut.HarqDetectionStatus    = bHarqDetectionStatus.addr();
            DataOut.CsiP1DetectionStatus   = bCsiP1DetectionStatus.addr();
            DataOut.CsiP2DetectionStatus   = bCsiP2DetectionStatus.addr();

            // input parameters
            cuphyDataType_t cplxTypeDataRx = fp16Mode ? CUPHY_C_16F : CUPHY_C_32F;
            tDataRxVec[0]                    = cuphy::tensor_from_dataset(fInput.open_dataset("DataRx"), cplxTypeDataRx, cuphy::tensor_flags::align_tight, cuStrm);
            cudaStreamSynchronize(cuStrm);

            tPrmDataRxVec[0].desc  = tDataRxVec[0].desc().handle();
            tPrmDataRxVec[0].pAddr = tDataRxVec[0].addr();
            DataIn.pTDataRx        = &tPrmDataRxVec[0];

            puschDynPrm.procModeBmsk   = procMode;
            puschDynPrm.pDataIn        = &DataIn;
            puschDynPrm.pDataOut       = &DataOut;
            puschDynPrm.pCellGrpDynPrm = &cellGrpDynPrm;
            puschDynPrm.cpuCopyOn      = static_cast<uint8_t>(cpuCopyOn);
            puschDynPrm.pDbg           = &dbgPrm;

            StatusOutput = {cuphyPuschStatusType_t::CUPHY_PUSCH_STATUS_SUCCESS_OR_UNTRACKED_ISSUE, MAX_UINT16, MAX_UINT16};
            puschDynPrm.pStatusOut = &StatusOutput;

            puschDynPrm.waitTimeOutPreEarlyHarqUs = 1100; //in us
            puschDynPrm.waitTimeOutPostEarlyHarqUs = 1500; //in us
        }
    }
}

// default constructor
DynApiDataset::DynApiDataset() :
    cellDynPrmVec{},
    dmrsPrmVec{},
    ueGrpPrmsVec{},
    uePrmsVec{},
    cellGrpDynPrm{},
    puschDynPrm{},
    DataOut{},
    DataIn{},
    tPrmDataRxVec{},
    tDataRxVec{},
    bCbCrcs{},
    bTbCrcs{},
    bTbPayloads{},
    bHarqDetectionStatus{},
    bCsiP1DetectionStatus{},
    bCsiP2DetectionStatus{},
    bStartOffsetsCbCrc{},
    bStartOffsetsTbCrc{},
    bStartOffsetsTbPayload{},
    ueGrpToUeIdxs{}
{
}

DynApiDataset::~DynApiDataset()
{
    CUDA_CHECK_NO_THROW(cudaFreeHost(pPreEarlyHarqWaitKernelStatus));
    CUDA_CHECK_NO_THROW(cudaFreeHost(pPostEarlyHarqWaitKernelStatus));
}

// reset pointers after a copy or move
void DynApiDataset::ResetPointers()
{
    for(uint32_t i = 0; i < tPrmDataRxVec.size(); i++)
    {
        tPrmDataRxVec[i].pAddr = tDataRxVec[i].addr();
        tPrmDataRxVec[i].desc  = tDataRxVec[i].desc().handle();
    }

    for(int i = 0; i < cellGrpDynPrm.nUeGrps; ++i)
    {
        ueGrpPrmsVec[i].pCellPrm    = &cellDynPrmVec[0];
        ueGrpPrmsVec[i].pDmrsDynPrm = &dmrsPrmVec[i];
        ueGrpPrmsVec[i].pUePrmIdxs  = ueGrpToUeIdxs[i].data();
    }

    for(int i = 0; i < cellGrpDynPrm.nUes; ++i)
    {
        uePrmsVec[i].pUeGrpPrm = &(ueGrpPrmsVec[uePrmsVec[i].ueGrpIdx]);
    }

    cellGrpDynPrm.pCellPrms  = &cellDynPrmVec[0];
    cellGrpDynPrm.pUeGrpPrms = ueGrpPrmsVec.data();
    cellGrpDynPrm.pUePrms    = uePrmsVec.data();

    puschDynPrm.pCellGrpDynPrm = &cellGrpDynPrm;
    puschDynPrm.pDataIn        = &DataIn;
    puschDynPrm.pDataOut       = &DataOut;

    DataOut.pDataRx                = nullptr;
    DataOut.pChannelEsts           = nullptr;
    DataOut.pChannelEstSizes       = nullptr;
    DataOut.pCbCrcs                = bCbCrcs.addr();
    DataOut.pTbCrcs                = bTbCrcs.addr();
    DataOut.pTbPayloads            = bTbPayloads.addr();
    DataOut.HarqDetectionStatus    = bHarqDetectionStatus.addr();
    DataOut.CsiP1DetectionStatus   = bCsiP1DetectionStatus.addr();
    DataOut.CsiP2DetectionStatus   = bCsiP2DetectionStatus.addr();
    DataOut.pStartOffsetsCbCrc     = bStartOffsetsCbCrc.addr();
    DataOut.pStartOffsetsTbCrc     = bStartOffsetsTbCrc.addr();
    DataOut.pStartOffsetsTbPayload = bStartOffsetsTbPayload.addr();

    DataIn.pTDataRx   = &tPrmDataRxVec[0];
}

void DynApiDataset::EasyAllocHarqBuffers(cudaStream_t strm)
{
    harqBuffers.clear();

    // allocate the buffers.  std::vector will call cuphy::buffer destructor when DynApiDataset goes out of scope
    for(int k = 0; k < DataOut.totNumTbs; k++)
    {
        harqBuffers.push_back(std::move(cuphy::buffer<uint8_t, cuphy::device_alloc>(DataOut.h_harqBufferSizeInBytes[k])));

        // Note: the following code won't work for HARQ TVs 7324-7326 where multiple back to back slots are tested.
        // This is ok since HARQ is tested in phase-2 test bench where EasyAllocHarqBuffers is unused.

        // For 1st transmission invalidate the entire HARQ buffer to test if its correctly initialized internally in cuPHY
        if(uePrmsVec[k].ndi)
        {
            CUDA_CHECK(cudaMemsetAsync(harqBuffers[k].addr(), 0xFF, DataOut.h_harqBufferSizeInBytes[k], strm));
        }
    }

    // Copy the pointer of each buffer to DataInOut
    for(int k = 0; k < DataOut.totNumTbs; k++)
    {
        DataInOut.pHarqBuffersInOut[k] = harqBuffers[k].addr();
    }
}

// move operator
DynApiDataset& DynApiDataset::operator=(DynApiDataset&& dynApiDataset)
{
    tDataRxVec             = std::move(dynApiDataset.tDataRxVec);
    tPrmDataRxVec          = std::move(dynApiDataset.tPrmDataRxVec);
    bCbCrcs                = std::move(dynApiDataset.bCbCrcs);
    bTbCrcs                = std::move(dynApiDataset.bTbCrcs);
    bTbPayloads            = std::move(dynApiDataset.bTbPayloads);
    bHarqDetectionStatus   = std::move(dynApiDataset.bHarqDetectionStatus);
    bCsiP1DetectionStatus  = std::move(dynApiDataset.bCsiP1DetectionStatus);
    bCsiP2DetectionStatus  = std::move(dynApiDataset.bCsiP2DetectionStatus);
    bStartOffsetsCbCrc     = std::move(dynApiDataset.bStartOffsetsCbCrc);
    bStartOffsetsTbCrc     = std::move(dynApiDataset.bStartOffsetsTbCrc);
    bStartOffsetsTbPayload = std::move(dynApiDataset.bStartOffsetsTbPayload);
    ueGrpToUeIdxs          = std::move(dynApiDataset.ueGrpToUeIdxs);
    cellDynPrmVec          = std::move(dynApiDataset.cellDynPrmVec);
    dmrsPrmVec             = std::move(dynApiDataset.dmrsPrmVec);
    ueGrpPrmsVec           = std::move(dynApiDataset.ueGrpPrmsVec);
    uePrmsVec              = std::move(dynApiDataset.uePrmsVec);
    cellGrpDynPrm          = std::move(dynApiDataset.cellGrpDynPrm);
    puschDynPrm            = std::move(dynApiDataset.puschDynPrm);
    dbgPrm                 = std::move(dynApiDataset.dbgPrm);
    DataOut                = std::move(dynApiDataset.DataOut);
    DataIn                 = std::move(dynApiDataset.DataIn);

    ResetPointers();
    return *this;
}

// copy constructor
DynApiDataset::DynApiDataset(const DynApiDataset& dynApiDataset) :
    cellDynPrmVec(dynApiDataset.cellDynPrmVec),
    dmrsPrmVec(dynApiDataset.dmrsPrmVec),
    ueGrpPrmsVec(dynApiDataset.ueGrpPrmsVec),
    uePrmsVec(dynApiDataset.uePrmsVec),
    cellGrpDynPrm(dynApiDataset.cellGrpDynPrm),
    puschDynPrm(dynApiDataset.puschDynPrm),
    dbgPrm(dynApiDataset.dbgPrm),
    DataOut(dynApiDataset.DataOut),
    DataIn(dynApiDataset.DataIn),
    tPrmDataRxVec(dynApiDataset.tPrmDataRxVec),
    tDataRxVec(dynApiDataset.tDataRxVec),
    bCbCrcs(dynApiDataset.bCbCrcs),
    bTbCrcs(dynApiDataset.bTbCrcs),
    bTbPayloads(dynApiDataset.bTbPayloads),
    bHarqDetectionStatus(dynApiDataset.bHarqDetectionStatus),
    bCsiP1DetectionStatus(dynApiDataset.bCsiP1DetectionStatus),
    bCsiP2DetectionStatus(dynApiDataset.bCsiP2DetectionStatus),
    bStartOffsetsCbCrc(dynApiDataset.bStartOffsetsCbCrc),
    bStartOffsetsTbCrc(dynApiDataset.bStartOffsetsTbCrc),
    bStartOffsetsTbPayload(dynApiDataset.bStartOffsetsTbPayload),
    ueGrpToUeIdxs(dynApiDataset.ueGrpToUeIdxs)
{
    // synch default stream, used to copy
    cudaStreamSynchronize(0);

    // update pointers
    ResetPointers();
}

//-----------------------------------------------------------------------------------------------------------
//  Dataset holds static api parameters/data

void StaticApiDataset::puschInitCellStatPrm(const std::vector<std::string>& inputFileNameVec, int apiTVflag, cudaStream_t cuStrm, const maxPUSCHPrms* puschPrms)
{
    puschStatPrms.nMaxCbsPerTb = 0;
    puschStatPrms.nMaxTbs      = 0;
    puschStatPrms.nMaxTotCbs   = 0;
    puschStatPrms.nMaxRx       = 0;
    puschStatPrms.nMaxPrb      = 0;

    if(puschPrms != nullptr)
    {
        puschStatPrms.nMaxCbsPerTb = puschPrms->maxNCbsPerTb ? puschPrms->maxNCbsPerTb : puschStatPrms.nMaxCbsPerTb;
        puschStatPrms.nMaxTbs      = puschPrms->maxNTbs ? puschPrms->maxNTbs : puschStatPrms.nMaxTbs;
        puschStatPrms.nMaxTotCbs   = puschPrms->maxNCbs ? puschPrms->maxNCbs : puschStatPrms.nMaxTotCbs;
        puschStatPrms.nMaxRx       = puschPrms->maxNRx ? puschPrms->maxNRx : puschStatPrms.nMaxRx;
        puschStatPrms.nMaxPrb      = puschPrms->maxNPrbs ? puschPrms->maxNPrbs : puschStatPrms.nMaxPrb;
    }

    cellStatPrmVec.resize(inputFileNameVec.size());
    puschCellStatPrmVec.resize(inputFileNameVec.size());
    gpuCsi2Maps.resize(inputFileNameVec.size());
    gpuCsi2MapPrms.resize(inputFileNameVec.size());

    for(uint32_t i = 0; i < inputFileNameVec.size(); i++)
    {
        // load h5 file
        hdf5hpp::hdf5_file fInput = hdf5hpp::hdf5_file::open(inputFileNameVec[i].c_str());
        if(apiTVflag == 0)
        {
            cuphy::cuphyHDF5_struct gnbConfig = cuphy::get_HDF5_struct(fInput, "gnb_pars");
            hdf5hpp::hdf5_dataset   gnbDset   = fInput.open_dataset("gnb_pars");

            // pusch cell static parameters:
            puschCellStatPrmVec[i].nCsirsPorts      = static_cast<uint8_t>(4);
            puschCellStatPrmVec[i].N1               = static_cast<uint8_t>(2);
            puschCellStatPrmVec[i].N2               = static_cast<uint8_t>(1);
            puschCellStatPrmVec[i].csiReportingBand = static_cast<uint8_t>(0);
            puschCellStatPrmVec[i].codebookType     = static_cast<uint8_t>(0);
            puschCellStatPrmVec[i].codebookMode     = static_cast<uint8_t>(1);
            puschCellStatPrmVec[i].isCqi            = static_cast<uint8_t>(0);
            puschCellStatPrmVec[i].isLi             = static_cast<uint8_t>(0);

            // number of CSI2 Maps:
            puschCellStatPrmVec[i].nCsi2Maps = gnbConfig.get_value_as<uint8_t>("nCsi2Maps");

            // CSI2 Map buffer:
            std::vector<uint16_t> csi2MapsVec = gnbDset[0]["csi2Maps_buffer"].as<std::vector<uint16_t>>();
            uint16_t csi2MapBufferSize        = csi2MapsVec.size();

            gpuCsi2Maps[i] = std::move(cuphy::buffer<uint16_t, cuphy::device_alloc>(csi2MapBufferSize));
            CUDA_CHECK(cudaMemcpyAsync(gpuCsi2Maps[i].addr(), csi2MapsVec.data(), sizeof(uint16_t) * csi2MapBufferSize, cudaMemcpyHostToDevice, cuStrm));
            puschCellStatPrmVec[i].pCsi2MapBuffer = static_cast<uint16_t*>(gpuCsi2Maps[i].addr());

            // CSI2 Map paramaters:
            std::vector<cuphyCsi2MapPrm_t>                 csi2MapPrmsVec(puschCellStatPrmVec[i].nCsi2Maps);
            std::vector<uint8_t>  csi2Maps_sumOfPrmSizes   = gnbDset[0]["csi2Maps_sumOfPrmSizes"].as<std::vector<uint8_t>>();
            std::vector<uint32_t> csi2Maps_bufferStartIdxs = gnbDset[0]["csi2Maps_bufferStartIdxs"].as<std::vector<uint32_t>>();

            for(int csi2MapIdx = 0; csi2MapIdx < puschCellStatPrmVec[i].nCsi2Maps; ++csi2MapIdx)
            {
                csi2MapPrmsVec[csi2MapIdx].csi2MapSize     = pow(2, csi2Maps_sumOfPrmSizes[csi2MapIdx]);
                csi2MapPrmsVec[csi2MapIdx].csi2MapStartIdx = csi2Maps_bufferStartIdxs[csi2MapIdx];
            }

            gpuCsi2MapPrms[i] = std::move(cuphy::buffer<cuphyCsi2MapPrm_t, cuphy::device_alloc>(puschCellStatPrmVec[i].nCsi2Maps));
            CUDA_CHECK(cudaMemcpyAsync(gpuCsi2MapPrms[i].addr(), csi2MapPrmsVec.data(), sizeof(cuphyCsi2MapPrm_t) * puschCellStatPrmVec[i].nCsi2Maps, cudaMemcpyHostToDevice, cuStrm));
            puschCellStatPrmVec[i].pCsi2MapPrm = static_cast<cuphyCsi2MapPrm_t*>(gpuCsi2MapPrms[i].addr());

            // static cell parameters
            cellStatPrmVec[i].phyCellId          = gnbConfig.get_value_as<uint16_t>("cellId");
            cellStatPrmVec[i].nRxAnt             = gnbConfig.get_value_as<uint16_t>("nRx");
            cellStatPrmVec[i].nTxAnt             = gnbConfig.get_value_as<uint16_t>("nRx");
            cellStatPrmVec[i].nPrbUlBwp          = gnbConfig.get_value_as<uint16_t>("nPrb");
            cellStatPrmVec[i].nPrbDlBwp          = gnbConfig.get_value_as<uint16_t>("nPrb");
            cellStatPrmVec[i].mu                 = gnbConfig.get_value_as<uint8_t>("mu");
            cellStatPrmVec[i].pPuschCellStatPrms = &(puschCellStatPrmVec[i]);
            cellStatPrmVec[i].pPucchCellStatPrms = nullptr;
        }
        else
        {
            // static cell parameters
            cuphy::cuphyHDF5_struct cellStatic = cuphy::get_HDF5_struct(fInput, "cellStat_pars");
            cellStatPrmVec[i].phyCellId        = cellStatic.get_value_as<uint16_t>("phyCellId");
            cellStatPrmVec[i].nRxAnt           = cellStatic.get_value_as<uint16_t>("nRxAnt");
            cellStatPrmVec[i].nTxAnt           = cellStatic.get_value_as<uint16_t>("nTxAnt");
            cellStatPrmVec[i].nPrbUlBwp        = cellStatic.get_value_as<uint16_t>("nPrbUlBwp");
            cellStatPrmVec[i].nPrbDlBwp        = cellStatic.get_value_as<uint16_t>("nPrbDlBwp");
            cellStatPrmVec[i].mu               = cellStatic.get_value_as<uint8_t>("mu");
        }
    }
}

void StaticApiDataset::setWorkCancelValue(uint8_t work_cancellation)
{
    memset(hWorkCancelPtr.get(), work_cancellation, sizeof(uint8_t));
}

// construct from h5 file
StaticApiDataset::StaticApiDataset(const std::vector<std::string>& inputFileNameVec, cudaStream_t cuStrm, std::string outputFileName,
                                   int descramblingOn, int apiTVflag, bool enableLdpcThroughputMode, const maxPUSCHPrms* puschPrms,
                                   cuphyPuschLdpcKernelLaunch_t ldpcLaunchMode)
{
    puschInitCellStatPrm(inputFileNameVec, apiTVflag, cuStrm, puschPrms);
    // load h5 file
    hdf5hpp::hdf5_file fInput = hdf5hpp::hdf5_file::open(inputFileNameVec[0].c_str());

    // API logging
    dbgPrm.enableApiLogging = 0;

    // GPU memory footprint tracking
    puschTracker.pMemoryFootprint = nullptr;
    puschStatPrms.pOutInfo        = &puschTracker;

    // Allocate host pinned memory for work cancellation and memset to 0
    hWorkCancelPtr = make_unique_pinned<uint8_t>(1);
    memset(hWorkCancelPtr.get(), 0, sizeof(uint8_t));
    puschStatPrms.pWorkCancelInfo = hWorkCancelPtr.get();
    // Update work cancellation mode (disabled by default)
    puschStatPrms.workCancelMode = PUSCH_NO_WORK_CANCEL; // Alternatives of interest PUSCH_COND_IF_NODES_W_KERNEL, PUSCH_DEVICE_GRAPHS
    //puschStatPrms.workCancelMode = PUSCH_COND_IF_NODES_W_KERNEL;

    // stream priority
    puschStatPrms.stream_priority = PUSCH_STREAM_PRIORITY;

    // default list size for polar decoder in PUSCH is set to 8
    puschStatPrms.polarDcdrListSz = 8;

    // For now, it is assumed all Rx symbols are ready in dataset, hence symStats initialized with 1
    std::vector<uint32_t> symStats(OFDM_SYMBOLS_PER_SLOT, SYM_RX_DONE);
    bSymRxStatus = std::move(cuphy::buffer<uint32_t, cuphy::device_alloc>(symStats));
    puschStatPrms.pSymRxStatus = static_cast<uint32_t const*>(bSymRxStatus.addr());
    puschStatPrms.subSlotCompletedEvent = subSlotCompletedEvent.handle();
    puschStatPrms.waitCompletedSubSlotEvent = waitCompletedSubSlotEvent.handle();
    puschStatPrms.waitCompletedFullSlotEvent = waitCompletedFullSlotEvent.handle();

    puschStatPrms.enableEarlyHarq          = 1;
    puschStatPrms.enableDeviceGraphLaunch  = 0; // since compute-sanitizer (as of version 2023.3.1.0) for synccheck causes crash in the app when enableDeviceGraphLaunch=1, keep it disabled for now
    puschStatPrms.enableDebugEqOutput      = 0; // disable post-eq debugging output by default
    puschStatPrms.enableBatchedMemcpy      = 1; //FIXME decide on an appropriate default value
    puschStatPrms.nMaxLdpcHetConfigs       = 32; // default number LDPC config nodes to some nominal value
    puschStatPrms.nMaxTbPerNode            = CUPHY_LDPC_DECODE_DESC_MAX_TB; // Use maximum number of transport blocks per node

    if(apiTVflag == 0)
    {
        cuphy::cuphyHDF5_struct gnbConfig = cuphy::get_HDF5_struct(fInput, "gnb_pars");
#if 0
        // static cell parameters
        cellStatPrmVec.phyCellId                = gnbConfig.get_value_as<uint16_t>("cellId");
        cellStatPrmVec.nRxAnt                   = gnbConfig.get_value_as<uint16_t>("nRx");
        cellStatPrmVec.nTxAnt                   = gnbConfig.get_value_as<uint16_t>("nRx");
        cellStatPrmVec.nPrbUlBwp                = gnbConfig.get_value_as<uint16_t>("nPrb");
        cellStatPrmVec.nPrbDlBwp                = gnbConfig.get_value_as<uint16_t>("nPrb");
        cellStatPrmVec.mu                       = gnbConfig.get_value_as<uint8_t>("mu");
#endif
        // debug parameters
        cuphy::cuphyHDF5_struct debug_pars = cuphy::get_HDF5_struct(fInput, "debug_pars");

        dbgPrm.forcedNumCsi2Bits = debug_pars.get_value_as<uint16_t>("forcedNumCsi2Bits");
        bOutputFileName          = outputFileName;
        dbgPrm.pOutFileName      = bOutputFileName.empty() ? nullptr : bOutputFileName.c_str();
        dbgPrm.descrmOn          = static_cast<uint8_t>(descramblingOn);

        // static pusch parameters
        tWFreq      = cuphy::tensor_from_dataset(fInput.open_dataset("WFreq"), CUPHY_R_32F, cuphy::tensor_flags::align_tight, cuStrm);
        tWFreq4     = cuphy::tensor_from_dataset(fInput.open_dataset("WFreq4"), CUPHY_R_32F, cuphy::tensor_flags::align_tight, cuStrm);
        tWFreqSmall = cuphy::tensor_from_dataset(fInput.open_dataset("WFreqSmall"), CUPHY_R_32F, cuphy::tensor_flags::align_tight, cuStrm);

        tShiftSeq    = cuphy::tensor_from_dataset(fInput.open_dataset("ShiftSeq"), CUPHY_C_16F, cuphy::tensor_flags::align_tight, cuStrm);
        tUnShiftSeq  = cuphy::tensor_from_dataset(fInput.open_dataset("UnShiftSeq"), CUPHY_C_16F, cuphy::tensor_flags::align_tight, cuStrm);
        tShiftSeq4   = cuphy::tensor_from_dataset(fInput.open_dataset("ShiftSeq4"), CUPHY_C_16F, cuphy::tensor_flags::align_tight, cuStrm);
        tUnShiftSeq4 = cuphy::tensor_from_dataset(fInput.open_dataset("UnShiftSeq4"), CUPHY_C_16F, cuphy::tensor_flags::align_tight, cuStrm);

        cudaStreamSynchronize(cuStrm);

        tPrmWFreq.desc       = tWFreq.desc().handle();
        tPrmWFreq.pAddr      = tWFreq.addr();
        puschStatPrms.pWFreq = &tPrmWFreq;

        tPrmWFreq4.desc       = tWFreq4.desc().handle();
        tPrmWFreq4.pAddr      = tWFreq4.addr();
        puschStatPrms.pWFreq4 = &tPrmWFreq4;

        tPrmWFreqSmall.desc       = tWFreqSmall.desc().handle();
        tPrmWFreqSmall.pAddr      = tWFreqSmall.addr();
        puschStatPrms.pWFreqSmall = &tPrmWFreqSmall;

        tPrmShiftSeq.desc       = tShiftSeq.desc().handle();
        tPrmShiftSeq.pAddr      = tShiftSeq.addr();
        puschStatPrms.pShiftSeq = &tPrmShiftSeq;

        tPrmShiftSeq4.desc       = tShiftSeq4.desc().handle();
        tPrmShiftSeq4.pAddr      = tShiftSeq4.addr();
        puschStatPrms.pShiftSeq4 = &tPrmShiftSeq4;

        tPrmUnShiftSeq.desc       = tUnShiftSeq.desc().handle();
        tPrmUnShiftSeq.pAddr      = tUnShiftSeq.addr();
        puschStatPrms.pUnShiftSeq = &tPrmUnShiftSeq;

        tPrmUnShiftSeq4.desc       = tUnShiftSeq4.desc().handle();
        tPrmUnShiftSeq4.pAddr      = tUnShiftSeq4.addr();
        puschStatPrms.pUnShiftSeq4 = &tPrmUnShiftSeq4;

        puschStatPrms.polarDcdrListSz      = gnbConfig.get_value_as<uint8_t>("listLength");
        puschStatPrms.enableCfoCorrection  = gnbConfig.get_value_as<uint8_t>("enableCfoCorrection");
        puschStatPrms.enableWeightedAverageCfo  = gnbConfig.get_value_as<uint8_t>("enableWeightedAverageCfo");
        puschStatPrms.enableToEstimation   = gnbConfig.get_value_as<uint8_t>("enableToEstimation");
        puschStatPrms.enablePuschTdi       = gnbConfig.get_value_as<uint8_t>("TdiMode") == 1;
        puschStatPrms.enableDftSOfdm       = gnbConfig.get_value_as<uint8_t>("enableDftSOfdm");
        puschStatPrms.enableTbSizeCheck    = 1;
        puschStatPrms.enableMassiveMIMO    = gnbConfig.get_value_as<uint8_t>("enable_static_dynamic_beamforming");
        puschStatPrms.enableEarlyHarq      = gnbConfig.get_value_as<uint8_t>("enableEarlyHarq");
        puschStatPrms.ldpcEarlyTermination = gnbConfig.get_value_as<uint8_t>("ldpcEarlyTermination");
        puschStatPrms.ldpcAlgoIndex        = gnbConfig.get_value_as<uint16_t>("ldpcAlgoIndex");
        puschStatPrms.ldpcFlags            = gnbConfig.get_value_as<uint32_t>("ldpcFlags");
        puschStatPrms.ldpcKernelLaunch     = ldpcLaunchMode; //it could be changed to be read from h5 file instead
        uint8_t ldpcMaxNumItrAlgoIdx       = gnbConfig.get_value_as<uint8_t>("ldpcMaxNumItrAlgIdx");
        switch(ldpcMaxNumItrAlgoIdx){
            case 0:
            puschStatPrms.ldpcMaxNumItrAlgo = LDPC_MAX_NUM_ITR_ALGO_TYPE_FIXED;
            break;

            case 1:
            puschStatPrms.ldpcMaxNumItrAlgo = LDPC_MAX_NUM_ITR_ALGO_TYPE_LUT;
            break;

            case 2:
            puschStatPrms.ldpcMaxNumItrAlgo = LDPC_MAX_NUM_ITR_ALGO_TYPE_PER_UE;
            break;
        }
        puschStatPrms.fixedMaxNumLdpcItrs = gnbConfig.get_value_as<uint8_t>("ldpcMaxNumItr");

        if(enableLdpcThroughputMode)
        {
            // For throughput mode to be set, the algorithm index needs to be 0 and CUPHY_LDPC_DECODE_CHOOSE_THROUGHPUT needs to be set
#ifdef ENABLE_PUSCH_LDPC_OUTPUT_H5_DUMP
            puschStatPrms.ldpcFlags     = CUPHY_LDPC_DECODE_WRITE_SOFT_OUTPUTS;
#else
            puschStatPrms.ldpcFlags     = CUPHY_LDPC_DECODE_CHOOSE_THROUGHPUT;
#endif
            puschStatPrms.ldpcAlgoIndex = 0;
            NVLOGC_FMT(NVLOG_PUSCH, "LDPC throughput mode enabled");
        }
        else
        {
            puschStatPrms.ldpcFlags = 0;
            NVLOGC_FMT(NVLOG_PUSCH, "LDPC throughput mode disabled");
        }

        if(getenv("LDPC_USE_HALF"))
        {
            puschStatPrms.ldpcUseHalf = 1;
        }
        else
        {
            puschStatPrms.ldpcUseHalf = gnbConfig.get_value_as<uint8_t>("ldpcUseHalf");
        }
        puschStatPrms.ldpcClampValue = 32.0f;
        puschStatPrms.enableRssiMeasurement = gnbConfig.get_value_as<uint8_t>("enableRssiMeasurement");
        puschStatPrms.enableSinrMeasurement = gnbConfig.get_value_as<uint8_t>("enableSinrMeasurement");
        puschStatPrms.enableCsiP2Fapiv3     = gnbConfig.get_value_as<uint8_t>("enableCsiP2Fapiv3");
        uint8_t eqCoeffAlgoIdx = gnbConfig.get_value_as<uint8_t>("eqCoeffAlgoIdx");
        switch(eqCoeffAlgoIdx){
            case 0:
            puschStatPrms.eqCoeffAlgo = PUSCH_EQ_ALGO_TYPE_RZF;
            break;

            case 1:
            puschStatPrms.eqCoeffAlgo = PUSCH_EQ_ALGO_TYPE_NOISE_DIAG_MMSE;
            break;

            case 2:
            puschStatPrms.eqCoeffAlgo = PUSCH_EQ_ALGO_TYPE_MMSE_IRC;
            break;

            case 3:
            puschStatPrms.eqCoeffAlgo = PUSCH_EQ_ALGO_TYPE_MMSE_IRC_SHRINK_RBLW;
            break;

            case 4:
            puschStatPrms.eqCoeffAlgo = PUSCH_EQ_ALGO_TYPE_MMSE_IRC_SHRINK_OAS;
            break;
        }
        uint8_t chEstAlgoIdx = gnbConfig.get_value_as<uint8_t>("dmrsChEstAlgIdx");
        switch(chEstAlgoIdx){
            case 0:
            puschStatPrms.chEstAlgo = PUSCH_CH_EST_ALGO_TYPE_LEGACY_MMSE;
            break;

            case 1:
            puschStatPrms.chEstAlgo = PUSCH_CH_EST_ALGO_TYPE_MULTISTAGE_MMSE_WITH_DELAY_EST;
            break;

            case 2:
            puschStatPrms.chEstAlgo = PUSCH_CH_EST_ALGO_TYPE_RKHS;
            break;
        }
        
        puschStatPrms.enablePerPrgChEst= gnbConfig.get_value_as<uint8_t>("enablePerPrgChEst");
                
        //printf("file[%s]enablePerPrgChEst[%d]\n", inputFileNameVec[0].c_str(), puschStatPrms.enablePerPrgChEst);

        puschStatPrms.nMaxCells        = inputFileNameVec.size();
        puschStatPrms.nMaxCellsPerSlot = puschPrms ? puschPrms->maxNCellsPerSlot : puschStatPrms.nMaxCells;
        puschStatPrms.pCellStatPrms    = &cellStatPrmVec[0];
        puschStatPrms.pDbg             = &dbgPrm;
        puschStatPrms.nMaxCells        = cellStatPrmVec.size();

        // CSI-P2 FAPIv3 API paramaters
        puschStatPrms.enableCsiP2Fapiv3 = gnbConfig.get_value_as<uint8_t>("enableCsiP2Fapiv3");

        // RKHS paramaters
        if(puschStatPrms.chEstAlgo == PUSCH_CH_EST_ALGO_TYPE_RKHS)
        {
            puschStatPrms.pPuschRkhsPrms        = &puschRkhsPrms;
            cuphy::cuphyHDF5_struct rkhsPrmsCfg = cuphy::get_HDF5_struct(fInput, "rkhsPrms");
            puschRkhsPrms.nZpSizes              = rkhsPrmsCfg.get_value_as<uint8_t>("nZpSizes");
            puschRkhsPrms.nPrbSizes             = rkhsPrmsCfg.get_value_as<uint16_t>("nPrbSizes");

            hdf5hpp::hdf5_dataset prbPrmsRkhsDset = fInput.open_dataset("prbRksPrms");
            cpuBufferRkhsPrbPrms                  = std::move(cuphy::buffer<rkhsPrbPrms_t, cuphy::pinned_alloc>(puschRkhsPrms.nPrbSizes));
            puschRkhsPrms.pPerPrbRkhsPrms         = cpuBufferRkhsPrbPrms.addr();
            puschStatPrms.pPuschRkhsPrms          = &puschRkhsPrms;

            for(int i = 0; i < puschRkhsPrms.nPrbSizes; ++i)
            {
                cuphy::cuphyHDF5_struct prbRkhsPrmsStruct = cuphy::get_HDF5_struct_index(prbPrmsRkhsDset, i);
                rkhsPrbPrms_t& prbPrms = puschRkhsPrms.pPerPrbRkhsPrms[i];

                prbPrms.zpIdx               = prbRkhsPrmsStruct.get_value_as<uint8_t>("zpIdx");
                prbPrms.sumEigValues        = static_cast<__half>(prbRkhsPrmsStruct.get_value_as<float>("sumEigValues"));

                std::string str1 = "corr_half_nZpDmrsSc";
                std::string str2 = std::to_string (i + 1);
                std::string str3 = str1 + str2;
                corr_half_nZpDmrsScTable[i]       =  cuphy::tensor_from_dataset(fInput.open_dataset(str3.c_str()), CUPHY_C_16F, cuphy::tensor_flags::align_tight, cuStrm);
                prbPrms.corr_half_nZpDmrsSc.desc  = corr_half_nZpDmrsScTable[i].desc().handle();
                prbPrms.corr_half_nZpDmrsSc.pAddr = corr_half_nZpDmrsScTable[i].addr();

                str1 = "eigVecCob";
                str3 = str1 + str2;
                eigVecCobTable[i]       = cuphy::tensor_from_dataset(fInput.open_dataset(str3.c_str()), CUPHY_R_16F, cuphy::tensor_flags::align_tight, cuStrm);
                prbPrms.eigVecCob.desc  = eigVecCobTable[i] .desc().handle();
                prbPrms.eigVecCob.pAddr = eigVecCobTable[i].addr();

                str1 = "corr";
                str3 = str1 + str2;
                corrTable[i]       = cuphy::tensor_from_dataset(fInput.open_dataset(str3.c_str()), CUPHY_C_16F, cuphy::tensor_flags::align_tight, cuStrm);
                prbPrms.corr.desc  = corrTable[i] .desc().handle();
                prbPrms.corr.pAddr = corrTable[i].addr();

                str1 = "eigVal";
                str3 = str1 + str2;
                eigValTable[i]       =  cuphy::tensor_from_dataset(fInput.open_dataset(str3.c_str()), CUPHY_R_16F, cuphy::tensor_flags::align_tight, cuStrm);
                prbPrms.eigVal.desc  = eigValTable[i] .desc().handle();
                prbPrms.eigVal.pAddr = eigValTable[i].addr();

                str1 = "interpCob";
                str3 = str1 + str2;
                interpCobTable[i]       =  cuphy::tensor_from_dataset(fInput.open_dataset(str3.c_str()), CUPHY_R_16F, cuphy::tensor_flags::align_tight, cuStrm);
                prbPrms.interpCob.desc  = interpCobTable[i] .desc().handle();
                prbPrms.interpCob.pAddr = interpCobTable[i].addr();
            }

            hdf5hpp::hdf5_dataset zpPrmsRkhsDset  = fInput.open_dataset("zpRksPrms");
            cpuBufferRkhsZpPrms                   = std::move(cuphy::buffer<rkhsZpPrms_t, cuphy::pinned_alloc>(puschRkhsPrms.nZpSizes));
            puschRkhsPrms.pPerZpRkhsPrms          = cpuBufferRkhsZpPrms.addr();

            for(int i = 0; i < puschRkhsPrms.nZpSizes; ++i)
            {
                cuphy::cuphyHDF5_struct zpRkhsPrmsStruct = cuphy::get_HDF5_struct_index(prbPrmsRkhsDset, i);
                rkhsZpPrms_t& zpPrms                     = puschRkhsPrms.pPerZpRkhsPrms[i];

                // zpPrms.nZpDmrsSc              = zpRkhsPrmsStruct.get_value_as<uint16_t>("nZpDmrsSc");
                // zpPrms.secondStageFourierSize = zpRkhsPrmsStruct.get_value_as<uint8_t>("secondStageFourierSize");

                std::string str1 = "zpInterpVec";
                std::string str2 = std::to_string(i);
                std::string str3 = str1 + str2;
                zpInterpVecTable[i]           = cuphy::tensor_from_dataset(fInput.open_dataset(str3.c_str()), CUPHY_R_16F, cuphy::tensor_flags::align_tight, cuStrm);
                zpPrms.zpInterpVec.desc       = zpInterpVecTable[i].desc().handle();
                zpPrms.zpInterpVec.pAddr      = zpInterpVecTable[i].addr();

                str1 = "zpDmrsScEigenVec";
                str3 = str1 + str2;
                zpDmrsScEigenVecTable[i]           = cuphy::tensor_from_dataset(fInput.open_dataset(str3.c_str()), CUPHY_R_16F, cuphy::tensor_flags::align_tight, cuStrm);
                zpPrms.zpDmrsScEigenVec.desc       = zpDmrsScEigenVecTable[i].desc().handle();
                zpPrms.zpDmrsScEigenVec.pAddr      = zpDmrsScEigenVecTable[i].addr();

                str1 = "zpSecondStageTwiddleFactors";
                str3 = str1 + str2;
                secondStageTwiddleFactors[i]           = cuphy::tensor_from_dataset(fInput.open_dataset(str3.c_str()), CUPHY_C_16F, cuphy::tensor_flags::align_tight, cuStrm);
                zpPrms.secondStageTwiddleFactors.desc  = secondStageTwiddleFactors[i].desc().handle();
                zpPrms.secondStageTwiddleFactors.pAddr = secondStageTwiddleFactors[i].addr();

                str1 = "zpSecondStageFourierPerm";
                str3 = str1 + str2;
                secondStageFourierPerm[i]           = cuphy::tensor_from_dataset(fInput.open_dataset(str3.c_str()), CUPHY_R_8U, cuphy::tensor_flags::align_tight, cuStrm);
                zpPrms.secondStageFourierPerm.desc  = secondStageFourierPerm[i].desc().handle();
                zpPrms.secondStageFourierPerm.pAddr = secondStageFourierPerm[i].addr();
            }
        }else
        {
            puschStatPrms.pPuschRkhsPrms = NULL;
        }
    }
    else
    {
        // static cell parameters
        cuphy::cuphyHDF5_struct cellStatic = cuphy::get_HDF5_struct(fInput, "cellStat_pars");
#if 0
        cellStatPrmVec.phyCellId              = cellStatic.get_value_as<uint16_t>("phyCellId");
        cellStatPrmVec.nRxAnt                 = cellStatic.get_value_as<uint16_t>("nRxAnt");
        cellStatPrmVec.nTxAnt                 = cellStatic.get_value_as<uint16_t>("nTxAnt");
        cellStatPrmVec.nPrbUlBwp              = cellStatic.get_value_as<uint16_t>("nPrbUlBwp");
        cellStatPrmVec.nPrbDlBwp              = cellStatic.get_value_as<uint16_t>("nPrbDlBwp");
        cellStatPrmVec.mu                     = cellStatic.get_value_as<uint8_t>("mu");
#endif
        // debug parameters
        bOutputFileName         = outputFileName;
        dbgPrm.pOutFileName     = bOutputFileName.empty() ? nullptr : bOutputFileName.c_str();
        dbgPrm.descrmOn         = static_cast<uint8_t>(descramblingOn);

        // static pusch parameters
        cuphy::cuphyHDF5_struct puschStat_pars = cuphy::get_HDF5_struct(fInput, "puschStat_pars");
        tWFreq                                 = cuphy::tensor_from_dataset(fInput.open_dataset("WFreq"), CUPHY_R_32F, cuphy::tensor_flags::align_tight, cuStrm);
        tShiftSeq                              = cuphy::tensor_from_dataset(fInput.open_dataset("ShiftSeq"), CUPHY_C_16F, cuphy::tensor_flags::align_tight, cuStrm);
        tUnShiftSeq                            = cuphy::tensor_from_dataset(fInput.open_dataset("UnShiftSeq"), CUPHY_C_16F, cuphy::tensor_flags::align_tight, cuStrm);
        cudaStreamSynchronize(cuStrm);

        tPrmWFreq.desc       = tWFreq.desc().handle();
        tPrmWFreq.pAddr      = tWFreq.addr();
        puschStatPrms.pWFreq = &tPrmWFreq;

        tPrmShiftSeq.desc       = tShiftSeq.desc().handle();
        tPrmShiftSeq.pAddr      = tShiftSeq.addr();
        puschStatPrms.pShiftSeq = &tPrmShiftSeq;

        tPrmUnShiftSeq.desc       = tUnShiftSeq.desc().handle();
        tPrmUnShiftSeq.pAddr      = tUnShiftSeq.addr();
        puschStatPrms.pUnShiftSeq = &tPrmUnShiftSeq;

        puschStatPrms.enableCfoCorrection  = puschStat_pars.get_value_as<uint8_t>("enableCfoCorrection");
        puschStatPrms.enableWeightedAverageCfo  = puschStat_pars.get_value_as<uint8_t>("enableWeightedAverageCfo");
        puschStatPrms.enableToEstimation   = puschStat_pars.get_value_as<uint8_t>("enableToEstimation");
        puschStatPrms.enablePuschTdi       = puschStat_pars.get_value_as<uint8_t>("TdiMode") == 1;
        puschStatPrms.enableDftSOfdm       = puschStat_pars.get_value_as<uint8_t>("enableDftSOfdm");
        puschStatPrms.enableTbSizeCheck    = 1;
        puschStatPrms.enableMassiveMIMO    = puschStat_pars.get_value_as<uint8_t>("enable_static_dynamic_beamforming");
        puschStatPrms.ldpcEarlyTermination = puschStat_pars.get_value_as<uint8_t>("ldpcEarlyTermination");
        puschStatPrms.ldpcAlgoIndex        = puschStat_pars.get_value_as<uint16_t>("ldpcAlgoIndex");
        puschStatPrms.ldpcFlags            = puschStat_pars.get_value_as<uint32_t>("ldpcFlags");
        puschStatPrms.ldpcKernelLaunch     = ldpcLaunchMode;
        if(getenv("LDPC_USE_HALF"))
        {
            puschStatPrms.ldpcUseHalf = 1;
        }
        else
        {
            puschStatPrms.ldpcUseHalf = puschStat_pars.get_value_as<uint8_t>("ldpcUseHalf");
        }
        puschStatPrms.ldpcClampValue  = 32.0f;
        puschStatPrms.enableRssiMeasurement = puschStat_pars.get_value_as<uint8_t>("enableRssiMeasurement");
        puschStatPrms.enableSinrMeasurement = puschStat_pars.get_value_as<uint8_t>("enableSinrMeas");

        puschStatPrms.nMaxCells        = inputFileNameVec.size();
        puschStatPrms.nMaxCellsPerSlot = puschPrms ? puschPrms->maxNCellsPerSlot : puschStatPrms.nMaxCells;
        puschStatPrms.pCellStatPrms    = &cellStatPrmVec[0];
        puschStatPrms.pDbg             = &dbgPrm;
        puschStatPrms.nMaxCells        = cellStatPrmVec.size();
    }
}

// default constructor
StaticApiDataset::StaticApiDataset() :
    puschStatPrms{},
    dbgPrm{},
    cellStatPrmVec{},
    tWFreq{},
    tWFreq4{},
    tWFreqSmall{},
    tShiftSeq{},
    tUnShiftSeq{},
    tShiftSeq4{},
    tUnShiftSeq4{},
    tPrmWFreq{},
    tPrmWFreq4{},
    tPrmWFreqSmall{},
    tPrmShiftSeq{},
    tPrmUnShiftSeq{},
    tPrmShiftSeq4{},
    tPrmUnShiftSeq4{},
    bOutputFileName{}
{}

// Reset pointers after a move or copy
void StaticApiDataset::ResetPointers()
{
    tPrmWFreq.pAddr = tWFreq.addr();
    tPrmWFreq.desc  = tWFreq.desc().handle();

    tPrmWFreq4.pAddr = tWFreq4.addr();
    tPrmWFreq4.desc  = tWFreq4.desc().handle();

    tPrmWFreqSmall.pAddr = tWFreqSmall.addr();
    tPrmWFreqSmall.desc  = tWFreqSmall.desc().handle();

    tPrmShiftSeq.pAddr = tShiftSeq.addr();
    tPrmShiftSeq.desc  = tShiftSeq.desc().handle();

    tPrmShiftSeq4.pAddr = tShiftSeq4.addr();
    tPrmShiftSeq4.desc  = tShiftSeq4.desc().handle();

    tPrmUnShiftSeq.pAddr = tUnShiftSeq.addr();
    tPrmUnShiftSeq.desc  = tUnShiftSeq.desc().handle();

    tPrmUnShiftSeq4.pAddr = tUnShiftSeq4.addr();
    tPrmUnShiftSeq4.desc  = tUnShiftSeq4.desc().handle();

    puschStatPrms.pWFreq        = &tPrmWFreq;
    puschStatPrms.pWFreq4       = &tPrmWFreq4;
    puschStatPrms.pWFreqSmall   = &tPrmWFreqSmall;
    puschStatPrms.pShiftSeq     = &tPrmShiftSeq;
    puschStatPrms.pShiftSeq4    = &tPrmShiftSeq4;
    puschStatPrms.pUnShiftSeq   = &tPrmUnShiftSeq;
    puschStatPrms.pUnShiftSeq4  = &tPrmUnShiftSeq4;
    puschStatPrms.pCellStatPrms = &cellStatPrmVec[0];
    puschStatPrms.pDbg          = &dbgPrm;

    dbgPrm.pOutFileName = bOutputFileName.empty() ? nullptr : bOutputFileName.c_str();
}

// copy constructor
StaticApiDataset::StaticApiDataset(const StaticApiDataset& staticApiDataset) :
    puschStatPrms(staticApiDataset.puschStatPrms),
    dbgPrm(staticApiDataset.dbgPrm),
    cellStatPrmVec(staticApiDataset.cellStatPrmVec),
    tWFreq(staticApiDataset.tWFreq),
    tWFreq4(staticApiDataset.tWFreq4),
    tWFreqSmall(staticApiDataset.tWFreqSmall),
    tShiftSeq(staticApiDataset.tShiftSeq),
    tUnShiftSeq(staticApiDataset.tUnShiftSeq),
    tShiftSeq4(staticApiDataset.tShiftSeq4),
    tUnShiftSeq4(staticApiDataset.tUnShiftSeq4),
    tPrmWFreq(staticApiDataset.tPrmWFreq),
    tPrmWFreq4(staticApiDataset.tPrmWFreq4),
    tPrmWFreqSmall(staticApiDataset.tPrmWFreqSmall),
    tPrmShiftSeq(staticApiDataset.tPrmShiftSeq),
    tPrmUnShiftSeq(staticApiDataset.tPrmUnShiftSeq),
    tPrmShiftSeq4(staticApiDataset.tPrmShiftSeq4),
    tPrmUnShiftSeq4(staticApiDataset.tPrmUnShiftSeq4),
    bOutputFileName(staticApiDataset.bOutputFileName)
{
    // synch default stream, used to copy
    cudaStreamSynchronize(0);

    // update pointers
    ResetPointers();
}

// move operator
StaticApiDataset& StaticApiDataset::operator=(StaticApiDataset&& staticApiDataset)
{
    puschStatPrms   = std::move(staticApiDataset.puschStatPrms);
    dbgPrm          = std::move(staticApiDataset.dbgPrm);
    cellStatPrmVec  = std::move(staticApiDataset.cellStatPrmVec);
    tWFreq          = std::move(staticApiDataset.tWFreq);
    tWFreq4         = std::move(staticApiDataset.tWFreq4);
    tWFreqSmall     = std::move(staticApiDataset.tWFreqSmall);
    tShiftSeq       = std::move(staticApiDataset.tShiftSeq);
    tShiftSeq4      = std::move(staticApiDataset.tShiftSeq4);
    tUnShiftSeq     = std::move(staticApiDataset.tUnShiftSeq);
    tUnShiftSeq4    = std::move(staticApiDataset.tUnShiftSeq4);
    tPrmWFreq       = std::move(staticApiDataset.tPrmWFreq);
    tPrmWFreq4      = std::move(staticApiDataset.tPrmWFreq4);
    tPrmWFreqSmall  = std::move(staticApiDataset.tPrmWFreqSmall);
    tPrmShiftSeq    = std::move(staticApiDataset.tPrmShiftSeq);
    tPrmShiftSeq4   = std::move(staticApiDataset.tPrmShiftSeq4);
    tPrmUnShiftSeq  = std::move(staticApiDataset.tPrmUnShiftSeq);
    tPrmUnShiftSeq4 = std::move(staticApiDataset.tPrmUnShiftSeq4);
    bOutputFileName = std::move(staticApiDataset.bOutputFileName);

    ResetPointers();
    return *this;
}

//-------------------------------------------------------------------------------------
// Dataset holds parameters/data used to evaluate bler

// construct from h5 file

EvalDataset::EvalDataset(const std::vector<std::string>& inputFileNameVec, cudaStream_t cuStrm, int apiTVflag, bool drmDebug)
{
    m_drmDebug = drmDebug;
    nTbs = nCbs = 0;

    tRefUciPayloadBytesVec.resize(inputFileNameVec.size());
    tRefUciCrcFlagsVec.resize(inputFileNameVec.size());
    tRefUciDTXsVec.resize(inputFileNameVec.size());
    tRefUciHarqDetStatusVec.resize(inputFileNameVec.size());
    tRefUciCsi1DetStatusVec.resize(inputFileNameVec.size());
    tRefUciCsi2DetStatusVec.resize(inputFileNameVec.size());

    tTrueTbBytesVec.resize(inputFileNameVec.size());
    tTrueTbCrcErrVec.resize(inputFileNameVec.size());
    nBytesVec.resize(inputFileNameVec.size());
    nTbsInFileVec.resize(inputFileNameVec.size());
    uint32_t globalTbIdx = 0;

    nUes                    = 0;
    nSchUes                 = 0;
    uint16_t nHarqUcis      = 0;
    uint16_t nCsi1Ucis      = 0;
    nCsi2Ues                = 0;
    uint16_t nSchAndUciUes  = 0;
    uint16_t globalUeGrpIdx = 0;

    for(uint32_t i = 0; i < inputFileNameVec.size(); i++)
    {
        // load 5h file
        uint16_t nSchAndUciUesFile  = 0;
        uint16_t nHarqUcisFile      = 0;
        uint16_t nCsi1UcisFile      = 0;
        uint16_t nCsi2UesFile       = 0;
        bool     IsUciUe            = false;
        bool     IsSchUe            = false;

        hdf5hpp::hdf5_file fInput = hdf5hpp::hdf5_file::open(inputFileNameVec[i].c_str());
        nBytesVec[i]              = 0;
        if(apiTVflag == 0) // legacy test-vector
        {
            cuphy::cuphyHDF5_struct gnbConfig          = cuphy::get_HDF5_struct(fInput, "gnb_pars");
            hdf5hpp::hdf5_dataset   tbpDset            = fInput.open_dataset("tb_pars");
            hdf5hpp::hdf5_dataset   uciRmSizesDset     = fInput.open_dataset("reference_uci_pars");
            hdf5hpp::hdf5_dataset   ueRefBuffOffsetDst = fInput.open_dataset("ue_refBufferOffsets");

            // compute sizes
            uint32_t nTbsOld = nTbs;
            nTbsInFileVec[i] = gnbConfig.get_value_as<uint32_t>("numTb");
            nTbs += gnbConfig.get_value_as<uint32_t>("numTb");
            nUes += gnbConfig.get_value_as<uint32_t>("numTb");
            nUeGrps = gnbConfig.get_value_as<uint32_t>("nUserGroups");

            ueRefBuffOffsetsVec.resize(nTbs);
            perTbPrmsRef.resize(nTbs);
            nBytesPerCbVec.resize(nTbs);
            nCbsPerTbVec.resize(nTbs);
            csi2UeIdxsVec.resize(nTbs);
            uciSizesVec.resize(nTbs);
            tTrueCbCrcErrVec.resize(nTbs);

            for(uint32_t tbIdx = 0; tbIdx < nTbsInFileVec[i]; ++tbIdx, globalTbIdx++)
            {
                cuphy::cuphyHDF5_struct tbConfig          = cuphy::get_HDF5_struct_index(tbpDset, tbIdx);
                cuphy::cuphyHDF5_struct uciRmSizes        = cuphy::get_HDF5_struct_index(uciRmSizesDset, tbIdx);
                cuphy::cuphyHDF5_struct ueRefBuffOffsetH5 = cuphy::get_HDF5_struct_index(ueRefBuffOffsetDst, tbIdx);

                ueRefBuffOffsetsVec[globalTbIdx].harqPayloadByteOffset = ueRefBuffOffsetH5.get_value_as<uint32_t>("harqPayloadByteOffset");
                ueRefBuffOffsetsVec[globalTbIdx].nHarqBytes            = ueRefBuffOffsetH5.get_value_as<uint32_t>("nHarqBytes");
                ueRefBuffOffsetsVec[globalTbIdx].harqCrcFlagOffset     = ueRefBuffOffsetH5.get_value_as<uint32_t>("harqCrcFlagOffset");
                ueRefBuffOffsetsVec[globalTbIdx].csi1PayloadByteOffset = ueRefBuffOffsetH5.get_value_as<uint32_t>("csi1PayloadByteOffset");
                ueRefBuffOffsetsVec[globalTbIdx].nCsi1Bytes            = ueRefBuffOffsetH5.get_value_as<uint32_t>("nCsi1Bytes");
                ueRefBuffOffsetsVec[globalTbIdx].csi1CrcFlagOffset     = ueRefBuffOffsetH5.get_value_as<uint32_t>("csi1CrcFlagOffset");
                ueRefBuffOffsetsVec[globalTbIdx].csi2PayloadByteOffset = ueRefBuffOffsetH5.get_value_as<uint32_t>("csi2PayloadByteOffset");
                ueRefBuffOffsetsVec[globalTbIdx].nCsi2Bytes            = ueRefBuffOffsetH5.get_value_as<uint32_t>("nCsi2Bytes");
                ueRefBuffOffsetsVec[globalTbIdx].csi2CrcFlagOffset     = ueRefBuffOffsetH5.get_value_as<uint32_t>("csi2CrcFlagOffset");

                nCbsPerTbVec[globalTbIdx]   = tbConfig.get_value_as<uint32_t>("nCb");
                nBytesPerCbVec[globalTbIdx] = tbConfig.get_value_as<uint32_t>("nTbByte") / tbConfig.get_value_as<uint32_t>("nCb");

                nCbs += tbConfig.get_value_as<uint32_t>("nCb");
                nBytesVec[i] += tbConfig.get_value_as<uint32_t>("nTbByte");

                perTbPrmsRef[globalTbIdx].G          = uciRmSizes.get_value_as<uint32_t>("G");
                perTbPrmsRef[globalTbIdx].G_harq     = uciRmSizes.get_value_as<uint32_t>("G_harq");
                perTbPrmsRef[globalTbIdx].G_csi1     = uciRmSizes.get_value_as<uint32_t>("G_csi1");
                perTbPrmsRef[globalTbIdx].G_harq_rvd = uciRmSizes.get_value_as<uint32_t>("G_harq_rvd");

                uciSizesVec[globalTbIdx].G         = uciRmSizes.get_value_as<uint32_t>("G");
                uciSizesVec[globalTbIdx].G_csi2    = uciRmSizes.get_value_as<uint32_t>("G_csi2");
                uciSizesVec[globalTbIdx].nBitsCsi2 = uciRmSizes.get_value_as<uint16_t>("nBitsCsi2");


                uint16_t pduBitmap = tbConfig.get_value_as<uint16_t>("pduBitmap");
                if(pduBitmap & 1) // check 0th bit for sch transmission
                {
                    IsSchUe = true;
                    std::string inputCbCrcErrName = "cbErr"+std::to_string(tbIdx);
                    tTrueCbCrcErrVec[globalTbIdx] = cuphy::tensor_from_dataset(fInput.open_dataset(inputCbCrcErrName.c_str()), CUPHY_R_32U, cuphy::tensor_flags::align_tight, cuStrm);
                }
                if(pduBitmap & 2) // check 1st bit for uci transmission
                {
                    IsUciUe = true;
                    if(pduBitmap & 1) // check 0th bit for sch transmission
                    {
                        std::string          datasetName = "reference_schLLRs" + std::to_string(nSchAndUciUesFile);
                        cuphy::tensor_pinned schLLRs     = cuphy::tensor_from_dataset(fInput.open_dataset(datasetName.c_str()), CUPHY_R_16F, cuphy::tensor_flags::align_tight, cuStrm);

                        schLLRsRef.emplace_back(schLLRs.layout());
                        schLLRsRef[nSchAndUciUes] = schLLRs;
                        nSchAndUciUes++;
                        nSchAndUciUesFile++;
                    }
                    if(perTbPrmsRef[globalTbIdx].G_harq > 0)
                    {
                        std::string          datasetName = "reference_harqLLRs" + std::to_string(nHarqUcisFile);
                        cuphy::tensor_pinned harqLLRs    = cuphy::tensor_from_dataset(fInput.open_dataset(datasetName.c_str()), CUPHY_R_16F, cuphy::tensor_flags::align_tight, cuStrm);

                        harqLLRsRef.emplace_back(harqLLRs.layout());
                        harqLLRsRef[nHarqUcis] = harqLLRs;
                        nHarqUcis++;
                        nHarqUcisFile++;
                    }
                    if(perTbPrmsRef[globalTbIdx].G_csi1 > 0)
                    {
                        std::string          datasetName = "reference_csi1LLRs" + std::to_string(nCsi1UcisFile);
                        cuphy::tensor_pinned csi1LLRs    = cuphy::tensor_from_dataset(fInput.open_dataset(datasetName.c_str()), CUPHY_R_16F, cuphy::tensor_flags::align_tight, cuStrm);

                        csi1LLRsRef.emplace_back(csi1LLRs.layout());
                        csi1LLRsRef[nCsi1Ucis] = csi1LLRs;
                        nCsi1Ucis++;
                        nCsi1UcisFile++;
                    }
                    if(uciSizesVec[globalTbIdx].G_csi2 > 0)
                    {
                        std::string          datasetName = "reference_csi2LLRs" + std::to_string(nCsi2UesFile);
                        cuphy::tensor_pinned csi2LLRs    = cuphy::tensor_from_dataset(fInput.open_dataset(datasetName.c_str()), CUPHY_R_16F, cuphy::tensor_flags::align_tight, cuStrm);

                        csi2LLRsRef.emplace_back(csi2LLRs.layout());
                        csi2LLRsRef[nCsi2Ues]   = csi2LLRs;
                        csi2UeIdxsVec[nCsi2Ues] = static_cast<uint16_t>(globalTbIdx);
                        nCsi2Ues++;
                        nCsi2UesFile++;
                    }
                }
            }

            for(int ueIdx = 0; ueIdx < nTbsInFileVec[i]; ++ueIdx)
            {
                cuphy::cuphyHDF5_struct tbConfig  = cuphy::get_HDF5_struct_index(tbpDset, ueIdx);
                uint16_t                pduBitmap = tbConfig.get_value_as<uint16_t>("pduBitmap");

                if(pduBitmap & 1)
                {
                    std::string          datasetName = "reference_rmOutLLRs" + std::to_string(ueIdx);
                    cuphy::tensor_pinned rmOutLLRs   = cuphy::tensor_from_dataset(fInput.open_dataset(datasetName.c_str()), CUPHY_R_16F, cuphy::tensor_flags::align_tight, cuStrm);
                    rmOutLLRsRef.emplace_back(rmOutLLRs.layout());
                    rmOutLLRsRef[nSchUes] = rmOutLLRs;
                    nSchUes += 1;
                }
            }

            // load uci reference buffers
            if(IsUciUe)
            {
                //tRefUciPayloadBytes = typed_tensor_from_dataset<CUPHY_R_8U, pinned_alloc>(fInput.open_dataset("reference_uciPayloads"), cuphy::tensor_flags::align_tight, cuStrm);  // only for cuPHY/examples/uciOnPusch_csi2_ctrl/cuphy_ex_uciOnPusch_csi2_ctrl.cpp:162:53
                tRefUciPayloadBytesVec[i] = cuphy::tensor_from_dataset(fInput.open_dataset("reference_uciPayloads"), cuphy::tensor_flags::align_tight, cuStrm);
                tRefUciCrcFlagsVec[i] = cuphy::tensor_from_dataset(fInput.open_dataset("reference_uciCrcFlags"), cuphy::tensor_flags::align_tight, cuStrm);
                tRefUciDTXsVec[i] = cuphy::tensor_from_dataset(fInput.open_dataset("reference_uciDTXs"), cuphy::tensor_flags::align_tight, cuStrm);
                tRefUciHarqDetStatusVec[i] = cuphy::tensor_from_dataset(fInput.open_dataset("reference_uciHarqDetStatus"), cuphy::tensor_flags::align_tight, cuStrm);
                tRefUciCsi1DetStatusVec[i] = cuphy::tensor_from_dataset(fInput.open_dataset("reference_uciCsi1DetStatus"), cuphy::tensor_flags::align_tight, cuStrm);
                tRefUciCsi2DetStatusVec[i] = cuphy::tensor_from_dataset(fInput.open_dataset("reference_uciCsi2DetStatus"), cuphy::tensor_flags::align_tight, cuStrm);
            }

            // load true Bytes
            if(IsSchUe)
            {
                tTrueTbBytesVec[i]  = cuphy::tensor_from_dataset(fInput.open_dataset("tb_data"), CUPHY_R_8U, cuphy::tensor_flags::align_tight, cuStrm);
                tTrueTbCrcErrVec[i] = cuphy::tensor_from_dataset(fInput.open_dataset("tbErr"), CUPHY_R_32U, cuphy::tensor_flags::align_tight, cuStrm);
            }

            for(int ueGrpIdx = 0; ueGrpIdx < nUeGrps; ++ueGrpIdx, ++globalUeGrpIdx)
            {
                // save equalizer output LLRs
                std::string          datasetName = "reference_eqOutLLRs" + std::to_string(ueGrpIdx);
                cuphy::tensor_pinned eqOutLLRs   = cuphy::tensor_from_dataset(fInput.open_dataset(datasetName.c_str()), CUPHY_R_16F, cuphy::tensor_flags::align_tight, cuStrm);

                eqOutLLRsRef.emplace_back(eqOutLLRs.layout());
                eqOutLLRsRef[globalUeGrpIdx] = eqOutLLRs;

                datasetName                  = "reference_rssi" + std::to_string(ueGrpIdx);
                cuphy::tensor_pinned refRssi = cuphy::tensor_from_dataset(fInput.open_dataset(datasetName.c_str()), cuphy::tensor_flags::align_tight, cuStrm);
                tRefRssi.emplace_back(refRssi.layout());
                tRefRssi[globalUeGrpIdx] = refRssi;
                
                datasetName                  = "reference_rssi_ehq" + std::to_string(ueGrpIdx);
                cuphy::tensor_pinned refRssiEhq = cuphy::tensor_from_dataset(fInput.open_dataset(datasetName.c_str()), cuphy::tensor_flags::align_tight, cuStrm);
                tRefRssiEhq.emplace_back(refRssiEhq.layout());
                tRefRssiEhq[globalUeGrpIdx] = refRssiEhq;

                datasetName                      = "reference_rssiFull" + std::to_string(ueGrpIdx);
                cuphy::tensor_pinned refRssiFull = cuphy::tensor_from_dataset(fInput.open_dataset(datasetName.c_str()), cuphy::tensor_flags::align_tight, cuStrm);
                tRefRssiFull.emplace_back(refRssiFull.layout());
                tRefRssiFull[globalUeGrpIdx] = refRssiFull;

                if(gnbConfig.get_value_as<uint8_t>("dmrsChEstAlgIdx") == PUSCH_CH_EST_ALGO_TYPE_MULTISTAGE_MMSE_WITH_DELAY_EST)
                {
                    datasetName                    = "reference_ChEst_w_delay_est_H_LS_est_save" + std::to_string(ueGrpIdx);
                    cuphy::tensor_pinned refChEstLS = cuphy::tensor_from_dataset(fInput.open_dataset(datasetName.c_str()), cuphy::tensor_flags::align_tight, cuStrm);
                    tRefChEstLSHest.emplace_back(refChEstLS.layout());
                    tRefChEstLSHest[globalUeGrpIdx] = refChEstLS;

                    datasetName                    = "reference_ChEst_w_delay_est_delay_mean" + std::to_string(ueGrpIdx);
                    cuphy::tensor_pinned refDelayMean = cuphy::tensor_from_dataset(fInput.open_dataset(datasetName.c_str()), cuphy::tensor_flags::align_tight, cuStrm);
                    tRefChEstDelayMean.emplace_back(refDelayMean.layout());
                    tRefChEstDelayMean[globalUeGrpIdx] = refDelayMean;
                }
                
                datasetName = "reference_H_est" + std::to_string(ueGrpIdx);
                cuphy::tensor_pinned Hest = cuphy::tensor_from_dataset(fInput.open_dataset(datasetName.c_str()), cuphy::tensor_flags::align_tight, cuStrm);
                HestRef.emplace_back(Hest.layout());
                HestRef[globalUeGrpIdx] = Hest;
                
                datasetName = "reference_Eq_coef" + std::to_string(ueGrpIdx);
                cuphy::tensor_pinned chEqCoef = cuphy::tensor_from_dataset(fInput.open_dataset(datasetName.c_str()), cuphy::tensor_flags::align_tight, cuStrm);
                chEqCoefRef.emplace_back(chEqCoef.layout());
                chEqCoefRef[globalUeGrpIdx] = chEqCoef;

                // @todo: CFO and TA estimates should be per UE but saved per UE group in the TV. Fix this.
                datasetName                    = "reference_cfoEst" + std::to_string(ueGrpIdx);
                cuphy::tensor_pinned refCfoEst = cuphy::tensor_from_dataset(fInput.open_dataset(datasetName.c_str()), cuphy::tensor_flags::align_tight, cuStrm);
                tRefCfoEst.emplace_back(refCfoEst.layout());
                tRefCfoEst[globalUeGrpIdx] = refCfoEst;

                // load reference TA estimate:
                datasetName                   = "reference_taEstSec" + std::to_string(ueGrpIdx);
                cuphy::tensor_pinned refTaEst = cuphy::tensor_from_dataset(fInput.open_dataset(datasetName.c_str()), cuphy::tensor_flags::align_tight, cuStrm);
                tRefTaEsts.emplace_back(refTaEst.layout());
                tRefTaEsts[globalUeGrpIdx] = refTaEst;

#if 0
            for(int32_t i = 0; i < tRefCfoEst[globalUeGrpIdx].desc().get_dim(0); ++i)
            {
                printf("ueGroup[%d] Symbol[%d] CFO Est %f+j%f\n", ueGrpIdx, i, tRefCfoEst[globalUeGrpIdx](i).x, tRefCfoEst[globalUeGrpIdx](i).y);
            }

            printf("ueGroup[%d] TA Est %f us\n", ueGrpIdx, tRefTaEsts[globalUeGrpIdx](0)*1000000); // seconds to microseconds

            // CFO frequency in Hz for debug
            float cfoEstHz = 0;
            try
            {
                cuphy::typed_tensor<CUPHY_R_32F, cuphy::pinned_alloc> tCfoEstHz;
                datasetName = "reference_cfoEstHz" + std::to_string(ueGrpIdx);
                tCfoEstHz = cuphy::typed_tensor_from_dataset<CUPHY_R_32F, cuphy::pinned_alloc>(fInput.open_dataset(datasetName.c_str()), cuphy::tensor_flags::align_tight, cuStrm);
                cfoEstHz  = tCfoEstHz(0);
            }
            catch(const cuphy::cuphyHDF5_exception& e)// catch(...)
            {
                cfoEstHz = 0;
                printf("WARNING: TV is missing CFO Est field for UE group %d.  Setting to 0Hz.\n", ueGrpIdx);
            }
            printf("ueGroup[%d] Reference CFO Est %07.4f Hz\n", ueGrpIdx, cfoEstHz);
#endif
            }

            // @todo: One tensor per metric for all the UEs in a cell. For cell groups these parameters (RSRP, NoiseVar, SINR) need to be "fused" before comparison with
            // the cuPHY values
            // @todo: Change these metrics to fp32 types in TV
            std::string          datasetName = "reference_rsrpdB";
            cuphy::tensor_pinned refRsrp     = cuphy::tensor_from_dataset(fInput.open_dataset(datasetName.c_str()), cuphy::tensor_flags::align_tight, cuStrm);
            tRefRsrp.emplace_back(refRsrp.layout());
            tRefRsrp[i] = refRsrp;
            
            datasetName = "reference_rsrpdB_ehq";
            cuphy::tensor_pinned refRsrpEhq     = cuphy::tensor_from_dataset(fInput.open_dataset(datasetName.c_str()), cuphy::tensor_flags::align_tight, cuStrm);
            tRefRsrpEhq.emplace_back(refRsrpEhq.layout());
            tRefRsrpEhq[i] = refRsrpEhq;

#if USE_PUSCH_PER_UE_PREQ_NOISE_VAR
            datasetName                           = "reference_noiseVardBPerUe";
#else
            datasetName                           = "reference_noiseVardB";
#endif
            cuphy::tensor_pinned refNoiseVarPreEq = cuphy::tensor_from_dataset(fInput.open_dataset(datasetName.c_str()), cuphy::tensor_flags::align_tight, cuStrm);
            tRefNoiseVarPreEq.emplace_back(refNoiseVarPreEq.layout());
            tRefNoiseVarPreEq[i] = refNoiseVarPreEq;

            datasetName                            = "reference_postEqNoiseVardB";
            cuphy::tensor_pinned refNoiseVarPostEq = cuphy::tensor_from_dataset(fInput.open_dataset(datasetName.c_str()), cuphy::tensor_flags::align_tight, cuStrm);
            tRefNoiseVarPostEq.emplace_back(refNoiseVarPostEq.layout());
            tRefNoiseVarPostEq[i] = refNoiseVarPostEq;

            datasetName                       = "reference_sinrdB";
            cuphy::tensor_pinned refSinrPreEq = cuphy::tensor_from_dataset(fInput.open_dataset(datasetName.c_str()), cuphy::tensor_flags::align_tight, cuStrm);
            tRefSinrPreEq.emplace_back(refSinrPreEq.layout());
            tRefSinrPreEq[i] = refSinrPreEq;
            
            datasetName                          = "reference_sinrdB_ehq";
            cuphy::tensor_pinned refSinrPreEqEhq = cuphy::tensor_from_dataset(fInput.open_dataset(datasetName.c_str()), cuphy::tensor_flags::align_tight, cuStrm);
            tRefSinrPreEqEhq.emplace_back(refSinrPreEqEhq.layout());
            tRefSinrPreEqEhq[i] = refSinrPreEqEhq;

            datasetName                        = "reference_postEqSinrdB";
            cuphy::tensor_pinned refSinrPostEq = cuphy::tensor_from_dataset(fInput.open_dataset(datasetName.c_str()), cuphy::tensor_flags::align_tight, cuStrm);
            tRefSinrPostEq.emplace_back(refSinrPostEq.layout());
            tRefSinrPostEq[i] = refSinrPostEq;

            //if(gnbConfig.get_value_as<uint8_t>("enableCfoCorrection"))
            {
                datasetName                           = "reference_cfoEstHzPerUe";
                cuphy::tensor_pinned refCfoEstHzPerUe = cuphy::tensor_from_dataset(fInput.open_dataset(datasetName.c_str()), cuphy::tensor_flags::align_tight, cuStrm);
                tRefCfoEstHzPerUe.emplace_back(refCfoEstHzPerUe.layout());
                tRefCfoEstHzPerUe[i] = refCfoEstHzPerUe;
            }

            if(gnbConfig.get_value_as<uint8_t>("enableToEstimation"))
            {
                datasetName                           = "reference_taEstMicroSecPerUe";
                cuphy::tensor_pinned refToEstMicroSecPerUe = cuphy::tensor_from_dataset(fInput.open_dataset(datasetName.c_str()), cuphy::tensor_flags::align_tight, cuStrm);
                tRefToEstMicroSecPerUe.emplace_back(refToEstMicroSecPerUe.layout());
                tRefToEstMicroSecPerUe[i] = refToEstMicroSecPerUe;
            }

            if(m_drmDebug && (nSchUes > 0))
            {
                datasetName = "reference_derateCbsIndicesSizes";
                if(fInput.is_valid_dataset(datasetName.c_str()))
                {
                    tReference_derateCbsIndices      = cuphy::tensor_from_dataset(fInput.open_dataset("reference_derateCbsIndices"), CUPHY_R_32U, cuphy::tensor_flags::align_tight, cuStrm);
                    tReference_derateCbsIndicesSizes = cuphy::tensor_from_dataset(fInput.open_dataset(datasetName.c_str()), CUPHY_R_32U, cuphy::tensor_flags::align_tight, cuStrm);
                }
                else
                {
                    NVLOGF_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "Old TV is missing {} field.  Can't check de-rate-match index results.",datasetName);
                }
            }

            CUDA_CHECK(cudaStreamSynchronize(cuStrm));
        }
        else // api test-vector
        {
            cuphy::cuphyHDF5_struct cellGrpDyn_pars = cuphy::get_HDF5_struct(fInput, "cellGrpDyn_pars");
            hdf5hpp::hdf5_dataset   output_pars     = fInput.open_dataset("output_pars");
            hdf5hpp::hdf5_dataset   eval_pars       = fInput.open_dataset("eval_pars");

            // load total sizes and offsets
            nCbs += output_pars[0]["totNumCbs"].as<uint32_t>();
            nTbs += output_pars[0]["totNumTbs"].as<uint32_t>();
            nBytesVec[i] += output_pars[0]["totNumPayloadBytes"].as<uint32_t>();
            uint32_t nTbsinFile   = output_pars[0]["totNumTbs"].as<uint32_t>();
            uint32_t nCbsinFile   = output_pars[0]["totNumCbs"].as<uint32_t>();
            uint32_t nBytesinFile = output_pars[0]["totNumPayloadBytes"].as<uint32_t>();

            std::vector<uint32_t> offsetCbCrc   = (output_pars[0]["offsetCbCrc"].as<std::vector<uint32_t>>());
            std::vector<uint32_t> offsetTbCrc   = (output_pars[0]["offsetTbCrc"].as<std::vector<uint32_t>>());
            std::vector<uint32_t> offsetPayload = (output_pars[0]["offsetPayload"].as<std::vector<uint32_t>>());

            // derive sizes
            nCbsPerTbVec.resize(nTbs);
            nBytesPerCbVec.resize(nTbs);

            for(int tbIdx = 0; tbIdx < (nTbsinFile - 1); ++tbIdx)
            {
                nCbsPerTbVec[tbIdx]   = offsetCbCrc[tbIdx + 1] - offsetCbCrc[tbIdx];
                nBytesPerCbVec[tbIdx] = (offsetPayload[tbIdx + 1] - offsetPayload[tbIdx]) / nCbsPerTbVec[tbIdx];
            }

            nCbsPerTbVec[nTbsinFile - 1]   = nCbsinFile - offsetCbCrc[nTbsinFile - 1];
            nBytesPerCbVec[nTbsinFile - 1] = (nBytesinFile - offsetPayload[nTbsinFile - 1]) / nCbsPerTbVec[nTbsinFile - 1];

            // load true Bytes
            tTrueTbBytesVec[i] = cuphy::tensor_from_dataset(fInput.open_dataset("tb_data"), CUPHY_R_8U, cuphy::tensor_flags::align_tight, cuStrm);
            CUDA_CHECK(cudaStreamSynchronize(cuStrm));

            // load true cb error pattern
            //std::string inputCbErrName = "cbErr"+std::to_string(i);
            //tTrueCbErrVec[i] = cuphy::tensor_from_dataset(fInput.open_dataset(inputCbErrName.c_str()), CUPHY_R_32U, cuphy::tensor_flags::align_tight, cuStrm);
            //CUDA_CHECK(cudaStreamSynchronize(cuStrm));

            // load intermediate buffers
            interBufferFlag = static_cast<bool>(eval_pars[0]["interBufferFlag"].as<uint8_t>());

            if(interBufferFlag)
            {
                nUes    = cellGrpDyn_pars.get_value_as<uint32_t>("nUes");
                nUeGrps = cellGrpDyn_pars.get_value_as<uint32_t>("nUeGrps");

                // load reference channel estimate:
                cuphy::tensor_pinned Hest = cuphy::tensor_from_dataset(fInput.open_dataset("reference_H_est"), CUPHY_C_32F, cuphy::tensor_flags::align_tight, cuStrm);
                HestRef.emplace_back(Hest.layout());
                HestRef[0] = Hest;

                // load reference CFO estimate:
                cuphy::tensor_pinned cfoEst = cuphy::tensor_from_dataset(fInput.open_dataset("reference_cfo_est"), CUPHY_C_32F, cuphy::tensor_flags::align_tight, cuStrm);
                tRefCfoEst.emplace_back(cfoEst.layout());
                tRefCfoEst[0] = cfoEst;

                // Load LDPC output:
                std::string          ldpcOutName("LDPC_out");
                cuphy::tensor_pinned ldpcOut = cuphy::tensor_from_dataset(fInput.open_dataset(ldpcOutName.c_str()), CUPHY_R_32U, cuphy::tensor_flags::align_tight, cuStrm);
                ldpcOutRef                   = std::move(cuphy::typed_tensor<CUPHY_R_32U, cuphy::pinned_alloc>(ldpcOut.layout()));
                ldpcOutRef                   = ldpcOut;
            }
        }
    }
}

// default constructor
EvalDataset::EvalDataset() :
    nCbs(0),
    nTbs(0),
    nBytesVec(0),
    nBytesPerCbVec{},
    nCbsPerTbVec{},
    tTrueTbBytesVec{}
{}

// copy constructor
EvalDataset::EvalDataset(const EvalDataset& evalDataset) :
    nCbs(evalDataset.nCbs),
    nTbs(evalDataset.nTbs),
    nBytesVec(evalDataset.nBytesVec),
    nBytesPerCbVec(evalDataset.nBytesPerCbVec),
    nCbsPerTbVec(evalDataset.nCbsPerTbVec),
    tTrueTbBytesVec(evalDataset.tTrueTbBytesVec)
{
}

// move operator
EvalDataset& EvalDataset::operator=(EvalDataset&& evalDataset)
{
    nBytesVec = std::move(evalDataset.nBytesVec);
    nCbs      = std::move(evalDataset.nCbs);
    nTbs      = std::move(evalDataset.nTbs);

    nBytesPerCbVec  = std::move(evalDataset.nBytesPerCbVec);
    nCbsPerTbVec    = std::move(evalDataset.nCbsPerTbVec);
    tTrueTbBytesVec = std::move(evalDataset.tTrueTbBytesVec);

    return *this;
}

//-------------------------------------------------------------------------------------
// functions compute channel estimation snrs

double EvalDataset::evalChEstDelayMean(std::vector<cuphy::tensor_device>& tDmrsDelayMeanGpu, int ueGrp, cudaStream_t cuStrm)
{
    // copy chEst buffer to cpu
    cuphy::typed_tensor<CUPHY_R_32F, cuphy::pinned_alloc> tDmrsDelayMeanGpu_copy(tDmrsDelayMeanGpu[ueGrp].layout());
    tDmrsDelayMeanGpu_copy.convert(tDmrsDelayMeanGpu[ueGrp], cuStrm);
    CUDA_CHECK(cudaStreamSynchronize(cuStrm));
    return computeSnr(tDmrsDelayMeanGpu_copy, tRefChEstDelayMean[ueGrp]);
}

double EvalDataset::evalChEstLS(std::vector<cuphy::tensor_device>& tLSEstGpu, int ueGrp, cudaStream_t cuStrm)
{
    // copy chEst buffer to cpu
    cuphy::typed_tensor<CUPHY_C_32F, cuphy::pinned_alloc> tLSEstGpu_copy(tLSEstGpu[ueGrp].layout());
    tLSEstGpu_copy.convert(tLSEstGpu[ueGrp], cuStrm);
    CUDA_CHECK(cudaStreamSynchronize(cuStrm));

    return computeSnr(tLSEstGpu_copy, tRefChEstLSHest[ueGrp]);
}

double EvalDataset::evalChEst(std::vector<cuphy::tensor_device>& tHEstGpu, int ueGrp, cudaStream_t cuStrm)
{
    // copy chEst buffer to cpu
    cuphy::typed_tensor<CUPHY_C_32F, cuphy::pinned_alloc> tHEstGpu_copy(tHEstGpu[ueGrp].layout());
    tHEstGpu_copy.convert(tHEstGpu[ueGrp], cuStrm);
    CUDA_CHECK(cudaStreamSynchronize(cuStrm));

    // compute snr
    double chEstSnr = computeSnr(tHEstGpu_copy, HestRef[ueGrp]);
    return chEstSnr;
}

double EvalDataset::evalChEqCoef(std::vector<cuphy::tensor_device>& tChEqCoefGpu, int ueGrp, cudaStream_t cuStrm)
{
    // copy ChEqCoef buffer to cpu
    cuphy::typed_tensor<CUPHY_C_32F, cuphy::pinned_alloc> tChEqCoefGpu_copy(tChEqCoefGpu[ueGrp].layout());
    tChEqCoefGpu_copy.convert(tChEqCoefGpu[ueGrp], cuStrm);
    CUDA_CHECK(cudaStreamSynchronize(cuStrm));

    // compute snr
    double chEqCoefSnr = computeChEqCoefSnr(tChEqCoefGpu_copy, chEqCoefRef[ueGrp]);
    return chEqCoefSnr;
}

double EvalDataset::evalChEqLLR(std::vector<cuphy::tensor_device>& tChEqLLRGpu, int ueGrp, cudaStream_t cuStrm)
{
    // copy LLR buffer to cpu
    cuphy::typed_tensor<CUPHY_R_16F, cuphy::pinned_alloc> tChEqLLRGpu_copy(tChEqLLRGpu[ueGrp].layout());
    tChEqLLRGpu_copy.convert(tChEqLLRGpu[ueGrp], cuStrm);
    CUDA_CHECK(cudaStreamSynchronize(cuStrm));

    // compute snr
    double chEqLLRSnr = computeChEqLLRSnr(tChEqLLRGpu_copy, eqOutLLRsRef[ueGrp]);
    return chEqLLRSnr;
}

//-------------------------------------------------------------------------------------
// function computes CFO estimation snr

void EvalDataset::evalCfoTaEst(std::vector<cuphy::tensor_device>& tCfoEstGpu, std::vector<cuphy::tensor_device>& tTaEstGpu, cudaStream_t cuStrm)
{
    for(int ueGrpIdx = 0; ueGrpIdx < nUeGrps; ++ueGrpIdx)
    {
        // copy CfoEst buffer to cpu
        cuphy::typed_tensor<CUPHY_C_32F, cuphy::pinned_alloc> tCfoEstCpu(tCfoEstGpu[ueGrpIdx].layout());
        tCfoEstCpu.convert(tCfoEstGpu[ueGrpIdx], cuStrm);
        cudaStreamSynchronize(cuStrm);

        double cfoEstSnr = computeSnr(tCfoEstCpu, tRefCfoEst[ueGrpIdx]);
        printf("ueGroup[%d] CFO Est reference comparison SNR %7.04f dB.", ueGrpIdx, cfoEstSnr);
        if(cfoEstSnr < 30.0f /*cfoSnrTol*/)
        {
            printf(" Mismatch detected, Failed!");
        }
        printf("\n");
        //---------------------------
        // copy TaEst buffer to cpu
        cuphy::typed_tensor<CUPHY_R_32F, cuphy::pinned_alloc> tTaEstCpu(tTaEstGpu[ueGrpIdx].layout());
        tTaEstCpu.convert(tTaEstGpu[ueGrpIdx], cuStrm);
        cudaStreamSynchronize(cuStrm);
        printf("ueGroup[%d] TA Est Ref %f, GPU %f\n", ueGrpIdx, tRefTaEsts[ueGrpIdx](0) * 1000000, tTaEstCpu(ueGrpIdx));

    }

}


double EvalDataset::evalCfoEst(tensor_pinned_C_32F& tRefCfoEst, cuphy::tensor_pinned& tCfoEstRes, cudaStream_t cuStrm)
{
    // copy CFO buffer to cpu
    tensor_pinned_C_32F tCfoEstCpu(tCfoEstRes.layout());
    tCfoEstCpu.convert(tCfoEstRes, cuStrm);
    CUDA_CHECK(cudaStreamSynchronize(cuStrm));

    // compute snr
    double cfoEstSnr = computeSnr(tCfoEstCpu, tRefCfoEst);

    return cfoEstSnr;
}

//-------------------------------------------------------------------------------------
// function computes RSSI comparision snr
double EvalDataset::evalRssi(tensor_pinned_R_32F& tRssiRef, cuphy::tensor_pinned& tRssiRes, cudaStream_t cuStrm, bool verbose)
{
    // copy RSSI buffer to cpu
    tensor_pinned_R_32F tRssiCpu(tRssiRes.layout());
    tRssiCpu.convert(tRssiRes, cuStrm);
    CUDA_CHECK(cudaStreamSynchronize(cuStrm));

    // compute snr
    double rssiSnr = computeSnr(tRssiCpu, tRssiRef, verbose);

    return rssiSnr;
}

//-------------------------------------------------------------------------------------
// function computes RSRP reference check snr
double EvalDataset::evalRsrp(tensor_pinned_R_32F& tRsrpRef, cuphy::tensor_pinned& tRsrpRes, cudaStream_t cuStrm, bool verbose)
{
    // copy buffer to cpu
    tensor_pinned_R_32F tRsrpCpu(tRsrpRes.layout());
    tRsrpCpu.convert(tRsrpRes, cuStrm);
    CUDA_CHECK(cudaStreamSynchronize(cuStrm));

    // compute snr
    double rsrpSnr = computeSnr(tRsrpCpu, tRsrpRef, verbose);

    return rsrpSnr;
}

double EvalDataset::evalRsrpDiff(tensor_pinned_R_32F& tRsrpRef, cuphy::tensor_pinned& tRsrpRes, cudaStream_t cuStrm, bool verbose)
{
    // copy buffer to cpu
    tensor_pinned_R_32F tRsrpCpu(tRsrpRes.layout());
    tRsrpCpu.convert(tRsrpRes, cuStrm);
    CUDA_CHECK(cudaStreamSynchronize(cuStrm));

    // compute snr
    double rsrpDiff = computeDiff(tRsrpCpu, tRsrpRef, verbose);

    return rsrpDiff;
}

//-------------------------------------------------------------------------------------
// function computes Noise-interference power reference check snr
double EvalDataset::evalNoiseIntfVar(tensor_pinned_R_32F& tNoiseIntfRef, cuphy::tensor_pinned& tNoiseIntfRes, cudaStream_t cuStrm, bool verbose)
{
    // copy buffer to cpu
    tensor_pinned_R_32F tNoiseIntfVarCpu(tNoiseIntfRes.layout());
    tNoiseIntfVarCpu.convert(tNoiseIntfRes, cuStrm);
    CUDA_CHECK(cudaStreamSynchronize(cuStrm));

    // compute snr
    double noiseIntfSnr = computeSnr(tNoiseIntfVarCpu, tNoiseIntfRef, verbose);

    return noiseIntfSnr;
}

//-------------------------------------------------------------------------------------
// function computes SINR reference check snr
double EvalDataset::evalSinr(tensor_pinned_R_32F& tSinrRef, cuphy::tensor_pinned& tSinrRes, cudaStream_t cuStrm, bool verbose)
{
    // copy buffer to cpu
    tensor_pinned_R_32F tSinrCpu(tSinrRes.layout());
    tSinrCpu.convert(tSinrRes, cuStrm);
    CUDA_CHECK(cudaStreamSynchronize(cuStrm));

    // compute snr
    double sinrSnr = computeSnr(tSinrCpu, tSinrRef, verbose);

    return sinrSnr;
}

//-------------------------------------------------------------------------------------
// function computes CFO reference check Error Rate
double EvalDataset::evalCfoEstHzPerUe(tensor_pinned_R_32F& tCfoRef, cuphy::tensor_pinned& tCfoRes, cudaStream_t cuStrm, bool verbose)
{
    // copy buffer to cpu
    tensor_pinned_R_32F tCfoCpu(tCfoRes.layout());
    tCfoCpu.convert(tCfoRes, cuStrm);
    CUDA_CHECK(cudaStreamSynchronize(cuStrm));

    // compute Error
    double cfoError = computeDiff(tCfoCpu, tCfoRef, verbose);

    return cfoError;
}

//-------------------------------------------------------------------------------------
// function computes TO reference check Error Rate
double EvalDataset::evalToEstMicroSecPerUe(tensor_pinned_R_32F& tToRef, cuphy::tensor_pinned& tToRes, cudaStream_t cuStrm, bool verbose)
{
    // copy buffer to cpu
    tensor_pinned_R_32F tToCpu(tToRes.layout());
    tToCpu.convert(tToRes, cuStrm);
    CUDA_CHECK(cudaStreamSynchronize(cuStrm));

    // compute Error
    double toError = computeDiff(tToCpu, tToRef, verbose);

    return toError;
}

//-------------------------------------------------------------------------------------
// function evaluates pusch crc removal

void EvalDataset::evalPuschCrc(uint32_t* pTbCrcGpu, uint32_t* pCbCrcGpu, uint8_t* pTbPayloadGpu, cudaStream_t cuStrm)
{
    // copy from Gpu to Cpu
    cuphy::buffer<uint32_t, cuphy::pinned_alloc> tbCrcCpu_buffer(nTbs);
    cuphy::buffer<uint32_t, cuphy::pinned_alloc> cbCrcCpu_buffer(nCbs);

    CUDA_CHECK(cudaMemcpyAsync(tbCrcCpu_buffer.addr(), pTbCrcGpu, sizeof(uint32_t) * nTbs, cudaMemcpyDeviceToHost, cuStrm));
    CUDA_CHECK(cudaMemcpyAsync(cbCrcCpu_buffer.addr(), pCbCrcGpu, sizeof(uint32_t) * nCbs, cudaMemcpyDeviceToHost, cuStrm));
    CUDA_CHECK(cudaStreamSynchronize(cuStrm));

    // compute errors
    uint32_t  nTbCrcErrors = 0;
    uint32_t* pTbCrcCpu    = tbCrcCpu_buffer.addr();
    for(int tbIdx = 0; tbIdx < nTbs; ++tbIdx)
    {
        if(pTbCrcCpu[tbIdx] != static_cast<uint32_t>(0))
        {
            nTbCrcErrors = nTbCrcErrors + 1;
        }
    }

    uint32_t  nCbCrcErrors = 0;
    uint32_t* pCbCrcCpu    = cbCrcCpu_buffer.addr();
    for(int cbIdx = 0; cbIdx < nCbs; ++cbIdx)
    {
        NVLOGD_FMT(NVLOG_PUSCH, "cb idx: {}, cb crc: {}", cbIdx, pCbCrcCpu[cbIdx]);
        if(pCbCrcCpu[cbIdx] != 0)
        {
            nCbCrcErrors++;
        }
    }

    for(uint32_t i = 0; i < nBytesVec.size(); i++)
    {
        cuphy::buffer<uint8_t, cuphy::pinned_alloc> tbPayloadCpu_buffer(nBytesVec[i]);
        CUDA_CHECK(cudaMemcpyAsync(tbPayloadCpu_buffer.addr(), pTbPayloadGpu, sizeof(uint8_t) * (nBytesVec[i]), cudaMemcpyDeviceToHost, cuStrm));
        CUDA_CHECK(cudaStreamSynchronize(cuStrm));
        uint32_t nTbPayloadErrors = 0;
        uint8_t* pRefPayloadBytes = static_cast<uint8_t*>(tTrueTbBytesVec[i].addr());
        uint8_t* pTbPayloadCpu    = tbPayloadCpu_buffer.addr();
        for(int byteIdx = 0; byteIdx < nBytesVec[i]; ++byteIdx)
        {
            if(pTbPayloadCpu[byteIdx] != pRefPayloadBytes[byteIdx])
            {
                nTbPayloadErrors++;
            }
        }
        NVLOGC_FMT(NVLOG_PUSCH, "number of payload byte errors: {} out of {}", nTbPayloadErrors, nBytesVec[i]);
    }

    // print results
    NVLOGW_FMT(NVLOG_PUSCH, "number of Tb CRC errors      : {} out of {}", nTbCrcErrors, nTbs);
    NVLOGW_FMT(NVLOG_PUSCH, "number of Cb CRC errors      : {} out of {}", nCbCrcErrors, nCbs);
}

//-------------------------------------------------------------------------------------
// function to evaluate PuschRx results
void EvalDataset::evalPuschRx(std::string const& resultFileName, StaticApiDataset const& statApiDataset, DynApiDataset const& dynApiDataset, cudaStream_t cuStrm)
{
    // Currently, no evaluation of PuschRx when Factory file exists (!=nullptr)
    // Channel Estimate Factory file currently used to activate AI/ML Chest implementation (TrtEngine specifically)
    if (statApiDataset.puschStatPrms.puschrxChestFactorySettingsFilename) {
        return;
    }
    
    if(! resultFileName.empty())
    {
        // load h5 file
        hdf5hpp::hdf5_file fResult;
        try
        {
            fResult = hdf5hpp::hdf5_file::open(resultFileName.c_str());
        }
        catch(const std::exception& e) // catch(...)
        {
            NVLOGF_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "Error in opening output file (skipping verification of internal pipeline probes)");
            return;
        }

        // @todo: checks for CFO est implemented, need to expand to other datasets
        // std::string datasetName = "taEsts";
        // tensor_pinned_R_32F resTaEsts = cuphy::tensor_from_dataset(fResult.open_dataset(datasetName.c_str()), CUPHY_R_32F, cuphy::tensor_flags::align_tight, cuStrm);

        for(int ueGrpIdx = 0; ueGrpIdx < nUeGrps; ++ueGrpIdx)
        {
            // @todo: In cuPHY CFO and TA are fused in a single kernel, so cannot enable CFO and TA seperately.
            // Need to update MATLAB model to unify this flag
            if(statApiDataset.puschStatPrms.enableCfoCorrection)
            {
                // Verify CFO estimate
                cuphyPuschUeGrpPrm_t const& ueGrpPrms = dynApiDataset.ueGrpPrmsVec[ueGrpIdx];
                if(0 != ueGrpPrms.pDmrsDynPrm->dmrsAddlnPos)
                {
                    // @todo:  MATLAB TVs 7408 and 7409 save the CFO/TA estimates per UE (currently being saved per UE group and causes index exceeds tensor dim error)
                    std::string          datasetName = "CfoEst" + std::to_string(ueGrpIdx);
                    cuphy::tensor_pinned cfoEstRes   = cuphy::tensor_from_dataset(fResult.open_dataset(datasetName.c_str()), CUPHY_C_32F, cuphy::tensor_flags::align_tight, cuStrm);
                    double               cfoEstSnr   = evalCfoEst(tRefCfoEst[ueGrpIdx], cfoEstRes, cuStrm);
                    NVLOGC_FMT(NVLOG_PUSCH, "ueGroup[{}] CFO Est reference comparison SNR {:7.04f}", ueGrpIdx, cfoEstSnr);

                    // @todo: Verify TA estimate using SNR
                    // double taEstSnr =  evalTaEst(tRefTaEsts[ueGrpIdx], taEstsRes[ueGrpIdx], cuStrm);
                    NVLOGC_FMT(NVLOG_PUSCH, "ueGroup[{}] TA Est Ref {} GPU {}", ueGrpIdx, tRefTaEsts[ueGrpIdx](0) * 1000000, dynApiDataset.getOutputTaEst(ueGrpIdx));
                }
            }
        }

        if(statApiDataset.puschStatPrms.enableRssiMeasurement)
        {
            bool verbose = true;

            // Verify RSSI measurement
            std::string          datasetName = "Rssi";
            cuphy::tensor_pinned rssiRes     = cuphy::tensor_from_dataset(fResult.open_dataset(datasetName.c_str()), CUPHY_R_32F, cuphy::tensor_flags::align_tight, cuStrm);
            double               rssiSnr     = evalRssi(tRefRssi[0], rssiRes, cuStrm, verbose);
            NVLOGC_FMT(NVLOG_PUSCH, "Rssi reference comparison SNR {:7.04f}", rssiSnr);
            NVLOGC_FMT(NVLOG_PUSCH, "tRefRssi {}", tRefRssi[0].desc().get_info().to_string(false).c_str());

            datasetName                      = "RssiFull";
            cuphy::tensor_pinned rssiFullRes = cuphy::tensor_from_dataset(fResult.open_dataset(datasetName.c_str()), CUPHY_R_32F, cuphy::tensor_flags::align_tight, cuStrm);
            double               rssiFullSnr = evalRssi(tRefRssiFull[0], rssiFullRes, cuStrm, verbose);
            NVLOGC_FMT(NVLOG_PUSCH, "Rssi Full reference comparison SNR {:7.04f}", rssiFullSnr);
            NVLOGC_FMT(NVLOG_PUSCH, "tRefRssiFull {}", tRefRssiFull[0].desc().get_info().to_string(false).c_str());
            NVLOGC_FMT(NVLOG_PUSCH, "rssiFullRes {}", rssiFullRes.desc().get_info().to_string(false).c_str());
        }

        if(statApiDataset.puschStatPrms.enableSinrMeasurement)
        {
            bool verbose = true;

            // Verify RSRP, noise variance, SINR measurement
            std::string          datasetName = "Rsrp";
            cuphy::tensor_pinned rsrpRes     = cuphy::tensor_from_dataset(fResult.open_dataset(datasetName.c_str()), CUPHY_R_32F, cuphy::tensor_flags::align_tight, cuStrm);
            double               rsrpSnr     = evalRsrp(tRefRsrp[0], rsrpRes, cuStrm, verbose);
            NVLOGC_FMT(NVLOG_PUSCH, "Rsrp reference comparison SNR {:7.04f}", rsrpSnr);
            NVLOGC_FMT(NVLOG_PUSCH, "tRefRsrp {}", tRefRsrp[0].desc().get_info().to_string(false).c_str());

#if USE_PUSCH_PER_UE_PREQ_NOISE_VAR
            datasetName                           = "NoiseVarPreEqPerUe";
#else
            datasetName                           = "NoiseVarPreEq";
#endif
            cuphy::tensor_pinned noiseVarPreEqRes = cuphy::tensor_from_dataset(fResult.open_dataset(datasetName.c_str()), CUPHY_R_32F, cuphy::tensor_flags::align_tight, cuStrm);
            double               noiseVarPreEqSnr = evalNoiseIntfVar(tRefNoiseVarPreEq[0], noiseVarPreEqRes, cuStrm, verbose);
            NVLOGC_FMT(NVLOG_PUSCH, "{} reference comparison SNR {:7.04f}", datasetName, noiseVarPreEqSnr);


            datasetName                  = "SinrPreEq";
            cuphy::tensor_pinned sinrPreEqRes = cuphy::tensor_from_dataset(fResult.open_dataset(datasetName.c_str()), CUPHY_R_32F, cuphy::tensor_flags::align_tight, cuStrm);
            double               sinrPreEqSnr = evalSinr(tRefSinrPreEq[0], sinrPreEqRes, cuStrm, verbose);
            NVLOGC_FMT(NVLOG_PUSCH, "{} reference comparison SNR {:7.04f}", datasetName, sinrPreEqSnr);
            
            datasetName                      = "NoiseVarPostEq";
            cuphy::tensor_pinned noiseVarPostEqRes = cuphy::tensor_from_dataset(fResult.open_dataset(datasetName.c_str()), CUPHY_R_32F, cuphy::tensor_flags::align_tight, cuStrm);
            double               noiseVarPostEqSnr = evalNoiseIntfVar(tRefNoiseVarPostEq[0], noiseVarPostEqRes, cuStrm, verbose);
            NVLOGC_FMT(NVLOG_PUSCH, "{} reference comparison SNR {:7.04f}", datasetName, noiseVarPostEqSnr);

            datasetName                  = "SinrPostEq";
            cuphy::tensor_pinned sinrPostEqRes = cuphy::tensor_from_dataset(fResult.open_dataset(datasetName.c_str()), CUPHY_R_32F, cuphy::tensor_flags::align_tight, cuStrm);
            double               sinrPostEqSnr = evalSinr(tRefSinrPostEq[0], sinrPostEqRes, cuStrm, verbose);
            NVLOGC_FMT(NVLOG_PUSCH, "{} reference comparison SNR {:7.04f}", datasetName, sinrPostEqSnr);
        }

        if(statApiDataset.puschStatPrms.enableCfoCorrection)
        {
            // Verify CFO(Hz) estimates
            std::string          datasetName = "CfoEstHzPerUe";
            cuphy::tensor_pinned cfoHzRes     = cuphy::tensor_from_dataset(fResult.open_dataset(datasetName.c_str()), CUPHY_R_32F, cuphy::tensor_flags::align_tight, cuStrm);
            double               cfoHzError     = evalCfoEstHzPerUe(tRefCfoEstHzPerUe[0], cfoHzRes, cuStrm, true);
            NVLOGC_FMT(NVLOG_PUSCH, "{} reference comparison {:7.04f} Hz", datasetName, cfoHzError);
        }

        if(statApiDataset.puschStatPrms.enableToEstimation)
        {
            // Verify TO(microsecond) estimates
            std::string          datasetName     = "ToEstMicroSecPerUe";
            cuphy::tensor_pinned toMicroSecRes   = cuphy::tensor_from_dataset(fResult.open_dataset(datasetName.c_str()), CUPHY_R_32F, cuphy::tensor_flags::align_tight, cuStrm);
            double               toMicroSecError = evalToEstMicroSecPerUe(tRefToEstMicroSecPerUe[0], toMicroSecRes, cuStrm, true);
            NVLOGC_FMT(NVLOG_PUSCH, "{} reference comparison {:7.08f} uS", datasetName, toMicroSecError);
        }
    }

    if(statApiDataset.puschStatPrms.enableRssiMeasurement)
    {
        // Verify RSSI measurement
        for(int ueGrpIdx = 0; ueGrpIdx < dynApiDataset.cellGrpDynPrm.nUeGrps; ueGrpIdx++)
        {
            cuphy::tensor_pinned tRssiResIface(const_cast<float*>(dynApiDataset.getOutputRssiPtr())+ueGrpIdx, CUPHY_R_32F, dynApiDataset.cellGrpDynPrm.nUeGrps-ueGrpIdx, cuphy::tensor_flags::align_tight);
            double               rssiIfaceSnr = evalRssi(tRefRssi[ueGrpIdx], tRssiResIface, cuStrm, true);
            NVLOGC_FMT(NVLOG_PUSCH, "RSSI reference comparison SNR {:7.04f} for UEGRP[{}]", rssiIfaceSnr, ueGrpIdx);
            if(rssiIfaceSnr < 30) NVLOGF_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "RSSI mismatch, reference comparison SNR {:7.04f} for UEGRP[{}]", rssiIfaceSnr, ueGrpIdx);
        }
        
        //////////////////////////////// RSSI for Early-HARQ  ////////////////////////////////////////////////////////////////////////////
        if(dynApiDataset.DataOut.isEarlyHarqPresent)
        {
            for(int ueGrpIdx = 0; ueGrpIdx < dynApiDataset.cellGrpDynPrm.nUeGrps; ueGrpIdx++)
            {
                cuphy::tensor_pinned tRssiEarlyHarqResIface(const_cast<float*>(dynApiDataset.evalRssiEhq)+ueGrpIdx, CUPHY_R_32F, dynApiDataset.cellGrpDynPrm.nUeGrps-ueGrpIdx, cuphy::tensor_flags::align_tight);
                double               rssiEarlyHarqIfaceSnr = evalRssi(tRefRssiEhq[ueGrpIdx], tRssiEarlyHarqResIface, cuStrm, true);
                NVLOGC_FMT(NVLOG_PUSCH, "RSSI (Early-HARQ) reference comparison SNR {:7.04f} for UEGRP[{}]", rssiEarlyHarqIfaceSnr, ueGrpIdx);
                if(rssiEarlyHarqIfaceSnr < 30) NVLOGF_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "RSSI (early-HARQ) mismatch, reference comparison SNR {:7.04f} for UEGRP[{}]", rssiEarlyHarqIfaceSnr, ueGrpIdx);
            }
        }
    }

    if(statApiDataset.puschStatPrms.enableSinrMeasurement)
    {
        // Verify RSRP, noise variance, SINR measurement
        cuphy::tensor_pinned tRsrpResIface(const_cast<float*>(dynApiDataset.getOutputRsrpPtr()), CUPHY_R_32F, dynApiDataset.cellGrpDynPrm.nUes, cuphy::tensor_flags::align_tight);
        double               rsrpIfaceSnr     = evalRsrp(tRefRsrp[0], tRsrpResIface, cuStrm, true);
        NVLOGC_FMT(NVLOG_PUSCH, "RSRP reference comparison SNR {:7.04f}", rsrpIfaceSnr);
        if(rsrpIfaceSnr < 30) NVLOGF_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "RSRP mismatch, reference comparison SNR {:7.04f}", rsrpIfaceSnr);
        double               rsrpIfaceDiff    = evalRsrpDiff(tRefRsrp[0], tRsrpResIface, cuStrm);
        NVLOGC_FMT(NVLOG_PUSCH, "RSRP reference comparison difference (dB) {}", rsrpIfaceDiff);
        if(rsrpIfaceDiff > 0.01) NVLOGF_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "RSRP mismatch, reference comparison difference (dB) {}", rsrpIfaceDiff);
        
        //////////////////////////////// RSRP for Early-HARQ  ////////////////////////////////////////////////////////////////////////////
        if(dynApiDataset.DataOut.isEarlyHarqPresent)
        {
            cuphy::tensor_pinned tRsrpEarlyHarqResIface(const_cast<float*>(dynApiDataset.evalRsrpEhq), CUPHY_R_32F, dynApiDataset.cellGrpDynPrm.nUes, cuphy::tensor_flags::align_tight);
            double               rsrpEarlyHarqIfaceSnr = evalRsrp(tRefRsrpEhq[0], tRsrpEarlyHarqResIface, cuStrm, true);
            NVLOGC_FMT(NVLOG_PUSCH, "RSRP (Early-HARQ) reference comparison SNR {:7.04f}", rsrpEarlyHarqIfaceSnr);
            if(rsrpEarlyHarqIfaceSnr < 30) NVLOGW_FMT(NVLOG_PUSCH, "RSRP mismatch (Early-HARQ), reference comparison SNR {:7.04f}", rsrpEarlyHarqIfaceSnr);
            double               rsrpEarlyHarqIfaceDiff    = evalRsrpDiff(tRefRsrpEhq[0], tRsrpEarlyHarqResIface, cuStrm);
            NVLOGC_FMT(NVLOG_PUSCH, "RSRP (Early-HARQ) reference comparison difference (dB) {}", rsrpEarlyHarqIfaceDiff);
            if(rsrpEarlyHarqIfaceDiff > 0.01) NVLOGF_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "RSRP mismatch (Early-HARQ), reference comparison difference (dB) {}", rsrpEarlyHarqIfaceDiff);
        }

        // @todo1: TC 7409: with both CFO and TO errors, when the noise-power postEq is small (and hence SNR postEq is high), there is ~5dB error in cuPHY estimate relative to 5GModel.
        // This needs to be debugged. A characterization of error between cuPHY noise-power estimate (Ree) versus 5GModel is needed by sweeping SNR values with and without CFO/TO.
        // @todo2: The error relative to 5GModel in the cuPHY estimated (preEq and postEq) noise-power and hence SNR values vary as a function of the true SNR.
        // Thus reference check SNR threshold needs to be a function of true SNR.
        // For e.g. TC 7415 when signal power (RSRP= -39.683dB) is close to noise power (preEq noise-power = -40.0019dB) resulting in small SNR = 0.318869dB a
        // 0.5dB error in either signal-power or noise-power (sinrPreEqDb -0.1431656, rsrpDb -39.683, noiseIntfVarPreEqDb -39.53987) can result in reference check SNR
        // of 30dB to fail.
        // @todo3: PostEq noiseVar (and hence SNR) are not a good match with 5GModel reference for negative SINR levels
        static constexpr float PUSCH_SNR_MIN_THRESHOLD = 1.0f;
        static constexpr float PUSCH_SNR_REF_CHECK_SNR_MIN_THRESHOLD = 30.0f;
        static constexpr float PUSCH_POST_EQ_SNR_REF_CHECK_SNR_MAX_THRESHOLD = 20.0f;

        float minRefSinrPreEq = std::numeric_limits<float>::max();
        size_t nSinrPreEq = tRefSinrPreEq[0].dimensions()[0];
        float* pRefSinrPreEq = tRefSinrPreEq[0].addr();
        for(int32_t i = 0; i < nSinrPreEq; ++i)
        {
            minRefSinrPreEq = std::min(minRefSinrPreEq, pRefSinrPreEq[i]);
        }

        float maxRefSinrPostEq = std::numeric_limits<float>::min();
        size_t nSinrPostEq = tRefSinrPostEq[0].dimensions()[0];
        float* pRefSinrPostEq = tRefSinrPostEq[0].addr();
        for(int32_t i = 0; i < nSinrPostEq; ++i)
        {
            maxRefSinrPostEq = std::max(maxRefSinrPostEq, pRefSinrPostEq[i]);
        }

        if(minRefSinrPreEq > PUSCH_SNR_MIN_THRESHOLD)
        {
#if USE_PUSCH_PER_UE_PREQ_NOISE_VAR
            cuphy::tensor_pinned tNoiseVarPreEqResIface(const_cast<float*>(dynApiDataset.getOutputNoiseVarPreEqPtr()), CUPHY_R_32F, dynApiDataset.cellGrpDynPrm.nUes, cuphy::tensor_flags::align_tight);
#else
            cuphy::tensor_pinned tNoiseVarPreEqResIface(const_cast<float*>(dynApiDataset.getOutputNoiseVarPreEqPtr()), CUPHY_R_32F, dynApiDataset.cellGrpDynPrm.nUeGrps, cuphy::tensor_flags::align_tight);
#endif
            double               noiseVarPreEqIfaceSnr = evalNoiseIntfVar(tRefNoiseVarPreEq[0], tNoiseVarPreEqResIface, cuStrm);
            NVLOGC_FMT(NVLOG_PUSCH, "NoiseVarPreEqIntf interface reference comparison SNR {:7.04f}", noiseVarPreEqIfaceSnr);
            if(noiseVarPreEqIfaceSnr < 30) NVLOGF_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "NoiseVarPreEqIntf mismatch, reference comparison SNR {:7.04f}", noiseVarPreEqIfaceSnr);

            cuphy::tensor_pinned tSinrPreEqResIface(const_cast<float*>(dynApiDataset.getOutputSinrPreEqPtr()), CUPHY_R_32F, dynApiDataset.cellGrpDynPrm.nUes, cuphy::tensor_flags::align_tight);
            double               sinrPreEqIfaceSnr = evalSinr(tRefSinrPreEq[0], tSinrPreEqResIface, cuStrm, true);
            NVLOGC_FMT(NVLOG_PUSCH, "SINR PreEq interface reference comparison SNR {:7.04f}", sinrPreEqIfaceSnr);
            if(sinrPreEqIfaceSnr < 30) NVLOGF_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "SINR PreEq mismatch, reference comparison SNR {:7.04f}", sinrPreEqIfaceSnr);
            
            //////////////////////////////// Pre-Eq SINR for Early-HARQ  ////////////////////////////////////////////////////////////////////////////
            if(dynApiDataset.DataOut.isEarlyHarqPresent)
            {
                cuphy::tensor_pinned tSinrPreEqEarlyHarqResIface(const_cast<float*>(dynApiDataset.evalPreEqSinrEhq), CUPHY_R_32F, dynApiDataset.cellGrpDynPrm.nUes, cuphy::tensor_flags::align_tight);
                double               sinrPreEqEarlyHarqIfaceSnr = evalSinr(tRefSinrPreEqEhq[0], tSinrPreEqEarlyHarqResIface, cuStrm, true);
                NVLOGC_FMT(NVLOG_PUSCH, "SINR PreEq (Early-HARQ) interface reference comparison SNR {:7.04f}", sinrPreEqEarlyHarqIfaceSnr);
                if(sinrPreEqEarlyHarqIfaceSnr < 30) NVLOGF_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "SINR PreEq (Early-Harq) mismatch, reference comparison SNR {:7.04f}", sinrPreEqEarlyHarqIfaceSnr);
            }
            
            //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            if(statApiDataset.puschStatPrms.enableWeightedAverageCfo!=1) // bypass the post-Eq measurement comparison when enableWeightedAverageCfo is enabled.
            {
            cuphy::tensor_pinned tNoiseVarPostEqResIface(const_cast<float*>(dynApiDataset.getOutputNoiseVarPostEqPtr()), CUPHY_R_32F, dynApiDataset.cellGrpDynPrm.nUes, cuphy::tensor_flags::align_tight);
            double               noiseVarPostEqIfaceSnr = evalNoiseIntfVar(tRefNoiseVarPostEq[0], tNoiseVarPostEqResIface, cuStrm);
            NVLOGC_FMT(NVLOG_PUSCH, "NoiseVarPostEqIntf interface reference comparison SNR {:7.04f}", noiseVarPostEqIfaceSnr);
            if(noiseVarPostEqIfaceSnr < PUSCH_SNR_REF_CHECK_SNR_MIN_THRESHOLD)
            {
                // See @todo1 above
                if(!((statApiDataset.puschStatPrms.enableCfoCorrection) && (maxRefSinrPostEq > PUSCH_POST_EQ_SNR_REF_CHECK_SNR_MAX_THRESHOLD)))
                {
                    NVLOGF_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "NoiseVarPostEqIntf mismatch, reference comparison SNR {:7.04f}", noiseVarPostEqIfaceSnr);
                }
                else
                {
                    NVLOGC_FMT(NVLOG_PUSCH, "enableCfoCorrection {}, maxRefSinrPostEq {:7.04f}, skipping reference checks on postEq noise-power and SNR", statApiDataset.puschStatPrms.enableCfoCorrection, maxRefSinrPostEq);
                }
            }

            cuphy::tensor_pinned tSinrPostEqResIface(const_cast<float*>(dynApiDataset.getOutputSinrPostEqPtr()), CUPHY_R_32F, dynApiDataset.cellGrpDynPrm.nUes, cuphy::tensor_flags::align_tight);
            double               sinrPostEqIfaceSnr = evalSinr(tRefSinrPostEq[0], tSinrPostEqResIface, cuStrm);
            NVLOGC_FMT(NVLOG_PUSCH, "SINR PostEq interface reference comparison SNR {:7.04f}", sinrPostEqIfaceSnr);

            if(sinrPostEqIfaceSnr < PUSCH_SNR_REF_CHECK_SNR_MIN_THRESHOLD)
            {
                // See @todo1 above
                if(!((statApiDataset.puschStatPrms.enableCfoCorrection) && (maxRefSinrPostEq > PUSCH_POST_EQ_SNR_REF_CHECK_SNR_MAX_THRESHOLD)))
                {
                    NVLOGF_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "SINR PostEq mismatch, reference comparison SNR {:7.04f}", sinrPostEqIfaceSnr);
                }
                else
                {
                    NVLOGC_FMT(NVLOG_PUSCH, "enableCfoCorrection {}, maxRefSinrPostEq {:7.04f}, skipping reference checks on postEq noise-power and SNR", statApiDataset.puschStatPrms.enableCfoCorrection, maxRefSinrPostEq);
                }
            }
            
            }
            else
            {
                NVLOGC_FMT(NVLOG_PUSCH, "Bypass post-Eq SNR/noise var comparisons due to enableWeightedAverageCfo=1.");
            } // if(statApiDataset.puschStatPrms.enableWeightedAverageCfo!=1)


        }
        else
        {
            NVLOGC_FMT(NVLOG_PUSCH, "minRefSinrPreEq {:7.04f}, skipping reference checks on preEq/postEq SNR, noise-power", minRefSinrPreEq);
        }
    }

    if(statApiDataset.puschStatPrms.enableCfoCorrection)
    {
       // Verify CFO(Hz) estimates
        cuphy::tensor_pinned tCfoHzResIface(const_cast<float*>(dynApiDataset.getOutputCfoHzPtr()), CUPHY_R_32F, dynApiDataset.cellGrpDynPrm.nUes, cuphy::tensor_flags::align_tight);
        double               cfoHzIfaceError = evalCfoEstHzPerUe(tRefCfoEstHzPerUe[0], tCfoHzResIface, cuStrm);
        NVLOGC_FMT(NVLOG_PUSCH, "CFO(Hz) reference comparison err {:7.04f} Hz", cfoHzIfaceError);
        if(cfoHzIfaceError > CUPHY_PUSCH_RX_CFO_CHECK_THRESHOLD) NVLOGF_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "CFO(Hz) mismatch, reference comparison error {:7.04f} Hz", cfoHzIfaceError);
    }

    if(statApiDataset.puschStatPrms.enableToEstimation)
    {
       // Verify TO(microsecond) estimates
        cuphy::tensor_pinned tToMicroSecResIface(const_cast<float*>(dynApiDataset.getOutputToMicroSecPtr()), CUPHY_R_32F, dynApiDataset.cellGrpDynPrm.nUes, cuphy::tensor_flags::align_tight);
        double               toMicroSecIfaceError = evalToEstMicroSecPerUe(tRefToEstMicroSecPerUe[0], tToMicroSecResIface, cuStrm);
        NVLOGC_FMT(NVLOG_PUSCH, "TO(microsecond) reference comparison err {:7.08f} us", toMicroSecIfaceError);
        if(toMicroSecIfaceError > CUPHY_PUSCH_RX_TO_CHECK_THRESHOLD) NVLOGF_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "TO(microsecond) mismatch, reference comparison error {:7.08} us", toMicroSecIfaceError);
    }
}

//-------------------------------------------------------------------------------------
// function computes number of codeblock errors

uint32_t EvalDataset::computeNumCbErrors(DynApiDataset const& dynApiDataset)
{
    // parameters
    const std::vector<cuphyPuschUePrm_t>& uePrmsVec = dynApiDataset.uePrmsVec;

    // output of Rx pipeline:
    uint8_t*  pEstTbBytes            = dynApiDataset.DataOut.pTbPayloads;
    uint32_t* pStartOffsetsTbPayload = dynApiDataset.DataOut.pStartOffsetsTbPayload;
    uint32_t* pEstTbCrcs;
    uint32_t* pEstCbCrcs;
    if(nTbs>0)
    {
        pEstTbCrcs = dynApiDataset.DataOut.pTbCrcs;
    }
    if(nCbs>0)
    {
        pEstCbCrcs = dynApiDataset.DataOut.pCbCrcs;
    }
    // true TB bytes:
    uint32_t iterator     = 0;
    uint32_t oldTbsIdx    = 0;
    uint8_t* pTrueTbBytes = static_cast<uint8_t*>(tTrueTbBytesVec[iterator].addr());
    uint32_t* pTrueTbCrcErr;
    if(nSchUes>0)
    {
        pTrueTbCrcErr  = static_cast<uint32_t*>(tTrueTbCrcErrVec[iterator].addr());
    }

    // compute Bler
    uint32_t nCbErrors        = 0;
    uint32_t nCbErrMismatches = 0;
    uint32_t totalCbErrors    = 0;
    uint32_t nCbCrcMismatches = 0;
    uint32_t nTbCrcMismatches = 0;
    uint32_t numCbs           = 0;
    uint32_t numTotalCbs      = 0;
    for(int tbIdx = 0; tbIdx < nTbs; tbIdx++)
    {
        if(uePrmsVec[tbIdx].pduBitmap & 1) // check 0'th bit for SCH data
        {
            uint32_t nBytesPerCb        = nBytesPerCbVec[tbIdx];
            uint32_t ByteOffset         = pStartOffsetsTbPayload[tbIdx];
            uint32_t ByteOffsetDatasets = dynApiDataset.bStartOffsetsTbPayloadDatasetsVec[tbIdx];
            uint32_t* pTrueCbCrcErr  = static_cast<uint32_t*>(tTrueCbCrcErrVec[tbIdx].addr());

            //Tb CRC comparison agasin tbErr //
            if((pTrueTbCrcErr[tbIdx-oldTbsIdx]==0)&&(pEstTbCrcs[tbIdx]!=0)) nTbCrcMismatches +=1;
            if((pTrueTbCrcErr[tbIdx-oldTbsIdx]==1)&&(pEstTbCrcs[tbIdx]==0))
            {
                NVLOGC_FMT(NVLOG_PUSCH, "Cell # {} : TbIdx: {}: Warning: cuPHY can decode data with EstTbCrc {}, but 5GModel cannot with TrueTbCrc {}", iterator, tbIdx, pEstTbCrcs[tbIdx], pTrueTbCrcErr[tbIdx-oldTbsIdx]);
            }
            NVLOGD_FMT(NVLOG_PUSCH, "pTrueTbCrcErr={}, pEstTbCrcs={}", pTrueTbCrcErr[tbIdx-oldTbsIdx], pEstTbCrcs[tbIdx]);
            //////////////////////////////////
            numCbs    = nCbsPerTbVec[tbIdx];
            for(int cbIdx = 0; cbIdx < numCbs; ++cbIdx)
            {
                int cbErrorFlag = memcmp(pTrueTbBytes + ByteOffsetDatasets + cbIdx * nBytesPerCb, pEstTbBytes + ByteOffset + cbIdx * nBytesPerCb, nBytesPerCb);

                if(pEstCbCrcs[numTotalCbs]!=0)
                {
                    if(cbErrorFlag==0)
                    {
                        NVLOGC_FMT(NVLOG_PUSCH, "Cell # {} : TbIdx: {} cbIdx: {}: Warning: cuPHY and 5GModel have the same decoded results, but EstCbCrc {} and TrueCbCrc {}", iterator, tbIdx, cbIdx, pEstCbCrcs[numTotalCbs], pTrueCbCrcErr[cbIdx]);
                    }
                    //force cbErrorFlag non-zero if estimated Cb CRC is non-zero
                    cbErrorFlag = 1;
                }

                if(cbErrorFlag != 0) nCbErrors += 1;

                if(((cbErrorFlag != 0)&&(pTrueCbCrcErr[cbIdx]==0))||((cbErrorFlag == 0)&&(pTrueCbCrcErr[cbIdx]==1))) nCbErrMismatches += 1;

                if((pTrueCbCrcErr[cbIdx]==0)&&(pEstCbCrcs[numTotalCbs]!=0)) nCbCrcMismatches +=1;
                if((pTrueCbCrcErr[cbIdx]==1)&&(pEstCbCrcs[numTotalCbs]==0))
                {
                    NVLOGC_FMT(NVLOG_PUSCH, "Cell # {} : TbIdx: {} cbIdx: {}: Warning: cuPHY can decode data with EstCbCrc {}, but 5GModel cannot with TrueCbCrc {}", iterator, tbIdx, cbIdx, pEstCbCrcs[numTotalCbs], pTrueCbCrcErr[cbIdx]);
                }
                NVLOGD_FMT(NVLOG_PUSCH, "pTrueCbCrcErr={}, pEstCbCrcs={}", pTrueCbCrcErr[cbIdx], pEstCbCrcs[numTotalCbs]);
                numTotalCbs += 1;

                bool enableVerboseTbPayloadLog = false; // true; // (cbErrorFlag != 0)
                if(enableVerboseTbPayloadLog)
                {
                    for(int32_t byteIdx = 0; byteIdx < nBytesPerCb; ++byteIdx)
                    //for(int32_t byteIdx = 0; byteIdx < 10; ++byteIdx)
                    {
                        int32_t trueTbByteIdx = ByteOffsetDatasets + cbIdx * nBytesPerCb + byteIdx;
                        int32_t estTbByteIdx  = ByteOffset + cbIdx * nBytesPerCb + byteIdx;

                        if(pTrueTbBytes[trueTbByteIdx] != pEstTbBytes[estTbByteIdx])
                        {
                            NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "PuschRx: Error (tbIdx cbIdx) ({} {}) byteIdx[{}] Ref 0x{:02x} Gpu 0x{:02x}",
                                       tbIdx, cbIdx, byteIdx, pTrueTbBytes[trueTbByteIdx], pEstTbBytes[estTbByteIdx]);
                        }
                        else
                        {
                            NVLOGC_FMT(NVLOG_PUSCH, "PuschRx: (tbIdx cbIdx) ({} {}) byteIdx[{}] Ref 0x{:02x} Gpu 0x{:02x}", tbIdx, cbIdx, byteIdx, pTrueTbBytes[trueTbByteIdx], pEstTbBytes[estTbByteIdx]);
                        }
                    }
                }
            }

            NVLOGD_FMT(NVLOG_PUSCH, "Cell {} TbIdx {}: numCbs {} nBytesPerCb {} ByteOffset {} ByteOffsetDatasets {}", iterator, tbIdx, numCbs, nBytesPerCb, ByteOffset, ByteOffsetDatasets);
        }
        else
        {
            numCbs = 0;
        }
        if(numCbs>0)
        {   if((nCbErrMismatches>0)||(nCbCrcMismatches>0))
            {
                NVLOGF_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "Cell # {} : TbIdx: {} Metric - Block Error Rate      : {:4.4f} (Error CBs {}, Mismatched CBs {}, MismatchedCRC CBs {}, Total CBs {})",
                    iterator,
                    tbIdx,
                    static_cast<float>(nCbErrors) / static_cast<float>(numCbs),
                    nCbErrors,
                    nCbErrMismatches,
                    nCbCrcMismatches,
                    numCbs);
            }
            else
            {
                NVLOGC_FMT(NVLOG_PUSCH, "Cell # {} : TbIdx: {} Metric - Block Error Rate      : {:4.4f} (Error CBs {}, Mismatched CBs {}, MismatchedCRC CBs {}, Total CBs {})",
                    iterator,
                    tbIdx,
                    static_cast<float>(nCbErrors) / static_cast<float>(numCbs),
                    nCbErrors,
                    nCbErrMismatches,
                    nCbCrcMismatches,
                    numCbs);
            }
        }
        else if(numCbs==0)
        {
            NVLOGC_FMT(NVLOG_PUSCH, "Cell # {} : TbIdx: {} Metric - Block Error Rate      : {:4.4f} (Error CBs {}, Mismatched CBs {}, MismatchedCRC CBs {}, Total CBs {})",
                    iterator,
                    tbIdx,
                    0.0,
                    nCbErrors,
                    nCbErrMismatches,
                    nCbCrcMismatches,
                    numCbs);
        }
        totalCbErrors    += nCbErrors;
        nCbErrors        = 0;
        nCbErrMismatches = 0;
        nCbCrcMismatches = 0;

        if((tbIdx+1-oldTbsIdx) == nTbsInFileVec[iterator])
        {  
            if(nTbCrcMismatches>0)
            {
                NVLOGF_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "Cell # {} :          Metric - TB CRC Error      :(MismatchedCRC TBs {}, Total TBs {})",
                                 iterator,
                                 nTbCrcMismatches,
                                 nTbsInFileVec[iterator]);
            }
            else
            {
                NVLOGC_FMT(NVLOG_PUSCH, "Cell # {} :          Metric - TB CRC Error      :(MismatchedCRC TBs {}, Total TBs {})",
                                 iterator,
                                 nTbCrcMismatches,
                                 nTbsInFileVec[iterator]);
            }
            nTbCrcMismatches = 0;
            oldTbsIdx = tbIdx + 1;
            ++iterator;
            if(iterator < nTbsInFileVec.size())
            {
                pTrueTbBytes = static_cast<uint8_t*>(tTrueTbBytesVec[iterator].addr());
                pTrueTbCrcErr= static_cast<uint32_t*>(tTrueTbCrcErrVec[iterator].addr());
            }
        }
    }
    return totalCbErrors;
}

//-------------------------------------------------------------------------------------
// Function computes number of UCI codeblock errors

void EvalDataset::computeNumUciCbErrors(DynApiDataset const& dynApiDataset, bool evalEarlyHarqFlag)
{
    if(evalEarlyHarqFlag && (dynApiDataset.DataOut.isEarlyHarqPresent != 1))
        return;

    // cuphy output:
    const std::vector<cuphyPuschUePrm_t>& uePrmsVec             = dynApiDataset.uePrmsVec;
    const uint8_t*                        CsiP1DetectionStatus  = dynApiDataset.DataOut.CsiP1DetectionStatus;
    const uint8_t*                        CsiP2DetectionStatus  = dynApiDataset.DataOut.CsiP2DetectionStatus;
    const cuphyUciOnPuschOutOffsets_t*    pUciOnPuschOutOffsets = dynApiDataset.DataOut.pUciOnPuschOutOffsets;

    const uint8_t*                        pUciPayloads          = (evalEarlyHarqFlag) ? dynApiDataset.evalUciPayloads : dynApiDataset.DataOut.pUciPayloads;
    const uint8_t*                        pUciCrcFlags          = (evalEarlyHarqFlag) ? dynApiDataset.evalUciCrcFlags : dynApiDataset.DataOut.pUciCrcFlags;
    const uint8_t*                        HarqDetectionStatus   = (evalEarlyHarqFlag) ? dynApiDataset.evalHarqDetectionStatus : dynApiDataset.DataOut.HarqDetectionStatus;


    uint32_t oldTbsIdx    = 0;
    uint32_t iterator     = 0;

    uint8_t* pRefUciPayloadBytes = static_cast<uint8_t*>(tRefUciPayloadBytesVec[iterator].addr());
    uint8_t* pRefUciCrcFlags = static_cast<uint8_t*>(tRefUciCrcFlagsVec[iterator].addr());
    uint8_t* pRefUciDTXs = static_cast<uint8_t*>(tRefUciDTXsVec[iterator].addr());
    uint8_t* pRefUciHarqDetStatus = static_cast<uint8_t*>(tRefUciHarqDetStatusVec[iterator].addr());
    uint8_t* pRefUciCsi1DetStatus = static_cast<uint8_t*>(tRefUciCsi1DetStatusVec[iterator].addr());
    uint8_t* pRefUciCsi2DetStatus = static_cast<uint8_t*>(tRefUciCsi2DetStatusVec[iterator].addr());

    // compute errors
    uint32_t nUciCbErrors        = 0;
    uint32_t nUCiCbMismatches    = 0;  //include payload error, CRC mismatch, and DTX mismatch for UCI-on-PUSCH
    uint32_t nUciCbs             = 0;
    uint32_t nAccumTbs           = 0;
    nHarqUciErrors = 0;
    nCsi1UciErrors = 0;
    nCsi2UciErrors = 0;

    for(int ueIdx = 0; ueIdx < nTbs; ueIdx++)
    {
        if(uePrmsVec[ueIdx].pduBitmap & 2) // check 1st bit for uci
        {
            if(evalEarlyHarqFlag && (pUciOnPuschOutOffsets[ueIdx].isEarlyHarq==1))
            {
                NVLOGC_FMT(NVLOG_PUSCH, "UCI[{}] Early-HARQ Detection Status={}", ueIdx, HarqDetectionStatus[ueIdx]);
            }
            else if(evalEarlyHarqFlag && (pUciOnPuschOutOffsets[ueIdx].isEarlyHarq!=1))
            {
                 NVLOGC_FMT(NVLOG_PUSCH, "UCI[{}] NO Early-HARQ", ueIdx);
            }
            else
            {
                NVLOGC_FMT(NVLOG_PUSCH, "UCI[{}] HARQ Detection Status={}, CSIP1 Detection Status={}, CSIP2 Detection Status={}", ueIdx,
                                              dynApiDataset.DataOut.HarqDetectionStatus[ueIdx],
                                              dynApiDataset.DataOut.CsiP1DetectionStatus[ueIdx],
                                              dynApiDataset.DataOut.CsiP2DetectionStatus[ueIdx]);
            }

            if((!evalEarlyHarqFlag) || (evalEarlyHarqFlag && (pUciOnPuschOutOffsets[ueIdx].isEarlyHarq==1)))
            {
                uint32_t nHarqBytes = ueRefBuffOffsetsVec[ueIdx].nHarqBytes;
                if(nHarqBytes > 0)
                {
                    bool     harqPayloadCheck = true;
                    uint16_t HarqDetectionStatusOffset = pUciOnPuschOutOffsets[ueIdx].HarqDetectionStatusOffset;
                    uint8_t  cuphyHarqDetStatus        = *(HarqDetectionStatus + HarqDetectionStatusOffset);
                    uint8_t  refHarqDetStatus          = *(pRefUciHarqDetStatus+ HarqDetectionStatusOffset - nAccumTbs);
                    bool     mismatchHarqDetStatus     = (cuphyHarqDetStatus != refHarqDetStatus);
                    if(uePrmsVec[ueIdx].pUciPrms->nBitsHarq <= CUPHY_N_MAX_UCI_BITS_RM)
                    {
                        uint16_t refHarqDtxOffset          = cuphyUciDtxTypes_t::N_UCI_DTX*(ueIdx-nAccumTbs)+cuphyUciDtxTypes_t::UCI_HARQ_DTX;
                        uint8_t  refHarqDtx                = *(pRefUciDTXs+refHarqDtxOffset);
                        if(mismatchHarqDetStatus||((cuphyHarqDetStatus==CUPHY_FAPI_DTX)&&(refHarqDtx==0))||((cuphyHarqDetStatus==CUPHY_FAPI_NO_DTX)&&(refHarqDtx==1)))
                        {
                            nUCiCbMismatches += 1;
                            nHarqUciErrors   += 1;
                            harqPayloadCheck = false;
                        }
                        else if((refHarqDetStatus==CUPHY_FAPI_DTX)||(refHarqDtx!=0))
                        {
                            harqPayloadCheck = false;
                        }
                    }
                    else if(uePrmsVec[ueIdx].pUciPrms->nBitsHarq > CUPHY_N_MAX_UCI_BITS_RM)
                    {
                        uint16_t refHarqCrcFlagOffset          = ueRefBuffOffsetsVec[ueIdx].harqCrcFlagOffset;
                        uint16_t cuphyHarqCrcFlagOffset        = pUciOnPuschOutOffsets[ueIdx].harqCrcFlagOffset;
                        if(mismatchHarqDetStatus||(*(pRefUciCrcFlags+refHarqCrcFlagOffset)!=*(pUciCrcFlags+cuphyHarqCrcFlagOffset)))
                        {
                            nUCiCbMismatches += 1;
                            harqPayloadCheck = false;
                        }
                        else if((refHarqDetStatus==CUPHY_FAPI_CRC_FAILURE)||(*(pRefUciCrcFlags+refHarqCrcFlagOffset)!=0))
                        {
                            harqPayloadCheck = false;
                        }
                    }

                    if(harqPayloadCheck)
                    {
                        uint32_t refHarqOffset   = ueRefBuffOffsetsVec[ueIdx].harqPayloadByteOffset;
                        uint32_t cuphyHarqOffset = pUciOnPuschOutOffsets[ueIdx].harqPayloadByteOffset;
                        int cbErrorFlag = memcmp(pRefUciPayloadBytes + refHarqOffset, pUciPayloads + cuphyHarqOffset, nHarqBytes);

                        if(cbErrorFlag != 0)
                        {
                            nUciCbErrors     += 1;
                            nUCiCbMismatches += 1;
                        }
                    }

                    uint32_t refHarqOffset   = ueRefBuffOffsetsVec[ueIdx].harqPayloadByteOffset;
                    uint32_t cuphyHarqOffset = pUciOnPuschOutOffsets[ueIdx].harqPayloadByteOffset;
                    int cbErrorFlag = memcmp(pRefUciPayloadBytes + refHarqOffset, pUciPayloads + cuphyHarqOffset, nHarqBytes);

                    if(cbErrorFlag != 0)
                    {
                        nHarqUciErrors += 1;
                    }

                    nUciCbs++;
                }
            }//if((!evalEarlyHarqFlag) || (evalEarlyHarqFlag && (pUciOnPuschOutOffsets[ueIdx].isEarlyHarq==1)))

            if(!evalEarlyHarqFlag)
            {
                uint32_t nCsi1Bytes = ueRefBuffOffsetsVec[ueIdx].nCsi1Bytes;
                if(nCsi1Bytes > 0)
                {
                    bool     csi1PayloadCheck           = true;
                    uint16_t CsiP1DetectionStatusOffset = pUciOnPuschOutOffsets[ueIdx].CsiP1DetectionStatusOffset;
                    uint8_t  cuphyCsiP1DetStatus        = *(CsiP1DetectionStatus + CsiP1DetectionStatusOffset);
                    uint8_t  refCsiP1DetStatus          = *(pRefUciCsi1DetStatus + CsiP1DetectionStatusOffset - nAccumTbs);
                    bool     mismatchCsiP1DetStatus     = (cuphyCsiP1DetStatus!=refCsiP1DetStatus);
                    if(uePrmsVec[ueIdx].pUciPrms->nBitsCsi1 <= CUPHY_N_MAX_UCI_BITS_RM)
                    {
                        uint16_t refCsiP1DtxOffset          = cuphyUciDtxTypes_t::N_UCI_DTX*(ueIdx-nAccumTbs)+cuphyUciDtxTypes_t::UCI_CSI1_DTX;
                        uint8_t  refCsiP1Dtx                = *(pRefUciDTXs+refCsiP1DtxOffset);
                        if(mismatchCsiP1DetStatus||((cuphyCsiP1DetStatus==CUPHY_FAPI_DTX)&&(refCsiP1Dtx==0))||((cuphyCsiP1DetStatus==CUPHY_FAPI_NO_DTX)&&(refCsiP1Dtx==1)))
                        {
                            nUCiCbMismatches += 1;
                            csi1PayloadCheck = false;
                        }
                        else if((refCsiP1DetStatus==CUPHY_FAPI_DTX)||(refCsiP1Dtx!=0))
                        {
                            csi1PayloadCheck = false;
                        }

                    }
                    else if(uePrmsVec[ueIdx].pUciPrms->nBitsCsi1 > CUPHY_N_MAX_UCI_BITS_RM)
                    {
                        uint16_t refCsi1CrcFlagOffset          = ueRefBuffOffsetsVec[ueIdx].csi1CrcFlagOffset;
                        uint16_t cuphyCsi1CrcFlagOffset        = pUciOnPuschOutOffsets[ueIdx].csi1CrcFlagOffset;
                        if(mismatchCsiP1DetStatus||(*(pRefUciCrcFlags+refCsi1CrcFlagOffset)!=*(pUciCrcFlags+cuphyCsi1CrcFlagOffset)))
                        {
                            nUCiCbMismatches += 1;
                            csi1PayloadCheck = false;
                        }
                        else if((refCsiP1DetStatus==CUPHY_FAPI_CRC_FAILURE)||(*(pRefUciCrcFlags+refCsi1CrcFlagOffset)!=0))
                        {
                            csi1PayloadCheck = false;
                        }
                    }

                    if(csi1PayloadCheck)
                    {
                        uint32_t refCsi1Offset   = ueRefBuffOffsetsVec[ueIdx].csi1PayloadByteOffset;
                        uint32_t cuphyCsi1Offset = pUciOnPuschOutOffsets[ueIdx].csi1PayloadByteOffset;
                        int cbErrorFlag = memcmp(pRefUciPayloadBytes + refCsi1Offset, pUciPayloads + cuphyCsi1Offset, nCsi1Bytes);
                        if(cbErrorFlag != 0)
                        {
                            nUciCbErrors     += 1;
                            nUCiCbMismatches += 1;
                        }
                    }

                    uint32_t refCsi1Offset   = ueRefBuffOffsetsVec[ueIdx].csi1PayloadByteOffset;
                    uint32_t cuphyCsi1Offset = pUciOnPuschOutOffsets[ueIdx].csi1PayloadByteOffset;
                    int cbErrorFlag = memcmp(pRefUciPayloadBytes + refCsi1Offset, pUciPayloads + cuphyCsi1Offset, nCsi1Bytes);
                    if(cbErrorFlag != 0)
                    {
                        nCsi1UciErrors += 1;
                    }


                    nUciCbs++;
                }

                uint32_t nCsi2Bytes = ueRefBuffOffsetsVec[ueIdx].nCsi2Bytes;
                if(nCsi2Bytes > 0)
                {
                    bool csi2PayloadCheck = true;
                    uint16_t CsiP2DetectionStatusOffset = pUciOnPuschOutOffsets[ueIdx].CsiP2DetectionStatusOffset;
                    uint16_t refCsiP2DtxOffset          = cuphyUciDtxTypes_t::N_UCI_DTX*(ueIdx-nAccumTbs)+cuphyUciDtxTypes_t::UCI_CSI2_DTX;
                    uint8_t  cuphyCsiP2DetStatus        = *(CsiP2DetectionStatus + CsiP2DetectionStatusOffset);
                    uint8_t  refCsiP2DetStatus          = *(pRefUciCsi2DetStatus + CsiP2DetectionStatusOffset - nAccumTbs);
                    uint8_t  refCsiP2Dtx                = *(pRefUciDTXs+refCsiP2DtxOffset);
                    bool     mismatchCsi2DetStatus      = (cuphyCsiP2DetStatus != refCsiP2DetStatus);
                    if(mismatchCsi2DetStatus||((cuphyCsiP2DetStatus==CUPHY_FAPI_DTX)&&(refCsiP2Dtx==0))||((cuphyCsiP2DetStatus==CUPHY_FAPI_NO_DTX)&&(refCsiP2Dtx==1)))
                    {
                        nUCiCbMismatches += 1;
                        csi2PayloadCheck = false;
                    }
                    else if((refCsiP2DetStatus==CUPHY_FAPI_DTX)||(refCsiP2Dtx!=0))
                    {
                        csi2PayloadCheck = false;
                    }


                    if(csi2PayloadCheck)
                    {
                        uint32_t refCsi2Offset   = ueRefBuffOffsetsVec[ueIdx].csi2PayloadByteOffset;
                        uint32_t cuphyCsi2Offset = pUciOnPuschOutOffsets[ueIdx].csi2PayloadByteOffset;
                        int cbErrorFlag = memcmp(pRefUciPayloadBytes + refCsi2Offset, pUciPayloads + cuphyCsi2Offset, nCsi2Bytes);
                        if(cbErrorFlag != 0)
                        {
                            nUciCbErrors     += 1;
                            nUCiCbMismatches += 1;
                        }
                    }


                    uint32_t refCsi2Offset   = ueRefBuffOffsetsVec[ueIdx].csi2PayloadByteOffset;
                    uint32_t cuphyCsi2Offset = pUciOnPuschOutOffsets[ueIdx].csi2PayloadByteOffset;
                    int cbErrorFlag = memcmp(pRefUciPayloadBytes + refCsi2Offset, pUciPayloads + cuphyCsi2Offset, nCsi2Bytes);
                    if(cbErrorFlag != 0)
                    {
                        nCsi2UciErrors   += 1;
                    }
                    nUciCbs++;
                }
            }//if(!evalEarlyHarqFlag)
        }

        if(++oldTbsIdx == nTbsInFileVec[iterator])
        {
            if (nUciCbs > 0) 
            {   if((nUciCbErrors>0)||(nUCiCbMismatches>0))
                {
                    NVLOGF_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "Cell # {} : Metric - UCI Block Error Rate           : {:4.4f} (Error CBs {}, Mismatched CBs {}, Total CBs {})",
                        iterator,
                        static_cast<float>(nUciCbErrors) / static_cast<float>(nUciCbs),
                        nUciCbErrors,
                        nUCiCbMismatches,
                        nUciCbs);
                }
                else
                {
                    NVLOGC_FMT(NVLOG_PUSCH, "Cell # {} : Metric - UCI Block Error Rate           : {:4.4f} (Error CBs {}, Mismatched CBs {}, Total CBs {})",
                        iterator,
                        static_cast<float>(nUciCbErrors) / static_cast<float>(nUciCbs),
                        nUciCbErrors,
                        nUCiCbMismatches,
                        nUciCbs);
                }
            }
            else {
                NVLOGC_FMT(NVLOG_PUSCH, "Cell # {} : No UCI",
                        iterator);
            }
            ++iterator;
            nAccumTbs    += oldTbsIdx;
            oldTbsIdx    = 0;

            nUciCbErrors        = 0;
            nUCiCbMismatches    = 0;
            nUciCbs             = 0;
            if(iterator < nTbsInFileVec.size())
            {
                pRefUciPayloadBytes = static_cast<uint8_t*>(tRefUciPayloadBytesVec[iterator].addr());
                pRefUciCrcFlags = static_cast<uint8_t*>(tRefUciCrcFlagsVec[iterator].addr());
                pRefUciDTXs = static_cast<uint8_t*>(tRefUciDTXsVec[iterator].addr());
                pRefUciHarqDetStatus = static_cast<uint8_t*>(tRefUciHarqDetStatusVec[iterator].addr());
                pRefUciCsi1DetStatus = static_cast<uint8_t*>(tRefUciCsi1DetStatusVec[iterator].addr());
                pRefUciCsi2DetStatus = static_cast<uint8_t*>(tRefUciCsi2DetStatusVec[iterator].addr());
            }
        }
    }
}

//-------------------------------------------------------------------------------------
// function evaluates pusch rate-matching

void EvalDataset::evalPuschRm(void** pRmOutLLRAddrsGpu, const PerTbParams* pTbPrmsCpu, cudaStream_t cuStrm)
{
    for(int ueIdx = 0; ueIdx < nTbs; ++ueIdx)
    {
        // copy Rm LLRs to CPU
        uint32_t                                   nRmLLRs = (pTbPrmsCpu[ueIdx].Ncb + 2 * pTbPrmsCpu[ueIdx].Zc) * pTbPrmsCpu[ueIdx].num_CBs;
        cuphy::buffer<__half, cuphy::pinned_alloc> rmOutLLRsCpu(nRmLLRs);
        CUDA_CHECK(cudaMemcpyAsync(static_cast<void*>(rmOutLLRsCpu.addr()), pRmOutLLRAddrsGpu[ueIdx], sizeof(__half) * nRmLLRs, cudaMemcpyDeviceToHost, cuStrm));
        CUDA_CHECK(cudaStreamSynchronize(cuStrm));

        // compare cuphy output to reference
        uint32_t nMismatches = 0;
        uint32_t nLLRs       = 0;

        for(int rmLLRidx = 0; rmLLRidx < nRmLLRs; ++rmLLRidx)
        {
            nLLRs += 1;
            double LLR_ref   = static_cast<double>(rmOutLLRsRef[ueIdx](rmLLRidx));
            double LLR_cuphy = static_cast<double>(rmOutLLRsCpu[rmLLRidx]);

            double error = std::pow(std::abs(LLR_ref - LLR_cuphy), 2);
            // if LLR_cuphy is computed as sum of multiple llrs, we can expect deviation from reference values. Since in cuPHY fp16 is used, but ref values are accumulated as double,
            // then the final sum is converted to fp16
            // non-zero tol helps with TC7254, TC7445-8, TC7450-52, TC7470,73,87,88,89 among others
            double tol = 0.01 * LLR_ref * LLR_ref;
            if(error > tol)
            {
                NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "Mismatch detected for User {}. rmLLRidx = {},  LLR_ref = {},  LLR_cuphy = {}", ueIdx, rmLLRidx, LLR_ref, LLR_cuphy);
                nMismatches += 1;
            }
        }

        NVLOGC_FMT(NVLOG_PUSCH, "detected {} mismatches out of {} rateMatchedLLRs", nMismatches, nLLRs);
    }
}

//-------------------------------------------------------------------------------------
// function compare cuphy computation of uci rate-match seq lengths to reference

void EvalDataset::evalUciRmSizes(PerTbParams* pTbPrmsCuphy, PerTbParams* pTbPrmsRef, cuphyPuschUePrm_t* pUePuschPrms, uint16_t nUes)
{
    uint16_t nMismatches = 0;
    uint16_t nChecks     = 0;

    NVLOGC_FMT(NVLOG_PUSCH, "Comparing cuphy computed rm sequence lengths to reference....");
    for(int ueIdx = 0; ueIdx < nUes; ++ueIdx)
    {
        if(pUePuschPrms[ueIdx].pduBitmap & 3)
        {
            nChecks += 4;
            if(pTbPrmsCuphy[ueIdx].G != pTbPrmsRef[ueIdx].G)
            {
                NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "ue{} G mismatch.           cuphy = {}, ref = {}.", ueIdx, pTbPrmsCuphy[ueIdx].G, pTbPrmsRef[ueIdx].G);
                nMismatches += 1;
            }
            if(pTbPrmsCuphy[ueIdx].G_harq != pTbPrmsRef[ueIdx].G_harq)
            {
                NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "ue{} G_harq mismatch.      cuphy = {}, ref = {}.", ueIdx, pTbPrmsCuphy[ueIdx].G_harq, pTbPrmsRef[ueIdx].G_harq);
                nMismatches += 1;
            }
            if(pTbPrmsCuphy[ueIdx].G_csi1 != pTbPrmsRef[ueIdx].G_csi1)
            {
                NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "ue{} G_csi1 mismatch.      cuphy = {}, ref = {}.", ueIdx, pTbPrmsCuphy[ueIdx].G_csi1, pTbPrmsRef[ueIdx].G_csi1);
                nMismatches += 1;
            }
            if(pTbPrmsCuphy[ueIdx].G_harq_rvd != pTbPrmsRef[ueIdx].G_harq_rvd)
            {
                NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "ue{} G_harq_rvd mismatch. G_harq_rvd = {}, G_harq_rvd = {}.", ueIdx, pTbPrmsCuphy[ueIdx].G_harq_rvd, pTbPrmsRef[ueIdx].G_harq_rvd);
                nMismatches += 1;
            }
        }

        NVLOGC_FMT(NVLOG_PUSCH, "detected {} mismatches out of {} checks", nMismatches, nChecks);
    }
}

//-------------------------------------------------------------------------------------
// function compare cuphy computation uciOnPusch schLLRs, csi1LLRs, and harqLLRs to reference

void EvalDataset::evalUciOnPuschSegLLRs1(uint16_t nUciUes, uint16_t* pUciUserIdxs, PerTbParams* pTbPrmsCpu, cudaStream_t cuStrm)
{
    uint32_t maxNumLLRs = 273 * 12 * 14 * 4;

    cuphy::buffer<__half, cuphy::pinned_alloc> schLLRsCuphy(maxNumLLRs);
    cuphy::buffer<__half, cuphy::pinned_alloc> csi1LLRsCuphy(maxNumLLRs);
    cuphy::buffer<__half, cuphy::pinned_alloc> harqLLRsCuphy(maxNumLLRs);

    NVLOGC_FMT(NVLOG_PUSCH, "Comparing cuPHY uciOnPusch deScram + segmented LLRs to reference...");
    uint16_t nSch         = 0;
    uint16_t nCsi1        = 0;
    uint16_t nHarq        = 0;
    int      mismatchFlag = 0;
    for(int i = 0; i < nUciUes; ++i)
    {
        uint16_t ueIdx    = pUciUserIdxs[i];
        uint32_t G        = pTbPrmsCpu[ueIdx].G;
        uint32_t G_csi1   = pTbPrmsCpu[ueIdx].G_csi1;
        uint32_t G_harq   = pTbPrmsCpu[ueIdx].G_harq;
        uint8_t  csi2Flag = pTbPrmsCpu[ueIdx].csi2Flag;

        __half* d_schAndCsi2LLRs = pTbPrmsCpu[ueIdx].d_schAndCsi2LLRs;
        __half* d_csi1LLRs       = pTbPrmsCpu[ueIdx].d_csi1LLRs;
        __half* d_harqLLrs       = pTbPrmsCpu[ueIdx].d_harqLLrs;

        if((G > 0) && (csi2Flag == 0))
        {
            CUDA_CHECK(cudaMemcpyAsync(schLLRsCuphy.addr(), d_schAndCsi2LLRs, G * sizeof(__half), cudaMemcpyDeviceToHost, cuStrm));
            CUDA_CHECK(cudaStreamSynchronize(cuStrm));

            double signalEnergy = 0;
            double errorEnergy  = 0;

            for(int LLRidx = 0; LLRidx < G; ++LLRidx)
            {
                double cuphyLLR = static_cast<double>(schLLRsCuphy[LLRidx]);
                double refLLR   = static_cast<double>(schLLRsRef[nSch](LLRidx));

                signalEnergy += pow(abs(refLLR), 2);
                errorEnergy += pow(abs(refLLR - cuphyLLR), 2);
            }

            if(errorEnergy > 0)
            {
                double snr = 10 * log10(signalEnergy / errorEnergy);
                if(snr < 50)
                {
                    NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "mismatch detected for user {} sch LLRs", ueIdx);
                    mismatchFlag = 1;
                }
            }
            nSch += 1;
        }

        if(G_csi1 > 0)
        {
            CUDA_CHECK(cudaMemcpyAsync(csi1LLRsCuphy.addr(), d_csi1LLRs, G_csi1 * sizeof(__half), cudaMemcpyDeviceToHost, cuStrm));
            CUDA_CHECK(cudaStreamSynchronize(cuStrm));

            double signalEnergy = 0;
            double errorEnergy  = 0;

            for(int LLRidx = 0; LLRidx < G_csi1; ++LLRidx)
            {
                double cuphyLLR = static_cast<double>(csi1LLRsCuphy[LLRidx]);
                double refLLR   = static_cast<double>(csi1LLRsRef[nCsi1](LLRidx));

                signalEnergy += pow(abs(refLLR), 2);
                errorEnergy += pow(abs(refLLR - cuphyLLR), 2);
            }

            if(errorEnergy > 0)
            {
                double snr = 10 * log10(signalEnergy / errorEnergy);
                if(snr < 50)
                {
                    NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "mismatch detected for user {} csi1 LLRs", ueIdx);
                    mismatchFlag = 1;
                }
            }
            nCsi1 += 1;
        }

        if(G_harq > 0)
        {
            CUDA_CHECK(cudaMemcpyAsync(harqLLRsCuphy.addr(), d_harqLLrs, G_harq * sizeof(__half), cudaMemcpyDeviceToHost, cuStrm));
            CUDA_CHECK(cudaStreamSynchronize(cuStrm));

            double signalEnergy = 0;
            double errorEnergy  = 0;

            for(int LLRidx = 0; LLRidx < G_harq; ++LLRidx)
            {
                double cuphyLLR = static_cast<double>(harqLLRsCuphy[LLRidx]);
                double refLLR   = static_cast<double>(harqLLRsRef[nHarq](LLRidx));

                signalEnergy += pow(abs(refLLR), 2);
                errorEnergy += pow(abs(refLLR - cuphyLLR), 2);
            }

            if(errorEnergy > 0)
            {
                double snr = 10 * log10(signalEnergy / errorEnergy);
                if(snr < 50)
                {
                    NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "mismatch detected for user {} harq LLRs", ueIdx);
                    mismatchFlag = 1;
                }
            }
            nHarq += 1;
        }
    }

    if(mismatchFlag == 0)
    {
        NVLOGC_FMT(NVLOG_PUSCH, "no mismatches detected");
    }
}

//-------------------------------------------------------------------------------------
// function compare cuphy descrambled + segmented LLRs for CSI-P2 users to reference

void EvalDataset::evalUciOnPuschSegLLRs2(uint16_t nCsi2Ues, uint16_t* pCsi2UserIdxs, PerTbParams* pTbPrmsCpu, cudaStream_t cuStrm)
{
    uint32_t maxNumLLRs = 273 * 12 * 14 * 4;

    cuphy::buffer<__half, cuphy::pinned_alloc> schLLRsCuphy(maxNumLLRs);
    cuphy::buffer<__half, cuphy::pinned_alloc> csi2LLRsCuphy(maxNumLLRs);

    NVLOGC_FMT(NVLOG_PUSCH, "Comparing cuPHY CSI-P2 deScram + segmented LLRs to reference...");
    int mismatchFlag = 0;
    int nSch         = 0;
    int nCsi2        = 0;

    for(int i = 0; i < nCsi2Ues; ++i)
    {
        uint16_t ueIdx       = pCsi2UserIdxs[i];
        uint32_t G           = pTbPrmsCpu[ueIdx].G;
        uint32_t G_csi2      = pTbPrmsCpu[ueIdx].G_csi2;
        uint8_t  nBitsPerQam = pTbPrmsCpu[ueIdx].Qm;
        uint16_t nBitsCsi2   = pTbPrmsCpu[ueIdx].nBitsCsi2;

        __half* d_schLLRs  = pTbPrmsCpu[ueIdx].d_schAndCsi2LLRs;
        __half* d_csi2LLRs = pTbPrmsCpu[ueIdx].d_schAndCsi2LLRs + G;

        if(G > 0)
        {
            CUDA_CHECK(cudaMemcpyAsync(schLLRsCuphy.addr(), d_schLLRs, G * sizeof(__half), cudaMemcpyDeviceToHost, cuStrm));
            CUDA_CHECK(cudaStreamSynchronize(cuStrm));

            double signalEnergy = 0;
            double errorEnergy  = 0;

            for(int LLRidx = 0; LLRidx < G; ++LLRidx)
            {
                double cuphyLLR = static_cast<double>(schLLRsCuphy[LLRidx]);
                double refLLR   = static_cast<double>(schLLRsRef[nSch](LLRidx));

                signalEnergy += pow(abs(refLLR), 2);
                errorEnergy += pow(abs(refLLR - cuphyLLR), 2);
            }

            if(errorEnergy > 0)
            {
                double snr = 10 * log10(signalEnergy / errorEnergy);
                if(snr < 50)
                {
                    NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "mismatch detected for user {} sch LLRs", ueIdx);
                    mismatchFlag = 1;
                }
            }
            nSch += 1;
        }

        if(G_csi2 > 0)
        {
            CUDA_CHECK(cudaMemcpyAsync(csi2LLRsCuphy.addr(), d_csi2LLRs, G_csi2 * sizeof(__half), cudaMemcpyDeviceToHost, cuStrm));
            CUDA_CHECK(cudaStreamSynchronize(cuStrm));

            double signalEnergy = 0;
            double errorEnergy  = 0;

            for(int LLRidx = 0; LLRidx < G_csi2; ++LLRidx)
            {
                bool llrCheckFlag = false;
                if((nBitsCsi2 == 1) && (nBitsPerQam > 1))
                {
                    uint32_t r = LLRidx % nBitsPerQam;
                    if(r < 2)
                    {
                        llrCheckFlag = true;
                    }
                }
                else
                {
                    llrCheckFlag = true;
                }

                if(llrCheckFlag)
                {
                    double cuphyLLR = static_cast<double>(csi2LLRsCuphy[LLRidx]);
                    double refLLR   = static_cast<double>(csi2LLRsRef[nCsi2](LLRidx));

                    signalEnergy += pow(abs(refLLR), 2);
                    errorEnergy += pow(abs(refLLR - cuphyLLR), 2);
                }
            }

            if(errorEnergy > 0)
            {
                double snr = 10 * log10(signalEnergy / errorEnergy);
                if(snr < 50)
                {
                    NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "mismatch detected for user {} csi2 LLRs", ueIdx);
                    mismatchFlag = 1;
                }
            }
            nCsi2 += 1;
        }
    }

    if(mismatchFlag == 0)
    {
        NVLOGC_FMT(NVLOG_PUSCH, "no mismatches detected");
    }
}

void EvalDataset::evalUciOnPuschCsi2Ctrl(PerTbParams* pTbPrmsGpu, cudaStream_t cuStrm)
{
    // Copy tbParms to CPU:
    cuphy::buffer<PerTbParams, cuphy::pinned_alloc> tbPrmsCpu(nUes);
    CUDA_CHECK(cudaMemcpyAsync(tbPrmsCpu.addr(), pTbPrmsGpu, nUes * sizeof(PerTbParams), cudaMemcpyDeviceToHost, cuStrm));
    CUDA_CHECK(cudaStreamSynchronize(cuStrm));

    NVLOGC_FMT(NVLOG_PUSCH, "Comparing cuPHY computed G, G_csi2, and nBitsCsi2 to reference...");
    uint8_t mismatchFlag = 0;
    for(int csi2Idx = 0; csi2Idx < nCsi2Ues; ++csi2Idx)
    {
        uint16_t ueIdx           = csi2UeIdxsVec[csi2Idx];
        uint32_t G_ref           = uciSizesVec[ueIdx].G;
        uint32_t G_cuphy         = tbPrmsCpu[ueIdx].G;
        uint32_t G_csi2_ref      = uciSizesVec[ueIdx].G_csi2;
        uint32_t G_csi2_cuphy    = tbPrmsCpu[ueIdx].G_csi2;
        uint16_t nBitsCsi2_ref   = uciSizesVec[ueIdx].nBitsCsi2;
        uint16_t nBitsCsi2_cuphy = tbPrmsCpu[ueIdx].nBitsCsi2;

        if((G_ref != G_cuphy) || (G_csi2_ref != G_csi2_cuphy) || (nBitsCsi2_ref != nBitsCsi2_cuphy))
        {
            mismatchFlag = 1;
            NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "mismatch detected for ueIdx = {}:", ueIdx);

            if(G_ref != G_cuphy)
            {
                NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, " G_cuphy         = {},  G_ref         = {}", G_ref, G_cuphy);
            }
            if(G_csi2_ref != G_csi2_cuphy)
            {
                NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, " G_csi2_cuphy    = {},   G_csi2_ref    = {}", G_csi2_ref, G_csi2_cuphy);
            }
            if(nBitsCsi2_ref != nBitsCsi2_cuphy)
            {
                NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, " nBitsCsi2_cuphy = {},      nBitsCsi2_ref = {}", nBitsCsi2_ref, nBitsCsi2_cuphy);
            }
        }
    }

    if(mismatchFlag == 0)
    {
        NVLOGC_FMT(NVLOG_PUSCH, "no mistmatched detected!");
    }
}

void EvalDataset::reportPuschCrcErrors(cuphyPuschDynPrms_t const& dynPrms)
{
    uint16_t                 nUes               = dynPrms.pCellGrpDynPrm->nUes;
    cuphyPuschUePrm_t const* pUePrms            = dynPrms.pCellGrpDynPrm->pUePrms;
    uint32_t                 totNumCbs          = dynPrms.pDataOut->totNumCbs;
    uint32_t*                pCbCrcs            = dynPrms.pDataOut->pCbCrcs;
    uint32_t*                pTbCrcs            = dynPrms.pDataOut->pTbCrcs;
    uint32_t*                pStartOffsetsCbCrc = dynPrms.pDataOut->pStartOffsetsCbCrc;
    uint32_t*                pStartOffsetsTbCrc = dynPrms.pDataOut->pStartOffsetsTbCrc;

    uint8_t crcFailFlag = 0;
    for(int ueIdx = 0; ueIdx < nUes; ++ueIdx)
    {
        if(pUePrms[ueIdx].pduBitmap & 1) // check 0th bit for SCH transmission
        {
            uint32_t tbCrcOffset = pStartOffsetsTbCrc[ueIdx];
            if(pTbCrcs[tbCrcOffset] != 0)
            {
                crcFailFlag = 1;
                NVLOGC_FMT(NVLOG_PUSCH, "TB CRC fail for ueIdx = {}", ueIdx);
            }

            uint32_t cbCrcOffset = pStartOffsetsCbCrc[ueIdx];
            uint16_t nCbs        = (ueIdx == (nUes - 1)) ? (totNumCbs - cbCrcOffset) : (pStartOffsetsCbCrc[ueIdx + 1] - cbCrcOffset);
            for(int cbIdx = 0; cbIdx < nCbs; ++cbIdx)
            {
                if(pCbCrcs[cbIdx + cbCrcOffset] != 0)
                {
                    crcFailFlag = 1;
                    NVLOGC_FMT(NVLOG_PUSCH, "TB CRC fail for ueIdx = {}, cbIdx = {}", ueIdx, cbIdx);
                }
            }
        }
    }

    if(crcFailFlag == 0)
    {
        NVLOGC_FMT(NVLOG_PUSCH, "No PUSCH CRC errors detected!");
    }
}

/******************************************************************/ /**
 * \brief Helper to populate UCI parameters
 *
 * Copies UCI parameters from an HDF5 dataset into a specific UCI parameter struct for use by the PUCCH pipeline
 *
 * \param uciPrmsH5 - Reference to HDF5 struct containing UCI parameters
 * \param uciPrms   - Reference to PUCCH UCI struct to store UCI parameters for use in PUCCH pipeline
 * \param cellIdx   - Cell index number
 * \param uciIdx    - UCI index number
 * \param pucchFmt  - PUCCH Format number used to account for parameters that are handled differently for each format
 *
 */
void pucchDynApiDataset::populateUciParams(cuphy::cuphyHDF5_struct& uciPrmsH5, cuphyPucchUciPrm_t& uciPrms, int cellIdx, int uciIdx, uint8_t pucchFmt)
{
    uciPrms.cellPrmDynIdx        = cellIdx;
    uciPrms.uciOutputIdx         = uciIdx;
    uciPrms.rnti                 = uciPrmsH5.get_value_as<uint16_t>("rnti");
    try
    {
        uciPrms.bwpStart             = uciPrmsH5.get_value_as<uint16_t>("BWPStart");
    }
    catch(const cuphy::cuphyHDF5_exception& e)
    {
        NVLOGW_FMT(NVLOG_PUCCH, "TV is missing BWPStart field for uciIdx {}.  Defaulting BWPStart to 0.", uciIdx);
        uciPrms.bwpStart             = 0;
    }
    uciPrms.startPrb             = uciPrmsH5.get_value_as<uint16_t>("startPrb");
    uciPrms.startSym             = uciPrmsH5.get_value_as<uint8_t>("startSym");
    uciPrms.nSym                 = uciPrmsH5.get_value_as<uint8_t>("nSym");
    uciPrms.freqHopFlag          = uciPrmsH5.get_value_as<uint8_t>("freqHopFlag");
    uciPrms.secondHopPrb         = uciPrmsH5.get_value_as<uint16_t>("secondHopPrb");
    uciPrms.bitLenHarq           = uciPrmsH5.get_value_as<uint16_t>("bitLenHarq");
    uciPrms.DTXthreshold         = uciPrmsH5.get_value_as<float>("DTXthreshold");
    uciPrms.formatType           = pucchFmt;
    uciPrms.nUplinkStreams       = 1; //TODO
    // uciPrms.uciP1P2Crpd_t        = nullptr;
    switch(pucchFmt){
        case 0:
        case 1:
            uciPrms.multiSlotTxIndicator = uciPrmsH5.get_value_as<uint8_t>("multiSlotTxIndicator");
            uciPrms.pi2Bpsk              = uciPrmsH5.get_value_as<uint8_t>("pi2Bpsk");
            uciPrms.groupHopFlag         = uciPrmsH5.get_value_as<uint8_t>("groupHopFlag");
            uciPrms.sequenceHopFlag      = uciPrmsH5.get_value_as<uint8_t>("sequenceHopFlag");
            uciPrms.initialCyclicShift   = uciPrmsH5.get_value_as<uint16_t>("initialCyclicShift");
            uciPrms.timeDomainOccIdx     = uciPrmsH5.get_value_as<uint8_t>("timeDomainOccIdx");
            uciPrms.srFlag               = uciPrmsH5.get_value_as<uint8_t>("srFlag");
            uciPrms.maxCodeRate          = 0;
            break;
        case 2:
            uciPrms.prbSize          = uciPrmsH5.get_value_as<uint8_t>("prbSize");
            uciPrms.bitLenSr         = uciPrmsH5.get_value_as<uint16_t>("bitLenSr");
            uciPrms.bitLenCsiPart1   = uciPrmsH5.get_value_as<uint8_t>("bitLenCsiPart1");
            uciPrms.dataScramblingId = uciPrmsH5.get_value_as<uint8_t>("dataScramblingId");
            uciPrms.DmrsScramblingId = uciPrmsH5.get_value_as<uint8_t>("DmrsScramblingId");
            uciPrms.maxCodeRate      = uciPrmsH5.get_value_as<uint8_t>("maxCodeRate");
            break;
        case 3:
            uciPrms.pi2Bpsk          = uciPrmsH5.get_value_as<uint8_t>("pi2Bpsk");
            uciPrms.groupHopFlag     = uciPrmsH5.get_value_as<uint8_t>("groupHopFlag");
            uciPrms.sequenceHopFlag  = uciPrmsH5.get_value_as<uint8_t>("sequenceHopFlag");
            uciPrms.prbSize          = uciPrmsH5.get_value_as<uint8_t>("prbSize");
            uciPrms.bitLenSr         = uciPrmsH5.get_value_as<uint16_t>("bitLenSr");
            uciPrms.bitLenCsiPart1   = uciPrmsH5.get_value_as<uint8_t>("bitLenCsiPart1");
            uciPrms.AddDmrsFlag      = uciPrmsH5.get_value_as<uint8_t>("AddDmrsFlag");
            uciPrms.dataScramblingId = uciPrmsH5.get_value_as<uint8_t>("dataScramblingId");
            uciPrms.maxCodeRate      = uciPrmsH5.get_value_as<uint8_t>("maxCodeRate");
            uciPrms.uciP1P2Crpd_t.numPart2s = uciPrmsH5.get_value_as<uint16_t>("numPart2s");
            break;
        default:
            std::string error_msg = "Format " + std::to_string(pucchFmt) + " is not a supported PUCCH Format!\n";
            throw std::runtime_error(error_msg.c_str());
    }
}

//---------------------------------------------------------------------------------------------------
// Dataset holds dynamic PUCCH api parameters/data

// Construct dataset from h5 file
pucchDynApiDataset::pucchDynApiDataset(const std::vector<std::string>& inputFileNameVec, cudaStream_t cuStrm, uint64_t procMode)
{
    int nCells = static_cast<int>(inputFileNameVec.size());
    cellDynPrm.resize(nCells);
    tDataRxVec.resize(nCells);
    tPrmDataRxVec.resize(nCells);

    cellGrpDynPrm.nF0Ucis = 0;
    cellGrpDynPrm.nF1Ucis = 0;
    cellGrpDynPrm.nF2Ucis = 0;
    cellGrpDynPrm.nF3Ucis = 0;

    // loop through the cells
    for(int i = 0; i < nCells; i++)
    {
        // load 5h file
        hdf5hpp::hdf5_file      fInput          = hdf5hpp::hdf5_file::open(inputFileNameVec[i].c_str());
        cuphy::cuphyHDF5_struct cellDynPrmH5    = cuphy::get_HDF5_struct(fInput, "cellDynPrm");
        cuphy::cuphyHDF5_struct cellGrpDynPrmH5 = cuphy::get_HDF5_struct(fInput, "cellGrpDynPrm");

        // cell parameters
        cellDynPrm[i].cellPrmStatIdx = i;
        cellDynPrm[i].cellPrmDynIdx  = i;
        cellDynPrm[i].slotNum        = cellDynPrmH5.get_value_as<uint16_t>("slotNum");
        cellDynPrm[i].pucchHoppingId = cellDynPrmH5.get_value_as<uint16_t>("pucchHoppingId");

        //------------------------------
        // F0 uci parameters
        try
        {
            uint16_t              nF0UcisTmp  = cellGrpDynPrmH5.get_value_as<uint16_t>("nF0Ucis");
            hdf5hpp::hdf5_dataset F0UciPrmsH5 = fInput.open_dataset("F0UciPrms");
            uint16_t              offset      = static_cast<uint16_t>(F0UciPrmsVec.size());
            uint16_t              nF0Ucis     = offset + nF0UcisTmp;
            F0UciPrmsVec.resize(nF0Ucis);
            for(uint16_t uciIdx = 0; uciIdx < nF0UcisTmp; ++uciIdx)
            {
                cuphy::cuphyHDF5_struct uciPrmsH5 = cuphy::get_HDF5_struct_index(F0UciPrmsH5, uciIdx);

                populateUciParams(uciPrmsH5, F0UciPrmsVec[uciIdx + offset], i, uciIdx + offset, 0);
            }

            if(i == (nCells - 1))
            {
                cellGrpDynPrm.nF0Ucis    = nF0Ucis;
                cellGrpDynPrm.pF0UciPrms = F0UciPrmsVec.data();

        // output parameters
                if(nF0Ucis)
                {
                    bF0UciOut = std::move(cuphy::buffer<cuphyPucchF0F1UciOut_t, cuphy::pinned_alloc>(nF0Ucis));
                }
                DataOut.pF0UcisOut = bF0UciOut.addr();
            }
        }
        catch(const cuphy::cuphyHDF5_exception& e)
        {
            NVLOGW_FMT(NVLOG_PUCCH, "no PF0 Uci parameter object found in TV HDF5 file");
        }

        // F1 uci parameters
        try
        {
            uint16_t              nF1UcisTmp  = cellGrpDynPrmH5.get_value_as<uint16_t>("nF1Ucis");
            uint16_t              offset      = static_cast<uint16_t>(F1UciPrmsVec.size());
            uint16_t              nF1Ucis     = offset + nF1UcisTmp;
            hdf5hpp::hdf5_dataset F1UciPrmsH5 = fInput.open_dataset("F1UciPrms");
            F1UciPrmsVec.resize(nF1Ucis);
            for(uint16_t uciIdx = 0; uciIdx < nF1UcisTmp; ++uciIdx)
            {
                cuphy::cuphyHDF5_struct uciPrmsH5 = cuphy::get_HDF5_struct_index(F1UciPrmsH5, uciIdx);
                populateUciParams(uciPrmsH5, F1UciPrmsVec[uciIdx + offset], i, uciIdx + offset, 1);
            }

            if(i == (nCells - 1))
            {
                cellGrpDynPrm.nF1Ucis    = nF1Ucis;
                cellGrpDynPrm.pF1UciPrms = F1UciPrmsVec.data();

                if(nF1Ucis)
                {
                    bF1UciOut = std::move(cuphy::buffer<cuphyPucchF0F1UciOut_t, cuphy::pinned_alloc>(nF1Ucis));
                }
                DataOut.pF1UcisOut = bF1UciOut.addr();
            }
        }
        catch(const cuphy::cuphyHDF5_exception& e)
        {
            NVLOGW_FMT(NVLOG_PUCCH, "no PF1 Uci parameter object found in TV HDF5 file");
        }

        // F2 uci parameters
        try
        {
            uint16_t              nF2UcisTmp  = cellGrpDynPrmH5.get_value_as<uint16_t>("nF2Ucis");
            uint16_t              offset      = static_cast<uint16_t>(F2UciPrmsVec.size());
            uint16_t              nF2Ucis     = offset + nF2UcisTmp;
            hdf5hpp::hdf5_dataset F2UciPrmsH5 = fInput.open_dataset("F2UciPrms");
            F2UciPrmsVec.resize(nF2Ucis);
            for(uint16_t uciIdx = 0; uciIdx < nF2UcisTmp; ++uciIdx)
            {
                cuphy::cuphyHDF5_struct uciPrmsH5 = cuphy::get_HDF5_struct_index(F2UciPrmsH5, uciIdx);
                populateUciParams(uciPrmsH5, F2UciPrmsVec[uciIdx + offset], i, uciIdx + offset, 2);
            }

            if(i == (nCells - 1))
            {
                cellGrpDynPrm.nF2Ucis    = nF2Ucis;
                cellGrpDynPrm.pF2UciPrms = F2UciPrmsVec.data();
            }
        }
        catch(const cuphy::cuphyHDF5_exception& e)
        {
            NVLOGW_FMT(NVLOG_PUCCH, "no PF2 Uci parameter object found in TV HDF5 file");
        }

        // F3 uci parameters
        try
        {
            uint16_t              nF3UcisTmp  = cellGrpDynPrmH5.get_value_as<uint16_t>("nF3Ucis");
            uint16_t              offset      = static_cast<uint16_t>(F3UciPrmsVec.size());
            uint16_t              nF3Ucis     = offset + nF3UcisTmp;
            hdf5hpp::hdf5_dataset F3UciPrmsH5 = fInput.open_dataset("F3UciPrms");
            F3UciPrmsVec.resize(nF3Ucis);
            for(uint16_t uciIdx = 0; uciIdx < nF3UcisTmp; ++uciIdx)
            {
                cuphy::cuphyHDF5_struct uciPrmsH5 = cuphy::get_HDF5_struct_index(F3UciPrmsH5, uciIdx);
                populateUciParams(uciPrmsH5, F3UciPrmsVec[uciIdx + offset], i, uciIdx + offset, 3);
            }

            if(i == (nCells - 1))
            {
                cellGrpDynPrm.nF3Ucis    = nF3Ucis;
                cellGrpDynPrm.pF3UciPrms = F3UciPrmsVec.data();
            }
        }
        catch(const cuphy::cuphyHDF5_exception& e)
        {
            NVLOGW_FMT(NVLOG_PUCCH, "no PF3 Uci parameter object found in TV HDF5 file");
        }

        // input paramEters
        tDataRxVec[i] = cuphy::tensor_from_dataset(fInput.open_dataset("DataRx"), CUPHY_C_16F, cuphy::tensor_flags::align_tight, cuStrm);
        CUDA_CHECK(cudaStreamSynchronize(cuStrm));
        tPrmDataRxVec[i].desc  = tDataRxVec[i].desc().handle();
        tPrmDataRxVec[i].pAddr = tDataRxVec[i].addr();
    }

    // debug parameters
    dbgPrm.pOutFileName = nullptr;
    dbgPrm.enableDynApiLogging  = 0;
    dbgPrm.enableStatApiLogging = 0;

    // remaining DataOut pointers
    constexpr size_t MAX_N_F234_UCI  = CUPHY_PUCCH_F2_MAX_UCI + CUPHY_PUCCH_F3_MAX_UCI;
    //bDtxFlag = std::move(cuphy::buffer<uint8_t, cuphy::pinned_alloc>(MAX_N_F234_UCI));
    //DataOut.pDtxFlags = bDtxFlag.addr();
    bF2OutOffsets = std::move(cuphy::buffer<cuphyPucchF234OutOffsets_t, cuphy::pinned_alloc>(CUPHY_PUCCH_F2_MAX_UCI));
    DataOut.pPucchF2OutOffsets = bF2OutOffsets.addr();
    bF3OutOffsets = std::move(cuphy::buffer<cuphyPucchF234OutOffsets_t, cuphy::pinned_alloc>(CUPHY_PUCCH_F3_MAX_UCI));
    DataOut.pPucchF3OutOffsets = bF3OutOffsets.addr();
    bUciPayload = std::move(cuphy::buffer<uint8_t, cuphy::pinned_alloc>(MAX_N_F234_UCI * 1024));
    DataOut.pUciPayloads = bUciPayload.addr();
    bCrcFlag = std::move(cuphy::buffer<uint8_t, cuphy::pinned_alloc>(MAX_N_F234_UCI));
    DataOut.pCrcFlags = bCrcFlag.addr();
    bTaEst = std::move(cuphy::buffer<float, cuphy::pinned_alloc>(MAX_N_F234_UCI));
    DataOut.pTaEst = bTaEst.addr();
    bRssi = std::move(cuphy::buffer<float, cuphy::pinned_alloc>(MAX_N_F234_UCI));
    DataOut.pRssi = bRssi.addr();
    bSinr = std::move(cuphy::buffer<float, cuphy::pinned_alloc>(MAX_N_F234_UCI));
    DataOut.pSinr = bSinr.addr();
    bInterf = std::move(cuphy::buffer<float, cuphy::pinned_alloc>(MAX_N_F234_UCI));
    DataOut.pInterf = bInterf.addr();
    bRsrp = std::move(cuphy::buffer<float, cuphy::pinned_alloc>(MAX_N_F234_UCI));
    DataOut.pRsrp = bRsrp.addr();
    bHarqDetectionStatus = std::move(cuphy::buffer<uint8_t, cuphy::pinned_alloc>(MAX_N_F234_UCI));
    DataOut.HarqDetectionStatus = bHarqDetectionStatus.addr();
    bCsiP1DetectionStatus = std::move(cuphy::buffer<uint8_t, cuphy::pinned_alloc>(MAX_N_F234_UCI));
    DataOut.CsiP1DetectionStatus = bCsiP1DetectionStatus.addr();
    bCsiP2DetectionStatus = std::move(cuphy::buffer<uint8_t, cuphy::pinned_alloc>(MAX_N_F234_UCI));
    DataOut.CsiP2DetectionStatus = bCsiP2DetectionStatus.addr();
    // if pNumCsi2Bits is used later, uncomment the following 2 lines
    //bNumCsi2Bits = std::move(cuphy::buffer<uint16_t, cuphy::pinned_alloc>(MAX_N_F234_UCI));
    //DataOut.pNumCsi2Bits = bNumCsi2Bits.addr();

    // cell group parameters
    cellGrpDynPrm.nCells    = nCells;
    cellGrpDynPrm.pCellPrms = cellDynPrm.data();

    DataIn.pTDataRx = tPrmDataRxVec.data();

    pucchDynPrm.procModeBmsk   = procMode;
    pucchDynPrm.pDataIn        = &DataIn;
    pucchDynPrm.pDataOut       = &DataOut;
    pucchDynPrm.pCellGrpDynPrm = &cellGrpDynPrm;
    pucchDynPrm.cpuCopyOn      = 0;

    pucchDynPrm.pDbg           = &dbgPrm;

    StatusOutput = {cuphyPucchStatusType_t::CUPHY_PUCCH_STATUS_SUCCESS_OR_UNTRACKED_ISSUE, MAX_UINT16, MAX_UINT16};
    pucchDynPrm.pStatusOut = &StatusOutput;
}

//----------------------------------------------------------------------------------------------------------
//  Dataset holds static PUCCH api parameters/data

// construct from h5 file
pucchStaticApiDataset::pucchStaticApiDataset(const std::vector<std::string>& inputFileNameVec, cudaStream_t cuStrm, std::string outputFileName)
{
    pucchTracker.pMemoryFootprint = nullptr;
    pucchStatPrms.pOutInfo        = &pucchTracker;

    int nCells = static_cast<int>(inputFileNameVec.size());
    cellStatPrm.resize(nCells);

    pucchCellStatPrm.nCsirsPorts      = static_cast<uint8_t>(4);
    pucchCellStatPrm.N1               = static_cast<uint8_t>(2);
    pucchCellStatPrm.N2               = static_cast<uint8_t>(1);
    pucchCellStatPrm.csiReportingBand = static_cast<uint8_t>(0);
    pucchCellStatPrm.codebookType     = static_cast<uint8_t>(0);
    pucchCellStatPrm.codebookMode     = static_cast<uint8_t>(1);
    pucchCellStatPrm.isCqi            = static_cast<uint8_t>(0);
    pucchCellStatPrm.isLi             = static_cast<uint8_t>(0);

    // for storing listLength value from the TV of cell 0
    uint8_t tempPolarDcdrListSz;
    // loop through cells
    for(int i = 0; i < nCells; i++)
    {
        // load h5 file
        hdf5hpp::hdf5_file fInput = hdf5hpp::hdf5_file::open(inputFileNameVec[i].c_str());
        // static cell parameters
        cuphy::cuphyHDF5_struct cellStatPrmH5 = cuphy::get_HDF5_struct(fInput, "cellStatPrm");
        //read_cell_static_pars_from_file(cellStatPrm.data() + current_static_cells, cell_static_dataset, num_cells, current_static_cells);
        cellStatPrm[i].phyCellId = cellStatPrmH5.get_value_as<uint16_t>("phyCellId");
        cellStatPrm[i].nRxAnt    = cellStatPrmH5.get_value_as<uint16_t>("nRxAnt");
        cellStatPrm[i].nTxAnt    = cellStatPrmH5.get_value_as<uint16_t>("nTxAnt");
        cellStatPrm[i].nPrbUlBwp = cellStatPrmH5.get_value_as<uint16_t>("nPrbUlBwp");
        cellStatPrm[i].nPrbDlBwp = cellStatPrmH5.get_value_as<uint16_t>("nPrbDlBwp");
        cellStatPrm[i].mu        = cellStatPrmH5.get_value_as<uint8_t>("mu");
        // load listLength only from cell 0
        if (i == 0) tempPolarDcdrListSz = cellStatPrmH5.get_value_as<uint8_t>("listLength");
        cellStatPrm[i].pPuschCellStatPrms = nullptr;
        cellStatPrm[i].pPucchCellStatPrms = &pucchCellStatPrm;
    }

    // debug parameters
    bOutputFileName     = outputFileName;
    dbgPrm.pOutFileName = bOutputFileName.empty() ? nullptr : bOutputFileName.c_str();
    // dbgPrm.enableDynApiLogging  = 0;
    // dbgPrm.enableStatApiLogging = 0;

    // static pucch parameters
    pucchStatPrms.nMaxCells        = nCells;
    pucchStatPrms.nMaxCellsPerSlot = nCells;
    pucchStatPrms.uciOutputMode    = 0;
    pucchStatPrms.pCellStatPrms    = &cellStatPrm[0];
    pucchStatPrms.pDbg             = &dbgPrm;
    pucchStatPrms.polarDcdrListSz  = tempPolarDcdrListSz;
    pucchStatPrms.enableUlRxBf     = 0; //TODO
    pucchStatPrms.enableBatchedMemcpy = 1; //FIXME decide on appropriate default value. This will only be relevant for standalone cuPHY tests/examples
}

//----------------------------------------------------------------------------------------------------------
//  Dataset holds reference UCI output to evaluate cuPHY PUCCH receiver

// construct from h5 file
EvalPucchDataset::EvalPucchDataset(const std::vector<std::string>& inputFileNameVec, cudaStream_t cuStrm)
{
    int nCells = static_cast<int>(inputFileNameVec.size());
    tRefLLRs.resize(nCells);
    tRefHarqDetStat.resize(nCells);
    tRefCsiPart1DetStat.resize(nCells);
    tRefCsiPart2DetStat.resize(nCells);
    tRefDtxFlags.resize(nCells);
    tRefSinr.resize(nCells);
    tRefInterf.resize(nCells);
    tRefRssi.resize(nCells);
    tRefRsrp.resize(nCells);
    tRefTaEst.resize(nCells);
    tRefPayloadBytes.resize(nCells);
    tRefSeg1LLRs.resize(nCells);
    tRefSeg2LLRs.resize(nCells);

    for(int i = 0; i < nCells; i++)
    {
        // load h5 file
        hdf5hpp::hdf5_file      fInput          = hdf5hpp::hdf5_file::open(inputFileNameVec[i].c_str());
        cuphy::cuphyHDF5_struct cellGrpDynPrmH5 = cuphy::get_HDF5_struct(fInput, "cellGrpDynPrm");

        //PF0
        try
        {
            uint16_t              offset      = static_cast<uint16_t>(F0UcisOutRefVec.size());
            uint16_t              nF0UcisTmp  = cellGrpDynPrmH5.get_value_as<uint16_t>("nF0Ucis");
            hdf5hpp::hdf5_dataset F0UcisOutH5 = fInput.open_dataset("F0UcisOutRef");
            nF0Ucis                           = offset + nF0UcisTmp;
            F0UcisOutRefVec.resize(nF0Ucis);

            for(int uciIdx = 0; uciIdx < nF0UcisTmp; ++uciIdx)
            {
                cuphy::cuphyHDF5_struct uciOutH5 = cuphy::get_HDF5_struct_index(F0UcisOutH5, uciIdx);

                F0UcisOutRefVec[uciIdx + offset].taEstMicroSec       = uciOutH5.get_value_as<float_t>("taEstMicroSec");
                F0UcisOutRefVec[uciIdx + offset].SinrDB              = uciOutH5.get_value_as<float_t>("SinrDB");
                F0UcisOutRefVec[uciIdx + offset].InterfDB            = uciOutH5.get_value_as<float_t>("InterfDB");
                F0UcisOutRefVec[uciIdx + offset].RSSI                = uciOutH5.get_value_as<float_t>("RSSI");
                F0UcisOutRefVec[uciIdx + offset].RSRP                = uciOutH5.get_value_as<float_t>("RSRP");
                F0UcisOutRefVec[uciIdx + offset].SRindication        = uciOutH5.get_value_as<uint8_t>("SRindication");
                F0UcisOutRefVec[uciIdx + offset].NumHarq             = uciOutH5.get_value_as<uint8_t>("NumHarq");
                F0UcisOutRefVec[uciIdx + offset].HarqValues[0]       = uciOutH5.get_value_as<uint8_t>("HarqValue0");
                F0UcisOutRefVec[uciIdx + offset].HarqValues[1]       = uciOutH5.get_value_as<uint8_t>("HarqValue1");
                F0UcisOutRefVec[uciIdx + offset].SRconfidenceLevel   = uciOutH5.get_value_as<uint8_t>("SRconfidenceLevel");
                F0UcisOutRefVec[uciIdx + offset].HarqconfidenceLevel = uciOutH5.get_value_as<uint8_t>("HarqconfidenceLevel");
            }
        }
        catch(const cuphy::cuphyHDF5_exception& e)
        {
            NVLOGW_FMT(NVLOG_PUCCH, "no PF0 Uci parameter object found in TV HDF5 file\n");
            nF0Ucis = 0;
        }

        //PF1
        try
        {
            uint16_t              offset      = static_cast<uint16_t>(F1UcisOutRefVec.size());
            uint16_t              nF1UcisTmp  = cellGrpDynPrmH5.get_value_as<uint16_t>("nF1Ucis");
            hdf5hpp::hdf5_dataset F1UcisOutH5 = fInput.open_dataset("F1UcisOutRef");
            hdf5hpp::hdf5_dataset F1UciPrms   = fInput.open_dataset("F1UciPrms");
            nF1Ucis                           = offset + nF1UcisTmp;
            F1UcisOutRefVec.resize(nF1Ucis);
            pucchF1multiplexed.resize(nF1Ucis);
            std::unordered_map<uint32_t,uint8_t> uci_grp_counts;

            // Count entries in each UCI group
            for(int uciIdx = 0; uciIdx < nF1UcisTmp; ++uciIdx)
            {
                cuphy::cuphyHDF5_struct uciPrmsH5 = cuphy::get_HDF5_struct_index(F1UciPrms, uciIdx);
                // Create a key for determining UCI multiplexing
                uint32_t uci_prms    = uciPrmsH5.get_value_as<uint16_t>("BWPStart");
                uci_prms            += uciPrmsH5.get_value_as<uint16_t>("startPrb");
                uci_prms             = (uci_prms << 4) + uciPrmsH5.get_value_as<uint8_t>("startSym");
                uci_grp_counts[uci_prms]++;
            }

            for(int uciIdx = 0; uciIdx < nF1UcisTmp; ++uciIdx)
            {
                cuphy::cuphyHDF5_struct uciOutH5 = cuphy::get_HDF5_struct_index(F1UcisOutH5, uciIdx);

                F1UcisOutRefVec[uciIdx + offset].taEstMicroSec       = uciOutH5.get_value_as<float_t>("taEstMicroSec");
                F1UcisOutRefVec[uciIdx + offset].SinrDB              = uciOutH5.get_value_as<float_t>("SinrDB");
                F1UcisOutRefVec[uciIdx + offset].InterfDB            = uciOutH5.get_value_as<float_t>("InterfDB");
                F1UcisOutRefVec[uciIdx + offset].RSSI                = uciOutH5.get_value_as<float_t>("RSSI");
                F1UcisOutRefVec[uciIdx + offset].RSRP                = uciOutH5.get_value_as<float_t>("RSRP");
                F1UcisOutRefVec[uciIdx + offset].SRindication        = uciOutH5.get_value_as<uint8_t>("SRindication");
                F1UcisOutRefVec[uciIdx + offset].NumHarq             = uciOutH5.get_value_as<uint8_t>("NumHarq");
                F1UcisOutRefVec[uciIdx + offset].HarqValues[0]       = uciOutH5.get_value_as<uint8_t>("HarqValue0");
                F1UcisOutRefVec[uciIdx + offset].HarqValues[1]       = uciOutH5.get_value_as<uint8_t>("HarqValue1");
                F1UcisOutRefVec[uciIdx + offset].SRconfidenceLevel   = uciOutH5.get_value_as<uint8_t>("SRconfidenceLevel");
                F1UcisOutRefVec[uciIdx + offset].HarqconfidenceLevel = uciOutH5.get_value_as<uint8_t>("HarqconfidenceLevel");

                cuphy::cuphyHDF5_struct uciPrmsH5 = cuphy::get_HDF5_struct_index(F1UciPrms, uciIdx);
                // Create a key for determining UCI multiplexing
                uint32_t uci_prms    = uciPrmsH5.get_value_as<uint16_t>("BWPStart");
                uci_prms            += uciPrmsH5.get_value_as<uint16_t>("startPrb");
                uci_prms             = (uci_prms << 4) + uciPrmsH5.get_value_as<uint8_t>("startSym");
                bool multiplexed_uci = uci_grp_counts[uci_prms] > 1;
                pucchF1multiplexed[uciIdx + offset] = multiplexed_uci;
            }
        }
        catch(const cuphy::cuphyHDF5_exception& e)
        {
            NVLOGW_FMT(NVLOG_PUCCH, "no PF1 Uci parameter object found in TV HDF5 file\n");
            nF1Ucis = 0;
        }

        // PF2 and PF3
        // sizes
        uint16_t nF2UcisTmp = cellGrpDynPrmH5.get_value_as<uint16_t>("nF2Ucis");
        uint16_t nF3UcisTmp = cellGrpDynPrmH5.get_value_as<uint16_t>("nF3Ucis");

        uint16_t offsetF2 = static_cast<uint16_t>(pucchF2bufferOffsetsVec.size());
        uint16_t offsetF3 = static_cast<uint16_t>(pucchF3bufferOffsetsVec.size());

        nF2Ucis = offsetF2 + nF2UcisTmp;
        nF3Ucis = offsetF3 + nF3UcisTmp;

        // Load PUCCH F2 buffer offsets
        if(nF2UcisTmp > 0)
        {
            pucchF2bufferOffsetsVec.resize(nF2Ucis);
            hdf5hpp::hdf5_dataset F2offsetsH5 = fInput.open_dataset("pucchF2_refBufferOffsets");
            for(int uciIdx = 0; uciIdx < nF2UcisTmp; ++uciIdx)
            {
                cuphy::cuphyHDF5_struct offsets = cuphy::get_HDF5_struct_index(F2offsetsH5, uciIdx);

                pucchF2bufferOffsetsVec[uciIdx + offsetF2].harqDetStatOffset        = offsets.get_value_as<uint32_t>("harqDetStatOffset");
                pucchF2bufferOffsetsVec[uciIdx + offsetF2].csiPart1DetStatOffset    = offsets.get_value_as<uint32_t>("csiPart1DetStatOffset");
                pucchF2bufferOffsetsVec[uciIdx + offsetF2].csiPart2DetStatOffset    = offsets.get_value_as<uint32_t>("csiPart2DetStatOffset");
                pucchF2bufferOffsetsVec[uciIdx + offsetF2].dtxFlagOffset            = offsets.get_value_as<uint32_t>("dtxFlagOffset");
                pucchF2bufferOffsetsVec[uciIdx + offsetF2].snrOffset                = offsets.get_value_as<uint32_t>("snrOffset");
                pucchF2bufferOffsetsVec[uciIdx + offsetF2].RSRPoffset               = offsets.get_value_as<uint32_t>("RSRPoffset");
                pucchF2bufferOffsetsVec[uciIdx + offsetF2].RSSIoffset               = offsets.get_value_as<uint32_t>("RSSIoffset");
                pucchF2bufferOffsetsVec[uciIdx + offsetF2].InterfOffset             = offsets.get_value_as<uint32_t>("InterfOffset");
                pucchF2bufferOffsetsVec[uciIdx + offsetF2].taEstOffset              = offsets.get_value_as<uint32_t>("taOffset");
                pucchF2bufferOffsetsVec[uciIdx + offsetF2].uciSeg1PayloadByteOffset = offsets.get_value_as<uint32_t>("uciSeg1PayloadByteOffset");
                pucchF2bufferOffsetsVec[uciIdx + offsetF2].nUciSeg1Bytes            = offsets.get_value_as<uint32_t>("nUciSeg1Bytes");
                pucchF2bufferOffsetsVec[uciIdx + offsetF2].harqPayloadByteOffset    = offsets.get_value_as<uint32_t>("harqPayloadByteOffset");
                pucchF2bufferOffsetsVec[uciIdx + offsetF2].nHarqBytes               = offsets.get_value_as<uint32_t>("nHarqBytes");
                pucchF2bufferOffsetsVec[uciIdx + offsetF2].srPayloadByteOffset      = offsets.get_value_as<uint32_t>("srPayloadByteOffset");
                pucchF2bufferOffsetsVec[uciIdx + offsetF2].nSrBytes                 = offsets.get_value_as<uint32_t>("nSrBytes");
                pucchF2bufferOffsetsVec[uciIdx + offsetF2].csiP1PayloadByteOffset   = offsets.get_value_as<uint32_t>("csiP1PayloadByteOffset");
                pucchF2bufferOffsetsVec[uciIdx + offsetF2].nCsiP1Bytes              = offsets.get_value_as<uint32_t>("nCsiP1Bytes");
                pucchF2bufferOffsetsVec[uciIdx + offsetF2].LLRsoffset               = offsets.get_value_as<uint32_t>("LLRsoffset");
                pucchF2bufferOffsetsVec[uciIdx + offsetF2].nSegLLRs                 = offsets.get_value_as<uint32_t>("nSegLLRs");
                pucchF2bufferOffsetsVec[uciIdx + offsetF2].cellIdx                  = i;
            }
        }

        // Load PUCCH F3 buffer offsets
        uint16_t numPart2s = 0;
        if(nF3UcisTmp > 0)
        {
            pucchF3bufferOffsetsVec.resize(nF3Ucis);
            hdf5hpp::hdf5_dataset F3offsetsH5 = fInput.open_dataset("pucchF3_refBufferOffsets");
            hdf5hpp::hdf5_dataset F3UciPrmsH5 = fInput.open_dataset("F3UciPrms");
            for(int uciIdx = 0; uciIdx < nF3UcisTmp; ++uciIdx)
            {
                cuphy::cuphyHDF5_struct offsets   = cuphy::get_HDF5_struct_index(F3offsetsH5, uciIdx);
                cuphy::cuphyHDF5_struct uciPrmsH5 = cuphy::get_HDF5_struct_index(F3UciPrmsH5, uciIdx);
                uint16_t                temp      = uciPrmsH5.get_value_as<uint16_t>("numPart2s");
                numPart2s += temp;

                pucchF3bufferOffsetsVec[uciIdx + offsetF3].harqDetStatOffset        = offsets.get_value_as<uint32_t>("harqDetStatOffset");
                pucchF3bufferOffsetsVec[uciIdx + offsetF3].csiPart1DetStatOffset    = offsets.get_value_as<uint32_t>("csiPart1DetStatOffset");
                pucchF3bufferOffsetsVec[uciIdx + offsetF3].csiPart2DetStatOffset    = offsets.get_value_as<uint32_t>("csiPart2DetStatOffset");
                pucchF3bufferOffsetsVec[uciIdx + offsetF3].dtxFlagOffset            = offsets.get_value_as<uint32_t>("dtxFlagOffset");
                pucchF3bufferOffsetsVec[uciIdx + offsetF3].snrOffset                = offsets.get_value_as<uint32_t>("snrOffset");
                pucchF3bufferOffsetsVec[uciIdx + offsetF3].RSRPoffset               = offsets.get_value_as<uint32_t>("RSRPoffset");
                pucchF3bufferOffsetsVec[uciIdx + offsetF3].RSSIoffset               = offsets.get_value_as<uint32_t>("RSSIoffset");
                pucchF3bufferOffsetsVec[uciIdx + offsetF3].InterfOffset             = offsets.get_value_as<uint32_t>("InterfOffset");
                pucchF3bufferOffsetsVec[uciIdx + offsetF3].taEstOffset              = offsets.get_value_as<uint32_t>("taOffset");
                pucchF3bufferOffsetsVec[uciIdx + offsetF3].uciSeg1PayloadByteOffset = offsets.get_value_as<uint32_t>("uciSeg1PayloadByteOffset");
                pucchF3bufferOffsetsVec[uciIdx + offsetF3].nUciSeg1Bytes            = offsets.get_value_as<uint32_t>("nUciSeg1Bytes");
                pucchF3bufferOffsetsVec[uciIdx + offsetF3].harqPayloadByteOffset    = offsets.get_value_as<uint32_t>("harqPayloadByteOffset");
                pucchF3bufferOffsetsVec[uciIdx + offsetF3].nHarqBytes               = offsets.get_value_as<uint32_t>("nHarqBytes");
                pucchF3bufferOffsetsVec[uciIdx + offsetF3].srPayloadByteOffset      = offsets.get_value_as<uint32_t>("srPayloadByteOffset");
                pucchF3bufferOffsetsVec[uciIdx + offsetF3].nSrBytes                 = offsets.get_value_as<uint32_t>("nSrBytes");
                pucchF3bufferOffsetsVec[uciIdx + offsetF3].csiP1PayloadByteOffset   = offsets.get_value_as<uint32_t>("csiP1PayloadByteOffset");
                pucchF3bufferOffsetsVec[uciIdx + offsetF3].nCsiP1Bytes              = offsets.get_value_as<uint32_t>("nCsiP1Bytes");
                pucchF3bufferOffsetsVec[uciIdx + offsetF3].LLRsoffset               = offsets.get_value_as<uint32_t>("LLRsoffset");
                pucchF3bufferOffsetsVec[uciIdx + offsetF3].Seg1LLRsoffset           = offsets.get_value_as<uint32_t>("Seg1LLRsoffset");
                pucchF3bufferOffsetsVec[uciIdx + offsetF3].Seg2LLRsoffset           = offsets.get_value_as<uint32_t>("Seg2LLRsoffset");
                pucchF3bufferOffsetsVec[uciIdx + offsetF3].nSegLLRs                 = offsets.get_value_as<uint32_t>("nSegLLRs");
                pucchF3bufferOffsetsVec[uciIdx + offsetF3].nSeg1LLRs                = offsets.get_value_as<uint32_t>("nSeg1LLRs");
                pucchF3bufferOffsetsVec[uciIdx + offsetF3].nSeg2LLRs                = offsets.get_value_as<uint32_t>("nSeg2LLRs");
                pucchF3bufferOffsetsVec[uciIdx + offsetF3].cellIdx                  = i;
            }
        }

        // Load PUCCH F234 reference buffers
        if((nF2UcisTmp + nF3UcisTmp) > 0)
        {
            tRefLLRs[i] = typed_tensor_from_dataset<CUPHY_R_16F, pinned_alloc>(fInput.open_dataset("pucchF234_refLLRbuffer"), cuphy::tensor_flags::align_tight, cuStrm);
            if(nF3UcisTmp > 0)
            {
                tRefSeg1LLRs[i] = typed_tensor_from_dataset<CUPHY_R_16F, pinned_alloc>(fInput.open_dataset("pucchF234_refSeg1LLRbuffer"), cuphy::tensor_flags::align_tight, cuStrm);
                if(numPart2s > 0)
                {
                    tRefSeg2LLRs[i] = typed_tensor_from_dataset<CUPHY_R_16F, pinned_alloc>(fInput.open_dataset("pucchF234_refSeg2LLRbuffer"), cuphy::tensor_flags::align_tight, cuStrm);
                }
            }
            tRefHarqDetStat[i]     = typed_tensor_from_dataset<CUPHY_R_8U, pinned_alloc>(fInput.open_dataset("pucchF234_refHarqDetStatBuffer"), cuphy::tensor_flags::align_tight, cuStrm);
            tRefCsiPart1DetStat[i] = typed_tensor_from_dataset<CUPHY_R_8U, pinned_alloc>(fInput.open_dataset("pucchF234_refCsiPart1DetStatBuffer"), cuphy::tensor_flags::align_tight, cuStrm);
            tRefCsiPart2DetStat[i] = typed_tensor_from_dataset<CUPHY_R_8U, pinned_alloc>(fInput.open_dataset("pucchF234_refCsiPart2DetStatBuffer"), cuphy::tensor_flags::align_tight, cuStrm);
            tRefDtxFlags[i]        = typed_tensor_from_dataset<CUPHY_R_8U, pinned_alloc>(fInput.open_dataset("pucchF234_refDTXbuffer"), cuphy::tensor_flags::align_tight, cuStrm);
            tRefPayloadBytes[i]    = typed_tensor_from_dataset<CUPHY_R_8U, pinned_alloc>(fInput.open_dataset("pucchF234_refPayloadBuffer"), cuphy::tensor_flags::align_tight, cuStrm);
            tRefSinr[i]         = typed_tensor_from_dataset<CUPHY_R_32F, pinned_alloc>(fInput.open_dataset("pucchF234_refSnrBuffer"),    cuphy::tensor_flags::align_tight, cuStrm);
            tRefInterf[i]       = typed_tensor_from_dataset<CUPHY_R_32F, pinned_alloc>(fInput.open_dataset("pucchF234_refInterfBuffer"), cuphy::tensor_flags::align_tight, cuStrm);
            tRefRssi[i]         = typed_tensor_from_dataset<CUPHY_R_32F, pinned_alloc>(fInput.open_dataset("pucchF234_refRssiBuffer"),   cuphy::tensor_flags::align_tight, cuStrm);
            tRefRsrp[i]         = typed_tensor_from_dataset<CUPHY_R_32F, pinned_alloc>(fInput.open_dataset("pucchF234_refRsrpBuffer"),   cuphy::tensor_flags::align_tight, cuStrm);
            tRefTaEst[i]        = typed_tensor_from_dataset<CUPHY_R_32F, pinned_alloc>(fInput.open_dataset("pucchF234_refTaBuffer"),     cuphy::tensor_flags::align_tight, cuStrm);
        }
    }

}

uint16_t EvalPucchDataset::evalPucchF0Receiver(cuphyPucchF0F1UciOut_t* pF0UcisOutGpu, cudaStream_t cuStrm)
{
    uint16_t nMismatches = 0;
    if(nF0Ucis)
    {
        /// Copy GPU output to CPU
        cuphy::buffer<cuphyPucchF0F1UciOut_t, cuphy::pinned_alloc> bF0UcisOutCpu(nF0Ucis);
        CUDA_CHECK(cudaMemcpyAsync(bF0UcisOutCpu.addr(), pF0UcisOutGpu, sizeof(cuphyPucchF0F1UciOut_t) * nF0Ucis, cudaMemcpyDeviceToHost, cuStrm));
        CUDA_CHECK(cudaStreamSynchronize(cuStrm));

        // compare cuPHY UCI output to reference UCI output

        std::string pucchFormatName = "pucchF0";
        nMismatches = compareF0F1UciOutput(nF0Ucis, bF0UcisOutCpu.addr(), F0UcisOutRefVec, pucchFormatName);
    }
    return nMismatches;
}

uint16_t EvalPucchDataset::evalPucchF1Receiver(cuphyPucchF0F1UciOut_t* pF1UcisOutGpu, cudaStream_t cuStrm)
{
    uint16_t nMismatches = 0;
    if(nF1Ucis)
    {
        /// Copy GPU output to CPU
        cuphy::buffer<cuphyPucchF0F1UciOut_t, cuphy::pinned_alloc> bF1UcisOutCpu(nF1Ucis);
        CUDA_CHECK(cudaMemcpyAsync(bF1UcisOutCpu.addr(), pF1UcisOutGpu, sizeof(cuphyPucchF0F1UciOut_t) * nF1Ucis, cudaMemcpyDeviceToHost, cuStrm));
        CUDA_CHECK(cudaStreamSynchronize(cuStrm));

        // compare cuPHY UCI output to reference UCI output
        std::string pucchFormatName = "pucchF1";
        nMismatches = compareF0F1UciOutput(nF0Ucis, bF1UcisOutCpu.addr(), F1UcisOutRefVec, pucchFormatName);
    }
    return nMismatches;
}

uint16_t EvalPucchDataset::evalPucchF2FrontEnd(__half** pDescramLLRaddrs, uint8_t* pDTXflags, uint16_t* E_seg1, cudaStream_t cuStrm)
{
    uint32_t                                    maxNumLLRs = CUPHY_PUCCH_F2_MAX_E;
    cuphy::buffer<__half, cuphy::pinned_alloc>  bCuphyLLRs(maxNumLLRs);
    cuphy::buffer<uint8_t, cuphy::pinned_alloc> bCuphyDtxFlags(nF2Ucis);
    CUDA_CHECK(cudaMemcpyAsync(bCuphyDtxFlags.addr(), pDTXflags, nF2Ucis * sizeof(uint8_t), cudaMemcpyDeviceToHost, cuStrm));

    uint16_t nMismatches = 0;
    for(int uciIdx = 0; uciIdx < nF2Ucis; ++uciIdx)
    {
        uint8_t errorFlag = 0;

        // compare cuphy DTX flag to reference
        uint32_t dtxFlagOffset = pucchF2bufferOffsetsVec[uciIdx].dtxFlagOffset;
        uint16_t cellIdx       = pucchF2bufferOffsetsVec[uciIdx].cellIdx;

        if(tRefDtxFlags[cellIdx](dtxFlagOffset) == 1)
        {
            continue;
        }

        if(bCuphyDtxFlags[uciIdx] != tRefDtxFlags[cellIdx](dtxFlagOffset))
        {
            errorFlag = 1;
            printf("\n DTXflag mismatch detected for pucch F2 uci: %d", uciIdx);
        }

        // compare cuphy payload to reference payload:
        CUDA_CHECK(cudaMemcpyAsync(bCuphyLLRs.addr(), pDescramLLRaddrs[uciIdx], sizeof(__half) * E_seg1[uciIdx], cudaMemcpyDeviceToHost, cuStrm));
        CUDA_CHECK(cudaStreamSynchronize(cuStrm));

        double   signalEnergy  = 0;
        double   errorEnergy   = 0;
        uint32_t payloadOffset = pucchF2bufferOffsetsVec[uciIdx].LLRsoffset;
        cellIdx                = pucchF2bufferOffsetsVec[uciIdx].cellIdx;

        for(int bitIdx = 0; bitIdx < 2; ++bitIdx)
        {
            double cuphyLLR = static_cast<double>(bCuphyLLRs[bitIdx]);
            double refLLR   = static_cast<double>(tRefLLRs[cellIdx](payloadOffset + bitIdx));
            signalEnergy += pow(abs(refLLR), 2);
            errorEnergy += pow(abs(refLLR - cuphyLLR), 2);
        }

        double snr = 10 * log10(signalEnergy / errorEnergy);

        // 30 was chosen to give enough implementation error with fp16 vs double. 50dB was failing on correct results
        if(snr < 30)
        {
            errorFlag = 1;
            printf("\n LLR mismatch detected for pucch F2 uci: %d", uciIdx);
        }

        nMismatches += errorFlag;
    }
    return nMismatches;
}

uint16_t EvalPucchDataset::evalPucchF3FrontEnd(__half** pDescramLLRaddrs, uint8_t* pDTXflags, uint16_t* E_tot, cudaStream_t cuStrm)
{
    uint32_t                                    maxNumLLRs = CUPHY_PUCCH_F3_MAX_E;
    cuphy::buffer<__half, cuphy::pinned_alloc>  bCuphyLLRs(maxNumLLRs);
    cuphy::buffer<uint8_t, cuphy::pinned_alloc> bCuphyDtxFlags(nF3Ucis);
    CUDA_CHECK(cudaMemcpyAsync(bCuphyDtxFlags.addr(), pDTXflags, nF3Ucis * sizeof(uint8_t), cudaMemcpyDeviceToHost, cuStrm));

    uint16_t nMismatches = 0;
    for(int uciIdx = 0; uciIdx < nF3Ucis; ++uciIdx)
    {
        uint8_t errorFlag = 0;

        //  compare cuphy payload to reference payload:
        CUDA_CHECK(cudaMemcpyAsync(bCuphyLLRs.addr(), pDescramLLRaddrs[uciIdx], sizeof(__half) * E_tot[uciIdx], cudaMemcpyDeviceToHost, cuStrm));
        CUDA_CHECK(cudaStreamSynchronize(cuStrm));

        double   signalEnergy  = 0;
        double   errorEnergy   = 0;
        uint32_t payloadOffset = pucchF3bufferOffsetsVec[uciIdx].LLRsoffset;
        uint16_t cellIdx       = pucchF3bufferOffsetsVec[uciIdx].cellIdx;

        for(int bitIdx = 0; bitIdx < 2; ++bitIdx)
        {
            double cuphyLLR = static_cast<double>(bCuphyLLRs[bitIdx]);
            double refLLR   = static_cast<double>(tRefLLRs[cellIdx](payloadOffset + bitIdx));

            signalEnergy += pow(abs(refLLR), 2);
            errorEnergy += pow(abs(refLLR - cuphyLLR), 2);
        }

        double snr = 0;
        //ToDo?? why the code below is commented out?
        /* double snr = 10*log10(signalEnergy/errorEnergy);
        if(snr < 50)
        {
           errorFlag = 1;
           printf("\n LLR mismatch detected for pucch F3 uci: %d", uciIdx);
        } */

        // compare cuphy DTX flag to reference
        uint32_t dtxFlagOffset = pucchF3bufferOffsetsVec[uciIdx].dtxFlagOffset;
        if(bCuphyDtxFlags[uciIdx] != tRefDtxFlags[cellIdx](dtxFlagOffset))
        {
            errorFlag = 1;
            printf("\n DTXflag mismatch detected for pucch F3 uci: %d", uciIdx);
        }

        nMismatches += errorFlag;
    }
    return nMismatches;
}

uint16_t EvalPucchDataset::evalPucchF3SegLLRs(__half** pDescramLLRaddrs, uint16_t* E_seg1, uint16_t* E_seg2, cudaStream_t cuStrm)
{
    uint32_t                                    maxNumLLRs = CUPHY_PUCCH_F3_MAX_E;
    cuphy::buffer<__half, cuphy::pinned_alloc>  bCuphyLLRs(maxNumLLRs);

    uint16_t nMismatches = 0;
    for(int uciIdx = 0; uciIdx < nF3Ucis; ++uciIdx)
    {
        //  compare cuphy payload to reference payload:
        CUDA_CHECK(cudaMemcpyAsync(bCuphyLLRs.addr(), pDescramLLRaddrs[uciIdx], sizeof(__half) * (E_seg1[uciIdx] + E_seg2[uciIdx]), cudaMemcpyDeviceToHost, cuStrm));
        CUDA_CHECK(cudaStreamSynchronize(cuStrm));

        uint32_t Seg1payloadOffset = pucchF3bufferOffsetsVec[uciIdx].Seg1LLRsoffset;
        uint32_t Seg2payloadOffset = pucchF3bufferOffsetsVec[uciIdx].Seg2LLRsoffset;

        // compare LLR segment 1
        uint8_t  errorFlag = 0;
        double   signalEnergy  = 0;
        double   errorEnergy   = 0;
        uint16_t cellIdx       = pucchF3bufferOffsetsVec[uciIdx].cellIdx;

        for(int bitIdx = 0; bitIdx < E_seg1[uciIdx]; ++bitIdx)
        {
            double cuphyLLR = static_cast<double>(bCuphyLLRs[bitIdx]);
            double refLLR   = static_cast<double>(tRefSeg1LLRs[cellIdx](Seg1payloadOffset + bitIdx));

            signalEnergy += pow(abs(refLLR), 2);
            errorEnergy += pow(abs(refLLR - cuphyLLR), 2);
        }

        double snr = 10*log10(signalEnergy/errorEnergy);
        if(snr < 30)
        {
           errorFlag = 1;
           printf("\n LLR segment 1 mismatch detected for pucch F3 uci: %d\n", uciIdx);
        }

        nMismatches += errorFlag;

        // compare LLR segment 2
        if (E_seg2[uciIdx] > 0) {
            errorFlag = 0;
            signalEnergy  = 0;
            errorEnergy   = 0;
            for(int bitIdx = 0; bitIdx < E_seg2[uciIdx]; ++bitIdx)
            {
                double cuphyLLR = static_cast<double>(bCuphyLLRs[E_seg1[uciIdx] + bitIdx]);
                double refLLR   = static_cast<double>(tRefSeg2LLRs[cellIdx](Seg2payloadOffset + bitIdx));

                signalEnergy += pow(abs(refLLR), 2);
                errorEnergy += pow(abs(refLLR - cuphyLLR), 2);
            }

            snr = 10*log10(signalEnergy/errorEnergy);
            if(snr < 30)
            {
            errorFlag = 1;
            printf("\n LLR segment 2 mismatch detected for pucch F3 uci: %d\n", uciIdx);
            }

            nMismatches += errorFlag;
        }
    }

    return nMismatches;
}


uint16_t EvalPucchDataset::evalPucchF234UciSeg(uint8_t* uciPayloadsGpu, uint16_t nF2Ucis, uint16_t nF3Ucis, cuphyPucchF234OutOffsets_t* pF2Cuphyoffsets, cuphyPucchF234OutOffsets_t* pF3Cuphyoffsets, cudaStream_t cuStrm)
{
    int numBytesUciPayloads = tRefPayloadBytes[0].desc().get_size_in_bytes();
    uint8_t* uciPayloadsCpu = new uint8_t[numBytesUciPayloads];

    //  compare cuphy payload to reference payload:
    CUDA_CHECK(cudaMemcpyAsync(uciPayloadsCpu, uciPayloadsGpu, numBytesUciPayloads * sizeof(uint8_t), cudaMemcpyDeviceToHost, cuStrm));
    CUDA_CHECK(cudaStreamSynchronize(cuStrm));

    uint16_t nMismatches = 0;

    for (int uciIdx = 0; uciIdx < nF2Ucis; uciIdx++) {
        uint8_t errorFlag = 0;

        uint32_t nUciSeg1Bytes = pucchF2bufferOffsetsVec[uciIdx].nUciSeg1Bytes;
        uint32_t nHarqBytes    = pucchF2bufferOffsetsVec[uciIdx].nHarqBytes;
        uint32_t nSrBytes      = pucchF2bufferOffsetsVec[uciIdx].nSrBytes;
        uint32_t nCsiP1Bytes   = pucchF2bufferOffsetsVec[uciIdx].nCsiP1Bytes;

        if(nUciSeg1Bytes > 0)
        {
            uint32_t refUciSeg1PayloadOffset   = pucchF2bufferOffsetsVec[uciIdx].uciSeg1PayloadByteOffset;
            uint32_t cuphyUciSeg1PayloadOffset = pF2Cuphyoffsets[uciIdx].uciSeg1PayloadByteOffset;

            for(int byteIdx = 0; byteIdx < nUciSeg1Bytes; ++byteIdx)
            {
                if(tRefPayloadBytes[0](refUciSeg1PayloadOffset + byteIdx) != uciPayloadsCpu[cuphyUciSeg1PayloadOffset + byteIdx])
                {
                    errorFlag = 1;
                    printf("UciSeg1 payload mismatch for PF2 uci %d\n", uciIdx);
                }
            }
        }

        if(nHarqBytes > 0)
        {
            uint32_t refHarqPayloadByteOffset   = pucchF2bufferOffsetsVec[uciIdx].harqPayloadByteOffset;
            uint32_t cuphyHarqPayloadByteOffset = pF2Cuphyoffsets[uciIdx].harqPayloadByteOffset;

            for(int byteIdx = 0; byteIdx < nHarqBytes; ++byteIdx)
            {
                if(tRefPayloadBytes[0](refHarqPayloadByteOffset + byteIdx) != uciPayloadsCpu[cuphyHarqPayloadByteOffset + byteIdx])
                {
                    errorFlag = 1;
                    printf("HARQ payload mismatch for PF2 uci %d\n", uciIdx);
                }
            }
        }

        if(nSrBytes > 0)
        {
            uint32_t refSrPayloadByteOffset   = pucchF2bufferOffsetsVec[uciIdx].srPayloadByteOffset;
            uint32_t cuphySrPayloadByteOffset = pF2Cuphyoffsets[uciIdx].srPayloadByteOffset;

            for(int byteIdx = 0; byteIdx < nSrBytes; ++byteIdx)
            {
                if(tRefPayloadBytes[0](refSrPayloadByteOffset + byteIdx) != uciPayloadsCpu[cuphySrPayloadByteOffset + byteIdx])
                {
                    errorFlag = 1;
                    printf("SR payload mismatch for PF2 uci %d\n", uciIdx);
                }
            }
        }

        if(nCsiP1Bytes > 0)
        {
            uint32_t refCsi1PayloadByteOffset   = pucchF2bufferOffsetsVec[uciIdx].csiP1PayloadByteOffset;
            uint32_t cuphyCsi1PayloadByteOffset = pF2Cuphyoffsets[uciIdx].csi1PayloadByteOffset;

            for(int byteIdx = 0; byteIdx < nCsiP1Bytes; ++byteIdx)
            {
                if(tRefPayloadBytes[0](refCsi1PayloadByteOffset + byteIdx) != uciPayloadsCpu[cuphyCsi1PayloadByteOffset + byteIdx])
                {
                    errorFlag = 1;
                    printf("CSI part 1 payload mismatch for PF2 uci %d\n", uciIdx);
                }
            }
        }
        nMismatches += errorFlag;
    }

    for (int uciIdx = 0; uciIdx < nF3Ucis; uciIdx++) {
        uint8_t errorFlag = 0;

        uint32_t nUciSeg1Bytes = pucchF3bufferOffsetsVec[uciIdx].nUciSeg1Bytes;
        uint32_t nHarqBytes    = pucchF3bufferOffsetsVec[uciIdx].nHarqBytes;
        uint32_t nSrBytes      = pucchF3bufferOffsetsVec[uciIdx].nSrBytes;
        uint32_t nCsiP1Bytes   = pucchF3bufferOffsetsVec[uciIdx].nCsiP1Bytes;

        if(nUciSeg1Bytes > 0)
        {
            uint32_t refUciSeg1PayloadOffset   = pucchF3bufferOffsetsVec[uciIdx].uciSeg1PayloadByteOffset;
            uint32_t cuphyUciSeg1PayloadOffset = pF3Cuphyoffsets[uciIdx].uciSeg1PayloadByteOffset;

            for(int byteIdx = 0; byteIdx < nUciSeg1Bytes; ++byteIdx)
            {
                if(tRefPayloadBytes[0](refUciSeg1PayloadOffset + byteIdx) != uciPayloadsCpu[cuphyUciSeg1PayloadOffset + byteIdx])
                {
                    errorFlag = 1;
                    printf("UciSeg1 payload mismatch for PF3 uci %d\n", uciIdx);
                }
            }
        }

        if(nHarqBytes > 0)
        {
            uint32_t refHarqPayloadByteOffset   = pucchF3bufferOffsetsVec[uciIdx].harqPayloadByteOffset;
            uint32_t cuphyHarqPayloadByteOffset = pF3Cuphyoffsets[uciIdx].harqPayloadByteOffset;

            for(int byteIdx = 0; byteIdx < nHarqBytes; ++byteIdx)
            {
                if(tRefPayloadBytes[0](refHarqPayloadByteOffset + byteIdx) != uciPayloadsCpu[cuphyHarqPayloadByteOffset + byteIdx])
                {
                    errorFlag = 1;
                    printf("HARQ payload mismatch for PF3 uci %d\n", uciIdx);
                }
            }
        }

        if(nSrBytes > 0)
        {
            uint32_t refSrPayloadByteOffset   = pucchF3bufferOffsetsVec[uciIdx].srPayloadByteOffset;
            uint32_t cuphySrPayloadByteOffset = pF3Cuphyoffsets[uciIdx].srPayloadByteOffset;

            for(int byteIdx = 0; byteIdx < nSrBytes; ++byteIdx)
            {
                if(tRefPayloadBytes[0](refSrPayloadByteOffset + byteIdx) != uciPayloadsCpu[cuphySrPayloadByteOffset + byteIdx])
                {
                    errorFlag = 1;
                    printf("SR payload mismatch for PF3 uci %d\n", uciIdx);
                }
            }
        }

        if(nCsiP1Bytes > 0)
        {
            uint32_t refCsi1PayloadByteOffset   = pucchF3bufferOffsetsVec[uciIdx].csiP1PayloadByteOffset;
            uint32_t cuphyCsi1PayloadByteOffset = pF3Cuphyoffsets[uciIdx].csi1PayloadByteOffset;

            for(int byteIdx = 0; byteIdx < nCsiP1Bytes; ++byteIdx)
            {
                if(tRefPayloadBytes[0](refCsi1PayloadByteOffset + byteIdx) != uciPayloadsCpu[cuphyCsi1PayloadByteOffset + byteIdx])
                {
                    errorFlag = 1;
                    printf("CSI part 1 payload mismatch for PF3 uci %d\n", uciIdx);
                }
            }
        }
        nMismatches += errorFlag;
    }

    delete[] uciPayloadsCpu;
    return nMismatches;
}


uint16_t EvalPucchDataset::compareF0F1UciOutput(uint16_t nUcis, cuphyPucchF0F1UciOut_t* uciOutMeas, std::vector<cuphyPucchF0F1UciOut_t> uciOutRef, std::string pucchFormatName)
{
    uint16_t nMismatches = 0;
    for(int uciIdx = 0; uciIdx < nUcis; ++uciIdx)
    {
        uint8_t errorFlag = 0;
        float refSINR  = uciOutRef[uciIdx].SinrDB;
        float measSINR = uciOutMeas[uciIdx].SinrDB;
        // Clip SINR between -12dB and +55dB until performance characterization is done
        float conf     = std::fabs(std::min(std::max(refSINR, -12.0f),55.0f)
                                 - std::min(std::max(measSINR,-12.0f),55.0f));
        if(conf > 1.5) { // TODO the tolerance is high for now for functional checks.  Will explore performance in future work
            NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT, "SINR doesn't match for uci {}, expected={} != actual={}",uciIdx , uciOutRef[uciIdx].SinrDB, uciOutMeas[uciIdx].SinrDB);
            errorFlag = 1;
        }

        float refRSSI  = uciOutRef[uciIdx].RSSI;
        float measRSSI = uciOutMeas[uciIdx].RSSI;
        conf           = std::fabs(refRSSI - measRSSI);
        if(conf > 1.0) {
            NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT, "RSSI doesn't match for uci {}, expected={} != actual={}",uciIdx,uciOutRef[uciIdx].RSSI,uciOutMeas[uciIdx].RSSI);
            errorFlag = 1;
        }

        float refRSRP  = uciOutRef[uciIdx].RSRP;
        float measRSRP = uciOutMeas[uciIdx].RSRP;
        conf           = std::fabs(refRSRP - measRSRP);
        if(conf > 1.0) {
            NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT, "RSRP doesn't match for uci {}, expected={} != actual={}",uciIdx,uciOutRef[uciIdx].RSRP,uciOutMeas[uciIdx].RSRP);
            errorFlag = 1;
        }

        // Clip Interference + noise at -30dB until performance characterization is done
        // For PF1: when RSRP is too large (e.g., > 20 dB), there will be a power leakage from the PF1 noise filter which will affect the accuracy of the interference + noise measurement
        float refInterf  = std::max( uciOutRef[uciIdx].InterfDB, -30.0f);
        float measInterf = std::max(uciOutMeas[uciIdx].InterfDB, -30.0f);
        conf             = std::fabs(refInterf - measInterf);

        // For PF1: when RSRP > 20 dB, an up to 3 dB difference between the measured values from 5GModel and cuPHY is expected (due to the use of half-precision data in PF1 receiver)
        float confTolerance = 1.0;
        if (measRSRP > 20.0) {
            confTolerance = 3.0;
        }

        if(conf > confTolerance) {
            NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT, "Interference + noise level doesn't match for uci {}, expected={} != actual={}",
                       uciIdx, uciOutRef[uciIdx].InterfDB, uciOutMeas[uciIdx].InterfDB);
            errorFlag = 1;
        }

        if(pucchFormatName=="pucchF1") // Only check PF1 TA estimation if not multiplexed (PF0 not currently implemented)
        {
            if(!pucchF1multiplexed[uciIdx])
            {
                float refTaEst   = uciOutRef[uciIdx].taEstMicroSec;
                float measTaEst  = uciOutMeas[uciIdx].taEstMicroSec;
                conf             = std::fabs(refTaEst - measTaEst);
                if(((conf > 1.0) && (refSINR > -10.0)) || ((conf > 4.0) && (refSINR > -60.0)))
                {
                    NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT, "Timing Advance doesn't match for uci {}, expected={} != actual={}",
                               uciIdx,uciOutRef[uciIdx].taEstMicroSec,uciOutMeas[uciIdx].taEstMicroSec);
                    errorFlag = 1;
                }
            }
        }

        if(uciOutRef[uciIdx].SRindication != uciOutMeas[uciIdx].SRindication)
        {
            NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT, "SR indication not matching for uci {}, got={} wanted={}",
                       uciIdx, uciOutMeas[uciIdx].SRindication, uciOutRef[uciIdx].SRindication);
            errorFlag = 1;
        }
        if(uciOutRef[uciIdx].NumHarq != uciOutMeas[uciIdx].NumHarq)
        {
            NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT, "Num harq not matching for uci {}, got={} wanted={}",
                       uciIdx, uciOutMeas[uciIdx].NumHarq, uciOutRef[uciIdx].NumHarq);
            errorFlag = 1;
        }
        if (uciOutRef[uciIdx].HarqconfidenceLevel != uciOutMeas[uciIdx].HarqconfidenceLevel)
        {
            NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT, "HarqconfidenceLevel not matching for uci {}, got={} wanted={}",
                       uciIdx, uciOutMeas[uciIdx].HarqconfidenceLevel, uciOutRef[uciIdx].HarqconfidenceLevel);
            errorFlag = 1;
        }
        if (uciOutRef[uciIdx].SRconfidenceLevel != uciOutMeas[uciIdx].SRconfidenceLevel)
        {
            NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT, "SRconfidenceLevel not matching for uci {}, got={} wanted={}",
                       uciIdx, uciOutMeas[uciIdx].SRconfidenceLevel, uciOutRef[uciIdx].SRconfidenceLevel);
            errorFlag = 1;
        }
        for(int bitIdx = 0; bitIdx < uciOutRef[uciIdx].NumHarq; ++bitIdx)
        {
            if(uciOutRef[uciIdx].HarqValues[bitIdx] != uciOutMeas[uciIdx].HarqValues[bitIdx])
            {
                NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT, "Uci {} HARQ bit not matching bit={} got={} wanted={}",
                           uciIdx, bitIdx, uciOutMeas[uciIdx].HarqValues[bitIdx], uciOutRef[uciIdx].HarqValues[bitIdx]);
                errorFlag = 1;
            }
        }
        nMismatches+=errorFlag;
    }
    return nMismatches;
}

uint16_t EvalPucchDataset::compareF234UciOutput(uint16_t nUcis, cuphyPucchDynPrms_t& pucchDynPrm, cuphyPucchF234OutOffsets_t* pCuphyoffsets, pucchF234bufferOffsets* pRefoffsets, std::string pucchFormatName)
{
    // cuphy output buffers:
    uint8_t* pCuphyUciPayloads     = pucchDynPrm.pDataOut->pUciPayloads;
    uint8_t* pCuphyHarqDetStat     = pucchDynPrm.pDataOut->HarqDetectionStatus;
    uint8_t* pCuphyCsiPart1DetStat = pucchDynPrm.pDataOut->CsiP1DetectionStatus;
    uint8_t* pCuphyCsiPart2DetStat = pucchDynPrm.pDataOut->CsiP2DetectionStatus;
    float*   pCuphySinr        = pucchDynPrm.pDataOut->pSinr;
    float*   pCuphyRsrp        = pucchDynPrm.pDataOut->pRsrp;
    float*   pCuphyRssi        = pucchDynPrm.pDataOut->pRssi;
    float*   pCuphyInterf      = pucchDynPrm.pDataOut->pInterf;
    float*   pCuphyTaEst       = pucchDynPrm.pDataOut->pTaEst;

    // compare cuphy output to reference:
    uint16_t nMismatches = 0;
    for(int uciIdx = 0; uciIdx < nUcis; ++uciIdx)
    {
        uint8_t errorFlag = 0;
        // Compare measurements
        uint32_t refSinrOffset   =   pRefoffsets[uciIdx].snrOffset;
        uint16_t cuphySinrOffset = pCuphyoffsets[uciIdx].snrOffset;
        uint16_t cellIdx         = pRefoffsets[uciIdx].cellIdx;
        float    refSinr         = tRefSinr[cellIdx](refSinrOffset);
        float    measSinr        = pCuphySinr[cuphySinrOffset];
        float    conf            = std::fabs(refSinr - measSinr);
        if(conf > 3.0 )
        {
            NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT, "SINR doesn't match for uci {}, expected={} != actual={}",uciIdx, refSinr, measSinr);
            nMismatches += 1;
        }

        uint32_t refInterfOffset   =   pRefoffsets[uciIdx].InterfOffset;
        uint16_t cuphyInterfOffset = pCuphyoffsets[uciIdx].InterfOffset;
        float    refInterf         = tRefInterf[cellIdx](refInterfOffset);
        float    measInterf        = pCuphyInterf[cuphySinrOffset];
        conf                       = std::fabs(refInterf - measInterf);
        if(conf > 3.0 )
        {
            NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT, "Interference + noise doesn't match for uci {}, expected={} != actual={}",uciIdx, refInterf, measInterf);
            nMismatches += 1;
        }

        uint32_t refRssiOffset   =   pRefoffsets[uciIdx].RSSIoffset;
        uint16_t cuphyRssiOffset = pCuphyoffsets[uciIdx].RSSIoffset;
        float    refRssi         = tRefRssi[cellIdx](refRssiOffset);
        float    measRssi        = pCuphyRssi[cuphyRssiOffset];
        conf                     = std::fabs(refRssi - measRssi);
        if(conf > 3.0 )
        {
            NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT, "RSSI doesn't match for uci {}, expected={} != actual={}",uciIdx, refRssi, measRssi);
            nMismatches += 1;
        }

        uint32_t refRsrpOffset   =   pRefoffsets[uciIdx].RSRPoffset;
        uint16_t cuphyRsrpOffset = pCuphyoffsets[uciIdx].RSRPoffset;
        float    refRsrp         = tRefRsrp[cellIdx](refRsrpOffset);
        float    measRsrp        = pCuphyRsrp[cuphyRsrpOffset];
        conf                     = std::fabs(refRsrp - measRsrp);
        if(conf > 3.0 )
        {
            NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT, "RSRP doesn't match for uci {}, expected={} != actual={}",uciIdx, refRsrp, measRsrp);
            nMismatches += 1;
        }

        uint32_t refTaEstOffset   =   pRefoffsets[uciIdx].taEstOffset;
        uint16_t cuphyTaEstOffset = pCuphyoffsets[uciIdx].taEstOffset;
        float    refTaEst         = tRefTaEst[cellIdx](refTaEstOffset);
        float    measTaEst        = pCuphyTaEst[cuphyTaEstOffset];
        conf                      = std::fabs(refTaEst - measTaEst);
        if(((conf > 0.5) && (refSinr > 0.0)) || (conf > 1.0))
        {
            NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT, "Timing advance doesn't match for uci {}, expected={} != actual={}",uciIdx, refTaEst, measTaEst);
            NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT, "\tRSSI: {} SINR: {}",refRssi,refSinr);
            nMismatches += 1;
        }

        // Compare detection status
        uint32_t refHarqDetStatOffset       = pRefoffsets[uciIdx].harqDetStatOffset;
        uint32_t refCsiPart1DetStatOffset   = pRefoffsets[uciIdx].csiPart1DetStatOffset;
        uint32_t refCsiPart2DetStatOffset   = pRefoffsets[uciIdx].csiPart2DetStatOffset;
        uint16_t cuphyHarqDetStatOffset     = pCuphyoffsets[uciIdx].HarqDetectionStatusOffset;
        uint16_t cuphyCsiPart1DetStatOffset = pCuphyoffsets[uciIdx].CsiP1DetectionStatusOffset;
        uint16_t cuphyCsiPart2DetStatOffset = pCuphyoffsets[uciIdx].CsiP2DetectionStatusOffset;

        // Compare Uci Segment 1
        uint32_t nUciSeg1Bytes = pRefoffsets[uciIdx].nUciSeg1Bytes;
        uint32_t nHarqBytes    = pRefoffsets[uciIdx].nHarqBytes;
        uint32_t nSrBytes      = pRefoffsets[uciIdx].nSrBytes;
        uint32_t nCsiP1Bytes   = pRefoffsets[uciIdx].nCsiP1Bytes;

        if(nHarqBytes>0)
        {
            if (tRefHarqDetStat[cellIdx](refHarqDetStatOffset) != pCuphyHarqDetStat[cuphyHarqDetStatOffset]) {
                nMismatches += 1;
                NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT, "HARQ detectionStatus mismatch for uci {}", uciIdx);
                continue;
            }
        }

        if(nCsiP1Bytes>0){
            if (tRefCsiPart1DetStat[cellIdx](refCsiPart1DetStatOffset) != pCuphyCsiPart1DetStat[cuphyCsiPart1DetStatOffset]) {
                nMismatches += 1;
                NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT, "CSI part 1 detectionStatus mismatch for uci {}", uciIdx);
                continue;
            }
        }

        if((tRefHarqDetStat[cellIdx](refHarqDetStatOffset)==CUPHY_FAPI_DTX)||(tRefCsiPart1DetStat[cellIdx](refCsiPart1DetStatOffset)==CUPHY_FAPI_DTX))
        {
            continue;
        }

        // if (tRefCsiPart2DetStat[cellIdx](refCsiPart2DetStatOffset) != pCuphyCsiPart2DetStat[cuphyCsiPart2DetStatOffset]) {
        //     nMismatches += 1;
        //     NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT, "CSI part 2 detectionStatus mismatch for uci {}", uciIdx);
        //     continue;
        // }

//        // Compare DTX flag
//        uint32_t refDtxFlagOffset   = pRefoffsets[uciIdx].dtxFlagOffset;
//        uint16_t cuphyDtxFlagOffset = pCuphyoffsets[uciIdx].dtxFlagOffset;
//
//        if(tRefDtxFlags[cellIdx](refDtxFlagOffset) != pCuphyDtxFlags[cuphyDtxFlagOffset])
//        {
//            nMismatches += 1;
//            NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT, "dtx flag mismatch for uci {}", uciIdx);
//            continue;
//        }
//
//        if(tRefDtxFlags[cellIdx](refDtxFlagOffset)==1)
//        {
//            continue;
//        }

        if(nUciSeg1Bytes > 0)
        {
            uint32_t refUciSeg1PayloadOffset   = pRefoffsets[uciIdx].uciSeg1PayloadByteOffset;
            uint32_t cuphyUciSeg1PayloadOffset = pCuphyoffsets[uciIdx].uciSeg1PayloadByteOffset;
            for(int byteIdx = 0; byteIdx < nUciSeg1Bytes; ++byteIdx)
            {
                if(tRefPayloadBytes[cellIdx](refUciSeg1PayloadOffset + byteIdx) != pCuphyUciPayloads[cuphyUciSeg1PayloadOffset + byteIdx])
                {
                    errorFlag = 1;
                    NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT, "UciSeg1 payload mismatch for {} uci {}", pucchFormatName.c_str(), uciIdx);
                }
            }
        }

        if(nHarqBytes > 0)
        {
            uint32_t refHarqPayloadByteOffset   = pRefoffsets[uciIdx].harqPayloadByteOffset;
            uint32_t cuphyHarqPayloadByteOffset = pCuphyoffsets[uciIdx].harqPayloadByteOffset;
            for(int byteIdx = 0; byteIdx < nHarqBytes; ++byteIdx)
            {
                if(tRefPayloadBytes[cellIdx](refHarqPayloadByteOffset + byteIdx) != pCuphyUciPayloads[cuphyHarqPayloadByteOffset + byteIdx])
                {
                    errorFlag = 1;
                    NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT, "HARQ payload mismatch for {} uci {}", pucchFormatName.c_str(), uciIdx);
                }
            }
        }

        if(nSrBytes > 0)
        {
            uint32_t refSrPayloadByteOffset   = pRefoffsets[uciIdx].srPayloadByteOffset;
            uint32_t cuphySrPayloadByteOffset = pCuphyoffsets[uciIdx].srPayloadByteOffset;
            for(int byteIdx = 0; byteIdx < nSrBytes; ++byteIdx)
            {
                if(tRefPayloadBytes[cellIdx](refSrPayloadByteOffset + byteIdx) != pCuphyUciPayloads[cuphySrPayloadByteOffset + byteIdx])
                {
                    errorFlag = 1;
                    NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT, "SR payload mismatch for {} uci {}", pucchFormatName.c_str(), uciIdx);
                }
            }
        }

        if(nCsiP1Bytes > 0)
        {
            uint32_t refCsiP1PayloadByteOffset   = pRefoffsets[uciIdx].csiP1PayloadByteOffset;
            uint32_t cuphyCsi1PayloadByteOffset  = pCuphyoffsets[uciIdx].csi1PayloadByteOffset;
            for(int byteIdx = 0; byteIdx < nCsiP1Bytes; ++byteIdx)
            {
                if(tRefPayloadBytes[cellIdx](refCsiP1PayloadByteOffset + byteIdx) != pCuphyUciPayloads[cuphyCsi1PayloadByteOffset + byteIdx])
                {
                    errorFlag = 1;
                    NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT, "CSI part 1 payload mismatch for {} uci {}", pucchFormatName.c_str(), uciIdx);
                }
            }
        }

        nMismatches += errorFlag;
    }

    return nMismatches;
}

int EvalPucchDataset::evalPucchRxPipeline(cuphyPucchDynPrms_t& pucchDynPrm)
{
    cuphyPucchF0F1UciOut_t* pF0UcisOut = pucchDynPrm.pDataOut->pF0UcisOut;
    cuphyPucchF0F1UciOut_t* pF1UcisOut = pucchDynPrm.pDataOut->pF1UcisOut;

    uint16_t nF0Mismatches = 0;
    uint16_t nF1Mismatches = 0;
    uint16_t nF2Mismatches = 0;
    uint16_t nF3Mismatches = 0;

    std::string pucchFormatName  = "pucchF0";
    nF0Mismatches += compareF0F1UciOutput(nF0Ucis, pF0UcisOut, F0UcisOutRefVec, pucchFormatName);
    pucchFormatName  = "pucchF1";
    nF1Mismatches += compareF0F1UciOutput(nF1Ucis, pF1UcisOut, F1UcisOutRefVec, pucchFormatName);

    cuphyPucchF234OutOffsets_t* pCuphyPF23offsets = pucchDynPrm.pDataOut->pPucchF2OutOffsets;
    pucchF234bufferOffsets*     pRefPF23offsets   = pucchF2bufferOffsetsVec.data();
    pucchFormatName  = "pucchF2";
    nF2Mismatches += compareF234UciOutput(nF2Ucis, pucchDynPrm, pCuphyPF23offsets, pRefPF23offsets, pucchFormatName);

    pCuphyPF23offsets = pucchDynPrm.pDataOut->pPucchF3OutOffsets;
    pRefPF23offsets   = pucchF3bufferOffsetsVec.data();
    pucchFormatName  = "pucchF3";
    nF3Mismatches += compareF234UciOutput(nF3Ucis, pucchDynPrm, pCuphyPF23offsets, pRefPF23offsets, pucchFormatName);

    std::string summary = fmt::format("\n PUCCH format 0: found {} mismatches out of {} Ucis", nF0Mismatches, nF0Ucis);
    summary += fmt::format("\n PUCCH format 1: found {} mismatches out of {} Ucis", nF1Mismatches, nF1Ucis);
    summary += fmt::format("\n PUCCH format 2: found {} mismatches out of {} Ucis", nF2Mismatches, nF2Ucis);
    summary += fmt::format("\n PUCCH format 3: found {} mismatches out of {} Ucis", nF3Mismatches, nF3Ucis);
    summary += "\n\n";
    NVLOGC_FMT(NVLOG_PUCCH, "{}",summary);

    if(nF0Mismatches || nF1Mismatches || nF2Mismatches || nF3Mismatches)
    {
        NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT, "PUCCH reference check: FAILED! exiting");
    }
    return nF0Mismatches + nF1Mismatches + nF2Mismatches + nF3Mismatches;
}

//----------------------------------------------------------------------------------------------------------
// UciPolarDataset
// 1.) contains parameters and input data consumed by cuPHY uci polar functions
// 2.) contains reference intermediate buffers
// 3.) contains validation functions comparing cuPHY output against reference buffers

UciPolarDataset::UciPolarDataset(const std::string& inputFileName, cudaStream_t cuStrm)
{
    // load h5 file
    hdf5hpp::hdf5_file      fInput               = hdf5hpp::hdf5_file::open(inputFileName.c_str());
    cuphy::cuphyHDF5_struct sizesH5              = cuphy::get_HDF5_struct(fInput, "sizes");
    hdf5hpp::hdf5_dataset   polUciSegPrmsArrayH5 = fInput.open_dataset("polarUciSegPrms");

    // sizes
    nPolUciSegs = sizesH5.get_value_as<uint16_t>("nPolUciSegs");
    nPolCws     = sizesH5.get_value_as<uint16_t>("nPolCws");

    // Polar uci segment parameters
    polUciSegPrmsVec.resize(nPolUciSegs);
    for(uint16_t segIdx = 0; segIdx < nPolUciSegs; ++segIdx)
    {
        cuphy::cuphyHDF5_struct polUciSegPrmsH5 = cuphy::get_HDF5_struct_index(polUciSegPrmsArrayH5, segIdx);

        polUciSegPrmsVec[segIdx].nCbs           = polUciSegPrmsH5.get_value_as<uint8_t>("nCbs");
        polUciSegPrmsVec[segIdx].E_cw           = polUciSegPrmsH5.get_value_as<uint32_t>("E_cw");
        polUciSegPrmsVec[segIdx].K_cw           = polUciSegPrmsH5.get_value_as<uint16_t>("K_cw");
        polUciSegPrmsVec[segIdx].N_cw           = polUciSegPrmsH5.get_value_as<uint16_t>("N_cw");
        polUciSegPrmsVec[segIdx].n_cw           = polUciSegPrmsH5.get_value_as<uint8_t>("n_cw");
        polUciSegPrmsVec[segIdx].E_seg          = polUciSegPrmsH5.get_value_as<uint32_t>("E_seg");
        polUciSegPrmsVec[segIdx].nCrcBits       = polUciSegPrmsH5.get_value_as<uint16_t>("nCrcBits");
        polUciSegPrmsVec[segIdx].zeroInsertFlag = polUciSegPrmsH5.get_value_as<uint8_t>("zeroInsertFlag");

    }

    // Reference cwTreeTypes
    for(uint16_t segIdx = 0; segIdx < nPolUciSegs; ++segIdx)
    {
        std::string          treeName    = "cwTreeTypes" + std::to_string(segIdx);
        cuphy::tensor_pinned cwTreeTypes = cuphy::tensor_from_dataset(fInput.open_dataset(treeName.c_str()), CUPHY_R_8U, cuphy::tensor_flags::align_tight, cuStrm);

        refCwTreeTypesVec.emplace_back(cwTreeTypes.layout());
        refCwTreeTypesVec[segIdx] = cwTreeTypes;
    }

    // Reference uciSegLLRs
    for(uint16_t segIdx = 0; segIdx < nPolUciSegs; ++segIdx)
    {
        std::string          datasetName = "uciSegLLRs" + std::to_string(segIdx);
        cuphy::tensor_pinned uciSegLLRs  = cuphy::tensor_from_dataset(fInput.open_dataset(datasetName.c_str()), CUPHY_R_16F, cuphy::tensor_flags::align_tight, cuStrm);

        refUciSegLLRsVec.emplace_back(uciSegLLRs.layout());
        refUciSegLLRsVec[segIdx] = uciSegLLRs;
    }

    // Reference cwLLRs
    for(uint16_t cwIdx = 0; cwIdx < nPolCws; ++cwIdx)
    {
        std::string          datasetName = "cwLLRs" + std::to_string(cwIdx);
        cuphy::tensor_pinned cwLLRs      = cuphy::tensor_from_dataset(fInput.open_dataset(datasetName.c_str()), CUPHY_R_16F, cuphy::tensor_flags::align_tight, cuStrm);

        refCwLLRsVec.emplace_back(cwLLRs.layout());
        refCwLLRsVec[cwIdx] = cwLLRs;
    }

    // Reference cbEsts
    for(uint16_t cbIdx = 0; cbIdx < nPolCws; ++cbIdx)
    {
        std::string          datasetName = "cbEst" + std::to_string(cbIdx);
        cuphy::tensor_pinned cbEst       = cuphy::tensor_from_dataset(fInput.open_dataset(datasetName.c_str()), CUPHY_R_32U, cuphy::tensor_flags::align_tight, cuStrm);

        refCbEstsVec.emplace_back(cbEst.layout());
        refCbEstsVec[cbIdx] = cbEst;
    }

    // Reference uciSegEsts
    for(uint16_t uciSegIdx = 0; uciSegIdx < nPolUciSegs; ++uciSegIdx)
    {
        std::string          datasetName = "uciSegEst" + std::to_string(uciSegIdx);
        cuphy::tensor_pinned uciSegEst   = cuphy::tensor_from_dataset(fInput.open_dataset(datasetName.c_str()), CUPHY_R_32U, cuphy::tensor_flags::align_tight, cuStrm);

        refUciSegEstsVec.emplace_back(uciSegEst.layout());
        refUciSegEstsVec[uciSegIdx] = uciSegEst;
    }

    // Reference crc error flags
    std::string          datasetName   = "crcErrorFlags";
    cuphy::tensor_pinned crcErrorFlags = (cuphy::tensor_from_dataset(fInput.open_dataset(datasetName.c_str()), CUPHY_R_8U, cuphy::tensor_flags::align_tight, cuStrm));
    refCrcErrorFlags                   = std::move(cuphy::typed_tensor<CUPHY_R_8U, cuphy::pinned_alloc>(crcErrorFlags.layout()));
    refCrcErrorFlags                   = crcErrorFlags;
}

void UciPolarDataset::evalCwTreeTypes(uint8_t** pCwTreeTypesAddrsGpu, cudaStream_t cuStrm)
{
    NVLOGC_FMT(NVLOG_PUSCH, "Comparing cuPHY output cwTreeTypes to reference cwTreeTypes...");
    uint16_t                                    max_N_cw        = 1024;
    uint16_t                                    max_cwTree_size = 2 * max_N_cw;
    cuphy::buffer<uint8_t, cuphy::pinned_alloc> cuphyCwTreeTypesCpu(max_cwTree_size);

    for(uint16_t segIdx = 0; segIdx < nPolUciSegs; ++segIdx)
    {
        uint16_t N_cw       = polUciSegPrmsVec[segIdx].N_cw;
        uint16_t nCwsInTree = 2 * N_cw;

        CUDA_CHECK(cudaMemcpyAsync(cuphyCwTreeTypesCpu.addr(), pCwTreeTypesAddrsGpu[segIdx], nCwsInTree, cudaMemcpyDeviceToHost, cuStrm));
        CUDA_CHECK(cudaStreamSynchronize(cuStrm));

        uint16_t nMismatches = 0;
        for(uint16_t cwIdx = 2; cwIdx < nCwsInTree; ++cwIdx)
        {
            if(refCwTreeTypesVec[segIdx](cwIdx - 2) != cuphyCwTreeTypesCpu[cwIdx])
            {
                NVLOGD_FMT(NVLOG_PUSCH, "segIdx {} cwIdx {} mismatches expected {} got {}",
                           segIdx,cwIdx,cuphyCwTreeTypesCpu[cwIdx],refCwTreeTypesVec[segIdx](cwIdx - 2));
                nMismatches += 1;
            }
        }

        NVLOGC_FMT(NVLOG_PUSCH, "UCI segment {} has {} codeword type mismatches out of {} codewords", segIdx, nMismatches, nCwsInTree);
    }
}

void UciPolarDataset::evalCwLLRs(uint16_t nPolCws, cuphyPolarCwPrm_t* pPolarCwPrms, __half** pCwLLRsAddrsGpu, cudaStream_t cuStrm)
{
    uint16_t                                   max_N_cw = 1024;
    cuphy::buffer<__half, cuphy::pinned_alloc> cwLLRsCpu(max_N_cw);

    int nMismatches = 0;
    NVLOGC_FMT(NVLOG_PUSCH, "Comparing cuPHY output codeword LLRs to reference...");

    for(int cwIdx = 0; cwIdx < nPolCws; ++cwIdx)
    {
        uint16_t N_cw   = pPolarCwPrms[cwIdx].N_cw;
        size_t   nBytes = N_cw * sizeof(__half);

        CUDA_CHECK(cudaMemcpyAsync(cwLLRsCpu.addr(), pCwLLRsAddrsGpu[cwIdx], nBytes, cudaMemcpyDeviceToHost, cuStrm));
        CUDA_CHECK(cudaStreamSynchronize(cuStrm));

        double signalEnergy = 0;
        double errorEnergy  = 0;

        for(int bitIdx = 0; bitIdx < N_cw; ++bitIdx)
        {
            double cuphyLLR = static_cast<double>(cwLLRsCpu[bitIdx]);
            double refLLR   = static_cast<double>(refCwLLRsVec[cwIdx](bitIdx));

            signalEnergy += pow(abs(refLLR), 2);
            errorEnergy += pow(abs(refLLR - cuphyLLR), 2);
        }

        double snr = 10 * log10(signalEnergy / errorEnergy);
        if(snr < 50)
        {
            nMismatches += 1;
            NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "mismatch detected for cw {}", cwIdx);
        }
    }
    NVLOGC_FMT(NVLOG_PUSCH, "detected {} mismatches out of {} codewords", nMismatches, nPolCws);
}

void UciPolarDataset::evalDecoderOutput(uint16_t nPolCbs, cuphyPolarCwPrm_t* pPolarCwPrms, uint32_t** pCbEstGpuAddrs, uint8_t* pCrcErrorFlagsGpu, uint16_t nPolSegs, cuphyPolarUciSegPrm_t* pUciSegPrms, uint32_t** pUciSegEstGpuAddrs, cudaStream_t cuStrm)
{
    uint16_t                                     max_words_cb = 32;
    cuphy::buffer<uint32_t, cuphy::pinned_alloc> cbEstCpu(max_words_cb);
    cuphy::buffer<uint32_t, cuphy::pinned_alloc> uciSegEstCpu(2*max_words_cb);
    cuphy::buffer<uint8_t, cuphy::pinned_alloc>  crcErrorFlagCpu(nPolCbs);
    CUDA_CHECK(cudaMemcpyAsync(crcErrorFlagCpu.addr(), pCrcErrorFlagsGpu, nPolCbs, cudaMemcpyDeviceToHost, cuStrm));

    int nMismatches = 0;
    NVLOGC_FMT(NVLOG_PUSCH, "Comparing cuPHY decoder output to reference...");

    for(int cbIdx = 0; cbIdx < nPolCbs; ++cbIdx)
    {
        uint16_t A_cw   = pPolarCwPrms[cbIdx].A_cw;
        uint16_t nWords = div_round_up(A_cw, static_cast<uint16_t>(32));

        CUDA_CHECK(cudaMemcpyAsync(cbEstCpu.addr(), pCbEstGpuAddrs[cbIdx], nWords * sizeof(uint32_t), cudaMemcpyDeviceToHost, cuStrm));
        CUDA_CHECK(cudaStreamSynchronize(cuStrm));
        uint8_t mismatchFlag = 0;

        if(crcErrorFlagCpu[cbIdx] != refCrcErrorFlags(cbIdx))
        {
            mismatchFlag = 1;
            NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "crcErrorFlag mismatch detected for cb {}", cbIdx);
        }

        for(int wordIdx = 0; wordIdx < nWords; ++wordIdx)
        {
            if(cbEstCpu[wordIdx] != refCbEstsVec[cbIdx](wordIdx))
            {
                mismatchFlag = 1;
                NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "cbEst mismatch detected for cb {}", cbIdx);
                break;
            }
        }

        if(mismatchFlag)
        {
            nMismatches += 1;
        }
    }
    NVLOGC_FMT(NVLOG_PUSCH, "detected {} mismatches out of {} codeblocks", nMismatches, nPolCbs);

    nMismatches = 0;
    for(int segIdx = 0; segIdx < nPolSegs; ++segIdx)
    {
        uint8_t  mismatchFlag = 0;
        uint32_t nSegBits     = pUciSegPrms[segIdx].nCbs * (pUciSegPrms[segIdx].K_cw - pUciSegPrms[segIdx].nCrcBits) - pUciSegPrms[segIdx].zeroInsertFlag;
        uint32_t nSegWords    =  div_round_up(nSegBits, static_cast<uint32_t>(32));

        CUDA_CHECK(cudaMemcpyAsync(uciSegEstCpu.addr(), pUciSegEstGpuAddrs[segIdx], nSegWords * sizeof(uint32_t), cudaMemcpyDeviceToHost, cuStrm));
        CUDA_CHECK(cudaStreamSynchronize(cuStrm));

        for(int wordIdx = 0; wordIdx < nSegWords; ++wordIdx)
        {
            if(uciSegEstCpu[wordIdx] != refUciSegEstsVec[segIdx](wordIdx))
            {
                mismatchFlag = 1;
                break;
            }
        }

        if(mismatchFlag)
        {
            nMismatches += 1;
        }
    }
    NVLOGC_FMT(NVLOG_PUSCH, "detected {} mismatches out of {} uciSegs", nMismatches, nPolSegs);

}

//----------------------------------------------------------------------------------------------------------
// simplexDataset
// 1.) contains parameters and input data consumed by cuPHY simplex functions
// 2.) contains validation functions comparing cuPHY output against reference buffers

simplexDataset::simplexDataset(const std::string& inputFileName, cudaStream_t cuStrm)
{
    hdf5hpp::hdf5_file fInput = hdf5hpp::hdf5_file::open(inputFileName.c_str());

    // Sizes:
    cuphy::cuphyHDF5_struct SimplexCwPar = cuphy::get_HDF5_struct(fInput, "SimplexCwPar");
    nCws                                 = static_cast<uint16_t>(SimplexCwPar.get_value_as<uint32_t>("numCW"));

    // Load per simplex paramters:
    simplexCwPrmsVec.resize(nCws);
    hdf5hpp::hdf5_dataset SimplexPar = fInput.open_dataset("SimplexPar");

    for(int cwIdx = 0; cwIdx < nCws; ++cwIdx)
    {
        cuphy::cuphyHDF5_struct cwPar = cuphy::get_HDF5_struct_index(SimplexPar, cwIdx);

        simplexCwPrmsVec[cwIdx].exitFlag    = 0;
        simplexCwPrmsVec[cwIdx].K           = cwPar.get_value_as<uint8_t>("K");
        simplexCwPrmsVec[cwIdx].E           = cwPar.get_value_as<uint32_t>("E");
        simplexCwPrmsVec[cwIdx].nBitsPerQam = cwPar.get_value_as<uint8_t>("nBitsPerQam");
    }

    // Load codeword LLRs:
    for(int cwIdx = 0; cwIdx < nCws; ++cwIdx)
    {
        std::string          cwLLRsName = "cwLLRs" + std::to_string(cwIdx);
        cuphy::tensor_pinned cwLLRs     = cuphy::tensor_from_dataset(fInput.open_dataset(cwLLRsName.c_str()), CUPHY_R_16F, cuphy::tensor_flags::align_tight, cuStrm);

        refCwLLRsVec.emplace_back(cwLLRs.layout());
        refCwLLRsVec[cwIdx] = cwLLRs;
    }

    // Load codeblocks
    std::string          cbName = "cb_vec";
    cuphy::tensor_pinned tCbs   = cuphy::tensor_from_dataset(fInput.open_dataset(cbName.c_str()), CUPHY_R_32U, cuphy::tensor_flags::align_tight, cuStrm);
    refCbs                      = std::move(cuphy::typed_tensor<CUPHY_R_32U, cuphy::pinned_alloc>(tCbs.layout()));
    refCbs                      = tCbs;
}

void simplexDataset::evalDecoderOutput(cuphySimplexCwPrm_t* h_simplexCwPrms, cudaStream_t cuStrm)
{
    uint32_t cbEst;

    uint16_t nMismatches = 0;
    for(int cbIdx = 0; cbIdx < nCws; ++cbIdx)
    {
        CUDA_CHECK(cudaMemcpyAsync(&cbEst, h_simplexCwPrms[cbIdx].d_cbEst, sizeof(uint32_t), cudaMemcpyDeviceToHost, cuStrm));
        CUDA_CHECK(cudaStreamSynchronize(cuStrm));

        if(cbEst != refCbs(cbIdx))
        {
            nMismatches += 1;
            NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "Error! Mismatch detected for codeblock {}", cbIdx);
        }
    }

    NVLOGC_FMT(NVLOG_PUSCH, "Simplex code: found {} mismatches out of {} codeblocks \n", nMismatches, nCws);
}


//----------------------------------------------------------------------------------------------------------
// Dataset holds dynamic SRS API parameters/data/buffers

srsDynApiDataset::srsDynApiDataset(const std::vector<std::string>& inputFileNameVec, cudaStream_t cuStrm, uint64_t procModeBmsk)
{
    uint16_t nCells           = inputFileNameVec.size();
    uint16_t nSrsUesInCellGrp = 0;
    std::vector<uint16_t> nUesPerCellVec(nCells);
    cellDynPrmVec.resize(nCells);

    // load SRS cell dynamic parameters:
    for(int cellIdx = 0; cellIdx < nCells; cellIdx++)
    {
        hdf5hpp::hdf5_file      fInput        = hdf5hpp::hdf5_file::open(inputFileNameVec[cellIdx].c_str());
        cuphy::cuphyHDF5_struct srsCellParams = cuphy::get_HDF5_struct(fInput, "srsCellParams");

        cellDynPrmVec[cellIdx].cellPrmStatIdx = cellIdx;
        cellDynPrmVec[cellIdx].cellPrmDynIdx  = cellIdx;
        cellDynPrmVec[cellIdx].slotNum        = srsCellParams.get_value_as<uint16_t>("slotNum");
        cellDynPrmVec[cellIdx].frameNum       = srsCellParams.get_value_as<uint16_t>("frameNum");
        cellDynPrmVec[cellIdx].srsStartSym    = srsCellParams.get_value_as<uint8_t>("srsStartSym");
        cellDynPrmVec[cellIdx].nSrsSym        = srsCellParams.get_value_as<uint8_t>("nSrsSym");

        nUesPerCellVec[cellIdx]  =  srsCellParams.get_value_as<uint16_t>("nSrsUes");
        nSrsUesInCellGrp        +=  nUesPerCellVec[cellIdx];
    }

    // load SRS user dynamic parameters, compute srsChEstBufferSize:
    uint16_t ueIdx_within_cellGrp = 0;
    uint32_t srsChEstBufferSize   = 0;
    ueSrsPrmVec.resize(nSrsUesInCellGrp);

    for(int cellIdx = 0; cellIdx < nCells; cellIdx++)
    {
        hdf5hpp::hdf5_file    fInput            = hdf5hpp::hdf5_file::open(inputFileNameVec[cellIdx].c_str());
        hdf5hpp::hdf5_dataset srsUePrms_dataset = fInput.open_dataset("srsUePrms");

        for(int ueIdx_within_cell = 0; ueIdx_within_cell < nUesPerCellVec[cellIdx]; ++ueIdx_within_cell)
        {
            cuphy::cuphyHDF5_struct ueSrsPrm = cuphy::get_HDF5_struct_index(srsUePrms_dataset, ueIdx_within_cell);

            ueSrsPrmVec[ueIdx_within_cellGrp].cellIdx                = cellIdx;
            ueSrsPrmVec[ueIdx_within_cellGrp].nAntPorts              = ueSrsPrm.get_value_as<uint8_t>("nAntPorts");
            ueSrsPrmVec[ueIdx_within_cellGrp].nSyms                  = ueSrsPrm.get_value_as<uint8_t>("nSyms");
            ueSrsPrmVec[ueIdx_within_cellGrp].nRepetitions           = ueSrsPrm.get_value_as<uint8_t>("nRepetitions");
            ueSrsPrmVec[ueIdx_within_cellGrp].combSize               = ueSrsPrm.get_value_as<uint8_t>("combSize");
            ueSrsPrmVec[ueIdx_within_cellGrp].startSym               = ueSrsPrm.get_value_as<uint8_t>("startSym");
            ueSrsPrmVec[ueIdx_within_cellGrp].sequenceId             = ueSrsPrm.get_value_as<uint16_t>("sequenceId");
            ueSrsPrmVec[ueIdx_within_cellGrp].configIdx              = ueSrsPrm.get_value_as<uint8_t>("configIdx");
            ueSrsPrmVec[ueIdx_within_cellGrp].bandwidthIdx           = ueSrsPrm.get_value_as<uint8_t>("bandwidthIdx");
            ueSrsPrmVec[ueIdx_within_cellGrp].combOffset             = ueSrsPrm.get_value_as<uint8_t>("combOffset");
            ueSrsPrmVec[ueIdx_within_cellGrp].cyclicShift            = ueSrsPrm.get_value_as<uint8_t>("cyclicShift");
            ueSrsPrmVec[ueIdx_within_cellGrp].frequencyPosition      = ueSrsPrm.get_value_as<uint8_t>("frequencyPosition");
            ueSrsPrmVec[ueIdx_within_cellGrp].frequencyShift         = ueSrsPrm.get_value_as<uint16_t>("frequencyShift");
            ueSrsPrmVec[ueIdx_within_cellGrp].frequencyHopping       = ueSrsPrm.get_value_as<uint8_t>("frequencyHopping");
            ueSrsPrmVec[ueIdx_within_cellGrp].resourceType           = ueSrsPrm.get_value_as<uint8_t>("resourceType");
            ueSrsPrmVec[ueIdx_within_cellGrp].Tsrs                   = ueSrsPrm.get_value_as<uint16_t>("Tsrs");
            ueSrsPrmVec[ueIdx_within_cellGrp].Toffset                = ueSrsPrm.get_value_as<uint16_t>("Toffset");
            ueSrsPrmVec[ueIdx_within_cellGrp].groupOrSequenceHopping = ueSrsPrm.get_value_as<uint8_t>("groupOrSequenceHopping");
            ueSrsPrmVec[ueIdx_within_cellGrp].prgSize                = ueSrsPrm.get_value_as<uint16_t>("prgSize");
            ueSrsPrmVec[ueIdx_within_cellGrp].RNTI                   = ueSrsPrm.get_value_as<uint16_t>("RNTI");
            ueSrsPrmVec[ueIdx_within_cellGrp].chEstBuffIdx           = ueIdx_within_cellGrp;

            uint32_t srsAntPortToUeAntMap = ueSrsPrm.get_value_as<uint32_t>("srsAntPortToUeAntMap");
            for(int antPortIdx = 0; antPortIdx < ueSrsPrmVec[ueIdx_within_cellGrp].nAntPorts; ++antPortIdx)
            {
                uint8_t bitShift = antPortIdx * 8;
                ueSrsPrmVec[ueIdx_within_cellGrp].srsAntPortToUeAntMap[antPortIdx] = static_cast<uint8_t>(srsAntPortToUeAntMap >> bitShift);
            }

            ueIdx_within_cellGrp += 1;
        }
    }

    // cell group api parameters:
    cellGrpDynPrm.nCells     = nCells;
    cellGrpDynPrm.pCellPrms  = cellDynPrmVec.data();
    cellGrpDynPrm.nSrsUes    = nSrsUesInCellGrp;
    cellGrpDynPrm.pUeSrsPrms = ueSrsPrmVec.data();

    // load input tensors:
    tDataRxVec.resize(nCells);
    tPrmDataRxVec.resize(nCells);
    for(int cellIdx = 0; cellIdx < nCells; cellIdx++)
    {
        hdf5hpp::hdf5_file  fInput = hdf5hpp::hdf5_file::open(inputFileNameVec[cellIdx].c_str());

       tDataRxVec[cellIdx]          = cuphy::tensor_from_dataset(fInput.open_dataset("DataRx"), CUPHY_C_16F , cuphy::tensor_flags::align_tight, cuStrm);
       tPrmDataRxVec[cellIdx].desc  = tDataRxVec[cellIdx].desc().handle();
       tPrmDataRxVec[cellIdx].pAddr = tDataRxVec[cellIdx].addr();
    }

    // input API parameters:
    dataIn.pTDataRx = tPrmDataRxVec.data();

    // allocate output buffers:
    srsChEstBuffInfoVec.resize(nSrsUesInCellGrp);
    rbSnrVec = std::move(cuphy::buffer<float, cuphy::pinned_alloc>(nSrsUesInCellGrp * 273)); // TODO: assuming each UE use a buffer for 273 PRBs

    rbSnrBuffOffsetVec.resize(nSrsUesInCellGrp);
    srsReportVec = std::move(cuphy::buffer<cuphySrsReport_t, cuphy::pinned_alloc>(nSrsUesInCellGrp));
    chEstCpuBuffVec.resize(nSrsUesInCellGrp);
    chEstToL2Vec.resize(nSrsUesInCellGrp);

    ueIdx_within_cellGrp = 0;
    for(int cellIdx = 0; cellIdx < nCells; cellIdx++)
    {
        hdf5hpp::hdf5_file      fInput                     = hdf5hpp::hdf5_file::open(inputFileNameVec[cellIdx].c_str());
        hdf5hpp::hdf5_dataset   srsChEstBufferInfo_dataset = fInput.open_dataset("srsChEstBufferInfo");

        for(int ueIdx_within_cell = 0; ueIdx_within_cell < nUesPerCellVec[cellIdx]; ++ueIdx_within_cell)
        {
            // Allocate GPU buffer for ChEst:
            cuphy::cuphyHDF5_struct ueSrsBuffInfo = cuphy::get_HDF5_struct_index(srsChEstBufferInfo_dataset, ueIdx_within_cell);

            uint16_t nRxAnt      = ueSrsBuffInfo.get_value_as<uint16_t>("nRxAnt");
            uint16_t nEsts       = ueSrsBuffInfo.get_value_as<uint16_t>("nPrbGrps");
            uint8_t  nUeAnt      = ueSrsBuffInfo.get_value_as<uint8_t>("nUeAnt");
            uint16_t startPrbGrp = ueSrsBuffInfo.get_value_as<uint16_t>("startPrbGrp");

            tSrsChEstVec.push_back(cuphy::tensor_device(CUPHY_C_16F,
                                                        nEsts,
                                                        nRxAnt,
                                                        nUeAnt,
                                                        cuphy::tensor_flags::align_tight));

            size_t nBytesInChEstBuffer = nEsts * nRxAnt * nUeAnt * 4;
            CUDA_CHECK(cudaMemsetAsync(tSrsChEstVec[ueIdx_within_cellGrp].addr(), 0, nBytesInChEstBuffer, cuStrm)); // init ChEst buffer to zero

            // Allocate CPU buffer for ChEst:
            size_t maxBufferSize = 273 * 128 * 4 * sizeof(short2); // nPrgs x maxGnbAnts x maxUeAnts x sizeof(float2), TODO: assuming each UE use a buffer for 273 PRBs
            chEstCpuBuffVec[ueIdx_within_cellGrp] = std::move(cuphy::buffer<uint8_t, cuphy::pinned_alloc>(maxBufferSize));

            // pointers:
            srsChEstBuffInfoVec[ueIdx_within_cellGrp].tChEstBuffer.desc  = tSrsChEstVec[ueIdx_within_cellGrp].desc().handle();
            srsChEstBuffInfoVec[ueIdx_within_cellGrp].tChEstBuffer.pAddr = tSrsChEstVec[ueIdx_within_cellGrp].addr();
            srsChEstBuffInfoVec[ueIdx_within_cellGrp].startPrbGrp        = startPrbGrp;
            rbSnrBuffOffsetVec[ueIdx_within_cellGrp]                     = 273 * ueIdx_within_cellGrp; // TODO: assuming each UE use a buffer for 273 PRBs
            chEstToL2Vec[ueIdx_within_cellGrp].pChEstCpuBuff             = chEstCpuBuffVec[ueIdx_within_cellGrp].addr();
            ueIdx_within_cellGrp += 1;
        }
    }

    // output API parameters:
    dataOut.pChEstBuffInfo     = srsChEstBuffInfoVec.data();
    dataOut.pSrsReports        = srsReportVec.addr();
    dataOut.pRbSnrBuffer       = rbSnrVec.addr();
    dataOut.pRbSnrBuffOffsets  = rbSnrBuffOffsetVec.data();
    dataOut.pSrsChEstToL2      = chEstToL2Vec.data();
    dataOut.pDataRxSrs         = nullptr;

    // debug parameters:
    dynDbgPrm.enableApiLogging = 0;

    // dynamic API parameters:
    srsDynPrm.cuStream        = cuStrm;
    srsDynPrm.procModeBmsk    = procModeBmsk;
    srsDynPrm.pCellGrpDynPrm  = &cellGrpDynPrm;
    srsDynPrm.pDataIn         = &dataIn;
    srsDynPrm.pDataOut        = &dataOut;
    srsDynPrm.cpuCopyOn       = 1;
    srsDynPrm.pDynDbg         = &dynDbgPrm;

    StatusOutput = {cuphySrsStatusType_t::CUPHY_SRS_STATUS_SUCCESS_OR_UNTRACKED_ISSUE, MAX_UINT16, MAX_UINT16};
    srsDynPrm.pStatusOut = &StatusOutput;
}

//----------------------------------------------------------------------------------------------------------
// Dataset holds static SRS API parameters/data

srsStaticApiDataset::srsStaticApiDataset(const std::vector<std::string>& inputFileNameVec, cudaStream_t cuStrm, std::string outFileName)
{
    uint16_t nCells = inputFileNameVec.size();
    cellStatPrmVec.resize(nCells);

    // cell parameters:
    for(int cellIdx = 0; cellIdx < nCells; ++cellIdx)
    {
        hdf5hpp::hdf5_file      fInput        = hdf5hpp::hdf5_file::open(inputFileNameVec[cellIdx].c_str());
        cuphy::cuphyHDF5_struct srsCellParams = cuphy::get_HDF5_struct(fInput, "srsCellParams");

        cellStatPrmVec[cellIdx].mu         = srsCellParams.get_value_as<uint8_t>("mu");
        cellStatPrmVec[cellIdx].nRxAnt     = srsCellParams.get_value_as<uint16_t>("nRxAnt");
        cellStatPrmVec[cellIdx].nRxAntSrs  = srsCellParams.get_value_as<uint16_t>("nRxAntSrs");
        cellStatPrmVec[cellIdx].nPrbUlBwp  = srsCellParams.get_value_as<uint16_t>("nPrbs"); // number of PRBs
    }

    // chEst algorithm
    hdf5hpp::hdf5_file      fInput        = hdf5hpp::hdf5_file::open(inputFileNameVec[0].c_str());
    cuphy::cuphyHDF5_struct srsCellParams = cuphy::get_HDF5_struct(fInput, "srsCellParams");
    uint8_t chEstAlgoIdx = srsCellParams.get_value_as<uint8_t>("chEstAlgoIdx");
    switch(chEstAlgoIdx){
        default:
            throw std::runtime_error("Unknown Channel Estimate Algorithm Index: " + std::to_string(chEstAlgoIdx));
        case 0:
        srsStatPrms.chEstAlgo = SRS_CH_EST_ALGO_TYPE_MMSE;
        break;

        case 1:
        srsStatPrms.chEstAlgo = SRS_CH_EST_ALGO_TYPE_RKHS;
        break;
    }
    
    srsStatPrms.chEstToL2NormalizationAlgo = srsCellParams.get_value_as<uint8_t>("chEstToL2NormalizationAlgo");
    srsStatPrms.chEstToL2ConstantScaler = srsCellParams.get_value_as<float>("chEstToL2ConstantScaler");
    
    srsStatPrms.enableDelayOffsetCorrection = 1;

    tFocc_table          = cuphy::tensor_from_dataset(fInput.open_dataset("focc_table"), CUPHY_C_16F , cuphy::tensor_flags::align_tight, cuStrm);
    tPrmFocc_table.desc  = tFocc_table.desc().handle();
    tPrmFocc_table.pAddr = tFocc_table.addr();
    srsStatPrms.srsFilterPrms.tPrmFocc_table = tPrmFocc_table;

    tFocc_comb2_table          = cuphy::tensor_from_dataset(fInput.open_dataset("focc_table_comb2"), CUPHY_C_16F , cuphy::tensor_flags::align_tight, cuStrm);
    tPrmFocc_comb2_table.desc  = tFocc_comb2_table.desc().handle();
    tPrmFocc_comb2_table.pAddr = tFocc_comb2_table.addr();
    srsStatPrms.srsFilterPrms.tPrmFocc_comb2_table = tPrmFocc_comb2_table;

    tFocc_comb4_table          = cuphy::tensor_from_dataset(fInput.open_dataset("focc_table_comb4"), CUPHY_C_16F , cuphy::tensor_flags::align_tight, cuStrm);
    tPrmFocc_comb4_table.desc  = tFocc_comb4_table.desc().handle();
    tPrmFocc_comb4_table.pAddr = tFocc_comb4_table.addr();
    srsStatPrms.srsFilterPrms.tPrmFocc_comb4_table = tPrmFocc_comb4_table;

    tW_comb2_nPorts1_wide          = cuphy::tensor_from_dataset(fInput.open_dataset("W_comb2_nPorts1_wide"), CUPHY_C_16F , cuphy::tensor_flags::align_tight, cuStrm);
    tPrmW_comb2_nPorts1_wide.desc  = tW_comb2_nPorts1_wide.desc().handle();
    tPrmW_comb2_nPorts1_wide.pAddr = tW_comb2_nPorts1_wide.addr();
    srsStatPrms.srsFilterPrms.tPrmW_comb2_nPorts1_wide = tPrmW_comb2_nPorts1_wide;

    tW_comb2_nPorts2_wide          = cuphy::tensor_from_dataset(fInput.open_dataset("W_comb2_nPorts2_wide"), CUPHY_C_16F , cuphy::tensor_flags::align_tight, cuStrm);
    tPrmW_comb2_nPorts2_wide.desc  = tW_comb2_nPorts2_wide.desc().handle();
    tPrmW_comb2_nPorts2_wide.pAddr = tW_comb2_nPorts2_wide.addr();
    srsStatPrms.srsFilterPrms.tPrmW_comb2_nPorts2_wide = tPrmW_comb2_nPorts2_wide;

    tW_comb2_nPorts4_wide          = cuphy::tensor_from_dataset(fInput.open_dataset("W_comb2_nPorts4_wide"), CUPHY_C_16F , cuphy::tensor_flags::align_tight, cuStrm);
    tPrmW_comb2_nPorts4_wide.desc  = tW_comb2_nPorts4_wide.desc().handle();
    tPrmW_comb2_nPorts4_wide.pAddr = tW_comb2_nPorts4_wide.addr();
    srsStatPrms.srsFilterPrms.tPrmW_comb2_nPorts4_wide = tPrmW_comb2_nPorts4_wide;

    tW_comb2_nPorts8_wide          = cuphy::tensor_from_dataset(fInput.open_dataset("W_comb2_nPorts8_wide"), CUPHY_C_16F , cuphy::tensor_flags::align_tight, cuStrm);
    tPrmW_comb2_nPorts8_wide.desc  = tW_comb2_nPorts8_wide.desc().handle();
    tPrmW_comb2_nPorts8_wide.pAddr = tW_comb2_nPorts8_wide.addr();
    srsStatPrms.srsFilterPrms.tPrmW_comb2_nPorts8_wide = tPrmW_comb2_nPorts8_wide;

    tW_comb4_nPorts1_wide          = cuphy::tensor_from_dataset(fInput.open_dataset("W_comb4_nPorts1_wide"), CUPHY_C_16F , cuphy::tensor_flags::align_tight, cuStrm);
    tPrmW_comb4_nPorts1_wide.desc  = tW_comb4_nPorts1_wide.desc().handle();
    tPrmW_comb4_nPorts1_wide.pAddr = tW_comb4_nPorts1_wide.addr();
    srsStatPrms.srsFilterPrms.tPrmW_comb4_nPorts1_wide = tPrmW_comb4_nPorts1_wide;

    tW_comb4_nPorts2_wide          = cuphy::tensor_from_dataset(fInput.open_dataset("W_comb4_nPorts2_wide"), CUPHY_C_16F , cuphy::tensor_flags::align_tight, cuStrm);
    tPrmW_comb4_nPorts2_wide.desc  = tW_comb4_nPorts2_wide.desc().handle();
    tPrmW_comb4_nPorts2_wide.pAddr = tW_comb4_nPorts2_wide.addr();
    srsStatPrms.srsFilterPrms.tPrmW_comb4_nPorts2_wide = tPrmW_comb4_nPorts2_wide;

    tW_comb4_nPorts4_wide          = cuphy::tensor_from_dataset(fInput.open_dataset("W_comb4_nPorts4_wide"), CUPHY_C_16F , cuphy::tensor_flags::align_tight, cuStrm);
    tPrmW_comb4_nPorts4_wide.desc  = tW_comb4_nPorts4_wide.desc().handle();
    tPrmW_comb4_nPorts4_wide.pAddr = tW_comb4_nPorts4_wide.addr();
    srsStatPrms.srsFilterPrms.tPrmW_comb4_nPorts4_wide = tPrmW_comb4_nPorts4_wide;

    tW_comb4_nPorts6_wide          = cuphy::tensor_from_dataset(fInput.open_dataset("W_comb4_nPorts6_wide"), CUPHY_C_16F , cuphy::tensor_flags::align_tight, cuStrm);
    tPrmW_comb4_nPorts6_wide.desc  = tW_comb4_nPorts6_wide.desc().handle();
    tPrmW_comb4_nPorts6_wide.pAddr = tW_comb4_nPorts6_wide.addr();
    srsStatPrms.srsFilterPrms.tPrmW_comb4_nPorts6_wide = tPrmW_comb4_nPorts6_wide;

    tW_comb4_nPorts12_wide          = cuphy::tensor_from_dataset(fInput.open_dataset("W_comb4_nPorts12_wide"), CUPHY_C_16F , cuphy::tensor_flags::align_tight, cuStrm);
    tPrmW_comb4_nPorts12_wide.desc  = tW_comb4_nPorts12_wide.desc().handle();
    tPrmW_comb4_nPorts12_wide.pAddr = tW_comb4_nPorts12_wide.addr();
    srsStatPrms.srsFilterPrms.tPrmW_comb4_nPorts12_wide = tPrmW_comb4_nPorts12_wide;

    tW_comb2_nPorts1_narrow          = cuphy::tensor_from_dataset(fInput.open_dataset("W_comb2_nPorts1_narrow"), CUPHY_C_16F , cuphy::tensor_flags::align_tight, cuStrm);
    tPrmW_comb2_nPorts1_narrow.desc  = tW_comb2_nPorts1_narrow.desc().handle();
    tPrmW_comb2_nPorts1_narrow.pAddr = tW_comb2_nPorts1_narrow.addr();
    srsStatPrms.srsFilterPrms.tPrmW_comb2_nPorts1_narrow = tPrmW_comb2_nPorts1_narrow;

    tW_comb2_nPorts2_narrow          = cuphy::tensor_from_dataset(fInput.open_dataset("W_comb2_nPorts2_narrow"), CUPHY_C_16F , cuphy::tensor_flags::align_tight, cuStrm);
    tPrmW_comb2_nPorts2_narrow.desc  = tW_comb2_nPorts2_narrow.desc().handle();
    tPrmW_comb2_nPorts2_narrow.pAddr = tW_comb2_nPorts2_narrow.addr();
    srsStatPrms.srsFilterPrms.tPrmW_comb2_nPorts2_narrow = tPrmW_comb2_nPorts2_narrow;

    tW_comb2_nPorts4_narrow          = cuphy::tensor_from_dataset(fInput.open_dataset("W_comb2_nPorts4_narrow"), CUPHY_C_16F , cuphy::tensor_flags::align_tight, cuStrm);
    tPrmW_comb2_nPorts4_narrow.desc  = tW_comb2_nPorts4_narrow.desc().handle();
    tPrmW_comb2_nPorts4_narrow.pAddr = tW_comb2_nPorts4_narrow.addr();
    srsStatPrms.srsFilterPrms.tPrmW_comb2_nPorts4_narrow = tPrmW_comb2_nPorts4_narrow;

    tW_comb2_nPorts8_narrow          = cuphy::tensor_from_dataset(fInput.open_dataset("W_comb2_nPorts8_narrow"), CUPHY_C_16F , cuphy::tensor_flags::align_tight, cuStrm);
    tPrmW_comb2_nPorts8_narrow.desc  = tW_comb2_nPorts8_narrow.desc().handle();
    tPrmW_comb2_nPorts8_narrow.pAddr = tW_comb2_nPorts8_narrow.addr();
    srsStatPrms.srsFilterPrms.tPrmW_comb2_nPorts8_narrow = tPrmW_comb2_nPorts8_narrow;

    tW_comb4_nPorts1_narrow          = cuphy::tensor_from_dataset(fInput.open_dataset("W_comb4_nPorts1_narrow"), CUPHY_C_16F , cuphy::tensor_flags::align_tight, cuStrm);
    tPrmW_comb4_nPorts1_narrow.desc  = tW_comb4_nPorts1_narrow.desc().handle();
    tPrmW_comb4_nPorts1_narrow.pAddr = tW_comb4_nPorts1_narrow.addr();
    srsStatPrms.srsFilterPrms.tPrmW_comb4_nPorts1_narrow = tPrmW_comb4_nPorts1_narrow;

    tW_comb4_nPorts2_narrow          = cuphy::tensor_from_dataset(fInput.open_dataset("W_comb4_nPorts2_narrow"), CUPHY_C_16F , cuphy::tensor_flags::align_tight, cuStrm);
    tPrmW_comb4_nPorts2_narrow.desc  = tW_comb4_nPorts2_narrow.desc().handle();
    tPrmW_comb4_nPorts2_narrow.pAddr = tW_comb4_nPorts2_narrow.addr();
    srsStatPrms.srsFilterPrms.tPrmW_comb4_nPorts2_narrow = tPrmW_comb4_nPorts2_narrow;

    tW_comb4_nPorts4_narrow          = cuphy::tensor_from_dataset(fInput.open_dataset("W_comb4_nPorts4_narrow"), CUPHY_C_16F , cuphy::tensor_flags::align_tight, cuStrm);
    tPrmW_comb4_nPorts4_narrow.desc  = tW_comb4_nPorts4_narrow.desc().handle();
    tPrmW_comb4_nPorts4_narrow.pAddr = tW_comb4_nPorts4_narrow.addr();
    srsStatPrms.srsFilterPrms.tPrmW_comb4_nPorts4_narrow = tPrmW_comb4_nPorts4_narrow;

    tW_comb4_nPorts6_narrow          = cuphy::tensor_from_dataset(fInput.open_dataset("W_comb4_nPorts6_narrow"), CUPHY_C_16F , cuphy::tensor_flags::align_tight, cuStrm);
    tPrmW_comb4_nPorts6_narrow.desc  = tW_comb4_nPorts6_narrow.desc().handle();
    tPrmW_comb4_nPorts6_narrow.pAddr = tW_comb4_nPorts6_narrow.addr();
    srsStatPrms.srsFilterPrms.tPrmW_comb4_nPorts6_narrow = tPrmW_comb4_nPorts6_narrow;

    tW_comb4_nPorts12_narrow          = cuphy::tensor_from_dataset(fInput.open_dataset("W_comb4_nPorts12_narrow"), CUPHY_C_16F , cuphy::tensor_flags::align_tight, cuStrm);
    tPrmW_comb4_nPorts12_narrow.desc  = tW_comb4_nPorts12_narrow.desc().handle();
    tPrmW_comb4_nPorts12_narrow.pAddr = tW_comb4_nPorts12_narrow.addr();
    srsStatPrms.srsFilterPrms.tPrmW_comb4_nPorts12_narrow = tPrmW_comb4_nPorts12_narrow;

    cuphy::cuphyHDF5_struct debiasPrms = cuphy::get_HDF5_struct(fInput, "debiasPrms");
    srsStatPrms.srsFilterPrms.noisEstDebias_comb2_nPorts1 = debiasPrms.get_value_as<float>("noisEstDebias_comb2_nPorts1");
    srsStatPrms.srsFilterPrms.noisEstDebias_comb2_nPorts2 = debiasPrms.get_value_as<float>("noisEstDebias_comb2_nPorts2");
    srsStatPrms.srsFilterPrms.noisEstDebias_comb2_nPorts4 = debiasPrms.get_value_as<float>("noisEstDebias_comb2_nPorts4");
    srsStatPrms.srsFilterPrms.noisEstDebias_comb2_nPorts8 = debiasPrms.get_value_as<float>("noisEstDebias_comb2_nPorts8");

    srsStatPrms.srsFilterPrms.noisEstDebias_comb4_nPorts1  = debiasPrms.get_value_as<float>("noisEstDebias_comb4_nPorts1");
    srsStatPrms.srsFilterPrms.noisEstDebias_comb4_nPorts2  = debiasPrms.get_value_as<float>("noisEstDebias_comb4_nPorts2");
    srsStatPrms.srsFilterPrms.noisEstDebias_comb4_nPorts4  = debiasPrms.get_value_as<float>("noisEstDebias_comb4_nPorts4");
    srsStatPrms.srsFilterPrms.noisEstDebias_comb4_nPorts6  = debiasPrms.get_value_as<float>("noisEstDebias_comb4_nPorts6");
    srsStatPrms.srsFilterPrms.noisEstDebias_comb4_nPorts12 = debiasPrms.get_value_as<float>("noisEstDebias_comb4_nPorts12");

    // debug parameters:
    bOutputFileName             = outFileName;
    statDbgPrm.pOutFileName     = bOutputFileName.empty() ? nullptr : bOutputFileName.c_str();
    statDbgPrm.enableApiLogging = 0;

    // static parameters:
    srsStatPrms.nMaxCells        = nCells;
    srsStatPrms.pCellStatPrms    = cellStatPrmVec.data();
    srsStatPrms.nMaxCellsPerSlot = nCells;
    srsStatPrms.pStatDbg         = &statDbgPrm;

    // GPU memory footprint tracking
    srsTracker.pMemoryFootprint  = nullptr;
    srsStatPrms.pOutInfo       = &srsTracker;
    srsStatPrms.enableBatchedMemcpy = 1;

    // rkhs paramaters
    if(srsStatPrms.chEstAlgo == SRS_CH_EST_ALGO_TYPE_RKHS)
    {
        cuphy::cuphyHDF5_struct  refSrsRkhsPrms = cuphy::get_HDF5_struct(fInput, "srsRkhsPrms");
        srsRkhsPrms.nGridSizes                  = refSrsRkhsPrms.get_value_as<uint8_t>("nGrids");
        srsStatPrms.pSrsRkhsPrms                = &srsRkhsPrms;

        cpuBufferRkhsGridPrms     = std::move(cuphy::buffer<cuphyRkhsGridPrms_t, cuphy::pinned_alloc>(srsRkhsPrms.nGridSizes));
        srsRkhsPrms.pRkhsGridPrms = cpuBufferRkhsGridPrms.addr();

        for(int i = 0; i < srsRkhsPrms.nGridSizes; ++i)
        {
            cuphyRkhsGridPrms_t& gridPrms = cpuBufferRkhsGridPrms[i];

            std::string str1 = "srsRkhs_eigenVecs_grid";
            std::string str2 = std::to_string(i);
            std::string str3 = str1 + str2;
            eigenVecTable[i]         = cuphy::tensor_from_dataset(fInput.open_dataset(str3.c_str()), CUPHY_R_16F, cuphy::tensor_flags::align_tight, cuStrm);
            gridPrms.eigenVecs.desc  = eigenVecTable[i].desc().handle();
            gridPrms.eigenVecs.pAddr = eigenVecTable[i].addr();

            str1 = "srsRkhs_eigValues_grid";
            str3 = str1 + str2;
            eigenValueTable[i]         = cuphy::tensor_from_dataset(fInput.open_dataset(str3.c_str()), CUPHY_R_16F, cuphy::tensor_flags::align_tight, cuStrm);
            gridPrms.eigenValues.desc  = eigenValueTable[i].desc().handle();
            gridPrms.eigenValues.pAddr = eigenValueTable[i].addr();

            str1 = "srsRkhs_eigenCorr_grid";
            str3 = str1 + str2;
            eigenCorrTable[i]        = cuphy::tensor_from_dataset(fInput.open_dataset(str3.c_str()), CUPHY_C_16F, cuphy::tensor_flags::align_tight, cuStrm);
            gridPrms.eigenCorr.desc  = eigenCorrTable[i].desc().handle();
            gridPrms.eigenCorr.pAddr = eigenCorrTable[i].addr();
        }

        cuphyRkhsGridPrms_t& gridPrms = cpuBufferRkhsGridPrms[2];

        std::string str = "srsRkhs_secondStageFourierPerm_grid2";
        secondStageFourierPermTable = cuphy::tensor_from_dataset(fInput.open_dataset(str.c_str()), CUPHY_R_8U, cuphy::tensor_flags::align_tight, cuStrm);
        gridPrms.secondStageFourierPerm.desc  = secondStageFourierPermTable.desc().handle();
        gridPrms.secondStageFourierPerm.pAddr = secondStageFourierPermTable.addr();

        str = "srsRkhs_secondStageTwiddleFactors_grid2";
        secondStageTwiddleFactorsTable = cuphy::tensor_from_dataset(fInput.open_dataset(str.c_str()), CUPHY_C_16F, cuphy::tensor_flags::align_tight, cuStrm);
        gridPrms.secondStageTwiddleFactors.desc  = secondStageTwiddleFactorsTable.desc().handle();
        gridPrms.secondStageTwiddleFactors.pAddr = secondStageTwiddleFactorsTable.addr();
    }
}


//----------------------------------------------------------------------------------------------------------
// Dataset holds static SRS API parameters/data

srsEvalDataset::srsEvalDataset(const std::vector<std::string>& inputFileNameVec, cudaStream_t cuStrm)
{
    uint16_t nCells                = inputFileNameVec.size();
    uint16_t ueIdx_within_cellGrp  = 0;

    for(int cellIdx = 0; cellIdx < nCells; ++cellIdx)
    {
        hdf5hpp::hdf5_file      fInput                 = hdf5hpp::hdf5_file::open(inputFileNameVec[cellIdx].c_str());
        cuphy::cuphyHDF5_struct srsCellParams          = cuphy::get_HDF5_struct(fInput, "srsCellParams");
        uint16_t                nUesInCell             = srsCellParams.get_value_as<uint16_t>("nSrsUes");
        hdf5hpp::hdf5_dataset   widebandReport_dataset = fInput.open_dataset("widebandSrsStats");

        for(int ueIdx_within_cell = 0; ueIdx_within_cell < nUesInCell; ++ueIdx_within_cell)
        {
            std::string          datasetName = "HestUe" + std::to_string(ueIdx_within_cell);
            cuphy::tensor_pinned Hest        = cuphy::tensor_from_dataset(fInput.open_dataset(datasetName.c_str()), CUPHY_C_32F, cuphy::tensor_flags::align_tight, cuStrm);
            srsChEstBuffsRef.emplace_back(Hest.layout());
            srsChEstBuffsRef[ueIdx_within_cellGrp] = Hest;

            datasetName = "HestNormToL2Ue" + std::to_string(ueIdx_within_cell);
            cuphy::tensor_pinned HestToL2    = cuphy::tensor_from_dataset(fInput.open_dataset(datasetName.c_str()), CUPHY_C_32I, cuphy::tensor_flags::align_tight, cuStrm);
            srsChEstToL2BuffsRef.emplace_back(HestToL2.layout());
            srsChEstToL2BuffsRef[ueIdx_within_cellGrp] = HestToL2;

            datasetName = "rbSnrsUe" + std::to_string(ueIdx_within_cell);
            cuphy::tensor_pinned rbSnrs      = cuphy::tensor_from_dataset(fInput.open_dataset(datasetName.c_str()), CUPHY_R_32F, cuphy::tensor_flags::align_tight, cuStrm);
            rbSnrsRef.emplace_back(rbSnrs.layout());
            rbSnrsRef[ueIdx_within_cellGrp] = rbSnrs;

            cuphy::cuphyHDF5_struct widebandReport = cuphy::get_HDF5_struct_index(widebandReport_dataset, ueIdx_within_cell);
            cuphySrsReport_t ueSrsReport;
            ueSrsReport.widebandSnr   = widebandReport.get_value_as<float>("widebandSnr");
            ueSrsReport.toEstMicroSec = widebandReport.get_value_as<float>("toEstMicroSec");
            ueSrsReport.widebandCsCorrRatioDb   = widebandReport.get_value_as<float>("widebandCsCorrRatioDb");
            srsReportsRef.emplace_back(ueSrsReport);
            srsStartValidPrgRef.emplace_back(widebandReport.get_value_as<uint16_t>("startValidPrg"));
            srsNValidPrgRef.emplace_back(widebandReport.get_value_as<uint16_t>("nValidPrg"));
            ueIdx_within_cellGrp += 1;
        }
    }
}

void srsEvalDataset::evalSrsRx(cuphySrsDynPrms_t& srsDynPrm, std::vector<cuphy::tensor_device>& tSrsChEstVec, float* h_rbSnrsCuphy, cuphySrsReport_t* h_srsReportsCuphy, cudaStream_t cuStrm)
{
    cuphySrsDataOut_t* pDataOut = srsDynPrm.pDataOut;
    uint16_t           nSrsUes  = srsDynPrm.pCellGrpDynPrm->nSrsUes;

    // compare wideband SNR reports:
    float tolTimingOffset = 0.01;
    float tolRbSnr        = 1.0;
    float tolCsCorrDb     = 0.2; // TODO: need to check the error tolerance to pass
    bool mismatchFlag     = false;
    bool anyMismatch      = false;
    printf("\n\n comparing cuphy and reference SRS reports...");
    for(int ueIdx = 0; ueIdx < nSrsUes; ++ueIdx)
    {
        uint8_t highDensityAntPortFlag = h_srsReportsCuphy[ueIdx].highDensityAntPortFlag;

        float errorTimingOffset = abs(h_srsReportsCuphy[ueIdx].toEstMicroSec - srsReportsRef[ueIdx].toEstMicroSec);
        if(!(isnan(h_srsReportsCuphy[ueIdx].toEstMicroSec) && isnan(srsReportsRef[ueIdx].toEstMicroSec)) &&
           (!isfinite(h_srsReportsCuphy[ueIdx].toEstMicroSec) || errorTimingOffset > tolTimingOffset))
        {
            printf("\n Timing offset mismatch detected for UE %d: toEstMicroSecCuphy = %f, toEstMicroSecRef = %f", ueIdx, h_srsReportsCuphy[ueIdx].toEstMicroSec, srsReportsRef[ueIdx].toEstMicroSec);
            mismatchFlag = true;
        }


        if(h_srsReportsCuphy[ueIdx].highDensityAntPortFlag == 0)
        {
            float errorWidebandSnr = abs(h_srsReportsCuphy[ueIdx].widebandSnr - srsReportsRef[ueIdx].widebandSnr);
            if(!(isnan(h_srsReportsCuphy[ueIdx].widebandSnr) && isnan(srsReportsRef[ueIdx].widebandSnr)) &&
            (!isfinite(h_srsReportsCuphy[ueIdx].widebandSnr) || errorWidebandSnr > tolRbSnr))
            {
                printf("\n Wideband SNR mismatch detected for UE %d: widebandSnrCuphy = %f, widebandSnrRef = %f", ueIdx, h_srsReportsCuphy[ueIdx].widebandSnr, srsReportsRef[ueIdx].widebandSnr);
                mismatchFlag = true;
            }
        }

        // estimation might not be accurate above 50dB, capped at 50dB for further comparison
        float refCsCorrRatioDb           = std::min(srsReportsRef[ueIdx].widebandCsCorrRatioDb,50.0f);
        float resCsCorrRatioDb           = std::min(h_srsReportsCuphy[ueIdx].widebandCsCorrRatioDb,50.0f);
        float errorwidebandCsCorrRatioDb = abs(resCsCorrRatioDb - refCsCorrRatioDb);
        if(!(isnan(h_srsReportsCuphy[ueIdx].widebandCsCorrRatioDb) && isnan(srsReportsRef[ueIdx].widebandCsCorrRatioDb)) &&
           (errorwidebandCsCorrRatioDb > tolCsCorrDb))
        {
            printf("\n Wideband cyclic shift correlation ratio mismatch detected for UE %d: widebandCsCorrRatioDbCuphy = %f, widebandCsCorrRatioDbRef = %f", ueIdx, h_srsReportsCuphy[ueIdx].widebandCsCorrRatioDb, srsReportsRef[ueIdx].widebandCsCorrRatioDb);
            mismatchFlag = true;
        }
    }
    if(mismatchFlag == false)
    {
        printf("\n no mismatches detected");
    }
    printf("\n");
    anyMismatch |= mismatchFlag;

    // compare Rb SNRs:
    uint32_t* pRbSnrBuffOffsets = pDataOut->pRbSnrBuffOffsets;
    float rbSnrTol = 20;
    mismatchFlag = false;

    printf("\n comparing cuphy and reference Rb Snrs...");
    for(int ueIdx = 0; ueIdx < nSrsUes; ++ueIdx)
    {
        float totError = 0;
        float totEnegy = 0;
        for(int prbIdx = 0; prbIdx < 272; ++prbIdx)
        {
            float rbSnrCuphy = h_rbSnrsCuphy[pRbSnrBuffOffsets[ueIdx] + prbIdx];
            float rbSnrRef   = rbSnrsRef[ueIdx](prbIdx);

            float e = rbSnrRef*rbSnrRef;
            if(e > 0.1f)
            {
                totEnegy += rbSnrRef*rbSnrRef;
                totError += (rbSnrRef - rbSnrCuphy) * (rbSnrRef - rbSnrCuphy);
            }
        }
        float evalSnr = 10*log10(totEnegy / totError);
        if(h_srsReportsCuphy[ueIdx].highDensityAntPortFlag == 0)
        {
            if(!(totEnegy == 0 && totError == 0) && (!isfinite(evalSnr) || evalSnr < rbSnrTol))
            {
                printf("\n Rb SNR mismatch detected for UE %d: evalSnr = %f", ueIdx, evalSnr);
                mismatchFlag = true;
            }
        }
    }
    if(mismatchFlag == false)
    {
        printf("\n no mismatches detected");
    }
    printf("\n");
    anyMismatch |= mismatchFlag;

    // compare Srs ChEstToL2:
    float chEstToL2SnrTol = 30;
    mismatchFlag          = false;
    printf("\n comparing cuphy and reference srs ChEstToL2...");
    for(int ueIdx = 0; ueIdx < nSrsUes; ++ueIdx)
    {
        const vec<int, CUPHY_DIM_MAX>& dim = srsChEstToL2BuffsRef[ueIdx].dimensions();

        int nPrbGrps  = dim[0];
        int nRxAnts   = dim[1];
        int nAntPorts = dim[2];

        cuphy::tensor_ref tRefSrsChEstToL2BuffCuphy;
        tRefSrsChEstToL2BuffCuphy.desc().set(CUPHY_C_16I, nPrbGrps, nRxAnts, nAntPorts, cuphy::tensor_flags::align_tight);
        tRefSrsChEstToL2BuffCuphy.set_addr(pDataOut->pSrsChEstToL2[ueIdx].pChEstCpuBuff);

        cuphy::typed_tensor<CUPHY_C_16I, cuphy::pinned_alloc> tSrsChEstToL2Buff(nPrbGrps, nRxAnts, nAntPorts, cuphy::tensor_flags::align_tight);
        tSrsChEstToL2Buff = tRefSrsChEstToL2BuffCuphy;
        
        double avgSignalEnergy, avgErrorEnergy;
        computeSnrComponentsShort(avgSignalEnergy, avgErrorEnergy, tSrsChEstToL2Buff, srsChEstToL2BuffsRef[ueIdx]);
        const double chEstToL2Snr = 10*log10(avgSignalEnergy / avgErrorEnergy);
        if(!((avgSignalEnergy == 0 && avgErrorEnergy == 0) || (isnan(avgSignalEnergy) && isnan(avgErrorEnergy))) &&
           (!isfinite(chEstToL2Snr) || chEstToL2Snr < chEstToL2SnrTol))
        {
            printf("\n chEstToL2 mismatch detected for UE %d: chEstToL2Snr = %f", ueIdx, chEstToL2Snr);
            mismatchFlag = true;
        }

    }
    if(mismatchFlag == false)
    {
        printf("\n no mismatches detected");
    }
    printf("\n");
    anyMismatch |= mismatchFlag;

    // compare Srs ChEst:
    cuphySrsChEstBuffInfo_t* pChEstBuffInfo = pDataOut->pChEstBuffInfo;
    float chEstSnrTol = 30;
    mismatchFlag      = false;
    printf("\n comparing cuphy and reference srs ChEst...");
    for(int ueIdx = 0; ueIdx < nSrsUes; ++ueIdx)
    {
        // copy chEst buffer to cpu
        cuphy::typed_tensor<CUPHY_C_16F, cuphy::pinned_alloc> tSrsHestCuphy(tSrsChEstVec[ueIdx].layout());
        tSrsHestCuphy.convert(tSrsChEstVec[ueIdx], cuStrm);
        CUDA_CHECK(cudaStreamSynchronize(cuStrm));

        // compute snr
        double avgSignalEnergy, avgErrorEnergy;
        computeSnrComponentsHalf(avgSignalEnergy, avgErrorEnergy, tSrsHestCuphy, srsChEstBuffsRef[ueIdx]);
        const double chEstSnr = 10*log10(avgSignalEnergy / avgErrorEnergy);
        if(!((avgSignalEnergy == 0 && avgErrorEnergy == 0) || (isnan(avgSignalEnergy) && isnan(avgErrorEnergy))) &&
           (!isfinite(chEstSnr) || chEstSnr < chEstSnrTol))
        {
            printf("\n chEst mismatch detected for UE %d: chEstSnr = %f", ueIdx, chEstSnr);
            mismatchFlag = true;
        }
    }
    
    
//    for(int ueIdx = 0; ueIdx < nSrsUes; ++ueIdx)
//    {
//        if(pChEstBuffInfo[ueIdx].startValidPrg!=srsStartValidPrgRef[ueIdx])
//        {
//            printf("\n UE[%d]: startValidPrg mismatch cuPHY = %d, ref = %d", ueIdx, pChEstBuffInfo[ueIdx].startValidPrg, srsStartValidPrgRef[ueIdx]);
//            mismatchFlag = true;
//            break;
//        }
//    }
//    
//    for(int ueIdx = 0; ueIdx < nSrsUes; ++ueIdx)
//    {
//        if(pChEstBuffInfo[ueIdx].nValidPrg!=srsNValidPrgRef[ueIdx])
//        {
//            printf("\n UE[%d]: nValidPrg mismatch cuPHY = %d, ref = %d", ueIdx, pChEstBuffInfo[ueIdx].nValidPrg, srsNValidPrgRef[ueIdx]);
//            mismatchFlag = true;
//            break;
//        }
//    }
    
    anyMismatch |= mismatchFlag;
    if(anyMismatch == false)
    {
        printf("\n SRS reference check: PASSED!");
    }
    printf("\n");

    if(anyMismatch == true)
    {
        printf("SRS reference check: FAILED! exiting\n");
        throw cuphy::cuphy_exception(CUPHY_STATUS_REF_MISMATCH);
    }
}

//----------------------------------------------------------------------------------------------------------
// Dataset holds dynamic BFW API parameters/data/buffers

bfwDynApiDataset::bfwDynApiDataset(const std::vector<std::string>& inputFileNameVec, cudaStream_t cuStrm, uint64_t procModeBmsk, OutputBufferMode outputBufferMode, int16_t beamIdOffset) 
: m_outputBufferMode(outputBufferMode)
{
    uint16_t nCells           = inputFileNameVec.size();
    uint16_t nUeGrpsInCellGrp = 0;
    uint16_t nLayerInCellGrp  = 0;
    uint16_t nSrsUesInCellGrp = 0;
    constexpr uint8_t bfwCoefBFP = 9;

    std::vector<uint16_t> nUeGrpsPerCellVec(nCells);
    std::vector<uint16_t> nLayersPerCellVec(nCells);
    std::vector<uint16_t> nSrsUesPerCellVec(nCells);

    // count total number of ueGrps and layers:
    uint16_t srsPrbGrpSize;
    for(int cellIdx = 0; cellIdx < nCells; cellIdx++)
    {
        hdf5hpp::hdf5_file      fInput = hdf5hpp::hdf5_file::open(inputFileNameVec[cellIdx].c_str());
        cuphy::cuphyHDF5_struct sizes  = cuphy::get_HDF5_struct(fInput, "sizes");

        nUeGrpsPerCellVec[cellIdx] = sizes.get_value_as<uint16_t>("nUeGrps");
        nLayersPerCellVec[cellIdx] = sizes.get_value_as<uint16_t>("nLayersTot");
        nSrsUesPerCellVec[cellIdx] = sizes.get_value_as<uint16_t>("nSrsUes");
        srsPrbGrpSize              = sizes.get_value_as<uint16_t>("srsPrbGrpSize");

        nUeGrpsInCellGrp  += nUeGrpsPerCellVec[cellIdx];
        nLayerInCellGrp   += nLayersPerCellVec[cellIdx];
        nSrsUesInCellGrp  += nSrsUesPerCellVec[cellIdx];
    }

    // load ueGrp and layer paramaters:
    bfwUeGrpPrmVec.resize(nUeGrpsInCellGrp);
    bfwLayerPrmVec.resize(nLayerInCellGrp);

    uint16_t            layerIdx_within_cellGrp   = 0;
    uint16_t            ueGrpIdx_within_cellGrp   = 0;
    cuphyBfwLayerPrm_t* pCellLayerPrms            = bfwLayerPrmVec.data();
    uint32_t            total_coefs               = 0;

    uint16_t chEstBufCellOffset = 0, coefBufCellOffset = 0;
    for(int cellIdx = 0; cellIdx < nCells; cellIdx++)
    {
        hdf5hpp::hdf5_file    fInput            = hdf5hpp::hdf5_file::open(inputFileNameVec[cellIdx].c_str());
        hdf5hpp::hdf5_dataset ueGrpPrms_dataset = fInput.open_dataset("cuphyBfwUeGrpPrm");
        hdf5hpp::hdf5_dataset layerPrms_dataset = fInput.open_dataset("cuphyBfwLayerPrm");

        for(int ueGrpIdx_within_cell = 0; ueGrpIdx_within_cell < nUeGrpsPerCellVec[cellIdx]; ueGrpIdx_within_cell++)
        {
            cuphy::cuphyHDF5_struct ueGrpPrm = cuphy::get_HDF5_struct_index(ueGrpPrms_dataset, ueGrpIdx_within_cell);

            bfwUeGrpPrmVec[ueGrpIdx_within_cellGrp].startPrbGrp    = ueGrpPrm.get_value_as<uint16_t>("startPrbGrp");
            bfwUeGrpPrmVec[ueGrpIdx_within_cellGrp].nPrbGrp        = ueGrpPrm.get_value_as<uint16_t>("nPrbGrp");
            bfwUeGrpPrmVec[ueGrpIdx_within_cellGrp].startPrb       = ueGrpPrm.get_value_as<uint16_t>("startPrb");
            bfwUeGrpPrmVec[ueGrpIdx_within_cellGrp].nRxAnt         = ueGrpPrm.get_value_as<uint16_t>("nRxAnt");
            bfwUeGrpPrmVec[ueGrpIdx_within_cellGrp].nBfLayers      = ueGrpPrm.get_value_as<uint8_t>("nBfLayers");
            bfwUeGrpPrmVec[ueGrpIdx_within_cellGrp].pBfLayerPrm    = pCellLayerPrms + ueGrpPrm.get_value_as<uint16_t>("bfLayerPrmStartIdx");
            bfwUeGrpPrmVec[ueGrpIdx_within_cellGrp].coefBufIdx     = coefBufCellOffset + ueGrpPrm.get_value_as<uint16_t>("coefBufIdx");
            bfwUeGrpPrmVec[ueGrpIdx_within_cellGrp].bfwPrbGrpSize  = ueGrpPrm.get_value_as<uint16_t>("bfwPrbGrpSize");
            bfwUeGrpPrmVec[ueGrpIdx_within_cellGrp].beamIdOffset   = beamIdOffset;
            total_coefs += bfwUeGrpPrmVec[ueGrpIdx_within_cellGrp].nRxAnt * bfwUeGrpPrmVec[ueGrpIdx_within_cellGrp].nBfLayers * bfwUeGrpPrmVec[ueGrpIdx_within_cellGrp].nPrbGrp;

            NVLOGD_FMT(NVLOG_BFW, "BfwCoefBuf: cell[{:02d}] ueGrp[{:02d}] nBfLayers {:01d} coefBufIdx {:03d}", cellIdx, ueGrpIdx_within_cellGrp, bfwUeGrpPrmVec[ueGrpIdx_within_cellGrp].nBfLayers, bfwUeGrpPrmVec[ueGrpIdx_within_cellGrp].coefBufIdx);
            ueGrpIdx_within_cellGrp += 1;
        }

        for(int layerIdx_within_cell = 0; layerIdx_within_cell < nLayersPerCellVec[cellIdx]; layerIdx_within_cell++)
        {
            cuphy::cuphyHDF5_struct layerPrm = cuphy::get_HDF5_struct_index(layerPrms_dataset, layerIdx_within_cell);

            bfwLayerPrmVec[layerIdx_within_cellGrp].chEstInfoBufIdx = chEstBufCellOffset + layerPrm.get_value_as<uint16_t>("chEstInfoBufIdx");
            bfwLayerPrmVec[layerIdx_within_cellGrp].ueLayerIndex    = layerPrm.get_value_as<uint8_t>("ueLayerIndex");

            NVLOGD_FMT(NVLOG_BFW, "ChEstInfoBuf: cell[{:02d}] ueLayerIdx[{:02d}] (cellGrpLayerIdx {:03d}) chEstInfoBufIdx {:03d}", cellIdx, bfwLayerPrmVec[layerIdx_within_cellGrp].ueLayerIndex, layerIdx_within_cellGrp, bfwLayerPrmVec[layerIdx_within_cellGrp].chEstInfoBufIdx);

            layerIdx_within_cellGrp += 1;
        }
        chEstBufCellOffset += nSrsUesPerCellVec[cellIdx];
        coefBufCellOffset += nUeGrpsPerCellVec[cellIdx];
        pCellLayerPrms +=  nLayersPerCellVec[cellIdx];
    }

   // populate cuphyBfwDynPrm
   bfwDynPrm.nUeGrps     = nUeGrpsInCellGrp;
   bfwDynPrm.pUeGrpPrms  = bfwUeGrpPrmVec.data();

   // load SRS databatabase
   tSrsChEstVec.resize(nSrsUesInCellGrp);
   srsChEstBufInfoVec.resize(nSrsUesInCellGrp);
   uint16_t srsIdx_within_cellGrp = 0;

    layerIdx_within_cellGrp = 0;
    for(int cellIdx = 0; cellIdx < nCells; cellIdx++)
    {
        hdf5hpp::hdf5_file   fInput        = hdf5hpp::hdf5_file::open(inputFileNameVec[cellIdx].c_str());
        cuphy::tensor_pinned tStartPrbGrp = cuphy::tensor_from_dataset(fInput.open_dataset("chEstBuf_startPrbGrps"), CUPHY_R_16U , cuphy::tensor_flags::align_tight, cuStrm);
        cudaStreamSynchronize(cuStrm);
        uint16_t*            pStartPrbGrp = static_cast<uint16_t*>(tStartPrbGrp.addr());
        
        cuphy::tensor_pinned tStartValidPrg = cuphy::tensor_from_dataset(fInput.open_dataset("startValidPrg"), CUPHY_R_16U , cuphy::tensor_flags::align_tight, cuStrm);
        cudaStreamSynchronize(cuStrm);
        uint16_t*            pStartValidPrg = static_cast<uint16_t*>(tStartValidPrg.addr());
        
        cuphy::tensor_pinned tNValidPrg = cuphy::tensor_from_dataset(fInput.open_dataset("nValidPrg"), CUPHY_R_16U , cuphy::tensor_flags::align_tight, cuStrm);
        cudaStreamSynchronize(cuStrm);
        uint16_t*            pNValidPrg = static_cast<uint16_t*>(tNValidPrg.addr());

        for(int srsIdx_within_cell = 0; srsIdx_within_cell < nSrsUesPerCellVec[cellIdx]; ++srsIdx_within_cell)
        {
            std::string  datasetName             = "chEstBufHalf" + std::to_string(srsIdx_within_cell);
            tSrsChEstVec[srsIdx_within_cellGrp]  = cuphy::tensor_from_dataset(fInput.open_dataset(datasetName.c_str()), CUPHY_C_16F , cuphy::tensor_flags::align_tight, cuStrm);

            srsChEstBufInfoVec[srsIdx_within_cellGrp].tChEstBuffer.desc  = tSrsChEstVec[srsIdx_within_cellGrp].desc().handle();
            srsChEstBufInfoVec[srsIdx_within_cellGrp].tChEstBuffer.pAddr = tSrsChEstVec[srsIdx_within_cellGrp].addr();
            srsChEstBufInfoVec[srsIdx_within_cellGrp].startPrbGrp        = pStartPrbGrp[srsIdx_within_cell];
            srsChEstBufInfoVec[srsIdx_within_cellGrp].srsPrbGrpSize      = srsPrbGrpSize;
            srsChEstBufInfoVec[srsIdx_within_cellGrp].startValidPrg      = pStartValidPrg[srsIdx_within_cell];
            srsChEstBufInfoVec[srsIdx_within_cellGrp].nValidPrg          = pNValidPrg[srsIdx_within_cell];

            NVLOGD_FMT(NVLOG_BFW, "SRSChEst: cell[{:02d}] ue[{:02d}] buf[{:03d}] Ref {}", cellIdx, srsIdx_within_cell, srsIdx_within_cellGrp, tSrsChEstVec[srsIdx_within_cellGrp].desc().get_info().to_string(false).c_str());

            srsIdx_within_cellGrp += 1;
        }
    }

    // allocate bfw buffers:
    bfwCompBufferVec.resize(nUeGrpsInCellGrp);
    bfwComppBufVec.resize(nUeGrpsInCellGrp);

    for(int ueGrpIdx_within_cellGrp = 0; ueGrpIdx_within_cellGrp < nUeGrpsInCellGrp; ++ueGrpIdx_within_cellGrp)
    {
        uint16_t nPrbGrp   = bfwUeGrpPrmVec[ueGrpIdx_within_cellGrp].nPrbGrp;
        uint16_t nRxAnt    = bfwUeGrpPrmVec[ueGrpIdx_within_cellGrp].nRxAnt;
        uint8_t  nBfLayers = bfwUeGrpPrmVec[ueGrpIdx_within_cellGrp].nBfLayers;
        uint16_t coefBufIdx = bfwUeGrpPrmVec[ueGrpIdx_within_cellGrp].coefBufIdx;

        size_t compBytes = nPrbGrp*nBfLayers*(div_round_up(nRxAnt*bfwCoefBFP*2,8) + 3);
        if(m_outputBufferMode == OutputBufferMode::Host)
        {
            bfwCompBufferVec[coefBufIdx] = std::make_unique<buffer_impl<cuphy::pinned_alloc>>(compBytes);
        }
        else
        {
            bfwCompBufferVec[coefBufIdx] = std::make_unique<buffer_impl<cuphy::device_alloc>>(compBytes);
        }
        bfwComppBufVec[coefBufIdx]   = static_cast<uint8_t*>(bfwCompBufferVec[coefBufIdx]->addr());
        // Reset the entire output buffer.
        CUDA_CHECK(cudaMemsetAsync(bfwComppBufVec[coefBufIdx], 0, compBytes, cuStrm)); // This is done in cuStrm
    }

    // input and output paramaters:
    dataIn.pChEstInfo = srsChEstBufInfoVec.data();
    dataOut.pBfwCoef  = bfwComppBufVec.data();

    // debug paramaters:
    dynDbgPrm.enableApiLogging = 1;


    // populate bfwDynPrms
    bfwDynPrms.cuStream     = cuStrm;
    bfwDynPrms.procModeBmsk = procModeBmsk;
    bfwDynPrms.pDynPrm      = &bfwDynPrm;
    bfwDynPrms.pDataIn      = &dataIn;
    bfwDynPrms.pDataOut     = &dataOut;
    bfwDynPrms.pDynDbg      = &dynDbgPrm;

    StatusOutput = {cuphyBfwStatusType_t::CUPHY_BFW_STATUS_SUCCESS_OR_UNTRACKED_ISSUE, MAX_UINT16, MAX_UINT16};
    bfwDynPrms.pStatusOut = &StatusOutput;
}


//----------------------------------------------------------------------------------------------------------
// Dataset holds static bfw API parameters/data

bfwStaticApiDataset::bfwStaticApiDataset(const std::vector<std::string>& inputFileNameVec, cudaStream_t cuStrm, std::string outFileName)
{
    // debug paramaters:
    bOutputFileName     = outFileName;
    statDbgPrm.enableApiLogging = 1;
    statDbgPrm.pOutFileName     = bOutputFileName.empty() ? nullptr : bOutputFileName.c_str();

    // Static paramaters:
    hdf5hpp::hdf5_file      fInput     = hdf5hpp::hdf5_file::open(inputFileNameVec[0].c_str());
    cuphy::cuphyHDF5_struct statPrmsH5 = cuphy::get_HDF5_struct(fInput, "bfwStatParms");

    bfwStatPrms.lambda          = statPrmsH5.get_value_as<float>("lambda");
    try
    {   
        bfwStatPrms.beta            = statPrmsH5.get_value_as<float>("beta");
    }
    catch(...)
    {
        bfwStatPrms.beta            = 16384;  // same with SimCtrl.bfw.beta in cfgSimCtrl.m
    }
    bfwStatPrms.bfwPowerNormAlg_selector = statPrmsH5.get_value_as<uint8_t>("bfwPowerNormAlg_selector");
    // Find max antennas and PRB Groups across all cells/UE groups
    uint8_t  nMaxGnbAnt  = 0;
    uint16_t nMaxPrbGrps = 0;
    for(int cellIdx = 0; cellIdx < inputFileNameVec.size(); cellIdx++)
    {
        hdf5hpp::hdf5_file    fInput            = hdf5hpp::hdf5_file::open(inputFileNameVec[cellIdx].c_str());
        cuphy::cuphyHDF5_struct sizes           = cuphy::get_HDF5_struct(fInput, "sizes");
        hdf5hpp::hdf5_dataset ueGrpPrms_dataset = fInput.open_dataset("cuphyBfwUeGrpPrm");

        uint16_t nUeGrps = sizes.get_value_as<uint16_t>("nUeGrps");

        for(int ueGrpIdx_within_cell = 0; ueGrpIdx_within_cell < nUeGrps; ueGrpIdx_within_cell++)
        {
            cuphy::cuphyHDF5_struct ueGrpPrm = cuphy::get_HDF5_struct_index(ueGrpPrms_dataset, ueGrpIdx_within_cell);

            nMaxGnbAnt  = std::max(nMaxGnbAnt,  (uint8_t)ueGrpPrm.get_value_as<uint16_t>("nRxAnt"));
            nMaxPrbGrps = std::max(nMaxPrbGrps, ueGrpPrm.get_value_as<uint16_t>("nPrbGrp"));
        }
    }
    bfwStatPrms.nMaxGnbAnt       = nMaxGnbAnt;
    bfwStatPrms.nMaxPrbGrps      = nMaxPrbGrps;
    bfwStatPrms.compressBitwidth = 9;
    bfwStatPrms.nMaxUeGrps       = CUPHY_BFW_COEF_COMP_N_MAX_USER_GRPS;
    bfwStatPrms.nMaxTotalLayers  = CUPHY_BFW_COEF_COMP_N_MAX_USER_GRPS * MAX_DL_LAYERS; //MAX_DL_LAYERS = 32
    bfwStatPrms.pStatDbg         = &statDbgPrm;

    // GPU memory footprint tracking
    bfwTracker.pMemoryFootprint  = nullptr;
    bfwStatPrms.pOutInfo         = &bfwTracker;
    bfwStatPrms.enableBatchedMemcpy = 1;
}

//----------------------------------------------------------------------------------------------------------
// Dataset holds buffers to evaluate bfw compute

bfwEvalDataset::bfwEvalDataset(const std::vector<std::string>& inputFileNameVec, cudaStream_t cuStrm)
{
    uint16_t nCells                   = inputFileNameVec.size();
    uint16_t ueGrpIdx_within_cellGrp  = 0;
    m_nCells = nCells;
    m_nUeGrpsInCell.resize(m_nCells);

    for(int cellIdx = 0; cellIdx < m_nCells; ++cellIdx)
    {
        hdf5hpp::hdf5_file      fInput        = hdf5hpp::hdf5_file::open(inputFileNameVec[cellIdx].c_str());
        cuphy::cuphyHDF5_struct sizes         = cuphy::get_HDF5_struct(fInput, "sizes");
        uint16_t                nUeGrpsInCell  = sizes.get_value_as<uint16_t>("nUeGrps");
        m_nUeGrpsInCell[cellIdx] = nUeGrpsInCell;

        for(int ueGrpIdx_within_cell = 0; ueGrpIdx_within_cell < nUeGrpsInCell; ++ueGrpIdx_within_cell)
        {
            std::string          datasetName = "bfwBuf" + std::to_string(ueGrpIdx_within_cell);
            cuphy::tensor_pinned tBfw        = cuphy::tensor_from_dataset(fInput.open_dataset(datasetName.c_str()), CUPHY_C_32F, cuphy::tensor_flags::align_tight, cuStrm);
            datasetName = "bfwCompBuf" + std::to_string(ueGrpIdx_within_cell);
            cuphy::tensor_pinned tBfwComp    = cuphy::tensor_from_dataset(fInput.open_dataset(datasetName.c_str()), CUPHY_R_8U,  cuphy::tensor_flags::align_tight, cuStrm);

            bfwBufRefVec.emplace_back(tBfw.layout());
            bfwBufRefVec[ueGrpIdx_within_cellGrp] = tBfw;
            bfwCompBufRefVec.emplace_back(tBfwComp.layout());
            bfwCompBufRefVec[ueGrpIdx_within_cellGrp] = tBfwComp;

            ueGrpIdx_within_cellGrp += 1;
        }
    }
}

bool bfwEvalDataset::bfwDecompressCompare(uint16_t ueGrpIdx, float beta, int bundleSize, uint8_t* input, int16_t beamIdOffset)
{
    // TODO do decompression using bfw_decompress_blockFP() function

    const cuphy::vec<int, CUPHY_DIM_MAX> refDim =  bfwBufRefVec[ueGrpIdx].layout().dimensions();
    const cuphy::vec<int, CUPHY_DIM_MAX> dim    =  bfwCompBufRefVec[ueGrpIdx].layout().dimensions();

    const int nAnt = refDim[0];
    const int nLayer = refDim[1];
    const int nPrbGrp = refDim[2];
    const int bfpBits = ((bundleSize - 1) * 8)/(nAnt*2);
    const bool pack_beams = beamIdOffset >= 0;

    if(bfpBits != 9){
        NVLOGE_FMT(NVLOG_BFW, AERIAL_CUPHY_EVENT, "Only 9 bit decompression validation supported");
        return false;
    }
    if(nAnt % 4 != 0){
        NVLOGE_FMT(NVLOG_BFW, AERIAL_CUPHY_EVENT, "Only multiples of 4 antennae supported for validation");
        return false;
    }

    NVLOGD_FMT(NVLOG_BFW, "nAnt {} nLayer {} nPrbGrp {}", nAnt, nLayer, nPrbGrp);
    float invbeta = 1.0 / beta;
    int mismatch = 0;

    for(int iLayer = 0; iLayer < nLayer; iLayer++)
    {
        for(int iPrbGrp = 0; iPrbGrp < nPrbGrp; iPrbGrp++)
        {
            int bundleOffset = bundleSize * (iLayer*nPrbGrp + iPrbGrp);
            int shift = input[bundleOffset];
            if(pack_beams){
                uint32_t beamId =  ntohs(*reinterpret_cast<uint16_t*>(input + bundleOffset + 1));
                const auto ref_beam_id = iLayer*CUPHY_BFW_N_MAX_PRB_GRPS + iPrbGrp + beamIdOffset;
                if(ref_beam_id != beamId){
                    NVLOGE_FMT(NVLOG_BFW, AERIAL_CUPHY_EVENT, "Error! Mismatch for beamId: expected={} vs. gpu={}", ref_beam_id, beamId);
                    mismatch++;
                }
            }
            
            for(int iAnt = 0; iAnt*4 < nAnt; iAnt++)
            {
                int32_t  vi[4],vq[4];
                int offset = bundleOffset + iAnt*bfpBits + 1 + 2*(pack_beams);

                // Unpack
                vi[0] = (input[offset] << 1) | (input[offset + 1] >> 7);              // 8 + 1, remains 7
                vq[0] = ((input[offset + 1] & 0x7f) << 2) | (input[offset + 2] >> 6); // 7 + 2, remains 6
                vi[1] = ((input[offset + 2] & 0x3f) << 3) | (input[offset + 3] >> 5); // 6 + 3, remains 5
                vq[1] = ((input[offset + 3] & 0x1f) << 4) | (input[offset + 4] >> 4); // 5 + 4, remains 4
                vi[2] = ((input[offset + 4] & 0x0f) << 5) | (input[offset + 5] >> 3); // 4 + 5, remains 3
                vq[2] = ((input[offset + 5] & 0x07) << 6) | (input[offset + 6] >> 2); // 3 + 6, remains 2
                vi[3] = ((input[offset + 6] & 0x03) << 7) | (input[offset + 7] >> 1); // 2 + 7, remains 1
                vq[3] = ((input[offset + 7] & 0x01) << 8) |  input[offset + 8];       // 1 + 8

                // Sign extend to 32-bits
                for(int i = 0; i < 4; i++)
                {
                    float2 bfw_val;
                    vi[i] = (vi[i] << (32 - bfpBits)) >> (32 - bfpBits - shift);
                    vq[i] = (vq[i] << (32 - bfpBits)) >> (32 - bfpBits - shift);
                    bfw_val.x = ((float)vi[i] * invbeta);
                    bfw_val.y = ((float)vq[i] * invbeta);
                    float2 refBfwVal = bfwBufRefVec[ueGrpIdx](4*iAnt+i,iLayer,iPrbGrp);
                    //if(!complex_approx_equal<float2,float>(bfw_val,refBfwVal,1.0/256.0)) // TODO calculate precision based on BFP compression
                    if(!complex_approx_equal<float2,float>(bfw_val,refBfwVal,1.0/100.0)) // relax the tol from 1/256.0 to 1/100.0 for BFW using bfwPowerNormAlg_selector = 1
                    // bfwPowerNormAlg_selector = 0 can pass 1.0/200.0
                    // TODO: check suitable tolerance for bfwPowerNormAlg_selector = 1
                    {
                        NVLOGE_FMT(NVLOG_BFW, AERIAL_CUPHY_EVENT, "Error! Mismatch for weight (antenna {}, layer {}, PRB group {}):  expected={} + i {} vs. gpu={} + i {}",
                                4*iAnt+i, iLayer, iPrbGrp,
                               (float) refBfwVal.x, (float) refBfwVal.y,
                               (float) bfw_val.x, (float) bfw_val.y);
                        mismatch += 1;
                    }
                    if(mismatch > 0)
                    {
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

void bfwEvalDataset::bfwEvalCoefs(bfwStaticApiDataset& statApiDataset, bfwDynApiDataset& dynApiDataset, cudaStream_t cuStrm, float refCheckSnrThd, bool verbose)
{
    constexpr uint8_t bfwCoefBFP = 9;
    uint32_t beta = statApiDataset.bfwStatPrms.beta;
    uint16_t ueGrpIdx_within_cellGrp  = 0;
    bool failedTest = false;
    for(int cellIdx = 0; cellIdx < m_nCells; ++cellIdx)
    {
        for(int ueGrpIdx_within_cell = 0; ueGrpIdx_within_cell < m_nUeGrpsInCell[cellIdx]; ++ueGrpIdx_within_cell)
        {
            // Copy compressed BFW coefficents to CPU
            cuphyBfwUeGrpPrm_t& ueGrpParam = dynApiDataset.bfwUeGrpPrmVec[ueGrpIdx_within_cell];
            uint16_t nPrbGrp   = ueGrpParam.nPrbGrp;
            uint16_t nBfLayers = ueGrpParam.nBfLayers;
            // Determine if beams are packed based on dynamic UE group param; assume uniform across run
            const bool packBeams = (ueGrpParam.beamIdOffset >= 0);
            uint16_t nByteAnt  = (ueGrpParam.nRxAnt*bfwCoefBFP*2+7)/8 + 1 + (packBeams ? 2 : 0);
            uint16_t startPrb = ueGrpParam.startPrb;
            int16_t beamIdOffset = packBeams ? (int16_t)(startPrb/CUPHY_BFW_MIN_PRB_GRP_SIZE + ueGrpParam.beamIdOffset) : -1;

            std::vector<uint8_t> compressed_bfw_coefs(nPrbGrp*nBfLayers*nByteAnt);
            uint8_t* bfw_comp_ptr = dynApiDataset.bfwComppBufVec[ueGrpIdx_within_cellGrp];
            if(dynApiDataset.m_outputBufferMode == OutputBufferMode::Device)
            {
                CUDA_CHECK(cudaMemcpyAsync(compressed_bfw_coefs.data(), dynApiDataset.bfwComppBufVec[ueGrpIdx_within_cellGrp], nPrbGrp*nBfLayers*nByteAnt, cudaMemcpyDeviceToHost, cuStrm));
                bfw_comp_ptr = compressed_bfw_coefs.data();
            }
            CUDA_CHECK(cudaStreamSynchronize(cuStrm));

            if(bfwDecompressCompare(ueGrpIdx_within_cellGrp, beta, nByteAnt, bfw_comp_ptr, beamIdOffset))
            {
                NVLOGC_FMT(NVLOG_BFW, "Test PASS: BFW cell[{:02d}] ueGrp[{:02d}]: decompressed BFW coefficents match", cellIdx, ueGrpIdx_within_cell);
            }else{
                NVLOGW_FMT(NVLOG_BFW, "Test FAIL: BFW cell[{:02d}] ueGrp[{:02d}]: comparing decompressed BFW coefficents don't match.", cellIdx, ueGrpIdx_within_cell);
                failedTest = true;
            }
            ueGrpIdx_within_cellGrp += 1;
        }
    }

    if(failedTest)
    {
        NVLOGF_FMT(NVLOG_BFW, AERIAL_CUPHY_EVENT, "BWC reference check: FAILED! exiting\n");
        // in case of mismatch, NVLOGF_FMT will call exit(1), no need to throw exception
        // throw cuphy::cuphy_exception(CUPHY_STATUS_REF_MISMATCH);
    }
}

//---------------------------------------------------------------------------------------------------
// Dataset holds dynamic PDSCH API parameters/data

// Construct dataset from HDF5 file
pdschDynApiDataset::pdschDynApiDataset(const std::string& inputFileName, uint32_t cfg_max_cells, cudaStream_t cuStrm, cuphyPdschProcMode_t pdsch_proc_mode, cuphyPdschStatPrms_t& stat_params)
{
    // Output buffer allocation will happen based on the number of max cells.
    max_cells = cfg_max_cells;

    cell_grp_dyn_params.nCells             = 0;
    cell_grp_dyn_params.nUeGrps            = 0;
    cell_grp_dyn_params.nUes               = 0;
    cell_grp_dyn_params.nCws               = 0;
    cell_grp_dyn_params.nCsiRsPrms         = 0;
    cell_grp_dyn_params.nPrecodingMatrices = 0;

    max_UEs_per_cell_group = (stat_params.nMaxUesPerCellGroup == 0) ? PDSCH_MAX_UES_PER_CELL_GROUP : stat_params.nMaxUesPerCellGroup;

    // Reserve space up front to avoid the need to update the parent pointers on every cumulative update.
    // If we exceed the reserved space, the various parent pointers will be incorrect.
    CellPrms.reserve(max_cells);
    CellMetrics.reserve(max_cells);
    UeGrpPrms.reserve(PDSCH_MAX_UE_GROUPS_PER_CELL_GROUP);
    UePrms.reserve(max_UEs_per_cell_group);
    CwPrms.reserve(max_UEs_per_cell_group); // Reminder: for now # CWs is the same as # UEs. Max. value is PDSCH_MAX_CWS_PER_CELL_GROUP

    // Large buffer added as cuPHY tools needs a buffer with a power of 2 size.
    // Allocating once with the assumption that there will be at most max_cells cells
    large_buffer_elements = MAX_N_PRBS_SUPPORTED * CUPHY_N_TONES_PER_PRB * OFDM_SYMBOLS_PER_SLOT * MAX_DL_LAYERS;
    large_buffer_bytes    = large_buffer_elements * sizeof(__half2);
    large_buffer          = make_unique_device<__half2>(max_cells * large_buffer_elements);
    // Reset the entire output buffer.
    CUDA_CHECK(cudaMemsetAsync(large_buffer.get(), 0, max_cells * large_buffer_bytes, cuStrm)); // This is done in cuStrm

    crc_input_data_ptr = std::make_unique<uint8_t*[]>(max_cells);
    crc_input_data.resize(max_cells);
    data_in.resize(max_cells);
    tb_crc_in.resize(max_cells);
    output_data.resize(max_cells);
    output_status.resize(1);
    output_tensorPrm.resize(max_cells);
    data_tx_tensor.resize(max_cells);

    PmwPrms.resize(max_UEs_per_cell_group);

    // OK to do the next two actions  here given overprovisioned allocations above.
    pdsch_dyn_params                    = {cuStrm, pdsch_proc_mode, &cell_grp_dyn_params, data_in.data(), tb_crc_in.data(), output_data.data(), output_status.data()};
    pdsch_dyn_params.pDataOut->pTDataTx = output_tensorPrm.data();
    output_status[0] = {cuphyPdschStatusType_t::CUPHY_PDSCH_STATUS_SUCCESS_OR_UNTRACKED_ISSUE, MAX_UINT16, MAX_UINT16};

    // Update dynamic parameters. This method will also be called to stitch together multiple one-cell TVs.
    cumulativeUpdate(inputFileName, cuStrm, pdsch_proc_mode);
}

void pdschDynApiDataset::cumulativeUpdate(const std::string& inputFileName, cudaStream_t cuStrm, cuphyPdschProcMode_t pdsch_proc_mode)
{
    int current_cells        = CellPrms.size();
    int current_UE_groups    = UeGrpPrms.size();
    int current_UEs          = UePrms.size();
    int current_CWs          = CwPrms.size();
    int current_CSIRS_params = CsirsPrms.size();
    int current_PM_W_params  = PmwPrms.size(); // not used

    // Open HDF5 file
    hdf5hpp::hdf5_file    fInput                    = hdf5hpp::hdf5_file::open(inputFileName.c_str());
    hdf5hpp::hdf5_dataset cell_grp_dyn_pars_dataset = fInput.open_dataset("cellGrpDyn_pars");
    int                   num_cell_groups           = cell_grp_dyn_pars_dataset.get_dataspace().get_dimensions()[0];

    if(num_cell_groups != 1)
    {
        throw std::runtime_error("PDSCH: Only a single cell group is supported per pipeline!");
    }
    int                     cell_group_id       = 0;
    cuphy::cuphyHDF5_struct cell_grp_dyn_config = cuphy::get_HDF5_struct_index(cell_grp_dyn_pars_dataset, cell_group_id);

    //Read # cells, UE groups, UEs, and CWs and update the relevant fields in this cell group.
    //FIXME Add precoding too.
    uint16_t num_cells = cell_grp_dyn_config.get_value_as<uint16_t>("nCells");
    cell_grp_dyn_params.nCells += num_cells;
    CellPrms.resize(cell_grp_dyn_params.nCells);
    CellMetrics.resize(cell_grp_dyn_params.nCells);
    cell_grp_dyn_params.pCellPrms = CellPrms.data();
    cell_grp_dyn_params.pCellMetrics = CellMetrics.data();

    if(CellPrms.size() > max_cells)
    {
        throw std::runtime_error("PDSCH: More max. # cells than expected!");
    }

    uint16_t num_ue_groups = cell_grp_dyn_config.get_value_as<uint16_t>("nUeGrps");
    cell_grp_dyn_params.nUeGrps += num_ue_groups;
    UeGrpPrms.resize(current_UE_groups + num_ue_groups);
    pdsch_dmrs_pars.resize(current_UE_groups + num_ue_groups);
    cell_grp_dyn_params.pUeGrpPrms = UeGrpPrms.data();
    NVLOGD_FMT(NVLOG_PDSCH, "pUeGrpPrms {:p}", static_cast<void*>(cell_grp_dyn_params.pUeGrpPrms));

    uint16_t num_ues = cell_grp_dyn_config.get_value_as<uint16_t>("nUes");
    cell_grp_dyn_params.nUes += num_ues;
    UePrms.resize(cell_grp_dyn_params.nUes);
    cell_grp_dyn_params.pUePrms = UePrms.data();

    uint16_t num_cws = cell_grp_dyn_config.get_value_as<uint16_t>("nCws");
    cell_grp_dyn_params.nCws += num_cws;
    CwPrms.resize(cell_grp_dyn_params.nCws);
    cell_grp_dyn_params.pCwPrms = CwPrms.data();

    hdf5hpp::hdf5_dataset csirs_dyn_pars_dataset = fInput.open_dataset("csirs_pars");
    uint16_t              num_csirs              = csirs_dyn_pars_dataset.get_dataspace().get_dimensions()[0];
    if(num_csirs != 0)
    { // Unlike other datasets, it is possible there are no CSI-RS PDUs scheduled for this cell group
        cell_grp_dyn_params.nCsiRsPrms += num_csirs;
        CsirsPrms.resize(cell_grp_dyn_params.nCsiRsPrms);
        cell_grp_dyn_params.pCsiRsPrms = CsirsPrms.data();
        //Populate the new num_csirs CSI-RS parameters. Only fields needed for CSI-RS RE location are filled in. The rest are set to zero.
        _cuphyCsirsRrcDynPrm* csirs_params = cell_grp_dyn_params.pCsiRsPrms + current_CSIRS_params;
        read_pdsch_csirs_pars_from_file(csirs_params, num_csirs, csirs_dyn_pars_dataset);
    }

    cell_grp_dyn_params.pPmwPrms = PmwPrms.data();

    // Check we haven't exceeded reserved capacity
    if((cell_grp_dyn_params.nCells > CellPrms.capacity()) ||
       (cell_grp_dyn_params.nCells > CellMetrics.capacity()) ||
       (cell_grp_dyn_params.nUeGrps > UeGrpPrms.capacity()) ||
       (cell_grp_dyn_params.nUes > UePrms.capacity()) ||
       (cell_grp_dyn_params.nCws > CwPrms.capacity()))
    {
        throw std::runtime_error("PDSCH: Expected  max. values exceeded. Parent pointers will be invalid!");
    }

    // TODO The *v2 are versions of the functions from pdsch_tx.cpp that existin common/cuphy_hdf5.hpp. The former are still used by the cuPHY control plane, so will keep both until they remove them.  They are also used by some other cuPHY examples too.
    // FIXME There may be small difference across them too.
    read_dmrs_pars_from_file(pdsch_dmrs_pars, fInput, current_UE_groups); //FIXME resizing pdsch_dmrs_pars

    // Populate arrays. Includes setting pointer to parent/children structs etc., so it should
    // happen *after* all previous mem. allocations.
    read_cell_dynamic_pars_from_file(&cell_grp_dyn_params.pCellPrms[current_cells], fInput, current_cells, current_CSIRS_params); // No allocations internally

    //TODO remove comment  Allocations (new) of UE group's pUePrmIdxs and pDmrsDynPrm
    read_ue_groups_pars_from_file(&cell_grp_dyn_params.pUeGrpPrms[current_UE_groups], fInput, &cell_grp_dyn_params.pCellPrms[current_cells], pdsch_dmrs_pars, current_UE_groups, current_UEs);

    //TODO remove comment  Allocations (new) of pCwIdxs
    read_ue_pars_from_file(&cell_grp_dyn_params.pUePrms[current_UEs], fInput, &cell_grp_dyn_params.pUeGrpPrms[current_UE_groups], cell_grp_dyn_params, current_UEs);

    //TODO remove comment  No new allocations
    read_cw_pars_from_file(&cell_grp_dyn_params.pCwPrms[current_CWs], fInput, &cell_grp_dyn_params.pUePrms[current_UEs], current_CWs);

    // For dataset updates assuming single cell in each dataset (FIXME make consistent everywhere)
    //Parse input dataset
    hdf5hpp::hdf5_dataset crc_dataset = fInput.open_dataset("InputData");
    //crc_input_data.emplace_back(typed_tensor_from_dataset<CUPHY_R_8U, pinned_alloc>(crc_dataset, cuphy::tensor_flags::align_default, cuStrm));
    crc_input_data[current_cells]     = typed_tensor_from_dataset<CUPHY_R_8U, pinned_alloc>(crc_dataset, cuphy::tensor_flags::align_default, cuStrm);
    crc_input_data_ptr[current_cells] = (uint8_t*)crc_input_data[current_cells].addr();

    data_in[current_cells]   = {&crc_input_data_ptr[current_cells], cuphyPdschDataIn_t::CPU_BUFFER};
    tb_crc_in[current_cells] = {nullptr, cuphyPdschDataIn_t::CPU_BUFFER};

    int num_REs                   = cuphy::get_HDF5_dataset_info(fInput.open_dataset("Xtf")).layout().dimensions()[0];
    data_tx_tensor[current_cells] = tensor_device(large_buffer.get() + current_cells * large_buffer_elements, CUPHY_C_16F, num_REs, OFDM_SYMBOLS_PER_SLOT, MAX_DL_PORTS, cuphy::tensor_flags::align_tight);
    pdsch_dyn_params.pDataOut->pTDataTx[current_cells].desc  = data_tx_tensor[current_cells].desc().handle();

    pdsch_dyn_params.pDataOut->pTDataTx[current_cells].desc  = data_tx_tensor[current_cells].desc().handle();
    pdsch_dyn_params.pDataOut->pTDataTx[current_cells].pAddr = data_tx_tensor[current_cells].addr();
    //NVLOGC_FMT(NVLOG_PDSCH, "addr() {:p}", (void*)data_tx_tensor[current_cells].addr());

    CUDA_CHECK(cudaStreamSynchronize(cuStrm)); // ensure crc_input_data is ready FIXME?
}

void pdschDynApiDataset::print()
{
    print_pdsch_dynamic_cell_group(&cell_grp_dyn_params);
}

void pdschDynApiDataset::resetOutputTensors(cudaStream_t cuStrm)
{
    // Reset the entire output buffer.
    CUDA_CHECK(cudaMemsetAsync(large_buffer.get(), 0, max_cells * large_buffer_bytes, cuStrm));
}

// default constructor
pdschDynApiDataset::pdschDynApiDataset() :
    CellPrms{},
    CellMetrics{},
    UeGrpPrms{},
    UePrms{},
    CwPrms{},
    pdsch_dmrs_pars{},
    cell_grp_dyn_params{},
    pdsch_dyn_params{},
    data_in{},
    output_data{},
    output_tensorPrm{},
    max_cells(1)
{}

// reset pointers after a copy or move
void pdschDynApiDataset::ResetPointers()
{
    NVLOGC_FMT(NVLOG_PDSCH, "PDSCH dynamic dataset: reset pointers");
    cell_grp_dyn_params.pCellPrms  = CellPrms.data();
    cell_grp_dyn_params.pCellMetrics  = CellMetrics.data();
    cell_grp_dyn_params.pUeGrpPrms = UeGrpPrms.data();
    cell_grp_dyn_params.pUePrms    = UePrms.data();
    cell_grp_dyn_params.pCwPrms    = CwPrms.data();
}

// move operator
pdschDynApiDataset& pdschDynApiDataset::operator=(pdschDynApiDataset&& pdschdynApiDataset)
{
    NVLOGC_FMT(NVLOG_PDSCH, "PDSCH dynamic dataset: move");
    CellPrms            = std::move(pdschdynApiDataset.CellPrms);
    CellMetrics   = std::move(pdschdynApiDataset.CellMetrics);
    UeGrpPrms           = std::move(pdschdynApiDataset.UeGrpPrms);
    UePrms              = std::move(pdschdynApiDataset.UePrms);
    CwPrms              = std::move(pdschdynApiDataset.CwPrms);
    pdsch_dmrs_pars     = std::move(pdschdynApiDataset.pdsch_dmrs_pars);
    cell_grp_dyn_params = std::move(pdschdynApiDataset.cell_grp_dyn_params);
    pdsch_dyn_params    = std::move(pdschdynApiDataset.pdsch_dyn_params);
    output_data         = std::move(pdschdynApiDataset.output_data);
    output_tensorPrm    = std::move(pdschdynApiDataset.output_tensorPrm);
    data_in             = std::move(pdschdynApiDataset.data_in);
    ResetPointers();
    return *this;
}

// copy constructor
pdschDynApiDataset::pdschDynApiDataset(const pdschDynApiDataset& pdschdynApiDataset) :
    CellPrms(pdschdynApiDataset.CellPrms),
    CellMetrics(pdschdynApiDataset.CellMetrics),
    UeGrpPrms(pdschdynApiDataset.UeGrpPrms),
    UePrms(pdschdynApiDataset.UePrms),
    CwPrms(pdschdynApiDataset.CwPrms),
    pdsch_dmrs_pars(pdschdynApiDataset.pdsch_dmrs_pars),
    cell_grp_dyn_params(pdschdynApiDataset.cell_grp_dyn_params),
    pdsch_dyn_params(pdschdynApiDataset.pdsch_dyn_params),
    data_in(pdschdynApiDataset.data_in),
    output_data(pdschdynApiDataset.output_data),
    output_tensorPrm(pdschdynApiDataset.output_tensorPrm)
{
    NVLOGC_FMT(NVLOG_PDSCH, "PDSCH dynamic dataset: copy");
    // synch default stream, used to copy
    CUDA_CHECK(cudaStreamSynchronize(0));

    // update pointers
    ResetPointers();
}

//----------------------------------------------------------------------------------------------------------
//  Dataset holds static api parameters/data

// construct from h5 file
pdschStaticApiDataset::pdschStaticApiDataset(const std::string& inputFileName, std::string outputFileName, bool ref_check, bool identical_ldpc_configs, int stream_priority, uint32_t max_CBs_per_TB, uint32_t max_UEs_per_cell_group, uint32_t max_PRBs)
{
    pdschStatPrms.nCells               = 0;
    pdschStatPrms.read_TB_CRC          = false;
    pdschStatPrms.full_slot_processing = true;

    pdschStatPrms.nMaxUesPerCellGroup = max_UEs_per_cell_group;
    pdschStatPrms.nMaxCBsPerTB        = max_CBs_per_TB;
    pdschStatPrms.nMaxPrb             = max_PRBs;

    pdschStatPrms.stream_priority = stream_priority;
    compute_max_values            = false;

    pdschTracker.pMemoryFootprint = nullptr;
    pdschStatPrms.pOutInfo        = &pdschTracker;
    pdschStatPrms.enableBatchedMemcpy = 0; // update as needed

    cumulativeUpdate(inputFileName, outputFileName, ref_check, identical_ldpc_configs);
}

void pdschStaticApiDataset::cumulativeUpdate(const std::string& inputFileName, std::string outputFileName, bool ref_check, bool identical_ldpc_configs)
{
    int current_static_cells = cellStatPrm.size();

    hdf5hpp::hdf5_file    fInput              = hdf5hpp::hdf5_file::open(inputFileName.c_str());
    hdf5hpp::hdf5_dataset cell_static_dataset = fInput.open_dataset("cellStat_pars");
    int                   num_cells           = cell_static_dataset.get_dataspace().get_dimensions()[0];

    cellStatPrm.resize(current_static_cells + num_cells);
    uint16_t max_PRBs_from_TV = 0;
    read_cell_static_pars_from_file(cellStatPrm.data() + current_static_cells, cell_static_dataset, num_cells, current_static_cells, max_PRBs_from_TV);
    if((pdschStatPrms.nMaxPrb != 0) && (max_PRBs_from_TV > pdschStatPrms.nMaxPrb))
    {
        throw std::runtime_error("nPRBs in TV > nMaxPrb specified in yaml file!");
    }

    CfgFileName.push_back(inputFileName);
    int j = CfgFileName.size() - 1;

    dbgPrm.resize(current_static_cells + num_cells);
    for(int i = 0; i < num_cells; i++)
    {
        dbgPrm[current_static_cells + i].pCfgFileName            = CfgFileName[j].empty() ? nullptr : CfgFileName[j].c_str();
        dbgPrm[current_static_cells + i].refCheck                = ref_check;
        dbgPrm[current_static_cells + i].cfgIdenticalLdpcEncCfgs = identical_ldpc_configs;
    }

    pdschStatPrms.nCells += num_cells;
    pdschStatPrms.pCellStatPrms = cellStatPrm.data();
    pdschStatPrms.pDbg          = dbgPrm.data();

    // Max. parameters. Set to 0 if you want to use default time constants
    pdschStatPrms.nMaxCellsPerSlot = pdschStatPrms.nCells;

    if(compute_max_values)
    {
        // Currently number of UEs is the same as number of CWs. Update when 2 CW per UE are supported.
        hdf5hpp::hdf5_dataset      cell_grp_dataset    = fInput.open_dataset("cellGrpDyn_pars");
        hdf5hpp::hdf5_dataset_elem cell_grp_dyn_config = cell_grp_dataset[0];
        int16_t                    new_UEs             = cell_grp_dyn_config["nUes"].as<uint16_t>();
        pdschStatPrms.nMaxUesPerCellGroup += new_UEs;

        // Find max. number of CBs per TB from the first dimension of tb*_cbs dataset (or tb*_codecbs).
        // FIXME Is it realistic to expect this value to be specified up front?

        hdf5hpp::hdf5_dataset_elem  cell_dyn_config = fInput.open_dataset("cellDyn_pars")[0];
        uint8_t cell_testing_mode = 0;
        try {
            cell_testing_mode                   = cell_dyn_config["testModel"].as<uint8_t>();
        } catch(...) {
            cell_testing_mode = 0;
        }
        for(int UE_idx = 0; UE_idx < new_UEs; UE_idx++)
        {
            // If the cell this UE belongs to is in testing mode, then use TB size (in bits) / 25344, rounded up, to determine number of CBs.
            // (Another option is to set nMaxCBsPerTB to 0.)
            std::string           CBs_dataset_name = "tb" + std::to_string(UE_idx) + ((cell_testing_mode == 0) ? "_cbs" : "_inputdata");
            hdf5hpp::hdf5_dataset CBs_dataset      = fInput.open_dataset(CBs_dataset_name.c_str());
            uint16_t              num_CBs_for_TB   = (cell_testing_mode == 0) ? CBs_dataset.get_dataspace().get_dimensions()[0] : \
                                                     (div_round_up<int>(CBs_dataset.get_dataspace().get_dimensions()[1], MAX_ENCODED_CODE_BLOCK_BIT_SIZE));
            pdschStatPrms.nMaxCBsPerTB             = std::max(pdschStatPrms.nMaxCBsPerTB, num_CBs_for_TB);
        }
    }
}

void pdschStaticApiDataset::print()
{
    print_pdsch_static(&pdschStatPrms);
}

// default constructor
pdschStaticApiDataset::pdschStaticApiDataset() :
    pdschStatPrms{},
    dbgPrm{},
    cellStatPrm{},
    CfgFileName{},
    compute_max_values(false)
{}

// Reset pointers after a move or copy
void pdschStaticApiDataset::ResetPointers()
{
    NVLOGC_FMT(NVLOG_PDSCH, "PDSCH static dataset: resetPointers");
#if 0
    pdschStatPrms.pCellStatPrms = &cellStatPrm;
    pdschStatPrms.pDbg          = &dbgPrm;
    dbgPrm.pCfgFileName = CfgFileName.empty() ? nullptr : CfgFileName.c_str();
#else
    pdschStatPrms.pCellStatPrms = cellStatPrm.data();
    pdschStatPrms.pDbg          = dbgPrm.data();
    //dbgPrm.pCfgFileName = CfgFileName.empty() ? nullptr : CfgFileName.c_str(); //FIXME Why??
#endif
}

// move operator
pdschStaticApiDataset& pdschStaticApiDataset::operator=(pdschStaticApiDataset&& pdschstaticApiDataset)
{
    NVLOGC_FMT(NVLOG_PDSCH, "PDSCH static dataset: move");
    pdschStatPrms      = std::move(pdschstaticApiDataset.pdschStatPrms);
    dbgPrm             = std::move(pdschstaticApiDataset.dbgPrm);
    cellStatPrm        = std::move(pdschstaticApiDataset.cellStatPrm);
    CfgFileName        = std::move(pdschstaticApiDataset.CfgFileName);
    compute_max_values = pdschstaticApiDataset.compute_max_values;

    ResetPointers();
    return *this;
}

// copy constructor
pdschStaticApiDataset::pdschStaticApiDataset(const pdschStaticApiDataset& pdschstaticApiDataset) :
    pdschStatPrms(pdschstaticApiDataset.pdschStatPrms),
    dbgPrm(pdschstaticApiDataset.dbgPrm),
    cellStatPrm(pdschstaticApiDataset.cellStatPrm),
    CfgFileName(pdschstaticApiDataset.CfgFileName)
{
    NVLOGC_FMT(NVLOG_PDSCH, "PDSCH static dataset: copy");
    // synch default stream, used to copy
    CUDA_CHECK(cudaStreamSynchronize(0));

    // update pointers
    ResetPointers();
}

// PDCCH datasets

// construct from h5 file
pdcchStaticApiDataset::pdcchStaticApiDataset(int cfg_max_cells_per_slot)
{
    pdcchStatPrms.nMaxCellsPerSlot = cfg_max_cells_per_slot; // Should be specified. Currently no fallback if 0.
    pdcchTracker.pMemoryFootprint  = nullptr;
    pdcchStatPrms.pOutInfo         = &pdcchTracker;
}

void pdcchStaticApiDataset::print()
{
}

//---------------------------------------------------------------------------------------------------
// Dataset holds dynamic PDCCH API parameters/data

// Construct dataset from HDF5 file
pdcchDynApiDataset::pdcchDynApiDataset(const std::string& inputFileName, uint32_t cfg_max_cells, cudaStream_t cuStrm, uint64_t procModeBmsk)
{
    // Output buffer allocation will happen based on the number of max cells.
    max_cells = cfg_max_cells;

    pdcch_dyn_params.nCells    = 0;
    pdcch_dyn_params.nCoresets = 0;
    pdcch_dyn_params.nDci      = 0;
    pdcch_dyn_params.nPrecodingMatrices = 0;

    // Reserve space up front to avoid the need to update the parent pointers on every cumulative update.
    // If we exceed the reserved space, the various parent pointers will be incorrect.
    coreset_params.reserve(cfg_max_cells * CUPHY_PDCCH_N_MAX_CORESETS_PER_CELL);
    dci_params.reserve(cfg_max_cells * CUPHY_PDCCH_N_MAX_CORESETS_PER_CELL * CUPHY_PDCCH_MAX_DCIS_PER_CORESET);

    // Large buffer added as cuPHY tools needs a buffer with a power of 2 size.
    // Allocating once with the assumption that there will be at most max_cells cells
    large_buffer_elements = MAX_N_PRBS_SUPPORTED * CUPHY_N_TONES_PER_PRB * OFDM_SYMBOLS_PER_SLOT * MAX_DL_LAYERS;
    large_buffer_bytes    = large_buffer_elements * sizeof(__half2);
    large_buffer          = make_unique_device<__half2>(max_cells * large_buffer_elements);
    // Reset the entire output buffer.
    CUDA_CHECK(cudaMemsetAsync(large_buffer.get(), 0, max_cells * large_buffer_bytes, cuStrm)); // This is done in cuStrm

    output_data.resize(max_cells);
    output_tensorPrm.resize(max_cells);
    data_tx_tensor.resize(max_cells);

    data_in.resize(1);

    int input_dims[2] = {CUPHY_PDCCH_MAX_DCI_PAYLOAD_BYTES, (int) dci_params.capacity()};
    input_data.emplace_back(cuphy::tensor_layout(2, input_dims, nullptr));
    data_in[0] = {input_data[0].addr(), cuphyPdcchDataIn_t::CPU_BUFFER};

    // OK to do the next two actions  here given overprovisioned allocations above.
#if 0
    pdcch_dyn_params = {cuStrm, 0 /* pdcch_proc_mode*/, 0 /* updated later */,
                        dci_params.data(), 0 /* updated later */, coreset_params.data(),
                        1 /*nCells*/, data_in.data(), nullptr};
#endif
    pdcch_dyn_params.procModeBmsk = procModeBmsk;
    pdcch_dyn_params.pDataIn      = data_in.data();
    pdcch_dyn_params.pDataOut     = output_data.data();
    pdcch_dyn_params.pDataOut->pTDataTx = output_tensorPrm.data();
    pdcch_dyn_params.cuStream     = cuStrm;

    // Update dynamic parameters. This method will also be called to stitch together multiple one-cell TVs.
    cumulativeUpdate(inputFileName, cuStrm);
}

void pdcchDynApiDataset::cumulativeUpdate(const std::string& inputFileName, cudaStream_t cuStrm) {

    int current_cells     = pdcch_dyn_params.nCells;
    int current_coresets  = pdcch_dyn_params.nCoresets;
    int current_dci       = pdcch_dyn_params.nDci;

    // Open HDF5 file
    hdf5hpp::hdf5_file fInput = hdf5hpp::hdf5_file::open(inputFileName.c_str());
    CfgFileName.push_back(inputFileName);

    // Every TV corresponds to a separate cell
    pdcch_dyn_params.nCells += 1;
    if (pdcch_dyn_params.nCells > max_cells) {
        std::string error_msg = "Current PDCCH dyn. cells are " + std::to_string(pdcch_dyn_params.nCells) + " but max. cells was " + std::to_string(max_cells) + "\n";
        throw std::runtime_error(error_msg.c_str());
    }

    // Figure out nCoresets of this TV. Resize coreset_params and read values. Update nCoresets
    hdf5hpp::hdf5_dataset dset = fInput.open_dataset("PdcchParams");
    int num_coresets = dset.get_dataspace().get_dimensions()[0];
    pdcch_dyn_params.nCoresets += num_coresets;
    coreset_params.resize(pdcch_dyn_params.nCoresets);
    pdcch_precoding_matrix.resize(max_cells * CUPHY_PDCCH_N_MAX_CORESETS_PER_CELL * CUPHY_PDCCH_MAX_DCIS_PER_CORESET);
    cuphy::read_pdcch_coreset_dyn_params_from_file_v2(coreset_params, fInput, current_coresets, current_dci, current_cells);

    bool use_new_dset = fInput.is_valid_dataset("DciPayload_coreset_0_dci_0");

    int cumulative_dci = current_dci;
    for (int i = 0; i < num_coresets; i++) {
        //Figure out nDCIs of this TV; Resize dci_params; read them; update nDci

        int num_dci = coreset_params[current_coresets + i].nDci;
        pdcch_dyn_params.nDci += num_dci;
        dci_params.resize(pdcch_dyn_params.nDci);
        cuphy::read_dci_pdcch_params(dci_params.data() + cumulative_dci, fInput, num_dci, pdcch_precoding_matrix, pdcch_dyn_params.nPrecodingMatrices, i);

        // Read num_dci DCI payloads for coreset i
        for (int dci_idx = 0; dci_idx < num_dci; dci_idx++) {

            std::string   dataset_name = (use_new_dset) ? ("DciPayload_coreset_" + std::to_string(i) + "_dci_" + std::to_string(dci_idx)) : ("DciPayload" + std::to_string(dci_idx + 1));
            uint32_t      dci_bits     = dci_params[cumulative_dci + dci_idx].Npayload;
            uint8_t byte_id      = 0;
            int     element_size = 8;
            uint32_t dci_bytes = div_round_up<uint32_t>(dci_bits, element_size);

            if(use_new_dset)
            {
               cuphy::typed_tensor<CUPHY_R_8U, cuphy::pinned_alloc> payload_dci_idx = cuphy::typed_tensor_from_dataset<CUPHY_R_8U, cuphy::pinned_alloc>((fInput).open_dataset(dataset_name.c_str()));
                // Added extra check to ensure the payload size in bytes matches the expected dci_bytes.
                if (payload_dci_idx.layout().dimensions()[0] != dci_bytes) {
                    std::string error_msg = "Coreset " + std::to_string(i) + " with DCI " + std::to_string(dci_idx) + " payload size mismatch: " +\
                                            "expected " + std::to_string(dci_bytes) + "B but got " + std::to_string(payload_dci_idx.layout().dimensions()[0]) + "B instead.\n";
                    throw std::runtime_error(error_msg.c_str());
                }

                for(int element_start = 0; element_start < dci_bytes; element_start++)
                {
                    //Note: if element 0 is 0x0c when printed means input was:  0 0 0 0 1 1 0 0
                    input_data[0](element_start, cumulative_dci + dci_idx) = payload_dci_idx(element_start);
                }
            }
            else
            {
                // TODO: Will eventually remove once all new PDCCH TVs are generated from Matlab
                // Pack properly to uint8_t elements. FIXME We could change the Matlab TV to avoid this work.
                cuphy::typed_tensor<CUPHY_R_32U, cuphy::pinned_alloc> payload_dci_idx = cuphy::typed_tensor_from_dataset<CUPHY_R_32U, cuphy::pinned_alloc>((fInput).open_dataset(dataset_name.c_str()));
                // Added extra check to ensure the payload size in bytes matches the expected dci_bytes.
                if (payload_dci_idx.layout().dimensions()[0] != dci_bits) {
                    std::string error_msg = "Coreset " + std::to_string(i) + " with DCI " + std::to_string(dci_idx) + " payload size mismatch: " +\
                                            "expected " + std::to_string(dci_bytes) + "bits but got " + std::to_string(payload_dci_idx.layout().dimensions()[0]) + "bits instead.\n";
                    throw std::runtime_error(error_msg.c_str());
                }

                for(int element_start = 0; element_start < dci_bits; element_start += element_size)
                {
                    uint8_t val = 0;
                    for(int offset = 0; offset < element_size; offset++)
                    {
                        if(element_start + offset < dci_bits)
                        {
                            uint8_t bit = payload_dci_idx(element_start + offset);
                            val |= (bit << (element_size - 1 - offset));
                        }
                    }
                    //Note: if element 0 is 0x0c when printed means input from (0, 0) to (0, 7) was:  0 0 0 0 1 1 0 0
                    //NVLOGC_FMT(NVLOG_PDCCH, "element_start {}, val {:#x}", element_start, val);

                    input_data[0](element_start / element_size, cumulative_dci + dci_idx) = val;
                }
            }
            // Populate bytes beyond payload with 0xff to ensure code does not make any invalid assumptions about the payload's
            // contents beyond its bytes.
            memset(&input_data[0](dci_bytes, cumulative_dci + dci_idx), 0xff, CUPHY_PDCCH_MAX_DCI_PAYLOAD_BYTES - dci_bytes);
        }
        cumulative_dci += num_dci;
     }

     //Update p pointers
     pdcch_dyn_params.pDciPrms       = dci_params.data();
     pdcch_dyn_params.pCoresetDynPrm = coreset_params.data();
     pdcch_dyn_params.pPmwParams     = pdcch_precoding_matrix.data();

     //Update pDataOut
     int num_REs = use_new_dset ?
                   cuphy::get_HDF5_dataset_info(fInput.open_dataset("X_tf_fp16")).layout().dimensions()[0] :
                   cuphy::get_HDF5_dataset_info(fInput.open_dataset("Xtf")).layout().dimensions()[0];
     int ports = use_new_dset ?
                 cuphy::get_HDF5_dataset_info(fInput.open_dataset("X_tf_fp16")).layout().dimensions()[2] :
                 cuphy::get_HDF5_dataset_info(fInput.open_dataset("Xtf")).layout().dimensions()[2];

     data_tx_tensor[current_cells] = tensor_device(large_buffer.get() + current_cells * large_buffer_elements, CUPHY_C_16F,
                                                   num_REs, OFDM_SYMBOLS_PER_SLOT, ports, cuphy::tensor_flags::align_tight);

     pdcch_dyn_params.pDataOut->pTDataTx[current_cells].desc  = data_tx_tensor[current_cells].desc().handle();
     pdcch_dyn_params.pDataOut->pTDataTx[current_cells].pAddr = data_tx_tensor[current_cells].addr();

     //CUDA_CHECK(cudaStreamSynchronize(cuStrm));
}

void pdcchDynApiDataset::printPayload() {
   //print payload
   for (int dci = 0; dci < dci_params.size(); dci++) {
       std::stringstream dci_payload_strm;
       dci_payload_strm << "DCI " << std::to_string(dci) << " payload: ";
       for (int i = 0; i < CUPHY_PDCCH_MAX_DCI_PAYLOAD_BYTES; i++) {
           dci_payload_strm << std::hex << input_data[0](i, dci) << " ";
       }
       NVLOGC_FMT(NVLOG_PDCCH, "{}", dci_payload_strm.str().c_str());
   }
}

/* Code to change the relationship of coreset i (with i in [0, num_coresetss)  with its dciStartIdx field.
   By default when parsing TVs, coreset[i].dciStartIdx = (i == 0) ? 0 : coreset[i-1].dciStartIdx + coreset[i-1].num_dl_dci.
   This function reverses this, and updates dciStartIdx and reorganizes pDciParams and input payload,
   as a first step to enable testing for more general scenarios.
*/
void pdcchDynApiDataset::revDciOrder() {

   // It's safe to call this function in the testbench, after all cumulative updates have been called
   // and before setup is called on a pipeline. You can call printPayload() before/after to see the effect this function has on the payload.

   int num_coresets = pdcch_dyn_params.nCoresets;

   std::vector<int> old_dci_start_idx(num_coresets);
   std::vector<cuphyPdcchDciPrm_t> backup_dci_params(dci_params.size());
   std::vector<cuphy::typed_tensor<CUPHY_R_8U, cuphy::pinned_alloc>>  temp_input_data;

   int input_dims[2] = {CUPHY_PDCCH_MAX_DCI_PAYLOAD_BYTES, (int) dci_params.capacity()};
   temp_input_data.emplace_back(cuphy::tensor_layout(2, input_dims, nullptr));

   for (int i = 0; i < num_coresets; i++) {
       old_dci_start_idx[i] = coreset_params[i].dciStartIdx;
       // Set-up a reverse mapping *in place*. NB reverse is just one option, could do any valid shuffle for more variation.
       coreset_params[i].dciStartIdx = ((i == 0) ? pdcch_dyn_params.nDci : coreset_params[i-1].dciStartIdx) - coreset_params[i].nDci;

       // Reorganize dci_params and input payload. Change is not doen in place as there can be overlap.
       std::copy(dci_params.data() + old_dci_start_idx[i],
                 dci_params.data() + old_dci_start_idx[i] + coreset_params[i].nDci,
                 backup_dci_params.data() + coreset_params[i].dciStartIdx);

       std::copy(&input_data[0](0, old_dci_start_idx[i]),
                 &input_data[0](0, old_dci_start_idx[i] + coreset_params[i].nDci),
                 &temp_input_data[0](0, coreset_params[i].dciStartIdx));
   }

   // Copy back to original DCI params and payload structures. No size change.
   std::copy(&temp_input_data[0](0, 0),
             &temp_input_data[0](0, dci_params.size()),
             &input_data[0](0, 0));

   std::copy(backup_dci_params.data(),  backup_dci_params.data() + dci_params.size(), dci_params.data());

}


int pdcchDynApiDataset::refCheck(bool verbose) {
    int total_err_cnt = 0;
    //NVLOGC_FMT(NVLOG_PDCCH, "ref check for {} buffers", pdcch_dyn_params.nCells);
    using tensor_pinned_C_16F          = typed_tensor<CUPHY_C_16F, pinned_alloc>;
    for (int output_buffer = 0; output_buffer < pdcch_dyn_params.nCells; output_buffer++) {
        //NVLOGC_FMT(NVLOG_PDCCH, "buffer {} with name {}", output_buffer, CfgFileName[output_buffer].c_str());
        hdf5hpp::hdf5_file fInput = hdf5hpp::hdf5_file::open(CfgFileName[output_buffer].c_str());
        tensor_pinned_C_16F pdcch_tx_ref_output = typed_tensor_from_dataset<CUPHY_C_16F, pinned_alloc>(fInput.open_dataset("X_tf_fp16"));

        typed_tensor<CUPHY_C_16F, pinned_alloc> h_pdcch_out_tensor(pdcch_tx_ref_output.layout());
        h_pdcch_out_tensor.convert(data_tx_tensor[output_buffer], pdcch_dyn_params.cuStream);
        CUDA_CHECK(cudaStreamSynchronize(pdcch_dyn_params.cuStream));
        NVLOGD_FMT(NVLOG_PDCCH, "PDCCH dims {} {} {}",
                h_pdcch_out_tensor.layout().dimensions()[0],
                h_pdcch_out_tensor.layout().dimensions()[1],
                h_pdcch_out_tensor.layout().dimensions()[2]);

        int checked_symbols = 0, gpu_mismatch = 0;
        int dmrs_err_cnt = 0, qam_err_cnt = 0;
        for(int layer_id = 0; layer_id <  h_pdcch_out_tensor.layout().dimensions()[2]; layer_id++)
        {
            for(int symbol_id = 0; symbol_id < OFDM_SYMBOLS_PER_SLOT; symbol_id++) //  h_pdcch_out_tensor.layout().dimensions()[1] compare with
            {
                for(int freq_idx = 0; freq_idx <  h_pdcch_out_tensor.layout().dimensions()[0]; freq_idx++)
                {
                    __half2 gpu_symbol = h_pdcch_out_tensor(freq_idx, symbol_id, layer_id);
                    __half2 ref_symbol;
                    /* The reference HDF5 dataset should contain all the number of PRBs for this BWP. Any unallocated PRB should be empty. */
                    ref_symbol.x = (half)pdcch_tx_ref_output(freq_idx, symbol_id, layer_id).x;
                    ref_symbol.y = (half)pdcch_tx_ref_output(freq_idx, symbol_id, layer_id).y;
                    checked_symbols += 1;
                    if(!complex_approx_equal<__half2, __half>(gpu_symbol, ref_symbol, 0.0001f))
                    {
                        NVLOGD_FMT(NVLOG_PDCCH, "Error! Mismatch for symbol (freq_bin {}, symbol {}, layer {}):  expected={} + i {} vs. gpu={} + i {}",
                                freq_idx, symbol_id, layer_id,
                               (float) ref_symbol.x, (float) ref_symbol.y,
                               (float) gpu_symbol.x, (float) gpu_symbol.y);
                        gpu_mismatch += 1;
                        if ((freq_idx & 0x3) == 1) {
                            dmrs_err_cnt += 1;
                        } else {
                            qam_err_cnt += 1;
                        }
                    }
                 }
            }
       }
       total_err_cnt += gpu_mismatch;

#if 0
       //TODO update pipeline check script if I use this
       NVLOGC_FMT(NVLOG_PDCCH, "PDCCH: Found {} mismatched symbols out of {} in {{{}, {}, {}}} output tensor; compared with  <X_tf_fp16> dataset from {}",
                  gpu_mismatch, checked_symbols, h_pdcch_out_tensor.layout().dimensions()[1], h_pdcch_out_tensor.layout().dimensions()[2],
                  CfgFileName[output_buffer].c_str());
       NVLOGD_FMT(NVLOG_PDCCH, "Found {} mismatched symbols ({} dmrs {} qam) out of {}", gpu_mismatch, dmrs_err_cnt, qam_err_cnt, checked_symbols);
       NVLOGC_FMT(NVLOG_PDCCH, "Found {} mismatched symbols out of {}", gpu_mismatch, checked_symbols);
       NVLOGC_FMT(NVLOG_PDCCH, "in {{{}, {}, {}}} output tensor; ", h_pdcch_out_tensor.layout().dimensions()[0],
              h_pdcch_out_tensor.layout().dimensions()[1], h_pdcch_out_tensor.layout().dimensions()[2]);
       NVLOGC_FMT(NVLOG_PDCCH, " compared with <X_tf_fp16> dataset from <{}>", CfgFileName[output_buffer].c_str());
#else
       //Same output as orig. phase-0 example
       if (verbose && (gpu_mismatch == 0)) {
           NVLOGC_FMT(NVLOG_PDCCH, "====> TV {}: Test PASS",  CfgFileName[output_buffer].c_str());
       } else if (gpu_mismatch != 0) {
           NVLOGC_FMT(NVLOG_PDCCH, "====> TV {}: Test FAIL. Found {} mismatched symbols: {} dmrs {} qam", CfgFileName[output_buffer].c_str(), gpu_mismatch, dmrs_err_cnt, qam_err_cnt);
       }
#endif
    }
    return total_err_cnt;
}

//----------------------------------------------------------------------------------------------------------
//  Dataset holds PRACH api paramaters/data

// construct from h5 file
PrachApiDataset::PrachApiDataset(const std::string& inputFileName, cudaStream_t cuStrm, uint64_t procModeBmsk, bool refCheck, std::string outputFileName)
                    : procModeBmsk_(procModeBmsk), enable_ref_check(refCheck)
{
    prachTracker.pMemoryFootprint = nullptr;
    prachStatPrms.pOutInfo        = &prachTracker;
    prachStatPrms.enableUlRxBf    = 0; //TODO
    cumulativeUpdate(inputFileName, outputFileName, cuStrm);
}

void PrachApiDataset::cumulativeUpdate(const std::string& inputFileName, std::string outputFileName, cudaStream_t cuStrm)
{
    prach_file = hdf5hpp::hdf5_file::open(inputFileName.c_str());

    int numOccaInCell = 0;
    int nDynParams = prachOccaDynPrms.size();
    try
    {
        hdf5hpp::hdf5_dataset nPrach_dataset = prach_file.open_dataset("nPrach");
        nPrach_dataset.read(&numOccaInCell);
        prachCellStatPrms.resize(nCells + 1);
        prachOccaStatPrms.resize(nOccasaions + numOccaInCell);
        prachOccaDynPrms.resize(nDynParams + numOccaInCell);
        dataRxTensor.resize(nOccasaions + numOccaInCell);
        dataRxTensorPrm.resize(nOccasaions + numOccaInCell);

        cuphy::cuphyHDF5_struct prachCellParamsHdf5 = cuphy::get_HDF5_struct(prach_file, "PrachCellStatPrms");

        prachCellStatPrms[nCells].configurationIndex = prachCellParamsHdf5.get_value_as<uint8_t>("configurationIndex");
        prachCellStatPrms[nCells].restrictedSet = prachCellParamsHdf5.get_value_as<uint8_t>("restrictedSet");
        prachCellStatPrms[nCells].FR = prachCellParamsHdf5.get_value_as<uint8_t>("FR");
        prachCellStatPrms[nCells].duplex = prachCellParamsHdf5.get_value_as<uint8_t>("duplex");
        prachCellStatPrms[nCells].mu = prachCellParamsHdf5.get_value_as<uint8_t>("mu");
        prachCellStatPrms[nCells].N_ant = prachCellParamsHdf5.get_value_as<uint32_t>("N_ant");

        prachCellStatPrms[nCells].occaStartIdx = nOccasaions;
        prachCellStatPrms[nCells].nFdmOccasions = numOccaInCell;

        for(int i = 0; i < numOccaInCell; ++i)
        {
            std::string dataSetName = "prachCuphyParams_" + std::to_string(i);
            cuphy::cuphyHDF5_struct prachParamsHdf5 = cuphy::get_HDF5_struct(prach_file, dataSetName.c_str());
            prachOccaStatPrms[nOccasaions + i].prachRootSequenceIndex = prachParamsHdf5.get_value_as<uint16_t>("prachRootSequenceIndex");
            prachOccaStatPrms[nOccasaions + i].prachZeroCorrConf = prachParamsHdf5.get_value_as<uint8_t>("prachZeroCorrConf");
            prachOccaStatPrms[nOccasaions + i].cellPrmStatIdx = nCells;

            prachOccaDynPrms[nDynParams + i].occaPrmStatIdx = nOccasaions + i;
            prachOccaDynPrms[nDynParams + i].occaPrmDynIdx = nOccasaions + i;
            prachOccaDynPrms[nDynParams + i].force_thr0 = prachParamsHdf5.get_value_as<float>("force_thr0");
            prachOccaDynPrms[nDynParams + i].nUplinkStreams = 1; //TODO

            dataSetName = "y_uv_rx_" + std::to_string(i);

            dataRxTensor[nOccasaions + i] = tensor_from_dataset(prach_file.open_dataset(dataSetName.c_str()), CUPHY_C_16F, cuphy::tensor_flags::align_tight, cuStrm);
            dataRxTensorPrm[nOccasaions + i].desc = dataRxTensor[nOccasaions + i].desc().handle();
            dataRxTensorPrm[nOccasaions + i].pAddr = dataRxTensor[nOccasaions + i].addr();
        }

        if(enable_ref_check)
        {
            readReferenceValues(prach_file, numOccaInCell);
        }
            ++nCells;
    }
    catch (hdf5hpp::hdf5_exception e)
    {
        NVLOGW_FMT(NVLOG_PRACH, "Caught HDF5 exception");
        // For the sake of testing more complex configurations, if a file fails to load,
        //    add an empty static occasion config and no dynamic occasion
        numOccaInCell = 1;
        prachOccaStatPrms.resize(nOccasaions + numOccaInCell);
        dataRxTensor.resize(nOccasaions + numOccaInCell);
        dataRxTensorPrm.resize(nOccasaions + numOccaInCell);
        for(int i = 0; i < numOccaInCell; ++i)
        {
            prachOccaStatPrms[nOccasaions + i].prachRootSequenceIndex = 0;
            prachOccaStatPrms[nOccasaions + i].prachZeroCorrConf      = 0;
            prachOccaStatPrms[nOccasaions + i].cellPrmStatIdx         = 0;
        }

    }
    statDbgPrm.enableApiLogging = !outputFileName.empty();
    statDbgPrm.pOutFileName     = nullptr;
    if(!outputFileName.empty())
    {
        bOutputFileName         = outputFileName;
        statDbgPrm.pOutFileName = bOutputFileName.c_str();
    }


    nOccasaions += numOccaInCell;
}

void PrachApiDataset::finalize(cudaStream_t cuStrm)
{
    prachStatPrms.nMaxCells = nCells;
    prachStatPrms.nMaxOccaProc = nOccasaions;
    prachStatPrms.pCellPrms = prachCellStatPrms.data();
    prachStatPrms.pOccaPrms = prachOccaStatPrms.data();
    prachStatPrms.pDbg = &statDbgPrm;

    dataIn.pTDataRx = dataRxTensorPrm.data();

    // Allocate output
    num_detectedPrmb = tensor_device(CUPHY_R_32U,
                                   nOccasaions,
                                   cuphy::tensor_flags::align_tight);
    prmbIndex_estimates = tensor_device(CUPHY_R_32U,
                                      CUPHY_PRACH_RX_NUM_PREAMBLE,
                                      nOccasaions,
                                      cuphy::tensor_flags::align_tight);
    prmbDelay_estimates = tensor_device(CUPHY_R_32F,
                                      CUPHY_PRACH_RX_NUM_PREAMBLE,
                                      nOccasaions,
                                      cuphy::tensor_flags::align_tight);
    prmbPower_estimates = tensor_device(CUPHY_R_32F,
                                      CUPHY_PRACH_RX_NUM_PREAMBLE,
                                      nOccasaions,
                                      cuphy::tensor_flags::align_tight);
    ant_rssi = tensor_pinned(CUPHY_R_32F,
                            MAX_N_ANTENNAS_SUPPORTED,
                            nOccasaions,
                            cuphy::tensor_flags::align_tight);
    rssi = tensor_pinned(CUPHY_R_32F,
                        nOccasaions,
                        cuphy::tensor_flags::align_tight);

    interference = tensor_pinned(CUPHY_R_32F,
                            nOccasaions,
                            cuphy::tensor_flags::align_tight);

    dataOut.numDetectedPrmb.desc  = num_detectedPrmb.desc().handle();
    dataOut.numDetectedPrmb.pAddr = num_detectedPrmb.addr();
    dataOut.prmbIndexEstimates.desc  = prmbIndex_estimates.desc().handle();
    dataOut.prmbIndexEstimates.pAddr = prmbIndex_estimates.addr();
    dataOut.prmbDelayEstimates.desc  = prmbDelay_estimates.desc().handle();
    dataOut.prmbDelayEstimates.pAddr = prmbDelay_estimates.addr();
    dataOut.prmbPowerEstimates.desc  = prmbPower_estimates.desc().handle();
    dataOut.prmbPowerEstimates.pAddr = prmbPower_estimates.addr();
    dataOut.rssi.desc  = rssi.desc().handle();
    dataOut.rssi.pAddr = rssi.addr();
    dataOut.antRssi.desc  = ant_rssi.desc().handle();
    dataOut.antRssi.pAddr = ant_rssi.addr();
    dataOut.interference.desc  = interference.desc().handle();
    dataOut.interference.pAddr = interference.addr();

    prachDynPrms.nOccaProc = prachOccaDynPrms.size();
    prachDynPrms.pOccaPrms = prachOccaDynPrms.data();
    prachDynPrms.pDataIn = &dataIn;
    prachDynPrms.pDataOut = &dataOut;
    prachDynPrms.cuStream = cuStrm;
    prachDynPrms.procModeBmsk = procModeBmsk_;

    StatusOutput = {cuphyPrachStatusType_t::CUPHY_PRACH_STATUS_SUCCESS_OR_UNTRACKED_ISSUE, MAX_UINT16, MAX_UINT16};
    prachDynPrms.pStatusOut = &StatusOutput;
    dynDbgPrm.enableApiLogging = 0;
    prachDynPrms.pDbg = &dynDbgPrm;
}

void PrachApiDataset::readReferenceValues(hdf5hpp::hdf5_file& prach_file, int numOccaInCell)
{
    ref_ant_rssi.resize(nOccasaions + numOccaInCell);
    ref_prmb_index.resize(nOccasaions + numOccaInCell);
    ref_delay_time.resize(nOccasaions + numOccaInCell);
    ref_peak_power.resize(nOccasaions + numOccaInCell);

    using tensor_pinned_R_32U = typed_tensor<CUPHY_R_32U, pinned_alloc>;
    using tensor_pinned_R_32F = typed_tensor<CUPHY_R_32F, pinned_alloc>;

    int dynIdx = prachOccaDynPrms.size() - numOccaInCell;

    for(int i = 0; i < numOccaInCell; ++i)
    {
        std::string dataSetName = "detIdx_" + std::to_string(i);
        tensor_pinned_R_32U matlab_numPrmb = typed_tensor_from_dataset<CUPHY_R_32U,
                                                    pinned_alloc>(prach_file.open_dataset(dataSetName.c_str()));
        ref_num_prmb.push_back(*(matlab_numPrmb.addr()));

        uint32_t numAnt;
        dataSetName = "numAnt_" + std::to_string(i);
        prach_file.open_dataset(dataSetName.c_str()).read(&numAnt);

        ref_num_ant.push_back(numAnt);

        dataSetName = "noise_det_" + std::to_string(i);
        tensor_pinned_R_32F matlab_interference = typed_tensor_from_dataset<CUPHY_R_32F,
            pinned_alloc>(prach_file.open_dataset(dataSetName.c_str()));

        ref_interference.push_back(*(matlab_interference.addr()));

        dataSetName = "rssi_det_" + std::to_string(i);
        tensor_pinned_R_32F matlab_rssi = typed_tensor_from_dataset<CUPHY_R_32F,
            pinned_alloc>(prach_file.open_dataset(dataSetName.c_str()));

        ref_rssi.push_back(*(matlab_rssi.addr()));

        dataSetName = "antRssi_det_" + std::to_string(i);
        using tensor_pinned_R_32F = typed_tensor<CUPHY_R_32F, pinned_alloc>;
        tensor_pinned_R_32F matlab_ant_rssi = typed_tensor_from_dataset<CUPHY_R_32F,
            pinned_alloc>(prach_file.open_dataset(dataSetName.c_str()));

        ref_ant_rssi[dynIdx].resize(numAnt);

        memcpy(ref_ant_rssi[dynIdx].data(), matlab_ant_rssi.addr(), sizeof(float) * numAnt);

        if(ref_num_prmb.back() > 0)
        {
            // Read results from matlab test vector
            dataSetName = "prmbIdx_det_" + std::to_string(i);
            tensor_pinned_R_32U matlab_prmbIndex_estimates = typed_tensor_from_dataset<CUPHY_R_32U,
                pinned_alloc>(prach_file.open_dataset(dataSetName.c_str()));

            dataSetName = "delay_time_det_" + std::to_string(i);
            tensor_pinned_R_32F matlab_prmbDelay_estimates = typed_tensor_from_dataset<CUPHY_R_32F,
                pinned_alloc>(prach_file.open_dataset(dataSetName.c_str()));

            dataSetName = "peak_det_" + std::to_string(i);
            tensor_pinned_R_32F matlab_prmbPower_estimates = typed_tensor_from_dataset<CUPHY_R_32F,
                pinned_alloc>(prach_file.open_dataset(dataSetName.c_str()));

            ref_prmb_index[dynIdx].resize(ref_num_prmb.back());
            ref_delay_time[dynIdx].resize(ref_num_prmb.back());
            ref_peak_power[dynIdx].resize(ref_num_prmb.back());

            memcpy(ref_prmb_index[dynIdx].data(), matlab_prmbIndex_estimates.addr(), sizeof(int) * ref_num_prmb.back());
            memcpy(ref_delay_time[dynIdx].data(), matlab_prmbDelay_estimates.addr(), sizeof(float) * ref_num_prmb.back());
            memcpy(ref_peak_power[dynIdx].data(), matlab_prmbPower_estimates.addr(), sizeof(float) * ref_num_prmb.back());
            dynIdx++;
        }
    }
}

int PrachApiDataset::evaluateOutput()
{
    if(!enable_ref_check)
    {
        return 1;
    }

    // Copy output from GPU to CPU for reference comparison
    std::vector<uint32_t> gpu_num_detectedPrmb(nOccasaions);
    CUDA_CHECK(cudaMemcpy(gpu_num_detectedPrmb.data(), num_detectedPrmb.addr(), sizeof(uint32_t) * nOccasaions, cudaMemcpyDeviceToHost));

    float* gpu_ant_rssi = (float*)ant_rssi.addr();
    float* gpu_rssi = (float*)rssi.addr();
    float* gpu_interference = (float*)interference.addr();

    // Compare results between GPU and Matlab output
    NVLOGC_FMT(NVLOG_PRACH, "---------------------------------------------------------------");
    NVLOGC_FMT(NVLOG_PRACH, "Comparing test vectors ...");

    int index_est_mismatch = 0;
    int delay_est_mismatch = 0;
    int power_est_mismatch = 0;

    int ant_rssi_mismatch = 0;
    int rssi_mismatch = 0;
    int interference_mismatch = 0;

    for(int dynIdx = 0; dynIdx < prachOccaDynPrms.size(); ++dynIdx)
    {
        uint16_t statIdx = prachOccaDynPrms[dynIdx].occaPrmStatIdx;
        NVLOGC_FMT(NVLOG_PRACH, "Comparing PRACH occasion: {}", statIdx);
        int numPrmb = gpu_num_detectedPrmb[statIdx];

        if (numPrmb != ref_num_prmb[dynIdx]) {
            NVLOGC_FMT(NVLOG_PRACH, "Preamble count - matlab = {} vs gpu = {}", ref_num_prmb[dynIdx], numPrmb);
            index_est_mismatch += 1;
            delay_est_mismatch += 1;
            power_est_mismatch += 1;
        }
        else if(numPrmb > 0)
        {
            using r32u_t = type_traits<CUPHY_R_32U>::type;
            using f32_t = type_traits<CUPHY_R_32U>::type;

            cuphy::typed_tensor<CUPHY_R_32U, pinned_alloc> gpu_prmbIndex_estimates(numPrmb, cuphy::tensor_flags::align_tight);
            cuphy::typed_tensor<CUPHY_R_32F, pinned_alloc> gpu_prmbDelay_estimates(numPrmb, cuphy::tensor_flags::align_tight);
            cuphy::typed_tensor<CUPHY_R_32F, pinned_alloc> gpu_prmbPower_estimates(numPrmb, cuphy::tensor_flags::align_tight);

            CUDA_CHECK(cudaMemcpy(gpu_prmbIndex_estimates.addr(), (r32u_t*)prmbIndex_estimates.addr() + statIdx * CUPHY_PRACH_RX_NUM_PREAMBLE, numPrmb * sizeof(uint32_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(gpu_prmbDelay_estimates.addr(),  (f32_t*)prmbDelay_estimates.addr() + statIdx * CUPHY_PRACH_RX_NUM_PREAMBLE, numPrmb * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(gpu_prmbPower_estimates.addr(),  (f32_t*)prmbPower_estimates.addr() + statIdx * CUPHY_PRACH_RX_NUM_PREAMBLE, numPrmb * sizeof(float), cudaMemcpyDeviceToHost));

            for (int prmbCounter = 0; prmbCounter < numPrmb; prmbCounter++) {
                int matlab_prmbIndex_val = ref_prmb_index[dynIdx][prmbCounter];

                int gpuCounterFound = -1;
                for (int prmbCounterGpu= 0; prmbCounterGpu < numPrmb; prmbCounterGpu++) {
                    int gpu_prmbIndex_val = gpu_prmbIndex_estimates(prmbCounterGpu);
                    if (matlab_prmbIndex_val == gpu_prmbIndex_val) {
                        gpuCounterFound = prmbCounterGpu;
                    }
                }
                if (gpuCounterFound == -1) {
                    index_est_mismatch += 1;
                    delay_est_mismatch += 1;
                    power_est_mismatch += 1;
                }
                else {
                    int gpu_prmbIndex_val = gpu_prmbIndex_estimates(gpuCounterFound);
                    NVLOGC_FMT(NVLOG_PRACH, "prmbIndex - matlab = {:6d} vs. gpu = {:6d}", matlab_prmbIndex_val, gpu_prmbIndex_val);
                    if (matlab_prmbIndex_val != gpu_prmbIndex_val) {
                        NVLOGE_FMT(NVLOG_PRACH, AERIAL_CUPHY_EVENT, " => Mismatch");
                        index_est_mismatch += 1;
                    }
                    float matlab_prmbDelay_val = ref_delay_time[dynIdx][prmbCounter]*1e6;
                    float gpu_prmbDelay_val = gpu_prmbDelay_estimates(gpuCounterFound)*1e6;
                    NVLOGC_FMT(NVLOG_PRACH, "prmbDelay - matlab = {:6.4f} vs. gpu = {:6.4f}", matlab_prmbDelay_val, gpu_prmbDelay_val);
                    if (!compare_approx(matlab_prmbDelay_val, gpu_prmbDelay_val, 0.0001f)) {
                        NVLOGE_FMT(NVLOG_PRACH, AERIAL_CUPHY_EVENT, " => Mismatch");
                        delay_est_mismatch += 1;
                    }
                    float matlab_prmbPower_val = ref_peak_power[dynIdx][prmbCounter];
                    float gpu_prmbPower_val = gpu_prmbPower_estimates(gpuCounterFound);
                    NVLOGC_FMT(NVLOG_PRACH, "prmbPower - matlab = {:6.4f} vs. gpu = {:6.4f}", matlab_prmbPower_val, gpu_prmbPower_val);
                    if (!compare_approx(matlab_prmbPower_val, gpu_prmbPower_val, 0.001f)) {
                        NVLOGE_FMT(NVLOG_PRACH, AERIAL_CUPHY_EVENT, " => Mismatch");
                        power_est_mismatch += 1;
                    }
                }
            }
        }

        NVLOGC_FMT(NVLOG_PRACH, "interference - matlab = {:6.4f} vs. gpu = {:6.4f}", ref_interference[dynIdx], gpu_interference[statIdx]);
        if (!compare_approx(ref_interference[dynIdx], gpu_interference[statIdx], 0.001f))
        {
            NVLOGE_FMT(NVLOG_PRACH, AERIAL_CUPHY_EVENT, " => Mismatch");
            interference_mismatch += 1;
        }

        NVLOGC_FMT(NVLOG_PRACH, "rssi - matlab = {:6.4f} vs. gpu = {:6.4f}", ref_rssi[dynIdx], gpu_rssi[statIdx]);
        if (!compare_approx(ref_rssi[dynIdx], gpu_rssi[statIdx], 0.002f))
        {
            NVLOGE_FMT(NVLOG_PRACH, AERIAL_CUPHY_EVENT, " => Mismatch");
            rssi_mismatch += 1;
        }

        uint32_t numAnt = ref_num_ant[dynIdx];
        for(int antIndex = 0; antIndex < numAnt; ++antIndex)
        {
            NVLOGC_FMT(NVLOG_PRACH, "ant rssi - matlab = {:6.4f} vs. gpu = {:6.4f}", ref_ant_rssi[dynIdx][antIndex], gpu_ant_rssi[statIdx * MAX_N_ANTENNAS_SUPPORTED + antIndex]);
            if (!compare_approx(ref_ant_rssi[dynIdx][antIndex], gpu_ant_rssi[statIdx * MAX_N_ANTENNAS_SUPPORTED + antIndex], 0.002f))
            {
                NVLOGE_FMT(NVLOG_PRACH, AERIAL_CUPHY_EVENT, " => Mismatch");
                ant_rssi_mismatch += 1;
            }
        }
    }

    int errCount = index_est_mismatch+delay_est_mismatch+power_est_mismatch+rssi_mismatch+ant_rssi_mismatch+interference_mismatch;
    if (errCount == 0) {
        NVLOGC_FMT(NVLOG_PRACH, "========> Test PASS");
    }
    else {
        NVLOGE_FMT(NVLOG_PRACH, AERIAL_CUPHY_EVENT, "========> Test FAIL");
    }

    return errCount;
}

// SSB datasets

// construct from h5 file
ssbStaticApiDataset::ssbStaticApiDataset(int cfg_max_cells_per_slot)
{
    ssbStatPrms.nMaxCellsPerSlot = cfg_max_cells_per_slot; // Should be specified. Currently no fallback if 0.
    ssbTracker.pMemoryFootprint  = nullptr;
    ssbStatPrms.pOutInfo         = &ssbTracker;
    ssbStatPrms.pDbgPrms         = nullptr;
}

//---------------------------------------------------------------------------------------------------
// Dataset holds dynamic PDCCH API paramaters/data

// Construct dataset from HDF5 file
ssbDynApiDataset::ssbDynApiDataset(const std::string& inputFileName, uint32_t cfg_max_cells, cudaStream_t cuStrm, uint64_t procModeBmsk)
{
    //NVLOGC_FMT(NVLOG_SSB, "max cells per slot {}", cfg_max_cells);
    // Output buffer allocation will happen based on the number of max cells.
    max_cells = cfg_max_cells;

    ssb_dyn_params.nCells = 0;
    ssb_dyn_params.nSSBlocks = 0;
    ssb_dyn_params.nPrecodingMatrices = 0;

    // Reserve space up front to avoid the need to update the parent pointers on every cumulative update.
    // If we exceed the reserved space, the various parent pointers will be incorrect.
    per_cell_SSB_params.reserve(cfg_max_cells);
    per_SS_block_params.reserve(cfg_max_cells * CUPHY_SSB_MAX_SSBS_PER_CELL_PER_SLOT);
    ssb_precoding_matrix.reserve(cfg_max_cells * CUPHY_SSB_MAX_SSBS_PER_CELL_PER_SLOT);

    // Large buffer added as cuPHY tools needs a buffer with a power of 2 size.
    // Allocating once with the assumption that there will be at most max_cells cells
    large_buffer_elements = MAX_N_PRBS_SUPPORTED * CUPHY_N_TONES_PER_PRB * OFDM_SYMBOLS_PER_SLOT * MAX_DL_LAYERS;
    large_buffer_bytes    = large_buffer_elements * sizeof(__half2);
    large_buffer          = make_unique_device<__half2>(max_cells * large_buffer_elements);
    // Reset the entire output buffer.
    CUDA_CHECK(cudaMemsetAsync(large_buffer.get(), 0, max_cells * large_buffer_bytes, cuStrm)); // This is done in cuStrm

    output_data.resize(max_cells);
    output_tensorPrm.resize(max_cells);
    data_tx_tensor.resize(max_cells);

    data_in.resize(1);

    int input_dims[1] = {(int) per_SS_block_params.capacity()};
    input_data.emplace_back(cuphy::tensor_layout(1, input_dims, nullptr));
    data_in[0] = {input_data[0].addr(), cuphySsbDataIn_t::CPU_BUFFER};

    // OK to do the next actions  here given overprovisioned allocations above.
    ssb_dyn_params.pDataIn            = data_in.data();
    ssb_dyn_params.pDataOut           = output_data.data();
    ssb_dyn_params.pDataOut->pTDataTx = output_tensorPrm.data();
    ssb_dyn_params.cuStream           = cuStrm;
    ssb_dyn_params.procModeBmsk       = procModeBmsk;

    // Update dynamic parameters. This method will also be called to stitch together multiple one-cell TVs.
    cumulativeUpdate(inputFileName, cuStrm);
}

void ssbDynApiDataset::cumulativeUpdate(const std::string& inputFileName, cudaStream_t cuStrm){

    int current_cells        = ssb_dyn_params.nCells;
    int current_SSBs         = ssb_dyn_params.nSSBlocks;

    // Open HDF5 file
    hdf5hpp::hdf5_file fInput = hdf5hpp::hdf5_file::open(inputFileName.c_str());
    CfgFileName.push_back(inputFileName);

    // Every TV corresponds to a separate cell
    ssb_dyn_params.nCells += 1;
    if (ssb_dyn_params.nCells > max_cells) {
        std::string error_msg = "Current SSB dyn. cells are " + std::to_string(ssb_dyn_params.nCells) + " but max. cells was " + std::to_string(max_cells) + "\n";
        throw std::runtime_error(error_msg.c_str());
    }

    // Figure out nSSBs of this TV. Resize  SSB params and read values. Update nSSBs
    hdf5hpp::hdf5_dataset dset = fInput.open_dataset("SSTxParams"); //FIXME
    int num_SSBs = dset.get_dataspace().get_dimensions()[0];
    ssb_dyn_params.nSSBlocks += num_SSBs;
    per_SS_block_params.resize(ssb_dyn_params.nSSBlocks);

    int num_REs = cuphy::get_HDF5_dataset_info(fInput.open_dataset("X_tf_fp16")).layout().dimensions()[0];
    // FIXME write functions to fill in the new API structs (params first). TODO: Matlab should be updated too
    cuphy::read_SSB_dyn_params_from_file(per_SS_block_params, per_cell_SSB_params, ssb_precoding_matrix, fInput, num_REs, ssb_dyn_params.nPrecodingMatrices, current_SSBs, current_cells);

    // Stitch together the MIB from the input
    using tensor_pinned_R_32U          = typed_tensor<CUPHY_R_32U, pinned_alloc>;
    tensor_pinned_R_32U h5_pbch_mib = typed_tensor_from_dataset<CUPHY_R_32U, pinned_alloc>(fInput.open_dataset("x_mib")); // payload bits for SSB

    input_data[0](current_SSBs) = 0; // Pack 24 elements with one bit per uint32_t to 1 uint32_t element with the most significant? (FIXME) 24 bits set.

    int N_mib_elements_per_SSB = h5_pbch_mib.layout().dimensions()[0];
    int N_MIBs = h5_pbch_mib.layout().dimensions()[1];
    if (num_SSBs != N_MIBs) {
        throw std::runtime_error("Mismatch between number of SSBs and number of MIBs");
    }

    // For now support both old and new MIB dataset layout.
    if (N_mib_elements_per_SSB == CUPHY_SSB_N_MIB_BITS) { //Old dataset; one bit per uint32_t element
        if (N_MIBs != 1) {
            throw std::runtime_error("Multiple SSB per cell not supported in old TV format!");
        }
        NVLOGC_FMT(NVLOG_SSB, "You are using the old TV format with 24 uint32_t elements with one valid bit each for MIB. It is still supported but it's better to update your TV!");
        for(int i = 0; i < N_mib_elements_per_SSB; ++i)  {
            input_data[0](current_SSBs) |= ((h5_pbch_mib(i) & 0x1U) << (CUPHY_SSB_N_MIB_BITS - i - 1));
        }
    } else if (N_mib_elements_per_SSB == 1) {
        for (int j = 0; j < N_MIBs; j++) {
            input_data[0](current_SSBs + j) = h5_pbch_mib(0, j); // least significatn CUPHY_SSB_N_MIB_BITS are valid
        }
    } else {
        throw std::runtime_error("Unsupported MIB layout! TV's x_mib dataset should either contain 1 uint32_t element (packed 24 bits) or 24 uint32_t elements with one value bit each.");
    }

     //Update p pointers
     ssb_dyn_params.pPerCellSsbDynParams = per_cell_SSB_params.data();
     ssb_dyn_params.pPerSsBlockParams    = per_SS_block_params.data();
     ssb_dyn_params.pPmwParams           = ssb_precoding_matrix.data();

     //Update pDataOut
     int ports = cuphy::get_HDF5_dataset_info(fInput.open_dataset("X_tf_fp16")).layout().dimensions()[2];

     data_tx_tensor[current_cells] = tensor_device(large_buffer.get() + current_cells * large_buffer_elements, CUPHY_C_16F,
                                                   num_REs, OFDM_SYMBOLS_PER_SLOT, ports, cuphy::tensor_flags::align_tight);

     ssb_dyn_params.pDataOut->pTDataTx[current_cells].desc  = data_tx_tensor[current_cells].desc().handle();
     ssb_dyn_params.pDataOut->pTDataTx[current_cells].pAddr = data_tx_tensor[current_cells].addr();
}

int ssbDynApiDataset::refCheck(bool verbose) {
    int total_err_cnt = 0;
    NVLOGD_FMT(NVLOG_SSB, "ref check for {} buffers", ssb_dyn_params.nCells);
    using tensor_pinned_C_16F          = typed_tensor<CUPHY_C_16F, pinned_alloc>;
    for (int output_buffer = 0; output_buffer < ssb_dyn_params.nCells; output_buffer++) {
        NVLOGD_FMT(NVLOG_SSB, "buffer {} with name {}", output_buffer, CfgFileName[output_buffer].c_str());
        hdf5hpp::hdf5_file fInput = hdf5hpp::hdf5_file::open(CfgFileName[output_buffer].c_str());
        tensor_pinned_C_16F ssb_tx_ref_output = typed_tensor_from_dataset<CUPHY_C_16F, pinned_alloc>(fInput.open_dataset("X_tf_fp16"));

        typed_tensor<CUPHY_C_16F, pinned_alloc> h_ssb_out_tensor(ssb_tx_ref_output.layout());
        h_ssb_out_tensor.convert(data_tx_tensor[output_buffer], ssb_dyn_params.cuStream);
        CUDA_CHECK(cudaStreamSynchronize(ssb_dyn_params.cuStream));
        NVLOGD_FMT(NVLOG_SSB, "output tensor dims {} {} {} ",
                h_ssb_out_tensor.layout().dimensions()[0],
                h_ssb_out_tensor.layout().dimensions()[1],
                h_ssb_out_tensor.layout().dimensions()[2]);

        int checked_symbols = 0, gpu_mismatch = 0;
        for(int layer_id = 0; layer_id <  h_ssb_out_tensor.layout().dimensions()[2]; layer_id++)
        //int layer_id = 0;
        {
            for(int symbol_id = 0; symbol_id < OFDM_SYMBOLS_PER_SLOT; symbol_id++)
            {
                for(int freq_idx = 0; freq_idx <  h_ssb_out_tensor.layout().dimensions()[0]; freq_idx++)
                {
                    __half2 gpu_symbol = h_ssb_out_tensor(freq_idx, symbol_id, layer_id);
                    __half2 ref_symbol;
                    /* The reference HDF5 dataset should contain all the number of PRBs for this BWP. Any unallocated PRB should be empty. */
                    ref_symbol.x = (half)ssb_tx_ref_output(freq_idx, symbol_id, layer_id).x;
                    ref_symbol.y = (half)ssb_tx_ref_output(freq_idx, symbol_id, layer_id).y;
                    checked_symbols += 1;
                    if(!complex_approx_equal<__half2, __half>(gpu_symbol, ref_symbol, 0.0001f))
                    {
                        NVLOGD_FMT(NVLOG_SSB, "Error! Mismatch for symbol (freq_bin {}, symbol {}, layer {}):  expected={} + i {} vs. gpu={} + i {}",
                                freq_idx, symbol_id, layer_id,
                               (float) ref_symbol.x, (float) ref_symbol.y,
                               (float) gpu_symbol.x, (float) gpu_symbol.y);
                        gpu_mismatch += 1;
                     }
                 }
            }
        }
        total_err_cnt += gpu_mismatch;

       //Same output as orig. phase-0 example
       if (verbose && (gpu_mismatch == 0)) {
           NVLOGC_FMT(NVLOG_SSB, "====> TV {}: Test PASS",  CfgFileName[output_buffer].c_str());
       } else if (gpu_mismatch != 0) {
           NVLOGC_FMT(NVLOG_SSB, "====> TV {}: Test FAIL. Found {} mismatched symbols", CfgFileName[output_buffer].c_str(), gpu_mismatch);
       }
       fflush(stdout);
    }
    return total_err_cnt;
}


// CSI-RS datasets

// construct from h5 file
csirsStaticApiDataset::csirsStaticApiDataset(const std::string& inputFileName, int cfg_max_cells_per_slot)
{
    csirsStatPrms.nMaxCellsPerSlot     = cfg_max_cells_per_slot; // Should be specified. Currently no fallback if 0.
    csirsStatPrms.nCells               = 0;
    csirsTracker.pMemoryFootprint      = nullptr;
    csirsStatPrms.pOutInfo             = &csirsTracker;

    cumulativeUpdate(inputFileName);
}

void csirsStaticApiDataset::cumulativeUpdate(const std::string& inputFileName)
{
    //csirs cuPHY TVs do not have cellStat_pars dataset so I'll infer dlBwp from Xtf's dataset dimension
    int current_static_cells = cellStatPrm.size();

    hdf5hpp::hdf5_file    fInput              = hdf5hpp::hdf5_file::open(inputFileName.c_str());
    int num_REs = cuphy::get_HDF5_dataset_info(fInput.open_dataset("X_tf")).layout().dimensions()[0];

    int num_cells = 1; // every cumulative update, updates 1 cell

    cellStatPrm.resize(current_static_cells + num_cells);
    memset(&cellStatPrm[current_static_cells], 0, sizeof(cuphyCellStatPrm_t)); // all fields but nPrbDlBwp will have 0 values for CSI-RS (unused)
    cellStatPrm[current_static_cells].nPrbDlBwp = num_REs / CUPHY_N_TONES_PER_PRB;  // convert to PRBs
    cellStatPrm[current_static_cells].nTxAnt = cuphy::get_HDF5_dataset_info(fInput.open_dataset("X_tf")).layout().dimensions()[2];

    csirsStatPrms.nCells             += num_cells;
    csirsStatPrms.pCellStatPrms      = cellStatPrm.data();

}

void csirsStaticApiDataset::print()
{
}

//---------------------------------------------------------------------------------------------------
// Dataset that holds dynamic CSI-RS TX API paramaters/data

// Construct dataset from HDF5 file
csirsDynApiDataset::csirsDynApiDataset(const std::string& inputFileName, uint32_t cfg_max_cells, cudaStream_t cuStrm, uint64_t procModeBmsk)
{

    // Output buffer allocation will happen based on the number of max cells.
    max_cells = cfg_max_cells;

    csirs_dyn_params.nCells    = 0;
    csirs_dyn_params.nPrecodingMatrices = 0;

    // Reserve space up front to avoid the need to update the parent pointers on every cumulative update.
    // If we exceed the reserved space, the various parent pointers will be incorrect.
    cell_params.reserve(cfg_max_cells);
    rrc_params.reserve(cfg_max_cells * CUPHY_CSIRS_MAX_NUM_PARAMS);
    csirs_precoding_matrix.resize(cfg_max_cells * CUPHY_CSIRS_MAX_NUM_PARAMS);

    // Large buffer added as cuPHY tools needs a buffer with a power of 2 size.
    // Allocating once with the assumption that there will be at most max_cells cells
    large_buffer_elements = MAX_N_PRBS_SUPPORTED * CUPHY_N_TONES_PER_PRB * OFDM_SYMBOLS_PER_SLOT * CUPHY_CSIRS_MAX_ANTENNA_PORTS;
    large_buffer_bytes    = large_buffer_elements * sizeof(__half2);
    large_buffer          = make_unique_device<__half2>(max_cells * large_buffer_elements);

    // Reset the entire output buffer.
    CUDA_CHECK(cudaMemsetAsync(large_buffer.get(), 0, max_cells * large_buffer_bytes, cuStrm)); // This is done in cuStrm

    output_data.resize(max_cells);
    output_tensorPrm.resize(max_cells);
    data_tx_tensor.resize(max_cells);

    // OK to do the next two actions  here given overprovisioned allocations above.
    csirs_dyn_params.pDataOut           = output_data.data();
    csirs_dyn_params.pDataOut->pTDataTx = output_tensorPrm.data();

    csirs_dyn_params.procModeBmsk = procModeBmsk;
    csirs_dyn_params.cuStream     = cuStrm;
    total_rrc_params = 0;

    // Update dynamic parameters. This method will also be called to stitch together multiple one-cell TVs.
    cumulativeUpdate(inputFileName, cuStrm);
}

void csirsDynApiDataset::cumulativeUpdate(const std::string& inputFileName, cudaStream_t cuStrm) {


    int current_cells     = csirs_dyn_params.nCells;

    // Open HDF5 file
    hdf5hpp::hdf5_file fInput = hdf5hpp::hdf5_file::open(inputFileName.c_str());
    CfgFileName.push_back(inputFileName);

    // Every TV corresponds to a separate cell
    csirs_dyn_params.nCells += 1;
    if (csirs_dyn_params.nCells > max_cells) {
        std::string error_msg = "Current CSI-RS dyn. cells are " + std::to_string(csirs_dyn_params.nCells) + " but max. cells was " + std::to_string(max_cells) + "\n";
        throw std::runtime_error(error_msg.c_str());
    }

    hdf5hpp::hdf5_dataset dset = fInput.open_dataset("CsirsParamsList");
    int num_rrc_params = dset.get_num_elements();

    // Resize and Populate cell_params
    cell_params.resize(csirs_dyn_params.nCells);
    cell_params[current_cells].rrcParamsOffset = total_rrc_params; // should happen before total_rrc_params updates
    cell_params[current_cells].nRrcParams = num_rrc_params;
    cell_params[current_cells].slotBufferIdx = current_cells;
    cell_params[current_cells].cellPrmStatIdx = current_cells;

    // Resize and populate RRC params for this cell
    total_rrc_params += num_rrc_params;
    rrc_params.resize(total_rrc_params);
    cuphyCsirsRrcDynPrm_t* current_csirs_rrc_dyn_params = &rrc_params[total_rrc_params-num_rrc_params];
    cuphy::read_csirs_dynamic_params_from_file(current_csirs_rrc_dyn_params, num_rrc_params, fInput, csirs_precoding_matrix, csirs_dyn_params.nPrecodingMatrices);

    //Update p pointers
    csirs_dyn_params.pRrcDynPrm     = rrc_params.data();
    csirs_dyn_params.pCellParam     = cell_params.data();
    csirs_dyn_params.pPmwParams     = csirs_precoding_matrix.data();

    //Update pDataOut
    int num_REs = cuphy::get_HDF5_dataset_info(fInput.open_dataset("X_tf_fp16")).layout().dimensions()[0];
    int ports = cuphy::get_HDF5_dataset_info(fInput.open_dataset("X_tf_fp16")).layout().dimensions()[2];

    data_tx_tensor[current_cells] = tensor_device(large_buffer.get() + current_cells * large_buffer_elements, CUPHY_C_16F,
                                                  num_REs, OFDM_SYMBOLS_PER_SLOT, ports, cuphy::tensor_flags::align_tight);

    csirs_dyn_params.pDataOut->pTDataTx[current_cells].desc  = data_tx_tensor[current_cells].desc().handle();
    csirs_dyn_params.pDataOut->pTDataTx[current_cells].pAddr = data_tx_tensor[current_cells].addr();

     //CUDA_CHECK(cudaStreamSynchronize(cuStrm));
}

int csirsDynApiDataset::refCheck(bool verbose) {
    int total_err_cnt = 0;

    //NVLOGC_FMT(NVLOG_CSIRS, "ref check for {} buffers", csirs_dyn_params.nCells);
    using tensor_pinned_C_16F          = typed_tensor<CUPHY_C_16F, pinned_alloc>;
    for (int output_buffer = 0; output_buffer < csirs_dyn_params.nCells; output_buffer++) {
        //NVLOGC_FMT(NVLOG_CSIRS, "buffer {} with name {}", output_buffer, CfgFileName[output_buffer].c_str());
        hdf5hpp::hdf5_file fInput = hdf5hpp::hdf5_file::open(CfgFileName[output_buffer].c_str());
        tensor_pinned_C_16F csirs_tx_ref_output = typed_tensor_from_dataset<CUPHY_C_16F, pinned_alloc>(fInput.open_dataset("X_tf_fp16"));


        typed_tensor<CUPHY_C_16F, pinned_alloc> h_csirs_out_tensor(data_tx_tensor[output_buffer].layout());
        h_csirs_out_tensor.convert(data_tx_tensor[output_buffer], csirs_dyn_params.cuStream);
        CUDA_CHECK(cudaStreamSynchronize(csirs_dyn_params.cuStream));
        NVLOGC_FMT(NVLOG_CSIRS, "CSI-RS dims {} {} {}",
                h_csirs_out_tensor.layout().dimensions()[0],
                h_csirs_out_tensor.layout().dimensions()[1],
                h_csirs_out_tensor.layout().dimensions()[2]);
        // It's possible that the reference tensor has one layer while the tensor provided has more. Compare only with what's available in the ref. tensor.

        int checked_symbols = 0, gpu_mismatch = 0;
        for(int layer_id = 0; layer_id <  csirs_tx_ref_output.layout().dimensions()[2]; layer_id++)
        {
            for(int symbol_id = 0; symbol_id < csirs_tx_ref_output.layout().dimensions()[1]; symbol_id++)
            {
                for(int freq_idx = 0; freq_idx < csirs_tx_ref_output.layout().dimensions()[0]; freq_idx++)
                {
                    __half2 gpu_symbol = h_csirs_out_tensor(freq_idx, symbol_id, layer_id);
                    __half2 ref_symbol;
                    /* The reference HDF5 dataset should contain all the number of PRBs for this BWP. Any unallocated PRB should be empty. */
                    ref_symbol.x = (half)csirs_tx_ref_output(freq_idx, symbol_id, layer_id).x;
                    ref_symbol.y = (half)csirs_tx_ref_output(freq_idx, symbol_id, layer_id).y;
                    checked_symbols += 1;
                    if(!complex_approx_equal<__half2, __half>(gpu_symbol, ref_symbol, 0.0001f))
                    {
                        NVLOGD_FMT(NVLOG_CSIRS, "Error! Mismatch for symbol (freq_bin {}, symbol {}, layer {}):  expected={} + i {} vs. gpu={} + i {}",
                                freq_idx, symbol_id, layer_id,
                               (float) ref_symbol.x, (float) ref_symbol.y,
                               (float) gpu_symbol.x, (float) gpu_symbol.y);
                        gpu_mismatch += 1;
                    }
                 }
            }
       }
       if (verbose) {
           if (gpu_mismatch == 0) {
               NVLOGC_FMT(NVLOG_CSIRS, "====> TV {}: Test PASS", CfgFileName[output_buffer].c_str());
           } else {
               NVLOGC_FMT(NVLOG_CSIRS, "====> TV {}: Test FAIL. Found {} mismatched symbols", CfgFileName[output_buffer].c_str(), gpu_mismatch);
           }
       }
       total_err_cnt += gpu_mismatch;
    }
    return total_err_cnt;
}

//---------------------------------------------------------------------------------------------------
// Dataset that holds dynamic CSI-RS RX API paramaters/data

// Construct dataset from HDF5 file
csirsRxDynApiDataset::csirsRxDynApiDataset(const std::string& inputFileName, uint32_t cfg_max_cells, uint32_t cfg_max_ues, cudaStream_t cuStrm, uint64_t procModeBmsk)
{
    max_cells = cfg_max_cells;
    max_ues   = cfg_max_ues;

    csirs_rx_dyn_params.nCells    = 0;
    csirs_rx_dyn_params.nUes      = 0;
    nPrecodingMatrices = 0;
    
    // Reserve space up front to avoid the need to update the parent pointers on every cumulative update.
    // If we exceed the reserved space, the various parent pointers will be incorrect.
    cell_params.reserve(cfg_max_cells);
    rrc_params.reserve(cfg_max_cells * CUPHY_CSIRS_MAX_NUM_PARAMS);
    ue_params.reserve(cfg_max_ues);
    csirs_precoding_matrix.resize(cfg_max_cells * CUPHY_CSIRS_MAX_NUM_PARAMS);

    input_data.resize(1);
    input_tensorPrm.resize(max_ues);
    data_rx_tensor.resize(max_ues);
    
    output_data.resize(1);
    output_chest_data.resize(max_ues);
    output_chest_tensorPrm.resize(max_ues*CUPHY_CSIRS_MAX_NUM_PARAMS);
    startPrbChEst.resize(max_ues*CUPHY_CSIRS_MAX_NUM_PARAMS);
    sizePrbChEst.resize(max_ues*CUPHY_CSIRS_MAX_NUM_PARAMS);
    ch_est_tensor.resize(max_ues*CUPHY_CSIRS_MAX_NUM_PARAMS);
    
    // OK to do the next two actions  here given overprovisioned allocations above.
    csirs_rx_dyn_params.pDataIn                   = input_data.data();
    csirs_rx_dyn_params.pDataIn->pTDataRx         = input_tensorPrm.data();
    
    csirs_rx_dyn_params.pDataOut                  = output_data.data();
    csirs_rx_dyn_params.pDataOut->pChEstBuffInfo  = output_chest_data.data();

    csirs_rx_dyn_params.procModeBmsk = procModeBmsk;
    csirs_rx_dyn_params.cuStream     = cuStrm;
    total_rrc_params = 0;

    // Update dynamic parameters. This method will also be called to stitch together multiple one-cell TVs.
    cumulativeUpdate(inputFileName, cuStrm);
}

void csirsRxDynApiDataset::cumulativeUpdate(const std::string& inputFileName, cudaStream_t cuStrm) {


    int current_cells     = csirs_rx_dyn_params.nCells;
    int current_ues       = csirs_rx_dyn_params.nUes;

    // Open HDF5 file
    hdf5hpp::hdf5_file fInput = hdf5hpp::hdf5_file::open(inputFileName.c_str());
    CfgFileName.push_back(inputFileName);

    // Every TV corresponds to a separate cell
    csirs_rx_dyn_params.nCells += 1;
    if (csirs_rx_dyn_params.nCells > max_cells) {        
        NVLOGE_FMT(NVLOG_CSIRSRX, AERIAL_CUPHY_EVENT, "Current CSI-RS Rx dyn. cells are {} but max. cells was {}", csirs_rx_dyn_params.nCells, max_cells);
        return;
    }
    
    csirs_rx_dyn_params.nUes += 1;
    if (csirs_rx_dyn_params.nUes > max_ues) {
        NVLOGE_FMT(NVLOG_CSIRSRX, AERIAL_CUPHY_EVENT, "Current CSI-RS Rx dyn. ues are {} but max. ues was {}", csirs_rx_dyn_params.nUes, max_ues);
        return;
    }

    hdf5hpp::hdf5_dataset dset = fInput.open_dataset("CsirsParamsList");
    int num_rrc_params = dset.get_num_elements();

    // Resize and Populate cell_params
    cell_params.resize(csirs_rx_dyn_params.nCells);
    cell_params[current_cells].rrcParamsOffset = total_rrc_params; // should happen before total_rrc_params updates
    cell_params[current_cells].nRrcParams = num_rrc_params;
    cell_params[current_cells].slotBufferIdx = current_cells;
    cell_params[current_cells].cellPrmStatIdx = current_cells;

    // Resize and populate RRC params for this cell
    total_rrc_params += num_rrc_params;
    rrc_params.resize(total_rrc_params);
    cuphyCsirsRrcDynPrm_t* current_csirs_rrc_dyn_params = &rrc_params[total_rrc_params-num_rrc_params];
    cuphy::read_csirs_dynamic_params_from_file(current_csirs_rrc_dyn_params, num_rrc_params, fInput, csirs_precoding_matrix, nPrecodingMatrices); //no precoding is considered

    //Update p pointers
    csirs_rx_dyn_params.pRrcDynPrm     = rrc_params.data();
    csirs_rx_dyn_params.pCellParam     = cell_params.data();
    csirs_rx_dyn_params.pUeParam       = ue_params.data();

    //Update pDataIn
    data_rx_tensor[current_ues] = cuphy::tensor_from_dataset(fInput.open_dataset("X_tf"), CUPHY_C_32F , cuphy::tensor_flags::align_tight, cuStrm);
    csirs_rx_dyn_params.pDataIn->pTDataRx[current_ues].desc  = data_rx_tensor[current_ues].desc().handle();
    csirs_rx_dyn_params.pDataIn->pTDataRx[current_ues].pAddr = data_rx_tensor[current_ues].addr();
    uint8_t nRxAnt = cuphy::get_HDF5_dataset_info(fInput.open_dataset("X_tf")).layout().dimensions()[2];
    CUDA_CHECK(cudaStreamSynchronize(cuStrm));
    
    // update UE parameters for this ue
    ue_params.resize(csirs_rx_dyn_params.nUes);
    ue_params[current_ues].cellPrmStatIdx = current_cells;
    ue_params[current_ues].nRxAnt = nRxAnt;
    
    uint8_t nPortsLUT[18] = {1, 1, 2, 4, 4, 8, 8, 8, 12, 12, 16, 16, 24, 24, 24, 32, 32, 32};
    
    //update pDataOut
    csirs_rx_dyn_params.pDataOut->pChEstBuffInfo[current_ues].nCsirs       = num_rrc_params;
    csirs_rx_dyn_params.pDataOut->pChEstBuffInfo[current_ues].startPrb     = startPrbChEst.data() + current_ues*CUPHY_CSIRS_MAX_NUM_PARAMS;
    csirs_rx_dyn_params.pDataOut->pChEstBuffInfo[current_ues].sizePrb      = sizePrbChEst.data()  + current_ues*CUPHY_CSIRS_MAX_NUM_PARAMS;
    csirs_rx_dyn_params.pDataOut->pChEstBuffInfo[current_ues].tChEstBuffer = output_chest_tensorPrm.data() + current_ues*CUPHY_CSIRS_MAX_NUM_PARAMS;
    for(int rrc_idx = 0; rrc_idx < num_rrc_params; rrc_idx++)
    {
        csirs_rx_dyn_params.pDataOut->pChEstBuffInfo[current_ues].startPrb[rrc_idx] = current_csirs_rrc_dyn_params[rrc_idx].startRb;
        
        uint16_t nRb         = current_csirs_rrc_dyn_params[rrc_idx].nRb;
        uint8_t  row         = current_csirs_rrc_dyn_params[rrc_idx].row; 
        uint8_t  freqDensity = current_csirs_rrc_dyn_params[rrc_idx].freqDensity; 
        uint8_t  nPort       = nPortsLUT[row-1];
        
        if((row==1)&&(freqDensity==3))
        {
            nRb = nRb*3;
        }
        else if((freqDensity==0)||(freqDensity==1))
        {
            nRb = (nRb>>1);
        }
        
        csirs_rx_dyn_params.pDataOut->pChEstBuffInfo[current_ues].sizePrb[rrc_idx] = nRb; 
        ch_est_tensor[current_ues*CUPHY_CSIRS_MAX_NUM_PARAMS + rrc_idx] = cuphy::tensor_device(CUPHY_C_32F, 
                                                                                               nRb,
                                                                                               nPort,  // gNB ports
                                                                                               nRxAnt, // UE antennas
                                                                                               cuphy::tensor_flags::align_tight);

        size_t nBytesChEstBuffer = nRb*nPort*nRxAnt*8;
        CUDA_CHECK(cudaMemsetAsync(ch_est_tensor[current_ues*CUPHY_CSIRS_MAX_NUM_PARAMS + rrc_idx].addr(), 0, nBytesChEstBuffer, cuStrm)); // init ChEst buffer to zero

        csirs_rx_dyn_params.pDataOut->pChEstBuffInfo[current_ues].tChEstBuffer[rrc_idx].desc = ch_est_tensor[current_ues*CUPHY_CSIRS_MAX_NUM_PARAMS + rrc_idx].desc().handle();
        csirs_rx_dyn_params.pDataOut->pChEstBuffInfo[current_ues].tChEstBuffer[rrc_idx].pAddr = ch_est_tensor[current_ues*CUPHY_CSIRS_MAX_NUM_PARAMS + rrc_idx].addr();  

        // check CHEST dimension
        std::string  datasetName = "Csirs_Hest" + std::to_string(rrc_idx);
        if((nRb!=cuphy::get_HDF5_dataset_info(fInput.open_dataset(datasetName.c_str())).layout().dimensions()[0])||
           (nPort!=cuphy::get_HDF5_dataset_info(fInput.open_dataset(datasetName.c_str())).layout().dimensions()[1])||
           (nRxAnt!=cuphy::get_HDF5_dataset_info(fInput.open_dataset(datasetName.c_str())).layout().dimensions()[2]))
        {
            NVLOGE_FMT(NVLOG_CSIRSRX, AERIAL_CUPHY_EVENT, "CSIRS CHEST dimension mismatch, {}, {}, {} vs {}, {}, {}", 
                           cuphy::get_HDF5_dataset_info(fInput.open_dataset(datasetName.c_str())).layout().dimensions()[0],
                           cuphy::get_HDF5_dataset_info(fInput.open_dataset(datasetName.c_str())).layout().dimensions()[1],
                           cuphy::get_HDF5_dataset_info(fInput.open_dataset(datasetName.c_str())).layout().dimensions()[2],
                           nRb, nPort, nRxAnt);
            return;
        }
            
            
    }
    CUDA_CHECK(cudaStreamSynchronize(cuStrm));
}

int csirsRxDynApiDataset::refCheck(bool verbose) {
    int total_err_cnt = 0;
    
    using tensor_pinned_C_32F = typed_tensor<CUPHY_C_32F, pinned_alloc>;
    for (int output_ue_idx = 0; output_ue_idx < csirs_rx_dyn_params.nUes; output_ue_idx++) {
        
        hdf5hpp::hdf5_file fInput = hdf5hpp::hdf5_file::open(CfgFileName[output_ue_idx].c_str());
        
        uint8_t nCsirs = csirs_rx_dyn_params.pDataOut->pChEstBuffInfo[output_ue_idx].nCsirs;
        
        for(int output_csirs_idx = 0; output_csirs_idx < nCsirs; output_csirs_idx++)
        {
            std::string  datasetName = "Csirs_Hest" + std::to_string(output_csirs_idx);
            tensor_pinned_C_32F csirs_rx_chest_ref = typed_tensor_from_dataset<CUPHY_C_32F, pinned_alloc>(fInput.open_dataset(datasetName.c_str()));
    
            typed_tensor<CUPHY_C_32F, pinned_alloc> csirs_rx_chest_output_tensor(ch_est_tensor[output_ue_idx*CUPHY_CSIRS_MAX_NUM_PARAMS + output_csirs_idx].layout());
            
            csirs_rx_chest_output_tensor.convert(ch_est_tensor[output_ue_idx*CUPHY_CSIRS_MAX_NUM_PARAMS + output_csirs_idx], csirs_rx_dyn_params.cuStream);
            CUDA_CHECK(cudaStreamSynchronize(csirs_rx_dyn_params.cuStream));
            NVLOGC_FMT(NVLOG_CSIRSRX, "CSI-RS CHEST dims {} {} {}",
                    csirs_rx_chest_output_tensor.layout().dimensions()[0],
                    csirs_rx_chest_output_tensor.layout().dimensions()[1],
                    csirs_rx_chest_output_tensor.layout().dimensions()[2]);
    
            int checked_res = 0, gpu_mismatch = 0;
            for(int ant_id = 0; ant_id <  csirs_rx_chest_ref.layout().dimensions()[2]; ant_id++)
            {
                for(int port_id = 0; port_id < csirs_rx_chest_ref.layout().dimensions()[1]; port_id++)
                {
                    for(int prb_idx = 0; prb_idx < csirs_rx_chest_ref.layout().dimensions()[0]; prb_idx++)
                    {
                        float2 gpu_res = csirs_rx_chest_output_tensor(prb_idx, port_id, ant_id);
                        float2 ref_res;
                        /* The reference HDF5 dataset should contain all the number of PRBs for this BWP. Any unallocated PRB should be empty. */
                        ref_res.x = (float)csirs_rx_chest_ref(prb_idx, port_id, ant_id).x;
                        ref_res.y = (float)csirs_rx_chest_ref(prb_idx, port_id, ant_id).y;
                        checked_res += 1;
                        if(!complex_approx_equal<float2, float>(gpu_res, ref_res, 0.0001f))
                        {
                            NVLOGC_FMT(NVLOG_CSIRSRX, "Error! CHEST Mismatch for (prb {}, port {}, ant {}):  expected={} + i {} vs. gpu={} + i {}",
                                    prb_idx, port_id, ant_id,
                                   (float) ref_res.x, (float) ref_res.y,
                                   (float) gpu_res.x, (float) gpu_res.y);
                            gpu_mismatch += 1;
                        }
                     }
                }
           }
           if (verbose) {
               if (gpu_mismatch == 0) {
                   NVLOGC_FMT(NVLOG_CSIRSRX, "====> TV {} CSIRS {}: Test PASS", CfgFileName[output_ue_idx].c_str(), output_csirs_idx);
               } else {
                   NVLOGC_FMT(NVLOG_CSIRSRX, "====> TV {} CSIRS {}: Test FAIL. Found {} mismatched symbols", CfgFileName[output_ue_idx].c_str(), output_csirs_idx, gpu_mismatch);
               }
           }
           total_err_cnt += gpu_mismatch;
        }
    }
    return total_err_cnt;
}

// SRS-TX datasets

// construct from h5 file
srsTxStaticApiDataset::srsTxStaticApiDataset(const std::string& inputFileName, int cfg_max_ues_per_slot)
{
    srsTxStatPrms.nMaxSrsUes          = cfg_max_ues_per_slot; // Should be specified. Currently no fallback if 0.
    srsTxTracker.pMemoryFootprint     = nullptr;
    srsTxStatPrms.pOutInfo            = &srsTxTracker;
    srsTxStatPrms.nSlotsPerFrame      = 20;
    srsTxStatPrms.nSymbsPerSlot       = 14;
}


void srsTxStaticApiDataset::print()
{
}

//---------------------------------------------------------------------------------------------------
// Dataset that holds dynamic SRS-TX API paramaters/data

// Construct dataset from HDF5 file
srsTxDynApiDataset::srsTxDynApiDataset(const std::string& inputFileName, uint32_t cfg_max_ues, cudaStream_t cuStrm)
{

    // Output buffer allocation will happen based on the number of max cells.
    max_ues = cfg_max_ues;
    srstx_dyn_params.nSrsUes   = 0;

    ue_srstx_params.reserve(max_ues);

    // Large buffer added as cuPHY tools needs a buffer with a power of 2 size.
    // Allocating once with the assumption that there will be at most max_cells cells
    large_buffer_elements = MAX_N_PRBS_SUPPORTED * CUPHY_N_TONES_PER_PRB * OFDM_SYMBOLS_PER_SLOT * 4;
    large_buffer_bytes    = large_buffer_elements * sizeof(__half2);
    large_buffer          = make_unique_device<__half2>(large_buffer_elements*max_ues);

    // Reset the entire output buffer.
    CUDA_CHECK(cudaMemsetAsync(large_buffer.get(), 0, max_ues*large_buffer_bytes, cuStrm)); // This is done in cuStrm

    output_tensorPrm.resize(max_ues);
    data_tx_tensor.resize(max_ues);

    srstx_dyn_params.pTDataSrsTx = output_tensorPrm.data();

    srstx_dyn_params.cuStream = cuStrm;

    // Update dynamic parameters. This method will also be called to stitch together multiple one-cell TVs.
    cumulativeUpdate(inputFileName, cuStrm);
}

void srsTxDynApiDataset::cumulativeUpdate(const std::string& inputFileName, cudaStream_t cuStrm) {


    int current_ues     = srstx_dyn_params.nSrsUes;

    // Open HDF5 file
    hdf5hpp::hdf5_file fInput = hdf5hpp::hdf5_file::open(inputFileName.c_str());
    CfgFileName.push_back(inputFileName);

    // Every TV corresponds to a separate UE
    srstx_dyn_params.nSrsUes += 1;
    if (srstx_dyn_params.nSrsUes > max_ues) {
        std::string error_msg = "The current number of SRS UEs is " + std::to_string(srstx_dyn_params.nSrsUes) + " but max. UEs was " + std::to_string(max_ues) + "\n";
        throw std::runtime_error(error_msg.c_str());
    }

    ue_srstx_params.resize(srstx_dyn_params.nSrsUes);
    cuphyUeSrsTxPrm_t* current_ue_srstx_dyn_params = &ue_srstx_params[srstx_dyn_params.nSrsUes-1];
    cuphy::read_ue_srstx_dynamic_params_from_file(current_ue_srstx_dyn_params, fInput);

    srstx_dyn_params.pUeSrsTxPrms     = ue_srstx_params.data();

    data_tx_tensor[current_ues] = tensor_device(large_buffer.get() + current_ues * large_buffer_elements, CUPHY_C_16F, 3276, OFDM_SYMBOLS_PER_SLOT, 4, cuphy::tensor_flags::align_tight);

    srstx_dyn_params.pTDataSrsTx[current_ues].desc  = data_tx_tensor[current_ues].desc().handle();
    srstx_dyn_params.pTDataSrsTx[current_ues].pAddr = data_tx_tensor[current_ues].addr();
}

int srsTxDynApiDataset::refCheck(bool verbose) {
    int total_err_cnt = 0;

    using tensor_pinned_C_16F          = typed_tensor<CUPHY_C_16F, pinned_alloc>;
    for (int output_buffer = 0; output_buffer < srstx_dyn_params.nSrsUes; output_buffer++) {
        hdf5hpp::hdf5_file fInput = hdf5hpp::hdf5_file::open(CfgFileName[output_buffer].c_str());
        tensor_pinned_C_16F srs_tx_ref_output = typed_tensor_from_dataset<CUPHY_C_16F, pinned_alloc>(fInput.open_dataset("X_tf_fp16"));


        typed_tensor<CUPHY_C_16F, pinned_alloc> h_srstx_out_tensor(data_tx_tensor[output_buffer].layout());
        h_srstx_out_tensor.convert(data_tx_tensor[output_buffer], srstx_dyn_params.cuStream);
        CUDA_CHECK(cudaStreamSynchronize(srstx_dyn_params.cuStream));
        NVLOGD_FMT(NVLOG_SRSTX, "SRS TX dims {} {} {}",
                h_srstx_out_tensor.layout().dimensions()[0],
                h_srstx_out_tensor.layout().dimensions()[1],
                h_srstx_out_tensor.layout().dimensions()[2]);
        // It's possible that the reference tensor has one layer while the tensor provided has more. Compare only with what's available in the ref. tensor.

        int checked_symbols = 0, gpu_mismatch = 0;
        for(int layer_id = 0; layer_id <  srs_tx_ref_output.layout().dimensions()[2]; layer_id++)
        {
            for(int symbol_id = 0; symbol_id < srs_tx_ref_output.layout().dimensions()[1]; symbol_id++)
            {
                for(int freq_idx = 0; freq_idx < srs_tx_ref_output.layout().dimensions()[0]; freq_idx++)
                {
                    __half2 gpu_symbol = h_srstx_out_tensor(freq_idx, symbol_id, layer_id);
                    __half2 ref_symbol;
                    /* The reference HDF5 dataset should contain all the number of PRBs for this BWP. Any unallocated PRB should be empty. */
                    ref_symbol.x = (half)srs_tx_ref_output(freq_idx, symbol_id, layer_id).x;
                    ref_symbol.y = (half)srs_tx_ref_output(freq_idx, symbol_id, layer_id).y;
                    checked_symbols += 1;
                    if(!complex_approx_equal<__half2, __half>(gpu_symbol, ref_symbol, 0.001f))
                    {
                        NVLOGE_FMT(NVLOG_SRSTX, AERIAL_CUPHY_EVENT, "Error! Mismatch for symbol (freq_bin {}, symbol {}, layer {}):  expected={} + i {} vs. gpu={} + i {}",
                                freq_idx, symbol_id, layer_id,
                               (float) ref_symbol.x, (float) ref_symbol.y,
                               (float) gpu_symbol.x, (float) gpu_symbol.y);
                        gpu_mismatch += 1;
                    }
                 }
            }
       }
       if (verbose) {
           if (gpu_mismatch == 0) {
               NVLOGC_FMT(NVLOG_SRSTX, "====> TV {}: Test PASS OK!!!", CfgFileName[output_buffer].c_str());
           } else {
               NVLOGC_FMT(NVLOG_SRSTX, "====> TV {}: Test FAIL. Found {} mismatched symbols", CfgFileName[output_buffer].c_str(), gpu_mismatch);
           }
       }
       total_err_cnt += gpu_mismatch;
    }
    return total_err_cnt;
}
