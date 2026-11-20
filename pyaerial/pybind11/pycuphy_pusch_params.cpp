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

#include <cstring>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "cuphy.h"
#include "cuphy.hpp"
#include "cuphy_api.h"
#include "cuphy_hdf5.hpp"
#include "fmtlog.h"
#include "pusch_rx.hpp"
#include "pusch_utils.hpp"
#include "pycuphy_util.hpp"
#include "pycuphy_params.hpp"
#include "cuda_array_interface.hpp"
#include "tensor_desc.hpp"

namespace py = pybind11;

namespace pycuphy {


void printPuschStatPrms(const cuphyPuschStatPrms_t& statPrms) {

    int nCells = statPrms.nMaxCells;
    printf("======================================\n");
    printf("PUSCH static parameters:\n");
    printf("======================================\n");

    // Parameters common across all cells.
    printf("enableCfoCorrection:        %4d\n", statPrms.enableCfoCorrection);
    printf("enableWeightedAverageCfo:   %4d\n", statPrms.enableWeightedAverageCfo);
    printf("enableToEstimation:         %4d\n", statPrms.enableToEstimation);
    printf("enablePuschTdi:             %4d\n", statPrms.enablePuschTdi);
    printf("enableDftSOfdm:             %4d\n", statPrms.enableDftSOfdm);
    printf("enableTbSizeCheck:          %4d\n", statPrms.enableTbSizeCheck);
    printf("enableMassiveMIMO:          %4d\n", statPrms.enableMassiveMIMO);
    printf("enableDebugEqOutput:        %4d\n", statPrms.enableDebugEqOutput);
    printf("ldpcEarlyTermination:       %4u\n", statPrms.ldpcEarlyTermination);
    printf("ldpcUseHalf:                %4d\n", statPrms.ldpcUseHalf);
    printf("ldpcAlgoIndex:              %4d\n", statPrms.ldpcAlgoIndex);
    printf("ldpcFlags:                  %4d\n", statPrms.ldpcFlags);
    printf("ldpcKernelLaunch:           %4u\n", statPrms.ldpcKernelLaunch);
    printf("ldpcMaxNumItrAlgo:          %4d\n", statPrms.ldpcMaxNumItrAlgo);
    printf("fixedMaxNumLdpcItrs:        %4d\n", statPrms.fixedMaxNumLdpcItrs);
    printf("ldpcClampValue:             %4f\n", statPrms.ldpcClampValue);
    printf("nMaxLdpcHetConfigs:         %4d\n", statPrms.nMaxLdpcHetConfigs);
    printf("polarDcdrListSz:            %4d\n", statPrms.polarDcdrListSz);
    printf("chEstAlgo:                  %4d\n", statPrms.chEstAlgo);
    printf("enablePerPrgChEst:          %4d\n", statPrms.enablePerPrgChEst);
    printf("eqCoeffAlgo:                %4u\n", statPrms.eqCoeffAlgo);
    printf("enableRssiMeasurement:      %4d\n", statPrms.enableRssiMeasurement);
    printf("enableSinrMeasurement:      %4d\n", statPrms.enableSinrMeasurement);
    printf("enableCsiP2Fapiv3:          %4d\n", statPrms.enableCsiP2Fapiv3);
    printf("stream_priority:            %4d\n", statPrms.stream_priority);
    printf("nMaxCells:                  %4u\n", statPrms.nMaxCells);
    printf("nMaxCellsPerSlot:           %4u\n", statPrms.nMaxCellsPerSlot);
    printf("nMaxTbs:                    %4d\n", statPrms.nMaxTbs);
    printf("nMaxCbsPerTb:               %4d\n", statPrms.nMaxCbsPerTb);
    printf("nMaxTotCbs:                 %4d\n", statPrms.nMaxTotCbs);
    printf("nMaxRx:                     %4u\n", statPrms.nMaxRx);
    printf("enableDeviceGraphLaunch:    %4u\n", statPrms.enableDeviceGraphLaunch);
    printf("enableEarlyHarq:            %4u\n", statPrms.enableEarlyHarq);
    printf("earlyHarqProcNodePriority:  %4u\n", statPrms.earlyHarqProcNodePriority);
    printf("workCancelMode:             %4u\n", statPrms.workCancelMode);
    printf("puschrxChestFactorySettingsFilename:          %s\n", statPrms.puschrxChestFactorySettingsFilename);

    // Cell specific parameters.
    printf("PUSCH static parameters for %d cells\n", nCells);
    cuphyCellStatPrm_t* cellStatPrms = statPrms.pCellStatPrms;
    cuphyPuschStatDbgPrms_t* cellDbgPrms = statPrms.pDbg;
    for (int cellIdx = 0; cellIdx < nCells; cellIdx++) {
        printf("  ------------------------------------\n");
        printf("  Cell %d\n", cellIdx);
        printf("  ------------------------------------\n");
        printf("  phyCellId:      %4d\n", cellStatPrms[cellIdx].phyCellId);
        printf("  nRxAnt:         %4d\n", cellStatPrms[cellIdx].nRxAnt);
        printf("  nRxAntSrs:      %4d\n", cellStatPrms[cellIdx].nRxAntSrs);
        printf("  nTxAnt:         %4d\n", cellStatPrms[cellIdx].nTxAnt);
        printf("  nPrbUlBwp:      %4d\n", cellStatPrms[cellIdx].nPrbUlBwp);
        printf("  nPrbDlBwp:      %4d\n", cellStatPrms[cellIdx].nPrbDlBwp);
        printf("  mu:             %4d\n", cellStatPrms[cellIdx].mu);
    }

    // Debug parameters.
    if(cellDbgPrms) {
        printf("\nDBG:\n");
        printf("pOutFileName:      %s\n", cellDbgPrms->pOutFileName);
        printf("descrmOn:          %d\n", cellDbgPrms->descrmOn);
        printf("enableApiLogging:  %d\n", cellDbgPrms->enableApiLogging);
        printf("forcedNumCsi2Bits: %d\n", cellDbgPrms->forcedNumCsi2Bits);
    }
    printf("======================================\n\n");
}


void printPuschDynPrms(const cuphyPuschDynPrms_t& dynPrms) {
    printf("======================================\n");
    printf("PUSCH dynamic parameters:\n");
    printf("======================================\n");
    printf("phase1Stream:               %ld\n", (uint64_t)dynPrms.phase1Stream);
    printf("phase2Stream:               %ld\n", (uint64_t)dynPrms.phase2Stream);
    printf("setupPhase:                 %d\n", dynPrms.setupPhase);
    printf("procModeBmsk:               %lx\n", dynPrms.procModeBmsk);
    printf("waitTimeOutPreEarlyHarqUs:  %d\n", dynPrms.waitTimeOutPreEarlyHarqUs);
    printf("waitTimeOutPostEarlyHarqUs: %d\n", dynPrms.waitTimeOutPostEarlyHarqUs);

    const cuphyPuschCellGrpDynPrm_t* pCellGrpDynPrm = dynPrms.pCellGrpDynPrm;
    printf("PUSCH cell group dynamic parameters for %d cells\n", pCellGrpDynPrm->nCells);
    for (uint16_t i = 0 ; i < pCellGrpDynPrm->nCells; i++)
    {
        printf("  ------------------------------------\n");
        printf("  Cell %d\n", i);
        printf("  ------------------------------------\n");

        cuphyPuschCellDynPrm_t* pCellDynPrm = &pCellGrpDynPrm->pCellPrms[i];
        printf("  pCellPrms:              %p\n", pCellDynPrm);
        printf("  cellPrmStatIdx:         %d\n", pCellDynPrm->cellPrmStatIdx);
        printf("  cellPrmDynIdx:          %d\n", pCellDynPrm->cellPrmDynIdx);
        printf("  slotNum:                %d\n", pCellDynPrm->slotNum);
    }

    printf("PUSCH cell group dynamic parameters for %d UE groups\n", pCellGrpDynPrm->nUeGrps);

    for (uint16_t i=0; i< pCellGrpDynPrm->nUeGrps; i++)
    {
        printf("  ------------------------------------\n");
        printf("  UE group %d\n", i);
        printf("  ------------------------------------\n");

        const cuphyPuschUeGrpPrm_t* pUeGrpDynPrm = &pCellGrpDynPrm->pUeGrpPrms[i];
        printf("  pUeGrpPrms:             %p\n", pUeGrpDynPrm);
        printf("  pUeGrpDynPrm->pCellPrm: %p\n", pUeGrpDynPrm->pCellPrm);
        printf("  pDmrsDynPrm:            %p\n", pUeGrpDynPrm->pDmrsDynPrm);
        printf("  startPrb:               %d\n", pUeGrpDynPrm->startPrb);
        printf("  nPrb:                   %d\n", pUeGrpDynPrm->nPrb);
        printf("  prgSize:                %d\n", pUeGrpDynPrm->prgSize);
        printf("  nUplinkStreams:         %d\n", pUeGrpDynPrm->nUplinkStreams);
        printf("  puschStartSym:          %d\n", pUeGrpDynPrm->puschStartSym);
        printf("  nPuschSym:              %d\n", pUeGrpDynPrm->nPuschSym);
        printf("  dmrsSymLocBmsk:         %d\n", pUeGrpDynPrm->dmrsSymLocBmsk);
        printf("  rssiSymLocBmsk:         %d\n", pUeGrpDynPrm->rssiSymLocBmsk);
        printf("  nUes:                   %d\n", pUeGrpDynPrm->nUes);

        for (uint16_t j = 0; j < pUeGrpDynPrm->nUes; j++) {
            printf("  pUePrmIdxs:             %d\n", pUeGrpDynPrm->pUePrmIdxs[j]);
        }

        const cuphyPuschDmrsPrm_t* pDmrsDynPrm = pUeGrpDynPrm->pDmrsDynPrm;
        if(pDmrsDynPrm != nullptr) {
            printf("  dmrsAddlnPos:           %d\n", pDmrsDynPrm->dmrsAddlnPos);
            printf("  dmrsMaxLen:             %d\n", pDmrsDynPrm->dmrsMaxLen);
            printf("  nDmrsCdmGrpsNoData:     %d\n", pDmrsDynPrm->nDmrsCdmGrpsNoData);
            printf("  dmrsScrmId:             %d\n", pDmrsDynPrm->dmrsScrmId);
        }
    }

    // Print information for all UEs
    int numUes = pCellGrpDynPrm->nUes;
    printf("PUSCH cell group dynamic parameters for %d UEs\n", numUes);

    for (uint16_t i = 0; i < pCellGrpDynPrm->nUes; i++)
    {
        printf("  ------------------------------------\n");
        printf("  UE %d\n", i);
        printf("  ------------------------------------\n");

        const cuphyPuschUePrm_t* pUeDynPrm = &pCellGrpDynPrm->pUePrms[i];
        printf("  pUePrms:                %p\n", pUeDynPrm);
        printf("  pUePrms->pUeGrpPrm:     %p\n", pUeDynPrm->pUeGrpPrm);
        printf("  pduBitmap:              %x\n", pUeDynPrm->pduBitmap);
        printf("  ueGrpIdx:               %d\n", pUeDynPrm->ueGrpIdx);
        printf("  enableTfPrcd:           %d\n", pUeDynPrm->enableTfPrcd);
        printf("  puschIdentity:          %d\n", pUeDynPrm->puschIdentity);
        printf("  groupOrSequenceHopping: %d\n", pUeDynPrm->groupOrSequenceHopping);
        printf("  N_symb_slot:            %d\n", pUeDynPrm->N_symb_slot);
        printf("  N_slot_frame:           %d\n", pUeDynPrm->N_slot_frame);
        printf("  lowPaprGroupNumber:     %d\n", pUeDynPrm->lowPaprGroupNumber);
        printf("  lowPaprSequenceNumber:  %d\n", pUeDynPrm->lowPaprSequenceNumber);
        printf("  scid:                   %d\n", pUeDynPrm->scid);
        printf("  dmrsPortBmsk:           %d\n", pUeDynPrm->dmrsPortBmsk);
        printf("  mcsTableIndex:          %d\n", pUeDynPrm->mcsTableIndex);
        printf("  mcsIndex:               %d\n", pUeDynPrm->mcsIndex);
        printf("  targetCodeRate:         %d\n", pUeDynPrm->targetCodeRate);
        printf("  qamModOrder:            %d\n", pUeDynPrm->qamModOrder);
        printf("  TBSize:                 %d\n", pUeDynPrm->TBSize);
        printf("  rv:                     %d\n", pUeDynPrm->rv);
        printf("  rnti:                   %d\n", pUeDynPrm->rnti);
        printf("  dataScramId:            %d\n", pUeDynPrm->dataScramId);
        printf("  nUeLayers:              %d\n", pUeDynPrm->nUeLayers);
        printf("  ndi:                    %d\n", pUeDynPrm->ndi);
        printf("  harqProcessId:          %d\n", pUeDynPrm->harqProcessId);
        printf("  i_lbrm:                 %d\n", pUeDynPrm->i_lbrm);
        printf("  maxLayers:              %d\n", pUeDynPrm->maxLayers);
        printf("  maxQm:                  %d\n", pUeDynPrm->maxQm);
        printf("  n_PRB_LBRM:             %d\n", pUeDynPrm->n_PRB_LBRM);

        if(pUeDynPrm->pUciPrms != nullptr) {
            const cuphyUciOnPuschPrm_t* pUci = pUeDynPrm->pUciPrms;
            printf("  pUciPrms:              %p\n", pUci);
            printf("  nBitsHarq:             %d\n", pUci->nBitsHarq);
            printf("  nBitsCsi1:             %d\n", pUci->nBitsCsi1);
            printf("  alphaScaling:          %d\n", pUci->alphaScaling);
            printf("  betaOffsetHarqAck:     %d\n", pUci->betaOffsetHarqAck);
            printf("  betaOffsetCsi1:        %d\n", pUci->betaOffsetCsi1);
            printf("  betaOffsetCsi2:        %d\n", pUci->betaOffsetCsi2);
            printf("  rankBitOffset:         %d\n", pUci->rankBitOffset);
            printf("  nRanksBits:            %d\n", pUci->nRanksBits);
            printf("  nCsiReports:           %d\n", pUci->nCsiReports);
        }
    }
    printf("======================================\n\n");
}

void printPerTbParams(const char *desciption, const uint16_t nSchUes, const PerTbParams* tbPrms)
{
    printf("============= %s ============\n",desciption);
    printf("PUSCH PerTbParams:\n");
    printf("======================================\n");
    for(uint16_t ueIdx = 0; ueIdx < nSchUes ; ueIdx++)
    {
        printf("===============UEIdx %u =======================\n",ueIdx);
        printf("ndi                     = %u\n",tbPrms[ueIdx].ndi);
        printf("rv                      = %u\n",tbPrms[ueIdx].rv);
        printf("Qm                      = %u\n",tbPrms[ueIdx].Qm);
        printf("bg                      = %u\n",tbPrms[ueIdx].bg);
        printf("Nl                      = %u\n",tbPrms[ueIdx].Nl);
        printf("num_CBs                 = %u\n",tbPrms[ueIdx].num_CBs);
        printf("Zc                      = %u\n",tbPrms[ueIdx].Zc);
        printf("N                       = %u\n",tbPrms[ueIdx].N);
        printf("Ncb                     = %u\n",tbPrms[ueIdx].Ncb);
        printf("Ncb_padded              = %u\n",tbPrms[ueIdx].Ncb_padded);
        printf("G                       = %u\n",tbPrms[ueIdx].G);
        printf("K                       = %u\n",tbPrms[ueIdx].K);
        printf("F                       = %u\n",tbPrms[ueIdx].F);
        printf("cinit                   = %u\n",tbPrms[ueIdx].cinit);
        printf("nDataBytes              = %u\n",tbPrms[ueIdx].nDataBytes);
        printf("nZpBitsPerCb            = %u\n",tbPrms[ueIdx].nZpBitsPerCb);
        printf("firstCodeBlockIndex     = %u\n",tbPrms[ueIdx].firstCodeBlockIndex);
        printf("encodedSize             = %u\n",tbPrms[ueIdx].encodedSize);
        printf("layer_map_array         = {%u %u %u %u %u %u %u %u}\n",
            tbPrms[ueIdx].layer_map_array[0],tbPrms[ueIdx].layer_map_array[1],
            tbPrms[ueIdx].layer_map_array[2],tbPrms[ueIdx].layer_map_array[3],
            tbPrms[ueIdx].layer_map_array[4],tbPrms[ueIdx].layer_map_array[5],
            tbPrms[ueIdx].layer_map_array[6],tbPrms[ueIdx].layer_map_array[7]);
        printf("userGroupIndex          = %u\n",tbPrms[ueIdx].userGroupIndex);
        printf("nBBULayers              = %u\n",tbPrms[ueIdx].nBBULayers);
        printf("startLLR                = %u\n",tbPrms[ueIdx].startLLR);
        printf("nDmrsCdmGrpsNoData      = %u\n",(uint32_t)tbPrms[ueIdx].nDmrsCdmGrpsNoData);
        printf("======================================\n");
    }
}

void printPuschDrvdUeGrpPrms(const cuphyPuschRxUeGrpPrms_t& puschRxUeGrpPrms) {
    printf("======================================\n");
    printf("PUSCH derived UE group parameters:\n");
    printf("======================================\n");

    // Common parameters
    printf("statCellIdx:                %d\n", puschRxUeGrpPrms.statCellIdx);
    printf("nRxAnt:                     %d\n", puschRxUeGrpPrms.nRxAnt);
    printf("nLayers:                    %d\n", puschRxUeGrpPrms.nLayers);
    printf("slotNum:                    %d\n", puschRxUeGrpPrms.slotNum);

    // DFT-s-OFDM parameters
    printf("enableTfPrcd:               %d\n", puschRxUeGrpPrms.enableTfPrcd);
    printf("optionalDftSOfdm:           %d\n", puschRxUeGrpPrms.optionalDftSOfdm);
    printf("puschIdentity:              %d\n", puschRxUeGrpPrms.puschIdentity);
    printf("groupOrSequenceHopping:     %d\n", puschRxUeGrpPrms.groupOrSequenceHopping);
    printf("N_symb_slot:                %d\n", puschRxUeGrpPrms.N_symb_slot);
    printf("N_slot_frame:               %d\n", puschRxUeGrpPrms.N_slot_frame);
    printf("lowPaprGroupNumber:         %d\n", puschRxUeGrpPrms.lowPaprGroupNumber);
    printf("lowPaprSequenceNumber:      %d\n", puschRxUeGrpPrms.lowPaprSequenceNumber);

    // Resource allocation
    printf("startPrb:                   %d\n", puschRxUeGrpPrms.startPrb);
    printf("nPrb:                       %d\n", puschRxUeGrpPrms.nPrb);
    printf("mu:                         %d\n", puschRxUeGrpPrms.mu);

    printf("puschStartSym:              %d\n", puschRxUeGrpPrms.puschStartSym);
    printf("nPuschSym:                  %d\n", puschRxUeGrpPrms.nPuschSym);

    printf("nUes:                       %d\n", puschRxUeGrpPrms.nUes);

    // DMRS parameters
    printf("nDmrsSyms:                  %d\n", puschRxUeGrpPrms.nDmrsSyms);
    printf("dmrsSymLocBmsk:             %d\n", puschRxUeGrpPrms.dmrsSymLocBmsk);
    printf("dmrsScrmId:                 %d\n", puschRxUeGrpPrms.dmrsScrmId);
    printf("dmrsMaxLen:                 %d\n", puschRxUeGrpPrms.dmrsMaxLen);
    printf("dmrsAddlnPos:               %d\n", puschRxUeGrpPrms.dmrsAddlnPos);
    printf("dmrsSymLoc:                 ");
    for(int i = 0; i < N_MAX_DMRS_SYMS; i++) {
        if(i != 0) {
            printf(", ");
        }
        printf("%d", puschRxUeGrpPrms.dmrsSymLoc[i]);
    }
    printf("\n");
    printf("dmrsCnt:                    %d\n", puschRxUeGrpPrms.dmrsCnt);
    printf("scid:                       %d\n", puschRxUeGrpPrms.scid);
    printf("nDmrsCdmGrpsNoData:         %d\n", puschRxUeGrpPrms.nDmrsCdmGrpsNoData);
    printf("nDmrsGridsPerPrb:           %d\n", puschRxUeGrpPrms.nDmrsGridsPerPrb);
    printf("activeDMRSGridBmsk:         %d\n", puschRxUeGrpPrms.activeDMRSGridBmsk);
    printf("activeTOCCBmsk:             %d, %d\n", puschRxUeGrpPrms.activeTOCCBmsk[0], puschRxUeGrpPrms.activeTOCCBmsk[1]);
    printf("activeFOCCBmsk:             %d, %d\n", puschRxUeGrpPrms.activeFOCCBmsk[0], puschRxUeGrpPrms.activeFOCCBmsk[1]);
    printf("OCCIdx:                     ");
    for(int i = 0; i < MAX_N_LAYERS_PUSCH; i++) {
        if(i != 0) {
            printf(", ");
        }
        printf("%d", puschRxUeGrpPrms.OCCIdx[i]);
    }
    printf("\n");
    printf("dmrsPortIdxs:               ");
    for(int i = 0; i < MAX_N_LAYERS_PUSCH; i++) {
        if(i != 0) {
            printf(", ");
        }
        printf("%d", puschRxUeGrpPrms.dmrsPortIdxs[i]);
    }
    printf("\n");
    printf("nTimeChEsts:                %d\n", puschRxUeGrpPrms.nTimeChEsts);
    printf("invNoiseVarLin:             %f\n", puschRxUeGrpPrms.invNoiseVarLin);
    printf("nDataSym:                   %d\n", puschRxUeGrpPrms.nDataSym);
    printf("enableCfoCorrection:        %d\n", puschRxUeGrpPrms.enableCfoCorrection);
    printf("enableWeightedAverageCfo:   %d\n", puschRxUeGrpPrms.enableWeightedAverageCfo);
    printf("enableToEstimation:         %d\n", puschRxUeGrpPrms.enableToEstimation);
    printf("enablePuschTdi:             %d\n", puschRxUeGrpPrms.enablePuschTdi);
    printf("qam:                        [");
    for(int i = 0; i < CUPHY_PUSCH_RX_MAX_N_LAYERS_PER_UE_GROUP; i++) {
        if(i != 0) {
            printf(", ");
        }
        printf("%d", puschRxUeGrpPrms.qam[i]);
    }
    printf("]\n");
    printf("======================================\n\n");
}


PuschParams::PuschParams():
m_LinearAlloc(getBufferSize()) {

    // Zero-initialize all structures to prevent uninitialized memory issues
    memset(&m_puschStatPrms, 0, sizeof(m_puschStatPrms));
    memset(&m_puschStatDbgPrms, 0, sizeof(m_puschStatDbgPrms));
    memset(&m_tracker, 0, sizeof(m_tracker));
    memset(&m_puschDynPrms, 0, sizeof(m_puschDynPrms));

    // Allocate CPU and GPU memory for the PUSCH Rx UE group parameters.
    m_descrSizeBytes = sizeof(cuphyPuschRxUeGrpPrms_t) * MAX_N_USER_GROUPS_SUPPORTED;
    size_t descrAlignBytes = alignof(cuphyPuschRxUeGrpPrms_t);
    m_descrSizeBytes = ((m_descrSizeBytes + (descrAlignBytes - 1)) / descrAlignBytes) * descrAlignBytes;

    m_drvdUeGrpPrmsCpuBuf = cuphy::buffer<uint8_t, cuphy::pinned_alloc>(m_descrSizeBytes);
    m_drvdUeGrpPrmsGpuBuf = cuphy::buffer<uint8_t, cuphy::device_alloc>(m_descrSizeBytes);
    m_drvdUeGrpPrmsCpu = (cuphyPuschRxUeGrpPrms_t*)m_drvdUeGrpPrmsCpuBuf.addr();
    m_drvdUeGrpPrmsGpu = (cuphyPuschRxUeGrpPrms_t*)m_drvdUeGrpPrmsGpuBuf.addr();

    m_tbPrmsCpu = cuphy::buffer<PerTbParams, cuphy::pinned_alloc>(MAX_N_TBS_SUPPORTED);
    m_tbPrmsGpu = cuphy::buffer<PerTbParams, cuphy::device_alloc>(MAX_N_TBS_SUPPORTED);

    // Allocate host pinned buffer for early exit and memset it to 0 (i.e., no early exit)
    m_workCancelBuffer = cuphy::buffer<uint8_t, cuphy::pinned_alloc>(sizeof(uint8_t));
    memset(m_workCancelBuffer.addr(), 0, sizeof(uint8_t));

}


size_t PuschParams::getBufferSize() const {

    size_t nBytesBuffer = 0;

    // HARQ buffer size. FIXME: Figure out the "correct" number here. The current number should be high enough though.
    const uint32_t maxNumRmLLRsPerCb = 26112;
    nBytesBuffer += maxNumRmLLRsPerCb * MAX_N_CBS_PER_TB_SUPPORTED + LINEAR_ALLOC_PAD_BYTES;

    // Allocate memory for Rx data. One cell.
    nBytesBuffer += MAX_N_PRBS_SUPPORTED * CUPHY_N_TONES_PER_PRB * OFDM_SYMBOLS_PER_SLOT * MAX_N_ANTENNAS_SUPPORTED + LINEAR_ALLOC_PAD_BYTES;

    return nBytesBuffer;
}


void PuschParams::setFilters(const py::array& WFreq,
                             const py::array& WFreq4,
                             const py::array& WFreqSmall,
                             const py::array& shiftSeq,
                             const py::array& shiftSeq4,
                             const py::array& unShiftSeq,
                             const py::array& unShiftSeq4,
                             uint64_t cuStream) {
    m_tWFreq      = deviceFromNumpy<float, py::array::f_style | py::array::forcecast>(WFreq, CUPHY_R_32F, CUPHY_R_32F, cuphy::tensor_flags::align_tight, (cudaStream_t)cuStream);
    m_tWFreq4     = deviceFromNumpy<float, py::array::f_style | py::array::forcecast>(WFreq4, CUPHY_R_32F, CUPHY_R_32F, cuphy::tensor_flags::align_tight, (cudaStream_t)cuStream);
    m_tWFreqSmall = deviceFromNumpy<float, py::array::f_style | py::array::forcecast>(WFreqSmall, CUPHY_R_32F, CUPHY_R_32F, cuphy::tensor_flags::align_tight, (cudaStream_t)cuStream);

    m_tPrmWFreq.desc = m_tWFreq.desc().handle();
    m_tPrmWFreq.pAddr = m_tWFreq.addr();
    m_puschStatPrms.pWFreq = &m_tPrmWFreq;

    m_tPrmWFreq4.desc = m_tWFreq4.desc().handle();
    m_tPrmWFreq4.pAddr = m_tWFreq4.addr();
    m_puschStatPrms.pWFreq4 = &m_tPrmWFreq4;

    m_tPrmWFreqSmall.desc = m_tWFreqSmall.desc().handle();
    m_tPrmWFreqSmall.pAddr = m_tWFreqSmall.addr();
    m_puschStatPrms.pWFreqSmall = &m_tPrmWFreqSmall;

    m_tShiftSeq    = deviceFromNumpy<std::complex<float>, py::array::f_style | py::array::forcecast>(shiftSeq, CUPHY_C_32F, CUPHY_C_16F, cuphy::tensor_flags::align_tight, (cudaStream_t)cuStream);
    m_tUnshiftSeq  = deviceFromNumpy<std::complex<float>, py::array::f_style | py::array::forcecast>(unShiftSeq, CUPHY_C_32F, CUPHY_C_16F, cuphy::tensor_flags::align_tight, (cudaStream_t)cuStream);
    m_tShiftSeq4   = deviceFromNumpy<std::complex<float>, py::array::f_style | py::array::forcecast>(shiftSeq4, CUPHY_C_32F, CUPHY_C_16F, cuphy::tensor_flags::align_tight, (cudaStream_t)cuStream);
    m_tUnshiftSeq4 = deviceFromNumpy<std::complex<float>, py::array::f_style | py::array::forcecast>(unShiftSeq4, CUPHY_C_32F, CUPHY_C_16F, cuphy::tensor_flags::align_tight, (cudaStream_t)cuStream);

    m_tPrmShiftSeq.desc = m_tShiftSeq.desc().handle();
    m_tPrmShiftSeq.pAddr = m_tShiftSeq.addr();
    m_puschStatPrms.pShiftSeq = &m_tPrmShiftSeq;

    m_tPrmUnShiftSeq.desc = m_tUnshiftSeq.desc().handle();
    m_tPrmUnShiftSeq.pAddr = m_tUnshiftSeq.addr();
    m_puschStatPrms.pUnShiftSeq = &m_tPrmUnShiftSeq;

    m_tPrmShiftSeq4.desc = m_tShiftSeq4.desc().handle();
    m_tPrmShiftSeq4.pAddr = m_tShiftSeq4.addr();
    m_puschStatPrms.pShiftSeq4 = &m_tPrmShiftSeq4;

    m_tPrmUnShiftSeq4.desc = m_tUnshiftSeq4.desc().handle();
    m_tPrmUnShiftSeq4.pAddr = m_tUnshiftSeq4.addr();
    m_puschStatPrms.pUnShiftSeq4 = &m_tPrmUnShiftSeq4;
}


void PuschParams::setStatPrms(const py::object& statPrms) {
    // Zero-initialize the entire structure to prevent uninitialized memory bugs
    memset(&m_puschStatPrms, 0, sizeof(m_puschStatPrms));

    const py::list cellStatPrmList = statPrms.attr("cellStatPrms");
    uint16_t nCells = cellStatPrmList.size();
    m_cellStatPrms.resize(nCells);
    m_puschCellStatPrms.resize(nCells);

    // TODO: Support this properly.
    m_tracker.pMemoryFootprint                = nullptr;
    m_puschStatPrms.pOutInfo                  = &m_tracker;
    m_puschStatPrms.pWorkCancelInfo           = static_cast<uint8_t*>(m_workCancelBuffer.addr());
    m_puschStatPrms.pPuschRkhsPrms            = nullptr;  // TODO: Support RKHS.
    m_puschStatPrms.enableCfoCorrection       = statPrms.attr("enableCfoCorrection").cast<uint8_t>();
    m_puschStatPrms.enableWeightedAverageCfo  = statPrms.attr("enableWeightedAverageCfo").cast<uint8_t>();
    m_puschStatPrms.enableToEstimation        = statPrms.attr("enableToEstimation").cast<uint8_t>();
    m_puschStatPrms.enablePuschTdi            = statPrms.attr("enablePuschTdi").cast<uint8_t>();
    m_puschStatPrms.enableDftSOfdm            = statPrms.attr("enableDftSOfdm").cast<uint8_t>();
    m_puschStatPrms.enableTbSizeCheck         = statPrms.attr("enableTbSizeCheck").cast<uint8_t>();
    m_puschStatPrms.enableMassiveMIMO         = statPrms.attr("enableUlRxBf").cast<uint8_t>(); //for PUSCH enableMassiveMIMO = enableUlRxBf
    m_puschStatPrms.enableDebugEqOutput       = statPrms.attr("enableDebugEqOutput").cast<uint8_t>();
    m_puschStatPrms.ldpcEarlyTermination      = statPrms.attr("ldpcEarlyTermination").cast<uint8_t>();
    m_puschStatPrms.ldpcUseHalf               = statPrms.attr("ldpcUseHalf").cast<uint8_t>();
    m_puschStatPrms.ldpcAlgoIndex             = statPrms.attr("ldpcAlgoIndex").cast<uint8_t>();
    m_puschStatPrms.ldpcFlags                 = statPrms.attr("ldpcFlags").cast<uint32_t>();
    m_puschStatPrms.ldpcKernelLaunch          = statPrms.attr("ldpcKernelLaunch").cast<cuphyPuschLdpcKernelLaunch_t>();
    m_puschStatPrms.ldpcMaxNumItrAlgo         = statPrms.attr("ldpcMaxNumItrAlgo").cast<cuphyLdpcMaxItrAlgoType_t>();
    m_puschStatPrms.fixedMaxNumLdpcItrs       = statPrms.attr("fixedMaxNumLdpcItrs").cast<uint8_t>();
    m_puschStatPrms.ldpcClampValue            = statPrms.attr("ldpcClampValue").cast<float>();
    m_puschStatPrms.polarDcdrListSz           = statPrms.attr("polarDcdrListSz").cast<uint8_t>();
    m_puschStatPrms.nMaxTbPerNode             = statPrms.attr("nMaxTbPerNode").cast<uint8_t>();
    m_puschStatPrms.chEstAlgo                 = statPrms.attr("chEstAlgo").cast<cuphyPuschChEstAlgoType_t>();
    m_puschStatPrms.enablePerPrgChEst         = statPrms.attr("enablePerPrgChEst").cast<uint8_t>();
    m_puschStatPrms.eqCoeffAlgo               = statPrms.attr("eqCoeffAlgo").cast<cuphyPuschEqCoefAlgoType_t>();
    m_puschStatPrms.enableRssiMeasurement     = statPrms.attr("enableRssiMeasurement").cast<uint8_t>();
    m_puschStatPrms.enableSinrMeasurement     = statPrms.attr("enableSinrMeasurement").cast<uint8_t>();
    m_puschStatPrms.enableCsiP2Fapiv3         = statPrms.attr("enableCsiP2Fapiv3").cast<uint8_t>();
    m_puschStatPrms.stream_priority           = statPrms.attr("stream_priority").cast<int>();
    m_puschStatPrms.nMaxCells                 = statPrms.attr("nMaxCells").cast<uint16_t>();
    m_puschStatPrms.nMaxCellsPerSlot          = statPrms.attr("nMaxCellsPerSlot").cast<uint16_t>();
    m_puschStatPrms.pCellStatPrms             = m_cellStatPrms.data();
    m_puschStatPrms.nMaxTbs                   = statPrms.attr("nMaxTbs").cast<uint32_t>();
    m_puschStatPrms.nMaxCbsPerTb              = statPrms.attr("nMaxCbsPerTb").cast<uint32_t>();
    m_puschStatPrms.nMaxTotCbs                = statPrms.attr("nMaxTotCbs").cast<uint32_t>();
    m_puschStatPrms.nMaxRx                    = statPrms.attr("nMaxRx").cast<uint32_t>();
    m_puschStatPrms.nMaxPrb                   = statPrms.attr("nMaxPrb").cast<uint32_t>();
    m_puschStatPrms.nMaxLdpcHetConfigs        = statPrms.attr("nMaxLdpcHetConfigs").cast<uint8_t>();
    m_puschStatPrms.enableDeviceGraphLaunch   = statPrms.attr("enableDeviceGraphLaunch").cast<uint8_t>();
    m_puschStatPrms.enableEarlyHarq           = statPrms.attr("enableEarlyHarq").cast<uint8_t>();
    m_puschStatPrms.earlyHarqProcNodePriority = statPrms.attr("earlyHarqProcNodePriority").cast<int32_t>();
    m_puschStatPrms.workCancelMode            = statPrms.attr("workCancelMode").cast<cuphyPuschWorkCancelMode_t>();
    m_puschStatPrms.enableBatchedMemcpy       = statPrms.attr("enableBatchedMemcpy").cast<uint8_t>();
    m_puschStatPrms.openRanFunctionalSplitOption = PUSCH_7_2_A;
    m_puschStatPrms.kernelSelOption              = PUSCH_ALL;
    m_puschStatPrms.uciKernelSelOption           = PUSCH_UCI_ALL;
    m_puschStatPrms.delayUs                      = 0;
    m_puschStatPrms.subSlotDelayUs               = 0;
    
    if (py::object attr = py::getattr(statPrms, "chestFactorySettingsFilename"); !attr.is_none())
    {
        m_puschrxChestFactorySettingsFilename = statPrms.attr("chestFactorySettingsFilename").cast<
            std::string>();
        m_puschStatPrms.puschrxChestFactorySettingsFilename = m_puschrxChestFactorySettingsFilename.c_str();
    }

    // For now, it is assumed all Rx symbols are ready in the pycuphy test, hence symStats initialized with 1
    std::vector<uint32_t> symStats(OFDM_SYMBOLS_PER_SLOT, SYM_RX_DONE);
    m_bSymRxStatus = std::move(cuphy::buffer<uint32_t, cuphy::device_alloc>(symStats));
    m_puschStatPrms.pSymRxStatus = static_cast<uint32_t const*>(m_bSymRxStatus.addr());
    CUDA_CHECK_EXCEPTION_PRINTF_VERSION(cudaEventCreate(&m_subSlotCompletedEvent));
    CUDA_CHECK_EXCEPTION_PRINTF_VERSION(cudaEventCreate(&m_waitCompletedSubSlotEvent));
    CUDA_CHECK_EXCEPTION_PRINTF_VERSION(cudaEventCreate(&m_waitCompletedFullSlotEvent));
    CUDA_CHECK_EXCEPTION_PRINTF_VERSION(cudaEventCreate(&m_uciOnPuschCompletedEvent));
    m_puschStatPrms.subSlotCompletedEvent = m_subSlotCompletedEvent;
    m_puschStatPrms.waitCompletedSubSlotEvent = m_waitCompletedSubSlotEvent;
    m_puschStatPrms.waitCompletedFullSlotEvent = m_waitCompletedFullSlotEvent;
    m_puschStatPrms.uciOnPuschCompletedEvent = m_uciOnPuschCompletedEvent;

    // Channel estimation filters.
    py::array WFreq       = statPrms.attr("WFreq");
    py::array WFreq4      = statPrms.attr("WFreq4");
    py::array WFreqSmall  = statPrms.attr("WFreqSmall");
    py::array shiftSeq    = statPrms.attr("ShiftSeq");
    py::array unshiftSeq  = statPrms.attr("UnShiftSeq");
    py::array shiftSeq4   = statPrms.attr("ShiftSeq4");
    py::array unshiftSeq4 = statPrms.attr("UnShiftSeq4");
    setFilters(WFreq, WFreq4, WFreqSmall, shiftSeq, shiftSeq4, unshiftSeq, unshiftSeq4);

    // Cell static parameters.
    for (int cellIdx = 0; cellIdx < nCells; cellIdx ++ ) {
        // Zero-initialize cell structures to prevent uninitialized memory
        memset(&m_cellStatPrms[cellIdx], 0, sizeof(m_cellStatPrms[cellIdx]));
        memset(&m_puschCellStatPrms[cellIdx], 0, sizeof(m_puschCellStatPrms[cellIdx]));

        const py::object cellStatPrms = cellStatPrmList[cellIdx];

        m_cellStatPrms[cellIdx].phyCellId = cellStatPrms.attr("phyCellId").cast<uint16_t>();
        m_cellStatPrms[cellIdx].nRxAnt    = cellStatPrms.attr("nRxAnt").cast<uint16_t>();
        m_cellStatPrms[cellIdx].nRxAntSrs = cellStatPrms.attr("nRxAntSrs").cast<uint16_t>();
        m_cellStatPrms[cellIdx].nTxAnt    = cellStatPrms.attr("nTxAnt").cast<uint16_t>();
        m_cellStatPrms[cellIdx].nPrbUlBwp = cellStatPrms.attr("nPrbUlBwp").cast<uint16_t>();
        m_cellStatPrms[cellIdx].nPrbDlBwp = cellStatPrms.attr("nPrbDlBwp").cast<uint16_t>();
        m_cellStatPrms[cellIdx].mu        = cellStatPrms.attr("mu").cast<uint8_t>();
        if (m_cellStatPrms[cellIdx].mu > 1) {
            throw std::runtime_error("Unsupported numerology value!");
        }

        m_cellStatPrms[cellIdx].pPuschCellStatPrms = &m_puschCellStatPrms[cellIdx];

        // TODO: UCI is not supported
        try {
            const py::object puschCellStatPrms = cellStatPrms.attr("puschCellStatPrms");

            m_cellStatPrms[cellIdx].pPuschCellStatPrms->nCsirsPorts      = puschCellStatPrms.attr("nCsirsPorts").cast<uint8_t>();
            m_cellStatPrms[cellIdx].pPuschCellStatPrms->N1               = puschCellStatPrms.attr("N1").cast<uint8_t>();
            m_cellStatPrms[cellIdx].pPuschCellStatPrms->N2               = puschCellStatPrms.attr("N2").cast<uint8_t>();
            m_cellStatPrms[cellIdx].pPuschCellStatPrms->csiReportingBand = puschCellStatPrms.attr("csiReportingBand").cast<uint8_t>();
            m_cellStatPrms[cellIdx].pPuschCellStatPrms->codebookType     = puschCellStatPrms.attr("codebookType").cast<uint8_t>();
            m_cellStatPrms[cellIdx].pPuschCellStatPrms->codebookMode     = puschCellStatPrms.attr("codebookMode").cast<uint8_t>();
            m_cellStatPrms[cellIdx].pPuschCellStatPrms->isCqi            = puschCellStatPrms.attr("isCqi").cast<uint8_t>();
            m_cellStatPrms[cellIdx].pPuschCellStatPrms->isLi             = puschCellStatPrms.attr("isLi").cast<uint8_t>();
        }
        catch(...) {
            m_cellStatPrms[cellIdx].pPuschCellStatPrms->nCsirsPorts      = 4;
            m_cellStatPrms[cellIdx].pPuschCellStatPrms->N1               = 2;
            m_cellStatPrms[cellIdx].pPuschCellStatPrms->N2               = 1;
            m_cellStatPrms[cellIdx].pPuschCellStatPrms->csiReportingBand = 0;
            m_cellStatPrms[cellIdx].pPuschCellStatPrms->codebookType     = 0;
            m_cellStatPrms[cellIdx].pPuschCellStatPrms->codebookMode     = 1;
            m_cellStatPrms[cellIdx].pPuschCellStatPrms->isCqi            = 0;
            m_cellStatPrms[cellIdx].pPuschCellStatPrms->isLi             = 0;
            m_cellStatPrms[cellIdx].pPucchCellStatPrms = nullptr;
        }
    }

    // Next, debug parameters.
    // cuPHY library requires pDbg to always be non-null, so always provide a valid structure
    // Zero-initialize the structure to prevent random garbage values causing crashes
    memset(&m_puschStatDbgPrms, 0, sizeof(m_puschStatDbgPrms));
    m_puschStatPrms.pDbg = &m_puschStatDbgPrms;

    py::object dbg = statPrms.attr("dbg");
    if (std::string(py::str(dbg)) == "None") {
        // Set default values when debug object is None
        m_puschStatPrms.pDbg->descrmOn = 1;
        m_puschStatPrms.pDbg->enableApiLogging = 0;
        m_puschStatPrms.pDbg->forcedNumCsi2Bits = 0;
        m_puschStatPrms.pDbg->pOutFileName = nullptr;
    }
    else {
        m_puschStatPrms.pDbg->descrmOn = dbg.attr("descrmOn").cast<uint8_t>();
        m_puschStatPrms.pDbg->enableApiLogging = dbg.attr("enableApiLogging").cast<uint8_t>();
        m_puschStatPrms.pDbg->forcedNumCsi2Bits = dbg.attr("forcedNumCsi2Bits").cast<uint16_t>();
        m_debugFilename = std::string(py::str(dbg.attr("outFileName")));
        if (m_debugFilename == "None") {
            m_puschStatPrms.pDbg->pOutFileName = nullptr;
        }
        else {
            m_puschStatPrms.pDbg->pOutFileName = m_debugFilename.c_str();
        }
    }
}


void PuschParams::setStatPrms(const cuphyPuschStatPrms_t& statPrms) {
    m_puschStatPrms = statPrms;
    m_ldpcParams = std::move(cuphyLDPCParams(&statPrms));
}


void PuschParams::setDynPrms(const cuphyPuschDynPrms_t& dynPrms) {
    m_puschDynPrms = dynPrms;
    if (m_puschDynPrms.setupPhase == PUSCH_SETUP_PHASE_1) {
        m_nMaxPrb = updatePuschRxUeGrpPrms(m_puschDynPrms.phase1Stream);
    }

    // Set Rx data tensors.
    int nCells = m_puschDynPrms.pCellGrpDynPrm->nCells;
    m_tDataRx.resize(nCells);
    m_tPrmDataRx = std::vector<cuphyTensorPrm_t>(m_puschDynPrms.pDataIn->pTDataRx, m_puschDynPrms.pDataIn->pTDataRx + nCells);
    for(int cellIdx = 0; cellIdx < nCells; cellIdx++) {
        cuphyTensorPrm_t* tPrm = &m_puschDynPrms.pDataIn->pTDataRx[cellIdx];
        cuphyTensorDescriptor_t desc = tPrm->desc;
        void* addr = tPrm->pAddr;

        cuphyDataType_t dtype;
        int rank;
        cuphy::vec<int, CUPHY_DIM_MAX> dimensions;
        cuphy::vec<int, CUPHY_DIM_MAX> strides;
        cuphyStatus_t s = cuphyGetTensorDescriptor(desc,
                                                   CUPHY_DIM_MAX,
                                                   &dtype,
                                                   &rank,
                                                   dimensions.begin(),
                                                   strides.begin());
        if(CUPHY_STATUS_SUCCESS != s) {
            throw cuphy::cuphy_fn_exception(s, "cuphyGetTensorDescriptor()");
        }
        cuphy::tensor_info tInfo = cuphy::tensor_info(dtype, cuphy::tensor_layout(rank, dimensions.begin(), strides.begin()));
        m_tDataRx[cellIdx] = cuphy::tensor_device(addr, tInfo);
    }

    uint32_t maxNCbs = getMaxNCbs(&m_puschStatPrms);
    uint32_t maxNCbsPerTb = getMaxNCbsPerTb(&m_puschStatPrms);
    cuphyStatus_t status = PuschRx::expandBackEndParameters(&m_puschDynPrms, &m_puschStatPrms, m_drvdUeGrpPrmsCpu, m_tbPrmsCpu.addr(), m_ldpcParams, maxNCbs, maxNCbsPerTb);
    if(CUPHY_STATUS_SUCCESS != status) {
        throw cuphy::cuphy_fn_exception(status, "expandBackEndParameters()");
    }

    PerTbParams* pPerTbPrms = m_tbPrmsCpu.addr();
    for(uint32_t iterator = 0; iterator < m_puschDynPrms.pCellGrpDynPrm->nUeGrps; iterator++)
    {
        cuphyPuschUeGrpPrm_t ueGrpPrms = m_puschDynPrms.pCellGrpDynPrm->pUeGrpPrms[iterator];
        for(int i = 0; i < ueGrpPrms.nUes; i++)
        {
            uint16_t ueIdx = ueGrpPrms.pUePrmIdxs[i];
            uint8_t  Qm    = static_cast<uint8_t>(pPerTbPrms[ueIdx].Qm);
            for(int j = 0; j < pPerTbPrms[ueIdx].Nl; ++j)
            {
                m_drvdUeGrpPrmsCpu[iterator].qam[pPerTbPrms[ueIdx].layer_map_array[j]] = Qm;
            }
        }

        // Need to calculate N here ??
        //pPerTbPrms[iterator].N = (pPerTbPrms[iterator].bg == 1) ?
        //                        CUPHY_LDPC_MAX_BG1_UNPUNCTURED_VAR_NODES * pPerTbPrms[iterator].Zc :
        //                        CUPHY_LDPC_MAX_BG2_UNPUNCTURED_VAR_NODES * pPerTbPrms[iterator].Zc;
        // printf("N %u\n", pPerTbPrms[iterator].N);
    }

    uint32_t* harqBufferSizeInBytes    = dynPrms.pDataOut->h_harqBufferSizeInBytes;
    size_t        NUM_BYTES_PER_LLR    = 2; // fp16 LLRs
    for(uint32_t tbIdx = 0; tbIdx < dynPrms.pCellGrpDynPrm->nUes; ++tbIdx)
    {
        size_t cur_deRmLLRSize;
        cur_deRmLLRSize                = NUM_BYTES_PER_LLR * (m_tbPrmsCpu[tbIdx].Ncb_padded) * m_tbPrmsCpu[tbIdx].num_CBs;
        harqBufferSizeInBytes[tbIdx] = cur_deRmLLRSize;
    }

    uint32_t totNumTbs = 0;
    uint32_t totNumCbs = 0;
    uint32_t totNumPayloadBytes = 0;

    for(uint32_t ueIdx = 0; ueIdx < dynPrms.pCellGrpDynPrm->nUes; ++ueIdx)
    {
        if(dynPrms.pCellGrpDynPrm->pUePrms[ueIdx].pduBitmap & 1)
        {
            dynPrms.pDataOut->pStartOffsetsTbPayload[ueIdx] = totNumPayloadBytes;
            dynPrms.pDataOut->pStartOffsetsTbCrc[ueIdx]     = totNumTbs;
            dynPrms.pDataOut->pStartOffsetsCbCrc[ueIdx]     = totNumCbs;
            totNumTbs += 1;

            // TBD
            // m_schUserIdxsVec[m_nSchUes] =  ueIdx;
            // m_nSchUes                   += 1;

            totNumCbs += m_tbPrmsCpu[ueIdx].num_CBs;

            uint8_t  crcSizeBytes = m_tbPrmsCpu[ueIdx].tbSize > 3824 ? 3 : 2;     // 38.212, section 7.2.1
            uint32_t tbSizeBytes  = m_tbPrmsCpu[ueIdx].tbSize / 8 + crcSizeBytes; // in cuPHY each TB includes TB payload + TB CRC
            totNumPayloadBytes += tbSizeBytes;

            uint32_t tbWordAlignPaddingBytes = (sizeof(uint32_t) - (tbSizeBytes % sizeof(uint32_t))) % sizeof(uint32_t);
            totNumPayloadBytes += tbWordAlignPaddingBytes;
        }
    }
    dynPrms.pDataOut->totNumTbs             = totNumTbs;
    dynPrms.pDataOut->totNumCbs             = totNumCbs;
    dynPrms.pDataOut->totNumPayloadBytes    = totNumPayloadBytes;

    copyPerTbPrms();

}


void PuschParams::copyPuschRxUeGrpPrms() {
     CUDA_CHECK(cudaMemcpyAsync(m_drvdUeGrpPrmsGpu,
                                m_drvdUeGrpPrmsCpu,
                                sizeof(cuphyPuschRxUeGrpPrms_t) * MAX_N_USER_GROUPS_SUPPORTED,
                                cudaMemcpyHostToDevice,
                                m_puschDynPrms.phase1Stream));
}


void PuschParams::copyPerTbPrms() {
     CUDA_CHECK(cudaMemcpyAsync(m_tbPrmsGpu.addr(),
                                m_tbPrmsCpu.addr(),
                                sizeof(PerTbParams) * MAX_N_TBS_SUPPORTED,
                                cudaMemcpyHostToDevice,
                                m_puschDynPrms.phase1Stream));
}


void PuschParams::setDynPrmsPhase1(const py::object& dynPrms) {

    m_LinearAlloc.reset();

    // Cell group dynamic parameters.
    const py::object cellGrpDynPrm = dynPrms.attr("cellGrpDynPrm");

    // Cell parameters.
    const py::list cellPrmsList = cellGrpDynPrm.attr("cellPrms");
    uint16_t nCells = cellPrmsList.size();
    m_cellDynPrms.resize(nCells);
    for (int cellIdx = 0; cellIdx < nCells; cellIdx ++ ) {
        const py::object cellDynPrms = cellPrmsList[cellIdx];

        m_cellDynPrms[cellIdx].cellPrmStatIdx = cellDynPrms.attr("cellPrmStatIdx").cast<uint16_t>();
        m_cellDynPrms[cellIdx].cellPrmDynIdx  = cellDynPrms.attr("cellPrmDynIdx").cast<uint16_t>();
        m_cellDynPrms[cellIdx].slotNum        = cellDynPrms.attr("slotNum").cast<uint16_t>();
    }
    m_cellGrpDynPrms.pCellPrms = m_cellDynPrms.data();
    m_cellGrpDynPrms.nCells = nCells;

    // UE group parameters.
    const py::list ueGrpPrmsList = cellGrpDynPrm.attr("ueGrpPrms");
    uint16_t nUeGrps = ueGrpPrmsList.size();
    m_ueGrpPrms.resize(nUeGrps);
    m_dmrsPrms.resize(nUeGrps);
    m_uePrmIdxs.resize(nUeGrps);
    for (int ueGroupIdx = 0; ueGroupIdx < nUeGrps; ueGroupIdx++) {
        const py::object ueGrpPrms = ueGrpPrmsList[ueGroupIdx];

        int cellPrmIdx = ueGrpPrms.attr("cellPrmIdx").cast<int>();
        m_ueGrpPrms[ueGroupIdx].pCellPrm = &m_cellGrpDynPrms.pCellPrms[cellPrmIdx];
        m_ueGrpPrms[ueGroupIdx].startPrb = ueGrpPrms.attr("startPrb").cast<uint16_t>();
        m_ueGrpPrms[ueGroupIdx].nPrb = ueGrpPrms.attr("nPrb").cast<uint16_t>();
        m_ueGrpPrms[ueGroupIdx].prgSize = ueGrpPrms.attr("prgSize").cast<uint16_t>();
        m_ueGrpPrms[ueGroupIdx].enablePerPrgChEstPerUeg = 0;
        m_ueGrpPrms[ueGroupIdx].nUplinkStreams = ueGrpPrms.attr("nUplinkStreams").cast<uint16_t>();
        m_ueGrpPrms[ueGroupIdx].puschStartSym = ueGrpPrms.attr("puschStartSym").cast<uint8_t>();
        m_ueGrpPrms[ueGroupIdx].nPuschSym = ueGrpPrms.attr("nPuschSym").cast<uint8_t>();
        m_ueGrpPrms[ueGroupIdx].dmrsSymLocBmsk = ueGrpPrms.attr("dmrsSymLocBmsk").cast<uint16_t>();
        m_ueGrpPrms[ueGroupIdx].rssiSymLocBmsk = ueGrpPrms.attr("rssiSymLocBmsk").cast<uint16_t>();

        // pUePrmIdxs
        py::list uePrmIdxsList = ueGrpPrms.attr("uePrmIdxs");
        uint16_t nUes = uePrmIdxsList.size();
        m_ueGrpPrms[ueGroupIdx].nUes = nUes;
        m_uePrmIdxs[ueGroupIdx].resize(nUes);
        for (int idx = 0; idx < nUes; idx++) {
            m_uePrmIdxs[ueGroupIdx][idx] = uePrmIdxsList[idx].cast<uint16_t>();
        }
        m_ueGrpPrms[ueGroupIdx].pUePrmIdxs = m_uePrmIdxs[ueGroupIdx].data();

        // DMRS
        py::object dmrsDynPrm = ueGrpPrms.attr("dmrsDynPrm");
        m_dmrsPrms[ueGroupIdx].nDmrsCdmGrpsNoData = dmrsDynPrm.attr("numDmrsCdmGrpsNoData").cast<uint8_t>();
        m_dmrsPrms[ueGroupIdx].dmrsScrmId         = dmrsDynPrm.attr("dmrsScrmId").cast<uint16_t>();
        try {
            m_dmrsPrms[ueGroupIdx].dmrsAddlnPos   = dmrsDynPrm.attr("dmrsAddlnPos").cast<uint8_t>();
            m_dmrsPrms[ueGroupIdx].dmrsMaxLen     = dmrsDynPrm.attr("dmrsMaxLen").cast<uint8_t>();
        }
        catch(...) {
            // TODO eventually these two fields will be removed from the TVs.
            // Added this to help with the transition.
            m_dmrsPrms[ueGroupIdx].dmrsAddlnPos   = 0;
            m_dmrsPrms[ueGroupIdx].dmrsMaxLen     = 0;
        }
        m_ueGrpPrms[ueGroupIdx].pDmrsDynPrm = &m_dmrsPrms[ueGroupIdx];
    }
    m_cellGrpDynPrms.pUeGrpPrms = m_ueGrpPrms.data();
    m_cellGrpDynPrms.nUeGrps = nUeGrps;

    // UE parameters.
    const py::list uePrmsList = cellGrpDynPrm.attr("uePrms");
    uint16_t nUes = uePrmsList.size();
    m_uePrms.resize(nUes);
    for (int ueIdx = 0; ueIdx < nUes; ueIdx++) {
        const py::object uePrms = uePrmsList[ueIdx];

        m_uePrms[ueIdx].pduBitmap          = uePrms.attr("pduBitmap").cast<uint16_t>();
        uint16_t ueGrpIdx                  = uePrms.attr("ueGrpIdx").cast<uint16_t>();
        m_uePrms[ueIdx].ueGrpIdx           = ueGrpIdx;
        m_uePrms[ueIdx].pUeGrpPrm          = &m_cellGrpDynPrms.pUeGrpPrms[ueGrpIdx];
        m_uePrms[ueIdx].enableTfPrcd       = uePrms.attr("enableTfPrcd").cast<uint8_t>();
        // TODO: Add DFT-s-OFDM.
        m_uePrms[ueIdx].scid               = uePrms.attr("scid").cast<uint8_t>();
        m_uePrms[ueIdx].dmrsPortBmsk       = uePrms.attr("dmrsPortBmsk").cast<uint16_t>();
        m_uePrms[ueIdx].mcsTableIndex      = uePrms.attr("mcsTable").cast<uint8_t>();
        m_uePrms[ueIdx].mcsIndex           = uePrms.attr("mcsIndex").cast<uint8_t>();
        m_uePrms[ueIdx].targetCodeRate     = uePrms.attr("targetCodeRate").cast<uint16_t>();
        m_uePrms[ueIdx].qamModOrder        = uePrms.attr("qamModOrder").cast<uint8_t>();
        m_uePrms[ueIdx].TBSize             = uePrms.attr("TBSize").cast<uint32_t>();
        m_uePrms[ueIdx].rv                 = uePrms.attr("rv").cast<uint8_t>();
        m_uePrms[ueIdx].rnti               = uePrms.attr("rnti").cast<uint16_t>();
        m_uePrms[ueIdx].dataScramId        = uePrms.attr("dataScramId").cast<uint16_t>();
        m_uePrms[ueIdx].nUeLayers          = uePrms.attr("nUeLayers").cast<uint8_t>();
        m_uePrms[ueIdx].ndi                = uePrms.attr("ndi").cast<uint8_t>();
        m_uePrms[ueIdx].harqProcessId      = uePrms.attr("harqProcessId").cast<uint8_t>();
        m_uePrms[ueIdx].i_lbrm             = uePrms.attr("i_lbrm").cast<uint8_t>();
        m_uePrms[ueIdx].maxLayers          = uePrms.attr("maxLayers").cast<uint8_t>();
        m_uePrms[ueIdx].maxQm              = uePrms.attr("maxQm").cast<uint8_t>();
        m_uePrms[ueIdx].n_PRB_LBRM         = uePrms.attr("n_PRB_LBRM").cast<uint16_t>();
        m_uePrms[ueIdx].pUciPrms           = nullptr;  // UCI not supported.
        m_uePrms[ueIdx].debug_d_derateCbsIndices = nullptr;  // Not supported.
        m_uePrms[ueIdx].foForgetCoeff      = 0.0f; // Not supported
        m_uePrms[ueIdx].ldpcEarlyTerminationPerUe = 0; // Not supported
        m_uePrms[ueIdx].ldpcMaxNumItrPerUe  = 10; // Not supported
    }
    m_cellGrpDynPrms.nUes = nUes;
    m_cellGrpDynPrms.pUePrms = m_uePrms.data();

    m_puschDynPrms.pCellGrpDynPrm = &m_cellGrpDynPrms;

    // Input
    py::object dataIn  = dynPrms.attr("dataIn");
    py::list tDataRxList = dataIn.attr("tDataRx");
    m_tPrmDataRx.resize(nCells);
    m_tDataRx.resize(nCells);
    for (int cellIdx = 0; cellIdx < nCells; cellIdx++) {
        const cuda_array_t<std::complex<float>>& tDataRx = tDataRxList[cellIdx].cast<cuda_array_t<std::complex<float>>>();
        m_tDataRx[cellIdx] = deviceFromCudaArray<std::complex<float>>(
            tDataRx,
            nullptr,
            CUPHY_C_32F,
            CUPHY_C_16F,
            cuphy::tensor_flags::align_tight,
            m_puschDynPrms.phase1Stream);
        m_tPrmDataRx[cellIdx].desc = m_tDataRx[cellIdx].desc().handle();
        m_tPrmDataRx[cellIdx].pAddr = m_tDataRx[cellIdx].addr();
    }
    m_puschDataIn.pTDataRx = m_tPrmDataRx.data();
    m_puschDataIn.pTX_72e      = nullptr;
    m_puschDataIn.pTReeInv_72e = nullptr;

    m_puschDynPrms.pDataIn = &m_puschDataIn;

    // Output
    py::object dataOut = dynPrms.attr("dataOut");
    m_puschDataOut.isEarlyHarqPresent               = 0;
    m_puschDataOut.pDataRx                          = 0;
    m_puschDataOut.dataRxHostCellStride             = 0;
    m_puschDataOut.pCbCrcs                          = numpyArrayToPtr<uint32_t>(dataOut.attr("cbCrcs"));
    m_puschDataOut.pTbCrcs                          = numpyArrayToPtr<uint32_t>(dataOut.attr("tbCrcs"));
    m_puschDataOut.pTbPayloads                      = numpyArrayToPtr<uint8_t>(dataOut.attr("tbPayloads"));
    m_puschDataOut.h_harqBufferSizeInBytes          = numpyArrayToPtr<uint32_t>(dataOut.attr("harqBufferSizeInBytes"));
    m_puschDataOut.pUciPayloads                     = nullptr;
    m_puschDataOut.pUciCrcFlags                     = nullptr;
    m_puschDataOut.pNumCsi2Bits                     = nullptr;
    m_puschDataOut.pStartOffsetsCbCrc               = numpyArrayToPtr<uint32_t>(dataOut.attr("startOffsetsCbCrc"));
    m_puschDataOut.pStartOffsetsTbCrc               = numpyArrayToPtr<uint32_t>(dataOut.attr("startOffsetsTbCrc"));
    m_puschDataOut.pStartOffsetsTbPayload           = numpyArrayToPtr<uint32_t>(dataOut.attr("startOffsetsTbPayload"));
    m_puschDataOut.pUciOnPuschOutOffsets            = nullptr;
    m_puschDataOut.pTaEsts                          = numpyArrayToPtr<float>(dataOut.attr("taEsts"));
    m_puschDataOut.pRssi                            = numpyArrayToPtr<float>(dataOut.attr("rssi"));
    m_puschDataOut.pRsrp                            = numpyArrayToPtr<float>(dataOut.attr("rsrp"));
    m_puschDataOut.pNoiseVarPreEq                   = numpyArrayToPtr<float>(dataOut.attr("noiseVarPreEq"));
    m_puschDataOut.pNoiseVarPostEq                  = numpyArrayToPtr<float>(dataOut.attr("noiseVarPostEq"));
    m_puschDataOut.pSinrPreEq                       = numpyArrayToPtr<float>(dataOut.attr("sinrPreEq"));
    m_puschDataOut.pSinrPostEq                      = numpyArrayToPtr<float>(dataOut.attr("sinrPostEq"));
    m_puschDataOut.pCfoHz                           = numpyArrayToPtr<float>(dataOut.attr("cfoHz"));
    m_puschDataOut.pChannelEsts                     = nullptr;
    m_puschDataOut.pChannelEstSizes                 = nullptr;
    m_puschDataOut.HarqDetectionStatus              = numpyArrayToPtr<uint8_t>(dataOut.attr("HarqDetectionStatus"));
    m_puschDataOut.CsiP1DetectionStatus             = numpyArrayToPtr<uint8_t>(dataOut.attr("CsiP1DetectionStatus"));
    m_puschDataOut.CsiP2DetectionStatus             = numpyArrayToPtr<uint8_t>(dataOut.attr("CsiP2DetectionStatus"));
    m_puschDataOut.pPreEarlyHarqWaitKernelStatusGpu = nullptr;//numpyArrayToPtr<uint8_t>(dataOut.attr("preEarlyHarqWaitStatus"));
    m_puschDataOut.pPostEarlyHarqWaitKernelStatusGpu= nullptr;//numpyArrayToPtr<uint8_t>(dataOut.attr("postEarlyHarqWaitStatus"));

    m_puschDataOut.nPerCellTbDests = 0;
    std::memset(m_puschDataOut.pPerCellTbPayloads, 0, sizeof(m_puschDataOut.pPerCellTbPayloads));

    m_puschDynPrms.pDataOut = &m_puschDataOut;

    // Input/output.
    m_puschDataInOut.pHarqBuffersInOut = m_bHarqBufferPtrs.addr(); // Currently not returning HARQ to Python.
    m_puschDataInOut.pFoCompensationBuffersInOut = m_bFoCompensationBufferPtrs.addr();
    m_puschDynPrms.pDataInOut = &m_puschDataInOut;

    // Status parameter.
    m_puschDynPrms.pStatusOut = &m_puschStatusOutput;

    // Debug parameters.
    try {
        py::object dbgPrms = dynPrms.attr("dbg");
        m_puschDynDbgPrms.enableApiLogging = dbgPrms.attr("enableApiLogging").cast<uint8_t>();
    }
    catch(...) {
        printf("Couldn't read PuschDynDbgPrms!\n");
        m_puschDynDbgPrms.enableApiLogging = 0;
    }
    m_puschDynPrms.pDbg = &m_puschDynDbgPrms;
}

void PuschParams::setDynPrmsPhase2(const py::object& dynPrms) {

    py::object dataInOut = dynPrms.attr("dataInOut");

    // Allocate HARQ buffers based on the calculated requirements from setupPhase 1
    for(int k = 0; k < m_puschDynPrms.pDataOut->totNumTbs; k++) {
        try {
            py::list harqBuffersInOut = dataInOut.attr("harqBuffersInOut");
            uint8_t* harqBuffers = (uint8_t*)harqBuffersInOut[k].cast<uint64_t>();
            m_puschDataInOut.pHarqBuffersInOut[k] = harqBuffers;
        }
        catch(...) {
            printf("Warning: Cannot read harqBuffersInOut, set harqBuffers to 0... \n");
            uint8_t* harqBuffers;
            harqBuffers = static_cast<uint8_t*>(m_LinearAlloc.alloc(m_puschDynPrms.pDataOut->h_harqBufferSizeInBytes[k] * sizeof(uint8_t)));
            printf("h_harqBufferSizeInBytes[k]: %d \n", m_puschDynPrms.pDataOut->h_harqBufferSizeInBytes[k]);
            CUDA_CHECK(cudaMemsetAsync(harqBuffers, 0, m_puschDynPrms.pDataOut->h_harqBufferSizeInBytes[k] * sizeof(uint8_t), m_puschDynPrms.phase1Stream));
            m_puschDynPrms.pDataInOut->pHarqBuffersInOut[k] = harqBuffers;
        }
    }
}


void PuschParams::setDynPrms(const py::object& dynPrms) {

    m_puschDynPrms.phase1Stream               = (cudaStream_t)dynPrms.attr("phase1Stream").cast<uint64_t>();
    m_puschDynPrms.phase2Stream               = (cudaStream_t)dynPrms.attr("phase2Stream").cast<uint64_t>();
    m_puschDynPrms.setupPhase                 = dynPrms.attr("setupPhase").cast<cuphyPuschSetupPhase_t>();
    m_puschDynPrms.procModeBmsk               = dynPrms.attr("procModeBmsk").cast<uint64_t>();
    m_puschDynPrms.waitTimeOutPreEarlyHarqUs  = dynPrms.attr("waitTimeOutPreEarlyHarqUs").cast<uint16_t>();
    m_puschDynPrms.waitTimeOutPostEarlyHarqUs = dynPrms.attr("waitTimeOutPostEarlyHarqUs").cast<uint16_t>();
    m_puschDynPrms.cpuCopyOn                  = dynPrms.attr("cpuCopyOn").cast<uint8_t>();

    if (m_puschDynPrms.setupPhase == PUSCH_SETUP_PHASE_1) {
        setDynPrmsPhase1(dynPrms);
        m_nMaxPrb = updatePuschRxUeGrpPrms(m_puschDynPrms.phase1Stream);
    }
    else {  // PUSCH_SETUP_PHASE_2
        setDynPrmsPhase2(dynPrms);
    }
    copyPerTbPrms();
}


uint32_t PuschParams::updatePuschRxUeGrpPrms(cudaStream_t cuStream) {

    bool subSlotProcessingFrontLoadedDmrsEnabled = false;
    uint8_t maxDmrsMaxLen = 1;
    uint32_t maxNPrbAlloc = getMaxNPrbAlloc(&m_puschStatPrms);
    uint32_t nMaxPrb = PuschRx::expandFrontEndParameters(&m_puschDynPrms, &m_puschStatPrms, m_drvdUeGrpPrmsCpu, subSlotProcessingFrontLoadedDmrsEnabled, maxDmrsMaxLen, false, maxNPrbAlloc);

    // Set modulation orders.
    cuphyPuschUePrm_t* uePrmsArray = m_puschDynPrms.pCellGrpDynPrm->pUePrms;
    for(uint32_t iterator = 0; iterator < m_puschDynPrms.pCellGrpDynPrm->nUeGrps; iterator++)
    {
        cuphyPuschUeGrpPrm_t* ueGrpPrms = &m_puschDynPrms.pCellGrpDynPrm->pUeGrpPrms[iterator];

        uint8_t ueGrpLayerIdx = 0;

        for(int i = 0; i < ueGrpPrms->nUes; ++i) {
            uint16_t ueIdx = ueGrpPrms->pUePrmIdxs[i];
            for(int j = 0; j < m_drvdUeGrpPrmsCpu[iterator].nUeLayers[i]; ++j) {
                m_drvdUeGrpPrmsCpu[iterator].qam[ueGrpLayerIdx] = uePrmsArray[ueIdx].qamModOrder;
                ueGrpLayerIdx++;
            }
        }
    }
    return nMaxPrb;
}


void PuschParams::printStatPrms() const {
    printPuschStatPrms(m_puschStatPrms);
}


void PuschParams::printDynPrms() const {
    printPuschDynPrms(m_puschDynPrms);
}

void PuschParams::printPerPuschTbParams(const char* desc) const {
    printPerTbParams(desc, m_puschDynPrms.pCellGrpDynPrm->nUes, m_tbPrmsCpu.addr());
}

void PuschParams::printDrvdUeGrpPrms() const {
    for(uint32_t iterator = 0; iterator < m_puschDynPrms.pCellGrpDynPrm->nUeGrps; iterator++) {
        printf("Derived PUSCH UE group parameters, UE group %d: ", iterator);
        printPuschDrvdUeGrpPrms(m_drvdUeGrpPrmsCpu[iterator]);
    }
}


}
