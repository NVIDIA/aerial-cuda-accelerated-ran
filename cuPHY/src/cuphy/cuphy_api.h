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

/** \file cuphy_api.h
 *  \brief PHY Layer library header file
 *
 *  Header file for the cuPHY API
 */

#if !defined(CUPHY_API_H_INCLUDED_)
#define CUPHY_API_H_INCLUDED_

#include <cuda_runtime.h>
#include <stdint.h>
#include "cuComplex.h"
#include "cuda_fp16.h"
#include "cuphy.h"

#ifndef CUPHYWINAPI
#ifdef _WIN32
#define CUPHYWINAPI __stdcall
#else
#define CUPHYWINAPI
#endif
#endif

#if defined(__cplusplus)
extern "C" {
#endif /* defined(__cplusplus) */


#define PRINT_GPU_MEMORY_CUPHY_CHANNEL 1 // Set to 1 to print to standard output after PHY channel object creation or 0 otherwise

/**
 * CUPHY per-cell static parameters
 *
 * @brief Struct tracks static, per-cell information, needed both for downlink (DL) and uplink (UL).
 */

// Per cell static parameters

typedef struct _cuphyCellStatPrm
{
    uint16_t phyCellId; /*!< physical cell Id */
    uint16_t nRxAnt;    /*!< number of receiving antenna ports for PUSCH/PUCCH/PRACH */
    uint16_t nRxAntSrs; /*!< number of receiving antenna ports for SRS. SRS UL/DL beamforming also assumes this many gNB ports  */
    uint16_t nTxAnt;    /*!< number of transmitting antenna ports */
    uint16_t nPrbUlBwp; /*!< number of PRBs (Physical Resource Blocks) allocated in UL BWP (bandwidth part) */
    uint16_t nPrbDlBwp; /*!< number of PRBs allocated in DL BWP */
    uint8_t  mu;        /*!< numerology [0, 3] */
    // uint8_t  cpType;
    // uint8_t  duplexingMode;

    cuphyPuschCellStatPrm_t* pPuschCellStatPrms; /*!< Per-cell PUSCH static parameters */
    cuphyPucchCellStatPrm_t* pPucchCellStatPrms; /*!< Per-cell PUCCH static parameters */
} cuphyCellStatPrm_t;

/**
 * cuPHY static and quasi-static state update types
 */
typedef enum _cuphyStateUpdateType
{
    CUPHY_STATE_UPDATE_TYPE_CREATE_STATIC       = 0x0, /*!< To be used only at creation time. Result: (a) Memory allocation for static state (b) State initialization. */
    CUPHY_STATE_UPDATE_TYPE_CREATE_QUASI_STATIC = 0x1, /*!< To be used only at creation time. Result: (a) Memory allocation for quasi-static state (b) Quasi-static state initialization. */
    CUPHY_STATE_UPDATE_TYPE_MODIFY              = 0x2, /*!< To be used only during (re)configuration and is relevant only to quasi-static state. Results in state update only (no memory allocation). */
    CUPHY_STATE_UPDATE_TYPE_DESTROY             = 0x3, /*!< Cleanup allocated memory and state handles. */
    CUPHY_MAX_STATE_UPDATE_TYPES
} cuphyStateUpdateType_t;


/**
 * Memory Tracker
 */
typedef struct _cuphyTracker
{
    const void* pMemoryFootprint; /* pointer to cuphyMemoryFootprint object of a given cuPHY channel object */
} cuphyTracker_t;

/**
 * \defgroup CUPHY_PUSCH_RECEIVER  PUSCH Receiver
 *
 * This section describes the PUSCH receive pipeline functions of the cuPHY
 * application programming interface.
 *
 * @{
 */

struct cuphyPuschRx;
/**
 * cuPHY PUSCH Receiver handle
 */
typedef struct cuphyPuschRx* cuphyPuschRxHndl_t;

/**
 * PUSCH processing modes
 */
typedef enum _cuphyPuschProcMode
{
    PUSCH_PROC_MODE_FULL_SLOT           = 0,           /// Full slot processing
    PUSCH_PROC_MODE_FULL_SLOT_GRAPHS    = (1ULL << 0), /// Full slot processing with graphs
    PUSCH_PROC_MODE_SUB_SLOT            = (1ULL << 1), /// Enable early-HARQ (HARQ bits fully resident on symbols 0-3) sub-slot processing
    PUSCH_MAX_PROC_MODES
} cuphyPuschProcMode_t;

/**
 * PUSCH Static Parameters API logging
 */
typedef struct _cuphyPuschStatDbgPrms
{
    const char* pOutFileName;       /// output file capturing pipeline intermediate states. No capture if null.
    uint8_t     descrmOn;           /// Descrambling enable/disable
    uint8_t     enableApiLogging;   /// control the API logging of PUSCH static parameters
    uint16_t    forcedNumCsi2Bits;  /// if > 0 cuPHY assumes all csi2 UCIs have forcedNumCsi2Bits bits
} cuphyPuschStatDbgPrms_t;

typedef enum _cuphyPuschLdpcKernelLaunch
{
    PUSCH_RX_ENABLE_DRIVER_LDPC_LAUNCH = 0x1,         // driver API is used to launch the LDPC decoder kernel on a single stream. Note: requires CUDA 11.0 or higher
    PUSCH_RX_LDPC_STREAM_POOL          = 0x2,         // LDPC kernel launch will run on multiple-stream via internal PUSCH stream-pool
    PUSCH_RX_LDPC_STREAM_SEQUENTIAL    = 0x4,         // LDPC kernel launches will occur in a single stream, and each transport block will be
                                                      // processed via a separate kernel launch. In general, this may result in using fewer
                                                      // GPU resources, but may increase latency
    PUSCH_RX_ENABLE_LDPC_DEC_SINGLE_STREAM_OPT = 0x8, // inputs that have certain conditions will result in an LDPC decoder kernel
                                                      // launch in the same stream as other components. (For this to occur, there must
                                                      // be fewer than CUPHY_LDPC_DECODE_DESC_MAX_TB transport blocks, and all transport
                                                      // blocks must have the same LDPC configuration.) Only used when
                                                      // PUSCH_RX_LDPC_STREAM_SEQUENTIAL is NOT selected
} cuphyPuschLdpcKernelLaunch_t;

/**
 * PUSCH Symbol RX state
 */
typedef enum _cuphySymbolRxState
{
    SYM_RX_NOT_DONE = 0, /// OFDM symbol to be received
    SYM_RX_DONE     = 1, /// OFDM symbol received succesfully
    SYM_RX_TIMEOUT  = 2, /// OFDM symbol reception timed out
    SYM_RX_ERROR    = 3, /// Error during OFDM symbol reception
    SYM_RX_MAX
} cuphySymbolRxState_t;


/**
 * PUSCH Work Cancellation mode
 */
typedef enum _cuphyPuschWorkCancelMode
{
    PUSCH_NO_WORK_CANCEL          = 0x0, /*!< No work cancellation from PUSCH graph(s) support */
    PUSCH_COND_IF_NODES_W_KERNEL  = 0x1, /*!< Conditional IF nodes in PUSCH graph(s) with cond. value set via kernel. Cannot be used if cuphyPuschStatPrms_t enabledDeviceGraphLaunch is set. */
    PUSCH_DEVICE_GRAPHS           = 0x2, /*!< Device graphs launched from the device. */
    PUSCH_MAX_WORK_CANCEL_MODES   = 0x3
} cuphyPuschWorkCancelMode_t;


/**
 * PUSCH Static Parameters
 */
typedef struct _cuphyPuschStatPrms
{
    cuphyTracker_t*     pOutInfo;      /*!< pointer to cuphyTracker_t. Its pMemoryFootprint pointer will be updated by cuPHY in channel objection creation. The caller will thus have access to the cuphyMemoryFootprint object of that cuPHY PUSCH channel object. The cuphyMemoryFootprint object tracks the size of the GPU memory allocations owned by that cuPHY PUSCH channel object. */

    uint8_t*          pWorkCancelInfo; /*!< pointer to a host-pinned (initially) memory location that has a value of 1 if PUSCH run should early exit (e.g., if conditional graph nodes or device graph launches are used) or 0 if it should continue as usual */

    /// Channel estimation filters
    cuphyTensorPrm_t* pWFreq;
    cuphyTensorPrm_t* pWFreq4;
    cuphyTensorPrm_t* pWFreqSmall;

    /// Channel estimation sequences
    cuphyTensorPrm_t* pShiftSeq;
    cuphyTensorPrm_t* pUnShiftSeq;
    cuphyTensorPrm_t* pShiftSeq4;
    cuphyTensorPrm_t* pUnShiftSeq4;

    /// rkhs paramaters
    cuphyPuschRkhsPrms_t* pPuschRkhsPrms;

    /// CFO correction enable/disable flag: 0 - disable, 1 - enable
    uint8_t enableCfoCorrection;
    
    /// Weighted Average CFO estimation enable/disable flag: 0 - disable, 1 - enable
    uint8_t enableWeightedAverageCfo;

    /// TO estimation enable/disable flag: 0 - disable, 1 - enable
    uint8_t enableToEstimation;

    /// Time domain interpolation on PUSCH: 0 - disable, 1 - enable
    uint8_t  enablePuschTdi;

    /// DFT-s-OFDM enable/disable flag: 0 - disable, 1 - enable
    uint8_t enableDftSOfdm;

    /// Tb size verification enable/disable flag: 0 - disable, 1 - enable
    uint8_t enableTbSizeCheck; 
    
    /// Enable UL RX beamforming
    uint8_t enableMassiveMIMO;

    /// Debugging output of post-equalized receive data samples flag: 0 - disable, 1 - enable
    uint8_t enableDebugEqOutput;

    /// LDPC static parameters
    uint8_t                      ldpcEarlyTermination;
    uint8_t                      ldpcUseHalf;
    uint8_t                      ldpcAlgoIndex;
    uint32_t                     ldpcFlags;
    cuphyPuschLdpcKernelLaunch_t ldpcKernelLaunch;
    cuphyLdpcMaxItrAlgoType_t    ldpcMaxNumItrAlgo;
    uint8_t                      fixedMaxNumLdpcItrs; // Used when ldpcMaxNumItrAlgo = FIXED
    float                        ldpcClampValue;

    /// Polar decoder list size
    uint8_t polarDcdrListSz;        // default size for PUSCH to be set to 1

    uint8_t nMaxTbPerNode; // Maximum number of transport blocks per node
    /// channel estimation algorithm selection: 0 - legacy MMSE, 1 - multi-stage MMSE with delay estimation, 2 - RKHS, 3 - LS estimation only
    cuphyPuschChEstAlgoType_t chEstAlgo;
    
    /// Per-PRG CHEST enable/disable flag: 0 - disable, 1 - enable
    uint8_t enablePerPrgChEst;

    /// equalizer coefficients algorithm selection
    cuphyPuschEqCoefAlgoType_t  eqCoeffAlgo;

    /// RSSI enable/disable flag: 0 - disable, 1 - enable
    uint8_t  enableRssiMeasurement;

    /// SINR enable/disable flag:  0 - disable, 1 - enable
    uint8_t  enableSinrMeasurement;

    /// FAPIv3 method for computing CSI-P2 size flag: 0 - disable, 1 - enable
    uint8_t enableCsiP2Fapiv3;

    int stream_priority;            /// CUDA stream priority for internal to PUSCH stream pool. Should match the priority of CUDA stream passed in cuphyCreatePuschRx()

    /// Cell specific static parameters
    uint16_t nMaxCells; /*!< Total # of cell configurations supported by the pipeline during its lifetime
                             Maximum # of cells scheduled in a slot. Out of nMaxCells, the nMaxCellsPerSlot most resource hungry cells are
                             used for resource provisioning purposes */

    uint16_t            nMaxCellsPerSlot; /// nMaxCellsPerSlot <= nMaxCells
    cuphyCellStatPrm_t* pCellStatPrms;

    /// Max values to be used for static memory allocation of PuschRx object
    uint32_t nMaxTbs;      /*!< Maximum number of transport blocks that will be supported by PuschRx object */
    uint32_t nMaxCbsPerTb; /*!< Maximum number of code blocks per transport block that will be supported by PuschRx object */
    uint32_t nMaxTotCbs;   /*!< Total number of code blocks (sum of # code blocks across all transport blocks) that will be supported by PuschRx object */

    uint32_t nMaxRx;  /*!< Maximum number of Rx antennas that will be supported by PuschRx object */
    uint32_t nMaxPrb; /*!< Maximum number of PRBs that will be supported by PuschRx object */
    
    uint32_t nMaxLdpcHetConfigs; /*!<  maximum number of allowed heterogenous workload configs for LDPC */

    /// Debug parameters
    cuphyPuschStatDbgPrms_t* pDbg;

    uint8_t     enableDeviceGraphLaunch;    /*!< Static flag to allow device graph launch in PUSCH */

    uint8_t     enableEarlyHarq;            /*!< Static flag to control construction of early-HARQ related members in PUSCH */

    int32_t     earlyHarqProcNodePriority;  /*!< Elevated priority used for early-HARQ processing nodes in graphs mode.
                                                The priority values are same as CUDA stream priorities with lower numbers imply greater priorities */
    cudaEvent_t subSlotCompletedEvent;      /*!< Event used to signal completion of sub-slot processing */

    cudaEvent_t waitCompletedSubSlotEvent;  /*!< Event used to signal the completion of the initial wait time in sub-slot, prior to start the pipeline */

    cudaEvent_t waitCompletedFullSlotEvent; /*!< Event used to signal the completion of the initial wait time in full-slot, prior to start the pipeline */

    uint32_t const* pSymRxStatus;           /*!< - Pointer to a GPU readable array (length OFDM_SYMBOLS_PER_SLOT) containing per UL symbol reception status.
                                                   Each symbol status is updated when the symbol is received on all receivers of all cells scheduled for the slot.
                                                 - This is a read only flag for cuPHY
                                                 - Status values from cuphySymbolRxState_t
                                                Note: successful reception of symbol i does not necessarily imply successful reception of prior symbols 0,...,i-1 */

    cuphyPuschWorkCancelMode_t    workCancelMode; /*!< Controls how, if at all, PUSCH graph(s) can support work cancellation. For any value other than PUSCH_NO_WORK_CANCEL, the exit cond. will be controlled by the value pointed by pWorkCancelInfo. */

    const char* puschrxChestFactorySettingsFilename;  /*!< Points to a filename to be parsed, and will provide information related to the PuschRx Channel Estimate Factory
                                                           If the file does not exist, the cuphy channel estimate factory will fallback to the vanilla channel estimate.
                                                           i.e. instantiating the existing/legacy/vanilla puschRx Channel estimate type. */

    uint8_t enableBatchedMemcpy;  /*!< Enable batched memory copy. Values: 0->1. */

} cuphyPuschStatPrms_t;

//-----------------------------------------------------------------------------------------------------------
// PUSCH Dynamic Parameters

/// PUSCH Dynamic Parameters API logging
typedef struct _cuphyPuschDynDbgPrms
{
    uint8_t     enableApiLogging;   /// control the API logging of PUSCH static parameters
} cuphyPuschDynDbgPrms_t;

/// DMRS information
typedef struct _cuphyPuschDmrsPrm
{
    /// DMRS resource information
    uint8_t dmrsAddlnPos;
    uint8_t dmrsMaxLen;

    // uint8_t  dmrsType; // only DMRS type-A supported
    uint8_t  nDmrsCdmGrpsNoData; /// Used to calculate DMRS energy (via table lookup)
    uint16_t dmrsScrmId;
} cuphyPuschDmrsPrm_t;

/// Per cell dynamic parameter
typedef struct _cuphyPuschCellDynPrm
{
    uint16_t cellPrmStatIdx; /// Index to cell-static parameter information
    uint16_t cellPrmDynIdx;  /// Index to cell-dynamic parameter information
    uint16_t slotNum;

} cuphyPuschCellDynPrm_t;

/// Co-scheduled UE group parameters
typedef struct _cuphyPuschUeGrpPrm
{
    cuphyPuschCellDynPrm_t* pCellPrm; /// Pointer to UE group’s parent cell dynamic parameters

    /// DMRS information
    cuphyPuschDmrsPrm_t* pDmrsDynPrm;

    /// PUSCH frequency resource allocation
    uint16_t startPrb;
    uint16_t nPrb;
    
    uint16_t prgSize;
    uint8_t  enablePerPrgChEstPerUeg;
    
    uint16_t nUplinkStreams;

    /// PUSCH time domain resource allocation
    uint8_t puschStartSym;
    uint8_t nPuschSym; // PUSCH DMRS + data symbol count

    uint16_t dmrsSymLocBmsk; /*!< DMRS location bitmask (LSB 14 bits)
                                  PUSCH symbol locations derived from dmrsSymLocBmsk. Bit i is "1" if symbol i is DMRS.
                                  For example if there are DMRS are symbols 2 and 3, then: dmrsSymLocBmsk = 0000 0000 0000 1100 */

    /// RSSI measurement
    uint16_t rssiSymLocBmsk; /*!< Symbol location bitmask for RSSI measurement (LSB 14 bits)
                                  Bit i is "1" if symbol i needs be to measured, 0 disables RSSI calculation
                                  For example to measure RSSI on DMRS symbols 2, 6 and 9, use: rssiSymLocBmsk = 0000 0010 0100 0100 */

    /// Per UE information in co-scheduled group
    uint16_t  nUes;
    uint16_t* pUePrmIdxs;
} cuphyPuschUeGrpPrm_t;


/// Uci on pusch parameters
typedef struct _cuphyUciOnPusch
{
    uint16_t nBitsHarq;
    uint16_t nBitsCsi1;
    uint8_t  alphaScaling;
    uint8_t  betaOffsetHarqAck;
    uint8_t  betaOffsetCsi1;

    uint8_t betaOffsetCsi2;
    uint8_t rankBitOffset;
    uint8_t nRanksBits;
    uint8_t nCsiReports;
    float   DTXthreshold;

    // FAPIv3 method for computing number of CSI-P2 bits:
    uint16_t                nCsi2Reports;      // number of CSI-2 reports. 0-100.
    cuphyCalcCsi2SizePrm_t* pCalcCsi2SizePrms; // pointer to paramaters used to compute CSI-P2 size. Null if zero CSI-2 reports

} cuphyUciOnPuschPrm_t;

/// Per UE parameters
typedef struct _cuphyPuschUePrm
{
    /// Pusch tx options
    uint16_t pduBitmap; /*!< - Bit 0 indicates if data present. Bit 1 indicates if uci present.
                             - Bit 2 indicates if ptrs present. Bit 3 indicates DFT-S transmission.
                             - Bit 4 indicates if sch data present. Bit 5 indicates if CSI-P2 present */

    cuphyPuschUeGrpPrm_t* pUeGrpPrm; /// pointer to parent UE Group
    uint16_t              ueGrpIdx;  /// index of parent UE group

    /// DFT-s-OFDM
    uint8_t  enableTfPrcd;
    uint32_t puschIdentity;
    uint8_t  groupOrSequenceHopping;
    uint8_t  N_symb_slot;
    uint8_t  N_slot_frame;
    uint8_t  lowPaprGroupNumber;
    uint16_t lowPaprSequenceNumber;

    uint8_t  scid;
    uint16_t dmrsPortBmsk; /// Use to map DMRS port to fOCC/DMRS-grid/tOCC
    uint8_t  nlAbove16{0}; // This is not used for pusch. Keeping it in parity with pdsch. Will always be set to 0. This helps to use a common function track_eaxcids_fh for both PDSCH and PUSCH.

    /// Backend parameters
    uint8_t  mcsTableIndex; /*!< mcsTableIndex and mcsIndx only used for enabling Tb size checking (mcsTableIndex 0:Table 5.1.3.1-1, 1:Table 5.1.3.1-2, 2:Table 5.1.3.1-3, 3:Table 6.1.4.1-1, 4:Table 6.1.4.1-2) */
    uint8_t  mcsIndex;

    /// Parameters to enable MCS greater than 28
    uint16_t targetCodeRate;   /*!< Assuming the code rate is x/1024.0 where x contains a single digit after decimal point, then targetCodeRate = static_cast<uint16_t>(x * 10) = static_cast<uint16_t>(codeRate * 1024 * 10) */
    uint8_t  qamModOrder; /*!< Value: 2,4,6,8 if transform precoding is disabled; 1,2,4,6,8 if transform precoding is enabled */
    uint32_t TBSize; /*!< transport block size in bytes provided by L2, refer to Table 3–96 Optional puschData information in FAPI 10.04 */

    uint8_t  rv;
    uint16_t rnti;
    uint16_t dataScramId;
    uint8_t  nUeLayers;
    uint8_t  ndi;           /*!< 1 - new data, 0 - retx
                                 ndi is updated in setupCmnPhase2() after HARQ buffer allocation instead of setupCmnPhase1() */
    uint8_t  harqProcessId; /// value 0-15

    /// Parameters used for LBRM
    uint8_t  i_lbrm;     /// Boolean to use LBRM per 38.212 5.4.2.1 and 6.2.5
    uint8_t  maxLayers;  /// used for LBRM Nref calculation
    uint8_t  maxQm;      /// used for LBRM Nref calculation
    uint16_t n_PRB_LBRM; /// used for LBRM Nref calculation

    /// Parameters for UCI
    cuphyUciOnPuschPrm_t* pUciPrms; // pointer to uci parameters. Null if uci on pusch not configured.

    /// debug
    uint32_t* debug_d_derateCbsIndices;
    
    /// Weighted average CFO estimation
    float     foForgetCoeff;
    
    /// LDPC decoder
    uint8_t  ldpcEarlyTerminationPerUe;
    uint8_t  ldpcMaxNumItrPerUe;
} cuphyPuschUePrm_t;

/// Cell group dynamic parameters
struct cuphyPuschCellGrpDynPrm
{
    /// Cell group parameters
    uint16_t                nCells; // # of cells to be processed; nCells <= nMaxCellsPerSlot specified in cuphyPuschStatPrms_t
    cuphyPuschCellDynPrm_t* pCellPrms;

    /// Co-scheduled UE group parameters
    uint16_t              nUeGrps;
    cuphyPuschUeGrpPrm_t* pUeGrpPrms;

    /// UE parameters
    uint16_t           nUes;
    cuphyPuschUePrm_t* pUePrms;
};
typedef struct cuphyPuschCellGrpDynPrm cuphyPuschCellGrpDynPrm_t;

/**
 * PUSCH pipeline input data
 */
typedef struct _cuphyPuschDataIn
{
    cuphyTensorPrm_t* pTDataRx;   /*!< array of tensors with each tensor (indexed by cellPrmDynIdx) representing the receive slot buffer of a cell in the cell group
                                       Each cell's tensor may have a different geometry */
    cuphyTensorPrm_t* pTNoisePwr; /*!< array of noise power metric tensors with each tensor (indexed by cellPrmDynIdx) for given a cell in the cell group */
} cuphyPuschDataIn_t;


/// cuPHY PUSCH setup phase
typedef enum _cuphyPuschSetupPhase
{
    PUSCH_SETUP_PHASE_INVALID    = 0,
    PUSCH_SETUP_PHASE_1          = 1,
    PUSCH_SETUP_PHASE_2          = 2,
    PUSCH_SETUP_MAX_PHASES       = 3,
    PUSCH_SETUP_MAX_VALID_PHASES = PUSCH_SETUP_MAX_PHASES - 1
} cuphyPuschSetupPhase_t;


/// cuPHY PUSCH run phase
typedef enum _cuphyPuschRunPhase
{
    PUSCH_RUN_PHASE_INVALID                      = 0,
    PUSCH_RUN_SUB_SLOT_PROC                      = 1, // sub-slot processing for early-HARQ + D2H copies of early-HARQ result or for front-loaded DMRS symbol(s)
    PUSCH_RUN_FULL_SLOT_PROC                     = 2, // full-slot processing
    PUSCH_RUN_FULL_SLOT_COPY                     = 3, // copying output of PUSCH_RUN_FULL_SLOT_PROC from GPU to CPU
    PUSCH_RUN_ALL_PHASES                         = 4, // PUSCH_RUN_FULL_SLOT_PROC + PUSCH_RUN_FULL_SLOT_COPY
    PUSCH_RUN_MAX_PHASES                         = 5,
    PUSCH_RUN_MAX_VALID_PHASES = PUSCH_RUN_MAX_PHASES - 1
} cuphyPuschRunPhase_t;

/**
 * PUSCH pipeline output data
 */
typedef struct _cuphyPuschDataOut
{
    uint32_t* h_harqBufferSizeInBytes;  /*!< calculated in Setup() PUSCH_SETUP_PHASE_1
                                             pointer to array of HARQ buffer sizes */

    uint8_t isEarlyHarqPresent; /*< calculated in Setup() PUSCH_SETUP_PHASE_1
                                    Flag when set indicates that the slot contains UEs with early-HARQ (HARQ bits fully resident on symbols 0-3) */

    uint8_t isFrontLoadedDmrsPresent; /*< calculated in Setup() PUSCH_SETUP_PHASE_1
                                    Flag when set indicates that the slot contains UEs with frontloaded DMRS symbol(s) */
    uint64_t isEarlyHarqPresentCellMask; /*< indicate which cells have early-HARQ UEs
    Bit 0 - Cell 0, Bit - Cell 1, Bit 2 - Cell 2 and so on.*/

    /// output size
    uint32_t totNumTbs;     //totNumTbs is the number of UEs with ULSCH-on-PUSCH. totNumTbs is not always equal to nUes.
    uint32_t totNumCbs;
    uint32_t totNumPayloadBytes;
    uint16_t totNumUciSegs;

    /// PUSCH processing results
    __half2*  pDataRx;
    uint32_t* pCbCrcs;
    uint32_t* pTbCrcs;
    uint8_t*  pTbPayloads;

    /// UCI processing results
    uint8_t*  pUciPayloads;
    uint8_t*  pUciCrcFlags;
    uint16_t* pNumCsi2Bits;

    /*< PUSCH results layout information
        nUes offsets providing start offset of UE CB-CRCs, UE TB-CRCs, UE TB-payload within a
        container pointed by pCbCrc, pTbCrc and pTbPayload. The UE ordering is identical to input UE ordering
        in pUePrms within cuphyPuschCellGrpDynPrm_t */
    uint32_t*                    pStartOffsetsCbCrc;
    uint32_t*                    pStartOffsetsTbCrc;
    uint32_t*                    pStartOffsetsTbPayload;
    cuphyUciOnPuschOutOffsets_t* pUciOnPuschOutOffsets;

    float* pTaEsts; /*< Timing Advance: Pointer to nUes estimates in microseconds. UE ordering identical to input UE ordering
                        in pUePrms within cuphyPuschCellGrpDynPrm_t */

    float* pRssi; /*< RSSI: Pointer to nUeGrps estimates in dB. Per UE group total power (signal + noise + interference)
                      averaged over allocated PRBs, DMRS additional positions and summed over Rx antenna */

    float* pRsrp;  /*< RSRP: Pointer to nUes RSRP estimates in dB. Per UE signal power averaged over allocated PRBs,
                       DMRS additional positions, Rx antenna and summed over layers */

    float* pNoiseVarPreEq; /*< Pointer to nUes/nUeGrps pre-equalizer noise variance estimates in dB. The reported value is per-UE or per-UE-group depending on whether build option ENABLE_PUSCH_PER_UE_PREQ_NOISE_VAR is enabled or not (default enabled) */
    float* pNoiseVarPostEq; /// Pointer to nUes post equalizer noise variance estimates in dB.

    float* pSinrPreEq; /// Pointer to nUes pre-equalizer SINR estimates in dB
    float* pSinrPostEq; /// Pointer to nUes post-equalizer estimates SINR in dB

    float* pCfoHz;        /// Pointer to nUes CFO estimates in Hz

    /// HARQ/CSI part 1/CSI part 2 detection status. Refer to SCF FAPIv10.04
    uint8_t*  HarqDetectionStatus; /*< Value: 1 = CRC Pass, 2 = CRC Failure, 3 = DTX, 4 = No DTX (indicates UCI detection). Note that FAPI also defined value 5 to be "DTX not checked", which is not considered in cuPHY since DTX detection is present. */

    uint8_t*  CsiP1DetectionStatus; /*< Value: 1 = CRC Pass, 2 = CRC Failure, 3 = DTX, 4 = No DTX (indicates UCI detection). Note that FAPI also defined value 5 to be "DTX not checked", which is not considered in cuPHY since DTX detection is present. */

    uint8_t*  CsiP2DetectionStatus; /*< Value: 1 = CRC Pass, 2 = CRC Failure, 3 = DTX, 4 = No DTX (indicates UCI detection). Note that FAPI also defined value 5 to be "DTX not checked", which is not considered in cuPHY since DTX detection is present. */

    uint8_t* pPreEarlyHarqWaitKernelStatusGpu;  /// Pointer to device memory which holds the status of preEarlyHarqWaitKernel
    uint8_t* pPostEarlyHarqWaitKernelStatusGpu; /// Pointer to device memory which holds the status of postEarlyHarqWaitKernel

    float2* pChannelEsts; /*< Channel Estimates: Pointer to H matrix estimates, all UE groups concatenated.
                              Physical memory layout per group: (nDmrsEstimates, nSubcarriers, nRxAnt, nLayers) in row-major order.
                              Note: PyAerial transposes this to (nRxAnt, nLayers, nSubcarriers, nDmrsEstimates) for user convenience.
                              Use pChannelEstSizes to determine per-group data size in elements */

    uint32_t* pChannelEstSizes; /*< Channel Estimate Sizes: Array of nUeGrps elements, each containing the
                                     corresponding UE group's channel estimate data size in elements (float2) */

} cuphyPuschDataOut_t;

/**
 * PUSCH bidirectional (in/out) data buffers
 */
typedef struct
{
    uint8_t** pHarqBuffersInOut;  /*< pointer to array of In/Out HARQ buffers
                                      The In/Out HARQ buffers will be read or written depending on ndi and TB CRC pass result
                                      The In/Out HARQ buffers themselves are located in GPU memory
                                      The “array of pointers” must be read-able from a GPU kernel.  An allocation from cudaHostAlloc with cudaHostAllocPortable | cudaHostAllocMapped is sufficient. */
                                      
    float**   pFoCompensationBuffersInOut; /*< Weighted average CFO estimation buffers */
} cuphyPuschDataInOut_t;

/**
 * PUSCH pipeline dynamic parameters
 */
typedef struct _cuphyPuschDynPrms
{
    /// CUDA stream on which phase1/phase2 launched - only consumed when running setup for phase1
    cudaStream_t phase1Stream;
    cudaStream_t phase2Stream;

    /*< Setup Phases: PUSCH_SETUP_PHASE_1 – calculate HARQ buffer sizes PUSCH_SETUP_PHASE_2 – perform rest of the setup */
    cuphyPuschSetupPhase_t setupPhase;

    /// Control parameters
    uint64_t                   procModeBmsk; /// Processing modes
    uint16_t                   waitTimeOutPreEarlyHarqUs;       /// time-out threshold for wait kernel prior to starting early HARQ processing
    uint16_t                   waitTimeOutPostEarlyHarqUs;      /// time-out threshold for wait kernel after finishing early HARQ processing

    cuphyPuschCellGrpDynPrm_t const* pCellGrpDynPrm;

    /// Data parameters
    cuphyPuschDataIn_t const* pDataIn;
    cuphyPuschDataOut_t*      pDataOut;
    cuphyPuschDataInOut_t*    pDataInOut;
    uint8_t                   cpuCopyOn; /// Flag. Indicates if reciever output copied to cpu.

    /// status parameter
    cuphyPuschStatusOut_t* pStatusOut;

    /// Debug parameters
    cuphyPuschDynDbgPrms_t*   pDbg;
} cuphyPuschDynPrms_t;

/// Batching configuration parameters
struct cuphyPuschBatchPrm;
typedef struct cuphyPuschBatchPrm* cuphyPuschBatchPrmHndl_t;

//-----------------------------------------------------------------------------------------------------------
// Functions

/******************************************************************/ /**
 * \brief Allocates and initializes a cuPHY PUSCH pipeline
 *
 * Allocates a cuPHY PUSCH receiver pipeline and returns a handle in the
 * address provided by the caller.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pPuschRxHndl and/or \p pStatPrms is NULL.
 *
 * Returns ::CUPHY_STATUS_ALLOC_FAILED if a PuschRx object cannot be allocated
 *
 * Returns ::CUPHY_STATUS_SUCCESS if allocation and initialization were
 * successful.
 *
 * \param pPuschRxHndl - Address to return the new PuschRx instance
 * \param pStatPrms    - Pointer to PUSCH static parameters to be used in pipeline creation
 * \param cuStream     - CUDA stream used for creation time work (e.g static tensor copy, conversion)
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_ALLOC_FAILED,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphySetupPuschRx,::cuphyRunPuschRx,::cuphyDestroyPuschRx
 */
cuphyStatus_t CUPHYWINAPI cuphyCreatePuschRx(cuphyPuschRxHndl_t* pPuschRxHndl, cuphyPuschStatPrms_t const* pStatPrms, cudaStream_t cuStream);

#if 0
/******************************************************************/ /**
 * \brief Return pointer to cuphyMemoryFootprint of cuPHY PUSCH
 *
 * Return pointer to cuphyMemoryFootprint of cuPHY PUSCH receiver pipeline which tracks
 * the size of GPU memory allocations owned by that object.
 *
 * Returns nullptr if \p puschRxHndl is NULL
 *
 * \param puschRxHndl  - Handle of PuschRx instance
 *
 * \return
 * \p to cuphyMemoryFootprint
 *
 */
const void*  cuphyGetMemoryFootprintTrackerPuschRx(cuphyPuschRxHndl_t puschRxHndl);
#endif

/******************************************************************/ /**
 * \brief Allocate a container for PUSCH batch parameters
 *
 * Allocate storage to hold PUSCH batch parameters and return a handle in the
 * address provided by the caller.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pBatchPrmHndl is NULL.
 *
 * Returns ::CUPHY_STATUS_ALLOC_FAILED if a batch parameter container object cannot be allocated
 *
 * Returns ::CUPHY_STATUS_SUCCESS if allocation were successful.
 *
 * \param pBatchPrmHndl - Address to return the container for PUSCH batch parameters
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_ALLOC_FAILED,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyDestroyPuschBatchPrm
 */
cuphyStatus_t CUPHYWINAPI cuphyCreatePuschBatchPrm(cuphyPuschBatchPrmHndl_t* pBatchPrmHndl);

/******************************************************************/ /**
 * \brief Destroys container for PUSCH batch parameters
 *
 * Destroy a cuPHY context object that was previously created by a call
 * to ::cuphyCreatePuschBatchPrm. The handle provided to this function should
 * not be used for any operations after this function returns.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p batchPrmHndl is NULL.
 *
 * Returns ::CUPHY_STATUS_SUCCESS if destruction was successful.
 *
 * \param batchPrmHndl - previously allocated PUSCH batch parameter instance
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyCreatePuschBatchPrm
 */
cuphyStatus_t CUPHYWINAPI cuphyDestroyPuschBatchPrm(cuphyPuschBatchPrmHndl_t batchPrmHndl);

#if 0 //added so as not to break doxygen
/******************************************************************/ /**
 * \brief Configure/Reconfigure cuPHY PUSCH pipeline
 *
 * Configures a cuPHY PUSCH receiver pipeline and returns a handle in the
 * address provided by the caller.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pPuschRxHndl and/or \p pStatPrms is NULL.
 *
 * Returns ::CUPHY_STATUS_ALLOC_FAILED if a PuschRx object cannot be allocated
 *
 * Returns ::CUPHY_STATUS_SUCCESS if allocation and initialization were
 * successful.
 *
 * \param pPuschRxHndl - Address to return the new PuschRx instance
 * \param pStatPrms  - Pointer to PUSCH static parameters to be used in pipeline configuration
 * \param pQuasiStatPrms - Pointer to PUSCH quasi-static parameters to be used in pipeline configuration
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_ALLOC_FAILED,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphySetupPuschRx,::cuphyRunPuschRx,::cuphyDestroyPuschRx
 */
// cuphyStatus_t CUPHYWINAPI cuphyConfigPuschRx(cuphyPuschRxHndl_t* pPuschRxHndl, cuphyPuschStatPrms_t const* pStatPrms, cuphyPuschQuasiStatPrms_t const* pQuasiStatPrms);
#endif

/******************************************************************/ /**
 * \brief Batch PUSCH workoad
 *
 * Batch PUSCH workload across one or more cells, UE-groups and UEs.
 * The batched configuration is used during slot execution
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p puschRxHndl and/or \p pDynPrms (or its components)
 * and/or batchPrmHndl is NULL
 *
 * Returns ::CUPHY_STATUS_SUCCESS if setup is successful.
 *
 * \param puschRxHndl  - Handle of PuschRx instance to be setup
 * \param pDynPrms     - Dynamic parameters carrying information needed for slot processing
 * \param batchPrmHndl - Workload batching information
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyCreatePuschRx,::cuphyRunPuschRx,::cuphyDestroyPuschRx
 */
cuphyStatus_t CUPHYWINAPI cuphyBatchPuschRx(cuphyPuschRxHndl_t puschRxHndl, cuphyPuschDynPrms_t* pDynPrms, cuphyPuschBatchPrmHndl_t batchPrmHndl);

/******************************************************************/ /**
 * \brief Setup cuPHY PUSCH pipeline for slot processing
 *
 * Setup cuPHY PUSCH receiver pipeline (and its components) state in
 * preparation towards slot execution
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p puschRxHndl and/or \p pDynPrms
 * and/or batchPrmHndl is NULL
 *
 * Returns ::CUPHY_STATUS_SUCCESS if allocation and initialization were
 * successful.
 *
 * \param puschRxHndl  - Handle of PuschRx instance to be setup
 * \param pDynPrms     - Dynamic parameters carrying information needed for slot processing
 * \param batchPrmHndl - Workload batching information
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyCreatePuschRx,::cuphyRunPuschRx,::cuphyDestroyPuschRx
 */
cuphyStatus_t CUPHYWINAPI cuphySetupPuschRx(cuphyPuschRxHndl_t puschRxHndl, cuphyPuschDynPrms_t* pDynPrms, cuphyPuschBatchPrmHndl_t const batchPrmHndl);

/******************************************************************/ /**
 * \brief Run cuPHY PUSCH pipeline processing in specified mode
 *
 * Call triggers cuPHY PUSCH receiver pipeline execution in mode specified
 * by procModeBmsk
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p puschRxHndl is NULL and/or
 * procModeBmsk is not supported.
 *
 * Returns ::CUPHY_STATUS_SUCCESS if PuschRx execution is successful.
 *
 * \param puschRxHndl  - Handle of PuschRx instance which is to be triggered
 * \param runPhase     - Run phase for cuPHY PUSCH pipeline
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyCreatePuschRx,::cuphySetupPuschRx,::cuphyDestroyPuschRx
 */
cuphyStatus_t CUPHYWINAPI cuphyRunPuschRx(cuphyPuschRxHndl_t puschRxHndl, cuphyPuschRunPhase_t runPhase);

/******************************************************************/ /**
 * \brief Run cuPHY save Pusch debug buffer
 * \param puschRxHndl - Handle of PuschRx instance which saves the debug buffer
 * \param cuStream   - CUDA stream used for PuschRx pipeline execution
 * Note: requires stream synchronization durring call
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 */
cuphyStatus_t CUPHYWINAPI cuphyWriteDbgBufSynch(cuphyPuschRxHndl_t puschRxHndl, cudaStream_t cuStream);

/******************************************************************/ /**
 * \brief Destroys a cuPHY PUSCH receiver pipeline object
 *
 * Destroys a cuPHY PUSCH receiver pipeline object that was previously
 * created by ::cuphyCreatePuschRx. The handle provided to this function
 * should not be used for any operations after this function returns.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p puschRxHndl is NULL.
 *
 * Returns ::CUPHY_STATUS_SUCCESS if destruction was successful.
 *
 * \param puschRxHndl - handle to previously allocated PuschRx instance
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyCreatePuschRx
 */
cuphyStatus_t CUPHYWINAPI cuphyDestroyPuschRx(cuphyPuschRxHndl_t puschRxHndl);

/** @} */ /* END CUPHY_PUSCH_RECEIVER */

// Following snippet goes into a source file
#if 0
struct cuphyPuschBatchPrm
{
    // nBatchTypes - Number of configurations to be batched

    // Type-Length-IdxValue
    // batchTypeX      - Batch type identifier
    // batchTypeXSz    - Batch size of type batchTypeX
    // batchTypeValues - Batch type batchTypeX indices 0:batchTypeXSz-1

    // | nBatchTypes = N+1 |
    // | batchType0  | batchType0Sz | ueGrpIdx[0] | ueGrpIdx[1] | ... | ueGrpIdx[batchType0Sz-1]
    // | batchType1  | batchType1Sz | ueGrpIdx[0] | ueGrpIdx[1] | ... | ueGrpIdx[batchType1Sz-1]
    // ...
    // | batchTypeN  | batchTypeNSz | ueGrpIdx[0] | ueGrpIdx[1] | ... | ueGrpIdx[batchTypeNSz-1]
    uint16_t* pChEstBatchPrms;
    uint16_t* pChEqBatchPrms;

    // | nBatchTypes = N+1 |
    // | batchType0  | batchType0Sz | ueIdx[0] | ueIdx[1] | ... | ueIdx[batchType0Sz-1]
    // | batchType1  | batchType1Sz | ueIdx[0] | ueIdx[1] | ... | ueIdx[batchType1Sz-1]
    // ...
    // | batchTypeN  | batchTypeNSz | ueIdx[0] | ueIdx[1] | ... | ueIdx[batchTypeNSz-1]
    uint16_t* pLdpcBatchPrms;
};
typedef struct cuphyPuschBatchPrm cuphyPuschBatchPrm_t;
#endif
// End source file snippet

/**
 * \defgroup CUPHY_PUCCH_RECEIVER  PUCCH Receiver
 *
 * This section describes the PUCCH receive pipeline functions of the cuPHY
 * application programming interface.
 *
 * @{
 */

struct cuphyPucchRx;
/**
 * cuPHY PUCCH Receiver handle
 */
typedef struct cuphyPucchRx* cuphyPucchRxHndl_t;

/// PUCCH processing modes
typedef enum _cuphyPucchProcMode
{
    PUCCH_PROC_MODE_FULL_SLOT = 0x0,        /*!< stream processing */
    PUCCH_PROC_MODE_FULL_SLOT_GRAPHS = 0x1, /*!< graph processing */
} cuphyPucchProcMode_t;

//-----------------------------------------------------------------------------------------------------------
// PUCCH Static Parameters

/**
 * PUCCH debug parameters
 */
typedef struct _cuphyPucchDbgPrms
{
    const char* pOutFileName;           /*!< output file capturing pipeline intermediate states. No capture if null. */
    uint8_t     enableDynApiLogging;    /*!< control the API logging of PUCCH dynamic parameters */
    uint8_t     enableStatApiLogging;   /*!< control the API logging of PUCCH static parameters */
} cuphyPucchDbgPrms_t;

// Cell-group API
/**
 * PUCCH Static parameters, shall remain constant throughout the pipeline lifecycle
 */
typedef struct _cuphyPucchStatPrms
{
    cuphyTracker_t*     pOutInfo;      /*!< pointer to cuphyTracker_t. Its pMemoryFootprint pointer will be updated by cuPHY in channel objection creation. The caller will thus have access to the cuphyMemoryFootprint object of that cuPHY PUCCH channel object. The cuphyMemoryFootprint object tracks the size of the GPU memory allocations owned by that cuPHY PUCCH channel object. */

    // Cell specific static parameters
    uint16_t            nMaxCells;     /*!< Total # of cell configurations supported by the pipeline during its lifetime */
    cuphyCellStatPrm_t* pCellStatPrms; /*!< Per cell static parameters */

    // 
    // 
    uint16_t nMaxCellsPerSlot;  /*!< Maximum # of cells scheduled in a slot. 
                                    Out of nMaxCells, the nMaxCellsPerSlot most resource hungry cells are used for resource provisioning purposes
                                    nMaxCellsPerSlot <= nMaxCells */

    uint8_t uciOutputMode; /*!< 0 --> decoded UCI segment1 outputed in a single buffer \n
                                1 --> decoded UCI segment1 seperated into three buffers (HARQ, SR, CSI-P1) */

    uint8_t polarDcdrListSz; /*!<  Polar decoder list size \n
                                   default size for PUCCH to be set to 8 */
    
    
    uint8_t enableUlRxBf;      /*!< Enable UL RX beamforming */


    uint8_t enableBatchedMemcpy;  /*!< Enable batched memory copy. Values: 0->1. */

    cuphyPucchDbgPrms_t* pDbg; /*!< Debug parameters */
} cuphyPucchStatPrms_t;

//-----------------------------------------------------------------------------------------------------------
// PUCCH Dynamic Parameters

/**
 * Per cell PUCCH dynamic parameter
 */
typedef struct _cuphyPucchCellDynPrm
{
    uint16_t cellPrmStatIdx; /*!<  Index to cell-static parameter information */
    uint16_t cellPrmDynIdx;  /*!<  Index to cell-dynamic parameter information */
    uint16_t slotNum;        /*!<  Slot number */
    uint16_t pucchHoppingId; /*!<  Hopping ID */

} cuphyPucchCellDynPrm_t;

/**
 * PUCCH Cell group dynamic parameters for all formats
 */
typedef struct _cuphyPucchCellGrpDynPrm
{
    // Cell group parameters
    uint16_t                nCells;    /*!< # of cells to be batch processed */
    cuphyPucchCellDynPrm_t* pCellPrms; /*!< Array of per cell dynamic parameters of length \p nCells */

    // UE parameters
    uint16_t                nF0Ucis;    /*!< Number of Format 0 UCIs */
    cuphyPucchUciPrm_t*     pF0UciPrms; /*!< Array of per-UCI PUCCH Format 0 parameters */

    uint16_t                nF1Ucis;    /*!< Number of Format 1 UCIs */
    cuphyPucchUciPrm_t*     pF1UciPrms; /*!< Array of per-UCI PUCCH Format 1 parameters */

    uint16_t                nF2Ucis;    /*!< Number of Format 2 UCIs */
    cuphyPucchUciPrm_t*     pF2UciPrms; /*!< Array of per-UCI PUCCH Format 2 parameters */

    uint16_t                nF3Ucis;    /*!< Number of Format 3 UCIs */
    cuphyPucchUciPrm_t*     pF3UciPrms; /*!< Array of per-UCI PUCCH Format 3 parameters */

    uint16_t                nF4Ucis;    /*!< Number of Format 4 UCIs */
    cuphyPucchUciPrm_t*     pF4UciPrms; /*!< Array of per-UCI PUCCH Format 4 parameters */
} cuphyPucchCellGrpDynPrm_t;

/**
 * PUCCH input data
 */
typedef struct _cuphyPucchDataIn
{
    cuphyTensorPrm_t* pTDataRx; /*!< array of tensors with each tensor (indexed by cellPrmDynIdx) representing the receive slot buffer of a cell in the cell group \n
                                     Each cell's tensor may have a different geometry */
} cuphyPucchDataIn_t;

/**
 * PUCCH output data. The UE ordering in buffers is identical to input UCI parameter (pFxUciPrms within cuphyPucchCellGrpDynPrm_t) input ordering
 */
typedef struct _cuphyPucchDataOut
{
    cuphyPucchF0F1UciOut_t* pF0UcisOut; /*!< pointer to buffers containing F0 UCI output with ordering identical to input ordering within pF0UciPrms in cuphyPucchCellGrpDynPrm_t, dim:nF0Ucis */
    cuphyPucchF0F1UciOut_t* pF1UcisOut; /*!< pointer to buffers containing F1 UCI output with ordering identical to input ordering within pF1UciPrms in cuphyPucchCellGrpDynPrm_t, dim:nF1Ucis */

    cuphyPucchF234OutOffsets_t* pPucchF2OutOffsets; /*!< pointer to buffers containing offset information for F2 UCI output with ordering identical to input ordering within pF2UciPrms in cuphyPucchCellGrpDynPrm_t, dim:nF2Ucis */
    cuphyPucchF234OutOffsets_t* pPucchF3OutOffsets; /*!< pointer to buffers containing offset information for F3 UCI output with ordering identical to input ordering within pF3UciPrms in cuphyPucchCellGrpDynPrm_t, dim:nF3Ucis */
    cuphyPucchF234OutOffsets_t* pPucchF4OutOffsets; /*!< pointer to buffers containing offset information for F4 UCI output with ordering identical to input ordering within pF4UciPrms in cuphyPucchCellGrpDynPrm_t, dim:nF4Ucis */

    uint8_t*  pUciPayloads; /*!< pointer to buffer containing UCI decoded payload bits for F2, F3, F4 with offset specified by pPucchF2OutOffsets, pPucchF2OutOffsets, pPucchF2OutOffsets */
    uint8_t*  pCrcFlags;    /*!< Pointer to CRC results per UCI for F2, F3, F4 */
    // uint8_t*  pDtxFlags;
    // For the following definitions of RSSI and SINR, refer to SCF FAPIv10.04, Table 3–121 UCI PUCCH format 2, 3 or 4 PDU
    // Note that pInterf (interference level) is not a SCF FAPI defined field
    float*    pRssi;        /*!< Rssi reported in dB */
    float*    pSinr;        /*!< Sinr reported in dB */
    float*    pInterf;      /*!< interference level is not a SCF FAPI defined field. Here we adopt a value setting as follows: report is in dB */
    float*    pRsrp;        /*!< Rsrp reported in dB */
    float*    pTaEst;       /*!< Timing advance reported in uS */
    uint16_t* pNumCsi2Bits; /*!< Array containing per-UCI number of CSI-P2 bits */

    // HARQ/CSI part 1/CSI part 2 detection status. Refer to SCF FAPIv10.04
    // Lengths of the following arrays are equal to the number of PF2 and PF3 UCIs
    uint8_t*  HarqDetectionStatus;  /*!< Value: 1 = CRC Pass, 2 = CRC Failure, 3 = DTX, 4 = No DTX (indicates UCI detection). Note that FAPI also defined value 5 to be "DTX not checked", which is not considered in cuPHY since DTX detection is present. */
    uint8_t*  CsiP1DetectionStatus; /*!< Value: 1 = CRC Pass, 2 = CRC Failure, 3 = DTX, 4 = No DTX (indicates UCI detection). Note that FAPI also defined value 5 to be "DTX not checked", which is not considered in cuPHY since DTX detection is present. */
    uint8_t*  CsiP2DetectionStatus; /*!< Value: 1 = CRC Pass, 2 = CRC Failure, 3 = DTX, 4 = No DTX (indicates UCI detection). Note that FAPI also defined value 5 to be "DTX not checked", which is not considered in cuPHY since DTX detection is present. */
} cuphyPucchDataOut_t;

/**
 * PUCCH Dynamic Parameters
 */
typedef struct _cuphyPucchDynPrms
{

    cudaStream_t cuStream; /*!< CUDA stream on which pipeline is launched */

    // Control parameters
    uint64_t                         procModeBmsk; /*!< Processing modes */
    cuphyPucchCellGrpDynPrm_t const* pCellGrpDynPrm; /*!< Cell group dynamic parameter inputs */

    // Data parameters
    cuphyPucchDataIn_t const* pDataIn;   /*!< Input sample data */
    cuphyPucchDataOut_t*      pDataOut;  /*!< Processed result outputs */
    uint8_t                   cpuCopyOn; /*!< Flag. Indicates if reciever output copied to cpu. */

    /// status parameter
    cuphyPucchStatusOut_t* pStatusOut;

    cuphyPucchDbgPrms_t* pDbg; /*!< Debug parameters */
} cuphyPucchDynPrms_t;

// Batching configuration parameters
struct cuphyPucchBatchPrm;
/// @brief Handle for batching - TODO currently unused/supported
typedef struct cuphyPucchBatchPrm* cuphyPucchBatchPrmHndl_t;

//-----------------------------------------------------------------------------------------------------------
// Functions

/******************************************************************/ /**
 * \brief Allocates and initializes a cuPHY PUCCH pipeline
 *
 * Allocates a cuPHY PUCCH receiver pipeline and returns a handle in the
 * address provided by the caller.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pPucchRxHndl and/or \p pStatPrms is NULL.
 *
 * Returns ::CUPHY_STATUS_ALLOC_FAILED if a PucchRx object cannot be allocated
 *
 * Returns ::CUPHY_STATUS_SUCCESS if allocation and initialization were
 * successful.
 *
 * \param pPucchRxHndl - Address to return the new PucchRx instance
 * \param pStatPrms    - Pointer to PUCCH static parameters to be used in pipeline creation
 * \param cuStream     - CUDA stream used for creation time work (e.g static tensor copy, conversion)
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_ALLOC_FAILED,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphySetupPuschRx,::cuphyRunPuschRx,::cuphyDestroyPuschRx
 */
cuphyStatus_t CUPHYWINAPI cuphyCreatePucchRx(cuphyPucchRxHndl_t* pPucchRxHndl, cuphyPucchStatPrms_t const* pStatPrms, cudaStream_t cuStream);

#if 0
/******************************************************************/ /**
 * \brief Return pointer to cuphyMemoryFootprint of cuPHY PUCCH
 *
 * Return pointer to cuphyMemoryFootprint of cuPHY PUCCH receiver pipeline which tracks
 * the size of GPU memory allocations owned by that object.
 *
 * Returns nullptr if \p pucchRxHndl is NULL
 *
 * \param pucchRxHndl  - Handle of PucchRx instance
 *
 * \return
 * \p to cuphyMemoryFootprint
 *
 */
const void*  cuphyGetMemoryFootprintTrackerPucchRx(cuphyPucchRxHndl_t pucchRxHndl);
#endif

#if 0 //added so as not to break doxygen
/******************************************************************/ /**
 * \brief Configure/Reconfigure cuPHY PUCCH pipeline
 *
 * Configures a cuPHY PUCCH receiver pipeline and returns a handle in the
 * address provided by the caller.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pPucchRxHndl and/or \p pStatPrms is NULL.
 *
 * Returns ::CUPHY_STATUS_ALLOC_FAILED if a PucchRx object cannot be allocated
 *
 * Returns ::CUPHY_STATUS_SUCCESS if allocation and initialization were
 * successful.
 *
 * \param pPucchRxHndl   - Address to return the new PucchRx instance
 * \param pStatPrms      - Pointer to PUCCH static parameters to be used in pipeline configuration
 * \param pQuasiStatPrms - Pointer to PUCCH quasi-static parameters to be used in pipeline configuration
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_ALLOC_FAILED,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphySetupPucchRx,::cuphyRunPucchRx,::cuphyDestroyPucchRx
 */
// cuphyStatus_t CUPHYWINAPI cuphyConfigPucchRx(cuphyPucchRxHndl_t* pPucchRxHndl, cuphyPucchStatPrms_t const* pStatPrms, cuphyPucchQuasiStatPrms_t const* pQuasiStatPrms);
#endif

/******************************************************************/ /**
 * \brief Setup cuPHY PUCCH pipeline for slot processing
 *
 * Setup cuPHY PUCCH receiver pipeline (and its components) state in
 * preparation towards slot execution
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pucchRxHdnl and/or \p pDynPrms
 * and/or batchPrmHndl is NULL
 *
 * Returns ::CUPHY_STATUS_SUCCESS if allocation and initialization were
 * successful.
 *
 * \param pucchRxHndl  - Handle of PucchRx instance to be setup
 * \param pDynPrms     - Dynamic parameters carrying information needed for slot processing
 * \param batchPrmHndl - Workload batching information
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyCreatePucchRx,::cuphyRunPucchRx,::cuphyDestroyPucchRx
 */
cuphyStatus_t CUPHYWINAPI cuphySetupPucchRx(cuphyPucchRxHndl_t pucchRxHndl, cuphyPucchDynPrms_t* pDynPrms, cuphyPucchBatchPrmHndl_t const batchPrmHndl);

/******************************************************************/ /**
 * \brief Run cuPHY PUCCH pipeline processing in specified mode
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pucchRxHdnl is NULL
 *
 * Returns ::CUPHY_STATUS_SUCCESS if PucchRx execution is successful.
 *
 * \param pucchRxHndl  - Handle of PucchRx instance which is to be triggered
 * \param procModeBmsk - Note: \p procModeBmsk is unused here.  It is set as a dynamic parameter TODO: Remove
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyCreatePucchRx,::cuphySetupPucchRx,::cuphyDestroyPucchRx
 */
cuphyStatus_t CUPHYWINAPI cuphyRunPucchRx(cuphyPucchRxHndl_t pucchRxHndl, uint64_t procModeBmsk);

/******************************************************************/ /**
 * \brief Run cuPHY save Pucch debug buffer
 *
 * \param pucchRxHndl - Handle of PucchRx instance which saves the debug buffer
 * \param cuStream   - CUDA stream used for PucchRx pipeline execution
 * Note: requires stream synchronization durring call
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 * Note: requires stream synchronization durring call
 */
cuphyStatus_t CUPHYWINAPI cuphyWriteDbgBufSynchPucch(cuphyPucchRxHndl_t pucchRxHndl, cudaStream_t cuStream);

/******************************************************************/ /**
 * \brief Destroys a cuPHY PUCCH receiver pipeline object
 *
 * Destroys a cuPHY PUCCH receiver pipeline object that was previously
 * created by ::cuphyCreatePucchRx. The handle provided to this function
 * should not be used for any operations after this function returns.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pucchRxHndl is NULL.
 *
 * Returns ::CUPHY_STATUS_SUCCESS if destruction was successful.
 *
 * \param pucchRxHndl - handle to previously allocated PucchRx instance
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyCreatePucchRx
 */
cuphyStatus_t CUPHYWINAPI cuphyDestroyPucchRx(cuphyPucchRxHndl_t pucchRxHndl);

/** @} */ /* END CUPHY_PUCCH_RECEIVER */

/**
 * \defgroup CUPHY_SRS_RECEIVER  SRS Receiver
 *
 * This section describes the SRS receive pipeline functions of the cuPHY
 * application programming interface.
 *
 * @{
 */

struct cuphySrsRx;
/**
 * cuPHY SRS Receiver handle
 */
typedef struct cuphySrsRx* cuphySrsRxHndl_t;

// SRS processing modes
typedef enum _cuphySrsProcMode
{
    SRS_PROC_MODE_FULL_SLOT        = 0x0, // Full slot processing
    SRS_PROC_MODE_FULL_SLOT_GRAPHS = 0x1, // Full slot processing with graphs
    SRS_MAX_PROC_MODES
} cuphySrsProcMode_t;

//-----------------------------------------------------------------------------------------------------------
// SRS Static Parameters

typedef struct _cuphySrsDbgPrms // TODO: deleate after CP removes reference.
{
    const char* pOutFileName; // output file capturing pipeline intermediate states. No capture if null.
} cuphySrsDbgPrms_t;

typedef struct _cuphySrsStatDbgPrms
{
    const char* pOutFileName;     // output file capturing pipeline intermediate states. No capture if null.
    uint8_t     enableApiLogging; // control the API logging of SRS static parameters
} cuphySrsStatDbgPrms_t;

// Cell-group API
typedef struct _cuphySrsStatPrms
{
    cuphySrsFilterPrms_t srsFilterPrms;

    // Cell specific static parameters
    uint16_t            nMaxCells;     // Total # of cell configurations supported by the pipeline during its lifetime
    cuphyCellStatPrm_t* pCellStatPrms; //

    // Maximum # of cells scheduled in a slot. Out of nMaxCells, the nMaxCellsPerSlot most resource hungry cells are
    // used for resource provisioning purposes
    uint16_t nMaxCellsPerSlot;  // nMaxCellsPerSlot <= nMaxCells

    // Debug parameters
    cuphySrsDbgPrms_t*     pDbg;    // TODO: DELEATE.
    cuphySrsStatDbgPrms_t* pStatDbg;

    // channel estimation algorithm selection: 0 - MMSE, 1 - RKHS
    cuphySrsChEstAlgoType_t chEstAlgo; 
    
    // normalization/scaling algorithm selection: 0 - constant scaling, 1 - peak value normalization
    uint8_t chEstToL2NormalizationAlgo;
    
    // constant scaling factor
    float  chEstToL2ConstantScaler;
    
    // delay offset correction enable/disable flag: 0 - disable, 1 - enable
    uint8_t enableDelayOffsetCorrection; 

    // RKHS paramaters
    cuphySrsRkhsPrms_t* pSrsRkhsPrms;

    // memory tracker
    cuphyTracker_t* pOutInfo;      /*!< pointer to cuphyTracker_t. Its pMemoryFootprint pointer will be updated by cuPHY in channel objection creation. The caller will thus have access to the cuphyMemoryFootprint object of that cuPHY SRS channel object. The cuphyMemoryFootprint object tracks the size of the GPU memory allocations owned by that cuPHY SRS channel object. */

    uint8_t enableBatchedMemcpy;  /*!< Enable batched memory copy. Values: 0->1. */
} cuphySrsStatPrms_t;

//-----------------------------------------------------------------------------------------------------------
// SRS Dynamic Parameters

// API logging
typedef struct _cuphySrsDynDbgPrms
{
    uint8_t enableApiLogging; // control the API logging of SRS dynamic parameters
} cuphySrsDynDbgPrms_t;


// Per cell dynamic parameter
typedef struct _cuphySrsCellDynPrm
{
    uint16_t cellPrmStatIdx; // Index to cell-static parameter information
    uint16_t cellPrmDynIdx;  // Index to cell-dynamic parameter information
    uint16_t slotNum;
    uint16_t frameNum;

    uint8_t srsStartSym;   // starting srs symbol (for all users in the cell)
    uint8_t nSrsSym;       // number of srs symbols (for all users in the cell)
} cuphySrsCellDynPrm_t;

// Cell group dynamic parameters
typedef struct _cuphySrsCellGrpDynPrm
{
    // Cell group parameters
    uint16_t              nCells; // # of cells to be batch processed
    cuphySrsCellDynPrm_t* pCellPrms;

    // SRS user parameters
    uint16_t         nSrsUes;
    cuphyUeSrsPrm_t* pUeSrsPrms;

} cuphySrsCellGrpDynPrm_t;

typedef struct _cuphySrsDataIn
{
    cuphyTensorPrm_t* pTDataRx; // array of tensors with each tensor (indexed by cellPrmDynIdx) representing the received SRS symbols of a cell in the cell group
                                // Each cell's tensor may have a different geometry
} cuphySrsDataIn_t;

typedef struct _cuphySrsDataOut
{
    __half2*   pDataRxSrs;         // host-pinned buffer for raw SRS IQ from buf_st_2 (GPU->host copy, gated on DataLake)

    cuphySrsChEstBuffInfo_t* pChEstBuffInfo;  // array of ChEst buffers of all users
    cuphySrsReport_t*        pSrsReports;     // array containing SRS reports of all users
    cuphySrsChEstToL2_t*     pSrsChEstToL2;   // array of CPU ChEst to L2 of all users

    float*     pRbSnrBuffer;       // buffer containing RB SNRs of all users
    uint32_t*  pRbSnrBuffOffsets;  // buffer containing user offsets into pRbSnrBuffer
} cuphySrsDataOut_t;

typedef struct _cuphySrsDynPrms
{
    // CUDA stream on which pipeline is launched
    // @todo: cuPHY internally uses a CUDA stream pool to launch multiple parallel CUDA kernels from the same
    // component. So cuStream provided below is not the only stream where workload would be launched. To be
    // closed after consensus with a wider group
    cudaStream_t cuStream;

    // Control parameters
    uint64_t                       procModeBmsk; // Processing modes
    cuphySrsCellGrpDynPrm_t const* pCellGrpDynPrm;

    // Data parameters
    cuphySrsDataIn_t const* pDataIn;
    cuphySrsDataOut_t*      pDataOut;
    uint8_t                 cpuCopyOn; // Flag. Indicates if reciever output copied to cpu.

    /// status parameter
    cuphySrsStatusOut_t*    pStatusOut;

    // Debug paramaters:
    cuphySrsDynDbgPrms_t* pDynDbg;
} cuphySrsDynPrms_t;

// Batching configuration parameters
struct cuphySrsBatchPrm;
typedef struct cuphySrsBatchPrm* cuphySrsBatchPrmHndl_t;

//-----------------------------------------------------------------------------------------------------------
// Functions

/******************************************************************/ /**
 * \brief Allocates and initializes a cuPHY SRS pipeline
 *
 * Allocates a cuPHY SRS receiver pipeline and returns a handle in the
 * address provided by the caller.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pSrsRxHndl and/or \p pStatPrms is NULL.
 *
 * Returns ::CUPHY_STATUS_ALLOC_FAILED if a SrsRx object cannot be allocated
 *
 * Returns ::CUPHY_STATUS_SUCCESS if allocation and initialization were
 * successful.
 *
 * \param pSrsRxHndl   - Address to return the new PucchRx instance
 * \param pStatPrms    - Pointer to SRS static parameters to be used in pipeline creation
 * \param cuStream     - CUDA stream used for creation time work (e.g static tensor copy, conversion)
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_ALLOC_FAILED,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphySetupSrsRx,::cuphyRunSrsRx,::cuphyDestroySrsRx
 */
cuphyStatus_t CUPHYWINAPI cuphyCreateSrsRx(cuphySrsRxHndl_t* pSrsRxHndl, cuphySrsStatPrms_t const* pStatPrms, cudaStream_t cuStream);

#if 0 //added so as not to break doxygen
/******************************************************************/ /**
 * \brief Configure/Reconfigure cuPHY SRS pipeline
 *
 * Configures a cuPHY SRS receiver pipeline and returns a handle in the
 * address provided by the caller.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pSrsRxHndl and/or \p pStatPrms is NULL.
 *
 * Returns ::CUPHY_STATUS_ALLOC_FAILED if a SrsRx object cannot be allocated
 *
 * Returns ::CUPHY_STATUS_SUCCESS if allocation and initialization were
 * successful.
 *
 * \param pSrsRxHndl     - Address to return the new SrsRx instance
 * \param pStatPrms      - Pointer to SRS static parameters to be used in pipeline configuration
 * \param pQuasiStatPrms - Pointer to SRS quasi-static parameters to be used in pipeline configuration
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_ALLOC_FAILED,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphySetupSrsRx,::cuphyRunSrsRx,::cuphyDestroySrsRx
 */
// cuphyStatus_t CUPHYWINAPI cuphyConfigPucchRx(cuphySrsRxHndl_t* pSrsRxHndl, cuphySrsStatPrms_t const* pStatPrms, cuphySrsQuasiStatPrms_t const* pQuasiStatPrms);
#endif

/******************************************************************/ /**
 * \brief Setup cuPHY SRS receive pipeline for slot processing
 *
 * Setup cuPHY SRS receiver pipeline (and its components) state in
 * preparation towards slot execution
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p srsRxHdnl and/or \p pDynPrms
 * and/or batchPrmHndl is NULL
 *
 * Returns ::CUPHY_STATUS_SUCCESS if allocation and initialization were
 * successful.
 *
 * \param srsRxHndl    - Handle of SrsRx instance to be setup
 * \param pDynPrms     - Dynamic parameters carrying information needed for slot processing
 * \param batchPrmHndl - Workload batching information
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyCreateSrsRx,::cuphyRunSrsRx,::cuphyDestroySrsRx
 */
cuphyStatus_t CUPHYWINAPI cuphySetupSrsRx(cuphySrsRxHndl_t srsRxHndl, const cuphySrsDynPrms_t* pDynPrms, cuphySrsBatchPrmHndl_t batchPrmHndl);

/******************************************************************/ /**
 * \brief Run cuPHY SRS receive pipeline processing in specified mode
 *
 * Call triggers cuPHY SRS receive pipeline exeuction in mode specified
 * by procModeBmsk
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p srsRxHdnl is NULL and/or
 * procModeBmsk is not supported.
 *
 * Returns ::CUPHY_STATUS_SUCCESS if SrsRx execution is successful.
 *
 * \param srsRxHndl    - Handle of SrsRx instance which is to be triggered
 * \param procModeBmsk - Processing mode bitmask containing one or more processing modes applicable during this execution
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyCreateSrsRx,::cuphySetupSrsRx,::cuphyDestroySrsRx
 */
cuphyStatus_t CUPHYWINAPI cuphyRunSrsRx(cuphySrsRxHndl_t srsRxHndl, uint64_t procModeBmsk);

/******************************************************************/ /**
 * \brief Run cuPHY save SRS receiver debug buffer
 *
 * \param srsRxHndl  - Handle of SrsRx instance which saves the debug buffer
 * \param cuStream   - CUDA stream used for PucchRx pipeline execution
 * Note: requires stream synchronization durring call
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 * Note: requires stream synchronization durring call
 */
cuphyStatus_t CUPHYWINAPI cuphyWriteDbgBufSynchSrs(cuphySrsRxHndl_t srsRxHndl, cudaStream_t cuStream);

/******************************************************************/ /**
 * \brief Destroys a cuPHY SRS receiver pipeline object
 *
 * Destroys a cuPHY SRS receiver pipeline object that was previously
 * created by ::cuphyCreateSrsRx. The handle provided to this function
 * should not be used for any operations after this function returns.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p srsRxHndl is NULL.
 *
 * Returns ::CUPHY_STATUS_SUCCESS if destruction was successful.
 *
 * \param srsRxHndl - handle to previously allocated SrsRx instance
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyCreatePucchRx
 */
cuphyStatus_t CUPHYWINAPI cuphyDestroySrsRx(cuphySrsRxHndl_t srsRxHndl);

/** @} */ /* END CUPHY_SRS_RECEIVER */

/**
 * \defgroup CUPHY_BEAMFORMING_WEIGHT_COMPUTE Beamforming Weight computation
 *
 * This section describes the Beamforming weight computation pipeline functions of the cuPHY
 * application programming interface.
 *
 * @{
 */

struct cuphyBfwTx;
/**
 * cuPHY BfwTx handle
 */
typedef struct cuphyBfwTx* cuphyBfwTxHndl_t;

// BFW processing modes
typedef enum _cuphyBfwTxProcMode
{
    BFW_PROC_MODE_NO_GRAPH   = 0x0, // stream based workload submission
    BFW_PROC_MODE_WITH_GRAPH = 0x1, // CUDA graph based workload submission
    BFW_MAX_PROC_MODES
} cuphyBfwTxProcMode_t;

//-----------------------------------------------------------------------------------------------------------
// BFW Static Parameters

typedef struct _cuphyBfwDbgPrms
{
    const char* pOutFileName; // output file capturing pipeline intermediate states. No capture if null.
} cuphyBfwDbgPrms_t;

typedef struct _cuphyBfwStatDbgPrms
{
    const char* pOutFileName;     // output file capturing pipeline intermediate states. No capture if null.
    uint8_t     enableApiLogging; // control the API logging of BFW static parameters
} cuphyBfwStatDbgPrms_t;


typedef struct _cuphyBfwStatPrms
{
    float                  lambda;           // Regularization constant used in regularized zero-forcing beamformer
    uint8_t                nMaxGnbAnt;       // Max number of gNB antenna used for beam-forming
    uint16_t               nMaxPrbGrps;      // Max number of PRB groups supported for beam-forming
    uint16_t               nMaxUeGrps;       // Max total number of UE groups to be processed per beamforming weight compute pipeline
    uint16_t               nMaxTotalLayers;  // Maximum total beamformed layers (i.e. sum of layer count across all UE groups)
                                             // to be processed per beamforming weight compute pipeline
    uint8_t                compressBitwidth; // 0 => none, 9 => 9b, other values are not supported
    float                  beta;             // Coefficient amplitude scaling used during compression
    uint8_t                bfwPowerNormAlg_selector; // Selector of power normalization algorithm 
    uint8_t                useKernelCopy;    /*!< When 1, use kernel-based D2H copy for BFW outputs. Overrides 'enableBatchedMemcpy' for the D2H path. Applies only when Host output buffers are used. Values: 0->1. */
    cuphyBfwDbgPrms_t*     pDbg;             // Debug parameters
    cuphyBfwStatDbgPrms_t* pStatDbg;         // Debug parameters
    cuphyTracker_t*        pOutInfo;      /*!< pointer to cuphyTracker_t. Its pMemoryFootprint pointer will be updated by cuPHY in channel objection creation. The caller will thus have access to the cuphyMemoryFootprint object of that cuPHY BFW channel object. The cuphyMemoryFootprint object tracks the size of the GPU memory allocations owned by that cuPHY BFW channel object. */

    uint8_t                enableBatchedMemcpy;  /*!< When 1, enables batched memcpys where applicable. Affects: D2H of BFW outputs only when useKernelCopy==0 and Host output buffers are used; and H2D transfers during BFW CoefComp setup. When 0, fall back to individual memcpys. Values: 0->1. */
} cuphyBfwStatPrms_t;

//-----------------------------------------------------------------------------------------------------------
// BFW Dynamic Parameters

typedef struct _cuphyBfwDynDbgPrms
{
    uint8_t     enableApiLogging; // control the API logging of BFW dynamic parameters
} cuphyBfwDynDbgPrms_t;


typedef struct _cuphyBfwDynPrm
{
    uint16_t                  nUeGrps;     // Number of beamforming groups to process
    cuphyBfwUeGrpPrm_t const* pUeGrpPrms;  // Pointer to an array of configuration parameters of length nUeGrps
} cuphyBfwDynPrm_t;

typedef struct _cuphyBfwDataIn
{
    cuphySrsChEstBuffInfo_t* pChEstInfo; // pointer to an array of SRS channel estimation information
                                         // (indexed by chEstInfoBufIdx in cuphyBfwLayerPrm_t)
                                         // SRS channel estimate dimensions: nPrbGrpChEsts x nGnbAnt x nUeLayers (FAPI based)
                                         // Note: in the representation: nPrbGrpChEsts x nGnbAnt x nUeLayers, nPrbGrpChEsts is the innermost i.e. fastest changing dimension
                                         // and nUeLayers is the outermost dimension i.e. slowest changing dimension
                                         // Each SRS channel estimate tensor may have a different geometry
} cuphyBfwDataIn_t;

typedef struct _cuphyBfwDataOut
{
    uint8_t** pBfwCoef;     // Array of nUeGrps pointers each representing a buffer of CPU pinned memory for beamforming coefficients of a UE group
                            //  and indexed by coefBuffIdx in cuphyBfwPrm_t
                            //  Each buffer has geometry: nGnbAnt x nPrbGrpBfw x nLayers (as specified in cuphyBfwGrpPrm_t)
                            //  Each beamforming coefficient buffer may have different dimensions
                            // Note1: in the representation: nGnbAnt x nPrbGrpBfw x nLayers, nGnbAnt is the innermost i.e. fastest changing dimension
                            // and nLayers is the outermost dimension i.e. slowest changing dimension
                            // Note2: for compressed beamforming coefficients, the block floating point exponent is prefixed to every nGnbAnt coefficients
                            // (see field bfwCompParam in Table 7.7.11.1-1 in O-RAN.WG4.CUS.0-v10.00)
#ifdef BFW_BOTH_COMP_FLOAT
    uint8_t** pBfwCompCoef; // TODO for testing, second buffer to store BFP compressed coefficients
#endif
} cuphyBfwDataOut_t;

typedef struct _cuphyBfwDynPrms
{
    // CUDA stream on which pipeline is launched
    cudaStream_t cuStream;

    // Control parameters
    uint64_t                 procModeBmsk; // Processing modes
    cuphyBfwDynPrm_t const*  pDynPrm;

    // Data parameters
    cuphyBfwDataIn_t const* pDataIn;
    cuphyBfwDataOut_t*      pDataOut;

    // debug paramaters
    cuphyBfwDynDbgPrms_t*   pDynDbg; // Dynamic debug paramaters

    /// status parameter
    cuphyBfwStatusOut_t*    pStatusOut;

} cuphyBfwDynPrms_t;

// Batching configuration parameters
struct cuphyBfwBatchPrm;
typedef struct cuphyBfwBatchPrm* cuphyBfwBatchPrmHndl_t;

//-----------------------------------------------------------------------------------------------------------
// Functions

/******************************************************************/ /**
 * \brief Allocates and initializes a cuPHY BFW pipeline
 *
 * Allocates a cuPHY Downlink Beamforming Weight (BFW) computation pipeline for Downlink and returns a handle in the
 * address provided by the caller.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pBfwTxHndl and/or \p pStatPrms is NULL.
 *
 * Returns ::CUPHY_STATUS_ALLOC_FAILED if a BfwTx object cannot be allocated
 *
 * Returns ::CUPHY_STATUS_SUCCESS if allocation and initialization were
 * successful.
 *
 * \param pBfwTxHndl - Address to return the new BFW pipeline instance
 * \param pStatPrms  - Pointer to BFW static parameters to be used in pipeline creation
 * \param cuStream   - CUDA stream used for creation time work (e.g static tensor copy, conversion)
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_ALLOC_FAILED,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphySetupBfwTx,::cuphyRunBfwTx,::cuphyDestroyBfwTx
 */
cuphyStatus_t CUPHYWINAPI cuphyCreateBfwTx(cuphyBfwTxHndl_t* pBfwTxHndl, cuphyBfwStatPrms_t const* pStatPrms, cudaStream_t cuStream);

/******************************************************************/ /**
 * \brief Setup cuPHY BFW computation pipeline
 *
 * Setup cuPHY Downlink Beamforming Weight (BFW) computation pipeline (and its components) state in
 * preparation towards slot execution
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p bfwTxHndl and/or \p pDynPrms
 * and/or \p batchPrmHndl is NULL
 *
 * Returns ::CUPHY_STATUS_SUCCESS if allocation and initialization were
 * successful.
 *
 * \param bfwTxHndl    - Handle of BFW instance to be setup
 * \param pDynPrms     - Dynamic parameters carrying information needed for slot processing
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyCreateBfwTx,::cuphyRunBfwTx,::cuphyDestroyBfwTx
 */
cuphyStatus_t CUPHYWINAPI cuphySetupBfwTx(cuphyBfwTxHndl_t bfwTxHndl, cuphyBfwDynPrms_t* pDynPrms);

/******************************************************************/ /**
 * \brief Run cuPHY BFW computation pipeline in specified mode
 *
 * Call triggers cuPHY Downlink Beamforming Weight (BFW) computation pipeline execution in mode specified
 * by procModeBmsk
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p bfwTxHndl is NULL and/or \p procModeBmsk
 * is not supported.
 *
 * Returns ::CUPHY_STATUS_SUCCESS if BfwTx execution is successful.
 *
 * \param bfwTxHndl    - Handle of BfwTx instance which is to be triggered
 * \param procModeBmsk - Processing mode bitmask containing one or more processing modes applicable during this execution
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyCreateBfwTx,::cuphySetupBfwTx,::cuphyDestroyBfwTx
 */
cuphyStatus_t CUPHYWINAPI cuphyRunBfwTx(cuphyBfwTxHndl_t bfwTxHndl, uint64_t procModeBmsk);

/******************************************************************/ /**
 * \brief Run cuPHY save BFW computation debug buffer
 *
 * \param bfwTxHndl  - Handle of BfwTx instance which saves the debug buffer
 * \param cuStream   - CUDA stream used for BfwTx pipeline execution
 * Note: requires stream synchronization durring call
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 * Note: requires stream synchronization during call
 */
cuphyStatus_t CUPHYWINAPI cuphyWriteDbgBufSynchBfw(cuphyBfwTxHndl_t bfwTxHndl, cudaStream_t cuStream);

/******************************************************************/ /**
 * \brief Destroys a cuPHY BFW receiver pipeline object
 *
 * Destroys a cuPHY Downlink Beamforming Weight (BFW) computation object that was previously
 * created by ::cuphyCreateBfwTx. The handle provided to this function
 * should not be used for any operations after this function returns.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p bfwTxHndl is NULL.
 *
 * Returns ::CUPHY_STATUS_SUCCESS if destruction was successful.
 *
 * \param bfwTxHndl - handle to previously allocated BfwTx instance
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyCreateBfwTx,::cuphySetupBfwTx,::cuphyRunBfwTx
 */
cuphyStatus_t CUPHYWINAPI cuphyDestroyBfwTx(cuphyBfwTxHndl_t bfwTxHndl);

/** @} */ /* END CUPHY_BEAMFORMING_WEIGHT_COMPUTE */

/**
 * \defgroup CUPHY_PDSCH_TRANSMITTER PDSCH Transmitter
 *
 * This section describes the PDSCH transmit pipeline functions of the cuPHY
 * application programming interface.
 *
 * @{
 */

struct cuphyPdschTx;
/**
 * cuPHY PDSCH transmitter handle
 */
typedef struct cuphyPdschTx* cuphyPdschTxHndl_t;

/**
 * PDSCH processing modes
 */
typedef enum _cuphyPdschProcMode
{
    /** Think of it as a bitmask [...] B2 B1 B0, where B0 is the least significant bit.
     *  B0: streams (0) or graphs (1) mode
     *  B1: setup once fallback if 1; default 0
     *  B2: inter-cell kernel batching if 1; default 0. Only applicable when there are multiple cells in a cell group.
     */

    PDSCH_PROC_MODE_NO_GRAPHS           = 0, /*!< processing without CUDA GRAPHS */
    PDSCH_PROC_MODE_GRAPHS              = 1, /*!< processing with CUDA graphs */
    PDSCH_PROC_MODE_SETUP_ONCE_FALLBACK = 2, /*!< Force reset of TB-CRC buffers when running the same TV back to back with only a single setup; used to guarantee correctness. Use this mode with a bitwise or with any of the other modes (e.g., 0 to 1). Ensure value, currently 2, has only one bit set which does not overlap with any of the other modes. */
    PDSCH_INTER_CELL_BATCHING           = 4, /*!< Deprecated; has no effect. */
    PDSCH_MAX_PROC_MODES                     /*!< maximum number of processing modes supported */
} cuphyPdschProcMode_t;

//-----------------------------------------------------------------------------------------------------------
// PDSCH Static Parameters

/**
 * PDSCH debug parameters
 */
typedef struct _cuphyPdschDbgPrms
{
    char const* pCfgFileName;            /*!< name of HDF5 file that drives the DL pipeline. No file, if null. */
    uint8_t     checkTbSize;             /*!< If 1, cuPHY PDSCH will recompute TB size for initial transmission. Value: 0 or 1.*/
    bool        refCheck;                /*!< If set, compare the output of each pipeline component with the reference output from the pCfgFileName file that drives the pipeline. */
    bool        cfgIdenticalLdpcEncCfgs; /*!< Deprecated; has no effect */
} cuphyPdschDbgPrms_t;

/**
 * PDSCH static parameters
 */
typedef struct _cuphyPdschStatPrms
{
    cuphyTracker_t*     pOutInfo;      /*!< pointer to cuphyTracker_t. Its pMemoryFootprint pointer will be updated by cuPHY in channel objection creation. The caller will thus have access to the cuphyMemoryFootprint object of that cuPHY PDSCH channel object. The cuphyMemoryFootprint object tracks the size of the GPU memory allocations owned by that cuPHY PDSCH channel object. */

    uint16_t            nCells;        /*!< number of supported cells. TODO May rename to nMaxCells to be consistent with PUSCH. */
    cuphyCellStatPrm_t* pCellStatPrms; /*!< array of cell-specific static parameters with nCells elements */

    cuphyPdschDbgPrms_t* pDbg;                 /*!< array of cell-specific debug parameters with nCells elements */
    bool                 read_TB_CRC;          /*!< if true, TB crcs are read from input buffers and not computed */
    bool                 full_slot_processing; /*!< If false, all cells ran on this PdschTx will undergo: TB-CRC + CB-CRC/segmentation + LDPC encoding + rate-matching/scrambling.
                                    If true, all cells ran on this PdschTx will undergo full slot processing:  TB-CRC + CB-CRC/segmentation + LDPC encoding + rate-matching/scrambling/layer-mapping + modulation + DMRS
                                    NB: This mode is an a priori known characteristic of the cell; a cell will never switch between modes.
                                    We may consider moving this parameter to cuphyCellStatPrm_t in the future. */

    int stream_priority; /*!< CUDA stream priority for all internal to PDSCH streams. Should match the priority of CUDA stream passed in
                                    cuphyPdschDynPrms_t during setup. */

    // Max. parameters to limit overprovisioned buffer allocation
    uint16_t nMaxCellsPerSlot;    /*!< Maximum number of cells supported. nCells <= nMaxCellsPerSlot and nMaxCellsPerSlot <= PDSCH_MAX_CELLS_PER_CELL_GROUP. If 0, compile-time constant PDSCH_MAX_CELLS_PER_CELL_GROUP is used. */
    uint16_t nMaxUesPerCellGroup; /*!< Maximum number of UEs supported in a cell group, i.e., across all the cells. nMaxUesPerCellGroup <= PDSCH_MAX_UES_PER_CELL_GROUP. If 0, the compile-time constant PDSCH_MAX_UES_PER_CELL_GROUP is used. */
    uint16_t nMaxCBsPerTB;        /*!< Maximum number of CBs supported per TB; limit valid for any UE in that cell. nMaxCBsPerTb <= MAX_N_CBS_PER_TB_SUPPORTED. If 0, the compile-time constant MAX_N_CBS_PER_TB_SUPPORTED is used. */
    uint16_t nMaxPrb;             /*!< Maximum value of cuphyCellStatPrm_t.nPrbDlBwp supported by PdschTx object. nMaxPrb <= 273. If 0, 273 is used. */
    uint8_t enableBatchedMemcpy;  /*!< Enable batched memory copy. Values: 0->1. */
} cuphyPdschStatPrms_t;

//-----------------------------------------------------------------------------------------------------------
// PDSCH Dynamic Parameters

/**
 * PDSCH DRMS (Demodulation Reference Signal) parameters
 */
typedef struct _cuphyPdschDmrsPrm
{
    /** DMRS resource information */
    // uint8_t  dmrsType;        /*!< dmrs type (only support type 1)

    uint8_t  nDmrsCdmGrpsNoData; /*!< used to calculate DMRS energy (via table lookup). Value: 1->3 */

} cuphyPdschDmrsPrm_t;

struct _cuphyCsirsRrcDynPrm;

/**
 * PDSCH per-cell dynamic parameters
 */
typedef struct _cuphyPdschCellDynPrm
{
    /** CSI-RS information for current cell */
    uint16_t nCsiRsPrms;      /*!< number of CSI-RS params co-scheduled for this cell */
    uint16_t csiRsPrmsOffset; /*!< start index for this cell's nCsiRsPrms elements in the pCsiRsPrms array of cuphyPdschCellGrpDynPrm_t; all elements are allocated continuously. */

    uint16_t cellPrmStatIdx; /*!< Index to cell-static parameter information, i.e., to the pCellStatPrms array of the
                                    cuphyPdschStatPrms_t struct. */
    uint16_t cellPrmDynIdx;  /*!< Index to cell-dynamic parameter information, i.e., to the pCellPrms array of the
                                    cuphyPdschCellGrpDynPrm_t struct */
    uint16_t slotNum;        /*!< slot number. Value: 0->319. */

    /** PDSCH time domain resource allocation */
    /** The pdschStartSym, nPdschSym and dmrsSymLocBmsk fields are also added at the user group level.
     *  The current expectation is that the caller uses the UE-group fields only if nPdschSym (cell level) and dmrsSymLocBmsk (cell level)
     *  are zero. If these fields are not zero, then the cell-level fields are used, and the implementation assumes these
     *  values are identical across all UEs and all UE groups belonging to this cell.
     *
     *  TODO: Other possible design choices:
     *        * Wrap the time domain resource allocation fields in a seperate struct, and add a pointer both here
     *          and at the UE group level. Use one of the two fields depending on which pointer is not nullptr.
     *          Note: that from a memory perspective storing the pointer would likely be more expensive, as these 3
     *          fields take up only 4B.
     *
     */

    uint8_t pdschStartSym; /*!< PDSCH start symbol location (0-indexing). Value: 0->13 */

    uint8_t nPdschSym; /*!< PDSCH DMRS + data symbol count. Value: 1->14 */

    uint16_t dmrsSymLocBmsk; /*!< DMRS symbol location bitmask (least significant 14 bits are valid);
                                  A set bit i, specifies symbol i is DMRS.
                                  For example if symbols 2 and 3 are DMRS, then: dmrsSymLocBmsk = 0000 0000 0000 1100 */

    uint8_t testModel;      /*!< Specifies the cell is in testing mode if set to 1. Value: 0-1
                                 For cells in testing mode, the TB payload buffers hold bits from the PN23 (pseudorandom) sequence
                                 instead of a TB payload. */

} cuphyPdschCellDynPrm_t;

/**
 * Co-scheduled UE (User-Equipment) group parameters
 */
typedef struct _cuphyPdschUeGrpPrm
{
    cuphyPdschCellDynPrm_t* pCellPrm; /*!< Pointer to UE group's parent cell dynamic parameters */

    cuphyPdschDmrsPrm_t* pDmrsDynPrm; /*!< DMRS information */

    uint8_t  resourceAlloc; /*!< Resource Allocation Type. Value: 0-1 */
    uint8_t* rbBitmap; /*!< Pointer to Resource Allocation bitmap of MAX_RBMASK_BYTE_SIZE bytes.  Each set bit indicates the RBG is
                            allocated to the UE starting with the lowest bit on the lowest byte being RB0.
                            For example, if RB 3 (0-index) were allocated, bit 3 on byte 0 would be set (mask of 0x8)
                            and if RB 273 were allocated, bit 1 on byte 34 would be set
                            RB0 is with reference to BWPStart not CRB0
                            Also see FAPI Table 3-70 item rbBitmap (Section 3.4.2.2)
                            Note: cuPHY assumes this pointer is always valid and points to memory populated by L2
                            of at least MAX_RBMASK_BYTE_SIZE bytes*/

    /** PDSCH frequency resource allocation (contiguous)*/
    uint16_t startPrb; /*!< start PRB (0-indexing). Not valid for resourceAlloc 0 Value: 0-274  */

    uint16_t nPrb;     /*!< number of allocated PRBs.  Must be populated correctly regardless of RA Type. Value: 1-275 */

    uint16_t dmrsSymLocBmsk; /*!< DMRS symbol location bitmask (least significant 14 bits are valid);
                                  A set bit i, specifies symbol i is DMRS.
                                  For example if symbols 2 and 3 are DMRS, then: dmrsSymLocBmsk = 0000 0000 0000 1100
                                  This field will only have a valid value if the corresponding cell level field is zero. */

    /** PDSCH time domain resource allocation */
    // The pdschStartSym, nPdschSym here will only have valid values if the nPdschSym at cell level is 0.
    uint8_t pdschStartSym; /*!< PDSCH start symbol location (0-indexing). Value: 0->13 */

    uint8_t nPdschSym; /*!< PDSCH DMRS + data symbol count. Value: 1->14 */

    /** Per UE information in co-scheduled group */
    uint16_t  nUes;       /*!< number of UEs co-scheduled in this group */
    uint16_t* pUePrmIdxs; /*!< nUes element wide array; it contains indices into the pUePrms array of cuphyPdschCellGrpDynPrm_t */

    // beamforming parameters
    //**More parameters to be added when beamforming supported**

} cuphyPdschUeGrpPrm_t;

/**
 * Per UE parameters
 */
typedef struct _cuphyPdschUePrm
{
    cuphyPdschUeGrpPrm_t* pUeGrpPrm; /*!< pointer to parent UE group */

    /** DMRS parameters: */
    uint16_t dmrsScrmId; /*!< DMRS scrambling Id. Value: 0-65535 */
    uint16_t dmrsPortBmsk; /*!< bitmask's set bits specify DMRS ports used. Least significant bit corresponds to port 0. */
    uint8_t  nUeLayers;  /*!< total number of user layers. Value: 1->8 */
    uint8_t  scid;       /*!< dmrs sequence initialization. Value: 0->1 */

    uint16_t BWPStart; /*!< Bandwidth part start (PRB number starting from 0). Used only if ref. point is 1. */
    float    beta_dmrs;   /*!< Fronthaul DMRS amplitude scaling */
    float    beta_qam;    /*!< Fronthaul QAM amplitude scaling */

    uint8_t  refPoint; /*!< DMRS reference point. Value 0->1. */

    /** ID parameters: */
    uint16_t rnti;        /*!< RNTI (Radio Network Temporary Identifier). Value: 1->65535. */
    uint16_t dataScramId; /*!< used to compute bit scrambling seed. Value: 0->65535. */

    /** codeword parameters: */
    uint8_t   nCw;     /*!< number of codewords. Value: 1->2. */
    uint16_t* pCwIdxs; /*!< nCw element wide array; it contains indices into the pCwPrms array of cuphyPdschCellGrpDynPrm_t */

    /** Pre-coding parameters: */
    uint8_t enablePrcdBf; /*!< Enable pre-coding for this UE*/

    uint16_t pmwPrmIdx; /*!< Index to pre-coding matrix array, i.e., to the pPmwPrms array of the
                                    cuphyPdschCellGrpDynPrm_t struct */
    uint8_t  nlAbove16{0};
} cuphyPdschUePrm_t;

/**
 * Per Codeword (CW) parameters
 */
typedef struct _cuphyPdschCwPrm
{
    cuphyPdschUePrm_t* pUePrm; /*!< pointer to parent UE */

    /** Coding parameters: */
    uint8_t mcsTableIndex; /*!< Solely used in optional TB size checking; use targetCodeRate and qamModOrder for everything else
                                MCS (Modulation and Coding Scheme) Table Id. Value: 0->2
                                0: Table 5.1.3.1-1
                                1: Table 5.1.3.1-2
                                2: Table 5.1.3.1-3 */
    uint8_t mcsIndex;      /*!< Solely used in optional TB size checking; use targetCodeRate and qamModOrer for everything else
                                MCS index within the mcsTableIndex table. Value: 0->31 */

    // Parameters to enable MCS greater than 28
    uint16_t targetCodeRate; /*!< Target code rate. Assuming code rate is codeRate = x/1024.0 where x contains a single digit after decimal point,
                                  then targetCodeRate = static_cast<uint16_t>(x * 10) = static_cast<uint16_t>(codeRate * 1024 * 10) */
    uint8_t  qamModOrder;    /*!< Modulation order. Values: 2, 4, 6 or 8. */

    uint8_t rv;            /*!< redundancy version. Value: 0->3 */

    /** TB (Transport Block) location: */
    uint32_t tbStartOffset; /*!< starting index (in bytes) of transport block within pTbInput array in cuphyPdschDataIn_t */
    uint32_t tbSize;        /*!< transport block size in bytes; if the TB is for a cell in testing mode this holds length of PN sequence */

    /** Parameters used for LBRM (Limited Buffer Rate-Matching) transport block (TB) size computation */
    uint16_t n_PRB_LBRM; /*!< number of PRBs used for LBRM TB size computation. Possible values: {32, 66, 107, 135, 162, 217, 273}. */
    uint8_t  maxLayers;  /*!< number of layers used for LBRM TB size computation (at most 4). */
    uint8_t  maxQm;      /*!< modulation order used for LBRM TB size computation. Value: 6 or 8. */

} cuphyPdschCwPrm_t;

/**
 * Structure to define pre-coding matrix
 */

typedef struct _cuphyPmW_t
{
    /*!< Pre-coding matrix used only if cuphyPdschUePrm_t.enablePrcdBf is true.
    Layout of the data is such that cuphyPdschUePrm_t.nUeLayers is slower dimension.
    The cuphyPmW_t.nPorts is the number of columns.
    Memory layout in expected to be in following manner with row-major layout.
    If a transport block has 2 layers and output tensor has 4 ports, the following layout should be used
    --------------------------------------------
    |       | Port 0 | Port 1 | Port 3 | Port 4|
    --------------------------------------------
    | TB L0 |        |        |        |       |
    |-------|-----------------------------------
    | TB L1 |        |        |        |       |
    --------------------------------------------
    */
    __half2 matrix[MAX_DL_LAYERS_PER_TB * MAX_DL_PORTS];
    uint8_t nPorts; /*!< number of ports for this UE. */
} cuphyPmW_t;

/**
 * DL Pre-coding matrices
 */
typedef struct _cuphyPmWOneLayer_t
{
     /*!< Pre-coding matrix used only if enablePrcdBf is true and one layer of input is considered. */
    __half2 matrix[MAX_DL_PORTS];
    uint8_t nPorts; /*!< number of ports for this UE. */
} cuphyPmWOneLayer_t;

typedef struct _cuphyPdschCellAerialMetrics
{
    uint32_t   nTBs; /*!< # of TBs to be batch processed for a given cell. */
    uint32_t   tbSize; /*!< # of TB bytes per cell */
} cuphyPdschCellAerialMetrics_t;

/**
 * Cell group dynamic parameters
 */
typedef struct _cuphyPdschCellGrpDynPrm
{
    /** Cell group parameters */
    uint16_t                nCells;    /*!< # of cells to be batch processed. Should be <= nMaxCellsPerSlot from static parameters. */
    cuphyPdschCellDynPrm_t* pCellPrms; /*!< array of per-cell dynamic parameters with nCells elements */

    cuphyPdschCellAerialMetrics_t* pCellMetrics; /*!< array of per-cell metrics with nCells elements */

    /** Co-scheduled UE group parameters */
    uint16_t              nUeGrps;    /*!< # of co-scheduled UE groups */
    cuphyPdschUeGrpPrm_t* pUeGrpPrms; /*!< array of per-UE-group parameters with nUeGrps elements */

    /** UE parameters */
    uint16_t           nUes;    /*!< number of UEs */
    cuphyPdschUePrm_t* pUePrms; /*!< array of per-UE parameters with nUes elements */

    /** CW parameters */
    uint16_t           nCws;    /*!< number of code-words */
    cuphyPdschCwPrm_t* pCwPrms; /*!< array of per-CW parameters with nCws elements */

    /** CSI-RS parameters */
    uint16_t              nCsiRsPrms; /*!< number of CSI-RS parameters for all cells */
    _cuphyCsirsRrcDynPrm* pCsiRsPrms; /*!< array of per-cell CSI-RS parameters with nCsiRsPrms elements
                                              NB: a few of the cuphyCsirsRrcDynPrm_t fields will not be needed
                                                  as no symbols will be written. We could use a different struct too.
                                          */

    /** Pre-coding parameters: */
    uint16_t nPrecodingMatrices; /*!< number of precoding matrices */

    cuphyPmW_t* pPmwPrms; /*!< array of pre-coding matrices */

} cuphyPdschCellGrpDynPrm_t;

/**
 * PDSCH Data Input
 */
typedef struct _cuphyPdschDataIn
{
    uint8_t** pTbInput; /*!< array of transport block input buffers, one buffer per cell, indexed by cellPrmDynIdx.
                             Each pTbInput[] element points to a flat array with all TBs for that cell.
                             Currently per-cell TB allocations are contiguous, zero-padded to byte boundary.
                             When a cell is in test mode, then the buffer contains the PN23 (pseudorandom sequence)
                             rather than a payload. */
    enum
    {
        CPU_BUFFER,
        GPU_BUFFER
    } pBufferType; /*!< pTbInput[] buffer type */
} cuphyPdschDataIn_t;


/**
 * PDSCH Data Output
 */
typedef struct _cuphyPdschDataOut
{
    cuphyTensorPrm_t* pTDataTx; /*!< array of tensors with each tensor (indexed by cellPrmDynIdx) representing the transmit slot buffer of a cell in the cell group.
                                     Each cell's tensor may have a different geometry */
} cuphyPdschDataOut_t;

/**
 * PDSCH Dynamic Parameters
 */
typedef struct _cuphyPdschDynPrms
{
    cudaStream_t cuStream; /*!< CUDA stream on which pipeline is launched.
                                 @todo: cuPHY internally uses a CUDA stream pool to launch multiple parallel CUDA kernels from the same
                                 component. So cuStream provided below is not the only stream where workload would be launched. To be
                                 closed after consensus with a wider group */

    /** Control parameters */
    uint64_t procModeBmsk; /*!< Processing modes (e.g., full-slot processing w/ profile 0 PDSCH_PROC_MODE_FULL_SLOT|PDSCH_PROC_MODE_PROFILE0)*/

    cuphyPdschCellGrpDynPrm_t const* pCellGrpDynPrm; /*!< Pointer to cell group configuration parameters. Each pipeline will process a single cell-group. */

    /** Data parameters */
    cuphyPdschDataIn_t const* pDataIn;      /*!< Pointer to PDSCH data input */
    cuphyPdschDataIn_t const* pTbCRCDataIn; /*!< Pointer to optional TB CRCs */
    cuphyPdschDataOut_t*      pDataOut;     /*!< Pointer to PDSCH data output that will contain pCellGrpDynPrm->nCells tensors */
    cuphyPdschStatusOut_t*    pStatusInfo;  /*!< pointer to struct holding status information; currently specifies if PDSCH setup ran into MAX_ER_CB issue . */
} cuphyPdschDynPrms_t;

/**
 * PDSCH batching configuration parameters
 */
struct cuphyPdschBatchPrm;

/**
 * PDSCH batch configuration handle
 */
typedef struct cuphyPdschBatchPrm* cuphyPdschBatchPrmHndl_t;

//-----------------------------------------------------------------------------------------------------------
// Functions

/******************************************************************/ /**
 * \brief Allocate and initialize a cuPHY PDSCH pipeline
 *
 * Allocate a cuPHY PDSCH transmitter pipeline and returns a handle in the
 * address provided by the caller.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pPdschTxHdnl and/or \p pStatPrms is NULL.
 *
 * Returns ::CUPHY_STATUS_ALLOC_FAILED if a PdschTx object cannot be allocated
 *
 * Returns ::CUPHY_STATUS_SUCCESS if allocation and initialization were
 * successful.
 *
 * \param pPdschTxHndl - Address to return the new PdschTx instance
 * \param pStatPrms  - Pointer to PDSCH static parameters to be used in pipeline creation
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_ALLOC_FAILED,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphySetupPdschTx,::cuphyRunPdschTx,::cuphyDestroyPdschTx
 */
cuphyStatus_t CUPHYWINAPI cuphyCreatePdschTx(cuphyPdschTxHndl_t* pPdschTxHndl, cuphyPdschStatPrms_t const* pStatPrms);

size_t cuphyGetGpuMemoryFootprintPdschTx(cuphyPdschTxHndl_t pdschTxHndl);

#if 0
/******************************************************************/ /**
 * \brief Return pointer to cuphyMemoryFootprint of cuPHY PDSCH
 *
 * Return pointer to cuphyMemoryFootprint of cuPHY PDSCH transmitter pipeline which tracks
 * the size of GPU memory allocations owned by that object.
 *
 * Returns nullptr if \p pdschTxHndl is NULL
 *
 * \param pdschTxHndl  - Handle of PdschTx instance
 *
 * \return
 * \p to cuphyMemoryFootprint
 *
 */
const void*  cuphyGetMemoryFootprintTrackerPdschTx(cuphyPdschTxHndl_t pdschTxHndl);
#endif

/******************************************************************/ /**
 * \brief Allocate a container for PDSCH batch parameters
 *
 * Allocate storage to hold PDSCH batch parameters and return a handle in the
 * address provided by the caller.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pBatchPrmHndl is NULL.
 *
 * Returns ::CUPHY_STATUS_ALLOC_FAILED if a batch parameter container object cannot be allocated
 *
 * Returns ::CUPHY_STATUS_SUCCESS if allocation were successful.
 *
 * \param pBatchPrmHndl - Address to return the container for PDSCH batch parameters
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_ALLOC_FAILED,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyDestroyPdschBatchPrm
 */
cuphyStatus_t CUPHYWINAPI cuphyCreatePdschBatchPrm(cuphyPdschBatchPrmHndl_t* pBatchPrmHndl);

/******************************************************************/ /**
 * \brief Destroy container for PDSCH batch parameters
 *
 * Destroy a cuPHY context object that was previously created by a call
 * to ::cuphyCreatePdschBatchPrm. The handle provided to this function should
 * not be used for any operations after this function returns.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p batchPrmHndl is NULL.
 *
 * Returns ::CUPHY_STATUS_SUCCESS if destruction was successful.
 *
 * \param batchPrmHndl - previously allocated PDSCH batch parameter instance
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyCreatePdschBatchPrm
 */
cuphyStatus_t CUPHYWINAPI cuphyDestroyPdschBatchPrm(cuphyPdschBatchPrmHndl_t batchPrmHndl);

#if 0
/******************************************************************/ /**
 * \brief Configure/Reconfigure cuPHY PDSCH pipeline
 *
 * Configure a cuPHY PDSCH transmitter pipeline and returns a handle in the
 * address provided by the caller.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pPdschTxHdnl and/or \p pStatPrms is NULL.
 *
 * Returns ::CUPHY_STATUS_ALLOC_FAILED if a PdschTx object cannot be allocated
 *
 * Returns ::CUPHY_STATUS_SUCCESS if allocation and initialization were
 * successful.
 *
 * \param pPdschTxHndl - Address to return the new PdschTx instance
 * \param pStatPrms  - Pointer to PDSCH static parameters to be used in pipeline configuration
 * \param pQuasiStatPrms - Pointer to PDSCH quasi-static parameters to be used in pipeline configuration
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_ALLOC_FAILED,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphySetupPdschTx,::cuphyRunPdschTx,::cuphyDestroyPdschTx
 */
cuphyStatus_t CUPHYWINAPI cuphyConfigPdschTx(cuphyPdschTxHndl_t* pPdschTxHndl, cuphyPdschStatPrms_t const* pStatPrms, cuphyPdschQuasiStatPrms_t const* pQuasiStatPrms);
#endif

/******************************************************************/ /**
 * \brief Batch PDSCH workoad
 *
 * Batch PDSCH workload across one or more cells, UE-groups and UEs.
 * The batched configuration is used during slot execution
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pdschTxHndl and/or \p pDynPrms (or its components)
 * and/or batchPrmHndl is NULL
 *
 * Returns ::CUPHY_STATUS_SUCCESS if setup is successful.
 *
 * \param pdschTxHndl  - Handle of PdschTx instance to be setup
 * \param pDynPrms     - Dynamic parameters carrying information needed for slot processing
 * \param batchPrmHndl - Workload batching information
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyCreatePdschTx,::cuphyRunPdschTx,::cuphyDestroyPdschTx
 */
cuphyStatus_t CUPHYWINAPI cuphyBatchPdschTx(cuphyPdschTxHndl_t pdschTxHndl, cuphyPdschDynPrms_t* pDynPrms, cuphyPdschBatchPrmHndl_t batchPrmHndl);

/******************************************************************/ /**
 * \brief Setup cuPHY PDSCH pipeline for slot processing
 *
 * Setup cuPHY PDSCH transmitter pipeline (and its components) state in
 * preparation towards slot execution
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pdschTxHndl and/or \p pDynPrms
 * and/or batchPrmHndl is NULL
 *
 * Returns ::CUPHY_STATUS_SUCCESS if allocation and initialization were
 * successful.
 *
 * \param pdschTxHndl  - Handle of PdschTx instance to be setup
 * \param pDynPrms     - Dynamic parameters carrying information needed for slot processing
 * \param batchPrmHndl - Workload batching information
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyCreatePdschTx,::cuphyRunPdschTx,::cuphyDestroyPdschTx
 */
cuphyStatus_t CUPHYWINAPI cuphySetupPdschTx(cuphyPdschTxHndl_t pdschTxHndl, cuphyPdschDynPrms_t* pDynPrms, cuphyPdschBatchPrmHndl_t const batchPrmHndl);

/******************************************************************/ /**
 * \brief Fallback single-cell output buffer setup for cuPHY PDSCH pipeline for slot processing
 *
 * Setup the output buffer address for cuPHY PDSCH transmitter pipeline in
 * preparation towards slot execution. This function should only be used in
 * PDSCH_PROC_MODE_SETUP_ONCE_FALLBACK mode and requires that
 * cuphySetupPdschTx() has been called once before it for each pipeline object.
 * It supports only one cell per pipeline.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pdschTxHndl and/or \p pAddr is NULL
 *
 * Returns ::CUPHY_STATUS_SUCCESS if buffer setup was successful.
 *
 * \param pdschTxHndl  - Handle of PdschTx instance to be setup
 * \param pAddr        - New address for pipeline's output tensor buffer (single cell only)
 * \param strm         - CUDA stream for async. memory copy
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 */
cuphyStatus_t CUPHYWINAPI cuphyFallbackBufferSetupPdschTx(cuphyPdschTxHndl_t pdschTxHndl, void* pAddr, cudaStream_t strm);

/******************************************************************/ /**
 * \brief Fallback output buffers setup for multiple cells in cell group for cuPHY PDSCH pipeline for slot processing
 *
 * Setup the output buffer addresses for cuPHY PDSCH transmitter pipeline in
 * preparation towards slot execution. This function should only be used in
 * PDSCH_PROC_MODE_SETUP_ONCE_FALLBACK mode and requires that
 * cuphySetupPdschTx() has been called once before it for each pipeline object.
 * It supports multiple cells per pipeline.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pdschTxHndl and/or \p pAddr or its elements are NULL
 *
 * Returns ::CUPHY_STATUS_SUCCESS if buffers setup was successful.
 *
 * \param pdschTxHndl     - Handle of PdschTx instance to be setup
 * \param pAddr           - Array of addresses for pipeline's output tensor buffers (one buffer per cell). Order of addresses should
 *                          match the order of dynamic cells in initial, non fallback, setup
 * \param fallback_cells  - Number of cells (same as valid elements in pAddr array). Should match number of cells during initial, non fallback, setup.
 * \param strm            - CUDA stream for async. memory copy
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 */
cuphyStatus_t CUPHYWINAPI cuphyFallbackBuffersSetupPdschTx(cuphyPdschTxHndl_t pdschTxHndl, void** pAddr, int fallback_cells, cudaStream_t strm);

/******************************************************************/ /**
 * \brief Run cuPHY PDSCH pipeline processing in specified mode
 *
 * Call triggers cuPHY PDSCH transmitter pipeline execution in mode specified
 * by procModeBmsk
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pdschTxHndl is NULL and/or
 * procModeBmsk is not supported.
 *
 * Returns ::CUPHY_STATUS_SUCCESS if PdschTx execution is successful.
 *
 * \param pdschTxHndl  - Handle of PdschTx instance which is to be triggered
 * \param procModeBmsk - Processing mode bitmask containing one or more processing modes applicable during this execution
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyCreatePdschTx,::cuphySetupPdschTx,::cuphyDestroyPdschTx
 */
cuphyStatus_t CUPHYWINAPI cuphyRunPdschTx(cuphyPdschTxHndl_t pdschTxHndl, uint64_t procModeBmsk);

/******************************************************************/ /**
 * \brief Destroy a cuPHY PDSCH transmit pipeline object
 *
 * Destroy a cuPHY PDSCH transmitter pipeline object that was previously
 * created by ::cuphyCreatePdschTx. The handle provided to this function
 * should not be used for any operations after this function returns.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pdschTxHndl is NULL.
 *
 * Returns ::CUPHY_STATUS_SUCCESS if destruction was successful.
 *
 * \param pdschTxHndl - handle to previously allocated PdschTx instance
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyCreatePdschTx
 */
cuphyStatus_t CUPHYWINAPI cuphyDestroyPdschTx(cuphyPdschTxHndl_t pdschTxHndl);

/** @} */ /* END CUPHY_PDSCH_TRANSMITTER */

/**
 * \defgroup CUPHY_PDCCH_TRANSMITTER PDCCH Transmitter
 *
 * This section describes the PDCCH transmit pipeline functions of the cuPHY
 * application programming interface.
 *
 * @{
 */

struct cuphyPdcchTx;
/**
 * cuPHY PDCCH transmitter handle
 */
typedef struct cuphyPdcchTx* cuphyPdcchTxHndl_t;

/**
 * PDCCH static parameters
 */
typedef struct _cuphyPdcchStatPrms
{
    cuphyTracker_t*     pOutInfo;      /*!< pointer to cuphyTracker_t. Its pMemoryFootprint pointer will be updated by cuPHY in channel objection creation. The caller will thus have access to the cuphyMemoryFootprint object of that cuPHY PDCCH channel object. The cuphyMemoryFootprint object tracks the size of the GPU memory allocations owned by that cuPHY PDCCH channel object. */
    uint16_t nMaxCellsPerSlot; /*!< Maximum number of supported cells (used to define upper limits on number of coresets, number of DCIs etc)
                                    nMaxCoresetsPerSlot = nMaxCellsPerSlot * CUPHY_PDCCH_N_MAX_CORESETS_PER_CELL
                                    nMaxDcisPerSlot     = nMaxCoresetsPerSlot * CUPHY_PDCCH_MAX_DCIS_PER_CORESET */
} cuphyPdcchStatPrms_t;

/**
 * PDCCH DCI parameters
 */
//typedef struct _cuphyPdcchDciDynPrm
struct _cuphyPdcchDciDynPrm
{
    uint32_t                   Npayload;    /*!< number of bits for PDCCH payload */
    uint32_t                   rntiCrc;     /*!< rnti number for CRC scrambling */
    uint32_t                   rntiBits;    /*!< rnti number for bit scrambling */
    uint32_t                   dmrs_id;     /*!< dmrs scrambling id */ //dmrsId
    uint32_t                   aggr_level;     /*!< aggregation level */ //aggrL
    uint32_t                   cce_index;     /*!<  */ //cce_index

    float                      beta_qam;    /*!< amplitude factor of qam signal */
    float                      beta_dmrs;   /*!< amplitude factor of dmrs signal */
    uint8_t                    enablePrcdBf;     /*!< Enable pre-coding for this PDCCH DCI*/
    uint16_t                   pmwPrmIdx;        /*!< Index to pre-coding matrix array, i.e., to the pPmwPrms array of the
                                                      cuphyPdcchDynPrms_t struct */
//} cuphyPdcchDciPrm_t;
};

/**
 * PDCCH Coreset dynamic parameters
 */
typedef struct _cuphyPdcchCoresetDynPrm
{
    uint32_t n_f;         /*!< number of subcarriers in full BW */
    uint32_t slot_number; /*!< slot number */
    uint32_t start_rb;    /*!< starting RB */
    uint32_t start_sym;   /*!< starting OFDM symbol number */
    uint32_t n_sym;       /*!< number of pdcch OFDM symbols (1-3) */
    uint32_t bundle_size; /*!< bundle size for PDCCH. It is in REGs. */
    uint32_t interleaver_size; /*!< Interleaving happens at the bundle granularity. Values: 2, 3 or 6 in interleaved mode. */
    uint32_t shift_index; /*!< shift index for interleaved mode */
    uint32_t interleaved; /*!< interleaved mode. Value: 0 or 1 */

    uint64_t freq_domain_resource; /*!< frequency domain resource */ //use to derive coreset-map, n_CCE and rb_coreset

    uint16_t  coreset_type;  /*!< Coreset Type. Values: 0, 1. */

    uint8_t testModel;      /*!< Specifies the cell this coreset belongs to is in testing mode if set to 1.  Value: 0-1
                                 For cells in testing mode, the DCI payload buffers hold bits from the PN23 (pseudorandom) sequence
                                 instead of a DCI payload. */

    uint8_t   nDci;          /*!< number of DCIs in this coreset. Value: 1->91. */
    uint32_t  dciStartIdx;   /*!< index of the first DCI (from this coreset), DCI indices of a given coreset are assumed to be
                                  continuous with indices: dciStartIdx, dciStartIdx+1, ... (dciStartIdx+nDcis-1)
                                  - Index into per DCI parameters (pDciPrms in cuphyPdcchDynPrms_t)
                                    E.g. Parameters in 2nd DCI of this coresets are accessed as pDciPrms[dciStartIdx+1]
                                  - Strided index into input DCI payload (pDciInput in cuphyPdcchDataIn_t) with stride = CUPHY_PDCCH_MAX_DCI_PAYLOAD_BYTES
                                    E.g. The first payload byte of the 2nd DCI in this coreset is accessed as pDciInput[(dciStartIdx+1)*CUPHY_PDCCH_MAX_DCI_PAYLOAD_BYTES] */
    uint32_t  slotBufferIdx; /*<! Index into output slot buffer tensor (pTDataTx in cuphyPdcchDataOut_t) where the prepared DCI payload needs to be written
                                  slotBufferIdx < nCells */
} cuphyPdcchCoresetDynPrm_t;

/**
 * PDCCH Data Input
 */
typedef struct _cuphyPdcchDataIn
{
    uint8_t* pDciInput; /*!< Pointer to DCI payloads, payload of each DCI is at stride of CUPHY_PDCCH_MAX_DCI_PAYLOAD_BYTES bytes from previous.
                             When a cell is in test mode, then the buffer contains bits from the PN23 (pseudorandom sequence).*/
    enum
    {
        CPU_BUFFER,
        GPU_BUFFER
    } pBufferType; /*!< pDciInput buffer type; currently only CPU_BUFFER is supported */
} cuphyPdcchDataIn_t;

/**
 * PDCCH Data Output
 */
typedef struct _cuphyPdcchDataOut
{
    cuphyTensorPrm_t* pTDataTx; /*!< Array of tensors with each tensor (indexed by slotBufferIdx) representing the slot buffer to be transmitted */
                                /*!< Note: Each tensor may have a different geometry */
} cuphyPdcchDataOut_t;

/**
 * PDCCH Dynamic Parameters
 */
typedef struct _cuphyPdcchDynPrms
{
    cudaStream_t cuStream; /*!< CUDA stream on which pipeline is launched. */

    /** Control parameters */
    uint64_t procModeBmsk;

    uint32_t                         nDci;      /*!< total number of DCIs to be processed and transmitted */
    cuphyPdcchDciPrm_t const*        pDciPrms; /*!< array of per-DCI parameters with nDCIs elements */

    uint16_t                         nCoresets;      /*!< total number of PDCCH coresets to be processed and transmitted */
    cuphyPdcchCoresetDynPrm_t const* pCoresetDynPrm; /*!< Pointer to array of Coreset configuration parameters */

    uint16_t                         nCells;  /*!< Number of cells for which PDCCH needs to be processed in this slot */

    uint16_t                      nPrecodingMatrices; /*!< number of precoding matrices */
    cuphyPmWOneLayer_t*           pPmwParams;         /*!< array of pre-coding matrices */

    CUgraph                         *chan_graph;   /* Pointer to root node of graph */

    /** Data parameters */
    cuphyPdcchDataIn_t const*        pDataIn;  /*!< Pointer to PDCCH data input */
    cuphyPdcchDataOut_t*             pDataOut; /*!< Pointer to PDCCH data output */
} cuphyPdcchDynPrms_t;

// PDCCH processing modes
typedef enum _cuphyPdcchProcMode
{
    PDCCH_PROC_MODE_STREAMS = 0x0, // stream processing
    PDCCH_PROC_MODE_GRAPHS  = 0x1, // graph processing
} cuphyPdcchProcMode_t;

/******************************************************************/ /**
 * \brief Allocate and initialize a cuPHY PDCCH pipeline
 *
 * Allocate a cuPHY PDCCH transmitter pipeline and return a handle in the
 * address provided by the caller.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pPdcchTxHdnl or \p pStatPrms is NULL.
 *
 * Returns ::CUPHY_STATUS_ALLOC_FAILED if a PdcchTx object cannot be allocated
 *
 * Returns ::CUPHY_STATUS_SUCCESS if allocation and initialization were
 * successful.
 *
 * \param pPdcchTxHndl - Address to return the new PdcchTx instance
 * \param pStatPrms    - Pointer to PDCCH static parameters to be used in pipeline creation
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_ALLOC_FAILED,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphySetupPdcchTx,::cuphyRunPdcchTx,::cuphyDestroyPdcchTx
 */
cuphyStatus_t CUPHYWINAPI cuphyCreatePdcchTx(cuphyPdcchTxHndl_t* pPdcchTxHndl, cuphyPdcchStatPrms_t const* pStatPrms);

#if 0
/******************************************************************/ /**
 * \brief Return pointer to cuphyMemoryFootprint of cuPHY PDCCH
 *
 * Return pointer to cuphyMemoryFootprint of cuPHY PDCCH transmitter pipeline which tracks
 * the size of GPU memory allocations owned by that object.
 *
 * Returns nullptr if \p pdcchTxHndl is NULL
 *
 * \param pdcchTxHndl  - Handle of PdcchTx instance
 *
 * \return
 * \p to cuphyMemoryFootprint
 *
 */
const void*  cuphyGetMemoryFootprintTrackerPdcchTx(cuphyPdcchTxHndl_t pdcchTxHndl);
#endif

/******************************************************************/ /**
 * \brief Setup cuPHY PDCCH pipeline for slot processing
 *
 * Setup cuPHY PDCCH transmitter pipeline (and its components) state in
 * preparation towards slot execution
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pdcchTxHndl and/or \p pDynPrms
 * is NULL
 *
 * Returns ::CUPHY_STATUS_SUCCESS if allocation and initialization were
 * successful.
 *
 * \param pdcchTxHndl  - Handle of PdcchTx instance to be setup
 * \param pDynPrms     - Dynamic parameters carrying information needed for slot processing
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyCreatePdcchTx,::cuphyRunPdcchTx,::cuphyDestroyPdcchTx
 */
cuphyStatus_t CUPHYWINAPI cuphySetupPdcchTx(cuphyPdcchTxHndl_t pdcchTxHndl, cuphyPdcchDynPrms_t* pDynPrms);

/******************************************************************/ /**
 * \brief Run cuPHY PDCCH pipeline processing in specified mode
 *
 * Call triggers cuPHY PDCCH transmitter pipeline execution in mode specified
 * by procModeBmsk
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pdcchTxHndl is NULL and/or
 * procModeBmsk is not supported.
 *
 * Returns ::CUPHY_STATUS_SUCCESS if PdcchTx execution is successful.
 *
 * \param pdcchTxHndl  - Handle of PdcchTx instance which is to be triggered
 * \param procModeBmsk - Processing mode bitmask containing one or more processing modes applicable during this execution. Currently unused.
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyCreatePdcchTx,::cuphySetupPdcchTx,::cuphyDestroyPdcchTx
 */
cuphyStatus_t CUPHYWINAPI cuphyRunPdcchTx(cuphyPdcchTxHndl_t pdcchTxHndl, uint64_t procModeBmsk);

/******************************************************************/ /**
 * \brief Destroy a cuPHY PDCCH transmit pipeline object
 *
 * Destroy a cuPHY PDCCH transmitter pipeline object that was previously
 * created by ::cuphyCreatePdcchTx. The handle provided to this function
 * should not be used for any operations after this function returns.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pdcchTxHndl is NULL.
 *
 * Returns ::CUPHY_STATUS_SUCCESS if destruction was successful.
 *
 * \param pdcchTxHndl - handle to previously allocated PdcchTx instance
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyCreatePdcchTx
 */
cuphyStatus_t CUPHYWINAPI cuphyDestroyPdcchTx(cuphyPdcchTxHndl_t pdcchTxHndl);

/** @} */ /* END CUPHY_PDCCH_TRANSMITTER */

/**
 * \defgroup CUPHY_CSIRS CSI-RS Transmitter/Receiver
 *
 * This section describes the CSI-RS TX/RX pipeline functions of the cuPHY
 * application programming interface.
 *
 * @{
 */

struct cuphyCsirsTx;
/**
 * cuPHY CSIRS transmitter handle
 */
typedef struct cuphyCsirsTx* cuphyCsirsTxHndl_t;

struct cuphyCsirsRx;
/**
 * cuPHY CSIRS receiver handle
 */
typedef struct cuphyCsirsRx* cuphyCsirsRxHndl_t;

/**
 * CSI-RS RRC dynamic parameters
 */
typedef struct _cuphyCsirsRrcDynPrm
{
    uint16_t       startRb;        /*!< RB where this CSI resource starts. Expected value < 273 */
    uint16_t       nRb;            /*!< Number of RBs across which this CSI resource spans. Expected value <= 273-startRb */
    uint16_t       freqDomain;     /*!< Bitmap defining the freqDomainAllocation. Counting is started from least significant bit */
    uint8_t        row;            /*!< Row entry into the CSI resource location table. Valid values 1-18 */
    uint8_t        symbL0;         /*!< Time domain location L0 and firstOFDMSymbolInTimeDomain. 0 <= Valid value < OFDM_SYMBOLS_PER_SLOT */
    uint8_t        symbL1;         /*!< Time domain location L1 and firstOFDMSymbolInTimeDomain2. 0 <= Valid value < OFDM_SYMBOLS_PER_SLOT */
    uint8_t        freqDensity;    /*!< The density field, p and comb offset (for dot5), 0: dot5(even RB), 1: dot5 (odd RB), 2: One, 3: three */
    uint16_t       scrambId;       /*!< ScramblingId of CSI-RS */
    uint8_t        idxSlotInFrame; /*!< slot index in frame */
    cuphyCsiType_t csiType;        /*!< CSI Type, 0: TRS, 1: CSI-RS NZP, 2: CSI-RS ZP. Only  CSI-RS NZP supported currently */
    cuphyCdmType_t cdmType;        /*!< CDM Type, 0: noCDM, 1: fd-CDM2, 2: cdm4-FD2-TD2, 3: cdm8-FD2-TD4 */
    float          beta;           /*!< Power scaling factor */
    /** Pre-coding parameters for CSI-RS TX: */
    uint8_t enablePrcdBf;         /*!< Enable pre-coding for this CSI-RS*/
    uint16_t pmwPrmIdx;           /*!< Index to pre-coding matrix array, i.e., to the pPmwParams array of the
                                    cuphyCsirsDynPrms_t struct */
} cuphyCsirsRrcDynPrm_t;

/**
 * CSI-RS TX Data Output
 */
typedef struct _cuphyCsirsDataOut
{
    cuphyTensorPrm_t* pTDataTx; /*!< Array of nCells tensors with each tensor (indexed by slotBufferIdx) representing the slot buffer to be transmitted */
                                /*!< Note: Each tensor may have a different geometry */
} cuphyCsirsDataOut_t;

/**
 * CSI-RS RX Data Input
 */
typedef struct _cuphyCsirsRxDataIn
{
    cuphyTensorPrm_t* pTDataRx; /*!< Array of nUes tensors with each tensor representing the slot buffer to be received */
                                /*!< Note: Each tensor may have a different geometry */
} cuphyCsirsRxDataIn_t;

/**
 * CSI-RS RX Data Output
 */
typedef struct _cuphyCsirsRxDataOut
{
    cuphyCsirsRxChEstBuffInfo_t* pChEstBuffInfo;  // array of ChEst buffers of all users
} cuphyCsirsRxDataOut_t;

/**
 * Per-cell CSI-RS Dynamic Parameters
 */
typedef struct _cuphyCsirsCellDynPrm
{
    uint16_t rrcParamsOffset; /*!< start index for this cell's nRrcParams in the pRrcDynPrm array of cuphyCsirsDynPrms_t; all elements are allocated continuously. */
    uint8_t  nRrcParams;      /*!< number of RRC parameters co-scheduled for this cell. Maximum allowed: CUPHY_CSIRS_MAX_NUM_PARAMS. */
    uint16_t slotBufferIdx;   /*!< index into output slot buffer tensor array pDataOut->pTDataTx in cuphyCsirsCellDynPrms_t for this cell. Values: [0, nCells). */
    uint16_t cellPrmStatIdx;  /*!< Index to cell-static parameter information, i.e., to the pCellStatPrms array of the cuphyCsirsStatPrms_t struct. */
} cuphyCsirsCellDynPrm_t;

typedef struct _cuphyCsirsUeDynPrm
{
    uint16_t cellPrmStatIdx; 
    uint8_t  nRxAnt; 
} cuphyCsirsUeDynPrm_t;

typedef struct _cuphyCsirsDynPrms
{
    cudaStream_t cuStream; /*!< CUDA stream on which pipeline is launched. */

    // cuphyCsirsRrcDynPrm_t is also used in PDSCH
    cuphyCsirsRrcDynPrm_t *pRrcDynPrm; /*!< Pointer to RRC parameters across all nCells cells. Note: the length of this array is the sum of the field nRrcParams in cuphyCsirsCellDynPrm_t across nCells */
    // FIXME we could also explicitly specify the length of the pRrcDynPrm array; see below
    //uint16_t                   nRrcDynPrm; /*!< Number of elements in pRrcDynPrm array. Maximum allowed: CUPHY_CSIRS_MAX_NUM_PARAMS * max cells per slot specified during CSI-RS channel creation */

    uint16_t nCells; /*!< Number of cells for which CSI-RS will be computed in this slot */

    cuphyCsirsCellDynPrm_t* pCellParam; /*!< Array with nCells elements */

    uint64_t procModeBmsk; /*!< Processing modes */

    CUgraph *chan_graph;   /* Pointer to root node of graph */

    /** Data parameters */
    cuphyCsirsDataOut_t* pDataOut; /*!< Pointer to CSI-RS data output. */

    /** Pre-coding parameters: */
    uint16_t nPrecodingMatrices;    /*!< number of precoding matrices */
    cuphyPmWOneLayer_t* pPmwParams; /*!< array of pre-coding matrices */
} cuphyCsirsDynPrms_t;


typedef struct _cuphyCsirsRxDynPrms
{
    cudaStream_t cuStream; /*!< CUDA stream on which pipeline is launched. */

    // cuphyCsirsRrcDynPrm_t is also used in PDSCH
    cuphyCsirsRrcDynPrm_t *pRrcDynPrm; /*!< Pointer to RRC parameters across all nCells cells. Note: the length of this array is the sum of the field nRrcParams in cuphyCsirsCellDynPrm_t across nCells */

    uint16_t nCells; /*!< Number of cells for which CSI-RS will be computed in this slot */

    cuphyCsirsCellDynPrm_t* pCellParam; /*!< Array with nCells elements */
    
    uint16_t nUes;  /*!< Number of UEs for which CSI-RS will be processed in this slot */
    
    cuphyCsirsUeDynPrm_t* pUeParam; /*!< Array with nUes elements */

    uint64_t procModeBmsk; /*!< Processing modes */

    CUgraph *chan_graph;   /* Pointer to root node of graph */

    /** Data parameters */
    cuphyCsirsRxDataIn_t* pDataIn; /*!< Pointer to CSI-RS Rx data input. */
    cuphyCsirsRxDataOut_t* pDataOut; /*!< Pointer to CSI-RS Rx data output. */
} cuphyCsirsRxDynPrms_t;

// PDCCH processing modes
typedef enum _cuphyCsirsProcMode
{
    CSIRS_PROC_MODE_STREAMS = 0x0, // stream processing
    CSIRS_PROC_MODE_GRAPHS  = 0x1, // graph processing
} cuphyCsirsProcMode_t;

//-----------------------------------------------------------------------------------------------------------
// Functions

/**
 * CSI-RS static parameters
 */
typedef struct _cuphyCsirsStatPrms
{
    cuphyTracker_t*     pOutInfo;      /*!< pointer to cuphyTracker_t. Its pMemoryFootprint pointer will be updated by cuPHY in channel objection creation. The caller will thus have access to the cuphyMemoryFootprint object of that cuPHY CSIRS channel object. The cuphyMemoryFootprint object tracks the size of the GPU memory allocations owned by that cuPHY CSIRS channel object. */
    cuphyCellStatPrm_t* pCellStatPrms; /*!< array of cell-specific static parameters with nCells elements. Currently only nPrbDlBwp field is used */
    uint16_t nCells;                   /*!< number of supported cells. TODO May rename to nMaxCells to be consistent with PUSCH. */
    uint16_t nMaxCellsPerSlot; /*!< Maximum number of supported cells (used to define upper limits on number of CSIRS parameters etc. */
} cuphyCsirsStatPrms_t;

/******************************************************************/ /**
 * \brief Allocate and initialize a cuPHY CSI-RS TX pipeline
 *
 * Allocate a cuPHY CSI-RS transmitter pipeline and return a handle in the
 * address provided by the caller.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pCsirsTxHndl or \p pStatPrms is NULL.
 *
 * Returns ::CUPHY_STATUS_ALLOC_FAILED if a CsirsTx object cannot be allocated
 *
 * Returns ::CUPHY_STATUS_SUCCESS if allocation and initialization were
 * successful.
 *
 * \param pCsirsTxHndl - Address to return the new CsirsTx instance
 * \param pStatPrms    - Pointer to CSI-RS static parameters to be used in pipeline creation
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_ALLOC_FAILED,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphySetupCsirsTx,::cuphyRunCsirsTx,::cuphyDestroyCsirsTx
 */
cuphyStatus_t CUPHYWINAPI cuphyCreateCsirsTx(cuphyCsirsTxHndl_t* pCsirsTxHndl, cuphyCsirsStatPrms_t const* pStatPrms);

#if 0
/******************************************************************/ /**
 * \brief Return pointer to cuphyMemoryFootprint of cuPHY CSI-RS TX
 *
 * Return pointer to cuphyMemoryFootprint of cuPHY CSI-RS transmitter pipeline which tracks
 * the size of GPU memory allocations owned by that object.
 *
 * Returns nullptr if \p csirsTxHndl is NULL
 *
 * \param csirsTxHndl  - Handle of CsirsTx instance
 *
 * \return
 * \p to cuphyMemoryFootprint
 *
 */
const void*  cuphyGetMemoryFootprintTrackerCsirsTx(cuphyCsirsTxHndl_t csirsTxHndl);
#endif

/******************************************************************/ /**
 * \brief Setup cuPHY CSI-RS TX pipeline for slot processing
 *
 * Setup cuPHY CSI-RS transmitter pipeline (and its components) state in
 * preparation towards slot execution
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p csirsTxHndl and/or \p pDynPrms
 * is NULL
 *
 * Returns ::CUPHY_STATUS_SUCCESS if allocation and initialization were
 * successful.
 *
 * \param csirsTxHndl  - Handle of csirsTx instance to be setup
 * \param pDynPrms     - Dynamic parameters carrying information needed for slot processing
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyCreateCsirsTx,::cuphyRunCsirsTx,::cuphyDestroyCsirsTx
 */
cuphyStatus_t CUPHYWINAPI cuphySetupCsirsTx(cuphyCsirsTxHndl_t csirsTxHndl, cuphyCsirsDynPrms_t* pDynPrms);

/******************************************************************/ /**
 * \brief Run cuPHY CSI-RS TX pipeline processing in specified mode
 *
 * Call triggers cuPHY CSI-RS transmitter pipeline execution
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p csirsTxHndl
 *
 * Returns ::CUPHY_STATUS_SUCCESS if CsirsTx execution is successful.
 *
 * \param csirsTxHndl  - Handle of CsirsTx instance which is to be triggered
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyCreateCsirsTx,::cuphySetupCsirsTx,::cuphyDestroyCsirsTx
 */
cuphyStatus_t CUPHYWINAPI cuphyRunCsirsTx(cuphyCsirsTxHndl_t csirsTxHndl);

/******************************************************************/ /**
 * \brief Destroy a cuPHY CSI-RS TX pipeline object
 *
 * Destroy a cuPHY CSI-RS transmitter pipeline object that was previously
 * created by ::cuphyCreateCsirsTx. The handle provided to this function
 * should not be used for any operations after this function returns.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p csirsTxHndl is NULL.
 *
 * Returns ::CUPHY_STATUS_SUCCESS if destruction was successful.
 *
 * \param csirsTxHndl - handle to previously allocated csirsTx instance
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyCreateCsirsTx
 */
cuphyStatus_t CUPHYWINAPI cuphyDestroyCsirsTx(cuphyCsirsTxHndl_t csirsTxHndl);

/******************************************************************/ /**
 * \brief Allocate and initialize a cuPHY CSI-RS RX pipeline
 *
 * Allocate a cuPHY CSI-RS RX pipeline and return a handle in the
 * address provided by the caller.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pCsirsRxHndl or \p pStatPrms is NULL.
 *
 * Returns ::CUPHY_STATUS_ALLOC_FAILED if a CsirsRx object cannot be allocated
 *
 * Returns ::CUPHY_STATUS_SUCCESS if allocation and initialization were
 * successful.
 *
 * \param pCsirsRxHndl - Address to return the new CsirsRx instance
 * \param pStatPrms    - Pointer to CSI-RS static parameters to be used in pipeline creation
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_ALLOC_FAILED,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphySetupCsirsRx,::cuphyRunCsirsRx,::cuphyDestroyCsirsRx
 */
cuphyStatus_t CUPHYWINAPI cuphyCreateCsirsRx(cuphyCsirsRxHndl_t* pCsirsRxHndl, cuphyCsirsStatPrms_t const* pStatPrms);

/******************************************************************/ /**
 * \brief Setup cuPHY CSI-RS RX pipeline for slot processing
 *
 * Setup cuPHY CSI-RS receiver pipeline (and its components) state in
 * preparation towards slot execution
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p csirsRxHndl and/or \p pDynPrms
 * is NULL
 *
 * Returns ::CUPHY_STATUS_SUCCESS if allocation and initialization were
 * successful.
 *
 * \param csirsRxHndl  - Handle of csirsRx instance to be setup
 * \param pDynPrms     - Dynamic parameters carrying information needed for slot processing
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyCreateCsirsRx,::cuphyRunCsirsRx,::cuphyDestroyCsirsRx
 */
cuphyStatus_t CUPHYWINAPI cuphySetupCsirsRx(cuphyCsirsRxHndl_t csirsRxHndl, cuphyCsirsRxDynPrms_t* pDynPrms);

/******************************************************************/ /**
 * \brief Run cuPHY CSI-RS RX pipeline processing in specified mode
 *
 * Call triggers cuPHY CSI-RS receiver pipeline execution
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p csirsRxHndl
 *
 * Returns ::CUPHY_STATUS_SUCCESS if CsirsRx execution is successful.
 *
 * \param csirsRxHndl  - Handle of CsirsRx instance which is to be triggered
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyCreateCsirsRx,::cuphySetupCsirsRx,::cuphyDestroyCsirsRx
 */
cuphyStatus_t CUPHYWINAPI cuphyRunCsirsRx(cuphyCsirsRxHndl_t csirsRxHndl);

/******************************************************************/ /**
 * \brief Destroy a cuPHY CSI-RS RX pipeline object
 *
 * Destroy a cuPHY CSI-RS receiver pipeline object that was previously
 * created by ::cuphyCreateCsirsRx. The handle provided to this function
 * should not be used for any operations after this function returns.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p csirsRxHndl is NULL.
 *
 * Returns ::CUPHY_STATUS_SUCCESS if destruction was successful.
 *
 * \param csirsRxHndl - handle to previously allocated csirsTx instance
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyCreateCsirsRx
 */
cuphyStatus_t CUPHYWINAPI cuphyDestroyCsirsRx(cuphyCsirsRxHndl_t csirsRxHndl);

/** @} */ /* END CUPHY_CSIRS */

/**
 * \defgroup CUPHY_SRS_TRANSMITTER SRS Transmitter
 *
 * This section describes the SRS transmit pipeline functions of the cuPHY
 * application programming interface.
 *
 * @{
 */

struct cuphySrsTx;
/**
 * cuPHY SRS transmitter handle
 */
typedef struct cuphySrsTx* cuphySrsTxHndl_t;

typedef struct _cuphySrsTxStatPrms
{
    uint16_t            nMaxSrsUes;
    uint16_t            nSlotsPerFrame;
    uint16_t            nSymbsPerSlot;
    cuphyTracker_t*     pOutInfo;   
} cuphySrsTxStatPrms_t;

typedef struct _cuphySrsTxDynPrms
{
    // CUDA stream on which pipeline is launched
    cudaStream_t       cuStream;
    
    CUgraph *chan_graph;   /* Pointer to root node of graph */

    uint16_t           nSrsUes;
    // Control parameters
    cuphyUeSrsTxPrm_t* pUeSrsTxPrms;

    // Data parameters
    cuphyTensorPrm_t*  pTDataSrsTx; 
} cuphySrsTxDynPrms_t;

//-----------------------------------------------------------------------------------------------------------
// Functions
/******************************************************************/ /**
 * \brief Allocate and initialize a cuPHY SRS TX pipeline
 *
 * Allocate a cuPHY SRS transmitter pipeline and return a handle in the
 * address provided by the caller.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pSrsTxHndl or \p pStatPrms is NULL.
 *
 * Returns ::CUPHY_STATUS_ALLOC_FAILED if a SrsTx object cannot be allocated
 *
 * Returns ::CUPHY_STATUS_SUCCESS if allocation and initialization were
 * successful.
 *
 * \param pSrsTxHndl   - Address to return the new SrsTx instance
 * \param pStatPrms    - Pointer to SrsTx static parameters to be used in pipeline creation
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_ALLOC_FAILED,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphySetupSrsTx,::cuphyRunSrsTx,::cuphyDestroySrsTx
 */
cuphyStatus_t CUPHYWINAPI cuphyCreateSrsTx(cuphySrsTxHndl_t* pSrsTxHndl, cuphySrsTxStatPrms_t const* pStatPrms);

/******************************************************************/ /**
 * \brief Setup cuPHY SRS TX pipeline for slot processing
 *
 * Setup cuPHY SRS transmitter pipeline (and its components) state in
 * preparation towards slot execution
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p srsTxHndl and/or \p pDynPrms
 * is NULL
 *
 * Returns ::CUPHY_STATUS_SUCCESS if allocation and initialization were
 * successful.
 *
 * \param srsTxHndl    - Handle of SrsTx instance to be setup
 * \param pDynPrms     - Dynamic parameters carrying information needed for slot processing
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyCreateSrsTx,::cuphyRunSrsTx,::cuphyDestroySrsTx
 */
cuphyStatus_t CUPHYWINAPI cuphySetupSrsTx(cuphySrsTxHndl_t srsTxHndl, cuphySrsTxDynPrms_t* pDynPrms);

/******************************************************************/ /**
 * \brief Run cuPHY SRS TX pipeline processing in specified mode
 *
 * Call triggers cuPHY SRS transmitter pipeline execution
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p srsTxHndl
 *
 * Returns ::CUPHY_STATUS_SUCCESS if SrsTx execution is successful.
 *
 * \param srsTxHndl  - Handle of SrsTx instance which is to be triggered
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyCreateSrsTx,::cuphySetupSrsTx,::cuphyDestroySrsTx
 */
cuphyStatus_t CUPHYWINAPI cuphyRunSrsTx(cuphySrsTxHndl_t srsTxHndl);

/******************************************************************/ /**
 * \brief Destroy a cuPHY SRS transmit pipeline object
 *
 * Destroy a cuPHY SRS transmitter pipeline object that was previously
 * created by ::cuphyCreateSrsTx. The handle provided to this function
 * should not be used for any operations after this function returns.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p srsTxHndl is NULL.
 *
 * Returns ::CUPHY_STATUS_SUCCESS if destruction was successful.
 *
 * \param srsTxHndl - handle to previously allocated srsTx instance
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyCreateSrsTx
 */
cuphyStatus_t CUPHYWINAPI cuphyDestroySrsTx(cuphySrsTxHndl_t srsTxHndl);

/** @} */ /* END CUPHY_SRS_TRANSMITTER */

/**
 * \defgroup UL_CUPHY_PRACH_RECEIVER PRACH Receiver
 *
 * This section describes the structs and functions of the uplink
 * cuPHY PRACH channel receiver.
 *
 * @{
 */

struct cuphyPrachRx;
/**
 * cuPHY PRACH receiver handle
 */
typedef struct cuphyPrachRx* cuphyPrachRxHndl_t;

#define CUPHY_PRACH_RX_NUM_PREAMBLE (64) /*!<  Maximum number of PRACH preambles */

/**
 * @brief PRACH processing modes
 *
 */
typedef enum _cuphyPrachRxProcMode
{
    PRACH_PROC_MODE_NO_GRAPH   = 0x0, // stream based workload submission
    PRACH_PROC_MODE_WITH_GRAPH = 0x1, // CUDA graph based workload submission
    PRACH_MAX_PROC_MODES
} cuphyPrachRxProcMode_t;

/**
 * @brief PRACH Static Debug Parameters
 *
 */
typedef struct _cuphyPrachStatDbgPrms
{
    const char* pOutFileName;       // output file capturing pipeline intermediate states. No capture if null.
    uint8_t     enableApiLogging;   // control the API logging of PRACH static parameters
} cuphyPrachStatDbgPrms_t;

// #define CUPHY_PRACH_RECONFIG_API
#ifndef CUPHY_PRACH_RECONFIG_API

typedef struct _cuphyPrachOccaStatPrms
{
    uint16_t cellPrmStatIdx;         /*!< Index to cell-static parameter information (index into pCellPrms in cuphyPrachStatPrms_t) */
    uint16_t prachRootSequenceIndex; /*!< 0-137 for short preamble, 0-837 for long preamble */
    uint8_t  prachZeroCorrConf;      /*!< valid values 0-15 */
} cuphyPrachOccaStatPrms_t;

/**
 * @brief Cell specific static parameters for PRACH receiver processing
 */
typedef struct _cuphyPrachCellStatPrms
{
    uint8_t  occaStartIdx;           /*!< Start index of the occasion in cuphyPrachStatPrms.pOccaPrms */
    uint8_t  nFdmOccasions;          /*!< Number of FDM occasions for this cell (upto 8 per cell) */
    uint32_t N_ant;                  /*!< number of antennas */
    uint8_t  FR;                     /*!< FR1: sub-6G, FR2: mm-wave. valid values: 1, 2 */
    uint8_t  duplex;                 /*!< 0: FDD, 1: TDD */
    uint8_t  mu;                     /*!< numerology. Only supported value 0 and 1 */
    uint8_t  configurationIndex;     /*!< valid values 0-255 */
    uint8_t  restrictedSet;          /*!< Only supported value 0 */
} cuphyPrachCellStatPrms_t;

/**
 * @brief Static parameters to process all cell-group PRACH receiver
 */
typedef struct _cuphyPrachStatPrms
{
    cuphyTracker_t*     pOutInfo;      /*!< pointer to cuphyTracker_t. Its pMemoryFootprint pointer will be updated by cuPHY in channel objection creation. The caller will thus have access to the cuphyMemoryFootprint object of that cuPHY PRACH channel object. The cuphyMemoryFootprint object tracks the size of the GPU memory allocations owned by that cuPHY PRACH channel object. */

    // Cell specific
    uint16_t                        nMaxCells; /*!< Number of cells for which the cell specific static parameter is provided, length of array pointed to by pCellPrms */
    cuphyPrachCellStatPrms_t const* pCellPrms; /*!< Pointer to array of cell specific static parameters whose dimension is nMaxCells */

    // Occasion specific
    cuphyPrachOccaStatPrms_t const* pOccaPrms;     /*!< Pointer to array of occasion specific static parameters. Note: the length of this array is the sum of the field nFdmOccasions in cuphyPrachCellStatPrms_t across nMaxCells */

    uint16_t                        nMaxOccaProc; /*!< Maximum number of occasions to be processed in a single pipeline invocation, nMaxOccaProc most resource hungry occasions
                                                       out of total number of occasions are used for resource provisioning purposes */
    cuphyPrachStatDbgPrms_t*        pDbg;         /*!< Debug Interface */
    
    uint8_t                         enableUlRxBf;
} cuphyPrachStatPrms_t;

#else // CUPHY_PRACH_RECONFIG_API

/**
 * @brief PrachRx Static state
 */
struct cuphyPrachRxStatState;
/**
 * @brief PrachRx Static state handle
 */
typedef struct cuphyPrachRxStatState* cuphyPrachRxStatStateHndl_t;

/**
 * @brief PrachRx Quasi-static state
 */
struct cuphyPrachRxQuasiStatState;
/**
 * @brief PrachRx Quasi-static state handle
 */
typedef struct cuphyPrachRxQuasiStatState* cuphyPrachRxQuasiStatStateHndl_t;

/**
 * @brief Cell specific static parameters for PRACH receiver processing
 */
typedef struct _cuphyPrachCellStatPrms
{
    uint16_t N_ant;                  /*!< number of antennas */
    uint8_t  FR;                     /*!< FR1: sub-6G, FR2: mm-wave. valid values: 1, 2 */
    uint8_t  duplex;                 /*!< 0: FDD, 1: TDD */
    uint8_t  mu;                     /*!< numerology. Only supported value 0 and 1 */
    uint8_t  nMaxFdmOccasions;       /*!< Maximum number of FDM occasions for this cell (upto 8 per cell) */
} cuphyPrachCellStatPrms_t;

/**
 * @brief Static parameters to process all cell-group PRACH receiver
 */
typedef struct _cuphyPrachStatPrms
{
    // Cell specific
    uint16_t                        nMaxCells;    /*!< Number of cells for which the cell specific static parameter is provided, length of array pointed to by pCellPrms
                                                       Note: For a given cell, cuPHY will reserve space for upto cuphyPrachCellStatPrms.nMaxFdmOccasions occassions which
                                                       shall later be configured by the cuphyConfigPrachRx API. Each occassion may be referred by occaPrmStatIdx specified in the dynamic API
                                                       Suppose cuphyPrachCellStatPrms.nMaxFdmOccasions = 8 for all cells then occaPrmStatIdx for the first occasion of cell[i] = i*8 */
    cuphyPrachCellStatPrms_t const* pCellPrms;    /*!< Pointer to array of cell specific static parameters whose dimension is nMaxCells */
    cuphyPrachStatDbgPrms_t*        pDbg;         /*!< Debug Interface */
} cuphyPrachStatPrms_t;

/**
 * @brief Occasion specific Quasi-static parameters for PRACH receiver processing
 */
typedef struct _cuphyPrachOccaQuasiStatPrms
{
    // uint16_t cellPrmStatIdx;         /*!< Index to cell-static parameter information (index into pCellPrms in cuphyPrachStatPrms_t) */
    uint16_t prachRootSequenceIndex; /*!< 0-137 for short preamble, 0-837 for long preamble */
    uint8_t  prachZeroCorrConf;      /*!< valid values 0-15 */
} cuphyPrachOccaQuasiStatPrms_t;

/**
 * @brief Cell specific Quasi-static parameters for PRACH receiver processing
 */
typedef struct _cuphyPrachCellQuasiStatPrms
{
    uint16_t                             cellPrmStatIdx;     /*!< Index to cell-static parameter information (index into pCellPrms in cuphyPrachStatPrms_t) */
    uint8_t                              configurationIndex; /*!< valid values 0-255 */
    uint8_t                              restrictedSet;      /*!< Only supported value 0 */
    uint8_t                              nFdmOccasions;      /*!< Number of FDM occasions for this cell <= cuphyPrachCellStatPrms.nMaxFdmOccasions, length of array pointed by pOccaPrms */
    cuphyPrachOccaQuasiStatPrms_t const* pOccaPrms;          /*!< Pointer to array of nFdmOccasions occasion specific parameters */
} cuphyPrachCellQuasiStatPrms_t;

/**
 * @brief Quasi-static parameters to process cell-group PRACH receiver
 */
typedef struct _cuphyPrachQuasiStatPrms
{
    // Cell specific
    uint16_t                             nCells;      /*!< Number of cells for which the cell specific quasi-static parameter is provided, length of array pointed to by pCellPrms */
    cuphyPrachCellQuasiStatPrms_t const* pCellPrms;   /*!< Pointer to array of cell specific Quasi-static parameters whose dimension is nCells */
} cuphyPrachQuasiStatPrms_t;

/*!< Parameters for update type CUPHY_STATE_UPDATE_TYPE_CREATE_STATIC */
typedef struct _cuphyPrachStatStateCreatePrms
{
    cuphyPrachStatPrms_t const*  pStatPrms;      /*!< Pointer to input PrachRx static parameters to be used to configure static state. */

    uint8_t                      nStatStates;     /*!< Number of static states to be created/initialized. */
    cuphyPrachRxStatStateHndl_t* pStatStateHndls; /*!< Pointer to an array (length atleast nStatStates) of handles to PrachRx internal static state.
                                                       Create and initialize nStatStates handles in pStatStateHndls array. */
} cuphyPrachStatStateCreatePrms_t;

/*!< Parameters for update type CUPHY_STATE_UPDATE_TYPE_CREATE_QUASI_STATIC */
typedef struct _cuphyPrachQuasiStatStateCreatePrms
{
    cuphyPrachRxStatStateHndl_t const  statStateHndl;       /*!< Handle to static state used during quasi-static state creation. Must be non-null. */
    cuphyPrachRxStatStateHndl_t const  quasiStatStateHndl;  /*!< If non-null: Handle to quasi-static state to be copied in during quasi-static state creation. This
                                                                              input quasi-static state is read-only and hence can be in use by cuPHY object Setup/Run processing.
                                                                 If null: no quasi-static state to import. */
    cuphyPrachQuasiStatPrms_t const*   pQuasiStatPrms;      /*!< Pointer to input PrachRx quasi-static parameters to be used to (re)configure quasi-static state. */

    uint8_t                           nQuasiStatStates;     /*!< Number of quasi-static states to be created/initialized. */
    cuphyPrachRxQuasiStatStateHndl_t* pQuasiStatStateHndls; /*!< Pointer to an array (length atleast nQuasiStatStates) of handles to PrachRx internal quasi-static state.
                                                                 if quasiStatStateHndl handle is null:
                                                                 (a) Allocate/initialize quasi-static state using configuration pointed by pQuasiStatPrms and statStateHndl
                                                                 (b) Write the state's handle into pQuasiStatStateHndls[i]

                                                                 if quasiStatStateHndl handle is non-null:
                                                                 (a) Allocates a container sized to house existing information in quasi-static state with handle quasiStatStateHndl
                                                                     and the new information determined using configuration pointed by pQuasiStatPrms and statStateHndl.
                                                                 (b) Copy information in existing quasi-static state into the newly allocated container.
                                                                 (c) Append the new quasi-static information into the container.
                                                                 (d) Write the new state's handle into pQuasiStatStateHndls[i] */
} cuphyPrachQuasiStatStateCreatePrms_t;

/*!< Parameters for update type CUPHY_STATE_UPDATE_TYPE_MODIFY */
typedef struct _cuphyPrachQuasiStatStateModifyPrms
{
    cuphyPrachRxStatStateHndl_t const  statStateHndl;       /*!< Handle to static state used during quasi-static state modification. Must be non-null. */
    cuphyPrachQuasiStatPrms_t const*   pQuasiStatPrms;      /*!< Pointer to input PrachRx quasi-static parameters to be used to (re)configure quasi-static state. */

    uint8_t                           nQuasiStatStates;     /*!< Number of quasi-static states to be modified. */
    cuphyPrachRxQuasiStatStateHndl_t* pQuasiStatStateHndls; /*!< Pointer to an array (length atleast nQuasiStatStates) of handles to PrachRx internal quasi-static state
                                                                 to be modified based on information determined using configuration pointed by pQuasiStatPrms and statStateHndl. */
} cuphyPrachQuasiStatStateModifyPrms_t;

/*!< Parameters for update type CUPHY_STATE_UPDATE_TYPE_DESTROY */
typedef struct _cuphyPrachStatStateDestroyPrms
{
    uint8_t                      nStatStates;     /*!< Number of static states to be destroyed. */
    cuphyPrachRxStatStateHndl_t* pStatStateHndls; /*!< Pointer to an array (length atleast nStatStates) of handles to PrachRx internal static state.
                                                       Free resources and destroy the nStatStates handles in pStatStateHndls array. */
} cuphyPrachStatStateDestroyPrms_t;

typedef struct _cuphyPrachQuasiStatStateDestroyPrms
{
    uint8_t                           nQuasiStatStates;     /*!< Number of quasi-static states to be destroyed.*/
    cuphyPrachRxQuasiStatStateHndl_t* pQuasiStatStateHndls; /*!< Pointer to an array (length atleast nQuasiStatStates) of handles to PrachRx internal quasi-static state.
                                                                 nQuasiStatStates handles in pStatStateHndls array are used to free resources and destroyed. */
} cuphyPrachQuasiStatStateDestroyPrms_t;

typedef struct _cuphyPrachStateDestroyPrms
{
    cuphyPrachStatStateDestroyPrms_t*      pStatStatePrms;      /*!< If non-null, results in destruction of static states and their handles.
                                                                     If null, then static state destroy operations are skipped. */
    cuphyPrachQuasiStatStateDestroyPrms_t* pQuasiStatStatePrms; /*!< If non-null, results in destruction of quasi-static states and their handles.
                                                                     If null, then quasi-static state destroy operations are skipped. */
} cuphyPrachStateDestroyPrms_t;

typedef struct _cuphyPrachStatePrms
{
    cuphyStateUpdateType_t            stateUpdateType;  /*!< Determines the type of operation (Create/Modify/Destroy) and hence the type of
                                                             parameters from the union prms to be used. */
    union
    {
        cuphyPrachStatStateCreatePrms_t*      pStaticCreatePrms;      /*!< Parameters used for state update type: CUPHY_STATE_UPDATE_TYPE_CREATE_STATIC */
        cuphyPrachQuasiStatStateCreatePrms_t* pQuasiStaticCreatePrms; /*!< Parameters used for state update type: CUPHY_STATE_UPDATE_TYPE_CREATE_QUASI_STATIC */
        cuphyPrachQuasiStatStateModifyPrms_t* pModifyPrms;            /*!< Parameters used for state update type: CUPHY_STATE_UPDATE_TYPE_MODIFY */
        cuphyPrachStateDestroyPrms_t*         pDestroyPrms;           /*!< Parameters used for state update type: CUPHY_STATE_UPDATE_TYPE_DESTROY */
    } prms;
} cuphyPrachStatePrms_t;

typedef struct _cuphyPrachStateUpdatePrms
{
    cudaStream_t          cuStream;  /*!< CUDA stream on which state update work occurs */
    cuphyPrachStatePrms_t statePrms; /*!< State update parameters */
} cuphyPrachStateUpdatePrms_t;

#endif // CUPHY_PRACH_RECONFIG_API

typedef struct _cuphyPrachOccaDynPrms
{
    /* Note on usage of occaPrmStatIdx, occaPrmDynIdx: these indices are used to index the static and dynamic API parameters as described below.

       For a given PRACH pipeline setup invocation, the static and dynamic parameters for occasion "i" (i being the ith occasion cuphyPrachDynPrms.pOccaPrms[i]) are accessed as follows:
       - Indexing input static API structure: cuphyPrachStatPrms.pOccaPrms[cuphyPrachDynPrms.pOccaPrms[i].occaPrmStatIdx]
       - Indexing input dynamic tensor parameter array: pTDataRx[cuphyPrachDynPrms.pOccaPrms[i].occaPrmDynIdx]
       - Indexing output dynamic tensor (derived from tensor parameters in cuphyPrachDataOut): for e.g. tNumDetectedPrmb(cuphyPrachDynPrms.pOccaPrms[i].occaPrmDynIdx)
       - Note: occaPrmDynIdx is used for indexing occasion specific input and output valued in the dynamic API

       Example:
       Suppose cuphyPrachStatPrms.nMaxCells = 5 with 2 occasions each per cell.
       The 2 occasion static parameters for cell "c" (c = [0,4]) are: {cuphyPrachStatPrms.pOccaPrms[c*2], cuphyPrachStatPrms.pOccaPrms[c*2 + 1]}
       Suppose PRACH for cell c = 3 needs to be processed and input occasion buffers are pTDataRx[0], pTDataRx[3]
       - cell3 occasion0: cuphyPrachDynPrms.pOccaPrms[0].occaPrmStatIdx = 6, cuphyPrachDynPrms.pOccaPrms[0].occaPrmDynIdx = 0
       - cell3 occasion1: cuphyPrachDynPrms.pOccaPrms[0].occaPrmStatIdx = 7, cuphyPrachDynPrms.pOccaPrms[1].occaPrmDynIdx = 3
       - Since occaPrmDynIdx is used for both input and output indexing, number of detected preambles for cell3, occasion1 would be located at tNumDetectedPrmb(3)
    */
    uint16_t occaPrmStatIdx; /*!< Index to occasion static/quasi-static parameter information (index into pOccaPrms created during PrachRx object construction) */
    uint16_t occaPrmDynIdx;  /*!< Index to occasion-dynamic parameter information (index into: pTDataRx in cuphyPrachDataIn_t, numDetectedPrmb in cuphyPrachDataOut_t, ...) */

    float    force_thr0;     /*!< 0: use the default threshold computed by cuPHY, > 0: use this value as threshold (overwrite cuPHY computed threshold) */
    
    uint16_t nUplinkStreams;
} cuphyPrachOccaDynPrms_t;

/**
 * PRACH Data Input
 */
typedef struct _cuphyPrachDataIn
{
    cuphyTensorPrm_t* pTDataRx; /*!< Array of tensors with each tensor (indexed by occaPrmDynIdx) representing the PRACH occasion buffer */
} cuphyPrachDataIn_t;

/**
 * PRACH Data Output
 */
typedef struct _cuphyPrachDataOut
{
    cuphyTensorPrm_t numDetectedPrmb;     /*!< 1D Tensor containing the number of detected preambles (<= CUPHY_PRACH_RX_NUM_PREAMBLE), dim: nOccaProc */
    cuphyTensorPrm_t prmbIndexEstimates;  /*!< 2D Tensor containing per cell preamble indices (length NumDetectedPrmb), dim: CUPHY_PRACH_RX_NUM_PREAMBLE, nOccaProc */
    cuphyTensorPrm_t prmbDelayEstimates;  /*!< 2D Tensor containing per cell delay estimate for the detected preamble (length NumDetectedPrmb), dim: CUPHY_PRACH_RX_NUM_PREAMBLE, nOccaProc */
    cuphyTensorPrm_t prmbPowerEstimates;  /*!< 2D Tensor containing per cell power estimate for the detected preamble (length NumDetectedPrmb), dim: CUPHY_PRACH_RX_NUM_PREAMBLE, nOccaProc */
    cuphyTensorPrm_t rssi;                /*!< 1D Tensor containing per occasion RSSI, dim: nOccaProc; use pinned memory */
    cuphyTensorPrm_t antRssi;             /*!< 2D Tensor containing per antenna, per occasion RSSI indices (length N_ant in cuphyPrachCellStatPrms_t), dim: MAX_N_ANTENNAS_SUPPORTED, nOccaProc; , use pinned memory */
    cuphyTensorPrm_t interference;        /*!< 1D Tensor containing per occasion interference in dB, dim: nOccaProc; use pinned memory */
} cuphyPrachDataOut_t;

/**
 * PRACH Dynamic Debug Parameters
 */
typedef struct _cuphyPrachDynDbgPrms
{
    uint8_t     enableApiLogging;   /// control the API logging of PRACH static parameters
} cuphyPrachDynDbgPrms_t;

/**
 * PRACH Dynamic Parameters
 */
typedef struct _cuphyPrachDynPrms
{
    cudaStream_t             cuStream;      /*!< CUDA stream on which pipeline is launched */

    uint64_t                 procModeBmsk;  /*!< Processing modes */

    uint16_t                 nOccaProc; /*!< Number of occasions to be processed. Length of array pointed to by pOccaPrms. nOccaProc <= nMaxOccaProc */
    cuphyPrachOccaDynPrms_t* pOccaPrms; /*!< Pointer to array of occasion specific dynamic parameters */

    /** Data parameters */
    cuphyPrachDataIn_t*       pDataIn;  /*!< Pointer to PRACH data input */
    cuphyPrachDataOut_t*      pDataOut; /*!< Pointer to PRACH data output */

    /** status parameter */
    cuphyPrachStatusOut_t*    pStatusOut;
    cuphyPrachDynDbgPrms_t*   pDbg;     /*!< Debug parameters */

} cuphyPrachDynPrms_t;

//-----------------------------------------------------------------------------------------------------------
// Functions

/**
 * @brief validate the PRACH parameters
 * @param pPrachParams - pointer to the PRACH static parameters for validation
 * @return CUPHY_STATUS_SUCCESS if the parameters are valid, CUPHY_STATUS_INVALID_ARGUMENT otherwise
 */
cuphyStatus_t CUPHYWINAPI cuphyValidatePrachParams(cuphyPrachStatPrms_t const* pPrachParams);

#ifndef CUPHY_PRACH_RECONFIG_API

/******************************************************************/ /**
 * \brief Allocate and initialize a cuPHY PRACH pipeline
 *
 * Allocate a cuPHY PRACH receiver pipeline and returns a handle in the
 * address provided by the caller.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pPrachRxHndl and/or \p pStatPrms is NULL.
 *
 * Returns ::CUPHY_STATUS_ALLOC_FAILED if a PrachRx object cannot be allocated
 *
 * Returns ::CUPHY_STATUS_SUCCESS if allocation and initialization were
 * successful.
 *
 * \param pPrachRxHndl - Address to return the new PrachRx instance
 * \param pStatPrms  - Pointer to PRACH static parameters to be used in pipeline creation
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_ALLOC_FAILED,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphySetupPrachRx,::cuphyRunPrachRx,::cuphyDestroyPrachRx
 */
cuphyStatus_t CUPHYWINAPI cuphyCreatePrachRx(cuphyPrachRxHndl_t* pPrachRxHndl, cuphyPrachStatPrms_t const* pStatPrms);

/******************************************************************/ /**
 * \brief Run cuPHY save Prach debug buffer
 * \param pPrachRxHndl - Handle of PrachRx instance which saves the debug buffer
 * \param cuStream   - CUDA stream used for PrachRx pipeline execution
 * Note: requires stream synchronization during call
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 */
cuphyStatus_t CUPHYWINAPI cuphyWriteDbgBufSynchPrach(cuphyPrachRxHndl_t pPrachRxHndl, cudaStream_t cuStream);

#if 0
/******************************************************************/ /**
 * \brief Return pointer to cuphyMemoryFootprint of cuPHY PRACH
 *
 * Return pointer to cuphyMemoryFootprint of cuPHY PRACH receiver pipeline which tracks
 * the size of GPU memory allocations owned by that object.
 *
 * Returns nullptr if \p prachRxHndl is NULL
 *
 * \param prachRxHndl  - Handle of PrachTx instance
 *
 * \return
 * \p to cuphyMemoryFootprint
 *
 */
const void*  cuphyGetMemoryFootprintTrackerPrachRx(cuphyPrachRxHndl_t prachRxHndl);
#endif

/******************************************************************/ /**
 * \brief Setup cuPHY PRACH pipeline for slot processing
 *
 * Setup cuPHY PRACH receiver pipeline (and its components) state in
 * preparation towards slot execution
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pPrachRxHndl and/or \p pDynPrms
 * is NULL
 *
 * Returns ::CUPHY_STATUS_SUCCESS if allocation and initialization were
 * successful.
 *
 * \param pPrachRxHndl  - Handle of prachRx instance to be setup
 * \param pDynPrms     - Dynamic parameters carrying information needed for slot processing
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyCreatePrachRx,::cuphyRunPrachRx,::cuphyDestroyPrachRx
 */
cuphyStatus_t CUPHYWINAPI cuphySetupPrachRx(cuphyPrachRxHndl_t pPrachRxHndl, cuphyPrachDynPrms_t* pDynPrms);

/******************************************************************/ /**
 * \brief Run cuPHY PRACH pipeline processing in specified mode
 *
 * Call triggers cuPHY PRACH transmitter pipeline execution
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pPrachRxHndl
 *
 * Returns ::CUPHY_STATUS_SUCCESS if PrachRx execution is successful.
 *
 * \param pPrachRxHndl  - Handle of PrachRx instance which is to be triggered
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyCreatePrachRx,::cuphySetupPrachRx,::cuphyDestroyPrachRx
 */
cuphyStatus_t CUPHYWINAPI cuphyRunPrachRx(cuphyPrachRxHndl_t pPrachRxHndl);

/******************************************************************/ /**
 * \brief Destroy a cuPHY PRACH receiver pipeline object
 *
 * Destroy a cuPHY PRACH receiver pipeline object that was previously
 * created by ::cuphyCreatePrachRx. The handle provided to this function
 * should not be used for any operations after this function returns.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pPrachRxHndl is NULL.
 *
 * Returns ::CUPHY_STATUS_SUCCESS if destruction was successful.
 *
 * \param pPrachRxHndl - handle to previously allocated prachRx instance
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyCreatePrachRx
 */
cuphyStatus_t CUPHYWINAPI cuphyDestroyPrachRx(cuphyPrachRxHndl_t pPrachRxHndl);

#else // CUPHY_PRACH_RECONFIG_API

/******************************************************************/ /**
 * \brief Create/Modify/Destroy cuPHY PRACH pipeline internal static/quasi-static states
 *
 * Depending on the type of state update selected:
 * CUPHY_STATE_UPDATE_TYPE_CREATE_STATIC      : Create/allocate/initialize static internal states and their handles using input parameters.
 * CUPHY_STATE_UPDATE_TYPE_CREATE_QUASI_STATIC: Create/allocate/initialize quasi-static internal states and their handles using input parameters.
 * CUPHY_STATE_UPDATE_TYPE_MODIFY             : Modify quasi-static state corresponding to supplied handles using input parameters.
 * CUPHY_STATE_UPDATE_TYPE_DESTROY            : Free resources and destroy static and/or quasi-static internal states and their handles.
 *
 * The PrachRx state can only be used in a PrachRx object after the state update operations (both synchronous and asynchronous on the supplied CUDA stream) are complete.
 * Note that undefined behavior will result if:
 * (a) A state is used in a PrachRx object (as part of cuphySetupPrachRx, cuphyRunPrachRx processing) before state
 * update operations complete.
 * (b) State update operations such as modify/destroy are attempted when the state is in use in a PrachRx object
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pPrachStateUpdatePrms and/or pointers are NULL.
 *
 * Returns ::CUPHY_STATUS_ALLOC_FAILED if internal states cannot be allocated during create stage
 *
 * Returns ::CUPHY_STATUS_SUCCESS if Create/Modify/Destroy operations were successful.
 *
 * \param pPrachStateUpdatePrms - Pointer to PRACH state update parameters
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_ALLOC_FAILED,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyCreatePrachRx,::cuphyConfigPrachRx,::cuphySetupPrachRx,::cuphyRunPrachRx,::cuphyDestroyPrachRx
 */
cuphyStatus_t CUPHYWINAPI cuphyStateUpdatePrachRx(cuphyPrachStateUpdatePrms_t* pPrachStateUpdatePrms);

/******************************************************************/ /**
 * \brief Allocate and initialize a cuPHY PRACH pipeline
 *
 * Allocate a cuPHY PRACH receiver pipeline and returns a handle in the
 * address provided by the caller.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pPrachRxHndl and/or \p pStatPrms is NULL.
 *
 * Returns ::CUPHY_STATUS_ALLOC_FAILED if a PrachRx object cannot be allocated
 *
 * Returns ::CUPHY_STATUS_SUCCESS if allocation and initialization were
 * successful.
 *
 * \param pPrachRxHndl  - Address to return the new PrachRx instance
 * \param pStatPrms     - Pointer to PRACH static parameters to be used in pipeline creation
 * \param statStateHndl - Handle to PrachRx static state to be used by the PrachRx instance
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_ALLOC_FAILED,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyStateUpdatePrachRx,::cuphyConfigPrachRx,::cuphySetupPrachRx,::cuphyRunPrachRx,::cuphyDestroyPrachRx
 */
cuphyStatus_t CUPHYWINAPI cuphyCreatePrachRx(cuphyPrachRxHndl_t* pPrachRxHndl, cuphyPrachStatPrms_t const* pStatPrms, cuphyPrachRxStatStateHndl_t statStateHndl);


/******************************************************************/ /**
 * \brief Configure/Reconfigure cuPHY PRACH pipeline with Quasi-static state
 *
 * Applies the supplied quasi-static state to all the PrachRx cuPHY objects. This (re)config API processing
 * is expected to be very quick (pointer updates) since the heavy lifting involved in (re)configuration is
 * performed during cuphyStateUpdate create/modify calls. These cuphyStateUpdate create/modify calls used to
 * prepare the supplied quasi-static state must be complete (both synchronous and asynchronous activities) before
 * involing this API.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pPrachRxHndls is NULL.
 *
 * Returns ::CUPHY_STATUS_SUCCESS if reconiguration is successful.
 *
 * \param quasiStatStateHndl - Quasi-static state handle to be applied to PrachRx pipeline objects.
 * \param pPrachRxHndls      - Pointer to array of handles to PrachRx pipeline object instances to be associated with quasi-static state.
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyStateUpdatePrachRx,::cuphyCreatePrachRx,::cuphySetupPrachRx,::cuphyRunPrachRx,::cuphyDestroyPrachRx
 */
cuphyStatus_t CUPHYWINAPI cuphyConfigPrachRx(cuphyPrachRxQuasiStatStateHndl_t quasiStatStateHndl, cuphyPrachRxHndl_t* pPrachRxHndls);

/******************************************************************/ /**
 * \brief Setup cuPHY PRACH pipeline for slot processing
 *
 * Setup cuPHY PRACH receiver pipeline (and its components) state in
 * preparation towards slot execution
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pPrachRxHndl and/or \p pDynPrms
 * is NULL
 *
 * Returns ::CUPHY_STATUS_SUCCESS if allocation and initialization were
 * successful.
 *
 * \param prachRxHndl  - Handle of prachRx instance to be setup
 * \param pDynPrms     - Dynamic parameters carrying information needed for slot processing
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyStateUpdatePrachRx,::cuphyCreatePrachRx,::cuphyConfigPrachRx,::cuphyRunPrachRx,::cuphyDestroyPrachRx
 */
cuphyStatus_t CUPHYWINAPI cuphySetupPrachRx(cuphyPrachRxHndl_t prachRxHndl, cuphyPrachDynPrms_t* pDynPrms);

/******************************************************************/ /**
 * \brief Run cuPHY PRACH pipeline processing in specified mode
 *
 * Call triggers cuPHY PRACH transmitter pipeline execution
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pPrachRxHndl
 *
 * Returns ::CUPHY_STATUS_SUCCESS if PrachRx execution is successful.
 *
 * \param prachRxHndl  - Handle of PrachRx instance which is to be triggered
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyStateUpdatePrachRx,::cuphyCreatePrachRx,::cuphyConfigPrachRx,::cuphySetupPrachRx,::cuphyDestroyPrachRx
 */
cuphyStatus_t CUPHYWINAPI cuphyRunPrachRx(cuphyPrachRxHndl_t prachRxHndl);

/******************************************************************/ /**
 * \brief Destroy a cuPHY PRACH receiver pipeline object
 *
 * Destroy a cuPHY PRACH receiver pipeline object that was previously
 * created by ::cuphyCreatePrachRx. The handle provided to this function
 * should not be used for any operations after this function returns.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pPrachRxHndl is NULL.
 *
 * Returns ::CUPHY_STATUS_SUCCESS if destruction was successful.
 *
 * \param prachRxHndl - Handle to previously allocated prachRx instance
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyStateUpdatePrachRx,::cuphyCreatePrachRx,::cuphyConfigPrachRx,::cuphySetupPrachRx,::cuphyRunPrachRx
 */
cuphyStatus_t CUPHYWINAPI cuphyDestroyPrachRx(cuphyPrachRxHndl_t prachRxHndl);

#endif // CUPHY_PRACH_RECONFIG_API

/** @} */ /* END UL_CUPHY_PRACH_RECEIVER */

// Following snippet goes into a source file
#if 0
struct cuphyPdschBatchPrm
{
    // nBatchTypes - Number of configurations to be batched

    // Type-Length-IdxValue
    // batchTypeX      - Batch type identifier
    // batchTypeXSz    - Batch size of type batchTypeX
    // batchTypeValues - Batch type batchTypeX indices 0:batchTypeXSz-1

    // | nBatchTypes = N+1 |
    // | batchType0  | batchType0Sz | ueIdx[0] | ueIdx[1] | ... | ueIdx[batchType0Sz-1]
    // | batchType1  | batchType1Sz | ueIdx[0] | ueIdx[1] | ... | ueIdx[batchType1Sz-1]
    // ...
    // | batchTypeN  | batchTypeNSz | ueIdx[0] | ueIdx[1] | ... | ueIdx[batchTypeNSz-1]
    uint16_t* pLdpcEncBatchPrms;
};
typedef struct cuphyPdschBatchPrm cuphyPdschBatchPrm_t;
#endif
// End source file snippet

/**
 * @brief: Populate PdschDmrsParams struct on the host.
 * @param[in, out] h_dmrs_params: pointer to DMRS config params struct on the host. This struct is also used in modulation.
 * @param[in] dyn_params: pointer to dymanic parameters struct, provided during PDSCH setup, on the host.
 * @param[in] static_params: pointer to static cell parameters, provided during PDSCH creation, on the host.
 * @param[in,out] pdsch_ue_group_params: pointer to UE group specific parameters needed when CSI-RS parameters are present
 * @return CUPHY_STATUS_SUCCESS or CUPHY_STATUS_INVALID_ARGUMENT
 */
cuphyStatus_t CUPHYWINAPI cuphyUpdatePdschDmrsParams(PdschDmrsParams*            h_dmrs_params,
                                                     cuphyPdschDynPrms_t*        dyn_params,
                                                     const cuphyPdschStatPrms_t* static_params,
                                                     PdschUeGrpParams*           pdsch_ue_group_params);


/**
 * \defgroup CUPHY_PBCCH_TRANSMITTER SSB Transmitter
 *
 * This section describes the SS/SSB transmit pipeline functions of the cuPHY
 * application programming interface.
 *
 * @{
 */

struct cuphySsbTx;
/**
 * cuPHY PBCCH transmitter handle
 */
typedef struct cuphySsbTx* cuphySsbTxHndl_t;

/**
 * SSB Static Parameters API logging
 */
typedef struct _cuphySsbStatDbgPrms
{
    const char* pOutFileName;       /// output file capturing pipeline intermediate states. No capture if null.
    uint8_t     enableApiLogging;   /// control the API logging of SSB static parameters
} cuphySsbStatDbgPrms_t;

/**
 * SSB static parameters
 */
typedef struct _cuphySsbStatPrms
{
    cuphyTracker_t*     pOutInfo;      /*!< pointer to cuphyTracker_t. Its pMemoryFootprint pointer will be updated by cuPHY in channel objection creation. The caller will thus have access to the cuphyMemoryFootprint object of that cuPHY SSB channel object. The cuphyMemoryFootprint object tracks the size of the GPU memory allocations owned by that cuPHY SSB channel object. */
    uint16_t nMaxCellsPerSlot; /*!< Maximum number of supported cells FIXME expand */
    cuphySsbStatDbgPrms_t* pDbgPrms;  /// pointer to SSB static parameters API logging parameters
} cuphySsbStatPrms_t;


/**
 * SSB Dynamic parameters that are SSB block specific
 */
typedef struct _cuphyPerSsBlockDynPrms
{
    uint16_t f0;         /*!< Index of initial SSB subcarrier in [0, 273*12 - 240) range; PBCH spans 240 subcarriers.
                              This is where the PBCH starts; PSS and SSS start at (f0 + 56) */
    uint8_t  t0;         /*!< Index of initial SSB OFDM symbol in [0, OFDM_SYMBOLS_PER_SLOT - 4] range, as each SSB spans 4 OFDM symbols.
                              PSS is at t0; SSS is at t0 + 2, and PBCH in [t0+1, t0+3] OFDM symbols. */
    uint8_t  blockIndex; /*!< SS block index (0 - L_max); L_max can be at most 64 */
    float    beta_pss;   /*!< scaling factor for PSS (primary synchronization signal) */
    float    beta_sss;   /*!< scaling factor for SSS (secondary synchronization signal), PBCH data and DMRS */
    uint16_t cell_index; /*!< index into pPerCellSsbDynParams nCells wide array to retrieve cell information for the cell this SSB belongs to.
                              FIXME Could alternatively (or additionally) add a startOffset  and nSSBs to the cuphyPerCellSsbDynPrms_t
                              to specify where a cell's SSBs start in the pPerSsBlockParams array and their number respectively. */
    uint8_t  enablePrcdBf; /*!< Enable pre-coding for this SSB*/
    uint16_t pmwPrmIdx;    /*!< Index to pre-coding matrix array, i.e., to the pPmwPrms array of the cuphySsbDynPrms_t struct.
                                pmwPrmIdx will be irrelevant if enablePrcdBf is 0.*/

} cuphyPerSsBlockDynPrms_t;

/**
 * Cell-specific SSB dynamic parameters
 */
typedef struct _cuphyPerCellSsbDynPrms
{
    uint16_t NID;           /*!< Physical cell identifier */
    uint16_t nHF;           /*!< Half frame index (0 or 1) */
    uint16_t Lmax;          /*!< Max number of SS blocks in PBCH period (4,8,or 64) */
    uint16_t SFN;           /*!< frame index */
    uint16_t k_SSB;         /*!< SSB subcarrier offset [0, 31]*/
    uint16_t nF;            /*!< number of subcarriers for one slot [0, 273*12) */
    uint16_t slotBufferIdx; /*!< Index into output slot buffer tensor array (pTDataTx in cuphySsbDataOut_t) where the prepared SSBs for that cell should be written */
} cuphyPerCellSsbDynPrms_t;

/**
 * SSB Data Input
 */
typedef struct _cuphySsbDataIn
{
    uint32_t* pMibInput; /*!< Pointer to array of nSSBlocks MIB payloads across all cells, one element (the least significant 24 bits of 32 bits valid) per SSB.
                              The order of payloads in the array should match the order of SSBs in the pPerSsBlockParams array in cuphySsbDynPrms_t.
                              Reminder: the 24-bit MIB content is identical for different SSBs in the same slot for the same cell (at most 3 SSB can exist per slot for the same cell), so with the current API there is a slight data replication. cuPHY does not check the correctness of the MIB contents. */
    enum
    {
        CPU_BUFFER,
        GPU_BUFFER
    } pBufferType; /*!< pMibInput buffer type; currently only CPU_BUFFER is supported */
} cuphySsbDataIn_t;


/**
 * SSB Data Output
 */
typedef struct _cuphySsbDataOut
{
    cuphyTensorPrm_t* pTDataTx; /*!< Array of tensors with each tensor (indexed by slotBufferIdx) representing the slot buffer to be transmitted */
                                /*!< Note: Each tensor may have a different geometry */
} cuphySsbDataOut_t;


/**
 * SSB Dynamic Parameters
 */
typedef struct _cuphySsbDynPrms
{
    cudaStream_t cuStream; /*!< CUDA stream on which pipeline is launched. */

    /** Control parameters */
    uint64_t procModeBmsk;

    //FIXME Since the number of SSBs per cell is small (~3?), we could also consider consolidating the per-cell parameters with the SSB specific ones, if we'll end
    // up processing them in cuPHY at the SSB granularity.
    uint16_t                      nCells;               /*!< Number of cells for which SSB needs to be processed in this slot */
    cuphyPerCellSsbDynPrms_t*     pPerCellSsbDynParams; /*!< Array with nCells elements; cell-specific parameters are common across all SSBs in a given cell */

    uint16_t                      nSSBlocks;            /*!< Number of SSBs across all nCells */
    cuphyPerSsBlockDynPrms_t*     pPerSsBlockParams;    /*!< Array with nSSBlocks SSB-specific elements spanning all nCells cells */

    uint16_t                      nPrecodingMatrices;   /*!< number of precoding matrices */
    cuphyPmWOneLayer_t*           pPmwParams;           /*!< array of pre-coding matrices */

    CUgraph                       *chan_graph;          /* Pointer to root node of graph */

    /** Data parameters */
    cuphySsbDataIn_t const*       pDataIn;  /*!< Pointer to SSB data input  */
    cuphySsbDataOut_t*            pDataOut; /*!< Pointer to SSB data output */
} cuphySsbDynPrms_t;

// SSB processing modes
typedef enum _cuphySsbProcMode
{
    SSB_PROC_MODE_STREAMS = 0x0, // stream processing
    SSB_PROC_MODE_GRAPHS  = 0x1, // graph processing
} cuphySsbProcMode_t;

/******************************************************************/ /**
 * \brief Allocate and initialize a cuPHY SSB pipeline
 *
 * Allocate a cuPHY SSB transmitter pipeline and return a handle in the
 * address provided by the caller.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pSsbTxHdnl or \p pStatPrms is NULL.
 *
 * Returns ::CUPHY_STATUS_ALLOC_FAILED if a SsbTx object cannot be allocated
 *
 * Returns ::CUPHY_STATUS_SUCCESS if allocation and initialization were
 * successful.
 *
 * \param pSsbTxHndl - Address to return the new SsbTx instance
 * \param pStatPrms    - Pointer to SSB static parameters to be used in pipeline creation
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_ALLOC_FAILED,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphySetupSsbTx,::cuphyRunSsbTx,::cuphyDestroySsbTx
 */
cuphyStatus_t CUPHYWINAPI cuphyCreateSsbTx(cuphySsbTxHndl_t* pSsbTxHndl, cuphySsbStatPrms_t const* pStatPrms);

#if 0
/******************************************************************/ /**
 * \brief Return pointer to cuphyMemoryFootprint of cuPHY SSB
 *
 * Return pointer to cuphyMemoryFootprint of cuPHY SSB transmitter pipeline which tracks
 * the size of GPU memory allocations owned by that object.
 *
 * Returns nullptr if \p ssbTxHndl is NULL
 *
 * \param ssbTxHndl  - Handle of SsbTx instance
 *
 * \return
 * \p to cuphyMemoryFootprint
 *
 */
const void*  cuphyGetMemoryFootprintTrackerSsbTx(cuphySsbTxHndl_t ssbTxHndl);
#endif

/******************************************************************/ /**
 * \brief Setup cuPHY SSB pipeline for slot processing
 *
 * Setup cuPHY SSB transmitter pipeline (and its components) state in
 * preparation towards slot execution
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p ssbTxHndl and/or \p pDynPrms
 * is NULL
 *
 * Returns ::CUPHY_STATUS_SUCCESS if allocation and initialization were
 * successful.
 *
 * \param ssbTxHndl  - Handle of SsbTx instance to be setup
 * \param pDynPrms     - Dynamic parameters carrying information needed for slot processing
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyCreateSsbTx,::cuphyRunSsbTx,::cuphyDestroySsbTx
 */
cuphyStatus_t CUPHYWINAPI cuphySetupSsbTx(cuphySsbTxHndl_t ssbTxHndl, cuphySsbDynPrms_t* pDynPrms);


/******************************************************************/ /**
 * \brief Run cuPHY SSB pipeline processing in specified mode
 *
 * Call triggers cuPHY SSB transmitter pipeline execution in mode specified
 * by procModeBmsk
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p ssbTxHndl is NULL and/or
 * procModeBmsk is not supported.
 *
 * Returns ::CUPHY_STATUS_SUCCESS if SsbTx execution is successful.
 *
 * \param ssbTxHndl  - Handle of SsbTx instance which is to be triggered
 * \param procModeBmsk - Processing mode bitmask containing one or more processing modes applicable during this execution. Currently unused.
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyCreateSsbTx,::cuphySetupSsbTx,::cuphyDestroySsbTx
 */
cuphyStatus_t CUPHYWINAPI cuphyRunSsbTx(cuphySsbTxHndl_t ssbTxHndl, uint64_t procModeBmsk);

/******************************************************************/ /**
 * \brief Destroy a cuPHY SSB transmit pipeline object
 *
 * Destroy a cuPHY SSB transmitter pipeline object that was previously
 * created by ::cuphyCreateSsbTx. The handle provided to this function
 * should not be used for any operations after this function returns.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p ssbTxHndl is NULL.
 *
 * Returns ::CUPHY_STATUS_SUCCESS if destruction was successful.
 *
 * \param ssbTxHndl - handle to previously allocated SsbTx instance
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyCreateSsbTx
 */
cuphyStatus_t CUPHYWINAPI cuphyDestroySsbTx(cuphySsbTxHndl_t ssbTxHndl);

/******************************************************************/ /**
 * \brief Write SSB debug data to HDF5 file
 *
 * Write SSB transmitter debug data to HDF5 file if debug output is enabled.
 * This function should be called after cuphyRunSsbTx to capture the 
 * pipeline intermediate states and output data.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p ssbTxHndl is NULL.
 *
 * Returns ::CUPHY_STATUS_SUCCESS if debug data writing is successful.
 *
 * \param ssbTxHndl - Handle of SsbTx instance
 * \param cuStream - CUDA stream for synchronization
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyCreateSsbTx,::cuphySetupSsbTx,::cuphyRunSsbTx,::cuphyDestroySsbTx
 */
cuphyStatus_t CUPHYWINAPI cuphyWriteDbgBufSynchSsb(cuphySsbTxHndl_t ssbTxHndl, cudaStream_t cuStream);

/** @} */ /* END CUPHY_SSB_TRANSMITTER */

#if defined(__cplusplus)
} /* extern "C" */
#endif /* defined(__cplusplus) */

#endif /* !defined(CUPHY_API_H_INCLUDED_) */
