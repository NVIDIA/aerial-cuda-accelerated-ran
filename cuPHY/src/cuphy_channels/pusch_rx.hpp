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

#if !defined(PUSCH_RX_HPP_INCLUDED_)
#define PUSCH_RX_HPP_INCLUDED_

#include "pusch_utils.hpp"
#include <vector>
#include <string>
#include <variant>

#include <gsl-lite/gsl-lite.hpp>

#include "ldpc/ldpc_api.hpp"
#include "ldpc/ldpc_cb_kernel_api.h"
#include "rate_matching/rate_matching.hpp"
#include "cuphy_hdf5.hpp"
#include "cuphy.hpp"
#include "tensor_desc.hpp"
#include "ch_est/IModule.hpp"
#include "ch_est/ch_est_settings.hpp"

// for memory tracing
#include "memtrace.h"

#define LLR_FP16
#define FRONT_END_DESCR 1

// ldpc multi-stream test for huge TB, temporary hack
#define SPLIT_LDPC 0


// Used in case of cond. if nodes or device graphs to allow us to enable a subset of the 3 cond. IF nodes or device graphs. By default all 3 should be enabled.
// Also a value of 1 does not say anything about the condition of an IF node, but whether this IF node exists in the first place.
// Please note that not all possible permutations of USE_COND_GRAPH_NODE-C{0-2} are meaningful/supported, esp. in device graphs mode.
#if 1
#define USE_COND_GRAPH_NODE_C0 1
#define USE_COND_GRAPH_NODE_C1 1
#define USE_COND_GRAPH_NODE_C2 1
#else
#define USE_COND_GRAPH_NODE_C0 0
#define USE_COND_GRAPH_NODE_C1 0
#define USE_COND_GRAPH_NODE_C2 0
#endif

#define DEFAULT_COND_VAL 1 // Default value used for cond. handle, if CU_GRAPH_COND_ASSIGN_DEFAULT flags was used during handle creation.
                           // Not currently exercised, but needs to be passed to conditional handle creation call.

#define PUSCH_USE_BATCHED_MEMCPY 1 // enables batched async memcpy along with static parameter field
#define PUSCH_MAX_OUTPUT_TO_CPU_COPIES (17 + MAX_CELLS_PER_SLOT - 1 + MAX_N_USER_GROUPS_SUPPORTED) // GT-11677 Phase 2: per-cell TB D2H (up to MAX_CELLS_PER_SLOT) + per-UE-group H-est D2H (up to MAX_N_USER_GROUPS_SUPPORTED)

static constexpr unsigned int BYTES_PER_WORD               = sizeof(uint32_t) / sizeof(uint8_t);

struct cuphyPuschRx
{};

struct condGraphInfo final {
//#if TRY_COND_GRAPH_NODES
    // Conditional graph nodes
    CUgraphNode m_graph_G0_cond_node{}, m_graph_G1_cond_node{}, m_graph_G2_cond_node{};
    // Conditional handles for the previous nodes
#if CUDA_VERSION >= 12040
    CUgraphConditionalHandle m_conditional_node_C0_handle{}, m_conditional_node_C1_handle{}, m_conditional_node_C2_handle{};

    // Graph node parameters
    CUgraphNodeParams m_conditional_node_C0_params = { CU_GRAPH_NODE_TYPE_CONDITIONAL };
    CUgraphNodeParams m_conditional_node_C1_params = { CU_GRAPH_NODE_TYPE_CONDITIONAL };
    CUgraphNodeParams m_conditional_node_C2_params = { CU_GRAPH_NODE_TYPE_CONDITIONAL };
#endif
//#endif

    // Kernel nodes that initialize the condition for the conditional graph nodes
    // or trigger a device graph launch (tail launch)
//#if (TRY_COND_GRAPH_NODES && USE_KERNEL_TO_SET_COND_VAL) || TRY_DGL_INSTEAD_OF_COND_GRAPHS
    CUgraphNode m_graph_G0_init_cond_node{}, m_graph_G1_init_cond_node{}, m_graph_G2_init_cond_node{};
    CUDA_KERNEL_NODE_PARAMS m_init_C0_node_params{}, m_init_C1_node_params{}, m_init_C2_node_params{};
//#endif

    //Helper pointers
//#if TRY_DGL_INSTEAD_OF_COND_GRAPHS
    CUgraphExec m_graphExec[3]{};
    CUgraph m_graph[3]{};
//#endif
    CUgraph* m_pGraph[3]{};
};

// helper to represent the graph + parent nodes for a conditional stage (C0/C1/C2)
struct CondStage
{
    CUgraph* pGraph = nullptr;              // graph where subsequent nodes should be added
    std::vector<CUgraphNode> parents;       // parent nodes for the next stage
};

// Helper to return the terminal nodes of different pipeline stage functions
struct StageResult
{
    std::vector<CUgraphNode> terminalNodes;  // Last nodes of this stage
};

/** State for the transport-block LDPC decoder path. */
struct TbLdpcState
{
    std::unique_ptr<cuphy::LDPC_decoder>       decoder;          ///< TB decoder instance.
    cuphy::LDPC_decode_desc_set                decode_desc_set;  ///< Reusable TB decode descriptors.
    std::vector<cuphyLDPCDecodeLaunchConfig_t> launch_configs;   ///< Per-TB launch configurations.
    cuphy::stream_pool                         stream_pool;      ///< Streams used by the TB decoder.
    std::vector<std::vector<CUgraphNode>>      decoder_nodes;    ///< Graph nodes for decoder launches.
    std::vector<uint8_t> decoder_nodes_enabled[CUPHY_MAX_PUSCH_FULL_SLOT_PROC_MODES];  ///< Node-enable flags by mode.

    /**
     * Constructs TB decoder state with descriptor capacity for the requested number of transport blocks.
     *
     * @param[in] ctx cuPHY context that owns the decoder resources.
     * @param[in] maxTbsPerDesc Maximum transport blocks represented by one decode descriptor.
     */
    TbLdpcState(cuphy::context& ctx, uint8_t maxTbsPerDesc)
        : decoder(std::make_unique<cuphy::LDPC_decoder>(ctx))
        , decode_desc_set(maxTbsPerDesc)
        , stream_pool()
    {}

    /**
     * Moves TB decoder state.
     *
     * @param[in] other State whose resources are transferred to this object.
     */
    TbLdpcState(TbLdpcState&& other) = default;
    /**
     * Replaces this object with moved TB decoder state.
     *
     * @param[in] other State whose resources are transferred to this object.
     * @return This object after the assignment.
     */
    TbLdpcState& operator=(TbLdpcState&& other) = default;
    TbLdpcState(const TbLdpcState&) = delete;
    TbLdpcState& operator=(const TbLdpcState&) = delete;
};

/** State for the codeblock-centric LDPC decoder path. */
struct CbLdpcState
{
    /** Per-phase set of prepared codeblock decoder launches. */
    struct CbDecodeBatch
    {
        /** Storage and placement for one group of codeblocks with an equivalent kernel choice. */
        struct Config
        {
            cuphyLdpcCbSubgroupKey_t key{};             ///< Kernel-selection attributes for the group.
            uint16_t                 num_cb = 0;        ///< Number of codeblocks in the group.
            size_t                   data_offset = 0;   ///< Offset into @ref cb_data.
            uint16_t                 partition_id = 0;  ///< Separates otherwise-equivalent groups when required.
        };

        std::vector<Config>                                configs;            ///< Group configurations.
        std::vector<cuphyLdpcCbData_t>                     cb_data;            ///< Flattened CB data by group.
        std::vector<std::vector<cuphyLdpcCbData_t>>         cb_data_by_config;  ///< CB data retained per group.
        std::vector<std::vector<cuphyLdpcCbBufferSpan_t>>   span_storage;       ///< Backing span storage per group.
        std::vector<cuphyLdpcCbSubgroupDesc_t>             subgroups;          ///< Subgroup descriptors per group.
        std::vector<cuphyLdpcCbLaunchBatchDesc_t>          batch_descs;        ///< Prepared-launch batch descriptors.
        std::vector<cuphyLdpcCbLaunchFamily_t>             families;           ///< Launch family selected per group.
        std::vector<cuphyLdpcCbPreparedLaunch_t>           prepared_launches;  ///< Prepared launch handles.
        std::vector<std::vector<uint8_t>>                  launch_workspaces;  ///< Aligned backing workspace per group.
        std::vector<CUDA_KERNEL_NODE_PARAMS>               node_params;        ///< CUDA graph node parameters per group.
        uint16_t                                           total_cb_count = 0;  ///< Total codeblocks in this batch.

        /** Releases every prepared-launch handle owned by this batch. */
        void deinitLaunches()
        {
            for (auto& launch : prepared_launches)
            {
                if (launch != nullptr)
                {
                    cuphyLdpcCbPreparedLaunchDeinit(launch);
                    launch = nullptr;
                }
            }
        }

        /**
         * Preallocates per-group storage for the maximum expected batch shape.
         *
         * @param[in] max_configs Maximum number of kernel-choice groups.
         * @param[in] max_codeblocks Maximum codeblocks retained by each group.
         * @param[in] max_workspace_bytes Maximum workspace bytes required by one group.
         * @param[in] workspace_alignment_bytes Additional bytes reserved for workspace alignment.
         */
        void reserve(const size_t max_configs,
                     const size_t max_codeblocks,
                     const size_t max_workspace_bytes,
                     const size_t workspace_alignment_bytes)
        {
            configs.reserve(max_configs);
            cb_data.reserve(max_codeblocks);
            cb_data_by_config.resize(max_configs);
            span_storage.resize(max_configs);
            subgroups.resize(max_configs);
            batch_descs.resize(max_configs);
            families.resize(max_configs);
            prepared_launches.resize(max_configs, nullptr);
            launch_workspaces.resize(max_configs);
            node_params.resize(max_configs);

            const size_t workspace_capacity = max_workspace_bytes + workspace_alignment_bytes;
            for (size_t idx = 0; idx < max_configs; ++idx)
            {
                cb_data_by_config[idx].reserve(max_codeblocks);
                span_storage[idx].reserve(max_codeblocks);
                launch_workspaces[idx].resize(workspace_capacity);
            }
        }

        /** Clears setup-specific data while retaining preallocated capacity. */
        void resetForSetup()
        {
            deinitLaunches();
            configs.clear();
            cb_data.clear();
            for (auto& per_config_data : cb_data_by_config)
            {
                per_config_data.clear();
            }
            total_cb_count = 0;
        }
    };

    struct CachedChoice
    {
        cuphyLdpcCbSubgroupKey_t  key{};     ///< Attributes used for kernel selection.
        cuphyLdpcCbKernelChoice_t choice{};  ///< Selected kernel implementation.
        cuphyLdpcCbLaunchFamily_t family{};  ///< Launch family for @ref choice.
    };

    cuphyLdpcCbLaunchPreparer_t                preparer;      ///< API object that prepares CB launches.
    CbDecodeBatch                              full_batch;    ///< Full-slot CB decode batch.
    CbDecodeBatch                              early_batch;   ///< Early-SCH CB decode batch.
    std::vector<CachedChoice>                  choice_cache;  ///< Intra-slot selection cache.
    std::vector<std::vector<CUgraphNode>>      full_nodes;    ///< Graph nodes for full-slot batches.
    std::vector<CUgraphNode>                   early_nodes;   ///< Graph nodes for early-SCH batches.

    /** Constructs empty codeblock decoder state. */
    CbLdpcState()
        : preparer(nullptr)
    {
    }

    /** Releases prepared launches and the launch preparer. */
    ~CbLdpcState()
    {
        full_batch.deinitLaunches();
        early_batch.deinitLaunches();
        if (preparer)
        {
            cuphyDestroyLdpcCbLaunchPreparer(preparer);
            preparer = nullptr;
        }
    }

    /**
     * Moves codeblock decoder state, transferring ownership of the preparer.
     *
     * @param[in] other State whose resources are transferred to this object.
     */
    CbLdpcState(CbLdpcState&& other) noexcept
        : preparer(other.preparer)
        , full_batch(std::move(other.full_batch))
        , early_batch(std::move(other.early_batch))
        , choice_cache(std::move(other.choice_cache))
        , full_nodes(std::move(other.full_nodes))
        , early_nodes(std::move(other.early_nodes))
    {
        other.preparer = nullptr;
    }

    /**
     * Replaces this object with moved codeblock decoder state.
     *
     * @param[in] other State whose resources are transferred to this object.
     * @return This object after the assignment.
     */
    CbLdpcState& operator=(CbLdpcState&& other) noexcept
    {
        if (this != &other)
        {
            full_batch.deinitLaunches();
            early_batch.deinitLaunches();
            if (preparer != nullptr)
            {
                cuphyDestroyLdpcCbLaunchPreparer(preparer);
            }
            preparer = other.preparer;
            full_batch = std::move(other.full_batch);
            early_batch = std::move(other.early_batch);
            choice_cache = std::move(other.choice_cache);
            full_nodes = std::move(other.full_nodes);
            early_nodes = std::move(other.early_nodes);
            other.preparer = nullptr;
        }
        return *this;
    }

    CbLdpcState(const CbLdpcState&) = delete;
    CbLdpcState& operator=(const CbLdpcState&) = delete;
};

class PuschRx : public cuphyPuschRx {
public:
    enum DescriptorTypes
    {
        PUSCH_CH_EST                        = 0,
        PUSCH_NOISE_INTF_EST                = PUSCH_CH_EST + CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST,
        PUSCH_CH_EQ_COEF                    = PUSCH_NOISE_INTF_EST + 1,
        PUSCH_CH_EQ_SOFT_DEMAP              = PUSCH_CH_EQ_COEF + CUPHY_PUSCH_RX_MAX_N_TIME_CH_EQ,
        PUSCH_CH_EQ_IDFT                    = PUSCH_CH_EQ_SOFT_DEMAP + 1,
        PUSCH_CH_EQ_AFTER_IDFT              = PUSCH_CH_EQ_IDFT + 1,
        PUSCH_RATE_MATCH                    = PUSCH_CH_EQ_AFTER_IDFT + 1,
        PUSCH_RATE_MATCH_EARLY              = PUSCH_RATE_MATCH + 1,
        PUSCH_LDPC_DEC                      = PUSCH_RATE_MATCH_EARLY + 1,
        PUSCH_CRC                           = PUSCH_LDPC_DEC + 1,
        PUSCH_CFO_TA_EST                    = PUSCH_CRC + 1,
        PUSCH_RSSI                          = PUSCH_CFO_TA_EST + 1,
        PUSCH_RSRP                          = PUSCH_RSSI + 1,
        PUSCH_FRONT_END_PARAMS              = PUSCH_RSRP + 1,
        PUSCH_SEG_UCI_LLRS0                 = PUSCH_FRONT_END_PARAMS + 1,
        PUSCH_SEG_EARLY_UCI_LLRS0           = PUSCH_SEG_UCI_LLRS0 + 1,
        PUSCH_SEG_UCI_LLRS2                 = PUSCH_SEG_EARLY_UCI_LLRS0 + 1,
        PUSCH_UCI_CSI2_CTRL                 = PUSCH_SEG_UCI_LLRS2 + 1,
        POL_COMP_CW_TREE                    = PUSCH_UCI_CSI2_CTRL + 1,
        POL_COMP_CW_TREE_ADDRS              = POL_COMP_CW_TREE + 1,
        POL_SEG_DERM_DEITL                  = POL_COMP_CW_TREE_ADDRS + 1,
        POL_SEG_DERM_DEITL_CW_ADDRS         = POL_SEG_DERM_DEITL + 1,
        POL_SEG_DERM_DEITL_UCI_ADDRS        = POL_SEG_DERM_DEITL_CW_ADDRS + 1,
        POL_DECODE                          = POL_SEG_DERM_DEITL_UCI_ADDRS + 1,
        POL_DECODE_LLR_ADDRS                = POL_DECODE + 1,
        POL_DECODE_CB_ADDRS                 = POL_DECODE_LLR_ADDRS + 1,
        LIST_POL_DECODE_SCRATCH_ADDRS       = POL_DECODE_CB_ADDRS + 1,
        POL_COMP_CW_TREE_CSI2               = LIST_POL_DECODE_SCRATCH_ADDRS + 1,
        POL_COMP_CW_TREE_ADDRS_CSI2         = POL_COMP_CW_TREE_CSI2 + 1,
        POL_SEG_DERM_DEITL_CSI2             = POL_COMP_CW_TREE_ADDRS_CSI2 + 1,
        POL_SEG_DERM_DEITL_CW_ADDRS_CSI2    = POL_SEG_DERM_DEITL_CSI2 + 1,
        POL_SEG_DERM_DEITL_UCI_ADDRS_CSI2   = POL_SEG_DERM_DEITL_CW_ADDRS_CSI2 + 1,
        POL_DECODE_CSI2                     = POL_SEG_DERM_DEITL_UCI_ADDRS_CSI2 + 1,
        POL_DECODE_LLR_ADDRS_CSI2           = POL_DECODE_CSI2 + 1,
        POL_DECODE_CB_ADDRS_CSI2            = POL_DECODE_LLR_ADDRS_CSI2 + 1,
        LIST_POL_DECODE_SCRATCH_ADDRS_CSI2  = POL_DECODE_CB_ADDRS_CSI2 + 1,
        POL_COMP_CW_TREE_EARLY              = LIST_POL_DECODE_SCRATCH_ADDRS_CSI2 + 1,
        POL_COMP_CW_TREE_ADDRS_EARLY        = POL_COMP_CW_TREE_EARLY + 1,
        POL_SEG_DERM_DEITL_EARLY            = POL_COMP_CW_TREE_ADDRS_EARLY + 1,
        POL_SEG_DERM_DEITL_CW_ADDRS_EARLY   = POL_SEG_DERM_DEITL_EARLY + 1,
        POL_SEG_DERM_DEITL_UCI_ADDRS_EARLY  = POL_SEG_DERM_DEITL_CW_ADDRS_EARLY + 1,
        POL_DECODE_EARLY                    = POL_SEG_DERM_DEITL_UCI_ADDRS_EARLY + 1,
        POL_DECODE_LLR_ADDRS_EARLY          = POL_DECODE_EARLY + 1,
        POL_DECODE_CB_ADDRS_EARLY           = POL_DECODE_LLR_ADDRS_EARLY  + 1,
        LIST_POL_DECODE_SCRATCH_ADDRS_EARLY = POL_DECODE_CB_ADDRS_EARLY + 1,
        SPX_DECODE                          = LIST_POL_DECODE_SCRATCH_ADDRS_EARLY + 1,
        RM_DECODE                           = SPX_DECODE + 1,
        SPX_DECODE_CSI2                     = RM_DECODE + 1,
        RM_DECODE_CSI2                      = SPX_DECODE_CSI2 + 1,
        SPX_DECODE_EARLY                    = RM_DECODE_CSI2 + 1,
        RM_DECODE_EARLY                     = SPX_DECODE_EARLY + 1,
        N_PUSCH_DESCR_TYPES                 = RM_DECODE_EARLY + 1
    };
    struct OutputParams
    {
        // flag to copy outputs to host (cpu)
        bool                         cpuCopyOn;

        // size of outputs
        uint32_t                     totNumTbs;
        uint32_t                     totNumCbs;
        uint32_t                     totNumPayloadBytes;
        uint32_t                     totNumUciSegs;
        uint32_t                     totNumUciPayloadBytes;
        uint32_t                     dataRxHostCellStride;

        // device (GPU) output addresses
        uint32_t* pCbCrcsDevice;
        uint32_t* pTbCrcsDevice;
        uint8_t*  pTbPayloadsDevice;
        float*    pTaEstsDevice;
        float*    pRssiDevice;
        float*    pRssiEhqDevice;
        float*    pRsrpDevice;
        float*    pRsrpEhqDevice;
        float*    pNoiseVarPreEqDevice;
        float*    pNoiseVarPostEqDevice;
        float*    pSinrPreEqDevice;
        float*    pSinrPostEqDevice;
        float*    pCfoHzDevice;
        uint8_t*  pUciPayloadsDevice;
        uint8_t*  pUciCrcFlagsDevice_csi2;
        uint8_t*  pUciCrcFlagsDevice;
        uint8_t*  pUciCrcFlagsDevice_early;
        uint16_t* pNumCsi2BitsDevice;

        uint8_t*  pHarqDetectionStatusDevice;
        uint8_t*  pCsiP1DetectionStatusDevice;
        uint8_t*  pCsiP2DetectionStatusDevice;
        uint8_t*  pUciDTXsDevice;

        // host (CPU) output addresses
        __half2*                     pDataRxHost;
        uint32_t*                    pCbCrcsHost;
        uint32_t*                    pTbCrcsHost;
        uint8_t*                     pTbPayloadsHost;
        float*                       pTaEstsHost;
        float*                       pRssiHost;
        float*                       pRsrpHost;
        float*                       pNoiseVarPreEqHost;
        float*                       pNoiseVarPostEqHost;
        float*                       pSinrPreEqHost;
        float*                       pSinrPostEqHost;
        float*                       pCfoHzHost;
        float2*                      pChannelEstsHost;
        uint32_t*                    pChannelEstSizesHost;
        uint8_t*                     pUciPayloadsHost;
        uint8_t*                     pUciCrcFlagsHost;
        uint16_t*                    pNumCsi2BitsHost;

        uint8_t*                     pHarqDetectionStatusHost;
        uint8_t*                     pCsiP1DetectionStatusHost;
        uint8_t*                     pCsiP2DetectionStatusHost;
        //uint8_t*                     pUciDTXsHost;

        cuphyUciOnPuschOutOffsets_t* pUciOnPuschOutOffsets;

        uint32_t                     nPerCellTbDests{}; //!< Number of per-cell D2H TB payload destinations (GT-11677 Phase 2)
        uint8_t*                     pPerCellTbPayloadsHost[MAX_CELLS_PER_SLOT]{}; //!< Per-cell host destination pointers for batched D2H
        uint32_t                     perCellTbPayloadOffset[MAX_CELLS_PER_SLOT]{}; //!< Byte offset of each cell's TB data in device buffer
        uint32_t                     perCellTbPayloadSize[MAX_CELLS_PER_SLOT]{}; //!< Byte size of each cell's TB payload

        // output parameters
        bool                         debugOutputFlag;
        hdf5hpp::hdf5_file           outHdf5File;
    };

    enum H2DDataTypes
    {
        // note that PUSCH_TB_PRMS is assumed to be the first item in this list and should not be reordered
        PUSCH_TB_PRMS            = 0,
        PUSCH_SPX_PRMS           = PUSCH_TB_PRMS + 1,
        PUSCH_RM_CW_PRMS         = PUSCH_SPX_PRMS + 1,
        PUSCH_UCI_SEG_PRMS       = PUSCH_RM_CW_PRMS + 1,
        PUSCH_UCI_CW_PRMS        = PUSCH_UCI_SEG_PRMS + 1,
        PUSCH_UCI_SEG_CSI2_PRMS  = PUSCH_UCI_CW_PRMS + 1,
        PUSCH_UCI_CW_CSI2_PRMS   = PUSCH_UCI_SEG_CSI2_PRMS + 1,
        PUSCH_SPX_CSI2_PRMS      = PUSCH_UCI_CW_CSI2_PRMS + 1,
        PUSCH_RM_CW_CSI2_PRMS    = PUSCH_SPX_CSI2_PRMS + 1,
        PUSCH_UCI_SEG_EARLY_PRMS = PUSCH_RM_CW_CSI2_PRMS + 1,
        PUSCH_UCI_CW_EARLY_PRMS  = PUSCH_UCI_SEG_EARLY_PRMS + 1,
        PUSCH_SPX_EARLY_PRMS     = PUSCH_UCI_CW_EARLY_PRMS + 1,
        PUSCH_RM_CW_EARLY_PRMS   = PUSCH_SPX_EARLY_PRMS + 1,
        N_PUSCH_H2D_DATA_TYPES   = PUSCH_RM_CW_EARLY_PRMS + 1
    };
    template<size_t N_DATA> using dataBundle = cuphy::kernelDescrs<N_DATA>;

    PuschRx(cuphyPuschStatPrms_t const* pStatPrms, cudaStream_t cuStream);
    PuschRx(PuschRx const&) = delete;
    PuschRx& operator=(PuschRx const&) = delete;
    ~PuschRx();

    [[nodiscard]] cuphyStatus_t copyEarlyHarqOutputToCPU(cudaStream_t cuStrm);
    [[nodiscard]] cuphyStatus_t copyOutputToCPU(cudaStream_t cuStrm);

    // determine HARQ/CSI part 1/CSI part 2 detection status
    // void detStatus();

    void writeDbgBufSynch(cudaStream_t cuStrm);

    cuphyStatus_t setup(cuphyPuschDynPrms_t* pDynPrm);
    cuphyStatus_t run(cuphyPuschRunPhase_t runPhase);

    // These two are callable without instantiating PuschRx.
    static uint32_t expandFrontEndParameters(cuphyPuschDynPrms_t* pDynPrm,
                                             cuphyPuschStatPrms_t* pStatPrm,
                                             cuphyPuschRxUeGrpPrms_t* pDrvdUeGrpPrms,
                                             bool& subSlotProcessingFrontLoadedDmrsEnabled,
                                             uint8_t& maxDmrsMaxLen,
                                             const uint8_t enableRssiMeasurement,
                                             const uint32_t maxNPrbAlloc);
    static cuphyStatus_t expandBackEndParameters(cuphyPuschDynPrms_t* pDynPrm,
                                                 cuphyPuschStatPrms_t* pStatPrm,
                                                 cuphyPuschRxUeGrpPrms_t* pDrvdUeGrpPrms,
                                                 PerTbParams* pPerTbPrms,
                                                 cuphyLDPCParams& ldpcPrms,
                                                 const uint32_t maxNCbs,
                                                 const uint32_t maxNCbsPerTb);

    const void* getMemoryTracker();

    template <fmtlog::LogLevel log_level=fmtlog::DBG>
    static void printDynApiPrms(cuphyPuschDynPrms_t* pDynPrm);
    template <fmtlog::LogLevel log_level=fmtlog::DBG>
    static void printStaticApiPrms(cuphyPuschStatPrms_t const* pStaticPrm);
    template <typename T>
    void copyTensorRef2Info(cuphy::tensor_ref& tRef, T& tInfo)
    {
        tInfo.pAddr              = tRef.addr();
        const tensor_desc& tDesc = static_cast<const tensor_desc&>(*(tRef.desc().handle()));
        tInfo.elemType           = tDesc.type();
        std::copy_n(tDesc.layout().strides.begin(), std::extent<decltype(tInfo.strides)>::value, tInfo.strides);
    }

private:
    cuphyMemoryFootprint m_memoryFootprint;

    // setup functions
    cuphyStatus_t   setupCmnPhase1(cuphyPuschDynPrms_t* pDynPrm);
    cuphyStatus_t   setupCmnPhase2(cuphyPuschDynPrms_t* pDynPrm);
    void   allocateDeviceMemory(cuphyPuschDynPrms_t* pDynPrm);
    void   allocateDescr(void);
    void   allocateInputBuf(uint32_t nMaxTbs, uint32_t maxNumTbsSupported);
    void   updateInputBuf(void);
    size_t getBufferSize(cuphyPuschStatPrms_t const* pStatPrms);
    size_t getBufferSizeBluesteinWorkspace(cuphyPuschStatPrms_t const* pStatPrms);
    void   expandUciParameters(bool updateOnlyNumInputPrms);
    void   expandUciCodingPrms(uint32_t nInfoBits, uint32_t nRmBits, uint8_t Qm, float DTXthreshold, bool updateOnlyNumInputPrms,
    uint16_t& nRmCws, cuphyRmCwPrm_t* pRmCwPrms, uint16_t& nSpxCws, cuphySimplexCwPrm_t* pSpxCwPrms,
    uint16_t& nPolUciSegs, uint16_t& nPolCbs, cuphyPolarUciSegPrm_t* pUciSegPrms, cuphyPolarCwPrm_t* pUciCwPrms);

    void   allocAndLinkPolBuffers(cuphyPolarUciSegPrm_t&     uciSegPrms,
                                     const uint16_t&         polSegIdx,
                                     void*                   pSegLLRs,
                                     uint8_t*                pCrcStatus,
                                     uint32_t*               pUciSegEst,
                                     cuphyPolarCwPrm_t*      pUciCwPrmsCpu,
                                     std::vector<uint8_t*>&  cwTreeTypesAddrVec,
                                     std::vector<__half*>&   uciSegLLRsAddrVec,
                                     std::vector<__half*>&   cwTreeLLRsAddrVec,
                                     std::vector<__half*>&   cwLLRsAddrVec,
                                     std::vector<bool*>&     listPolScratchAddrVec,
                                     std::vector<uint32_t*>& cbEstAddrVec);

    // component functions (non-LDPC)
    void createComponents(cudaStream_t cuStrm,
                          int          rmFPconfig     = 3,
                          int          descramblingOn = 1);
    cuphyStatus_t setupComponents(bool enableCpuToGpuDescrAsyncCpy, cuphyPuschDynPrms_t* pDynPrm);
    void destroyComponents();

    // LDPC component functions
    [[nodiscard]] uint16_t computeLdpcMaxIters(uint16_t ueIdx) const;
    void prepareLDPCStreamsTB();
    void computeEarlyHarqCbDecodePlan();
    cuphyStatus_t prepareCbLdpcBatches(bool prepareEarlyHarqBatch);
    void addCbLdpcNodes(CUgraph graph,
                        const std::vector<CUgraphNode>& parents,
                        std::vector<CUgraphNode>& nodes);
    void updateCbLdpcNodes(CUgraphExec graphExec,
                           const std::vector<CUgraphNode>& nodes,
                           const std::vector<CUDA_KERNEL_NODE_PARAMS>& params,
                           size_t activeParamCount,
                           std::vector<uint8_t>& enabled,
                           bool enableNodes);
    void launchLDPCStreamsCB(cudaStream_t strm);
    void launchLDPCStreamsTensor(cudaStream_t strm);
    void launchLDPCStreamsTB(cudaStream_t strm);

    // CUDA graph building blocks =================================================================
    StageResult buildFrontEndStage(cuphyPuschFullSlotProcMode_t    fullSlotProcMode,
                                   CUgraph*                        pGraph,
                                   const std::vector<CUgraphNode>& initialParents,
                                   void*&                          arg);

    StageResult buildSoftDemapStage(cuphyPuschFullSlotProcMode_t    fullSlotProcMode,
                                    CUgraph*                        pGraph,
                                    const std::vector<CUgraphNode>& eqCoefDeps,
                                    void*&                          arg);

    StageResult buildSchBackendStage(cuphyPuschFullSlotProcMode_t    fullSlotProcMode,
                                     CUgraph*                        pGraph,
                                     const std::vector<CUgraphNode>& schParents);

    StageResult buildUciP1BackendStage(cuphyPuschFullSlotProcMode_t    fullSlotProcMode,
                                       CUgraph*                        pGraph,
                                       const std::vector<CUgraphNode>& softDemapParents);
                                       
    StageResult buildUciP1OffloadingBackendStage(cuphyPuschFullSlotProcMode_t    fullSlotProcMode,
                                                 CUgraph*                        pGraph,
                                                 const std::vector<CUgraphNode>& softDemapParents);

    StageResult buildCsiP2BackendStage(cuphyPuschFullSlotProcMode_t    fullSlotProcMode,
                                       CUgraph*                        pGraph,
                                       const std::vector<CUgraphNode>& uciP1Parents,
                                       bool                            useCondIfOrDglC2);
    // Conditional Graph helper functions ==========================================================
    CondStage enterConditionalStage0(CUgraph&                        fullSlotGraph,
                                     CUgraphNode&                    emptyRootNode,
                                     condGraphInfo&                  condInfo,
                                     CUcontext                       current_context,
                                     unsigned int                    cond_handle_flags,
                                     const std::vector<CUgraphNode>& initialParents);

    CondStage enterConditionalStage1(condGraphInfo&                  condInfo,
                                     CUcontext                       current_context,
                                     unsigned int                    cond_handle_flags,
                                     const std::vector<CUgraphNode>& parents,
                                     std::vector<CUgraphNode>&       dglParentsOut);

    CondStage enterConditionalStage2(condGraphInfo&                  condInfo,
                                     CUcontext                       current_context,
                                     unsigned int                    cond_handle_flags,
                                     bool                            use_cond_if_node_c2,
                                     const std::vector<CUgraphNode>& parentsForC2,
                                     std::vector<CUgraphNode>&       dglParentsOut);
    // ============================================================================================


    // graph functions
    void createFullSlotGraph(cuphyPuschFullSlotProcMode_t fullSlotProcMode, CUgraph& fullSlotGraph, CUgraphNode& emptyRootNode, condGraphInfo& condInfo);
    void createFullSlotOffloadingGraph(cuphyPuschFullSlotProcMode_t fullSlotProcMode, CUgraph& fullSlotGraph, CUgraphNode& emptyRootNode, condGraphInfo& condInfo);
    void updateFullSlotGraph(bool disableAllNodes, cuphyPuschFullSlotProcMode_t fullSlotProcMode, CUgraphExec graphExec, condGraphInfo& condInfo);
    void updateFullSlotOffloadingGraph(bool disableAllNodes, cuphyPuschFullSlotProcMode_t fullSlotProcMode, CUgraphExec graphExec, condGraphInfo& condInfo);
    
    /// Creates the CUDA graph for 72e full-slot processing mode.
    /// `@param`[in,out] fullSlotGraph The CUDA graph to populate
    /// `@param`[in,out] emptyRootNode Root node for the graph
    /// `@param`[in,out] condInfo Conditional graph information
    void create72eFullSlotGraph(CUgraph& fullSlotGraph, CUgraphNode& emptyRootNode, condGraphInfo& condInfo);
    void create72eFullSlotOffloadingGraph(CUgraph& fullSlotGraph, CUgraphNode& emptyRootNode, condGraphInfo& condInfo);
    /// Updates the CUDA graph for 72e full-slot processing mode.
    /// `@param`[in] disableAllNodes Whether to disable all nodes in the graph
    /// `@param`[in] fullSlotProcMode Full slot processing mode
    /// `@param`[in,out] graphExec Executable graph to update
    /// `@param`[in,out] condInfo Conditional graph information
    void update72eFullSlotGraph(bool disableAllNodes,  cuphyPuschFullSlotProcMode_t fullSlotProcMode, CUgraphExec graphExec, condGraphInfo& condInfo);
    void update72eFullSlotOffloadingGraph(bool disableAllNodes,  cuphyPuschFullSlotProcMode_t fullSlotProcMode, CUgraphExec graphExec, condGraphInfo& condInfo);

    void createEarlyHarqGraph();  // for early-HARQ portion of PUSCH pipeline
    void createEarlyHarqOffloadingGraph(); 
    void updateEarlyHarqGraph(bool disableAllNodes = false);
    void updateEarlyHarqOffloadingGraph(bool disableAllNodes = false);
    
    /// Creates the CUDA graph for 72e early HARQ processing mode.
    void create72eEarlyHarqGraph();
    void create72eEarlyHarqOffloadingGraph();
    /// Updates the CUDA graph for 72e early HARQ processing mode.
    /// `@param`[in] disableAllNodes Whether to disable all nodes in the graph (default: false)
    void update72eEarlyHarqGraph(bool disableAllNodes = false);
    void update72eEarlyHarqOffloadingGraph(bool disableAllNodes = false);
    
    void createFrontLoadedDmrsGraph();  // for front-loaded DMRS portion of PUSCH pipeline
    void updateFrontLoadedDmrsGraph(bool disableAllNodes =false);

    // for the parent nodes of device graph launches
    void createLaunchGraph(CUgraph &graph,
                           CUgraphNode &graphWaitNode, CUgraphNode &graphEventNode, CUgraphNode &graphDglNode,
                           cuphyPuschRxWaitLaunchCfg_t& waitCfg, cuphyPuschRxDglLaunchCfg_t& dglCfg,
                           uint8_t puschRxProcMode, CUevent waitEndEvent, bool enableDeviceGraphLaunch);
    void updateLaunchGraph(CUgraphExec &graphExec, CUgraphNode & graphWaitNode, CUgraphNode &graphDglNode,
                           cuphyPuschRxWaitLaunchCfg_t& waitCfg, cuphyPuschRxDglLaunchCfg_t& dglCfg, bool enableDeviceGraphLaunch);

    CUresult addGraphNodeHelper(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, CUgraphNodeParams* nodeParams);

    // kernel launch for early-HARQ or frontloaded DMRS
    void subSlotKernelLaunch();

    // kernel launch for early-HARQ full slot proc or post-frontloaded DMRS full slot proc
    void fullSlotKernelLaunch();

    // pipeline inputs
    cuphyTensorPrm_t  m_tPrmDataRx;

    // pipeline parameters
    const cuphyPuschStatPrms_t m_cuphyPuschStatPrms;
    cuphyPuschDynPrms_t        m_cuphyPuschDynPrms;
    cuphyPuschCellGrpDynPrm_t  m_cuphyPuschCellGrpDynPrm;
    cuphyLDPCParams            m_ldpcPrms;
    uint32_t                   m_nMaxPrb;
    uint32_t                   m_maxNPrbAlloc;
    uint32_t                   m_maxNRx;
    gsl_lite::span<cuphyPuschRxUeGrpPrms_t> m_drvdUeGrpPrmsCpu;
    gsl_lite::span<cuphyPuschRxUeGrpPrms_t> m_drvdUeGrpPrmsGpu;
    const cuphyChEstSettings   m_chEstSettings;
    cuphyChEqParams            m_chEqPrms;
    uint32_t*                  m_harqBufferSizeInBytes;

    PerTbParams*                                                m_pTbPrmsCpu;
    PerTbParams*                                                m_pTbPrmsGpu;
    cuphy::buffer<cuphyPuschCellStatPrm_t, cuphy::device_alloc> m_puschCellStatPrmBufGpu;
    cuphy::buffer<cuphyPuschCellStatPrm_t, cuphy::pinned_alloc> m_puschCellStatPrmCpu;
    const std::vector<cuphyCellStatPrm_t>                       m_cuphyCellStatPrmVecCpu;

    // CPU to GPU data copies
    dataBundle<N_PUSCH_H2D_DATA_TYPES> m_h2dBuffer;
    // zero-initialize
    std::array<size_t, N_PUSCH_H2D_DATA_TYPES> m_inputBufSizeBytes{};
    std::array<size_t, N_PUSCH_H2D_DATA_TYPES> m_inputBufAlignBytes{};

    // intermediate/output buffers
    static constexpr uint32_t LINEAR_ALLOC_PAD_BYTES = 128;
    cuphy::linear_alloc<LINEAR_ALLOC_PAD_BYTES, cuphy::device_alloc> m_LinearAlloc, m_LinearAllocBluesteinWorkspace; // linear buffer holding intermediate results

    std::vector<cuphyTensorPrm_t>  m_tPrmLLRVec, m_tPrmLLRCdm1Vec;
    std::vector<cuphy::tensor_ref> m_tRefDataRx, m_tRefLLRVec, m_tRefLLRCdm1Vec, m_tRefHEstVec, m_tRefPerPrbNoiseVarVec, m_tRefLwInvVec, m_tRefChEstDbgVec, m_tRefCfoEstVec, m_tRefReeDiagInvVec, m_tRefDataEqVec, m_tRefDataEqDftVec, m_tRefCoefVec, m_tRefEqDbgVec;
    std::vector<cuphy::tensor_ref> m_tRefDmrsLSEstVec, m_tRefDmrsDelayMeanVec, m_tRefDmrsAccumVec;

    cuphy::tensor_ref m_tRefCfoPhaseRot, m_tRefTaPhaseRot, m_tRefTaEst, m_tRefCfoTaEstInterCtaSyncCnt, m_tRefCfoEstInterCtaSyncCnt, m_tRefTaEstInterCtaSyncCnt;
    cuphy::tensor_ref m_tRefRssi, m_tRefRssiEhq, m_tRefRssiFull, m_tRefRssiInterCtaSyncCnt;
    cuphy::tensor_ref m_tRefRsrp, m_tRefRsrpEhq, m_tRefRsrpInterCtaSyncCnt, m_tRefNoiseVarPreEq, m_tRefNoiseVarPostEq, m_tRefNoiseIntfEstInterCtaSyncCnt, m_tRefSinrPreEq, m_tRefSinrPostEq;
    cuphy::tensor_ref m_tRefCfoHz;
    cuphy::tensor_ref m_tRefBluesteinWorkspaceTime, m_tRefBluesteinWorkspaceFreq;
    void**            m_pHarqBuffers;
    float**           m_pFoCompensationBuffers;
    void*             d_pLDPCOut;
#ifdef ENABLE_PUSCH_LDPC_OUTPUT_H5_DUMP
    void*             d_pLDPCSoftOut;
#endif
    OutputParams      m_outputPrms; // CPU and GPU adreasses of output buffers.

    cuphyTensorInfo2_t tInfoDftBluesteinWorkspaceTime;
    cuphyTensorInfo2_t tInfoDftBluesteinWorkspaceFreq;

    // Component handles (non-LDPC)
    cuphyPuschRxNoiseIntfEstHndl_t m_noiseIntfEstHndl;
    cuphyPuschRxCfoTaEstHndl_t     m_cfoTaEstHndl;
    cuphyPuschRxChEqHndl_t         m_chEqHndl;
    cuphyPuschRxRateMatchHndl_t    m_rateMatchHndl;
    cuphyPuschRxCrcDecodeHndl_t    m_crcDecodeHndl;
    cuphyPuschRxRssiHndl_t         m_rssiHndl;
    cuphyRmDecoderHndl_t           m_rmDecodeHndl;
    cuphySimplexDecoderHndl_t      m_spxDecoderHndl;
    cuphyRmDecoderHndl_t           m_rmDecodeHndl_csi2;
    cuphySimplexDecoderHndl_t      m_spxDecoderHndl_csi2;
    cuphyRmDecoderHndl_t           m_rmDecodeHndl_early;
    cuphySimplexDecoderHndl_t      m_spxDecoderHndl_early;
    cuphyUciOnPuschSegLLRs0Hndl_t  m_uciOnPuschSegLLRs0Hndl;
    cuphyUciOnPuschSegLLRs0Hndl_t  m_uciOnPuschEarlySegLLRs0Hndl;
    cuphyUciOnPuschSegLLRs2Hndl_t  m_uciOnPuschSegLLRs2Hndl;
    cuphyUciOnPuschCsi2CtrlHndl_t  m_uciOnPuschCsi2CtrlHndl;
    cuphyCompCwTreeTypesHndl_t     m_compCwTreeTypesHndl;
    cuphyPolSegDeRmDeItlHndl_t     m_polSegDeRmDeItlHndl;
    cuphyPolarDecoderHndl_t        m_polarDecoderHndl;
    cuphyCompCwTreeTypesHndl_t     m_compCwTreeTypesHndl_csi2;
    cuphyPolSegDeRmDeItlHndl_t     m_polSegDeRmDeItlHndl_csi2;
    cuphyPolarDecoderHndl_t        m_polarDecoderHndl_csi2;
    cuphyCompCwTreeTypesHndl_t     m_compCwTreeTypesHndl_early;
    cuphyPolSegDeRmDeItlHndl_t     m_polSegDeRmDeItlHndl_early;
    cuphyPolarDecoderHndl_t        m_polarDecoderHndl_early;

    uint32_t m_nUes;

    // SCH-on-PUSCH parameters
    uint16_t m_nSchUes;
    std::vector<uint16_t> m_schUserIdxsVec;
    std::vector<uint16_t> m_earlyDecodedCbCountPerTb;
    std::vector<cuphyPuschRxRateMatchCbRange_t> m_rateMatchCbRangesEarly;
    std::vector<cuphyPuschRxRateMatchCbRange_t> m_rateMatchCbRangesFull;
    uint32_t              m_nEarlyDecodedSchCbs{};
    uint32_t              m_nFullSlotRemainingSchCbs{};

    // UCI-on-PUSCH LLR seg parameters.
    uint8_t               m_enableCsiP2Fapiv3;
    std::vector<uint16_t> m_nPrbsVec;
    uint16_t              m_nUciUes, m_nEarlyHarqUes, m_nCsi2Ues;
    std::vector<uint16_t> m_uciUserIdxsVec, m_csi2UeIdxsVec;
    //cudaDataType_t        m_llrDataType = CUDA_R_16F; // if to be uncommented, move initialization to constructor

    // Simplex decoder parameters (HARQ + CSI1)
    uint16_t             m_nSpxCws;
    cuphySimplexCwPrm_t* m_pSpxCwPrmsCpu;
    cuphySimplexCwPrm_t* m_pSpxPrmsGpu;

    // simplex decoder parameters (CSI2)
    uint16_t             m_nSpxCws_csi2;
    cuphySimplexCwPrm_t* m_pSpxCwPrmsCpu_csi2;
    cuphySimplexCwPrm_t* m_pSpxPrmsGpu_csi2;

    // simplex decoder parameters (early)
    uint16_t             m_nSpxCws_early;
    cuphySimplexCwPrm_t* m_pSpxCwPrmsCpu_early;
    cuphySimplexCwPrm_t* m_pSpxPrmsGpu_early;

    // Reed Muller decoder parameters (HARQ + CSI1)
    cuphyRmCwPrm_t* m_pRmCwPrmsCpu;
    cuphyRmCwPrm_t* m_pRmCwPrmsGpu;
    uint16_t        m_nRmCws;

    // Reed Muller decoder parameters (CSI2)
    cuphyRmCwPrm_t* m_pRmCwPrmsCpu_csi2;
    cuphyRmCwPrm_t* m_pRmCwPrmsGpu_csi2;
    uint16_t        m_nRmCws_csi2;

    // Reed Muller decoder parameters (early)
    cuphyRmCwPrm_t* m_pRmCwPrmsCpu_early;
    cuphyRmCwPrm_t* m_pRmCwPrmsGpu_early;
    uint16_t        m_nRmCws_early;

    // List size for Polar Decoder
    uint8_t m_polDcdrListSz;

    // Polar decoder parameters (HARQ + CSI1)
    uint16_t               m_nPolUciSegs;
    uint16_t               m_nPolCbs;
    cuphyPolarUciSegPrm_t* m_pUciSegPrmsCpu;
    cuphyPolarUciSegPrm_t* m_pUciSegPrmsGpu;
    cuphyPolarCwPrm_t*     m_pUciCwPrmsCpu;
    cuphyPolarCwPrm_t*     m_pUciCwPrmsGpu;
    std::vector<uint8_t*>  m_cwTreeTypesAddrVec;
    std::vector<__half*>   m_uciSegLLRsAddrVec;
    std::vector<__half*>   m_cwLLRsAddrVec;
    std::vector<__half*>   m_cwTreeLLRsAddrVec;
    std::vector<uint32_t*> m_cbEstAddrVec;
    std::vector<bool*>     m_listPolScratchAddrVec;
    uint8_t*               m_pPolCrcFlags;
    std::vector<uint32_t*> m_pUciSegEst;

    // Polar decoder parameters (CSI2)
    uint16_t               m_nPolUciSegs_csi2;
    uint16_t               m_nPolCbs_csi2;
    cuphyPolarUciSegPrm_t* m_pUciSegPrmsCpu_csi2;
    cuphyPolarUciSegPrm_t* m_pUciSegPrmsGpu_csi2;
    cuphyPolarCwPrm_t*     m_pUciCwPrmsCpu_csi2;
    cuphyPolarCwPrm_t*     m_pUciCwPrmsGpu_csi2;
    std::vector<uint8_t*>  m_cwTreeTypesAddrVec_csi2;
    std::vector<__half*>   m_uciSegLLRsAddrVec_csi2;
    std::vector<__half*>   m_cwLLRsAddrVec_csi2;
    std::vector<__half*>   m_cwTreeLLRsAddrVec_csi2;
    std::vector<uint32_t*> m_cbEstAddrVec_csi2;
    std::vector<bool*>     m_listPolScratchAddrVec_csi2;
    uint8_t*               m_pPolCrcFlags_csi2;
    std::vector<uint32_t*> m_pUciSegEst_csi2;

    // Polar decoder parameters (early)
    uint16_t               m_nPolUciSegs_early;
    uint16_t               m_nPolCbs_early;
    cuphyPolarUciSegPrm_t* m_pUciSegPrmsCpu_early;
    cuphyPolarUciSegPrm_t* m_pUciSegPrmsGpu_early;
    cuphyPolarCwPrm_t*     m_pUciCwPrmsCpu_early;
    cuphyPolarCwPrm_t*     m_pUciCwPrmsGpu_early;
    std::vector<uint8_t*>  m_cwTreeTypesAddrVec_early;
    std::vector<__half*>   m_uciSegLLRsAddrVec_early;
    std::vector<__half*>   m_cwLLRsAddrVec_early;
    std::vector<__half*>   m_cwTreeLLRsAddrVec_early;
    std::vector<uint32_t*> m_cbEstAddrVec_early;
    std::vector<bool*>     m_listPolScratchAddrVec_early;
    uint8_t*               m_pPolCrcFlags_early;
    std::vector<uint32_t*> m_pUciSegEst_early;

    // LDPC decoder
    size_t                                   m_ldpcWorkspaceSize;
    cuphy::context                           m_ctx;
    bool                                     m_useCbLdpc;
    std::variant<TbLdpcState, CbLdpcState>   m_ldpcState;
    cuphyPuschLdpcKernelLaunch_t             m_LDPCkernelLaunchMode;

    // kernel launch configurationsen
    cuphyPuschRxWaitLaunchCfg_t          m_preSubSlotWaitCfgs;
    cuphyPuschRxWaitLaunchCfg_t          m_postSubSlotWaitCfgs;
    cuphyPuschRxDglLaunchCfg_t           m_preSubSlotDglCfgs;
    cuphyPuschRxDglLaunchCfg_t           m_postSubSlotDglCfgs;
    cuphyPuschRxNoiseIntfEstLaunchCfgs_t m_noiseIntfEstLaunchCfgs[CUPHY_MAX_PUSCH_EXECUTION_PATHS];
    cuphyPuschRxChEqLaunchCfgs_t         m_chEqCoefCompLaunchCfgs[CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST];
    cuphyPuschRxChEqLaunchCfgs_t         m_chEqSoftDemapLaunchCfgs[CUPHY_MAX_PUSCH_EXECUTION_PATHS];
    cuphyPuschRxChEqLaunchCfgs_t         m_chEqSoftDemapIdftLaunchCfgs[CUPHY_MAX_PUSCH_EXECUTION_PATHS];
    cuphyPuschRxChEqLaunchCfgs_t         m_chEqSoftDemapAfterDftLaunchCfgs[CUPHY_MAX_PUSCH_EXECUTION_PATHS];
    cuphyPuschRxCfoTaEstLaunchCfgs_t     m_cfoTaEstLaunchCfgs;
    cuphyPuschRxRateMatchLaunchCfg_t     m_rateMatchLaunchCfg;
    cuphyPuschRxRateMatchLaunchCfg_t     m_rateMatchLaunchCfgEarly;
    cuphyPuschRxCrcDecodeLaunchCfg_t     m_crcLaunchCfgs[2]; // CB CRC + TB CRC
    cuphyPuschRxRssiLaunchCfgs_t         m_rssiLaunchCfgs[CUPHY_MAX_PUSCH_EXECUTION_PATHS];
    cuphyPuschRxRsrpLaunchCfgs_t         m_rsrpLaunchCfgs[CUPHY_MAX_PUSCH_EXECUTION_PATHS];
    cuphyUciOnPuschSegLLRs0LaunchCfg_t   m_uciOnPuschSegLLRs0LaunchCfg;
    cuphyUciOnPuschSegLLRs0LaunchCfg_t   m_uciOnPuschEarlySegLLRs0LaunchCfg;
    cuphyUciOnPuschSegLLRs2LaunchCfg_t   m_uciOnPuschSegLLRs2LaunchCfg;
    cuphyUciOnPuschCsi2CtrlLaunchCfg_t   m_uciOnPuschCsi2CtrlLaunchCfg;
    cuphyCompCwTreeTypesLaunchCfg_t      m_compCwTreeTypesLaunchCfg;
    cuphyPolSegDeRmDeItlLaunchCfg_t      m_polSegDeRmDeItlLaunchCfg;
    cuphyPolarDecoderLaunchCfg_t         m_polarDecoderLaunchCfg;
    cuphySimplexDecoderLaunchCfg_t       m_simplexDecoderLaunchCfg;
    cuphyRmDecoderLaunchCfg_t            m_rmDecoderLaunchCfg;
    cuphySimplexDecoderLaunchCfg_t       m_simplexDecoderLaunchCfg_csi2;
    cuphySimplexDecoderLaunchCfg_t       m_simplexDecoderLaunchCfg_early;
    cuphyRmDecoderLaunchCfg_t            m_rmDecoderLaunchCfg_csi2;
    cuphyCompCwTreeTypesLaunchCfg_t      m_compCwTreeTypesLaunchCfg_csi2;
    cuphyPolSegDeRmDeItlLaunchCfg_t      m_polSegDeRmDeItlLaunchCfg_csi2;
    cuphyPolarDecoderLaunchCfg_t         m_polarDecoderLaunchCfg_csi2;
    cuphyRmDecoderLaunchCfg_t            m_rmDecoderLaunchCfg_early;
    cuphyCompCwTreeTypesLaunchCfg_t      m_compCwTreeTypesLaunchCfg_early;
    cuphyPolSegDeRmDeItlLaunchCfg_t      m_polSegDeRmDeItlLaunchCfg_early;
    cuphyPolarDecoderLaunchCfg_t         m_polarDecoderLaunchCfg_early;

    // kernel descriptors
    cuphy::kernelDescrs<N_PUSCH_DESCR_TYPES>         m_kernelStatDescr;
    cuphy::kernelDescrs<N_PUSCH_DESCR_TYPES>         m_kernelDynDescr;

    //the enabling flag for the early-HARQ mode
    bool m_earlyHarqModeEnabled;
    //the enabling flag for the sub-slot processing of front-loaded DMRS
    bool m_subSlotProcessingFrontLoadedDmrsEnabled;

    //the flag to enable device graph launch mode
    bool m_deviceGraphLaunchEnabled;
    
    uint8_t m_maxDmrsMaxLen;

    // graph parameters
    //======================================================================================================
    bool            m_cudaGraphModeEnabled;
    // for full-slot
    CUgraph         m_fullSlotGraph;
    CUgraphExec     m_fullSlotGraphExec;
    CUgraph         m_frontLoadedDmrsFullSlotGraph;
    CUgraphExec     m_frontLoadedDmrsFullSlotGraphExec;
    // for early-HARQ
    CUgraph         m_ehqGraph;
    CUgraphExec     m_ehqGraphExec;
    // for front-loaded DMRS
    CUgraph         m_frontLoadedDmrsGraph;
    CUgraphExec     m_frontLoadedDmrsGraphExec;
    // for symbol wait and device graph launch
    CUgraph         m_preFullSlotGraph;
    CUgraphExec     m_preFullSlotGraphExec;
    CUgraph         m_preSubSlotGraph;
    CUgraphExec     m_preSubSlotGraphExec;

    // graph kernel nodes
    CUgraphNode     m_emptyFullSlotRootNode, m_emptyFrontLoadedDmrsFullSlotRootNode;
    CUgraphNode     m_noiseIntfEstNodes[CUPHY_PUSCH_RX_NOISE_INTF_EST_N_MAX_HET_CFGS];
    CUgraphNode     m_cfoTaEstNodes[CUPHY_PUSCH_RX_CFO_EST_N_MAX_HET_CFGS];
    CUgraphNode     m_chEqCoefCompNodes[CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST][CUPHY_PUSCH_RX_CH_EQ_N_MAX_HET_CFGS];
    // full-slot proc + front-loaded DMRS full-slot proc
    CUgraphNode     m_chEqSoftDemapNodes[cuphyPuschFullSlotProcMode_t::CUPHY_MAX_PUSCH_FULL_SLOT_PROC_MODES][CUPHY_PUSCH_RX_CH_EQ_N_MAX_HET_CFGS];
    CUgraphNode     m_chEqSoftDemapIdftNodes[cuphyPuschFullSlotProcMode_t::CUPHY_MAX_PUSCH_FULL_SLOT_PROC_MODES][CUPHY_PUSCH_RX_CH_EQ_N_MAX_HET_CFGS];
    CUgraphNode     m_chEqSoftDemapAfterDftNodes[cuphyPuschFullSlotProcMode_t::CUPHY_MAX_PUSCH_FULL_SLOT_PROC_MODES][CUPHY_PUSCH_RX_CH_EQ_N_MAX_HET_CFGS];
    CUgraphNode     m_resetRateMatchNode[cuphyPuschFullSlotProcMode_t::CUPHY_MAX_PUSCH_FULL_SLOT_PROC_MODES];
    CUgraphNode     m_clampRateMatchNode[cuphyPuschFullSlotProcMode_t::CUPHY_MAX_PUSCH_FULL_SLOT_PROC_MODES];
    CUgraphNode     m_rateMatchNode[cuphyPuschFullSlotProcMode_t::CUPHY_MAX_PUSCH_FULL_SLOT_PROC_MODES];
    CUgraphNode     m_subSlotDelayNode, m_fullSlotDelayNode;
    CUgraphNode     m_crcNodes[cuphyPuschFullSlotProcMode_t::CUPHY_MAX_PUSCH_FULL_SLOT_PROC_MODES][2];  // CB CRC + TB CRC
    CUgraphNode     m_uciSegLLRs0Node[cuphyPuschFullSlotProcMode_t::CUPHY_MAX_PUSCH_FULL_SLOT_PROC_MODES];
    CUgraphNode     m_simplexDecoderNode[cuphyPuschFullSlotProcMode_t::CUPHY_MAX_PUSCH_FULL_SLOT_PROC_MODES];
    CUgraphNode     m_rmDecoderNode[cuphyPuschFullSlotProcMode_t::CUPHY_MAX_PUSCH_FULL_SLOT_PROC_MODES];
    CUgraphNode     m_compCwTreeTypesNode[cuphyPuschFullSlotProcMode_t::CUPHY_MAX_PUSCH_FULL_SLOT_PROC_MODES];
    CUgraphNode     m_polSegDeRmDeItlNode[cuphyPuschFullSlotProcMode_t::CUPHY_MAX_PUSCH_FULL_SLOT_PROC_MODES];
    CUgraphNode     m_polarDecoderNode[cuphyPuschFullSlotProcMode_t::CUPHY_MAX_PUSCH_FULL_SLOT_PROC_MODES];
    // CSI-P2
    CUgraphNode     m_uciOnPuschCsi2CtrlNode[cuphyPuschFullSlotProcMode_t::CUPHY_MAX_PUSCH_FULL_SLOT_PROC_MODES];
    CUgraphNode     m_uciOnPuschCsi2SegLLRs2Node[cuphyPuschFullSlotProcMode_t::CUPHY_MAX_PUSCH_FULL_SLOT_PROC_MODES];
    CUgraphNode     m_uciOnPuschCsi2rmDecoderNode[cuphyPuschFullSlotProcMode_t::CUPHY_MAX_PUSCH_FULL_SLOT_PROC_MODES];
    CUgraphNode     m_uciOnPuschCsi2simplexDecoderNode[cuphyPuschFullSlotProcMode_t::CUPHY_MAX_PUSCH_FULL_SLOT_PROC_MODES];
    CUgraphNode     m_uciOnPuschCsi2CompCwTreeTypesNode[cuphyPuschFullSlotProcMode_t::CUPHY_MAX_PUSCH_FULL_SLOT_PROC_MODES];
    CUgraphNode     m_uciOnPuschCsi2PolSegDeRmDeItlNode[cuphyPuschFullSlotProcMode_t::CUPHY_MAX_PUSCH_FULL_SLOT_PROC_MODES];
    CUgraphNode     m_uciOnPuschCsi2PolarDecoderNode[cuphyPuschFullSlotProcMode_t::CUPHY_MAX_PUSCH_FULL_SLOT_PROC_MODES];
    CUgraphNode     m_uciOnPuschCompletedEventNode[cuphyPuschFullSlotProcMode_t::CUPHY_MAX_PUSCH_FULL_SLOT_PROC_MODES];
    //
    CUgraphNode     m_rssiNodes[cuphyPuschFullSlotProcMode_t::CUPHY_MAX_PUSCH_FULL_SLOT_PROC_MODES][CUPHY_PUSCH_RX_RSSI_N_MAX_HET_CFGS];
    CUgraphNode     m_rsrpNodes[cuphyPuschFullSlotProcMode_t::CUPHY_MAX_PUSCH_FULL_SLOT_PROC_MODES][CUPHY_PUSCH_RX_RSRP_N_MAX_HET_CFGS];

    // node states used in updateFullSlotGraph()
    std::vector<uint8_t>                 m_noiseIntfEstNodesEnabled;
    std::vector<uint8_t>                 m_cfoTaEstNodesEnabled;
    std::vector<std::vector<uint8_t>>    m_chEqCoefCompNodesEnabled;
    // full-slot proc + fornt-loaded DMRS full-slot proc
    std::vector<uint8_t>                 m_chEqSoftDemapNodesEnabled[cuphyPuschFullSlotProcMode_t::CUPHY_MAX_PUSCH_FULL_SLOT_PROC_MODES];
    std::vector<uint8_t>                 m_chEqSoftDemapIdftNodesEnabled[cuphyPuschFullSlotProcMode_t::CUPHY_MAX_PUSCH_FULL_SLOT_PROC_MODES];
    std::vector<uint8_t>                 m_chEqSoftDemapAfterDftNodesEnabled[cuphyPuschFullSlotProcMode_t::CUPHY_MAX_PUSCH_FULL_SLOT_PROC_MODES];
    uint8_t                              m_rateMatchNodeEnabled[cuphyPuschFullSlotProcMode_t::CUPHY_MAX_PUSCH_FULL_SLOT_PROC_MODES];
    std::vector<uint8_t>                 m_cbLdpcNodesEnabled[cuphyPuschFullSlotProcMode_t::CUPHY_MAX_PUSCH_FULL_SLOT_PROC_MODES];
    std::vector<uint8_t>                 m_ehqCbLdpcNodesEnabled;
    uint8_t                              m_crcNodesEnabled[cuphyPuschFullSlotProcMode_t::CUPHY_MAX_PUSCH_FULL_SLOT_PROC_MODES];
    uint8_t                              m_uciSegLLRs0NodeEnabled[cuphyPuschFullSlotProcMode_t::CUPHY_MAX_PUSCH_FULL_SLOT_PROC_MODES];
    uint8_t                              m_simplexDecoderNodeEnabled[cuphyPuschFullSlotProcMode_t::CUPHY_MAX_PUSCH_FULL_SLOT_PROC_MODES];
    uint8_t                              m_rmDecoderNodeEnabled[cuphyPuschFullSlotProcMode_t::CUPHY_MAX_PUSCH_FULL_SLOT_PROC_MODES];
    uint8_t                              m_polarNodeEnabled[cuphyPuschFullSlotProcMode_t::CUPHY_MAX_PUSCH_FULL_SLOT_PROC_MODES];//used for m_compCwTreeTypesNode, m_polSegDeRmDeItlNode and m_polarDecoderNode
    uint8_t                              m_csi2NodeEnabled[cuphyPuschFullSlotProcMode_t::CUPHY_MAX_PUSCH_FULL_SLOT_PROC_MODES]; // used for all csi2 nodes
    std::vector<uint8_t>                 m_rssiNodesEnabled[cuphyPuschFullSlotProcMode_t::CUPHY_MAX_PUSCH_FULL_SLOT_PROC_MODES];
    std::vector<uint8_t>                 m_rsrpNodesEnabled[cuphyPuschFullSlotProcMode_t::CUPHY_MAX_PUSCH_FULL_SLOT_PROC_MODES];
    uint8_t                              m_fullSlotDelayNodeEnabled[cuphyPuschFullSlotProcMode_t::CUPHY_MAX_PUSCH_FULL_SLOT_PROC_MODES];
    uint8_t                              m_subSlotDelayNodeEnabled;

    // nodes used in launch graphs
    CUgraphNode     m_preSubSlotWaitNode;
    CUgraphNode     m_preSubSlotEventNode;
    CUgraphNode     m_preSubSlotDglNode;
    CUgraphNode     m_postSubSlotWaitNode;
    CUgraphNode     m_postSubSlotEventNode;
    CUgraphNode     m_postSubSlotDglNode;

    // for early HARQ sub-slot processing using CUDA graphs
    //======================================================================================================
    // early-HARQ related nodes
    CUgraphNode     m_ehqRootNode;
    CUgraphNode     m_ehqNoiseIntfEstNodes[CUPHY_PUSCH_RX_NOISE_INTF_EST_N_MAX_HET_CFGS];
    CUgraphNode     m_ehqChEqCoefCompNodes[CUPHY_PUSCH_RX_CH_EQ_N_MAX_HET_CFGS];
    CUgraphNode     m_ehqChEqSoftDemapNodes[CUPHY_PUSCH_RX_CH_EQ_N_MAX_HET_CFGS];
    CUgraphNode     m_ehqChEqSoftDemapIdftNodes[CUPHY_PUSCH_RX_CH_EQ_N_MAX_HET_CFGS];
    CUgraphNode     m_ehqChEqSoftDemapAfterDftNodes[CUPHY_PUSCH_RX_CH_EQ_N_MAX_HET_CFGS];
    CUgraphNode     m_ehqUciSegLLRs0Node;
    CUgraphNode     m_ehqResetRateMatchNode;
    CUgraphNode     m_ehqRateMatchNode;
    CUgraphNode     m_ehqClampRateMatchNode;
    CUgraphNode     m_ehqSimplexDecoderNode;
    CUgraphNode     m_ehqRmDecoderNode;
    CUgraphNode     m_ehqCompCwTreeTypesNode;
    CUgraphNode     m_ehqPolSegDeRmDeItlNode;
    CUgraphNode     m_ehqPolarDecoderNode;
    CUgraphNode     m_ehqRsrpNodes[CUPHY_PUSCH_RX_RSRP_N_MAX_HET_CFGS];
    CUgraphNode     m_ehqRssiNodes[CUPHY_PUSCH_RX_RSRP_N_MAX_HET_CFGS];

    // front-loaded DMRS related nodes
    CUgraphNode     m_frontLoadedDmrsRootNode;
    CUgraphNode     m_frontLoadedDmrsNoiseIntfEstNodes[CUPHY_PUSCH_RX_NOISE_INTF_EST_N_MAX_HET_CFGS];
    CUgraphNode     m_frontLoadedDmrsChEqCoefCompNodes[CUPHY_PUSCH_RX_CH_EQ_N_MAX_HET_CFGS];

    // node states used in updateEarlyHarqGraph()
    std::vector<uint8_t> m_ehqNoiseIntfEstNodesEnabled;
    std::vector<uint8_t> m_ehqChEqCoefCompNodesEnabled;
    std::vector<uint8_t> m_ehqChEqSoftDemapNodesEnabled;
    std::vector<uint8_t> m_ehqChEqSoftDemapIdftNodesEnabled;
    std::vector<uint8_t> m_ehqChEqSoftDemapAfterDftNodesEnabled;
    uint8_t              m_ehqUciSegLLRs0NodeEnabled;
    uint8_t              m_ehqRateMatchNodeEnabled;
    uint8_t              m_ehqCbDecoderNodeEnabled;
    uint8_t              m_ehqSimplexDecoderNodeEnabled;
    uint8_t              m_ehqRmDecoderNodeEnabled;
    uint8_t              m_ehqPolarNodeEnabled;     // used for m_ehqCompCwTreeTypesNode, m_ehqPolSegDeRmDeItlNode and m_ehqPolarDecoderNode
    std::vector<uint8_t> m_ehqRsrpNodesEnabled;
    std::vector<uint8_t> m_ehqRssiNodesEnabled;

    // node states used in updateFrontLoadedDmrsGraph()
    std::vector<uint8_t> m_frontLoadedDmrsNoiseIntfEstNodesEnabled;
    std::vector<uint8_t> m_frontLoadedDmrsChEqCoefCompNodesEnabled;

    // node used in launch graph
    uint8_t m_DeviceGraphLaunchNodeEnabled;

    // node parameters
    CUDA_KERNEL_NODE_PARAMS m_emptyNode0paramDriver;
    CUDA_KERNEL_NODE_PARAMS m_emptyNode1paramDriver;
    CUDA_KERNEL_NODE_PARAMS m_emptyNode2paramsDriver;

    // Used for full slot graph version when conditional IF nodes or device graphs are used
    // Please note that the extra kernels launching a device graph or setting a condition
    // will only be present when PUSCH runs in graphs mode. These kernels are not present in streams mode.
    condGraphInfo m_fullSlotGraphCondInfo;
    condGraphInfo m_frontLoadedDmrsFullSlotGraphCondInfo;

    uint32_t m_maxNTbs;
    uint32_t m_maxNCbs;
    uint32_t m_maxNCbsPerTb;

    // CUDA stream used for GPU operations
    cudaStream_t phase1Stream{};
    cuphy::event m_phase1Complete;
    cudaStream_t phase2Stream{};
    cuphy::stream_pool m_G0streamPool; // Use a pool of CUDA streams to concurrently perform early-HARQ output copy and PUSCH_RUN_FULL_SLOT_PROC
    cuphy::stream_pool m_G1streamPool; // Use a pool of CUDA streams to concurrently launch some kernels in CSI-P1 and CSI-P2
    cuphy::stream_pool m_G2streamPool; // Use a pool of CUDA streams to concurrently launch some other kernels in CSI-P1 and CSI-P2

    // event for stream syncs
    cuphy::event m_uciOnPuschSegLLRs0Event[CUPHY_MAX_PUSCH_EXECUTION_PATHS];
    cuphy::event m_compCwTreeTypesEvent[CUPHY_MAX_PUSCH_EXECUTION_PATHS];
    cuphy::event m_rateMatchEvent;

    // GPU Architectures
    const cuphyDeviceArchInfo m_cudaDeviceArchInfo;

    // For work cancellation
    uint8_t* m_workCancelPtr; // from static parameters
    cuphyPuschWorkCancelMode_t  m_workCancelMode; // from static parameters

    // puschrx channel estimate abstractions
    std::unique_ptr<ch_est::IKernelBuilder> m_chestKernelBuilder;
    std::unique_ptr<ch_est::IModule> m_chest;

    enum batchedMemcpyHelperTarget{
        EH = 0, /* early harq */
        NON_EH = 1, /* not early harq */
        H2D_SETUP = 2,
        TOTAL_CNT = 3
    };

    // Batched memcpy helper objects
    const bool m_useBatchedMemcpy;
    //  Array of 3, index 0 for early harq, index  1 for non EH, and 2 for H2D setup; see batchedmMemcpyHelperTarget
    cuphyBatchedMemcpyHelper m_batchedMemcpyHelper[batchedMemcpyHelperTarget::TOTAL_CNT];

    // Determine from PuschRx ctor passed stream if green context is used
    bool m_useGreenContext{};
    
    // Open RAN Functional Splits Option for PUSCH
    uint8_t m_openRanFunctionalSplitOption{};
    uint8_t m_kernelSelOption{};
    uint8_t m_uciKernelSelOption{};
    uint32_t m_delayUs{};
    uint32_t m_subSlotDelayUs{};
    
    void*    m_subSlotDelayKernelArgs[1];
    CUDA_KERNEL_NODE_PARAMS m_subSlotDelayKernelParamsDriver;
    
    void*    m_fullSlotDelayKernelArgs[1];
    CUDA_KERNEL_NODE_PARAMS m_fullSlotDelayKernelParamsDriver;

};

#endif // !defined(PUSCH_RX_HPP_INCLUDED_)
