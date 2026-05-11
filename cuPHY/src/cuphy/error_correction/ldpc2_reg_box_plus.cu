/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

//#define CUPHY_DEBUG 1

#include <assert.h>
#include "ldpc2_desc.cuh"
#include "ldpc2_c2v.cuh"
#include "ldpc2_app_address_fp_desc.cuh"
#include "ldpc2_app_address_fp_dp_desc.cuh"
#include "ldpc2_min_sum_update_half_0.cuh"
#include "ldpc2_box_plus.cuh"
#include "ldpc2_reg_box_plus.hpp"
#include "ldpc2_schedule_dynamic_desc.cuh"
#include "ldpc2_c2v_cache_register.cuh"

using namespace ldpc2;

#define USE_APP_ADDR_FP_DP 1

#if defined(__CUDACC__) && (__CUDACC_VER_MAJOR__ >= 11)
#define LDPC2_GRID_CONST __grid_constant__ const
#else
#define LDPC2_GRID_CONST const
#endif

namespace
{
    // Single set of values for all kernels in this module, for now...
    const int MAX_THREADS_PER_CTA = 384;
    const int MIN_CTA_PER_SM      = 1;

    //------------------------------------------------------------------
    // Maximum number of parity nodes supported by this kernel
    const int MAX_NUM_PARITY_BG1      = 46;
    const int MAX_NUM_PARITY_BG2      = 42;

    //------------------------------------------------------------------
    // Number of per-row storage words
    [[maybe_unused]] const int NUM_STORAGE_WORDS_BG1 = 10;
    [[maybe_unused]] const int NUM_STORAGE_WORDS_BG2 = 5;
    [[maybe_unused]] const int NUM_STORAGE_WORDS_BG1_FAST_CORE    = NUM_STORAGE_WORDS_BG1;
    [[maybe_unused]] const int NUM_STORAGE_WORDS_BG1_FAST_NONCORE = 8;

    // APP address calculation
    // Using floating point instruction APP address calculation
    // sequence
#if USE_APP_ADDR_FP_DP
    template <int BG> using app_loc_t = app_loc_address_fp_dp_desc<__half, BG>;
#else
    template <int BG> using app_loc_t = app_loc_address_fp_desc<__half, BG>;
#endif

    //------------------------------------------------------------------
    // Alias template for compressed C2V row processors, with a template
    // parameter for the C2V row storage. The sign processor and min
    // sum updater have been chosen to be the "fastest" for some
    // architecture and lifting size combinations.
    //typedef ldpc2::cC2V_row_proc<__half,
    //                             ldpc2::cC2V_row_context<__half,
    //                                                     sign_dst_fp_t,
    //                                                     ldpc2::min_sum_update_half_0,
    //                                                     C2V_storage_t>
    //                            > cC2V_row_proc;
    //------------------------------------------------------------------
    // box_plus_all_row_map_t
    // The C2V_row_proc template requires a row map template with template
    // arguments BG (int), CHECK_IDX (int), TStorage (per-row storage
    // structure. We use simple_row_map to indicate that all rows should
    // use the same template. (Other kernels might choose differently for
    // different rows.)
    template <int   BG,
              int   CHECK_IDX,
              class TC2VStorage> using box_plus_all_row_map_t = simple_row_map<BG,
                                                                               CHECK_IDX,
                                                                               TC2VStorage,
                                                                               box_plus_row_proc<box_plus_op>>;

    //------------------------------------------------------------------
    // Kernel configuration structure with split C2V storage for core
    // and non-core parity rows.
    template <int   BG_,
              int   NUM_STORAGE_WORDS_CORE,
              int   NUM_STORAGE_WORDS_NONCORE,
              int   MAX_PARITY_ROWS,
              class TKernelParams>
    struct ldpc2_reg_box_plus_kernel_config_core_split
    {
        static constexpr int BG              = BG_;
        static constexpr int MIN_PARITY_ROWS = 4;

        typedef ldpc2::C2V_storage_t<__half, NUM_STORAGE_WORDS_CORE>    c2v_storage_core_t;
        typedef ldpc2::C2V_storage_t<__half, NUM_STORAGE_WORDS_NONCORE> c2v_storage_noncore_t;
        typedef TKernelParams                                            kernel_params_t;

        typedef C2V_row_proc<__half,
                             BG,
                             box_plus_all_row_map_t,
                             app_loader,
                             app_writer> c2v_t;

        typedef ldpc2::c2v_cache_register_core<BG,
                                               MAX_PARITY_ROWS,
                                               c2v_t,
                                               c2v_storage_core_t,
                                               c2v_storage_noncore_t,
                                               kernel_params_t> c2v_cache_t;

        typedef ldpc2::llr_loader_variable_batch<__half, 4, llr_op_clamp> llr_loader_t;
        typedef typename llr_loader_t::app_buf_t                          app_buf_t;

        typedef ldpc2::ldpc_schedule_dynamic_desc<BG,
                                                  app_loc_t<BG_>,
                                                  c2v_cache_t,
                                                  kernel_params_t,
                                                  typename app_loc_t<BG_>::bg_desc_t,
                                                  MIN_PARITY_ROWS,
                                                  MAX_PARITY_ROWS> sched_t;
    };
    //------------------------------------------------------------------
    // Kernel configuration structure, with typedefs for kernel execution
    // BG_: base graph (1 or 2)
    // TKernelParams: Class/struct used for kernel parameters
    // NUM_STORAGE_WORDS: Number of storage words for each parity row
    // MAX_PARITY_ROWS: Maximum number of parity rows supported by the kernel
    template <int   BG_,
              int   NUM_STORAGE_WORDS,
              int   MAX_PARITY_ROWS,
              class TKernelParams>
    struct ldpc2_reg_box_plus_kernel_config
    {
        static constexpr int BG              = BG_;
        static constexpr int MIN_PARITY_ROWS = 4;

        // C2V per-row storage. Larger storage allows faster row
        // processing, but increases register pressure (and may incur
        // register spills).
        typedef ldpc2::C2V_storage_t<__half, NUM_STORAGE_WORDS> c2v_storage_t;
        typedef TKernelParams                                   kernel_params_t;
        
        typedef C2V_row_proc<__half,
                             BG,
                             box_plus_all_row_map_t,
                             app_loader,
                             app_writer>                                  C2V_t;
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // C2V message cache (register memory here)
        typedef ldpc2::c2v_cache_register<BG,
                                          MAX_PARITY_ROWS,
                                          C2V_t,
                                          c2v_storage_t,
                                          kernel_params_t>                c2v_cache_t;
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // LLR loader, used to load LLR data from global to shared memory
        typedef ldpc2::llr_loader_variable_batch<__half, 4, llr_op_clamp> llr_loader_t;
        // Data type in APP shared memory buffer (__half or __half2)
        typedef typename llr_loader_t::app_buf_t                          app_buf_t;
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // "Dynamic" schedule, with the number of parity rows not known until runtime.
        typedef ldpc2::ldpc_schedule_dynamic_desc<BG,
                                                  app_loc_t<BG_>,
                                                  c2v_cache_t,
                                                  kernel_params_t,
                                                  typename app_loc_t<BG_>::bg_desc_t,
                                                  MIN_PARITY_ROWS,
                                                  MAX_PARITY_ROWS> sched_t;
    };
} // namespace


////////////////////////////////////////////////////////////////////////
// ldpc2_BG1_reg_box_plus()
// Base graph 1 kernel, "legacy" tensor interface
extern "C"
__global__ __launch_bounds__(MAX_THREADS_PER_CTA, MIN_CTA_PER_SM)
void ldpc2_BG1_reg_box_plus(LDPC_kernel_params params, app_loc_t<1>::bg_desc_t bgdesc)
{
    // Shared memory is allocated dynamically
    extern __shared__ char smem[];

    //------------------------------------------------------------------
    // Kernel configuration template
    typedef ldpc2_reg_box_plus_kernel_config<1,
                                             NUM_STORAGE_WORDS_BG1,
                                             MAX_NUM_PARITY_BG1,
                                             ldpc2::LDPC_kernel_params> kernel_config_t;

    //------------------------------------------------------------------
    // Load LLR data from global to shared memory
    kernel_config_t::llr_loader_t::load_sync(smem, params, blockIdx.x);

    //------------------------------------------------------------------
    // Perform iterations
    kernel_config_t::sched_t sched(params,
                                   bgdesc,
                                   static_cast<int>(__cvta_generic_to_shared(smem)),
                                   threadIdx.x);
    for(int iter = 0; iter < params.max_iterations; ++iter)
    {
        sched.do_iteration();
    }

    //------------------------------------------------------------------
    // Write hard output based on APP values
    //ldpc_dec_output_variable(params, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    ldpc_dec_output_variable_loop(params, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    //------------------------------------------------------------------
    // Write soft outputs if the caller provided a buffer
    if(params.soft_out != nullptr)
    {
        ldpc_dec_soft_output(params, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    }
}

////////////////////////////////////////////////////////////////////////
// ldpc2_BG2_reg_box_plus()
// Base graph 2 kernel, "legacy" tensor interface
extern "C"
__global__ __launch_bounds__(MAX_THREADS_PER_CTA, MIN_CTA_PER_SM)
void ldpc2_BG2_reg_box_plus(LDPC_kernel_params params, app_loc_t<2>::bg_desc_t bgdesc)
{
    // Shared memory is allocated dynamically
    extern __shared__ char smem[];
    
    //------------------------------------------------------------------
    // Kernel configuration template
    typedef ldpc2_reg_box_plus_kernel_config<2, // BG
                                             NUM_STORAGE_WORDS_BG2,
                                             MAX_NUM_PARITY_BG2,
                                             ldpc2::LDPC_kernel_params> kernel_config_t;

    //------------------------------------------------------------------
    // Load LLR data from global to shared memory
    kernel_config_t::llr_loader_t::load_sync(smem, params, blockIdx.x);

    //------------------------------------------------------------------
    // Perform iterations
    kernel_config_t::sched_t sched(params,
                                   bgdesc,
                                   static_cast<int>(__cvta_generic_to_shared(smem)),
                                   threadIdx.x);
    for(int iter = 0; iter < params.max_iterations; ++iter)
    {
        sched.do_iteration();
    }

    //------------------------------------------------------------------
    // Write hard output based on APP values
    ldpc_dec_output_variable(params, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    // No loop needed for BG2 with Z>= 32
    //ldpc_dec_output_variable_loop(params, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    //------------------------------------------------------------------
    // Write soft outputs if the caller provided a buffer
    if(params.soft_out != nullptr)
    {
        ldpc_dec_soft_output(params, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    }
}

////////////////////////////////////////////////////////////////////////
// ldpc2_BG1_reg_box_plus_tb()
// Base graph 1 kernel, transport block interface
extern "C"
__global__ __launch_bounds__(MAX_THREADS_PER_CTA, MIN_CTA_PER_SM)
void ldpc2_BG1_reg_box_plus_tb(cuphyLDPCDecodeDesc_t decodeDesc, app_loc_t<1>::bg_desc_t bgdesc)
{
    // Shared memory is allocated dynamically
    extern __shared__ char smem[];

    //------------------------------------------------------------------
    // Kernel configuration template
    typedef ldpc2_reg_box_plus_kernel_config<1,
                                             NUM_STORAGE_WORDS_BG1,
                                             MAX_NUM_PARITY_BG1,
                                             cuphyLDPCDecodeConfigDesc_t> kernel_config_t;

    //------------------------------------------------------------------
    // Load LLR data from global to shared memory
    kernel_config_t::llr_loader_t::load_sync(smem, decodeDesc, blockIdx.x);

    //------------------------------------------------------------------
    // Perform iterations
    kernel_config_t::sched_t sched(decodeDesc.config,
                                   bgdesc,
                                   static_cast<int>(__cvta_generic_to_shared(smem)),
                                   threadIdx.x);
    for(int iter = 0; iter < decodeDesc.config.max_iterations; ++iter)
    {
        sched.do_iteration();
    }

    //------------------------------------------------------------------
    // Write hard output based on APP values
    //ldpc_dec_output_variable(decodeDesc, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    ldpc_dec_output_variable_loop(decodeDesc, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    //------------------------------------------------------------------
    // Write soft outputs if the caller requested
    if(0 != (decodeDesc.config.flags & CUPHY_LDPC_DECODE_WRITE_SOFT_OUTPUTS))
    {
        ldpc_dec_soft_output(decodeDesc, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    }
}

////////////////////////////////////////////////////////////////////////
// ldpc2_BG1_reg_box_plus_tb_Z384_fast()
// Base graph 1 kernel, transport block interface, optimized dispatch
// target for Z=384 parity[4..46] max_iterations>0.
extern "C"
__global__ __launch_bounds__(MAX_THREADS_PER_CTA, MIN_CTA_PER_SM)
void ldpc2_BG1_reg_box_plus_tb_Z384_fast(LDPC2_GRID_CONST cuphyLDPCDecodeDesc_t decodeDesc,
                                         LDPC2_GRID_CONST app_loc_t<1>::bg_desc_t bgdesc)
{
    // Shared memory is allocated dynamically
    extern __shared__ char smem[];

    //------------------------------------------------------------------
    // Kernel configuration template
    typedef ldpc2_reg_box_plus_kernel_config_core_split<1,
                                                        NUM_STORAGE_WORDS_BG1_FAST_CORE,
                                                        NUM_STORAGE_WORDS_BG1_FAST_NONCORE,
                                                        MAX_NUM_PARITY_BG1,
                                                        cuphyLDPCDecodeConfigDesc_t> kernel_config_t;

    //------------------------------------------------------------------
    // Load LLR data from global to shared memory
    kernel_config_t::llr_loader_t::load_sync(smem, decodeDesc, blockIdx.x);

    //------------------------------------------------------------------
    // Perform iterations
    kernel_config_t::sched_t sched(decodeDesc.config,
                                   bgdesc,
                                   static_cast<int>(__cvta_generic_to_shared(smem)),
                                   threadIdx.x);
    for(int iter = 0; iter < decodeDesc.config.max_iterations; ++iter)
    {
        sched.do_iteration();
    }

    //------------------------------------------------------------------
    // Write hard output based on APP values
    ldpc_dec_output_variable_loop(decodeDesc, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    //------------------------------------------------------------------
    // Write soft outputs if the caller requested
    if(0 != (decodeDesc.config.flags & CUPHY_LDPC_DECODE_WRITE_SOFT_OUTPUTS))
    {
        ldpc_dec_soft_output(decodeDesc, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    }
}

////////////////////////////////////////////////////////////////////////
// ldpc2_BG2_reg_box_plus_tb()
// Base graph 2 kernel, transport block interface
extern "C"
__global__ __launch_bounds__(MAX_THREADS_PER_CTA, MIN_CTA_PER_SM)
void ldpc2_BG2_reg_box_plus_tb(cuphyLDPCDecodeDesc_t decodeDesc, app_loc_t<2>::bg_desc_t bgdesc)
{
    // Shared memory is allocated dynamically
    extern __shared__ char smem[];
    
    //------------------------------------------------------------------
    // Kernel configuration template
    typedef ldpc2_reg_box_plus_kernel_config<2, // BG
                                             NUM_STORAGE_WORDS_BG2,
                                             MAX_NUM_PARITY_BG2,
                                             cuphyLDPCDecodeConfigDesc_t> kernel_config_t;

    //------------------------------------------------------------------
    // Load LLR data from global to shared memory
    kernel_config_t::llr_loader_t::load_sync(smem, decodeDesc, blockIdx.x);

    //------------------------------------------------------------------
    // Perform iterations
    kernel_config_t::sched_t sched(decodeDesc.config,
                                   bgdesc,
                                   static_cast<int>(__cvta_generic_to_shared(smem)),
                                   threadIdx.x);
    for(int iter = 0; iter < decodeDesc.config.max_iterations; ++iter)
    {
        sched.do_iteration();
    }

    //------------------------------------------------------------------
    // Write hard output based on APP values
    ldpc_dec_output_variable(decodeDesc, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    // No loop needed for BG2 with Z>= 32
    //ldpc_dec_output_variable_loop(decodeDesc, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    //------------------------------------------------------------------
    // Write soft outputs if the caller requested
    if(0 != (decodeDesc.config.flags & CUPHY_LDPC_DECODE_WRITE_SOFT_OUTPUTS))
    {
        ldpc_dec_soft_output(decodeDesc, reinterpret_cast<const kernel_config_t::app_buf_t*>(smem));
    }
}

namespace ldpc2
{

////////////////////////////////////////////////////////////////////////
// reg_box_plus::decode()
cuphyStatus_t reg_box_plus::decode(ldpc::decoder&                     dec,
                                   LDPC_output_t&                     tDst,
                                   const_tensor_pair&                 tLLR,
                                   const cuphy_optional<tensor_pair>& optSoftOutputs,
                                   const cuphyLDPCDecodeConfigDesc_t& config,
                                   cudaStream_t                       strm)
{
    DEBUG_PRINTF("ldpc2::reg_box_plus::decode()\n");
    //------------------------------------------------------------------
    cuphyDataType_t llrType = tLLR.first.get().type();
    const int       NUM_CW  = tLLR.first.get().layout().dimensions[1];
    //------------------------------------------------------------------
    dim3 grdDim(NUM_CW);
    dim3 blkDim(config.Z);

    //------------------------------------------------------------------
    // Initialize the kernel params struct
    LDPC_kernel_params params(config, tLLR, tDst, optSoftOutputs);

    cuphyStatus_t s = CUPHY_STATUS_NOT_SUPPORTED;
    
    //------------------------------------------------------------------
    // Determine the dynamic amount of shared memory, which is the same
    // for both fp16 and fp32 (after conversion).
    const uint32_t SHMEM_SIZE = shmem_llr_buffer_size(params.num_var_nodes, // num shared memory nodes
                                                      params.Z,             // lifting size
                                                      sizeof(__half));      // element size
    if(llrType == CUPHY_R_16F)
    {
        switch(config.BG)
        {
        case 1:
            {
                //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
                // Retrieve the base graph descriptor
                const app_loc_t<1>::bg_desc_t* bgdesc = app_loc_t<1>::get_bg_desc(params.Z);
                if(!bgdesc) break;
                
                DEBUG_PRINT_FUNC_MAX_BLOCKS(ldpc2_BG1_reg_box_plus, blkDim, SHMEM_SIZE);
                
                //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
                // Launch the kernel
                ldpc2_BG1_reg_box_plus<<<grdDim, blkDim, SHMEM_SIZE, strm>>>(params, *bgdesc);
                s = CUPHY_STATUS_SUCCESS;

            }
            break;
        case 2:
            {
                //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
                // Retrieve the base graph descriptor
                const app_loc_t<2>::bg_desc_t* bgdesc = app_loc_t<2>::get_bg_desc(params.Z);
                if(!bgdesc) break;

                DEBUG_PRINT_FUNC_MAX_BLOCKS(ldpc2_BG2_reg_box_plus, blkDim, SHMEM_SIZE);
                
                //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
                // Launch the kernel
                ldpc2_BG2_reg_box_plus<<<grdDim, blkDim, SHMEM_SIZE, strm>>>(params, *bgdesc);
                s = CUPHY_STATUS_SUCCESS;
            }
            break;
        default:
            break;
        }
    }
    
    if(CUPHY_STATUS_SUCCESS != s)
    {
        return s;
    }

#if CUPHY_DEBUG
    cudaDeviceSynchronize();
#endif
    cudaError_t e = cudaGetLastError();
    DEBUG_PRINTF("CUDA STATUS (%s:%i): %s\n", __FILE__, __LINE__, cudaGetErrorString(e));
    return (e == cudaSuccess) ? CUPHY_STATUS_SUCCESS : CUPHY_STATUS_INTERNAL_ERROR;
}

////////////////////////////////////////////////////////////////////////
// reg_box_plus::decode_tb()
cuphyStatus_t reg_box_plus::decode_tb(ldpc::decoder&               dec,
                                      const cuphyLDPCDecodeDesc_t& decodeDesc,
                                      cudaStream_t                 strm)
{
    DEBUG_PRINTF("ldpc2::reg_box_plus::decode_tb()\n");    
    cuphyStatus_t s = CUPHY_STATUS_NOT_SUPPORTED;
    //------------------------------------------------------------------
    // Make sure that at least the first output pointer is non-NULL if
    // writing soft outputs is requested.
    assert((0 == (decodeDesc.config.flags & CUPHY_LDPC_DECODE_WRITE_SOFT_OUTPUTS)) ||
           (decodeDesc.llr_output[0].addr));
    //------------------------------------------------------------------
    dim3 grdDim(ldpc::decoder::get_total_num_codewords(decodeDesc));
    dim3 blkDim(decodeDesc.config.Z);

    if(decodeDesc.config.llr_type == CUPHY_R_16F)
    {
        switch(decodeDesc.config.BG)
        {
        case 1:
            {
                //------------------------------------------------------------------
                // Fast-path condition for native BG1 TB kernel dispatch.
                const bool fast_path_eligible =
                    (decodeDesc.config.Z == 384) &&
                    (decodeDesc.config.num_parity_nodes >= 4) &&
                    (decodeDesc.config.num_parity_nodes <= MAX_NUM_PARITY_BG1) &&
                    (decodeDesc.config.max_iterations > 0);

                //------------------------------------------------------------------
                // Determine the dynamic amount of shared memory
                const uint32_t SHMEM_SIZE = shmem_llr_buffer_size(decodeDesc.config.num_parity_nodes + max_info_nodes<1>::value, // num shared memory nodes
                                                                  decodeDesc.config.Z,                                           // lifting size
                                                                  sizeof(__half));                                               // element size

                //------------------------------------------------------------------
                // Retrieve the base graph descriptor
                const app_loc_t<1>::bg_desc_t* bgdesc = app_loc_t<1>::get_bg_desc(decodeDesc.config.Z);
                if(!bgdesc) break;

                DEBUG_PRINTF("ldpc2::reg_box_plus::decode_tb() BG1 dispatch: Z=%d parity=%d max_iter=%d fast=%d\n",
                             decodeDesc.config.Z,
                             decodeDesc.config.num_parity_nodes,
                             decodeDesc.config.max_iterations,
                             static_cast<int>(fast_path_eligible));

                //------------------------------------------------------------------
                // Launch the kernel
                if(fast_path_eligible)
                {
                    DEBUG_PRINT_FUNC_MAX_BLOCKS(ldpc2_BG1_reg_box_plus_tb_Z384_fast, blkDim, SHMEM_SIZE);
                    ldpc2_BG1_reg_box_plus_tb_Z384_fast<<<grdDim, blkDim, SHMEM_SIZE, strm>>>(decodeDesc, *bgdesc);
                }
                else
                {
                    DEBUG_PRINT_FUNC_MAX_BLOCKS(ldpc2_BG1_reg_box_plus_tb, blkDim, SHMEM_SIZE);
                    ldpc2_BG1_reg_box_plus_tb<<<grdDim, blkDim, SHMEM_SIZE, strm>>>(decodeDesc, *bgdesc);
                }
                s = CUPHY_STATUS_SUCCESS;
            }
            break;
        case 2:
            {
                //------------------------------------------------------------------
                // Determine the dynamic amount of shared memory
                const uint32_t SHMEM_SIZE = shmem_llr_buffer_size(decodeDesc.config.num_parity_nodes + max_info_nodes<2>::value, // num shared memory nodes
                                                                  decodeDesc.config.Z,                                           // lifting size
                                                                  sizeof(__half));                                               // element size

                //------------------------------------------------------------------
                // Retrieve the base graph descriptor
                const app_loc_t<2>::bg_desc_t* bgdesc = app_loc_t<2>::get_bg_desc(decodeDesc.config.Z);
                if(!bgdesc) break;
                
                DEBUG_PRINT_FUNC_MAX_BLOCKS(ldpc2_BG2_reg_box_plus_tb, blkDim, SHMEM_SIZE);
                
                //------------------------------------------------------------------
                // Launch the kernel
                ldpc2_BG2_reg_box_plus_tb<<<grdDim, blkDim, SHMEM_SIZE, strm>>>(decodeDesc, *bgdesc);
                s = CUPHY_STATUS_SUCCESS;
            }
            break;
        default:
            break;
        }
    }
    if(CUPHY_STATUS_SUCCESS != s)
    {
        return s;
    }

#if CUPHY_DEBUG
    cudaDeviceSynchronize();
#endif
    cudaError_t e = cudaGetLastError();
    DEBUG_PRINTF("CUDA STATUS (%s:%i): %s\n", __FILE__, __LINE__, cudaGetErrorString(e));
    return (e == cudaSuccess) ? CUPHY_STATUS_SUCCESS : CUPHY_STATUS_INTERNAL_ERROR;
}

////////////////////////////////////////////////////////////////////////
// reg_box_plus::get_workspace_size()
std::pair<bool, size_t> reg_box_plus::get_workspace_size(const ldpc::decoder&               dec,
                                                         const cuphyLDPCDecodeConfigDesc_t& config,
                                                         int                                num_cw)
{
    return std::pair<bool, size_t>(true, 0);
}

////////////////////////////////////////////////////////////////////////
// reg_box_plus::reg_box_plus()
reg_box_plus::reg_box_plus(ldpc::decoder& desc)
{
    const uint32_t MAX_VAR_NODES_BG1 = ldpc2::max_variable_nodes<1>::value;
    const uint32_t MAX_VAR_NODES_BG2 = ldpc2::max_variable_nodes<2>::value;
    //------------------------------------------------------------------
    // Determine the maximum amount of shared memory that could be used
    // by a kernel
    const int MAX_BG1_SHMEM_SIZE = static_cast<int>(shmem_llr_buffer_size(MAX_VAR_NODES_BG1,           // num shared memory nodes
                                                                          CUPHY_LDPC_MAX_LIFTING_SIZE, // lifting size
                                                                          sizeof(__half)));            // element size
    const int MAX_BG2_SHMEM_SIZE = static_cast<int>(shmem_llr_buffer_size(MAX_VAR_NODES_BG2,           // num shared memory nodes
                                                                          CUPHY_LDPC_MAX_LIFTING_SIZE, // lifting size
                                                                          sizeof(__half)));            // element size

    //------------------------------------------------------------------
    // For each kernel, set the maximum dynamic shared memory size
    typedef std::pair<const void*, int> func_attr_t;
    std::array<func_attr_t, 5> func_attrs =
    {
        func_attr_t((const void*)ldpc2_BG1_reg_box_plus,    MAX_BG1_SHMEM_SIZE),
        func_attr_t((const void*)ldpc2_BG2_reg_box_plus,    MAX_BG2_SHMEM_SIZE),
        func_attr_t((const void*)ldpc2_BG1_reg_box_plus_tb, MAX_BG1_SHMEM_SIZE),
        func_attr_t((const void*)ldpc2_BG2_reg_box_plus_tb, MAX_BG2_SHMEM_SIZE),
        func_attr_t((const void*)ldpc2_BG1_reg_box_plus_tb_Z384_fast, MAX_BG1_SHMEM_SIZE)
    };
    for(func_attr_t f_a : func_attrs)
    {
        cudaError_t e = cudaFuncSetAttribute(f_a.first,
                                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                                             f_a.second);
        if(cudaSuccess != e)
        {
            throw cuphy_i::cuda_exception(e);
        }
    }
    //------------------------------------------------------------------
    // Favor shared memory for the BG1 Z384 fast kernel.
    {
        cudaError_t e = cudaFuncSetAttribute((const void*)ldpc2_BG1_reg_box_plus_tb_Z384_fast,
                                             cudaFuncAttributePreferredSharedMemoryCarveout,
                                             0);
        if(cudaSuccess != e)
        {
            throw cuphy_i::cuda_exception(e);
        }
    }
    //------------------------------------------------------------------
    DEBUG_PRINT_FUNC_ATTRIBUTES(ldpc2_BG1_reg_box_plus);
    DEBUG_PRINT_FUNC_ATTRIBUTES(ldpc2_BG2_reg_box_plus);
    DEBUG_PRINT_FUNC_ATTRIBUTES(ldpc2_BG1_reg_box_plus_tb);
    DEBUG_PRINT_FUNC_ATTRIBUTES(ldpc2_BG2_reg_box_plus_tb);
    DEBUG_PRINT_FUNC_ATTRIBUTES(ldpc2_BG1_reg_box_plus_tb_Z384_fast);
}

////////////////////////////////////////////////////////////////////////
// reg_box_plus::get_launch_config()
cuphyStatus_t reg_box_plus::get_launch_config(const ldpc::decoder&           dec,
                                              cuphyLDPCDecodeLaunchConfig_t& launchConfig)
{
    const int Z                = launchConfig.decode_desc.config.Z;
    const int BG               = launchConfig.decode_desc.config.BG;
    const int NUM_PARITY_NODES = launchConfig.decode_desc.config.num_parity_nodes;
    const int MAX_PARITY_NODES = (1 == BG)                  ?
                                 max_parity_nodes<1>::value :
                                 max_parity_nodes<2>::value;
    const int NUM_VAR_NODES    = ldpc::decoder::get_num_variable_nodes(BG,
                                                                       NUM_PARITY_NODES);
    const bool FAST_BG1_Z384_ELIGIBLE =
        (1 == BG) &&
        (launchConfig.decode_desc.config.llr_type == CUPHY_R_16F) &&
        (Z == 384) &&
        (NUM_PARITY_NODES >= 4) &&
        (NUM_PARITY_NODES <= MAX_NUM_PARITY_BG1) &&
        (launchConfig.decode_desc.config.max_iterations > 0);
    //------------------------------------------------------------------
    // Validate input arguments
    if((Z < 2)                              ||
       (Z > CUPHY_LDPC_MAX_LIFTING_SIZE)    ||
       (NUM_PARITY_NODES < 4)               ||
       (NUM_PARITY_NODES > MAX_PARITY_NODES))
    {
        return CUPHY_STATUS_UNSUPPORTED_CONFIG;
    }

    //------------------------------------------------------------------
    // Set up launch geometry and the kernel function (driver)
    #if CUDART_VERSION >= 11000
    launchConfig.kernel_node_params_driver.blockDimX = Z;
    launchConfig.kernel_node_params_driver.blockDimY = 1;
    launchConfig.kernel_node_params_driver.blockDimZ = 1;

    launchConfig.kernel_node_params_driver.gridDimX = ldpc::decoder::get_total_num_codewords(launchConfig.decode_desc);
    launchConfig.kernel_node_params_driver.gridDimY = 1;
    launchConfig.kernel_node_params_driver.gridDimZ = 1;

    launchConfig.kernel_node_params_driver.extra          = nullptr;
    launchConfig.kernel_node_params_driver.kernelParams   = launchConfig.kernel_args;
    launchConfig.kernel_node_params_driver.sharedMemBytes = shmem_llr_buffer_size(NUM_VAR_NODES,   // num shared memory nodes
                                                                                  Z,               // lifting size
                                                                                  sizeof(__half)); // element size

    cudaFunction_t deviceFunction;
    MemtraceDisableScope md;
    cudaError_t e;
    if(BG == 1)
    {
        const void* bg1_func = FAST_BG1_Z384_ELIGIBLE ?
                               (const void*)ldpc2_BG1_reg_box_plus_tb_Z384_fast :
                               (const void*)ldpc2_BG1_reg_box_plus_tb;
        e = cudaGetFuncBySymbol(&deviceFunction, bg1_func);
    }
    else
    {
        e = cudaGetFuncBySymbol(&deviceFunction, (void*)ldpc2_BG2_reg_box_plus_tb);
    }
    if (e != cudaSuccess) 
    {
        return CUPHY_STATUS_INTERNAL_ERROR;
    }
    launchConfig.kernel_node_params_driver.func = static_cast<CUfunction>(deviceFunction);
    #endif
    //------------------------------------------------------------------
    // Set kernel arguments:
    // arg 0: decode descriptor
    launchConfig.kernel_args[0] = &launchConfig.decode_desc;
    // arg 1: base graph descriptor
    if(1 == BG)
    {
        const app_loc_t<1>::bg_desc_t* bgdesc = app_loc_t<1>::get_bg_desc(Z);
        launchConfig.kernel_args[1] = const_cast<void*>(reinterpret_cast<const void*>(bgdesc));
    }
    else
    {
        const app_loc_t<2>::bg_desc_t* bgdesc = app_loc_t<2>::get_bg_desc(Z);
        launchConfig.kernel_args[1] = const_cast<void*>(reinterpret_cast<const void*>(bgdesc));
    }
    return CUPHY_STATUS_SUCCESS;
}

} // namespace ldpc2
