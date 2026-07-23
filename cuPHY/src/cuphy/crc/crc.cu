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

#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>
#include "G_CRC_16_P_LUT.h"
#include "G_CRC_24_A_P_LUT.h"
#include "G_CRC_24_B_P_LUT.h"
#include "crc.cuh"
#include "crc.hpp"
#include "cuphy.h"
#include "cuphy.hpp"
#include "cuphy_internal.h"
#include "CRC_256_LUTS.h"
#include "crc_decode.hpp"
#include "crc_encode.hpp"
#define _WITH_LARGE_RADIX_TABLE_ 0

#include <cooperative_groups.h>

#if((__CUDACC_VER_MAJOR__ >= 12) || (__CUDACC_VER_MAJOR__ == 11 && (__CUDACC_VER_MINOR__ >= 1)))
#include <cooperative_groups/memcpy_async.h>
#endif

#define BIT_FLIP 1

namespace cg = cooperative_groups;

using namespace cuphy_i;

namespace crc
{
// 5G spec 24-bit polynomial a (Transport Block) and b (Code Block)
__constant__ uint32_t POLY_A[1]  = {G_CRC_24_A};
__constant__ uint32_t POLY_B[1]  = {G_CRC_24_B};
__constant__ uint32_t POLY_16[1] = {G_CRC_16};
__device__ uint32_t   NTBS[1];

// KERNEL copmputing CRC of Code Blocks and assembling the Transport Block
__global__ void crcUplinkPuschCodeBlocksKernel(const __grid_constant__ puschRxCrcDecodeDescr_t desc)
{
    uint32_t*          outputCBCRCs    = desc.pOutputCBCRCs;
    uint8_t*           outputTBs       = desc.pOutputTBs;
    const uint32_t*    inputCodeBlocks = desc.pInputCodeBlocks;
    uint32_t*          outputTBCRCs    = desc.pOutputTBCRCs;
    const PerTbParams* tbPrmsArray     = desc.pTbPrmsArray;
    bool               reverseBytes    = desc.reverseBytes;
    uint16_t           ueIdx           = desc.schUserIdxs[blockIdx.y];

    // exit if blockIdx.x is less than number of code blocks for current transport block
    uint32_t codeBlockIdx = blockIdx.x + tbPrmsArray[ueIdx].firstCodeBlockIndex;
    if(blockIdx.x >= tbPrmsArray[ueIdx].num_CBs)
        return;

    // Shared memory for CRC xor-based reduction of partial results
    extern __shared__ uint32_t shmemBuf[];

    uint32_t crcPolyBitSize = (tbPrmsArray[ueIdx].K - tbPrmsArray[ueIdx].F) > (MAX_SMALL_A_BITS + SMALL_L_BITS) || tbPrmsArray[ueIdx].num_CBs > 1 ? LARGE_L_BITS : SMALL_L_BITS;

    // CB CRC kernel removes CB CRC as part of CB desegmentation. The TB CRC kernel does not remove the TB CRC.
    // However the TB CRC kernel does not run if the TB has a single CB.
    // Consequently, if nCBs in a TB are > 1, TB payload = TB data + TB CRC
    // To force the same behavior (i.e. TB payload = TB data + CRC) for nCBs = 1, do not skip CB CRC
    if(1 == tbPrmsArray[ueIdx].num_CBs) crcPolyBitSize = 0;

    uint32_t codeBlockDataByteSize = (tbPrmsArray[ueIdx].K - tbPrmsArray[ueIdx].F - crcPolyBitSize + 8 - 1) / 8;
    int      tid                   = threadIdx.x;
    uint32_t size                  = (tbPrmsArray[ueIdx].K - tbPrmsArray[ueIdx].F + 32 - 1) / 32; // pad to 32-bit boundary

    // base addresses for transport block and code block for block y
    uint32_t tbBase         = 0;
    uint32_t cbBase         = 0;
    uint32_t nCodeBlocksSum = 0;
    for(int i = 1; i <= blockIdx.y; i++)
    {
        uint16_t prevUeIdx = desc.schUserIdxs[i - 1];
        uint32_t prevCrcPolyBitSize = (tbPrmsArray[prevUeIdx].K - tbPrmsArray[prevUeIdx].F) > (MAX_SMALL_A_BITS + SMALL_L_BITS) || tbPrmsArray[prevUeIdx].num_CBs > 1 ? LARGE_L_BITS : SMALL_L_BITS;
        if(1 == tbPrmsArray[prevUeIdx].num_CBs) prevCrcPolyBitSize = 0;

        uint32_t prevCodeBlockDataByteSize = (tbPrmsArray[prevUeIdx].K - tbPrmsArray[prevUeIdx].F - prevCrcPolyBitSize + 8 - 1) / 8;

        tbBase += tbPrmsArray[prevUeIdx].num_CBs * prevCodeBlockDataByteSize;
        tbBase += (sizeof(uint32_t) - tbBase % sizeof(uint32_t)) % sizeof(uint32_t);
        cbBase += tbPrmsArray[prevUeIdx].num_CBs * MAX_WORDS_PER_CODE_BLOCK;
        nCodeBlocksSum += tbPrmsArray[prevUeIdx].num_CBs;

        // if(0 == threadIdx.x) printf("tb %d blockIdx (x, y) (%d %d) firstCodeBlockIndex %d nCBs %d K %d F %d size %d tbBase %d cbBase %d nCodeBlocksSum %d prevCodeBlockDataByteSize %d prevCrcPolyBitSize %d\n", i, blockIdx.x, blockIdx.y, tbPrmsArray[blockIdx.y].firstCodeBlockIndex, tbPrmsArray[blockIdx.y].num_CBs, tbPrmsArray[blockIdx.y].K, tbPrmsArray[blockIdx.y].F, size, tbBase, cbBase, nCodeBlocksSum, prevCodeBlockDataByteSize, prevCrcPolyBitSize);
    }

    // 1) Compute CRCs of code blocks; assume size of code block data + crc +
    // right zero padding (in bytes) bytes is divisible by 4. Also Assemble
    // Transport Block
    uint32_t inVal = 0;
    uint32_t crc   = 0;

    uint8_t* tb = (uint8_t*)outputTBs;
    tb += tbBase;

    while(tid < size)
    {
        // CRC "map"

        inVal = inputCodeBlocks[cbBase + codeBlockIdx * MAX_WORDS_PER_CODE_BLOCK + tid];
        if(reverseBytes)
        {
            inVal = __brev(inVal);
            inVal = swap<32>(inVal);
        }
        // if(((12 == blockIdx.y) || (13 == blockIdx.y) || (14 == blockIdx.y) || (15 == blockIdx.y)) && (0 == blockIdx.x) && (0 == threadIdx.x)) printf("blockIdx (x,y) (%d,%d) tid %d size %d tbVal[%d] 0x%08x\n", blockIdx.x, blockIdx.y, tid, size, cbBase + codeBlockIdx * MAX_WORDS_PER_CODE_BLOCK + tid, inVal);

        // If transport block is small (single code block) compute CRC using TB polynomial and skip TB CRC computation in next kernel
        if(tbPrmsArray[ueIdx].num_CBs <= 1)
        {
            // for CBs of size less than 3808 bits use 16-bit CRC
            if(codeBlockDataByteSize <= MAX_CB_BYTE_SIZE_FOR_CRC16)
            {
                crc ^= mulModCRCPolyLUT<uint16_t, 16>(inVal,
                                                      G_CRC_16_P_LUT[(tid)],
                                                      G_CRC_16_256_LUT,
                                                      *POLY_16);
            }
            else
                crc ^= mulModCRCPolyLUT<uint32_t, 24>(inVal,
                                                      G_CRC_24_A_P_LUT[(tid)],
                                                      G_CRC_24_A_256_LUT,
                                                      *POLY_A);
        }
        else
            crc ^= mulModCRCPolyLUT<uint32_t, 24>(inVal,
                                                  G_CRC_24_B_P_LUT[tid],
                                                  G_CRC_24_B_256_LUT,
                                                  *POLY_B);

        // Transport Block assembly
        if((tid * 4) < codeBlockDataByteSize)
            tb[codeBlockIdx * codeBlockDataByteSize + 4 * tid] = (uint8_t)inVal & 0xFF;
        if((tid * 4 + 1) < codeBlockDataByteSize)
            tb[codeBlockIdx * codeBlockDataByteSize + 4 * tid + 1] = (uint8_t)(inVal >> 8) & 0xFF;
        if((tid * 4 + 2) < codeBlockDataByteSize)
            tb[codeBlockIdx * codeBlockDataByteSize + 4 * tid + 2] = (uint8_t)(inVal >> 16) & 0xFF;
        if((tid * 4 + 3) < codeBlockDataByteSize)
            tb[codeBlockIdx * codeBlockDataByteSize + 4 * tid + 3] = (uint8_t)(inVal >> 24) & 0xFF;

        tid += blockDim.x;
    }

    crc = xorReductionWarpShared<uint32_t>(crc, shmemBuf);
    // zero pad TB up to bytesize divsible by 4
    if(threadIdx.x == 0)
    {
        *(outputCBCRCs + nCodeBlocksSum + codeBlockIdx) = crc;
        if(blockIdx.x == 0)
        {
            // zero out TB crc array for next kernel
            outputTBCRCs[blockIdx.y] = tbPrmsArray[ueIdx].num_CBs == 1 ? crc : 0;
            int rem4                 = (codeBlockDataByteSize * tbPrmsArray[ueIdx].num_CBs) % 4;
            if(rem4)
            {
                for(int i = 0; i < 4 - rem4; i++)
                {
                    tb[codeBlockDataByteSize * tbPrmsArray[ueIdx].num_CBs + i] = 0;
                }
            }
        }
    }
}

// KERNEL computing CRC of transport block; two levels (block and grid) of
__global__ void crcUplinkPuschTransportBlockKernel(const __grid_constant__ puschRxCrcDecodeDescr_t desc)
{
    const uint32_t*    inputTBs     = (uint32_t*)desc.pOutputTBs;
    uint32_t*          outputTBCRCs = desc.pOutputTBCRCs;
    const PerTbParams* tbPrmsArray  = desc.pTbPrmsArray;
    uint16_t           ueIdx        = desc.schUserIdxs[blockIdx.y];
    
    // Do nothing in the single code block per transport block case
    if(tbPrmsArray[ueIdx].num_CBs == 1)
    {
        return;
    }

    uint32_t crcPolyBitSize        = 24;
    uint32_t codeBlockDataByteSize = (tbPrmsArray[ueIdx].K - tbPrmsArray[ueIdx].F - crcPolyBitSize + 8 - 1) / 8;

    uint32_t tbSize = tbPrmsArray[ueIdx].num_CBs * codeBlockDataByteSize;
    tbSize += (sizeof(uint32_t) - tbSize % sizeof(uint32_t)) % sizeof(uint32_t);
    tbSize /= sizeof(uint32_t);

    extern __shared__ uint32_t shmemBuf[];
    int                        tid = blockDim.x * blockIdx.x + threadIdx.x;

    uint32_t tbBase = 0;

    for(int i = 1; i <= blockIdx.y; i++)
    {
        uint16_t prevUeIdx                 = desc.schUserIdxs[i - 1];
        uint32_t prevCrcPolyBitSize        = (tbPrmsArray[prevUeIdx].K - tbPrmsArray[prevUeIdx].F) > (MAX_SMALL_A_BITS + SMALL_L_BITS) || tbPrmsArray[prevUeIdx].num_CBs > 1 ? LARGE_L_BITS : SMALL_L_BITS;
        uint32_t prevCodeBlockDataByteSize = (tbPrmsArray[prevUeIdx].K - tbPrmsArray[prevUeIdx].F - prevCrcPolyBitSize + 8 - 1) / 8;

        tbBase += tbPrmsArray[prevUeIdx].num_CBs * prevCodeBlockDataByteSize + (tbPrmsArray[prevUeIdx].num_CBs == 1 ? (prevCrcPolyBitSize / 8) : 0);
        tbBase += (sizeof(uint32_t) - tbBase % sizeof(uint32_t)) % sizeof(uint32_t);
    }
    // Word size
    tbBase /= sizeof(uint32_t);

    int crc = 0;

    if(tid < tbSize)
    {
        crc = mulModCRCPolyLUT<uint32_t, 24>(inputTBs[tbBase + tid],
                                             G_CRC_24_A_P_LUT[(tid)],
                                             G_CRC_24_A_256_LUT,
                                             *POLY_A);
    }

    crc = xorReductionWarpShared<uint32_t>(crc, shmemBuf);

    // reduction
    if(threadIdx.x == 0)
    {
        atomicXor(&outputTBCRCs[blockIdx.y], crc);
    }
}

cuphyStatus_t launch(
    uint32_t*          d_cbCRCs,
    uint32_t*          d_tbCRCs,
    uint8_t*           d_transportBlocks,
    const uint32_t*    d_inputCodeBlocks,
    const PerTbParams* d_tbPrmsArray,
    uint32_t           nTBs,
    uint32_t           maxNCBsPerTB,
    uint32_t           maxTBByteSize,
    bool               reverseBytes,
    bool               timeIt,
    uint32_t           NRUNS,
    bool               codeBlocksOnly,
    cudaStream_t       strm)
{
    cuphyStatus_t status = CUPHY_STATUS_SUCCESS;
    if(nTBs > MAX_N_TBS_PER_CELL_GROUP_SUPPORTED)
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: Maximum number of Transport Blocks supported is {}", __FUNCTION__, MAX_N_TBS_PER_CELL_GROUP_SUPPORTED);
        return CUPHY_STATUS_NOT_SUPPORTED;
    }

    if(maxNCBsPerTB > MAX_N_CBS_PER_TB_SUPPORTED)
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: Maximum number of Code Blocks per Transport Block supported is {}", __FUNCTION__, MAX_N_CBS_PER_TB_SUPPORTED);
        return CUPHY_STATUS_NOT_SUPPORTED;
    }

    if(maxTBByteSize > MAX_BYTES_PER_TRANSPORT_BLOCK)
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: Transport Block size = {}. Maximum Transport Block size in bytes supported is {}", __FUNCTION__, maxTBByteSize, MAX_BYTES_PER_TRANSPORT_BLOCK);
        return CUPHY_STATUS_NOT_SUPPORTED;
    }

    const uint32_t crcPolyDegree = 24;
    const uint32_t blockSize     = GLOBAL_BLOCK_SIZE;
    uint32_t       gridSizeCBX   = maxNCBsPerTB;
    uint32_t       gridSizeCBY   = nTBs;
    uint32_t       tbSize        = (maxTBByteSize + sizeof(uint32_t) - 1) / sizeof(uint32_t);
    dim3           gCBSize(gridSizeCBX, gridSizeCBY);
    uint32_t       gridSizeTBX = (tbSize + blockSize - 1) / blockSize;
    uint32_t       gridSizeTBY = nTBs;
    dim3           gTBSize(gridSizeTBX, gridSizeTBY);

    uint8_t* descb;

    size_t descSize       = 0;
    size_t descAlignBytes = 0;

    status = cuphyPuschRxCrcDecodeGetDescrInfo(&descSize, &descAlignBytes);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: status is {}", __FUNCTION__, cuphyGetErrorString(status));
        return status;
    }

    unique_device_ptr<uint8_t> d_desc = make_unique_device<uint8_t>(descSize);
    CUDA_CHECK(cudaHostAlloc((void**)&descb, descSize, cudaHostAllocDefault));

    puschRxCrcDecodeDescr_t& desc = *(static_cast<puschRxCrcDecodeDescr_t*>((void*)descb));

    desc.pOutputCBCRCs    = d_cbCRCs;
    desc.pOutputTBs       = d_transportBlocks;
    desc.pInputCodeBlocks = d_inputCodeBlocks;
    desc.pOutputTBCRCs    = d_tbCRCs;
    desc.pTbPrmsArray     = d_tbPrmsArray;
    desc.reverseBytes     = reverseBytes;
    // Initialize schUserIdxs to identity for the simple API path; without this,
    // the kernel reads uninitialized indices and produces wrong CRCs on
    // heterogeneous-TB inputs (single-CB and multi-CB TBs in the same batch).
    for(uint32_t i = 0; i < nTBs; ++i) desc.schUserIdxs[i] = i;

    // Because the kernel is now grid constant, we do not need to copy the descriptor to GPU memory

    crcUplinkPuschCodeBlocksKernel<<<gCBSize, blockSize, sizeof(uint32_t) * WARP_SIZE, strm>>>(desc);

    if(!codeBlocksOnly)
    {
        crcUplinkPuschTransportBlockKernel<<<gTBSize, blockSize, sizeof(uint32_t) * WARP_SIZE, strm>>>(desc);
    }

    if(timeIt)
    {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        float time1 = 0.0;
        float time2 = 0.0;
        cudaEventRecord(start);

        for(int i = 0; i < NRUNS; i++)
        {
            crcUplinkPuschCodeBlocksKernel<<<gCBSize, blockSize, sizeof(uint32_t) * WARP_SIZE, strm>>>(desc);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time1, start, stop);

        if(!codeBlocksOnly)
        {
            cudaEventRecord(start);
            for(int i = 0; i < NRUNS; i++)
            {
                crcUplinkPuschTransportBlockKernel<<<gTBSize, blockSize, sizeof(uint32_t) * WARP_SIZE, strm>>>(desc);
            }
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&time2, start, stop);
        }
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        time1 /= NRUNS;
        time2 /= NRUNS;

        NVLOGI_FMT(NVLOG_PUSCH, "{}: CB KERNEL GRID_SIZE_X {}", __FUNCTION__, gridSizeCBX);
        if(!codeBlocksOnly)
            NVLOGI_FMT(NVLOG_PUSCH, "{}: TB KERNEL GRID_SIZE_X {}", __FUNCTION__, gridSizeTBX);

        NVLOGI_FMT(NVLOG_PUSCH, "{}: Kernel 1: Code blocks CRC and TB assembly({}-bit crc): {} us", __FUNCTION__, crcPolyDegree, time1 * 1000);

        if(!codeBlocksOnly)
            NVLOGI_FMT(NVLOG_PUSCH, "{}: Kernel 2: TB CRC({}-bit crc): {} us", __FUNCTION__, crcPolyDegree, time2 * 1000);
    }
    return status;
}

// KERNEL computing CRC of Code Blocks and assembling the Transport Block
__global__ void crcDownlinkPdschCodeBlocksKernel(const __grid_constant__ crcEncodeDescr_t desc)
{
    uint32_t*          outputCBCRCs = desc.d_cbCRCs;
    const PdschPerTbParams* tbPrmsArray  = desc.d_tbPrmsArray;
    bool               reverseBytes = desc.reverseBytes;
    uint32_t*          tbCRCs       = desc.d_tbCRCs;

    // exit if blockIdx.x is less than number of code blocks for current transport block or for TBs whose cells are in testing mode
    uint32_t codeBlockIdx = blockIdx.x + tbPrmsArray[blockIdx.y].firstCodeBlockIndex;
    if((blockIdx.x >= tbPrmsArray[blockIdx.y].num_CBs) || (tbPrmsArray[blockIdx.y].testModel != 0))
        return;

    // Shared memory for CRC xor-based reduction of partial results
    extern __shared__ uint32_t shmemBuf[];

    const uint8_t* __restrict__ d_inputTransportBlocks = (uint8_t*)desc.d_inputTransportBlocks;
    uint8_t* d_codeBlocks                              = desc.d_codeBlocks;

    int TB_id = blockIdx.y;
    int CB_id = blockIdx.x;

    uint32_t total_src_tb_offset = (TB_id == 0) ? 0 : tbPrmsArray[TB_id-1].cumulativeTbSizePadding;

    __shared__ uint32_t sh_total_dst_tb_offset;
    __shared__ uint32_t sh_nCodeBlocksSum;

    if(threadIdx.x == 0)
    {
        sh_total_dst_tb_offset = 0;
        sh_nCodeBlocksSum = 0;
    }
    __syncthreads();

    for(int i = threadIdx.x; i < TB_id; i+= blockDim.x)
    {
        // round up K so it's 32-bit aligned; offset is still in bytes
        atomicAdd(&sh_total_dst_tb_offset, (((tbPrmsArray[i].K + 31) >> 5) * sizeof(uint32_t) * tbPrmsArray[i].num_CBs));
    }
    __syncthreads();

    uint32_t per_CB_CRC_byte_size    = (tbPrmsArray[TB_id].num_CBs == 1) ? 0 : 3;
    //total_CB_data_byte_size has to by divisible by 4, as LDPC encode expects each CB to be 32-bit aligned.
    uint32_t total_CB_data_byte_size = sizeof(uint32_t) * ((tbPrmsArray[TB_id].K + 31) >> 5); // round up if K (in bits) not evenly divisible by 32, thus +31, i.e., 32-1
    // CB_data_byte_size is K - F but rounded up to nearest byte. It's valid data + padding copied from the input.
    uint32_t CB_data_byte_size       = ((tbPrmsArray[TB_id].K - tbPrmsArray[TB_id].F + 7) >> 3) - per_CB_CRC_byte_size; // + 7 is to round up if not evenly divisible by 8
    uint32_t CB_data_word_size       = (CB_data_byte_size - (CB_id == tbPrmsArray[TB_id].num_CBs - 1) * per_CB_CRC_byte_size + 4 - 1) / 4 + (CB_id > 0) * (((CB_id * CB_data_byte_size) % 4) > 0);
    uint32_t extra_padding           = total_CB_data_byte_size - CB_data_byte_size - per_CB_CRC_byte_size;

    /*if ((threadIdx.x == 0) && (blockIdx.x == 0))
        printf("TB %d, K %d, F %d, CB_data_byte_size %d, CB_data_word_size %d,  total_CB_data_byte_size %d, extra_padding %d\n",
                TB_id, tbPrmsArray[TB_id].K, tbPrmsArray[TB_id].F, CB_data_byte_size, CB_data_word_size, total_CB_data_byte_size, extra_padding);*/

    const uint8_t*  shmem_CB_input_bytes = (uint8_t*)(shmemBuf + crc::MAX_WORDS_PER_CODE_BLOCK) + ((CB_id * CB_data_byte_size) % 4);
    const uint32_t* CB_input_words       = (((uint32_t*)(d_inputTransportBlocks)) + (total_src_tb_offset + CB_id * CB_data_byte_size) / 4);
    uint8_t*        CB_output_bytes      = d_codeBlocks + sh_total_dst_tb_offset + CB_id * total_CB_data_byte_size;

    uint32_t crcPolyBitSize        = (tbPrmsArray[blockIdx.y].K - tbPrmsArray[blockIdx.y].F) > (MAX_SMALL_A_BITS + SMALL_L_BITS) || tbPrmsArray[blockIdx.y].num_CBs > 1 ? LARGE_L_BITS : SMALL_L_BITS;
    uint32_t codeBlockDataByteSize = (tbPrmsArray[blockIdx.y].K - tbPrmsArray[blockIdx.y].F - crcPolyBitSize + 8 - 1) >> 3;
    int      tid                   = threadIdx.x;
    uint32_t size                  = tbPrmsArray[blockIdx.y].K - tbPrmsArray[blockIdx.y].F; // pad to 32-bit boundary
    uint32_t cbShiftBits           = (32 - (size % 32)) % 32;
    uint32_t totalSize             = (tbPrmsArray[blockIdx.y].K + 32 - 1) / 32; // padded size including filler bits
    size                           = (size + cbShiftBits) >> 5;
    uint32_t cbShiftBytes          = cbShiftBits >> 3;

    uint8_t* dataBytes = reinterpret_cast<uint8_t*>(shmemBuf);

    // base addresses for transport block and code block for block y
    uint32_t cbBase         = 0;
    for(int i = threadIdx.x + 1; i <= blockIdx.y; i+=blockDim.x)
    {
        // cbBase is used to assemble final codeblocks
        // so the filler bits must be included
        // also needs to be padded to 32-bit boundary
        atomicAdd(&sh_nCodeBlocksSum, tbPrmsArray[i - 1].num_CBs);
    }
    //__syncthreads();
    cbBase += codeBlockIdx * totalSize;

    // 1) Compute CRCs of code blocks; assume size of code block data + crc +
    // right zero padding (in bytes) bytes is divisible by 4. Also Assemble
    // Transport Block
    uint32_t inVal = 0;
    uint32_t crc   = 0;

    uint8_t* tbCRCBytes = reinterpret_cast<uint8_t*>(tbCRCs);

#if((__CUDACC_VER_MAJOR__ >= 12) || (__CUDACC_VER_MAJOR__ == 11 && (__CUDACC_VER_MINOR__ >= 1)))
    auto group = cg::this_thread_block();
    cg::memcpy_async(group, shmemBuf + MAX_WORDS_PER_CODE_BLOCK, CB_input_words, sizeof(uint32_t) * CB_data_word_size);
    cg::wait(group);
#else
    for(int i = threadIdx.x; i < CB_data_word_size; i += blockDim.x)
    {
        shmemBuf[MAX_WORDS_PER_CODE_BLOCK + i] = CB_input_words[i];
    }
    __syncthreads();
#endif

    for(int i = threadIdx.x; i < CB_data_byte_size; i += blockDim.x)
    {
        uint8_t byte;

        if((CB_id == (tbPrmsArray[TB_id].num_CBs - 1)) && tbPrmsArray[TB_id].num_CBs > 1 && i >= CB_data_byte_size - per_CB_CRC_byte_size) // last CB, read TB CRC
        {
            byte = tbCRCBytes[TB_id * sizeof(uint32_t) + (i - (CB_data_byte_size - per_CB_CRC_byte_size))];
        }
        else
            byte = shmem_CB_input_bytes[i];
        CB_output_bytes[i]          = byte;
        dataBytes[i + cbShiftBytes] = byte;
        //if(i < (((tbPrmsArray[TB_id]. F + 7)>> 3))) // number of filler bits may not be evenly divisible by 8; round up
        if(i < extra_padding)  // number of filler bits may not be evenly divisible by 8; round up
            CB_output_bytes[CB_data_byte_size + per_CB_CRC_byte_size + i] = 0x0;
        if(i < (per_CB_CRC_byte_size))
            dataBytes[i + cbShiftBytes + CB_data_byte_size] = 0x0;
        /*if ((threadIdx.x == 0) && (blockIdx.x == 0)) {
            printf("TB %d, CB_data_byte_size %d (copying input), per_CB_CRC_byte_size %d, filler bits in bytes [%d, %d)\n",
                   TB_id, CB_data_byte_size, per_CB_CRC_byte_size, CB_data_byte_size + per_CB_CRC_byte_size, CB_data_byte_size + per_CB_CRC_byte_size+ min(extra_padding, CB_data_byte_size));
          }*/
    }

    /* It is possible that CB_data_byte_size < extra_padding, so not all padding was added above
       In the for-loop above, the max value of i can be CB_data_byte_size -1. When CB_data_byte_size < extra_padding, then
       at most CB_data_byte_size bytes out of the extra_padding bytes added in that loop.
       Padding in the loop is added up to and just before this position
       CB_output_bytes[CB_data_byte_size + per_CB_CRC_byte_size + (extra_padding > CB_data_byte_size) ?  CB_data_byte_size : extra_padding]

       The if-statement below is only executed if extra_padding > CB_data_byte_size and adds the leftover padding starting at the  index mentioned above.
       The extra CB_data_byte_size (2 multiplier below) is to account for the fact that CB_data_byte_size of padding was already added in the for loop.

       PDSCH TV 3303 is an example of a case where this is relevant. TB 0 has K=70, F=30, CB_data_byte_size = function of K - F is 5B;
       total_CB_data_byte_size 12 (K rounded up so it's divisible by 32 is 96bits = 12B). extra padding=7B which is greater than CB_data_byte_size.
       The first 5B of extra_padding from [5 to 10) will be added in the for loop. The remaining 2B at [10, 12) will be added via the following if statement.
    */
    int leftover_padding = extra_padding - CB_data_byte_size;
    if((leftover_padding > 0) && (threadIdx.x < leftover_padding)) {
        CB_output_bytes[2*CB_data_byte_size + per_CB_CRC_byte_size + threadIdx.x] = 0x0; // the extra CB_data_byte_size is the padding covered in the prev. for-loop
    /*  if ((threadIdx.x == 0) && (blockIdx.x == 0))
           printf("TB %d, leftover_padding %d writing to bytes [%d, %d)\n",
                 TB_id, leftover_padding, 2*CB_data_byte_size + per_CB_CRC_byte_size,
                 2*CB_data_byte_size + per_CB_CRC_byte_size+ leftover_padding);*/
    }
    if(threadIdx.x < cbShiftBytes)
    {
        dataBytes[threadIdx.x] = 0x0;
    }
    __syncthreads();

    while(tid < size)
    {
        // CRC "map"

        // inVal = inputCodeBlocks[tbBase + cbBase + tid];
        // CRC "map"
        inVal = shmemBuf[tid];

        if(reverseBytes)
        {
            inVal = __brev(inVal);
            inVal = swap<32>(inVal);
        }

        // If transport block is small (single code block) compute CRC using TB polynomial and skip TB CRC computation in next kernel
        if(tbPrmsArray[blockIdx.y].num_CBs <= 1)
        {
            // for TBs of size less than 3824 bits use 16-bit CRC
            if(codeBlockDataByteSize <= MAX_SMALL_A_BYTES)
            {
                int offset = G_CRC_16_P_LUT_SIZE - size + tid;

                uint32_t tabVal;
                if(tid == size - 1)
                {
                    inVal <<= 16;
                    tabVal = 1;
                }
                else
                    tabVal = G_CRC_16_P_LUT[offset + 1];
                crc ^= mulModCRCPolyLUT<uint16_t, 16>(inVal,
                                                      tabVal,
                                                      G_CRC_16_256_LUT,
                                                      *POLY_16);
            }
            else
            {
                int offset = (G_CRC_24_A_P_LUT_SIZE)-size + tid;

                uint32_t tabVal;
                if(tid == size - 1)
                {
                    inVal <<= 24;
                    tabVal = 1;
                }
                else
                    tabVal = G_CRC_24_A_P_LUT[offset + 1];

                crc ^= mulModCRCPolyLUT<uint32_t, 24>(inVal,
                                                      tabVal,
                                                      G_CRC_24_A_256_LUT,
                                                      *POLY_A);
            }
        }
        else
        {
            int offset = G_CRC_24_B_P_LUT_SIZE - size + tid;

            uint32_t tabVal;
            if(tid == size - 1)
            {
                inVal <<= 24;
                tabVal = 1;
            }
            else
                tabVal = G_CRC_24_B_P_LUT[offset + 1];

            crc ^= mulModCRCPolyLUT<uint32_t, 24>(inVal,
                                                  tabVal,
                                                  G_CRC_24_B_256_LUT,
                                                  *POLY_B);
        }

        tid += blockDim.x;
    }

    //__syncthreads();

    //crc = xorReductionWarpShared<uint32_t>(crc, shmemBuf);
    crc = xorReductionWarpShared<uint32_t>(crc, &shmemBuf[2 * MAX_WORDS_PER_CODE_BLOCK + 2]);

    // zero pad TB up to bytesize divsible by 4
    if(threadIdx.x == 0)
    {
        if(reverseBytes)
        { // FIXME might not be related to reverseBytes
            crc = __brev(crc) >> (32 - crcPolyBitSize);
        }

        // Separate per-CB CRCs are only useful for debugging. Disable if outputCBCRCs is nullptr.
        if(outputCBCRCs != nullptr)
        {
            *(outputCBCRCs + sh_nCodeBlocksSum + codeBlockIdx) = crc;
        }
        CB_output_bytes[codeBlockDataByteSize]     = (uint8_t)crc & 0xff;
        CB_output_bytes[codeBlockDataByteSize + 1] = (uint8_t)(crc >> 8) & 0xff;
        if(crcPolyBitSize == LARGE_L_BITS)
        { // Not executed in single-CB case with 16-bit per-TB CRC.
            CB_output_bytes[codeBlockDataByteSize + 2] = (uint8_t)(crc >> 16) & 0xff;
        }
    }
}

// KERNEL computing CRC of transport block; two levels (block and grid) of
__global__ void crcDownlinkPdschTransportBlockKernel(const __grid_constant__ crcEncodeDescr_t desc)
{
    uint32_t*          outputTBCRCs = desc.d_tbCRCs;
    const uint32_t*    inputTBs     = desc.d_inputTransportBlocks;
    const PdschPerTbParams* tbPrmsArray  = desc.d_tbPrmsArray;
    bool               reverseBytes = desc.reverseBytes;

    uint32_t crcPolyBitSize        = 24;
    uint32_t codeBlockDataByteSize = (tbPrmsArray[blockIdx.y].K - tbPrmsArray[blockIdx.y].F - crcPolyBitSize + 8 - 1) / 8;

    uint32_t tbSize = tbPrmsArray[blockIdx.y].num_CBs * codeBlockDataByteSize;

    uint32_t tbShiftBytes = (sizeof(uint32_t) - tbSize % sizeof(uint32_t)) % sizeof(uint32_t);
    tbSize += tbShiftBytes;
    tbSize /= sizeof(uint32_t);

    // Do nothing in the single code block per transport block case or for those TBs whose cells are in testing mode
    //if(tbSize <= MAX_WORDS_PER_CODE_BLOCK)
    if((tbPrmsArray[blockIdx.y].num_CBs == 1) || (tbPrmsArray[blockIdx.y].testModel != 0))
        return;

    extern __shared__ uint32_t shmemBuf[];
    int                        tid = blockDim.x * blockIdx.x + threadIdx.x;

#if 0
    uint32_t tbBase = 0;
    for(int i = 0; i < blockIdx.y; i++)
    {
        tbBase += (tbPrmsArray[i].tbSize + tbPrmsArray[i].paddingBytes);
    }
#else
    uint32_t tbBase = (blockIdx.y == 0) ? 0 : tbPrmsArray[blockIdx.y-1].cumulativeTbSizePadding;
#endif
    // Word size
    tbBase >>= 2; // divide by sizeof(uint32_t)

    int      crc   = 0;
    uint32_t in    = 0;
    uint32_t inVal = 0;
    uint64_t inPad = 0;
    /*
    if(tid < tbSize){
	    shmemBuf[threadIdx.x + 1] = *(inputTBs + tbBase + tid);
    if(threadIdx.x == 0)
	    shmemBuf[0] = tid == 0 ? 0 : *(inputTBs + tbBase + tid - 1);
    }
    __syncthreads();
*/
    if(tid < tbSize)
    {
        if(tid == 0)
        {
            in = *(inputTBs + tbBase + tid);
            //   in    = shmemBuf[threadIdx.x + 1];
            inVal = (in << (tbShiftBytes * 8)) & 0xffffffff;
        }
        else
        {
            in = *(inputTBs + tbBase + tid);
            // in    = shmemBuf[threadIdx.x + 1];
            inPad = (uint64_t)in << sizeof(uint32_t) * 8;
            inPad ^= *(inputTBs + tbBase + tid - 1);
            // inPad ^= shmemBuf[threadIdx.x];
            inVal = (inPad >> (32 - (tbShiftBytes * 8))) & 0xffffffff;
        }

        if(reverseBytes)
        { // reverseBytes reverses bit order and reorders bytes (endianness change)
            inVal = __brev(inVal);
            inVal = swap<32>(inVal);
        }

        int      offset = G_CRC_24_A_P_LUT_SIZE - tbSize + tid;
        uint32_t tabVal;

        if(tid == tbSize - 1)
        {
            inVal  = inVal << 24;
            tabVal = 1;
        }
        else
            tabVal = G_CRC_24_A_P_LUT[offset + 1];

        crc = mulModCRCPolyLUT<uint32_t, 24>(inVal,
                                             tabVal,
                                             G_CRC_24_A_256_LUT,
                                             *POLY_A);
    }

    //    __syncthreads();
    crc = xorReductionWarpShared<uint32_t>(crc, shmemBuf);

    // reduction
    if(threadIdx.x == 0)
    {
        if(reverseBytes)
        { //FIXME might not be related to reverseBytes
            crc = __brev(crc) >> 8;
        }

        //Per-TB CRC needs to be atomically updated for every TB (blockIdx.y).
        //Separate TB-CRCs are used by CB kernel.
        atomicXor(&outputTBCRCs[blockIdx.y], crc);
    }
}

} // namespace crc

cuphyStatus_t CUPHYWINAPI cuphySetupCrcEncode(cuphyCrcEncodeLaunchConfig_t crcEncodeLaunchConfig,
                                              uint32_t*                    d_cbCRCs,
                                              uint32_t*                    d_tbCRCs,
                                              const uint32_t*              d_inputTransportBlocks,
                                              uint8_t*                     d_codeBlocks,
                                              const PdschPerTbParams*      d_tbPrmsArray,
                                              uint32_t                     nTBs,
                                              uint32_t                     maxNCBsPerTB,
                                              uint32_t                     maxTBByteSize,
                                              uint8_t                      reverseBytes,
                                              uint8_t                      codeBlocksOnly,
                                              void*                        cpu_desc,
                                              void*                        gpu_desc,
                                              uint8_t                      enable_desc_async_copy,
                                              cudaStream_t                 strm)
{
    if(d_tbCRCs == nullptr)
    {
        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "{}: d_tbCRCs is equal to nullptr", __FUNCTION__);
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    if(d_inputTransportBlocks == nullptr)
    {
        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "{}: d_inputTransportBlocks is equal to nullptr", __FUNCTION__);
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    if(d_codeBlocks == nullptr)
    {
        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "{}: d_codeBlocks is equal to nullptr", __FUNCTION__);
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    if(d_tbPrmsArray == nullptr)
    {
        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "{}: d_tbPrmsArray is equal to nullptr", __FUNCTION__);
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    if(cpu_desc == nullptr)
    {
        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "{}: cpu_desc is equal to nullptr", __FUNCTION__);
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    if(gpu_desc == nullptr)
    {
        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "{}: gpu_desc is equal to nullptr", __FUNCTION__);
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    if(nTBs > PDSCH_MAX_UES_PER_CELL_GROUP) //FIXME Increase limit when grouping - maybe pass a batching param
    {
        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "{}: Maximum number of Transport Blocks supported is {}", __FUNCTION__, PDSCH_MAX_UES_PER_CELL_GROUP);
        return CUPHY_STATUS_NOT_SUPPORTED; // As this limit might change in the future, return a NOT SUPPORTED error
    }

    if(maxNCBsPerTB > MAX_N_CBS_PER_TB_SUPPORTED)
    {
        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "{}: Maximum number of Code Blocks per Transport Block supported is {}", __FUNCTION__, MAX_N_CBS_PER_TB_SUPPORTED);
        return CUPHY_STATUS_NOT_SUPPORTED;
    }

    if(maxTBByteSize > MAX_BYTES_PER_TRANSPORT_BLOCK)
    {
        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "{}: Transport Block size = {}. Maximum Transport Block size in bytes supported is {}", __FUNCTION__, maxTBByteSize, MAX_BYTES_PER_TRANSPORT_BLOCK);
        return CUPHY_STATUS_NOT_SUPPORTED;
    }

    // All 2 kernels will share the same descriptor as kernel argument
    crcEncodeLaunchConfig->m_kernelArgs[0]                    = cpu_desc;
    crcEncodeLaunchConfig->m_kernelNodeParams[0].extra        = nullptr;
    crcEncodeLaunchConfig->m_kernelNodeParams[0].kernelParams = &(crcEncodeLaunchConfig->m_kernelArgs[0]);

    crcEncodeLaunchConfig->m_kernelNodeParams[1].extra        = nullptr;
    crcEncodeLaunchConfig->m_kernelNodeParams[1].kernelParams = &(crcEncodeLaunchConfig->m_kernelArgs[0]);

    //Set up CPU descriptor. Assumes it has been pre-allocated.
    crcEncodeDescr_t& desc      = *(static_cast<crcEncodeDescr_t*>(cpu_desc));
    desc.d_cbCRCs               = d_cbCRCs;
    desc.d_tbCRCs               = d_tbCRCs;
    desc.d_inputTransportBlocks = d_inputTransportBlocks;
    desc.d_codeBlocks           = d_codeBlocks;
    desc.d_tbPrmsArray          = d_tbPrmsArray;
    desc.reverseBytes           = (reverseBytes == 1);

    // Optional descriptor copy to GPU memory
    // When running as part of a pipeline, it's better to do a single copy of all descriptors in the pipeline,
    // so the code below won't be exercised. The pipeline will also memset the TB-CRCs as needed.
    if(enable_desc_async_copy)
    {
        CUDA_CHECK(cudaMemcpyAsync(gpu_desc, cpu_desc, sizeof(crcEncodeDescr_t), cudaMemcpyHostToDevice, strm));
        if(!codeBlocksOnly)
        {
            CUDA_CHECK(cudaMemsetAsync(d_tbCRCs, 0, sizeof(uint32_t) * nTBs, strm));
        }
    }
    crcEncodeLaunchConfig->m_desc = static_cast<crcEncodeDescr_t*>(cpu_desc);

    const uint32_t blockSize   = crc::GLOBAL_BLOCK_SIZE;
    uint32_t       gridSizeCBX = maxNCBsPerTB;
    uint32_t       gridSizeCBY = nTBs;
    dim3           gCBSize(gridSizeCBX, gridSizeCBY);
    uint32_t       tbSize = (maxTBByteSize + sizeof(uint32_t) - 1) / sizeof(uint32_t);

    uint32_t gridSizeTBX = (tbSize + blockSize - 1) / blockSize;
    uint32_t gridSizeTBY = nTBs;
    dim3     gTBSize(gridSizeTBX, gridSizeTBY);

    uint32_t shmem_size = sizeof(uint32_t) * crc::WARP_SIZE;

    cudaFunction_t TB_device_function, CB_device_function;

    // When running as part of the PDSCH pipeline (!enable_desc_async_copy),
    // update the kernel functions only during the first call to cuphySetupCrcEncode. Doing so is OK as they are not templated and do not change.
    if(enable_desc_async_copy || (crcEncodeLaunchConfig->m_kernelNodeParams[0].func == nullptr) || (crcEncodeLaunchConfig->m_kernelNodeParams[1].func == nullptr))
    {
        {MemtraceDisableScope md; CUDA_CHECK(cudaGetFuncBySymbol(&TB_device_function, reinterpret_cast<void*>(crc::crcDownlinkPdschTransportBlockKernel)));}
        {MemtraceDisableScope md; CUDA_CHECK(cudaGetFuncBySymbol(&CB_device_function, reinterpret_cast<void*>(crc::crcDownlinkPdschCodeBlocksKernel)));}

        crcEncodeLaunchConfig->m_kernelNodeParams[0].func = TB_device_function;
        crcEncodeLaunchConfig->m_kernelNodeParams[1].func = CB_device_function;
    }

    // For crcDownlinkPdschTransportBlockKernel. Should be launched only if (!codeBlocksonly)
    crcEncodeLaunchConfig->m_kernelNodeParams[0].blockDimX      = blockSize;
    crcEncodeLaunchConfig->m_kernelNodeParams[0].blockDimY      = 1;
    crcEncodeLaunchConfig->m_kernelNodeParams[0].blockDimZ      = 1;
    crcEncodeLaunchConfig->m_kernelNodeParams[0].gridDimX       = gTBSize.x;
    crcEncodeLaunchConfig->m_kernelNodeParams[0].gridDimY       = gTBSize.y;
    crcEncodeLaunchConfig->m_kernelNodeParams[0].gridDimZ       = gTBSize.z;
    crcEncodeLaunchConfig->m_kernelNodeParams[0].sharedMemBytes = shmem_size;

    // For crcDownlinkPdschCodeBlocksKernel
    crcEncodeLaunchConfig->m_kernelNodeParams[1].blockDimX = blockSize;
    crcEncodeLaunchConfig->m_kernelNodeParams[1].blockDimY = 1;
    crcEncodeLaunchConfig->m_kernelNodeParams[1].blockDimZ = 1;
    crcEncodeLaunchConfig->m_kernelNodeParams[1].gridDimX  = gCBSize.x;
    crcEncodeLaunchConfig->m_kernelNodeParams[1].gridDimY  = gCBSize.y;
    crcEncodeLaunchConfig->m_kernelNodeParams[1].gridDimZ  = gCBSize.z;
    // Shared memory will contain two copies a of a code block with one that might not start at a 32-bit boundary, so 2 extra words are needed to store head and tail fragments
    //crcEncodeLaunchConfig->m_kernelNodeParams[1].sharedMemBytes = (crc::MAX_WORDS_PER_CODE_BLOCK + (crc::MAX_WORDS_PER_CODE_BLOCK + 2)) * sizeof(uint32_t);
    //Also add blockSize / WARP_SIZE elements to avoid syncthreads before xorReductionWarpShared.
    crcEncodeLaunchConfig->m_kernelNodeParams[1].sharedMemBytes = (crc::MAX_WORDS_PER_CODE_BLOCK + (crc::MAX_WORDS_PER_CODE_BLOCK + 2) + (blockSize / crc::WARP_SIZE)) * sizeof(uint32_t);

    return CUPHY_STATUS_SUCCESS;
}

cuphyStatus_t cuphyCRCDecode(
    /* DEVICE MEMORY*/
    uint32_t*          d_outputCBCRCs,
    uint32_t*          d_outputTBCRCs,
    uint8_t*           d_outputTransportBlocks,
    const uint32_t*    d_inputCodeBlocks,
    const PerTbParams* d_tbPrmsArray,
    /* END DEVICE MEMORY*/
    uint32_t     nTBs,
    uint32_t     maxNCBsPerTB,  // Maximum number of code blocks per transport block for current launch
    uint32_t     maxTBByteSize, // Maximum size in bytes of transport block for current launch
    int          reverseBytes,
    int          timeIt,
    uint32_t     NRUNS,
    uint32_t     codeBlocksOnly, // Only compute CRC of code blocks. Skip transport block CRC computation
    cudaStream_t strm)
{
    cuphyStatus_t status = crc::launch(
        d_outputCBCRCs,
        d_outputTBCRCs,
        d_outputTransportBlocks,
        d_inputCodeBlocks,
        d_tbPrmsArray,
        nTBs,
        maxNCBsPerTB,
        maxTBByteSize,
        reverseBytes,
        timeIt,
        NRUNS,
        codeBlocksOnly,
        strm);
    return status;
}

#if 1
//Support only BIT_FLIP
/* The prepare_crc_buffers kernel is part of PDSCH setup and it prepares the PDSCH input so it is in the format
   expected by the PDSCH CRC kernel(s): crcDownlinkPdschTransportBlockKernel (optional if TB-CRC provided) and crcDownlinkPdschCodeBlocksKernel.
   Currently the CRC kernel(s) expect the starting address of each transport block (TB) to be 32-bit aligned and to have some additional padding.
   They also require some bit manipulations. However, the PDSCH API only ensures that each TB is byte-aligned.

   There are two ways to do this processing without changing the cuPHY API or the CRC kernels:
   (a) perform H2D copies before calling the prepare_crc_buffers kernel and then have this kernel
   operate on (read) device memory directly or (b) have this kernel read host pinned memory directly.
   Option (a) needs multiple H2D copies when there are multiple cells in a cell group, as the TBs of each cell reside in separate buffers based on cuPHY API.
   Note that the number of H2D copies, which will be seralized, scales with the number of cells.

   Currently, pdsch_tx.cpp has a REVERT_TO_ASYNC_COPIES flag to determine if there will be H2D copies (set to 1, default) or not (set to 0).
   The kernel is exercised no matter the value of REVERT_TO_ASYNC_COPIES. If it's 1, then this kernel accesses device memory and does a form of D2D copy
   with extra processing. H2D copies, one per cell, are called before that. If it's 0, then this kernel accesses host pinned memory directly.

   Some preliminary PDSCH-only benchmarking had shown that avoiding the H2D copies was beneficial; the duration of the prepare_crc_buffers would "absorb"
   the H2D copies. However, when running PDSCH with PUSCH, H2D copies appeared to be better as the former resulted in dropped packets due to PCI-e pressure.

   The kernel has since been updated so each thread does one 32-bit read in the common case with some exceptions. The reduced number of requests
   can be beneficial irrespective the REVERT_TO_ASYNC_COPIES value.
   TODO: it is worth exploring whether setting REVERT_TO_ASYNC_COPIES to 0 after the kernel change performs better when PUSCH is also running.
*/
__global__ void prepare_crc_buffers(prepareCrcEncodeDescr_t* p_desc)
{
    prepareCrcEncodeDescr_t& desc = *p_desc;
    const uint32_t* __restrict__ d_inputOrigTBs = desc.d_inputOrigTBs;
    const PdschPerTbParams* __restrict__ cfg_workspace = desc.d_tbPrmsArray;

    uint32_t           TB_id     = blockIdx.y;
    const PdschPerTbParams* TB_params = &cfg_workspace[TB_id];
    uint32_t           tb_size   = TB_params->tbSize;
    uint32_t           tb_offset = TB_params->tbStartOffset; //might not be divisible by sizeof(uint32_t)
    const uint8_t* __restrict__    TB_addr   = (d_inputOrigTBs == nullptr) ? TB_params->tbStartAddr : ((uint8_t*)d_inputOrigTBs + tb_offset);
    const bool         not_in_testing_mode = (TB_params->testModel == 0);

    // For TBs in cells in testing mode, the input payload (bits from PN23 sequence) should be copied to a different output buffer,
    // the one used as input for the fused rate-matching and modulation kernel.
    uint32_t* d_inputTBs = not_in_testing_mode ? desc.d_inputTBs : desc.d_inputTBsTM;
    // The desc.offset array has valid values only for TM TBs
    uint32_t total_src_tb_offset = (TB_id == 0) ? 0 : (not_in_testing_mode ? cfg_workspace[TB_id-1].cumulativeTbSizePadding >> 2 : desc.offset[TB_id] >> 2);
    uint32_t tb_size_plus_padding_bytes =  TB_params->cumulativeTbSizePadding - ((TB_id == 0) ? 0 : cfg_workspace[TB_id-1].cumulativeTbSizePadding);
    uint32_t CB_element_offset = blockIdx.x * blockDim.x + threadIdx.x;

    int addr_modulo_4 = reinterpret_cast<uintptr_t>(TB_addr) & 0x3;
#if 0
    int  element_offset_general_case = (CB_element_offset < (tb_size >> 2)) ? 1 : 0;
#else
    //only relevant if addr_modulo_4 != 0. And there, there's a difference between cases where original tb_size % 4 == 0 and tb_size % 4 != 0.
    int  element_offset_general_case = (CB_element_offset < ((tb_size - addr_modulo_4 - (tb_size & 0x3))>> 2)) ? 1 : 0;
#endif
    int pred = 0;
    unsigned int match_mask = __match_all_sync(0xFFFFFFFF, element_offset_general_case, &pred);

    if ((match_mask != 0) && (element_offset_general_case == 1)) {
        if (addr_modulo_4 == 0) {
            const uint32_t* ptr = (uint32_t*)TB_addr + CB_element_offset;
            uint32_t temp_value = *ptr;
            uint32_t value = __byte_perm(temp_value, 0, 0x0123); //preferable
            //uint32_t value = ((temp_value >> 24) & 0xFF) | ((temp_value >> 8) & 0xFF00) | ((temp_value << 8) & 0xFF0000) | ((temp_value << 24) & 0xFF000000);
            d_inputTBs[total_src_tb_offset + CB_element_offset] = __brev(value);
        } else {
            const uint32_t* ptr = (uint32_t*)(TB_addr + (CB_element_offset << 2) - addr_modulo_4);
            uint32_t temp_value = *ptr;
            uint32_t new_value  = __shfl_down_sync(0xffffffff, temp_value, 1);
            if ((threadIdx.x % 32) == 31) {
                // The element_offset_general_case computation above has been updated to ensure all bytes of this uint32_t have been initialized.
                // Before this update, i.e., when code was: int  element_offset_general_case = (CB_element_offset < (tb_size >> 2)) ? 1 : 0;
                // the 32-bit value read below could contain uninitialized bytes which can trigger an initcheck error. Please note that these bytes were never actually used though.
                new_value = *(ptr + 1);
            }
            // perm_mask used should be 0x5670 or 0x6701 or 0x7012 for addr_modulo_4 equal to 1, 2 or 3 respectively.
            uint32_t perm_mask = (0x5670123 >> ((4 - addr_modulo_4)*4)) & 0xffff;
            uint32_t value = __byte_perm(new_value, temp_value, perm_mask);
            d_inputTBs[total_src_tb_offset + CB_element_offset] = __brev(value);
        }
    } else { // FIXME still byte reads in the first two clauses
        if(CB_element_offset < (tb_size >> 2)) // could be misaligned; still byte reads
        {
            const uint8_t* ptr = TB_addr + (CB_element_offset << 2);
            uint32_t value = 0;

            for(int byte_id = 0; byte_id < 4; byte_id++)
            {
                value |= (ptr[byte_id] << (3 - byte_id) * 8);
            }
            d_inputTBs[total_src_tb_offset + CB_element_offset] = __brev(value);
        }
        else if(CB_element_offset < div_round_up<uint32_t>(tb_size, 4)) //still byte reads
        {
            uint32_t value = 0;
            const uint8_t* ptr = TB_addr + (CB_element_offset << 2);
            for(int byte_id = 0; byte_id < (tb_size & 0x3U); byte_id++)
            {
                value |= (ptr[byte_id] << (3 - byte_id) * 8);
            }
            d_inputTBs[total_src_tb_offset + CB_element_offset] = __brev(value);
        }
        else if(CB_element_offset< tb_size_plus_padding_bytes >> 2) // just writes the padding; no reads
        {
            if (not_in_testing_mode) // No need for extra padding in TM as there won't be any CRC inserted
                d_inputTBs[total_src_tb_offset + CB_element_offset] = 0;
        }
    }
}
#else
__global__ void prepare_crc_buffers(const uint32_t* __restrict__ d_inputOrigTBs,
                                    uint32_t* d_inputTBs,
                                    const PdschPerTbParams* __restrict__ cfg_workspace)
{
    uint32_t           TB_id     = blockIdx.y;
    const PdschPerTbParams* TB_params = &cfg_workspace[TB_id];
    uint32_t           tb_size   = TB_params->tbSize;
    uint32_t           tb_offset = TB_params->tbStartOffset; //might not be divisible by sizeof(uint32_t)
    const uint8_t* __restrict__    TB_addr   = (d_inputOrigTBs == nullptr) ? TB_params->tbStartAddr : ((uint8_t*)d_inputOrigTBs + tb_offset);

    //temp. read value
    uint32_t total_src_tb_offset = 0;
    // todo read global offset
    for(int i = 0; i < TB_id; i++)
    {
        total_src_tb_offset += ((cfg_workspace[i].tbSize + cfg_workspace[i].paddingBytes) >> 2);
    }
    uint32_t CB_element_offset = blockIdx.x * blockDim.x + threadIdx.x;

    uint32_t value = 0;
#if BIT_FLIP
    if(CB_element_offset < (tb_size >> 2))
    {
        const uint8_t* ptr = TB_addr + (CB_element_offset << 2);
        for(int byte_id = 0; byte_id < 4; byte_id++)
        {
            value |= (ptr[byte_id] << (3 - byte_id) * 8);
        }
        d_inputTBs[total_src_tb_offset + CB_element_offset] = __brev(value);
    }
    else if(CB_element_offset < div_round_up<uint32_t>(tb_size, 4))
    {
        const uint8_t* ptr = TB_addr + (CB_element_offset << 2);
        for(int byte_id = 0; byte_id < (tb_size & 0x3U); byte_id++)
        {
            value |= (ptr[byte_id] << (3 - byte_id) * 8);
        }
        d_inputTBs[total_src_tb_offset + CB_element_offset] = __brev(value);
    }
    else if(CB_element_offset<(tb_size + TB_params->paddingBytes) >> 2)
    {
        d_inputTBs[total_src_tb_offset + CB_element_offset] = 0;
    }
#else
    if(CB_element_offset < (tb_size >> 2))
    {
        const uint8_t* ptr = TB_addr + (CB_element_offset << 2);
        for(int byte_id = 0; byte_id < 4; byte_id++)
        {
            value |= (ptr[byte_id] << byte_id * 8);
        }
        d_inputTBs[total_src_tb_offset + CB_element_offset] = value;
    }
    else if(CB_element_offset < div_round_up<uint32_t>(tb_size, 4))
    {
        const int8_t* ptr = TB_addr + (CB_element_offset << 2);
        for(int byte_id = 0; byte_id < (tb_size & 0x3U); byte_id++)
        {
            value |= (ptr[byte_id] << byte_id * 8);
        }
        d_inputTBs[total_src_tb_offset + CB_element_offset] = value;
    }
    else if(CB_element_offset<(tb_size + TB_params->paddingBytes)> > 2)
    {
        d_inputTBs[total_src_tb_offset + CB_element_offset] = 0;
    }
#endif
}
#endif

cuphyStatus_t cuphySetupPrepareCRCEncode(
    cuphyPrepareCrcEncodeLaunchConfig_t prepareCrcEncodeLaunchConfig,
    const uint32_t* d_inputOrigTBs, // Array containing input, if set to nullptr the d_tbPrmsArray->tbStartAddr (pinned host memory is used)
    uint32_t*       d_inputTBs,     // Array containing input after preparation
    uint32_t*       d_inputTBsTM,     // Array containing input after preparation for TBs of cells in testing mode

    const PdschPerTbParams* d_tbPrmsArray, // array of PerTbParams structs describing each input transport block
    uint32_t           nTBs,          // total number of input transport blocks
    uint32_t           maxNCBsPerTB,  // Maximum number of code blocks per transport block for current launch // FIXME not currently used
    uint32_t           maxTbSizeBytes,
    void*              cpu_desc,
    void*              gpu_desc,
    uint8_t            enable_desc_async_copy,
    cudaStream_t       strm)
{

    prepareCrcEncodeLaunchConfig->m_kernelArgs[0] = &(prepareCrcEncodeLaunchConfig->m_desc);
    prepareCrcEncodeLaunchConfig->m_kernelNodeParams.extra = nullptr;
    prepareCrcEncodeLaunchConfig->m_kernelNodeParams.kernelParams = &(prepareCrcEncodeLaunchConfig->m_kernelArgs[0]);

    //Set up CPU descriptor. Assumes it has been pre-allocated.
    prepareCrcEncodeDescr_t& desc = *(static_cast<prepareCrcEncodeDescr_t*>(cpu_desc));
    desc.d_inputOrigTBs = d_inputOrigTBs;
    desc.d_inputTBs     = d_inputTBs;
    desc.d_inputTBsTM   = d_inputTBsTM;
    desc.d_tbPrmsArray  = d_tbPrmsArray;

    // Optional descriptor copy to GPU memory
    // When running as part of a pipeline, it's better to do a single copy of all descriptors in the pipeline.
    if (enable_desc_async_copy) {
        cudaError_t res = cudaMemcpyAsync(gpu_desc, cpu_desc, sizeof(prepareCrcEncodeDescr_t), cudaMemcpyHostToDevice, strm);
        if (res != cudaSuccess) {
            return CUPHY_STATUS_MEMCPY_ERROR;
        }
    }

    prepareCrcEncodeLaunchConfig->m_desc = static_cast<prepareCrcEncodeDescr_t*>(gpu_desc);

    cudaFunction_t prepare_crc_kernel_device_function;
    {
    MemtraceDisableScope md;
    cudaError_t res = cudaGetFuncBySymbol(&prepare_crc_kernel_device_function, reinterpret_cast<void*>(prepare_crc_buffers));
    if (res != cudaSuccess) { // Currently cudaGetFuncBySymbol only returns cudaSuccess.
        return CUPHY_STATUS_INTERNAL_ERROR;
    }
    }
    prepareCrcEncodeLaunchConfig->m_kernelNodeParams.func = prepare_crc_kernel_device_function;

    const uint32_t threads         = 256;
    uint32_t       max_TB_elements = div_round_up<uint32_t>(maxTbSizeBytes, sizeof(uint32_t));
    prepareCrcEncodeLaunchConfig->m_kernelNodeParams.blockDimX = threads;
    prepareCrcEncodeLaunchConfig->m_kernelNodeParams.blockDimY = 1;
    prepareCrcEncodeLaunchConfig->m_kernelNodeParams.blockDimZ = 1;
    prepareCrcEncodeLaunchConfig->m_kernelNodeParams.gridDimX = div_round_up<uint32_t>(max_TB_elements, threads);
    prepareCrcEncodeLaunchConfig->m_kernelNodeParams.gridDimY = nTBs;
    prepareCrcEncodeLaunchConfig->m_kernelNodeParams.gridDimZ = 1;
    prepareCrcEncodeLaunchConfig->m_kernelNodeParams.sharedMemBytes = 0;

    cuphyStatus_t  status          = CUPHY_STATUS_SUCCESS;
#if 0
    const uint32_t threads         = 256;
    uint32_t       max_TB_elements = div_round_up<uint32_t>(maxTbSizeBytes, sizeof(uint32_t));
    dim3           num_thread_blocks(div_round_up<uint32_t>(max_TB_elements, threads), nTBs);

    prepare_crc_buffers<<<num_thread_blocks, threads, 0, strm>>>(d_inputOrigTBs, d_inputTBs, d_tbPrmsArray);

    if(cudaGetLastError() != cudaSuccess)
    {
        status = CUPHY_STATUS_INTERNAL_ERROR;
    }
#endif

    return status;
}

void puschRxCrcDecode::getDescrInfo(size_t& descrSizeBytes, size_t& descrAlignBytes)
{
    descrSizeBytes  = sizeof(puschRxCrcDecodeDescr_t);
    descrAlignBytes = alignof(puschRxCrcDecodeDescr_t);
}

void puschRxCrcDecode::init(int reverseBytes)
{
    m_reverseBytes = static_cast<bool>(reverseBytes);
    {MemtraceDisableScope md; CUDA_CHECK(cudaGetFuncBySymbol(&m_cbCrcKernelFunc, reinterpret_cast<void*>(crc::crcUplinkPuschCodeBlocksKernel)));}
    {MemtraceDisableScope md; CUDA_CHECK(cudaGetFuncBySymbol(&m_tbCrcKernelFunc, reinterpret_cast<void*>(crc::crcUplinkPuschTransportBlockKernel)));}
}

void puschRxCrcDecode::setup(uint16_t                          nSchUes,
                             uint16_t*                         pSchUserIdxsCpu,
                             uint32_t*                         pOutputCBCRCs,
                             uint8_t*                          pOutputTBs,
                             const uint32_t*                   pInputCodeBlocks,
                             uint32_t*                         pOutputTBCRCs,
                             const PerTbParams*                pTbPrmsCpu,
                             const PerTbParams*                pTbPrmsGpu,
                             void*                             pCpuDesc,
                             void*                             pGpuDesc,
                             uint8_t                           enableCpuToGpuDescrAsyncCpy,
                             cuphyPuschRxCrcDecodeLaunchCfg_t* pCbCrcLaunchCfg,
                             cuphyPuschRxCrcDecodeLaunchCfg_t* pTbCrcLaunchCfg,
                             cudaStream_t                      strm)
{
    // setup CPU descriptor
    puschRxCrcDecodeDescr_t& desc = *(static_cast<puschRxCrcDecodeDescr_t*>(pCpuDesc));

    desc.pOutputCBCRCs    = pOutputCBCRCs;
    desc.pOutputTBs       = pOutputTBs;
    desc.pInputCodeBlocks = pInputCodeBlocks;
    desc.pOutputTBCRCs    = pOutputTBCRCs;
    desc.pTbPrmsArray     = pTbPrmsGpu;
    desc.reverseBytes     = m_reverseBytes;
    for(int i = 0; i < nSchUes; ++i)
    {
        desc.schUserIdxs[i] = pSchUserIdxsCpu[i];
    }

    // Because the kernel is now grid constant, we do not need to copy the descriptor to GPU memory

    // Setup Launch Geometry
    uint32_t nTbs          = nSchUes;
    uint32_t maxNCBsPerTB  = 0;
    uint32_t maxTBByteSize = 0;

    for(int i = 0; i < nTbs; ++i)
    {
        uint16_t ueIdx = pSchUserIdxsCpu[i];
        maxNCBsPerTB  = pTbPrmsCpu[ueIdx].num_CBs > maxNCBsPerTB ? pTbPrmsCpu[ueIdx].num_CBs : maxNCBsPerTB;
        maxTBByteSize = pTbPrmsCpu[ueIdx].nDataBytes > maxTBByteSize ? pTbPrmsCpu[ueIdx].nDataBytes : maxTBByteSize;
    }

    uint32_t tbSize = (maxTBByteSize + sizeof(uint32_t) - 1) / sizeof(uint32_t);

    dim3 cbCrcGridDim(maxNCBsPerTB, nTbs, 1);
    dim3 cbCrcBlockDim(crc::GLOBAL_BLOCK_SIZE, 1, 1);

    // printf("CB CRC: grid (x y z) (%d %d %d), block (x y z) (%d %d %d)\n", cbCrcGridDim.x, cbCrcGridDim.y, cbCrcGridDim.z, cbCrcBlockDim.x, cbCrcBlockDim.y, cbCrcBlockDim.z);

    pCbCrcLaunchCfg->desc                                  = pGpuDesc;
    pCbCrcLaunchCfg->kernelArgs[0]                         = &desc;
    pCbCrcLaunchCfg->kernelNodeParamsDriver.gridDimX       = cbCrcGridDim.x;
    pCbCrcLaunchCfg->kernelNodeParamsDriver.gridDimY       = cbCrcGridDim.y;
    pCbCrcLaunchCfg->kernelNodeParamsDriver.gridDimZ       = cbCrcGridDim.z;
    pCbCrcLaunchCfg->kernelNodeParamsDriver.blockDimX      = cbCrcBlockDim.x;
    pCbCrcLaunchCfg->kernelNodeParamsDriver.blockDimY      = cbCrcBlockDim.y;
    pCbCrcLaunchCfg->kernelNodeParamsDriver.blockDimZ      = cbCrcBlockDim.z;
    pCbCrcLaunchCfg->kernelNodeParamsDriver.func           = m_cbCrcKernelFunc;
    pCbCrcLaunchCfg->kernelNodeParamsDriver.kernelParams   = &(pCbCrcLaunchCfg->kernelArgs[0]);
    pCbCrcLaunchCfg->kernelNodeParamsDriver.sharedMemBytes = sizeof(uint32_t) * crc::WARP_SIZE;
    pCbCrcLaunchCfg->kernelNodeParamsDriver.extra          = nullptr;

    dim3 tbCrcGridDim((tbSize + crc::GLOBAL_BLOCK_SIZE - 1) / crc::GLOBAL_BLOCK_SIZE, nTbs, 1);
    dim3 tbCrcBlockDim(crc::GLOBAL_BLOCK_SIZE, 1, 1);

    // printf("TB CRC: grid (x y z) (%d %d %d), block (x y z) (%d %d %d)\n", tbCrcGridDim.x, tbCrcGridDim.y, tbCrcGridDim.z, tbCrcBlockDim.x, tbCrcBlockDim.y, tbCrcBlockDim.z);

    pTbCrcLaunchCfg->desc                                  = pGpuDesc;
    pTbCrcLaunchCfg->kernelArgs[0]                         = &desc;
    pTbCrcLaunchCfg->kernelNodeParamsDriver.gridDimX       = tbCrcGridDim.x;
    pTbCrcLaunchCfg->kernelNodeParamsDriver.gridDimY       = tbCrcGridDim.y;
    pTbCrcLaunchCfg->kernelNodeParamsDriver.gridDimZ       = tbCrcGridDim.z;
    pTbCrcLaunchCfg->kernelNodeParamsDriver.blockDimX      = tbCrcBlockDim.x;
    pTbCrcLaunchCfg->kernelNodeParamsDriver.blockDimY      = tbCrcBlockDim.y;
    pTbCrcLaunchCfg->kernelNodeParamsDriver.blockDimZ      = tbCrcBlockDim.z;
    pTbCrcLaunchCfg->kernelNodeParamsDriver.func           = m_tbCrcKernelFunc;
    pTbCrcLaunchCfg->kernelNodeParamsDriver.kernelParams   = &(pTbCrcLaunchCfg->kernelArgs[0]);
    pTbCrcLaunchCfg->kernelNodeParamsDriver.sharedMemBytes = sizeof(uint32_t) * crc::WARP_SIZE;
    pTbCrcLaunchCfg->kernelNodeParamsDriver.extra          = nullptr;
}
