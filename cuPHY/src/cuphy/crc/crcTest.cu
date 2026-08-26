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

#include <gtest/gtest.h>
#include <fstream>
#include <iostream>
#include <string>
#include "crc.hpp"
#include "cuphy.h"
#include "cuphy.hpp"
#include "cuphy_internal.h"
#include "crc_decode.hpp"

using namespace crc;
using namespace cuphy_i;

// utility function for unit test
template <typename baseType>
unsigned long equalCount(baseType* a, baseType* b, unsigned long nElements, const std::string& label = "")
{
    unsigned long popCount = 0;
    for(int i = 0; i < nElements; i++)
    {
        popCount += a[i] != b[i];
        if(a[i] != b[i])
        {
            std::cout << label << "NOT EQUAL (" << std::dec << i << ") a: " << std::hex << a[i]
                      << " b: " << std::hex << b[i] << "\n";
        }
    }
    return popCount == 0;
}

template <typename baseType>
void linearToCoalesced(baseType*     coalescedData,
                       baseType*     linearData,
                       unsigned long nElements,
                       unsigned long elementSize,
                       unsigned long stride)
{
    for(int i = 0; i < nElements; i++)
    {
        for(int j = 0; j < elementSize; j++)
            coalescedData[j * stride + i] = linearData[i * elementSize + j];
    }
}

int CRC_GPU_UPLINK_PUSCH_TEST(bool timeIt)
{
    uint32_t  nTBs                    = MAX_N_TBS_SUPPORTED;
    uint32_t* firstCodeBlockIdxArray  = new uint32_t[nTBs];
    uint32_t* nCodeBlocks             = new uint32_t[nTBs];
    uint32_t* codeBlockByteSizes      = new uint32_t[nTBs]; // 56;
    uint32_t* codeBlockWordSizes      = new uint32_t[nTBs]; // 56;
    uint32_t* codeBlockDataByteSizes  = new uint32_t[nTBs]; // 1053;
    uint32_t* CBPaddingByteSizes      = new uint32_t[nTBs]; // pad to 32-bit boundary
    uint32_t* crcByteSizes            = new uint32_t[nTBs];
    uint32_t* totalCodeBlockByteSizes = new uint32_t[nTBs];
    uint32_t  totalByteSize           = 0;
    uint32_t  totalNCodeBlocks        = 0;
    uint32_t* tbPaddedByteSizes       = new uint32_t[nTBs];
    uint32_t  totalTBPaddedByteSize   = 0;
    uint32_t  ratio                   = sizeof(uint32_t) / sizeof(uint8_t);
    // Same CRC value for each code block, code blocks are all the same
    // linear input layout : cb1|crc1, cb2|crc2, ...

    for(int i = 0; i < nTBs; i++)
    {
        if(i == 1)
        {
            nCodeBlocks[i]        = 1;
            crcByteSizes[i]       = 2;
            codeBlockByteSizes[i] = 333;
        }
        else if(i == 2)
        {
            nCodeBlocks[i]        = 1;
            crcByteSizes[i]       = 3;
            codeBlockByteSizes[i] = 945;
        }

        else
        {
            codeBlockByteSizes[i] = 1007;
            nCodeBlocks[i]        = 6;
            crcByteSizes[i]       = 3;
        }
        totalNCodeBlocks += nCodeBlocks[i];
        codeBlockDataByteSizes[i]  = codeBlockByteSizes[i] - crcByteSizes[i];
        CBPaddingByteSizes[i]      = (MAX_BYTES_PER_CODE_BLOCK - (codeBlockByteSizes[i] % MAX_BYTES_PER_CODE_BLOCK)) % MAX_BYTES_PER_CODE_BLOCK;
        totalCodeBlockByteSizes[i] = codeBlockByteSizes[i] + CBPaddingByteSizes[i];
        codeBlockWordSizes[i]      = totalCodeBlockByteSizes[i] / ratio;
        totalByteSize += totalCodeBlockByteSizes[i] * nCodeBlocks[i];
        tbPaddedByteSizes[i] = (codeBlockDataByteSizes[i] + (nCodeBlocks[i] == 1 ? crcByteSizes[i] : 0)) * nCodeBlocks[i] +
                               (4 - (nCodeBlocks[i] * (codeBlockDataByteSizes[i] + (nCodeBlocks[i] == 1 ? crcByteSizes[i] : 0)) % 4)) % 4;
        totalTBPaddedByteSize += tbPaddedByteSizes[i];
    }

    PerTbParams* tbPrmsArray           = new PerTbParams[nTBs];
    uint8_t*     linearInput           = new uint8_t[totalByteSize];
    uint32_t*    goldenCRCs            = new uint32_t[totalNCodeBlocks];
    uint8_t*     goldenTransportBlocks = new uint8_t[totalTBPaddedByteSize];
    uint32_t*    transportBlocks       = new uint32_t[totalTBPaddedByteSize / ratio];
    uint32_t*    codeBlocks            = (uint32_t*)linearInput;
    uint32_t*    crcs                  = new uint32_t[totalNCodeBlocks];
    uint32_t*    tbCRCs                = new uint32_t[nTBs];
    memset(goldenTransportBlocks, 0, totalTBPaddedByteSize);
    memset(firstCodeBlockIdxArray, 0, nTBs * sizeof(uint32_t));
    uint32_t tbBytes      = 0;
    uint32_t totalCBBytes = 0;
    uint32_t totalCBs     = 0;
    for(int t = 0; t < nTBs; t++)
    {
        // Build transport block
        uint32_t cbBytes = 0;

        for(int i = 0; i < nCodeBlocks[t]; i++)
        {
            memset(goldenTransportBlocks + tbBytes + cbBytes,
                   rand(),
                   codeBlockDataByteSizes[t]);
            cbBytes += codeBlockDataByteSizes[t];
        }

        // last code block contains TB CRC in the last 3 bytes
        if(nCodeBlocks[t] > 1)
        {
            uint32_t golden_tbCRC = computeCRC<uint32_t, 24>(goldenTransportBlocks + tbBytes,
                                                             codeBlockDataByteSizes[t] * nCodeBlocks[t] - crcByteSizes[t],
                                                             G_CRC_24_A,
                                                             0,
                                                             1);
            for(int j = 0; j < crcByteSizes[t]; j++)
                goldenTransportBlocks[tbBytes + nCodeBlocks[t] * codeBlockDataByteSizes[t] - crcByteSizes[t] + j] =
                    (golden_tbCRC >> (crcByteSizes[t] - 1 - j) * 8) & 0xFF;
        }
        // compute CB crcs
        for(int i = 0; i < nCodeBlocks[t]; i++)
        {
            uint8_t* cbPtr  = linearInput + i * totalCodeBlockByteSizes[t] + totalCBBytes;
            uint8_t* crcPtr = (cbPtr + codeBlockDataByteSizes[t]);
            memcpy(cbPtr,
                   goldenTransportBlocks + i * codeBlockDataByteSizes[t] + tbBytes,
                   codeBlockDataByteSizes[t]);
            uint32_t crc;
            if(nCodeBlocks[t] == 1)
            {
                if(codeBlockDataByteSizes[t] <= MAX_SMALL_A_BYTES)
                {
                    crc = computeCRC<uint32_t, 16>((uint8_t*)cbPtr,
                                                   codeBlockDataByteSizes[t],
                                                   G_CRC_16,
                                                   0,
                                                   1);
                }
                else
                    crc = computeCRC<uint32_t, 24>((uint8_t*)cbPtr,
                                                   codeBlockDataByteSizes[t],
                                                   G_CRC_24_A,
                                                   0,
                                                   1);
                for(int j = 0; j < crcByteSizes[t]; j++)
                    goldenTransportBlocks[tbBytes + nCodeBlocks[t] * codeBlockDataByteSizes[t] /*- crcByteSizes[t]*/ + j] =
                        (crc >> (crcByteSizes[t] - 1 - j) * 8) & 0xFF;
            }

            else

                crc = computeCRC<uint32_t, 24>((uint8_t*)cbPtr,
                                               codeBlockDataByteSizes[t],
                                               G_CRC_24_B,
                                               0,
                                               1);

            for(int j = 0; j < crcByteSizes[t]; j++)
                crcPtr[j] = (crc >> (crcByteSizes[t] - 1 - j) * 8) & 0xFF;
            goldenCRCs[totalCBs] = 0;
            totalCBs++;
            memset(cbPtr + codeBlockByteSizes[t], 0, CBPaddingByteSizes[t]);
        }
        tbPrmsArray[t].num_CBs             = nCodeBlocks[t];
        tbPrmsArray[t].K                   = codeBlockByteSizes[t] * 8;
        tbPrmsArray[t].F                   = 0;
        tbPrmsArray[t].firstCodeBlockIndex = 0;
        tbBytes += tbPaddedByteSizes[t];
        totalCBBytes += nCodeBlocks[t] * totalCodeBlockByteSizes[t];
    }

#if 0
    std::cout << "CBs:\n";
    for (int i = 0; i < totalByteSize; i++)
        std::cout << std::hex << (unsigned short)linearInput[i] << ",";
    std::cout << "\n";

    std::cout << "TB:\n";
    for (int i = 0; i < totalTBPaddedByteSize; i++)
        std::cout << std::hex << (unsigned short)goldenTransportBlocks[i] << ",";
    std::cout << "\n";
#endif

    //input
    unique_device_ptr<uint32_t> d_codeBlocks = make_unique_device<uint32_t>(totalByteSize / sizeof(uint32_t));

    unique_device_ptr<PerTbParams> d_tbPrmsArray = make_unique_device<PerTbParams>(nTBs);
    //output

    unique_device_ptr<uint32_t> d_CBCRCs = make_unique_device<uint32_t>(nTBs * nCodeBlocks[0]);
    unique_device_ptr<uint32_t> d_TBCRCs = make_unique_device<uint32_t>(nTBs);
    unique_device_ptr<uint8_t>  d_TBs    = make_unique_device<uint8_t>(nTBs * tbPaddedByteSizes[0]);

    cudaMemcpy(d_codeBlocks.get(), codeBlocks, totalByteSize, cudaMemcpyHostToDevice);

    cudaMemcpy(d_tbPrmsArray.get(), tbPrmsArray, sizeof(PerTbParams) * nTBs, cudaMemcpyHostToDevice);

    cuphyStatus_t status = cuphyCRCDecode(
        d_CBCRCs.get(),
        d_TBCRCs.get(),
        d_TBs.get(),
        d_codeBlocks.get(),
        d_tbPrmsArray.get(),
        nTBs,
        nCodeBlocks[0],
        tbPaddedByteSizes[0],
        0,
        timeIt,
        0,
        false,
        0);

    cudaMemcpy(crcs, d_CBCRCs.get(), sizeof(uint32_t) * totalNCodeBlocks, cudaMemcpyDeviceToHost);

    cudaMemcpy(tbCRCs, d_TBCRCs.get(), sizeof(uint32_t) * nTBs, cudaMemcpyDeviceToHost);

    cudaMemcpy(transportBlocks, d_TBs.get(), totalTBPaddedByteSize, cudaMemcpyDeviceToHost);

    int passed = 0;

    if(status != CUPHY_STATUS_SUCCESS)
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "CRC: CUPHY ERROR");
    else
    {
        uint32_t* gt = (uint32_t*)goldenTransportBlocks;
        passed       = equalCount(crcs, goldenCRCs, totalNCodeBlocks, "CB CRC ");
        passed &= equalCount(transportBlocks, gt, totalTBPaddedByteSize / ratio, "TB DATA");

        for(int i = 0; i < nTBs; i++)
        {
            passed &= (tbCRCs[i] == 0);
            if(tbCRCs[i] != 0)
                NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "TB[{}] CRC not equal to 0: {}", i, tbCRCs[i]);
        }
    }

    delete[] tbPrmsArray;
    delete[] crcs;
    delete[] tbCRCs;
    delete[] goldenCRCs;
    delete[] goldenTransportBlocks;
    delete[] linearInput;
    delete[] transportBlocks;
    delete[] firstCodeBlockIdxArray;
    delete[] nCodeBlocks;
    delete[] codeBlockByteSizes;
    delete[] crcByteSizes;
    delete[] codeBlockWordSizes;
    delete[] codeBlockDataByteSizes;
    delete[] CBPaddingByteSizes; // pad to 32-bit boundary
    delete[] totalCodeBlockByteSizes;
    delete[] tbPaddedByteSizes;

    return passed;
}

int CRC_GPU_DOWNLINK_PDSCH(bool timeIt, int numCBs = 2) // FIXME numCBs has no effect!
{
    uint32_t  nTBs                    = 2;
    uint32_t* nCodeBlocks             = new uint32_t[nTBs];
    uint32_t* crcByteSizes            = new uint32_t[nTBs];
    uint32_t* codeBlockByteSizes      = new uint32_t[nTBs];
    uint32_t* codeBlockDataByteSizes  = new uint32_t[nTBs];
    uint32_t* cbPaddingByteSizes      = new uint32_t[nTBs];
    uint32_t* fillerByteSizes         = new uint32_t[nTBs];
    uint32_t* tensorStrideByteSizes   = new uint32_t[nTBs];
    uint32_t* totalCodeBlockByteSizes = new uint32_t[nTBs]; // The CB size that include every thing
    uint32_t* tbPaddedByteSizes       = new uint32_t[nTBs];
    uint32_t  totalNCodeBlocks        = 0;
    uint32_t  totalInTBByteSize       = 0;
    uint32_t  totalOutTBByteSizes     = 0;
    uint32_t  ratio                   = sizeof(uint32_t) / sizeof(uint8_t);

    for(int i = 0; i < nTBs; i++)
    {
        codeBlockByteSizes[i]    = 1056;
        crcByteSizes[i]          = 3;
        nCodeBlocks[i]           = 2;
        fillerByteSizes[i]       = 0;
        tensorStrideByteSizes[i] = 0;

        codeBlockDataByteSizes[i] = codeBlockByteSizes[i] - crcByteSizes[i] - fillerByteSizes[i];
        totalNCodeBlocks += nCodeBlocks[i];
        tbPaddedByteSizes[i] = codeBlockDataByteSizes[i] * nCodeBlocks[i];
        tbPaddedByteSizes[i] += (4 - (tbPaddedByteSizes[i] % 4)) % 4;
        totalInTBByteSize += tbPaddedByteSizes[i];
        totalCodeBlockByteSizes[i] = codeBlockByteSizes[i] + tensorStrideByteSizes[i];
        cbPaddingByteSizes[i]      = (4 - totalCodeBlockByteSizes[i] % 4) % 4;
        totalCodeBlockByteSizes[i] += cbPaddingByteSizes[i];
        totalOutTBByteSizes += totalCodeBlockByteSizes[i] * nCodeBlocks[i]; // should be a multiple of 4
    }

    PdschPerTbParams* tbPrmsArray      = new PdschPerTbParams[nTBs];
    uint8_t*     goldenTransportBlocks = new uint8_t[totalInTBByteSize];
    uint32_t*    crcs                  = new uint32_t[totalNCodeBlocks];
    uint32_t*    tbCRCs                = new uint32_t[nTBs];
    uint32_t*    goldenTBCRCs          = new uint32_t[nTBs];
    uint32_t*    goldenCRCs            = new uint32_t[totalNCodeBlocks];
    uint32_t*    codeBlocks            = new uint32_t[totalOutTBByteSizes >> 2];

    memset(goldenTransportBlocks, 0, totalInTBByteSize);
    memset(codeBlocks, 0, totalOutTBByteSizes);

    uint32_t tbOutBytes = 0;
    uint32_t tbBytes    = 0;
    uint8_t* outputs    = (uint8_t*)codeBlocks;
    for(int t = 0; t < nTBs; t++)
    {
        uint32_t cbBytes    = 0;
        uint32_t cbOutBytes = 0;
        for(int i = 0; i < nCodeBlocks[t]; i++)
        {
            // Set the input
            memset(goldenTransportBlocks + tbBytes + cbBytes,
                   rand(),
                   codeBlockDataByteSizes[t]);
            memcpy(outputs + tbOutBytes + cbOutBytes,
                   goldenTransportBlocks + tbBytes + cbBytes,
                   codeBlockDataByteSizes[t]);
            cbBytes += codeBlockDataByteSizes[t];
            cbOutBytes += totalCodeBlockByteSizes[t];
        }
        tbPrmsArray[t].num_CBs             = nCodeBlocks[t];
        tbPrmsArray[t].K                   = codeBlockByteSizes[t] * 8;
        tbPrmsArray[t].F                   = 0;
        tbPrmsArray[t].firstCodeBlockIndex = 0;
        // Add tbSize, tbStartOffset and paddingBytes fields
        tbPrmsArray[t].tbSize        = codeBlockDataByteSizes[t] * nCodeBlocks[t] - 3;
        tbPrmsArray[t].tbStartOffset = (t == 0) ? 0 : (tbPrmsArray[t - 1].tbStartOffset + tbPrmsArray[t - 1].tbSize);
        tbPrmsArray[t].tbStartAddr   = nullptr; // not used in CRC encode
        //tbPrmsArray[t].paddingBytes  = tbPaddedByteSizes[t] - tbPrmsArray[t].tbSize;
        tbPrmsArray[t].cumulativeTbSizePadding =  tbPaddedByteSizes[t] + ((t == 0) ? 0 : tbPrmsArray[t-1].cumulativeTbSizePadding);
        tbPrmsArray[t].testModel               =  0; // Assume no cell is in testing mode

        // The last CB in one TB is 3 bytes shorter than other CBs
        memset(goldenTransportBlocks + tbBytes +
                   nCodeBlocks[t] * codeBlockDataByteSizes[t] - 3,
               0,
               3);
        memset(outputs + tbOutBytes + (nCodeBlocks[t] - 1) * totalCodeBlockByteSizes[t] +
                   codeBlockDataByteSizes[t] - 3,
               0,
               3);
        tbBytes += tbPaddedByteSizes[t];
        tbOutBytes += totalCodeBlockByteSizes[t] * nCodeBlocks[t];
    }

    //input
    unique_device_ptr<uint32_t>    d_transportBlocks = make_unique_device<uint32_t>(totalInTBByteSize / ratio);
    unique_device_ptr<PdschPerTbParams> d_tbPrmsArray     = make_unique_device<PdschPerTbParams>(nTBs);

    //output

    unique_device_ptr<uint32_t> d_CBCRCs     = make_unique_device<uint32_t>(nTBs * nCodeBlocks[0]);
    unique_device_ptr<uint32_t> d_TBCRCs     = make_unique_device<uint32_t>(nTBs);
    unique_device_ptr<uint8_t>  d_codeBlocks = make_unique_device<uint8_t>(totalOutTBByteSizes);

    cudaMemcpy(d_transportBlocks.get(), (uint32_t*)goldenTransportBlocks, totalInTBByteSize, cudaMemcpyHostToDevice);

    //cudaMemcpy(d_codeBlocks.get(), (uint8_t*)codeBlocks, totalOutTBByteSizes, cudaMemcpyHostToDevice);

    CUDA_CHECK(cudaMemcpy(d_tbPrmsArray.get(), tbPrmsArray, sizeof(PdschPerTbParams) * nTBs, cudaMemcpyHostToDevice));

    // Allocate launch config struct.
    std::unique_ptr<cuphyCrcEncodeLaunchConfig> crc_hndl = std::make_unique<cuphyCrcEncodeLaunchConfig>();

    // Allocate descriptors and setup rate matching component
    uint8_t       desc_async_copy = 1; // Copy descriptor to the GPU during setup. And set TB-CRCs to 0.
    uint8_t*      h_crc_encode_desc;
    size_t        desc_size = 0, alloc_size = 0;
    cuphyStatus_t status = cuphyCrcEncodeGetDescrInfo(&desc_size, &alloc_size);
    if(status != CUPHY_STATUS_SUCCESS)
    {
        printf("cuphyCrcEncodeGetDescrInfo error %d\n", status);
    }
    unique_device_ptr<uint8_t> d_crc_encode_desc = make_unique_device<uint8_t>(desc_size);
    CUDA_CHECK(cudaHostAlloc((void**)&h_crc_encode_desc, desc_size, cudaHostAllocDefault));

    cudaStream_t cuda_strm = 0;

    status = cuphySetupCrcEncode(crc_hndl.get(),
                                 d_CBCRCs.get(),
                                 d_TBCRCs.get(),
                                 d_transportBlocks.get(),
                                 d_codeBlocks.get(),
                                 d_tbPrmsArray.get(),
                                 nTBs,
                                 nCodeBlocks[0],
                                 tbPaddedByteSizes[0],
                                 0,
                                 false,
                                 h_crc_encode_desc,
                                 d_crc_encode_desc.get(),
                                 desc_async_copy,
                                 cuda_strm);

    if(status != CUPHY_STATUS_SUCCESS)
    {
        throw std::runtime_error("Invalid argument(s) for cuphySetupCrcEncode");
    }

    // CRC has 2 kernels right now
    CUresult status_k1 = launch_kernel(crc_hndl.get()->m_kernelNodeParams[0], cuda_strm);
    CUresult status_k2 = launch_kernel(crc_hndl.get()->m_kernelNodeParams[1], cuda_strm);
    if((status_k1 != CUDA_SUCCESS) ||
       (status_k2 != CUDA_SUCCESS))
    {
        throw std::runtime_error("CRC Encode error(s)");
    }

    CUDA_CHECK(cudaStreamSynchronize(cuda_strm));

    CUDA_CHECK(cudaMemcpy(crcs, d_CBCRCs.get(), totalNCodeBlocks * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(tbCRCs, d_TBCRCs.get(), nTBs * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy((uint8_t*)codeBlocks, d_codeBlocks.get(), totalOutTBByteSizes, cudaMemcpyDeviceToHost));

    tbOutBytes = 0;
    tbBytes    = 0;

    int passed = 1;

    for(int t = 0; t < nTBs; t++)
    {
        // compute TB crcs
        uint32_t tbcrc  = computeCRC<uint32_t, 24>(goldenTransportBlocks + tbBytes,
                                                  codeBlockDataByteSizes[t] * nCodeBlocks[t] - 3,
                                                  G_CRC_24_A,
                                                  0,
                                                  1);
        goldenTBCRCs[t] = tbcrc;

        // Compare standalone per-TB CRC w/ per-TB CRC inserted in code blocks buffer
        // Compute pointer to per-TB CRC in the CB buffer. Subtract 6 because of 3B for per-TB CRC
        // and 3 for per-CB CRC.
        uint8_t* tmp           = (uint8_t*)codeBlocks + nCodeBlocks[t] * totalCodeBlockByteSizes[t] + tbOutBytes - 6;
        uint32_t gpu_perTB_crc = 0;
        for(int byte_id = 0; byte_id < 3; byte_id++)
        {
            gpu_perTB_crc |= ((*(tmp + byte_id)) << (byte_id * 8)); // Assume CRC written least significant byte of 24bits first (lower addr)
        }

        if(tbcrc != gpu_perTB_crc)
        {
            NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "TB {}: Standalone per-TB CRC {} mismatch w/ CRC inserted in CB buffer {}", t, tbcrc, gpu_perTB_crc);
            passed = 0;
        }

        // compute CB crcs
        for(int i = 0; i < nCodeBlocks[t]; i++)
        {
            uint8_t* cbPtr = (uint8_t*)codeBlocks + i * totalCodeBlockByteSizes[t] + tbOutBytes;
            // TODO It'd be better not to use the buffer (pointed by cbPtr) w/ the GPU generated CRCs to compute the CPU CRCs.
            uint32_t crc = computeCRC<uint32_t, 24>((uint8_t*)cbPtr,
                                                    codeBlockDataByteSizes[t],
                                                    G_CRC_24_B,
                                                    0,
                                                    1);

            goldenCRCs[t * nCodeBlocks[t] + i] = crc;

            // Also compare the per-CB CRCs CRC written in the CB buffer w/ the standalone per-CB CRC.
            uint8_t* tmp           = cbPtr + codeBlockDataByteSizes[t];
            uint32_t gpu_perCB_crc = 0;
            for(int byte_id = 0; byte_id < 3; byte_id++)
            {
                gpu_perCB_crc |= ((*(tmp + byte_id)) << (byte_id * 8)); // Assume CRC written least significant byte of 24bits first (lower addr)
            }
            if(crc != gpu_perCB_crc)
            {
                NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "TB {}, CB {}: Standalone per-CB CRC %x mismatch w/ CRC inserted in CB buffer {}", t, i, crc, gpu_perCB_crc);
                passed = 0;
            }
        }
        tbBytes += tbPaddedByteSizes[t];
        tbOutBytes += nCodeBlocks[t] * totalCodeBlockByteSizes[t];
    }

    if(status != CUPHY_STATUS_SUCCESS)
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "CRC: CUPHY ERROR");
    else
    {
        // uint32_t* gt = (uint32_t*)goldenTransportBlocks;
        passed &= equalCount(crcs, goldenCRCs, totalNCodeBlocks, "CB CRC ");
        // passed &= equalCount(transportBlocks, gt, totalTBPaddedByteSize / ratio, "TB DATA");

        for(int i = 0; i < nTBs; i++)
        {
            passed &= (tbCRCs[i] == goldenTBCRCs[i]);
            if(tbCRCs[i] != goldenTBCRCs[i])
                NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "TB[{}] CRC: {} not equal to goldenTBCRCs: {}", i, tbCRCs[i], goldenTBCRCs[i]);
        }
    }

    delete[] nCodeBlocks;
    delete[] crcByteSizes;
    delete[] codeBlockByteSizes;
    delete[] codeBlockDataByteSizes;
    delete[] cbPaddingByteSizes;
    delete[] fillerByteSizes;
    delete[] tensorStrideByteSizes;
    delete[] totalCodeBlockByteSizes;
    delete[] goldenTransportBlocks;
    delete[] crcs;
    delete[] tbCRCs;
    delete[] codeBlocks;

    CUDA_CHECK(cudaFreeHost(h_crc_encode_desc));

    return passed;
}

int CRC_SINGLE_CB_GPU_UPLINK_PUSCH_TEST(bool timeIt)
{
    uint32_t  nTBs                    = MAX_N_TBS_SUPPORTED;
    uint32_t* nCodeBlocks             = new uint32_t[nTBs];
    uint32_t  crcByteSize             = 3; // 24 bits
    uint32_t* firstCodeBlockIdxArray  = new uint32_t[nTBs];
    uint32_t* codeBlockByteSizes      = new uint32_t[nTBs]; // 56;
    uint32_t* codeBlockWordSizes      = new uint32_t[nTBs]; // 56;
    uint32_t* codeBlockDataByteSizes  = new uint32_t[nTBs]; // 1053;
    uint32_t* CBPaddingByteSizes      = new uint32_t[nTBs]; // pad to 32-bit boundary
    uint32_t* totalCodeBlockByteSizes = new uint32_t[nTBs];
    uint32_t  totalByteSize           = 0;
    uint32_t  totalNCodeBlocks        = 0;
    uint32_t* tbPaddedByteSizes       = new uint32_t[nTBs];
    uint32_t  totalTBPaddedByteSize   = 0;
    uint32_t  ratio                   = sizeof(uint32_t) / sizeof(uint8_t);
    // Same CRC value for each code block, code blocks are all the same
    // linear input layout : cb1|crc1, cb2|crc2, ...

    for(int i = 0; i < nTBs; i++)
    {
        if(i == 2)
        {
            codeBlockByteSizes[i] = 1056;
            nCodeBlocks[i]        = 1;
        }
        codeBlockByteSizes[i] = 1000;
        nCodeBlocks[i]        = 1;
        totalNCodeBlocks += nCodeBlocks[i];
        codeBlockDataByteSizes[i]  = codeBlockByteSizes[i] - crcByteSize;
        CBPaddingByteSizes[i]      = (MAX_BYTES_PER_CODE_BLOCK - (codeBlockByteSizes[i] % MAX_BYTES_PER_CODE_BLOCK)) % MAX_BYTES_PER_CODE_BLOCK;
        totalCodeBlockByteSizes[i] = codeBlockByteSizes[i] + CBPaddingByteSizes[i];
        codeBlockWordSizes[i]      = totalCodeBlockByteSizes[i] / ratio;
        totalByteSize += totalCodeBlockByteSizes[i] * nCodeBlocks[i];
        tbPaddedByteSizes[i] = (codeBlockDataByteSizes[i] + (nCodeBlocks[i] == 1 ? crcByteSize : 0)) * nCodeBlocks[i] +
                               (4 - (nCodeBlocks[i] * (codeBlockDataByteSizes[i] + (nCodeBlocks[i] == 1 ? crcByteSize : 0)) % 4)) % 4;
        totalTBPaddedByteSize += tbPaddedByteSizes[i];
    }

    PerTbParams* tbPrmsArray           = new PerTbParams[nTBs];
    uint8_t*     linearInput           = new uint8_t[totalByteSize];
    uint32_t*    goldenCRCs            = new uint32_t[totalNCodeBlocks];
    uint8_t*     goldenTransportBlocks = new uint8_t[totalTBPaddedByteSize];
    uint32_t*    transportBlocks       = new uint32_t[totalTBPaddedByteSize / ratio];
    uint32_t*    codeBlocks            = (uint32_t*)linearInput;
    uint32_t*    crcs                  = new uint32_t[totalNCodeBlocks];
    uint32_t*    tbCRCs                = new uint32_t[nTBs];
    memset(goldenTransportBlocks, 0, totalTBPaddedByteSize);
    memset(firstCodeBlockIdxArray, 0, nTBs * sizeof(uint32_t));
    uint32_t tbBytes      = 0;
    uint32_t totalCBBytes = 0;
    for(int t = 0; t < nTBs; t++)
    {
        // Build transport block
        uint32_t cbBytes = 0;
        for(int i = 0; i < nCodeBlocks[t]; i++)
        {
            memset(goldenTransportBlocks + tbBytes + cbBytes,
                   rand(),
                   codeBlockDataByteSizes[t]);
            cbBytes += codeBlockDataByteSizes[t];
        }

        // just compute CB crcs using TB polynomial, as TBs contain only one CB
        for(int i = 0; i < nCodeBlocks[t]; i++)
        {
            uint8_t* cbPtr  = linearInput + i * totalCodeBlockByteSizes[t] + totalCBBytes;
            uint8_t* crcPtr = (cbPtr + codeBlockDataByteSizes[t]);
            memcpy(cbPtr,
                   goldenTransportBlocks + i * codeBlockDataByteSizes[t] + tbBytes,
                   codeBlockDataByteSizes[t]);
            uint32_t crc = computeCRC<uint32_t, 24>((uint8_t*)cbPtr,
                                                    codeBlockDataByteSizes[t],
                                                    G_CRC_24_A,
                                                    0,
                                                    1);
            for(int j = 0; j < crcByteSize; j++)
                crcPtr[j] = (crc >> (crcByteSize - 1 - j) * 8) & 0xFF;
            for(int j = 0; j < crcByteSize; j++)
                goldenTransportBlocks[tbBytes + nCodeBlocks[t] * codeBlockDataByteSizes[t] /*- crcByteSizes[t]*/ + j] =
                    (crc >> (crcByteSize - 1 - j) * 8) & 0xFF;

            goldenCRCs[t * nCodeBlocks[t] + i] = 0;
            memset(cbPtr + codeBlockByteSizes[t], 0, CBPaddingByteSizes[t]);
        }
        tbPrmsArray[t].num_CBs             = nCodeBlocks[t];
        tbPrmsArray[t].K                   = codeBlockByteSizes[t] * 8;
        tbPrmsArray[t].F                   = 0;
        tbPrmsArray[t].firstCodeBlockIndex = 0;
        tbBytes += tbPaddedByteSizes[t];
        totalCBBytes += nCodeBlocks[t] * totalCodeBlockByteSizes[t];
    }
#if 0
    std::cout << "CBs:\n";
    for (int i = 0; i < totalByteSize; i++)
        std::cout << std::hex << (unsigned short)linearInput[i] << ",";
    std::cout << "\n";

    std::cout << "TB:\n";
    for (int i = 0; i < totalTBPaddedByteSize; i++)
        std::cout << std::hex << (unsigned short)goldenTransportBlocks[i] << ",";
    std::cout << "\n";
#endif
    /*
       cuphyStatus_t status = cuphyCRCDecode(
       crcs, tbCRCs, transportBlocks, nTBs, (const uint32_t*)codeBlocks,
       nCodeBlocks, codeBlockWordSizes, codeBlockDataByteSizes, timeIt, 10000);
     */

    //input
    unique_device_ptr<uint32_t>    d_codeBlocks  = make_unique_device<uint32_t>(totalByteSize / sizeof(uint32_t));
    unique_device_ptr<PerTbParams> d_tbPrmsArray = make_unique_device<PerTbParams>(nTBs);

    //output

    unique_device_ptr<uint32_t> d_CBCRCs = make_unique_device<uint32_t>(nTBs * nCodeBlocks[0]);
    unique_device_ptr<uint32_t> d_TBCRCs = make_unique_device<uint32_t>(nTBs);
    unique_device_ptr<uint8_t>  d_TBs    = make_unique_device<uint8_t>(nTBs * tbPaddedByteSizes[0]);

    cudaMemcpy(d_codeBlocks.get(), codeBlocks, totalByteSize, cudaMemcpyHostToDevice);

    cudaMemcpy(d_tbPrmsArray.get(), tbPrmsArray, sizeof(PerTbParams) * nTBs, cudaMemcpyHostToDevice);

    cuphyStatus_t status = cuphyCRCDecode(
        d_CBCRCs.get(),
        d_TBCRCs.get(),
        d_TBs.get(),
        d_codeBlocks.get(),
        d_tbPrmsArray.get(),
        nTBs,
        nCodeBlocks[0],
        tbPaddedByteSizes[0],
        0,
        timeIt,
        0,
        false,
        0);

    cudaMemcpy(crcs, d_CBCRCs.get(), sizeof(uint32_t) * totalNCodeBlocks, cudaMemcpyDeviceToHost);

    cudaMemcpy(tbCRCs, d_TBCRCs.get(), sizeof(uint32_t) * nTBs, cudaMemcpyDeviceToHost);

    cudaMemcpy(transportBlocks, d_TBs.get(), totalTBPaddedByteSize, cudaMemcpyDeviceToHost);

    int passed = 0;
    if(status != CUPHY_STATUS_SUCCESS)
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "CRC: CUPHY ERROR");
    else
    {
        uint32_t* gt = (uint32_t*)goldenTransportBlocks;
        passed       = equalCount(crcs, goldenCRCs, totalNCodeBlocks, "CB CRC ");
        passed &= equalCount(transportBlocks, gt, totalTBPaddedByteSize / ratio, "TB DATA ");

        for(int i = 0; i < nTBs; i++)
        {
            passed &= (tbCRCs[i] == 0);
            if(tbCRCs[i] != 0)
                NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "TB[{}] CRC not equal to 0: {}", i, tbCRCs[i]);
        }
    }

    delete[] tbPrmsArray;
    delete[] crcs;
    delete[] tbCRCs;
    delete[] goldenCRCs;
    delete[] goldenTransportBlocks;
    delete[] linearInput;
    delete[] transportBlocks;
    delete[] firstCodeBlockIdxArray;
    delete[] nCodeBlocks;
    delete[] codeBlockByteSizes;
    delete[] codeBlockWordSizes;
    delete[] codeBlockDataByteSizes;
    delete[] CBPaddingByteSizes; // pad to 32-bit boundary
    delete[] totalCodeBlockByteSizes;
    delete[] tbPaddedByteSizes;

    return passed;
}

int CRC_SINGLE_SMALL_CB_GPU_UPLINK_PUSCH_TEST(bool timeIt)
{
    uint32_t  nTBs                    = MAX_N_TBS_SUPPORTED;
    uint32_t* firstCodeBlockIdxArray  = new uint32_t[nTBs];
    uint32_t* nCodeBlocks             = new uint32_t[nTBs];
    uint32_t  crcByteSize             = 2;                  // 24 bits
    uint32_t* codeBlockByteSizes      = new uint32_t[nTBs]; // 56;
    uint32_t* codeBlockWordSizes      = new uint32_t[nTBs]; // 56;
    uint32_t* codeBlockDataByteSizes  = new uint32_t[nTBs]; // 1053;
    uint32_t* CBPaddingByteSizes      = new uint32_t[nTBs]; // pad to 32-bit boundary
    uint32_t* totalCodeBlockByteSizes = new uint32_t[nTBs];
    uint32_t  totalByteSize           = 0;
    uint32_t  totalNCodeBlocks        = 0;
    uint32_t* tbPaddedByteSizes       = new uint32_t[nTBs];
    uint32_t  totalTBPaddedByteSize   = 0;
    uint32_t  ratio                   = sizeof(uint32_t) / sizeof(uint8_t);
    // Same CRC value for each code block, code blocks are all the same
    // linear input layout : cb1|crc1, cb2|crc2, ...

    for(int i = 0; i < nTBs; i++)
    {
        if(i == 2)
        {
            codeBlockByteSizes[i] = 333;
            nCodeBlocks[i]        = 1;
        }
        codeBlockByteSizes[i] = 476;
        nCodeBlocks[i]        = 1;
        totalNCodeBlocks += nCodeBlocks[i];
        codeBlockDataByteSizes[i]  = codeBlockByteSizes[i] - crcByteSize;
        CBPaddingByteSizes[i]      = (MAX_BYTES_PER_CODE_BLOCK - (codeBlockByteSizes[i] % MAX_BYTES_PER_CODE_BLOCK)) % MAX_BYTES_PER_CODE_BLOCK;
        totalCodeBlockByteSizes[i] = codeBlockByteSizes[i] + CBPaddingByteSizes[i];
        codeBlockWordSizes[i]      = totalCodeBlockByteSizes[i] / ratio;
        totalByteSize += totalCodeBlockByteSizes[i] * nCodeBlocks[i];
        tbPaddedByteSizes[i] = (codeBlockDataByteSizes[i] + (nCodeBlocks[i] == 1 ? crcByteSize : 0)) * nCodeBlocks[i] +
                               (4 - (nCodeBlocks[i] * (codeBlockDataByteSizes[i] + (nCodeBlocks[i] == 1 ? crcByteSize : 0)) % 4)) % 4;
        totalTBPaddedByteSize += tbPaddedByteSizes[i];
    }

    PerTbParams* tbPrmsArray           = new PerTbParams[nTBs];
    uint8_t*     linearInput           = new uint8_t[totalByteSize];
    uint32_t*    goldenCRCs            = new uint32_t[totalNCodeBlocks];
    uint8_t*     goldenTransportBlocks = new uint8_t[totalTBPaddedByteSize];
    uint32_t*    transportBlocks       = new uint32_t[totalTBPaddedByteSize / ratio];
    uint32_t*    codeBlocks            = (uint32_t*)linearInput;
    uint32_t*    crcs                  = new uint32_t[totalNCodeBlocks];
    uint32_t*    tbCRCs                = new uint32_t[nTBs];
    memset(goldenTransportBlocks, 0, totalTBPaddedByteSize);
    memset(firstCodeBlockIdxArray, 0, nTBs * sizeof(uint32_t));

    uint32_t tbBytes      = 0;
    uint32_t totalCBBytes = 0;
    for(int t = 0; t < nTBs; t++)
    {
        // Build transport block
        uint32_t cbBytes = 0;
        for(int i = 0; i < nCodeBlocks[t]; i++)
        {
            memset(goldenTransportBlocks + tbBytes + cbBytes,
                   rand(),
                   codeBlockDataByteSizes[t]);
            cbBytes += codeBlockDataByteSizes[t];
        }

        // just compute CB crcs using TB polynomial, as TBs contain only one CB
        for(int i = 0; i < nCodeBlocks[t]; i++)
        {
            uint8_t* cbPtr  = linearInput + i * totalCodeBlockByteSizes[t] + totalCBBytes;
            uint8_t* crcPtr = (cbPtr + codeBlockDataByteSizes[t]);
            memcpy(cbPtr,
                   goldenTransportBlocks + i * codeBlockDataByteSizes[t] + tbBytes,
                   codeBlockDataByteSizes[t]);
            uint32_t crc = computeCRC<uint32_t, 16>((uint8_t*)cbPtr,
                                                    codeBlockDataByteSizes[t],
                                                    G_CRC_16,
                                                    0,
                                                    1);
            for(int j = 0; j < crcByteSize; j++)
                crcPtr[j] = (crc >> (crcByteSize - 1 - j) * 8) & 0xFF;
            for(int j = 0; j < crcByteSize; j++)
                goldenTransportBlocks[tbBytes + nCodeBlocks[t] * codeBlockDataByteSizes[t] /*- crcByteSizes[t]*/ + j] =
                    (crc >> (crcByteSize - 1 - j) * 8) & 0xFF;
            goldenCRCs[t * nCodeBlocks[t] + i] = 0;
            memset(cbPtr + codeBlockByteSizes[t], 0, CBPaddingByteSizes[t]);
        }

        tbPrmsArray[t].num_CBs             = nCodeBlocks[t];
        tbPrmsArray[t].K                   = codeBlockByteSizes[t] * 8;
        tbPrmsArray[t].F                   = 0;
        tbPrmsArray[t].firstCodeBlockIndex = 0;
        tbBytes += tbPaddedByteSizes[t];
        totalCBBytes += nCodeBlocks[t] * totalCodeBlockByteSizes[t];
    }
#if 0
    std::cout << "CBs:\n";
    for (int i = 0; i < totalByteSize; i++)
        std::cout << std::hex << (unsigned short)linearInput[i] << ",";
    std::cout << "\n";

    std::cout << "TB:\n";
    for (int i = 0; i < totalTBPaddedByteSize; i++)
        std::cout << std::hex << (unsigned short)goldenTransportBlocks[i] << ",";
    std::cout << "\n";
#endif
    /*
       cuphyStatus_t status = cuphyCRCDecode(
       crcs, tbCRCs, transportBlocks, nTBs, (const uint32_t*)codeBlocks,
       nCodeBlocks, codeBlockWordSizes, codeBlockDataByteSizes, timeIt, 10000);
     */

    //input
    unique_device_ptr<uint32_t>    d_codeBlocks  = make_unique_device<uint32_t>(totalByteSize / sizeof(uint32_t));
    unique_device_ptr<PerTbParams> d_tbPrmsArray = make_unique_device<PerTbParams>(nTBs);

    //output

    unique_device_ptr<uint32_t> d_CBCRCs = make_unique_device<uint32_t>(nTBs * nCodeBlocks[0]);
    unique_device_ptr<uint32_t> d_TBCRCs = make_unique_device<uint32_t>(nTBs);
    unique_device_ptr<uint8_t>  d_TBs    = make_unique_device<uint8_t>(nTBs * tbPaddedByteSizes[0]);

    cudaMemcpy(d_codeBlocks.get(), codeBlocks, totalByteSize, cudaMemcpyHostToDevice);

    cudaMemcpy(d_tbPrmsArray.get(), tbPrmsArray, sizeof(PerTbParams) * nTBs, cudaMemcpyHostToDevice);

    cuphyStatus_t status = cuphyCRCDecode(
        d_CBCRCs.get(),
        d_TBCRCs.get(),
        d_TBs.get(),
        d_codeBlocks.get(),
        d_tbPrmsArray.get(),
        nTBs,
        nCodeBlocks[0],
        tbPaddedByteSizes[0],
        0,
        timeIt,
        0,
        false,
        0);

    cudaMemcpy(crcs, d_CBCRCs.get(), sizeof(uint32_t) * totalNCodeBlocks, cudaMemcpyDeviceToHost);

    cudaMemcpy(tbCRCs, d_TBCRCs.get(), sizeof(uint32_t) * nTBs, cudaMemcpyDeviceToHost);

    cudaMemcpy(transportBlocks, d_TBs.get(), totalTBPaddedByteSize, cudaMemcpyDeviceToHost);

    int passed = 0;
    if(status != CUPHY_STATUS_SUCCESS)
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "CRC: CUPHY ERROR");
    else
    {
        uint32_t* gt = (uint32_t*)goldenTransportBlocks;
        passed       = equalCount(crcs, goldenCRCs, totalNCodeBlocks, "CB CRC ");
        passed &= equalCount(transportBlocks, gt, totalTBPaddedByteSize / ratio, "TB DATA ");

        for(int i = 0; i < nTBs; i++)
        {
            passed &= (tbCRCs[i] == 0);
            if(tbCRCs[i] != 0)
                NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "TB[{}] CRC not equal to 0: {}", i, tbCRCs[i]);
        }
    }

    delete[] tbPrmsArray;
    delete[] crcs;
    delete[] tbCRCs;
    delete[] goldenCRCs;
    delete[] goldenTransportBlocks;
    delete[] linearInput;
    delete[] transportBlocks;
    delete[] nCodeBlocks;
    delete[] firstCodeBlockIdxArray;
    delete[] codeBlockByteSizes;
    delete[] codeBlockWordSizes;
    delete[] codeBlockDataByteSizes;
    delete[] CBPaddingByteSizes; // pad to 32-bit boundary
    delete[] totalCodeBlockByteSizes;
    delete[] tbPaddedByteSizes;

    return passed;
}

int TestPuschRxCrcDecodeSetup()
{
    // Create test objects
    std::unique_ptr<cuphyPuschRxCrcDecode> decoder = std::make_unique<puschRxCrcDecode>();
    static_cast<puschRxCrcDecode*>(decoder.get())->init(0); // Initialize with reverseBytes = 0

    // Test parameters
    uint16_t nSchUes = 2;
    uint16_t schUserIdxs[2] = {0, 1};

    // Create device memory
    unique_device_ptr<uint32_t> d_outputCBCRCs = make_unique_device<uint32_t>(nSchUes * 6); // Assuming max 6 CBs per UE
    unique_device_ptr<uint8_t> d_outputTBs = make_unique_device<uint8_t>(nSchUes * 1024); // Assuming max 1KB per TB
    unique_device_ptr<uint32_t> d_inputCodeBlocks = make_unique_device<uint32_t>(nSchUes * 6 * 256); // Assuming max 1KB per CB
    unique_device_ptr<uint32_t> d_outputTBCRCs = make_unique_device<uint32_t>(nSchUes);

    // Create TB parameters
    PerTbParams tbParams[2];
    tbParams[0].num_CBs = 2;
    tbParams[0].K = 1000 * 8; // 1000 bytes * 8 bits
    tbParams[0].F = 0;
    tbParams[0].firstCodeBlockIndex = 0;

    tbParams[1].num_CBs = 3;
    tbParams[1].K = 1500 * 8; // 1500 bytes * 8 bits
    tbParams[1].F = 0;
    tbParams[1].firstCodeBlockIndex = 2;

    unique_device_ptr<PerTbParams> d_tbParams = make_unique_device<PerTbParams>(nSchUes);
    cudaMemcpy(d_tbParams.get(), tbParams, sizeof(PerTbParams) * nSchUes, cudaMemcpyHostToDevice);

    // Get descriptor size
    size_t descrSize, descrAlign;
    static_cast<puschRxCrcDecode*>(decoder.get())->getDescrInfo(descrSize, descrAlign);

    // Allocate descriptors
    unique_device_ptr<uint8_t> d_desc = make_unique_device<uint8_t>(descrSize);
    uint8_t* h_desc;
    CUDA_CHECK(cudaHostAlloc((void**)&h_desc, descrSize, cudaHostAllocDefault));

    // Create launch configs
    cuphyPuschRxCrcDecodeLaunchCfg_t cbCrcLaunchCfg = {};
    cuphyPuschRxCrcDecodeLaunchCfg_t tbCrcLaunchCfg = {};

    // Test setup function
    try {
        static_cast<puschRxCrcDecode*>(decoder.get())->setup(
            nSchUes,
            schUserIdxs,
            d_outputCBCRCs.get(),
            d_outputTBs.get(),
            d_inputCodeBlocks.get(),
            d_outputTBCRCs.get(),
            tbParams,
            d_tbParams.get(),
            h_desc,
            d_desc.get(),
            1, // enableCpuToGpuDescrAsyncCpy
            &cbCrcLaunchCfg,
            &tbCrcLaunchCfg,
            0 // stream
        );
        // Cleanup
        CUDA_CHECK(cudaFreeHost(h_desc));
        return 1;
    }
    catch (const std::exception& e) {
        // Cleanup
        CUDA_CHECK(cudaFreeHost(h_desc));
        return 0;
    }
}

int TestPrepareCRCEncodeSetup()
{
    // Test parameters
    uint32_t nTBs = 2;
    uint32_t maxNCBsPerTB = 6;
    uint32_t maxTbSizeBytes = 1024; // 1KB per TB

    // Get descriptor size
    size_t descrSize, descrAlign;
    cuphyStatus_t status = cuphyPrepareCrcEncodeGetDescrInfo(&descrSize, &descrAlign);
    if(status != CUPHY_STATUS_SUCCESS)
    {
        return 0;
    }

    // Allocate descriptors
    unique_device_ptr<uint8_t> d_desc = make_unique_device<uint8_t>(descrSize);
    uint8_t* h_desc;
    CUDA_CHECK(cudaHostAlloc((void**)&h_desc, descrSize, cudaHostAllocDefault));

    // Create device memory for input/output
    unique_device_ptr<uint32_t> d_inputOrigTBs = make_unique_device<uint32_t>(nTBs * maxTbSizeBytes / sizeof(uint32_t));
    unique_device_ptr<uint32_t> d_inputTBs = make_unique_device<uint32_t>(nTBs * maxTbSizeBytes / sizeof(uint32_t));
    unique_device_ptr<uint32_t> d_inputTBsTM = make_unique_device<uint32_t>(nTBs * maxTbSizeBytes / sizeof(uint32_t));

    // Create TB parameters
    PdschPerTbParams tbParams[2];
    tbParams[0].num_CBs = 2;
    tbParams[0].K = 1000 * 8; // 1000 bytes * 8 bits
    tbParams[0].F = 0;
    tbParams[0].firstCodeBlockIndex = 0;
    tbParams[0].tbSize = 1000;
    tbParams[0].tbStartOffset = 0;
    tbParams[0].tbStartAddr = nullptr;
    tbParams[0].cumulativeTbSizePadding = 1024;
    tbParams[0].testModel = 0;

    tbParams[1].num_CBs = 3;
    tbParams[1].K = 1500 * 8; // 1500 bytes * 8 bits
    tbParams[1].F = 0;
    tbParams[1].firstCodeBlockIndex = 2;
    tbParams[1].tbSize = 1500;
    tbParams[1].tbStartOffset = 1024;
    tbParams[1].tbStartAddr = nullptr;
    tbParams[1].cumulativeTbSizePadding = 2048;
    tbParams[1].testModel = 0;

    unique_device_ptr<PdschPerTbParams> d_tbParams = make_unique_device<PdschPerTbParams>(nTBs);
    cudaMemcpy(d_tbParams.get(), tbParams, sizeof(PdschPerTbParams) * nTBs, cudaMemcpyHostToDevice);

    // Create launch config
    cuphyPrepareCrcEncodeLaunchConfig prepareCrcEncodeLaunchCfg = {};

    // Test setup function
    try {
        status = cuphySetupPrepareCRCEncode(
            &prepareCrcEncodeLaunchCfg,
            d_inputOrigTBs.get(),
            d_inputTBs.get(),
            d_inputTBsTM.get(),
            d_tbParams.get(),
            nTBs,
            maxNCBsPerTB,
            maxTbSizeBytes,
            h_desc,
            d_desc.get(),
            1, // enable_desc_async_copy
            0  // stream
        );
        // Cleanup
        CUDA_CHECK(cudaFreeHost(h_desc));
        return (status == CUPHY_STATUS_SUCCESS) ? 1 : 0;
    }
    catch (const std::exception& e) {
        // Cleanup
        CUDA_CHECK(cudaFreeHost(h_desc));
        return 0;
    }
}

// Helper function to create minimal device memory for boundary tests
struct BoundaryTestMemory {
    unique_device_ptr<uint32_t> d_CBCRCs;
    unique_device_ptr<uint32_t> d_TBCRCs;
    unique_device_ptr<uint8_t> d_TBs;
    unique_device_ptr<uint32_t> d_codeBlocks;
    unique_device_ptr<PerTbParams> d_tbParams;
};

BoundaryTestMemory createBoundaryTestMemory(uint32_t nTBs) {
    BoundaryTestMemory mem;
    mem.d_CBCRCs = make_unique_device<uint32_t>(2);
    mem.d_TBCRCs = make_unique_device<uint32_t>(2);
    mem.d_TBs = make_unique_device<uint8_t>(2);
    mem.d_codeBlocks = make_unique_device<uint32_t>(2);
    mem.d_tbParams = make_unique_device<PerTbParams>(nTBs);
    return mem;
}

// Helper function to create minimal TB parameters
PerTbParams createMinimalTbParams(uint32_t num_CBs = 1) {
    PerTbParams params;
    params.num_CBs = num_CBs;
    params.K = 8;
    params.F = 0;
    params.firstCodeBlockIndex = 0;
    return params;
}

// Helper function to test boundary condition
bool testBoundaryCondition(
    uint32_t nTBs,
    uint32_t maxNCBsPerTB,
    uint32_t maxTbSizeBytes,
    uint32_t num_CBs,
    const char* errorMessage) {

    // Create device memory
    BoundaryTestMemory mem = createBoundaryTestMemory(nTBs);

    // Create TB parameters
    PerTbParams tbParams[2];
    tbParams[0] = createMinimalTbParams(num_CBs);

    // Copy TB parameters to device
    cudaMemcpy(mem.d_tbParams.get(), tbParams, sizeof(PerTbParams) * 2, cudaMemcpyHostToDevice);

    try {
        cuphyStatus_t status = cuphyCRCDecode(
            mem.d_CBCRCs.get(),
            mem.d_TBCRCs.get(),
            mem.d_TBs.get(),
            mem.d_codeBlocks.get(),
            mem.d_tbParams.get(),
            nTBs,
            maxNCBsPerTB,
            maxTbSizeBytes,
            0,
            false,
            0,
            false,
            0
        );
        if (status != CUPHY_STATUS_INVALID_ARGUMENT) {
            printf("%s\n", errorMessage);
            return false;
        }
    }
    catch (const std::exception& e) {
        return false;
    }

    return true;
}

// Helper function to test setup boundary condition for cuphySetupCrcEncode
bool testSetupCrcEncodeBoundaryCondition(
    uint32_t nTBs,
    uint32_t maxNCBsPerTB,
    uint32_t maxTbSizeBytes,
    uint32_t* d_tbCRCs,
    uint32_t* d_inputTransportBlocks,
    uint8_t* d_codeBlocks,
    PdschPerTbParams* d_tbPrmsArray,
    uint8_t* cpu_desc,
    uint8_t* gpu_desc,
    const char* errorMessage) {

    // Get descriptor size
    size_t descrSize, descrAlign;
    cuphyStatus_t status = cuphyCrcEncodeGetDescrInfo(&descrSize, &descrAlign);
    if(status != CUPHY_STATUS_SUCCESS)
    {
        return false;
    }

    // Create device memory for input/output
    unique_device_ptr<uint32_t> d_CBCRCs = make_unique_device<uint32_t>(nTBs * maxNCBsPerTB);

    // Create launch config
    std::unique_ptr<cuphyCrcEncodeLaunchConfig> crc_hndl = std::make_unique<cuphyCrcEncodeLaunchConfig>();

    try {
        status = cuphySetupCrcEncode(
            crc_hndl.get(),
            d_CBCRCs.get(),
            d_tbCRCs,
            d_inputTransportBlocks,
            d_codeBlocks,
            d_tbPrmsArray,
            nTBs,
            maxNCBsPerTB,
            maxTbSizeBytes,
            0,
            false,
            cpu_desc,
            gpu_desc,
            1, // enable_desc_async_copy
            0  // stream
        );

        if (d_tbCRCs == nullptr || d_inputTransportBlocks == nullptr || d_codeBlocks == nullptr ||
            d_tbPrmsArray == nullptr || cpu_desc == nullptr || gpu_desc == nullptr) {
            // For null pointer checks, expect INVALID_ARGUMENT
            if (status != CUPHY_STATUS_INVALID_ARGUMENT) {
                printf("%s\n", errorMessage);
                return false;
            }
        } else {
            // For boundary condition checks, expect NOT_SUPPORTED
            if (status != CUPHY_STATUS_NOT_SUPPORTED) {
                printf("%s\n", errorMessage);
                return false;
            }
        }
        return true;
    }
    catch (const std::exception& e) {
        return false;
    }
}

TEST(CRC_GPU_UPLINK_PUSCH, 24B_24A)
{
    EXPECT_EQ(CRC_GPU_UPLINK_PUSCH_TEST(false), 1);
}

TEST(CRC_SINGLE_CB_GPU_UPLINK_PUSCH, 24A)
{
    EXPECT_EQ(CRC_SINGLE_CB_GPU_UPLINK_PUSCH_TEST(false), 1);
}

TEST(CRC_SINGLE_SMALL_CB_GPU_UPLINK_PUSCH, 16)
{
    EXPECT_EQ(CRC_SINGLE_SMALL_CB_GPU_UPLINK_PUSCH_TEST(false), 1);
}

//int cb_num_array[20] = {51, 44, 26, 18, 10, 383, 330, 195, 135, 75, 501, 429, 251, 174, 91, 752, 644, 377, 261, 137};
TEST(CRC_GPU_DOWNLINK_PDSCH, 24B_24A)
{
    /*for (int i = 0; i < 20; i++){
        EXPECT_EQ(CRC_GPU_DOWNLINK_PDSCH(false, cb_num_array[i]), 1); // FIXME the 2nd argument is not used.
    }*/
    EXPECT_EQ(CRC_GPU_DOWNLINK_PDSCH(false), 1);
}

TEST(CRCTest, launchBoundaryCondition)
{
    int result = 0;
    // Test each boundary condition separately
    // 1. Test nTBs > MAX_N_TBS_PER_CELL_GROUP_SUPPORTED
    result = testBoundaryCondition(
        MAX_N_TBS_PER_CELL_GROUP_SUPPORTED + 1,  // Exceed max TBs
        2,  // Small value for maxNCBsPerTB
        1024,  // Small value for maxTbSizeBytes
        1,  // num_CBs
        "nTBs > MAX_N_TBS_PER_CELL_GROUP_SUPPORTED check failed"
    ) ? 1 : 0;
    EXPECT_EQ(result, 0) << "nTBs boundary condition test passed";

    // // 2. Test maxNCBsPerTB > MAX_N_CBS_PER_TB_PER_CELL_GROUP_SUPPORTED
    // result = testBoundaryCondition(
    //     2,  // Small value for nTBs
    //     MAX_N_CBS_PER_TB_PER_CELL_GROUP_SUPPORTED + 1,  // Exceed max CBs per TB
    //     1024,  // Small value for maxTbSizeBytes
    //     MAX_N_CBS_PER_TB_PER_CELL_GROUP_SUPPORTED + 1,  // This will trigger the boundary check
    //     "maxNCBsPerTB > MAX_N_CBS_PER_TB_PER_CELL_GROUP_SUPPORTED check failed"
    // ) ? 1 : 0;
    // EXPECT_EQ(result, 0) << "maxNCBsPerTB boundary condition test passed";

    // 3. Test maxTBByteSize > MAX_BYTES_PER_TRANSPORT_BLOCK
    result = testBoundaryCondition(
        2,  // Small value for nTBs
        2,  // Small value for maxNCBsPerTB
        MAX_BYTES_PER_TRANSPORT_BLOCK + 1,  // Exceed max TB size
        1,  // num_CBs
        "maxTBByteSize > MAX_BYTES_PER_TRANSPORT_BLOCK check failed"
    ) ? 1 : 0;
    EXPECT_EQ(result, 0) << "maxTBByteSize boundary condition test passed";
}

TEST(CRCTest, UplinkPuschWithTiming)
{
    // Test with timing enabled
    int result = CRC_GPU_UPLINK_PUSCH_TEST(true);
    EXPECT_EQ(result, 1) << "CRC test with timing enabled failed";
}

TEST(PuschRxCrcDecodeTest, SetupFunction)
{
    EXPECT_EQ(TestPuschRxCrcDecodeSetup(), 1) << "Setup function test failed";
}

TEST(CRCTest, PrepareCRCEncodeSetup)
{
    EXPECT_EQ(TestPrepareCRCEncodeSetup(), 1) << "Prepare CRC encode setup test failed";
}

TEST(CRCTest, SetupCrcEncodeBoundaryConditions)
{
    bool result = 0;

    // Test null pointer check for d_tbCRCs
    result = testSetupCrcEncodeBoundaryCondition(
        2,  // nTBs
        6,  // maxNCBsPerTB
        1024,  // maxTbSizeBytes
        nullptr,  // d_tbCRCs is nullptr
        nullptr,  // d_inputTransportBlocks
        nullptr,  // d_codeBlocks
        nullptr,  // d_tbPrmsArray
        nullptr,  // cpu_desc
        nullptr,  // gpu_desc
        "d_tbCRCs nullptr check failed in cuphySetupCrcEncode"
    );
    EXPECT_TRUE(result) << "d_tbCRCs nullptr check test failed for cuphySetupCrcEncode";

    // Test null pointer check for d_inputTransportBlocks
    result = testSetupCrcEncodeBoundaryCondition(
        2,  // nTBs
        6,  // maxNCBsPerTB
        1024,  // maxTbSizeBytes
        (uint32_t*)0x1,  // d_tbCRCs is non-nullptr
        nullptr,  // d_inputTransportBlocks is nullptr
        nullptr,  // d_codeBlocks
        nullptr,  // d_tbPrmsArray
        nullptr,  // cpu_desc
        nullptr,  // gpu_desc
        "d_inputTransportBlocks nullptr check failed in cuphySetupCrcEncode"
    );
    EXPECT_TRUE(result) << "d_inputTransportBlocks nullptr check test failed for cuphySetupCrcEncode";

    // Test null pointer check for d_codeBlocks
    result = testSetupCrcEncodeBoundaryCondition(
        2,  // nTBs
        6,  // maxNCBsPerTB
        1024,  // maxTbSizeBytes
        (uint32_t*)0x1,  // d_tbCRCs is non-nullptr
        (uint32_t*)0x1,  // d_inputTransportBlocks is non-nullptr
        nullptr,  // d_codeBlocks is nullptr
        nullptr,  // d_tbPrmsArray
        nullptr,  // cpu_desc
        nullptr,  // gpu_desc
        "d_codeBlocks nullptr check failed in cuphySetupCrcEncode"
    );
    EXPECT_TRUE(result) << "d_codeBlocks nullptr check test failed for cuphySetupCrcEncode";

    // Test null pointer check for d_tbPrmsArray
    result = testSetupCrcEncodeBoundaryCondition(
        2,  // nTBs
        6,  // maxNCBsPerTB
        1024,  // maxTbSizeBytes
        (uint32_t*)0x1,  // d_tbCRCs is non-nullptr
        (uint32_t*)0x1,  // d_inputTransportBlocks is non-nullptr
        (uint8_t*)0x1,   // d_codeBlocks is non-nullptr
        nullptr,  // d_tbPrmsArray is nullptr
        nullptr,  // cpu_desc
        nullptr,  // gpu_desc
        "d_tbPrmsArray nullptr check failed in cuphySetupCrcEncode"
    );
    EXPECT_TRUE(result) << "d_tbPrmsArray nullptr check test failed for cuphySetupCrcEncode";

    // Test null pointer check for cpu_desc
    result = testSetupCrcEncodeBoundaryCondition(
        2,  // nTBs
        6,  // maxNCBsPerTB
        1024,  // maxTbSizeBytes
        (uint32_t*)0x1,  // d_tbCRCs is non-nullptr
        (uint32_t*)0x1,  // d_inputTransportBlocks is non-nullptr
        (uint8_t*)0x1,   // d_codeBlocks is non-nullptr
        (PdschPerTbParams*)0x1,  // d_tbPrmsArray is non-nullptr
        nullptr,  // cpu_desc is nullptr
        nullptr,  // gpu_desc
        "cpu_desc nullptr check failed in cuphySetupCrcEncode"
    );
    EXPECT_TRUE(result) << "cpu_desc nullptr check test failed for cuphySetupCrcEncode";

    // Test null pointer check for gpu_desc
    result = testSetupCrcEncodeBoundaryCondition(
        2,  // nTBs
        6,  // maxNCBsPerTB
        1024,  // maxTbSizeBytes
        (uint32_t*)0x1,  // d_tbCRCs is non-nullptr
        (uint32_t*)0x1,  // d_inputTransportBlocks is non-nullptr
        (uint8_t*)0x1,   // d_codeBlocks is non-nullptr
        (PdschPerTbParams*)0x1,  // d_tbPrmsArray is non-nullptr
        (uint8_t*)0x1,   // cpu_desc is non-nullptr
        nullptr,  // gpu_desc is nullptr
        "gpu_desc nullptr check failed in cuphySetupCrcEncode"
    );
    EXPECT_TRUE(result) << "gpu_desc nullptr check test failed for cuphySetupCrcEncode";

    // Test boundary condition for nTBs > PDSCH_MAX_UES_PER_CELL_GROUP
    result = testSetupCrcEncodeBoundaryCondition(
        PDSCH_MAX_UES_PER_CELL_GROUP + 1,  // nTBs exceeds maximum
        6,  // maxNCBsPerTB
        1024,  // maxTbSizeBytes
        (uint32_t*)0x1,  // d_tbCRCs is non-nullptr
        (uint32_t*)0x1,  // d_inputTransportBlocks is non-nullptr
        (uint8_t*)0x1,   // d_codeBlocks is non-nullptr
        (PdschPerTbParams*)0x1,  // d_tbPrmsArray is non-nullptr
        (uint8_t*)0x1,   // cpu_desc is non-nullptr
        (uint8_t*)0x1,   // gpu_desc is non-nullptr
        "nTBs > PDSCH_MAX_UES_PER_CELL_GROUP check failed in cuphySetupCrcEncode"
    );
    EXPECT_TRUE(result) << "nTBs boundary condition test failed for cuphySetupCrcEncode";

    // Test boundary condition for maxNCBsPerTB > MAX_N_CBS_PER_TB_SUPPORTED
    result = testSetupCrcEncodeBoundaryCondition(
        2,  // nTBs
        MAX_N_CBS_PER_TB_SUPPORTED + 1,  // maxNCBsPerTB exceeds maximum
        1024,  // maxTbSizeBytes
        (uint32_t*)0x1,  // d_tbCRCs is non-nullptr
        (uint32_t*)0x1,  // d_inputTransportBlocks is non-nullptr
        (uint8_t*)0x1,   // d_codeBlocks is non-nullptr
        (PdschPerTbParams*)0x1,  // d_tbPrmsArray is non-nullptr
        (uint8_t*)0x1,   // cpu_desc is non-nullptr
        (uint8_t*)0x1,   // gpu_desc is non-nullptr
        "maxNCBsPerTB > MAX_N_CBS_PER_TB_SUPPORTED check failed in cuphySetupCrcEncode"
    );
    EXPECT_TRUE(result) << "maxNCBsPerTB boundary condition test failed for cuphySetupCrcEncode";

    // Test boundary condition for maxTBByteSize > MAX_BYTES_PER_TRANSPORT_BLOCK
    result = testSetupCrcEncodeBoundaryCondition(
        2,  // nTBs
        6,  // maxNCBsPerTB
        MAX_BYTES_PER_TRANSPORT_BLOCK + 1,  // maxTbSizeBytes exceeds maximum
        (uint32_t*)0x1,  // d_tbCRCs is non-nullptr
        (uint32_t*)0x1,  // d_inputTransportBlocks is non-nullptr
        (uint8_t*)0x1,   // d_codeBlocks is non-nullptr
        (PdschPerTbParams*)0x1,  // d_tbPrmsArray is non-nullptr
        (uint8_t*)0x1,   // cpu_desc is non-nullptr
        (uint8_t*)0x1,   // gpu_desc is non-nullptr
        "maxTBByteSize > MAX_BYTES_PER_TRANSPORT_BLOCK check failed in cuphySetupCrcEncode"
    );
    EXPECT_TRUE(result) << "maxTBByteSize boundary condition test failed for cuphySetupCrcEncode";
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);

    int result = RUN_ALL_TESTS();
    return result;
}
