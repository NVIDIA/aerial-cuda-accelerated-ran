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

#if !defined(LDPC2_APP_ADDRESS_FP_DP_DESC_CUH_INCLUDED_)
#define LDPC2_APP_ADDRESS_FP_DP_DESC_CUH_INCLUDED_

#include "ldpc2_bg_desc.hpp"

////////////////////////////////////////////////////////////////////////
// APP address calculation using the floating point (FP) unit (with
// denormalized floats) and the dot product (DP) instruction.
//
// addr_app = [(col_app * Z) + (shift + threadIdx)    ] * sizeof(T)    threadIdx < (Z-shift)    (threadIdx + shift) < Z
// addr_app = [(col_app * Z) + (shift + threadIdx - Z)] * sizeof(T)    threadIdx >= (Z-shift)   (threadIdx + shift) >= Z

////////////////////////////////////////////////////////////////////////
// GENERAL CONSIDERATIONS FOR 2x SIMD ADDRESS CALCULATIONS
// 1.) Overall maximum APP address (as integer value):
//     BG1: ((68 * 384) - 1) * sizeof(T)
//         half:   52,222
//         half2: 104,444
//     BG2: ((52 * 384) - 1) * sizeof(T)
//         half:   39,934
//         half2:  79,868
//     Therefore:
//         - the maximum address can be stored in 16 bits for all (BG, Z) for half
//         - the maximum address CANNOT be stored in 16 bits for half2
// 2.) Calculating an APP address for a given (col_idx, shift) pair
//
// addr_app = [(col_idx * Z) + (shift + threadIdx)    ] * sizeof(T)    threadIdx < (Z-shift)
// addr_app = [(col_idx * Z) + (shift + threadIdx - Z)] * sizeof(T)    threadIdx >= (Z-shift)
//
// 3.) When shift is zero:
// addr_app = [(col_idx * Z) + (threadIdx)] * sizeof(T) = (col_idx * Z * sizeof(T)) + threadIdx
// (Depending on what base graph data is stored, this can be a simple ADD instruction.)
//
// Note also that the shift for the rightmost (last) element of each parity node
// row is zero. As such, when the row degree is odd, we can avoid the conditional
// processing for the last row element. When the row degree is even, this zero
// shift value will be paired with another (non-zero) shift value.
// BG1: Total edges  = 316
//      num rows     = 46
//      num odd rows = 30
//
// 4.) Max column for the address of a 32-bit APP element (e.g. float or half2)
//     that can be stored in 16 bits:
// USHRT_MAX = 2^16 - 1 = 65535
// MAX_COL_IDX_32 = floor(USHRT_MAX / (384 * sizeof(half2)))
//                = 42
//
// The APP address (where address = index * sizeof(T)) is given by:
//
// addr_app = [(col_app * Z) + (shift + threadIdx)    ] * sizeof(T)    threadIdx < (Z-shift)
// addr_app = [(col_app * Z) + (shift + threadIdx - Z)] * sizeof(T)    threadIdx >= (Z-shift)
//
//
// addr_app = [(col_idx * Z * sizeof(T)) + (shift + threadIdx    ) * sizeof(T)]     threadIdx < (Z-shift)    (Z - threadIdx) >  shift
// addr_app = [(col_idx * Z * sizeof(T)) + (shift + threadIdx - Z) * sizeof(T)]     threadIdx >= (Z-shift)   (Z - threadIdx) <= shift
//
// addr_app = [(col_idx * Z * sizeof(T)) + (shift + threadIdx) * sizeof(T)                ]     threadIdx < (Z-shift)    (Z - threadIdx) >  shift
// addr_app = [(col_idx * Z * sizeof(T)) + (shift + threadIdx) * sizeof(T) - Z * sizeof(T)]     threadIdx >= (Z-shift)   (Z - threadIdx) <= shift
//
// addr_app = [((col_idx * Z  + shift) * sizeof(T)) + (threadIdx * sizeof(T))                         ]     threadIdx < (Z-shift)    (Z - threadIdx) >  shift
// addr_app = [((col_idx * Z) + shift) * sizeof(T)) + (threadIdx * sizeof(T)) + (-1) * (Z * sizeof(T))]     threadIdx >= (Z-shift)   (Z - threadIdx) <= shift
//
// addr_app = [((col_idx * Z + shift) * sizeof(T)) + (threadIdx * sizeof(T))                         ]     threadIdx < (Z-shift)    (Z - threadIdx) >  shift
// addr_app = [((col_idx * Z + shift) * sizeof(T)) + (threadIdx * sizeof(T)) + (-1) * (Z * sizeof(T))]     threadIdx >= (Z-shift)   (Z - threadIdx) <= shift
//
// --base offset---->|
//     <---shift---->|
//     | - - - - - - - - - - - - - - - - - - - - |
//  0  | . . . . . . x                           |
//  1  |     (6)       x                         |
//  2  |                 x                       |
//  3  |                   x                     |
//  4  |                     x                   |
//  5  |                       x                 |
//  6  |                         x               |
//  7  |                           x             |
//  8  |                             x           |
//  9  |                               x         |
// 10  |                                 x       |
// 11  |                                   x     |
// 12  |                                     x   |
// 13  |                                       x |
// 14  | x.......................................|<-- wrap index (Z - shift)
// 15  |   x                                     |
// 16  |     x                                   |
// 17  |       x                                 |
// 18  |         x                               |
// 19  |           x                             |
//     | - - - - - - - - - - - - - - - - - - - - |
//
// Below, COND(Z), means:
//    COND(Z) = Z * [0 ? (threadIdx < (Z-shift)) : 1]
//
// APPROACH:
// R0 =  threadIdx + Z                                               (constant for all addresses, unique per thread, adding "extra" Z to keep FMA positive)
// COL_IDX_SHIFT_LOW  = ((col_idx[0] - 1) * Z + shift) * sizeof(T)   (signed, stored as full int32) (Subtracting "extra" Z)
// COL_IDX_SHIFT_HIGH = ((col_idx[1] - 1) * Z + shift) * sizeof(T)   (signed, stored as full int32) (Subtracting "extra" Z)
//
// R1 = HSET2.GE.BF(threadIdx, wrapIndex)                             (threadIdx >= wrap_index) ? : 1.0 : 0.0
// R2 = R0 - (R1 * Z)                                                 threadIdx + Z - COND(Z)                    (always >= 0, can be fp16 denormalized)
// dp2a.lo.s32.s32 ADDR_0, R2, 0x00 00 00 sz, COL_IDX_SHIFT_LOW;      [col_idx * Z + shift + threadIdx - COND(Z)] * sizeof(T)
// dp2a.hi.s32.s32 ADDR_1, R2, 0x00 sz 00 00, COL_IDX_SHIFT_HIGH;
//
// In practice, this ends up being 6 instructions because the source value for
// dp2a (COL_IDX_SHIFT_LOW/HIGH above) needs to be in a register, as opposed
// to a kernel argument in constant memory. Therefore, two extra mov
// instructions are generated in addition to the instructions above.

namespace ldpc2
{

////////////////////////////////////////////////////////////////////////
// app_address_fp_dp_desc
//   -------------------------
//   |    high   |    low    |  RA
//   -------------------------
//
//   -------------------------
//   |  a  |  b  |  c  |  d  |  RB
//   -------------------------
//
//   dp2a.hi: output = (RB.a * RA.high) + (RB.b * RA.low)
//   dp2a.lo: output = (RB.c * RA.high) + (RB.d * RA.low)
// We want the dot product instruction to:
// a.) select between high and low, and
// b.) multiply by sizeof(T)
template <typename T> struct app_address_fp_dp_const
{
    static const int low_value =  static_cast<int>(sizeof(T));
    static const int high_value = static_cast<int>(sizeof(T) << 24);
};

////////////////////////////////////////////////////////////////////////
// app_loc_address_fp_dp_desc
// Manager for calculation and storage of APP locations (in shared memory)
// This implementation stores shared memory addresses (as opposed to
// storing INDICES into the APP array). The addresses are stored in
// registers (to avoid recalculation), so use will result in increased
// register pressure.
template <typename T, int BG>
struct app_loc_address_fp_dp_desc
{
    //------------------------------------------------------------------
    // Base graph descriptor type used by this app address calculator
    typedef BG_adj_desc<BG> bg_desc_t;
    // Z=384 packed as raw fp16x2 denormalized integer payload.
    static constexpr uint32_t Z384_RAW_FP16X2 = (384u << 16) | 384u;
    //------------------------------------------------------------------
    // app_loc_address_fp_dp_desc()
    // Constructor using original LDPC_kernel_params struct
    __device__
    app_loc_address_fp_dp_desc(const LDPC_kernel_params& params,
                               const bg_desc_t&          bgd,
                               unsigned int              t_idx) : bg_desc(bgd),
                                                                  Z_Z(h0_h0(params.Z)),
                                                                  tIdx_tIdx(h0_h0(t_idx)),
                                                                  tIdx_Z(get_tIdx_plus_Z(tIdx_tIdx, params.Z))
    {
    }
    //------------------------------------------------------------------
    // app_loc_address_fp_dp_desc()
    // Constructor using descriptor config struct
    __device__
    app_loc_address_fp_dp_desc(const cuphyLDPCDecodeConfigDesc_t& config,
                               const bg_desc_t&                   bgd,
                               unsigned int                       t_idx) : bg_desc(bgd),
                                                                           Z_Z(h0_h0(config.Z)),
                                                                           tIdx_tIdx(h0_h0(t_idx)),
                                                                           tIdx_Z(get_tIdx_plus_Z(tIdx_tIdx, config.Z))
    {
    }
    //------------------------------------------------------------------
    // get_tIdx_plus_Z()
    __device__ static
    word_t get_tIdx_plus_Z(word_t tIdx_tIdx_in, unsigned int Z)
    {
        if(Z == 384u)
        {
            word_t tIdx_Z_out;
            tIdx_Z_out.u32 = tIdx_tIdx_in.u32 + Z384_RAW_FP16X2;
            return tIdx_Z_out;
        }
        return h0_h0(static_cast<unsigned int>(tIdx_tIdx_in.u16x2.x) + Z);
    }
    //------------------------------------------------------------------
    // generate()
    template <int CHECK_IDX>
    __device__
    void generate(int (&app_addr)[row_degree<BG, CHECK_IDX>::value])
    {
        const int ROW_DEGREE  = row_degree<BG, CHECK_IDX>::value;
        const int PAIR_OFFSET = row_pair_index<BG, CHECK_IDX>::value;
        #pragma unroll
        for(int i = 0; i < (ROW_DEGREE + 1) / 2; ++i)
        {
            // R0 =  threadIdx + Z                                               (constant for all addresses, unique per thread, adding "extra" Z to keep FMA positive)
            // COL_IDX_SHIFT_LOW  = ((col_idx[0] - 1) * Z + shift) * sizeof(T)   (signed, stored as full int32) (Subtracting "extra" Z)
            // COL_IDX_SHIFT_HIGH = ((col_idx[1] - 1) * Z + shift) * sizeof(T)   (signed, stored as full int32) (Subtracting "extra" Z)
            //
            // R1 = HSET2.GE.BF(threadIdx, wrapIndex)                             (threadIdx >= wrap_index) ? : 1.0 : 0.0
            // R2 = R0 - (R1 * Z)                                                 threadIdx + Z - COND(Z)                    (always positive, can be fp16 denormalized)
            // dp2a.lo.s32.s32 ADDR_0, R2, 0x00 00 00 sz, COL_IDX_SHIFT_LOW;      [col_idx * Z + shift + threadIdx - COND(Z)] * sizeof(T)
            // dp2a.hi.s32.s32 ADDR_1, R2, 0x00 sz 00 00, COL_IDX_SHIFT_HIGH; 

            word_t WRAP_INDEX, R1, R2;
            
            const int32_t COL_IDX_SHIFT_LOW  = bg_desc.nodes[PAIR_OFFSET + i].col_Z_shift_low;
            const int32_t COL_IDX_SHIFT_HIGH = bg_desc.nodes[PAIR_OFFSET + i].col_Z_shift_high;
            WRAP_INDEX.u32                   = bg_desc.nodes[PAIR_OFFSET + i].wrap_index;
            
            R1                   = hset2_bf_ge(tIdx_tIdx, WRAP_INDEX);          // R1 = HSET2.GE.BF(threadIdx, WRAP_INDEX)  (HSET2.BF instruction)
            R2                   = hfma2(hneg2(R1), Z_Z, tIdx_Z);               // R2 = threadIdx + Z - COND(Z)

            //if((0 == CHECK_IDX) && (0 == threadIdx.x) && (0 == i))
            //{
            //    printf("threadIdx = %u, i = (%i, %i), WRAP_INDEX = (%u, %u), COL_Z_SHIFT = (%i, %i), R1 = (%.1f, %.1f), R2 = (%u, %u)\n",
            //           threadIdx.x,
            //           i * 2 + 1,
            //           i * 2,
            //           WRAP_INDEX.u32 >> 16,
            //           WRAP_INDEX.u32 & 0x0000FFFF,
            //           COL_IDX_SHIFT_HIGH,
            //           COL_IDX_SHIFT_LOW,
            //           __high2float(R1.f16x2),
            //           __low2float(R1.f16x2),
            //           R2.u16x2.y,
            //           R2.u16x2.x);
            //}

            //   -------------------------
            //   |    high   |    low    |  RA
            //   -------------------------
            //
            //   -------------------------
            //   |  a  |  b  |  c  |  d  |  RB
            //   -------------------------
            //
            //   dp2a.hi: output = (RB.a * RA.high) + (RB.b * RA.low)
            //   dp2a.lo: output = (RB.c * RA.high) + (RB.d * RA.low)
            
            LDPC2_ASM("\t"
                      "{\n\t\t"
                      "dp2a.lo.s32.s32 %0, %1, %2, %3;\n\t"
                      "}\n"
                      : "=r"(app_addr[i*2])
                      : "r"(R2.i32),
                        "n"(app_address_fp_dp_const<T>::low_value),
                        "r"(COL_IDX_SHIFT_LOW));


            if(((i * 2) + 1) < ROW_DEGREE)
            {
                LDPC2_ASM("\t"
                          "{\n\t\t"
                          "dp2a.hi.s32.s32 %0, %1, %2, %3;\n\t"
                          "}\n"
                          : "=r"(app_addr[i*2 + 1])
                          : "r"(R2.i32),
                            "n"(app_address_fp_dp_const<T>::high_value),
                            "r"(COL_IDX_SHIFT_HIGH));
            }
        }
    }
    //------------------------------------------------------------------
    // get_bg_desc()
    static const bg_desc_t* get_bg_desc(int Z)
    {
        return get_adj_BG_desc<T, BG>(Z);
    }
    //------------------------------------------------------------------
    // get_bg_desc_small()
    static const bg_desc_t* get_bg_desc_small(int Z)
    {
        return get_adj_BG_desc_small<T, BG>(Z);
    }
    //------------------------------------------------------------------
    // Data
    const bg_desc_t& bg_desc;
    const word_t     Z_Z;
    const word_t     tIdx_tIdx;
    const word_t     tIdx_Z;
};


} // namespace ldpc2

#endif // !defined(LDPC2_APP_ADDRESS_FP_DP_DESC_CUH_INCLUDED_)
