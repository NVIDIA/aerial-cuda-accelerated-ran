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

#if !defined(LDPC2_BOX_PLUS_CUH_INCLUDED_)
#define LDPC2_BOX_PLUS_CUH_INCLUDED_

#include "ldpc2.cuh"

namespace ldpc2
{

//#if __CUDA_ARCH__ != 860
////////////////////////////////////////////////////////////////////////
// box_plus()
// Returns a word with pairwise (hi/lo) box_plus fp16 values, where each
// component is given by:
//
// box_plus(a, b) = min(abs(a), abs(b)) * xorsign(a, b)
//
// where
//
// xorsign(a,b) =  1.0      if signbit(a) == signbit(b)
//                -1.0      otherwise
__device__ inline word_t box_plus(word_t a, word_t b)
{
    word_t   out;
#if __CUDA_ARCH__ >= 860
    asm volatile("min.xorsign.abs.f16x2 %0, %1, %2;\n"
                 : "=r"(out.u32)
                 : "r"(a.u32),
                   "r"(b.u32));
#elif __CUDA_ARCH__ >= 800
    out.f16x2 =  __hmin2(__habs2(a.f16x2), __habs2(b.f16x2));
    out.u32   |= ((a.u32 ^ b.u32) & 0x80008000);
#else
    word_t   MIN_VALUES = select_from_mask(hset2_bm_lt(fp16x2_abs(a), fp16x2_abs(b)), a, b);
    uint32_t SIGNS_XOR  = a.u32 ^ b.u32;
    out.u32 = (SIGNS_XOR & ~0x7FFF7FFF) | (MIN_VALUES.u32 & 0x7FFF7FFF);
#endif

    //printf("box_plus: inputs = (%f, %f)  (%f %f), output = (%f %f)\n",
    //       __high2float(a.f16x2),
    //       __low2float(a.f16x2),
    //       __high2float(b.f16x2),
    //       __low2float(b.f16x2),
    //       __high2float(out.f16x2),
    //       __low2float(out.f16x2));
    return out;
}
//#endif //__CUDA_ARCH__ != 860

struct box_plus_op
{
    static __device__ word_t box_plus(word_t a, word_t b)
    {
        word_t   out;
#if __CUDA_ARCH__ >= 860
        asm volatile("min.xorsign.abs.f16x2 %0, %1, %2;\n"
                     : "=r"(out.u32)
                     : "r"(a.u32),
                       "r"(b.u32));
#elif __CUDA_ARCH__ >= 800
        out.f16x2 =  __hmin2(__habs2(a.f16x2), __habs2(b.f16x2));
        out.u32   |= ((a.u32 ^ b.u32) & 0x80008000);
#else
        word_t   MIN_VALUES = select_from_mask(hset2_bm_lt(fp16x2_abs(a), fp16x2_abs(b)), a, b);
        uint32_t SIGNS_XOR  = a.u32 ^ b.u32;
        out.u32 = (SIGNS_XOR & ~0x7FFF7FFF) | (MIN_VALUES.u32 & 0x7FFF7FFF);
#endif
        //printf("box_plus: inputs = (%f, %f)  (%f %f), output = (%f %f)\n",
        //       __high2float(a.f16x2),
        //       __low2float(a.f16x2),
        //       __high2float(b.f16x2),
        //       __low2float(b.f16x2),
        //       __high2float(out.f16x2),
        //       __low2float(out.f16x2));
        return out;
    }

    // Overwrite-and-return variant used to reduce temporary register pressure.
    static __device__ word_t box_plus_consume(word_t a, word_t b)
    {
#if __CUDA_ARCH__ >= 860
        asm volatile("min.xorsign.abs.f16x2 %0, %0, %1;\n"
                     : "+r"(a.u32)
                     : "r"(b.u32));
        return a;
#else
        return box_plus(a, b);
#endif
    }
};

template <typename T, class TBoxPlusOp, int ROW_DEGREE, int UPDATE_ROW_DEGREE> struct box_plus_seq_gen;

//template <int ROW_DEGREE, int UPDATE_ROW_DEGREE> __device__ void box_plus_seq(word_t (&seq)[(UPDATE_ROW_DEGREE + 1) / 2],
//                                                                              word_t (&app)[(ROW_DEGREE + 1) / 2]);

////////////////////////////////////////////////////////////////////////
// box_plus_seq_gen<__half, 3, 2>()
// Input:
// [    X | VAL_2 ]  [ VAL_1 | VAL_0 ]
// Output:
// [ box_plus(0, 2) | box_plus(1, 2) ]
template <class TBoxPlusOp> struct box_plus_seq_gen<__half, TBoxPlusOp, 3, 2>
{
    typedef TBoxPlusOp op_t;

    __device__ static void generate(word_t (&seq)[1], word_t (&app)[2])
    {
        const word_t& VAL_1_0  = app[0];
        const word_t& VAL_2    = app[1];
        word_t        VAL_2_2  = duplicate_low(VAL_2);
        word_t        VAL_0_1  = swap_high_low(VAL_1_0);
        word_t        BP_02_12 = op_t::box_plus(VAL_2_2, VAL_0_1);
        seq[0] = BP_02_12;
        //printf("box_plus_seq_gen<3, 2>: inputs = (    %.0f) (%.0f %.0f), outputs = (%.0f %.0f)\n",
        //       __low2float(app[1].f16x2),
        //       __high2float(app[0].f16x2),
        //       __low2float(app[0].f16x2),
        //       __high2float(seq[0].f16x2),
        //       __low2float(seq[0].f16x2));
    }
};

////////////////////////////////////////////////////////////////////////
// box_plus_seq_gen<__half, 3, 3>()
// Input:
// [    X | VAL_2 ]  [ VAL_1 | VAL_0 ]
// Output:
// [ X  | box_plus(0, 1) ] [ box_plus(0, 2) | box_plus(1, 2) ]
template <class TBoxPlusOp> struct box_plus_seq_gen<__half, TBoxPlusOp, 3, 3>
{
    typedef TBoxPlusOp op_t;

    __device__ static void generate(word_t (&seq)[2], word_t (&app)[2])
    {
        const word_t& VAL_1_0  = app[0];
        const word_t& VAL_2    = app[1];
        word_t        VAL_2_2  = duplicate_low(VAL_2);
        word_t        VAL_0_1  = swap_high_low(VAL_1_0);
        word_t        BP_02_12 = op_t::box_plus(VAL_2_2, VAL_0_1);
        word_t        BP_01_01 = op_t::box_plus(VAL_1_0, VAL_0_1);
        seq[0] = BP_02_12;
        seq[1].f16x2.x = BP_01_01.f16x2.x;
        seq[1].f16x2.y = 0;
        //printf("box_plus_seq_gen<3, 3>: inputs = (    %.0f) (%.0f %.0f), outputs = (    %.0f) (%.0f %.0f)\n",
        //       __low2float(app[1].f16x2),
        //       __high2float(app[0].f16x2),
        //       __low2float(app[0].f16x2),
        //       __low2float(seq[1].f16x2),
        //       __high2float(seq[0].f16x2),
        //       __low2float(seq[0].f16x2));
    }
};

////////////////////////////////////////////////////////////////////////
// box_plus_seq_gen<__half, 4, 3>()
// Input:
// [ VAL_3 | VAL_2 ]  [ VAL_1 | VAL_0 ]
// Output:
// [ 0  | box_plus(0, 1, 3) ] [ box_plus(0, 2, 3) | box_plus(1, 2, 3) ]
template <class TBoxPlusOp> struct box_plus_seq_gen<__half, TBoxPlusOp, 4, 3>
{
    typedef TBoxPlusOp op_t;

    __device__ static void generate(word_t (&seq)[2], word_t (&app)[2])
    {
        const word_t& VAL_1_0 = app[0];
        const word_t& VAL_3_2 = app[1];
        word_t BP_13_02       = op_t::box_plus(VAL_3_2,  VAL_1_0);
        word_t BP_02_13       = swap_high_low(BP_13_02);
        word_t BP_012_013     = op_t::box_plus(BP_02_13, VAL_1_0);
        word_t BP_023_123     = op_t::box_plus_consume(BP_02_13, VAL_3_2);
        seq[0] = BP_023_123;
        seq[1].f16x2.x = BP_012_013.f16x2.x;
        seq[1].f16x2.y = 0;
        //printf("box_plus_seq_gen<4, 3>: inputs = (%.0f %.0f)  (%.0f %.0f), outputs = (%.0f %.0f) (%.0f %.0f)\n",
        //       __high2float(app[1].f16x2),
        //       __low2float(app[1].f16x2),
        //       __high2float(app[0].f16x2),
        //       __low2float(app[0].f16x2),
        //       __high2float(seq[1].f16x2),
        //       __low2float(seq[1].f16x2),
        //       __high2float(seq[0].f16x2),
        //       __low2float(seq[0].f16x2));
    }
};

////////////////////////////////////////////////////////////////////////
// box_plus_seq_gen<__half, 4, 4>()
// Input:
// [ VAL_3 | VAL_2 ]  [ VAL_1 | VAL_0 ]
// Output:
// [ box_plus(0, 1, 2) | box_plus(0, 1, 3) ] [ box_plus(0, 2, 3) | box_plus(1, 2, 3) ]
template <class TBoxPlusOp> struct box_plus_seq_gen<__half, TBoxPlusOp, 4, 4>
{
    typedef TBoxPlusOp op_t;

    __device__ static void generate(word_t (&seq)[2], word_t (&app)[2])
    {
        const word_t& VAL_1_0 = app[0];
        const word_t& VAL_3_2 = app[1];
        word_t BP_13_02       = op_t::box_plus(VAL_3_2,  VAL_1_0);
        word_t BP_02_13       = swap_high_low(BP_13_02);
        word_t BP_012_013     = op_t::box_plus(BP_02_13, VAL_1_0);
        word_t BP_023_123     = op_t::box_plus_consume(BP_02_13, VAL_3_2);
        seq[0] = BP_023_123;
        seq[1] = BP_012_013;
        //printf("box_plus_seq_gen<4, 4>: inputs = (%.0f %.0f)  (%.0f %.0f), outputs = (%.0f %.0f) (%.0f %.0f)\n",
        //       __high2float(app[1].f16x2),
        //       __low2float(app[1].f16x2),
        //       __high2float(app[0].f16x2),
        //       __low2float(app[0].f16x2),
        //       __high2float(seq[1].f16x2),
        //       __low2float(seq[1].f16x2),
        //       __high2float(seq[0].f16x2),
        //       __low2float(seq[0].f16x2));
    }
};

////////////////////////////////////////////////////////////////////////
// box_plus_seq_gen<__half, 5, 4>()
// Input:
// [   X  | VAL_4 ] [ VAL_3 | VAL_2 ]  [ VAL_1 | VAL_0 ]
// Output:
// [ box_plus(0, 1, 2, 4) | box_plus(0, 1, 3, 4) ] [ box_plus(0, 2, 3, 4) | box_plus(1, 2, 3, 4) ]
template <class TBoxPlusOp> struct box_plus_seq_gen<__half, TBoxPlusOp, 5, 4>
{
    typedef TBoxPlusOp op_t;

    __device__ static void generate(word_t (&seq)[2], word_t (&app)[3])
    {
        const word_t& VAL_1_0 = app[0];
        const word_t& VAL_3_2 = app[1];
        const word_t& VAL_4   = app[2];
        word_t        VAL_4_4 = duplicate_low(VAL_4);
        word_t BP_13_02       = op_t::box_plus(VAL_3_2,    VAL_1_0);
        word_t BP_02_13       = swap_high_low(BP_13_02);
        word_t BP_024_134     = op_t::box_plus(BP_02_13,   VAL_4_4);
        word_t BP_0124_0134   = op_t::box_plus(BP_024_134, VAL_1_0);
        word_t BP_0234_1234   = op_t::box_plus_consume(BP_024_134, VAL_3_2);
        seq[0] = BP_0234_1234;
        seq[1] = BP_0124_0134;
        //printf("box_plus_seq_gen<5, 4>: inputs = (    %.0f) (%.0f %.0f) (%.0f %.0f), outputs = (%.0f %.0f) (%.0f %.0f)\n",
        //       __low2float(app[2].f16x2),
        //       __high2float(app[1].f16x2),
        //       __low2float(app[1].f16x2),
        //       __high2float(app[0].f16x2),
        //       __low2float(app[0].f16x2),
        //       __high2float(seq[1].f16x2),
        //       __low2float(seq[1].f16x2),
        //       __high2float(seq[0].f16x2),
        //       __low2float(seq[0].f16x2));
    }
};

////////////////////////////////////////////////////////////////////////
// box_plus_seq_gen<__half, 5, 5>()
// Input:
// [   X  | VAL_4 ] [ VAL_3 | VAL_2 ]  [ VAL_1 | VAL_0 ]
// Output:
// [    X   box_plus(0, 1, 2, 3) ] [ box_plus(0, 1, 2, 4) | box_plus(0, 1, 3, 4) ] [ box_plus(0, 2, 3, 4) | box_plus(1, 2, 3, 4) ]
template <class TBoxPlusOp> struct box_plus_seq_gen<__half, TBoxPlusOp, 5, 5>
{
    typedef TBoxPlusOp op_t;

    __device__ static void generate(word_t (&seq)[3], word_t (&app)[3])
    {
        const word_t& VAL_1_0 = app[0];
        const word_t& VAL_3_2 = app[1];
        const word_t& VAL_4   = app[2];
        word_t        VAL_4_4 = duplicate_low(VAL_4);
        word_t BP_13_02       = op_t::box_plus(VAL_3_2,    VAL_1_0);
        word_t BP_02_13       = swap_high_low(BP_13_02);
        word_t BP_024_134     = op_t::box_plus(BP_02_13,   VAL_4_4);
        word_t BP_0124_0134   = op_t::box_plus(BP_024_134, VAL_1_0);
        word_t BP_0234_1234   = op_t::box_plus_consume(BP_024_134, VAL_3_2);
        word_t BP_0123_0123   = op_t::box_plus(BP_13_02,   BP_02_13);
        seq[0] = BP_0234_1234;
        seq[1] = BP_0124_0134;
        seq[2].f16x2.x = BP_0123_0123.f16x2.x;
        seq[2].f16x2.y = 0;
        //printf("box_plus_seq_gen<5, 5>: inputs = (    %.0f) (%.0f %.0f) (%.0f %.0f), outputs = (    %.0f) (%.0f %.0f) (%.0f %.0f)\n",
        //       __low2float(app[2].f16x2),
        //       __high2float(app[1].f16x2),
        //       __low2float(app[1].f16x2),
        //       __high2float(app[0].f16x2),
        //       __low2float(app[0].f16x2),
        //       __low2float(seq[2].f16x2),
        //       __high2float(seq[1].f16x2),
        //       __low2float(seq[1].f16x2),
        //       __high2float(seq[0].f16x2),
        //       __low2float(seq[0].f16x2));
    }
};

////////////////////////////////////////////////////////////////////////
// box_plus_seq_gen<__half, 6, 5>()
// Input:
// [ VAL_5 | VAL_4 ] [ VAL_3 | VAL_2 ]  [ VAL_1 | VAL_0 ]
// Output:
// 0: [ box_plus(0, 2, 3, 4, 5) | box_plus(1, 2, 3, 4, 5) ]
// 1: [ box_plus(0, 1, 2, 4, 5) | box_plus(0, 1, 3, 4, 5) ]
// 2: [           0             | box_plus(0, 1, 2, 3, 5) ]
template <class TBoxPlusOp> struct box_plus_seq_gen<__half, TBoxPlusOp, 6, 5>
{
    typedef TBoxPlusOp op_t;

    __device__ static void generate(word_t (&seq)[3], word_t (&app)[3])
    {
        const word_t& VAL_1_0 = app[0];
        const word_t& VAL_3_2 = app[1];
        const word_t& VAL_5_4 = app[2];
        word_t BP_13_02       = op_t::box_plus(VAL_3_2,      VAL_1_0);
        word_t BP_135_024     = op_t::box_plus(BP_13_02,     VAL_5_4);
        word_t BP_024_135     = swap_high_low(BP_135_024);
        word_t BP_01234_01235 = op_t::box_plus(BP_024_135,   BP_13_02);
        word_t BP_0245_1345   = op_t::box_plus(BP_024_135,   VAL_5_4);
        word_t BP_01245_01345 = op_t::box_plus(BP_0245_1345, VAL_1_0);
        word_t BP_02345_12345 = op_t::box_plus_consume(BP_0245_1345, VAL_3_2);
        seq[0] = BP_02345_12345;
        seq[1] = BP_01245_01345;
        seq[2].f16x2.x = BP_01234_01235.f16x2.x;
        seq[2].f16x2.y = 0;
        //printf("box_plus_seq_gen<6, 5>: inputs = (%.0f %.0f)  (%.0f %.0f) (%.0f %.0f), outputs = ( %.0f) (%.0f %.0f) (%.0f %0.f)\n",
        //       __high2float(app[2].f16x2),
        //       __low2float(app[2].f16x2),
        //       __high2float(app[1].f16x2),
        //       __low2float(app[1].f16x2),
        //       __high2float(app[0].f16x2),
        //       __low2float(app[0].f16x2),
        //       __low2float(seq[2].f16x2),
        //       __high2float(seq[1].f16x2),
        //       __low2float(seq[1].f16x2),
        //       __high2float(seq[0].f16x2),
        //       __low2float(seq[0].f16x2));
    }
};

////////////////////////////////////////////////////////////////////////
// box_plus_seq_gen<__half, 6, 6>()
// Input:
// [ VAL_5 | VAL_4 ] [ VAL_3 | VAL_2 ]  [ VAL_1 | VAL_0 ]
// Output:
// 0: [ box_plus(0, 2, 3, 4, 5) | box_plus(1, 2, 3, 4, 5) ]
// 1: [ box_plus(0, 1, 2, 4, 5) | box_plus(0, 1, 3, 4, 5) ]
// 2: [ box_plus(0, 1, 2, 3, 4) | box_plus(0, 1, 2, 3, 5) ]
template <class TBoxPlusOp> struct box_plus_seq_gen<__half, TBoxPlusOp, 6, 6>
{
    typedef TBoxPlusOp op_t;

    __device__ static void generate(word_t (&seq)[3], word_t (&app)[3])
    {
        const word_t& VAL_1_0 = app[0];
        const word_t& VAL_3_2 = app[1];
        const word_t& VAL_5_4 = app[2];
        word_t BP_13_02       = op_t::box_plus(VAL_3_2,      VAL_1_0);
        word_t BP_135_024     = op_t::box_plus(BP_13_02,     VAL_5_4);
        word_t BP_024_135     = swap_high_low(BP_135_024);
        word_t BP_01234_01235 = op_t::box_plus(BP_024_135,   BP_13_02);
        word_t BP_0245_1345   = op_t::box_plus(BP_024_135,   VAL_5_4);
        word_t BP_01245_01345 = op_t::box_plus(BP_0245_1345, VAL_1_0);
        word_t BP_02345_12345 = op_t::box_plus_consume(BP_0245_1345, VAL_3_2);
        seq[0] = BP_02345_12345;
        seq[1] = BP_01245_01345;
        seq[2] = BP_01234_01235;
        //printf("box_plus_seq_gen<6, 6>: inputs = (%.0f %.0f)  (%.0f %.0f) (%.0f %.0f), outputs = (%.0f %.0f) (%.0f %.0f) (%.0f %0.f)\n",
        //       __high2float(app[2].f16x2),
        //       __low2float(app[2].f16x2),
        //       __high2float(app[1].f16x2),
        //       __low2float(app[1].f16x2),
        //       __high2float(app[0].f16x2),
        //       __low2float(app[0].f16x2),
        //       __high2float(seq[2].f16x2),
        //       __low2float(seq[2].f16x2),
        //       __high2float(seq[1].f16x2),
        //       __low2float(seq[1].f16x2),
        //       __high2float(seq[0].f16x2),
        //       __low2float(seq[0].f16x2));
    }
};

////////////////////////////////////////////////////////////////////////
// box_plus_seq_gen<__half, 7, 6>()
// Input:
// [ X | VAL_6 ] [ VAL_5 | VAL_4 ] [ VAL_3 | VAL_2 ]  [ VAL_1 | VAL_0 ]
// Output:
// 0: [ box_plus(0, 2, 3, 4, 5, 6) | box_plus(1, 2, 3, 4, 5, 6) ]
// 1: [ box_plus(0, 1, 2, 4, 5, 6) | box_plus(0, 1, 3, 4, 5, 6) ]
// 2: [ box_plus(0, 1, 2, 3, 4, 6) | box_plus(0, 1, 2, 3, 5, 6) ]
template <class TBoxPlusOp> struct box_plus_seq_gen<__half, TBoxPlusOp, 7, 6>
{
    typedef TBoxPlusOp op_t;

    __device__ static void generate(word_t (&seq)[3], word_t (&app)[4])
    {
        const word_t& VAL_1_0   = app[0];
        const word_t& VAL_3_2   = app[1];
        const word_t& VAL_5_4   = app[2];
        const word_t& VAL_6     = app[3];
        word_t VAL_6_6          = duplicate_low(VAL_6);
        word_t BP_13_02         = op_t::box_plus(VAL_3_2,        VAL_1_0);
        word_t BP_135_024       = op_t::box_plus(BP_13_02,       VAL_5_4);
        word_t BP_024_135       = swap_high_low(BP_135_024);
        word_t BP_0246_1356     = op_t::box_plus(BP_024_135, VAL_6_6);
        //word_t BP_012345_012345 = op_t::box_plus(BP_135_024, BP_024_135);
        word_t BP_012346_012356 = op_t::box_plus(BP_0246_1356, BP_13_02);
        word_t BP_02456_13456   = op_t::box_plus(BP_0246_1356, VAL_5_4);
        word_t BP_012456_013456 = op_t::box_plus(BP_02456_13456, VAL_1_0);
        word_t BP_023456_123456 = op_t::box_plus_consume(BP_02456_13456, VAL_3_2);
        seq[0] = BP_023456_123456;
        seq[1] = BP_012456_013456;
        seq[2] = BP_012346_012356;
        //seq[3].f16x2.x = BP_012345_012345.f16x2.x;
        //seq[3].f16x2.y = 0;
        //printf("box_plus_seq_gen<7, 7>: inputs = (%.0f) (%.0f %.0f)  (%.0f %.0f) (%.0f %.0f), outputs = (%.0f %.0f) (%.0f %.0f) (%.0f %0.f)\n",
        //       __low2float(app[3].f16x2),
        //       __high2float(app[2].f16x2),
        //       __low2float(app[2].f16x2),
        //       __high2float(app[1].f16x2),
        //       __low2float(app[1].f16x2),
        //       __high2float(app[0].f16x2),
        //       __low2float(app[0].f16x2),
        //       __high2float(seq[2].f16x2),
        //       __low2float(seq[2].f16x2),
        //       __high2float(seq[1].f16x2),
        //       __low2float(seq[1].f16x2),
        //       __high2float(seq[0].f16x2),
        //       __low2float(seq[0].f16x2));
    }
};

////////////////////////////////////////////////////////////////////////
// box_plus_seq_gen<__half, 7, 7>()
// Input:
// [ X | VAL_6 ] [ VAL_5 | VAL_4 ] [ VAL_3 | VAL_2 ]  [ VAL_1 | VAL_0 ]
// Output:
// 0: [ box_plus(0, 2, 3, 4, 5, 6) | box_plus(1, 2, 3, 4, 5, 6) ]
// 1: [ box_plus(0, 1, 2, 4, 5, 6) | box_plus(0, 1, 3, 4, 5, 6) ]
// 2: [ box_plus(0, 1, 2, 3, 4, 6) | box_plus(0, 1, 2, 3, 5, 6) ]
// 3: [                            | box_plus(0, 1, 2, 3, 4, 5) ]
template <class TBoxPlusOp> struct box_plus_seq_gen<__half, TBoxPlusOp, 7, 7>
{
    typedef TBoxPlusOp op_t;

    __device__ static void generate(word_t (&seq)[4], word_t (&app)[4])
    {
        const word_t& VAL_1_0   = app[0];
        const word_t& VAL_3_2   = app[1];
        const word_t& VAL_5_4   = app[2];
        const word_t& VAL_6     = app[3];
        word_t VAL_6_6          = duplicate_low(VAL_6);
        word_t BP_13_02         = op_t::box_plus(VAL_3_2,        VAL_1_0);
        word_t BP_135_024       = op_t::box_plus(BP_13_02,       VAL_5_4);
        word_t BP_024_135       = swap_high_low(BP_135_024);
        word_t BP_0246_1356     = op_t::box_plus(BP_024_135, VAL_6_6);
        word_t BP_012345_012345 = op_t::box_plus(BP_135_024, BP_024_135);
        word_t BP_012346_012356 = op_t::box_plus(BP_0246_1356, BP_13_02);
        word_t BP_02456_13456   = op_t::box_plus(BP_0246_1356, VAL_5_4);
        word_t BP_012456_013456 = op_t::box_plus(BP_02456_13456, VAL_1_0);
        word_t BP_023456_123456 = op_t::box_plus_consume(BP_02456_13456, VAL_3_2);
        seq[0] = BP_023456_123456;
        seq[1] = BP_012456_013456;
        seq[2] = BP_012346_012356;
        seq[3].f16x2.x = BP_012345_012345.f16x2.x;
        seq[3].f16x2.y = 0;
        //printf("box_plus_seq_gen<7, 7>: inputs = (%.0f) (%.0f %.0f)  (%.0f %.0f) (%.0f %.0f), outputs = (%.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %0.f)\n",
        //       __low2float(app[3].f16x2),
        //       __high2float(app[2].f16x2),
        //       __low2float(app[2].f16x2),
        //       __high2float(app[1].f16x2),
        //       __low2float(app[1].f16x2),
        //       __high2float(app[0].f16x2),
        //       __low2float(app[0].f16x2),
        //       __low2float(seq[3].f16x2),
        //       __high2float(seq[2].f16x2),
        //       __low2float(seq[2].f16x2),
        //       __high2float(seq[1].f16x2),
        //       __low2float(seq[1].f16x2),
        //       __high2float(seq[0].f16x2),
        //       __low2float(seq[0].f16x2));
    }
};


////////////////////////////////////////////////////////////////////////
// box_plus_seq_gen<__half, 8, 7>()
// Input:
// [ VAL_7 | VAL_6 ] [ VAL_5 | VAL_4 ] [ VAL_3 | VAL_2 ]  [ VAL_1 | VAL_0 ]
// Output:
// 0: [ box_plus(0, 2, 3, 4, 5, 6, 7) | box_plus(1, 2, 3, 4, 5, 6, 7) ]
// 1: [ box_plus(0, 1, 2, 4, 5, 6, 7) | box_plus(0, 1, 3, 4, 5, 6, 7) ]
// 2: [ box_plus(0, 1, 2, 3, 4, 6, 7) | box_plus(0, 1, 2, 3, 5, 6, 7) ]
// 3: [                0              | box_plus(0, 1, 2, 3, 4, 5, 7) ]
template <class TBoxPlusOp> struct box_plus_seq_gen<__half, TBoxPlusOp, 8, 7>
{
    typedef TBoxPlusOp op_t;

    __device__ static void generate(word_t (&seq)[4], word_t (&app)[4])
    {
        const word_t& VAL_1_0     = app[0];
        const word_t& VAL_3_2     = app[1];
        const word_t& VAL_5_4     = app[2];
        const word_t& VAL_7_6     = app[3];
        word_t BP_13_02           = op_t::box_plus(VAL_3_2,        VAL_1_0);
        word_t BP_57_46           = op_t::box_plus(VAL_7_6,        VAL_5_4);
        word_t BP_1357_0246       = op_t::box_plus(BP_13_02,       BP_57_46);
        word_t BP_0246_1357       = swap_high_low(BP_1357_0246);
        word_t BP_012346_012357   = op_t::box_plus(BP_0246_1357, BP_13_02);
        word_t BP_024567_134567   = op_t::box_plus(BP_0246_1357, BP_57_46);
        word_t BP_0123467_0123567 = op_t::box_plus(BP_012346_012357, VAL_7_6);
        word_t BP_0123456_0123457 = op_t::box_plus(BP_012346_012357, VAL_5_4);
        word_t BP_0124567_0134567 = op_t::box_plus(BP_024567_134567, VAL_1_0);
        word_t BP_0234567_1234567 = op_t::box_plus_consume(BP_024567_134567, VAL_3_2);
        seq[0] = BP_0234567_1234567;
        seq[1] = BP_0124567_0134567;
        seq[2] = BP_0123467_0123567;
        seq[3].f16x2.x = BP_0123456_0123457.f16x2.x;
        seq[3].f16x2.y = 0;
        //printf("box_plus_seq_gen<8, 7>: inputs = (%.0f %0.f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f), outputs = (%.0f %0.f) (%.0f %.0f) (%.0f %.0f) (%.0f %0.f)\n",
        //       __high2float(app[3].f16x2),
        //       __low2float(app[3].f16x2),
        //       __high2float(app[2].f16x2),
        //       __low2float(app[2].f16x2),
        //       __high2float(app[1].f16x2),
        //       __low2float(app[1].f16x2),
        //       __high2float(app[0].f16x2),
        //       __low2float(app[0].f16x2),
        //       __high2float(seq[3].f16x2),
        //       __low2float(seq[3].f16x2),
        //       __high2float(seq[2].f16x2),
        //       __low2float(seq[2].f16x2),
        //       __high2float(seq[1].f16x2),
        //       __low2float(seq[1].f16x2),
        //       __high2float(seq[0].f16x2),
        //       __low2float(seq[0].f16x2));
    }
};

////////////////////////////////////////////////////////////////////////
// box_plus_seq_gen<__half, 8, 8>()
// Input:
// [ VAL_7 | VAL_6 ] [ VAL_5 | VAL_4 ] [ VAL_3 | VAL_2 ]  [ VAL_1 | VAL_0 ]
// Output:
// 0: [ box_plus(0, 2, 3, 4, 5, 6, 7) | box_plus(1, 2, 3, 4, 5, 6, 7) ]
// 1: [ box_plus(0, 1, 2, 4, 5, 6, 7) | box_plus(0, 1, 3, 4, 5, 6, 7) ]
// 2: [ box_plus(0, 1, 2, 3, 4, 6, 7) | box_plus(0, 1, 2, 3, 5, 6, 7) ]
// 3: [ box_plus(0, 1, 2, 3, 4, 5, 6) | box_plus(0, 1, 2, 3, 4, 5, 7) ]
template <class TBoxPlusOp> struct box_plus_seq_gen<__half, TBoxPlusOp, 8, 8>
{
    typedef TBoxPlusOp op_t;

    __device__ static void generate(word_t (&seq)[4], word_t (&app)[4])
    {
        const word_t& VAL_1_0     = app[0];
        const word_t& VAL_3_2     = app[1];
        const word_t& VAL_5_4     = app[2];
        const word_t& VAL_7_6     = app[3];
        word_t BP_13_02           = op_t::box_plus(VAL_3_2,        VAL_1_0);
        word_t BP_57_46           = op_t::box_plus(VAL_7_6,        VAL_5_4);
        word_t BP_1357_0246       = op_t::box_plus(BP_13_02,       BP_57_46);
        word_t BP_0246_1357       = swap_high_low(BP_1357_0246);
        word_t BP_012346_012357   = op_t::box_plus(BP_0246_1357, BP_13_02);
        word_t BP_024567_134567   = op_t::box_plus(BP_0246_1357, BP_57_46);
        word_t BP_0123467_0123567 = op_t::box_plus(BP_012346_012357, VAL_7_6);
        word_t BP_0123456_0123457 = op_t::box_plus(BP_012346_012357, VAL_5_4);
        word_t BP_0124567_0134567 = op_t::box_plus(BP_024567_134567, VAL_1_0);
        word_t BP_0234567_1234567 = op_t::box_plus_consume(BP_024567_134567, VAL_3_2);
        seq[0] = BP_0234567_1234567;
        seq[1] = BP_0124567_0134567;
        seq[2] = BP_0123467_0123567;
        seq[3] = BP_0123456_0123457;
        //printf("box_plus_seq_gen<8, 8>: inputs = (%.0f %0.f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f), outputs = (%.0f %0.f) (%.0f %.0f) (%.0f %.0f) (%.0f %0.f)\n",
        //       __high2float(app[3].f16x2),
        //       __low2float(app[3].f16x2),
        //       __high2float(app[2].f16x2),
        //       __low2float(app[2].f16x2),
        //       __high2float(app[1].f16x2),
        //       __low2float(app[1].f16x2),
        //       __high2float(app[0].f16x2),
        //       __low2float(app[0].f16x2),
        //       __high2float(seq[3].f16x2),
        //       __low2float(seq[3].f16x2),
        //       __high2float(seq[2].f16x2),
        //       __low2float(seq[2].f16x2),
        //       __high2float(seq[1].f16x2),
        //       __low2float(seq[1].f16x2),
        //       __high2float(seq[0].f16x2),
        //       __low2float(seq[0].f16x2));
    }
};

////////////////////////////////////////////////////////////////////////
// box_plus_seq_gen<__half, 9, 8>()
// Input:
// [  X  | VAL_8 ] [ VAL_7 | VAL_6 ] [ VAL_5 | VAL_4 ] [ VAL_3 | VAL_2 ]  [ VAL_1 | VAL_0 ]
// Output:
// 0: [ box_plus(0, 2, 3, 4, 5, 6, 7, 8) | box_plus(1, 2, 3, 4, 5, 6, 7, 8) ]
// 1: [ box_plus(0, 1, 2, 4, 5, 6, 7, 8) | box_plus(0, 1, 3, 4, 5, 6, 7, 8) ]
// 2: [ box_plus(0, 1, 2, 3, 4, 6, 7, 8) | box_plus(0, 1, 2, 3, 5, 6, 7, 8) ]
// 3: [ box_plus(0, 1, 2, 3, 4, 5, 6, 8) | box_plus(0, 1, 2, 3, 4, 5, 7, 8) ]
template <class TBoxPlusOp> struct box_plus_seq_gen<__half, TBoxPlusOp, 9, 8>
{
    typedef TBoxPlusOp op_t;

    __device__ static void generate(word_t (&seq)[4], word_t (&app)[5])
    {
        const word_t& VAL_1_0       = app[0];
        const word_t& VAL_3_2       = app[1];
        const word_t& VAL_5_4       = app[2];
        const word_t& VAL_7_6       = app[3];
        const word_t& VAL_8         = app[4];
        word_t        VAL_8_8       = duplicate_low(VAL_8);
        word_t BP_13_02             = op_t::box_plus(VAL_3_2,            VAL_1_0);
        word_t BP_57_46             = op_t::box_plus(VAL_7_6,            VAL_5_4);
        word_t BP_1357_0246         = op_t::box_plus(BP_13_02,           BP_57_46);
        word_t BP_0246_1357         = swap_high_low(BP_1357_0246);
        //word_t BP_01234567_01234567 = op_t::box_plus(BP_0246_1357,       BP_1357_0246);
        word_t BP_02468_13578       = op_t::box_plus(BP_0246_1357,       VAL_8_8);
        word_t BP_0123468_0123578   = op_t::box_plus(BP_02468_13578,     BP_13_02);
        word_t BP_0245678_1345678   = op_t::box_plus(BP_02468_13578,     BP_57_46);
        word_t BP_01234678_01235678 = op_t::box_plus(BP_0123468_0123578, VAL_7_6);
        word_t BP_01234568_01234578 = op_t::box_plus(BP_0123468_0123578, VAL_5_4);
        word_t BP_01245678_01345678 = op_t::box_plus(BP_0245678_1345678, VAL_1_0);
        word_t BP_02345678_12345678 = op_t::box_plus_consume(BP_0245678_1345678, VAL_3_2);
        seq[0] = BP_02345678_12345678;
        seq[1] = BP_01245678_01345678;
        seq[2] = BP_01234678_01235678;
        seq[3] = BP_01234568_01234578;
        //seq[4].f16x2.x = BP_01234567_01234567.f16x2.x;
        //seq[4].f16x2.y = 0;
        //printf("box_plus_seq_gen<9, 9>: inputs = (%.0f) (%.0f %0.f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f), outputs = (%.0f) (%.0f %0.f) (%.0f %.0f) (%.0f %.0f) (%.0f %0.f)\n",
        //       __low2float(app[4].f16x2),
        //       __high2float(app[3].f16x2),
        //       __low2float(app[3].f16x2),
        //       __high2float(app[2].f16x2),
        //       __low2float(app[2].f16x2),
        //       __high2float(app[1].f16x2),
        //       __low2float(app[1].f16x2),
        //       __high2float(app[0].f16x2),
        //       __low2float(app[0].f16x2),
        //       __low2float(seq[4].f16x2),
        //       __high2float(seq[3].f16x2),
        //       __low2float(seq[3].f16x2),
        //       __high2float(seq[2].f16x2),
        //       __low2float(seq[2].f16x2),
        //       __high2float(seq[1].f16x2),
        //       __low2float(seq[1].f16x2),
        //       __high2float(seq[0].f16x2),
        //       __low2float(seq[0].f16x2));
    }
};

////////////////////////////////////////////////////////////////////////
// box_plus_seq_gen<__half, 9, 9>()
// Input:
// [  X  | VAL_8 ] [ VAL_7 | VAL_6 ] [ VAL_5 | VAL_4 ] [ VAL_3 | VAL_2 ]  [ VAL_1 | VAL_0 ]
// Output:
// 0: [ box_plus(0, 2, 3, 4, 5, 6, 7, 8) | box_plus(1, 2, 3, 4, 5, 6, 7, 8) ]
// 1: [ box_plus(0, 1, 2, 4, 5, 6, 7, 8) | box_plus(0, 1, 3, 4, 5, 6, 7, 8) ]
// 2: [ box_plus(0, 1, 2, 3, 4, 6, 7, 8) | box_plus(0, 1, 2, 3, 5, 6, 7, 8) ]
// 3: [ box_plus(0, 1, 2, 3, 4, 5, 6, 8) | box_plus(0, 1, 2, 3, 4, 5, 7, 8) ]
// 4: [                                  | box_plus(0, 1, 2, 3, 4, 5, 6, 7) ]
template <class TBoxPlusOp> struct box_plus_seq_gen<__half, TBoxPlusOp, 9, 9>
{
    typedef TBoxPlusOp op_t;

    __device__ static void generate(word_t (&seq)[5], word_t (&app)[5])
    {
        const word_t& VAL_1_0       = app[0];
        const word_t& VAL_3_2       = app[1];
        const word_t& VAL_5_4       = app[2];
        const word_t& VAL_7_6       = app[3];
        const word_t& VAL_8         = app[4];
        word_t        VAL_8_8       = duplicate_low(VAL_8);
        word_t BP_13_02             = op_t::box_plus(VAL_3_2,            VAL_1_0);
        word_t BP_57_46             = op_t::box_plus(VAL_7_6,            VAL_5_4);
        word_t BP_1357_0246         = op_t::box_plus(BP_13_02,           BP_57_46);
        word_t BP_0246_1357         = swap_high_low(BP_1357_0246);
        word_t BP_01234567_01234567 = op_t::box_plus(BP_0246_1357,       BP_1357_0246);
        word_t BP_02468_13578       = op_t::box_plus(BP_0246_1357,       VAL_8_8);
        word_t BP_0123468_0123578   = op_t::box_plus(BP_02468_13578,     BP_13_02);
        word_t BP_0245678_1345678   = op_t::box_plus(BP_02468_13578,     BP_57_46);
        word_t BP_01234678_01235678 = op_t::box_plus(BP_0123468_0123578, VAL_7_6);
        word_t BP_01234568_01234578 = op_t::box_plus(BP_0123468_0123578, VAL_5_4);
        word_t BP_01245678_01345678 = op_t::box_plus(BP_0245678_1345678, VAL_1_0);
        word_t BP_02345678_12345678 = op_t::box_plus_consume(BP_0245678_1345678, VAL_3_2);
        seq[0] = BP_02345678_12345678;
        seq[1] = BP_01245678_01345678;
        seq[2] = BP_01234678_01235678;
        seq[3] = BP_01234568_01234578;
        seq[4].f16x2.x = BP_01234567_01234567.f16x2.x;
        seq[4].f16x2.y = 0;
        //printf("box_plus_seq_gen<9, 9>: inputs = (%.0f) (%.0f %0.f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f), outputs = (%.0f) (%.0f %0.f) (%.0f %.0f) (%.0f %.0f) (%.0f %0.f)\n",
        //       __low2float(app[4].f16x2),
        //       __high2float(app[3].f16x2),
        //       __low2float(app[3].f16x2),
        //       __high2float(app[2].f16x2),
        //       __low2float(app[2].f16x2),
        //       __high2float(app[1].f16x2),
        //       __low2float(app[1].f16x2),
        //       __high2float(app[0].f16x2),
        //       __low2float(app[0].f16x2),
        //       __low2float(seq[4].f16x2),
        //       __high2float(seq[3].f16x2),
        //       __low2float(seq[3].f16x2),
        //       __high2float(seq[2].f16x2),
        //       __low2float(seq[2].f16x2),
        //       __high2float(seq[1].f16x2),
        //       __low2float(seq[1].f16x2),
        //       __high2float(seq[0].f16x2),
        //       __low2float(seq[0].f16x2));
    }
};

////////////////////////////////////////////////////////////////////////
// box_plus_seq_gen<__half, 10, 9>()
// Input:
// [ VAL_9 | VAL_8 ] [ VAL_7 | VAL_6 ] [ VAL_5 | VAL_4 ] [ VAL_3 | VAL_2 ]  [ VAL_1 | VAL_0 ]
// Output:
// 0: [ box_plus(0, 2, 3, 4, 5, 6, 7, 8, 9) | box_plus(1, 2, 3, 4, 5, 6, 7, 8, 9) ]
// 1: [ box_plus(0, 1, 2, 4, 5, 6, 7, 8, 9) | box_plus(0, 1, 3, 4, 5, 6, 7, 8, 9) ]
// 2: [ box_plus(0, 1, 2, 3, 4, 6, 7, 8, 9) | box_plus(0, 1, 2, 3, 5, 6, 7, 8, 9) ]
// 3: [ box_plus(0, 1, 2, 3, 4, 5, 6, 8, 9) | box_plus(0, 1, 2, 3, 4, 5, 7, 8, 9) ]
// 4: [                   0                 | box_plus(0, 1, 2, 3, 4, 5, 6, 7, 9) ]
template <class TBoxPlusOp> struct box_plus_seq_gen<__half, TBoxPlusOp, 10, 9>
{
    typedef TBoxPlusOp op_t;

    __device__ static void generate(word_t (&seq)[5], word_t (&app)[5])
    {
        const word_t& VAL_1_0         = app[0];
        const word_t& VAL_3_2         = app[1];
        const word_t& VAL_5_4         = app[2];
        const word_t& VAL_7_6         = app[3];
        const word_t& VAL_9_8         = app[4];
        word_t BP_13_02               = op_t::box_plus(VAL_3_2, VAL_1_0);
        word_t BP_57_46               = op_t::box_plus(VAL_5_4, VAL_7_6);

        word_t BP_1357_0246           = op_t::box_plus(BP_13_02, BP_57_46);
        word_t BP_13579_02468         = op_t::box_plus(BP_1357_0246, VAL_9_8);
        word_t BP_02468_13579         = swap_high_low(BP_13579_02468);
        word_t BP_024689_135789       = op_t::box_plus(BP_02468_13579, VAL_9_8);
        word_t BP_01234689_01235789   = op_t::box_plus(BP_024689_135789, BP_13_02);
        word_t BP_02456789_13456789   = op_t::box_plus(BP_024689_135789, BP_57_46);
    
        word_t BP_012345689_012345789 = op_t::box_plus(BP_01234689_01235789, VAL_5_4);
        word_t BP_012345678_012345679 = op_t::box_plus(BP_02468_13579, BP_1357_0246);
        word_t BP_012456789_013456789 = op_t::box_plus(BP_02456789_13456789, VAL_1_0);
        word_t BP_023456789_123456789 = op_t::box_plus(BP_02456789_13456789, VAL_3_2);
        word_t BP_012346789_012356789 = op_t::box_plus(BP_01234689_01235789, VAL_7_6);
    
        seq[0] = BP_023456789_123456789;
        seq[1] = BP_012456789_013456789;
        seq[2] = BP_012346789_012356789;
        seq[3] = BP_012345689_012345789;
        seq[4].f16x2.x = BP_012345678_012345679.f16x2.x;
        seq[4].f16x2.y = 0;

        //printf("box_plus_seq_gen<10, 10>: inputs = (%.0f %0.f) (%.0f %0.f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f), outputs = (%.0f %0.f) (%.0f %0.f) (%.0f %.0f) (%.0f %.0f) (%.0f %0.f)\n",
        //       __high2float(app[4].f16x2),
        //       __low2float(app[4].f16x2),
        //       __high2float(app[3].f16x2),
        //       __low2float(app[3].f16x2),
        //       __high2float(app[2].f16x2),
        //       __low2float(app[2].f16x2),
        //       __high2float(app[1].f16x2),
        //       __low2float(app[1].f16x2),
        //       __high2float(app[0].f16x2),
        //       __low2float(app[0].f16x2),
        //       __high2float(seq[4].f16x2),
        //       __low2float(seq[4].f16x2),
        //       __high2float(seq[3].f16x2),
        //       __low2float(seq[3].f16x2),
        //       __high2float(seq[2].f16x2),
        //       __low2float(seq[2].f16x2),
        //       __high2float(seq[1].f16x2),
        //       __low2float(seq[1].f16x2),
        //       __high2float(seq[0].f16x2),
        //       __low2float(seq[0].f16x2));
    }
};

////////////////////////////////////////////////////////////////////////
// box_plus_seq_gen<__half, 10, 10>()
// Input:
// [ VAL_9 | VAL_8 ] [ VAL_7 | VAL_6 ] [ VAL_5 | VAL_4 ] [ VAL_3 | VAL_2 ]  [ VAL_1 | VAL_0 ]
// Output:
// 0: [ box_plus(0, 2, 3, 4, 5, 6, 7, 8, 9) | box_plus(1, 2, 3, 4, 5, 6, 7, 8, 9) ]
// 1: [ box_plus(0, 1, 2, 4, 5, 6, 7, 8, 9) | box_plus(0, 1, 3, 4, 5, 6, 7, 8, 9) ]
// 2: [ box_plus(0, 1, 2, 3, 4, 6, 7, 8, 9) | box_plus(0, 1, 2, 3, 5, 6, 7, 8, 9) ]
// 3: [ box_plus(0, 1, 2, 3, 4, 5, 6, 8, 9) | box_plus(0, 1, 2, 3, 4, 5, 7, 8, 9) ]
// 4: [ box_plus(0, 1, 2, 3, 4, 5, 6, 7, 8) | box_plus(0, 1, 2, 3, 4, 5, 6, 7, 9) ]
template <class TBoxPlusOp> struct box_plus_seq_gen<__half, TBoxPlusOp, 10, 10>
{
    typedef TBoxPlusOp op_t;

    __device__ static void generate(word_t (&seq)[5], word_t (&app)[5])
    {
        const word_t& VAL_1_0         = app[0];
        const word_t& VAL_3_2         = app[1];
        const word_t& VAL_5_4         = app[2];
        const word_t& VAL_7_6         = app[3];
        const word_t& VAL_9_8         = app[4];
        word_t BP_13_02               = op_t::box_plus(VAL_3_2, VAL_1_0);
        word_t BP_57_46               = op_t::box_plus(VAL_5_4, VAL_7_6);

        word_t BP_1357_0246           = op_t::box_plus(BP_13_02, BP_57_46);
        word_t BP_13579_02468         = op_t::box_plus(BP_1357_0246, VAL_9_8);
        word_t BP_02468_13579         = swap_high_low(BP_13579_02468);
        word_t BP_024689_135789       = op_t::box_plus(BP_02468_13579, VAL_9_8);
        word_t BP_01234689_01235789   = op_t::box_plus(BP_024689_135789, BP_13_02);
        word_t BP_02456789_13456789   = op_t::box_plus(BP_024689_135789, BP_57_46);
    
        word_t BP_012345689_012345789 = op_t::box_plus(BP_01234689_01235789, VAL_5_4);
        word_t BP_012345678_012345679 = op_t::box_plus(BP_02468_13579, BP_1357_0246);
        word_t BP_012456789_013456789 = op_t::box_plus(BP_02456789_13456789, VAL_1_0);
        word_t BP_023456789_123456789 = op_t::box_plus(BP_02456789_13456789, VAL_3_2);
        word_t BP_012346789_012356789 = op_t::box_plus(BP_01234689_01235789, VAL_7_6);
    
        seq[0] = BP_023456789_123456789;
        seq[1] = BP_012456789_013456789;
        seq[2] = BP_012346789_012356789;
        seq[3] = BP_012345689_012345789;
        seq[4] = BP_012345678_012345679;
    
        //printf("box_plus_seq_gen<10, 10>: inputs = (%.0f %0.f) (%.0f %0.f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f), outputs = (%.0f %0.f) (%.0f %0.f) (%.0f %.0f) (%.0f %.0f) (%.0f %0.f)\n",
        //       __high2float(app[4].f16x2),
        //       __low2float(app[4].f16x2),
        //       __high2float(app[3].f16x2),
        //       __low2float(app[3].f16x2),
        //       __high2float(app[2].f16x2),
        //       __low2float(app[2].f16x2),
        //       __high2float(app[1].f16x2),
        //       __low2float(app[1].f16x2),
        //       __high2float(app[0].f16x2),
        //       __low2float(app[0].f16x2),
        //       __high2float(seq[4].f16x2),
        //       __low2float(seq[4].f16x2),
        //       __high2float(seq[3].f16x2),
        //       __low2float(seq[3].f16x2),
        //       __high2float(seq[2].f16x2),
        //       __low2float(seq[2].f16x2),
        //       __high2float(seq[1].f16x2),
        //       __low2float(seq[1].f16x2),
        //       __high2float(seq[0].f16x2),
        //       __low2float(seq[0].f16x2));
    }
};

////////////////////////////////////////////////////////////////////////
// box_plus_seq_gen<__half, 19, 19>()
// Input:
// [ X | VAL_18 ] [ VAL_17 | VAL_16 ] [ VAL_15 | VAL_14 ] [ VAL_13 | VAL_12 ]  [ VAL_11 | VAL_10 ] [ VAL_9 | VAL_8 ] [ VAL_7 | VAL_6 ] [ VAL_5 | VAL_4 ] [ VAL_3 | VAL_2 ]  [ VAL_1 | VAL_0 ]
// Output:
//  0: [ box_plus( 0,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18) | box_plus( 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18) ]
//  1: [ box_plus( 0,  1,  2,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18) | box_plus( 0,  1,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18) ]
//  2: [ box_plus( 0,  1,  2,  3,  4,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18) | box_plus( 0,  1,  2,  3,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18) ]
//  3: [ box_plus( 0,  1,  2,  3,  4,  5,  6,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18) | box_plus( 0,  1,  2,  3,  4,  5,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18) ]
//  4: [ box_plus( 0,  1,  2,  3,  4,  5,  6,  7,  8, 10, 11, 12, 13, 14, 15, 16, 17, 18) | box_plus( 0,  1,  2,  3,  4,  5,  6,  7,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18) ]
//  5: [ box_plus( 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 12, 13, 14, 15, 16, 17, 18) | box_plus( 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 11, 12, 13, 14, 15, 16, 17, 18) ]
//  6: [ box_plus( 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 14, 15, 16, 17, 18) | box_plus( 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 13, 14, 15, 16, 17, 18) ]
//  7: [ box_plus( 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 16, 17, 18) | box_plus( 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 15, 16, 17, 18) ]
//  8: [ box_plus( 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 18) | box_plus( 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 17, 18) ]
//  9: [                                          X                                       | box_plus( 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17) ]
// Theoretical number of box plus operation: 3N - 6 = 51
// Achieved number of 2-way fp16 box plus operations: 26 (+ 1 high/low swap operation)
template <class TBoxPlusOp> struct box_plus_seq_gen<__half, TBoxPlusOp, 19, 19>
{
    typedef TBoxPlusOp op_t;

    __device__ static void generate(word_t (&seq)[10], word_t (&app)[10])
    {
        const word_t& VAL_1_0         = app[0];
        const word_t& VAL_3_2         = app[1];
        const word_t& VAL_5_4         = app[2];
        const word_t& VAL_7_6         = app[3];
        const word_t& VAL_9_8         = app[4];
        const word_t& VAL_11_10       = app[5];
        const word_t& VAL_13_12       = app[6];
        const word_t& VAL_15_14       = app[7];
        const word_t& VAL_17_16       = app[8];
        const word_t& VAL_18          = app[9];
        word_t        VAL_18_18       = duplicate_low(VAL_18);

        word_t BP_1_3_x_0_2               = op_t::box_plus(VAL_3_2,   VAL_1_0);            // (1) 2 = 1 + 1
        word_t BP_5_7_x_4_6               = op_t::box_plus(VAL_5_4,   VAL_7_6);            // (2) 2 = 1 + 1
        word_t BP_9_11_x_8_10             = op_t::box_plus(VAL_9_8,   VAL_11_10);          // (3) 2 = 1 + 1
        word_t BP_13_15_x_12_14           = op_t::box_plus(VAL_13_12, VAL_15_14);          // (4) 2 = 1 + 1

        word_t BP_1_3_5_7_x_0_2_4_6       = op_t::box_plus(BP_1_3_x_0_2,   BP_5_7_x_4_6);     // (5) 4 = 2 + 2
        word_t BP_9_11_13_15_x_8_10_12_14 = op_t::box_plus(BP_9_11_x_8_10, BP_13_15_x_12_14); // (6) 4 = 2 + 2

        word_t BP_9_11_13_15_17_x_8_10_12_14_16 = op_t::box_plus(BP_9_11_13_15_x_8_10_12_14, VAL_17_16); // (7) 5 = 4 + 1

        word_t BP_1_3_5_7_9_11_13_15_17_x_0_2_4_6_8_10_12_14_16 = op_t::box_plus(BP_1_3_5_7_x_0_2_4_6, BP_9_11_13_15_17_x_8_10_12_14_16);  // (8) 9 = 5 + 4
        word_t BP_0_2_4_6_8_10_12_14_16_x_1_3_5_7_9_11_13_15_17 = swap_high_low(BP_1_3_5_7_9_11_13_15_17_x_0_2_4_6_8_10_12_14_16);

        // output 9 (without 18)
        word_t BP_0_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17_x_0_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17 = op_t::box_plus(BP_1_3_5_7_9_11_13_15_17_x_0_2_4_6_8_10_12_14_16,  // (9) 18 = 9 + 9
                                                                                                                             BP_0_2_4_6_8_10_12_14_16_x_1_3_5_7_9_11_13_15_17); 

        // without [(1, 3, 5, 7, 9, 11, 13, 15, 17), (0, 2, 4, 6, 8, 10, 12, 14, 16)]
        word_t BP_0_2_4_6_8_10_12_14_16_18_x_1_3_5_7_9_11_13_15_17_18 = op_t::box_plus(BP_0_2_4_6_8_10_12_14_16_x_1_3_5_7_9_11_13_15_17,  // (10)  10 = 9 + 1
                                                                                       VAL_18_18);

        // without [(9, 11, 13, 15, 17), (8, 10, 12, 14, 16)]
        word_t BP_0_1_2_3_4_5_6_7_8_10_12_14_16_18_x_0_1_2_3_4_5_6_7_9_11_13_15_17_18 = op_t::box_plus(BP_0_2_4_6_8_10_12_14_16_18_x_1_3_5_7_9_11_13_15_17_18,  // (11)  14 = 10 + 4
                                                                                                       BP_1_3_5_7_x_0_2_4_6);

        // without [(1, 3, 5, 7), (0, 2, 4, 6)]
        word_t BP_0_2_4_6_8_9_10_11_12_13_14_15_16_17_18_x_1_3_5_7_8_9_10_11_12_13_14_15_16_17_18 = op_t::box_plus(BP_0_2_4_6_8_10_12_14_16_18_x_1_3_5_7_9_11_13_15_17_18,  // (12) 14 = 10 + 4
                                                                                                                   BP_9_11_13_15_17_x_8_10_12_14_16);

        // without [(17), (16)] (output 8)
        word_t BP_0_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_18_x_0_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_17_18 = op_t::box_plus(BP_0_1_2_3_4_5_6_7_8_10_12_14_16_18_x_0_1_2_3_4_5_6_7_9_11_13_15_17_18,  // (13) 18 = 14 + 4
                                                                                                                             BP_9_11_13_15_x_8_10_12_14);

        // without [(9, 11, 13, 15), (8, 10, 12, 14)]
        word_t BP_0_1_2_3_4_5_6_7_8_10_12_14_16_17_18_x_0_1_2_3_4_5_6_7_9_11_13_15_16_17_18 = op_t::box_plus(BP_0_1_2_3_4_5_6_7_8_10_12_14_16_18_x_0_1_2_3_4_5_6_7_9_11_13_15_17_18,  // (14)  15 = 14 + 1
                                                                                                             VAL_17_16);

        // without [(13, 15), (12, 14)]
        word_t BP_0_1_2_3_4_5_6_7_8_9_10_11_12_14_16_17_18_x_0_1_2_3_4_5_6_7_8_9_10_11_13_15_16_17_18 = op_t::box_plus(BP_0_1_2_3_4_5_6_7_8_10_12_14_16_17_18_x_0_1_2_3_4_5_6_7_9_11_13_15_16_17_18,  // (15)  17 = 15 + 2
                                                                                                                       BP_9_11_x_8_10);

        // without [(15), (14)] (output 7)
        word_t BP_0_1_2_3_4_5_6_7_8_9_10_11_12_13_14_16_17_18_x_0_1_2_3_4_5_6_7_8_9_10_11_12_13_15_16_17_18 = op_t::box_plus(BP_0_1_2_3_4_5_6_7_8_9_10_11_12_14_16_17_18_x_0_1_2_3_4_5_6_7_8_9_10_11_13_15_16_17_18,  // (16)  18 = 17 + 1
                                                                                                                             VAL_13_12);

        // without [(13), (12)] (output 6)
        word_t BP_0_1_2_3_4_5_6_7_8_9_10_11_12_14_15_16_17_18_x_0_1_2_3_4_5_6_7_8_9_10_11_13_14_15_16_17_18 = op_t::box_plus(BP_0_1_2_3_4_5_6_7_8_9_10_11_12_14_16_17_18_x_0_1_2_3_4_5_6_7_8_9_10_11_13_15_16_17_18,  // (17)  18 = 17 + 1
                                                                                                                             VAL_15_14);

        // without [(9, 11), (8, 10)]
        word_t BP_0_1_2_3_4_5_6_7_8_10_12_13_14_15_16_17_18_x_0_1_2_3_4_5_6_7_9_11_12_13_14_15_16_17_18 = op_t::box_plus(BP_0_1_2_3_4_5_6_7_8_10_12_14_16_17_18_x_0_1_2_3_4_5_6_7_9_11_13_15_16_17_18,  // (18)  17 = 15 + 2
                                                                                                                         BP_13_15_x_12_14);

        // without [(11), (10)] (output 5)
        word_t BP_0_1_2_3_4_5_6_7_8_9_10_12_13_14_15_16_17_18_x_0_1_2_3_4_5_6_7_8_9_11_12_13_14_15_16_17_18 = op_t::box_plus(BP_0_1_2_3_4_5_6_7_8_10_12_13_14_15_16_17_18_x_0_1_2_3_4_5_6_7_9_11_12_13_14_15_16_17_18,  // (19)  18 = 17 + 1
                                                                                                                             VAL_9_8);

        // without [(9), (8)] (output 4)
        word_t BP_0_1_2_3_4_5_6_7_8_10_11_12_13_14_15_16_17_18_x_0_1_2_3_4_5_6_7_9_10_11_12_13_14_15_16_17_18 = op_t::box_plus(BP_0_1_2_3_4_5_6_7_8_10_12_13_14_15_16_17_18_x_0_1_2_3_4_5_6_7_9_11_12_13_14_15_16_17_18,  // (20)  18 = 17 + 1
                                                                                                                               VAL_11_10);

        // without [(5, 7), (4, 6)]
        word_t BP_0_1_2_3_4_6_8_9_10_11_12_13_14_15_16_17_18_x_0_1_2_3_5_7_8_9_10_11_12_13_14_15_16_17_18 = op_t::box_plus(BP_0_2_4_6_8_9_10_11_12_13_14_15_16_17_18_x_1_3_5_7_8_9_10_11_12_13_14_15_16_17_18,  // (21) 17 = 15 + 2
                                                                                                                           BP_1_3_x_0_2);

        // without [(7), (6)] (output 3)
        word_t BP_0_1_2_3_4_5_6_8_9_10_11_12_13_14_15_16_17_18_x_0_1_2_3_4_5_7_8_9_10_11_12_13_14_15_16_17_18 = op_t::box_plus(BP_0_1_2_3_4_6_8_9_10_11_12_13_14_15_16_17_18_x_0_1_2_3_5_7_8_9_10_11_12_13_14_15_16_17_18,  // (22) 18 = 17 + 1
                                                                                                                               VAL_5_4);

        // without [(5), (4)] (output 2)
        word_t BP_0_1_2_3_4_6_7_8_9_10_11_12_13_14_15_16_17_18_x_0_1_2_3_5_6_7_8_9_10_11_12_13_14_15_16_17_18 = op_t::box_plus(BP_0_1_2_3_4_6_8_9_10_11_12_13_14_15_16_17_18_x_0_1_2_3_5_7_8_9_10_11_12_13_14_15_16_17_18,  // (23) 18 = 17 + 1
                                                                                                                               VAL_7_6);

        // without [(1, 3), (0, 2)]
        word_t BP_0_2_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18_x_1_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18 = op_t::box_plus(BP_0_2_4_6_8_9_10_11_12_13_14_15_16_17_18_x_1_3_5_7_8_9_10_11_12_13_14_15_16_17_18,  // (24) 17 = 15 + 2
                                                                                                                           BP_5_7_x_4_6);

        // without [(3), (2)] (output 1)
        word_t BP_0_1_2_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18_x_0_1_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18 = op_t::box_plus(BP_0_2_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18_x_1_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18,  // (25) 18 = 17 + 1
                                                                                                                               VAL_1_0);

        // without [(1), (0)] (output 0)
        word_t BP_0_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18_x_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18 = op_t::box_plus(BP_0_2_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18_x_1_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18,  // (26) 18 = 17 + 1
                                                                                                                               VAL_3_2);

        seq[0]         = BP_0_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18_x_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18;
        seq[1]         = BP_0_1_2_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18_x_0_1_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18;
        seq[2]         = BP_0_1_2_3_4_6_7_8_9_10_11_12_13_14_15_16_17_18_x_0_1_2_3_5_6_7_8_9_10_11_12_13_14_15_16_17_18;
        seq[3]         = BP_0_1_2_3_4_5_6_8_9_10_11_12_13_14_15_16_17_18_x_0_1_2_3_4_5_7_8_9_10_11_12_13_14_15_16_17_18;
        seq[4]         = BP_0_1_2_3_4_5_6_7_8_10_11_12_13_14_15_16_17_18_x_0_1_2_3_4_5_6_7_9_10_11_12_13_14_15_16_17_18;
        seq[5]         = BP_0_1_2_3_4_5_6_7_8_9_10_12_13_14_15_16_17_18_x_0_1_2_3_4_5_6_7_8_9_11_12_13_14_15_16_17_18;
        seq[6]         = BP_0_1_2_3_4_5_6_7_8_9_10_11_12_14_15_16_17_18_x_0_1_2_3_4_5_6_7_8_9_10_11_13_14_15_16_17_18;
        seq[7]         = BP_0_1_2_3_4_5_6_7_8_9_10_11_12_13_14_16_17_18_x_0_1_2_3_4_5_6_7_8_9_10_11_12_13_15_16_17_18;
        seq[8]         = BP_0_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_18_x_0_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_17_18;
        seq[9].f16x2.x = BP_0_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17_x_0_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17.f16x2.x;
        seq[9].f16x2.y = 0;
        //printf("box_plus_seq_gen<19, 19>: inputs = (%0.f) (%.0f %0.f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %0.f) (%.0f %0.f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f), "
        //       "outputs = (%.0f %0.f) (%.0f %0.f) (%.0f %.0f) (%.0f %.0f) (%.0f %0.f) (%.0f %0.f) (%.0f %0.f) (%.0f %.0f) (%.0f %.0f) (%.0f %0.f)\n",
        //       __low2float(app[9].f16x2),
        //       __high2float(app[8].f16x2),
        //       __low2float(app[8].f16x2),
        //       __high2float(app[7].f16x2),
        //       __low2float(app[7].f16x2),
        //       __high2float(app[6].f16x2),
        //       __low2float(app[6].f16x2),
        //       __high2float(app[5].f16x2),
        //       __low2float(app[5].f16x2),
        //       __high2float(app[4].f16x2),
        //       __low2float(app[4].f16x2),
        //       __high2float(app[3].f16x2),
        //       __low2float(app[3].f16x2),
        //       __high2float(app[2].f16x2),
        //       __low2float(app[2].f16x2),
        //       __high2float(app[1].f16x2),
        //       __low2float(app[1].f16x2),
        //       __high2float(app[0].f16x2),
        //       __low2float(app[0].f16x2),
        //       __high2float(seq[9].f16x2),
        //       __low2float(seq[9].f16x2),
        //       __high2float(seq[8].f16x2),
        //       __low2float(seq[8].f16x2),
        //       __high2float(seq[7].f16x2),
        //       __low2float(seq[7].f16x2),
        //       __high2float(seq[6].f16x2),
        //       __low2float(seq[6].f16x2),
        //       __high2float(seq[5].f16x2),
        //       __low2float(seq[5].f16x2),
        //       __high2float(seq[4].f16x2),
        //       __low2float(seq[4].f16x2),
        //       __high2float(seq[3].f16x2),
        //       __low2float(seq[3].f16x2),
        //       __high2float(seq[2].f16x2),
        //       __low2float(seq[2].f16x2),
        //       __high2float(seq[1].f16x2),
        //       __low2float(seq[1].f16x2),
        //       __high2float(seq[0].f16x2),
        //       __low2float(seq[0].f16x2));
    }
};

////////////////////////////////////////////////////////////////////////
// box_plus_seq_gen<__half2, 3, 2>()
// Input:
// [ VAL2.HI,LO ] [ VAL1.HI,LO ] [ VAL0.HI,LO ]
// Output:
// [ BP(0,2).hi,lo ] [ BP(1,2).hi,lo ]
template <class TBoxPlusOp> struct box_plus_seq_gen<__half2, TBoxPlusOp, 3, 2>
{
    typedef TBoxPlusOp op_t;

    __device__ static void generate(word_t (&seq)[2], word_t (&app)[3])
    {
        const word_t& VAL_0    = app[0];
        const word_t& VAL_1    = app[1];
        const word_t& VAL_2    = app[2];
        word_t        BP_02    = op_t::box_plus(VAL_0, VAL_2);
        word_t        BP_12    = op_t::box_plus(VAL_1, VAL_2);
        seq[0] = BP_12;
        seq[1] = BP_02;
        //printf("box_plus_seq_gen<3, 2>: inputs = (%.0f %.0f) (%.0f %.0f) (%.0f %.0f), outputs = (%.0f %.0f) (%.0f %.0f)\n",
        //       __low2float(app[2].f16x2),
        //       __high2float(app[2].f16x2),
        //       __low2float(app[1].f16x2),
        //       __high2float(app[1].f16x2),
        //       __low2float(app[0].f16x2),
        //       __high2float(app[0].f16x2),
        //       __high2float(seq[1].f16x2),
        //       __low2float(seq[1].f16x2),
        //       __high2float(seq[0].f16x2),
        //       __low2float(seq[0].f16x2));
    }
};

////////////////////////////////////////////////////////////////////////
// box_plus_seq_gen<__half2, 3, 3>()
// Input:
// [ VAL2.HI,LO ] [ VAL1.HI,LO ] [ VAL0.HI,LO ]
// Output:
// [ BP(0,1).hi,lo ] [ BP(0,2).hi,lo ] [ BP(1,2).hi,lo ]
template <class TBoxPlusOp> struct box_plus_seq_gen<__half2, TBoxPlusOp, 3, 3>
{
    typedef TBoxPlusOp op_t;

    __device__ static void generate(word_t (&seq)[3], word_t (&app)[3])
    {
        const word_t& VAL_0    = app[0];
        const word_t& VAL_1    = app[1];
        const word_t& VAL_2    = app[2];
        word_t        BP_01    = op_t::box_plus(VAL_0, VAL_1);
        word_t        BP_02    = op_t::box_plus(VAL_0, VAL_2);
        word_t        BP_12    = op_t::box_plus(VAL_1, VAL_2);
        seq[0] = BP_12;
        seq[1] = BP_02;
        seq[2] = BP_01;
        //printf("box_plus_seq_gen<3, 3>: inputs = (%.0f %.0f) (%.0f %.0f) (%.0f %.0f), outputs = (%.0f %.0f) (%.0f %.0f) (%.0f %.0f)\n",
        //       __low2float(app[2].f16x2),
        //       __high2float(app[2].f16x2),
        //       __low2float(app[1].f16x2),
        //       __high2float(app[1].f16x2),
        //       __low2float(app[0].f16x2),
        //       __high2float(app[0].f16x2),
        //       __high2float(seq[2].f16x2),
        //       __low2float(seq[2].f16x2),
        //       __high2float(seq[1].f16x2),
        //       __low2float(seq[1].f16x2),
        //       __high2float(seq[0].f16x2),
        //       __low2float(seq[0].f16x2));
    }
};

////////////////////////////////////////////////////////////////////////
// box_plus_seq_gen<__half2, 4, 3>()
// Input:
// [ VAL3.HI,LO ] [ VAL2.HI,LO ] [ VAL1.HI,LO ] [ VAL0.HI,LO ]
// Output:
// [ BP(0,1,3).hi,lo ] [ BP(0,2,3).hi,lo ] [ BP(1,2,3).hi,lo ]
template <class TBoxPlusOp> struct box_plus_seq_gen<__half2, TBoxPlusOp, 4, 3>
{
    typedef TBoxPlusOp op_t;

    __device__ static void generate(word_t (&seq)[3], word_t (&app)[4])
    {
        const word_t& VAL_0 = app[0];
        const word_t& VAL_1 = app[1];
        const word_t& VAL_2 = app[2];
        const word_t& VAL_3 = app[3];

        word_t BP_13  = op_t::box_plus(VAL_1, VAL_3);
        word_t BP_02  = op_t::box_plus(VAL_0, VAL_2);

        word_t BP_123 = op_t::box_plus(BP_13, VAL_2);
        word_t BP_023 = op_t::box_plus(BP_02, VAL_3);
        word_t BP_013 = op_t::box_plus(BP_13, VAL_0);
        
        seq[0] = BP_123;
        seq[1] = BP_023;
        seq[2] = BP_013;
        //printf("box_plus_seq_gen<4, 3>: inputs = (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f), outputs = (%.0f %.0f) (%.0f %.0f) (%.0f %.0f)\n",
        //       __low2float(app[3].f16x2),
        //       __high2float(app[3].f16x2),
        //       __low2float(app[2].f16x2),
        //       __high2float(app[2].f16x2),
        //       __low2float(app[1].f16x2),
        //       __high2float(app[1].f16x2),
        //       __low2float(app[0].f16x2),
        //       __high2float(app[0].f16x2),
        //       __high2float(seq[2].f16x2),
        //       __low2float(seq[2].f16x2),
        //       __high2float(seq[1].f16x2),
        //       __low2float(seq[1].f16x2),
        //       __high2float(seq[0].f16x2),
        //       __low2float(seq[0].f16x2));
    }
};

////////////////////////////////////////////////////////////////////////
// box_plus_seq_gen<__half2, 4, 4>()
// Input:
// [ VAL3.HI,LO ] [ VAL2.HI,LO ] [ VAL1.HI,LO ] [ VAL0.HI,LO ]
// Output:
// [ BP(0,1,2).hi,lo ] [ BP(0,1,3).hi,lo ] [ BP(0,2,3).hi,lo ] [ BP(1,2,3).hi,lo ]
template <class TBoxPlusOp> struct box_plus_seq_gen<__half2, TBoxPlusOp, 4, 4>
{
    typedef TBoxPlusOp op_t;

    __device__ static void generate(word_t (&seq)[4], word_t (&app)[4])
    {
        const word_t& VAL_0 = app[0];
        const word_t& VAL_1 = app[1];
        const word_t& VAL_2 = app[2];
        const word_t& VAL_3 = app[3];

        word_t BP_13  = op_t::box_plus(VAL_1, VAL_3);
        word_t BP_02  = op_t::box_plus(VAL_0, VAL_2);

        word_t BP_123 = op_t::box_plus(BP_13, VAL_2);
        word_t BP_023 = op_t::box_plus(BP_02, VAL_3);
        word_t BP_013 = op_t::box_plus(BP_13, VAL_0);
        word_t BP_012 = op_t::box_plus(BP_02, VAL_1);

        seq[0] = BP_123;
        seq[1] = BP_023;
        seq[2] = BP_013;
        seq[3] = BP_012;
        //printf("box_plus_seq_gen<4, 4>: inputs = (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f), outputs = (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f)\n",
        //       __low2float(app[3].f16x2),
        //       __high2float(app[3].f16x2),
        //       __low2float(app[2].f16x2),
        //       __high2float(app[2].f16x2),
        //       __low2float(app[1].f16x2),
        //       __high2float(app[1].f16x2),
        //       __low2float(app[0].f16x2),
        //       __high2float(app[0].f16x2),
        //       __high2float(seq[3].f16x2),
        //       __low2float(seq[3].f16x2),
        //       __high2float(seq[2].f16x2),
        //       __low2float(seq[2].f16x2),
        //       __high2float(seq[1].f16x2),
        //       __low2float(seq[1].f16x2),
        //       __high2float(seq[0].f16x2),
        //       __low2float(seq[0].f16x2));
    }
};

////////////////////////////////////////////////////////////////////////
// box_plus_seq_gen<__half2, 5, 4>()
// Input:
// [ VAL4.HI,LO ] [ VAL3.HI,LO ] [ VAL2.HI,LO ] [ VAL1.HI,LO ] [ VAL0.HI,LO ]
// Output:
// [ BP(0,1,2,4).hi,lo ] [ BP(0,1,3,4).hi,lo ] [ BP(0,2,3,4).hi,lo ] [ BP(1,2,3,4).hi,lo ]
template <class TBoxPlusOp> struct box_plus_seq_gen<__half2, TBoxPlusOp, 5, 4>
{
    typedef TBoxPlusOp op_t;

    __device__ static void generate(word_t (&seq)[4], word_t (&app)[5])
    {
        const word_t& VAL_0 = app[0];
        const word_t& VAL_1 = app[1];
        const word_t& VAL_2 = app[2];
        const word_t& VAL_3 = app[3];
        const word_t& VAL_4 = app[4];
        
        word_t BP_13  = op_t::box_plus(VAL_1, VAL_3);
        word_t BP_02  = op_t::box_plus(VAL_0, VAL_2);
        word_t BP_134 = op_t::box_plus(BP_13, VAL_4);
        word_t BP_024 = op_t::box_plus(BP_02, VAL_4);

        word_t BP_1234 = op_t::box_plus(BP_134, VAL_2);
        word_t BP_0234 = op_t::box_plus(BP_024, VAL_3);
        word_t BP_0134 = op_t::box_plus(BP_134, VAL_0);
        word_t BP_0124 = op_t::box_plus(BP_024, VAL_1);

        seq[0] = BP_1234;
        seq[1] = BP_0234;
        seq[2] = BP_0134;
        seq[3] = BP_0124;
        //printf("box_plus_seq_gen<5, 4>: inputs = (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f), "
        //       "outputs = (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f)\n",
        //       __low2float(app[4].f16x2),
        //       __high2float(app[4].f16x2),
        //       __low2float(app[3].f16x2),
        //       __high2float(app[3].f16x2),
        //       __low2float(app[2].f16x2),
        //       __high2float(app[2].f16x2),
        //       __low2float(app[1].f16x2),
        //       __high2float(app[1].f16x2),
        //       __low2float(app[0].f16x2),
        //       __high2float(app[0].f16x2),
        //       __high2float(seq[3].f16x2),
        //       __low2float(seq[3].f16x2),
        //       __high2float(seq[2].f16x2),
        //       __low2float(seq[2].f16x2),
        //       __high2float(seq[1].f16x2),
        //       __low2float(seq[1].f16x2),
        //       __high2float(seq[0].f16x2),
        //       __low2float(seq[0].f16x2));
    }
};

////////////////////////////////////////////////////////////////////////
// box_plus_seq_gen<__half2, 5, 5>()
// Input:
// [ VAL4.HI,LO ] [ VAL3.HI,LO ] [ VAL2.HI,LO ] [ VAL1.HI,LO ] [ VAL0.HI,LO ]
// Output:
// [ BP(0,1,2,3).hi,lo ] [ BP(0,1,2,4).hi,lo ] [ BP(0,1,3,4).hi,lo ] [ BP(0,2,3,4).hi,lo ] [ BP(1,2,3,4).hi,lo ]
template <class TBoxPlusOp> struct box_plus_seq_gen<__half2, TBoxPlusOp, 5, 5>
{
    typedef TBoxPlusOp op_t;

    __device__ static void generate(word_t (&seq)[5], word_t (&app)[5])
    {
        const word_t& VAL_0 = app[0];
        const word_t& VAL_1 = app[1];
        const word_t& VAL_2 = app[2];
        const word_t& VAL_3 = app[3];
        const word_t& VAL_4 = app[4];
        
        word_t BP_13  = op_t::box_plus(VAL_1, VAL_3);
        word_t BP_02  = op_t::box_plus(VAL_0, VAL_2);
        word_t BP_134 = op_t::box_plus(BP_13, VAL_4);
        word_t BP_024 = op_t::box_plus(BP_02, VAL_4);

        word_t BP_1234 = op_t::box_plus(BP_134, VAL_2);
        word_t BP_0234 = op_t::box_plus(BP_024, VAL_3);
        word_t BP_0134 = op_t::box_plus(BP_134, VAL_0);
        word_t BP_0124 = op_t::box_plus(BP_024, VAL_1);
        word_t BP_0123 = op_t::box_plus(BP_02,  BP_13);

        seq[0] = BP_1234;
        seq[1] = BP_0234;
        seq[2] = BP_0134;
        seq[3] = BP_0124;
        seq[4] = BP_0123;
        //printf("box_plus_seq_gen<5, 5>: inputs = (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f), "
        //       "outputs = (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f)\n",
        //       __low2float(app[4].f16x2),
        //       __high2float(app[4].f16x2),
        //       __low2float(app[3].f16x2),
        //       __high2float(app[3].f16x2),
        //       __low2float(app[2].f16x2),
        //       __high2float(app[2].f16x2),
        //       __low2float(app[1].f16x2),
        //       __high2float(app[1].f16x2),
        //       __low2float(app[0].f16x2),
        //       __high2float(app[0].f16x2),
        //       __high2float(seq[4].f16x2),
        //       __low2float(seq[4].f16x2),
        //       __high2float(seq[3].f16x2),
        //       __low2float(seq[3].f16x2),
        //       __high2float(seq[2].f16x2),
        //       __low2float(seq[2].f16x2),
        //       __high2float(seq[1].f16x2),
        //       __low2float(seq[1].f16x2),
        //       __high2float(seq[0].f16x2),
        //       __low2float(seq[0].f16x2));
    }
};

////////////////////////////////////////////////////////////////////////
// box_plus_seq_gen<__half2, 6, 5>()
// Input:
//                                                             [ VAL5.HI,LO ]
// [ VAL4.HI,LO ] [ VAL3.HI,LO ] [ VAL2.HI,LO ] [ VAL1.HI,LO ] [ VAL0.HI,LO ]
// Output:
// [ BP(0,1,2,3,5).hi,lo ] [ BP(0,1,2,4,5).hi,lo ] [ BP(0,1,3,4,5).hi,lo ] [ BP(0,2,3,4,5).hi,lo ] [ BP(1,2,3,4,5).hi,lo ]
template <class TBoxPlusOp> struct box_plus_seq_gen<__half2, TBoxPlusOp, 6, 5>
{
    typedef TBoxPlusOp op_t;

    __device__ static void generate(word_t (&seq)[5], word_t (&app)[6])
    {
        const word_t& VAL_0 = app[0];
        const word_t& VAL_1 = app[1];
        const word_t& VAL_2 = app[2];
        const word_t& VAL_3 = app[3];
        const word_t& VAL_4 = app[4];
        const word_t& VAL_5 = app[5];
        
        word_t BP_13   = op_t::box_plus(VAL_1, VAL_3);
        word_t BP_02   = op_t::box_plus(VAL_0, VAL_2);
        word_t BP_45   = op_t::box_plus(VAL_4, VAL_5);
        word_t BP_1345 = op_t::box_plus(BP_13, BP_45);
        word_t BP_0245 = op_t::box_plus(BP_02, BP_45);
        word_t BP_0123 = op_t::box_plus(BP_02, BP_13);

        word_t BP_12345 = op_t::box_plus(BP_1345, VAL_2);
        word_t BP_02345 = op_t::box_plus(BP_0245, VAL_3);
        word_t BP_01345 = op_t::box_plus(BP_1345, VAL_0);
        word_t BP_01245 = op_t::box_plus(BP_0245, VAL_1);
        word_t BP_01235 = op_t::box_plus(BP_0123, VAL_5);

        seq[0] = BP_12345;
        seq[1] = BP_02345;
        seq[2] = BP_01345;
        seq[3] = BP_01245;
        seq[4] = BP_01235;
        //printf("box_plus_seq_gen<6, 5>: inputs = (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f), "
        //       "outputs = (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f)\n",
        //       __low2float(app[5].f16x2),
        //       __high2float(app[5].f16x2),
        //       __low2float(app[4].f16x2),
        //       __high2float(app[4].f16x2),
        //       __low2float(app[3].f16x2),
        //       __high2float(app[3].f16x2),
        //       __low2float(app[2].f16x2),
        //       __high2float(app[2].f16x2),
        //       __low2float(app[1].f16x2),
        //       __high2float(app[1].f16x2),
        //       __low2float(app[0].f16x2),
        //       __high2float(app[0].f16x2),
        //       __high2float(seq[4].f16x2),
        //       __low2float(seq[4].f16x2),
        //       __high2float(seq[3].f16x2),
        //       __low2float(seq[3].f16x2),
        //       __high2float(seq[2].f16x2),
        //       __low2float(seq[2].f16x2),
        //       __high2float(seq[1].f16x2),
        //       __low2float(seq[1].f16x2),
        //       __high2float(seq[0].f16x2),
        //       __low2float(seq[0].f16x2));
    }
};

////////////////////////////////////////////////////////////////////////
// box_plus_seq_gen<__half2, 6, 6>()
// Input:
// [ VAL5.HI,LO ] [ VAL4.HI,LO ] [ VAL3.HI,LO ] [ VAL2.HI,LO ] [ VAL_1.HI,LO ] [ VAL_0.HI,LO ]
// Output:
// [ BP(0,1,2,3,4).hi,lo ] [ BP(0,1,2,3,5).hi,lo ] [ BP(0,1,2,4,5).hi,lo ] [ BP(0,1,3,4,5).hi,lo ] [ BP(0,2,3,4,5).hi,lo ] [ BP(1,2,3,4,5).hi,lo ]
template <class TBoxPlusOp> struct box_plus_seq_gen<__half2, TBoxPlusOp, 6, 6>
{
    typedef TBoxPlusOp op_t;

    __device__ static void generate(word_t (&seq)[6], word_t (&app)[6])
    {
        const word_t& VAL_0 = app[0];
        const word_t& VAL_1 = app[1];
        const word_t& VAL_2 = app[2];
        const word_t& VAL_3 = app[3];
        const word_t& VAL_4 = app[4];
        const word_t& VAL_5 = app[5];
        
        word_t BP_13   = op_t::box_plus(VAL_1, VAL_3);
        word_t BP_02   = op_t::box_plus(VAL_0, VAL_2);
        word_t BP_45   = op_t::box_plus(VAL_4, VAL_5);
        word_t BP_1345 = op_t::box_plus(BP_13, BP_45);
        word_t BP_0245 = op_t::box_plus(BP_02, BP_45);
        word_t BP_0123 = op_t::box_plus(BP_02, BP_13);

        word_t BP_12345 = op_t::box_plus(BP_1345, VAL_2);
        word_t BP_02345 = op_t::box_plus(BP_0245, VAL_3);
        word_t BP_01345 = op_t::box_plus(BP_1345, VAL_0);
        word_t BP_01245 = op_t::box_plus(BP_0245, VAL_1);
        word_t BP_01235 = op_t::box_plus(BP_0123, VAL_5);
        word_t BP_01234 = op_t::box_plus(BP_0123, VAL_4);

        seq[0] = BP_12345;
        seq[1] = BP_02345;
        seq[2] = BP_01345;
        seq[3] = BP_01245;
        seq[4] = BP_01235;
        seq[5] = BP_01234;
        //printf("box_plus_seq_gen<6, 6>: inputs = (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f), "
        //       "outputs = (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f)\n",
        //       __low2float(app[5].f16x2),
        //       __high2float(app[5].f16x2),
        //       __low2float(app[4].f16x2),
        //       __high2float(app[4].f16x2),
        //       __low2float(app[3].f16x2),
        //       __high2float(app[3].f16x2),
        //       __low2float(app[2].f16x2),
        //       __high2float(app[2].f16x2),
        //       __low2float(app[1].f16x2),
        //       __high2float(app[1].f16x2),
        //       __low2float(app[0].f16x2),
        //       __high2float(app[0].f16x2),
        //       __high2float(seq[5].f16x2),
        //       __low2float(seq[5].f16x2),
        //       __high2float(seq[4].f16x2),
        //       __low2float(seq[4].f16x2),
        //       __high2float(seq[3].f16x2),
        //       __low2float(seq[3].f16x2),
        //       __high2float(seq[2].f16x2),
        //       __low2float(seq[2].f16x2),
        //       __high2float(seq[1].f16x2),
        //       __low2float(seq[1].f16x2),
        //       __high2float(seq[0].f16x2),
        //       __low2float(seq[0].f16x2));
    }
};

////////////////////////////////////////////////////////////////////////
// box_plus_seq_gen<__half2, 7, 6>()
// Input:
//                                              [ VAL6.HI,LO ] [ VAL5.HI,LO ]
// [ VAL4.HI,LO ] [ VAL3.HI,LO ] [ VAL2.HI,LO ] [ VAL1.HI,LO ] [ VAL0.HI,LO ]
// Output:
// [ BP(0,1,2,3,4,6).hi,lo ] [ BP(0,1,2,3,5,6).hi,lo ] [ BP(0,1,2,4,5,6).hi,lo ] [ BP(0,1,3,4,5,6).hi,lo ] [ BP(0,2,3,4,5,6).hi,lo ] [ BP(1,2,3,4,5,6).hi,lo ]
template <class TBoxPlusOp> struct box_plus_seq_gen<__half2, TBoxPlusOp, 7, 6>
{
    typedef TBoxPlusOp op_t;

    __device__ static void generate(word_t (&seq)[6], word_t (&app)[7])
    {
        const word_t& VAL_0 = app[0];
        const word_t& VAL_1 = app[1];
        const word_t& VAL_2 = app[2];
        const word_t& VAL_3 = app[3];
        const word_t& VAL_4 = app[4];
        const word_t& VAL_5 = app[5];
        const word_t& VAL_6 = app[6];
        
        word_t BP_13    = op_t::box_plus(VAL_1,  VAL_3);
        word_t BP_02    = op_t::box_plus(VAL_0,  VAL_2);
        word_t BP_46    = op_t::box_plus(VAL_4,  VAL_6);

        word_t BP_0246  = op_t::box_plus(BP_02,  BP_46);
        word_t BP_135   = op_t::box_plus(BP_13,  VAL_5);

        word_t BP_02456 = op_t::box_plus(BP_0246, VAL_5);
        word_t BP_01235 = op_t::box_plus(BP_02,   BP_135);
        word_t BP_13456 = op_t::box_plus(BP_135,  BP_46);
        
        word_t BP_123456 = op_t::box_plus(BP_13456, VAL_2);
        word_t BP_023456 = op_t::box_plus(BP_02456, VAL_3);
        word_t BP_013456 = op_t::box_plus(BP_13456, VAL_0);
        word_t BP_012456 = op_t::box_plus(BP_02456, VAL_1);
        word_t BP_012356 = op_t::box_plus(BP_01235, VAL_6);
        word_t BP_012346 = op_t::box_plus(BP_0246,  BP_13);
        
        seq[0] = BP_123456;
        seq[1] = BP_023456;
        seq[2] = BP_013456;
        seq[3] = BP_012456;
        seq[4] = BP_012356;
        seq[5] = BP_012346;
        //printf("box_plus_seq_gen<7, 6>: inputs = (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f), "
        //       "outputs = (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f)\n",
        //       __low2float(app[6].f16x2),
        //       __high2float(app[6].f16x2),
        //       __low2float(app[5].f16x2),
        //       __high2float(app[5].f16x2),
        //       __low2float(app[4].f16x2),
        //       __high2float(app[4].f16x2),
        //       __low2float(app[3].f16x2),
        //       __high2float(app[3].f16x2),
        //       __low2float(app[2].f16x2),
        //       __high2float(app[2].f16x2),
        //       __low2float(app[1].f16x2),
        //       __high2float(app[1].f16x2),
        //       __low2float(app[0].f16x2),
        //       __high2float(app[0].f16x2),
        //       __high2float(seq[5].f16x2),
        //       __low2float(seq[5].f16x2),
        //       __high2float(seq[4].f16x2),
        //       __low2float(seq[4].f16x2),
        //       __high2float(seq[3].f16x2),
        //       __low2float(seq[3].f16x2),
        //       __high2float(seq[2].f16x2),
        //       __low2float(seq[2].f16x2),
        //       __high2float(seq[1].f16x2),
        //       __low2float(seq[1].f16x2),
        //       __high2float(seq[0].f16x2),
        //       __low2float(seq[0].f16x2));
    }
};

////////////////////////////////////////////////////////////////////////
// box_plus_seq_gen<__half2, 7, 7>()
// Input:
//                                              [ VAL6.HI,LO ] [ VAL5.HI,LO ]
// [ VAL4.HI,LO ] [ VAL3.HI,LO ] [ VAL2.HI,LO ] [ VAL1.HI,LO ] [ VAL0.HI,LO ]
// Output:
//                           [ BP(0,1,2,3,4,5).hi,lo ] [ BP(0,1,2,3,4,6).hi,lo ] [ BP(0,1,2,3,5,6).hi,lo ]
// [ BP(0,1,2,4,5,6).hi,lo ] [ BP(0,1,3,4,5,6).hi,lo ] [ BP(0,2,3,4,5,6).hi,lo ] [ BP(1,2,3,4,5,6).hi,lo ]
template <class TBoxPlusOp> struct box_plus_seq_gen<__half2, TBoxPlusOp, 7, 7>
{
    typedef TBoxPlusOp op_t;

    __device__ static void generate(word_t (&seq)[7], word_t (&app)[7])
    {
        const word_t& VAL_0 = app[0];
        const word_t& VAL_1 = app[1];
        const word_t& VAL_2 = app[2];
        const word_t& VAL_3 = app[3];
        const word_t& VAL_4 = app[4];
        const word_t& VAL_5 = app[5];
        const word_t& VAL_6 = app[6];
        
        word_t BP_13    = op_t::box_plus(VAL_1,  VAL_3);
        word_t BP_02    = op_t::box_plus(VAL_0,  VAL_2);
        word_t BP_46    = op_t::box_plus(VAL_4,  VAL_6);

        word_t BP_0246  = op_t::box_plus(BP_02,  BP_46);
        word_t BP_135   = op_t::box_plus(BP_13,  VAL_5);

        word_t BP_02456 = op_t::box_plus(BP_0246, VAL_5);
        word_t BP_01235 = op_t::box_plus(BP_02,   BP_135);
        word_t BP_13456 = op_t::box_plus(BP_135,  BP_46);
        
        word_t BP_123456 = op_t::box_plus(BP_13456, VAL_2);
        word_t BP_023456 = op_t::box_plus(BP_02456, VAL_3);
        word_t BP_013456 = op_t::box_plus(BP_13456, VAL_0);
        word_t BP_012456 = op_t::box_plus(BP_02456, VAL_1);
        word_t BP_012356 = op_t::box_plus(BP_01235, VAL_6);
        word_t BP_012346 = op_t::box_plus(BP_0246,  BP_13);
        word_t BP_012345 = op_t::box_plus(BP_01235, VAL_4);
        
        seq[0] = BP_123456;
        seq[1] = BP_023456;
        seq[2] = BP_013456;
        seq[3] = BP_012456;
        seq[4] = BP_012356;
        seq[5] = BP_012346;
        seq[6] = BP_012345;
        //printf("box_plus_seq_gen<7, 7>: inputs = (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f), "
        //       "outputs = (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f)\n",
        //       __low2float(app[6].f16x2),
        //       __high2float(app[6].f16x2),
        //       __low2float(app[5].f16x2),
        //       __high2float(app[5].f16x2),
        //       __low2float(app[4].f16x2),
        //       __high2float(app[4].f16x2),
        //       __low2float(app[3].f16x2),
        //       __high2float(app[3].f16x2),
        //       __low2float(app[2].f16x2),
        //       __high2float(app[2].f16x2),
        //       __low2float(app[1].f16x2),
        //       __high2float(app[1].f16x2),
        //       __low2float(app[0].f16x2),
        //       __high2float(app[0].f16x2),
        //       __high2float(seq[6].f16x2),
        //       __low2float(seq[6].f16x2),
        //       __high2float(seq[5].f16x2),
        //       __low2float(seq[5].f16x2),
        //       __high2float(seq[4].f16x2),
        //       __low2float(seq[4].f16x2),
        //       __high2float(seq[3].f16x2),
        //       __low2float(seq[3].f16x2),
        //       __high2float(seq[2].f16x2),
        //       __low2float(seq[2].f16x2),
        //       __high2float(seq[1].f16x2),
        //       __low2float(seq[1].f16x2),
        //       __high2float(seq[0].f16x2),
        //       __low2float(seq[0].f16x2));
    }
};

////////////////////////////////////////////////////////////////////////
// box_plus_seq_gen<__half2, 8, 7>()
// Input:
//                               [ VAL7.HI,LO ] [ VAL6.HI,LO ] [ VAL5.HI,LO ]
// [ VAL4.HI,LO ] [ VAL3.HI,LO ] [ VAL2.HI,LO ] [ VAL1.HI,LO ] [ VAL0.HI,LO ]
// Output:
//                             [ BP(0,1,2,3,4,5,7).hi,lo ] [ BP(0,1,2,3,4,6,7).hi,lo ] [ BP(0,1,2,3,5,6,7).hi,lo ]
// [ BP(0,1,2,4,5,6,7).hi,lo ] [ BP(0,1,3,4,5,6,7).hi,lo ] [ BP(0,2,3,4,5,6,7).hi,lo ] [ BP(1,2,3,4,5,6,7).hi,lo ]
template <class TBoxPlusOp> struct box_plus_seq_gen<__half2, TBoxPlusOp, 8, 7>
{
    typedef TBoxPlusOp op_t;

    __device__ static void generate(word_t (&seq)[7], word_t (&app)[8])
    {
        const word_t& VAL_0 = app[0];
        const word_t& VAL_1 = app[1];
        const word_t& VAL_2 = app[2];
        const word_t& VAL_3 = app[3];
        const word_t& VAL_4 = app[4];
        const word_t& VAL_5 = app[5];
        const word_t& VAL_6 = app[6];
        const word_t& VAL_7 = app[7];
        
        word_t BP_13    = op_t::box_plus(VAL_1,  VAL_3);
        word_t BP_02    = op_t::box_plus(VAL_0,  VAL_2);
        word_t BP_46    = op_t::box_plus(VAL_4,  VAL_6);
        word_t BP_57    = op_t::box_plus(VAL_5,  VAL_7);

        word_t BP_0246  = op_t::box_plus(BP_02,  BP_46);
        word_t BP_1357  = op_t::box_plus(BP_13,  BP_57);

        word_t BP_134567 = op_t::box_plus(BP_1357, BP_46);
        word_t BP_024567 = op_t::box_plus(BP_0246, BP_57);
        word_t BP_012357 = op_t::box_plus(BP_1357, BP_02);
        word_t BP_012346 = op_t::box_plus(BP_0246, BP_13);

        word_t BP_1234567 = op_t::box_plus(BP_134567, VAL_2);
        word_t BP_0234567 = op_t::box_plus(BP_024567, VAL_3);
        word_t BP_0134567 = op_t::box_plus(BP_134567, VAL_0);
        word_t BP_0124567 = op_t::box_plus(BP_024567, VAL_1);
        word_t BP_0123567 = op_t::box_plus(BP_012357, VAL_6);
        word_t BP_0123467 = op_t::box_plus(BP_012346, VAL_7);
        word_t BP_0123457 = op_t::box_plus(BP_012357, VAL_4);

        seq[0] = BP_1234567;
        seq[1] = BP_0234567;
        seq[2] = BP_0134567;
        seq[3] = BP_0124567;
        seq[4] = BP_0123567;
        seq[5] = BP_0123467;
        seq[6] = BP_0123457;
        //printf("box_plus_seq_gen<8, 7>: inputs = (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f), "
        //       "outputs = (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f)\n",
        //       __low2float(app[7].f16x2),
        //       __high2float(app[7].f16x2),
        //       __low2float(app[6].f16x2),
        //       __high2float(app[6].f16x2),
        //       __low2float(app[5].f16x2),
        //       __high2float(app[5].f16x2),
        //       __low2float(app[4].f16x2),
        //       __high2float(app[4].f16x2),
        //       __low2float(app[3].f16x2),
        //       __high2float(app[3].f16x2),
        //       __low2float(app[2].f16x2),
        //       __high2float(app[2].f16x2),
        //       __low2float(app[1].f16x2),
        //       __high2float(app[1].f16x2),
        //       __low2float(app[0].f16x2),
        //       __high2float(app[0].f16x2),
        //       __high2float(seq[6].f16x2),
        //       __low2float(seq[6].f16x2),
        //       __high2float(seq[5].f16x2),
        //       __low2float(seq[5].f16x2),
        //       __high2float(seq[4].f16x2),
        //       __low2float(seq[4].f16x2),
        //       __high2float(seq[3].f16x2),
        //       __low2float(seq[3].f16x2),
        //       __high2float(seq[2].f16x2),
        //       __low2float(seq[2].f16x2),
        //       __high2float(seq[1].f16x2),
        //       __low2float(seq[1].f16x2),
        //       __high2float(seq[0].f16x2),
        //       __low2float(seq[0].f16x2));
    }
};

////////////////////////////////////////////////////////////////////////
// box_plus_seq_gen<__half2, 8, 8>()
// Input:
//                               [ VAL7.HI,LO ] [ VAL6.HI,LO ] [ VAL5.HI,LO ]
// [ VAL4.HI,LO ] [ VAL3.HI,LO ] [ VAL2.HI,LO ] [ VAL1.HI,LO ] [ VAL0.HI,LO ]
// Output:
// [ BP(0,1,2,3,4,5,6).hi,lo ] [ BP(0,1,2,3,4,5,7).hi,lo ] [ BP(0,1,2,3,4,6,7).hi,lo ] [ BP(0,1,2,3,5,6,7).hi,lo ]
// [ BP(0,1,2,4,5,6,7).hi,lo ] [ BP(0,1,3,4,5,6,7).hi,lo ] [ BP(0,2,3,4,5,6,7).hi,lo ] [ BP(1,2,3,4,5,6,7).hi,lo ]
template <class TBoxPlusOp> struct box_plus_seq_gen<__half2, TBoxPlusOp, 8, 8>
{
    typedef TBoxPlusOp op_t;

    __device__ static void generate(word_t (&seq)[8], word_t (&app)[8])
    {
        const word_t& VAL_0 = app[0];
        const word_t& VAL_1 = app[1];
        const word_t& VAL_2 = app[2];
        const word_t& VAL_3 = app[3];
        const word_t& VAL_4 = app[4];
        const word_t& VAL_5 = app[5];
        const word_t& VAL_6 = app[6];
        const word_t& VAL_7 = app[7];
        
        word_t BP_13    = op_t::box_plus(VAL_1,  VAL_3);
        word_t BP_02    = op_t::box_plus(VAL_0,  VAL_2);
        word_t BP_46    = op_t::box_plus(VAL_4,  VAL_6);
        word_t BP_57    = op_t::box_plus(VAL_5,  VAL_7);

        word_t BP_0246  = op_t::box_plus(BP_02,  BP_46);
        word_t BP_1357  = op_t::box_plus(BP_13,  BP_57);

        word_t BP_134567 = op_t::box_plus(BP_1357, BP_46);
        word_t BP_024567 = op_t::box_plus(BP_0246, BP_57);
        word_t BP_012357 = op_t::box_plus(BP_1357, BP_02);
        word_t BP_012346 = op_t::box_plus(BP_0246, BP_13);

        word_t BP_1234567 = op_t::box_plus(BP_134567, VAL_2);
        word_t BP_0234567 = op_t::box_plus(BP_024567, VAL_3);
        word_t BP_0134567 = op_t::box_plus(BP_134567, VAL_0);
        word_t BP_0124567 = op_t::box_plus(BP_024567, VAL_1);
        word_t BP_0123567 = op_t::box_plus(BP_012357, VAL_6);
        word_t BP_0123467 = op_t::box_plus(BP_012346, VAL_7);
        word_t BP_0123457 = op_t::box_plus(BP_012357, VAL_4);
        word_t BP_0123456 = op_t::box_plus(BP_012346, VAL_5);

        seq[0] = BP_1234567;
        seq[1] = BP_0234567;
        seq[2] = BP_0134567;
        seq[3] = BP_0124567;
        seq[4] = BP_0123567;
        seq[5] = BP_0123467;
        seq[6] = BP_0123457;
        seq[7] = BP_0123456;
        //printf("box_plus_seq_gen<8, 8>: inputs = (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f), "
        //       "outputs = (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f)\n",
        //       __low2float(app[7].f16x2),
        //       __high2float(app[7].f16x2),
        //       __low2float(app[6].f16x2),
        //       __high2float(app[6].f16x2),
        //       __low2float(app[5].f16x2),
        //       __high2float(app[5].f16x2),
        //       __low2float(app[4].f16x2),
        //       __high2float(app[4].f16x2),
        //       __low2float(app[3].f16x2),
        //       __high2float(app[3].f16x2),
        //       __low2float(app[2].f16x2),
        //       __high2float(app[2].f16x2),
        //       __low2float(app[1].f16x2),
        //       __high2float(app[1].f16x2),
        //       __low2float(app[0].f16x2),
        //       __high2float(app[0].f16x2),
        //       __high2float(seq[7].f16x2),
        //       __low2float(seq[7].f16x2),
        //       __high2float(seq[6].f16x2),
        //       __low2float(seq[6].f16x2),
        //       __high2float(seq[5].f16x2),
        //       __low2float(seq[5].f16x2),
        //       __high2float(seq[4].f16x2),
        //       __low2float(seq[4].f16x2),
        //       __high2float(seq[3].f16x2),
        //       __low2float(seq[3].f16x2),
        //       __high2float(seq[2].f16x2),
        //       __low2float(seq[2].f16x2),
        //       __high2float(seq[1].f16x2),
        //       __low2float(seq[1].f16x2),
        //       __high2float(seq[0].f16x2),
        //       __low2float(seq[0].f16x2));
    }
};

////////////////////////////////////////////////////////////////////////
// box_plus_seq_gen<__half2, 9, 8>()
// Input:
//                [ VAL8.HI,LO ] [ VAL7.HI,LO ] [ VAL6.HI,LO ] [ VAL5.HI,LO ]
// [ VAL4.HI,LO ] [ VAL3.HI,LO ] [ VAL2.HI,LO ] [ VAL1.HI,LO ] [ VAL0.HI,LO ]
// Output:
// [ BP(0,1,2,3,4,5,6,8).hi,lo ] [ BP(0,1,2,3,4,5,7,8).hi,lo ] [ BP(0,1,2,3,4,6,7,8).hi,lo ] [ BP(0,1,2,3,5,6,7,8).hi,lo ]
// [ BP(0,1,2,4,5,6,7,8).hi,lo ] [ BP(0,1,3,4,5,6,7,8).hi,lo ] [ BP(0,2,3,4,5,6,7,8).hi,lo ] [ BP(1,2,3,4,5,6,7,8).hi,lo ]
template <class TBoxPlusOp> struct box_plus_seq_gen<__half2, TBoxPlusOp, 9, 8>
{
    typedef TBoxPlusOp op_t;

    __device__ static void generate(word_t (&seq)[8], word_t (&app)[9])
    {
        const word_t& VAL_0 = app[0];
        const word_t& VAL_1 = app[1];
        const word_t& VAL_2 = app[2];
        const word_t& VAL_3 = app[3];
        const word_t& VAL_4 = app[4];
        const word_t& VAL_5 = app[5];
        const word_t& VAL_6 = app[6];
        const word_t& VAL_7 = app[7];
        const word_t& VAL_8 = app[8];
        
        word_t BP_13    = op_t::box_plus(VAL_1,  VAL_3);
        word_t BP_02    = op_t::box_plus(VAL_0,  VAL_2);
        word_t BP_46    = op_t::box_plus(VAL_4,  VAL_6);
        word_t BP_57    = op_t::box_plus(VAL_5,  VAL_7);

        word_t BP_0246  = op_t::box_plus(BP_02,  BP_46);
        word_t BP_1357  = op_t::box_plus(BP_13,  BP_57);

        word_t BP_02468 = op_t::box_plus(BP_0246,  VAL_8);
        word_t BP_13578 = op_t::box_plus(BP_1357,  VAL_8);

        word_t BP_1345678 = op_t::box_plus(BP_13578, BP_46);
        word_t BP_0245678 = op_t::box_plus(BP_02468, BP_57);
        word_t BP_0123578 = op_t::box_plus(BP_13578, BP_02);
        word_t BP_0123468 = op_t::box_plus(BP_02468, BP_13);

        word_t BP_12345678 = op_t::box_plus(BP_1345678, VAL_2);
        word_t BP_02345678 = op_t::box_plus(BP_0245678, VAL_3);
        word_t BP_01345678 = op_t::box_plus(BP_1345678, VAL_0);
        word_t BP_01245678 = op_t::box_plus(BP_0245678, VAL_1);
        word_t BP_01235678 = op_t::box_plus(BP_0123578, VAL_6);
        word_t BP_01234678 = op_t::box_plus(BP_0123468, VAL_7);
        word_t BP_01234578 = op_t::box_plus(BP_0123578, VAL_4);
        word_t BP_01234568 = op_t::box_plus(BP_0123468, VAL_5);

        seq[0] = BP_12345678;
        seq[1] = BP_02345678;
        seq[2] = BP_01345678;
        seq[3] = BP_01245678;
        seq[4] = BP_01235678;
        seq[5] = BP_01234678;
        seq[6] = BP_01234578;
        seq[7] = BP_01234568;
        //printf("box_plus_seq_gen<9, 8>: inputs = (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f), "
        //       "outputs = (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f)\n",
        //       __low2float(app[8].f16x2),
        //       __high2float(app[8].f16x2),
        //       __low2float(app[7].f16x2),
        //       __high2float(app[7].f16x2),
        //       __low2float(app[6].f16x2),
        //       __high2float(app[6].f16x2),
        //       __low2float(app[5].f16x2),
        //       __high2float(app[5].f16x2),
        //       __low2float(app[4].f16x2),
        //       __high2float(app[4].f16x2),
        //       __low2float(app[3].f16x2),
        //       __high2float(app[3].f16x2),
        //       __low2float(app[2].f16x2),
        //       __high2float(app[2].f16x2),
        //       __low2float(app[1].f16x2),
        //       __high2float(app[1].f16x2),
        //       __low2float(app[0].f16x2),
        //       __high2float(app[0].f16x2),
        //       __high2float(seq[7].f16x2),
        //       __low2float(seq[7].f16x2),
        //       __high2float(seq[6].f16x2),
        //       __low2float(seq[6].f16x2),
        //       __high2float(seq[5].f16x2),
        //       __low2float(seq[5].f16x2),
        //       __high2float(seq[4].f16x2),
        //       __low2float(seq[4].f16x2),
        //       __high2float(seq[3].f16x2),
        //       __low2float(seq[3].f16x2),
        //       __high2float(seq[2].f16x2),
        //       __low2float(seq[2].f16x2),
        //       __high2float(seq[1].f16x2),
        //       __low2float(seq[1].f16x2),
        //       __high2float(seq[0].f16x2),
        //       __low2float(seq[0].f16x2));
    }
};

////////////////////////////////////////////////////////////////////////
// box_plus_seq_gen<__half2, 9, 9>()
// Input:
//                [ VAL8.HI,LO ] [ VAL7.HI,LO ] [ VAL6.HI,LO ] [ VAL5.HI,LO ]
// [ VAL4.HI,LO ] [ VAL3.HI,LO ] [ VAL2.HI,LO ] [ VAL1.HI,LO ] [ VAL0.HI,LO ]
// Output:
//                                                                                           [ BP(0,1,2,3,4,5,6,7).hi,lo ]
// [ BP(0,1,2,3,4,5,6,8).hi,lo ] [ BP(0,1,2,3,4,5,7,8).hi,lo ] [ BP(0,1,2,3,4,6,7,8).hi,lo ] [ BP(0,1,2,3,5,6,7,8).hi,lo ]
// [ BP(0,1,2,4,5,6,7,8).hi,lo ] [ BP(0,1,3,4,5,6,7,8).hi,lo ] [ BP(0,2,3,4,5,6,7,8).hi,lo ] [ BP(1,2,3,4,5,6,7,8).hi,lo ]
template <class TBoxPlusOp> struct box_plus_seq_gen<__half2, TBoxPlusOp, 9, 9>
{
    typedef TBoxPlusOp op_t;

    __device__ static void generate(word_t (&seq)[9], word_t (&app)[9])
    {
        const word_t& VAL_0 = app[0];
        const word_t& VAL_1 = app[1];
        const word_t& VAL_2 = app[2];
        const word_t& VAL_3 = app[3];
        const word_t& VAL_4 = app[4];
        const word_t& VAL_5 = app[5];
        const word_t& VAL_6 = app[6];
        const word_t& VAL_7 = app[7];
        const word_t& VAL_8 = app[8];
        
        word_t BP_13    = op_t::box_plus(VAL_1,  VAL_3);
        word_t BP_02    = op_t::box_plus(VAL_0,  VAL_2);
        word_t BP_46    = op_t::box_plus(VAL_4,  VAL_6);
        word_t BP_57    = op_t::box_plus(VAL_5,  VAL_7);

        word_t BP_0246  = op_t::box_plus(BP_02,  BP_46);
        word_t BP_1357  = op_t::box_plus(BP_13,  BP_57);

        word_t BP_02468 = op_t::box_plus(BP_0246,  VAL_8);
        word_t BP_13578 = op_t::box_plus(BP_1357,  VAL_8);

        word_t BP_1345678 = op_t::box_plus(BP_13578, BP_46);
        word_t BP_0245678 = op_t::box_plus(BP_02468, BP_57);
        word_t BP_0123578 = op_t::box_plus(BP_13578, BP_02);
        word_t BP_0123468 = op_t::box_plus(BP_02468, BP_13);

        word_t BP_12345678 = op_t::box_plus(BP_1345678, VAL_2);
        word_t BP_02345678 = op_t::box_plus(BP_0245678, VAL_3);
        word_t BP_01345678 = op_t::box_plus(BP_1345678, VAL_0);
        word_t BP_01245678 = op_t::box_plus(BP_0245678, VAL_1);
        word_t BP_01235678 = op_t::box_plus(BP_0123578, VAL_6);
        word_t BP_01234678 = op_t::box_plus(BP_0123468, VAL_7);
        word_t BP_01234578 = op_t::box_plus(BP_0123578, VAL_4);
        word_t BP_01234568 = op_t::box_plus(BP_0123468, VAL_5);
        word_t BP_01234567 = op_t::box_plus(BP_0246,    BP_1357);
        
        seq[0] = BP_12345678;
        seq[1] = BP_02345678;
        seq[2] = BP_01345678;
        seq[3] = BP_01245678;
        seq[4] = BP_01235678;
        seq[5] = BP_01234678;
        seq[6] = BP_01234578;
        seq[7] = BP_01234568;
        seq[8] = BP_01234567;
        //printf("box_plus_seq_gen<9, 9>: inputs = (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f), "
        //       "outputs = (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f)\n",
        //       __low2float(app[8].f16x2),
        //       __high2float(app[8].f16x2),
        //       __low2float(app[7].f16x2),
        //       __high2float(app[7].f16x2),
        //       __low2float(app[6].f16x2),
        //       __high2float(app[6].f16x2),
        //       __low2float(app[5].f16x2),
        //       __high2float(app[5].f16x2),
        //       __low2float(app[4].f16x2),
        //       __high2float(app[4].f16x2),
        //       __low2float(app[3].f16x2),
        //       __high2float(app[3].f16x2),
        //       __low2float(app[2].f16x2),
        //       __high2float(app[2].f16x2),
        //       __low2float(app[1].f16x2),
        //       __high2float(app[1].f16x2),
        //       __low2float(app[0].f16x2),
        //       __high2float(app[0].f16x2),
        //       __high2float(seq[8].f16x2),
        //       __low2float(seq[8].f16x2),
        //       __high2float(seq[7].f16x2),
        //       __low2float(seq[7].f16x2),
        //       __high2float(seq[6].f16x2),
        //       __low2float(seq[6].f16x2),
        //       __high2float(seq[5].f16x2),
        //       __low2float(seq[5].f16x2),
        //       __high2float(seq[4].f16x2),
        //       __low2float(seq[4].f16x2),
        //       __high2float(seq[3].f16x2),
        //       __low2float(seq[3].f16x2),
        //       __high2float(seq[2].f16x2),
        //       __low2float(seq[2].f16x2),
        //       __high2float(seq[1].f16x2),
        //       __low2float(seq[1].f16x2),
        //       __high2float(seq[0].f16x2),
        //       __low2float(seq[0].f16x2));
    }
};

////////////////////////////////////////////////////////////////////////
// box_plus_seq_gen<__half2, 10, 9>()
// Input:
// [ VAL9.HI,LO ] [ VAL8.HI,LO ] [ VAL7.HI,LO ] [ VAL6.HI,LO ] [ VAL5.HI,LO ]
// [ VAL4.HI,LO ] [ VAL3.HI,LO ] [ VAL2.HI,LO ] [ VAL1.HI,LO ] [ VAL0.HI,LO ]
// Output:
//                                                                                                 [ BP(0,1,2,3,4,5,6,7,9).hi,lo ]
// [ BP(0,1,2,3,4,5,6,8,9).hi,lo ] [ BP(0,1,2,3,4,5,7,8,9).hi,lo ] [ BP(0,1,2,3,4,6,7,8,9).hi,lo ] [ BP(0,1,2,3,5,6,7,8,9).hi,lo ]
// [ BP(0,1,2,4,5,6,7,8,9).hi,lo ] [ BP(0,1,3,4,5,6,7,8,9).hi,lo ] [ BP(0,2,3,4,5,6,7,8,9).hi,lo ] [ BP(1,2,3,4,5,6,7,8,9).hi,lo ]
template <class TBoxPlusOp> struct box_plus_seq_gen<__half2, TBoxPlusOp, 10, 9>
{
    typedef TBoxPlusOp op_t;

    __device__ static void generate(word_t (&seq)[9], word_t (&app)[10])
    {
        const word_t& VAL_0 = app[0];
        const word_t& VAL_1 = app[1];
        const word_t& VAL_2 = app[2];
        const word_t& VAL_3 = app[3];
        const word_t& VAL_4 = app[4];
        const word_t& VAL_5 = app[5];
        const word_t& VAL_6 = app[6];
        const word_t& VAL_7 = app[7];
        const word_t& VAL_8 = app[8];
        const word_t& VAL_9 = app[9];
        
        word_t BP_13    = op_t::box_plus(VAL_1,  VAL_3);
        word_t BP_02    = op_t::box_plus(VAL_0,  VAL_2);
        word_t BP_46    = op_t::box_plus(VAL_4,  VAL_6);
        word_t BP_57    = op_t::box_plus(VAL_5,  VAL_7);

        word_t BP_0246  = op_t::box_plus(BP_02,  BP_46);
        word_t BP_1357  = op_t::box_plus(BP_13,  BP_57);
        
        word_t BP_02468 = op_t::box_plus(BP_0246, VAL_8);
        word_t BP_13579 = op_t::box_plus(BP_1357, VAL_9);

        word_t BP_135789 = op_t::box_plus(BP_13579, VAL_8);
        word_t BP_024689 = op_t::box_plus(BP_02468, VAL_9);

        word_t BP_13456789 = op_t::box_plus(BP_135789, BP_46);
        word_t BP_02456789 = op_t::box_plus(BP_024689, BP_57);
        word_t BP_01235789 = op_t::box_plus(BP_135789, BP_02);
        word_t BP_01234689 = op_t::box_plus(BP_024689, BP_13);
        
        word_t BP_123456789 = op_t::box_plus(BP_13456789, VAL_2);
        word_t BP_023456789 = op_t::box_plus(BP_02456789, VAL_3);
        word_t BP_013456789 = op_t::box_plus(BP_13456789, VAL_0);
        word_t BP_012456789 = op_t::box_plus(BP_02456789, VAL_1);
        word_t BP_012356789 = op_t::box_plus(BP_01235789, VAL_6);
        word_t BP_012346789 = op_t::box_plus(BP_01234689, VAL_7);
        word_t BP_012345789 = op_t::box_plus(BP_01235789, VAL_4);
        word_t BP_012345689 = op_t::box_plus(BP_01234689, VAL_5);
        word_t BP_012345679 = op_t::box_plus(BP_0246,     BP_13579);
        
        seq[0] = BP_123456789;
        seq[1] = BP_023456789;
        seq[2] = BP_013456789;
        seq[3] = BP_012456789;
        seq[4] = BP_012356789;
        seq[5] = BP_012346789;
        seq[6] = BP_012345789;
        seq[7] = BP_012345689;
        seq[8] = BP_012345679;
        //printf("box_plus_seq_gen<10, 9>: inputs = (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f)"
        //       "(%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f), "
        //       "outputs = (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f)\n",
        //       __low2float(app[9].f16x2),
        //       __high2float(app[9].f16x2),
        //       __low2float(app[8].f16x2),
        //       __high2float(app[8].f16x2),
        //       __low2float(app[7].f16x2),
        //       __high2float(app[7].f16x2),
        //       __low2float(app[6].f16x2),
        //       __high2float(app[6].f16x2),
        //       __low2float(app[5].f16x2),
        //       __high2float(app[5].f16x2),
        //       __low2float(app[4].f16x2),
        //       __high2float(app[4].f16x2),
        //       __low2float(app[3].f16x2),
        //       __high2float(app[3].f16x2),
        //       __low2float(app[2].f16x2),
        //       __high2float(app[2].f16x2),
        //       __low2float(app[1].f16x2),
        //       __high2float(app[1].f16x2),
        //       __low2float(app[0].f16x2),
        //       __high2float(app[0].f16x2),
        //       __high2float(seq[8].f16x2),
        //       __low2float(seq[8].f16x2),
        //       __high2float(seq[7].f16x2),
        //       __low2float(seq[7].f16x2),
        //       __high2float(seq[6].f16x2),
        //       __low2float(seq[6].f16x2),
        //       __high2float(seq[5].f16x2),
        //       __low2float(seq[5].f16x2),
        //       __high2float(seq[4].f16x2),
        //       __low2float(seq[4].f16x2),
        //       __high2float(seq[3].f16x2),
        //       __low2float(seq[3].f16x2),
        //       __high2float(seq[2].f16x2),
        //       __low2float(seq[2].f16x2),
        //       __high2float(seq[1].f16x2),
        //       __low2float(seq[1].f16x2),
        //       __high2float(seq[0].f16x2),
        //       __low2float(seq[0].f16x2));
    }
};

////////////////////////////////////////////////////////////////////////
// box_plus_seq_gen<__half2, 10, 10>()
// Input:
// [ VAL9.HI,LO ] [ VAL8.HI,LO ] [ VAL7.HI,LO ] [ VAL6.HI,LO ] [ VAL5.HI,LO ]
// [ VAL4.HI,LO ] [ VAL3.HI,LO ] [ VAL2.HI,LO ] [ VAL1.HI,LO ] [ VAL0.HI,LO ]
// Output:
//                                                                 [ BP(0,1,2,3,4,5,6,7,8).hi,lo ] [ BP(0,1,2,3,4,5,6,7,9).hi,lo ]
// [ BP(0,1,2,3,4,5,6,8,9).hi,lo ] [ BP(0,1,2,3,4,5,7,8,9).hi,lo ] [ BP(0,1,2,3,4,6,7,8,9).hi,lo ] [ BP(0,1,2,3,5,6,7,8,9).hi,lo ]
// [ BP(0,1,2,4,5,6,7,8,9).hi,lo ] [ BP(0,1,3,4,5,6,7,8,9).hi,lo ] [ BP(0,2,3,4,5,6,7,8,9).hi,lo ] [ BP(1,2,3,4,5,6,7,8,9).hi,lo ]
template <class TBoxPlusOp> struct box_plus_seq_gen<__half2, TBoxPlusOp, 10, 10>
{
    typedef TBoxPlusOp op_t;

    __device__ static void generate(word_t (&seq)[10], word_t (&app)[10])
    {
        const word_t& VAL_0 = app[0];
        const word_t& VAL_1 = app[1];
        const word_t& VAL_2 = app[2];
        const word_t& VAL_3 = app[3];
        const word_t& VAL_4 = app[4];
        const word_t& VAL_5 = app[5];
        const word_t& VAL_6 = app[6];
        const word_t& VAL_7 = app[7];
        const word_t& VAL_8 = app[8];
        const word_t& VAL_9 = app[9];
        
        word_t BP_13    = op_t::box_plus(VAL_1,  VAL_3);
        word_t BP_02    = op_t::box_plus(VAL_0,  VAL_2);
        word_t BP_46    = op_t::box_plus(VAL_4,  VAL_6);
        word_t BP_57    = op_t::box_plus(VAL_5,  VAL_7);

        word_t BP_0246  = op_t::box_plus(BP_02,  BP_46);
        word_t BP_1357  = op_t::box_plus(BP_13,  BP_57);
        
        word_t BP_02468  = op_t::box_plus(BP_0246, VAL_8);
        word_t BP_13579  = op_t::box_plus(BP_1357, VAL_9);

        word_t BP_024689 = op_t::box_plus(BP_02468, VAL_9);
        word_t BP_135789 = op_t::box_plus(BP_13579, VAL_8);

        word_t BP_13456789 = op_t::box_plus(BP_135789, BP_46);
        word_t BP_02456789 = op_t::box_plus(BP_024689, BP_57);
        word_t BP_01235789 = op_t::box_plus(BP_135789, BP_02);
        word_t BP_01234689 = op_t::box_plus(BP_024689, BP_13);
        
        word_t BP_123456789 = op_t::box_plus(BP_13456789, VAL_2);
        word_t BP_023456789 = op_t::box_plus(BP_02456789, VAL_3);
        word_t BP_013456789 = op_t::box_plus(BP_13456789, VAL_0);
        word_t BP_012456789 = op_t::box_plus(BP_02456789, VAL_1);
        word_t BP_012356789 = op_t::box_plus(BP_01235789, VAL_6);
        word_t BP_012346789 = op_t::box_plus(BP_01234689, VAL_7);
        word_t BP_012345789 = op_t::box_plus(BP_01235789, VAL_4);
        word_t BP_012345689 = op_t::box_plus(BP_01234689, VAL_5);
        word_t BP_012345679 = op_t::box_plus(BP_0246,     BP_13579);
        word_t BP_012345678 = op_t::box_plus(BP_02468,    BP_1357);
        
        seq[0] = BP_123456789;
        seq[1] = BP_023456789;
        seq[2] = BP_013456789;
        seq[3] = BP_012456789;
        seq[4] = BP_012356789;
        seq[5] = BP_012346789;
        seq[6] = BP_012345789;
        seq[7] = BP_012345689;
        seq[8] = BP_012345679;
        seq[9] = BP_012345678;
        //printf("box_plus_seq_gen<10, 10>: inputs = (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f)"
        //       "(%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f), "
        //       "outputs = (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) "
        //       "(%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f)\n",
        //       __low2float(app[9].f16x2),
        //       __high2float(app[9].f16x2),
        //       __low2float(app[8].f16x2),
        //       __high2float(app[8].f16x2),
        //       __low2float(app[7].f16x2),
        //       __high2float(app[7].f16x2),
        //       __low2float(app[6].f16x2),
        //       __high2float(app[6].f16x2),
        //       __low2float(app[5].f16x2),
        //       __high2float(app[5].f16x2),
        //       __low2float(app[4].f16x2),
        //       __high2float(app[4].f16x2),
        //       __low2float(app[3].f16x2),
        //       __high2float(app[3].f16x2),
        //       __low2float(app[2].f16x2),
        //       __high2float(app[2].f16x2),
        //       __low2float(app[1].f16x2),
        //       __high2float(app[1].f16x2),
        //       __low2float(app[0].f16x2),
        //       __high2float(app[0].f16x2),
        //       __high2float(seq[9].f16x2),
        //       __low2float(seq[9].f16x2),
        //       __high2float(seq[8].f16x2),
        //       __low2float(seq[8].f16x2),
        //       __high2float(seq[7].f16x2),
        //       __low2float(seq[7].f16x2),
        //       __high2float(seq[6].f16x2),
        //       __low2float(seq[6].f16x2),
        //       __high2float(seq[5].f16x2),
        //       __low2float(seq[5].f16x2),
        //       __high2float(seq[4].f16x2),
        //       __low2float(seq[4].f16x2),
        //       __high2float(seq[3].f16x2),
        //       __low2float(seq[3].f16x2),
        //       __high2float(seq[2].f16x2),
        //       __low2float(seq[2].f16x2),
        //       __high2float(seq[1].f16x2),
        //       __low2float(seq[1].f16x2),
        //       __high2float(seq[0].f16x2),
        //       __low2float(seq[0].f16x2));
    }
};

////////////////////////////////////////////////////////////////////////
// box_plus_seq_gen<__half2, 19, 19>()
// Input:
//                 [ VAL18.HI,LO ] [ VAL17.HI,LO ] [ VAL16.HI,LO ] [ VAL15.HI,LO ]
// [ VAL14.HI,LO ] [ VAL13.HI,LO ] [ VAL12.HI,LO ] [ VAL11.HI,LO ] [ VAL10.HI,LO ]
// [ VAL9.HI,LO  ] [ VAL8.HI,LO  ] [ VAL7.HI,LO  ] [ VAL6.HI,LO  ] [ VAL5.HI,LO  ]
// [ VAL4.HI,LO  ] [ VAL3.HI,LO  ] [ VAL2.HI,LO  ] [ VAL1.HI,LO  ] [ VAL0.HI,LO  ]
template <class TBoxPlusOp> struct box_plus_seq_gen<__half2, TBoxPlusOp, 19, 19>
{
    typedef TBoxPlusOp op_t;

    __device__ static void generate(word_t (&seq)[19], word_t (&app)[19])
    {
        const word_t& VAL_0  = app[0];
        const word_t& VAL_1  = app[1];
        const word_t& VAL_2  = app[2];
        const word_t& VAL_3  = app[3];
        const word_t& VAL_4  = app[4];
        const word_t& VAL_5  = app[5];
        const word_t& VAL_6  = app[6];
        const word_t& VAL_7  = app[7];
        const word_t& VAL_8  = app[8];
        const word_t& VAL_9  = app[9];
        const word_t& VAL_10 = app[10];
        const word_t& VAL_11 = app[11];
        const word_t& VAL_12 = app[12];
        const word_t& VAL_13 = app[13];
        const word_t& VAL_14 = app[14];
        const word_t& VAL_15 = app[15];
        const word_t& VAL_16 = app[16];
        const word_t& VAL_17 = app[17];
        const word_t& VAL_18 = app[18];
        
        word_t BP_0_2    = op_t::box_plus(VAL_0,  VAL_2);                                                                      // 1
        word_t BP_1_3    = op_t::box_plus(VAL_1,  VAL_3);                                                                      // 2
        word_t BP_4_6    = op_t::box_plus(VAL_4,  VAL_6);                                                                      // 3
        word_t BP_5_7    = op_t::box_plus(VAL_5,  VAL_7);                                                                      // 4
        word_t BP_8_10   = op_t::box_plus(VAL_8,  VAL_10);                                                                     // 5
        word_t BP_9_11   = op_t::box_plus(VAL_9,  VAL_11);                                                                     // 6
        word_t BP_12_14  = op_t::box_plus(VAL_12, VAL_14);                                                                     // 7
        word_t BP_13_15  = op_t::box_plus(VAL_13, VAL_15);                                                                     // 8
        
        word_t BP_0_2_4_6    = op_t::box_plus(BP_0_2,  BP_4_6);                                                                // 9
        word_t BP_1_3_5_7    = op_t::box_plus(BP_1_3,  BP_5_7);                                                                // 10
        word_t BP_8_10_12_14 = op_t::box_plus(BP_8_10, BP_12_14);                                                              // 11
        word_t BP_9_11_13_15 = op_t::box_plus(BP_9_11, BP_13_15);                                                              // 12

        word_t BP_0_2_4_6_8_10_12_14 = op_t::box_plus(BP_0_2_4_6, BP_8_10_12_14);                                              // 13
        word_t BP_1_3_5_7_9_11_13_15 = op_t::box_plus(BP_1_3_5_7, BP_9_11_13_15);                                              // 14
        
        word_t BP_0_2_4_6_8_10_12_14_16 = op_t::box_plus(BP_0_2_4_6_8_10_12_14, VAL_16);                                       // 15
        word_t BP_1_3_5_7_9_11_13_15_17 = op_t::box_plus(BP_1_3_5_7_9_11_13_15, VAL_17);                                       // 16

        word_t BP_0_2_4_6_8_10_12_14_16_18 = op_t::box_plus(BP_0_2_4_6_8_10_12_14_16, VAL_18);                                 // 17
        word_t BP_1_3_5_7_9_11_13_15_17_18 = op_t::box_plus(BP_1_3_5_7_9_11_13_15_17, VAL_18);                                 // 18

        word_t BP_1_3_5_7_9_11_13_15_16_17_18 = op_t::box_plus(BP_1_3_5_7_9_11_13_15_17_18, VAL_16);                           // 19
        word_t BP_0_2_4_6_8_10_12_14_16_17_18 = op_t::box_plus(BP_0_2_4_6_8_10_12_14_16_18, VAL_17);                           // 20

        // - missing 0,2,4,6
        word_t BP_1_3_5_7_8_9_10_11_12_13_14_15_16_17_18 = op_t::box_plus(BP_1_3_5_7_9_11_13_15_16_17_18,                      // 21
                                                                          BP_8_10_12_14);
        // - - missing 0,2
        word_t BP_1_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18 = op_t::box_plus(BP_1_3_5_7_8_9_10_11_12_13_14_15_16_17_18,       // 22
                                                                              BP_4_6);
        // - - - missing 0
        word_t BP_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18 = op_t::box_plus(BP_1_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18, // 23
                                                                                VAL_2);
        // - - - missing 2
        word_t BP_0_1_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18 = op_t::box_plus(BP_1_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18, // 24
                                                                                VAL_0);
        // - - missing 4,6
        word_t BP_0_1_2_3_5_7_8_9_10_11_12_13_14_15_16_17_18 = op_t::box_plus(BP_1_3_5_7_8_9_10_11_12_13_14_15_16_17_18,       // 25
                                                                              BP_0_2);
        // - - - missing 4
        word_t BP_0_1_2_3_5_6_7_8_9_10_11_12_13_14_15_16_17_18 = op_t::box_plus(BP_0_1_2_3_5_7_8_9_10_11_12_13_14_15_16_17_18, // 26
                                                                                VAL_6);
        // - - - missing 6
        word_t BP_0_1_2_3_4_5_7_8_9_10_11_12_13_14_15_16_17_18 = op_t::box_plus(BP_0_1_2_3_5_7_8_9_10_11_12_13_14_15_16_17_18, // 27
                                                                                VAL_4);
        // - missing 8,10,12,14
        word_t BP_0_1_2_3_4_5_6_7_9_11_13_15_16_17_18 = op_t::box_plus(BP_1_3_5_7_9_11_13_15_16_17_18,                         // 28
                                                                       BP_0_2_4_6);
        // - - missing 8,10
        word_t BP_0_1_2_3_4_5_6_7_9_11_12_13_14_15_16_17_18 = op_t::box_plus(BP_0_1_2_3_4_5_6_7_9_11_13_15_16_17_18,           // 29
                                                                             BP_12_14);
        // - - - missing 8
        word_t BP_0_1_2_3_4_5_6_7_9_10_11_12_13_14_15_16_17_18 = op_t::box_plus(BP_0_1_2_3_4_5_6_7_9_11_12_13_14_15_16_17_18,  // 30
                                                                                VAL_10);
        // - - - missing 10
        word_t BP_0_1_2_3_4_5_6_7_8_9_11_12_13_14_15_16_17_18 = op_t::box_plus(BP_0_1_2_3_4_5_6_7_9_11_12_13_14_15_16_17_18,   // 31
                                                                               VAL_8);
        // - - missing 12,14
        word_t BP_0_1_2_3_4_5_6_7_8_9_10_11_13_15_16_17_18 = op_t::box_plus(BP_0_1_2_3_4_5_6_7_9_11_13_15_16_17_18,            // 32
                                                                            BP_8_10);
        // - - - missing 12
        word_t BP_0_1_2_3_4_5_6_7_8_9_10_11_13_14_15_16_17_18 = op_t::box_plus(BP_0_1_2_3_4_5_6_7_8_9_10_11_13_15_16_17_18,    // 33
                                                                               VAL_14);
        // - - - missing 14
        word_t BP_0_1_2_3_4_5_6_7_8_9_10_11_12_13_15_16_17_18 = op_t::box_plus(BP_0_1_2_3_4_5_6_7_8_9_10_11_13_15_16_17_18,    // 34
                                                                               VAL_12);
        // - missing 1,3,5,7
        word_t BP_0_2_4_6_8_9_10_11_12_13_14_15_16_17_18 = op_t::box_plus(BP_0_2_4_6_8_10_12_14_16_17_18,                      // 35
                                                                          BP_9_11_13_15);
        // - - missing 1,3
        word_t BP_0_2_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18 = op_t::box_plus(BP_0_2_4_6_8_9_10_11_12_13_14_15_16_17_18,       // 36
                                                                              BP_5_7);
        // - - - missing 1
        word_t BP_0_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18 = op_t::box_plus(BP_0_2_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18, // 37
                                                                                VAL_3);
        // - - - missing 3
        word_t BP_0_1_2_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18 = op_t::box_plus(BP_0_2_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18, // 38
                                                                                VAL_1);
        // - - missing 5,7
        word_t BP_0_1_2_3_4_6_8_9_10_11_12_13_14_15_16_17_18 = op_t::box_plus(BP_0_2_4_6_8_9_10_11_12_13_14_15_16_17_18,       // 39
                                                                              BP_1_3);
        // - - - missing 5
        word_t BP_0_1_2_3_4_6_7_8_9_10_11_12_13_14_15_16_17_18 = op_t::box_plus(BP_0_1_2_3_4_6_8_9_10_11_12_13_14_15_16_17_18, // 40
                                                                                VAL_7);
        // - - - missing 7
        word_t BP_0_1_2_3_4_5_6_8_9_10_11_12_13_14_15_16_17_18 = op_t::box_plus(BP_0_1_2_3_4_6_8_9_10_11_12_13_14_15_16_17_18, // 41
                                                                                VAL_5);
        // - missing 9,11,13,15
        word_t BP_0_1_2_3_4_5_6_7_8_10_12_14_16_17_18 = op_t::box_plus(BP_0_2_4_6_8_10_12_14_16_17_18,                         // 42
                                                                       BP_1_3_5_7);
        // - - missing 9,11
        word_t BP_0_1_2_3_4_5_6_7_8_10_12_13_14_15_16_17_18 = op_t::box_plus(BP_0_1_2_3_4_5_6_7_8_10_12_14_16_17_18,           // 43
                                                                             BP_13_15);
        // - - - missing 9
        word_t BP_0_1_2_3_4_5_6_7_8_10_11_12_13_14_15_16_17_18 = op_t::box_plus(BP_0_1_2_3_4_5_6_7_8_10_12_13_14_15_16_17_18,  // 44
                                                                                VAL_11);
        // - - - missing 11
        word_t BP_0_1_2_3_4_5_6_7_8_9_10_12_13_14_15_16_17_18 = op_t::box_plus(BP_0_1_2_3_4_5_6_7_8_10_12_13_14_15_16_17_18,   // 45
                                                                               VAL_9);
        // - - missing 13,15
        word_t BP_0_1_2_3_4_5_6_7_8_9_10_11_12_14_16_17_18 = op_t::box_plus(BP_0_1_2_3_4_5_6_7_8_10_12_14_16_17_18,            // 46
                                                                            BP_9_11);
        // - - - missing 13
        word_t BP_0_1_2_3_4_5_6_7_8_9_10_11_12_14_15_16_17_18 = op_t::box_plus(BP_0_1_2_3_4_5_6_7_8_9_10_11_12_14_16_17_18,    // 47
                                                                               VAL_15);
        // - - - missing 15
        word_t BP_0_1_2_3_4_5_6_7_8_9_10_11_12_13_14_16_17_18 = op_t::box_plus(BP_0_1_2_3_4_5_6_7_8_9_10_11_12_14_16_17_18,    // 48
                                                                               VAL_13);
        // missing 16
        word_t BP_0_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_17_18 = op_t::box_plus(BP_1_3_5_7_9_11_13_15_17_18,                    // 49
                                                                               BP_0_2_4_6_8_10_12_14);
        // missing 17
        word_t BP_0_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_18 = op_t::box_plus(BP_0_2_4_6_8_10_12_14_16_18,                    // 50
                                                                               BP_1_3_5_7_9_11_13_15);
        // missing 18
        word_t BP_0_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17 = op_t::box_plus(BP_0_2_4_6_8_10_12_14_16,                       // 51
                                                                               BP_1_3_5_7_9_11_13_15_17);

        seq[0]  = BP_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18;
        seq[1]  = BP_0_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18;
        seq[2]  = BP_0_1_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18;
        seq[3]  = BP_0_1_2_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18;
        seq[4]  = BP_0_1_2_3_5_6_7_8_9_10_11_12_13_14_15_16_17_18;
        seq[5]  = BP_0_1_2_3_4_6_7_8_9_10_11_12_13_14_15_16_17_18;
        seq[6]  = BP_0_1_2_3_4_5_7_8_9_10_11_12_13_14_15_16_17_18;
        seq[7]  = BP_0_1_2_3_4_5_6_8_9_10_11_12_13_14_15_16_17_18;
        seq[8]  = BP_0_1_2_3_4_5_6_7_9_10_11_12_13_14_15_16_17_18;
        seq[9]  = BP_0_1_2_3_4_5_6_7_8_10_11_12_13_14_15_16_17_18;
        seq[10] = BP_0_1_2_3_4_5_6_7_8_9_11_12_13_14_15_16_17_18;
        seq[11] = BP_0_1_2_3_4_5_6_7_8_9_10_12_13_14_15_16_17_18;
        seq[12] = BP_0_1_2_3_4_5_6_7_8_9_10_11_13_14_15_16_17_18;
        seq[13] = BP_0_1_2_3_4_5_6_7_8_9_10_11_12_14_15_16_17_18;
        seq[14] = BP_0_1_2_3_4_5_6_7_8_9_10_11_12_13_15_16_17_18;
        seq[15] = BP_0_1_2_3_4_5_6_7_8_9_10_11_12_13_14_16_17_18;
        seq[16] = BP_0_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_17_18;
        seq[17] = BP_0_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_18;
        seq[18] = BP_0_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17;
                  
        
        //printf("box_plus_seq_gen<19, 19>: inputs = (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f)"
        //       "(%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f), "
        //       "(%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f), "
        //       "(%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f), "
        //       "outputs = (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) "
        //       "(%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f)\n",
        //       "(%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f)\n",
        //       "(%.0f %.0f) (%.0f %.0f) (%.0f %.0f) (%.0f %.0f)\n",
        //       __low2float(app[18].f16x2),
        //       __high2float(app[18].f16x2),
        //       __low2float(app[17].f16x2),
        //       __high2float(app[17].f16x2),
        //       __low2float(app[16].f16x2),
        //       __high2float(app[16].f16x2),
        //       __low2float(app[15].f16x2),
        //       __high2float(app[15].f16x2),
        //       __low2float(app[14].f16x2),
        //       __high2float(app[14].f16x2),
        //       __low2float(app[13].f16x2),
        //       __high2float(app[13].f16x2),
        //       __low2float(app[12].f16x2),
        //       __high2float(app[12].f16x2),
        //       __low2float(app[11].f16x2),
        //       __high2float(app[11].f16x2),
        //       __low2float(app[10].f16x2),
        //       __high2float(app[10].f16x2),
        //       __low2float(app[9].f16x2),
        //       __high2float(app[9].f16x2),
        //       __low2float(app[8].f16x2),
        //       __high2float(app[8].f16x2),
        //       __low2float(app[7].f16x2),
        //       __high2float(app[7].f16x2),
        //       __low2float(app[6].f16x2),
        //       __high2float(app[6].f16x2),
        //       __low2float(app[5].f16x2),
        //       __high2float(app[5].f16x2),
        //       __low2float(app[4].f16x2),
        //       __high2float(app[4].f16x2),
        //       __low2float(app[3].f16x2),
        //       __high2float(app[3].f16x2),
        //       __low2float(app[2].f16x2),
        //       __high2float(app[2].f16x2),
        //       __low2float(app[1].f16x2),
        //       __high2float(app[1].f16x2),
        //       __low2float(app[0].f16x2),
        //       __high2float(app[0].f16x2),
        //       __high2float(seq[18].f16x2),
        //       __low2float(seq[18].f16x2),
        //       __high2float(seq[17].f16x2),
        //       __low2float(seq[17].f16x2),
        //       __high2float(seq[16].f16x2),
        //       __low2float(seq[16].f16x2),
        //       __high2float(seq[15].f16x2),
        //       __low2float(seq[15].f16x2),
        //       __high2float(seq[14].f16x2),
        //       __low2float(seq[14].f16x2),
        //       __high2float(seq[13].f16x2),
        //       __low2float(seq[13].f16x2),
        //       __high2float(seq[12].f16x2),
        //       __low2float(seq[12].f16x2),
        //       __high2float(seq[11].f16x2),
        //       __low2float(seq[11].f16x2),
        //       __high2float(seq[10].f16x2),
        //       __low2float(seq[10].f16x2),
        //       __high2float(seq[9].f16x2),
        //       __low2float(seq[9].f16x2),
        //       __high2float(seq[8].f16x2),
        //       __low2float(seq[8].f16x2),
        //       __high2float(seq[7].f16x2),
        //       __low2float(seq[7].f16x2),
        //       __high2float(seq[6].f16x2),
        //       __low2float(seq[6].f16x2),
        //       __high2float(seq[5].f16x2),
        //       __low2float(seq[5].f16x2),
        //       __high2float(seq[4].f16x2),
        //       __low2float(seq[4].f16x2),
        //       __high2float(seq[3].f16x2),
        //       __low2float(seq[3].f16x2),
        //       __high2float(seq[2].f16x2),
        //       __low2float(seq[2].f16x2),
        //       __high2float(seq[1].f16x2),
        //       __low2float(seq[1].f16x2),
        //       __high2float(seq[0].f16x2),
        //       __low2float(seq[0].f16x2));
    }
};

} // namespace ldpc2

#endif // !defined(LDPC2_BOX_PLUS_CUH_INCLUDED_)


