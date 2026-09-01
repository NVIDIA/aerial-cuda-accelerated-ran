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

#if !defined(LDPC2_C2V_CACHE_REGISTER_CUH_INCLUDED_)
#define LDPC2_C2V_CACHE_REGISTER_CUH_INCLUDED_

#include "ldpc2.cuh"

namespace ldpc2
{

////////////////////////////////////////////////////////////////////////
// c2v_cache_register
// Check to variable (C2V) messages are stored in registers. (No loading
// or storing required, but causes register pressure.) Assumes that all
// APP values are in shared memory.
template <int          BG_,
          unsigned int NUM_PARITY_NODES,
          class        TC2V,
          class        TC2VStorage,
          class        TKernelParams>
struct c2v_cache_register
{
    //------------------------------------------------------------------
    typedef TC2V                  c2v_t;
    typedef TC2VStorage           c2v_storage_t;
    typedef typename c2v_t::app_t app_t;
    static const int              BG = BG_;
    //------------------------------------------------------------------
    // c2v_cache_register()
    __device__
    c2v_cache_register(const TKernelParams& /*params*/) { }
    //------------------------------------------------------------------
    // init()
    __device__
    void init()
    {
        #pragma unroll
        for(int i = 0; i < NUM_PARITY_NODES; ++i)
        {
            c2v_storage[i].init();
        }
    }
    //------------------------------------------------------------------
    // Out-of-bounds read
    template <unsigned int CHECK_IDX, int NUM_APP_WORDS, int ROW_DEGREE>
    __device__
    void process_row(const TKernelParams& params,
                     word_t               (&app)[NUM_APP_WORDS],
                     int                  (&app_addr)[ROW_DEGREE],
                     int                  smem_offset)
    {
        static_assert(ROW_DEGREE == row_degree<BG, CHECK_IDX>::value,
                      "APP address size incorrect for row degree");
        static_assert(CHECK_IDX < NUM_PARITY_NODES,
                      "Parity check index exceeds allocation");
        c2v_t c2v;
        c2v.template process_row<CHECK_IDX, TKernelParams, c2v_storage_t>(params,
                                                                          app,
                                                                          app_addr,
                                                                          c2v_storage[CHECK_IDX],
                                                                          smem_offset);
    }
    //------------------------------------------------------------------
    // Data
    c2v_storage_t c2v_storage[NUM_PARITY_NODES];
};

////////////////////////////////////////////////////////////////////////
// c2v_cache_register_core
// Check to variable (C2V) messages are stored in registers, with a
// different type for C2V storage for "core" parity rows. (No loading
// or storing required, but causes register pressure.) Assumes that all
// APP values are in shared memory.
// "Core" parity check nodes in BG1 are high-degree (19), and this
// require more storage to retain the signs.
// Note that the compiler may be able to accomplish the same results
// as specifying two different storage types by eliminating unread
// variables.
template <int          BG_,
          unsigned int NUM_PARITY_NODES,
          class        TC2V,
          class        TC2VStorageCore,
          class        TC2VStorageNonCore,
          class        TKernelParams>
struct c2v_cache_register_core
{
    //------------------------------------------------------------------
    typedef TC2V                  c2v_t;
    typedef TC2VStorageCore       c2v_storage_core_t;
    typedef TC2VStorageNonCore    c2v_storage_noncore_t;
    typedef typename c2v_t::app_t app_t;
    static const int              BG = BG_;
    //------------------------------------------------------------------
    // c2v_cache_register_core()
    __device__
    c2v_cache_register_core(const TKernelParams& /*params*/) { }
    //------------------------------------------------------------------
    // init()
    __device__
    void init()
    {
        #pragma unroll
        for(int i = 0; i < 4; ++i)
        {
            c2v_storage_core[i].init();
        }
        #pragma unroll
        for(int i = 4; i < NUM_PARITY_NODES; ++i)
        {
            c2v_storage_noncore[i-4].init();
        }
    }
    //------------------------------------------------------------------
    // Out-of-bounds read
    template <unsigned int CHECK_IDX, int NUM_APP_WORDS, int ROW_DEGREE>
    __device__
    void process_row(const TKernelParams& params,
                     word_t               (&app)[NUM_APP_WORDS],
                     int                  (&app_addr)[ROW_DEGREE],
                     int                  smem_offset)
    {
        static_assert(ROW_DEGREE == row_degree<BG, CHECK_IDX>::value,
                      "APP address size incorrect for row degree");
        c2v_t c2v;
        if(CHECK_IDX < 4)
        {
            c2v.template process_row<CHECK_IDX, TKernelParams, c2v_storage_core_t>(params,
                                                                                   app,
                                                                                   app_addr,
                                                                                   c2v_storage_core[CHECK_IDX],
                                                                                   smem_offset);
        }
        else
        {
            // coverity[event tag:FALSE]
            c2v.template process_row<CHECK_IDX, TKernelParams, c2v_storage_noncore_t>(params,
                                                                                      app,
                                                                                      app_addr,
                                                                                      c2v_storage_noncore[CHECK_IDX-4],
                                                                                      smem_offset);
        }
                                                                              
    }
    //------------------------------------------------------------------
    // Data
    c2v_storage_core_t    c2v_storage_core[4];
    c2v_storage_noncore_t c2v_storage_noncore[NUM_PARITY_NODES - 4];
};

////////////////////////////////////////////////////////////////////////
// c2v_flat_offset_bg1_words
// Compile-time word offset for BG1 flat C2V layout, computed using
// update-row degrees (not full row degrees).
template <int CHECK_IDX>
struct c2v_flat_offset_bg1_words
{
    static const int value = c2v_flat_offset_bg1_words<CHECK_IDX - 1>::value +
                             div_round_up_t<update_row_degree<1, CHECK_IDX - 1>::value, 2>::value;
};

template <>
struct c2v_flat_offset_bg1_words<0>
{
    static const int value = 0;
};

////////////////////////////////////////////////////////////////////////
// c2v_flat_row_words_bg1
template <int CHECK_IDX>
struct c2v_flat_row_words_bg1
{
    static const int value = div_round_up_t<update_row_degree<1, CHECK_IDX>::value, 2>::value;
};

////////////////////////////////////////////////////////////////////////
// c2v_flat_row_storage_view_bg1
// Lightweight row-storage view that keeps compatibility with
// box_plus_row_proc by exposing row_storage.v.w[i].
template <int NUM_WORDS_>
struct c2v_flat_row_storage_view_bg1
{
    static constexpr int NUM_WORDS = NUM_WORDS_;

    struct value
    {
        word_t* w;
    };

    __device__ explicit c2v_flat_row_storage_view_bg1(word_t* ptr)
    {
        v.w = ptr;
    }

    value v;
};

////////////////////////////////////////////////////////////////////////
// c2v_cache_register_flat_bg1
// Check to variable (C2V) messages are stored in a BG1-specific flat
// register layout, split into [rows 0..3] and [rows 4..45].
// This keeps the existing cache interface and row-proc pipeline.
template <unsigned int NUM_PARITY_NODES,
          class        TC2V,
          class        TKernelParams>
struct c2v_cache_register_flat_bg1
{
    typedef TC2V                  c2v_t;
    typedef typename c2v_t::app_t app_t;
    static const int              BG = 1;

    static constexpr int BG1_MAX_PARITY_NODES = 46;
    static constexpr int BG1_CORE_ROWS         = 4;
    static constexpr int WORDS_ROWS03          = c2v_flat_offset_bg1_words<BG1_CORE_ROWS>::value;
    static constexpr int WORDS_TOTAL           = c2v_flat_offset_bg1_words<BG1_MAX_PARITY_NODES>::value;
    static constexpr int WORDS_ROWS4PLUS       = WORDS_TOTAL - WORDS_ROWS03;

    static_assert(NUM_PARITY_NODES <= BG1_MAX_PARITY_NODES,
                  "BG1 flat cache only supports up to 46 parity rows");
    static_assert(WORDS_ROWS03 == 40,
                  "BG1 flat cache rows[0..3] expected to occupy 40 words");
    static_assert(WORDS_TOTAL == 147,
                  "BG1 flat cache expected total word count mismatch");

    //------------------------------------------------------------------
    // c2v_cache_register_flat_bg1()
    __device__
    c2v_cache_register_flat_bg1(const TKernelParams& /*params*/) {}

    //------------------------------------------------------------------
    // init()
    __device__
    void init()
    {
        #pragma unroll
        for(int i = 0; i < WORDS_ROWS03; ++i)
        {
            c2v_rows03[i].u32 = 0;
        }
        #pragma unroll
        for(int i = 0; i < WORDS_ROWS4PLUS; ++i)
        {
            c2v_rows4plus[i].u32 = 0;
        }
    }

    //------------------------------------------------------------------
    template <unsigned int CHECK_IDX, int NUM_APP_WORDS, int ROW_DEGREE>
    __device__
    void process_row(const TKernelParams& params,
                     word_t               (&app)[NUM_APP_WORDS],
                     int                  (&app_addr)[ROW_DEGREE],
                     int                  smem_offset)
    {
        static_assert(ROW_DEGREE == row_degree<BG, CHECK_IDX>::value,
                      "APP address size incorrect for row degree");
        static_assert(CHECK_IDX < NUM_PARITY_NODES,
                      "Parity check index exceeds allocation");

        constexpr int NUM_ROW_WORDS = c2v_flat_row_words_bg1<CHECK_IDX>::value;
        typedef c2v_flat_row_storage_view_bg1<NUM_ROW_WORDS> row_storage_t;

        row_storage_t row_storage(get_row_word_ptr<CHECK_IDX>());
        c2v_t         c2v;
        c2v.template process_row<CHECK_IDX, TKernelParams, row_storage_t>(params,
                                                                           app,
                                                                           app_addr,
                                                                           row_storage,
                                                                           smem_offset);
    }

private:
    //------------------------------------------------------------------
    template <unsigned int CHECK_IDX>
    __device__
    word_t* get_row_word_ptr()
    {
        constexpr int WORD_OFFSET = c2v_flat_offset_bg1_words<CHECK_IDX>::value;
        if constexpr (CHECK_IDX < BG1_CORE_ROWS)
        {
            static_assert(WORD_OFFSET < WORDS_ROWS03,
                          "BG1 flat cache rows[0..3] offset out of range");
            return (c2v_rows03 + WORD_OFFSET);
        }
        else
        {
            static_assert((WORD_OFFSET - WORDS_ROWS03) < WORDS_ROWS4PLUS,
                          "BG1 flat cache rows[4..45] offset out of range");
            return (c2v_rows4plus + (WORD_OFFSET - WORDS_ROWS03));
        }
    }

    //------------------------------------------------------------------
    // Data
    word_t c2v_rows03[WORDS_ROWS03];
    word_t c2v_rows4plus[WORDS_ROWS4PLUS];
};

} // namespace ldpc2

#endif // !defined(LDPC2_C2V_CACHE_REGISTER_CUH_INCLUDED_)
