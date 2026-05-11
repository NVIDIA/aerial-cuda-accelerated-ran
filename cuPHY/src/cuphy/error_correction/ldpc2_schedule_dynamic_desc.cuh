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

#if !defined(LDPC2_SCHEDULE_DYNAMIC_DESC_CUH_INCLUDED_)
#define LDPC2_SCHEDULE_DYNAMIC_DESC_CUH_INCLUDED_

// Dynamic LDPC schedules are those for which the number of parity nodes
// is not known until runtime. (In contrast, "fixed" schedules will have
// a separate kernel for each number of parity nodes, and the appropriate
// kernel is called from the host.)
// This dynamic schedule also uses a "descriptor" structure, typically
// passed as a kernel argument, to calculate APP addresses.

namespace ldpc2
{

////////////////////////////////////////////////////////////////////////
// row_seq_sync
// Structure to return a (compile-time) value indicating whether a
// parity row requires a syncthreads() call (assuming a sequential
// schedule). If the variable nodes of a row do not overlap the variable
// nodes of the next row, a syncthreads() call is not required.
template <int BG, int CHECK_IDX> struct row_seq_sync        { static const bool value = true; };
template <>                      struct row_seq_sync<1, 16> { static const bool value = false; };
template <>                      struct row_seq_sync<1, 20> { static const bool value = false; };
template <>                      struct row_seq_sync<1, 22> { static const bool value = false; };
template <>                      struct row_seq_sync<1, 24> { static const bool value = false; };
template <>                      struct row_seq_sync<1, 26> { static const bool value = false; };
template <>                      struct row_seq_sync<1, 28> { static const bool value = false; };
template <>                      struct row_seq_sync<1, 30> { static const bool value = false; };
template <>                      struct row_seq_sync<1, 32> { static const bool value = false; };
template <>                      struct row_seq_sync<1, 34> { static const bool value = false; };
template <>                      struct row_seq_sync<1, 36> { static const bool value = false; };
template <>                      struct row_seq_sync<1, 38> { static const bool value = false; };
template <>                      struct row_seq_sync<1, 40> { static const bool value = false; };
template <>                      struct row_seq_sync<1, 42> { static const bool value = false; };
template <>                      struct row_seq_sync<1, 44> { static const bool value = false; };

template <>                      struct row_seq_sync<2, 11> { static const bool value = false; };
template <>                      struct row_seq_sync<2, 17> { static const bool value = false; };
template <>                      struct row_seq_sync<2, 20> { static const bool value = false; };
template <>                      struct row_seq_sync<2, 22> { static const bool value = false; };
template <>                      struct row_seq_sync<2, 24> { static const bool value = false; };
template <>                      struct row_seq_sync<2, 26> { static const bool value = false; };
template <>                      struct row_seq_sync<2, 28> { static const bool value = false; };
template <>                      struct row_seq_sync<2, 30> { static const bool value = false; };
template <>                      struct row_seq_sync<2, 32> { static const bool value = false; };
template <>                      struct row_seq_sync<2, 34> { static const bool value = false; };
template <>                      struct row_seq_sync<2, 36> { static const bool value = false; };
template <>                      struct row_seq_sync<2, 38> { static const bool value = false; };
template <>                      struct row_seq_sync<2, 40> { static const bool value = false; };

////////////////////////////////////////////////////////////////////////
// ldpc_schedule_dynamic_desc_base
template <int                  BG,
          class                TAPPLoc,
          class                TC2VCache,
          class                TKernelParams,
          class                BGDesc,
          int                  MIN_PARITY_ROWS,
          int                  MAX_PARITY_ROWS>
struct ldpc_schedule_dynamic_desc_base
{
    typedef typename TC2VCache::app_t app_t;
    typedef BGDesc                    bg_desc_t;
    //------------------------------------------------------------------
    // ldpc_schedule_dynamic_desc_base()
    __device__
    ldpc_schedule_dynamic_desc_base(const TKernelParams& p,
                                    const bg_desc_t&     bg_desc,
                                    int                  soffset,
                                    unsigned int         t_idx) :
        c2v_cache(p),
        app_addr_gen(p, bg_desc, t_idx),
        params(p),
        num_parity_nodes_i32(static_cast<int>(p.num_parity_nodes)),
        smem_offset(soffset)
    {
        c2v_cache.init();
    }
    //------------------------------------------------------------------
    // ldpc_schedule_dynamic_desc_base()
    __device__
    ldpc_schedule_dynamic_desc_base(char*                smem,
                                    const TKernelParams& p,
                                    const bg_desc_t&     bg_desc,
                                    int                  soffset,
                                    unsigned int         t_idx) :
        c2v_cache(smem, p),
        app_addr_gen(p, bg_desc, t_idx),
        params(p),
        num_parity_nodes_i32(static_cast<int>(p.num_parity_nodes)),
        smem_offset(soffset)
    {
        c2v_cache.init();
    }
    //------------------------------------------------------------------
    // process_row()
    template <int CHECK_IDX>
    __device__
    void process_row()
    {
        int    app_addr[row_degree<BG, CHECK_IDX>::value];      // shared memory (byte) addresses
        word_t app[app_num_words<app_t, BG, CHECK_IDX>::value]; // APP values

        // Note: conditional here causes 5-10% perf decrease for Z values
        // that are multiples of 32. (Other sizes not measured.)
        // Alternative: launch with blockDim.x == Z, and make the hard
        // output function tolerate blockDims that are not a multiple
        // of 32.
        //if(threadIdx.x < params.Z)
        {
            // Generate APP locations/address
            app_addr_gen.template generate<CHECK_IDX>(app_addr);
            // Process the C2V message for this row
            c2v_cache.process_row<CHECK_IDX>(params, app, app_addr, smem_offset);
        }
    }
    template <int CHECK_IDX>
    __device__
    bool iter_sync_check_done()
    {
        // For parity check nodes before the "minimum supported" by the
        // kernel compilation, we always return false to indicate that
        // the iteration is not done, and we sync if the row requires it.
        if((CHECK_IDX + 1) < MIN_PARITY_ROWS)
        {
            if(row_seq_sync<BG, CHECK_IDX>::value)
            {
                __syncthreads();
            }
            return false;
        }
        else if((CHECK_IDX+1) == MAX_PARITY_ROWS)
        {
            // For the maximum supported parity row, we always sync, and
            // we always return true to indicate completion of the
            // iteration.
            __syncthreads();
            return true;
        }
        else
        {
            // All other parity check rows: check the runtime number of
            // parity nodes to indicate completion.
            // In some cases, row APP updates will not overlap with
            // those of the following row. We can skip the sync there,
            // UNLESS it is the last row.
            const bool IS_LAST_ROW = ((CHECK_IDX + 1) == num_parity_nodes_i32);
        
            if(IS_LAST_ROW || row_seq_sync<BG, CHECK_IDX>::value)
            {
                __syncthreads();
            }
            return IS_LAST_ROW;
        }
    }
    //------------------------------------------------------------------
    // Data
    TC2VCache            c2v_cache;
    TAPPLoc              app_addr_gen;
    const TKernelParams& params;
    const int            num_parity_nodes_i32;
    const int            smem_offset;
};

////////////////////////////////////////////////////////////////////////
// ldpc_schedule_dynamic_desc
template <int                  BG,
          class                TAPPLoc,
          class                TC2VCache,
          class                TKernelParams,
          class                BGDesc,
          int                  MIN_PARITY_ROWS,
          int                  MAX_PARITY_ROWS>
struct ldpc_schedule_dynamic_desc;

// ldpc_schedule_dynamic_desc specialization for base graph 1
template <class TAPPLoc, class TC2VCache, class TKernelParams, class BGDesc, int MIN_PARITY_ROWS, int MAX_PARITY_ROWS>
struct ldpc_schedule_dynamic_desc<1,
                                  TAPPLoc,
                                  TC2VCache,
                                  TKernelParams,
                                  BGDesc,
                                  MIN_PARITY_ROWS,
                                  MAX_PARITY_ROWS> :
    ldpc_schedule_dynamic_desc_base<1,
                                    TAPPLoc,
                                    TC2VCache,
                                    TKernelParams,
                                    BGDesc,
                                    MIN_PARITY_ROWS,
                                    MAX_PARITY_ROWS>
{
    typedef ldpc_schedule_dynamic_desc_base<1,
                                           TAPPLoc,
                                           TC2VCache,
                                           TKernelParams,
                                           BGDesc,
                                           MIN_PARITY_ROWS,
                                           MAX_PARITY_ROWS> inherited_t;
    typedef typename TC2VCache::app_t app_t;
    typedef BGDesc                    bg_desc_t;
    //------------------------------------------------------------------
    // ldpc_schedule_dynamic_desc()
    __device__
    ldpc_schedule_dynamic_desc(const TKernelParams& params,
                               const bg_desc_t&     bg_desc,
                               int                  soffset,
                               unsigned int         t_idx) : inherited_t(params, bg_desc, soffset, t_idx)
    {
    }
    //------------------------------------------------------------------
    // ldpc_schedule_dynamic_desc()
    __device__
    ldpc_schedule_dynamic_desc(char*                smem,
                               const TKernelParams& params,
                               const bg_desc_t&     bg_desc,
                               int                  soffset,
                               unsigned int         t_idx) : inherited_t(smem, params, bg_desc, soffset, t_idx)
    {
    }
    //------------------------------------------------------------------
    // do_row()
    template <int CHECK_IDX>
    __device__
    void do_row()
    {
        (*this).template process_row<CHECK_IDX>();
        (void)(*this).template iter_sync_check_done<CHECK_IDX>();
    }
    //------------------------------------------------------------------
    // process_rows_4_9_until()
    __device__
    void process_rows_4_9_until(int last_row)
    {
        if(last_row < 4)
        {
            return;
        }

        switch(last_row)
        {
        case 4:
            do_row<4>();
            break;
        case 5:
            do_row<4>();
            do_row<5>();
            break;
        case 6:
            do_row<4>();
            do_row<5>();
            do_row<6>();
            break;
        case 7:
            do_row<4>();
            do_row<5>();
            do_row<6>();
            do_row<7>();
            break;
        case 8:
            do_row<4>();
            do_row<5>();
            do_row<6>();
            do_row<7>();
            do_row<8>();
            break;
        default: // >= 9
            do_row<4>();
            do_row<5>();
            do_row<6>();
            do_row<7>();
            do_row<8>();
            do_row<9>();
            break;
        }
    }
    //------------------------------------------------------------------
    // process_rows_10_15_until()
    __device__
    void process_rows_10_15_until(int last_row)
    {
        if(last_row < 10)
        {
            return;
        }

        switch(last_row)
        {
        case 10:
            do_row<10>();
            break;
        case 11:
            do_row<10>();
            do_row<11>();
            break;
        case 12:
            do_row<10>();
            do_row<11>();
            do_row<12>();
            break;
        case 13:
            do_row<10>();
            do_row<11>();
            do_row<12>();
            do_row<13>();
            break;
        case 14:
            do_row<10>();
            do_row<11>();
            do_row<12>();
            do_row<13>();
            do_row<14>();
            break;
        default: // >= 15
            do_row<10>();
            do_row<11>();
            do_row<12>();
            do_row<13>();
            do_row<14>();
            do_row<15>();
            break;
        }
    }
    //------------------------------------------------------------------
    // process_rows_16_19_until()
    __device__
    void process_rows_16_19_until(int last_row)
    {
        if(last_row < 16)
        {
            return;
        }

        switch(last_row)
        {
        case 16:
            do_row<16>();
            break;
        case 17:
            do_row<16>();
            do_row<17>();
            break;
        case 18:
            do_row<16>();
            do_row<17>();
            do_row<18>();
            break;
        default: // >= 19
            do_row<16>();
            do_row<17>();
            do_row<18>();
            do_row<19>();
            break;
        }
    }
    //------------------------------------------------------------------
    // process_rows_20_29_until()
    __device__
    void process_rows_20_29_until(int last_row)
    {
        if(last_row < 20)
        {
            return;
        }

        switch(last_row)
        {
        case 20:
            do_row<20>();
            break;
        case 21:
            do_row<20>();
            do_row<21>();
            break;
        case 22:
            do_row<20>();
            do_row<21>();
            do_row<22>();
            break;
        case 23:
            do_row<20>();
            do_row<21>();
            do_row<22>();
            do_row<23>();
            break;
        case 24:
            do_row<20>();
            do_row<21>();
            do_row<22>();
            do_row<23>();
            do_row<24>();
            break;
        case 25:
            do_row<20>();
            do_row<21>();
            do_row<22>();
            do_row<23>();
            do_row<24>();
            do_row<25>();
            break;
        case 26:
            do_row<20>();
            do_row<21>();
            do_row<22>();
            do_row<23>();
            do_row<24>();
            do_row<25>();
            do_row<26>();
            break;
        case 27:
            do_row<20>();
            do_row<21>();
            do_row<22>();
            do_row<23>();
            do_row<24>();
            do_row<25>();
            do_row<26>();
            do_row<27>();
            break;
        case 28:
            do_row<20>();
            do_row<21>();
            do_row<22>();
            do_row<23>();
            do_row<24>();
            do_row<25>();
            do_row<26>();
            do_row<27>();
            do_row<28>();
            break;
        default: // >= 29
            do_row<20>();
            do_row<21>();
            do_row<22>();
            do_row<23>();
            do_row<24>();
            do_row<25>();
            do_row<26>();
            do_row<27>();
            do_row<28>();
            do_row<29>();
            break;
        }
    }
    //------------------------------------------------------------------
    // process_rows_30_39_until()
    __device__
    void process_rows_30_39_until(int last_row)
    {
        if(last_row < 30)
        {
            return;
        }

        switch(last_row)
        {
        case 30:
            do_row<30>();
            break;
        case 31:
            do_row<30>();
            do_row<31>();
            break;
        case 32:
            do_row<30>();
            do_row<31>();
            do_row<32>();
            break;
        case 33:
            do_row<30>();
            do_row<31>();
            do_row<32>();
            do_row<33>();
            break;
        case 34:
            do_row<30>();
            do_row<31>();
            do_row<32>();
            do_row<33>();
            do_row<34>();
            break;
        case 35:
            do_row<30>();
            do_row<31>();
            do_row<32>();
            do_row<33>();
            do_row<34>();
            do_row<35>();
            break;
        case 36:
            do_row<30>();
            do_row<31>();
            do_row<32>();
            do_row<33>();
            do_row<34>();
            do_row<35>();
            do_row<36>();
            break;
        case 37:
            do_row<30>();
            do_row<31>();
            do_row<32>();
            do_row<33>();
            do_row<34>();
            do_row<35>();
            do_row<36>();
            do_row<37>();
            break;
        case 38:
            do_row<30>();
            do_row<31>();
            do_row<32>();
            do_row<33>();
            do_row<34>();
            do_row<35>();
            do_row<36>();
            do_row<37>();
            do_row<38>();
            break;
        default: // >= 39
            do_row<30>();
            do_row<31>();
            do_row<32>();
            do_row<33>();
            do_row<34>();
            do_row<35>();
            do_row<36>();
            do_row<37>();
            do_row<38>();
            do_row<39>();
            break;
        }
    }
    //------------------------------------------------------------------
    // process_rows_40_45_until()
    __device__
    void process_rows_40_45_until(int last_row)
    {
        if(last_row < 40)
        {
            return;
        }

        switch(last_row)
        {
        case 40:
            do_row<40>();
            break;
        case 41:
            do_row<40>();
            do_row<41>();
            break;
        case 42:
            do_row<40>();
            do_row<41>();
            do_row<42>();
            break;
        case 43:
            do_row<40>();
            do_row<41>();
            do_row<42>();
            do_row<43>();
            break;
        case 44:
            do_row<40>();
            do_row<41>();
            do_row<42>();
            do_row<43>();
            do_row<44>();
            break;
        default: // >= 45
            do_row<40>();
            do_row<41>();
            do_row<42>();
            do_row<43>();
            do_row<44>();
            do_row<45>();
            break;
        }
    }
    //------------------------------------------------------------------
    // do_iteration()
    __device__
    void do_iteration()
    {
        (*this).template process_row<0>(); __syncthreads();
        (*this).template process_row<1>(); __syncthreads();
        (*this).template process_row<2>(); __syncthreads();
        (*this).template process_row<3>(); (void)(*this).template iter_sync_check_done<3>();

        const int last_row = (*this).num_parity_nodes_i32 - 1;
        process_rows_4_9_until(last_row);
        process_rows_10_15_until(last_row);
        process_rows_16_19_until(last_row);
        process_rows_20_29_until(last_row);
        process_rows_30_39_until(last_row);
        process_rows_40_45_until(last_row);
    }
};

// ldpc_schedule_dynamic_desc specialization for base graph 2
template <class TAPPLoc, class TC2VCache, class TKernelParams, class BGDesc, int MIN_PARITY_ROWS, int MAX_PARITY_ROWS>
struct ldpc_schedule_dynamic_desc<2,
                                  TAPPLoc,
                                  TC2VCache,
                                  TKernelParams,
                                  BGDesc,
                                  MIN_PARITY_ROWS,
                                  MAX_PARITY_ROWS> :
    ldpc_schedule_dynamic_desc_base<2,
                                    TAPPLoc,
                                    TC2VCache,
                                    TKernelParams,
                                    BGDesc,
                                    MIN_PARITY_ROWS,
                                    MAX_PARITY_ROWS>
{
    typedef ldpc_schedule_dynamic_desc_base<2,
                                           TAPPLoc,
                                           TC2VCache,
                                           TKernelParams,
                                           BGDesc,
                                           MIN_PARITY_ROWS,
                                           MAX_PARITY_ROWS> inherited_t;
    typedef typename TC2VCache::app_t app_t;
    typedef BGDesc                    bg_desc_t;
    //------------------------------------------------------------------
    // ldpc_schedule_dynamic_desc()
    __device__
    ldpc_schedule_dynamic_desc(const TKernelParams& params,
                               const bg_desc_t&     bg_desc,
                               int                  soffset,
                               unsigned int         t_idx) : inherited_t(params, bg_desc, soffset, t_idx)
    {
    }
    //------------------------------------------------------------------
    // ldpc_schedule_dynamic_desc()
    __device__
    ldpc_schedule_dynamic_desc(char*                smem,
                               const TKernelParams& params,
                               const bg_desc_t&     bg_desc,
                               int                  soffset,
                               unsigned int         t_idx) : inherited_t(smem, params, bg_desc, soffset, t_idx)
    {
    }
    //------------------------------------------------------------------
    // do_iteration()
    __device__
    void do_iteration()
    {
        (*this).template process_row<0> (); __syncthreads();
        (*this).template process_row<1> (); __syncthreads();
        (*this).template process_row<2> (); __syncthreads();
        (*this).template process_row<3> (); if((*this).template iter_sync_check_done< 3>()) return;
        (*this).template process_row<4> (); if((*this).template iter_sync_check_done< 4>()) return;
        (*this).template process_row<5> (); if((*this).template iter_sync_check_done< 5>()) return;
        (*this).template process_row<6> (); if((*this).template iter_sync_check_done< 6>()) return;
        (*this).template process_row<7> (); if((*this).template iter_sync_check_done< 7>()) return;
        (*this).template process_row<8> (); if((*this).template iter_sync_check_done< 8>()) return;
        (*this).template process_row<9> (); if((*this).template iter_sync_check_done< 9>()) return;
        (*this).template process_row<10>(); if((*this).template iter_sync_check_done<10>()) return;
        (*this).template process_row<11>(); if((*this).template iter_sync_check_done<11>()) return;
        (*this).template process_row<12>(); if((*this).template iter_sync_check_done<12>()) return;
        (*this).template process_row<13>(); if((*this).template iter_sync_check_done<13>()) return;
        (*this).template process_row<14>(); if((*this).template iter_sync_check_done<14>()) return;
        (*this).template process_row<15>(); if((*this).template iter_sync_check_done<15>()) return;
        (*this).template process_row<16>(); if((*this).template iter_sync_check_done<16>()) return;
        (*this).template process_row<17>(); if((*this).template iter_sync_check_done<17>()) return;
        (*this).template process_row<18>(); if((*this).template iter_sync_check_done<18>()) return;
        (*this).template process_row<19>(); if((*this).template iter_sync_check_done<19>()) return;
        (*this).template process_row<20>(); if((*this).template iter_sync_check_done<20>()) return;
        (*this).template process_row<21>(); if((*this).template iter_sync_check_done<21>()) return;
        (*this).template process_row<22>(); if((*this).template iter_sync_check_done<22>()) return;
        (*this).template process_row<23>(); if((*this).template iter_sync_check_done<23>()) return;
        (*this).template process_row<24>(); if((*this).template iter_sync_check_done<24>()) return;
        (*this).template process_row<25>(); if((*this).template iter_sync_check_done<25>()) return;
        (*this).template process_row<26>(); if((*this).template iter_sync_check_done<26>()) return;
        (*this).template process_row<27>(); if((*this).template iter_sync_check_done<27>()) return;
        (*this).template process_row<28>(); if((*this).template iter_sync_check_done<28>()) return;
        (*this).template process_row<29>(); if((*this).template iter_sync_check_done<29>()) return;
        (*this).template process_row<30>(); if((*this).template iter_sync_check_done<30>()) return;
        (*this).template process_row<31>(); if((*this).template iter_sync_check_done<31>()) return;
        (*this).template process_row<32>(); if((*this).template iter_sync_check_done<32>()) return;
        (*this).template process_row<33>(); if((*this).template iter_sync_check_done<33>()) return;
        (*this).template process_row<34>(); if((*this).template iter_sync_check_done<34>()) return;
        (*this).template process_row<35>(); if((*this).template iter_sync_check_done<35>()) return;
        (*this).template process_row<36>(); if((*this).template iter_sync_check_done<36>()) return;
        (*this).template process_row<37>(); if((*this).template iter_sync_check_done<37>()) return;
        (*this).template process_row<38>(); if((*this).template iter_sync_check_done<38>()) return;
        (*this).template process_row<39>(); if((*this).template iter_sync_check_done<39>()) return;
        (*this).template process_row<40>(); if((*this).template iter_sync_check_done<40>()) return;
        (*this).template process_row<41>(); __syncthreads();
    }
};

} // namespace ldpc2

#endif // !defined(LDPC2_SCHEDULE_DYNAMIC_DESC_CUH_INCLUDED_)

