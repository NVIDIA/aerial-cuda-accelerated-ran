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

#include <cuda.h>
#include <cuda_runtime.h>
#include "aerial-fh-driver/api.hpp"
#include "aerial-fh-driver/oran.hpp"
#include "aerial-fh-driver/sem.hpp"
#include "gpu_blockFP.h" //Compression Decompression repo
#include "gpu_fixed.h"  //Compression Decompression repo
#include "nvlog.hpp"
#include "order_kernel_functions.hpp"
#include "cuda_driver_utils/cuda_driver_utils.hpp"

#pragma nv_diag_suppress 177 // warning #177-D: variable "wqe_id" was declared but never referenced
#pragma nv_diag_suppress 550 // #550-D: variable "wqe_id_last" was set but never used
#include <doca_gpunetio.h>
#include <doca_gpunetio_dev_eth_rxq.cuh>
#include <doca_gpunetio_dev_sem.cuh>
#pragma nv_diag_restore 177
#pragma nv_diag_restore 550

#define TAG (NVLOG_TAG_BASE_CUPHY_DRIVER + 19) // "DRV.ORDER_CUDA"

// The ping-pong order kernels use a single CTA to both receive and process the packets,
// alternating between reading and processing in a ping-pong fashion.
#define ORDER_KERNEL_PINGPONG_NUM_THREADS (320)
#define ORDER_KERNEL_PINGPONG_SRS_NUM_THREADS (1024)

#define ORDER_KERNEL_PUSCH_ONLY (0)
#define ORDER_KERNEL_SRS_ENABLE (1)
#define ORDER_KERNEL_PUSCH (1)
#define ORDER_KERNEL_SRS   (2)
#define ORDER_KERNEL_SRS_AND_PUSCH (ORDER_KERNEL_SRS | ORDER_KERNEL_PUSCH)

#ifdef __cplusplus
extern "C" {
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// DEVICE FUNCTIONS
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ __forceinline__ unsigned long long __globaltimer()
{
    unsigned long long globaltimer;
    // 64-bit GPU global nanosecond timer
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(globaltimer));
    return globaltimer;
}

__global__ void order_kernel_printf_warmup()
{
    if(blockIdx.x == 0 && threadIdx.x == 0)
    {
        printf("[ORDER] printf warmup\n");
    }
}

__device__ __forceinline__ uint16_t get_eaxc_index(uint16_t* eAxC_map, int eAxC_num, uint16_t eAxC_id)
{
    for(int i = 0; i < eAxC_num; i++)
    {
        if(eAxC_map[i] == eAxC_id)
            return i;
    }

    return 0;
}

// CWE-787 mitigation for the O-RAN U-plane copy/decompress paths below.
//
// oran_get_offset_from_hdr() converts symbolId/startPrb (read from an untrusted
// fronthaul packet header) plus num_prb into a byte offset of the form
//   flow_index*symbols_x_slot*prbs_per_symbol*prb_size
//     + symbol_id *prbs_per_symbol*prb_size
//     + start_prb *prb_size
// and the subsequent copy/decompress writes num_prb*prb_size bytes there. The
// per-eAxC region the buffer is sized for spans [0, symbols_x_slot) symbols and
// [0, prbs_per_symbol) PRBs, and get_eaxc_index() already clamps flow_index into
// [0, eAxC_num). So a write stays inside the allocation iff the symbol index and
// the [start_prb, start_prb+num_prb) PRB span stay inside that region. A packet
// that violates this (e.g. symbolId=63, startPrbc=1023, numPrbc=255) would write
// past the antenna buffer into adjacent GPU allocations. These helpers return
// false for such packets so the caller can drop the write (num_prb=0).
//
// The predicate depends only on packet-header fields + uniform geometry, so it is
// warp-uniform and safe to use without introducing divergence at collectives.
//
// Two overloads: the symbolId overload is used where the caller already holds
// symbolId in a local; the packet overload forwards, reading symbolId from the
// header for callers that don't have it handy. The two are equivalent in codegen
// (ptxas hoists the repeated header read), so pick whichever reads better.
//
// These are C++ overloads, so force C++ linkage: the surrounding order kernels
// live in an extern "C" block (opened above), which does not permit overloaded
// functions ("more than one instance ... has C linkage").
// The predicates take unsigned parameters so the (>= 0) arms disappear by
// construction instead of relying on the compiler to fold them. symbol_id,
// start_prb and num_prb do originate in unsigned packet-header fields, but the
// geometry arguments (symbols_x_slot, prbs_per_symbol, start_symbol) are still
// declared int at the kernel boundary and convert implicitly here. That is safe
// only because they are positive values derived from the cell config, never
// attacker-controlled. A caller passing a *computed* start must reject the
// negative case itself before converting - see adj_start in order_kernel_doca.
extern "C++" {

// PRB-span half of the predicate. start_prb and num_prb come from <=16-bit header
// fields, so the sum cannot overflow uint32_t.
[[nodiscard]] __device__ __forceinline__ bool ul_oran_prb_span_in_bounds(
    uint32_t start_prb, uint32_t num_prb, uint32_t prbs_per_symbol)
{
    return (start_prb + num_prb) <= prbs_per_symbol;
}

[[nodiscard]] __device__ __forceinline__ bool ul_oran_offset_in_bounds(
    uint32_t symbol_id, uint32_t start_prb, uint32_t num_prb, uint32_t symbols_x_slot, uint32_t prbs_per_symbol)
{
    return (symbol_id < symbols_x_slot) &&
           ul_oran_prb_span_in_bounds(start_prb, num_prb, prbs_per_symbol);
}

[[nodiscard]] __device__ __forceinline__ bool ul_oran_offset_in_bounds(
    uint8_t* pkt, uint32_t start_prb, uint32_t num_prb, uint32_t symbols_x_slot, uint32_t prbs_per_symbol)
{
    return ul_oran_offset_in_bounds((uint32_t)oran_umsg_get_symbol_id(pkt), start_prb, num_prb,
                                    symbols_x_slot, prbs_per_symbol);
}

// SRS variant: the offset is computed relative to srs start symbol, so the
// effective symbol index is (symbolId - start_symbol).
[[nodiscard]] __device__ __forceinline__ bool ul_oran_srs_offset_in_bounds(
    uint32_t symbol_id, uint32_t start_prb, uint32_t num_prb, uint32_t symbols_x_slot, uint32_t prbs_per_symbol, uint32_t start_symbol)
{
    // The lower bound (symbolId >= start_symbol) is still required -- a packet
    // with symbolId < start_symbol would otherwise index *before* the SRS buffer.
    // Unsigned arithmetic enforces it for free: the subtraction wraps to a huge
    // value, which fails the < comparison. Do not "fix" this to signed.
    const uint32_t eff_symbol = symbol_id - start_symbol;
    return (eff_symbol < symbols_x_slot) &&
           ul_oran_prb_span_in_bounds(start_prb, num_prb, prbs_per_symbol);
}

[[nodiscard]] __device__ __forceinline__ bool ul_oran_srs_offset_in_bounds(
    uint8_t* pkt, uint32_t start_prb, uint32_t num_prb, uint32_t symbols_x_slot, uint32_t prbs_per_symbol, uint32_t start_symbol)
{
    return ul_oran_srs_offset_in_bounds((uint32_t)oran_umsg_get_symbol_id(pkt), start_prb, num_prb,
                                        symbols_x_slot, prbs_per_symbol, start_symbol);
}

// Symbols below start_symbol wrap to a huge unsigned value, so one compare bounds both ends.
[[nodiscard]] __device__ __forceinline__ bool ul_oran_srs_symbol_in_window(
    uint32_t symbol_id, uint32_t start_symbol, uint32_t symbols_x_slot)
{
    return (symbol_id - start_symbol) < symbols_x_slot;
}
} // extern "C++"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// Order Kernel multi block
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ void ib_barrier(int * barrier_flag, int barrier_signal, int& barrier_idx) {

    atomicAdd(barrier_flag, 1);
    while(atomicAdd(barrier_flag, 0) < (barrier_signal * barrier_idx))
        ;
    barrier_idx++;
}

__device__ int order_kernel_cpu_init_comms_wait_packets(
int cell_id,
uint8_t frameId,
uint8_t subframeId,
uint8_t slotId,
uint8_t& start_loop, 
unsigned long long& kernel_start, 
unsigned long long& first_packet_start,
uint32_t* ready_list,
uint32_t& ready_sync_sh,
int& sem_idx_order,
int  last_sem_idx_order,
uint32_t timeout_no_pkt_ns,
uint32_t timeout_first_pkt_ns
)
{
    int ret = 0;
    unsigned long long current_time = 0;
    int ready_local;

    while(1)
    {
        current_time = __globaltimer();

        //Max timeout to wait for the very first slot packet
        if((start_loop == 1 && ((current_time - kernel_start) > timeout_no_pkt_ns)))
        {
            printf("Cell %d Order kernel No pkt timeout after %d ms F%dS%dS%d\n",cell_id,timeout_no_pkt_ns,frameId, subframeId, slotId);
            ready_sync_sh = (int)SYNC_PACKET_STATUS_EXIT;
            __threadfence();
            ret = -1;
            break;
        }

        //Max timeout to wait to receive all expected packets for this slot
        if((start_loop == 2 && ((current_time - first_packet_start) > timeout_first_pkt_ns)))
        {
            printf("Cell %d Order kernel Partial pkt timeout after %d ms F%dS%dS%d\n", cell_id,timeout_first_pkt_ns,frameId, subframeId, slotId);
            ready_sync_sh = (int)SYNC_PACKET_STATUS_EXIT;
            __threadfence();
            ret = -2;
            break;
        }

        if(start_loop == 0)
        {
            sem_idx_order   = last_sem_idx_order;
            start_loop       = 1;
        }
        else
        {
            // printf("Order kernel cell %d polling on item %d, ordered %d/%d addr %lx\n", cell_id, rx_queue_index, ordered_prbs, prb_x_slot, &ready_list[rx_queue_index]);
            ready_local = ACCESS_ONCE(ready_list[sem_idx_order]);
            if(ready_local == SYNC_PACKET_STATUS_READY)
            {
                if(start_loop == 1)
                {
                    start_loop = 2;
                    first_packet_start  = __globaltimer();
                }
                ready_sync_sh     = (int)SYNC_PACKET_STATUS_READY;
                //printf("Order kernel block %d cell %d ready on item %d at F%dS%dS%d\n", blockIdx.x,cell_id,sem_idx_order,frameId, subframeId, slotId);
                __threadfence();
                ret = 0;
                break;
            }
        }
    }

    return ret;
}

//Determines the number of slots between the specified ORAN details (ORAN details 2 - ORAN details 1)
//Result is in the range of [-2560, 2560]
//Positive means ORAN details 2 is later in time than ORAN details 1
//Negative means ORAN details 2 is earlier in time than ORAN details 1
inline __device__ int32_t calculate_slot_difference(
    uint8_t frameId1, uint8_t frameId2,
    uint8_t subframeId1, uint8_t subframeId2,
    uint8_t slotId1, uint8_t slotId2
) {
    // Calculate frame difference accounting for wrap-around
    // Using (ORAN_FRAME_WRAP/2) ensures we get the shortest path around the wrap
    int32_t frame_diff = ((frameId2 - frameId1 + (ORAN_FRAME_WRAP/2)) % ORAN_FRAME_WRAP) - (ORAN_FRAME_WRAP/2);

    // Calculate absolute slot positions within their respective frames
    int32_t slot_count1 = (ORAN_SLOT_WRAP * subframeId1 + slotId1);
    int32_t slot_count2 = (ORAN_SLOT_WRAP * subframeId2 + slotId2);

    // Calculate slot difference based on frame difference
    int32_t slot_diff = slot_count2 - slot_count1;

    // Combine frame and slot differences
    return frame_diff * ORAN_SLOTS_PER_FRAME + slot_diff;
}


__global__ void order_kernel_doca(
	/* DOCA objects */
	struct doca_gpu_eth_rxq *doca_rxq,
	struct doca_gpu_semaphore_gpu *sem_gpu,
	const uint16_t sem_order_num,

	/* Cell */
	const int		cell_id,
    const int		ru_type,
	uint32_t		*start_cuphy_d,
	uint32_t		*exit_cond_d,
	uint32_t		*last_sem_idx_rx_h,
	uint32_t		*last_sem_idx_order_h,
	const int		comp_meth,
    const int		bit_width,
	const float		beta,
	const int		prb_size,

	/* Timeout */
	const uint32_t	timeout_no_pkt_ns,
	const uint32_t	timeout_first_pkt_ns,
    const uint32_t	max_rx_pkts,

	/* ORAN */
	const uint8_t		frameId,
	const uint8_t		subframeId,
	const uint8_t		slotId,

	/* Order kernel specific */
	int			*barrier_flag,
	uint8_t		*done_shared,

	/* Timer */
	uint32_t 		*early_rx_packets,
	uint32_t 		*on_time_rx_packets,
	uint32_t 		*late_rx_packets,
	uint64_t		slot_start,
	uint64_t		ta4_min_ns,
	uint64_t		ta4_max_ns,
	uint64_t		slot_duration,

	/* PUSCH */
	uint16_t		*pusch_eAxC_map,
	int			pusch_eAxC_num,
	uint8_t		*pusch_buffer,
	int			pusch_prb_x_slot,
	int			pusch_prb_x_symbol,
	int			pusch_prb_x_symbol_x_antenna,
	int			pusch_symbols_x_slot,
	int			pusch_prb_x_port_x_symbol,
	uint32_t		*pusch_ordered_prbs,

	/* PRACH */
	uint16_t 		*prach_eAxC_map,
	int			prach_eAxC_num,
	uint8_t		*prach_buffer_0,
	uint8_t		*prach_buffer_1,
	uint8_t 		*prach_buffer_2,
	uint8_t 		*prach_buffer_3,
	uint16_t		prach_section_id_0,
	uint16_t		prach_section_id_1,
	uint16_t		prach_section_id_2,
	uint16_t		prach_section_id_3,
	int			prach_prb_x_slot,
	int			prach_prb_x_symbol,
	int			prach_prb_x_symbol_x_antenna,
	int			prach_symbols_x_slot,
	int			prach_prb_x_port_x_symbol,
	uint32_t		*prach_ordered_prbs)
{
	int laneId = threadIdx.x % 32;
	int warpId = threadIdx.x / 32;
	int nwarps = blockDim.x / 32;
	uint8_t first_packet = 0;
	unsigned long long first_packet_start = 0;
	unsigned long long current_time = 0;
	unsigned long long kernel_start = __globaltimer();
	uint8_t* pkt_offset_ptr, *gbuf_offset_ptr;
	uint8_t* buffer;
	int prb_x_slot=pusch_prb_x_slot+prach_prb_x_slot;
	doca_error_t ret = (doca_error_t)0;
	// Restart from last semaphore item
	int sem_idx_rx = *last_sem_idx_rx_h;
	int sem_idx_order = *last_sem_idx_order_h;
	int last_sem_idx_order = *last_sem_idx_order_h;
	const uint64_t timeout_ns = 100000;
	int  barrier_idx = 1, barrier_signal = gridDim.x;
	//unsigned long long t0, t1, t2, t3, t4, t5, t6, t7;

	__shared__ uint32_t rx_pkt_num;
	__shared__ uint64_t rx_buf_idx;
	__shared__ uint32_t done_shared_sh;
	__shared__ uint32_t last_stride_idx;
	__shared__ uint64_t rx_timestamp;
	__shared__ struct doca_gpu_dev_eth_rxq_attr out_attr_sh[512];

    /* Note: WIP for a more generic approach to calculate and pass the startRB from the cuPHY-CP */
	uint16_t startPRB_offset_idx_0 = 0 * 12 * prb_size;
	uint16_t startPRB_offset_idx_1 = 1 * 12 * prb_size;
	uint16_t startPRB_offset_idx_2 = 2 * 12 * prb_size;
	uint16_t startPRB_offset_idx_3 = 3 * 12 * prb_size;

	if(blockIdx.x == 1 && threadIdx.x == 0) {
		DOCA_GPUNETIO_VOLATILE(done_shared_sh) = 1;
		DOCA_GPUNETIO_VOLATILE(pusch_ordered_prbs[0]) = 0;
		DOCA_GPUNETIO_VOLATILE(prach_ordered_prbs[0]) = 0;
		DOCA_GPUNETIO_VOLATILE(rx_pkt_num) = 0;
	}
	__syncthreads();

	while (DOCA_GPUNETIO_VOLATILE(*exit_cond_d) == ORDER_KERNEL_RUNNING) {
		/* Block 0 receives packets and forward them to Block 1 */
		if(blockIdx.x == 0) {
			if (threadIdx.x == 0) {
				DOCA_GPUNETIO_VOLATILE(rx_pkt_num) = 0;

				current_time = __globaltimer();
				if (first_packet && ((current_time - first_packet_start) > timeout_first_pkt_ns)) {
					printf("%d Cell %d Order kernel sem_idx_rx %d last_sem_idx_rx %d PUSCH PRBs %d/%d PRACH PRBs %d/%d. Wait first packet timeout after %d ns F%dS%dS%d\n",__LINE__,
						cell_id, sem_idx_rx, *last_sem_idx_rx_h,
						DOCA_GPUNETIO_VOLATILE(pusch_ordered_prbs[0]), pusch_prb_x_slot,
						DOCA_GPUNETIO_VOLATILE(prach_ordered_prbs[0]), prach_prb_x_slot,
						timeout_first_pkt_ns, frameId, subframeId, slotId);

					DOCA_GPUNETIO_VOLATILE(*exit_cond_d) = ORDER_KERNEL_EXIT_TIMEOUT_RX_PKT;
				} else if (((current_time - kernel_start) > timeout_no_pkt_ns)) {
					printf("%d Cell %d Order kernel sem_idx_rx %d last_sem_idx_rx %d PUSCH PRBs %d/%d PRACH PRBs %d/%d. Receive more packets timeout after %d ns F%dS%dS%d\n",__LINE__,
						cell_id, sem_idx_rx, *last_sem_idx_rx_h,
						DOCA_GPUNETIO_VOLATILE(pusch_ordered_prbs[0]), pusch_prb_x_slot,
						DOCA_GPUNETIO_VOLATILE(prach_ordered_prbs[0]), prach_prb_x_slot,
						timeout_no_pkt_ns, frameId, subframeId, slotId);

					DOCA_GPUNETIO_VOLATILE(*exit_cond_d) = ORDER_KERNEL_EXIT_TIMEOUT_NO_PKT;
				}
				// printf("Timeout check Done Exit condition (%d)\n",DOCA_GPUNETIO_VOLATILE(*exit_cond_d));
			}
			__syncthreads();

			if (DOCA_GPUNETIO_VOLATILE(*exit_cond_d) != ORDER_KERNEL_RUNNING)
				break;

/*
			if (threadIdx.x == 0 || threadIdx.x == 32 || threadIdx.x == 64)
				t0 = __globaltimer();
*/
		// Add rx timestamp
		if (threadIdx.x == 0) {
			ret = doca_gpu_dev_eth_rxq_recv<DOCA_GPUNETIO_ETH_EXEC_SCOPE_THREAD,DOCA_GPUNETIO_ETH_MCST_DISABLED,DOCA_GPUNETIO_ETH_NIC_HANDLER_AUTO,DOCA_GPUNETIO_ETH_RX_ATTR_NONE>(doca_rxq, max_rx_pkts, timeout_ns, &rx_buf_idx, &rx_pkt_num, out_attr_sh);
		/* If any thread returns receive error, the whole execution stops */
		if (ret != DOCA_SUCCESS) {
			doca_gpu_dev_semaphore_set_status(sem_gpu, sem_idx_rx, DOCA_GPU_SEMAPHORE_STATUS_ERROR);
			DOCA_GPUNETIO_VOLATILE(*exit_cond_d) = ORDER_KERNEL_EXIT_ERROR_LEGACY;
				printf("Exit from rx kernel block %d threadIdx %d ret %d sem_idx_rx %d\n",
						blockIdx.x, threadIdx.x, ret, sem_idx_rx);
			}
		}
/*
			if (threadIdx.x == 0 || threadIdx.x == 32 || threadIdx.x == 64)
				t1 = __globaltimer();
*/
			__threadfence();
			__syncthreads();
/*
			if (threadIdx.x == 0 || threadIdx.x == 32 || threadIdx.x == 64)
				t2 = __globaltimer();
*/
		if (DOCA_GPUNETIO_VOLATILE(rx_pkt_num) > 0) {
			if (threadIdx.x == 0)
				doca_gpu_dev_semaphore_set_packet_info(sem_gpu, sem_idx_rx, DOCA_GPU_SEMAPHORE_STATUS_READY, rx_pkt_num, rx_buf_idx);
			sem_idx_rx = (sem_idx_rx+1) & (sem_order_num - 1);
/*
				if (threadIdx.x == 0 || threadIdx.x == 32 || threadIdx.x == 64) {
					printf("ThreadIdx %d Rx %d sem_idx_rx %d pkts took %lld us sync %lld us\n",
						threadIdx.x, DOCA_GPUNETIO_VOLATILE(rx_pkt_num), sem_idx_rx, t1-t0, t2-t1);
				}
*/
				if (threadIdx.x == 0 && first_packet == 0) {
					first_packet = 1;
					first_packet_start  = __globaltimer();
				}
			}

/*
			if (threadIdx.x == 0 || threadIdx.x == 32 || threadIdx.x == 64) {
				if (DOCA_GPUNETIO_VOLATILE(rx_pkt_num) > 0)
				printf("ThreadIdx %d Rx %d sem_idx_rx %d pkts took %lld us sync %lld us\n",
					threadIdx.x, DOCA_GPUNETIO_VOLATILE(rx_pkt_num), sem_idx_rx, t1-t0, t2-t1);
			}
			__syncthreads();
*/

   		} else {
			/* Block 1 waits on semaphore for new packets and process them */

			/* Semaphore wait */
		if (threadIdx.x == 0) {
			do {
				ret = doca_gpu_dev_semaphore_get_packet_info_status(sem_gpu, sem_idx_order, DOCA_GPU_SEMAPHORE_STATUS_READY, &rx_pkt_num, &rx_buf_idx);
			} while (ret == DOCA_ERROR_NOT_FOUND && DOCA_GPUNETIO_VOLATILE(*exit_cond_d) == ORDER_KERNEL_RUNNING);
			}
            // COVERITY_DEVIATION: blockIdx.x is uniform across all threads in a block.
            // All threads in this block will reach __syncthreads(), no actual divergence.
            // coverity[CUDA.DIVERGENCE_AT_COLLECTIVE_OPERATION]
			__syncthreads();

			/* Check error or exit condition */
			if (DOCA_GPUNETIO_VOLATILE(*exit_cond_d) != ORDER_KERNEL_RUNNING) {
				if (threadIdx.x == 0) {
					// printf("EXIT FROM Block %d Sem %d pkt_addr %lx pkt_num %d status_proxy %d exit %d\n",
					// 	blockIdx.x, sem_idx_order, rx_pkt_addr, rx_pkt_num, status_proxy, *exit_cond_d);
					DOCA_GPUNETIO_VOLATILE(*exit_cond_d) = ORDER_KERNEL_EXIT_ERROR_LEGACY;
				}
				break;
			}


			if(DOCA_GPUNETIO_VOLATILE(rx_pkt_num) == 0)
				continue;

		/* Order & decompress packets */
		for (uint32_t pkt_idx = warpId; pkt_idx < rx_pkt_num; pkt_idx += nwarps) {
            uint8_t *pkt_thread = (uint8_t*)doca_gpu_dev_eth_rxq_get_pkt_addr(doca_rxq, rx_buf_idx + pkt_idx);
				uint16_t section_id_pkt  = oran_umsg_get_section_id(pkt_thread);
				uint8_t frameId_pkt     = oran_umsg_get_frame_id(pkt_thread);
				uint8_t subframeId_pkt  = oran_umsg_get_subframe_id(pkt_thread);
				uint8_t slotId_pkt      = oran_umsg_get_slot_id(pkt_thread);
				#if 0
				if (laneId == 0 && warpId == 0)
					printf("pkt_thread %lx: src %x:%x:%x:%x:%x:%x dst %x:%x:%x:%x:%x:%x proto %x:%x vlan %x:%x ecpri %x:%x hdr %x:%x:%x:%x:%x:%x:%x:%x\n",
						// "pkt_idx %d stride_start_idx %d section_id_pkt %d/%d frameId_pkt %d/%d subframeId_pkt %d/%d slotId_pkt %d/%d\n",
						pkt_thread,
						pkt_thread[0], pkt_thread[1], pkt_thread[2], pkt_thread[3], pkt_thread[4], pkt_thread[5],
						pkt_thread[6], pkt_thread[7], pkt_thread[8], pkt_thread[9], pkt_thread[10], pkt_thread[11],
						pkt_thread[12], pkt_thread[13], pkt_thread[14], pkt_thread[15],
						pkt_thread[16], pkt_thread[17],
						pkt_thread[18], pkt_thread[19], pkt_thread[20], pkt_thread[21], pkt_thread[22], pkt_thread[23],
						pkt_idx, stride_start_idx, section_id_pkt, prach_section_id_0, frameId_pkt, frameId, subframeId_pkt, subframeId, slotId_pkt, slotId);
				#endif

				// if (threadIdx.x == 0)
				// 	t2 = __globaltimer();
				/* If current frame */
				if (
					((frameId_pkt == frameId) && (subframeId_pkt == subframeId) && (slotId_pkt > slotId)) ||
					((frameId_pkt == frameId) && (subframeId_pkt == ((subframeId+1) % 10)))
				) {
					if (laneId == 0 && DOCA_GPUNETIO_VOLATILE(done_shared_sh) == 1) {
						 //printf("F%d/%d SF %d/%d SL %d/%d\n", frameId_pkt, frameId, subframeId_pkt, subframeId, slotId_pkt, slotId);
						DOCA_GPUNETIO_VOLATILE(done_shared_sh) = 0;
					}
				} else {
					if (!((frameId_pkt == frameId) && (subframeId_pkt == subframeId) && (slotId_pkt == slotId)))
						continue;
				  /* if this is the right slot, order & decompress */
					uint16_t num_prb = oran_umsg_get_num_prb(pkt_thread);
                    if(num_prb==0)
                    {
                        num_prb = ORAN_MAX_PRB_X_SLOT;
                    }
					// Check rx timestamp
					#if 0
						uint8_t symbol_id_pkt = oran_umsg_get_symbol_id(pkt_thread);
						uint64_t packet_early_thres = slot_start + ta4_min_ns + (slot_duration * symbol_id_pkt / ORAN_MAX_SYMBOLS);
						uint64_t packet_late_thres  = slot_start + ta4_max_ns + (slot_duration * symbol_id_pkt / ORAN_MAX_SYMBOLS);

						if (rx_timestamp < packet_early_thres)
							atomicAdd(early_rx_packets, 1);
						else if (rx_timestamp > packet_late_thres)
							atomicAdd(late_rx_packets, 1);
						else
							atomicAdd(on_time_rx_packets, 1);
					#endif

					pkt_offset_ptr  = pkt_thread + ORAN_IQ_HDR_SZ;

					if(section_id_pkt < prach_section_id_0)
					{
						buffer = pusch_buffer;
						gbuf_offset_ptr = buffer + oran_get_offset_from_hdr(pkt_thread, (uint16_t)get_eaxc_index(pusch_eAxC_map, pusch_eAxC_num,
																		oran_umsg_get_flowid(pkt_thread)),
																		pusch_symbols_x_slot, pusch_prb_x_port_x_symbol, prb_size);
					}
					else {
						if(section_id_pkt == prach_section_id_0) buffer = prach_buffer_0;
						else if(section_id_pkt == prach_section_id_1) buffer = prach_buffer_1;
						else if(section_id_pkt == prach_section_id_2) buffer = prach_buffer_2;
						else if(section_id_pkt == prach_section_id_3) buffer = prach_buffer_3;
						else {
							// Invalid section_id - skip this packet
                            printf("ERROR invalid section_id %d for Cell %d F%dS%dS%d\n", section_id_pkt, blockIdx.x, frameId_pkt, subframeId_pkt, slotId_pkt);
							continue;
						}
						gbuf_offset_ptr = buffer + oran_get_offset_from_hdr(pkt_thread, (uint16_t)get_eaxc_index(prach_eAxC_map, prach_eAxC_num,
																		oran_umsg_get_flowid(pkt_thread)),
																		prach_symbols_x_slot, prach_prb_x_port_x_symbol, prb_size);
						 /* prach_buffer_x_cell is populated based on number of PRACH PDU's, hence the index can be used as "Frequency domain occasion index"
                                                  and mutiplying with num_prb i.e. NRARB=12 (NumRB's (PRACH SCS=30kHz) for each FDM ocassion) will yeild the corrosponding PRB start for each Frequency domain index
                                                  Note: WIP for a more generic approach to caluclate and pass the startRB from the cuPHY-CP */
                                               if(section_id_pkt == prach_section_id_0) gbuf_offset_ptr -= startPRB_offset_idx_0;
                                               else if(section_id_pkt == prach_section_id_1) gbuf_offset_ptr -= startPRB_offset_idx_1;
                                               else if(section_id_pkt == prach_section_id_2) gbuf_offset_ptr -= startPRB_offset_idx_2;
                                               else if(section_id_pkt == prach_section_id_3) gbuf_offset_ptr -= startPRB_offset_idx_3;
					}

					// if (threadIdx.x == 0)
					// 	t3 = __globaltimer();
					// CWE-787: drop copy/decompress if untrusted symbolId/startPrb/numPrb
					// would index past the per-eAxC antenna buffer. For PRACH the pointer is
					// shifted back by startPRB_offset_idx_* (bytes), so validate the effective
					// post-shift start PRB, which also rejects an underflow before the buffer.
					{
						const int chk_sxs = (section_id_pkt < prach_section_id_0) ? pusch_symbols_x_slot : prach_symbols_x_slot;
						const int chk_pps = (section_id_pkt < prach_section_id_0) ? pusch_prb_x_port_x_symbol : prach_prb_x_port_x_symbol;
						int shift_bytes = 0;
						if(section_id_pkt == prach_section_id_1)      shift_bytes = startPRB_offset_idx_1;
						else if(section_id_pkt == prach_section_id_2) shift_bytes = startPRB_offset_idx_2;
						else if(section_id_pkt == prach_section_id_3) shift_bytes = startPRB_offset_idx_3;
						const int adj_start = (int)oran_umsg_get_start_prb(pkt_thread) - (shift_bytes / prb_size);
						// adj_start is the only computed (signed) start in this file, so its
						// negative case is rejected here rather than inside the helper: the
						// helper now takes unsigned, and a negative value would convert to a
						// huge uint32_t that could wrap back into range across the span add.
						if(adj_start < 0 ||
						   !ul_oran_offset_in_bounds(pkt_thread, (uint32_t)adj_start, num_prb, chk_sxs, chk_pps))
							num_prb = 0;
					}
                    if(comp_meth == static_cast<uint8_t>(aerial_fh::UserDataCompressionMethod::BLOCK_FLOATING_POINT))
                    {
                        if(bit_width == BFP_NO_COMPRESSION) // BFP with 16 bits is a special case and uses FP16, so copy the values
                        {
                            for(int index_copy = laneId; index_copy < (num_prb * prb_size); index_copy += 32)
                                gbuf_offset_ptr[index_copy] = pkt_offset_ptr[index_copy];
                        }
                        else
                        {
                            decompress_scale_blockFP<false>((unsigned char*)pkt_offset_ptr, (__half*)gbuf_offset_ptr, beta, num_prb, bit_width, (int)(threadIdx.x & 31), 32);
                        }
                    } else // aerial_fh::UserDataCompressionMethod::NO_COMPRESSION
                    {
                        decompress_scale_fixed<false>(pkt_offset_ptr, (__half*)gbuf_offset_ptr, beta, num_prb, bit_width, (int)(threadIdx.x & 31), 32);
                    }
					// if (threadIdx.x == 0)
					// 	t4 = __globaltimer();

					// Only first warp thread increases the number of tot PRBs
					if(laneId == 0) {
						int oprb_ch1 = 0;
						int oprb_ch2 = 0;

						if(section_id_pkt < prach_section_id_0) {
							oprb_ch1 = atomicAdd(pusch_ordered_prbs, num_prb);
							oprb_ch2 = atomicAdd(prach_ordered_prbs, 0);
						} else {
							oprb_ch1 = atomicAdd(pusch_ordered_prbs, 0);
							oprb_ch2 = atomicAdd(prach_ordered_prbs, num_prb);
						}

						// printf("Lane ID = %d Warp ID = %d oprb_ch1 %d oprb_ch2 %d num_prb %d prb_x_slot %d\n",
						//     laneId, warpId, oprb_ch1, oprb_ch2, num_prb, prb_x_slot);
						if(oprb_ch1 + oprb_ch2 + num_prb >= prb_x_slot)
							DOCA_GPUNETIO_VOLATILE(*exit_cond_d) = ORDER_KERNEL_EXIT_PRB;
					}
/*
					if (threadIdx.x == 0) {
						t5 = __globaltimer();
						printf("Sem %lld ns rx_pkt_num %d Compress %lld ns Prbs %lld ns\n",
								t1-t0, rx_pkt_num, t4-t3, t5-t4);
					}
*/
				}
			}
			__threadfence();
			__syncthreads();

		if(threadIdx.x == 0) {
			if(DOCA_GPUNETIO_VOLATILE(done_shared_sh) == 1) {
				doca_gpu_dev_semaphore_set_status(sem_gpu, last_sem_idx_order, DOCA_GPU_SEMAPHORE_STATUS_DONE);
				last_sem_idx_order = (last_sem_idx_order + 1) & (sem_order_num - 1);
					__threadfence();
				}
				DOCA_GPUNETIO_VOLATILE(rx_pkt_num) = 0;
			}
			__syncthreads();

			sem_idx_order = (sem_idx_order+1) & (sem_order_num - 1);
		}
	}

	///////////////////////////////////////////////////////////
	// Inter-block barrier
	///////////////////////////////////////////////////////////
	__threadfence();
	__syncthreads();
	if(threadIdx.x == 0)
		ib_barrier(&(barrier_flag[0]), barrier_signal, barrier_idx);
	__syncthreads();
	///////////////////////////////////////////////////////////

	if (threadIdx.x == 0) {
		if(blockIdx.x == 0) {
			DOCA_GPUNETIO_VOLATILE(*last_sem_idx_rx_h) = sem_idx_rx;
			__threadfence_system();
		} else if(blockIdx.x == 1) {
			DOCA_GPUNETIO_VOLATILE(*last_sem_idx_order_h) = last_sem_idx_order;
			DOCA_GPUNETIO_VOLATILE(*start_cuphy_d) = 1;
			// t7 = __globaltimer();
			// printf("BlockIdx 1 thread 0 %lld ns\n", t7-t6);
			__threadfence_system();
			/* printf("Order kernel cell %d exit after %d/%d items, rx item: %d last rx item: %d frame %d subframe %d slot %d buffer %x%x%x%x %x%x%x%x %x%x%x%x\n",
				cell_id, ordered_prbs[0], prb_x_slot, rx_queue_index, last_sem_idx, frameId, subframeId, slotId,
				buffer_0[0], buffer_0[1], buffer_0[2], buffer_0[3],
				buffer_0[4], buffer_0[5], buffer_0[6], buffer_0[7],
				buffer_0[8], buffer_0[9], buffer_0[10], buffer_0[11]
				);*/
			// printf("Order kernel cell %d exit after %d/%d items, rx item: %d last rx item: %d frame %d subframe %d slot %d\n",
			//     cell_id, ordered_prbs[0], prb_x_slot, rx_queue_index, last_sem_idx, frameId, subframeId, slotId);
		}
	}

	return;
}

__global__ void order_kernel_doca_single(
	/* DOCA objects */
	struct doca_gpu_eth_rxq **doca_rxq,
	struct doca_gpu_semaphore_gpu **sem_gpu,
	const uint16_t* sem_order_num,

	/* Cell */
	const int*		cell_id,
    const int*		ru_type,
    const bool*     cell_health,
	uint32_t		**start_cuphy_d,
	uint32_t		**exit_cond_d,
	uint32_t		**last_sem_idx_rx_h,
	uint32_t		**last_sem_idx_order_h,
	const int*		comp_meth,
    const int*		bit_width,
	const float*		beta,
	const int		prb_size,


	/* Timeout */
	const uint32_t	timeout_no_pkt_ns,
	const uint32_t	timeout_first_pkt_ns,
    const uint32_t  timeout_log_interval_ns,
    const uint8_t   timeout_log_enable,
    const uint32_t  max_rx_pkts,
    const uint32_t  rx_pkts_timeout_ns,
    bool                  commViaCpu,

	/* ORAN */
	const uint8_t		frameId,
	const uint8_t		subframeId,
	const uint8_t		slotId,

	/* Order kernel specific */
	int			*barrier_flag,
	uint8_t		**done_shared,

	/* Timer */
	uint32_t 		**early_rx_packets,
	uint32_t 		**on_time_rx_packets,
	uint32_t 		**late_rx_packets,
	uint32_t 		**next_slot_early_rx_packets,
	uint32_t 		**next_slot_on_time_rx_packets,
	uint32_t 		**next_slot_late_rx_packets,	
	uint64_t*		slot_start,
	uint64_t*		ta4_min_ns,
	uint64_t*		ta4_max_ns,
	uint64_t*		slot_duration,
    uint64_t      **order_kernel_last_timeout_error_time,
    uint8_t                ul_rx_pkt_tracing_level,
	uint64_t**             rx_packets_ts,
	uint32_t**             rx_packets_count,
    uint32_t**             rx_bytes_count,
    uint64_t**             rx_packets_ts_earliest,
    uint64_t**             rx_packets_ts_latest,
	uint64_t**             next_slot_rx_packets_ts,
	uint32_t**             next_slot_rx_packets_count,
    uint32_t**             next_slot_rx_bytes_count,
	uint32_t**             next_slot_num_prb_ch1,
	uint32_t**             next_slot_num_prb_ch2,	

	/* PUSCH */
	uint16_t		**pusch_eAxC_map,
	int*			pusch_eAxC_num,
	uint8_t		**pusch_buffer,
	int*			pusch_prb_x_slot,
	int*			pusch_prb_x_symbol,
	int*			pusch_prb_x_symbol_x_antenna,
	int			pusch_symbols_x_slot,
	uint32_t*			pusch_prb_x_port_x_symbol,
	uint32_t		**pusch_ordered_prbs,

	/* PRACH */
	uint16_t 	**prach_eAxC_map,
	int*		prach_eAxC_num,
	uint8_t		**prach_buffer_0,
	uint8_t		**prach_buffer_1,
	uint8_t 	**prach_buffer_2,
	uint8_t 	**prach_buffer_3,
	uint16_t	prach_section_id_0,
	uint16_t	prach_section_id_1,
	uint16_t	prach_section_id_2,
	uint16_t	prach_section_id_3,
	int*		prach_prb_x_slot,
	int*		prach_prb_x_symbol,
	int*		prach_prb_x_symbol_x_antenna,
	int			prach_symbols_x_slot,
	uint32_t*	prach_prb_x_port_x_symbol,
	uint32_t	**prach_ordered_prbs,
    uint8_t** pcap_buffer,
    uint8_t** pcap_buffer_ts,
    uint32_t** pcap_buffer_index,
    uint8_t pcap_capture_enable,
    uint64_t pcap_capture_cell_bitmask,
    uint16_t max_pkt_size)
{
	int laneId = threadIdx.x % 32;
	int warpId = threadIdx.x / 32;
	int nwarps = blockDim.x / 32;
	int cell_idx = blockIdx.x / 2;

	uint8_t first_packet = 0;
	unsigned long long first_packet_start = 0;
	unsigned long long current_time = 0;
	unsigned long long kernel_start = __globaltimer();
	uint8_t* pkt_offset_ptr, *gbuf_offset_ptr;
	uint8_t* buffer;
	int prb_x_slot=pusch_prb_x_slot[cell_idx]+prach_prb_x_slot[cell_idx];
	doca_error_t ret = (doca_error_t)0;
	// Restart from last semaphore item
	int sem_idx_rx = (int)(*(*(last_sem_idx_rx_h+cell_idx)));
	int sem_idx_order = (int)(*(*(last_sem_idx_order_h+cell_idx)));
	int last_sem_idx_order = (int)(*(*(last_sem_idx_order_h+cell_idx)));
	const uint64_t timeout_ns = rx_pkts_timeout_ns;
	//int  barrier_idx = 1, barrier_signal = gridDim.x;
	//unsigned long long t0, t1, t2, t3, t4, t5, t6, t7;

	__shared__ uint32_t rx_pkt_num;
	__shared__ uint32_t rx_pkt_num_total;
    __shared__ uint32_t pcap_pkt_num_total;

	__shared__ uint64_t rx_buf_idx;
	__shared__ uint32_t done_shared_sh;
	__shared__ uint32_t last_stride_idx;
	__shared__ uint32_t num_prb_ch1_sh;
	__shared__ uint32_t num_prb_ch2_sh;
	__shared__ uint32_t next_slot_num_prb_ch1_sh;
	__shared__ uint32_t next_slot_num_prb_ch2_sh;       
	__shared__ uint32_t exit_rx_cta_sh;
	uint64_t rx_timestamp;
    __shared__ uint32_t early_rx_packets_count_sh;
    __shared__ uint32_t on_time_rx_packets_count_sh;
    __shared__ uint32_t late_rx_packets_count_sh;
    __shared__ uint32_t next_slot_early_rx_packets_count_sh;
    __shared__ uint32_t next_slot_on_time_rx_packets_count_sh;
    __shared__ uint32_t next_slot_late_rx_packets_count_sh;    
    __shared__ uint64_t rx_packets_ts_sh[ORDER_KERNEL_MAX_PKTS_PER_OFDM_SYM*ORAN_MAX_SYMBOLS];
    __shared__ uint32_t rx_packets_count_sh[ORAN_MAX_SYMBOLS];
    __shared__ uint32_t rx_bytes_count_sh[ORAN_MAX_SYMBOLS];
    __shared__ uint64_t rx_packets_ts_earliest_sh[ORAN_MAX_SYMBOLS];
    __shared__ uint64_t rx_packets_ts_latest_sh[ORAN_MAX_SYMBOLS];
    __shared__ uint64_t next_slot_rx_packets_ts_sh[ORDER_KERNEL_MAX_PKTS_PER_OFDM_SYM*ORAN_MAX_SYMBOLS];
    __shared__ uint32_t next_slot_rx_packets_count_sh[ORAN_MAX_SYMBOLS];
    __shared__ uint32_t next_slot_rx_bytes_count_sh[ORAN_MAX_SYMBOLS];
    __shared__ struct doca_gpu_dev_eth_rxq_attr out_attr_sh[512];

	//Cell specific (de-reference from host pinned memory once)
	uint8_t* done_shared_cell=*(done_shared+cell_idx);
	uint32_t* pusch_ordered_prbs_cell=*(pusch_ordered_prbs+cell_idx);
	uint32_t* prach_ordered_prbs_cell=*(prach_ordered_prbs+cell_idx);
	uint32_t* exit_cond_d_cell=*(exit_cond_d+cell_idx);
	uint32_t* last_sem_idx_rx_h_cell=*(last_sem_idx_rx_h+cell_idx);
	uint32_t* last_sem_idx_order_h_cell=*(last_sem_idx_order_h+cell_idx);
	struct doca_gpu_eth_rxq* doca_rxq_cell=*(doca_rxq+cell_idx);
	struct doca_gpu_semaphore_gpu* sem_gpu_cell=*(sem_gpu+cell_idx);
	int pusch_prb_x_slot_cell=pusch_prb_x_slot[cell_idx];
	int prach_prb_x_slot_cell=prach_prb_x_slot[cell_idx];

	uint64_t		slot_start_cell=slot_start[cell_idx];
	uint64_t		ta4_min_ns_cell=ta4_min_ns[cell_idx];
	uint64_t		ta4_max_ns_cell=ta4_max_ns[cell_idx];
	uint64_t		slot_duration_cell=slot_duration[cell_idx];
	uint32_t 		*early_rx_packets_cell= *(early_rx_packets+cell_idx);
	uint32_t 		*on_time_rx_packets_cell= *(on_time_rx_packets+cell_idx);
	uint32_t 		*late_rx_packets_cell= *(late_rx_packets+cell_idx);
	uint32_t 		*next_slot_early_rx_packets_cell= *(next_slot_early_rx_packets+cell_idx);
	uint32_t 		*next_slot_on_time_rx_packets_cell= *(next_slot_on_time_rx_packets+cell_idx);
	uint32_t 		*next_slot_late_rx_packets_cell= *(next_slot_late_rx_packets+cell_idx);    
    uint64_t*       rx_packets_ts_cell=*(rx_packets_ts+cell_idx);
    uint32_t*       rx_packets_count_cell=*(rx_packets_count+cell_idx);
    uint32_t*       rx_bytes_count_cell=*(rx_bytes_count+cell_idx);
    uint64_t*       rx_packets_ts_earliest_cell = *(rx_packets_ts_earliest+cell_idx);
    uint64_t*       rx_packets_ts_latest_cell = *(rx_packets_ts_latest+cell_idx);
    uint64_t*       next_slot_rx_packets_ts_cell=*(next_slot_rx_packets_ts+cell_idx);
    uint32_t*       next_slot_rx_packets_count_cell=*(next_slot_rx_packets_count+cell_idx);
    uint32_t*       next_slot_rx_bytes_count_cell=*(next_slot_rx_bytes_count+cell_idx);
    uint32_t*       next_slot_num_prb_ch1_cell=*(next_slot_num_prb_ch1+cell_idx);
    uint32_t*       next_slot_num_prb_ch2_cell=*(next_slot_num_prb_ch2+cell_idx);     

	uint8_t			*pcap_buffer_cell=*(pcap_buffer+cell_idx);
	uint8_t			*pcap_buffer_ts_cell=*(pcap_buffer_ts+cell_idx);
	uint32_t		*pcap_buffer_index_cell=*(pcap_buffer_index+cell_idx);
	uint8_t			*pusch_buffer_cell=*(pusch_buffer+cell_idx);
	uint16_t		*pusch_eAxC_map_cell=*(pusch_eAxC_map+cell_idx);
	int			pusch_eAxC_num_cell=pusch_eAxC_num[cell_idx];
	uint32_t		pusch_prb_x_port_x_symbol_cell=	pusch_prb_x_port_x_symbol[cell_idx];
	uint8_t			*prach_buffer_0_cell=*(prach_buffer_0+cell_idx);
	uint8_t			*prach_buffer_1_cell=*(prach_buffer_1+cell_idx);
	uint8_t			*prach_buffer_2_cell=*(prach_buffer_2+cell_idx);
	uint8_t			*prach_buffer_3_cell=*(prach_buffer_3+cell_idx);
	uint16_t		*prach_eAxC_map_cell=*(prach_eAxC_map+cell_idx);
	int			prach_eAxC_num_cell=prach_eAxC_num[cell_idx];
	uint32_t		prach_prb_x_port_x_symbol_cell=	prach_prb_x_port_x_symbol[cell_idx];
    const int		ru_type_cell=ru_type[cell_idx];
	const int		comp_meth_cell=comp_meth[cell_idx];
    const int		bit_width_cell=bit_width[cell_idx];
	const float		beta_cell=beta[cell_idx];
	uint32_t		*start_cuphy_d_cell=*(start_cuphy_d+cell_idx);
	const uint16_t sem_order_num_cell=sem_order_num[cell_idx];
    uint64_t* order_kernel_last_timeout_error_time_cell=order_kernel_last_timeout_error_time[cell_idx];
    bool            cell_healthy = cell_health[cell_idx];

    // PRACH start_prbu = 0
	uint16_t startPRB_offset_idx_0 = 0;
	uint16_t startPRB_offset_idx_1 = 0;
	uint16_t startPRB_offset_idx_2 = 0;
	uint16_t startPRB_offset_idx_3 = 0;
    
    uint8_t *pkt_thread = NULL;
    uint8_t frameId_pkt = 0;
    uint8_t symbol_id_pkt = 0;
    uint8_t subframeId_pkt  = 0;
    uint8_t slotId_pkt      = 0;                    
    uint64_t packet_early_thres = 0;
    uint64_t packet_late_thres  = 0;                    
    int rx_packets_ts_idx=0,next_slot_rx_packets_ts_idx=0,max_pkt_idx=ORDER_KERNEL_MAX_PKTS_PER_OFDM_SYM*ORAN_MAX_SYMBOLS;
    int32_t slot_count_input=(2*subframeId+slotId),slot_count_curr;
    uint8_t* section_buf;
    uint16_t ecpri_payload_length;
    // 4 bytes for ecpriPcid, ecprSeqid, ecpriEbit, ecpriSubSeqid
    uint16_t current_length = 4 + sizeof(oran_umsg_iq_hdr);
    uint16_t num_prb = 0;
    uint16_t start_prb = 0;
    uint16_t section_id = 0;
    uint16_t compressed_prb_size = (bit_width_cell == BFP_NO_COMPRESSION) ? PRB_SIZE_16F : (bit_width_cell == BFP_COMPRESSION_14_BITS) ? PRB_SIZE_14F : PRB_SIZE_9F;
    uint16_t num_sections = 0;    

    uint32_t cell_idx_mask = (0x1<<cell_idx);
    uint32_t start_pcap_pkt_offset = 0;
    uint4* s_addr;
    uint4* d_addr;
    int index_copy, index_copy_new;

	// if (threadIdx.x==0) {
	// 	printf("[Single Order kernel start]sem_idx_rx %d addr %lx sem_idx_order %d addr %lx last_sem_idx_order %d F%dS%dS%d first_packet %d timeout_first_pkt_ns %llu current_time %llu first_packet_start %llu\n",
	// 		sem_idx_rx, &(sem_gpu_cell[sem_idx_rx]),
	// 		sem_idx_order, &(sem_gpu_cell[sem_idx_rx]),
	// 		last_sem_idx_order,frameId, subframeId, slotId,first_packet,timeout_first_pkt_ns,current_time,first_packet_start);
	// }

	if((blockIdx.x & 0x1) == 1 && threadIdx.x == 0) {
		DOCA_GPUNETIO_VOLATILE(done_shared_cell[0]) = 1;
		DOCA_GPUNETIO_VOLATILE(pusch_ordered_prbs_cell[0]) = 0;
		DOCA_GPUNETIO_VOLATILE(prach_ordered_prbs_cell[0]) = 0;
        if (!cell_healthy)
        {
            DOCA_GPUNETIO_VOLATILE(*exit_cond_d_cell)=ORDER_KERNEL_EXIT_PRB;
        }
		DOCA_GPUNETIO_VOLATILE(done_shared_sh) = 1;
		DOCA_GPUNETIO_VOLATILE(last_stride_idx) = 0;
	}
    
	if((blockIdx.x & 0x1) == 0 && threadIdx.x == 0) {
		DOCA_GPUNETIO_VOLATILE(rx_pkt_num_total) = 0;
        DOCA_GPUNETIO_VOLATILE(pcap_pkt_num_total) = DOCA_GPUNETIO_VOLATILE(*pcap_buffer_index_cell);
        early_rx_packets_count_sh=DOCA_GPUNETIO_VOLATILE(*next_slot_early_rx_packets_cell);
        on_time_rx_packets_count_sh=DOCA_GPUNETIO_VOLATILE(*next_slot_on_time_rx_packets_cell);
        late_rx_packets_count_sh=DOCA_GPUNETIO_VOLATILE(*next_slot_late_rx_packets_cell);
        next_slot_early_rx_packets_count_sh=0;
        next_slot_late_rx_packets_count_sh=0;
        next_slot_on_time_rx_packets_count_sh=0;
        num_prb_ch1_sh=DOCA_GPUNETIO_VOLATILE(*next_slot_num_prb_ch1_cell);
        num_prb_ch2_sh=DOCA_GPUNETIO_VOLATILE(*next_slot_num_prb_ch2_cell);
        next_slot_num_prb_ch1_sh=0;
        next_slot_num_prb_ch2_sh=0;
        exit_rx_cta_sh=0;
	}

    if(ul_rx_pkt_tracing_level){
        if((blockIdx.x & 0x1) == 0)
        {
            if(threadIdx.x<ORAN_MAX_SYMBOLS)
            {
                rx_packets_count_sh[threadIdx.x]=DOCA_GPUNETIO_VOLATILE(next_slot_rx_packets_count_cell[threadIdx.x]);
                rx_bytes_count_sh[threadIdx.x]=DOCA_GPUNETIO_VOLATILE(next_slot_rx_bytes_count_cell[threadIdx.x]);
                rx_packets_ts_earliest_sh[threadIdx.x]=0xFFFFFFFFFFFFFFFFLLU;
                rx_packets_ts_latest_sh[threadIdx.x]=0;
                next_slot_rx_packets_count_sh[threadIdx.x]=0;
                next_slot_rx_bytes_count_sh[threadIdx.x]=0;
            }
            // COVERITY_DEVIATION: blockIdx.x is uniform across all threads in a block.
            // All threads in this block will reach __syncthreads(), no actual divergence.
            // coverity[CUDA.DIVERGENCE_AT_COLLECTIVE_OPERATION]
            __syncthreads();
            for(uint32_t pkt_idx=threadIdx.x;pkt_idx<max_pkt_idx;pkt_idx+=blockDim.x)
            {
                uint32_t symbol_idx=pkt_idx/ORDER_KERNEL_MAX_PKTS_PER_OFDM_SYM;
                rx_packets_ts_sh[pkt_idx]=DOCA_GPUNETIO_VOLATILE(next_slot_rx_packets_ts_cell[pkt_idx]);
                __threadfence_block();
                if(rx_packets_ts_sh[pkt_idx]!=0)
                    atomicMin((unsigned long long*) &rx_packets_ts_earliest_sh[symbol_idx],(unsigned long long) rx_packets_ts_sh[pkt_idx]);
                atomicMax((unsigned long long*) &rx_packets_ts_latest_sh[symbol_idx],(unsigned long long) rx_packets_ts_sh[pkt_idx]);
                __threadfence_block();
                next_slot_rx_packets_ts_sh[pkt_idx]=0;
            }
            __syncthreads();
            if(threadIdx.x<ORAN_MAX_SYMBOLS)
            {
                if(rx_packets_ts_earliest_sh[threadIdx.x]==0)
                    rx_packets_ts_earliest_sh[threadIdx.x]=0xFFFFFFFFFFFFFFFFLLU;
            }
        }            
    }
	__syncthreads();

	while (DOCA_GPUNETIO_VOLATILE(*exit_cond_d_cell) == ORDER_KERNEL_RUNNING) {
		/* Even receives packets and forward them to Odd Block */
		if ((blockIdx.x & 0x1) == 0) {
			if (threadIdx.x == 0) {
				current_time = __globaltimer();
				if (first_packet && ((current_time - first_packet_start) > timeout_first_pkt_ns)) {
                    if(timeout_log_enable){
                        if((current_time-DOCA_GPUNETIO_VOLATILE(*order_kernel_last_timeout_error_time_cell))>timeout_log_interval_ns){
                            printf("%d Cell %d Order kernel sem_idx_rx %d last_sem_idx_rx %d sem_idx_order %d last_sem_idx_order %d PUSCH PRBs %d/%d PRACH PRBs %d/%d.  First packet received timeout after %d ns F%dS%dS%d done = %d stride = %d,current_time=%llu,last_timeout_log_time=%llu total_rx_pkts=%d\n",__LINE__,
                                cell_idx, sem_idx_rx, *last_sem_idx_rx_h_cell,
                                sem_idx_order,*last_sem_idx_order_h_cell,
                                DOCA_GPUNETIO_VOLATILE(pusch_ordered_prbs_cell[0]), pusch_prb_x_slot_cell,
                                DOCA_GPUNETIO_VOLATILE(prach_ordered_prbs_cell[0]), prach_prb_x_slot_cell,
                                timeout_first_pkt_ns, frameId, subframeId, slotId,
                                DOCA_GPUNETIO_VOLATILE(done_shared_sh), DOCA_GPUNETIO_VOLATILE(last_stride_idx),
                                current_time,DOCA_GPUNETIO_VOLATILE(*order_kernel_last_timeout_error_time_cell),DOCA_GPUNETIO_VOLATILE(rx_pkt_num_total));
                                DOCA_GPUNETIO_VOLATILE(*order_kernel_last_timeout_error_time_cell)=current_time;
                        }
                    }
					DOCA_GPUNETIO_VOLATILE(*exit_cond_d_cell) = ORDER_KERNEL_EXIT_TIMEOUT_RX_PKT;
                    //Latch last_sem_idx_order to sem_idx_rx in case of this timeout
                    // last_sem_idx_order=sem_idx_rx;
				} else if ((!first_packet) && ((current_time - kernel_start) > timeout_no_pkt_ns)) {
                    if(timeout_log_enable){
                        if((current_time-DOCA_GPUNETIO_VOLATILE(*order_kernel_last_timeout_error_time_cell))>timeout_log_interval_ns){
                        printf("%d Cell %d Order kernel sem_idx_rx %d last_sem_idx_rx %d sem_idx_order %d last_sem_idx_order %d PUSCH PRBs %d/%d PRACH PRBs %d/%d. No packet received timeout after %d ns F%dS%dS%d done = %d stride = %d,current_time=%llu,last_timeout_log_time=%llu\n",__LINE__,
                            cell_idx, sem_idx_rx, *last_sem_idx_rx_h_cell,
                            sem_idx_order,*last_sem_idx_order_h_cell,
                            DOCA_GPUNETIO_VOLATILE(pusch_ordered_prbs_cell[0]), pusch_prb_x_slot_cell,
                            DOCA_GPUNETIO_VOLATILE(prach_ordered_prbs_cell[0]), prach_prb_x_slot_cell,
                            timeout_no_pkt_ns, frameId, subframeId, slotId,
                            DOCA_GPUNETIO_VOLATILE(done_shared_sh), DOCA_GPUNETIO_VOLATILE(last_stride_idx),
                            current_time,DOCA_GPUNETIO_VOLATILE(*order_kernel_last_timeout_error_time_cell));
                            DOCA_GPUNETIO_VOLATILE(*order_kernel_last_timeout_error_time_cell)=current_time;
                        }
                    }
 					DOCA_GPUNETIO_VOLATILE(*exit_cond_d_cell) = ORDER_KERNEL_EXIT_TIMEOUT_NO_PKT;
				}
				// printf("Timeout check Done Exit condition (%d)\n",DOCA_GPUNETIO_VOLATILE(*exit_cond_d_cell));
				DOCA_GPUNETIO_VOLATILE(rx_pkt_num) = 0;
			}

            // COVERITY_DEVIATION: blockIdx.x is uniform across all threads in a block.
            // All threads in this block will reach __syncthreads(), no actual divergence.
            // coverity[CUDA.DIVERGENCE_AT_COLLECTIVE_OPERATION]
			__syncthreads();

			if (DOCA_GPUNETIO_VOLATILE(*exit_cond_d_cell) != ORDER_KERNEL_RUNNING)
				break;

            if(commViaCpu)
        {
            ret = doca_gpu_dev_eth_rxq_recv<DOCA_GPUNETIO_ETH_EXEC_SCOPE_BLOCK,DOCA_GPUNETIO_ETH_MCST_DISABLED,DOCA_GPUNETIO_ETH_NIC_HANDLER_AUTO,DOCA_GPUNETIO_ETH_RX_ATTR_NONE>(doca_rxq_cell, max_rx_pkts, timeout_ns, &rx_buf_idx, &rx_pkt_num, out_attr_sh);
            /* If any thread returns receive error, the whole execution stops */
            if (ret != DOCA_SUCCESS) {
                doca_gpu_dev_semaphore_set_status(sem_gpu_cell, sem_idx_rx, DOCA_GPU_SEMAPHORE_STATUS_ERROR);
                DOCA_GPUNETIO_VOLATILE(*exit_cond_d_cell) = ORDER_KERNEL_EXIT_ERROR_LEGACY;
                printf("Exit from rx kernel block %d threadIdx %d ret %d sem_idx_rx %d\n",
                        blockIdx.x, threadIdx.x, ret, sem_idx_rx);
            }
        }
        else
        {
            if (threadIdx.x == 0) {
                ret = doca_gpu_dev_eth_rxq_recv<DOCA_GPUNETIO_ETH_EXEC_SCOPE_THREAD,DOCA_GPUNETIO_ETH_MCST_DISABLED,DOCA_GPUNETIO_ETH_NIC_HANDLER_AUTO,DOCA_GPUNETIO_ETH_RX_ATTR_NONE>(doca_rxq_cell, max_rx_pkts, timeout_ns, &rx_buf_idx, &rx_pkt_num, out_attr_sh);
                /* If any thread returns receive error, the whole execution stops */
                if (ret != DOCA_SUCCESS) {
                    doca_gpu_dev_semaphore_set_status(sem_gpu_cell, sem_idx_rx, DOCA_GPU_SEMAPHORE_STATUS_ERROR);
                    DOCA_GPUNETIO_VOLATILE(*exit_cond_d_cell) = ORDER_KERNEL_EXIT_ERROR_LEGACY;
                    printf("Exit from rx kernel block %d threadIdx %d ret %d sem_idx_rx %d\n",
                            blockIdx.x, threadIdx.x, ret, sem_idx_rx);
                }
            }

                if(pcap_capture_enable && (pcap_capture_cell_bitmask & (cell_idx_mask)) != 0)
                {
                    start_pcap_pkt_offset = pcap_pkt_num_total;
                }
            }
			__threadfence();
		__syncthreads();

		if (DOCA_GPUNETIO_VOLATILE(rx_pkt_num) > 0) {
            if (threadIdx.x == 0){
				doca_gpu_dev_semaphore_set_packet_info(sem_gpu_cell, sem_idx_rx, DOCA_GPU_SEMAPHORE_STATUS_READY, rx_pkt_num, rx_buf_idx);                        
                atomicAdd(&rx_pkt_num_total,rx_pkt_num);
                //printf("Cell Idx %d F%dS%dS%d rx_pkt_num %d\n",cell_idx,frameId, subframeId, slotId,rx_pkt_num);    
            }    
			// Check rx timestamp --  can be done for all packets
                for(uint32_t pkt_idx=threadIdx.x;pkt_idx<rx_pkt_num;pkt_idx+=blockDim.x)
                {
					pkt_thread = (uint8_t*)doca_gpu_dev_eth_rxq_get_pkt_addr(doca_rxq_cell, rx_buf_idx + pkt_idx);
                    frameId_pkt     = oran_umsg_get_frame_id(pkt_thread);
                        subframeId_pkt  = oran_umsg_get_subframe_id(pkt_thread);
                        slotId_pkt      = oran_umsg_get_slot_id(pkt_thread);
    					symbol_id_pkt = oran_umsg_get_symbol_id(pkt_thread);

                        // Bounds check for symbol_id_pkt from untrusted packet data
                        if(__builtin_expect(symbol_id_pkt >= ORAN_ALL_SYMBOLS, 0))
                        {
                            printf("ERROR invalid symbol_id %d (max %d) for Cell %d F%dS%dS%d\n",
                                   symbol_id_pkt,
                                   ORAN_ALL_SYMBOLS - 1,
                                   cell_idx,
                                   frameId_pkt,
                                   subframeId_pkt,
                                   slotId_pkt);
                            continue; // Skip this packet
                        }

                        slot_count_curr = (2*subframeId_pkt+slotId_pkt);
                        section_buf = oran_umsg_get_first_section_buf(pkt_thread);
                        ecpri_payload_length = min(oran_umsg_get_ecpri_payload(pkt_thread),ORAN_ECPRI_MAX_PAYLOAD_LEN);

    					rx_timestamp = out_attr_sh[pkt_idx].timestamp_ns;

                        // Store received packets in buffer
                        if(pcap_capture_enable && (pcap_capture_cell_bitmask & (cell_idx_mask)) != 0)
                        {
                            uint32_t offset = (start_pcap_pkt_offset + pkt_idx) % MAX_PKTS_PER_PCAP_BUFFER;
                            uint8_t* pkt_dst_buf = pcap_buffer_cell+((offset)*(max_pkt_size));
                            uint16_t pkt_size = (ORAN_ETH_HDR_SIZE+sizeof(oran_ecpri_hdr)+ecpri_payload_length);
                            for(index_copy = 0; index_copy < (pkt_size-16); index_copy+=16)
                            {
                                s_addr = reinterpret_cast<uint4*>((uint8_t*)pkt_thread+index_copy);
                                d_addr = reinterpret_cast<uint4*>((uint8_t*)pkt_dst_buf+index_copy);
                                *d_addr = *s_addr;
                            }
                            for(index_copy_new = index_copy; index_copy_new < (pkt_size); index_copy_new ++)
                            {
                                pkt_dst_buf[index_copy_new] = pkt_thread[index_copy_new];
                            }
                            // Append timestamp at the end
                            uint64_t* timestamp_ptr = reinterpret_cast<uint64_t*>((uint8_t*)pcap_buffer_ts_cell + (offset * sizeof(rx_timestamp)));
                            *timestamp_ptr = rx_timestamp;
                            atomicAdd(&pcap_pkt_num_total,1);
                        }

                        if((frameId_pkt!=frameId) || (((slot_count_curr-slot_count_input+20)%20)>1)) //Drop scoring if packets from 2 or greater slots away are received during current slot reception or if the frame IDs mis-match
                        {
                            continue;
                        }

                        if(((slot_count_curr-slot_count_input+20)%20)==1) //TODO: Fix magic number 20 => number of slots in a radio frame for mu=1
                        {
                            packet_early_thres = slot_start_cell+ slot_duration_cell + ta4_min_ns_cell + (slot_duration_cell * symbol_id_pkt / ORAN_MAX_SYMBOLS);
                            packet_late_thres  = slot_start_cell+ slot_duration_cell + ta4_max_ns_cell + (slot_duration_cell * symbol_id_pkt / ORAN_MAX_SYMBOLS);
        					if (rx_timestamp < packet_early_thres)
        						atomicAdd(&next_slot_early_rx_packets_count_sh, 1);
        					else if (rx_timestamp > packet_late_thres)
        						atomicAdd(&next_slot_late_rx_packets_count_sh, 1);
        					else
        						atomicAdd(&next_slot_on_time_rx_packets_count_sh, 1);
                            __threadfence_block();
                            if(ul_rx_pkt_tracing_level)
                            {
                                next_slot_rx_packets_ts_idx = atomicAdd(&next_slot_rx_packets_count_sh[symbol_id_pkt], 1);
                                __threadfence_block();
                                next_slot_rx_packets_ts_idx += ORDER_KERNEL_MAX_PKTS_PER_OFDM_SYM * symbol_id_pkt;
                                next_slot_rx_packets_ts_sh[next_slot_rx_packets_ts_idx] = rx_timestamp;
                            }
                            uint16_t section_buf_size = 0;
                            current_length = 4 + sizeof(oran_umsg_iq_hdr);
                            while(current_length < ecpri_payload_length)
                            {
                                num_prb = oran_umsg_get_num_prb_from_section_buf(section_buf);
                                section_id = oran_umsg_get_section_id_from_section_buf(section_buf);
                                if(num_prb==0)
                                    num_prb=ORAN_MAX_PRB_X_SLOT;
                                if(section_id < prach_section_id_0)
                                {
                                    atomicAdd(&next_slot_num_prb_ch1_sh,num_prb);
                                }
                                else
                                {
                                    atomicAdd(&next_slot_num_prb_ch2_sh,num_prb);
                                }
                                __threadfence_block();
                                section_buf_size = static_cast<uint32_t>(compressed_prb_size) * num_prb + ORAN_IQ_UNCOMPRESSED_SECTION_OVERHEAD;
                                current_length += section_buf_size;
                                section_buf += section_buf_size;
                            }                              
                        }
                        else //Same Frame,sub-frame, slot
                        {
                            packet_early_thres = slot_start_cell + ta4_min_ns_cell + (slot_duration_cell * symbol_id_pkt / ORAN_MAX_SYMBOLS);
                            packet_late_thres  = slot_start_cell + ta4_max_ns_cell + (slot_duration_cell * symbol_id_pkt / ORAN_MAX_SYMBOLS);
        					if (rx_timestamp < packet_early_thres)
        						atomicAdd(&early_rx_packets_count_sh, 1);
        					else if (rx_timestamp > packet_late_thres)
        						atomicAdd(&late_rx_packets_count_sh, 1);
        					else
        						atomicAdd(&on_time_rx_packets_count_sh, 1);
                            __threadfence_block();                        
                            if(ul_rx_pkt_tracing_level){    
                                rx_packets_ts_idx = atomicAdd(&rx_packets_count_sh[symbol_id_pkt], 1);
                                atomicAdd(&rx_bytes_count_sh[symbol_id_pkt], ORAN_ETH_HDR_SIZE + ecpri_payload_length);
                                __threadfence_block();
                                rx_packets_ts_idx+=ORDER_KERNEL_MAX_PKTS_PER_OFDM_SYM*symbol_id_pkt;
                                rx_packets_ts_sh[rx_packets_ts_idx]=rx_timestamp;
                                atomicMin((unsigned long long*) &rx_packets_ts_earliest_sh[symbol_id_pkt],(unsigned long long) rx_timestamp);
                                atomicMax((unsigned long long*) &rx_packets_ts_latest_sh[symbol_id_pkt],(unsigned long long) rx_timestamp);
                                __threadfence_block();
    							// printf("doca_rxq_cell=%p, rx_buf_idx+pkt_idx=%llu, pkt_thread=%p, symbol_id_pkt:%d, rx_packets_ts_idx=%i, rx_packets_count_sh[symbol_id_pkt]=%d, ORDER_KERNEL_MAX_PKTS_PER_OFDM_SYM*symbol_id_pkt=%i, rx_timestamp=%llu\n",
                                //        doca_rxq_cell,     rx_buf_idx+pkt_idx,     pkt_thread,     symbol_id_pkt,    rx_packets_ts_idx,    rx_packets_count_sh[symbol_id_pkt],    ORDER_KERNEL_MAX_PKTS_PER_OFDM_SYM*symbol_id_pkt,    rx_timestamp);
                            	// printf("symbol_id_pkt:%d,rx_packets_count_cell[symbol_id_pkt]=%d,rx_packets_ts_cell[ORDER_KERNEL_MAX_PKTS_PER_OFDM_SYM*symbol_id_pkt+rx_packets_count_cell[symbol_id_pkt]]=%llu\n",
                            	//        symbol_id_pkt,rx_packets_count_cell[symbol_id_pkt],rx_packets_ts_cell[ORDER_KERNEL_MAX_PKTS_PER_OFDM_SYM*symbol_id_pkt+rx_packets_count_cell[symbol_id_pkt]]);
                            }
                            uint16_t section_buf_size = 0;
                            current_length           = 4 + sizeof(oran_umsg_iq_hdr);
                            while(current_length < ecpri_payload_length)
                            {
                                num_prb = oran_umsg_get_num_prb_from_section_buf(section_buf);
                                section_id = oran_umsg_get_section_id_from_section_buf(section_buf);
                                if(num_prb==0)
                                    num_prb=ORAN_MAX_PRB_X_SLOT;
                                if(section_id < prach_section_id_0)
                                {
                                    atomicAdd(&num_prb_ch1_sh,num_prb);
                                }
                                else
                                {
                                    atomicAdd(&num_prb_ch2_sh,num_prb);
                                }
                                __threadfence_block();
                                section_buf_size = static_cast<uint32_t>(compressed_prb_size) * num_prb + ORAN_IQ_UNCOMPRESSED_SECTION_OVERHEAD;
                                current_length += section_buf_size;
                                section_buf += section_buf_size;
                            }
                        }
                    }                    
					// printf("Block idx %d received %d packets on sem %d addr %lx\n", blockIdx.x, rx_pkt_num,sem_idx_rx, &(sem_gpu_cell[sem_idx_rx]));

					if (first_packet == 0) {
						first_packet = 1;
						first_packet_start  = __globaltimer();
					}
				sem_idx_rx = (sem_idx_rx+1) & (sem_order_num_cell - 1);
			}
			else if(sem_idx_rx == sem_idx_order)
				continue;
			__syncthreads();
            if(threadIdx.x==0){
                if((num_prb_ch1_sh+num_prb_ch2_sh)>=prb_x_slot)
                {
                    exit_rx_cta_sh=1;
                }
            }            
            __syncthreads();            
            if(exit_rx_cta_sh)
                break;            
		} else {
			/* Block 1 waits on semaphore for new packets and process them */

			/* Semaphore wait */
			if (threadIdx.x == 0) {
			// printf("Block idx %d waiting on sem %d addr %lx\n", blockIdx.x, sem_idx_order, &(sem_gpu_cell[sem_idx_order]));
			do {
				ret = doca_gpu_dev_semaphore_get_packet_info_status(sem_gpu_cell, sem_idx_order, DOCA_GPU_SEMAPHORE_STATUS_READY, &rx_pkt_num, &rx_buf_idx);
			} while (ret == DOCA_ERROR_NOT_FOUND && DOCA_GPUNETIO_VOLATILE(*exit_cond_d_cell) == ORDER_KERNEL_RUNNING);
				// FIXME: check timeout
			}
			__syncthreads();

			/* Check error or exit condition */
			if (DOCA_GPUNETIO_VOLATILE(*exit_cond_d_cell) != ORDER_KERNEL_RUNNING) {
				/* don't overwrite the error code set by block-0
                if (threadIdx.x == 0) {
					DOCA_GPUNETIO_VOLATILE(*exit_cond_d_cell) = ORDER_KERNEL_EXIT_ERROR_LEGACY;}*/
				break;
			}

			if(DOCA_GPUNETIO_VOLATILE(rx_pkt_num) == 0)
				continue;

		/* Order & decompress packets */
		for (uint32_t pkt_idx = warpId; pkt_idx < rx_pkt_num; pkt_idx += nwarps) {
			pkt_thread = (uint8_t*)doca_gpu_dev_eth_rxq_get_pkt_addr(doca_rxq_cell, rx_buf_idx + pkt_idx);
				frameId_pkt     = oran_umsg_get_frame_id(pkt_thread);
				subframeId_pkt  = oran_umsg_get_subframe_id(pkt_thread);
				slotId_pkt      = oran_umsg_get_slot_id(pkt_thread);
				#if 0
				if (laneId == 0 && warpId == 0)
					printf("pkt_thread %lx: src %x:%x:%x:%x:%x:%x dst %x:%x:%x:%x:%x:%x proto %x:%x vlan %x:%x ecpri %x:%x hdr %x:%x:%x:%x:%x:%x:%x:%x\n",
						// "pkt_idx %d stride_start_idx %d frameId_pkt %d/%d subframeId_pkt %d/%d slotId_pkt %d/%d\n",
						pkt_thread,
						pkt_thread[0], pkt_thread[1], pkt_thread[2], pkt_thread[3], pkt_thread[4], pkt_thread[5],
						pkt_thread[6], pkt_thread[7], pkt_thread[8], pkt_thread[9], pkt_thread[10], pkt_thread[11],
						pkt_thread[12], pkt_thread[13], pkt_thread[14], pkt_thread[15],
						pkt_thread[16], pkt_thread[17],
						pkt_thread[18], pkt_thread[19], pkt_thread[20], pkt_thread[21], pkt_thread[22], pkt_thread[23],
						pkt_idx, stride_start_idx, frameId_pkt, frameId, subframeId_pkt, subframeId, slotId_pkt, slotId);
				#endif

				/* If current frame */
				if (
					((frameId_pkt == frameId) && (subframeId_pkt == subframeId) && (slotId_pkt > slotId)) ||
					((frameId_pkt == frameId) && (subframeId_pkt == ((subframeId+1) % 10)))
				) {
					if (laneId == 0 && DOCA_GPUNETIO_VOLATILE(done_shared_sh) == 1){
						//printf("[DONE Shared 0]F%d/%d SF %d/%d SL %d/%d last_sem_idx_order %d, sem_idx_order %d,sem_idx_rx %d, threadIdx.x %d\n", frameId_pkt, frameId, subframeId_pkt, subframeId, slotId_pkt, slotId,last_sem_idx_order, sem_idx_order, sem_idx_rx,threadIdx.x);
						DOCA_GPUNETIO_VOLATILE(done_shared_sh) = 0;
						DOCA_GPUNETIO_VOLATILE(last_stride_idx) = DOCA_GPUNETIO_VOLATILE(rx_buf_idx);
					}
				} else {
					if (!((frameId_pkt == frameId) && (subframeId_pkt == subframeId) && (slotId_pkt == slotId)))
						 continue;

                    section_buf = oran_umsg_get_first_section_buf(pkt_thread);
					ecpri_payload_length = oran_umsg_get_ecpri_payload(pkt_thread);
					// 4 bytes for ecpriPcid, ecprSeqid, ecpriEbit, ecpriSubSeqid
					current_length = 4 + sizeof(oran_umsg_iq_hdr);
					num_prb = 0;
					start_prb = 0;
					section_id = 0;
					num_sections = 0;
                    uint16_t prb_buffer_size = 0;
                    bool sanity_check = (current_length < ecpri_payload_length);
                    if(ecpri_hdr_sanity_check(pkt_thread) == false)
                    {
                        printf("ERROR malformatted eCPRI header... block %d thread %d\n", blockIdx.x, threadIdx.x);
                        //break;
                    }
					while(current_length < ecpri_payload_length)
					{
						current_time = __globaltimer();
                        if(current_length + ORAN_IQ_UNCOMPRESSED_SECTION_OVERHEAD >= ecpri_payload_length)
                        {
                            sanity_check = false;
                            break;
                        }
                        num_prb = oran_umsg_get_num_prb_from_section_buf(section_buf);
						section_id = oran_umsg_get_section_id_from_section_buf(section_buf);
						start_prb = oran_umsg_get_start_prb_from_section_buf(section_buf);
						if(num_prb==0)
							num_prb=ORAN_MAX_PRB_X_SLOT;
                        prb_buffer_size = compressed_prb_size * num_prb;

                        //WAR added for ru_type::SINGLE_SECT_MODE O-RU to pass. Will remove it when new FW is applied to fix the erronous ecpri payload length
                        if(ru_type_cell != SINGLE_SECT_MODE && current_length + prb_buffer_size + ORAN_IQ_UNCOMPRESSED_SECTION_OVERHEAD > ecpri_payload_length)
                        {
                            sanity_check = false;
                            break;
                        }
                        pkt_offset_ptr = section_buf + ORAN_IQ_UNCOMPRESSED_SECTION_OVERHEAD;
						if(section_id < prach_section_id_0)
						{
							buffer = pusch_buffer_cell;
							gbuf_offset_ptr = buffer + oran_get_offset_from_hdr(pkt_thread, (uint16_t)get_eaxc_index(pusch_eAxC_map_cell, pusch_eAxC_num_cell,
																		 oran_umsg_get_flowid(pkt_thread)),
													    pusch_symbols_x_slot, pusch_prb_x_port_x_symbol_cell, prb_size, start_prb);
						}
						else {
							if(section_id == prach_section_id_0) buffer = prach_buffer_0_cell;
							else if(section_id == prach_section_id_1) buffer = prach_buffer_1_cell;
							else if(section_id == prach_section_id_2) buffer = prach_buffer_2_cell;
							else if(section_id == prach_section_id_3) buffer = prach_buffer_3_cell;
							else {
								// Invalid section_id - skip this section
                                printf("ERROR invalid section_id %d for Cell %d F%dS%dS%d\n", section_id, cell_idx, frameId_pkt, subframeId_pkt, slotId_pkt);
								break;
							}
							gbuf_offset_ptr = buffer + oran_get_offset_from_hdr(pkt_thread, (uint16_t)get_eaxc_index(prach_eAxC_map_cell, prach_eAxC_num_cell,
																		 oran_umsg_get_flowid(pkt_thread)),
													    prach_symbols_x_slot, prach_prb_x_port_x_symbol_cell, prb_size, start_prb);
    
							/* prach_buffer_x_cell is populated based on number of PRACH PDU's, hence the index can be used as "Frequency domain occasion index"
							   and mutiplying with num_prb i.e. NRARB=12 (NumRB's (PRACH SCS=30kHz) for each FDM ocassion) will yeild the corrosponding PRB start for each Frequency domain index
							   Note: WIP for a more generic approach to caluclate and pass the startRB from the cuPHY-CP */
							if(section_id == prach_section_id_0) gbuf_offset_ptr -= startPRB_offset_idx_0;
							else if(section_id == prach_section_id_1) gbuf_offset_ptr -= startPRB_offset_idx_1;
							else if(section_id == prach_section_id_2) gbuf_offset_ptr -= startPRB_offset_idx_2;
							else if(section_id == prach_section_id_3) gbuf_offset_ptr -= startPRB_offset_idx_3;
						}
    
						// CWE-787: drop copy/decompress if untrusted symbolId/startPrb/numPrb
						// would index past the per-eAxC antenna buffer.
						{
							const int chk_sxs = (section_id < prach_section_id_0) ? pusch_symbols_x_slot : prach_symbols_x_slot;
							const int chk_pps = (section_id < prach_section_id_0) ? pusch_prb_x_port_x_symbol_cell : prach_prb_x_port_x_symbol_cell;
							if(!ul_oran_offset_in_bounds(pkt_thread, start_prb, num_prb, chk_sxs, chk_pps))
								num_prb = 0;
						}
                        if(comp_meth_cell == static_cast<uint8_t>(aerial_fh::UserDataCompressionMethod::BLOCK_FLOATING_POINT))
                        {
                            if(bit_width_cell == BFP_NO_COMPRESSION) // BFP with 16 bits is a special case and uses FP16, so copy the values
                            {
                                for(int index_copy = laneId; index_copy < (num_prb * prb_size); index_copy += 32)
                                    gbuf_offset_ptr[index_copy] = pkt_offset_ptr[index_copy];
                            }
                            else
                            {
                                decompress_scale_blockFP<false>((unsigned char*)pkt_offset_ptr, (__half*)gbuf_offset_ptr, beta_cell, num_prb, bit_width_cell, (int)(threadIdx.x & 31), 32);
                            }
                        } else // aerial_fh::UserDataCompressionMethod::NO_COMPRESSION
                        {
                            decompress_scale_fixed<false>(pkt_offset_ptr, (__half*)gbuf_offset_ptr, beta_cell, num_prb, bit_width_cell, (int)(threadIdx.x & 31), 32);
                        }

						// Only first warp thread increases the number of tot PRBs
						if(laneId == 0) {
							int oprb_ch1 = 0;
							int oprb_ch2 = 0;

							if(section_id < prach_section_id_0) {
								oprb_ch1 = atomicAdd(pusch_ordered_prbs_cell, num_prb);
								oprb_ch2 = atomicAdd(prach_ordered_prbs_cell, 0);
							} else {
								oprb_ch1 = atomicAdd(pusch_ordered_prbs_cell, 0);
								oprb_ch2 = atomicAdd(prach_ordered_prbs_cell, num_prb);
							}

							// printf("Lane ID = %d Warp ID = %d oprb_ch1 %d oprb_ch2 %d num_prb %d prb_x_slot %d\n",
							//     laneId, warpId, oprb_ch1, oprb_ch2, num_prb, prb_x_slot);
							if(oprb_ch1 + oprb_ch2 + num_prb >= prb_x_slot)
								DOCA_GPUNETIO_VOLATILE(*exit_cond_d_cell) = ORDER_KERNEL_EXIT_PRB;
						}
						current_length += prb_buffer_size + ORAN_IQ_UNCOMPRESSED_SECTION_OVERHEAD;
						section_buf = pkt_offset_ptr + prb_buffer_size;
						++num_sections;
						if(num_sections > ORAN_MAX_PRB_X_SLOT)
						{
							printf("Invalid U-Plane packet, num_sections %d > 273 for Cell %d F%dS%dS%d\n", num_sections, cell_idx, frameId_pkt, subframeId_pkt, slotId_pkt);
							break;
						}
					}
                    if(!sanity_check)
                    {
                        printf("ERROR uplane pkt sanity check failed, it could be erroneous BFP, numPrb or ecpri payload len, or other reasons... block %d thread %d\n", blockIdx.x, threadIdx.x);
                        DOCA_GPUNETIO_VOLATILE(*exit_cond_d_cell) = ORDER_KERNEL_EXIT_ERROR_LEGACY;
                        break;
                    }
				}
		}
		__syncthreads();

		if(threadIdx.x == 0 && DOCA_GPUNETIO_VOLATILE(done_shared_sh) == 1) {
			doca_gpu_dev_semaphore_set_status(sem_gpu_cell, last_sem_idx_order, DOCA_GPU_SEMAPHORE_STATUS_DONE);
			last_sem_idx_order = (last_sem_idx_order + 1) & (sem_order_num_cell - 1);
		}

		sem_idx_order = (sem_idx_order+1) & (sem_order_num_cell - 1);
		}
	}

	///////////////////////////////////////////////////////////
	// Inter-block barrier
	///////////////////////////////////////////////////////////
	// __threadfence();
	// __syncthreads();
	// if(threadIdx.x == 0)
	// 	ib_barrier(&(barrier_flag[0]), barrier_signal, barrier_idx);
	__syncthreads();
	///////////////////////////////////////////////////////////
	if(ul_rx_pkt_tracing_level){
        if((blockIdx.x & 0x1) == 0)
        {
            for(uint32_t pkt_idx=threadIdx.x;pkt_idx<max_pkt_idx;pkt_idx+=blockDim.x)
            {
                DOCA_GPUNETIO_VOLATILE(rx_packets_ts_cell[pkt_idx])=rx_packets_ts_sh[pkt_idx];
                DOCA_GPUNETIO_VOLATILE(next_slot_rx_packets_ts_cell[pkt_idx])=next_slot_rx_packets_ts_sh[pkt_idx];
            }
            __syncthreads();
            if(threadIdx.x<ORAN_MAX_SYMBOLS)
            {
                DOCA_GPUNETIO_VOLATILE(rx_packets_count_cell[threadIdx.x])=rx_packets_count_sh[threadIdx.x];
                DOCA_GPUNETIO_VOLATILE(rx_bytes_count_cell[threadIdx.x])=rx_bytes_count_sh[threadIdx.x];
                DOCA_GPUNETIO_VOLATILE(next_slot_rx_packets_count_cell[threadIdx.x])=next_slot_rx_packets_count_sh[threadIdx.x];
                DOCA_GPUNETIO_VOLATILE(next_slot_rx_bytes_count_cell[threadIdx.x])=next_slot_rx_bytes_count_sh[threadIdx.x];
                DOCA_GPUNETIO_VOLATILE(rx_packets_ts_earliest_cell[threadIdx.x])=rx_packets_ts_earliest_sh[threadIdx.x];
                DOCA_GPUNETIO_VOLATILE(rx_packets_ts_latest_cell[threadIdx.x])=rx_packets_ts_latest_sh[threadIdx.x];
            }
            __syncthreads();
        }	        
    }
    
	if (threadIdx.x == 0) {
		if((blockIdx.x & 0x1) == 0) {
			DOCA_GPUNETIO_VOLATILE(*last_sem_idx_rx_h_cell) = sem_idx_rx;
            DOCA_GPUNETIO_VOLATILE(*early_rx_packets_cell) = early_rx_packets_count_sh;
            DOCA_GPUNETIO_VOLATILE(*on_time_rx_packets_cell) = on_time_rx_packets_count_sh;
            DOCA_GPUNETIO_VOLATILE(*late_rx_packets_cell) = late_rx_packets_count_sh;
            DOCA_GPUNETIO_VOLATILE(*next_slot_early_rx_packets_cell)=next_slot_early_rx_packets_count_sh;
            DOCA_GPUNETIO_VOLATILE(*next_slot_on_time_rx_packets_cell)=next_slot_on_time_rx_packets_count_sh;
            DOCA_GPUNETIO_VOLATILE(*next_slot_late_rx_packets_cell)=next_slot_late_rx_packets_count_sh;  
            DOCA_GPUNETIO_VOLATILE(*next_slot_num_prb_ch1_cell)=next_slot_num_prb_ch1_sh;
            DOCA_GPUNETIO_VOLATILE(*next_slot_num_prb_ch2_cell)=next_slot_num_prb_ch2_sh;            
            DOCA_GPUNETIO_VOLATILE(*pcap_buffer_index_cell) = pcap_pkt_num_total % MAX_PKTS_PER_PCAP_BUFFER;
            //printf("Exiting OK for F%dS%dS%d : rx_pkt_num_total=%d\n",frameId,subframeId,slotId,rx_pkt_num_total);
		} else {
			DOCA_GPUNETIO_VOLATILE(*last_sem_idx_order_h_cell) = last_sem_idx_order;
			DOCA_GPUNETIO_VOLATILE(*start_cuphy_d_cell) = 1;
		}
	}

	// __syncthreads();

	// if (threadIdx.x == 0) {
	// 	DOCA_GPUNETIO_VOLATILE(*last_sem_idx_rx_h_cell) = sem_idx_rx;
	// 	DOCA_GPUNETIO_VOLATILE(*last_sem_idx_order_h_cell) = last_sem_idx_order;
	// 	DOCA_GPUNETIO_VOLATILE(*start_cuphy_d_cell) = 1;
	// }

	return;
}


__global__ void receive_kernel_for_test_bench(
	/* DOCA objects */
	struct doca_gpu_eth_rxq **doca_rxq,
	struct doca_gpu_semaphore_gpu **sem_gpu,
    const uint16_t* sem_order_num,

	/* Cell */
	const int*		cell_id,
	uint32_t		**exit_cond_d,
	uint32_t		**last_sem_idx_rx_h,
    const int*		bit_width,

	/* Timeout */
	const uint32_t	timeout_no_pkt_ns,
	const uint32_t	timeout_first_pkt_ns,
    const uint32_t  max_rx_pkts,
    const uint32_t  max_pkt_size,

	/* ORAN */
	const uint8_t		frameId,
	const uint8_t		subframeId,
	const uint8_t		slotId,

	uint32_t**             rx_packets_count,
    uint32_t**             next_slot_rx_packets_count,
	uint32_t**             next_slot_num_prb_ch1,
	uint32_t**             next_slot_num_prb_ch2,

    uint8_t**               tb_fh_buf,
    uint8_t**               tb_fh_buf_next_slot,

	int*			pusch_prb_x_slot,
	int*		prach_prb_x_slot,
	uint16_t	prach_section_id_0,
	uint16_t	prach_section_id_1,
	uint16_t	prach_section_id_2,
	uint16_t	prach_section_id_3,
    int*                   srs_prb_x_slot
)
{
	int laneId = threadIdx.x % 32;
	int warpId = threadIdx.x / 32;
	int nwarps = blockDim.x / 32;
	int cell_idx = blockIdx.x;  
    int srs_mode = (srs_prb_x_slot != nullptr) ? 1 : 0;  
    uint32_t* exit_cond_d_cell=*(exit_cond_d+cell_idx);
    struct doca_gpu_eth_rxq* doca_rxq_cell=*(doca_rxq+cell_idx);
    struct doca_gpu_semaphore_gpu* sem_gpu_cell=*(sem_gpu+cell_idx);
    int sem_idx_rx = (int)(*(*(last_sem_idx_rx_h+cell_idx)));
    uint8_t* tb_fh_buf_cell=*(tb_fh_buf+cell_idx);
    uint8_t* tb_fh_buf_next_slot_cell=*(tb_fh_buf_next_slot+cell_idx);
    uint32_t* next_slot_num_prb_ch1_cell = nullptr;
    uint32_t* next_slot_num_prb_ch2_cell = nullptr;
    uint32_t* rx_packets_count_cell = *(rx_packets_count+cell_idx);
    uint32_t* next_slot_rx_packets_count_cell = *(next_slot_rx_packets_count+cell_idx);
    uint32_t* last_sem_idx_rx_h_cell=*(last_sem_idx_rx_h+cell_idx);
    const int		bit_width_cell=bit_width[cell_idx];
    int prb_x_slot=0;
    const uint16_t sem_order_num_cell=sem_order_num[cell_idx];
    doca_error_t ret = (doca_error_t)0;
    uint32_t o_next_slot_rx_pkt_count;
    __shared__ uint32_t next_slot_first_pkt_rcvd;
    __shared__ uint32_t next_slot_first_pkt_idx;
    __shared__ uint32_t next_slot_rx_pkt_count;
	__shared__ uint32_t rx_pkt_num;
    __shared__ uint32_t rx_pkt_num_total;
    __shared__ uint64_t rx_buf_idx;
	__shared__ uint32_t num_prb_ch1_sh;
	__shared__ uint32_t num_prb_ch2_sh;
	__shared__ uint32_t next_slot_num_prb_ch1_sh;
	__shared__ uint32_t next_slot_num_prb_ch2_sh;
    __shared__ uint32_t exit_rx_cta_sh;      
    __shared__ uint32_t pkt_idx_offset_from_prev_slot;
    __shared__ struct doca_gpu_dev_eth_rxq_attr out_attr_sh[512];

    next_slot_num_prb_ch1_cell = *(next_slot_num_prb_ch1+cell_idx);
    if(srs_mode)
    {
        prb_x_slot=srs_prb_x_slot[cell_idx];
    }
    else 
    {
        next_slot_num_prb_ch2_cell = *(next_slot_num_prb_ch2+cell_idx);    
        prb_x_slot=pusch_prb_x_slot[cell_idx]+prach_prb_x_slot[cell_idx];
    }

	uint8_t first_packet = 0;
	unsigned long long first_packet_start = 0;
	unsigned long long current_time = 0;
	unsigned long long kernel_start = __globaltimer();
    const uint64_t timeout_ns = 100000;
    uint8_t *pkt_thread = NULL;
    uint8_t *pkt_dst_buf = NULL;
    uint8_t *pkt_thread_temp = NULL;
    uint8_t *pkt_dst_buf_temp = NULL;    
    uint8_t frameId_pkt=0;
    uint8_t symbol_id_pkt = 0;
    uint8_t subframeId_pkt  = 0;
    uint8_t slotId_pkt      = 0; 
    int32_t slot_count_input=(2*subframeId+slotId),slot_count_curr;
    uint8_t* section_buf;
    uint16_t ecpri_payload_length;
    // 4 bytes for ecpriPcid, ecprSeqid, ecpriEbit, ecpriSubSeqid
    uint16_t current_length = 4 + sizeof(oran_umsg_iq_hdr);
    uint16_t num_prb = 0;
    uint16_t start_prb = 0;
    uint16_t section_id = 0;
    uint16_t compressed_prb_size = (bit_width_cell == BFP_NO_COMPRESSION) ? PRB_SIZE_16F : (bit_width_cell == BFP_COMPRESSION_14_BITS) ? PRB_SIZE_14F : PRB_SIZE_9F;
    uint16_t num_sections = 0;
    uint4* s_addr;
    uint4* d_addr;
    int index_copy,index_copy_new;

    if(threadIdx.x==0)
    {
        next_slot_first_pkt_rcvd=0;
        next_slot_first_pkt_idx=0;
        next_slot_rx_pkt_count=0;
        rx_pkt_num=0;
        rx_pkt_num_total=0;
        num_prb_ch1_sh=DOCA_GPUNETIO_VOLATILE(*next_slot_num_prb_ch1_cell);
        if(!srs_mode)
        {
            num_prb_ch2_sh=DOCA_GPUNETIO_VOLATILE(*next_slot_num_prb_ch2_cell);
        }
        
        pkt_idx_offset_from_prev_slot=DOCA_GPUNETIO_VOLATILE(*next_slot_rx_packets_count_cell);
        DOCA_GPUNETIO_VOLATILE(*rx_packets_count_cell)=DOCA_GPUNETIO_VOLATILE(*next_slot_rx_packets_count_cell);
        next_slot_num_prb_ch1_sh=0;
        next_slot_num_prb_ch2_sh=0;  
        exit_rx_cta_sh=0;
    }
    __syncthreads();
	while (DOCA_GPUNETIO_VOLATILE(*exit_cond_d_cell) == ORDER_KERNEL_RUNNING) {
		/* Even receives packets and forward them to Odd Block */
			if (threadIdx.x == 0) {
				current_time = __globaltimer();
				if (first_packet && ((current_time - first_packet_start) > timeout_first_pkt_ns)) {
                        printf("Cell %d Order kernel sem_idx_rx %d last_sem_idx_rx %d. First packet received timeout after %d ns F%dS%dS%d ,total_rx_pkts=%d\n",
                            cell_idx, sem_idx_rx, *last_sem_idx_rx_h_cell,
                            timeout_first_pkt_ns, frameId, subframeId, slotId,
                            DOCA_GPUNETIO_VOLATILE(rx_pkt_num_total));
                            DOCA_GPUNETIO_VOLATILE(*exit_cond_d_cell) = ORDER_KERNEL_EXIT_TIMEOUT_RX_PKT;
				} else if ((!first_packet) && ((current_time - kernel_start) > timeout_no_pkt_ns)) {
                    printf("Cell %d Order kernel sem_idx_rx %d last_sem_idx_rx %d. No packet received timeout after %d ns F%dS%dS%d\n",
                        cell_idx, sem_idx_rx, *last_sem_idx_rx_h_cell,
                        timeout_no_pkt_ns, frameId, subframeId, slotId);
                        DOCA_GPUNETIO_VOLATILE(*exit_cond_d_cell) = ORDER_KERNEL_EXIT_TIMEOUT_NO_PKT;
				}
				// printf("Timeout check Done Exit condition (%d)\n",DOCA_GPUNETIO_VOLATILE(*exit_cond_d_cell));
				DOCA_GPUNETIO_VOLATILE(rx_pkt_num) = 0;
			}
			__syncthreads();

			if (DOCA_GPUNETIO_VOLATILE(*exit_cond_d_cell) != ORDER_KERNEL_RUNNING)
				break;

			// Add rx timestamp
            ret = doca_gpu_dev_eth_rxq_recv<DOCA_GPUNETIO_ETH_EXEC_SCOPE_BLOCK,DOCA_GPUNETIO_ETH_MCST_DISABLED,DOCA_GPUNETIO_ETH_NIC_HANDLER_AUTO,DOCA_GPUNETIO_ETH_RX_ATTR_NONE>(doca_rxq_cell, max_rx_pkts, timeout_ns, &rx_buf_idx, &rx_pkt_num, out_attr_sh);
            /* If any thread returns receive error, the whole execution stops */
            if (ret != DOCA_SUCCESS) {
                doca_gpu_dev_semaphore_set_status(sem_gpu_cell, sem_idx_rx, DOCA_GPU_SEMAPHORE_STATUS_ERROR);
                DOCA_GPUNETIO_VOLATILE(*exit_cond_d_cell) = ORDER_KERNEL_EXIT_ERROR_LEGACY;
                //printf("Exit from rx kernel block %d threadIdx %d ret %d sem_idx_rx %d\n",
                //	blockIdx.x, threadIdx.x, ret, sem_idx_rx);
            }
            //printf("[receive_kernel_for_test_bench]F%dS%dS%d Received packet count %d pkt_dst_buf for curr slot %p Ordered PRB count %d\n",frameId, subframeId, slotId,rx_pkt_num,(void*)(tb_fh_buf_cell+((pkt_idx_offset_from_prev_slot)*max_pkt_size)),(num_prb_ch1_sh+num_prb_ch2_sh));

		if (DOCA_GPUNETIO_VOLATILE(rx_pkt_num) > 0) {
        if (threadIdx.x == 0){
            doca_gpu_dev_semaphore_set_packet_info(sem_gpu_cell, sem_idx_rx, DOCA_GPU_SEMAPHORE_STATUS_READY, rx_pkt_num, rx_buf_idx);
        }
            // Check rx timestamp --  can be done for all packets
            for(uint32_t pkt_idx=threadIdx.x;pkt_idx<rx_pkt_num;pkt_idx+=blockDim.x)
            {
                pkt_thread = (uint8_t*)doca_gpu_dev_eth_rxq_get_pkt_addr(doca_rxq_cell, rx_buf_idx + pkt_idx);
                frameId_pkt     = oran_umsg_get_frame_id(pkt_thread);
                subframeId_pkt  = oran_umsg_get_subframe_id(pkt_thread);
                slotId_pkt      = oran_umsg_get_slot_id(pkt_thread);                    
                symbol_id_pkt = oran_umsg_get_symbol_id(pkt_thread);
                // Warning for invalid symbol_id (not used for array indexing here, only for packet metadata)
                if(__builtin_expect(symbol_id_pkt >= ORAN_ALL_SYMBOLS, 0))
                {
                    printf("WARNING invalid symbol_id %d (max %d) for Cell %d F%dS%dS%d\n",
                           symbol_id_pkt,
                           ORAN_ALL_SYMBOLS - 1,
                           cell_idx,
                           frameId_pkt,
                           subframeId_pkt,
                           slotId_pkt);
                }
                slot_count_curr = (2*subframeId_pkt+slotId_pkt);
                section_buf = oran_umsg_get_first_section_buf(pkt_thread);
                ecpri_payload_length = min(oran_umsg_get_ecpri_payload(pkt_thread),ORAN_ECPRI_MAX_PAYLOAD_LEN);
                // Network endianness to CPU endianness
                if((frameId_pkt!=frameId) || (((slot_count_curr-slot_count_input+20)%20)>1)) //Drop scoring if packets from 2 or greater slots away are received during current slot reception or if the frame IDs mis-match
                    {
                        continue;
                    }
            
                    if(((slot_count_curr-slot_count_input+20)%20)==1) //TODO: Fix magic number 20 => number of slots in a radio frame for mu=1
                    {
                        if(!next_slot_first_pkt_rcvd)
                        {
                            next_slot_first_pkt_rcvd=1;
                            next_slot_first_pkt_idx=pkt_idx;
                            __threadfence_block();
                        }
                        o_next_slot_rx_pkt_count=atomicAdd(&next_slot_rx_pkt_count,1);
                        uint8_t* pkt_dst_buf = (uint8_t*)(tb_fh_buf_next_slot_cell+((o_next_slot_rx_pkt_count)*max_pkt_size));
                        uint16_t pkt_size = (ORAN_ETH_HDR_SIZE+sizeof(oran_ecpri_hdr)+ecpri_payload_length);
                        //printf("F%dS%dS%d threadIdx.x %d pkt_idx %d pkt_dst_buf for next slot %p\n",frameId_pkt,subframeId_pkt,slotId_pkt,threadIdx.x,(o_next_slot_rx_pkt_count+1),(void*)pkt_dst_buf);
                        for(index_copy = 0; index_copy < (pkt_size-16); index_copy+=16)
                        {
                            s_addr = reinterpret_cast<uint4*>((uint8_t*)pkt_thread+index_copy);
                            d_addr = reinterpret_cast<uint4*>((uint8_t*)pkt_dst_buf+index_copy);
                            *d_addr = *s_addr;
                        }
                        for(int index_copy_new = index_copy; index_copy_new < (pkt_size); index_copy_new ++)
                        {
                            pkt_dst_buf[index_copy_new] = pkt_thread[index_copy_new];
                        }                        
                        uint16_t section_buf_size = 0;
                        current_length = 4 + sizeof(oran_umsg_iq_hdr);
                        while(current_length < ecpri_payload_length)
                        {
                            num_prb = oran_umsg_get_num_prb_from_section_buf(section_buf);                            
                            section_id = oran_umsg_get_section_id_from_section_buf(section_buf);
                            if(num_prb==0)
                                num_prb=ORAN_MAX_PRB_X_SLOT;
                            if(srs_mode)
                            {
                                atomicAdd(&next_slot_num_prb_ch1_sh,num_prb);
                            }
                            else
                            {
                                if(section_id < prach_section_id_0)
                                {
                                    atomicAdd(&next_slot_num_prb_ch1_sh,num_prb);
                                }
                                else
                                {
                                    atomicAdd(&next_slot_num_prb_ch2_sh,num_prb);
                                }                                
                            }
                            __threadfence_block();
                            section_buf_size = static_cast<uint32_t>(compressed_prb_size) * num_prb + ORAN_IQ_UNCOMPRESSED_SECTION_OVERHEAD;
                            current_length += section_buf_size;
                            section_buf += section_buf_size;
                        }                        
                    }
                    else //Same Frame,sub-frame, slot
                    {
                        #if 0
                        if (laneId == 0 && warpId == 0){
                            printf("threadIdx.x %d pkt_thread %lx: src %x:%x:%x:%x:%x:%x dst %x:%x:%x:%x:%x:%x proto %x:%x vlan %x:%x ecpri %x:%x hdr %x:%x:%x:%x:%x:%x:%x:%x pkt_idx %d\n",
                                threadIdx.x,pkt_thread,
                                pkt_thread[0], pkt_thread[1], pkt_thread[2], pkt_thread[3], pkt_thread[4], pkt_thread[5],
                                pkt_thread[6], pkt_thread[7], pkt_thread[8], pkt_thread[9], pkt_thread[10], pkt_thread[11],
                                pkt_thread[12], pkt_thread[13], pkt_thread[14], pkt_thread[15],
                                pkt_thread[16], pkt_thread[17],
                                pkt_thread[18], pkt_thread[19], pkt_thread[20], pkt_thread[21], pkt_thread[22], pkt_thread[23],pkt_thread[24], pkt_thread[25],
                                pkt_idx);
                        }
                        #endif                        
                        uint8_t* pkt_dst_buf = (uint8_t*)(tb_fh_buf_cell+((pkt_idx+rx_pkt_num_total+pkt_idx_offset_from_prev_slot)*max_pkt_size));
                        uint16_t pkt_size = (ORAN_ETH_HDR_SIZE+sizeof(oran_ecpri_hdr)+ecpri_payload_length);
                        for(index_copy = 0; index_copy < (pkt_size-16); index_copy+=16)
                        {
                            s_addr = reinterpret_cast<uint4*>((uint8_t*)pkt_thread+index_copy);
                            d_addr = reinterpret_cast<uint4*>((uint8_t*)pkt_dst_buf+index_copy);
                            *d_addr = *s_addr;
                        }
                        for(int index_copy_new = index_copy; index_copy_new < (pkt_size); index_copy_new ++)
                        {
                            pkt_dst_buf[index_copy_new] = pkt_thread[index_copy_new];
                        }                        
                        uint16_t section_buf_size = 0;
                        current_length = 4 + sizeof(oran_umsg_iq_hdr);
                        while(current_length < ecpri_payload_length)
                        {
                            num_prb = oran_umsg_get_num_prb_from_section_buf(section_buf);
                            section_id = oran_umsg_get_section_id_from_section_buf(section_buf);
    						if(num_prb==0)
    							num_prb=ORAN_MAX_PRB_X_SLOT;
                            if(srs_mode)
                            {
                                atomicAdd(&num_prb_ch1_sh,num_prb);
                            }
                            else
                            {
                                if(section_id < prach_section_id_0)
                                {
                                    atomicAdd(&num_prb_ch1_sh,num_prb);
                                }
                                else
                                {
                                    atomicAdd(&num_prb_ch2_sh,num_prb);
                                }                                
                            }
                            __threadfence_block();
                            section_buf_size = static_cast<uint32_t>(compressed_prb_size) * num_prb + ORAN_IQ_UNCOMPRESSED_SECTION_OVERHEAD;
                            current_length += section_buf_size;
                            section_buf += section_buf_size;
                        }
                    }
                }                    
                // printf("Block idx %d received %d packets on sem %d addr %lx\n", blockIdx.x, rx_pkt_num,sem_idx_rx, &(sem_gpu_cell[sem_idx_rx]));
            
                if (first_packet == 0) {
                    first_packet = 1;
                    first_packet_start  = __globaltimer();
                }
                sem_idx_rx = (sem_idx_rx+1) & (sem_order_num_cell - 1);
			}

			__syncthreads();
			if(threadIdx.x==0){
                if(srs_mode)
                {
                    if(num_prb_ch1_sh>=prb_x_slot)
                    {
                        exit_rx_cta_sh=1;
                        DOCA_GPUNETIO_VOLATILE(*exit_cond_d_cell) = ORDER_KERNEL_EXIT_PRB;
                    }
                }
                else
                {
                    if((num_prb_ch1_sh+num_prb_ch2_sh)>=prb_x_slot)
                    {
                        exit_rx_cta_sh=1;
                        DOCA_GPUNETIO_VOLATILE(*exit_cond_d_cell) = ORDER_KERNEL_EXIT_PRB;
                    }
                }                
                atomicAdd(&rx_pkt_num_total,rx_pkt_num);                        
			}            
			__syncthreads();            
			if(exit_rx_cta_sh)
						break;
		}
        __syncthreads();
        if (threadIdx.x == 0){
            DOCA_GPUNETIO_VOLATILE(*last_sem_idx_rx_h_cell) = sem_idx_rx;
            if(srs_mode)
            {
                DOCA_GPUNETIO_VOLATILE(*next_slot_num_prb_ch1_cell)=next_slot_num_prb_ch1_sh;
            }
            else
            {
                DOCA_GPUNETIO_VOLATILE(*next_slot_num_prb_ch1_cell)=next_slot_num_prb_ch1_sh;
                DOCA_GPUNETIO_VOLATILE(*next_slot_num_prb_ch2_cell)=next_slot_num_prb_ch2_sh;
            }
            DOCA_GPUNETIO_VOLATILE(*next_slot_rx_packets_count_cell)=next_slot_rx_pkt_count;
            DOCA_GPUNETIO_VOLATILE(*rx_packets_count_cell)=DOCA_GPUNETIO_VOLATILE(*rx_packets_count_cell)+(rx_pkt_num_total-next_slot_rx_pkt_count);                  
            printf("rx_pkt_num_total %d next_slot_rx_pkt_count %d\n",rx_pkt_num_total,next_slot_rx_pkt_count);
            printf("[receive_kernel_for_test_bench]F%dS%dS%d Exiting kernel next_slot_rx_pkt_count %d Ordered PRB count %d exit_cond_d_cell %d\n",frameId, subframeId, slotId,next_slot_rx_pkt_count,(num_prb_ch1_sh+num_prb_ch2_sh),DOCA_GPUNETIO_VOLATILE(*exit_cond_d_cell));      
        } 
}

int launch_receive_kernel_for_test_bench(
	CUfunction func,
	cudaStream_t stream,

	/* DOCA objects */
	struct doca_gpu_eth_rxq **doca_rxq,
	struct doca_gpu_semaphore_gpu **sem_gpu,
    const uint16_t* sem_order_num,

	/* Cell */
	const int*		cell_id,
	uint32_t		**exit_cond_d,
	uint32_t		**last_sem_idx_rx_h,
    const int*		bit_width,

	/* Timeout */
	const uint32_t	timeout_no_pkt_ns,
	const uint32_t	timeout_first_pkt_ns,
    const uint32_t  max_rx_pkts,
    const uint32_t  max_pkt_size,

	/* ORAN */
	const uint8_t		frameId,
	const uint8_t		subframeId,
	const uint8_t		slotId,

    uint32_t**             rx_packets_count,
	uint32_t**             next_slot_rx_packets_count,
	uint32_t**             next_slot_num_prb_ch1,
	uint32_t**             next_slot_num_prb_ch2,

    uint8_t**               tb_fh_buf,
    uint8_t**               tb_fh_buf_next_slot,

	int*			pusch_prb_x_slot,
	int*		prach_prb_x_slot,
	uint16_t	prach_section_id_0,
	uint16_t	prach_section_id_1,
	uint16_t	prach_section_id_2,
	uint16_t	prach_section_id_3,
    int*                   srs_prb_x_slot,
	uint8_t num_order_cells
    )
    {
	int cudaBlocks = (num_order_cells);
    int numThreads = 128;

    void* args[] = {
        /* DOCA objects */
        &doca_rxq, &sem_gpu, &sem_order_num,
        /* Cell specific */
        &cell_id, &exit_cond_d, &last_sem_idx_rx_h, &bit_width,
        /* Timeout */
        const_cast<uint32_t*>(&timeout_no_pkt_ns), const_cast<uint32_t*>(&timeout_first_pkt_ns),
        const_cast<uint32_t*>(&max_rx_pkts), const_cast<uint32_t*>(&max_pkt_size),
        /* Time specific */
        const_cast<uint8_t*>(&frameId), const_cast<uint8_t*>(&subframeId), const_cast<uint8_t*>(&slotId),
        /* Order kernel specific */
        &rx_packets_count, &next_slot_rx_packets_count, &next_slot_num_prb_ch1, &next_slot_num_prb_ch2,
        /* FH buffer specific */
        &tb_fh_buf, &tb_fh_buf_next_slot,
        /* PUSCH/PRACH Output buffer specific */
        &pusch_prb_x_slot, &prach_prb_x_slot,
        &prach_section_id_0, &prach_section_id_1, &prach_section_id_2, &prach_section_id_3,
        &srs_prb_x_slot
    };

    CUresult cuResult = cuLaunchKernel(func,
        cudaBlocks, 1, 1,
        numThreads, 1, 1,
        0, stream,
        args, nullptr);
    CUDA_DRIVER_CHECK(cuResult);

    return 0;
}

__device__ __forceinline__ void order_kernel_doca_receive_packets_subSlot(
bool ok_tb_enable,  
uint8_t* tb_fh_buf_cell,
const uint32_t  max_pkt_size,  
uint8_t& first_packet_received,
unsigned long long& first_packet_received_time,
const uint32_t&	timeout_first_pkt_ns,
const uint32_t&	timeout_no_pkt_ns,
const uint8_t&               timeout_log_enable,
uint64_t* order_kernel_last_timeout_error_time_cell,
const uint32_t&  timeout_log_interval_ns,
unsigned long long& kernel_start,
int& cell_idx,
int& sem_idx_rx,
uint32_t* last_sem_idx_rx_h_cell,
int& sem_idx_order,
uint32_t* last_sem_idx_order_h_cell,
uint32_t* pusch_ordered_prbs_cell,
int& pusch_prb_x_slot_cell,
uint32_t* prach_ordered_prbs_cell,
int& prach_prb_x_slot_cell,
const uint8_t&		frameId,
const uint8_t&		subframeId,
const uint8_t&		slotId,
uint32_t& done_shared_sh,
uint32_t last_stride_idx,
uint32_t& rx_pkt_num_total,
uint32_t* 			   sym_ord_done_sig_arr,
uint32_t* exit_cond_d_cell,
uint32_t& rx_pkt_num,
bool&      commViaCpu,
struct doca_gpu_eth_rxq* doca_rxq_cell,
const uint32_t&  max_rx_pkts,
const uint64_t& timeout_ns,
uint64_t& rx_buf_idx,
struct doca_gpu_semaphore_gpu* sem_gpu_cell,
uint64_t&		slot_start_cell,
uint64_t&		ta4_min_ns_cell,
uint64_t&		ta4_max_ns_cell,
uint64_t&		slot_duration_cell,
uint32_t* next_slot_rx_packets_count_sh,
uint32_t* next_slot_rx_bytes_count_sh,
uint32_t* rx_packets_count_sh,
uint32_t* rx_bytes_count_sh,
uint32_t& next_slot_early_rx_packets_count_sh,
uint32_t& next_slot_on_time_rx_packets_count_sh,
uint32_t& next_slot_late_rx_packets_count_sh,
uint32_t& early_rx_packets_count_sh,
uint32_t& on_time_rx_packets_count_sh,
uint32_t& late_rx_packets_count_sh,
uint8_t&                ul_rx_pkt_tracing_level,
uint64_t* next_slot_rx_packets_ts_sh,
uint64_t* rx_packets_ts_sh,
const int&		bit_width_cell,
uint64_t* rx_packets_ts_earliest_sh,
uint64_t* rx_packets_ts_latest_sh,
uint32_t& num_prb_ch1_sh,
uint32_t& num_prb_ch2_sh,
uint32_t& next_slot_num_prb_ch1_sh,
uint32_t& next_slot_num_prb_ch2_sh,
const uint16_t& sem_order_num_cell,
uint16_t	prach_section_id_0,
int& prb_x_slot,
uint32_t& exit_rx_cta_sh,
uint8_t *pcap_buffer_cell,
uint8_t *pcap_buffer_ts_cell,
uint8_t& pcap_capture_enable,
uint64_t& pcap_capture_cell_bitmask,
uint32_t& pcap_pkt_num_total,
struct doca_gpu_dev_eth_rxq_attr *out_attr_sh
)
{
    unsigned long long current_time = 0;
	doca_error_t ret = (doca_error_t)0;
    uint8_t *pkt_thread = NULL;
    uint8_t frameId_pkt=0;
    uint8_t symbol_id_pkt = 0;
    uint8_t subframeId_pkt  = 0;
    uint8_t slotId_pkt      = 0;
    uint8_t* section_buf;
    uint16_t ecpri_payload_length;
	int32_t slot_count_input=(2*subframeId+slotId),slot_count_curr;
	uint64_t rx_timestamp;
    uint64_t packet_early_thres = 0;
    uint64_t packet_late_thres  = 0;                    	
	int rx_packets_ts_idx=0,next_slot_rx_packets_ts_idx=0;
	uint16_t current_length = 4 + sizeof(oran_umsg_iq_hdr);
    uint16_t num_prb = 0;
    uint16_t section_id = 0;
	uint16_t compressed_prb_size = (bit_width_cell == BFP_NO_COMPRESSION) ? PRB_SIZE_16F : (bit_width_cell == BFP_COMPRESSION_14_BITS) ? PRB_SIZE_14F : PRB_SIZE_9F;
    uint32_t cell_idx_mask = (0x1<<cell_idx);
    uint32_t start_pcap_pkt_offset = 0;
    uint4* s_addr;
    uint4* d_addr;
    int index_copy, index_copy_new;
    while (DOCA_GPUNETIO_VOLATILE(*exit_cond_d_cell) == ORDER_KERNEL_RUNNING)
    {
        if (threadIdx.x == 0) {
            current_time = __globaltimer();
            if (first_packet_received && ((current_time - first_packet_received_time) > timeout_first_pkt_ns)) {
                if(timeout_log_enable){
                    if((current_time-DOCA_GPUNETIO_VOLATILE(*order_kernel_last_timeout_error_time_cell))>timeout_log_interval_ns){
                        printf("%d Cell %d Order kernel sem_idx_rx %d last_sem_idx_rx %d sem_idx_order %d last_sem_idx_order %d PUSCH PRBs %d/%d PRACH PRBs %d/%d.  First packet received timeout after %d ns F%dS%dS%d done = %d stride = %d,current_time=%llu,last_timeout_log_time=%llu,total_rx_pkts=%d\n",__LINE__,
                            cell_idx, sem_idx_rx, *last_sem_idx_rx_h_cell,
                            sem_idx_order,*last_sem_idx_order_h_cell,
                            DOCA_GPUNETIO_VOLATILE(pusch_ordered_prbs_cell[0]), pusch_prb_x_slot_cell,
                            DOCA_GPUNETIO_VOLATILE(prach_ordered_prbs_cell[0]), prach_prb_x_slot_cell,
                            timeout_first_pkt_ns, frameId, subframeId, slotId,
                            DOCA_GPUNETIO_VOLATILE(done_shared_sh), DOCA_GPUNETIO_VOLATILE(last_stride_idx),
                            current_time,DOCA_GPUNETIO_VOLATILE(*order_kernel_last_timeout_error_time_cell),DOCA_GPUNETIO_VOLATILE(rx_pkt_num_total));
                            DOCA_GPUNETIO_VOLATILE(*order_kernel_last_timeout_error_time_cell)=current_time;
                    }
                }
                for(uint32_t idx=0;idx<ORAN_PUSCH_SYMBOLS_X_SLOT;idx++)
                {
                    DOCA_GPUNETIO_VOLATILE(sym_ord_done_sig_arr[idx])=(uint32_t)SYM_RX_TIMEOUT;
                }
                                    
                DOCA_GPUNETIO_VOLATILE(*exit_cond_d_cell) = ORDER_KERNEL_EXIT_TIMEOUT_RX_PKT;
                //Latch last_sem_idx_order to sem_idx_rx in case of this timeout
                // last_sem_idx_order=sem_idx_rx;
            } else if ((!first_packet_received) && ((current_time - kernel_start) > timeout_no_pkt_ns)) {
                if(timeout_log_enable){
                    if((current_time-DOCA_GPUNETIO_VOLATILE(*order_kernel_last_timeout_error_time_cell))>timeout_log_interval_ns){
                    printf("%d Cell %d Order kernel sem_idx_rx %d last_sem_idx_rx %d sem_idx_order %d last_sem_idx_order %d PUSCH PRBs %d/%d PRACH PRBs %d/%d. No packet received timeout after %d ns F%dS%dS%d done = %d stride = %d,current_time=%llu,last_timeout_log_time=%llu\n",__LINE__,
                        cell_idx, sem_idx_rx, *last_sem_idx_rx_h_cell,
                        sem_idx_order,*last_sem_idx_order_h_cell,
                        DOCA_GPUNETIO_VOLATILE(pusch_ordered_prbs_cell[0]), pusch_prb_x_slot_cell,
                        DOCA_GPUNETIO_VOLATILE(prach_ordered_prbs_cell[0]), prach_prb_x_slot_cell,
                        timeout_no_pkt_ns, frameId, subframeId, slotId,
                        DOCA_GPUNETIO_VOLATILE(done_shared_sh), DOCA_GPUNETIO_VOLATILE(last_stride_idx),
                        current_time,DOCA_GPUNETIO_VOLATILE(*order_kernel_last_timeout_error_time_cell));
                        DOCA_GPUNETIO_VOLATILE(*order_kernel_last_timeout_error_time_cell)=current_time;
                    }
                }
                for(uint32_t idx=0;idx<ORAN_PUSCH_SYMBOLS_X_SLOT;idx++)
                {
                    DOCA_GPUNETIO_VOLATILE(sym_ord_done_sig_arr[idx])=(uint32_t)SYM_RX_TIMEOUT;
                }
                                    
                DOCA_GPUNETIO_VOLATILE(*exit_cond_d_cell) = ORDER_KERNEL_EXIT_TIMEOUT_NO_PKT;
            }
            // printf("Timeout check Done Exit condition (%d)\n",DOCA_GPUNETIO_VOLATILE(*exit_cond_d_cell));
            DOCA_GPUNETIO_VOLATILE(rx_pkt_num) = 0;
        }
        __syncthreads();

        if (DOCA_GPUNETIO_VOLATILE(*exit_cond_d_cell) != ORDER_KERNEL_RUNNING)
            break;
        // Add rx timestamp
        if(commViaCpu)
        {
            ret = doca_gpu_dev_eth_rxq_recv<DOCA_GPUNETIO_ETH_EXEC_SCOPE_BLOCK,DOCA_GPUNETIO_ETH_MCST_DISABLED,DOCA_GPUNETIO_ETH_NIC_HANDLER_AUTO,DOCA_GPUNETIO_ETH_RX_ATTR_NONE>(doca_rxq_cell, max_rx_pkts, timeout_ns, &rx_buf_idx, &rx_pkt_num, out_attr_sh);
            /* If any thread returns receive error, the whole execution stops */
            if (ret != DOCA_SUCCESS) {
                doca_gpu_dev_semaphore_set_status(sem_gpu_cell, sem_idx_rx, DOCA_GPU_SEMAPHORE_STATUS_ERROR);
                DOCA_GPUNETIO_VOLATILE(*exit_cond_d_cell) = ORDER_KERNEL_EXIT_ERROR1;
                printf("Exit from rx kernel block %d threadIdx %d ret %d sem_idx_rx %d\n",
                    blockIdx.x, threadIdx.x, ret, sem_idx_rx);
            }
    }
        else
        {
        if (threadIdx.x == 0) {
            ret = doca_gpu_dev_eth_rxq_recv<DOCA_GPUNETIO_ETH_EXEC_SCOPE_THREAD,DOCA_GPUNETIO_ETH_MCST_DISABLED,DOCA_GPUNETIO_ETH_NIC_HANDLER_AUTO,DOCA_GPUNETIO_ETH_RX_ATTR_NONE>(doca_rxq_cell, max_rx_pkts, timeout_ns, &rx_buf_idx, &rx_pkt_num, out_attr_sh);
            /* If any thread returns receive error, the whole execution stops */
            if (ret != DOCA_SUCCESS) {
                doca_gpu_dev_semaphore_set_status(sem_gpu_cell, sem_idx_rx, DOCA_GPU_SEMAPHORE_STATUS_ERROR);
                DOCA_GPUNETIO_VOLATILE(*exit_cond_d_cell) = ORDER_KERNEL_EXIT_ERROR1;
                printf("Exit from rx kernel block %d threadIdx %d ret %d sem_idx_rx %d\n",
                    blockIdx.x, threadIdx.x, ret, sem_idx_rx);
            }
        }
            if(pcap_capture_enable && (pcap_capture_cell_bitmask & (cell_idx_mask)) != 0)
            {
                start_pcap_pkt_offset = pcap_pkt_num_total;
            }
        }
        __threadfence();
        __syncthreads();
    

    if (DOCA_GPUNETIO_VOLATILE(rx_pkt_num) > 0) {
        if (threadIdx.x == 0){
            doca_gpu_dev_semaphore_set_packet_info(sem_gpu_cell, sem_idx_rx, DOCA_GPU_SEMAPHORE_STATUS_READY, rx_pkt_num, rx_buf_idx);
            atomicAdd(&rx_pkt_num_total,rx_pkt_num);                        
            // printf("Receive kernel Cell Idx %d F%dS%dS%d rx_pkt_num %d rx_buf_idx %lu sem_idx_rx %d READY\n",cell_idx,frameId, subframeId, slotId, rx_pkt_num, rx_buf_idx, sem_idx_rx);
        }

        // Check rx timestamp --  can be done for all packets
            for(uint32_t pkt_idx=threadIdx.x;pkt_idx<rx_pkt_num;pkt_idx+=blockDim.x)
            {
                if(ok_tb_enable)
                {
                    pkt_thread = (uint8_t*)(tb_fh_buf_cell+((pkt_idx)*max_pkt_size));
            }
            else
            {
                pkt_thread = (uint8_t*)doca_gpu_dev_eth_rxq_get_pkt_addr(doca_rxq_cell, rx_buf_idx + pkt_idx);
            }
            
            frameId_pkt     = oran_umsg_get_frame_id(pkt_thread);
                subframeId_pkt  = oran_umsg_get_subframe_id(pkt_thread);
                slotId_pkt      = oran_umsg_get_slot_id(pkt_thread);                    
                symbol_id_pkt = oran_umsg_get_symbol_id(pkt_thread);

                // Bounds check for symbol_id_pkt from untrusted packet data
                if(__builtin_expect(symbol_id_pkt >= ORAN_ALL_SYMBOLS, 0))
                {
                    printf("ERROR invalid symbol_id %d (max %d) for Cell %d F%dS%dS%d\n",
                           symbol_id_pkt,
                           ORAN_ALL_SYMBOLS - 1,
                           cell_idx,
                           frameId_pkt,
                           subframeId_pkt,
                           slotId_pkt);
                    continue; // Skip this packet
                }

                slot_count_curr = (2*subframeId_pkt+slotId_pkt);
                section_buf = oran_umsg_get_first_section_buf(pkt_thread);
                ecpri_payload_length = min(oran_umsg_get_ecpri_payload(pkt_thread),ORAN_ECPRI_MAX_PAYLOAD_LEN);

                rx_timestamp = out_attr_sh[pkt_idx].timestamp_ns;

                // Store received packets in buffer
                if(!ok_tb_enable && pcap_capture_enable && (pcap_capture_cell_bitmask & (cell_idx_mask)) != 0)
                {
                    uint32_t offset = (start_pcap_pkt_offset + pkt_idx) % MAX_PKTS_PER_PCAP_BUFFER;
                    uint8_t* pkt_dst_buf = pcap_buffer_cell+((offset)*(max_pkt_size));
                    uint16_t pkt_size = (ORAN_ETH_HDR_SIZE+sizeof(oran_ecpri_hdr)+ecpri_payload_length);
                    for(index_copy = 0; index_copy < (pkt_size-16); index_copy+=16)
                    {
                        s_addr = reinterpret_cast<uint4*>((uint8_t*)pkt_thread+index_copy);
                        d_addr = reinterpret_cast<uint4*>((uint8_t*)pkt_dst_buf+index_copy);
                        *d_addr = *s_addr;
                    }
                    for(index_copy_new = index_copy; index_copy_new < (pkt_size); index_copy_new ++)
                    {
                        pkt_dst_buf[index_copy_new] = pkt_thread[index_copy_new];
                    }
                    // Append timestamp at the end
                    uint64_t* timestamp_ptr = reinterpret_cast<uint64_t*>((uint8_t*)pcap_buffer_ts_cell + (offset * sizeof(rx_timestamp)));
                    *timestamp_ptr = rx_timestamp;
                    atomicAdd(&pcap_pkt_num_total,1);
                }

                if((frameId_pkt!=frameId) || (((slot_count_curr-slot_count_input+20)%20)>1)) //Drop scoring if packets from 2 or greater slots away are received during current slot reception or if the frame IDs mis-match
                {
                    continue;
                }

                if(((slot_count_curr-slot_count_input+20)%20)==1) //TODO: Fix magic number 20 => number of slots in a radio frame for mu=1
                {
                    packet_early_thres = slot_start_cell+ slot_duration_cell + ta4_min_ns_cell + (slot_duration_cell * symbol_id_pkt / ORAN_MAX_SYMBOLS);
                    packet_late_thres  = slot_start_cell+ slot_duration_cell + ta4_max_ns_cell + (slot_duration_cell * symbol_id_pkt / ORAN_MAX_SYMBOLS);
    #if 0
        printf("%s:%d ONE frameId_pkt %u subframeId_pkt %u slotId_pkt %u symbol_id_pkt %u "
                                "slot_start_cell %lu slot_duration_cell %lu ta4_min_ns_cell %lu  "
                                "packet_early_thres %lu packet_late_thres %lu rx_timestamp %lu\n",
                                __func__,__LINE__,frameId_pkt, subframeId_pkt, slotId_pkt, symbol_id_pkt,
                                slot_start_cell, slot_duration_cell, ta4_min_ns_cell,
                                packet_early_thres, packet_late_thres, rx_timestamp);
    #endif   
            if(rx_timestamp < packet_early_thres)
                        atomicAdd(&next_slot_early_rx_packets_count_sh, 1);
                    else if(rx_timestamp > packet_late_thres)
                        atomicAdd(&next_slot_late_rx_packets_count_sh, 1);
                    else
                        atomicAdd(&next_slot_on_time_rx_packets_count_sh, 1);
                    __threadfence_block();
                    if(ul_rx_pkt_tracing_level)
                    {
                        next_slot_rx_packets_ts_idx = atomicAdd(&next_slot_rx_packets_count_sh[symbol_id_pkt], 1);
                        atomicAdd(&next_slot_rx_bytes_count_sh[symbol_id_pkt], ORAN_ETH_HDR_SIZE + ecpri_payload_length);
                        __threadfence_block();
                        next_slot_rx_packets_ts_idx += ORDER_KERNEL_MAX_PKTS_PER_OFDM_SYM * symbol_id_pkt;
                        next_slot_rx_packets_ts_sh[next_slot_rx_packets_ts_idx] = rx_timestamp;
                    }
                    uint16_t section_buf_size = 0;
                    current_length = 4 + sizeof(oran_umsg_iq_hdr);
                    while(current_length < ecpri_payload_length)
                    {
                        num_prb = oran_umsg_get_num_prb_from_section_buf(section_buf);                            
                        section_id = oran_umsg_get_section_id_from_section_buf(section_buf);
                        if(num_prb==0)
                            num_prb=ORAN_MAX_PRB_X_SLOT;
                        if(section_id < prach_section_id_0)
                        {
                            atomicAdd(&next_slot_num_prb_ch1_sh,num_prb);
                        }
                        else
                        {
                            atomicAdd(&next_slot_num_prb_ch2_sh,num_prb);
                        }
                        __threadfence_block();
                        section_buf_size = static_cast<uint32_t>(compressed_prb_size) * num_prb + ORAN_IQ_UNCOMPRESSED_SECTION_OVERHEAD;
                        current_length += section_buf_size;
                        section_buf += section_buf_size;
                    }                        
                }
                else //Same Frame,sub-frame, slot
                {
                    packet_early_thres = slot_start_cell + ta4_min_ns_cell + (slot_duration_cell * symbol_id_pkt / ORAN_MAX_SYMBOLS);
                    packet_late_thres  = slot_start_cell + ta4_max_ns_cell + (slot_duration_cell * symbol_id_pkt / ORAN_MAX_SYMBOLS);
    #if 0
            printf("%s:%d TWO frameId_pkt %u subframeId_pkt %u slotId_pkt %u symbol_id_pkt %u "
                                "slot_start_cell %lu slot_duration_cell %lu ta4_min_ns_cell %lu  "
                                "packet_early_thres %lu packet_late_thres %lu rx_timestamp %lu\n",
                                __func__,__LINE__,frameId_pkt, subframeId_pkt, slotId_pkt, symbol_id_pkt,
                                slot_start_cell, slot_duration_cell, ta4_min_ns_cell,
                                packet_early_thres, packet_late_thres, rx_timestamp);
    #endif    
                if (rx_timestamp < packet_early_thres)
                        atomicAdd(&early_rx_packets_count_sh, 1);
                    else if (rx_timestamp > packet_late_thres)
                        atomicAdd(&late_rx_packets_count_sh, 1);
                    else
                        atomicAdd(&on_time_rx_packets_count_sh, 1);
                    __threadfence_block();                              
                    if(ul_rx_pkt_tracing_level){    
                        rx_packets_ts_idx = atomicAdd(&rx_packets_count_sh[symbol_id_pkt], 1);
                        atomicAdd(&rx_bytes_count_sh[symbol_id_pkt], ORAN_ETH_HDR_SIZE + ecpri_payload_length);
                        __threadfence_block();
                        rx_packets_ts_idx+=ORDER_KERNEL_MAX_PKTS_PER_OFDM_SYM*symbol_id_pkt;
                        rx_packets_ts_sh[rx_packets_ts_idx]=rx_timestamp;
                        atomicMin((unsigned long long*) &rx_packets_ts_earliest_sh[symbol_id_pkt],(unsigned long long) rx_timestamp);
                        atomicMax((unsigned long long*) &rx_packets_ts_latest_sh[symbol_id_pkt],(unsigned long long) rx_timestamp);                        
                        __threadfence_block();
                    //printf("symbol_id_pkt:%d,rx_packets_count_cell[symbol_id_pkt]=%d,rx_packets_ts_cell[ORDER_KERNEL_MAX_PKTS_PER_OFDM_SYM*symbol_id_pkt+rx_packets_count_cell[symbol_id_pkt]]=%llu\n",
                    //       symbol_id_pkt,rx_packets_count_cell[symbol_id_pkt],rx_packets_ts_cell[ORDER_KERNEL_MAX_PKTS_PER_OFDM_SYM*symbol_id_pkt+rx_packets_count_cell[symbol_id_pkt]]);
                    }
                    uint16_t section_buf_size = 0;
                    current_length = 4 + sizeof(oran_umsg_iq_hdr);
                    while(current_length < ecpri_payload_length)
                    {
                        num_prb = oran_umsg_get_num_prb_from_section_buf(section_buf);
                        section_id = oran_umsg_get_section_id_from_section_buf(section_buf);
                        if(num_prb==0)
                            num_prb=ORAN_MAX_PRB_X_SLOT;
                        if(section_id < prach_section_id_0)
                        {
                            atomicAdd(&num_prb_ch1_sh,num_prb);
                        }
                        else
                        {
                            atomicAdd(&num_prb_ch2_sh,num_prb);
                        }
                        __threadfence_block();
                        section_buf_size = static_cast<uint32_t>(compressed_prb_size) * num_prb + ORAN_IQ_UNCOMPRESSED_SECTION_OVERHEAD;
                        current_length += section_buf_size;
                        section_buf += section_buf_size;
                    }

                    if (threadIdx.x == 0 && first_packet_received == 0) {
                        first_packet_received = 1;
                        first_packet_received_time  = __globaltimer();
                    }
                }
            }                    
            // printf("Block idx %d received %d packets on sem %d addr %lx\n", blockIdx.x, rx_pkt_num,sem_idx_rx, &(sem_gpu_cell[sem_idx_rx]));

            sem_idx_rx = (sem_idx_rx+1) & (sem_order_num_cell - 1);
        }
        else if(sem_idx_rx == sem_idx_order)
        {
            // printf("EQUAL INDEX Cell %d Order kernel sem_idx_rx %d last_sem_idx_rx %d sem_idx_order %d last_sem_idx_order %d PUSCH PRBs %d/%d PRACH PRBs %d/%d done = %d\n",
            // 	cell_idx, sem_idx_rx, *last_sem_idx_rx_h_cell,
            // 	sem_idx_order,*last_sem_idx_order_h_cell,
            // 	DOCA_GPUNETIO_VOLATILE(pusch_ordered_prbs_cell[0]), pusch_prb_x_slot_cell,
            // 	DOCA_GPUNETIO_VOLATILE(prach_ordered_prbs_cell[0]), prach_prb_x_slot_cell,
            // 	DOCA_GPUNETIO_VOLATILE(done_shared_sh));
            continue;
        }
        __syncthreads();
        if(threadIdx.x==0){
            if((num_prb_ch1_sh+num_prb_ch2_sh)>=prb_x_slot)
            {
                exit_rx_cta_sh=1;
            }
        }            
        __syncthreads();            
        if(exit_rx_cta_sh)
        {
            break;
        }
    }
}

__device__ __forceinline__ void order_kernel_doca_process_receive_packets_subSlot(
    bool ok_tb_enable,
    uint8_t* tb_fh_buf_cell,
    const uint32_t  max_pkt_size,
    uint32_t  rx_pkt_num,
    int warpId,
    int nwarps,
    int laneId,
    struct doca_gpu_eth_rxq* doca_rxq_cell,
    uint64_t rx_buf_idx,
    uint32_t* exit_cond_d_cell,
	const uint8_t		frameId,
	const uint8_t		subframeId,
	const uint8_t		slotId,
    uint32_t* done_shared_sh,
    uint32_t* last_stride_idx,
    const int		bit_width_cell,
    const int		comp_meth_cell,
    const float		beta_cell,
    const int		ru_type_cell,
    uint8_t* pkt_offset_ptr,
    uint8_t* gbuf_offset_ptr,
    const int		prb_size,
    uint32_t cell_idx_mask,
    int prb_x_slot,

    uint16_t		*pusch_eAxC_map_cell,
    int			pusch_eAxC_num_cell,
    int			pusch_symbols_x_slot,
    uint32_t		pusch_prb_x_port_x_symbol_cell,
    uint8_t			*pusch_buffer_cell,
    uint32_t* pusch_ordered_prbs_cell,

    uint32_t* pusch_prb_symbol_map_cell,
    uint32_t* pusch_prb_symbol_ordered,
    uint32_t* pusch_prb_symbol_ordered_done,
    uint32_t* 			   sym_ord_done_sig_arr,
    uint32_t*              sym_ord_done_mask_arr,    
    uint32_t* 			   num_order_cells_sym_mask_arr,
    uint8_t                pusch_prb_non_zero,

    uint16_t		*prach_eAxC_map_cell,
    int			prach_eAxC_num_cell,
    int			prach_symbols_x_slot,
    uint32_t		prach_prb_x_port_x_symbol_cell,
	uint16_t	prach_section_id_0,
	uint16_t	prach_section_id_1,
	uint16_t	prach_section_id_2,
	uint16_t	prach_section_id_3,
    uint8_t			*prach_buffer_0_cell,    
    uint8_t			*prach_buffer_1_cell,
    uint8_t			*prach_buffer_2_cell,
    uint8_t			*prach_buffer_3_cell,
    uint32_t* prach_ordered_prbs_cell
)
{
    uint8_t *pkt_thread = NULL;
    uint8_t frameId_pkt=0;
    uint8_t symbol_id_pkt = 0;
    uint8_t subframeId_pkt  = 0;
    uint8_t slotId_pkt      = 0;
    uint8_t* section_buf;
    uint16_t ecpri_payload_length;
    // 4 bytes for ecpriPcid, ecprSeqid, ecpriEbit, ecpriSubSeqid
    uint16_t current_length;
    uint16_t num_prb = 0;
    uint16_t start_prb = 0;
    uint16_t section_id = 0;
    uint16_t compressed_prb_size = (bit_width_cell == BFP_NO_COMPRESSION) ? PRB_SIZE_16F : (bit_width_cell == BFP_COMPRESSION_14_BITS) ? PRB_SIZE_14F : PRB_SIZE_9F;
    uint16_t num_sections = 0;  
    uint8_t* buffer;
    uint32_t tot_pusch_prb_symbol_ordered=0;  
    // PRACH start_prbu = 0
	uint16_t startPRB_offset_idx_0 = 0;
	uint16_t startPRB_offset_idx_1 = 0;
	uint16_t startPRB_offset_idx_2 = 0;
	uint16_t startPRB_offset_idx_3 = 0;

    doca_error_t ret = (doca_error_t)0;
    unsigned long long current_time = 0;
    /* Order & decompress packets */
    for (uint32_t pkt_idx = warpId; pkt_idx < rx_pkt_num; pkt_idx += nwarps) {
        if(ok_tb_enable)
        {
            pkt_thread = (uint8_t*)(tb_fh_buf_cell+((pkt_idx)*max_pkt_size));
    }
    else
    {
        pkt_thread = (uint8_t*)doca_gpu_dev_eth_rxq_get_pkt_addr(doca_rxq_cell, rx_buf_idx + pkt_idx);
    }
    frameId_pkt     = oran_umsg_get_frame_id(pkt_thread);
        subframeId_pkt  = oran_umsg_get_subframe_id(pkt_thread);
        slotId_pkt      = oran_umsg_get_slot_id(pkt_thread);
        symbol_id_pkt = oran_umsg_get_symbol_id(pkt_thread);

        // Bounds check for symbol_id_pkt from untrusted packet data
        if(__builtin_expect(symbol_id_pkt >= ORAN_ALL_SYMBOLS, 0))
        {
            printf("ERROR invalid symbol_id %d (max %d) for Cell %d F%dS%dS%d\n",
                   symbol_id_pkt,
                   ORAN_ALL_SYMBOLS - 1,
                   (blockIdx.x / 2),
                   frameId_pkt,
                   subframeId_pkt,
                   slotId_pkt);
            continue; // Skip this packet
        }

#if 0
        if (laneId == 0 && warpId == 0)
            printf("pkt_thread %lx: src %x:%x:%x:%x:%x:%x dst %x:%x:%x:%x:%x:%x proto %x:%x vlan %x:%x ecpri %x:%x hdr %x:%x:%x:%x:%x:%x:%x:%x\n",
                // "pkt_idx %d stride_start_idx %d section_id_pkt %d/%d frameId_pkt %d/%d subframeId_pkt %d/%d slotId_pkt %d/%d\n",
                pkt_thread,
                pkt_thread[0], pkt_thread[1], pkt_thread[2], pkt_thread[3], pkt_thread[4], pkt_thread[5],
                pkt_thread[6], pkt_thread[7], pkt_thread[8], pkt_thread[9], pkt_thread[10], pkt_thread[11],
                pkt_thread[12], pkt_thread[13], pkt_thread[14], pkt_thread[15],
                pkt_thread[16], pkt_thread[17],
                pkt_thread[18], pkt_thread[19], pkt_thread[20], pkt_thread[21], pkt_thread[22], pkt_thread[23],
                pkt_idx, stride_start_idx, section_id_pkt, prach_section_id_0, frameId_pkt, frameId, subframeId_pkt, subframeId, slotId_pkt, slotId);
        #endif

        /* If current frame */
        if (
            ((frameId_pkt == frameId) && (subframeId_pkt == subframeId) && (slotId_pkt > slotId)) ||
            ((frameId_pkt == frameId) && (subframeId_pkt == ((subframeId+1) % 10)))
        ) {
            if (laneId == 0 && DOCA_GPUNETIO_VOLATILE(*done_shared_sh) == 1){
                //printf("[DONE Shared 0]F%d/%d SF %d/%d SL %d/%d last_sem_idx_order %d, sem_idx_order %d,sem_idx_rx %d, threadIdx.x %d\n", frameId_pkt, frameId, subframeId_pkt, subframeId, slotId_pkt, slotId,last_sem_idx_order, sem_idx_order, sem_idx_rx,threadIdx.x);
                DOCA_GPUNETIO_VOLATILE(*done_shared_sh) = 0;
                DOCA_GPUNETIO_VOLATILE(*last_stride_idx) = DOCA_GPUNETIO_VOLATILE(rx_buf_idx);
            }
        } else {
            if (!((frameId_pkt == frameId) && (subframeId_pkt == subframeId) && (slotId_pkt == slotId)))
            {
                continue;
            }
            /* if this is the right slot, order & decompress */
            section_buf = oran_umsg_get_first_section_buf(pkt_thread);
            ecpri_payload_length = oran_umsg_get_ecpri_payload(pkt_thread);
            // 4 bytes for ecpriPcid, ecprSeqid, ecpriEbit, ecpriSubSeqid
            current_length = 4 + sizeof(oran_umsg_iq_hdr);
            num_prb = 0;
            start_prb = 0;
            section_id = 0;
            num_sections = 0;
            uint16_t prb_buffer_size = 0;
            bool sanity_check = (current_length < ecpri_payload_length);
            if(ecpri_hdr_sanity_check(pkt_thread) == false)
            {
                printf("ERROR malformatted eCPRI header... block %d thread %d\n", blockIdx.x, threadIdx.x);
                //break;
            }
            while(current_length < ecpri_payload_length)
            {
                current_time = __globaltimer();
                if(current_length + ORAN_IQ_UNCOMPRESSED_SECTION_OVERHEAD >= ecpri_payload_length)
                {
                    sanity_check = false;
                    break;
                }

                num_prb = oran_umsg_get_num_prb_from_section_buf(section_buf);
                section_id = oran_umsg_get_section_id_from_section_buf(section_buf);
                start_prb = oran_umsg_get_start_prb_from_section_buf(section_buf);
                if(num_prb==0)
                    num_prb=ORAN_MAX_PRB_X_SLOT;
                prb_buffer_size = compressed_prb_size * num_prb;

                //WAR added for ru_type::SINGLE_SECT_MODE O-RU to pass. Will remove it when new FW is applied to fix the erronous ecpri payload length
                if(ru_type_cell != SINGLE_SECT_MODE && current_length + prb_buffer_size + ORAN_IQ_UNCOMPRESSED_SECTION_OVERHEAD > ecpri_payload_length)
                {
                    sanity_check = false;
                    break;
                }
                pkt_offset_ptr = section_buf + ORAN_IQ_UNCOMPRESSED_SECTION_OVERHEAD;
                if(section_id < prach_section_id_0)
                {
                    buffer = pusch_buffer_cell;
                    gbuf_offset_ptr = buffer + oran_get_offset_from_hdr(pkt_thread, (uint16_t)get_eaxc_index(pusch_eAxC_map_cell, pusch_eAxC_num_cell,
                                                                    oran_umsg_get_flowid(pkt_thread)),
                                                pusch_symbols_x_slot, pusch_prb_x_port_x_symbol_cell, prb_size, start_prb);
                }
                else {
                    if(section_id == prach_section_id_0) buffer = prach_buffer_0_cell;
                    else if(section_id == prach_section_id_1) buffer = prach_buffer_1_cell;
                    else if(section_id == prach_section_id_2) buffer = prach_buffer_2_cell;
                    else if(section_id == prach_section_id_3) buffer = prach_buffer_3_cell;
                    else {
                        // Invalid section_id - skip this section
                        printf("ERROR invalid section_id %d for Cell %d F%dS%dS%d\n", section_id, (blockIdx.x / 2), frameId_pkt, subframeId_pkt, slotId_pkt);
                        break;
                    }
                    gbuf_offset_ptr = buffer + oran_get_offset_from_hdr(pkt_thread, (uint16_t)get_eaxc_index(prach_eAxC_map_cell, prach_eAxC_num_cell,
                                                                    oran_umsg_get_flowid(pkt_thread)),
                                                prach_symbols_x_slot, prach_prb_x_port_x_symbol_cell, prb_size, start_prb);

                    /* prach_buffer_x_cell is populated based on number of PRACH PDU's, hence the index can be used as "Frequency domain occasion index"
                        and mutiplying with num_prb i.e. NRARB=12 (NumRB's (PRACH SCS=30kHz) for each FDM ocassion) will yeild the corrosponding PRB start for each Frequency domain index
                        Note: WIP for a more generic approach to caluclate and pass the startRB from the cuPHY-CP */
                    if(section_id == prach_section_id_0) gbuf_offset_ptr -= startPRB_offset_idx_0;
                    else if(section_id == prach_section_id_1) gbuf_offset_ptr -= startPRB_offset_idx_1;
                    else if(section_id == prach_section_id_2) gbuf_offset_ptr -= startPRB_offset_idx_2;
                    else if(section_id == prach_section_id_3) gbuf_offset_ptr -= startPRB_offset_idx_3;
                }

                // CWE-787: drop copy/decompress if untrusted symbolId/startPrb/numPrb
                // would index past the per-eAxC antenna buffer.
                {
                    const int chk_sxs = (section_id < prach_section_id_0) ? pusch_symbols_x_slot : prach_symbols_x_slot;
                    const int chk_pps = (section_id < prach_section_id_0) ? pusch_prb_x_port_x_symbol_cell : prach_prb_x_port_x_symbol_cell;
                    if(!ul_oran_offset_in_bounds(pkt_thread, start_prb, num_prb, chk_sxs, chk_pps))
                        num_prb = 0;
                }
                if(comp_meth_cell == static_cast<uint8_t>(aerial_fh::UserDataCompressionMethod::BLOCK_FLOATING_POINT))
                {
                    if(bit_width_cell == BFP_NO_COMPRESSION) // BFP with 16 bits is a special case and uses FP16, so copy the values
                    {
                        for(int index_copy = laneId; index_copy < (num_prb * prb_size); index_copy += 32)
                            gbuf_offset_ptr[index_copy] = pkt_offset_ptr[index_copy];
                    }
                    else
                    {
                        decompress_scale_blockFP<false>((unsigned char*)pkt_offset_ptr, (__half*)gbuf_offset_ptr, beta_cell, num_prb, bit_width_cell, (int)(threadIdx.x & 31), 32);
                    }
                } else // aerial_fh::UserDataCompressionMethod::NO_COMPRESSION
                {
                    decompress_scale_fixed<false>(pkt_offset_ptr, (__half*)gbuf_offset_ptr, beta_cell, num_prb, bit_width_cell, (int)(threadIdx.x & 31), 32);
                }
                // Only first warp thread increases the number of tot PRBs
                if(laneId == 0) {
                    int oprb_ch1 = 0;
                    int oprb_ch2 = 0;

                    if(section_id < prach_section_id_0) {
                        tot_pusch_prb_symbol_ordered = atomicAdd(&pusch_prb_symbol_ordered[symbol_id_pkt],num_prb);
                        tot_pusch_prb_symbol_ordered += num_prb;							
                        if(pusch_prb_non_zero && tot_pusch_prb_symbol_ordered >= pusch_prb_symbol_map_cell[symbol_id_pkt] && DOCA_GPUNETIO_VOLATILE(pusch_prb_symbol_ordered_done[symbol_id_pkt])==0){
                            DOCA_GPUNETIO_VOLATILE(pusch_prb_symbol_ordered_done[symbol_id_pkt])=1;
                            atomicOr(&sym_ord_done_mask_arr[symbol_id_pkt],cell_idx_mask);
                            //printf("Lane ID = %d Warp ID = %d symbol_id_pkt = %d pusch_prb_symbol_ordered_done[symbol_id_pkt] = %d sym_ord_done_mask_arr[symbol_id_pkt]=%d\n",laneId,warpId,symbol_id_pkt,DOCA_GPUNETIO_VOLATILE(pusch_prb_symbol_ordered_done[symbol_id_pkt]),DOCA_GPUNETIO_VOLATILE(sym_ord_done_mask_arr[symbol_id_pkt]));                                                                
                            if(DOCA_GPUNETIO_VOLATILE(sym_ord_done_mask_arr[symbol_id_pkt])==num_order_cells_sym_mask_arr[symbol_id_pkt] && DOCA_GPUNETIO_VOLATILE(sym_ord_done_sig_arr[symbol_id_pkt])==(uint32_t)SYM_RX_NOT_DONE){
                                DOCA_GPUNETIO_VOLATILE(sym_ord_done_sig_arr[symbol_id_pkt])=(uint32_t)SYM_RX_DONE;
                            }
                        }                                                       
                        oprb_ch1 = atomicAdd(pusch_ordered_prbs_cell, num_prb);
                        oprb_ch2 = atomicAdd(prach_ordered_prbs_cell, 0);
                    } else {
                        oprb_ch1 = atomicAdd(pusch_ordered_prbs_cell, 0);
                        oprb_ch2 = atomicAdd(prach_ordered_prbs_cell, num_prb);
                    }

                    // printf("Lane ID = %d Warp ID = %d oprb_ch1 %d oprb_ch2 %d num_prb %d prb_x_slot %d\n",
                    //     laneId, warpId, oprb_ch1, oprb_ch2, num_prb, prb_x_slot);
                    if(oprb_ch1 + oprb_ch2 + num_prb >= prb_x_slot)
                        DOCA_GPUNETIO_VOLATILE(*exit_cond_d_cell) = ORDER_KERNEL_EXIT_PRB;
                }
                current_length += prb_buffer_size + ORAN_IQ_UNCOMPRESSED_SECTION_OVERHEAD;
                section_buf = pkt_offset_ptr + prb_buffer_size;
                ++num_sections;
                if(num_sections > ORAN_MAX_PRB_X_SLOT)
                {
                    printf("Invalid U-Plane packet, num_sections %d > 273 for Cell %d F%dS%dS%d\n", num_sections, (blockIdx.x / 2), frameId_pkt, subframeId_pkt, slotId_pkt);
                    break;
                }
            }
            if(!sanity_check)
            {
                printf("ERROR uplane pkt sanity check failed, it could be erroneous BFP, numPrb or ecpri payload len, or other reasons... block %d thread %d\n", blockIdx.x, threadIdx.x);
                DOCA_GPUNETIO_VOLATILE(*exit_cond_d_cell) = ORDER_KERNEL_EXIT_ERROR7;
                break;
            }
        }
    }
}

__global__ void receive_process_kernel_for_test_bench(
	/* Cell */
	const int*		cell_id,
	uint32_t		**exit_cond_d,
    const uint16_t* sem_order_num,
    const int*		ru_type,

	/* ORAN */
	const uint8_t		frameId,
	const uint8_t		subframeId,
	const uint8_t		slotId,

    const int		prb_size,
    const int*		comp_meth,
    const int*		bit_width,
    const float*		beta,
    uint32_t		**last_sem_idx_order_h,
    
    uint32_t*       rx_pkt_num_slot,
    uint8_t**       tb_fh_buf,
    const uint32_t  max_pkt_size,


    /* Sub-slot processing*/
    uint32_t* 			   sym_ord_done_sig_arr,
    uint32_t*              sym_ord_done_mask_arr,
    uint32_t*              pusch_prb_symbol_map,
	uint32_t* 			   num_order_cells_sym_mask_arr,    
    
    /*PUSCH*/
    uint8_t**       pusch_buffer,
	uint16_t		**pusch_eAxC_map,
	int*			pusch_eAxC_num,    
    int			pusch_symbols_x_slot,
    uint32_t*			pusch_prb_x_port_x_symbol,
    uint32_t		**pusch_ordered_prbs,
    int*			pusch_prb_x_slot,

    /*PRACH*/
	uint16_t 	**prach_eAxC_map,
	int*		prach_eAxC_num,
	uint8_t		**prach_buffer_0,
	uint8_t		**prach_buffer_1,
	uint8_t 	**prach_buffer_2,
	uint8_t 	**prach_buffer_3,
    int*	    prach_prb_x_slot,
    int			prach_symbols_x_slot,
    uint32_t*   prach_prb_x_port_x_symbol,
    uint32_t	**prach_ordered_prbs,
	uint16_t	prach_section_id_0,
	uint16_t	prach_section_id_1,
	uint16_t	prach_section_id_2,
	uint16_t	prach_section_id_3,

    /*Receive CTA params*/
    const uint32_t	timeout_no_pkt_ns,
    const uint32_t	timeout_first_pkt_ns,
	const uint32_t  timeout_log_interval_ns,
	const uint8_t   timeout_log_enable,
    uint64_t      **order_kernel_last_timeout_error_time,
    uint32_t		**last_sem_idx_rx_h,
    bool            commViaCpu,
    struct doca_gpu_eth_rxq **doca_rxq,
    const uint32_t  max_rx_pkts,
    const uint32_t  rx_pkts_timeout_ns,
    struct doca_gpu_semaphore_gpu **sem_gpu,
	uint64_t*		slot_start,
	uint64_t*		ta4_min_ns,
	uint64_t*		ta4_max_ns,
	uint64_t*		slot_duration,
    uint8_t                ul_rx_pkt_tracing_level
)
{
    if(cell_id == nullptr)
    {
        return;
    }
    int laneId = threadIdx.x % 32;
	int warpId = threadIdx.x / 32;
	int nwarps = blockDim.x / 32;
	int cell_idx = blockIdx.x/2;
    unsigned long long current_time = 0;
    int last_sem_idx_order = (int)(*(*(last_sem_idx_order_h+cell_idx)));
    int sem_idx_order = (int)(*(*(last_sem_idx_order_h+cell_idx)));

    uint32_t rx_pkt_num_slot_cell = *(rx_pkt_num_slot+cell_idx);
    const int		comp_meth_cell=comp_meth[cell_idx];
    const float		beta_cell=beta[cell_idx];
    uint8_t *pkt_thread = NULL; 
    uint8_t* tb_fh_buf_cell=*(tb_fh_buf+cell_idx);
    const int		bit_width_cell=bit_width[cell_idx];
    uint32_t* pusch_prb_symbol_map_cell = pusch_prb_symbol_map+(ORAN_PUSCH_SYMBOLS_X_SLOT*cell_idx);
    uint8_t* section_buf;
    uint16_t ecpri_payload_length;
    // 4 bytes for ecpriPcid, ecprSeqid, ecpriEbit, ecpriSubSeqid
    uint16_t current_length = 4 + sizeof(oran_umsg_iq_hdr);
    uint16_t num_prb = 0;
    uint16_t start_prb = 0;
    uint16_t section_id = 0;
    uint16_t compressed_prb_size = (bit_width_cell == BFP_NO_COMPRESSION) ? PRB_SIZE_16F : (bit_width_cell == BFP_COMPRESSION_14_BITS) ? PRB_SIZE_14F : PRB_SIZE_9F;
    uint16_t num_sections = 0;
    uint8_t* pkt_offset_ptr, *gbuf_offset_ptr;
    uint8_t	*pusch_buffer_cell=*(pusch_buffer+cell_idx);
    uint32_t pusch_prb_x_port_x_symbol_cell=pusch_prb_x_port_x_symbol[cell_idx];
    uint32_t		prach_prb_x_port_x_symbol_cell=	prach_prb_x_port_x_symbol[cell_idx];
	uint32_t* pusch_ordered_prbs_cell=*(pusch_ordered_prbs+cell_idx);
	uint32_t* prach_ordered_prbs_cell=*(prach_ordered_prbs+cell_idx);
    uint16_t		*prach_eAxC_map_cell=*(prach_eAxC_map+cell_idx);
	uint8_t			*prach_buffer_0_cell=*(prach_buffer_0+cell_idx);
	uint8_t			*prach_buffer_1_cell=*(prach_buffer_1+cell_idx);
	uint8_t			*prach_buffer_2_cell=*(prach_buffer_2+cell_idx);
	uint8_t			*prach_buffer_3_cell=*(prach_buffer_3+cell_idx);
    int			prach_eAxC_num_cell=prach_eAxC_num[cell_idx];
    uint32_t* exit_cond_d_cell=*(exit_cond_d+cell_idx);    
    const int		ru_type_cell=ru_type[cell_idx];
    uint16_t		*pusch_eAxC_map_cell=*(pusch_eAxC_map+cell_idx);
    int			pusch_eAxC_num_cell=pusch_eAxC_num[cell_idx];
    const uint16_t sem_order_num_cell=sem_order_num[cell_idx];
    uint32_t tot_pusch_prb_symbol_ordered=0;
    uint8_t frameId_pkt=0;
    uint8_t symbol_id_pkt = 0;
    uint8_t subframeId_pkt  = 0;
    uint8_t slotId_pkt      = 0;  
    uint8_t* buffer;
    doca_error_t ret = (doca_error_t)0;

    uint8_t first_packet_received = 0;
    unsigned long long first_packet_received_time = 0;
    uint64_t* order_kernel_last_timeout_error_time_cell=order_kernel_last_timeout_error_time[cell_idx];   
    unsigned long long kernel_start = __globaltimer();

    __shared__ uint32_t done_shared_sh;
    __shared__ uint32_t exit_rx_cta_sh;      
    __shared__ uint32_t pusch_prb_symbol_ordered[ORAN_PUSCH_SYMBOLS_X_SLOT];
    __shared__ uint32_t last_stride_idx;
    __shared__ uint64_t rx_buf_idx;
    __shared__ uint32_t pusch_prb_symbol_ordered_done[ORAN_PUSCH_SYMBOLS_X_SLOT];

	uint16_t startPRB_offset_idx_0 = 0;
	uint16_t startPRB_offset_idx_1 = 0;
	uint16_t startPRB_offset_idx_2 = 0;
	uint16_t startPRB_offset_idx_3 = 0;
    uint32_t cell_idx_mask = (0x1<<cell_idx);
    int prb_x_slot=pusch_prb_x_slot[cell_idx]+prach_prb_x_slot[cell_idx];
	int pusch_prb_x_slot_cell=pusch_prb_x_slot[cell_idx];
	int prach_prb_x_slot_cell=prach_prb_x_slot[cell_idx];    
    int sem_idx_rx = (int)(*(*(last_sem_idx_rx_h+cell_idx)));
    uint32_t* last_sem_idx_rx_h_cell=*(last_sem_idx_rx_h+cell_idx);
    uint32_t* last_sem_idx_order_h_cell=*(last_sem_idx_order_h+cell_idx);
    struct doca_gpu_eth_rxq* doca_rxq_cell=*(doca_rxq+cell_idx);
    const uint64_t timeout_ns = rx_pkts_timeout_ns;
    struct doca_gpu_semaphore_gpu* sem_gpu_cell=*(sem_gpu+cell_idx);
    uint64_t		slot_start_cell=slot_start[cell_idx];
	uint64_t		ta4_min_ns_cell=ta4_min_ns[cell_idx];
	uint64_t		ta4_max_ns_cell=ta4_max_ns[cell_idx];
	uint64_t		slot_duration_cell=slot_duration[cell_idx];
    uint8_t dummy8 = 0;
    uint32_t dummy32 = 0;
    uint64_t dummy64 = 0;
    __shared__ uint32_t dummy32_sh;

    __shared__ uint32_t rx_pkt_num;
    __shared__ uint32_t rx_pkt_num_total;
    __shared__ uint32_t next_slot_rx_packets_count_sh[ORAN_MAX_SYMBOLS];
    __shared__ uint32_t next_slot_rx_bytes_count_sh[ORAN_MAX_SYMBOLS];
    __shared__ uint32_t rx_packets_count_sh[ORAN_MAX_SYMBOLS];
    __shared__ uint32_t rx_bytes_count_sh[ORAN_MAX_SYMBOLS];
    __shared__ uint32_t early_rx_packets_count_sh;
    __shared__ uint32_t on_time_rx_packets_count_sh;
    __shared__ uint32_t late_rx_packets_count_sh;
    __shared__ uint32_t next_slot_early_rx_packets_count_sh;
    __shared__ uint32_t next_slot_on_time_rx_packets_count_sh;
    __shared__ uint32_t next_slot_late_rx_packets_count_sh;   
    __shared__ uint64_t next_slot_rx_packets_ts_sh[ORDER_KERNEL_MAX_PKTS_PER_OFDM_SYM*ORAN_MAX_SYMBOLS];
    __shared__ uint64_t rx_packets_ts_sh[ORDER_KERNEL_MAX_PKTS_PER_OFDM_SYM*ORAN_MAX_SYMBOLS];
    __shared__ uint64_t rx_packets_ts_earliest_sh[ORAN_MAX_SYMBOLS];
    __shared__ uint64_t rx_packets_ts_latest_sh[ORAN_MAX_SYMBOLS];  
	__shared__ uint32_t num_prb_ch1_sh;
	__shared__ uint32_t num_prb_ch2_sh;
	__shared__ uint32_t next_slot_num_prb_ch1_sh;
	__shared__ uint32_t next_slot_num_prb_ch2_sh;
    __shared__ struct doca_gpu_dev_eth_rxq_attr out_attr_sh[512];

    // COVERITY_DEVIATION: blockIdx.x is uniform across all threads in a block, so all threads
    // in this block will take the same branch. No actual thread divergence occurs.
    if((blockIdx.x & 0x1) == 0) //Receive CTA
    {
        if(threadIdx.x==0){
            printf("[Receive CTA]Entered receive_process_kernel_for_test_bench() for F%dS%dS%d cell ID %d\n",frameId,subframeId,slotId,cell_idx);            
            printf("[Receive CTA]Exited receive_process_kernel_for_test_bench() for F%dS%dS%d cell ID %d\n",frameId,subframeId,slotId,cell_idx);            
        }
        if(threadIdx.x==0)
        {
            rx_pkt_num=rx_pkt_num_slot_cell;
            exit_rx_cta_sh=0;
        }
        sem_idx_rx=0;
        // COVERITY_DEVIATION: blockIdx.x is uniform across all threads in a block.
        // All threads in this block will reach __syncthreads(), no actual divergence.
        // coverity[CUDA.DIVERGENCE_AT_COLLECTIVE_OPERATION]
        __syncthreads();
        order_kernel_doca_receive_packets_subSlot(true, tb_fh_buf_cell, max_pkt_size, first_packet_received, first_packet_received_time, timeout_first_pkt_ns, timeout_no_pkt_ns, timeout_log_enable, order_kernel_last_timeout_error_time_cell, timeout_log_interval_ns, kernel_start, cell_idx, sem_idx_rx, last_sem_idx_rx_h_cell, sem_idx_order, last_sem_idx_order_h_cell, pusch_ordered_prbs_cell, pusch_prb_x_slot_cell, prach_ordered_prbs_cell, prach_prb_x_slot_cell, frameId, subframeId, slotId, done_shared_sh, last_stride_idx, rx_pkt_num_total, sym_ord_done_sig_arr, exit_cond_d_cell, rx_pkt_num, commViaCpu, doca_rxq_cell, max_rx_pkts, timeout_ns, rx_buf_idx, sem_gpu_cell, slot_start_cell, ta4_min_ns_cell, ta4_max_ns_cell, slot_duration_cell, next_slot_rx_packets_count_sh, next_slot_rx_bytes_count_sh, rx_packets_count_sh, rx_bytes_count_sh, next_slot_early_rx_packets_count_sh, next_slot_on_time_rx_packets_count_sh, next_slot_late_rx_packets_count_sh, early_rx_packets_count_sh, on_time_rx_packets_count_sh, late_rx_packets_count_sh, ul_rx_pkt_tracing_level, next_slot_rx_packets_ts_sh, rx_packets_ts_sh, bit_width_cell, rx_packets_ts_earliest_sh, rx_packets_ts_latest_sh, num_prb_ch1_sh, num_prb_ch2_sh, next_slot_num_prb_ch1_sh, next_slot_num_prb_ch2_sh, sem_order_num_cell, prach_section_id_0, prb_x_slot, exit_rx_cta_sh, nullptr, nullptr, dummy8, dummy64, dummy32_sh, out_attr_sh);
        if(threadIdx.x == 0)
        {
            printf("[Receive CTA]Exited receive_process_kernel_for_test_bench() for F%dS%dS%d cell ID %d\n", frameId, subframeId, slotId, cell_idx);
        }                                                    
        __syncthreads();
    }
    else //Process CTA
    {
        if(threadIdx.x==0){
            printf("[Process CTA]Entered receive_process_kernel_for_test_bench() for F%dS%dS%d cell ID %d\n",frameId,subframeId,slotId,cell_idx);
        }
        __syncthreads();
        while (DOCA_GPUNETIO_VOLATILE(*exit_cond_d_cell) == ORDER_KERNEL_RUNNING)
        {
                if(threadIdx.x==0){
                printf("Entered while loop processing inside receive_process_kernel_for_test_bench() for F%dS%dS%d cell ID%d rx_pkt_num_slot_cell %d pusch_ordered_prbs_cell %d prach_ordered_prbs_cell %d prb_x_slot %d pusch_prb_symbol_map_cell[0] %d num_order_cells_sym_mask_arr[0] %d\n",frameId,subframeId,slotId,cell_idx,rx_pkt_num_slot_cell,pusch_ordered_prbs_cell[0],prach_ordered_prbs_cell[0],prb_x_slot,pusch_prb_symbol_map_cell[0],num_order_cells_sym_mask_arr[0]);
                do {
                    ret = doca_gpu_dev_semaphore_get_packet_info_status(sem_gpu_cell, 0, DOCA_GPU_SEMAPHORE_STATUS_READY, &rx_pkt_num, &rx_buf_idx);
                } while (ret == DOCA_ERROR_NOT_FOUND && DOCA_GPUNETIO_VOLATILE(*exit_cond_d_cell) == ORDER_KERNEL_RUNNING);
                }
                /* Order & decompress packets */
                order_kernel_doca_process_receive_packets_subSlot(true,tb_fh_buf_cell,max_pkt_size,rx_pkt_num_slot_cell,warpId,nwarps,laneId,nullptr,0,exit_cond_d_cell,frameId,subframeId,slotId,
                                                                &done_shared_sh,&last_stride_idx,bit_width_cell,comp_meth_cell,beta_cell,ru_type_cell,pkt_offset_ptr,gbuf_offset_ptr,prb_size,cell_idx_mask,prb_x_slot,
                                                                pusch_eAxC_map_cell,pusch_eAxC_num_cell,pusch_symbols_x_slot,pusch_prb_x_port_x_symbol_cell,pusch_buffer_cell,pusch_ordered_prbs_cell,
                                                                pusch_prb_symbol_map_cell,pusch_prb_symbol_ordered,pusch_prb_symbol_ordered_done,sym_ord_done_sig_arr,sym_ord_done_mask_arr,num_order_cells_sym_mask_arr,1,
                                                                prach_eAxC_map_cell,prach_eAxC_num_cell,prach_symbols_x_slot,prach_prb_x_port_x_symbol_cell,prach_section_id_0,prach_section_id_1,prach_section_id_2,prach_section_id_3,
                                                                prach_buffer_0_cell,prach_buffer_1_cell,prach_buffer_2_cell,prach_buffer_3_cell,prach_ordered_prbs_cell);
                __syncthreads();
                if(threadIdx.x == 0 && DOCA_GPUNETIO_VOLATILE(done_shared_sh) == 1) {
                    last_sem_idx_order = (last_sem_idx_order + 1) & (sem_order_num_cell - 1);
                }
                sem_idx_order = (sem_idx_order+1) & (sem_order_num_cell - 1);
        }            
        if(threadIdx.x==0){
            printf("[Process CTA]Exited receive_process_kernel_for_test_bench() for F%dS%dS%d cell ID%d exit_cond_d_cell %d\n",frameId,subframeId,slotId,cell_idx,DOCA_GPUNETIO_VOLATILE(*exit_cond_d_cell));
        }
        __syncthreads();
    }    
}



__global__ void order_kernel_doca_single_subSlot(
	/* DOCA objects */
	struct doca_gpu_eth_rxq **doca_rxq,
	struct doca_gpu_semaphore_gpu **sem_gpu,
	const uint16_t* sem_order_num,

	/* Cell */
	const int*		cell_id,
    const int*		ru_type,
    const bool*     cell_health,

	uint32_t		**start_cuphy_d,
	uint32_t		**exit_cond_d,
	uint32_t		**last_sem_idx_rx_h,
	uint32_t		**last_sem_idx_order_h,
	const int*		comp_meth,
    const int*		bit_width,
	const float*		beta,
	const int		prb_size,

	/* Timeout */
	const uint32_t	timeout_no_pkt_ns,
	const uint32_t	timeout_first_pkt_ns,
	const uint32_t  timeout_log_interval_ns,
	const uint8_t   timeout_log_enable,
	const uint32_t  max_rx_pkts,
    const uint32_t  rx_pkts_timeout_ns,
    bool            commViaCpu,

	/* ORAN */
	const uint8_t		frameId,
	const uint8_t		subframeId,
	const uint8_t		slotId,

	/* Order kernel specific */
	int			*barrier_flag,
	uint8_t		**done_shared,

	/* Timer */
	uint32_t 		**early_rx_packets,
	uint32_t 		**on_time_rx_packets,
	uint32_t 		**late_rx_packets,
	uint32_t 		**next_slot_early_rx_packets,
	uint32_t 		**next_slot_on_time_rx_packets,
	uint32_t 		**next_slot_late_rx_packets,	
	uint64_t*		slot_start,
	uint64_t*		ta4_min_ns,
	uint64_t*		ta4_max_ns,
	uint64_t*		slot_duration,
    uint64_t      **order_kernel_last_timeout_error_time,
    uint8_t                ul_rx_pkt_tracing_level,
	uint64_t**             rx_packets_ts,
	uint32_t**              rx_packets_count,
    uint32_t**              rx_bytes_count,
    uint64_t**             rx_packets_ts_earliest,
    uint64_t**             rx_packets_ts_latest,	
	uint64_t**             next_slot_rx_packets_ts,
	uint32_t**             next_slot_rx_packets_count,
    uint32_t**             next_slot_rx_bytes_count,
	uint32_t**             next_slot_num_prb_ch1,
	uint32_t**             next_slot_num_prb_ch2,

    /* Sub-slot processing*/
    uint32_t* 			   sym_ord_done_sig_arr,
    uint32_t*              sym_ord_done_mask_arr,
    uint32_t*              pusch_prb_symbol_map,
	uint32_t* 			   num_order_cells_sym_mask_arr,	    
	uint8_t                pusch_prb_non_zero,

	/* PUSCH */
	uint16_t		**pusch_eAxC_map,
	int*			pusch_eAxC_num,
	uint8_t		**pusch_buffer,
	int*			pusch_prb_x_slot,
	int*			pusch_prb_x_symbol,
	int*			pusch_prb_x_symbol_x_antenna,
	int			pusch_symbols_x_slot,
	uint32_t*			pusch_prb_x_port_x_symbol,
	uint32_t		**pusch_ordered_prbs,

	/* PRACH */
	uint16_t 	**prach_eAxC_map,
	int*		prach_eAxC_num,
	uint8_t		**prach_buffer_0,
	uint8_t		**prach_buffer_1,
	uint8_t 	**prach_buffer_2,
	uint8_t 	**prach_buffer_3,
	uint16_t	prach_section_id_0,
	uint16_t	prach_section_id_1,
	uint16_t	prach_section_id_2,
	uint16_t	prach_section_id_3,
	int*		prach_prb_x_slot,
	int*		prach_prb_x_symbol,
	int*		prach_prb_x_symbol_x_antenna,
	int			prach_symbols_x_slot,
	uint32_t*	prach_prb_x_port_x_symbol,
	uint32_t	**prach_ordered_prbs,
    uint8_t num_order_cells,
    uint8_t **pcap_buffer,
    uint8_t **pcap_buffer_ts,
    uint32_t **pcap_buffer_index,
    uint8_t pcap_capture_enable,
    uint64_t pcap_capture_cell_bitmask,
    uint16_t max_pkt_size
)
{
	int laneId = threadIdx.x % 32;
	int warpId = threadIdx.x / 32;
	int nwarps = blockDim.x / 32;
	int cell_idx = blockIdx.x / 2;
    uint32_t cell_idx_mask = (0x1<<cell_idx);

	uint8_t first_packet_received = 0;
	unsigned long long first_packet_received_time = 0;
	unsigned long long current_time = 0;
	unsigned long long kernel_start = __globaltimer();
	uint8_t* pkt_offset_ptr, *gbuf_offset_ptr;
	uint8_t* buffer;
	int prb_x_slot=pusch_prb_x_slot[cell_idx]+prach_prb_x_slot[cell_idx];
	doca_error_t ret = (doca_error_t)0;
	// Restart from last semaphore item
	int sem_idx_rx = (int)(*(*(last_sem_idx_rx_h+cell_idx)));
	int sem_idx_order = (int)(*(*(last_sem_idx_order_h+cell_idx)));
	int last_sem_idx_order = (int)(*(*(last_sem_idx_order_h+cell_idx)));
	const uint64_t timeout_ns = rx_pkts_timeout_ns;
	//int  barrier_idx = 1, barrier_signal = gridDim.x;
	//unsigned long long t0, t1, t2, t3, t4, t5, t6, t7;

	__shared__ uint32_t rx_pkt_num;
    __shared__ uint32_t rx_pkt_num_total;
	__shared__ uint64_t rx_buf_idx;
	__shared__ uint32_t done_shared_sh;
	__shared__ uint32_t last_stride_idx;
	__shared__ uint32_t num_prb_ch1_sh;
	__shared__ uint32_t num_prb_ch2_sh;
	__shared__ uint32_t next_slot_num_prb_ch1_sh;
	__shared__ uint32_t next_slot_num_prb_ch2_sh;    
	__shared__ uint32_t exit_rx_cta_sh;    
	uint64_t rx_timestamp;
    __shared__ uint32_t early_rx_packets_count_sh;
    __shared__ uint32_t on_time_rx_packets_count_sh;
    __shared__ uint32_t late_rx_packets_count_sh;
    __shared__ uint32_t next_slot_early_rx_packets_count_sh;
    __shared__ uint32_t next_slot_on_time_rx_packets_count_sh;
    __shared__ uint32_t next_slot_late_rx_packets_count_sh;       
    __shared__ uint64_t rx_packets_ts_sh[ORDER_KERNEL_MAX_PKTS_PER_OFDM_SYM*ORAN_MAX_SYMBOLS];
    __shared__ uint32_t rx_packets_count_sh[ORAN_MAX_SYMBOLS];   
    __shared__ uint32_t rx_bytes_count_sh[ORAN_MAX_SYMBOLS];     
    __shared__ uint64_t rx_packets_ts_earliest_sh[ORAN_MAX_SYMBOLS];
    __shared__ uint64_t rx_packets_ts_latest_sh[ORAN_MAX_SYMBOLS];
    __shared__ uint64_t next_slot_rx_packets_ts_sh[ORDER_KERNEL_MAX_PKTS_PER_OFDM_SYM*ORAN_MAX_SYMBOLS];
    __shared__ uint32_t next_slot_rx_packets_count_sh[ORAN_MAX_SYMBOLS];
    __shared__ uint32_t next_slot_rx_bytes_count_sh[ORAN_MAX_SYMBOLS];
	__shared__ uint32_t pusch_prb_symbol_ordered[ORAN_PUSCH_SYMBOLS_X_SLOT];
	__shared__ uint32_t pusch_prb_symbol_ordered_done[ORAN_PUSCH_SYMBOLS_X_SLOT];
    __shared__ uint32_t pcap_pkt_num_total;
    __shared__ struct doca_gpu_dev_eth_rxq_attr out_attr_sh[512];

	//Cell specific (de-reference from host pinned memory once)
	uint8_t* done_shared_cell=*(done_shared+cell_idx);
	uint32_t* pusch_ordered_prbs_cell=*(pusch_ordered_prbs+cell_idx);
	uint32_t* prach_ordered_prbs_cell=*(prach_ordered_prbs+cell_idx);
	uint32_t* exit_cond_d_cell=*(exit_cond_d+cell_idx);
	uint32_t* last_sem_idx_rx_h_cell=*(last_sem_idx_rx_h+cell_idx);
	uint32_t* last_sem_idx_order_h_cell=*(last_sem_idx_order_h+cell_idx);
	struct doca_gpu_eth_rxq* doca_rxq_cell=*(doca_rxq+cell_idx);
	struct doca_gpu_semaphore_gpu* sem_gpu_cell=*(sem_gpu+cell_idx);
	int pusch_prb_x_slot_cell=pusch_prb_x_slot[cell_idx];
	int prach_prb_x_slot_cell=prach_prb_x_slot[cell_idx];

	uint64_t		slot_start_cell=slot_start[cell_idx];
	uint64_t		ta4_min_ns_cell=ta4_min_ns[cell_idx];
	uint64_t		ta4_max_ns_cell=ta4_max_ns[cell_idx];
	uint64_t		slot_duration_cell=slot_duration[cell_idx];
	uint32_t 		*early_rx_packets_cell= *(early_rx_packets+cell_idx);
	uint32_t 		*on_time_rx_packets_cell= *(on_time_rx_packets+cell_idx);
	uint32_t 		*late_rx_packets_cell= *(late_rx_packets+cell_idx);
	uint32_t 		*next_slot_early_rx_packets_cell= *(next_slot_early_rx_packets+cell_idx);
	uint32_t 		*next_slot_on_time_rx_packets_cell= *(next_slot_on_time_rx_packets+cell_idx);
	uint32_t 		*next_slot_late_rx_packets_cell= *(next_slot_late_rx_packets+cell_idx);      
    uint64_t*       rx_packets_ts_cell=*(rx_packets_ts+cell_idx);
    uint32_t*        rx_packets_count_cell=*(rx_packets_count+cell_idx);    
    uint32_t*        rx_bytes_count_cell=*(rx_bytes_count+cell_idx);    
    uint64_t*       rx_packets_ts_earliest_cell = *(rx_packets_ts_earliest+cell_idx);
    uint64_t*       rx_packets_ts_latest_cell = *(rx_packets_ts_latest+cell_idx);
    uint64_t*       next_slot_rx_packets_ts_cell=*(next_slot_rx_packets_ts+cell_idx);
    uint32_t*       next_slot_rx_packets_count_cell=*(next_slot_rx_packets_count+cell_idx);
    uint32_t*       next_slot_rx_bytes_count_cell=*(next_slot_rx_bytes_count+cell_idx);
    uint32_t*       next_slot_num_prb_ch1_cell=*(next_slot_num_prb_ch1+cell_idx);
    uint32_t*       next_slot_num_prb_ch2_cell=*(next_slot_num_prb_ch2+cell_idx);    

	uint8_t			*pusch_buffer_cell=*(pusch_buffer+cell_idx);
	uint8_t			*pcap_buffer_cell=*(pcap_buffer+cell_idx);
	uint8_t			*pcap_buffer_ts_cell=*(pcap_buffer_ts+cell_idx);
	uint32_t		*pcap_buffer_index_cell=*(pcap_buffer_index+cell_idx);
	uint16_t		*pusch_eAxC_map_cell=*(pusch_eAxC_map+cell_idx);
	int			pusch_eAxC_num_cell=pusch_eAxC_num[cell_idx];
	uint32_t		pusch_prb_x_port_x_symbol_cell=	pusch_prb_x_port_x_symbol[cell_idx];
	uint8_t			*prach_buffer_0_cell=*(prach_buffer_0+cell_idx);
	uint8_t			*prach_buffer_1_cell=*(prach_buffer_1+cell_idx);
	uint8_t			*prach_buffer_2_cell=*(prach_buffer_2+cell_idx);
	uint8_t			*prach_buffer_3_cell=*(prach_buffer_3+cell_idx);
	uint16_t		*prach_eAxC_map_cell=*(prach_eAxC_map+cell_idx);
	int			prach_eAxC_num_cell=prach_eAxC_num[cell_idx];
	uint32_t		prach_prb_x_port_x_symbol_cell=	prach_prb_x_port_x_symbol[cell_idx];
    const int		ru_type_cell=ru_type[cell_idx];
	const int		comp_meth_cell=comp_meth[cell_idx];
    const int		bit_width_cell=bit_width[cell_idx];
	const float		beta_cell=beta[cell_idx];
	uint32_t		*start_cuphy_d_cell=*(start_cuphy_d+cell_idx);
	const uint16_t sem_order_num_cell=sem_order_num[cell_idx];
    uint64_t* order_kernel_last_timeout_error_time_cell=order_kernel_last_timeout_error_time[cell_idx];

    uint32_t* pusch_prb_symbol_map_cell = pusch_prb_symbol_map+(ORAN_PUSCH_SYMBOLS_X_SLOT*cell_idx);
    bool            cell_healthy = cell_health[cell_idx];
    // PRACH start_prbu = 0
	uint16_t startPRB_offset_idx_0 = 0;
	uint16_t startPRB_offset_idx_1 = 0;
	uint16_t startPRB_offset_idx_2 = 0;
	uint16_t startPRB_offset_idx_3 = 0;

    uint8_t *pkt_thread = NULL;
    uint8_t frameId_pkt=0;
    uint8_t symbol_id_pkt = 0;
    uint8_t subframeId_pkt  = 0;
    uint8_t slotId_pkt      = 0;                    
    uint64_t packet_early_thres = 0;
    uint64_t packet_late_thres  = 0;                    
    int rx_packets_ts_idx=0,next_slot_rx_packets_ts_idx=0,max_pkt_idx=ORDER_KERNEL_MAX_PKTS_PER_OFDM_SYM*ORAN_MAX_SYMBOLS;    

	uint32_t tot_pusch_prb_symbol_ordered=0;
    int32_t slot_count_input=(2*subframeId+slotId),slot_count_curr;
    uint8_t* section_buf;
    uint16_t ecpri_payload_length;
    // 4 bytes for ecpriPcid, ecprSeqid, ecpriEbit, ecpriSubSeqid
    uint16_t current_length = 4 + sizeof(oran_umsg_iq_hdr);
    uint16_t num_prb = 0;
    uint16_t start_prb = 0;
    uint16_t section_id = 0;
    uint16_t compressed_prb_size = (bit_width_cell == BFP_NO_COMPRESSION) ? PRB_SIZE_16F : (bit_width_cell == BFP_COMPRESSION_14_BITS) ? PRB_SIZE_14F : PRB_SIZE_9F;
    uint16_t num_sections = 0;        

    // if (threadIdx.x==0) {
    //     printf("[Single Order kernel start]sem_idx_rx %d sem_idx_order %d last_sem_idx_order %d sem_idx_pcap %d last_sem_idx_pcap %d F%dS%dS%d\n",
    //         sem_idx_rx,sem_idx_order,last_sem_idx_order,sem_idx_pcap,last_sem_idx_pcap,frameId, subframeId, slotId);
    // }


	if((blockIdx.x & 0x1) == 1) {
		if(threadIdx.x==0){
			DOCA_GPUNETIO_VOLATILE(done_shared_cell[0]) = 1;
			DOCA_GPUNETIO_VOLATILE(pusch_ordered_prbs_cell[0]) = 0;
			DOCA_GPUNETIO_VOLATILE(prach_ordered_prbs_cell[0]) = 0;
			DOCA_GPUNETIO_VOLATILE(done_shared_sh) = 1;
			DOCA_GPUNETIO_VOLATILE(last_stride_idx) = 0;
            DOCA_GPUNETIO_VOLATILE(pusch_prb_symbol_ordered[threadIdx.x])=0;
            DOCA_GPUNETIO_VOLATILE(pusch_prb_symbol_ordered_done[threadIdx.x])=0;
            if(!cell_healthy)
            {
                DOCA_GPUNETIO_VOLATILE(*exit_cond_d_cell)=ORDER_KERNEL_EXIT_PRB;
                if(pusch_prb_non_zero)
                {
                    for(uint32_t idx=0;idx<ORAN_PUSCH_SYMBOLS_X_SLOT;idx++)
                    {
                        atomicOr(&sym_ord_done_mask_arr[idx],cell_idx_mask);
                    }
                }
            }
		}
		else if(threadIdx.x < ORAN_PUSCH_SYMBOLS_X_SLOT){
            DOCA_GPUNETIO_VOLATILE(pusch_prb_symbol_ordered[threadIdx.x])=0;
            DOCA_GPUNETIO_VOLATILE(pusch_prb_symbol_ordered_done[threadIdx.x])=0;
		}
	}
	if((blockIdx.x & 0x1) == 0 && threadIdx.x == 0) {
		DOCA_GPUNETIO_VOLATILE(rx_pkt_num_total) = 0;
		DOCA_GPUNETIO_VOLATILE(pcap_pkt_num_total) = DOCA_GPUNETIO_VOLATILE(*pcap_buffer_index_cell);
        early_rx_packets_count_sh=DOCA_GPUNETIO_VOLATILE(*next_slot_early_rx_packets_cell);
        on_time_rx_packets_count_sh=DOCA_GPUNETIO_VOLATILE(*next_slot_on_time_rx_packets_cell);
        late_rx_packets_count_sh=DOCA_GPUNETIO_VOLATILE(*next_slot_late_rx_packets_cell);
        next_slot_early_rx_packets_count_sh=0;
        next_slot_late_rx_packets_count_sh=0;
        next_slot_on_time_rx_packets_count_sh=0;
        num_prb_ch1_sh=DOCA_GPUNETIO_VOLATILE(*next_slot_num_prb_ch1_cell);
        num_prb_ch2_sh=DOCA_GPUNETIO_VOLATILE(*next_slot_num_prb_ch2_cell);
        next_slot_num_prb_ch1_sh=0;
        next_slot_num_prb_ch2_sh=0;
        exit_rx_cta_sh=0;
    }


    if(ul_rx_pkt_tracing_level){
        if((blockIdx.x & 0x1) == 0)
        {
            if(threadIdx.x<ORAN_MAX_SYMBOLS)
            {
                rx_packets_count_sh[threadIdx.x]=DOCA_GPUNETIO_VOLATILE(next_slot_rx_packets_count_cell[threadIdx.x]);
                rx_bytes_count_sh[threadIdx.x]=DOCA_GPUNETIO_VOLATILE(next_slot_rx_bytes_count_cell[threadIdx.x]);
                rx_packets_ts_earliest_sh[threadIdx.x]=0xFFFFFFFFFFFFFFFFLLU;
                rx_packets_ts_latest_sh[threadIdx.x]=0;
                next_slot_rx_packets_count_sh[threadIdx.x]=0;
                next_slot_rx_bytes_count_sh[threadIdx.x]=0;
            }
            // COVERITY_DEVIATION: blockIdx.x is uniform across all threads in a block.
            // All threads in this block will reach __syncthreads(), no actual divergence.
            // coverity[CUDA.DIVERGENCE_AT_COLLECTIVE_OPERATION]
            __syncthreads();
            for(uint32_t pkt_idx=threadIdx.x;pkt_idx<max_pkt_idx;pkt_idx+=blockDim.x)
            {
                uint32_t symbol_idx=pkt_idx/ORDER_KERNEL_MAX_PKTS_PER_OFDM_SYM;
                rx_packets_ts_sh[pkt_idx]=DOCA_GPUNETIO_VOLATILE(next_slot_rx_packets_ts_cell[pkt_idx]);
                __threadfence_block();
                if(rx_packets_ts_sh[pkt_idx]!=0)
                    atomicMin((unsigned long long*) &rx_packets_ts_earliest_sh[symbol_idx],(unsigned long long) rx_packets_ts_sh[pkt_idx]);
                atomicMax((unsigned long long*) &rx_packets_ts_latest_sh[symbol_idx],(unsigned long long) rx_packets_ts_sh[pkt_idx]);
                __threadfence_block();
                next_slot_rx_packets_ts_sh[pkt_idx]=0;
            }
            __syncthreads();
            if(threadIdx.x<ORAN_MAX_SYMBOLS)
            {
                if(rx_packets_ts_earliest_sh[threadIdx.x]==0)
                    rx_packets_ts_earliest_sh[threadIdx.x]=0xFFFFFFFFFFFFFFFFLLU;
            }            
        }            
    }    
	__syncthreads();

		/* Even receives packets and forward them to Odd Block */
    if ((blockIdx.x & 0x1) == 0) 
    {            
        order_kernel_doca_receive_packets_subSlot(false,nullptr,max_pkt_size,first_packet_received,first_packet_received_time,timeout_first_pkt_ns,timeout_no_pkt_ns,timeout_log_enable,order_kernel_last_timeout_error_time_cell,timeout_log_interval_ns,kernel_start,
                                                    cell_idx,sem_idx_rx,last_sem_idx_rx_h_cell,sem_idx_order,last_sem_idx_order_h_cell,pusch_ordered_prbs_cell,pusch_prb_x_slot_cell,prach_ordered_prbs_cell,prach_prb_x_slot_cell,
                                                    frameId,subframeId,slotId,done_shared_sh,last_stride_idx,rx_pkt_num_total,sym_ord_done_sig_arr,exit_cond_d_cell,rx_pkt_num,commViaCpu,doca_rxq_cell,max_rx_pkts,timeout_ns,
                                                    rx_buf_idx,sem_gpu_cell,slot_start_cell,ta4_min_ns_cell,ta4_max_ns_cell,slot_duration_cell,next_slot_rx_packets_count_sh,next_slot_rx_bytes_count_sh,rx_packets_count_sh,rx_bytes_count_sh,
                                                    next_slot_early_rx_packets_count_sh,next_slot_on_time_rx_packets_count_sh,next_slot_late_rx_packets_count_sh,early_rx_packets_count_sh,on_time_rx_packets_count_sh,late_rx_packets_count_sh,
                                                    ul_rx_pkt_tracing_level,next_slot_rx_packets_ts_sh,rx_packets_ts_sh,bit_width_cell,rx_packets_ts_earliest_sh,rx_packets_ts_latest_sh,num_prb_ch1_sh,num_prb_ch2_sh,next_slot_num_prb_ch1_sh,next_slot_num_prb_ch2_sh,sem_order_num_cell,prach_section_id_0,prb_x_slot,exit_rx_cta_sh,
                                                    pcap_buffer_cell, pcap_buffer_ts_cell, pcap_capture_enable, pcap_capture_cell_bitmask, pcap_pkt_num_total, out_attr_sh
                                                );
    }
    else
    {
        while (DOCA_GPUNETIO_VOLATILE(*exit_cond_d_cell) == ORDER_KERNEL_RUNNING)
        {
            /* Block 1 waits on semaphore for new packets and process them */

            /* Semaphore wait */
            if (threadIdx.x == 0) {
            // printf("Process kernel Block idx %d waiting on sem %d sem_gpu_cell %p\n", blockIdx.x, sem_idx_order, sem_gpu_cell);
            do {
                ret = doca_gpu_dev_semaphore_get_packet_info_status(sem_gpu_cell, sem_idx_order, DOCA_GPU_SEMAPHORE_STATUS_READY, &rx_pkt_num, &rx_buf_idx);
            } while (ret == DOCA_ERROR_NOT_FOUND && DOCA_GPUNETIO_VOLATILE(*exit_cond_d_cell) == ORDER_KERNEL_RUNNING);
                // printf("Process kernel Cell Idx %d F%dS%dS%d rx_pkt_num %d sem_idx_order %d rx_buf_idx %lu, pcap_pkt_num_total %d\n",cell_idx,frameId, subframeId, slotId, rx_pkt_num, sem_idx_order, rx_buf_idx, pcap_pkt_num_total);
            }
            // COVERITY_DEVIATION: blockIdx.x is uniform across all threads in a block.
            // All threads in this block will reach __syncthreads(), no actual divergence.
            // coverity[CUDA.DIVERGENCE_AT_COLLECTIVE_OPERATION]
            __syncthreads();

            /* Check error or exit condition */
            if (DOCA_GPUNETIO_VOLATILE(*exit_cond_d_cell) != ORDER_KERNEL_RUNNING) {
                break;
            }

            // This synchronization is needed because otherwise some thread may progress into order_kernel_doca_process_receive_packets_subSlot() and
            // set *exit_cond_d_cell before some other thread has read *exit_cond_d_cell above. If that happens, then the threads will diverge and
            // deadlock waiting on two different syncthreads.
            __syncthreads();

            if(DOCA_GPUNETIO_VOLATILE(rx_pkt_num) == 0)
                continue;

            order_kernel_doca_process_receive_packets_subSlot(false,nullptr,max_pkt_size,rx_pkt_num,warpId,nwarps,laneId,doca_rxq_cell,rx_buf_idx,exit_cond_d_cell,frameId,subframeId,slotId,
                                                            &done_shared_sh,&last_stride_idx,bit_width_cell,comp_meth_cell,beta_cell,ru_type_cell,pkt_offset_ptr,gbuf_offset_ptr,prb_size,cell_idx_mask,prb_x_slot,
                                                            pusch_eAxC_map_cell,pusch_eAxC_num_cell,pusch_symbols_x_slot,pusch_prb_x_port_x_symbol_cell,pusch_buffer_cell,pusch_ordered_prbs_cell,
                                                            pusch_prb_symbol_map_cell,pusch_prb_symbol_ordered,pusch_prb_symbol_ordered_done,sym_ord_done_sig_arr,sym_ord_done_mask_arr,num_order_cells_sym_mask_arr,pusch_prb_non_zero,
                                                            prach_eAxC_map_cell,prach_eAxC_num_cell,prach_symbols_x_slot,prach_prb_x_port_x_symbol_cell,prach_section_id_0,prach_section_id_1,prach_section_id_2,prach_section_id_3,
                                                        prach_buffer_0_cell,prach_buffer_1_cell,prach_buffer_2_cell,prach_buffer_3_cell,prach_ordered_prbs_cell);
        __syncthreads();

        if(threadIdx.x == 0 && DOCA_GPUNETIO_VOLATILE(done_shared_sh) == 1) {
            doca_gpu_dev_semaphore_set_status(sem_gpu_cell, last_sem_idx_order, DOCA_GPU_SEMAPHORE_STATUS_DONE);
            last_sem_idx_order = (last_sem_idx_order + 1) & (sem_order_num_cell - 1);
        }

        sem_idx_order = (sem_idx_order+1) & (sem_order_num_cell - 1);
        }
    }

	///////////////////////////////////////////////////////////
	// Inter-block barrier
	///////////////////////////////////////////////////////////
	// __threadfence();
	// __syncthreads();
	// if(threadIdx.x == 0)
	// 	ib_barrier(&(barrier_flag[0]), barrier_signal, barrier_idx);
	__syncthreads();
	///////////////////////////////////////////////////////////

	if(ul_rx_pkt_tracing_level){
        if((blockIdx.x & 0x1) == 0)
        {
            if(threadIdx.x<ORAN_MAX_SYMBOLS)
            {
                DOCA_GPUNETIO_VOLATILE(rx_packets_count_cell[threadIdx.x])=rx_packets_count_sh[threadIdx.x];
                DOCA_GPUNETIO_VOLATILE(rx_bytes_count_cell[threadIdx.x])=rx_bytes_count_sh[threadIdx.x];
                DOCA_GPUNETIO_VOLATILE(next_slot_rx_packets_count_cell[threadIdx.x])=next_slot_rx_packets_count_sh[threadIdx.x];
                DOCA_GPUNETIO_VOLATILE(next_slot_rx_bytes_count_cell[threadIdx.x])=next_slot_rx_bytes_count_sh[threadIdx.x];
                DOCA_GPUNETIO_VOLATILE(rx_packets_ts_earliest_cell[threadIdx.x])=rx_packets_ts_earliest_sh[threadIdx.x];
                DOCA_GPUNETIO_VOLATILE(rx_packets_ts_latest_cell[threadIdx.x])=rx_packets_ts_latest_sh[threadIdx.x];                
            }
            __syncthreads();
            for(uint32_t pkt_idx=threadIdx.x;pkt_idx<max_pkt_idx;pkt_idx+=blockDim.x)
            {
                DOCA_GPUNETIO_VOLATILE(rx_packets_ts_cell[pkt_idx])=rx_packets_ts_sh[pkt_idx];
                DOCA_GPUNETIO_VOLATILE(next_slot_rx_packets_ts_cell[pkt_idx])=next_slot_rx_packets_ts_sh[pkt_idx];
            }
            __syncthreads();
        }	        
    }    

	if (threadIdx.x == 0) {
		if((blockIdx.x & 0x1) == 0) {
			DOCA_GPUNETIO_VOLATILE(*last_sem_idx_rx_h_cell) = sem_idx_rx;
            DOCA_GPUNETIO_VOLATILE(*early_rx_packets_cell) = early_rx_packets_count_sh;
            DOCA_GPUNETIO_VOLATILE(*on_time_rx_packets_cell) = on_time_rx_packets_count_sh;
            DOCA_GPUNETIO_VOLATILE(*late_rx_packets_cell) = late_rx_packets_count_sh;      
            DOCA_GPUNETIO_VOLATILE(*next_slot_early_rx_packets_cell)=next_slot_early_rx_packets_count_sh;
            DOCA_GPUNETIO_VOLATILE(*next_slot_on_time_rx_packets_cell)=next_slot_on_time_rx_packets_count_sh;
            DOCA_GPUNETIO_VOLATILE(*next_slot_late_rx_packets_cell)=next_slot_late_rx_packets_count_sh;   
            DOCA_GPUNETIO_VOLATILE(*next_slot_num_prb_ch1_cell)=next_slot_num_prb_ch1_sh;
            DOCA_GPUNETIO_VOLATILE(*next_slot_num_prb_ch2_cell)=next_slot_num_prb_ch2_sh;
            DOCA_GPUNETIO_VOLATILE(*pcap_buffer_index_cell) = pcap_pkt_num_total % MAX_PKTS_PER_PCAP_BUFFER;
            // printf("[EXIT] RECEIVE kernel block F%dS%dS%d cell ID %d Order kernel exit with pcap_pkt_num_total %d buffer index %d\n", frameId,subframeId,slotId,cell_idx,DOCA_GPUNETIO_VOLATILE(pcap_pkt_num_total),DOCA_GPUNETIO_VOLATILE(*pcap_buffer_index_cell));
		} else {
			DOCA_GPUNETIO_VOLATILE(*last_sem_idx_order_h_cell) = last_sem_idx_order;
			DOCA_GPUNETIO_VOLATILE(*start_cuphy_d_cell) = 1;
            // printf("[EXIT] PROCESS kernel F%dS%dS%d cell ID %d Order kernel exit with pcap_pkt_num_total %d overlap_pkt_num_total %d\n", frameId,subframeId,slotId,cell_idx,DOCA_GPUNETIO_VOLATILE(pcap_pkt_num_total),DOCA_GPUNETIO_VOLATILE(overlap_pkt_num_total));
            /*
            for(uint32_t idx=0;idx<ORAN_PUSCH_SYMBOLS_X_SLOT;idx++){
                printf("Order kernel exit Cell %d F%dS%dS%d sym_idx %d sym_ord_done_sig_arr[sym_idx] %d\n",cell_idx,frameId, subframeId, slotId,idx,DOCA_GPUNETIO_VOLATILE(sym_ord_done_sig_arr[idx]));
            }
            */
		}
	}

	// __syncthreads();

	// if (threadIdx.x == 0) {
	// 	DOCA_GPUNETIO_VOLATILE(*last_sem_idx_rx_h_cell) = sem_idx_rx;
	// 	DOCA_GPUNETIO_VOLATILE(*last_sem_idx_order_h_cell) = last_sem_idx_order;
	// 	DOCA_GPUNETIO_VOLATILE(*start_cuphy_d_cell) = 1;
	// }

	return;
}

#ifdef __cplusplus
} // end extern "C" -- we need C++ linkage for the templates used in order_kernel_doca_single_subSlot_pingpong
#endif

struct order_kernel_pkt_tracing_info {
    uint32_t** rx_packets_count;
    uint32_t** rx_bytes_count;
    uint32_t** next_slot_rx_packets_count;
    uint32_t** next_slot_rx_bytes_count;
    uint64_t** rx_packets_ts_earliest;
    uint64_t** rx_packets_ts_latest;
    uint64_t** rx_packets_ts;
    uint64_t** next_slot_rx_packets_ts;
};

// Define launch bounds to support 2 CTAs per SM, even though we typically allocate one SM
// per CTA. This is intended to minimize the possibility of the order kernel being SM-starved.
// We would typically prefer to run two CTAs on a single SM rather than delay the launch of
// one or more CTAs.
template <bool ok_tb_enable, uint8_t ul_rx_pkt_tracing_level,uint8_t srs_enable, int NUM_THREADS, int NUM_CTAS_PER_SM>
__global__ void __launch_bounds__(NUM_THREADS,NUM_CTAS_PER_SM) order_kernel_doca_single_subSlot_pingpong(
    /* DOCA objects */
    struct doca_gpu_eth_rxq **doca_rxq,
    struct doca_gpu_semaphore_gpu **sem_gpu,
    struct aerial_fh_gpu_semaphore_gpu **sem_gpu_aerial_fh,
    const uint16_t* sem_order_num,

    /* Cell */
    const int*        cell_id,
    const int*        ru_type,
    const bool*     cell_health,

    uint32_t        **start_cuphy_d,
    uint32_t        **exit_cond_d,
    uint32_t        **last_sem_idx_rx_h,
    uint32_t        **last_sem_idx_order_h,
    const int*        comp_meth,
    const int*        bit_width,
    const float*        beta,
    const int        prb_size,

    /* Timeout */
    const uint32_t    timeout_no_pkt_ns,
    const uint32_t    timeout_first_pkt_ns,
    const uint32_t  timeout_log_interval_ns,
    const uint8_t   timeout_log_enable,
    const uint32_t  max_rx_pkts,
    const uint32_t  rx_pkts_timeout_ns,

    /* ORAN */
    const uint8_t        frameId,
    const uint8_t        subframeId,
    const uint8_t        slotId,

    /* Timer */
    uint32_t         **early_rx_packets,
    uint32_t         **on_time_rx_packets,
    uint32_t         **late_rx_packets,
    uint32_t         **next_slot_early_rx_packets,
    uint32_t         **next_slot_on_time_rx_packets,
    uint32_t         **next_slot_late_rx_packets,
    uint64_t*        slot_start,
    uint64_t*        ta4_min_ns,
    uint64_t*        ta4_max_ns,
    uint64_t*        slot_duration,
    uint64_t      **order_kernel_last_timeout_error_time,
    order_kernel_pkt_tracing_info pkt_tracing_info,
    uint32_t**             rx_packets_dropped_count,
    /* Sub-slot processing*/
    uint32_t*                sym_ord_done_sig_arr,
    uint32_t*              sym_ord_done_mask_arr,
    uint32_t*              pusch_prb_symbol_map,
    uint32_t*                num_order_cells_sym_mask_arr,
    uint8_t                pusch_prb_non_zero,

    /* PUSCH */
    uint16_t        **pusch_eAxC_map,
    int*            pusch_eAxC_num,
    uint8_t        **pusch_buffer,
    int*            pusch_prb_x_slot,
    int            pusch_symbols_x_slot,
    uint32_t*            pusch_prb_x_port_x_symbol,
    uint32_t        **pusch_ordered_prbs,

    /* PRACH */
    uint16_t     **prach_eAxC_map,
    int*        prach_eAxC_num,
    uint8_t        **prach_buffer_0,
    uint8_t        **prach_buffer_1,
    uint8_t     **prach_buffer_2,
    uint8_t     **prach_buffer_3,
    uint16_t    prach_section_id_0,
    uint16_t    prach_section_id_1,
    uint16_t    prach_section_id_2,
    uint16_t    prach_section_id_3,
    int*        prach_prb_x_slot,
    int            prach_symbols_x_slot,
    uint32_t*    prach_prb_x_port_x_symbol,
    uint32_t    **prach_ordered_prbs,

	/* SRS */
	uint16_t		**srs_eAxC_map,
	int*			srs_eAxC_num,
	uint8_t		**srs_buffer,
	int*			srs_prb_x_slot,
	int			srs_symbols_x_slot,
	uint32_t*			srs_prb_x_port_x_symbol,
	uint32_t		**srs_ordered_prbs,
    uint8_t*          srs_start_sym,

    uint8_t num_order_cells,
    /* PCAP Capture */
    uint8_t** pcap_buffer,
    uint8_t** pcap_buffer_ts,
    uint32_t** pcap_buffer_index,
    uint8_t pcap_capture_enable,
    uint64_t pcap_capture_cell_bitmask,
    /* Test bench */
    uint8_t     **tb_fh_buf,
    uint32_t    max_pkt_size,
    uint32_t*    rx_pkt_num_slot)
{
    int cell_idx = blockIdx.x;
    uint32_t cell_idx_mask = (0x1<<cell_idx);

    const unsigned long long kernel_start = __globaltimer();
    int prb_x_slot;
    // Restart from last semaphore item
    int sem_idx_rx = (int)(*(*(last_sem_idx_rx_h+cell_idx)));
    int sem_idx_order = (int)(*(*(last_sem_idx_order_h+cell_idx)));
    int last_sem_idx_order = (int)(*(*(last_sem_idx_order_h+cell_idx)));
    if constexpr (srs_enable==ORDER_KERNEL_SRS_ENABLE) {
        prb_x_slot = srs_prb_x_slot[cell_idx];
    } else if constexpr (srs_enable==ORDER_KERNEL_PUSCH_ONLY) {
        prb_x_slot = pusch_prb_x_slot[cell_idx] + prach_prb_x_slot[cell_idx];        
    } else {
        prb_x_slot = pusch_prb_x_slot[cell_idx] + prach_prb_x_slot[cell_idx] + srs_prb_x_slot[cell_idx];
    }

    __shared__ uint32_t rx_pkt_num;
    __shared__ uint64_t rx_buf_idx;
    __shared__ uint32_t done_shared_sh;
    __shared__ uint32_t early_rx_packets_count_sh;
    __shared__ uint32_t on_time_rx_packets_count_sh;
    __shared__ uint32_t late_rx_packets_count_sh;
    __shared__ uint32_t next_slot_early_rx_packets_count_sh;
    __shared__ uint32_t next_slot_on_time_rx_packets_count_sh;
    __shared__ uint32_t next_slot_late_rx_packets_count_sh;
    __shared__ uint32_t pusch_prb_symbol_ordered[ORAN_PUSCH_SYMBOLS_X_SLOT];
    __shared__ uint32_t pusch_prb_symbol_ordered_done[ORAN_PUSCH_SYMBOLS_X_SLOT];
    __shared__ uint32_t rx_packets_dropped_count_sh;
    __shared__ uint32_t pcap_pkt_num_total;
    // These shared memory arrays are only referenced when template parameter ul_rx_pkt_tracing_level
    // is non-zero. When ul_rx_pkt_tracing_level is 0, the compiler will not allocate this static shared memory.
    __shared__ uint32_t rx_packets_count_sh[ORAN_MAX_SYMBOLS];
    __shared__ uint32_t rx_bytes_count_sh[ORAN_MAX_SYMBOLS];
    __shared__ uint32_t next_slot_rx_packets_count_sh[ORAN_MAX_SYMBOLS];
    __shared__ uint32_t next_slot_rx_bytes_count_sh[ORAN_MAX_SYMBOLS];
    __shared__ uint64_t rx_packets_ts_earliest_sh[ORAN_MAX_SYMBOLS];
    __shared__ uint64_t rx_packets_ts_latest_sh[ORAN_MAX_SYMBOLS];
    __shared__ uint64_t rx_packets_ts_sh[ORDER_KERNEL_MAX_PKTS_PER_OFDM_SYM*ORAN_MAX_SYMBOLS];
    __shared__ uint64_t next_slot_rx_packets_ts_sh[ORDER_KERNEL_MAX_PKTS_PER_OFDM_SYM*ORAN_MAX_SYMBOLS];

    //Cell specific (de-reference from host pinned memory once)
    uint32_t* pusch_ordered_prbs_cell=*(pusch_ordered_prbs+cell_idx);
    uint32_t* prach_ordered_prbs_cell=*(prach_ordered_prbs+cell_idx);
    uint32_t* srs_ordered_prbs_cell=*(srs_ordered_prbs+cell_idx);
    uint32_t* exit_cond_d_cell=*(exit_cond_d+cell_idx);
    uint32_t* last_sem_idx_rx_h_cell=*(last_sem_idx_rx_h+cell_idx);
    uint32_t* last_sem_idx_order_h_cell=*(last_sem_idx_order_h+cell_idx);
    struct doca_gpu_eth_rxq* doca_rxq_cell=*(doca_rxq+cell_idx);
    struct doca_gpu_semaphore_gpu* sem_gpu_cell=*(sem_gpu+cell_idx);
    struct aerial_fh_gpu_semaphore_gpu* sem_gpu_aerial_fh_cell=*(sem_gpu_aerial_fh+cell_idx);
    int pusch_prb_x_slot_cell=pusch_prb_x_slot[cell_idx];
    int prach_prb_x_slot_cell=prach_prb_x_slot[cell_idx];

    uint64_t        slot_start_cell=slot_start[cell_idx];
    uint64_t        ta4_min_ns_cell=ta4_min_ns[cell_idx];
    uint64_t        ta4_max_ns_cell=ta4_max_ns[cell_idx];
    uint64_t        slot_duration_cell=slot_duration[cell_idx];
    uint32_t         *early_rx_packets_cell= *(early_rx_packets+cell_idx);
    uint32_t         *on_time_rx_packets_cell= *(on_time_rx_packets+cell_idx);
    uint32_t         *late_rx_packets_cell= *(late_rx_packets+cell_idx);
    uint32_t         *next_slot_early_rx_packets_cell= *(next_slot_early_rx_packets+cell_idx);
    uint32_t         *next_slot_on_time_rx_packets_cell= *(next_slot_on_time_rx_packets+cell_idx);
    uint32_t         *next_slot_late_rx_packets_cell= *(next_slot_late_rx_packets+cell_idx);
    uint32_t         *rx_packets_dropped_count_cell= *(rx_packets_dropped_count+cell_idx);

    uint8_t            *pusch_buffer_cell=*(pusch_buffer+cell_idx);
    uint16_t        *pusch_eAxC_map_cell=*(pusch_eAxC_map+cell_idx);
    int                pusch_eAxC_num_cell=pusch_eAxC_num[cell_idx];
    uint32_t        pusch_prb_x_port_x_symbol_cell=    pusch_prb_x_port_x_symbol[cell_idx];
    uint8_t            *prach_buffer_0_cell=*(prach_buffer_0+cell_idx);
    uint8_t            *prach_buffer_1_cell=*(prach_buffer_1+cell_idx);
    uint8_t            *prach_buffer_2_cell=*(prach_buffer_2+cell_idx);
    uint8_t            *prach_buffer_3_cell=*(prach_buffer_3+cell_idx);
    uint16_t        *prach_eAxC_map_cell=*(prach_eAxC_map+cell_idx);
    int                prach_eAxC_num_cell=prach_eAxC_num[cell_idx];
    uint32_t        prach_prb_x_port_x_symbol_cell=    prach_prb_x_port_x_symbol[cell_idx];

    uint8_t		*srs_buffer_cell=*(srs_buffer+cell_idx);
	uint16_t		*srs_eAxC_map_cell=*(srs_eAxC_map+cell_idx);
	int			srs_eAxC_num_cell=srs_eAxC_num[cell_idx];
    uint32_t			srs_prb_x_port_x_symbol_cell=	srs_prb_x_port_x_symbol[cell_idx];
    uint8_t         srs_start_sym_cell=srs_start_sym[cell_idx];
    int srs_prb_x_slot_cell=srs_prb_x_slot[cell_idx];
    // Combined ordering is selected for the slot; only cells with SRS resources
    // should use the SRS destination.
    const bool combined_srs_active_for_cell =
        (srs_enable == ORDER_KERNEL_SRS_AND_PUSCH) &&
        (srs_prb_x_slot_cell > 0) &&
        (srs_buffer_cell != nullptr);

    uint8_t*  pcap_buffer_cell       = nullptr;
    uint8_t*  pcap_buffer_ts_cell    = nullptr;
    uint32_t* pcap_buffer_index_cell = nullptr;

    if constexpr (!ok_tb_enable)
    {
        if(pcap_capture_enable)
        {
            pcap_buffer_cell       = *(pcap_buffer + cell_idx);
            pcap_buffer_ts_cell    = *(pcap_buffer_ts + cell_idx);
            pcap_buffer_index_cell = *(pcap_buffer_index + cell_idx);
        }
    }

    const int        ru_type_cell=ru_type[cell_idx];
    const int        comp_meth_cell=comp_meth[cell_idx];
    const int        bit_width_cell=bit_width[cell_idx];
    const float        beta_cell=beta[cell_idx];
    const uint16_t  sem_order_num_cell=sem_order_num[cell_idx];

    uint32_t* pusch_prb_symbol_map_cell = pusch_prb_symbol_map+(ORAN_PUSCH_SYMBOLS_X_SLOT*cell_idx);

    const uint32_t tid = threadIdx.x;
    const uint32_t nthreads = blockDim.x;
    const uint32_t warpId = threadIdx.x / 32;
    const uint32_t laneId = threadIdx.x % 32;
    const uint32_t nwarps = blockDim.x / 32;

    uint32_t max_rx_pkts_ = max(max_rx_pkts, blockDim.x);

    if(tid == 0){
        if constexpr (srs_enable==ORDER_KERNEL_SRS_ENABLE) {
            DOCA_GPUNETIO_VOLATILE(srs_ordered_prbs_cell[0]) = 0;
        } else if constexpr (srs_enable==ORDER_KERNEL_PUSCH_ONLY) {
            DOCA_GPUNETIO_VOLATILE(pusch_ordered_prbs_cell[0]) = 0;
            DOCA_GPUNETIO_VOLATILE(prach_ordered_prbs_cell[0]) = 0;
            DOCA_GPUNETIO_VOLATILE(pusch_prb_symbol_ordered[tid])=0;
            DOCA_GPUNETIO_VOLATILE(pusch_prb_symbol_ordered_done[tid])=0;
            const bool cell_healthy = cell_health[cell_idx];
            if(!cell_healthy)
            {
                atomicCAS(exit_cond_d_cell, ORDER_KERNEL_RUNNING, ORDER_KERNEL_EXIT_PRB);
                if(pusch_prb_non_zero)
                {
                    for(uint32_t idx=0;idx<ORAN_PUSCH_SYMBOLS_X_SLOT;idx++)
                    {
                        atomicOr(&sym_ord_done_mask_arr[idx],cell_idx_mask);
                    }
                }
            }
        } else {
            DOCA_GPUNETIO_VOLATILE(srs_ordered_prbs_cell[0]) = 0;
            DOCA_GPUNETIO_VOLATILE(pusch_ordered_prbs_cell[0]) = 0;
            DOCA_GPUNETIO_VOLATILE(prach_ordered_prbs_cell[0]) = 0;
            DOCA_GPUNETIO_VOLATILE(pusch_prb_symbol_ordered[tid])=0;
            DOCA_GPUNETIO_VOLATILE(pusch_prb_symbol_ordered_done[tid])=0;
            const bool cell_healthy = cell_health[cell_idx];
            if(!cell_healthy)
            {
                atomicCAS(exit_cond_d_cell, ORDER_KERNEL_RUNNING, ORDER_KERNEL_EXIT_PRB);
                if(pusch_prb_non_zero)
                {
                    for(uint32_t idx=0;idx<ORAN_PUSCH_SYMBOLS_X_SLOT;idx++)
                    {
                        atomicOr(&sym_ord_done_mask_arr[idx],cell_idx_mask);
                    }
                }
            }
        }
        DOCA_GPUNETIO_VOLATILE(done_shared_sh) = 1;

        early_rx_packets_count_sh=DOCA_GPUNETIO_VOLATILE(*next_slot_early_rx_packets_cell);
        on_time_rx_packets_count_sh=DOCA_GPUNETIO_VOLATILE(*next_slot_on_time_rx_packets_cell);
        late_rx_packets_count_sh=DOCA_GPUNETIO_VOLATILE(*next_slot_late_rx_packets_cell);
        next_slot_early_rx_packets_count_sh=0;
        next_slot_late_rx_packets_count_sh=0;
        next_slot_on_time_rx_packets_count_sh=0;
        rx_packets_dropped_count_sh=0;
        if constexpr (!ok_tb_enable)
        {
            if(pcap_capture_enable)
            {
                DOCA_GPUNETIO_VOLATILE(pcap_pkt_num_total) = DOCA_GPUNETIO_VOLATILE(*pcap_buffer_index_cell);
                //printf("BEGIN PCAP OK for Cell %d F%dS%dS%d : pcap_pkt_num_total=%d\n",blockIdx.x,frameId,subframeId,slotId,pcap_pkt_num_total);
            }
        }
    } else if(tid < ORAN_PUSCH_SYMBOLS_X_SLOT){
        if constexpr (srs_enable==ORDER_KERNEL_PUSCH_ONLY || srs_enable==ORDER_KERNEL_SRS_AND_PUSCH) {
            DOCA_GPUNETIO_VOLATILE(pusch_prb_symbol_ordered[tid])=0;
            DOCA_GPUNETIO_VOLATILE(pusch_prb_symbol_ordered_done[tid])=0;
        }
    }

    __syncthreads();

    uint64_t* order_kernel_last_timeout_error_time_cell=order_kernel_last_timeout_error_time[cell_idx];
    const uint16_t compressed_prb_size = (bit_width_cell == BFP_NO_COMPRESSION) ? PRB_SIZE_16F : (bit_width_cell == BFP_COMPRESSION_14_BITS) ? PRB_SIZE_14F : PRB_SIZE_9F;
    uint8_t first_packet_received = 0;
    unsigned long long first_packet_received_time = 0;

    uint32_t early_rx_packets_count = 0;
    uint32_t late_rx_packets_count = 0;
    uint32_t on_time_rx_packets_count = 0;
    uint32_t next_slot_early_rx_packets_count = 0;
    uint32_t next_slot_late_rx_packets_count = 0;
    uint32_t next_slot_on_time_rx_packets_count = 0;
    uint32_t packets_dropped_count = 0;

    // Only referenced by tid 0
    uint32_t rx_pkt_num_total{0};

    __shared__ bool sh_have_data_to_process;
    __shared__ uint32_t sh_next_pkt_ind;

    static constexpr int WARP_SIZE = 32;
    uint32_t warp_pusch_ordered_prbs_cell = 0;
    uint32_t warp_prach_ordered_prbs_cell = 0;
    uint32_t warp_srs_ordered_prbs_cell = 0;
    __shared__ uint32_t smem_pusch_prach_ordered_prbs_cell;
    __shared__ uint32_t smem_srs_ordered_prbs_cell;
    // Separate PUSCH/PRACH running accumulators, maintained purely so the timeout
    // log below can report the real in-loop ordered PRB count. The global
    // pusch_ordered_prbs_cell[0]/prach_ordered_prbs_cell[0] are only flushed after
    // the main while(1) loop (atomicAdd at end of kernel), so they always read 0
    // for a timed-out slot and are useless for triage.
    __shared__ uint32_t smem_pusch_ordered_prbs_cell;
    __shared__ uint32_t smem_prach_ordered_prbs_cell;
    if (tid == 0) {
        if constexpr (srs_enable==ORDER_KERNEL_SRS_ENABLE) {
            smem_srs_ordered_prbs_cell = *srs_ordered_prbs_cell;
        } else if constexpr (srs_enable==ORDER_KERNEL_PUSCH_ONLY) {
            smem_pusch_prach_ordered_prbs_cell = *pusch_ordered_prbs_cell + *prach_ordered_prbs_cell;
            smem_pusch_ordered_prbs_cell = *pusch_ordered_prbs_cell;
            smem_prach_ordered_prbs_cell = *prach_ordered_prbs_cell;
        } else {
            smem_srs_ordered_prbs_cell = *srs_ordered_prbs_cell;
            smem_pusch_prach_ordered_prbs_cell = *pusch_ordered_prbs_cell + *prach_ordered_prbs_cell;
            smem_pusch_ordered_prbs_cell = *pusch_ordered_prbs_cell;
            smem_prach_ordered_prbs_cell = *prach_ordered_prbs_cell;
        }
    }
    __syncthreads();

    if(ul_rx_pkt_tracing_level){
        if(tid<ORAN_MAX_SYMBOLS)
        {
            rx_packets_count_sh[tid]=DOCA_GPUNETIO_VOLATILE(pkt_tracing_info.next_slot_rx_packets_count[cell_idx][tid]);
            rx_bytes_count_sh[tid]=DOCA_GPUNETIO_VOLATILE(pkt_tracing_info.next_slot_rx_bytes_count[cell_idx][tid]);
            rx_packets_ts_earliest_sh[tid]=0xFFFFFFFFFFFFFFFFLLU;
            rx_packets_ts_latest_sh[tid]=0;
            next_slot_rx_packets_count_sh[tid]=0;
            next_slot_rx_bytes_count_sh[tid]=0;
        }
        __syncthreads();
        const int max_pkt_idx = ORDER_KERNEL_MAX_PKTS_PER_OFDM_SYM*ORAN_MAX_SYMBOLS;
        for(uint32_t pkt_idx=tid;pkt_idx<max_pkt_idx;pkt_idx+=blockDim.x)
        {
            uint32_t symbol_idx=pkt_idx/ORDER_KERNEL_MAX_PKTS_PER_OFDM_SYM;
            uint64_t* next_slot_rx_packets_ts_cell = *(pkt_tracing_info.next_slot_rx_packets_ts + cell_idx);            
            rx_packets_ts_sh[pkt_idx]=DOCA_GPUNETIO_VOLATILE(next_slot_rx_packets_ts_cell[pkt_idx]);
            __threadfence_block();
            if(rx_packets_ts_sh[pkt_idx]!=0)
                atomicMin((unsigned long long*) &rx_packets_ts_earliest_sh[symbol_idx],(unsigned long long) rx_packets_ts_sh[pkt_idx]);
            atomicMax((unsigned long long*) &rx_packets_ts_latest_sh[symbol_idx],(unsigned long long) rx_packets_ts_sh[pkt_idx]);
            __threadfence_block();
            next_slot_rx_packets_ts_sh[pkt_idx]=0;
        }
        __syncthreads();
        if(tid<ORAN_MAX_SYMBOLS)
        {
            if(rx_packets_ts_earliest_sh[tid]==0)
                rx_packets_ts_earliest_sh[tid]=0xFFFFFFFFFFFFFFFFLLU;
        }            
    }     

    // For the packets that are already available in semaphores before the first receive
    // call, we do not want to record packet stats because they were recoreded when the
    // packets were initially read.
    bool record_packet_stats = false;

    // Breaks when *exit_cond_d_cell != ORDER_KERNEL_RUNNING
    while (1) {

        // Breaks when no more packet buffers are ready to process. This loop may process buffers
        // stored in semaphores from a previous order kernel invocation.
        while (1) {
            uint32_t warp_pusch_prach_ordered_prbs_cell_this_burst = 0;
            uint32_t warp_srs_ordered_prbs_cell_this_burst = 0;
            // Per-burst PUSCH/PRACH split, folded into the log-only shared accumulators below.
            uint32_t warp_pusch_ordered_prbs_cell_this_burst = 0;
            uint32_t warp_prach_ordered_prbs_cell_this_burst = 0;

            if (tid == 0) {
                const doca_error_t ret = aerial_fh_gpu_dev_semaphore_get_packet_info_status(
                    sem_gpu_aerial_fh_cell, sem_idx_order, AERIAL_FH_GPU_SEMAPHORE_STATUS_READY, &rx_pkt_num, &rx_buf_idx);
                if constexpr (ok_tb_enable)
                {
                    rx_pkt_num = rx_pkt_num_slot[cell_idx];
                }
                sh_have_data_to_process = (ret != DOCA_ERROR_NOT_FOUND) && (rx_pkt_num > 0) &&
                    DOCA_GPUNETIO_VOLATILE(*exit_cond_d_cell) == ORDER_KERNEL_RUNNING;
                sh_next_pkt_ind = 0;
            }

            __syncthreads();

            if (! sh_have_data_to_process) {
                break;
            }

            // Breaks when no more packets are ready for processing in the current buffer.
            // Each warp handles one packet at a time, with packets dynamically scheduled to
            // warps using atomicAdd().
            while (1) {
                uint32_t pkt_idx{0};  // Initialize to prevent Coverity warning
                if (laneId == 0) {
                    pkt_idx = atomicAdd(&sh_next_pkt_ind, 1);
                }
                pkt_idx = __shfl_sync(0xffffffff, pkt_idx, 0, 32);
                if (pkt_idx >= rx_pkt_num) {
                    break;
                }
                uint8_t *pkt_thread = NULL;
                uint64_t rx_timestamp {0};
                if constexpr (ok_tb_enable)
                {
                    uint8_t* tb_fh_buf_cell=*(tb_fh_buf+cell_idx);
                    pkt_thread = (uint8_t*)(tb_fh_buf_cell+((pkt_idx)*max_pkt_size));
                }
                else
                {
                    pkt_thread = (uint8_t*)doca_gpu_dev_eth_rxq_get_pkt_addr(doca_rxq_cell, rx_buf_idx + pkt_idx);
                    rx_timestamp=doca_gpu_dev_eth_rxq_get_pkt_ts(doca_rxq_cell, rx_buf_idx + pkt_idx);
                    // PCAP Capture
                    if constexpr(!ok_tb_enable)
                    {
                        if(pcap_capture_enable && (pcap_capture_cell_bitmask & cell_idx_mask) != 0)
                        {
                            uint32_t offset{0};  // Initialize to prevent Coverity warning
                            if(laneId == 0)
                            {
                                offset = atomicAdd(&pcap_pkt_num_total, 1) % MAX_PKTS_PER_PCAP_BUFFER;
                            }
                            offset = __shfl_sync(0xffffffff, offset, 0, 32);

                            const uint16_t ecpri_payload_length_tmp = oran_umsg_get_ecpri_payload(pkt_thread);
                            uint8_t*       pkt_dst_buf              = pcap_buffer_cell + (offset * max_pkt_size);
                            const uint16_t pkt_size                 = (ORAN_ETH_HDR_SIZE + sizeof(oran_ecpri_hdr) + ecpri_payload_length_tmp);

                            for(uint32_t index_copy = laneId; index_copy < pkt_size; index_copy += WARP_SIZE)
                            {
                                pkt_dst_buf[index_copy] = pkt_thread[index_copy];
                            }

                            if(laneId == 0)
                            {
                                uint64_t* timestamp_ptr = reinterpret_cast<uint64_t*>(pcap_buffer_ts_cell + (offset * sizeof(uint64_t)));
                                *timestamp_ptr          = rx_timestamp;
                            }
                        }
                    }                    
                }

                uint8_t frameId_pkt     = oran_umsg_get_frame_id(pkt_thread);
                uint8_t subframeId_pkt  = oran_umsg_get_subframe_id(pkt_thread);
                uint8_t slotId_pkt      = oran_umsg_get_slot_id(pkt_thread);
                uint8_t symbol_id_pkt   = oran_umsg_get_symbol_id(pkt_thread);

                // Bounds check for symbol_id_pkt from untrusted packet data
                if(__builtin_expect(symbol_id_pkt >= ORAN_ALL_SYMBOLS, 0))
                {
                    printf("ERROR invalid symbol_id %d (max %d) for Cell %d F%dS%dS%d\n",
                           symbol_id_pkt,
                           ORAN_ALL_SYMBOLS - 1,
                           cell_idx,
                           frameId_pkt,
                           subframeId_pkt,
                           slotId_pkt);
                    continue; // Skip this packet
                }

                const uint8_t seq_id_pkt = oran_get_sequence_id(pkt_thread);
                const uint16_t ecpri_payload_length = oran_umsg_get_ecpri_payload(pkt_thread);
                int32_t full_slot_diff = calculate_slot_difference(frameId, frameId_pkt, subframeId, subframeId_pkt, slotId, slotId_pkt);

                if ( full_slot_diff > 0 ) //Only keep packets that are in the future slots
                {
                    if (record_packet_stats && laneId == 0) {
                        const uint64_t packet_early_thres = slot_start_cell+ slot_duration_cell + ta4_min_ns_cell + (slot_duration_cell * symbol_id_pkt / ORAN_MAX_SYMBOLS);
                        const uint64_t packet_late_thres  = slot_start_cell+ slot_duration_cell + ta4_max_ns_cell + (slot_duration_cell * symbol_id_pkt / ORAN_MAX_SYMBOLS);
                        if(rx_timestamp < packet_early_thres) next_slot_early_rx_packets_count++;
                        else if(rx_timestamp > packet_late_thres) next_slot_late_rx_packets_count++;
                        else next_slot_on_time_rx_packets_count++;
                        if constexpr (ul_rx_pkt_tracing_level)
                        {
                            uint32_t next_slot_rx_packets_ts_idx = atomicAdd(&next_slot_rx_packets_count_sh[symbol_id_pkt], 1);
                            atomicAdd(&next_slot_rx_bytes_count_sh[symbol_id_pkt], ORAN_ETH_HDR_SIZE + ecpri_payload_length);
                            __threadfence_block();
                            next_slot_rx_packets_ts_idx += ORDER_KERNEL_MAX_PKTS_PER_OFDM_SYM * symbol_id_pkt;
                            next_slot_rx_packets_ts_sh[next_slot_rx_packets_ts_idx] = rx_timestamp;
                        }
                    }
                    if (laneId == 0) {
                        atomicCAS(&done_shared_sh, 1, 0);
                    }
                } else if (full_slot_diff == 0) {
                    /* if this is the right slot, order & decompress */
                    uint8_t *section_buf = oran_umsg_get_first_section_buf(pkt_thread);
                    // 4 bytes for ecpriPcid, ecprSeqid, ecpriEbit, ecpriSubSeqid
                    uint16_t current_length = 4 + sizeof(oran_umsg_iq_hdr);
                    uint16_t num_sections = 0;
                    bool sanity_check = (current_length < ecpri_payload_length);
                    if(ecpri_hdr_sanity_check(pkt_thread) == false)
                    {
                        printf("ERROR malformatted eCPRI header... block %d thread %d\n", blockIdx.x, threadIdx.x);
                        //break;
                    }
                    const uint16_t startPRB_offset_idx_0 = 0;
                    const uint16_t startPRB_offset_idx_1 = 0;
                    const uint16_t startPRB_offset_idx_2 = 0;
                    const uint16_t startPRB_offset_idx_3 = 0;
                    while(current_length < ecpri_payload_length)
                    {
                        if(current_length + ORAN_IQ_UNCOMPRESSED_SECTION_OVERHEAD >= ecpri_payload_length)
                        {
                            sanity_check = false;
                            break;
                        }

                        uint16_t num_prb = oran_umsg_get_num_prb_from_section_buf(section_buf);
                        const uint16_t section_id = oran_umsg_get_section_id_from_section_buf(section_buf);
                        const uint16_t start_prb = oran_umsg_get_start_prb_from_section_buf(section_buf);
                        if(num_prb==0)
                            num_prb=ORAN_MAX_PRB_X_SLOT;
                        const uint16_t prb_buffer_size = compressed_prb_size * num_prb;

                        //WAR added for ru_type::SINGLE_SECT_MODE O-RU to pass. Will remove it when new FW is applied to fix the erronous ecpri payload length
                        if(ru_type_cell != SINGLE_SECT_MODE && current_length + prb_buffer_size + ORAN_IQ_UNCOMPRESSED_SECTION_OVERHEAD > ecpri_payload_length)
                        {
                            sanity_check = false;
                            break;
                        }
                        uint8_t *pkt_offset_ptr = section_buf + ORAN_IQ_UNCOMPRESSED_SECTION_OVERHEAD;
                        uint8_t *gbuf_offset_ptr;
                        uint8_t *gbuf_offset_ptr_srs = NULL;
                        uint8_t *buffer;
                        bool combined_srs_packet = false;

                        if constexpr (srs_enable==ORDER_KERNEL_SRS_ENABLE){
                            buffer = srs_buffer_cell;
                            gbuf_offset_ptr = buffer + oran_srs_get_offset_from_hdr(pkt_thread, (uint16_t)get_eaxc_index(srs_eAxC_map_cell, srs_eAxC_num_cell,
                                                                            oran_umsg_get_flowid(pkt_thread)),
                                                        srs_symbols_x_slot, srs_prb_x_port_x_symbol_cell, prb_size,start_prb,srs_start_sym_cell);
                        }
                        else
                        {
                            if(section_id < prach_section_id_0)
                            {
    
                                if(srs_enable==ORDER_KERNEL_PUSCH_ONLY) {
                                buffer = pusch_buffer_cell;
                                gbuf_offset_ptr = buffer + oran_get_offset_from_hdr(pkt_thread, (uint16_t)get_eaxc_index(pusch_eAxC_map_cell, pusch_eAxC_num_cell,
                                                                                oran_umsg_get_flowid(pkt_thread)),
                                                            pusch_symbols_x_slot, pusch_prb_x_port_x_symbol_cell, prb_size, start_prb);                                
                                } else if(srs_enable==ORDER_KERNEL_SRS_AND_PUSCH) {
                                    buffer = pusch_buffer_cell;
                                    gbuf_offset_ptr = buffer + oran_get_offset_from_hdr(pkt_thread,
                                                             (uint16_t)get_eaxc_index(pusch_eAxC_map_cell, pusch_eAxC_num_cell, oran_umsg_get_flowid(pkt_thread)),
                                                             pusch_symbols_x_slot, pusch_prb_x_port_x_symbol_cell, prb_size, start_prb);
                                    if(combined_srs_active_for_cell &&
                                       ul_oran_srs_symbol_in_window(symbol_id_pkt, srs_start_sym_cell, srs_symbols_x_slot)) {
                                        combined_srs_packet = true;
                                        gbuf_offset_ptr_srs = srs_buffer_cell + oran_srs_get_offset_from_hdr(pkt_thread,
                                                                    (uint16_t)get_eaxc_index(srs_eAxC_map_cell, srs_eAxC_num_cell, oran_umsg_get_flowid(pkt_thread)),
                                                                    srs_symbols_x_slot, srs_prb_x_port_x_symbol_cell, prb_size, start_prb, srs_start_sym_cell);

                                    }
                                }
                            }
                            else {
                                if(section_id == prach_section_id_0) buffer = prach_buffer_0_cell;
                                else if(section_id == prach_section_id_1) buffer = prach_buffer_1_cell;
                                else if(section_id == prach_section_id_2) buffer = prach_buffer_2_cell;
                                else if(section_id == prach_section_id_3) buffer = prach_buffer_3_cell;
                                else {
                                    // Invalid section_id - skip this section
                                    printf("ERROR invalid section_id %d for Cell %d F%dS%dS%d\n", section_id, cell_idx, frameId_pkt, subframeId_pkt, slotId_pkt);
                                    break;
                                }
                                gbuf_offset_ptr = buffer + oran_get_offset_from_hdr(pkt_thread, (uint16_t)get_eaxc_index(prach_eAxC_map_cell, prach_eAxC_num_cell,
                                                                                oran_umsg_get_flowid(pkt_thread)),
                                                            prach_symbols_x_slot, prach_prb_x_port_x_symbol_cell, prb_size, start_prb);
    
                                /* prach_buffer_x_cell is populated based on number of PRACH PDU's, hence the index can be used as "Frequency domain occasion index"
                                    and mutiplying with num_prb i.e. NRARB=12 (NumRB's (PRACH SCS=30kHz) for each FDM ocassion) will yeild the corrosponding PRB start for each Frequency domain index
                                    Note: WIP for a more generic approach to caluclate and pass the startRB from the cuPHY-CP */
                                if(section_id == prach_section_id_0) gbuf_offset_ptr -= startPRB_offset_idx_0;
                                else if(section_id == prach_section_id_1) gbuf_offset_ptr -= startPRB_offset_idx_1;
                                else if(section_id == prach_section_id_2) gbuf_offset_ptr -= startPRB_offset_idx_2;
                                else if(section_id == prach_section_id_3) gbuf_offset_ptr -= startPRB_offset_idx_3;
                            }
                        }                        

                        // CWE-787: drop copy/decompress if untrusted symbolId/startPrb/numPrb
                        // would index past the per-eAxC antenna buffer (PUSCH/PRACH/SRS).
                        {
                            bool ob;
                            if constexpr (srs_enable==ORDER_KERNEL_SRS_ENABLE)
                                ob = !ul_oran_srs_offset_in_bounds(symbol_id_pkt, start_prb, num_prb, srs_symbols_x_slot, srs_prb_x_port_x_symbol_cell, srs_start_sym_cell);
                            else if(section_id < prach_section_id_0)
                            {
                                ob = !ul_oran_offset_in_bounds(symbol_id_pkt, start_prb, num_prb, pusch_symbols_x_slot, pusch_prb_x_port_x_symbol_cell);
                                if(combined_srs_packet)
                                    ob = ob || !ul_oran_srs_offset_in_bounds(symbol_id_pkt, start_prb, num_prb, srs_symbols_x_slot, srs_prb_x_port_x_symbol_cell, srs_start_sym_cell);
                            }
                            else
                                ob = !ul_oran_offset_in_bounds(symbol_id_pkt, start_prb, num_prb, prach_symbols_x_slot, prach_prb_x_port_x_symbol_cell);
                            if(ob)
                                num_prb = 0;
                        }
                        if(comp_meth_cell == static_cast<uint8_t>(aerial_fh::UserDataCompressionMethod::BLOCK_FLOATING_POINT))
                        {
                            if(bit_width_cell == BFP_NO_COMPRESSION) // BFP with 16 bits is a special case and uses FP16, so copy the values
                            {
                                for(int index_copy = laneId; index_copy < (num_prb * prb_size); index_copy += 32)
                                    gbuf_offset_ptr[index_copy] = pkt_offset_ptr[index_copy];

                                if(combined_srs_packet) {
                                    for(int index_copy = laneId; index_copy < (num_prb * prb_size); index_copy += 32)
                                        gbuf_offset_ptr_srs[index_copy] = pkt_offset_ptr[index_copy];
                                }
                            }
                            else
                            {
                                decompress_scale_blockFP<false>((unsigned char*)pkt_offset_ptr, (__half*)gbuf_offset_ptr, beta_cell, num_prb, bit_width_cell, (int)(threadIdx.x & 31), 32);

                                if(combined_srs_packet) {
                                    decompress_scale_blockFP<false>((unsigned char*)pkt_offset_ptr, (__half*)gbuf_offset_ptr_srs, beta_cell, num_prb, bit_width_cell, (int)(threadIdx.x & 31), 32);
                                }
                            }
                        } else // aerial_fh::UserDataCompressionMethod::NO_COMPRESSION
                        {
                            decompress_scale_fixed<false>(pkt_offset_ptr, (__half*)gbuf_offset_ptr, beta_cell, num_prb, bit_width_cell, (int)(threadIdx.x & 31), 32);

                            if(combined_srs_packet) {
                                decompress_scale_fixed<false>(pkt_offset_ptr, (__half*)gbuf_offset_ptr_srs, beta_cell, num_prb, bit_width_cell, (int)(threadIdx.x & 31), 32);
                            }
                        }
                        if (laneId == 0) {
                            if constexpr (srs_enable==ORDER_KERNEL_SRS_ENABLE)
                            {
                                warp_srs_ordered_prbs_cell += num_prb;
                                warp_srs_ordered_prbs_cell_this_burst += num_prb;
                            }
                            else
                            {
                                if(section_id < prach_section_id_0) {
                                    uint32_t tot_pusch_prb_symbol_ordered = atomicAdd(&pusch_prb_symbol_ordered[symbol_id_pkt], num_prb);
                                    tot_pusch_prb_symbol_ordered += num_prb;
                                    if(pusch_prb_non_zero && tot_pusch_prb_symbol_ordered >= pusch_prb_symbol_map_cell[symbol_id_pkt] && DOCA_GPUNETIO_VOLATILE(pusch_prb_symbol_ordered_done[symbol_id_pkt])==0){
                                        DOCA_GPUNETIO_VOLATILE(pusch_prb_symbol_ordered_done[symbol_id_pkt])=1;
                                        atomicOr(&sym_ord_done_mask_arr[symbol_id_pkt],cell_idx_mask);
                                        if(DOCA_GPUNETIO_VOLATILE(sym_ord_done_mask_arr[symbol_id_pkt])==num_order_cells_sym_mask_arr[symbol_id_pkt] && DOCA_GPUNETIO_VOLATILE(sym_ord_done_sig_arr[symbol_id_pkt])==(uint32_t)SYM_RX_NOT_DONE){
                                            DOCA_GPUNETIO_VOLATILE(sym_ord_done_sig_arr[symbol_id_pkt])=(uint32_t)SYM_RX_DONE;
                                        }
                                    }
                                    warp_pusch_ordered_prbs_cell += num_prb;
                                    warp_pusch_ordered_prbs_cell_this_burst += num_prb;
                                    if(combined_srs_packet) {
                                        warp_srs_ordered_prbs_cell += num_prb;
                                    }
                                } else {
                                    warp_prach_ordered_prbs_cell += num_prb;
                                    warp_prach_ordered_prbs_cell_this_burst += num_prb;
                                }
                                warp_pusch_prach_ordered_prbs_cell_this_burst += num_prb;
                            }
                        }
                        
                        current_length += prb_buffer_size + ORAN_IQ_UNCOMPRESSED_SECTION_OVERHEAD;
                        section_buf = pkt_offset_ptr + prb_buffer_size;
                        ++num_sections;
                        if(num_sections > ORAN_MAX_PRB_X_SLOT)
                        {
                            printf("Invalid U-Plane packet, num_sections %d > 273 for Cell %d F%dS%dS%d\n", num_sections, blockIdx.x, frameId_pkt, subframeId_pkt, slotId_pkt);
                            break;
                        }
                        
                    }
                    if (record_packet_stats && laneId == 0) {
                        const uint64_t packet_early_thres = slot_start_cell + ta4_min_ns_cell + (slot_duration_cell * symbol_id_pkt / ORAN_MAX_SYMBOLS);
                        const uint64_t packet_late_thres  = slot_start_cell + ta4_max_ns_cell + (slot_duration_cell * symbol_id_pkt / ORAN_MAX_SYMBOLS);
                        if (rx_timestamp < packet_early_thres) early_rx_packets_count++;
                        else if (rx_timestamp > packet_late_thres) late_rx_packets_count++;
                        else on_time_rx_packets_count++;
                        if constexpr (ul_rx_pkt_tracing_level) {
                            uint32_t rx_packets_ts_idx = atomicAdd(&rx_packets_count_sh[symbol_id_pkt], 1);
                            atomicAdd(&rx_bytes_count_sh[symbol_id_pkt], ORAN_ETH_HDR_SIZE + ecpri_payload_length);
                            __threadfence_block();
                            rx_packets_ts_idx+=ORDER_KERNEL_MAX_PKTS_PER_OFDM_SYM*symbol_id_pkt;
                            rx_packets_ts_sh[rx_packets_ts_idx]=rx_timestamp;
                            atomicMin((unsigned long long*) &rx_packets_ts_earliest_sh[symbol_id_pkt],(unsigned long long) rx_timestamp);
                            atomicMax((unsigned long long*) &rx_packets_ts_latest_sh[symbol_id_pkt],(unsigned long long) rx_timestamp);
                            __threadfence_block();
                        }
                    }
                    if(!sanity_check)
                    {
                        printf("ERROR uplane pkt sanity check failed, it could be erroneous BFP, numPrb or ecpri payload len, or other reasons... block %d thread %d\n", blockIdx.x, threadIdx.x);
                        atomicCAS(exit_cond_d_cell, ORDER_KERNEL_RUNNING, ORDER_KERNEL_EXIT_ERROR7);
                        break;
                    }
                }
                else //Drop packets that are in the past slots
                {
                    if(record_packet_stats && laneId == 0){
                        packets_dropped_count++;
                    }
                }
            }

            if (laneId == 0) {
                uint32_t old_prb_count, num_prb_added;
                if constexpr (srs_enable==ORDER_KERNEL_SRS_ENABLE) {
                    num_prb_added = warp_srs_ordered_prbs_cell_this_burst;
                    old_prb_count = atomicAdd(&smem_srs_ordered_prbs_cell, num_prb_added);
                }
                else {
                    num_prb_added = warp_pusch_prach_ordered_prbs_cell_this_burst;
                    old_prb_count = atomicAdd(&smem_pusch_prach_ordered_prbs_cell, num_prb_added);
                    // Keep the log-only split accumulators in sync (does not affect the exit check).
                    atomicAdd(&smem_pusch_ordered_prbs_cell, warp_pusch_ordered_prbs_cell_this_burst);
                    atomicAdd(&smem_prach_ordered_prbs_cell, warp_prach_ordered_prbs_cell_this_burst);
                }

                if(old_prb_count < prb_x_slot && old_prb_count + num_prb_added >= prb_x_slot) {
                    atomicCAS(exit_cond_d_cell, ORDER_KERNEL_RUNNING, ORDER_KERNEL_EXIT_PRB);
                }
            }
            __syncthreads();

            if(tid == 0 && DOCA_GPUNETIO_VOLATILE(done_shared_sh) == 1) {
                aerial_fh_gpu_dev_semaphore_set_status(sem_gpu_aerial_fh_cell, last_sem_idx_order, AERIAL_FH_GPU_SEMAPHORE_STATUS_DONE);
                last_sem_idx_order = (last_sem_idx_order + 1) & (sem_order_num_cell - 1);
            }

            sem_idx_order = (sem_idx_order+1) & (sem_order_num_cell - 1);
        }
        if (tid == 0) {
            const unsigned long long current_time = __globaltimer();
            if (first_packet_received && ((current_time - first_packet_received_time) > timeout_first_pkt_ns)) {
                if constexpr (srs_enable==ORDER_KERNEL_SRS_ENABLE) {
                    if(timeout_log_enable){
                        if((current_time-DOCA_GPUNETIO_VOLATILE(*order_kernel_last_timeout_error_time_cell))>timeout_log_interval_ns){
                            printf("%d Cell %d Order kernel sem_idx_rx %d last_sem_idx_rx %d sem_idx_order %d last_sem_idx_order %d SRS PRBs %d/%d.  First packet received timeout after %d ns F%dS%dS%d done = %d current_time=%llu,last_timeout_log_time=%llu,total_rx_pkts=%d\n",__LINE__,
                                cell_idx, sem_idx_rx, *last_sem_idx_rx_h_cell,
                                sem_idx_order,*last_sem_idx_order_h_cell,
                                smem_srs_ordered_prbs_cell, srs_prb_x_slot_cell,
                                timeout_first_pkt_ns, frameId, subframeId, slotId,
                                DOCA_GPUNETIO_VOLATILE(done_shared_sh),
                                current_time,DOCA_GPUNETIO_VOLATILE(*order_kernel_last_timeout_error_time_cell),rx_pkt_num_total);
                                DOCA_GPUNETIO_VOLATILE(*order_kernel_last_timeout_error_time_cell)=current_time;
                        }
                    }                    
                }
                else
                {
                    if(timeout_log_enable){
                        if((current_time-DOCA_GPUNETIO_VOLATILE(*order_kernel_last_timeout_error_time_cell))>timeout_log_interval_ns){
                            printf("%d Cell %d Order kernel sem_idx_rx %d last_sem_idx_rx %d sem_idx_order %d last_sem_idx_order %d PUSCH PRBs %d/%d PRACH PRBs %d/%d.  First packet received timeout after %d ns F%dS%dS%d done = %d current_time=%llu,last_timeout_log_time=%llu,total_rx_pkts=%d\n",__LINE__,
                                cell_idx, sem_idx_rx, *last_sem_idx_rx_h_cell,
                                sem_idx_order,*last_sem_idx_order_h_cell,
                                smem_pusch_ordered_prbs_cell, pusch_prb_x_slot_cell,
                                smem_prach_ordered_prbs_cell, prach_prb_x_slot_cell,
                                timeout_first_pkt_ns, frameId, subframeId, slotId,
                                DOCA_GPUNETIO_VOLATILE(done_shared_sh),
                                current_time,DOCA_GPUNETIO_VOLATILE(*order_kernel_last_timeout_error_time_cell),rx_pkt_num_total);
                                DOCA_GPUNETIO_VOLATILE(*order_kernel_last_timeout_error_time_cell)=current_time;
                        }
                    }
                    if(pusch_prb_non_zero)
                    {
                        for(uint32_t idx=0;idx<ORAN_PUSCH_SYMBOLS_X_SLOT;idx++)
                        {
                            DOCA_GPUNETIO_VOLATILE(sym_ord_done_sig_arr[idx])=(uint32_t)SYM_RX_TIMEOUT;
                        }
                    }
                }
                atomicCAS(exit_cond_d_cell, ORDER_KERNEL_RUNNING, ORDER_KERNEL_EXIT_TIMEOUT_RX_PKT);
            } else if ((!first_packet_received) && ((current_time - kernel_start) > timeout_no_pkt_ns)) {
                if constexpr (srs_enable==ORDER_KERNEL_SRS_ENABLE) {
                    if(timeout_log_enable){
                        if((current_time-DOCA_GPUNETIO_VOLATILE(*order_kernel_last_timeout_error_time_cell))>timeout_log_interval_ns){
                        printf("%d Cell %d Order kernel sem_idx_rx %d last_sem_idx_rx %d sem_idx_order %d last_sem_idx_order %d SRS PRBs %d/%d. No packet received timeout after %d ns F%dS%dS%d done = %d current_time=%llu,last_timeout_log_time=%llu\n",__LINE__,
                            cell_idx, sem_idx_rx, *last_sem_idx_rx_h_cell,
                            sem_idx_order,*last_sem_idx_order_h_cell,
                            smem_srs_ordered_prbs_cell, srs_prb_x_slot_cell,
                            timeout_no_pkt_ns, frameId, subframeId, slotId,
                            DOCA_GPUNETIO_VOLATILE(done_shared_sh),
                            current_time,DOCA_GPUNETIO_VOLATILE(*order_kernel_last_timeout_error_time_cell));
                            DOCA_GPUNETIO_VOLATILE(*order_kernel_last_timeout_error_time_cell)=current_time;
                        }
                    }                    
                }
                else
                {
                    if(timeout_log_enable){
                        if((current_time-DOCA_GPUNETIO_VOLATILE(*order_kernel_last_timeout_error_time_cell))>timeout_log_interval_ns){
                        printf("%d Cell %d Order kernel sem_idx_rx %d last_sem_idx_rx %d sem_idx_order %d last_sem_idx_order %d PUSCH PRBs %d/%d PRACH PRBs %d/%d. No packet received timeout after %d ns F%dS%dS%d done = %d current_time=%llu,last_timeout_log_time=%llu\n",__LINE__,
                            cell_idx, sem_idx_rx, *last_sem_idx_rx_h_cell,
                            sem_idx_order,*last_sem_idx_order_h_cell,
                            smem_pusch_ordered_prbs_cell, pusch_prb_x_slot_cell,
                            smem_prach_ordered_prbs_cell, prach_prb_x_slot_cell,
                            timeout_no_pkt_ns, frameId, subframeId, slotId,
                            DOCA_GPUNETIO_VOLATILE(done_shared_sh),
                            current_time,DOCA_GPUNETIO_VOLATILE(*order_kernel_last_timeout_error_time_cell));
                            DOCA_GPUNETIO_VOLATILE(*order_kernel_last_timeout_error_time_cell)=current_time;
                        }
                    }
                    if(pusch_prb_non_zero)
                    {
                        for(uint32_t idx=0;idx<ORAN_PUSCH_SYMBOLS_X_SLOT;idx++)
                        {
                            DOCA_GPUNETIO_VOLATILE(sym_ord_done_sig_arr[idx])=(uint32_t)SYM_RX_TIMEOUT;
                        }
                    }
                }
                atomicCAS(exit_cond_d_cell, ORDER_KERNEL_RUNNING, ORDER_KERNEL_EXIT_TIMEOUT_NO_PKT);
            }
            DOCA_GPUNETIO_VOLATILE(rx_pkt_num) = 0;
        }

        __syncthreads();

        if (DOCA_GPUNETIO_VOLATILE(*exit_cond_d_cell) != ORDER_KERNEL_RUNNING)
            break;

        __syncthreads();

        {
            doca_error_t ret = doca_gpu_dev_eth_rxq_recv<DOCA_GPUNETIO_ETH_EXEC_SCOPE_BLOCK,DOCA_GPUNETIO_ETH_MCST_DISABLED,DOCA_GPUNETIO_ETH_NIC_HANDLER_AUTO,DOCA_GPUNETIO_ETH_RX_ATTR_NONE>(doca_rxq_cell,max_rx_pkts_,rx_pkts_timeout_ns,&rx_buf_idx,&rx_pkt_num,nullptr);
            /* If any thread returns receive error, the whole execution stops */
            if (ret != DOCA_SUCCESS) {
                aerial_fh_gpu_dev_semaphore_set_status(sem_gpu_aerial_fh_cell, sem_idx_rx, AERIAL_FH_GPU_SEMAPHORE_STATUS_ERROR);
                atomicCAS(exit_cond_d_cell, ORDER_KERNEL_RUNNING, ORDER_KERNEL_EXIT_ERROR1);
                printf("Exit from rx kernel block %d threadIdx %d ret %d sem_idx_rx %d\n",
                    blockIdx.x, threadIdx.x, ret, sem_idx_rx);
            } else {
                if (rx_pkt_num > 0) {
                    if (tid == 0){
                        aerial_fh_gpu_dev_semaphore_set_packet_info(sem_gpu_aerial_fh_cell, sem_idx_rx, AERIAL_FH_GPU_SEMAPHORE_STATUS_READY, rx_pkt_num, rx_buf_idx);
                        rx_pkt_num_total += rx_pkt_num;
                        if (first_packet_received == 0) {
                            first_packet_received = 1;
                            first_packet_received_time  = __globaltimer();
                        }                        
                    }
                }
            }
        }

        __syncthreads();

        if (DOCA_GPUNETIO_VOLATILE(rx_pkt_num) == 0) {
            continue;
        }

        // For newly read packets, we want to record packet stats the first time
        // we process them.
        record_packet_stats = true;
        sem_idx_rx = (sem_idx_rx+1) & (sem_order_num_cell - 1);

        __syncthreads();
    }

    // Only laneId == 0 threads should have non-zero packet stat counts
    if (early_rx_packets_count) atomicAdd(&early_rx_packets_count_sh, early_rx_packets_count);
    if (late_rx_packets_count) atomicAdd(&late_rx_packets_count_sh, late_rx_packets_count);
    if (on_time_rx_packets_count) atomicAdd(&on_time_rx_packets_count_sh, on_time_rx_packets_count);
    if (next_slot_early_rx_packets_count) atomicAdd(&next_slot_early_rx_packets_count_sh, next_slot_early_rx_packets_count);
    if (next_slot_late_rx_packets_count) atomicAdd(&next_slot_late_rx_packets_count_sh, next_slot_late_rx_packets_count);
    if (next_slot_on_time_rx_packets_count) atomicAdd(&next_slot_on_time_rx_packets_count_sh, next_slot_on_time_rx_packets_count);
    if (packets_dropped_count) atomicAdd(&rx_packets_dropped_count_sh, packets_dropped_count);
    __syncthreads();

    if constexpr (ul_rx_pkt_tracing_level) {
        __syncthreads();

        if (tid < ORAN_MAX_SYMBOLS) {
            pkt_tracing_info.rx_packets_count[cell_idx][tid] = rx_packets_count_sh[tid];
            pkt_tracing_info.rx_bytes_count[cell_idx][tid] = rx_bytes_count_sh[tid];
            pkt_tracing_info.next_slot_rx_packets_count[cell_idx][tid] = next_slot_rx_packets_count_sh[tid];
            pkt_tracing_info.next_slot_rx_bytes_count[cell_idx][tid] = next_slot_rx_bytes_count_sh[tid];
            pkt_tracing_info.rx_packets_ts_earliest[cell_idx][tid] = rx_packets_ts_earliest_sh[tid];
            pkt_tracing_info.rx_packets_ts_latest[cell_idx][tid] = rx_packets_ts_latest_sh[tid];
        }

        __syncthreads();
        const int max_pkt_idx = ORDER_KERNEL_MAX_PKTS_PER_OFDM_SYM*ORAN_MAX_SYMBOLS;
        uint64_t* rx_packets_ts_cell = pkt_tracing_info.rx_packets_ts[cell_idx];
        uint64_t* next_slot_rx_packets_ts_cell = pkt_tracing_info.next_slot_rx_packets_ts[cell_idx];
        for(uint32_t pkt_idx=tid;pkt_idx<max_pkt_idx;pkt_idx+=blockDim.x)
        {
            DOCA_GPUNETIO_VOLATILE(rx_packets_ts_cell[pkt_idx])=rx_packets_ts_sh[pkt_idx];
            DOCA_GPUNETIO_VOLATILE(next_slot_rx_packets_ts_cell[pkt_idx])=next_slot_rx_packets_ts_sh[pkt_idx];
        }
        __syncthreads();
    }

    if (laneId == 0) {
        if constexpr (srs_enable==ORDER_KERNEL_SRS_ENABLE) {
            atomicAdd(srs_ordered_prbs_cell, warp_srs_ordered_prbs_cell);
        }
        else {
            atomicAdd(pusch_ordered_prbs_cell, warp_pusch_ordered_prbs_cell);
            atomicAdd(prach_ordered_prbs_cell, warp_prach_ordered_prbs_cell);
            if(srs_enable==ORDER_KERNEL_SRS_AND_PUSCH) {
                atomicAdd(srs_ordered_prbs_cell, warp_srs_ordered_prbs_cell);
            }
        }
    }

    if (tid == 0) {
        DOCA_GPUNETIO_VOLATILE(*last_sem_idx_rx_h_cell) = sem_idx_rx;
        DOCA_GPUNETIO_VOLATILE(*early_rx_packets_cell) = early_rx_packets_count_sh;
        DOCA_GPUNETIO_VOLATILE(*on_time_rx_packets_cell) = on_time_rx_packets_count_sh;
        DOCA_GPUNETIO_VOLATILE(*late_rx_packets_cell) = late_rx_packets_count_sh;
        DOCA_GPUNETIO_VOLATILE(*next_slot_early_rx_packets_cell)=next_slot_early_rx_packets_count_sh;
        DOCA_GPUNETIO_VOLATILE(*next_slot_on_time_rx_packets_cell)=next_slot_on_time_rx_packets_count_sh;
        DOCA_GPUNETIO_VOLATILE(*next_slot_late_rx_packets_cell)=next_slot_late_rx_packets_count_sh;
        DOCA_GPUNETIO_VOLATILE(*rx_packets_dropped_count_cell)=rx_packets_dropped_count_sh;
        if constexpr (!ok_tb_enable)
        {
            if(pcap_capture_enable)
            {
                DOCA_GPUNETIO_VOLATILE(*pcap_buffer_index_cell) = pcap_pkt_num_total % MAX_PKTS_PER_PCAP_BUFFER;
                //printf("END PCAP OK for Cell %d F%dS%dS%d : pcap_pkt_num_total=%d\n",blockIdx.x,frameId,subframeId,slotId,pcap_pkt_num_total);
            }
        }
        DOCA_GPUNETIO_VOLATILE(*last_sem_idx_order_h_cell) = last_sem_idx_order;
        uint32_t *start_cuphy_d_cell=*(start_cuphy_d+cell_idx);
        DOCA_GPUNETIO_VOLATILE(*start_cuphy_d_cell) = 1;
    }
}

#ifdef __cplusplus
extern "C" {
#endif

__global__ void order_kernel_cpu_init_comms_single_subSlot(
uint32_t**             start_cuphy_d,
uint32_t**             order_kernel_exit_cond_d,
uint32_t**             ready_list,
struct aerial_fh::rx_queue_sync** rx_queue_sync_list,
uint32_t**                  last_ordered_item_h,
const uint16_t* sem_order_num,

uint8_t               frameId,
uint8_t               subframeId,
uint8_t               slotId,
int*                   comp_meth,
int*                   bit_width,
int                   prb_size,
float*                 beta,
int*                  barrier_flag,
uint8_t**              done_shared,

uint32_t              timeout_no_pkt_ns,
uint32_t              timeout_first_pkt_ns,

uint32_t*              sym_ord_done_sig_arr,
uint32_t*              sym_ord_done_mask_arr,
uint32_t*              pusch_prb_symbol_map,
uint32_t*              num_order_cells_sym_mask_arr,

/* Timer */
uint32_t 		**early_rx_packets,
uint32_t 		**on_time_rx_packets,
uint32_t 		**late_rx_packets,
uint32_t 		**next_slot_early_rx_packets,
uint32_t 		**next_slot_on_time_rx_packets,
uint32_t 		**next_slot_late_rx_packets,	
uint64_t*		slot_start,
uint64_t*		ta4_min_ns,
uint64_t*		ta4_max_ns,
uint64_t*		slot_duration,
uint8_t                ul_rx_pkt_tracing_level,
uint64_t**             rx_packets_ts,
uint32_t**              rx_packets_count,
uint32_t**              rx_bytes_count,
uint64_t**             rx_packets_ts_earliest,
uint64_t**             rx_packets_ts_latest,	
uint64_t**             next_slot_rx_packets_ts,
uint32_t**             next_slot_rx_packets_count,
uint32_t**             next_slot_rx_bytes_count,

uint16_t        **pusch_eAxC_map,
int*            pusch_eAxC_num,
uint8_t     **pusch_buffer,
int*            pusch_prb_x_slot,
int*            pusch_prb_x_symbol,
int*            pusch_prb_x_symbol_x_antenna,
int         pusch_symbols_x_slot,
uint32_t*           pusch_prb_x_port_x_symbol,
uint32_t        **pusch_ordered_prbs,

uint16_t    **prach_eAxC_map,
int*        prach_eAxC_num,
uint8_t     **prach_buffer_0,
uint8_t     **prach_buffer_1,
uint8_t     **prach_buffer_2,
uint8_t     **prach_buffer_3,
uint16_t    prach_section_id_0,
uint16_t    prach_section_id_1,
uint16_t    prach_section_id_2,
uint16_t    prach_section_id_3,
int*        prach_prb_x_slot,
int*        prach_prb_x_symbol,
int*        prach_prb_x_symbol_x_antenna,
int         prach_symbols_x_slot,
uint32_t*   prach_prb_x_port_x_symbol,
uint32_t    **prach_ordered_prbs,
uint8_t num_order_cells
)    
{
	int laneId = threadIdx.x % 32;
	int warpId = threadIdx.x / 32;
	int nwarps = blockDim.x / 32;
	int cell_idx = blockIdx.x;
    uint32_t cell_idx_mask = (0x1<<cell_idx);

	unsigned long long first_packet_start = 0;
	unsigned long long current_time = 0;
	unsigned long long kernel_start = __globaltimer();
	uint8_t* pkt_offset_ptr, *gbuf_offset_ptr;
	uint8_t* buffer;
	int prb_x_slot=pusch_prb_x_slot[cell_idx]+prach_prb_x_slot[cell_idx];
	int ret = 0;
    
	// Restart from last semaphore item
	int sem_idx_order = -1;

	__shared__ uint32_t rx_pkt_num;
    __shared__ uint32_t rx_pkt_num_total;    
	__shared__ uint32_t done_shared_sh;
    __shared__ uint32_t ready_sync_sh;
	__shared__ uint32_t num_prb_ch1_sh;
	__shared__ uint32_t num_prb_ch2_sh;
	__shared__ uint32_t exit_rx_timeout_sh;
    __shared__ uint32_t exit_ord_done_sh;
    __shared__ int      sem_idx_order_sh;
	uint64_t rx_timestamp;
    __shared__ uint32_t early_rx_packets_count_sh;
    __shared__ uint32_t on_time_rx_packets_count_sh;
    __shared__ uint32_t late_rx_packets_count_sh;
    __shared__ uint32_t next_slot_early_rx_packets_count_sh;
    __shared__ uint32_t next_slot_on_time_rx_packets_count_sh;
    __shared__ uint32_t next_slot_late_rx_packets_count_sh;       
    __shared__ uint64_t rx_packets_ts_sh[ORDER_KERNEL_MAX_PKTS_PER_OFDM_SYM*ORAN_MAX_SYMBOLS];
    __shared__ uint32_t rx_packets_count_sh[ORAN_MAX_SYMBOLS];     
    __shared__ uint32_t rx_bytes_count_sh[ORAN_MAX_SYMBOLS];     
    __shared__ uint64_t rx_packets_ts_earliest_sh[ORAN_MAX_SYMBOLS];
    __shared__ uint64_t rx_packets_ts_latest_sh[ORAN_MAX_SYMBOLS];
    __shared__ uint64_t next_slot_rx_packets_ts_sh[ORDER_KERNEL_MAX_PKTS_PER_OFDM_SYM*ORAN_MAX_SYMBOLS];
    __shared__ uint32_t next_slot_rx_packets_count_sh[ORAN_MAX_SYMBOLS];
    __shared__ uint32_t next_slot_rx_bytes_count_sh[ORAN_MAX_SYMBOLS];

	__shared__ uint32_t pusch_prb_symbol_ordered[ORAN_PUSCH_SYMBOLS_X_SLOT];
	__shared__ uint32_t pusch_prb_symbol_ordered_done[ORAN_PUSCH_SYMBOLS_X_SLOT];
	uintptr_t rx_pkt_addr;

	//Cell specific (de-reference from host pinned memory once)
	uint8_t* done_shared_cell=*(done_shared+cell_idx);
	uint32_t* pusch_ordered_prbs_cell=*(pusch_ordered_prbs+cell_idx);
	uint32_t* prach_ordered_prbs_cell=*(prach_ordered_prbs+cell_idx);
	uint32_t* exit_cond_d_cell=*(order_kernel_exit_cond_d+cell_idx);
	uint32_t* last_sem_idx_order_h_cell=(uint32_t*)(*(last_ordered_item_h+cell_idx));
    int last_sem_idx_order = (int)(*last_sem_idx_order_h_cell);

	uint8_t			*pusch_buffer_cell=*(pusch_buffer+cell_idx);
	uint16_t		*pusch_eAxC_map_cell=*(pusch_eAxC_map+cell_idx);
	int			    pusch_eAxC_num_cell=pusch_eAxC_num[cell_idx];
	uint32_t		pusch_prb_x_port_x_symbol_cell=	pusch_prb_x_port_x_symbol[cell_idx];
	uint8_t			*prach_buffer_0_cell=*(prach_buffer_0+cell_idx);
	uint8_t			*prach_buffer_1_cell=*(prach_buffer_1+cell_idx);
	uint8_t			*prach_buffer_2_cell=*(prach_buffer_2+cell_idx);
	uint8_t			*prach_buffer_3_cell=*(prach_buffer_3+cell_idx);
	uint16_t		*prach_eAxC_map_cell=*(prach_eAxC_map+cell_idx);
	int			    prach_eAxC_num_cell=prach_eAxC_num[cell_idx];
	uint32_t		prach_prb_x_port_x_symbol_cell=	prach_prb_x_port_x_symbol[cell_idx];
	const int		comp_meth_cell=comp_meth[cell_idx];
    const int		bit_width_cell=bit_width[cell_idx];
	const float		beta_cell=beta[cell_idx];
	uint32_t		*start_cuphy_d_cell=*(start_cuphy_d+cell_idx);
	const uint16_t sem_order_num_cell=sem_order_num[cell_idx];
    struct aerial_fh::rx_queue_sync* rx_queue_sync_list_cell=rx_queue_sync_list[cell_idx];
    uint32_t* ready_list_cell = ready_list[cell_idx]; 

    uint32_t* pusch_prb_symbol_map_cell = pusch_prb_symbol_map+(ORAN_PUSCH_SYMBOLS_X_SLOT*cell_idx);

	uint64_t		slot_start_cell=slot_start[cell_idx];
	uint64_t		ta4_min_ns_cell=ta4_min_ns[cell_idx];
	uint64_t		ta4_max_ns_cell=ta4_max_ns[cell_idx];
	uint64_t		slot_duration_cell=slot_duration[cell_idx];
	uint32_t 		*early_rx_packets_cell= *(early_rx_packets+cell_idx);
	uint32_t 		*on_time_rx_packets_cell= *(on_time_rx_packets+cell_idx);
	uint32_t 		*late_rx_packets_cell= *(late_rx_packets+cell_idx);
	uint32_t 		*next_slot_early_rx_packets_cell= *(next_slot_early_rx_packets+cell_idx);
	uint32_t 		*next_slot_on_time_rx_packets_cell= *(next_slot_on_time_rx_packets+cell_idx);
	uint32_t 		*next_slot_late_rx_packets_cell= *(next_slot_late_rx_packets+cell_idx);      
    uint64_t*       rx_packets_ts_cell=*(rx_packets_ts+cell_idx);
    uint32_t*       rx_packets_count_cell=*(rx_packets_count+cell_idx); 
    uint32_t*       rx_bytes_count_cell=*(rx_bytes_count+cell_idx);    
    uint64_t*       rx_packets_ts_earliest_cell = *(rx_packets_ts_earliest+cell_idx);
    uint64_t*       rx_packets_ts_latest_cell = *(rx_packets_ts_latest+cell_idx);
    uint64_t*       next_slot_rx_packets_ts_cell=*(next_slot_rx_packets_ts+cell_idx);
    uint32_t*       next_slot_rx_packets_count_cell=*(next_slot_rx_packets_count+cell_idx);
    uint32_t*       next_slot_rx_bytes_count_cell=*(next_slot_rx_bytes_count+cell_idx);

    // PRACH start_prbu = 0
	uint16_t startPRB_offset_idx_0 = 0;
	uint16_t startPRB_offset_idx_1 = 0;
	uint16_t startPRB_offset_idx_2 = 0;
	uint16_t startPRB_offset_idx_3 = 0;

    uint8_t *pkt_thread = NULL;
    uint8_t frameId_pkt=0;
    uint8_t symbol_id_pkt = 0;
    uint8_t subframeId_pkt  = 0;
    uint8_t slotId_pkt      = 0;                    
    uint64_t packet_early_thres = 0;
    uint64_t packet_late_thres  = 0;  

	uint32_t tot_pusch_prb_symbol_ordered=0;
    uint8_t* section_buf;
    uint16_t ecpri_payload_length;
    // 4 bytes for ecpriPcid, ecprSeqid, ecpriEbit, ecpriSubSeqid
    uint16_t current_length = 4 + sizeof(oran_umsg_iq_hdr);
    uint16_t num_prb = 0;
    uint16_t start_prb = 0;
    uint16_t section_id = 0;
    uint16_t compressed_prb_size = (bit_width_cell == BFP_NO_COMPRESSION) ? PRB_SIZE_16F : (bit_width_cell == BFP_COMPRESSION_14_BITS) ? PRB_SIZE_14F : PRB_SIZE_9F;
    uint16_t num_sections = 0;    
    uint8_t start_loop=0;
    int rx_packets_ts_idx=0,next_slot_rx_packets_ts_idx=0,max_pkt_idx=ORDER_KERNEL_MAX_PKTS_PER_OFDM_SYM*ORAN_MAX_SYMBOLS;
    int32_t slot_count_input=(2*subframeId+slotId),slot_count_curr;

    /*
	 if (threadIdx.x==0) {
	 	printf("[order_kernel_cpu_init_comms_single_subSlot start]F%dS%dS%d Cell Idx %d\n",
                frameId, subframeId, slotId,cell_idx);
	 }
    */

	if(threadIdx.x==0){
		DOCA_GPUNETIO_VOLATILE(pusch_ordered_prbs_cell[0]) = 0;
		DOCA_GPUNETIO_VOLATILE(prach_ordered_prbs_cell[0]) = 0;
		done_shared_sh = 1;
        pusch_prb_symbol_ordered[threadIdx.x]=0;
        pusch_prb_symbol_ordered_done[threadIdx.x]=0;        
        rx_pkt_num_total = 0;
        exit_rx_timeout_sh=0;
        exit_ord_done_sh=0;
        early_rx_packets_count_sh=0;//DOCA_GPUNETIO_VOLATILE(*next_slot_early_rx_packets_cell);
        on_time_rx_packets_count_sh=0;//DOCA_GPUNETIO_VOLATILE(*next_slot_on_time_rx_packets_cell);
        late_rx_packets_count_sh=0;//DOCA_GPUNETIO_VOLATILE(*next_slot_late_rx_packets_cell);
        next_slot_early_rx_packets_count_sh=0;
        next_slot_late_rx_packets_count_sh=0;
        next_slot_on_time_rx_packets_count_sh=0;                
	}
	else if(threadIdx.x < ORAN_PUSCH_SYMBOLS_X_SLOT){
        pusch_prb_symbol_ordered[threadIdx.x]=0;
        pusch_prb_symbol_ordered_done[threadIdx.x]=0;
	}

    if(ul_rx_pkt_tracing_level>1){
            if(threadIdx.x<ORAN_MAX_SYMBOLS)
            {
                rx_packets_count_sh[threadIdx.x]=0;//DOCA_GPUNETIO_VOLATILE(next_slot_rx_packets_count_cell[threadIdx.x]);
                rx_bytes_count_sh[threadIdx.x]=0;//DOCA_GPUNETIO_VOLATILE(next_slot_rx_bytes_count_cell[threadIdx.x]);
                rx_packets_ts_earliest_sh[threadIdx.x]=0xFFFFFFFFFFFFFFFFLLU;
                rx_packets_ts_latest_sh[threadIdx.x]=0;
                next_slot_rx_packets_count_sh[threadIdx.x]=0;
                next_slot_rx_bytes_count_sh[threadIdx.x]=0;
            }
            __syncthreads();
            for(uint32_t pkt_idx=threadIdx.x;pkt_idx<max_pkt_idx;pkt_idx+=blockDim.x)
            {
                uint32_t symbol_idx=pkt_idx/ORDER_KERNEL_MAX_PKTS_PER_OFDM_SYM;
                rx_packets_ts_sh[pkt_idx]=DOCA_GPUNETIO_VOLATILE(next_slot_rx_packets_ts_cell[pkt_idx]);
                __threadfence_block();
#if 0                
                if(rx_packets_ts_sh[pkt_idx]!=0)
                    atomicMin((unsigned long long*) &rx_packets_ts_earliest_sh[symbol_idx],(unsigned long long) rx_packets_ts_sh[pkt_idx]);
                atomicMax((unsigned long long*) &rx_packets_ts_latest_sh[symbol_idx],(unsigned long long) rx_packets_ts_sh[pkt_idx]);
                __threadfence_block();
#endif                
                next_slot_rx_packets_ts_sh[pkt_idx]=0;
            }
            __syncthreads();
            if(threadIdx.x<ORAN_MAX_SYMBOLS)
            {
                if(rx_packets_ts_earliest_sh[threadIdx.x]==0)
                    rx_packets_ts_earliest_sh[threadIdx.x]=0xFFFFFFFFFFFFFFFFLLU;
            }            
    }        
    __syncthreads();

	while (1) {
        if(DOCA_GPUNETIO_VOLATILE(*exit_cond_d_cell) == ORDER_KERNEL_START)
        {
            if(threadIdx.x==0)
            {
                ret=order_kernel_cpu_init_comms_wait_packets(
                cell_idx,
                frameId,
                subframeId,
                slotId,
                start_loop,
                kernel_start,
                first_packet_start,
                ready_list_cell,
                ready_sync_sh,
                sem_idx_order, 
                last_sem_idx_order,
                timeout_no_pkt_ns, 
                timeout_first_pkt_ns);
                if(ret<0)
                {
                    DOCA_GPUNETIO_VOLATILE(*exit_cond_d_cell)=(ret==-1)?ORDER_KERNEL_EXIT_TIMEOUT_NO_PKT:ORDER_KERNEL_EXIT_TIMEOUT_RX_PKT; 
                    exit_rx_timeout_sh=1;
                }
                else
                {
                    rx_pkt_num=DOCA_GPUNETIO_VOLATILE(rx_queue_sync_list_cell[sem_idx_order].umsg_num);
                    sem_idx_order_sh=sem_idx_order;
                    //printf("Cell Idx %d, F%dS%dS%d rx_pkt_num %d on_time_rx_packets_count_sh %d from wait_packets\n",
                    //      cell_idx,frameId, subframeId, slotId,rx_pkt_num,on_time_rx_packets_count_sh);
                }
            }
            __syncthreads();
			if(exit_rx_timeout_sh)
            {
                break;
            }
            sem_idx_order=sem_idx_order_sh;

            if(ul_rx_pkt_tracing_level)
            {
                for(uint32_t pkt_idx=threadIdx.x;pkt_idx<rx_pkt_num;pkt_idx+=blockDim.x)
                {
                    rx_pkt_addr=DOCA_GPUNETIO_VOLATILE(rx_queue_sync_list_cell[sem_idx_order].addr[pkt_idx]);
                    rx_timestamp=DOCA_GPUNETIO_VOLATILE(rx_queue_sync_list_cell[sem_idx_order].rx_timestamp[pkt_idx]);
                    pkt_thread = (uint8_t*)rx_pkt_addr;
                    frameId_pkt     = oran_umsg_get_frame_id(pkt_thread);
                    subframeId_pkt  = oran_umsg_get_subframe_id(pkt_thread);
                    slotId_pkt      = oran_umsg_get_slot_id(pkt_thread);
                    symbol_id_pkt = oran_umsg_get_symbol_id(pkt_thread);

                    // Bounds check for symbol_id_pkt from untrusted packet data
                    if(__builtin_expect(symbol_id_pkt >= ORAN_ALL_SYMBOLS, 0))
                    {
                        printf("ERROR invalid symbol_id %d (max %d) for Cell %d F%dS%dS%d\n",
                               symbol_id_pkt,
                               ORAN_ALL_SYMBOLS - 1,
                               cell_idx,
                               frameId_pkt,
                               subframeId_pkt,
                               slotId_pkt);
                        continue; // Skip this packet
                    }

                    ecpri_payload_length = min(oran_umsg_get_ecpri_payload(pkt_thread),ORAN_ECPRI_MAX_PAYLOAD_LEN);

                    slot_count_curr = (2*subframeId_pkt+slotId_pkt);       
                    if((frameId_pkt!=frameId) || (((slot_count_curr-slot_count_input+20)%20)>1)) //Drop scoring if packets from 2 or greater slots away are received during current slot reception or if the frame IDs mis-match
                    {
                        continue;
                    }                                 
                    if(((slot_count_curr-slot_count_input+20)%20)==1) //TODO: Fix magic number 20 => number of slots in a radio frame for mu=1
                    {
                        packet_early_thres = slot_start_cell+ slot_duration_cell + ta4_min_ns_cell + (slot_duration_cell * symbol_id_pkt / ORAN_MAX_SYMBOLS);
                        packet_late_thres  = slot_start_cell+ slot_duration_cell + ta4_max_ns_cell + (slot_duration_cell * symbol_id_pkt / ORAN_MAX_SYMBOLS);
                        if(rx_timestamp < packet_early_thres)
                            atomicAdd(&next_slot_early_rx_packets_count_sh, 1);
                        else if(rx_timestamp > packet_late_thres)
                            atomicAdd(&next_slot_late_rx_packets_count_sh, 1);
                        else
                            atomicAdd(&next_slot_on_time_rx_packets_count_sh, 1);
                        __threadfence_block();
                        if(ul_rx_pkt_tracing_level>1)
                        {
                            next_slot_rx_packets_ts_idx = atomicAdd(&next_slot_rx_packets_count_sh[symbol_id_pkt], 1);
                            __threadfence_block();
                            next_slot_rx_packets_ts_idx += ORDER_KERNEL_MAX_PKTS_PER_OFDM_SYM * symbol_id_pkt;
                            next_slot_rx_packets_ts_sh[next_slot_rx_packets_ts_idx] = rx_timestamp;
                        }
                    }      
                    else
                    {
                        packet_early_thres = slot_start_cell + ta4_min_ns_cell + (slot_duration_cell * symbol_id_pkt / ORAN_MAX_SYMBOLS);
                        packet_late_thres  = slot_start_cell + ta4_max_ns_cell + (slot_duration_cell * symbol_id_pkt / ORAN_MAX_SYMBOLS);
    					if (rx_timestamp < packet_early_thres)
    						atomicAdd(&early_rx_packets_count_sh, 1);
    					else if (rx_timestamp > packet_late_thres)
    						atomicAdd(&late_rx_packets_count_sh, 1);
    					else
    						atomicAdd(&on_time_rx_packets_count_sh, 1);
                        __threadfence_block();                              
                        if(ul_rx_pkt_tracing_level>1){    
                            rx_packets_ts_idx = atomicAdd(&rx_packets_count_sh[symbol_id_pkt], 1);
                            atomicAdd(&rx_bytes_count_sh[symbol_id_pkt], ORAN_ETH_HDR_SIZE + ecpri_payload_length);
                            __threadfence_block();
                            rx_packets_ts_idx+=ORDER_KERNEL_MAX_PKTS_PER_OFDM_SYM*symbol_id_pkt;
                            rx_packets_ts_sh[rx_packets_ts_idx]=rx_timestamp;
                            atomicMin((unsigned long long*) &rx_packets_ts_earliest_sh[symbol_id_pkt],(unsigned long long) rx_timestamp);
                            atomicMax((unsigned long long*) &rx_packets_ts_latest_sh[symbol_id_pkt],(unsigned long long) rx_timestamp);                        
                            __threadfence_block();
                        //printf("symbol_id_pkt:%d,rx_packets_count_cell[symbol_id_pkt]=%d,rx_packets_ts_cell[ORDER_KERNEL_MAX_PKTS_PER_OFDM_SYM*symbol_id_pkt+rx_packets_count_cell[symbol_id_pkt]]=%llu\n",
                        //       symbol_id_pkt,rx_packets_count_cell[symbol_id_pkt],rx_packets_ts_cell[ORDER_KERNEL_MAX_PKTS_PER_OFDM_SYM*symbol_id_pkt+rx_packets_count_cell[symbol_id_pkt]]);
                        }   
                    }
                }
            }
            __syncthreads();

			/* Order & decompress packets */
			for (uint32_t pkt_idx = warpId; pkt_idx < rx_pkt_num; pkt_idx += nwarps) {                
                #if 0
                if(laneId==0)
                {
                    printf("Cell Idx %d, threadIdx.x %d F%dS%dS%d pkt_idx %d addr %p\n",
                          cell_idx,threadIdx.x,frameId, subframeId, slotId,pkt_idx,(void*)rx_queue_sync_list_cell[sem_idx_order].addr[pkt_idx]);                    
                }                
                #endif
                rx_pkt_addr=DOCA_GPUNETIO_VOLATILE(rx_queue_sync_list_cell[sem_idx_order].addr[pkt_idx]);
                rx_timestamp=DOCA_GPUNETIO_VOLATILE(rx_queue_sync_list_cell[sem_idx_order].rx_timestamp[pkt_idx]);
				pkt_thread = (uint8_t*)rx_pkt_addr;
				frameId_pkt     = oran_umsg_get_frame_id(pkt_thread);
				subframeId_pkt  = oran_umsg_get_subframe_id(pkt_thread);
				slotId_pkt      = oran_umsg_get_slot_id(pkt_thread);
				symbol_id_pkt = oran_umsg_get_symbol_id(pkt_thread);

                // Bounds check for symbol_id_pkt from untrusted packet data
                if(__builtin_expect(symbol_id_pkt >= ORAN_ALL_SYMBOLS, 0))
                {
                    printf("ERROR invalid symbol_id %d (max %d) for Cell %d F%dS%dS%d\n",
                           symbol_id_pkt,
                           ORAN_ALL_SYMBOLS - 1,
                           cell_idx,
                           frameId_pkt,
                           subframeId_pkt,
                           slotId_pkt);
                    continue; // Skip this packet
                }

                slot_count_curr = (2*subframeId_pkt+slotId_pkt);
                #if 0
                if(laneId==0)
                {
                    printf("Cell Idx %d, threadIdx.x %d F%dS%dS%d From Packet:F%dS%dS%d symbol Id %d\n",
                          cell_idx,threadIdx.x,frameId, subframeId, slotId,frameId_pkt, subframeId_pkt, slotId_pkt,symbol_id_pkt);                    
                }
                #endif
				/* If current frame */
				if (
					((frameId_pkt == frameId) && (subframeId_pkt == subframeId) && (slotId_pkt > slotId)) ||
					((frameId_pkt == frameId) && (subframeId_pkt == ((subframeId+1) % 10)))
				) {              
					if (laneId == 0 && DOCA_GPUNETIO_VOLATILE(done_shared_sh) == 1){
						//printf("[DONE Shared 0]F%d/%d SF %d/%d SL %d/%d last_sem_idx_order %d, sem_idx_order %d,sem_idx_rx %d, threadIdx.x %d\n", frameId_pkt, frameId, subframeId_pkt, subframeId, slotId_pkt, slotId,last_sem_idx_order, sem_idx_order, sem_idx_rx,threadIdx.x);
						DOCA_GPUNETIO_VOLATILE(done_shared_sh) = 0;
					}
				} else {
					if (!((frameId_pkt == frameId) && (subframeId_pkt == subframeId) && (slotId_pkt == slotId)))
						 continue;                 

					/* if this is the right slot, order & decompress */
					section_buf = oran_umsg_get_first_section_buf(pkt_thread);
					ecpri_payload_length = oran_umsg_get_ecpri_payload(pkt_thread);
					// 4 bytes for ecpriPcid, ecprSeqid, ecpriEbit, ecpriSubSeqid
					current_length = 4 + sizeof(oran_umsg_iq_hdr);
					num_prb = 0;
					start_prb = 0;
					section_id = 0;
					num_sections = 0;
                    uint16_t prb_buffer_size = 0;
					while(current_length < ecpri_payload_length)
					{
						current_time = __globaltimer();
						num_prb = oran_umsg_get_num_prb_from_section_buf(section_buf);
						section_id = oran_umsg_get_section_id_from_section_buf(section_buf);
						start_prb = oran_umsg_get_start_prb_from_section_buf(section_buf);
						if(num_prb==0)
							num_prb=ORAN_MAX_PRB_X_SLOT;
                        prb_buffer_size = compressed_prb_size * num_prb;

						pkt_offset_ptr = section_buf + ORAN_IQ_UNCOMPRESSED_SECTION_OVERHEAD;
						if(section_id < prach_section_id_0)
						{
							buffer = pusch_buffer_cell;
							gbuf_offset_ptr = buffer + oran_get_offset_from_hdr(pkt_thread, (uint16_t)get_eaxc_index(pusch_eAxC_map_cell, pusch_eAxC_num_cell,
																		 oran_umsg_get_flowid(pkt_thread)),
													    pusch_symbols_x_slot, pusch_prb_x_port_x_symbol_cell, prb_size, start_prb);
						}
						else {
							if(section_id == prach_section_id_0) buffer = prach_buffer_0_cell;
							else if(section_id == prach_section_id_1) buffer = prach_buffer_1_cell;
							else if(section_id == prach_section_id_2) buffer = prach_buffer_2_cell;
							else if(section_id == prach_section_id_3) buffer = prach_buffer_3_cell;
							else {
								// Invalid section_id - skip this section
                                printf("ERROR invalid section_id %d for Cell %d F%dS%dS%d\n", section_id, cell_idx, frameId_pkt, subframeId_pkt, slotId_pkt);
								break;
							}
							gbuf_offset_ptr = buffer + oran_get_offset_from_hdr(pkt_thread, (uint16_t)get_eaxc_index(prach_eAxC_map_cell, prach_eAxC_num_cell,
																		 oran_umsg_get_flowid(pkt_thread)),
													    prach_symbols_x_slot, prach_prb_x_port_x_symbol_cell, prb_size, start_prb);
    
							/* prach_buffer_x_cell is populated based on number of PRACH PDU's, hence the index can be used as "Frequency domain occasion index"
							   and mutiplying with num_prb i.e. NRARB=12 (NumRB's (PRACH SCS=30kHz) for each FDM ocassion) will yeild the corrosponding PRB start for each Frequency domain index
							   Note: WIP for a more generic approach to caluclate and pass the startRB from the cuPHY-CP */
							if(section_id == prach_section_id_0) gbuf_offset_ptr -= startPRB_offset_idx_0;
							else if(section_id == prach_section_id_1) gbuf_offset_ptr -= startPRB_offset_idx_1;
							else if(section_id == prach_section_id_2) gbuf_offset_ptr -= startPRB_offset_idx_2;
							else if(section_id == prach_section_id_3) gbuf_offset_ptr -= startPRB_offset_idx_3;
						}
    
						// CWE-787: drop copy/decompress if untrusted symbolId/startPrb/numPrb
						// would index past the per-eAxC antenna buffer.
						{
							const int chk_sxs = (section_id < prach_section_id_0) ? pusch_symbols_x_slot : prach_symbols_x_slot;
							const int chk_pps = (section_id < prach_section_id_0) ? pusch_prb_x_port_x_symbol_cell : prach_prb_x_port_x_symbol_cell;
							if(!ul_oran_offset_in_bounds(pkt_thread, start_prb, num_prb, chk_sxs, chk_pps))
								num_prb = 0;
						}
                        if(comp_meth_cell == static_cast<uint8_t>(aerial_fh::UserDataCompressionMethod::BLOCK_FLOATING_POINT))
                        {
                            if(bit_width_cell == BFP_NO_COMPRESSION) // BFP with 16 bits is a special case and uses FP16, so copy the values
                            {
                                for(int index_copy = laneId; index_copy < (num_prb * prb_size); index_copy += 32)
                                    gbuf_offset_ptr[index_copy] = pkt_offset_ptr[index_copy];
                            }
                            else
                            {
                                decompress_scale_blockFP<false>((unsigned char*)pkt_offset_ptr, (__half*)gbuf_offset_ptr, beta_cell, num_prb, bit_width_cell, (int)(threadIdx.x & 31), 32);
                            }
                        } else // aerial_fh::UserDataCompressionMethod::NO_COMPRESSION
                        {
                            decompress_scale_fixed<false>(pkt_offset_ptr, (__half*)gbuf_offset_ptr, beta_cell, num_prb, bit_width_cell, (int)(threadIdx.x & 31), 32);
                        }
						// Only first warp thread increases the number of tot PRBs
						if(laneId == 0) {
							int oprb_ch1 = 0;
							int oprb_ch2 = 0;

							if(section_id < prach_section_id_0) {
								tot_pusch_prb_symbol_ordered = atomicAdd(&pusch_prb_symbol_ordered[symbol_id_pkt],num_prb);
								tot_pusch_prb_symbol_ordered += num_prb;							
								if(tot_pusch_prb_symbol_ordered >= pusch_prb_symbol_map_cell[symbol_id_pkt] && DOCA_GPUNETIO_VOLATILE(pusch_prb_symbol_ordered_done[symbol_id_pkt])==0){
									DOCA_GPUNETIO_VOLATILE(pusch_prb_symbol_ordered_done[symbol_id_pkt])=1;
									atomicOr(&sym_ord_done_mask_arr[symbol_id_pkt],cell_idx_mask);
									//printf("Lane ID = %d Warp ID = %d symbol_id_pkt = %d pusch_prb_symbol_ordered_done[symbol_id_pkt] = %d sym_ord_done_mask_arr[symbol_id_pkt]=%d\n",laneId,warpId,symbol_id_pkt,DOCA_GPUNETIO_VOLATILE(pusch_prb_symbol_ordered_done[symbol_id_pkt]),DOCA_GPUNETIO_VOLATILE(sym_ord_done_mask_arr[symbol_id_pkt]));                                                                
									if(DOCA_GPUNETIO_VOLATILE(sym_ord_done_mask_arr[symbol_id_pkt])==num_order_cells_sym_mask_arr[symbol_id_pkt] && DOCA_GPUNETIO_VOLATILE(sym_ord_done_sig_arr[symbol_id_pkt])==(uint32_t)SYM_RX_NOT_DONE){
										DOCA_GPUNETIO_VOLATILE(sym_ord_done_sig_arr[symbol_id_pkt])=(uint32_t)SYM_RX_DONE;
									}
								}                                                       
								oprb_ch1 = atomicAdd(pusch_ordered_prbs_cell, num_prb);
								oprb_ch2 = atomicAdd(prach_ordered_prbs_cell, 0);
							} else {
								oprb_ch1 = atomicAdd(pusch_ordered_prbs_cell, 0);
								oprb_ch2 = atomicAdd(prach_ordered_prbs_cell, num_prb);
							}

							//printf("Lane ID = %d threadIdx.x =%d Warp ID = %d oprb_ch1 %d oprb_ch2 %d num_prb %d prb_x_slot %d\n",
							//        laneId, threadIdx.x,warpId, oprb_ch1, oprb_ch2, num_prb, prb_x_slot);
							if(oprb_ch1 + oprb_ch2 + num_prb >= prb_x_slot){
								DOCA_GPUNETIO_VOLATILE(*exit_cond_d_cell) = ORDER_KERNEL_EXIT_PRB;
                                exit_ord_done_sh=1;
                            }
						}
						current_length += prb_buffer_size + ORAN_IQ_UNCOMPRESSED_SECTION_OVERHEAD;
						section_buf = pkt_offset_ptr + prb_buffer_size;
						++num_sections;
						if(num_sections > ORAN_MAX_PRB_X_SLOT)
						{
							printf("Invalid U-Plane packet, num_sections %d > 273 for Cell %d F%dS%dS%d\n", num_sections, cell_idx, frameId_pkt, subframeId_pkt, slotId_pkt);
							break;
						}
					}
				}
			}
			__syncthreads();
			if(threadIdx.x == 0 && DOCA_GPUNETIO_VOLATILE(done_shared_sh) == 1) {
                ACCESS_ONCE(ready_list_cell[sem_idx_order])                = SYNC_PACKET_STATUS_DONE; //Do not set it if there are packets for the next slot
                ACCESS_ONCE(rx_queue_sync_list_cell[sem_idx_order].status) = SYNC_PACKET_STATUS_DONE;                
				last_sem_idx_order = (last_sem_idx_order + 1) & (sem_order_num_cell - 1);
			}
			sem_idx_order = (sem_idx_order+1) & (sem_order_num_cell - 1);
            if(exit_ord_done_sh)
            {
                break;
            }
        }
        else if(DOCA_GPUNETIO_VOLATILE(*exit_cond_d_cell) == ORDER_KERNEL_IDLE)
        {
            continue;
        }
        else if(DOCA_GPUNETIO_VOLATILE(*exit_cond_d_cell) == ORDER_KERNEL_ABORT)
        {
            //TODO: Set the subSlot proc flags here
            break;
        }
	}
    __syncthreads();

	if(ul_rx_pkt_tracing_level){
        if(threadIdx.x<ORAN_MAX_SYMBOLS)
        {
            DOCA_GPUNETIO_VOLATILE(rx_packets_count_cell[threadIdx.x])=rx_packets_count_sh[threadIdx.x];
            DOCA_GPUNETIO_VOLATILE(rx_bytes_count_cell[threadIdx.x])=rx_bytes_count_sh[threadIdx.x];
            DOCA_GPUNETIO_VOLATILE(next_slot_rx_packets_count_cell[threadIdx.x])=next_slot_rx_packets_count_sh[threadIdx.x];
            DOCA_GPUNETIO_VOLATILE(next_slot_rx_bytes_count_cell[threadIdx.x])=next_slot_rx_bytes_count_sh[threadIdx.x];
            DOCA_GPUNETIO_VOLATILE(rx_packets_ts_earliest_cell[threadIdx.x])=rx_packets_ts_earliest_sh[threadIdx.x];
            DOCA_GPUNETIO_VOLATILE(rx_packets_ts_latest_cell[threadIdx.x])=rx_packets_ts_latest_sh[threadIdx.x];                
        }
        __syncthreads();
        for(uint32_t pkt_idx=threadIdx.x;pkt_idx<max_pkt_idx;pkt_idx+=blockDim.x)
        {
            DOCA_GPUNETIO_VOLATILE(rx_packets_ts_cell[pkt_idx])=rx_packets_ts_sh[pkt_idx];
            DOCA_GPUNETIO_VOLATILE(next_slot_rx_packets_ts_cell[pkt_idx])=next_slot_rx_packets_ts_sh[pkt_idx];
        }
        __syncthreads();        
    }        

	if (threadIdx.x == 0) {
			DOCA_GPUNETIO_VOLATILE(*last_sem_idx_order_h_cell) = last_sem_idx_order;
			DOCA_GPUNETIO_VOLATILE(*start_cuphy_d_cell) = 1;
            DOCA_GPUNETIO_VOLATILE(*early_rx_packets_cell) = early_rx_packets_count_sh;
            DOCA_GPUNETIO_VOLATILE(*on_time_rx_packets_cell) = on_time_rx_packets_count_sh;
            DOCA_GPUNETIO_VOLATILE(*late_rx_packets_cell) = late_rx_packets_count_sh;      
            DOCA_GPUNETIO_VOLATILE(*next_slot_early_rx_packets_cell)=next_slot_early_rx_packets_count_sh;
            DOCA_GPUNETIO_VOLATILE(*next_slot_on_time_rx_packets_cell)=next_slot_on_time_rx_packets_count_sh;
            DOCA_GPUNETIO_VOLATILE(*next_slot_late_rx_packets_cell)=next_slot_late_rx_packets_count_sh;               
            //printf("[order_kernel_cpu_init_comms_single_subSlot end]F%dS%dS%d Cell Idx %d Exit cond %d\n",
            //        frameId, subframeId, slotId,cell_idx,DOCA_GPUNETIO_VOLATILE(*exit_cond_d_cell));    
    }
    return;
}


__global__ void order_kernel_doca_single_srs(
	/* DOCA objects */
	struct doca_gpu_eth_rxq **doca_rxq,
	struct doca_gpu_semaphore_gpu **sem_gpu,
	const uint16_t* sem_order_num,

	/* Cell */
	const int*		cell_id,
    const int*		ru_type,
	uint32_t		**start_cuphy_d,
	uint32_t		**exit_cond_d,
	uint32_t		**last_sem_idx_rx_h,
	uint32_t		**last_sem_idx_order_h,
	const int*		comp_meth,
    const int*		bit_width,
	const float*		beta,
	const int		prb_size,

	/* Timeout */
	const uint32_t	timeout_no_pkt_ns,
	const uint32_t	timeout_first_pkt_ns,
    const uint32_t  max_rx_pkts,

	/* ORAN */
	const uint8_t		frameId,
	const uint8_t		subframeId,
	const uint8_t		slotId,

	/* Order kernel specific */
	int			*barrier_flag,
	uint8_t		**done_shared,

        /* SRS Timers */
        uint32_t        **early_rx_packets_srs,
        uint32_t        **on_time_rx_packets_srs,
        uint32_t        **late_rx_packets_srs,
        uint32_t        **next_slot_early_rx_packets_srs,
	uint32_t        **next_slot_on_time_rx_packets_srs,
	uint32_t        **next_slot_late_rx_packets_srs,

        /* SRS stats */
        uint32_t        **rx_packets_count_srs,
        uint32_t        **rx_bytes_count_srs,
        uint32_t        **next_slot_rx_packets_count_srs,
        uint32_t        **next_slot_rx_bytes_count_srs,
    uint8_t                ul_rx_pkt_tracing_level,
    uint64_t**             rx_packets_ts_srs,
    uint32_t**             rx_packets_count_per_sym_srs,
    uint64_t**             rx_packets_ts_earliest_srs,
    uint64_t**             rx_packets_ts_latest_srs,	
    uint64_t**             next_slot_rx_packets_ts_srs,
    uint32_t**             next_slot_rx_packets_count_per_sym_srs,        

        uint64_t*	slot_start,
	uint64_t*       ta4_min_ns,
	uint64_t*       ta4_max_ns,
	uint64_t*       slot_duration,
	/* SRS */
	uint16_t		**srs_eAxC_map,
	int*			srs_eAxC_num,
	uint8_t		**srs_buffer,
	int*			srs_prb_x_slot,
	int			srs_symbols_x_slot,
	uint32_t*			srs_prb_x_port_x_symbol,
	uint32_t		**srs_ordered_prbs,
    uint8_t*          srs_start_sym
    )
{
	int laneId = threadIdx.x % 32;
	int warpId = threadIdx.x / 32;
	int nwarps = blockDim.x / 32;
	int cell_idx=blockIdx.x;

	uint8_t first_packet = 0;
	unsigned long long first_packet_start = 0;
	unsigned long long current_time = 0;
	unsigned long long kernel_start = __globaltimer();
	uint8_t* pkt_offset_ptr, *gbuf_offset_ptr;
	uint8_t* buffer;
	int prb_x_slot=srs_prb_x_slot[cell_idx];
	doca_error_t ret = (doca_error_t)0;
	// Restart from last semaphore item
	int sem_idx_rx = (int)(*(*(last_sem_idx_rx_h+cell_idx)));
	int sem_idx_order = (int)(*(*(last_sem_idx_order_h+cell_idx)));
	int last_sem_idx_order = (int)(*(*(last_sem_idx_order_h+cell_idx)));
	const uint64_t timeout_ns = 100000;
	//int  barrier_idx = 1, barrier_signal = gridDim.x;

	__shared__ uint32_t rx_pkt_num;
	__shared__ uint64_t rx_buf_idx;
	__shared__ uint32_t done_shared_sh;
	uint64_t rx_timestamp;
    __shared__ uint32_t early_rx_packets_count_sh;
	__shared__ uint32_t on_time_rx_packets_count_sh;
	__shared__ uint32_t late_rx_packets_count_sh;
    __shared__ uint32_t next_slot_early_rx_packets_count_sh;
    __shared__ uint32_t next_slot_on_time_rx_packets_count_sh;
    __shared__ uint32_t next_slot_late_rx_packets_count_sh;
    __shared__ uint32_t rx_packets_count_sh;
    __shared__ uint32_t rx_bytes_count_sh;
    __shared__ uint32_t next_slot_rx_packets_count_sh;
    __shared__ uint32_t next_slot_rx_bytes_count_sh;
    __shared__ uint64_t rx_packets_ts_sh[ORDER_KERNEL_MAX_PKTS_PER_OFDM_SYM*ORAN_MAX_SYMBOLS];
    __shared__ uint32_t rx_packets_count_per_sym_sh[ORAN_MAX_SYMBOLS];     
    __shared__ uint64_t rx_packets_ts_earliest_sh[ORAN_MAX_SYMBOLS];
    __shared__ uint64_t rx_packets_ts_latest_sh[ORAN_MAX_SYMBOLS];
    __shared__ uint64_t next_slot_rx_packets_ts_sh[ORDER_KERNEL_MAX_PKTS_PER_OFDM_SYM*ORAN_MAX_SYMBOLS];
    __shared__ uint32_t next_slot_rx_packets_count_per_sym_sh[ORAN_MAX_SYMBOLS];    
    __shared__ uint32_t dropped_packets_printed;
    __shared__ struct doca_gpu_dev_eth_rxq_attr out_attr_sh[512];

	//Cell specific (de-reference from host pinned memory once)
	uint8_t* done_shared_cell=*(done_shared+cell_idx);
	uint32_t* srs_ordered_prbs_cell=*(srs_ordered_prbs+cell_idx);
	uint32_t* exit_cond_d_cell=*(exit_cond_d+cell_idx);
	uint32_t* last_sem_idx_rx_h_cell=*(last_sem_idx_rx_h+cell_idx);
	uint32_t* last_sem_idx_order_h_cell=*(last_sem_idx_order_h+cell_idx);
	struct doca_gpu_eth_rxq* doca_rxq_cell=*(doca_rxq+cell_idx);
	struct doca_gpu_semaphore_gpu* sem_gpu_cell=*(sem_gpu+cell_idx);
	int srs_prb_x_slot_cell=srs_prb_x_slot[cell_idx];

	uint8_t		*srs_buffer_cell=*(srs_buffer+cell_idx);
	uint16_t		*srs_eAxC_map_cell=*(srs_eAxC_map+cell_idx);
	int			srs_eAxC_num_cell=srs_eAxC_num[cell_idx];
	uint32_t			srs_prb_x_port_x_symbol_cell=	srs_prb_x_port_x_symbol[cell_idx];
    const int		ru_type_cell=ru_type[cell_idx];
	const int		comp_meth_cell=comp_meth[cell_idx];
    const int		bit_width_cell=bit_width[cell_idx];
	const float		beta_cell=beta[cell_idx];
    uint8_t         srs_start_sym_cell=srs_start_sym[cell_idx];
	uint32_t		*start_cuphy_d_cell=*(start_cuphy_d+cell_idx);
	const uint16_t sem_order_num_cell=sem_order_num[cell_idx];
    uint32_t offset;
    uint64_t		slot_start_cell=slot_start[cell_idx];
    uint64_t		ta4_min_ns_cell=ta4_min_ns[cell_idx];
    uint64_t		ta4_max_ns_cell=ta4_max_ns[cell_idx];
    uint64_t		slot_duration_cell=slot_duration[cell_idx];
    uint32_t 		*early_rx_packets_cell= *(early_rx_packets_srs+cell_idx);
    uint32_t 		*on_time_rx_packets_cell= *(on_time_rx_packets_srs+cell_idx);
    uint32_t 		*late_rx_packets_cell= *(late_rx_packets_srs+cell_idx);
    uint32_t 		*next_slot_early_rx_packets_cell= *(next_slot_early_rx_packets_srs+cell_idx);
    uint32_t 		*next_slot_on_time_rx_packets_cell= *(next_slot_on_time_rx_packets_srs+cell_idx);
    uint32_t 		*next_slot_late_rx_packets_cell= *(next_slot_late_rx_packets_srs+cell_idx);
    uint32_t 		*rx_packets_count_cell= *(rx_packets_count_srs+cell_idx);
    uint32_t 		*rx_bytes_count_cell= *(rx_bytes_count_srs+cell_idx);
    uint32_t 		*next_slot_rx_packets_count_cell= *(next_slot_rx_packets_count_srs+cell_idx);
    uint32_t 		*next_slot_rx_bytes_count_cell= *(next_slot_rx_bytes_count_srs+cell_idx);
    uint64_t*       rx_packets_ts_cell=*(rx_packets_ts_srs+cell_idx);
    uint64_t*       next_slot_rx_packets_ts_cell=*(next_slot_rx_packets_ts_srs+cell_idx);
    uint32_t*        rx_packets_count_per_sym_cell=*(rx_packets_count_per_sym_srs+cell_idx);    
    uint32_t*       next_slot_rx_packets_count_per_sym_cell=*(next_slot_rx_packets_count_per_sym_srs+cell_idx);
    uint64_t*       rx_packets_ts_earliest_cell = *(rx_packets_ts_earliest_srs+cell_idx);
    uint64_t*       rx_packets_ts_latest_cell = *(rx_packets_ts_latest_srs+cell_idx);     
    uint64_t packet_early_thres_srs = 0;
    uint64_t packet_late_thres_srs  = 0;
    int rx_packets_ts_idx=0,next_slot_rx_packets_ts_idx=0;
    int max_pkt_idx=ORDER_KERNEL_MAX_PKTS_PER_OFDM_SYM*ORAN_MAX_SYMBOLS;
/*
	if(threadIdx.x==0){
			printf("[Single Order kernel SRS start]sem_idx_rx %d sem_idx_order %d last_sem_idx_order %d F%dS%dS%d first_packet %d timeout_first_pkt_ns %llu current_time %llu first_packet_start %llu\n",sem_idx_rx,sem_idx_order,last_sem_idx_order,frameId, subframeId, slotId,first_packet,timeout_first_pkt_ns,current_time,first_packet_start);
	}
*/

	if(threadIdx.x == 0) {
        //First thread in each block

		DOCA_GPUNETIO_VOLATILE(done_shared_cell[0]) = 1;
		DOCA_GPUNETIO_VOLATILE(srs_ordered_prbs_cell[0]) = 0;
		DOCA_GPUNETIO_VOLATILE(done_shared_sh) = 1;
        // DOCA_GPUNETIO_VOLATILE(total_rx_packets)=0;

	    early_rx_packets_count_sh=DOCA_GPUNETIO_VOLATILE(*next_slot_early_rx_packets_cell);
        on_time_rx_packets_count_sh=DOCA_GPUNETIO_VOLATILE(*next_slot_on_time_rx_packets_cell);
        late_rx_packets_count_sh=DOCA_GPUNETIO_VOLATILE(*next_slot_late_rx_packets_cell);
        rx_packets_count_sh=DOCA_GPUNETIO_VOLATILE(*next_slot_rx_packets_count_cell);
        rx_bytes_count_sh=DOCA_GPUNETIO_VOLATILE(*next_slot_rx_bytes_count_cell);
        next_slot_early_rx_packets_count_sh=0;
        next_slot_late_rx_packets_count_sh=0;
        next_slot_on_time_rx_packets_count_sh=0;
        next_slot_rx_packets_count_sh=0;
        next_slot_rx_bytes_count_sh=0;

        dropped_packets_printed = 0;
	}
    if(ul_rx_pkt_tracing_level)
    {
            if(threadIdx.x<ORAN_MAX_SYMBOLS)
            {
                rx_packets_count_per_sym_sh[threadIdx.x]=DOCA_GPUNETIO_VOLATILE(next_slot_rx_packets_count_per_sym_cell[threadIdx.x]);
                rx_packets_ts_earliest_sh[threadIdx.x]=0xFFFFFFFFFFFFFFFFLLU;
                rx_packets_ts_latest_sh[threadIdx.x]=0;
                next_slot_rx_packets_count_per_sym_sh[threadIdx.x]=0;
            }
            __syncthreads();
            for(uint32_t pkt_idx=threadIdx.x;pkt_idx<max_pkt_idx;pkt_idx+=blockDim.x)
            {
                uint32_t symbol_idx=pkt_idx/ORDER_KERNEL_MAX_PKTS_PER_OFDM_SYM;
                rx_packets_ts_sh[pkt_idx]=DOCA_GPUNETIO_VOLATILE(next_slot_rx_packets_ts_cell[pkt_idx]);
                __threadfence_block();
                if(rx_packets_ts_sh[pkt_idx]!=0)
                    atomicMin((unsigned long long*) &rx_packets_ts_earliest_sh[symbol_idx],(unsigned long long) rx_packets_ts_sh[pkt_idx]);
                atomicMax((unsigned long long*) &rx_packets_ts_latest_sh[symbol_idx],(unsigned long long) rx_packets_ts_sh[pkt_idx]);
                __threadfence_block();
                next_slot_rx_packets_ts_sh[pkt_idx]=0;
            }
            __syncthreads();
            if(threadIdx.x<ORAN_MAX_SYMBOLS)
            {
                if(rx_packets_ts_earliest_sh[threadIdx.x]==0)
                    rx_packets_ts_earliest_sh[threadIdx.x]=0xFFFFFFFFFFFFFFFFLLU;
            }            
    }
	__syncthreads();

	while (DOCA_GPUNETIO_VOLATILE(*exit_cond_d_cell) == ORDER_KERNEL_RUNNING) {
		// This synchronization is needed to prevent threadIdx.x == 0 from modifying *exit_cond_d_cell
		// below (in case of a timeout) before some other thread has read *exit_cond_d_cell to continue
		// this while loop. If threadIdx.x == 0 modifies *exit_cond_d_cell before all threads have
		// read *exit_cond_d_cell, then some threads will break from the loop while others will
		// continue and we will have threads waiting at two different __syncthreads().
		__syncthreads();

		/* Even receives packets and forward them to Odd Block */
		//if((blockIdx.x&0x1) == 0) {
			if (threadIdx.x == 0) {
				current_time = __globaltimer();
				if (first_packet && ((current_time - first_packet_start) > timeout_first_pkt_ns)) {
					printf("%d Cell %d SRS Order kernel sem_idx_rx %d last_sem_idx_rx %d sem_idx_order %d last_sem_idx_order %d SRS PRBs %d/%d . Wait first packet timeout after %d ns F%dS%dS%d done = %d\n",__LINE__,
						cell_idx, sem_idx_rx, *last_sem_idx_rx_h_cell,
						sem_idx_order,*last_sem_idx_order_h_cell,
						DOCA_GPUNETIO_VOLATILE(srs_ordered_prbs_cell[0]), srs_prb_x_slot_cell,
						timeout_first_pkt_ns, frameId, subframeId, slotId,
						DOCA_GPUNETIO_VOLATILE(done_shared_sh));

					DOCA_GPUNETIO_VOLATILE(*exit_cond_d_cell) = ORDER_KERNEL_EXIT_TIMEOUT_RX_PKT;
                    //Latch last_sem_idx_order to sem_idx_rx in case of this timeout
                    // last_sem_idx_order=sem_idx_rx;
				} else if (((current_time - kernel_start) > timeout_no_pkt_ns)) {
					printf("%d Cell %d SRS Order kernel sem_idx_rx %d last_sem_idx_rx %d sem_idx_order %d last_sem_idx_order %d SRS PRBs %d/%d . Receive more packets timeout after %d ns F%dS%dS%d done = %d\n",__LINE__,
						cell_idx, sem_idx_rx, *last_sem_idx_rx_h_cell,
                        sem_idx_order,*last_sem_idx_order_h_cell,
						DOCA_GPUNETIO_VOLATILE(srs_ordered_prbs_cell[0]), srs_prb_x_slot_cell,
						timeout_no_pkt_ns, frameId, subframeId, slotId,
						DOCA_GPUNETIO_VOLATILE(done_shared_sh));

					DOCA_GPUNETIO_VOLATILE(*exit_cond_d_cell) = ORDER_KERNEL_EXIT_TIMEOUT_NO_PKT;
				}
				// printf("Timeout check Done Exit condition (%d)\n",DOCA_GPUNETIO_VOLATILE(*exit_cond_d_cell));
				DOCA_GPUNETIO_VOLATILE(rx_pkt_num) = 0;
			}
			__syncthreads();

			if (DOCA_GPUNETIO_VOLATILE(*exit_cond_d_cell) != ORDER_KERNEL_RUNNING)
				break;

			// This synchronization is needed because otherwise some thread may progress into order_kernel_doca_process_receive_packets_subSlot() and
			// set *exit_cond_d_cell before some other thread has read *exit_cond_d_cell above. If that happens, then the threads will diverge and
			// deadlock waiting on two different syncthreads.
			__syncthreads();

		if (threadIdx.x == 0) {
			ret = doca_gpu_dev_eth_rxq_recv<DOCA_GPUNETIO_ETH_EXEC_SCOPE_THREAD,DOCA_GPUNETIO_ETH_MCST_DISABLED,DOCA_GPUNETIO_ETH_NIC_HANDLER_AUTO,DOCA_GPUNETIO_ETH_RX_ATTR_NONE>(doca_rxq_cell, max_rx_pkts, timeout_ns, &rx_buf_idx, &rx_pkt_num, out_attr_sh);
			/* If any thread returns receive error, the whole execution stops */
			if (ret != DOCA_SUCCESS) {
				doca_gpu_dev_semaphore_set_status(sem_gpu_cell, sem_idx_rx, DOCA_GPU_SEMAPHORE_STATUS_ERROR);
				DOCA_GPUNETIO_VOLATILE(*exit_cond_d_cell) = ORDER_KERNEL_EXIT_ERROR_LEGACY;
				printf("Exit from SRS rx kernel block %d threadIdx %d ret %d sem_idx_rx %d\n",
						blockIdx.x, threadIdx.x, ret, sem_idx_rx);
			}
		}
			__threadfence();
			__syncthreads();

		if (DOCA_GPUNETIO_VOLATILE(rx_pkt_num) > 0) {
			if (threadIdx.x == 0) {
				doca_gpu_dev_semaphore_set_packet_info(sem_gpu_cell, sem_idx_rx, DOCA_GPU_SEMAPHORE_STATUS_READY, rx_pkt_num, rx_buf_idx);
				if (first_packet == 0) {
						first_packet = 1;
						first_packet_start  = __globaltimer();
					}
					// atomicAdd(&total_rx_packets,rx_pkt_num);
				}
				sem_idx_rx= (sem_idx_rx+1) & (sem_order_num_cell - 1);
			}
			else if(sem_idx_rx == sem_idx_order)
				continue;
			__syncthreads();

		//} else {
			/* Block 1 waits on semaphore for new packets and process them */

			/* Semaphore wait */
		if (threadIdx.x == 0) {
			do {
				ret = doca_gpu_dev_semaphore_get_packet_info_status(sem_gpu_cell, sem_idx_order, DOCA_GPU_SEMAPHORE_STATUS_READY, &rx_pkt_num, &rx_buf_idx);
			} while (ret == DOCA_ERROR_NOT_FOUND && DOCA_GPUNETIO_VOLATILE(*exit_cond_d_cell) == ORDER_KERNEL_RUNNING);
			}
			__syncthreads();

			/* Check error or exit condition */
			if (DOCA_GPUNETIO_VOLATILE(*exit_cond_d_cell) != ORDER_KERNEL_RUNNING) {
				if (threadIdx.x == 0) {
					// printf("EXIT FROM Block %d Sem %d pkt_addr %lx pkt_num %d status_proxy %d exit %d\n",
					// 	blockIdx.x, sem_idx_order, rx_pkt_addr, rx_pkt_num, status_proxy, *exit_cond_d_cell);
					DOCA_GPUNETIO_VOLATILE(*exit_cond_d_cell) = ORDER_KERNEL_EXIT_ERROR_LEGACY;
				}
				break;
			}

			if(DOCA_GPUNETIO_VOLATILE(rx_pkt_num) == 0)
				continue;

			/* Order & decompress packets */
			for (uint32_t pkt_idx = warpId; pkt_idx < rx_pkt_num; pkt_idx += nwarps) {

            ////
            ////PACKET RECEPTION
            ////

			uint8_t *pkt_thread = (uint8_t*)doca_gpu_dev_eth_rxq_get_pkt_addr(doca_rxq_cell, rx_buf_idx + pkt_idx);
				uint16_t section_id_pkt  = oran_umsg_get_section_id(pkt_thread);

				uint8_t frameId_pkt      = oran_umsg_get_frame_id(pkt_thread);
				uint8_t subframeId_pkt   = oran_umsg_get_subframe_id(pkt_thread);
				uint8_t slotId_pkt       = oran_umsg_get_slot_id(pkt_thread);
                int32_t full_slot_diff = calculate_slot_difference(frameId, frameId_pkt, subframeId, subframeId_pkt, slotId, slotId_pkt);

		 		uint8_t symbol_id_pkt    = oran_umsg_get_symbol_id(pkt_thread);

                // Warning for invalid symbol_id (not used for array indexing here, only for timing calculations)
                if(__builtin_expect(symbol_id_pkt >= ORAN_ALL_SYMBOLS, 0))
                {
                    printf("WARNING invalid symbol_id %d (max %d) for Cell %d F%dS%dS%d\n",
                           symbol_id_pkt,
                           ORAN_ALL_SYMBOLS - 1,
                           cell_idx,
                           frameId_pkt,
                           subframeId_pkt,
                           slotId_pkt);
                }

                // calculate SRS Symbol Tx delay
                uint64_t srs_symbol_tx_delay_ns = (symbol_id_pkt * slot_duration_cell) / ORAN_MAX_SYMBOLS;

                if(laneId==0)
                {

                    // if(full_slot_diff != 0) {
                    //     printf("[%d] Full slot diff %d, Desired frame %d/%d/%d, Packet frame %d/%d/%d\n", pkt_idx, full_slot_diff, frameId, subframeId, slotId, frameId_pkt, subframeId_pkt, slotId_pkt);
                    // }

                    rx_timestamp = out_attr_sh[pkt_idx].timestamp_ns;
                    uint16_t ecpri_payload_length = oran_umsg_get_ecpri_payload(pkt_thread);
                    if(full_slot_diff > 0) //Only keep packets that are in the future slots
                    {
                        packet_early_thres_srs = slot_start_cell + ta4_min_ns_cell + srs_symbol_tx_delay_ns;
                        packet_late_thres_srs  = slot_start_cell + ta4_max_ns_cell + srs_symbol_tx_delay_ns;

                        if(rx_timestamp < packet_early_thres_srs){
                            atomicAdd(&next_slot_early_rx_packets_count_sh, 1);
                        }
                        else if(rx_timestamp > packet_late_thres_srs){
                            atomicAdd(&next_slot_late_rx_packets_count_sh, 1);
                        }
                        else{
                            atomicAdd(&next_slot_on_time_rx_packets_count_sh, 1);
                        }
                        atomicAdd(&next_slot_rx_packets_count_sh, 1);
                        atomicAdd(&next_slot_rx_bytes_count_sh, ORAN_ETH_HDR_SIZE + ecpri_payload_length);
                        __threadfence_block();
                        if(ul_rx_pkt_tracing_level)
                        {
                            bool failed_indexing = true;
                            if(symbol_id_pkt < ORAN_MAX_SYMBOLS) {
                                next_slot_rx_packets_ts_idx = ORDER_KERNEL_MAX_PKTS_PER_OFDM_SYM * symbol_id_pkt;
                                next_slot_rx_packets_ts_idx += atomicAdd(&next_slot_rx_packets_count_per_sym_sh[symbol_id_pkt], 1);
                                __threadfence_block();
                                if(next_slot_rx_packets_ts_idx >= 0 && next_slot_rx_packets_ts_idx < max_pkt_idx) {
                                    next_slot_rx_packets_ts_sh[next_slot_rx_packets_ts_idx] = rx_timestamp;
                                    failed_indexing = false;
                                }
                            }
                            if(failed_indexing) {
                                printf("Indexing error: symbol_id_pkt %d ORDER_KERNEL_MAX_PKTS_PER_OFDM_SYM %d next_slot_rx_packets_ts_idx %d max_pkt_idx %d\n",
                                       symbol_id_pkt, ORDER_KERNEL_MAX_PKTS_PER_OFDM_SYM, next_slot_rx_packets_ts_idx, max_pkt_idx);
                            }
                        }                        
                    }
                    else if(full_slot_diff == 0) //Same Frame,sub-frame, slot as the desired frame,sub-frame, slot
                    {
                        packet_early_thres_srs = slot_start_cell + ta4_min_ns_cell + srs_symbol_tx_delay_ns;
                        packet_late_thres_srs  = slot_start_cell + ta4_max_ns_cell + srs_symbol_tx_delay_ns;

                        if (rx_timestamp < packet_early_thres_srs){
                            atomicAdd(&early_rx_packets_count_sh, 1);
                        }
                        else if (rx_timestamp > packet_late_thres_srs){
                            atomicAdd(&late_rx_packets_count_sh, 1);
                        }
                        else{
                            atomicAdd(&on_time_rx_packets_count_sh, 1);
                        }
                        atomicAdd(&rx_packets_count_sh, 1);
                        atomicAdd(&rx_bytes_count_sh, ORAN_ETH_HDR_SIZE + ecpri_payload_length);
                        __threadfence_block();
                        if(ul_rx_pkt_tracing_level)
                        {
                            bool failed_indexing = true;
                            if(symbol_id_pkt < ORAN_MAX_SYMBOLS) {
                                rx_packets_ts_idx = ORDER_KERNEL_MAX_PKTS_PER_OFDM_SYM * symbol_id_pkt;
                                rx_packets_ts_idx += atomicAdd(&rx_packets_count_per_sym_sh[symbol_id_pkt], 1);
                                __threadfence_block();
                                if(rx_packets_ts_idx >= 0 && rx_packets_ts_idx < max_pkt_idx) {
                                    rx_packets_ts_sh[rx_packets_ts_idx] = rx_timestamp;
                                    atomicMin((unsigned long long*) &rx_packets_ts_earliest_sh[symbol_id_pkt],(unsigned long long) rx_timestamp);
                                    atomicMax((unsigned long long*) &rx_packets_ts_latest_sh[symbol_id_pkt],(unsigned long long) rx_timestamp);
                                    failed_indexing = false;
                                }
                            }
                            if(failed_indexing) {
                                printf("Indexing error: symbol_id_pkt %d ORDER_KERNEL_MAX_PKTS_PER_OFDM_SYM %d rx_packets_ts_idx %d max_pkt_idx %d\n",
                                       symbol_id_pkt, ORDER_KERNEL_MAX_PKTS_PER_OFDM_SYM, rx_packets_ts_idx, max_pkt_idx);
                            }
                            __threadfence_block();
                        }                        
                    }
                    else { //Drop packets that are in the past slots
                        if(atomicExch(&dropped_packets_printed,1) == 0) {
                            //Only print once per cell
                            printf("Warning: Dropping packet for cell %d, Desired frame %d/%d/%d, Packet frame %d/%d/%d. Only one print per cell.\n",
                                   blockIdx.x, frameId, subframeId, slotId, frameId_pkt, subframeId_pkt, slotId_pkt);
                            
                        }

                        // printf("WARNING: Ignoring packet %d cell %d symbol %d with full_slot_diff %d. Desired frame %d/%d/%d, Packet frame %d/%d/%d\n", 
                        //        pkt_idx, blockIdx.x, symbol_id_pkt, full_slot_diff, frameId, subframeId, slotId, frameId_pkt, subframeId_pkt, slotId_pkt);
                    }
                }
				#if 0
				if (laneId == 0 && warpId == 0)
					printf("pkt_thread %lx: src %x:%x:%x:%x:%x:%x dst %x:%x:%x:%x:%x:%x proto %x:%x vlan %x:%x ecpri %x:%x hdr %x:%x:%x:%x:%x:%x:%x:%x\n",
						// "pkt_idx %d rx_buf_idx %d section_id_pkt %d/%d frameId_pkt %d/%d subframeId_pkt %d/%d slotId_pkt %d/%d\n",
						pkt_thread,
						pkt_thread[0], pkt_thread[1], pkt_thread[2], pkt_thread[3], pkt_thread[4], pkt_thread[5],
						pkt_thread[6], pkt_thread[7], pkt_thread[8], pkt_thread[9], pkt_thread[10], pkt_thread[11],
						pkt_thread[12], pkt_thread[13], pkt_thread[14], pkt_thread[15],
						pkt_thread[16], pkt_thread[17],
						pkt_thread[18], pkt_thread[19], pkt_thread[20], pkt_thread[21], pkt_thread[22], pkt_thread[23],
						pkt_idx, rx_buf_idx, section_id_pkt, prach_section_id_0, frameId_pkt, frameId, subframeId_pkt, subframeId, slotId_pkt, slotId);
				#endif


                ////
                ////PACKET PROCESSING
                ////

				if ( full_slot_diff > 0 ) {
                    //These packets will be processed in the next order kernel

					if (laneId == 0 && DOCA_GPUNETIO_VOLATILE(done_shared_sh) == 1){
						//printf("[DONE Shared 0]F%d/%d SF %d/%d SL %d/%d last_sem_idx_order %d, sem_idx_order %d,sem_idx_rx %d, threadIdx.x %d\n", frameId_pkt, frameId, subframeId_pkt, subframeId, slotId_pkt, slotId,last_sem_idx_order, sem_idx_order, sem_idx_rx,threadIdx.x);
						DOCA_GPUNETIO_VOLATILE(done_shared_sh) = 0;
					}

				} else if(full_slot_diff == 0) {
                    //We are processing a packet for this order kernel

					uint8_t* section_buf = oran_umsg_get_first_section_buf(pkt_thread);
					uint16_t ecpri_payload_length = oran_umsg_get_ecpri_payload(pkt_thread);
					// 4 bytes for ecpriPcid, ecprSeqid, ecpriEbit, ecpriSubSeqid
					uint16_t current_length = 4 + sizeof(oran_umsg_iq_hdr);
					uint16_t num_prb = 0;
					uint16_t start_prb = 0;
					uint16_t section_id = 0;
					uint16_t compressed_prb_size = (bit_width_cell == BFP_NO_COMPRESSION) ? PRB_SIZE_16F : (bit_width_cell == BFP_COMPRESSION_14_BITS) ? PRB_SIZE_14F : PRB_SIZE_9F;
					uint16_t prb_buffer_size = 0;
                    bool sanity_check = (current_length < ecpri_payload_length);
                    if(ecpri_hdr_sanity_check(pkt_thread) == false)
                    {
                        printf("ERROR malformatted eCPRI header... block %d thread %d\n", blockIdx.x, threadIdx.x);
                        //break;
                    }
                    while(current_length < ecpri_payload_length)
					{
						current_time = __globaltimer();
                        if(current_length + ORAN_IQ_UNCOMPRESSED_SECTION_OVERHEAD >= ecpri_payload_length)
                        {
                            sanity_check = false;
                            break;
                        }
						num_prb = oran_umsg_get_num_prb_from_section_buf(section_buf);
						section_id = oran_umsg_get_section_id_from_section_buf(section_buf);
						start_prb = oran_umsg_get_start_prb_from_section_buf(section_buf);
						if(num_prb==0)
							num_prb=ORAN_MAX_PRB_X_SLOT;
                        prb_buffer_size = compressed_prb_size * num_prb;

                        //WAR added for ru_type::SINGLE_SECT_MODE O-RU to pass. Will remove it when new FW is applied to fix the erronous ecpri payload length
                        if(ru_type_cell != SINGLE_SECT_MODE && current_length + prb_buffer_size + ORAN_IQ_UNCOMPRESSED_SECTION_OVERHEAD > ecpri_payload_length)
                        {
                            sanity_check = false;
                            break;
                        }
						pkt_offset_ptr = section_buf + ORAN_IQ_UNCOMPRESSED_SECTION_OVERHEAD;
						buffer = srs_buffer_cell;
						offset=oran_srs_get_offset_from_hdr(pkt_thread, (uint16_t)get_eaxc_index(srs_eAxC_map_cell, srs_eAxC_num_cell,
															 oran_umsg_get_flowid(pkt_thread)),
										    srs_symbols_x_slot, srs_prb_x_port_x_symbol_cell, prb_size,start_prb,srs_start_sym_cell);
						gbuf_offset_ptr = buffer + offset;

						// CWE-787: drop copy/decompress if untrusted symbolId/startPrb/numPrb
						// would index past the per-eAxC SRS antenna buffer.
						if(!ul_oran_srs_offset_in_bounds(pkt_thread, start_prb, num_prb, srs_symbols_x_slot, srs_prb_x_port_x_symbol_cell, srs_start_sym_cell))
							num_prb = 0;
                        if(comp_meth_cell == static_cast<uint8_t>(aerial_fh::UserDataCompressionMethod::BLOCK_FLOATING_POINT))
                        {
                            if(bit_width_cell == BFP_NO_COMPRESSION) // BFP with 16 bits is a special case and uses FP16, so copy the values
                            {
                                for(int index_copy = laneId; index_copy < (num_prb * prb_size); index_copy += 32)
                                    gbuf_offset_ptr[index_copy] = pkt_offset_ptr[index_copy];
                            }
                            else
                            {
                                decompress_scale_blockFP<false>((unsigned char*)pkt_offset_ptr, (__half*)gbuf_offset_ptr, beta_cell, num_prb, bit_width_cell, (int)(threadIdx.x & 31), 32);
                            }
                        } else // aerial_fh::UserDataCompressionMethod::NO_COMPRESSION
                        {
                            decompress_scale_fixed<false>(pkt_offset_ptr, (__half*)gbuf_offset_ptr, beta_cell, num_prb, bit_width_cell, (int)(threadIdx.x & 31), 32);
                        }
						// Only first warp thread increases the number of tot PRBs
						if(laneId == 0) {
							int oprb_ch = 0;

							oprb_ch = atomicAdd(srs_ordered_prbs_cell, num_prb);

							// printf("Lane ID = %d Warp ID = %d oprb_ch1 %d oprb_ch2 %d num_prb %d prb_x_slot %d\n",
							//     laneId, warpId, oprb_ch1, oprb_ch2, num_prb, prb_x_slot);
							if(oprb_ch + num_prb >= prb_x_slot)
								DOCA_GPUNETIO_VOLATILE(*exit_cond_d_cell) = ORDER_KERNEL_EXIT_PRB;
						}
						current_length += prb_buffer_size + ORAN_IQ_UNCOMPRESSED_SECTION_OVERHEAD;
						section_buf = pkt_offset_ptr + prb_buffer_size;
					}
                    if(!sanity_check)
                    {
                        printf("ERROR uplane pkt sanity check failed, it could be erroneous BFP, numPrb or ecpri payload len, or other reasons... block %d thread %d\n", blockIdx.x, threadIdx.x);
                        DOCA_GPUNETIO_VOLATILE(*exit_cond_d_cell) = ORDER_KERNEL_EXIT_ERROR_LEGACY;
                        break;
                    }
				}
			}
			__syncthreads();

		if(threadIdx.x == 0 && DOCA_GPUNETIO_VOLATILE(done_shared_sh) == 1) {
			doca_gpu_dev_semaphore_set_status(sem_gpu_cell, last_sem_idx_order, DOCA_GPU_SEMAPHORE_STATUS_DONE);
			last_sem_idx_order = (last_sem_idx_order + 1) & (sem_order_num_cell - 1);
			if (first_packet == 0) {
						first_packet = 1;
						first_packet_start  = __globaltimer();
				}
			}

			sem_idx_order = (sem_idx_order+1) & (sem_order_num_cell - 1);
		//}
	}

	__syncthreads();

	if(ul_rx_pkt_tracing_level)
    {
        if(threadIdx.x<ORAN_MAX_SYMBOLS)
        {
            DOCA_GPUNETIO_VOLATILE(rx_packets_count_per_sym_cell[threadIdx.x])=rx_packets_count_per_sym_sh[threadIdx.x];
            DOCA_GPUNETIO_VOLATILE(next_slot_rx_packets_count_per_sym_cell[threadIdx.x])=next_slot_rx_packets_count_per_sym_sh[threadIdx.x];
            DOCA_GPUNETIO_VOLATILE(rx_packets_ts_earliest_cell[threadIdx.x])=rx_packets_ts_earliest_sh[threadIdx.x];
            DOCA_GPUNETIO_VOLATILE(rx_packets_ts_latest_cell[threadIdx.x])=rx_packets_ts_latest_sh[threadIdx.x];                
        }
        __syncthreads();
        for(uint32_t pkt_idx=threadIdx.x;pkt_idx<max_pkt_idx;pkt_idx+=blockDim.x)
        {
            DOCA_GPUNETIO_VOLATILE(rx_packets_ts_cell[pkt_idx])=rx_packets_ts_sh[pkt_idx];
            DOCA_GPUNETIO_VOLATILE(next_slot_rx_packets_ts_cell[pkt_idx])=next_slot_rx_packets_ts_sh[pkt_idx];
        }
        __syncthreads();
    }       

	if (threadIdx.x == 0) {
		DOCA_GPUNETIO_VOLATILE(*last_sem_idx_rx_h_cell) = sem_idx_rx;
		DOCA_GPUNETIO_VOLATILE(*last_sem_idx_order_h_cell) = last_sem_idx_order;
		DOCA_GPUNETIO_VOLATILE(*start_cuphy_d_cell) = 1;
        DOCA_GPUNETIO_VOLATILE(*early_rx_packets_cell) = early_rx_packets_count_sh;
        DOCA_GPUNETIO_VOLATILE(*on_time_rx_packets_cell) = on_time_rx_packets_count_sh;
        DOCA_GPUNETIO_VOLATILE(*late_rx_packets_cell) = late_rx_packets_count_sh;
        DOCA_GPUNETIO_VOLATILE(*next_slot_early_rx_packets_cell)=next_slot_early_rx_packets_count_sh;
        DOCA_GPUNETIO_VOLATILE(*next_slot_on_time_rx_packets_cell)=next_slot_on_time_rx_packets_count_sh;
        DOCA_GPUNETIO_VOLATILE(*next_slot_late_rx_packets_cell)=next_slot_late_rx_packets_count_sh;
        DOCA_GPUNETIO_VOLATILE(*rx_packets_count_cell) = rx_packets_count_sh;
        DOCA_GPUNETIO_VOLATILE(*rx_bytes_count_cell) = rx_bytes_count_sh;
        DOCA_GPUNETIO_VOLATILE(*next_slot_rx_packets_count_cell) = next_slot_rx_packets_count_sh;
        DOCA_GPUNETIO_VOLATILE(*next_slot_rx_bytes_count_cell) = next_slot_rx_bytes_count_sh;

        //printf("[Single Order kernel SRS end]total_rx_packet %d F%dS%dS%d first_packet %d timeout_first_pkt_ns %llu current_time %llu first_packet_start %llu\n",DOCA_GPUNETIO_VOLATILE(total_rx_packets),frameId, subframeId, slotId);
	}

    //Sequentially print the rx_packets_count_per_sym_sh for each block
    // for(int ii=0; ii<gridDim.x; ii++)
    // {
        
    //     if(threadIdx.x == 0 && blockIdx.x == 0)
    //     {
        
    //         printf("rx_packets_count_per_sym_srs[%i]: ", ii);
    //         for(int i=0;i<ORAN_MAX_SYMBOLS;i++)
    //         {
    //             printf("%d ", rx_packets_count_per_sym_srs[ii][i]);
    //         }
    //         printf("\n");

    //     }

    // }

	return;
}

int launch_order_kernel_doca(
	CUfunction            func,
	cudaStream_t          stream,

    struct doca_gpu_eth_rxq *doca_rxq,
	struct doca_gpu_semaphore_gpu *sem_gpu,
	const uint16_t sem_order_num,

    int                   cell_id,
    int                   ru_type,

	uint32_t*             start_cuphy_d,
	uint32_t*             order_kernel_exit_cond_d,
	uint32_t		*last_sem_idx_rx_h,
	uint32_t		*last_sem_idx_order_h,

	uint32_t              timeout_no_pkt_ns,
	uint32_t              timeout_first_pkt_ns,
    uint32_t              max_rx_pkts,

	uint8_t               frameId,
	uint8_t               subframeId,
	uint8_t               slotId,
	int                   comp_meth,
    int                   bit_width,
	int                   prb_size,
	float                 beta,
	int*                  barrier_flag,
	uint8_t*              done_shared,

	uint32_t*             early_rx_packets,
	uint32_t*             on_time_rx_packets,
	uint32_t*             late_rx_packets,
	uint64_t              slot_start,
	uint64_t              ta4_min_ns,
	uint64_t              ta4_max_ns,
	uint64_t              slot_duration,

	uint16_t*             pusch_eAxC_map,
	int                   pusch_eAxC_num,
	uint8_t*              pusch_buffer,
	int                   pusch_prb_x_slot,
	int                   pusch_prb_x_symbol,
	int                   pusch_prb_x_symbol_x_antenna,
	uint32_t              pusch_prb_stride,
	uint32_t*             pusch_ordered_prbs,

	uint16_t*             prach_eAxC_map,
	int                   prach_eAxC_num,
	// uint8_t*              prach_buffer,
	// uint16_t              prach_section_id,

    uint8_t * prach_buffer_0,
    uint8_t * prach_buffer_1,
    uint8_t * prach_buffer_2,
    uint8_t * prach_buffer_3,

    uint16_t prach_section_id_0,
    uint16_t prach_section_id_1,
    uint16_t prach_section_id_2,
    uint16_t prach_section_id_3,

	int                   prach_prb_x_slot,
	int                   prach_prb_x_symbol,
	int                   prach_prb_x_symbol_x_antenna,
	uint32_t              prach_prb_stride,
	uint32_t*             prach_ordered_prbs
    )
    {
	if(
	    (pusch_buffer == nullptr || pusch_prb_x_slot == 0) &&
	    ((prach_buffer_0 == nullptr && prach_buffer_1 == nullptr && prach_buffer_2 == nullptr && prach_buffer_3 == nullptr) || prach_prb_x_slot == 0)
	)
	    return EINVAL;

	if(
	    (pusch_buffer != nullptr && pusch_prb_x_slot == 0) ||
	    ( (prach_buffer_0 != nullptr || prach_buffer_1 != nullptr || prach_buffer_2 != nullptr || prach_buffer_3 != nullptr) && prach_prb_x_slot == 0)
	)
	    return EINVAL;

	int pusch_symbols_x_slot = ORAN_PUSCH_SYMBOLS_X_SLOT;
	int prach_b4_symbols_x_slot = ORAN_PRACH_B4_SYMBOLS_X_SLOT;

        // block 0 to receive, block 1 to process
	void* args[] = {
	    /* DOCA objects */
	    &doca_rxq, &sem_gpu, (void*)&sem_order_num,
	    /* Cell specific */
	    &cell_id, &ru_type, &start_cuphy_d, &order_kernel_exit_cond_d, &last_sem_idx_rx_h, &last_sem_idx_order_h, &comp_meth, &bit_width, &beta, &prb_size,
	    /* Timeout */
	    &timeout_no_pkt_ns, &timeout_first_pkt_ns, &max_rx_pkts,
	    /* Time specific */
	    &frameId, &subframeId, &slotId,
	    /* Order kernel specific */
	    &barrier_flag, &done_shared,
	    &early_rx_packets, &on_time_rx_packets, &late_rx_packets,
	    &slot_start, &ta4_min_ns, &ta4_max_ns, &slot_duration,
	    /* PUSCH Output buffer specific */
	    &pusch_eAxC_map, &pusch_eAxC_num,
	    &pusch_buffer, &pusch_prb_x_slot, &pusch_prb_x_symbol, &pusch_prb_x_symbol_x_antenna,
	    &pusch_symbols_x_slot, &pusch_prb_stride, &pusch_ordered_prbs,
	    /* PRACH Output buffer specific */
	    &prach_eAxC_map, &prach_eAxC_num,
	    &prach_buffer_0, &prach_buffer_1, &prach_buffer_2, &prach_buffer_3,
	    &prach_section_id_0, &prach_section_id_1, &prach_section_id_2, &prach_section_id_3,
	    &prach_prb_x_slot, &prach_prb_x_symbol, &prach_prb_x_symbol_x_antenna,
	    &prach_b4_symbols_x_slot, &prach_prb_stride, &prach_ordered_prbs
	};

	CUresult cuResult = cuLaunchKernel(func,
	    2, 1, 1,
	    512, 1, 1,
	    0, stream,
	    args, nullptr);
	CUDA_DRIVER_CHECK_NON_FATAL(cuResult);
	if(cuResult != CUDA_SUCCESS)
	    return -1;

	return 0;
}

int launch_order_kernel_doca_single(
	CUfunction            func,
	cudaStream_t          stream,

    struct doca_gpu_eth_rxq **doca_rxq,
	struct doca_gpu_semaphore_gpu **sem_gpu,
	const uint16_t* sem_order_num,

    int*                   cell_id,
    int*                   ru_type,
    bool*                  cell_health,

	uint32_t**             start_cuphy_d,
	uint32_t**             order_kernel_exit_cond_d,
	uint32_t		**last_sem_idx_rx_h,
    uint32_t		**last_sem_idx_order_h,
	uint32_t              timeout_no_pkt_ns,
	uint32_t              timeout_first_pkt_ns,
    uint32_t              timeout_log_interval_ns,
    uint8_t               timeout_log_enable,
    uint32_t              max_rx_pkts,
    uint32_t              rx_pkts_timeout_ns,
    bool                  commViaCpu,

	uint8_t               frameId,
	uint8_t               subframeId,
	uint8_t               slotId,
	int*                   comp_meth,
    int*                   bit_width,
	int                   prb_size,
	float*                 beta,
	int*                  barrier_flag,
	uint8_t**              done_shared,

	uint32_t**             early_rx_packets,
	uint32_t**             on_time_rx_packets,
	uint32_t**             late_rx_packets,
	uint32_t**             next_slot_early_rx_packets,
	uint32_t**             next_slot_on_time_rx_packets,
	uint32_t**             next_slot_late_rx_packets,	
	uint64_t*              slot_start,
	uint64_t*              ta4_min_ns,
	uint64_t*              ta4_max_ns,
	uint64_t*              slot_duration,
    uint64_t**             order_kernel_last_timeout_error_time,
    uint8_t                ul_rx_pkt_tracing_level,
	uint64_t**             rx_packets_ts,
	uint32_t**             rx_packets_count,
    uint32_t**             rx_bytes_count,
    uint64_t**             rx_packets_ts_earliest,
    uint64_t**             rx_packets_ts_latest,
	uint64_t**             next_slot_rx_packets_ts,
	uint32_t**             next_slot_rx_packets_count,
    uint32_t**             next_slot_rx_bytes_count,
	uint32_t**             next_slot_num_prb_ch1,
	uint32_t**             next_slot_num_prb_ch2,

	uint16_t**             pusch_eAxC_map,
	int*                   pusch_eAxC_num,
	uint8_t**              pusch_buffer,
	int*                   pusch_prb_x_slot,
	int*                   pusch_prb_x_symbol,
	int*                   pusch_prb_x_symbol_x_antenna,
	uint32_t*              pusch_prb_stride,
	uint32_t**             pusch_ordered_prbs,

	uint16_t**             prach_eAxC_map,
	int*                   prach_eAxC_num,
	// uint8_t*              prach_buffer,
	// uint16_t              prach_section_id,

    uint8_t** prach_buffer_0,
    uint8_t** prach_buffer_1,
    uint8_t** prach_buffer_2,
    uint8_t** prach_buffer_3,

    uint16_t prach_section_id_0,
    uint16_t prach_section_id_1,
    uint16_t prach_section_id_2,
    uint16_t prach_section_id_3,

	int*                   prach_prb_x_slot,
	int*                   prach_prb_x_symbol,
	int*                   prach_prb_x_symbol_x_antenna,
	uint32_t*              prach_prb_stride,
	uint32_t**             prach_ordered_prbs,
	uint8_t num_order_cells,
    uint8_t** pcap_buffer,
    uint8_t** pcap_buffer_ts,
    uint32_t** pcap_buffer_index,
    uint8_t pcap_capture_enable,
    uint64_t pcap_capture_cell_bitmask,
    uint16_t max_pkt_size
    )
    {
	int cudaBlocks = (num_order_cells); //# of Thread blocks should be twice the number of cells
    int numThreads = (commViaCpu==true)?256:128;

	int pusch_symbols_x_slot = ORAN_PUSCH_SYMBOLS_X_SLOT;
	int prach_b4_symbols_x_slot = ORAN_PRACH_B4_SYMBOLS_X_SLOT;

        // block 0 to receive, block 1 to process
	void* args[] = {
	    /* DOCA objects */
	    &doca_rxq, &sem_gpu, &sem_order_num,
	    /* Cell specific */
	    &cell_id, &ru_type, &cell_health, &start_cuphy_d, &order_kernel_exit_cond_d, &last_sem_idx_rx_h, &last_sem_idx_order_h, &comp_meth, &bit_width, &beta, &prb_size,
	    /* Timeout */
	    &timeout_no_pkt_ns, &timeout_first_pkt_ns, &timeout_log_interval_ns, &timeout_log_enable, &max_rx_pkts, &rx_pkts_timeout_ns, &commViaCpu,
	    /* Time specific */
	    &frameId, &subframeId, &slotId,
	    /* Order kernel specific */
	    &barrier_flag, &done_shared,
	    &early_rx_packets, &on_time_rx_packets, &late_rx_packets, &next_slot_early_rx_packets, &next_slot_on_time_rx_packets, &next_slot_late_rx_packets,
	    &slot_start, &ta4_min_ns, &ta4_max_ns, &slot_duration, &order_kernel_last_timeout_error_time, &ul_rx_pkt_tracing_level, &rx_packets_ts, &rx_packets_count, &rx_bytes_count, &rx_packets_ts_earliest, &rx_packets_ts_latest, &next_slot_rx_packets_ts, &next_slot_rx_packets_count, &next_slot_rx_bytes_count,
	    &next_slot_num_prb_ch1, &next_slot_num_prb_ch2,
	    /* PUSCH Output buffer specific */
	    &pusch_eAxC_map, &pusch_eAxC_num,
	    &pusch_buffer, &pusch_prb_x_slot, &pusch_prb_x_symbol, &pusch_prb_x_symbol_x_antenna,
	    &pusch_symbols_x_slot, &pusch_prb_stride, &pusch_ordered_prbs,
	    /* PRACH Output buffer specific */
	    &prach_eAxC_map, &prach_eAxC_num,
	    &prach_buffer_0, &prach_buffer_1, &prach_buffer_2, &prach_buffer_3,
	    &prach_section_id_0, &prach_section_id_1, &prach_section_id_2, &prach_section_id_3,
	    &prach_prb_x_slot, &prach_prb_x_symbol, &prach_prb_x_symbol_x_antenna,
	    &prach_b4_symbols_x_slot, &prach_prb_stride, &prach_ordered_prbs,
	    &pcap_buffer, &pcap_buffer_ts, &pcap_buffer_index, &pcap_capture_enable, &pcap_capture_cell_bitmask, &max_pkt_size
	};

	CUresult cuResult = cuLaunchKernel(func,
	    cudaBlocks * 2, 1, 1,
	    numThreads, 1, 1,
	    0, stream,
	    args, nullptr);
	CUDA_DRIVER_CHECK_NON_FATAL(cuResult);
	if(cuResult != CUDA_SUCCESS)
	    return -1;

	return 0;
}

int launch_order_kernel_doca_single_srs(
	CUfunction            func,
	cudaStream_t          stream,

    	struct doca_gpu_eth_rxq **doca_rxq,
	struct doca_gpu_semaphore_gpu **sem_gpu,
	const uint16_t* sem_order_num,

    int*                   cell_id,
    int*                   ru_type,

	uint32_t**             start_cuphy_d,
	uint32_t**             order_kernel_exit_cond_d,
	uint32_t		**last_sem_idx_rx_h,
    uint32_t		**last_sem_idx_order_h,
	uint32_t              timeout_no_pkt_ns,
	uint32_t              timeout_first_pkt_ns,
    uint32_t              max_rx_pkts,

	uint8_t               frameId,
	uint8_t               subframeId,
	uint8_t               slotId,
	int*                   comp_meth,
    int*                   bit_width,
	int                   prb_size,
	float*                 beta,
	int*                  barrier_flag,
	uint8_t**              done_shared,
	uint32_t**             early_rx_packets_srs,
	uint32_t**             on_time_rx_packets_srs,
	uint32_t**             late_rx_packets_srs,
        uint32_t**             next_slot_early_rx_packets_srs,
	uint32_t**             next_slot_on_time_rx_packets_srs,
	uint32_t**             next_slot_late_rx_packets_srs,
    uint32_t**             rx_packets_count_srs,
    uint32_t**             rx_bytes_count_srs,
    uint32_t**             next_slot_rx_packets_count_srs,
    uint32_t**             next_slot_rx_bytes_count_srs,
    uint8_t                ul_rx_pkt_tracing_level,
    uint64_t**             rx_packets_ts_srs,
    uint32_t**             rx_packets_count_per_sym_srs,
    uint64_t**             rx_packets_ts_earliest_srs,
    uint64_t**             rx_packets_ts_latest_srs,	
    uint64_t**             next_slot_rx_packets_ts_srs,
    uint32_t**             next_slot_rx_packets_count_per_sym_srs,      
	uint64_t*              slot_start_srs,
	uint64_t*              ta4_min_ns,
	uint64_t*              ta4_max_ns,
        uint64_t*              slot_duration,
	uint16_t**             srs_eAxC_map,
	int*                   srs_eAxC_num,
	uint8_t**              srs_buffer,
	int*                   srs_prb_x_slot,
	uint32_t*              srs_prb_stride,
	uint32_t**             srs_ordered_prbs,
    uint8_t*                srs_start_sym,
	uint8_t num_order_cells
    )
    {
	int cudaBlocks = (num_order_cells); //# of Thread blocks should be twice the number of cells

	int oran_max_srs_symbols = ORAN_MAX_SRS_SYMBOLS;

        // block 0 to receive, block 1 to process
	void* args[] = {
	    /* DOCA objects */
	    &doca_rxq, &sem_gpu, &sem_order_num,
	    /* Cell specific */
	    &cell_id, &ru_type, &start_cuphy_d, &order_kernel_exit_cond_d, &last_sem_idx_rx_h, &last_sem_idx_order_h, &comp_meth, &bit_width, &beta, &prb_size,
	    /* Timeout */
	    &timeout_no_pkt_ns, &timeout_first_pkt_ns, &max_rx_pkts,
	    /* Time specific */
	    &frameId, &subframeId, &slotId,
	    /* Order kernel specific */
	    &barrier_flag, &done_shared,
	    /* SRS packet stats */
	    &early_rx_packets_srs, &on_time_rx_packets_srs, &late_rx_packets_srs,
	    &next_slot_early_rx_packets_srs, &next_slot_on_time_rx_packets_srs, &next_slot_late_rx_packets_srs,
	    &rx_packets_count_srs, &rx_bytes_count_srs, &next_slot_rx_packets_count_srs, &next_slot_rx_bytes_count_srs,
	    &ul_rx_pkt_tracing_level, &rx_packets_ts_srs, &rx_packets_count_per_sym_srs, &rx_packets_ts_earliest_srs, &rx_packets_ts_latest_srs, &next_slot_rx_packets_ts_srs, &next_slot_rx_packets_count_per_sym_srs,
	    &slot_start_srs, &ta4_min_ns, &ta4_max_ns, &slot_duration,
	    /* SRS Output buffer specific */
	    &srs_eAxC_map, &srs_eAxC_num,
	    &srs_buffer, &srs_prb_x_slot,
	    &oran_max_srs_symbols, &srs_prb_stride, &srs_ordered_prbs, &srs_start_sym
	};

	CUresult cuResult = cuLaunchKernel(func,
	    cudaBlocks, 1, 1,
	    256, 1, 1,
	    0, stream,
	    args, nullptr);
	CUDA_DRIVER_CHECK_NON_FATAL(cuResult);
	if(cuResult != CUDA_SUCCESS)
	    return -1;

	return 0;
}

int launch_order_kernel_doca_single_subSlot(
	CUfunction func,
	CUfunction pingpong_trace_srs,
	CUfunction pingpong_trace_srs_pusch,
	CUfunction pingpong_trace_no_srs,
	CUfunction pingpong_no_trace_srs,
	CUfunction pingpong_no_trace_srs_pusch,
	CUfunction pingpong_no_trace_no_srs,
	cudaStream_t stream,

	struct doca_gpu_eth_rxq **doca_rxq,
	struct doca_gpu_semaphore_gpu **sem_gpu,
	struct aerial_fh_gpu_semaphore_gpu **sem_gpu_aerial_fh,
	const uint16_t* sem_order_num,

	int*                   cell_id,
    int*                   ru_type,
    bool*                  cell_health,

	uint32_t**             start_cuphy_d,
	uint32_t**             order_kernel_exit_cond_d,
	uint32_t		**last_sem_idx_rx_h,
	uint32_t		**last_sem_idx_order_h,
    uint8_t               ul_order_kernel_mode,
	uint32_t              timeout_no_pkt_ns,
	uint32_t              timeout_first_pkt_ns,
	uint32_t              timeout_log_interval_ns,
	uint8_t               timeout_log_enable,
	uint32_t              max_rx_pkts,
    uint32_t              rx_pkts_timeout_ns,
    bool                  commViaCpu,

	uint8_t               frameId,
	uint8_t               subframeId,
	uint8_t               slotId,
	int*                   comp_meth,
    int*                   bit_width,
	int                   prb_size,
	float*                 beta,
	int*                  barrier_flag,
	uint8_t**              done_shared,

	uint32_t**             early_rx_packets,
	uint32_t**             on_time_rx_packets,
	uint32_t**             late_rx_packets,
	uint32_t**             next_slot_early_rx_packets,
	uint32_t**             next_slot_on_time_rx_packets,
	uint32_t**             next_slot_late_rx_packets,		
	uint64_t*              slot_start,
	uint64_t*              ta4_min_ns,
	uint64_t*              ta4_max_ns,
	uint64_t*              slot_duration,
    uint64_t**             order_kernel_last_timeout_error_time,
    uint8_t                ul_rx_pkt_tracing_level,
	uint64_t**             rx_packets_ts,
	uint32_t**              rx_packets_count,
    uint32_t**              rx_bytes_count,
    uint64_t**             rx_packets_ts_earliest,
    uint64_t**             rx_packets_ts_latest,	
	uint64_t**             next_slot_rx_packets_ts,
	uint32_t**             next_slot_rx_packets_count,
    uint32_t**             next_slot_rx_bytes_count,
    uint32_t**             rx_packets_dropped_count,
	uint32_t**             next_slot_num_prb_ch1,
	uint32_t**             next_slot_num_prb_ch2,

    uint32_t* 			   sym_ord_done_sig_arr,
    uint32_t*              sym_ord_done_mask_arr,
    uint32_t*              pusch_prb_symbol_map,
	uint32_t* 			   num_order_cells_sym_mask_arr,
    uint8_t                pusch_prb_non_zero,

	uint16_t**             pusch_eAxC_map,
	int*                   pusch_eAxC_num,
	uint8_t**              pusch_buffer,
	int*                   pusch_prb_x_slot,
	int*                   pusch_prb_x_symbol,
	int*                   pusch_prb_x_symbol_x_antenna,
	uint32_t*              pusch_prb_stride,
	uint32_t**             pusch_ordered_prbs,


	uint16_t**             prach_eAxC_map,
	int*                   prach_eAxC_num,
	// uint8_t*              prach_buffer,
	// uint16_t              prach_section_id,

    uint8_t** prach_buffer_0,
    uint8_t** prach_buffer_1,
    uint8_t** prach_buffer_2,
    uint8_t** prach_buffer_3,

    uint16_t prach_section_id_0,
    uint16_t prach_section_id_1,
    uint16_t prach_section_id_2,
    uint16_t prach_section_id_3,

	int*                   prach_prb_x_slot,
	int*                   prach_prb_x_symbol,
	int*                   prach_prb_x_symbol_x_antenna,
	uint32_t*              prach_prb_stride,
	uint32_t**             prach_ordered_prbs,

	uint16_t		**srs_eAxC_map,
	int*			srs_eAxC_num,
	uint8_t		**srs_buffer,
	int*			srs_prb_x_slot,
    uint32_t*              srs_prb_stride,	
    uint32_t		**srs_ordered_prbs,
    uint8_t*          srs_start_sym,

	uint8_t num_order_cells,
    uint8_t** pcap_buffer,
    uint8_t** pcap_buffer_ts,
    uint32_t** pcap_buffer_index,
    uint8_t pcap_capture_enable,
    uint64_t pcap_capture_cell_bitmask,
    uint16_t max_pkt_size,
    uint8_t srs_enable
    )
    {
	int cudaBlocks = (num_order_cells); //# of Thread blocks should be twice the number of cells

    if(ul_order_kernel_mode == 0) {
        order_kernel_pkt_tracing_info pkt_tracing_info = {
            .rx_packets_count = rx_packets_count,
            .rx_bytes_count = rx_bytes_count,
            .next_slot_rx_packets_count = next_slot_rx_packets_count,
            .next_slot_rx_bytes_count = next_slot_rx_bytes_count,
            .rx_packets_ts_earliest = rx_packets_ts_earliest,
            .rx_packets_ts_latest = rx_packets_ts_latest,
            .rx_packets_ts = rx_packets_ts,
            .next_slot_rx_packets_ts = next_slot_rx_packets_ts,
        };

        CUfunction pp_func{};
        int pp_threads;
        uint16_t** pp_srs_eAxC_map_arg = srs_eAxC_map;
        int* pp_srs_eAxC_num_arg = srs_eAxC_num;

        if(ul_rx_pkt_tracing_level) {
            if(srs_enable == ORDER_KERNEL_SRS_ENABLE) {
                pp_func = pingpong_trace_srs;
                pp_threads = ORDER_KERNEL_PINGPONG_SRS_NUM_THREADS;
            } else if(srs_enable == ORDER_KERNEL_SRS_AND_PUSCH) {
                pp_func = pingpong_trace_srs_pusch;
                pp_threads = ORDER_KERNEL_PINGPONG_SRS_NUM_THREADS;
                pp_srs_eAxC_map_arg = pusch_eAxC_map;
                pp_srs_eAxC_num_arg = pusch_eAxC_num;
            } else {
                pp_func = pingpong_trace_no_srs;
                pp_threads = ORDER_KERNEL_PINGPONG_NUM_THREADS;
                pp_srs_eAxC_map_arg = pusch_eAxC_map;
                pp_srs_eAxC_num_arg = pusch_eAxC_num;
            }
        } else {
            if(srs_enable == ORDER_KERNEL_SRS_ENABLE) {
                pp_func = pingpong_no_trace_srs;
                pp_threads = ORDER_KERNEL_PINGPONG_SRS_NUM_THREADS;
            } else if(srs_enable == ORDER_KERNEL_SRS_AND_PUSCH) {
                pp_func = pingpong_no_trace_srs_pusch;
                pp_threads = ORDER_KERNEL_PINGPONG_SRS_NUM_THREADS;
                pp_srs_eAxC_map_arg = pusch_eAxC_map;
            } else {
                pp_func = pingpong_no_trace_no_srs;
                pp_threads = ORDER_KERNEL_PINGPONG_NUM_THREADS;
            }
        }

        int pusch_symbols_x_slot = ORAN_PUSCH_SYMBOLS_X_SLOT;
        int prach_b4_symbols_x_slot = ORAN_PRACH_B4_SYMBOLS_X_SLOT;
        int srs_symbols_x_slot = ORAN_MAX_SRS_SYMBOLS;
        uint8_t** tb_fh_buf_null = nullptr;
        uint32_t* rx_pkt_num_slot_null = nullptr;

        void* pp_args[] = {
            /* DOCA objects */
            &doca_rxq, &sem_gpu, &sem_gpu_aerial_fh, &sem_order_num,
            /* Cell specific */
            &cell_id, &ru_type, &cell_health, &start_cuphy_d, &order_kernel_exit_cond_d, &last_sem_idx_rx_h, &last_sem_idx_order_h, &comp_meth, &bit_width, &beta, &prb_size,
            /* Timeout */
            &timeout_no_pkt_ns, &timeout_first_pkt_ns, &timeout_log_interval_ns, &timeout_log_enable, &max_rx_pkts, &rx_pkts_timeout_ns,
            /* Time specific */
            &frameId, &subframeId, &slotId,
            /* Order kernel specific */
            &early_rx_packets, &on_time_rx_packets, &late_rx_packets, &next_slot_early_rx_packets, &next_slot_on_time_rx_packets, &next_slot_late_rx_packets,
            &slot_start, &ta4_min_ns, &ta4_max_ns, &slot_duration, &order_kernel_last_timeout_error_time, &pkt_tracing_info, &rx_packets_dropped_count,
            /* Sub-slot processing specific */
            &sym_ord_done_sig_arr, &sym_ord_done_mask_arr, &pusch_prb_symbol_map, &num_order_cells_sym_mask_arr, &pusch_prb_non_zero,
            /* PUSCH Output buffer specific */
            &pusch_eAxC_map, &pusch_eAxC_num, &pusch_buffer, &pusch_prb_x_slot,
            &pusch_symbols_x_slot, &pusch_prb_stride, &pusch_ordered_prbs,
            /* PRACH Output buffer specific */
            &prach_eAxC_map, &prach_eAxC_num,
            &prach_buffer_0, &prach_buffer_1, &prach_buffer_2, &prach_buffer_3,
            &prach_section_id_0, &prach_section_id_1, &prach_section_id_2, &prach_section_id_3,
            &prach_prb_x_slot,
            &prach_b4_symbols_x_slot, &prach_prb_stride, &prach_ordered_prbs,
            /* SRS Output buffer specific */
            &pp_srs_eAxC_map_arg, &pp_srs_eAxC_num_arg, &srs_buffer, &srs_prb_x_slot, &srs_symbols_x_slot, &srs_prb_stride, &srs_ordered_prbs, &srs_start_sym,
            &num_order_cells,
            /* PCAP Capture specific */
            &pcap_buffer, &pcap_buffer_ts, &pcap_buffer_index, &pcap_capture_enable, &pcap_capture_cell_bitmask,
            /* Test bench values */
            &tb_fh_buf_null, &max_pkt_size, &rx_pkt_num_slot_null
        };

        MemtraceDisableScope md;
        CUresult cuResult = cuLaunchKernel(pp_func,
            cudaBlocks, 1, 1,
            pp_threads, 1, 1,
            0, stream,
            pp_args, nullptr);
        CUDA_DRIVER_CHECK_NON_FATAL(cuResult);
        if(cuResult != CUDA_SUCCESS)
            return -1;
    }
    else if(ul_order_kernel_mode == 1) {
        // Dual CTA mode
            // block 0 to receive, block 1 to process
        const int numThreads = (commViaCpu==true)?256:128;
        int pusch_symbols_x_slot = ORAN_PUSCH_SYMBOLS_X_SLOT;
        int prach_b4_symbols_x_slot = ORAN_PRACH_B4_SYMBOLS_X_SLOT;

        void* args[] = {
            &doca_rxq, &sem_gpu, &sem_order_num,
            &cell_id, &ru_type, &cell_health, &start_cuphy_d, &order_kernel_exit_cond_d, &last_sem_idx_rx_h, &last_sem_idx_order_h, &comp_meth, &bit_width, &beta, &prb_size,
            &timeout_no_pkt_ns, &timeout_first_pkt_ns, &timeout_log_interval_ns, &timeout_log_enable, &max_rx_pkts, &rx_pkts_timeout_ns, &commViaCpu,
            &frameId, &subframeId, &slotId,
            &barrier_flag, &done_shared,
            &early_rx_packets, &on_time_rx_packets, &late_rx_packets, &next_slot_early_rx_packets, &next_slot_on_time_rx_packets, &next_slot_late_rx_packets,
            &slot_start, &ta4_min_ns, &ta4_max_ns, &slot_duration, &order_kernel_last_timeout_error_time, &ul_rx_pkt_tracing_level, &rx_packets_ts, &rx_packets_count, &rx_bytes_count, &rx_packets_ts_earliest, &rx_packets_ts_latest, &next_slot_rx_packets_ts, &next_slot_rx_packets_count, &next_slot_rx_bytes_count,
            &next_slot_num_prb_ch1, &next_slot_num_prb_ch2,
            &sym_ord_done_sig_arr, &sym_ord_done_mask_arr, &pusch_prb_symbol_map, &num_order_cells_sym_mask_arr, &pusch_prb_non_zero,
            &pusch_eAxC_map, &pusch_eAxC_num,
            &pusch_buffer, &pusch_prb_x_slot, &pusch_prb_x_symbol, &pusch_prb_x_symbol_x_antenna,
            &pusch_symbols_x_slot, &pusch_prb_stride, &pusch_ordered_prbs,
            &prach_eAxC_map, &prach_eAxC_num,
            &prach_buffer_0, &prach_buffer_1, &prach_buffer_2, &prach_buffer_3,
            &prach_section_id_0, &prach_section_id_1, &prach_section_id_2, &prach_section_id_3,
            &prach_prb_x_slot, &prach_prb_x_symbol, &prach_prb_x_symbol_x_antenna,
            &prach_b4_symbols_x_slot, &prach_prb_stride, &prach_ordered_prbs, &num_order_cells,
            &pcap_buffer, &pcap_buffer_ts, &pcap_buffer_index, &pcap_capture_enable, &pcap_capture_cell_bitmask, &max_pkt_size
        };

        CUresult cuResult = cuLaunchKernel(func,
            cudaBlocks * 2, 1, 1,
            numThreads, 1, 1,
            0, stream,
            args, nullptr);
        CUDA_DRIVER_CHECK_NON_FATAL(cuResult);
        if(cuResult != CUDA_SUCCESS)
            return -1;
    }
    else {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT,"Invalid UL Order Kernel Mode: {}", ul_order_kernel_mode);
        return -1;
    }

	return 0;
}

    
int launch_order_kernel_cpu_init_comms_single_subSlot(
CUfunction func,
cudaStream_t stream,

uint32_t**             start_cuphy_d,
uint32_t**             order_kernel_exit_cond_d,
uint32_t**             ready_list,
struct aerial_fh::rx_queue_sync** rx_queue_sync_list,
uint32_t**                  last_ordered_item_h,
uint16_t*              sem_order_num,

uint8_t               frameId,
uint8_t               subframeId,
uint8_t               slotId,
int*                   comp_meth,
int*                   bit_width,
int                   prb_size,
float*                 beta,
int*                  barrier_flag,
uint8_t**              done_shared,

uint32_t              timeout_no_pkt_ns,
uint32_t              timeout_first_pkt_ns,

uint32_t*              sym_ord_done_sig_arr,
uint32_t*              sym_ord_done_mask_arr,
uint32_t*              pusch_prb_symbol_map,
uint32_t*              num_order_cells_sym_mask_arr,

/* Timer */
uint32_t 		**early_rx_packets,
uint32_t 		**on_time_rx_packets,
uint32_t 		**late_rx_packets,
uint32_t 		**next_slot_early_rx_packets,
uint32_t 		**next_slot_on_time_rx_packets,
uint32_t 		**next_slot_late_rx_packets,	
uint64_t*		slot_start,
uint64_t*		ta4_min_ns,
uint64_t*		ta4_max_ns,
uint64_t*		slot_duration,
uint8_t                ul_rx_pkt_tracing_level,
uint64_t**             rx_packets_ts,
uint32_t**              rx_packets_count,
uint32_t**              rx_bytes_count,
uint64_t**             rx_packets_ts_earliest,
uint64_t**             rx_packets_ts_latest,	
uint64_t**             next_slot_rx_packets_ts,
uint32_t**             next_slot_rx_packets_count,
uint32_t**             next_slot_rx_bytes_count,

uint16_t**             pusch_eAxC_map,
int*                   pusch_eAxC_num,
uint8_t**              pusch_buffer,
int*                   pusch_prb_x_slot,
int*                   pusch_prb_x_symbol,
int*                   pusch_prb_x_symbol_x_antenna,
uint32_t*              pusch_prb_stride,
uint32_t**             pusch_ordered_prbs,

uint16_t**             prach_eAxC_map,
int*                   prach_eAxC_num,
// uint8_t*              prach_buffer,
// uint16_t              prach_section_id,

uint8_t** prach_buffer_0,
uint8_t** prach_buffer_1,
uint8_t** prach_buffer_2,
uint8_t** prach_buffer_3,

uint16_t prach_section_id_0,
uint16_t prach_section_id_1,
uint16_t prach_section_id_2,
uint16_t prach_section_id_3,

int*                   prach_prb_x_slot,
int*                   prach_prb_x_symbol,
int*                   prach_prb_x_symbol_x_antenna,
uint32_t*              prach_prb_stride,
uint32_t**             prach_ordered_prbs,
uint8_t num_order_cells
)
{
	int cudaBlocks = (num_order_cells); //# of Thread blocks should be equal to the num of cells
	int pusch_symbols_x_slot = ORAN_PUSCH_SYMBOLS_X_SLOT;
	int prach_b4_symbols_x_slot = ORAN_PRACH_B4_SYMBOLS_X_SLOT;

	void* args[] = {
	    /* Cell specific */
	    &start_cuphy_d, &order_kernel_exit_cond_d,
	    /* Rx objects */
	    &ready_list, &rx_queue_sync_list, &last_ordered_item_h, &sem_order_num,
	    /* Time specific */
	    &frameId, &subframeId, &slotId, &comp_meth, &bit_width, &prb_size, &beta, &barrier_flag, &done_shared,
	    &timeout_no_pkt_ns, &timeout_first_pkt_ns,
	    /*sub-slot processing specific*/
	    &sym_ord_done_sig_arr, &sym_ord_done_mask_arr, &pusch_prb_symbol_map, &num_order_cells_sym_mask_arr,
	    /*Timer*/
	    &early_rx_packets, &on_time_rx_packets, &late_rx_packets, &next_slot_early_rx_packets, &next_slot_on_time_rx_packets, &next_slot_late_rx_packets,
	    &slot_start, &ta4_min_ns, &ta4_max_ns, &slot_duration, &ul_rx_pkt_tracing_level, &rx_packets_ts, &rx_packets_count, &rx_bytes_count, &rx_packets_ts_earliest, &rx_packets_ts_latest, &next_slot_rx_packets_ts, &next_slot_rx_packets_count, &next_slot_rx_bytes_count,
	    /* PUSCH Output buffer specific */
	    &pusch_eAxC_map, &pusch_eAxC_num,
	    &pusch_buffer, &pusch_prb_x_slot, &pusch_prb_x_symbol, &pusch_prb_x_symbol_x_antenna,
	    &pusch_symbols_x_slot, &pusch_prb_stride, &pusch_ordered_prbs,
	    /* PRACH Output buffer specific */
	    &prach_eAxC_map, &prach_eAxC_num,
	    &prach_buffer_0, &prach_buffer_1, &prach_buffer_2, &prach_buffer_3,
	    &prach_section_id_0, &prach_section_id_1, &prach_section_id_2, &prach_section_id_3,
	    &prach_prb_x_slot, &prach_prb_x_symbol, &prach_prb_x_symbol_x_antenna,
	    &prach_b4_symbols_x_slot, &prach_prb_stride, &prach_ordered_prbs, &num_order_cells
	};

	CUresult cuResult = cuLaunchKernel(func,
	    cudaBlocks, 1, 1,
	    256, 1, 1,
	    0, stream,
	    args, nullptr);
	CUDA_DRIVER_CHECK_NON_FATAL(cuResult);
	if(cuResult != CUDA_SUCCESS)
	    return -1;

    return 0;
}

/* OLD VERSION */

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// Order Kernel single block
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void kernel_order(
    int fake_run, int cell_id, //debug reason, should be an array when  moving to batching
    uint8_t*              order_kernel_end_cuphy_d,
    uint8_t*              order_completed_h,
    uint32_t*                  order_start_kernel_d,
    uint32_t*             ready_list,
    struct aerial_fh::rx_queue_sync* rx_queue_sync_list,
    int*                  last_ordered_item_h, //With batching, this should be a list one item per cell
    uint8_t*              buffer,
    int                   prb_x_slot,
    int                   prb_x_symbol,
    int                   prb_x_symbol_x_antenna,
    uint8_t               frameId,
    uint8_t               subframeId,
    uint8_t               slotId,
    uint16_t*             eAxC_map,
    int                   eAxC_num,
    int                   comp_meth,
    int                   bit_width,
    int                   prb_size,
    float                 beta)
{
    int                rx_queue_index = -1, ready_local = 0, last_queue_index = 0, index_copy = 0, index_mbuf = 0;
    int                laneId       = threadIdx.x % 32;
    int                warpId       = threadIdx.x / 32;
    int                nwarps       = blockDim.x / 32;
    unsigned long long start        = 0;
    unsigned long long kernel_start = __globaltimer(), current_time = 0;

    uint8_t frameId_pkt, subframeId_pkt, slotId_pkt, *pkt_offset_ptr, *gbuf_offset_ptr, start_loop = 0;

    __shared__ int ready_shared[1];
    __shared__ int rx_queue_index_s[1];
    __shared__ uint8_t done_shared[1];
    __shared__ uint16_t msg_prb[CK_ORDER_PKTS_BUFFERING];
    __shared__ uintptr_t msg_addr[CK_ORDER_PKTS_BUFFERING];
    __shared__ uint32_t gbuf_offset[CK_ORDER_PKTS_BUFFERING];
    __shared__ int      ordered_prbs[1]; //FIXME in case of multiple CUDA blocks, let's move this into vidmem

    if(fake_run == 1)
        return;

    done_shared[0]  = 1;
    ordered_prbs[0] = 0;

    __syncthreads();

    while(1)
    {
        if(threadIdx.x == 0)
        {
            while(1)
            {
                //Should we add a global timer despite of the first packet? Assuming LSU always sends them
                current_time = __globaltimer();

                if((start_loop == 1 && ((current_time - kernel_start) > ORDER_KERNEL_WAIT_TIMEOUT_MS * NS_X_MS))) //8ms max timeout for kernel to wait for packets
                {
                    printf("Order kernel wait timeout after 8 ms\n");
                    ready_shared[0] = (int)SYNC_PACKET_STATUS_EXIT;
                    __threadfence_block();
                    break;
                }

                if((start_loop == 2 && ((current_time - start) > ORDER_KERNEL_RECV_TIMEOUT_MS * NS_X_MS))) //4ms max timeout for receiving packets
                {
                    printf("Order kernel recv timeout after 4 ms\n");
                    ready_shared[0] = (int)SYNC_PACKET_STATUS_EXIT;
                    __threadfence_block();
                    break;
                }

                if(start_loop == 0)
                {
                    rx_queue_index = ACCESS_ONCE(order_start_kernel_d[0]);
                    if(rx_queue_index != -1)
                    {
                        rx_queue_index   = last_ordered_item_h[0];
                        last_queue_index = rx_queue_index;
                        start_loop       = 1;
                    }
                }
                else
                {
                    // printf("Order kernel cell %d polling on item %d, ordered %d/%d addr %lx\n", cell_id, rx_queue_index, ordered_prbs, prb_x_slot, &ready_list[rx_queue_index]);
                    ready_local = ACCESS_ONCE(ready_list[rx_queue_index]);
                    if(ready_local == SYNC_PACKET_STATUS_READY)
                    {
                        if(start_loop == 1)
                        {
                            // start_flag[0] = PT_SLOT_START;
                            // __threadfence_system();
                            start_loop = 2;
                            start      = __globaltimer();
                            __threadfence();
                        }

                        ready_shared[0]     = (int)SYNC_PACKET_STATUS_READY;
                        rx_queue_index_s[0] = rx_queue_index;

                        // printf("Order kernel cell %d ready on item %d, ordered %d/%d\n", cell_id, rx_queue_index, ordered_prbs[0], prb_x_slot);

                        __threadfence_block();
                        break;
                    }
                }
            }
        }
        __syncthreads();

        //Exit condition from host
        if(ready_shared[0] != SYNC_PACKET_STATUS_READY)
            goto exit;

        rx_queue_index = rx_queue_index_s[0];

        if(threadIdx.x < CK_ORDER_PKTS_BUFFERING)
        {
            msg_addr[threadIdx.x] = (uintptr_t)ACCESS_ONCE(rx_queue_sync_list[rx_queue_index].addr[threadIdx.x]);
            if(msg_addr[threadIdx.x] != 0)
            {
                frameId_pkt    = oran_umsg_get_frame_id((uint8_t*)msg_addr[threadIdx.x]);
                subframeId_pkt = oran_umsg_get_subframe_id((uint8_t*)msg_addr[threadIdx.x]);
                slotId_pkt     = oran_umsg_get_slot_id((uint8_t*)msg_addr[threadIdx.x]);

                //Previous slot: it's ok to ignore and advance
                if(
                    (frameId_pkt < frameId) ||
                    (frameId_pkt == frameId && subframeId_pkt < subframeId) ||
                    (frameId_pkt == frameId && subframeId_pkt == subframeId && slotId_pkt < slotId))
                {
                    // printf("Order kernel cell %d Item %d pkt %d has previous frame %d/%d subframe %d/%d slot %d/%d\n",
                    //     cell_id, rx_queue_index, threadIdx.x,
                    //     oran_umsg_get_frame_id((uint8_t*)msg_addr[threadIdx.x]), frameId,
                    //     oran_umsg_get_subframe_id((uint8_t*)msg_addr[threadIdx.x]), subframeId,
                    //     oran_umsg_get_slot_id((uint8_t*)msg_addr[threadIdx.x]), slotId
                    // );
                    msg_addr[threadIdx.x] = 0;
                }
                //Next slot: don't advance
                else if(
                    (frameId_pkt > frameId) ||
                    (frameId_pkt == frameId && subframeId_pkt > subframeId) ||
                    (frameId_pkt == frameId && subframeId_pkt == subframeId && slotId_pkt > slotId))
                {
                    // printf("Order kernel cell %d Item %d pkt %d has next frame %d/%d subframe %d/%d slot %d/%d\n",
                    //     cell_id, rx_queue_index, threadIdx.x,
                    //     oran_umsg_get_frame_id((uint8_t*)msg_addr[threadIdx.x]), frameId,
                    //     oran_umsg_get_subframe_id((uint8_t*)msg_addr[threadIdx.x]), subframeId,
                    //     oran_umsg_get_slot_id((uint8_t*)msg_addr[threadIdx.x]), slotId
                    // );
                    msg_addr[threadIdx.x] = 0;
                    done_shared[0]        = 0;
                }
                else
                {
                    msg_prb[threadIdx.x] = oran_umsg_get_num_prb((uint8_t*)msg_addr[threadIdx.x]);
                    if(msg_prb[threadIdx.x] == 0) msg_prb[threadIdx.x] = 273;

                    uint16_t flow_index      = oran_umsg_get_flowid((uint8_t*)msg_addr[threadIdx.x]);
                    gbuf_offset[threadIdx.x] = oran_get_offset_from_hdr(
                        (uint8_t*)msg_addr[threadIdx.x],
                        (uint16_t)get_eaxc_index(eAxC_map, eAxC_num, flow_index),
                        ORAN_PUSCH_SYMBOLS_X_SLOT,
                        ORAN_PUSCH_PRBS_X_PORT_X_SYMBOL, // <--- per antenna
                        prb_size);

                    // CWE-787: discard packets whose untrusted symbolId/startPrb/numPrb would
                    // index past the antenna buffer (consumer skips entries with msg_addr==0).
                    if(!ul_oran_offset_in_bounds((uint8_t*)msg_addr[threadIdx.x],
                                                 (int)oran_umsg_get_start_prb((uint8_t*)msg_addr[threadIdx.x]),
                                                 (int)msg_prb[threadIdx.x],
                                                 ORAN_PUSCH_SYMBOLS_X_SLOT, ORAN_PUSCH_PRBS_X_PORT_X_SYMBOL))
                    {
                        msg_addr[threadIdx.x] = 0;
                        msg_prb[threadIdx.x] = 0;
                        gbuf_offset[threadIdx.x] = 0; // don't leave a stale OOB offset behind
                    }

                    // if(gbuf_offset[threadIdx.x] == 0)
                    //     printf("Offset item %d pkt %d (addr %lx) cell %d pkt %d, flow index %d offset %d flow %d symbol %d startPrb %d numPrb %d\n",
                    //     rx_queue_index, threadIdx.x, msg_addr[threadIdx.x],
                    //     cell_id, threadIdx.x,
                    //     ACCESS_ONCE(rx_queue_sync_list[rx_queue_index].flow[threadIdx.x]),
                    //     gbuf_offset[threadIdx.x],
                    //     oran_umsg_get_flowid((uint8_t*)msg_addr[threadIdx.x]),
                    //     oran_umsg_get_symbol_id((uint8_t*)msg_addr[threadIdx.x]),
                    //     oran_umsg_get_start_prb((uint8_t*)msg_addr[threadIdx.x]),
                    //     oran_umsg_get_num_prb((uint8_t*)msg_addr[threadIdx.x])
                    // );
                }
            }
            else
                msg_addr[threadIdx.x] = 0;
        }
        __syncthreads();

        /*
         * Each warp gets a different pkt
         */

        for(index_mbuf = warpId; index_mbuf < CK_ORDER_PKTS_BUFFERING; index_mbuf += nwarps)
        {
            if(msg_addr[index_mbuf] == 0)
                continue;
    #if 0
            if(laneId == 0)
                printf("Lane ID = %d Warp ID = %d, Packet index = %d Ordered PRBs = %d prb_size=%d prb_pkt=%d comp_meth=%d bit_width=%d\n",
                                laneId, warpId, index_mbuf, atomicAdd(ordered_prbs, 0), prb_size, msg_prb[index_mbuf], comp_meth, bit_width);
    #endif
            pkt_offset_ptr  = ((uint8_t*)msg_addr[index_mbuf]) + ORAN_IQ_HDR_SZ;
            gbuf_offset_ptr = buffer + gbuf_offset[index_mbuf];

            if(comp_meth == static_cast<uint8_t>(aerial_fh::UserDataCompressionMethod::BLOCK_FLOATING_POINT))
            {
                if(bit_width == BFP_NO_COMPRESSION) // BFP with 16 bits is a special case and uses FP16, so copy the values
                {
                    for(int index_copy = laneId; index_copy < (msg_prb[index_mbuf] * prb_size); index_copy += 32)
                        gbuf_offset_ptr[index_copy] = pkt_offset_ptr[index_copy];
                }
                else
                {
                    decompress_scale_blockFP<false>((unsigned char*)pkt_offset_ptr, (__half*)gbuf_offset_ptr, beta, msg_prb[index_mbuf], bit_width, (int)(threadIdx.x & 31), 32);
                }
            } else // aerial_fh::UserDataCompressionMethod::NO_COMPRESSION
            {
                decompress_scale_fixed<false>(pkt_offset_ptr, (__half*)gbuf_offset_ptr, beta, msg_prb[index_mbuf], bit_width, (int)(threadIdx.x & 31), 32);
            }

            //Each block increases the number of tot PRBs
            if(laneId == 0)
            {
                int tmp = atomicAdd(ordered_prbs, msg_prb[index_mbuf]);
                // printf("Lane ID = %d Warp ID = %d, Packet index = %d PRBs = %d\n", laneId, warpId, index_mbuf, atomicAdd(ordered_prbs, 0));
                if(tmp + msg_prb[index_mbuf] >= prb_x_slot)
                    ready_shared[0] = SYNC_PACKET_STATUS_EXIT;
            }
        }

        __syncthreads();
        if(threadIdx.x == 0)
        {
            if(done_shared[0] == 1)
            {
                ACCESS_ONCE(ready_list[rx_queue_index])                = SYNC_PACKET_STATUS_DONE; //Do not set it if there are packets for the next slot
                ACCESS_ONCE(rx_queue_sync_list[rx_queue_index].status) = SYNC_PACKET_STATUS_DONE;
                last_queue_index                                       = (last_queue_index + 1) % RX_QUEUE_SYNC_LIST_ITEMS;
            }
        }
        __syncthreads();

        rx_queue_index = (rx_queue_index + 1) % RX_QUEUE_SYNC_LIST_ITEMS;

        if(ready_shared[0] == SYNC_PACKET_STATUS_EXIT)
            goto exit;
    }

exit:
    if(threadIdx.x == 0)
    {
        order_kernel_end_cuphy_d[0]              = 1;
        order_completed_h[0]                     = 1;
        order_start_kernel_d[0]                  = 0xffffffff;
        ACCESS_ONCE(*last_ordered_item_h) = last_queue_index;
        // printf("Order kernel cell %d exit after %d/%d items, rx item: %d last rx item: %d frame %d subframe %d slot %d\n",
        //     cell_id, ordered_prbs[0], prb_x_slot, rx_queue_index, last_queue_index, frameId, subframeId, slotId);
    }

    return;
}

void launch_kernel_order(
    CUfunction            func,
    cudaStream_t          stream,
    int                   fake_run,
    int                   cell_id,
    uint8_t*              order_kernel_end_cuphy_d,
    uint8_t*              order_completed_h,
    uint32_t*             order_start_kernel_d,
    uint32_t*             ready_list,
    struct aerial_fh::rx_queue_sync* rx_queue_sync_list,
    int*                  last_ordered_item_h,
    uint8_t*              buffer,
    int                   prb_x_slot,
    int                   prb_x_symbol,
    int                   prb_x_symbol_x_antenna,
    uint8_t               frameId,
    uint8_t               subframeId,
    uint8_t               slotId,
    uint16_t*             eAxC_map,
    int                   eAxC_num,
    int                   comp_meth,
    int                   bit_width,
    int                   prb_size,
    float                 beta)
{
    void* args[] = {
        &fake_run, &cell_id, &order_kernel_end_cuphy_d, &order_completed_h,
        &order_start_kernel_d, &ready_list, &rx_queue_sync_list, &last_ordered_item_h,
        &buffer, &prb_x_slot, &prb_x_symbol, &prb_x_symbol_x_antenna,
        &frameId, &subframeId, &slotId, &eAxC_map, &eAxC_num,
        &comp_meth, &bit_width, &prb_size, &beta
    };

    CUresult res = cuLaunchKernel(func, 1, 1, 1, 512, 1, 1, 0, stream, args, nullptr);
    if(CUDA_SUCCESS != res)
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "[{}:{}] cuLaunchKernel failed for kernel_order with {} ", __FILE__, __LINE__, res);
}


__device__ int wait_packets(uint8_t& start_loop, unsigned long long& kernel_start, unsigned long long& first_packet_start,
                            uint32_t* ready_list,
                            int* ready_shared, int* rx_queue_index_s, int& rx_queue_index, int& last_queue_index, int* last_ordered_item_h,
                            uint32_t timeout_no_pkt_ns, uint32_t timeout_first_pkt_ns
                        )
{
    int ret = 0;
    unsigned long long current_time = 0;
    int ready_local;

    while(1)
    {
        //Should we add a global timer despite of the first packet? Assuming LSU always sends them
        current_time = __globaltimer();

        //Max timeout to wait for the very first slot packet
        if((start_loop == 1 && ((current_time - kernel_start) > timeout_no_pkt_ns)))
        {
            // printf("Cell %d Order kernel wait timeout after 8 ms F%dS%dS%d\n", cell_id, frameId, subframeId, slotId);
            ACCESS_ONCE(*ready_shared) = (int)SYNC_PACKET_STATUS_EXIT;
            __threadfence();
            ret = -1;
            break;
        }

        //Max timeout to wait to receive all expected packets for this slot
        if((start_loop == 2 && ((current_time - first_packet_start) > timeout_first_pkt_ns)))
        {
            // printf("Cell %d Order kernel recv timeout after 4 ms F%dS%dS%d\n", cell_id, frameId, subframeId, slotId);
            ACCESS_ONCE(*ready_shared) = (int)SYNC_PACKET_STATUS_EXIT;
            __threadfence();
            ret = -2;
            break;
        }

        if(start_loop == 0)
        {
            rx_queue_index   = *last_ordered_item_h;
            last_queue_index = rx_queue_index;
            start_loop       = 1;
        }
        else
        {
            // printf("Order kernel cell %d polling on item %d, ordered %d/%d addr %lx\n", cell_id, rx_queue_index, ordered_prbs, prb_x_slot, &ready_list[rx_queue_index]);
            ready_local = ACCESS_ONCE(ready_list[rx_queue_index]);
            if(ready_local == SYNC_PACKET_STATUS_READY)
            {
                if(start_loop == 1)
                {
                    start_loop = 2;
                    first_packet_start  = __globaltimer();
                }

                ACCESS_ONCE(*ready_shared)     = (int)SYNC_PACKET_STATUS_READY;
                ACCESS_ONCE(*rx_queue_index_s) = rx_queue_index;

                // printf("Order kernel block %d cell %d ready on item %d, ordered %d/%d\n", cell_id, blockIdx.x, rx_queue_index, ordered_prbs[0], prb_x_slot);

                __threadfence();
                ret = 0;
                break;
            }
        }
    }

    return ret;
}

__device__ void populate_addrs_ch1(
    uint16_t * msg_prb, uintptr_t * msg_addr, uint32_t * gbuf_offset,
    struct aerial_fh::rx_queue_sync* rx_queue_sync_list, int& rx_queue_index,
    uint8_t& frameId, uint8_t& subframeId, uint8_t& slotId,
    uint16_t* eAxC_map, int& eAxC_num,
    int& symbols_x_slot, int& prb_x_port_x_symbol, int& prb_size,
    uint8_t* done_shared,
    uint32_t* early_rx_packets, uint32_t* on_time_rx_packets, uint32_t* late_rx_packets,
    uint64_t& slot_start, uint64_t& ta4_min_ns, uint64_t& ta4_max_ns, uint64_t& slot_duration
)
{
    uint8_t frameId_pkt, subframeId_pkt, slotId_pkt;

    msg_prb[threadIdx.x] = 0;
    msg_addr[threadIdx.x] = 0;

    msg_addr[threadIdx.x] = (uintptr_t)ACCESS_ONCE(rx_queue_sync_list[rx_queue_index].addr[(ORDER_KERNEL_MAX_PKTS_BLOCK * blockIdx.x) + threadIdx.x]);
    if(msg_addr[threadIdx.x] != 0)
    {
        frameId_pkt    = oran_umsg_get_frame_id((uint8_t*)msg_addr[threadIdx.x]);
        subframeId_pkt = oran_umsg_get_subframe_id((uint8_t*)msg_addr[threadIdx.x]);
        slotId_pkt     = oran_umsg_get_slot_id((uint8_t*)msg_addr[threadIdx.x]);

        if(frameId_pkt == frameId)
        {
            if(
                (subframeId_pkt > subframeId) ||
                (subframeId_pkt == subframeId && slotId_pkt > slotId)
            )
            {
                done_shared[0]              = 0;
            }
            else if(subframeId_pkt == subframeId && slotId_pkt == slotId)
            {
                msg_prb[threadIdx.x] = oran_umsg_get_num_prb((uint8_t*)msg_addr[threadIdx.x]);
                if(msg_prb[threadIdx.x] == 0) msg_prb[threadIdx.x] = 273;

                uint16_t flow_index      = oran_umsg_get_flowid((uint8_t*)msg_addr[threadIdx.x]);
                gbuf_offset[threadIdx.x] = oran_get_offset_from_hdr(
                    (uint8_t*)msg_addr[threadIdx.x],
                    (uint16_t)get_eaxc_index(eAxC_map, eAxC_num, flow_index),
                    symbols_x_slot, prb_x_port_x_symbol, prb_size);
                    // printf("Order kernel item %d section id %d flow_index %d single CH\n", rx_queue_index, oran_umsg_get_section_id((uint8_t*)msg_addr[threadIdx.x]), flow_index);

                uint64_t rx_timestamp       = rx_queue_sync_list[rx_queue_index].rx_timestamp[(ORDER_KERNEL_MAX_PKTS_BLOCK * blockIdx.x) + threadIdx.x];
                uint8_t symbol_id_pkt       = oran_umsg_get_symbol_id((uint8_t*)msg_addr[threadIdx.x]);

                // Warning for invalid symbol_id (not used for array indexing here, only for timing calculations)
                if(__builtin_expect(symbol_id_pkt >= ORAN_ALL_SYMBOLS, 0))
                {
                    printf("WARNING invalid symbol_id %d (max %d) in single-packet kernel\n",
                           symbol_id_pkt,
                           ORAN_ALL_SYMBOLS - 1);
                }

                uint64_t packet_early_thres = slot_start + ta4_min_ns + (slot_duration * symbol_id_pkt / ORAN_MAX_SYMBOLS);
                uint64_t packet_late_thres  = slot_start + ta4_max_ns + (slot_duration * symbol_id_pkt / ORAN_MAX_SYMBOLS);

                if (rx_timestamp < packet_early_thres)
                {
                    atomicAdd(early_rx_packets, 1);
                }
                else if (rx_timestamp > packet_late_thres)
                {
                    // printf("late packet symbol %d flow index %d eaxc_index %d how late %llu ns, rx time %llu packet_early_thres %llu\n", symbol_id_pkt, flow_index, (uint16_t)get_eaxc_index(eAxC_map, eAxC_num, flow_index), rx_timestamp - packet_late_thres, rx_timestamp, packet_early_thres);
                    atomicAdd(late_rx_packets, 1);
                }
                else
                {
                    atomicAdd(on_time_rx_packets, 1);
                }

                /*
                printf("symbol = %d\tpacket_early_thres = %" PRIu64 "\tpacket_late_thres = %" PRIu64 "\trx_timestamp = %" PRIu64 "\n",
                    symbol_id_pkt, packet_early_thres, packet_late_thres, rx_timestamp
                    );
                */

                // CWE-787: discard packets whose untrusted symbolId/startPrb/numPrb would
                // index past the antenna buffer (consumer skips entries with msg_addr==0).
                // Done after the timing bookkeeping above, which still needs msg_addr.
                if(!ul_oran_offset_in_bounds((uint8_t*)msg_addr[threadIdx.x],
                                             (int)oran_umsg_get_start_prb((uint8_t*)msg_addr[threadIdx.x]),
                                             (int)msg_prb[threadIdx.x],
                                             symbols_x_slot, prb_x_port_x_symbol))
                {
                    msg_addr[threadIdx.x] = 0;
                    msg_prb[threadIdx.x] = 0;
                    gbuf_offset[threadIdx.x] = 0; // don't leave a stale OOB offset behind
                }
            }
        }

#if 0
        //Previous slot: it's ok to ignore and advance
        if(
            (frameId_pkt < frameId) ||
            (frameId_pkt == frameId && subframeId_pkt < subframeId) ||
            (frameId_pkt == frameId && subframeId_pkt == subframeId && slotId_pkt < slotId))
        {
            // printf("Order kernel cell %d Item %d pkt %d has previous frame %d/%d subframe %d/%d slot %d/%d\n",
            //     cell_id, rx_queue_index, threadIdx.x,
            //     oran_umsg_get_frame_id((uint8_t*)msg_addr[threadIdx.x]), frameId,
            //     oran_umsg_get_subframe_id((uint8_t*)msg_addr[threadIdx.x]), subframeId,
            //     oran_umsg_get_slot_id((uint8_t*)msg_addr[threadIdx.x]), slotId
            // );
            msg_addr[threadIdx.x] = 0;
        }
        //Next slot: don't advance
        else if(
            (frameId_pkt > frameId) ||
            (frameId_pkt == frameId && subframeId_pkt > subframeId) ||
            (frameId_pkt == frameId && subframeId_pkt == subframeId && slotId_pkt > slotId))
        {
            // printf("Order kernel cell %d Item %d pkt %d has next frame %d/%d subframe %d/%d slot %d/%d\n",
            //     cell_id, rx_queue_index, threadIdx.x,
            //     oran_umsg_get_frame_id((uint8_t*)msg_addr[threadIdx.x]), frameId,
            //     oran_umsg_get_subframe_id((uint8_t*)msg_addr[threadIdx.x]), subframeId,
            //     oran_umsg_get_slot_id((uint8_t*)msg_addr[threadIdx.x]), slotId
            // );
            msg_addr[threadIdx.x] = 0;
            done_shared[0]        = 0;
        }
        else
        {
            msg_prb[threadIdx.x] = oran_umsg_get_num_prb((uint8_t*)msg_addr[threadIdx.x]);
            if(msg_prb[threadIdx.x] == 0) msg_prb[threadIdx.x] = 273;

            uint16_t flow_index      = oran_umsg_get_flowid((uint8_t*)msg_addr[threadIdx.x]);
            gbuf_offset[threadIdx.x] = oran_get_offset_from_hdr(
                (uint8_t*)msg_addr[threadIdx.x],
                (uint16_t)get_eaxc_index(eAxC_map, eAxC_num, flow_index),
                symbols_x_slot, prb_x_port_x_symbol, prb_size);

            // if(gbuf_offset[threadIdx.x] == 0)
            //     printf("Offset item %d pkt %d (addr %lx) cell %d pkt %d, flow index %d offset %d flow %d symbol %d startPrb %d numPrb %d\n",
            //     rx_queue_index, threadIdx.x, msg_addr[threadIdx.x],
            //     cell_id, threadIdx.x,
            //     ACCESS_ONCE(rx_queue_sync_list[rx_queue_index].flow[threadIdx.x]),
            //     gbuf_offset[threadIdx.x],
            //     oran_umsg_get_flowid((uint8_t*)msg_addr[threadIdx.x]),
            //     oran_umsg_get_symbol_id((uint8_t*)msg_addr[threadIdx.x]),
            //     oran_umsg_get_start_prb((uint8_t*)msg_addr[threadIdx.x]),
            //     oran_umsg_get_num_prb((uint8_t*)msg_addr[threadIdx.x])
            // );
        }
#endif
    }
}

__device__ void populate_addrs_ch2(
    uint16_t * msg_prb_ch1, uintptr_t * msg_addr_ch1, uint32_t * gbuf_offset_ch1, uint16_t& sectionId_prach,
    uint16_t * msg_prb_ch2, uintptr_t * msg_addr_ch2, uint32_t * gbuf_offset_ch2,
    struct aerial_fh::rx_queue_sync* rx_queue_sync_list, int& rx_queue_index,
    uint8_t& frameId, uint8_t& subframeId, uint8_t& slotId,
    uint16_t* ch1_eAxC_map, int& ch1_eAxC_num,
    int& symbols_x_slot_ch1, int& prb_x_port_x_symbol_ch1, int& prb_size_ch1,
    uint16_t* ch2_eAxC_map, int& ch2_eAxC_num,
    int& symbols_x_slot_ch2, int& prb_x_port_x_symbol_ch2, int& prb_size_ch2,
    uint8_t* done_shared,
    uint32_t* early_rx_packets, uint32_t* on_time_rx_packets, uint32_t* late_rx_packets,
    uint64_t& slot_start, uint64_t& ta4_min_ns, uint64_t& ta4_max_ns, uint64_t& slot_duration
)
{
    uint8_t frameId_pkt, subframeId_pkt, slotId_pkt;
    int tmp_section_id;
    uint16_t flow_index;

    msg_addr_ch1[threadIdx.x] = 0;
    msg_prb_ch1[threadIdx.x] = 0;
    msg_addr_ch2[threadIdx.x] = 0;
    msg_prb_ch2[threadIdx.x] = 0;

    uintptr_t tmp_addr = (uintptr_t)ACCESS_ONCE(rx_queue_sync_list[rx_queue_index].addr[(ORDER_KERNEL_MAX_PKTS_BLOCK * blockIdx.x) + threadIdx.x]);
    if(tmp_addr != 0)
    {
        tmp_section_id  = oran_umsg_get_section_id((uint8_t*)tmp_addr);
        flow_index      = oran_umsg_get_flowid((uint8_t*)tmp_addr);
        frameId_pkt     = oran_umsg_get_frame_id((uint8_t*)tmp_addr);
        subframeId_pkt  = oran_umsg_get_subframe_id((uint8_t*)tmp_addr);
        slotId_pkt      = oran_umsg_get_slot_id((uint8_t*)tmp_addr);

        if(frameId_pkt == frameId)
        {
            if(
                (subframeId_pkt > subframeId) ||
                (subframeId_pkt == subframeId && slotId_pkt > slotId)
            )
            {
                done_shared[0] = 0;
            }
            //Here we assume that PUSCH has only one sectionId
            else if(subframeId_pkt == subframeId && slotId_pkt == slotId)
            {
                if(tmp_section_id < sectionId_prach)
                {
                    msg_addr_ch1[threadIdx.x]       = tmp_addr;
                    msg_prb_ch1[threadIdx.x]        = oran_umsg_get_num_prb((uint8_t*)tmp_addr);

                    if(msg_prb_ch1[threadIdx.x] == 0)
                        msg_prb_ch1[threadIdx.x]    = 273;

                    gbuf_offset_ch1[threadIdx.x]    = oran_get_offset_from_hdr(
                        (uint8_t*)tmp_addr,
                        (uint16_t)get_eaxc_index(ch1_eAxC_map, ch1_eAxC_num, flow_index),
                        symbols_x_slot_ch1, prb_x_port_x_symbol_ch1, prb_size_ch1);

                    // CWE-787: discard packets whose untrusted symbolId/startPrb/numPrb would
                    // index past the antenna buffer (consumer skips entries with msg_addr==0).
                    if(!ul_oran_offset_in_bounds((uint8_t*)tmp_addr,
                                                 (int)oran_umsg_get_start_prb((uint8_t*)tmp_addr),
                                                 (int)msg_prb_ch1[threadIdx.x],
                                                 symbols_x_slot_ch1, prb_x_port_x_symbol_ch1))
                    {
                        msg_addr_ch1[threadIdx.x] = 0;
                        msg_prb_ch1[threadIdx.x] = 0;
                        gbuf_offset_ch1[threadIdx.x] = 0; // don't leave a stale OOB offset behind
                    }

                }
                //PRACH may have various section Id in case of multiple occasions
                else //if(tmp_section_id == sectionId_ch2)
                {
                    msg_addr_ch2[threadIdx.x]       = tmp_addr;
                    msg_prb_ch2[threadIdx.x]        = oran_umsg_get_num_prb((uint8_t*)tmp_addr);

                    if(msg_prb_ch2[threadIdx.x] == 0)
                        msg_prb_ch2[threadIdx.x]    = 273;

                    gbuf_offset_ch2[threadIdx.x]    = oran_get_offset_from_hdr(
                        (uint8_t*)tmp_addr,
                        (uint16_t)get_eaxc_index(ch2_eAxC_map, ch2_eAxC_num, flow_index),
                        symbols_x_slot_ch2, prb_x_port_x_symbol_ch2, prb_size_ch2);

                    // CWE-787: discard packets whose untrusted symbolId/startPrb/numPrb would
                    // index past the antenna buffer (consumer skips entries with msg_addr==0).
                    if(!ul_oran_offset_in_bounds((uint8_t*)tmp_addr,
                                                 (int)oran_umsg_get_start_prb((uint8_t*)tmp_addr),
                                                 (int)msg_prb_ch2[threadIdx.x],
                                                 symbols_x_slot_ch2, prb_x_port_x_symbol_ch2))
                    {
                        msg_addr_ch2[threadIdx.x] = 0;
                        msg_prb_ch2[threadIdx.x] = 0;
                        gbuf_offset_ch2[threadIdx.x] = 0; // don't leave a stale OOB offset behind
                    }
                }

                uint64_t rx_timestamp       = rx_queue_sync_list[rx_queue_index].rx_timestamp[(ORDER_KERNEL_MAX_PKTS_BLOCK * blockIdx.x) + threadIdx.x];
                uint8_t symbol_id_pkt       = oran_umsg_get_symbol_id((uint8_t*)tmp_addr);

                // Warning for invalid symbol_id (not used for array indexing here, only for timing calculations)
                if(__builtin_expect(symbol_id_pkt >= ORAN_ALL_SYMBOLS, 0))
                {
                    printf("WARNING invalid symbol_id %d (max %d) in dual-channel kernel\n",
                           symbol_id_pkt,
                           ORAN_ALL_SYMBOLS - 1);
                }

                uint64_t packet_early_thres = slot_start + ta4_min_ns + (slot_duration * symbol_id_pkt / ORAN_MAX_SYMBOLS);
                uint64_t packet_late_thres  = slot_start + ta4_max_ns + (slot_duration * symbol_id_pkt / ORAN_MAX_SYMBOLS);

                if (rx_timestamp < packet_early_thres)
                {
                    atomicAdd(early_rx_packets, 1);
                }
                else if (rx_timestamp > packet_late_thres)
                {
                    // printf("late packet symbol %d flow index %d eaxc_index %d how late %llu ns, rx time %llu packet_early_thres %llu\n", symbol_id_pkt, flow_index, (uint16_t)get_eaxc_index(eAxC_map, eAxC_num, flow_index), rx_timestamp - packet_late_thres, rx_timestamp, packet_early_thres);
                    atomicAdd(late_rx_packets, 1);
                }
                else
                {
                    atomicAdd(on_time_rx_packets, 1);
                }

                /*
                printf("symbol = %d\tpacket_early_thres = %" PRIu64 "\tpacket_late_thres = %" PRIu64 "\trx_timestamp = %" PRIu64 "\n",
                    symbol_id_pkt, packet_early_thres, packet_late_thres, rx_timestamp
                    );
                */
            }
        }
    }
    else
    {
        msg_addr_ch1[threadIdx.x]   = 0;
        msg_addr_ch2[threadIdx.x]   = 0;
    }
}

__device__ void copy_decompress_packets(
        int& warpId, int& nwarps, int& laneId, int& comp_meth, int& bit_width, float& beta, int prb_x_slot, int& prb_size,
        uint8_t* buffer, uint16_t * msg_prb, uintptr_t * msg_addr, uint32_t * gbuf_offset,
        uint32_t* ordered_prbs, uint32_t* ordered_prbs_other, int* ready_shared, uint16_t start_prach_section_id
)
{
    uint8_t *pkt_offset_ptr, *gbuf_offset_ptr;

    for(int index_mbuf = warpId; index_mbuf < ORDER_KERNEL_MAX_PKTS_BLOCK; index_mbuf += nwarps)
    {
        if(msg_addr[index_mbuf] == 0)
            continue;
        #if 0
            if(laneId == 0)
                printf("Lane ID = %d Warp ID = %d, Packet index = %d Ordered PRBs = %d prb_size=%d prb_pkt=%d comp_meth=%d, bit_width=%d\n",
                                laneId, warpId, index_mbuf, atomicAdd(ordered_prbs, 0), prb_size, msg_prb[index_mbuf], comp_meth, bit_width);
        #endif

        uint16_t section_id = oran_umsg_get_section_id((uint8_t*)msg_addr[index_mbuf]);
        if (section_id >= start_prach_section_id)
            continue;

        pkt_offset_ptr  = ((uint8_t*)msg_addr[index_mbuf]) + ORAN_IQ_HDR_SZ;
        gbuf_offset_ptr = buffer + gbuf_offset[index_mbuf];

        /*
        * If no compression, just copy the data. Can the decompress function take care of it?
        */
        if(comp_meth == static_cast<uint8_t>(aerial_fh::UserDataCompressionMethod::BLOCK_FLOATING_POINT))
        {
            if(bit_width == BFP_NO_COMPRESSION) // BFP with 16 bits is a special case and uses FP16, so copy the values
            {
                for(int index_copy = laneId; index_copy < (msg_prb[index_mbuf] * prb_size); index_copy += 32)
                    gbuf_offset_ptr[index_copy] = pkt_offset_ptr[index_copy];
            }
            else
            {
                decompress_scale_blockFP<false>((unsigned char*)pkt_offset_ptr, (__half*)gbuf_offset_ptr, beta, msg_prb[index_mbuf], bit_width, (int)(threadIdx.x & 31), 32);
            }
        } else // aerial_fh::UserDataCompressionMethod::NO_COMPRESSION
        {
            decompress_scale_fixed<false>(pkt_offset_ptr, (__half*)gbuf_offset_ptr, beta, msg_prb[index_mbuf], bit_width, (int)(threadIdx.x & 31), 32);
        }

        //Each block increases the number of tot PRBs
        if(laneId == 0)
        {
            int oprb_ch1 = atomicAdd(ordered_prbs, msg_prb[index_mbuf]);
            int oprb_ch2 = 0;
            // IB barrier after this function guarantees no other thread is working on channel 2
            if(ordered_prbs_other != NULL)
                oprb_ch2 = *ordered_prbs_other;
            // printf("Lane ID = %d Warp ID = %d, Packet index = %d PRBs = %d\n", laneId, warpId, index_mbuf, atomicAdd(ordered_prbs, 0));
            if(oprb_ch1 + oprb_ch2 + msg_prb[index_mbuf] >= prb_x_slot)
                ready_shared[0] = SYNC_PACKET_STATUS_EXIT;
        }
    }
}

__device__ void copy_decompress_packets_prach(
    int& warpId, int& nwarps, int& laneId, int& comp_meth, int& bit_width, float& beta, int prb_x_slot, int& prb_size,
    uint8_t * buffer_0, uint8_t * buffer_1, uint8_t * buffer_2, uint8_t * buffer_3,
    uint16_t section_id_0, uint16_t section_id_1, uint16_t section_id_2, uint16_t section_id_3,
    uint16_t * msg_prb, uintptr_t * msg_addr, uint32_t * gbuf_offset,
    uint32_t* ordered_prbs, uint32_t* ordered_prbs_other, int* ready_shared
)
{
    uint8_t *pkt_offset_ptr, *gbuf_offset_ptr, *buffer;
    uint16_t section_id;

    for(int index_mbuf = warpId; index_mbuf < ORDER_KERNEL_MAX_PKTS_BLOCK; index_mbuf += nwarps)
    {
        if(msg_addr[index_mbuf] == 0)
            continue;
        #if 0
            if(laneId == 0)
                printf("Lane ID = %d Warp ID = %d, Packet index = %d Ordered PRBs = %d prb_size=%d prb_pkt=%d comp_meth=%d bit_width=%d\n",
                                laneId, warpId, index_mbuf, atomicAdd(ordered_prbs, 0), prb_size, msg_prb[index_mbuf], comp_meth, bit_width);
        #endif

        pkt_offset_ptr  = ((uint8_t*)msg_addr[index_mbuf]) + ORAN_IQ_HDR_SZ;
        section_id  = oran_umsg_get_section_id((uint8_t*)msg_addr[index_mbuf]);
        if(section_id == section_id_0) buffer = buffer_0;
        else if(section_id == section_id_1) buffer = buffer_1;
        else if(section_id == section_id_2) buffer = buffer_2;
        else if(section_id == section_id_3) buffer = buffer_3;
        else continue;

        gbuf_offset_ptr = buffer + gbuf_offset[index_mbuf];

        /*
        * If no compression, just copy the data. Can the decompress function take care of it?
        */
        if(comp_meth == static_cast<uint8_t>(aerial_fh::UserDataCompressionMethod::BLOCK_FLOATING_POINT))
        {
            if(bit_width == BFP_NO_COMPRESSION) // BFP with 16 bits is a special case and uses FP16, so copy the values
            {
                for(int index_copy = laneId; index_copy < (msg_prb[index_mbuf] * prb_size); index_copy += 32)
                    gbuf_offset_ptr[index_copy] = pkt_offset_ptr[index_copy];
            }
            else
            {
                decompress_scale_blockFP<false>((unsigned char*)pkt_offset_ptr, (__half*)gbuf_offset_ptr, beta, msg_prb[index_mbuf], bit_width, (int)(threadIdx.x & 31), 32);
            }
        } else // aerial_fh::UserDataCompressionMethod::NO_COMPRESSION
        {
            decompress_scale_fixed<false>(pkt_offset_ptr, (__half*)gbuf_offset_ptr, beta, msg_prb[index_mbuf], bit_width, (int)(threadIdx.x & 31), 32);
        }

        //Each block increases the number of tot PRBs
        if(laneId == 0)
        {
            int oprb_ch1 = atomicAdd(ordered_prbs, msg_prb[index_mbuf]);
            int oprb_ch2 = 0;
            // IB barrier after this function guarantees no other thread is working on channel 2
            if(ordered_prbs_other != NULL)
                oprb_ch2 = *ordered_prbs_other;
            // printf("Lane ID = %d Warp ID = %d, Packet index = %d PRBs = %d\n", laneId, warpId, index_mbuf, atomicAdd(ordered_prbs, 0));
            if(oprb_ch1 + oprb_ch2 + msg_prb[index_mbuf] >= prb_x_slot)
                ready_shared[0] = SYNC_PACKET_STATUS_EXIT;
        }
    }
}

__global__ void kernel_order_mb_one_ch(
    /* Cell specific */
    int                   cell_id,
    uint32_t*             order_kernel_end_cuphy_d,
    uint32_t*             order_start_kernel_d,
    uint32_t*             ready_list,
    struct aerial_fh::rx_queue_sync* rx_queue_sync_list,
    int*                  last_ordered_item_h,
    int                   comp_meth,
    int                   bit_width,
    float                 beta,
    int                   prb_size,

    /* Timeout */
    uint32_t              timeout_no_pkt_ns,
    uint32_t              timeout_first_pkt_ns,

    /* Time specific */
    uint8_t               frameId,
    uint8_t               subframeId,
    uint8_t               slotId,

    /* Order kernel specific */
    int*                  barrier_flag,
    uint8_t*              done_shared,
    int*                  ready_shared,
    int*                  rx_queue_index_s,

    uint32_t*             early_rx_packets,
    uint32_t*             on_time_rx_packets,
    uint32_t*             late_rx_packets,
    uint64_t              slot_start,
    uint64_t              ta4_min_ns,
    uint64_t              ta4_max_ns,
    uint64_t              slot_duration,

    uint32_t*             ordered_prbs,

    /* Output buffer specific */
    uint16_t*             eAxC_map,
    int                   eAxC_num,

    //PUSCH uses only 0, PRACH may use all of the 4 buffers
    uint8_t * buffer_0, uint8_t * buffer_1, uint8_t * buffer_2, uint8_t * buffer_3,
    uint16_t section_id_0, uint16_t section_id_1, uint16_t section_id_2, uint16_t section_id_3,
    int                   prb_x_slot,
    int                   prb_x_symbol,
    int                   prb_x_symbol_x_antenna,
    int                   symbols_x_slot,
    int                   prb_x_port_x_symbol,
    uint8_t channel_prach
)
{
    int                rx_queue_index = -1, last_queue_index = 0;
    int                laneId       = threadIdx.x % 32;
    int                warpId       = threadIdx.x / 32;
    int                nwarps       = blockDim.x / 32;
    unsigned long long first_packet_start        = 0;
    unsigned long long kernel_start = __globaltimer();
    uint8_t            start_loop = 0;
    int                barrier_idx = 1, barrier_signal = gridDim.x;
    int __attribute__((unused)) ret = 0;

    __shared__ uint16_t msg_prb[ORDER_KERNEL_MAX_PKTS_BLOCK];
    __shared__ uintptr_t msg_addr[ORDER_KERNEL_MAX_PKTS_BLOCK];
    __shared__ uint32_t gbuf_offset[ORDER_KERNEL_MAX_PKTS_BLOCK];

    if(ACCESS_ONCE(*order_start_kernel_d) == ORDER_KERNEL_ABORT)
    {
        if(blockIdx.x == 0 && threadIdx.x == 0)
        {
            ACCESS_ONCE(*order_kernel_end_cuphy_d)   = 1;
            __threadfence_system();
        }

        return;
    }

    if(blockIdx.x == 0 && threadIdx.x == 0)
    {
        ACCESS_ONCE(done_shared[0])  = 1;
        ACCESS_ONCE(ordered_prbs[0]) = 0;
        __threadfence();
        // printf("ORDER KERNEL buf %lx, sectionId %d, prach %d\n", buffer_0, section_id_0, channel_prach);
    }
    __syncthreads();

    while(1)
    {
        ///////////////////////////////////////////////////////////
        // Wait to receive new packets
        ///////////////////////////////////////////////////////////
        if(blockIdx.x == 0 && threadIdx.x == 0)
        {
            ret = wait_packets(start_loop, kernel_start, first_packet_start,
                &(ready_list[0]), &(ready_shared[0]), &(rx_queue_index_s[0]),
                rx_queue_index, last_queue_index, &(last_ordered_item_h[0]),
                timeout_no_pkt_ns, timeout_first_pkt_ns
            );

#if 0  //Enable for debug
            if(ret == -1)
                printf("Cell %d Order kernel %d/%d items wait timeout after %d ns F%dS%dS%d\n",
                    cell_id, ordered_prbs[0], prb_x_slot, timeout_no_pkt_ns, frameId, subframeId, slotId);
            if(ret == -2)
                printf("Cell %d Order kernel %d/%d items recv timeout after %d ns F%dS%dS%d\n",
                    cell_id, ordered_prbs[0], prb_x_slot, timeout_first_pkt_ns, frameId, subframeId, slotId);
#endif
        }

        ///////////////////////////////////////////////////////////
        // Inter-block barrier
        ///////////////////////////////////////////////////////////
        __threadfence();
        __syncthreads();
        if(threadIdx.x == 0)
            ib_barrier(&(barrier_flag[0]), barrier_signal, barrier_idx);
        __syncthreads();
        ///////////////////////////////////////////////////////////

        //Exit condition from host
        if(ready_shared[0] != SYNC_PACKET_STATUS_READY)
            goto exit;

        rx_queue_index = ACCESS_ONCE(rx_queue_index_s[0]);

        if(threadIdx.x < ORDER_KERNEL_MAX_PKTS_BLOCK)
        {
            populate_addrs_ch1(
                &(msg_prb[0]), &(msg_addr[0]), &(gbuf_offset[0]),
                rx_queue_sync_list, rx_queue_index,
                frameId, subframeId, slotId,
                eAxC_map, eAxC_num,
                symbols_x_slot, prb_x_port_x_symbol, prb_size,
                &(done_shared[0]),
                early_rx_packets, on_time_rx_packets, late_rx_packets,
                slot_start, ta4_min_ns, ta4_max_ns, slot_duration
            );
        }

        ///////////////////////////////////////////////////////////
        // Inter-block barrier
        ///////////////////////////////////////////////////////////
        __threadfence();
        __syncthreads();
        if(threadIdx.x == 0)
            ib_barrier(&(barrier_flag[0]), barrier_signal, barrier_idx);
        __syncthreads();
        ///////////////////////////////////////////////////////////

        ///////////////////////////////////////////////////////////
        // Copy or Decompress PRBs (1 pkt per warp)
        ///////////////////////////////////////////////////////////
        if(channel_prach == 1)
        {
            copy_decompress_packets_prach(
                warpId, nwarps, laneId, comp_meth, bit_width, beta, prb_x_slot, prb_size,
                &(buffer_0[0]), &(buffer_1[0]), &(buffer_2[0]), &(buffer_3[0]),
                section_id_0, section_id_1, section_id_2, section_id_3,
                &(msg_prb[0]), &(msg_addr[0]), &(gbuf_offset[0]),
                &(ordered_prbs[0]), NULL, &(ready_shared[0])
            );
        }
        else
        {
            copy_decompress_packets(
                warpId, nwarps, laneId, comp_meth, bit_width, beta, prb_x_slot, prb_size,
                &(buffer_0[0]), &(msg_prb[0]), &(msg_addr[0]), &(gbuf_offset[0]),
                &(ordered_prbs[0]), NULL, &(ready_shared[0]), section_id_0
            );
        }
        ///////////////////////////////////////////////////////////
        // Inter-block barrier
        ///////////////////////////////////////////////////////////
        __threadfence();
        __syncthreads();
        if(threadIdx.x == 0)
        {
            if(blockIdx.x == 0 && done_shared[0] == 1)
            {
                ACCESS_ONCE(ready_list[rx_queue_index])                = SYNC_PACKET_STATUS_DONE; //Do not set it if there are packets for the next slot
                ACCESS_ONCE(rx_queue_sync_list[rx_queue_index].status) = SYNC_PACKET_STATUS_DONE;
                last_queue_index                                       = (last_queue_index + 1) % RX_QUEUE_SYNC_LIST_ITEMS;
                __threadfence();
            }

            ib_barrier(&(barrier_flag[0]), barrier_signal, barrier_idx);
        }
        __syncthreads();
        ///////////////////////////////////////////////////////////

        rx_queue_index = (rx_queue_index + 1) % RX_QUEUE_SYNC_LIST_ITEMS;

        if(ready_shared[0] == SYNC_PACKET_STATUS_EXIT)
            goto exit;
    }

exit:
    if(blockIdx.x == 0 && threadIdx.x == 0)
    {
        ACCESS_ONCE(*last_ordered_item_h) = last_queue_index;
        ACCESS_ONCE(*order_kernel_end_cuphy_d) = 1;
        __threadfence_system();
        /* printf("Order kernel cell %d exit after %d/%d items, rx item: %d last rx item: %d frame %d subframe %d slot %d buffer %x%x%x%x %x%x%x%x %x%x%x%x\n",
             cell_id, ordered_prbs[0], prb_x_slot, rx_queue_index, last_queue_index, frameId, subframeId, slotId,
             buffer_0[0], buffer_0[1], buffer_0[2], buffer_0[3],
             buffer_0[4], buffer_0[5], buffer_0[6], buffer_0[7],
             buffer_0[8], buffer_0[9], buffer_0[10], buffer_0[11]
         );*/
        // printf("Order kernel cell %d exit after %d/%d items, rx item: %d last rx item: %d frame %d subframe %d slot %d\n",
        //     cell_id, ordered_prbs[0], prb_x_slot, rx_queue_index, last_queue_index, frameId, subframeId, slotId);
    }

    return;
}

__global__ void kernel_order_mb_two_ch(
    /* Cell specific */
    int                   cell_id,
    uint32_t*             order_kernel_end_cuphy_d,
    uint32_t*             order_start_kernel_d,
    uint32_t*             ready_list,
    struct aerial_fh::rx_queue_sync* rx_queue_sync_list,
    int*                  last_ordered_item_h,
    int                   comp_meth,
    int                   bit_width,
    float                 beta,
    int                   prb_size,
    /* Timeout */
    uint32_t              timeout_no_pkt_ns,
    uint32_t              timeout_first_pkt_ns,

    /* Time specific */
    uint8_t               frameId,
    uint8_t               subframeId,
    uint8_t               slotId,

    /* Order kernel specific */
    int*                  barrier_flag,
    uint8_t*              done_shared,
    int*                  ready_shared,
    int*                  rx_queue_index_s,

    uint32_t*             early_rx_packets,
    uint32_t*             on_time_rx_packets,
    uint32_t*             late_rx_packets,
    uint64_t              slot_start,
    uint64_t              ta4_min_ns,
    uint64_t              ta4_max_ns,
    uint64_t              slot_duration,

    /* CH1, Output buffer specific */
    uint16_t*             ch1_eAxC_map,
    int                   ch1_eAxC_num,
    uint8_t*              ch1_buffer,
    int                   ch1_prb_x_slot,
    int                   ch1_prb_x_symbol,
    int                   ch1_prb_x_symbol_x_antenna,
    int                   ch1_symbols_x_slot,
    int                   ch1_prb_x_port_x_symbol,
    uint32_t*             ch1_ordered_prbs,

    /* CH2, Output buffer specific */
    uint16_t*             ch2_eAxC_map,
    int                   ch2_eAxC_num,
    uint8_t * ch2_buffer_0, uint8_t * ch2_buffer_1, uint8_t * ch2_buffer_2, uint8_t * ch2_buffer_3,
    uint16_t ch2_section_id_0, uint16_t ch2_section_id_1, uint16_t ch2_section_id_2, uint16_t ch2_section_id_3,
    int                   ch2_prb_x_slot,
    int                   ch2_prb_x_symbol,
    int                   ch2_prb_x_symbol_x_antenna,
    int                   ch2_symbols_x_slot,
    int                   ch2_prb_x_port_x_symbol,
    uint32_t*             ch2_ordered_prbs
)
{
    int                rx_queue_index = -1, last_queue_index = 0;
    int                laneId       = threadIdx.x % 32;
    int                warpId       = threadIdx.x / 32;
    int                nwarps       = blockDim.x / 32;
    unsigned long long first_packet_start        = 0;
    unsigned long long kernel_start = __globaltimer();
    uint8_t            start_loop = 0;
    int                barrier_idx = 1, barrier_signal = gridDim.x;
    int __attribute__((unused)) ret = 0;

    __shared__ uint16_t msg_prb_ch1[ORDER_KERNEL_MAX_PKTS_BLOCK];
    __shared__ uint16_t msg_prb_ch2[ORDER_KERNEL_MAX_PKTS_BLOCK];
    __shared__ uintptr_t msg_addr_ch1[ORDER_KERNEL_MAX_PKTS_BLOCK];
    __shared__ uintptr_t msg_addr_ch2[ORDER_KERNEL_MAX_PKTS_BLOCK];
    __shared__ uint32_t gbuf_offset_ch1[ORDER_KERNEL_MAX_PKTS_BLOCK];
    __shared__ uint32_t gbuf_offset_ch2[ORDER_KERNEL_MAX_PKTS_BLOCK];

    if(ACCESS_ONCE(*order_start_kernel_d) == ORDER_KERNEL_ABORT)
    {
        if(blockIdx.x == 0 && threadIdx.x == 0)
        {
            ACCESS_ONCE(*order_kernel_end_cuphy_d) = 1;
            __threadfence_system();
        }

        return;
    }

    if(blockIdx.x == 0 && threadIdx.x == 0)
    {
        ACCESS_ONCE(done_shared[0])  = 1;
        ACCESS_ONCE(ch1_ordered_prbs[0]) = 0;
        ACCESS_ONCE(ch2_ordered_prbs[0]) = 0;
        __threadfence();
    }
    __syncthreads();

    while(1)
    {
        ///////////////////////////////////////////////////////////
        // Wait to receive new packets
        ///////////////////////////////////////////////////////////
        if(blockIdx.x == 0 && threadIdx.x == 0)
        {
            ret = wait_packets(start_loop, kernel_start, first_packet_start,
                &(ready_list[0]), &(ready_shared[0]), &(rx_queue_index_s[0]),
                rx_queue_index, last_queue_index, &(last_ordered_item_h[0]),
                timeout_no_pkt_ns, timeout_first_pkt_ns
            );

#if 0  //Enable for debug
            if(ret == -1)
                printf("Cell %d Order kernel %d/%d items wait timeout after %d ns F%dS%dS%d\n",
                        cell_id, ch1_ordered_prbs[0] + ch2_ordered_prbs[0], ch1_prb_x_slot+ch2_prb_x_slot, timeout_no_pkt_ns, frameId, subframeId, slotId);
            if(ret == -2)
                printf("Cell %d Order kernel %d/%d items recv timeout after %d ns F%dS%dS%d\n",
                        cell_id, ch1_ordered_prbs[0] + ch2_ordered_prbs[0], ch1_prb_x_slot+ch2_prb_x_slot, timeout_first_pkt_ns, frameId, subframeId, slotId);
#endif
        }

        ///////////////////////////////////////////////////////////
        // Inter-block barrier
        ///////////////////////////////////////////////////////////
        __threadfence();
        __syncthreads();
        if(threadIdx.x == 0)
            ib_barrier(&(barrier_flag[0]), barrier_signal, barrier_idx);
        __syncthreads();
        ///////////////////////////////////////////////////////////

        //Exit condition from host
        if(ready_shared[0] != SYNC_PACKET_STATUS_READY)
            goto exit;

        rx_queue_index = ACCESS_ONCE(rx_queue_index_s[0]);

        if(threadIdx.x < ORDER_KERNEL_MAX_PKTS_BLOCK)
        {
            populate_addrs_ch2(
                &(msg_prb_ch1[0]), &(msg_addr_ch1[0]), &(gbuf_offset_ch1[0]), ch2_section_id_0,
                &(msg_prb_ch2[0]), &(msg_addr_ch2[0]), &(gbuf_offset_ch2[0]),
                rx_queue_sync_list, rx_queue_index,
                frameId, subframeId, slotId,
                ch1_eAxC_map, ch1_eAxC_num,
                ch1_symbols_x_slot, ch1_prb_x_port_x_symbol, prb_size,
                ch2_eAxC_map, ch2_eAxC_num,
                ch2_symbols_x_slot, ch2_prb_x_port_x_symbol, prb_size,
                &(done_shared[0]),
                early_rx_packets, on_time_rx_packets, late_rx_packets,
                slot_start, ta4_min_ns, ta4_max_ns, slot_duration
            );
        }

        ///////////////////////////////////////////////////////////
        // Inter-block barrier
        ///////////////////////////////////////////////////////////
        __threadfence();
        __syncthreads();
        if(threadIdx.x == 0)
            ib_barrier(&(barrier_flag[0]), barrier_signal, barrier_idx);
        __syncthreads();
        ///////////////////////////////////////////////////////////

        ///////////////////////////////////////////////////////////
        // Copy or Decompress PRBs CH1 (1 pkt per warp)
        ///////////////////////////////////////////////////////////
        copy_decompress_packets(
            warpId, nwarps, laneId, comp_meth, bit_width, beta, ch1_prb_x_slot+ch2_prb_x_slot, prb_size,
            &(ch1_buffer[0]), &(msg_prb_ch1[0]), &(msg_addr_ch1[0]), &(gbuf_offset_ch1[0]),
            &(ch1_ordered_prbs[0]), &(ch2_ordered_prbs[0]), &(ready_shared[0]), ch2_section_id_0
        );

        ///////////////////////////////////////////////////////////
        // Inter-block barrier
        ///////////////////////////////////////////////////////////
        __threadfence();
        __syncthreads();
        if(threadIdx.x == 0)
            ib_barrier(&(barrier_flag[0]), barrier_signal, barrier_idx);
        __syncthreads();
        ///////////////////////////////////////////////////////////

        ///////////////////////////////////////////////////////////
        // Copy or Decompress PRBs CH2 (1 pkt per warp)
        ///////////////////////////////////////////////////////////
        if(ready_shared[0] != SYNC_PACKET_STATUS_EXIT)
        {
            copy_decompress_packets_prach(
                warpId, nwarps, laneId, comp_meth, bit_width, beta, ch1_prb_x_slot+ch2_prb_x_slot, //Avoid to cleanup ordered_prbs
                prb_size,
                &(ch2_buffer_0[0]), &(ch2_buffer_1[0]), &(ch2_buffer_2[0]), &(ch2_buffer_3[0]),
                ch2_section_id_0, ch2_section_id_1, ch2_section_id_2, ch2_section_id_3,
                &(msg_prb_ch2[0]), &(msg_addr_ch2[0]), &(gbuf_offset_ch2[0]),
                &(ch2_ordered_prbs[0]), &(ch1_ordered_prbs[0]), &(ready_shared[0])
            );
        }


        ///////////////////////////////////////////////////////////
        // Inter-block barrier
        ///////////////////////////////////////////////////////////
        __threadfence();
        __syncthreads();
        if(threadIdx.x == 0)
        {
            if(blockIdx.x == 0 && done_shared[0] == 1)
            {
                ACCESS_ONCE(ready_list[rx_queue_index])                = SYNC_PACKET_STATUS_DONE; //Do not set it if there are packets for the next slot
                ACCESS_ONCE(rx_queue_sync_list[rx_queue_index].status) = SYNC_PACKET_STATUS_DONE;
                last_queue_index                                       = (last_queue_index + 1) % RX_QUEUE_SYNC_LIST_ITEMS;
                __threadfence();
            }

            ib_barrier(&(barrier_flag[0]), barrier_signal, barrier_idx);
        }
        __syncthreads();
        ///////////////////////////////////////////////////////////

        rx_queue_index = (rx_queue_index + 1) % RX_QUEUE_SYNC_LIST_ITEMS;

        if(ready_shared[0] == SYNC_PACKET_STATUS_EXIT)
            goto exit;
    }

exit:
    if(blockIdx.x == 0 && threadIdx.x == 0)
    {
        ACCESS_ONCE(*last_ordered_item_h)       = last_queue_index;
        ACCESS_ONCE(*order_kernel_end_cuphy_d)  = 1;
        __threadfence_system();
        // printf("Order kernel cell %d exit after %d/%d items, rx item: %d last rx item: %d frame %d subframe %d slot %d\n",
        //     cell_id, ch1_ordered_prbs[0] + ch2_ordered_prbs[0], prb_x_slot, rx_queue_index, last_queue_index, frameId, subframeId, slotId);
    }

    return;
}

int launch_kernel_order_mb(
    CUfunction            func_one_ch,
    CUfunction            func_two_ch,
    cudaStream_t          stream,
    int                   fake_run,
    int                   cell_id,
    uint32_t*             order_kernel_end_cuphy_d,
    uint32_t*             order_start_kernel_d,
    uint32_t*             ready_list,
    struct aerial_fh::rx_queue_sync* rx_queue_sync_list,
    int*                  last_ordered_item_h,
    uint32_t              timeout_no_pkt_ns,
    uint32_t              timeout_first_pkt_ns,

    uint8_t               frameId,
    uint8_t               subframeId,
    uint8_t               slotId,
    int                   comp_meth,
    int                   bit_width,
    int                   prb_size,
    float                 beta,
    int*                  barrier_flag,
    uint8_t*              done_shared,
    int*                  ready_shared,
    int*                  rx_queue_index,

    uint32_t*             early_rx_packets,
    uint32_t*             on_time_rx_packets,
    uint32_t*             late_rx_packets,
    uint64_t              slot_start,
    uint64_t              ta4_min_ns,
    uint64_t              ta4_max_ns,
    uint64_t              slot_duration,

    uint16_t*             ST1_eAxC_map,
    int                   ST1_eAxC_num,
    uint8_t*              ST1_buffer,
    int                   ST1_prb_x_slot,
    int                   ST1_prb_x_symbol,
    int                   ST1_prb_x_symbol_x_antenna,
    uint32_t              ST1_prb_stride,
    uint32_t*             ST1_ordered_prbs,

    uint16_t*             prach_eAxC_map,
    int                   prach_eAxC_num,
    // uint8_t*              prach_buffer,
    // uint16_t              prach_section_id,
    uint8_t * prach_buffer_o0, uint8_t * prach_buffer_o1, uint8_t * prach_buffer_o2, uint8_t * prach_buffer_o3,
    uint16_t prach_section_id_o0, uint16_t prach_section_id_o1, uint16_t prach_section_id_o2, uint16_t prach_section_id_o3,
    int                   prach_prb_x_slot,
    int                   prach_prb_x_symbol,
    int                   prach_prb_x_symbol_x_antenna,
    uint32_t              prach_prb_stride,
    uint32_t*             prach_ordered_prbs
)
{
    if(
        (ST1_buffer == nullptr || ST1_prb_x_slot == 0) &&
        ((prach_buffer_o0 == nullptr && prach_buffer_o1 == nullptr && prach_buffer_o2 == nullptr && prach_buffer_o3 == nullptr) || prach_prb_x_slot == 0)
    )
        return EINVAL;

    if(
        (ST1_buffer != nullptr && ST1_prb_x_slot == 0) ||
        ( (prach_buffer_o0 != nullptr || prach_buffer_o1 != nullptr || prach_buffer_o2 != nullptr || prach_buffer_o3 != nullptr) && prach_prb_x_slot == 0)
    )
        return EINVAL;

    int pusch_symbols_x_slot = ORAN_PUSCH_SYMBOLS_X_SLOT;
    int prach_b4_symbols_x_slot = ORAN_PRACH_B4_SYMBOLS_X_SLOT;

    // PUSCH only
    if(prach_buffer_o0 == nullptr && prach_buffer_o1 == nullptr && prach_buffer_o2 == nullptr && prach_buffer_o3 == nullptr)
    {
        uint8_t* null_buf = nullptr;
        uint16_t zero_sec_id = 0;
        int is_prach = 0;
        void* args[] = {
            /* Cell specific */
            &cell_id, &order_kernel_end_cuphy_d, &order_start_kernel_d, &ready_list, &rx_queue_sync_list, &last_ordered_item_h, &comp_meth, &bit_width, &beta, &prb_size,
            /* Timeout */
            &timeout_no_pkt_ns, &timeout_first_pkt_ns,
            /* Time specific */
            &frameId, &subframeId, &slotId,
            /* Order kernel specific */
            &barrier_flag, &done_shared, &ready_shared, &rx_queue_index,
            &early_rx_packets, &on_time_rx_packets, &late_rx_packets,
            &slot_start, &ta4_min_ns, &ta4_max_ns, &slot_duration,
            &ST1_ordered_prbs,
            /* Output buffer specific */
            &ST1_eAxC_map, &ST1_eAxC_num,
            &ST1_buffer, &null_buf, &null_buf, &null_buf,
            &prach_section_id_o0, &zero_sec_id, &zero_sec_id, &zero_sec_id,
            &ST1_prb_x_slot, &ST1_prb_x_symbol, &ST1_prb_x_symbol_x_antenna,
            &pusch_symbols_x_slot, &ST1_prb_stride, &is_prach
        };
        CUresult cuResult = cuLaunchKernel(func_one_ch,
            ORDER_KERNEL_MB, 1, 1,
            512, 1, 1,
            0, stream,
            args, nullptr);
        CUDA_DRIVER_CHECK_NON_FATAL(cuResult);
        if(cuResult != CUDA_SUCCESS)
            return -1;
    }
    // PRACH only
    else if(ST1_buffer == nullptr)
    {
        int is_prach = 1;
        void* args[] = {
            /* Cell specific */
            &cell_id, &order_kernel_end_cuphy_d, &order_start_kernel_d, &ready_list, &rx_queue_sync_list, &last_ordered_item_h, &comp_meth, &bit_width, &beta, &prb_size,
            /* Timeout */
            &timeout_no_pkt_ns, &timeout_first_pkt_ns,
            /* Time specific */
            &frameId, &subframeId, &slotId,
            /* Order kernel specific */
            &barrier_flag, &done_shared, &ready_shared, &rx_queue_index,
            &early_rx_packets, &on_time_rx_packets, &late_rx_packets,
            &slot_start, &ta4_min_ns, &ta4_max_ns, &slot_duration,
            &prach_ordered_prbs,
            /* Output buffer specific */
            &prach_eAxC_map, &prach_eAxC_num,
            &prach_buffer_o0, &prach_buffer_o1, &prach_buffer_o2, &prach_buffer_o3,
            &prach_section_id_o0, &prach_section_id_o1, &prach_section_id_o2, &prach_section_id_o3,
            &prach_prb_x_slot, &prach_prb_x_symbol, &prach_prb_x_symbol_x_antenna,
            &prach_b4_symbols_x_slot, &prach_prb_stride, &is_prach
        };
        CUresult cuResult = cuLaunchKernel(func_one_ch,
            ORDER_KERNEL_MB, 1, 1,
            512, 1, 1,
            0, stream,
            args, nullptr);
        CUDA_DRIVER_CHECK_NON_FATAL(cuResult);
        if(cuResult != CUDA_SUCCESS)
            return -1;
    }
    // PUSCH + PRACH
    else
    {
        void* args[] = {
            /* Cell specific */
            &cell_id, &order_kernel_end_cuphy_d, &order_start_kernel_d, &ready_list, &rx_queue_sync_list, &last_ordered_item_h, &comp_meth, &bit_width, &beta, &prb_size,
            /* Timeout */
            &timeout_no_pkt_ns, &timeout_first_pkt_ns,
            /* Time specific */
            &frameId, &subframeId, &slotId,
            /* Order kernel specific */
            &barrier_flag, &done_shared, &ready_shared, &rx_queue_index,
            &early_rx_packets, &on_time_rx_packets, &late_rx_packets,
            &slot_start, &ta4_min_ns, &ta4_max_ns, &slot_duration,
            /* PUSCH Output buffer specific */
            &ST1_eAxC_map, &ST1_eAxC_num,
            &ST1_buffer, &ST1_prb_x_slot, &ST1_prb_x_symbol, &ST1_prb_x_symbol_x_antenna,
            &pusch_symbols_x_slot, &ST1_prb_stride, &ST1_ordered_prbs,
            /* PRACH Output buffer specific */
            &prach_eAxC_map, &prach_eAxC_num,
            &prach_buffer_o0, &prach_buffer_o1, &prach_buffer_o2, &prach_buffer_o3,
            &prach_section_id_o0, &prach_section_id_o1, &prach_section_id_o2, &prach_section_id_o3,
            &prach_prb_x_slot, &prach_prb_x_symbol, &prach_prb_x_symbol_x_antenna,
            &prach_b4_symbols_x_slot, &prach_prb_stride, &prach_ordered_prbs
        };
        CUresult cuResult = cuLaunchKernel(func_two_ch,
            ORDER_KERNEL_MB, 1, 1,
            512, 1, 1,
            0, stream,
            args, nullptr);
        CUDA_DRIVER_CHECK_NON_FATAL(cuResult);
        if(cuResult != CUDA_SUCCESS)
            return -1;
    }

    return 0;
}

void launch_receive_process_kernel_for_test_bench(
    CUfunction tb_pingpong_srs_func,
    CUfunction tb_pingpong_no_srs_func,
    CUfunction recv_process_func,
    cudaStream_t stream,
	/* Cell */
	int*		cell_id,
	uint32_t		**exit_cond_d,
    uint16_t* sem_order_num,
    int*		ru_type,

	/* ORAN */
	uint8_t		frameId,
	uint8_t		subframeId,
	uint8_t		slotId,

    int		prb_size,
    int*		comp_meth,
    int*		bit_width,
    float*		beta,
    uint32_t		**last_sem_idx_order_h,
    
    uint32_t*       rx_pkt_num_slot,
    uint8_t**       tb_fh_buf,
    uint32_t  max_pkt_size,

    /* Timer */
    uint32_t         **early_rx_packets,
    uint32_t         **on_time_rx_packets,
    uint32_t         **late_rx_packets,
    uint32_t         **next_slot_early_rx_packets,
    uint32_t         **next_slot_on_time_rx_packets,
    uint32_t         **next_slot_late_rx_packets,
    uint32_t         **rx_packets_dropped_count,
    bool*             cell_health,
    uint32_t         **start_cuphy_d,

    /* Sub-slot processing*/
    uint32_t* 			   sym_ord_done_sig_arr,
    uint32_t*              sym_ord_done_mask_arr,
    uint32_t*              pusch_prb_symbol_map,
	uint32_t* 			   num_order_cells_sym_mask_arr,    
    
    /*PUSCH*/
    uint8_t**       pusch_buffer,
	uint16_t		**pusch_eAxC_map,
	int*			pusch_eAxC_num,    
    int			pusch_symbols_x_slot,
    uint32_t*			pusch_prb_x_port_x_symbol,
    uint32_t		**pusch_ordered_prbs,
    int*			pusch_prb_x_slot,

    /*PRACH*/
	uint16_t 	**prach_eAxC_map,
	int*		prach_eAxC_num,
	uint8_t		**prach_buffer_0,
	uint8_t		**prach_buffer_1,
	uint8_t 	**prach_buffer_2,
	uint8_t 	**prach_buffer_3,
    int*	    prach_prb_x_slot,
    int			prach_symbols_x_slot,
    uint32_t*   prach_prb_x_port_x_symbol,
    uint32_t	**prach_ordered_prbs,
	uint16_t	prach_section_id_0,
	uint16_t	prach_section_id_1,
	uint16_t	prach_section_id_2,
	uint16_t	prach_section_id_3,
    uint8_t num_order_cells,    

    /*SRS*/
	uint16_t		**srs_eAxC_map,
	int*			srs_eAxC_num,
	uint8_t		**srs_buffer,
	int*			srs_prb_x_slot,
    uint32_t*              srs_prb_stride,	
    uint32_t		**srs_ordered_prbs,
    uint8_t*          srs_start_sym,

    /*Receive CTA params*/
    uint32_t	timeout_no_pkt_ns,
    uint32_t	timeout_first_pkt_ns,
	uint32_t  timeout_log_interval_ns,
	uint8_t   timeout_log_enable,
    uint64_t      **order_kernel_last_timeout_error_time,
    uint32_t		**last_sem_idx_rx_h,
    bool            commViaCpu,
    struct doca_gpu_eth_rxq **doca_rxq,
    uint32_t  max_rx_pkts,
    uint32_t  rx_pkts_timeout_ns,
    struct doca_gpu_semaphore_gpu **sem_gpu,
    struct aerial_fh_gpu_semaphore_gpu **sem_gpu_aerial_fh,
	uint64_t*		slot_start,
	uint64_t*		ta4_min_ns,
	uint64_t*		ta4_max_ns,
	uint64_t*		slot_duration,
    uint8_t                ul_rx_pkt_tracing_level,
    uint8_t                ul_order_kernel_mode,
    uint8_t                enable_srs
)
{
	int cudaBlocks = (num_order_cells);

    if(ul_order_kernel_mode == 0) {
        // The pingpong kernel is currently disabled in the test bench due to several missing buffers (e.g., packet
        // stat buffers).
        order_kernel_pkt_tracing_info pkt_tracing_info = {
            .rx_packets_count = nullptr,
            .rx_bytes_count = nullptr,
            .next_slot_rx_packets_count = nullptr,
            .next_slot_rx_bytes_count = nullptr,
            .rx_packets_ts_earliest = nullptr,
            .rx_packets_ts_latest = nullptr,
            .rx_packets_ts = nullptr,
            .next_slot_rx_packets_ts = nullptr,
        };

        CUfunction pp_func;
        int pp_threads;
        uint8_t pusch_prb_non_zero;

        if(enable_srs)
        {
            pp_func = tb_pingpong_srs_func;
            pp_threads = ORDER_KERNEL_PINGPONG_SRS_NUM_THREADS;
            pusch_prb_non_zero = 0;
        }
        else
        {
            pp_func = tb_pingpong_no_srs_func;
            pp_threads = ORDER_KERNEL_PINGPONG_NUM_THREADS;
            pusch_prb_non_zero = 1;
        }

        int pusch_sym_x_slot = ORAN_PUSCH_SYMBOLS_X_SLOT;
        int prach_b4_symbols_x_slot = ORAN_PRACH_B4_SYMBOLS_X_SLOT;
        int srs_symbols_x_slot = ORAN_MAX_SRS_SYMBOLS;
        uint8_t* pcap_buffer_null = nullptr;
        uint8_t* pcap_buffer_ts_null = nullptr;
        uint32_t* pcap_buffer_index_null = nullptr;
        uint8_t pcap_capture_enable = 0;
        uint64_t pcap_capture_cell_bitmask = 0;

        void* pp_args[] = {
            /* DOCA objects */
            &doca_rxq, &sem_gpu, &sem_gpu_aerial_fh, &sem_order_num,
            /* Cell specific */
            &cell_id, &ru_type, &cell_health, &start_cuphy_d, &exit_cond_d, &last_sem_idx_rx_h, &last_sem_idx_order_h, &comp_meth, &bit_width, &beta, &prb_size,
            /* Timeout */
            &timeout_no_pkt_ns, &timeout_first_pkt_ns, &timeout_log_interval_ns, &timeout_log_enable, &max_rx_pkts, &rx_pkts_timeout_ns,
            /* Time specific */
            &frameId, &subframeId, &slotId,
            /* Order kernel specific */
            &early_rx_packets, &on_time_rx_packets, &late_rx_packets, &next_slot_early_rx_packets, &next_slot_on_time_rx_packets, &next_slot_late_rx_packets,
            &slot_start, &ta4_min_ns, &ta4_max_ns, &slot_duration, &order_kernel_last_timeout_error_time, &pkt_tracing_info, &rx_packets_dropped_count,
            /* Sub-slot processing specific */
            &sym_ord_done_sig_arr, &sym_ord_done_mask_arr, &pusch_prb_symbol_map, &num_order_cells_sym_mask_arr, &pusch_prb_non_zero,
            /* PUSCH Output buffer specific */
            &pusch_eAxC_map, &pusch_eAxC_num, &pusch_buffer, &pusch_prb_x_slot,
            &pusch_sym_x_slot, &pusch_prb_x_port_x_symbol, &pusch_ordered_prbs,
            /* PRACH Output buffer specific */
            &prach_eAxC_map, &prach_eAxC_num,
            &prach_buffer_0, &prach_buffer_1, &prach_buffer_2, &prach_buffer_3,
            &prach_section_id_0, &prach_section_id_1, &prach_section_id_2, &prach_section_id_3,
            &prach_prb_x_slot,
            &prach_b4_symbols_x_slot, &prach_prb_x_port_x_symbol, &prach_ordered_prbs,
            /* SRS Output buffer specific */
            &srs_eAxC_map, &srs_eAxC_num, &srs_buffer, &srs_prb_x_slot, &srs_symbols_x_slot, &srs_prb_stride, &srs_ordered_prbs, &srs_start_sym,
            &num_order_cells,
            /* PCAP Capture specific */
            &pcap_buffer_null, &pcap_buffer_ts_null, &pcap_buffer_index_null, &pcap_capture_enable, &pcap_capture_cell_bitmask,
            /* Test bench values */
            &tb_fh_buf, &max_pkt_size, &rx_pkt_num_slot
        };

        MemtraceDisableScope md;
        CUresult cuResult = cuLaunchKernel(pp_func,
            cudaBlocks, 1, 1,
            pp_threads, 1, 1,
            0, stream,
            pp_args, nullptr);
        CUDA_DRIVER_CHECK(cuResult);
    }
    else {
        // Dual CTA mode
        // block 0 to receive, block 1 to process
        int numThreads = 128;

        void* args[] = {
            /* Cell */
            &cell_id, &exit_cond_d, &sem_order_num, &ru_type,
            /* ORAN */
            &frameId, &subframeId, &slotId,
            &prb_size, &comp_meth, &bit_width, &beta, &last_sem_idx_order_h,
            &rx_pkt_num_slot, &tb_fh_buf, &max_pkt_size,
            /* Sub-slot processing */
            &sym_ord_done_sig_arr, &sym_ord_done_mask_arr, &pusch_prb_symbol_map, &num_order_cells_sym_mask_arr,
            /* PUSCH */
            &pusch_buffer, &pusch_eAxC_map, &pusch_eAxC_num, &pusch_symbols_x_slot,
            &pusch_prb_x_port_x_symbol, &pusch_ordered_prbs, &pusch_prb_x_slot,
            /* PRACH */
            &prach_eAxC_map, &prach_eAxC_num,
            &prach_buffer_0, &prach_buffer_1, &prach_buffer_2, &prach_buffer_3,
            &prach_prb_x_slot, &prach_symbols_x_slot, &prach_prb_x_port_x_symbol, &prach_ordered_prbs,
            &prach_section_id_0, &prach_section_id_1, &prach_section_id_2, &prach_section_id_3,
            /* Receive CTA params */
            &timeout_no_pkt_ns, &timeout_first_pkt_ns, &timeout_log_interval_ns, &timeout_log_enable,
            &order_kernel_last_timeout_error_time, &last_sem_idx_rx_h, &commViaCpu,
            &doca_rxq, &max_rx_pkts, &rx_pkts_timeout_ns, &sem_gpu,
            &slot_start, &ta4_min_ns, &ta4_max_ns, &slot_duration,
            &ul_rx_pkt_tracing_level
        };

        CUresult cuResult = cuLaunchKernel(recv_process_func,
            cudaBlocks * 2, 1, 1,
            numThreads, 1, 1,
            0, stream,
            args, nullptr);
        CUDA_DRIVER_CHECK(cuResult);
    }
}

#ifdef __cplusplus
}
#endif

#include "cuda_driver_utils/cuda_kernel_utils.cuh"

bool init_order_kernel_functions(OrderKernelFunctions& funcs)
{
    bool ok = true;
    ok &= resolve_kernel_func<TAG>(&funcs.order_kernel_doca,                        reinterpret_cast<void*>(order_kernel_doca),                        "order_kernel_doca");
    ok &= resolve_kernel_func<TAG>(&funcs.order_kernel_doca_single,                 reinterpret_cast<void*>(order_kernel_doca_single),                 "order_kernel_doca_single");
    ok &= resolve_kernel_func<TAG>(&funcs.order_kernel_doca_single_srs,             reinterpret_cast<void*>(order_kernel_doca_single_srs),             "order_kernel_doca_single_srs");
    ok &= resolve_kernel_func<TAG>(&funcs.order_kernel_doca_single_subSlot,         reinterpret_cast<void*>(order_kernel_doca_single_subSlot),         "order_kernel_doca_single_subSlot");
    ok &= resolve_kernel_func<TAG>(&funcs.order_kernel_cpu_init_comms_single_subSlot, reinterpret_cast<void*>(order_kernel_cpu_init_comms_single_subSlot), "order_kernel_cpu_init_comms_single_subSlot");
    ok &= resolve_kernel_func<TAG>(&funcs.kernel_order_mb_one_ch,                   reinterpret_cast<void*>(kernel_order_mb_one_ch),                   "kernel_order_mb_one_ch");
    ok &= resolve_kernel_func<TAG>(&funcs.kernel_order_mb_two_ch,                   reinterpret_cast<void*>(kernel_order_mb_two_ch),                   "kernel_order_mb_two_ch");
    ok &= resolve_kernel_func<TAG>(&funcs.receive_kernel_for_test_bench,            reinterpret_cast<void*>(receive_kernel_for_test_bench),            "receive_kernel_for_test_bench");
    ok &= resolve_kernel_func<TAG>(&funcs.receive_process_kernel_for_test_bench,    reinterpret_cast<void*>(receive_process_kernel_for_test_bench),    "receive_process_kernel_for_test_bench");

    // Ping-pong kernel instantiations: <is_test_bench, pkt_trace, srs_enable, threads, ctas>
    ok &= resolve_kernel_func<TAG>(&funcs.pingpong_trace_srs,
        reinterpret_cast<void*>(&order_kernel_doca_single_subSlot_pingpong<false, 1, 1, ORDER_KERNEL_PINGPONG_SRS_NUM_THREADS, 1>),
        "pingpong<false,1,1,SRS_THREADS,1>");
    ok &= resolve_kernel_func<TAG>(&funcs.pingpong_trace_srs_pusch,
        reinterpret_cast<void*>(&order_kernel_doca_single_subSlot_pingpong<false, 1, 3, ORDER_KERNEL_PINGPONG_SRS_NUM_THREADS, 1>),
        "pingpong<false,1,3,SRS_THREADS,1>");
    ok &= resolve_kernel_func<TAG>(&funcs.pingpong_trace_no_srs,
        reinterpret_cast<void*>(&order_kernel_doca_single_subSlot_pingpong<false, 1, 0, ORDER_KERNEL_PINGPONG_NUM_THREADS, 2>),
        "pingpong<false,1,0,THREADS,2>");
    ok &= resolve_kernel_func<TAG>(&funcs.pingpong_no_trace_srs,
        reinterpret_cast<void*>(&order_kernel_doca_single_subSlot_pingpong<false, 0, 1, ORDER_KERNEL_PINGPONG_SRS_NUM_THREADS, 1>),
        "pingpong<false,0,1,SRS_THREADS,1>");
    ok &= resolve_kernel_func<TAG>(&funcs.pingpong_no_trace_srs_pusch,
        reinterpret_cast<void*>(&order_kernel_doca_single_subSlot_pingpong<false, 0, 3, ORDER_KERNEL_PINGPONG_SRS_NUM_THREADS, 1>),
        "pingpong<false,0,3,SRS_THREADS,1>");
    ok &= resolve_kernel_func<TAG>(&funcs.pingpong_no_trace_no_srs,
        reinterpret_cast<void*>(&order_kernel_doca_single_subSlot_pingpong<false, 0, 0, ORDER_KERNEL_PINGPONG_NUM_THREADS, 2>),
        "pingpong<false,0,0,THREADS,2>");

    ok &= resolve_kernel_func<TAG>(&funcs.order_kernel_printf_warmup,
        reinterpret_cast<void*>(order_kernel_printf_warmup), "order_kernel_printf_warmup");

    return ok;
}

extern "C" void launch_order_kernel_printf_warmup(CUfunction func, cudaStream_t stream)
{
    CUDA_DRIVER_CHECK(cuLaunchKernel(func, 1, 1, 1, 1, 1, 1, 0, stream, nullptr, nullptr));
}

bool resolve_kernel_order_handle(CUfunction* out)
{
    return resolve_kernel_func<TAG>(out, reinterpret_cast<void*>(kernel_order), "kernel_order");
}

bool init_order_kernel_tb_functions(OrderKernelTbFunctions& funcs)
{
    bool ok = true;

    cudaError_t e = cudaGetFuncBySymbol(&funcs.tb_pingpong_srs,
        reinterpret_cast<void*>(&order_kernel_doca_single_subSlot_pingpong<true, 0, 1, ORDER_KERNEL_PINGPONG_SRS_NUM_THREADS, 1>));
    if(cudaSuccess != e)
    {
        NVLOGE_FMT(TAG, AERIAL_CUDA_KERNEL_EVENT, "[{}:{}] cudaGetFuncBySymbol failed for tb_pingpong<true,0,1,1024,1>: {}", __FILE__, __LINE__, cudaGetErrorString(e));
        ok = false;
    }

    e = cudaGetFuncBySymbol(&funcs.tb_pingpong_no_srs,
        reinterpret_cast<void*>(&order_kernel_doca_single_subSlot_pingpong<true, 0, 0, ORDER_KERNEL_PINGPONG_NUM_THREADS, 2>));
    if(cudaSuccess != e)
    {
        NVLOGE_FMT(TAG, AERIAL_CUDA_KERNEL_EVENT, "[{}:{}] cudaGetFuncBySymbol failed for tb_pingpong<true,0,0,320,2>: {}", __FILE__, __LINE__, cudaGetErrorString(e));
        ok = false;
    }

    e = cudaGetFuncBySymbol(&funcs.recv_process, reinterpret_cast<void*>(receive_process_kernel_for_test_bench));
    if(cudaSuccess != e)
    {
        NVLOGE_FMT(TAG, AERIAL_CUDA_KERNEL_EVENT, "[{}:{}] cudaGetFuncBySymbol failed for receive_process_kernel_for_test_bench: {}", __FILE__, __LINE__, cudaGetErrorString(e));
        ok = false;
    }

    return ok;
}
