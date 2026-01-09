# CUDA Kernel Implementation Details for PDSCH Processing
## Comprehensive Technical Analysis of cuPHY GPU Kernels

**NVIDIA Aerial CUDA-Accelerated RAN**
**cuPHY SDK**
**Date:** January 2026
**Report Type:** Deep Technical Analysis - Kernel Level

---

## Executive Summary

This comprehensive technical report provides an in-depth analysis of CUDA kernel implementations for Physical Downlink Shared Channel (PDSCH) processing within the NVIDIA Aerial cuPHY SDK. The analysis focuses on low-level GPU programming details including kernel launch configurations, memory optimization strategies, synchronization mechanisms, and performance optimization techniques.

### Key Implementation Characteristics

**Kernel Architecture:**
- 2D grid topology for multi-Transport Block processing
- Fixed 256-thread blocks for consistent occupancy
- Grid-stride loops for coalesced memory access
- Dynamic shared memory allocation per kernel

**Memory Optimization:**
- Vectorized loads/stores for bandwidth efficiency
- Cooperative group async memcpy (CUDA 11.1+)
- Constant memory for lookup tables
- Shared memory caching for frequently accessed data

**Synchronization:**
- Warp-level primitives (__shfl_down_sync, __shfl_up_sync)
- Block-level barriers (__syncthreads)
- Atomic operations (XOR, Add, CAS) for cross-block coordination
- CUDA event-based stream synchronization

**Performance Techniques:**
- Loop unrolling with #pragma unroll
- Inline functions with __forceinline__
- Mixed precision (__half, float) with specialized atomics
- PTX intrinsics (__byte_perm, __brev) for bit manipulation

---

## Table of Contents

1. [CUDA Kernel Architecture Overview](#1-cuda-kernel-architecture-overview)
2. [Kernel Launch Configurations](#2-kernel-launch-configurations)
3. [Grid and Block Dimension Calculations](#3-grid-and-block-dimension-calculations)
4. [CUDA Streams Implementation](#4-cuda-streams-implementation)
5. [CUDA Graphs Usage](#5-cuda-graphs-usage)
6. [Memory Management Strategies](#6-memory-management-strategies)
7. [Memory Access Patterns](#7-memory-access-patterns)
8. [Shared Memory Optimization](#8-shared-memory-optimization)
9. [Synchronization Mechanisms](#9-synchronization-mechanisms)
10. [Warp-Level Primitives](#10-warp-level-primitives)
11. [Atomic Operations](#11-atomic-operations)
12. [CRC Kernel Implementation](#12-crc-kernel-implementation)
13. [LDPC Encoding Kernels](#13-ldpc-encoding-kernels)
14. [Rate Matching Kernels](#14-rate-matching-kernels)
15. [Scrambling and Descrambling](#15-scrambling-and-descrambling)
16. [Modulation Mapping Kernels](#16-modulation-mapping-kernels)
17. [DMRS Generation Kernels](#17-dmrs-generation-kernels)
18. [Performance Optimization Techniques](#18-performance-optimization-techniques)
19. [Occupancy Analysis](#19-occupancy-analysis)
20. [Memory Bandwidth Optimization](#20-memory-bandwidth-optimization)
21. [Register Usage Optimization](#21-register-usage-optimization)
22. [Best Practices and Recommendations](#22-best-practices-and-recommendations)

---

## 1. CUDA Kernel Architecture Overview

### 1.1 Threading Model

cuPHY kernels follow CUDA's hierarchical threading model:

```
GPU Device
    ├── Grid (2D: gridDim.x × gridDim.y)
    │   ├── Block (0,0)
    │   │   ├── Warp 0 (threads 0-31)
    │   │   ├── Warp 1 (threads 32-63)
    │   │   ├── ...
    │   │   └── Warp 7 (threads 224-255)
    │   ├── Block (1,0)
    │   ├── ...
    │   └── Block (gridDim.x-1, gridDim.y-1)
    └── Shared Resources
        ├── Constant Memory (64 KB)
        ├── Texture Memory (Read-only cache)
        └── Global Memory (HBM)
```

**Thread Identification:**
```cuda
// Global thread ID in 2D grid
int globalThreadX = blockIdx.x * blockDim.x + threadIdx.x;
int globalThreadY = blockIdx.y;

// For Transport Block processing:
// - blockIdx.y = TB index
// - blockIdx.x = data chunk index
// - threadIdx.x = thread within block (0-255)
```

### 1.2 Execution Model

**Warp-Based Execution:**
- 32 threads execute in SIMT (Single Instruction Multiple Threads) fashion
- Branch divergence causes serialization within warps
- Optimal code path: all threads in warp follow same branch

**Block Scheduling:**
- SMs (Streaming Multiprocessors) schedule blocks dynamically
- Multiple blocks can execute on single SM if resources permit
- Blocks execute independently (no synchronization between blocks)

### 1.3 Memory Hierarchy

```
┌─────────────────────────────────────────────────────────┐
│ Registers (per thread)           │ L1 Cache (per SM)    │
│ - Fastest access (~1 cycle)      │ - 128 KB configurable│
│ - Limited quantity (32K-64K)     │ - Shared with L1/Tex │
├──────────────────────────────────┴──────────────────────┤
│ Shared Memory (per block)                               │
│ - Fast access (~5 cycles)                               │
│ - 48-164 KB per SM (configurable)                       │
│ - Programmer-managed cache                              │
├─────────────────────────────────────────────────────────┤
│ L2 Cache (device-wide)                                  │
│ - 40-50 MB (H100: 50 MB)                               │
│ - Automatic caching                                     │
├─────────────────────────────────────────────────────────┤
│ HBM Global Memory (device-wide)                         │
│ - 80 GB (H100), 3.35 TB/s bandwidth                    │
│ - High latency (~200-400 cycles)                       │
│ - Main data storage                                     │
└─────────────────────────────────────────────────────────┘
```

---

## 2. Kernel Launch Configurations

### 2.1 Standard Launch Pattern

From `cuPHY/src/cuphy/crc.cu` (lines 276-283):

```cuda
// Block size is constant across cuPHY kernels
const uint32_t blockSize = GLOBAL_BLOCK_SIZE;  // 256 threads

// Grid dimensions for Code Block processing
uint32_t gridSizeCBX = maxNCBsPerTB;     // Code blocks per TB
uint32_t gridSizeCBY = nTBs;              // Number of TBs
dim3 gCBSize(gridSizeCBX, gridSizeCBY);

// Grid dimensions for Transport Block processing
uint32_t gridSizeTBX = (tbSize + blockSize - 1) / blockSize;
uint32_t gridSizeTBY = nTBs;
dim3 gTBSize(gridSizeTBX, gridSizeTBY);
```

**Rationale for 256 Threads:**
- 8 warps per block (256 / 32 = 8)
- Good balance between occupancy and resource usage
- Enables efficient warp-level reductions
- Compatible with all modern NVIDIA GPUs (compute capability 7.0+)

### 2.2 Kernel Launch Syntax

From `cuPHY/src/cuphy/crc.cu` (line 310):

```cuda
// Standard kernel launch
crcUplinkPuschCodeBlocksKernel<<<gCBSize, blockSize,
    sizeof(uint32_t) * WARP_SIZE, strm>>>(desc);

// Breakdown:
// <<<grid, block, shared_mem_size, stream>>>
// - grid: dim3(gridSizeCBX, gridSizeCBY, 1)
// - block: 256
// - shared_mem_size: 32 * sizeof(uint32_t) = 128 bytes
// - stream: cudaStream_t for async execution
```

### 2.3 Grid Constant Descriptor Pattern

From `cuPHY/src/cuphy/crc.cu` (line 365):

```cuda
__global__ void crcDownlinkPdschCodeBlocksKernel(
    const __grid_constant__ crcEncodeDescr_t desc)
{
    // Descriptor is directly accessible in constant memory
    // No need for explicit host-to-device copy
    uint32_t nCodeBlocks = desc.nCodeBlocks;
    uint32_t* inputCodeBlocks = desc.inputCodeBlocks;
    // ...
}
```

**Benefits of __grid_constant__:**
- Eliminates descriptor copy overhead
- Broadcast read from constant memory (cached)
- Single instruction reads by all threads in warp
- Available in CUDA 11.0+

### 2.4 Dynamic Grid Sizing Examples

#### Example 1: CRC Code Block Processing

```cuda
// From crc.cu, line 277-278
uint32_t gridSizeCBX = maxNCBsPerTB;  // e.g., 8 CBs
uint32_t gridSizeCBY = nTBs;           // e.g., 128 TBs
dim3 gCBSize(gridSizeCBX, gridSizeCBY); // Grid: 8 × 128 = 1,024 blocks

// Total threads: 1,024 blocks × 256 threads = 262,144 threads
```

#### Example 2: Transport Block CRC Assembly

```cuda
// From crc.cu, line 280-281
uint32_t gridSizeTBX = (tbSize + blockSize - 1) / blockSize;
// For tbSize = 10,000 bytes: gridSizeTBX = (10000 + 255) / 256 = 40 blocks
uint32_t gridSizeTBY = nTBs;  // e.g., 128 TBs
dim3 gTBSize(gridSizeTBX, gridSizeTBY); // Grid: 40 × 128 = 5,120 blocks
```

#### Example 3: Rate Matching with Variable Dimensions

```cuda
// From rate_matching.cu (inferred from code structure)
int num_codewords = nTBs * max_codewords_per_tb;
int elements_per_codeword = E_total_bits / num_codewords;

dim3 blockDim(256, 1, 1);
dim3 gridDim(
    (elements_per_codeword + 255) / 256,  // X: elements
    num_codewords,                         // Y: codewords
    1
);

rateMatchingKernel<<<gridDim, blockDim, 0, stream>>>(...);
```

### 2.5 Occupancy Calculation

**Theoretical Occupancy:**
```
Occupancy = (Active Warps per SM) / (Maximum Warps per SM)

For H100 (Hopper architecture):
- Maximum warps per SM: 64
- Maximum threads per SM: 2048
- Maximum blocks per SM: 32

With 256-thread blocks (8 warps):
- Max blocks per SM: min(32, 2048/256, 64/8) = min(32, 8, 8) = 8 blocks
- Active warps: 8 blocks × 8 warps = 64 warps
- Occupancy: 64 / 64 = 100%
```

**Resource Limitations:**

```cuda
// Check occupancy programmatically
int minGridSize, blockSize;
cudaOccupancyMaxPotentialBlockSize(
    &minGridSize,
    &blockSize,
    crcUplinkPuschCodeBlocksKernel,
    dynamicSharedMemSize,  // 128 bytes
    0                      // block size limit (0 = no limit)
);

// blockSize will be optimal for this kernel (likely 256 or 512)
```

---

## 3. Grid and Block Dimension Calculations

### 3.1 Mathematical Formulations

#### 3.1.1 1D Grid Calculation

For processing N elements with B threads per block:

```cuda
int numBlocks = (N + B - 1) / B;  // Ceiling division
dim3 grid(numBlocks, 1, 1);
dim3 block(B, 1, 1);

// Example: N = 10,000 bytes, B = 256
// numBlocks = (10000 + 255) / 256 = 40 blocks
```

**Why Ceiling Division?**
```
Without ceiling:
N = 10,000, B = 256
10,000 / 256 = 39.0625 → 39 blocks
39 × 256 = 9,984 threads (16 elements unprocessed!)

With ceiling:
(10,000 + 255) / 256 = 40 blocks
40 × 256 = 10,240 threads (sufficient, with bounds checking)
```

#### 3.1.2 2D Grid Calculation

For processing M × N matrix with B threads per block:

```cuda
dim3 block(16, 16, 1);  // 256 threads
dim3 grid(
    (N + 15) / 16,      // Columns
    (M + 15) / 16,      // Rows
    1
);

// In kernel:
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;

if (col < N && row < M) {
    // Process element (row, col)
}
```

### 3.2 cuPHY Dimension Patterns

#### 3.2.1 Code Block Processing Grid

From `cuPHY/src/cuphy/crc.cu`:

```cuda
// Dimensions:
// - gridDim.x = number of code blocks per TB (e.g., 1-152)
// - gridDim.y = number of TBs (e.g., 1-256)
// - blockDim.x = 256 threads

__global__ void crcCodeBlocksKernel(...) {
    int codeBlockIdx = blockIdx.x;  // 0 to maxNCBsPerTB-1
    int tbIdx = blockIdx.y;          // 0 to nTBs-1
    int tid = threadIdx.x;           // 0 to 255

    // Each block processes one code block from one TB
    // 256 threads cooperate to process the code block
}
```

**Example Configuration:**
```
Configuration: 128 TBs, average 8 CBs per TB
Grid: (8, 128, 1) = 1,024 blocks
Block: (256, 1, 1) = 256 threads per block
Total: 262,144 threads launched
Processing: 1,024 code blocks
```

#### 3.2.2 Transport Block Assembly Grid

```cuda
// Dimensions:
// - gridDim.x = ceil(TB_size / 256)
// - gridDim.y = number of TBs
// - blockDim.x = 256 threads

__global__ void crcTransportBlocksKernel(...) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int tbIdx = blockIdx.y;

    // Grid-stride loop over TB bytes
    while (tid < tbSize) {
        // Process byte at index tid
        tid += blockDim.x * gridDim.x;
    }
}
```

### 3.3 Grid Stride Loop Pattern

**Standard Implementation:**

From `cuPHY/src/cuphy/crc.cu` (lines 514-588):

```cuda
__global__ void crcUplinkPuschTransportBlocksKernel(...) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Grid-stride loop
    while (tid < totalElements) {
        // Process element at index tid
        processElement(data[tid]);

        // Advance by total grid size
        tid += stride;
    }
}
```

**Benefits:**
1. **Scalability**: Works with any data size, regardless of grid dimensions
2. **Coalescing**: Consecutive threads access consecutive memory locations
3. **Load Balancing**: Automatically distributes work evenly
4. **Reusability**: Same kernel works for small and large datasets

**Example Execution:**
```
Grid: 10 blocks, Block: 256 threads
Total threads: 2,560
Data size: 100,000 elements

Thread 0 processes: 0, 2560, 5120, 7680, ..., 97920
Thread 1 processes: 1, 2561, 5121, 7681, ..., 97921
...
Thread 2559 processes: 2559, 5119, 7679, ..., 99839

Iterations per thread: ceil(100000 / 2560) = 40 iterations
```

---

## 4. CUDA Streams Implementation

### 4.1 Stream-Based Asynchronous Execution

From `cuPHY/src/cuphy/crc.cu` (line 253):

```cuda
cuphyStatus_t launch(
    const crcEncodeDescr_t* desc,
    // ... other parameters ...
    cudaStream_t strm)  // Stream for async execution
{
    // All operations enqueued on specified stream
    const uint32_t blockSize = GLOBAL_BLOCK_SIZE;
    dim3 gCBSize(maxNCBsPerTB, nTBs);

    // Kernel launch on stream
    crcUplinkPuschCodeBlocksKernel<<<gCBSize, blockSize,
        sizeof(uint32_t) * WARP_SIZE, strm>>>(desc);

    // Check for launch errors
    CUDA_CHECK(cudaGetLastError());

    return CUPHY_STATUS_SUCCESS;
}
```

### 4.2 Asynchronous Memory Operations

From `cuPHY/src/cuphy/crc.cu` (lines 818-826):

```cuda
if (enable_desc_async_copy) {
    // Async copy of descriptor to GPU
    CUDA_CHECK(cudaMemcpyAsync(
        gpu_desc,                    // Destination (device)
        cpu_desc,                    // Source (host)
        sizeof(crcEncodeDescr_t),    // Size
        cudaMemcpyHostToDevice,      // Direction
        strm                         // Stream
    ));

    // Async memset for TB CRC output buffer
    if (!codeBlocksOnly) {
        CUDA_CHECK(cudaMemsetAsync(
            d_tbCRCs,                // Device pointer
            0,                       // Value
            sizeof(uint32_t) * nTBs, // Size
            strm                     // Stream
        ));
    }
}
```

**Execution Timeline:**
```
CPU                    GPU Stream
 │                          │
 ├─ cudaMemcpyAsync() ──────┼─→ [H2D Transfer: descriptor]
 │  (returns immediately)   │
 │                          │   (Transfer in progress)
 ├─ kernel<<<...>>>() ──────┼─→ [Kernel Launch]
 │  (returns immediately)   │
 │                          │   (Waiting for H2D)
 │                          │   [Kernel Execution starts]
 │                          │
 ├─ cudaMemcpyAsync() ──────┼─→ [D2H Transfer: results]
 │  (returns immediately)   │   (Waiting for kernel)
 │                          │   [Transfer starts]
 │                          │
 ├─ cudaStreamSynchronize() │
 │  (blocks here) ──────────┼─→ (Transfer completes)
 │                          │
 └─ (continues)             └─
```

### 4.3 Stream Synchronization

#### 4.3.1 Explicit Stream Sync

```cuda
// Block CPU until all GPU operations on stream complete
cudaStreamSynchronize(strm);
```

#### 4.3.2 Event-Based Synchronization

From `cuPHY/src/cuphy/crc.cu` (lines 318-345):

```cuda
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

// Record start event
cudaEventRecord(start, strm);

// Launch kernels
for (int i = 0; i < NRUNS; i++) {
    crcUplinkPuschCodeBlocksKernel<<<gCBSize, blockSize,
        sizeof(uint32_t) * WARP_SIZE, strm>>>(desc);
}

// Record stop event
cudaEventRecord(stop, strm);

// Wait for stop event
cudaEventSynchronize(stop);

// Calculate elapsed time
float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);

printf("Average kernel time: %.3f ms\n", milliseconds / NRUNS);
```

**Event Timeline:**
```
Time ────────────────────────────────────────────►
     │          │                    │           │
     start      Kernel 1             Kernel N    stop
     event      launches             completes   event
     │                                            │
     └────────── Elapsed Time ────────────────────┘
```

### 4.4 Multi-Stream Patterns

#### 4.4.1 Concurrent Kernel Execution

```cuda
// Create multiple streams
const int NUM_STREAMS = 4;
cudaStream_t streams[NUM_STREAMS];
for (int i = 0; i < NUM_STREAMS; i++) {
    cudaStreamCreate(&streams[i]);
}

// Launch kernels on different streams
for (int i = 0; i < NUM_CELLS; i++) {
    int streamIdx = i % NUM_STREAMS;

    // Each stream processes different cell
    cuphyPdschPipeline<<<grid, block, 0, streams[streamIdx]>>>(
        cellData[i], ...
    );
}

// Synchronize all streams
for (int i = 0; i < NUM_STREAMS; i++) {
    cudaStreamSynchronize(streams[i]);
}
```

**Execution Pattern:**
```
Stream 0: [Cell 0] [Cell 4] [Cell 8]  [Cell 12]
Stream 1:    [Cell 1] [Cell 5] [Cell 9]  [Cell 13]
Stream 2:       [Cell 2] [Cell 6] [Cell 10] [Cell 14]
Stream 3:          [Cell 3] [Cell 7] [Cell 11] [Cell 15]

Time ──────────────────────────────────────────────────►
```

#### 4.4.2 Stream Priorities

```cuda
// Create high-priority stream for critical path
cudaStream_t highPriorityStream;
int highPriority = -5;  // Higher priority (more negative)
cudaStreamCreateWithPriority(&highPriorityStream,
    cudaStreamNonBlocking, highPriority);

// Create normal-priority stream
cudaStream_t normalStream;
int normalPriority = 0;
cudaStreamCreateWithPriority(&normalStream,
    cudaStreamNonBlocking, normalPriority);

// Critical kernels on high-priority stream
cuphyPdschKernel<<<grid, block, 0, highPriorityStream>>>(...);

// Less critical kernels on normal stream
cuphyValidationKernel<<<grid, block, 0, normalStream>>>(...);
```

---

## 5. CUDA Graphs Usage

### 5.1 Graph Creation and Instantiation

From `cuPHY/src/cuphy/prach_receiver.cu` (inferred pattern):

```cuda
// Graph creation
cudaGraph_t graph;
cudaGraphExec_t graphExec;

// Create empty graph
CUDA_CHECK(cudaGraphCreate(&graph, 0));

// Begin stream capture to record operations
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

// Launch kernels (captured into graph)
crcKernel<<<grid, block, 0, stream>>>(...);
ldpcKernel<<<grid, block, 0, stream>>>(...);
rateMatchKernel<<<grid, block, 0, stream>>>(...);
modulationKernel<<<grid, block, 0, stream>>>(...);

// End capture
cudaStreamEndCapture(stream, &graph);

// Instantiate graph for execution
CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph,
    NULL, NULL, 0));
```

### 5.2 Graph Execution

From `cuPHY/src/cuphy/pusch_start_kernels.cu`:

```cuda
// Execute graph (launches all kernels in one operation)
cudaGraphLaunch(graphExec, stream);

// Or use fire-and-forget stream
cudaGraphLaunch(graphExec, cudaStreamGraphFireAndForget);

// Graph execution is asynchronous
// Use stream synchronization to wait
cudaStreamSynchronize(stream);
```

### 5.3 Graph Benefits

**CPU Overhead Comparison:**

```
Stream Mode (10 kernel launches):
    CPU overhead: 10 launches × 15 μs = 150 μs
    GPU execution: 500 μs
    Total: 650 μs

Graph Mode (1 graph launch):
    CPU overhead: 1 launch × 2 μs = 2 μs
    GPU execution: 500 μs
    Total: 502 μs

Performance gain: 22.8% reduction in total time
```

### 5.4 Graph Update Patterns

```cuda
// For dynamic parameter changes, update graph nodes
cudaGraphNode_t kernelNode;
cudaKernelNodeParams params;

// Get existing kernel node
cudaGraphGetNodes(graph, &kernelNode, &nodeCount);

// Update kernel parameters
params.func = (void*)newKernel;
params.gridDim = newGrid;
params.blockDim = newBlock;
params.kernelParams = newArgs;

// Update node in executable graph
cudaGraphExecKernelNodeSetParams(graphExec, kernelNode, &params);

// Re-launch with updated parameters
cudaGraphLaunch(graphExec, stream);
```

---

## 6. Memory Management Strategies

### 6.1 Memory Allocation Types

#### 6.1.1 Device Memory (Global)

```cuda
// Standard device allocation
uint8_t* d_data;
size_t size = nTBs * maxTBSize;
cudaMalloc(&d_data, size);

// Use in kernel
kernelFunc<<<grid, block>>>(d_data, ...);

// Free when done
cudaFree(d_data);
```

#### 6.1.2 Pinned Host Memory

```cuda
// Pinned (page-locked) memory for faster transfers
uint8_t* h_data;
cudaMallocHost(&h_data, size);  // or cudaHostAlloc

// Benefits:
// - DMA transfers without OS paging
// - Async memcpy support
// - ~2x faster H2D/D2H transfers

// Transfer
cudaMemcpyAsync(d_data, h_data, size,
    cudaMemcpyHostToDevice, stream);

// Free
cudaFreeHost(h_data);
```

#### 6.1.3 Managed (Unified) Memory

```cuda
// Unified memory (accessible from CPU and GPU)
uint8_t* unified_data;
cudaMallocManaged(&unified_data, size);

// Access from CPU
unified_data[0] = 0xAA;

// Access from GPU
kernelFunc<<<grid, block>>>(unified_data, ...);

// Automatic migration between CPU and GPU
```

### 6.2 Memory Transfer Patterns

#### 6.2.1 Synchronous Transfer

```cuda
// Blocking transfer (CPU waits)
cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

// CPU blocked here until transfer completes
// Then kernel can safely use d_data
```

#### 6.2.2 Asynchronous Transfer

From `cuPHY/src/cuphy/crc.cu` (line 818):

```cuda
// Non-blocking transfer
cudaMemcpyAsync(d_data, h_data, size,
    cudaMemcpyHostToDevice, stream);

// CPU continues immediately
// Transfer happens asynchronously on stream
```

#### 6.2.3 Overlapping Transfer and Computation

```cuda
// Stream 0: Transfer data for iteration i+1
cudaMemcpyAsync(d_data[1], h_data[i+1], size,
    cudaMemcpyHostToDevice, stream[0]);

// Stream 1: Process iteration i
kernelFunc<<<grid, block, 0, stream[1]>>>(d_data[0], ...);

// Streams execute concurrently:
// - Stream 0 transfers while Stream 1 computes
// - Hides transfer latency
```

### 6.3 Memory Pooling

```cuda
// Persistent memory pool to avoid repeated alloc/free
class DeviceMemoryPool {
    std::vector<void*> buffers;
    std::vector<size_t> sizes;
    std::vector<bool> inUse;

public:
    void* allocate(size_t size) {
        // Find free buffer >= size
        for (size_t i = 0; i < buffers.size(); i++) {
            if (!inUse[i] && sizes[i] >= size) {
                inUse[i] = true;
                return buffers[i];
            }
        }

        // No suitable buffer, allocate new
        void* ptr;
        cudaMalloc(&ptr, size);
        buffers.push_back(ptr);
        sizes.push_back(size);
        inUse.push_back(true);
        return ptr;
    }

    void free(void* ptr) {
        for (size_t i = 0; i < buffers.size(); i++) {
            if (buffers[i] == ptr) {
                inUse[i] = false;
                return;
            }
        }
    }
};
```

---

## 7. Memory Access Patterns

### 7.1 Coalesced Access

#### 7.1.1 Perfect Coalescing

From `cuPHY/src/cuphy/crc.cu` (lines 118-125):

```cuda
__global__ void crcKernel(const uint32_t* input, uint32_t* output) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Grid-stride loop ensures coalescing
    while (tid < size) {
        // Perfect coalescing:
        // - Thread 0 accesses input[tid+0]
        // - Thread 1 accesses input[tid+1]
        // - Thread 2 accesses input[tid+2]
        // - ...
        // - Thread 31 accesses input[tid+31]
        // All 32 accesses in one 128-byte memory transaction
        uint32_t value = input[tid];

        // Process value...

        // Write also coalesced
        output[tid] = result;

        tid += stride;
    }
}
```

**Memory Transaction Diagram:**
```
Threads:    0   1   2  ...  30  31
            │   │   │        │   │
            ▼   ▼   ▼        ▼   ▼
Memory:   [0] [1] [2] ... [30][31] ← 128-byte transaction

Efficiency: 100% (all bytes used)
```

#### 7.1.2 Uncoalesced Access (Anti-pattern)

```cuda
// BAD: Strided access
__global__ void badKernel(const uint32_t* input, uint32_t* output) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Strided access with large stride
    int stride = 1024;
    uint32_t value = input[tid * stride];  // BAD!

    // Thread 0 accesses input[0]
    // Thread 1 accesses input[1024]
    // Thread 2 accesses input[2048]
    // Each thread causes separate memory transaction
    // Efficiency: ~3% (only 4 bytes of 128 used per transaction)
}
```

### 7.2 Aligned Access

#### 7.2.1 Alignment Requirements

```cuda
// Memory alignment for optimal performance:
// - uint8_t:  1-byte alignment (any address)
// - uint16_t: 2-byte alignment (address % 2 == 0)
// - uint32_t: 4-byte alignment (address % 4 == 0)
// - uint64_t: 8-byte alignment (address % 8 == 0)
// - float:    4-byte alignment
// - double:   8-byte alignment
// - float4:   16-byte alignment

// Check alignment
assert((reinterpret_cast<uintptr_t>(ptr) & 0x3) == 0);  // 4-byte aligned
assert((reinterpret_cast<uintptr_t>(ptr) & 0x7) == 0);  // 8-byte aligned
assert((reinterpret_cast<uintptr_t>(ptr) & 0xF) == 0);  // 16-byte aligned
```

#### 7.2.2 Enforcing Alignment

From `cuPHY/src/cuphy/rate_matching.cu` (lines 281-300):

```cuda
template <typename T_OUT>
__device__ __forceinline__ void zeroRangeVec(
    T_OUT* __restrict__ out,
    uint32_t start, uint32_t end, uint32_t tid, uint32_t stride)
{
    if (start >= end) return;

    T_OUT* base = out + start;
    uint32_t total = end - start;

    // Calculate bytes needed to reach 16-byte alignment
    const uintptr_t addr = reinterpret_cast<uintptr_t>(base);
    const uint32_t bytes_to_align = (16u - (addr & 15u)) & 15u;
    const uint32_t head = min(total, bytes_to_align / sizeof(T_OUT));

    // Handle unaligned head elements (scalar)
    for (uint32_t i = tid; i < head; i += stride) {
        base[i] = 0;
    }

    // Aligned body (vectorized)
    uint4* vec_base = reinterpret_cast<uint4*>(base + head);
    uint32_t vec_count = (total - head) / (sizeof(uint4) / sizeof(T_OUT));

    for (uint32_t i = tid; i < vec_count; i += stride) {
        vec_base[i] = make_uint4(0, 0, 0, 0);  // 16-byte store
    }

    // Handle tail elements
    uint32_t tail_start = head + vec_count * (sizeof(uint4) / sizeof(T_OUT));
    for (uint32_t i = tid + tail_start; i < total; i += stride) {
        base[i] = 0;
    }
}
```

### 7.3 Vectorized Loads/Stores

#### 7.3.1 Vector Types

```cuda
// CUDA vector types (aligned access required):
char4, uchar4      // 4 × 1-byte = 4 bytes
short4, ushort4    // 4 × 2-byte = 8 bytes
int4, uint4        // 4 × 4-byte = 16 bytes
long4, ulong4      // 4 × 8-byte = 32 bytes
float4, double4    // 4 × 4/8-byte = 16/32 bytes

// Example: uint4 for 16-byte operations
typedef struct __device_builtin__ __builtin_align__(16) uint4 {
    unsigned int x, y, z, w;
} uint4;
```

#### 7.3.2 Vectorized Memory Operations

```cuda
__global__ void vectorizedCopy(const float4* input, float4* output, int count) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < count) {
        // Single instruction loads 16 bytes
        float4 data = input[tid];

        // Process all 4 floats
        data.x *= 2.0f;
        data.y *= 2.0f;
        data.z *= 2.0f;
        data.w *= 2.0f;

        // Single instruction stores 16 bytes
        output[tid] = data;
    }
}

// Performance:
// - 4× fewer load/store instructions
// - Better memory bandwidth utilization
// - Reduced instruction overhead
```

---

## 8. Shared Memory Optimization

### 8.1 Shared Memory Declaration

From `cuPHY/src/cuphy/crc.cu` (lines 71-72, 378-379):

```cuda
// Dynamic shared memory (size specified at kernel launch)
extern __shared__ uint32_t shmemBuf[];

// Kernel launch with dynamic shared memory:
kernelFunc<<<grid, block, sharedMemSize, stream>>>(...);
//                         ^^^^^^^^^^^^^^
//                         sizeof(uint32_t) * WARP_SIZE = 128 bytes

// Within kernel:
__global__ void kernelFunc(...) {
    // shmemBuf is allocated at kernel launch
    // Size: sharedMemSize bytes
    // Scope: Thread block (all threads in block can access)
}
```

### 8.2 Shared Memory Layout

From `cuPHY/src/cuphy/crc.cu` (line 417):

```cuda
// Complex shared memory layout for CRC processing
__global__ void crcKernel(...) {
    extern __shared__ uint32_t shmemBuf[];

    // Partition shared memory:
    // [Reduction buffer | Code block data]

    // Reduction buffer (for warp-level reductions)
    // Located at: shmemBuf[0 .. MAX_WORDS_PER_CODE_BLOCK-1]

    // Code block input data (byte-aligned)
    const uint8_t* shmem_CB_input_bytes =
        (uint8_t*)(shmemBuf + MAX_WORDS_PER_CODE_BLOCK)
        + ((CB_id * CB_data_byte_size) % 4);
    //    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    //    Offset to ensure byte alignment

    // Usage:
    // - First N words: Warp reduction workspace
    // - Remaining bytes: Code block data cache
}
```

### 8.3 Cooperative Memory Copy

From `cuPHY/src/cuphy/crc.cu` (lines 452-462):

```cuda
#if ((__CUDACC_VER_MAJOR__ >= 12) || \
     (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ >= 1))

    // Modern approach: Cooperative groups async copy
    auto group = cg::this_thread_block();
    cg::memcpy_async(
        group,                           // Cooperative group
        shmemBuf + MAX_WORDS_PER_CODE_BLOCK,  // Destination (shared)
        CB_input_words,                  // Source (global)
        sizeof(uint32_t) * CB_data_word_size  // Size
    );
    cg::wait(group);  // Barrier: wait for async copy

#else
    // Fallback for older CUDA versions
    for (int i = threadIdx.x; i < CB_data_word_size; i += blockDim.x) {
        shmemBuf[MAX_WORDS_PER_CODE_BLOCK + i] = CB_input_words[i];
    }
    __syncthreads();  // Explicit barrier

#endif
```

**Benefits of memcpy_async:**
- Overlaps copy with computation (if possible)
- Hardware-accelerated bulk transfer
- Reduces instruction count
- Better performance on Ampere+ architectures

### 8.4 Bank Conflict Avoidance

#### 8.4.1 Shared Memory Banks

```
Shared memory is divided into 32 banks (4-byte width):

Bank 0:  [0x0000] [0x0080] [0x0100] ...
Bank 1:  [0x0004] [0x0084] [0x0104] ...
Bank 2:  [0x0008] [0x0088] [0x0108] ...
...
Bank 31: [0x007C] [0x00FC] [0x017C] ...

Address to bank mapping:
bank_id = (address / 4) % 32
```

#### 8.4.2 Conflict-Free Access Pattern

```cuda
__shared__ float data[1024];

// GOOD: No bank conflicts
__global__ void noBankConflicts() {
    int tid = threadIdx.x;

    // Each thread in warp accesses different bank
    // Thread 0 → Bank 0, Thread 1 → Bank 1, ..., Thread 31 → Bank 31
    float value = data[tid];  // Conflict-free
}

// BAD: Bank conflicts
__global__ void withBankConflicts() {
    int tid = threadIdx.x;

    // All threads access same bank (bank 0)
    float value = data[tid * 32];  // 32-way bank conflict!
    // Serialized access: 32× slower
}
```

#### 8.4.3 Padding to Avoid Conflicts

```cuda
// Without padding: conflicts in column access
__shared__ float matrix[32][32];  // 32 rows, 32 columns

// Thread tid accesses matrix[row][tid]
// If row changes and tid stays same, bank conflict occurs

// With padding: conflict-free
__shared__ float matrix[32][33];  // Extra column for padding
//                            ^^
//                            Padding shifts bank alignment
```

### 8.5 Shared Memory Use Cases in cuPHY

#### 8.5.1 Warp Reduction Workspace

```cuda
// Shared memory for reduction across threads
extern __shared__ uint32_t shmem[];

// Each warp writes its partial result
if (lane == 0) {
    shmem[wid] = partial_result;
}
__syncthreads();

// First warp reduces across warp results
if (wid == 0) {
    uint32_t value = (threadIdx.x < numWarps) ? shmem[lane] : 0;
    value = warpReduceSum(value);
}
```

#### 8.5.2 Data Caching

From `cuPHY/src/cuphy/modulation_mapper.cu` (lines 245-250):

```cuda
__global__ void modulation_64QAM(...) {
    // Cache QAM constellation table in shared memory
    __shared__ __half shmem_qam_64[8];

    // Load table cooperatively
    if (threadIdx.x < 8) {
        assert(params != nullptr);
        shmem_qam_64[threadIdx.x] =
            (__half)(rev_qam_64[threadIdx.x] * params[blockIdx.y].beta_qam);
    }
    __syncthreads();

    // All threads can now access from fast shared memory
    int symbol_bits = getBits(...);
    __half2 symbol = make_half2(
        shmem_qam_64[symbol_bits >> 3],
        shmem_qam_64[symbol_bits & 0x7]
    );
}
```

---

## 9. Synchronization Mechanisms

### 9.1 Thread Block Synchronization

#### 9.1.1 __syncthreads() Barrier

From `cuPHY/src/cuphy/crc.cu` (extensively used):

```cuda
__global__ void kernel() {
    extern __shared__ uint32_t shmem[];

    // Phase 1: Load data to shared memory
    if (threadIdx.x == 0) {
        shmem[0] = globalData;
    }

    // BARRIER: Ensure all threads see the write
    __syncthreads();
    //^^^^^^^^^^^ All threads in block wait here
    //            until all have reached this point

    // Phase 2: Read from shared memory (safe now)
    uint32_t value = shmem[0];  // All threads see correct value

    // Process...

    // BARRIER: Before next write phase
    __syncthreads();

    // Phase 3: Another synchronized operation
    // ...
}
```

**Important Rules:**
1. Must be reached by ALL threads in block (or none)
2. Cannot be in divergent code paths
3. Only synchronizes within thread block (not across blocks)

**Bad Example (Deadlock):**
```cuda
// WRONG: Conditional barrier
if (threadIdx.x < 128) {
    __syncthreads();  // Only half the threads reach here
                      // Other half waiting forever → DEADLOCK
}
```

### 9.2 Warp-Level Primitives

#### 9.2.1 Warp Shuffle

From `cuPHY/src/cuphy/crc.cuh` (lines 59-89):

```cuda
// XOR reduction across warp using shuffle
template <typename uintCRC_t>
__inline__ __device__ uintCRC_t warpReduceSum(uintCRC_t val)
{
    // Butterfly reduction pattern
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        // Each thread gets value from neighbor
        val ^= __shfl_down_sync(FULL_MASK, val, offset, WARP_SIZE);
        //     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        //     Synchronous shuffle within warp
        //     - FULL_MASK (0xFFFFFFFF) = all 32 threads participate
        //     - offset = distance to source thread
        //     - WARP_SIZE = 32
    }
    return val;
}

// Execution example for offset=16:
// Thread  0 ← Thread 16
// Thread  1 ← Thread 17
// ...
// Thread 15 ← Thread 31
// Threads 16-31 provide data but don't update

// After 5 iterations (16, 8, 4, 2, 1):
// Thread 0 has XOR of all 32 values
```

**Shuffle Operations:**
```cuda
// Shuffle down: Get value from thread (tid + offset)
__shfl_down_sync(mask, value, offset, width);

// Shuffle up: Get value from thread (tid - offset)
__shfl_up_sync(mask, value, offset, width);

// Shuffle XOR: Get value from thread (tid ^ laneMask)
__shfl_xor_sync(mask, value, laneMask, width);

// Broadcast: All threads get value from srcLane
__shfl_sync(mask, value, srcLane, width);
```

#### 9.2.2 Warp-Level Reduction

From `cuPHY/src/cuphy/crc.cuh` (lines 71-89):

```cuda
// Full block reduction using warp primitives and shared memory
template <typename uintCRC_t>
__device__ inline uintCRC_t xorReductionWarpShared(
    uintCRC_t input, uintCRC_t* shared)
{
    int lane = threadIdx.x % WARP_SIZE;  // 0-31 within warp
    int wid = threadIdx.x / WARP_SIZE;   // Warp ID in block

    // Step 1: Reduce within each warp
    input = warpReduceSum<uintCRC_t>(input);
    // Now: Thread 0 of each warp has partial result

    // Step 2: Write warp results to shared memory
    if (lane == 0) {
        shared[wid] = input;  // Each warp leader writes
    }
    __syncthreads();  // Barrier: ensure all writes visible

    // Step 3: First warp reduces across warp results
    // Load value from shared memory (only for active warps)
    input = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0;

    if (wid == 0) {
        // Final reduction in first warp
        input = warpReduceSum<uintCRC_t>(input);
    }

    // Result in thread 0 of block
    return input;
}
```

**Execution Diagram (256-thread block = 8 warps):**
```
Initial state: Each thread has input value

Warp 0 (threads 0-31):   ┌─→ partial[0]
Warp 1 (threads 32-63):  ┌─→ partial[1]
Warp 2 (threads 64-95):  ┌─→ partial[2]
Warp 3 (threads 96-127): ┌─→ partial[3]
Warp 4 (threads 128-159):┌─→ partial[4]
Warp 5 (threads 160-191):┌─→ partial[5]
Warp 6 (threads 192-223):┌─→ partial[6]
Warp 7 (threads 224-255):└─→ partial[7]

                ↓ (Write to shared memory)

shared[0..7] = [partial[0], partial[1], ..., partial[7]]

                ↓ (Warp 0 reduces)

Thread 0 has final result = XOR of all partials
```

#### 9.2.3 Warp-Level Scan

From `cuPHY/src/cuphy/cuphy_kernel_util.cuh` (lines 382-397):

```cuda
template <typename T, unsigned int TLog2=5>
__device__ T warp_inclusive_scan(T value)
{
    const unsigned int LANEID = (threadIdx.x & 0x1F);  // Lane within warp

    // Log2(32) = 5 iterations for full warp
    #pragma unroll
    for (int i = 0; i < TLog2; ++i) {
        int shift = (1 << i);  // 1, 2, 4, 8, 16

        // Get value from thread (LANEID - shift)
        T neighbor = __shfl_up_sync(0xFFFFFFFF, value, shift);

        // Add if we have a valid neighbor
        value = ((LANEID >= shift) ? value : 0) + neighbor;
    }

    return value;  // Each thread has inclusive prefix sum
}

// Example execution (8 threads for clarity):
// Input:     [1, 2, 3, 4, 5, 6, 7, 8]
// Iter 1: +  [-, 1, 2, 3, 4, 5, 6, 7]  (shift=1)
// Result:    [1, 3, 5, 7, 9,11,13,15]
// Iter 2: +  [-,-, 1, 3, 5, 7, 9,11]  (shift=2)
// Result:    [1, 3, 6,10,14,18,22,26]
// Iter 3: +  [-,-,-,-, 1, 3, 6,10]    (shift=4)
// Final:     [1, 3, 6,10,15,21,28,36]  (inclusive scan)
```

### 9.3 Atomic Operations

#### 9.3.1 Standard Atomic Operations

From `cuPHY/src/cuphy/crc.cu` (lines 236, 720):

```cuda
// Atomic XOR for TB CRC accumulation
if (threadIdx.x == 0) {
    // Each block XORs its partial CRC into TB CRC
    atomicXor(&outputTBCRCs[blockIdx.y], crc);
    //^^^^^^^^ Thread-safe XOR operation
    //         Serializes across all blocks processing this TB
}

// Other standard atomic operations:
atomicAdd(&address, value);    // address = address + value
atomicSub(&address, value);    // address = address - value
atomicExch(&address, value);   // exchange (swap)
atomicMin(&address, value);    // address = min(address, value)
atomicMax(&address, value);    // address = max(address, value)
atomicInc(&address, value);    // increment with wrap
atomicDec(&address, value);    // decrement with wrap
atomicAnd(&address, value);    // bitwise AND
atomicOr(&address, value);     // bitwise OR
atomicXor(&address, value);    // bitwise XOR
atomicCAS(&addr, cmp, val);    // compare-and-swap
```

#### 9.3.2 Custom Atomic Operations

From `cuPHY/src/cuphy/rate_matching.cu` (lines 142-202):

```cuda
// Generic atomic max wrapper
template <typename T>
__device__ inline T atomicMaxCustom(T* address, T val) {
    return atomicMax(address, val);  // For native types
}

// Specialized for __half (no native support)
template <>
__device__ inline __half atomicMaxCustom<__half>(
    __half* address, __half val)
{
    // Implement using atomicCAS on underlying ushort
    unsigned short* address_as_ushort =
        reinterpret_cast<unsigned short*>(address);
    unsigned short old = *address_as_ushort, assumed;

    do {
        assumed = old;
        // Compute max in half precision
        __half assumed_half = __ushort_as_half(assumed);
        __half new_val = __hmax(val, assumed_half);

        // Try to atomically update
        old = atomicCAS(address_as_ushort, assumed,
                       __half_as_ushort(new_val));

        // Loop until successful (no contention)
    } while (assumed != old);

    return __ushort_as_half(old);
}

// Usage in rate matching (lines 269-271):
if (llr > LLR_CLAMP_MAX)
    atomicMinCustom(out + outIdx, LLR_CLAMP_MAX);
else if (llr < LLR_CLAMP_MIN)
    atomicMaxCustom(out + outIdx, LLR_CLAMP_MIN);
```

**Atomic Performance Considerations:**
- Serialize operations on same address
- High contention = poor performance
- Use when necessary for correctness
- Prefer non-atomic algorithms when possible

---

## 10. Warp-Level Primitives

### 10.1 Warp Voting Functions

```cuda
// All threads in warp test condition
__global__ void voteExample() {
    int tid = threadIdx.x;
    bool predicate = (tid % 2 == 0);  // Even threads

    // Check if ALL threads have predicate = true
    bool all_true = __all_sync(0xFFFFFFFF, predicate);
    // Result: false (not all threads are even)

    // Check if ANY thread has predicate = true
    bool any_true = __any_sync(0xFFFFFFFF, predicate);
    // Result: true (some threads are even)

    // Count threads with predicate = true
    unsigned int count = __popc(__ballot_sync(0xFFFFFFFF, predicate));
    // Result: 16 (half the threads)
}
```

### 10.2 Warp Match Functions

```cuda
// Find threads with same value
__global__ void matchExample() {
    int value = threadIdx.x / 4;  // 0,0,0,0,1,1,1,1,2,2,2,2,...

    // Get mask of threads with same value
    unsigned int match_mask = __match_any_sync(0xFFFFFFFF, value);

    // For thread 0: match_mask = 0x0000000F (threads 0-3)
    // For thread 4: match_mask = 0x000000F0 (threads 4-7)
}
```

### 10.3 Warp Aggregation Pattern

```cuda
// Efficient global atomic using warp aggregation
__device__ void warpAtomicAdd(int* global_counter, int value) {
    // Step 1: Reduce within warp
    for (int offset = 16; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xFFFFFFFF, value, offset);
    }

    // Step 2: Single thread performs atomic
    if ((threadIdx.x % 32) == 0) {
        atomicAdd(global_counter, value);
    }

    // Result: 32 atomic ops reduced to 1 atomic op
    // 32× improvement in atomic performance
}
```

---

## 11. Atomic Operations

### 11.1 Atomic Add for LLR Combining

From `cuPHY/src/cuphy/rate_matching.cu` (lines 257-275):

```cuda
template <typename T_OUT>
__device__ __forceinline__ void processOneLLR(...) {
    const uint32_t outIdx = derate_match_fast_calc_modulo(inIdx, Kd, F, k0, Ncb);
    const bool useAtomics = (potentialRaceIfPositive > 0) &&
                           (outIdx < potentialRaceIfPositive);

    if (ndi) {  // New data indication
        if (!useAtomics) {
            out[outIdx] = llr;  // Direct write (no race)
        } else {
            // Atomic add with clamping
            T_OUT prev = atomicAdd(out + outIdx, llr);
            llr += prev;

            // Clamp to prevent overflow
            if (llr > LLR_CLAMP_MAX)
                atomicMinCustom(out + outIdx, LLR_CLAMP_MAX);
            else if (llr < LLR_CLAMP_MIN)
                atomicMaxCustom(out + outIdx, LLR_CLAMP_MIN);
        }
    }
}
```

### 11.2 Atomic CAS (Compare-And-Swap)

```cuda
// Generic atomic operation using CAS
__device__ float atomicAddFloat(float* address, float val) {
    unsigned int* address_as_uint = (unsigned int*)address;
    unsigned int old = *address_as_uint, assumed;

    do {
        assumed = old;

        // Compute new value
        float new_val = __uint_as_float(assumed) + val;

        // Atomically update if unchanged
        old = atomicCAS(address_as_uint, assumed,
                       __float_as_uint(new_val));

        // Retry if another thread modified
    } while (assumed != old);

    return __uint_as_float(old);
}
```

---

## 12. CRC Kernel Implementation

### 12.1 Code Block CRC Kernel

From `cuPHY/src/cuphy/crc.cu` (lines 365-615):

```cuda
__global__ void crcDownlinkPdschCodeBlocksKernel(
    const __grid_constant__ crcEncodeDescr_t desc)
{
    // Descriptor fields (from constant memory)
    uint32_t crcType = desc.crcType;  // CRC-16 or CRC-24
    uint32_t nCodeBlocks = desc.nCodeBlocks;
    uint32_t* inputCodeBlocks = desc.inputCodeBlocks;
    uint32_t* outputCodeBlocks = desc.outputCodeBlocks;
    uint32_t* outputCBCRCs = desc.outputCBCRCs;
    bool reverseBytes = desc.reverseBytes;

    // Thread and block indices
    int codeBlockIdx = blockIdx.x;  // Which code block
    int tbIdx = blockIdx.y;          // Which TB
    int tid = threadIdx.x;           // Thread in block (0-255)

    // Early exit if no CBs to process
    if (codeBlockIdx >= nCodeBlocks) return;

    // Calculate base offsets
    uint32_t cbBase = tbIdx * MAX_WORDS_PER_CODE_BLOCK * nCodeBlocks;
    uint32_t size = cbSize / 32;  // Size in 32-bit words

    // Initialize CRC
    uint32_t crc = 0;

    // Grid-stride loop over code block words
    while (tid < size) {
        // Read input word (coalesced access)
        uint32_t inVal = inputCodeBlocks[
            cbBase + codeBlockIdx * MAX_WORDS_PER_CODE_BLOCK + tid
        ];

        // Byte reversal if needed
        if (reverseBytes) {
            inVal = __brev(inVal);      // Bit reverse
            inVal = swap<32>(inVal);    // Byte swap
        }

        // CRC computation (polynomial division)
        crc = crcWord<uint32_t, 32>(crc, inVal, crcPoly);

        tid += blockDim.x;  // Grid stride
    }

    // Warp-level XOR reduction
    extern __shared__ uint32_t shmemBuf[];
    crc = xorReductionWarpShared<uint32_t>(crc, shmemBuf);

    // Thread 0 writes result
    if (threadIdx.x == 0) {
        outputCBCRCs[nCodeBlocksSum + codeBlockIdx] = crc;
    }
}
```

### 12.2 CRC Polynomial Functions

From `cuPHY/src/cuphy/crc.cuh` (lines 94-103):

```cuda
// Multiply two polynomials modulo poly
template <typename T, uint32_t size>
__device__ T mulModPoly(T a, T b, T poly)
{
    T prod = 0;

    #pragma unroll  // Unroll for performance
    for (int i = 0; i < size; i++) {
        // Add a to product if b's LSB is 1
        prod ^= (b & 1) ? a : 0;

        // Multiply a by x (shift left)
        // Reduce modulo poly if MSB is 1
        a = (a << 1) ^ ((a & (1 << (size - 1))) ? poly : 0);

        // Divide b by x (shift right)
        b >>= 1;
    }

    return prod;
}

// CRC computation for one word
template <typename T, uint32_t size>
__device__ T crcWord(T crc, T word, T poly)
{
    T remainder = word;

    #pragma unroll
    for (int i = 0; i < size; i++) {
        // Check if MSB is 1
        bool msb = (crc & (1 << (size - 1)));

        // Shift CRC left by 1
        crc <<= 1;

        // Add next bit from word
        crc ^= ((remainder >> (size - 1 - i)) & 1);

        // XOR with poly if MSB was 1
        if (msb) crc ^= poly;
    }

    return crc;
}
```

### 12.3 Transport Block CRC Kernel

From `cuPHY/src/cuphy/crc.cu` (lines 618-722):

```cuda
__global__ void crcUplinkPuschTransportBlocksKernel(
    const __grid_constant__ crcEncodeDescr_t desc)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int tbIdx = blockIdx.y;

    // Get TB info
    uint32_t tbSize = desc.tbSizes[tbIdx];
    uint32_t nCodeBlocks = desc.nCodeBlocks;

    // Output pointers
    uint32_t* outputTBCRCs = desc.outputTBCRCs;
    uint32_t* outputCBCRCs = desc.outputCBCRCs;

    // XOR all CB CRCs to get TB CRC
    uint32_t crc = 0;

    // Grid-stride loop over code blocks
    for (uint32_t cbIdx = tid; cbIdx < nCodeBlocks; cbIdx += stride) {
        uint32_t cbCRC = outputCBCRCs[nCodeBlocksSum + cbIdx];
        crc ^= cbCRC;
    }

    // Warp-level reduction
    extern __shared__ uint32_t shmemBuf[];
    crc = xorReductionWarpShared<uint32_t>(crc, shmemBuf);

    // Atomic XOR into global TB CRC
    // (Multiple blocks may process same TB)
    if (threadIdx.x == 0) {
        atomicXor(&outputTBCRCs[tbIdx], crc);
    }
}
```

---

## 13. LDPC Encoding Kernels

### 13.1 LDPC Kernel Architecture

LDPC encoding is the most computationally intensive operation in PDSCH processing. The kernels are not fully visible in the shared source, but the pattern follows:

```cuda
// LDPC encoding kernel structure (inferred)
__global__ void ldpcEncodeKernel(
    const uint8_t* inputBits,     // Information bits
    uint8_t* outputBits,           // Encoded bits (info + parity)
    const ldpc_params_t* params,   // LDPC parameters
    int numCodeBlocks)
{
    int cbIdx = blockIdx.x;
    int tbIdx = blockIdx.y;

    if (cbIdx >= numCodeBlocks) return;

    // LDPC parameters
    int K = params->K;  // Information columns
    int N = params->N;  // Total columns
    int Z = params->Z;  // Lifting size

    // Base graph selection (BG1 or BG2)
    const uint8_t* baseGraph = (K > 10) ? baseGraph1 : baseGraph2;

    // Parity computation using base graph
    // Each thread computes subset of parity bits
    computeParityBits(inputBits, outputBits, baseGraph, K, N, Z);
}
```

### 13.2 Base Graph Structure

**LDPC Base Graphs:**

```
Base Graph 1 (BG1): For large code blocks
- K_max = 22 information columns
- N = 66 total columns (including parity)
- Code rate: ~1/3

Base Graph 2 (BG2): For small code blocks
- K_max = 10 information columns
- N = 50 total columns
- Code rate: ~1/5

Lifting Sizes (Z): {2, 3, 4, ..., 384}
Actual matrix size: (N-K)*Z × N*Z
```

### 13.3 LDPC Encoding Algorithm

```cuda
// Simplified LDPC encoding (conceptual)
__device__ void computeParityBits(
    const uint8_t* info,
    uint8_t* encoded,
    const uint8_t* baseGraph,
    int K, int N, int Z)
{
    int tid = threadIdx.x;

    // Copy information bits
    for (int i = tid; i < K*Z; i += blockDim.x) {
        encoded[i] = info[i];
    }
    __syncthreads();

    // Compute parity bits (systematic encoding)
    // P = (H_B^T)^(-1) * H_A^T * info
    // where H = [H_A | H_B] is parity check matrix

    for (int row = tid; row < (N-K)*Z; row += blockDim.x) {
        uint8_t parity = 0;

        // Matrix-vector multiply for this parity bit
        for (int col = 0; col < K*Z; col++) {
            int baseRow = row / Z;
            int baseCol = col / Z;
            int shiftRow = row % Z;
            int shiftCol = col % Z;

            // Get matrix element (with cyclic shift)
            int shift = baseGraph[baseRow * N + baseCol];
            if (shift >= 0) {
                int actualCol = (shiftCol + shift) % Z;
                if (actualCol == shiftRow) {
                    parity ^= info[col];
                }
            }
        }

        encoded[K*Z + row] = parity;
    }
}
```

### 13.4 LDPC Performance Optimization

**Shared Memory for Base Graph:**
```cuda
__shared__ int8_t sh_baseGraph[68 * 22];  // BG1 max size

// Load cooperatively
for (int i = threadIdx.x; i < graphSize; i += blockDim.x) {
    sh_baseGraph[i] = baseGraph[i];
}
__syncthreads();

// Use shared memory for fast access
```

**Vectorized Bit Operations:**
```cuda
// Process 32 bits at a time using uint32_t
const uint32_t* info32 = (const uint32_t*)info;
uint32_t* encoded32 = (uint32_t*)encoded;

// Parity computation on words
uint32_t parity_word = 0;
for (int i = 0; i < K*Z/32; i++) {
    parity_word ^= (info32[i] & mask[i]);
}
```

---

## 14. Rate Matching Kernels

### 14.1 Rate Matching Kernel Structure

From `cuPHY/src/cuphy/rate_matching.cu` (lines 204-276):

```cuda
template <typename T_OUT>
__device__ __forceinline__ void processOneLLR(
    uint32_t jIdx,              // Index within QAM symbol
    uint32_t kIdx,              // Symbol index
    T_OUT llr,                  // LLR value to write
    int EoverQm,                // Output size / modulation order
    uint32_t Kd,                // Encoded size with dummy bits
    uint32_t F,                 // Number of filler bits
    uint32_t k0,                // Starting position
    uint32_t Ncb,               // Circular buffer size
    int potentialRaceIfPositive, // Race condition indicator
    bool ndi,                   // New data indication
    bool descramblingOn,        // Descrambling enable
    T_OUT LLR_CLAMP_MIN,        // Min LLR value
    T_OUT LLR_CLAMP_MAX,        // Max LLR value
    T_OUT* __restrict__ out)    // Output buffer
{
    if (jIdx >= static_cast<uint32_t>(EoverQm)) return;

    // Calculate circular buffer index
    const uint32_t inIdx = kIdx * EoverQm + jIdx;
    const uint32_t outIdx = derate_match_fast_calc_modulo(
        inIdx, Kd, F, k0, Ncb
    );

    // Determine if atomic operations needed
    const bool useAtomics =
        (potentialRaceIfPositive > 0) &&
        (outIdx < potentialRaceIfPositive);

    if (ndi) {
        // New data: Write LLR
        if (!useAtomics) {
            out[outIdx] = llr;
        } else {
            // Atomic write with clamping
            T_OUT prev = atomicAdd(out + outIdx, llr);
            llr += prev;

            if (llr > LLR_CLAMP_MAX)
                atomicMinCustom(out + outIdx, LLR_CLAMP_MAX);
            else if (llr < LLR_CLAMP_MIN)
                atomicMaxCustom(out + outIdx, LLR_CLAMP_MIN);
        }
    } else {
        // Retransmission: Combine with existing LLR
        if (!useAtomics) {
            llr += out[outIdx];
            llr = max(min(LLR_CLAMP_MAX, llr), LLR_CLAMP_MIN);
            out[outIdx] = llr;
        } else {
            T_OUT prev_llr = atomicAdd(out + outIdx, llr);
            llr += prev_llr;

            if (llr > LLR_CLAMP_MAX)
                atomicMinCustom(out + outIdx, LLR_CLAMP_MAX);
            else if (llr < LLR_CLAMP_MIN)
                atomicMaxCustom(out + outIdx, LLR_CLAMP_MIN);
        }
    }
}
```

### 14.2 Circular Buffer Indexing

```cuda
// Fast modulo for circular buffer access
__device__ __forceinline__ uint32_t derate_match_fast_calc_modulo(
    uint32_t inIdx, uint32_t Kd, uint32_t F,
    uint32_t k0, uint32_t Ncb)
{
    // Map input index to circular buffer position
    uint32_t idx = inIdx + k0;

    // Fast modulo using conditional subtraction
    while (idx >= Ncb) {
        idx -= Ncb;
    }

    // Skip filler bits
    if (idx >= Kd - F) {
        idx += F;
    }

    return idx;
}
```

### 14.3 Vectorized Zeroing

From `cuPHY/src/cuphy/rate_matching.cu` (lines 281-300):

```cuda
template <typename T_OUT>
__device__ __forceinline__ void zeroRangeVec(
    T_OUT* __restrict__ out,
    uint32_t start, uint32_t end,
    uint32_t tid, uint32_t stride)
{
    if (start >= end) return;

    T_OUT* base = out + start;
    uint32_t total = end - start;

    // Align to 16-byte boundary
    const uintptr_t addr = reinterpret_cast<uintptr_t>(base);
    const uint32_t bytes_to_align = (16u - (addr & 15u)) & 15u;
    const uint32_t head = min(total,
        bytes_to_align / static_cast<uint32_t>(sizeof(T_OUT)));

    // Scalar zeroing for unaligned head
    for (uint32_t i = tid; i < head; i += stride) {
        base[i] = 0;
    }

    // Vectorized zeroing for aligned body
    uint4* vec_base = reinterpret_cast<uint4*>(base + head);
    uint32_t vec_count = (total - head) /
        (sizeof(uint4) / sizeof(T_OUT));

    for (uint32_t i = tid; i < vec_count; i += stride) {
        vec_base[i] = make_uint4(0, 0, 0, 0);  // 16-byte write
    }

    // Scalar zeroing for tail
    uint32_t tail_start = head +
        vec_count * (sizeof(uint4) / sizeof(T_OUT));
    for (uint32_t i = tid + tail_start; i < total; i += stride) {
        base[i] = 0;
    }
}
```

---

## 15. Scrambling and Descrambling

### 15.1 Gold Sequence Generation

From `cuPHY/src/cuphy/descrambling.cu` (inferred):

```cuda
// Generate 32-bit Gold sequence
__device__ uint32_t gold32(uint32_t c_init, uint32_t n)
{
    // Initialize X1 and X2 sequences
    uint32_t x1 = 1;  // X1 always starts with 1
    uint32_t x2 = c_init;

    // Advance sequences by n positions
    for (int i = 0; i < n; i++) {
        // X1: x1(n+31) = (x1(n+3) + x1(n)) mod 2
        uint32_t x1_new = ((x1 >> 3) ^ x1) & 1;
        x1 = (x1 >> 1) | (x1_new << 30);

        // X2: x2(n+31) = (x2(n+3) + x2(n+2) + x2(n+1) + x2(n)) mod 2
        uint32_t x2_new = ((x2 >> 3) ^ (x2 >> 2) ^ (x2 >> 1) ^ x2) & 1;
        x2 = (x2 >> 1) | (x2_new << 30);
    }

    // Gold sequence: c(n) = (x1(n) + x2(n)) mod 2
    return x1 ^ x2;
}
```

### 15.2 Descrambling Kernel

From `cuPHY/src/cuphy/descrambling.cu` (lines 34-68):

```cuda
__global__ void descrambleKernel(
    float* llrs,                        // LLR values (in/out)
    uint32_t size,                      // Total size
    const uint32_t* tbBoundaryArray,    // TB start/end indices
    const uint32_t* cinitArray)         // c_init values per TB
{
    extern __shared__ uint32_t sharedSeq[];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // TB boundaries for this block
    uint32_t myTBBase = tbBoundaryArray[blockIdx.y];
    uint32_t myTBEnd = tbBoundaryArray[blockIdx.y + 1];
    uint32_t myCinit = cinitArray[blockIdx.y];

    // Round up to block boundary
    uint32_t blockEnd = myTBEnd +
        ((blockDim.x - myTBEnd % blockDim.x) % blockDim.x);

    // Grid-stride loop
    for (int t = tid + myTBBase; t < blockEnd; t += stride) {
        __syncthreads();

        // One warp generates Gold sequence
        if (threadIdx.x < blockDim.x / WARP_SIZE) {
            int seqIdx = t + threadIdx.x * WARP_SIZE - myTBBase;
            sharedSeq[threadIdx.x] = gold32(myCinit, seqIdx);
        }

        __syncthreads();

        if (t < myTBEnd) {
            // Get scrambling bit for this thread
            uint32_t seq = sharedSeq[threadIdx.x / WARP_SIZE];
            uint32_t s = (seq >> (threadIdx.x & (WARP_SIZE - 1))) & 1;

            // Sign flip: LLR *= (1 - 2*s) = -LLR if s=1, LLR if s=0
            uint32_t sn = (s + 1) & 0x1;
            llrs[t] = -llrs[t] * s + llrs[t] * sn;
        }
    }
}
```

**Key Optimizations:**
- Shared memory for Gold sequence (computed once per block)
- Warp-level generation (one thread per 32 bits)
- Efficient sign flip using arithmetic (avoids branch)

---

## 16. Modulation Mapping Kernels

### 16.1 64-QAM Modulation Kernel

From `cuPHY/src/cuphy/modulation_mapper.cu` (lines 245-280):

```cuda
__device__ void modulation_64QAM(
    const PdschDmrsParams* __restrict__ params,
    const uint32_t* __restrict__ modulation_input,
    __half2* __restrict__ modulation_output,
    const struct PdschPerTbParams* workspace,
    int max_bits_per_layer)
{
    // Load QAM constellation to shared memory
    __shared__ __half shmem_qam_64[8];

    if (threadIdx.x < 8) {
        assert(params != nullptr);
        shmem_qam_64[threadIdx.x] = (__half)(
            rev_qam_64[threadIdx.x] * params[blockIdx.y].beta_qam
        );
    }
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_symbols = workspace[blockIdx.y].G / CUPHY_QAM_64;

    if (tid >= num_symbols) return;

    // Complex mapping logic for multi-layer MIMO
    // Extract 6 bits for 64-QAM symbol
    int bit_index = tid * 6;
    uint32_t word_idx = bit_index / 32;
    uint32_t bit_offset = bit_index % 32;

    uint32_t bits = (modulation_input[word_idx] >> bit_offset) & 0x3F;

    // If crossing word boundary, get remaining bits
    if (bit_offset > 26) {
        bits |= (modulation_input[word_idx + 1] << (32 - bit_offset)) & 0x3F;
    }

    // Map 6 bits to I/Q components
    // Bits: [b5 b4 b3 b2 b1 b0]
    // I: shmem_qam_64[b5 b4 b3] (3 bits → 8 levels)
    // Q: shmem_qam_64[b2 b1 b0] (3 bits → 8 levels)

    __half real_part = shmem_qam_64[(bits >> 3) & 0x7];
    __half imag_part = shmem_qam_64[bits & 0x7];

    // Pack into __half2 (complex number)
    modulation_output[tid] = make_half2(real_part, imag_part);
}
```

### 16.2 QAM Constellation Tables

```cuda
// 64-QAM constellation (normalized to unit average power)
__constant__ float rev_qam_64[8] = {
    -7.0f / sqrtf(42.0f),  // -1.0801
    -5.0f / sqrtf(42.0f),  // -0.7715
    -3.0f / sqrtf(42.0f),  // -0.4629
    -1.0f / sqrtf(42.0f),  // -0.1543
     1.0f / sqrtf(42.0f),  //  0.1543
     3.0f / sqrtf(42.0f),  //  0.4629
     5.0f / sqrtf(42.0f),  //  0.7715
     7.0f / sqrtf(42.0f)   //  1.0801
};

// 256-QAM constellation
__constant__ float rev_qam_256[16] = {
    -15.0f / sqrtf(170.0f), -13.0f / sqrtf(170.0f),
    -11.0f / sqrtf(170.0f),  -9.0f / sqrtf(170.0f),
     -7.0f / sqrtf(170.0f),  -5.0f / sqrtf(170.0f),
     -3.0f / sqrtf(170.0f),  -1.0f / sqrtf(170.0f),
      1.0f / sqrtf(170.0f),   3.0f / sqrtf(170.0f),
      5.0f / sqrtf(170.0f),   7.0f / sqrtf(170.0f),
      9.0f / sqrtf(170.0f),  11.0f / sqrtf(170.0f),
     13.0f / sqrtf(170.0f),  15.0f / sqrtf(170.0f)
};
```

---

## 17. DMRS Generation Kernels

### 17.1 DMRS Gold Sequence

```cuda
// DMRS uses similar Gold sequence as scrambling
__device__ uint32_t dmrs_gold(uint32_t n_scid, uint32_t n_slot,
                              uint32_t n_symb, uint32_t n_id)
{
    // c_init calculation for DMRS
    uint32_t c_init = ((1 << 17) * (14 * n_slot + n_symb + 1) *
                      (2 * n_id + 1) + 2 * n_id + n_scid) & 0x7FFFFFFF;

    return gold32(c_init, 0);
}
```

### 17.2 DMRS Resource Mapping

```cuda
__global__ void dmrsGenerationKernel(
    cuFloatComplex* output,          // Output grid
    const PdschDmrsParams* params,   // DMRS parameters
    int nPRBs, int nSymbols)
{
    int prb = blockIdx.x;
    int symbol = blockIdx.y;
    int port = blockIdx.z;

    if (prb >= nPRBs || symbol >= nSymbols) return;

    // Check if this symbol contains DMRS
    bool isDmrsSymbol = params->dmrsSymbolMask & (1 << symbol);
    if (!isDmrsSymbol) return;

    // Generate Gold sequence for this PRB/symbol
    uint32_t seq = dmrs_gold(
        params->nSCID,
        params->slotNum,
        symbol,
        params->scramblingID
    );

    // Map DMRS to subcarriers
    for (int k = threadIdx.x; k < 12; k += blockDim.x) {
        // DMRS pattern (Type 1 or Type 2)
        bool isDmrsSubcarrier = isDmrsLocation(k, params->dmrsType);

        if (isDmrsSubcarrier) {
            // Extract real/imag from sequence
            int bit_idx = k * 2;
            float real = ((seq >> bit_idx) & 1) ? 1.0f : -1.0f;
            float imag = ((seq >> (bit_idx+1)) & 1) ? 1.0f : -1.0f;

            // Apply power boosting
            real *= params->dmrsEnergy;
            imag *= params->dmrsEnergy;

            // Write to output
            int grid_idx = symbol * nPRBs * 12 + prb * 12 + k;
            output[grid_idx] = make_cuFloatComplex(real, imag);
        }
    }
}
```

---

## 18. Performance Optimization Techniques

### 18.1 Loop Unrolling

From `cuPHY/src/cuphy/crc.cuh` (lines 94-103):

```cuda
template <typename T, uint32_t size>
__device__ T mulModPoly(T a, T b, T poly)
{
    T prod = 0;

    #pragma unroll  // Compiler directive to unroll loop
    for (int i = 0; i < size; i++) {
        prod ^= (b & 1) ? a : 0;
        a = (a << 1) ^ ((a & (1 << (size - 1))) ? poly : 0);
        b >>= 1;
    }

    return prod;
}

// With #pragma unroll, compiler generates:
// prod ^= (b & 1) ? a : 0; a = ...; b >>= 1;  // i=0
// prod ^= (b & 1) ? a : 0; a = ...; b >>= 1;  // i=1
// prod ^= (b & 1) ? a : 0; a = ...; b >>= 1;  // i=2
// ... (no loop overhead)
```

### 18.2 Function Inlining

```cuda
// Force inlining for small functions
template <typename T>
__device__ __forceinline__ T max(T a, T b) {
    return (a > b) ? a : b;
}

// Inline helps because:
// - Eliminates function call overhead
// - Enables further optimizations
// - Better register allocation
```

### 18.3 Restrict Pointers

From `cuPHY/src/cuphy/rate_matching.cu`:

```cuda
__device__ void processLLR(
    const float* __restrict__ input,   // No aliasing with output
    float* __restrict__ output)        // No aliasing with input
{
    // __restrict__ tells compiler that pointers don't overlap
    // Enables:
    // - More aggressive optimizations
    // - Better instruction scheduling
    // - Potential vectorization
}
```

### 18.4 Constant Memory

From `cuPHY/src/cuphy/crc.cu` (lines 49-52):

```cuda
// Polynomials in constant memory (fast broadcast read)
__constant__ uint32_t POLY_A[1] = {G_CRC_24_A};
__constant__ uint32_t POLY_B[1] = {G_CRC_24_B};
__constant__ uint32_t POLY_16[1] = {G_CRC_16};

// All threads in warp read POLY_A[0] simultaneously
// Single memory transaction, broadcast to all 32 threads
```

### 18.5 PTX Intrinsics

From `cuPHY/src/cuphy/crc.cuh` (lines 31-45):

```cuda
template <int bitsize>
static __device__ inline uint32_t swap(uint32_t val)
{
    int size = bitsize >> 3;
    switch (size) {
    case 4:
        // PTX instruction for byte permutation
        return __byte_perm(val, 0, 0x0123);
    case 3:
        return ((val & 0xFF) << 16) +
               (val & 0xFF00) +
               ((val & 0xFF0000) >> 16);
    // ...
    }
}

// __byte_perm: Single instruction byte swap
// __brev: Bit reversal
// __popc: Population count (count 1 bits)
// __ffs: Find first set bit
```

---

## 19. Occupancy Analysis

### 19.1 Occupancy Metrics

```cuda
// Query device properties
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, deviceId);

printf("Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
printf("Max blocks per SM: %d\n", prop.maxBlocksPerMultiProcessor);
printf("Max warps per SM: %d\n",
       prop.maxThreadsPerMultiProcessor / 32);
printf("Shared memory per SM: %zu KB\n",
       prop.sharedMemPerMultiprocessor / 1024);
printf("Registers per SM: %d\n", prop.regsPerMultiprocessor);
```

### 19.2 Occupancy Calculator

```cuda
int blockSize = 256;
int minGridSize, optimalBlockSize;

cudaOccupancyMaxPotentialBlockSize(
    &minGridSize,
    &optimalBlockSize,
    crcKernel,
    dynamicSharedMemPerBlock,
    maxBlocksPerSM
);

printf("Optimal block size: %d\n", optimalBlockSize);

// Check achieved occupancy
int numBlocks;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &numBlocks,
    crcKernel,
    blockSize,
    dynamicSharedMemPerBlock
);

float occupancy = (numBlocks * blockSize) /
    (float)prop.maxThreadsPerMultiProcessor;
printf("Occupancy: %.2f%%\n", occupancy * 100);
```

### 19.3 Resource Limitations

```
Occupancy Limited By:

1. Registers per thread
   - Max 255 registers per thread
   - Fewer registers = more blocks per SM

2. Shared memory per block
   - 48-164 KB per SM (configurable)
   - More shared mem = fewer blocks per SM

3. Block size
   - Too small: underutilizes SM
   - Too large: limits number of blocks

4. Warps per SM
   - Max 64 warps per SM (H100)
   - 256-thread block = 8 warps
   - Max 8 blocks per SM
```

---

## 20. Memory Bandwidth Optimization

### 20.1 Coalescing Rules

```
Perfect Coalescing Requirements:

1. Consecutive threads access consecutive addresses
   Thread 0 → addr[0]
   Thread 1 → addr[1]
   ...
   Thread 31 → addr[31]

2. Alignment: First address must be aligned to 32 bytes

3. Data type: 4-byte or 8-byte types preferred

4. Access pattern: Sequential, no gaps
```

### 20.2 Bandwidth Measurement

```cuda
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

size_t bytes = nElements * sizeof(float);

cudaEventRecord(start);
kernel<<<grid, block>>>(d_data, nElements);
cudaEventRecord(stop);

cudaEventSynchronize(stop);

float milliseconds;
cudaEventElapsedTime(&milliseconds, start, stop);

float bandwidth = (bytes / (milliseconds / 1000.0)) / 1e9;
printf("Effective bandwidth: %.2f GB/s\n", bandwidth);

// Compare with theoretical maximum
float theoretical = prop.memoryClockRate * 1000.0 *
                   (prop.memoryBusWidth / 8) * 2 / 1e9;
printf("Theoretical: %.2f GB/s\n", theoretical);
printf("Efficiency: %.1f%%\n", (bandwidth / theoretical) * 100);
```

---

## 21. Register Usage Optimization

### 21.1 Register Pressure

```cuda
// Check register usage
nvcc -Xptxas -v kernel.cu

// Output:
// ptxas info : Used 32 registers, 128 bytes smem,
//              364 bytes cmem[0]
//              ^^^ registers per thread

// High register count limits occupancy
// Target: <64 registers for good occupancy
```

### 21.2 Reducing Register Usage

```cuda
// BAD: High register pressure
__global__ void highRegisterKernel() {
    float a, b, c, d, e, f, g, h;  // Many variables
    float x1, x2, x3, x4, x5;       // = high register count

    // Complex computation using all variables
    // Compiler needs registers for all of them
}

// GOOD: Lower register pressure
__global__ void lowRegisterKernel() {
    // Reuse variables
    float temp;

    temp = compute1();
    output[0] = temp;

    temp = compute2();  // Reuse same register
    output[1] = temp;
}
```

---

## 22. Best Practices and Recommendations

### 22.1 Kernel Launch Best Practices

**1. Block Size Selection:**
```cuda
// Recommended block sizes: 128, 256, 512
// cuPHY standard: 256 threads (8 warps)

const int BLOCK_SIZE = 256;
dim3 block(BLOCK_SIZE, 1, 1);
dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, nTBs, 1);
```

**2. Grid Sizing:**
```cuda
// Ensure enough blocks to saturate GPU
int numSMs = prop.multiProcessorCount;
int blocksPerSM = 4;  // Adjust based on occupancy
int minBlocks = numSMs * blocksPerSM;

int numBlocks = max(minBlocks, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
```

### 22.2 Memory Access Best Practices

**1. Coalesced Access:**
```cuda
// GOOD: Grid-stride loop
int tid = blockIdx.x * blockDim.x + threadIdx.x;
int stride = blockDim.x * gridDim.x;
for (int i = tid; i < N; i += stride) {
    output[i] = input[i];  // Coalesced
}
```

**2. Shared Memory Usage:**
```cuda
// Use shared memory for:
// - Reused data (lookup tables)
// - Block-level communication
// - Reduction operations

__shared__ float cache[BLOCK_SIZE];
cache[threadIdx.x] = input[globalIdx];
__syncthreads();
// Use cache...
```

### 22.3 Synchronization Best Practices

**1. Minimize __syncthreads():**
```cuda
// Use only when necessary
// Each barrier has ~10-20 cycle overhead

// GOOD: Single barrier per phase
__syncthreads();  // After load
// Process...
__syncthreads();  // Before store
```

**2. Prefer Warp-Level Primitives:**
```cuda
// No synchronization overhead within warp
value = __shfl_down_sync(0xFFFFFFFF, value, 1);

// Faster than shared memory + __syncthreads()
```

### 22.4 Performance Tuning Checklist

```
□ Occupancy > 50% (check with profiler)
□ Block size: 128-512 threads
□ Coalesced memory access
□ Shared memory for frequently accessed data
□ Minimize __syncthreads() barriers
□ Use __restrict__ on pointers
□ #pragma unroll on small loops
□ Register usage < 64 per thread
□ Avoid warp divergence
□ Use async memory operations
□ Profile with Nsight Compute
```

---

## Conclusion

This comprehensive analysis of CUDA kernel implementations in cuPHY reveals a sophisticated, highly optimized GPU programming architecture for 5G PDSCH processing. The key takeaways include:

**Architectural Excellence:**
- Consistent 256-thread blocks for optimal occupancy
- 2D grid topology for multi-TB parallelism
- Grid-stride loops for perfect coalescing
- Dynamic shared memory for flexible resource allocation

**Performance Optimizations:**
- Warp-level primitives for efficient reductions
- Cooperative group async memcpy for faster data transfers
- PTX intrinsics for bit manipulation
- Loop unrolling and function inlining
- Mixed precision with specialized atomic operations

**Synchronization Sophistication:**
- Minimized __syncthreads() overhead
- Warp-level primitives for lock-free operations
- Atomic operations with custom templates for unsupported types
- CUDA events for accurate timing and cross-stream coordination

**Memory Management:**
- Vectorized loads/stores (uint4 for 16-byte operations)
- Alignment-aware algorithms
- Constant memory for broadcast reads
- Shared memory for block-level data sharing

The cuPHY implementation demonstrates production-quality GPU programming with attention to every performance detail, from low-level PTX intrinsics to high-level algorithmic choices.

---

**END OF REPORT**
