# Fronthaul Receive Processing Chain Analysis

## NVIDIA Aerial CUDA-Accelerated RAN: O-RU to UL Channel Processing Input

**Report Scope:** End-to-end analysis of the uplink fronthaul receive path — from O-RU eCPRI connection establishment, through packet reception and IQ decompression, to the preparation of IQ sample buffers consumed by PUSCH, PUCCH, PRACH, and SRS processing chains.

**Codebase Version:** 25-3 (aerial-sdk-version: `25-3-cubb`)

> **Disclaimer:** This report is based on analysis of the open-source codebase. Some GPU kernel implementations (e.g., the order kernel `.cu` files) are compiled into libraries and not available as source. Behavioral descriptions of these kernels are inferred from their API surfaces, configuration structures, and calling code. Performance figures are architectural expectations, not benchmarked measurements.

---

## 1. Architecture Overview

The fronthaul receive chain spans three major software components in a layered architecture:

```
 O-RU (Radio Unit)
   │
   │  eCPRI over Ethernet (U-plane IQ data + C-plane control)
   │
   ▼
┌──────────────────────────────────────────────────────────────┐
│  aerial-fh-driver                                            │
│  DPDK/DOCA-based fronthaul driver                            │
│  - Ethernet RX via DPDK or DOCA GPUNetIO                     │
│  - eCPRI header parsing                                      │
│  - O-RAN C-plane/U-plane demux                               │
│  - Hardware flow steering (eAxC-based)                       │
│  - Packet delivery to GPU memory (zero-copy via GPUDirect)   │
│  Source: cuPHY-CP/aerial-fh-driver/                          │
└──────────────────────┬───────────────────────────────────────┘
                       │  Packets in GPU-accessible memory
                       │  DOCA semaphore signaling
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  cuphydriver                                                 │
│  PHY driver with UL worker threads                           │
│  - Order kernel: GPU-based packet sorting, decompression,    │
│    and PRB placement into per-channel output buffers          │
│  - Slot map management and timing validation                 │
│  - UL task pipeline orchestration                            │
│  Source: cuPHY-CP/cuphydriver/                               │
└──────────────────────┬───────────────────────────────────────┘
                       │  3D IQ tensors in GPU device memory
                       │  [subcarriers × symbols × antennas]
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  cuPHY channel processing                                    │
│  GPU-accelerated PHY receive pipelines                       │
│  - PUSCH RX: channel estimation, equalization, LDPC decode   │
│  - PUCCH RX: UCI detection (Formats 0-4)                     │
│  - PRACH RX: preamble detection, timing advance estimation   │
│  - SRS RX: channel estimation, beamforming weight computation│
│  Source: cuPHY/src/cuphy/, cuPHY/src/cuphy_channels/         │
└──────────────────────────────────────────────────────────────┘
```

**Key source files referenced throughout this report:**

| Component | File | Purpose |
|-----------|------|---------|
| FH driver API | `cuPHY-CP/aerial-fh-driver/include/aerial-fh-driver/api.hpp` | Public API: open/close, NIC, peer, flow management |
| O-RAN headers | `cuPHY-CP/aerial-fh-driver/include/aerial-fh-driver/oran.hpp` | eCPRI, O-RAN C/U-plane packet structures |
| DOCA structures | `cuPHY-CP/aerial-fh-driver/include/aerial-fh-driver/doca_structs.hpp` | GPU RX queue, semaphore, buffer constants |
| Peer management | `cuPHY-CP/aerial-fh-driver/lib/peer.cpp` | O-RU peer lifecycle, flow rules, GPU memory |
| GPU comm | `cuPHY-CP/aerial-fh-driver/lib/gpu_comm.hpp` | GPU-accelerated DL TX packet construction |
| Flow rules | `cuPHY-CP/aerial-fh-driver/lib/flow.cpp` | DPDK flow steering rules per eAxC |
| FH proxy | `cuPHY-CP/cuphydriver/src/common/fh.cpp` | cuphydriver-to-FH-driver integration layer |
| Cell init | `cuPHY-CP/cuphydriver/src/common/cell.cpp` | Cell bring-up, peer/flow registration, UL buffers |
| Order entity | `cuPHY-CP/cuphydriver/include/order_entity.hpp` | Order kernel config and launch interface |
| UL task aggr | `cuPHY-CP/cuphydriver/src/uplink/task_function_ul_aggr.cpp` | UL worker task functions |
| PUSCH driver | `cuPHY-CP/cuphydriver/src/uplink/phypusch_aggr.cpp` | PUSCH aggregation and cuPHY API binding |
| cuPHY API | `cuPHY/src/cuphy/cuphy_api.h` | PUSCH/PUCCH/PRACH/SRS receive API |
| Compression | `cuPHY-CP/compression_decompression/` | GPU BFP/uLaw/block-scaling kernels |

---

## 2. O-RU Connection Establishment

### 2.1 Fronthaul Driver Initialization

The fronthaul subsystem is initialized via `aerial_fh::open()` (`api.hpp:119`), which creates a DPDK EAL (Environment Abstraction Layer) instance and registers CUDA devices for GPU-accelerated packet processing.

**`FronthaulInfo` structure** (`api.hpp:77-92`) controls initialization:

```
FronthaulInfo:
  dpdk_thread           → CPU core for DPDK polling thread
  cuda_device_ids       → GPU devices for DOCA GPUNetIO packet RX
  cuda_device_ids_for_compute → GPU devices for PHY compute
  rivermax              → Rivermax RX mode (alternative to DPDK RX)
  cpu_rx_only           → CPU-only memory for RX (UE mode)
  enable_gpu_comm_via_cpu → GPU comm routed through CPU (no P2P)
```

The `FhProxy` class in cuphydriver (`fh.cpp:56-88`) wraps the FH driver API and connects it to the PHY driver context:

```cpp
// fh.cpp:85
aerial_fh::open(&fh_info, &fhi);
```

### 2.2 NIC Registration

Each physical NIC port is registered via `aerial_fh::add_nic()` (`api.hpp:167`). The `NicInfo` structure specifies:

- **PCIe bus address** — identifies the Mellanox/NVIDIA ConnectX NIC
- **MTU** — determines maximum eCPRI packet size (typically 1500 bytes)
- **Queue counts** — separate TX and RX queue pools
- **CUDA device** — which GPU receives packets via DOCA GPUNetIO
- **Queue sizes** — depth of TX/RX descriptor rings

The NIC is initialized through DPDK with `rte_eth_dev_configure()` and the appropriate number of TX/RX queues are created. For GPU-accelerated RX, DOCA objects (`doca_eth_rxq`, `doca_gpu_eth_rxq`) are created to enable direct NIC-to-GPU DMA.

```cpp
// FhProxy::registerNic() in fh.cpp:190-231
aerial_fh::NicInfo ninfo{
    cfg.nic_bus_addr, cfg.nic_mtu, false,
    cfg.cpu_mbuf_num, 0, 0, 0, cfg.tx_req_num,
    txq_cpu, txq_gpu, cfg.rxq_count,
    cfg.txq_size, cfg.rxq_size, gpu_id, false
};
aerial_fh::add_nic(fhi, &ninfo, &nic);
```

### 2.3 Peer (O-RU) Registration

Each O-RU is modeled as a **Peer** in the fronthaul driver. A peer is identified by:

- **Source/destination MAC addresses** — Ethernet L2 identity
- **VLAN TCI** — 802.1Q tag (PCP + VID) for traffic isolation
- **eAxC ID lists** — separate lists for UL, SRS, and DL antenna-carrier streams

Peer creation (`peer.cpp:38-81`) performs:

1. **PRB size calculation** based on compression method and IQ bit width
2. **Source MAC address adjustment** (if not explicitly provided)
3. **NIC resource allocation** — TX/RX queues assigned from the NIC's queue pool
4. **DOCA GPU semaphore creation** — for GPU-based RX synchronization
5. **Flow rule creation** — hardware-level packet steering per eAxC ID
6. **GPU slot info allocation** — per-slot metadata structures in GPU memory
7. **C-plane section cache creation** — for matching U-plane to C-plane

```cpp
// peer.cpp constructor:
Peer::Peer(Nic* nic, PeerInfo const* info,
    std::vector<uint16_t>& eAxC_list_ul,
    std::vector<uint16_t>& eAxC_list_srs,
    std::vector<uint16_t>& eAxC_list_dl)
{
    prb_size_upl_ = get_prb_size(info->ud_comp_info.iq_sample_size,
                                  info->ud_comp_info.method);
    prbs_per_pkt_upl_ = (nic->get_mtu() - ORAN_IQ_HDR_SZ) / prb_size_upl_;
    ...
    request_nic_resources();
    doca_gpu_sem_create();
    create_rx_rules(eAxC_list_ul, eAxC_list_srs, eAxC_list_dl);
    gpu_comm_create_up_slot_list();
    create_cplane_sections_cache();
}
```

### 2.4 Cell Bring-Up and Flow Registration

At the cuphydriver level, each cell is created via the `Cell` constructor (`cell.cpp:29-250`), which orchestrates fronthaul registration:

```
Cell bring-up sequence:
  1. Extract M-plane config (MAC, VLAN, eAxC IDs, timing params)
  2. Build UL eAxC list from PUSCH + PUCCH + PRACH channels
  3. Register peer with FH driver (per NIC)
  4. Register flows for each eAxC ID and channel type
  5. Allocate UL input buffers (GPU device memory):
     - Section Type 1 buffers (PUSCH/PUCCH): 273 PRB × 12 RE × 14 sym × 4 bytes/RE × N_ant
     - Section Type 2 buffers (SRS): 273 PRB × 12 RE × 6 sym × 4 bytes/RE × N_ant_srs
     - Section Type 3 buffers (PRACH): 24 PRB × 12 RE × 12 rep × 4 bytes/RE × N_ant
  6. Initialize GPU semaphore tracking structures
  7. Configure beamforming weight buffers (NO_CHAINING / CPU / GPU chaining)
```

### 2.5 Hardware Flow Steering

For each eAxC ID, a DPDK flow rule (`flow.cpp`) is programmed into the NIC hardware to steer incoming packets to the correct RX queue. The flow match criteria are:

```
Flow match fields:
  - Destination MAC address (O-DU MAC)
  - VLAN ID
  - eCPRI RTC ID (maps to eAxC ID in O-RAN)
```

This hardware-level classification means different antenna ports and channel types (PUSCH vs. SRS) are steered to separate RX queues, enabling parallel processing paths.

---

## 3. eCPRI Packet Reception

### 3.1 Packet Structure

Each uplink fronthaul packet has the following wire format:

```
┌──────────────────────────────────────────────────────────────────┐
│ Ethernet Header (14B) │ VLAN Tag (4B) │ eCPRI Header (8B) │    │
│  dst_mac  src_mac type│ tci  proto    │ ver msg len rtcid seq│  │
├────────────────────────┴───────────────┴──────────────────────┤  │
│ O-RAN Radio Application Header (4B)                           │  │
│  dataDir│payloadVer│filterIdx│frameId│subframeId│slotId│symId │  │
├──────────────────────────────────────────────────────────────────┤
│ O-RAN Section Header (variable)                                 │
│  sectionId│rb│symInc│startPrbc│numPrbc│udCompHdr│reserved      │
├──────────────────────────────────────────────────────────────────┤
│ Compressed IQ Payload                                           │
│  [BFP exponent + compressed I/Q pairs] × numPrbc × 12 RE      │
└──────────────────────────────────────────────────────────────────┘
```

The O-RAN header structures are defined in `oran.hpp`:

- **`oran_ecpri_hdr`** (line 221): eCPRI transport — version, message type (`0x00` = IQ data, `0x02` = RTC), payload length, RTC ID (carries eAxC), sequence ID
- **`oran_cmsg_radio_app_hdr`**: Data direction, payload version, filter index, frame/subframe/slot/symbol IDs
- **Section Type 1** (`oran_cmsg_sect1`): startPrbc, numPrbc, reMask, numSymbol, beamId — used for PUSCH/PUCCH
- **Section Type 3** (`oran_cmsg_sect3`): Adds `freqOffset` for PRACH with frequency offset
- **Section Type 5** (`oran_cmsg_sect5`): Used for UE-specific beamforming

### 3.2 Two Receive Modes

The FH driver supports two primary RX paths:

#### Mode 1: DOCA GPUNetIO Direct (Default for DU)

Packets are received directly into GPU memory via DOCA GPUNetIO, bypassing CPU entirely:

```
NIC → PCIe DMA → GPU Memory (via DOCA doca_gpu_eth_rxq)
                          │
                GPU Order Kernel reads packets from GPU RX ring
```

Key DOCA structures (`doca_structs.hpp:114-129`):

```c
typedef struct doca_rx_items {
    struct doca_gpu *gpu_dev;               // GPU device
    struct doca_dev *ddev;                  // Network DOCA device
    struct doca_eth_rxq *eth_rxq_cpu;       // CPU handle
    struct doca_gpu_eth_rxq *eth_rxq_gpu;   // GPU handle (for kernels)
    struct doca_mmap *pkt_buff_mmap;        // Memory map for packet buffer
    void *gpu_pkt_addr;                     // GPU memory for packets
    struct doca_gpu_semaphore *sem_cpu;      // Semaphore (CPU side)
    struct doca_gpu_semaphore_gpu *sem_gpu;  // Semaphore (GPU side)
} doca_rx_items_t;
```

#### Mode 2: CPU RX (UE Mode or fallback)

When GPU-direct is not available (e.g., no P2P support, UE mode):

```
NIC → DPDK rte_eth_rx_burst() → CPU mbufs → cudaMemcpy → GPU Memory
```

Selected via `cpu_rx_only` flag or `enable_gpu_comm_via_cpu` in `FronthaulInfo`.

### 3.3 Timing Constraints

The FH driver timestamps incoming packets using PTP-synchronized hardware timestamps. Key constants from `doca_structs.hpp`:

```c
SYMBOL_DURATION_NS = 35714;           // ~35.7 μs per OFDM symbol
CK_ORDER_PKTS_BUFFERING_NS = 40000;   // 40 μs buffering window
ORDER_KERNEL_RECV_TIMEOUT_MS = 4;      // 4 ms receive timeout
ORDER_KERNEL_MAX_PKTS_PER_OFDM_SYM = 100;  // Safety limit per symbol
```

The order kernel validates each packet's arrival time against the O-RAN timing window defined by `Ta4_min` and `Ta4_max` per cell, counting packets as early, on-time, or late.

---

## 4. The Order Kernel: Packet Sorting, Decompression, and PRB Placement

### 4.1 Purpose and Architecture

The **order kernel** is the central GPU kernel in the UL receive chain. It is a persistent CUDA kernel that runs on the GPU, directly consuming packets from DOCA GPUNetIO RX queues and producing decompressed, sorted IQ sample buffers ready for cuPHY channel processing.

The kernel performs all of the following in a single GPU launch:

1. **Packet reception** from DOCA GPU Ethernet RX queues
2. **eCPRI/O-RAN header parsing** on the GPU
3. **Timing validation** against Ta4 window
4. **IQ decompression** (BFP, block scaling, u-law, or modulation compression)
5. **PRB placement** into per-channel output buffers at the correct [antenna, symbol, subcarrier] location
6. **Signaling completion** via GDR (GPU Direct RDMA) flags

### 4.2 Configuration

The order kernel is configured via `orderKernelConfigParams` (`order_entity.hpp:46-100`), which carries per-cell parameters:

```
Per-cell configuration:
  rxq_info_gpu[]           → DOCA GPU RX queue handles
  sem_gpu[]                → Semaphores for RX synchronization
  comp_meth[]              → Compression method (BFP=1, BlockScaling=2, uLaw=3, ModComp=4)
  bit_width[]              → IQ sample bit width (e.g., 9 for 9-bit BFP)
  beta[]                   → BFP beta scaling parameter
  slot_start[]             → Expected slot start time (ns)
  ta4_min_ns[] / ta4_max_ns[] → Timing acceptance window

  PUSCH/PUCCH output:
    pusch_buffer[]         → GPU device memory pointer
    pusch_prb_x_slot[]     → Total PRBs allocated
    pusch_prb_stride[]     → Memory stride between PRBs (bytes)
    pusch_eAxC_num[]       → Number of antenna ports

  PRACH output:
    prach_buffer_0..3[]    → Up to 4 PRACH occasions per slot
    prach_prb_x_slot[]     → PRACH PRB count
    prach_prb_stride[]     → PRACH buffer stride

  SRS output:
    srs_buffer[]           → SRS output buffer
    srs_prb_x_slot[]       → SRS PRB count
    srs_prb_stride[]       → SRS buffer stride

  GDR synchronization:
    start_cuphy_d[]        → Flag set by order kernel when ordering complete
    order_kernel_exit_cond_d[] → Exit condition (0=normal, >0=error/timeout)
    on_time_rx_packets[]   → Counter of packets within Ta4 window
    early_rx_packets[]     → Counter of packets arriving before Ta4 min
    late_rx_packets[]      → Counter of packets arriving after Ta4 max
```

### 4.3 Launch and Execution

The order kernel is launched by `OrderEntity::runOrder()` from UL task function 1 (`task_function_ul_aggr.cpp:1879`). It runs only when at least one of PUSCH, PUCCH, PRACH, or SRS is scheduled in the slot:

```cpp
// task_function_ul_aggr.cpp:1769
if(pusch||pucch||prach||srs)
{
    oentity->runOrder(slot_oran_ind,
        numPrbPuschPucch, buf_st_1,
        buf_pcap_capture, buf_pcap_capture_ts,
        numPrbPrach,
        buf_st_3_o0, buf_st_3_o1, buf_st_3_o2, buf_st_3_o3,
        startSectionIdPrach, ...
        numPrbSrs, buf_st_2,
        slot_start, ta4_min_ns, ta4_max_ns,
        ta4_min_ns_srs, ta4_max_ns_srs,
        num_order_cells, srsMask, srs_start_symbol,
        nonSrsUlMask, ...);
}
```

The kernel operates on a cell mask, processing only cells that have active UL channels:
- `nonSrsUlMask`: Bitmask of cells with PUSCH/PUCCH/PRACH data
- `srsMask`: Bitmask of cells with SRS data

### 4.4 Packet-to-Buffer Mapping

The order kernel maps each incoming eCPRI packet to the correct location in the output buffer using information from the O-RAN headers:

```
Packet header fields → Buffer location:
  eAxC ID (RTC ID)     → antenna port index
  Symbol ID             → symbol dimension
  Section ID / startPrbc → PRB offset within symbol
  numPrbc               → Number of contiguous PRBs in this packet
```

The C-plane sections cache (created during peer registration) provides the expected section layout, enabling the kernel to validate U-plane data against the C-plane allocation.

### 4.5 Output Buffer Layout

The order kernel writes decompressed IQ samples into three types of output buffers:

**Section Type 1 — PUSCH/PUCCH buffer:**
```
Size per cell: ORAN_MAX_PRB(273) × 12 RE × 14 symbols × sizeof(uint32_t) × N_antenna_ports
Layout: [antenna_port][symbol][subcarrier] — complex int16 pairs packed as uint32
```

**Section Type 2 — SRS buffer:**
```
Size per cell: ORAN_MAX_PRB(273) × 12 RE × 6 symbols × sizeof(uint32_t) × N_srs_antenna_ports
Layout: [antenna_port][symbol][subcarrier]
```

**Section Type 3 — PRACH buffer (per occasion):**
```
Size per cell: 24 PRB × 12 RE × 12 repetitions × sizeof(uint32_t) × N_antenna_ports
Up to 4 occasions per slot (prach_buffer_0 through prach_buffer_3)
```

---

## 5. IQ Decompression

### 5.1 Supported Compression Methods

The fronthaul driver supports the following O-RAN compression methods (`api.hpp:239-249`):

| Method | Enum Value | Description |
|--------|-----------|-------------|
| `NO_COMPRESSION` | `0b0000` | Raw 16-bit I/Q samples |
| `BLOCK_FLOATING_POINT` | `0b0001` | BFP: 1-byte exponent + N-bit mantissas per PRB |
| `BLOCK_SCALING` | `0b0010` | Block scaling with shared scale factor |
| `U_LAW` | `0b0011` | μ-law companding |
| `MODULATION_COMPRESSION` | `0b0100` | Modulation-aware compression |
| `BFP_SELECTIVE_RE_SENDING` | `0b0101` | BFP with selective RE |
| `MOD_COMPR_SELECTIVE_RE_SENDING` | `0b0110` | ModComp with selective RE |

### 5.2 Decompression in the Order Kernel

Decompression happens **inline** within the order kernel as packets are processed. For each PRB in a U-plane packet:

1. Read the `udCompHdr` from the section header (compression method + bit width)
2. For BFP: read the per-PRB exponent byte, then unpack N-bit mantissa pairs
3. Scale to int16 complex pairs and write to the output buffer at the correct [ant, sym, sc] location

The GPU kernel templates for compression/decompression are in `cuPHY-CP/compression_decompression/`:

**Block Floating Point** (`gpu_blockFP.h`):
- Per-PRB: 1 exponent byte + 24 × (compbits/8) mantissa bytes
- Decompression: `mantissa << (exponent - compbits + 1)` to recover int16
- GPU implementation: 64 PRBs per CTA, 256 threads

**Block Scaling** (`gpu_blockScaling.h`):
- Similar to BFP but with a linear scale factor instead of exponent

**μ-law** (`gpu_uLaw.h`):
- ITU-T G.711 μ-law companding applied per I/Q sample

**Modulation Compression** (`api.hpp:245`):
- Special handling: `prb_size_upl_` and `prbs_per_pkt_upl_` set to 0
- Partial slot info tracking per message (`partial_up_slot_info_`)
- Modulation-specific parameters carried per packet

### 5.3 Data Type After Decompression

After decompression, IQ samples are stored as **complex int16 pairs** (packed as `uint32_t`):

```
Each uint32_t = [Q_sample(int16) | I_sample(int16)]
Each PRB = 12 resource elements = 12 × uint32_t = 48 bytes
```

This format is the native input format expected by the cuPHY channel processing kernels.

---

## 6. UL Slot Processing Pipeline

### 6.1 Slot Map and Task Scheduling

The UL processing pipeline is driven by `SlotMapUl` objects that encapsulate all state for a single uplink slot across multiple cells. The pipeline executes as a sequence of tasks on dedicated UL worker threads:

```
UL Task Pipeline:
  Task 0: C-plane processing (UL_TTI.request from L2)
  Task 1: Order kernel launch (packet RX + decompress + sort)
  Task 2: Order kernel completion wait + cuPHY channel launch
          → PUSCH, PUCCH, PRACH, SRS processing
  Task 3+: Completion callbacks and L2 indication
```

### 6.2 Task 1: Order Kernel Launch (task_function_ul_aggr.cpp)

Task 1 prepares and launches the order kernel. Key steps:

1. **Collect per-cell parameters** (lines 1784-1871):
   - Get UL input buffer pointers (Section Type 1, 2, 3)
   - Count PRBs per channel from C-plane sections
   - Set Ta4 timing windows per cell
   - Build cell masks for SRS vs non-SRS processing

2. **Count PRBs from FH sections** (line 1840):
   ```cpp
   numPrbPuschPucch[i] = fhproxy->countPuschPucchPrbs(
       *(slot_map->aggr_slot_info[i]),
       cell_ptr->geteAxCNumPusch(),
       cell_ptr->geteAxCNumPucch(), ...);
   numPrbPrach[i] = fhproxy->countPrachPrbs(...);
   numPrbSrs[i] = fhproxy->countSrsPrbs(...);
   ```

3. **Launch order kernel** (line 1879): `oentity->runOrder(...)` — asynchronous GPU launch

4. **Unlock next task** (line 1912): Signals task 2 to begin

### 6.3 Task 2: Order Completion and Channel Processing Launch

Task 2 waits for the order kernel to complete, then launches cuPHY channel processing:

1. **Wait for ULC (UL C-plane) tasks** (line 1978)
2. **Wait for order kernel completion** — polls GDR `start_cuphy_d` flag
3. **Check order kernel exit condition** — `order_kernel_exit_cond_d` for timeouts/errors
4. **Bind IQ buffers to cuPHY tensors** and launch:
   - `cuphySetupPuschRx()` + `cuphyRunPuschRx()`
   - `cuphySetupPucchRx()` + `cuphyRunPucchRx()`
   - `cuphySetupPrachRx()` + `cuphyRunPrachRx()`
   - `cuphySetupSrsChEst()` + `cuphyRunSrsChEst()`

### 6.4 UL Worker Thread Architecture

UL workers are created during context initialization (`context.cpp:283-295`):

```cpp
for(int ulc = 0; ulc < workers_ul_cores.size(); ulc++) {
    Worker w(pdh, wid, WORKER_UL, name,
             workers_ul_cores[ulc],      // CPU core affinity
             ctx_cfg.workers_sched_priority,  // SCHED_FIFO priority
             ...);
}
```

Workers run with:
- **CPU core affinity** — pinned to specific cores
- **SCHED_FIFO real-time priority** — configured via `ENABLE_SCHED_FIFO_ALL_RT`
- **PMU metrics** — optional performance counter monitoring

GPU SM allocation for UL processing is controlled via NVIDIA MPS:

```yaml
# From YAML configuration:
mps_sm_pusch: 50        # SMs for PUSCH
mps_sm_pucch: 10        # SMs for PUCCH
mps_sm_prach: 4         # SMs for PRACH
mps_sm_srs: 20          # SMs for SRS
mps_sm_ul_order: 8      # SMs for order kernel
```

---

## 7. Handoff to cuPHY Channel Processing

### 7.1 IQ Buffer Format for cuPHY

cuPHY receives UL IQ data through `cuphyTensorPrm_t` structures (`cuphy.h:1242-1246`):

```c
typedef struct _cuphyTensorPrm {
    cuphyTensorDescriptor_t desc;  // Layout descriptor
    void*                   pAddr; // GPU device memory pointer
} cuphyTensorPrm_t;
```

The tensor descriptor defines a 3D layout:

```
Dimension 0 (fastest): Subcarriers = ORAN_MAX_PRB × 12 = up to 3276
Dimension 1 (medium):  OFDM Symbols = 14 per slot
Dimension 2 (slowest): Antenna Ports = up to 16 (or 64 for SRS)
Data type: CUPHY_C_16F (complex float16, __half2)
```

### 7.2 PUSCH Input Setup

`PhyPuschAggr` (`phypusch_aggr.cpp:230-244`) creates per-cell tensor descriptors:

```cpp
pusch_data_rx_desc[idx] = {
    CUPHY_C_16F,
    static_cast<int>(ORAN_MAX_PRB * CUPHY_N_TONES_PER_PRB),  // 3276 subcarriers
    static_cast<int>(OFDM_SYMBOLS_PER_SLOT),                  // 14 symbols
    static_cast<int>(MAX_AP_PER_SLOT),                         // antenna ports
    cuphy::tensor_flags::align_tight
};
DataIn.pTDataRx[idx].desc = pusch_data_rx_desc[idx].handle();
DataIn.pTDataRx[idx].pAddr = nullptr;  // Set at runtime to UL input buffer
```

At slot processing time, the `pAddr` field is pointed to the `ULInputBuffer::getBufD()` GPU device address that was populated by the order kernel.

### 7.3 PUSCH Receiver Pipeline (cuphy_api.h:463-617)

```c
typedef struct _cuphyPuschDataIn {
    cuphyTensorPrm_t* pTDataRx;    // Per-cell IQ tensor array
    cuphyTensorPrm_t* pTNoisePwr;  // Optional noise power metrics
} cuphyPuschDataIn_t;
```

The PUSCH receiver pipeline then performs:
1. DMRS-based channel estimation (MMSE, RKHS, or LS algorithms)
2. CFO estimation and correction
3. Timing advance estimation
4. Channel equalization
5. Soft demapping
6. LDPC decoding
7. CRC checking and TB assembly

### 7.4 PUCCH Input (cuphy_api.h:987-991)

```c
typedef struct _cuphyPucchDataIn {
    cuphyTensorPrm_t* pTDataRx;  // Same 3D tensor format as PUSCH
} cuphyPucchDataIn_t;
```

PUCCH shares the Section Type 1 buffer with PUSCH. The cuPHY PUCCH receiver extracts the relevant symbols and PRBs based on the PUCCH format and resource allocation.

### 7.5 PRACH Input (cuphy_api.h:3394-3397)

```c
typedef struct _cuphyPrachDataIn {
    cuphyTensorPrm_t* pTDataRx;  // Per-occasion buffer array
} cuphyPrachDataIn_t;
```

PRACH uses separate Section Type 3 buffers per occasion (up to 4 per slot). Each occasion buffer contains:
- Up to 24 PRACH PRBs
- 12 repetitions
- N antenna ports

### 7.6 SRS Input (cuphy_api.h:1316-1320)

```c
typedef struct _cuphySrsDataIn {
    cuphyTensorPrm_t* pTDataRx;  // SRS symbols per cell
} cuphySrsDataIn_t;
```

SRS uses Section Type 2 buffers with up to 64 antenna ports (for massive MIMO) and up to 6 SRS symbols per slot.

---

## 8. Timing and Synchronization Model

### 8.1 O-RAN Timing Parameters

Each cell is configured with O-RAN-defined timing parameters (`cell_mplane_info` in `cuphydriver_api.hpp:117-195`):

| Parameter | Description |
|-----------|-------------|
| `ta4_min_ns` | Minimum delay from UL slot boundary for U-plane arrival |
| `ta4_max_ns` | Maximum delay from UL slot boundary for U-plane arrival |
| `ta4_min_ns_srs` | SRS-specific Ta4 minimum |
| `ta4_max_ns_srs` | SRS-specific Ta4 maximum |
| `t1a_max_cp_ul_ns` | Max advance time for UL C-plane transmission |
| `t1a_min_cp_ul_ns` | Min advance time for UL C-plane transmission |
| `ul_u_plane_tx_offset_ns` | UL U-plane TX offset (for RU emulator) |

### 8.2 Order Kernel Timing Validation

The order kernel uses `slot_start[]` (computed from L2 slot tick + Ta offset) to determine the expected arrival window:

```
Expected UL arrival window:
  earliest = slot_start[cell] + ta4_min_ns[cell]
  latest   = slot_start[cell] + ta4_max_ns[cell]
```

Packets are classified as:
- **Early**: arrival < `slot_start + ta4_min_ns`
- **On-time**: `ta4_min` <= arrival <= `ta4_max`
- **Late**: arrival > `slot_start + ta4_max_ns`

The packet counts are reported via GDR flags for health monitoring:

```c
// order_entity.hpp:92-97
uint32_t* early_rx_packets[UL_MAX_CELLS_PER_SLOT];
uint32_t* on_time_rx_packets[UL_MAX_CELLS_PER_SLOT];
uint32_t* late_rx_packets[UL_MAX_CELLS_PER_SLOT];
```

### 8.3 O-RU Health Monitoring

The system tracks consecutive unhealthy slots per cell. If a cell is marked unhealthy (based on missing or late packets), the order kernel processing may be skipped:

```cpp
// task_function_ul_aggr.cpp:1861
if(pdctx->ru_health_check_enabled() && !cell_ptr->isHealthy()) {
    cell_ptr->num_consecutive_unhealthy_slots++;
    // Order kernel processing skipped for this cell
}
```

---

## 9. Multi-Cell and Massive MIMO Considerations

### 9.1 Multi-Cell Processing

The order kernel processes multiple cells in a single launch. Cell-level parallelism is achieved through:

- **Per-cell DOCA RX queues** — separate hardware queues per cell
- **Cell mask bitmaps** — `nonSrsUlMask` and `srsMask` control which cells are active
- **Cell grouping** — cells can be grouped for inter-cell batching in cuPHY

The system supports up to:
- 16 cells (default), 32 cells (`ENABLE_20C`), or 64 cells (`ENABLE_64C`)
- Up to `UL_MAX_CELLS_PER_SLOT` cells processed per slot map

### 9.2 Massive MIMO (32T32R / 64T64R)

For massive MIMO configurations:

- **PUSCH/PUCCH**: Up to 16 RX antenna ports per cell (`MAX_RX_ANT_PUSCH_PUCCH_PRACH_64T64R = 16`)
- **SRS**: Up to 64 RX antenna ports per cell (`MAX_RX_ANT_SRS_64T64R = 64`)
- **Separate SRS RX queues** — SRS streams use dedicated DOCA RX queues due to the higher antenna count
- **BFW (Beamforming Weight) C-plane**: Supports chaining modes (NO_CHAINING, CPU_CHAINING, GPU_CHAINING) for transmitting large beamforming weight matrices

### 9.3 Buffer Sizes

From `doca_structs.hpp:80-82`:

```c
UL_ST1_AP_BUF_SIZE = 273 × 12 × 14 × 4 = 183,456 bytes per antenna port  (PUSCH/PUCCH)
UL_ST2_AP_BUF_SIZE = 273 × 12 × 6  × 4 =  78,624 bytes per antenna port  (SRS)
UL_ST3_AP_BUF_SIZE = 24  × 12 × 12 × 4 =  13,824 bytes per antenna port  (PRACH)
```

Total per-cell buffer for 16 PUSCH antennas: ~2.9 MB GPU memory per slot.

---

## 10. Error Handling and Recovery

### 10.1 Order Kernel Exit Codes

The order kernel can exit with several conditions (`doca_structs.hpp:93-106`):

| Exit Code | Meaning |
|-----------|---------|
| `ORDER_KERNEL_RUNNING (0)` | Still running |
| `ORDER_KERNEL_EXIT_PRB (1)` | Normal completion (all PRBs received) |
| `ORDER_KERNEL_EXIT_TIMEOUT_RX_PKT (3)` | Timeout waiting for packets |
| `ORDER_KERNEL_EXIT_TIMEOUT_NO_PKT (4)` | Timeout, no packets at all |
| `ORDER_KERNEL_EXIT_ERROR1..7 (5-11)` | Various error conditions |

### 10.2 Timeout Handling

Timeouts are configured per context (`context.cpp`):

```
ul_order_timeout_cpu_ns  → CPU-side polling timeout for order completion
ul_order_timeout_gpu_ns  → GPU kernel internal timeout
ul_order_timeout_gpu_srs_ns → SRS-specific GPU timeout
```

If the order kernel times out, the UL task pipeline logs the error, increments the unhealthy slot counter, and continues to the next slot. There is currently no automatic pipeline recovery from CUDA/FH errors — a fatal failure causes process exit.

### 10.3 PCAP Capture

For debugging, the order kernel can capture raw packets into a GPU buffer (`pcap_buffer[]`) with timestamps (`pcap_buffer_ts[]`). This is enabled via `ul_pcap_capture_enable` in the YAML configuration.

---

## 11. Summary: Complete Data Flow

```
1. SYSTEM INIT
   aerial_fh::open() → DPDK EAL + DOCA GPU init
   aerial_fh::add_nic() → NIC port setup, RX/TX queues
   aerial_fh::add_peer() → O-RU registration, flow rules, GPU memory

2. CELL BRING-UP
   Cell::Cell() → Peer + flow registration
                → UL input buffers (GPU device memory)
                → Semaphore and GDR flag allocation

3. PER-SLOT UL PROCESSING
   L2 → UL_TTI.request (FAPI) → C-plane sections parsed
                                → Expected PRB counts known

4. ORDER KERNEL LAUNCH (GPU)
   ┌─────────────────────────────────────────────────────┐
   │  For each incoming eCPRI packet:                     │
   │    a. Read packet from DOCA GPU RX ring              │
   │    b. Parse eCPRI header → eAxC ID, seq ID           │
   │    c. Parse O-RAN header → frame/slot/symbol/section │
   │    d. Validate timing against Ta4 window             │
   │    e. Decompress IQ payload (BFP/BS/uLaw/ModComp)   │
   │    f. Write int16 IQ pairs to output buffer at       │
   │       [antenna][symbol][PRB×12] location             │
   │  Signal completion via GDR flag                      │
   └─────────────────────────────────────────────────────┘

5. HANDOFF TO cuPHY
   UL Task 2 detects order completion
   → Binds UL input buffer to cuphyTensorPrm_t
   → cuphySetupPuschRx() + cuphyRunPuschRx()
   → cuphySetupPucchRx() + cuphyRunPucchRx()
   → cuphySetupPrachRx() + cuphyRunPrachRx()
   → cuphySetupSrsChEst() + cuphyRunSrsChEst()

6. CHANNEL PROCESSING (GPU)
   PUSCH: ChEst → Equalization → Soft Demod → LDPC Decode → CRC
   PUCCH: ChEst → UCI Detection (Format-specific)
   PRACH: Preamble Detection → TA Estimation
   SRS:   ChEst → BFW Computation
```

---

## Appendix A: Key Constants

| Constant | Value | Source |
|----------|-------|--------|
| `ORAN_MAX_PRB` | 273 | `doca_structs.hpp:68` |
| `ORAN_MAX_SYMBOLS` | 14 | `doca_structs.hpp:69` |
| `ORAN_RE` (REs per PRB) | 12 | `doca_structs.hpp:67` |
| `ORAN_PRACH_PRB` | 24 | `doca_structs.hpp:71` |
| `ORAN_MAX_SRS_SYMBOLS` | 6 | `doca_structs.hpp:70` |
| `MAX_RX_ANT_4T4R` | 4 | `doca_structs.hpp:74` |
| `MAX_RX_ANT_PUSCH_PUCCH_PRACH_64T64R` | 16 | `doca_structs.hpp:75` |
| `MAX_RX_ANT_SRS_64T64R` | 64 | `doca_structs.hpp:76` |
| `SYMBOL_DURATION_NS` | 35,714 | `doca_structs.hpp:33` |
| `DEFAULT_PRB_STRIDE` | 48 bytes | `doca_structs.hpp:77` |
| `UL_MAX_CELLS_PER_SLOT` | 16/32/64 | Config-dependent |
| `PRACH_MAX_OCCASIONS` | 4 | `constant.hpp` |

## Appendix B: YAML Configuration Parameters (UL-Relevant)

```yaml
# From cuPHY-CP/cuphycontroller/config/*.yaml
cuphydriver_config:
  workers_ul: [4, 5]              # CPU cores for UL worker threads
  mps_sm_pusch: 50                # GPU SMs for PUSCH
  mps_sm_pucch: 10                # GPU SMs for PUCCH
  mps_sm_prach: 4                 # GPU SMs for PRACH
  mps_sm_srs: 20                  # GPU SMs for SRS
  mps_sm_ul_order: 8              # GPU SMs for order kernel
  ul_order_timeout_cpu_ns: 4000000  # CPU timeout for order kernel
  ul_order_timeout_gpu_ns: 4000000  # GPU timeout for order kernel
  ul_order_kernel_mode: 0         # 0=Ping-Pong, 1=Dual CTA
  ul_input_buffer_per_cell: 4     # Circular buffer depth
  ul_input_buffer_per_cell_srs: 4 # SRS circular buffer depth
  enable_ul_cuphy_graphs: 0       # CUDA graph mode for UL
  ru_health_check_enable: 1       # O-RU health monitoring
  ul_pcap_capture_enable: 0       # Debug packet capture

# Per-cell timing (from M-plane config)
cells:
  - ta4_min_ns: 25000
    ta4_max_ns: 300000
    ta4_min_ns_srs: 25000
    ta4_max_ns_srs: 300000
    ul_comp_meth: 1               # BFP compression
    ul_bit_width: 9               # 9-bit mantissa
```
