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

#ifndef PHYDRIVER_CTX_H
#define PHYDRIVER_CTX_H

#include <memory>
#include <algorithm>
#include <unordered_map>
#include <condition_variable>
#include "constant.hpp"
#include "cuphydriver_api.hpp"
#include "worker.hpp"
#include "task.hpp"
#include "cell.hpp"
#include "gpudevice.hpp"
#include "phychannel.hpp"
#include "phypusch_aggr.hpp"
#include "phypucch_aggr.hpp"
#include "phyprach_aggr.hpp"
#include "phypdsch_aggr.hpp"
#include "phypdcch_aggr.hpp"
#include "phypbch_aggr.hpp"
#include "phycsirs_aggr.hpp"
#include "physrs_aggr.hpp"
#include "dl_validation_params.hpp"

#include "fh.hpp"
#include "slot_map_ul.hpp"
#include "slot_map_dl.hpp"
#include "dlbuffer.hpp"
#include "ulbuffer.hpp"
#include "mps.hpp"
#include "order_entity.hpp"
#include "memfoot.hpp"
#include "harq_pool.hpp"
#include "wavgcfo_pool.hpp"
#include <slot_command/slot_command.hpp>
#include "metrics.hpp"
#include "cv_memory_bank_srs_chest.hpp"
#include "ul_pcap_capture_thread.hpp"
#include "perf_metrics/percentile_tracker.hpp"

// Current max. limit of green contexts used in cuphydriver. 7 green contexts are currently used.
// Affects size of various vectors/arrays in PhyDriverCtx. Increase if you plan to add more green contexts.
#define CURRENT_MAX_GREEN_CTXS 10

#define CUPHYDRIVER_PDSCH_USE_BATCHED_COPY 1
#define BATCHED_COPY_THREAD_COPY_GRANULARITY 20 // Used only when CUPHYDRIVER_PDSCH_USE_BATCHED_COPY is set; can set to something lower than cells scheduled to explore perf. Could use DL_MAX_CELLS_PER_SLOT instead too.

void default_err_hndl(const char*);
void default_inf_hndl(const char*);

struct fh_memreg_buf {
    MemRegHandle        memreg;
    MemRegInfo memreg_info;
    void* buffer_ptr;
    size_t buffer_size;
};

typedef struct aggr_obj_error_info{
    bool prevSlotNonAvail;
    uint32_t nonAvailCount;
    uint32_t availCount;
    uint32_t l1RecoverySlots;
}aggr_obj_error_info_t;

typedef enum aggr_stream_pusch_index
{
    PHASE1_SPLIT_STREAM1 = 0,
    PHASE1_SPLIT_STREAM2 = 1,
    PHASE2_SPLIT_STREAM1 = 2,
    PHASE2_SPLIT_STREAM2 = 3
} aggr_stream_pusch_index;

typedef struct ok_tb_config_info
{
    uint32_t num_rx_packets[MAX_UL_SLOTS_OK_TB];
    uint32_t num_pusch_prbs[MAX_UL_SLOTS_OK_TB];
    uint32_t num_prach_prbs[MAX_UL_SLOTS_OK_TB];
    uint8_t		frameId[MAX_UL_SLOTS_OK_TB];
    uint8_t		subframeId[MAX_UL_SLOTS_OK_TB];
    uint8_t		slotId[MAX_UL_SLOTS_OK_TB];
    uint16_t    pusch_eAxC_map[MAX_AP_PER_SLOT];
    uint32_t    pusch_eAxC_num;
    uint16_t    prach_eAxC_map[MAX_AP_PER_SLOT];
    uint32_t    prach_eAxC_num;
    uint32_t    pusch_prb_symbol_map[MAX_UL_SLOTS_OK_TB][ORAN_PUSCH_SYMBOLS_X_SLOT];
    uint32_t    num_order_cells_sym_mask[MAX_UL_SLOTS_OK_TB][ORAN_PUSCH_SYMBOLS_X_SLOT];        
    int cell_id;
}ok_tb_config_info_t;

typedef struct ok_tb_config_srs_info
{
    uint32_t num_rx_packets[MAX_UL_SLOTS_OK_TB];
    uint32_t num_srs_prbs[MAX_UL_SLOTS_OK_TB];
    uint8_t		frameId[MAX_UL_SLOTS_OK_TB];
    uint8_t		subframeId[MAX_UL_SLOTS_OK_TB];
    uint8_t		slotId[MAX_UL_SLOTS_OK_TB];
    uint16_t    srs_eAxC_map[MAX_AP_PER_SLOT_SRS];
    uint32_t    srs_eAxC_num;
    int cell_id;
}ok_tb_config_srs_info_t;

typedef struct fapi_srs_stats_info
{
    uint32_t slot_on_time_fapi_packets_srs;
    uint32_t slot_early_fapi_packets_srs;
    uint32_t slot_late_fapi_packets_srs;
}fapi_srs_stats_info_t;

enum timing_type
{
    EARLY,
    ONTIME,
    LATE,
    MAX_TIMING_TYPES,
};

/*
 * General cuPHYDriver context class (one context per execution)
 */
class PhyDriverCtx {
public:
    PhyDriverCtx(const context_config& ctx_cfg);
    PhyDriverCtx(const context_config & ctx_cfg, bool minimal); 
    ~PhyDriverCtx();
    bool     isValidation();
    bool     isCPlaneDisabled() const;

    /////////////////////////////////////////////////////////////////////
    //// Start
    /////////////////////////////////////////////////////////////////////
    int  start();
    int  isActive() { return active; }
    void setActive() { active = true; }
    void setInactive() { active = false; }

    /////////////////////////////////////////////////////////////////////
    //// L2 Callbacks
    /////////////////////////////////////////////////////////////////////
    int  setUlCb(slot_command_api::ul_slot_callbacks& _ul_cb);
    bool getUlCb(slot_command_api::ul_slot_callbacks& cb);
    int  setDlCb(slot_command_api::dl_slot_callbacks& _dl_cb);
    bool getDlCb(slot_command_api::dl_slot_callbacks& cb);
    int setCellUpdateCb(::CellUpdateCallBackFn& callback);
    bool cellUpdateCbExists();
    /////////////////////////////////////////////////////////////////////
    //// SlotMap management
    /////////////////////////////////////////////////////////////////////
    SlotMapUl* getNextSlotMapUl();
    SlotMapDl* getNextSlotMapDl();

    struct slot_params_aggr* getNextSlotCmd();

    /////////////////////////////////////////////////////////////////////
    //// TaskList management
    /////////////////////////////////////////////////////////////////////
    TaskList* getTaskListUl();
    TaskList* getTaskListDl();
    TaskList* getTaskListDlVal();
    TaskList* getTaskListDebug();
    Task*     getNextTask();

    /////////////////////////////////////////////////////////////////////
    //// Cell management
    /////////////////////////////////////////////////////////////////////
    int     addNewCell(const cell_mplane_info& m_plane_info,uint32_t idx);
    int     setCellPhyByMplane(struct cell_phy_info& cell_pinfo);
    Cell *  getCellById(cell_id_t c_id);
    Cell *  getCellByPhyId(uint16_t c_phy_id);
    Cell *  getCellByMplaneId(uint16_t mplane_id);
    int     setCellPhyId(uint16_t c_phy_id_old, uint16_t c_phy_id_new, cell_id_t cell_id);
    int     removeCell(uint16_t cid);
    int     getCellNum();
    int     getCellList(Cell **clist, uint32_t *pcellCount);
    [[nodiscard]]
    int     getCellIdxList(std::array<uint32_t,MAX_CELLS_PER_SLOT>& cell_idx_list);
    uint8_t getUeMode() const;

    /////////////////////////////////////////////////////////////////////
    //// Worker management
    /////////////////////////////////////////////////////////////////////
    int       addGenericWorker(std::unique_ptr<Worker> w);
    Worker*   getWorkerById(worker_id id);
    worker_id getULWorkerID(int worker_index);
    worker_id getDLWorkerID(int worker_index);
    int       getNumDLWorkers();
    int       getNumULWorkers();
    int       removeWorker(worker_id id);

    /////////////////////////////////////////////////////////////////////
    //// GPU management
    /////////////////////////////////////////////////////////////////////
    GpuDevice* getGpuById(int id);
    int        getGpuNum();
    GpuDevice* getFirstGpu();

    /////////////////////////////////////////////////////////////////////
    //// PhyChannel
    /////////////////////////////////////////////////////////////////////
    int                 createIOBuffers();
    OrderEntity*        getNextOrderEntity(int32_t* cell_idx_list, uint8_t cell_idx_list_size, OrderEntity* oe,bool new_order_entity);
    uint16_t            getStartSectionIdSrs() const;
    uint16_t            getStartSectionIdPrach() const;
    uint8_t             getEnableUlCuphyGraphs() const;
    uint8_t             getEnableDlCuphyGraphs() const;
    uint32_t            getUlOrderTimeoutCPU() const;
    uint32_t            getUlOrderTimeoutGPU() const;
    uint32_t            getUlOrderTimeoutGPUSrs() const;
    uint32_t            getUlSrsAggr3TaskLaunchOffsetNs() const;
    uint8_t             getUlOrderTimeoutGPULogEnable() const;
    uint32_t            getUlOrderMaxRxPkts() const;
    uint32_t            getUlOrderRxPktsTimeout() const;
    uint32_t            getUlOrderTimeoutLogInterval() const;
    uint8_t             getUlOrderKernelMode() const;
    uint32_t            getUlOrderTimeoutFirstPktGPU() const;
    uint32_t            getUlOrderTimeoutFirstPktGPUSrs() const;
    HarqPoolManager *   getHarPoolManager() const;
    WAvgCfoPoolManager * getWAvgCfoPoolManager() const;
    CvSrsChestMemoryBank* getCvSrsChestMemoryBank() const;
    /**
     * @brief Return if green contexts (GC) mode is enabled.
     * @return Non-zero if GC enabled, zero otherwise
     */
    uint8_t             getUseGreenContexts() const;
    /**
     * @brief Return if GC workqueue feature is enabled.
     * Only relevant if green context mode is enabled
     * @return Non-zero if GC WQs enabled, zero otherwise
     */
    uint8_t             getUseGCWorkqueues() const;
    uint8_t             getUseBatchedMemcpy() const;
    int                 getMpsSmPusch() const;
    int                 getMpsSmPucch() const;
    int                 getMpsSmPrach() const;
    int                 getMpsSmUlOrder() const;
    int                 getMpsSmSrs() const;
    int                 getMpsSmPdsch() const;
    int                 getMpsSmPdcch() const;
    int                 getMpsSmPbch() const;
    int                 getMpsSmCsiRs() const;
    int                 getMpsSmDlCtrl() const;
    int                 getMpsSmGpuComms() const;

    uint8_t             getPdschFallback() const;
    MpsCtx*             getUlCtx();
    void                setUlCtx();
    MpsCtx*             getDlCtx();
    void                setDlCtx();
    MpsCtx*             getGpuCommsCtx();
    void                setGpuCommsCtx();
    cudaStream_t        getUlOrderStreamPd();
    cudaStream_t        getUlOrderStreamSrsPd();
    cudaStream_t*       getUlOrderStreamsPusch();
    cudaStream_t*       getUlOrderStreamsPucch();
    cudaStream_t*       getUlOrderStreamsPrach();
    cudaStream_t&       get_stream_timing_dl() {return stream_timing_dl;};
    cudaStream_t&       get_stream_timing_ul() {return stream_timing_ul;};
    void                warmupStream(cudaStream_t stream);

    PhyUlBfwAggr*       getNextUlBfwAggr(slot_params_aggr* aggr_slot_params);
    PhyPuschAggr*       getNextPuschAggr(slot_params_aggr* aggr_slot_params);
    PhyPucchAggr*       getNextPucchAggr(slot_params_aggr* aggr_slot_params);
    PhyPrachAggr*       getNextPrachAggr(slot_params_aggr* aggr_slot_params);
    PhySrsAggr*         getNextSrsAggr(slot_params_aggr* aggr_slot_params);
    PhyDlBfwAggr*       getNextDlBfwAggr(slot_params_aggr* aggr_slot_params);
    void                recordDlBFWCompletion(int slot);
    int                 queryDlBFWCompletion(int slot);
    cudaError_t         queryDlBFWCompletion_v2(int slot);
    void                recordUlBFWCompletion(int slot);
    int                 queryUlBFWCompletion(int slot);
    PhyPdschAggr*       getNextPdschAggr(slot_params_aggr* aggr_slot_params);
    PhyPdcchAggr*       getNextPdcchDlAggr(slot_params_aggr* aggr_slot_params);
    PhyPdcchAggr*       getNextPdcchUlAggr(slot_params_aggr* aggr_slot_params);
    PhyPbchAggr*        getNextPbchAggr(slot_params_aggr* aggr_slot_params);
    PhyCsiRsAggr*       getNextCsiRsAggr(slot_params_aggr* aggr_slot_params);
    void*               getDlHBuffersAddr(int index);
    void*               getDlDBuffersAddr(int index);
    int                 createPrachObjects();
    int                 deletePrachObjects();
    int                 replacePrachObjects();
    int                 updateCellConfig(cell_id_t cell_id, cell_phy_info& cell_pinfo);
    Mutex               updateCellConfigMutex;
    int                 updateCellConfigCellId;
    ::CellUpdateCallBackFn cell_update_cb;

    uint8_t             getPuschWorkCancelMode(void) const;
    uint8_t             getPuschTdi(void) const;
    uint8_t             getPuschCfo(void) const;
    uint8_t             getPuschEqCoeffAlgo(void) const;
    uint8_t             getPuschChEstAlgo(void) const;
    uint8_t             getPuschEnablePerPrgChEst(void) const;
    uint8_t             getPuschTo(void) const;
    uint8_t             getPuschRssi(void) const;
    uint8_t             getPuschSinr(void) const;
    uint8_t             getPuschDftSOfdm(void) const;
    uint8_t             getPuschTbSizeCheck(void) const;
    uint8_t             getPuschEarlyHarqEn(void) const;
    uint8_t             getPuschDeviceGraphLaunchEn(void) const;
    void                setPuschEarlyHarqEn(bool is_early_harq_detection_enabled);
    uint16_t            getPuschWaitTimeOutPreEarlyHarqUs(void) const;
    uint16_t            getPuschWaitTimeOutPostEarlyHarqUs(void) const;
    uint8_t             getPuxchPolarDcdrListSz(void) const;
    [[nodiscard]]
    const std::string&  getPuschrxChestFactorySettingsFilename() const noexcept;
    uint8_t             getNotifyUlHarqBufferRelease(void) const;

    bool                gpuCommDlEnabled(void) const;
    bool                gpuCommEnabledViaCpu(void) const;
    bool                cpuCommEnabled(void) const;

    bool                fixBetaDl(void) const;

    bool                enableL1ParamSanityCheck(void) const;
    uint8_t             enableCPUTaskTracing(void) const;
    bool                enablePrepareTracing(void) const;
    bool                cuptiTracingEnabled(void) const;
    uint64_t            cuptiBufferSize(void) const;
    uint16_t            cuptiNumBuffers(void) const;
    bool                disableEmpw(void) const;
    bool                enableDlCqeTracing(void) const;
    uint64_t            get_cqe_trace_cell_mask(void) const;
    uint32_t            get_cqe_trace_slot_mask(void) const;
    bool                enableOKTb(void) const;
    uint32_t            get_num_ok_tb_slot(void) const;    
    uint8_t             getUlRxPktTracingLevel(void) const;
    uint8_t             getUlRxPktTracingLevelSrs(void) const;
    uint32_t            getUlWarmupFrameCount(void) const;
    uint8_t             getPMUMetrics(void) const;

    bool                debug_worker_enabled() const { return debug_worker != -1; };
    bool                datalake_enabled() const { return data_core != -1; };
    bool                splitUlCudaStreamsEnabled(void) const { return !(!(split_ul_cuda_streams)); };
    bool                serializePucchPusch(void) const { return !(!(serialize_pucch_pusch)); };
    uint16_t            getForcedNumCsi2Bits(void) const;
    uint32_t            getPuschMaxNumLdpcHetConfigs(void) const;
    uint8_t             getPuschMaxNumTbPerNode(void) const;
    uint8_t             getPuschAggrPerCtx(void) const;
    uint8_t             getPrachAggrPerCtx(void) const;
    uint8_t             getPucchAggrPerCtx(void) const;
    uint8_t             getSrsAggrPerCtx(void) const;
    uint8_t             getUlbfwAggrPerCtx(void) const;
    uint16_t            getMaxHarqPools() const;
    uint16_t            getMaxHarqTxCountBundled() const;
    uint16_t            getMaxHarqTxCountNonBundled() const;
    uint8_t             getUlInputBufferPerCell(void) const;
    uint8_t             getUlInputBufferPerCellSrs(void) const;

    /////////////////////////////////////////////////////////////////////
    //// FhProxy
    /////////////////////////////////////////////////////////////////////
    int                 startFhProxy(void);
    FhProxy *           getFhProxy();
    int                 registerBufferToFh(void* buffer_ptr, size_t buffer_size);

    /////////////////////////////////////////////////////////////////////
    //// Logger
    /////////////////////////////////////////////////////////////////////
    log_handler_fn_t get_error_logger() const;
    log_handler_fn_t get_info_logger() const;
    log_handler_fn_t get_debug_logger() const;
    bool             error_log_enabled() const;
    bool             info_log_enabled() const;
    bool             debug_log_enabled() const;
    void             set_error_logger(log_handler_fn_t fn);
    void             set_info_logger(log_handler_fn_t fn);
    void             set_debug_logger(log_handler_fn_t fn);
    void             set_level_logger(l1_log_level _log_lvl);

    cudaEvent_t         getGpuCommsPrepareDoneEvt(void) const { return gpu_comm_prepare_done; };
    cudaStream_t        getH2DCpyStream(void)const { return H2D_TB_CPY_stream;};
    MpsCtx *            getPdschMpsCtx(void)const{return pdschMpsCtx;};
    cudaEvent_t         get_event_pdsch_tb_cpy_complete(uint8_t slot_idx){return pdsch_tb_cpy_complete[slot_idx%MAX_PDSCH_TB_CPY_CUDA_EVENTS];};
    cudaEvent_t         get_event_pdsch_tb_cpy_start(uint8_t slot_idx){return pdsch_tb_cpy_start[slot_idx%MAX_PDSCH_TB_CPY_CUDA_EVENTS];};

    uint32_t             getAggr_obj_non_avail_th(void) const;
    h2d_copy_prepone_info_t* get_h2d_copy_prepone_info(uint16_t idx){return &h2d_cpy_info[idx];};
    void reset_h2d_copy_prepone_info();
    
    uint8_t                 getmMIMO_enable() const;
    uint8_t                 get_enable_srs() const;
    uint8_t                 get_enable_dl_core_affinity() const;
    uint8_t                 get_dlc_core_packing_scheme() const;
    uint8_t                 getCellGroupNum() const;
    uint8_t                 get_ch_segment_proc_enable() const;
    uint32_t                geth2d_copy_wait_th(void) const;
    uint32_t                getcuphy_dl_channel_wait_th(void) const;
    uint32_t                getSendCPlane_timing_error_th_ns(void) const;
    uint32_t                getSendCPlane_ulbfw_backoff_th_ns(void) const;
    uint32_t                getSendCPlane_dlbfw_backoff_th_ns(void) const;
    uint64_t                get_gps_alpha() {return gps_alpha_;};
    int64_t                 get_gps_beta() {return gps_beta_;};
    void                    set_gps_alpha(uint64_t val){gps_alpha_=val;};
    void                    set_gps_beta(int64_t val){gps_beta_=val;};
    aggr_obj_error_info_t*  getAggrObjErrInfo(bool isDl);
    void                    set_slot_advance(uint8_t val){slot_advance=val;};
    uint8_t                 get_slot_advance(){return slot_advance;};
    bool                    get_exit_dl_validation(){return exit_dl_validation.load();};
    Packet_Statistics*   getULPacketStatistics() {return &ul_stats;};
    Packet_Statistics*   getSRSPacketStatistics() {return &srs_stats;};
    Packet_Statistics*   getDLPacketStatistics(int type) {return &dl_stats[type];};
    bool                    ru_health_check_enabled(void) const;
    uint32_t                get_max_ru_unhealthy_slots(void) const;
    cuphySrsChEstAlgoType_t get_srs_chest_algo_type() const;
    uint8_t                 get_srs_chest_tol2_normalization_algo_type() const;
    float                   get_srs_chest_tol2_constant_scaler() const;
    uint8_t                 get_bfw_power_normalization_alg_selector() const;
    float                   get_bfw_beta_prescaler() const;
    uint32_t                get_total_num_srs_chest_buffers() const;
    uint8_t                 get_send_static_bfw_wt_all_cplane() const;
    ru_type                 get_ru_type_for_srs_proc() const;
    void                    set_ru_type_for_srs_proc(ru_type ru_type);

    Mutex h2d_copy_prepone_mutex;
    Mutex dl_aggr_compression_task;
    Mutex dl_cpu_db_task;
    ///////////////////////////////////////////////////////////////
    //// Memory footprint
    //////////////////////////////////////////////////////////////
    size_t ctx_tot_cpu_regular_memory;
    size_t ctx_tot_cpu_pinned_memory;
    size_t ctx_tot_gpu_pinned_memory;
    size_t ctx_tot_gpu_regular_memory;
    MemFoot mf;
    MemFoot cuphyChannelsAccumMf; // cumulative GPU footprint for cuPHY channel objects
    MemFoot wip_accum_mf;  //FIXME TBD if GPU or general
    uint8_t num_pdsch_buff_copy;
    bool enable_prepone_h2d_cpy;
    uint8_t h2d_copy_thread_enable;
    std::thread h2d_cpy_thread;
    std::atomic<uint16_t> h2d_write_idx;
    uint16_t h2d_read_idx; //Currently the read index is non-atomic. However, this would have to change if copy task is offloaded to multile threads vs the current single thread scheme
    std::array<std::atomic<bool>, PDSCH_MAX_GPU_BUFFS> h2d_copy_cuda_event_rec_done;
    std::array<std::atomic<int>, PDSCH_MAX_GPU_BUFFS> h2d_copy_done_cur_slot_idx;
    uint8_t h2d_copy_done_cur_slot_read_idx;
    uint8_t h2d_copy_done_cur_slot_write_idx;
    std::array<bfw_buffer_info*, MAX_CELLS_PER_SLOT> bfw_coeff_buffer;
    bfw_buffer_info* getBfwCoeffBuffer(uint8_t cell_idx) const;
    void setBfwCoeffBuffer(uint8_t cell_idx, bfw_buffer_info* buffer_info) noexcept;
    MpsCtx* getSrsMpsCtx(void)const{return srsMpsCtx;};
    std::vector<MpsCtx*> mpsCtxList;
    //MIB cycle is defined as the time during which MIB contents do not change - which is 16SFNs. For u=1 each SFN has 20 slots.
    const uint32_t default_mib_cycle = 16*SLOTS_PER_FRAME;

    void updateBatchedMemcpyInfo(void* dst_addr, void* src_addr, size_t count);
    [[nodiscard]] cuphyStatus_t performBatchedMemcpy();
    void     resetBatchedMemcpyBatches();
    [[nodiscard]] bool getEnableTxNotification() const;

    DataLake* getDataLake(void);
    bool getAggrObjFreeStatus();
    bool incrL1RecoverySlots();
    bool incrAllObjFreeSlots();
    void resetAllObjFreeSlots();
    void resetL1RecoverySlots();
    uint8_t* getFhBufOkTb(uint8_t cell_idx) const;
    uint8_t* getFhBufOkTbSrs(uint8_t cell_idx) const;
    ok_tb_config_info_t* getOkTbConfig(uint8_t cell_idx) const;
    ok_tb_config_srs_info_t* getOkTbConfigSrs(uint8_t cell_idx) const;
    uint32_t getConfigOkTbNumSlots() const;
    void setConfigOkTbNumSlots(uint32_t num_slots);
    uint32_t getConfigOkTbSrsNumSlots() const;
    void setConfigOkTbSrsNumSlots(uint32_t num_slots);
    uint32_t getConfigOkTbMaxPacketSize() const;
    void setConfigOkTbMaxPacketSize(uint32_t max_packet_size);
    uint8_t getEnableWeightedAverageCfo(void) const;
    /////////////////////////////////////////////////////////////////////
    //// UL PCAP capture
    /////////////////////////////////////////////////////////////////////
    std::thread ul_pcap_capture_thread;
    ul_pcap_capture_context_info_t ul_pcap_capture_context_info;
    uint8_t                 get_ul_pcap_capture_enable() const;
    uint16_t                 get_ul_pcap_capture_mtu() const;
    int launch_ul_capture_thread();
    std::atomic<bool> stop_ul_pcap_thread{false};
    uint8_t                 getPuschAggrFactor() const;
    void                    setPuschAggrFactor(uint8_t pusch_aggr_factor_);
    
    perf_metrics::PercentileTracker order_kernel_timing_tracker{0, 3000000, 1000, 80};  ///< Track Order Kernel completion times per slot (0-3ms range, 1μs buckets, 80 slots)

private:
    std::array<cudaEvent_t,MAX_PDSCH_TB_CPY_CUDA_EVENTS> pdsch_tb_cpy_start;
    std::array<cudaEvent_t,MAX_PDSCH_TB_CPY_CUDA_EVENTS> pdsch_tb_cpy_complete;
    std::array<cudaEvent_t,SLOTS_PER_FRAME> dlbfw_run_completion_event;
    std::array<cudaEvent_t,SLOTS_PER_FRAME> ulbfw_run_completion_event;
    //List of pre-allocated SlotMaps
    std::array<SlotMapUl*, SLOT_MAP_NUM> slot_map_ul_array;
    std::array<SlotMapDl*, SLOT_MAP_NUM> slot_map_dl_array;
    //List of pre-allocated Tasks
    std::array<Task*, TASK_ITEM_NUM> task_item_array;

    //List of tasks list (flexible desing: possibility to have differet task lists)
    std::unique_ptr<TaskList> task_list_ul;
    std::unique_ptr<TaskList> task_list_dl;
    std::unique_ptr<TaskList> task_list_dl_validation;
    std::unique_ptr<TaskList> task_list_debug;
    //List of cells registered in the context
    std::unordered_map<cell_id_t, std::unique_ptr<Cell>> cell_map;
    //List of cells id <> cell Phy Id
    std::unordered_map<uint16_t, cell_id_t> cell_index_map;
    //List of associated workers
    std::unordered_map<worker_id, std::unique_ptr<Worker>> worker_ul_map;
    std::vector<worker_id> worker_ul_ordering;
    std::unordered_map<worker_id, std::unique_ptr<Worker>> worker_dl_map;
    std::vector<worker_id> worker_dl_ordering;
    std::unordered_map<worker_id, std::unique_ptr<Worker>> worker_dl_validation_map;
    std::unordered_map<worker_id, std::unique_ptr<Worker>> worker_generic_map;
    //List of associated GPUs
    std::unordered_map<int, std::unique_ptr<GpuDevice>> gpu_map;

    std::array<h2d_copy_prepone_info_t,(DL_MAX_CELLS_PER_SLOT*PDSCH_MAX_GPU_BUFFS)> h2d_cpy_info;

    //batched memory copy
    uint8_t  use_batched_memcpy; // configuration requested via yaml file or default fall back. Batched async copy use relies on other factors too.
    cuphyBatchedMemcpyHelper m_batchedMemcpyHelper;
    int batched_copies = 0; // number of copies that happen in a batch for PDSCH H2D per cell TB copies.

    // Callbacks
    slot_command_api::ul_slot_callbacks ul_cb;
    slot_command_api::dl_slot_callbacks dl_cb;

    //Logger
    l1_log_level     log_lvl;
    log_handler_fn_t log_err_fn_;
    log_handler_fn_t log_inf_fn_;
    log_handler_fn_t log_dbg_fn_;

    //FhProxy
    std::unique_ptr<FhProxy> fh_proxy; //FIXME: with multiple NICs, one proxy per NIC
    std::vector<struct fh_memreg_buf> fh_memregs_list;

    //Status
    bool active;

    //LockItems
    Mutex mlock_slot_map_ul;
    Mutex mlock_slot_map_dl;
    int   slot_map_ul_index;
    int   slot_map_dl_index;
    Mutex mlock_task;
    int   task_item_index;

    std::array<OrderEntity*, ORDER_ENTITY_NUM> order_entity_list;
    // std::vector<std::unique_ptr<OrderEntity>> order_entity_list;
    Mutex mlock_oentity;
    int   oentity_index;

    //Others
    bool cplane_disable;
    bool standalone;
    bool validation;

    //MPS
    MpsCtx *ulMpsCtx;
    MpsCtx *dlMpsCtx;
    MpsCtx *gpuCommsMpsCtx;
    MpsCtx *puschMpsCtx;
    MpsCtx *pucchMpsCtx;
    MpsCtx *prachMpsCtx;
    MpsCtx *srsMpsCtx;
    MpsCtx *pdschMpsCtx; // + CSI_RS
    MpsCtx *pdcchMpsCtx;
    MpsCtx *pbchMpsCtx;
    MpsCtx *csiRsMpsCtx;
    MpsCtx *dlCtrlMpsCtx;
    MpsCtx *dlBfwMpsCtx;
    MpsCtx *ulBfwMpsCtx;

    uint8_t  use_green_contexts;
    uint8_t  use_gc_workqueues;
    std::array<cuphy::cudaGreenContext, CURRENT_MAX_GREEN_CTXS> tmpGreenContextsForResplit;
#if CUDA_VERSION >= 12040
    CUdevResource initial_device_GPU_resources;
    CUdevResourceType default_resource_type = CU_DEV_RESOURCE_TYPE_SM;
    // All 3 vectors below will be resized to 2*CURRENT_MAX_GREEN_CTXS.
    std::vector<CUdevResource> devResources;
    std::vector<unsigned int> actual_split_groups;
    std::vector<unsigned int> min_sm_counts;
#endif

    //Worker per DL slot
    std::vector<uint8_t> workers_ul_cores;
    std::vector<uint8_t> workers_dl_cores;
    
    std::vector<DLValidationParams> workers_dl_validation_params;
    std::atomic<bool> exit_dl_validation;
    std::vector<uint8_t> workers_dl_validation_cores;
    int16_t debug_worker;

    ////////////////////////////////////////////
    //// Data Collection
    ////////////////////////////////////////////

    std::shared_ptr<DataLake> dataLake;
    int16_t data_core;
    std::thread datalake_pusch_thread;
    std::thread datalake_srs_thread;

    // Section Id per UL channel. ORAN SectionId 12bits
    uint16_t start_section_id_srs;
    uint16_t start_section_id_prach;

    // Order kernel timeout for CPU and GPU
    uint32_t ul_order_timeout_gpu_ns;
    uint32_t ul_order_timeout_gpu_srs_ns;
    uint32_t ul_srs_aggr3_task_launch_offset_ns;
    uint32_t ul_order_timeout_first_pkt_gpu_ns;
    uint32_t ul_order_timeout_first_pkt_gpu_srs_ns;
    uint32_t ul_order_timeout_log_interval_ns;
    uint32_t ul_order_timeout_cpu_ns;
    uint8_t ul_order_timeout_gpu_log_enable;
    uint8_t ul_order_kernel_mode;
    uint32_t ul_order_max_rx_pkts;
    uint32_t ul_order_rx_pkts_timeout_ns;

    //Gather metrics
    int prometheus_cpu_core;
    std::unique_ptr<Metrics> metrics;
    uint8_t enable_ul_cuphy_graphs;
    uint8_t enable_dl_cuphy_graphs;

    //HARQ manager
    std::unique_ptr<HarqPoolManager> hq_manager;

    //Weighted Average CFO Pool Manager
    std::unique_ptr<WAvgCfoPoolManager> wavgcfo_manager;

    //CV Srs Channel estimates Memory Bank
    std::unique_ptr<CvSrsChestMemoryBank> cv_srs_chest_memory_bank;

    uint32_t mps_sm_pusch;
    uint32_t mps_sm_pucch;
    uint32_t mps_sm_prach;
    uint32_t mps_sm_ul_order;
    uint32_t mps_sm_srs;
    uint32_t mps_sm_pdsch;
    uint32_t mps_sm_pdcch;
    uint32_t mps_sm_pbch;
    uint32_t mps_sm_csirs;
    uint32_t mps_sm_dl_ctrl;
    uint32_t mps_sm_gpu_comms;

    uint8_t pdsch_fallback;

    cudaEvent_t                            gpu_comm_prepare_done;

    //////////////////////////////////////////////////////////////////////////////////
    //// cuPHY aggregate
    //////////////////////////////////////////////////////////////////////////////////
    uint8_t                                cell_group;
    uint8_t                                cell_group_num;

    struct slot_params_aggr*               sc_aggr_array;
    Mutex                                  sc_aggr_lock;
    int                                    sc_aggr_index;

    cudaStream_t                           stream_order_pd;
    cudaStream_t                           aggr_stream_ulbfw;
    cudaStream_t                           stream_order_srs_pd;
    cudaStream_t                           aggr_stream_pusch[4];
    cudaStream_t                           aggr_stream_pucch[2];
    cudaStream_t                           aggr_stream_prach[2];
    cudaStream_t                           aggr_stream_srs;

    Mutex                                  aggr_lock_cell_phy_ulbfw;
    Mutex                                  aggr_lock_cell_phy_pusch;
    Mutex                                  aggr_lock_cell_phy_pucch;
    Mutex                                  aggr_lock_cell_phy_prach;
    Mutex                                  aggr_lock_cell_phy_srs;
    std::vector<std::unique_ptr<PhyUlBfwAggr>> aggr_ulbfw_items;
    std::vector<std::unique_ptr<PhyPuschAggr>> aggr_pusch_items;
    std::vector<std::unique_ptr<PhyPucchAggr>> aggr_pucch_items;
    std::vector<std::unique_ptr<PhyPrachAggr>> aggr_prach_items;
    std::vector<std::unique_ptr<PhySrsAggr>> aggr_srs_items;
    int                                    aggr_last_ulbfw;
    int                                    aggr_last_pusch;
    int                                    aggr_last_pucch;
    int                                    aggr_last_prach;
    int                                    aggr_last_srs;

    cudaStream_t                           aggr_stream_dlbfw;
    cudaStream_t                           aggr_stream_pdsch;
    cudaStream_t                           aggr_stream_pdcch;
    cudaStream_t                           aggr_stream_pbch;
    cudaStream_t                           aggr_stream_csirs;
    cudaStream_t                           H2D_TB_CPY_stream;

    Mutex                                  aggr_lock_cell_phy_dlbfw;
    Mutex                                  aggr_lock_cell_phy_pdsch;
    Mutex                                  aggr_lock_cell_phy_pdcch_dl;
    Mutex                                  aggr_lock_cell_phy_pdcch_ul;
    Mutex                                  aggr_lock_cell_phy_pbch;
    Mutex                                  aggr_lock_cell_phy_csirs;
    std::vector<std::unique_ptr<PhyDlBfwAggr>> aggr_dlbfw_items;
    std::vector<std::unique_ptr<PhyPdschAggr>> aggr_pdsch_items;
    std::vector<std::unique_ptr<PhyPdcchAggr>> aggr_pdcch_dl_items;
    std::vector<std::unique_ptr<PhyPdcchAggr>> aggr_pdcch_ul_items;
    std::vector<std::unique_ptr<PhyPbchAggr>> aggr_pbch_items;
    std::vector<std::unique_ptr<PhyCsiRsAggr>> aggr_csirs_items;

    int                                    aggr_last_dlbfw;
    int                                    aggr_last_pdsch;
    int                                    aggr_last_pdcch_dl;
    int                                    aggr_last_pdcch_ul;
    int                                    aggr_last_pbch;
    int                                    aggr_last_csirs;

    // Helper buffers used only in DL context to memset cells' DL output buffers
    cuphy::unique_pinned_ptr<CleanupDlBufInfo> h_dl_buffers_addr;
    cuphy::unique_device_ptr<CleanupDlBufInfo> d_dl_buffers_addr;

    uint32_t                               num_new_prach_handles;

    uint8_t                                pusch_workCancelMode;
    uint8_t                                enable_pusch_tdi;
    uint8_t                                enable_pusch_cfo;
    uint8_t                                select_pusch_eqcoeffalgo;
    uint8_t                                select_pusch_chestalgo;
    uint8_t                                enable_pusch_perprgchest;
    uint8_t                                enable_pusch_to;
    uint8_t                                enable_pusch_rssi;
    uint8_t                                enable_pusch_sinr;
    uint8_t                                enable_weighted_average_cfo;
    uint8_t                                enable_pusch_dftsofdm;
    uint8_t                                enable_pusch_tbsizecheck;
    uint8_t                                pusch_earlyHarqEn;
    uint8_t                                pusch_deviceGraphLaunchEn;
    uint16_t                               pusch_waitTimeOutPreEarlyHarqUs;
    uint16_t                               pusch_waitTimeOutPostEarlyHarqUs;
    uint8_t                                mPuxchPolarDcdrListSz;
    std::string                            mPuschrxChestFactorySettingsFilename;
    uint8_t                                notify_ul_harq_buffer_release;
    uint8_t                                enable_gpu_comm_dl;
    uint8_t                                enable_gpu_comm_via_cpu;
    uint8_t                                enable_cpu_init_comms;
    uint8_t                                fix_beta_dl;
    uint8_t                                disable_empw; 
    uint8_t                                enable_cpu_task_tracing;
    uint8_t                                enable_l1_param_sanity_check;
    uint8_t                                enable_prepare_tracing;
    uint8_t                                cupti_enable_tracing;
    uint64_t                               cupti_buffer_size;
    uint16_t                               cupti_num_buffers;
    uint8_t                                enable_dl_cqe_tracing;
    uint64_t                               cqe_trace_cell_mask;
    uint32_t                               cqe_trace_slot_mask;
    uint8_t                                enable_ok_tb;
    uint32_t                               num_ok_tb_slot;        
    uint8_t                                ul_rx_pkt_tracing_level;
    uint8_t                                ul_rx_pkt_tracing_level_srs;
    uint32_t                               ul_warmup_frame_count;
    uint8_t                                pmu_metrics;
    uint8_t                                mMIMO_enable;
    uint8_t                                enable_srs;
    uint8_t                                enable_dl_core_affinity;
    uint8_t                                dlc_core_packing_scheme;
    uint8_t                                ue_mode;
    uint8_t                                mCh_segment_proc_enable;
    uint32_t                               aggr_obj_non_avail_th;
    uint8_t                                split_ul_cuda_streams;
    uint8_t                                serialize_pucch_pusch;
    uint32_t                               sendCPlane_timing_error_th_ns;
    uint32_t                               sendCPlane_ulbfw_backoff_th_ns;
    uint32_t                               sendCPlane_dlbfw_backoff_th_ns;
    uint64_t                               gps_alpha_ = 0;
    int64_t                                gps_beta_ = 0;
    uint8_t                                pusch_aggr_per_ctx;
    uint8_t                                pucch_aggr_per_ctx;
    uint8_t                                srs_aggr_per_ctx;
    uint8_t                                ulbfw_aggr_per_ctx;
    uint16_t                               max_harq_pools;
    uint16_t                               max_harq_tx_count_bundled;
    uint16_t                               max_harq_tx_count_non_bundled;
    uint8_t                                prach_aggr_per_ctx;
    uint8_t                                ul_input_buffer_per_cell;
    uint8_t                                ul_input_buffer_per_cell_srs;
    uint8_t                                send_static_bfw_wt_all_cplane;
    uint32_t                               max_ru_unhealthy_ul_slots;
    uint8_t                                ul_pcap_capture_enable;
    uint16_t                               ul_pcap_capture_mtu;
    uint8_t                                ul_pcap_capture_thread_cpu_affinity;
    uint8_t                                ul_pcap_capture_thread_sched_priority;
    uint8_t                                pcap_logger_ul_cplane_enable;
    uint8_t                                pcap_logger_dl_cplane_enable;  
    uint8_t                                pcap_logger_thread_cpu_affinity;
    uint8_t                                pcap_logger_thread_sched_prio;   
    std::string                            pcap_logger_file_save_dir;
    cuphySrsChEstAlgoType_t                srs_chest_algo_type;
    uint8_t                                srs_chest_tol2_normalization_algo_type;
    float                                  srs_chest_tol2_constant_scaler;
    uint8_t                                bfw_power_normalization_alg_selector;
    float                                  bfw_beta_prescaler;
    uint32_t                               total_num_srs_chest_buffers;
    /*UL/DL Aggregated Objects error handling specific*/
    aggr_obj_error_info_t                  aggr_error_info_dl;
    aggr_obj_error_info_t                  aggr_error_info_ul;  
    //DL wait thresholds(ns)
    uint32_t                               h2d_copy_wait_th;
    uint32_t                               cuphy_dl_channel_wait_th;
    uint16_t                               forcedNumCsi2Bits;
    uint32_t                               pusch_nMaxLdpcHetConfigs;
    uint8_t                                pusch_nMaxTbPerNode;
    cudaStream_t                           stream_timing_dl;
    cudaStream_t                           stream_timing_ul;
    uint8_t                                slot_advance;
    Packet_Statistics ul_stats;
    Packet_Statistics srs_stats;
    Packet_Statistics dl_stats[Packet_Statistics::MAX_DL_PACKET_TYPES];
    std::array<uint8_t*,UL_MAX_CELLS_PER_SLOT> fh_buf_ok_tb;
    std::array<uint8_t*,UL_MAX_CELLS_PER_SLOT> fh_buf_ok_tb_srs;
    std::array<ok_tb_config_info_t*,UL_MAX_CELLS_PER_SLOT> config_ok_tb;
    std::array<ok_tb_config_srs_info_t*,UL_MAX_CELLS_PER_SLOT> config_ok_tb_srs;
    uint32_t                               config_ok_tb_num_slots;
    uint32_t                               config_ok_tb_srs_num_slots;
    uint32_t                               config_ok_tb_max_packet_size;
    std::array<std::array<fapi_srs_stats_info_t,MAX_LAUNCH_PATTERN_SLOTS>, MAX_CELLS_PER_SLOT> srs_fapi_stats_info;
    std::array<std::array<std::array<std::atomic<uint64_t>, MAX_TIMING_TYPES>, MAX_LAUNCH_PATTERN_SLOTS>, MAX_CELLS_PER_SLOT> srs_fapi_stats;
    std::array<bool, MAX_LAUNCH_PATTERN_SLOTS> srs_fapi_active_slots;
    ru_type ru_type_for_srs_proc;
    uint8_t pusch_aggr_factor;
    bool minimal_phydriver = false; // Default is false, minimal constructor sets it to true
    uint8_t enable_tx_notification{};  ///< Enable TX notification flag
};

struct phydriverwh
{
    Worker*       w;
    PhyDriverCtx* pdctx;
};

#endif
