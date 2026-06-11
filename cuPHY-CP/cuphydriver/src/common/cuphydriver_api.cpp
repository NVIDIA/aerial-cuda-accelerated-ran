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

// Must be included before #define TAG, since it has template with parameter <TAG> and the below
// will impact the parser.
#include "backward.hpp"

#define TAG (NVLOG_TAG_BASE_CUPHY_DRIVER + 4) // "DRV.API"
#define TAG_STARTUP_TIMES (NVLOG_TAG_BASE_CUPHY_CONTROLLER + 5) // "CTL.STARTUP_TIMES"

#include "app_config.hpp"
#include "app_utils.hpp"
#include "constant.hpp"
#include "context.hpp"
#include "time.hpp"
#include "task.hpp"
#include "cell.hpp"
#include "slot_map_ul.hpp"
#include "slot_map_dl.hpp"
#include "worker.hpp"
#include "gpudevice.hpp"
#include "time.hpp"
#include "cuphydriver_api.hpp"
#include "data_lake.hpp"   // DataLake::notifyBfw (for l1_datalake_notify_bfw)
#include "exceptions.hpp"
#include "nvlog.hpp"
#include "cuphyoam.hpp"
#include "oran_utils/conversion.hpp"
#include <cuda_profiler_api.h>
#include <unistd.h>
#include "scf_5g_fapi.h"
#include "ti_generic.hpp"
#include <rte_trace.h>

#include "ptp_service_status_checking.hpp"
#include "rhocp_ptp_event_consumer.hpp"

#define COMBINE_DL_TASKS_WITH_GPU_INIT_COMMS 0

phydriver_handle l1_pdh;
pthread_t gBg_thread_id;

void* ptp_svc_monitoring_func(void* arg);
void* rhocp_ptp_events_monitoring_func(void* arg);

int l1_init(phydriver_handle* _pdh, const context_config& ctx_cfg)
{
    TI_GENERIC_INIT("l1_init",8);

    TI_GENERIC_ADD("Start Task");

    PhyDriverCtx* pdctx    = nullptr;
    int           ret      = 0;
    int           gpu_id   = DEFAULT_GPU_ID;
    bool          init_gdr = false;
    GpuDevice * gDev;
    uint32_t cell_idx=0;

    try
    {
        TI_GENERIC_ADD("PhyDriverCtx construct");
        pdctx = new PhyDriverCtx(ctx_cfg);
        TI_GENERIC_ADD("PhyDriverCtx convert");
        *_pdh = StaticConversion<void>(pdctx).get();

        TI_GENERIC_ADD("Add each cell");
        uint32_t count = 0;
        // Add cells with M-plane parameters

        if(ctx_cfg.cell_mplane_list.size() > 0 && ctx_cfg.mMIMO_enable == 1)
        {
            auto t1a_min_cp_dl_ns = ctx_cfg.cell_mplane_list[0].t1a_min_cp_dl_ns;
            auto t1a_max_cp_dl_ns = ctx_cfg.cell_mplane_list[0].t1a_max_cp_dl_ns;
            auto t1a_min_cp_ul_ns = ctx_cfg.cell_mplane_list[0].t1a_min_cp_ul_ns;
            auto t1a_max_cp_ul_ns = ctx_cfg.cell_mplane_list[0].t1a_max_cp_ul_ns;

            for(auto& m : ctx_cfg.cell_mplane_list)
            {
                if(m.t1a_min_cp_dl_ns != t1a_min_cp_dl_ns)
                {
                    NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "We don't support different t1a_min_cp_dl_ns across cells");
                    goto exit;
                }
                if(m.t1a_max_cp_dl_ns != t1a_max_cp_dl_ns)
                {
                    NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "We don't support different t1a_max_cp_dl_ns across cells");
                    goto exit;
                }
                if(ctx_cfg.mMIMO_enable)
                {
                    if(m.t1a_min_cp_ul_ns != t1a_min_cp_ul_ns)
                    {
                        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "We don't support different t1a_min_cp_ul_ns across cells for MIMO case");
                        goto exit;
                    }
                    if(m.t1a_max_cp_ul_ns != t1a_max_cp_ul_ns)
                    {
                        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "We don't support different t1a_max_cp_ul_ns across cells for MIMO case");
                        goto exit;
                    }
                }
            }
        }

        for(auto& m : ctx_cfg.cell_mplane_list)
        {
            ret = pdctx->addNewCell(m,cell_idx);
            if(ret)
            {
                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "New cell creation error {}", ret);
                goto exit;
            }
            cell_idx++;
            if(++count == ctx_cfg.cell_group_num)
                break;
        }

        TI_GENERIC_ADD("PhyDriverCtx start");
        ret = pdctx->start();
        if(ret)
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Couldn't start cuPHYDriver context {}", ret);
            goto exit;
        }
        if(AppConfig::getInstance().isPtpSvcMonitoringEnabled())
        {
            pthread_t thread_id;
            int       status = pthread_create(&thread_id, nullptr, ptp_svc_monitoring_func, *_pdh);
            if(status)
            {
                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "pthread_create ptp_service_monitoring_thread_func failed with status : {}", std::strerror(status));
                goto exit;
            }
        } 
        else if (AppConfig::getInstance().isRhocpPtpEventsMonitoringEnabled())
        {
            pthread_t thread_id;
            NVLOGC_FMT(TAG, "Now will start RHOCP PTP events monitoring thread");
            int       status = pthread_create(&thread_id, nullptr, rhocp_ptp_events_monitoring_func, *_pdh);
            if(status)
            {
                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "pthread_create rhocp_ptp_events_monitoring_thread_func failed with status : {}", std::strerror(status));
                goto exit;
            }
        }


        TI_GENERIC_ADD("End Task");
        TI_GENERIC_ALL_NVLOGI(TAG_STARTUP_TIMES);
        return ret;
    }
    PHYDRIVER_CATCH_EXCEPTIONS();

exit:
    // All goto triggers are from errors above
    return -1;
}

int l1_finalize(phydriver_handle pdh)
{
    PhyDriverCtx* pdctx = nullptr;

    try
    {
        pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
        delete pdctx;
    }
    PHYDRIVER_CATCH_EXCEPTIONS();

    return 0;
}

int l1_worker_start_generic(phydriver_handle pdh, phydriverwrk_handle* _wh, const char* name, uint8_t affinity_core, uint32_t sched_priority, worker_routine wr, void* args)
{
    PhyDriverCtx* pdctx = nullptr;
    Worker*       w     = nullptr;
    int           ret   = 0;
    worker_id     wid   = 0;

    wid = create_worker_id();
    try
    {
        if(_wh == nullptr)
            PHYDRIVER_THROW_EXCEPTIONS(errno, "Worker handler provided is nullptr");

        pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
        w     = new Worker(pdh, wid, WORKER_GENERIC, name, affinity_core,
                           sched_priority, 0, wr, args); //Type here is useless
        *_wh  = w;

        // Add this metrics generic worker to the generic worker map - will be started in context.cpp
        if(pdctx->addGenericWorker(std::unique_ptr<Worker>(std::move(w))))
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "New generic worker can't be added to the context {}: {}", errno , std::strerror(errno));
            PHYDRIVER_THROW_EXCEPTIONS(errno, "addNewWorker");
        }
    }
    PHYDRIVER_CATCH_EXCEPTIONS();

    return 0;
}

worker_id l1_worker_get_id(phydriverwrk_handle whandler)
{
    Worker* w = nullptr;

    try
    {
        w = StaticConversion<Worker>(whandler).get();
        return (worker_id)w->getId();
    }
    PHYDRIVER_CATCH_EXCEPTIONS();

    return 0;
}

phydriver_handle l1_worker_get_phydriver_handler(phydriverwrk_handle whandler)
{
    Worker* w = nullptr;

    try
    {
        w = StaticConversion<Worker>(whandler).get();
        return w->getPhyDriverHandler();
    }
    PHYDRIVER_CATCH_EXCEPTIONS_RETVAL(nullptr);

    return nullptr;
}

bool l1_worker_check_exit(phydriverwrk_handle whandler)
{
    Worker* w = nullptr;

    try
    {
        w = StaticConversion<Worker>(whandler).get();
        return w->getExitValue();
    }
    PHYDRIVER_CATCH_EXCEPTIONS();

    return true;
}

int l1_worker_stop(phydriverwrk_handle whandler)
{
    int           ret   = 0;
    worker_id     wid   = 0;
    Worker*       w     = nullptr;
    PhyDriverCtx* pdctx = nullptr;

    try
    {
        w     = StaticConversion<Worker>(whandler).get();
        pdctx = StaticConversion<PhyDriverCtx>(w->getPhyDriverHandler()).get();
        wid   = l1_worker_get_id(whandler);
        if(wid == 0)
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Worker doesn't exist");
            ret = -1;
        }

        ret = pdctx->removeWorker(wid);
        if(ret)
            PHYDRIVER_THROW_EXCEPTIONS(-1, "Can't remove worker");
    }
    PHYDRIVER_CATCH_EXCEPTIONS();

exit:
    return ret;
}

int l1_set_output_callback(phydriver_handle pdh, struct slot_command_api::callbacks& cb)
{
    int           ret   = 0;
    PhyDriverCtx* pdctx = nullptr;

    try
    {
        pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    }
    PHYDRIVER_CATCH_EXCEPTIONS();

    ret = pdctx->setUlCb(cb.ul_cb);
    if(ret)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Couldn't set Uplink callback {}", ret);
        return -1;
    }

    ret = pdctx->setDlCb(cb.dl_cb);
    if(ret)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Couldn't set Downlink callback {}", ret);
        return -1;
    }

    return 0;
}

int get_num_ulc_tasks(int num_workers) {
    //Previosly we scaled this based on number of cells
    //odd -> round up  (num_cells=15 > num_cplane_tasks=8)
    //even -> num_cells == 2*num_cplane_tasks (num_cells=16 > num_cplane_tasks=8)
    // return (num_cells / 2) + (num_cells%2);
    return num_workers;
}

int get_num_dlc_tasks(int num_workers,bool commViaCpu, uint8_t mMIMO_enable) {
    // Ensure minimum number of workers required
    if (num_workers < 2) {
        return 0;  // Not enough workers to handle DLC tasks
    }

    //Previosly we scaled this based on number of cells
    //odd -> round up  (num_cells=15 > num_cplane_tasks=8)
    //even -> num_cells == 2*num_cplane_tasks (num_cells=16 > num_cplane_tasks=8)
    // return (num_cells / 2) + (num_cells%2);
    // For muMIMO case keeping the num dlc tasks to 2 as here total number of DL 
    // cores are set to 4
    if(commViaCpu)
    {
        if(mMIMO_enable)
            return num_workers-3;
        else
            return num_workers-2;
    }
    else if(mMIMO_enable)
    {
        if(num_workers > 5)
            return num_workers-2;
        else
            return num_workers;
    }
    else
        return num_workers-1;
}

int l1_enqueue_phy_work(phydriver_handle pdh, struct slot_command_api::slot_command* sc)
{
    PhyDriverCtx*                           pdctx       = nullptr;
    SlotMapUl*                              slot_map_ul = nullptr;
    SlotMapDl*                              slot_map_dl = nullptr;
    int                                     tentative = 0, mu = 0, num_cells = 0, first_cell = 0, task_index = 0, max_ul_uc_delay = 0, min_slot_ahead = 100,dl_task_count=0,ul_task_count=0;
    bool pucch_or_pusch_found = false;
    TaskList*                               tListUl = nullptr;
    TaskList*                               tListDl = nullptr;
    std::array<t_ns, TASK_MAX_PER_SLOT + 1> task_ts_exec;
    std::array<t_ns, TASK_MAX_PER_SLOT + 1> task_ts_enq;
    t_ns                                    waitns(10 * 1000);
    t_ns                                    t0, t1, t2, t3, t4, t5, t6;
    t_ns                                    start_task;
    struct slot_params*                     current_slot_params = nullptr;
    struct slot_params_aggr*                current_slot_params_aggr = nullptr;
    std::array<Task*, TASK_MAX_PER_SLOT>    task_dl_ptr_list;
    std::array<Task*, TASK_MAX_PER_SLOT>    task_ul_ptr_list;
    bool                                    ulbuffer_st1_needed = true;
    std::array<ULInputBuffer*, PRACH_MAX_OCCASIONS> ulbuf_st3_v = {nullptr};
    int rach_occasion = 0;
    slot_command_api::slot_info& slot {sc->cell_groups.slot};
    bool order_entity_set=false;
    /*
     * Here we assume L1 supports only homogeneous cells with MU = 1
     */
    ULInputBuffer * ulbuf_st1    = nullptr;
    ULInputBuffer * ulbuf_st2    = nullptr;
    ULInputBuffer * ulbuf_pcap_capture    = nullptr;
    ULInputBuffer * ulbuf_pcap_capture_ts  = nullptr;
    DLOutputBuffer* dlbuf        = nullptr;
    OrderEntity*    oentity_ptr  = nullptr;
    OrderEntity*    oentity_ptr_tmp  = nullptr;

    PhyPuschAggr*   aggr_pusch_ptr = nullptr;
    PhyPucchAggr*   aggr_pucch_ptr = nullptr;
    PhyPrachAggr*   aggr_prach_ptr = nullptr;
    PhySrsAggr*   aggr_srs_ptr   = nullptr;
    PhyPdschAggr*   aggr_pdsch_ptr = nullptr;
    PhyPdcchAggr*   aggr_pdcch_dl_ptr = nullptr;
    PhyPdcchAggr*   aggr_pdcch_ul_ptr = nullptr;
    PhyPbchAggr*    aggr_pbch_ptr = nullptr;
    PhyDlBfwAggr*   aggr_dlbfw_ptr=nullptr;
    PhyUlBfwAggr*   aggr_ulbfw_ptr=nullptr;
    PhyCsiRsAggr*   aggr_csirs_ptr = nullptr;
    int32_t cell_dl_list[DL_MAX_CELLS_PER_SLOT];
    uint32_t cell_dl_list_idx = 0, tmpdl = 0;
    int32_t cell_ul_list[UL_MAX_CELLS_PER_SLOT];
    uint32_t cell_ul_list_idx = 0, tmpul = 0 ;
    std::vector<int32_t> * phy_cell_index_list;
    std::vector<int32_t> * cell_index_list;
    bool isAggrObjAvail=true;
    bool isUlDlBufAvail=true;
    bool isOKobjAvail=true;
    bool en_orderKernel_tb=false;
    ru_type ru_type_for_srs_proc = OTHER_MODE;


    if(pdh == nullptr) {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "l1_enqueue_phy_work returned error for pdh == nullptr");
        return EINVAL;
    }

    t0 = Time::nowNs();

    try
    {
        pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
        if(pdctx->isActive() == false)
        {
            NVLOGF_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "This cuPHYDriver context is not active, can't enqueue any work");
            return -1;
        }
    }
    PHYDRIVER_CATCH_EXCEPTIONS();
    aggr_obj_error_info_t* errorInfoDl = pdctx->getAggrObjErrInfo(true);
    aggr_obj_error_info_t* errorInfoUl = pdctx->getAggrObjErrInfo(false);

    min_slot_ahead = pdctx->get_slot_advance();
    en_orderKernel_tb = pdctx->enableOKTb();

    t1 = Time::nowNs();

    // t3 = Time::nowNs();

    for (auto &cell : sc->cells)
    {
        if (slot.type != slot_command_api::slot_type::SLOT_NONE)
            break;
        else
            slot = cell.slot;
    }

    if (slot.type != slot_command_api::slot_type::SLOT_NONE)
    {
        // Clean-up sc->tick_original if L2 Adapter screwed it up
        uint64_t t0_slot = sfn_to_tai(slot.slot_3gpp.sfn_, slot.slot_3gpp.slot_, sc->tick_original.count() + AppConfig::getInstance().getTaiOffset(), (int64_t)pdctx->get_gps_alpha(),pdctx->get_gps_beta(), 1) - AppConfig::getInstance().getTaiOffset();
        uint64_t correct_tick = t0_slot - ((min_slot_ahead) * Cell::getTtiNsFromMu(MU_SUPPORTED));
        slot.slot_3gpp.t0_ = t0_slot;
        slot.slot_3gpp.t0_valid_ = true;

        NVLOGI_FMT(TAG,"SFN {}.{} L2A tick {} correct tick {} error {}, gps_alpha={}, gps_beta={}",
               slot.slot_3gpp.sfn_,
               slot.slot_3gpp.slot_,
               sc->tick_original.count(),
               correct_tick,
               sc->tick_original.count()-correct_tick,
               pdctx->get_gps_alpha(),
               pdctx->get_gps_beta());

        if (correct_tick != sc->tick_original.count())
        {
            sc->tick_original = t_ns(correct_tick);
        }
    }

    uint64_t t0_slot = slot.slot_3gpp.t0_valid_ ? slot.slot_3gpp.t0_ : (sc->tick_original + t_ns(min_slot_ahead * Cell::getTtiNsFromMu(MU_SUPPORTED))).count();

    ////////////////////////////////////////////////////////////////////////////
    //// Split cells by direction UL/DL
    ////////////////////////////////////////////////////////////////////////////
    {
        CuphyOAM *oam = CuphyOAM::getInstance();
        // FIXME: HACK for the H5Dump ACK
        if(oam->puschH5dumpInProgress.load())
        {
            static int drop_count = 0;
            NVLOGI_FMT(TAG, "SFN {}.{} Drop slot command due to H5Dump Mechanism (Total dropped this run {})", static_cast<unsigned>(sc->cells[0].slot.slot_3gpp.sfn_), static_cast<unsigned>(sc->cells[0].slot.slot_3gpp.slot_), drop_count++);
            goto cleanup_err;
        }
        current_slot_params_aggr = pdctx->getNextSlotCmd(); //new slot_params_aggr(sc->cell_groups.slot.slot_3gpp, &sc->cell_groups);
        current_slot_params_aggr->populate(&(sc->cell_groups.slot.slot_3gpp), &(sc->cell_groups));
        for (uint8_t i = 0; i < sc->cell_groups.channel_array_size; i++)
        // for(auto& ch : sc->cell_groups.channels)
        {
            auto ch = sc->cell_groups.channels[i];
            /* Generic DL items */
            if(ch == slot_command_api::PDSCH || ch == slot_command_api::PDCCH_DL || ch == slot_command_api::PDCCH_UL || ch == slot_command_api::PBCH || ch == slot_command_api::CSI_RS || (ch == slot_command_api::BFW && current_slot_params_aggr->cgcmd->get_bfw_params()->bfw_cvi_type==slot_command_api::DL_BFW))
            {
                if(slot_map_dl == nullptr)
                {
                    slot_map_dl = pdctx->getNextSlotMapDl();
                    if(slot_map_dl == nullptr)
                    {
                        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "SFN {}.{} SlotMap UL error", sc->cell_groups.slot.slot_3gpp.sfn_, sc->cell_groups.slot.slot_3gpp.slot_);
                        goto cleanup_err;
                    }
                    else
                        NVLOGI_FMT(TAG, "SFN {}.{} Map {} direction DL at {}", static_cast<unsigned>(sc->cell_groups.slot.slot_3gpp.sfn_), static_cast<unsigned>(sc->cell_groups.slot.slot_3gpp.slot_), slot_map_dl->getId(), Time::nowNs().count());

                    slot_map_dl->setSlot3GPP(*current_slot_params_aggr->si);
                    auto dyn_beam_id_offset = pdctx->getFhProxy()->getDynamicBeamIdOffsetOfPrevSlot();
                    slot_map_dl->setDynBeamIdOffset(dyn_beam_id_offset);
                }

                phy_cell_index_list = nullptr;
                cell_index_list = nullptr;
                if(ch == slot_command_api::PDSCH)
                {
                    phy_cell_index_list = &(current_slot_params_aggr->cgcmd->pdsch->phy_cell_index_list);
                    cell_index_list = &(current_slot_params_aggr->cgcmd->pdsch->cell_index_list);
                    //Enable this code snippet to Start/Stop profiling in the middle of a test run

                    // if (3==(slot_map_dl->getId()))    {
                    //     NVLOGC_FMT(TAG, "Starting profiler");
                    //     cudaProfilerStart();
                    // }
                    // if (10==(slot_map_dl->getId())) {
                    //     NVLOGC_FMT(TAG, "Stopping profiler");
                    //     cudaProfilerStop();
                    // }
#if 0 
                     if ((sc->cell_groups.slot.slot_3gpp.sfn_ == 0) && (sc->cell_groups.slot.slot_3gpp.slot_ == 7)) {
                        NVLOGC(TAG, "Starting profiler");
                        cudaProfilerStart();
                    }
                     if ((sc->cell_groups.slot.slot_3gpp.sfn_ == 1) && (sc->cell_groups.slot.slot_3gpp.slot_ == 2)) {
                        NVLOGC(TAG, "Stopping profiler");
                        cudaProfilerStop();
                    }
 
#endif

                }
                else if(ch == slot_command_api::PDCCH_DL || ch == slot_command_api::PDCCH_UL)
                {
                    phy_cell_index_list = &(current_slot_params_aggr->cgcmd->pdcch->phy_cell_index_list);
                    cell_index_list = &(current_slot_params_aggr->cgcmd->pdcch->cell_index_list);
                }
                else if(ch == slot_command_api::PBCH)
                {
                    phy_cell_index_list = &(current_slot_params_aggr->cgcmd->pbch->phy_cell_index_list);
                    cell_index_list = &(current_slot_params_aggr->cgcmd->pbch->cell_index_list);
                }
                else if(ch == slot_command_api::CSI_RS)
                {
                    phy_cell_index_list = &(current_slot_params_aggr->cgcmd->csirs->phy_cell_index_list);
                    cell_index_list = &(current_slot_params_aggr->cgcmd->csirs->cell_index_list);
                }

                if(ch != slot_command_api::BFW)
                {
                    int cell_index=0;
                    for(auto& cell_phy_id : *phy_cell_index_list)
                    {
                        // auto findptr = std::find(std::begin(cell_dl_list), std::end(cell_dl_list), (int) cell_phy_id);
                        // if (findptr != std::end(cell_dl_list)) {
                        //     cell_index++;
                        //     continue;
                        // }

                        for (tmpdl=0; tmpdl < cell_dl_list_idx; tmpdl++) {
                            if (cell_dl_list[tmpdl] == (int) cell_phy_id)
                                break;
                        }

                        if (tmpdl < cell_dl_list_idx) {
                            cell_index++;
                            continue;
                        }

                        dlbuf = nullptr;

                        Cell* cell_ptr = pdctx->getCellByPhyId(cell_phy_id);
                        if(cell_ptr == nullptr)
                        {
                            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Cell {} is not present in the PhyDriver context", cell_phy_id);
                            cell_index++;
                            continue;
                        }

                        if(cell_ptr->isActive() == false)
                        {
                            NVLOGW_FMT(TAG, "Cell {} is not active, can't run anything", cell_ptr->getPhyId());
                            cell_index++;
                            continue;
                        }

                        // Assume homogeneous cells
                        mu = cell_ptr->getMu();

                        dlbuf = cell_ptr->getNextDlBuffer();
                        if(dlbuf == nullptr)
                        {
                            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "SFN {}.{} No available DL output buffers for cell {}", sc->cell_groups.slot.slot_3gpp.sfn_, sc->cell_groups.slot.slot_3gpp.slot_, cell_ptr->getPhyId());
                            isUlDlBufAvail=false;
                            goto cleanup_err;
                        }
                        else
                            NVLOGI_FMT(TAG, "SFN {}.{} Map {} Cell {} with MU {} DLBuffer {} at {}",
                                        static_cast<unsigned>(sc->cell_groups.slot.slot_3gpp.sfn_), static_cast<unsigned>(sc->cell_groups.slot.slot_3gpp.slot_),
                                        slot_map_dl->getId(), cell_ptr->getPhyId(), mu, dlbuf->getId(), Time::nowNs().count());

                        if(slot_map_dl->aggrSetCells(cell_ptr, &sc->cells[(*cell_index_list)[cell_index]].params, dlbuf))
                        {
                            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "SlotMap DL can't set another cell");
                            goto cleanup_err;
                        }

                        // Avoid to waste UL thread time starting it too early
                        if(max_ul_uc_delay < cell_ptr->getT1aMaxCpUlNs())
                            max_ul_uc_delay = cell_ptr->getT1aMaxCpUlNs();

                        // cell_dl_list.push_back(cell_phy_id);
                        cell_dl_list[cell_dl_list_idx] = cell_phy_id;
                        cell_dl_list_idx++;
                        cell_index++;
                    }
                }
            }

            /* Generic UL items */
            if(ch == slot_command_api::PUSCH ||ch == slot_command_api::PUCCH || ch == slot_command_api::PRACH || ch == slot_command_api::SRS || (ch == slot_command_api::BFW && current_slot_params_aggr->cgcmd->get_bfw_params()->bfw_cvi_type==slot_command_api::UL_BFW))
            {
                if(slot_map_ul == nullptr)
                {
                    slot_map_ul = pdctx->getNextSlotMapUl();
                    if(slot_map_ul == nullptr)
                    {
                        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "SFN {}.{} SlotMap UL error", sc->cell_groups.slot.slot_3gpp.sfn_, sc->cell_groups.slot.slot_3gpp.slot_);
                        goto cleanup_err;
                    }
                    else
                        NVLOGI_FMT(TAG, "SFN {}.{} Map {} direction UL at {}", static_cast<unsigned>(sc->cell_groups.slot.slot_3gpp.sfn_), static_cast<unsigned>(sc->cell_groups.slot.slot_3gpp.slot_), slot_map_ul->getId(), Time::nowNs().count());

                    auto dyn_beam_id_offset = pdctx->getFhProxy()->getDynamicBeamIdOffsetOfPrevSlot();
                    slot_map_ul->setDynBeamIdOffset(dyn_beam_id_offset);
                }

                slot_map_ul->setSlot3GPP(*current_slot_params_aggr->si);

                int cell_index=0;
                /*By default */
                phy_cell_index_list = nullptr;
                cell_index_list = nullptr;
                if(ch == slot_command_api::PUSCH)
                {
                    phy_cell_index_list = &(current_slot_params_aggr->cgcmd->pusch->phy_cell_index_list);
                    cell_index_list = &(current_slot_params_aggr->cgcmd->pusch->cell_index_list);
                }
                else if(ch == slot_command_api::PUCCH)
                {
                    phy_cell_index_list = &(current_slot_params_aggr->cgcmd->pucch->phy_cell_index_list);
                    cell_index_list = &(current_slot_params_aggr->cgcmd->pucch->cell_index_list);
                }
                else if(ch == slot_command_api::PRACH)
                {
                    phy_cell_index_list = &(current_slot_params_aggr->cgcmd->prach->phy_cell_index_list);
                    cell_index_list = &(current_slot_params_aggr->cgcmd->prach->cell_index_list);
                }
                else if(ch == slot_command_api::SRS)
                {
                    phy_cell_index_list = &(current_slot_params_aggr->cgcmd->srs->phy_cell_index_list);
                    cell_index_list = &(current_slot_params_aggr->cgcmd->srs->cell_index_list);
                }

                if(ch != slot_command_api::BFW)
                {
                    bool ru_type_found = false;
                    /* Check if all the cells are valid */
                    for(auto& cell_phy_id : *phy_cell_index_list)
                    {
                        for(tmpul = 0; tmpul < cell_ul_list_idx; tmpul++)
                        {
                            if(cell_ul_list[tmpul] == (int) cell_phy_id)
                            {
                                break;
                            }
                        }
                        if (tmpul < cell_ul_list_idx)
                        {
                            cell_index++;
                            continue;
                        }

                        oentity_ptr = nullptr;
                        // ulbuf_st3_v.clear();
                        rach_occasion = 0;

                        Cell* cell_ptr = pdctx->getCellByPhyId(cell_phy_id);
                        //NVLOGC_FMT(TAG, "Cell phy id = {}, cell index = {}", cell_phy_id, (*cell_index_list)[cell_index]);
                        if(cell_ptr == nullptr)
                        {
                            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Cell {} is not present in the PhyDriver context", cell_phy_id);
                            cell_index++;
                            continue;
                        }

                        if(ru_type_found == false)
                        {
                            ru_type_for_srs_proc = cell_ptr->getRUType();
                            pdctx->set_ru_type_for_srs_proc(ru_type_for_srs_proc);
                            ru_type_found = true;
                        }

                        if(cell_ptr->isActive() == false)
                        {
                            NVLOGW_FMT(TAG, "Cell {} is not active, can't run anything", cell_ptr->getPhyId());
                            cell_index++;
                            continue;
                        }

                        // Assume homogeneous cells
                        mu = cell_ptr->getMu();

                        if(ch == slot_command_api::PUSCH ||ch == slot_command_api::PUCCH)
                        {
                            ulbuf_st1 = nullptr;
                            ulbuf_st2 = nullptr;
                            cell_ptr->setPuschDynPrmIndex(sc->cell_groups.slot.slot_3gpp.slot_, -1);
                            // ulbuf_st3_v.clear();
                            rach_occasion = 0;
                            ulbuf_st1 = cell_ptr->getNextUlBufferST1();
                            if(ulbuf_st1 == nullptr)
                            {
                                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "SFN {}.{} No available UL input buffers st1 for cell {}", sc->cell_groups.slot.slot_3gpp.sfn_, sc->cell_groups.slot.slot_3gpp.slot_, cell_ptr->getPhyId());
                                isUlDlBufAvail=false;
                                goto cleanup_err;
                            }
                            else
                                NVLOGI_FMT(TAG, "SFN {}.{} Map {} Cell {} with MU {} ULBuffer ST1 {} at {} ch = PUSCH|PUCCH", static_cast<unsigned>(sc->cell_groups.slot.slot_3gpp.sfn_), static_cast<unsigned>(sc->cell_groups.slot.slot_3gpp.slot_), slot_map_ul->getId(), cell_ptr->getPhyId(), mu, ulbuf_st1->getId(), Time::nowNs().count());

                            /* Is the same cell there also for PRACH? */
                            if(current_slot_params_aggr->cgcmd->prach) {
                                auto findcell = std::find(std::begin(current_slot_params_aggr->cgcmd->prach->phy_cell_index_list), std::end(current_slot_params_aggr->cgcmd->prach->phy_cell_index_list), (int) cell_phy_id);
                                if (findcell != std::end(current_slot_params_aggr->cgcmd->prach->phy_cell_index_list)) {
                                    // ulbuf_st3_v.clear();
                                    rach_occasion = 0;
                                   // if(current_slot_params_aggr->cgcmd->prach->rach.size() > PRACH_MAX_OCCASIONS)
                                    if(current_slot_params_aggr->cgcmd->prach->nOccasion > PRACH_MAX_OCCASIONS)
    				                {
                                        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "SFN {}.{} Too many RACH objects (occasions) {} for this cell {}", sc->cell_groups.slot.slot_3gpp.sfn_, sc->cell_groups.slot.slot_3gpp.slot_, current_slot_params_aggr->cgcmd->prach->nOccasion, cell_ptr->getPhyId());
                                        goto cleanup_err;
                                    }

                                    rach_occasion = cell_ptr->getPrachOccaSize();
                                    //NVLOGI_FMT(TAG,"NUmber of RO per cell = {}", rach_occasion);

                                    //Create UL buffer entries for each rach occasion
                                    for(int ro = 0; ro < rach_occasion; ro++)
                                    {
                                        ULInputBuffer * ulbuf_st3 = cell_ptr->getNextUlBufferST3();
                                        if(ulbuf_st3 == nullptr)
                                        {
                                            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "SFN {}.{} No available UL input buffers st3 for cell {}", sc->cell_groups.slot.slot_3gpp.sfn_, sc->cell_groups.slot.slot_3gpp.slot_, cell_ptr->getPhyId());
                                            isUlDlBufAvail=false;
                                            goto cleanup_err;
                                        }
                                        else
                                            NVLOGI_FMT(TAG, "SFN {}.{} Map {} Cell {} with MU {} ULBuffer ST3 {} at {} ch = PRACH", static_cast<unsigned>(sc->cell_groups.slot.slot_3gpp.sfn_), static_cast<unsigned>(sc->cell_groups.slot.slot_3gpp.slot_), slot_map_ul->getId(), cell_ptr->getPhyId(), mu, ulbuf_st3->getId(), Time::nowNs().count());

                                        // ulbuf_st3_v.push_back(ulbuf_st3);
                                        ulbuf_st3_v[ro] = ulbuf_st3;
                                    }
                                }
                            }
                            if( current_slot_params_aggr->cgcmd->srs) {
                                auto findcell = std::find(std::begin(current_slot_params_aggr->cgcmd->srs->phy_cell_index_list), std::end(current_slot_params_aggr->cgcmd->srs->phy_cell_index_list), (int) cell_phy_id);
                                if (findcell != std::end(current_slot_params_aggr->cgcmd->srs->phy_cell_index_list)) {
                                    ulbuf_st2 = nullptr;
                                    ulbuf_st2 = cell_ptr->getNextUlBufferST2();
                                    if(ulbuf_st2 == nullptr)
                                    {
                                        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "SFN {}.{} No available UL input buffers st2 for cell {}", sc->cell_groups.slot.slot_3gpp.sfn_, sc->cell_groups.slot.slot_3gpp.slot_, cell_ptr->getPhyId());
                                        isUlDlBufAvail=false;
                                        goto cleanup_err;
                                    }
                                    else
                                        NVLOGI_FMT(TAG, "SFN {}.{} Map {} Cell {} with MU {} ULBuffer ST2 {} at {} ch = SRS", static_cast<unsigned>(sc->cell_groups.slot.slot_3gpp.sfn_), static_cast<unsigned>(sc->cell_groups.slot.slot_3gpp.slot_), slot_map_ul->getId(), cell_ptr->getPhyId(), mu, ulbuf_st2->getId(), Time::nowNs().count());
                                    }
                            }
                        }
                        else if(ch == slot_command_api::PRACH)
                        {
                            // ulbuf_st3_v.clear();
                            rach_occasion = 0;
                            //if(current_slot_params_aggr->cgcmd->prach->rach.size() > PRACH_MAX_OCCASIONS)
                            if(current_slot_params_aggr->cgcmd->prach->nOccasion > PRACH_MAX_OCCASIONS)
                            {
                                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "SFN {}.{} Too many RACH objects (occasions) {} for this cell {}", sc->cell_groups.slot.slot_3gpp.sfn_, sc->cell_groups.slot.slot_3gpp.slot_, current_slot_params_aggr->cgcmd->prach->nOccasion, cell_ptr->getPhyId());
                                goto cleanup_err;
                            }

                            rach_occasion = cell_ptr->getPrachOccaSize();
                            //NVLOGC_FMT(TAG,"NUmber of RO per cell  {} with PHY ID {} = {}", cell_index, cell_phy_id, rach_occasion);

                            //Create UL buffer entries for each rach occasion
                            for(int ro = 0; ro < rach_occasion; ro++)
                            {
                                ULInputBuffer * ulbuf_st3 = cell_ptr->getNextUlBufferST3();
                                if(ulbuf_st3 == nullptr)
                                {
                                    NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "SFN {}.{} No available UL input buffers st3 for cell {}", sc->cell_groups.slot.slot_3gpp.sfn_, sc->cell_groups.slot.slot_3gpp.slot_, cell_ptr->getPhyId());
                                    isUlDlBufAvail=false;
                                    goto cleanup_err;
                                }
                                else
                                    NVLOGI_FMT(TAG, "SFN {}.{} Map {} Cell {} with MU {} ULBuffer ST3 {} at {} ch = PRACH", static_cast<unsigned>(sc->cell_groups.slot.slot_3gpp.sfn_), static_cast<unsigned>(sc->cell_groups.slot.slot_3gpp.slot_), slot_map_ul->getId(), cell_ptr->getPhyId(), mu, ulbuf_st3->getId(), Time::nowNs().count());

                                // ulbuf_st3_v.push_back(ulbuf_st3);
                                ulbuf_st3_v[ro] = ulbuf_st3;
                            }

                            bool is_pusch = false;
                            bool is_pucch = false;
                            bool is_srs = false;
                            ulbuf_st1 = nullptr;
                            ulbuf_st2 = nullptr;

                            /* Is the same cell there also for PUSCH? */
                            if(current_slot_params_aggr->cgcmd->pusch) {
                                auto findcell = std::find(std::begin(current_slot_params_aggr->cgcmd->pusch->phy_cell_index_list), std::end(current_slot_params_aggr->cgcmd->pusch->phy_cell_index_list), (int) cell_phy_id);
                                if (findcell != std::end(current_slot_params_aggr->cgcmd->pusch->phy_cell_index_list)) {
                                    is_pusch = true;
                                }
                            }

                            /* Is the same cell there also for PUSCH? */
                            if(is_pusch == false && current_slot_params_aggr->cgcmd->pucch) {
                                auto findcell = std::find(std::begin(current_slot_params_aggr->cgcmd->pucch->phy_cell_index_list), std::end(current_slot_params_aggr->cgcmd->pucch->phy_cell_index_list), (int) cell_phy_id);
                                if (findcell != std::end(current_slot_params_aggr->cgcmd->pucch->phy_cell_index_list)) {
                                    is_pucch = true;
                                }
                            }

                            if(current_slot_params_aggr->cgcmd->srs) {
                                auto findcell = std::find(std::begin(current_slot_params_aggr->cgcmd->srs->phy_cell_index_list), std::end(current_slot_params_aggr->cgcmd->srs->phy_cell_index_list), (int) cell_phy_id);
                                if (findcell != std::end(current_slot_params_aggr->cgcmd->srs->phy_cell_index_list)) {
                                    is_srs = true;
                                }
                            }
                            /* Get ULBuffer ST1 */
                            if(is_pusch == true || is_pucch == true ) {
                                ulbuf_st1 = nullptr;
                                ulbuf_st1 = cell_ptr->getNextUlBufferST1();
                                if(ulbuf_st1 == nullptr)
                                {
                                    NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "SFN {}.{} No available UL input buffers st1 for cell {}", sc->cell_groups.slot.slot_3gpp.sfn_, sc->cell_groups.slot.slot_3gpp.slot_, cell_ptr->getPhyId());
                                    isUlDlBufAvail=false;
                                    goto cleanup_err;
                                }
                                else
                                    NVLOGI_FMT(TAG, "SFN {}.{} Map {} Cell {} with MU {} ULBuffer ST1 {} at {} ch = PUSCH|PUCCH", static_cast<unsigned>(sc->cell_groups.slot.slot_3gpp.sfn_), static_cast<unsigned>(sc->cell_groups.slot.slot_3gpp.slot_), slot_map_ul->getId(), cell_ptr->getPhyId(), mu, ulbuf_st1->getId(), Time::nowNs().count());
                            }

                            if(is_srs == true){
                                ulbuf_st2 = nullptr;
                                ulbuf_st2 = cell_ptr->getNextUlBufferST2();
                                if(ulbuf_st2 == nullptr)
                                {
                                    NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "SFN {}.{} No available UL input buffers st2 for cell {}", sc->cell_groups.slot.slot_3gpp.sfn_, sc->cell_groups.slot.slot_3gpp.slot_, cell_ptr->getPhyId());
                                    isUlDlBufAvail=false;
                                    goto cleanup_err;
                                }
                                else
                                    NVLOGI_FMT(TAG, "SFN {}.{} Map {} Cell {} with MU {} ULBuffer ST2 {} at {} ch = SRS", static_cast<unsigned>(sc->cell_groups.slot.slot_3gpp.sfn_), static_cast<unsigned>(sc->cell_groups.slot.slot_3gpp.slot_), slot_map_ul->getId(), cell_ptr->getPhyId(), mu, ulbuf_st2->getId(), Time::nowNs().count());
                            }
                        }
                        else if (ch == slot_command_api::SRS){
                            ulbuf_st2 = nullptr;
                            ulbuf_st2 = cell_ptr->getNextUlBufferST2();
                            if(ulbuf_st2 == nullptr)
                            {
                                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "SFN {}.{} No available UL input buffers st2 for cell {}", sc->cell_groups.slot.slot_3gpp.sfn_, sc->cell_groups.slot.slot_3gpp.slot_, cell_ptr->getPhyId());
                                isUlDlBufAvail=false;
                                goto cleanup_err;
                            }
                            else
                                NVLOGI_FMT(TAG, "SFN {}.{} Map {} Cell {} with MU {} ULBuffer ST2 {} at {} ch = SRS", static_cast<unsigned>(sc->cell_groups.slot.slot_3gpp.sfn_), static_cast<unsigned>(sc->cell_groups.slot.slot_3gpp.slot_), slot_map_ul->getId(), cell_ptr->getPhyId(), mu, ulbuf_st2->getId(), Time::nowNs().count());
                            /* Is the same cell there also for PRACH? */
                            if(current_slot_params_aggr->cgcmd->prach) {
                                auto findcell = std::find(std::begin(current_slot_params_aggr->cgcmd->prach->phy_cell_index_list), std::end(current_slot_params_aggr->cgcmd->prach->phy_cell_index_list), (int) cell_phy_id);
                                if (findcell != std::end(current_slot_params_aggr->cgcmd->prach->phy_cell_index_list)) {
                                    // ulbuf_st3_v.clear();
                                    rach_occasion = 0;
                                   // if(current_slot_params_aggr->cgcmd->prach->rach.size() > PRACH_MAX_OCCASIONS)
                                    if(current_slot_params_aggr->cgcmd->prach->nOccasion > PRACH_MAX_OCCASIONS)
                                    {
                                        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "SFN {}.{} Too many RACH objects (occasions) {} for this cell {}", sc->cell_groups.slot.slot_3gpp.sfn_, sc->cell_groups.slot.slot_3gpp.slot_, current_slot_params_aggr->cgcmd->prach->nOccasion, cell_ptr->getPhyId());
                                        goto cleanup_err;
                                    }

                                    rach_occasion = cell_ptr->getPrachOccaSize();
                                    //NVLOGI_FMT(TAG,"NUmber of RO per cell = {}", rach_occasion);

                                    //Create UL buffer entries for each rach occasion
                                    for(int ro = 0; ro < rach_occasion; ro++)
                                    {
                                        ULInputBuffer * ulbuf_st3 = cell_ptr->getNextUlBufferST3();
                                        if(ulbuf_st3 == nullptr)
                                        {
                                            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "SFN {}.{} No available UL input buffers st3 for cell {}", sc->cell_groups.slot.slot_3gpp.sfn_, sc->cell_groups.slot.slot_3gpp.slot_, cell_ptr->getPhyId());
                                            isUlDlBufAvail=false;
                                            goto cleanup_err;
                                        }
                                        else
                                            NVLOGI_FMT(TAG, "SFN {}.{} Map {} Cell {} with MU {} ULBuffer ST3 {} at {} ch = PRACH", static_cast<unsigned>(sc->cell_groups.slot.slot_3gpp.sfn_), static_cast<unsigned>(sc->cell_groups.slot.slot_3gpp.slot_), slot_map_ul->getId(), cell_ptr->getPhyId(), mu, ulbuf_st3->getId(), Time::nowNs().count());

                                        // ulbuf_st3_v.push_back(ulbuf_st3);
                                        ulbuf_st3_v[ro] = ulbuf_st3;
                                    }
                                }
                            }

                            bool is_pusch = false;
                            bool is_pucch = false;
                            ulbuf_st1 = nullptr;
                            /* Is the same cell there also for PUSCH? */
                            if(current_slot_params_aggr->cgcmd->pusch) {
                                auto findcell = std::find(std::begin(current_slot_params_aggr->cgcmd->pusch->phy_cell_index_list), std::end(current_slot_params_aggr->cgcmd->pusch->phy_cell_index_list), (int) cell_phy_id);
                                if (findcell != std::end(current_slot_params_aggr->cgcmd->pusch->phy_cell_index_list)) {
                                    is_pusch = true;
                                }
                            }

                            /* Is the same cell there also for PUSCH? */
                            if(is_pusch == false && current_slot_params_aggr->cgcmd->pucch) {
                                auto findcell = std::find(std::begin(current_slot_params_aggr->cgcmd->pucch->phy_cell_index_list), std::end(current_slot_params_aggr->cgcmd->pucch->phy_cell_index_list), (int) cell_phy_id);
                                if (findcell != std::end(current_slot_params_aggr->cgcmd->pucch->phy_cell_index_list)) {
                                    is_pucch = true;
                                }
                            }
                            /* Get ULBuffer ST1 */
                            if(is_pusch == true || is_pucch == true ) {
                                ulbuf_st1 = nullptr;
                                ulbuf_st1 = cell_ptr->getNextUlBufferST1();
                                if(ulbuf_st1 == nullptr)
                                {
                                    NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "SFN {}.{} No available UL input buffers st1 for cell {}", sc->cell_groups.slot.slot_3gpp.sfn_, sc->cell_groups.slot.slot_3gpp.slot_, cell_ptr->getPhyId());
                                    isUlDlBufAvail=false;
                                    goto cleanup_err;
                                }
                                else
                                    NVLOGI_FMT(TAG, "SFN {}.{} Map {} Cell {} with MU {} ULBuffer ST1 {} at {} ch = PUSCH|PUCCH", static_cast<unsigned>(sc->cell_groups.slot.slot_3gpp.sfn_), static_cast<unsigned>(sc->cell_groups.slot.slot_3gpp.slot_), slot_map_ul->getId(), cell_ptr->getPhyId(), mu, ulbuf_st1->getId(), Time::nowNs().count());
                            }
                        }

                        ulbuf_pcap_capture = nullptr;
                        ulbuf_pcap_capture_ts = nullptr;
                        if(pdctx->get_ul_pcap_capture_enable())
                        {
                            ulbuf_pcap_capture = cell_ptr->getUlBufferPcap();
                            ulbuf_pcap_capture_ts = cell_ptr->getUlBufferPcapTs();
                        }
                        //FIXME: When PUSCH + PDSCH, should move cells params at the beginning into one place, not cell by cell per channel
                        if(slot_map_ul->aggrSetCells(cell_ptr, &sc->cells[(*cell_index_list)[cell_index]].params,
                                                    ulbuf_st1,ulbuf_st2,ulbuf_st3_v, rach_occasion, ulbuf_pcap_capture, ulbuf_pcap_capture_ts))
                        {
                            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "SlotMap UL can't set another cell");
                            goto cleanup_err;
                        }


                        // Avoid to waste UL thread time starting it too early
                        if(max_ul_uc_delay < cell_ptr->getT1aMaxCpUlNs())
                            max_ul_uc_delay = cell_ptr->getT1aMaxCpUlNs();
                        cell_ul_list[cell_ul_list_idx] = cell_phy_id;
                        cell_ul_list_idx++;
                        cell_index++;
                    }
                    //Single order kernel entity per slot
                    if(!order_entity_set)
                    {
                        oentity_ptr = pdctx->getNextOrderEntity(cell_ul_list,cell_ul_list_idx,nullptr,true);
                        if(oentity_ptr == nullptr)
                        {
                            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "No available Order Kernel object for SFN {}.{} Map {} ", sc->cell_groups.slot.slot_3gpp.sfn_, sc->cell_groups.slot.slot_3gpp.slot_, slot_map_ul->getId());
                            isOKobjAvail=false;
                            goto cleanup_err;
                        }
                        else
                        {
                            NVLOGI_FMT(TAG, "SFN {}.{} Map {} Oentity {} at {} for ch({})", sc->cell_groups.slot.slot_3gpp.sfn_, sc->cell_groups.slot.slot_3gpp.slot_, slot_map_ul->getId(),oentity_ptr->getId(), Time::nowNs().count(),(int)ch);
                            slot_map_ul->aggrSetOrderEntity(oentity_ptr);
                            order_entity_set=true;
                        }
                    }
                    else
                    {
                        oentity_ptr = pdctx->getNextOrderEntity(cell_ul_list,cell_ul_list_idx, slot_map_ul->aggrGetOrderEntity(),false);
                        if(oentity_ptr!=slot_map_ul->aggrGetOrderEntity())
                        {
                            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Order entity cannot be different for the same slot");
                            goto cleanup_err;
                        }
                    }
                }
            }

            /* Uplink */
            if(ch == slot_command_api::BFW)
            {
                if(current_slot_params_aggr->cgcmd->get_bfw_params()->bfw_cvi_type==slot_command_api::UL_BFW)
                {
                    aggr_ulbfw_ptr = pdctx->getNextUlBfwAggr(current_slot_params_aggr);
                    if(aggr_ulbfw_ptr == nullptr)
                    {
                        NVLOGI_FMT(TAG, "No available Aggr ULBFW objects");
                        goto cleanup_err;
                    }
                    else
                        NVLOGI_FMT(TAG, "SFN {}.{} Map {} with MU {} got ULBFW Aggr obj {:x} at {}",
                                sc->cell_groups.slot.slot_3gpp.sfn_, sc->cell_groups.slot.slot_3gpp.slot_, slot_map_ul->getId(),
                                mu, aggr_ulbfw_ptr->getId(), Time::nowNs().count()
                            );
                    ul_task_count+=2; //ULBFW + AGGR3_ULBFW
                }
            }

            if(ch == slot_command_api::PUSCH || ch == slot_command_api::PUCCH) {
                if(!pucch_or_pusch_found) {
                    pucch_or_pusch_found = true;
                    ul_task_count += 1;
                }
            }

            if(ch == slot_command_api::PUSCH)
            {
                aggr_pusch_ptr = pdctx->getNextPuschAggr(current_slot_params_aggr);
                if(aggr_pusch_ptr == nullptr)
                {
                    NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "SFN {}.{} No available Aggr PUSCH objects", sc->cell_groups.slot.slot_3gpp.sfn_, sc->cell_groups.slot.slot_3gpp.slot_);
                    isAggrObjAvail=false;
                    goto cleanup_err;
                }
                else
                    NVLOGI_FMT(TAG, "SFN {}.{} Map {} with MU {} got PUSCH Aggr obj {:x} at {}",
                            sc->cell_groups.slot.slot_3gpp.sfn_, sc->cell_groups.slot.slot_3gpp.slot_, slot_map_ul->getId(),
                            mu, aggr_pusch_ptr->getId(), Time::nowNs().count()
                        );
                ul_task_count+=1; //Early UCI IND Task
            }

            if(ch == slot_command_api::PUCCH)
            {
                aggr_pucch_ptr = pdctx->getNextPucchAggr(current_slot_params_aggr);
                if(aggr_pucch_ptr == nullptr)
                {
                    NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "SFN {}.{} No available Aggr PUCCH objects", sc->cell_groups.slot.slot_3gpp.sfn_, sc->cell_groups.slot.slot_3gpp.slot_);
                    isAggrObjAvail=false;
                    goto cleanup_err;
                }
                else
                    NVLOGI_FMT(TAG, "SFN {}.{} Map {} with MU {} got PUCCH Aggr obj {:x} at {}",
                            sc->cell_groups.slot.slot_3gpp.sfn_, sc->cell_groups.slot.slot_3gpp.slot_, slot_map_ul->getId(),
                            mu, aggr_pucch_ptr->getId(), Time::nowNs().count()
                        );
            }

            if(ch == slot_command_api::PRACH)
            {
                aggr_prach_ptr = pdctx->getNextPrachAggr(current_slot_params_aggr);
                if(aggr_prach_ptr == nullptr)
                {
                    NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "SFN {}.{} No available Aggr PRACH objects", sc->cell_groups.slot.slot_3gpp.sfn_, sc->cell_groups.slot.slot_3gpp.slot_);
                    isAggrObjAvail=false;
                    goto cleanup_err;
                }
                else
                    NVLOGI_FMT(TAG, "SFN {}.{} Map {} with MU {} got PRACH Aggr obj {:x} at {}",
                            sc->cell_groups.slot.slot_3gpp.sfn_, sc->cell_groups.slot.slot_3gpp.slot_, slot_map_ul->getId(),
                            mu, aggr_prach_ptr->getId(), Time::nowNs().count()
                        );
                ul_task_count++;
            }

            if(ch == slot_command_api::SRS)
            {
                aggr_srs_ptr = pdctx->getNextSrsAggr(current_slot_params_aggr);
                if(aggr_srs_ptr == nullptr)
                {
                    NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "SFN {}.{} No available Aggr SRS objects", sc->cell_groups.slot.slot_3gpp.sfn_, sc->cell_groups.slot.slot_3gpp.slot_);
                    isAggrObjAvail=false;
                    goto cleanup_err;
                }
                else
                    NVLOGI_FMT(TAG, "SFN {}.{} Map {} with MU {} got SRS Aggr obj {:x} at {}",
                            sc->cell_groups.slot.slot_3gpp.sfn_, sc->cell_groups.slot.slot_3gpp.slot_, slot_map_ul->getId(),
                            mu, aggr_srs_ptr->getId(), Time::nowNs().count()
                        );
                if(pdctx->get_ru_type_for_srs_proc() == SINGLE_SECT_MODE)
                {
                    ul_task_count+=1; //SRS only
                }
                else
                {
                    ul_task_count+=2; //SRS + AGGR3_SRS
                }
            }

            /* Downlink */

            if(ch == slot_command_api::BFW)
            {
                if(current_slot_params_aggr->cgcmd->get_bfw_params()->bfw_cvi_type==slot_command_api::DL_BFW)
                {
                    aggr_dlbfw_ptr = pdctx->getNextDlBfwAggr(current_slot_params_aggr);
                    if(aggr_dlbfw_ptr == nullptr)
                    {
                        NVLOGI_FMT(TAG, "No available Aggr DLBFW objects");
                        goto cleanup_err;
                    }
                    else
                        NVLOGI_FMT(TAG, "SFN {}.{} Map {} with MU {} got DLBFW Aggr obj {:x} at {}",
                                sc->cell_groups.slot.slot_3gpp.sfn_, sc->cell_groups.slot.slot_3gpp.slot_, slot_map_dl->getId(),
                                mu, aggr_dlbfw_ptr->getId(), Time::nowNs().count()
                            );
                    dl_task_count++;
                }
            }
            else if(ch == slot_command_api::PDSCH)
            {
                aggr_pdsch_ptr = pdctx->getNextPdschAggr(current_slot_params_aggr);
                if(aggr_pdsch_ptr == nullptr)
                {
                    NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "SFN {}.{} No available Aggr PDSCH objects", sc->cell_groups.slot.slot_3gpp.sfn_, sc->cell_groups.slot.slot_3gpp.slot_);
                    isAggrObjAvail=false;
                    goto cleanup_err;
                }
                else
                    NVLOGI_FMT(TAG, "SFN {}.{} Map {} with MU {} got PDSCH Aggr obj {:x} at {}",
                            sc->cell_groups.slot.slot_3gpp.sfn_, sc->cell_groups.slot.slot_3gpp.slot_, slot_map_dl->getId(),
                            mu, aggr_pdsch_ptr->getId(), Time::nowNs().count()
                        );
                dl_task_count++;
                if(pdctx->enable_prepone_h2d_cpy)
                {
                    if(!pdctx->h2d_copy_thread_enable)
                    {
                        pdctx->getPdschMpsCtx()->setCtx();

                        CUDA_CHECK(cudaEventRecord(pdctx->get_event_pdsch_tb_cpy_start(sc->cell_groups.slot.slot_3gpp.slot_), pdctx->getH2DCpyStream()));

                        // If there is no seperate copy thread, launch the copy here. The information about the copy batches is filled in in the offload funciton.
                        cuphyStatus_t batched_memcpy_status = pdctx->performBatchedMemcpy(); // still a no-op if yaml flag is 0
                        if (batched_memcpy_status != CUPHY_STATUS_SUCCESS)
                        {
                            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Error! cuMemcpyBatchAsync returned {}!", batched_memcpy_status);
                        }
                        pdctx->resetBatchedMemcpyBatches(); // reset batches count to 0

                        CUDA_CHECK(cudaEventRecord(pdctx->get_event_pdsch_tb_cpy_complete(sc->cell_groups.slot.slot_3gpp.slot_), pdctx->getH2DCpyStream()));
                    }
                    else
                    {
                        l1_set_h2d_copy_done_cur_slot_flag(pdh,(int)sc->cell_groups.slot.slot_3gpp.slot_); //Set h2d copy done for current slot flag
                    }
                }
            }
            else if(ch == slot_command_api::PDCCH_DL)
            {
                aggr_pdcch_dl_ptr = pdctx->getNextPdcchDlAggr(current_slot_params_aggr);
                if(aggr_pdcch_dl_ptr == nullptr)
                {
                    NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "SFN {}.{} No available Aggr PDCCH DL objects", sc->cell_groups.slot.slot_3gpp.sfn_, sc->cell_groups.slot.slot_3gpp.slot_);
                    isAggrObjAvail=false;
                    goto cleanup_err;
                }
                else
                    NVLOGI_FMT(TAG, "SFN {}.{} Map {} with MU {} got Aggr PDCCH DL obj {:x} at {}",
                            sc->cell_groups.slot.slot_3gpp.sfn_, sc->cell_groups.slot.slot_3gpp.slot_, slot_map_dl->getId(),
                            mu, aggr_pdcch_dl_ptr->getId(), Time::nowNs().count()
                        );
            }
            else if(ch == slot_command_api::PDCCH_UL)
            {
                aggr_pdcch_ul_ptr = pdctx->getNextPdcchUlAggr(current_slot_params_aggr);
                if(aggr_pdcch_ul_ptr == nullptr)
                {
                    NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "SFN {}.{} No available Aggr PDCCH UL objects", sc->cell_groups.slot.slot_3gpp.sfn_, sc->cell_groups.slot.slot_3gpp.slot_);
                    isAggrObjAvail=false;
                    goto cleanup_err;
                }
                else
                    NVLOGI_FMT(TAG, "SFN {}.{} Map {} with MU {} got Aggr PDCCH UL obj {:x} at {}",
                            sc->cell_groups.slot.slot_3gpp.sfn_, sc->cell_groups.slot.slot_3gpp.slot_, slot_map_dl->getId(),
                            mu, aggr_pdcch_ul_ptr->getId(), Time::nowNs().count()
                        );
            }
            else if(ch == slot_command_api::PBCH)
            {
                aggr_pbch_ptr = pdctx->getNextPbchAggr(current_slot_params_aggr);
                if(aggr_pbch_ptr == nullptr)
                {
                    NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "SFN {}.{} No available Aggr PBCH objects", sc->cell_groups.slot.slot_3gpp.sfn_, sc->cell_groups.slot.slot_3gpp.slot_);
                    isAggrObjAvail=false;
                    goto cleanup_err;
                }
                else
                    NVLOGI_FMT(TAG, "SFN {}.{} Map {} with MU {} got Aggr PBCH obj {:x} at {}",
                            sc->cell_groups.slot.slot_3gpp.sfn_, sc->cell_groups.slot.slot_3gpp.slot_, slot_map_dl->getId(),
                            mu, aggr_pbch_ptr->getId(), Time::nowNs().count()
                        );
            }
            else if(ch == slot_command_api::CSI_RS)
            {
                aggr_csirs_ptr = pdctx->getNextCsiRsAggr(current_slot_params_aggr);
                if(aggr_csirs_ptr == nullptr)
                {
                    NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "SFN {}.{} No available Aggr CSI_RS objects", sc->cell_groups.slot.slot_3gpp.sfn_, sc->cell_groups.slot.slot_3gpp.slot_);
                    isAggrObjAvail=false;
                    goto cleanup_err;
                }
                else
                    NVLOGI_FMT(TAG, "SFN {}.{} Map {} with MU {} got Aggr CSI_RS obj {:x} at {}",
                            sc->cell_groups.slot.slot_3gpp.sfn_, sc->cell_groups.slot.slot_3gpp.slot_, slot_map_dl->getId(),
                            mu, aggr_csirs_ptr->getId(), Time::nowNs().count()
                        );
            }

        }

        if (aggr_pdcch_dl_ptr || aggr_pdcch_ul_ptr || aggr_pbch_ptr || aggr_csirs_ptr) {
            dl_task_count++;
        }
    }

    task_ts_exec[0] = sc->tick_original;

    ////////////////////////////////////////////////////////////////////////
    //// CreateMap + CreateTask + EnqueueTask()
    ////////////////////////////////////////////////////////////////////////

    if(slot_map_ul != nullptr)
    {
        auto num_cells = slot_map_ul->getNumCells();
        int num_ul_workers = pdctx->getNumULWorkers();
        int num_ulc_tasks = get_num_ulc_tasks(num_ul_workers);
        bool launch_second_order_kernel_task = false;
        bool isSrsOnly = ((aggr_srs_ptr != nullptr) && (aggr_pusch_ptr==nullptr) && (aggr_pucch_ptr==nullptr) && (aggr_prach_ptr==nullptr));


        if(num_cells > 0 || aggr_ulbfw_ptr)
        {
            if(slot_map_ul->aggrSetPhy(aggr_pusch_ptr,aggr_pucch_ptr,aggr_prach_ptr, aggr_srs_ptr,aggr_ulbfw_ptr,current_slot_params_aggr))
            {
                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "SlotMapUL aggrSetPhy");
                goto cleanup_err;
            }
            //Check for Exit handler in flight and skip enqueing of UL tasks accordingly
            if(pExitHandler.test_exit_in_flight())
            {
                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "L1 Exit in flight during Slot Map {}",slot_map_ul->getId());
                goto cleanup_err;
            }

            t_ns map_ul_start = Time::nowNs();
            /*
             * Ref times for ops within the task.
             */
            PUSH_RANGE_PHYDRV("SLOT_UL", 4);
            num_cells = slot_map_ul->getNumCells();

            if(en_orderKernel_tb)
            {
                ul_task_count=2+num_ulc_tasks;//Cplane+Order kernel+UL3
                // task_index == 0..num_ulc_tasks-1: CPlane - launch at tick + 1 slot
                for(task_index=0;task_index<(num_ulc_tasks);task_index++)
                {
                    task_ts_exec[task_index]= sc->tick_original + ((1) * t_ns(Cell::getTtiNsFromMu(MU_SUPPORTED)));
                }
                // Order kernel
                task_ts_exec[task_index]= t_ns(t0_slot - UL_TASK1_ORDER_LAUNCH_OFFSET_FROM_T0_NS);
                task_index++;
            }
            else
            {
                if(pdctx->cpuCommEnabled())
                    ul_task_count+=3+num_ulc_tasks;//Cplane+Order kernel+UL2+UL3
                else
                    ul_task_count+=2+num_ulc_tasks;//Cplane+Order kernel+UL3
                    
                // task_index == 0..num_ulc_tasks-1: CPlane - launch at tick + 1 slot
                for(task_index=0;task_index<(num_ulc_tasks);task_index++)
                {
                    task_ts_exec[task_index]= sc->tick_original + ((1) * t_ns(Cell::getTtiNsFromMu(MU_SUPPORTED)));
                }

                if(aggr_srs_ptr && (ru_type_for_srs_proc != SINGLE_SECT_MODE))
                {
                    if(isSrsOnly)
                    {
                        // Launch only SRS UL Order kernel
                        if(pdctx->gpuCommEnabledViaCpu())
                        {
                            task_ts_exec[task_index]= t_ns(t0_slot + UL_TASK1_SRS_ORDER_LAUNCH_OFFSET_FROM_T0_NS);
                        }
                        else
                        {
                            task_ts_exec[task_index]= t_ns(t0_slot + UL_TASK1_ORDER_LAUNCH_OFFSET_FROM_T0_NS);
                        }
                        task_index++;
                    }
                    else
                    {
                        // Launch both SRS and non-SRS UL Order kernel
                        task_ts_exec[task_index]= t_ns(t0_slot - UL_TASK1_ORDER_LAUNCH_OFFSET_FROM_T0_NS);
                        task_index++;
                        if(pdctx->gpuCommEnabledViaCpu())
                        {
                            task_ts_exec[task_index]= t_ns(t0_slot + UL_TASK1_SRS_ORDER_LAUNCH_OFFSET_FROM_T0_NS);
                        }
                        else
                        {
                            task_ts_exec[task_index]= t_ns(t0_slot + UL_TASK1_ORDER_LAUNCH_OFFSET_FROM_T0_NS);
                        }
                        task_index++;
                        ul_task_count+=1;      
                        launch_second_order_kernel_task = true;
                    }
                }
                else
                {
                    // Launch only non-SRS UL Order kernel, or combined if ru_type::SINGLE_SECT_MODE
                    task_ts_exec[task_index]= t_ns(t0_slot - UL_TASK1_ORDER_LAUNCH_OFFSET_FROM_T0_NS);
                    task_index++;
                }

                if(aggr_pucch_ptr || aggr_pusch_ptr)
                {
                    //PUCCH and PUSCH task
                    task_ts_exec[task_index]= t_ns(t0_slot - UL_TASK1_PUCCH_LAUNCH_OFFSET_FROM_T0_NS);
                    task_index++;
                }

                if(pdctx->cpuCommEnabled())
                {
                    task_ts_exec[task_index]= t_ns(t0_slot + UL_TASK2_OFFSET_FROM_T0_NS);
                    task_index++;                
                }

                if(aggr_pusch_ptr)
                {
                    //Early UCI task
                    task_ts_exec[task_index]= t_ns(t0_slot + UL_TASK3_EARLY_UCI_IND_TASK_LAUNCH_OFFSET_FROM_T0_NS);
                    task_index++;
                }
                if(aggr_ulbfw_ptr) 
                {
                    task_ts_exec[task_index]= sc->tick_original; //Immediate action time for BFW
                    task_index++;
                    task_ts_exec[task_index]= t_ns(t0_slot - UL_AGGR3_ULBFW_OFFSET_FROM_T0_NS); //Launch UL_AGGR3_ULBFW task at UL_AGGR3_ULBFW_OFFSET_FROM_T0_NS before T0
                    task_index++;                    
                }
                int task_index_loop_limit;

                if(ru_type_for_srs_proc == SINGLE_SECT_MODE) {
                    task_index_loop_limit = ul_task_count-1;
                } else {
                   // No SRS : task_index num_ulc_tasks+1..ul_task_count-2 : Channel tasks, launch at T0 slot boundary + 1
                   // With SRS : task_index num_ulc_tasks+1..ul_task_count-3 : Channel tasks, launch at T0 slot boundary + 1
                   task_index_loop_limit = (aggr_srs_ptr) ? ul_task_count-2 : ul_task_count-1;
                }
                for(task_index=task_index;task_index<(task_index_loop_limit);task_index++)
                {
                    //Set all UL channel tasks except PUSCH  to launch at the same time (T0+500us)
                    task_ts_exec[task_index]= t_ns(t0_slot + Cell::getTtiNsFromMu(MU_SUPPORTED));
                }
            }
            // TaskUL3Aggr - launch 1 slot after the channel tasks
            task_ts_exec[task_index] = t_ns(t0_slot + UL_TASK3_AGGR3_OFFSET_FROM_T0_NS);

            // TaskUL3AggrSrs - launch at Yaml configured offset from T0
            if(aggr_srs_ptr && !en_orderKernel_tb && ru_type_for_srs_proc != SINGLE_SECT_MODE) //FX RU requires awaiting for SRS completion in the same AGGR3 task as non-SRS
            {
                uint32_t task_offset_from_t0= (pdctx->getUlSrsAggr3TaskLaunchOffsetNs() > UL_TASK3_AGGR3_MAX_BACKOFF_FROM_SRS_COMPLETION_TH_NS) ? UL_TASK3_AGGR3_MAX_BACKOFF_FROM_SRS_COMPLETION_TH_NS : pdctx->getUlSrsAggr3TaskLaunchOffsetNs();
                task_offset_from_t0 = SRS_COMPLETION_TH_FROM_T0_NS - task_offset_from_t0;
                task_ts_exec[task_index+1] = t_ns(t0_slot + task_offset_from_t0);
                task_index++;
            }
            slot_map_ul->setTasksTs(ul_task_count, task_ts_exec, Time::nowNs());
            task_index = 0;

            ///////////////////////////////////////////////////////////////////////
            //// Task1: C-plane & CUDA tasks
            ///////////////////////////////////////////////////////////////////////

            first_cell = 0;
            {
                if(task_index>=ul_task_count)
                {
                    NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Task index exceeds UL task count");
                    goto cleanup_err;
                }
                else
                {
                    for (int c = 0; c < num_ulc_tasks; c++) {
                        task_ul_ptr_list[task_index] = pdctx->getNextTask();
                        char buf[32];
                        sprintf(buf, "TaskUL1AggrCplane%d", c + 1);
                        task_ul_ptr_list[task_index]->init(task_ts_exec[task_index] + (t_ns)task_index, buf, task_work_function_ul_aggr_1_cplane, static_cast<void*>(slot_map_ul),
                                                            c, first_cell, num_ulc_tasks, 0);
                        task_index++;
                    }
                }
                if(task_index>=ul_task_count)
                {
                    NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Task index exceeds UL task count");
                    goto cleanup_err;
                }
                else
                {
                    task_ul_ptr_list[task_index] = pdctx->getNextTask();
                    int o=1;
                    char buf[32];
                    sprintf(buf, "TaskUL1AggrOrderkernel%d", o);                    
                    task_ul_ptr_list[task_index]->init(task_ts_exec[task_index]+(t_ns)(task_index), buf, task_work_function_ul_aggr_1_orderKernel, static_cast<void*>(slot_map_ul),
                                                       first_cell,o,(isSrsOnly)?1:0, 0);
                    task_index++;
                    if(launch_second_order_kernel_task)
                    {
                        task_ul_ptr_list[task_index] = pdctx->getNextTask();
                        o++;
                        sprintf(buf, "TaskUL1AggrOrderkernel%d", o);                    
                        task_ul_ptr_list[task_index]->init(task_ts_exec[task_index]+(t_ns)(task_index), buf, task_work_function_ul_aggr_1_orderKernel, static_cast<void*>(slot_map_ul),
                                                           first_cell,o,1, 0);
                        task_index++;
                    }
                }
                if(!en_orderKernel_tb)
                {
                    if(aggr_pusch_ptr || aggr_pucch_ptr)
                    {
                        if(task_index>=ul_task_count)
                        {
                            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Task index exceeds UL task count");
                            goto cleanup_err;
                        }
                        else
                        {
                            task_ul_ptr_list[task_index] = pdctx->getNextTask();
                            task_ul_ptr_list[task_index]->init(task_ts_exec[task_index]+(t_ns)(task_index), "TaskUL1AggrPucchPusch", task_work_function_ul_aggr_1_pucch_pusch, static_cast<void*>(slot_map_ul),
                                                            first_cell, num_cells, ul_task_count, 0);
                            task_index++;
                        }
                    }
                    if(pdctx->cpuCommEnabled())
                    {
                        if(task_index>=ul_task_count)
                        {
                            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Task index exceeds UL task count");
                            goto cleanup_err;
                        }
                        else
                        {
                            task_ul_ptr_list[task_index] = pdctx->getNextTask();
                            task_ul_ptr_list[task_index]->init(task_ts_exec[task_index]+(t_ns)(task_index), "TaskUL2Aggr", task_work_function_ul_aggr_2, static_cast<void*>(slot_map_ul),
                                                            first_cell, num_cells, ul_task_count, 0);
                            task_index++;
                        }                
                    }
                    if(aggr_pusch_ptr) {
                        if(task_index>=ul_task_count)
                        {
                            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Task index exceeds UL task count");
                            goto cleanup_err;
                        }
                        else
                        {
                            task_ul_ptr_list[task_index] = pdctx->getNextTask();
                            task_ul_ptr_list[task_index]->init(task_ts_exec[task_index]+(t_ns)(task_index), "TaskUL3AggrEarlyUciInd", task_work_function_ul_aggr_3_early_uci_ind, static_cast<void*>(slot_map_ul),
                                                            first_cell, num_cells, ul_task_count, 0);
                            task_index++;
                        }
                    }
                    if(aggr_ulbfw_ptr)
                    {
                        if(task_index>=ul_task_count)
                        {
                            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Task index exceeds UL task count");
                            goto cleanup_err;
                        }
                        else
                        {
                            task_ul_ptr_list[task_index] = pdctx->getNextTask();
                            task_ul_ptr_list[task_index]->init(task_ts_exec[task_index]+(t_ns)(task_index), "TaskULAggrUlBfw", task_work_function_ul_aggr_bfw, static_cast<void*>(slot_map_ul),
                                                            first_cell, num_cells, ul_task_count, 0);
                            task_index++;
                            task_ul_ptr_list[task_index] = pdctx->getNextTask();
                            task_ul_ptr_list[task_index]->init(task_ts_exec[task_index]+(t_ns)(task_index), "TaskUL3AggrUlBfw", task_work_function_ul_aggr_3_ulbfw, static_cast<void*>(slot_map_ul),
                                                            first_cell, num_cells, ul_task_count, 0);
                            task_index++;
                        }
                    }
                    if(aggr_prach_ptr)
                    {
                        if(task_index>=ul_task_count)
                        {
                            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Task index exceeds UL task count");
                            goto cleanup_err;
                        }
                        else
                        {
                            task_ul_ptr_list[task_index] = pdctx->getNextTask();
                            task_ul_ptr_list[task_index]->init(task_ts_exec[task_index]+(t_ns)(task_index), "TaskUL1AggrPrach", task_work_function_ul_aggr_1_prach, static_cast<void*>(slot_map_ul),
                                                            first_cell, num_cells, ul_task_count, 0);
                            task_index++;
                        }
                    }
                    if(aggr_srs_ptr)
                    {
                        if(task_index>=ul_task_count)
                        {
                            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Task index exceeds UL task count");
                            goto cleanup_err;
                        }
                        else
                        {
                            task_ul_ptr_list[task_index] = pdctx->getNextTask();
                            if(pdctx->gpuCommEnabledViaCpu())
                            {
                                task_ts_exec[task_index]= t_ns(t0_slot + UL_TASK1_SRS_LAUNCH_OFFSET_FROM_T0_NS);
                            }
                            task_ul_ptr_list[task_index]->init(task_ts_exec[task_index]+(t_ns)(task_index), "TaskUL1AggrSrs", task_work_function_ul_aggr_1_srs, static_cast<void*>(slot_map_ul),
                                                            first_cell, num_cells, ul_task_count, 0);
                            task_index++;
                        }
                    }
                }
            }
            ///////////////////////////////////////////////////////////////////////
            //// Task3: Wait & L2 Callback
            ///////////////////////////////////////////////////////////////////////
            num_cells = slot_map_ul->getNumCells();

            first_cell = 0;
            do
            {
                task_ul_ptr_list[task_index] = pdctx->getNextTask();

                task_ul_ptr_list[task_index]->init(task_ts_exec[task_index]+(t_ns)(task_index), "TaskUL3Aggr", task_work_function_ul_aggr_3, static_cast<void*>(slot_map_ul),
                                                   first_cell, num_cells, ul_task_count, 0);

                first_cell += num_cells;
                task_index++;
            } while(first_cell < slot_map_ul->getNumCells() && task_index < TASK_MAX_PER_SLOT);

            if(aggr_srs_ptr && !en_orderKernel_tb && ru_type_for_srs_proc != SINGLE_SECT_MODE) //FX RU requires awaiting for SRS completion in the same AGGR3 task as non-SRS
            {
                first_cell = 0;
                do
                {
                    task_ul_ptr_list[task_index] = pdctx->getNextTask();

                    task_ul_ptr_list[task_index]->init(task_ts_exec[task_index]+(t_ns)(task_index), "TaskUL3AggrSrs", task_work_function_ul_aggr_3_srs, static_cast<void*>(slot_map_ul),
                                                    first_cell, num_cells, ul_task_count, 0);

                    first_cell += num_cells;
                    task_index++;
                } while(first_cell < slot_map_ul->getNumCells() && task_index < TASK_MAX_PER_SLOT);                
            }

            ///////////////////////////////////////////////////////////////////////
            //// Enqueue tasks in the list
            ///////////////////////////////////////////////////////////////////////
            tListUl = pdctx->getTaskListUl();
            tListUl->lock();
            for(int tIndex = 0; tIndex < task_index; tIndex++)
                tListUl->push(task_ul_ptr_list[tIndex]);
            tListUl->unlock();

            POP_RANGE
            t_ns map_ul_end = Time::nowNs();

            NVLOGI_FMT(TAG, "Enqueue UL tasks START: {} END: {} DURATION: {} us", map_ul_start.count(), map_ul_end.count(), Time::NsToUs(map_ul_end - map_ul_start).count());
        }
        else
            goto cleanup_err;
    }

    task_ts_exec[0] = sc->tick_original; //Re-assign for DL

    if(slot_map_dl != nullptr)
    {
        auto num_cells = slot_map_dl->getNumCells();
        int num_dl_workers = pdctx->getNumDLWorkers();
        int num_dlc_tasks = get_num_dlc_tasks(num_dl_workers,pdctx->gpuCommEnabledViaCpu(), pdctx->getmMIMO_enable());
        int dl_worker_offset;
        int dlbfw_core_index = 0;
        const bool ENABLE_DL_AFFINITY = (pdctx->get_enable_dl_core_affinity() != 0);

        if(pdctx->getmMIMO_enable() && pdctx->gpuCommEnabledViaCpu())
        {
            if(num_dl_workers < 3) {
                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Insufficient DL workers ({}) for mMIMO with GPU comm via CPU", num_dl_workers);
                goto cleanup_err;
            }
            dlbfw_core_index = num_dl_workers - 3;
        }
        else
        {
            if(num_dl_workers < 2) {
                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Insufficient DL workers ({})", num_dl_workers);
                goto cleanup_err;
            }
            dlbfw_core_index = num_dl_workers - 2;
        }

        if(num_cells > 0 || aggr_dlbfw_ptr)
        {
            if(slot_map_dl->aggrSetPhy(aggr_pdsch_ptr, aggr_pdcch_dl_ptr, aggr_pdcch_ul_ptr, aggr_pbch_ptr, aggr_csirs_ptr,aggr_dlbfw_ptr,current_slot_params_aggr))
            {
                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "SlotMapDL aggrSetPhy");
                goto cleanup_err;
            }

            //Check for Exit handler in flight and skip enqueing of DL tasks accordingly
            if(pExitHandler.test_exit_in_flight())
            {
                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "L1 Exit in flight during Slot Map {}",slot_map_dl->getId());
                goto cleanup_err;
            }

            t_ns map_dl_start = Time::nowNs();

            PUSH_RANGE_PHYDRV("SLOT_DL", 5);
            num_cells = slot_map_dl->getNumCells();
            first_cell = 0;
            bool dlbfw_only = ((aggr_dlbfw_ptr!=nullptr) && !(aggr_pdsch_ptr || aggr_pdcch_dl_ptr || aggr_pdcch_ul_ptr || aggr_pbch_ptr || aggr_csirs_ptr));

            if(pdctx->gpuCommEnabledViaCpu()){
                if(dlbfw_only) //Test for only DLBFW presence
                {
                    //Account for only DLBFW + DL Task 3 Buf clean up tasks
                    dl_task_count+=1;    
                }
                else
                {
                    dl_task_count+= 5 + (num_dlc_tasks<<1); //Add Compression Task + DL Task 2 (Tx) + DL Task 3 + C-Plane Tasks (x2 to factor in UPlane prepare) + FHCB + CPU Door bell task
                }                
                dl_worker_offset=2;
            }
            else
            {
                if(dlbfw_only) //Test for only DLBFW presence
                {
                    //Account for only DLBFW + DL Task 3 Buf clean up tasks
                    dl_task_count+=1;    
                }
                else
                {
                    dl_task_count+= 4 + (num_dlc_tasks<<1); //Add Compression Task + DL Task 2 (Tx) + DL Task 3 + C-Plane Tasks (x2 to factor in UPlane prepare) + FHCB
                }                
                dl_worker_offset=1;
            }
            NVLOGI_FMT(TAG, "Map {} dl_task_count {} dlbfw_only {}", slot_map_dl->getId(),dl_task_count,dlbfw_only);

            if(dl_task_count>=TASK_MAX_PER_SLOT)
                goto cleanup_err;
            for(task_index=1;task_index<dl_task_count;task_index++)
                task_ts_exec[task_index]=task_ts_exec[0];
            slot_map_dl->setTasksTs(dl_task_count, task_ts_exec, Time::nowNs());
            task_index = 0;

            ///////////////////////////////////////////////////////////////////////
            //// Task1: C-plane & CUDA tasks
            ///////////////////////////////////////////////////////////////////////
            //Push Parallelized DL Task 1s into Task queue
            {
                if(!dlbfw_only) //Do not schedule the below DL tasks if DLBFW is the only DL work scheduled 
                {
                    // FH callback
                    if(task_index>=dl_task_count)
                    {
                        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Task index exceeds DL task count");
                        goto cleanup_err;
                    }
                    else
                    {
                        task_dl_ptr_list[task_index] = pdctx->getNextTask();
                        if (!pdctx->getmMIMO_enable())
                        {
                            task_dl_ptr_list[task_index]->init(task_ts_exec[task_index]+ (t_ns)task_index, "TaskDLFHCb", task_work_function_dl_fh_cb, static_cast<void*>(slot_map_dl),
                                                            first_cell, num_cells, dl_task_count, (ENABLE_DL_AFFINITY) ? pdctx->getDLWorkerID(num_dl_workers-dl_worker_offset) : 0);
                        }
                        else
                        {
                            task_dl_ptr_list[task_index]->init(task_ts_exec[task_index]+ (t_ns)task_index, "TaskDLFHCb", task_work_function_dl_fh_cb, static_cast<void*>(slot_map_dl),
                                                            first_cell, num_cells, dl_task_count, (ENABLE_DL_AFFINITY) ? pdctx->getDLWorkerID(num_dl_workers-dl_worker_offset) : 0);
                        }
                        task_index++;
                    }

                    if(aggr_pdsch_ptr)
                    {
                        if(task_index>=dl_task_count)
                        {
                            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Task index exceeds DL task count");
                            goto cleanup_err;
                        }
                        else
                        {
                            task_dl_ptr_list[task_index] = pdctx->getNextTask();
                            if (!pdctx->getmMIMO_enable())
                            {
                                task_dl_ptr_list[task_index]->init(task_ts_exec[task_index]+ (t_ns)task_index, "TaskDL1AggrPdsch", task_work_function_dl_aggr_1_pdsch, static_cast<void*>(slot_map_dl),
                                first_cell, num_cells, dl_task_count, (ENABLE_DL_AFFINITY) ? pdctx->getDLWorkerID(0) : 0);
                            }
                            else
                            {
                                task_dl_ptr_list[task_index]->init(task_ts_exec[task_index]+ (t_ns)task_index, "TaskDL1AggrPdsch", task_work_function_dl_aggr_1_pdsch, static_cast<void*>(slot_map_dl),
                                first_cell, num_cells, dl_task_count, (ENABLE_DL_AFFINITY) ? pdctx->getDLWorkerID(num_dl_workers-dl_worker_offset) : 0);
                            }
                            task_index++;
                        }
                    }

                    if(aggr_pdcch_dl_ptr || aggr_pdcch_ul_ptr || aggr_pbch_ptr || aggr_csirs_ptr)
                    {
                        if(task_index >= dl_task_count)
                        {
                            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Task index exceeds DL task count");
                            goto cleanup_err;
                        }
                        else
                        {
                            task_dl_ptr_list[task_index] = pdctx->getNextTask();
                            if (!pdctx->getmMIMO_enable())
                            {
                                task_dl_ptr_list[task_index]->init(task_ts_exec[task_index]+ (t_ns)task_index, "TaskDL1AggrControl", task_work_function_dl_aggr_control, static_cast<void*>(slot_map_dl),
                                first_cell, num_cells, dl_task_count, (ENABLE_DL_AFFINITY) ? pdctx->getDLWorkerID(1) : 0);
                            }
                            else
                            {
                                task_dl_ptr_list[task_index]->init(task_ts_exec[task_index]+ (t_ns)task_index, "TaskDL1AggrControl", task_work_function_dl_aggr_control, static_cast<void*>(slot_map_dl),
                                first_cell, num_cells, dl_task_count, (ENABLE_DL_AFFINITY) ? pdctx->getDLWorkerID(dlbfw_core_index) : 0);
                            }
                            task_index++;
                        }
                    }
                }

                if(aggr_dlbfw_ptr)
                {
                    if(task_index>=dl_task_count)
                    {
                        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Task index exceeds DL task count\n");
                        goto cleanup_err;
                    }
                    else
                    {
                        task_dl_ptr_list[task_index] = pdctx->getNextTask();
                        task_dl_ptr_list[task_index]->init(task_ts_exec[task_index]+ (t_ns)task_index, "TaskDLAggrDlBfw", task_work_function_dl_aggr_bfw, static_cast<void*>(slot_map_dl),
                                                        first_cell, num_cells, dl_task_count, (ENABLE_DL_AFFINITY) ? pdctx->getDLWorkerID(dlbfw_core_index) : 0);
                        task_index++;
                    }
                }                                    

                if(!dlbfw_only) //Do not schedule the below DL tasks if DLBFW is the only DL work scheduled
                {
                    ///////////////////////////////////////////////////////////////////////
                    //// Task2: Prepare U-plane, wait DL channels, TX U-plane pkts
                    ///////////////////////////////////////////////////////////////////////
                    // /* Only 1 thread with GComm per DL slot */
                    // if(!pdctx->gpuCommEnabled()) {
                    if(!pdctx->gpuCommDlEnabled() || (pdctx->gpuCommDlEnabled() && COMBINE_DL_TASKS_WITH_GPU_INIT_COMMS == 0)) {
                        num_cells = slot_map_dl->getNumCells();

                        first_cell = 0;

                    }
                    if(task_index>=dl_task_count)
                    {
                        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Task index exceeds DL task count");
                        goto cleanup_err;
                    }
                    else
                    {
                        
                        if(pdctx->gpuCommDlEnabled())
                        {
                            if(!pdctx->getmMIMO_enable()) //Only schedule GPU Comm Prepare tasks on GPU Comms Tx core if mMIMO is disabled
                            {
                                for (int c = num_dlc_tasks-1; c >= 0; c--) 
                                {
                                    char buf[32];
                                    task_dl_ptr_list[task_index] = pdctx->getNextTask();
                                    sprintf(buf, "TaskDL2AggrPrepare%d", c + 1);
                                    task_dl_ptr_list[task_index]->init(task_ts_exec[task_index]+ (t_ns)task_index, buf, task_work_function_dl_aggr_2_gpu_comm_prepare, static_cast<void*>(slot_map_dl),
                                                                    c,first_cell,num_dlc_tasks, (ENABLE_DL_AFFINITY) ? pdctx->getDLWorkerID(num_dl_workers-dl_worker_offset) : 0);
                                    task_index++;
                                }                                
                            }
                            task_dl_ptr_list[task_index] = pdctx->getNextTask();
                            task_dl_ptr_list[task_index]->init(task_ts_exec[task_index]+ (t_ns)task_index, "TaskDL2AggrTx", task_work_function_dl_aggr_2_gpu_comm_tx, static_cast<void*>(slot_map_dl),
                                                            first_cell, num_cells, num_dlc_tasks, (ENABLE_DL_AFFINITY) ? pdctx->getDLWorkerID(num_dl_workers-dl_worker_offset) : 0);
                        }
                        else
                        {
                            task_dl_ptr_list[task_index] = pdctx->getNextTask();
                            task_dl_ptr_list[task_index]->init(task_ts_exec[task_index]+ (t_ns)task_index, "TaskDL2Aggr", task_work_function_dl_aggr_2, static_cast<void*>(slot_map_dl),
                                                            first_cell, num_cells, dl_task_count, (ENABLE_DL_AFFINITY) ? pdctx->getDLWorkerID(num_dl_workers-dl_worker_offset) : 0);
                        }
                        task_index++;
                    }

                    if(task_index>=dl_task_count)
                    {
                        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Task index exceeds DL task count");
                        goto cleanup_err;
                    }
                    else
                    {
                        for (int c = num_dlc_tasks-1; c >= 0; c--) {
                            task_dl_ptr_list[task_index] = pdctx->getNextTask();
                            char buf[32];
                            sprintf(buf, "TaskDL1AggrCplane%d", c + 1);
                            int affinity_worker_index = c % (num_dl_workers-dl_worker_offset);//Use all workers except last one (which is reserved for fhcb/comms/compression)
                            task_dl_ptr_list[task_index]->init(task_ts_exec[task_index] + (t_ns)task_index, buf, task_work_function_cplane, static_cast<void*>(slot_map_dl),
                                                                c, first_cell, num_dlc_tasks, (ENABLE_DL_AFFINITY) ? pdctx->getDLWorkerID(affinity_worker_index) : 0);
                            task_index++;
                            if(pdctx->getmMIMO_enable()) //Only schedule GPU Comm Prepare tasks on DLC cores if mMIMO is enabled
                            {
                                task_dl_ptr_list[task_index] = pdctx->getNextTask();
                                sprintf(buf, "TaskDL2AggrPrepare%d", c + 1);
                                task_dl_ptr_list[task_index]->init(task_ts_exec[task_index]+ (t_ns)task_index, buf, task_work_function_dl_aggr_2_gpu_comm_prepare, static_cast<void*>(slot_map_dl),
                                                                c,first_cell,num_dlc_tasks, (ENABLE_DL_AFFINITY) ? pdctx->getDLWorkerID(affinity_worker_index) : 0);
                                task_index++;
                            }
                        }
                    }

                    if(task_index>=dl_task_count)
                    {
                        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Task index exceeds DL task count");
                        goto cleanup_err;
                    }
                    else
                    {
                        worker_id dl_compression_worker_id = 0;
                        if (ENABLE_DL_AFFINITY)
                        {
                            dl_compression_worker_id = ((aggr_dlbfw_ptr)?pdctx->getDLWorkerID(dlbfw_core_index):pdctx->getDLWorkerID(num_dl_workers-dl_worker_offset));
                        }
                        task_dl_ptr_list[task_index] = pdctx->getNextTask();
                        task_dl_ptr_list[task_index]->init(task_ts_exec[task_index]+ (t_ns)task_index, "TaskDL1AggrCompression", task_work_function_dl_aggr_1_compression, static_cast<void*>(slot_map_dl),
                                                        first_cell, num_cells, dl_task_count, dl_compression_worker_id);
                        task_index++;
                    }

                    if(pdctx->gpuCommEnabledViaCpu())
                    {
                        if(task_index>=dl_task_count)
                        {
                            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Task index exceeds DL task count");
                            goto cleanup_err;
                        }
                        else
                        {
                            task_dl_ptr_list[task_index] = pdctx->getNextTask();
                            // We use slot_advance + 1 to guarantee that the packets in the buffer are no longer used and the packets have been transmitted
                            task_dl_ptr_list[task_index]->init(task_ts_exec[task_index]+ (t_ns)task_index, "TaskDL2RingCpuDoorbell", task_work_function_dl_aggr_2_ring_cpu_doorbell, static_cast<void*>(slot_map_dl),
                                                                    first_cell, num_cells, dl_task_count, (ENABLE_DL_AFFINITY) ? pdctx->getDLWorkerID(num_dl_workers-1) : 0);
                            task_index++;
                        }
                    }
                }
            }

            if(task_index>=dl_task_count)
            {
                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Task index exceeds DL task count");
                goto cleanup_err;
            }
            else
            {
                task_dl_ptr_list[task_index] = pdctx->getNextTask();
                // We use slot_advance + 1 to guarantee that the packets in the buffer are no longer used and the packets have been transmitted
                task_dl_ptr_list[task_index]->init(t_ns(t0_slot + Cell::getTtiNsFromMu(MU_SUPPORTED)), "TaskDL3Aggr", task_work_function_dl_aggr_3_buf_cleanup, static_cast<void*>(slot_map_dl),
                                                        first_cell, num_cells, dl_task_count, (ENABLE_DL_AFFINITY) ? pdctx->getDLWorkerID(num_dl_workers-dl_worker_offset) : 0);
                task_index++;
            }
            ///////////////////////////////////////////////////////////////////////
            //// Enqueue tasks in the list
            ///////////////////////////////////////////////////////////////////////
            tListDl = pdctx->getTaskListDl();
            tListDl->lock();
            for(int tIndex = 0; tIndex < dl_task_count; tIndex++)
                tListDl->push(task_dl_ptr_list[tIndex]);
            tListDl->unlock();

            if (pdctx->debug_worker_enabled()) {
                auto debug_task = pdctx->getNextTask();
                debug_task->init(task_ts_exec[0], "Debug", task_work_function_debug, static_cast<void*>(slot_map_dl),
                                 first_cell, num_cells, dl_task_count, 0);
                auto dl = pdctx->getTaskListDebug();
                dl->lock();
                dl->push(debug_task);
                dl->unlock();
            }
            POP_RANGE

            t_ns map_dl_end = Time::nowNs();

            NVLOGI_FMT(TAG, "Enqueue DL tasks START: {} END: {} DURATION: {} us",map_dl_start.count() , map_dl_end.count() ,Time::NsToUs(map_dl_end - map_dl_start).count());
        }
        else
            goto cleanup_err;
    }


    if(slot_map_dl!=nullptr){
        errorInfoDl->prevSlotNonAvail=false;
        errorInfoDl->nonAvailCount=0;
        NVLOGD_FMT(TAG,"[DL Slot] Aggr Objects Available prevSlotNonAvail {},nonAvailCount {}",errorInfoDl->prevSlotNonAvail,errorInfoDl->nonAvailCount);
    }
    if(slot_map_ul!=nullptr){
        errorInfoUl->prevSlotNonAvail=false;
        errorInfoUl->nonAvailCount=0;
        NVLOGD_FMT(TAG,"[UL Slot] Aggr Objects Available prevSlotNonAvail {},nonAvailCount {}",errorInfoUl->prevSlotNonAvail,errorInfoUl->nonAvailCount);
    }

    pdctx->getFhProxy()->updateDynamicBeamIdOffset();

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    t6 = Time::nowNs();
    NVLOGI_FMT(TAG, "l1_enqueue_phy_work START: {} END: {} DURATION: {} us", t0.count(), t6.count(), Time::NsToUs(t6 - t0).count());

    return 0;

cleanup_err:
    // NVSLOGE(TAG, AERIAL_CUPHYDRV_API_EVENT) << "Exit error";

    pdctx->getFhProxy()->updateDynamicBeamIdOffset();

    if(slot_map_ul) slot_map_ul->release(UL_MAX_CELLS_PER_SLOT,false);
    if(oentity_ptr) oentity_ptr->release();
    if(slot_map_dl) slot_map_dl->release(DL_MAX_CELLS_PER_SLOT);
    if(dlbuf) dlbuf->release();
    if(current_slot_params) delete current_slot_params;

    if(ulbuf_st1) ulbuf_st1->release();
    // if(ulbuf_st3_v.size() > 0)
    // {
    //     for(auto& p : ulbuf_st3_v)
    //     {
    //         if(p)
    //         {
    //             p->release();
    //             p = nullptr;
    //         }
    //     }
    // }
    // ulbuf_st3_v.clear();

    if(aggr_pusch_ptr) aggr_pusch_ptr->release();
    if(aggr_pucch_ptr) aggr_pucch_ptr->release();
    if(aggr_prach_ptr) aggr_prach_ptr->release();
    if(aggr_srs_ptr)   aggr_srs_ptr->release();
    if(aggr_pdsch_ptr) aggr_pdsch_ptr->release();

    /*Error handling for Non-availability of DL or UL Aggr Object OR DL or UL Buffers OR Order kernel object for this slot*/
    if(!isAggrObjAvail || !isUlDlBufAvail || !isOKobjAvail){
        if(slot_map_dl!=nullptr)
        {
            errorInfoDl->nonAvailCount++;
            if(errorInfoDl->prevSlotNonAvail){
                if(errorInfoDl->nonAvailCount>pdctx->getAggr_obj_non_avail_th()) //Greater than Yaml configured threshold, Declare fatal error and exit
                {
                    NVLOGE_FMT(TAG,AERIAL_CUPHYDRV_API_EVENT,"[DL] Successive Non-availability of Aggregated DL objects OR DL Buffers reached Threshold {}, Exiting!!!",pdctx->getAggr_obj_non_avail_th());
                    ENTER_L1_RECOVERY()
                }
            }
            errorInfoDl->prevSlotNonAvail=true;
            NVLOGD_FMT(TAG,"[DL] Aggr Objects OR DL Buffers Non-available prevSlotNonAvail {},nonAvailCount {} isAggrObjAvail {} isUlDlBufAvail {}",errorInfoDl->prevSlotNonAvail,errorInfoDl->nonAvailCount,isAggrObjAvail,isUlDlBufAvail);
            return -2;//Change error code to -2 s.t L2A can handle this error case differently
        }
        if(slot_map_ul!=nullptr)
        {
            errorInfoUl->nonAvailCount++;
            if(errorInfoUl->prevSlotNonAvail){
                if(errorInfoUl->nonAvailCount>pdctx->getAggr_obj_non_avail_th()) //Greater than Yaml configured threshold, Declare fatal error and exit
                {
                    NVLOGE_FMT(TAG,AERIAL_CUPHYDRV_API_EVENT,"[UL] Successive Non-availability of Aggregated UL objects OR UL Buffers OR Order Kernel Objects reached Threshold {}, Exiting!!!",pdctx->getAggr_obj_non_avail_th());
                    ENTER_L1_RECOVERY()
                }
            }
            errorInfoUl->prevSlotNonAvail=true;
            NVLOGD_FMT(TAG,"[UL] Aggr Objects OR UL Buffers OR Order Kernel Objects Non-available prevSlotNonAvail {},nonAvailCount {} isAggrObjAvail {} isUlDlBufAvail {} isOKobjAvail {}",errorInfoUl->prevSlotNonAvail,errorInfoUl->nonAvailCount,isAggrObjAvail,isUlDlBufAvail,isOKobjAvail);
            return -2;//Change error code to -2 s.t L2A can handle this error case differently
        }
    }

    return -1;
}



int l1_cell_create(phydriver_handle pdh, struct cell_phy_info& cell_pinfo)
{
    PhyDriverCtx* pdctx = nullptr;
    int           ret   = 0;

    try
    {
        pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
        pdctx->setPuschEarlyHarqEn(cell_pinfo.is_early_harq_detection_enabled);
        pdctx->setPuschAggrFactor(cell_pinfo.pusch_aggr_factor);

        ret = pdctx->setCellPhyByMplane(cell_pinfo);
        if(ret)
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "setCellPhyByMplane() failed with error {}", ret);
            return ret;
        }

        Cell* c = pdctx->getCellByPhyId(cell_pinfo.phy_stat.phyCellId);
        if(c == nullptr)
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Could't getCellByPhyId {}", cell_pinfo.phy_stat.phyCellId);
            return -1;
        }

    }
    PHYDRIVER_CATCH_EXCEPTIONS();

    return 0;
}

int l1_cell_destroy(phydriver_handle pdh, uint16_t cell_id)
{
    PhyDriverCtx* pdctx = nullptr;
    try
    {
        pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    }
    PHYDRIVER_CATCH_EXCEPTIONS();

    return pdctx->removeCell(cell_id);
}

int l1_cell_start(phydriver_handle pdh, uint16_t cell_id)
{
    PhyDriverCtx* pdctx = nullptr;
    try
    {
        pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    }
    PHYDRIVER_CATCH_EXCEPTIONS();

    Cell* c = pdctx->getCellByPhyId(cell_id);
    if(c == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Could't find any cell with id {}", cell_id);
        return -1;
    }

    c->start();
    AppConfig::getInstance().cellActivated(c->getMplaneId());
    return 0;
}

int l1_cell_stop(phydriver_handle pdh, uint16_t cell_id)
{
    PhyDriverCtx* pdctx = nullptr;

    try
    {
        pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    }
    PHYDRIVER_CATCH_EXCEPTIONS();

    Cell* c = pdctx->getCellByPhyId(cell_id);
    if(c == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Could't find any cell with id {}", cell_id);
        return -1;
    }
    c->stop();
    AppConfig::getInstance().cellDeactivated(c->getMplaneId());
    return 0;
}

int l1_set_log_error_handler(phydriver_handle pdh, log_handler_fn_t log_fn)
{
    PhyDriverCtx* pdctx = nullptr;

    try
    {
        pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
        try
        {
            pdctx->set_error_logger(log_fn);
        }
        PHYDRIVER_CATCH_EXCEPTIONS();
    }
    PHYDRIVER_CATCH_EXCEPTIONS();
    return 0;
}

int l1_set_log_info_handler(phydriver_handle pdh, log_handler_fn_t log_fn)
{
    PhyDriverCtx* pdctx = nullptr;
    try
    {
        pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
        try
        {
            pdctx->set_info_logger(log_fn);
        }
        PHYDRIVER_CATCH_EXCEPTIONS();
    }
    PHYDRIVER_CATCH_EXCEPTIONS();
    return 0;
}

int l1_set_log_debug_handler(phydriver_handle pdh, log_handler_fn_t log_fn)
{
    PhyDriverCtx* pdctx = nullptr;
    try
    {
        pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
        try
        {
            pdctx->set_debug_logger(log_fn);
        }
        PHYDRIVER_CATCH_EXCEPTIONS();
    }
    PHYDRIVER_CATCH_EXCEPTIONS();
    return 0;
}

int l1_set_log_level(phydriver_handle pdh, l1_log_level log_lvl)
{
    PhyDriverCtx* pdctx = nullptr;
    try
    {
        pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
        pdctx->set_level_logger(log_lvl);
        NVLOGC_FMT(TAG, "Log level set to {}", +log_lvl);
    }
    PHYDRIVER_CATCH_EXCEPTIONS();
    return 0;
}

int l1_cell_update_cell_config(phydriver_handle pdh, uint16_t mplane_id, uint16_t grid_sz, bool dl)
{
    try
    {
        PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
        Cell* c = pdctx->getCellByMplaneId(mplane_id);
        if(c == nullptr)
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "{}: Could't getCellByMplaneId ", __func__, mplane_id);
            return -1;
        }

        if(c->isActive())
        {
           NVLOGC_FMT(TAG, "Cell active, cannot update config!");
           return -1;
        }

        if(dl)
        {
            NVLOGC_FMT(TAG, "Update cell: mplane_id={} dl_grid_sz={} ", mplane_id, grid_sz);
            c->setDLGridSize(grid_sz);
        }
        else
        {
            NVLOGC_FMT(TAG, "Update cell: mplane_id={} ul_grid_sz={} ", mplane_id, grid_sz);
            c->setULGridSize(grid_sz);
        }
    }
    PHYDRIVER_CATCH_EXCEPTIONS();

    return 0;
}

int l1_cell_update_cell_config(phydriver_handle pdh, uint16_t mplane_id, std::string dst_mac, uint16_t vlan_tci)
{
    try
    {
        PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
        Cell* c = pdctx->getCellByMplaneId(mplane_id);
        if(c == nullptr)
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "{}: Could't getCellByMplaneId ", __func__, mplane_id);
            return -1;
        }

        if(c->isActive())
        {
           NVLOGC_FMT(TAG, "Cell active, cannot update config!");
           return -1;
        }

        NVLOGC_FMT(TAG, "Update cell: mplane_id={} dst_mac={} vlan_tci=0x{:X}", mplane_id, dst_mac.c_str(), vlan_tci);
        c->updateCellConfig(dst_mac, vlan_tci);
    }
    PHYDRIVER_CATCH_EXCEPTIONS();

    return 0;
}

int l1_cell_update_attenuation(phydriver_handle pdh, uint16_t mplane_id, float attenuation_dB)
{
    try
    {
        PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
        Cell* c = pdctx->getCellByMplaneId(mplane_id);
        if(c == nullptr)
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "{}: Could't getCellByMplaneId ", __func__, mplane_id);
            return -1;
        }

        NVLOGC_FMT(TAG, "Update cell: mplane_id={} attenuation_dB={} dB", mplane_id, attenuation_dB);
        c->setAttenuation_dB(attenuation_dB);
    }
    PHYDRIVER_CATCH_EXCEPTIONS();

    return 0;
}

int l1_update_gps_alpha_beta(phydriver_handle pdh,uint64_t alpha,int64_t beta)
{
    PhyDriverCtx* pdctx =  StaticConversion<PhyDriverCtx>(pdh).get();
    pdctx->set_gps_alpha(alpha);
    pdctx->set_gps_beta(beta);

    return 0;
}

void* ptp_svc_monitoring_func(void* arg)
{
    NVLOGI_FMT(TAG, "ptp_svc_monitoring_func thread");

    // Switch to a low_priority_core to avoid blocking time critical thread
    auto& appConfig = AppConfig::getInstance();
    auto low_priority_core = appConfig.getLowPriorityCore();
    NVLOGD_FMT(TAG, "cuphydriver thread {} affinity set to cpu core {}", __func__, low_priority_core);
    nv_assign_thread_cpu_core(low_priority_core);

    if(pthread_setname_np(pthread_self(), "PtpMonitoring") != 0)
    {
        NVLOGW_FMT(TAG, "{}: set thread name failed", __func__);
    }
    phydriver_handle pdh = reinterpret_cast<phydriver_handle>(arg);
    PhyDriverCtx* pdctx =  StaticConversion<PhyDriverCtx>(pdh).get();

    std::string syslogPath = "/host/var/log/syslog";
    auto ptp_rms_threshold = appConfig.getPtpRmsThreshold();
    int ptp_status = 0;
    /* Single thread runs this (one pthread from l1_initialize); no synchronization needed. */
    static time_t last_reported_link_down_ts = time(nullptr);
    static time_t last_reported_link_up_ts = time(nullptr);

    while(1)
    {
        int new_ptp_status = AppUtils::checkPtpServiceStatus(syslogPath, ptp_rms_threshold, ptp_rms_threshold);
        if(ptp_status != new_ptp_status)
        {
            //Send PTP ERROR.indication
            slot_command_api::dl_slot_callbacks      dl_cb{};
            std::array<uint32_t, MAX_CELLS_PER_SLOT> cell_idx_list = {};
            const auto                               cell_count    = pdctx->getCellIdxList(cell_idx_list);
            auto                                     error_code    = new_ptp_status ? SCF_ERROR_CODE_PTP_SVC_ERROR : SCF_ERROR_CODE_PTP_SYNCED;
            if(pdctx->getDlCb(dl_cb))
            {
                if(cell_count > 0)
                {
                    dl_cb.l1_exit_error_fn(SCF_FAPI_ERROR_INDICATION, error_code, cell_idx_list, cell_count);
                }
            }
            ptp_status = new_ptp_status;
        }

        time_t link_down_ts = static_cast<time_t>(-1);
        time_t link_up_ts = static_cast<time_t>(-1);
        AppUtils::getPtpPortLinkEvents(syslogPath, link_down_ts, link_up_ts);
        const bool new_down = (link_down_ts != static_cast<time_t>(-1) && link_down_ts > last_reported_link_down_ts);
        const bool new_up = (link_up_ts != static_cast<time_t>(-1) && link_up_ts > last_reported_link_up_ts);
        if (new_down || new_up) {
            if (new_down) last_reported_link_down_ts = link_down_ts;
            if (new_up) last_reported_link_up_ts = link_up_ts;

            slot_command_api::dl_slot_callbacks dl_cb{};
            std::array<uint32_t, MAX_CELLS_PER_SLOT> cell_idx_list = {};
            const auto cell_count = pdctx->getCellIdxList(cell_idx_list);
            if (pdctx->getDlCb(dl_cb) && cell_count > 0 && dl_cb.l1_exit_error_fn) {
                if (new_down && new_up) {
                    /* Ordering is best-effort when link_down_ts == link_up_ts (syslog has 1s granularity). */
                    if (link_down_ts <= link_up_ts) {
                        dl_cb.l1_exit_error_fn(SCF_FAPI_ERROR_INDICATION, SCF_ERROR_CODE_FH_PORT_DOWN, cell_idx_list, cell_count);
                        NVLOGI_FMT(TAG, "FH port link down indication (0x99)");
                        dl_cb.l1_exit_error_fn(SCF_FAPI_ERROR_INDICATION, SCF_ERROR_CODE_FH_PORT_UP, cell_idx_list, cell_count);
                        NVLOGI_FMT(TAG, "FH port link up indication (0x9A)");
                    } else {
                        dl_cb.l1_exit_error_fn(SCF_FAPI_ERROR_INDICATION, SCF_ERROR_CODE_FH_PORT_UP, cell_idx_list, cell_count);
                        NVLOGI_FMT(TAG, "FH port link up indication (0x9A)");
                        dl_cb.l1_exit_error_fn(SCF_FAPI_ERROR_INDICATION, SCF_ERROR_CODE_FH_PORT_DOWN, cell_idx_list, cell_count);
                        NVLOGI_FMT(TAG, "FH port link down indication (0x99)");
                    }
                } else if (new_down) {
                    dl_cb.l1_exit_error_fn(SCF_FAPI_ERROR_INDICATION, SCF_ERROR_CODE_FH_PORT_DOWN, cell_idx_list, cell_count);
                    NVLOGI_FMT(TAG, "FH port link down indication (0x99)");
                } else if (new_up) {
                    dl_cb.l1_exit_error_fn(SCF_FAPI_ERROR_INDICATION, SCF_ERROR_CODE_FH_PORT_UP, cell_idx_list, cell_count);
                    NVLOGI_FMT(TAG, "FH port link up indication (0x9A)");
                }
            }
        }

        usleep(200000);  /* 200 ms poll for faster port link indication */
    }

    NVLOGI_FMT(TAG, "ptp_svc_monitoring thread exit");
    return nullptr;
}

void* rhocp_ptp_events_monitoring_func(void* arg)
{
    NVLOGC_FMT(TAG, "RhocpPtpMonitor thread");
    // Switch to a low_priority_core to avoid blocking time critical thread
    auto& appConfig = AppConfig::getInstance();
    auto low_priority_core = appConfig.getLowPriorityCore();
    NVLOGC_FMT(TAG, "cuphydriver thread {} affinity set to cpu core {}", __func__, low_priority_core);
    // Set thread affinity to low priority core
    nv_assign_thread_cpu_core(low_priority_core); 

    if(pthread_setname_np(pthread_self(), "RhocpPtpMon") != 0)
    {
        NVLOGW_FMT(TAG, "{}: set thread name failed", __func__);
    }

    phydriver_handle pdh = reinterpret_cast<phydriver_handle>(arg);
    PhyDriverCtx* pdctx =  StaticConversion<PhyDriverCtx>(pdh).get();

    auto rhocp_ptp_publisher = appConfig.getRhocpPtpPublisher();
    auto rhocp_ptp_node_name = appConfig.getRhocpPtpNodeName();
    auto rhocp_ptp_consumer = appConfig.getRhocpPtpConsumer();

    // FYI: startEventServer will create another event consumer thread.
    std::shared_ptr<httplib::Server>  svr = AppUtils::startEventServer(rhocp_ptp_consumer);
    if (!svr) {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Failed to start event consumer server for RHOCP PTP events");
        return nullptr;
    }

    // Give it a moment to start before subscription
    std::this_thread::sleep_for(std::chrono::seconds(1));
    // To pull, we need subscriptions exist first.
    AppUtils::subscribeToEvents(rhocp_ptp_publisher, rhocp_ptp_node_name, rhocp_ptp_consumer);
    // Starting from RHOCP v4.18, pull also requires the consumer server to be running, otherwise subscription will be deleted. 
    // svr->stop();

    int ptp_status = 0;

    while (1) {
        int new_ptp_status = AppUtils::pullEvents(rhocp_ptp_publisher, rhocp_ptp_node_name);
        NVLOGD_FMT(TAG, "RHOCP PTP events status: {} -> {}", ptp_status, new_ptp_status); 
        if(ptp_status != new_ptp_status)
        {
             //Send RHOCP PTP ERROR.indication
            slot_command_api::dl_slot_callbacks      dl_cb{};
            std::array<uint32_t, MAX_CELLS_PER_SLOT> cell_idx_list = {};
            const auto                               cell_count    = pdctx->getCellIdxList(cell_idx_list);
            auto                                     error_code    = new_ptp_status ? SCF_ERROR_CODE_RHOCP_PTP_EVENTS_ERROR : SCF_ERROR_CODE_RHOCP_PTP_EVENTS_SYNCED;
            if(pdctx->getDlCb(dl_cb))
            {
                if(cell_count > 0)
                {
                    dl_cb.l1_exit_error_fn(SCF_FAPI_ERROR_INDICATION, error_code, cell_idx_list, cell_count);
                    if (new_ptp_status == 0)
                        NVLOGW_FMT(TAG, "RHOCP PTP events status backed to normal. Sending indication SCF_ERROR_CODE_RHOCP_PTP_EVENTS_SYNCED code=0x{:X}", +error_code);   
                    else
                        NVLOGW_FMT(TAG, "RHOCP PTP events ERROR detected. Sending indication SCF_ERROR_CODE_RHOCP_PTP_EVENTS_ERROR code=0x{:X}", +error_code);
                }
            }
            ptp_status = new_ptp_status;
        }
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }
    NVLOGC_FMT(TAG, "rhocp_ptp_events_monitoring thread exited");
    return nullptr;
}

void* cell_update_config_func(void* arg)
{
    NVLOGI_FMT(TAG, "cell_update_config_func thread");

    // Switch to a low_priority_core to avoid blocking time critical thread
    auto& appConfig = AppConfig::getInstance();
    auto low_priority_core = appConfig.getLowPriorityCore();
    NVLOGD_FMT(TAG, "cuphydriver thread {} affinity set to cpu core {}", __func__, low_priority_core);
    nv_assign_thread_cpu_core(low_priority_core);

    if(pthread_setname_np(pthread_self(), "cell_update_cfg") != 0)
    {
        NVLOGW_FMT(TAG, "{}: set thread name failed", __func__);
    }
    phydriver_handle pdh = reinterpret_cast<phydriver_handle>(arg);
    PhyDriverCtx* pdctx =  StaticConversion<PhyDriverCtx>(pdh).get();
    if (pdctx->createPrachObjects()!= 0)
    {
        if(pdctx->cellUpdateCbExists())
        {
            NVLOGW_FMT(TAG, "{}: createPrachObjects failed, calling cell_update_cb for cell_id={} with error_code=0x{:X}", __func__, pdctx->updateCellConfigCellId, +SCF_ERROR_CODE_MSG_INVALID_CONFIG);
            pdctx->cell_update_cb(pdctx->updateCellConfigCellId, SCF_ERROR_CODE_MSG_INVALID_CONFIG);
        }
        pdctx->updateCellConfigMutex.unlock();
        NVLOGI_FMT(TAG, "cell_update_config_func: createPrachObjects failed. Send INVALID_CONFIG in CONFIG.RSP. Thread exit");
        return nullptr;
    }

    Cell* cell_list[MAX_CELLS_PER_SLOT];
    uint32_t cellCount = 0;
    pdctx->getCellList(cell_list,&cellCount);

    bool any_cell_active = false;
    for(uint32_t i = 0; i < cellCount; i++)
    {
        auto& cell_ptr = cell_list[i];
        if(cell_ptr->isActive())
            any_cell_active = true;
    }

    if(!any_cell_active)
    {
        pdctx->replacePrachObjects();
    }

    NVLOGI_FMT(TAG, "cell_update_config_func thread exit");
    return nullptr;
}

int l1_lock_update_cell_config_mutex(phydriver_handle pdh)
{

    PhyDriverCtx* pdctx = nullptr;
    int           ret   = 0;

    pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    return pdctx->updateCellConfigMutex.try_lock();
}

int l1_unlock_update_cell_config_mutex(phydriver_handle pdh)
{

    PhyDriverCtx* pdctx = nullptr;
    int           ret   = 0;

    pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    return pdctx->updateCellConfigMutex.unlock();
}

int l1_cell_update_cell_config(phydriver_handle pdh, uint16_t mplane_id, std::unordered_map<int, std::vector<uint16_t>>& eaxcids_ch_map)
{
    PhyDriverCtx* pdctx = nullptr;
    int           ret   = 0;

    pdctx = StaticConversion<PhyDriverCtx>(pdh).get();

    Cell* c = pdctx->getCellByMplaneId(mplane_id);
    if(c == nullptr)
    {
        NVLOGC_FMT(TAG, "Cell does not exist, cannot update config!");
        return -1;
    }

    if(c->isActive())
    {
        NVLOGC_FMT(TAG, "Cell active, cannot update eAxCIDs!");
        return -1;
    }
    else
    {
        c->updateeAxCIds(eaxcids_ch_map);
    }
    return 0;
}

int l1_cell_update_cell_config(phydriver_handle pdh, uint16_t mplane_id, std::unordered_map<std::string, double>& attrs, std::unordered_map<std::string, int>& res)
{
    PhyDriverCtx* pdctx = nullptr;
    int           ret   = 0;

    pdctx = StaticConversion<PhyDriverCtx>(pdh).get();

    Cell* c = pdctx->getCellByMplaneId(mplane_id);
    if(c == nullptr)
    {
        NVLOGC_FMT(TAG, "Cell does not exist, cannot update config!");
        return -1;
    }

    if(c->isActive())
    {
        std::unordered_set<std::string> del;
        for(auto& p : attrs)
        {
            if(strcmp(p.first.c_str(), CELL_PARAM_NIC) == 0) continue;
            NVLOGC_FMT(TAG, "Cell active, skip updating '{}' ", p.first);
            res[p.first] = -1;
            del.insert(p.first);
        }
        for(auto key : del)
        {
            attrs.erase(key);
        }
    }

    if(attrs.find(CELL_PARAM_DL_COMP_METH) != attrs.end() || attrs.find(CELL_PARAM_DL_BIT_WIDTH) != attrs.end())
    {
        if(attrs.find(CELL_PARAM_DL_COMP_METH) == attrs.end())
        {
            NVLOGC_FMT(TAG, "dl_comp_meth is missing, skip updating dl IQ data format...");
            res[CELL_PARAM_DL_BIT_WIDTH] = -1;
        }
        else if(attrs.find(CELL_PARAM_DL_BIT_WIDTH) == attrs.end())
        {
            NVLOGC_FMT(TAG, "dl_bit_width is missing, skip updating dl IQ data format...");
            res[CELL_PARAM_DL_COMP_METH] = -1;
        }
        else
        {
            auto    comp_meth = static_cast<UserDataCompressionMethod>(attrs[CELL_PARAM_DL_COMP_METH]);
            uint8_t bit_width = attrs[CELL_PARAM_DL_BIT_WIDTH];
            if(comp_meth == UserDataCompressionMethod::NO_COMPRESSION && bit_width != 16)
            {
                NVLOGC_FMT(TAG, "Error dl_bit_width {}, Fix point currently only supports 16 bit_width, skip updating dl IQ data format...", bit_width);
                res[CELL_PARAM_DL_BIT_WIDTH] = -1;
                res[CELL_PARAM_DL_COMP_METH] = -1;
            }
            else if(comp_meth == UserDataCompressionMethod::BLOCK_FLOATING_POINT && (bit_width != 9 && bit_width != 14 && bit_width != 16))
            {
                NVLOGC_FMT(TAG, "Error dl_bit_width {}, BFP currently only supports 9, 14, 16 bit_width, skip updating dl IQ data format...", bit_width);
                res[CELL_PARAM_DL_BIT_WIDTH] = -1;
                res[CELL_PARAM_DL_COMP_METH] = -1;
            }
            else
            {
                NVLOGC_FMT(TAG, "{} updated to {:.0f} ", CELL_PARAM_DL_COMP_METH, attrs[CELL_PARAM_DL_COMP_METH]);
                NVLOGC_FMT(TAG, "{} updated to {:.0f} ", CELL_PARAM_DL_BIT_WIDTH, attrs[CELL_PARAM_DL_BIT_WIDTH]);
                c->setDLIQDataFmt(comp_meth, bit_width);
            }
        }
        attrs.erase(CELL_PARAM_DL_COMP_METH);
        attrs.erase(CELL_PARAM_DL_BIT_WIDTH);
    }

    if(attrs.find(CELL_PARAM_UL_COMP_METH) != attrs.end() || attrs.find(CELL_PARAM_UL_BIT_WIDTH) != attrs.end())
    {
        if(attrs.find(CELL_PARAM_UL_COMP_METH) == attrs.end())
        {
            NVLOGC_FMT(TAG, "ul_comp_meth is missing, skip updating ul IQ data format...");
            res[CELL_PARAM_UL_BIT_WIDTH] = -1;
        }
        else if(attrs.find(CELL_PARAM_UL_BIT_WIDTH) == attrs.end())
        {
            NVLOGC_FMT(TAG, "ul_bit_width is missing, skip updating ul IQ data format...");
            res[CELL_PARAM_UL_COMP_METH] = -1;
        }
        else
        {
            auto    comp_meth = static_cast<UserDataCompressionMethod>(attrs[CELL_PARAM_UL_COMP_METH]);
            uint8_t bit_width = attrs[CELL_PARAM_UL_BIT_WIDTH];
            if(comp_meth == UserDataCompressionMethod::NO_COMPRESSION && bit_width != 16)
            {
                NVLOGC_FMT(TAG, "Error ul_bit_width {}, Fix point currently only supports 16 bit_width, skip updating ul IQ data format...", bit_width);
                res[CELL_PARAM_UL_BIT_WIDTH] = -1;
                res[CELL_PARAM_UL_COMP_METH] = -1;
            }
            else if(comp_meth == UserDataCompressionMethod::BLOCK_FLOATING_POINT && (bit_width != 9 && bit_width != 14 && bit_width != 16))
            {
                NVLOGC_FMT(TAG, "Error ul_bit_width {}, BFP currently only supports 9, 14, 16 bit_width, skip updating ul IQ data format...", bit_width);
                res[CELL_PARAM_UL_BIT_WIDTH] = -1;
                res[CELL_PARAM_UL_COMP_METH] = -1;
            }
            else
            {
                NVLOGC_FMT(TAG, "{} updated to {:.0f} ", CELL_PARAM_UL_COMP_METH, attrs[CELL_PARAM_UL_COMP_METH]);
                NVLOGC_FMT(TAG, "{} updated to {:.0f} ", CELL_PARAM_UL_BIT_WIDTH, attrs[CELL_PARAM_UL_BIT_WIDTH]);
                c->setULIQDataFmt(comp_meth, bit_width);
            }
        }
        attrs.erase(CELL_PARAM_UL_COMP_METH);
        attrs.erase(CELL_PARAM_UL_BIT_WIDTH);
    }

    for(auto& p : attrs)
    {
        if(strcmp(p.first.c_str(), CELL_PARAM_UL_GAIN_CALIBRATION) == 0 || strcmp(p.first.c_str(), CELL_PARAM_LOWER_GUARD_BW) == 0)
        {
            continue;
        }

        if(strcmp(p.first.c_str(), CELL_PARAM_NIC) != 0 && strcmp(p.first.c_str(), CELL_PARAM_DST_MAC_ADDR) != 0
        && strcmp(p.first.c_str(), CELL_PARAM_VLAN_ID) != 0 && strcmp(p.first.c_str(), CELL_PARAM_PCP) != 0)
        {
            NVLOGC_FMT(TAG, "{} updated to {:.0f} ", p.first.c_str(), p.second);
        }

        if(strcmp(p.first.c_str(), CELL_PARAM_DST_MAC_ADDR) == 0)
        {
            if(attrs.find(CELL_PARAM_VLAN_ID) == attrs.end() || attrs.find(CELL_PARAM_PCP) == attrs.end())
            {
                NVLOGC_FMT(TAG, "No vlan_id/pcp provided, skip update mac ... ");
            }
            uint64_t mac = p.second;
            std::string dst_mac;
            for(int i = 0; i < 6; i++)
            {
                dst_mac = (dst_mac.size() > 0 ? ":" : "") + dst_mac;
                std::ostringstream ss;
                ss << std::setfill('0') << std::setw(2) << std::hex << (mac & 0xFF);
                dst_mac = ss.str() + dst_mac;
                mac >>= 8;
            }
            uint32_t vlan_id = attrs[CELL_PARAM_VLAN_ID];
            uint32_t pcp = attrs[CELL_PARAM_PCP];
            uint32_t vlan_tci = (pcp << 13) | vlan_id;
            NVLOGC_FMT(TAG, "dst_mac updated to {} ", dst_mac.c_str());
            NVLOGC_FMT(TAG, "vlan_id updated to {} ", vlan_id);
            NVLOGC_FMT(TAG, "pcp updated to {} ", pcp);
            //NVLOGC_FMT(TAG, "dst_mac={} vlan_tci=0x{:X}", dst_mac.c_str(), vlan_tci);
            c->updateCellConfig(dst_mac, vlan_tci);
        }

        if(strcmp(p.first.c_str(), CELL_PARAM_RU_TYPE) == 0)
        {
            c->setRUType(static_cast<ru_type>(p.second));
        }
        else if(strcmp(p.first.c_str(), CELL_PARAM_EXPONENT_DL) == 0)
        {
            c->setDlExponent(p.second);
        }
        else if(strcmp(p.first.c_str(), CELL_PARAM_EXPONENT_UL) == 0)
        {
            c->setUlExponent(p.second);
        }
        else if(strcmp(p.first.c_str(), CELL_PARAM_MAX_AMP_UL) == 0)
        {
            c->setUlMaxAmp(p.second);
        }
        else if(strcmp(p.first.c_str(), CELL_PARAM_PUSCH_PRB_STRIDE) == 0)
        {
            if(pdctx->enableL1ParamSanityCheck())
            {
                if(p.second>ORAN_MAX_PRB) //Exceeds spec limit
                {
                    NVLOGW_FMT(TAG,"Invalid L1 Param value {} provided for CELL_PARAM_PUSCH_PRB_STRIDE.Setting default value of {}",p.second,ORAN_PUSCH_PRBS_X_PORT_X_SYMBOL);
                    p.second=ORAN_PUSCH_PRBS_X_PORT_X_SYMBOL;
                }
                c->setPuschPrbStride(p.second);
            }
            else
                c->setPuschPrbStride(p.second);
        }
        else if(strcmp(p.first.c_str(), CELL_PARAM_PRACH_PRB_STRIDE) == 0)
        {
            if(pdctx->enableL1ParamSanityCheck())
            {
                if(p.second>ORAN_PRACH_PRB) //Exceeds spec limit
                {
                    NVLOGW_FMT(TAG,"Invalid L1 Param value {} provided for CELL_PARAM_PRACH_PRB_STRIDE.Setting default value of {}",p.second,ORAN_PRACH_B4_PRBS_X_PORT_X_SYMBOL);
                    p.second=ORAN_PRACH_B4_PRBS_X_PORT_X_SYMBOL;
                }
                c->setPrachPrbStride(p.second);
            }
            else
                c->setPrachPrbStride(p.second);
        }
        else if(strcmp(p.first.c_str(), CELL_PARAM_SECTION_3_TIME_OFFSET) == 0)
        {
            c->setSection3TimeOffset(p.second);
        }
        else if(strcmp(p.first.c_str(), CELL_PARAM_FH_DISTANCE_RANGE) == 0)
        {
            c->updateFhLenConfig(p.second);
        }
        else if(strcmp(p.first.c_str(), CELL_PARAM_REF_DL) == 0)
        {
            c->setRefDl(p.second);
        }
        else if(strcmp(p.first.c_str(), CELL_PARAM_NIC) == 0)
        {
            uint64_t address = p.second;
            uint32_t domain = (address >> 20) & 0xFFFF;
            uint32_t bus = (address >> 12) & 0xFF;
            uint32_t device = (address >> 4) & 0xFF;
            uint32_t function = address & 0xF;

            std::stringstream ss;
            ss << std::hex << std::setw(4) << std::setfill('0') << domain << ":"
               << std::setw(2) << std::setfill('0') << bus << ":"
               << std::setw(2) << std::setfill('0') << device << "."
               << std::setw(1) << std::setfill('0') << function;

            std::string pcie_address = ss.str();
            NVLOGC_FMT(TAG, "nic updated to {} ", pcie_address.c_str());
            auto ret = c->setNicName(pcie_address);
            if(ret != 0)
            {
                res[p.first] = ret;
            }
        }
    }
    return 0;
}

int l1_cell_update_cell_config(phydriver_handle pdh, struct cell_phy_info& cell_pinfo, CellUpdateCallBackFn& callback)
{
    PhyDriverCtx* pdctx = nullptr;
    int           ret   = 0;

    pdctx = StaticConversion<PhyDriverCtx>(pdh).get();

    Cell* cell_ptr = pdctx->getCellByMplaneId(cell_pinfo.mplane_id);
    if(cell_ptr == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "{}:Could't getCellByPhyId {}", __func__ , cell_pinfo.phy_stat.phyCellId);
        return -1;
    }

    NVLOGI_FMT(TAG, "Update cell config received for PCI {} new PCI {}",cell_ptr->getPhyId(), cell_pinfo.phy_stat.phyCellId);

    if(cell_ptr->getPhyId() != cell_pinfo.phy_stat.phyCellId)
    {
        //If the PCI has been changed, update cell_index_map to use the new phyCellId for this cell
        ret = pdctx->setCellPhyId(cell_ptr->getPhyId(),cell_pinfo.phy_stat.phyCellId,cell_ptr->getId());
        if(ret)
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "{}:setCellPhyId error {}", __func__ , ret);
            return ret;
        }
    }

    //Update the static parameters of the cell
    ret = cell_ptr->setPhyStatic(cell_pinfo);
    if(ret)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "{}:setPhyStaticInfo error {}", __func__ , ret);
        return ret;
    }

    if(pdctx->updateCellConfig(cell_ptr->getId(),cell_pinfo) == 1)
    {
        pdctx->setCellUpdateCb(callback);
        pdctx->updateCellConfigCellId = cell_pinfo.mplane_id;
        pthread_t thread_id;
        int status=pthread_create(&thread_id, nullptr, cell_update_config_func, pdh);

        if(status == 0)
        {
            NVLOGC_FMT(TAG, "launch a new thread cell_update_config_func for mplane_id={}. Return 1", cell_pinfo.mplane_id);
            // Switch to a low_priority_core to avoid blocking time critical thread
            auto&     appConfig         = AppConfig::getInstance();
            auto      low_priority_core = appConfig.getLowPriorityCore();
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(low_priority_core, &cpuset);
            status = pthread_setaffinity_np(thread_id, sizeof(cpu_set_t), &cpuset);
            if(status)
            {
                NVLOGW_FMT(TAG, "cell_update_config_func setaffinity_np failed with status : {}", std::strerror(status));
            }
            return 1;
        }
    }
    else if(ret == -1)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "updateCellConfig failed");
        return ret;
    }
    NVLOGI_FMT(TAG, "Successful update of cell config. Return 0") ;
    return 0;
}

uint8_t l1_get_prach_start_ro_index(phydriver_handle pdh, uint16_t phyCellId)
{
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();

    Cell* c = pdctx->getCellByPhyId(phyCellId);
    if(c == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Could't find any cell with id {}", phyCellId);
        return -1;
    }
    return c->getPrachOccaPrmStatIdx();
}

bool l1_allocSrsChesBuffPool(phydriver_handle pdh, uint32_t requestedBy, uint16_t phyCellId, uint32_t poolSize)
{
    bool retVal = false;
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    if(pdctx != nullptr)
    {
        CvSrsChestMemoryBank* cv = pdctx->getCvSrsChestMemoryBank();
        if(cv != nullptr)
        {
            retVal = cv->memPoolAllocatePerCell(requestedBy, phyCellId, poolSize);
        }
    }
    return retVal;
}

bool l1_deAllocSrsChesBuffPool(phydriver_handle pdh, uint16_t phyCellId)
{
    bool retVal = false;
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    if(pdctx != nullptr)
    {
        CvSrsChestMemoryBank* cv = pdctx->getCvSrsChestMemoryBank();
        if(cv != nullptr)
        {
            retVal = cv->memPoolDeAllocatePerCell(phyCellId);
        }
    }
    return retVal;
}

void l1_copy_TB_to_gpu_buf(phydriver_handle pdh, uint16_t phy_cell_id, uint8_t * tb_buff, uint8_t ** gpu_buff_ref, uint32_t tb_len, uint8_t slot_index)
{
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    Cell* c = pdctx->getCellByPhyId(phy_cell_id);
    (*gpu_buff_ref) = (uint8_t *)c->get_pdsch_tb_buffer(slot_index);
    /*All cuda operation should happen in PDSCH context*/
    pdctx->enable_prepone_h2d_cpy = true;
    NVLOGI_FMT(TAG, "l1_copy_TB_to_gpu_buf pdh={} cell_id={} tb_buff={} gpu_buff_ref={} tb_len={} slot_index={}",(void*)pdh,phy_cell_id,(void*)tb_buff,(void*)gpu_buff_ref,tb_len,slot_index);
    pdctx->getPdschMpsCtx()->setCtx();
    
    if(pdctx->num_pdsch_buff_copy == 0)
    {
        CUDA_CHECK(cudaEventRecord(pdctx->get_event_pdsch_tb_cpy_start(slot_index), pdctx->getH2DCpyStream()));
    }
    
    CUDA_CHECK(cudaMemcpyAsync((uint32_t*)(* gpu_buff_ref),
                                            tb_buff,
                                            tb_len,
                                            cudaMemcpyHostToDevice,
                                            pdctx->getH2DCpyStream()));
    if(++(pdctx->num_pdsch_buff_copy) == pdctx->getCellNum())
    {
        pdctx->num_pdsch_buff_copy = 0;

        CUDA_CHECK(cudaEventRecord(pdctx->get_event_pdsch_tb_cpy_complete(slot_index), pdctx->getH2DCpyStream()));
    }

}

void* l1_copy_TB_to_gpu_buf_thread_func(void* arg)
{
    NVLOGI_FMT(TAG,"l1_copy_TB_to_gpu_buf_thread_func Entry");
    phydriver_handle pdh = reinterpret_cast<phydriver_handle>(arg);
    PhyDriverCtx* pdctx =  StaticConversion<PhyDriverCtx>(pdh).get();
    h2d_copy_prepone_info_t h2d_cpy_info;
    Cell* c;
    uint16_t h2d_read_idx,h2d_write_idx;
    int h2d_copy_done_cur_slot_idx;

    std::vector<cuphyBatchedMemcpyHelper> batched_memcpy_helper;
    batched_memcpy_helper.reserve(PDSCH_MAX_GPU_BUFFS);
    std::generate_n(std::back_inserter(batched_memcpy_helper), PDSCH_MAX_GPU_BUFFS,
        [pdctx]() { 
            return cuphyBatchedMemcpyHelper(DL_MAX_CELLS_PER_SLOT, batchedMemcpySrcHint::srcIsHost, batchedMemcpyDstHint::dstIsDevice, (CUPHYDRIVER_PDSCH_USE_BATCHED_COPY == 1) && (pdctx->getUseBatchedMemcpy() == 1)); 
        });
    std::for_each(batched_memcpy_helper.begin(), batched_memcpy_helper.end(), 
        [](auto& helper) { helper.reset(); });

    //NVLOGE(TAG, AERIAL_CUPHYDRV_API_EVENT, "Thread func has CUPHYDRIVER_PDSCH_USE_BATCHED_COPY=%d and BATCHING_GRANULARITY %d for CUDA_VERSION %d\n",
    //           CUPHYDRIVER_PDSCH_USE_BATCHED_COPY, BATCHED_COPY_THREAD_COPY_GRANULARITY, CUDA_VERSION);

    //bool do_batched_memcpy = (CUPHYDRIVER_PDSCH_USE_BATCHED_COPY == 1) && (pdctx->getUseBatchedMemcpy() == 1) && (CUDA_VERSION >= 12080);
    cudaStream_t h2d_copy_stream = pdctx->getH2DCpyStream();
    std::array<int32_t, PDSCH_MAX_GPU_BUFFS> active_batch_sfn;
    active_batch_sfn.fill(-1);

    while(1)
    {
            h2d_read_idx=pdctx->h2d_read_idx;
            h2d_write_idx=pdctx->h2d_write_idx;
            h2d_copy_done_cur_slot_idx=pdctx->h2d_copy_done_cur_slot_idx[pdctx->h2d_copy_done_cur_slot_read_idx].load(std::memory_order_acquire);
            if(h2d_read_idx!=h2d_write_idx)
            {
                h2d_cpy_info=(h2d_copy_prepone_info_t)(*pdctx->get_h2d_copy_prepone_info(pdctx->h2d_read_idx));
                c = pdctx->getCellByPhyId(h2d_cpy_info.phy_cell_id);
                NVLOGD_FMT(TAG, "l1_copy_TB_to_gpu_buf_thread_func pdh={} cell_id={} tb_buff={} gpu_buff_ref={} tb_len={} slot_index={} h2d_read_idx={} h2d_write_idx={}",pdh,h2d_cpy_info.phy_cell_id,(void*)h2d_cpy_info.tb_buff,(void*)h2d_cpy_info.gpu_buff_ref,h2d_cpy_info.tb_len,h2d_cpy_info.slot_index,h2d_read_idx,h2d_write_idx);
                /*All cuda operation should happen in PDSCH context*/
                pdctx->getPdschMpsCtx()->setCtx();

                uint8_t si = h2d_cpy_info.slot_index;
                if (batched_memcpy_helper[si].getMemcpyCount() > 0 &&
                    active_batch_sfn[si] != static_cast<int32_t>(h2d_cpy_info.sfn))
                {
                    NVLOGW_FMT(TAG, "{}: Stale batch detected for slot_index={} (SFN {} vs {}), discarding {}/{} copies (slot was likely dropped)",
                               __func__, (int)si, active_batch_sfn[si], (int)h2d_cpy_info.sfn,
                               batched_memcpy_helper[si].getMemcpyCount(),
                               batched_memcpy_helper[si].getMaxMemcopiesCount());
                    batched_memcpy_helper[si].reset();
                }
                active_batch_sfn[si] = static_cast<int32_t>(h2d_cpy_info.sfn);

                // orig memcpy had a (uint32_t*)(c->get_pdsch_tb_buffer(h2d_cpy_info.slot_index))
                batched_memcpy_helper[h2d_cpy_info.slot_index].updateMemcpy((uint32_t*)c->get_pdsch_tb_buffer(h2d_cpy_info.slot_index),
                                                   h2d_cpy_info.tb_buff,
                                                   h2d_cpy_info.tb_len,
                                                   cudaMemcpyHostToDevice,
                                                   h2d_copy_stream);

    /*
                if(++(pdctx->num_pdsch_buff_copy) == pdctx->getCellNum())
                {
                    pdctx->num_pdsch_buff_copy = 0;

                    CUDA_CHECK(cudaEventRecord(pdctx->get_event_pdsch_tb_cpy_complete(), pdctx->getH2DCpyStream()));
                    pdctx->h2d_copy_cuda_event_rec_done=true; //now an array; did not update commented out code
                }
    */
                pdctx->h2d_read_idx=(pdctx->h2d_read_idx+1)%(DL_MAX_CELLS_PER_SLOT*PDSCH_MAX_GPU_BUFFS);
                NVLOGD_FMT(TAG, "l1_copy_TB_to_gpu_buf_thread_func cell_id={},h2d_read_idx={},h2d_write_idx={}", h2d_cpy_info.phy_cell_id,(int)h2d_read_idx,(int)h2d_write_idx);
            }
            if(h2d_copy_done_cur_slot_idx>=0)
            {
                uint8_t slot_index = h2d_copy_done_cur_slot_idx % PDSCH_MAX_GPU_BUFFS;
                h2d_cpy_info=(h2d_copy_prepone_info_t)(*pdctx->get_h2d_copy_prepone_info(pdctx->h2d_read_idx));
                if ((slot_index == h2d_cpy_info.slot_index) && (h2d_read_idx!=h2d_write_idx))
                {
                    continue; //We do not yet want to launch the batched memcpy if the slot index is the same as the one we are currently updating the batched memcpy object with (yet to transition to the next slot) OR if the h2d_read_idx and h2d_write_idx are the same (read index yet to catch up with write index)
                }
                pdctx->getPdschMpsCtx()->setCtx();

                if (batched_memcpy_helper[slot_index].getMemcpyCount() != 0)
                {
                    CUDA_CHECK(cudaEventRecord(pdctx->get_event_pdsch_tb_cpy_start(h2d_copy_done_cur_slot_idx), pdctx->getH2DCpyStream()));
                    
                    NVLOGI_FMT(TAG, "Launching batched memcpy with {} copies for slot {} ", batched_memcpy_helper[slot_index].getMemcpyCount(), slot_index);
                    cuphyStatus_t status = batched_memcpy_helper[slot_index].launchBatchedMemcpy(h2d_copy_stream);
                    if (status != CUPHY_STATUS_SUCCESS)
                    {
                        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "{}: Launching batched memcpy returned an error", __func__);
                        // not throwing an exception just logging an error as the only side-effect is that PDSCH will process invalid data, but this will not cause any other issues
                    }
                    batched_memcpy_helper[slot_index].reset();
                }
                active_batch_sfn[slot_index] = -1;

                NVLOGD_FMT(TAG, "l1_copy_TB_to_gpu_buf_thread_func triggering cudaEventRecord h2d_read_idx={},h2d_write_idx={}",(int)h2d_read_idx,(int)h2d_write_idx);
                CUDA_CHECK(cudaEventRecord(pdctx->get_event_pdsch_tb_cpy_complete(h2d_copy_done_cur_slot_idx), pdctx->getH2DCpyStream()));
                pdctx->h2d_copy_cuda_event_rec_done[slot_index].store(true,std::memory_order_relaxed);
                pdctx->h2d_copy_done_cur_slot_idx[pdctx->h2d_copy_done_cur_slot_read_idx].store(-1,std::memory_order_relaxed);
                pdctx->h2d_copy_done_cur_slot_read_idx = (pdctx->h2d_copy_done_cur_slot_read_idx + 1) % PDSCH_MAX_GPU_BUFFS;
            }
            else
            {
                std::this_thread::sleep_for(std::chrono::nanoseconds(5000));
            }
    }
    return nullptr;
}

void l1_copy_TB_to_gpu_buf_thread_offload(phydriver_handle pdh, uint16_t phy_cell_id, uint8_t * tb_buff, uint8_t ** gpu_buff_ref, uint32_t tb_len, uint8_t slot_index, uint16_t sfn)
{
    h2d_copy_prepone_info_t h2d_cpy_info;
    h2d_copy_prepone_info_t* h2d_cpy_info_pdctx;
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    h2d_cpy_info.pdh=pdh;
    h2d_cpy_info.phy_cell_id=phy_cell_id;
    h2d_cpy_info.tb_buff=tb_buff;
    h2d_cpy_info.gpu_buff_ref=gpu_buff_ref;
    h2d_cpy_info.tb_len=tb_len;
    h2d_cpy_info.slot_index=slot_index;
    h2d_cpy_info.sfn=sfn;
    Cell* c = pdctx->getCellByPhyId(phy_cell_id);
    (*gpu_buff_ref) = (uint8_t *)c->get_pdsch_tb_buffer(slot_index);
    pdctx->enable_prepone_h2d_cpy = true;
    NVLOGD_FMT(TAG, "l1_copy_TB_to_gpu_buf_thread_offload pdh={} cell_id={} tb_buff={} gpu_buff_ref={} tb_len={} slot_index={} h2d_copy_thread_enable={}",(void*)pdh,phy_cell_id,(void*)tb_buff,(void*)gpu_buff_ref,tb_len,slot_index,pdctx->h2d_copy_thread_enable);
    if(!pdctx->h2d_copy_thread_enable)
    {
        // This is not taking into consideration BATCHED_COPY_THREAD_COPY_GRANULARITY.
        pdctx->updateBatchedMemcpyInfo((uint32_t*)c->get_pdsch_tb_buffer(h2d_cpy_info.slot_index),
                                h2d_cpy_info.tb_buff,
                                h2d_cpy_info.tb_len);

    }
    else
    {
        uint16_t h2d_widx=pdctx->h2d_write_idx;
        h2d_cpy_info_pdctx=pdctx->get_h2d_copy_prepone_info(h2d_widx);
        *h2d_cpy_info_pdctx=h2d_cpy_info;
        h2d_widx=(h2d_widx+1)%(DL_MAX_CELLS_PER_SLOT*PDSCH_MAX_GPU_BUFFS);
        pdctx->h2d_write_idx=h2d_widx;
    }
}

void l1_set_h2d_copy_done_cur_slot_flag(phydriver_handle pdh,int slot_idx)
{
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    pdctx->h2d_copy_done_cur_slot_idx[pdctx->h2d_copy_done_cur_slot_write_idx].store(slot_idx,std::memory_order_release);
    pdctx->h2d_copy_done_cur_slot_write_idx = (pdctx->h2d_copy_done_cur_slot_write_idx + 1) % PDSCH_MAX_GPU_BUFFS;
    NVLOGD_FMT(TAG, "l1_set_h2d_copy_done_cur_slot_flag Set h2d_copy_done_cur_slot to true for slot_idx(%d)",slot_idx);
}

int l1_cv_mem_bank_update(phydriver_handle pdh,uint32_t cell_id,uint16_t rnti,uint16_t buffer_idx,uint16_t reportType,uint16_t startPrbGrp,uint32_t srsPrbGrpSize ,uint16_t numPrgs,
        uint8_t nGnbAnt,uint8_t nUeAnt,uint32_t offset, uint8_t* srsChEsts, uint16_t startValidPrg, uint16_t nValidPrg)
{
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    CvSrsChestMemoryBank* cv = pdctx->getCvSrsChestMemoryBank();

    if(cv == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "{}: CV Memory Bank does not exist", __func__);
        return -1;
    }

    NVLOGD_FMT(TAG, "cell_id {} rnti {} startPrbGrp {} srsPrbGrpSize{} ",cell_id,rnti,startPrbGrp,srsPrbGrpSize);

    CVSrsChestBuff *buffer = nullptr;
    if(cv->preAllocateBuffer(cell_id, rnti, buffer_idx, reportType, &buffer))
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "{}: allocateBuffer returned error", __func__);
        return -1;
    }
    buffer->configSrsInfo(numPrgs, nGnbAnt, nUeAnt, srsPrbGrpSize, startPrbGrp, startValidPrg, nValidPrg);

    //uint8_t* ptr = srsChEsts+offset;
    //NVLOGD(TAG, "%s: srsChEst = %d %d %d %d\n",__func__,*ptr, *(ptr+1), *(ptr+2), *(ptr+3));
    uint8_t* dst = nullptr;
    /* Each ChEst Buffer is of geometry - nPrbGrp x nGnbAnt x nUeLayer with nPrbGrp being the fastest changing dimension
     * In C style ChEst buffer dimensions are -  [nUeLayer][nGnbAnt][nPrbGrp].
     */
    uint32_t size_of_half2 = sizeof(uint32_t);

    MemtraceDisableScope mds;

    pdctx->getSrsMpsCtx()->setCtx();
    for(uint32_t i=0; i < nUeAnt; i++)
    {
        for(uint32_t j=0; j < nGnbAnt; j++)
        {
            uint32_t k = startPrbGrp;
            uint32_t addrOffset=size_of_half2 * (i*nGnbAnt*numPrgs + j*numPrgs + k);
            dst = buffer->getAddr() + addrOffset;
            // TODO replace w/ 3D memcopy
            CUDA_CHECK(cudaMemcpy(dst,srsChEsts+offset+addrOffset,size_of_half2*numPrgs,cudaMemcpyHostToDevice));
        }
    }
    return 0;
}
    
int l1_cv_mem_bank_retrieve_buffer(phydriver_handle pdh,uint32_t cell_id, uint16_t rnti, uint16_t buffer_idx, uint16_t reportType,uint8_t *pSrsPrgSize, uint16_t* pSrsStartPrg, uint16_t* pSrsStartValidPrg, uint16_t* pSrsNValidPrg, cuphyTensorDescriptor_t* descr, uint8_t** ptr)
{
    if(ptr == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "{}: Invalid input - nullptr pointer", __func__);
        return -1;
    }
    else
        *ptr = nullptr;

    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    CvSrsChestMemoryBank* cv = pdctx->getCvSrsChestMemoryBank();

    if(cv == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "{}: CV Memory Bank does not exist", __func__);
        return -1;
    }

    CVSrsChestBuff *buffer = nullptr;
    if(cv->retrieveBuffer(cell_id, rnti, buffer_idx, reportType, &buffer))
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "{}: retrieveBuffer returned error for cell_id {} rnti {} reportType {}", __func__, cell_id, rnti, reportType);
        return -1;
    }

    *ptr = buffer->getAddr();
    buffer->getSrsPrgInfo(pSrsPrgSize, pSrsStartPrg, pSrsStartValidPrg, pSrsNValidPrg);
    *descr = buffer->getSrsDescr();
    return 0;
}

int l1_cv_mem_bank_update_buffer_state(phydriver_handle pdh,uint32_t cell_id, uint16_t buffer_idx, slot_command_api::srsChestBuffState srs_chest_buff_state)
{
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();

    CvSrsChestMemoryBank* cv = pdctx->getCvSrsChestMemoryBank();

    if(cv == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "{}: CV Memory Bank does not exist", __func__);
        return -1;
    }
    cv->updateSrsChestBufferState(cell_id, buffer_idx, srs_chest_buff_state);
    return 0;
    
}

int l1_cv_mem_bank_get_buffer_state(phydriver_handle pdh,uint32_t cell_id, uint16_t buffer_idx, slot_command_api::srsChestBuffState* srs_chest_buff_state)
{
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();

    CvSrsChestMemoryBank* cv = pdctx->getCvSrsChestMemoryBank();

    if(cv == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "{}: CV Memory Bank does not exist", __func__);
        return -1;
    }
    *srs_chest_buff_state = cv->getSrsChestBufferState(cell_id, buffer_idx);
    return 0;

}

int l1_datalake_notify_bfw(phydriver_handle pdh, uint16_t cell_id, uint16_t rnti, uint16_t sfn, uint16_t slot,
                           uint16_t beam_id, uint8_t n_gnb_ant, uint16_t n_prg, uint16_t prg_size,
                           uint8_t bfw_iq_width, uint8_t n_layers, const uint8_t* bfp, uint32_t bfp_len)
{
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    DataLake* dl = pdctx->getDataLake();
    if(dl == nullptr)
    {
        return -1;  // data lake disabled
    }
    dl->notifyBfw(cell_id, rnti, sfn, slot, beam_id, n_gnb_ant, n_prg, prg_size, bfw_iq_width, n_layers, bfp, bfp_len);
    return 0;
}

int l1_cv_mem_bank_update_buffer_usage(phydriver_handle pdh,uint32_t cell_id, uint16_t rnti, uint16_t buffer_idx, uint32_t usage)
{
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();

    CvSrsChestMemoryBank* cv = pdctx->getCvSrsChestMemoryBank();

    if(cv == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "{}: CV Memory Bank does not exist", __func__);
        return -1;
    }
    cv->updateSrsChestBufferUsage(cell_id, rnti, buffer_idx, usage);
    return 0;
}

int l1_cv_mem_bank_get_buffer_usage(phydriver_handle pdh,uint32_t cell_id, uint16_t rnti, uint16_t buffer_idx, uint32_t* usage)
{
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();

    CvSrsChestMemoryBank* cv = pdctx->getCvSrsChestMemoryBank();

    if(cv == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "{}: CV Memory Bank does not exist", __func__);
        return -1;
    }
    *usage = cv->getSrsChestBufferUsage(cell_id, rnti, buffer_idx);
    return 0;
}

int l1_bfw_coeff_retrieve_buffer(phydriver_handle pdh, uint32_t cell_id, bfw_buffer_info* bfw_buffer_info)
{
    if(bfw_buffer_info == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "{}: Invalid input - nullptr pointer", __func__);
        return -1;
    }
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    *bfw_buffer_info = *pdctx->getBfwCoeffBuffer(cell_id);
    return 0;
}


int l1_mMIMO_enable_info(phydriver_handle pdh, uint8_t *pMuMIMO_enable)
{
    PhyDriverCtx* pdctx =  StaticConversion<PhyDriverCtx>(pdh).get();
    *pMuMIMO_enable = pdctx->getmMIMO_enable();
    return 0;
}

int l1_enable_srs_info(phydriver_handle pdh, uint8_t *pEnable_srs)
{
    PhyDriverCtx* pdctx =  StaticConversion<PhyDriverCtx>(pdh).get();
    *pEnable_srs = pdctx->get_enable_srs();
    return 0;
}

int l1_get_cell_group_num(phydriver_handle pdh, uint8_t *cell_group_num)
{
    PhyDriverCtx* pdctx =  StaticConversion<PhyDriverCtx>(pdh).get();
    *cell_group_num = pdctx->getCellGroupNum();
    return 0;
}

int l1_get_ch_segment_proc_enable_info(phydriver_handle pdh, uint8_t* ch_seg_proc_enable)
{
    PhyDriverCtx* pdctx =  StaticConversion<PhyDriverCtx>(pdh).get();
    *ch_seg_proc_enable = pdctx->get_ch_segment_proc_enable();
    return 0;
}

bool l1_incr_recovery_slots(phydriver_handle pdh)
{
    PhyDriverCtx* pdctx =  StaticConversion<PhyDriverCtx>(pdh).get();
    return pdctx->incrL1RecoverySlots();
}

bool l1_incr_all_obj_free_slots(phydriver_handle pdh)
{
    PhyDriverCtx* pdctx =  StaticConversion<PhyDriverCtx>(pdh).get();
    return pdctx->incrAllObjFreeSlots();
}

void l1_reset_all_obj_free_slots(phydriver_handle pdh)
{
    PhyDriverCtx* pdctx =  StaticConversion<PhyDriverCtx>(pdh).get();
    pdctx->resetAllObjFreeSlots();
}

void l1_reset_recovery_slots(phydriver_handle pdh)
{
    PhyDriverCtx* pdctx =  StaticConversion<PhyDriverCtx>(pdh).get();
    pdctx->resetL1RecoverySlots();
}

int l1_storeDBTPduInFH(phydriver_handle pdh, uint16_t cell_id, void* data_buf)
{
    PhyDriverCtx* pdctx =  StaticConversion<PhyDriverCtx>(pdh).get();
    FhProxy * fhproxy = pdctx->getFhProxy();
    if (fhproxy == nullptr)
    {
        return -1;
    }
    return fhproxy->storeDBTPdu(cell_id, data_buf);
}

int l1_resetDBTStorageInFH(phydriver_handle pdh, uint16_t cell_id)
{
    PhyDriverCtx* pdctx =  StaticConversion<PhyDriverCtx>(pdh).get();
    FhProxy * fhproxy = pdctx->getFhProxy();
    if (fhproxy == nullptr)
    {
        return -1;
    }
    return fhproxy->resetDBTStorage(cell_id);
}

int l1_getBeamWeightsSentFlagInFH(phydriver_handle pdh, uint16_t cell_id, uint16_t beamIdx)
{
    PhyDriverCtx* pdctx =  StaticConversion<PhyDriverCtx>(pdh).get();
    FhProxy * fhproxy = pdctx->getFhProxy();
    if (fhproxy == nullptr)
    {
        return -1;
    }
    return fhproxy->getBeamWeightsSentFlag(cell_id,beamIdx);
}

int l1_setBeamWeightsSentFlagInFH(phydriver_handle pdh, uint16_t cell_id, uint16_t beamIdx)
{
    PhyDriverCtx* pdctx =  StaticConversion<PhyDriverCtx>(pdh).get();
    FhProxy * fhproxy = pdctx->getFhProxy();
    if (fhproxy == nullptr)
    {
        return -1;
    }
    return fhproxy->setBeamWeightsSentFlag(cell_id,beamIdx);
}

int16_t l1_getDynamicBeamIdOffset(phydriver_handle pdh, uint16_t cell_id)
{
    PhyDriverCtx* pdctx =  StaticConversion<PhyDriverCtx>(pdh).get();
    FhProxy * fhproxy = pdctx->getFhProxy();
    if (fhproxy == nullptr)
    {
        return -1;
    }
    return (fhproxy->getBfwCPlaneChainingMode() == aerial_fh::BfwCplaneChainingMode::NO_CHAINING) ? -1 : fhproxy->getDynamicBeamIdOffset();
}

int l1_staticBFWConfiguredInFH(phydriver_handle pdh, uint16_t cell_id)
{
    PhyDriverCtx* pdctx =  StaticConversion<PhyDriverCtx>(pdh).get();
    FhProxy * fhproxy = pdctx->getFhProxy();
    if (fhproxy == nullptr)
    {
        return -1;
    }
    return fhproxy->staticBFWConfigured(cell_id);
}

int l1_clear_task_list(phydriver_handle pdh)
{
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    TaskList*     tListUL = pdctx->getTaskListUl();
    TaskList*     tListDL = pdctx->getTaskListDl();

    tListUL->lock();
    tListUL->clear_task_all();
    tListUL->unlock();

    tListDL->lock();
    tListDL->clear_task_all();
    tListDL->unlock();
    NVLOGI_FMT(TAG,"{}: Clearing DL/UL Task Lists", __func__);
    return 0;
}


phydriver_handle l1_getPhydriverHandle()
{
    return l1_pdh;
}

pthread_t l1_getFmtLogThreadId()
{
    return gBg_thread_id;
}

int l1_get_send_static_bfw_wt_all_cplane(phydriver_handle pdh)
{
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    return pdctx->get_send_static_bfw_wt_all_cplane();
}

// Reason for macro is not to add extra call of function to be part of the stack.
// And since we have few locations where we need it, macro is a proper use here.
#define AERIAL_PRINT_BACKTRACE(TRACE_CNT_MAX)      \
    do {                                           \
        backward::StackTrace st;                   \
        std::ignore = st.load_here(TRACE_CNT_MAX); \
        backward::Printer p;                       \
        p.print(st);                               \
    } while(false)

void l1_exit_handler()
{
    static constexpr auto trace_cnt_max = 32ULL;
    NVLOGC_FMT(TAG,"Triggering L1 exit handler");

    //PhyDriver initialization failure
    if(l1_getPhydriverHandle() == nullptr)
    {
        AERIAL_PRINT_BACKTRACE(32ULL);
        NVLOGW_FMT(TAG, "L1 exit handler: PhyDriver handle is null, cleanup may be incomplete");
        return;
    }

#ifdef ENABLE_DPDK_TX_PKT_TRACING
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    rte_trace_save();
#pragma GCC diagnostic pop 
#endif   

    //Step 1 : Send ERROR.indication (Part1) here
    auto* pdctx = StaticConversion<PhyDriverCtx>(l1_getPhydriverHandle()).get();
    slot_command_api::dl_slot_callbacks dl_cb{};
    std::array<uint32_t,MAX_CELLS_PER_SLOT> cell_idx_list={};
    const auto cell_count = pdctx->getCellIdxList(cell_idx_list);
    if(pdctx->getDlCb(dl_cb))
    {
        if(cell_count>0)
        {
            dl_cb.l1_exit_error_fn(SCF_FAPI_ERROR_INDICATION,SCF_ERROR_CODE_L1_P1_EXIT_ERROR,cell_idx_list,cell_count);
        }
    }
    NVLOGC_FMT(TAG,"L1 exit handler: end of step 1");

    //Step 2 : Clear UL/DL Tasks which are still in the Task queue
    l1_clear_task_list(l1_getPhydriverHandle());

    NVLOGC_FMT(TAG,"L1 exit handler: end of step 2");

    //Step 3 : Issue cuCtxSynchronize if CUDA coredump env variables are set
    if(const auto *enable_core = getenv("CUDA_ENABLE_COREDUMP_ON_EXCEPTION"); enable_core)
    {
        if(1==std::stoi(enable_core))
        {
            MemtraceDisableScope md;
            NVLOGC_FMT(TAG,"L1 exit handler: step 3 (CUDA_ENABLE_COREDUMP_ON_EXCEPTION = 1)");

            // If a CUDA coredump generation is triggered, it is possible some of these steps won't take place,
            // if an abort is triggered by default at the end of the GPU core dump generation process.

            //Synchronize for all MPS contexts
            for(int i=0;i<pdctx->mpsCtxList.size();i++)
            {
                NVLOGC_FMT(TAG,"L1 exit handler: step 3 - MPS context {} work start", i);
                pdctx->mpsCtxList[i]->setCtx();
                CU_CHECK_L1_EXIT_PHYDRIVER_NONFATAL(cuCtxSynchronize());
                NVLOGC_FMT(TAG,"L1 exit handler: step 3 - MPS context {} work end", i);
            }

            //Synchronize Primary context as well
            {
                NVLOGC_FMT(TAG,"L1 exit handler: step 3 - primary context work start");
                CUcontext  CurCtx;
                CUcontext  PrimaryCtx;
                CUdevice   cuDev;
                CU_CHECK_L1_EXIT_PHYDRIVER_NONFATAL(cuCtxGetDevice(&cuDev));
                CU_CHECK_L1_EXIT_PHYDRIVER_NONFATAL(cuDevicePrimaryCtxRetain(&PrimaryCtx,cuDev));
                CU_CHECK_L1_EXIT_PHYDRIVER_NONFATAL(cuCtxSynchronize());
                NVLOGC_FMT(TAG,"L1 exit handler: step 3 - primary context work end");
                CU_CHECK_L1_EXIT_PHYDRIVER_NONFATAL(cuDevicePrimaryCtxRelease(cuDev));
            }
            //Step 4 : Send ERROR.indication (Part2) here
            NVLOGC_FMT(TAG,"L1 exit handler: step 4");
            if(pdctx->getDlCb(dl_cb))
            {
                if(cell_count>0)
                {
                    dl_cb.l1_exit_error_fn(SCF_FAPI_ERROR_INDICATION,SCF_ERROR_CODE_L1_P2_EXIT_ERROR,cell_idx_list,cell_count);
                }
            }
            NVLOGC_FMT(TAG,"L1 exit handler: will close nvlog");
            nvlog_fmtlog_close(l1_getFmtLogThreadId());
            printf("L1 exit handler: closed nvlog\n");
            std::fflush(nullptr);
            asm volatile("" : : : "memory"); //Memory barrier inserted here to prevent compiler from reordering and thereby prematurely triggering the system abort before SCF_ERROR_CODE_L1_P2_EXIT_ERROR is sent to L2
            AERIAL_PRINT_BACKTRACE(trace_cnt_max);
            return;
        }
        else
        {
            NVLOGC_FMT(TAG,"L1 exit handler: step 3 (CUDA_ENABLE_COREDUMP_ON_EXCEPTION = 0)");
            asm volatile("" : : : "memory"); //Memory barrier inserted here to prevent compiler from reordering
            MemtraceDisableScope md;
            AERIAL_PRINT_BACKTRACE(trace_cnt_max);
            return;
        }
    }
    else // core dump set to 0
    {
        NVLOGC_FMT(TAG,"L1 exit handler: step 3 (CUDA_ENABLE_COREDUMP_ON_EXCEPTION unset)");
        asm volatile("" : : : "memory"); //Memory barrier inserted here to prevent compiler from reordering
        MemtraceDisableScope md;
        AERIAL_PRINT_BACKTRACE(trace_cnt_max);
        return;
    }
}

bool l1_check_cuphy_objects_status(phydriver_handle pdh)
{
    auto* pdctx =  StaticConversion<PhyDriverCtx>(pdh).get();
    return pdctx->getAggrObjFreeStatus(); 
}

void l1_resetBatchedMemcpyBatches(phydriver_handle pdh)
{
    auto* pdctx =  StaticConversion<PhyDriverCtx>(pdh).get();
    pdctx->resetBatchedMemcpyBatches();
}

uint8_t l1_get_enable_weighted_average_cfo(phydriver_handle pdh)
{
    auto* pdctx =  StaticConversion<PhyDriverCtx>(pdh).get();
    return pdctx->getEnableWeightedAverageCfo();
}

bool l1_get_split_ul_cuda_streams(phydriver_handle pdh)
{
    auto* pdctx =  StaticConversion<PhyDriverCtx>(pdh).get();
    return pdctx->splitUlCudaStreamsEnabled();
}

bool l1_get_dl_tx_notification(phydriver_handle pdh)
{
    auto* pdctx =  StaticConversion<PhyDriverCtx>(pdh).get();
    return pdctx->getEnableTxNotification();
}