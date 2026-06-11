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

#include "scf_5g_slot_commands_pdsch_csirs.hpp"

#include "nvlog.h"
//#include "ti_generic.hpp"
#include "nv_phy_module.hpp"

#define TAG (NVLOG_TAG_BASE_SCF_L2_ADAPTER + 4) // "SCF.SLOTCMD"

// #define DEBUG_SYM_PRB_INFO_STRUCT 1
namespace scf_5g_fapi {

    static uint32_t next_pdsch_cw_offset = 0;
    static void prepare_ra_type0_info(pdsch_params* info, cuphyPdschUeGrpPrm_t& grp, uint32_t ue_grp_index);
    inline void update_pm_weights_cuphy(cuphyPdschUePrm_t& ue, cuphyPdschCellGrpDynPrm_t& cell_grp,
        prc_weights_list_t& list, const pm_weight_map_t & pm_weight_map,
        prc_weights_idx_list_t& cache, const scf_fapi_tx_precoding_beamforming_t& pmi_bf_pdu,
        int32_t cell_index);
    inline void update_fh_params_csirs(const cuphyCsirsRrcDynPrm_t& csirs_inst, scf_fapi_tx_precoding_beamforming_t& pmi_bf_pdu, cell_sub_command& cell_cmd, bool bf_enabled = false, enum ru_type ru = OTHER_MODE, nv::slot_detail_t* slot_detail = nullptr, int32_t cell_index = 0);
    inline void update_pm_weights_prbs(cuphyPdschUePrm_t& ue, cuphyPdschCellGrpDynPrm_t& cell_grp,
        prc_weights_list_t& list, const pm_weight_map_t & pm_weight_map,
        prc_weights_idx_list_t& cache, scf_fapi_tx_precoding_beamforming_t& pmi_bf_pdu,
        prb_info_common_t& prbs, int32_t cell_index);

    inline void csirs_all_sym_ru_handler(slot_info_t* sym_prbs, nv::slot_detail_t* slot_detail, uint16_t num_dl_prb);
    inline void add_beam_id_csirs(beamid_array_t& array, size_t& array_size, tx_precoding_beamforming_t& pmi_bf_pdu, prb_info_t& prb_info, uint16_t beam_id, int32_t cell_idx);
#ifdef ENABLE_L2_SLT_RSP
    void reset_pdsch_cw_offset()
    {
        next_pdsch_cw_offset = 0;
    }
#endif

   template <bool modcomp_enabled = false>
   void update_non_overlapping_csirs(slot_info_t* slot_info)
    {
        auto& alt_csirs_prb_info_list{slot_info->alt_csirs_prb_info_list};
        auto& alt_csirs_prb_info_idx_list{slot_info->alt_csirs_prb_info_idx_list};
        auto& prbs{slot_info->prbs};
        for(int symbol_id = 0 ; symbol_id < OFDM_SYMBOLS_PER_SLOT; symbol_id++)
        {
    
            prb_info_idx_list_t &csirs_prb_index_info = slot_info->symbols[symbol_id][slot_command_api::channel_type::CSI_RS];
            auto csirs_size = csirs_prb_index_info.size();
            if (csirs_size == 0) {
                continue; 
            }

            uint8_t rb_map[MAX_N_PRBS_SUPPORTED] = {0};
            prb_info_idx_list_t &prb_index_info = slot_info->symbols[symbol_id][slot_command_api::channel_type::PDSCH_CSIRS];
            auto size = prb_index_info.size();
            for(int index = 0; index < size; index++)
            {
                int prb_index = prb_index_info[index];
                if(!prbs[prb_index].common.useAltPrb)
                {
                    std::memset(rb_map + prbs[prb_index].common.startPrbc, 1 , prbs[prb_index].common.numPrbc);
                }
                else
                {
                    for(int i = prbs[prb_index].common.startPrbc; i < prbs[prb_index].common.numPrbc * 2; i+= 2 )
                    {
                        rb_map[i] = 1;
                    }
                }
            }
            
            for(int index = 0; index < csirs_size; ++index)
            {
                int prb_index = csirs_prb_index_info[index];
                auto& csirs_prb_info{prbs[prb_index]};
                uint16_t i = csirs_prb_info.common.startPrbc;
                if(!csirs_prb_info.common.useAltPrb)
                {
                    uint16_t lastRB = i + csirs_prb_info.common.numPrbc;
                    while ( i < lastRB)
                    {
                        uint16_t j = i;
                        uint16_t numPrb = 0;
                        while((!rb_map[j]) && (j < lastRB))
                        {
                            ++j;
                            ++numPrb;
                        }
                        if(numPrb)
                        {
                            alt_csirs_prb_info_list[slot_info->alt_csirs_prb_info_list_size] = prb_info_t(i, numPrb);
                            auto& prb_info{alt_csirs_prb_info_list[slot_info->alt_csirs_prb_info_list_size]};
                            prb_info.common.direction = csirs_prb_info.common.direction;
                            prb_info.common.reMask = csirs_prb_info.common.reMask;
                            prb_info.common.numApIndices = csirs_prb_info.common.numApIndices;
                            prb_info.common.portMask = csirs_prb_info.common.portMask;
                            prb_info.common.useAltPrb = csirs_prb_info.common.useAltPrb;
                            prb_info.beams_array = csirs_prb_info.beams_array;
                            prb_info.beams_array_size = csirs_prb_info.beams_array_size;
                            if(csirs_prb_info.common.extType == ORAN_CMSG_SECTION_EXT_TYPE_11)
                            {
                                prb_info.common.extType = csirs_prb_info.common.extType;
                                prb_info.common.isStaticBfwEncoded = csirs_prb_info.common.isStaticBfwEncoded;
                                prb_info.static_bfwCoeff_buf_info.num_prgs = csirs_prb_info.static_bfwCoeff_buf_info.num_prgs;
                                prb_info.static_bfwCoeff_buf_info.prg_size = csirs_prb_info.static_bfwCoeff_buf_info.prg_size;
                                prb_info.static_bfwCoeff_buf_info.dig_bf_interfaces = csirs_prb_info.static_bfwCoeff_buf_info.dig_bf_interfaces;
                                NVLOGD_FMT(TAG, "{} prb_info={}, prb_info.beams_array_size = {}, prb_info.beams_array.size() = {}", __FUNCTION__, static_cast<void *>(&prb_info), prb_info.beams_array_size, prb_info.beams_array.size());
                                for (int i = 0; i < prb_info.beams_array_size; i++)
                                {
                                    NVLOGD_FMT(TAG, "{} prb_info.beams_array[{}] = {}, csirs_prb_info.common.extType = {}, prb_info.static_bfwCoeff_buf_info.num_prgs = {}, prb_info.static_bfwCoeff_buf_info.prg_size = {}, prb_info.static_bfwCoeff_buf_info.dig_bf_interfaces = {}",
                                                    __FUNCTION__, i, prb_info.beams_array[i], csirs_prb_info.common.extType, prb_info.static_bfwCoeff_buf_info.num_prgs, prb_info.static_bfwCoeff_buf_info.prg_size, prb_info.static_bfwCoeff_buf_info.dig_bf_interfaces);
                                }
                            }

                            if (modcomp_enabled) 
                            {
                                prb_info.comp_info.bwScaler = csirs_prb_info.comp_info.bwScaler;
                                prb_info.comp_info.common.extType = csirs_prb_info.comp_info.common.extType;
                                prb_info.comp_info.common.nSections = csirs_prb_info.comp_info.common.nSections;
                                prb_info.comp_info.common.udIqWidth = csirs_prb_info.comp_info.common.udIqWidth;
                                for (int i = 0; i < prb_info.comp_info.common.nSections; i++)
                                {
                                    auto& csirssection = csirs_prb_info.comp_info.sections[i];
                                    auto& prbsection = prb_info.comp_info.sections[i];
                                    prbsection = csirssection; 
                                    prb_info.comp_info.modCompScalingValue[i] = csirs_prb_info.comp_info.modCompScalingValue[i];
                                }
                            }
                            alt_csirs_prb_info_idx_list[symbol_id].emplace_back(slot_info->alt_csirs_prb_info_list_size++);
                        }
                        i = j + 1;
                    }
                }
                else
                {
                    uint16_t lastRB = i + (csirs_prb_info.common.numPrbc << 1);
                    while ( i < lastRB)
                    {
                        uint16_t j = i;
                        uint16_t numPrb = 0;
                        while((!rb_map[j]) && (j < lastRB))
                        {
                            j+=2;
                            ++numPrb;
                        }
                        if(numPrb)
                        {
                            alt_csirs_prb_info_list[slot_info->alt_csirs_prb_info_list_size] = prb_info_t(i, numPrb);
                            auto& prb_info{alt_csirs_prb_info_list[slot_info->alt_csirs_prb_info_list_size]};
                            prb_info.common.direction = csirs_prb_info.common.direction;
                            prb_info.common.reMask = csirs_prb_info.common.reMask;
                            prb_info.common.numApIndices = csirs_prb_info.common.numApIndices;
                            prb_info.common.portMask = csirs_prb_info.common.portMask;
                            prb_info.common.useAltPrb = csirs_prb_info.common.useAltPrb;
                            prb_info.beams_array = csirs_prb_info.beams_array;
                            prb_info.beams_array_size = csirs_prb_info.beams_array_size;
                            if(csirs_prb_info.common.extType == ORAN_CMSG_SECTION_EXT_TYPE_11)
                            {
                                prb_info.common.extType = csirs_prb_info.common.extType;
                                prb_info.common.isStaticBfwEncoded = csirs_prb_info.common.isStaticBfwEncoded;
                                prb_info.static_bfwCoeff_buf_info.num_prgs = csirs_prb_info.static_bfwCoeff_buf_info.num_prgs;
                                prb_info.static_bfwCoeff_buf_info.prg_size = csirs_prb_info.static_bfwCoeff_buf_info.prg_size;
                                prb_info.static_bfwCoeff_buf_info.dig_bf_interfaces = csirs_prb_info.static_bfwCoeff_buf_info.dig_bf_interfaces;
                                NVLOGD_FMT(TAG, "{} prb_info={}, prb_info.beams_array_size = {}, prb_info.beams_array.size() = {}", __FUNCTION__, static_cast<void *>(&prb_info), prb_info.beams_array_size, prb_info.beams_array.size());
                                for (int i = 0; i < prb_info.beams_array_size; i++)
                                {
                                    NVLOGD_FMT(TAG, "{} prb_info.beams_array[{}] = {}, csirs_prb_info.common.extType = {}, prb_info.static_bfwCoeff_buf_info.num_prgs = {}, prb_info.static_bfwCoeff_buf_info.prg_size = {}, prb_info.static_bfwCoeff_buf_info.dig_bf_interfaces = {}",
                                                    __FUNCTION__, i, prb_info.beams_array[i], csirs_prb_info.common.extType, prb_info.static_bfwCoeff_buf_info.num_prgs, prb_info.static_bfwCoeff_buf_info.prg_size, prb_info.static_bfwCoeff_buf_info.dig_bf_interfaces);
                                }
                            }
                           if (modcomp_enabled) 
                            {
                                // Copy mod-comp common (value types) and sections; prbsection = csirssection copies all section fields.
                                prb_info.comp_info.bwScaler = csirs_prb_info.comp_info.bwScaler;
                                prb_info.comp_info.common.extType = csirs_prb_info.comp_info.common.extType;
                                prb_info.comp_info.common.nSections = csirs_prb_info.comp_info.common.nSections;
                                prb_info.comp_info.common.udIqWidth = csirs_prb_info.comp_info.common.udIqWidth;
                                for (int i = 0; i < prb_info.comp_info.common.nSections; i++)
                                {
                                    auto& csirssection = csirs_prb_info.comp_info.sections[i];
                                    auto& prbsection = prb_info.comp_info.sections[i];
                                    prbsection = csirssection;
                                    prb_info.comp_info.modCompScalingValue[i] = csirs_prb_info.comp_info.modCompScalingValue[i];
                                }
                            }
                            alt_csirs_prb_info_idx_list[symbol_id].emplace_back(slot_info->alt_csirs_prb_info_list_size++);
                        }
                        i = j + 2;
                    }

                }
            }
        }
    }
    bool update_cell_command(cell_group_command* cell_grp_cmd, cell_sub_command& cell_sub_cmd, const scf_fapi_pdsch_pdu_t& msg, uint8_t testMode, slot_indication& slotinfo,
                                        int32_t cell_index, pm_weight_map_t& pm_map, bool pm_enabled, bool bf_enabled, uint16_t num_dl_prb, bfw_coeff_mem_info_t *bfwCoeff_mem_info,
                                        bool mmimo_enabled, nv::slot_detail_t*  slot_detail)
    {
        // cell_sub_cmd.create_if(channel_type::PDSCH);
        cell_sub_cmd.slot.type = SLOT_DOWNLINK;
        cell_sub_cmd.slot.slot_3gpp = slotinfo;
        slot_command_api::pdsch_params*  info  = cell_grp_cmd->get_pdsch_params();
        if (info == nullptr)
        {
            NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "no pdsch command");
            return false;
        }

        auto pdsch_ue_idx = info->cell_grp_info.nUes;
        auto pdsch_cw_idx = info->cell_grp_info.nCws;

        auto& ue    = info->ue_info[pdsch_ue_idx];
        // SCF FAPI sticks a variable-length structure in the middle of the struct. Capture the data we need
        // from this struct and move the pointer past it.
        ue.nCw = msg.num_codewords;
        const uint8_t *ptr = reinterpret_cast<const uint8_t*>(&msg.codewords[0]);
        for (uint8_t cw = 0; cw < msg.num_codewords; cw++) {
            const scf_fapi_pdsch_codeword_t *p_cw = reinterpret_cast<const scf_fapi_pdsch_codeword_t*>(ptr);
            auto& ue_cw = info->ue_cw_info[pdsch_cw_idx];
            ue_cw.pUePrm = &ue;
            ue_cw.mcsIndex       = p_cw->mcs_index; // value only used for optional TB size check in cuPHY
            ue_cw.mcsTableIndex  = p_cw->mcs_table; // value only used for optional TB size check in cuPHY, also used to determine maxQm below
            ue_cw.targetCodeRate = p_cw->target_code_rate;
            ue_cw.qamModOrder    = p_cw->qam_mod_order;

            ue_cw.rv            = p_cw->rv_index;
            ue_cw.tbSize        = p_cw->tb_size;
            ue_cw.tbStartOffset = next_pdsch_cw_offset;

            NVLOGD_FMT(TAG, "{}: SFN {}.{} cell_id={} DL_TTI.req: PDU {}-{}-{} tb_size={} pdu_offset={}",
                    __func__, slotinfo.sfn_, slotinfo.slot_, cell_index, pdsch_cw_idx, msg.num_codewords, cw, ue_cw.tbSize, ue_cw.tbStartOffset);

            ue_cw.maxLayers = 4;
            ue_cw.maxQm = p_cw->mcs_table == 1 ? 8 : 6; // We are assuming the MCS table value in the FAPI PDU is valid and so we can use it in maxQm computation
            ue_cw.n_PRB_LBRM = compute_N_prb_lbrm(msg.bwp.bwp_size);

            info->ue_cw_index_info[pdsch_cw_idx] = pdsch_cw_idx;
            ue.pCwIdxs = &(info->ue_cw_index_info[pdsch_cw_idx]);
            next_pdsch_cw_offset += ue_cw.tbSize;
            pdsch_cw_idx++;
            info->cell_grp_info.nCws++;
            ptr += sizeof(scf_fapi_pdsch_codeword_t);
        }

        // We're past the variable-length part
        const scf_fapi_pdsch_pdu_end_t *end = reinterpret_cast<const scf_fapi_pdsch_pdu_end_t*>(ptr);

        ue.BWPStart = msg.bwp.bwp_start;
        ue.scid = end->sc_id;
        ue.dmrsScrmId = end->dl_dmrs_scrambling_id;
        ue.rnti = msg.rnti;
        ue.nUeLayers = end->num_of_layers;
        ue.dmrsPortBmsk = static_cast<uint16_t>(end->dmrs_ports & 0xFFF);
        ue.nlAbove16 = static_cast<uint8_t>((end->dmrs_ports >> PDSCH_ABOVE_16_LAYERS_DMRSPORTS_BIT_LOC) & 0x1);
#if 0
        if (ue.rnti == UINT16_MAX) {
            ue.nUeLayers = 2;
        }
#endif
        ue.dataScramId = end->data_scrambling_id;
        ue.refPoint = end->ref_point;

        // update UE Group - no group present use 0 as default
        auto found = [&end, &msg] (const auto& e) {
            if(end->resource_alloc == 1)
            {
                // NVLOGI_FMT(TAG, "GRP pdschStartSym {} nPdschSym {} startPrb {}, nPrb{}",
                //     e.pdschStartSym,
                //     e.nPdschSym,
                //     e.startPrb,
                //     e.nPrb);
                // NVLOGI_FMT(TAG, "MSG pdschStartSym {} nPdschSym {} bwp_start {} startPrb {}, nPrb{}",
                //     end->start_sym_index,
                //     end->num_symbols,
                //     msg.bwp.bwp_start, end->rb_start,
                //     end->rb_size);
                return (e.pdschStartSym == end->start_sym_index &&
                        e.nPdschSym == end->num_symbols &&
                        e.startPrb == (msg.bwp.bwp_start + end->rb_start) &&
                        e.nPrb ==  end->rb_size);
            }
            else
            {
                return (e.pdschStartSym == end->start_sym_index &&
                        e.nPdschSym == end->num_symbols &&
                        e.startPrb == (msg.bwp.bwp_start) &&
                        !std::memcmp(e.rbBitmap, end->rb_bitmap,sizeof(uint8_t)*MAX_RBMASK_BYTE_SIZE));
            }
        };

        auto iter = std::find_if(info->ue_grp_info + info->cell_ue_group_idx_start, info->ue_grp_info + MAX_PDSCH_UE_GROUPS, found);
        std::size_t ueGrpIndex;
        std::size_t bfwUeGrpIndex;
        bool newUeGrp = false;
        if (iter != std::end(info->ue_grp_info))
        {
            ueGrpIndex = std::distance(info->ue_grp_info, iter);
            bfwUeGrpIndex = info->ue_grp_idx_bfw_id_map[ueGrpIndex];
            //NVLOGD_FMT(TAG, "PDSCH ueGrpIndex={}, info->cell_grp_info.nUeGrps={}", ueGrpIndex, info->cell_grp_info.nUeGrps);
        }
        else
        {
            ueGrpIndex = info->cell_grp_info.nUeGrps;
            bfwUeGrpIndex = info->nue_grps_per_cell[cell_index];
            info->ue_grp_idx_bfw_id_map[ueGrpIndex] = bfwUeGrpIndex;
            ++info->cell_grp_info.nUeGrps;
            ++info->nue_grps_per_cell[cell_index];
            newUeGrp = true;
            //NVLOGD_FMT(TAG, "PDSCH new ueGrpIndex={}, info->cell_grp_info.nUeGrps={} newUeGrp = true", ueGrpIndex, info->cell_grp_info.nUeGrps);
        }
#if 0
        if(mmimo_enabled && (is_latest_bfw_coff_avail(cell_sub_cmd.slot.slot_3gpp.sfn_ , cell_sub_cmd.slot.slot_3gpp.slot_, bfwCoeff_mem_info->sfn, bfwCoeff_mem_info->slot)))
        {
            if((bfwCoeff_mem_info->pdu_idx_rnti_list[ueGrpIndex][pdsch_ue_idx].pdu_idx != pdsch_ue_idx) || (bfwCoeff_mem_info->pdu_idx_rnti_list[ueGrpIndex][pdsch_ue_idx].rnti != msg.rnti))
            {
                NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT,"PDSCH Index or RNTI mismatch between CVI_REQ & DL_TTI slotIndex={} BFW Idx={} DL_TTI Idx={} DL CVI RNTI={} PDSCH RNTI={}",
                                bfwCoeff_mem_info->slotIndex, static_cast<uint16_t>(bfwCoeff_mem_info->pdu_idx_rnti_list[ueGrpIndex][pdsch_ue_idx].pdu_idx), static_cast<uint16_t>(pdsch_ue_idx),
                                static_cast<uint16_t>(bfwCoeff_mem_info->pdu_idx_rnti_list[ueGrpIndex][pdsch_ue_idx].rnti), static_cast<uint16_t>(msg.rnti));
                return;
            }
            else
            {
                NVLOGD_FMT(TAG,"ueGrpIndex={} slotIndex={} BFW Idx={} DL_TTI Idx={} DL CVI RNTI={} PDSCH RNTI={}",ueGrpIndex, bfwCoeff_mem_info->slotIndex,
                                static_cast<uint16_t>(bfwCoeff_mem_info->pdu_idx_rnti_list[ueGrpIndex][pdsch_ue_idx].pdu_idx), static_cast<uint16_t>(pdsch_ue_idx),
                                static_cast<uint16_t>(bfwCoeff_mem_info->pdu_idx_rnti_list[ueGrpIndex][pdsch_ue_idx].rnti), static_cast<uint16_t>(msg.rnti));
            }
        }
#endif
        ue.pUeGrpPrm = &info->ue_grp_info[ueGrpIndex];
        // update UE Group - no group present use 0 as default
        cuphyPdschUeGrpPrm_t& ue_grp =  info->ue_grp_info[ueGrpIndex];
        ue_grp.pUePrmIdxs[ue_grp.nUes] = pdsch_ue_idx;
        ue_grp.nUes++;

        // Update DMRS Info
        cuphyPdschDmrsPrm_t& ue_dmrs = info->ue_dmrs_info[ueGrpIndex];
        ue_dmrs.nDmrsCdmGrpsNoData = end->num_dmrs_cdm_grps_no_data;
        // Update DMRS Index if new Group
        if (newUeGrp) {
            ue_grp.pDmrsDynPrm = &ue_dmrs;
            ue_grp.resourceAlloc = end->resource_alloc;
            if(!ue_grp.resourceAlloc) // RA type-0
            {
                memcpy(ue_grp.rbBitmap,end->rb_bitmap,sizeof(uint8_t)*MAX_RBMASK_BYTE_SIZE);
                //nPrb & startPrb for ra-type0 is updated here
                prepare_ra_type0_info(info, ue_grp, ueGrpIndex);
                ue_grp.startPrb = msg.bwp.bwp_start;
            }
            else
            {
                ue_grp.nPrb = end->rb_size;//update nPrb for ra-type1 only from msg
                ue_grp.startPrb = msg.bwp.bwp_start + end->rb_start;
            }
            ue_grp.pdschStartSym = end->start_sym_index;
            ue_grp.nPdschSym = end->num_symbols;
            ue_grp.dmrsSymLocBmsk = end->dl_dmrs_sym_pos;
            ue_grp.pCellPrm = &info->cell_dyn_info[info->cell_grp_info.nCells-1];
            ue_grp.pCellPrm->testModel = testMode;
            NVLOGD_FMT(TAG,"{}:{} PDSCH testMode={}",__func__,__LINE__,testMode);
            //NVLOGD_FMT(TAG, "PDSCH pCellPrm indexing into cell_dyn_info[{}]",info->cell_grp_info.nCells-1);
        }
        else
        {
            // TODO: Check why this was needed.
            #if 0
            if (mmimo_enabled && static_bfwCoeff_mem_info!=NULL)
            {
                NVLOGD_FMT(TAG, "Static Beamforming is disabled for this PDU as dynamic BFW is enabled");
                NVLOGD_FMT(TAG, "{} static_bfwCoeff_mem_info:{}",__func__,reinterpret_cast<void*>(static_bfwCoeff_mem_info));
                *static_bfwCoeff_mem_info->header = BFW_COFF_MEM_FREE;
                static_bfwCoeff_mem_info = NULL;
            }
            #endif
        }

        /// Skip Ptrs
        const scf_fapi_tx_precoding_beamforming_t* pm_bf = nullptr;
        if (msg.pdu_bitmap & 0x1)
        {
            const scf_fapi_pdsch_ptrs_t& ptrs = *reinterpret_cast<const scf_fapi_pdsch_ptrs_t*>(end->next);
            pm_bf = reinterpret_cast<const scf_fapi_tx_precoding_beamforming_t*>(ptrs.next);
        } else {
            pm_bf = reinterpret_cast<const scf_fapi_tx_precoding_beamforming_t*>(end->next);
        }

        uint16_t numPRGs = pm_bf->num_prgs;
        uint16_t prgSize = pm_bf->prg_size;
        uint8_t digBFInterfaces = pm_bf->dig_bf_interfaces;

        if(!check_bf_pc_params(numPRGs, digBFInterfaces, mmimo_enabled))
        {
            NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{} line {}: check_bf_pc_params failed: numPRGs={} digBFInterfaces={} mmimo_enabled={}",
                __FUNCTION__, __LINE__, static_cast<uint16_t>(numPRGs), static_cast<uint16_t>(digBFInterfaces), mmimo_enabled);
            /* Rollback state to keep pdsch_params and pdsch_fh_params consistent */
            ue_grp.nUes--;
            for (uint8_t cw = 0; cw < msg.num_codewords; cw++) {
                next_pdsch_cw_offset -= info->ue_cw_info[pdsch_cw_idx - 1 - cw].tbSize;
                info->cell_grp_info.nCws--;
            }
            if (newUeGrp) {
                --info->cell_grp_info.nUeGrps;
                --info->nue_grps_per_cell[cell_index];
                /* Rollback cell addition (done in on_dl_tti_request) when rolling back the only UE group */
                if (info->cell_grp_info.nUeGrps == 0 && info->cell_grp_info.nCells > 0) {
                    --info->cell_grp_info.nCells;
                    if (!info->cell_index_list.empty())
                        info->cell_index_list.pop_back();
                    if (!info->phy_cell_index_list.empty())
                        info->phy_cell_index_list.pop_back();
                }
            }
            return false;
        }

        auto& cell_grp_info = info->cell_grp_info;
        cell_grp_info.nUes++;
        info->tb_data.pBufferType = cuphyPdschDataIn_t::CPU_BUFFER;

        const uint8_t *next = reinterpret_cast<const uint8_t*>(pm_bf);
        uint16_t bf_size = 0;
        if (mmimo_enabled == 0 || digBFInterfaces != 0)
        {
            /* PMI feature is not supported for users when MU-MIMO feature is enabled. 
             * The precoding is done using the static beamforming weights recieved from L2 which are already precoded by L2 
             * and sent during cell config request in Digtal Beam Table (DBT) PDU. */
            bf_size = sizeof(numPRGs) + sizeof(prgSize) + sizeof(digBFInterfaces) + numPRGs * sizeof(uint16_t) + numPRGs * digBFInterfaces * sizeof(uint16_t);
            if (mmimo_enabled)
            {
                bfwCoeff_mem_info = NULL;
            }
        }
        else
        {
            /* When dynamic beamforming weight cacluation is requested for the paired users from L2 then digBFInterfaces is set to 0 and PMI and BeamId's
             * are not encoded in the TxPrecoding and Beamforming PDU.  Hence only numPRGs, prgSize and digBFInterfaces are encoded.
             * PMI feature is not used for paired users when MU-MIMO feature is enabled. */
            bf_size = sizeof(numPRGs) + sizeof(prgSize) + sizeof(digBFInterfaces);
        }
        next = next + bf_size;

        const scf_fapi_tx_power_info_t* tx_power_info = reinterpret_cast<const scf_fapi_tx_power_info_t*>(next);
        ue.beta_qam = std::pow(10.0, ((tx_power_info->power_control_offset - 8) + (tx_power_info->power_control_offset_ss - 1)*3.0)/20.0);
        ue.beta_dmrs = std::pow(10.0, ((tx_power_info->power_control_offset - 8) + (tx_power_info->power_control_offset_ss - 1)*3.0)/20.0);

        // if (pm_enabled) {
        //     update_pm_weights(ue, cell_grp_info, info->pm_info, pm_map, info->pmw_idx_cache, *pm_bf);
        // }
        // New Cell arrived
        if(cell_grp_cmd->fh_params.num_pdsch_fh_params.at(cell_index) == 0)
        {
            //Record the start Index
            cell_grp_cmd->fh_params.start_index_pdsch_fh_params.at(cell_index) = cell_grp_cmd->fh_params.total_num_pdsch_pdus;
        }
        auto& pdsch_fh_param = cell_grp_cmd->fh_params.pdsch_fh_params.at(cell_grp_cmd->fh_params.total_num_pdsch_pdus);
        auto& pc_bf = cell_grp_cmd->fh_params.pc_bf_arr.at(cell_grp_cmd->fh_params.total_num_pdsch_pdus);

        ++cell_grp_cmd->fh_params.num_pdsch_fh_params.at(cell_index); 
        ++cell_grp_cmd->fh_params.total_num_pdsch_pdus;

        pdsch_fh_param.grp = &ue_grp;
        pdsch_fh_param.is_new_grp = newUeGrp;
        pdsch_fh_param.ue = &ue;
        pdsch_fh_param.cell_cmd = &cell_sub_cmd;
        pdsch_fh_param.bf_enabled = bf_enabled;
        pdsch_fh_param.pm_enabled = pm_enabled;
        pdsch_fh_param.mmimo_enabled = mmimo_enabled;
        pdsch_fh_param.cell_index = cell_index;
        pdsch_fh_param.num_dl_prb = num_dl_prb;
        pdsch_fh_param.ue_grp_index = ueGrpIndex;
        pdsch_fh_param.ue_grp_bfw_index_per_cell = bfwUeGrpIndex;
        pdsch_fh_param.bfwCoeff_mem_info = bfwCoeff_mem_info;

        ue.enablePrcdBf = pm_enabled;

        pdsch_fh_param.pc_bf = &pc_bf;
        pdsch_fh_param.pc_bf->num_prgs = numPRGs;
        pdsch_fh_param.pc_bf->prg_size = prgSize;
        pdsch_fh_param.pc_bf->dig_bf_interfaces = digBFInterfaces;

        if (mmimo_enabled == 0 || digBFInterfaces != 0)
        {
            memcpy(pdsch_fh_param.pc_bf->pm_idx_and_beam_idx, pm_bf->pm_idx_and_beam_idx, sizeof(uint16_t) * (numPRGs + numPRGs * digBFInterfaces));
        }
        if (likely(!pm_enabled)) {}
        else
        {
            update_pm_weights_cuphy(ue, info->cell_grp_info, info->pm_info, pm_map, info->pmw_idx_cache, *pm_bf, cell_index);
        }
        return true;
    }

   static void prepare_ra_type0_info(pdsch_params* info, cuphyPdschUeGrpPrm_t& grp, uint32_t ue_grp_index)
    {
        uint32_t byte_index = 0;
        int32_t start_prb = -1;
        uint32_t num_prb = 0;
        uint32_t info_index = 0;

        grp.nPrb = 0;

        while(byte_index < MAX_RBMASK_BYTE_SIZE)
        {
            //skip the bytes that don't have any bit set
            while(grp.rbBitmap[byte_index] == 0 && byte_index < MAX_RBMASK_BYTE_SIZE)
            {
                byte_index++;
            }

            //go byte by byte
            while(grp.rbBitmap[byte_index] && byte_index < MAX_RBMASK_BYTE_SIZE)
            {
                uint32_t bitmask = 1;
                uint8_t rb_bitmap = grp.rbBitmap[byte_index];
                while(rb_bitmap)
                {
                   if(bitmask & rb_bitmap)
                   {
                       if(start_prb == -1)
                       {
                            start_prb = byte_index*8 + __builtin_ffs(bitmask)-1;
                       }
                       num_prb++;
                       rb_bitmap &= ~bitmask;
                   }
                   else
                   {
                       if(start_prb != -1)
                       {
                           //hit end of consecutive 1s
                           NVLOGD_FMT(TAG,"start_prb={} num_prb={}",start_prb, num_prb);
                           info->ra_type0_info[info_index][ue_grp_index].start_prb = start_prb;
                           info->ra_type0_info[info_index][ue_grp_index].num_prb = num_prb;
                           grp.nPrb += num_prb;
                           info_index++;
                           start_prb = -1;
                           num_prb = 0;
                       }
                   }
                   bitmask = bitmask << 1;
                }
                if(bitmask != 0x00000100) //leading 0s in rb_bitmap
                {
                   //hit end of consecutive 1s
                   NVLOGD_FMT(TAG,"start_prb={} num_prb={}",start_prb, num_prb);
                   info->ra_type0_info[info_index][ue_grp_index].start_prb = start_prb;
                   info->ra_type0_info[info_index][ue_grp_index].num_prb = num_prb;
                   grp.nPrb += num_prb;
                   info_index++;
                   start_prb = -1;
                   num_prb = 0;
                }
                byte_index++;
            }
            if(start_prb != -1)
            {
                //hit end of consecutive 1s
                NVLOGD_FMT(TAG,"start_prb={} num_prb={}",start_prb, num_prb);
                info->ra_type0_info[info_index][ue_grp_index].start_prb = start_prb;
                info->ra_type0_info[info_index][ue_grp_index].num_prb = num_prb;
                grp.nPrb += num_prb;
                info_index++;
                start_prb = -1;
                num_prb = 0;
            }
        }
        info->num_ra_type0_info[ue_grp_index] = info_index;
        NVLOGD_FMT(TAG,"num_ra_type0_info[{}]={} nPrb={}",ue_grp_index,info_index,grp.nPrb);
    }

    inline void update_pm_weights_cuphy(cuphyPdschUePrm_t& ue, cuphyPdschCellGrpDynPrm_t& cell_grp,
        prc_weights_list_t& list, const pm_weight_map_t & pm_weight_map,
        prc_weights_idx_list_t& cache, const scf_fapi_tx_precoding_beamforming_t& pmi_bf_pdu,
        int32_t cell_index
        )
    {
        uint16_t offset = 0;

        auto restore_defaults = [&ue] () {
            ue.enablePrcdBf = false;
        };

        for (uint16_t i = 0; i < pmi_bf_pdu.num_prgs; i++) {
#if DBG_PRECODER
            NVLOGC_FMT(TAG, "{} PMI Index={} cell Index={} final pmidx={}", __FUNCTION__, pmi_bf_pdu.pm_idx_and_beam_idx[i + offset], cell_index, pmi_bf_pdu.pm_idx_and_beam_idx[i + offset] | cell_index << 16);
#endif
            if (pmi_bf_pdu.pm_idx_and_beam_idx[i + offset] == 0 || pmi_bf_pdu.dig_bf_interfaces == 0) {  // dig_bf_interfaces == 0 means dynamic beamforming or ZP-CSI-RS, which doesn't need precoding
                offset+= (pmi_bf_pdu.dig_bf_interfaces + 1);
                restore_defaults();
                continue;
            }
            uint32_t pmi = pmi_bf_pdu.pm_idx_and_beam_idx[i + offset] | cell_index << 16; /// PMI Unused
            offset+= (pmi_bf_pdu.dig_bf_interfaces + 1);
            uint16_t matrix_index = UINT16_MAX;
            auto iter = std::find(cache.begin(), cache.end(), pmi);
            if (iter != cache.end()) {
                matrix_index = std::distance(cache.begin(), iter);
#if DBG_PRECODER
                NVLOGC_FMT(TAG, "{} PMI Index={} found in cache matrix_index={}", __FUNCTION__, pmi, matrix_index);
#endif
            } else {
                auto pmw_iter = pm_weight_map.find(pmi);
                if (pmw_iter == pm_weight_map.end()){
                    restore_defaults();
                    continue ;
                }
                auto& val = list[cell_grp.nPrecodingMatrices];
                matrix_index = cell_grp.nPrecodingMatrices;
                cache.push_back(pmi);
                val.nPorts = pmw_iter->second.weights.nPorts;
                std::copy(pmw_iter->second.weights.matrix, pmw_iter->second.weights.matrix + (pmw_iter->second.layers * pmw_iter->second.ports), val.matrix);
                cell_grp.nPrecodingMatrices++;
#if DBG_PRECODER
                NVLOGC_FMT(TAG, "{} PMI Index={} not found in cache matrix_index={}", __FUNCTION__, cache.back(), matrix_index);
                for (uint i = 0; i < pmw_iter->second.layers; i++) {
                    for (uint j = 0; j < pmw_iter->second.ports; j++) {
                        __half2& value{val.matrix[i* pmw_iter->second.ports + j]};
                        NVLOGC_FMT(TAG, " layer {} port {} index {}. real = {} imag = {}", i, j, i * pmw_iter->second.ports + j,  static_cast<float>(value.x), static_cast<float>(value.y));
                    }
                }
#endif
            }

            if (matrix_index != UINT16_MAX){
                ue.pmwPrmIdx = matrix_index;
                cell_grp.pPmwPrms = list.data();
            }
        }
    }

    std::size_t get_current_prb_index(slot_info_t* sym_prbs) {
        return !!sym_prbs->prbs_size? sym_prbs->prbs_size - 1: MAX_PRB_INFO;
    }

    inline void pdsch_csirs_all_sym_handler(slot_info_t* sym_prbs, nv::slot_detail_t* slot_detail, bool is_new_grp, cuphyPdschUeGrpPrm_t& grp, cuphyPdschUePrm_t& ue, uint16_t num_dl_prb, bool mmimo_enabled) {

        auto& prbs{sym_prbs->prbs};

        bool value = ifAnySymbolPresent(sym_prbs->symbols, DL_CHANNEL_MASK);
        if (value) {
            return;
        }
        if (is_new_grp && grp.resourceAlloc == 1) {

            check_prb_info_size(sym_prbs->prbs_size);
            prbs[sym_prbs->prbs_size] = prb_info_t(0, num_dl_prb);
            sym_prbs->prbs_size++;
            auto index = get_current_prb_index(sym_prbs);
            prb_info_t& prb_info{prbs[index]};
            prb_info.common.ap_index = 0; //Init AP_INDEX as this is the first UE in the new UE-Group
            prb_info.common.direction = fh_dir_t::FH_DIR_DL;
            prb_info.common.numApIndices = 0;
            prb_info.common.numSymbols = (slot_detail == nullptr || slot_detail->max_dl_symbols == 0 ? OFDM_SYMBOLS_PER_SLOT: slot_detail->max_dl_symbols);

            if(ue.rnti == std::numeric_limits<uint16_t>::max())
            {
                prb_info.common.portMask |= (1 << (2 * ue.nUeLayers)) - 1;
            }
            else
            {
                prb_info.common.portMask |= calculate_dmrs_port_mask(ue.dmrsPortBmsk, ue.scid, ue.nlAbove16);
                if(mmimo_enabled)
                {
                    //NVLOGD_FMT(TAG, "{}:{} ue.rnti {} ue.dmrsPortBmsk {} ue.scid {}", __FILE__, __LINE__, ue.rnti, ue.dmrsPortBmsk, ue.scid);
                    track_eaxcids_fh(ue, prb_info.common);
                }
            }
            uint8_t start_symbol = (slot_detail == nullptr ? 0 : slot_detail->start_sym_dl);
                        
            update_prb_sym_list(sym_prbs, index, start_symbol, prb_info.common.numSymbols, channel_type::PDSCH, ru_type::SINGLE_SECT_MODE);
        }
    }

    void update_prc_fh_params_pdsch_with_csirs_mod_comp(const pm_weight_map_t & pm_map, const IFhCallbackContext& fh_context, const PdschFhParamsView& pdsch_fh_param, nv::slot_detail_t* slot_detail, ru_type ru, comp_method dl_comp_method, uint8_t num_csirs_eaxcids, bool csirs_compact_mode) {
        auto& grp = pdsch_fh_param.grp();
        bool is_new_grp = pdsch_fh_param.is_new_grp();
        auto & ue = pdsch_fh_param.ue();
        uint16_t num_dl_prb = pdsch_fh_param.num_dl_prb();
        auto cell_index = pdsch_fh_param.cell_index();

        auto numSym = grp.nPdschSym;
        auto pdschSym = grp.pdschStartSym;
        auto sym_prbs{pdsch_fh_param.sym_prb_info()};

        if (!is_new_grp) {
            return ;
        }

        switch(ru) {
            case ru_type::SINGLE_SECT_MODE:
                pdsch_csirs_all_sym_handler(sym_prbs, slot_detail, is_new_grp, grp, ue, num_dl_prb, pdsch_fh_param.mmimo_enabled());
                break;
            case ru_type::MULTI_SECT_MODE:
            case ru_type::OTHER_MODE: {
 
                uint16_t pdschSymMask = ((1 << numSym) - 1) << pdschSym;
                uint16_t pdschOnlySymMask = pdschSymMask & ~grp.dmrsSymLocBmsk;
                uint16_t csirsOnlySymMask = fh_context.csirs_symbol_map(cell_index);
                pdschOnlySymMask &= ~csirsOnlySymMask;

                ru_type ru = ru_type::OTHER_MODE;
                auto channel = channel_type::NONE;
                NVLOGD_FMT(TAG, "{} pdschSymMask {} pdschOnlySymMask {} csirsOnlySymMask {} dmrsSymLocBmsk {} \n", __FUNCTION__, pdschSymMask, pdschOnlySymMask, csirsOnlySymMask, grp.dmrsSymLocBmsk);
                auto func = [&pm_map, sym_prbs, &fh_context, &pdsch_fh_param, &slot_detail, &ru, &channel](uint16_t start, uint16_t length ) {
                    auto startPrbIndex = get_current_prb_index(sym_prbs);
                    auto endPrbIndex = startPrbIndex;
                    handleNewPdschSegment(start, length, pm_map, fh_context, pdsch_fh_param, slot_detail, ru, channel);
                    endPrbIndex = get_current_prb_index(sym_prbs);
                    if (endPrbIndex != startPrbIndex && startPrbIndex == MAX_PRB_INFO) { 
                        startPrbIndex = 0;
                    } else {
                        startPrbIndex++;
                    }
                    NVLOGD_FMT(TAG, "start {} length {} startPrbIndex {} endPrbIndex {}\n", start, length, startPrbIndex, endPrbIndex);
                    for (auto index = startPrbIndex; index <= endPrbIndex; index++) {
                        update_prb_sym_list(sym_prbs, index, start, length, channel_type::PDSCH, ru);
                    }
                };

                auto func_pdsch_csirs = [&pm_map, sym_prbs, &fh_context, &pdsch_fh_param, &slot_detail, &ru, &channel, num_csirs_eaxcids, csirs_compact_mode](uint16_t start, uint16_t length ) {
                    handleNewPdschCsirsSegment(start, length, pm_map, fh_context, pdsch_fh_param, slot_detail, ru, channel, num_csirs_eaxcids, csirs_compact_mode);
                };

                switch (dl_comp_method)
                {
                    case comp_method::MODULATION_COMPRESSION: {
                        channel = channel_type::PDSCH_DMRS;
                        processConsecutiveBits(grp.dmrsSymLocBmsk, func);
                        channel = channel_type::PDSCH;
                        processConsecutiveBits(pdschOnlySymMask, func);
                    }
                        break;
                    case comp_method::NO_COMPRESSION:
                    case comp_method::BLOCK_FLOATING_POINT: {
                        channel = channel_type::PDSCH;
                        processConsecutiveBits(pdschSymMask & ~csirsOnlySymMask, func);
                        break;
                    }
                    default:
                        break;
                }

                channel = channel_type::PDSCH_CSIRS;
                processSetBits(csirsOnlySymMask, func_pdsch_csirs);
            }
            default:
                break;

        }
    }

    inline void update_prc_fh_params_pdsch_mod_comp(const pm_weight_map_t & pm_map, const IFhCallbackContext& fh_context, const PdschFhParamsView& pdsch_fh_param, nv::slot_detail_t* slot_detail, ru_type ru, comp_method dl_comp_method) {

        cuphyPdschUeGrpPrm_t& grp = pdsch_fh_param.grp();
        bool is_new_grp = pdsch_fh_param.is_new_grp();
        cuphyPdschUePrm_t& ue = pdsch_fh_param.ue();
        uint16_t num_dl_prb = pdsch_fh_param.num_dl_prb();

        if (!is_new_grp) {
            return ;
        }
        auto numSym = grp.nPdschSym;
        auto pdschSym = grp.pdschStartSym;
        auto sym_prbs{pdsch_fh_param.sym_prb_info()};
        auto tempPdschSym = pdschSym;
        auto tempPdschNum = 0;
        auto prevChanType = channel_type::NONE;
        auto currentChanType = channel_type::NONE;
        switch(ru) {
            case ru_type::SINGLE_SECT_MODE:
                pdsch_csirs_all_sym_handler(sym_prbs, slot_detail, is_new_grp, grp, ue, num_dl_prb, pdsch_fh_param.mmimo_enabled());
                break;
            case ru_type::MULTI_SECT_MODE:
            case ru_type::OTHER_MODE: {
                uint16_t pdschSymMask = ((1 << numSym) - 1) << pdschSym;
                uint16_t pdschOnlySymMask = pdschSymMask & ~grp.dmrsSymLocBmsk;
                uint16_t csirsOnlySymMask = 0;
                
                auto channel = channel_type::PDSCH_DMRS;
                NVLOGD_FMT(TAG, " {} pdschSymMask {} pdschOnlySymMask {} csirsOnlySymMask {}\n", __FUNCTION__, pdschSymMask, pdschOnlySymMask, csirsOnlySymMask);
                auto func = [&pm_map, sym_prbs, &fh_context, &pdsch_fh_param, &slot_detail, &ru, &channel](uint16_t start, uint16_t length ) {
                    auto startPrbIndex = get_current_prb_index(sym_prbs);
                    auto endPrbIndex = startPrbIndex;
                    handleNewPdschSegment(start, length, pm_map, fh_context, pdsch_fh_param, slot_detail, ru, channel);
                    endPrbIndex = get_current_prb_index(sym_prbs);
                    if (endPrbIndex != startPrbIndex && startPrbIndex == MAX_PRB_INFO) { 
                        startPrbIndex = 0;
                    } else {
                        startPrbIndex++;
                    }
                    NVLOGD_FMT(TAG, "start {} length {} startPrbIndex {} endPrbIndex {}\n", start, length, startPrbIndex, endPrbIndex);
                    for (auto index = startPrbIndex; index <= endPrbIndex; index++) {
                        update_prb_sym_list(sym_prbs, index, start, length, channel_type::PDSCH, ru);
                    }
                };

                switch (dl_comp_method) {
                    case comp_method::MODULATION_COMPRESSION:
                        channel = channel_type::PDSCH_DMRS;
                        processConsecutiveBits(grp.dmrsSymLocBmsk, func);
                        channel = channel_type::PDSCH;
                        processConsecutiveBits(pdschOnlySymMask, func);
                        break;
                    case comp_method::NO_COMPRESSION:
                    case comp_method::BLOCK_FLOATING_POINT:
                        channel = channel_type::PDSCH;
                        processConsecutiveBits(pdschSymMask, func);
                        break;
                    default:
                        break;
                }
            }
            default:
                break;

        }
    }

    template <bool mplane_configured_ru_type>
    void update_prc_fh_params_pdsch(const pm_weight_map_t & pm_map, const IFhCallbackContext& fh_context, const PdschFhParamsView& pdsch_fh_param, nv::slot_detail_t* slot_detail)
    {
        ru_type ru;
        aerial_fh::UserDataCompressionMethod dl_comp_method = aerial_fh::UserDataCompressionMethod::BLOCK_FLOATING_POINT;
        if (mplane_configured_ru_type)
        {
            nv::PHYDriverProxy& phyDriver = nv::PHYDriverProxy::getInstance();
            const auto & mplane_info = phyDriver.getMPlaneConfig(pdsch_fh_param.cell_index());
            dl_comp_method = mplane_info.dl_comp_meth;
            ru = mplane_info.ru;
        }
        else
        {
            ru = OTHER_MODE;
        }

        switch (dl_comp_method) {
            case comp_method::MODULATION_COMPRESSION:
                update_prc_fh_params_pdsch_mod_comp(pm_map, fh_context, pdsch_fh_param, slot_detail, ru, comp_method::MODULATION_COMPRESSION);
                break;
            case comp_method::NO_COMPRESSION:
            case comp_method::BLOCK_FLOATING_POINT: {
            cuphyPdschUeGrpPrm_t& grp = pdsch_fh_param.grp();
            bool is_new_grp = pdsch_fh_param.is_new_grp();
            cuphyPdschUePrm_t& ue = pdsch_fh_param.ue();
            auto& pmi_bf_pdu = pdsch_fh_param.pc_bf();
            // pm_weight_map_t & pm_map = pm_map
            auto& slot_3gpp = pdsch_fh_param.slot_3gpp();
            bool bf_enabled = pdsch_fh_param.bf_enabled();
            bool pm_enabled = pdsch_fh_param.pm_enabled();
            bool mmimo_enabled = pdsch_fh_param.mmimo_enabled();
            int32_t cell_index = pdsch_fh_param.cell_index();
            uint32_t ue_grp_index = pdsch_fh_param.ue_grp_index();
            uint32_t ue_grp_bfw_index_per_cell = pdsch_fh_param.ue_grp_bfw_index_per_cell();
            bfw_coeff_mem_info_t* bfwCoeff_mem_info = pdsch_fh_param.bfw_coeff_mem_info();
            uint16_t num_dl_prb = pdsch_fh_param.num_dl_prb();

            // NVLOGC_FMT(TAG, "pdsch_params* info {}, cuphyPdschCellGrpDynPrm_t& cell_grp_info {},"
            // "cuphyPdschUeGrpPrm_t& grp {}, bool is_new_grp {}, cuphyPdschUePrm_t& ue {},"
            // "scf_fapi_tx_precoding_beamforming_t& pmi_bf_pdu {}, pm_weight_map_t & pm_map {},"
            // "cell_sub_command& cell_cmd {}, bool bf_enabled {}, bool pm_enabled {}, int32_t cell_index {}\n",
            // info, &cell_grp_info,
            // &grp, (is_new_grp)?1:0, &ue,
            // &pmi_bf_pdu, &pm_map,
            // &cell_cmd, bf_enabled?1:0, pm_enabled?1:0, cell_index);

            slot_info_t *sym_prbs = pdsch_fh_param.sym_prb_info();
            prb_info_t (&prbs)[MAX_PRB_INFO] = sym_prbs->prbs; 

            if(ru == SINGLE_SECT_MODE)
            {
            bool value = ifAnySymbolPresent(sym_prbs->symbols, DL_CHANNEL_MASK);
            if (value)
                return;
            }

            if (is_new_grp) {
                // NVLOGC_FMT(TAG, "PDSCH new Grp");

            if(grp.resourceAlloc == 1) //RA type 1
            {
                    check_prb_info_size(sym_prbs->prbs_size);
                    if(ru == SINGLE_SECT_MODE)
                    {
                    prbs[sym_prbs->prbs_size] = prb_info_t(0, num_dl_prb);
                    sym_prbs->prbs_size++;
                    }
                    else
                    {
                    prbs[sym_prbs->prbs_size] = prb_info_t(grp.startPrb, grp.nPrb);
                    sym_prbs->prbs_size++;
                    }

                    std::size_t index{sym_prbs->prbs_size - 1};
                    prb_info_t& prb_info{prbs[index]};
                    prb_info.common.direction = fh_dir_t::FH_DIR_DL;
                    prb_info.common.numApIndices = 0;
                    prb_info.common.ap_index = 0; //Init AP_INDEX as this is the first UE in the new UE-Group
                    if(ru == SINGLE_SECT_MODE)
                    {
                        prb_info.common.numSymbols = (slot_detail == nullptr || slot_detail->max_dl_symbols == 0 ? OFDM_SYMBOLS_PER_SLOT: slot_detail->max_dl_symbols);
                    }
                    if (likely(!pm_enabled)) {
                        if(ue.rnti == UINT16_MAX && !pdsch_fh_param.mmimo_enabled())  // For 4T4R O-RUs, SI-RNTI date with a single layer transmission will be replicated for the second antenna
                        {
                            prb_info.common.portMask |= (1 << (2 * ue.nUeLayers)) - 1;
                        }
                        else
                        {
                            prb_info.common.portMask |= calculate_dmrs_port_mask(ue.dmrsPortBmsk, ue.scid, ue.nlAbove16);
                            if(mmimo_enabled)
                            {
                                //NVLOGD_FMT(TAG, "{}:{} is_new_grp {} ue.rnti {} ue.dmrsPortBmsk {} ue.scid {}", __FILE__, __LINE__, (is_new_grp)?1:0, ue.rnti, ue.dmrsPortBmsk, ue.scid);
                                track_eaxcids_fh(ue, prb_info.common);
                            }
                        }
                    } else {
                        update_pm_weights_fh(ue, pm_map, pmi_bf_pdu, prb_info.common, cell_index, mmimo_enabled);
                    }
                    if (bf_enabled) {
                        update_beam_list(prb_info.beams_array, prb_info.beams_array_size, pmi_bf_pdu, mmimo_enabled, prb_info, cell_index);
                        if (bfwCoeff_mem_info != NULL)
                        {
                            NVLOGD_FMT(TAG, "Header {} SFN {} SLOT {} BFW_SFN {} BFW_SLOT {}", *bfwCoeff_mem_info->header, slot_3gpp.sfn_, slot_3gpp.slot_, bfwCoeff_mem_info->sfn, bfwCoeff_mem_info->slot);
                        }
                        else
                        {
                            NVLOGD_FMT(TAG, "bfwCoeff_mem_info is NULL");
                        }
                        if (bfwCoeff_mem_info != NULL && *bfwCoeff_mem_info->header == BFW_COFF_MEM_BUSY)
                        {
                            if ((is_latest_bfw_coff_avail(slot_3gpp.sfn_ , slot_3gpp.slot_,
                                    bfwCoeff_mem_info->sfn, bfwCoeff_mem_info->slot)) && (pmi_bf_pdu.dig_bf_interfaces==0))
                            {
                                NVLOGD_FMT(TAG, "PDSCH new Grp :: Header {} SFN {} SLOT {} BFW_SFN {} BFW_SLOT {}",
                                                *bfwCoeff_mem_info->header, slot_3gpp.sfn_, slot_3gpp.slot_,
                                                bfwCoeff_mem_info->sfn, bfwCoeff_mem_info->slot);
                                prb_info.common.extType = 11;
                                //prb_info.common.portMask = (1 << pmi_bf_pdu.dig_bf_interfaces) -1;
                                prb_info.bfwCoeff_buf_info.num_prgs = pmi_bf_pdu.num_prgs;
                                prb_info.bfwCoeff_buf_info.prg_size = pmi_bf_pdu.prg_size;
                                prb_info.bfwCoeff_buf_info.dig_bf_interfaces = pmi_bf_pdu.dig_bf_interfaces;
                                prb_info.bfwCoeff_buf_info.nGnbAnt = bfwCoeff_mem_info->nGnbAnt;
                                prb_info.bfwCoeff_buf_info.header = bfwCoeff_mem_info->header;
                                prb_info.bfwCoeff_buf_info.p_buf_bfwCoef_h = bfwCoeff_mem_info->buff_addr_chunk_h[ue_grp_bfw_index_per_cell];
                                prb_info.bfwCoeff_buf_info.p_buf_bfwCoef_d = bfwCoeff_mem_info->buff_addr_chunk_d[ue_grp_bfw_index_per_cell];
                                // Dump the SRS-derived dynamic BFW weights for this UE into the data lake
                                // (dyn_bfw_weights, keyed by rnti). Host coeff buffer layout per (layer,prg)
                                // bundle = [exponent byte + 2*nGnbAnt*bfwIqWidth/8 packed mantissas]. nLayers
                                // is 1 today (single-layer); wire the real layer count when SU-MIMO lands.
                                {
                                    const uint8_t  _dl_iqw    = 9;   // dynamic BFW BFP width (matches fh.cpp)
                                    const uint8_t  _dl_layers = 1;
                                    const uint8_t  _dl_ngnb   = bfwCoeff_mem_info->nGnbAnt;
                                    const uint16_t _dl_nprg   = pmi_bf_pdu.num_prgs;
                                    const uint32_t _dl_bundle = 1u + (uint32_t)(2 * _dl_ngnb * _dl_iqw) / 8u;
                                    nv::PHYDriverProxy& _dl_phy = nv::PHYDriverProxy::getInstance();
                                    l1_datalake_notify_bfw(_dl_phy.get_driver(), (uint16_t)cell_index, ue.rnti,
                                        slot_3gpp.sfn_, slot_3gpp.slot_, 0 /*beamId assigned later in FH*/,
                                        _dl_ngnb, _dl_nprg, (uint16_t)pmi_bf_pdu.prg_size, _dl_iqw, _dl_layers,
                                        (const uint8_t*)bfwCoeff_mem_info->buff_addr_chunk_h[ue_grp_bfw_index_per_cell],
                                        (uint32_t)_dl_nprg * _dl_layers * _dl_bundle);
                                }
                                NVLOGD_FMT(TAG, "PDSCH new Grp dig_bf_interfaces={}, prb_info.common.portMask {}, num_prgs={}, prg_size={}, buff_addr_chunk_h[{}]={}",
                                                pmi_bf_pdu.dig_bf_interfaces, static_cast<uint32_t>(prb_info.common.portMask), 
                                                static_cast<uint16_t>(pmi_bf_pdu.num_prgs), static_cast<uint16_t>(pmi_bf_pdu.prg_size), ue_grp_bfw_index_per_cell, reinterpret_cast<void*>(bfwCoeff_mem_info->buff_addr_chunk_h[ue_grp_bfw_index_per_cell]));

                                #if 0
                                // 35072
                                uint8_t* buffer_ptr = (uint8_t*)bfwCoeff_mem_info->buff_addr_chunk_h[ue_grp_index];
                                float2* value = (float2*)buffer_ptr;
                                uint32_t count = 0;
                                for (uint8_t i=0; i < 1 ; i++)
                                {
                                    for ( uint8_t j = 0; j < 1; j++ )
                                    {
                                        for ( uint8_t k = 0; k < 1; k++ )
                                        {
                                            NVLOGD_FMT(TAG, "####Port {} GnBlayer {} PRBG {} IQ = {} + j{} ", i, j, k, value[count].x, value[count].y);
                                            count++;
                                        }
                                    }
                                }
                                #endif
                            }
                        }
                    }
                    if(ru == SINGLE_SECT_MODE)
                    {
                        uint8_t start_symbol = (slot_detail == nullptr ? 0 : slot_detail->start_sym_dl);
                        
                        update_prb_sym_list(sym_prbs, index, start_symbol, prb_info.common.numSymbols, channel_type::PDSCH, ru);
                    }
                    else
                    {
                        if(prb_info.common.extType == ORAN_CMSG_SECTION_EXT_TYPE_11 || mmimo_enabled)
                        {
                            prb_info.common.numSymbols = grp.nPdschSym;
                        }
                        else
                        {
                            prb_info.common.numSymbols = 1;
                        }
                        update_prb_sym_list(sym_prbs, index, grp.pdschStartSym, grp.nPdschSym, channel_type::PDSCH, ru);
                    }
    #if 0
                    //DMRS - to be enabled for Mod Comp
                    uint16_t bitmap = grp.dmrsSymLocBmsk >> grp.pdschStartSym;
                    // Set dmrs_max_len to 0 to reflect old grp.dmrsMaxLen value which was also always 0.
                    // FIXME Loop below was (and still is) a no-op and should be removed or other code should be updated as needed.
                    uint8_t dmrs_max_len = 0;
                    //uint8_t dmrs_max_len = ((grp.dmrsSymLocBmsk & (grp.dmrsSymLocBmsk >> 1)) > 0) ? 2 : 1;
                    uint8_t dmrs_pos = grp.pdschStartSym;
                    while(bitmap)
                    {
                        if(1 & bitmap)
                        {
                            update_prb_sym_list(sym_prbs, index, dmrs_pos, dmrs_max_len, channel_type::PDSCH_DMRS, ru);
                            if(dmrs_max_len > 1)
                            {
                                dmrs_pos += dmrs_max_len;
                                bitmap >>= dmrs_max_len;
                                continue;
                            }
                        }
                        dmrs_pos++;
                        bitmap >>= 1;
                    }
    #endif
                } //RA type 1
                else
                { // RA type 0
                    //for(uint32_t i=0; i < MAX_RBMASK_BYTE_SIZE; i++)
                        //NVLOGD_FMT(TAG, "{}: rbBitmap[{}]=0x{:x}",__func__,i, grp.rbBitmap[i]);
                    //prepare_ra_type0_info(info,grp,ue_grp_index);
                    for(uint32_t i=0; i < fh_context.pdsch_num_ra_type0_info(ue_grp_index); i++)
                    {
                        check_prb_info_size(sym_prbs->prbs_size);
                        prbs[sym_prbs->prbs_size] =
                            prb_info_t(fh_context.pdsch_ra_type0_info(i, ue_grp_index).start_prb + grp.startPrb, fh_context.pdsch_ra_type0_info(i, ue_grp_index).num_prb);
                        sym_prbs->prbs_size++;

                        std::size_t index{sym_prbs->prbs_size - 1};
                        prb_info_t& prb_info{prbs[index]};
                        prb_info.common.direction = fh_dir_t::FH_DIR_DL;
                        prb_info.common.numApIndices = 0;
                        if (likely(!pm_enabled)) {
                            if(ue.rnti == UINT16_MAX && !pdsch_fh_param.mmimo_enabled())  // For 4T4R O-RUs, SI-RNTI date with a single layer transmission will be replicated for the second antenna
                            {
                                prb_info.common.portMask |= (1 << (2 * ue.nUeLayers)) - 1;
                            }
                            else
                            {
                                prb_info.common.portMask |= calculate_dmrs_port_mask(ue.dmrsPortBmsk, ue.scid, ue.nlAbove16);
                                if(mmimo_enabled)
                                {
                                    //NVLOGD_FMT(TAG, "{}:{} ue.rnti {} ue.dmrsPortBmsk {} ue.scid {}", __FILE__, __LINE__, ue.rnti, ue.dmrsPortBmsk, ue.scid);
                                    track_eaxcids_fh(ue, prb_info.common);
                                }
                            }
                        } else {
                            update_pm_weights_fh(ue, pm_map, pmi_bf_pdu, prb_info.common, cell_index, mmimo_enabled);
                        }
                        if (bf_enabled) {
                            update_beam_list(prb_info.beams_array, prb_info.beams_array_size, pmi_bf_pdu, mmimo_enabled, prb_info, cell_index);
                        }
                        update_prb_sym_list(*sym_prbs, index, grp.pdschStartSym, grp.nPdschSym, channel_type::PDSCH, ru);
    #if 0
                        //DMRS - to be enabled for Mod Comp
                        uint16_t dmrs_bitmap = grp.dmrsSymLocBmsk >> grp.pdschStartSym;
                        // Set dmrs_max_len to 0 to reflect old grp.dmrsMaxLen value which was also always 0.
                        // FIXME Loop below was (and still is) a no-op and should be removed or other code should be updated as needed.
                        uint8_t dmrs_max_len = 0;
                        //uint8_t dmrs_max_len = ((grp.dmrsSymLocBmsk & (grp.dmrsSymLocBmsk >> 1)) > 0) ? 2 : 1;
                        uint8_t dmrs_pos = grp.pdschStartSym;
                        while(dmrs_bitmap)
                        {
                            if(1 & dmrs_bitmap)
                            {
                                update_prb_sym_list(*sym_prbs, index, dmrs_pos, dmrs_max_len, channel_type::PDSCH_DMRS, ru);
                                if(dmrs_max_len > 1)
                                {
                                    dmrs_pos += dmrs_max_len;
                                    dmrs_bitmap >>= dmrs_max_len;
                                    continue;
                                }
                            }
                            dmrs_pos++;
                            dmrs_bitmap >>= 1;
                        }
    #endif
                    }//for num_ra_type0_info
                } // RA type 0
            }//new group = true
            else
            {
                if(grp.resourceAlloc == 1) //RA type 1
                {
                    auto& idxlist{sym_prbs->symbols[grp.pdschStartSym][channel_type::PDSCH]};

                    auto iter = std::find_if(idxlist.begin(), idxlist.end(),[&sym_prbs, &prbs, &grp, &ru, &num_dl_prb](const auto& e){
                        bool retval = (e < sym_prbs->prbs_size);
                        if (retval) {
                            auto& prb{prbs[e]};
                            if(ru == SINGLE_SECT_MODE)
                            {
                                retval = (prb.common.startPrbc == 0 && prb.common.numPrbc == num_dl_prb);
                            }
                            else
                            {
                                retval = (prb.common.startPrbc == grp.startPrb && prb.common.numPrbc == grp.nPrb);
                            }
                        }
                        return retval;
                    });

                    if (iter != idxlist.end()){
                        auto& prb{prbs[*iter]};
                        if (likely(!pm_enabled)) {
                                //NVLOGD_FMT(TAG, "{}:{} old prb.common.portMask {}", __FILE__, __LINE__, ue.rnti, static_cast<uint32_t>(prb.common.portMask), static_cast<uint64_t>(ue.dmrsPortBmsk), ue.scid);
                                prb.common.portMask |= calculate_dmrs_port_mask(ue.dmrsPortBmsk, ue.scid, ue.nlAbove16);
                                if(mmimo_enabled)
                                {
                                    //NVLOGD_FMT(TAG, "{}:{} not pm_enabled ue.rnti {} prb_info.common.portMask {}, ue.dmrsPortBmsk {}, ue.scid {}", __FILE__, __LINE__, ue.rnti, static_cast<uint32_t>(prb.common.portMask), static_cast<uint64_t>(ue.dmrsPortBmsk), ue.scid);
                                    track_eaxcids_fh(ue, prb.common);
                                }
                        } else {
                            update_pm_weights_fh(ue, pm_map, pmi_bf_pdu, prb.common, cell_index, mmimo_enabled);
                        }
                        //NVLOGD_FMT(TAG, "PDSCH old Grp dig_bf_interfaces={}, prb_info.common.portMask {}",
                        //                        pmi_bf_pdu.dig_bf_interfaces, static_cast<uint32_t>(prb.common.portMask));
                    }
                }//RA type 1
                else
                {
                    auto& idxlist{sym_prbs->symbols[grp.pdschStartSym][channel_type::PDSCH]};
                    for(uint32_t i=0; i < fh_context.pdsch_num_ra_type0_info(ue_grp_index); i++)
                    {
                        uint32_t startPrb = fh_context.pdsch_ra_type0_info(i, ue_grp_index).start_prb + grp.startPrb;
                        uint32_t nPrb = fh_context.pdsch_ra_type0_info(i, ue_grp_index).num_prb;
                        auto iter = std::find_if(idxlist.begin(), idxlist.end(),[&sym_prbs, &prbs, startPrb, nPrb](const auto& e){
                            bool retval = (e < sym_prbs->prbs_size);
                            if (retval) {
                                auto& prb{prbs[e]};
                                retval = (prb.common.startPrbc == startPrb && prb.common.numPrbc == nPrb);
                            }
                            return retval;
                        });

                        if (iter != idxlist.end()){
                            auto& prb{prbs[*iter]};
                            if (likely(!pm_enabled)) {
                                prb.common.portMask |= calculate_dmrs_port_mask(ue.dmrsPortBmsk, ue.scid, ue.nlAbove16);
                                if(mmimo_enabled)
                                {
                                    //NVLOGD_FMT(TAG, "{}:{} ue.rnti {} ue.dmrsPortBmsk {} ue.scid {}", __FILE__, __LINE__, ue.rnti, ue.dmrsPortBmsk, ue.scid);
                                    track_eaxcids_fh(ue, prb.common);
                                }
                            } else {
                                update_pm_weights_fh(ue, pm_map, pmi_bf_pdu, prb.common, cell_index, mmimo_enabled);
                            }
                        }
                    }//for 0->num_ra_type0_info
                }//RA type 0
            }//new group = false

            }
        }
    }

    /**
     * @brief Encode static beamforming weights for a CSI-RS beam into prb_info.
     *
     * Checks the DBT (dynamic beamforming table) for the given beam_id and, if
     * present, populates the static BFW coefficient metadata in prb_info.  When
     * sendOncePerBeam is true the beam-sent flag is set so that subsequent calls
     * for the same beam are skipped.  In per-port iteration mode
     * (modcomp_enabled == true) the flag is deferred until the last port to
     * avoid premature skipping of later ports.
     *
     * @param[in]     cell_index           Cell index.
     * @param[in]     pdu                  Precoding / beamforming PDU.
     * @param[in,out] prb_info             PRB info to update with BFW metadata.
     * @param[in]     beam_id              Beam identifier to look up in DBT.
     * @param[in]     modcomp_enabled      True when per-port sent-flag gating is needed.
     * @param[in]     phyDriver            PHY driver proxy reference.
     * @param[in]     static_bfw_configured Whether static BFW is configured.
     * @param[in]     sendOncePerBeam      If true, mark beam as sent after encoding.
     */
    inline void update_static_bf_wt_csi_rs_impl(int32_t cell_index, tx_precoding_beamforming_t& pdu, prb_info_t& prb_info, uint16_t beam_id, bool modcomp_enabled,
        nv::PHYDriverProxy& phyDriver, bool static_bfw_configured, bool sendOncePerBeam)
    {
        if (!static_bfw_configured)
        {
            NVLOGD_FMT(TAG, "{} Static Beamforming is disabled for this PDU prb_info={}", __func__, reinterpret_cast<void*>(&prb_info.common));
            return;
        }

        NVLOGD_FMT(TAG, "num_prgs={}, dig_bf_interfaces={}", static_cast<uint16_t>(pdu.num_prgs), static_cast<uint8_t>(pdu.dig_bf_interfaces));

        // Check if beamId is part of static DBT table
        // Return values: -1 = not in DBT table (predefined beam), 0 = in DBT table but not sent, 1 = in DBT table and sent
        const int isbeamInDBT = phyDriver.l1_getBeamWeightsSentFlag(cell_index, beam_id);

        if (isbeamInDBT == -1)
        {
            NVLOGD_FMT(TAG, "Beamidx={} is a predefined beam ID (not in static DBT table)", beam_id);
            return;
        }

        if (sendOncePerBeam && (isbeamInDBT == 1))
        {
            // Beam weights already sent, skip
            return;
        }

        NVLOGD_FMT(TAG, "Beamidx={} entry available in DBT PDU IQ sent using extType=11 prb_info={}", beam_id, reinterpret_cast<void*>(&prb_info.common));
        prb_info.common.extType = 11;
        prb_info.common.isStaticBfwEncoded = true;
        prb_info.static_bfwCoeff_buf_info.num_prgs = pdu.num_prgs;
        prb_info.static_bfwCoeff_buf_info.prg_size = pdu.prg_size;
        prb_info.static_bfwCoeff_buf_info.dig_bf_interfaces = pdu.dig_bf_interfaces;

        if (!sendOncePerBeam)
        {
            return;
        }

        if (!modcomp_enabled)
        {
            phyDriver.l1_setBeamWeightsSentFlag(cell_index, beam_id);
            return;
        }

        // In the case of ModComp, this function is called per-port; if we mark sent on the first port,
        // all beam IDs are treated as sent and later ports would skip static BFW encoding.
        // Gate "mark sent" on the last port.
        const uint64_t last_port_mask = (1ULL << (pdu.dig_bf_interfaces - 1U));
        if (prb_info.common.portMask == last_port_mask)
        {
            NVLOGD_FMT(TAG, "CSI-RS BFW gate: mark sent beam_id={} portMask=0x{:x}",
                beam_id,
                static_cast<uint64_t>(prb_info.common.portMask));
            // Mark as sent once the final CSI-RS port is processed
            phyDriver.l1_setBeamWeightsSentFlag(cell_index, beam_id);
        }
    }

    /**
     * @brief Convenience wrapper for update_static_bf_wt_csi_rs_impl.
     *
     * Fetches PHY driver state (static BFW config flag, sendOncePerBeam policy)
     * and delegates to update_static_bf_wt_csi_rs_impl.
     *
     * @param[in]     cell_index       Cell index.
     * @param[in]     pdu              Precoding / beamforming PDU.
     * @param[in,out] prb_info         PRB info to update with BFW metadata.
     * @param[in]     beam_id          Beam identifier to look up in DBT.
     * @param[in]     modcomp_enabled  True when per-port sent-flag gating is needed.
     */
    void update_static_bf_wt_csi_rs(int32_t cell_index, tx_precoding_beamforming_t& pdu, prb_info_t& prb_info, uint16_t beam_id, bool modcomp_enabled)
    {
        if (pdu.dig_bf_interfaces == 0U)
        {
            NVLOGD_FMT(TAG, "{} Static Beamforming is disabled for this PDU prb_info={}", __func__, reinterpret_cast<void*>(&prb_info.common));
            return;
        }
        nv::PHYDriverProxy& phyDriver = nv::PHYDriverProxy::getInstance();
        const bool static_bfw_configured = phyDriver.l1_staticBFWConfigured(cell_index);
        const bool sendOncePerBeam = !phyDriver.l1_get_send_static_bfw_wt_all_cplane();
        update_static_bf_wt_csi_rs_impl(cell_index, pdu, prb_info, beam_id, modcomp_enabled, phyDriver, static_bfw_configured, sendOncePerBeam);
    }

   template <bool mplane_configured_ru_type>
    void update_prc_fh_params_pdsch_with_csirs(const pm_weight_map_t & pm_map, const IFhCallbackContext& fh_context, const PdschFhParamsView& pdsch_fh_param, nv::slot_detail_t* slot_detail, bool csirs_compact_mode)
    {
        if(pdsch_fh_param.mmimo_enabled())
            update_prc_fh_params_pdsch_with_csirs_mmimo<mplane_configured_ru_type>(pm_map, fh_context, pdsch_fh_param, slot_detail, csirs_compact_mode);
        else
            update_prc_fh_params_pdsch_with_csirs_nonmmimo<mplane_configured_ru_type>(pm_map, fh_context, pdsch_fh_param, slot_detail, csirs_compact_mode);
    }

    // non mimo_enabled case
    template <bool mplane_configured_ru_type>
    void update_prc_fh_params_pdsch_with_csirs_nonmmimo(const pm_weight_map_t & pm_map, const IFhCallbackContext& fh_context, const PdschFhParamsView& pdsch_fh_param, nv::slot_detail_t *slot_detail, bool csirs_compact_mode) 
    {
    
        const cuphyPdschCellGrpDynPrm_t& cell_grp_info = fh_context.pdsch_cell_grp_info();
        cuphyPdschUeGrpPrm_t& grp = pdsch_fh_param.grp(); 
        bool is_new_grp = pdsch_fh_param.is_new_grp(); 
        cuphyPdschUePrm_t& ue = pdsch_fh_param.ue();
        auto& pmi_bf_pdu = pdsch_fh_param.pc_bf();
        auto& slot_3gpp = pdsch_fh_param.slot_3gpp();
        bool bf_enabled = pdsch_fh_param.bf_enabled();
        bool pm_enabled = pdsch_fh_param.pm_enabled(); 
        int32_t cell_index = pdsch_fh_param.cell_index();
        uint16_t num_dl_prb = pdsch_fh_param.num_dl_prb();
        uint32_t ue_grp_index = pdsch_fh_param.ue_grp_index();
        uint32_t ue_grp_bfw_index_per_cell = pdsch_fh_param.ue_grp_bfw_index_per_cell();
        bfw_coeff_mem_info_t* bfwCoeff_mem_info = pdsch_fh_param.bfw_coeff_mem_info();

        ru_type ru;
        aerial_fh::UserDataCompressionMethod dl_comp_method = aerial_fh::UserDataCompressionMethod::BLOCK_FLOATING_POINT;
        if (mplane_configured_ru_type)
        {
            nv::PHYDriverProxy& phyDriver = nv::PHYDriverProxy::getInstance();
            const auto& mplane_info = phyDriver.getMPlaneConfig(cell_index);
            ru = mplane_info.ru;
        }
        else
        {
            ru = OTHER_MODE;
        }

        // NVLOGC_FMT(TAG, "pdsch_params* pdsch_info {}, cuphyPdschCellGrpDynPrm_t& cell_grp_info {},"
        // "cuphyPdschUeGrpPrm_t& grp {}, bool is_new_grp {}, cuphyPdschUePrm_t& ue {},"
        // "scf_fapi_tx_precoding_beamforming_t& pmi_bf_pdu {}, pm_weight_map_t & pm_map {},"
        // "cell_sub_command& cell_cmd {}, bool bf_enabled {}, bool pm_enabled {}, int32_t cell_index {}, csirs_params* csirs_info {}, uint16_t num_dl_prb {}\n",
        // pdsch_info, &cell_grp_info,
        // &grp, (is_new_grp)?1:0, &ue,
        // &pmi_bf_pdu, &pm_map,
        // &cell_cmd, bf_enabled?1:0, pm_enabled?1:0, cell_index, csirs_info, num_dl_prb);
        auto sym_prbs{pdsch_fh_param.sym_prb_info()};
        auto& prbs{sym_prbs->prbs};
        //slot_command_api::prb_info_common_t * csi_rs_prb_common = nullptr;
        int  prb_index= 0;
        uint64_t csirs_portMask = 0;
        const uint16_t csirs_symbol_map = fh_context.csirs_symbol_map(cell_index);
        //TI_GENERIC_INIT("pdsch_with_csirs",15);

        if (is_new_grp)
        {
            uint32_t numSym = grp.nPdschSym;
            uint32_t pdschSym = grp.pdschStartSym;
            uint32_t tempPdschSym = 0;
            uint32_t tempNumSym = 0;
            /* pdsch_only is an array of 14 symbols with intention to record - 
            0xFF000000 - highest order byte - PDSCH exists
            0x00FF0000 - second highest order byte - PDSCH start symbol
            0x0000FFFF - lower order 2 bytes - num of consecutive PDSCH symbols at the PDSCH start symbol */
            std::array <uint32_t, 14> pdsch_only = {0}; // lower 16 bits have the num symbols, higher 16 bits has the start symbol
            uint32_t index = 0; //index into pdsch_only
            while(numSym--) {
                uint32_t mask = 1 << pdschSym;
                //NVLOGD_FMT(TAG, "upate_fh_params_pdsch_with_csirs: cell_index={} numSym={} mask=0x{:x} csirs-symbolMap=0x{:x} start rb {} num rb {}",cell_index,numSym,mask,static_cast<uint16_t>(csirs_info->symbolMapArray[cell_index]),grp.startPrb,grp.nPrb );
                if ((mask & csirs_symbol_map) != 0U)
                {
                    //TI_GENERIC_ADD("pdsch_csirs symbol");
                    ++index;
                    //NVLOGC_FMT(TAG, "upate_fh_params_pdsch_with_csirs: cell_index={} numSym={} mask=0x{:x} csirs-symbolMap=0x{:x} start rb {} num rb {} sym match",cell_index,numSym,mask,static_cast<uint16_t>(csirs_info->symbolMapArray[cell_index]),grp.startPrb,grp.nPrb );
                    prb_info_idx_list_t & csi_rs_prb_info = sym_prbs->symbols[pdschSym][channel_type::CSI_RS];
                    auto csi_rs_prb_info_size = csi_rs_prb_info.size();
                    if(grp.resourceAlloc == 1) //RA type 1
                    {
                        uint16_t rb_idx = grp.startPrb;
                        uint16_t csirs_remap = fh_context.csirs_remap(cell_index, pdschSym * num_dl_prb + rb_idx) & 0xFFF;
                        bool use_alt_prb = (fh_context.csirs_remap(cell_index, pdschSym * num_dl_prb + rb_idx) & (1 << 15)) ? true : false;
                        uint32_t tempnPrb = 0;
                        uint32_t tempStartPrb = rb_idx;
                        uint32_t counter = grp.nPrb;
                        bool isPdschSplitAcrossPrbInfo = false;
                        while(counter) {
                            while(((csirs_remap == (fh_context.csirs_remap(cell_index, pdschSym * num_dl_prb + rb_idx) & 0xFFF)) ||
                                (use_alt_prb && (csirs_remap == (fh_context.csirs_remap(cell_index, pdschSym * num_dl_prb + rb_idx + 1) & 0xFFF))))
                                && (counter > 0)) {
                                    //NVLOGD_FMT(TAG, "csirs_remap={} csirs_info->reMap={}",csirs_remap, (csirs_info->reMap[cell_index][pdschSym*num_dl_prb + rb_idx] & 0xFFF));
                                tempnPrb++; rb_idx++; counter--;
                                continue;
                            }

                            if(counter)
                            {
                                isPdschSplitAcrossPrbInfo = true;
                            }

                            //NVLOGC_FMT(TAG, "upate_fh_params_pdsch_with_csirs: cell_index={} counter={} isPdschSplitAcrossPrbInfo={} tempnPrb={} numSym={} mask=0x{:x} csirs-symbolMap=0x{:x} start rb {} num rb {} re match",
                                            //cell_index,counter,isPdschSplitAcrossPrbInfo,tempnPrb,numSym,mask,static_cast<uint16_t>(csirs_info->symbolMapArray[cell_index]),grp.startPrb,grp.nPrb );
                            check_prb_info_size(sym_prbs->prbs_size);
                            prbs[sym_prbs->prbs_size] = (use_alt_prb) ? prb_info_t(tempStartPrb, ((tempnPrb + 1) >> 1)): prb_info_t(tempStartPrb, tempnPrb);
                            sym_prbs->prbs_size++;
                            std::size_t index_pdsch_csirs{sym_prbs->prbs_size - 1};
                            prb_info_t& prb_info{prbs[sym_prbs->prbs_size - 1]};
                            prb_info.common.direction = fh_dir_t::FH_DIR_DL;
                            prb_info.common.numApIndices = 0;
                            prb_info.common.reMask = ~csirs_remap;
                            prb_info.common.useAltPrb = use_alt_prb;
                            prb_info.common.isPdschSplitAcrossPrbInfo = isPdschSplitAcrossPrbInfo;
                            if (likely(!pm_enabled)) {
                                prb_info.common.portMask |= calculate_dmrs_port_mask(ue.dmrsPortBmsk, ue.scid, ue.nlAbove16);
                            } else {
                                update_pm_weights_fh(ue, pm_map, pmi_bf_pdu, prb_info.common, cell_index, false);
                            }
                            if (bf_enabled) {
                                update_beam_list(prb_info.beams_array, prb_info.beams_array_size, pmi_bf_pdu, false, prb_info, cell_index);

                                if (bfwCoeff_mem_info != NULL && *bfwCoeff_mem_info->header == BFW_COFF_MEM_BUSY)
                                {
                                    NVLOGD_FMT(TAG, "PDSCH with CSI-RS new Grp :: Header {} SFN {} SLOT {} BFW_SFN {} BFW_SLOT {}",
                                                *bfwCoeff_mem_info->header, slot_3gpp.sfn_, slot_3gpp.slot_,
                                                bfwCoeff_mem_info->sfn, bfwCoeff_mem_info->slot);
                                    if ((is_latest_bfw_coff_avail(slot_3gpp.sfn_ , slot_3gpp.slot_,
                                            bfwCoeff_mem_info->sfn, bfwCoeff_mem_info->slot) && (pmi_bf_pdu.dig_bf_interfaces==0)))
                                    {
                                        prb_info.common.extType = 11;
                                        prb_info.bfwCoeff_buf_info.num_prgs = pmi_bf_pdu.num_prgs;
                                        prb_info.bfwCoeff_buf_info.prg_size = pmi_bf_pdu.prg_size;
                                        prb_info.bfwCoeff_buf_info.dig_bf_interfaces = pmi_bf_pdu.dig_bf_interfaces;
                                        prb_info.bfwCoeff_buf_info.nGnbAnt = bfwCoeff_mem_info->nGnbAnt;
                                        prb_info.bfwCoeff_buf_info.header = bfwCoeff_mem_info->header;
                                        prb_info.bfwCoeff_buf_info.p_buf_bfwCoef_h = bfwCoeff_mem_info->buff_addr_chunk_h[ue_grp_bfw_index_per_cell];
                                        prb_info.bfwCoeff_buf_info.p_buf_bfwCoef_d = bfwCoeff_mem_info->buff_addr_chunk_d[ue_grp_bfw_index_per_cell];
                                        NVLOGD_FMT(TAG, "PDSCH with CSI-RS new Grp dig_bf_interfaces={}, prb_info.common.portMask {}",
                                                         pmi_bf_pdu.dig_bf_interfaces, static_cast<uint32_t>(prb_info.common.portMask));
                                    }
                                }
                            }

                            //Code to add CSI_RS antenna ports which are not covered by overlapping PDSCH
                            prb_index= 0;
                            while(csi_rs_prb_info_size > prb_index)
                            {
                                if((csirs_remap & prbs[csi_rs_prb_info[prb_index]].common.reMask) == prbs[csi_rs_prb_info[prb_index]].common.reMask)
                                {
                                    prb_info.common.portMask |= prbs[csi_rs_prb_info[prb_index]].common.portMask;
                                    break;
                                }
                                ++prb_index;
                            }
                            update_prb_sym_list(*sym_prbs, index_pdsch_csirs, pdschSym, 1, channel_type::PDSCH_CSIRS, ru);
                            NVLOGD_FMT(TAG, "PDSCH with CSI-RS new Grp : at symbol={} added index={} startPrb={} numPrb={} nSym=1 prb_info={}",
                                    pdschSym, index_pdsch_csirs, tempStartPrb, tempnPrb, static_cast<void *>(&prb_info.common));

                            if(use_alt_prb && tempnPrb > 2)
                            {
                                prbs[sym_prbs->prbs_size] = prb_info_t(tempStartPrb + 1, (tempnPrb >> 1));
                                std::size_t index_pdsch{sym_prbs->prbs_size};
                                prb_info_t& prb_info{prbs[sym_prbs->prbs_size]};
                                prb_info_t& prb_info_ref{prbs[sym_prbs->prbs_size - 1]};
                                ++(sym_prbs->prbs_size);
                                prb_info.common.direction = prb_info_ref.common.direction;
                                prb_info.common.numApIndices = prb_info_ref.common.numApIndices;
                                prb_info.common.useAltPrb = prb_info_ref.common.useAltPrb;
                                prb_info.common.portMask = prb_info_ref.common.portMask;
                                prb_info.beams_array = prb_info_ref.beams_array;
                                prb_info.beams_array_size = prb_info_ref.beams_array_size;
                                prb_info.common.reMask = 0xFFF;
                                update_prb_sym_list(*sym_prbs, index_pdsch, pdschSym, 1, channel_type::PDSCH, ru);
                                NVLOGD_FMT(TAG, "PDSCH with CSIRS new Grp but PDSCH only: at symbol={} added index={} startPrb={} numPrb={} nSym=1",
                                        pdschSym, index_pdsch, tempStartPrb+1, tempnPrb>>1);

                            }

                            tempStartPrb = rb_idx;
                            tempnPrb = 0;
                            csirs_remap = fh_context.csirs_remap(cell_index, pdschSym * num_dl_prb + rb_idx) & 0xFFF;
                            use_alt_prb = (fh_context.csirs_remap(cell_index, pdschSym * num_dl_prb + rb_idx) & (1 << 15)) ? true : false;
            #if 0
                            //DMRS
                            uint16_t bitmap = grp.dmrsSymLocBmsk >> grp.pdschStartSym;
                            uint8_t dmrs_pos = grp.pdschStartSym;
                            while(bitmap)
                            {
                                if(1 & bitmap)
                                {
                                    update_prb_sym_list(*sym_prbs, index, dmrs_pos, grp.pDmrsDynPrm->dmrsMaxLen, channel_type::PDSCH_DMRS);
                                    if(grp.pDmrsDynPrm->dmrsMaxLen > 1)
                                    {
                                        dmrs_pos += grp.pDmrsDynPrm->dmrsMaxLen;
                                        bitmap >>= grp.pDmrsDynPrm->dmrsMaxLen;
                                        continue;
                                    }
                                }
                                dmrs_pos++;
                                bitmap >>= 1;
                            }
            #endif
                        } //while(counter)
                    } // RA type 1
                    else
                    { // RA type 0
                        for(uint32_t i=0; i < fh_context.pdsch_num_ra_type0_info(ue_grp_index); i++)
                        {
                            uint16_t rb_idx = fh_context.pdsch_ra_type0_info(i, ue_grp_index).start_prb + grp.startPrb;
                            uint16_t csirs_remap = fh_context.csirs_remap(cell_index, pdschSym * num_dl_prb + rb_idx) & 0xFFF;
                            bool use_alt_prb = (fh_context.csirs_remap(cell_index, pdschSym * num_dl_prb + rb_idx) & (1 << 15)) ? true : false;
                            uint32_t tempnPrb = 0;
                            uint32_t tempStartPrb = rb_idx;
                            uint32_t counter = fh_context.pdsch_ra_type0_info(i, ue_grp_index).num_prb;
                            while(counter) {
                                while(((csirs_remap == (fh_context.csirs_remap(cell_index, pdschSym*num_dl_prb + rb_idx) & 0xFFF)) ||
                                (use_alt_prb && (csirs_remap == (fh_context.csirs_remap(cell_index, pdschSym*num_dl_prb + rb_idx + 1) & 0xFFF))))
                                && (counter > 0)){
                                    tempnPrb++; rb_idx++; counter--;
                                    continue;
                                }
                                // NVLOGC_FMT(TAG, "upate_fh_params_pdsch_with_csirs: cell_index={} numSym={} mask=0x{:x} csirs-symbolMap=0x{:x} start rb {} num rb {} re match",cell_index,numSym,mask,csirs_info->symbolMap,grp.startPrb,grp.nPrb );
                                check_prb_info_size(sym_prbs->prbs_size);
                                prbs[sym_prbs->prbs_size] = (use_alt_prb) ? prb_info_t(tempStartPrb, ((tempnPrb + 1) >> 1)): prb_info_t(tempStartPrb, tempnPrb);
                                sym_prbs->prbs_size++;
                                std::size_t index_pdsch_csirs{sym_prbs->prbs_size - 1};
                                prb_info_t& prb_info{prbs[sym_prbs->prbs_size - 1]};

                                prb_info.common.direction = fh_dir_t::FH_DIR_DL;
                                prb_info.common.numApIndices = 0;
                                prb_info.common.reMask = ~csirs_remap;
                                prb_info.common.useAltPrb = use_alt_prb;
                                if (likely(!pm_enabled)) {
                                    prb_info.common.portMask |= calculate_dmrs_port_mask(ue.dmrsPortBmsk, ue.scid, ue.nlAbove16);
                                } else {
                                    update_pm_weights_fh(ue, pm_map, pmi_bf_pdu, prb_info.common, cell_index, false);
                                }
                                if (bf_enabled) {
                                    update_beam_list(prb_info.beams_array, prb_info.beams_array_size, pmi_bf_pdu, false, prb_info, cell_index);
                                }

                                //Code to add CSI_RS antenna ports which are not covered by overlapping PDSCH
                                prb_index= 0;
                                while(csi_rs_prb_info_size > prb_index)
                                {
                                    if((csirs_remap & prbs[csi_rs_prb_info[prb_index]].common.reMask) == prbs[csi_rs_prb_info[prb_index]].common.reMask)
                                    {
                                        prb_info.common.portMask|= prbs[csi_rs_prb_info[prb_index]].common.portMask;

                                        break;
                                    }
                                    ++prb_index;
                                }
                                update_prb_sym_list(*sym_prbs, index_pdsch_csirs, pdschSym, 1, channel_type::PDSCH_CSIRS, ru);
                                NVLOGD_FMT(TAG, "PDSCH with CSI-RS new Grp : at symbol={} added index={} startPrb={} numPrb={} nSym=1",
                                        pdschSym, index_pdsch_csirs, tempStartPrb, tempnPrb);

                                if(use_alt_prb && tempnPrb > 2)
                                {
                                    prbs[sym_prbs->prbs_size] = prb_info_t(tempStartPrb + 1, (tempnPrb >> 1));
                                    std::size_t index_pdsch{sym_prbs->prbs_size};
                                    prb_info_t& prb_info{prbs[sym_prbs->prbs_size]};
                                    prb_info_t& prb_info_ref{prbs[sym_prbs->prbs_size - 1]};
                                    sym_prbs->prbs_size++;
                                    prb_info.common.direction = prb_info_ref.common.direction;
                                    prb_info.common.numApIndices = prb_info_ref.common.numApIndices;
                                    prb_info.common.useAltPrb = prb_info_ref.common.useAltPrb;
                                    prb_info.common.portMask = prb_info_ref.common.portMask;
                                    prb_info.beams_array = prb_info_ref.beams_array;
                                    prb_info.beams_array_size = prb_info_ref.beams_array_size;
                                    prb_info.common.reMask = 0xFFF;
                                    update_prb_sym_list(*sym_prbs, index_pdsch, pdschSym, 1, channel_type::PDSCH, ru);
                                    NVLOGD_FMT(TAG, "PDSCH with CSIRS new Grp but PDSCH only: at symbol={} added index={} startPrb={} numPrb={} nSym=1",
                                        pdschSym, index_pdsch, tempStartPrb+1, tempnPrb>>1);

                                }
                                tempStartPrb = rb_idx;
                                tempnPrb = 0;
                                csirs_remap = fh_context.csirs_remap(cell_index, pdschSym*num_dl_prb + rb_idx) & 0xFFF;
                                use_alt_prb = (fh_context.csirs_remap(cell_index, pdschSym*num_dl_prb + rb_idx) & (1 << 15)) ? true : false;
                #if 0
                                //DMRS
                                uint16_t bitmap = grp.dmrsSymLocBmsk >> grp.pdschStartSym;
                                uint8_t dmrs_pos = grp.pdschStartSym;
                                while(bitmap)
                                {
                                    if(1 & bitmap)
                                    {
                                        update_prb_sym_list(*sym_prbs, index, dmrs_pos, grp.pDmrsDynPrm->dmrsMaxLen, channel_type::PDSCH_DMRS);
                                        if(grp.pDmrsDynPrm->dmrsMaxLen > 1)
                                        {
                                            dmrs_pos += grp.pDmrsDynPrm->dmrsMaxLen;
                                            bitmap >>= grp.pDmrsDynPrm->dmrsMaxLen;
                                            continue;
                                        }
                                    }
                                    dmrs_pos++;
                                    bitmap >>= 1;
                                }
                #endif
                            } //while(counter)
                        }//for num_ra_type0_info
                    } //RA type 0
                    //TI_GENERIC_ADD("pdsch_csirs symbol end");
                } //pdsch+csirs
                else
                {
                    //if the highest byte is not set yet, set it to indicate PDSCH exists and record the start symbol for PDSCH
                    if(!(pdsch_only[index] & 0x01000000)) {
                        pdsch_only[index] = pdschSym << 16;
                        pdsch_only[index] |= 0x01000000;
                    }

                    pdsch_only[index]++;
                    NVLOGD_FMT(TAG, "pdsch_only[{}]:{}:{}", index,(pdsch_only[index]&0x00FF0000)>>16,pdsch_only[index]&0xFFFF);
                }
                pdschSym++;
            } // while(numSym)

            //TI_GENERIC_ADD("pdsch only symbol");
            //NVLOGD_FMT(TAG, "upate_fh_params_pdsch_with_csirs: cell_index={} numSym={} mask=0x{:x} csirs {} csirs-symbolMap=0x{:x} start rb {} num rb {} no match",cell_index,numSym,mask,static_cast<void *>(csirs_info),static_cast<uint16_t>(csirs_info->symbolMapArray[cell_index]),grp.startPrb,grp.nPrb );
            if(grp.resourceAlloc == 1) // RA type 1
            {
                check_prb_info_size(sym_prbs->prbs_size);
                prbs[sym_prbs->prbs_size] = prb_info_t(grp.startPrb, grp.nPrb); // TODO update for RAT0
                sym_prbs->prbs_size++;
                std::size_t index_pdsch_only{sym_prbs->prbs_size - 1};
                prb_info_t& prb_info{prbs[index_pdsch_only]};
                prb_info.common.ap_index = 0; //Init AP_INDEX as this is the first UE in the new UE-Group
                prb_info.common.direction = fh_dir_t::FH_DIR_DL;
                prb_info.common.numApIndices = 0;
                if (likely(!pm_enabled)) {
                    if(ue.rnti == UINT16_MAX)  // For 4T4R O-RUs, SI-RNTI date with a single layer transmission will be replicated for the second antenna
                    {
                        prb_info.common.portMask |= (1 << (2 * ue.nUeLayers)) - 1;
                    }
                    else
                    {
                        prb_info.common.portMask |= calculate_dmrs_port_mask(ue.dmrsPortBmsk, ue.scid, ue.nlAbove16);
                    }
                } else {
                    update_pm_weights_fh(ue, pm_map, pmi_bf_pdu, prb_info.common, cell_index, false);
                }
                if (bf_enabled) {
                    update_beam_list(prb_info.beams_array, prb_info.beams_array_size, pmi_bf_pdu, false, prb_info, cell_index);

                    if (bfwCoeff_mem_info != NULL && *bfwCoeff_mem_info->header == BFW_COFF_MEM_BUSY)
                    {
                        if ((is_latest_bfw_coff_avail(slot_3gpp.sfn_ , slot_3gpp.slot_,
                            bfwCoeff_mem_info->sfn, bfwCoeff_mem_info->slot)) && (pmi_bf_pdu.dig_bf_interfaces==0))
                            {
                                NVLOGD_FMT(TAG, "PDSCH with CSI-RS new Grp :: Header {} SFN {} SLOT {} BFW_SFN {} BFW_SLOT {}",
                                    *bfwCoeff_mem_info->header, slot_3gpp.sfn_, slot_3gpp.slot_,
                                    bfwCoeff_mem_info->sfn, bfwCoeff_mem_info->slot);
                                    //prb_info.common.numApIndices = pmi_bf_pdu.dig_bf_interfaces;
                                    //prb_info.common.portMask = (1 << pmi_bf_pdu.dig_bf_interfaces) -1;
                                    prb_info.common.extType = 11;
                                    prb_info.bfwCoeff_buf_info.num_prgs = pmi_bf_pdu.num_prgs;
                                    prb_info.bfwCoeff_buf_info.prg_size = pmi_bf_pdu.prg_size;
                                    prb_info.bfwCoeff_buf_info.dig_bf_interfaces = pmi_bf_pdu.dig_bf_interfaces;
                                    prb_info.bfwCoeff_buf_info.nGnbAnt = bfwCoeff_mem_info->nGnbAnt;
                                    prb_info.bfwCoeff_buf_info.header = bfwCoeff_mem_info->header;
                                    prb_info.bfwCoeff_buf_info.p_buf_bfwCoef_h = bfwCoeff_mem_info->buff_addr_chunk_h[ue_grp_bfw_index_per_cell];
                                    prb_info.bfwCoeff_buf_info.p_buf_bfwCoef_d = bfwCoeff_mem_info->buff_addr_chunk_d[ue_grp_bfw_index_per_cell];
                                    NVLOGD_FMT(TAG, "PDSCH new Grp dig_bf_interfaces={}, prb_info.common.portMask {}, buff_addr_chunk_h={}",
                                    pmi_bf_pdu.dig_bf_interfaces, static_cast<uint32_t>(prb_info.common.portMask), reinterpret_cast<void*>(bfwCoeff_mem_info->buff_addr_chunk_h[ue_grp_bfw_index_per_cell]));
                            }
                    }
                }
                uint32_t i = 0;
                while(i <= index) {
                    //for the case where CSI-RS is on the first and last symbol of PDSCH allocation, index gets incremented but the highest byte of pdsch_only
                    //does not get set. Hence check this condition to avoid adding a bogus entry in these special cases
                    if(pdsch_only[i] & 0x01000000)
                    {
                        update_prb_sym_list(*sym_prbs, index_pdsch_only, (pdsch_only[i]&0x00FF0000)>>16, (pdsch_only[i]&0xFFFF), channel_type::PDSCH, ru);

                        NVLOGD_FMT(TAG, "PDSCH new Grp : RA Type-1 : at symbol={} added index={} startPrb={} nPrb={} numSym={} prb_info={}",
                        (pdsch_only[i]&0x00FF0000)>>16, index_pdsch_only, grp.startPrb, grp.nPrb, (pdsch_only[i]&0xFFFF),static_cast<void *>(&prb_info.common));
                    }
                    i++;
                }
            }//RA type 1
            else
            { // RA type 0
                for(uint32_t i=0; i < fh_context.pdsch_num_ra_type0_info(ue_grp_index); i++)
                {
                    check_prb_info_size(sym_prbs->prbs_size);
                    prbs[sym_prbs->prbs_size] =
                    prb_info_t(fh_context.pdsch_ra_type0_info(i, ue_grp_index).start_prb + grp.startPrb, fh_context.pdsch_ra_type0_info(i, ue_grp_index).num_prb);
                    sym_prbs->prbs_size++;
                    std::size_t index_pdsch_only{sym_prbs->prbs_size - 1};
                    prb_info_t& prb_info{prbs[index_pdsch_only]};
                    prb_info.common.direction = fh_dir_t::FH_DIR_DL;
                    prb_info.common.numApIndices = 0;
                    if (likely(!pm_enabled)) {
                        if(ue.rnti == UINT16_MAX)  // For 4T4R O-RUs, SI-RNTI date with a single layer transmission will be replicated for the second antenna
                        {
                            prb_info.common.portMask |= (1 << (2 * ue.nUeLayers)) - 1;
                        }
                        else
                        {
                            prb_info.common.portMask |= calculate_dmrs_port_mask(ue.dmrsPortBmsk, ue.scid, ue.nlAbove16);
                        }
                    } else {
                        update_pm_weights_fh(ue, pm_map, pmi_bf_pdu, prb_info.common, cell_index, false);
                    }
                    if (bf_enabled) {
                        update_beam_list(prb_info.beams_array, prb_info.beams_array_size, pmi_bf_pdu, false, prb_info, cell_index);
                    }
                    uint32_t j = 0;
                    while(j <= index) {
                        //for the case where CSI-RS is on the first and last symbol of PDSCH allocation, index gets incremented but the highest byte of pdsch_only
                        //does not get set. Hence check this condition to avoid adding a bogus entry in these special cases
                        if(pdsch_only[j] & 0x01000000)
                        {
                            update_prb_sym_list(*sym_prbs, index_pdsch_only, (pdsch_only[j]&0x00FF0000)>>16, (pdsch_only[j]&0xFFFF), channel_type::PDSCH, ru);
                            NVLOGD_FMT(TAG, "PDSCH new Grp : RA Type-0 : at symbol={} added index={} startPrb={} nPrb={} numSym={} prb_info={}",
                            (pdsch_only[j]&0x00FF0000)>>16, index_pdsch_only, fh_context.pdsch_ra_type0_info(i, ue_grp_index).start_prb + grp.startPrb,
                            fh_context.pdsch_ra_type0_info(i, ue_grp_index).num_prb, (pdsch_only[j]&0xFFFF), static_cast<void *>(&prb_info.common));
                        }
                        j++;
                    }
                }//for num_ra_type0_info
            } // RA type 0
        //TI_GENERIC_ADD("pdsch only symbol end");
        } 
        else 
        {
            {
                if(grp.resourceAlloc == 1) //RA type 1
                {
                    auto& idxlist{sym_prbs->symbols[grp.pdschStartSym][channel_type::PDSCH]};

                    auto iter = std::find_if(idxlist.begin(), idxlist.end(),[&sym_prbs, &prbs, &grp](const auto& e){
                        bool retval = (e < sym_prbs->prbs_size);
                        if (retval) {
                            auto& prb{prbs[e]};
                            if(prb.common.useAltPrb)
                            {
                                retval = (prb.common.startPrbc == grp.startPrb && prb.common.numPrbc == ((grp.nPrb + 1) >> 2 ));
                            }
                            else
                            {
                                retval = (prb.common.startPrbc == grp.startPrb && prb.common.numPrbc == grp.nPrb);
                            }
                        }
                        return retval;
                    });
                    if (iter != idxlist.end()){
                        auto& prb{prbs[*iter]};
                            if (likely(!pm_enabled)) {
                            prb.common.portMask |= calculate_dmrs_port_mask(ue.dmrsPortBmsk, ue.scid, ue.nlAbove16);
                            //NVLOGD_FMT(TAG, "##### prb_info={}, portMask = {}", static_cast<void *>(&prb), static_cast<uint64_t>(prb.common.portMask));
                        } else {
                            update_pm_weights_fh(ue, pm_map, pmi_bf_pdu, prb.common, cell_index, false);
                        }
                    }
                }//RA type 1
                else
                {// RA type 0
                    auto& idxlist{sym_prbs->symbols[grp.pdschStartSym][channel_type::PDSCH]};
                    for(uint32_t i=0; i < fh_context.pdsch_num_ra_type0_info(ue_grp_index); i++)
                    {
                        uint32_t startPrb = fh_context.pdsch_ra_type0_info(i, ue_grp_index).start_prb + grp.startPrb;
                        uint32_t nPrb = fh_context.pdsch_ra_type0_info(i, ue_grp_index).num_prb;

                        auto iter = std::find_if(idxlist.begin(), idxlist.end(),[&sym_prbs, &prbs, startPrb, nPrb](const auto& e){
                            bool retval = (e < sym_prbs->prbs_size);
                            if (retval) {
                                auto& prb{prbs[e]};
                                retval = (prb.common.startPrbc == startPrb && prb.common.numPrbc == nPrb);
                            }
                            return retval;
                        });
                        if (iter != idxlist.end()){
                            auto& prb{prbs[*iter]};
                            if (likely(!pm_enabled)) {
                                prb.common.portMask |= calculate_dmrs_port_mask(ue.dmrsPortBmsk, ue.scid, ue.nlAbove16);
                            } else {
                                update_pm_weights_fh(ue, pm_map, pmi_bf_pdu, prb.common, cell_index, false);
                            }
                        }
                    }//for 0->num_ra_type0_info
                }// RA type 0
            }
        } // new grp = false
        //TI_GENERIC_ALL_NVLOGW(TAG);
    }

    template <bool mplane_configured_ru_type>
    void update_prc_fh_params_pdsch_with_csirs_mmimo(const pm_weight_map_t & pm_map, const IFhCallbackContext& fh_context, const PdschFhParamsView& pdsch_fh_param, nv::slot_detail_t* slot_detail, bool csirs_compact_mode) {
        bool mmimo_enabled = true; 
        int32_t cell_index = pdsch_fh_param.cell_index();

        uint8_t num_csirs_eaxcids = 0;
        aerial_fh::UserDataCompressionMethod dl_comp_method = aerial_fh::UserDataCompressionMethod::BLOCK_FLOATING_POINT;
        ru_type ru;
        uint8_t num_eaxcids = 0;
        if (mplane_configured_ru_type)
        {
            nv::PHYDriverProxy& phyDriver = nv::PHYDriverProxy::getInstance();
            const auto & mplane_info = phyDriver.getMPlaneConfig(cell_index);
            dl_comp_method = mplane_info.dl_comp_meth;
            ru = mplane_info.ru;
            num_eaxcids = mplane_info.eAxC_ids[slot_command_api::channel_type::CSI_RS].size();
        }
        else
        {
            ru = OTHER_MODE;
        }
        if(csirs_compact_mode) {
            dl_comp_method = comp_method::MODULATION_COMPRESSION;
        }
        switch (dl_comp_method) {
            case comp_method::MODULATION_COMPRESSION:
                    update_prc_fh_params_pdsch_with_csirs_mod_comp(pm_map, fh_context, pdsch_fh_param, slot_detail, ru_type::OTHER_MODE, comp_method::MODULATION_COMPRESSION, num_eaxcids, csirs_compact_mode);
                break;
            case comp_method::NO_COMPRESSION:
            case comp_method::BLOCK_FLOATING_POINT: {
                const cuphyPdschCellGrpDynPrm_t& cell_grp_info = fh_context.pdsch_cell_grp_info();
                cuphyPdschUeGrpPrm_t& grp = pdsch_fh_param.grp();
                bool is_new_grp = pdsch_fh_param.is_new_grp();
                cuphyPdschUePrm_t& ue = pdsch_fh_param.ue();
                auto& pmi_bf_pdu = pdsch_fh_param.pc_bf();
                // pm_weight_map_t & pm_map = pm_map
                auto& slot_3gpp = pdsch_fh_param.slot_3gpp();
                bool bf_enabled = pdsch_fh_param.bf_enabled();
                bool pm_enabled = pdsch_fh_param.pm_enabled();
                uint32_t ue_grp_index = pdsch_fh_param.ue_grp_index();
                uint32_t ue_grp_bfw_index_per_cell = pdsch_fh_param.ue_grp_bfw_index_per_cell();
                bfw_coeff_mem_info_t* bfwCoeff_mem_info = pdsch_fh_param.bfw_coeff_mem_info();
                uint16_t num_dl_prb = pdsch_fh_param.num_dl_prb();

                // NVLOGC_FMT(TAG, "pdsch_params* pdsch_info {}, cuphyPdschCellGrpDynPrm_t& cell_grp_info {},"
                // "cuphyPdschUeGrpPrm_t& grp {}, bool is_new_grp {}, cuphyPdschUePrm_t& ue {},"
                // "scf_fapi_tx_precoding_beamforming_t& pmi_bf_pdu {}, pm_weight_map_t & pm_map {},"
                // "cell_sub_command& cell_cmd {}, bool bf_enabled {}, bool pm_enabled {}, int32_t cell_index {}, csirs_params* csirs_info {}, uint16_t num_dl_prb {}\n",
                // pdsch_info, &cell_grp_info,
                // &grp, (is_new_grp)?1:0, &ue,
                // &pmi_bf_pdu, &pm_map,
                // &cell_cmd, bf_enabled?1:0, pm_enabled?1:0, cell_index, csirs_info, num_dl_prb);
                auto sym_prbs{pdsch_fh_param.sym_prb_info()};
                auto& prbs{sym_prbs->prbs};
                //slot_command_api::prb_info_common_t * csi_rs_prb_common = nullptr;
                int  prb_index= 0;
                uint64_t csirs_portMask = 0;
                const uint16_t csirs_symbol_map = fh_context.csirs_symbol_map(cell_index);

                if (is_new_grp) {
                    uint32_t numSym = grp.nPdschSym;
                    uint32_t pdschSym = grp.pdschStartSym;
                    uint32_t tempPdschSym = 0;
                    uint32_t tempNumSym = 0;
                    std::size_t index_pdsch_only{sym_prbs->prbs_size - 1};
                    //NVLOGD_FMT(TAG, "{}:{} is_new_grp={} sym_prbs->prbs_size={} index_pdsch_only={}", 
                    //    __FILE__, __LINE__, is_new_grp, sym_prbs->prbs_size, index_pdsch_only);
                    while(numSym--) {
                        uint32_t mask = 1 << pdschSym;
                        //NVLOGD_FMT(TAG, "upate_fh_params_pdsch_with_csirs: cell_index={} numSym={} mask=0x{:x} csirs-symbolMap=0x{:x} start rb {} num rb {}",cell_index,numSym,mask,static_cast<uint16_t>(csirs_info->symbolMapArray[cell_index]),grp.startPrb,grp.nPrb );
                        if ((mask & csirs_symbol_map) == 0U)
                        {
                            //NVLOGD_FMT(TAG, "upate_fh_params_pdsch_with_csirs: cell_index={} numSym={} mask=0x{:x} csirs {} csirs-symbolMap=0x{:x} start rb {} num rb {} no match",cell_index,numSym,mask,static_cast<void *>(csirs_info),static_cast<uint16_t>(csirs_info->symbolMapArray[cell_index]),grp.startPrb,grp.nPrb );
                            if(grp.resourceAlloc == 1) // RA type 1
                            {
                                check_prb_info_size(sym_prbs->prbs_size);
                                prbs[sym_prbs->prbs_size] = prb_info_t(grp.startPrb, grp.nPrb); // TODO update for RAT0
                                sym_prbs->prbs_size++;
                                std::size_t index_pdsch_only{sym_prbs->prbs_size - 1};
                                prb_info_t& prb_info{prbs[index_pdsch_only]};
                                prb_info.common.ap_index = 0; //Init AP_INDEX as this is the first UE in the new UE-Group
                                prb_info.common.direction = fh_dir_t::FH_DIR_DL;
                                prb_info.common.numApIndices = 0;
                                if (likely(!pm_enabled)) {
                                    if(ue.rnti == UINT16_MAX && !pdsch_fh_param.mmimo_enabled())  // For 4T4R O-RUs, SI-RNTI date with a single layer transmission will be replicated for the second antenna
                                    {
                                        prb_info.common.portMask |= (1 << (2 * ue.nUeLayers)) - 1;
                                    }
                                    else
                                    {
                                        prb_info.common.portMask |= calculate_dmrs_port_mask(ue.dmrsPortBmsk, ue.scid, ue.nlAbove16);
                                        //NVLOGD_FMT(TAG, "{}:{} is_new_grp={} ue.rnti {} ue.dmrsPortBmsk {} ue.scid {}",
                                        //    __FILE__, __LINE__, is_new_grp, ue.rnti, ue.dmrsPortBmsk, ue.scid);
                                        track_eaxcids_fh(ue, prb_info.common);
                                    }
                                } else {
                                    update_pm_weights_fh(ue, pm_map, pmi_bf_pdu, prb_info.common, cell_index, mmimo_enabled);
                                }
                                if (bf_enabled) {
                                    update_beam_list(prb_info.beams_array, prb_info.beams_array_size, pmi_bf_pdu, mmimo_enabled, prb_info, cell_index);

                                    if (bfwCoeff_mem_info != NULL && *bfwCoeff_mem_info->header == BFW_COFF_MEM_BUSY)
                                    {
                                        if ((is_latest_bfw_coff_avail(slot_3gpp.sfn_ , slot_3gpp.slot_,
                                                bfwCoeff_mem_info->sfn, bfwCoeff_mem_info->slot)) && (pmi_bf_pdu.dig_bf_interfaces==0))
                                        {
                                            NVLOGD_FMT(TAG, "PDSCH with CSI-RS new Grp :: Header {} SFN {} SLOT {} BFW_SFN {} BFW_SLOT {}",
                                                        *bfwCoeff_mem_info->header, slot_3gpp.sfn_, slot_3gpp.slot_,
                                                        bfwCoeff_mem_info->sfn, bfwCoeff_mem_info->slot);
                                            //prb_info.common.numApIndices = pmi_bf_pdu.dig_bf_interfaces;
                                            //prb_info.common.portMask = (1 << pmi_bf_pdu.dig_bf_interfaces) -1;
                                            prb_info.common.extType = 11;
                                            prb_info.bfwCoeff_buf_info.num_prgs = pmi_bf_pdu.num_prgs;
                                            prb_info.bfwCoeff_buf_info.prg_size = pmi_bf_pdu.prg_size;
                                            prb_info.bfwCoeff_buf_info.dig_bf_interfaces = pmi_bf_pdu.dig_bf_interfaces;
                                            prb_info.bfwCoeff_buf_info.nGnbAnt = bfwCoeff_mem_info->nGnbAnt;
                                            prb_info.bfwCoeff_buf_info.header = bfwCoeff_mem_info->header;
                                            prb_info.bfwCoeff_buf_info.p_buf_bfwCoef_h = bfwCoeff_mem_info->buff_addr_chunk_h[ue_grp_bfw_index_per_cell];
                                            prb_info.bfwCoeff_buf_info.p_buf_bfwCoef_d = bfwCoeff_mem_info->buff_addr_chunk_d[ue_grp_bfw_index_per_cell];
                                            NVLOGD_FMT(TAG, "PDSCH new Grp dig_bf_interfaces={}, prb_info.common.portMask {}, buff_addr_chunk_h={}",
                                                            pmi_bf_pdu.dig_bf_interfaces, static_cast<uint32_t>(prb_info.common.portMask), reinterpret_cast<void*>(bfwCoeff_mem_info->buff_addr_chunk_h[ue_grp_bfw_index_per_cell]));
                                        }
                                    }
                                }
                                if (mmimo_enabled)
                                {
                                    prb_info.common.pdschPortMask = prb_info.common.portMask;
                                    tempPdschSym = pdschSym;
                                    // Incrementing it by 1 because the outer while loop decrements it. We need to adjust the symbols to combine them for muMIMO ORU.
                                    numSym++;
                                    while ((mask & csirs_symbol_map) == 0U)
                                    {
                                        tempNumSym++;
                                        tempPdschSym++;
                                        if (tempPdschSym == (grp.pdschStartSym + grp.nPdschSym))
                                        {
                                            numSym=0;
                                            break;
                                        }
                                        mask = 1 << tempPdschSym;
                                        // Decrementing the symbols which are combined together.
                                        numSym--;
                                        NVLOGD_FMT(TAG, "mask=0x{:x}, tempNumSym={}, tempPdschSym={}, numSym={}", mask,  tempNumSym, tempPdschSym, numSym);
                                    }

                                    prb_info.common.numSymbols = tempNumSym;

                                    update_prb_sym_list(*sym_prbs, index_pdsch_only, pdschSym, tempNumSym, channel_type::PDSCH, ru);

                                    NVLOGD_FMT(TAG, "PDSCH new Grp : RA Type-1 : at symbol={} added index={} startPrb={} nPrb={} tempNumSym={} prb_info={} portMask={} numSym={}",
                                            pdschSym, index_pdsch_only, grp.startPrb, grp.nPrb, tempNumSym, static_cast<void *>(&prb_info.common),static_cast<uint32_t>(prb_info.common.portMask), numSym);

                                    pdschSym = tempPdschSym - 1;
                                    tempNumSym = 0;
                                }
                                
                            }//RA type 1
                            else
                            { // RA type 0
                                for(uint32_t i=0; i < fh_context.pdsch_num_ra_type0_info(ue_grp_index); i++)
                                {
                                    check_prb_info_size(sym_prbs->prbs_size);
                                    prbs[sym_prbs->prbs_size] =
                                        prb_info_t(fh_context.pdsch_ra_type0_info(i, ue_grp_index).start_prb + grp.startPrb, fh_context.pdsch_ra_type0_info(i, ue_grp_index).num_prb);
                                    sym_prbs->prbs_size++;
                                    std::size_t index_pdsch_only{sym_prbs->prbs_size - 1};
                                    prb_info_t& prb_info{prbs[index_pdsch_only]};
                                    prb_info.common.direction = fh_dir_t::FH_DIR_DL;
                                    prb_info.common.numApIndices = 0;
                                    prb_info.common.ap_index = 0; //Init AP_INDEX as this is the first UE in the new UE-Group
                                    if (likely(!pm_enabled)) {
                                        if(ue.rnti == UINT16_MAX && !pdsch_fh_param.mmimo_enabled())  // For 4T4R O-RUs, SI-RNTI date with a single layer transmission will be replicated for the second antenna
                                        {
                                            prb_info.common.portMask |= (1 << (2 * ue.nUeLayers)) - 1;
                                        }
                                        else
                                        {
                                            prb_info.common.portMask |= calculate_dmrs_port_mask(ue.dmrsPortBmsk, ue.scid, ue.nlAbove16);
                                            if(mmimo_enabled)
                                            {
                                                //NVLOGC_FMT(TAG, "{}:{} RA Type-0: ue.rnti {} ue.dmrsPortBmsk {} ue.scid {}", __FILE__, __LINE__, ue.rnti, ue.dmrsPortBmsk, ue.scid);
                                                track_eaxcids_fh(ue, prb_info.common);
                                            }
                                        }
                                    } else {
                                        update_pm_weights_fh(ue, pm_map, pmi_bf_pdu, prb_info.common, cell_index, mmimo_enabled);
                                    }
                                    if (bf_enabled) {
                                        update_beam_list(prb_info.beams_array, prb_info.beams_array_size, pmi_bf_pdu, mmimo_enabled, prb_info, cell_index);
                                    }
                                    update_prb_sym_list(*sym_prbs, index_pdsch_only, pdschSym, 1, channel_type::PDSCH, ru);
                                    NVLOGD_FMT(TAG, "PDSCH new Grp : RA Type-0 : at symbol={} added index={} startPrb={} nPrb={} numSym=1 prb_info={}",
                                        pdschSym, index_pdsch_only,fh_context.pdsch_ra_type0_info(i, ue_grp_index).start_prb + grp.startPrb,
                                        fh_context.pdsch_ra_type0_info(i, ue_grp_index).num_prb, static_cast<void *>(&prb_info.common));
                                }//for num_ra_type0_info
                            } // RA type 0
                        } //PDSCH only
                        else
                        {
                            //NVLOGD_FMT(TAG, "upate_fh_params_pdsch_with_csirs: cell_index={} numSym={} mask=0x{:x} csirs-symbolMap=0x{:x} start rb {} num rb {} sym match",cell_index,numSym,mask,static_cast<uint16_t>(csirs_info->symbolMapArray[cell_index]),grp.startPrb,grp.nPrb );
                            prb_info_idx_list_t & csi_rs_prb_info = sym_prbs->symbols[pdschSym][channel_type::CSI_RS];
                            auto csi_rs_prb_info_size = csi_rs_prb_info.size();
                            if(grp.resourceAlloc == 1) //RA type 1
                            {
                                uint16_t rb_idx = grp.startPrb;
                                uint16_t csirs_remap = fh_context.csirs_remap(cell_index, pdschSym*num_dl_prb + rb_idx) & 0xFFF;
                                bool use_alt_prb = (fh_context.csirs_remap(cell_index, pdschSym*num_dl_prb + rb_idx) & (1 << 15)) ? true : false;
                                uint32_t tempnPrb = 0;
                                uint32_t tempStartPrb = rb_idx;
                                uint32_t counter = grp.nPrb;
                                bool isPdschSplitAcrossPrbInfo = false;
                                while(counter) {
                                while(((csirs_remap == (fh_context.csirs_remap(cell_index, pdschSym * num_dl_prb + rb_idx) & 0xFFF)) ||
                                (use_alt_prb && (csirs_remap == (fh_context.csirs_remap(cell_index, pdschSym * num_dl_prb + rb_idx + 1) & 0xFFF))))
                                        && (counter > 0)) {
                                            //NVLOGD_FMT(TAG, "csirs_remap={} csirs_info->reMap={}",csirs_remap, (csirs_info->reMap[cell_index][pdschSym*num_dl_prb + rb_idx] & 0xFFF));
                                        tempnPrb++; rb_idx++; counter--;
                                        continue;
                                    }

                                    if(counter)
                                    {
                                        isPdschSplitAcrossPrbInfo = true;
                                    }

                                    //NVLOGC_FMT(TAG, "upate_fh_params_pdsch_with_csirs: cell_index={} counter={} isPdschSplitAcrossPrbInfo={} tempnPrb={} numSym={} mask=0x{:x} csirs-symbolMap=0x{:x} start rb {} num rb {} re match",
                                                    //cell_index,counter,isPdschSplitAcrossPrbInfo,tempnPrb,numSym,mask,static_cast<uint16_t>(csirs_info->symbolMapArray[cell_index]),grp.startPrb,grp.nPrb );
                                    check_prb_info_size(sym_prbs->prbs_size);
                                    prbs[sym_prbs->prbs_size] = (use_alt_prb) ? prb_info_t(tempStartPrb, ((tempnPrb + 1) >> 1)): prb_info_t(tempStartPrb, tempnPrb);
                                    sym_prbs->prbs_size++;
                                    std::size_t index_pdsch_csirs{sym_prbs->prbs_size - 1};
                                    prb_info_t& prb_info{prbs[sym_prbs->prbs_size - 1]};
                                    prb_info.common.direction = fh_dir_t::FH_DIR_DL;
                                    prb_info.common.numApIndices = 0;
                                    prb_info.common.reMask = ~csirs_remap;
                                    prb_info.common.useAltPrb = use_alt_prb;
                                    prb_info.common.isPdschSplitAcrossPrbInfo = isPdschSplitAcrossPrbInfo;
                                    prb_info.common.ap_index = 0; //Init AP_INDEX as this is the first UE in the new UE-Group
                                    if (likely(!pm_enabled)) {
                                        prb_info.common.portMask |= calculate_dmrs_port_mask(ue.dmrsPortBmsk, ue.scid, ue.nlAbove16);
                                        if(mmimo_enabled)
                                        {
                                            //NVLOGD_FMT(TAG, "{}:{} is_new_grp={} ue.rnti {} ue.dmrsPortBmsk {} ue.scid {}", __FILE__, __LINE__, is_new_grp, ue.rnti, ue.dmrsPortBmsk, ue.scid);
                                            track_eaxcids_fh(ue, prb_info.common);
                                        }
                                    } else {
                                        update_pm_weights_fh(ue, pm_map, pmi_bf_pdu, prb_info.common, cell_index, mmimo_enabled);
                                    }
                                    if (bf_enabled) {
                                        update_beam_list(prb_info.beams_array, prb_info.beams_array_size, pmi_bf_pdu, mmimo_enabled, prb_info, cell_index);

                                        if (bfwCoeff_mem_info != NULL && *bfwCoeff_mem_info->header == BFW_COFF_MEM_BUSY)
                                        {
                                            NVLOGD_FMT(TAG, "PDSCH with CSI-RS new Grp :: Header {} SFN {} SLOT {} BFW_SFN {} BFW_SLOT {}",
                                                        *bfwCoeff_mem_info->header, slot_3gpp.sfn_, slot_3gpp.slot_,
                                                        bfwCoeff_mem_info->sfn, bfwCoeff_mem_info->slot);
                                            if ((is_latest_bfw_coff_avail(slot_3gpp.sfn_ , slot_3gpp.slot_,
                                                    bfwCoeff_mem_info->sfn, bfwCoeff_mem_info->slot) && (pmi_bf_pdu.dig_bf_interfaces==0)))
                                            {
                                                prb_info.common.extType = 11;
                                                prb_info.bfwCoeff_buf_info.num_prgs = pmi_bf_pdu.num_prgs;
                                                prb_info.bfwCoeff_buf_info.prg_size = pmi_bf_pdu.prg_size;
                                                prb_info.bfwCoeff_buf_info.dig_bf_interfaces = pmi_bf_pdu.dig_bf_interfaces;
                                                prb_info.bfwCoeff_buf_info.nGnbAnt = bfwCoeff_mem_info->nGnbAnt;
                                                prb_info.bfwCoeff_buf_info.header = bfwCoeff_mem_info->header;
                                                prb_info.bfwCoeff_buf_info.p_buf_bfwCoef_h = bfwCoeff_mem_info->buff_addr_chunk_h[ue_grp_bfw_index_per_cell];
                                                prb_info.bfwCoeff_buf_info.p_buf_bfwCoef_d = bfwCoeff_mem_info->buff_addr_chunk_d[ue_grp_bfw_index_per_cell];
                                                NVLOGD_FMT(TAG, "PDSCH with CSI-RS new Grp dig_bf_interfaces={}, prb_info.common.portMask {}",
                                                                pmi_bf_pdu.dig_bf_interfaces, static_cast<uint32_t>(prb_info.common.portMask));
                                            }
                                        }
                                    }

                                    //Code to add CSI_RS antenna ports which are not covered by overlapping PDSCH
                                    prb_index= 0;
                                    if(mmimo_enabled)
                                    {
                                        prb_info.common.pdschPortMask = prb_info.common.portMask;
                                    }
                                    /// Not required for ModComp
                                    while(csi_rs_prb_info_size > prb_index)
                                    {
                                        if((csirs_remap & prbs[csi_rs_prb_info[prb_index]].common.reMask) == prbs[csi_rs_prb_info[prb_index]].common.reMask)
                                        {
                                            prb_info.common.portMask |= prbs[csi_rs_prb_info[prb_index]].common.portMask;
                                            if (mmimo_enabled)
                                            {
                                                NVLOGD_FMT(TAG, "prb_info={}, reMask=0x{:x}, csi_rs_portMask=0x{:x}, portMask=0x{:x} pdschPortMask=0x{:x}", static_cast<void *>(&prb_info),
                                                                    static_cast<uint16_t>(prbs[csi_rs_prb_info[prb_index]].common.reMask),
                                                                    static_cast<uint16_t>(prbs[csi_rs_prb_info[prb_index]].common.portMask),
                                                                    static_cast<uint64_t>(prb_info.common.portMask), static_cast<uint64_t>(prb_info.common.pdschPortMask));
                                            }
                                            break;
                                        }
                                        ++prb_index;
                                    }
                                    update_prb_sym_list(*sym_prbs, index_pdsch_csirs, pdschSym, 1, channel_type::PDSCH_CSIRS, ru);
                                    NVLOGD_FMT(TAG, "PDSCH with CSI-RS new Grp : at symbol={} added index={} startPrb={} numPrb={} nSym=1 prb_info={} numSym={}",
                                            pdschSym, index_pdsch_csirs, tempStartPrb, tempnPrb, static_cast<void *>(&prb_info.common), numSym);

                                    if(use_alt_prb && tempnPrb > 2)
                                    {
                                        prbs[sym_prbs->prbs_size] = prb_info_t(tempStartPrb + 1, (tempnPrb >> 1));
                                        std::size_t index_pdsch{sym_prbs->prbs_size};
                                        prb_info_t& prb_info{prbs[sym_prbs->prbs_size]};
                                        prb_info_t& prb_info_ref{prbs[sym_prbs->prbs_size - 1]};
                                        ++(sym_prbs->prbs_size);
                                        prb_info.common.direction = prb_info_ref.common.direction;
                                        prb_info.common.numApIndices = prb_info_ref.common.numApIndices;
                                        prb_info.common.useAltPrb = prb_info_ref.common.useAltPrb;
                                        prb_info.common.portMask = prb_info_ref.common.portMask;
                                        prb_info.beams_array = prb_info_ref.beams_array;
                                        prb_info.beams_array_size = prb_info_ref.beams_array_size;
                                        prb_info.common.reMask = 0xFFF;
                                        prb_info.common.ap_index = prb_info_ref.common.ap_index;
                                        prb_info.common.pdschPortMask = prb_info.common.portMask;
                                        //NVLOGD_FMT(TAG, "{}:{} start ap_index={} is_new_grp={} ", 
                                        //    __FILE__, __LINE__, static_cast<uint32_t>(prb_info.common.ap_index), is_new_grp);
                                        track_eaxcids_fh(ue, prb_info.common);  // Track eAxC for PDSCH
                                        update_prb_sym_list(*sym_prbs, index_pdsch, pdschSym, 1, channel_type::PDSCH, ru);
                                    }

                                    tempStartPrb = rb_idx;
                                    tempnPrb = 0;
                                    csirs_remap = fh_context.csirs_remap(cell_index, pdschSym*num_dl_prb + rb_idx) & 0xFFF;
                                    use_alt_prb = (fh_context.csirs_remap(cell_index, pdschSym*num_dl_prb + rb_idx) & (1 << 15)) ? true : false;
                    #if 0
                                    //DMRS
                                    uint16_t bitmap = grp.dmrsSymLocBmsk >> grp.pdschStartSym;
                                    uint8_t dmrs_pos = grp.pdschStartSym;
                                    while(bitmap)
                                    {
                                        if(1 & bitmap)
                                        {
                                            update_prb_sym_list(*sym_prbs, index, dmrs_pos, grp.pDmrsDynPrm->dmrsMaxLen, channel_type::PDSCH_DMRS);
                                            if(grp.pDmrsDynPrm->dmrsMaxLen > 1)
                                            {
                                                dmrs_pos += grp.pDmrsDynPrm->dmrsMaxLen;
                                                bitmap >>= grp.pDmrsDynPrm->dmrsMaxLen;
                                                continue;
                                            }
                                        }
                                        dmrs_pos++;
                                        bitmap >>= 1;
                                    }
                    #endif
                                } //while(counter)
                            } // RA type 1
                            else
                            { // RA type 0
                                for(uint32_t i=0; i < fh_context.pdsch_num_ra_type0_info(ue_grp_index); i++)
                                {
                                    uint16_t rb_idx = fh_context.pdsch_ra_type0_info(i, ue_grp_index).start_prb + grp.startPrb;
                                    uint16_t csirs_remap = fh_context.csirs_remap(cell_index, pdschSym*num_dl_prb + rb_idx) & 0xFFF;
                                    bool use_alt_prb = (fh_context.csirs_remap(cell_index, pdschSym*num_dl_prb + rb_idx) & (1 << 15)) ? true : false;
                                    uint32_t tempnPrb = 0;
                                    uint32_t tempStartPrb = rb_idx;
                                    uint32_t counter = fh_context.pdsch_ra_type0_info(i, ue_grp_index).num_prb;
                                    while(counter) {
                                        while(((csirs_remap == (fh_context.csirs_remap(cell_index, pdschSym*num_dl_prb + rb_idx) & 0xFFF)) ||
                                        (use_alt_prb && (csirs_remap == (fh_context.csirs_remap(cell_index, pdschSym*num_dl_prb + rb_idx + 1) & 0xFFF))))
                                        && (counter > 0)){
                                            tempnPrb++; rb_idx++; counter--;
                                            continue;
                                        }
                                        // NVLOGC_FMT(TAG, "upate_fh_params_pdsch_with_csirs: cell_index={} numSym={} mask=0x{:x} csirs-symbolMap=0x{:x} start rb {} num rb {} re match",cell_index,numSym,mask,csirs_info->symbolMap,grp.startPrb,grp.nPrb );
                                        check_prb_info_size(sym_prbs->prbs_size);
                                        prbs[sym_prbs->prbs_size] = (use_alt_prb) ? prb_info_t(tempStartPrb, ((tempnPrb + 1) >> 1)): prb_info_t(tempStartPrb, tempnPrb);
                                        sym_prbs->prbs_size++;
                                        std::size_t index_pdsch_csirs{sym_prbs->prbs_size - 1};
                                        prb_info_t& prb_info{prbs[sym_prbs->prbs_size - 1]};

                                        prb_info.common.direction = fh_dir_t::FH_DIR_DL;
                                        prb_info.common.numApIndices = 0;
                                        prb_info.common.reMask = ~csirs_remap;
                                        prb_info.common.useAltPrb = use_alt_prb;
                                        if (likely(!pm_enabled)) {
                                            prb_info.common.portMask |= calculate_dmrs_port_mask(ue.dmrsPortBmsk, ue.scid, ue.nlAbove16);
                                            //NVLOGD_FMT(TAG, "line {} ue.rnti {} ue.dmrsPortBmsk {} ue.scid {}", __LINE__, ue.rnti, ue.dmrsPortBmsk, ue.scid);
                                            track_eaxcids_fh(ue, prb_info.common);
                                        } else {
                                            update_pm_weights_fh(ue, pm_map, pmi_bf_pdu, prb_info.common, cell_index, mmimo_enabled);
                                        }
                                        if (bf_enabled) {
                                            update_beam_list(prb_info.beams_array, prb_info.beams_array_size, pmi_bf_pdu, mmimo_enabled, prb_info, cell_index);
                                        }

                                        //Code to add CSI_RS antenna ports which are not covered by overlapping PDSCH
                                        prb_index= 0;
                                        while(csi_rs_prb_info_size > prb_index)
                                        {
                                            if((csirs_remap & prbs[csi_rs_prb_info[prb_index]].common.reMask) == prbs[csi_rs_prb_info[prb_index]].common.reMask)
                                            {
                                                prb_info.common.portMask|= prbs[csi_rs_prb_info[prb_index]].common.portMask;

                                                break;
                                            }
                                            ++prb_index;
                                        }
                                        update_prb_sym_list(*sym_prbs, index_pdsch_csirs, pdschSym, 1, channel_type::PDSCH_CSIRS, ru);
                                        NVLOGD_FMT(TAG, "PDSCH with CSI-RS new Grp : at symbol={} added index={} startPrb={} numPrb={} nSym=1",
                                                pdschSym, index_pdsch_csirs, tempStartPrb, tempnPrb);

                                        if(use_alt_prb && tempnPrb > 2)
                                        {
                                            prbs[sym_prbs->prbs_size] = prb_info_t(tempStartPrb + 1, (tempnPrb >> 1));
                                            std::size_t index_pdsch{sym_prbs->prbs_size};
                                            prb_info_t& prb_info{prbs[sym_prbs->prbs_size]};
                                            prb_info_t& prb_info_ref{prbs[sym_prbs->prbs_size - 1]};
                                            sym_prbs->prbs_size++;
                                            prb_info.common.direction = prb_info_ref.common.direction;
                                            prb_info.common.numApIndices = prb_info_ref.common.numApIndices;
                                            prb_info.common.useAltPrb = prb_info_ref.common.useAltPrb;
                                            prb_info.common.portMask = prb_info_ref.common.portMask;
                                            prb_info.beams_array = prb_info_ref.beams_array;
                                            prb_info.beams_array_size = prb_info_ref.beams_array_size;
                                            prb_info.common.reMask = 0xFFF;
                                            prb_info.common.ap_index = prb_info_ref.common.ap_index;
                                            prb_info.common.pdschPortMask = prb_info.common.portMask;
                                            track_eaxcids_fh(ue, prb_info.common);  // Track eAxC for PDSCH
                                            update_prb_sym_list(*sym_prbs, index_pdsch, pdschSym, 1, channel_type::PDSCH, ru);
                                        }
                                        tempStartPrb = rb_idx;
                                        tempnPrb = 0;
                                csirs_remap = fh_context.csirs_remap(cell_index, pdschSym * num_dl_prb + rb_idx) & 0xFFF;
                                use_alt_prb = (fh_context.csirs_remap(cell_index, pdschSym * num_dl_prb + rb_idx) & (1 << 15)) ? true : false;
                        #if 0
                                        //DMRS
                                        uint16_t bitmap = grp.dmrsSymLocBmsk >> grp.pdschStartSym;
                                        uint8_t dmrs_pos = grp.pdschStartSym;
                                        while(bitmap)
                                        {
                                            if(1 & bitmap)
                                            {
                                                update_prb_sym_list(*sym_prbs, index, dmrs_pos, grp.pDmrsDynPrm->dmrsMaxLen, channel_type::PDSCH_DMRS);
                                                if(grp.pDmrsDynPrm->dmrsMaxLen > 1)
                                                {
                                                    dmrs_pos += grp.pDmrsDynPrm->dmrsMaxLen;
                                                    bitmap >>= grp.pDmrsDynPrm->dmrsMaxLen;
                                                    continue;
                                                }
                                            }
                                            dmrs_pos++;
                                            bitmap >>= 1;
                                        }
                        #endif
                                    } //while(counter)
                                }//for num_ra_type0_info
                            } //RA type 0
                        } //pdsch+csirs
                        pdschSym++;
                    } // while(numSym)
                } 
                else 
                {
                    if(mmimo_enabled)
                    {
                        uint32_t numSym = grp.nPdschSym;
                        uint32_t pdschSym = grp.pdschStartSym;
                        //NVLOGD_FMT(TAG, "{}:{} numSym={} pdschSym={}", __FILE__, __LINE__, numSym, pdschSym);
                        slot_command_api::prb_info_t* p_prb_info_ref = nullptr;
                        while(numSym--)
                        {
                            if(grp.resourceAlloc == 1) //RA type 1
                            {
                                auto& idxlist{sym_prbs->symbols[pdschSym][channel_type::PDSCH]};

                                auto iter = std::find_if(idxlist.begin(), idxlist.end(),[&sym_prbs, &prbs, &grp](const auto& e){
                                    bool retval = (e < sym_prbs->prbs_size);
                                    if (retval) {
                                        auto& prb{prbs[e]};
                                        if(prb.common.useAltPrb)
                                        {
                                            retval = (prb.common.startPrbc == grp.startPrb && prb.common.numPrbc == ((grp.nPrb + 1) >> 2 ));
                                        }
                                        else
                                        {
                                            retval = (prb.common.startPrbc == grp.startPrb && prb.common.numPrbc == grp.nPrb);
                                        }
                                    }
                                    return retval;
                                });
                                if (iter != idxlist.end()){
                                        auto& prb{prbs[*iter]};
                                        //NVLOGD_FMT(TAG, "{}:{} ap_index={}", __FILE__, __LINE__, static_cast<uint32_t>(prb.common.ap_index));
                                        if (likely(!pm_enabled)) {
                                        prb.common.portMask |= calculate_dmrs_port_mask(ue.dmrsPortBmsk, ue.scid, ue.nlAbove16);
                                        prb.common.pdschPortMask |= calculate_dmrs_port_mask(ue.dmrsPortBmsk, ue.scid, ue.nlAbove16);
                                        //Avoid updating the same PRB info multiple times
                                        if((p_prb_info_ref == nullptr) || (&prb != p_prb_info_ref))
                                        {
                                            //NVLOGD_FMT(TAG, "{}:{} is_new_grp={} ue.rnti {} ue.dmrsPortBmsk {} ue.scid {}", __FILE__, __LINE__, is_new_grp, ue.rnti, ue.dmrsPortBmsk, ue.scid);
                                            track_eaxcids_fh(ue, prb.common);
                                            p_prb_info_ref = &prb;
                                        }
                                        //NVLOGD_FMT(TAG, "##### channel_type=PDSCH prb_info={}, portMask = {}", static_cast<void *>(&prb), static_cast<uint64_t>(prb.common.portMask));
                                    } else {
                                        update_pm_weights_fh(ue, pm_map, pmi_bf_pdu, prb.common, cell_index, mmimo_enabled);
                                    }
                                }
                            }//RA type 1
                            else
                            {// RA type 0
                                auto& idxlist{sym_prbs->symbols[grp.pdschStartSym][channel_type::PDSCH]};
                                for(uint32_t i=0; i < fh_context.pdsch_num_ra_type0_info(ue_grp_index); i++)
                                {
                                    uint32_t startPrb = fh_context.pdsch_ra_type0_info(i, ue_grp_index).start_prb + grp.startPrb;
                                    uint32_t nPrb = fh_context.pdsch_ra_type0_info(i, ue_grp_index).num_prb;

                                    auto iter = std::find_if(idxlist.begin(), idxlist.end(),[&sym_prbs, &prbs, startPrb, nPrb](const auto& e){
                                        bool retval = (e < sym_prbs->prbs_size);
                                        if (retval) {
                                            auto& prb{prbs[e]};
                                            retval = (prb.common.startPrbc == startPrb && prb.common.numPrbc == nPrb);
                                        }
                                        return retval;
                                    });
                                    if (iter != idxlist.end()){
                                        auto& prb{prbs[*iter]};
                                        if (likely(!pm_enabled)) {
                                            prb.common.portMask |= calculate_dmrs_port_mask(ue.dmrsPortBmsk, ue.scid, ue.nlAbove16);
                                            prb.common.pdschPortMask |= calculate_dmrs_port_mask(ue.dmrsPortBmsk, ue.scid, ue.nlAbove16);
                                            //NVLOGD_FMT(TAG, "{}:{} RA Type-0: ue.rnti {} ue.dmrsPortBmsk {} ue.scid {}", __FILE__, __LINE__, ue.rnti, ue.dmrsPortBmsk, ue.scid);
                                            track_eaxcids_fh(ue, prb.common);
                                        } else {
                                            update_pm_weights_fh(ue, pm_map, pmi_bf_pdu, prb.common, cell_index, mmimo_enabled);
                                        }
                                    }
                                }//for 0->num_ra_type0_info
                            }// RA type 0
                            if(grp.resourceAlloc == 1) //RA type 1
                            {
                                auto& idxlist{sym_prbs->symbols[pdschSym][channel_type::PDSCH_CSIRS]};

                                uint16_t rb_idx = grp.startPrb;
                                uint16_t csirs_remap = fh_context.csirs_remap(cell_index, pdschSym*num_dl_prb + rb_idx) & 0xFFF;
                                bool use_alt_prb = (fh_context.csirs_remap(cell_index, pdschSym*num_dl_prb + rb_idx) & (1 << 15)) ? true : false;
                                uint32_t tempnPrb = 0;
                                uint32_t tempStartPrb = rb_idx;
                                uint32_t counter = grp.nPrb;
                                while(counter) {
                                    while(((csirs_remap == (fh_context.csirs_remap(cell_index, pdschSym*num_dl_prb + rb_idx) & 0xFFF)) ||
                                        (use_alt_prb && (csirs_remap == (fh_context.csirs_remap(cell_index, pdschSym*num_dl_prb + rb_idx + 1) & 0xFFF))))
                                        && (counter > 0)) {
                                        tempnPrb++; rb_idx++; counter--;
                                        continue;
                                    }

                                    //NVLOGD_FMT(TAG, "counter = {}, tempnPrb={}, rb_idx={}, pdschSym={} numSym={}", counter, tempnPrb, rb_idx, pdschSym, numSym);

                                    auto iter = std::find_if(idxlist.begin(), idxlist.end(),[&sym_prbs, &prbs, &grp, &tempStartPrb, &tempnPrb](const auto& e){
                                        bool retval = (e < sym_prbs->prbs_size);
                                        if (retval) {
                                            auto& prb{prbs[e]};
                                            if(prb.common.useAltPrb)
                                            {
                                                retval = (prb.common.startPrbc == grp.startPrb && prb.common.numPrbc == ((grp.nPrb + 1) >> 2 ));
                                            }
                                            else
                                            {
                                                retval = (prb.common.startPrbc == tempStartPrb && prb.common.numPrbc == tempnPrb);
                                            }
                                        }
                                        return retval;
                                    });
                                    if (iter != idxlist.end()){
                                        auto& prb{prbs[*iter]};
                                            if (likely(!pm_enabled)) {
                                            prb.common.portMask |= calculate_dmrs_port_mask(ue.dmrsPortBmsk, ue.scid, ue.nlAbove16);
                                            prb.common.pdschPortMask |= calculate_dmrs_port_mask(ue.dmrsPortBmsk, ue.scid, ue.nlAbove16);
                                            //NVLOGD_FMT(TAG, "{}:{} ue.rnti {} ue.dmrsPortBmsk {} ue.scid {}", __FILE__, __LINE__, ue.rnti, ue.dmrsPortBmsk, ue.scid);
                                            track_eaxcids_fh(ue, prb.common);
                                            //NVLOGD_FMT(TAG, "##### channel_type=PDSCH_CSIRS prb_info={}, portMask = {}", static_cast<void *>(&prb), static_cast<uint64_t>(prb.common.portMask));
                                        } else {
                                            update_pm_weights_fh(ue, pm_map, pmi_bf_pdu, prb.common, cell_index, mmimo_enabled);
                                        }
                                    }
                                    tempStartPrb = rb_idx;
                                    tempnPrb = 0;
                                    csirs_remap = fh_context.csirs_remap(cell_index, pdschSym*num_dl_prb + rb_idx) & 0xFFF;
                                    use_alt_prb = (fh_context.csirs_remap(cell_index, pdschSym*num_dl_prb + rb_idx) & (1 << 15)) ? true : false;
                                }
                            }
                            else
                            {// RA type 0
                                auto& idxlist{sym_prbs->symbols[grp.pdschStartSym][channel_type::PDSCH_CSIRS]};
                                for(uint32_t i=0; i < fh_context.pdsch_num_ra_type0_info(ue_grp_index); i++)
                                {
                                    uint32_t startPrb = fh_context.pdsch_ra_type0_info(i, ue_grp_index).start_prb + grp.startPrb;
                                    uint32_t nPrb = fh_context.pdsch_ra_type0_info(i, ue_grp_index).num_prb;

                                    auto iter = std::find_if(idxlist.begin(), idxlist.end(),[&sym_prbs, &prbs, startPrb, nPrb](const auto& e){
                                        bool retval = (e < sym_prbs->prbs_size);
                                        if (retval) {
                                            auto& prb{prbs[e]};
                                            retval = (prb.common.startPrbc == startPrb && prb.common.numPrbc == nPrb);
                                        }
                                        return retval;
                                    });
                                    if (iter != idxlist.end()){
                                        auto& prb{prbs[*iter]};
                                        if (likely(!pm_enabled)) {
                                            prb.common.portMask |= calculate_dmrs_port_mask(ue.dmrsPortBmsk, ue.scid, ue.nlAbove16);
                                            prb.common.pdschPortMask |= calculate_dmrs_port_mask(ue.dmrsPortBmsk, ue.scid, ue.nlAbove16);
                                            //NVLOGC_FMT(TAG, "{}:{} RA Type-0: ue.rnti {} ue.dmrsPortBmsk {} ue.scid {}", __FILE__, __LINE__, ue.rnti, ue.dmrsPortBmsk, ue.scid);
                                            track_eaxcids_fh(ue, prb.common);
                                        } else {
                                            update_pm_weights_fh(ue, pm_map, pmi_bf_pdu, prb.common, cell_index, mmimo_enabled);
                                        }
                                    }
                                }//for 0->num_ra_type0_info
                            }
                            pdschSym++;
                        } //while loop
                    } // mmimo_enabled
                    else
                    {
                        if(grp.resourceAlloc == 1) //RA type 1
                        {
                            auto& idxlist{sym_prbs->symbols[grp.pdschStartSym][channel_type::PDSCH]};

                            auto iter = std::find_if(idxlist.begin(), idxlist.end(),[&sym_prbs, &prbs, &grp](const auto& e){
                                bool retval = (e < sym_prbs->prbs_size);
                                if (retval) {
                                    auto& prb{prbs[e]};
                                    if(prb.common.useAltPrb)
                                    {
                                        retval = (prb.common.startPrbc == grp.startPrb && prb.common.numPrbc == ((grp.nPrb + 1) >> 2 ));
                                    }
                                    else
                                    {
                                        retval = (prb.common.startPrbc == grp.startPrb && prb.common.numPrbc == grp.nPrb);
                                    }
                                }
                                return retval;
                            });
                            if (iter != idxlist.end()){
                                auto& prb{prbs[*iter]};
                                    if (likely(!pm_enabled)) {
                                    prb.common.portMask |= calculate_dmrs_port_mask(ue.dmrsPortBmsk, ue.scid, ue.nlAbove16);
                                    //NVLOGD_FMT(TAG, "{}:{} ue.rnti {} ue.dmrsPortBmsk {} ue.scid {}", __FILE__, __LINE__, ue.rnti, ue.dmrsPortBmsk, ue.scid);
                                    track_eaxcids_fh(ue, prb.common);
                                    //NVLOGD_FMT(TAG, "##### prb_info={}, portMask = {}", static_cast<void *>(&prb), static_cast<uint64_t>(prb.common.portMask));
                                } else {
                                    update_pm_weights_fh(ue, pm_map, pmi_bf_pdu, prb.common, cell_index, mmimo_enabled);
                                }
                            }
                        }//RA type 1
                        else
                        {// RA type 0
                            auto& idxlist{sym_prbs->symbols[grp.pdschStartSym][channel_type::PDSCH]};
                            for(uint32_t i=0; i < fh_context.pdsch_num_ra_type0_info(ue_grp_index); i++)
                            {
                                uint32_t startPrb = fh_context.pdsch_ra_type0_info(i, ue_grp_index).start_prb + grp.startPrb;
                                uint32_t nPrb = fh_context.pdsch_ra_type0_info(i, ue_grp_index).num_prb;

                                auto iter = std::find_if(idxlist.begin(), idxlist.end(),[&sym_prbs, &prbs, startPrb, nPrb](const auto& e){
                                    bool retval = (e < sym_prbs->prbs_size);
                                    if (retval) {
                                        auto& prb{prbs[e]};
                                        retval = (prb.common.startPrbc == startPrb && prb.common.numPrbc == nPrb);
                                    }
                                    return retval;
                                });
                                if (iter != idxlist.end()){
                                    auto& prb{prbs[*iter]};
                                    if (likely(!pm_enabled)) {
                                        prb.common.portMask |= calculate_dmrs_port_mask(ue.dmrsPortBmsk, ue.scid, ue.nlAbove16);
                                        //NVLOGD_FMT(TAG, "{}:{} ue.rnti {} ue.dmrsPortBmsk {} ue.scid {}", __FILE__, __LINE__, ue.rnti, ue.dmrsPortBmsk, ue.scid);
                                        track_eaxcids_fh(ue, prb.common);
                                    } else {
                                        update_pm_weights_fh(ue, pm_map, pmi_bf_pdu, prb.common, cell_index, mmimo_enabled);
                                    }
                                }
                            }//for 0->num_ra_type0_info
                        }// RA type 0
                    }
                } // new grp = false

                }
                break;
        }
    }

    struct CsirsFhMetaData {
        inline static constexpr float rho_vals[4] = {0.5f, 0.5f, 1, 3};
    };


    inline void csirs_all_sym_ru_handler(slot_info_t* sym_prbs, nv::slot_detail_t* slot_detail, uint16_t num_dl_prb) {
        auto& prbs{sym_prbs->prbs};
        bool value = ifAnySymbolPresent(sym_prbs->symbols, DL_CHANNEL_MASK);
        if (value)
            return;

        prbs[sym_prbs->prbs_size] = prb_info_t(0, num_dl_prb);
        sym_prbs->prbs_size++;
        std::size_t index{sym_prbs->prbs_size - 1};
        prb_info_t& prb_info{prbs[index]};

        uint8_t start_symbol = (slot_detail == nullptr ? 0: slot_detail->start_sym_dl); 
        
        prb_info.common.direction = fh_dir_t::FH_DIR_DL;
        // prb_info.common.numSymbols = OFDM_SYMBOLS_PER_SLOT;
        prb_info.common.numSymbols = (slot_detail == nullptr || slot_detail->max_dl_symbols == 0 ? OFDM_SYMBOLS_PER_SLOT: slot_detail->max_dl_symbols);

        update_prb_sym_list(*sym_prbs, index, start_symbol, prb_info.common.numSymbols, channel_type::CSI_RS, ru_type::SINGLE_SECT_MODE);
    }

    inline void updateCsirsReSymbMap(uint16_t num_dl_prb, uint16_t* reMapRow, uint16_t& symbolMap, uint8_t l, uint16_t reMask, uint16_t startRb, uint16_t nRB, uint8_t rbInc) {
        uint32_t reMapIndex = l*num_dl_prb + startRb;
        NVLOGD_FMT(TAG, "CSIRS reMap start [{}] 0x{:x} rbInc {}", reMapIndex, reMapRow[reMapIndex], rbInc);
        
        // Hot-path split: rbInc=1 is the common case; rbInc=2 handles density 0.5 bit.
        auto* reMapPtr = &reMapRow[reMapIndex];
        const int endRb = startRb + nRB;
        if (likely(rbInc == 1)) {
            for (int idxRb = startRb; idxRb < endRb; idxRb++) {
                *reMapPtr++ |= reMask;
            }
        } else {
            const uint16_t combinedMask = reMask | (1 << 15);
            for (int idxRb = startRb; idxRb < endRb; idxRb += 2) {
                *reMapPtr |= combinedMask;
                reMapPtr += 2;
            }
        }
        symbolMap |= static_cast<uint16_t>(1U << l);
    }

    template <typename Func, typename ...Args>
    inline void updateCsirsPortApIndexes(Func func, Args&&... args) {
        func(std::forward<Args>(args)...);
    }

    inline void createCsiRsPrbInfo(slot_info_t* sym_prbs, uint16_t startRb, uint16_t nRB, uint8_t rbInc, uint16_t reMask, uint16_t l) {
        auto& prbs{sym_prbs->prbs};

        check_prb_info_size(sym_prbs->prbs_size);
        
        // Optimization: Branchless calculation for 32-port CSI-RS hot path
        const uint16_t nRB_adjusted = (rbInc == 2) ? ((nRB + 1) >> 1) : nRB;
        
        prbs[sym_prbs->prbs_size] = prb_info_t(startRb, nRB_adjusted);
        
        const std::size_t index = sym_prbs->prbs_size++;
        prb_info_t& prb_info{prbs[index]};
        prb_info.common.direction = fh_dir_t::FH_DIR_DL;
        prb_info.common.reMask = reMask;
        prb_info.common.useAltPrb = rbInc - 1;
    }

    inline void csirs_bfp_compress_fh_helper(
        uint16_t*                              reMapRow,
        uint16_t&                              symbolMap,
        tx_precoding_beamforming_t&            pc_and_bf,
        const cuphyCsirsRrcDynPrm_t&           csirs_param,
        const pm_group*                        pm_grp,
        uint16_t                               cell_idx,
        slot_info_t*                           sym_prbs,
        bool                                   config_options_precoding_enabled,
        bool                                   config_options_bf_enabled,
        bool                                   bf_enabled,
        const csirs_lookup_api::CsirsPortData* csirs_port_data,
        uint16_t                               num_dl_prb,
        uint16_t                               startRb,
        uint16_t                               nRB,
        uint8_t                                rbInc,
        bool                                   mmimo_enabled)
    {
        auto& prbs{sym_prbs->prbs};
        const uint8_t num_ports = csirs_port_data->num_ports;
        uint16_t reMask = 0;
        uint16_t l_mask = 0;
        
        // Optimization: Cache CSI-RS parameter reference
        const bool is_zp_csirs = (csirs_param.csiType == cuphyCsiType_t::ZP_CSI_RS);
        
        for(uint8_t port = 0; port < num_ports; port++) {
            reMask |= csirs_port_data->port_tx_locations[port].re_mask;
            l_mask |= csirs_port_data->port_tx_locations[port].symbol_mask;
        }
        const uint16_t validSymbolMask = static_cast<uint16_t>((1U << OFDM_SYMBOLS_PER_SLOT) - 1U);
        l_mask &= validSymbolMask;

        uint8_t num_ap_indices = num_ports;
        uint64_t port_mask = (1ULL << num_ports) - 1ULL;
        if (config_options_precoding_enabled && csirs_param.enablePrcdBf)
        {
            num_ap_indices = pm_grp->csirs_list[csirs_param.pmwPrmIdx].nPorts;
            port_mask = (1ULL << num_ap_indices) - 1ULL;
        }
        else if (config_options_bf_enabled)
        {
            const uint8_t dig_bf_interfaces = pc_and_bf.dig_bf_interfaces;
            num_ap_indices = dig_bf_interfaces;
            port_mask = (1ULL << dig_bf_interfaces) - 1ULL;
        }
 
        if (unlikely(is_zp_csirs)) {
            // ZP-CSI-RS still requires reMap/symbolMap updates, but no prb_info/beam work.
            while (l_mask != 0) {
                const uint8_t l = static_cast<uint8_t>(__builtin_ctz(static_cast<unsigned int>(l_mask)));
                l_mask &= static_cast<uint16_t>(l_mask - 1);
                updateCsirsReSymbMap(num_dl_prb, reMapRow, symbolMap, l, reMask, startRb, nRB, rbInc);
            }
            return;
        }

        while (l_mask != 0) {
            const uint8_t l = static_cast<uint8_t>(__builtin_ctz(static_cast<unsigned int>(l_mask)));
            l_mask &= static_cast<uint16_t>(l_mask - 1);
            updateCsirsReSymbMap(num_dl_prb, reMapRow, symbolMap, l, reMask, startRb, nRB, rbInc);

            createCsiRsPrbInfo(sym_prbs, startRb, nRB, rbInc, reMask, l);
            std::size_t index{sym_prbs->prbs_size - 1};
            prb_info_t& prb_info{prbs[index]};
            prb_info.common.numApIndices = num_ap_indices;
            prb_info.common.portMask = port_mask;
            NVLOGD_FMT(TAG, "CSI-RS new Grp dig_bf_interfaces={}, prb_info.common.portMask {}",
                        pc_and_bf.dig_bf_interfaces, static_cast<uint32_t>(prb_info.common.portMask));
            //if(csi_rs_params[csirs_idx].csiType == cuphyCsiType_t::TRS)
            //    prb_info.common.numApIndices *= 2;
            update_prb_sym_list(*sym_prbs, index, l, 1, channel_type::CSI_RS, OTHER_MODE);
            if (bf_enabled)
            {
                const bool modcomp_enabled = false;
                update_beam_list_csirs(prb_info.beams_array, prb_info.beams_array_size, pc_and_bf, prb_info, cell_idx, mmimo_enabled, modcomp_enabled);
            }
        }
    }

    template <bool config_options_bf_enabled>
    inline void csirs_mod_compress_fh_helper(uint16_t* reMapRow, uint16_t& symbolMap, tx_precoding_beamforming_t& pc_and_bf, const cuphyCsirsRrcDynPrm_t &csirs_param, uint16_t cell_idx, slot_info_t* sym_prbs,
        bool bf_enabled,
        const csirs_lookup_api::CsirsPortData* csirs_port_data, uint16_t num_dl_prb, uint16_t startRb, uint16_t nRB, uint8_t rbInc, bool mmimo_enabled, aerial_fh::UserDataCompressionMethod dl_comp_method, float bw_scaler = 0.0f) {
        
        auto& prbs{sym_prbs->prbs};
        const uint8_t num_ports = csirs_port_data->num_ports;
        
        // Optimization: Cache bw_scaler to avoid repeated function calls (critical for 32-port CSI-RS)
        const float bw_scaler_val = (bw_scaler > 0.0f) ? bw_scaler : getBwScaler(num_dl_prb);
        
        // Optimization: Cache CSI-RS parameter and check modcomp once
        const bool is_zp_csirs = (csirs_param.csiType == cuphyCsiType_t::ZP_CSI_RS);

        const bool is_modcomp = (dl_comp_method == comp_method::MODULATION_COMPRESSION);
        if (unlikely(!is_modcomp)) {
            NVLOGD_FMT(TAG, "Do not update mod comp info for {} dl_comp_method {}", __func__, static_cast<uint32_t>(dl_comp_method));
        }

        // Optimization: Cache port location array pointer to reduce indirection
        const auto* port_tx_locations = csirs_port_data->port_tx_locations;
        const uint16_t validSymbolMask = static_cast<uint16_t>((1U << OFDM_SYMBOLS_PER_SLOT) - 1U);

        if (unlikely(is_zp_csirs)) {
            // ZP-CSI-RS still requires reMap/symbolMap updates, but no prb_info/beam/modcomp work.
            for (uint8_t port = 0; port < num_ports; port++) {
                uint16_t symbolMask = static_cast<uint16_t>(port_tx_locations[port].symbol_mask & validSymbolMask);
                const uint16_t reMask = port_tx_locations[port].re_mask;

                while (symbolMask != 0)
                {
                    const uint8_t l = static_cast<uint8_t>(__builtin_ctz(static_cast<unsigned int>(symbolMask)));
                    symbolMask &= static_cast<uint16_t>(symbolMask - 1);
                    updateCsirsReSymbMap(num_dl_prb, reMapRow, symbolMap, l, reMask, startRb, nRB, rbInc);
                }
            }
            return;
        }

        // Performance: Split modcomp/non-modcomp branches to avoid per-iteration branching
        // in the hot inner loop (per-port x per-symbol). The modcomp path calls
        // update_mod_comp_info_common/section which the non-modcomp path skips entirely.
        if (likely(is_modcomp)) {
            for (uint8_t port = 0; port < num_ports; port++) {
                uint16_t symbolMask = static_cast<uint16_t>(port_tx_locations[port].symbol_mask & validSymbolMask);
                const uint16_t reMask = port_tx_locations[port].re_mask;
                
                while (symbolMask != 0)
                {
                    const uint8_t l = static_cast<uint8_t>(__builtin_ctz(static_cast<unsigned int>(symbolMask)));
                    symbolMask &= static_cast<uint16_t>(symbolMask - 1);
                    updateCsirsReSymbMap(num_dl_prb, reMapRow, symbolMap, l, reMask, startRb, nRB, rbInc);
                    createCsiRsPrbInfo(sym_prbs, startRb, nRB, rbInc, reMask, l);
                    const std::size_t index = sym_prbs->prbs_size - 1;
                    prb_info_t& prb_info{prbs[index]};
                    overlap_csirs_port_info_t& overlap_csirs_port_info = sym_prbs->overlap_csirs_port_info[index];
                    overlap_csirs_port_info.num_ports = num_ports;
                    
                    if constexpr (config_options_bf_enabled)
                    {
                        prb_info.common.numApIndices = 1;
                        prb_info.common.portMask = static_cast<uint64_t>(0x1ULL << port);
                        prb_info.common.ap_index = port;
                    }
                    
                    update_prb_sym_list(*sym_prbs, index, l, 1, channel_type::CSI_RS, OTHER_MODE);
                    if (bf_enabled)
                    {
                        const bool modcomp_enabled = true;
                        update_beam_list_csirs(prb_info.beams_array, prb_info.beams_array_size, pc_and_bf, prb_info, cell_idx, mmimo_enabled, modcomp_enabled);
                    }
                    update_mod_comp_info_common(prb_info, bw_scaler_val);
                    update_mod_comp_info_section(prb_info, reMask, csirs_param.beta, CUPHY_QAM_4, DEFAULT_CSF); // QPSK
                }
            }
        } else {
            for (uint8_t port = 0; port < num_ports; port++) {
                uint16_t symbolMask = static_cast<uint16_t>(port_tx_locations[port].symbol_mask & validSymbolMask);
                const uint16_t reMask = port_tx_locations[port].re_mask;
                
                while (symbolMask != 0)
                {
                    const uint8_t l = static_cast<uint8_t>(__builtin_ctz(static_cast<unsigned int>(symbolMask)));
                    symbolMask &= static_cast<uint16_t>(symbolMask - 1);
                    updateCsirsReSymbMap(num_dl_prb, reMapRow, symbolMap, l, reMask, startRb, nRB, rbInc);
                    createCsiRsPrbInfo(sym_prbs, startRb, nRB, rbInc, reMask, l);
                    const std::size_t index = sym_prbs->prbs_size - 1;
                    prb_info_t& prb_info{prbs[index]};
                    overlap_csirs_port_info_t& overlap_csirs_port_info = sym_prbs->overlap_csirs_port_info[index];
                    overlap_csirs_port_info.num_ports = num_ports;
                    
                    if constexpr (config_options_bf_enabled)
                    {
                        prb_info.common.numApIndices = 1;
                        prb_info.common.portMask = static_cast<uint64_t>(0x1ULL << port);
                        prb_info.common.ap_index = port;
                    }
                    
                    update_prb_sym_list(*sym_prbs, index, l, 1, channel_type::CSI_RS, OTHER_MODE);
                    if (bf_enabled)
                    {
                        // Even in non-modcomp data compression, this function uses per-port
                        // iteration (portMask = 1 port), so modcomp_enabled=true is required
                        // to defer beam-sent marking until the last port.
                        const bool modcomp_enabled = true;
                        update_beam_list_csirs(prb_info.beams_array, prb_info.beams_array_size, pc_and_bf, prb_info, cell_idx, mmimo_enabled, modcomp_enabled);
                    }
                }
            }
        }
    }

   template <bool mplane_configured_ru_type, bool config_options_precoding_enabled, bool config_options_bf_enabled>
    void update_fh_params_csirs_remap(IFhCallbackContext& fh_context, nv::slot_detail_t* slot_detail, const CsirsFhParamsView& csirs_fh_params, bool &csirs_compact_mode)
    {
        uint32_t cell_idx              = csirs_fh_params.cell_idx();
        int32_t cuphy_params_cell_idx  = csirs_fh_params.cuphy_params_cell_idx();
        bool bf_enabled                = csirs_fh_params.bf_enabled();
        uint16_t num_dl_prb            = csirs_fh_params.num_dl_prb();
        bool mmimo_enabled             = csirs_fh_params.mmimo_enabled();

        uint32_t num_params = 0;
        uint32_t csirs_params_offset =  0;
        csirs_lookup_api::CsirsLookup& lookup = csirs_lookup_api::CsirsLookup::getInstance();
        const cuphyCsirsRrcDynPrm_t *csi_rs_params = NULL;
        ru_type ru;
        aerial_fh::UserDataCompressionMethod dl_comp_method = aerial_fh::UserDataCompressionMethod::BLOCK_FLOATING_POINT;
        uint8_t num_eaxcids = 0;
        if (mplane_configured_ru_type)
        {
            nv::PHYDriverProxy& phyDriver = nv::PHYDriverProxy::getInstance();
            const auto& mplane_info = phyDriver.getMPlaneConfig(cell_idx);
            ru = mplane_info.ru;
            dl_comp_method = mplane_info.dl_comp_meth;
            num_eaxcids = mplane_info.eAxC_ids[slot_command_api::channel_type::CSI_RS].size();
            NVLOGD_FMT(TAG, "{}cell-idx={} num_eaxcids={}",__func__,cell_idx,num_eaxcids);
        }
        else
        {
            ru = OTHER_MODE;
        }

        // Optimization: Cache cell index list reference for loop
        uint32_t cell_info_idx = 0;
        const auto& cell_idx_list = fh_context.csirs_cell_index_list();
        const size_t cell_list_size = cell_idx_list.size();
        
        while(cell_info_idx < cell_list_size && cell_idx_list[cell_info_idx] != cell_idx)
            ++cell_info_idx;


        if(cuphy_params_cell_idx != -1)
        {
            csi_rs_params = fh_context.pdsch_csirs_rrc_dyn_params();
            num_params = fh_context.pdsch_cell_num_csirs_params(cuphy_params_cell_idx);
            csirs_params_offset = fh_context.pdsch_cell_csirs_params_offset(cuphy_params_cell_idx);
            NVLOGD_FMT(TAG, "{}:Looking for CSI-RS RrcDynPrms in PDSCH - cell-idx={} cell_dyn_indx={} num CSI-RS params={} CSI-RS offset={}",
                __func__,cell_idx,cuphy_params_cell_idx,num_params,csirs_params_offset);
        }
        else
        {
            csi_rs_params = fh_context.csirs_rrc_dyn_params();
            num_params = fh_context.csirs_cell_num_rrc_params(cell_info_idx);
            csirs_params_offset = fh_context.csirs_cell_rrc_params_offset(cell_info_idx);
            NVLOGD_FMT(TAG, "{}:Looking for CSI-RS RrcDynPrms in CSI-RS cell-idx={} num CSI-RS params={} CSI-RS offset={}",__func__,cell_idx,num_params,csirs_params_offset);
        }

        auto pm_grp = fh_context.pm_group_ptr();
        auto* csirs_remap_row = fh_context.csirs_remap_row(cell_idx);
        auto& csirs_symbol_map_ref = fh_context.csirs_symbol_map_ref(cell_idx);
        auto sym_prbs{csirs_fh_params.sym_prb_info()};
        auto& prbs{sym_prbs->prbs};
        //pc_and_bf_idx gets incremented only when csirs type is not ZP_CSI_RS
        int pc_and_bf_idx = fh_context.csirs_cell_rrc_params_offset(cell_info_idx);
        switch (ru) {
            case SINGLE_SECT_MODE:
                csirs_all_sym_ru_handler(sym_prbs, slot_detail, num_dl_prb);
            break;
            case MULTI_SECT_MODE:
            case OTHER_MODE:
            default: {
                    // Optimization: Pre-compute BW scaler for 32-port CSI-RS (common case)
                    const float bw_scaler_cached = getBwScaler(num_dl_prb);
                    
                    // Hoist loop-invariant values for 32-port optimization
                    const bool is_multi_sect = (ru == MULTI_SECT_MODE);
                    const bool use_bfp_helper = (dl_comp_method == comp_method::NO_COMPRESSION) ||
                        (dl_comp_method == comp_method::BLOCK_FLOATING_POINT);
                    const bool use_modcomp_helper = (dl_comp_method == comp_method::MODULATION_COMPRESSION);
                    
                    for(int j = 0; j < num_params; j++)
                    {
                        const uint16_t csirs_idx = csirs_params_offset + j;
                        NVLOGD_FMT(TAG, "CSI-RS parameter {} of {} , offset {}, csirs_idx {}", j, num_params, csirs_params_offset, csirs_idx);

                        // Optimization: Cache CSI-RS parameter access to reduce pointer indirection
                        const auto& csirs_param = csi_rs_params[csirs_idx];
                        
                        if(csirs_param.row > CUPHY_CSIRS_SYMBOL_LOCATION_TABLE_LENGTH)
                        {
                            NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "row {}  > CUPHY_CSIRS_SYMBOL_LOCATION_TABLE_LENGTH", csirs_param.row);
                            return;
                        }
                        const csirs_lookup_api::CsirsPortData *csirs_port_data = nullptr;

                        if(false == lookup.getPortInfo(csirs_param.row, csirs_param.freqDomain, csirs_param.symbL0, csirs_param.symbL1, csirs_port_data))
                        {
                            NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{} getPortInfo failed for row={} freqDomain={} symbL0={} symbL1={}", __func__, csirs_param.row, csirs_param.freqDomain, csirs_param.symbL0, csirs_param.symbL1);
                            return;
                        }
                        
                        const uint8_t num_ports = csirs_port_data->num_ports;
                        uint16_t nRB = csirs_param.nRb;
                        num_eaxcids = num_eaxcids == 0? num_ports : num_eaxcids;
                        uint16_t startRb = csirs_param.startRb;
                        
                        // Adjust startRb/nRB if generating odd RBs but startRb is even, or generating even RBs but startRb is odd.
                        // Optimization: Hoist frequency density check for better branch prediction
                        const uint8_t freq_density = csirs_param.freqDensity;
                        const bool startRb_is_odd = (startRb & 0x1);
                        uint8_t rbInc = (freq_density < 2) ? 2 : 1;
                        
                        if (((freq_density == 1) && !startRb_is_odd) ||
                            ((freq_density == 0) && startRb_is_odd))
                        {
                            startRb += 1;
                            nRB -= 1;
                        }
                        // Separate KBar and LBar loops.
                        // Note: this only works due to the structure of 38.211 Table 7.4.1.5.3-1.
                        auto& csirs_pc_and_bf = fh_context.csirs_pc_and_bf(pc_and_bf_idx);
                        if((num_ports > num_eaxcids) && is_multi_sect)
                        {
                            csirs_compact_mode = true;
                            csirs_mod_compress_fh_helper<config_options_bf_enabled>(csirs_remap_row, csirs_symbol_map_ref, csirs_pc_and_bf, csirs_param, cell_idx, sym_prbs,
                                    bf_enabled,
                                    csirs_port_data, num_dl_prb, startRb, nRB, rbInc, mmimo_enabled, dl_comp_method, bw_scaler_cached);
                        }
                        else
                        {
                            if (use_bfp_helper) {
                                csirs_bfp_compress_fh_helper(csirs_remap_row, csirs_symbol_map_ref, csirs_pc_and_bf, csirs_param, pm_grp, cell_idx, sym_prbs,
                                    config_options_precoding_enabled, config_options_bf_enabled, bf_enabled, 
                                    csirs_port_data, num_dl_prb, startRb, nRB, rbInc,  mmimo_enabled);
                            } else if (use_modcomp_helper) {
                                csirs_mod_compress_fh_helper<config_options_bf_enabled>(csirs_remap_row, csirs_symbol_map_ref, csirs_pc_and_bf, csirs_param, cell_idx, sym_prbs,
                                    bf_enabled,
                                    csirs_port_data, num_dl_prb, startRb, nRB, rbInc, mmimo_enabled,dl_comp_method, bw_scaler_cached);
                            }
                        }
                        // Increment only for non-ZP CSI-RS (use cached parameter)
                        if(likely(csirs_param.csiType != cuphyCsiType_t::ZP_CSI_RS))
                        {
                            pc_and_bf_idx++;
                        }
                    }
            }
            break;
        }
    }

    /**
     * @brief Append a beam ID to the CSI-RS beam list and update static BFW weights.
     *
     * @param[in,out] array       Beam ID array to append to.
     * @param[in,out] array_size  Current size of the array; incremented on return.
     * @param[in]     pmi_bf_pdu  Precoding / beamforming PDU.
     * @param[in,out] prb_info    PRB info to update with BFW metadata.
     * @param[in]     beam_id     Beam identifier to store and look up.
     * @param[in]     cell_idx    Cell index.
     */
    inline void add_beam_id_csirs(beamid_array_t& array, size_t& array_size, tx_precoding_beamforming_t& pmi_bf_pdu, prb_info_t& prb_info, uint16_t beam_id, int32_t cell_idx) {
        array[array_size] = beam_id;
        ++array_size;
        update_static_bf_wt_csi_rs(cell_idx, pmi_bf_pdu, prb_info, beam_id, false);
    }

    void update_beam_list_csirs(beamid_array_t& array, size_t& array_size, tx_precoding_beamforming_t& pmi_bf_pdu, prb_info_t& prb_info, int32_t cell_idx, bool mmimo_enabled, bool modcomp_enabled) {
        //NVLOGI_FMT(TAG, "{} TX BeamForming PDU num_prgs = {} dig_bf_interfaces ={}", __FUNCTION__, pmi_bf_pdu.num_prgs, pmi_bf_pdu.dig_bf_interfaces);
        
        // Optimization: Hoist invariants and branch once (mmimo_enabled is constant per call)
        const uint32_t num_prgs = static_cast<uint32_t>(pmi_bf_pdu.num_prgs);
        const uint32_t dig_bf_interfaces = static_cast<uint32_t>(pmi_bf_pdu.dig_bf_interfaces);
        if (unlikely((num_prgs == 0U) || (dig_bf_interfaces == 0U)))
        {
            return;
        }
        uint32_t idx = 0U;

        if (mmimo_enabled) {
            nv::PHYDriverProxy& phyDriver = nv::PHYDriverProxy::getInstance();
            const bool static_bfw_configured = phyDriver.l1_staticBFWConfigured(cell_idx);
            const bool sendOncePerBeam = !phyDriver.l1_get_send_static_bfw_wt_all_cplane();
            if (likely(static_bfw_configured)) {
                for (uint32_t prg = 0U; prg < num_prgs; ++prg) {
                    ++idx; // Skip PMI entry
                    for (uint32_t j = 0U; j < dig_bf_interfaces; ++j) {
                        const uint16_t beam_id = pmi_bf_pdu.pm_idx_and_beam_idx[idx++];
                        array[array_size] = beam_id;
                        ++array_size;
                        update_static_bf_wt_csi_rs_impl(cell_idx, pmi_bf_pdu, prb_info, beam_id, modcomp_enabled, phyDriver, static_bfw_configured, sendOncePerBeam);
                    }
                }
            } else {
                for (uint32_t prg = 0U; prg < num_prgs; ++prg) {
                    ++idx; // Skip PMI entry
                    for (uint32_t j = 0U; j < dig_bf_interfaces; ++j) {
                        array[array_size] = pmi_bf_pdu.pm_idx_and_beam_idx[idx++];
                        ++array_size;
                    }
                }
            }
        } else {
            for (uint32_t prg = 0U; prg < num_prgs; ++prg) {
                ++idx; // Skip PMI entry
                for (uint32_t j = 0U; j < dig_bf_interfaces; ++j) {
                    array[array_size] = pmi_bf_pdu.pm_idx_and_beam_idx[idx++];
                    ++array_size;
                }
            }
        }
        
#if 0
        NVLOGI_FMT(TAG, "{} TX BeamForming beam list size = {}", __FUNCTION__, array_size);
        for (std::size_t i = 0; i < array_size; i++) {
            NVLOGI_FMT(TAG, "{} beam [{}] = {}", __FUNCTION__, i, array_size);
        }
#endif
    }

    inline void update_fh_params_csirs(cuphyCsirsRrcDynPrm_t& csirs_inst, scf_fapi_tx_precoding_beamforming_t& pmi_bf_pdu, cell_sub_command& cell_cmd, bool bf_enabled, 
                                       enum ru_type ru, nv::slot_detail_t* slot_detail, int32_t cell_index) {
        auto sym_prbs{cell_cmd.sym_prb_info()};
        auto& prbs{sym_prbs->prbs};
        cuphyCdmType_t cdmType{csirs_inst.cdmType};
        uint8_t nSym = 0;
        if(ru == SINGLE_SECT_MODE)
        {
            bool value = ifAnySymbolPresent(sym_prbs->symbols, DL_CHANNEL_MASK);
            if(value)
                return;

            check_prb_info_size(sym_prbs->prbs_size);
            prbs[sym_prbs->prbs_size] = prb_info_t(0, 273);
            sym_prbs->prbs_size++;
            std::size_t index{sym_prbs->prbs_size - 1};
            prb_info_t& prb_info{prbs[index]};
            prb_info.common.direction = fh_dir_t::FH_DIR_DL;
            // prb_info.common.numSymbols = OFDM_SYMBOLS_PER_SLOT;
            prb_info.common.numSymbols = (slot_detail == nullptr || slot_detail->max_dl_symbols == 0 ? OFDM_SYMBOLS_PER_SLOT: slot_detail->max_dl_symbols);
            uint8_t start_symbol = (slot_detail == nullptr ? 0: slot_detail->start_sym_dl);
            update_prb_sym_list(*sym_prbs, index, start_symbol, prb_info.common.numSymbols, channel_type::CSI_RS, ru);
        }
        else
        {
            check_prb_info_size(sym_prbs->prbs_size);
            prbs[sym_prbs->prbs_size] = prb_info_t(csirs_inst.startRb, csirs_inst.nRb);
            sym_prbs->prbs_size++;
            std::size_t index{sym_prbs->prbs_size - 1};
            prb_info_t& prb_info{prbs[index]};

            prb_info.common.direction = fh_dir_t::FH_DIR_DL;

            if (bf_enabled) {
                update_beam_list(prb_info.beams_array, prb_info.beams_array_size, pmi_bf_pdu,false, prb_info, cell_index);
            }
            nSym = std::max(CSI_RS_L0_NUM_SYM[csirs_inst.row], CSI_RS_L_PRIME_NUM_SYM[csirs_inst.row]);
            update_prb_sym_list(*sym_prbs, index, csirs_inst.symbL0, nSym, channel_type::CSI_RS, ru);

            if(CSI_RS_L1_NUM_SYM[csirs_inst.row] != 0)
            {
                nSym = std::max(CSI_RS_L1_NUM_SYM[csirs_inst.row], CSI_RS_L_PRIME_NUM_SYM[csirs_inst.row]);
                update_prb_sym_list(*sym_prbs, index, csirs_inst.symbL1, nSym, channel_type::CSI_RS, ru);
            }
        }
    }

    inline void update_pm_weights_prbs(cuphyPdschUePrm_t& ue, cuphyPdschCellGrpDynPrm_t& cell_grp,
        prc_weights_list_t& list, pm_weight_map_t & pm_weight_map,
        prc_weights_idx_list_t& cache, scf_fapi_tx_precoding_beamforming_t& pmi_bf_pdu,
        prb_info_common_t& prbs, int32_t cell_index
        ) {
        uint16_t offset = 0;

        auto restore_defaults = [&ue, &prbs] () {
            ue.enablePrcdBf = false;
            if(ue.rnti == UINT16_MAX)  // For 4T4R O-RUs, SI-RNTI date with a single layer transmission will be replicated for the second antenna
                prbs.numApIndices += ue.nUeLayers * 2;
            else
                prbs.numApIndices += ue.nUeLayers;
        };

        for (uint16_t i = 0; i < pmi_bf_pdu.num_prgs; i++) {
#if DBG_PRECODER
            NVLOGC_FMT(TAG, "{} PMI Index={} cell Index={} final pmidx={}", __FUNCTION__, pmi_bf_pdu.pm_idx_and_beam_idx[i + offset], cell_index, pmi_bf_pdu.pm_idx_and_beam_idx[i + offset] | cell_index << 16);
#endif
            if (pmi_bf_pdu.pm_idx_and_beam_idx[i + offset] == 0 || pmi_bf_pdu.dig_bf_interfaces == 0) {  // dig_bf_interfaces == 0 means dynamic beamforming or ZP-CSI-RS, which doesn't need precoding
                offset+= (pmi_bf_pdu.dig_bf_interfaces + 1);
                restore_defaults();
                continue;
            }
            uint32_t pmi = pmi_bf_pdu.pm_idx_and_beam_idx[i + offset] | cell_index << 16; /// PMI Unused
            offset+= (pmi_bf_pdu.dig_bf_interfaces + 1);
            uint16_t matrix_index = UINT16_MAX;
            auto iter = std::find(cache.begin(), cache.end(), pmi);
            if (iter != cache.end()) {
                matrix_index = std::distance(cache.begin(), iter);
                prbs.numApIndices = std::max (prbs.numApIndices,reinterpret_cast<uint16_t&> (pm_weight_map[pmi].ports));
#if DBG_PRECODER
                NVLOGC_FMT(TAG, "{} PMI Index={} found in cache matrix_index={}", __FUNCTION__, pmi, matrix_index);
#endif
            } else {
                auto pmw_iter = pm_weight_map.find(pmi);
                if (pmw_iter == pm_weight_map.end()){
                    restore_defaults();
                    continue ;
                }
                auto& val = list[cell_grp.nPrecodingMatrices];
                matrix_index = cell_grp.nPrecodingMatrices;
                cache.push_back(pmi);
                prbs.numApIndices = std::max(prbs.numApIndices,reinterpret_cast<uint16_t&>(pmw_iter->second.ports));
                val.nPorts = pmw_iter->second.weights.nPorts;
                std::copy(pmw_iter->second.weights.matrix, pmw_iter->second.weights.matrix + (pmw_iter->second.layers * pmw_iter->second.ports), val.matrix);
                cell_grp.nPrecodingMatrices++;
#if DBG_PRECODER
                NVLOGC_FMT(TAG, "{} PMI Index={} not found in cache matrix_index={}", __FUNCTION__, cache.back(), matrix_index);
                for (uint i = 0; i < pmw_iter->second.layers; i++) {
                    for (uint j = 0; j < pmw_iter->second.ports; j++) {
                        __half2& value{val.matrix[i* pmw_iter->second.ports + j]};
                        NVLOGC_FMT(TAG, " layer {} port {} index {}. real = {} imag = {}", i, j, i * pmw_iter->second.ports + j,  static_cast<float>(value.x), static_cast<float>(value.y));
                    }
                }
#endif
            }

            if (matrix_index != UINT16_MAX){
                ue.pmwPrmIdx = matrix_index;
                cell_grp.pPmwPrms = list.data();
            }
        }
    }

    inline void update_new_csirs_pm(const scf_fapi_csi_rsi_pdu_t& msg_csirs, cuphyCsirsRrcDynPrm_t& csirs_rrc_dyn_params,  pm_group* prec_group, const pm_weight_map_t& pm_map,  nv::phy_config_option& config_options, int32_t cell_index) {
        const auto& pdu = *reinterpret_cast<const scf_fapi_tx_precoding_beamforming_t*>(&msg_csirs.pc_and_bf);
        csirs_rrc_dyn_params.enablePrcdBf = config_options.precoding_enabled;

        auto default_values = [&csirs_rrc_dyn_params]() {
            csirs_rrc_dyn_params.enablePrcdBf = false;
        };

        uint16_t offset = 0;

        for (uint16_t i = 0; i < pdu.num_prgs; i++) {
            uint16_t pdu_pmi = pdu.pm_idx_and_beam_idx[i + offset];
            uint32_t cache_pmi = pdu_pmi | cell_index << 16; /// PMI Unused
            csirs_rrc_dyn_params.enablePrcdBf = csirs_rrc_dyn_params.enablePrcdBf && (pdu_pmi != 0);
            if (csirs_rrc_dyn_params.enablePrcdBf) {
                auto pmw_iter = pm_map.find(cache_pmi);
                if (pmw_iter == pm_map.end()){
                    default_values();
                    continue ;
                }

                if (pmw_iter->second.layers != 1) {
                    default_values();
                    continue;
                }

                //auto iter = std::find_if(prec_group->csirs_pmw_idx_cache.begin(), prec_group->csirs_pmw_idx_cache.begin()+ prec_group->nCacheEntries, [&cache_pmi](const auto& e ) {
                //    return e.pmwIdx == cache_pmi;
                //});
                auto iter = std::find_if(prec_group->csirs_pmw_idx_cache.begin(), prec_group->csirs_pmw_idx_cache.end(), [&cache_pmi](const auto& e ) {
                    return e.pmwIdx == cache_pmi;
                });

                if (iter == prec_group->csirs_pmw_idx_cache.end()) {
                    auto& cache_entry = prec_group->csirs_pmw_idx_cache.at(prec_group->nCacheEntries);
                    cache_entry.pmwIdx = cache_pmi;
                    cache_entry.nIndex = prec_group->nPmCsirs;
                    prec_group->nCacheEntries++;
                    auto& val = prec_group->csirs_list[prec_group->nPmCsirs];
                    val.nPorts = pmw_iter->second.weights.nPorts;
                    csirs_rrc_dyn_params.pmwPrmIdx = prec_group->nPmCsirs;
                    std::copy(pmw_iter->second.weights.matrix, pmw_iter->second.weights.matrix + (pmw_iter->second.layers * pmw_iter->second.ports), val.matrix);
                    prec_group->nPmCsirs++;
                } else {
                    csirs_rrc_dyn_params.pmwPrmIdx = iter->nIndex;
                }
            }
            offset+=(pdu.dig_bf_interfaces + 1);
        }
    }

    void update_cell_command(cell_group_command* cell_grp_cmd, cell_sub_command& cell_cmd, const scf_fapi_csi_rsi_pdu_t& msg, slot_indication & slotinfo, int32_t cell_index, cuphyCellStatPrm_t cell_params,
                                nv::phy_config_option& config_option, pm_weight_map_t& pm_map,uint32_t csirs_offset, bool pdsch_exist, uint16_t cell_stat_prm_idx, bool mmimo_enabled, nv::slot_detail_t*  slot_detail)
    {

        /* Following are the cases for CSI-RS and PDSCH -
         * ZP CSI-RS + PDSCH - PDSCH will look into the RrcDynPrms. CSI-RS pipeline is not called.
         * NZP CSI-RS only - only CSI-RS look into RrcDynPrms. PDSCH pipeline is not called.
         * PDSCH only - neither PDSCH nor CSI-RS look into RrcDynPrms. CSI-RS pipeline is not called. This function will not be called
         * NZP CSI-RS + PDSCH - both PDSCH & CSI-RS look into RrcDynPrms. Both pipelines called indepedently
         * ZP CSI-RS - no pipeline called. Return from top of this function
         */

        cuphyCsiType_t csi_type = static_cast<cuphyCsiType_t>(msg.csi_type);
        if((csi_type == cuphyCsiType_t::ZP_CSI_RS) && (pdsch_exist == false))
        {
            NVLOGD_FMT(TAG, "update_cell_cmd for CSI-RS: ZP CSI-RS & pdsch does not exit. return.");
            return;
        }

        cuphyCsirsRrcDynPrm_t* pdsch_rrc_dyn_params = NULL;
        cuphyCsirsRrcDynPrm_t* csirs_rrc_dyn_params = NULL;
        cell_cmd.slot.type = SLOT_DOWNLINK;
        cell_cmd.slot.slot_3gpp = slotinfo;
        cell_grp_cmd->slot.type = SLOT_DOWNLINK;
        cell_grp_cmd->slot.slot_3gpp = slotinfo;
        auto& staticCsiRsSlotNum = config_option.staticCsiRsSlotNum;

        if(pdsch_exist)
        {
            pdsch_params* pdsch_params  = cell_grp_cmd->get_pdsch_params();
            auto& pdsch_grp = pdsch_params->cell_grp_info;
            pdsch_grp.nCsiRsPrms++;
            pdsch_params->cell_dyn_info[pdsch_grp.nCells-1].csiRsPrmsOffset = csirs_offset;
            pdsch_params->cell_dyn_info[pdsch_grp.nCells-1].nCsiRsPrms++;

            pdsch_rrc_dyn_params = &pdsch_grp.pCsiRsPrms[pdsch_params->num_csirs_info];
            pdsch_params->num_csirs_info++;

            pdsch_rrc_dyn_params->startRb          = msg.start_rb + msg.bwp.bwp_start;
            pdsch_rrc_dyn_params->nRb              = msg.num_of_rbs;
            pdsch_rrc_dyn_params->freqDomain       = msg.freq_domain;
            pdsch_rrc_dyn_params->row              = msg.row;
            pdsch_rrc_dyn_params->symbL0           = msg.sym_l0;
            pdsch_rrc_dyn_params->symbL1           = msg.sym_l1;
            pdsch_rrc_dyn_params->freqDensity      = msg.freq_density;
            pdsch_rrc_dyn_params->scrambId         = msg.scrambling_id;
            pdsch_rrc_dyn_params->idxSlotInFrame   = ((staticCsiRsSlotNum > -1)? staticCsiRsSlotNum:slotinfo.slot_);
            pdsch_rrc_dyn_params->csiType          = static_cast<cuphyCsiType_t>(msg.csi_type);
            pdsch_rrc_dyn_params->cdmType          = static_cast<cuphyCdmType_t>(msg.cdm_type);
            pdsch_rrc_dyn_params->beta             = std::pow(10.0, (msg.tx_power.power_control_offset_ss - 1)*3.0/20.0);
            pdsch_rrc_dyn_params->enablePrcdBf     = false;
            pdsch_rrc_dyn_params->pmwPrmIdx        = 0;

            NVLOGD_FMT(TAG, "CSI-RS pdsch->cell_grp_info.nCsiRsPrms={}, cell_dyn_info[{}].nCsiRsPrms={}, offset={}",
                pdsch_params->cell_grp_info.nCsiRsPrms,pdsch_grp.nCells-1, pdsch_params->cell_dyn_info[pdsch_grp.nCells-1].nCsiRsPrms,csirs_offset);
            NVLOGD_FMT(TAG, "update_cell_cmd for CSI-RS: Added entry {} into rrc_dyn_params of PDSCH for CSI-Type={}",pdsch_params->num_csirs_info, +csi_type);
        }

        if(csi_type != cuphyCsiType_t::ZP_CSI_RS)
        {
            if(!check_bf_pc_params(msg.pc_and_bf.num_prgs, msg.pc_and_bf.dig_bf_interfaces, mmimo_enabled))
            {
                NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{} line {}: check_bf_pc_params failed: numPRGs={} digBFInterfaces={} mmimo_enabled={}",
                    __FUNCTION__, __LINE__, static_cast<uint16_t>(msg.pc_and_bf.num_prgs), static_cast<uint16_t>(msg.pc_and_bf.dig_bf_interfaces), mmimo_enabled);
                return;
            }

            cell_grp_cmd->create_if(channel_type::CSI_RS);
            csirs_params* csirs_params = cell_grp_cmd->csirs.get();
            if (csirs_params == nullptr)
            {
                NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "no csirs command");
                return;
            }

            if(csirs_params->symbolMapArray[cell_index])
                csirs_params->symbolMapArray[cell_index] = 0;

            auto it = std::find(csirs_params->cell_index_list.begin(),csirs_params->cell_index_list.end(), cell_index);
            if(it == csirs_params->cell_index_list.end())
            {
                csirs_params->cell_index_list.push_back(cell_index);
                csirs_params->phy_cell_index_list.push_back(cell_cmd.cell);
            }

            cuphyCsirsRrcDynPrm_t& csirs_rrc_dyn_params = csirs_params->csirsList[csirs_params->nCsirsRrcDynPrm];
            csirs_params->nCsirsRrcDynPrm++;

            csirs_rrc_dyn_params.startRb          = msg.start_rb;
            csirs_rrc_dyn_params.nRb              = msg.num_of_rbs;
            csirs_rrc_dyn_params.freqDomain       = msg.freq_domain;
            csirs_rrc_dyn_params.row              = msg.row;
            csirs_rrc_dyn_params.symbL0           = msg.sym_l0;
            csirs_rrc_dyn_params.symbL1           = msg.sym_l1;
            csirs_rrc_dyn_params.freqDensity      = msg.freq_density;
            csirs_rrc_dyn_params.scrambId         = msg.scrambling_id;
            csirs_rrc_dyn_params.idxSlotInFrame   = ((staticCsiRsSlotNum > -1)? staticCsiRsSlotNum:slotinfo.slot_);
            csirs_rrc_dyn_params.csiType          = static_cast<cuphyCsiType_t>(msg.csi_type);
            csirs_rrc_dyn_params.cdmType          = static_cast<cuphyCdmType_t>(msg.cdm_type);
            csirs_rrc_dyn_params.beta             = std::pow(10.0, (msg.tx_power.power_control_offset_ss - 1)*3.0/20.0);
            update_new_csirs_pm(msg, csirs_rrc_dyn_params, cell_grp_cmd->get_pm_group(), pm_map, config_option, cell_index);
            if(pdsch_exist)
            {
                pdsch_rrc_dyn_params->enablePrcdBf     = csirs_rrc_dyn_params.enablePrcdBf;
                pdsch_rrc_dyn_params->pmwPrmIdx        = csirs_rrc_dyn_params.pmwPrmIdx;
            }

            csirs_params->cellInfo[csirs_params->nCells].nRrcParams++;
            csirs_params->cellInfo[csirs_params->nCells].cellPrmStatIdx = cell_stat_prm_idx;
            NVLOGD_FMT(TAG, "CSI-RS pCellParam[{}].rrcParamOffset={},nRrcParams={}",csirs_params->nCells,
                csirs_params->cellInfo[csirs_params->nCells].rrcParamsOffset,csirs_params->cellInfo[csirs_params->nCells].nRrcParams);
#if 0
        if(rrc_dyn_params.csiType != cuphyCsiType_t::ZP_CSI_RS)
        {
            if (unlikely(params->nCsirsRrcDynPrm >= CUPHY_CSIRS_MAX_NUM_PARAMS * PDSCH_MAX_CELLS_PER_CELL_GROUP))
            {
                NVLOGW_FMT(TAG , "Max CSI-RS params reached");
            }
            else
            {
                grp.nCsiRsPrms++;
                if(!params->cellInfo[params->nCells].nRrcParams)o
                    params->cellInfo[params->nCells].rrcParamsOffset = params->nCsirsRrcDynPrm;
                NVLOGD_FMT(TAG, "CSI-RS pCellParam[{}].rrcParamsOffset={}",params->nCells,params->nCsirsRrcDynPrm);
                params->cellInfo[params->nCells].nRrcParams++;
                params->csirsList[params->nCsirsRrcDynPrm++] = rrc_dyn_params;
                NVLOGD_FMT(TAG, "CSI-RS pCellParam[{}].nRrcParams={}",params->nCells,params->cellInfo[params->nCells].nRrcParams);
            }
        }
#endif
            tx_precoding_beamforming_t * pcBf = &csirs_params->pcAndBf[csirs_params->numPcBf];
            //csirs_params->static_bfwCoeff_mem_info[csirs_params->numPcBf] = static_bfwCoeff_mem_info;
            pcBf->num_prgs = msg.pc_and_bf.num_prgs;
            pcBf->prg_size = msg.pc_and_bf.prg_size;
            pcBf->dig_bf_interfaces = msg.pc_and_bf.dig_bf_interfaces;
            uint16_t bf_size = 0;
            bf_size = pcBf->num_prgs * (pcBf->dig_bf_interfaces+1);
            std::memcpy(&(pcBf->pm_idx_and_beam_idx[0]),msg.pc_and_bf.pm_idx_and_beam_idx,bf_size*sizeof(uint16_t));
            NVLOGD_FMT(TAG, "CSI-RS with cell group: pcAndBf[{}].num_prgs:{} prg_size:{} dig_bf_interfaces:{}",csirs_params->numPcBf,
                static_cast<unsigned short>(csirs_params->pcAndBf[csirs_params->numPcBf].num_prgs),static_cast<unsigned short>(csirs_params->pcAndBf[csirs_params->numPcBf].prg_size),
                static_cast<unsigned short>(csirs_params->pcAndBf[csirs_params->numPcBf].dig_bf_interfaces));
            csirs_params->numPcBf++;
        }
        return;
    }


    template <bool mplane_configured_ru_type, bool config_options_precoding_enabled, bool config_options_bf_enabled>
    void fh_callback(IFhCallbackContext& fh_context, nv::slot_detail_t* slot_detail)
    {
        //TI_GENERIC_INIT("fh_callback",15);
        //TI_GENERIC_ADD("remap");
        const bool is_csirs_cell = fh_context.is_csirs_cell();
        const int num_pdsch_params = fh_context.num_pdsch_fh_params();
        auto csirs_fh_params = fh_context.csirs_fh_params_view();

        bool csirs_compact_mode = false;
        if (unlikely(is_csirs_cell)) {
            update_fh_params_csirs_remap<mplane_configured_ru_type,config_options_precoding_enabled,config_options_bf_enabled>(
                fh_context, slot_detail, csirs_fh_params, csirs_compact_mode);
        }
        //TI_GENERIC_ADD("pdsch_with_csirs");
        //NVLOGC_FMT(TAG, "fh_callback: slot info: sfn={}, slot={}", fh_context.fh_slot_indication().sfn_, fh_context.fh_slot_indication().slot_);
        if (fh_context.has_csirs_params() && num_pdsch_params > 0) {
            uint32_t start_index = fh_context.start_index_pdsch_fh_params();
            uint32_t end_index = start_index + num_pdsch_params;
            const auto& pm_map = nv::PHY_module::pm_map();
            if (fh_context.csirs_symbol_map() == 0) {
                for (int i = start_index; i < end_index; ++i) {
                    update_prc_fh_params_pdsch<mplane_configured_ru_type>(pm_map, fh_context, fh_context.pdsch_fh_params_view(i), slot_detail);
                }
            } else {
                for (int i = start_index; i < end_index; ++i) {
                    update_prc_fh_params_pdsch_with_csirs<mplane_configured_ru_type>(pm_map, fh_context, fh_context.pdsch_fh_params_view(i), slot_detail, csirs_compact_mode);
                }
            }
        }
        //TI_GENERIC_ADD("non-overlapping csirs");
        if (unlikely(is_csirs_cell)) {
            if(get_comp_method(csirs_fh_params.cell_idx()) == comp_method::MODULATION_COMPRESSION) {
                update_non_overlapping_csirs<true>(csirs_fh_params.sym_prb_info());
            } else {
                update_non_overlapping_csirs<false>(csirs_fh_params.sym_prb_info());
            }
        }
        //TI_GENERIC_ADD("fh callback end");
        //TI_GENERIC_ALL_NVLOGW(TAG);
#ifdef DEBUG_SYM_PRB_INFO_STRUCT
        if (is_csirs_cell)
        {
            print_sym_prb_info(fh_context.slot_sfn(), fh_context.slot_slot(), fh_context.csirs_fh_params_view().sym_prb_info(), 0);
        }

        if(!is_csirs_cell)
        {
            slot_command_api::slot_info_* prev_cmd = nullptr;
            for(int i = 0; i < fh_context.total_num_pdsch_pdus(); ++i)
            {
                if(fh_context.pdsch_fh_params_view(i).sym_prb_info() != prev_cmd)
                {
                    print_sym_prb_info(fh_context.slot_sfn(), fh_context.slot_slot(), fh_context.pdsch_fh_params_view(i).sym_prb_info(), i);
                }
                prev_cmd = fh_context.pdsch_fh_params_view(i).sym_prb_info();
            }
        }
#endif
    }
    template void fh_callback<false,false,false>(IFhCallbackContext& fh_context, nv::slot_detail_t* slot_detail);
    template void fh_callback<false,false,true>(IFhCallbackContext& fh_context, nv::slot_detail_t* slot_detail);
    template void fh_callback<false,true,false>(IFhCallbackContext& fh_context, nv::slot_detail_t* slot_detail);
    template void fh_callback<false,true,true>(IFhCallbackContext& fh_context, nv::slot_detail_t* slot_detail);
    template void fh_callback<true,false,false>(IFhCallbackContext& fh_context, nv::slot_detail_t* slot_detail);
    template void fh_callback<true,false,true>(IFhCallbackContext& fh_context, nv::slot_detail_t* slot_detail);
    template void fh_callback<true,true,false>(IFhCallbackContext& fh_context, nv::slot_detail_t* slot_detail);
    template void fh_callback<true,true,true>(IFhCallbackContext& fh_context, nv::slot_detail_t* slot_detail);

}
