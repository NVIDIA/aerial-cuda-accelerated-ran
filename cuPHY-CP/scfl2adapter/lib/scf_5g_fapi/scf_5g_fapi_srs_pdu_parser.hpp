/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#if !defined(SCF_5G_FAPI_SRS_PDU_PARSER_HPP_INCLUDED_)
#define SCF_5G_FAPI_SRS_PDU_PARSER_HPP_INCLUDED_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iterator>
#include <ranges>

#include <gsl-lite/gsl-lite.hpp>

#include "aerial/casts/casts.hpp"
#include "scf_5g_fapi_parser_helpers.hpp"
#include "scf_5g_fapi_slot_concepts.hpp"
#include "scf_5g_fapi_srs_consts.hpp"
#include "scf_5g_fapi_tags.hpp"
#include "scf_5g_fapi.h"
#include "nv_phy_mac_transport.hpp"
#include "slot_command/slot_command.hpp"

namespace scf_5g_fapi
{

namespace detail {

// SRS FAPI lookup tables live in scf_5g_fapi_srs_consts.hpp so the parser,
// legacy scf_5g_slot_commands.cpp, and ru-emulator can converge on a single
// definition. See that header for spec citations.

#ifdef SCF_FAPI_10_04
// SCF FAPI 10.04 indication-policy constants kept local until the shared
// nv::* enum values are exported by aerial_common. See GT-12547.
inline constexpr uint8_t srs_data_ind_idx           = 4u;
inline constexpr uint8_t multi_msg_instance_per_slot = 2u;
#endif

[[nodiscard]] inline bool ipc_buffer_size(nv::phy_mac_transport& transport,
                                          nv_ipc_mempool_id_t pool,
                                          std::size_t& out) noexcept
{
    const int size = nv_ipc_get_buf_size(transport.get_nv_ipc_config(), pool);
    if (size <= 0)
    {
        NVLOGE_FMT(k_tag, AERIAL_L2ADAPTER_EVENT,
                   "SrsPduParser: failed to get NVIPC buffer size pool={} size={}",
                   static_cast<int>(pool), size);
        return false;
    }
    out = static_cast<std::size_t>(size);
    return true;
}

#ifdef SCF_FAPI_10_04_SRS
// trp_scheme and the v4 params section exist only in the 10.04 SRS wire format.
[[nodiscard]] inline const scs_fapi_v4_srs_params_t*
decode_v4_srs_params(const scf_fapi_srs_pdu_t& pdu) noexcept
{
    const uint8_t* next = &pdu.payload[0];
    // If trp_scheme == 0, beamforming PDU is present at payload[0] and v4 params follow it.
    // If trp_scheme != 0, beamforming PDU is absent and payload[0] is the v4 params directly;
    // the initial cast to scf_fapi_rx_beamforming_t only reads trp_scheme and is otherwise ignored.
    const auto* srs_rx_bf = aerial::casts::assume_cast<scf_fapi_rx_beamforming_t>(next);
    if (srs_rx_bf->trp_scheme == 0)
    {
        next += sizeof(scf_fapi_rx_beamforming_t)
                + (sizeof(uint16_t) * srs_rx_bf->num_prgs * srs_rx_bf->dig_bf_interfaces);
    }
    return aerial::casts::assume_cast<scs_fapi_v4_srs_params_t>(next);
}
#endif

[[nodiscard]] inline uint16_t calc_srs_start_prb(const scf_fapi_srs_pdu_t& pdu,
                                                 const slot_command_api::slot_indication& slot,
                                                 slot_command_api::srs_rb_info_t srs_rb_info[],
                                                 uint8_t mu,
                                                 uint8_t& n_hops) noexcept
{
    uint16_t hop_start_prbs[slot_command_api::MAX_SRS_SYM] = {0};
    uint16_t num_prbs[slot_command_api::MAX_SRS_SYM] = {0};
    uint16_t unique_hop_start_prbs[slot_command_api::MAX_SRS_SYM] = {0};
    uint16_t n_hops_in_slot = 0;
    uint16_t hop_idx = 0;
    uint16_t nb = 0;
    uint32_t n_srs = 0;
    const uint16_t n_syms = srs_symb_idx_to_num_symb[pdu.num_symbols];
    const uint16_t n_repetitions = srs_rep_factor_idx_to_num_rep_factor[pdu.num_repetitions];
    if (n_repetitions == 0u)
    {
        return 0u;
    }

    for (uint8_t i = 0; i < slot_command_api::MAX_SRS_SYM; ++i)
    {
        hop_start_prbs[i] = pdu.frequency_shift;
    }

    n_hops_in_slot = n_syms / n_repetitions;
    if (n_hops_in_slot == 0u)
    {
        return 0u;
    }
    for (hop_idx = 0; hop_idx < n_hops_in_slot; ++hop_idx)
    {
        for (uint8_t b = 0; b <= pdu.bandwidth_index; ++b)
        {
            const uint16_t nb_cfg = slot_command_api::srs_bw_table[pdu.config_index].bsrs_info[b].nb;
            const uint16_t m_srs_b = slot_command_api::srs_bw_table[pdu.config_index].bsrs_info[b].mSRS;
            if (nb_cfg == 0u || m_srs_b == 0u)
            {
                return 0u;
            }
            if (pdu.frequency_hopping >= pdu.bandwidth_index)
            {
                nb = ((4u * pdu.frequency_position / m_srs_b) % nb_cfg);
            }
            else if (b <= pdu.frequency_hopping)
            {
                nb = ((4u * pdu.frequency_position / m_srs_b) % nb_cfg);
            }
            else
            {
                if (pdu.resource_type == 0)
                {
                    n_srs = hop_idx;
                }
                else
                {
                    if (pdu.t_srs == 0u)
                    {
                        return 0u;
                    }
                    const uint32_t n_slots_per_frame = 10u << mu;
                    const uint32_t absolute_slot = n_slots_per_frame * slot.sfn_ + slot.slot_;
                    if (absolute_slot < pdu.t_offset)
                    {
                        return 0u;
                    }
                    const uint32_t slot_idx = absolute_slot - pdu.t_offset;
                    if ((slot_idx % pdu.t_srs) == 0u)
                    {
                        n_srs = (slot_idx / pdu.t_srs) * (n_syms / n_repetitions) + hop_idx;
                    }
                    else
                    {
                        NVLOGC_FMT(k_tag, "SrsPduParser: not an SRS slot");
                        n_srs = 0;
                        return 0;
                    }
                }

                uint16_t pi_bm1 = 1;
                for (uint8_t b_prime = pdu.frequency_hopping + 1u; b_prime <= b - 1u; ++b_prime)
                {
                    pi_bm1 = static_cast<uint16_t>(pi_bm1 * slot_command_api::srs_bw_table[pdu.config_index].bsrs_info[b_prime].nb);
                }
                const uint16_t pi_b = static_cast<uint16_t>(pi_bm1 * nb_cfg);
                uint16_t fb = 0;
                if ((nb_cfg % 2u) == 0u)
                {
                    fb = static_cast<uint16_t>((nb_cfg / 2u) * ((n_srs % pi_b) / pi_bm1)
                         + ((n_srs % pi_b) / (2u * pi_bm1)));
                }
                else
                {
                    fb = static_cast<uint16_t>((nb_cfg / 2u) * (n_srs / pi_bm1));
                }
                nb = static_cast<uint16_t>((fb + (4u * pdu.frequency_position / m_srs_b)) % nb_cfg);
            }
            hop_start_prbs[hop_idx] = static_cast<uint16_t>(hop_start_prbs[hop_idx] + m_srs_b * nb + pdu.bwp.bwp_start);
            num_prbs[hop_idx] = m_srs_b;
        }
    }

    hop_idx = 0;
    for (uint8_t symb_idx = 0; symb_idx < n_syms; symb_idx = static_cast<uint8_t>(symb_idx + n_repetitions))
    {
        for (uint8_t rep_idx = 0; rep_idx < n_repetitions; ++rep_idx)
        {
            srs_rb_info[symb_idx + rep_idx].srs_start_prbs = hop_start_prbs[hop_idx];
            srs_rb_info[symb_idx + rep_idx].num_srs_prbs = num_prbs[hop_idx];
        }
        ++hop_idx;
    }

    n_hops = 1;
    unique_hop_start_prbs[0] = hop_start_prbs[0];
    for (uint16_t hop_idx0 = 1; hop_idx0 < n_hops_in_slot; ++hop_idx0)
    {
        const auto* unique_begin = std::begin(unique_hop_start_prbs);
        const auto* unique_end   = unique_begin + n_hops;
        if (std::find(unique_begin, unique_end, hop_start_prbs[hop_idx0]) == unique_end)
        {
            unique_hop_start_prbs[n_hops] = hop_start_prbs[hop_idx0];
            ++n_hops;
        }
    }
    return n_syms;
}

// Outcome of try_alloc_additional_srs_ind. Each label names one failure
// mode so the caller's switch reads as a small table of policy + capacity
// invariants and how each one maps to logging / error-indication.
enum class SrsIndAllocOutcome : uint8_t
{
    Ok,
    PolicyDisallowed,    // view->indication_instances_per_slot != MULTI_MSG_INSTANCE_PER_SLOT
    CapacityExhausted,   // srs_ind_index_ reached MAX_SRS_IND_PER_SLOT - 1
    AllocFailed,         // transport.tx_alloc returned < 0
};

// Derived inputs passed to populate_ue_params. Bundling these into a
// struct keeps the helper's signature short and lets the call site
// document each field by name.
struct PopulateUeInputs
{
    uint8_t  n_ant_ports;
    uint16_t prg_size;
    uint16_t prg_size_l2;
    uint16_t num_prg;
    uint16_t first_hop_start_prb;
    uint16_t cell_prm_dyn_idx;
    uint16_t chest_buff_idx;
};

// Copies the FAPI SRS PDU fields into the per-UE slot-command record
// (cuphyUeSrsPrm_t) and applies the SCF_FAPI_10_04 chest-buffer index
// derivation. The SCF_FAPI_10_04_SRS v4 `usage` override is applied at
// the call site (it depends on a decoded v4 params block that only
// exists under that build flag).
inline void populate_ue_params(cuphyUeSrsPrm_t& ue,
                               const scf_fapi_srs_pdu_t& pdu,
                               const PopulateUeInputs& in) noexcept
{
    ue.cellIdx                = in.cell_prm_dyn_idx;
    ue.nAntPorts              = in.n_ant_ports;
    ue.nSyms                  = srs_symb_idx_to_num_symb[pdu.num_symbols];
    ue.nRepetitions           = srs_rep_factor_idx_to_num_rep_factor[pdu.num_repetitions];
    ue.combSize               = srs_comb_idx_to_comb_size[pdu.comb_size];
    ue.startSym               = pdu.time_start_position;
    ue.sequenceId             = pdu.sequenceId;
    ue.configIdx              = pdu.config_index;
    ue.bandwidthIdx           = pdu.bandwidth_index;
    ue.combOffset             = pdu.comb_offset;
    ue.cyclicShift            = pdu.cyclic_shift;
    ue.frequencyPosition      = pdu.frequency_position;
    ue.frequencyShift         = pdu.frequency_shift;
    ue.frequencyHopping       = pdu.frequency_hopping;
    ue.resourceType           = pdu.resource_type;
    ue.Tsrs                   = pdu.t_srs;
    ue.Toffset                = pdu.t_offset;
    ue.groupOrSequenceHopping = pdu.group_or_sequence_hopping;
    ue.chEstBuffIdx           = in.chest_buff_idx;
    ue.rnti                   = pdu.rnti;
    ue.handle                 = pdu.handle;
    ue.prgSize                = in.prg_size_l2;
    ue.usage                  = SRS_REPORT_FOR_BEAM_MANAGEMENT;
    ue.nValidPrg              = in.num_prg;
    ue.startValidPrg          = static_cast<uint16_t>(in.first_hop_start_prb / in.prg_size);
    ue.srsStartPrg            = 0;
#ifdef SCF_FAPI_10_04
    ue.srsChestBufferIndexL2 = static_cast<uint16_t>((pdu.handle >> 8) & 0xFFFF);
#else
    ue.srsChestBufferIndexL2 = static_cast<uint16_t>((pdu.rnti - 1) % slot_command_api::MAX_SRS_CHEST_BUFFERS_PER_4T4R_CELL);
#endif
}

// Expands the cell's [srsStartSym, srsStartSym + nSrsSym) window to also
// cover the [ue_start_sym, ue_start_sym + ue_n_syms) window. For a new
// cell, the window is set to the UE's exactly. Guard-clauses replace
// the nested-if shape so each case stands alone.
inline void merge_srs_sym_window(cuphySrsCellDynPrm_t& cell_info,
                                 uint8_t ue_start_sym,
                                 uint8_t ue_n_syms,
                                 bool is_new_cell) noexcept
{
    if (is_new_cell)
    {
        cell_info.srsStartSym = ue_start_sym;
        cell_info.nSrsSym = ue_n_syms;
        return;
    }
    if (cell_info.srsStartSym == ue_start_sym && cell_info.nSrsSym == ue_n_syms)
    {
        return;
    }
    if (ue_start_sym < cell_info.srsStartSym)
    {
        const uint8_t num_sym = static_cast<uint8_t>((cell_info.srsStartSym + cell_info.nSrsSym) - ue_start_sym);
        cell_info.nSrsSym = std::max(num_sym, ue_n_syms);
        cell_info.srsStartSym = ue_start_sym;
        return;
    }
    const uint8_t ue_num_sym = static_cast<uint8_t>((ue_start_sym + ue_n_syms) - cell_info.srsStartSym);
    cell_info.nSrsSym = std::max(cell_info.nSrsSym, ue_num_sym);
}

// Merges sorted (start, end-inclusive) RB intervals into `merged`, in place.
// Pre: `intervals` is sorted ascending by .first. Pure — no side effects.
// Two intervals [a, b] and [c, d] merge into [a, max(b, d)] iff c <= b + 1
// (touching or overlapping). Otherwise [c, d] becomes a new merged entry.
inline void merge_sorted_intervals(const std::vector<std::pair<uint16_t, uint16_t>>& intervals,
                                   std::vector<std::pair<uint16_t, uint16_t>>& merged) noexcept
{
    merged.clear();
    for (const auto& interval : intervals)
    {
        const auto [start_rb, end_rb] = interval;
        if (merged.empty() || start_rb > static_cast<uint16_t>(merged.back().second + 1u))
        {
            merged.push_back(interval);
        }
        else
        {
            merged.back().second = std::max(merged.back().second, end_rb);
        }
    }
}

// Appends merged (start, end-inclusive) PRB intervals to `sym_prbs` for `sym_idx`,
// tagged FH_DIR_UL and registered in the per-symbol SRS index list. Pre: caller has
// verified `sym_prbs.prbs_size + merged.size() <= MAX_PRB_INFO`.
inline void write_prb_entries(slot_command_api::slot_info_t& sym_prbs,
                              uint16_t sym_idx,
                              const std::vector<std::pair<uint16_t, uint16_t>>& merged) noexcept
{
    for (const auto& interval : merged)
    {
        const auto index = sym_prbs.prbs_size++;
        const auto start_rb = interval.first;
        const auto num_rb = static_cast<uint16_t>(interval.second - interval.first + 1u);
        sym_prbs.prbs[index] = slot_command_api::prb_info_t(start_rb, num_rb);
        sym_prbs.prbs[index].common.direction = slot_command_api::FH_DIR_UL;
        sym_prbs.symbols[sym_idx][slot_command_api::channel_type::SRS].push_back(index);
        NVLOGD_FMT(detail::k_tag,
                   "SrsPduParser: order PRB symbol={} start_rb={} num_rb={} index={}",
                   sym_idx, start_rb, num_rb, index);
    }
}

} // namespace detail

/**
 * Parses a single SRS PDU from a UL_TTI.request message.
 *
 * Populates the channel-pipeline SRS slot-command state for the parallel
 * UL aggregation path. The legacy update_cell_command(...SRS...) path is
 * intentionally left untouched.
 *
 * @tparam V  Type satisfying UlModuleView.
 */
template<UlModuleView V>
class SrsPduParser final
{
public:
    using pdu_t = scf_fapi_srs_pdu_t;
    static constexpr scf_fapi_ul_tti_pdu_type_t pdu_type = UL_TTI_PDU_TYPE_SRS;

    explicit SrsPduParser(V& v) noexcept : view_{&v} {}

    /**
     * Per-message setup for the current UL_TTI.request.
     *
     * SRS parsing needs the message cell id and request-level PUSCH presence
     * before individual SRS PDUs are dispatched.
     *
     * @param[in] req          UL_TTI.request for the current message.
     * @param[in] msg_cell_id  Logical cell id from the message envelope.
     * @param[in] payload_end  One-past-end of the validated message body for
     *                         bounded payload walks (10.02); nullptr = unbounded.
     * @param[in] expected_srs Store-time sidecar SRS count for this message; when
     *                         >= 0 it is used directly (avoids a 10.02 re-walk).
     *                         -1 (default) means "unknown" -> walk the payload.
     */
    void setup_cell(const scf_fapi_ul_tti_req_t& req, uint32_t msg_cell_id,
                    const void* payload_end = nullptr, int32_t expected_srs = -1) noexcept;

    /**
     * Process one SRS PDU for the given sfn/slot.
     *
     * @param[in] sfn   System Frame Number.
     * @param[in] slot  Slot number.
     * @param[in] pdu   SRS PDU to process.
     * @return true on success; false if processing could not be completed.
     *         Return value must be checked.
     */
    [[nodiscard]] bool parse(uint16_t sfn, uint16_t slot, const pdu_t& pdu) noexcept;

private:
    gsl_lite::not_null<V*> view_; //!< Non-owning; must outlive this parser.

    slot_command_api::slot_indication slot_ind_{};
    uint32_t logical_cell_id_{};
    uint32_t cell_index_{};
    /// Captured value of params.cell_grp_info.nCells AT THE MOMENT this cell's
    /// first SRS PDU enters ensure_srs_ind_buffer for the current slot.
    /// Used as the row index into params.srs_indications[]/num_srs_ind_indexes[]
    /// for all subsequent SRS PDUs from the same cell in this slot. Not "first
    /// SRS overall" — it's per-cell-per-slot. Reset to 0 in setup_cell() at slot
    /// boundary.
    uint32_t first_srs_cell_store_idx_{};
    int srs_ind_index_{};
    uint16_t srs_ues_at_ind_alloc_{};
    size_t nvipc_alloc_buff_len_{};
    bool first_srs_{};
    bool has_pusch_{};
    bool stop_srs_{};
    bool srs_enabled_{};
    ru_type ru_{OTHER_MODE};
    bool setup_valid_{};
    uint16_t expected_srs_pdus_{};
    uint16_t seen_srs_pdus_{};
    uint16_t accepted_srs_pdus_{};
    bool srs_order_prbs_finalized_{};
    bool publish_srs_order_prbs_{};

    // Helpers below are NOT marked noexcept: they call view methods
    // (send_fapi_error_indication, classify_srs_chest_buffer, transport) that
    // can throw. noexcept on these would bypass parse()'s try/catch and
    // terminate the process instead of letting parse() log and return.
    [[nodiscard]] bool validate_pdu(uint16_t sfn, uint16_t slot, const pdu_t& pdu);
    [[nodiscard]] bool ensure_srs_ind_buffer(slot_command_api::srs_params& params);
    [[nodiscard]] bool ensure_srs_ind_capacity(slot_command_api::srs_params& params,
                                               uint32_t,
                                               uint8_t n_ant_ports,
                                               uint16_t num_prg_l2,
                                               uint16_t n_rx_ant_srs);
#ifdef SCF_FAPI_10_04
    // Pure decision: returns the outcome and, on Ok, fills out_msg.
    // No logging / error-indication / state-mutation side effects —
    // caller (ensure_srs_ind_capacity) dispatches on the outcome.
    [[nodiscard]] detail::SrsIndAllocOutcome
    try_alloc_additional_srs_ind(nv::phy_mac_msg_desc& out_msg) noexcept;
#endif
    void record_srs_rb_info(slot_command_api::srs_params& params,
                            const slot_command_api::srs_rb_info_t srs_rb_info[],
                            uint16_t num_sym_info,
                            uint8_t start_sym) noexcept;
    void finalize_srs_order_prbs(slot_command_api::srs_params& params) noexcept;
    void finalize_srs_order_prbs_if_batch_complete() noexcept;
    void release_unaccepted_srs_ind_buffers(slot_command_api::srs_params& params) noexcept;
};

template<UlModuleView V>
void SrsPduParser<V>::setup_cell(const scf_fapi_ul_tti_req_t& req, uint32_t msg_cell_id,
                                 [[maybe_unused]] const void* payload_end,
                                 [[maybe_unused]] int32_t expected_srs) noexcept
{
    if (setup_valid_ && first_srs_ && slot_ind_.sfn_ == req.sfn
        && slot_ind_.slot_ == req.slot && logical_cell_id_ == msg_cell_id)
    {
#ifdef SCF_FAPI_10_04
        has_pusch_ = req.nPDUsOfEachType[UL_TTI_NPDUS_IDX_PUSCH] != 0u;
#else
        has_pusch_ = req.num_ulsch != 0u;
#endif
        srs_enabled_ = view_->srs_enabled();
        ru_ = view_->ru(cell_index_);
        return;
    }

    slot_ind_ = slot_command_api::slot_indication{req.sfn, req.slot, 0};
    logical_cell_id_ = msg_cell_id;
    const int32_t carrier_id = view_->carrier_id(msg_cell_id);
    if (carrier_id < 0)
    {
        setup_valid_ = false;
        stop_srs_ = true;
        return;
    }
    cell_index_ = static_cast<uint32_t>(carrier_id);
    first_srs_cell_store_idx_ = 0;
    srs_ind_index_ = 0;
    srs_ues_at_ind_alloc_ = 0;
    nvipc_alloc_buff_len_ = 0;
#ifdef SCF_FAPI_10_04
    expected_srs_pdus_ = req.nPDUsOfEachType[UL_TTI_NPDUS_IDX_SRS];
#else
    // 10.02 has no per-type wire counters. Prefer the store-time sidecar count
    // (single walk at ingress) when the caller supplies it; otherwise walk here.
    expected_srs_pdus_ = (expected_srs >= 0)
                             ? static_cast<uint16_t>(expected_srs)
                             : detail::count_ul_tti_pdus(req, UL_TTI_PDU_TYPE_SRS, payload_end);
#endif
    seen_srs_pdus_ = 0;
    accepted_srs_pdus_ = 0;
    srs_order_prbs_finalized_ = false;
    // The channel-task path publishes SRS order PRBs into the per-cell order scratch,
    // which merge_ul_order_scratch folds into the slot command sym_prb_info. The UL
    // order kernel reads those PRBs to launch SRS ordering, so finalize must always run.
    publish_srs_order_prbs_ = true;
    first_srs_ = false;
    stop_srs_ = false;
    setup_valid_ = true;
#ifdef SCF_FAPI_10_04
    has_pusch_ = req.nPDUsOfEachType[UL_TTI_NPDUS_IDX_PUSCH] != 0u;
#else
    has_pusch_ = req.num_ulsch != 0u;
#endif
    srs_enabled_ = view_->srs_enabled();
    ru_ = view_->ru(cell_index_);
}

template<UlModuleView V>
bool SrsPduParser<V>::validate_pdu(uint16_t sfn, uint16_t slot, const pdu_t& pdu)
{
    if (!setup_valid_)
    {
        return false;
    }

    if (pdu.num_ant_ports >= std::size(detail::srs_ant_idx_to_port)
        || pdu.num_symbols >= std::size(detail::srs_symb_idx_to_num_symb)
        || pdu.num_repetitions >= std::size(detail::srs_rep_factor_idx_to_num_rep_factor)
        || pdu.comb_size >= std::size(detail::srs_comb_idx_to_comb_size)
        || pdu.config_index >= std::size(slot_command_api::srs_bw_table)
        || pdu.bandwidth_index >= std::size(slot_command_api::srs_bw_table[0].bsrs_info))
    {
        NVLOGE_FMT(detail::k_tag, AERIAL_L2ADAPTER_EVENT,
                   "SrsPduParser: invalid SRS table index/period ant_ports={} num_symbols={} "
                   "num_repetitions={} comb_size={} config_index={} bandwidth_index={}",
                   pdu.num_ant_ports, pdu.num_symbols, pdu.num_repetitions, pdu.comb_size,
                   pdu.config_index, pdu.bandwidth_index);
        return false;
    }

    if (!srs_enabled_)
    {
        NVLOGI_FMT(detail::k_tag,
                   "SrsPduParser: cell_id={} received SRS while enable_srs=false; dropping PDU",
                   logical_cell_id_);
        return false;
    }

#if defined(SCF_FAPI_10_04_SRS) || !defined(SCF_FAPI_10_04)
    if (ru_ == SINGLE_SECT_MODE && !has_pusch_)
    {
        NVLOGI_FMT(detail::k_tag,
                   "SrsPduParser: SRS without PUSCH unsupported in SINGLE_SECT_MODE sfn={} slot={} cell_id={}",
                   sfn, slot, logical_cell_id_);
        view_->send_fapi_error_indication(logical_cell_id_, SCF_FAPI_UL_TTI_REQUEST,
                                          SCF_ERROR_CODE_SRS_WITHOUT_PUSCH_UNSUPPORTED,
                                          sfn, slot);
        stop_srs_ = true;
        return false;
    }
#elif defined(SCF_FAPI_10_04)
    if (ru_ == SINGLE_SECT_MODE)
    {
        return false;
    }
#endif

#ifdef ENABLE_L2_SLT_RSP
    auto& cell_l1_limit = view_->get_cell_limit_errors(static_cast<uint16_t>(cell_index_));
    if (cell_l1_limit.srs_errors.parsed < slot_command_api::MAX_SRS_PDU_PER_SLOT)
    {
        ++cell_l1_limit.srs_errors.parsed;
    }
    else
    {
        ++cell_l1_limit.srs_errors.errors;
        NVLOGD_FMT(detail::k_tag,
                   "SrsPduParser: SRS PDU L1 limit error sfn={} slot={} cell_index={}",
                   sfn, slot, cell_index_);
        view_->send_fapi_error_indication(logical_cell_id_, SCF_FAPI_UL_TTI_REQUEST,
                                          SCF_FAPI_SRS_L1_LIMIT_EXCEEDED,
                                          sfn, slot);
        stop_srs_ = true;
        return false;
    }
#endif

#ifdef SCF_FAPI_10_04
    switch (view_->classify_srs_chest_buffer(logical_cell_id_, pdu))
    {
    case SrsChestBuffVerdict::Accept:
        break;
    case SrsChestBuffVerdict::LookupFailed:
        NVLOGE_FMT(detail::k_tag, AERIAL_L2ADAPTER_EVENT,
                   "SrsPduParser: buffer state check failed cell_id={} rnti={}",
                   logical_cell_id_, pdu.rnti);
        return false;
    case SrsChestBuffVerdict::AlreadyRequested:
        NVLOGE_FMT(detail::k_tag, AERIAL_L2ADAPTER_EVENT,
                   "SrsPduParser: SRS chest buffer already requested cell_id={} rnti={}",
                   logical_cell_id_, pdu.rnti);
        view_->send_fapi_error_indication(logical_cell_id_, SCF_FAPI_UL_TTI_REQUEST,
                                          SCF_ERROR_CODE_SRS_CHEST_BUFF_BAD_STATE,
                                          sfn, slot);
        return false;
    }
#endif

#ifdef SCF_FAPI_10_04_SRS
    const auto* srs_v4_params = detail::decode_v4_srs_params(pdu);
    if (srs_v4_params->rep_scope != 0u)
    {
        NVLOGE_FMT(detail::k_tag, AERIAL_L2ADAPTER_EVENT,
                   "SrsPduParser: invalid reportScope={}; dropping PDU",
                   srs_v4_params->rep_scope);
        return false;
    }
    if (!(srs_v4_params->usage & (SRS_REPORT_FOR_BEAM_MANAGEMENT
                                  | SRS_REPORT_FOR_CODEBOOK
                                  | SRS_REPORT_FOR_NON_CODEBOOK)))
    {
        NVLOGE_FMT(detail::k_tag, AERIAL_L2ADAPTER_EVENT,
                   "SrsPduParser: unsupported SRS usage={}; dropping PDU",
                   static_cast<int>(srs_v4_params->usage));
        return false;
    }
#endif

    return true;
}

template<UlModuleView V>
bool SrsPduParser<V>::ensure_srs_ind_buffer(slot_command_api::srs_params& params)
{
    if (first_srs_)
    {
        return true;
    }

    auto& transport = view_->transport(static_cast<int>(cell_index_));
    nv::phy_mac_msg_desc msg_desc;
    msg_desc.data_pool = NV_IPC_MEMPOOL_CPU_LARGE;
    if (transport.tx_alloc(msg_desc) < 0)
    {
        NVLOGE_FMT(detail::k_tag, AERIAL_L2ADAPTER_EVENT,
                   "SrsPduParser: SRS.IND tx_alloc failed cell_index={}",
                   cell_index_);
        stop_srs_ = true;
        return false;
    }

    if (!detail::ipc_buffer_size(transport,
                                 static_cast<nv_ipc_mempool_id_t>(msg_desc.data_pool),
                                 nvipc_alloc_buff_len_))
    {
        NVLOGE_FMT(detail::k_tag, AERIAL_L2ADAPTER_EVENT,
                   "SrsPduParser::ensure_srs_ind_buffer: ipc_buffer_size failed for "
                   "cell_index={} pool={} - releasing buffer and stopping SRS for this slot",
                   cell_index_, static_cast<int>(msg_desc.data_pool));
        transport.tx_release(msg_desc);
        stop_srs_ = true;
        return false;
    }
    first_srs_cell_store_idx_ = params.cell_grp_info.nCells;
    srs_ues_at_ind_alloc_ = params.cell_grp_info.nSrsUes;
    params.num_srs_ind_indexes[first_srs_cell_store_idx_] = 0;
    srs_ind_index_ = params.num_srs_ind_indexes[first_srs_cell_store_idx_];
    params.srs_indications[first_srs_cell_store_idx_][srs_ind_index_] = static_cast<nv_ipc_msg_t>(msg_desc);
    for (int i = 0; i < slot_command_api::MAX_SRS_IND_PER_SLOT; ++i)
    {
        params.num_srs_pdus_per_srs_ind[first_srs_cell_store_idx_][i] = 0;
    }
    first_srs_ = true;
    return true;
}

template<UlModuleView V>
bool SrsPduParser<V>::ensure_srs_ind_capacity(slot_command_api::srs_params& params,
                                              uint32_t,
                                              uint8_t n_ant_ports,
                                              uint16_t num_prg_l2,
                                              uint16_t n_rx_ant_srs)
{
    const auto required = sizeof(scf_fapi_norm_ch_iq_matrix_info_t)
        + (static_cast<std::size_t>(n_ant_ports) * num_prg_l2 * n_rx_ant_srs
           * IQ_REPR_32BIT_NORMALIZED_IQ_SIZE_4);
    auto& desc = params.srs_indications[first_srs_cell_store_idx_][srs_ind_index_];
    if ((desc.data_len + required) <= nvipc_alloc_buff_len_)
    {
        return true;
    }

#ifdef SCF_FAPI_10_04
    nv::phy_mac_msg_desc new_msg;
    switch (try_alloc_additional_srs_ind(new_msg))
    {
    case detail::SrsIndAllocOutcome::Ok:
        ++srs_ind_index_;
        params.num_srs_ind_indexes[first_srs_cell_store_idx_] = srs_ind_index_;
        params.srs_indications[first_srs_cell_store_idx_][srs_ind_index_] = static_cast<nv_ipc_msg_t>(new_msg);
        return true;
    case detail::SrsIndAllocOutcome::PolicyDisallowed:
        NVLOGI_FMT(detail::k_tag,
                   "SrsPduParser: SRS PDUs overflow NVIPC buffer and multiple SRS.IND is disabled");
        view_->send_fapi_error_indication(logical_cell_id_, SCF_FAPI_UL_TTI_REQUEST,
                                          SCF_ERROR_CODE_PARTIAL_SRS_IND_ERR,
                                          slot_ind_.sfn_, slot_ind_.slot_);
        stop_srs_ = true;
        return false;
    case detail::SrsIndAllocOutcome::CapacityExhausted:
        NVLOGW_FMT(detail::k_tag,
                   "SrsPduParser: SRS PDUs overflow NVIPC buffer; all SRS.IND slots allocated={}",
                   srs_ind_index_);
        view_->send_fapi_error_indication(logical_cell_id_, SCF_FAPI_UL_TTI_REQUEST,
                                          SCF_ERROR_CODE_PARTIAL_SRS_IND_ERR,
                                          slot_ind_.sfn_, slot_ind_.slot_);
        stop_srs_ = true;
        return false;
    case detail::SrsIndAllocOutcome::AllocFailed:
        NVLOGE_FMT(detail::k_tag, AERIAL_L2ADAPTER_EVENT,
                   "SrsPduParser: multi-SRS.IND tx_alloc failed for cell_index={} "
                   "(srs_ind_index would advance to {}) - stopping SRS for this slot",
                   cell_index_, srs_ind_index_ + 1);
        stop_srs_ = true;
        return false;
    }
    return false;  // unreachable; all enumerators handled above
#else
    view_->send_fapi_error_indication(logical_cell_id_, SCF_FAPI_UL_TTI_REQUEST,
                                      SCF_ERROR_CODE_PARTIAL_SRS_IND_ERR,
                                      slot_ind_.sfn_, slot_ind_.slot_);
    stop_srs_ = true;
    return false;
#endif
}

#ifdef SCF_FAPI_10_04
template<UlModuleView V>
detail::SrsIndAllocOutcome
SrsPduParser<V>::try_alloc_additional_srs_ind(nv::phy_mac_msg_desc& out_msg) noexcept
{
    if (view_->indication_instances_per_slot(detail::srs_data_ind_idx) != detail::multi_msg_instance_per_slot)
    {
        return detail::SrsIndAllocOutcome::PolicyDisallowed;
    }
    if (srs_ind_index_ >= (slot_command_api::MAX_SRS_IND_PER_SLOT - 1))
    {
        return detail::SrsIndAllocOutcome::CapacityExhausted;
    }
    out_msg.data_pool = NV_IPC_MEMPOOL_CPU_LARGE;
    if (view_->transport(static_cast<int>(cell_index_)).tx_alloc(out_msg) < 0)
    {
        return detail::SrsIndAllocOutcome::AllocFailed;
    }
    return detail::SrsIndAllocOutcome::Ok;
}
#endif

template<UlModuleView V>
void SrsPduParser<V>::record_srs_rb_info(slot_command_api::srs_params& params,
                                         const slot_command_api::srs_rb_info_t srs_rb_info[],
                                         uint16_t num_sym_info,
                                         uint8_t start_sym) noexcept
{
    for (uint16_t sym_idx = 0; sym_idx < num_sym_info; ++sym_idx)
    {
        const auto start_rb = srs_rb_info[sym_idx].srs_start_prbs;
        const auto num_rb = srs_rb_info[sym_idx].num_srs_prbs;
        if (num_rb == 0u)
        {
            continue;
        }

        const auto symbol = static_cast<uint16_t>(start_sym + sym_idx);
        if (symbol >= OFDM_SYMBOLS_PER_SLOT)
        {
            continue;
        }

        params.rb_info_per_sym[cell_index_][symbol].push_back(
            {start_rb, static_cast<uint16_t>(start_rb + num_rb - 1u)});
    }
}

template<UlModuleView V>
void SrsPduParser<V>::finalize_srs_order_prbs(slot_command_api::srs_params& params) noexcept
{
    if (srs_order_prbs_finalized_)
    {
        return;
    }

    auto* sym_prbs = view_->order_sym_prb_info(cell_index_);
    if (sym_prbs == nullptr)
    {
        return;
    }

    auto& rb_info_per_sym = params.rb_info_per_sym[cell_index_];
    auto& final_rb_info_per_sym = params.final_rb_info_per_sym[cell_index_];

#ifdef SCF_FAPI_10_04_SRS
    const bool use_full_band_order_prbs = ru_ == SINGLE_SECT_MODE;
    const uint16_t ul_bandwidth_prbs = use_full_band_order_prbs
        ? view_->cell_view(cell_index_, slot_ind_).cell_params().nPrbUlBwp
        : 0u;
#else
    const bool use_full_band_order_prbs = false;
    const uint16_t ul_bandwidth_prbs = 0u;
#endif
    if (use_full_band_order_prbs && ul_bandwidth_prbs == 0u)
    {
        NVLOGE_FMT(detail::k_tag, AERIAL_L2ADAPTER_EVENT,
                   "SrsPduParser: UL bandwidth is zero while finalizing "
                   "single-section SRS order PRBs cell_index={}",
                   cell_index_);
        srs_order_prbs_finalized_ = true;
        return;
    }

    // Phase 1: per symbol, sort the recorded intervals and merge them into final_rb_info_per_sym.
    std::size_t merged_interval_count = 0;
    for (uint16_t sym_idx = 0; sym_idx < OFDM_SYMBOLS_PER_SLOT; ++sym_idx)
    {
        auto& intervals = rb_info_per_sym[sym_idx];
        if (intervals.empty())
        {
            continue;
        }

        auto& merged = final_rb_info_per_sym[sym_idx];
        if (use_full_band_order_prbs)
        {
            // Single-section packets cover the full UL bandwidth. The recorded
            // intervals identify which symbols carry SRS.
            merged.clear();
            merged.emplace_back(0u, static_cast<uint16_t>(ul_bandwidth_prbs - 1u));
        }
        else
        {
            std::ranges::sort(intervals);
            detail::merge_sorted_intervals(intervals, merged);
        }
        merged_interval_count += merged.size();
    }

    // Phase 2: single up-front capacity gate. No partial writes possible on overflow.
    const auto current_prb_count = static_cast<std::size_t>(sym_prbs->prbs_size);
    if (current_prb_count > MAX_PRB_INFO || merged_interval_count > (MAX_PRB_INFO - current_prb_count))
    {
        NVLOGE_FMT(detail::k_tag, AERIAL_L2ADAPTER_EVENT,
                   "SrsPduParser: sym_prb_info capacity exceeded while finalizing SRS order PRBs "
                   "current_size={} additional_entries={} capacity={}",
                   current_prb_count, merged_interval_count, MAX_PRB_INFO);
        srs_order_prbs_finalized_ = true;
        return;
    }

    // Phase 3: append the merged intervals to sym_prb_info.
    for (uint16_t sym_idx = 0; sym_idx < OFDM_SYMBOLS_PER_SLOT; ++sym_idx)
    {
        const auto& merged = final_rb_info_per_sym[sym_idx];
        if (merged.empty())
        {
            continue;
        }
        detail::write_prb_entries(*sym_prbs, sym_idx, merged);
    }

    srs_order_prbs_finalized_ = true;
}

template<UlModuleView V>
void SrsPduParser<V>::finalize_srs_order_prbs_if_batch_complete() noexcept
{
    if (!setup_valid_ || !publish_srs_order_prbs_ || srs_order_prbs_finalized_ || expected_srs_pdus_ == 0u
        || seen_srs_pdus_ < expected_srs_pdus_)
    {
        return;
    }

    auto* grp_cmd = view_->group_command();
    if (grp_cmd == nullptr)
    {
        return;
    }

    // Use raw unique_ptr access instead of get_srs_params() to avoid
    // create_if(SRS)'s side effect (lazy allocation + channel_array_size++)
    // on every RAII-fired finalize, including validation-drop paths where
    // the SRS params were never touched.
    auto* params = grp_cmd->srs.get();
    if (params == nullptr)
    {
        return;
    }

    finalize_srs_order_prbs(*params);
}

template<UlModuleView V>
void SrsPduParser<V>::release_unaccepted_srs_ind_buffers(slot_command_api::srs_params& params) noexcept
{
    if (!first_srs_ || params.cell_grp_info.nSrsUes != srs_ues_at_ind_alloc_)
    {
        return;
    }

    auto& transport = view_->transport(static_cast<int>(cell_index_));
    const auto used = static_cast<std::size_t>(params.num_srs_ind_indexes[first_srs_cell_store_idx_]) + 1u;
    for (const auto& ipc_msg : params.srs_indications[first_srs_cell_store_idx_] | std::views::take(used))
    {
        nv::phy_mac_msg_desc msg_desc(ipc_msg);
        transport.tx_release(msg_desc);
    }
    params.num_srs_ind_indexes[first_srs_cell_store_idx_] = 0;
    first_srs_ = false;
}

template<UlModuleView V>
bool SrsPduParser<V>::parse(uint16_t sfn, uint16_t slot, const pdu_t& pdu) noexcept
{
    ++seen_srs_pdus_;
    struct FinalizeSrsOrderPrbsOnExit
    {
        SrsPduParser& parser;
        ~FinalizeSrsOrderPrbsOnExit() noexcept
        {
            parser.finalize_srs_order_prbs_if_batch_complete();
        }
    } finalize_on_exit{*this};

    if (stop_srs_)
    {
        return true;
    }

    try
    {
        if (!validate_pdu(sfn, slot, pdu))
        {
            return true;
        }

        auto* grp_cmd = view_->group_command();
        if (grp_cmd == nullptr)
        {
            return false;
        }

        // get_srs_params() lazily allocates via create_if() and the underlying
        // unique_ptr is held by the cell_group_command (no external reset), so
        // the returned pointer is guaranteed non-null here.
        auto* params = grp_cmd->get_srs_params();

        const auto cell_view = view_->cell_view(cell_index_, slot_ind_);
        const auto& cell_params = cell_view.cell_params();

        slot_command_api::srs_rb_info_t srs_rb_info[slot_command_api::MAX_SRS_SYM] = {{0}};
        uint8_t n_hops = 0;
        const uint16_t num_sym_info = detail::calc_srs_start_prb(pdu, slot_ind_, srs_rb_info, cell_params.mu, n_hops);
        if (num_sym_info == 0u || n_hops == 0u)
        {
            return true;
        }

        uint16_t prg_size = slot_command_api::MIN_PRG_SIZE;
        uint16_t prg_size_l2 = slot_command_api::MIN_PRG_SIZE;
#ifdef SCF_FAPI_10_04_SRS
        const auto* srs_v4_params = detail::decode_v4_srs_params(pdu);
        prg_size_l2 = std::max<uint16_t>(prg_size_l2, srs_v4_params->prg_size);
        prg_size = ((prg_size_l2 > 4u) || (prg_size_l2 == 3u)) ? 2u : prg_size_l2;
#endif
        if (prg_size_l2 == 0u || prg_size == 0u)
        {
            NVLOGE_FMT(detail::k_tag, AERIAL_L2ADAPTER_EVENT,
                       "SrsPduParser: invalid PRG size prg_size_l2={} prg_size={}; dropping PDU",
                       prg_size_l2, prg_size);
            return true;
        }

        const uint16_t num_prg_l2 = static_cast<uint16_t>((n_hops * srs_rb_info[0].num_srs_prbs) / prg_size_l2);
        const uint16_t num_prg = static_cast<uint16_t>((n_hops * srs_rb_info[0].num_srs_prbs) / prg_size);
        if (srs_rb_info[0].num_srs_prbs == 0u || num_prg_l2 == 0u || num_prg == 0u)
        {
            NVLOGE_FMT(detail::k_tag, AERIAL_L2ADAPTER_EVENT,
                       "SrsPduParser: invalid SRS PRB allocation num_srs_prbs={} num_prg_l2={} num_prg={}; dropping PDU",
                       srs_rb_info[0].num_srs_prbs, num_prg_l2, num_prg);
            return true;
        }
        const uint8_t n_ant_ports = detail::srs_ant_idx_to_port[pdu.num_ant_ports];

        if (params->cell_grp_info.nSrsUes >= static_cast<uint16_t>(slot_command_api::MAX_SRS_UE_PER_TTI * slot_command_api::MAX_CELLS_PER_CELL_GROUP))
        {
            NVLOGE_FMT(detail::k_tag, AERIAL_L2ADAPTER_EVENT,
                       "SrsPduParser: nSrsUes={} exceeds capacity; dropping PDU",
                       params->cell_grp_info.nSrsUes);
            return false;
        }

        auto it = std::find(params->cell_index_list.begin(), params->cell_index_list.end(), static_cast<int32_t>(cell_index_));
        if (it == params->cell_index_list.end())
        {
            if (params->cell_grp_info.nCells >= slot_command_api::MAX_CELLS_PER_CELL_GROUP)
            {
                NVLOGE_FMT(detail::k_tag, AERIAL_L2ADAPTER_EVENT,
                           "SrsPduParser: nCells={} exceeds capacity; dropping PDU",
                           params->cell_grp_info.nCells);
                return false;
            }
        }

        if (!ensure_srs_ind_buffer(*params))
        {
            return false;
        }

        if (!ensure_srs_ind_capacity(*params, 0, n_ant_ports, num_prg_l2, cell_params.nRxAntSrs))
        {
            release_unaccepted_srs_ind_buffers(*params);
            return true;
        }

        bool new_cell = false;
        if (it == params->cell_index_list.end())
        {
            auto& cell_cmd = view_->cell_sub_command(cell_index_);
            cell_cmd.cell = view_->phy_cell_id(logical_cell_id_);
            params->cell_index_list.push_back(static_cast<int32_t>(cell_index_));
            params->phy_cell_index_list.push_back(cell_cmd.cell);
            ++params->cell_grp_info.nCells;
            new_cell = true;
        }

        const auto cell_store_idx = new_cell
            ? static_cast<uint32_t>(params->cell_grp_info.nCells - 1u)
            : static_cast<uint32_t>(it - params->cell_index_list.begin());

        auto& cell_cmd = view_->cell_sub_command(cell_index_);
        cell_cmd.cell = view_->phy_cell_id(logical_cell_id_);
        cell_cmd.slot.type = slot_command_api::SLOT_UPLINK;
        cell_cmd.slot.slot_3gpp = slot_ind_;
        grp_cmd->slot.type = slot_command_api::SLOT_UPLINK;
        grp_cmd->slot.slot_3gpp = slot_ind_;

        auto& cell_info = params->cell_dyn_info[cell_store_idx];
        if (new_cell)
        {
            cell_info.cellPrmDynIdx = static_cast<uint16_t>(cell_store_idx);
            cell_info.cellPrmStatIdx = view_->cell_stat_prm_idx(logical_cell_id_);
            cell_info.srsStartSym = 0;
            cell_info.nSrsSym = 0;
        }
        cell_info.slotNum = slot_ind_.slot_;
        cell_info.frameNum = slot_ind_.sfn_;

        auto& ue = params->ue_info[params->cell_grp_info.nSrsUes];
        detail::populate_ue_params(ue, pdu, detail::PopulateUeInputs{
            .n_ant_ports         = n_ant_ports,
            .prg_size            = prg_size,
            .prg_size_l2         = prg_size_l2,
            .num_prg             = num_prg,
            .first_hop_start_prb = srs_rb_info[0].srs_start_prbs,
            .cell_prm_dyn_idx    = cell_info.cellPrmDynIdx,
            .chest_buff_idx      = params->cell_grp_info.nSrsUes,
        });
#ifdef SCF_FAPI_10_04_SRS
        ue.usage = srs_v4_params->usage;
#endif

        params->dl_ul_bwp_max_prg[cell_index_] = detail::round_up_u16(cell_params.nPrbDlBwp, prg_size);
        params->nGnbAnt = cell_params.nRxAntSrs;
        detail::merge_srs_sym_window(cell_info, ue.startSym, ue.nSyms, new_cell);

        uint8_t ant_port_idx = 0;
        for ([[maybe_unused]] const uint8_t bit_pos : std::views::iota(uint8_t{0}, ue.nAntPorts))
        {
#ifdef SCF_FAPI_10_04_SRS
            if (srs_v4_params->samp_ue_ant & (1u << bit_pos))
#endif
            {
                ue.srsAntPortToUeAntMap[ant_port_idx] = ant_port_idx;
                ++ant_port_idx;
            }
        }

        auto& desc = params->srs_indications[first_srs_cell_store_idx_][srs_ind_index_];
        desc.data_len += sizeof(scf_fapi_norm_ch_iq_matrix_info_t);
        params->srs_chest_buffer[params->cell_grp_info.nSrsUes] =
            static_cast<uint8_t*>(desc.data_buf) + sizeof(uint8_t) * desc.data_len;
        desc.data_len += static_cast<uint32_t>(ue.nAntPorts * num_prg_l2 * cell_params.nRxAntSrs
                          * IQ_REPR_32BIT_NORMALIZED_IQ_SIZE_4);

        record_srs_rb_info(*params, srs_rb_info, num_sym_info, ue.startSym);
        ++params->cell_grp_info.nSrsUes;
        ++accepted_srs_pdus_;
        params->srs_ue_per_cell[cell_store_idx].cell_idx = static_cast<uint8_t>(cell_index_);
        ++params->srs_ue_per_cell[cell_store_idx].num_srs_ues;
        params->scf_ul_tti_handle_list.push_back(pdu.handle);
        ++params->num_srs_pdus_per_srs_ind[cell_store_idx][srs_ind_index_];
        return true;
    }
    catch (const std::exception& ex)
    {
        NVLOGE_FMT(detail::k_tag, AERIAL_L2ADAPTER_EVENT,
                   "SrsPduParser::parse sfn={} slot={} threw: {}",
                   sfn, slot, ex.what());
        return false;
    }
    catch (...)
    {
        NVLOGE_FMT(detail::k_tag, AERIAL_L2ADAPTER_EVENT,
                   "SrsPduParser::parse sfn={} slot={} threw unknown exception",
                   sfn, slot);
        return false;
    }
}

} // namespace scf_5g_fapi

#endif // SCF_5G_FAPI_SRS_PDU_PARSER_HPP_INCLUDED_
