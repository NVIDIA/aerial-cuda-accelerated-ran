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

/**
 * @file test_srs_pdu_parser.cpp
 * @brief Initial tests for SrsPduParser and ULSlotProcessor SRS dispatch.
 *
 * These tests intentionally focus on validation/drop paths that do not
 * allocate nvIPC buffers. Accepted-PDU descriptor accounting is covered by the
 * container build/integration matrix because nv::phy_mac_transport is not a
 * lightweight fakeable interface.
 */

#include <gtest/gtest.h>

#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include "aerial/casts/casts.hpp"
#include "fmtlog.h"
#include "scf_5g_fapi_srs_pdu_parser.hpp"
#include "scf_5g_fapi_ul_slot_processor.hpp"
#include "nv_phy_config_option.hpp"
#include "nv_phy_limit_errors.hpp"
#include "nv_phy_mac_transport.hpp"
#include "scf_5g_fapi.h"

namespace {

struct MockSrsCellView
{
    cuphyCellStatPrm_t stat_prm_{};

    MockSrsCellView()
    {
        stat_prm_.phyCellId = 42u;
        stat_prm_.nRxAntSrs = 4u;
        stat_prm_.nPrbDlBwp = 106u;
    }

    [[nodiscard]] uint16_t num_dl_prb() const noexcept { return stat_prm_.nPrbDlBwp; }
    [[nodiscard]] nv::slot_detail_t* slot_detail() const noexcept { return nullptr; }
    [[nodiscard]] const cuphyCellStatPrm_t& cell_params() const noexcept { return stat_prm_; }

    // Single-sector FH accessors — exercised by the PUSCH parser body that
    // ULSlotProcessor instantiates for every UL parser; never hit by SRS dispatch.
    [[nodiscard]] uint8_t ul_start_symbol() const noexcept { return 0u; }
    [[nodiscard]] uint8_t ul_max_symbols(uint8_t fallback) const noexcept { return fallback; }
};

static_assert(scf_5g_fapi::CellView<MockSrsCellView>,
              "MockSrsCellView must satisfy CellView");

struct MockUlModuleView
{
    bool srs_enabled_{false};
    bool chest_state_query_ok_{true};
    bool fapi_to_cplane_direct_enabled_{false};
    ru_type ru_{OTHER_MODE};
    slot_command_api::srsChestBuffState chest_state_{slot_command_api::SRS_CHEST_BUFF_NONE};
    mutable uint32_t srs_enabled_calls_{};
    mutable uint32_t error_indication_calls_{};
    mutable std::optional<scf_fapi_message_id_e> last_error_msg_id_{};
    mutable std::optional<scf_fapi_error_codes_t> last_error_code_{};
    mutable uint16_t last_error_sfn_{};
    mutable uint16_t last_error_slot_{};

    mutable slot_command_api::slot_command slot_cmd_{};
    mutable slot_command_api::cell_sub_command cell_sub_cmd_{};
    // Distinct from cell_sub_cmd_.sym_prb_info(): SRS publishes its Order PRB
    // metadata to the UL Order scratch, never to the cell command's FH buffer, so
    // the two destinations must be separate objects for a test to tell them apart.
    // Heap-allocated because slot_info_t is large (3 x MAX_PRB_INFO arrays).
    mutable std::unique_ptr<slot_command_api::slot_info_t> order_sym_prbs_{
        std::make_unique<slot_command_api::slot_info_t>()};
    mutable nv::phy_config_option config_opt_{};
    mutable nv::slot_limit_cell_error_t limit_errors_{};
    mutable nv::slot_limit_group_error_t group_limit_errors_{};
    mutable MockSrsCellView cell_view_{};
    mutable nv::pucch_dtx_t_list dtx_thresholds_{};
    mutable nv_ipc_config_t transport_config_{};
    mutable nv::phy_mac_transport transport_{transport_config_, 1u};

    [[nodiscard]] slot_command_api::cell_group_command* group_command() const noexcept
    {
        return &slot_cmd_.cell_groups;
    }

    [[nodiscard]] slot_command_api::cell_sub_command& cell_sub_command(uint32_t) const noexcept
    {
        return cell_sub_cmd_;
    }

    [[nodiscard]] slot_command_api::slot_info_t* order_sym_prb_info(uint32_t) const noexcept
    {
        return order_sym_prbs_.get();
    }

    [[nodiscard]] slot_command_api::slot_command& slot_command() const noexcept
    {
        return slot_cmd_;
    }

    [[nodiscard]] bool mmimo_enabled() const noexcept { return true; }
    [[nodiscard]] bool bf_enabled() const noexcept { return false; }
    [[nodiscard]] nv::phy_config_option& config_options() const noexcept { return config_opt_; }
    [[nodiscard]] int staticPuschSlotNum() const noexcept { return -1; }
    [[nodiscard]] uint8_t lbrm() const noexcept { return 0u; }
    [[nodiscard]] const nv::pucch_dtx_t_list& dtx_thresholds() const noexcept { return dtx_thresholds_; }
    [[nodiscard]] const float& dtx_thresholds_pusch() const noexcept
    {
        static const float threshold = 0.0F;
        return threshold;
    }

    [[nodiscard]] nv::phy_mac_transport& transport(int) const noexcept
    {
        return transport_;
    }

    [[nodiscard]] uint16_t cell_stat_prm_idx(uint32_t) const noexcept { return 7u; }
    [[nodiscard]] int32_t carrier_id(uint32_t) const noexcept { return 0; }
    [[nodiscard]] uint16_t phy_cell_id(uint32_t) const noexcept { return cell_view_.stat_prm_.phyCellId; }
    [[nodiscard]] nv::slot_limit_cell_error_t& get_cell_limit_errors(uint16_t) const noexcept
    {
        return limit_errors_;
    }
    // Required by the UlModuleView concept (extended by PUCCH MR1 to cover the
    // group-level L1-limit counters PUCCH validation needs). SRS does not exercise
    // this accessor; presence-only to satisfy the concept.
    [[nodiscard]] nv::slot_limit_group_error_t& get_group_limit_errors() const noexcept
    {
        return group_limit_errors_;
    }

    [[nodiscard]] uint8_t indication_instances_per_slot(uint32_t) const noexcept
    {
        return 0u;
    }

    [[nodiscard]] bool srs_enabled() const noexcept
    {
        ++srs_enabled_calls_;
        return srs_enabled_;
    }

    [[nodiscard]] ru_type ru(uint32_t) const noexcept { return ru_; }

    [[nodiscard]] scf_5g_fapi::SrsChestBuffVerdict
    classify_srs_chest_buffer(uint32_t, const scf_fapi_srs_pdu_t&) const noexcept
    {
        if (!chest_state_query_ok_)
        {
            return scf_5g_fapi::SrsChestBuffVerdict::LookupFailed;
        }
        if (chest_state_ == slot_command_api::SRS_CHEST_BUFF_REQUESTED)
        {
            return scf_5g_fapi::SrsChestBuffVerdict::AlreadyRequested;
        }
        return scf_5g_fapi::SrsChestBuffVerdict::Accept;
    }

    [[nodiscard]] bool fapi_to_cplane_direct_enabled() const noexcept
    {
        return fapi_to_cplane_direct_enabled_;
    }

    void send_fapi_error_indication(uint32_t,
                                    scf_fapi_message_id_e msg_id,
                                    scf_fapi_error_codes_t error_code,
                                    uint16_t sfn,
                                    uint16_t slot) const noexcept
    {
        ++error_indication_calls_;
        last_error_msg_id_ = msg_id;
        last_error_code_ = error_code;
        last_error_sfn_ = sfn;
        last_error_slot_ = slot;
    }

    [[nodiscard]] MockSrsCellView cell_view(uint32_t,
        const slot_command_api::slot_indication&) const noexcept
    {
        return cell_view_;
    }

    // PrachModuleView refinement stubs — required to satisfy the concept;
    // never invoked by the SRS test path.
    mutable nv::phy_config              phy_config_obj_{};
    mutable nv::prach_addln_config_t    prach_addln_config_obj_{};
    mutable nv::slot_limit_cell_error_t prach_limit_errors_{};

    [[nodiscard]] const nv::phy_config& phy_config(uint32_t) const noexcept
    {
        return phy_config_obj_;
    }
    [[nodiscard]] const nv::prach_addln_config_t& prach_addln_config(uint32_t) const noexcept
    {
        return prach_addln_config_obj_;
    }
    [[nodiscard]] ru_type ru_type_for_cell(uint32_t) const noexcept { return OTHER_MODE; }
    [[nodiscard]] bool is_fapi_to_cplane_direct() const noexcept { return false; }

    // PuschModuleView refinement stubs — required to satisfy the concept via
    // ULSlotProcessor's PuschPduParser; never invoked by the SRS test path.
    slot_command_api::bfw_coeff_mem_info_t* bfw_coeff_mem_info_{nullptr};

    [[nodiscard]] bool enable_weighted_avg_cfo() const noexcept { return false; }
    [[nodiscard]] slot_command_api::bfw_coeff_mem_info_t*
    bfw_coeff_mem_info(uint32_t, uint8_t) const noexcept { return bfw_coeff_mem_info_; }
};

static_assert(scf_5g_fapi::UlModuleView<MockUlModuleView>,
              "MockUlModuleView must satisfy UlModuleView");

void suppress_fmtlog_output() noexcept
{
    fmtlog::setLogLevel(fmtlog::OFF);
}

[[nodiscard]] scf_fapi_srs_pdu_t make_valid_srs_pdu() noexcept
{
    scf_fapi_srs_pdu_t pdu{};
    pdu.rnti = 0x1234u;
    pdu.handle = 0u;
    pdu.bwp.bwp_start = 0u;
    pdu.bwp.bwp_size = 106u;
    pdu.num_ant_ports = 1u;
    pdu.num_symbols = 0u;
    pdu.num_repetitions = 0u;
    pdu.time_start_position = 10u;
    pdu.config_index = 0u;
    pdu.sequenceId = 1u;
    pdu.bandwidth_index = 0u;
    pdu.comb_size = 0u;
    pdu.resource_type = 0u;
    pdu.t_srs = 1u;
    return pdu;
}

#ifdef SCF_FAPI_10_04_SRS
struct SrsPduWithPayload
{
    std::vector<uint8_t> bytes;

    [[nodiscard]] const scf_fapi_srs_pdu_t& pdu() const noexcept
    {
        return *aerial::casts::assume_cast<const scf_fapi_srs_pdu_t>(bytes.data());
    }
};

[[nodiscard]] SrsPduWithPayload make_srs_pdu_with_v4_params(uint8_t rep_scope,
                                                            uint8_t usage)
{
    SrsPduWithPayload result{};
    result.bytes.resize(sizeof(scf_fapi_srs_pdu_t)
                        + sizeof(scf_fapi_rx_beamforming_t)
                        + sizeof(scs_fapi_v4_srs_params_t),
                        0u);

    auto base = make_valid_srs_pdu();
    std::memcpy(result.bytes.data(), &base, sizeof(base));

    auto* pdu = aerial::casts::assume_cast<scf_fapi_srs_pdu_t>(result.bytes.data());
    auto* rx_bf = aerial::casts::assume_cast<scf_fapi_rx_beamforming_t>(&pdu->payload[0]);
    rx_bf->trp_scheme = 0u;
    rx_bf->num_prgs = 0u;
    rx_bf->dig_bf_interfaces = 0u;

    auto* v4 = aerial::casts::assume_cast<scs_fapi_v4_srs_params_t>(
        &pdu->payload[sizeof(scf_fapi_rx_beamforming_t)]);
    v4->rep_scope = rep_scope;
    v4->usage = usage;
    return result;
}
#endif

void set_ul_tti_pdu_counts(scf_fapi_ul_tti_req_t& req,
                           uint16_t srs_count,
                           uint16_t pusch_count) noexcept
{
#ifdef SCF_FAPI_10_04
    req.nPDUsOfEachType[UL_TTI_NPDUS_IDX_SRS]   = srs_count;
    req.nPDUsOfEachType[UL_TTI_NPDUS_IDX_PUSCH] = pusch_count;
#else
    static_cast<void>(srs_count);
    req.num_ulsch = static_cast<uint8_t>(pusch_count);
#endif
}

[[nodiscard]] scf_fapi_ul_tti_req_t make_ul_tti_req(uint16_t sfn,
                                                     uint16_t slot,
                                                     uint8_t num_pdus,
                                                     uint16_t srs_count,
                                                     uint16_t pusch_count = 0u) noexcept
{
    scf_fapi_ul_tti_req_t req{};
    req.sfn = sfn;
    req.slot = slot;
    req.num_pdus = num_pdus;
    set_ul_tti_pdu_counts(req, srs_count, pusch_count);
    return req;
}

void expect_no_srs_state_mutation(const MockUlModuleView& view)
{
    const auto* srs = view.slot_cmd_.cell_groups.srs.get();
    if (srs == nullptr)
    {
        EXPECT_EQ(view.slot_cmd_.cell_groups.channel_array_size, 0u);
        return;
    }
    ASSERT_NE(srs, nullptr);
    EXPECT_EQ(srs->cell_grp_info.nCells, 0u);
    EXPECT_EQ(srs->cell_grp_info.nSrsUes, 0u);
    EXPECT_TRUE(srs->cell_index_list.empty());
    EXPECT_TRUE(srs->phy_cell_index_list.empty());
    EXPECT_TRUE(srs->scf_ul_tti_handle_list.empty());
    EXPECT_EQ(view.slot_cmd_.cell_groups.channel_array_size, 0u);
    for (const auto& per_cell : srs->rb_info_per_sym)
    {
        for (const auto& per_sym : per_cell)
        {
            EXPECT_TRUE(per_sym.empty());
        }
    }
    for (const auto& per_cell : srs->final_rb_info_per_sym)
    {
        for (const auto& per_sym : per_cell)
        {
            EXPECT_TRUE(per_sym.empty());
        }
    }
}

#ifdef SCF_FAPI_10_04_SRS
void expect_srs_order_prbs_cover_band(const slot_command_api::slot_info_t& slot_info,
                                      uint16_t bandwidth,
                                      uint8_t expected_symbol)
{
    ASSERT_LT(expected_symbol, slot_info.symbols.size());
    const auto& expected_entries =
        slot_info.symbols[expected_symbol][slot_command_api::channel_type::SRS];
    ASSERT_EQ(expected_entries.size(), 1u);

    std::size_t count = 0u;
    for (const auto& per_symbol : slot_info.symbols)
    {
        for (const auto index : per_symbol[slot_command_api::channel_type::SRS])
        {
            ASSERT_LT(index, slot_info.prbs_size);
            const auto& prb = slot_info.prbs[index].common;
            EXPECT_EQ(prb.startPrbc, 0u);
            EXPECT_EQ(prb.numPrbc, bandwidth);
            ++count;
        }
    }
    EXPECT_EQ(count, 1u);
}
#endif

// Builds an SRS PDU whose payload is wired up for the SCF_FAPI_10_04_SRS
// accept path. Under that build flag, the parser calls
// detail::decode_v4_srs_params(pdu) which reads
// scf_fapi_rx_beamforming_t + scs_fapi_v4_srs_params_t off pdu.payload[];
// a bare make_valid_srs_pdu() does not allocate that block, so decoding
// reads garbage and the parser silently drops the PDU. Pre-MR5
// validation-drop tests don't reach the v4 decode and can keep using
// make_valid_srs_pdu(); MR5 accept-path tests (e.g. OrderPrbs_*) need
// this helper.
[[nodiscard]] const scf_fapi_srs_pdu_t& make_valid_srs_pdu_for_parse(
    [[maybe_unused]] scf_fapi_srs_pdu_t& pdu
#ifdef SCF_FAPI_10_04_SRS
    ,
    SrsPduWithPayload& pdu_with_payload
#endif
) noexcept
{
#ifdef SCF_FAPI_10_04_SRS
    pdu_with_payload = make_srs_pdu_with_v4_params(0u, SRS_REPORT_FOR_BEAM_MANAGEMENT);
    return pdu_with_payload.pdu();
#else
    pdu = make_valid_srs_pdu();
    return pdu;
#endif
}

[[nodiscard]] std::size_t srs_order_prb_entry_count(const slot_command_api::slot_info_t& slot_info)
{
    std::size_t count = 0;
    for (const auto& per_symbol : slot_info.symbols)
    {
        count += per_symbol[slot_command_api::channel_type::SRS].size();
    }
    return count;
}

struct FapiUlSrsMsg
{
    std::vector<uint8_t> buf;
    nv::phy_mac_msg_desc desc{};

    FapiUlSrsMsg(uint16_t sfn,
                 uint16_t slot,
                 const scf_fapi_srs_pdu_t& srs_pdu,
                 uint16_t wire_srs_count = 1u,
                 uint16_t pdu_size_override = 0u)
    {
        const auto hdr_sz     = sizeof(scf_fapi_header_t);
        const auto req_sz     = sizeof(scf_fapi_ul_tti_req_t);
        const auto gen_hdr_sz = sizeof(scf_fapi_generic_pdu_info_t);
        const auto total      = hdr_sz + req_sz + gen_hdr_sz + sizeof(scf_fapi_srs_pdu_t);

        buf.resize(total, 0u);

        auto* req = aerial::casts::assume_cast<scf_fapi_ul_tti_req_t>(buf.data() + hdr_sz);
        req->sfn = sfn;
        req->slot = slot;
        req->num_pdus = 1u;
        set_ul_tti_pdu_counts(*req, wire_srs_count, 0u);

        auto* gen = aerial::casts::assume_cast<scf_fapi_generic_pdu_info_t>(buf.data() + hdr_sz + req_sz);
        gen->pdu_type = static_cast<uint16_t>(UL_TTI_PDU_TYPE_SRS);
        gen->pdu_size = pdu_size_override != 0u
            ? pdu_size_override
            : static_cast<uint16_t>(gen_hdr_sz + sizeof(scf_fapi_srs_pdu_t));

        std::memcpy(buf.data() + hdr_sz + req_sz + gen_hdr_sz, &srs_pdu, sizeof(srs_pdu));

        desc.msg_buf = buf.data();
        desc.msg_len = static_cast<uint32_t>(buf.size());
        desc.cell_id = 0u;
    }
};

} // namespace

TEST(SrsPduParser, Validation_SrsDisabledDropsPdu)
{
    suppress_fmtlog_output();

    MockUlModuleView view{};
    scf_5g_fapi::SrsPduParser<MockUlModuleView> parser{view};

    const auto req = make_ul_tti_req(12u, 3u, 1u, 1u);

    parser.setup_cell(req, 0u);
    EXPECT_EQ(view.srs_enabled_calls_, 1u);

    EXPECT_TRUE(parser.parse(req.sfn, req.slot, make_valid_srs_pdu()));
    EXPECT_EQ(view.error_indication_calls_, 0u);

    expect_no_srs_state_mutation(view);
}

TEST(SrsPduParser, Validation_BadTableIndexDropsPdu)
{
    suppress_fmtlog_output();

    struct Case
    {
        const char* name;
        uint8_t num_ant_ports;
        uint8_t num_symbols;
        uint8_t num_repetitions;
        uint8_t comb_size;
    };

    constexpr Case cases[] = {
        {"num_ant_ports_oob", 3u, 0u, 0u, 0u},
        {"num_symbols_oob", 1u, 3u, 0u, 0u},
        {"num_repetitions_oob", 1u, 0u, 3u, 0u},
        {"comb_size_oob", 1u, 0u, 0u, 3u},
    };

    for (const auto& tc : cases)
    {
        SCOPED_TRACE(tc.name);

        MockUlModuleView view{};
        view.srs_enabled_ = true;
        scf_5g_fapi::SrsPduParser<MockUlModuleView> parser{view};

        const auto req = make_ul_tti_req(18u, 5u, 1u, 1u, 1u);
        parser.setup_cell(req, 0u);

        auto pdu = make_valid_srs_pdu();
        pdu.num_ant_ports = tc.num_ant_ports;
        pdu.num_symbols = tc.num_symbols;
        pdu.num_repetitions = tc.num_repetitions;
        pdu.comb_size = tc.comb_size;

        EXPECT_TRUE(parser.parse(req.sfn, req.slot, pdu));
        EXPECT_EQ(view.error_indication_calls_, 0u);

        expect_no_srs_state_mutation(view);
    }
}

TEST(SrsPduParser, Validation_SingleSectModePolicy)
{
    suppress_fmtlog_output();

    MockUlModuleView view{};
    view.srs_enabled_ = true;
    view.ru_ = SINGLE_SECT_MODE;
    scf_5g_fapi::SrsPduParser<MockUlModuleView> parser{view};

    const auto req = make_ul_tti_req(19u, 7u, 1u, 1u, 0u);
    parser.setup_cell(req, 0u);

    EXPECT_TRUE(parser.parse(req.sfn, req.slot, make_valid_srs_pdu()));
#ifdef SCF_FAPI_10_04_SRS
    ASSERT_EQ(view.error_indication_calls_, 1u);
    ASSERT_TRUE(view.last_error_msg_id_.has_value());
    ASSERT_TRUE(view.last_error_code_.has_value());
    EXPECT_EQ(*view.last_error_msg_id_, SCF_FAPI_UL_TTI_REQUEST);
    EXPECT_EQ(*view.last_error_code_, SCF_ERROR_CODE_SRS_WITHOUT_PUSCH_UNSUPPORTED);
    EXPECT_EQ(view.last_error_sfn_, req.sfn);
    EXPECT_EQ(view.last_error_slot_, req.slot);

    EXPECT_TRUE(parser.parse(req.sfn, req.slot, make_valid_srs_pdu()));
    EXPECT_EQ(view.error_indication_calls_, 1u);
#else
    EXPECT_EQ(view.error_indication_calls_, 0u);
#endif

    expect_no_srs_state_mutation(view);
}

TEST(SrsPduParser, Validation_SingleSectWithPuschPolicy)
{
    suppress_fmtlog_output();

    MockUlModuleView view{};
    view.srs_enabled_ = true;
    view.ru_ = SINGLE_SECT_MODE;
    scf_5g_fapi::SrsPduParser<MockUlModuleView> parser{view};

    const auto req = make_ul_tti_req(20u, 8u, 1u, 1u, 1u);
    parser.setup_cell(req, 0u, nullptr, 1);

#ifdef SCF_FAPI_10_04_SRS
    scf_fapi_srs_pdu_t pdu{};
    SrsPduWithPayload pdu_with_payload{};
    const auto& parsed_pdu = make_valid_srs_pdu_for_parse(pdu, pdu_with_payload);
    EXPECT_TRUE(parser.parse(req.sfn, req.slot, parsed_pdu));
    EXPECT_EQ(view.error_indication_calls_, 0u);
    ASSERT_NE(view.slot_cmd_.cell_groups.srs.get(), nullptr);
    const auto* order_prbs = view.order_sym_prb_info(0u);
    ASSERT_NE(order_prbs, nullptr);
    expect_srs_order_prbs_cover_band(
        *order_prbs, view.cell_view_.stat_prm_.nPrbUlBwp, parsed_pdu.time_start_position);
#else
    EXPECT_TRUE(parser.parse(req.sfn, req.slot, make_valid_srs_pdu()));
    EXPECT_EQ(view.error_indication_calls_, 0u);
#ifdef SCF_FAPI_10_04
    expect_no_srs_state_mutation(view);
#else
    // Rel-26-2 still parses legacy 10.02 single-section SRS with PUSCH.
    ASSERT_NE(view.slot_cmd_.cell_groups.srs.get(), nullptr);
    EXPECT_EQ(view.slot_cmd_.cell_groups.srs->cell_grp_info.nSrsUes, 1u);
#endif
#endif
}

#ifdef ENABLE_L2_SLT_RSP
TEST(SrsPduParser, Validation_L1LimitExceededDropsPdu)
{
    suppress_fmtlog_output();

    MockUlModuleView view{};
    view.srs_enabled_ = true;
    view.limit_errors_.srs_errors.parsed =
        static_cast<uint8_t>(slot_command_api::MAX_SRS_PDU_PER_SLOT);
    scf_5g_fapi::SrsPduParser<MockUlModuleView> parser{view};

    const auto req = make_ul_tti_req(20u, 9u, 1u, 1u, 1u);
    parser.setup_cell(req, 0u);

    EXPECT_TRUE(parser.parse(req.sfn, req.slot, make_valid_srs_pdu()));
    ASSERT_EQ(view.error_indication_calls_, 1u);
    ASSERT_TRUE(view.last_error_code_.has_value());
    EXPECT_EQ(*view.last_error_code_, SCF_FAPI_SRS_L1_LIMIT_EXCEEDED);
    EXPECT_EQ(view.limit_errors_.srs_errors.errors, 1u);

    EXPECT_TRUE(parser.parse(req.sfn, req.slot, make_valid_srs_pdu()));
    EXPECT_EQ(view.error_indication_calls_, 1u);

    expect_no_srs_state_mutation(view);
}
#endif

#ifdef SCF_FAPI_10_04
TEST(SrsPduParser, Validation_ChestBufferBadStateDropsPdu)
{
    suppress_fmtlog_output();

    MockUlModuleView view{};
    view.srs_enabled_ = true;
    view.chest_state_ = slot_command_api::SRS_CHEST_BUFF_REQUESTED;
    scf_5g_fapi::SrsPduParser<MockUlModuleView> parser{view};

    const auto req = make_ul_tti_req(21u, 11u, 1u, 1u, 1u);
    parser.setup_cell(req, 0u);

    EXPECT_TRUE(parser.parse(req.sfn, req.slot, make_valid_srs_pdu()));
    ASSERT_EQ(view.error_indication_calls_, 1u);
    ASSERT_TRUE(view.last_error_code_.has_value());
    EXPECT_EQ(*view.last_error_code_, SCF_ERROR_CODE_SRS_CHEST_BUFF_BAD_STATE);

    expect_no_srs_state_mutation(view);
}
#endif

#ifdef SCF_FAPI_10_04_SRS
TEST(SrsPduParser, Validation_UnsupportedRepScopeAndUsage)
{
    suppress_fmtlog_output();

    struct Case
    {
        const char* name;
        uint8_t rep_scope;
        uint8_t usage;
    };

    constexpr Case cases[] = {
        {"rep_scope_nonzero", 1u, SRS_REPORT_FOR_BEAM_MANAGEMENT},
        {"usage_zero", 0u, 0u},
        {"usage_only_unknown", 0u, 0x80u},
    };

    for (const auto& tc : cases)
    {
        SCOPED_TRACE(tc.name);

        MockUlModuleView view{};
        view.srs_enabled_ = true;
        scf_5g_fapi::SrsPduParser<MockUlModuleView> parser{view};

        const auto req = make_ul_tti_req(22u, 13u, 1u, 1u, 1u);
        parser.setup_cell(req, 0u);

        const auto pdu = make_srs_pdu_with_v4_params(tc.rep_scope, tc.usage);

        EXPECT_TRUE(parser.parse(req.sfn, req.slot, pdu.pdu()));
        EXPECT_EQ(view.error_indication_calls_, 0u);

        expect_no_srs_state_mutation(view);
    }
}
#endif

// SRS routes its Order PRB metadata to order_sym_prb_info() unconditionally: the
// UL Order kernel consumes it in both modes, and the per-channel scratch keeps
// parallel UL channel tasks from mutating the shared cell-command slot_info_t
// (merge_ul_order_scratch folds it in later).  The direct-C-plane flag therefore
// must not change the destination — these two tests pin that from both settings.
TEST(SrsPduParser, OrderPrbs_PublishedToOrderScratch_DirectCplaneDisabled)
{
    suppress_fmtlog_output();

    MockUlModuleView view{};
    view.srs_enabled_ = true;
    view.fapi_to_cplane_direct_enabled_ = false;
    scf_5g_fapi::SrsPduParser<MockUlModuleView> parser{view};

    const auto req = make_ul_tti_req(23u, 4u, 1u, 1u, 1u);
    parser.setup_cell(req, 0u);

    scf_fapi_srs_pdu_t pdu{};
#ifdef SCF_FAPI_10_04_SRS
    SrsPduWithPayload pdu_with_payload{};
    ASSERT_TRUE(parser.parse(req.sfn, req.slot, make_valid_srs_pdu_for_parse(pdu, pdu_with_payload)));
#else
    ASSERT_TRUE(parser.parse(req.sfn, req.slot, make_valid_srs_pdu_for_parse(pdu)));
#endif

    auto* srs = view.slot_cmd_.cell_groups.srs.get();
    ASSERT_NE(srs, nullptr);
    EXPECT_EQ(srs->cell_grp_info.nSrsUes, 1u);
    auto* sym_prbs = view.order_sym_prb_info(0u);
    ASSERT_NE(sym_prbs, nullptr);
    EXPECT_GT(sym_prbs->prbs_size, 0u);
    EXPECT_GT(srs_order_prb_entry_count(*sym_prbs), 0u);
    EXPECT_EQ(sym_prbs->prbs[0].common.direction, slot_command_api::FH_DIR_UL);
    EXPECT_EQ(sym_prbs->prbs[0].common.numPrbc,
              slot_command_api::srs_bw_table[0].bsrs_info[0].mSRS);

    // Order metadata never lands on the cell command's own FH buffer.
    auto* cell_fh = view.cell_sub_cmd_.sym_prb_info();
    ASSERT_NE(cell_fh, nullptr);
    EXPECT_EQ(cell_fh->prbs_size, 0u);
}

TEST(SrsPduParser, OrderPrbs_PublishedToOrderScratch_DirectCplaneEnabled)
{
    suppress_fmtlog_output();

    MockUlModuleView view{};
    view.srs_enabled_ = true;
    view.fapi_to_cplane_direct_enabled_ = true;
    scf_5g_fapi::SrsPduParser<MockUlModuleView> parser{view};

    const auto req = make_ul_tti_req(23u, 4u, 1u, 1u, 1u);
    parser.setup_cell(req, 0u);

    scf_fapi_srs_pdu_t pdu{};
#ifdef SCF_FAPI_10_04_SRS
    SrsPduWithPayload pdu_with_payload{};
    ASSERT_TRUE(parser.parse(req.sfn, req.slot, make_valid_srs_pdu_for_parse(pdu, pdu_with_payload)));
#else
    ASSERT_TRUE(parser.parse(req.sfn, req.slot, make_valid_srs_pdu_for_parse(pdu)));
#endif

    auto* srs = view.slot_cmd_.cell_groups.srs.get();
    ASSERT_NE(srs, nullptr);
    EXPECT_EQ(srs->cell_grp_info.nSrsUes, 1u);
    // Same destination as the direct-disabled case: the flag does not gate this.
    auto* sym_prbs = view.order_sym_prb_info(0u);
    ASSERT_NE(sym_prbs, nullptr);
    EXPECT_GT(sym_prbs->prbs_size, 0u);
    EXPECT_GT(srs_order_prb_entry_count(*sym_prbs), 0u);
    EXPECT_EQ(sym_prbs->prbs[0].common.direction, slot_command_api::FH_DIR_UL);

    // The cell command's FH buffer stays empty: in direct mode the framework builds
    // the O-RAN C-plane straight from FAPI.
    auto* cell_fh = view.cell_sub_cmd_.sym_prb_info();
    ASSERT_NE(cell_fh, nullptr);
    EXPECT_EQ(cell_fh->prbs_size, 0u);

    bool accumulated_rb_info = false;
    for (const auto& per_symbol : srs->rb_info_per_sym[0])
    {
        accumulated_rb_info = accumulated_rb_info || !per_symbol.empty();
    }
    EXPECT_TRUE(accumulated_rb_info);
}

TEST(MergeSortedIntervals, EmptyInputProducesEmptyOutput)
{
    const std::vector<std::pair<uint16_t, uint16_t>> intervals{};
    std::vector<std::pair<uint16_t, uint16_t>> merged{{99u, 100u}};  // pre-seeded; must be cleared
    scf_5g_fapi::detail::merge_sorted_intervals(intervals, merged);
    EXPECT_TRUE(merged.empty());
}

TEST(MergeSortedIntervals, DisjointIntervalsStayDistinct)
{
    const std::vector<std::pair<uint16_t, uint16_t>> intervals{
        {0u, 5u}, {10u, 15u}, {20u, 25u}};
    std::vector<std::pair<uint16_t, uint16_t>> merged{};
    scf_5g_fapi::detail::merge_sorted_intervals(intervals, merged);
    ASSERT_EQ(merged.size(), 3u);
    EXPECT_EQ(merged[0], (std::pair<uint16_t, uint16_t>{0u, 5u}));
    EXPECT_EQ(merged[1], (std::pair<uint16_t, uint16_t>{10u, 15u}));
    EXPECT_EQ(merged[2], (std::pair<uint16_t, uint16_t>{20u, 25u}));
}

TEST(MergeSortedIntervals, AdjacentIntervalsMerge)
{
    // (0,5) is touching (6,10) — end+1 == next.start; should merge to (0,10).
    const std::vector<std::pair<uint16_t, uint16_t>> intervals{
        {0u, 5u}, {6u, 10u}};
    std::vector<std::pair<uint16_t, uint16_t>> merged{};
    scf_5g_fapi::detail::merge_sorted_intervals(intervals, merged);
    ASSERT_EQ(merged.size(), 1u);
    EXPECT_EQ(merged[0], (std::pair<uint16_t, uint16_t>{0u, 10u}));
}

TEST(MergeSortedIntervals, OverlappingIntervalsMergeExtendingEnd)
{
    // (0,5) overlaps (3,8); should merge to (0,8). (10,12) stays disjoint.
    const std::vector<std::pair<uint16_t, uint16_t>> intervals{
        {0u, 5u}, {3u, 8u}, {10u, 12u}};
    std::vector<std::pair<uint16_t, uint16_t>> merged{};
    scf_5g_fapi::detail::merge_sorted_intervals(intervals, merged);
    ASSERT_EQ(merged.size(), 2u);
    EXPECT_EQ(merged[0], (std::pair<uint16_t, uint16_t>{0u, 8u}));
    EXPECT_EQ(merged[1], (std::pair<uint16_t, uint16_t>{10u, 12u}));
}

TEST(MergeSortedIntervals, ContainedIntervalDoesNotShrinkExisting)
{
    // (0,20) contains (5,10); merged end must stay at 20, not shrink to 10.
    const std::vector<std::pair<uint16_t, uint16_t>> intervals{
        {0u, 20u}, {5u, 10u}};
    std::vector<std::pair<uint16_t, uint16_t>> merged{};
    scf_5g_fapi::detail::merge_sorted_intervals(intervals, merged);
    ASSERT_EQ(merged.size(), 1u);
    EXPECT_EQ(merged[0], (std::pair<uint16_t, uint16_t>{0u, 20u}));
}

TEST(SrsPduParser, OrderPrbs_CapacityOverflowPreservesPrbsSize)
{
    suppress_fmtlog_output();

    MockUlModuleView view{};
    view.srs_enabled_ = true;
    view.fapi_to_cplane_direct_enabled_ = false;
    scf_5g_fapi::SrsPduParser<MockUlModuleView> parser{view};

    const auto req = make_ul_tti_req(24u, 5u, 1u, 1u, 1u);
    parser.setup_cell(req, 0u);

    // Pre-seed prbs_size to MAX_PRB_INFO so any additional entry would overflow.
    // The finalize step must early-return without writing partial entries.
    // Seeded on the Order scratch: that is where SRS publishes.
    auto* sym_prbs = view.order_sym_prb_info(0u);
    ASSERT_NE(sym_prbs, nullptr);
    sym_prbs->prbs_size = MAX_PRB_INFO;
    const auto pre_size = sym_prbs->prbs_size;

    ASSERT_TRUE(parser.parse(req.sfn, req.slot, make_valid_srs_pdu()));

    // prbs_size must be unchanged — overflow path skipped all writes.
    EXPECT_EQ(sym_prbs->prbs_size, pre_size);
    EXPECT_EQ(srs_order_prb_entry_count(*sym_prbs), 0u);
}

TEST(ULSlotProcessor, Process_SrsAbsentMessageSkipsParser)
{
    suppress_fmtlog_output();

    MockUlModuleView view{};
    scf_5g_fapi::ULSlotProcessor<MockUlModuleView> processor{view};
    FapiUlSrsMsg msg{44u, 6u, make_valid_srs_pdu(), 0u};
    auto* req = aerial::casts::assume_cast<scf_fapi_ul_tti_req_t>(
        msg.buf.data() + sizeof(scf_fapi_header_t));
    req->num_pdus = 0u;

    auto result = processor.process<UL_TTI_PDU_TYPE_SRS>(std::span{&msg.desc, 1u});

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(view.srs_enabled_calls_, 0u);

    expect_no_srs_state_mutation(view);
}

TEST(ULSlotProcessor, Process_SrsFilterSkipsNonSrsPdu)
{
    suppress_fmtlog_output();

    MockUlModuleView view{};
    view.srs_enabled_ = true;
    scf_5g_fapi::ULSlotProcessor<MockUlModuleView> processor{view};

    FapiUlSrsMsg msg{45u, 8u, make_valid_srs_pdu(), 0u};
    auto* req = aerial::casts::assume_cast<scf_fapi_ul_tti_req_t>(
        msg.buf.data() + sizeof(scf_fapi_header_t));
    req->num_pdus = 1u;
    set_ul_tti_pdu_counts(*req, 0u, 1u);
    auto* gen = aerial::casts::assume_cast<scf_fapi_generic_pdu_info_t>(
        msg.buf.data() + sizeof(scf_fapi_header_t) + sizeof(scf_fapi_ul_tti_req_t));
    gen->pdu_type = static_cast<uint16_t>(UL_TTI_PDU_TYPE_PUSCH);

    auto result = processor.process<UL_TTI_PDU_TYPE_SRS>(std::span{&msg.desc, 1u});

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(view.srs_enabled_calls_, 0u);
    EXPECT_EQ(view.error_indication_calls_, 0u);
    expect_no_srs_state_mutation(view);
}
