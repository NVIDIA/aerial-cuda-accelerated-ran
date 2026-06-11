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

#include "yamlparser.hpp"
#include "cuphydriver.hpp"
#include <algorithm>
#include <sstream>
#include <fstream>
#include "app_config.hpp"
#include "nvlog.hpp"
#include <arpa/inet.h>

#ifdef AERIAL_METRICS
#include "aerial_metrics.hpp"
#endif
#include "yaml_sdk_version.hpp"

#define TAG (NVLOG_TAG_BASE_CUPHY_CONTROLLER + 4) // "CTL.YAML"

static inline bool is_port_in_use(int port)
{
    // Create a socket to test if the port is available
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        NVLOGC_FMT(TAG, "Failed to create socket for port availability check");
        return false; // Assume port is available if we can't check
    }

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);

    // Convert host string to IP address
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);

    //Bind to the port
    if (bind(sock, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0) {
        bool is_port_in_use = (errno == EADDRINUSE);
        close(sock);
        return is_port_in_use; // Port is in use
    }

    close(sock);
    return false; // Port is available
}

static inline bool is_valid_backend_address_format(std::string_view backend_addr, std::string_view key_name)
{
    if (backend_addr.empty()) {
        NVLOGE_FMT(TAG, AERIAL_YAML_PARSER_EVENT, "Backend address is empty for key '{}'", key_name);
        return false;
    }

    // Check for host:port format
    size_t colon_pos = backend_addr.find(':');
    if (colon_pos == std::string::npos || colon_pos == 0 || colon_pos == backend_addr.length() - 1) {
        NVLOGE_FMT(TAG, AERIAL_YAML_PARSER_EVENT, "Invalid backend address format for key '{}': {} (missing or invalid host:port)", key_name, backend_addr);
        return false;
    }

    std::string_view host = backend_addr.substr(0, colon_pos);
    std::string_view port_str = backend_addr.substr(colon_pos + 1);

    // Validate host
    if (host.empty()) {
        NVLOGE_FMT(TAG, AERIAL_YAML_PARSER_EVENT, "Invalid backend address for key '{}': empty host in {}", key_name, backend_addr);
        return false;
    }

    // Host IP (loopback IP) will always be 127.0.0.1
    if (host != "127.0.0.1") {
        NVLOGE_FMT(TAG, AERIAL_YAML_PARSER_EVENT, "Invalid backend address for key '{}': host must be 127.0.0.1, got {} in {}", key_name, host, backend_addr);
        return false;
    }

    // Validate port number format and range (without checking availability)
    try {
        int port = std::stoi(std::string(port_str));
        if (port <= 0 || port > 65535) {
            NVLOGE_FMT(TAG, AERIAL_YAML_PARSER_EVENT, "Invalid backend address for key '{}': port {} out of range [1-65535] in {}", key_name, port, backend_addr);
            return false;
        }
    } catch (const std::invalid_argument&) {
        NVLOGE_FMT(TAG, AERIAL_YAML_PARSER_EVENT, "Invalid backend address for key '{}': non-numeric port {} in {}", key_name, port_str, backend_addr);
        return false;
    } catch (const std::out_of_range&) {
        NVLOGE_FMT(TAG, AERIAL_YAML_PARSER_EVENT, "Invalid backend address for key '{}': port {} too large in {}", key_name, port_str, backend_addr);
        return false;
    }

    return true;
}

static inline bool is_valid_backend_address(std::string_view backend_addr, std::string_view key_name)
{
    if (!is_valid_backend_address_format(backend_addr, key_name)) {
        return false;
    }

    // Extract port for availability check
    size_t colon_pos = backend_addr.find(':');
    std::string_view port_str = backend_addr.substr(colon_pos + 1);
    
    int port;
    try {
        port = std::stoi(std::string(port_str));
    } catch (const std::exception&) {
        NVLOGE_FMT(TAG, AERIAL_YAML_PARSER_EVENT, "Unexpected error parsing port {} for key '{}'", port_str, key_name);
        return false;
    }

    // Check if port is already in use by another process
    if (is_port_in_use(port)) {
        NVLOGE_FMT(TAG, AERIAL_YAML_PARSER_EVENT, "Backend address port is already in use by another process (possibly by another instance of cuphycontroller). Use a different port by changing YAML key: {}.", key_name);
        return false;
    }

    return true;
}

int YamlParser::parse_file(const char* filename)
{

    try
    {
        yaml::file_parser fp(filename);
        yaml::document doc = fp.next_document();
        yaml::node root = doc.root();

        aerial::check_yaml_version(root, filename);

        char l2adapter_config_full_path[MAX_PATH_LEN];
        get_full_path_file(l2adapter_config_full_path, CONFIG_YAML_FILE_PATH, static_cast<std::string>(root[YAML_PARAM_L2ADAPTER_FILENAME]).c_str(), CONFIG_CUBB_ROOT_DIR_RELATIVE_NUM);
        l2adapter_config_filename = std::string(l2adapter_config_full_path);

#ifdef AERIAL_METRICS
        std::string aerial_metrics_backend_addr = static_cast<std::string>(root[YAML_PARAM_AERIAL_METRICS_BACKEND_ADDRESS]);
        
        // Validate backend address before setting it
        if (!is_valid_backend_address(aerial_metrics_backend_addr, YAML_PARAM_AERIAL_METRICS_BACKEND_ADDRESS)) {
            return -1;
        }
        
        auto& metrics_manager = aerial_metrics::AerialMetricsRegistrationManager::getInstance();
        try
        {
            metrics_manager.changeBackendAddress(aerial_metrics_backend_addr);
        }
        catch (std::exception &e)
        {
            std::cout << e.what() << '\n';
            NVLOGE_FMT(TAG, AERIAL_CONFIG_EVENT, "Error setting Aerial metrics backend address to {}, please make sure the port is not occupied or use a different port.", aerial_metrics_backend_addr);
            return -1;
        }
#endif

        yaml::node pd_config_root = root[YAML_PARAM_CUPHYDRIVER_CONFIG];

        if(parse_cuphydriver_configs(pd_config_root) != 0)
        {
            NVLOGE_FMT(TAG, AERIAL_CONFIG_EVENT, "Failed to parse cuphydriver_config section in YAML file: {}", filename);
            return -1;
        };

        if(parse_cell_configs(pd_config_root) != 0)
        {
            NVLOGE_FMT(TAG, AERIAL_CONFIG_EVENT, "Failed to parse cells config section in YAML file: {}", filename);
            return -1;
        };
    }
    catch (std::exception& e)
    {
        std::cout << e.what() << '\n';
        return -1;
    }
    return 0;
}

int YamlParser::parse_standalone_config_file(const char* filename)
{
    try
    {
        yaml::file_parser fp(filename);
        yaml::document doc = fp.next_document();
        yaml::node root = doc.root();
        char standalone_full_path[MAX_PATH_LEN];
        get_full_path_file(standalone_full_path, CONFIG_YAML_FILE_PATH, static_cast<std::string>(root[YAML_PARAM_STANDALONE_FILENAME]).c_str(), CONFIG_CUBB_ROOT_DIR_RELATIVE_NUM);
        standalone_config_filename = std::string(standalone_full_path);
    }
    catch(const std::exception& e)
    {
        NVLOGC_FMT(TAG, "Standalone config file not found. {}", e.what());
        return -1;
    }
    return 0;
}

static void insert_unique_nic_info(std::string& nic_name,std::string *p_unique_nic_info,size_t length)
{
    bool insert = false;
    size_t i=0;
    for( ;i < length; i++)
    {
        if(p_unique_nic_info[i] == "")
        {
            insert = true;
            break;
        }
        else if(p_unique_nic_info[i] == nic_name)
        {
            break;
        }
    }
    if(insert)
    {
        p_unique_nic_info[i] = nic_name;
    }
}


static void parse_nic_info(yaml::node& yaml_cells,std::string *p_unique_nic_info)
{
    for(size_t i = 0; i < yaml_cells.length(); ++i)
    {
        yaml::node  cell = yaml_cells[i];
        std::string nic_name = static_cast<std::string>(cell[YAML_PARAM_CELL_NIC]);
        insert_unique_nic_info(nic_name,p_unique_nic_info,yaml_cells.length());
    }
}

static uint32_t get_nic_index(std::string& nic_name,std::string *p_unique_nic_info,size_t length)
{
    uint32_t i=0;
    for( ;i < length; i++)
    {
        if(p_unique_nic_info[i] == nic_name)
            break;
    }
    return i;
}

static bool isSupportedCompression(aerial_fh::UserDataCompressionMethod method) {
    switch (method) {
        case aerial_fh::UserDataCompressionMethod::NO_COMPRESSION:
        case aerial_fh::UserDataCompressionMethod::BLOCK_FLOATING_POINT:
        case aerial_fh::UserDataCompressionMethod::MODULATION_COMPRESSION:
            return true;
        default:
            return false;
    }
    return false;
}

static inline bool is_valid_mac_address(std::string_view mac_addr)
{
    // MAC address must be exactly 17 characters: XX:XX:XX:XX:XX:XX
    if (mac_addr.length() != 17) {
        return false;
    }

    // Check each position for expected format
    for (size_t i = 0; i < 17; ++i) {
        char c = mac_addr[i];
        
        if (i % 3 == 2) {
            // Position 2, 5, 8, 11, 14 should be colons
            if (c != ':') {
                return false;
            }
        } else {
            // Positions 0,1, 3,4, 6,7, 9,10, 12,13, 15,16 should be hex digits
            if (!std::isxdigit(static_cast<unsigned char>(c))) {
                return false;
            }
        }
    }
    
    return true;
}

/**
 * @brief Parses and loads cell configuration parameters from the provided YAML node.
 *
 * This function reads the cell configuration section from the YAML input, extracts relevant parameters,
 * and populates the corresponding member variables within the class. It is responsible for interpreting
 * the YAML structure for cell configurations and ensuring that all required parameters are stored for
 * later use.
 *
 * @param root The YAML node representing the root of the cell configuration section.
 * @return Returns 0 on successful parsing and population of configuration variables, or -1 if an error occurs.
 */
int YamlParser::parse_cell_configs(yaml::node root)
{
    yaml::node yaml_cells = root[YAML_PARAM_CELLS];
    std::string *unique_nic_info = new std::string[yaml_cells.length()];
    parse_nic_info(yaml_cells,unique_nic_info);
    for (size_t i = 0; i < phydriver_config.cell_group_num; ++i)
    {
        if (parse_single_cell(yaml_cells[i], unique_nic_info, phydriver_config.cell_group_num) != 0)
        {
            return -1;
        }
    }
    delete[] unique_nic_info;
    return 0;
}

static void parse_eth_addr(const std::string &eth_addr_str, std::array<uint8_t, 6> &eth_addr_bytes)
{
    char * pch = nullptr;
    int i=0, tmp=0;

    pch = strtok (const_cast<char*>(eth_addr_str.c_str()),":");
    while (pch != nullptr)
    {
        std::stringstream str;
        std::string stok;
        stok.assign(pch);
        str << stok;
        str >> std::hex >> tmp;
        eth_addr_bytes[i] = static_cast<uint8_t>(tmp);
        pch = strtok (nullptr,":");
        i++;
    }
}

/**
 * @brief Parses and populates a single cell's configuration parameters from a YAML node.
 *
 * This formatter reads the YAML node corresponding to a single cell, extracts all relevant configuration
 * parameters, validates their formats, and stores them in the appropriate member variables. It ensures
 * that each cell's configuration is correctly interpreted and integrated into the system's configuration
 * state.
 *
 * @param cell The YAML node containing configuration parameters for a single cell.
 * @param p_unique_nic_info Pointer to an array of unique NIC information strings.
 * @param length The number of unique NIC entries.
 * @return Returns 0 on successful parsing and population of the cell configuration, or -1 if an error occurs.
 */
int YamlParser::parse_single_cell(yaml::node cell,std::string *p_unique_nic_info,size_t length)
{
    struct cell_phy_info cell_cfg;
    cell_mplane_info mplane_cfg;
    int i=0;

    try
    {
        mplane_cfg.mplane_id = static_cast<uint16_t>(cell[YAML_PARAM_CELL_ID]);
        cell_cfg.mplane_id = mplane_cfg.mplane_id;
        cell_cfg.name = static_cast<std::string>(cell[YAML_PARAM_CELL_NAME]);

        if (cell_id_set.count(mplane_cfg.mplane_id))
        {
            NVLOGE_FMT(TAG, AERIAL_CONFIG_EVENT, 
                "Duplicate cell id '{}' detected (YAML key: {}).", mplane_cfg.mplane_id, YAML_PARAM_CELL_ID);
            return -1;
        }
        cell_id_set.insert(mplane_cfg.mplane_id);

        int ru = static_cast<int>(cell[YAML_PARAM_CELL_RU_TYPE]);
        if(ru > OTHER_MODE || ru < SINGLE_SECT_MODE)
        {
            NVLOGE_FMT(TAG, AERIAL_CONFIG_EVENT, "RU type not supported: {}. Check cell id {} configuration for YAML key: {}", ru, mplane_cfg.mplane_id, YAML_PARAM_CELL_RU_TYPE);
            return -1;
        }
        mplane_cfg.ru = static_cast<ru_type>(ru);


        mplane_cfg.nic_name = static_cast<std::string>(cell[YAML_PARAM_CELL_NIC]);
        //mplane_cfg.name = static_cast<std::string>(cell[YAML_PARAM_CELL_NAME]);
        
        // Validate that the NIC PCI address exists on the system
        std::string sys_path = "/sys/bus/pci/devices/" + mplane_cfg.nic_name;
        std::ifstream nic_check(sys_path + "/vendor");
        if (!nic_check.good()) {
            NVLOGE_FMT(TAG, AERIAL_CONFIG_EVENT, "PCI device {} not found for cell id {} (YAML key: {}). Please verify that the specified PCI address exists on this system.", mplane_cfg.nic_name, mplane_cfg.mplane_id, YAML_PARAM_CELL_NIC);
            return -1;
        }
        
        mplane_cfg.nic_index = get_nic_index(mplane_cfg.nic_name,p_unique_nic_info,length);
        NVLOGC_FMT(TAG,"cell_id {} nic_index :{}",mplane_cfg.mplane_id,mplane_cfg.nic_index);

        cell_cfg.phy_stat.mu = static_cast<uint8_t>(static_cast<uint16_t>(cell[YAML_PARAM_CELL_MU]));

        // Validate source MAC address format
        std::string src_eth_addr_str = static_cast<std::string>(cell[YAML_PARAM_CELL_SRC_MAC_ADDR]);
        if (!is_valid_mac_address(src_eth_addr_str)) {
            NVLOGE_FMT(TAG, AERIAL_CONFIG_EVENT, "Invalid source MAC address format found for cell id {} (YAML key: {})", mplane_cfg.mplane_id, YAML_PARAM_CELL_SRC_MAC_ADDR);
            return -1;
        }

        parse_eth_addr(src_eth_addr_str, mplane_cfg.src_eth_addr);

        // cell_cfg.vlan_tci = static_cast<uint16_t>(cell[YAML_PARAM_CELL_VLAN]);
        mplane_cfg.vlan_tci = (static_cast<uint16_t>(cell[YAML_PARAM_CELL_VLAN]) & 0xfff) |
                (static_cast<uint16_t>(cell[YAML_PARAM_CELL_PCP]) << 13);

        mplane_cfg.nic_cfg.txq_count_uplane = (uint8_t)static_cast<uint16_t>(cell[YAML_PARAM_CELL_TXQ_COUNT_UPLANE]);

        bool has_destinations = cell.has_key(YAML_PARAM_CELL_DESTINATIONS);
        bool has_legacy_dst = cell.has_key(YAML_PARAM_CELL_DST_MAC_ADDR);

        if (!has_destinations && !has_legacy_dst) {
            NVLOGE_FMT(TAG, AERIAL_CONFIG_EVENT,
                "Cell id {} must have either 'destinations' or 'dst_mac_addr'. "
                "Use 'destinations' for multi-destination routing, or 'dst_mac_addr' for single destination.",
                mplane_cfg.mplane_id);
            return -1;
        }

        if (has_destinations)
        {
            yaml::node dests_node = cell[YAML_PARAM_CELL_DESTINATIONS];
            for (size_t d = 0; d < dests_node.length(); ++d)
            {
                cell_mplane_destination dest_cfg{};
                if (!dests_node[d].has_key(YAML_PARAM_CELL_DST_MAC_ADDR)) {
                    NVLOGE_FMT(TAG, AERIAL_CONFIG_EVENT,
                        "Destination entry [{}] for cell id {} is missing required key 'dst_mac_addr'",
                        d, mplane_cfg.mplane_id);
                    return -1;
                }
                std::string dst_eth_addr_str = static_cast<std::string>(dests_node[d][YAML_PARAM_CELL_DST_MAC_ADDR]);
                if (!is_valid_mac_address(dst_eth_addr_str)) {
                    NVLOGE_FMT(TAG, AERIAL_CONFIG_EVENT, "Invalid destination MAC address format in destinations[{}] for cell id {} (YAML key: {})", d, mplane_cfg.mplane_id, YAML_PARAM_CELL_DST_MAC_ADDR);
                    return -1;
                }
                if (phydriver_config.ue_mode == 0 && cell_configs.size() < phydriver_config.cell_group_num)
                {
                    if (dst_mac_set.count(dst_eth_addr_str))
                    {
                        NVLOGE_FMT(TAG, AERIAL_CONFIG_EVENT, "Duplicate destination MAC address '{}' in destinations[{}] for cell id {}", dst_eth_addr_str, d, mplane_cfg.mplane_id);
                        return -1;
                    }
                    dst_mac_set.insert(dst_eth_addr_str);
                }
                parse_eth_addr(dst_eth_addr_str, dest_cfg.dst_eth_addr);

                parse_eAxC_to_beam_map(dests_node[d][YAML_PARAM_CELL_EAXC_ID_SSB_PBCH], std::vector<slot_command_api::channel_type>{slot_command_api::channel_type::PBCH}, dest_cfg);
                parse_eAxC_to_beam_map(dests_node[d][YAML_PARAM_CELL_EAXC_ID_PDCCH], std::vector<slot_command_api::channel_type>{slot_command_api::channel_type::PDCCH_DL, slot_command_api::channel_type::PDCCH_UL}, dest_cfg);
                parse_eAxC_to_beam_map(dests_node[d][YAML_PARAM_CELL_EAXC_ID_PDSCH], std::vector<slot_command_api::channel_type>{slot_command_api::channel_type::PDSCH, slot_command_api::channel_type::PDSCH_CSIRS}, dest_cfg);
                parse_eAxC_to_beam_map(dests_node[d][YAML_PARAM_CELL_EAXC_ID_CSIRS], std::vector<slot_command_api::channel_type>{slot_command_api::channel_type::CSI_RS}, dest_cfg);
                parse_eAxC_to_beam_map(dests_node[d][YAML_PARAM_CELL_EAXC_ID_PUSCH], std::vector<slot_command_api::channel_type>{slot_command_api::channel_type::PUSCH}, dest_cfg);
                parse_eAxC_to_beam_map(dests_node[d][YAML_PARAM_CELL_EAXC_ID_PUCCH], std::vector<slot_command_api::channel_type>{slot_command_api::channel_type::PUCCH}, dest_cfg);
                parse_eAxC_to_beam_map(dests_node[d][YAML_PARAM_CELL_EAXC_ID_SRS], std::vector<slot_command_api::channel_type>{slot_command_api::channel_type::SRS}, dest_cfg);
                parse_eAxC_to_beam_map(dests_node[d][YAML_PARAM_CELL_EAXC_ID_PRACH], std::vector<slot_command_api::channel_type>{slot_command_api::channel_type::PRACH}, dest_cfg);

                mplane_cfg.destinations.push_back(dest_cfg);

                for (size_t ch = 0; ch < slot_command_api::channel_type::CHANNEL_MAX; ++ch)
                {
                    for (auto eaxc : dest_cfg.eAxC_ids[ch])
                    {
                        if (std::find(mplane_cfg.eAxC_ids[ch].begin(), mplane_cfg.eAxC_ids[ch].end(), eaxc) == mplane_cfg.eAxC_ids[ch].end())
                        {
                            mplane_cfg.eAxC_ids[ch].push_back(eaxc);
                        }
                    }
                }
            }
            if (mplane_cfg.destinations.empty()) {
                NVLOGE_FMT(TAG, AERIAL_CONFIG_EVENT, "destinations list is empty for cell id {}", mplane_cfg.mplane_id);
                return -1;
            }
            mplane_cfg.dst_eth_addr = mplane_cfg.destinations[0].dst_eth_addr;
        }
        else
        {
            std::string dst_eth_addr_str = static_cast<std::string>(cell[YAML_PARAM_CELL_DST_MAC_ADDR]);
            if (!is_valid_mac_address(dst_eth_addr_str)) {
                NVLOGE_FMT(TAG, AERIAL_CONFIG_EVENT, "Invalid destination MAC address format found for cell id {} (YAML key: {})", mplane_cfg.mplane_id, YAML_PARAM_CELL_DST_MAC_ADDR);
                return -1;
            }
            if (phydriver_config.ue_mode == 0 && cell_configs.size() < phydriver_config.cell_group_num)
            {
                if (dst_mac_set.count(dst_eth_addr_str))
                {
                    NVLOGE_FMT(TAG, AERIAL_CONFIG_EVENT, "Duplicate destination MAC address '{}' detected for cell id {} (YAML key: {}). Each cell must have a unique destination MAC address.", dst_eth_addr_str, mplane_cfg.mplane_id, YAML_PARAM_CELL_DST_MAC_ADDR);
                    return -1;
                }
                dst_mac_set.insert(dst_eth_addr_str);
            }
            parse_eth_addr(dst_eth_addr_str, mplane_cfg.dst_eth_addr);

            parse_eAxC_to_beam_map(cell[YAML_PARAM_CELL_EAXC_ID_SSB_PBCH], std::vector<slot_command_api::channel_type>{slot_command_api::channel_type::PBCH}, mplane_cfg);
            parse_eAxC_to_beam_map(cell[YAML_PARAM_CELL_EAXC_ID_PDCCH], std::vector<slot_command_api::channel_type>{slot_command_api::channel_type::PDCCH_DL, slot_command_api::channel_type::PDCCH_UL}, mplane_cfg);
            parse_eAxC_to_beam_map(cell[YAML_PARAM_CELL_EAXC_ID_PDSCH], std::vector<slot_command_api::channel_type>{slot_command_api::channel_type::PDSCH, slot_command_api::channel_type::PDSCH_CSIRS}, mplane_cfg);
            parse_eAxC_to_beam_map(cell[YAML_PARAM_CELL_EAXC_ID_CSIRS], std::vector<slot_command_api::channel_type>{slot_command_api::channel_type::CSI_RS}, mplane_cfg);
            parse_eAxC_to_beam_map(cell[YAML_PARAM_CELL_EAXC_ID_PUSCH], std::vector<slot_command_api::channel_type>{slot_command_api::channel_type::PUSCH}, mplane_cfg);
            parse_eAxC_to_beam_map(cell[YAML_PARAM_CELL_EAXC_ID_PUCCH], std::vector<slot_command_api::channel_type>{slot_command_api::channel_type::PUCCH}, mplane_cfg);
            parse_eAxC_to_beam_map(cell[YAML_PARAM_CELL_EAXC_ID_SRS], std::vector<slot_command_api::channel_type>{slot_command_api::channel_type::SRS}, mplane_cfg);
            parse_eAxC_to_beam_map(cell[YAML_PARAM_CELL_EAXC_ID_PRACH], std::vector<slot_command_api::channel_type>{slot_command_api::channel_type::PRACH}, mplane_cfg);
        }

        char pusch_tv_full_path[MAX_PATH_LEN];
        char srs_tv_full_path[MAX_PATH_LEN];

        get_full_path_file(pusch_tv_full_path, CONFIG_TEST_VECTOR_PATH, static_cast<std::string>(cell[YAML_PARAM_CELL_TV_PUSCH]).c_str(), CONFIG_CUBB_ROOT_DIR_RELATIVE_NUM);
        get_full_path_file(srs_tv_full_path, CONFIG_TEST_VECTOR_PATH, static_cast<std::string>(cell[YAML_PARAM_CELL_TV_SRS]).c_str(), CONFIG_CUBB_ROOT_DIR_RELATIVE_NUM);

        mplane_cfg.dl_comp_meth = static_cast<aerial_fh::UserDataCompressionMethod>(static_cast<int>(cell[YAML_PARAM_CELL_DL_IQ_DATA_FMT][YAML_PARAM_CELL_COMP_METH]));
        if (!isSupportedCompression(mplane_cfg.dl_comp_meth))
        {
            NVLOGE_FMT(TAG, AERIAL_CONFIG_EVENT, "DL IQ data format is not supported {}, Check cell id {} configuration for YAML key: {}", (int)mplane_cfg.dl_comp_meth, mplane_cfg.mplane_id, YAML_PARAM_CELL_DL_IQ_DATA_FMT);
            return -1;
        }
        int dl_bit_width = static_cast<int>(cell[YAML_PARAM_CELL_DL_IQ_DATA_FMT][YAML_PARAM_CELL_BIT_WIDTH]);
        if(mplane_cfg.dl_comp_meth == aerial_fh::UserDataCompressionMethod::NO_COMPRESSION && dl_bit_width != 16)
        {
            NVLOGE_FMT(TAG, AERIAL_CONFIG_EVENT, "Fixed Point only support bit width 16 for now. Check cell id {} configuration for YAML key: {}", mplane_cfg.mplane_id, YAML_PARAM_CELL_DL_IQ_DATA_FMT);
            return -1;
        }
        if(mplane_cfg.dl_comp_meth == aerial_fh::UserDataCompressionMethod::BLOCK_FLOATING_POINT && (dl_bit_width < BFP_COMPRESSION_9_BITS || dl_bit_width > BFP_NO_COMPRESSION))
        {
            NVLOGE_FMT(TAG, AERIAL_CONFIG_EVENT, "BFP compression method does not support DL bit width {}. Check cell id {} configuration for YAML key: {}", dl_bit_width, mplane_cfg.mplane_id, YAML_PARAM_CELL_DL_IQ_DATA_FMT);
            return -1;
        }
        mplane_cfg.dl_bit_width = static_cast<uint8_t>(dl_bit_width);

        mplane_cfg.ul_comp_meth = static_cast<aerial_fh::UserDataCompressionMethod>(static_cast<int>(cell[YAML_PARAM_CELL_UL_IQ_DATA_FMT][YAML_PARAM_CELL_COMP_METH]));
        if (!isSupportedCompression(mplane_cfg.ul_comp_meth))
        {
            NVLOGE_FMT(TAG, AERIAL_CONFIG_EVENT, "UL IQ data format is not supported {}. Check cell id {} configuration for YAML key: {}", (int)mplane_cfg.ul_comp_meth, mplane_cfg.mplane_id, YAML_PARAM_CELL_UL_IQ_DATA_FMT);
            return -1;
        }

        int ul_bit_width = static_cast<int>(cell[YAML_PARAM_CELL_UL_IQ_DATA_FMT][YAML_PARAM_CELL_BIT_WIDTH]);
        if(mplane_cfg.ul_comp_meth == aerial_fh::UserDataCompressionMethod::NO_COMPRESSION && ul_bit_width != 16)
        {
            NVLOGE_FMT(TAG, AERIAL_CONFIG_EVENT, "Fixed Point only support bit width 16 for now. Check cell id {} configuration for YAML key: {}", mplane_cfg.mplane_id, YAML_PARAM_CELL_UL_IQ_DATA_FMT);
            return -1;
        }
        if(mplane_cfg.ul_comp_meth == aerial_fh::UserDataCompressionMethod::BLOCK_FLOATING_POINT && (ul_bit_width < BFP_COMPRESSION_9_BITS || ul_bit_width > BFP_NO_COMPRESSION))
        {
            NVLOGE_FMT(TAG, AERIAL_CONFIG_EVENT, "BFP compression method does not support UL bit width {}. Check cell id {} configuration for YAML key: {}", ul_bit_width, mplane_cfg.mplane_id, YAML_PARAM_CELL_UL_IQ_DATA_FMT);
            return -1;
        }
        mplane_cfg.ul_bit_width = static_cast<uint8_t>(ul_bit_width);

        mplane_cfg.fs_offset_dl = static_cast<int>(cell[YAML_PARAM_CELL_FS_OFFSET_DL]);
        mplane_cfg.exponent_dl = static_cast<int>(cell[YAML_PARAM_CELL_EXPONENT_DL]);
        mplane_cfg.ref_dl = static_cast<int>(cell[YAML_PARAM_CELL_REF_DL]);
        mplane_cfg.fs_offset_ul = static_cast<int>(cell[YAML_PARAM_CELL_FS_OFFSET_UL]);
        mplane_cfg.exponent_ul = static_cast<int>(cell[YAML_PARAM_CELL_EXPONENT_UL]);
        mplane_cfg.section_3_time_offset = htons(static_cast<uint16_t>(cell[YAML_PARAM_CELL_SECTION3_TIME_OFFSET]));
        mplane_cfg.max_amp_ul = static_cast<int>(cell[YAML_PARAM_CELL_MAX_AMP_UL]);
        mplane_cfg.t1a_max_up_ns = static_cast<uint32_t>(cell[YAML_PARAM_CELL_T1A_MAX_UP_NS]);
        mplane_cfg.t1a_max_cp_ul_ns = static_cast<uint32_t>(cell[YAML_PARAM_CELL_T1A_MAX_CP_UL_NS]);
        mplane_cfg.ta4_min_ns = static_cast<uint32_t>(cell[YAML_PARAM_CELL_TA4_MIN_NS]);
        mplane_cfg.ta4_max_ns = static_cast<uint32_t>(cell[YAML_PARAM_CELL_TA4_MAX_NS]);
        try{
            mplane_cfg.ta4_min_ns_srs = static_cast<uint32_t>(cell[YAML_PARAM_CELL_TA4_MIN_NS_SRS]);
            mplane_cfg.ta4_max_ns_srs = static_cast<uint32_t>(cell[YAML_PARAM_CELL_TA4_MAX_NS_SRS]);
        }
        catch (std::exception& e)
        {
            mplane_cfg.ta4_min_ns_srs =  621 * NS_X_US;
            mplane_cfg.ta4_max_ns_srs = 1831 * NS_X_US;
            NVLOGC_FMT(TAG, "ta4_min_ns_srs/ta4_max_ns_srs not set in config file, using default ta4_min_ns_srs = {} us and ta4_max_ns_srs = {} us", mplane_cfg.ta4_min_ns_srs / NS_X_US, mplane_cfg.ta4_max_ns_srs / NS_X_US);
        }

        try{
            mplane_cfg.t1a_min_cp_dl_ns = static_cast<uint32_t>(cell[YAML_PARAM_CELL_T1A_MIN_CP_DL_NS]);
            mplane_cfg.t1a_max_cp_dl_ns = static_cast<uint32_t>(cell[YAML_PARAM_CELL_T1A_MAX_CP_DL_NS]);
        }
        catch ([[maybe_unused]] const std::exception& e)
        {
            mplane_cfg.t1a_min_cp_dl_ns =  419 * NS_X_US;
            mplane_cfg.t1a_max_cp_dl_ns = 669 * NS_X_US;
            NVLOGC_FMT(TAG, "t1a_min_cp_dl_ns/t1a_max_cp_dl_ns not set in config file, using default t1a_min_cp_dl_ns = {} us and t1a_max_cp_dl_ns = {} us", mplane_cfg.t1a_min_cp_dl_ns / NS_X_US, mplane_cfg.t1a_max_cp_dl_ns / NS_X_US);
        }

        try
        {
            mplane_cfg.t1a_min_cp_ul_ns = static_cast<uint32_t>(cell[YAML_PARAM_CELL_T1A_MIN_CP_UL_NS]);
        }
        catch ([[maybe_unused]] const std::exception& e)
        {
            mplane_cfg.t1a_min_cp_ul_ns = 285 * NS_X_MS;
            NVLOGC_FMT(TAG, "t1a_min_cp_ul_ns not set in config file, using default of 285 us");
        }

        mplane_cfg.tcp_adv_dl_ns = static_cast<uint32_t>(cell[YAML_PARAM_CELL_TCP_ADV_DL_NS]);
        mplane_cfg.fh_len_range = static_cast<uint32_t>(cell[YAML_PARAM_CELL_MAX_FH_LEN]);
        mplane_cfg.pusch_prb_stride = static_cast<uint32_t>(cell[YAML_PARAM_CELL_PUSCH_PRB_STRIDE]);
        mplane_cfg.prach_prb_stride = static_cast<uint32_t>(cell[YAML_PARAM_CELL_PRACH_PRB_STRIDE]);
        mplane_cfg.srs_prb_stride = static_cast<uint32_t>(cell[YAML_PARAM_CELL_SRS_PRB_STRIDE]);
        mplane_cfg.pusch_ldpc_max_num_itr_algo_type = static_cast<uint32_t>(cell[YAML_PARAM_CELL_PUSCH_LDPC_MAX_NUM_ITR_ALGO_TYPE]);
        mplane_cfg.pusch_fixed_max_num_ldpc_itrs = static_cast<uint32_t>(cell[YAML_PARAM_CELL_PUSCH_FIXED_MAX_NUM_LDPC_ITRS]);
        mplane_cfg.pusch_ldpc_early_termination = static_cast<uint32_t>(cell[YAML_PARAM_CELL_PUSCH_LDPC_EARLY_TERMINATION]);
        mplane_cfg.pusch_ldpc_algo_index = static_cast<uint32_t>(cell[YAML_PARAM_CELL_PUSCH_LDPC_ALGO_INDEX]);
        mplane_cfg.pusch_ldpc_flags = static_cast<uint32_t>(cell[YAML_PARAM_CELL_PUSCH_LDPC_FLAGS]);
        mplane_cfg.pusch_ldpc_use_half = static_cast<uint32_t>(cell[YAML_PARAM_CELL_PUSCH_LDPC_USE_HALF]);
        mplane_cfg.ul_gain_calibration = static_cast<float>(cell[YAML_PARAM_CELL_UL_GAIN_CALIBRATION]);
        mplane_cfg.lower_guard_bw = static_cast<uint32_t>(cell[YAML_PARAM_CELL_LOWER_GUARD_BW]);

        try
        {
            mplane_cfg.pusch_nMaxPrb = static_cast<uint16_t>(cell[YAML_PARAM_CELL_PUSCH_NMAXPRB]);
        }
        catch (std::exception& e)
        {
            mplane_cfg.pusch_nMaxPrb = 273;
            NVLOGC_FMT(TAG, "pusch_nMaxPrb not set in config file, using default of 273 PRB allocation");
        }

        try
        {
            mplane_cfg.pusch_nMaxRx = static_cast<uint16_t>(cell[YAML_PARAM_CELL_PUSCH_NMAXRX]);
        }
        catch (std::exception& e)
        {
            mplane_cfg.pusch_nMaxRx = 0;
            NVLOGC_FMT(TAG, "pusch_nMaxRx not set in config file, using default value of 0");
        }

        try
        {
            mplane_cfg.ul_u_plane_tx_offset_ns = static_cast<uint32_t>(cell[YAML_PARAM_CELL_UL_U_PLANE_TX_OFFSET_NS]);
        }
        catch (std::exception& e)
        {
            mplane_cfg.ul_u_plane_tx_offset_ns = 280000;
            NVLOGC_FMT(TAG, "ul_u_plane_tx_offset_ns not set in config file, using default of 280 us");
        }        

        mplane_cfg.tv_pusch_h5 = std::string(pusch_tv_full_path);
        mplane_cfg.tv_srs_h5 = std::string(srs_tv_full_path);

        try
        {
            mplane_cfg.nMaxRxAnt = static_cast<uint32_t>(cell["nMaxRxAnt"]);
        }
        catch(const std::exception& e)
        {
            NVLOGI_FMT(TAG, " Using default value of {} for nMaxRxAnt", DEFAULT_MAX_4T4R_RXANT);
            mplane_cfg.nMaxRxAnt = DEFAULT_MAX_4T4R_RXANT;
        }

        try
        {
            mplane_cfg.dlc_core_index = static_cast<uint8_t>(static_cast<uint16_t>(cell[YAML_PARAM_DLC_CORE_INDEX]));
        }
        catch(const std::exception& e)
        {
            // Default to 0; actual value should be set via test_config.sh when scheme=1
            mplane_cfg.dlc_core_index = 0;
        }

        cell_configs.push_back(cell_cfg);
        mplane_configs.push_back(mplane_cfg);
    }
    catch (std::exception& e)
    {
        NVLOGE_FMT(TAG, AERIAL_CONFIG_EVENT, "{}", e.what());
        return -1;
    }
    return 0;
}

int YamlParser::parse_eAxC_to_beam_map(yaml::node node, const std::vector<slot_command_api::channel_type>& channels, cell_mplane_info& mplane_cfg)
{
    for(int i = 0; i < node.length(); ++i)
    {
        auto eAxC_id = static_cast<int>(node[i]);
        for (auto channel : channels)
        {
            mplane_cfg.eAxC_ids[channel].push_back(eAxC_id);
        }
    }

    return 0;
}

int YamlParser::parse_eAxC_to_beam_map(yaml::node node, const std::vector<slot_command_api::channel_type>& channels, cell_mplane_destination& dest_cfg)
{
    for(int i = 0; i < node.length(); ++i)
    {
        auto eAxC_id = static_cast<int>(node[i]);
        for (auto channel : channels)
        {
            dest_cfg.eAxC_ids[channel].push_back(eAxC_id);
        }
    }

    return 0;
}

/**
 * @brief Parses and loads cuPHY driver configuration parameters from the provided YAML node.
 *
 * This function reads the cuPHY driver configuration section from the YAML input, extracts relevant parameters,
 * and populates the corresponding member variables within the class. It is responsible for interpreting
 * the YAML structure for driver configurations and ensuring that all required parameters are stored for
 * later use. Default values are applied for missing keys where appropriate, and errors are logged if parsing fails.
 *
 * @param root The YAML node representing the root of the cuPHY driver configuration section.
 * @return Returns 0 on successful parsing and population of configuration variables, or -1 if an error occurs.
 */
int YamlParser::parse_cuphydriver_configs(yaml::node root)
{
    try
    {
        phydriver_config.validation                 = static_cast<uint16_t>(root[YAML_PARAM_VALIDATION]);
        phydriver_config.standalone                 = static_cast<uint16_t>(root[YAML_PARAM_STANDALONE]);
        phydriver_config.profiler_sec               = static_cast<int>(root[YAML_PARAM_PROFILER_SEC]);
        phydriver_config.num_slots                  = static_cast<uint16_t>(root[YAML_PARAM_NSLOTS]);
        phydriver_config.prometheus_thread          = static_cast<int>(root[YAML_PARAM_PROMETHEUS_THREAD]);
        phydriver_config.workers_sched_priority     = static_cast<uint32_t>(root[YAML_PARAM_WORKERS_SCHED_PRIORITY]);
        phydriver_config.start_section_id_srs       = static_cast<uint16_t>(root[YAML_PARAM_START_SECTION_ID_SRS]);
        phydriver_config.start_section_id_prach     = static_cast<uint16_t>(root[YAML_PARAM_START_SECTION_ID_PRACH]);
        phydriver_config.enable_ul_cuphy_graphs     = (uint8_t)static_cast<uint16_t>(root[YAML_PARAM_ENABLE_UL_CUPHY_GRAPHS]);
        phydriver_config.enable_dl_cuphy_graphs     = (uint8_t)static_cast<uint16_t>(root[YAML_PARAM_ENABLE_DL_CUPHY_GRAPHS]);
        phydriver_config.ul_order_timeout_cpu_ns    = static_cast<uint32_t>(root[YAML_PARAM_UL_ORDER_TIMEOUT_CPU_NS]);
        phydriver_config.ul_order_timeout_gpu_ns    = static_cast<uint32_t>(root[YAML_PARAM_UL_ORDER_TIMEOUT_GPU_NS]);
        phydriver_config.cplane_disable             = (uint8_t)static_cast<uint16_t>(root[YAML_PARAM_CPLANE_DISABLE]);
        phydriver_config.dpdk_thread                = static_cast<uint32_t>(root[YAML_PARAM_DPDK_THREAD]);
        phydriver_config.dpdk_verbose_logs          = (uint8_t)static_cast<uint16_t>(root[YAML_PARAM_DPDK_VERBOSE_LOGS]);
        phydriver_config.accu_tx_sched_res_ns       = static_cast<uint32_t>(root[YAML_PARAM_ACCU_TX_SCHED_RES_NS]);
        phydriver_config.accu_tx_sched_disable      = (uint8_t)static_cast<uint16_t>(root[YAML_PARAM_ACCU_TX_SCHED_DISABLE]);
        phydriver_config.fh_stats_dump_cpu_core     = static_cast<int>(root[YAML_PARAM_FH_STATS_DUMP_CPU_CORE]);
        phydriver_config.pdump_client_thread        = static_cast<int>(root[YAML_PARAM_PDUMP_CLIENT_THREAD]);
        phydriver_config.dpdk_file_prefix           = static_cast<std::string>(root[YAML_PARAM_DPDK_FILE_PREFIX]);
        phydriver_config.mps_sm_pusch               = static_cast<uint32_t>(root[YAML_PARAM_MPS_SM_PUSCH]);
        phydriver_config.mps_sm_pucch               = static_cast<uint32_t>(root[YAML_PARAM_MPS_SM_PUCCH]);
        phydriver_config.mps_sm_prach               = static_cast<uint32_t>(root[YAML_PARAM_MPS_SM_PRACH]);
        // mps_sm_ul_order assigned below with a fallback default value if the key is missing
        phydriver_config.mps_sm_srs                 = static_cast<uint32_t>(root[YAML_PARAM_MPS_SM_SRS]);
        phydriver_config.mps_sm_pdsch               = static_cast<uint32_t>(root[YAML_PARAM_MPS_SM_PDSCH]);
        phydriver_config.mps_sm_pdcch               = static_cast<uint32_t>(root[YAML_PARAM_MPS_SM_PDCCH]);
        phydriver_config.mps_sm_pbch                = static_cast<uint32_t>(root[YAML_PARAM_MPS_SM_PBCH]);
        // mps_sm_gpu_comms assigned below with a fallback default value if the key is missing
        phydriver_config.pdsch_fallback             = (uint8_t)static_cast<uint16_t>(root[YAML_PARAM_PDSCH_FACLLBACK]);
        phydriver_config.gpu_init_comms_dl          = (uint8_t)static_cast<uint16_t>(root[YAML_PARAM_GPU_INIT_COMMS_DL]);
        phydriver_config.cell_group_num             = (uint8_t)static_cast<uint16_t>(root[YAML_PARAM_CELL_GROUP_NUM]);
        phydriver_config.cell_group                 = (uint8_t)static_cast<uint16_t>(root[YAML_PARAM_CELL_GROUP]);

        if(root.has_key("ul_order_timeout_gpu_srs_ns"))
        {
            phydriver_config.ul_order_timeout_gpu_srs_ns = static_cast<uint32_t>(root[YAML_PARAM_UL_ORDER_TIMEOUT_GPU_SRS_NS]);
        }
        else
        {
            phydriver_config.ul_order_timeout_gpu_srs_ns = 5200000; // default value as the key is currently only present in a few config. files.
            NVLOGC_FMT(TAG, "cuphycontroller config. yaml does not have ul_order_timeout_gpu_srs_ns key; defaulting to {} ns", phydriver_config.ul_order_timeout_gpu_srs_ns);
        }

        if(root.has_key("ul_srs_aggr3_task_launch_offset_ns"))
        {
            phydriver_config.ul_srs_aggr3_task_launch_offset_ns = static_cast<uint32_t>(root[YAML_PARAM_UL_SRS_AGGR3_TASK_LAUNCH_OFFSET_NS]);
        }
        else
        {
            phydriver_config.ul_srs_aggr3_task_launch_offset_ns = 500000; // default value as the key is currently only present in a few config. files.
            NVLOGC_FMT(TAG, "cuphycontroller config. yaml does not have ul_srs_aggr3_task_launch_offset_ns key; defaulting to {} ns", phydriver_config.ul_srs_aggr3_task_launch_offset_ns);
        }

        if(root.has_key("gpu_init_comms_via_cpu"))
        {
            phydriver_config.gpu_init_comms_via_cpu = static_cast<uint8_t>(root[YAML_PARAM_GPU_INIT_COMMS_VIA_CPU]);
        }
        else
        {
            phydriver_config.gpu_init_comms_via_cpu = 0; // default value as the key is currently only present in a few config. files.
            NVLOGC_FMT(TAG, "cuphycontroller config. yaml does not have gpu_init_comms_via_cpu key; defaulting to {}.", phydriver_config.gpu_init_comms_via_cpu);
        }

        if(root.has_key("cpu_init_comms"))
        {
            phydriver_config.cpu_init_comms = static_cast<uint8_t>(root[YAML_PARAM_CPU_INIT_COMMS]);
        }
        else
        {
            phydriver_config.cpu_init_comms = 0; // default value as the key is currently only present in a few config. files.
            NVLOGC_FMT(TAG, "cuphycontroller config. yaml does not have cpu_init_comms key; defaulting to {}.", phydriver_config.cpu_init_comms);
        }

        if(root.has_key("use_green_contexts"))
        {
            phydriver_config.use_green_contexts = static_cast<uint8_t>(root[YAML_PARAM_USE_GREEN_CONTEXTS]);
        }
        else
        {
            phydriver_config.use_green_contexts = 0; // default value as the key is currently only present in a few config. files.
            NVLOGW_FMT(TAG, "cuphycontroller config. yaml does not have use_green_contexts key (experimental feature); defaulting to {}.", phydriver_config.use_green_contexts);
        }

        if(root.has_key("use_gc_workqueues"))
        {
            phydriver_config.use_gc_workqueues = static_cast<uint8_t>(root[YAML_PARAM_USE_GC_WORKQUEUES]);
            if((phydriver_config.use_green_contexts == 0) && (phydriver_config.use_gc_workqueues != 0))
            {
                NVLOGW_FMT(TAG, "use_gc_workqueues={} is set but use_green_contexts is 0; forcing use_gc_workqueues to 0.",
                           phydriver_config.use_gc_workqueues);
                phydriver_config.use_gc_workqueues = 0;
            }
        }
        else
        {
            phydriver_config.use_gc_workqueues = 0; // default value as the key is currently only present in a few config. files.
            NVLOGW_FMT(TAG, "cuphycontroller config. yaml does not have use_gc_workqueues key (experimental feature); defaulting to {}.", phydriver_config.use_gc_workqueues);
        }

        if(root.has_key("use_batched_memcpy"))
        {
            phydriver_config.use_batched_memcpy = static_cast<uint8_t>(root[YAML_PARAM_USE_BATCHED_MEMCPY]);
        }
        else
        {
            phydriver_config.use_batched_memcpy = 0; // default value as the key is currently only present in a few config. files.
            NVLOGW_FMT(TAG, "cuphycontroller config. yaml does not have use_batched_memcpy key; defaulting to {}.", phydriver_config.use_batched_memcpy);
        }

        if(root.has_key("mps_sm_ul_order"))
        {
            phydriver_config.mps_sm_ul_order = static_cast<uint32_t>(root[YAML_PARAM_MPS_SM_UL_ORDER]);
        }
        else
        {
            phydriver_config.mps_sm_ul_order = 16; // default value as the key is currently only present in the F08_R750 config. file
            NVLOGW_FMT(TAG, "cuphycontroller config. yaml does not have mps_sm_ul_order key; defaulting to 16.");
        }

        if(root.has_key("mps_sm_gpu_comms"))
        {
            phydriver_config.mps_sm_gpu_comms = static_cast<uint32_t>(root[YAML_PARAM_MPS_SM_GPU_COMMS]);
        }
        else
        {
            phydriver_config.mps_sm_gpu_comms = 8; // Set to 16 for F08_R750 config. file; 8 is the old value
            NVLOGW_FMT(TAG, "cuphycontroller config. yaml does not have mps_sm_gpu_comms key; defaulting to old count of 8 SMs.");
        }

        if(root.has_key("fix_beta_dl"))
        {
            phydriver_config.fix_beta_dl = static_cast<int>(root["fix_beta_dl"]);
        }
        else
        {
            phydriver_config.fix_beta_dl = 0;
        }

        if(root.has_key("ul_order_kernel_mode"))
        {
            phydriver_config.ul_order_kernel_mode = static_cast<uint8_t>(root[YAML_PARAM_UL_ORDER_KERNEL_MODE]);
        }
        else
        {
            phydriver_config.ul_order_kernel_mode = 0;
            NVLOGW_FMT(TAG, "cuphycontroller config. yaml does not have ul_order_kernel_mode key; defaulting to 0 (Ping-Pong mode).");
        }

        try
        {
            phydriver_config.ul_order_timeout_gpu_log_enable    = static_cast<uint8_t>(root[YAML_PARAM_UL_ORDER_TIMEOUT_GPU_LOG_ENABLE]);
        }
        catch(const std::exception& e)
        {
            NVLOGW_FMT(TAG, "{} Using default value of 0 to YAML_PARAM_UL_ORDER_TIMEOUT_GPU_LOG_ENABLE", e.what());
            phydriver_config.ul_order_timeout_gpu_log_enable = 0;
        }

        try
        {
            phydriver_config.ue_mode    = static_cast<uint8_t>(root[YAML_PARAM_UE_MODE]);
            if(phydriver_config.ue_mode)
            {
                // Parse DL Validation threads
                yaml::node workers_dl_validation = root[YAML_PARAM_DL_VALIDATION_WORKERS];
                for(size_t i = 0; i < workers_dl_validation.length(); ++i)
                {
                    phydriver_config.workers_dl_validation.push_back((uint8_t)static_cast<uint16_t>(workers_dl_validation[i]));
                    printf("Adding %d to DL validation worker because UE mode is enabled\n", phydriver_config.workers_dl_validation[i]);
                }
            }
        }
        catch(const std::exception& e)
        {
            NVLOGW_FMT(TAG, "{} Using default value of 0 to YAML_PARAM_UE_MODE", e.what());
            phydriver_config.ue_mode = 0;
        }

        try
        {
            phydriver_config.enable_l1_param_sanity_check = static_cast<uint8_t>(static_cast<uint16_t>(root[YAML_PARAM_ENABLE_L1_PARAM_SANITY_CHECK]));
        }
        catch(const std::exception& e)
        {
            NVLOGW_FMT(TAG, "{} Using default value of 0 to YAML_PARAM_ENABLE_L1_PARAM_SANITY_CHECK", e.what());
            phydriver_config.enable_l1_param_sanity_check = 0;
        }

        try
        {
            phydriver_config.enable_cpu_task_tracing = static_cast<uint8_t>(static_cast<uint16_t>(root[YAML_PARAM_ENABLE_CPU_TASK_TRACING]));
        }
        catch(const std::exception& e)
        {
            NVLOGW_FMT(TAG, "{} Using default value of 0 to ENABLE_CPU_TASK_TRACING", e.what());
            phydriver_config.enable_cpu_task_tracing = 0;
        }

        try
        {
            phydriver_config.enable_prepare_tracing = static_cast<uint8_t>(static_cast<uint16_t>(root[YAML_PARAM_ENABLE_PREPARE_TRACING]));
        }
        catch(const std::exception& e)
        {
            NVLOGW_FMT(TAG, "{} Using default value of 0 to ENABLE_PREPARE_TRACING", e.what());
            phydriver_config.enable_prepare_tracing = 0;
        }

        try
        {
            phydriver_config.cupti_enable_tracing = static_cast<uint8_t>(static_cast<uint16_t>(root[YAML_PARAM_CUPTI_ENABLE_TRACING]));
        }
        catch(const std::exception& e)
        {
            NVLOGW_FMT(TAG, "{} Using default value of 0 to CUPTI_ENABLE_TRACING", e.what());
            phydriver_config.cupti_enable_tracing = 0;
        }

        try
        {
            phydriver_config.cupti_buffer_size = static_cast<uint64_t>(root[YAML_PARAM_CUPTI_BUFFER_SIZE]);
        }
        catch(const std::exception& e)
        {
            NVLOGW_FMT(TAG, "{} Using default value of 2GB for cupti_buffer_size", e.what());
            phydriver_config.cupti_buffer_size = 2ULL * 1024 * 1024 * 1024;
        }

        try
        {
            phydriver_config.cupti_num_buffers = static_cast<uint16_t>(root[YAML_PARAM_CUPTI_NUM_BUFFERS]);
        }
        catch(const std::exception& e)
        {
            NVLOGW_FMT(TAG, "{} Using default value of 2 for cupti_num_buffers", e.what());
            phydriver_config.cupti_num_buffers = 2;
        }


        if(root.has_key(YAML_PARAM_CQE_TRACER_CONFIG))
        {
            yaml::node cqe_tracer_cfg = root[YAML_PARAM_CQE_TRACER_CONFIG];
            phydriver_config.enable_dl_cqe_tracing = cqe_tracer_cfg["enable_dl_cqe_tracing"].as<uint8_t>();
            phydriver_config.cqe_trace_cell_mask = cqe_tracer_cfg["cqe_trace_cell_mask"].as<uint64_t>();
            phydriver_config.cqe_trace_slot_mask = cqe_tracer_cfg["cqe_trace_slot_mask"].as<uint32_t>();            
        }
        else
        {
            NVLOGI_FMT(TAG," Using default value of 0 for CQE tracing enable,cell mask, slot mask");
            phydriver_config.enable_dl_cqe_tracing = 0;
            phydriver_config.cqe_trace_cell_mask = 0;
            phydriver_config.cqe_trace_slot_mask = 0;
        }

        if(root.has_key(YAML_PARAM_OK_TESTBENCH_CONFIG))
        {
            yaml::node ok_testbench_cfg = root[YAML_PARAM_OK_TESTBENCH_CONFIG];
            phydriver_config.enable_ok_tb = ok_testbench_cfg["enable_ok_tb"].as<uint8_t>();
            phydriver_config.num_ok_tb_slot = ok_testbench_cfg["num_ok_tb_slot"].as<uint32_t>();            
        }
        else
        {
            NVLOGI_FMT(TAG," Using default value of 0 for enable ok tb,num of OK tb slots");
            phydriver_config.enable_ok_tb = 0;
            phydriver_config.num_ok_tb_slot = 0;
        }        

        try
        {
            phydriver_config.disable_empw = static_cast<uint8_t>(static_cast<uint16_t>(root[YAML_PARAM_DISABLE_EMPW]));
        }
        catch(const std::exception& e)
        {
            NVLOGW_FMT(TAG, "{} Using default value of 0 to YAML_PARAM_DISABLE_EMPW", e.what());
            phydriver_config.disable_empw = 0;
        }
        
        try
        {
            phydriver_config.ul_rx_pkt_tracing_level = static_cast<uint8_t>(static_cast<uint16_t>(root[YAML_PARAM_UL_RX_PKT_TRACING_LEVEL]));
        }
        catch(const std::exception& e)
        {
            NVLOGW_FMT(TAG, "{} Using default value of 0 to YAML_PARAM_UL_RX_PKT_TRACING_LEVEL", e.what());
            phydriver_config.ul_rx_pkt_tracing_level = 0;
        }

        try
        {
            phydriver_config.ul_rx_pkt_tracing_level_srs = static_cast<uint8_t>(static_cast<uint16_t>(root[YAML_PARAM_UL_RX_PKT_TRACING_LEVEL_SRS]));
        }
        catch(const std::exception& e)
        {
            NVLOGW_FMT(TAG, "{} Using default value of 0 to YAML_PARAM_UL_RX_PKT_TRACING_LEVEL_SRS", e.what());
            phydriver_config.ul_rx_pkt_tracing_level_srs = 0;
        }

        try
        {
            phydriver_config.ul_warmup_frame_count = static_cast<uint32_t>(root[YAML_PARAM_UL_WARMUP_FRAME_COUNT]);
        }
        catch(const std::exception& e)
        {
            NVLOGW_FMT(TAG, "{} Using default value of 8 to YAML_PARAM_UL_WARMUP_FRAME_COUNT", e.what());
            phydriver_config.ul_warmup_frame_count = 8;
        }

        try
        {
            phydriver_config.pmu_metrics = static_cast<uint8_t>(static_cast<uint16_t>(root[YAML_PARAM_PMU_METRICS]));
        }
        catch(const std::exception& e)
        {
            NVLOGW_FMT(TAG, "{} Using default value of 0 to YAML_PARAM_PMU_METRICS", e.what());
            phydriver_config.pmu_metrics = 0;
        }

        try
        {
            phydriver_config.h2d_cpy_th_cfg.enable_h2d_copy_thread = static_cast<uint8_t>(static_cast<uint16_t>(root[YAML_PARAM_ENABLE_H2D_COPY_THREAD]));
        }
        catch(const std::exception& e)
        {
            NVLOGW_FMT(TAG, "{} Using default value of 0 to YAML_PARAM_ENABLE_H2D_COPY_THREAD", e.what());
            phydriver_config.h2d_cpy_th_cfg.enable_h2d_copy_thread = 0;
        }

        try
        {
            phydriver_config.h2d_cpy_th_cfg.h2d_copy_thread_cpu_affinity = static_cast<uint8_t>(static_cast<uint16_t>(root[YAML_PARAM_H2D_COPY_THREAD_CPU_AFFINITY]));
        }
        catch(const std::exception& e)
        {
            NVLOGW_FMT(TAG, "{} Using default value of 29 to YAML_PARAM_H2D_COPY_THREAD_CPU_AFFINITY", e.what());
            phydriver_config.h2d_cpy_th_cfg.h2d_copy_thread_cpu_affinity = 29;
        }

        try
        {
            phydriver_config.h2d_cpy_th_cfg.h2d_copy_thread_sched_priority = static_cast<uint8_t>(static_cast<uint16_t>(root[YAML_PARAM_H2D_COPY_THREAD_SCHED_PRIORITY]));
        }
        catch(const std::exception& e)
        {
            NVLOGW_FMT(TAG, "{} Using default value of 0 to YAML_PARAM_H2D_COPY_THREAD_SCHED_PRIORITY", e.what());
            phydriver_config.h2d_cpy_th_cfg.h2d_copy_thread_sched_priority = 0;
        }

        try
        {
            phydriver_config.mMIMO_enable = static_cast<uint8_t>(static_cast<uint16_t>(root[YAML_PARAM_MMIMO_ENABLE]));
        }
        catch(const std::exception& e)
        {
            NVLOGW_FMT(TAG, "{} Using default value of 0 to YAML_PARAM_MMIMO_ENABLE", e.what());
            phydriver_config.mMIMO_enable = 0;
        }

        try
        {
            phydriver_config.enable_srs = static_cast<uint8_t>(static_cast<uint16_t>(root[YAML_PARAM_ENABLE_SRS]));
        }
        catch(const std::exception& e)
        {
            NVLOGW_FMT(TAG, "{} Using default value of 0 to YAML_PARAM_ENABLE_SRS", e.what());
            phydriver_config.enable_srs = 0;
        }

        try
        {
            phydriver_config.enable_dl_core_affinity = static_cast<uint8_t>(static_cast<uint16_t>(root[YAML_PARAM_ENABLE_DL_CORE_AFFINITY]));
        }
        catch(const std::exception& e)
        {
            NVLOGW_FMT(TAG, "{} Using default value of 1 to YAML_PARAM_ENABLE_DL_CORE_AFFINITY", e.what());
            phydriver_config.enable_dl_core_affinity = 1;
        }

        try
        {
            phydriver_config.dlc_core_packing_scheme = static_cast<uint8_t>(static_cast<uint16_t>(root[YAML_PARAM_DLC_CORE_PACKING_SCHEME]));
        }
        catch(const std::exception& e)
        {
            NVLOGW_FMT(TAG, "{} Using default value of 0 to YAML_PARAM_DLC_CORE_PACKING_SCHEME", e.what());
            phydriver_config.dlc_core_packing_scheme = 0;
        }

        try
        {
            phydriver_config.aggr_obj_non_avail_th = static_cast<uint32_t>(root[YAML_PARAM_AGGR_OBJ_NON_AVAIL_TH]);
        }
        catch(const std::exception& e)
        {
            NVLOGW_FMT(TAG, "{} Using default value of 5 to YAML_PARAM_AGGR_OBJ_NON_AVAIL_TH", e.what());
            phydriver_config.aggr_obj_non_avail_th = 5;
        }
        try
        {
            phydriver_config.sendCPlane_timing_error_th_ns = static_cast<uint32_t>(static_cast<uint32_t>(root[YAML_PARAM_SENDCPLANE_TIMING_ERROR_TH_NS]));
        }
        catch(const std::exception& e)
        {
            NVLOGW_FMT(TAG, "{} Using default value of 0 us to YAML_PARAM_SENDCPLANE_TIMING_ERROR_TH_NS", e.what());
            phydriver_config.sendCPlane_timing_error_th_ns = 0;
        }

        try
        {
            phydriver_config.sendCPlane_ulbfw_backoff_th_ns = static_cast<uint32_t>(static_cast<uint32_t>(root[YAML_PARAM_SENDCPLANE_ULBFW_BACKOFF_TH_NS]));
        }
        catch(const std::exception& e)
        {
            NVLOGW_FMT(TAG, "{} Using default value of 100000 ns (100 us) to YAML_PARAM_SENDCPLANE_ULBFW_BACKOFF_TH_NS", e.what());
            phydriver_config.sendCPlane_ulbfw_backoff_th_ns = 100000;
        }

        try
        {
            phydriver_config.sendCPlane_dlbfw_backoff_th_ns = static_cast<uint32_t>(static_cast<uint32_t>(root[YAML_PARAM_SENDCPLANE_DLBFW_BACKOFF_TH_NS]));
        }
        catch(const std::exception& e)
        {
            NVLOGW_FMT(TAG, "{} Using default value of 100000 ns (100 us) to YAML_PARAM_SENDCPLANE_DLBFW_BACKOFF_TH_NS", e.what());
            phydriver_config.sendCPlane_dlbfw_backoff_th_ns = 100000;
        }

        if(root.has_key("pusch_workCancelMode"))
        {
            phydriver_config.pusch_workCancelMode          = static_cast<uint8_t>(root[YAML_PARAM_PUSCH_WORKCANCELMODE]);
            if(phydriver_config.pusch_workCancelMode >= cuphyPuschWorkCancelMode_t::PUSCH_MAX_WORK_CANCEL_MODES)
            {
                NVLOGE_FMT(TAG, AERIAL_CONFIG_EVENT, "cuphycontroller config. yaml has invalid value ({}) for pusch_workCancelMode key. Supported range is [0, {}).",
                           phydriver_config.pusch_workCancelMode, +cuphyPuschWorkCancelMode_t::PUSCH_MAX_WORK_CANCEL_MODES);
                return -1;
            }
        }
        else
        {
            phydriver_config.pusch_workCancelMode = cuphyPuschWorkCancelMode_t::PUSCH_NO_WORK_CANCEL;
            NVLOGW_FMT(TAG, "cuphycontroller config. yaml does not have pusch_workCancelMode key (experimental feature); defaulting to {}.", phydriver_config.pusch_workCancelMode);
        }

        try
        {
            phydriver_config.puschTdi                   = static_cast<uint8_t>(static_cast<uint16_t>(root[YAML_PARAM_PUSCH_TDI]));
        }
        catch(const std::exception& e)
        {
            NVLOGW_FMT(TAG, "{} Using default value of 0 to PUSCH-TDI", e.what());
            phydriver_config.puschTdi = 0;
        }

        try
        {
            phydriver_config.puschCfo = static_cast<uint8_t>(static_cast<uint16_t>(root[YAML_PARAM_PUSCH_CFO]));
        }
        catch(const std::exception& e)
        {
            NVLOGW_FMT(TAG, "{} Using default value of 0 to PUSCH-CFO", e.what());
            phydriver_config.puschCfo = 0;
        }

        try
        {
            phydriver_config.puschSelectEqCoeffAlgo                   = static_cast<uint8_t>(static_cast<uint16_t>(root[YAML_PARAM_PUSCH_SELECT_EQCOEFFALGO]));
        }
        catch(const std::exception& e)
        {
            NVLOGW_FMT(TAG, "{} Using default value of 1 to PUSCH-EQCOEFFALGO", e.what());
            phydriver_config.puschSelectEqCoeffAlgo = 1;
        }

        try
        {
            phydriver_config.puschSelectChEstAlgo                   = static_cast<uint8_t>(static_cast<uint16_t>(root[YAML_PARAM_PUSCH_SELECT_CHESTALGO]));
        }
        catch(const std::exception& e)
        {
            NVLOGW_FMT(TAG, "{} Using default value of 1 to PUSCH-CHESTALGO", e.what());
            phydriver_config.puschSelectChEstAlgo = 1;
        }
        
        try
        {
            phydriver_config.puschEnablePerPrgChEst                = static_cast<uint8_t>(static_cast<uint16_t>(root[YAML_PARAM_PUSCH_ENABLE_PERPRGCHEST]));
        }
        catch(const std::exception& e)
        {
            NVLOGW_FMT(TAG, "{} Using default value of 0 to PUSCH-ENABLEPERPRGCHEST", e.what());
            phydriver_config.puschEnablePerPrgChEst = 0;
        }

        try
        {
            phydriver_config.puschDftSOfdm = static_cast<uint8_t>(static_cast<uint16_t>(root[YAML_PARAM_PUSCH_DFTSOFDM]));
        }
        catch(const std::exception& e)
        {
            NVLOGW_FMT(TAG, "{} Using default value of 0 to PUSCH-DFTSOFDM", e.what());
            phydriver_config.puschDftSOfdm = 0;
        }

        try
        {
            phydriver_config.puschTbSizeCheck = static_cast<uint8_t>(static_cast<uint16_t>(root[YAML_PARAM_PUSCH_TBSIZECHECK]));
        }
        catch(const std::exception& e)
        {
            NVLOGW_FMT(TAG, "{} Using default value of 0 to PUSCH-TBSIZECHECK", e.what());
            phydriver_config.puschTbSizeCheck = 0;
        }


        try
        {
            phydriver_config.pusch_deviceGraphLaunchEn = static_cast<uint8_t>(static_cast<uint16_t>(root[YAML_PARAM_PUSCH_DEVICEGRAPHLAUNCHEN]));
        }
        catch(const std::exception& e)
        {
            NVLOGW_FMT(TAG, "{} Using default value of 1 to PUSCH-DEVICEGRAPHLAUNCHEN", e.what());
            phydriver_config.pusch_deviceGraphLaunchEn = 1;
        }

        try
        {
            phydriver_config.pusch_waitTimeOutPreEarlyHarqUs = static_cast<uint16_t>(root[YAML_PARAM_PUSCH_WAIT_TIMEOUT_PRE_EARLY_HARQ_US]);
        }
        catch(const std::exception& e)
        {
            NVLOGW_FMT(TAG, "{} Using default value of 3000 to PUSCH-WAITTIMEOUTPREEHQUS", e.what());
            phydriver_config.pusch_waitTimeOutPreEarlyHarqUs = 3000;
        }

        try
        {
            phydriver_config.pusch_waitTimeOutPostEarlyHarqUs = static_cast<uint16_t>(root[YAML_PARAM_PUSCH_WAIT_TIMEOUT_POST_EARLY_HARQ_US]);
        }
        catch(const std::exception& e)
        {
            NVLOGW_FMT(TAG, "{} Using default value of 3000 to PUSCH-WAITTIMEOUTPOSTEHQUS", e.what());
            phydriver_config.pusch_waitTimeOutPostEarlyHarqUs = 3000;
        }

        try
        {
            phydriver_config.puschTo = static_cast<uint8_t>(static_cast<uint16_t>(root[YAML_PARAM_PUSCH_TO]));
        }
        catch(const std::exception& e)
        {
            NVLOGW_FMT(TAG, "{} Using default value of 0 to PUSCH-TO", e.what());
            phydriver_config.puschTo = 0;
        }

        try
        {
            phydriver_config.puschRssi                   = static_cast<uint8_t>(static_cast<uint16_t>(root[YAML_PARAM_PUSCH_RSSI]));
        }
        catch(const std::exception& e)
        {
            NVLOGW_FMT(TAG, "{} Using default value of 0 to PUSCH-RSSI", e.what());
            phydriver_config.puschRssi = 0;
        }

        try
        {
            phydriver_config.puschSinr                   = static_cast<uint8_t>(static_cast<uint16_t>(root[YAML_PARAM_PUSCH_SINR]));
        }
        catch(const std::exception& e)
        {
            NVLOGW_FMT(TAG, "{} Using default value of 0 to PUSCH-SINR", e.what());
            phydriver_config.puschSinr = 0;
        }

        try
        {
            phydriver_config.puschWeightedAverageCfo = static_cast<uint8_t>(static_cast<uint16_t>(root[YAML_PARAM_PUSCH_WEIGHTED_AVERAGE_CFO]));
        }
        catch(const std::exception& e)
        {
            NVLOGW_FMT(TAG, "{} Using default value of 0 to PUSCH-WEIGHTED-AVERAGE-CFO", e.what());
            phydriver_config.puschWeightedAverageCfo = 0;
        }

        try
        {
            if (root.has_key(YAML_PARAM_PUSCHRX_CHEST_FACTORY_SETTINGS_FILENAME))
            {
                phydriver_config.puschrxChestFactorySettingsFilename = root[
                    YAML_PARAM_PUSCHRX_CHEST_FACTORY_SETTINGS_FILENAME].operator std::string();
            }
        }
        catch(const std::exception& e)
        {
            NVLOGW_FMT(TAG, "{} PuschRx Chest Factory settings filename was not provided", e.what());
        }

        try
        {
            phydriver_config.puxchPolarDcdrListSz                   = static_cast<uint8_t>(root[YAML_PARAM_PUXCH_POLAR_DCDR_LIST_SZ]);
        }
        catch(const std::exception& e)
        {
            NVLOGW_FMT(TAG, "{} Using default value of 1 to PUSCH_POLAR_DCDR_LIST_SZ", e.what());
            phydriver_config.puxchPolarDcdrListSz = 1;
        }

        try
        {
            phydriver_config.split_ul_cuda_streams = static_cast<uint8_t>(static_cast<uint16_t>(root[YAML_PARAM_SPLIT_UL_CUDA_STREAMS]));
        }
        catch(const std::exception& e)
        {
            NVLOGW_FMT(TAG, "{} Using default value of 0 to YAML_PARAM_SPLIT_UL_CUDA_STREAMS", e.what());
            phydriver_config.split_ul_cuda_streams = 0;
        }

        try
        {
            phydriver_config.serialize_pucch_pusch = static_cast<uint8_t>(static_cast<uint16_t>(root[YAML_PARAM_SERIALIZE_PUCCH_PUSCH]));
        }
        catch(const std::exception& e)
        {
            NVLOGW_FMT(TAG, "{} Using default value of 0 to YAML_PARAM_SERIALIZE_PUCCH_PUSCH", e.what());
            phydriver_config.serialize_pucch_pusch = 0;
        }

        try
        {
            phydriver_config.ul_order_max_rx_pkts                   = static_cast<uint32_t>(root[YAML_PARAM_UL_ORDER_MAX_RX_PKTS]);
        }
        catch(const std::exception& e)
        {
            NVLOGW_FMT(TAG, "{} Using default value of 512 to UL_ORDER_MAX_RX_PKTS", e.what());
            phydriver_config.ul_order_max_rx_pkts = 512;
        }

        try
        {
            phydriver_config.ul_order_rx_pkts_timeout_ns                   = static_cast<uint32_t>(root[YAML_PARAM_UL_ORDER_RX_PKTS_TIMEOUT_NS]);
        }
        catch(const std::exception& e)
        {
            NVLOGW_FMT(TAG, "{} Using default value of 100us to YAML_PARAM_UL_ORDER_RX_PKTS_TIMEOUT_NS", e.what());
            phydriver_config.ul_order_rx_pkts_timeout_ns = 100000;
        }        

        try
        {
            phydriver_config.ul_order_timeout_log_interval_ns                   = static_cast<uint32_t>(root[YAML_PARAM_UL_ORDER_TIMEOUT_LOG_INTERVAL_NS]);
        }
        catch(const std::exception& e)
        {
            NVLOGW_FMT(TAG, "{} Using default value of 1s to YAML_PARAM_UL_ORDER_TIMEOUT_LOG_INTERVAL_NS", e.what());
            phydriver_config.ul_order_timeout_log_interval_ns = 1000000000;
        }

        try
        {
            phydriver_config.mCh_segment_proc_enable = static_cast<uint8_t>(static_cast<uint16_t>(root[YAML_PARAM_MCH_SEGMENT_PROC_ENABLE]));
        }
        catch(const std::exception& e)
        {
            NVLOGW_FMT(TAG, "{} Using default value of 0 to YAML_PARAM_MCH_SEGMENT_PROC_ENABLE", e.what());
            phydriver_config.mCh_segment_proc_enable = 0;
        }

        try
        {
            phydriver_config.pusch_aggr_per_ctx = static_cast<uint8_t>(static_cast<uint16_t>(root[YAML_PARAM_PUSCH_AGGR_PER_CTX]));
        }
        catch(const std::exception& e)
        {
            NVLOGW_FMT(TAG, "{} Using default value of 3 to YAML_PARAM_PUSCH_AGGR_PER_CTX", e.what());
            phydriver_config.pusch_aggr_per_ctx = 3;
        }

        try
        {
            phydriver_config.max_harq_pools = static_cast<uint16_t>(static_cast<uint16_t>(root[YAML_PARAM_MAX_HARQ_POOLS]));
        }
        catch(const std::exception& e)
        {
            NVLOGW_FMT(TAG, "{} Using default value of 384 to YAML_PARAM_MAX_HARQ_POOLS", e.what());
            phydriver_config.max_harq_pools = 384;
        }        

        try
        {
            phydriver_config.prach_aggr_per_ctx = static_cast<uint8_t>(static_cast<uint16_t>(root[YAML_PARAM_PRACH_AGGR_PER_CTX]));
        }
        catch(const std::exception& e)
        {
            NVLOGW_FMT(TAG, "{} Using default value of 2 to YAML_PARAM_PRACH_AGGR_PER_CTX", e.what());
            phydriver_config.prach_aggr_per_ctx = 2;
        }

        try
        {
            phydriver_config.ul_input_buffer_per_cell = static_cast<uint8_t>(static_cast<uint16_t>(root[YAML_PARAM_UL_INPUT_BUFFER_NUM_PER_CELL]));
        }
        catch(const std::exception& e)
        {
            NVLOGW_FMT(TAG, "{} Using default value of 10 to YAML_PARAM_UL_INPUT_BUFFER_NUM_PER_CELL", e.what());
            phydriver_config.ul_input_buffer_per_cell = 10;
        }

        try
        {
            phydriver_config.pucch_aggr_per_ctx = static_cast<uint8_t>(static_cast<uint16_t>(root[YAML_PARAM_PUCCH_AGGR_PER_CTX]));
        }
        catch(const std::exception& e)
        {
            NVLOGW_FMT(TAG, "{} Using default value of 4 to YAML_PARAM_PUCCH_AGGR_PER_CTX", e.what());
            phydriver_config.pucch_aggr_per_ctx = 4;
        }

        try
        {
            phydriver_config.srs_aggr_per_ctx = static_cast<uint8_t>(static_cast<uint16_t>(root[YAML_PARAM_SRS_AGGR_PER_CTX]));
        }
        catch(const std::exception& e)
        {
            NVLOGW_FMT(TAG, "{} Using default value of 2 to YAML_PARAM_SRS_AGGR_PER_CTX", e.what());
            phydriver_config.srs_aggr_per_ctx = 2;
        }

        try
        {
            phydriver_config.ul_input_buffer_per_cell_srs = static_cast<uint8_t>(static_cast<uint16_t>(root[YAML_PARAM_UL_INPUT_BUFFER_NUM_PER_CELL_SRS]));
        }
        catch(const std::exception& e)
        {
            NVLOGW_FMT(TAG, "{} Using default value of 4 to YAML_PARAM_UL_INPUT_BUFFER_NUM_PER_CELL_SRS", e.what());
            phydriver_config.ul_input_buffer_per_cell_srs = 4;
        }

        try
        {
            phydriver_config.max_ru_unhealthy_ul_slots = static_cast<uint32_t>(static_cast<uint32_t>(root[YAML_PARAM_MAX_RU_UNHEALTHY_UL_SLOTS]));
        }
        catch(const std::exception& e)
        {
            NVLOGW_FMT(TAG, "{} Using default value of 0 for YAML_PARAM_MAX_RU_UNHEALTHY_UL_SLOTS", e.what());
            phydriver_config.max_ru_unhealthy_ul_slots = 0;
        }

        try
        {
            phydriver_config.srs_chest_algo_type = static_cast<uint32_t>(static_cast<uint8_t>(root[YAML_PARAM_SRS_CHEST_ALGO_TYPE]));
        }
        catch(const std::exception& e)
        {
            NVLOGW_FMT(TAG, "{} Using default value of 0 for YAML_PARAM_SRS_CHEST_ALGO_TYPE", e.what());
            phydriver_config.srs_chest_algo_type = 0;
        }
        
        try
        {
            phydriver_config.srs_chest_tol2_normalization_algo_type = static_cast<uint8_t>(root[YAML_PARAM_SRS_CHEST_TOL2_NORMALIZATION_ALGO_TYPE]);
        }
        catch(const std::exception& e)
        {
            NVLOGW_FMT(TAG, "{} Using default value of 1 for YAML_PARAM_SRS_CHEST_TOL2_NORMALIZATION_ALGO_TYPE", e.what());
            phydriver_config.srs_chest_tol2_normalization_algo_type = 1;
        }
        
        try
        {
            phydriver_config.srs_chest_tol2_constant_scaler = static_cast<float>(root[YAML_PARAM_SRS_CHEST_TOL2_CONSTANT_SCALER]);
        }
        catch(const std::exception& e)
        {
            NVLOGW_FMT(TAG, "{} Using default value of 32768.0 for YAML_PARAM_SRS_CHEST_TOL2_CONSTANT_SCALER", e.what());
            phydriver_config.srs_chest_tol2_constant_scaler = 32768.0f;
        }
        
        try
        {
            phydriver_config.bfw_power_normalization_alg_selector = static_cast<uint8_t>(static_cast<uint8_t>(root[YAML_PARAM_BFW_POWER_NORMALIZATION_ALG_SELECTOR]));
        }
        catch(const std::exception& e)
        {
            NVLOGW_FMT(TAG, "{} Using default value of 1 for YAML_PARAM_BFW_POWER_NORMALIZATION_ALG_SELECTOR", e.what());
            phydriver_config.bfw_power_normalization_alg_selector = 1;
        }

        try
        {
            phydriver_config.bfw_beta_prescaler = static_cast<float>(static_cast<float>(root[YAML_PARAM_BFW_BETA_PRESCALER]));
        }
        catch(const std::exception& e)
        {
            phydriver_config.bfw_beta_prescaler = 16384;  // same with SimCtrl.bfw.beta in cfgSimCtrl.m
            NVLOGW_FMT(TAG, "{} Using default value of 16384 for YAML_PARAM_BFW_BETA_PRESCALER {}", e.what(), phydriver_config.bfw_beta_prescaler);
        }

	try
    {
        phydriver_config.total_num_srs_chest_buffers = static_cast<uint32_t>(root[YAML_PARAM_TOTAL_NUM_SRS_CHEST_BUFFERS]);
        if (phydriver_config.total_num_srs_chest_buffers > slot_command_api::MAX_SRS_CHEST_BUFFERS)
        {
            NVLOGW_FMT(TAG, "ERROR!! YAML_PARAM_TOTAL_NUM_SRS_CHEST_BUFFERS is greater than the maximum allowed value of {}", slot_command_api::MAX_SRS_CHEST_BUFFERS);
            phydriver_config.total_num_srs_chest_buffers = slot_command_api::MAX_SRS_CHEST_BUFFERS;
            NVLOGC_FMT(TAG, "Setting YAML_PARAM_TOTAL_NUM_SRS_CHEST_BUFFERS to the maximum allowed value of {}", slot_command_api::MAX_SRS_CHEST_BUFFERS);
        }
    }
    catch (const std::exception &e)
    {
        phydriver_config.total_num_srs_chest_buffers = slot_command_api::MAX_SRS_CHEST_BUFFERS;
        NVLOGW_FMT(TAG, "{} Using default value of {} for YAML_PARAM_TOTAL_NUM_SRS_CHEST_BUFFERS", e.what(), phydriver_config.total_num_srs_chest_buffers);
    }

    try
    {
            phydriver_config.send_static_bfw_wt_all_cplane = static_cast<uint8_t>(static_cast<uint8_t>(root[YAML_PARAM_SEND_STATIC_BFW_WT_ALL_CPLANE]));
        }
        catch(const std::exception& e)
        {
            NVLOGW_FMT(TAG, "{} Using default value of 0 for YAML_PARAM_SEND_STATIC_BFW_WT_ALL_CPLANE", e.what());
            phydriver_config.send_static_bfw_wt_all_cplane = 1;
        }

        try
        {
            phydriver_config.ul_pcap_capture_enable = static_cast<uint8_t>(static_cast<uint8_t>(root[YAML_PARAM_UL_PCAP_CAPTURE_ENABLE]));
        }
        catch(const std::exception& e)
        {
            NVLOGW_FMT(TAG, "{} Using default value of 0 for YAML_PARAM_UL_PCAP_CAPTURE_ENABLE", e.what());
            phydriver_config.ul_pcap_capture_enable = 0;
        }

        if(phydriver_config.ul_pcap_capture_enable == 1)
        {
            try
            {
                phydriver_config.ul_pcap_capture_thread_cpu_affinity = static_cast<uint8_t>(static_cast<uint8_t>(root[YAML_PARAM_UL_PCAP_CAPTURE_THREAD_CPU_AFFINITY]));
                phydriver_config.ul_pcap_capture_thread_sched_priority = static_cast<uint8_t>(static_cast<uint8_t>(root[YAML_PARAM_UL_PCAP_CAPTURE_THREAD_SCHED_PRIORITY]));
            }
            catch(const std::exception& e)
            {
                NVLOGW_FMT(TAG, "{} UL PCAP feature enabled but required fields not enabled, please set {}, {}", e.what(), YAML_PARAM_UL_PCAP_CAPTURE_THREAD_CPU_AFFINITY, YAML_PARAM_UL_PCAP_CAPTURE_THREAD_SCHED_PRIORITY);
                exit(1);
            }
        }

        try
        {
            phydriver_config.pcap_logger_ul_cplane_enable       = static_cast<uint8_t>(root[YAML_PARAM_PCAP_LOGGER__ENABLE_UL_CPLANE]); 
            phydriver_config.pcap_logger_dl_cplane_enable       = static_cast<uint8_t>(root[YAML_PARAM_PCAP_LOGGER__ENABLE_DL_CPLANE]); 
            phydriver_config.pcap_logger_thread_cpu_affinity    = static_cast<uint8_t>(root[YAML_PARAM_PCAP_LOGGER__THREAD_CPU_AFFINITY]); 
            phydriver_config.pcap_logger_thread_sched_prio      = static_cast<uint8_t>(root[YAML_PARAM_PCAP_LOGGER__THREAD_SCHED_PRIO]); 
            phydriver_config.pcap_logger_file_save_dir          = static_cast<std::string>(root[YAML_PARAM_PCAP_LOGGER__FILE_SAVE_DIR]);
        }
        catch(const std::exception& e)
        {
            phydriver_config.pcap_logger_ul_cplane_enable       = 0; 
            phydriver_config.pcap_logger_dl_cplane_enable       = 0;
            phydriver_config.pcap_logger_thread_cpu_affinity    = 0;
            phydriver_config.pcap_logger_thread_sched_prio      = 0;
            phydriver_config.pcap_logger_file_save_dir          = "."; 
            NVLOGW_FMT(TAG, "{} PCAP Logging parameters are set incorrectly, so logger is disabled by default", e.what());
        }

        try
        {
            phydriver_config.dlc_bfw_enable_divide_per_cell = static_cast<uint8_t>(root[YAML_PARAM_DLC_BFW_ENABLE_DIVIDE_PER_CELL]);
        }
        catch(const std::exception& e)
        {
            NVLOGW_FMT(TAG, "{} Using default value of 0 for YAML_PARAM_DLC_BFW_ENABLE_DIVIDE_PER_CELL", e.what());
            phydriver_config.dlc_bfw_enable_divide_per_cell = 0;
        }

        try
        {
            phydriver_config.ulc_bfw_enable_divide_per_cell = static_cast<uint8_t>(root[YAML_PARAM_ULC_BFW_ENABLE_DIVIDE_PER_CELL]);
        }
        catch(const std::exception& e)
        {
            NVLOGW_FMT(TAG, "{} Using default value of 0 for YAML_PARAM_ULC_BFW_ENABLE_DIVIDE_PER_CELL", e.what());
            phydriver_config.ulc_bfw_enable_divide_per_cell = 0;
        }

        try
        {
            phydriver_config.dlc_alloc_cplane_bfw_txq = static_cast<uint8_t>(root[YAML_PARAM_DLC_ALLOC_CPLANE_BFW_TXQ]);
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what()<<" Using default value of 0 for YAML_PARAM_DLC_ALLOC_CPLANE_BFW_TXQ" << std::endl;
            phydriver_config.dlc_alloc_cplane_bfw_txq = 0;
        }
        
        try
        {
            phydriver_config.ulc_alloc_cplane_bfw_txq = static_cast<uint8_t>(root[YAML_PARAM_ULC_ALLOC_CPLANE_BFW_TXQ]);
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what()<<" Using default value of 0 for YAML_PARAM_ULC_ALLOC_CPLANE_BFW_TXQ" << std::endl;
            phydriver_config.ulc_alloc_cplane_bfw_txq = 0;
        }

        try
        {
            auto static_beam_id_start_val = root[YAML_PARAM_STATIC_BEAM_ID_START].as<int>();
            auto static_beam_id_end_val = root[YAML_PARAM_STATIC_BEAM_ID_END].as<int>();
            auto dynamic_beam_id_start_val = root[YAML_PARAM_DYNAMIC_BEAM_ID_START].as<int>();
            auto dynamic_beam_id_end_val = root[YAML_PARAM_DYNAMIC_BEAM_ID_END].as<int>();

            // 1. All values must be in [1, 32767]
            if (static_beam_id_start_val < 1 || static_beam_id_start_val > 32767 ||
                static_beam_id_end_val   < 1 || static_beam_id_end_val   > 32767 ||
                dynamic_beam_id_start_val < 1 || dynamic_beam_id_start_val > 32767 ||
                dynamic_beam_id_end_val   < 1 || dynamic_beam_id_end_val   > 32767) {
                NVLOGE_FMT(TAG, AERIAL_CONFIG_EVENT, "CONFIG ERROR: All beam IDs must be in the range 1 to 32767");
                return -1;
            }

            // 2. Ranges must be valid (start ≤ end)
            if (static_beam_id_start_val > static_beam_id_end_val) {
                NVLOGE_FMT(TAG, AERIAL_CONFIG_EVENT, "CONFIG ERROR: static_beam_id_start must be <= static_beam_id_end");
                return -1;
            }
            if (dynamic_beam_id_start_val > dynamic_beam_id_end_val) {
                NVLOGE_FMT(TAG, AERIAL_CONFIG_EVENT, "CONFIG ERROR: dynamic_beam_id_start must be <= dynamic_beam_id_end");
                return -1;
            }

            // 3. Static range must be lower and not overlap with dynamic range
            if (static_beam_id_end_val >= dynamic_beam_id_start_val) {
                NVLOGE_FMT(TAG, AERIAL_CONFIG_EVENT, "CONFIG ERROR: static beam ID range [{}, {}] overlaps with or is not lower than dynamic beam ID range [{}, {}]",
                    static_beam_id_start_val, static_beam_id_end_val, dynamic_beam_id_start_val, dynamic_beam_id_end_val);
                return -1;
            }
            // Assign after all checks pass
            phydriver_config.static_beam_id_start = static_cast<uint16_t>(static_beam_id_start_val);
            phydriver_config.static_beam_id_end = static_cast<uint16_t>(static_beam_id_end_val);
            phydriver_config.dynamic_beam_id_start = static_cast<uint16_t>(dynamic_beam_id_start_val);
            phydriver_config.dynamic_beam_id_end = static_cast<uint16_t>(dynamic_beam_id_end_val);
        }
        catch(const std::exception& e)
        {
            NVLOGW_FMT(TAG, "{} Using default value of 0 for YAML_PARAM_BEAMID_CONFIG", e.what());
            phydriver_config.static_beam_id_start = 1;
            phydriver_config.static_beam_id_end = 16527;
            phydriver_config.dynamic_beam_id_start = 16528;
            phydriver_config.dynamic_beam_id_end = 32767;
        }

        try
        {
            phydriver_config.bfw_c_plane_chaining_mode = static_cast<uint8_t>(root[YAML_PARAM_BFW_C_PLANE_CHAINING_MODE]);
        }
        catch(const std::exception& e)
        {
            NVLOGW_FMT(TAG, "{} Using default value of 0 for YAML_PARAM_BFW_C_PLANE_CHAINING_MODE", e.what());
            phydriver_config.bfw_c_plane_chaining_mode = 0;
        }

        try
        {
            phydriver_config.enable_tx_notification = static_cast<uint8_t>(root[YAML_PARAM_ENABLE_TX_NOTIFICATION]);
        }
        catch(const std::exception& e)
        {
            NVLOGW_FMT(TAG, "{} Using default value of 0 for YAML_PARAM_ENABLE_TX_NOTIFICATION", e.what());
            phydriver_config.enable_tx_notification = 0;
        }

        std::string log_str = static_cast<std::string>(root[YAML_PARAM_LOGLVL]);

        if(log_str.compare("ERROR") == 0)
            phydriver_config.log_level = L1_LOG_LVL_ERROR;
        else if(log_str.compare("INFO") == 0)
            phydriver_config.log_level = L1_LOG_LVL_INFO;
        else if(log_str.compare("DBG") == 0)
            phydriver_config.log_level = L1_LOG_LVL_DBG;
        else
        {
            NVLOGE_FMT(TAG, AERIAL_CONFIG_EVENT, "Input log level is not valid. Setting ERROR");
            phydriver_config.log_level = L1_LOG_LVL_ERROR;
        }

        yaml::node workers_ul = root[YAML_PARAM_WORKERS_UL];
        for(size_t i = 0; i < workers_ul.length(); ++i)
        {
            phydriver_config.workers_list_ul.push_back((uint8_t)static_cast<uint16_t>(workers_ul[i]));
        }

        try
        {
            yaml::node dl_wait_th_list = root[YAML_PARAM_DL_WAIT_TH_NS];
            for(size_t i = 0; i < dl_wait_th_list.length(); ++i)
            {
                phydriver_config.dl_wait_th_list.push_back((uint32_t)static_cast<uint32_t>(dl_wait_th_list[i]));
            }
        }
        catch(const std::exception& e)
        {
            NVLOGW_FMT(TAG,"Exception {} DL Wait Thresholds will be set to default values", e.what());
        }

       yaml::node workers_dl = root[YAML_PARAM_WORKERS_DL];
        for(size_t i = 0; i < workers_dl.length(); ++i)
        {
            phydriver_config.workers_list_dl.push_back((uint8_t)static_cast<uint16_t>(workers_dl[i]));
        }

        try
        {
            phydriver_config.debug_worker = static_cast<uint16_t>(root[YAML_PARAM_DEBUG_WORKER]);
        }
        catch(const std::exception& e)
        {
            phydriver_config.debug_worker = -1;
        }

        if(root.has_key(YAML_PARAM_DATA_CONFIG)) {
            yaml::node data_cfg = root[YAML_PARAM_DATA_CONFIG];
            // Handle optional data_core (set to -1 to disable)
            // Support deprecated "datalake_core" for backward compatibility
            if(data_cfg.has_key("data_core")) {
                phydriver_config.data_core = data_cfg["data_core"].as<uint16_t>();
            } else if(data_cfg.has_key("datalake_core")) {
                NVLOGW_FMT(TAG, "Config key 'datalake_core' is deprecated, use 'data_core' instead.");
                phydriver_config.data_core = data_cfg["datalake_core"].as<uint16_t>();
            } else {
                phydriver_config.data_core = -1;
            }
            
            if(data_cfg.has_key("datalake_db_write_enable")) {
                phydriver_config.datalake_db_write_enable = data_cfg["datalake_db_write_enable"].as<uint8_t>();
            } else {
                phydriver_config.datalake_db_write_enable = 0;
            }
            
            if(data_cfg.has_key("datalake_samples")) {
                phydriver_config.datalake_samples = data_cfg["datalake_samples"].as<uint32_t>();
            } else {
                phydriver_config.datalake_samples = 1000000;
            }
            
            if(data_cfg.has_key("datalake_address")) {
                phydriver_config.datalake_address = data_cfg["datalake_address"].as<std::string>();
            } else {
                phydriver_config.datalake_address = "localhost";
            }
            
            if(data_cfg.has_key("datalake_engine")) {
                phydriver_config.datalake_engine = data_cfg["datalake_engine"].as<std::string>();
            } else {
                phydriver_config.datalake_engine = "Memory";
            }
            
            if(data_cfg.has_key("datalake_store_failed_pdu")) {
                phydriver_config.datalake_store_failed_pdu = data_cfg["datalake_store_failed_pdu"].as<uint8_t>();
            } else {
                phydriver_config.datalake_store_failed_pdu = 0;
            }
            
            if(data_cfg.has_key("num_rows_fh")) {
                phydriver_config.num_rows_fh = data_cfg["num_rows_fh"].as<uint32_t>();
            } else {
                phydriver_config.num_rows_fh = 120;
            }
            
            if(data_cfg.has_key("num_rows_pusch")) {
                phydriver_config.num_rows_pusch = data_cfg["num_rows_pusch"].as<uint32_t>();
            } else {
                phydriver_config.num_rows_pusch = 400;
            }
            
            if(data_cfg.has_key("num_rows_hest")) {
                phydriver_config.num_rows_hest = data_cfg["num_rows_hest"].as<uint32_t>();
            } else {
                phydriver_config.num_rows_hest = 140;
            }
            
            // E3 Agent enable flag (runtime configurable)
            if(data_cfg.has_key("e3_agent_enable")) {
                phydriver_config.e3_agent_enabled = data_cfg["e3_agent_enable"].as<uint8_t>() != 0;
            } else {
                phydriver_config.e3_agent_enabled = false;
            }
            
            if(data_cfg.has_key("e3_rep_port")) {
                phydriver_config.e3_rep_port = data_cfg["e3_rep_port"].as<uint16_t>();
            } else {
                phydriver_config.e3_rep_port = 5555;
            }
            
            if(data_cfg.has_key("e3_pub_port")) {
                phydriver_config.e3_pub_port = data_cfg["e3_pub_port"].as<uint16_t>();
            } else {
                phydriver_config.e3_pub_port = 5556;
            }
            
            if(data_cfg.has_key("e3_sub_port")) {
                phydriver_config.e3_sub_port = data_cfg["e3_sub_port"].as<uint16_t>();
            } else {
                phydriver_config.e3_sub_port = 5557;
            }
            
            // Handle optional datalake_drop_tables
            if(data_cfg.has_key("datalake_drop_tables")) {
                phydriver_config.datalake_drop_tables = data_cfg["datalake_drop_tables"].as<uint8_t>();
            } else {
                phydriver_config.datalake_drop_tables = 0;
            }
            
            // Parse enabled data types array
            if(data_cfg.has_key("datalake_data_types")) {
                yaml::node enabled_types = data_cfg["datalake_data_types"];
                for(size_t i = 0; i < enabled_types.length(); ++i) {
                    phydriver_config.datalake_data_types.push_back(enabled_types[i].as<std::string>());
                }
            } else {
                // Default: enable all data types
                phydriver_config.datalake_data_types = {"fh", "pusch", "hest"};
            }
        } else {
            NVLOGI_FMT(TAG," Using default values for data_config");
            phydriver_config.data_core = -1;
            phydriver_config.datalake_db_write_enable = 0;
            phydriver_config.datalake_samples = 1000000;
            phydriver_config.datalake_address = "localhost";
            phydriver_config.datalake_engine = "Memory";
            phydriver_config.datalake_data_types = {"fh", "pusch", "hest"};
            phydriver_config.datalake_store_failed_pdu = 0;
            phydriver_config.num_rows_fh = 120;
            phydriver_config.num_rows_pusch = 400;
            phydriver_config.num_rows_hest = 140;
            phydriver_config.datalake_drop_tables = 0;
            phydriver_config.e3_agent_enabled = false;
            phydriver_config.e3_rep_port = 5555;
            phydriver_config.e3_pub_port = 5556;
            phydriver_config.e3_sub_port = 5557;
        }
    
        yaml::node gpus_list_y = root[YAML_PARAM_GPUS];
        for(size_t i = 0; i < gpus_list_y.length(); ++i)
        {
            uint16_t gpu_id = static_cast<uint16_t>(gpus_list_y[i]);
            
            // Check if GPU device exists (basic check)
            std::string sys_path = "/dev/nvidia" + std::to_string(gpu_id);
            std::ifstream gpu_check(sys_path);
            if (!gpu_check.good()) {
                NVLOGE_FMT(TAG, AERIAL_CONFIG_EVENT, "GPU device {} not found (YAML key: {}[{}]). Please verify that the specified GPU exists on this system.", gpu_id, YAML_PARAM_GPUS, i);
                return -1;
            }
            
            phydriver_config.gpus_list.push_back(gpu_id);
        }

        yaml::node nics_list_y = root[YAML_PARAM_NICS];
        for(size_t i = 0; i < nics_list_y.length(); ++i)
        {
            struct nic_config nc;
            nc.address = static_cast<std::string>(nics_list_y[i][YAML_PARAM_NICS_NIC]);
            
            // Validate that the NIC PCI address exists on the system
            std::string sys_path = "/sys/bus/pci/devices/" + nc.address;
            std::ifstream nic_check(sys_path + "/vendor");
            if (!nic_check.good()) {
                NVLOGE_FMT(TAG, AERIAL_CONFIG_EVENT, "PCI device {} not found (YAML key: {}[{}].{}). Please verify that the specified PCI address exists on this system.", nc.address, YAML_PARAM_NICS, i, YAML_PARAM_NICS_NIC);
                return -1;
            }
            
            nc.mtu = static_cast<uint16_t>(nics_list_y[i][YAML_PARAM_NICS_MTU]);
            nc.cpu_mbuf_num = static_cast<uint32_t>(nics_list_y[i][YAML_PARAM_NICS_CPU_MBUFS]);
            nc.tx_req_num = static_cast<uint32_t>(nics_list_y[i][YAML_PARAM_NICS_UPLANE_TX_HANDLES]);
            nc.txq_size = static_cast<uint16_t>(nics_list_y[i][YAML_PARAM_NICS_TXQ_SIZE]);
            nc.rxq_size = static_cast<uint16_t>(nics_list_y[i][YAML_PARAM_NICS_RXQ_SIZE]);
            nc.gpu = static_cast<int>(nics_list_y[i][YAML_PARAM_NICS_GPU]);
            phydriver_config.nics_list.push_back(nc);
        }

        try
        {
            auto cus_port_failover = static_cast<bool>(root[YAML_PARAM_CUS_PORT_FAILOVER].as<int>());
            AppConfig::getInstance().setCUSPortFailover(cus_port_failover);
        }
        catch (const std::exception &e)
        {
            NVLOGW_FMT(TAG, "{} Using default value of false to YAML_PARAM_CUS_PORT_FAILOVER", e.what());
            AppConfig::getInstance().setCUSPortFailover(false);
        }

        try 
        {
            phydriver_config.forcedNumCsi2Bits = static_cast<uint16_t>(root[YAML_PARAM_PUSCH_FORCE_NUM_CSI2_BITS]);

        } 
        catch (const std::exception& e) 
        {
            NVLOGW_FMT(TAG, "{} Using default value of 0 to PUSCH-FORCE-NUM-CSI2-BITS", e.what());
            phydriver_config.forcedNumCsi2Bits = 0;
        }
        
        try
        {
            phydriver_config.pusch_nMaxLdpcHetConfigs = static_cast<uint32_t>(root[YAML_PARAM_PUSCH_N_MAX_LDPC_HET_CONFIGS]);

        } 
        catch (const std::exception& e) 
        {
            NVLOGW_FMT(TAG, "{} Using default value of 32 to PUSCH-N-MAX-LDPC-HET-CONFIGS", e.what());
            phydriver_config.pusch_nMaxLdpcHetConfigs = 32;
        }
        try
        {
            uint8_t temp_value = static_cast<uint8_t>(root[YAML_PARAM_PUSCH_N_MAX_TB_PER_NODE]);
            if (temp_value > CUPHY_LDPC_DECODE_DESC_MAX_TB) {
                NVLOGW_FMT(TAG, "pusch_nMaxTbPerNode value {} exceeds maximum {}, using maximum", temp_value, CUPHY_LDPC_DECODE_DESC_MAX_TB);
                phydriver_config.pusch_nMaxTbPerNode = CUPHY_LDPC_DECODE_DESC_MAX_TB;
            } else if (temp_value == 0) {
                NVLOGW_FMT(TAG, "pusch_nMaxTbPerNode cannot be zero, using default value 32");
                phydriver_config.pusch_nMaxTbPerNode = 32;
            } else {
                phydriver_config.pusch_nMaxTbPerNode = temp_value;
            }
        }
        catch (const std::exception& e)
        {
            NVLOGW_FMT(TAG, "{} Using default value of 32 to PUSCH-N-MAX-TB-PER-NODE", e.what());
            phydriver_config.pusch_nMaxTbPerNode = 32;
        }

        try
        {
            phydriver_config.notify_ul_harq_buffer_release = static_cast<uint8_t>(root[YAML_PARAM_NOTIFY_UL_HARQ_BUFFER_RELEASE]);
        }
        catch (const std::exception& e)
        {
            NVLOGW_FMT(TAG, "{} Using default value of 0 (not enabled) to NOTIFY_UL_HARQ_BUFFER_RELEASE", e.what());
            phydriver_config.notify_ul_harq_buffer_release = 0;
        }

        // Validate that cell_group_num is not greater than the number of cells in the YAML file
        if (root.has_key(YAML_PARAM_CELLS))
        {
            yaml::node yaml_cells = root[YAML_PARAM_CELLS];
            if (yaml_cells.length() < phydriver_config.cell_group_num)
            {
                NVLOGE_FMT(TAG, AERIAL_CONFIG_EVENT, 
                    "Configuration error: cell_group_num ({}) exceeds the number of cells defined in the YAML file ({}). "
                    "Please ensure that 'cell_group_num' is less than or equal to the number of cells specified under the '{}' key.",
                    phydriver_config.cell_group_num, yaml_cells.length(), YAML_PARAM_CELLS);
                return -1;
            }
        }

        try
        {
            phydriver_config.max_harq_tx_count_bundled = static_cast<uint16_t>(root[YAML_PARAM_MAX_HARQ_TX_COUNT_BUNDLED]);
        }
        catch(const std::exception& e)
        {
            NVLOGW_FMT(TAG, "{} Using default value of 10 to YAML_PARAM_MAX_HARQ_TX_COUNT_BUNDLED", e.what());
            phydriver_config.max_harq_tx_count_bundled = 10;
        }

        try
        {
            phydriver_config.max_harq_tx_count_non_bundled = static_cast<uint16_t>(root[YAML_PARAM_MAX_HARQ_TX_COUNT_NON_BUNDLED]);
        }
        catch(const std::exception& e)
        {
            NVLOGW_FMT(TAG, "{} Using default value of 4 to YAML_PARAM_MAX_HARQ_TX_COUNT_NON_BUNDLED", e.what());
            phydriver_config.max_harq_tx_count_non_bundled = 4;
        }
    }
    catch (const std::exception& e)
    {
        std::cout << e.what() << '\n';
        return -1;
    }
    return 0;
}

//!
// \brief Prints the populated configurations
//
//
void YamlParser::print_configs() const
{
    // NVLOGC_FMT(TAG, "Standalone mode: {}", (phydriver_config.standalone == 1 ? Yes : No));
    NVLOGC_FMT(TAG, "Num Slots: {}", phydriver_config.num_slots);
    NVLOGC_FMT(TAG, "Enable UL cuPHY Graphs: {}", phydriver_config.enable_ul_cuphy_graphs);
    NVLOGC_FMT(TAG, "Enable DL cuPHY Graphs: {}", phydriver_config.enable_dl_cuphy_graphs);
    NVLOGC_FMT(TAG, "Accurate TX scheduling clock resolution (ns): {}", phydriver_config.accu_tx_sched_res_ns);
    NVLOGC_FMT(TAG, "DPDK core: {}", phydriver_config.dpdk_thread);
    NVLOGC_FMT(TAG, "Prometheus core: {}", phydriver_config.prometheus_thread);
    NVLOGC_FMT(TAG, "UL cores: ");
    for(int i = 0; i < phydriver_config.workers_list_ul.size(); ++i)
        NVLOGC_FMT(TAG, "\t- {}", phydriver_config.workers_list_ul[i]);

    NVLOGC_FMT(TAG, "DL cores: ");
    for(int i = 0; i < phydriver_config.workers_list_dl.size(); ++i)
        NVLOGC_FMT(TAG, "\t- {}", phydriver_config.workers_list_dl[i]);
    NVLOGC_FMT(TAG, "Debug worker: {}", phydriver_config.debug_worker);
    NVLOGC_FMT(TAG, "Data Lake core: {}", phydriver_config.data_core);

    NVLOGC_FMT(TAG, "SRS starting Section ID: {}", phydriver_config.start_section_id_srs);
    NVLOGC_FMT(TAG, "PRACH starting Section ID: {}", phydriver_config.start_section_id_prach);

    NVLOGC_FMT(TAG, "USE GREEN CONTEXTS: {}", phydriver_config.use_green_contexts);
    NVLOGC_FMT(TAG, "USE GC WORKQUEUES: {}", phydriver_config.use_gc_workqueues);
    NVLOGC_FMT(TAG, "USE BATCHED MEMCPY: {}", phydriver_config.use_batched_memcpy);
    NVLOGC_FMT(TAG, "MPS SM PUSCH: {}", phydriver_config.mps_sm_pusch);
    NVLOGC_FMT(TAG, "MPS SM PUCCH: {}", phydriver_config.mps_sm_pucch);
    NVLOGC_FMT(TAG, "MPS SM PRACH: {}", phydriver_config.mps_sm_prach);
    NVLOGC_FMT(TAG, "MPS SM UL ORDER: {}", phydriver_config.mps_sm_ul_order);
    NVLOGC_FMT(TAG, "MPS SM PDSCH: {}", phydriver_config.mps_sm_pdsch);
    NVLOGC_FMT(TAG, "MPS SM PDCCH: {}", phydriver_config.mps_sm_pdcch);
    NVLOGC_FMT(TAG, "MPS SM PBCH: {}", phydriver_config.mps_sm_pbch);
    NVLOGC_FMT(TAG, "MPS SM GPU_COMMS: {}", phydriver_config.mps_sm_gpu_comms);
    NVLOGC_FMT(TAG, "MPS SM SRS: {}", phydriver_config.mps_sm_srs);
    NVLOGC_FMT(TAG, "UL Order Kernel Mode: {}", phydriver_config.ul_order_kernel_mode);
    NVLOGC_FMT(TAG, "PDSCH fallback: {}", phydriver_config.pdsch_fallback);

    NVLOGC_FMT(TAG, "Massive MIMO enable: {}", phydriver_config.mMIMO_enable);

    NVLOGC_FMT(TAG, "mMIMO_enable feature {}", phydriver_config.mMIMO_enable);

    NVLOGC_FMT(TAG, "Enable SRS : {}", phydriver_config.enable_srs);

    NVLOGC_FMT(TAG, "ul_order_timeout_gpu_log_enable: {}", phydriver_config.ul_order_timeout_gpu_log_enable);
    NVLOGC_FMT(TAG, "ue_mode: {}", phydriver_config.ue_mode);
    NVLOGC_FMT(TAG, "Aggr Obj Non-availability threshold: {}", phydriver_config.aggr_obj_non_avail_th);
    NVLOGC_FMT(TAG, "sendCPlane_timing_error_th_ns: {}", phydriver_config.sendCPlane_timing_error_th_ns);
    NVLOGC_FMT(TAG, "pusch_aggr_per_ctx: {}", phydriver_config.pusch_aggr_per_ctx);
    NVLOGC_FMT(TAG, "prach_aggr_per_ctx: {}", phydriver_config.prach_aggr_per_ctx);
    NVLOGC_FMT(TAG, "pucch_aggr_per_ctx: {}", phydriver_config.pucch_aggr_per_ctx);
    NVLOGC_FMT(TAG, "srs_aggr_per_ctx: {}", phydriver_config.srs_aggr_per_ctx);
    NVLOGC_FMT(TAG, "max_harq_pools: {}", phydriver_config.max_harq_pools);
    NVLOGC_FMT(TAG, "max_harq_tx_count_bundled: {}", phydriver_config.max_harq_tx_count_bundled);
    NVLOGC_FMT(TAG, "max_harq_tx_count_non_bundled: {}", phydriver_config.max_harq_tx_count_non_bundled);
    NVLOGC_FMT(TAG, "ul_input_buffer_per_cell: {}", phydriver_config.ul_input_buffer_per_cell);
    NVLOGC_FMT(TAG, "ul_input_buffer_per_cell_srs: {}", phydriver_config.ul_input_buffer_per_cell_srs);
    NVLOGC_FMT(TAG, "max_ru_unhealthy_ul_slots: {}", phydriver_config.max_ru_unhealthy_ul_slots);
    NVLOGC_FMT(TAG, "srs_chest_algo_type: {}", phydriver_config.srs_chest_algo_type);
    NVLOGC_FMT(TAG, "srs_chest_tol2_normalization_algo_type: {}", phydriver_config.srs_chest_tol2_normalization_algo_type);
    NVLOGC_FMT(TAG, "srs_chest_tol2_constant_scaler: {}", phydriver_config.srs_chest_tol2_constant_scaler);
    NVLOGC_FMT(TAG, "bfw_power_normalization_alg_selector: {}", phydriver_config.bfw_power_normalization_alg_selector);
    NVLOGC_FMT(TAG, "bfw_beta_prescaler: {}", phydriver_config.bfw_beta_prescaler);
    NVLOGC_FMT(TAG, "total_num_srs_chest_buffers: {}", phydriver_config.total_num_srs_chest_buffers);
    NVLOGC_FMT(TAG, "send_static_bfw_wt_all_cplane: {}", phydriver_config.send_static_bfw_wt_all_cplane);
    NVLOGC_FMT(TAG, "ul_pcap_capture_enable: {}", phydriver_config.ul_pcap_capture_enable);
    NVLOGC_FMT(TAG, "ul_pcap_capture_thread_cpu_affinity: {}", phydriver_config.ul_pcap_capture_thread_cpu_affinity);
    NVLOGC_FMT(TAG, "ul_pcap_capture_thread_sched_priority: {}", phydriver_config.ul_pcap_capture_thread_sched_priority);
    
    NVLOGC_FMT(TAG, "pcap_logger_ul_cplane_enable: {}", phydriver_config.pcap_logger_ul_cplane_enable);
    NVLOGC_FMT(TAG, "pcap_logger_dl_cplane_enable: {}", phydriver_config.pcap_logger_dl_cplane_enable);
    NVLOGC_FMT(TAG, "pcap_logger_thread_cpu_affinity: {}", phydriver_config.pcap_logger_thread_cpu_affinity);
    NVLOGC_FMT(TAG, "pcap_logger_thread_sched_prio: {}", phydriver_config.pcap_logger_thread_sched_prio);
    NVLOGC_FMT(TAG, "pcap_logger_file_save_dir: {}", phydriver_config.pcap_logger_file_save_dir);

    NVLOGC_FMT(TAG, "static_beam_id_start: {}", phydriver_config.static_beam_id_start);
    NVLOGC_FMT(TAG, "static_beam_id_end: {}", phydriver_config.static_beam_id_end);
    NVLOGC_FMT(TAG, "dynamic_beam_id_start: {}", phydriver_config.dynamic_beam_id_start);
    NVLOGC_FMT(TAG, "dynamic_beam_id_end: {}", phydriver_config.dynamic_beam_id_end);

    NVLOGC_FMT(TAG, "ul_order_timeout_gpu_log_enable: {}", phydriver_config.ul_order_timeout_gpu_log_enable);

    NVLOGC_FMT(TAG, "pusch_workCancelMode: {}", phydriver_config.pusch_workCancelMode);

    NVLOGC_FMT(TAG, "GPU-initiated comms DL: {}", phydriver_config.gpu_init_comms_dl);
    NVLOGC_FMT(TAG, "GPU-initiated comms (via CPU): {}", phydriver_config.gpu_init_comms_via_cpu);
    NVLOGC_FMT(TAG, "CPU-initiated comms : {}", phydriver_config.cpu_init_comms);
    NVLOGC_FMT(TAG, "Cell group: {}", phydriver_config.cell_group);
    NVLOGC_FMT(TAG, "Cell group num: {}", phydriver_config.cell_group_num);
    NVLOGC_FMT(TAG, "puxchPolarDcdrListSz: {}", phydriver_config.puxchPolarDcdrListSz);
    NVLOGC_FMT(TAG, "split_ul_cuda_streams: {}", phydriver_config.split_ul_cuda_streams);
    NVLOGC_FMT(TAG, "serialize_pucch_pusch: {}", phydriver_config.serialize_pucch_pusch);
    NVLOGC_FMT(TAG, "bfw_c_plane_chaining_mode: {}", phydriver_config.bfw_c_plane_chaining_mode);

    // Added missing yaml parameters. May want to reorganize the order in which they are printed
    NVLOGC_FMT(TAG, "fh_stats_dump_cpu_core: {}", phydriver_config.fh_stats_dump_cpu_core);
    NVLOGC_FMT(TAG, "fix_beta_dl: {}", phydriver_config.fix_beta_dl);
    NVLOGC_FMT(TAG, "pdump_client_thread: {}", phydriver_config.pdump_client_thread);
    NVLOGC_FMT(TAG, "profiler_sec: {}", phydriver_config.profiler_sec);
    //NVLOGC_FMT(TAG, "log_level: {}", phydriver_config.log_level); // determine how this should be displayed
    NVLOGC_FMT(TAG, "datalake_address: {}", phydriver_config.datalake_address);
    NVLOGC_FMT(TAG, "dpdk_file_prefix: {}", phydriver_config.dpdk_file_prefix);
    NVLOGC_FMT(TAG, "puschrxChestFactorySettingsFilename: {}", phydriver_config.puschrxChestFactorySettingsFilename);
    //NVLOGC_FMT(TAG, "nics_list: {}", phydriver_config.nics_list); // determine how this should be displayed
    //NVLOGC_FMT(TAG, "workers_dl_validation: {}", phydriver_config.workers_dl_validation); // determine how this should be displayed
    //NVLOGC_FMT(TAG, "gpus_list: {}", phydriver_config.gpus_list); // determine how this should be displayed
    //NVLOGC_FMT(TAG, "dl_wait_th_list: {}", phydriver_config.dl_wait_th_list); // determine how this should be displayed
    //NVLOGC_FMT(TAG, "h2d_cpy_th_cfg: {}", phydriver_config.h2d_cpy_th_cfg); // determine how this should be displayed
    NVLOGC_FMT(TAG, "accu_tx_sched_disable: {}", phydriver_config.accu_tx_sched_disable);
    NVLOGC_FMT(TAG, "cplane_disable: {}", phydriver_config.cplane_disable);
    NVLOGC_FMT(TAG, "disable_empw: {}", phydriver_config.disable_empw);
    NVLOGC_FMT(TAG, "dlc_alloc_cplane_bfw_txq: {}", phydriver_config.dlc_alloc_cplane_bfw_txq);
    NVLOGC_FMT(TAG, "dlc_bfw_enable_divide_per_cell: {}", phydriver_config.dlc_bfw_enable_divide_per_cell);
    NVLOGC_FMT(TAG, "dpdk_verbose_logs: {}", phydriver_config.dpdk_verbose_logs);
    NVLOGC_FMT(TAG, "enable_cpu_task_tracing: {}", phydriver_config.enable_cpu_task_tracing);
    NVLOGC_FMT(TAG, "enable_dl_cqe_tracing: {}", phydriver_config.enable_dl_cqe_tracing);
    NVLOGC_FMT(TAG, "enable_l1_param_sanity_check: {}", phydriver_config.enable_l1_param_sanity_check);
    NVLOGC_FMT(TAG, "enable_ok_tb: {}", phydriver_config.enable_ok_tb);
    NVLOGC_FMT(TAG, "enable_prepare_tracing: {}", phydriver_config.enable_prepare_tracing);
    NVLOGC_FMT(TAG, "cupti_enable_tracing: {}", phydriver_config.cupti_enable_tracing);
    NVLOGC_FMT(TAG, "cupti_buffer_size: {}", phydriver_config.cupti_buffer_size);
    NVLOGC_FMT(TAG, "cupti_num_buffers: {}", phydriver_config.cupti_num_buffers);
    NVLOGC_FMT(TAG, "mCh_segment_proc_enable: {}", phydriver_config.mCh_segment_proc_enable);
    NVLOGC_FMT(TAG, "pmu_metrics: {}", phydriver_config.pmu_metrics);
    NVLOGC_FMT(TAG, "puschCfo: {}", phydriver_config.puschCfo);
    NVLOGC_FMT(TAG, "puschDftSOfdm: {}", phydriver_config.puschDftSOfdm);
    NVLOGC_FMT(TAG, "puschEnablePerPrgChEst: {}", phydriver_config.puschEnablePerPrgChEst);
    NVLOGC_FMT(TAG, "puschRssi: {}", phydriver_config.puschRssi);
    NVLOGC_FMT(TAG, "puschSelectChEstAlgo: {}", phydriver_config.puschSelectChEstAlgo);
    NVLOGC_FMT(TAG, "puschSelectEqCoeffAlgo: {}", phydriver_config.puschSelectEqCoeffAlgo);
    NVLOGC_FMT(TAG, "puschSinr: {}", phydriver_config.puschSinr);
    NVLOGC_FMT(TAG, "puschTbSizeCheck: {}", phydriver_config.puschTbSizeCheck);
    NVLOGC_FMT(TAG, "puschTdi: {}", phydriver_config.puschTdi);
    NVLOGC_FMT(TAG, "puschTo: {}", phydriver_config.puschTo);
    NVLOGC_FMT(TAG, "pusch_deviceGraphLaunchEn: {}", phydriver_config.pusch_deviceGraphLaunchEn);
    NVLOGC_FMT(TAG, "ulc_alloc_cplane_bfw_txq: {}", phydriver_config.ulc_alloc_cplane_bfw_txq);
    NVLOGC_FMT(TAG, "ulc_bfw_enable_divide_per_cell: {}", phydriver_config.ulc_bfw_enable_divide_per_cell);
    NVLOGC_FMT(TAG, "ul_rx_pkt_tracing_level: {}", phydriver_config.ul_rx_pkt_tracing_level);
    NVLOGC_FMT(TAG, "ul_rx_pkt_tracing_level_srs: {}", phydriver_config.ul_rx_pkt_tracing_level_srs);
    NVLOGC_FMT(TAG, "ul_warmup_frame_count: {}", phydriver_config.ul_warmup_frame_count);
    NVLOGC_FMT(TAG, "forcedNumCsi2Bits: {}", phydriver_config.forcedNumCsi2Bits);
    NVLOGC_FMT(TAG, "pusch_waitTimeOutPostEarlyHarqUs: {}", phydriver_config.pusch_waitTimeOutPostEarlyHarqUs);
    NVLOGC_FMT(TAG, "pusch_waitTimeOutPreEarlyHarqUs: {}", phydriver_config.pusch_waitTimeOutPreEarlyHarqUs);
    NVLOGC_FMT(TAG, "validation: {}", phydriver_config.validation);
    NVLOGC_FMT(TAG, "cqe_trace_slot_mask: {}", phydriver_config.cqe_trace_slot_mask);
    NVLOGC_FMT(TAG, "datalake_samples: {}", phydriver_config.datalake_samples);
    NVLOGC_FMT(TAG, "num_ok_tb_slot: {}", phydriver_config.num_ok_tb_slot);
    NVLOGC_FMT(TAG, "pusch_nMaxLdpcHetConfigs: {}", phydriver_config.pusch_nMaxLdpcHetConfigs);
    NVLOGC_FMT(TAG, "pusch_nMaxTbPerNode: {}", phydriver_config.pusch_nMaxTbPerNode);
    NVLOGC_FMT(TAG, "sendCPlane_dlbfw_backoff_th_ns: {}", phydriver_config.sendCPlane_dlbfw_backoff_th_ns);
    NVLOGC_FMT(TAG, "sendCPlane_ulbfw_backoff_th_ns: {}", phydriver_config.sendCPlane_ulbfw_backoff_th_ns);
    NVLOGC_FMT(TAG, "ul_order_max_rx_pkts: {}", phydriver_config.ul_order_max_rx_pkts);
    NVLOGC_FMT(TAG, "ul_order_rx_pkts_timeout_ns: {}", phydriver_config.ul_order_rx_pkts_timeout_ns);
    NVLOGC_FMT(TAG, "ul_order_timeout_cpu_ns: {}", phydriver_config.ul_order_timeout_cpu_ns);
    NVLOGC_FMT(TAG, "ul_order_timeout_gpu_ns: {}", phydriver_config.ul_order_timeout_gpu_ns);
    NVLOGC_FMT(TAG, "ul_order_timeout_gpu_srs_ns: {}", phydriver_config.ul_order_timeout_gpu_srs_ns);
    NVLOGC_FMT(TAG, "ul_order_timeout_log_interval_ns: {}", phydriver_config.ul_order_timeout_log_interval_ns);
    NVLOGC_FMT(TAG, "ul_srs_aggr3_task_launch_offset_ns: {}", phydriver_config.ul_srs_aggr3_task_launch_offset_ns);
    NVLOGC_FMT(TAG, "workers_sched_priority: {}", phydriver_config.workers_sched_priority);
    NVLOGC_FMT(TAG, "cqe_trace_cell_mask: {}", phydriver_config.cqe_trace_cell_mask);
    // --

    NVLOGC_FMT(TAG, "Number of Cell Configs: {}", cell_configs.size());
    NVLOGC_FMT(TAG, "L2Adapter config file: {}", l2adapter_config_filename.c_str());
    for(const auto& cell_config: cell_configs)
    {
        NVLOGC_FMT(TAG, "Cell name: {}", cell_config.name.c_str());
        NVLOGC_FMT(TAG, "\tMU: {}", cell_config.phy_stat.mu);
        NVLOGC_FMT(TAG, "\tID: {}", cell_config.mplane_id);
    }

    NVLOGC_FMT(TAG, "Number of MPlane Configs: {}", mplane_configs.size());
    for(const auto& mplane_config: mplane_configs)
    {
        NVLOGC_FMT(TAG, "\tMplane ID: {}", mplane_config.mplane_id);
        NVLOGC_FMT(TAG, "\tVLAN ID: {}", mplane_config.vlan_tci & 0xfff);
        NVLOGC_FMT(TAG, "\tSource Eth Address: {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}",
            mplane_config.src_eth_addr[0],mplane_config.src_eth_addr[1],
            mplane_config.src_eth_addr[2],mplane_config.src_eth_addr[3],
            mplane_config.src_eth_addr[4],mplane_config.src_eth_addr[5]);
        NVLOGC_FMT(TAG, "\tDestination Eth Address: {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}",
            mplane_config.dst_eth_addr[0],mplane_config.dst_eth_addr[1],
            mplane_config.dst_eth_addr[2],mplane_config.dst_eth_addr[3],
            mplane_config.dst_eth_addr[4],mplane_config.dst_eth_addr[5]);
        NVLOGC_FMT(TAG, "\tNIC port: {}", mplane_config.nic_name.c_str());
        NVLOGC_FMT(TAG, "\tRU Type: {}", +mplane_config.ru);
        NVLOGC_FMT(TAG, "\tU-plane TXQs: {}", mplane_config.nic_cfg.txq_count_uplane);

        NVLOGC_FMT(TAG, "\tDL compression method: {}", (int)mplane_config.dl_comp_meth);
        NVLOGC_FMT(TAG, "\tDL iq bit width: {}", mplane_config.dl_bit_width);
        NVLOGC_FMT(TAG, "\tUL compression method: {}", (int)mplane_config.ul_comp_meth);
        NVLOGC_FMT(TAG, "\tUL iq bit width: {}", mplane_config.ul_bit_width);
        NVLOGC_FMT(TAG, "");

        NVLOGC_FMT(TAG, "\tFlow list SSB/PBCH: ");
        for(auto eAxC_id : mplane_config.eAxC_ids[slot_command_api::channel_type::PBCH])
            NVLOGC_FMT(TAG, "\t\t{}", eAxC_id);
        NVLOGC_FMT(TAG, "\tFlow list PDCCH: ");
        for(auto eAxC_id : mplane_config.eAxC_ids[slot_command_api::channel_type::PDCCH_DL])
            NVLOGC_FMT(TAG, "\t\t{}", eAxC_id);
        NVLOGC_FMT(TAG, "\tFlow list PDSCH: ");
        for(auto eAxC_id : mplane_config.eAxC_ids[slot_command_api::channel_type::PDSCH])
            NVLOGC_FMT(TAG, "\t\t{}", eAxC_id);
        NVLOGC_FMT(TAG, "\tFlow list CSIRS: ");
        for(auto eAxC_id : mplane_config.eAxC_ids[slot_command_api::channel_type::CSI_RS])
            NVLOGC_FMT(TAG, "\t\t{}", eAxC_id);
        NVLOGC_FMT(TAG, "\tFlow list PUSCH: ");
        for(auto eAxC_id : mplane_config.eAxC_ids[slot_command_api::channel_type::PUSCH])
            NVLOGC_FMT(TAG, "\t\t{}", eAxC_id);
        NVLOGC_FMT(TAG, "\tFlow list PUCCH: ");
        for(auto eAxC_id : mplane_config.eAxC_ids[slot_command_api::channel_type::PUCCH])
            NVLOGC_FMT(TAG, "\t\t{}", eAxC_id);
        NVLOGC_FMT(TAG, "\tFlow list SRS: ");
        for(auto eAxC_id : mplane_config.eAxC_ids[slot_command_api::channel_type::SRS])
            NVLOGC_FMT(TAG, "\t\t{}", eAxC_id);
        NVLOGC_FMT(TAG, "\tFlow list PRACH: ");
        for(auto eAxC_id : mplane_config.eAxC_ids[slot_command_api::channel_type::PRACH])
            NVLOGC_FMT(TAG, "\t\t{}", eAxC_id);

        if(!mplane_config.tv_pusch_h5.empty())
            NVLOGC_FMT(TAG, "\tPUSCH TV: {}", mplane_config.tv_pusch_h5.c_str());

        if(!mplane_config.tv_srs_h5.empty())
            NVLOGC_FMT(TAG, "\tSRS TV: {}", mplane_config.tv_srs_h5.c_str());

        NVLOGC_FMT(TAG, "\tSection_3 time offset: {}", mplane_config.section_3_time_offset);

        NVLOGC_FMT(TAG, "\tnMaxRxAnt: {}", mplane_config.nMaxRxAnt);
        NVLOGC_FMT(TAG, "\tPUSCH PRBs Stride: {}", mplane_config.pusch_prb_stride);
        NVLOGC_FMT(TAG, "\tPRACH PRBs Stride: {}", mplane_config.prach_prb_stride);
        NVLOGC_FMT(TAG, "\tSRS PRBs Stride: {}", mplane_config.srs_prb_stride);
        NVLOGC_FMT(TAG, "\tPUSCH nMaxPrb: {}", mplane_config.pusch_nMaxPrb);
        NVLOGC_FMT(TAG, "\tPUSCH nMaxRx: {}", mplane_config.pusch_nMaxRx);
        NVLOGC_FMT(TAG, "\tUL Gain Calibration: {}", mplane_config.ul_gain_calibration);
        NVLOGC_FMT(TAG, "\tLower guard bw: {}",  mplane_config.lower_guard_bw);
    }

}

std::vector<uint8_t>& YamlParser::get_cuphydriver_workers_ul() {
    return phydriver_config.workers_list_ul;
}

std::vector<uint8_t>& YamlParser::get_cuphydriver_workers_dl() {
    return phydriver_config.workers_list_dl;
}

std::vector<uint8_t>& YamlParser::get_cuphydriver_workers_dl_validation() {
    return phydriver_config.workers_dl_validation;
}

int16_t YamlParser::get_cuphydriver_debug_worker() {
    return phydriver_config.debug_worker;
}

int16_t YamlParser::get_cuphydriver_data_core() {
    return phydriver_config.data_core;
}
uint8_t YamlParser::get_cuphydriver_datalake_db_write_enable() {
    return phydriver_config.datalake_db_write_enable;
}
std::vector<std::string>& YamlParser::get_cuphydriver_datalake_data_types() {
    return phydriver_config.datalake_data_types;
}
std::string& YamlParser::get_cuphydriver_datalake_address() {
    return phydriver_config.datalake_address;
}
uint32_t YamlParser::get_cuphydriver_datalake_samples() {
    return phydriver_config.datalake_samples;
}
std::string& YamlParser::get_cuphydriver_datalake_engine() {
    return phydriver_config.datalake_engine;
}
uint8_t YamlParser::get_cuphydriver_datalake_store_failed_pdu() {
    return phydriver_config.datalake_store_failed_pdu;
}
uint32_t YamlParser::get_cuphydriver_num_rows_fh() {
    return phydriver_config.num_rows_fh;
}
uint32_t YamlParser::get_cuphydriver_num_rows_pusch() {
    return phydriver_config.num_rows_pusch;
}

uint32_t YamlParser::get_cuphydriver_num_rows_hest() {
    return phydriver_config.num_rows_hest;
}

uint8_t YamlParser::get_cuphydriver_e3_agent_enabled() {
    return phydriver_config.e3_agent_enabled;
}

uint16_t YamlParser::get_cuphydriver_e3_rep_port() {
    return phydriver_config.e3_rep_port;
}

uint16_t YamlParser::get_cuphydriver_e3_pub_port() {
    return phydriver_config.e3_pub_port;
}

uint16_t YamlParser::get_cuphydriver_e3_sub_port() {
    return phydriver_config.e3_sub_port;
}
uint8_t YamlParser::get_cuphydriver_datalake_drop_tables() {
    return phydriver_config.datalake_drop_tables;
}


std::vector<uint16_t>& YamlParser::get_cuphydriver_gpus() {
    return phydriver_config.gpus_list;
}

l1_log_level& YamlParser::get_cuphydriver_loglevel() {
    return phydriver_config.log_level;
}

uint16_t& YamlParser::get_cuphydriver_validation() {
    return phydriver_config.validation;
}

uint16_t& YamlParser::get_cuphydriver_standalone() {
    return phydriver_config.standalone;
}

int& YamlParser::get_cuphydriver_profiler_sec() {
    return phydriver_config.profiler_sec;
}

uint16_t& YamlParser::get_cuphydriver_slots() {
    return phydriver_config.num_slots;
}

uint8_t& YamlParser::get_cuphydriver_ul_cuphy_graphs() {
    return phydriver_config.enable_ul_cuphy_graphs;
}

uint8_t& YamlParser::get_cuphydriver_dl_cuphy_graphs() {
    return phydriver_config.enable_dl_cuphy_graphs;
}

int& YamlParser::get_cuphydriver_prometheusthread() {
    return phydriver_config.prometheus_thread;
}

uint32_t& YamlParser::get_cuphydriver_dpdk_thread() {
    return phydriver_config.dpdk_thread;
}

uint8_t& YamlParser::get_cuphydriver_dpdk_verbose_logs() {
    return phydriver_config.dpdk_verbose_logs;
}

uint32_t& YamlParser::get_cuphydriver_accu_tx_sched_res_ns() {
    return phydriver_config.accu_tx_sched_res_ns;
}

uint8_t& YamlParser::get_cuphydriver_accu_tx_sched_disable() {
    return phydriver_config.accu_tx_sched_disable;
}

int& YamlParser::get_cuphydriver_fh_stats_dump_cpu_core() {
    return phydriver_config.fh_stats_dump_cpu_core;
}

int& YamlParser::get_cuphydriver_pdump_client_thread() {
    return phydriver_config.pdump_client_thread;
}

std::string& YamlParser::get_cuphydriver_dpdk_file_prefix() {
    return phydriver_config.dpdk_file_prefix;
}

uint32_t& YamlParser::get_cuphydriver_workers_sched_priority() {
    return phydriver_config.workers_sched_priority;
}

std::vector<struct nic_config> YamlParser::get_cuphydriver_nics() {
    return phydriver_config.nics_list;
}

uint16_t& YamlParser::get_cuphydriver_start_section_id_srs() {
    return phydriver_config.start_section_id_srs;
}

uint16_t& YamlParser::get_cuphydriver_start_section_id_prach() {
    return phydriver_config.start_section_id_prach;
}

uint32_t& YamlParser::get_cuphydriver_timeout_cpu() {
    return phydriver_config.ul_order_timeout_cpu_ns;
}

uint32_t& YamlParser::get_cuphydriver_timeout_gpu() {
    return phydriver_config.ul_order_timeout_gpu_ns;
}

uint32_t& YamlParser::get_cuphydriver_timeout_gpu_srs() {
    return phydriver_config.ul_order_timeout_gpu_srs_ns;
}

uint32_t& YamlParser::get_cuphydriver_ul_srs_aggr3_task_launch_offset_ns() {
    return phydriver_config.ul_srs_aggr3_task_launch_offset_ns;
}

uint32_t& YamlParser::get_cuphydriver_timeout_log_interval() {
    return phydriver_config.ul_order_timeout_log_interval_ns;
}

uint8_t& YamlParser::get_cuphydriver_ul_order_kernel_mode() {
    return phydriver_config.ul_order_kernel_mode;
}

uint8_t& YamlParser::get_cuphydriver_timeout_gpu_log_enable() {
    return phydriver_config.ul_order_timeout_gpu_log_enable;
}

uint8_t& YamlParser::get_cuphydriver_ue_mode() {
    return phydriver_config.ue_mode;
}

uint32_t& YamlParser::get_cuphydriver_order_kernel_max_rx_pkts() {
    return phydriver_config.ul_order_max_rx_pkts;
}

uint32_t& YamlParser::get_cuphydriver_order_kernel_rx_pkts_timeout() {
    return phydriver_config.ul_order_rx_pkts_timeout_ns;
}

uint8_t& YamlParser::get_cplane_disable()
{
    return phydriver_config.cplane_disable;
}

uint8_t& YamlParser::get_cuphydriver_use_green_contexts() {
    return phydriver_config.use_green_contexts;
}

uint8_t& YamlParser::get_cuphydriver_use_gc_workqueues() {
    return phydriver_config.use_gc_workqueues;
}

uint8_t& YamlParser::get_cuphydriver_use_batched_memcpy() {
    return phydriver_config.use_batched_memcpy;
}

uint32_t& YamlParser::get_cuphydriver_mps_sm_pusch() {
    return phydriver_config.mps_sm_pusch;
}
uint32_t& YamlParser::get_cuphydriver_mps_sm_pucch() {
    return phydriver_config.mps_sm_pucch;
}
uint32_t& YamlParser::get_cuphydriver_mps_sm_prach() {
    return phydriver_config.mps_sm_prach;
}
uint32_t& YamlParser::get_cuphydriver_mps_sm_ul_order() {
    return phydriver_config.mps_sm_ul_order;
}
uint32_t& YamlParser::get_cuphydriver_mps_sm_srs() {
    return phydriver_config.mps_sm_srs;
}
uint32_t& YamlParser::get_cuphydriver_mps_sm_pdsch() {
    return phydriver_config.mps_sm_pdsch;
}
uint32_t& YamlParser::get_cuphydriver_mps_sm_pdcch() {
    return phydriver_config.mps_sm_pdcch;
}
uint32_t& YamlParser::get_cuphydriver_mps_sm_pbch() {
    return phydriver_config.mps_sm_pbch;
}
uint32_t& YamlParser::get_cuphydriver_mps_sm_gpu_comms() {
    return phydriver_config.mps_sm_gpu_comms;
}

uint8_t& YamlParser::get_cuphydriver_pdsch_fallback() {
    return phydriver_config.pdsch_fallback;
}

uint8_t& YamlParser::get_cuphydriver_gpu_init_comms_dl() {
    return phydriver_config.gpu_init_comms_dl;
}

uint8_t& YamlParser::get_cuphydriver_gpu_init_comms_via_cpu() {
    return phydriver_config.gpu_init_comms_via_cpu;
}

uint8_t& YamlParser::get_cuphydriver_cpu_init_comms() {
    return phydriver_config.cpu_init_comms;
}

uint8_t& YamlParser::get_cuphydriver_cell_group() {
    return phydriver_config.cell_group;
}
uint8_t& YamlParser::get_cuphydriver_cell_group_num() {
    return phydriver_config.cell_group_num;
}

uint16_t YamlParser::get_cuphydriver_total_uplane_txq_count() const {
    uint16_t total = 0;
    for (const auto& m : mplane_configs) {
        size_t peer_count = m.destinations.empty() ? 1u : m.destinations.size();
        total += static_cast<uint16_t>(peer_count * m.nic_cfg.txq_count_uplane);
    }
    return total;
}

uint16_t YamlParser::get_cuphydriver_total_peer_count() const {
    uint16_t total = 0;
    for (const auto& m : mplane_configs) {
        total += static_cast<uint16_t>(m.destinations.empty() ? 1u : m.destinations.size());
    }
    return total;
}

uint8_t YamlParser::get_cuphydriver_pusch_workCancelMode()const {
    return phydriver_config.pusch_workCancelMode;
}

uint8_t YamlParser::get_cuphydriver_pusch_tdi()const {
    return phydriver_config.puschTdi;
}

uint8_t YamlParser::get_cuphydriver_pusch_cfo()const {
    return phydriver_config.puschCfo;
}

uint8_t YamlParser::get_cuphydriver_pusch_dftsofdm()const {
    return phydriver_config.puschDftSOfdm;
}

uint8_t YamlParser::get_cuphydriver_pusch_tbsizecheck()const {
    return phydriver_config.puschTbSizeCheck;
}

uint8_t YamlParser::get_cuphydriver_pusch_deviceGraphLaunchEn()const {
    return phydriver_config.pusch_deviceGraphLaunchEn;
}

uint16_t YamlParser::get_cuphydriver_pusch_waitTimeOutPreEarlyHarqUs()const {
    return phydriver_config.pusch_waitTimeOutPreEarlyHarqUs;
}

uint16_t YamlParser::get_cuphydriver_pusch_waitTimeOutPostEarlyHarqUs()const {
    return phydriver_config.pusch_waitTimeOutPostEarlyHarqUs;
}

uint8_t YamlParser::get_cuphydriver_pusch_to()const {
    return phydriver_config.puschTo;
}

uint8_t YamlParser::get_cuphydriver_pusch_select_eqcoeffalgo()const {
    return phydriver_config.puschSelectEqCoeffAlgo;
}

uint8_t YamlParser::get_cuphydriver_pusch_select_chestalgo()const {
    return phydriver_config.puschSelectChEstAlgo;
}

uint8_t YamlParser::get_cuphydriver_pusch_enable_perprgchest()const {
    return phydriver_config.puschEnablePerPrgChEst;
}

uint8_t YamlParser::get_cuphydriver_pusch_rssi()const {
    return phydriver_config.puschRssi;
}

uint8_t YamlParser::get_cuphydriver_pusch_sinr()const {
    return phydriver_config.puschSinr;
}

uint8_t YamlParser::get_cuphydriver_pusch_weighted_average_cfo()const {
    return phydriver_config.puschWeightedAverageCfo;
}

const std::string& YamlParser::get_cuphydriver_puschrx_chest_factory_settings_filename() const noexcept {
    return phydriver_config.puschrxChestFactorySettingsFilename;
}

uint8_t YamlParser::get_cuphydriver_puxchPolarDcdrListSz()const {
    return phydriver_config.puxchPolarDcdrListSz;
}

uint8_t YamlParser::get_cuphydriver_fix_beta_dl()const {
    return phydriver_config.fix_beta_dl;
}

uint8_t YamlParser::get_cuphydriver_enable_cpu_task_tracing()const {
    return phydriver_config.enable_cpu_task_tracing;
}

uint8_t YamlParser::get_cuphydriver_enable_l1_param_sanity_check()const {
    return phydriver_config.enable_l1_param_sanity_check;
}


uint8_t YamlParser::get_cuphydriver_enable_prepare_tracing()const {
    return phydriver_config.enable_prepare_tracing;
}

uint8_t YamlParser::get_cuphydriver_cupti_enable_tracing()const {
    return phydriver_config.cupti_enable_tracing;
}

uint64_t YamlParser::get_cuphydriver_cupti_buffer_size()const {
    return phydriver_config.cupti_buffer_size;
}

uint16_t YamlParser::get_cuphydriver_cupti_num_buffers()const {
    return phydriver_config.cupti_num_buffers;
}

uint8_t YamlParser::get_cuphydriver_disable_empw()const {
    return phydriver_config.disable_empw;
}

uint8_t YamlParser::get_cuphydriver_enable_dl_cqe_tracing()const {
    return phydriver_config.enable_dl_cqe_tracing;
}

uint64_t YamlParser::get_cuphydriver_cqe_trace_cell_mask()const {
    return phydriver_config.cqe_trace_cell_mask;
}

uint32_t YamlParser::get_cuphydriver_cqe_trace_slot_mask()const {
    return phydriver_config.cqe_trace_slot_mask;
}

uint8_t YamlParser::get_cuphydriver_enable_ok_tb()const {
    return phydriver_config.enable_ok_tb;
}

uint32_t YamlParser::get_cuphydriver_num_ok_tb_slot()const {
    return phydriver_config.num_ok_tb_slot;
}

uint8_t YamlParser::get_cuphydriver_ul_rx_pkt_tracing_level()const {
    return phydriver_config.ul_rx_pkt_tracing_level;
}

uint8_t YamlParser::get_cuphydriver_ul_rx_pkt_tracing_level_srs()const {
    return phydriver_config.ul_rx_pkt_tracing_level_srs;
}

uint32_t YamlParser::get_cuphydriver_ul_warmup_frame_count() const {
    return phydriver_config.ul_warmup_frame_count;
}

uint8_t YamlParser::get_cuphydriver_pmu_metrics()const {
    return phydriver_config.pmu_metrics;
}


struct h2d_copy_thread_config YamlParser::get_cuphydriver_h2d_cpy_th_cfg()const {
    return phydriver_config.h2d_cpy_th_cfg;
}

uint8_t& YamlParser::get_cuphydriver_mMIMO_enable() {
    return phydriver_config.mMIMO_enable;
}

uint8_t& YamlParser::get_cuphydriver_enable_srs() {
    return phydriver_config.enable_srs;
}

uint8_t& YamlParser::get_cuphydriver_enable_dl_core_affinity() {
    return phydriver_config.enable_dl_core_affinity;
}

uint8_t& YamlParser::get_cuphydriver_dlc_core_packing_scheme() {
    return phydriver_config.dlc_core_packing_scheme;
}

uint32_t& YamlParser::get_cuphydriver_aggr_obj_non_avail_th() {
    return phydriver_config.aggr_obj_non_avail_th;
}

uint32_t& YamlParser::get_cuphydriver_sendCPlane_timing_error_th_ns() {
    return phydriver_config.sendCPlane_timing_error_th_ns;
}

uint32_t& YamlParser::get_cuphydriver_sendCPlane_ulbfw_backoff_th_ns() {
    return phydriver_config.sendCPlane_ulbfw_backoff_th_ns;
}

uint32_t& YamlParser::get_cuphydriver_sendCPlane_dlbfw_backoff_th_ns() {
    return phydriver_config.sendCPlane_dlbfw_backoff_th_ns;
}

uint8_t YamlParser::get_cuphydriver_split_ul_cuda_streams() {
    return phydriver_config.split_ul_cuda_streams;
}

uint8_t YamlParser::get_cuphydriver_serialize_pucch_pusch() {
    return phydriver_config.serialize_pucch_pusch;
}

std::vector<uint32_t>& YamlParser::get_cuphydriver_dl_wait_th() {
    return phydriver_config.dl_wait_th_list;
}

uint16_t YamlParser::get_cuphydriver_forcedNumCsi2Bits() {
    return phydriver_config.forcedNumCsi2Bits;
}

uint32_t YamlParser::get_cuphydriver_pusch_nMaxLdpcHetConfigs() {
    return phydriver_config.pusch_nMaxLdpcHetConfigs;
}

uint8_t YamlParser::get_cuphydriver_pusch_nMaxTbPerNode() {
    return phydriver_config.pusch_nMaxTbPerNode;
}

uint8_t& YamlParser::get_cuphydriver_ch_segment_proc_enable() {
    return phydriver_config.mCh_segment_proc_enable;
}

uint8_t& YamlParser::get_pusch_aggr_per_ctx() {
    return phydriver_config.pusch_aggr_per_ctx;
}

uint8_t& YamlParser::get_prach_aggr_per_ctx() {
    return phydriver_config.prach_aggr_per_ctx;
}

uint8_t& YamlParser::get_pucch_aggr_per_ctx() {
    return phydriver_config.pucch_aggr_per_ctx;
}

uint8_t& YamlParser::get_srs_aggr_per_ctx() {
    return phydriver_config.srs_aggr_per_ctx;
}

uint16_t& YamlParser::get_max_harq_pools() {
    return phydriver_config.max_harq_pools;
}

uint8_t& YamlParser::get_ul_input_buffer_per_cell() {
    return phydriver_config.ul_input_buffer_per_cell;
}

uint8_t& YamlParser::get_ul_input_buffer_per_cell_srs() {
    return phydriver_config.ul_input_buffer_per_cell_srs;
}

uint32_t& YamlParser::get_max_ru_unhealthy_ul_slots() {
    return phydriver_config.max_ru_unhealthy_ul_slots;
}

uint8_t& YamlParser::get_srs_chest_algo_type() {
    return phydriver_config.srs_chest_algo_type;
}

uint8_t& YamlParser::get_srs_chest_tol2_normalization_algo_type() {
    return phydriver_config.srs_chest_tol2_normalization_algo_type;
}

float& YamlParser::get_srs_chest_tol2_constant_scaler() {
    return phydriver_config.srs_chest_tol2_constant_scaler;
}

uint8_t& YamlParser::get_bfw_power_normalization_alg_selector() {
    return phydriver_config.bfw_power_normalization_alg_selector;
}

float& YamlParser::get_bfw_beta_prescaler() {
    return phydriver_config.bfw_beta_prescaler;
}

uint32_t& YamlParser::get_total_num_srs_chest_buffers() {
    return phydriver_config.total_num_srs_chest_buffers;
}

uint8_t& YamlParser::get_ul_pcap_capture_enable() {
    return phydriver_config.ul_pcap_capture_enable;
}

uint8_t& YamlParser::get_pcap_logger_ul_cplane_enable() {
    return phydriver_config.pcap_logger_ul_cplane_enable;
}

uint8_t& YamlParser::get_pcap_logger_dl_cplane_enable() {
    return phydriver_config.pcap_logger_dl_cplane_enable;
}

uint8_t& YamlParser::get_pcap_logger_thread_cpu_affinity() {
    return phydriver_config.pcap_logger_thread_cpu_affinity; 
}

uint8_t& YamlParser::get_pcap_logger_thread_sched_prio() {
    return phydriver_config.pcap_logger_thread_sched_prio; 
}

std::string& YamlParser::get_pcap_logger_file_save_dir() {
    return phydriver_config.pcap_logger_file_save_dir;
}

uint16_t YamlParser::get_static_beam_id_start() {
    return phydriver_config.static_beam_id_start;
}

uint16_t YamlParser::get_static_beam_id_end() {
    return phydriver_config.static_beam_id_end;
}

uint16_t YamlParser::get_dynamic_beam_id_start() {
    return phydriver_config.dynamic_beam_id_start;
}

uint16_t YamlParser::get_dynamic_beam_id_end() {
    return phydriver_config.dynamic_beam_id_end;
}

uint8_t& YamlParser::get_ul_pcap_capture_thread_cpu_affinity() {
    return phydriver_config.ul_pcap_capture_thread_cpu_affinity;
}

uint8_t& YamlParser::get_ul_pcap_capture_thread_sched_priority() {
    return phydriver_config.ul_pcap_capture_thread_sched_priority;
}

uint8_t& YamlParser::get_send_static_bfw_wt_all_cplane() {
    return phydriver_config.send_static_bfw_wt_all_cplane;
}

uint8_t YamlParser::get_dlc_bfw_enable_divide_per_cell() {
    return phydriver_config.dlc_bfw_enable_divide_per_cell;
}

uint8_t YamlParser::get_ulc_bfw_enable_divide_per_cell() {
    return phydriver_config.ulc_bfw_enable_divide_per_cell;
}

uint8_t YamlParser::get_dlc_alloc_cplane_bfw_txq() {
    return phydriver_config.dlc_alloc_cplane_bfw_txq;
}

uint8_t YamlParser::get_ulc_alloc_cplane_bfw_txq() {
    return phydriver_config.ulc_alloc_cplane_bfw_txq;
}

uint8_t YamlParser::get_enable_tx_notification() {
    return phydriver_config.enable_tx_notification;
}

int YamlParser::parse_launch_pattern_file(const char* filename)
{
    int cell_index, slot_num, pusch;

    try {

        yaml::file_parser fp(filename);
        yaml::document doc = fp.next_document();
        yaml::node root = doc.root();
        yaml::node slots_node = root[YAML_LP_SLOTS];

        NVLOGC_FMT(TAG, "Slots node length: {}", slots_node.length());
        for(int slot_idx = 0; slot_idx < slots_node.length(); ++slot_idx)
        {
            slot_command_list.push_back(std::unique_ptr<struct slot_command_api::slot_command>(new slot_command_api::slot_command));

            yaml::node cells_node_ul = slots_node[slot_idx][YAML_LP_PUSCH][YAML_LP_CELLS];
            for(int cell_idx = 0; cell_idx < cells_node_ul.length(); ++cell_idx)
            {
                slot_command_api::cell_sub_command csc_1;
                csc_1.cell = static_cast<int>(cells_node_ul[cell_idx]);
                csc_1.slot.type = slot_command_api::SLOT_UPLINK;
                csc_1.slot.slot_3gpp.sfn_ = 0;
                csc_1.slot.slot_3gpp.slot_ = 0;
                csc_1.create_if(slot_command_api::channel_type::PUSCH);
                slot_command_list[slot_idx]->cells.push_back(std::move(csc_1));
            }

            yaml::node cells_node_dl = slots_node[slot_idx][YAML_LP_PDSCH][YAML_LP_CELLS];
            for(int cell_idx = 0; cell_idx < cells_node_dl.length(); ++cell_idx)
            {
                slot_command_api::cell_sub_command csc_2;
                csc_2.cell = static_cast<int>(cells_node_dl[cell_idx]);
                csc_2.slot.type = slot_command_api::SLOT_DOWNLINK;
                csc_2.slot.slot_3gpp.sfn_ = 0;
                csc_2.slot.slot_3gpp.slot_ = 0;
                csc_2.create_if(slot_command_api::channel_type::PDSCH);

                // struct slot_command_api::channel * ch = new struct slot_command_api::channel[1];
                // ch->type = slot_command_api::PDSCH;
                // csc_2.channels.push_back(ch[0]);

                slot_command_list[slot_idx]->cells.push_back(std::move(csc_2));
            }

            yaml::node cells_node_pbch = slots_node[slot_idx][YAML_LP_PBCH][YAML_LP_CELLS];
            for(int cell_idx = 0; cell_idx < cells_node_pbch.length(); ++cell_idx)
            {
                slot_command_api::cell_sub_command csc_3;
                csc_3.cell = static_cast<int>(cells_node_pbch[cell_idx]);
                csc_3.slot.type = slot_command_api::SLOT_DOWNLINK;
                csc_3.slot.slot_3gpp.sfn_ = 0;
                csc_3.slot.slot_3gpp.slot_ = 0;
                csc_3.create_if(slot_command_api::channel_type::PBCH);

                // struct slot_command_api::channel * ch = new struct slot_command_api::channel[1];
                // ch->type = slot_command_api::PDSCH;
                // csc_2.channels.push_back(ch[0]);

                slot_command_list[slot_idx]->cells.push_back(std::move(csc_3));
            }

        }

    }
    catch (std::exception& e)
    {
        std::cout << e.what() << '\n';
        return -1;
    }

    return 0;
}

size_t YamlParser::get_cuphydriver_standalone_slot_command_size() {
    return slot_command_list.size();
}

struct slot_command_api::slot_command * YamlParser::get_cuphydriver_standalone_slot_command(int slot_num) {
    if(slot_num > slot_command_list.size())
        return nullptr;

    return slot_command_list[slot_num].get();
}

uint8_t YamlParser::get_bfw_c_plane_chaining_mode() {
    return phydriver_config.bfw_c_plane_chaining_mode;
}

uint16_t& YamlParser::get_max_harq_tx_count_bundled() {
    return phydriver_config.max_harq_tx_count_bundled;
}

uint16_t& YamlParser::get_max_harq_tx_count_non_bundled() {
    return phydriver_config.max_harq_tx_count_non_bundled;
}

uint8_t YamlParser::get_notify_ul_harq_buffer_release() const {
    return phydriver_config.notify_ul_harq_buffer_release;
}
