#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#
# install_nic_fw.sh - Verify and configure NIC firmware for NVIDIA Aerial
#
# On DGX Spark, this script verifies the ConnectX-7 firmware installed by
# mlnx-fw-updater and configures the documented NIC settings. On Supermicro
# GH200, it installs the BlueField3 BFB when needed and configures the
# documented BF3 settings.
#
# Prerequisites: DOCA/OFED must already be installed (provides flint, mlxconfig,
# and bfb-install on GH200).
#
# Usage: ./install_nic_fw.sh [--dry-run] [--verbose] [--check] [--rshim=N] [--help]
#
#   --dry-run        Show commands without executing
#   --verbose        Print commands before executing
#   --check          Check current FW version only; exit 0 if up-to-date, 1 if update needed
#   --rshim=N        GH only: update /dev/rshimN (default: all documented BF3 devices)
#   -h, --help       Show this help message
#

_SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
[[ -f "$_SCRIPT_DIR/includes.sh" ]] && source "$_SCRIPT_DIR/includes.sh" || { echo "ERROR: includes.sh not found: $_SCRIPT_DIR/includes.sh" >&2; exit 1; }
[[ -f "$_SCRIPT_DIR/versions.sh" ]] && source "$_SCRIPT_DIR/versions.sh" || { echo "ERROR: versions.sh not found: $_SCRIPT_DIR/versions.sh" >&2; exit 1; }

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Verify and configure NIC firmware for NVIDIA Aerial"
    echo ""
    echo "Options:"
    echo "  --dry-run        Show commands without executing"
    echo "  --verbose        Print commands before executing"
    echo "  --check          Check FW version only (exit 0 if current, 1 if update needed)"
    echo "  --rshim=N        GH only: update /dev/rshimN (default: ${BFB_RSHIM_NUMS:-not applicable})"
    echo "  -h, --help       Show this help message"
    echo ""
    echo "Expected NIC FW: ${NIC_FW_VERSION:-not set}"
    if [[ -n ${BFB_FILE:-} ]]; then
        echo "Expected BFB:    ${BFB_FILE}"
        echo "BFB URL:         https://content.mellanox.com/BlueField/FW-Bundle/${BFB_FILE}"
    fi
    exit "${1:-0}"
}

# Parse common arguments (--dry-run, --verbose) and populate REMAINING_ARGS
parse_common_args "$@"
verify_secure_boot_disabled

CHECK_ONLY=0
RSHIM_NUM=""
NIC_CONFIG_UPDATED=0

set -- "${REMAINING_ARGS[@]}"
while [[ $# -gt 0 ]]; do
    case $1 in
        --check) CHECK_ONLY=1; shift ;;
        --rshim=*) RSHIM_NUM="${1#*=}"; shift ;;
        --rshim) RSHIM_NUM="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1" >&2; usage 1 ;;
    esac
done

declare -a NIC_FW_CHECK_DEVICES
case "$PLATFORM" in
    "NVIDIA_DGX_Spark_P4242")
        NIC_FW_CHECK_DEVICES=("$NIC_DEV")
        ;;
    "Supermicro_ARS-111GL-NHR")
        if [[ -n $RSHIM_NUM ]]; then
            [[ $RSHIM_NUM =~ ^[0-9]+$ ]] || { echo "Invalid rshim number: $RSHIM_NUM" >&2; exit 1; }
            RSHIM_NUMS="$RSHIM_NUM"
            NIC_FW_CHECK_DEVICES=("/dev/mst/mt41692_pciconf${RSHIM_NUM}")
        else
            RSHIM_NUMS="${BFB_RSHIM_NUMS:-0}"
            read -r -a NIC_FW_CHECK_DEVICES <<< "${NIC_FW_DEVICES:-$NIC_DEV}"
        fi
        BFB_URL="https://content.mellanox.com/BlueField/FW-Bundle/${BFB_FILE}"
        ;;
    *)
        echo_and_log "[INFO] NIC firmware management is not supported on platform '$PLATFORM' — skipped"
        exit 0
        ;;
esac

start_mst() {
    echo_and_log "[INFO] Starting MST (Mellanox Software Tools) driver set..."
    execute_or_die sudo mst start
}

# Query the current NIC firmware version with the documented flint command.
# Returns the running FW version string, or empty if unavailable.
get_current_fw_version() {
    local nic_dev="$1"
    if [[ $DRYRUN -eq 1 ]]; then
        echo_and_log "[DRY-RUN] sudo flint -d ${nic_dev} q (would report current NIC FW)"
        echo ""
        return
    fi

    local fw_output
    fw_output=$(sudo flint -d "$nic_dev" q 2>/dev/null | awk -F: '/^[[:space:]]*FW Version:/ {gsub(/^[[:space:]]+/, "", $2); print $2; exit}') || true
    echo "$fw_output"
}

# Check whether every selected NIC matches the expected firmware version.
# Returns 0 if up-to-date, 1 if update is needed.
check_nic_firmware_versions() {
    echo_and_log "[INFO] Checking NIC firmware version..."
    echo_and_log "[INFO] Expected NIC FW version: ${NIC_FW_VERSION}"

    local nic_dev current_fw
    local update_needed=0
    for nic_dev in "${NIC_FW_CHECK_DEVICES[@]}"; do
        current_fw=$(get_current_fw_version "$nic_dev")

        if [[ -z $current_fw ]]; then
            if [[ $DRYRUN -eq 1 ]]; then
                echo_and_log "[DRY-RUN] Cannot determine current FW version on ${nic_dev} — assuming update needed"
            else
                echo_and_log "[WARN] Could not determine current NIC firmware version on ${nic_dev}"
            fi
            update_needed=1
            continue
        fi

        if [[ $current_fw == "$NIC_FW_VERSION" ]]; then
            echo_and_log "[INFO] ${nic_dev}: NIC firmware is up-to-date (${current_fw})"
        else
            echo_and_log "[INFO] ${nic_dev}: FW update required (current=${current_fw}, expected=${NIC_FW_VERSION})"
            update_needed=1
        fi
    done

    return $update_needed
}

# Verify that NIC_DEV (set in versions.sh per platform) matches the NIC present and exists.
check_nic_pciconf_device() {
    local mtdev_name current_dev
    mtdev_name=$(ibdev2netdev -v 2>/dev/null | head -1 | awk '{print $3}' | tr -d '(' | tr '[:upper:]' '[:lower:]') || {
        echo_and_log "[ERROR] Could not get MTDEV name from ibdev2netdev (is OFED/ibdev2netdev available?)" >&2
        return 1
    }
    current_dev="/dev/mst/${mtdev_name}_pciconf0"
    if [[ -e $current_dev && $current_dev == "$NIC_DEV" ]]; then
        echo_and_log "[INFO] Found expected device: $NIC_DEV"
        return 0
    fi
    if [[ $DRYRUN -eq 1 ]]; then
        echo_and_log "[DRY-RUN] Expected device $NIC_DEV not found (ibdev2netdev reports $current_dev)" >&2
        return 0
    fi
    echo_and_log "[ERROR] Expected device $NIC_DEV not found (ibdev2netdev reports $current_dev)" >&2
    return 1
}

# Configure NIC firmware features required for Aerial CUDA-Accelerated RAN.
configure_nic_firmware() {
    check_nic_pciconf_device || { FAILED=1; return; }

    local -a settings=(
        "FLEX_PARSER_PROFILE_ENABLE;4;4"
        "PROG_PARSE_GRAPH;1;(True|1)"
        "REAL_TIME_CLOCK_ENABLE;1;(True|1)"
        "ACCURATE_TX_SCHEDULER;1;(True|1)"
        "CQE_COMPRESSION;1;(AGGRESSIVE|1)"
    )
    if [[ $PLATFORM == "Supermicro_ARS-111GL-NHR" ]]; then
        settings+=(
            "LINK_TYPE_P1;2;(ETH|2)"
            "LINK_TYPE_P2;2;(ETH|2)"
            "INTERNAL_CPU_MODEL;1;(EMBEDDED_CPU|1)"
            "INTERNAL_CPU_PAGE_SUPPLIER;EXT_HOST_PF;(EXT_HOST_PF|1)"
            "INTERNAL_CPU_ESWITCH_MANAGER;EXT_HOST_PF;(EXT_HOST_PF|1)"
            "INTERNAL_CPU_IB_VPORT0;EXT_HOST_PF;(EXT_HOST_PF|1)"
            "INTERNAL_CPU_OFFLOAD_ENGINE;DISABLED;(DISABLED|1)"
            "EXP_ROM_VIRTIO_NET_PXE_ENABLE;0;(False|0)"
            "EXP_ROM_VIRTIO_NET_UEFI_ARM_ENABLE;0;(False|0)"
            "EXP_ROM_VIRTIO_NET_UEFI_x86_ENABLE;0;(False|0)"
            "EXP_ROM_VIRTIO_BLK_UEFI_ARM_ENABLE;0;(False|0)"
            "EXP_ROM_VIRTIO_BLK_UEFI_x86_ENABLE;0;(False|0)"
        )
    fi

    local query_pattern=""
    local setting name value expected
    for setting in "${settings[@]}"; do
        IFS=';' read -r name value expected <<< "$setting"
        query_pattern="${query_pattern:+${query_pattern}|}${name}"
    done

    local needs_update=0
    local output=""
    if [[ $DRYRUN -ne 1 ]]; then
        output=$(sudo mlxconfig -d "$NIC_DEV" q | grep -E "$query_pattern")

        echo_and_log "[INFO] Current NIC firmware settings:"
        echo_and_log "$output"

        for setting in "${settings[@]}"; do
            IFS=';' read -r name value expected <<< "$setting"
            if ! echo "$output" | grep -qE "${name}.*${expected}"; then
                echo_and_log "[INFO] ${name} needs update (want: ${value})"
                needs_update=1
            fi
        done

        if [[ $needs_update -eq 0 ]]; then
            echo_and_log "[INFO] Skipping NIC config changes"
        fi
    else
        echo_and_log "[DRY-RUN] Would query documented NIC firmware settings on $NIC_DEV"
        needs_update=1
    fi

    if [[ $needs_update -eq 1 ]]; then
        NIC_CONFIG_UPDATED=1
        echo_and_log "[INFO] Updating NIC firmware settings..."
        for setting in "${settings[@]}"; do
            IFS=';' read -r name value expected <<< "$setting"
            execute "sudo mlxconfig -d $NIC_DEV --yes set ${name}=${value} > /dev/null"
        done
    fi

    if [[ $needs_update -eq 1 ]]; then
        if [[ $PLATFORM == "Supermicro_ARS-111GL-NHR" ]]; then
            echo_and_log "[IMPORTANT] A full system POWER CYCLE is required to apply the BF3 settings."
            echo_and_log "[IMPORTANT] A soft reboot or mlxfwreset is not sufficient."
            [[ $DRYRUN -eq 0 ]] && return 0
        else
            echo_and_log "[INFO] Resetting NIC to apply firmware changes."
            execute "sudo mlxfwreset -d $NIC_DEV --yes --level 3 r > /dev/null"
        fi
    else
        echo_and_log "[INFO] NIC settings already correct"
    fi

    echo_and_log "[INFO] Verifying NIC firmware parameters..."
    if [[ $DRYRUN -ne 1 ]]; then
        output=$(sudo mlxconfig -d "$NIC_DEV" q | grep -E "$query_pattern")
        echo_and_log "$output"

        for setting in "${settings[@]}"; do
            IFS=';' read -r name value expected <<< "$setting"
            if ! echo "$output" | grep -qE "${name}.*${expected}"; then
                echo_and_log "[WARN] ${name} is not set to ${value}"
                FAILED=1
            fi
        done

        if [[ $FAILED -ne 1 ]]; then
            echo_and_log "[INFO] All NIC firmware parameters configured successfully"
        fi
    fi
}

# Download BFB file
download_bfb() {
    if [[ -f "${BFB_FILE}" && -s "${BFB_FILE}" ]]; then
        echo_and_log "[INFO] Using existing ${BFB_FILE} ($(du -h "${BFB_FILE}" | cut -f1))"
    else
        echo_and_log "[INFO] Downloading BFB from ${BFB_URL}..."
        execute_or_die "wget -nv -O ${BFB_FILE} ${BFB_URL}"
    fi
}

# Install BFB via bfb-install
install_bfb() {
    local rshim_num="$1"
    local rshim_dev="/dev/rshim${rshim_num}"
    echo_and_log "[INFO] Installing BFB firmware via bfb-install on ${rshim_dev}..."

    # Start the rshim daemon if the device isn't present yet.
    # rshim (userspace) creates /dev/rshimN as a directory; it is started via systemd.
    if [[ ! -e "${rshim_dev}" ]]; then
        echo_and_log "[INFO] ${rshim_dev} not found — starting rshim service..."
        execute "sudo systemctl start rshim"
        sleep 3
    fi

    if [[ ! -e "${rshim_dev}" ]] && [[ $DRYRUN -eq 0 ]]; then
        echo_and_log "[ERROR] rshim device not found after starting rshim service: ${rshim_dev}"
        echo_and_log "[ERROR] Available rshim devices: $(ls /dev/rshim* 2>/dev/null || echo 'none')"
        echo_and_log "[ERROR] Check rshim service: sudo systemctl status rshim"
        exit 1
    fi

    execute_or_die "sudo bfb-install -r ${rshim_dev} -b ${BFB_FILE}"
}

# Wait for BlueField3 to come back online after BFB install by polling mst status
wait_for_bfb() {
    echo_and_log "[INFO] Waiting for BlueField3 to come back online (up to 120s)..."

    if [[ $DRYRUN -eq 1 ]]; then
        echo_and_log "[DRY-RUN] Would poll: sudo mst status | grep -q mt41692"
        return 0
    fi

    local elapsed=0
    local max_wait=120
    while [[ $elapsed -lt $max_wait ]]; do
        if sudo mst status 2>/dev/null | grep -q "mt41692"; then
            echo_and_log "[INFO] BlueField3 is back online after ${elapsed}s"
            return 0
        fi
        sleep 5
        ((elapsed += 5))
        echo_and_log "[INFO] Still waiting... (${elapsed}s elapsed)"
    done

    echo_and_log "[WARN] BlueField3 did not come back online within ${max_wait}s"
    echo_and_log "[WARN] A full power cycle may be required"
    return 1
}

# Provision BlueField3 firmware and settings on Supermicro GH200.
provision_gh_nic_firmware_and_settings() {
    echo "============================================"
    echo_and_log "BlueField3 BFB Firmware Installation"
    echo_and_log "Platform: ${PLATFORM}"
    echo_and_log "BFB file: ${BFB_FILE}"
    echo_and_log "rshim:    ${RSHIM_NUMS}"
    echo "============================================"
    echo ""

    if ! check_nic_firmware_versions; then
        download_bfb
        local rshim_num
        for rshim_num in $RSHIM_NUMS; do
            install_bfb "$rshim_num"
        done
        wait_for_bfb

        if [[ $DRYRUN -eq 0 ]]; then
            echo ""
            echo "============================================"
            echo_and_log "[IMPORTANT] The BFB was installed on rshim ${RSHIM_NUMS}."
            echo_and_log "[IMPORTANT] Perform a full system POWER CYCLE, then run make install again."
            echo_and_log "[IMPORTANT] A soft reboot is NOT sufficient."
            echo "============================================"
            exit 2
        fi
    else
        echo_and_log "[INFO] BlueField3 FW is already up-to-date on all selected devices."
    fi

    configure_nic_firmware
    [[ $FAILED -eq 1 ]] && return 1
    if [[ $NIC_CONFIG_UPDATED -eq 1 && $DRYRUN -eq 0 ]]; then
        echo ""
        echo "============================================"
        echo_and_log "[IMPORTANT] The documented BF3 settings were updated."
        echo_and_log "[IMPORTANT] Perform a full system POWER CYCLE, then run make install again."
        echo_and_log "[IMPORTANT] A soft reboot or mlxfwreset is NOT sufficient."
        echo "============================================"
        return 2
    fi

    echo ""
    echo "============================================"
    echo_and_log "[INFO] BF3 firmware and configuration match the release guide."
    echo "============================================"
}

# Verify and configure ConnectX-7 firmware on DGX Spark after the updater reboot.
verify_spark_nic_firmware_then_configure_settings() {
    echo "============================================"
    echo_and_log "DGX Spark NIC Firmware Verification"
    echo_and_log "Platform: ${PLATFORM}"
    echo_and_log "NIC device: ${NIC_DEV}"
    echo "============================================"
    echo ""

    if ! check_nic_firmware_versions; then
        if [[ $DRYRUN -eq 0 ]]; then
            echo_and_log "[ERROR] NIC firmware does not match the expected post-reboot version."
            return 1
        fi
    fi

    configure_nic_firmware
    [[ $FAILED -eq 1 ]] && return 1

    echo ""
    echo "============================================"
    echo_and_log "[INFO] DGX Spark NIC firmware and configuration match the release guide."
    echo "============================================"
}

# Main execution
main() {
    start_mst

    if [[ $CHECK_ONLY -eq 1 ]]; then
        check_nic_firmware_versions
        return $?
    fi

    case "$PLATFORM" in
        "NVIDIA_DGX_Spark_P4242") verify_spark_nic_firmware_then_configure_settings ;;
        "Supermicro_ARS-111GL-NHR") provision_gh_nic_firmware_and_settings ;;
    esac
}

main "$@"
