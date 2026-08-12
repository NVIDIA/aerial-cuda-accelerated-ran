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
# versions.sh - Centralized version definitions for NVIDIA Aerial install scripts
#
# This file defines all software versions used by the install scripts.
# Versions can be customized per platform by setting PLATFORM before sourcing.
#
# Usage:
#   source includes.sh   # sets PLATFORM_ID, then sources this file
#   export PLATFORM="YourPlatformName" # optional override; defaults to PLATFORM_ID from includes.sh
#

# Idempotent when sourced: skip if already loaded (e.g. script sources includes.sh which sources this)
if [[ "${BASH_SOURCE[0]:-$0}" != "${0}" ]] && [[ -n "${VERSIONS_SH_LOADED:-}" ]]; then
    return 0
fi
VERSIONS_SH_LOADED=1

SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]:-$0}")")"
# When run directly, PLATFORM_ID is not set; compute it from DMI so we can show_versions
if [[ -z "${PLATFORM_ID:-}" ]]; then
    DMI_PATH="${DMI_PATH:-/sys/devices/virtual/dmi/id}"
    _read_dmi() { cat "$DMI_PATH/$1" 2>/dev/null | tr ' ' '_' | tr -s '_' || echo "unknown"; }
    _vb="$(_read_dmi board_vendor)_$(_read_dmi product_family)_$(_read_dmi product_name)_$(_read_dmi board_name)"
    case "$_vb" in
        #There are multiple variants of the Supermicro ARS-111GL-NHR server. Map them to the same platform ID.
        Supermicro_Family_Super_Server_G1SMH-G)  PLATFORM_ID="Supermicro_ARS-111GL-NHR" ;;
        Supermicro_Family_ARS-111GL-NHR_G1SMH-G) PLATFORM_ID="Supermicro_ARS-111GL-NHR" ;;
        NVIDIA_DGX_Spark_NVIDIA_DGX_Spark_P4242) PLATFORM_ID="NVIDIA_DGX_Spark_P4242" ;;
        *) PLATFORM_ID="$_vb" ;;
    esac
    VERBOSE_PLATFORM_ID="$_vb"
fi
# Default platform if not specified (PLATFORM_ID from includes.sh when sourced, or from DMI above when run directly)
PLATFORM="${PLATFORM:-$PLATFORM_ID}"

# =============================================================================
# Platform-specific versions
# =============================================================================

case "$PLATFORM" in
    "NVIDIA_DGX_Spark_P4242")
        # Kernel
        KERNEL_VERSION="6.17.0-1014-nvidia"

        # DOCA/OFED
        DOCA_VERSION="3.2.1"
        DOCA_BUILD="044000-25.10"
        UBUNTU_VERSION="ubuntu2404"
        ARCH="arm64"

        # Docker
        DOCKER_VERSION="29.1.5"

        # GPU Driver
        GPU_DRIVER_VERSION="590.48.01"
        GPU_DRIVER_ARCH="aarch64"
        CUDA_VERSION="13.1.1"
        CUDA_RUN_FILE_NAME="cuda_${CUDA_VERSION}_${GPU_DRIVER_VERSION}_linux_sbsa.run"

        # GDRCopy (GPU Direct RDMA)
        GDRDRV_VERSION="2.5.1-1"
        GDRDRV_CUDA_VERSION="13.0"
        GDRCOPY_UBUNTU_VER="ubuntu24_04"

        HUGEPAGES="24"
        LINUXPTP_VERSION="4.2"           # available via apt on Ubuntu 24.04

        NIC_DEV="/dev/mst/mt4129_pciconf0"
        NIC_FW_VERSION="28.47.1088"       # Used for post-reboot verification after mlnx-fw-updater
        ISOLCPUS="4-19"

        # PTP services are unpinned by default. Set PTP_CPU_AFFINITY to pin them explicitly.
        PTP_CPU_AFFINITY="${PTP_CPU_AFFINITY:-}"

        # build_aerial_sdk.sh args
        AERIAL_BUILD_FLAGS="${AERIAL_BUILD_FLAGS:- --cuda-archs 121}"
        ;;

    "Supermicro_ARS-111GL-NHR")
        # Kernel
        KERNEL_VERSION="6.8.0-1025-nvidia-64k"

        # DOCA/OFED
        DOCA_VERSION="3.2.1"
        DOCA_BUILD="044000-25.10"
        UBUNTU_VERSION="ubuntu2204"          # Ubuntu 22.04
        ARCH="arm64"

        # Docker
        DOCKER_VERSION="29.1.5"

        # GPU Driver
        GPU_DRIVER_VERSION="590.48.01"
        GPU_DRIVER_ARCH="aarch64"
        CUDA_VERSION="13.1.1"
        CUDA_RUN_FILE_NAME="cuda_${CUDA_VERSION}_${GPU_DRIVER_VERSION}_linux_sbsa.run"

        # GDRCopy (GPU Direct RDMA)
        GDRDRV_VERSION="2.5-1"
        GDRDRV_CUDA_VERSION="12.8"
        GDRCOPY_UBUNTU_VER="ubuntu22_04"

        HUGEPAGES="48"                       # 48 × 512M = 24GB (GH200)
        LINUXPTP_VERSION="4.2"           # must build from source on Ubuntu 22.04
        BFB_VERSION="3.2.1-34_25.11-prod"   # BlueField3 BFB bundle version
        NIC_FW_VERSION="32.47.1088"       # Used for post-BFB-install verification on both BF3 devices
        BFB_RSHIM_NUMS="0 1"
        NIC_FW_DEVICES="/dev/mst/mt41692_pciconf0 /dev/mst/mt41692_pciconf1"

        NIC_DEV="/dev/mst/mt41692_pciconf0"
        ISOLCPUS="4-64"
        # PTP services: unpinned for SMC (empty). Set PTP_CPU_AFFINITY= to pin if desired.
        PTP_CPU_AFFINITY="${PTP_CPU_AFFINITY:-}"

        # build_aerial_sdk.sh args
        AERIAL_BUILD_FLAGS="${AERIAL_BUILD_FLAGS:- --cuda-archs 90}"
        ;;

    # Add other platforms as needed
    # example_platform)
    #     KERNEL_VERSION="x.x.x"
    #     ...
    #     ;;

    *)
        echo "[ERROR] Unknown platform DMI ID: $PLATFORM"
        echo "[ERROR] Supported platforms and their DMI board_vendor_family_name_board_name:"
        echo "        NVIDIA_DGX_Spark_P4242:"
        echo "           NVIDIA_DGX_Spark_NVIDIA_DGX_Spark_P4242"
        echo "        Supermicro_ARS-111GL-NHR:"
        echo "           Supermicro_Family_Super_Server_G1SMH-G"
        echo "           Supermicro_Family_ARS-111GL-NHR_G1SMH-G"
        exit 1
        ;;
esac

# =============================================================================
# Common versions (tools used across platforms)
# =============================================================================

YQ_VERSION="${YQ_VERSION:-4.50.1}"
# ARCH for binary downloads (e.g. yq); set per-platform above or default from dpkg
ARCH="${ARCH:-$(dpkg --print-architecture 2>/dev/null)}"

# =============================================================================
# Derived values (computed from versions above)
# =============================================================================

# NVIDIA developer download base (keep constant; versioned paths appended)
NVIDIA_CUDA_BASE_URL="https://developer.download.nvidia.com/compute/cuda/"

# DOCA package filename and URL
DOCA_DEB="doca-host_${DOCA_VERSION}-${DOCA_BUILD}-${UBUNTU_VERSION}_${ARCH}.deb"
DOCA_URL="https://www.mellanox.com/downloads/DOCA/DOCA_v${DOCA_VERSION}/host/${DOCA_DEB}"

# ARM: download CUDA run file (CUDA_RUN_FILE_*), extract it, then run GPU_DRIVER_FILE from inside
GPU_DRIVER_FILE="NVIDIA-Linux-${GPU_DRIVER_ARCH}-${GPU_DRIVER_VERSION}.run"
GPU_DRIVER_URL="${NVIDIA_CUDA_BASE_URL}${CUDA_VERSION}/local_installers/${CUDA_RUN_FILE_NAME}"
GPU_DRIVER_DOWNLOAD_FILE="${CUDA_RUN_FILE_NAME}"

# GDRCopy driver filename and URL
# Derive label: "ubuntu24_04" -> "Ubuntu24_04"
_gdrcopy_label="${GDRCOPY_UBUNTU_VER^}"
GDRDRV_FILE="gdrdrv-dkms_${GDRDRV_VERSION}_${ARCH}.${_gdrcopy_label}.deb"
GDRDRV_URL="https://developer.download.nvidia.com/compute/redist/gdrcopy/CUDA%20${GDRDRV_CUDA_VERSION}/${GDRCOPY_UBUNTU_VER}/${GPU_DRIVER_ARCH}/${GDRDRV_FILE}"

# BlueField3 BFB firmware bundle (only set on Supermicro GH200)
BFB_FILE="${BFB_VERSION:+bf-fwbundle-${BFB_VERSION}.bfb}"

# Source Aerial container setup for version info
CUPHY_CP_SETUP="${SCRIPT_DIR}/../../cuPHY-CP/container/setup.sh"
AERIAL_VERSION_TAG="${AERIAL_VERSION_TAG:-26-1-cubb}"
if [[ -f "$CUPHY_CP_SETUP" ]]; then
    source "$CUPHY_CP_SETUP"
else
    # Fallback defaults if setup.sh not found
    AERIAL_REPO="${AERIAL_REPO:-gitlab-master.nvidia.com:5005/gputelecom/container/}"
    AERIAL_IMAGE_NAME="${AERIAL_IMAGE_NAME:-aerial_build_devel}"
fi
export AERIAL_VERSION_TAG



# =============================================================================
# Display versions (for debugging)
# =============================================================================

show_versions() {
    echo "Platform ID: $PLATFORM"
    echo "Platform DMI ID: $VERBOSE_PLATFORM_ID"
    echo ""
    echo "Versions:"
    echo "  Kernel:       $KERNEL_VERSION"
    echo "  ISOLCPUS:     ${ISOLCPUS:-not set}"
    echo "  PTP CPU aff:  ${PTP_CPU_AFFINITY:-<unpinned>}"
    echo "  Hugepages:    ${HUGEPAGES:-not set}"
    echo "  DOCA:         $DOCA_VERSION"
    echo "  DOCA Build:   $DOCA_BUILD"
    echo "  Ubuntu:       $UBUNTU_VERSION"
    echo "  Arch:         $ARCH"
    echo "  Docker:       $DOCKER_VERSION"
    echo "  GPU Driver:   $GPU_DRIVER_VERSION"
    echo "  GDRCopy:      $GDRDRV_VERSION (CUDA $GDRDRV_CUDA_VERSION)"
    echo "  linuxptp:     $LINUXPTP_VERSION"
    echo "  NIC FW:       ${NIC_FW_VERSION:-not set}"
    echo "  Aerial build: ${AERIAL_BUILD_FLAGS:-<none>}"
    echo ""
    echo "Aerial Container:"
    echo "  Version Tag:  AERIAL_VERSION_TAG=$AERIAL_VERSION_TAG"
    echo "  Repository:   AERIAL_REPO=$AERIAL_REPO"
    echo "  Image Name:   AERIAL_IMAGE_NAME=$AERIAL_IMAGE_NAME"
    echo "  Platform:     ${AERIAL_PLATFORM:-not set}"
    echo ""
    echo "Derived:"
    echo "  DOCA DEB:     $DOCA_DEB"
    echo "  DOCA URL:     $DOCA_URL"
    echo "  GPU File:     $GPU_DRIVER_FILE"
    echo "  GPU URL:      $GPU_DRIVER_URL"
    echo "  CUDA Run:     $CUDA_RUN_FILE_NAME (extract then run $GPU_DRIVER_FILE)"
    echo "  Download as:  $GPU_DRIVER_DOWNLOAD_FILE"
    echo "  GDRDRV File:  $GDRDRV_FILE"
    echo "  GDRDRV URL:   $GDRDRV_URL"
    echo "  BFB File:     ${BFB_FILE:-not set (non-Supermicro platform)}"
}

# If run directly (not sourced), show versions
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    show_versions
fi
