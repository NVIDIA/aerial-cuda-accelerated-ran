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
# install_drivers.sh - Install DOCA-Host and OFED drivers for NVIDIA Aerial
#
# This script installs:
#   - Build dependencies and tools
#   - DOCA-Host package
#   - DOCA OFED drivers
#   - Mellanox firmware updater
#   - NVIDIA GPU driver
#
# Usage: ./install_drivers.sh [--dry-run] [--verbose] [--uninstall]
#

# Source common functions and versions
_SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
[[ -f "$_SCRIPT_DIR/includes.sh" ]] && source "$_SCRIPT_DIR/includes.sh" || { echo "ERROR: includes.sh not found: $_SCRIPT_DIR/includes.sh" >&2; exit 1; }

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Install DOCA-Host, OFED drivers, and GPU driver for NVIDIA Aerial"
    echo ""
    echo "Options:"
    echo "  --dry-run               Show commands without executing"
    echo "  --verbose               Print commands before executing"
    echo "  --show-time             Show timing for each command (can also use SHOW_TIME=1)"
    echo "  --uninstall             Uninstall the GPU driver and exit"
    echo "  --docker                Install Docker and NVIDIA Container Toolkit only"
    echo "  --check-docker-login    Check Docker login only"
    echo "  --status                Show driver and interface status only"
    echo "  -h, --help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --dry-run --verbose       # Preview what will be done"
    echo "  SHOW_TIME=1 $0               # Show timing for each step"
    echo "  $0 --show-time --verbose     # Show commands and timing"
    exit "${1:-0}"
}

# Parse common arguments (--dry-run, --verbose) and populate REMAINING_ARGS
parse_common_args "$@"
verify_secure_boot_disabled

UNINSTALL_DRIVERS=0
DOCKER_ONLY=0
STATUS_ONLY=0
CHECK_DOCKER_LOGIN=0
FAILED=0

for arg in "${REMAINING_ARGS[@]}"; do
    case $arg in
        --uninstall) UNINSTALL_DRIVERS=1 ;;
        --docker) DOCKER_ONLY=1 ;;
        --check-docker-login) CHECK_DOCKER_LOGIN=1 ;;
        --status) STATUS_ONLY=1 ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $arg"; usage 1 ;;
    esac
done

# Fix broken dpkg state
# Parameters:
#   $1 - Mode: "uninstall" or "install" (default: install)
fix_broken_packages() {
    local mode="${1:-install}"
    echo_and_log "[INFO] Checking for broken package state..."

    # Check if dpkg is in a broken state
    # Status codes: iU=Unpacked, iF=Failed-config, iH=Half-installed, or R=Reinst-required
    local broken_packages
    broken_packages=$(dpkg -l | grep -E '^(iU|iF|iH|.[^ ]R)' | awk '{print $2}' || true)

    if [[ -n $broken_packages ]]; then
        echo_and_log "[WARN] Found packages in broken state:"
        echo_and_log "$broken_packages"
        echo_and_log "[INFO] Attempting to fix broken package state..."

        # Try to configure any unconfigured packages
        execute sudo dpkg --configure -a || true

        # Force remove packages that are in failed/half-installed state
        # This includes nvidia, doca, and related packages
        local failed_packages
        failed_packages=$(dpkg -l | grep -E '^(iU|iF|iH|.[^ ]R)' | grep -E '(nvidia|libnvidia|doca-|mlnx-)' | awk '{print $2}' || true)

        if [[ -n $failed_packages ]]; then
            echo_and_log "[INFO] Force removing broken nvidia/doca/mlnx packages:"
            echo_and_log "$failed_packages"
            echo_and_log "[INFO] NOTE: dpkg warnings about 'very bad inconsistent state' are expected during force-removal"
            for pkg in $failed_packages; do
                execute "sudo dpkg --remove --force-remove-reinstreq $pkg" || true
            done
            echo_and_log "[INFO] Broken packages removed - they will be cleanly reinstalled later"
        fi

        # Only try to fix dependencies during installation, not during uninstall
        if [[ "$mode" == "install" ]]; then
            echo_and_log "[INFO] Fixing remaining broken dependencies..."
            execute sudo apt --fix-broken install -y || true
        else
            echo_and_log "[INFO] Skipping dependency fix during uninstall mode"
        fi

        # Clean apt cache to remove problematic .deb files
        execute sudo apt clean

        echo_and_log "[INFO] Broken package state resolution complete"
    else
        echo_and_log "[INFO] No broken packages detected"
    fi
}

uninstall_docker() {
    echo_and_log "[INFO] Uninstalling Docker..."
    execute "sudo systemctl stop docker.socket" || true
    execute "sudo systemctl disable docker.socket" || true
    execute "sudo systemctl stop docker.service" || true
    execute "sudo systemctl disable docker.service" || true
    execute "sudo systemctl stop containerd.service" || true
    execute "sudo systemctl disable containerd.service" || true
    execute "sudo systemctl daemon-reload" || true
    execute sudo apt remove -y docker docker-engine docker.io containerd runc
    execute sudo apt purge -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    execute sudo apt remove $(dpkg --get-selections docker.io docker-compose docker-compose-v2 docker-doc podman-docker containerd runc | cut -f1)
    execute sudo apt autoremove -y
    execute sudo apt clean
    #execute sudo rm -rf /var/lib/docker
    #execute sudo rm -rf /var/lib/containerd
    echo_and_log "[INFO] Docker container images not removed"
    echo_and_log "[INFO] Docker uninstalled successfully"
}

# Install Docker
install_docker() {
    echo_and_log "[INFO] Installing Docker..."

    # Check if Docker is already installed with correct version
    if command -v docker &> /dev/null; then
        local current_version
        current_version=$(docker --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
        if [[ $current_version == "$DOCKER_VERSION" ]]; then
            echo_and_log "[INFO] Docker ${DOCKER_VERSION} already installed, skipping"
            return
        fi
    fi

    # Install Docker with specific version
    local ubuntu_version
    local codename
    ubuntu_version=$(. /etc/os-release && echo "$VERSION_ID")
    codename=$(. /etc/os-release && echo "$VERSION_CODENAME")

    # Construct Docker version string: 5:VERSION-1~ubuntu.XX.XX~CODENAME
    local VERSION_STRING="5:${DOCKER_VERSION}-1~ubuntu.${ubuntu_version}~${codename}"

    echo_and_log "[INFO] Installing Docker ${DOCKER_VERSION} (${VERSION_STRING})..."
    execute_or_die "sudo apt install -y --allow-downgrades \
        docker-ce=${VERSION_STRING} \
        docker-ce-cli=${VERSION_STRING} \
        containerd.io \
        docker-buildx-plugin \
        docker-compose-plugin \
        docker-model-plugin"

    # Enable and start Docker
    execute sudo systemctl enable docker
    execute sudo systemctl start docker

    add_user_to_docker

    # Re-initialize docker command wrapper now that the user is in the docker group.
    # init_docker_cmd ran at sourcing time (before add_user_to_docker) so its cached
    # DOCKER_PREFIX may be stale. Unsetting INIT_DOCKER_CMD_DONE forces a re-check,
    # which will detect the new group membership and switch to 'sg docker' if needed.
    unset INIT_DOCKER_CMD_DONE
    init_docker_cmd

    echo_and_log "[INFO] Docker installation complete"
}

check_docker_login() {
    echo_and_log "[INFO] Checking Docker login..."

    local current_user="${SUDO_USER:-$USER}"
    local user_home
    user_home=$(getent passwd "$current_user" | cut -d: -f6)

    # Create .docker directory for user if it doesn't exist
    if [[ ! -d "$user_home/.docker" ]]; then
        echo_and_log "[INFO] $user_home/.docker directory does not exist, creating..."
        execute sudo -u "$current_user" mkdir -p "$user_home/.docker"
    fi

    # Check if we can access the Aerial container image
    local aerial_image="${AERIAL_REPO}${AERIAL_IMAGE_NAME}:${AERIAL_VERSION_TAG}"
    echo_and_log "[INFO] Checking access to Aerial image: ${aerial_image}"

    local manifest_cmd="docker manifest inspect ${aerial_image}"
    if [[ $VERBOSE -eq 1 ]]; then
        echo_and_log "Executing: $manifest_cmd"
    fi

    if [[ $DRYRUN -eq 1 ]]; then
        echo_and_log "[DRY-RUN] $manifest_cmd"
        return 0
    fi

    local manifest_check
    manifest_check=$(docker manifest inspect "${aerial_image}" 2>&1)

    if echo "$manifest_check" | grep -q "no such manifest\|unauthorized\|authentication required"; then
        echo_and_log "[ERROR] Cannot access Aerial image: ${aerial_image}"
        echo_and_log "[ERROR] You may need to log in to the container registry:"
        echo_and_log "[ERROR]   docker login ${AERIAL_REPO%%/*}"
        FAILED=1
        return 1
    elif echo "$manifest_check" | grep -q "mediaType"; then
        echo_and_log "[INFO] Successfully verified access to Aerial image ${aerial_image}"
        return 0
    else
        echo_and_log "[WARN] Could not verify Aerial image access (unknown response)"
        echo_and_log "[WARN] Response: ${manifest_check}"
        return 0
    fi
}

# Setup APT repositories and GPG keys for Docker and NVIDIA Container Toolkit
setup_apt_repositories() {
    echo_and_log "[INFO] Setting up APT repositories..."
    echo_and_log "[INFO] Removing Docker repository /etc/apt/sources.list.d/docker.*"
    execute sudo rm -f /etc/apt/sources.list.d/docker.*
    echo_and_log "[INFO] Updating APT repositories after removing sources"
    execute sudo apt update

    # Create keyrings directory
    execute sudo install -m 0755 -d /etc/apt/keyrings

    local codename=$(. /etc/os-release && echo $VERSION_CODENAME)

    # Add Docker repository certificate
    echo_and_log "[INFO] Adding Docker certificate..."
    execute "sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --batch --yes --dearmor -o /etc/apt/keyrings/docker.gpg"
    execute sudo chmod a+r /etc/apt/keyrings/docker.gpg

    # Add Docker repository using .sources format
    echo_and_log "[INFO] Adding Docker repository..."
    if [[ $DRYRUN -ne 1 ]]; then
        sudo tee /etc/apt/sources.list.d/docker.sources > /dev/null <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: ${codename}
Components: stable
Signed-By: /etc/apt/keyrings/docker.gpg
#One-line style .list files are deprecated and may eventually be removed, but not before 2029.
EOF
    else
        echo_and_log "[DRY-RUN] sudo tee /etc/apt/sources.list.d/docker.sources"
    fi

    # Add NVIDIA container toolkit GPG key
    echo_and_log "[INFO] Adding NVIDIA Container Toolkit GPG key..."
    if [[ $DRYRUN -ne 1 ]]; then
        execute_or_die "curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --batch --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
    else
        echo_and_log "[DRY-RUN] curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --batch --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
    fi

    # Add NVIDIA container toolkit repository
    echo_and_log "[INFO] Adding NVIDIA Container Toolkit repository..."
    execute "curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null"

    # Update package lists with new repositories
    execute sudo apt update
    execute sudo apt --fix-broken install -y
    execute sudo dpkg --configure -a
}

# Install NVIDIA Container Toolkit
install_nvidia_container_toolkit() {
    echo_and_log "[INFO] Installing NVIDIA Container Toolkit..."

    execute sudo apt install -y nvidia-container-toolkit

    # Configure Docker runtime
    echo_and_log "[INFO] Configuring Docker runtime for NVIDIA..."
    execute sudo nvidia-ctk runtime configure --runtime=docker
    execute sudo systemctl restart docker

    # Verify installation
    echo_and_log "[INFO] Verifying NVIDIA container toolkit..."
    execute_retry_or_die 3 "docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi"
}

# Download and install DOCA-Host deb package
install_doca_host() {
    echo_and_log "[INFO] Installing DOCA ${DOCA_VERSION}..."
    if [[ -f "${DOCA_DEB}" && -s "${DOCA_DEB}" ]]; then
        echo_and_log "[INFO] Using existing ${DOCA_DEB} ($(du -h "${DOCA_DEB}" | cut -f1))"
    else
        execute_or_die "wget -nv ${DOCA_URL}"
    fi

    echo_and_log "[INFO] Installing DOCA-Host package..."
    execute_or_die sudo dpkg -i "${DOCA_DEB}"

    echo_and_log "[INFO] Updating package lists after DOCA repo addition..."
    execute_or_die sudo apt update

    echo_and_log "[INFO] Installing DOCA tools, OFED, and firmware updater..."
    local openibd_cur="/etc/infiniband/openib.conf"
    local openibd_orig="/etc/infiniband/openib.conf.dpkg-old"
    if [[ -f $openibd_orig ]]; then
        if diff -q "$openibd_cur" "$openibd_orig" >/dev/null 2>&1; then
            echo_and_log "[INFO] OFED configuration file is unchanged, skipping copy"
        else
            local openibd_backup="${openibd_cur}.old-$(date +%Y-%m-%dT%H%M%S)"
            echo_and_log "[INFO] OFED configuration file has changed, backing up current version to $openibd_backup, because this can break openibd installation"
            copy_file "$openibd_cur" "$openibd_backup"
            copy_file "$openibd_orig" "$openibd_cur"
       fi
    fi
    execute_or_die sudo apt -y install doca-tools doca-ofed mlnx-fw-updater
}

# Start MST (Mellanox Software Tools)
start_mst() {
    echo_and_log "[INFO] Starting MST (Mellanox Software Tools) driver set..."
    execute_or_die sudo mst start
}

# Display version and status information
show_status() {
    echo_and_log "[INFO] Docker version (expected: ${DOCKER_VERSION}):"
    local docker_version
    docker_version=$(docker --version 2>/dev/null) || { echo_and_log "[ERROR] Docker is not installed"; FAILED=1; return; }
    echo_and_log "  $docker_version"
    if ! echo "$docker_version" | grep -q "Docker version ${DOCKER_VERSION}"; then
        echo_and_log "[ERROR] Docker version mismatch. Expected ${DOCKER_VERSION}"
        FAILED=1
    fi

    echo ""
    echo_and_log "[INFO] OFED version:"
    ofed_info -s || { echo_and_log "[WARN] Could not get OFED version"; FAILED=1; }

    echo ""
    echo_and_log "[INFO] MST version:"
    execute "sudo mst version" || { echo_and_log "[WARN] Could not get MST version"; FAILED=1; }

    echo ""
    echo_and_log "[INFO] MST status:"
    execute "sudo mst status -v" || { echo_and_log "[WARN] Could not get MST status"; FAILED=1; }

    echo ""
    echo_and_log "[INFO] Checking ConnectX link status on aerial interfaces..."

    # Find all aerial0x interfaces
    local AERIAL_INTERFACES
    AERIAL_INTERFACES=$(ip -o link show | grep -oE 'aerial0[0-9]+' | sort -u)

    if [[ -z $AERIAL_INTERFACES ]]; then
        echo_and_log "[WARN] No aerial0x interfaces found (run 'make net' to create them)"
        return
    fi

    local interfaces_down=1
    for IFACE in $AERIAL_INTERFACES; do
        # Get PCI address for this interface
        local pci_addr
        pci_addr=$(ethtool -i "$IFACE" 2>/dev/null | grep "bus-info" | awk '{print $2}')

        if [[ -z $pci_addr ]]; then
            echo_and_log "[WARN] Could not get PCI address for $IFACE"
            FAILED=1
            continue
        fi

        local mlxlink_output
        mlxlink_output=$(sudo mlxlink -d "$pci_addr" 2>/dev/null) || {
            echo_and_log "[WARN] $IFACE ($pci_addr): mlxlink query failed"
            FAILED=1
            continue
        }

        local state physical_state speed
        state=$(echo "$mlxlink_output" | grep "^State" | awk -F: '{print $2}' | xargs)
        physical_state=$(echo "$mlxlink_output" | grep "^Physical state" | awk -F: '{print $2}' | xargs)
        speed=$(echo "$mlxlink_output" | grep "^Speed" | awk -F: '{print $2}' | xargs)

        if [[ "$state" == *"Active"* ]] && [[ "$physical_state" == *"LinkUp"* || "$physical_state" == *"ETH_AN_FSM_ENABLE"* ]]; then
            interfaces_down=0
            if [[ "$speed" == *"100G"* ]]; then
                echo_and_log "[INFO] $IFACE ($pci_addr): OK - ${speed}, ${physical_state}"
            elif [[ "$speed" == *"10G"* ]]; then
                echo_and_log "[INFO] $IFACE ($pci_addr): OK - ${speed}, ${physical_state} (note: 10G link speed)"
            fi
        else
            echo_and_log "[WARN] $IFACE ($pci_addr): ${state}, ${physical_state}, ${speed} (expected: Active, LinkUp or ETH_AN_FSM_ENABLE)"
        fi
    done

    if [[ $interfaces_down -eq 1 ]]; then
        echo_and_log "[ERROR] No aerial interfaces are up"
        FAILED=1
        return
    fi
}

# Track display managers stopped by unload_nvidia_modules so install_gpu_driver can restart them
STOPPED_DISPLAY_MANAGERS=()

# Unload nvidia kernel modules (non-fatal)
unload_nvidia_modules() {
    echo_and_log "[INFO] Attempting to unload nvidia kernel modules..."

    # Step 1: Mask and stop nvidia-persistenced — it auto-reloads modules and holds
    # nvidia-modeset open, causing "module in use" errors during rmmod.
    # Both mask (prevents restart) and stop (kills running instance) are required.
    echo_and_log "[INFO] Masking and stopping nvidia-persistenced to prevent module reload..."
    execute "sudo systemctl mask nvidia-persistenced 2>/dev/null || true"
    execute "sudo systemctl stop nvidia-persistenced 2>/dev/null || true"

    # Stop any active display managers — on desktop systems (e.g. DGX Spark with gdm3)
    # the display manager holds the GPU open via Wayland/X11, blocking module unload.
    # On headless servers none of these will be active, so this is a safe no-op there.
    #
    # IMPORTANT: if this script is running inside a GUI terminal, stopping the display
    # manager will kill the terminal and this script mid-execution. Detect that and abort.
    local dm dm_pid pid ppid
    STOPPED_DISPLAY_MANAGERS=()
    for dm in gdm3 gdm lightdm sddm xdm; do
        if ! systemctl is-active --quiet "$dm" 2>/dev/null; then
            continue
        fi

        # Walk our process ancestry to see if we are a descendant of this display manager.
        dm_pid=$(systemctl show -p MainPID --value "$dm" 2>/dev/null || true)
        if [[ -n "$dm_pid" && "$dm_pid" != "0" ]]; then
            pid=$$
            while [[ $pid -gt 1 ]]; do
                if [[ $pid -eq $dm_pid ]]; then
                    echo_and_log "[ERROR] This script is running inside a GUI session managed by $dm (PID $dm_pid)."
                    echo_and_log "[ERROR] Stopping $dm would kill your terminal and abort this script mid-install."
                    echo_and_log "[ERROR] Please re-run from an SSH session or a virtual console (Ctrl+Alt+F2)."
                    exit 1
                fi
                ppid=$(awk '{print $4}' /proc/$pid/stat 2>/dev/null) || break
                [[ "$ppid" == "$pid" ]] && break  # init is its own parent
                pid=$ppid
            done
        fi

        echo_and_log "[INFO] Stopping display manager $dm..."
        execute "sudo systemctl stop $dm || true"
        STOPPED_DISPLAY_MANAGERS+=("$dm")
    done

    # Check if any nvidia modules are currently loaded
    local modules
    modules=$(lsmod | awk '/^[^[:space:]]*(nvidia|nv_|gdrdrv)/ {print $1}')

    if [[ -z $modules ]]; then
        echo_and_log "[INFO] No nvidia kernel modules currently loaded"
        return
    fi

    # Step 2: Unload in the correct dependency order.
    # nvidia_uvm and nvidia_drm are non-fatal (may not be present);
    # nvidia_modeset and nvidia follow after.
    local ordered_modules=(nvidia_uvm nvidia_drm nvidia_modeset nvidia)
    for m in "${ordered_modules[@]}"; do
        if lsmod | grep -q "^${m} "; then
            echo_and_log "[INFO] Unloading $m..."
            if [[ $DRYRUN -eq 1 ]]; then
                echo_and_log "[DRY-RUN] sudo rmmod $m"
            else
                sudo rmmod "$m" 2>/dev/null || true
            fi
        fi
    done

    # Unload any remaining nvidia-related modules not covered by the ordered list
    local remaining
    remaining=$(lsmod | awk '/^[^[:space:]]*(nvidia|nv_|gdrdrv)/ {print $1}')
    for m in $remaining; do
        echo_and_log "[INFO] Unloading remaining module $m..."
        if [[ $DRYRUN -eq 1 ]]; then
            echo_and_log "[DRY-RUN] sudo rmmod $m"
        else
            sudo rmmod "$m" 2>/dev/null || true
        fi
    done

    # Step 3: Verify modules are gone before continuing
    local still_loaded
    still_loaded=$(lsmod | awk '/^[^[:space:]]*(nvidia|nv_|gdrdrv)/ {print $1}')

    if [[ -n $still_loaded ]]; then
        echo_and_log "[WARN] Some nvidia modules could not be unloaded (still in use):"
        echo_and_log "$still_loaded"
        echo_and_log "[INFO] This is OK - continuing with package operations"
        echo_and_log "[INFO] NOTE: Reboot required for driver changes to take full effect"
    else
        echo_and_log "[INFO] All nvidia kernel modules successfully unloaded"
    fi
}
# Restore services stopped by unload_nvidia_modules: unmask+start nvidia-persistenced
# and restart any display managers that were stopped.
restore_gpu_services() {
    echo_and_log "[INFO] Unmasking and restarting nvidia-persistenced..."
    execute "sudo systemctl unmask nvidia-persistenced || true"
    execute "sudo systemctl start nvidia-persistenced || true"

    local dm
    for dm in "${STOPPED_DISPLAY_MANAGERS[@]}"; do
        echo_and_log "[INFO] Restarting display manager $dm..."
        execute "sudo systemctl start $dm || true"
    done
}

# Uninstall GPU drivers
uninstall_drivers() {
    echo_and_log "[INFO] Uninstalling GPU drivers..."

    unload_nvidia_modules
    fix_broken_packages "uninstall"

    # Remove nvidia/gdrdrv packages - find installed packages matching patterns
    echo_and_log "[INFO] Finding installed nvidia/gdrdrv packages..."
    local nvidia_packages
    # Get package names and strip architecture suffix (e.g., :arm64, :amd64)
    # Use ^i. to match any install state (ii=installed, iF=failed, iU=unpacked, etc.)
    # Exclude nvidia-disable-aqc-nic (config package with known broken post-removal script)
    nvidia_packages=$(dpkg -l | \
        grep -E '^i..*(cuda-compat|cuda-drivers|gdrdrv|libnvidia-|libnvsdm|libxnvctrl|nv-docker-gpus|nvidia-compute-utils|nvidia-conf-xconfig|nvidia-container-toolkit|nvidia-dkms|nvidia-driver|nvidia-fabricmanager|nvidia-firmware|nvidia-headless|nvidia-imex|nvidia-kernel|nvidia-modprobe|nvidia-open|nvidia-persistenced|nvidia-settings|nvidia-system-core|nvidia-system-utils|nvidia-utils|nvidia-xconfig|xserver-xorg-video-nvidia|nvidia.*-580|nvidia.*-590)' | \
        grep -v 'nvidia-disable-aqc-nic' | \
        awk '{print $2}' | \
        sed 's/:.*$//' | \
        sort -u || true)

    if [[ -n $nvidia_packages ]]; then
        echo_and_log "[INFO] Removing nvidia packages using dpkg (bypassing dependency checks):"
        echo_and_log "$nvidia_packages"
        # Use dpkg directly to avoid apt trying to resolve dependencies by installing packages
        for pkg in $nvidia_packages; do
            echo_and_log "[INFO] Removing $pkg..."
            execute "sudo dpkg --purge --force-all $pkg"
        done

        autoremove_until_clean
    else
        echo_and_log "[INFO] No nvidia packages found to remove"
    fi

    restore_gpu_services

    echo_and_log "[INFO] GPU driver uninstallation complete"
}

# Install GPU driver
install_gpu_driver() {
    echo_and_log "[INFO] Installing GPU driver ${GPU_DRIVER_VERSION}..."

    # Skip if the correct version is already installed
    local installed_version
    installed_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)
    if [[ "$installed_version" == "$GPU_DRIVER_VERSION" ]]; then
        echo_and_log "[INFO] GPU driver ${GPU_DRIVER_VERSION} already installed, skipping"
        return
    fi

    unload_nvidia_modules

    local download_file="${GPU_DRIVER_DOWNLOAD_FILE:-$GPU_DRIVER_FILE}"
    local driver_run_to_install="$GPU_DRIVER_FILE"
    local extract_dir=

    # Download driver (or CUDA run file on ARM) if not already present
    if [[ -f "${download_file}" && -s "${download_file}" ]]; then
        echo_and_log "[INFO] Using existing ${download_file} ($(du -h "${download_file}" | cut -f1))"
    else
        echo_and_log "[INFO] Downloading from ${GPU_DRIVER_URL}..."
        execute_or_die "wget -nv -O ${download_file} ${GPU_DRIVER_URL}"
    fi

    if [[ -n "${CUDA_RUN_FILE_NAME:-}" && "$download_file" == "$CUDA_RUN_FILE_NAME" ]]; then
        extract_dir="$(mktemp -d)"
        echo_and_log "[INFO] Extracting CUDA run file to get driver .run..."
        execute_or_die "sh ${download_file} --extract=${extract_dir}"
        driver_run_to_install="${extract_dir}/${GPU_DRIVER_FILE}"
        if [[ ! -f "$driver_run_to_install" ]] && [[ $DRYRUN -eq 0 ]]; then
            echo_and_log "[ERROR] Driver run file not found after extract: $driver_run_to_install"
            exit 1
        fi
    fi

    # Install driver
    echo_and_log "[INFO] Running GPU driver installer..."
    # NOTE: The libglvnd warning from the installer is non-fatal — nvidia-smi working
    # afterwards confirms a successful install.
    execute_or_die sudo sh "${driver_run_to_install}" --silent -m kernel-open

    # Clean up ARM extract dir
    if [[ -n "${extract_dir:-}" && -d "${extract_dir:-}" ]]; then
        [[ $DRYRUN -eq 0 ]] && rm -rf "$extract_dir"
    fi

    restore_gpu_services

    echo_and_log "[INFO] GPU driver installation complete"
}

# Install GDRCopy driver (GPU Direct RDMA)
install_gdrdrv() {
    if [[ $PLATFORM == "NVIDIA_DGX_Spark_P4242" ]]; then
        echo_and_log "[INFO] Skipping GDRCopy driver installation on DGX Spark"
        return
    fi

    echo_and_log "[INFO] Installing GDRCopy driver ${GDRDRV_VERSION}..."

    if [[ -f "${GDRDRV_FILE}" && -s "${GDRDRV_FILE}" ]]; then
        echo_and_log "[INFO] Using existing ${GDRDRV_FILE} ($(du -h "${GDRDRV_FILE}" | cut -f1))"
    else
        echo_and_log "[INFO] Downloading GDRCopy driver from ${GDRDRV_URL}..."
        execute_or_die "wget -nv ${GDRDRV_URL}"
    fi

    # Install the deb package
    echo_and_log "[INFO] Installing GDRCopy driver package..."
    execute_or_die sudo dpkg -i "${GDRDRV_FILE}"

    echo_and_log "[INFO] GDRCopy driver installation complete"
}

# Configure NVLink for GH200 (Supermicro only)
configure_nvlink() {
    if [[ $PLATFORM == "Supermicro_ARS-111GL-NHR" ]]; then
        echo_and_log "[INFO] Configuring NVLink for GH200..."
        write_file /etc/modprobe.d/nvidia.conf \
            'options nvidia NVreg_RegistryDwords="RMNvLinkDisableLinks=0x3FFFF;"'
    fi
}

# Install rcu_affinity_manager.sh script
install_rcu_affinity_manager() {
    echo_and_log "[INFO] Installing rcu_affinity_manager.sh..."

    local SRC_FILE="${_SCRIPT_DIR}/../infra/rcu_affinity_manager.sh"
    local DST_FILE="/usr/local/bin/rcu_affinity_manager.sh"

    if [[ ! -f $SRC_FILE ]]; then
        echo "============================================"
        echo_and_log "[WARN] rcu_affinity_manager.sh will not be installed because source file not found: $SRC_FILE"
        echo "============================================"
        return
    fi

    copy_file "$SRC_FILE" "$DST_FILE"
    echo_and_log "[INFO] Installed $DST_FILE"
}

# Main execution
main() {
    # Wait for apt/dpkg lock unless this run only does status or docker-login check (no apt).
    if [[ $STATUS_ONLY -eq 0 && $CHECK_DOCKER_LOGIN -eq 0 ]]; then
        check_and_release_apt_lock 300 || exit 1
    fi

    # Handle uninstall option
    if [[ $UNINSTALL_DRIVERS -eq 1 ]]; then
        echo_and_log "[INFO] Starting driver removal..."
        echo ""
        uninstall_drivers
        echo ""
        echo "============================================"
        if [[ $FAILED -eq 1 ]]; then
            echo_and_log "[ERROR] Driver removal completed with errors"
            exit 1
        else
            echo_and_log "[INFO] Driver removal completed successfully!"
        fi
        return
    fi

    if [[ $CHECK_DOCKER_LOGIN -eq 1 ]]; then
        echo "============================================"
        echo "Checking docker login..."
        echo "============================================"
        check_docker_login
        exit $FAILED
    fi

    # Handle docker-only option
    if [[ $DOCKER_ONLY -eq 1 ]]; then
        echo_and_log "[INFO] Installing Docker and NVIDIA Container Toolkit..."
        echo ""
        install_docker
        check_docker_login
        install_nvidia_container_toolkit
        echo ""
        echo "============================================"
        if [[ $FAILED -eq 1 ]]; then
            echo_and_log "[ERROR] Docker installation completed with errors"
            exit 1
        else
            echo_and_log "[INFO] Docker installation completed successfully!"
        fi
        return
    fi

    # Handle status-only option
    if [[ $STATUS_ONLY -eq 1 ]]; then
        echo_and_log "[INFO] Checking driver and interface status..."
        echo ""
        show_status
        echo ""
        echo "============================================"
        if [[ $FAILED -eq 1 ]]; then
            echo_and_log "[ERROR] Status check completed with errors"
            exit 1
        else
            echo_and_log "[INFO] Status check completed successfully!"
        fi
        return
    fi

    echo_and_log "[INFO] Starting driver installation..."

    # Sub-stamp helpers for resumable installs.
    # Each major step group touches a sub-stamp so that if the script is interrupted
    # (e.g., docker group not yet active) and re-run, completed groups are skipped.
    # Sub-stamps share the same .stamps/ directory used by the Makefile.
    local _sub_stamp_dir="${_SCRIPT_DIR}/.stamps"
    _step_done()  { [[ -f "${_sub_stamp_dir}/drivers_${1}" ]]; }
    _stamp_step() { [[ $DRYRUN -eq 0 ]] && mkdir -p "$_sub_stamp_dir" && touch "${_sub_stamp_dir}/drivers_${1}"; true; }

    # These are idempotent / fast — run every time
    fix_broken_packages "install"
    setup_apt_repositories
    install_docker
    check_docker_login

    # Group 1: DOCA + NIC firmware (download-heavy, skip if already done)
    if _step_done "doca"; then
        echo_and_log "[INFO] Skipping DOCA/NIC firmware install (sub-stamp exists)"
    else
        install_doca_host
        start_mst
        install_rcu_affinity_manager
        # Persist this stage before the intentional Spark reboot exit so the
        # next run resumes with post-reboot firmware verification.
        _stamp_step "doca"

        if [[ $PLATFORM == "NVIDIA_DGX_Spark_P4242" ]]; then
            if [[ $DRYRUN -eq 1 ]]; then
                echo_and_log "[DRY-RUN] A system reboot would be required before verifying the updated NIC firmware"
            else
                echo ""
                echo "============================================"
                echo_and_log "[IMPORTANT] DOCA and mlnx-fw-updater installation completed."
                echo_and_log "[IMPORTANT] Reboot the system so the NIC firmware update can take effect, then run make install again."
                echo "============================================"
                exit 2
            fi
        fi
    fi

    # Group 2: GPU driver + GDRCopy (download-heavy, skip if already done)
    if _step_done "gpu"; then
        echo_and_log "[INFO] Skipping GPU driver / GDRCopy install (sub-stamp exists)"
    else
        install_gpu_driver
        configure_nvlink
        install_gdrdrv
        _stamp_step "gpu"
    fi

    # Group 3: NVIDIA container toolkit + Docker verification
    # This is the step that requires docker group membership to be active.
    # If it failed on a previous run (e.g., user not yet in docker group),
    # log out, log back in, then re-run make drivers to retry only this step.
    if _step_done "toolkit"; then
        echo_and_log "[INFO] Skipping NVIDIA container toolkit install (sub-stamp exists)"
    else
        install_nvidia_container_toolkit
        _stamp_step "toolkit"
    fi

    echo ""
    echo "============================================"
    echo_and_log "Installation complete! Displaying status..."
    echo "============================================"
    echo ""

    show_status

    echo ""
    echo "============================================"
    if [[ $FAILED -eq 1 ]]; then
        echo_and_log "[ERROR] Driver installation completed with errors"
        exit 1
    else
        echo_and_log "[INFO] Driver installation completed successfully!"
    fi
    echo "============================================"
}

main "$@"
