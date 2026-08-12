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

# install_services.sh - Install and configure services for NVIDIA Aerial
#
# This script installs and configures:
#   - linuxptp package (ptp4l, phc2sys)
#   - PTP configuration and services
#   - nvidia.service for GPU and system optimizations
#   - cpu-latency.service for low-latency operation
#   - ethtool TX timestamping optimizations
#
# Usage: ./install_services.sh [--dry-run] [--verbose] [--interface <name>] [--detect-ptp] [--status] [--check-sync]
#
# Reference: https://docs.nvidia.com/aerial/cuda-accelerated-ran/aerial_cubb/cubb_install/aerial_system_scripts.html
#

# Source common functions
_SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
[[ -f "$_SCRIPT_DIR/includes.sh" ]] && source "$_SCRIPT_DIR/includes.sh" || { echo "ERROR: includes.sh not found: $_SCRIPT_DIR/includes.sh" >&2; exit 1; }

FH_INTERFACE="aerial00"
DETECT_PTP_ONLY=0
STATUS_ONLY=0
CHECK_SYNC_ONLY=0

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Install and configure services for NVIDIA Aerial (PTP, GPU optimizations)"
    echo ""
    echo "Options:"
    echo "  --dry-run           Show commands without executing"
    echo "  --verbose           Print commands before executing"
    echo "  --interface=NAME    Specify fronthaul interface (default: aerial00)"
    echo "  --detect-ptp        Detect PTP traffic and exit"
    echo "  --status            Show service status and exit"
    echo "  --check-sync        Check PTP sync status and exit"
    echo "  -h, --help          Show this help message"
    exit "${1:-0}"
}

# Parse common arguments (--dry-run, --verbose) and populate REMAINING_ARGS
parse_common_args "$@"
verify_secure_boot_disabled
echo_and_log "$(date) - Starting service installation"

# Re-set positional parameters to REMAINING_ARGS for shift-based parsing for use with parse_common_args
set -- "${REMAINING_ARGS[@]}"
while [[ $# -gt 0 ]]; do
    case $1 in
        --detect-ptp) DETECT_PTP_ONLY=1; shift ;;
        --status) STATUS_ONLY=1; shift ;;
        --check-sync) CHECK_SYNC_ONLY=1; shift ;;
        --interface=*) FH_INTERFACE="${1#*=}"; shift ;;
        --interface) FH_INTERFACE="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1"; usage 1 ;;
    esac
done

# Disable NTP to prevent conflicts with PTP
disable_ntp() {
    echo_and_log "[INFO] Disabling NTP (PTP will handle time synchronization)..."
    execute sudo timedatectl set-ntp false
    echo_and_log "[INFO] NTP disabled"
}

# Install linuxptp package at the version specified in versions.sh.
# On Ubuntu 24.04, linuxptp 4.2 is available via apt.
# On Ubuntu 22.04, the apt package is older so we build 4.2 from source.
# Install linuxptp: if apt would install a version >= LINUXPTP_VERSION, use apt; otherwise build from source.
# apt-cache works for packages that are not installed (it queries the repository).
install_linuxptp() {
    echo_and_log "[INFO] Checking version of installed ptp4l. Required version: ${LINUXPTP_VERSION}"

    hash -r # Update bash hash in case ptp has moved.
    local apt_ver req_ver apt_installed ptp_prefix ptp_target_loc
    req_ver="${LINUXPTP_VERSION:-4.0}" # Minimum version is 4.0
    apt_policy=$(apt-cache policy linuxptp 2>/dev/null)
    apt_ver=$(echo "$apt_policy" | awk '/Candidate:/ {print $2; exit}' | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?')
    apt_installed=0
    if [[ $(echo "$apt_policy" | awk '/Installed:/ {print $2; exit}') != "(none)" ]]; then
        apt_installed=1
    fi
    ptp_installed=$(command -v ptp4l &>/dev/null && ptp4l -v )
    ptp_loc=$(command -v ptp4l)
    if [[ -n "${ptp_installed}" ]]; then
        if  [[ "$(printf '%s\n%s' "$ptp_installed" "$req_ver" | sort -V | head -n1)" == "$req_ver" ]]; then
            echo_and_log "[INFO] ptp4l ${ptp_installed} found at ${ptp_loc}. Not updating"
            return 0
        else
            echo_and_log "[INFO] ptp4l ${ptp_installed} found at ${ptp_loc} but is older than ${req_ver}. Updating."
        fi
    fi

    #This sorts the two versions smallest to largest, output is the lower of the two. So if apt_ver is req_ver, install via apt.
    if [[ -n "${apt_ver}" ]] && [[ "$(printf '%s\n%s' "$apt_ver" "$req_ver" | sort -V | head -n1)" == "$req_ver" ]]; then
        echo_and_log "[INFO] apt would install linuxptp ${apt_ver} (>= ${req_ver}) — installing via apt"
        execute sudo apt-get update
        execute sudo apt-get install -y linuxptp
    else
        echo_and_log "[INFO] apt has linuxptp ${apt_ver:-unknown}, need >= ${req_ver} — building from source"
        ptp_prefix=/usr
        ptp_target_loc=/usr/sbin
        #If ptp was installed to a non-standard location, use the same prefix for the new installation.
        if [[ -n "${ptp_loc}" ]] && [[ "${ptp_loc}" != "${ptp_target_loc}/ptp4l" ]]; then
            ptp_prefix=$(dirname "$ptp_loc")
            if [[ "${ptp_prefix}" == */sbin ]]; then
                ptp_prefix="${ptp_prefix%/sbin}"
            elif [[ "${ptp_prefix}" == */bin ]]; then
                ptp_prefix="${ptp_prefix%/bin}"
            fi
            echo_and_log "[INFO] Existing ptp installation directory: ${ptp_loc}, installing new version with prefix: ${ptp_prefix}"
        fi
        if [[ $apt_installed -eq 1 ]]; then
            execute sudo apt-get remove -y linuxptp 2>/dev/null || true
        fi
        # build-essential and wget are already installed by the install_deps script.

        local src_url="https://github.com/richardcochran/linuxptp/archive/refs/tags/v${LINUXPTP_VERSION}.tar.gz"
        local src_tar="linuxptp-${LINUXPTP_VERSION}.tar.gz"
        local src_dir="linuxptp-${LINUXPTP_VERSION}"

        if [[ -f "$src_tar" && -s "$src_tar" ]]; then
            echo_and_log "[INFO] Using existing ${src_tar}"
        else
            execute_or_die "wget -nv -O ${src_tar} ${src_url}"
        fi

        execute_or_die "tar -xzf ${src_tar}"
        execute_or_die "make -C ${src_dir}"
        # For make install, prefix is the install root; sbindir will be ${prefix}/sbin
        execute_or_die "sudo make -C ${src_dir} install prefix=${ptp_prefix}"
    fi

    hash -r # Refresh bash hash
    if command -v ptp4l &>/dev/null; then
        echo_and_log "[INFO] linuxptp installed: $(ptp4l -v) at $(command -v ptp4l)"
    else
        echo_and_log "[ERROR] linuxptp installation failed - ptp4l not found"
        exit 1
    fi

}

# Detect PTP traffic on aerial interfaces and set PTP_INTERFACE
# Runs tcpdump with PTP filter for 2 seconds on each interface
# Sets PTP_INTERFACE to the first interface with PTP traffic
detect_ptp_traffic() {
    echo_and_log "[INFO] Detecting PTP traffic on aerial interfaces (2 second timeout per interface)..."

    # Find all aerial0x interfaces
    local AERIAL_INTERFACES
    AERIAL_INTERFACES=$(ip -o link show | grep -oE 'aerial0[0-9]+' | sort -u | tr '\n' ' ')

    if [[ -z $AERIAL_INTERFACES ]]; then
        echo_and_log "[WARN] No aerial0x interfaces found"
        return 1
    fi
    echo_and_log "[INFO] Found interfaces: ${AERIAL_INTERFACES}"

    PTP_INTERFACE=""
    local ptp_interfaces=""

    for IFACE in $AERIAL_INTERFACES; do
        echo_and_log -n "[INFO] Checking ${IFACE}... "

        # Check if link is up first (|| true prevents set -e from exiting on grep failure)
        local link_state
        link_state=$(ip link show "$IFACE" 2>/dev/null | grep -o 'state [A-Z]*' | awk '{print $2}' || true)
        if [[ $link_state != "UP" ]]; then
            echo_and_log "link down (state: ${link_state:-UNKNOWN})"
            continue
        fi

        # Run tcpdump for 2 seconds, filter for PTP (ethertype 0x88f7)
        # PTP uses ethertype 0x88f7 for L2 transport
        # tcpdump -c 1 exits 0 if it captures a packet, timeout exits 124 if no packet
        # Note: Cannot use execute() here as we need to capture tcpdump result
        if [[ $DRYRUN -eq 1 ]]; then
            echo_and_log "[DRY-RUN] Would run: sudo timeout 2 tcpdump -i $IFACE -c 1 'ether proto 0x88f7'"
        else
            if sudo timeout 2 tcpdump -q -i "$IFACE" -c 1 'ether proto 0x88f7' >/dev/null 2>&1; then
                echo_and_log "PTP traffic detected!"
                # Set PTP_INTERFACE to first interface with traffic
                if [[ -z $PTP_INTERFACE ]]; then
                    PTP_INTERFACE="$IFACE"
                fi
                ptp_interfaces="$ptp_interfaces $IFACE"
            else
                echo_and_log "no PTP traffic"
            fi
        fi
    done

    echo ""
    if [[ -n $PTP_INTERFACE ]]; then
        echo_and_log "[INFO] PTP traffic detected on:$ptp_interfaces"
        echo_and_log "[INFO] Using ${PTP_INTERFACE} for PTP configuration"
        return 0
    else
        echo_and_log "[WARN] No PTP traffic detected on any aerial interface"
        echo_and_log "[WARN] Ensure PTP grandmaster is connected and sending traffic"
        return 1
    fi
}

# Create PTP configuration file
create_ptp_conf() {
    # Use detected PTP_INTERFACE if available, otherwise fall back to FH_INTERFACE
    local ptp_iface="${PTP_INTERFACE:-$FH_INTERFACE}"
    echo_and_log "[INFO] Creating /etc/ptp.conf for interface ${ptp_iface}..."

    write_file /etc/ptp.conf << EOF
# PTP configuration for NVIDIA Aerial
# Optimized for DGX Spark with ConnectX-7 NIC

[global]
dataset_comparison              G.8275.x
G.8275.defaultDS.localPriority  128
maxStepsRemoved                 255
logAnnounceInterval             -3
logSyncInterval                 -4
logMinDelayReqInterval          -4
G.8275.portDS.localPriority     128
network_transport               L2
domainNumber                    24
tx_timestamp_timeout            30
clientOnly                      1

clock_servo pi
step_threshold 1.0
egressLatency 28
pi_proportional_const 4.65
pi_integral_const 0.1

[${ptp_iface}]
announceReceiptTimeout 3
delay_mechanism E2E
network_transport L2
EOF

    echo_and_log "[INFO] Created /etc/ptp.conf"

    # Verify ptp4l can parse the configuration (skip in dry-run mode).
    # ptp4l is a daemon and never exits on its own, so use timeout to kill it after a few seconds.
    if [[ $DRYRUN -ne 1 ]]; then
        echo_and_log "[INFO] Verifying ptp4l configuration (5s timeout)..."
        local ptp4l_out
        ptp4l_out=$(sudo timeout 5 ptp4l -f /etc/ptp.conf -m -q 2>&1 || true)
        echo_and_log "$ptp4l_out"
        if echo "$ptp4l_out" | grep -qE "selected.*PTP clock|INITIALIZING"; then
            echo_and_log "[INFO] ptp4l configuration verified successfully"
        else
            echo_and_log "[WARN] ptp4l configuration check produced unexpected output (may be OK without network)"
        fi
    fi
}

# Create ptp4l systemd service
create_ptp4l_service() {
    local ptp_iface="${PTP_INTERFACE:-$FH_INTERFACE}"
    echo_and_log "[INFO] Creating /etc/systemd/system/ptp4l.service..."

    if [[ -f /lib/systemd/system/ptp4l.service ]]; then
        echo_and_log "[WARN] /lib/systemd/system/ptp4l.service already present."
        echo_and_log "[WARN] /etc/systemd/system/ptp4l.service takes precedence and overrides /lib/systemd/system/ptp4l.service"
        # Prepend a note to the /lib file so admins see it will be preempted by /etc
        if [[ $DRYRUN -eq 1 ]]; then
            echo_and_log "[DRY-RUN] Would prepend preemption note to /lib/systemd/system/ptp4l.service"
        else
            if ! sudo head -1 /lib/systemd/system/ptp4l.service | grep -q "preempted by the file in /etc"; then
                local _prepend_line='# This file is preempted by the file in /etc/systemd/system.'
                execute "sudo sed -i '1i$_prepend_line' /lib/systemd/system/ptp4l.service"
            fi
        fi
    fi

    local ptp_cpu_affinity_line=""
    [[ -n "${PTP_CPU_AFFINITY:-}" ]] && ptp_cpu_affinity_line="CPUAffinity=${PTP_CPU_AFFINITY}"
    write_file /etc/systemd/system/ptp4l.service << EOF
[Unit]
Description=Precision Time Protocol (PTP) service
Documentation=man:ptp4l
After=network.target nvidia.service
ConditionPathExistsGlob=/dev/ptp*

[Service]
Restart=always
RestartSec=5s
Type=simple
ExecStart=/usr/sbin/ptp4l -f /etc/ptp.conf
${ptp_cpu_affinity_line}
[Install]
WantedBy=multi-user.target
EOF

    echo "[INFO] Created ptp4l.service"
}

# Create phc2sys systemd service
create_phc2sys_service() {
    # Use detected PTP_INTERFACE if available, otherwise fall back to FH_INTERFACE
    local ptp_iface="${PTP_INTERFACE:-$FH_INTERFACE}"
    echo_and_log "[INFO] Creating /etc/systemd/system/phc2sys.service for interface ${ptp_iface}..."

    local ptp_cpu_affinity_line=""
    [[ -n "${PTP_CPU_AFFINITY:-}" ]] && ptp_cpu_affinity_line="CPUAffinity=${PTP_CPU_AFFINITY}"
    write_file /etc/systemd/system/phc2sys.service << EOF
[Unit]
Description=Synchronize system clock or PTP hardware clock (PHC)
Documentation=man:phc2sys
Requires=ptp4l.service
After=ptp4l.service

[Service]
Restart=always
RestartSec=5s
Type=simple
# Gives ptp4l a chance to stabilize
ExecStartPre=sleep 2
ExecStart=/bin/sh -c "/usr/sbin/phc2sys -s ${ptp_iface} -c CLOCK_REALTIME -n 24 -O 0 -R 256 -u 256"
${ptp_cpu_affinity_line}

[Install]
WantedBy=multi-user.target
EOF

    echo_and_log "[INFO] Created phc2sys.service"
}

# Install nvidia.sh script for GPU and system optimizations.
# nvidia.sh is a template; platform-specific commands are substituted before
# writing it to /usr/local/bin/nvidia.sh.
install_nvidia_script() {
    echo_and_log "[INFO] Installing /usr/local/bin/nvidia.sh..."

    local script_path
    script_path=$(realpath "${BASH_SOURCE[0]}")
    local SCRIPT_DIR=$(dirname "$script_path")
    local SRC_FILE="${SCRIPT_DIR}/nvidia.sh"
    local DST_FILE="/usr/local/bin/nvidia.sh"

    local gpu_clock_cmd mig_mode_cmd cpu_latency_cmd
    case "${PLATFORM:-}" in
        Supermicro_ARS-111GL-NHR)
            # GH200 requires --mode=1 to unlock the full clock range; query the hardware
            # maximum dynamically. Validate numeric result before calling -lgc.
            gpu_clock_cmd='{ _max_clk=$(nvidia-smi -i 0 --query-gpu=clocks.max.graphics --format=csv,noheader,nounits 2>/dev/null) && [[ "$_max_clk" =~ ^[0-9]+$ ]] && nvidia-smi -i 0 -lgc "$_max_clk" --mode=1; }'
            mig_mode_cmd='nvidia-smi -mig 0 || { echo "[ERROR] Failed to disable MIG mode"; FAILED=1; }'
            ;;
        NVIDIA_DGX_Spark_P4242)
            gpu_clock_cmd='nvidia-smi -lgc 2000'
            cpu_latency_cmd='systemctl start cpu-latency.service || { echo "[ERROR] Failed to start cpu-latency.service"; FAILED=1; }'
            ;;
        *)
            gpu_clock_cmd='nvidia-smi -lgc 2000'
            ;;
    esac

    echo_and_log "[INFO] GPU clock command: ${gpu_clock_cmd}"
    if [[ $DRYRUN -eq 1 ]]; then
        echo_and_log "[DRY-RUN] Would install platform-specific GPU and service commands in $DST_FILE"
    else
        # Escape sed replacement metacharacters (& means "matched text" in sed replacement)
        local escaped_cmd="${gpu_clock_cmd//&/\\&}"
        local escaped_mig_cmd="${mig_mode_cmd//&/\\&}"
        local escaped_cpu_latency_cmd="${cpu_latency_cmd//&/\\&}"
        sed -e "s#@GPU_CLOCK_CMD@#${escaped_cmd}#" \
            -e "s#@MIG_MODE_CMD@#${escaped_mig_cmd}#" \
            -e "s#@CPU_LATENCY_CMD@#${escaped_cpu_latency_cmd}#" \
            "$SRC_FILE" | sudo tee "$DST_FILE" > /dev/null
        sudo chmod +x "$DST_FILE"
        echo_and_log "  [install_nvidia_script] $DST_FILE"
    fi
    echo_and_log "[INFO] Installed $DST_FILE"
}

# Create nvidia.service systemd unit
create_nvidia_service() {
    echo_and_log "[INFO] Creating nvidia.service..."

    write_file /etc/systemd/system/nvidia.service << 'EOF'
[Unit]
Description=NVIDIA GPU and System Optimizations for Aerial
After=network.target openibd.service
Requires=openibd.service

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/local/bin/nvidia.sh

[Install]
WantedBy=default.target
EOF

    echo_and_log "[INFO] Created nvidia.service"
}

# Create nvidia-persistenced systemd service
install_nvidia_persistenced() {
    echo_and_log "[INFO] Creating nvidia-persistenced.service..."

    write_file /etc/systemd/system/nvidia-persistenced.service << 'EOF'
[Unit]
Description=NVIDIA Persistence Daemon
Wants=syslog.target

[Service]
Type=forking
ExecStart=/usr/bin/nvidia-persistenced
ExecStopPost=/bin/rm -rf /var/run/nvidia-persistenced

[Install]
WantedBy=multi-user.target
EOF

    echo_and_log "[INFO] Created nvidia-persistenced.service"
}

# Create CPU DMA latency service for low-latency operation
create_cpu_latency_service() {
    echo_and_log "[INFO] Creating cpu-latency.service..."

    write_file /etc/systemd/system/cpu-latency.service << 'EOF'
[Unit]
Description=Disable CPU DMA Latency
After=network.target

[Service]
Type=simple
# This keeps the shell open in the background holding the FD
ExecStart=/bin/bash -c "exec 3> /dev/cpu_dma_latency; echo 0 >&3; exec /usr/bin/sleep infinity"
Restart=always

[Install]
WantedBy=multi-user.target
EOF

    echo_and_log "[INFO] Created cpu-latency.service"
}

# Enable and start services
enable_services() {
    echo "[INFO] Enabling and starting services..."

    execute sudo systemctl daemon-reload

    # Restart RDMA/IB service first so nvidia.sh runs against a fresh, consistent module set.
    # Unload rpcrdma before restart so openibd's stop can unload rdma_cm (avoids "Module rdma_cm is in use by: rpcrdma").
    execute sudo modprobe -r rpcrdma 2>/dev/null || true
    execute sudo systemctl restart openibd.service

    # Enable and start nvidia service (GPU and system optimizations)
    execute sudo systemctl enable nvidia.service
    execute sudo systemctl restart nvidia.service
    execute sudo systemctl enable nvidia-persistenced.service
    execute sudo systemctl restart nvidia-persistenced.service


    if [[ $PLATFORM == "NVIDIA_DGX_Spark_P4242" ]]; then
        execute sudo systemctl enable cpu-latency.service
        execute sudo systemctl restart cpu-latency.service
    fi

    # Enable and start PTP services
    execute sudo systemctl enable ptp4l.service
    execute sudo systemctl enable phc2sys.service
    execute sudo systemctl restart ptp4l.service

    # Wait for ptp4l to start before starting phc2sys
    sleep 2
    execute sudo systemctl restart phc2sys.service

    echo "[INFO] All services enabled and started"
}

# Display service status
show_status() {
    echo ""
    echo_and_log "[INFO] nvidia service status:"
    systemctl status nvidia.service --no-pager --full || {
        echo_and_log "[WARN] Could not get nvidia status"
        FAILED=1
    }

    if [[ $PLATFORM == "NVIDIA_DGX_Spark_P4242" ]]; then
        echo ""
        echo_and_log "[INFO] cpu-latency service status:"
        systemctl status cpu-latency.service --no-pager --full || {
            echo_and_log "[WARN] Could not get cpu-latency status"
            FAILED=1
        }
    fi

    echo ""
    echo_and_log "[INFO] ptp4l service status:"
    systemctl status ptp4l.service --no-pager --full || {
        echo_and_log "[WARN] Could not get ptp4l status"
        FAILED=1
    }

    echo ""
    echo_and_log "[INFO] phc2sys service status:"
    systemctl status phc2sys.service --no-pager --full || {
        echo_and_log "[WARN] Could not get phc2sys status"
        FAILED=1
    }

    echo ""
    echo_and_log "[INFO] PTP configuration file:"
    if [[ -f /etc/ptp.conf ]]; then
        grep -E "^\[" /etc/ptp.conf || true
        echo_and_log "[INFO] /etc/ptp.conf exists and is configured"
    else
        echo_and_log "[WARN] /etc/ptp.conf not found"
        FAILED=1
    fi

    echo ""
    echo_and_log "[INFO] linuxptp version:"
    ptp4l -v | head -1 || {
        echo_and_log "[WARN] Could not get ptp4l version"
        FAILED=1
    }
}

# Extract RMS value from a log line
# Parameters:
#   $1 - log line containing "rms    VALUE max"
# Returns: outputs rms value, or empty if not found
extract_rms_value() {
    local log_line="$1"
    echo "$log_line" | grep -oP 'rms\s+\K[0-9]+'
}

extract_delay_value() {
    local log_line="$1"
    echo "$log_line" | grep -oP 'delay\s+\K[0-9]+'
}

# Check if a service is running
# Parameters:
#   $1 - service name
# Returns: 0 if running, 1 otherwise
check_service_running() {
    local service="$1"

    if ! systemctl is-active --quiet "$service" 2>/dev/null; then
        echo_and_log "[WARN] $service service is not running"
        FAILED=1
        return 1
    fi
    service_start_time=$(systemctl show -p ActiveEnterTimestamp --value "$service.service" 2>/dev/null)
    echo_and_log "[INFO] $service service is running since $service_start_time, time now is $(date)"
    return 0
}

# Get service logs (call journalctl once and cache)
# Parameters:
#   $1 - service name
#   $2 - minimum number of log lines required (default: 20)
#   $3 - max wait time in seconds (default: 30)
# Returns: 0 and outputs logs if successful, 1 otherwise
get_service_logs() {
    local service="$1"
    local min_lines="${2:-20}"
    local max_wait="${3:-30}"
    local invocation_id

    # Get the current invocation ID for this service instance
    invocation_id=$(systemctl show -p InvocationID --value "${service}.service" 2>/dev/null)

    if [[ -z $invocation_id ]]; then
        echo_and_log "[WARN] Could not get invocation ID for $service" >&2
        return 1
    fi

    local elapsed=0
    local log_count=0
    local logs=""

    # Wait for sufficient logs
    while [[ $elapsed -lt $max_wait ]]; do
        logs=$(journalctl _SYSTEMD_INVOCATION_ID="$invocation_id" --no-pager 2>/dev/null)
        log_count=$(echo "$logs" | wc -l)

        if [[ $log_count -ge $min_lines ]]; then
            echo "$logs"
            return 0
        fi

        if [[ $elapsed -eq 0 ]]; then
            echo_and_log "[INFO] Waiting for $service to generate sufficient logs ($log_count/$min_lines)..." >&2
        else
            echo_and_log "[INFO] Still waiting for $service logs... ($log_count/$min_lines, ${elapsed}s elapsed)" >&2
        fi

        sleep 10
        ((elapsed += 10))
    done

    echo_and_log "[WARN] $service only has $log_count log entries after ${max_wait}s (expected at least $min_lines)" >&2
    echo_and_log "[WARN] Proceeding with limited log data" >&2
    echo "$logs"
    return 1
}

# Check RMS values from log output
# Parameters:
#   $1 - service name
#   $2 - log output (from get_service_logs)
#   $3 - threshold value rms (ns)
#   $4 - threshold value delay (ns)
# Returns: 0 if check passes, 1 otherwise
check_service_timing_values() {
    local service="$1"
    local logs="$2"
    local threshold_rms="$3"
    local threshold_delay="$4"

    # Get only the most recent log line with rms data
    local last_line
    last_line=$(echo "$logs" | grep "rms" | tail -1)

    if [[ -z $last_line ]]; then
        echo_and_log "[WARN] No $service rms data found in journal"
        echo_and_log "[WARN] Check logs: journalctl -u $service -f"
        FAILED=1
        return 1
    fi

    # Extract and check the rms value
    local rms_val
    rms_val=$(extract_rms_value "$last_line")
    local delay_val
    delay_val=$(extract_delay_value "$last_line")

    if [[ -z $rms_val || -z $delay_val ]]; then
        echo_and_log "[WARN] Could not parse rms value or delay value from $service logs (rms='${rms_val:-}', delay='${delay_val:-}')"
        FAILED=1
        return 1
    fi

    # Compare against threshold
    if [[ $rms_val -lt $threshold_rms && $delay_val -lt $threshold_delay ]]; then
        echo_and_log "[INFO] Current $service locked: rms: ${rms_val}ns delay: ${delay_val}ns)"
        return 0
    else
        echo_and_log "[WARN] Current $service rms: ${rms_val}ns delay: ${delay_val}ns (expected < ${threshold_rms}ns rms and < ${threshold_delay}ns delay when locked)"
        return 1
    fi
}

# Check ptp4l master clock selection from log output
# Parameters:
#   $1 - log output (from get_service_logs)
# Returns: 0 if found, 1 otherwise
check_ptp4l_master_selection() {
    local logs="$1"
    local best_master
    best_master=$(echo "$logs" | grep "selected best master clock" | tail -1)

    if [[ -n $best_master ]]; then
        echo_and_log "[INFO] Best master clock selected: $(echo "$best_master" | grep -oP 'selected best master clock \K.*' || echo "found")"
        return 0
    else
        echo_and_log "[WARN] No 'selected best master clock' found in recent logs"
        echo_and_log "[WARN] PTP may not be synchronized to a grandmaster"
        FAILED=1
        return 1
    fi
}

# Check PTP sync status
check_ptp_sync() {
    echo_and_log "[INFO] Checking PTP synchronization status..."

    # Check ptp4l service
    check_service_running "ptp4l" || return

    local ptp4l_logs
    ptp4l_logs=$(get_service_logs "ptp4l" 20 30)

    if [[ -n $ptp4l_logs ]]; then
        check_ptp4l_master_selection "$ptp4l_logs"
        check_service_timing_values "ptp4l" "$ptp4l_logs" 100 150
    else
        echo_and_log "[ERROR] Could not retrieve ptp4l logs"
        FAILED=1
    fi

    # Check phc2sys service
    check_service_running "phc2sys" || return

    local phc2sys_logs
    phc2sys_logs=$(get_service_logs "phc2sys" 20 30)

    if [[ -n $phc2sys_logs ]]; then
        #Allow a little leeway on startup
        check_service_timing_values "phc2sys" "$phc2sys_logs" 50 1000
    else
        echo_and_log "[ERROR] Could not retrieve phc2sys logs"
        FAILED=1
    fi
}

# Main execution
main() {
    echo "============================================"
    echo_and_log "NVIDIA Aerial Services Installation"
    echo "============================================"

    # Handle --detect-ptp flag: just run detection and exit
    if [[ $DETECT_PTP_ONLY -eq 1 ]]; then
        detect_ptp_traffic
        exit $?
    fi

    # Handle --status flag: just show status and exit
    if [[ $STATUS_ONLY -eq 1 ]]; then
        show_status
        exit $FAILED
    fi

    # Handle --check-sync flag: just check PTP sync and exit
    if [[ $CHECK_SYNC_ONLY -eq 1 ]]; then
        check_ptp_sync
        exit $FAILED
    fi

    disable_ntp
    install_linuxptp

    # Detect which interface has PTP traffic
    detect_ptp_traffic || {
        echo_and_log "[WARN] Continuing with default interface: ${FH_INTERFACE}"
        PTP_INTERFACE="$FH_INTERFACE"
    }
    echo_and_log "[INFO] Using PTP interface: ${PTP_INTERFACE}"
    echo ""

    create_ptp_conf
    create_ptp4l_service
    create_phc2sys_service
    install_nvidia_script
    create_nvidia_service
    install_nvidia_persistenced
    if [[ $PLATFORM == "NVIDIA_DGX_Spark_P4242" ]]; then
        create_cpu_latency_service
    fi
    enable_services

    echo ""
    echo "============================================"
    echo_and_log "Installation complete! Displaying status..."
    echo "============================================"

    show_status

    echo ""
    echo "============================================"
    if [[ $FAILED -eq 1 ]]; then
        echo_and_log "[ERROR] Service installation completed with warnings/errors"
        exit 1
    else
        echo_and_log "[INFO] Service installation completed successfully!"
        echo_and_log "[INFO] To verify PTP sync after connecting a grandmaster: make check"
        echo "============================================"
    fi
}

main "$@"
