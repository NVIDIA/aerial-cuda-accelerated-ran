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
# nvidia.sh - GPU and system optimizations for NVIDIA Aerial
#

FAILED=0

# Start Mellanox Software Tools
mst start || { echo "[ERROR] Failed to start MST"; FAILED=1; }

# Bring up aerial0* interfaces and set ethtool options (detect which exist)
AERIAL_INTERFACES=$(ip -o link show 2>/dev/null | grep -oE 'aerial0[0-9]+' | sort -u)
for iface in $AERIAL_INTERFACES; do
    ip link set "$iface" up || { echo "[WARN] Failed to bring up $iface"; }
    ethtool --set-priv-flags "$iface" tx_port_ts on 2>/dev/null || echo "[WARN] tx_port_ts not set on $iface (may not be supported)"
    ethtool -A "$iface" rx off tx off 2>/dev/null || true
done

# Force max frequency on all GPUs (command is substituted per platform during install)
@GPU_CLOCK_CMD@ || { echo "[ERROR] Failed to lock GPU frequency"; FAILED=1; }

# Configure MIG mode when required by the platform guide
@MIG_MODE_CMD@

# Allow real-time tasks to take 100% CPU
echo -1 > /proc/sys/kernel/sched_rt_runtime_us || { echo "[ERROR] Failed to configure RT scheduling"; FAILED=1; }

# Disable timer migration
echo 0 | tee /proc/sys/kernel/timer_migration > /dev/null || { echo "[ERROR] Failed to disable timer migration"; FAILED=1; }

# Pin all RCU processes to core 1
[[ -x /usr/local/bin/rcu_affinity_manager.sh ]] && \
    /usr/local/bin/rcu_affinity_manager.sh -w -c 1 || echo "[WARN] RCU pinning failed (non-fatal)"

# Enable DPDK mapping of GPU memory
modprobe nvidia-peermem 2>/dev/null || lsmod | grep -q nvidia_peermem || { echo "[ERROR] Failed to load nvidia-peermem"; FAILED=1; }

# Start the CPU DMA latency service on platforms that require it
@CPU_LATENCY_CMD@

# Exit with status
[[ $FAILED -eq 1 ]] && { echo "[ERROR] Completed with errors"; exit 1; }
echo "[INFO] Completed successfully"
exit 0
