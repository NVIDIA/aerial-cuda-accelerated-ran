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

# Makefile for NVIDIA Aerial SDK Installation
#
# Usage:
#   make all          - Run full installation (requires reboot after kernel step)
#   make net          - Setup network interfaces (aerial0x)
#   make kernel       - Install Aerial CUDA kernel
#   make drivers      - Install DOCA, OFED, and GPU drivers
#   make services     - Install PTP and system services
#   make check        - Check installation status
#   make clean        - Remove build artifacts (../../build.$ARCH)
#   make help         - Show this help
#
# Options:
#   DRYRUN=1          - Show commands without executing
#   VERBOSE=1         - Print commands before executing
#   ISOLCPUS=LIST     - CPU list for kernel isolation (e.g. 4, 4-15, 2,4,5,7,8). Default: 4-19
#
# Example:
#   make all DRYRUN=1   - Preview full installation
#   make drivers        - Install only drivers
#

SHELL := /bin/bash
SCRIPT_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
# Current-arch build dir (aarch64 -> arm64 to match versions.sh)
ARCH := $(shell uname -m)
BUILD_ARCH := $(if $(filter aarch64,$(ARCH)),arm64,$(ARCH))
BUILD_DIR := $(SCRIPT_DIR)../../build.$(BUILD_ARCH)/

# Stamp files: created after each install step so make can skip steps already done.
# Stamps depend on versions.sh so any version bump automatically re-triggers the step.
STAMP_DIR := $(SCRIPT_DIR).stamps
STAMP_NET     := $(STAMP_DIR)/net
STAMP_KERNEL  := $(STAMP_DIR)/kernel
STAMP_DRIVERS := $(STAMP_DIR)/drivers
STAMP_NIC_FW  := $(STAMP_DIR)/nic_fw
STAMP_SERVICES := $(STAMP_DIR)/services

# Pass through options to scripts
OPTS :=
ifdef DRYRUN
	OPTS += --dry-run
endif
ifdef VERBOSE
	OPTS += --verbose
endif

# Environment variables to pass through
ENV_VARS :=
ifdef RU_MAC
	ENV_VARS += RU_MAC=$(RU_MAC)
endif
ifdef PROFILE
	ENV_VARS += PROFILE=$(PROFILE)
endif
ifdef BUILD_PRESET
	ENV_VARS += BUILD_PRESET=$(BUILD_PRESET)
endif
ifdef BUILD_CMAKE_FLAGS
	ENV_VARS += BUILD_CMAKE_FLAGS="$(BUILD_CMAKE_FLAGS)"
endif
ifdef AERIAL_BUILD_FLAGS
	ENV_VARS += AERIAL_BUILD_FLAGS="$(AERIAL_BUILD_FLAGS)"
endif

.PHONY: prepare all install build build_aerial build_oai net kernel drivers nic_fw services check help clean start_gnb start_cn start_all

$(STAMP_DIR):
	@mkdir -p $@

prepare: kernel
	@echo ""
	@echo "============================================"
	@echo "Prepare complete! Please reboot, and after startup:"
	@echo "To only install drivers, run: make install"
	@echo "To also build and start the gNB and CN5G, run: make all"
	@echo "============================================"
	@echo ""

all: install build start_all

install: services
	@echo ""
	@echo "============================================"
	@echo "Install complete!"
	@echo "============================================"
	@echo ""

# net: runs after drivers (requires OFED/mlx5 to be installed for aerial0x interfaces)
net: $(STAMP_NET)
$(STAMP_NET): $(SCRIPT_DIR)setup_net_ifs.sh $(SCRIPT_DIR)versions.sh $(STAMP_DRIVERS) | $(STAMP_DIR)
	@echo "========================================"
	@echo "Setting up network interfaces"
	@echo "========================================"
	$(SCRIPT_DIR)/setup_net_ifs.sh $(OPTS)
	@touch $@

# kernel: re-runs if install_aerial_kernel.sh or versions.sh changes
kernel: $(STAMP_KERNEL)
$(STAMP_KERNEL): $(SCRIPT_DIR)install_aerial_kernel.sh $(SCRIPT_DIR)versions.sh | $(STAMP_DIR)
	@echo "========================================"
	@echo "Installing Aerial CUDA kernel"
	@echo "========================================"
	$(if $(ISOLCPUS),$(info Using ISOLCPUS=$(ISOLCPUS)))
	$(SCRIPT_DIR)/install_aerial_kernel.sh $(OPTS) $(if $(ISOLCPUS),--isolcpus $(ISOLCPUS))
	@touch $@
	@echo "NOTE: If kernel was updated, reboot and re-run:"
	@echo "  make all"
	@echo ""

# drivers: re-runs if install_drivers.sh or versions.sh changes; kernel must be stamped first
drivers: $(STAMP_DRIVERS)
$(STAMP_DRIVERS): $(SCRIPT_DIR)install_drivers.sh $(SCRIPT_DIR)versions.sh $(STAMP_KERNEL) | $(STAMP_DIR)
	@echo "========================================"
	@echo "Installing drivers"
	@echo "========================================"
	@echo ""
	@echo "IMPORTANT: Have you configured the following?"
	@echo "  - PTP settings"
	@echo "  - VLAN on RU and gNB interfaces"
	@echo "  - Peer MAC address on RU"
	@echo "  - Docker login if necessary"
	@echo "Without these the gNB will not deploy properly."
	@echo ""
	@read -p "Continue with driver installation? (Y/n): " confirm && \
	[[ -z $$confirm || $$confirm == [yY] || $$confirm == [yY][eE][sS] ]] || { echo "Installation cancelled."; exit 1; }
	@echo ""
	$(SCRIPT_DIR)/install_aerial_kernel.sh --remove-other-kernels $(OPTS)
	$(SCRIPT_DIR)/install_drivers.sh $(OPTS)
	@touch $@

# nic_fw: verifies and configures NIC firmware; installs the BlueField3 BFB on GH200 when the firmware version differs from the expected version
# Depends on drivers since flint, mlxconfig, and bfb-install require DOCA/OFED to be present
nic_fw: $(STAMP_NIC_FW)
$(STAMP_NIC_FW): $(SCRIPT_DIR)install_nic_fw.sh $(SCRIPT_DIR)versions.sh $(STAMP_DRIVERS) | $(STAMP_DIR)
	@echo "========================================"
	@echo "Verifying and configuring NIC firmware"
	@echo "========================================"
	$(SCRIPT_DIR)/install_nic_fw.sh $(OPTS)
	@touch $@

# services: runs after both net (needs aerial0x) and nic_fw (needs NIC firmware/configuration)
services: $(STAMP_SERVICES)
$(STAMP_SERVICES): $(SCRIPT_DIR)install_services.sh $(SCRIPT_DIR)versions.sh $(STAMP_NET) $(STAMP_NIC_FW) | $(STAMP_DIR)
	@echo "========================================"
	@echo "Installing services"
	@echo "========================================"
	$(SCRIPT_DIR)/install_services.sh $(OPTS)
	@touch $@

build: build_aerial build_oai

build_aerial:
	@echo "========================================"
	@echo "Building Aerial SDK"
	@echo "========================================"
	LOGFILE=$(SCRIPT_DIR)build-aerial.log $(ENV_VARS) $(SCRIPT_DIR)/quickstart-aerial.sh $(OPTS)

build_oai:
	@echo "========================================"
	@echo "Building OAI"
	@echo "========================================"
	LOGFILE=$(SCRIPT_DIR)build-oai.log $(ENV_VARS) $(SCRIPT_DIR)/quickstart-oai.sh --build-only $(OPTS)

start_gnb:
	@echo "========================================"
	@echo "Starting gNB"
	@echo "========================================"
	@if [ -z "$(RU_MAC)" ]; then \
		echo "  RU_MAC is not set - will use default"; \
	else \
		echo "Using RU_MAC=$(RU_MAC)"; \
	fi
	$(ENV_VARS) $(SCRIPT_DIR)/quickstart-oai.sh --start gnb $(OPTS)

start_cn:
	@echo "========================================"
	@echo "Starting CN5G"
	@echo "========================================"
	$(ENV_VARS) $(SCRIPT_DIR)/quickstart-oai.sh --start cn $(OPTS)

start_all:
	@echo "========================================"
	@echo "Starting gNB and CN5G"
	@if [ -z "$(RU_MAC)" ]; then \
		echo "  RU_MAC is not set - will use default"; \
		read -p "Continue with gNB starting? (Y/n): " confirm && \
		[[ -z $$confirm || $$confirm == [yY] || $$confirm == [yY][eE][sS] ]] || { echo "Starting cancelled."; exit 1; }; \
	else \
		echo "Using RU_MAC=$(RU_MAC)"; \
	fi
	@echo "========================================"
	$(ENV_VARS) $(SCRIPT_DIR)/quickstart-oai.sh --start $(OPTS)

check:
	@echo "========================================"
	@echo "Checking installation status"
	@echo "========================================"
	@echo ""
	@echo "--- Kernel ---"
	$(SCRIPT_DIR)/install_aerial_kernel.sh --check $(OPTS) || true
	@echo ""
	@echo "--- Services ---"
	$(SCRIPT_DIR)/install_services.sh --check-sync --status $(OPTS) || true

help:
	@echo "NVIDIA Aerial SDK Installation"
	@echo ""
	@echo "Targets:"
	@echo "  all       - Full end-to-end: install + build + start"
	@echo "  prepare   - Install kernel (reboot required after)"
	@echo "  install   - Post-reboot: drivers + net + nic_fw + services"
	@echo "  kernel    - Install Aerial CUDA kernel"
	@echo "  drivers   - Install DOCA, OFED, and GPU drivers"
	@echo "  net       - Setup aerial0x network interfaces (requires OFED)"
	@echo "  nic_fw    - Verify/configure NIC firmware; install the GH200 BFB on firmware version mismatch"
	@echo "  services  - Install PTP and system services"
	@echo "  check     - Check installation status"
	@echo "  build     - Build Aerial SDK and OAI"
	@echo "  build_aerial - Build Aerial SDK (use PROFILE= for variant)"
	@echo "  build_oai  - Build OAI (use RU_MAC= if needed)"
	@echo "  start_gnb  - Start gNB (use RU_MAC=)"
	@echo "  start_cn   - Start CN5G"
	@echo "  start_all  - Start gNB and CN5G"
	@echo "  clean      - Remove build artifacts (../../build.\$$ARCH)"
	@echo "  help      - Show this help"
	@echo ""
	@echo "Options (pass as make variables):"
	@echo "  DRYRUN=1              - Show commands without executing"
	@echo "  VERBOSE=1             - Print commands before executing"
	@echo "  ISOLCPUS=LIST         - CPU list for isolation (e.g. 4, 4-15, 2,4,5,7,8); default 4-19"
	@echo "  RU_MAC=aa:bb:cc:dd:ee:ff - Set RU MAC address for gNB"
	@echo "  PROFILE=name          - Aerial build profile file (oai.conf, fapi_10_02.conf, fapi_10_04.conf; see cmake-profiles/)"
	@echo "  BUILD_PRESET=preset   - Override preset (perf, 10_02, 10_04, 10_04_32dl)"
	@echo "  BUILD_CMAKE_FLAGS=... - Additional CMake flags not defined in cmake-profiles/"
	@echo "  AERIAL_BUILD_FLAGS=... - Override build_aerial_sdk.sh args before -- (from versions.sh)"
	@echo ""
	@echo "Logging: install steps log to /var/log/aerial-installer.log (override with LOGFILE=); build steps to build-<target>.log."
	@echo ""
	@echo "Stamp files (.stamps/): each install step creates a stamp so make skips it if already done."
	@echo "  Stamps are invalidated automatically when versions.sh or the install script changes."
	@echo "  To force a step to re-run: rm .stamps/<target>   (e.g. rm .stamps/drivers)"
	@echo "  To force everything to re-run: make clean"
	@echo ""
	@echo "Examples:"
	@echo "  make all              # Full installation"
	@echo "  make all DRYRUN=1     # Preview full installation"
	@echo "  make kernel           # Install only kernel"
	@echo "  make kernel ISOLCPUS=4-15   # Install kernel with custom isolated CPUs"
	@echo "  make drivers services # Install drivers and services"
	@echo "  RU_MAC=e8:c7:cf:ac:58:32 make build_oai  # Build OAI with custom RU MAC"
	@echo "  make build_aerial                      # Aerial build, FAPI 10.02 with SRS 10.04"
	@echo "  PROFILE=fapi_10_02.conf make build_aerial   # Aerial build, FAPI 10.02"
	@echo "  PROFILE=fapi_10_04.conf make build_aerial   # Aerial build with FAPI 10.04"
	@echo ""
	@echo "Recommended flow:"
	@echo "  1. make prepare       # Install kernel"
	@echo "  2. sudo reboot        # Reboot to load new kernel"
	@echo "  3. make install       # Install drivers, net, NIC firmware, and services"
	@echo "  4. make check         # Verify installation"
	@echo ""

clean:
	@echo "========================================"
	@echo "Cleaning build artifacts"
	@echo "========================================"
	@rm -rf $(BUILD_DIR)
	@echo "Removed ../../build.$(BUILD_ARCH)/"
	@rm -rf $(STAMP_DIR)
	@echo "Removed install stamps (next make will re-run all install steps)"
