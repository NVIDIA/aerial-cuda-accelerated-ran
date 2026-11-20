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
# quickstart-oai.sh - Quick start script for OpenAirInterface with Aerial
#
# This script sets up and runs OpenAirInterface (OAI) gNB with NVIDIA Aerial
# acceleration, including the 5G Core Network.
#
# Usage:
#   ./quickstart-oai.sh                  # Full setup: clone, build, and run
#   ./quickstart-oai.sh --build-only     # Only build Docker images
#   ./quickstart-oai.sh --start          # Start containers (assumes built)
#   ./quickstart-oai.sh --dry-run        # Show commands without executing
#

# Identify SCRIPT_DIR and SDK root
SCRIPT=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT")
[[ -f "$SCRIPT_DIR/includes.sh" ]] && source "$SCRIPT_DIR/includes.sh" || { echo "ERROR: includes.sh not found: $SCRIPT_DIR/includes.sh" >&2; exit 1; }

# Log to current directory (override default /var/log/...)
LOGFILE="quickstart-oai.log"

# Configuration
OAI_BRANCH="${OAI_BRANCH:-ATB1.2_integration}"
OAI_REPO="${OAI_REPO:-https://github.com/duranta-project/openairinterface5g.git}"
OAI_DIR="${OAI_DIR:-$HOME/openairinterface5g}"
GNB_DIR="$OAI_DIR/ci-scripts/yaml_files/sa_gnb_aerial"
CN5G_DIR="$OAI_DIR/doc/tutorial_resources/oai-cn5g"
AERIAL_SDK_DIR="${AERIAL_SDK_DIR:-$HOME/aerial-cuda-accelerated-ran}"

# O-RU destination in cuphycontroller YAML is an Ethernet MAC (dst_mac_addr)
is_ru_mac_format() {
    [[ "${1:-}" =~ ^([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}$ ]]
}

# Expected Docker build times per platform (in minutes)
declare -A DOCKER_BUILD_TIMES=(
    ["DGX-Spark"]=70
)

# CN5G cpuset configuration per platform
# Format: "service:cpuset,service:cpuset,...,default:cpuset"
declare -A CN5G_CPUSET_CONFIG=(
    ["DGX-Spark"]="oai-amf:4,oai-upf:12-14,default:11"
    ["SMC-GraceHopper"]="oai-upf:42-43,default:41"
)

# gNB cpuset configuration per platform
# Format: "service:cpuset,service:cpuset,...,default:cpuset"
declare -A GNB_CPUSET_CONFIG=(
    ["DGX-Spark"]="oai-gnb-aerial:17-19"
)

# Help function
show_help() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Quick start script for OpenAirInterface with NVIDIA Aerial acceleration.
Sets up OAI gNB and 5G Core Network with Aerial GPU acceleration.

OPTIONS:
    --build-only         Only clone repo and build Docker images
    --start [gnb|cn]     Start containers (all, or just gnb/cn)
    --stop [gnb|cn]      Stop containers (all, or just gnb/cn)
    --logs               Show nv-cubb container logs after starting
    --no-logs            Don't show logs after starting (default shows logs)
    --dry-run            Show commands without executing
    --verbose            Enable verbose output
    -h, --help           Display this help message and exit

ENVIRONMENT VARIABLES:
    OAI_BRANCH           OAI git branch (default: ATB1.2_integration)
    OAI_REPO             OAI git repository (default: https://github.com/duranta-project/openairinterface5g.git)
    OAI_DIR              OAI clone directory (default: ~/openairinterface5g)
    AERIAL_SDK_DIR       Aerial SDK directory (default: ~/aerial-cuda-accelerated-ran)
    RU_MAC               O-RU Ethernet MAC for L1; yq sets it on docker-compose service nv-cubb

EXAMPLES:
    # Full setup from scratch
    $(basename "$0")

    # Only build the Docker images
    $(basename "$0") --build-only

    # Start all containers (assumes images already built)
    $(basename "$0") --start

    # Start only gNB
    $(basename "$0") --start gnb

    # Start only 5G Core Network
    $(basename "$0") --start cn

    # Stop all containers
    $(basename "$0") --stop

    # Stop only gNB containers
    $(basename "$0") --stop gnb

    # Stop only 5G Core Network
    $(basename "$0") --stop cn

    # Preview commands without executing
    $(basename "$0") --dry-run

EOF
}

# Parse common arguments (--dry-run, --verbose) and populate REMAINING_ARGS
parse_common_args "$@"
init_docker_cmd

# Parse script-specific arguments from REMAINING_ARGS
BUILD_ONLY=0
RUN_ONLY=0
CN5G_ONLY=0
GNB_ONLY=0
STOP_MODE=""
SHOW_LOGS=1

i=0
while [[ $i -lt ${#REMAINING_ARGS[@]} ]]; do
    arg="${REMAINING_ARGS[$i]}"
    case $arg in
        --build-only)
            BUILD_ONLY=1
            ;;
        --cn5g-only)
            CN5G_ONLY=1
            ;;
        --gnb-only)
            GNB_ONLY=1
            ;;
        --start)
            RUN_ONLY=1
            # Check if next arg is gnb or cn
            next_idx=$((i + 1))
            if [[ $next_idx -lt ${#REMAINING_ARGS[@]} ]]; then
                next_arg="${REMAINING_ARGS[$next_idx]}"
                case $next_arg in
                    gnb)
                        GNB_ONLY=1
                        RUN_ONLY=0
                        ((++i))
                        ;;
                    cn)
                        CN5G_ONLY=1
                        RUN_ONLY=0
                        ((++i))
                        ;;
                esac
            fi
            ;;
        --stop)
            STOP_MODE="all"
            # Check if next arg is gnb or cn
            next_idx=$((i + 1))
            if [[ $next_idx -lt ${#REMAINING_ARGS[@]} ]]; then
                next_arg="${REMAINING_ARGS[$next_idx]}"
                case $next_arg in
                    gnb)
                        STOP_MODE="gnb"
                        ((++i))
                        ;;
                    cn)
                        STOP_MODE="cn"
                        ((++i))
                        ;;
                esac
            fi
            ;;
        --logs)
            SHOW_LOGS=1
            ;;
        --no-logs)
            SHOW_LOGS=0
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $arg"
            show_help
            exit 1
            ;;
    esac
    ((++i))
done

echo "=============================================="
echo "OpenAirInterface + Aerial Quickstart"
echo "=============================================="
echo "OAI Branch:         $OAI_BRANCH"
echo "OAI Directory:      $OAI_DIR"
echo "Aerial SDK:         $AERIAL_SDK_DIR"
echo "Hostname:           $(hostname)"
echo "=============================================="
echo ""

# Function to clone and setup OAI repository
setup_oai_repo() {
    echo "[STEP] Cloning OpenAirInterface repository..."

    if [[ -d $OAI_DIR ]]; then
        echo "OAI directory already exists: $OAI_DIR"
        echo "Updating to branch $OAI_BRANCH..."
        execute_or_die "cd $OAI_DIR && git fetch && git checkout $OAI_BRANCH && git pull"
    else
        execute_or_die "git clone --branch $OAI_BRANCH $OAI_REPO $OAI_DIR"
    fi

    echo "[STEP] Copying NVIPC source tarball..."
    local nvipc_tarball
    local -a nvipc_tarballs
    local f
    shopt -s nullglob
    nvipc_tarballs=("$AERIAL_SDK_DIR/cuPHY-CP/gt_common_libs/nvipc_src."*.tar.gz)
    shopt -u nullglob
    for f in "${nvipc_tarballs[@]}"; do
        if [[ -z $nvipc_tarball || $f -nt $nvipc_tarball ]]; then
            nvipc_tarball=$f
        fi
    done

    if [[ -z $nvipc_tarball ]]; then
        echo "[ERROR] NVIPC tarball not found in $AERIAL_SDK_DIR/cuPHY-CP/gt_common_libs/"
        echo "Run quickstart-aerial.sh first to generate the tarball"
        exit 1
    fi

    echo "Using NVIPC tarball: $nvipc_tarball"
    execute "rm -rf $OAI_DIR/nvipc_src* 2>/dev/null"
    execute_or_die "cp $nvipc_tarball $OAI_DIR/"
}

# Function to build Docker images
build_docker_images() {
    echo -n "[STEP] Building OAI Docker images. "

    if [[ -n ${DOCKER_BUILD_TIMES[$PLATFORM_ID]} ]]; then
        echo "Expected build time for $PLATFORM_ID: ${DOCKER_BUILD_TIMES[$PLATFORM_ID]} minutes."
    else
        echo "Expected build time: unknown (platform: $PLATFORM_ID)"
    fi

    execute_or_die "cd $OAI_DIR && docker build . -f docker/Dockerfile.base.ubuntu --tag ran-base:latest"
    execute_or_die "cd $OAI_DIR && docker build . -f docker/Dockerfile.gNB.aerial.ubuntu --tag oai-gnb-aerial:latest"

    echo "Docker images built successfully:"
    echo "  - ran-base:latest"
    echo "  - oai-gnb-aerial:latest"
}

# Function to configure CN5G cpuset values using yq
configure_cn5g() {
    echo "[STEP] Configuring CN5G cpuset values..."
    if [[ ! -d $CN5G_DIR ]]; then
        echo "[ERROR] CN5G directory not found: $CN5G_DIR"
        exit 1
    fi

    local compose_file="$CN5G_DIR/docker-compose.yaml"
    if [[ ! -f $compose_file ]]; then
        echo "[ERROR] docker-compose.yaml not found: $compose_file"
        exit 1
    fi

    # Check if yq is available
    if ! command -v yq &> /dev/null; then
        echo "[ERROR] yq is required but not installed"
        exit 1
    fi

    # Check if platform has cpuset configuration
    if [[ -z ${CN5G_CPUSET_CONFIG[$PLATFORM_ID]} ]]; then
        echo "[WARN] No cpuset configuration for platform: $PLATFORM_ID, skipping"
        return
    fi

    # Parse the platform-specific cpuset configuration into an associative array
    local config_string="${CN5G_CPUSET_CONFIG[$PLATFORM_ID]}"
    declare -A cpuset_map
    IFS=',' read -ra config_entries <<< "$config_string"
    for entry in "${config_entries[@]}"; do
        local key="${entry%%:*}"
        local value="${entry#*:}"
        cpuset_map["$key"]="$value"
    done

    # Get default cpuset value
    local default_cpuset="${cpuset_map[default]:-0}"

    # Get all service names
    local services
    services=$(yq -r '.services | keys | .[]' "$compose_file") || { echo "[ERROR] failed to get services from $compose_file"; exit 1; }

    # Set cpuset for each service
    for service in $services; do
        local cpuset="${cpuset_map[$service]:-$default_cpuset}"
        echo "  Setting cpuset for $service to $cpuset"
        execute_or_die "yq -i --unwrapScalar=false '.services.\"$service\".cpuset = \"$cpuset\"' \"$compose_file\""
    done

    echo "[STEP] CN5G cpuset configuration complete"
}

# Function to configure gNB cpuset values using yq
configure_gnb() {
    echo "[STEP] Configuring gNB cpuset values..."
    if [[ ! -d $GNB_DIR ]]; then
        echo "[ERROR] gNB directory not found: $GNB_DIR"
        exit 1
    fi

    local compose_file="$GNB_DIR/docker-compose.yaml"
    if [[ ! -f $compose_file ]]; then
        echo "[ERROR] docker-compose.yaml not found: $compose_file"
        exit 1
    fi

    # Check if yq is available
    if ! command -v yq &> /dev/null; then
        echo "[ERROR] yq is required but not installed"
        exit 1
    fi

    # Check if platform has cpuset configuration
    if [[ -z ${GNB_CPUSET_CONFIG[$PLATFORM_ID]} ]]; then
        echo "[WARN] No gNB cpuset configuration for platform: $PLATFORM_ID, skipping"
        return
    fi

    # Parse the platform-specific cpuset configuration into an associative array
    local config_string="${GNB_CPUSET_CONFIG[$PLATFORM_ID]}"
    declare -A cpuset_map
    IFS=',' read -ra config_entries <<< "$config_string"
    for entry in "${config_entries[@]}"; do
        local key="${entry%%:*}"
        local value="${entry#*:}"
        cpuset_map["$key"]="$value"
    done

    # Get default cpuset value (empty means don't set)
    local default_cpuset="${cpuset_map[default]:-}"

    # Get all service names
    local services
    services=$(yq -r '.services | keys | .[]' "$compose_file") || { echo "[ERROR] failed to get services from $compose_file"; exit 1; }

    # Set cpuset for each service that has a configuration
    for service in $services; do
        local cpuset="${cpuset_map[$service]:-$default_cpuset}"
        if [[ -n $cpuset ]]; then
            echo "  Setting cpuset for $service to $cpuset"
            execute_or_die "yq -i --unwrapScalar=false '.services.\"$service\".cpuset = \"$cpuset\"' \"$compose_file\""
        fi
    done

    # Update tr_s_poll_core in gNB config file to match first core of oai-gnb-aerial cpuset
    local gnb_cpuset="${cpuset_map[oai-gnb-aerial]:-}"
    if [[ -n $gnb_cpuset ]]; then
        # Extract first core from cpuset (e.g., "17-19" -> "17", "4" -> "4")
        local first_core="${gnb_cpuset%%-*}"

        # Find the config file path from the volume mount that maps to /opt/oai-gnb/etc/gnb.conf
        local volume_mount
        volume_mount=$(yq eval '.services."oai-gnb-aerial".volumes[] | select(test(":/opt/oai-gnb/etc/gnb.conf$"))' "$compose_file") || { echo "[ERROR] failed to get gnb.conf volume mount from $compose_file"; exit 1; }
        if [[ -n $volume_mount ]]; then
            # Extract source path (part before the colon)
            local config_rel_path="${volume_mount%%:*}"
            # Resolve relative to GNB_DIR
            local gnb_conf
            gnb_conf=$(cd "$GNB_DIR" && readlink -f "$config_rel_path") || { echo "[ERROR] failed to resolve gNB config path: $config_rel_path"; exit 1; }
            if [[ -f $gnb_conf ]]; then
                echo "  Setting tr_s_poll_core to $first_core in $gnb_conf"
                execute_or_die "sed -i \"s/tr_s_poll_core[[:space:]]*=[[:space:]]*[0-9]*/tr_s_poll_core   = $first_core/\" \"$gnb_conf\""
            else
                echo "[WARN] gNB config file not found: $gnb_conf"
            fi
        else
            echo "[WARN] Could not find gnb.conf volume mount in docker-compose.yaml"
        fi
    fi

    echo "[STEP] gNB cpuset configuration complete"
}

# Set RU_MAC on gNB L1 service nv-cubb in docker-compose (list or map environment).
configure_gnb_compose_ru_mac() {
    local f="$GNB_DIR/docker-compose.yaml" _failed_before=$FAILED
    echo "[STEP] Setting RU_MAC in docker-compose service nv-cubb."
    export RU_MAC="$1"
    if [[ $(yq -r '.services.nv-cubb.environment | tag' "$f") == "!!seq" ]]; then
        # List env has no keys; drop prior RU_MAC= lines so we do not stack duplicates on re-run.
        execute "yq -i 'del(.services.nv-cubb.environment[] | select(test(\"^RU_MAC=\")))' \"$f\""
        execute "yq -i '.services.nv-cubb.environment += [\"RU_MAC=\" + strenv(RU_MAC)]' \"$f\""
    else
        # Map env: set or overwrite RU_MAC in one step.
        execute "yq -i '.services.nv-cubb.environment.RU_MAC = strenv(RU_MAC)' \"$f\""
    fi
    if [[ $FAILED -ne $_failed_before ]]; then
        echo "[WARN] Could not update RU_MAC in $f (service nv-cubb); set it in compose manually if the container needs it." >&2
        FAILED=$_failed_before
    fi
}

# Function to start 5G Core Network
start_cn5g() {
    echo "[STEP] Starting 5G Core Network..."
    if [[ ! -d $CN5G_DIR ]]; then
        echo "[ERROR] CN5G directory not found: $CN5G_DIR"
        exit 1
    fi

    configure_cn5g
    execute_or_die "cd $CN5G_DIR && docker compose up -d"
    echo "5G Core Network started"
}

# Function to configure and start gNB with Aerial
start_gnb_aerial() {
    echo "[STEP] Configuring and starting gNB with Aerial..."
    if [[ ! -d $GNB_DIR ]]; then
        echo "[ERROR] gNB Aerial directory not found: $GNB_DIR"
        exit 1
    fi

    # Create .env file
    echo "[STEP] Creating .env file for gNB..."
    execute_or_die "cd $GNB_DIR && echo 'REGISTRY=\"\"' > .env && echo 'TAG=\"latest\"' >> .env"

    # Configure cpuset values
    configure_gnb

    # RU_MAC: compose env for the container; run_l1.sh applies it to cuphycontroller YAML.
    if [[ -n ${RU_MAC:-} ]]; then
        if ! is_ru_mac_format "$RU_MAC"; then
            echo "[ERROR] RU_MAC must be an Ethernet MAC like aa:bb:cc:dd:ee:ff (for cuphycontroller dst_mac_addr)." >&2
            echo "[ERROR] Got: $RU_MAC" >&2
            exit 1
        fi
        configure_gnb_compose_ru_mac "$RU_MAC"
    fi

    # Show git diff for review
    echo "[STEP] Configuration changes:"
    execute "cd $GNB_DIR && git --no-pager diff -b ."

    # Start gNB containers
    echo "[STEP] Starting gNB containers..."
    execute_or_die "cd $GNB_DIR && docker compose up -d"

    echo "gNB with Aerial started"
}

# Function to stop 5G Core Network
stop_cn5g() {
    echo "[STEP] Stopping 5G Core Network..."
    if [[ ! -d $CN5G_DIR ]]; then
        echo "[WARN] CN5G directory not found: $CN5G_DIR"
        return
    fi
    if ! pushd "$CN5G_DIR" > /dev/null; then
        echo "[ERROR] pushd failed: $CN5G_DIR"
        exit 1
    fi
    echo "[INFO] Running: docker compose down"
    execute docker compose down
    popd > /dev/null
    echo "[INFO] 5G Core Network stopped"
}

# Function to stop gNB containers
stop_gnb() {
    echo "[STEP] Stopping gNB containers..."
    if [[ ! -d $GNB_DIR ]]; then
        echo "[WARN] gNB Aerial directory not found: $GNB_DIR"
        return
    fi
    if ! pushd "$GNB_DIR" > /dev/null; then
        echo "[ERROR] pushd failed: $GNB_DIR"
        exit 1
    fi
    echo "[INFO] Running: docker compose down"
    execute docker compose down
    popd > /dev/null
    echo "[INFO] gNB containers stopped"
}

# Function to show container logs
show_container_logs() {
    echo ""
    echo "=============================================="
    echo "Showing nv-cubb container logs (Ctrl+C to exit)"
    echo "=============================================="
    execute "docker logs -f nv-cubb"
}

# Main execution logic
if [[ -n $STOP_MODE ]]; then
    # Stop containers
    case $STOP_MODE in
        gnb)
            stop_gnb
            ;;
        cn)
            stop_cn5g
            ;;
        all)
            stop_gnb
            stop_cn5g
            ;;
    esac
elif [[ $RUN_ONLY -eq 1 ]]; then
    # Only run containers
    start_cn5g
    start_gnb_aerial
    if [[ $SHOW_LOGS -eq 1 ]]; then
        show_container_logs
    fi
elif [[ $CN5G_ONLY -eq 1 ]]; then
    # Only start CN5G
    start_cn5g
elif [[ $GNB_ONLY -eq 1 ]]; then
    # Only start gNB
    start_gnb_aerial
    if [[ $SHOW_LOGS -eq 1 ]]; then
        show_container_logs
    fi
elif [[ $BUILD_ONLY -eq 1 ]]; then
    # Only clone and build
    setup_oai_repo
    build_docker_images
else
    # Full setup: clone, build, and run, assumes default RU mac address.
    setup_oai_repo
    build_docker_images
    start_cn5g
    start_gnb_aerial
    if [[ $SHOW_LOGS -eq 1 ]]; then
        show_container_logs
    fi
    echo "=============================================="
    echo "OAI + Aerial quickstart complete!"
    echo "=============================================="
fi

