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

# (Re)start the container via compose. Pass -b after source edits to force a rebuild.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: ./restart_e3agent_standalone.sh [-b|--build] [-c|--config PATH] [--synth|--replay]
                                       [-t|--trace PATH] [--loops N] [--wait-subscriber]
                                       [--log-level LVL] [--feeder-core N] [--duration-slots N]

  -b, --build         Rebuild the image (auto-enabled if the image does not exist yet).
                      Needed after C++ or staged data_lake source changes.
  -c, --config PATH   Config file relative to the tool dir; a bare name resolves
                      under config/. Default: config/e3agent-standalone.example.yaml
  -h, --help          Show this help.

Overrides (take precedence over the config file):
  --synth, --replay   mode
  -t, --trace PATH    replay.path; any host file, mounted read only. Implies --replay.
  --loops N           replay.loops (0 = forever)
  --wait-subscriber   replay.wait_for_subscriber: start each pass on the first E3 subscription,
                      restart from the first slot when the last one is released
  --log-level LVL     log_level (VRB|DBG|INF|WRN|ERR|FAT)
  --feeder-core N     cpu.feeder_core (-1 = no pinning)
  --duration-slots N  synth.duration_slots (0 = forever)
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BUILD=0
CONFIG_FILE="config/e3agent-standalone.example.yaml"
TRACE=""
OVERRIDES=()
while [ $# -gt 0 ]; do
    case "$1" in
        -b|--build)   BUILD=1; shift;;
        -c|--config)  [ $# -ge 2 ] || { echo "Missing PATH for $1" >&2; usage; exit 1; }; CONFIG_FILE="$2"; shift 2;;
        --synth|--replay|--wait-subscriber) OVERRIDES+=("$1"); shift;;
        -t|--trace)
            [ $# -ge 2 ] || { echo "Missing PATH for $1" >&2; usage; exit 1; }
            TRACE="$(realpath -e -- "$2" 2>/dev/null)" && [ -f "$TRACE" ] && [ -r "$TRACE" ] \
                || { echo "ERROR: trace '$2' is not a readable file" >&2; exit 1; }
            shift 2;;
        --loops|--log-level|--feeder-core|--duration-slots)
            [ $# -ge 2 ] || { echo "Missing value for $1" >&2; usage; exit 1; }; OVERRIDES+=("$1" "$2"); shift 2;;
        -h|--help)    usage; exit 0;;
        *) echo "Unknown option: $1" >&2; usage; exit 1;;
    esac
done
cd "$SCRIPT_DIR"
case "$CONFIG_FILE" in */*) ;; *) CONFIG_FILE="config/$CONFIG_FILE";; esac

if [ -n "$TRACE" ]; then
    export E3SA_TRACE_DIR="$(dirname "$TRACE")"
    OVERRIDES+=(--trace "/trace/$(basename "$TRACE")")
fi

echo "=== restart e3agent-standalone ==="

IMAGE="e3agent-standalone:latest"
if [ "$BUILD" -eq 0 ] && ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "Image $IMAGE not found; building it this time."
    BUILD=1
fi

if [ "$BUILD" -eq 1 ]; then
    AERIAL_DATA_LAKE_DIR="${AERIAL_DATA_LAKE_DIR:-$SCRIPT_DIR/../data_lake}"
    if [ ! -f "$AERIAL_DATA_LAKE_DIR/e3_agent.cpp" ]; then
        echo "ERROR: invalid AERIAL_DATA_LAKE_DIR='$AERIAL_DATA_LAKE_DIR' (no e3_agent.cpp)" >&2
        echo "       expected the sibling cuPHY-CP/data_lake/, or set AERIAL_DATA_LAKE_DIR." >&2
        exit 1
    fi
    AERIAL_DATA_LAKE_DIR="$(cd "$AERIAL_DATA_LAKE_DIR" && pwd)"
    echo "Staging aerial sources from: $AERIAL_DATA_LAKE_DIR"
    STAGE_DIR="$SCRIPT_DIR/.aerial_src"
    rm -rf "$STAGE_DIR"
    mkdir -p "$STAGE_DIR"
    cp "$AERIAL_DATA_LAKE_DIR"/e3_agent.cpp \
       "$AERIAL_DATA_LAKE_DIR"/e3_agent.hpp \
       "$AERIAL_DATA_LAKE_DIR"/data_lake.hpp \
       "$STAGE_DIR/"
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "WARNING: config '$CONFIG_FILE' not found under $SCRIPT_DIR" >&2
fi
echo "Config: $CONFIG_FILE (build=$BUILD)"
[ -z "$TRACE" ] || echo "Trace: $TRACE"
[ ${#OVERRIDES[@]} -eq 0 ] || echo "Overrides: ${OVERRIDES[*]}"

NAME="e3agent-standalone"
docker compose down --remove-orphans 2>/dev/null || true
docker rm -f "$NAME" >/dev/null 2>&1 || true
trap 'echo; echo "Stopping container..."; docker stop "$NAME" >/dev/null 2>&1 || true; docker rm "$NAME" >/dev/null 2>&1 || true' EXIT

if [ "$BUILD" -eq 1 ]; then
    echo "Deleting existing image..."
    docker image rm "$IMAGE" 2>/dev/null || true
    docker compose build
else
    echo "Reusing existing image (pass -b to rebuild after source changes)."
fi
echo "Starting container (Ctrl-C to stop)..."
docker compose run --rm --name "$NAME" "$NAME" ./build/e3agent_standalone --config "$CONFIG_FILE" "${OVERRIDES[@]}"
