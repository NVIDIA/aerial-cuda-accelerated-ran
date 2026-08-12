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

# Stage the sibling cuPHY-CP/data_lake/ sources into .aerial_src/, then (re)build
# and start the container via compose. Override the source with AERIAL_DATA_LAKE_DIR.

set -euo pipefail
echo "=== restart e3agent-standalone ==="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Canonical e3_agent sources live in the sibling cuPHY-CP/data_lake/.
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

echo "Rebuilding and starting container (Ctrl-C to stop)..."
docker compose down --remove-orphans 2>/dev/null || true
trap 'echo; echo "Stopping container..."; docker compose down' EXIT

echo "Deleting existing image..."
docker image rm e3agent-standalone:latest 2>/dev/null || true

echo "Building and starting container..."
docker compose up --build
