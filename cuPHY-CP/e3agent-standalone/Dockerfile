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

# Builder image for the E3 Agent harness.

FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
# Staged aerial_sdk sources (see restart_e3agent_standalone.sh).
ARG AERIAL_SRC_DIR=.aerial_src

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        python3 \
        libzmq3-dev \
        cppzmq-dev \
        nlohmann-json3-dev \
        libfmt-dev \
        libyaml-cpp-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/e3agent-standalone
COPY . .

RUN cmake -B build -S . -DAERIAL_DATA_LAKE_DIR=/opt/e3agent-standalone/${AERIAL_SRC_DIR} \
    && cmake --build build -j"$(nproc)"

CMD ["./build/e3agent_standalone"]
