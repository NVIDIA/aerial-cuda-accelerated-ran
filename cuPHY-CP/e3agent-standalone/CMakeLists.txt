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

# Standalone E3 Agent harness — runs aerial_sdk's e3_agent.cpp without the RAN.
# Self-contained: configure directly with
#   cmake -B build -S cuPHY-CP/e3agent-standalone

cmake_minimum_required(VERSION 3.18)
project(e3agent_standalone LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(fmt REQUIRED)
find_package(nlohmann_json REQUIRED)
find_library(ZMQ_LIBRARY NAMES zmq REQUIRED)

# Canonical e3_agent sources live in the sibling cuPHY-CP/data_lake/. Override
# with -DAERIAL_DATA_LAKE_DIR=<path>.
set(AERIAL_DATA_LAKE_DIR "" CACHE PATH "Path to cuPHY-CP/data_lake/")
if(NOT AERIAL_DATA_LAKE_DIR)
    get_filename_component(AERIAL_DATA_LAKE_DIR
        "${CMAKE_CURRENT_SOURCE_DIR}/../data_lake" ABSOLUTE)
endif()
if(NOT EXISTS "${AERIAL_DATA_LAKE_DIR}/e3_agent.cpp")
    message(FATAL_ERROR
        "e3agent-standalone: invalid AERIAL_DATA_LAKE_DIR='${AERIAL_DATA_LAKE_DIR}' "
        "(no e3_agent.cpp). Pass -DAERIAL_DATA_LAKE_DIR=<path>.")
endif()
message(STATUS "e3agent-standalone: AERIAL_DATA_LAKE_DIR=${AERIAL_DATA_LAKE_DIR}")

# Stage e3_agent.{cpp,hpp} into the build tree (without data_lake.hpp) so the
# agent's quoted #include "data_lake.hpp" resolves to our stub on the include
# path instead of the real header beside the original source.
set(_staged ${CMAKE_BINARY_DIR}/aerial_staged)
configure_file(${AERIAL_DATA_LAKE_DIR}/e3_agent.cpp ${_staged}/e3_agent.cpp COPYONLY)
configure_file(${AERIAL_DATA_LAKE_DIR}/e3_agent.hpp ${_staged}/e3_agent.hpp COPYONLY)

add_executable(e3agent_standalone
    src/main.cpp
    src/config.cpp
    src/synth.cpp
    src/feeder.cpp
    src/replay.cpp
    ${_staged}/e3_agent.cpp)
target_include_directories(e3agent_standalone PRIVATE include ${_staged})
target_link_libraries(e3agent_standalone PRIVATE
    Threads::Threads
    fmt::fmt
    nlohmann_json::nlohmann_json
    ${ZMQ_LIBRARY})
# yaml-cpp target name differs across versions (0.7: yaml-cpp, 0.8: yaml-cpp::yaml-cpp).
if(TARGET yaml-cpp::yaml-cpp)
    target_link_libraries(e3agent_standalone PRIVATE yaml-cpp::yaml-cpp)
else()
    target_link_libraries(e3agent_standalone PRIVATE yaml-cpp)
endif()

# Drift guard: vendored include/e3_types.hpp must match aerial_sdk's canonical
# structs. Runs on every build; disable with -DE3SA_CHECK_DRIFT=OFF.
option(E3SA_CHECK_DRIFT "Verify vendored e3_types.hpp against aerial_sdk on build" ON)
if(E3SA_CHECK_DRIFT)
    find_package(Python3 COMPONENTS Interpreter)
    if(NOT Python3_Interpreter_FOUND)
        message(FATAL_ERROR
            "e3agent-standalone: E3SA_CHECK_DRIFT=ON but python3 not found. "
            "Install python3 or pass -DE3SA_CHECK_DRIFT=OFF.")
    endif()
    add_custom_target(check_drift ALL
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/scripts/check_drift.py
            --data-lake-dir ${AERIAL_DATA_LAKE_DIR}
            --vendored ${CMAKE_CURRENT_SOURCE_DIR}/include/e3_types.hpp
        COMMENT "Checking e3_types.hpp drift against aerial_sdk"
        VERBATIM)
    add_dependencies(e3agent_standalone check_drift)
endif()
