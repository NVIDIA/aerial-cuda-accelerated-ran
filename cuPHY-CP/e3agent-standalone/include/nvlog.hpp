/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Drop-in replacement for aerial_sdk's nvlog.hpp, shadowing it on the harness
// include path so e3_agent.cpp compiles without the full nvlog/fmtlog stack.
// Routes the NVLOG*_FMT macros to fmt-formatted stderr lines. Uses the same
// include guard as the real header so it wins if both are ever reachable.

#ifndef _NVLOG_HPP_
#define _NVLOG_HPP_

#include <cstdio>
#include <cstdlib>
#include <string>

#include <fmt/format.h>

// Symbols the agent pulls from nvlog.h / aerial_event_code.h.
#define NVLOG_TAG_BASE_CUPHY_CONTROLLER 100
#define AERIAL_SYSTEM_API_EVENT 17

namespace e3shim {
enum Level { VRB, DBG, INF, WRN, ERR, FAT, CON };

// Min level printed; set once from config at startup (default INF).
inline Level g_threshold = INF;
inline void setThreshold(Level l) { g_threshold = l; }

// Optional file sink (mirrors stderr). Opened from main; no-op if path unusable.
inline FILE* g_logfile = nullptr;
inline void setLogFile(const char* path) {
	if (g_logfile) { std::fclose(g_logfile); g_logfile = nullptr; }
	g_logfile = path ? std::fopen(path, "w") : nullptr;
}

inline void log(Level lvl, const char* name, int tag, const std::string& msg) {
	if (lvl != CON && lvl < g_threshold) return;  // CON always prints
	fmt::print(stderr, "[E3][{}][tag {}] {}\n", name, tag, msg);
	if (g_logfile) { fmt::print(g_logfile, "[E3][{}][tag {}] {}\n", name, tag, msg); std::fflush(g_logfile); }
	if (lvl == FAT) std::abort();  // mirror production NVLOGF_FMT: fatal never returns
}
}  // namespace e3shim

// Level macros: (tag, fmt, ...). fmt string is a literal at every call site,
// so fmt::format compile-time checking applies.
#define NVLOGV_FMT(tag, ...) ::e3shim::log(::e3shim::VRB, "VRB", tag, fmt::format(__VA_ARGS__))
#define NVLOGD_FMT(tag, ...) ::e3shim::log(::e3shim::DBG, "DBG", tag, fmt::format(__VA_ARGS__))
#define NVLOGI_FMT(tag, ...) ::e3shim::log(::e3shim::INF, "INF", tag, fmt::format(__VA_ARGS__))
#define NVLOGW_FMT(tag, ...) ::e3shim::log(::e3shim::WRN, "WRN", tag, fmt::format(__VA_ARGS__))
#define NVLOGC_FMT(tag, ...) ::e3shim::log(::e3shim::CON, "CON", tag, fmt::format(__VA_ARGS__))

// Event macros: (tag, event_level, fmt, ...). event_level is dropped.
#define NVLOGE_FMT(tag, evt, ...) ::e3shim::log(::e3shim::ERR, "ERR", tag, fmt::format(__VA_ARGS__))
#define NVLOGF_FMT(tag, evt, ...) ::e3shim::log(::e3shim::FAT, "FAT", tag, fmt::format(__VA_ARGS__))

#endif  // _NVLOG_HPP_
