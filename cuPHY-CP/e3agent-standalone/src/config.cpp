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

#include "config.hpp"

#include <yaml-cpp/yaml.h>

#include <initializer_list>
#include <set>
#include <stdexcept>
#include <string>

namespace e3sa {
namespace {

[[noreturn]] void fail(const std::string& msg) {
	throw std::runtime_error("config: " + msg);
}

// Reject any key not in `allowed`.
void only(const YAML::Node& node, const char* where,
          std::initializer_list<const char*> allowed) {
	if (!node.IsMap()) fail(std::string(where) + ": expected a mapping");
	const std::set<std::string> ok(allowed.begin(), allowed.end());
	for (const auto& kv : node) {
		const auto key = kv.first.as<std::string>();
		if (!ok.count(key)) fail(std::string(where) + ": unknown key '" + key + "'");
	}
}

const YAML::Node require(const YAML::Node& node, const char* where, const char* key) {
	const YAML::Node child = node[key];
	if (!child) fail(std::string(where) + ": missing required key '" + key + "'");
	return child;
}

template <typename T>
T scalar(const YAML::Node& node, const char* where, const char* key) {
	try {
		return require(node, where, key).as<T>();
	} catch (const YAML::Exception&) {
		fail(std::string(where) + "." + key + ": invalid value");
	}
}

template <typename T>
T scalar_or(const YAML::Node& node, const char* where, const char* key, T def) {
	const YAML::Node child = node[key];
	if (!child) return def;
	try {
		return child.as<T>();
	} catch (const YAML::Exception&) {
		fail(std::string(where) + "." + key + ": invalid value");
	}
}

void parse_agent(const YAML::Node& n, AgentCfg& a) {
	only(n, "agent", {"rep_port", "pub_port", "sub_port"});
	a.rep_port = scalar_or<uint16_t>(n, "agent", "rep_port", a.rep_port);
	a.pub_port = scalar_or<uint16_t>(n, "agent", "pub_port", a.pub_port);
	a.sub_port = scalar_or<uint16_t>(n, "agent", "sub_port", a.sub_port);
	if (!a.rep_port || !a.pub_port || !a.sub_port) fail("agent: ports must be non-zero");
}

void parse_rows(const YAML::Node& n, RowsCfg& r) {
	only(n, "rows", {"fh", "pusch", "hest", "srs_iq", "srs", "srs_hest"});
	r.fh = scalar_or<uint32_t>(n, "rows", "fh", r.fh);
	r.pusch = scalar_or<uint32_t>(n, "rows", "pusch", r.pusch);
	r.hest = scalar_or<uint32_t>(n, "rows", "hest", r.hest);
	r.srs_iq = scalar_or<uint32_t>(n, "rows", "srs_iq", r.srs_iq);
	r.srs = scalar_or<uint32_t>(n, "rows", "srs", r.srs);
	r.srs_hest = scalar_or<uint32_t>(n, "rows", "srs_hest", r.srs_hest);
	if (!r.fh || !r.pusch || !r.hest || !r.srs_iq || !r.srs || !r.srs_hest) {
		fail("rows: all row counts must be non-zero");
	}
}

void parse_cpu(const YAML::Node& n, CpuCfg& c) {
	only(n, "cpu", {"feeder_core"});
	c.feeder_core = scalar_or<int>(n, "cpu", "feeder_core", c.feeder_core);
	if (c.feeder_core < -1) fail("cpu.feeder_core: must be -1 (no pinning) or a core index");
}

void parse_synth(const YAML::Node& n, SynthCfg& s) {
	only(n, "synth", {"slot_tick_us", "tdd_pattern", "srs_periodicity_ms", "duration_slots"});
	s.slot_tick_us = scalar_or<uint32_t>(n, "synth", "slot_tick_us", s.slot_tick_us);
	s.tdd_pattern = scalar_or<std::string>(n, "synth", "tdd_pattern", s.tdd_pattern);
	s.srs_periodicity_ms = scalar_or<uint32_t>(n, "synth", "srs_periodicity_ms", s.srs_periodicity_ms);
	s.duration_slots = scalar_or<uint64_t>(n, "synth", "duration_slots", s.duration_slots);

	if (!s.slot_tick_us) fail("synth.slot_tick_us: must be non-zero");
	if (s.tdd_pattern.empty()) fail("synth.tdd_pattern: must be non-empty");
	bool has_u = false;
	for (char c : s.tdd_pattern) {
		if (c != 'D' && c != 'S' && c != 'U') {
			fail("synth.tdd_pattern: invalid char '" + std::string(1, c) + "' (allowed: D, S, U)");
		}
		has_u = has_u || c == 'U';
	}
	if (!has_u) fail("synth.tdd_pattern: must contain at least one 'U' (uplink)");
	if (!s.srs_periodicity_ms) fail("synth.srs_periodicity_ms: must be non-zero");
	if ((uint64_t(s.srs_periodicity_ms) * 1000u) % s.slot_tick_us != 0) {
		fail("synth.srs_periodicity_ms: must be an integer number of slots (multiple of slot_tick_us)");
	}
}

void parse_replay(const YAML::Node& n, ReplayCfg& r) {
	only(n, "replay", {"path", "loops"});
	r.path = scalar<std::string>(n, "replay", "path");
	r.loops = scalar_or<uint64_t>(n, "replay", "loops", r.loops);
	if (r.path.empty()) fail("replay.path: must be non-empty");
}

} // namespace

Config Config::load(const std::string& path) {
	YAML::Node root;
	try {
		root = YAML::LoadFile(path);
	} catch (const YAML::Exception& e) {
		fail("failed to load '" + path + "': " + e.what());
	}
	only(root, "root", {"mode", "log_level", "agent", "rows", "cpu", "synth", "replay"});

	Config c;
	const auto mode = scalar<std::string>(root, "root", "mode");
	if (mode == "synth") c.mode = Mode::Synth;
	else if (mode == "replay") c.mode = Mode::Replay;
	else fail("mode: must be 'synth' or 'replay' (got '" + mode + "')");

	c.log_level = scalar_or<std::string>(root, "root", "log_level", c.log_level);
	if (c.log_level != "VRB" && c.log_level != "DBG" && c.log_level != "INF" &&
	    c.log_level != "WRN" && c.log_level != "ERR" && c.log_level != "FAT")
		fail("log_level: must be VRB|DBG|INF|WRN|ERR|FAT (got '" + c.log_level + "')");

	if (root["agent"]) parse_agent(root["agent"], c.agent);
	if (root["rows"]) parse_rows(root["rows"], c.rows);
	if (root["cpu"]) parse_cpu(root["cpu"], c.cpu);

	// synth is required in synth mode, optional (fill source) in replay.
	if (c.mode == Mode::Synth) {
		parse_synth(require(root, "root", "synth"), c.synth);
	} else if (root["synth"]) {
		parse_synth(root["synth"], c.synth);
	}

	// replay is required in replay mode, optional in synth.
	if (c.mode == Mode::Replay) {
		parse_replay(require(root, "root", "replay"), c.replay);
	} else if (root["replay"]) {
		parse_replay(root["replay"], c.replay);
	}

	return c;
}

} // namespace e3sa
