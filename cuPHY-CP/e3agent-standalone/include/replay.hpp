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

// Replays a recorded trace (see replay_format.hpp) into DataLake/SHM,
// reconstructing PUSCH and SRS indications and pacing emission by the
// recorded timestamp_tai delta.

#ifndef E3SA_REPLAY_HPP
#define E3SA_REPLAY_HPP

#include "config.hpp"

#include <atomic>
#include <cstdint>

class DataLake;
class E3Agent;

namespace e3sa {

// Tag-dispatched trace reader: PUSCH writes fh+hest rings, SRS writes
// iq/hest/rb-snr rings. Absent blobs (len 0) zero-fill; SHM offsets/indices
// are recomputed here, not stored.
class Replay {
public:
	Replay(DataLake& dl, E3Agent& agent, const ReplayCfg& cfg, const CpuCfg& cpu,
	       const RowsCfg& rows);
	void run(std::atomic<bool>& stop);

private:
	DataLake& dl_;
	E3Agent& agent_;
	ReplayCfg cfg_;
	CpuCfg cpu_;
	RowsCfg rows_;
};

} // namespace e3sa

#endif // E3SA_REPLAY_HPP
