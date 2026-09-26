#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import aerial_mcore as NRSimulator
import matlab
import time
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Generate performance test vectors')
parser.add_argument('pattern', type=int, nargs='?', default=69, help='Pattern number to generate (default: 69)')
parser.add_argument('only_gen_lp', type=int, nargs='?', default=0, help='Only generate launch pattern files (1) or generate perf patterns (0)')
parser.add_argument('exec_cmd', type=int, nargs='?', default=1, help='Execute commands (1) or just print them (0)')
args = parser.parse_args()

print(f"Configuration:")
print(f"  Pattern number: {args.pattern}")
print(f"  Operation mode: {'Launch pattern only' if args.only_gen_lp else 'Launch pattern and test vector'}")
print(f"  MATLAB command execution: {'Enabled' if args.only_gen_lp or args.exec_cmd else 'Print only'}")

eng = NRSimulator.initialize()
#eng.cfg_parfor(0,nargout=0)

tic = time.time()

if args.only_gen_lp:
    # Only generate launch patterns
    eng.genLP_POC2(args.pattern)
else:
    # Generate perf pattern
    eng.genPerfPattern(args.pattern, 'allChannels', args.exec_cmd)  # Use exec_cmd from command line argument

    # Alternative method to generate perf pattern using runRegression
    # eng.runRegression(['PerfPattern'], args.pattern, ['allChannels'])

toc = time.time()
print(f"Elapsed time: {toc-tic} seconds")
