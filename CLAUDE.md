# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

NVIDIA Aerial CUDA-Accelerated RAN is a GPU-accelerated 5G physical layer (PHY) and MAC scheduler SDK. The codebase includes:
- **cuPHY**: CUDA-accelerated Physical Layer (L1) - channel coding, modulation, MIMO processing
- **cuPHY-CP**: Control Plane integration components (drivers, controllers, adapters)
- **cuMAC**: GPU-accelerated L2 scheduler (when enabled)
- **pyaerial**: Python API for ML/AI research
- **5GModel**: MATLAB-based test vector generation and 3GPP waveform validation
- **testBenches**: Performance testing and validation tools

## Development Environment

### Container Setup (Recommended)

All development should be done inside the Aerial container:

```bash
# Pull pre-built container from NGC
docker pull nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb

# Start interactive development container
./cuPHY-CP/container/run_aerial.sh

# Inside container: source code is mounted at /opt/nvidia/cuBB (also set as $cuBB_SDK)
```

**Note**: The container must be used for building - the build script explicitly checks for `/.dockerenv`.

### Building Custom Containers

If building containers from source:

```bash
# Install prerequisites: Docker, NVIDIA Container Toolkit, GDRCopy, HPCCM
pip3 install hpccm

# Build containers (run on target platform - x86_64 or aarch64)
cd cuPHY-CP/container
source ./setup.sh
export AERIAL_VERSION_TAG=<custom_tag>
./build_base.sh        # Base container with CUDA and dependencies
./build_devel.sh       # Development container
```

## Build System

### Primary Build Commands

```bash
# Standard build (must be run inside container)
./testBenches/phase4_test_scripts/build_aerial_sdk.sh

# With options
./testBenches/phase4_test_scripts/build_aerial_sdk.sh \
  --toolchain native \
  --preset perf \
  --build_dir build.$(uname -m)

# Build presets:
#   perf       - Performance build with FAPI 10_04, 20C enabled (default)
#   10_02      - FAPI 10_02
#   10_04      - FAPI 10_04
#   10_04_32dl - FAPI 10_04 with 32 DL layers

# CMake presets (minimal builds without tests/pyaerial/cuMAC)
cmake --preset minimal-x86    # x86_64 minimal build
cmake --preset minimal-arm    # ARM/Grace minimal build
```

### Toolchains

Must specify a toolchain via `-DCMAKE_TOOLCHAIN_FILE`:
- `cuPHY/cmake/toolchains/native` - Build and run on same system
- `cuPHY/cmake/toolchains/x86-64` - x86_64 architecture
- `cuPHY/cmake/toolchains/grace-cross` - ARM cross-compilation
- `cuPHY/cmake/toolchains/devkit` - x86_64 devkit
- `cuPHY/cmake/toolchains/bf3` - Bluefield-3
- `cuPHY/cmake/toolchains/r750` - Dell R750

### Manual CMake Build

```bash
# Configure
cmake -B<build_dir> -GNinja \
  -DCMAKE_TOOLCHAIN_FILE=cuPHY/cmake/toolchains/native \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="80-real 90-real"

# Build everything
cmake --build <build_dir> -j $(nproc)

# Build specific targets
cmake --build <build_dir> --target cuphy_examples        # All cuPHY examples
cmake --build <build_dir> --target testbenches_examples  # All testbench examples
cmake --build <build_dir> --target pdsch_tx              # Specific example
cmake --build <build_dir> --target cubb_gpu_test_bench   # GPU testbench
```

### Important Build Options

```cmake
# Architecture
-DCMAKE_CUDA_ARCHITECTURES="80-real 90-real"  # A100/H100, GH200

# Feature flags (see CMakeLists.txt for full list)
-DENABLE_CUMAC=ON/OFF           # Enable cuMAC support
-DENABLE_PYAERIAL=ON/OFF        # Enable Python bindings
-DENABLE_TESTS=ON/OFF           # Enable test building
-DENABLE_20C=ON/OFF             # Enable 20 cell support
-DENABLE_64C=ON/OFF             # Enable 64 cell support
-DENABLE_32DL=ON/OFF            # Enable 32-layer downlink
-DSCF_FAPI_10_04=ON/OFF         # FAPI version 10.04
-DAERIAL_METRICS=ON/OFF         # Enable metrics collection

# Compiler cache (enabled by default)
-DENABLE_CCACHE=ON/OFF
```

## Testing

### cuPHY Unit Tests

```bash
# From cuPHY test directory
cd <build_dir>/cuPHY/test

# Run all unit tests (requires test vectors at specified path)
./cuphy_unit_test.sh <TV_path> <GPU_id>

# Example with common options
./cuphy_unit_test.sh /mnt/cicd_tvs/develop/GPU_test_input/ 0 1 0 1

# Parameters:
#   TV_path:              Path to test vectors
#   GPU:                  CUDA device ID (default: 0)
#   CB_ERROR_CHECK:       Enable PUSCH CB error check (default: 1)
#   COMPONENT_TESTS:      Enable component tests (default: 0)
#   COMPUTE_SANITIZER:    Bitmask for sanitizers (default: 1=memcheck)
```

### GPU Performance Testing (cubb_gpu_test_bench)

Located in `testBenches/cubb_gpu_test_bench` - GPU-only testbench for latency/capacity measurement.

```bash
cd testBenches/perf

# Step 1: Generate test configuration
python3 generate_avg_TDD.py --peak 20 --avg 0 --case F08 --exact --fdm

# Step 2: Run performance test
python3 measure.py \
  --cuphy <build_dir> \
  --vectors <test_vectors_dir> \
  --config example_100_testcases_avg_F08.json \
  --uc uc_avg_F08_TDD.json \
  --gpu 0 --freq 1980 --power 900 \
  --start 20 --cap 20 \
  --slots 400 --iterations 1 \
  --target 8 12 8 117 132 12 \
  --graph

# Step 3: Visualize results
python3 compare.py --filename <result_json> --cells 20+0

# Collect Nsight traces (use fewer slots)
python3 measure.py ... --debug --debug_mode nsys --slots 8
```

### Phase-4 Tests (End-to-End cuBB Tests)

See `testBenches/phase4_test_scripts/README.md` for detailed instructions.

### pyAerial Tests

```bash
# Inside pyAerial container
cd $cuBB_SDK/pyaerial

# Build Python bindings first
cmake -Bbuild -GNinja -DCMAKE_TOOLCHAIN_FILE=cuPHY/cmake/toolchains/native
cmake --build build -t _pycuphy pycuphycpp

# Install pyaerial package
./scripts/install_dev_pkg.sh

# Run tests
./scripts/run_static_tests.sh  # Linting, type checking
./scripts/run_unit_tests.sh    # Unit tests (requires test vectors)
./scripts/run_notebooks.sh     # Execute example notebooks
```

## Test Vector Generation (5GModel)

Test vectors for validation are generated using MATLAB:

```matlab
cd <5GModel_root>/nr_matlab/
startup
runRegression  % Runs compliance tests and generates test vectors

% Generate performance test vectors
runRegression({'TestVector'}, {'perf_TVs'}, {'full'})
```

Test vectors are output to `<5GModel_root>/nr_matlab/GPU_test_input/`.

## Architecture

### Component Organization

```
cuPHY/                      # Core GPU-accelerated PHY library
├── src/cuphy/              # Main CUDA/C++ PHY functions and APIs
├── src/cuphy_channels/     # Channel-level aggregations
├── src/cuphy_hdf5/         # HDF5 utilities for test vectors
├── examples/               # Reference implementations (PDSCH, PUSCH, SRS, PRACH)
└── test/                   # Unit and component tests

cuPHY-CP/                   # Control Plane integration
├── aerial-fh-driver/       # O-RAN fronthaul driver (DPDK-based, O-RAN 5.0 subset)
├── cuphycontroller/        # Main application - system initialization, cell lifecycle
├── cuphydriver/            # PHY driver with DL/UL worker threads
├── cuphyl2adapter/         # L2 adapter for FAPI interface
├── scfl2adapter/           # Small Cell Forum FAPI adapter
├── ru-emulator/            # Radio Unit emulator
├── testMAC/                # Test MAC implementation
├── container/              # Docker container build scripts (HPCCM-based)
└── gt_common_libs/         # Common libraries (IPC, metrics, utilities)

cuMAC/                      # GPU-accelerated MAC scheduler (optional, enabled via ENABLE_CUMAC)
cuMAC-CP/                   # MAC Control Plane components

pyaerial/                   # Python API for ML/AI research
├── pybind11/               # C++ Python bindings
├── src/                    # Python package source
├── notebooks/              # Example Jupyter notebooks
├── container/              # pyAerial container with ML tools (Sionna, TensorFlow)
└── tests/                  # Unit tests

testBenches/
├── cubb_gpu_test_bench/    # GPU-only performance testbench
├── perf/                   # Python helpers for performance testing
├── chanModels/             # 3GPP 38.901 channel models (TDL/CDL/UMa/UMi/RMa)
└── phase4_test_scripts/    # End-to-end cuBB test automation

5GModel/                    # MATLAB test vector generation
└── nr_matlab/              # 5G NR waveform generation (3GPP compliance)
```

### Key Architectural Patterns

**GPU Acceleration**: cuPHY implements 5G NR PHY processing on GPU using CUDA. Channel-level operations (PDSCH, PUSCH, PRACH, etc.) are accelerated with specialized kernels for LDPC/Polar coding, modulation, MIMO, and channel estimation.

**Control/Data Plane Separation**: cuPHY-CP provides control plane components that integrate the GPU-accelerated PHY (cuPHY) with O-RAN fronthaul interfaces, L2 adapters (FAPI), and system management.

**Fronthaul Interface**: `aerial-fh-driver` implements O-RAN fronthaul using DPDK for high-performance packet processing. Supports Ethernet encapsulation, C-plane section types 0/1/3/5, static compression, and section extensions.

**Multi-Cell Architecture**: The system is designed for multi-cell operation with GPU resource sharing via NVIDIA MPS (Multi-Process Service). SM allocation is configurable per channel/subcontext.

**FAPI Interface**: L2 adapters (cuphyl2adapter/scfl2adapter) implement Small Cell Forum FAPI for L1-L2 communication. Supports both FAPI 10.02 and 10.04 (selected via build flags).

**Configuration Management**: System configuration uses YAML files (see `cuPHY-CP/cuphycontroller/config/` for examples). Configuration covers PHY parameters, fronthaul settings, CPU affinity, and resource allocation.

## Key Technologies

- **CUDA 12.6+** for GPU acceleration
- **CMake 3.25+** with Ninja build system
- **C++20** standard (C++17 for CUDA code)
- **DPDK** for fronthaul packet processing
- **DOCA** (Mellanox/NVIDIA Data Center on a Chip Architecture)
- **HDF5** for test vector storage
- **Python 3.8+** with pybind11 for pyaerial
- **MATLAB** for test vector generation (5GModel)

## Common Development Patterns

**Adding New PHY Features**: Implement in `cuPHY/src/cuphy/`, add examples in `cuPHY/examples/`, add unit tests in `cuPHY/test/`. Update MATLAB 5GModel for test vector generation if needed.

**Configuration Changes**: System configurations are in `cuPHY-CP/cuphycontroller/config/`. Each YAML includes PHY params, FH settings, L2 adapter config, and CPU/GPU resource allocation.

**Performance Optimization**: Use `cubb_gpu_test_bench` for GPU-only profiling. Adjust SM allocation via `--target` parameter. Use `--graph` mode for lower latency vs stream mode.

**Multi-Cell Testing**: Configure cell count via YAML or test scripts. Use Phase-4 scripts for end-to-end testing with RU emulator.

## Important Notes

- **Container Requirement**: All builds must run inside the Aerial container (checks for `/.dockerenv`)
- **Toolchain Mandatory**: CMake requires explicit toolchain specification
- **GPU Architecture**: Default CUDA architectures are 80 (A100) and 90 (H100/GH200)
- **Test Vectors**: Required for testing - generate with 5GModel or use pre-generated from CI/CD location
- **Real-Time Requirements**: cuphycontroller uses SCHED_FIFO for RT threads (configurable via ENABLE_SCHED_FIFO_ALL_RT)
- **Git LFS**: Large files use Git LFS - run `git lfs install && git lfs pull` after clone
- **Submodules**: Clone with `--recurse-submodules`

## Documentation

- Full documentation: https://docs.nvidia.com/aerial/
- API documentation: Build with `cmake --build <build_dir> --target docs_doxygen`
- NVIDIA NGC Container: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/aerial/containers/aerial-cuda-accelerated-ran
- 6G Developer Program: https://developer.nvidia.com/6g-program
