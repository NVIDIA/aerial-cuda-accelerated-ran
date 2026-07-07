# NVIDIA / EKONOMIK MOBIL,S.R.L , NVIDIA Aerial™ CUDA-Accelerated RAN

## Overview
NVIDIA Aerial™ CUDA-Accelerated RAN is a part of [NVIDIA AI Aerial™](https://developer.nvidia.com/industries/telecommunications/ai-aerial), a portfolio of accelerated computing platforms, software and tools to build, train, simulate, and deploy AI-native wireless networks.

Documentation for AI Aerial™ can be found [here](https://docs.nvidia.com/aerial/index.html).

The following AI Aerial™ software is available as open source:
- NVIDIA Aerial™ CUDA-Accelerated RAN (this repository)
- [NVIDIA Aerial™ Framework](https://github.com/NVIDIA/aerial-framework)

Updates on new software releases, NVIDIA 6G events and technical training for AI Aerial™ are available via the [NVIDIA 6G Developer Program](https://developer.nvidia.com/6g-program).

The **Aerial CUDA-Accelerated RAN** SDK includes:

- **GPU-Accelerated 5G PHY (cuPHY)**: CUDA-based physical layer processing for 5G NR including channel coding (LDPC, Polar), modulation/demodulation, MIMO processing, and channel estimation
- **GPU-Accelerated MAC Scheduler (cuMAC)**: High-performance L2 scheduler acceleration for resource allocation and scheduling
- **Python API (pyAerial)**: Python bindings for AI/ML research and integration with frameworks like TensorFlow and Sionna
- **5G Reference Models (5GModel)**: MATLAB-based 5G waveform generation and test vector creation based on 3GPP specifications
- **Containerized Environment**: Docker-based development and deployment with pre-built containers

### Repository Structure

```
aerial-cuda-accelerated-ran/
├── cuPHY/              # CUDA-accelerated Physical Layer (L1)
├── cuPHY-CP/           # Control Plane and integration components
│   ├── aerial-fh-driver/    # Fronthaul driver for O-RAN interfaces
│   ├── cuphycontroller/     # PHY controller
│   ├── cuphydriver/         # PHY driver
│   ├── cuphyl2adapter/      # L2 adapter
│   ├── ru-emulator/         # Radio Unit emulator
│   ├── testMAC/            # Test MAC implementation
│   └── container/          # Container build scripts and recipes
│   └── data_lake/          # data lake and E3 agent
├── cuMAC/              # CUDA-accelerated L2 Layer
├── cuMAC-CP/           # MAC Control Plane components
├── pyaerial/           # Python API and ML/AI tools
├── 5GModel/            # TV generation for cuPHY and cuBB verification
├── testBenches/        # Test benches and performance measurement tools
├── testVectors/        # Test vectors for validation
└── cubb_scripts/       # Build and automation scripts
```

## Getting Started

### Using Pre-Built Container (Recommended)

```bash
# Clone repository
git clone --recurse-submodules https://github.com/NVIDIA/aerial-cuda-accelerated-ran.git
cd aerial-cuda-accelerated-ran

# Enable git LFS and pull files
sudo apt install git-lfs && git lfs pull

# Start interactive development container
./cuPHY-CP/container/run_aerial.sh

# Inside container: Build SDK
./testBenches/phase4_test_scripts/build_aerial_sdk.sh
```

- Container versions available at [NVIDIA NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/aerial/containers/aerial-cuda-accelerated-ran)

### Further Information

Visit the full documentation at [NVIDIA Docs Hub](https://docs.nvidia.com/aerial/)

## Contribution Guidelines
- Aerial is not accepting contributions at this time.

## Security
- Vulnerability disclosure: [SECURITY.md](SECURITY.md)
- **Do not file public issues for security reports.**

## Support
- **Level**: Maintained
- **How to get help**:
  - File issues on GitHub for bugs and feature requests
  - Join discussions for questions and community support

## License
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

**Note**: Some dependencies may have different licenses. See [ATTRIBUTION.rst](ATTRIBUTION.rst) for third-party attributions in the source repository.

## Citation

If you use NVIDIA Aerial™ CUDA-Accelerated RAN in your research, please cite:

```bibtex
@software{nvidia_aerial_cuda_accelerated_ran,
  title = {NVIDIA Aerial™ CUDA-Accelerated RAN},
  author = {NVIDIA Corporation},
  year = {2025-2026},
  url = {https://github.com/NVIDIA/aerial-cuda-accelerated-ran}
}
```
