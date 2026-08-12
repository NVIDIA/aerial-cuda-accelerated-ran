# NVIDIA Aerial SDK Installation

## Quick start

First-time provisioning must be run in stages because the required reboot or power cycle interrupts the installation. Do not expect `make all` to complete in one invocation.

### DGX Spark

1. Install `make` if needed, prepare the Aerial CUDA kernel, and reboot to load it:

   ```bash
   sudo apt update && sudo apt install -y build-essential
   make prepare
   sudo reboot
   ```

2. Install DOCA and the NIC firmware, then reboot so the firmware update takes effect:

   ```bash
   make install
   sudo reboot
   ```

3. Resume and complete the installation:

   ```bash
   make install
   ```

4. Build and start the software:

   ```bash
   RU_MAC=<yourWncMac> make build
   RU_MAC=<yourWncMac> make start_all
   ```

### GH200

1. Install `make` if needed, prepare the Aerial CUDA kernel, and reboot to load it:

   ```bash
   sudo apt update && sudo apt install -y build-essential
   make prepare
   sudo reboot
   ```

2. Install the BFB on both BF3 devices:

   ```bash
   make install
   ```

   Perform a full cold power cycle, then power the host back on. A soft reboot is not sufficient.

3. Resume the installation to apply the BF3 NIC settings:

   ```bash
   make install
   ```

   Perform a second full cold power cycle, then power the host back on.

4. Resume and complete the installation:

   ```bash
   make install
   ```

5. Build and start the software:

   ```bash
   RU_MAC=<yourWncMac> make build
   RU_MAC=<yourWncMac> make start_all
   ```

Replace `<yourWncMac>` with your WNC RU MAC address (e.g. `e8:c7:cf:ac:58:32`).
If using a different RU, you may have to change PCP and VLAN in `cuphycontroller_P5G_WNC_DGX.yaml` before running `make build` or `make start_all` on DGX Spark.

---

## Make targets

| Target | Function |
|--------|----------|
| **prepare** | Installs the Aerial CUDA kernel. Reboot after this, then follow the platform-specific staged flow above. |
| **all** | Convenience flow: `install` -> `build` -> `start_all`. On a first-time installation it stops at required restart boundaries, so use the staged flow above. |
| **install** | Resumable installation of drivers, network configuration, NIC firmware, and system services. It stops when a platform restart is required; run it again afterward to resume. |
| **net** | Sets up network interfaces (e.g. `aerial0x`). |
| **kernel** | Installs the Aerial CUDA kernel. Reboot required after this if the kernel was updated. |
| **drivers** | Installs DOCA, OFED, and GPU drivers. Prompts for confirmation; ensure PTP, VLAN, RU peer MAC, and Docker login are configured first. |
| **services** | Installs PTP and system services. |
| **build** | Runs `build_aerial` and `build_oai`. |
| **build_aerial** | Builds the Aerial SDK (runs `quickstart-aerial.sh`). Use `PROFILE=` or `BUILD_PRESET`/`BUILD_CMAKE_FLAGS` for variants. |
| **build_oai** | Builds OAI (runs `quickstart-oai.sh --build-only`). Use `RU_MAC=<mac>` if needed. |
| **start_gnb** | Starts the gNB. Set `RU_MAC=<mac>` when invoking. |
| **start_cn** | Starts CN5G. |
| **start_all** | Starts both gNB and CN5G. Set `RU_MAC=<mac>` when invoking. |
| **check** | Checks installation status (kernel and services). |
| **help** | Prints target list and usage. |
| **clean** | Phony target (see `make help` for current behavior). |

## Options

- **DRYRUN=1** - Show commands without executing (e.g. `make install DRYRUN=1`).
- **VERBOSE=1** - Print commands before executing.
- **RU_MAC=aa:bb:cc:dd:ee:ff** - Set RU MAC address for gNB/OAI (required for `all`, `start_gnb`, `start_all`, and when building OAI with a specific MAC).
- **PROFILE=name** - Aerial build profile file: `oai.conf`, `fapi_10_02.conf`, `fapi_10_04.conf`, or a custom `<name>.conf` (see **Build profiles** below).
- **BUILD_PRESET=preset** - Override Aerial preset: `perf`, `10_02`, `10_04`, `10_04_32dl`.
- **BUILD_CMAKE_FLAGS="..."** - Override CMake flags for the Aerial build.

## Build profiles (Aerial configuration)

The Aerial build can use different configurations (OAI L2+ default, FAPI 10_02 only, or FAPI 10.04). Use either **make targets** or a **configuration profile**.

- **Make targets:**
  - `make build_aerial` — default (FAPI 10_02 + -DSCF_FAPI_10_04_SRS=ON).
  - `PROFILE=fapi_10_02.conf make build_aerial` — FAPI 10_02 only
  - `PROFILE=fapi_10_04.conf make build_aerial` — FAPI 10.04 (SCF_FAPI_10_04=ON).

- **Profile variable:**  
  Profiles are defined in `install/cmake-profiles/<name>.conf` (each sets `BUILD_PRESET` and `PROFILE_CMAKE_FLAGS`). See `install/cmake-profiles/README.md` for adding custom profiles.

## Scripts

The make targets run executable scripts in this directory (e.g. `install_aerial_kernel.sh`, `install_drivers.sh`, `install_services.sh`, `setup_net_ifs.sh`, `quickstart-aerial.sh`, `quickstart-oai.sh`). You can run any of these scripts directly. Each script supports a `-h` or `--help` option for usage and options.
