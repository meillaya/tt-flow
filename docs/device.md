# Device bring-up

This page starts at a machine with a Tenstorrent card and ends at a generated matmul program that reports its own PCC. The development machine has no card, so every step that touches one is untested here. The steps that need only a CPU ran on this machine and are marked tested.

## Host requirements

`spec/devices.json` records these minimums in the toolchain block of every device. They come from tt-metal `INSTALLING.md`.

| Item | Required |
|---|---|
| OS | Ubuntu 22.04 |
| IOMMU | enabled |
| Kernel-mode driver | tt-kmd 2.8.0 or newer |
| Firmware | 19.8.1 |
| tt-smi | 5.0.0 or newer |
| Device nodes | `/dev/tenstorrent/N`, one per card |
| CPU | x86_64 |

`deps/tt-metal.lock` pins the Ubuntu 24.04 amd64 release assets. The same lock records the four Ubuntu 22.04 asset names and sha256 digests, so a 22.04 card host can switch to them without another lookup.

## Steps

### 1. Prepare the host (untested, needs a card)

Enable IOMMU in the firmware and the kernel command line. An enabled host has entries under `/sys/kernel/iommu_groups`.

Install tt-kmd, the 19.8.1 firmware bundle, and tt-smi with Tenstorrent's installer or the manual instructions in tt-metal `INSTALLING.md`. Load the driver and flash the firmware. `tt-smi -s` prints the board when the host is ready.

A host that fails here fails before any TensixFlow step. Fix the driver or firmware version and re-run `tt-smi -s`.

### 2. Probe the card (card path untested, no-card path tested)

```sh
devenv shell -- python3 scripts/device_probe.py
```

With a supported card the probe parses `tt-smi -s` JSON, lists `/dev/tenstorrent/N`, matches the reported board to a `spec/devices.json` key or alias, and checks the reported firmware and tt-kmd versions against the recorded minimums. Exit 0 means the card matches the published spec. Exit 1 means a card is present but a board name is unknown, a version is below the minimum, or a version is missing from the snapshot. Exit 2 means the probe cannot run at all: no `tt-smi` on `PATH` or no device node. The probe never guesses an unknown board.

Expected end state: exit 0 and a printed device key that exists in `spec/devices.json`.

### 3. Fetch the pinned artifacts (tested, CPU only)

```sh
devenv shell -- bash scripts/fetch_tt_metal.sh
```

The script downloads the four `tt-metalium` debs at `v0.82.0-rc2`, verifies each sha256 against `deps/tt-metal.lock`, extracts them under `.deps/tt-metal/0.82.0-rc2/usr`, and downloads ttsim `v1.15.1` `libttsim_wh.so` plus the matching `soc_descriptor.yaml` into `.deps/ttsim/`. It then writes `.deps/env.sh`, which exports `CMAKE_PREFIX_PATH`, `LD_LIBRARY_PATH`, `TT_METAL_RUNTIME_ROOT`, `TT_METAL_HOME`, a writable `TT_METAL_CACHE`, `TT_METAL_SIMULATOR`, and `TT_METAL_SLOW_DISPATCH_MODE=1`.

The first run needs network. A re-run prints `cache complete` and touches nothing. This step needs no card.

### 4. Emit, compile, and link the project (tested, CPU only)

```sh
devenv shell -- bash scripts/build_generated.sh
```

The script emits a fresh matmul project from `examples/matmul.mlir` into `build-generated/project/`, configures it with CMake against the extracted release package, compiles the host program, and links it against `libtt_metal.so`. It checks that `build-generated/project/build/tensixflow_generated` exists, prints `built` for that path, and prints the matching `run_generated.sh` command. Without `.deps/`, the script prints the fetch command and exits 0 with an explicit skip message. The link step needs `TT_METAL_SYSTEM_LINK_FLAGS`, which `devenv.nix` sets; the script fails loudly when the variable is unset.

A failure here is a compile error in the emitted C++ or a missing system dependency. Recovery: fix the emitter or the environment and re-run; the script reconfigures from scratch.

### 5. Run the program (tested, CPU only)

Without a card, the program runs under ttsim:

```sh
devenv shell -- bash scripts/run_generated.sh
```

The simulator is the supported no-card run path. The script sources `.deps/env.sh`, runs the generated binary from the project directory under `libttsim_wh.so` with `TT_METAL_SLOW_DISPATCH_MODE=1`, parses the program's `TENSIXFLOW_RESULT` line, and prints one JSON object with `status`, `backend`, and `pcc`. It exits 0 only on a pass, 1 on a failed run, and 2 when no backend is available.

ttsim loads `soc_descriptor.yaml` from the directory that holds `libttsim_wh.so`, and `scripts/fetch_tt_metal.sh` puts the matching descriptor there. `TT_METAL_CACHE` must be writable; `.deps/env.sh` points it inside `.deps/`. A `No Physical Grouping Descriptor` warning on stderr is harmless.

The default example is `examples/matmul.mlir`: M=256, N=1024, K=512, bf16, emitted on one core as 8 x 32 output tiles over 16 K-tiles. The recorded run on this machine printed `Metalium vs Golden -- PCC = 0.9890072`, printed a `TENSIXFLOW_RESULT` line with status `pass` and threshold 0.97, and the script printed `{"status":"pass","backend":"ttsim","pcc":0.9890072}` with exit 0.

As a separate check, the upstream `matmul_single_core` example from the `v0.82.0-rc2` examples deb also ran under ttsim v1.15.1 on this machine. It JIT-compiled 13 kernels, reported `Metalium vs Golden -- PCC = 0.9791022` with `Test Passed`, and exited 0.

With a card, pass `--device`:

```sh
devenv shell -- bash scripts/run_generated.sh --device
```

The device backend requires `scripts/device_probe.py` to confirm a card and runs without the simulator variables. The program computes deterministic inputs, a CPU reference matmul, then the emitted kernels, and prints the PCC between the device result and the reference. The program's own check accepts a run when the PCC reaches 0.97, and `scripts/run_generated.sh` reports a pass only when the program prints its pass status. A device run is untested here. With no card, `--device` exits 2 with a message that names the probe and points at the default ttsim backend; it never falls back silently.

### 6. Record the result (untested)

A device run writes one JSON object: the backend, the status, and the PCC. Add the run to the table below when a card is available. A CPU simulator run does not go in this table.

| Date | Device | Backend | Status | PCC | Notes |
|---|---|---|---|---|---|
| - | none | - | - | - | no card on the development machine |

## Environment note

The release debs declare system dependencies: `nlohmann-json3-dev`, OpenMPI, libhwloc, libnuma, and zlib. A plain Ubuntu card host installs them from apt, which is what the release targets. This development host is not Ubuntu, so the devenv shell supplies the build dependencies; the compiler and build path do not depend on a system install.

## Related documents

[hardware.md](hardware.md) lists the device values and their sources. [verification.md](verification.md) lists what the CPU-only checks prove. [architecture.md](architecture.md) covers the emitted project layout.
