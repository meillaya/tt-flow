# The hardware path (Stretch C, conditional)

> **There is no Tenstorrent device on the box this repo is developed on.** The
> whole MVP - compiler, simulator, golden harness, reports - is CPU-only and
> passes with no hardware in the loop. This document is the path that opens up
> IF a Wormhole n150 ever becomes available: a validated bring-up runbook, the
> compile-only validation that stands in for hardware today, and the CI gate
> that runs the real on-device branch only where hardware exists.

The hardware wave is accepted DISJUNCTIVELY (drafts D12/D19):

```
(run on a real device AND PCC >= 0.9875 vs the golden)
OR
(validated bring-up runbook + compile-only kernel check + hardware-gated CI)
```

The disjunction is wired as an explicit if/else (todo 57), NEVER `&&...||`:

```bash
if [ -e /dev/tenstorrent ]; then
  <on-device branch: todo 55, may fail honestly>
else
  bash scripts/hw_bringup_dryrun.sh && cmake --build build --target check-generated-kernels-compile
fi
```

so a present-but-failing device run can never fall through to the passing
no-device branch and get masked.

## Branch 1: the no-device branch (ALWAYS available, must PASS)

This branch is what runs on this box and in ordinary CI. It has three parts,
none of which requires hardware, a tt-metal build, or network:

1. **The bring-up dry-run.** `bash scripts/hw_bringup_dryrun.sh` prints the
   validated bring-up steps (below), probes `/dev/tenstorrent`, and exits 0.
   It never depends on tt-metal; it only prints what WOULD be done.

2. **The compile-only kernel validation (reused from todo 33 / D16).**

   ```bash
   cmake --build build --target check-generated-kernels-compile
   ```

   This compiles the generated kernels (`kernels/generated/<op>/*.cpp`,
   emitted by `--tensixflow-emit-metalium=output-dir=`) plus the current-API
   baseline `-fsyntax-only` against the signature-only stub headers pinned to
   tt-metal `@df36cd6` (`kernels/handwritten_baselines/tt_metal_stubs/`).
   The check is non-vacuous by construction: it asserts the current-API
   baseline COMPILES while a kernel referencing the removed legacy symbol
   `EnqueueProgram` FAILS (naming the symbol). A typo in a generated kernel
   fails this target - it is a real gate, not a formality.

3. **The hardware-gated CI job.** `.github/workflows/ci-hardware.yml` defines
   a job that runs ONLY when both gates hold: the PR carries the
   `hardware.present` label AND a self-hosted runner tagged
   `tenstorrent` / `wormhole-n150` is available to pick it up. On this
   (no-device) box and in ordinary CI the job never runs. See
   [the CI gate](#the-hardware-gated-ci-job).

## Branch 2: the on-device branch (gated, todo 55)

IF a device is present, the on-device branch does what this document's runbook
prepares:

1. Compile the generated `kernels/generated/fused_rmsnorm_matmul` project
   (host.cpp + reader.cpp + compute.cpp + writer.cpp) against the REAL
   tt-metal runtime - NOT the stubs - using the project's own
   `CMakeLists.txt` pointed at the tt-metal build tree.
2. Enqueue it with `distributed::EnqueueMeshWorkload` on the Wormhole n150,
   read back the output tensor.
3. Validate PCC >= 0.9875 against the same NumPy golden reference the CPU
   harness uses (the identical bar; todo 40), and record the result in the
   [results log](#on-device-results-log) below.

The MVP codegen is NOT modified for the device: the emitter output that
compiles against the stubs on this box is exactly what gets built against the
real runtime.

## The bring-up runbook (validated steps)

Grounded in tt-metal `INSTALLING.md` at the pinned revision `@df36cd6`
(`df36cd6a7434854780eee60d81179535160342fc`). Five steps from zero to a
running generated kernel project:

### Step 1 - Obtain/access a Wormhole n150

Get shell access to a Linux box (Ubuntu 22.04 is the documented OS) with a
Tenstorrent Wormhole n150 card seated per the hardware setup guide at
`docs.tenstorrent.com`. This repo models exactly that part: 8x8 Tensix grid,
32x32 tiles, bf16/fp32, `MathFidelity::HiFi4`.

### Step 2 - Install the KMD (kernel-mode driver) and host tools

The KMD is what creates `/dev/tenstorrent` - the probe target the whole
disjunction keys on. Recommended route is the TT-Installer script
(INSTALLING.md prerequisite #2):

```bash
curl -fsSL https://github.com/tenstorrent/tt-installer/releases/download/v2.1.0/install.sh -O
chmod +x install.sh
./install.sh --install-container-runtime=no
```

This installs TT-KMD (>= v2.8.0 on the documented boards), `tt-smi`, and
`tt-flash`. Manual installation is documented at
`docs.tenstorrent.com/getting-started/manual-software-install.html`. After
the KMD loads, `ls -l /dev/tenstorrent` shows the device node.

### Step 3 - Build tt-metal WITH the runtime (from source)

The runtime (host API, device dispatch, `distributed::EnqueueMeshWorkload`)
is required to actually run kernels on device. Note this step is the ONLY
heavy build in the whole path, and per D19 it is a gated part of the
on-device branch - never a prerequisite for the no-device fallback:

```bash
git clone https://github.com/tenstorrent/tt-metal.git --recurse-submodules
cd tt-metal && git checkout df36cd6
./build_metal.sh     # or: cmake -G Ninja -DCMAKE_BUILD_TYPE=Release + ninja
```

### Step 4 - Flash the firmware

Flash the device firmware bundle before first use. TT-Installer drives
`tt-flash` (INSTALLING.md's board table pairs Wormhole-class systems with
`fw_pack-19.8.1.fwbundle`, i.e. firmware v19.8.1):

```bash
./install.sh --fw-version=19.8.1 --kmd-version=2.8.0 --install-container-runtime=no
tt-smi -s            # confirm the card, firmware version, and health
```

### Step 5 - Run the generated project

Build `kernels/generated/fused_rmsnorm_matmul/` against the real tt-metal
build tree (its checked-in `CMakeLists.txt` needs the tt-metal include/lib
paths instead of the stubs), enqueue with
`distributed::EnqueueMeshWorkload`, read back, and check PCC against the
golden:

```bash
# generate fresh project sources from the compiler (optional; they are
# also checked in under kernels/generated/fused_rmsnorm_matmul/)
build/bin/tensixflow-opt \
  --tensixflow-infer-tile-layout --tensixflow-fuse-rmsnorm-matmul \
  --tensixflow-plan-noc-sharding --tensixflow-allocate-circular-buffers \
  --tensixflow-lower-to-ttkernel \
  --tensixflow-emit-metalium=output-dir=/tmp/fused_rmsnorm_matmul \
  examples/rmsnorm_matmul.mlir

# then: build the project vs the tt-metal runtime, run, read back, and
# compare with the NumPy golden (PCC >= 0.9875) - the todo-55 on-device
# branch. Record the outcome in the results log below.
```

A dry-run of these five steps is always runnable via
`bash scripts/hw_bringup_dryrun.sh` (prints them, detects the missing
device, exits 0).

## The hardware-gated CI job

`.github/workflows/ci-hardware.yml` adds exactly one job, `on-device`, gated
TWICE so it can never run on a no-device box:

- **Label gate (job-level `if`):** the PR must carry the `hardware.present`
  label. Without it the job is skipped on every push/PR.
- **Runner gate (`runs-on`):** `[self-hosted, tenstorrent, wormhole-n150]` -
  only a self-hosted runner tagged as carrying the hardware can pick the job
  up. GitHub-hosted runners never match.

When (and only when) both hold, the job runs the todo-55 on-device branch:
it asserts `/dev/tenstorrent` exists (a gated job on a deviceless runner
fails honestly rather than silently passing), then executes the on-device
entry point (`scripts/hw_run_on_device.sh`, created by the todo-55 work when
a device box is available) to compile the generated project against real
tt-metal, run it, and assert PCC >= 0.9875. Per D19/D25 there is NO fallback
to the no-device branch inside this job: on a device box only the on-device
branch runs and may fail honestly.

The ordinary `ci.yml` (CPU-only) is untouched and still runs on every push.

## On-device results log

To be filled by the todo-55 on-device run (skipped while no device is owned):

| Date | Device | Project | Enqueue | PCC vs golden (bar 0.9875) | Status |
|------|--------|---------|---------|----------------------------|--------|
| -    | (none owned) | fused_rmsnorm_matmul | not run | n/a (schedule-level PCC only) | SKIPPED (no device) |

## Honesty

On this box the generated TT-Metalium kernels are validated STRUCTURALLY
(FileCheck against the current tt-metal API) and by the compile-only check
against signature-only stubs. They are never executed, and nothing here is
claimed to be hardware-verified numerics. Numerical correctness is proven at
the schedule level: the functional simulator vs the fp32-exact golden, PCC
floor 0.9875 (see `docs/architecture.md`, `docs/performance_model.md`,
`docs/tenstorrent_mapping.md`).
