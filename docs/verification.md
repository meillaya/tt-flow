# Verification

Every claim below comes from a check that runs on a CPU-only machine. The checks share one command, `devenv shell -- bash scripts/test.sh`, which builds the project and runs the lit suite, the GoogleTest suites, the Python reference checks, the prose checker, the device-spec check, and the emitted-project build. This page states what each layer proves and where the proof stops.

## Checks

| Layer | Command | Status on this machine | What it proves |
|---|---|---|---|
| Device data | `python3 tools/gen_device_spec.py --check` | checked | `spec/devices.json` has the required fields, every numeric group names a source, and the generated C++ header and Python module match the JSON. |
| Device unit tests | part of `ctest` | checked | The scheduler reads the per-device grid, usable L1, clock, matrix throughput, and DRAM bandwidth, and the circular-buffer allocation never exceeds the usable L1 budget. |
| IR suite | `cmake --build build --target check-tensixflow` | checked | The dialect parses and prints, verifiers reject malformed ops, layout inference attaches the device grid, both fusion patterns fire, the schedule attribute round-trips, the schedule IR lowers, and the emitter text carries the expected API tokens. |
| C++ unit tests | `ctest --test-dir build --output-on-failure` | checked | The plan schema serializes and parses, a missing required field is rejected, the scheduler planners obey the L1 budget, the rounding contract holds bit for bit, the simulator runs, and the cost model returns finite results for degenerate and normal plans. |
| Python reference | `python3 -m pytest python -q` | checked | The schedule-level simulator matches a NumPy golden within the stated PCC floor, the comparison cannot degenerate, the Python DSL builds IR structurally identical to the handwritten example, and the Python rounding twin matches the C++ rounding. See [the reference numerics](#reference-numerics). |
| Emitted-project build | `bash scripts/build_generated.sh` after `scripts/fetch_tt_metal.sh` | checked | A freshly emitted matmul project configures, compiles, and links against the pinned TT-Metalium release headers and `libtt_metal.so`. No stub headers remain in the repository. |
| Emitted-program run, 64 cores | `bash scripts/run_generated.sh` after the build | checked | The emitted matmul on the n150 8x8 grid runs under ttsim v1.15.1: PCC 0.9890072 against its own CPU reference, status `pass`, exit 0. |
| Emitted-program run, 16 of 64 cores | the same commands on `examples/matmul_small.mlir` | checked | The small shape uses 16 cores with one output tile each: PCC 0.999301, status `pass`, exit 0. This covers the idle-core and uneven-split paths. |
| Emission block | unit test, lit suite, report validation | checked | The plan JSON records the mapping, cores used, max tiles per core, and the upper-bound DRAM reads and writes of the emitted access pattern. The report rejects a stale byte count or an unknown mapping. |
| Fused-program run | the same commands on `examples/rmsnorm_matmul_small.mlir` | checked | The fused RMSNorm+MatMul program stages the activation rows in L1, normalizes them in the compute kernel, feeds the matmul from the normalized buffer, and runs under ttsim: PCC 0.9993997, status `pass`, exit 0. |
| Fused feasibility guard | lit suite | checked | The large `examples/rmsnorm_matmul.mlir` is rejected with a diagnostic that names its 256 staged tiles, 128 weight tiles, 128 gamma tiles, and the 700 available tiles per core. The plan JSON records `fused_feasible: false` for the same plan, so the plan and the emitter agree. |
| Fused-MLP run | the same commands on `examples/mlp_block_small.mlir` | checked | The full chain runs in one kernel: RMSNorm, first matmul with bias and accurate GELU, the GELU output kept in L1, second matmul, residual add. The run reports PCC 0.999981, status `pass`, exit 0. The large `examples/mlp_block.mlir` is rejected with its 3,072 intermediate tiles and 59,284 total tiles against 700 available. |
| Simulator toolchain | the same commands on the pinned upstream example | checked | The pinned release's own `matmul_single_core` example also builds and runs under ttsim: 13 JIT-compiled kernels, PCC 0.9791022 against its CPU golden, exit 0. This shows the result does not depend on this project's emitter. |
| TTIR parse | `python3 -m pytest python -q` with the wheel present | checked, wheel fetched | Emitted TTIR text parses with the pinned tt-mlir Python bindings. The test skips with a reason when the wheel or Python 3.11 is absent. See [ttir.md](ttir.md). |
| TTIR import | lit suite | checked | A documented bare-tensor TTIR subset imports into the tensixflow dialect and runs the planning pipeline. Twelve rejected constructs each produce a named diagnostic. See [ttir.md](ttir.md). |
| Prose | `python3 tools/check-prose.py` | checked for the whole tree | Every checked file follows the house prose rules. |

The emitted-project build compiles and links. A successful link against `libtt_metal.so` shows the host code resolves the release ABI. Runtime behavior is the next layer, the ttsim run.

## Reference numerics

The correctness check drives the functional tile simulator, which executes the schedule IR over modeled bf16 circular buffers. The simulator rounds values on push into a circular buffer, accumulates in fp32 within a tile, rounds each output tile to bf16, and stores fused intermediates as bf16.

The golden reference runs in fp32 and keeps every intermediate in fp32. The two paths differ on purpose.

The bar is one number for every run on every target:

```
PCC(simulator, golden) >= 0.9875
```

Three checks keep the bar from passing for nothing:

- The golden's fused intermediate is not representable in bf16, and the simulator output is, so the bf16 rounding happened on the simulator side only.
- `max_abs_diff(simulator, golden) > 0`, so the two outputs are not bit-identical.
- PCC stays below a trip-wire that the harness calibrates above the measured good-case PCC and below 1.0.

The targets are `rmsnorm_matmul` and the fused MLP prefix. The input shapes are reduced to 40 to 64 elements per dimension, because the non-BLAS tiled simulator cannot run the example shapes in a test. The simulator derives its tile grid from the input shapes it is given, and the IR supplies the fusion decision and the circular-buffer structure.

## What no CPU-only check proves

Device execution. The development machine has no Tenstorrent card, no `/dev/tenstorrent` node, and no `tt-smi`, so no kernel from this project has run on silicon. The hardware CI job is gated on a self-hosted runner with a card and has not run.

Measured performance. The cost model computes a roofline projection from published device limits and tensor shapes. No timing figure comes from a card. The report labels every cost figure as projected.

Fused kernel numerics on silicon. The fused program runs under ttsim, which is bit-exact numerically against the ISA but is not cycle-accurate and is not a card. A fused run on silicon remains untested, and the fused path is limited to shapes whose staging fits the per-core L1 budget.

The full MLP tail on silicon. The full `fused_mlp` chain runs under ttsim end to end, with a golden that applies RMSNorm, both matmuls, the bias add, the accurate GELU, and the residual add. The simulator execution is a functional check under simulation. The chain has not run on a card, and the emitted MLP keeps both weight blocks resident or staged without multicast.

Silicon behavior in any form. No kernel from this project has run on a card. ttsim is a CPU simulator that models the device, so a passing ttsim run shows functional execution under simulation. It says nothing about silicon timing or silicon behavior.

Activation multicast. The emitted matmul reuses its weight column block and multicasts it down the used row blocks, which is checked under ttsim. The activation row block is still read once per used column block, so activation replication remains. The plan's `noc` block counts the activation once; the `emission` block counts the emitted reads. The two differ by that replication on purpose.

The device backend. `scripts/device_probe.py` exits 2 on this machine because there is no card. Its success path, the `--device` backend, and the device JSON result are untested. The card path of `scripts/hardware_acceptance.sh`, and the table it prints for `docs/device.md`, are untested. On this machine the script stops at the probe with exit 2.
