# TensixFlow: An MLIR Compiler for Tile-Native LLM Kernels on Tenstorrent-Style Spatial Accelerators

> Lowers fused transformer operators into NoC-aware, circular-buffered reader/compute/writer kernels inspired by TT-Metalium and TT-MLIR.

TensixFlow is a from-scratch C++/MLIR compiler stack for Tenstorrent-style spatial accelerators. It takes hand-written MLIR in a custom `tensixflow` dialect, fuses transformer subgraphs, infers tile layouts, plans NoC-aware sharding under an L1 budget, allocates circular buffers, and emits TT-Metalium-style kernels plus a static performance report. The modeled device is a single Wormhole n150-class part: 8x8 core grid, 32x32 tiles, bf16/fp32 data, and a conservative 1 MiB modeled L1 budget per core (a tunable constant; the literal n150 L1 is larger, about 1.5 MB).

## Resume bullets

Three phrasings of the same work, for different audiences:

1. Built TensixFlow, a C++/MLIR compiler for Tenstorrent-style spatial AI accelerators that lowers fused transformer blocks into tile-native, NoC-aware TT-Metalium-style reader/compute/writer kernels.
2. Implemented MLIR dialects, fusion passes, tile-layout inference, NoC-aware sharding, circular-buffer allocation, TT-Metalium code generation, and a static performance model for fused RMSNorm+MatMul and MLP kernels targeting Tensix-like architectures.
3. Developed a mini TT-Forge-inspired compiler stack lowering transformer subgraphs through custom MLIR dialects into TT-Metalium-compatible kernels, reducing intermediate DRAM traffic through operator fusion and generating visual performance reports for core-grid placement, NoC movement, and circular-buffer utilization.

## Repository layout

```
include/tensixflow/            Public headers: dialect, ops, passes, codegen, simulator
lib/tensixflow/                Implementation of the dialect, passes, and backends
tools/
  tensixflow-opt/              MLIR optimizer driver (dialect + pass entry point)
  tensixflow-translate/        IR import/export translation tool
  tensixflow-report/           Performance report + HTML visualizer generator
test/                          lit/FileCheck IR tests (run via check-tensixflow)
examples/                      Hand-written .mlir inputs (rmsnorm_matmul, mlp_block)
benchmarks/                    C++ timing loops for the fused kernels
kernels/
  generated/                   TT-Metalium-style kernels emitted by codegen
  handwritten_baselines/       Reference kernels for comparison
reports/                       Generated performance reports (text + JSON + HTML)
docs/                          architecture, tenstorrent_mapping, performance_model
scripts/                       Helper scripts (environment, remote-box bring-up)
python/                        Python DSL + NumPy golden + correctness harness
```

## Building

TensixFlow builds against LLVM/MLIR 20.1.8 provided by a nix dev shell. Enter the shell first so `MLIR_DIR`, `LLVM_DIR`, and `mlir-tblgen` are on the environment:

```sh
nix develop        # or: devenv shell
```

Then configure and build. The `LLVM_EXTERNAL_LIT` pin points lit at the real python script (the nix `bin/lit` is a bash wrapper that breaks under LLVM's `add_lit_target`):

```sh
cmake -G Ninja -B build \
  -DMLIR_DIR=$MLIR_DIR -DLLVM_DIR=$LLVM_DIR \
  -DLLVM_EXTERNAL_LIT="$(dirname "$(which lit)")/.lit-wrapped"
cmake --build build
```

Built tools land in `build/bin/`:

| Tool | Purpose |
|---|---|
| `tensixflow-opt` | the MLIR optimizer driver: dialect + the seven-pass lowering pipeline |
| `tensixflow-translate` | serializes a planned module to `plan.json` (`--tensixflow-to-json`) |
| `tensixflow-simulate` | runs the pipeline + functional tile simulator over `.tft` tensors |
| `tensixflow-report` | runs the static perf model and renders the offline HTML report |

## Running the pipeline

Lower an example all the way to a standalone generated kernel project:

```sh
build/bin/tensixflow-opt \
  --tensixflow-infer-tile-layout \
  --tensixflow-fuse-rmsnorm-matmul \
  --tensixflow-plan-noc-sharding \
  --tensixflow-allocate-circular-buffers \
  --tensixflow-lower-to-ttkernel \
  --tensixflow-emit-metalium=output-dir=/tmp/rmsnorm_matmul \
  examples/rmsnorm_matmul.mlir
```

The output directory gets `host.cpp`, `reader.cpp`, `compute.cpp`, `writer.cpp`, a compile-only `CMakeLists.txt`, and a `README.md`. Omit `output-dir` to print the four kernels to stdout separated by `// ===== file: <name>.cpp =====` banners. A pre-generated project is checked in at `kernels/generated/fused_rmsnorm_matmul/`.

## Testing

Four layers, each green on a CPU-only box:

```sh
# 1. lit/FileCheck IR suite (23 tests: dialect, passes, schedule IR, codegen)
cmake --build build --target check-tensixflow

# 2. C++ unit suites (gtest): schema, scheduler planners, rounding, simulator, perf model
(cd build && ctest --output-on-failure)

# 3. generated-kernel compile check: the current-API baseline compiles,
#    a legacy-API fixture is rejected, and every generated project compiles
cmake --build build --target check-generated-kernels-compile

# 4. Python correctness harness (bf16 simulator vs fp32-exact golden, PCC >= 0.9875).
#    The rounding pytest reads a file the gtest writes, so run ctest -R rounding first.
(cd build && ctest -R rounding)
uv run --with numpy --with pytest pytest python/tensixflow/tests/ -q
```

No system Python ships numpy and pytest together here, so the harness runs under `uv`'s ephemeral environment (the command above). The harness drives the C++ simulator through `tensixflow-simulate` and compares against the NumPy golden; see `docs/performance_model.md` for the rounding contract.

## Benchmarks and the report

The benchmarks are simple timing loops, not a framework. Each prints the static perf model's unfused-vs-fused report (DRAM reads/writes, kernel launches, bottleneck, suggestions) and then times the functional simulator:

```sh
cmake --build build --target bench_rmsnorm_matmul bench_mlp_block
./build/benchmarks/bench_rmsnorm_matmul
./build/benchmarks/bench_mlp_block
```

Both carry an explicit disclaimer: the perf figures are static analytic estimates and the simulator timing is a functional-execution measure. Neither is wall-clock hardware performance.

### The report

`tensixflow-report` turns a `plan.json` into a single self-contained HTML visualizer. It inlines all CSS and JS, embeds the plan JSON, and makes zero network requests, so it opens offline from `file://`. It renders six sections: `#core-grid` (8x8 placement), `#shards`, `#noc-routes`, `#cb-usage`, `#timeline` (the reader/compute/writer pipeline), and `#bottlenecks` (the bound classification). The tool runs the static perf model to fill the perf section before rendering.

```sh
# plan.json: stop before lower-to-ttkernel (that pass erases the source func the
# translator walks for graph_id + tensors)
build/bin/tensixflow-opt \
  --tensixflow-plan-noc-sharding --tensixflow-allocate-circular-buffers \
  examples/rmsnorm_matmul.mlir \
  | build/bin/tensixflow-translate --tensixflow-to-json > reports/rmsnorm_matmul_plan.json

build/bin/tensixflow-report \
  --in reports/rmsnorm_matmul_plan.json \
  --out reports/rmsnorm_matmul_report.html

# or regenerate the checked-in report from the committed plan.json
cmake --build build --target regenerate-rmsnorm-report
```

The committed `reports/rmsnorm_matmul_report.html` is the deliverable artifact. Open it directly in a browser; there is no server and no CDN.

## Status / honesty

This is a compiler, a functional tile-level simulator, and a code generator. It is not a hardware run. There is no Tenstorrent device in the loop.

- Generated TT-Metalium kernels (`host.cpp`, `reader.cpp`, `compute.cpp`, `writer.cpp`) are validated **structurally only**: golden FileCheck patterns against the current tt-metal API, plus a compile-only `-fsyntax-only` check against signature-only stub headers pinned to tt-metal `@df36cd6`. They are **never executed** on hardware, and they are not claimed to be numerically verified.
- Numeric correctness is validated at the **schedule-IR level**. The functional simulator runs the lowered reader/compute/writer tile program over modeled circular buffers (bf16 spatial model) and is compared against an fp32-exact NumPy golden, with PCC as the primary metric and a floor of 0.9875.

In short: the schedule is numerically validated; the generated kernels are structurally aligned and compile-checked, never run. The performance figures are static analytic estimates, not measured hardware numbers. See `docs/architecture.md` for the full disclaimer, `docs/tenstorrent_mapping.md` for the API mapping, and `docs/performance_model.md` for the L1 budget, the formulas, and the rounding contract.
