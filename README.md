# TensixFlow: An MLIR Compiler for Tile-Native LLM Kernels on Tenstorrent-Style Spatial Accelerators

> Lowers fused transformer operators into NoC-aware, circular-buffered reader/compute/writer kernels inspired by TT-Metalium and TT-MLIR.

TensixFlow is a from-scratch C++/MLIR compiler stack for Tenstorrent-style spatial accelerators. It takes hand-written MLIR in a custom `tensixflow` dialect, fuses transformer subgraphs, infers tile layouts, plans NoC-aware sharding under an L1 budget, allocates circular buffers, and emits TT-Metalium-style kernels plus a static performance report. The modeled device is a single Wormhole n150-class part: 8x8 core grid, 32x32 tiles, roughly 1 MB of L1, bf16/fp32 data.

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

Then configure and build:

```sh
cmake -G Ninja -B build -DMLIR_DIR=$MLIR_DIR -DLLVM_DIR=$LLVM_DIR
cmake --build build
```

Built tools land in `build/bin/` (for example `build/bin/tensixflow-opt`).

Once the lit/FileCheck test infrastructure is in place, run the IR test suite with:

```sh
cmake --build build --target check-tensixflow
```

## Status / honesty

This is a compiler, a functional tile-level simulator, and a code generator. It is not a hardware run.

- Generated TT-Metalium kernels (`host.cpp`, `reader.cpp`, `compute.cpp`, `writer.cpp`) are validated **structurally only**: golden FileCheck patterns against the current tt-metal API, plus a compile-only syntax check against signature-only stub headers. They are not executed on hardware, and they are not claimed to be numerically verified.
- Numeric correctness is validated at the **schedule-IR level**. The functional simulator runs the lowered reader/compute/writer tile program over modeled circular buffers (bf16 spatial model) and is compared against an fp32-exact NumPy golden, using PCC as the primary metric.

In short: the schedule is numerically validated; the generated kernels are structurally aligned and compile-checked. See `docs/architecture.md` for the full disclaimer.
