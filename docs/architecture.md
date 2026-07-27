# TensixFlow Architecture

Stub. Full content lands in a later documentation pass. Each section below names what it will cover.

## Pipeline overview

The end-to-end flow from `tensixflow` dialect IR through fusion, tile-layout inference, NoC-aware scheduling, circular-buffer allocation, kernel lowering, and TT-Metalium codegen, plus where the functional simulator and performance model tap the schedule IR.

## The tensixflow dialect

The custom out-of-tree MLIR dialect: elementwise ops, matmul, rmsnorm, the fused `fused_rmsnorm_matmul` and `fused_mlp` ops, shape inference, verifiers, and the 32x32 tile-layout encoding attribute.

## Tile-layout inference

How `--infer-tile-layout` assigns the 32x32 tile-layout encoding (grid placement plus DRAM/L1 memory space) across the graph.

## Fusion

How `--fuse-rmsnorm-matmul` and the MLP-chain fusion eliminate intermediate tensors, keep results in on-chip SRAM, and record each elimination for the report.

## NoC-aware scheduler & circular-buffer allocation

How `--plan-noc-sharding` maps shapes onto the 8x8 grid, picks a weight strategy (replicate, shard, or column multicast) under the L1 budget, and how `--allocate-circular-buffers` assigns CB indices and depths.

## TT-Metalium codegen

How `--lower-to-ttkernel` produces the schedule IR and `--emit-metalium` emits the standalone generated project (host, reader, compute, writer, CMake) against the current tt-metal API.

## Functional simulator & correctness

The tile-level simulator that executes the schedule IR over modeled circular buffers (fp32 compute, bf16 round) and the correctness harness that compares it against the fp32-exact NumPy golden.

## Honesty disclaimer

Nothing in this project executes the generated C++ kernels. Generated kernels are validated structurally only (golden FileCheck against the current tt-metal API plus a compile-only check against stub headers). Numeric correctness is validated at the schedule-IR level, where the simulator is compared to the fp32-exact golden. The schedule is numerically validated; the kernels are structurally aligned and compile-checked, not numerically verified.
