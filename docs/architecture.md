# TensixFlow Architecture

TensixFlow is a from-scratch C++/MLIR compiler stack for Tenstorrent-style spatial
accelerators. It lowers hand-written MLIR in a custom `tensixflow` dialect through
fusion, tile-layout inference, NoC-aware sharding, and circular-buffer allocation,
then emits TT-Metalium-style reader/compute/writer kernels plus a static performance
report. The modeled device is a single Wormhole n150-class part: an 8x8 core grid,
32x32 tiles, a conservative 1 MiB of modeled L1 per core, and bf16 data with fp32
accumulation.

Three things run here, and only three: a compiler, a functional tile-level simulator,
and a code generator. There is no hardware run. Read the [Honesty disclaimer](#honesty-disclaimer)
at the bottom of this page before quoting any number.

## Pipeline overview

The lowering spine is seven `tensixflow-opt` passes. The first five stay at the
tensor and plan level, the sixth lowers to the tile-level schedule IR, and the
seventh emits standalone TT-Metalium kernel sources. Two consumers tap the schedule
IR and the plan: the functional simulator (numeric correctness) and the performance
model plus HTML report (static estimates).

```
            Hand-written MLIR in the tensixflow dialect
            (examples/rmsnorm_matmul.mlir, examples/mlp_block.mlir)
                                  |
                                  v
  --tensixflow-infer-tile-layout        attach #tensixflow.tile_layout
                                        (32x32 tile, 8x8 grid, DRAM/L1 space)
                                  |
                                  v
  --tensixflow-canonicalize             identity-add removal, constant folding
                                  |
                                  v
  --tensixflow-fuse-rmsnorm-matmul      rmsnorm+matmul -> fused_rmsnorm_matmul
                                        full MLP chain  -> fused_mlp
                                  |
                                  v
  --tensixflow-plan-noc-sharding        8x8 grid placement, weight strategy
                                        (replicate / shard / multicast),
                                        NoC traffic, tensixflow.schedule attr
                                  |
                                  v
  --tensixflow-allocate-circular-buffers  CB indices {c_0, c_1, c_16}, depth 160
                                  |
                                  v
  --tensixflow-lower-to-ttkernel        tile-level schedule IR
                                        (sched.kernel / reader / compute / writer)
                                  |
                +-----------------+-----------------+
                |                                   |
                v                                   v
  --tensixflow-emit-metalium            Functional tile simulator
  host.cpp, reader.cpp,                 (sim::runTileSimulator) executes
  compute.cpp, writer.cpp,              the schedule IR over modeled bf16
  CMakeLists.txt, README.md             circular buffers
                |                                   |
                v                                   v
  Structurally validated only           Compared against an fp32-exact
  (FileCheck goldens + compile-only     NumPy golden, PCC >= 0.9875
   against signature-only stubs)
```

The report path branches off after circular-buffer allocation, before the lowering
erases the source function:

```
  --tensixflow-plan-noc-sharding --tensixflow-allocate-circular-buffers
                                  |
                                  v
  tensixflow-translate --tensixflow-to-json        plan.json (plan + noc + cb)
                                  |
                                  v
  tensixflow-report --in plan.json --out report.html
        runs the static performance model (estimatePerf) to fill the perf
        section, then renders the self-contained offline HTML visualizer
```

## The tensixflow dialect

The custom out-of-tree dialect (namespace `mlir::tensixflow`) follows the
llvm-project standalone-example layout. Ops live in `include/tensixflow/TensixflowOps.td`:

| Op | Kind | Notes |
|---|---|---|
| `tensixflow.func` | container | isolated-from-above function, single-block body terminated by `tensixflow.return` |
| `tensixflow.return` | terminator | returns the function results |
| `tensixflow.constant` | value | typed constant, `ConstantLike` + foldable |
| `tensixflow.add` | elementwise | same-shape sum; canonicalizes away an identity zero operand |
| `tensixflow.gelu` | elementwise | exact-erf GELU/SiLU activation |
| `tensixflow.matmul` | compute | `[..., M, K] x [K, N] -> [..., M, N]`; verifier rejects mismatched contracting dims |
| `tensixflow.rmsnorm` | compute | RMSNorm over the last dim; verifier checks the 1-D gamma matches the last dim |
| `tensixflow.fused_rmsnorm_matmul` | fused | `matmul(rmsnorm(input, gamma), weight)` as one kernel |
| `tensixflow.fused_mlp` | fused | `(gelu(matmul(rmsnorm(x, gamma), w1) + b1) @ w2) + residual` as one kernel |

Every compute and fused op carries shape inference
(`InferShapedTypeOpInterface`) and a verifier. The dialect also defines two
attributes: the `#tensixflow.tile_layout` encoding (tile size, grid placement,
DRAM/L1 memory space) and the `#tensixflow.schedule` attribute that records the
plan (grid, sharding, weight strategy, circular buffers, NoC traffic).

## Tile-layout inference

`--tensixflow-infer-tile-layout` attaches a `#tensixflow.tile_layout<tile 32x32,
grid 8x8, mem DRAM|L1>` encoding across the graph. Function tensor arguments are
marked DRAM (they enter from off-chip), and op tensor results are marked L1 (they
live on-chip once scheduled). The pass is pure metadata: it never changes a shape.

## Fusion

`--tensixflow-fuse-rmsnorm-matmul` runs two greedy rewrite patterns. The MLP-chain
pattern (higher benefit) walks `rmsnorm -> matmul -> bias-add -> gelu -> matmul ->
residual-add` link by link, guarding single-use at every intermediate, and collapses
a full chain into one `tensixflow.fused_mlp`. A partial chain falls through to the
rmsnorm+matmul pattern, which rewrites a single-use `rmsnorm -> matmul` pair into one
`tensixflow.fused_rmsnorm_matmul`.

Each fusion erases the intermediate tensors and records the elimination on the fused
op (`tensixflow.eliminated_tensors`, `tensixflow.eliminated_bytes`) and on the module
(`tensixflow.fusion_report` count). This is the mechanism that keeps the normalized
intermediate resident in L1 instead of round-tripping DRAM, which is the whole point
of the exercise and the invariant the performance model reproduces.

## NoC-aware scheduler and circular-buffer allocation

`--tensixflow-plan-noc-sharding` maps the shapes onto the 8x8 grid and picks a weight
strategy under the L1 budget:

- **Replicate** when the weight fits the per-core circular-buffer budget (every core
  holds a full copy, zero multicast traffic).
- **Multicast** otherwise: column-shard N across the X-cores and move the whole weight
  once over the NoC via row master nodes, rather than re-reading it from DRAM per core.

The pass row-major shards the activation M across the Y-cores, runs the NoC traffic
estimator for both the unfused and fused paths, and writes everything into the
module-level `tensixflow.schedule` attribute.

`--tensixflow-allocate-circular-buffers` then assigns circular-buffer indices and
depths under the same budget. The default three-buffer pipeline uses the tt-metal
convention `in0 = c_0`, `in1 = c_1`, `out = c_16`, each at depth 160 tiles. The shared
L1-budget constants live in one header (`lib/tensixflow/Scheduler/L1Model.h`) and are
imported by both the allocator and the performance model. See
[performance_model.md](performance_model.md) for the budget arithmetic.

## The schedule IR

`--tensixflow-lower-to-ttkernel` is the true lowering: it consumes the source function
plus the schedule attribute and produces a tile-level schedule IR, then erases the
source function (the kernel becomes the sole symbol). The schedule dialect
(`tensixflow.sched.*`, defined in `include/tensixflow/TensixflowSchedule.td`) is the
contract between code generation and the simulator:

- **Containers:** `sched.kernel` holds three single-region program ops, one per
  tt-metal `kernel_main` role: `sched.reader`, `sched.compute`, `sched.writer`.
- **Structured loop:** `sched.tile_loop` carries a static trip count.
- **Circular-buffer ops:** `sched.cb_reserve_back`, `sched.cb_push_back`,
  `sched.cb_wait_front`, `sched.cb_pop_front`, keyed by static CB indices.
- **NoC ops:** `sched.noc_read`, `sched.noc_write`, `sched.noc_multicast`.
- **Compute ops:** `sched.matmul_tiles`, `sched.rmsnorm`, `sched.gelu`, over CB indices.

For the running `rmsnorm_matmul` example the reader and compute loops stream the K
dimension (trip 128) and the writer drains every output tile (trip 24576 = 64 x 384).

## TT-Metalium codegen

`--tensixflow-emit-metalium` walks the schedule IR and emits a standalone generated
project against the **current** tt-metal API: `host.cpp`, `reader.cpp`,
`compute.cpp`, `writer.cpp`, a compile-only `CMakeLists.txt`, and a `README.md`.
With no option it prints the four kernels to stdout separated by `// ===== file:
<name>.cpp =====` banners; with `output-dir=<dir>` it writes the whole tree. The host
uses the single `CreateKernel` with a `variant<ComputeConfig, DataMovementConfig>`,
`tt::CBIndex` circular buffers, and the `distributed::Enqueue*` mesh API. The full
op-by-op mapping and the legacy-API note are in
[tenstorrent_mapping.md](tenstorrent_mapping.md).

## Functional simulator and correctness

The functional tile simulator (`lib/tensixflow/Sim/TileSimulator.cpp`) executes the
schedule IR, not the emitted C++. It walks `sched.kernel`, reads the compute ops and
their CB indices, and performs the denoted tile computation over modeled circular
buffers: round-on-push into bf16 buffers, an fp32 reduce-and-scale RMSNorm, a tiled
matmul with fp32 accumulation within a tile and a bf16 round per output tile, and a
bf16 GELU. Fused intermediates between stages are stored as bf16.

The correctness harness (`python/tensixflow/harness.py`) drives the simulator through
the `tensixflow-simulate` tool and compares the result against the fp32-exact NumPy
golden (`python/tensixflow/golden.py`). The two paths are deliberately at different
precision so the PCC bar measures real divergence. The primary metric is PCC with a
floor of 0.9875; the rounding contract and the non-vacuity defense are documented in
[performance_model.md](performance_model.md).

## Performance model and report

The static performance model (`lib/tensixflow/Analysis/PerfModel.h`) is a first-order
analytic roofline over the plan: estimated cycles, DRAM/L1/NoC traffic, circular-buffer
occupancy, compute utilization, and a memory-vs-compute bound classification with
concrete suggestions. Every figure is a static estimate from the tensor shapes and the
shared L1 constants, never a measured hardware number. `tensixflow-report` runs the
model to fill the plan's perf section and renders the self-contained offline HTML
visualizer. The formulas are in [performance_model.md](performance_model.md).

## Honesty disclaimer

Nothing in this project executes the generated C++ kernels on hardware. There is no
Tenstorrent device in the loop, and no claim of on-device behavior anywhere.

The generated kernels (`host.cpp`, `reader.cpp`, `compute.cpp`, `writer.cpp`) are
validated **structurally** only. That validation is two layers: golden FileCheck
patterns that assert the current-API tokens (single `CreateKernel` + config variant,
`tt::CBIndex`, `distributed::Enqueue*`), and a compile-only `-fsyntax-only` check
against signature-only tt-metal stub headers pinned to `@df36cd6`. The stubs carry no
behavior. The kernels are **never executed**, and they are not claimed to be
numerically verified. A negative fixture that calls the removed legacy enqueue API is
deliberately kept failing, so the stub set cannot silently drift back to a legacy
surface.

Numeric correctness is established at the **schedule-IR level**, not the kernel level.
The functional simulator runs the lowered reader/compute/writer tile program over
modeled circular buffers (the bf16 spatial model) and is compared against an
fp32-exact NumPy golden, with PCC as the primary metric and a floor of 0.9875.

In short: the schedule is numerically validated; the generated kernels are
structurally aligned and compile-checked. The two are different claims, and this
project keeps them separate on purpose.
