# TensixFlow to Tenstorrent Mapping

TensixFlow targets the **current** TT-Metalium host and kernel API, studied from
tt-metal source at `@df36cd6`, and sits itself within the tt-mlir dialect family
studied at `@9f06802f`. This page maps each tensixflow op and pass to the TT-Metalium
surface it emits, records the removed legacy API the code generator deliberately
avoids, places each compilation stage relative to the TTIR / TTNN / D2M /
TTKernel / TTMetal dialect levels, maps every tensixflow op and pass across all
five levels, and traces the two tt-mlir lowering paths that leave TTIR.

Every mapping here is a structural alignment. The emitted kernels compile against
signature-only stub headers; they are not run on a device. See
[architecture.md](architecture.md#honesty-disclaimer) for the full disclaimer.

## tensixflow to TT-Metalium current API mapping

The emitted `host.cpp` lives in `namespace tt::tt_metal` and uses the current
distributed mesh API end to end:

| TensixFlow concept | Current TT-Metalium API |
|---|---|
| Open the modeled device | `distributed::MeshDevice::create_unit_mesh(device_id)` |
| Build a program | `CreateProgram()` |
| Register each of the three kernels | the **single** `CreateKernel(program, file, core_spec, variant<ComputeConfig, DataMovementConfig>)` |
| Compute kernel config (HiFi4, fp32 accumulate, K-tile trip) | `ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true, .compile_args = {...}}` |
| Reader / writer config (BRISC / NCRISC) | `DataMovementConfig{.processor, .noc, .compile_args}` |
| Circular buffers | `CreateCircularBuffer(program, grid, CircularBufferConfig(total, {{tt::CBIndex::c_N, bytes}, ...}))` |
| DRAM inputs / output | `CreateBuffer(device, InterleavedBufferConfig{.size, .page_size, .buffer_type = BufferType::DRAM})` |
| Per-kernel args | `SetCommonRuntimeArgs(...)` and `SetRuntimeArgs(...)` |
| Enqueue | `distributed::EnqueueWriteMeshBuffer`, `distributed::EnqueueMeshWorkload`, `distributed::EnqueueReadMeshBuffer` |

Circular-buffer indices use `tt::CBIndex::c_0 .. c_31` (the current enum), not the
older `tt::CB::c_in0` spelling. The default three-buffer pipeline maps `in0 = c_0`,
`in1 = c_1`, `out = c_16`, matching the scheduler's allocation.

### Kernel-side intrinsic mapping

The schedule IR ops lower one-to-one onto the global kernel intrinsics emitted in
`reader.cpp`, `compute.cpp`, and `writer.cpp`:

| Schedule IR op | Emitted TT-Metalium intrinsic | Role |
|---|---|---|
| `sched.cb_reserve_back` / `sched.cb_push_back` | `cb_reserve_back`, `cb_push_back`, `get_write_ptr` | stage tiles into a CB |
| `sched.cb_wait_front` / `sched.cb_pop_front` | `cb_wait_front`, `cb_pop_front`, `get_read_ptr` | take tiles out of a CB |
| `sched.noc_read` | `get_noc_addr`, `noc_async_read`, `noc_async_read_barrier` | unicast DRAM/L1 read |
| `sched.noc_write` | `get_noc_addr`, `noc_async_write`, `noc_async_write_barrier` | unicast write |
| `sched.noc_multicast` | `get_noc_multicast_addr`, `noc_async_write`, `noc_async_write_barrier` | row/column master multicast |
| `sched.matmul_tiles` | `matmul_init`, `matmul_tiles`, `tile_regs_acquire/commit/wait/release`, `pack_tile` | tile outer-product accumulate |
| `sched.rmsnorm` / `sched.gelu` | staged `cb_*` reads plus the documented tile math | fused elementwise stages |
| runtime / compile-time args | `get_arg_val<T>(i)`, `get_compile_time_arg_val<T>(i)` | kernel parameters |

Data-movement intrinsics (`cb_*`, `noc_async_*`, `get_noc_addr`, `get_arg_val`) are
global, as in real tt-metal; host symbols are namespaced under `tt::tt_metal` and
`tt::tt_metal::distributed`.

### Op and pass mapping

| TensixFlow op / pass | Maps toward |
|---|---|
| `tensixflow.matmul`, `tensixflow.fused_rmsnorm_matmul`, `tensixflow.fused_mlp` | the schedule IR, then `matmul_tiles` + `cb_*` + `noc_async_*` kernel intrinsics |
| `tensixflow.rmsnorm`, `tensixflow.gelu` | staged `cb_*` reads plus the fused tile math in `compute.cpp` |
| `--tensixflow-plan-noc-sharding` | the host grid (`CoreRange`), weight strategy, and multicast addressing |
| `--tensixflow-allocate-circular-buffers` | `CreateCircularBuffer` + `CircularBufferConfig` keyed by `tt::CBIndex` |
| `--tensixflow-lower-to-ttkernel` | the tile-level schedule IR (the TTKernel-level layer) |
| `--tensixflow-emit-metalium` | the host program (TTMetal-level) plus the three `kernel_main` bodies (TTKernel-level) |

## Legacy API note

Older tt-metal exposed a per-program enqueue and per-role kernel constructors. These
are **removed** from the current `host_api.hpp`, and TensixFlow deliberately does not
emit them:

- `EnqueueProgram(queue, program)`, the legacy single-program enqueue. The current
  API enqueues a mesh workload through `distributed::EnqueueMeshWorkload` instead.
- `CreateDataMovementKernel(...)` and `CreateComputeKernel(...)`, the per-role
  kernel constructors. The current API uses one `CreateKernel` overload that takes a
  `variant<DataMovementConfig, ComputeConfig>`; the role is selected by the config
  member, not by a separate function.

This is enforced, not just documented. The signature-only stub headers omit
`EnqueueProgram` and the per-role constructors entirely, and the compile-check target
keeps a negative fixture that calls the removed enqueue API and is required to fail
compilation naming `EnqueueProgram`. If the stubs ever drifted back to a legacy
surface, that fixture would start compiling and the check would fail. The emitted
`host.cpp` is additionally gated to contain zero occurrences of the legacy enqueue
token, even in comments.

## Dialect taxonomy

tt-mlir (`@9f06802f`) organizes its lowering into five dialect levels. Two lowering
paths leave TTIR: one through the TT-NN runtime, one direct to metal.

| Dialect | Level | Role |
|---|---|---|
| **TTIR** | high-level tensor IR | device-agnostic tensor ops, e.g. `"ttir.matmul"(...) : (tensor<32x128xbf16>, ...) -> ...`; layout via `#ttcore.metal_layout`, `#ttcore.memory_space<l1>` |
| **TTNN** | runtime dialect | ops mapped onto the TT-NN library; the `StableHLO -> TTIR -> TTNN` path |
| **D2M** | direct-to-metal tile ops | tile-level ops that bypass TT-NN; the `TTIR -> D2M` path |
| **TTKernel** | kernel primitives | the in-kernel CB / NoC / compute primitives (`cb_*`, `noc_async_*`, `matmul_tiles`) |
| **TTMetal** | host dispatch | the host runtime that builds programs, kernels, buffers, and enqueues |

### Where each TensixFlow stage maps

TensixFlow takes the direct-to-metal route: it never stops at a TT-NN-shaped runtime
dialect.

| TensixFlow stage | Nearest tt-mlir level |
|---|---|
| `tensixflow` dialect ops (`matmul`, `rmsnorm`, `gelu`, fused ops) | **TTIR**-level: high-level tensor ops with shape inference and verifiers |
| tile-layout inference + fusion passes | **TTIR**-level transforms (layout annotation + op fusion) |
| `--tensixflow-plan-noc-sharding` + `--tensixflow-allocate-circular-buffers` | the **D2M** bridge: tensor shapes become grid placement, sharding, and circular buffers |
| the schedule IR (`sched.*`) from `--tensixflow-lower-to-ttkernel` | **TTKernel**-level: tile-loop nests over CB / NoC / compute primitives |
| emitted `reader.cpp` / `compute.cpp` / `writer.cpp` | **TTKernel**-level: the three `kernel_main` bodies |
| emitted `host.cpp` | **TTMetal**-level: `CreateProgram` / `CreateKernel` / `CreateCircularBuffer` / `CreateBuffer` and the `distributed::Enqueue*` mesh dispatch |

The correspondence is by level, not by literal dialect: TensixFlow is a standalone
stack with its own dialects, and the tt-mlir names are the reference taxonomy that
names what each stage does.

## Per-op mapping across the five dialect levels

The todo-51 TTIR emitter (`--tensixflow-emit-ttir`) is the only code emitter in
the tt-mlir compat wave (draft D21): the TTIR column below is the emitted
contract, locked by `test/Tensixflow/emit_ttir.mlir`. The TTNN, D2M, TTKernel,
and TTMetal columns are prose mappings: where each op lands if the module takes
that level's path instead of stopping at TTIR text. The grounded syntax behind
each cell (op shapes, layout attrs, RUN lines) is quoted in
[ttmlir_study.md](ttmlir_study.md). One row per tensixflow op, in the order of
`include/tensixflow/TensixflowOps.td`:

| TensixFlow op | TTIR (emitted, todo 51) | TTNN (prose) | D2M (prose) | TTKernel (prose) | TTMetal (prose) |
|---|---|---|---|---|---|
| `tensixflow.func` | `func.func @name(...)` with `#ttcore.metal_layout`-encoded tensor types; argument tensors carry the `dram` encoding | the same func after `--ttir-to-ttnn-backend-pipeline`, tensors tiled by `ttnn.to_layout` | a func over tiled tensors `tensor<8x8x4x4x!ttcore.tile<32x32, bf16>, #layout>` with the grid in `#ttcore.grid` | the three `kernel_main` bodies (reader / compute / writer) | the host program builder: `CreateProgram` plus one `CreateKernel` per role |
| `tensixflow.return` | `return %v : <type>`; a poisoned operand falls back to the source result type rendered `dram` | return of the tiled value after layout conversion | return of the tiled, grid-placed value | the closing `return` of each `kernel_main` body | the `distributed::Enqueue*` mesh dispatch that ships the result buffer |
| `tensixflow.constant` | `"ttir.constant"() <{value = ...}>` with the value attr printed verbatim (plain type, no encoding) | a constant argument (`ttcore.argument_type<constant>`), tiled by `ttnn.to_layout` before first use | a constant buffer placed by `#ttcore.metal_layout` in `l1` or `dram` | compile-time args (`get_compile_time_arg_val`) or CB contents staged by the reader | the device-side write: `CreateBuffer` + `distributed::EnqueueWriteMeshBuffer` |
| `tensixflow.add` | `"ttir.add"(%a, %b) : (T, T) -> T`, result encoded `l1` | `ttnn.add` after `ttnn.to_layout` on the operands (the study's const-eval CHECK pair) | a tile add inside `d2m.generic`: `d2m.tile_add` over `!ttcore.tile<32x32>` operands | `add_binary_tile_init` + `add_binary_tile`, the study's D2MToTTKernel landing | host dispatch unchanged: the add rides inside the registered compute kernel |
| `tensixflow.gelu` | `// UNSUPPORTED: tensixflow.gelu has no TTIR mapping` (deliberate policy, see the note below the table) | the TT-NN elementwise gelu after layout conversion | an SFPU tile op inside `d2m.generic` (the D2MToTTKernel tests are SFPU-placement tests) | the SFPU instruction sequence for gelu staged between `cb_wait_front` and `cb_push_back` | host dispatch unchanged |
| `tensixflow.matmul` | `"ttir.matmul"(%a, %b)` in the bare generic form, shape carried by the function type | `ttnn.matmul` (the study's positive-test expectation) after `ttnn.to_layout` | a tile-level matmul over `!ttcore.tile<32x32, bf16>` operands inside `d2m.generic` on the planned grid | `matmul_init` + `matmul_tiles` with the `tile_regs_acquire/commit/wait/release` + `pack_tile` sequence | the `ComputeConfig{.math_fidelity = HiFi4, .fp32_dest_acc_en = true}` kernel registration |
| `tensixflow.rmsnorm` | `"ttir.rms_norm"(%x, %gamma) <{epsilon, normalized_shape, operandSegmentSizes}>`, the fused canonical op tt-mlir's own `--ttir-fusing` produces from the decomposition | the fused norm carries across the near 1:1 TTIR -> TTNN rename plus `ttnn.to_layout` | the decomposed tile chain (square, reduce, scale, add epsilon, rsqrt, broadcast, multiply) inside `d2m.generic` | the staged `cb_*` reads plus tile math emitted in `compute.cpp` | host dispatch unchanged |
| `tensixflow.fused_rmsnorm_matmul` | decomposed to `ttir.rms_norm` + `ttir.matmul`: the fusion survives as adjacency, not as a single TTIR op | the same two-op chain in TTNN form | the same chain at tile level, sharing the planned grid and circular buffers | the fused reader/compute/writer pipeline the tensixflow scheduler interleaves | one mesh workload instead of two enqueues: the DRAM round-trip the fusion removes |
| `tensixflow.fused_mlp` | the study-grounded prefix (`ttir.rms_norm` + `ttir.matmul` w1 + bias `ttir.add`) plus UNSUPPORTED markers for the gelu stage and its poisoned tail (matmul w2, residual add) | the mapped prefix in TTNN form; the unmapped gelu stage cuts the chain at the same point | the mapped prefix at tile level with the same cut | the fused MLP kernel bodies, gelu stage included at the schedule level | one mesh workload for the whole block |

The gelu row states the emitter's policy honestly. `ttir.gelu` exists upstream
(verified at the pin in tt-mlir's own fusing and model tests), but it sits
outside the study-grounded vocabulary of [ttmlir_study.md](ttmlir_study.md),
and that study is the todo-51 emitter's contract. The emitter therefore marks
gelu UNSUPPORTED rather than emit syntax the contract does not cover, and this
table supplies the prose mapping the emitter withholds. Dependent ops of an
unsupported result are marked transitively (`operand is unsupported`), so the
`fused_mlp` tail and the return mark too.

## Per-pass mapping across the five dialect levels

One row per pass declared in `include/tensixflow/TensixflowPasses.td`, in
pipeline order. The level names the tt-mlir layer each pass does the work of,
whether or not tensixflow routes through a literal tt-mlir dialect:

| TensixFlow pass | Nearest tt-mlir level | What that level does there |
|---|---|---|
| `--tensixflow-infer-tile-layout` | **TTIR** | layout annotation: `#tensixflow.tile_layout` is the tensixflow analog of `#ttcore.metal_layout` riding tensor types; tt-mlir's alignment selection happens on the TTIRToD2M hop (`alignments.mlir`) |
| `--tensixflow-canonicalize` | **TTIR** | canonicalization: identity-add removal, trivial constant folding, structural dedup, the same class as tt-mlir's TTIR canonicalize patterns |
| `--tensixflow-fuse-rmsnorm-matmul` | **TTIR** | op fusion: the analog of tt-mlir's `--ttir-fusing`, which builds `ttir.rms_norm` from the pow/sum/multiply/add/rsqrt/broadcast/multiply chain (`rms_norm_fusion.mlir`) |
| `--tensixflow-plan-noc-sharding` | **D2M** | grid placement + sharding under the L1 budget: `#ttcore.metal_layout<..., sharded>` + `#ttcore.grid` + `#ttcore.shard` |
| `--tensixflow-allocate-circular-buffers` | **D2M** | CB allocation: `d2m.get_cb` / `d2m.wait` / `d2m.reserve` over `!d2m.cb<memref<..., #ttcore.memory_space<l1>>>` |
| `--tensixflow-lower-to-ttkernel` | **TTKernel** | the D2M -> TTKernel analog: tensor ops become tile-loop nests over `cb_*` / `noc_async_*` / compute primitives |
| `--tensixflow-emit-metalium` | **TTKernel** + **TTMetal** | the reader/compute/writer `kernel_main` bodies are the TTKernel-level artifact; `host.cpp` is the TTMetal-level artifact, the D2MToTTMetal analog |
| `--tensixflow-emit-ttir` | **TTIR** | the StableHLOToTTIR analog in reverse: tensixflow ops become TTIR text either downstream path can consume |

## The two lowering paths out of TTIR

tt-mlir's Conversion directories (`include/ttmlir/Conversion/Passes.td` at
`@9f06802f`) define five conversions: StableHLOToTTIR, TTIRToTTNN, TTIRToD2M,
D2MToTTKernel, and D2MToTTMetal. They compose into the two paths that leave
TTIR. Each hop below is evidenced by the RUN line of a real conversion test,
quoted with its source path in [ttmlir_study.md](ttmlir_study.md#lowering-paths-with-the-ttmlir-opt-invocations).

### Path 1: StableHLO -> TTIR -> TTNN (the runtime path)

| Hop | ttmlir-opt invocation | Evidence (RUN line source) |
|---|---|---|
| StableHLO -> TTIR | `ttmlir-opt --stablehlo-to-ttir-pipeline` | `test/ttmlir/Conversion/StableHLOToTTIR/conv2d_op.mlir` |
| TTIR -> TTNN | `ttmlir-opt --ttir-to-ttnn-backend-pipeline` | `test/ttmlir/Conversion/TTIRToTTNN/conv3d_default_config.mlir` |

The TTIR -> TTNN hop is a near 1:1 op rename plus layout management:
`ttnn.to_layout` tiles the tensors, then each `ttir.*` op reappears as its
`ttnn.*` library counterpart (`ttir.add` -> `ttnn.add`, the study's const-eval
CHECK pair). TensixFlow enters this path at TTIR: `--tensixflow-emit-ttir`
emits text shaped for the backend pipeline, even though D21 keeps the TTNN hop
itself prose.

### Path 2: TTIR -> D2M -> TTKernel + TTMetal (the direct-to-metal path)

| Hop | ttmlir-opt invocation | Evidence |
|---|---|---|
| TTIR -> D2M | `ttmlir-opt --ttcore-register-device --ttir-to-d2m` | RUN line: `test/ttmlir/Conversion/TTIRToD2M/embedding_types.mlir` |
| D2M -> TTKernel | `ttmlir-opt --ttcore-register-device --convert-d2m-to-ttkernel` | RUN line: `test/ttmlir/Conversion/D2MToTTKernel/sfpu_op_placement_after_operands.mlir` (`d2m.tile_add` lands on `ttkernel.add_binary_tile_init` + `ttkernel.add_binary_tile`) |
| D2M -> TTMetal | the D2MToTTMetal conversion, the host-side emission of programs, kernels, and buffers | the Conversion directory listing at the pin (`include/ttmlir/Conversion/Passes.td` + `Conversion/D2MToTTMetal/`); no RUN line quoted in the study doc |

TensixFlow walks path 2 with its own dialects instead of tt-mlir's:
`--tensixflow-plan-noc-sharding` + `--tensixflow-allocate-circular-buffers` do
the TTIR -> D2M work (grid, shards, CBs), `--tensixflow-lower-to-ttkernel`
does the D2M -> TTKernel work (tile-loop nests over `cb_*` / `noc_async_*` /
compute primitives), and `--tensixflow-emit-metalium` does the D2M -> TTMetal
work (`host.cpp` with `CreateProgram` / `CreateKernel` / `CreateCircularBuffer`
and the `distributed::Enqueue*` dispatch). The `--tensixflow-emit-ttir` output
is the fork point: the same TTIR text could equally feed path 1's
`--ttir-to-ttnn-backend-pipeline`.
