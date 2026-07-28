# TensixFlow to Tenstorrent Mapping

TensixFlow targets the **current** TT-Metalium host and kernel API, studied from
tt-metal source at `@df36cd6`, and sits itself within the tt-mlir dialect family
studied at `@9f06802f`. This page maps each tensixflow op and pass to the TT-Metalium
surface it emits, records the removed legacy API the code generator deliberately
avoids, and places each compilation stage relative to the TTIR / TTNN / D2M /
TTKernel / TTMetal dialect levels.

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
