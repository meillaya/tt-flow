# Architecture

TensixFlow compiles one MLIR module in the `tensixflow` dialect into three artifacts: a tile-level schedule IR, a standalone TT-Metalium project, and a plan JSON that carries a projected cost model. The compiler selects a device table entry from `spec/devices.json`; the default is Wormhole n150. The supported emitted-project path covers a single matmul. Fused RMSNorm graphs lower to schedule IR and plan JSON, and the emitter rejects them with a diagnostic.

## Pipeline

Seven passes form the lowering spine. Each pass has one job and runs in this order.

| # | Flag | Job |
|---|---|---|
| 1 | `--tensixflow-infer-tile-layout` | Attach a `#tensixflow.tile_layout`, a 32x32 tile, the device grid, and a DRAM or L1 memory space, to every function argument and op result. Metadata only: no shape changes. |
| 2 | `--tensixflow-canonicalize` | Remove identity adds and fold constants on tensixflow ops. |
| 3 | `--tensixflow-fuse-rmsnorm-matmul` | Rewrite a single-use `rmsnorm -> matmul` pair into `tensixflow.fused_rmsnorm_matmul`, and a full MLP chain into `tensixflow.fused_mlp`. Record the eliminated tensors and bytes on the fused op and on the module. |
| 4 | `--tensixflow-plan-noc-sharding` | Place the graph on the device grid, shard the activation M dimension across the Y cores, pick a weight strategy, estimate NoC traffic, and write the `tensixflow.schedule` module attribute. |
| 5 | `--tensixflow-allocate-circular-buffers` | Assign circular-buffer indices and depths under the device's usable L1 budget, then complete the schedule attribute. |
| 6 | `--tensixflow-lower-to-ttkernel` | Build the `tensixflow.sched.kernel` schedule IR, stamp the matmul shape on the kernel, and erase the source function. |
| 7 | `--tensixflow-emit-metalium` | Write `host.cpp`, `kernels/reader.cpp`, `kernels/compute.cpp`, `kernels/writer.cpp`, `CMakeLists.txt`, and `README.md` under the directory given by `output-dir`, or print the four sources to stdout with file-delimiter banners. |

The device flag `--tensixflow-device=<name>` precedes the passes. It defaults to `n150` and records the choice as a `tensixflow.device` module attribute. An unknown name fails with the list of valid names.

Three passes sit outside the spine. `--tensixflow-emit-ttir` prints TTIR text for a tt-mlir frontend, and `--tensixflow-import-ttir` reads a documented bare-tensor TTIR subset; [ttir.md](ttir.md) documents both. The canonicalize pass changes nothing when the input has nothing to fold.

## The tensixflow dialect

The dialect lives in `include/tensixflow/`. Its ops carry shape inference and verifiers.

| Op | Kind | Notes |
|---|---|---|
| `tensixflow.func` | container | Isolated function with one block, terminated by `tensixflow.return`. |
| `tensixflow.return` | terminator | Returns the function results. |
| `tensixflow.constant` | value | Typed constant, foldable. |
| `tensixflow.add` | elementwise | Same-shape sum. The canonicalizer removes an identity zero operand. |
| `tensixflow.gelu` | elementwise | Exact-erf GELU. |
| `tensixflow.matmul` | compute | `[..., M, K] x [K, N] -> [..., M, N]`. The verifier rejects mismatched contracting dimensions. |
| `tensixflow.rmsnorm` | compute | RMSNorm over the last dimension. The verifier checks that the 1-D gamma matches that dimension. |
| `tensixflow.fused_rmsnorm_matmul` | fused | `matmul(rmsnorm(input, gamma), weight)`. |
| `tensixflow.fused_mlp` | fused | `(gelu(matmul(rmsnorm(x, gamma), w1) + b1) @ w2) + residual`. |

Two attributes carry scheduling decisions. `#tensixflow.tile_layout<tile HxW, grid YxX, mem DRAM|L1>` is the tensor encoding. `#tensixflow.schedule` records the grid, activation sharding, weight strategy, intermediate storage, fusion decision, circular buffers, NoC traffic, and fusion savings. It is pure metadata until the lowering pass reads it.

## The schedule IR

`--tensixflow-lower-to-ttkernel` produces one `tensixflow.sched.kernel` symbol with three regions, one per TT-Metalium kernel role.

| Op | Role |
|---|---|
| `tensixflow.sched.kernel` | Container symbol for the three tile programs. |
| `tensixflow.sched.reader` | Data-movement program that stages input tiles into circular buffers. |
| `tensixflow.sched.compute` | Tile math program that pops inputs, computes, and pushes outputs. |
| `tensixflow.sched.writer` | Data-movement program that drains output tiles. |
| `tensixflow.sched.tile_loop` | Static-trip-count loop. One iteration per tile or tile block. |
| `tensixflow.sched.cb_reserve_back`, `cb_push_back`, `cb_wait_front`, `cb_pop_front` | Circular-buffer handshake, keyed by a CB index. |
| `tensixflow.sched.noc_read`, `noc_write`, `noc_multicast` | Tile movement over the NoC. Multicast carries a fan-out count. |
| `tensixflow.sched.matmul_tiles` | Tile multiply-accumulate over two input CBs into an output CB. |
| `tensixflow.sched.rmsnorm`, `tensixflow.sched.gelu` | Fused prologue and epilogue tile math. |

The lowering stamps a `tensixflow.matmul_shape = [M, N, K]` attribute on the kernel. The emitter reads it to size the DRAM buffers and the CPU reference matmul, and refuses to emit when the attribute is missing.

For the running `examples/rmsnorm_matmul.mlir` example, with M=2048, K=4096, and N=12288, the reader and compute loops stream 128 K-tiles and the writer drains 24,576 output tiles (64 x 384):

```mlir
tensixflow.sched.kernel @rmsnorm_matmul {
  tensixflow.sched.reader {
    tensixflow.sched.tile_loop 128 {
      tensixflow.sched.cb_reserve_back 0, 1
      tensixflow.sched.noc_read 0, 0, 1
      tensixflow.sched.cb_push_back 0, 1
      tensixflow.sched.cb_reserve_back 1, 1
      tensixflow.sched.noc_multicast 1, 0, 1, 8
      tensixflow.sched.cb_push_back 1, 1
    }
  }
  tensixflow.sched.compute {
    tensixflow.sched.tile_loop 128 {
      tensixflow.sched.cb_wait_front 0, 1
      tensixflow.sched.cb_wait_front 1, 1
      tensixflow.sched.rmsnorm 0, 2, 0
      tensixflow.sched.matmul_tiles 0, 1, 16
      tensixflow.sched.cb_pop_front 0, 1
      tensixflow.sched.cb_pop_front 1, 1
    }
  }
  tensixflow.sched.writer {
    tensixflow.sched.tile_loop 24576 {
      tensixflow.sched.cb_wait_front 16, 1
      tensixflow.sched.noc_write 16, 0, 1
      tensixflow.sched.cb_pop_front 16, 1
    }
  }
}
```

The circular buffers use the TT-Metalium convention: inputs at `c_0` and `c_1`, output at `c_16`, and the RMSNorm scale at `c_2`. The simulator executes this IR. The emitter accepts the pure matmul schedule and emits the fused schedule when the activation staging fits L1; an infeasible fused plan exits with a diagnostic that prints the tile arithmetic.

## Data flow

```
examples/*.mlir
  |
  v
tensixflow-opt  (seven spine passes, plus import and emit)
  |-- schedule IR --> tensixflow-simulate --> .tft tensors --> python harness vs NumPy golden
  |-- plan JSON ----> tensixflow-report   --> self-contained HTML report (projected cost)
  `-- project tree --> scripts/build_generated.sh --> scripts/run_generated.sh (ttsim or card)
```

`tensixflow-translate --tensixflow-to-json` serializes a planned module to `plan.json`. The translate step must run before `--tensixflow-lower-to-ttkernel`, because the lowering erases the source function that carries `graph_id` and the tensor list. `tensixflow-report` fills the `perf` block by running the cost model, then renders the HTML.

## Mapping to TT-Metalium

The emitter targets the pinned TT-Metalium release, v0.82.0-rc2. `deps/tt-metal.lock` records the release assets and their sha256 digests. The generated project contains `host.cpp`, `kernels/reader.cpp`, `kernels/compute.cpp`, `kernels/writer.cpp`, `CMakeLists.txt`, and `README.md`. The build compiles and links only the host program; tt-metal's JIT compiles the three kernel sources at run time.

The host program uses the mesh API and realizes the plan grid.

| TensixFlow concept | TT-Metalium API |
|---|---|
| Open the device | `distributed::MeshDevice::create_unit_mesh(device_id)` |
| Command queue | `mesh_device->mesh_command_queue()` |
| Build a program | `CreateProgram(*mesh_device)` |
| Work split over the plan grid | `split_work_to_cores({gridX, gridY}, mTiles * nTiles)`, then contiguous output-tile ranges per core |
| Register a kernel | `CreateKernel(program, path, coreRange, DataMovementConfig or ComputeConfig)` on the grid's `CoreRangeSet` |
| Reader and writer configuration | `DataMovementConfig` with `DataMovementProcessor::RISCV_1` or `RISCV_0`, the matching NoC, and `TensorAccessorArgs` compile-time arguments |
| Compute configuration | `ComputeConfig` with `math_fidelity = MathFidelity::HiFi4` |
| Circular buffers | `CircularBufferConfig(bytes, {{tt::CBIndex, DataFormat::Float16_b}}).set_page_size(index, tile_bytes)`, then `CreateCircularBuffer` on the core range |
| DRAM buffers | `distributed::MeshBuffer::create(ReplicatedBufferConfig, DeviceLocalBufferConfig, mesh_device)` with a tile page size |
| Runtime arguments | `SetRuntimeArgs(program, kernel, core, args)` per core, carrying the tile range or the 2D block bounds |
| Enqueue | `workload.add_program`, then `distributed::EnqueueWriteMeshBuffer`, `EnqueueMeshWorkload`, and `EnqueueReadMeshBuffer` |

Kernel sources include `api/dataflow/dataflow_api.h`, `api/compute/compute_kernel_hw_startup.h`, `api/compute/matmul.h`, `api/compute/tile_move_copy.h`, and `hostdevcommon/kernel_structs.h`. The host includes the release's `<tt-metalium/...>` headers.

| Schedule IR op | Emitted TT-Metalium intrinsic |
|---|---|
| `sched.cb_reserve_back` / `cb_push_back` | `cb_reserve_back`, `cb_push_back` |
| `sched.cb_wait_front` / `cb_pop_front` | `cb_wait_front`, `cb_pop_front` |
| `sched.noc_read` | `TensorAccessor`, `noc_async_read_page` |
| `sched.noc_write` | `TensorAccessor`, `noc_async_write_page` |
| `sched.matmul_tiles` | `compute_kernel_hw_startup<SrcOrder::Reverse>`, `matmul_init`, `tile_regs_acquire`, `matmul_tiles`, `tile_regs_commit`, `tile_regs_wait`, `pack_tile`, `tile_regs_release` |

The emitted program is a multi-core matmul over the plan grid. The planning pass selects one of two mappings. `blocked-reuse` splits the grid into 2D blocks: each core stages its activation row block in `c_0` and its weight column block in `c_1`, and the reader multicasts the weight column block down the used row blocks from the top row block with a semaphore handshake. The emitter writes `reader_sender.cpp` for the sender and `reader_receiver.cpp` for the receivers. `output-tile-ranges` keeps contiguous output-tile ranges and unicast per-output-tile reads, and the fused path always uses it because the fused reader stages activation rows itself. Deterministic inputs come from seed 42. The program computes a CPU reference matmul, runs the kernels under the pinned release, prints the PCC, and prints a `TENSIXFLOW_RESULT` line that carries the status, the PCC, and the threshold of 0.97.

The activation row block is still read once per used column block, so the activation stays replicated across the grid. The plan JSON's `emission` block projects the emitted access pattern for the selected mapping, and the `noc` block projects the strategy the planner would prefer. Weight multicast is emitted; activation multicast is not, so the two blocks stay apart by the activation replication.

The host code uses the release's single `CreateKernel` overload and a config variant, so the kernel role comes from the config type. It does not use an older per-program enqueue or a per-role kernel constructor.

The emitter covers the matmul path and, when the activation staging fits the per-core L1 budget, the fused `fused_rmsnorm_matmul` path. The fused reader stages the core's activation rows, gamma tiles, and a scaler tile in L1, the compute kernel normalizes the staged rows in place, and the matmul consumes the normalized buffer. The activation is read from DRAM once per row tile and the normalized intermediate never reaches DRAM. The plan JSON marks the fused path feasible or infeasible from the tile arithmetic, and the emitter refuses an infeasible plan with a diagnostic that prints the required and available tiles.

The full MLP chain has its own emitter path. It reuses the blocked-reuse grid assignment, then stages and normalizes each activation row, runs the first matmul with the resident w1 block, adds the bias with a destination-reuse add, applies the accurate fp32 GELU, and packs the bf16 GELU output into L1 circular buffers. The second matmul consumes those buffers, adds the residual with a destination-reuse add, and packs the output. Twelve circular buffers hold the activation row, the squares, the inverse RMS, the normalized row, both weight blocks, gamma, the scaler, the bias, the residual row, the intermediate rows, and the output. Every buffer that the compute kernel reads by tile index pushes and pops in whole-depth cycles, because the tile-index read address does not wrap. The first weight block stays resident on every used core and the second is staged per column block, so MLP weight traffic is not multicast. The plan JSON records the intermediate row count as `fused_stage_tiles`, and an infeasible chain is rejected with the intermediate and total tile counts.

The generated `CMakeLists.txt` finds the release package with `find_package(TT-Metalium REQUIRED)` and links the `TT::Metalium` target. `scripts/fetch_tt_metal.sh` unpacks the release artifacts and writes `.deps/env.sh` with the package, library, runtime, and simulator variables.

## Tools

| Tool | Input | Output |
|---|---|---|
| `tensixflow-opt` | `tensixflow` MLIR or a bare-tensor TTIR subset, plus pass flags | transformed IR, schedule IR, emitted project, or stdout kernels |
| `tensixflow-translate` | a module carrying a `tensixflow.schedule` attribute | plan JSON (`--tensixflow-to-json`) |
| `tensixflow-simulate` | an example module and `.tft` input tensors | `.tft` output tensors from the schedule IR |
| `tensixflow-report` | plan JSON | self-contained offline HTML report |
| `scripts/normalize_ttir.sh` | layout-encoded TTIR | bare-tensor TTIR that `--tensixflow-import-ttir` accepts |

The tools build into `build/bin/`. [device.md](device.md) covers the run path on a simulator or a card, [verification.md](verification.md) covers the checks, and [performance.md](performance.md) covers the cost model.
