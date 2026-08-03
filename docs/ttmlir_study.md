# tt-mlir Study: TTIR/TTNN Op Shapes (Source-Study Fallback)

Status: **the heavy tt-mlir build was SKIPPED; the source-study fallback was
taken.** This page carries the skip reason, the build script that would run
the build, and the real TTIR/TTNN op shapes gathered from tt-mlir's own test
files. It grounds the TTIR emitter (todo 51) and the gated ttmlir-opt parse
(todo 53) without a device and without the 30-60 minute toolchain build.

tt-mlir is studied at `@9f06802f` (full SHA
`9f06802f5494f41ccdce4d3f725462f7ce13cdc1`, 2026-07-22), the same pin the
[tenstorrent_mapping.md](tenstorrent_mapping.md) taxonomy uses. All snippets
below are quoted verbatim from test files in `tenstorrent/tt-mlir` at that
era, via GitHub code search; the source path is cited above each block.

## Why the build was skipped

Todo 50 has a disjunctive acceptance: build+run `ttmlir-opt`, OR document the
skip reason and use source-study snippets from tt-mlir tests. The fallback is
taken for three reasons:

1. **The build is heavy.** tt-mlir does not use the system LLVM. Its
   `env/CMakeLists.txt` builds a private toolchain first: llvm-project
   `@4efe170d` (MinSizeRel, projects `mlir,lld`), flatbuffers, a Python venv,
   plus stablehlo/shardy source checkouts. That is the 30-60 minute cost,
   before the project itself compiles.
2. **The runtime needs a device we do not have.** `TTMLIR_ENABLE_RUNTIME`
   pulls in tt-metal and the TT-NN/TTMetal runtimes, which require Tenstorrent
   hardware and its system dependencies. The offline compiler is the only
   supported no-device configuration (the getting-started build matrix lists
   "Offline Compiler Only" as the column that works everywhere, including
   Ubuntu 24.04).
3. **It is off the MVP critical path.** Draft D10 resolves this as an
   OPTIONAL study task and the enabler for the later TT-MLIR compat wave.
   `check-tensixflow` and `check-generated-kernels-compile` never invoke
   `scripts/build-tt-mlir.sh`; nothing in the CMake build references it.

## The build script (correct, not run)

`scripts/build-tt-mlir.sh` is the script that WOULD run the study build. It is
standalone, idempotent, `bash -n`-valid, and mirrors the upstream flow exactly
as documented in tt-mlir `docs/src/getting-started.md` and
`.github/Dockerfile.ci` (the `environment_build` anchor) at the pinned commit:

| Step | Command | Source of truth |
|---|---|---|
| Clone + pin | `git clone https://github.com/tenstorrent/tt-mlir.git` + `git checkout 9f06802f...` | this study's pin |
| Toolchain dir | `sudo mkdir -p /opt/ttmlir-toolchain && sudo chown -R $USER ...` | getting-started.md, env/CMakeLists.txt FATAL_ERROR text |
| Env toolchain | `cmake -B env/build env -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++` then `cmake --build env/build` | Dockerfile.ci `environment_build` anchor |
| Activate | `source env/activate` (sets `TTMLIR_ENV_ACTIVATED`, `TTMLIR_TOOLCHAIN_DIR`, PATH) | top CMakeLists.txt env gate |
| Configure | `cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release -DTTMLIR_ENABLE_RUNTIME=OFF -DTTMLIR_ENABLE_STABLEHLO=ON -DTTMLIR_ENABLE_BINDINGS_PYTHON=OFF -DCMAKE_BUILD_PARALLEL_LEVEL=4` | RUNTIME=OFF is the upstream `option()` default; STABLEHLO=ON enables the StableHLO path; parallel level 4 is the documented OOM workaround |
| Build | `cmake --build build --target ttmlir-opt` | offline compiler only |
| Sample | `--stablehlo-to-ttir-pipeline`, `--ttir-to-ttnn-backend-pipeline`, `--ttcore-register-device --ttir-to-d2m` over a matmul+add graph | RUN lines below |

If the build is ever run, the captured TTIR/TTNN/D2M forms land under
`$TT_MLIR_DIR/study/` and can be pasted into the sections below, upgrading
them from source-study snippets to live captures.

## TTIR op shapes from tt-mlir tests

TTIR is the high-level tensor dialect: generic op names, ranked tensor types
carrying the shape, optional attributes in the `<{...}>` slot. Every snippet
below is quoted from a real tt-mlir test file.

### ttir.matmul

Source: `test/ttmlir/Dialect/TTIR/fusing/permute_slice_after_matmul.mlir`

```mlir
%0 = "ttir.matmul"(%arg0, %arg1) : (tensor<1024x4096xbf16>, tensor<4096x128256xbf16>) -> tensor<1024x128256xbf16>
```

The canonicalized form carries transpose attributes, from the CHECK lines of
the same file:

```mlir
"ttir.matmul"(%arg0, %arg1) <{transpose_a = false, transpose_b = false}>
    : (tensor<1024x4096xbf16>, tensor<4096x128256xbf16>) -> tensor<1024x128256xbf16>
```

Rank variants (1D/2D operands) and the TTIR-to-TTNN expectation, source:
`test/ttmlir/Dialect/TTNN/matmul/matmul_tests_positive.mlir`

```mlir
func.func @matmul_2d_2d(%arg0: tensor<64x128xbf16>, %arg1: tensor<128x64xbf16>) -> tensor<64x64xbf16> {
  // CHECK: "ttnn.matmul"
  %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<64x128xbf16>, tensor<128x64xbf16>) -> tensor<64x64xbf16>
  return %0 : tensor<64x64xbf16>
}
```

Batched matmul with a transposed B operand, source:
`test/ttmlir/Dialect/TTNN/optimizer/ttnn_fusing/split_qkv/split_qkv_matmul.mlir`

```mlir
%q0 = "ttir.matmul"(%0, %wq) <{transpose_a = false, transpose_b = true}>
    : (tensor<32x512xbf16>, tensor<512x512xbf16>) -> tensor<32x512xbf16>
```

Verifier diagnostics name the op, source:
`test/ttmlir/Dialect/TTIR/matmul/matmul_tests_negative.mlir`

```mlir
// CHECK: error: 'ttir.matmul' op Input A must be at least a 1D tensor
%1 = "ttir.matmul"(%arg0, %arg1) : (tensor<bf16>, tensor<64xbf16>) -> tensor<1xbf16>
```

### ttir.add

Source: `test/ttmlir/Dialect/TTIR/const-eval/const-eval.mlir`

```mlir
%1 = "ttir.add"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
```

The same file annotates function arguments with `ttcore.argument_type`,
the input/parameter/constant classification the runtime uses:

```mlir
func.func @forward_split(
    %arg0: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<input>},
    %arg1: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>},
    %arg3: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}) -> tensor<32x32xbf16>
```

Broadcasting adds (per-row bias over a 4D tensor), source:
`test/ttmlir/models/segformer.mlir`

```mlir
%31 = "ttir.add"(%30, %arg104) : (tensor<1x16384x32xbf16>, tensor<32xbf16>) -> tensor<1x16384x32xbf16>
```

### ttir.multiply

Elementwise multiply with rank-1 broadcasting, source:
`test/ttmlir/models/segformer.mlir`

```mlir
%29 = "ttir.multiply"(%22, %28) : (tensor<1x16384x32xbf16>, tensor<1x16384x1xbf16>) -> tensor<1x16384x32xbf16>
%30 = "ttir.multiply"(%29, %arg103) : (tensor<1x16384x32xbf16>, tensor<32xbf16>) -> tensor<1x16384x32xbf16>
```

Same-shape multiply, source:
`test/ttmlir/Dialect/TTIR/Transforms/HoistCPUOps/const_eval_hoist_multichip.mlir`

```mlir
%mul = "ttir.multiply"(%add, %arg0) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
```

### ttir.rsqrt and the RMSNorm decomposition

The RMSNorm fusion test shows the exact TTIR chain tt-mlir decomposes RMSNorm
into: square (`pow`), reduce (`sum`), scale, add epsilon, `rsqrt`, broadcast,
multiply. Source: `test/ttmlir/Dialect/TTIR/fusing/rms_norm_fusion.mlir`

```mlir
%28 = "ttir.pow"(%27, %5) : (tensor<32x1x2048xf32>, tensor<32x1x2048xf32>) -> tensor<32x1x2048xf32>
%29 = "ttir.sum"(%28) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<32x1x2048xf32>) -> tensor<32x1xf32>
%30 = "ttir.multiply"(%29, %4) : (tensor<32x1xf32>, tensor<32x1xf32>) -> tensor<32x1xf32>
%31 = "ttir.reshape"(%30) <{shape = [32 : i32, 1 : i32, 1 : i32]}> : (tensor<32x1xf32>) -> tensor<32x1x1xf32>
%32 = "ttir.add"(%31, %3) : (tensor<32x1x1xf32>, tensor<32x1x1xf32>) -> tensor<32x1x1xf32>
%33 = "ttir.rsqrt"(%32) : (tensor<32x1x1xf32>) -> tensor<32x1x1xf32>
%36 = "ttir.broadcast"(%35) <{broadcast_dimensions = array<i64: 1, 1, 2048>}> : (tensor<32x1x1xf32>) -> tensor<32x1x2048xf32>
%37 = "ttir.multiply"(%27, %36) : (tensor<32x1x2048xf32>, tensor<32x1x2048xf32>) -> tensor<32x1x2048xf32>
```

This is the same math TensixFlow fuses into `tensixflow.rmsnorm` and emits as
staged tile ops in `compute.cpp`, so the chain above is the direct reference
vocabulary for the todo-51 TTIR emitter. Constant-fold expectations for
`ttir.rsqrt` live in
`test/ttmlir/Dialect/TTIR/canonicalize/eltwise_unary_op_fold_tests.mlir`:

```mlir
%0 = "ttir.constant"() <{value = dense<[1.0, 4.0, 16.0, 0.25]> : tensor<4xf32>}> : () -> tensor<4xf32>
%1 = "ttir.rsqrt"(%0) : (tensor<4xf32>) -> tensor<4xf32>
```

## Layout and memory-space attributes

### #ttcore.metal_layout

The D2M path encodes placement with `#ttcore.metal_layout`: logical shape,
per-dim alignments, a collapsed-intervals map, the memory space (`l1`/`dram`),
and `sharded`/`interleaved`. Source:
`test/ttmlir/Dialect/D2M/lower_to_layout.mlir`

```mlir
#layout = #ttcore.metal_layout<logical_shape = 1024x1024, dim_alignments = 256x256,
    collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#layout_dram = #ttcore.metal_layout<logical_shape = 1024x1024, dim_alignments = 256x256,
    collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, dram, sharded>

func.func @tilize(%arg0: tensor<1024x1024xf32>) -> tensor<8x8x4x4x!ttcore.tile<32x32, f32>, #layout> {
  %0 = d2m.empty() : tensor<8x8x4x4x!ttcore.tile<32x32, f32>, #layout>
```

Note the tiled tensor form: `tensor<8x8x4x4x!ttcore.tile<32x32, f32>, #layout>`.
The outer `8x8` is the core grid, the inner `4x4` is the tile count per core,
and `!ttcore.tile<32x32, f32>` is the 32x32 tile type, exactly the geometry
TensixFlow's scheduler plans. Verifier negatives for the attribute (alignment
rank and positivity) are in
`test/ttmlir/Dialect/TTCore/metal_layout_attr_verify.mlir`, and the alignment
selection expectations (tile-sized alignments for small layouts, scaled
alignments when a dimension is split across cores) are in
`test/ttmlir/Conversion/TTIRToD2M/alignments.mlir`.

### #ttcore.memory_space<l1>

At the memref level (post-bufferization) the memory space becomes
`#ttcore.memory_space<l1>` (also `<dram>` and `<dst>` for the math
destination register), combined with `#ttcore.shard`. Source:
`test/ttmlir/Dialect/D2M/Transforms/insert_dst_register_access_multiple_nests.mlir`

```mlir
%alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<8x12x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>
d2m.generic {grid = #ttcore.grid<4x4>, ...}
    ins(%alloc_1 : memref<4x4x8x12x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #ttcore.memory_space<l1>>)
    outs(%alloc_5 : memref<4x4x8x12x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #ttcore.memory_space<l1>>)
```

Circular buffers appear as `d2m.get_cb` / `d2m.wait` / `d2m.reserve` over
`!d2m.cb<memref<..., #ttcore.memory_space<l1>>>` in the same file, the D2M
spelling of the `cb_*` primitives TensixFlow emits directly in its kernels.

## Lowering paths (with the ttmlir-opt invocations)

The two paths that leave TTIR, plus the kernel-level terminus, each evidenced
by the RUN line of a real conversion test:

| Path | ttmlir-opt invocation | Evidence (RUN line source) |
|---|---|---|
| StableHLO -> TTIR | `ttmlir-opt --stablehlo-to-ttir-pipeline` | `test/ttmlir/Conversion/StableHLOToTTIR/conv2d_op.mlir` |
| TTIR -> TTNN | `ttmlir-opt --ttir-to-ttnn-backend-pipeline` | `test/ttmlir/Conversion/TTIRToTTNN/conv3d_default_config.mlir` |
| TTIR -> D2M | `ttmlir-opt --ttcore-register-device --ttir-to-d2m` | `test/ttmlir/Conversion/TTIRToD2M/embedding_types.mlir` |
| D2M -> TTKernel | `ttmlir-opt --ttcore-register-device --convert-d2m-to-ttkernel` | `test/ttmlir/Conversion/D2MToTTKernel/sfpu_op_placement_after_operands.mlir` |

The TTIR -> TTNN hop is a near 1:1 op rename plus layout management. From
`test/ttmlir/Dialect/TTNN/const-eval/const-eval.mlir`:

```mlir
// CHECK: %[[TILED_INPUT1:.*]] = "ttnn.to_layout"(%arg0)
// CHECK: "ttnn.add"(%[[TILED_INPUT1]], %arg1)
%0 = "ttir.add"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
```

The D2M -> TTKernel hop lands on the kernel primitives TensixFlow emits by
hand. From `test/ttmlir/Conversion/D2MToTTKernel/sfpu_op_placement_after_operands.mlir`:

```mlir
// CHECK: ttkernel.add_binary_tile_init
// CHECK-NEXT: ttkernel.add_binary_tile
%4 = "d2m.tile_add"(%2, %3) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
```

## What TensixFlow takes from this

The dialect taxonomy and the stage-by-stage mapping live in
[tenstorrent_mapping.md](tenstorrent_mapping.md#dialect-taxonomy); this page
supplies the concrete syntax behind it. For the todo-51 TTIR emitter, the
grounded facts are:

- Op syntax is the generic MLIR form with the shape in the function type:
  `"ttir.matmul"(%a, %b) : (tensor<MxKxbf16>, tensor<KxNxbf16>) -> tensor<MxNxbf16>`,
  attributes in `<{...}>` (`transpose_a`, `transpose_b`, `dim_arg`,
  `broadcast_dimensions`, `shape`).
- Elementwise ops (`ttir.add`, `ttir.multiply`, `ttir.rsqrt`) are unary/binary
  tensor ops with implicit broadcasting; the RMSNorm chain above is the exact
  decomposition the emitter can target.
- Placement rides in attributes, not in the ops: `#ttcore.metal_layout` at the
  tensor level, `#ttcore.memory_space<l1>` + `#ttcore.shard` at the memref
  level, `!ttcore.tile<32x32, bf16>` as the tile type.
- TensixFlow takes the direct-to-metal route (TTIR-level ops straight to the
  D2M/TTKernel-level schedule), so the TTNN column stays a prose mapping per
  draft D21, while the TTIR syntax above is the real emitter contract.
