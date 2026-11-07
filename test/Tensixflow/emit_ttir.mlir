// RUN: tensixflow-opt --tensixflow-emit-ttir %s | FileCheck %s

// Structural test for --tensixflow-emit-ttir. The emitted text follows
// tt-mlir 0.9.0.dev20260221 (pinned in deps/tt-mlir.lock); the real parser gate
// for that text is python/tensixflow/tests/test_ttir_parse.py. This lit test
// needs only tensixflow-opt and FileCheck.
//
// Layout aliases are module-global and interned in first-use order, so the
// #layoutN numbers below are deterministic for this function order. Tensors
// below rank 2 keep a plain type: the pinned #ttcore.metal_layout verifier
// rejects a rank-1 logical_shape, as quoted in
// .deps/ttmlir/fixtures/test/ttmlir/Dialect/TTCore/metal_layout_attr_verify.mlir
// after scripts/fetch_ttmlir_fixtures.sh has run.

// CHECK: tt-mlir 0.9.0.dev20260221
// CHECK: #layout0 = #ttcore.metal_layout<logical_shape = 1x2048x4096, dim_alignments = 1x32x32, collapsed_intervals = dense<{{\[\[}}0, 2], [2, 3{{\]\]}}> : tensor<2x2xi64>, undef, dram, interleaved, index_map = map(0)>
// CHECK: #layout1 = #ttcore.metal_layout<logical_shape = 4096x12288, dim_alignments = 32x32, collapsed_intervals = dense<{{\[\[}}0, 1], [1, 2{{\]\]}}> : tensor<2x2xi64>, undef, dram, interleaved, index_map = map(0)>
// CHECK: #layout2 = #ttcore.metal_layout<logical_shape = 1x2048x4096, dim_alignments = 1x32x32, collapsed_intervals = dense<{{\[\[}}0, 2], [2, 3{{\]\]}}> : tensor<2x2xi64>, undef, l1, interleaved, index_map = map(0)>
// CHECK: #layout3 = #ttcore.metal_layout<logical_shape = 1x2048x12288, dim_alignments = 1x32x32, collapsed_intervals = dense<{{\[\[}}0, 2], [2, 3{{\]\]}}> : tensor<2x2xi64>, undef, l1, interleaved, index_map = map(0)>

// Happy path: the running example (rmsnorm -> matmul). rmsnorm maps to
// ttir.rms_norm in the pinned generic form: two operands (input and weight, no
// bias), so operandSegmentSizes is 1, 1, 0. The rank-1 gamma keeps a plain
// type. matmul maps to the generic ttir.matmul form.
// CHECK-LABEL: func.func @rmsnorm_matmul
// CHECK-SAME: (%arg0: tensor<1x2048x4096xbf16, #layout0>, %arg1: tensor<4096xbf16>, %arg2: tensor<4096x12288xbf16, #layout1>) -> tensor<1x2048x12288xbf16, #layout3>
// CHECK: %0 = "ttir.rms_norm"(%arg0, %arg1) <{epsilon = 9.99999997E-7 : f32, normalized_shape = array<i64: 4096>, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x2048x4096xbf16, #layout0>, tensor<4096xbf16>) -> tensor<1x2048x4096xbf16, #layout2>
// CHECK: %1 = "ttir.matmul"(%0, %arg2) : (tensor<1x2048x4096xbf16, #layout2>, tensor<4096x12288xbf16, #layout1>) -> tensor<1x2048x12288xbf16, #layout3>
// CHECK: return %1 : tensor<1x2048x12288xbf16, #layout3>
tensixflow.func @rmsnorm_matmul(%x: tensor<1x2048x4096xbf16>, %gamma: tensor<4096xbf16>, %w: tensor<4096x12288xbf16>) -> tensor<1x2048x12288xbf16> {
  %0 = tensixflow.rmsnorm %x, %gamma : tensor<1x2048x4096xbf16>, tensor<4096xbf16> -> tensor<1x2048x4096xbf16>
  %1 = tensixflow.matmul %0, %w : tensor<1x2048x4096xbf16> x tensor<4096x12288xbf16> -> tensor<1x2048x12288xbf16>
  tensixflow.return %1 : tensor<1x2048x12288xbf16>
}

// Elementwise add: two same-typed inputs share one dram alias, and the result
// gets a fresh l1 alias.
// CHECK: #layout4 = #ttcore.metal_layout<logical_shape = 2048x4096, dim_alignments = 32x32, collapsed_intervals = dense<{{\[\[}}0, 1], [1, 2{{\]\]}}> : tensor<2x2xi64>, undef, dram, interleaved, index_map = map(0)>
// CHECK: #layout5 = #ttcore.metal_layout<logical_shape = 2048x4096, dim_alignments = 32x32, collapsed_intervals = dense<{{\[\[}}0, 1], [1, 2{{\]\]}}> : tensor<2x2xi64>, undef, l1, interleaved, index_map = map(0)>
// CHECK-LABEL: func.func @just_add
// CHECK-SAME: (%arg0: tensor<2048x4096xbf16, #layout4>, %arg1: tensor<2048x4096xbf16, #layout4>) -> tensor<2048x4096xbf16, #layout5>
// CHECK: %0 = "ttir.add"(%arg0, %arg1) : (tensor<2048x4096xbf16, #layout4>, tensor<2048x4096xbf16, #layout4>) -> tensor<2048x4096xbf16, #layout5>
tensixflow.func @just_add(%a: tensor<2048x4096xbf16>, %b: tensor<2048x4096xbf16>) -> tensor<2048x4096xbf16> {
  %0 = tensixflow.add %a, %b : tensor<2048x4096xbf16>
  tensixflow.return %0 : tensor<2048x4096xbf16>
}

// Fused op composition: fused_rmsnorm_matmul decomposes to rms_norm + matmul,
// reusing the aliases func @rmsnorm_matmul interned (no new alias lines).
// CHECK-LABEL: func.func @fused
// CHECK: %0 = "ttir.rms_norm"
// CHECK: %1 = "ttir.matmul"
// CHECK: return %1 : tensor<1x2048x12288xbf16, #layout3>
tensixflow.func @fused(%x: tensor<1x2048x4096xbf16>, %gamma: tensor<4096xbf16>, %w: tensor<4096x12288xbf16>) -> tensor<1x2048x12288xbf16> {
  %0 = tensixflow.fused_rmsnorm_matmul %x, %gamma, %w : tensor<1x2048x4096xbf16>, tensor<4096xbf16>, tensor<4096x12288xbf16> -> tensor<1x2048x12288xbf16>
  tensixflow.return %0 : tensor<1x2048x12288xbf16>
}

// GELU is a supported op at the pin (fixture
// test/ttmlir/Dialect/TTIR/fusing/gelu_fusing.mlir checks the same generic
// form), so it maps instead of emitting a marker.
// CHECK-LABEL: func.func @gelu
// CHECK: %0 = "ttir.gelu"(%arg0) : (tensor<2048x4096xbf16, #layout4>) -> tensor<2048x4096xbf16, #layout5>
// CHECK: return %0 : tensor<2048x4096xbf16, #layout5>
tensixflow.func @gelu(%x: tensor<2048x4096xbf16>) -> tensor<2048x4096xbf16> {
  %0 = tensixflow.gelu %x : tensor<2048x4096xbf16>
  tensixflow.return %0 : tensor<2048x4096xbf16>
}

// A schedule op has no TTIR mapping (it belongs to the TTKernel path), so it
// emits the documented marker and does not affect the return. The marker colon
// is escaped so lit does not read the CHECK line as an UNSUPPORTED directive.
// CHECK-LABEL: func.func @unsupported_sched_op
// CHECK: // UNSUPPORTED{{:}} tensixflow.sched.gelu has no TTIR mapping
// CHECK: return %arg0 : tensor<2048x4096xbf16, #layout4>
tensixflow.func @unsupported_sched_op(%x: tensor<2048x4096xbf16>) -> tensor<2048x4096xbf16> {
  tensixflow.sched.gelu 0, 0
  tensixflow.return %x : tensor<2048x4096xbf16>
}

// The emitter writes TTIR only; no other dialect appears as code.
// CHECK-NOT: ttnn.
// CHECK-NOT: d2m.
// CHECK-NOT: ttkernel.
