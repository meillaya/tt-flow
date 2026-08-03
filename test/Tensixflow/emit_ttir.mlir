// RUN: tensixflow-opt --tensixflow-emit-ttir %s | FileCheck %s

// Structural test for --tensixflow-emit-ttir (todo 51, D21): tensixflow ops
// translate to TTIR-shaped MLIR text grounded in docs/ttmlir_study.md
// (tt-mlir @9f06802f). This test ALWAYS runs: it needs only tensixflow-opt +
// FileCheck, never ttmlir-opt (todo 53 adds the gated ttmlir-opt parse). The
// pass is a terminal emission (exit 0 after printing, the EmitMetaliumPass
// convention), so FileCheck sees ONLY the TTIR text, no module dump.
//
// Layout aliases are module-global and interned in first-use order, so the
// #layoutN numbers below are deterministic for this func order.

// CHECK: tensixflow -> TTIR
// CHECK: #ttcore.memory_space<l1>
// CHECK: #layout0 = #ttcore.metal_layout<logical_shape = 1x2048x4096, dim_alignments = 1x32x32, collapsed_intervals = dense<> : tensor<0x2xi64>, dram, interleaved>
// CHECK: #layout1 = #ttcore.metal_layout<logical_shape = 4096, dim_alignments = 32, collapsed_intervals = dense<{{\[\[}}0, 1{{\]\]}}> : tensor<1x2xi64>, dram, interleaved>

// Happy path: the running example (rmsnorm -> matmul). rmsnorm maps to the
// fused canonical "ttir.rms_norm" op (epsilon + normalized_shape attrs), the
// matmul to the generic-form "ttir.matmul", both over layout-encoded tensors.
// CHECK-LABEL: func.func @rmsnorm_matmul
// CHECK-SAME: (%arg0: tensor<1x2048x4096xbf16, #layout0>, %arg1: tensor<4096xbf16, #layout1>, %arg2: tensor<4096x12288xbf16, #layout2>) -> tensor<1x2048x12288xbf16, #layout4>
// CHECK: %0 = "ttir.rms_norm"(%arg0, %arg1) <{epsilon = 9.99999997E-7 : f32, normalized_shape = array<i64: 4096>, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x2048x4096xbf16, #layout0>, tensor<4096xbf16, #layout1>) -> tensor<1x2048x4096xbf16, #layout3>
// CHECK: %1 = "ttir.matmul"(%0, %arg2) : (tensor<1x2048x4096xbf16, #layout3>, tensor<4096x12288xbf16, #layout2>) -> tensor<1x2048x12288xbf16, #layout4>
// CHECK: return %1 : tensor<1x2048x12288xbf16, #layout4>
tensixflow.func @rmsnorm_matmul(%x: tensor<1x2048x4096xbf16>, %gamma: tensor<4096xbf16>, %w: tensor<4096x12288xbf16>) -> tensor<1x2048x12288xbf16> {
  %0 = tensixflow.rmsnorm %x, %gamma : tensor<1x2048x4096xbf16>, tensor<4096xbf16> -> tensor<1x2048x4096xbf16>
  %1 = tensixflow.matmul %0, %w : tensor<1x2048x4096xbf16> x tensor<4096x12288xbf16> -> tensor<1x2048x12288xbf16>
  tensixflow.return %1 : tensor<1x2048x12288xbf16>
}

// Elementwise add: same-shape binary op, shared dram alias for the two
// same-typed inputs, fresh l1 alias for the result.
// CHECK: #layout5 = #ttcore.metal_layout<logical_shape = 2048x4096, dim_alignments = 32x32, collapsed_intervals = dense<{{\[\[}}0, 1], [1, 2{{\]\]}}> : tensor<2x2xi64>, dram, interleaved>
// CHECK: #layout6 = #ttcore.metal_layout<logical_shape = 2048x4096, dim_alignments = 32x32, collapsed_intervals = dense<{{\[\[}}0, 1], [1, 2{{\]\]}}> : tensor<2x2xi64>, l1, interleaved>
// CHECK-LABEL: func.func @just_add
// CHECK-SAME: (%arg0: tensor<2048x4096xbf16, #layout5>, %arg1: tensor<2048x4096xbf16, #layout5>) -> tensor<2048x4096xbf16, #layout6>
// CHECK: %0 = "ttir.add"(%arg0, %arg1) : (tensor<2048x4096xbf16, #layout5>, tensor<2048x4096xbf16, #layout5>) -> tensor<2048x4096xbf16, #layout6>
tensixflow.func @just_add(%a: tensor<2048x4096xbf16>, %b: tensor<2048x4096xbf16>) -> tensor<2048x4096xbf16> {
  %0 = tensixflow.add %a, %b : tensor<2048x4096xbf16>
  tensixflow.return %0 : tensor<2048x4096xbf16>
}

// Fused op composition: fused_rmsnorm_matmul decomposes to rms_norm + matmul,
// reusing the aliases func @rmsnorm_matmul interned (no new alias lines).
// CHECK-LABEL: func.func @fused
// CHECK: %0 = "ttir.rms_norm"
// CHECK: %1 = "ttir.matmul"
// CHECK: return %1 : tensor<1x2048x12288xbf16, #layout4>
tensixflow.func @fused(%x: tensor<1x2048x4096xbf16>, %gamma: tensor<4096xbf16>, %w: tensor<4096x12288xbf16>) -> tensor<1x2048x12288xbf16> {
  %0 = tensixflow.fused_rmsnorm_matmul %x, %gamma, %w : tensor<1x2048x4096xbf16>, tensor<4096xbf16>, tensor<4096x12288xbf16> -> tensor<1x2048x12288xbf16>
  tensixflow.return %0 : tensor<1x2048x12288xbf16>
}

// Failure path (QA scenario): gelu has no study-grounded TTIR mapping, so it
// emits the documented UNSUPPORTED marker (not a crash), and the return whose
// operand is poisoned is marked transitively. The signature falls back to the
// source result type (dram).
// CHECK-LABEL: func.func @unsupported_gelu
// CHECK-SAME: (%arg0: tensor<2048x4096xbf16, #layout5>) -> tensor<2048x4096xbf16, #layout5>
// (The marker colon rides in {{:}} so lit does not read the CHECK line as an
// UNSUPPORTED lit directive.)
// CHECK-NEXT: // UNSUPPORTED{{:}} tensixflow.gelu has no TTIR mapping
// CHECK-NEXT: // UNSUPPORTED{{:}} tensixflow.return has no TTIR mapping (operand is unsupported)
// CHECK-NEXT: }
tensixflow.func @unsupported_gelu(%x: tensor<2048x4096xbf16>) -> tensor<2048x4096xbf16> {
  %0 = tensixflow.gelu %x : tensor<2048x4096xbf16>
  tensixflow.return %0 : tensor<2048x4096xbf16>
}

// D21 guard: TTIR only. TTNN/D2M/TTKernel/TTMetal never appear as code.
// CHECK-NOT: ttnn.
// CHECK-NOT: d2m.
// CHECK-NOT: ttkernel.
