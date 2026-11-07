// RUN: tensixflow-opt --tensixflow-emit-ttir %s | FileCheck %s

// Input module for the tt-mlir parser gate. Every function here maps to a TTIR
// op and ends with a return, so the emitted text must parse with the released
// tt-mlir bindings. python/tensixflow/tests/test_ttir_parse.py runs
// tensixflow-opt on this file and parses the output under python3.11 with the
// wheel pinned in deps/tt-mlir.lock; that test is the real syntax gate.
//
// This lit test keeps the structural expectations for the same text. It needs
// only tensixflow-opt and FileCheck: the release ships Python bindings and no
// ttmlir-opt executable, so a ttmlir-opt round trip is not available.
//
// Layout aliases are module-global and interned in first-use order. Tensors
// below rank 2 keep a plain type because the pinned #ttcore.metal_layout
// verifier rejects a rank-1 logical_shape.

// CHECK: #layout0 = #ttcore.metal_layout<logical_shape = 1x2048x4096, dim_alignments = 1x32x32, collapsed_intervals = dense<{{\[\[}}0, 2], [2, 3{{\]\]}}> : tensor<2x2xi64>, undef, dram, interleaved, index_map = map(0)>
// CHECK: #layout1 = #ttcore.metal_layout<logical_shape = 4096x12288, dim_alignments = 32x32, collapsed_intervals = dense<{{\[\[}}0, 1], [1, 2{{\]\]}}> : tensor<2x2xi64>, undef, dram, interleaved, index_map = map(0)>
// CHECK: #layout2 = #ttcore.metal_layout<logical_shape = 1x2048x4096, dim_alignments = 1x32x32, collapsed_intervals = dense<{{\[\[}}0, 2], [2, 3{{\]\]}}> : tensor<2x2xi64>, undef, l1, interleaved, index_map = map(0)>
// CHECK: #layout3 = #ttcore.metal_layout<logical_shape = 1x2048x12288, dim_alignments = 1x32x32, collapsed_intervals = dense<{{\[\[}}0, 2], [2, 3{{\]\]}}> : tensor<2x2xi64>, undef, l1, interleaved, index_map = map(0)>

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

// CHECK: #layout4 = #ttcore.metal_layout<logical_shape = 2048x4096, dim_alignments = 32x32, collapsed_intervals = dense<{{\[\[}}0, 1], [1, 2{{\]\]}}> : tensor<2x2xi64>, undef, dram, interleaved, index_map = map(0)>
// CHECK: #layout5 = #ttcore.metal_layout<logical_shape = 2048x4096, dim_alignments = 32x32, collapsed_intervals = dense<{{\[\[}}0, 1], [1, 2{{\]\]}}> : tensor<2x2xi64>, undef, l1, interleaved, index_map = map(0)>
// CHECK-LABEL: func.func @just_add
// CHECK-SAME: (%arg0: tensor<2048x4096xbf16, #layout4>, %arg1: tensor<2048x4096xbf16, #layout4>) -> tensor<2048x4096xbf16, #layout5>
// CHECK: %0 = "ttir.add"(%arg0, %arg1) : (tensor<2048x4096xbf16, #layout4>, tensor<2048x4096xbf16, #layout4>) -> tensor<2048x4096xbf16, #layout5>
tensixflow.func @just_add(%a: tensor<2048x4096xbf16>, %b: tensor<2048x4096xbf16>) -> tensor<2048x4096xbf16> {
  %0 = tensixflow.add %a, %b : tensor<2048x4096xbf16>
  tensixflow.return %0 : tensor<2048x4096xbf16>
}

// CHECK-LABEL: func.func @gelu
// CHECK: %0 = "ttir.gelu"(%arg0) : (tensor<2048x4096xbf16, #layout4>) -> tensor<2048x4096xbf16, #layout5>
// CHECK: return %0 : tensor<2048x4096xbf16, #layout5>
tensixflow.func @gelu(%x: tensor<2048x4096xbf16>) -> tensor<2048x4096xbf16> {
  %0 = tensixflow.gelu %x : tensor<2048x4096xbf16>
  tensixflow.return %0 : tensor<2048x4096xbf16>
}

// Fused rmsnorm + matmul reuses the aliases interned above.
// CHECK-LABEL: func.func @fused
// CHECK: %0 = "ttir.rms_norm"
// CHECK: %1 = "ttir.matmul"
// CHECK: return %1 : tensor<1x2048x12288xbf16, #layout3>
tensixflow.func @fused(%x: tensor<1x2048x4096xbf16>, %gamma: tensor<4096xbf16>, %w: tensor<4096x12288xbf16>) -> tensor<1x2048x12288xbf16> {
  %0 = tensixflow.fused_rmsnorm_matmul %x, %gamma, %w : tensor<1x2048x4096xbf16>, tensor<4096xbf16>, tensor<4096x12288xbf16> -> tensor<1x2048x12288xbf16>
  tensixflow.return %0 : tensor<1x2048x12288xbf16>
}

// A constant keeps a plain result type: ttir.constant requires the value
// attribute type and the result type to match, so no layout alias attaches.
// CHECK-LABEL: func.func @constant_add
// CHECK: %0 = "ttir.constant"() <{value = dense<1.000000e+00> : tensor<2048x4096xbf16>}> : () -> tensor<2048x4096xbf16>
// CHECK: %1 = "ttir.add"(%arg0, %0) : (tensor<2048x4096xbf16, #layout4>, tensor<2048x4096xbf16>) -> tensor<2048x4096xbf16, #layout5>
// CHECK: return %1 : tensor<2048x4096xbf16, #layout5>
tensixflow.func @constant_add(%x: tensor<2048x4096xbf16>) -> tensor<2048x4096xbf16> {
  %c = tensixflow.constant dense<1.0> : tensor<2048x4096xbf16>
  %0 = tensixflow.add %x, %c : tensor<2048x4096xbf16>
  tensixflow.return %0 : tensor<2048x4096xbf16>
}
