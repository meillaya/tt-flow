// RUN: split-file %s %t
// RUN: not tensixflow-opt --allow-unregistered-dialect --tensixflow-import-ttir %t/unsupported_ttir_op.mlir 2>&1 | FileCheck %t/unsupported_ttir_op.mlir
// RUN: not tensixflow-opt --allow-unregistered-dialect --tensixflow-import-ttir %t/unsupported_ttnn_op.mlir 2>&1 | FileCheck %t/unsupported_ttnn_op.mlir
// RUN: not tensixflow-opt --allow-unregistered-dialect --tensixflow-import-ttir %t/layout_encoded_type.mlir 2>&1 | FileCheck %t/layout_encoded_type.mlir
// RUN: not tensixflow-opt --allow-unregistered-dialect --tensixflow-import-ttir %t/module_device_attr.mlir 2>&1 | FileCheck %t/module_device_attr.mlir
// RUN: not tensixflow-opt --allow-unregistered-dialect --tensixflow-import-ttir %t/dynamic_shape.mlir 2>&1 | FileCheck %t/dynamic_shape.mlir
// RUN: not tensixflow-opt --allow-unregistered-dialect --tensixflow-import-ttir %t/bad_element_type.mlir 2>&1 | FileCheck %t/bad_element_type.mlir
// RUN: not tensixflow-opt --allow-unregistered-dialect --tensixflow-import-ttir %t/transpose_a.mlir 2>&1 | FileCheck %t/transpose_a.mlir
// RUN: not tensixflow-opt --allow-unregistered-dialect --tensixflow-import-ttir %t/transpose_b.mlir 2>&1 | FileCheck %t/transpose_b.mlir
// RUN: not tensixflow-opt --allow-unregistered-dialect --tensixflow-import-ttir %t/rank_one_arg.mlir 2>&1 | FileCheck %t/rank_one_arg.mlir
// RUN: not tensixflow-opt --allow-unregistered-dialect --tensixflow-import-ttir %t/multiple_functions.mlir 2>&1 | FileCheck %t/multiple_functions.mlir
// RUN: not tensixflow-opt --allow-unregistered-dialect --tensixflow-import-ttir %t/rms_norm_bias.mlir 2>&1 | FileCheck %t/rms_norm_bias.mlir
// RUN: not tensixflow-opt --allow-unregistered-dialect --tensixflow-import-ttir %t/normalized_shape.mlir 2>&1 | FileCheck %t/normalized_shape.mlir

// Negative cases for --tensixflow-import-ttir. Each case rejects one construct
// the importer does not accept, and every diagnostic names the construct and
// points at docs/ttir.md. The source is bare-tensor TTIR parsed with
// --allow-unregistered-dialect.

//--- unsupported_ttir_op.mlir
// CHECK: error: operation 'ttir.constant' is outside the import subset; the importer accepts ttir.matmul, ttir.add, ttir.gelu, and ttir.rms_norm (docs/ttir.md)
func.func @constant(%arg0: tensor<64x64xbf16>) -> tensor<64x64xbf16> {
  %0 = "ttir.constant"() <{value = dense<1.000000e+00> : tensor<64x64xbf16>}> : () -> tensor<64x64xbf16>
  return %0 : tensor<64x64xbf16>
}

//--- unsupported_ttnn_op.mlir
// CHECK: error: operation 'ttnn.relu' is outside the import subset; the importer accepts ttir.matmul, ttir.add, ttir.gelu, and ttir.rms_norm (docs/ttir.md)
func.func @relu(%arg0: tensor<64x64xbf16>) -> tensor<64x64xbf16> {
  %0 = "ttnn.relu"(%arg0) : (tensor<64x64xbf16>) -> tensor<64x64xbf16>
  return %0 : tensor<64x64xbf16>
}

//--- layout_encoded_type.mlir
// CHECK: error: function argument 0 is layout-encoded: 'tensor<32x32xbf16, #ttcore.metal_layout
// CHECK-SAME: this build does not link the ttcore or ttnn dialects
#layout = #ttcore.metal_layout<logical_shape = 32x32, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, interleaved, index_map = map(0)>
func.func @encoded(%arg0: tensor<32x32xbf16, #layout>) -> tensor<32x32xbf16, #layout> {
  return %arg0 : tensor<32x32xbf16, #layout>
}

//--- module_device_attr.mlir
// CHECK: error: module attribute 'ttcore.meshes' is outside the import subset; device and layout attributes from the ttcore or ttnn dialects are rejected because this build does not link them (docs/ttir.md)
module attributes {ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x1>]>} {
  func.func @f(%arg0: tensor<64x64xbf16>) -> tensor<64x64xbf16> {
    return %arg0 : tensor<64x64xbf16>
  }
}

//--- dynamic_shape.mlir
// CHECK: error: function argument 0 has a dynamic shape: 'tensor<?x64xbf16>'; the import subset requires static shapes (docs/ttir.md)
func.func @dynamic(%arg0: tensor<?x64xbf16>) -> tensor<?x64xbf16> {
  return %arg0 : tensor<?x64xbf16>
}

//--- bad_element_type.mlir
// CHECK: error: function argument 0 has element type 'f16'; the import subset accepts bf16 and f32 only (docs/ttir.md)
func.func @half(%arg0: tensor<64x64xf16>) -> tensor<64x64xf16> {
  return %arg0 : tensor<64x64xf16>
}

//--- transpose_a.mlir
// CHECK: error: 'ttir.matmul' attribute 'transpose_a = true' is outside the import subset; only untransposed operands are imported (docs/ttir.md)
func.func @transpose_a(%arg0: tensor<64x128xbf16>, %arg1: tensor<128x32xbf16>) -> tensor<64x32xbf16> {
  %0 = "ttir.matmul"(%arg0, %arg1) <{transpose_a = true}> : (tensor<64x128xbf16>, tensor<128x32xbf16>) -> tensor<64x32xbf16>
  return %0 : tensor<64x32xbf16>
}

//--- transpose_b.mlir
// CHECK: error: 'ttir.matmul' attribute 'transpose_b = true' is outside the import subset; only untransposed operands are imported (docs/ttir.md)
func.func @transpose_b(%arg0: tensor<64x128xbf16>, %arg1: tensor<32x128xbf16>) -> tensor<64x32xbf16> {
  %0 = "ttir.matmul"(%arg0, %arg1) <{transpose_b = true}> : (tensor<64x128xbf16>, tensor<32x128xbf16>) -> tensor<64x32xbf16>
  return %0 : tensor<64x32xbf16>
}

//--- rank_one_arg.mlir
// CHECK: error: function argument 0 has rank 1: 'tensor<64xbf16>'; rank-1 tensors are outside the import subset except the ttir.rms_norm weight (docs/ttir.md)
func.func @rank_one(%arg0: tensor<64xbf16>, %arg1: tensor<64x64xbf16>) -> tensor<64x64xbf16> {
  %0 = "ttir.gelu"(%arg1) : (tensor<64x64xbf16>) -> tensor<64x64xbf16>
  return %0 : tensor<64x64xbf16>
}

//--- multiple_functions.mlir
// CHECK: error: expected exactly one func.func in the module, found 2; multiple functions are outside the import subset (docs/ttir.md)
func.func @first(%arg0: tensor<64x64xbf16>) -> tensor<64x64xbf16> {
  return %arg0 : tensor<64x64xbf16>
}

func.func @second(%arg0: tensor<64x64xbf16>) -> tensor<64x64xbf16> {
  return %arg0 : tensor<64x64xbf16>
}

//--- rms_norm_bias.mlir
// CHECK: error: 'ttir.rms_norm' with a bias operand is outside the import subset; only the input and weight operands are imported (docs/ttir.md)
func.func @bias(%arg0: tensor<1x4x8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>) -> tensor<1x4x8xf32> {
  %0 = "ttir.rms_norm"(%arg0, %arg1, %arg2) <{normalized_shape = array<i64: 8>, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 1, 1>}> : (tensor<1x4x8xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<1x4x8xf32>
  return %0 : tensor<1x4x8xf32>
}

//--- normalized_shape.mlir
// CHECK: error: 'ttir.rms_norm' normalized_shape array<i64: 4, 8> is outside the import subset; tensixflow normalizes the last dimension (8) only (docs/ttir.md)
func.func @shape(%arg0: tensor<1x4x8xf32>, %arg1: tensor<8xf32>) -> tensor<1x4x8xf32> {
  %0 = "ttir.rms_norm"(%arg0, %arg1) <{normalized_shape = array<i64: 4, 8>, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x4x8xf32>, tensor<8xf32>) -> tensor<1x4x8xf32>
  return %0 : tensor<1x4x8xf32>
}
