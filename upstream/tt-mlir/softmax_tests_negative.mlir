// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for the TTIR softmax operation.

// CHECK: error: 'ttir.softmax' op Input and output shapes must be the same
func.func @softmax_shape_mismatch(%arg0: tensor<2x4x8xf32>) -> tensor<2x4x16xf32> {
  %0 = "ttir.softmax"(%arg0) <{dimension = 1 : si32}> : (tensor<2x4x8xf32>) -> tensor<2x4x16xf32>
  return %0 : tensor<2x4x16xf32>
}

// -----
// CHECK: error: 'ttir.softmax' op Dimension attribute must be within the bounds of the input tensor
func.func @softmax_dimension_too_large(%arg0: tensor<2x4x8xf32>) -> tensor<2x4x8xf32> {
  %0 = "ttir.softmax"(%arg0) <{dimension = 3 : si32}> : (tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
  return %0 : tensor<2x4x8xf32>
}

// -----
// CHECK: error: 'ttir.softmax' op Dimension attribute must be within the bounds of the input tensor
func.func @softmax_dimension_too_negative(%arg0: tensor<2x4x8xf32>) -> tensor<2x4x8xf32> {
  %0 = "ttir.softmax"(%arg0) <{dimension = -4 : si32}> : (tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
  return %0 : tensor<2x4x8xf32>
}
