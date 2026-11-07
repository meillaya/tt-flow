// Bare-tensor TTIR RMSNorm -> matmul pair in the subset that
// --tensixflow-import-ttir accepts. The generic forms mirror the pinned
// tt-mlir fixtures: rms_norm takes the input and the rank-1 weight and carries
// normalized_shape and epsilon, and matmul has no transpose attributes.
// Run:
//   tensixflow-opt --allow-unregistered-dialect --tensixflow-import-ttir \
//     --tensixflow-infer-tile-layout --tensixflow-plan-noc-sharding \
//     --tensixflow-allocate-circular-buffers examples/ttir/rmsnorm_matmul.mlir
func.func @rmsnorm_matmul(%arg0: tensor<1x2048x4096xbf16>, %arg1: tensor<4096xbf16>, %arg2: tensor<4096x12288xbf16>) -> tensor<1x2048x12288xbf16> {
  %0 = "ttir.rms_norm"(%arg0, %arg1) <{epsilon = 1.000000e-05 : f32, normalized_shape = array<i64: 4096>, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x2048x4096xbf16>, tensor<4096xbf16>) -> tensor<1x2048x4096xbf16>
  %1 = "ttir.matmul"(%0, %arg2) : (tensor<1x2048x4096xbf16>, tensor<4096x12288xbf16>) -> tensor<1x2048x12288xbf16>
  return %1 : tensor<1x2048x12288xbf16>
}
