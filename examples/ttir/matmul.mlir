// Bare-tensor TTIR matmul in the subset that --tensixflow-import-ttir accepts.
// Run:
//   tensixflow-opt --allow-unregistered-dialect --tensixflow-import-ttir \
//     --tensixflow-infer-tile-layout --tensixflow-plan-noc-sharding \
//     --tensixflow-allocate-circular-buffers examples/ttir/matmul.mlir
// The --allow-unregistered-dialect flag is required because this build does
// not link the ttir dialect; the ttir.matmul op parses as an unregistered
// generic operation and the import pass rebuilds it as tensixflow.matmul.
func.func @matmul(%arg0: tensor<1x2048x4096xbf16>, %arg1: tensor<4096x12288xbf16>) -> tensor<1x2048x12288xbf16> {
  %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<1x2048x4096xbf16>, tensor<4096x12288xbf16>) -> tensor<1x2048x12288xbf16>
  return %0 : tensor<1x2048x12288xbf16>
}
