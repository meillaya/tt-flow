// Small pure matmul used by scripts/build_generated.sh when examples/matmul.mlir
// is not present. 64x64 inputs are two tiles per dimension, small enough to run
// in seconds under the simulator while still exercising more than one tile.
tensixflow.func @matmul(%a: tensor<64x64xbf16>, %b: tensor<64x64xbf16>) -> tensor<64x64xbf16> {
  %0 = tensixflow.matmul %a, %b : tensor<64x64xbf16> x tensor<64x64xbf16> -> tensor<64x64xbf16>
  tensixflow.return %0 : tensor<64x64xbf16>
}
