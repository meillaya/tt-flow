// Unfused full MLP block: rmsnorm -> matmul(w1) -> add(bias) -> gelu ->
// matmul(w2) -> add(residual).
//
// This is the large analysis example. The fusion pass collapses the chain into
// a single tensixflow.fused_mlp:
//   tensixflow-opt --tensixflow-fuse-rmsnorm-matmul examples/mlp_block.mlir
// The emitted MLP program keeps the first weight block and the GELU
// intermediate rows in L1, and for this hidden width (12288) that footprint is
// far past the per-core circular-buffer budget. The emitter therefore refuses
// with a feasibility diagnostic that prints the staged intermediate tiles and
// the budget. examples/mlp_block_small.mlir is the emittable MLP example.
tensixflow.func @mlp_block(%x: tensor<1x2048x4096xbf16>, %gamma: tensor<4096xbf16>, %w1: tensor<4096x12288xbf16>, %b1: tensor<1x2048x12288xbf16>, %w2: tensor<12288x4096xbf16>, %residual: tensor<1x2048x4096xbf16>) -> tensor<1x2048x4096xbf16> {
  %0 = tensixflow.rmsnorm %x, %gamma : tensor<1x2048x4096xbf16>, tensor<4096xbf16> -> tensor<1x2048x4096xbf16>
  %1 = tensixflow.matmul %0, %w1 : tensor<1x2048x4096xbf16> x tensor<4096x12288xbf16> -> tensor<1x2048x12288xbf16>
  %2 = tensixflow.add %1, %b1 : tensor<1x2048x12288xbf16>
  %3 = tensixflow.gelu %2 : tensor<1x2048x12288xbf16>
  %4 = tensixflow.matmul %3, %w2 : tensor<1x2048x12288xbf16> x tensor<12288x4096xbf16> -> tensor<1x2048x4096xbf16>
  %5 = tensixflow.add %4, %residual : tensor<1x2048x4096xbf16>
  tensixflow.return %5 : tensor<1x2048x4096xbf16>
}
