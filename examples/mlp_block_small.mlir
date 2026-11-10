// Fused MLP block codegen quickstart: the full chain
//   rmsnorm -> matmul(w1) -> add(bias) -> gelu -> matmul(w2) -> add(residual)
// collapses to tensixflow.fused_mlp, and the emitter stages one activation row
// block in L1, keeps the GELU intermediate in L1 between the two matmuls, and
// adds the residual on the way out. Every dimension is a whole number of 32x32
// tiles, so the tile layout and circular-buffer depths align without padding.
//
// Emit (the fusion pass runs first so the planner and the emitter see one
// fused_mlp op):
//   tensixflow-opt --tensixflow-device=n150 \
//     --tensixflow-fuse-rmsnorm-matmul \
//     --tensixflow-plan-noc-sharding \
//     --tensixflow-allocate-circular-buffers \
//     --tensixflow-lower-to-ttkernel \
//     --tensixflow-emit-metalium=output-dir=<dir> \
//     examples/mlp_block_small.mlir
// then build and run it under the pinned simulator with
// scripts/build_generated.sh and scripts/run_generated.sh.
tensixflow.func @mlp_block_small(%x: tensor<1x128x128xbf16>, %gamma: tensor<128xbf16>, %w1: tensor<128x128xbf16>, %b1: tensor<1x128x128xbf16>, %w2: tensor<128x128xbf16>, %residual: tensor<1x128x128xbf16>) -> tensor<1x128x128xbf16> {
  %0 = tensixflow.rmsnorm %x, %gamma : tensor<1x128x128xbf16>, tensor<128xbf16> -> tensor<1x128x128xbf16>
  %1 = tensixflow.matmul %0, %w1 : tensor<1x128x128xbf16> x tensor<128x128xbf16> -> tensor<1x128x128xbf16>
  %2 = tensixflow.add %1, %b1 : tensor<1x128x128xbf16>
  %3 = tensixflow.gelu %2 : tensor<1x128x128xbf16>
  %4 = tensixflow.matmul %3, %w2 : tensor<1x128x128xbf16> x tensor<128x128xbf16> -> tensor<1x128x128xbf16>
  %5 = tensixflow.add %4, %residual : tensor<1x128x128xbf16>
  tensixflow.return %5 : tensor<1x128x128xbf16>
}
