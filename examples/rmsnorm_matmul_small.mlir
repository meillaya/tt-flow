// Fused RMSNorm -> MatMul codegen quickstart: bf16 operands, every dimension
// divisible by the 32x32 tile so the tile layout and CB depths align without
// padding. The planner fuses the single-use rmsnorm into the matmul, and the
// emitter stages each core's activation rows once in L1.
// Emit:
//   tensixflow-opt --tensixflow-device=n150 \
//     --tensixflow-plan-noc-sharding \
//     --tensixflow-allocate-circular-buffers \
//     --tensixflow-lower-to-ttkernel \
//     --tensixflow-emit-metalium=output-dir=<dir> \
//     examples/rmsnorm_matmul_small.mlir
// to write a standalone TT-Metalium project, then build and run it under the
// pinned simulator with scripts/build_generated.sh and scripts/run_generated.sh.
tensixflow.func @rmsnorm_matmul_small(%x: tensor<128x128xbf16>, %gamma: tensor<128xbf16>, %w: tensor<128x256xbf16>) -> tensor<128x256xbf16> {
  %0 = tensixflow.rmsnorm %x, %gamma : tensor<128x128xbf16>, tensor<128xbf16> -> tensor<128x128xbf16>
  %1 = tensixflow.matmul %0, %w : tensor<128x128xbf16> x tensor<128x256xbf16> -> tensor<128x256xbf16>
  tensixflow.return %1 : tensor<128x256xbf16>
}
