// RUN: split-file %s %t
// RUN: tensixflow-opt --tensixflow-fuse-rmsnorm-matmul %t/positive.mlir | FileCheck %t/positive.mlir
// RUN: tensixflow-opt --tensixflow-fuse-rmsnorm-matmul %t/negative_no_gelu.mlir | FileCheck %t/negative_no_gelu.mlir

// --tensixflow-fuse-rmsnorm-matmul collapses the full MLP chain
//   rmsnorm -> matmul -> add(bias) -> gelu -> matmul -> add(residual)
// into one tensixflow.fused_mlp (registered with higher benefit than the
// rmsnorm+matmul pattern so the full chain matches first). A broken chain
// (missing gelu) must NOT produce a fused_mlp.
//
// NOTE on the bias: tensixflow.add requires identical operand/result types
// (SameOperandsAndResultType), so the bias add in a chain materializes the bias
// as a full-rank [..., H] tensor (not a 1-D [H] vector). fused_mlp accepts that
// bias by its last dimension.

//--- positive.mlir

// Full MLP chain with every intermediate single-use: the whole chain collapses
// into one fused_mlp consuming (input, gamma, w1, b1, w2, residual). The greedy
// driver renumbers SSA names, so bind them with captures from the func sig.

// CHECK-LABEL: tensixflow.func @full_chain
// CHECK-SAME: (%[[X:[a-z0-9]+]]: tensor<1x2048x4096xbf16>, %[[GAMMA:[a-z0-9]+]]: tensor<4096xbf16>, %[[W1:[a-z0-9]+]]: tensor<4096x12288xbf16>, %[[B1:[a-z0-9]+]]: tensor<1x2048x12288xbf16>, %[[W2:[a-z0-9]+]]: tensor<12288x4096xbf16>, %[[RES:[a-z0-9]+]]: tensor<1x2048x4096xbf16>) -> tensor<1x2048x4096xbf16>
// CHECK: %[[R:[a-z0-9]+]] = tensixflow.fused_mlp %[[X]], %[[GAMMA]], %[[W1]], %[[B1]], %[[W2]], %[[RES]] : tensor<1x2048x4096xbf16>, tensor<4096xbf16>, tensor<4096x12288xbf16>, tensor<1x2048x12288xbf16>, tensor<12288x4096xbf16>, tensor<1x2048x4096xbf16> -> tensor<1x2048x4096xbf16>
// CHECK-NOT: tensixflow.rmsnorm
// CHECK-NOT: tensixflow.matmul
// CHECK-NOT: tensixflow.gelu
// CHECK-NOT: tensixflow.add
// CHECK: tensixflow.return %[[R]] : tensor<1x2048x4096xbf16>
tensixflow.func @full_chain(%x: tensor<1x2048x4096xbf16>, %gamma: tensor<4096xbf16>, %w1: tensor<4096x12288xbf16>, %b1: tensor<1x2048x12288xbf16>, %w2: tensor<12288x4096xbf16>, %residual: tensor<1x2048x4096xbf16>) -> tensor<1x2048x4096xbf16> {
  %0 = tensixflow.rmsnorm %x, %gamma : tensor<1x2048x4096xbf16>, tensor<4096xbf16> -> tensor<1x2048x4096xbf16>
  %1 = tensixflow.matmul %0, %w1 : tensor<1x2048x4096xbf16> x tensor<4096x12288xbf16> -> tensor<1x2048x12288xbf16>
  %2 = tensixflow.add %1, %b1 : tensor<1x2048x12288xbf16>
  %3 = tensixflow.gelu %2 : tensor<1x2048x12288xbf16>
  %4 = tensixflow.matmul %3, %w2 : tensor<1x2048x12288xbf16> x tensor<12288x4096xbf16> -> tensor<1x2048x4096xbf16>
  %5 = tensixflow.add %4, %residual : tensor<1x2048x4096xbf16>
  tensixflow.return %5 : tensor<1x2048x4096xbf16>
}

//--- negative_no_gelu.mlir

// Broken chain: rmsnorm -> matmul -> add -> matmul -> add with NO gelu between
// the adds. The full-chain pattern must NOT fire (no fused_mlp). NOTE: the head
// rmsnorm->matmul is single-use, so it MAY still fuse to fused_rmsnorm_matmul —
// that is fine; this test only asserts there is no fused_mlp. Every result is
// kept live (chained to the return) so DCE cannot drop a use.

// CHECK-LABEL: tensixflow.func @no_gelu
// CHECK-NOT: tensixflow.fused_mlp
// CHECK: tensixflow.return
tensixflow.func @no_gelu(%x: tensor<1x2048x4096xbf16>, %gamma: tensor<4096xbf16>, %w1: tensor<4096x12288xbf16>, %b1: tensor<1x2048x12288xbf16>, %w2: tensor<12288x4096xbf16>, %residual: tensor<1x2048x4096xbf16>) -> tensor<1x2048x4096xbf16> {
  %0 = tensixflow.rmsnorm %x, %gamma : tensor<1x2048x4096xbf16>, tensor<4096xbf16> -> tensor<1x2048x4096xbf16>
  %1 = tensixflow.matmul %0, %w1 : tensor<1x2048x4096xbf16> x tensor<4096x12288xbf16> -> tensor<1x2048x12288xbf16>
  %2 = tensixflow.add %1, %b1 : tensor<1x2048x12288xbf16>
  %3 = tensixflow.matmul %2, %w2 : tensor<1x2048x12288xbf16> x tensor<12288x4096xbf16> -> tensor<1x2048x4096xbf16>
  %4 = tensixflow.add %3, %residual : tensor<1x2048x4096xbf16>
  tensixflow.return %4 : tensor<1x2048x4096xbf16>
}
