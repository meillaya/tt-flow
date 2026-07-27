// RUN: split-file %s %t
// RUN: tensixflow-opt --tensixflow-fuse-rmsnorm-matmul %t/positive.mlir | FileCheck %t/positive.mlir
// RUN: tensixflow-opt --tensixflow-fuse-rmsnorm-matmul %t/negative_multi_use.mlir | FileCheck %t/negative_multi_use.mlir

// --tensixflow-fuse-rmsnorm-matmul collapses an adjacent rmsnorm -> matmul
// (rmsnorm result is the matmul LHS and has exactly one use) into a single
// tensixflow.fused_rmsnorm_matmul. A multi-use rmsnorm is left unfused.

//--- positive.mlir

// Single-use rmsnorm feeding a matmul as its LHS: both ops collapse into one
// fused_rmsnorm_matmul that consumes (input, gamma, weight) and feeds the
// return. The greedy driver renumbers SSA names, so bind them with captures.

// CHECK-LABEL: tensixflow.func @single_use
// CHECK-SAME: (%[[X:[a-z0-9]+]]: tensor<1x2048x4096xbf16>, %[[GAMMA:[a-z0-9]+]]: tensor<4096xbf16>, %[[W:[a-z0-9]+]]: tensor<4096x12288xbf16>) -> tensor<1x2048x12288xbf16>
// CHECK: %[[R:[a-z0-9]+]] = tensixflow.fused_rmsnorm_matmul %[[X]], %[[GAMMA]], %[[W]] {tensixflow.eliminated_bytes = 16777216 : i64, tensixflow.eliminated_tensors = 1 : i64} : tensor<1x2048x4096xbf16>, tensor<4096xbf16>, tensor<4096x12288xbf16> -> tensor<1x2048x12288xbf16>
// CHECK-NOT: tensixflow.rmsnorm
// CHECK-NOT: tensixflow.matmul
// CHECK: tensixflow.return %[[R]] : tensor<1x2048x12288xbf16>
tensixflow.func @single_use(%x: tensor<1x2048x4096xbf16>, %gamma: tensor<4096xbf16>, %w: tensor<4096x12288xbf16>) -> tensor<1x2048x12288xbf16> {
  %0 = tensixflow.rmsnorm %x, %gamma : tensor<1x2048x4096xbf16>, tensor<4096xbf16> -> tensor<1x2048x4096xbf16>
  %1 = tensixflow.matmul %0, %w : tensor<1x2048x4096xbf16> x tensor<4096x12288xbf16> -> tensor<1x2048x12288xbf16>
  tensixflow.return %1 : tensor<1x2048x12288xbf16>
}

//--- negative_multi_use.mlir

// The rmsnorm result feeds BOTH a matmul (its LHS) AND a gelu, so it has two
// uses and must stay materialized — the pair is NOT fused. Every result is
// kept live (gelu -> second matmul -> add with the first matmul -> return) so
// the greedy driver's DCE cannot remove the second use and accidentally enable
// fusion. Presence of rmsnorm AND matmul proves no fusion happened (a fused op
// would replace both).

// CHECK-LABEL: tensixflow.func @multi_use
// CHECK: tensixflow.rmsnorm
// CHECK: tensixflow.matmul
// CHECK: tensixflow.gelu
// CHECK: tensixflow.matmul
// CHECK: tensixflow.add
// CHECK-NOT: tensixflow.fused_rmsnorm_matmul
// CHECK: tensixflow.return
tensixflow.func @multi_use(%x: tensor<1x2048x4096xbf16>, %gamma: tensor<4096xbf16>, %w: tensor<4096x12288xbf16>) -> tensor<1x2048x12288xbf16> {
  %0 = tensixflow.rmsnorm %x, %gamma : tensor<1x2048x4096xbf16>, tensor<4096xbf16> -> tensor<1x2048x4096xbf16>
  %1 = tensixflow.matmul %0, %w : tensor<1x2048x4096xbf16> x tensor<4096x12288xbf16> -> tensor<1x2048x12288xbf16>
  %2 = tensixflow.gelu %0 : tensor<1x2048x4096xbf16>
  %3 = tensixflow.matmul %2, %w : tensor<1x2048x4096xbf16> x tensor<4096x12288xbf16> -> tensor<1x2048x12288xbf16>
  %4 = tensixflow.add %1, %3 : tensor<1x2048x12288xbf16>
  tensixflow.return %4 : tensor<1x2048x12288xbf16>
}
