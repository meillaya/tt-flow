// RUN: split-file %s %t
// RUN: tensixflow-opt --tensixflow-plan-noc-sharding --tensixflow-allocate-circular-buffers %t/happy.mlir | FileCheck %t/happy.mlir
// RUN: tensixflow-opt --tensixflow-plan-noc-sharding --tensixflow-allocate-circular-buffers %t/failure.mlir | FileCheck %t/failure.mlir

// Happy path: an unfused rmsnorm -> matmul pair is planned as fused (the planner
// decides fusion). The schedule attribute carries the n150 8x8 grid, row-major
// activation sharding, multicast weight strategy, L1 intermediate storage, the
// fused_rmsnorm_matmul decision, the 3-CB allocation {0,1,16} (depth 2, 2048
// bytes/tile from the published L1 budget), and a positive fusion savings
// (saved_read 16777216 = the 16 MiB rmsnorm output that no longer round-trips
// DRAM).

//--- happy.mlir
// CHECK: #tensixflow.schedule<grid 8x8, act "row-major", weight "multicast", storage "L1", fusion "fused_rmsnorm_matmul"
// CHECK-SAME: cbs [0:2:2048, 1:2:2048, 16:2:2048]
// CHECK-SAME: saved_read 16777216, saved_write 16777216, saved_intermediate 16777216>
tensixflow.func @rmsnorm_matmul(%x: tensor<1x2048x4096xbf16>, %gamma: tensor<4096xbf16>, %w: tensor<4096x12288xbf16>) -> tensor<1x2048x12288xbf16> {
  %0 = tensixflow.rmsnorm %x, %gamma : tensor<1x2048x4096xbf16>, tensor<4096xbf16> -> tensor<1x2048x4096xbf16>
  %1 = tensixflow.matmul %0, %w : tensor<1x2048x4096xbf16> x tensor<4096x12288xbf16> -> tensor<1x2048x12288xbf16>
  tensixflow.return %1 : tensor<1x2048x12288xbf16>
}

// Failure path: a lone matmul (no rmsnorm feeding it) cannot be fused, so it is
// planned unfused: fusion "none", the intermediate stays in DRAM, and every
// fusion saving is zero. The CB allocation is still filled by
// --tensixflow-allocate-circular-buffers.

//--- failure.mlir
// CHECK: #tensixflow.schedule<grid 8x8, act "row-major", weight "multicast", storage "DRAM", fusion "none"
// CHECK-SAME: cbs [0:2:2048, 1:2:2048, 16:2:2048]
// CHECK-SAME: saved_read 0, saved_write 0, saved_intermediate 0>
tensixflow.func @lone_matmul(%x: tensor<1x2048x4096xbf16>, %w: tensor<4096x12288xbf16>) -> tensor<1x2048x12288xbf16> {
  %0 = tensixflow.matmul %x, %w : tensor<1x2048x4096xbf16> x tensor<4096x12288xbf16> -> tensor<1x2048x12288xbf16>
  tensixflow.return %0 : tensor<1x2048x12288xbf16>
}
