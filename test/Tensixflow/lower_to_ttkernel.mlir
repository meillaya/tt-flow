// RUN: split-file %s %t
// RUN: tensixflow-opt --tensixflow-plan-noc-sharding --tensixflow-allocate-circular-buffers --tensixflow-lower-to-ttkernel %t/happy.mlir | FileCheck %t/happy.mlir
// RUN: not tensixflow-opt --tensixflow-lower-to-ttkernel %t/no_schedule.mlir 2>&1 | FileCheck %t/no_schedule.mlir

// Happy path: the full planning pipeline runs over an UNFUSED rmsnorm -> matmul
// pair (planned AS fused_rmsnorm_matmul), then --tensixflow-lower-to-ttkernel
// lowers the plan into a tensixflow.sched.kernel holding reader / compute /
// writer tile programs. The graph func is consumed (lowered away); the kernel is
// the sole symbol. CB indices {0,1,16} come from the schedule's 3-CB allocation;
// the weight strategy "multicast" selects sched.noc_multicast for the weight
// tiles; the compute region carries the rmsnorm prologue + matmul_tiles. Trip
// counts are the first-order tile decomposition over the 32x32 tile: M=2048 ->
// 64 row-tiles, K=4096 -> 128 K-tiles (reader/compute trip), N=12288 -> 384
// col-tiles, so the writer drains 64*384 = 24576 output tiles.

//--- happy.mlir
// CHECK-NOT: tensixflow.func
// CHECK-LABEL: tensixflow.sched.kernel @rmsnorm_matmul
// CHECK: tensixflow.sched.reader {
// CHECK: tensixflow.sched.tile_loop 128 {
// CHECK: tensixflow.sched.cb_reserve_back 0, 1
// CHECK: tensixflow.sched.noc_read 0, 0, 1
// CHECK: tensixflow.sched.cb_push_back 0, 1
// CHECK: tensixflow.sched.cb_reserve_back 1, 1
// CHECK: tensixflow.sched.noc_multicast 1, 0, 1, 8
// CHECK: tensixflow.sched.cb_push_back 1, 1
// CHECK: tensixflow.sched.compute {
// CHECK: tensixflow.sched.tile_loop 128 {
// CHECK: tensixflow.sched.cb_wait_front 0, 1
// CHECK: tensixflow.sched.cb_wait_front 1, 1
// CHECK: tensixflow.sched.rmsnorm 0, 2, 0
// CHECK: tensixflow.sched.matmul_tiles 0, 1, 16
// CHECK: tensixflow.sched.cb_pop_front 0, 1
// CHECK: tensixflow.sched.cb_pop_front 1, 1
// CHECK: tensixflow.sched.writer {
// CHECK: tensixflow.sched.tile_loop 24576 {
// CHECK: tensixflow.sched.cb_wait_front 16, 1
// CHECK: tensixflow.sched.noc_write 16, 0, 1
// CHECK: tensixflow.sched.cb_pop_front 16, 1
tensixflow.func @rmsnorm_matmul(%x: tensor<1x2048x4096xbf16>, %gamma: tensor<4096xbf16>, %w: tensor<4096x12288xbf16>) -> tensor<1x2048x12288xbf16> {
  %0 = tensixflow.rmsnorm %x, %gamma : tensor<1x2048x4096xbf16>, tensor<4096xbf16> -> tensor<1x2048x4096xbf16>
  %1 = tensixflow.matmul %0, %w : tensor<1x2048x4096xbf16> x tensor<4096x12288xbf16> -> tensor<1x2048x12288xbf16>
  tensixflow.return %1 : tensor<1x2048x12288xbf16>
}

// Failure path: a module WITHOUT a tensixflow.schedule attribute is rejected with
// a clear error (the required failure path). Isolated in its own split-file
// section; the error goes to stderr, so `not ... 2>&1 | FileCheck` asserts it.

//--- no_schedule.mlir
// CHECK: requires a tensixflow.schedule attribute
tensixflow.func @rmsnorm_matmul(%x: tensor<1x2048x4096xbf16>, %gamma: tensor<4096xbf16>, %w: tensor<4096x12288xbf16>) -> tensor<1x2048x12288xbf16> {
  %0 = tensixflow.rmsnorm %x, %gamma : tensor<1x2048x4096xbf16>, tensor<4096xbf16> -> tensor<1x2048x4096xbf16>
  %1 = tensixflow.matmul %0, %w : tensor<1x2048x4096xbf16> x tensor<4096x12288xbf16> -> tensor<1x2048x12288xbf16>
  tensixflow.return %1 : tensor<1x2048x12288xbf16>
}
