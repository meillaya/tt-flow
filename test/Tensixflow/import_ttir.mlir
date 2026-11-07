// RUN: split-file %s %t
// RUN: tensixflow-opt --allow-unregistered-dialect --tensixflow-import-ttir %t/elementwise.mlir | FileCheck %t/elementwise.mlir
// RUN: tensixflow-opt --allow-unregistered-dialect --tensixflow-import-ttir %t/f32_matmul.mlir | FileCheck %t/f32_matmul.mlir
// RUN: tensixflow-opt --allow-unregistered-dialect --tensixflow-import-ttir %S/../../examples/ttir/rmsnorm_matmul.mlir | FileCheck %s --check-prefix=IMPORTRMS
// RUN: tensixflow-opt --allow-unregistered-dialect --tensixflow-import-ttir --tensixflow-infer-tile-layout %S/../../examples/ttir/matmul.mlir | FileCheck %s --check-prefix=MATMUL
// RUN: tensixflow-opt --allow-unregistered-dialect --tensixflow-import-ttir --tensixflow-infer-tile-layout %S/../../examples/ttir/rmsnorm_matmul.mlir | FileCheck %s --check-prefix=RMSNORM
// RUN: tensixflow-opt --allow-unregistered-dialect --tensixflow-import-ttir --tensixflow-infer-tile-layout --tensixflow-plan-noc-sharding --tensixflow-allocate-circular-buffers %S/../../examples/ttir/matmul.mlir | FileCheck %s --check-prefix=PLANMATMUL
// RUN: tensixflow-opt --allow-unregistered-dialect --tensixflow-import-ttir --tensixflow-infer-tile-layout --tensixflow-plan-noc-sharding --tensixflow-allocate-circular-buffers %S/../../examples/ttir/rmsnorm_matmul.mlir | FileCheck %s --check-prefix=PLANRMSNORM

// Import test for --tensixflow-import-ttir. The pass turns bare-tensor TTIR
// into tensixflow IR so the planning pipeline runs unchanged. The
// --allow-unregistered-dialect flag is required because this build does not
// link the ttir dialect; the source ops parse as unregistered generic ops.

// IMPORTRMS: tensixflow.rmsnorm %{{.*}}, %{{.*}} {epsilon = 9.99999974E-6 : f32} : tensor<1x2048x4096xbf16>, tensor<4096xbf16> -> tensor<1x2048x4096xbf16>

// MATMUL: module attributes {tensixflow.device = "n150"} {
// MATMUL-LABEL: tensixflow.func @matmul
// MATMUL-SAME: (%[[A:.*]]: tensor<1x2048x4096xbf16> {tensixflow.layout = #tensixflow.tile_layout<tile 32x32, grid 8x8, mem DRAM>}, %[[B:.*]]: tensor<4096x12288xbf16> {tensixflow.layout = #tensixflow.tile_layout<tile 32x32, grid 8x8, mem DRAM>}) -> tensor<1x2048x12288xbf16>
// MATMUL: %[[R:.*]] = tensixflow.matmul %[[A]], %[[B]] {tensixflow.layout = #tensixflow.tile_layout<tile 32x32, grid 8x8, mem L1>} : tensor<1x2048x4096xbf16> x tensor<4096x12288xbf16> -> tensor<1x2048x12288xbf16>
// MATMUL: tensixflow.return %[[R]] : tensor<1x2048x12288xbf16>

// RMSNORM: module attributes {tensixflow.device = "n150"} {
// RMSNORM-LABEL: tensixflow.func @rmsnorm_matmul
// RMSNORM-SAME: (%[[X:.*]]: tensor<1x2048x4096xbf16> {tensixflow.layout = #tensixflow.tile_layout<tile 32x32, grid 8x8, mem DRAM>}, %[[G:.*]]: tensor<4096xbf16> {tensixflow.layout = #tensixflow.tile_layout<tile 32x32, grid 8x8, mem DRAM>}, %[[W:.*]]: tensor<4096x12288xbf16> {tensixflow.layout = #tensixflow.tile_layout<tile 32x32, grid 8x8, mem DRAM>}) -> tensor<1x2048x12288xbf16>
// RMSNORM: %[[N:.*]] = tensixflow.rmsnorm %[[X]], %[[G]] {epsilon = 9.99999974E-6 : f32, tensixflow.layout = #tensixflow.tile_layout<tile 32x32, grid 8x8, mem L1>} : tensor<1x2048x4096xbf16>, tensor<4096xbf16> -> tensor<1x2048x4096xbf16>
// RMSNORM: %[[M:.*]] = tensixflow.matmul %[[N]], %[[W]] {tensixflow.layout = #tensixflow.tile_layout<tile 32x32, grid 8x8, mem L1>} : tensor<1x2048x4096xbf16> x tensor<4096x12288xbf16> -> tensor<1x2048x12288xbf16>
// RMSNORM: tensixflow.return %[[M]] : tensor<1x2048x12288xbf16>

// The planner decides fusion from graph shape, not from the input dialect, so
// the imported rmsnorm -> matmul pair plans as fused and the imported single
// matmul plans as unfused. The expected schedule values match the native tests.
// PLANMATMUL: tensixflow.schedule = #tensixflow.schedule<grid 8x8, act "row-major", weight "multicast", storage "DRAM", fusion "none"
// PLANMATMUL-SAME: cbs [0:2:2048, 1:2:2048, 16:2:2048]
// PLANMATMUL-SAME: saved_read 0, saved_write 0, saved_intermediate 0>
// PLANRMSNORM: tensixflow.schedule = #tensixflow.schedule<grid 8x8, act "row-major", weight "multicast", storage "L1", fusion "fused_rmsnorm_matmul"
// PLANRMSNORM-SAME: cbs [0:2:2048, 1:2:2048, 16:2:2048]
// PLANRMSNORM-SAME: saved_read 16777216, saved_write 16777216, saved_intermediate 16777216>

//--- elementwise.mlir
// CHECK-LABEL: tensixflow.func @elementwise
// CHECK: %[[SUM:.*]] = tensixflow.add %{{.*}}, %{{.*}} : tensor<2048x4096xbf16>
// CHECK: %[[ACT:.*]] = tensixflow.gelu %[[SUM]] : tensor<2048x4096xbf16>
// CHECK: tensixflow.return %[[ACT]] : tensor<2048x4096xbf16>
func.func @elementwise(%arg0: tensor<2048x4096xbf16>, %arg1: tensor<2048x4096xbf16>) -> tensor<2048x4096xbf16> {
  %0 = "ttir.add"(%arg0, %arg1) : (tensor<2048x4096xbf16>, tensor<2048x4096xbf16>) -> tensor<2048x4096xbf16>
  %1 = "ttir.gelu"(%0) : (tensor<2048x4096xbf16>) -> tensor<2048x4096xbf16>
  return %1 : tensor<2048x4096xbf16>
}

//--- f32_matmul.mlir
// CHECK-LABEL: tensixflow.func @f32_matmul
// CHECK: tensixflow.matmul %{{.*}}, %{{.*}} : tensor<64x128xf32> x tensor<128x32xf32> -> tensor<64x32xf32>
// CHECK: tensixflow.return %{{.*}} : tensor<64x32xf32>
func.func @f32_matmul(%arg0: tensor<64x128xf32>, %arg1: tensor<128x32xf32>) -> tensor<64x32xf32> {
  %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<64x128xf32>, tensor<128x32xf32>) -> tensor<64x32xf32>
  return %0 : tensor<64x32xf32>
}
