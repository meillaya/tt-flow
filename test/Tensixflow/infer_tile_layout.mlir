// RUN: tensixflow-opt --tensixflow-infer-tile-layout %s | FileCheck %s

// --tensixflow-infer-tile-layout attaches a #tensixflow.tile_layout encoding
// as pure metadata: DRAM on function inputs (arg attrs, printed after the arg
// type), L1 on intermediate tensor results (op attrs). Shapes are unchanged.

// CHECK-LABEL: tensixflow.func @flow
// CHECK-SAME: (%[[X:[a-z0-9]+]]: tensor<1x2048x4096xbf16> {tensixflow.layout = #tensixflow.tile_layout<tile 32x32, grid 8x8, mem DRAM>}, %[[GAMMA:[a-z0-9]+]]: tensor<4096xbf16> {tensixflow.layout = #tensixflow.tile_layout<tile 32x32, grid 8x8, mem DRAM>}, %[[W:[a-z0-9]+]]: tensor<4096x12288xbf16> {tensixflow.layout = #tensixflow.tile_layout<tile 32x32, grid 8x8, mem DRAM>}) -> tensor<1x2048x12288xbf16>
tensixflow.func @flow(%x: tensor<1x2048x4096xbf16>, %gamma: tensor<4096xbf16>, %w: tensor<4096x12288xbf16>) -> tensor<1x2048x12288xbf16> {
  // CHECK: %[[N:[a-z0-9]+]] = tensixflow.rmsnorm %[[X]], %[[GAMMA]] {tensixflow.layout = #tensixflow.tile_layout<tile 32x32, grid 8x8, mem L1>} : tensor<1x2048x4096xbf16>, tensor<4096xbf16> -> tensor<1x2048x4096xbf16>
  %0 = tensixflow.rmsnorm %x, %gamma : tensor<1x2048x4096xbf16>, tensor<4096xbf16> -> tensor<1x2048x4096xbf16>
  // CHECK: %[[M:[a-z0-9]+]] = tensixflow.matmul %[[N]], %[[W]] {tensixflow.layout = #tensixflow.tile_layout<tile 32x32, grid 8x8, mem L1>} : tensor<1x2048x4096xbf16> x tensor<4096x12288xbf16> -> tensor<1x2048x12288xbf16>
  %1 = tensixflow.matmul %0, %w : tensor<1x2048x4096xbf16> x tensor<4096x12288xbf16> -> tensor<1x2048x12288xbf16>
  // CHECK: tensixflow.return %[[M]] : tensor<1x2048x12288xbf16>
  tensixflow.return %1 : tensor<1x2048x12288xbf16>
}

// Robustness: a non-tensor (i32) argument gets NO layout attr and does not
// crash. The [a-z0-9]+ SSA-name regex asserts nothing (no {attrs}) sits
// between %s and ": i32".
// CHECK-LABEL: tensixflow.func @scalar_arg
// CHECK-SAME: (%[[S:[a-z0-9]+]]: i32, %[[T:[a-z0-9]+]]: tensor<2048x4096xbf16> {tensixflow.layout = #tensixflow.tile_layout<tile 32x32, grid 8x8, mem DRAM>}) -> tensor<2048x4096xbf16>
tensixflow.func @scalar_arg(%s: i32, %t: tensor<2048x4096xbf16>) -> tensor<2048x4096xbf16> {
  // CHECK: tensixflow.gelu %[[T]] {tensixflow.layout = #tensixflow.tile_layout<tile 32x32, grid 8x8, mem L1>} : tensor<2048x4096xbf16>
  %0 = tensixflow.gelu %t : tensor<2048x4096xbf16>
  tensixflow.return %0 : tensor<2048x4096xbf16>
}
