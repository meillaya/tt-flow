// RUN: tensixflow-opt %s | FileCheck %s

// The tile-layout encoding attribute must parse and print round-trip when
// attached as a discardable attribute on an op.

// CHECK-LABEL: tensixflow.func @laid_out
// CHECK: %[[G:.*]] = tensixflow.gelu
// CHECK-SAME: {tensixflow.layout = #tensixflow.tile_layout<tile 32x32, grid 8x8, mem L1>}
// CHECK-SAME: : tensor<1x2048x4096xbf16>
// CHECK: tensixflow.return %[[G]] : tensor<1x2048x4096xbf16>
tensixflow.func @laid_out(%x: tensor<1x2048x4096xbf16>) -> tensor<1x2048x4096xbf16> {
  %0 = tensixflow.gelu %x {tensixflow.layout = #tensixflow.tile_layout<tile 32x32, grid 8x8, mem L1>} : tensor<1x2048x4096xbf16>
  tensixflow.return %0 : tensor<1x2048x4096xbf16>
}

// A DRAM-space layout with a non-default grid also round-trips.

// CHECK-LABEL: tensixflow.func @dram_layout
// CHECK: %[[A:.*]] = tensixflow.add
// CHECK-SAME: {tensixflow.layout = #tensixflow.tile_layout<tile 32x32, grid 4x2, mem DRAM>}
tensixflow.func @dram_layout(%x: tensor<2048x4096xbf16>, %y: tensor<2048x4096xbf16>) -> tensor<2048x4096xbf16> {
  %0 = tensixflow.add %x, %y {tensixflow.layout = #tensixflow.tile_layout<tile 32x32, grid 4x2, mem DRAM>} : tensor<2048x4096xbf16>
  tensixflow.return %0 : tensor<2048x4096xbf16>
}
