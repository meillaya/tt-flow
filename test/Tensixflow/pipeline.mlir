// RUN: tensixflow-opt --pass-pipeline='builtin.module(tensixflow-infer-tile-layout,tensixflow-canonicalize)' %s | FileCheck %s

// The D22 probe: proves the two-pass --pass-pipeline REGISTRATION +
// COMPOSITION works (the pipeline syntax was unverified during planning).
// infer-tile-layout now attaches layouts (DRAM on the func arg, L1 on the
// gelu result); canonicalize is still a no-op stub.

// CHECK-LABEL: tensixflow.func @probe
// CHECK-SAME: (%[[X:[a-z0-9]+]]: tensor<2048x4096xbf16> {tensixflow.layout = #tensixflow.tile_layout<tile 32x32, grid 8x8, mem DRAM>}) -> tensor<2048x4096xbf16>
tensixflow.func @probe(%x: tensor<2048x4096xbf16>) -> tensor<2048x4096xbf16> {
  // CHECK: %[[G:[a-z0-9]+]] = tensixflow.gelu %[[X]] {tensixflow.layout = #tensixflow.tile_layout<tile 32x32, grid 8x8, mem L1>} : tensor<2048x4096xbf16>
  %0 = tensixflow.gelu %x : tensor<2048x4096xbf16>
  // CHECK: tensixflow.return %[[G]] : tensor<2048x4096xbf16>
  tensixflow.return %0 : tensor<2048x4096xbf16>
}
