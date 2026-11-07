// RUN: split-file %s %t
// RUN: tensixflow-opt --tensixflow-device=n300 --tensixflow-infer-tile-layout %t/infer.mlir | FileCheck %t/infer.mlir --check-prefix=N300
// RUN: tensixflow-opt --tensixflow-device=p100a --tensixflow-infer-tile-layout %t/infer.mlir | FileCheck %t/infer.mlir --check-prefix=P100A
// RUN: tensixflow-opt --tensixflow-infer-tile-layout %t/infer.mlir | FileCheck %t/infer.mlir --check-prefix=DEFAULT
// RUN: not tensixflow-opt --tensixflow-device=bogus %t/infer.mlir 2>&1 | FileCheck %t/unknown.mlir

// The --tensixflow-device option selects the grid stamped on the values. The
// option defaults to n150 and an unknown name is rejected with the valid list.

//--- infer.mlir
// N300: module attributes {tensixflow.device = "n300"} {
// N300: #tensixflow.tile_layout<tile 32x32, grid 7x8, mem DRAM>
// P100A: module attributes {tensixflow.device = "p100a"} {
// P100A: #tensixflow.tile_layout<tile 32x32, grid 10x12, mem DRAM>
// DEFAULT: module attributes {tensixflow.device = "n150"} {
// DEFAULT: #tensixflow.tile_layout<tile 32x32, grid 8x8, mem DRAM>
tensixflow.func @add(%a: tensor<64x64xbf16>, %b: tensor<64x64xbf16>) -> tensor<64x64xbf16> {
  %0 = tensixflow.add %a, %b : tensor<64x64xbf16>
  tensixflow.return %0 : tensor<64x64xbf16>
}

//--- unknown.mlir
// CHECK: unknown device 'bogus'; valid devices: n150, n300, p100a, p150a
