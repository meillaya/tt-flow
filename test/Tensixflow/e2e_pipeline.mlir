// RUN: tensixflow-opt --tensixflow-infer-tile-layout --tensixflow-canonicalize %s | FileCheck %s

// End-to-end: parse -> --tensixflow-infer-tile-layout (attaches DRAM layouts to
// func args, L1 layouts to intermediate results) -> --tensixflow-canonicalize.
// No fusion yet (that is Wave 4); the rmsnorm and matmul stay separate, so the
// CHECK lines assert BOTH ops are still present and the graph is well-formed.
// The greedy canonicalizer renumbers SSA names, so they are bound with regex
// captures ([a-z0-9]+) and reused, never hardcoded.

// DRAM layout on every func arg (printed after each arg type).
// CHECK-LABEL: tensixflow.func @e2e
// CHECK-SAME: (%[[X:[a-z0-9]+]]: tensor<1x2048x4096xbf16> {tensixflow.layout = #tensixflow.tile_layout<tile 32x32, grid 8x8, mem DRAM>}, %[[W:[a-z0-9]+]]: tensor<4096x12288xbf16> {tensixflow.layout = #tensixflow.tile_layout<tile 32x32, grid 8x8, mem DRAM>}, %[[GAMMA:[a-z0-9]+]]: tensor<4096xbf16> {tensixflow.layout = #tensixflow.tile_layout<tile 32x32, grid 8x8, mem DRAM>}) -> tensor<1x2048x12288xbf16>
// L1 layout on the rmsnorm result; the rmsnorm op is present (not fused).
// CHECK: %[[N:[a-z0-9]+]] = tensixflow.rmsnorm %[[X]], %[[GAMMA]] {tensixflow.layout = #tensixflow.tile_layout<tile 32x32, grid 8x8, mem L1>} : tensor<1x2048x4096xbf16>, tensor<4096xbf16> -> tensor<1x2048x4096xbf16>
// L1 layout on the matmul result; the matmul op is present (not fused) and
// consumes the rmsnorm result.
// CHECK: %[[M:[a-z0-9]+]] = tensixflow.matmul %[[N]], %[[W]] {tensixflow.layout = #tensixflow.tile_layout<tile 32x32, grid 8x8, mem L1>} : tensor<1x2048x4096xbf16> x tensor<4096x12288xbf16> -> tensor<1x2048x12288xbf16>
// The graph is well-formed: the return uses the matmul result.
// CHECK: tensixflow.return %[[M]] : tensor<1x2048x12288xbf16>
tensixflow.func @e2e(%x: tensor<1x2048x4096xbf16>, %w: tensor<4096x12288xbf16>, %gamma: tensor<4096xbf16>) -> tensor<1x2048x12288xbf16> {
  %0 = tensixflow.rmsnorm %x, %gamma : tensor<1x2048x4096xbf16>, tensor<4096xbf16> -> tensor<1x2048x4096xbf16>
  %1 = tensixflow.matmul %0, %w : tensor<1x2048x4096xbf16> x tensor<4096x12288xbf16> -> tensor<1x2048x12288xbf16>
  tensixflow.return %1 : tensor<1x2048x12288xbf16>
}
