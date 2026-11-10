// Layout-encoded TTIR in the pinned tt-mlir 0.9.0.dev20260221 syntax. The
// tensors carry #ttcore.metal_layout encodings and the module carries a
// ttcore.meshes device attribute, so --tensixflow-import-ttir rejects this
// file. Normalize it first, then import and plan it:
//
//   bash scripts/normalize_ttir.sh examples/ttir/rmsnorm_matmul_layout.mlir \
//     -o /tmp/rmsnorm_matmul_bare.mlir
//   tensixflow-opt --allow-unregistered-dialect --tensixflow-import-ttir \
//     --tensixflow-infer-tile-layout --tensixflow-fuse-rmsnorm-matmul \
//     --tensixflow-plan-noc-sharding --tensixflow-allocate-circular-buffers \
//     /tmp/rmsnorm_matmul_bare.mlir
//
// The two attribute forms are quoted from the pinned fixtures fetched by
// scripts/fetch_ttmlir_fixtures.sh: the encoding field order comes from
// test/ttmlir/Dialect/TTCore/metal_layout_attr_verify.mlir and the module
// attribute from test/ttmlir/Dialect/TTIR/fusing/gelu_fusing.mlir. The
// placement convention (dram for arguments, l1 for results, no encoding on a
// rank-1 weight) matches what --tensixflow-emit-ttir writes.
#layout_input = #ttcore.metal_layout<logical_shape = 1x128x256, dim_alignments = 1x32x32, collapsed_intervals = dense<[[0, 2], [2, 3]]> : tensor<2x2xi64>, undef, dram, interleaved, index_map = map(0)>
#layout_normalized = #ttcore.metal_layout<logical_shape = 1x128x256, dim_alignments = 1x32x32, collapsed_intervals = dense<[[0, 2], [2, 3]]> : tensor<2x2xi64>, undef, l1, interleaved, index_map = map(0)>
#layout_weight = #ttcore.metal_layout<logical_shape = 256x128, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, dram, interleaved, index_map = map(0)>
#layout_result = #ttcore.metal_layout<logical_shape = 1x128x128, dim_alignments = 1x32x32, collapsed_intervals = dense<[[0, 2], [2, 3]]> : tensor<2x2xi64>, undef, l1, interleaved, index_map = map(0)>
module attributes {ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x1>]>} {
  func.func @rmsnorm_matmul(%arg0: tensor<1x128x256xbf16, #layout_input>, %arg1: tensor<256xbf16>, %arg2: tensor<256x128xbf16, #layout_weight>) -> tensor<1x128x128xbf16, #layout_result> {
    %0 = "ttir.rms_norm"(%arg0, %arg1) <{epsilon = 9.99999997E-7 : f32, normalized_shape = array<i64: 256>, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x128x256xbf16, #layout_input>, tensor<256xbf16>) -> tensor<1x128x256xbf16, #layout_normalized>
    %1 = "ttir.matmul"(%0, %arg2) : (tensor<1x128x256xbf16, #layout_normalized>, tensor<256x128xbf16, #layout_weight>) -> tensor<1x128x128xbf16, #layout_result>
    return %1 : tensor<1x128x128xbf16, #layout_result>
  }
}
