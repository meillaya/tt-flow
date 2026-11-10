// RUN: split-file %s %t
// RUN: tensixflow-opt --tensixflow-device=n150 --tensixflow-fuse-rmsnorm-matmul --tensixflow-plan-noc-sharding --tensixflow-allocate-circular-buffers --tensixflow-lower-to-ttkernel --tensixflow-emit-metalium %t/small.mlir | FileCheck %t/small.mlir
// RUN: rm -rf %t/project
// RUN: tensixflow-opt --tensixflow-device=n150 --tensixflow-fuse-rmsnorm-matmul --tensixflow-plan-noc-sharding --tensixflow-allocate-circular-buffers --tensixflow-lower-to-ttkernel --tensixflow-emit-metalium=output-dir=%t/project %t/small.mlir
// RUN: test -s %t/project/host.cpp
// RUN: test -s %t/project/kernels/reader.cpp
// RUN: test -s %t/project/kernels/compute.cpp
// RUN: test -s %t/project/kernels/writer.cpp
// RUN: grep -q 'fused MLP' %t/project/README.md
// RUN: grep -q 'GELU intermediate' %t/project/README.md
// RUN: grep -q 'accurate GELU' %t/project/README.md
// RUN: grep -q 'TENSIXFLOW_RESULT' %t/project/host.cpp
// RUN: not tensixflow-opt --tensixflow-device=n150 --tensixflow-fuse-rmsnorm-matmul --tensixflow-plan-noc-sharding --tensixflow-allocate-circular-buffers --tensixflow-lower-to-ttkernel --tensixflow-emit-metalium %t/large.mlir 2>&1 | FileCheck %t/large.mlir --check-prefix=INFEASIBLE

// Structural test for the emitted fused MLP project.
//
// Small graph: M=128, K=128, H=128, N=128 over the n150 grid. The fusion pass
// collapses the whole chain into tensixflow.fused_mlp. The planning pass picks
// the 2D block assignment (4 x 4 used blocks, one activation row and one output
// column tile per core) and records the emitted-path estimate, then the emitter
// allocates the fused MLP twelve-buffer layout and re-stamps the schedule copy.
//
// The generated compute kernel emits the RMSNorm prologue, the first matmul with
// the bias and the accurate GELU, the L1 intermediate pack, the second matmul,
// and the residual add. The emitted project is built and run under ttsim by
// scripts/test.sh, which requires the pass status.
//
// Large graph: M=2048, K=4096, H=12288, N=4096. The resident weight block and
// the intermediate rows do not fit the n150's per-core circular-buffer budget,
// so the emitter refuses with a diagnostic that prints the intermediate staging
// tiles, the whole-chain footprint, and the available tiles.

//--- small.mlir

// CHECK: constexpr uint32_t kMTiles = 4;
// CHECK: constexpr uint32_t kKTiles = 4;
// CHECK: constexpr uint32_t kHTiles = 4;
// CHECK: constexpr uint32_t kNTiles = 4;
// CHECK: constexpr uint32_t kUsedRows = 4;
// CHECK: constexpr uint32_t kUsedCols = 4;
// CHECK: constexpr uint32_t kRowsPerCore = 1;
// CHECK: constexpr uint32_t kColsPerCore = 1;
// CHECK: constexpr uint32_t kCbW1Tiles = 16;
// CHECK: constexpr uint32_t kCbInterTiles = 4;
// CHECK: // Zero-pad a row-major bf16 matrix to whole tiles.
// CHECK: std::erf(sum * 0.70710678f)
// CHECK: for (uint32_t rowBlock = 0; rowBlock < kUsedRows; ++rowBlock) {
// CHECK: {rowCount, colCount});
// CHECK: // Zero-pad the operands to whole tiles; the device computes on the
// CHECK: goldenMlp(activationPadded, gammaPadded, w1Padded, biasPadded, w2Padded,
// CHECK: TENSIXFLOW_RESULT
// CHECK: // ===== file: compute.cpp =====
// CHECK: constexpr tt::CBIndex cb_sq = tt::CBIndex::c_1
// CHECK: constexpr tt::CBIndex cb_invrms = tt::CBIndex::c_2
// CHECK: constexpr tt::CBIndex cb_norm = tt::CBIndex::c_3
// CHECK: constexpr tt::CBIndex cb_w1 = tt::CBIndex::c_4
// CHECK: constexpr tt::CBIndex cb_w2 = tt::CBIndex::c_8
// CHECK: constexpr tt::CBIndex cb_inter = tt::CBIndex::c_10
// CHECK: constexpr tt::CBIndex cb_out = tt::CBIndex::c_16
// CHECK: compute_kernel_hw_startup<SrcOrder::Reverse>(cb_norm, cb_w1, cb_out)
// CHECK: reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(
// CHECK: mul_unary_tile(0, kInverseKBits)
// CHECK: add_unary_tile(0, kEpsilonBits)
// CHECK: rsqrt_tile(0)
// CHECK: mul_tiles_bcast_cols(cb_act, cb_invrms, kt, 0, 0)
// CHECK: mul_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
// CHECK: matmul_tiles(cb_norm, cb_w1, kt, kt * kHt + h, 0)
// CHECK: add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
// CHECK: gelu_tile</*fast_and_approx=*/false>(0)
// CHECK: cb_reserve_back(cb_inter, 1)
// CHECK: matmul_tiles(cb_inter, cb_w2, row * kHt + ht,
// CHECK: // ===== file: reader.cpp =====
// CHECK: noc_async_read_page(rowStart * kKt + kt, act,
// CHECK: noc_async_read_page(kt * kHt + h, w1, get_write_ptr(cb_w1))
// CHECK: noc_async_read_page(ht * kNt + col, w2, get_write_ptr(cb_w2))
// CHECK: noc_async_read_page((rowStart + row) * kNt + col, residual,
// CHECK: // ===== file: writer.cpp =====
// CHECK: constexpr uint32_t cb_out = static_cast<uint32_t>(tt::CBIndex::c_16);
// CHECK: (rowStart + row) * kNt + colStart + localCol
tensixflow.func @mlp_block_small(%x: tensor<1x128x128xbf16>, %gamma: tensor<128xbf16>, %w1: tensor<128x128xbf16>, %b1: tensor<1x128x128xbf16>, %w2: tensor<128x128xbf16>, %residual: tensor<1x128x128xbf16>) -> tensor<1x128x128xbf16> {
  %0 = tensixflow.rmsnorm %x, %gamma : tensor<1x128x128xbf16>, tensor<128xbf16> -> tensor<1x128x128xbf16>
  %1 = tensixflow.matmul %0, %w1 : tensor<1x128x128xbf16> x tensor<128x128xbf16> -> tensor<1x128x128xbf16>
  %2 = tensixflow.add %1, %b1 : tensor<1x128x128xbf16>
  %3 = tensixflow.gelu %2 : tensor<1x128x128xbf16>
  %4 = tensixflow.matmul %3, %w2 : tensor<1x128x128xbf16> x tensor<128x128xbf16> -> tensor<1x128x128xbf16>
  %5 = tensixflow.add %4, %residual : tensor<1x128x128xbf16>
  tensixflow.return %5 : tensor<1x128x128xbf16>
}

//--- large.mlir
// INFEASIBLE: error: the fused plan 'fused_mlp'
// INFEASIBLE: keeps 3072 intermediate (GELU output) circular-buffer tiles
// INFEASIBLE: needs 59284 circular-buffer tiles per core
// INFEASIBLE: against 700 available tiles per core on 'n150'
// INFEASIBLE: examples/mlp_block_small.mlir
// INFEASIBLE: tensixflow-report
tensixflow.func @mlp_block(%x: tensor<1x2048x4096xbf16>, %gamma: tensor<4096xbf16>, %w1: tensor<4096x12288xbf16>, %b1: tensor<1x2048x12288xbf16>, %w2: tensor<12288x4096xbf16>, %residual: tensor<1x2048x4096xbf16>) -> tensor<1x2048x4096xbf16> {
  %0 = tensixflow.rmsnorm %x, %gamma : tensor<1x2048x4096xbf16>, tensor<4096xbf16> -> tensor<1x2048x4096xbf16>
  %1 = tensixflow.matmul %0, %w1 : tensor<1x2048x4096xbf16> x tensor<4096x12288xbf16> -> tensor<1x2048x12288xbf16>
  %2 = tensixflow.add %1, %b1 : tensor<1x2048x12288xbf16>
  %3 = tensixflow.gelu %2 : tensor<1x2048x12288xbf16>
  %4 = tensixflow.matmul %3, %w2 : tensor<1x2048x12288xbf16> x tensor<12288x4096xbf16> -> tensor<1x2048x4096xbf16>
  %5 = tensixflow.add %4, %residual : tensor<1x2048x4096xbf16>
  tensixflow.return %5 : tensor<1x2048x4096xbf16>
}
