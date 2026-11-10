// RUN: split-file %s %t
// RUN: tensixflow-opt --tensixflow-device=n150 --tensixflow-plan-noc-sharding --tensixflow-allocate-circular-buffers --tensixflow-lower-to-ttkernel --tensixflow-emit-metalium %t/small.mlir | FileCheck %t/small.mlir
// RUN: not tensixflow-opt --tensixflow-device=n150 --tensixflow-plan-noc-sharding --tensixflow-allocate-circular-buffers --tensixflow-lower-to-ttkernel --tensixflow-emit-metalium %t/large.mlir 2>&1 | FileCheck %t/large.mlir --check-prefix=INFEASIBLE
// RUN: not tensixflow-opt --tensixflow-device=n150 --tensixflow-plan-noc-sharding --tensixflow-allocate-circular-buffers --tensixflow-lower-to-ttkernel --tensixflow-emit-metalium %t/dest_bound.mlir 2>&1 | FileCheck %t/dest_bound.mlir --check-prefix=DEST

// Structural test for the emitted fused RMSNorm + MatMul project.
//
// Small graph: M=128, K=128, N=256 over the n150 grid. The planner fuses the
// single-use rmsnorm into the matmul and records the emitted-path estimate, then
// the emitter allocates the fused six-buffer layout (activation stage c_0,
// weight c_1, gamma c_2, scaler c_3, normalized activation c_4, output c_16).
// The generated compute kernel emits the normalization op sequence in front of
// the matmul; the emitted project is built and run under ttsim by
// scripts/test.sh, which requires the pass status.
//
// Destination-register graph: M=128, K=128, N=320 (Mt=4, Nt=10, Kt=4). The
// staging fits the n150 budget (27 of 700 tiles), so the plan gate passes and
// the emitter must reject on the eight-tile FP32 destination-register bound.
//
// Large graph: M=2048, K=4096, N=12288. The fused staging does not fit the
// n150's per-core circular-buffer budget, so the emitter refuses with a
// diagnostic that prints the staging counts and the available tiles and names
// the matmul-only and plan/report alternatives. The check keys on the phrases
// the diagnostic shares with the allocation gate, so it holds whichever gate
// rejects first.

//--- small.mlir
// CHECK: constexpr uint32_t kCbNorm = 4;
// CHECK: constexpr uint32_t kCbActTiles = 8;
// CHECK: std::vector<bfloat16> packGammaTiles(const std::vector<float>& values)
// CHECK: split_work_to_cores(coreGrid, kOutputTileCount)
// CHECK: .fp32_dest_acc_en = true,
// CHECK: // ===== file: compute.cpp =====
// CHECK: constexpr uint32_t kKt = 4;
// CHECK: constexpr tt::CBIndex cb_gamma = tt::CBIndex::c_2
// CHECK: constexpr tt::CBIndex cb_scaler = tt::CBIndex::c_3
// CHECK: constexpr tt::CBIndex cb_norm = tt::CBIndex::c_4
// CHECK: constexpr tt::CBIndex cb_out = tt::CBIndex::c_16
// CHECK: compute_kernel_hw_startup<SrcOrder::Reverse>(cb_act, cb_weight, cb_out)
// CHECK: reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_norm, cb_scaler, cb_norm)
// CHECK: mul_unary_tile(0, kInverseKBits)
// CHECK: add_unary_tile(0, kEpsilonBits)
// CHECK: rsqrt_tile(0)
// CHECK: mul_tiles_bcast_cols(cb_act, cb_norm, kt, 0, 0)
// CHECK: mul_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_gamma, kt, 0)
// CHECK: matmul_tiles(cb_norm, cb_weight, 0, 0, col - firstCol)
// CHECK: // ===== file: reader.cpp =====
// CHECK: noc_async_read_page(row * Kt + kt, act, get_write_ptr(cb_act))
// CHECK: noc_async_read_page(kt, gamma, get_write_ptr(cb_gamma))
// CHECK: noc_async_read_page(kt * Nt + col, weight, get_write_ptr(cb_weight))
// CHECK: // ===== file: writer.cpp =====
// CHECK: constexpr uint32_t cb_id_out0 = static_cast<uint32_t>(tt::CBIndex::c_16);
tensixflow.func @rmsnorm_matmul_small(%x: tensor<128x128xbf16>, %gamma: tensor<128xbf16>, %w: tensor<128x256xbf16>) -> tensor<128x256xbf16> {
  %0 = tensixflow.rmsnorm %x, %gamma : tensor<128x128xbf16>, tensor<128xbf16> -> tensor<128x128xbf16>
  %1 = tensixflow.matmul %0, %w : tensor<128x128xbf16> x tensor<128x256xbf16> -> tensor<128x256xbf16>
  tensixflow.return %1 : tensor<128x256xbf16>
}

//--- large.mlir
// INFEASIBLE: error: the fused plan 'fused_rmsnorm_matmul'
// INFEASIBLE: circular-buffer tiles
// INFEASIBLE: available tiles per core on 'n150'
// INFEASIBLE: Emit the matmul-only path from examples/matmul.mlir
// INFEASIBLE: tensixflow-report
tensixflow.func @rmsnorm_matmul(%x: tensor<1x2048x4096xbf16>, %gamma: tensor<4096xbf16>, %w: tensor<4096x12288xbf16>) -> tensor<1x2048x12288xbf16> {
  %0 = tensixflow.rmsnorm %x, %gamma : tensor<1x2048x4096xbf16>, tensor<4096xbf16> -> tensor<1x2048x4096xbf16>
  %1 = tensixflow.matmul %0, %w : tensor<1x2048x4096xbf16> x tensor<4096x12288xbf16> -> tensor<1x2048x12288xbf16>
  tensixflow.return %1 : tensor<1x2048x12288xbf16>
}

//--- dest_bound.mlir
// DEST: error: the fused plan 'fused_rmsnorm_matmul' has 10 output column tiles
// DEST: FP32 accumulation leaves 8
tensixflow.func @rmsnorm_matmul_dest_bound(%x: tensor<128x128xbf16>, %gamma: tensor<128xbf16>, %w: tensor<128x320xbf16>) -> tensor<128x320xbf16> {
  %0 = tensixflow.rmsnorm %x, %gamma : tensor<128x128xbf16>, tensor<128xbf16> -> tensor<128x128xbf16>
  %1 = tensixflow.matmul %0, %w : tensor<128x128xbf16> x tensor<128x320xbf16> -> tensor<128x320xbf16>
  tensixflow.return %1 : tensor<128x320xbf16>
}
