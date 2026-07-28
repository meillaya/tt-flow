// RUN: tensixflow-opt --tensixflow-plan-noc-sharding --tensixflow-allocate-circular-buffers --tensixflow-lower-to-ttkernel --tensixflow-emit-metalium %s | FileCheck %s

// Golden structural test for the compute.cpp emission (todo 36). The full
// planning + lowering pipeline runs over an UNFUSED rmsnorm -> matmul pair
// (planned as fused_rmsnorm_matmul), then --tensixflow-emit-metalium prints
// host.cpp + compute.cpp + reader.cpp + writer.cpp to stdout, separated by
// `// ===== file: <name>.cpp =====` delimiter banners (the pass flushes + exits
// before mlir-opt would append the module IR, so FileCheck sees ONLY the
// emitted artifacts). compute.cpp is emitted BETWEEN host and reader, so the
// compute banner precedes the reader banner.
//
// Assert the COMPUTE tile-math sequence (matmul_init -> tile_regs_acquire ->
// inner-K cb_wait_front -> matmul_tiles -> cb_pop_front -> tile_regs_commit ->
// cb_reserve_back -> tile_regs_wait -> pack_tile -> tile_regs_release ->
// cb_push_back) and, because the example is fused, the RMSNorm prologue's
// reduce-square-sum / rsqrt / scale-by-gamma markers + the gamma-tile circular-
// buffer staging. Assert NO data movement (noc_async_read / noc_async_write)
// leaks into the compute kernel -- that is the reader/writer. The CHECK-NOTs
// bracket the compute artifact's head and tail; the trailing NOT-region ends at
// the reader banner (reader.cpp legitimately contains noc_async_read).

// --- compute.cpp (TRISC tile math) ------------------------------------------
// CHECK: // ===== file: compute.cpp =====
// CHECK-NOT: noc_async_read
// CHECK-NOT: noc_async_write
// CHECK: matmul_init
// CHECK: tile_regs_acquire
// CHECK: cb_wait_front(in0_cb, 1)
// CHECK: cb_wait_front(in1_cb, 1)
// CHECK: RMSNorm prologue
// CHECK: reduce square-sum
// CHECK: rsqrt
// CHECK: scale by gamma
// CHECK: cb_wait_front(gamma_cb, 1)
// CHECK: cb_pop_front(gamma_cb, 1)
// CHECK: matmul_tiles(in0_cb, in1_cb
// CHECK: cb_pop_front(in0_cb, 1)
// CHECK: cb_pop_front(in1_cb, 1)
// CHECK: tile_regs_commit
// CHECK: cb_reserve_back(out_cb, 1)
// CHECK: tile_regs_wait
// CHECK: pack_tile
// CHECK: tile_regs_release
// CHECK: cb_push_back(out_cb, 1)
// CHECK-NOT: noc_async_read
// CHECK-NOT: noc_async_write
// CHECK: // ===== file: reader.cpp =====
tensixflow.func @rmsnorm_matmul(%x: tensor<1x2048x4096xbf16>, %gamma: tensor<4096xbf16>, %w: tensor<4096x12288xbf16>) -> tensor<1x2048x12288xbf16> {
  %0 = tensixflow.rmsnorm %x, %gamma : tensor<1x2048x4096xbf16>, tensor<4096xbf16> -> tensor<1x2048x4096xbf16>
  %1 = tensixflow.matmul %0, %w : tensor<1x2048x4096xbf16> x tensor<4096x12288xbf16> -> tensor<1x2048x12288xbf16>
  tensixflow.return %1 : tensor<1x2048x12288xbf16>
}
