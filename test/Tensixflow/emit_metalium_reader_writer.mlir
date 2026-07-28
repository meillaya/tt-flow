// RUN: tensixflow-opt --tensixflow-plan-noc-sharding --tensixflow-allocate-circular-buffers --tensixflow-lower-to-ttkernel --tensixflow-emit-metalium %s | FileCheck %s

// Golden structural test for the reader.cpp + writer.cpp emission (todo 35). The
// full planning + lowering pipeline runs over an UNFUSED rmsnorm -> matmul pair,
// then --tensixflow-emit-metalium prints host.cpp + reader.cpp + writer.cpp to
// stdout, separated by `// ===== file: <name>.cpp =====` delimiter banners (the
// pass flushes + exits before mlir-opt would append the module IR, so FileCheck
// sees ONLY the emitted artifacts).
//
// Assert the READER data-movement sequence (cb_reserve_back -> get_write_ptr ->
// noc_async_read -> noc_async_read_barrier -> cb_push_back, with the multicast
// weight via get_noc_multicast_addr + noc_async_write) and the WRITER sequence
// (cb_wait_front -> get_read_ptr -> noc_async_write -> noc_async_write_barrier
// -> cb_pop_front). Assert NO compute math (matmul_tiles) leaks into either
// data-movement kernel -- that is compute.cpp (todo 36). The CHECK-NOTs bracket
// each artifact's head and tail.

// --- reader.cpp (BRISC data movement) --------------------------------------
// CHECK: // ===== file: reader.cpp =====
// CHECK-NOT: matmul_tiles
// CHECK: cb_reserve_back
// CHECK: get_write_ptr
// CHECK: noc_async_read
// CHECK: noc_async_read_barrier
// CHECK: cb_push_back
// CHECK: get_noc_multicast_addr
// CHECK: noc_async_write
// CHECK-NOT: matmul_tiles

// --- writer.cpp (NCRISC data movement) -------------------------------------
// CHECK: // ===== file: writer.cpp =====
// CHECK-NOT: matmul_tiles
// CHECK: cb_wait_front
// CHECK: get_read_ptr
// CHECK: noc_async_write
// CHECK: noc_async_write_barrier
// CHECK: cb_pop_front
// CHECK-NOT: matmul_tiles
tensixflow.func @rmsnorm_matmul(%x: tensor<1x2048x4096xbf16>, %gamma: tensor<4096xbf16>, %w: tensor<4096x12288xbf16>) -> tensor<1x2048x12288xbf16> {
  %0 = tensixflow.rmsnorm %x, %gamma : tensor<1x2048x4096xbf16>, tensor<4096xbf16> -> tensor<1x2048x4096xbf16>
  %1 = tensixflow.matmul %0, %w : tensor<1x2048x4096xbf16> x tensor<4096x12288xbf16> -> tensor<1x2048x12288xbf16>
  tensixflow.return %1 : tensor<1x2048x12288xbf16>
}
