// RUN: split-file %s %t
// RUN: tensixflow-opt --tensixflow-plan-noc-sharding --tensixflow-allocate-circular-buffers --tensixflow-lower-to-ttkernel --tensixflow-emit-metalium %t/blocked.mlir | FileCheck %s --check-prefix=BLOCKED
// RUN: tensixflow-opt --tensixflow-plan-noc-sharding --tensixflow-allocate-circular-buffers --tensixflow-lower-to-ttkernel --tensixflow-emit-metalium %t/ranges.mlir | FileCheck %s --check-prefix=RANGES

// Structural test for the emitted compute kernel. The blocked-reuse module is a
// 2 x 2 output-tile matmul over the 8 x 8 plan grid, so each core owns a 1 x 1
// tile block: the kernel waits once for the resident activation row block and
// weight column block, then accumulates each output tile of the block from those
// buffers. The output-tile-ranges module is a 64 x 384 output-tile matmul whose
// resident blocks do not fit the budget, so its kernel keeps the streaming
// one-output-tile-per-iteration loop over the per-core tile count.
//
// The old stub headers are gone, so the checks name the pinned kernel API
// headers and intrinsics instead.

// BLOCKED: // ===== file: compute.cpp =====
// BLOCKED: Role: compute (tile math), blocked-reuse mapping.
// BLOCKED: #include "api/compute/compute_kernel_hw_startup.h"
// BLOCKED: #include "api/compute/matmul.h"
// BLOCKED: #include "api/compute/tile_move_copy.h"
// BLOCKED: #include "hostdevcommon/kernel_structs.h"
// BLOCKED: void kernel_main()
// BLOCKED: const uint32_t rowCount = get_arg_val<uint32_t>(0)
// BLOCKED: const uint32_t colCount = get_arg_val<uint32_t>(1)
// BLOCKED: const uint32_t Kt = get_arg_val<uint32_t>(2)
// BLOCKED: constexpr tt::CBIndex cb_in0 = tt::CBIndex::c_0
// BLOCKED: constexpr tt::CBIndex cb_in1 = tt::CBIndex::c_1
// BLOCKED: constexpr tt::CBIndex cb_out = tt::CBIndex::c_16
// BLOCKED: compute_kernel_hw_startup<SrcOrder::Reverse>(cb_in0, cb_in1, cb_out)
// BLOCKED: matmul_init(cb_in0, cb_in1)
// BLOCKED: cb_wait_front(cb_in0, rowCount * Kt)
// BLOCKED: cb_wait_front(cb_in1, Kt * colCount)
// BLOCKED: for (uint32_t mt = 0; mt < rowCount; ++mt)
// BLOCKED: for (uint32_t nt = 0; nt < colCount; ++nt)
// BLOCKED: tile_regs_acquire()
// BLOCKED: matmul_tiles(cb_in0, cb_in1, mt * Kt + kt,
// BLOCKED: kt * colCount + nt, 0)
// BLOCKED: tile_regs_commit()
// BLOCKED: tile_regs_wait()
// BLOCKED: cb_reserve_back(cb_out, 1)
// BLOCKED: pack_tile(0, cb_out)
// BLOCKED: cb_push_back(cb_out, 1)
// BLOCKED: tile_regs_release()
// BLOCKED: // ===== file: reader_sender.cpp =====
// BLOCKED-NOT: num_output_tiles
// BLOCKED-NOT: matmul_tiles(cb_in0, cb_in1, 0, 0, 0)

// RANGES: // ===== file: compute.cpp =====
// RANGES: Role: compute (tile math)
// RANGES: uint32_t num_output_tiles = get_arg_val<uint32_t>(0)
// RANGES: uint32_t Kt = get_arg_val<uint32_t>(1)
// RANGES: for (uint32_t i = 0; i < num_output_tiles; ++i)
// RANGES: matmul_tiles(cb_in0, cb_in1, 0, 0, 0)
// RANGES: cb_pop_front(cb_in0, 1)
// RANGES: cb_pop_front(cb_in1, 1)
// RANGES: // ===== file: reader.cpp =====
// RANGES-NOT: cb_wait_front(cb_in1, Kt * colCount)
// RANGES-NOT: rowCount

// The pure matmul path emits no fused prologue or epilogue, and it uses no
// stub headers.
// BLOCKED-NOT: rmsnorm
// BLOCKED-NOT: gelu
// BLOCKED-NOT: compute_kernel_api.h
// BLOCKED-NOT: kernel_types.hpp
// RANGES-NOT: rmsnorm
// RANGES-NOT: gelu
// RANGES-NOT: compute_kernel_api.h
// RANGES-NOT: kernel_types.hpp

//--- blocked.mlir
tensixflow.func @matmul(%a: tensor<64x64xbf16>, %b: tensor<64x64xbf16>) -> tensor<64x64xbf16> {
  %0 = tensixflow.matmul %a, %b : tensor<64x64xbf16> x tensor<64x64xbf16> -> tensor<64x64xbf16>
  tensixflow.return %0 : tensor<64x64xbf16>
}

//--- ranges.mlir
tensixflow.func @matmul(%a: tensor<2048x4096xbf16>, %b: tensor<4096x12288xbf16>) -> tensor<2048x12288xbf16> {
  %0 = tensixflow.matmul %a, %b : tensor<2048x4096xbf16> x tensor<4096x12288xbf16> -> tensor<2048x12288xbf16>
  tensixflow.return %0 : tensor<2048x12288xbf16>
}
