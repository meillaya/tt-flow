// RUN: tensixflow-opt --tensixflow-plan-noc-sharding --tensixflow-allocate-circular-buffers --tensixflow-lower-to-ttkernel --tensixflow-emit-metalium %s | FileCheck %s

// Structural test for the emitted compute kernel. The compute program is the
// tile-math half of the pinned multi-core matmul example: matrix-engine
// start-up, then one output tile per loop iteration, reducing over the K tiles
// of the output tiles this core owns. The per-core tile count and K arrive as
// runtime arguments from the host. The old stub headers are gone, so the checks
// name the pinned kernel API headers and intrinsics instead.

// CHECK: // ===== file: compute.cpp =====
// CHECK: #include "api/compute/compute_kernel_hw_startup.h"
// CHECK: #include "api/compute/matmul.h"
// CHECK: #include "api/compute/tile_move_copy.h"
// CHECK: #include "hostdevcommon/kernel_structs.h"
// CHECK: void kernel_main()
// CHECK: uint32_t num_output_tiles = get_arg_val<uint32_t>(0)
// CHECK: uint32_t Kt = get_arg_val<uint32_t>(1)
// CHECK: constexpr tt::CBIndex cb_in0 = tt::CBIndex::c_0
// CHECK: constexpr tt::CBIndex cb_in1 = tt::CBIndex::c_1
// CHECK: constexpr tt::CBIndex cb_out = tt::CBIndex::c_16
// CHECK: compute_kernel_hw_startup<SrcOrder::Reverse>(cb_in0, cb_in1, cb_out)
// CHECK: matmul_init(cb_in0, cb_in1)
// CHECK: for (uint32_t i = 0; i < num_output_tiles; ++i)
// CHECK: tile_regs_acquire()
// CHECK: cb_wait_front(cb_in0, 1)
// CHECK: cb_wait_front(cb_in1, 1)
// CHECK: matmul_tiles(cb_in0, cb_in1, 0, 0, 0)
// CHECK: cb_pop_front(cb_in0, 1)
// CHECK: cb_pop_front(cb_in1, 1)
// CHECK: tile_regs_commit()
// CHECK: tile_regs_wait()
// CHECK: cb_reserve_back(cb_out, 1)
// CHECK: pack_tile(0, cb_out)
// CHECK: cb_push_back(cb_out, 1)
// CHECK: tile_regs_release()

// The pure matmul path emits no fused prologue or epilogue, and it uses no
// stub headers.
// CHECK-NOT: rmsnorm
// CHECK-NOT: gelu
// CHECK-NOT: compute_kernel_api.h
// CHECK-NOT: kernel_types.hpp

// The compute artifact ends before the reader delimiter.
// CHECK: // ===== file: reader.cpp =====
tensixflow.func @matmul(%a: tensor<64x64xbf16>, %b: tensor<64x64xbf16>) -> tensor<64x64xbf16> {
  %0 = tensixflow.matmul %a, %b : tensor<64x64xbf16> x tensor<64x64xbf16> -> tensor<64x64xbf16>
  tensixflow.return %0 : tensor<64x64xbf16>
}
