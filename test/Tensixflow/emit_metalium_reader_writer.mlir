// RUN: tensixflow-opt --tensixflow-plan-noc-sharding --tensixflow-allocate-circular-buffers --tensixflow-lower-to-ttkernel --tensixflow-emit-metalium %s | FileCheck %s

// Structural test for the emitted data-movement kernels. The reader is the
// pinned multi-core partitioned reader: it walks this core's contiguous range
// of linear output-tile ids, converts each id to (out_row, out_col), and reads
// A[out_row][k] and B[k][out_col] with the TensorAccessor API. The writer is the
// pinned start-id writer: it stores each produced tile at linear id
// start_id + i. The one-core path has no multicast, so both kernels use plain
// unicast NoC reads and writes.

// The reader artifact follows its file delimiter.
// CHECK: // ===== file: reader.cpp =====
// CHECK: #include "api/dataflow/dataflow_api.h"
// CHECK: #include "hostdevcommon/kernel_structs.h"
// CHECK: void kernel_main()
// CHECK: uint32_t src0_addr = get_arg_val<uint32_t>(0)
// CHECK: uint32_t src1_addr = get_arg_val<uint32_t>(1)
// CHECK: uint32_t Mt = get_arg_val<uint32_t>(2)
// CHECK: uint32_t Kt = get_arg_val<uint32_t>(3)
// CHECK: uint32_t Nt = get_arg_val<uint32_t>(4)
// CHECK: uint32_t output_tile_start_id = get_arg_val<uint32_t>(5)
// CHECK: uint32_t num_output_tiles = get_arg_val<uint32_t>(6)
// CHECK: constexpr uint32_t cb_id_in0 = static_cast<uint32_t>(tt::CBIndex::c_0)
// CHECK: constexpr uint32_t cb_id_in1 = static_cast<uint32_t>(tt::CBIndex::c_1)
// CHECK: constexpr auto a_args = TensorAccessorArgs<0>()
// CHECK: const auto a = TensorAccessor(a_args, src0_addr)
// CHECK: constexpr auto b_args =
// CHECK: const auto b = TensorAccessor(b_args, src1_addr)
// CHECK: for (uint32_t output_tile = 0; output_tile < num_output_tiles;
// CHECK: uint32_t current_tile_id = output_tile_start_id + output_tile
// CHECK: uint32_t out_row = current_tile_id / Nt
// CHECK: uint32_t out_col = current_tile_id % Nt
// CHECK: uint32_t tile_a = out_row * Kt + k
// CHECK: cb_reserve_back(cb_id_in0, 1)
// CHECK: uint32_t l1_write_addr_in0 = get_write_ptr(cb_id_in0)
// CHECK: noc_async_read_page(tile_a, a, l1_write_addr_in0)
// CHECK: noc_async_read_barrier()
// CHECK: cb_push_back(cb_id_in0, 1)
// CHECK: uint32_t tile_b = k * Nt + out_col
// CHECK: cb_reserve_back(cb_id_in1, 1)
// CHECK: uint32_t l1_write_addr_in1 = get_write_ptr(cb_id_in1)
// CHECK: noc_async_read_page(tile_b, b, l1_write_addr_in1)
// CHECK: cb_push_back(cb_id_in1, 1)

// The writer artifact follows its delimiter.
// CHECK: // ===== file: writer.cpp =====
// CHECK: #include "api/dataflow/dataflow_api.h"
// CHECK: uint32_t dst_addr = get_arg_val<uint32_t>(0)
// CHECK: uint32_t num_tiles = get_arg_val<uint32_t>(1)
// CHECK: uint32_t start_id = get_arg_val<uint32_t>(2)
// CHECK: constexpr uint32_t cb_id_out0 = static_cast<uint32_t>(tt::CBIndex::c_16)
// CHECK: constexpr auto c_args = TensorAccessorArgs<0>()
// CHECK: const auto c = TensorAccessor(c_args, dst_addr)
// CHECK: uint32_t end_id = start_id + num_tiles
// CHECK: for (uint32_t i = start_id; i < end_id; ++i)
// CHECK: cb_wait_front(cb_id_out0, 1)
// CHECK: uint32_t l1_read_addr = get_read_ptr(cb_id_out0)
// CHECK: noc_async_write_page(i, c, l1_read_addr)
// CHECK: noc_async_write_barrier()
// CHECK: cb_pop_front(cb_id_out0, 1)

// The data-movement kernels never multicast and never compute.
// CHECK-NOT: noc_async_write_multicast
// CHECK-NOT: get_noc_multicast_addr
// CHECK-NOT: matmul_tiles
// CHECK-NOT: tt_metal_stubs
tensixflow.func @matmul(%a: tensor<64x64xbf16>, %b: tensor<64x64xbf16>) -> tensor<64x64xbf16> {
  %0 = tensixflow.matmul %a, %b : tensor<64x64xbf16> x tensor<64x64xbf16> -> tensor<64x64xbf16>
  tensixflow.return %0 : tensor<64x64xbf16>
}
