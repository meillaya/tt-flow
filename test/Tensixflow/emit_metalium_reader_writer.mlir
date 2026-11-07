// RUN: tensixflow-opt --tensixflow-plan-noc-sharding --tensixflow-allocate-circular-buffers --tensixflow-lower-to-ttkernel --tensixflow-emit-metalium %s | FileCheck %s

// Structural test for the emitted data-movement kernels. The reader streams the
// activation and weight tiles into the input circular buffers with the pinned
// TensorAccessor API; the writer drains result tiles to the output DRAM buffer.
// The one-core path has no multicast, so the weight path is a plain NoC read.

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
// CHECK: constexpr uint32_t cb_id_in0 = static_cast<uint32_t>(tt::CBIndex::c_0)
// CHECK: constexpr uint32_t cb_id_in1 = static_cast<uint32_t>(tt::CBIndex::c_1)
// CHECK: constexpr auto s0_args = TensorAccessorArgs<0>()
// CHECK: const auto s0 = TensorAccessor(s0_args, src0_addr)
// CHECK: constexpr auto s1_args =
// CHECK: const auto s1 = TensorAccessor(s1_args, src1_addr)
// CHECK: cb_reserve_back(cb_id_in0, 1)
// CHECK: uint32_t l1_write_addr_in0 = get_write_ptr(cb_id_in0)
// CHECK: noc_async_read_page(a_tile_index, s0, l1_write_addr_in0)
// CHECK: noc_async_read_barrier()
// CHECK: cb_push_back(cb_id_in0, 1)
// CHECK: cb_reserve_back(cb_id_in1, 1)
// CHECK: noc_async_read_page(b_tile_index, s1, l1_write_addr_in1)
// CHECK: cb_push_back(cb_id_in1, 1)

// The writer artifact follows its delimiter.
// CHECK: // ===== file: writer.cpp =====
// CHECK: #include "api/dataflow/dataflow_api.h"
// CHECK: uint32_t dst_addr = get_arg_val<uint32_t>(0)
// CHECK: uint32_t Mt = get_arg_val<uint32_t>(1)
// CHECK: uint32_t Nt = get_arg_val<uint32_t>(2)
// CHECK: constexpr uint32_t cb_id_out0 = static_cast<uint32_t>(tt::CBIndex::c_16)
// CHECK: constexpr auto s_args = TensorAccessorArgs<0>()
// CHECK: const auto s = TensorAccessor(s_args, dst_addr)
// CHECK: cb_wait_front(cb_id_out0, 1)
// CHECK: uint32_t l1_read_addr = get_read_ptr(cb_id_out0)
// CHECK: noc_async_write_page(m * Nt + n, s, l1_read_addr)
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
