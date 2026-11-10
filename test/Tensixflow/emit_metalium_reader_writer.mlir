// RUN: split-file %s %t
// RUN: tensixflow-opt --tensixflow-plan-noc-sharding --tensixflow-allocate-circular-buffers --tensixflow-lower-to-ttkernel --tensixflow-emit-metalium %t/blocked.mlir | FileCheck %s --check-prefix=BLOCKED
// RUN: tensixflow-opt --tensixflow-plan-noc-sharding --tensixflow-allocate-circular-buffers --tensixflow-lower-to-ttkernel --tensixflow-emit-metalium %t/single_row.mlir | FileCheck %s --check-prefix=SINGLE
// RUN: tensixflow-opt --tensixflow-plan-noc-sharding --tensixflow-allocate-circular-buffers --tensixflow-lower-to-ttkernel --tensixflow-emit-metalium %t/ranges.mlir | FileCheck %s --check-prefix=RANGES

// Structural test for the emitted data-movement kernels.
//
// The blocked-reuse module is a 2 x 2 output-tile matmul over the 8 x 8 plan
// grid, so each core owns a 1 x 1 tile block and two used row blocks exist. The
// reader is split by role: the top row block reads its weight column block and
// multicasts it down under a semaphore handshake (reader_sender.cpp), and the
// row block below receives it (reader_receiver.cpp). Both stage their
// activation row block unicast. The writer stores the block's result tiles at
// their linear output-tile ids.
//
// The single-row module has one used row block, so there is no weight
// replication to remove: its reader is the unicast blocked-reuse reader.
//
// The output-tile-ranges module is a 64 x 384 output-tile matmul whose resident
// blocks do not fit the budget. Its reader is the pinned multi-core partitioned
// reader: it walks this core's contiguous range of linear output-tile ids,
// converts each id to (out_row, out_col), and reads A[out_row][k] and
// B[k][out_col]. Its writer is the pinned start-id writer: it stores each
// produced tile at linear id start_id + i.

// The multicast sender artifact follows its delimiter.
// BLOCKED: // ===== file: reader_sender.cpp =====
// BLOCKED: //===- kernels/reader_sender.cpp - TensixFlow-generated weight sender
// BLOCKED: Role: reader (data movement), blocked-reuse mapping, top row block.
// BLOCKED: #include "api/dataflow/dataflow_api.h"
// BLOCKED: #include "hostdevcommon/common_values.hpp"
// BLOCKED: void kernel_main()
// BLOCKED: const uint32_t src0_addr = get_arg_val<uint32_t>(0)
// BLOCKED: const uint32_t src1_addr = get_arg_val<uint32_t>(1)
// BLOCKED: const uint32_t Nt = get_arg_val<uint32_t>(2)
// BLOCKED: const uint32_t rowStart = get_arg_val<uint32_t>(3)
// BLOCKED: const uint32_t rowCount = get_arg_val<uint32_t>(4)
// BLOCKED: const uint32_t colStart = get_arg_val<uint32_t>(5)
// BLOCKED: const uint32_t colCount = get_arg_val<uint32_t>(6)
// BLOCKED: const uint32_t Kt = get_arg_val<uint32_t>(7)
// BLOCKED: const uint32_t receiverStartX = get_arg_val<uint32_t>(8)
// BLOCKED: const uint32_t receiverStartY = get_arg_val<uint32_t>(9)
// BLOCKED: const uint32_t receiverEndX = get_arg_val<uint32_t>(10)
// BLOCKED: const uint32_t receiverEndY = get_arg_val<uint32_t>(11)
// BLOCKED: const uint32_t receiverCount = get_arg_val<uint32_t>(12)
// BLOCKED: const uint32_t weightSenderSemaphoreId = get_arg_val<uint32_t>(13)
// BLOCKED: const uint32_t weightReceiverSemaphoreId = get_arg_val<uint32_t>(14)
// BLOCKED: constexpr uint32_t cb_id_in0 = static_cast<uint32_t>(tt::CBIndex::c_0)
// BLOCKED: constexpr uint32_t cb_id_in1 = static_cast<uint32_t>(tt::CBIndex::c_1)
// BLOCKED: constexpr auto a_args = TensorAccessorArgs<0>()
// BLOCKED: const auto a = TensorAccessor(a_args, src0_addr)
// BLOCKED: const auto b = TensorAccessor(b_args, src1_addr)
// BLOCKED: noc_async_read_page((rowStart + mt) * Kt + kt, a,
// BLOCKED: cb_push_back(cb_id_in0, 1)
// BLOCKED: noc_async_read_page(kt * Nt + colStart + nt, b, weightReadAddr)
// BLOCKED: noc_semaphore_wait(weightSenderSemaphore, receiverCount)
// BLOCKED: noc_semaphore_set(weightSenderSemaphore, 0)
// BLOCKED: *weightReceiverSemaphore = VALID
// BLOCKED: const uint64_t weightMulticastAddr = get_noc_multicast_addr(
// BLOCKED: receiverEndX, receiverEndY, receiverStartX, receiverStartY,
// BLOCKED: noc_async_write_multicast(weightWriteAddr, weightMulticastAddr,
// BLOCKED: weightTiles * tileBytes, receiverCount)
// BLOCKED: noc_semaphore_set_multicast(weightReceiverSemaphoreAddr,
// BLOCKED: cb_push_back(cb_id_in1, weightTiles)

// The multicast receiver artifact follows its delimiter.
// BLOCKED: // ===== file: reader_receiver.cpp =====
// BLOCKED: //===- kernels/reader_receiver.cpp - TensixFlow-generated weight receiver
// BLOCKED: Role: reader (data movement), blocked-reuse mapping, rows below the top
// BLOCKED: const uint32_t src0_addr = get_arg_val<uint32_t>(0)
// BLOCKED: const uint32_t Nt = get_arg_val<uint32_t>(2)
// BLOCKED: const uint32_t rowStart = get_arg_val<uint32_t>(3)
// BLOCKED: const uint32_t rowCount = get_arg_val<uint32_t>(4)
// BLOCKED: const uint32_t colStart = get_arg_val<uint32_t>(5)
// BLOCKED: const uint32_t colCount = get_arg_val<uint32_t>(6)
// BLOCKED: const uint32_t Kt = get_arg_val<uint32_t>(7)
// BLOCKED: const uint32_t senderX = get_arg_val<uint32_t>(8)
// BLOCKED: const uint32_t senderY = get_arg_val<uint32_t>(9)
// BLOCKED: const uint32_t weightSenderSemaphoreId = get_arg_val<uint32_t>(10)
// BLOCKED: const uint32_t weightReceiverSemaphoreId = get_arg_val<uint32_t>(11)
// BLOCKED: constexpr uint32_t cb_id_in0 = static_cast<uint32_t>(tt::CBIndex::c_0)
// BLOCKED: constexpr uint32_t cb_id_in1 = static_cast<uint32_t>(tt::CBIndex::c_1)
// BLOCKED: noc_async_read_page((rowStart + mt) * Kt + kt, a,
// BLOCKED: cb_reserve_back(cb_id_in1, weightTiles)
// BLOCKED: noc_semaphore_set(weightReceiverSemaphore, INVALID)
// BLOCKED: get_noc_addr(senderX, senderY, get_semaphore(weightSenderSemaphoreId))
// BLOCKED: noc_semaphore_inc(weightSenderSemaphoreAddr, 1)
// BLOCKED: noc_semaphore_wait(weightReceiverSemaphore, VALID)
// BLOCKED: cb_push_back(cb_id_in1, weightTiles)

// The writer artifact follows the receiver delimiter.
// BLOCKED: // ===== file: writer.cpp =====
// BLOCKED: Role: writer (data movement), blocked-reuse mapping.
// BLOCKED: #include "api/dataflow/dataflow_api.h"
// BLOCKED: const uint32_t dst_addr = get_arg_val<uint32_t>(0)
// BLOCKED: const uint32_t Nt = get_arg_val<uint32_t>(1)
// BLOCKED: const uint32_t rowStart = get_arg_val<uint32_t>(2)
// BLOCKED: const uint32_t rowCount = get_arg_val<uint32_t>(3)
// BLOCKED: const uint32_t colStart = get_arg_val<uint32_t>(4)
// BLOCKED: const uint32_t colCount = get_arg_val<uint32_t>(5)
// BLOCKED: constexpr uint32_t cb_id_out0 = static_cast<uint32_t>(tt::CBIndex::c_16)
// BLOCKED: for (uint32_t mt = 0; mt < rowCount; ++mt)
// BLOCKED: for (uint32_t nt = 0; nt < colCount; ++nt)
// BLOCKED: (rowStart + mt) * Nt + (colStart + nt)
// BLOCKED: noc_async_write_page(outputTile, c, get_read_ptr(cb_id_out0))
// BLOCKED: noc_async_write_barrier()
// BLOCKED: cb_pop_front(cb_id_out0, 1)
// BLOCKED-NOT: output_tile_start_id
// BLOCKED-NOT: num_output_tiles
// BLOCKED-NOT: noc_async_write_multicast
// BLOCKED-NOT: matmul_tiles
// BLOCKED-NOT: tt_metal_stubs

// The single-row block assignment uses the unicast blocked-reuse reader.
// SINGLE: // ===== file: reader.cpp =====
// SINGLE: Role: reader (data movement), blocked-reuse mapping.
// SINGLE: #include "api/dataflow/dataflow_api.h"
// SINGLE: void kernel_main()
// SINGLE: const uint32_t src0_addr = get_arg_val<uint32_t>(0)
// SINGLE: const uint32_t src1_addr = get_arg_val<uint32_t>(1)
// SINGLE: const uint32_t Nt = get_arg_val<uint32_t>(2)
// SINGLE: const uint32_t rowStart = get_arg_val<uint32_t>(3)
// SINGLE: const uint32_t rowCount = get_arg_val<uint32_t>(4)
// SINGLE: const uint32_t colStart = get_arg_val<uint32_t>(5)
// SINGLE: const uint32_t colCount = get_arg_val<uint32_t>(6)
// SINGLE: const uint32_t Kt = get_arg_val<uint32_t>(7)
// SINGLE: noc_async_read_page((rowStart + mt) * Kt + kt, a,
// SINGLE: noc_async_read_page(kt * Nt + colStart + nt, b,
// SINGLE: cb_push_back(cb_id_in1, 1)
// SINGLE: // ===== file: writer.cpp =====
// SINGLE: noc_async_write_page(outputTile, c, get_read_ptr(cb_id_out0))
// SINGLE-NOT: noc_async_write_multicast
// SINGLE-NOT: noc_semaphore_inc
// SINGLE-NOT: common_values.hpp

// The output-tile-ranges reader is the pinned partitioned reader and the
// writer is the pinned start-id writer.
// RANGES: // ===== file: reader.cpp =====
// RANGES: Role: reader (data movement). Each core streams the activation and
// RANGES: uint32_t src0_addr = get_arg_val<uint32_t>(0)
// RANGES: uint32_t src1_addr = get_arg_val<uint32_t>(1)
// RANGES: uint32_t Mt = get_arg_val<uint32_t>(2)
// RANGES: uint32_t Kt = get_arg_val<uint32_t>(3)
// RANGES: uint32_t Nt = get_arg_val<uint32_t>(4)
// RANGES: uint32_t output_tile_start_id = get_arg_val<uint32_t>(5)
// RANGES: uint32_t num_output_tiles = get_arg_val<uint32_t>(6)
// RANGES: for (uint32_t output_tile = 0; output_tile < num_output_tiles;
// RANGES: uint32_t current_tile_id = output_tile_start_id + output_tile
// RANGES: uint32_t out_row = current_tile_id / Nt
// RANGES: uint32_t out_col = current_tile_id % Nt
// RANGES: uint32_t tile_a = out_row * Kt + k
// RANGES: noc_async_read_page(tile_a, a, l1_write_addr_in0)
// RANGES: uint32_t tile_b = k * Nt + out_col
// RANGES: noc_async_read_page(tile_b, b, l1_write_addr_in1)

// RANGES: // ===== file: writer.cpp =====
// RANGES: uint32_t dst_addr = get_arg_val<uint32_t>(0)
// RANGES: uint32_t num_tiles = get_arg_val<uint32_t>(1)
// RANGES: uint32_t start_id = get_arg_val<uint32_t>(2)
// RANGES: constexpr uint32_t cb_id_out0 = static_cast<uint32_t>(tt::CBIndex::c_16)
// RANGES: uint32_t end_id = start_id + num_tiles
// RANGES: for (uint32_t i = start_id; i < end_id; ++i)
// RANGES: noc_async_write_page(i, c, l1_read_addr)
// RANGES-NOT: rowStart
// RANGES-NOT: colCount
// RANGES-NOT: noc_async_write_multicast

//--- blocked.mlir
tensixflow.func @matmul(%a: tensor<64x64xbf16>, %b: tensor<64x64xbf16>) -> tensor<64x64xbf16> {
  %0 = tensixflow.matmul %a, %b : tensor<64x64xbf16> x tensor<64x64xbf16> -> tensor<64x64xbf16>
  tensixflow.return %0 : tensor<64x64xbf16>
}

//--- single_row.mlir
tensixflow.func @matmul(%a: tensor<32x64xbf16>, %b: tensor<64x64xbf16>) -> tensor<32x64xbf16> {
  %0 = tensixflow.matmul %a, %b : tensor<32x64xbf16> x tensor<64x64xbf16> -> tensor<32x64xbf16>
  tensixflow.return %0 : tensor<32x64xbf16>
}

//--- ranges.mlir
tensixflow.func @matmul(%a: tensor<2048x4096xbf16>, %b: tensor<4096x12288xbf16>) -> tensor<2048x12288xbf16> {
  %0 = tensixflow.matmul %a, %b : tensor<2048x4096xbf16> x tensor<4096x12288xbf16> -> tensor<2048x12288xbf16>
  tensixflow.return %0 : tensor<2048x12288xbf16>
}
