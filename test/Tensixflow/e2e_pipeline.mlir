// RUN: split-file %s %t
// RUN: tensixflow-opt --tensixflow-infer-tile-layout --tensixflow-canonicalize %t/layout.mlir | FileCheck %t/layout.mlir
// RUN: tensixflow-opt --tensixflow-plan-noc-sharding --tensixflow-allocate-circular-buffers --tensixflow-lower-to-ttkernel --tensixflow-emit-metalium %t/matmul.mlir | FileCheck %t/matmul.mlir --check-prefix=EMIT

// End-to-end pipeline from typed graph to emitted TT-Metalium source.
//
// Part 1: --tensixflow-infer-tile-layout attaches the DRAM layouts to the func
// arguments and the L1 layout to the intermediate result, then the greedy
// canonicalizer renumbers the SSA values. No fusion runs in this part, so the
// rmsnorm and matmul stay separate.
//
// Part 2: over a pure matmul the planning pipeline runs through to emission.
// The emitted host program realizes the plan's core grid, assigns each core its
// 2D output-tile block for the blocked-reuse mapping, multicasts the weight
// column block down the used row blocks under a semaphore handshake, uses the
// pinned mesh API, and carries the self-check result line. A fused graph is rejected by the
// emitter, so the matmul is the supported path.

//--- layout.mlir
// CHECK-LABEL: tensixflow.func @e2e
// CHECK-SAME: %[[X:[a-z0-9]+]]: tensor<1x2048x4096xbf16> {tensixflow.layout = #tensixflow.tile_layout<tile 32x32, grid 8x8, mem DRAM>}
// CHECK-SAME: %[[W:[a-z0-9]+]]: tensor<4096x12288xbf16> {tensixflow.layout = #tensixflow.tile_layout<tile 32x32, grid 8x8, mem DRAM>}
// CHECK-SAME: %[[GAMMA:[a-z0-9]+]]: tensor<4096xbf16> {tensixflow.layout = #tensixflow.tile_layout<tile 32x32, grid 8x8, mem DRAM>}
// CHECK: %[[N:[a-z0-9]+]] = tensixflow.rmsnorm %[[X]], %[[GAMMA]] {tensixflow.layout = #tensixflow.tile_layout<tile 32x32, grid 8x8, mem L1>}
// CHECK: %[[M:[a-z0-9]+]] = tensixflow.matmul %[[N]], %[[W]] {tensixflow.layout = #tensixflow.tile_layout<tile 32x32, grid 8x8, mem L1>}
// CHECK: tensixflow.return %[[M]]
tensixflow.func @e2e(%x: tensor<1x2048x4096xbf16>, %w: tensor<4096x12288xbf16>, %gamma: tensor<4096xbf16>) -> tensor<1x2048x12288xbf16> {
  %0 = tensixflow.rmsnorm %x, %gamma : tensor<1x2048x4096xbf16>, tensor<4096xbf16> -> tensor<1x2048x4096xbf16>
  %1 = tensixflow.matmul %0, %w : tensor<1x2048x4096xbf16> x tensor<4096x12288xbf16> -> tensor<1x2048x12288xbf16>
  tensixflow.return %1 : tensor<1x2048x12288xbf16>
}

//--- matmul.mlir
// EMIT: host.cpp - TensixFlow-generated TT-Metalium host program
// EMIT: Mapping: blocked-reuse
// EMIT: distributed::MeshCommandQueue& cq = meshDevice->mesh_command_queue()
// EMIT: Program program = CreateProgram(*meshDevice)
// EMIT: const CoreRangeSet allCores = CoreRangeSet(
// EMIT: CoreRange(CoreCoord{0, 0}, CoreCoord{kUsedCols - 1, kUsedRows - 1})
// EMIT: tt_metal::CreateSemaphore(program, allCores, INVALID)
// EMIT: distributed::MeshBuffer::create(
// EMIT: CircularBufferConfig(kCbIn0Tiles * kTileBytes,
// EMIT: tt_metal::CreateCircularBuffer(program, allCores, in0Config)
// EMIT: "kernels/reader_sender.cpp", topRow
// EMIT: "kernels/reader_receiver.cpp", receiverRows
// EMIT: "kernels/writer.cpp", allCores
// EMIT: "kernels/compute.cpp", allCores, computeConfig
// EMIT: for (uint32_t y = 0; y < kUsedRows; ++y)
// EMIT: workload.add_program(deviceRange, std::move(program))
// EMIT: distributed::EnqueueMeshWorkload(cq, workload, false)
// EMIT: distributed::MeshDevice::create_unit_mesh(kDeviceId)
// EMIT: TENSIXFLOW_RESULT
// EMIT: // ===== file: compute.cpp =====
// EMIT: cb_wait_front(cb_in1, Kt * colCount)
// EMIT: matmul_tiles(cb_in0, cb_in1, mt * Kt + kt,
// EMIT: // ===== file: reader_sender.cpp =====
// EMIT: noc_async_write_multicast(weightWriteAddr, weightMulticastAddr,
// EMIT: // ===== file: reader_receiver.cpp =====
// EMIT: noc_semaphore_wait(weightReceiverSemaphore, VALID)
// EMIT: // ===== file: writer.cpp =====
// EMIT: noc_async_write_page(outputTile, c, get_read_ptr(cb_id_out0))
tensixflow.func @matmul(%a: tensor<64x64xbf16>, %b: tensor<64x64xbf16>) -> tensor<64x64xbf16> {
  %0 = tensixflow.matmul %a, %b : tensor<64x64xbf16> x tensor<64x64xbf16> -> tensor<64x64xbf16>
  tensixflow.return %0 : tensor<64x64xbf16>
}
