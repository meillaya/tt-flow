// RUN: split-file %s %t
// RUN: tensixflow-opt --tensixflow-plan-noc-sharding --tensixflow-allocate-circular-buffers --tensixflow-lower-to-ttkernel --tensixflow-emit-metalium %t/blocked.mlir | FileCheck %s --check-prefix=BLOCKED
// RUN: tensixflow-opt --tensixflow-plan-noc-sharding --tensixflow-allocate-circular-buffers --tensixflow-lower-to-ttkernel --tensixflow-emit-metalium %t/ranges.mlir | FileCheck %s --check-prefix=RANGES
// RUN: tensixflow-opt --tensixflow-plan-noc-sharding --tensixflow-allocate-circular-buffers --tensixflow-lower-to-ttkernel --tensixflow-emit-metalium %t/single_row.mlir | FileCheck %s --check-prefix=SINGLE

// Structural test for the emitted host.cpp. The planning, allocation, lowering,
// and emission pipeline runs over a pure matmul and the pass prints the four
// generated translation units to stdout.
//
// The blocked-reuse module is a 2 x 2 output-tile matmul over the 8 x 8 plan
// grid: the resident buffers need (1 + 1) * 2 + 2 = 6 of the 700 budget tiles,
// so the selector picks blocked-reuse. The host assigns each core a 1 x 1 block
// of output tiles and passes the block in tile coordinates as runtime
// arguments. Two used row blocks exist, so the top row block multicasts each
// weight column block down to the row block below it under a semaphore
// handshake, and the host creates a sender and a receiver reader kernel.
//
// The single-row module has the same shape of block assignment but only one used
// row block, so the weight is read directly and the reader stays unicast.
//
// The output-tile-ranges module is a 64 x 384 output-tile matmul with 128
// reduction tiles: its resident buffers would need (8 + 48) * 128 + 2 = 7170
// tiles, so the selector keeps the contiguous-range mapping with
// split_work_to_cores and per-core tile ranges.
//
// Both host programs target the pinned tt-metal release (deps/tt-metal.lock),
// fill the declared shapes, zero-pad them to whole tiles, run the golden matmul
// over the same padded operands, and carry the self-check (CPU reference
// matmul, PCC print, machine-readable result line) that
// scripts/run_generated.sh parses.

// The legacy program-enqueue API, the removed per-role kernel creation
// overloads, the old interleaved buffer config, and stub headers never appear.
// BLOCKED-NOT: EnqueueProgram
// BLOCKED-NOT: CreateDataMovementKernel
// BLOCKED-NOT: CreateComputeKernel
// BLOCKED-NOT: InterleavedBufferConfig
// BLOCKED-NOT: tt_metal_stubs

// BLOCKED: host.cpp - TensixFlow-generated TT-Metalium host program
// BLOCKED: Emitted: multi-core matmul on the plan's 8 x 8 grid. Output tiles 2 x 2, reduction tiles 2.
// BLOCKED: Mapping: blocked-reuse. Each core owns a 1 x 1 tile block of the output.
// BLOCKED: #include <algorithm>
// BLOCKED: #include <tt-metalium/distributed.hpp>
// BLOCKED: #include <tt-metalium/host_api.hpp>
// BLOCKED: constexpr uint32_t kOutputRowTiles = 2
// BLOCKED: constexpr uint32_t kOutputColTiles = 2
// BLOCKED: constexpr uint32_t kReductionTiles = 2
// BLOCKED: constexpr uint32_t kLogicalM = 64
// BLOCKED: constexpr uint32_t kLogicalN = 64
// BLOCKED: constexpr uint32_t kLogicalK = 64
// BLOCKED: constexpr uint32_t kGridY = 8
// BLOCKED: constexpr uint32_t kGridX = 8
// BLOCKED: constexpr uint32_t kCbIn0 = 0
// BLOCKED: constexpr uint32_t kCbIn1 = 1
// BLOCKED: constexpr uint32_t kCbOut = 16
// BLOCKED: constexpr uint32_t kCbIn0Tiles = 2
// BLOCKED: constexpr uint32_t kCbIn1Tiles = 2
// BLOCKED: constexpr uint32_t kCbOutTiles = 2
// BLOCKED: constexpr uint32_t kRowsPerCore = 1
// BLOCKED: constexpr uint32_t kColsPerCore = 1
// BLOCKED: constexpr uint32_t kUsedRows = 2
// BLOCKED: constexpr uint32_t kUsedCols = 2
// BLOCKED: void goldenMatmul(
// BLOCKED: float pearsonCorrelation(
// BLOCKED: void runMatmul(
// BLOCKED-NOT: split_work_to_cores
// BLOCKED: const CoreRangeSet allCores = CoreRangeSet(
// BLOCKED: CoreRange(CoreCoord{0, 0}, CoreCoord{kUsedCols - 1, kUsedRows - 1})
// BLOCKED: const CoreRangeSet topRow = CoreRangeSet(
// BLOCKED: CoreRange(CoreCoord{0, 0}, CoreCoord{kUsedCols - 1, 0})
// BLOCKED: const CoreRangeSet receiverRows = CoreRangeSet(
// BLOCKED: CoreRange(CoreCoord{0, 1}, CoreCoord{kUsedCols - 1, kUsedRows - 1})
// BLOCKED: tt_metal::CreateSemaphore(program, allCores, INVALID)
// BLOCKED: distributed::DeviceLocalBufferConfig dramConfig
// BLOCKED: .page_size = kTileBytes
// BLOCKED: distributed::MeshBuffer::create(
// BLOCKED: CircularBufferConfig(kCbIn0Tiles * kTileBytes,
// BLOCKED: tt_metal::CreateCircularBuffer(program, allCores, in0Config)
// BLOCKED: tt_metal::CreateCircularBuffer(program, allCores, in1Config)
// BLOCKED: tt_metal::CreateCircularBuffer(program, allCores, outConfig)
// BLOCKED: TensorAccessorArgs(*activationBuffer).append_to(readerCompileTimeArgs)
// BLOCKED: TensorAccessorArgs(*weightBuffer).append_to(readerCompileTimeArgs)
// BLOCKED: "kernels/reader_sender.cpp", topRow
// BLOCKED: "kernels/reader_receiver.cpp", receiverRows
// BLOCKED: DataMovementProcessor::RISCV_1
// BLOCKED: "kernels/writer.cpp", allCores
// BLOCKED: DataMovementProcessor::RISCV_0
// BLOCKED: MathFidelity::HiFi4
// BLOCKED: "kernels/compute.cpp", allCores, computeConfig
// BLOCKED: for (uint32_t y = 0; y < kUsedRows; ++y)
// BLOCKED: const uint32_t rowStart = y * kRowsPerCore
// BLOCKED: for (uint32_t x = 0; x < kUsedCols; ++x)
// BLOCKED: const CoreCoord core{x, y}
// BLOCKED: {outputAddress, kOutputColTiles, rowStart, rowCount, colStart,
// BLOCKED: {rowCount, colCount, kReductionTiles});
// BLOCKED: meshDevice->worker_core_from_logical_core(CoreCoord{x, 1})
// BLOCKED: static_cast<uint32_t>(lastReceiver.y), kUsedRows - 1,
// BLOCKED: meshDevice->worker_core_from_logical_core(CoreCoord{x, 0})
// BLOCKED: static_cast<uint32_t>(sender.y), weightSenderSemaphore,
// BLOCKED: distributed::EnqueueWriteMeshBuffer(cq, activationBuffer
// BLOCKED: workload.add_program(deviceRange, std::move(program))
// BLOCKED: distributed::EnqueueMeshWorkload(cq, workload, false)
// BLOCKED: distributed::EnqueueReadMeshBuffer(cq, output, outputBuffer, true)
// BLOCKED: std::vector<bfloat16> activation(kLogicalM * kLogicalK)
// BLOCKED: std::vector<bfloat16> weight(kLogicalK * kLogicalN)
// BLOCKED: std::vector<bfloat16> activationPadded(kM * kK, bfloat16(0.0f))
// BLOCKED: activationPadded[row * kK + col] = activation[row * kLogicalK + col]
// BLOCKED: std::vector<bfloat16> weightPadded(kK * kN, bfloat16(0.0f))
// BLOCKED: weightPadded[row * kN + col] = weight[row * kLogicalN + col]
// BLOCKED: goldenMatmul(activationPadded, weightPadded, golden)
// BLOCKED: tilize_nfaces(activationPadded, kM, kK)
// BLOCKED: tilize_nfaces(weightPadded, kK, kN)
// BLOCKED: distributed::MeshDevice::create_unit_mesh(kDeviceId)
// BLOCKED: runMatmul(meshDevice, activationPadded, weightPadded, result)
// BLOCKED: untilize_nfaces(result, kM, kN)
// BLOCKED: pearsonCorrelation(golden, result)
// BLOCKED: TENSIXFLOW_RESULT
// BLOCKED: meshDevice->close()

// BLOCKED-NOT: EnqueueProgram
// BLOCKED-NOT: "kernels/reader.cpp"
// BLOCKED-NOT: CreateDataMovementKernel
// BLOCKED-NOT: CreateComputeKernel
// BLOCKED-NOT: InterleavedBufferConfig
// BLOCKED-NOT: tt_metal_stubs

// The output-tile-ranges path keeps the pinned work_split helper and the
// per-core contiguous linear output-tile ranges.
// RANGES: host.cpp - TensixFlow-generated TT-Metalium host program
// RANGES: Emitted: multi-core matmul on the plan's 8 x 8 grid. Output tiles 64 x 384, reduction tiles 128.
// RANGES: Mapping: output-tile-ranges. Each core owns a contiguous range of
// RANGES: constexpr uint32_t kCbIn0Tiles = 2
// RANGES: constexpr uint32_t kCbIn1Tiles = 2
// RANGES: constexpr uint32_t kCbOutTiles = 2
// RANGES: const CoreCoord coreGrid{kGridX, kGridY}
// RANGES: split_work_to_cores(coreGrid, kOutputTileCount)
// RANGES: const auto workGroups = {std::make_pair(coreGroup1, workPerCore1),
// RANGES: {activationAddress, weightAddress, kOutputRowTiles,
// RANGES: {outputAddress, workPerCore, workOffset});
// RANGES: {workPerCore, kReductionTiles});
// RANGES: workOffset += workPerCore
// RANGES: goldenMatmul(activationPadded, weightPadded, golden)
// RANGES: TENSIXFLOW_RESULT
// RANGES-NOT: kRowsPerCore

// The single-row-block geometry keeps the unicast blocked-reuse reader tested:
// one used row block has no weight replication, so no multicast is emitted.
// SINGLE: host.cpp - TensixFlow-generated TT-Metalium host program
// SINGLE: Mapping: blocked-reuse. Each core owns a 1 x 1 tile block of the output.
// SINGLE: constexpr uint32_t kRowsPerCore = 1
// SINGLE: constexpr uint32_t kColsPerCore = 1
// SINGLE: const CoreRangeSet allCores = CoreRangeSet(
// SINGLE: CoreRange(CoreCoord{0, 0}, CoreCoord{kGridX - 1, kGridY - 1})
// SINGLE: "kernels/reader.cpp", allCores
// SINGLE: for (uint32_t y = 0; y < kGridY; ++y)
// SINGLE: {activationAddress, weightAddress, kOutputColTiles, rowStart,
// SINGLE: TENSIXFLOW_RESULT
// SINGLE-NOT: CreateSemaphore
// SINGLE-NOT: kUsedRows
// SINGLE-NOT: reader_sender

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

//--- single_row.mlir
tensixflow.func @matmul(%a: tensor<32x64xbf16>, %b: tensor<64x64xbf16>) -> tensor<32x64xbf16> {
  %0 = tensixflow.matmul %a, %b : tensor<32x64xbf16> x tensor<64x64xbf16> -> tensor<32x64xbf16>
  tensixflow.return %0 : tensor<32x64xbf16>
}
