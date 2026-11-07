// RUN: tensixflow-opt --tensixflow-plan-noc-sharding --tensixflow-allocate-circular-buffers --tensixflow-lower-to-ttkernel --tensixflow-emit-metalium %s | FileCheck %s

// Structural test for the emitted host.cpp. The planning, allocation, lowering,
// and emission pipeline runs over a pure matmul, and the pass prints the four
// generated translation units to stdout. The emitted host program targets the
// pinned tt-metal release (deps/tt-metal.lock) and realizes the plan's 8x8 core
// grid: the output tiles are split over the grid with split_work_to_cores, the
// kernels and circular buffers are created on the whole core range, and each
// core receives its contiguous output-tile range as runtime arguments. The
// program also carries the self-check (CPU reference matmul, PCC print,
// machine-readable result line) that scripts/run_generated.sh parses.

// The legacy program-enqueue API, the removed per-role kernel creation
// overloads, the old interleaved buffer config, and stub headers never appear.
// CHECK-NOT: EnqueueProgram
// CHECK-NOT: CreateDataMovementKernel
// CHECK-NOT: CreateComputeKernel
// CHECK-NOT: InterleavedBufferConfig
// CHECK-NOT: tt_metal_stubs

// CHECK: host.cpp - TensixFlow-generated TT-Metalium host program
// CHECK: Emitted: multi-core matmul on the plan's 8 x 8 grid
// CHECK: #include <tt-metalium/distributed.hpp>
// CHECK: #include <tt-metalium/host_api.hpp>
// CHECK: #include <tt-metalium/work_split.hpp>
// CHECK: constexpr uint32_t kOutputRowTiles = 2
// CHECK: constexpr uint32_t kOutputColTiles = 2
// CHECK: constexpr uint32_t kReductionTiles = 2
// CHECK: constexpr uint32_t kGridY = 8
// CHECK: constexpr uint32_t kGridX = 8
// CHECK: constexpr uint32_t kOutputTileCount = kOutputRowTiles * kOutputColTiles
// CHECK: void goldenMatmul(
// CHECK: float pearsonCorrelation(
// CHECK: void runMatmul(
// CHECK: distributed::MeshCommandQueue& cq = meshDevice->mesh_command_queue()
// CHECK: distributed::MeshWorkload workload
// CHECK: distributed::MeshCoordinateRange deviceRange
// CHECK: Program program = CreateProgram(*meshDevice)
// CHECK: const CoreCoord coreGrid{kGridX, kGridY}
// CHECK: auto [numCores, allCores, coreGroup1, coreGroup2, workPerCore1,
// CHECK: split_work_to_cores(coreGrid, kOutputTileCount)
// CHECK: distributed::DeviceLocalBufferConfig dramConfig
// CHECK: .page_size = kTileBytes
// CHECK: .buffer_type = BufferType::DRAM
// CHECK: distributed::MeshBuffer::create(
// CHECK: distributed::ReplicatedBufferConfig{.size = sizeof(bfloat16) * activation.size()}
// CHECK: tt::DataFormat kCbDataFormat = tt::DataFormat::Float16_b
// CHECK: CircularBufferConfig(kCbIn0Tiles * kTileBytes,
// CHECK: .set_page_size(kCbIn0, kTileBytes)
// CHECK: tt_metal::CreateCircularBuffer(program, allCores, in0Config)
// CHECK: tt_metal::CreateCircularBuffer(program, allCores, in1Config)
// CHECK: tt_metal::CreateCircularBuffer(program, allCores, outConfig)
// CHECK: TensorAccessorArgs(*activationBuffer).append_to(readerCompileTimeArgs)
// CHECK: TensorAccessorArgs(*weightBuffer).append_to(readerCompileTimeArgs)
// CHECK: "kernels/reader.cpp", allCores
// CHECK: DataMovementProcessor::RISCV_1
// CHECK: TensorAccessorArgs(*outputBuffer).append_to(writerCompileTimeArgs)
// CHECK: "kernels/writer.cpp", allCores
// CHECK: DataMovementProcessor::RISCV_0
// CHECK: tt_metal::ComputeConfig
// CHECK: MathFidelity::HiFi4
// CHECK: "kernels/compute.cpp", allCores, computeConfig
// CHECK: const auto workGroups = {std::make_pair(coreGroup1, workPerCore1),
// CHECK: for (const auto& [ranges, workPerCore] : workGroups)
// CHECK: for (const auto& range : ranges.ranges())
// CHECK: for (const auto& core : range)
// CHECK: {activationAddress, weightAddress, kOutputRowTiles,
// CHECK: kReductionTiles, kOutputColTiles, workOffset, workPerCore});
// CHECK: {outputAddress, workPerCore, workOffset});
// CHECK: {workPerCore, kReductionTiles});
// CHECK: workOffset += workPerCore
// CHECK: distributed::EnqueueWriteMeshBuffer(cq, activationBuffer
// CHECK: distributed::EnqueueWriteMeshBuffer(cq, weightBuffer
// CHECK: workload.add_program(deviceRange, std::move(program))
// CHECK: distributed::EnqueueMeshWorkload(cq, workload, false)
// CHECK: distributed::EnqueueReadMeshBuffer(cq, output, outputBuffer, true)
// CHECK: std::mt19937 rng(kRandomSeed)
// CHECK: goldenMatmul(activation, weight, golden)
// CHECK: tilize_nfaces(activation, kM, kK)
// CHECK: distributed::MeshDevice::create_unit_mesh(kDeviceId)
// CHECK: runMatmul(meshDevice, activation, weight, result)
// CHECK: untilize_nfaces(result, kM, kN)
// CHECK: pearsonCorrelation(golden, result)
// CHECK: TENSIXFLOW_RESULT
// CHECK: meshDevice->close()

// The legacy tokens are absent to the end of the emitted host program too.
// CHECK-NOT: EnqueueProgram
// CHECK-NOT: CreateDataMovementKernel
// CHECK-NOT: CreateComputeKernel
// CHECK-NOT: InterleavedBufferConfig
// CHECK-NOT: tt_metal_stubs
tensixflow.func @matmul(%a: tensor<64x64xbf16>, %b: tensor<64x64xbf16>) -> tensor<64x64xbf16> {
  %0 = tensixflow.matmul %a, %b : tensor<64x64xbf16> x tensor<64x64xbf16> -> tensor<64x64xbf16>
  tensixflow.return %0 : tensor<64x64xbf16>
}
