// RUN: tensixflow-opt --tensixflow-plan-noc-sharding --tensixflow-allocate-circular-buffers --tensixflow-lower-to-ttkernel --tensixflow-emit-metalium %s | FileCheck %s

// Structural test for the emitted host.cpp. The planning, allocation, lowering,
// and emission pipeline runs over a pure matmul, and the pass prints the four
// generated translation units to stdout. The emitted host program targets the
// pinned tt-metal release (deps/tt-metal.lock): a unit mesh device, a mesh
// command queue, replicated DRAM buffers, a circular-buffer pipeline, the single
// CreateKernel entry point, and the distributed mesh enqueue calls. The program
// also carries the self-check (CPU reference matmul, PCC print, machine-readable
// result line) that scripts/run_generated.sh parses.

// The legacy program-enqueue API, the removed per-role kernel creation
// overloads, the old interleaved buffer config, and stub headers never appear.
// CHECK-NOT: EnqueueProgram
// CHECK-NOT: CreateDataMovementKernel
// CHECK-NOT: CreateComputeKernel
// CHECK-NOT: InterleavedBufferConfig
// CHECK-NOT: tt_metal_stubs

// CHECK: host.cpp - TensixFlow-generated TT-Metalium host program
// CHECK: #include <tt-metalium/bfloat16.hpp>
// CHECK: #include <tt-metalium/distributed.hpp>
// CHECK: #include <tt-metalium/host_api.hpp>
// CHECK: #include <tt-metalium/tensor_accessor_args.hpp>
// CHECK: #include <tt-metalium/tilize_utils.hpp>
// CHECK: constexpr uint32_t kOutputRowTiles = 2
// CHECK: constexpr uint32_t kOutputColTiles = 2
// CHECK: constexpr uint32_t kReductionTiles = 2
// CHECK: constexpr uint32_t kCbIn0 = 0
// CHECK: constexpr uint32_t kCbIn1 = 1
// CHECK: constexpr uint32_t kCbOut = 16
// CHECK: constexpr uint32_t kTileBytes = sizeof(bfloat16) * TILE_HEIGHT * TILE_WIDTH
// CHECK: void goldenMatmul(
// CHECK: float pearsonCorrelation(
// CHECK: void runMatmul(
// CHECK: distributed::MeshCommandQueue& cq = meshDevice->mesh_command_queue()
// CHECK: distributed::MeshWorkload workload
// CHECK: distributed::MeshCoordinateRange deviceRange
// CHECK: Program program = CreateProgram(*meshDevice)
// CHECK: const CoreCoord core{0, 0}
// CHECK: distributed::DeviceLocalBufferConfig dramConfig
// CHECK: .page_size = kTileBytes
// CHECK: .buffer_type = BufferType::DRAM
// CHECK: distributed::MeshBuffer::create(
// CHECK: distributed::ReplicatedBufferConfig{.size = sizeof(bfloat16) * activation.size()}
// CHECK: tt::DataFormat kCbDataFormat = tt::DataFormat::Float16_b
// CHECK: CircularBufferConfig(kCbIn0Tiles * kTileBytes,
// CHECK: .set_page_size(kCbIn0, kTileBytes)
// CHECK: tt_metal::CreateCircularBuffer(program, core, in0Config)
// CHECK: TensorAccessorArgs(*activationBuffer).append_to(readerCompileTimeArgs)
// CHECK: "kernels/reader.cpp"
// CHECK: tt_metal::DataMovementConfig
// CHECK: DataMovementProcessor::RISCV_1
// CHECK: "kernels/writer.cpp"
// CHECK: DataMovementProcessor::RISCV_0
// CHECK: tt_metal::ComputeConfig
// CHECK: MathFidelity::HiFi4
// CHECK: "kernels/compute.cpp"
// CHECK: tt_metal::SetRuntimeArgs(program, readerKernel, core
// CHECK: tt_metal::SetRuntimeArgs(program, writerKernel, core
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
