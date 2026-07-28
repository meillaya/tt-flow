// RUN: tensixflow-opt --tensixflow-plan-noc-sharding --tensixflow-allocate-circular-buffers --tensixflow-lower-to-ttkernel --tensixflow-emit-metalium %s | FileCheck %s

// Golden structural test for --tensixflow-emit-metalium (todo 34, D3/D23): the
// full planning + lowering pipeline runs over an UNFUSED rmsnorm -> matmul pair,
// then the emitter prints a host.cpp (current tt-metal API) to stdout. The pass
// flushes + exits before mlir-opt would append the module IR, so FileCheck sees
// ONLY the emitted host.cpp.
//
// Assert the CURRENT-API tokens are present (single CreateKernel + config
// variant, ComputeConfig/DataMovementConfig, tt::CBIndex circular buffers,
// InterleavedBufferConfig DRAM buffers, distributed::Enqueue* mesh enqueue) and
// the LEGACY tokens are ABSENT (EnqueueProgram, per-role CreateDataMovementKernel
// / CreateComputeKernel). The leading CHECK-NOTs cover [start -> first CHECK];
// the trailing CHECK-NOT covers [last CHECK -> end].

// CHECK-NOT: EnqueueProgram
// CHECK-NOT: CreateDataMovementKernel
// CHECK-NOT: CreateComputeKernel
// CHECK: CreateProgram()
// CHECK: ComputeConfig
// CHECK: MathFidelity::HiFi4
// CHECK: CreateKernel(program, "compute.cpp"
// CHECK: DataMovementConfig
// CHECK: CreateKernel(program, "reader.cpp"
// CHECK: CreateKernel(program, "writer.cpp"
// CHECK: CreateCircularBuffer
// CHECK: tt::CBIndex::c_0
// CHECK: tt::CBIndex::c_1
// CHECK: tt::CBIndex::c_16
// CHECK: InterleavedBufferConfig
// CHECK: BufferType::DRAM
// CHECK: SetCommonRuntimeArgs
// CHECK: SetRuntimeArgs
// CHECK: distributed::EnqueueWriteMeshBuffer
// CHECK: distributed::EnqueueMeshWorkload
// CHECK: distributed::EnqueueReadMeshBuffer
// CHECK-NOT: EnqueueProgram
// CHECK-NOT: CreateDataMovementKernel
// CHECK-NOT: CreateComputeKernel
tensixflow.func @rmsnorm_matmul(%x: tensor<1x2048x4096xbf16>, %gamma: tensor<4096xbf16>, %w: tensor<4096x12288xbf16>) -> tensor<1x2048x12288xbf16> {
  %0 = tensixflow.rmsnorm %x, %gamma : tensor<1x2048x4096xbf16>, tensor<4096xbf16> -> tensor<1x2048x4096xbf16>
  %1 = tensixflow.matmul %0, %w : tensor<1x2048x4096xbf16> x tensor<4096x12288xbf16> -> tensor<1x2048x12288xbf16>
  tensixflow.return %1 : tensor<1x2048x12288xbf16>
}
