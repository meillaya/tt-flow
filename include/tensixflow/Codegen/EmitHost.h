//===- Codegen/EmitHost.h - Emit a tt-metal host.cpp ----------*- C++ -*-===//
//
// TensixFlow: An MLIR Compiler for Tile-Native LLM Kernels on
// Tenstorrent-Style Spatial Accelerators.
//
//===----------------------------------------------------------------------===//
//
// The host.cpp emitter. Given the lowered schedule IR (a `tensixflow.sched.
// kernel`) and the module-level #tensixflow.schedule plan, emit a complete,
// self-checking TT-Metalium host program for the pinned tt-metal release.
//
// The emitted program mirrors tt-metal v0.82.0-rc2
// programming_examples/matmul/matmul_single_core:
//
//   distributed::MeshDevice::create_unit_mesh,
//   Program CreateProgram(distributed::MeshDevice&),
//   distributed::MeshCommandQueue& cq = device->mesh_command_queue(),
//   distributed::MeshBuffer::create(ReplicatedBufferConfig,
//       DeviceLocalBufferConfig, device),
//   CircularBufferConfig(total, {{CBIndex, DataFormat::Float16_b}})
//       .set_page_size(index, tile_bytes),
//   CreateKernel(program, path, core, DataMovementConfig | ComputeConfig),
//   SetRuntimeArgs, MeshWorkload::add_program,
//   distributed::EnqueueWriteMeshBuffer / EnqueueMeshWorkload /
//       EnqueueReadMeshBuffer.
//
// The program computes a deterministic bf16 matmul on one core, compares the
// device result with a CPU reference matmul, and prints a machine-readable
// result line:
//
//   TENSIXFLOW_RESULT {"status":"pass","pcc":<value>,"threshold":<value>}
//
// scripts/run_generated.sh parses that line into JSON and uses the process exit
// code to decide the status.
//
//===----------------------------------------------------------------------===//

#ifndef TENSIXFLOW_CODEGEN_EMITHOST_H
#define TENSIXFLOW_CODEGEN_EMITHOST_H

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace mlir {
namespace tensixflow {

class TensixflowSchedKernelOp;
class TensixflowScheduleAttr;

/// Emit the complete host.cpp translation unit for the given schedule-IR
/// kernel and plan. The caller has already validated the schedule attribute,
/// the kernel, the absence of a fused plan, and the tensixflow.matmul_shape
/// attribute (see EmitMetaliumPass.cpp).
void emitHost(TensixflowSchedKernelOp kernel, TensixflowScheduleAttr schedule,
              llvm::raw_ostream &os);

} // namespace tensixflow
} // namespace mlir

#endif // TENSIXFLOW_CODEGEN_EMITHOST_H
