//===- Codegen/EmitHost.h - Emit a tt-metal host.cpp ----------*- C++ -*-===//
//
// TensixFlow: An MLIR Compiler for Tile-Native LLM Kernels on
// Tenstorrent-Style Spatial Accelerators.
//
//===----------------------------------------------------------------------===//
//
// The host.cpp emitter (Wave 6, todo 34). Given the tile-level schedule IR
// (a `tensixflow.sched.kernel`) and the module-level #tensixflow.schedule plan,
// emit a complete, self-consistent host.cpp translation unit that uses the
// CURRENT tt-metal API only:
//
//   distributed::MeshDevice::create_unit_mesh + CreateProgram,
//   CreateKernel(program, file, core_spec, variant<DataMovementConfig,
//       ComputeConfig>)  -- the SINGLE current kernel entry point,
//   CreateCircularBuffer + CircularBufferConfig keyed by tt::CBIndex::c_*,
//   CreateBuffer(InterleavedBufferConfig) for the DRAM inputs/output,
//   SetCommonRuntimeArgs / SetRuntimeArgs,
//   distributed::EnqueueWriteMeshBuffer / EnqueueMeshWorkload /
//       EnqueueReadMeshBuffer.
//
// Legacy EnqueueProgram and the per-role CreateDataMovementKernel /
// CreateComputeKernel overloads are deliberately NEVER emitted (D3/D23).
//
// The emitted text compiles `-fsyntax-only` against the signature-only stub
// headers in kernels/handwritten_baselines/tt_metal_stubs/ (todo 33); the
// hand-written matmul_single_core_baseline.cpp is the proof-of-what-compiles
// this emitter mirrors. Todos 35/36 extend this directory with the reader /
// compute / writer kernel emitters; todo 37 writes the full project tree.
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

/// Emit a complete host.cpp translation unit (current tt-metal API) for the
/// given schedule-IR kernel + plan, writing it to `os`. The output is a
/// standalone TU: it includes "host_api.hpp", opens the host namespace, and
/// references only symbols the todo-33 stub headers declare, so it compiles
/// `-fsyntax-only` against them.
void emitHost(TensixflowSchedKernelOp kernel, TensixflowScheduleAttr schedule,
              llvm::raw_ostream &os);

} // namespace tensixflow
} // namespace mlir

#endif // TENSIXFLOW_CODEGEN_EMITHOST_H
