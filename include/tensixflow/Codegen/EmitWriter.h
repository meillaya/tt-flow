//===- Codegen/EmitWriter.h - Emit a tt-metal writer.cpp -------*- C++ -*-===//
//
// TensixFlow: An MLIR Compiler for Tile-Native LLM Kernels on
// Tenstorrent-Style Spatial Accelerators.
//
//===----------------------------------------------------------------------===//
//
// The writer.cpp emitter. Emits the data-movement kernel that drains this
// core's output tiles from the circular buffer to DRAM for the pinned tt-metal
// release: one TensorAccessor for the output buffer and the tile loop that
// waits on the output circular buffer, writes each tile with
// noc_async_write_page at its linear output id, and pops it. The emitted kernel
// is derived from the pinned release's matmul_multi_core writer.
//
// The kernel compiles with tt-metal's JIT compiler at run time, so it is checked
// by actually building and running the emitted project against the pinned
// release, not by a syntax-only stub pass.
//
//===----------------------------------------------------------------------===//

#ifndef TENSIXFLOW_CODEGEN_EMITWRITER_H
#define TENSIXFLOW_CODEGEN_EMITWRITER_H

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace mlir {
namespace tensixflow {

class TensixflowSchedKernelOp;
class TensixflowScheduleAttr;
struct MatmulEmission;

/// Emit the complete writer.cpp translation unit for the given schedule-IR
/// kernel and plan. The caller has validated the inputs first. `mapping` is the
/// selected output-tile mapping and, for blocked reuse, the 2D block geometry.
void emitWriter(TensixflowSchedKernelOp kernel, TensixflowScheduleAttr schedule,
                const MatmulEmission &mapping, llvm::raw_ostream &os);

} // namespace tensixflow
} // namespace mlir

#endif // TENSIXFLOW_CODEGEN_EMITWRITER_H
