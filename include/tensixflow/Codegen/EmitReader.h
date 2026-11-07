//===- Codegen/EmitReader.h - Emit a tt-metal reader.cpp -------*- C++ -*-===//
//
// TensixFlow: An MLIR Compiler for Tile-Native LLM Kernels on
// Tenstorrent-Style Spatial Accelerators.
//
//===----------------------------------------------------------------------===//
//
// The reader.cpp emitter. Emits the BRISC data-movement kernel for the pinned
// tt-metal release: one TensorAccessor per DRAM input buffer, and the tile loop
// that streams the A and B tiles of every output tile into the input circular
// buffers with cb_reserve_back / noc_async_read_page / cb_push_back. The
// emitted kernel is derived from the pinned release's
// matmul_single_core/kernels/dataflow/reader_single_core_mm.cpp; the circular
// buffer indices come from the plan's allocation.
//
// The kernel compiles with tt-metal's JIT compiler at run time, so it is checked
// by actually building and running the emitted project against the pinned
// release, not by a syntax-only stub pass.
//
//===----------------------------------------------------------------------===//

#ifndef TENSIXFLOW_CODEGEN_EMITREADER_H
#define TENSIXFLOW_CODEGEN_EMITREADER_H

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace mlir {
namespace tensixflow {

class TensixflowSchedKernelOp;
class TensixflowScheduleAttr;

/// Emit the complete reader.cpp translation unit for the given schedule-IR
/// kernel and plan. The caller has validated the inputs first.
void emitReader(TensixflowSchedKernelOp kernel, TensixflowScheduleAttr schedule,
                llvm::raw_ostream &os);

} // namespace tensixflow
} // namespace mlir

#endif // TENSIXFLOW_CODEGEN_EMITREADER_H
