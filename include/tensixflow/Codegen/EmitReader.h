//===- Codegen/EmitReader.h - Emit a tt-metal reader.cpp -------*- C++ -*-===//
//
// TensixFlow: An MLIR Compiler for Tile-Native LLM Kernels on
// Tenstorrent-Style Spatial Accelerators.
//
//===----------------------------------------------------------------------===//
//
// The reader.cpp emitter. Emits the BRISC data-movement kernel for the pinned
// tt-metal release: one TensorAccessor per DRAM input buffer, and the tile loop
// that streams the A and B tiles of this core's output-tile range into the
// input circular buffers with cb_reserve_back / noc_async_read_page /
// cb_push_back. The emitted kernel is derived from the pinned release's
// matmul_multi_core partitioned reader; the circular-buffer indices and the
// per-core tile range come from the plan.
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
struct MatmulEmission;

/// Emit the complete reader.cpp translation unit for the given schedule-IR
/// kernel and plan. The caller has validated the inputs first. `mapping` is the
/// selected output-tile mapping and, for blocked reuse, the 2D block geometry.
void emitReader(TensixflowSchedKernelOp kernel, TensixflowScheduleAttr schedule,
                const MatmulEmission &mapping, llvm::raw_ostream &os);

/// Emit kernels/reader_sender.cpp: the blocked-reuse weight sender that runs on
/// the top row block and multicasts each weight column block down the used row
/// blocks.
void emitReaderMulticastSender(TensixflowScheduleAttr schedule,
                               llvm::raw_ostream &os);

/// Emit kernels/reader_receiver.cpp: the blocked-reuse weight receiver that
/// runs on the row blocks below the top one and waits for that multicast.
void emitReaderMulticastReceiver(TensixflowScheduleAttr schedule,
                                 llvm::raw_ostream &os);

} // namespace tensixflow
} // namespace mlir

#endif // TENSIXFLOW_CODEGEN_EMITREADER_H
