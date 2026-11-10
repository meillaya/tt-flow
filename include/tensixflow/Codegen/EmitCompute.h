//===- Codegen/EmitCompute.h - Emit a tt-metal compute.cpp -----*- C++ -*-===//
//
// TensixFlow: An MLIR Compiler for Tile-Native LLM Kernels on
// Tenstorrent-Style Spatial Accelerators.
//
//===----------------------------------------------------------------------===//
//
// The compute.cpp emitter. Emits the TRISC compute kernel for the pinned
// tt-metal release: compute_kernel_hw_startup, matmul_init, and the outer
// product blocked matmul loop over this core's output tiles (tile_regs_* around
// matmul_tiles, then pack_tile into the output circular buffer). The emitted
// kernel is derived from the pinned release's matmul_multi_core compute kernel;
// the circular-buffer indices come from the plan.
//
// Only the pure matmul path is emitted. A fused RMSNorm or MLP plan is rejected
// by the pass with an actionable diagnostic before this emitter is called,
// because the pinned compute API has no fused rmsnorm/gelu tile intrinsic and
// fabricating one would be a false compile claim.
//
//===----------------------------------------------------------------------===//

#ifndef TENSIXFLOW_CODEGEN_EMITCOMPUTE_H
#define TENSIXFLOW_CODEGEN_EMITCOMPUTE_H

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace mlir {
namespace tensixflow {

class TensixflowSchedKernelOp;
class TensixflowScheduleAttr;
struct MatmulEmission;

/// Emit the complete compute.cpp translation unit for the given schedule-IR
/// kernel and plan. The caller has validated the inputs first. `mapping` is the
/// selected output-tile mapping and, for blocked reuse, the 2D block geometry.
void emitCompute(TensixflowSchedKernelOp kernel, TensixflowScheduleAttr schedule,
                 const MatmulEmission &mapping, llvm::raw_ostream &os);

} // namespace tensixflow
} // namespace mlir

#endif // TENSIXFLOW_CODEGEN_EMITCOMPUTE_H
