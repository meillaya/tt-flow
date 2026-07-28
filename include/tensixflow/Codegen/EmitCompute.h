//===- Codegen/EmitCompute.h - Emit a tt-metal compute.cpp -----*- C++ -*-===//
//
// TensixFlow: An MLIR Compiler for Tile-Native LLM Kernels on
// Tenstorrent-Style Spatial Accelerators.
//
//===----------------------------------------------------------------------===//
//
// The compute.cpp emitter (Wave 6, todo 36). Given the tile-level schedule IR
// (a `tensixflow.sched.kernel`) and the module-level #tensixflow.schedule plan,
// emit a complete, self-consistent compute.cpp translation unit: the TRISC
// compute kernel that consumes the tiles the reader staged into the input
// circular buffers, accumulates the tile matmul into a destination register
// (fp32), and packs result tiles into the output circular buffer for the
// writer to drain, mirroring the compute side of tt-metal's
// matmul_single_core kernel_main (matmul_init / tile_regs_* / matmul_tiles /
// pack_tile).
//
// The emitted kernel_main initializes the matmul engine (matmul_init) and runs
// an output tile loop:
//   tile_regs_acquire -> inner-K loop (cb_wait_front(in0/in1) -> matmul_tiles
//   -> cb_pop_front(in0/in1)) -> tile_regs_commit -> cb_reserve_back(out) ->
//   tile_regs_wait -> pack_tile(0, out) -> tile_regs_release -> cb_push_back(out)
// For a fused_rmsnorm_matmul / fused_mlp plan it additionally emits the RMSNorm
// prologue around the matmul as DOCUMENTED STRUCTURAL STAGING -- comments
// describing the reduce-square-sum / rsqrt / scale-by-gamma tile math (rounding
// contract: fp32 accumulate, bf16 round) plus the cb_* ops that stage the gamma
// tile from its circular buffer -- because the todo-33 stubs declare no rmsnorm
// / gelu intrinsic. fused_mlp adds a GELU epilogue marker likewise.
//
// This is TILE MATH + CB STAGING ONLY: no data movement (no noc_async_* /
// get_noc_addr) -- that is the reader/writer (todo 35).
//
// The emitted text compiles `-fsyntax-only` against the signature-only stub
// headers in kernels/handwritten_baselines/tt_metal_stubs/ (todo 33); the
// hand-written matmul_single_core_baseline.cpp compute side is the
// proof-of-what-compiles this emitter mirrors.
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

/// Emit a complete compute.cpp translation unit (TRISC compute kernel) for the
/// given schedule-IR kernel + plan, writing it to `os`. The output is a
/// standalone TU: it includes "compute_kernel_api.h" + "dataflow_api.h" +
/// "kernel_types.hpp", opens the compute namespace, and references only symbols
/// the todo-33 stub headers declare, so it compiles `-fsyntax-only` against
/// them. Emits no data movement.
void emitCompute(TensixflowSchedKernelOp kernel, TensixflowScheduleAttr schedule,
                 llvm::raw_ostream &os);

} // namespace tensixflow
} // namespace mlir

#endif // TENSIXFLOW_CODEGEN_EMITCOMPUTE_H
