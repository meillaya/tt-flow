//===- Codegen/EmitWriter.h - Emit a tt-metal writer.cpp -------*- C++ -*-===//
//
// TensixFlow: An MLIR Compiler for Tile-Native LLM Kernels on
// Tenstorrent-Style Spatial Accelerators.
//
//===----------------------------------------------------------------------===//
//
// The writer.cpp emitter (Wave 6, todo 35). Given the tile-level schedule IR
// (a `tensixflow.sched.kernel`) and the module-level #tensixflow.schedule plan,
// emit a complete, self-consistent writer.cpp translation unit: the NCRISC
// data-movement kernel that drains result tiles from the output circular buffer
// to their destination over the NoC, mirroring tt-metal's
// writer_single_core_mm.cpp kernel_main.
//
// The emitted kernel_main reads its runtime/compile-time args (get_arg_val /
// get_compile_time_arg_val) and runs a tile loop doing
//   cb_wait_front -> get_read_ptr -> noc_async_write ->
//   noc_async_write_barrier -> cb_pop_front
// for the output CB. The CB index comes from the schedule allocation (out),
// never hardcoded. This is DATA MOVEMENT ONLY: no compute math (matmul_tiles /
// rmsnorm / gelu) -- that is compute.cpp (todo 36).
//
// The emitted text compiles `-fsyntax-only` against the signature-only stub
// headers in kernels/handwritten_baselines/tt_metal_stubs/ (todo 33); the
// hand-written matmul_single_core_baseline.cpp dataflow side is the
// proof-of-what-compiles this emitter mirrors.
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

/// Emit a complete writer.cpp translation unit (NCRISC data-movement kernel) for
/// the given schedule-IR kernel + plan, writing it to `os`. The output is a
/// standalone TU: it includes "dataflow_api.h", opens the dataflow namespace,
/// and references only symbols the todo-33 stub headers declare, so it compiles
/// `-fsyntax-only` against them. Emits no compute math.
void emitWriter(TensixflowSchedKernelOp kernel, TensixflowScheduleAttr schedule,
                llvm::raw_ostream &os);

} // namespace tensixflow
} // namespace mlir

#endif // TENSIXFLOW_CODEGEN_EMITWRITER_H
