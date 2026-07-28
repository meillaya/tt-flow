//===- Codegen/EmitReader.h - Emit a tt-metal reader.cpp -------*- C++ -*-===//
//
// TensixFlow: An MLIR Compiler for Tile-Native LLM Kernels on
// Tenstorrent-Style Spatial Accelerators.
//
//===----------------------------------------------------------------------===//
//
// The reader.cpp emitter (Wave 6, todo 35). Given the tile-level schedule IR
// (a `tensixflow.sched.kernel`) and the module-level #tensixflow.schedule plan,
// emit a complete, self-consistent reader.cpp translation unit: the BRISC
// data-movement kernel that stages input tiles into the input circular buffers
// over the NoC, mirroring tt-metal's reader_single_core_mm.cpp kernel_main.
//
// The emitted kernel_main reads its runtime/compile-time args (get_arg_val /
// get_compile_time_arg_val) and runs a tile loop doing
//   cb_reserve_back -> get_write_ptr -> noc_async_read (activation) ->
//   noc_async_read_barrier -> cb_push_back
// for the activation CB, and for the weight CB either a multicast
//   get_noc_multicast_addr + noc_async_write (+ noc_async_write_barrier)
// when the plan's weight strategy is "multicast", or a plain noc_async_read
// otherwise. CB indices come from the schedule allocation (in0/in1), never
// hardcoded. This is DATA MOVEMENT ONLY: no compute math (matmul_tiles /
// rmsnorm / gelu) -- that is compute.cpp (todo 36).
//
// The emitted text compiles `-fsyntax-only` against the signature-only stub
// headers in kernels/handwritten_baselines/tt_metal_stubs/ (todo 33); the
// hand-written matmul_single_core_baseline.cpp dataflow side is the
// proof-of-what-compiles this emitter mirrors.
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

/// Emit a complete reader.cpp translation unit (BRISC data-movement kernel) for
/// the given schedule-IR kernel + plan, writing it to `os`. The output is a
/// standalone TU: it includes "dataflow_api.h", opens the dataflow namespace,
/// and references only symbols the todo-33 stub headers declare, so it compiles
/// `-fsyntax-only` against them. Emits no compute math.
void emitReader(TensixflowSchedKernelOp kernel, TensixflowScheduleAttr schedule,
                llvm::raw_ostream &os);

} // namespace tensixflow
} // namespace mlir

#endif // TENSIXFLOW_CODEGEN_EMITREADER_H
