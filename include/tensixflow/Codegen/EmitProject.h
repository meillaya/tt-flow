//===- Codegen/EmitProject.h - Assemble a standalone project ---*- C++ -*-===//
//
// TensixFlow: An MLIR Compiler for Tile-Native LLM Kernels on
// Tenstorrent-Style Spatial Accelerators.
//
//===----------------------------------------------------------------------===//
//
// The project assembler (Wave 6, todo 37). Given the tile-level schedule IR
// (a `tensixflow.sched.kernel`) and the module-level #tensixflow.schedule
// plan, write a complete standalone TT-Metalium-style project tree:
//
//   host.cpp / reader.cpp / compute.cpp / writer.cpp -- the four emitters
//       (todos 34-36), each rendered to its OWN file (no delimiter banners),
//   CMakeLists.txt -- compile-only: every kernel is checked `-fsyntax-only`
//       against the signature-only stub headers (todo 33). No link, no
//       device, no tt-metal library,
//   README.md -- kernel + plan summary, the current-API mapping, and the
//       structural-validation disclaimer.
//
// All six artifacts are rendered to in-memory strings FIRST; nothing touches
// the filesystem until every artifact exists in memory, so a render failure
// cannot leave a half-written tree. Input validation (schedule attr + kernel)
// is the caller's job and happens before emission entirely, so a validation
// failure writes nothing either.
//
//===----------------------------------------------------------------------===//

#ifndef TENSIXFLOW_CODEGEN_EMITPROJECT_H
#define TENSIXFLOW_CODEGEN_EMITPROJECT_H

#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"

#include <string>

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace mlir {
namespace tensixflow {

class TensixflowSchedKernelOp;
class TensixflowScheduleAttr;

/// Render the generated project's CMakeLists.txt: a compile-only target that
/// syntax-checks the four kernels against the stub headers (overridable via
/// -DTT_METAL_STUBS_DIR, defaulting to the in-repo stubs for a project that
/// lives at kernels/generated/<op>/).
void emitProjectCMakeLists(TensixflowSchedKernelOp kernel,
                           llvm::raw_ostream &os);

/// Render the generated project's README.md: the kernel (graph, fusion,
/// grid, circular-buffer allocation, tile program), the current-API mapping
/// per file, and the structural-validation disclaimer.
void emitProjectReadme(TensixflowSchedKernelOp kernel,
                       TensixflowScheduleAttr schedule, llvm::raw_ostream &os);

/// Write the complete project tree (host.cpp, reader.cpp, compute.cpp,
/// writer.cpp, CMakeLists.txt, README.md) for `kernel` + `schedule` into
/// `outputDir`, creating the directory (and parents) as needed. On any I/O
/// failure returns failure() and sets `error` to a descriptive message; the
/// caller reports it as a diagnostic. `error` is the failure-detail channel
/// (the kernel + schedule are the standard emitter source pair, outputDir the
/// sink -- three distinct roles, hence the four parameters).
LogicalResult emitProject(TensixflowSchedKernelOp kernel,
                          TensixflowScheduleAttr schedule,
                          llvm::StringRef outputDir, std::string &error);

} // namespace tensixflow
} // namespace mlir

#endif // TENSIXFLOW_CODEGEN_EMITPROJECT_H
