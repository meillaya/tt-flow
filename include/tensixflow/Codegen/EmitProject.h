//===- Codegen/EmitProject.h - Assemble a standalone project ---*- C++ -*-===//
//
// TensixFlow: An MLIR Compiler for Tile-Native LLM Kernels on
// Tenstorrent-Style Spatial Accelerators.
//
//===----------------------------------------------------------------------===//
//
// The project assembler. Given the lowered schedule IR (a `tensixflow.sched.
// kernel`), the module-level #tensixflow.schedule plan, and the device name from
// the module's tensixflow.device attribute, write a complete standalone
// TT-Metalium project tree:
//
//   host.cpp                       self-checking host program (EmitHost),
//   kernels/reader.cpp             data-movement kernel (EmitReader),
//   kernels/compute.cpp            matmul kernel (EmitCompute),
//   kernels/writer.cpp             data-movement kernel (EmitWriter),
//   CMakeLists.txt                 find_package(TT-Metalium) and TT::Metalium,
//   README.md                      plan summary, build/run steps, limitations.
//
// All six artifacts are rendered to memory before the first filesystem write, so
// a render failure cannot leave a half-written tree. The caller validates the
// schedule attribute, the kernel, the absence of a fused plan, and the matmul
// shape before this function runs, so a validation failure also writes nothing.
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
class TensixflowEmissionAttr;

/// Render the generated project's CMakeLists.txt: a TT-Metalium executable that
/// links the real package with find_package(TT-Metalium REQUIRED) and TT::Metalium
/// and JIT-compiles the kernel sources at run time.
void emitProjectCMakeLists(TensixflowSchedKernelOp kernel,
                           llvm::raw_ostream &os);

/// Render the generated project's README.md: the plan summary, the tile
/// geometry, the circular-buffer allocation against the device's usable L1
/// budget, the build/run commands, the one-core limitation, and the honesty
/// statement for the pinned upstream release.
void emitProjectReadme(TensixflowSchedKernelOp kernel,
                       TensixflowScheduleAttr schedule, llvm::StringRef deviceName,
                       TensixflowEmissionAttr emission, llvm::raw_ostream &os);

/// Write the complete project tree for `kernel` + `schedule` into `outputDir`,
/// creating the directory (and parents) as needed. On any I/O failure returns
/// failure() and sets `error` to a descriptive message.
LogicalResult emitProject(TensixflowSchedKernelOp kernel,
                          TensixflowScheduleAttr schedule,
                          llvm::StringRef deviceName,
                          TensixflowEmissionAttr emission,
                          llvm::StringRef outputDir, std::string &error);

} // namespace tensixflow
} // namespace mlir

#endif // TENSIXFLOW_CODEGEN_EMITPROJECT_H
