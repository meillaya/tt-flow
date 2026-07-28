//===- EmitMetaliumPass.cpp - Emit a tt-metal host.cpp ---------*- C++ -*-===//
//
// TensixFlow: An MLIR Compiler for Tile-Native LLM Kernels on
// Tenstorrent-Style Spatial Accelerators.
//
//===----------------------------------------------------------------------===//
//
// The --tensixflow-emit-metalium pass: the Wave-6 codegen entry point. It reads
// the tile-level schedule IR (the `tensixflow.sched.kernel` built by
// --tensixflow-lower-to-ttkernel) AND the module-level #tensixflow.schedule plan
// attribute, then emits the generated translation units to stdout: host.cpp
// (emitHost), reader.cpp (emitReader) and writer.cpp (emitWriter), separated by
// `// ===== file: <name>.cpp =====` delimiter banners (lib/tensixflow/Codegen/).
//
// This todo (35) emits HOST.CPP + READER.CPP + WRITER.CPP. Todo 36 extends this
// same pass to also emit compute.cpp; todo 37 writes the full project tree to
// disk (splitting stdout on the delimiter banners).
//
// This is the real body for the EmitMetaliumPass stub that previously lived in
// TensixflowPasses.cpp (that stub + its GEN_PASS_DEF_EMITMETALIUMPASS are removed
// so there is exactly one definition of the pass and its createXxx()).
//
//===----------------------------------------------------------------------===//

#include "tensixflow/TensixflowPasses.h"

#include "mlir/IR/BuiltinOps.h"
#include "tensixflow/Codegen/EmitHost.h"
#include "tensixflow/Codegen/EmitReader.h"
#include "tensixflow/Codegen/EmitWriter.h"
#include "tensixflow/TensixflowAttrs.h"
#include "tensixflow/TensixflowSchedule.h"

#include "llvm/Support/raw_ostream.h"

#include <cstdlib>

namespace mlir {
namespace tensixflow {

// Emit impl::EmitMetaliumPassBase<DerivedT> (in mlir::tensixflow::impl) and the
// public createEmitMetaliumPass() definition for this one pass only.
#define GEN_PASS_DEF_EMITMETALIUMPASS
#include "tensixflow/TensixflowPasses.h.inc"

namespace {

class EmitMetaliumPass final
    : public impl::EmitMetaliumPassBase<EmitMetaliumPass> {
public:
  using Base = impl::EmitMetaliumPassBase<EmitMetaliumPass>;
  using Base::Base;

  void runOnOperation() final {
    ModuleOp module = getOperation();

    // The plan attribute is required input (attached by --tensixflow-plan-noc-
    // sharding + --tensixflow-allocate-circular-buffers).
    auto schedule =
        module->getAttrOfType<TensixflowScheduleAttr>("tensixflow.schedule");
    if (!schedule) {
      module.emitError(
          "requires a tensixflow.schedule attribute; run "
          "--tensixflow-plan-noc-sharding "
          "--tensixflow-allocate-circular-buffers first");
      signalPassFailure();
      return;
    }

    // The schedule-IR kernel is required input (built by --tensixflow-lower-to-
    // ttkernel). Without it there is nothing to emit.
    TensixflowSchedKernelOp kernel;
    module.walk([&](TensixflowSchedKernelOp op) { kernel = op; });
    if (!kernel) {
      module.emitError(
          "requires a tensixflow.sched.kernel; run "
          "--tensixflow-lower-to-ttkernel first");
      signalPassFailure();
      return;
    }

    // Emit the three artifacts to stdout, each preceded by a file-delimiter
    // banner (a valid C++ comment line, so an extracted artifact compiles
    // whether the banner is kept or stripped). host.cpp is the prefix before the
    // first banner; reader.cpp / writer.cpp follow their banners. Todo 36 adds a
    // compute.cpp banner + emitter; todo 37's file-writer splits on these banners.
    emitHost(kernel, schedule, llvm::outs());
    llvm::outs() << "\n// ===== file: reader.cpp =====\n";
    emitReader(kernel, schedule, llvm::outs());
    llvm::outs() << "\n// ===== file: writer.cpp =====\n";
    emitWriter(kernel, schedule, llvm::outs());
    llvm::outs().flush();

    // mlir-opt prints the transformed module to stdout AFTER all passes run, and
    // this MLIR build has no -disable-output flag. This is a TERMINAL emission
    // pass whose stdout IS the artifact, so flush and exit here: the redirect
    // target is then a standalone-compilable set of translation units rather than
    // the artifacts followed by MLIR textual IR (which would fail the
    // -fsyntax-only stub compile). Exit 0 == success; the failure paths above
    // signalPassFailure() and let mlir-opt report + exit non-zero normally.
    std::exit(0);
  }
};

} // namespace
} // namespace tensixflow
} // namespace mlir
