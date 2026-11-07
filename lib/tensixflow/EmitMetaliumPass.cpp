//===- EmitMetaliumPass.cpp - Emit a TT-Metalium project --------*- C++ -*-===//
//
// TensixFlow: An MLIR Compiler for Tile-Native LLM Kernels on
// Tenstorrent-Style Spatial Accelerators.
//
//===----------------------------------------------------------------------===//
//
// The --tensixflow-emit-metalium pass: the code generation entry point. It reads
// the tile-level schedule IR (the `tensixflow.sched.kernel` built by
// --tensixflow-lower-to-ttkernel), the module-level #tensixflow.schedule plan
// attribute, and the module's tensixflow.device attribute, then emits the four
// generated translation units of the pinned-tt-metal project:
//
//   host.cpp            (emitHost),
//   kernels/compute.cpp (emitCompute),
//   kernels/reader.cpp  (emitReader),
//   kernels/writer.cpp  (emitWriter).
//
// With the `output-dir` option set (--tensixflow-emit-metalium=output-dir=<dir>)
// the pass writes a complete standalone project tree (the four sources plus a
// TT-Metalium CMakeLists.txt and a README.md, see lib/tensixflow/Codegen/
// EmitProject.cpp). With the option unset it prints the four artifacts to stdout
// separated by file-delimiter banners.
//
// Validation runs before any output in both modes: the schedule attribute, the
// sched.kernel, the matmul shape, and the unsupported fused path are all checked
// first, so a failure never leaves a half-written tree and never reports a
// compile that did not happen.
//
// The emitted project targets the pinned tt-metal release recorded in
// deps/tt-metal.lock. scripts/build_generated.sh compiles and links it, and
// scripts/run_generated.sh runs it under the official simulator.
//
//===----------------------------------------------------------------------===//

#include "tensixflow/TensixflowPasses.h"

#include "mlir/IR/BuiltinOps.h"
#include "tensixflow/Codegen/EmitCompute.h"
#include "tensixflow/Codegen/EmitCommon.h"
#include "tensixflow/Codegen/EmitHost.h"
#include "tensixflow/Codegen/EmitProject.h"
#include "tensixflow/Codegen/EmitReader.h"
#include "tensixflow/Codegen/EmitWriter.h"
#include "tensixflow/Generated/DeviceSpecs.h"
#include "tensixflow/TensixflowAttrs.h"
#include "tensixflow/TensixflowSchedule.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <string>

namespace mlir {
namespace tensixflow {

// Emit impl::EmitMetaliumPassBase<DerivedT> (in mlir::tensixflow::impl) and the
// public createEmitMetaliumPass() definition for this one pass only.
#define GEN_PASS_DEF_EMITMETALIUMPASS
#include "tensixflow/TensixflowPasses.h.inc"

namespace {

// The device name carried by the module's tensixflow.device attribute; the
// planner stamps it, and a default keeps a hand-written module emittable.
llvm::StringRef resolveDeviceName(ModuleOp module) {
  if (auto attr = module->getAttrOfType<StringAttr>("tensixflow.device")) {
    if (device::lookup(attr.getValue())) {
      return attr.getValue();
    }
  }
  return device::defaultDeviceName();
}

class EmitMetaliumPass final
    : public impl::EmitMetaliumPassBase<EmitMetaliumPass> {
public:
  using Base = impl::EmitMetaliumPassBase<EmitMetaliumPass>;
  using Base::Base;

  void runOnOperation() final {
    ModuleOp module = getOperation();

    // The plan attribute is required input.
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

    // The schedule-IR kernel is required input.
    TensixflowSchedKernelOp kernel;
    module.walk([&](TensixflowSchedKernelOp op) { kernel = op; });
    if (!kernel) {
      module.emitError(
          "requires a tensixflow.sched.kernel; run "
          "--tensixflow-lower-to-ttkernel first");
      signalPassFailure();
      return;
    }

    // The metal emitter covers the pure matmul path only. A fused graph would
    // need tile intrinsics the pinned compute API does not export, so fail with
    // the supported path and the planning alternative named instead of emitting
    // a program that cannot compute the fused graph.
    if (isFusedPlan(schedule)) {
      kernel.emitError()
          << "cannot emit a TT-Metalium program for the fused plan '"
          << fusionKind(schedule)
          << "'. The emitter supports the pure matmul path only "
             "(tensixflow.matmul with no fused prologue or epilogue; see "
             "examples/matmul.mlir). For a fused graph, use the plan and "
             "report path: run the planning passes from examples/rmsnorm_matmul.mlir, "
             "pipe the module into tensixflow-translate --tensixflow-to-json, "
             "then render it with tensixflow-report.";
      signalPassFailure();
      return;
    }

    // The matmul shape is what sizes the DRAM buffers and the reference matmul
    // in the emitted host program. Without it the emitter would have to invent
    // the dimensions, so refuse.
    if (!readMatmulShape(kernel)) {
      kernel.emitError()
          << "requires the " << kMatmulShapeAttrName
          << " attribute on " << kernel.getSymName()
          << "; run --tensixflow-lower-to-ttkernel to stamp it";
      signalPassFailure();
      return;
    }

    llvm::StringRef deviceName = resolveDeviceName(module);

    // File mode: assemble the complete standalone project tree.
    if (!outputDir.empty()) {
      std::string error;
      if (failed(emitProject(kernel, schedule, deviceName, outputDir, error))) {
        module.emitError("failed to write the Metalium project: ") << error;
        signalPassFailure();
        return;
      }
      std::exit(0);
    }

    // Stdout mode: the four artifacts with file-delimiter banners. host.cpp is
    // the prefix before the first banner; compute.cpp stays between host.cpp and
    // the two data-movement kernels so the FileCheck gates on reader/writer
    // output stay separate from the compute kernel.
    emitHost(kernel, schedule, llvm::outs());
    llvm::outs() << "\n// ===== file: compute.cpp =====\n";
    emitCompute(kernel, schedule, llvm::outs());
    llvm::outs() << "\n// ===== file: reader.cpp =====\n";
    emitReader(kernel, schedule, llvm::outs());
    llvm::outs() << "\n// ===== file: writer.cpp =====\n";
    emitWriter(kernel, schedule, llvm::outs());
    llvm::outs().flush();

    // mlir-opt prints the transformed module to stdout after all passes run, and
    // this MLIR build has no -disable-output flag. This is a terminal emission
    // pass whose stdout is the artifact set, so exit here with success; the
    // failure paths above signal failure and let mlir-opt report it.
    std::exit(0);
  }
};

} // namespace
} // namespace tensixflow
} // namespace mlir
