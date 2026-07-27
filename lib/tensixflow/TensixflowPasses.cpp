//===- TensixflowPasses.cpp - Tensixflow pass stubs -------------*- C++ -*-===//
//
// TensixFlow: An MLIR Compiler for Tile-Native LLM Kernels on
// Tenstorrent-Style Spatial Accelerators.
//
//===----------------------------------------------------------------------===//
//
// No-op stub definitions for the tensixflow passes declared in
// TensixflowPasses.td so that GEN_PASS_REGISTRATION links and multi-pass
// pipelines compose. Each later todo replaces its stub with a real body; the
// generated anchor a body must derive from is `impl::<Def>Base<Derived>`.
//
// InferTileLayoutPass has a real body in InferTileLayoutPass.cpp (its
// GEN_PASS_DEF_* + class live there), so it is intentionally absent here to
// keep exactly one definition of the pass and its createXxx().
//
//===----------------------------------------------------------------------===//

#include "tensixflow/TensixflowPasses.h"

#include "mlir/IR/BuiltinOps.h"

namespace mlir {
namespace tensixflow {

// Emit the impl::<Def>Base<DerivedT> base classes (in mlir::tensixflow::impl)
// and the public createXxx() definitions for the six stubbed passes.
// InferTileLayoutPass's GEN_PASS_DEF lives in InferTileLayoutPass.cpp.
#define GEN_PASS_DEF_CANONICALIZEPASS
#define GEN_PASS_DEF_FUSERMSNORMMATMULPASS
#define GEN_PASS_DEF_PLANNOCSHARDINGPASS
#define GEN_PASS_DEF_ALLOCATECIRCULARBUFFERSPASS
#define GEN_PASS_DEF_LOWERTOTTKERNELPASS
#define GEN_PASS_DEF_EMITMETALIUMPASS
#include "tensixflow/TensixflowPasses.h.inc"

namespace {

// One no-op stub per pass. The generated friend impl::createXxx() instantiates
// std::make_unique<DerivedT>() lazily, so each stub only needs the inherited
// base constructors and an empty runOnOperation().
#define TENSIXFLOW_STUB_PASS(Name)                                             \
  class Name final : public impl::Name##Base<Name> {                           \
  public:                                                                      \
    using Base = impl::Name##Base<Name>;                                       \
    using Base::Base;                                                          \
    void runOnOperation() final {}                                             \
  };

TENSIXFLOW_STUB_PASS(CanonicalizePass)
TENSIXFLOW_STUB_PASS(FuseRmsnormMatmulPass)
TENSIXFLOW_STUB_PASS(PlanNocShardingPass)
TENSIXFLOW_STUB_PASS(AllocateCircularBuffersPass)
TENSIXFLOW_STUB_PASS(LowerToTTKernelPass)
TENSIXFLOW_STUB_PASS(EmitMetaliumPass)

#undef TENSIXFLOW_STUB_PASS

} // namespace
} // namespace tensixflow
} // namespace mlir
