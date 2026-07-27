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
// CanonicalizePass has a real body below (it applies the dialect's
// canonicalization patterns via the greedy rewrite driver). InferTileLayoutPass,
// FuseRmsnormMatmulPass, PlanNocShardingPass, and AllocateCircularBuffersPass
// have real bodies in their own .cpp files (their GEN_PASS_DEF_* + classes live
// there), so they are intentionally absent here to keep exactly one definition
// of each pass and its createXxx().
//
//===----------------------------------------------------------------------===//

#include "tensixflow/TensixflowPasses.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensixflow/TensixflowOps.h"

namespace mlir {
namespace tensixflow {

// Emit the impl::<Def>Base<DerivedT> base classes (in mlir::tensixflow::impl)
// and the public createXxx() definitions for the passes whose bodies live in
// this TU. InferTileLayoutPass's GEN_PASS_DEF lives in InferTileLayoutPass.cpp,
// FuseRmsnormMatmulPass's in FuseRmsnormMatmulPass.cpp, PlanNocShardingPass's in
// PlanNocShardingPass.cpp, and AllocateCircularBuffersPass's in
// AllocateCircularBuffersPass.cpp.
#define GEN_PASS_DEF_CANONICALIZEPASS
#define GEN_PASS_DEF_LOWERTOTTKERNELPASS
#define GEN_PASS_DEF_EMITMETALIUMPASS
#include "tensixflow/TensixflowPasses.h.inc"

namespace {

// One no-op stub per pass that has no real body yet. The generated friend
// impl::createXxx() instantiates std::make_unique<DerivedT>() lazily, so each
// stub only needs the inherited base constructors and an empty runOnOperation().
#define TENSIXFLOW_STUB_PASS(Name)                                             \
  class Name final : public impl::Name##Base<Name> {                           \
  public:                                                                      \
    using Base = impl::Name##Base<Name>;                                       \
    using Base::Base;                                                          \
    void runOnOperation() final {}                                             \
  };

TENSIXFLOW_STUB_PASS(LowerToTTKernelPass)
TENSIXFLOW_STUB_PASS(EmitMetaliumPass)

#undef TENSIXFLOW_STUB_PASS

// Real body for --tensixflow-canonicalize. Applies the dialect's
// canonicalization patterns (the same ones the standard --canonicalize pass
// picks up via hasCanonicalizer) with the greedy rewrite driver, so this flag
// and --canonicalize produce identical results on tensixflow IR.
class CanonicalizePass final
    : public impl::CanonicalizePassBase<CanonicalizePass> {
public:
  using Base = impl::CanonicalizePassBase<CanonicalizePass>;
  using Base::Base;

  void runOnOperation() final {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    TensixflowAddOp::getCanonicalizationPatterns(patterns, ctx);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
} // namespace tensixflow
} // namespace mlir
