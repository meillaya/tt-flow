//===- TensixflowPasses.cpp - Tensixflow pass stubs -------------*- C++ -*-===//
//
// TensixFlow: An MLIR Compiler for Tile-Native LLM Kernels on
// Tenstorrent-Style Spatial Accelerators.
//
//===----------------------------------------------------------------------===//
//
// Pass registration for the tensixflow dialect.
//
// CanonicalizePass has its body below (it applies the dialect's canonicalization
// patterns through the greedy rewrite driver). InferTileLayoutPass,
// FuseRmsnormMatmulPass, PlanNocShardingPass, AllocateCircularBuffersPass,
// LowerToTTKernelPass, EmitMetaliumPass, EmitTTIRPass, and ImportTTIRPass have
// bodies in their own .cpp files, so they are absent here to keep one definition
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
// PlanNocShardingPass.cpp, AllocateCircularBuffersPass's in
// AllocateCircularBuffersPass.cpp, LowerToTTKernelPass's in
// LowerToTTKernelPass.cpp, and EmitMetaliumPass's in EmitMetaliumPass.cpp.
#define GEN_PASS_DEF_CANONICALIZEPASS
#include "tensixflow/TensixflowPasses.h.inc"

namespace {

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
