//===- FuseRmsnormMatmulPass.cpp - Fuse rmsnorm -> matmul -----*- C++ -*-===//
//
// TensixFlow: An MLIR Compiler for Tile-Native LLM Kernels on
// Tenstorrent-Style Spatial Accelerators.
//
//===----------------------------------------------------------------------===//
//
// The --tensixflow-fuse-rmsnorm-matmul pass. Fuses an adjacent
// `rmsnorm -> matmul` pair — where the rmsnorm result is the matmul's LHS and
// has exactly one use — into a single `tensixflow.fused_rmsnorm_matmul`,
// eliminating the DRAM round-trip of the normalized intermediate. The rmsnorm
// epsilon is preserved on the fused op. A multi-use rmsnorm (result consumed
// by anything besides the single matmul) is deliberately NOT fused; the MLP
// chain collapse to `fused_mlp` is a later todo (Wave 4, todo 21).
//
// This is the real body for the FuseRmsnormMatmulPass stub that previously
// lived in TensixflowPasses.cpp (that stub is removed so there is exactly one
// definition of the pass and its createXxx()).
//
//===----------------------------------------------------------------------===//

#include "tensixflow/TensixflowPasses.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensixflow/TensixflowOps.h"

namespace mlir {
namespace tensixflow {

// Emit impl::FuseRmsnormMatmulPassBase<DerivedT> (in mlir::tensixflow::impl)
// and the public createFuseRmsnormMatmulPass() definition for this one pass
// only — the other passes keep their GEN_PASS_DEF_* in TensixflowPasses.cpp
// (InferTileLayoutPass's lives in InferTileLayoutPass.cpp), so no createXxx()
// is defined twice.
#define GEN_PASS_DEF_FUSERMSNORMMATMULPASS
#include "tensixflow/TensixflowPasses.h.inc"

namespace {

/// Fuses `rmsnorm(%input, %gamma)` into a following `matmul(., %weight)` when
/// the rmsnorm result is the matmul's LHS and has exactly one use. The matmul
/// is replaced with `fused_rmsnorm_matmul(%input, %gamma, %weight)` (result
/// type inferred to the matmul's `[..., M, N]` shape) and the now-dead rmsnorm
/// is erased. Epsilon is copied from the rmsnorm onto the fused op.
struct FuseRmsnormMatmulPattern : OpRewritePattern<TensixflowRmsnormOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TensixflowRmsnormOp rmsnorm,
                                PatternRewriter &rewriter) const override {
    Value result = rmsnorm.getResult();

    // Guard: fuse only when the normalized intermediate feeds exactly one op.
    // A multi-use result (e.g. also consumed by a gelu) must stay materialized.
    if (!result.hasOneUse())
      return failure();

    // The single use must be a matmul ...
    auto matmul = dyn_cast<TensixflowMatmulOp>(*result.user_begin());
    if (!matmul)
      return failure();

    // ... and the rmsnorm result must be that matmul's LHS (the operand the
    // normalized intermediate normalizes), not its weight.
    if (matmul.getLhs() != result)
      return failure();

    // The fused op declares InferShapedTypeOpInterface (not
    // InferTypeOpInterface), so ODS emits no result-type-eliding builder: the
    // result type must be spelled out. The matmul's result type IS the fused
    // result shape (input leading/batch dims + weight's last dim), so passing
    // it keeps the replacement type-correct; epsilon rides the FloatAttr arg.
    rewriter.replaceOpWithNewOp<TensixflowFusedRmsnormMatmulOp>(
        matmul, matmul.getResult().getType(), rmsnorm.getInput(),
        rmsnorm.getGamma(), matmul.getRhs(), rmsnorm.getEpsilonAttr());

    // The rmsnorm result no longer has any use (the matmul was its only one),
    // so the rmsnorm is dead — drop it.
    rewriter.eraseOp(rmsnorm);
    return success();
  }
};

class FuseRmsnormMatmulPass final
    : public impl::FuseRmsnormMatmulPassBase<FuseRmsnormMatmulPass> {
public:
  using Base = impl::FuseRmsnormMatmulPassBase<FuseRmsnormMatmulPass>;
  using Base::Base;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<FuseRmsnormMatmulPattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
} // namespace tensixflow
} // namespace mlir
