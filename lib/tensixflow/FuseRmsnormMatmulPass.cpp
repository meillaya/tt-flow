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

/// Bytes occupied by the intermediate tensor a fusion eliminates: the element
/// count of a statically-shaped value's type times its element byte width
/// (bf16 = 2). Returns 0 for a dynamic shape or a non-int/float element type,
/// so such an intermediate simply contributes nothing to the byte tally.
static int64_t eliminatedBytesOf(Value v) {
  auto shaped = dyn_cast<ShapedType>(v.getType());
  if (!shaped || !shaped.hasStaticShape())
    return 0;
  Type elem = shaped.getElementType();
  if (!elem.isIntOrFloat())
    return 0;
  return shaped.getNumElements() * (elem.getIntOrFloatBitWidth() / 8);
}

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
    // Capture the eliminated intermediate's byte count BEFORE erasing the
    // rmsnorm so the type is still live to query.
    int64_t eliminatedBytes = eliminatedBytesOf(result);
    auto fused = rewriter.replaceOpWithNewOp<TensixflowFusedRmsnormMatmulOp>(
        matmul, matmul.getResult().getType(), rmsnorm.getInput(),
        rmsnorm.getGamma(), matmul.getRhs(), rmsnorm.getEpsilonAttr());

    // This fusion removes exactly one intermediate (the normalized rmsnorm
    // output); record the count and its byte size on the fused op.
    fused->setAttr("tensixflow.eliminated_tensors",
                   rewriter.getI64IntegerAttr(1));
    fused->setAttr("tensixflow.eliminated_bytes",
                   rewriter.getI64IntegerAttr(eliminatedBytes));

    // The rmsnorm result no longer has any use (the matmul was its only one),
    // so the rmsnorm is dead — drop it.
    rewriter.eraseOp(rmsnorm);
    return success();
  }
};

/// Fuses the full MLP chain
///   rmsnorm(x,gamma) -> matmul(.,w1) -> add(.,b1) -> gelu -> matmul(.,w2)
///     -> add(.,residual)
/// into one `fused_mlp(x, gamma, w1, b1, w2, residual)`. Every intermediate
/// must have exactly one use (so it is dead after the fuse and can be erased).
/// Matching is anchored on the rmsnorm and walks the chain link by link; any
/// missing or multi-use link aborts (returns failure) so a partial chain is
/// left for the rmsnorm+matmul pattern below. Registered with HIGHER benefit
/// than FuseRmsnormMatmulPattern so a full chain collapses to fused_mlp BEFORE
/// the head rmsnorm->matmul can be grabbed into a fused_rmsnorm_matmul.
struct FuseMlpPattern : OpRewritePattern<TensixflowRmsnormOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TensixflowRmsnormOp rmsnorm,
                                PatternRewriter &rewriter) const override {
    // rmsnorm result -> matmul1 (as LHS), single use.
    Value n = rmsnorm.getResult();
    if (!n.hasOneUse())
      return failure();
    auto matmul1 = dyn_cast<TensixflowMatmulOp>(*n.user_begin());
    if (!matmul1 || matmul1.getLhs() != n)
      return failure();

    // matmul1 result -> add1 (bias add), single use. The matmul result may be
    // either operand of the add; the OTHER operand is the bias.
    Value m1 = matmul1.getResult();
    if (!m1.hasOneUse())
      return failure();
    auto add1 = dyn_cast<TensixflowAddOp>(*m1.user_begin());
    if (!add1 || (add1.getLhs() != m1 && add1.getRhs() != m1))
      return failure();
    Value b1 = (add1.getLhs() == m1) ? add1.getRhs() : add1.getLhs();

    // add1 result -> gelu, single use.
    Value b = add1.getResult();
    if (!b.hasOneUse())
      return failure();
    auto gelu = dyn_cast<TensixflowGeluOp>(*b.user_begin());
    if (!gelu || gelu.getInput() != b)
      return failure();

    // gelu result -> matmul2 (as LHS), single use.
    Value g = gelu.getResult();
    if (!g.hasOneUse())
      return failure();
    auto matmul2 = dyn_cast<TensixflowMatmulOp>(*g.user_begin());
    if (!matmul2 || matmul2.getLhs() != g)
      return failure();

    // matmul2 result -> add2 (residual add), single use. As with the bias add,
    // the matmul result may be either operand; the OTHER operand is residual.
    Value m2 = matmul2.getResult();
    if (!m2.hasOneUse())
      return failure();
    auto add2 = dyn_cast<TensixflowAddOp>(*m2.user_begin());
    if (!add2 || (add2.getLhs() != m2 && add2.getRhs() != m2))
      return failure();
    Value residual = (add2.getLhs() == m2) ? add2.getRhs() : add2.getLhs();

    Value w1 = matmul1.getRhs();
    Value w2 = matmul2.getRhs();

    // The fused op declares InferShapedTypeOpInterface (not InferTypeOpInterface)
    // so ODS emits no result-type-eliding builder: spell the result type out.
    // The final add's result type IS the fused result shape (== residual shape),
    // so passing it keeps the replacement type-correct; epsilon rides the
    // FloatAttr arg, exactly like FuseRmsnormMatmulPattern. Operand order matches
    // the fused_mlp def: input, gamma, w1, b1, w2, residual.
    // Sum the byte sizes of the five eliminated intermediates BEFORE erasing
    // them: rmsnorm out (n), matmul1 out (m1), add1/bias out (b), gelu out (g),
    // matmul2 out (m2).
    int64_t eliminatedBytes = eliminatedBytesOf(n) + eliminatedBytesOf(m1) +
                              eliminatedBytesOf(b) + eliminatedBytesOf(g) +
                              eliminatedBytesOf(m2);
    auto fused = rewriter.replaceOpWithNewOp<TensixflowFusedMlpOp>(
        add2, add2.getResult().getType(), rmsnorm.getInput(),
        rmsnorm.getGamma(), w1, b1, w2, residual, rmsnorm.getEpsilonAttr());

    // This fusion removes five intermediates; record the count and total bytes.
    fused->setAttr("tensixflow.eliminated_tensors",
                   rewriter.getI64IntegerAttr(5));
    fused->setAttr("tensixflow.eliminated_bytes",
                   rewriter.getI64IntegerAttr(eliminatedBytes));

    // Erase the now-dead intermediates, consumers before producers (each is
    // dead once its single consumer above is gone).
    rewriter.eraseOp(matmul2);
    rewriter.eraseOp(gelu);
    rewriter.eraseOp(add1);
    rewriter.eraseOp(matmul1);
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
    ModuleOp module = getOperation();
    RewritePatternSet patterns(&getContext());
    // Higher benefit fires first: the full MLP chain collapses to fused_mlp
    // before the rmsnorm+matmul pattern can grab the chain head. A plain
    // rmsnorm->matmul that is NOT a full MLP chain fails FuseMlpPattern and
    // falls through to FuseRmsnormMatmulPattern.
    patterns.add<FuseMlpPattern>(&getContext(), /*benefit=*/2);
    patterns.add<FuseRmsnormMatmulPattern>(&getContext(), /*benefit=*/1);
    if (failed(applyPatternsGreedily(module, std::move(patterns))))
      signalPassFailure();

    // Count the fused ops the greedy rewrite produced and record the total as a
    // module-level fusion report counter. Counting AFTER the rewrite (rather
    // than bumping an attr from inside a pattern) keeps the patterns pure and
    // makes the counter correct even if a fused op already existed on input.
    int64_t fusions = 0;
    module.walk([&](Operation *op) {
      if (isa<TensixflowFusedRmsnormMatmulOp, TensixflowFusedMlpOp>(op))
        ++fusions;
    });
    OpBuilder builder(module.getBodyRegion());
    module->setAttr("tensixflow.fusion_report",
                    builder.getI64IntegerAttr(fusions));
  }
};

} // namespace
} // namespace tensixflow
} // namespace mlir
