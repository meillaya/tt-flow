//===- PlanNocShardingPass.cpp - Plan NoC sharding -------------*- C++ -*-===//
//
// TensixFlow: An MLIR Compiler for Tile-Native LLM Kernels on
// Tenstorrent-Style Spatial Accelerators.
//
//===----------------------------------------------------------------------===//
//
// The --tensixflow-plan-noc-sharding pass. Walks the module, finds the graph to
// plan — an already-fused op (fused_rmsnorm_matmul / fused_mlp), an UNFUSED
// rmsnorm -> matmul pair (planned AS fused, since the planner decides fusion),
// or a lone matmul (planned unfused) — derives the activation + weight operand
// shapes, runs the grid/weight/CB/NoC planners via buildPlanReport, attaches a
// #tensixflow.schedule attribute to the module (grid/sharding/weight/fusion/NoC
// populated; the CB arrays are left EMPTY for --tensixflow-allocate-circular-
// buffers to fill), and prints the human-readable "Chosen layout: ..." report to
// stderr. This pass only PLANS + reports; it emits NO kernels (Wave 6).
//
// This is the real body for the PlanNocShardingPass stub that previously lived
// in TensixflowPasses.cpp (that stub + its GEN_PASS_DEF are removed so there is
// exactly one definition of the pass and its createXxx()).
//
//===----------------------------------------------------------------------===//

#include "tensixflow/TensixflowPasses.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "tensixflow/TensixflowAttrs.h"
#include "tensixflow/TensixflowOps.h"

#include "Model.h"
#include "PlanReport.h"

#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace tensixflow {

// Emit impl::PlanNocShardingPassBase<DerivedT> (in mlir::tensixflow::impl) and
// the public createPlanNocShardingPass() definition for this one pass only — the
// other passes keep their GEN_PASS_DEF_* elsewhere, so no createXxx() is defined
// twice. (Double-N macro: PlanNoc -> PLANNOC.)
#define GEN_PASS_DEF_PLANNOCSHARDINGPASS
#include "tensixflow/TensixflowPasses.h.inc"

namespace {

/// Element-dtype name for the scheduler model ("bf16" / "fp16" / "fp32").
std::string dtypeString(Type elem) {
  if (elem.isBF16())
    return "bf16";
  if (elem.isF16())
    return "fp16";
  if (elem.isF32())
    return "fp32";
  return "bf16";
}

/// The scheduler shape of a value's ranked static tensor type (empty dims for a
/// non-tensor / dynamic value, which the planners guard against).
scheduler::TensorShape tensorShapeOf(Value value, const std::string &name) {
  scheduler::TensorShape shape;
  shape.name = name;
  auto ranked = dyn_cast<RankedTensorType>(value.getType());
  if (ranked && ranked.hasStaticShape()) {
    shape.dims.assign(ranked.getShape().begin(), ranked.getShape().end());
    shape.dtype = dtypeString(ranked.getElementType());
  }
  return shape;
}

class PlanNocShardingPass final
    : public impl::PlanNocShardingPassBase<PlanNocShardingPass> {
public:
  using Base = impl::PlanNocShardingPassBase<PlanNocShardingPass>;
  using Base::Base;

  void runOnOperation() final {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();

    std::string graphId = "graph";
    module.walk([&](TensixflowFuncOp func) {
      if (graphId == "graph")
        graphId = func.getName().str();
    });

    scheduler::TensorShape activation;
    scheduler::TensorShape weight;
    std::string fusion = "none";
    bool fused = false;
    bool found = false;

    // Case 1: an already-fused op carries its operand shapes directly.
    module.walk([&](TensixflowFusedRmsnormMatmulOp op) {
      if (found)
        return;
      activation = tensorShapeOf(op.getInput(), "activation");
      weight = tensorShapeOf(op.getWeight(), "weight");
      fusion = "fused_rmsnorm_matmul";
      fused = true;
      found = true;
    });
    if (!found)
      module.walk([&](TensixflowFusedMlpOp op) {
        if (found)
          return;
        activation = tensorShapeOf(op.getInput(), "activation");
        weight = tensorShapeOf(op.getW1(), "weight");
        fusion = "fused_mlp";
        fused = true;
        found = true;
      });

    // Case 2: an UNFUSED rmsnorm -> matmul pair (the running example). The
    // rmsnorm result feeds the matmul's LHS with a single use; the planner
    // decides fusion, so this is planned AS fused.
    if (!found)
      module.walk([&](TensixflowRmsnormOp rmsnorm) {
        if (found)
          return;
        Value result = rmsnorm.getResult();
        if (!result.hasOneUse())
          return;
        auto matmul = dyn_cast<TensixflowMatmulOp>(*result.user_begin());
        if (!matmul || matmul.getLhs() != result)
          return;
        activation = tensorShapeOf(rmsnorm.getInput(), "activation");
        weight = tensorShapeOf(matmul.getRhs(), "weight");
        fusion = "fused_rmsnorm_matmul";
        fused = true;
        found = true;
      });

    // Case 3: a lone matmul (no rmsnorm feeding it) — planned unfused, so the
    // fusion savings are zero (the required failure path).
    if (!found)
      module.walk([&](TensixflowMatmulOp matmul) {
        if (found)
          return;
        activation = tensorShapeOf(matmul.getLhs(), "activation");
        weight = tensorShapeOf(matmul.getRhs(), "weight");
        fusion = "none";
        fused = false;
        found = true;
      });

    if (!found)
      return;

    int bpe = scheduler::bytesPerElem(activation.dtype);
    if (bpe == 0)
      bpe = 2;

    scheduler::PlanRequest request{graphId, activation, weight,
                                   fusion,  fused,      bpe};
    scheduler::PlanReport report = scheduler::buildPlanReport(request);

    // Attach the schedule attribute. The CB arrays are left empty here;
    // --tensixflow-allocate-circular-buffers fills them from the CB allocator.
    ArrayAttr emptyCbs = ArrayAttr::get(ctx, SmallVector<Attribute>{});
    auto schedule = TensixflowScheduleAttr::get(
        ctx, static_cast<unsigned>(report.plan.shardGrid.gridY),
        static_cast<unsigned>(report.plan.shardGrid.gridX),
        StringAttr::get(ctx, report.plan.layout.activationSharding),
        StringAttr::get(ctx, report.plan.layout.weightStrategy),
        StringAttr::get(ctx, report.intermediateStorage),
        StringAttr::get(ctx, report.fusion), emptyCbs, emptyCbs, emptyCbs,
        report.noc.estimate.dramReadBytes, report.noc.estimate.dramWriteBytes,
        report.noc.estimate.l1Bytes, report.noc.estimate.multicastBytes,
        report.noc.savedDramReadBytes, report.noc.savedDramWriteBytes,
        report.noc.savedIntermediateBytes);
    module->setAttr("tensixflow.schedule", schedule);

    llvm::errs() << scheduler::formatPlanReport(report);
  }
};

} // namespace
} // namespace tensixflow
} // namespace mlir
