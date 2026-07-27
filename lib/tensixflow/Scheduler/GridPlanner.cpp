//===- GridPlanner.cpp ------------------------------------------*- C++ -*-===//
//
// Core-grid planner (todo 26). Row-major activation sharding over the modeled
// 8x8 grid + a CB-budget feasibility check against the shared L1 constants
// (L1Model.h). The depth is chosen as the max feasible depth, so a plan's
// cbBytes can never exceed kCBShardBudget — an oversized request is answered
// with a reduced depth (the documented "split-K / reduce depth" fallback), not
// an invalid plan.
//
//===----------------------------------------------------------------------===//

#include "GridPlanner.h"

#include "llvm/Support/MathExtras.h"

namespace mlir {
namespace tensixflow {
namespace scheduler {

int64_t maxCBDepth(int64_t numCBs) {
  if (numCBs <= 0)
    return 0;
  const int64_t bytesPerDepth = numCBs * kTileBytes;
  if (bytesPerDepth > kCBShardBudget)
    return 0;
  return kCBShardBudget / bytesPerDepth;
}

GridPlan planGrid(const TensorShape &activation, const plan::ShardGrid &grid,
                  int bytesPerElem, int64_t numCBs) {
  (void)bytesPerElem; // sizes the tile only at the report seam (toPlanDraft).

  GridPlan plan;
  plan.grid = grid;
  plan.sharding = "row-major";
  plan.numCBs = numCBs;
  plan.perCoreShard = activation;

  // Row-major: shard the M dimension (dims[rank-2]) across the grid's Y axis.
  const size_t rank = activation.dims.size();
  if (rank >= 2 && grid.gridY > 0) {
    const int64_t m = activation.dims[rank - 2];
    plan.perCoreShard.dims[rank - 2] = llvm::divideCeil(m, grid.gridY);
  }

  // Max feasible depth => cbBytes never overflows the per-core CB budget.
  plan.cbDepth = maxCBDepth(numCBs);
  plan.cbBytes = numCBs * plan.cbDepth * kTileBytes;
  plan.budgetFeasible = plan.cbDepth > 0 && plan.cbBytes <= kCBShardBudget;
  return plan;
}

PlanDraft toPlanDraft(const GridPlan &plan, int bytesPerElem) {
  PlanDraft draft;
  draft.grid = plan.grid;
  draft.activationSharding = plan.sharding;

  const TileShape tile; // 32x32 -> 2048 bytes at bf16.
  for (int64_t i = 0; i < plan.numCBs; ++i) {
    // tt-metal reader/compute/writer convention: in0 = c_0, in1 = c_1,
    // out = c_16 for the default 3-CB pipeline; sequential otherwise.
    const int64_t index = (plan.numCBs == 3 && i == 2) ? 16 : i;
    draft.cbSpecs.push_back(toSchemaCBSpec(index, plan.cbDepth, tile,
                                           bytesPerElem));
  }
  return draft;
}

} // namespace scheduler
} // namespace tensixflow
} // namespace mlir
