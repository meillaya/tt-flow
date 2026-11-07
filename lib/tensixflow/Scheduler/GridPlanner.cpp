//===- GridPlanner.cpp ------------------------------------------*- C++ -*-===//
//
// Core-grid planner. Row-major activation sharding over the device's compute
// grid plus a CB-budget feasibility check against the generated device table.
// The depth starts at the tt-metal example depth and clamps to the feasible
// maximum, so a plan's cbBytes can never exceed the per-core CB budget: an
// oversized request gets a reduced depth, not an invalid plan.
//
//===----------------------------------------------------------------------===//

#include "GridPlanner.h"

#include "llvm/Support/MathExtras.h"

#include <algorithm>

namespace mlir {
namespace tensixflow {
namespace scheduler {

int64_t maxCBDepth(const device::DeviceSpec &deviceSpec, int64_t numCBs,
                   int64_t reserveBytes) {
  if (numCBs <= 0) {
    return 0;
  }

  const int64_t budget = deviceSpec.cbBudget(reserveBytes);
  const int64_t bytesPerDepth = numCBs * deviceSpec.tileBytes;
  if (bytesPerDepth <= 0 || bytesPerDepth > budget) {
    return 0;
  }
  return budget / bytesPerDepth;
}

GridPlan planGrid(const TensorShape &activation,
                  const device::DeviceSpec &deviceSpec, int64_t numCBs,
                  int64_t reserveBytes) {
  GridPlan plan;
  plan.grid = plan::ShardGrid{deviceSpec.computeGrid.gridY,
                              deviceSpec.computeGrid.gridX};
  plan.sharding = "row-major";
  plan.numCBs = numCBs;
  plan.tileBytes = deviceSpec.tileBytes;
  plan.perCoreShard = activation;

  // Row-major: shard the M dimension (dims[rank-2]) across the grid's Y axis.
  const size_t rank = activation.dims.size();
  if (rank >= 2 && plan.grid.gridY > 0) {
    const int64_t m = activation.dims[rank - 2];
    plan.perCoreShard.dims[rank - 2] = llvm::divideCeil(m, plan.grid.gridY);
  }

  // Start from the example depth and clamp to the budget.
  const int64_t maxDepth = maxCBDepth(deviceSpec, numCBs, reserveBytes);
  plan.cbDepth = std::min(device::kDefaultCBDepthTiles, maxDepth);
  plan.cbBytes = numCBs * plan.cbDepth * deviceSpec.tileBytes;
  plan.budgetFeasible =
      plan.cbDepth > 0 && plan.cbBytes <= deviceSpec.cbBudget(reserveBytes);
  return plan;
}

PlanDraft toPlanDraft(const GridPlan &plan) {
  PlanDraft draft;
  draft.grid = plan.grid;
  draft.activationSharding = plan.sharding;

  for (int64_t i = 0; i < plan.numCBs; ++i) {
    // tt-metal reader/compute/writer convention: in0 = c_0, in1 = c_1,
    // out = c_16 for the default three-CB pipeline; sequential otherwise.
    const int64_t index = (plan.numCBs == kDefaultNumCBs && i == 2) ? kCBOut : i;
    draft.cbSpecs.push_back(plan::CBSpec{index, plan.cbDepth, plan.tileBytes});
  }
  return draft;
}

} // namespace scheduler
} // namespace tensixflow
} // namespace mlir
