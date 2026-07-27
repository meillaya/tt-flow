//===- GridPlanner.h --------------------------------------------*- C++ -*-===//
//
// Core-grid planner (todo 26): maps an activation onto the modeled 8x8 Wormhole
// n150 grid with row-major sharding of the M dimension (dims[rank-2]) across the
// grid's Y axis, and checks the per-core circular-buffer budget from L1Model.h.
// Pure data + math over the scheduler model (Model.h) — NO MLIR dependency.
//
// The planner only decides the activation grid shard + a feasible CB depth; the
// weight strategy (todo 27), CB index allocation (todo 28), and NoC estimates
// (todo 29) are separate planners that build on this.
//
//===----------------------------------------------------------------------===//

#ifndef TENSIXFLOW_SCHEDULER_GRIDPLANNER_H
#define TENSIXFLOW_SCHEDULER_GRIDPLANNER_H

#include "L1Model.h"
#include "Model.h"

#include <cstdint>
#include <string>

namespace mlir {
namespace tensixflow {
namespace scheduler {

/// The planner's decision for one activation: the grid it maps onto, the
/// sharding scheme, the per-core shard shape, and a CB-budget-feasible depth.
struct GridPlan {
  plan::ShardGrid grid;
  std::string sharding;
  TensorShape perCoreShard;
  int64_t numCBs = 3;
  int64_t cbDepth = 0;
  int64_t cbBytes = 0;
  bool budgetFeasible = false;
};

/// Maximum feasible CB depth (in tiles) shared across `numCBs` circular buffers
/// under the per-core CB budget: floor(kCBShardBudget / (numCBs * kTileBytes)).
/// Returns 0 when numCBs <= 0 or when even one tile per CB does not fit.
int64_t maxCBDepth(int64_t numCBs);

/// Plan a row-major activation shard over `grid`: for a rank>=2 activation the M
/// dimension (dims[rank-2]) is split across grid.gridY cores, perCoreRows =
/// ceilDiv(M, gridY) (e.g. [1,2048,4096] over 8x8 -> per-core [1,256,4096]). The
/// CB depth is the max feasible depth for `numCBs` buffers, so cbBytes never
/// exceeds kCBShardBudget (a too-large request gets a reduced depth, never an
/// overflow). `bytesPerElem` sizes the tile for the report seam (toPlanDraft).
GridPlan planGrid(const TensorShape &activation, const plan::ShardGrid &grid,
                  int bytesPerElem, int64_t numCBs = 3);

/// Convert a grid plan into a PlanDraft (the todo-30 report seam): populates the
/// grid, the activation-sharding label, and one CBSpec per buffer via
/// toSchemaCBSpec. For the default numCBs == 3 the CB indices follow the
/// tt-metal reader/compute/writer convention (in0 = c_0, in1 = c_1, out = c_16);
/// otherwise indices are sequential 0..numCBs-1.
PlanDraft toPlanDraft(const GridPlan &plan, int bytesPerElem);

} // namespace scheduler
} // namespace tensixflow
} // namespace mlir

#endif // TENSIXFLOW_SCHEDULER_GRIDPLANNER_H
