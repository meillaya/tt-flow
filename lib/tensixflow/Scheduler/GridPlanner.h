//===- GridPlanner.h --------------------------------------------*- C++ -*-===//
//
// Core-grid planner: maps an activation onto the selected device's Tensix
// compute grid with row-major sharding of the M dimension (dims[rank-2]) across
// the grid's Y axis, and checks the per-core circular-buffer budget from the
// generated device table. Pure data and math over the scheduler model
// (Model.h) with no MLIR dependency.
//
// The planner only decides the activation grid shard plus a feasible CB depth.
// The weight strategy, CB index allocation, and NoC estimates are separate
// planners that build on this one.
//
//===----------------------------------------------------------------------===//

#ifndef TENSIXFLOW_SCHEDULER_GRIDPLANNER_H
#define TENSIXFLOW_SCHEDULER_GRIDPLANNER_H

#include "Model.h"

#include "tensixflow/Generated/DeviceSpecs.h"

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
  int64_t numCBs = kDefaultNumCBs;
  int64_t cbDepth = 0;
  int64_t cbBytes = 0;
  int64_t tileBytes = 0;
  bool budgetFeasible = false;
};

/// Maximum feasible CB depth (in tiles) shared across `numCBs` circular buffers
/// under the per-core CB budget:
/// floor(cbBudget(reserveBytes) / (numCBs * device.tileBytes)).
/// Returns 0 when numCBs <= 0 or when even one tile per CB does not fit.
int64_t maxCBDepth(const device::DeviceSpec &deviceSpec, int64_t numCBs,
                   int64_t reserveBytes = device::kDefaultL1ReserveBytes);

/// Plan a row-major activation shard over the device's compute grid: for a
/// rank>=2 activation the M dimension (dims[rank-2]) splits across grid.gridY
/// cores, perCoreRows = ceilDiv(M, gridY) (e.g. [1,2048,4096] over an 8x8 grid
/// gives per-core [1,256,4096]). The CB depth starts at the tt-metal example
/// depth (two tiles) and clamps to maxCBDepth, so cbBytes never exceeds the
/// per-core CB budget; a too-large request gets a reduced depth, never an
/// overflow.
GridPlan planGrid(const TensorShape &activation,
                  const device::DeviceSpec &deviceSpec,
                  int64_t numCBs = kDefaultNumCBs,
                  int64_t reserveBytes = device::kDefaultL1ReserveBytes);

/// Convert a grid plan into a PlanDraft for the report: populates the grid, the
/// activation-sharding label, and one CBSpec per buffer with the device tile
/// size. For the default numCBs == 3 the CB indices follow the tt-metal
/// reader/compute/writer convention (in0 = c_0, in1 = c_1, out = c_16);
/// otherwise indices are sequential 0..numCBs-1.
PlanDraft toPlanDraft(const GridPlan &plan);

} // namespace scheduler
} // namespace tensixflow
} // namespace mlir

#endif // TENSIXFLOW_SCHEDULER_GRIDPLANNER_H
