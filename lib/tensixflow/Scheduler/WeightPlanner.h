//===- WeightPlanner.h ------------------------------------------*- C++ -*-===//
//
// Weight-strategy planner: chooses how a weight [K, N] is distributed across the
// selected device's compute grid. A weight that fits the per-core
// circular-buffer budget is replicated (every core holds a full copy); a weight
// that does not fit is column-sharded across the grid's X axis and multicast
// across the grid's rows via row master nodes (tt-metal
// get_noc_multicast_addr), so the weight moves once over the NoC instead of
// being re-read from DRAM per core.
//
// Pure data and math over the scheduler model (Model.h) with no MLIR
// dependency. The planner decides the strategy, the per-core weight shape, and
// an estimated multicast byte count; the CB allocator, NoC estimator, and
// planning passes build on this. The report seam reuses toString(WeightStrategy)
// as plan::Layout::weightStrategy.
//
//===----------------------------------------------------------------------===//

#ifndef TENSIXFLOW_SCHEDULER_WEIGHTPLANNER_H
#define TENSIXFLOW_SCHEDULER_WEIGHTPLANNER_H

#include "Model.h"

#include "tensixflow/Generated/DeviceSpecs.h"

#include <cstdint>
#include <string>

namespace mlir {
namespace tensixflow {
namespace scheduler {

/// The planner's decision for one weight: the distribution strategy, a human
/// rationale, the estimated bytes moved by multicast (0 when replicated), and
/// the per-core weight shape (a full copy when replicated, the column shard
/// when multicast).
struct WeightPlan {
  WeightStrategy strategy = WeightStrategy::Replicate;
  std::string rationale;
  int64_t multicastBytes = 0;
  TensorShape perCoreWeight;
};

/// Plan the distribution of a weight over the device compute grid. The weight
/// is [K, N] with K = dims[rank-2], N = dims[rank-1]; weightBytes =
/// numElements * bytesPerElem.
///   - Replicate when weightBytes <= cbBudget(reserveBytes): every core holds a
///     full copy, multicastBytes = 0, perCoreWeight = weight.
///   - Multicast otherwise: column-shard N across gridX cores
///     (perCoreCols = ceilDiv(N, gridX)) and multicast the columns across the
///     grid's rows; multicastBytes = weightBytes (moved once over the NoC rather
///     than read from DRAM per core), perCoreWeight = weight with dims[rank-1]
///     replaced by perCoreCols.
/// A rank < 2 weight or a grid with gridX <= 0 leaves the shape unchanged; the
/// guarded path never crashes. The combined shard+multicast decision reports as
/// WeightStrategy::Multicast, and the rationale names the column shard.
WeightPlan planWeight(const TensorShape &weight,
                      const device::DeviceSpec &deviceSpec, int bytesPerElem,
                      int64_t reserveBytes = device::kDefaultL1ReserveBytes);

} // namespace scheduler
} // namespace tensixflow
} // namespace mlir

#endif // TENSIXFLOW_SCHEDULER_WEIGHTPLANNER_H
