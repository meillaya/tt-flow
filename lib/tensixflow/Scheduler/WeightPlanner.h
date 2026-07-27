//===- WeightPlanner.h ------------------------------------------*- C++ -*-===//
//
// Weight-strategy planner (todo 27): chooses how a weight [K, N] is distributed
// across the modeled 8x8 Wormhole n150 grid. A weight that fits the per-core
// circular-buffer budget (L1Model.h) is REPLICATED (every core holds a full
// copy); a weight that does not fit is COLUMN-SHARDED across the grid's X axis
// and MULTICAST across the grid's rows via row master nodes (arXiv:2606.09879
// NoC master-node multicast; tt-metal get_noc_multicast_addr), so it is moved
// once over the NoC rather than re-read from DRAM per core.
//
// Pure data + math over the scheduler model (Model.h) — NO MLIR dependency. The
// planner only decides the weight strategy + a per-core weight shape + an
// estimated multicast-bytes figure; the CB allocator (todo 28), NoC estimator
// (todo 29), and passes (todo 30) build on this. The report seam reuses the
// existing toString(WeightStrategy) -> plan::Layout::weightStrategy string.
//
//===----------------------------------------------------------------------===//

#ifndef TENSIXFLOW_SCHEDULER_WEIGHTPLANNER_H
#define TENSIXFLOW_SCHEDULER_WEIGHTPLANNER_H

#include "L1Model.h"
#include "Model.h"

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

/// Plan the distribution of a weight over `grid`. The weight is [K, N] with
/// K = dims[rank-2], N = dims[rank-1]; weightBytes = numElements * bytesPerElem.
///   - Replicate when weightBytes <= kCBShardBudget: every core holds a full
///     copy, multicastBytes = 0, perCoreWeight = weight.
///   - Multicast otherwise: column-shard N across grid.gridX cores
///     (perCoreCols = ceilDiv(N, gridX)) and multicast the columns across the
///     grid's rows; multicastBytes = weightBytes (moved once over the NoC rather
///     than read from DRAM per core), perCoreWeight = weight with dims[rank-1]
///     replaced by perCoreCols.
/// rank < 2 or gridX <= 0 leaves the weight unchanged (guarded, no crash). The
/// combined shard+multicast decision reports as WeightStrategy::Multicast; the
/// rationale details the column shard.
WeightPlan planWeight(const TensorShape &weight, const plan::ShardGrid &grid,
                      int bytesPerElem);

} // namespace scheduler
} // namespace tensixflow
} // namespace mlir

#endif // TENSIXFLOW_SCHEDULER_WEIGHTPLANNER_H
