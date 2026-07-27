//===- WeightPlanner.cpp ----------------------------------------*- C++ -*-===//
//
// Weight-strategy planner (todo 27). Replicate a weight that fits the per-core
// CB budget; otherwise column-shard N across the grid's X axis and multicast the
// columns across the rows (arXiv:2606.09879 NoC master-node multicast), moving
// the weight once over the NoC instead of re-reading it from DRAM per core.
//
//===----------------------------------------------------------------------===//

#include "WeightPlanner.h"

#include "llvm/Support/MathExtras.h"

#include <string>

namespace mlir {
namespace tensixflow {
namespace scheduler {

WeightPlan planWeight(const TensorShape &weight, const plan::ShardGrid &grid,
                      int bytesPerElem) {
  WeightPlan plan;
  plan.perCoreWeight = weight;

  const int64_t weightBytes = weight.bytes(bytesPerElem);

  // Replicate: the whole weight fits in the per-core CB budget, so every core
  // holds a full copy and nothing is multicast.
  if (weightBytes <= kCBShardBudget) {
    plan.strategy = WeightStrategy::Replicate;
    plan.multicastBytes = 0;
    plan.rationale = "weight of " + std::to_string(weightBytes) +
                     " bytes fits in L1 (per-core CB budget " +
                     std::to_string(kCBShardBudget) +
                     " bytes); replicated (full copy on every core)";
    return plan;
  }

  // Multicast: column-shard N across the grid's X axis, then multicast the
  // columns across the rows via row master nodes. The whole weight is moved
  // once over the NoC rather than read from DRAM per core.
  plan.strategy = WeightStrategy::Multicast;
  plan.multicastBytes = weightBytes;

  const size_t rank = weight.dims.size();
  int64_t perCoreCols = 0;
  if (rank >= 2 && grid.gridX > 0) {
    const int64_t n = weight.dims[rank - 1];
    perCoreCols = llvm::divideCeil(n, grid.gridX);
    plan.perCoreWeight.dims[rank - 1] = perCoreCols;
  }

  plan.rationale = "weight of " + std::to_string(weightBytes) +
                   " bytes exceeds the per-core L1 CB budget (" +
                   std::to_string(kCBShardBudget) +
                   " bytes); column-sharded across " +
                   std::to_string(grid.gridX) + " X-cores (per-core N=" +
                   std::to_string(perCoreCols) + "), multicast across " +
                   std::to_string(grid.gridY) +
                   " rows via row master nodes";
  return plan;
}

} // namespace scheduler
} // namespace tensixflow
} // namespace mlir
