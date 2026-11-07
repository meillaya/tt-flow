//===- WeightPlanner.cpp ----------------------------------------*- C++ -*-===//
//
// Weight-strategy planner. Replicate a weight that fits the per-core CB budget;
// otherwise column-shard N across the grid's X axis and multicast the columns
// across the rows, moving the weight once over the NoC instead of re-reading it
// from DRAM on every core.
//
//===----------------------------------------------------------------------===//

#include "WeightPlanner.h"

#include "llvm/Support/MathExtras.h"

#include <string>

namespace mlir {
namespace tensixflow {
namespace scheduler {

WeightPlan planWeight(const TensorShape &weight,
                      const device::DeviceSpec &deviceSpec, int bytesPerElem,
                      int64_t reserveBytes) {
  WeightPlan plan;
  plan.perCoreWeight = weight;

  const int64_t weightBytes = weight.bytes(bytesPerElem);
  const int64_t budget = deviceSpec.cbBudget(reserveBytes);
  const int64_t gridX = deviceSpec.computeGrid.gridX;
  const int64_t gridY = deviceSpec.computeGrid.gridY;

  // Replicate: the whole weight fits the per-core CB budget, so every core
  // holds a full copy and nothing is moved over the NoC for the weight.
  if (weightBytes <= budget) {
    plan.strategy = WeightStrategy::Replicate;
    plan.multicastBytes = 0;
    plan.rationale = "weight of " + std::to_string(weightBytes) +
                     " bytes fits the per-core L1 CB budget (" +
                     std::to_string(budget) + " bytes) on " + deviceSpec.name +
                     "; replicated (full copy on every core)";
    return plan;
  }

  // Multicast: column-shard N across the grid's X axis, then multicast the
  // columns across the rows via row master nodes. The weight moves once over
  // the NoC instead of being re-read from DRAM per core.
  plan.strategy = WeightStrategy::Multicast;
  plan.multicastBytes = weightBytes;

  const size_t rank = weight.dims.size();
  int64_t perCoreCols = 0;
  if (rank >= 2 && gridX > 0) {
    const int64_t n = weight.dims[rank - 1];
    perCoreCols = llvm::divideCeil(n, gridX);
    plan.perCoreWeight.dims[rank - 1] = perCoreCols;
  }

  plan.rationale = "weight of " + std::to_string(weightBytes) +
                   " bytes exceeds the per-core L1 CB budget (" +
                   std::to_string(budget) + " bytes) on " + deviceSpec.name +
                   "; column-sharded across " + std::to_string(gridX) +
                   " X-cores (per-core N=" + std::to_string(perCoreCols) +
                   "), multicast across " + std::to_string(gridY) +
                   " rows via row master nodes";
  return plan;
}

} // namespace scheduler
} // namespace tensixflow
} // namespace mlir
