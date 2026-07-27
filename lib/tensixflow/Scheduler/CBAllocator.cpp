//===- CBAllocator.cpp ------------------------------------------*- C++ -*-===//
//
// Circular-buffer allocator (todo 28). Assigns tt-metal CB indices + a feasible
// shared depth from the per-core CB budget (L1Model.h). The depth is the max
// feasible depth (GridPlanner's maxCBDepth), so the total footprint never exceeds
// kCBShardBudget — an oversized request is answered with a reduced depth or
// flagged infeasible, never an overflow; no index past c_31 is ever emitted.
//
//===----------------------------------------------------------------------===//

#include "CBAllocator.h"

namespace mlir {
namespace tensixflow {
namespace scheduler {

namespace {

/// CB index for the i-th buffer of a `numCBs` allocation. The default 3-CB
/// pipeline follows the tt-metal reader/compute/writer convention (in0 = c_0,
/// in1 = c_1, out = c_16); any other count is sequential 0..numCBs-1.
int64_t cbIndex(int64_t i, int64_t numCBs) {
  if (numCBs == 3 && i == 2)
    return kCBOut;
  return i;
}

} // namespace

CBAllocation allocateCBs(int64_t numCBs, int bytesPerElem) {
  (void)bytesPerElem; // budget math is dtype-agnostic (kTileBytes is canonical).

  CBAllocation allocation;

  // Max feasible shared depth => totalBytes never overflows the CB budget.
  const int64_t depth = maxCBDepth(numCBs);

  bool indicesValid = numCBs > 0;
  for (int64_t i = 0; i < numCBs; ++i) {
    const int64_t index = cbIndex(i, numCBs);
    if (!isValidCBIndex(index)) {
      indicesValid = false;
      continue; // never emit an index past c_31.
    }
    allocation.cbs.push_back(plan::CBSpec{index, depth, kTileBytes});
  }

  allocation.totalBytes = numCBs * depth * kTileBytes;
  allocation.feasible =
      depth > 0 && allocation.totalBytes <= kCBShardBudget && indicesValid;
  return allocation;
}

std::vector<plan::CBSpec> toCBSpecs(const CBAllocation &allocation) {
  return allocation.cbs;
}

} // namespace scheduler
} // namespace tensixflow
} // namespace mlir
