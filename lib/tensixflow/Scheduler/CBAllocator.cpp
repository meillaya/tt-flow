//===- CBAllocator.cpp ------------------------------------------*- C++ -*-===//
//
// Circular-buffer allocator. Assigns tt-metal CB indices plus a depth that
// starts at the tt-metal example depth and clamps to the per-core CB budget
// from the generated device table. The total footprint never exceeds the
// budget; a request the device cannot hold is flagged infeasible instead of
// overflowing, and no index past c_31 is emitted.
//
//===----------------------------------------------------------------------===//

#include "CBAllocator.h"

#include <algorithm>

namespace mlir {
namespace tensixflow {
namespace scheduler {

namespace {

/// CB index for the i-th buffer of a `numCBs` allocation. The default three-CB
/// pipeline follows the tt-metal reader/compute/writer convention (in0 = c_0,
/// in1 = c_1, out = c_16); any other count is sequential 0..numCBs-1.
int64_t cbIndex(int64_t i, int64_t numCBs) {
  if (numCBs == kDefaultNumCBs && i == 2) {
    return kCBOut;
  }
  return i;
}

} // namespace

CBAllocation allocateCBs(const device::DeviceSpec &deviceSpec, int64_t numCBs,
                         int64_t requestedDepth, int64_t reserveBytes) {
  CBAllocation allocation;
  const int64_t budget = deviceSpec.cbBudget(reserveBytes);
  const int64_t maxDepth = maxCBDepth(deviceSpec, numCBs, reserveBytes);

  // Clamp the requested depth to the feasible maximum; a non-positive request
  // stays infeasible rather than silently becoming the default.
  int64_t depth = 0;
  if (maxDepth > 0 && requestedDepth > 0) {
    depth = std::min(requestedDepth, maxDepth);
  }
  allocation.depth = depth;

  bool indicesValid = numCBs > 0;
  for (int64_t i = 0; i < numCBs; ++i) {
    const int64_t index = cbIndex(i, numCBs);
    if (!isValidCBIndex(index)) {
      indicesValid = false;
      continue; // never emit an index past c_31.
    }
    allocation.cbs.push_back(
        plan::CBSpec{index, depth, deviceSpec.tileBytes});
  }

  allocation.totalBytes = numCBs * depth * deviceSpec.tileBytes;
  allocation.feasible =
      depth > 0 && allocation.totalBytes <= budget && indicesValid;
  return allocation;
}

std::vector<plan::CBSpec> toCBSpecs(const CBAllocation &allocation) {
  return allocation.cbs;
}

} // namespace scheduler
} // namespace tensixflow
} // namespace mlir
