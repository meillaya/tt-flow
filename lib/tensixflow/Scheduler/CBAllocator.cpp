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

int64_t fusedMlpCbTileCount(int64_t kTiles, int64_t hTiles,
                            int64_t rowsPerCore, int64_t colsPerCore) {
  if (kTiles <= 0 || hTiles <= 0 || rowsPerCore <= 0 || colsPerCore <= 0) {
    return 0;
  }
  const int64_t activationRowTiles = kTiles;
  const int64_t squaredRowTiles = kTiles;
  const int64_t inverseRmsTiles = kFusedMlpInverseRmsCbDepth;
  const int64_t normalizedRowTiles = kTiles;
  const int64_t weight1Tiles = kTiles * hTiles;
  const int64_t gammaTiles = kTiles;
  const int64_t scalerTiles = kFusedMlpScalerCbDepth;
  const int64_t biasTiles = hTiles;
  const int64_t weight2Tiles = hTiles * colsPerCore;
  const int64_t residualRowTiles = colsPerCore;
  const int64_t intermediateTiles = rowsPerCore * hTiles;
  return activationRowTiles + squaredRowTiles + inverseRmsTiles +
         normalizedRowTiles + weight1Tiles + gammaTiles + scalerTiles +
         biasTiles + weight2Tiles + residualRowTiles + intermediateTiles +
         kFusedOutputCbDepth;
}

CBAllocation allocateFusedMlpCBs(const device::DeviceSpec &deviceSpec,
                                 int64_t kTiles, int64_t hTiles,
                                 int64_t rowsPerCore, int64_t colsPerCore,
                                 int64_t reserveBytes) {
  CBAllocation allocation;
  if (kTiles <= 0 || hTiles <= 0 || rowsPerCore <= 0 || colsPerCore <= 0) {
    return allocation;
  }

  const int64_t depths[kFusedMlpCbCount] = {
      kTiles,
      kTiles,
      kFusedMlpInverseRmsCbDepth,
      kTiles,
      kTiles * hTiles,
      kTiles,
      kFusedMlpScalerCbDepth,
      hTiles,
      hTiles * colsPerCore,
      colsPerCore,
      rowsPerCore * hTiles,
      kFusedOutputCbDepth};
  const int64_t budget = deviceSpec.cbBudget(reserveBytes);
  for (std::size_t i = 0; i < kFusedMlpCbCount; ++i) {
    const int64_t index = kFusedMlpCBIndices[i];
    if (!isValidCBIndex(index)) {
      return CBAllocation{};
    }
    allocation.cbs.push_back(
        plan::CBSpec{index, depths[i], deviceSpec.tileBytes});
    allocation.totalBytes += depths[i] * deviceSpec.tileBytes;
  }

  // The reported depth is the intermediate staging depth, the buffer that
  // motivates the feasibility check.
  allocation.depth = rowsPerCore * hTiles;
  allocation.feasible = allocation.totalBytes <= budget;
  return allocation;
}

int64_t fusedCbTileCount(int64_t kTiles, int64_t stageTiles) {
  if (kTiles <= 0 || stageTiles <= 0) {
    return 0;
  }
  return 2 * stageTiles + 2 * kTiles + kFusedScalerCbDepth +
         kFusedOutputCbDepth;
}

CBAllocation allocateFusedCBs(const device::DeviceSpec &deviceSpec,
                              int64_t kTiles, int64_t stageTiles,
                              int64_t reserveBytes) {
  CBAllocation allocation;
  if (kTiles <= 0 || stageTiles <= 0) {
    return allocation;
  }

  const int64_t depths[kFusedCbCount] = {
      stageTiles, kTiles, kTiles, kFusedScalerCbDepth, stageTiles,
      kFusedOutputCbDepth};
  const int64_t budget = deviceSpec.cbBudget(reserveBytes);
  for (std::size_t i = 0; i < kFusedCbCount; ++i) {
    const int64_t index = kFusedCBIndices[i];
    if (!isValidCBIndex(index)) {
      return CBAllocation{};
    }
    allocation.cbs.push_back(
        plan::CBSpec{index, depths[i], deviceSpec.tileBytes});
    allocation.totalBytes += depths[i] * deviceSpec.tileBytes;
  }

  // The reported fused stage depth is the activation staging depth; the other
  // depths are fixed by the roles.
  allocation.depth = stageTiles;
  allocation.feasible = allocation.totalBytes <= budget;
  return allocation;
}

} // namespace scheduler
} // namespace tensixflow
} // namespace mlir
