//===- EmissionEstimator.cpp - Emitted-path accounting ---------*- C++ -*-===//
//
// Implementation of the emitted-path accounting. See EmissionEstimator.h for
// the formulas and the mapping they describe.
//
//===----------------------------------------------------------------------===//

#include "EmissionEstimator.h"

#include <algorithm>

namespace mlir {
namespace tensixflow {
namespace scheduler {

namespace {

/// Integer ceiling division, guarded for a non-positive divisor.
int64_t ceilDiv(int64_t numerator, int64_t denominator) {
  if (numerator <= 0 || denominator <= 0) {
    return 0;
  }
  return (numerator + denominator - 1) / denominator;
}

/// Integer floor division, guarded for a non-positive divisor.
int64_t floorDiv(int64_t numerator, int64_t denominator) {
  if (numerator <= 0 || denominator <= 0) {
    return 0;
  }
  return numerator / denominator;
}

} // namespace

EmissionPlan estimateEmission(const EmissionInputs &inputs) {
  EmissionPlan plan;

  const int64_t tileBytes =
      inputs.tileBytes > 0 ? inputs.tileBytes : kEmissionTileBytes;
  const int64_t totalOutputTiles = inputs.mTiles * inputs.nTiles;
  const int64_t gridCores = inputs.gridX * inputs.gridY;
  if (totalOutputTiles <= 0 || gridCores <= 0) {
    return plan;
  }

  // Whole output tiles are assigned to cores, so at most one core per output
  // tile is used and every core gets an even share up to one tile.
  plan.coresUsed = std::min(gridCores, totalOutputTiles);
  plan.maxTilesPerCore = ceilDiv(totalOutputTiles, plan.coresUsed);

  // Each output tile reads its K activation tiles and its K weight tiles,
  // unicast, with no reuse across output tiles. That upper bound is what the
  // emitted program actually moves.
  plan.dramReadBytes = totalOutputTiles * inputs.kTiles * 2 * tileBytes;
  plan.dramWriteBytes = totalOutputTiles * tileBytes;

  // A fused kernel would stage the rows touched by one core's output-tile range
  // at once. The two-row allowance covers a range that straddles a row boundary,
  // matching the worst case of a contiguous linear tile range.
  const int64_t maxRowsPerCore =
      std::min(inputs.mTiles, floorDiv(plan.maxTilesPerCore - 1, inputs.nTiles) + 2);
  plan.fusedStageTiles = maxRowsPerCore * inputs.kTiles;

  // The fused allocation holds the staged activation (stage tiles), the
  // normalized activation (stage tiles), the weight (kTiles), the gamma
  // (kTiles), one scaler tile, and two output tiles.
  const int64_t cbBudgetTiles = floorDiv(inputs.cbBudgetBytes, tileBytes);
  plan.fusedFeasible =
      2 * plan.fusedStageTiles + 2 * inputs.kTiles + 3 <= cbBudgetTiles;
  return plan;
}

} // namespace scheduler
} // namespace tensixflow
} // namespace mlir
