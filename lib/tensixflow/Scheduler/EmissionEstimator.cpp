//===- EmissionEstimator.cpp - Emitted-path accounting ---------*- C++ -*-===//
//
// Implementation of the emitted-path accounting. See EmissionEstimator.h for
// the formulas and the mappings they describe.
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

/// The rows of the y-th row block when the grid is split into ceil blocks.
int64_t rowBlockSize(const EmissionInputs &inputs, int64_t rowBlock) {
  const int64_t start = rowBlock * ceilDiv(inputs.mTiles, inputs.gridY);
  const int64_t remaining = inputs.mTiles - start;
  return std::min(ceilDiv(inputs.mTiles, inputs.gridY),
                  std::max<int64_t>(remaining, 0));
}

/// The columns of the x-th column block when the grid is split into ceil blocks.
int64_t colBlockSize(const EmissionInputs &inputs, int64_t colBlock) {
  const int64_t start = colBlock * ceilDiv(inputs.nTiles, inputs.gridX);
  const int64_t remaining = inputs.nTiles - start;
  return std::min(ceilDiv(inputs.nTiles, inputs.gridX),
                  std::max<int64_t>(remaining, 0));
}

} // namespace

BlockAssignment planBlockAssignment(const EmissionInputs &inputs) {
  BlockAssignment block;

  // A block assignment needs a full tile geometry and a grid to spread it over.
  if (inputs.mTiles <= 0 || inputs.nTiles <= 0 || inputs.kTiles <= 0 ||
      inputs.gridX <= 0 || inputs.gridY <= 0) {
    return block;
  }

  const int64_t tileBytes =
      inputs.tileBytes > 0 ? inputs.tileBytes : kEmissionTileBytes;
  block.rowsPerCore = ceilDiv(inputs.mTiles, inputs.gridY);
  block.colsPerCore = ceilDiv(inputs.nTiles, inputs.gridX);
  block.usedRows = ceilDiv(inputs.mTiles, block.rowsPerCore);
  block.usedCols = ceilDiv(inputs.nTiles, block.colsPerCore);

  // The deepest 2D block is the largest row block times the largest column
  // block: the first block of each dimension is the full per-core extent.
  const int64_t maxRows = std::min(block.rowsPerCore, inputs.mTiles);
  const int64_t maxCols = std::min(block.colsPerCore, inputs.nTiles);
  block.maxTilesPerCore = maxRows * maxCols;

  // Resident buffers: the core's full activation row block and full weight
  // column block, plus the output buffer depth the compute/writer pair uses.
  block.in0Tiles = block.rowsPerCore * inputs.kTiles;
  block.in1Tiles = block.colsPerCore * inputs.kTiles;
  block.cbTiles = block.in0Tiles + block.in1Tiles + kEmissionOutputCbTiles;

  const int64_t budgetTiles = floorDiv(inputs.cbBudgetBytes, tileBytes);
  block.feasible = budgetTiles > 0 && block.cbTiles <= budgetTiles;
  // The emitted blocked-reuse reader multicasts the weight column block down
  // the used row blocks whenever there is more than one of them.
  block.weightMulticast = block.usedRows > 1;
  return block;
}

EmissionPlan estimateEmission(const EmissionInputs &inputs) {
  EmissionPlan plan;

  const int64_t tileBytes =
      inputs.tileBytes > 0 ? inputs.tileBytes : kEmissionTileBytes;
  const int64_t totalOutputTiles = inputs.mTiles * inputs.nTiles;
  const int64_t gridCores = inputs.gridX * inputs.gridY;
  if (totalOutputTiles <= 0 || gridCores <= 0) {
    return plan;
  }

  // The contiguous-range work split underlies the fused-variant projection and
  // is the fallback mapping when the resident blocks do not fit the budget.
  const int64_t rangeCoresUsed = std::min(gridCores, totalOutputTiles);
  const int64_t rangeMaxTilesPerCore =
      ceilDiv(totalOutputTiles, rangeCoresUsed);

  // Blocked reuse: stage each used row block and column block once and
  // accumulate every output tile of the block from L1. The activation is
  // replicated across the used column blocks and the weight across the used row
  // blocks, so the traffic drops by the per-dimension grid extent only.
  const BlockAssignment block = planBlockAssignment(inputs);
  if (block.feasible && !inputs.fused) {
    plan.mapping = kBlockedReuseMapping;
    plan.coresUsed = block.usedRows * block.usedCols;
    plan.maxTilesPerCore = block.maxTilesPerCore;

    int64_t readTiles = 0;
    for (int64_t rowBlock = 0; rowBlock < block.usedRows; ++rowBlock) {
      readTiles += inputs.kTiles * block.usedCols *
                   rowBlockSize(inputs, rowBlock);
    }
    // A multicast weight block is read once per used column block; a unicast
    // one is read once per used row block in every used column block.
    const int64_t weightReplication =
        block.weightMulticast ? 1 : block.usedRows;
    for (int64_t colBlock = 0; colBlock < block.usedCols; ++colBlock) {
      readTiles += inputs.kTiles * weightReplication *
                   colBlockSize(inputs, colBlock);
    }
    plan.dramReadBytes = readTiles * tileBytes;
    plan.dramWriteBytes = totalOutputTiles * tileBytes;
  } else {
    // Every output tile reads its K activation tiles and its K weight tiles,
    // unicast, with no reuse across output tiles.
    plan.mapping = kOutputTileRangesMapping;
    plan.coresUsed = rangeCoresUsed;
    plan.maxTilesPerCore = rangeMaxTilesPerCore;
    plan.dramReadBytes = totalOutputTiles * inputs.kTiles * 2 * tileBytes;
    plan.dramWriteBytes = totalOutputTiles * tileBytes;
  }

  // A fused variant stages the rows touched by one core's contiguous
  // output-tile range at once. The two-row allowance covers a range that
  // straddles a row boundary, matching the worst case of a contiguous range.
  const int64_t maxRowsPerCore =
      std::min(inputs.mTiles,
               floorDiv(rangeMaxTilesPerCore - 1, inputs.nTiles) + 2);
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
