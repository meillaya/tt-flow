//===- EmissionEstimatorTest.cpp --------------------------------*- C++ -*-===//
//
// Unit tests for the emitted-path accounting: the mapping selector, the 2D
// block assignment, and the traffic the emitted TT-Metalium program will
// produce. The expected values are computed by hand from the documented
// formulas, not read back from the implementation.
//
//===----------------------------------------------------------------------===//

#include "EmissionEstimator.h"

#include "gtest/gtest.h"

using mlir::tensixflow::scheduler::BlockAssignment;
using mlir::tensixflow::scheduler::EmissionInputs;
using mlir::tensixflow::scheduler::EmissionPlan;
using mlir::tensixflow::scheduler::estimateEmission;
using mlir::tensixflow::scheduler::kBlockedReuseMapping;
using mlir::tensixflow::scheduler::kOutputTileRangesMapping;
using mlir::tensixflow::scheduler::planBlockAssignment;

namespace {

// The per-core circular-buffer budget of an n150: 1,499,136 usable L1 bytes
// minus the 65,536-byte planning reserve, so 1,433,600 bytes or 700 tiles.
constexpr int64_t kN150CbBudgetBytes = 1433600;

EmissionInputs inputs(int64_t mTiles, int64_t nTiles, int64_t kTiles,
                      int64_t gridY = 8, int64_t gridX = 8) {
  EmissionInputs in;
  in.mTiles = mTiles;
  in.nTiles = nTiles;
  in.kTiles = kTiles;
  in.gridY = gridY;
  in.gridX = gridX;
  in.tileBytes = 2048;
  in.cbBudgetBytes = kN150CbBudgetBytes;
  return in;
}

} // namespace

TEST(EmissionEstimatorTest, MatmulExampleBlockedReuseValues) {
  // examples/matmul.mlir: M=256, N=1024, K=512 on the n150 8x8 grid, so
  // 8 x 32 output tiles and 16 K tiles. The grid splits into 8 row blocks of
  // 1 and 8 column blocks of 4. A is read once per used column block; the
  // weight column block is multicast down the used row blocks, so it is read
  // once per used column block:
  //   A reads = 16 * 8 * (1 * 8) = 1024 tiles
  //   B reads = 16 * 1 * (4 * 8) = 512 tiles
  //   writes  = 8 * 32 = 256 tiles
  // The resident blocks need (1 + 4) * 16 + 2 = 82 of the 700 budget tiles.
  const EmissionPlan plan = estimateEmission(
      inputs(/*mTiles=*/8, /*nTiles=*/32, /*kTiles=*/16));
  EXPECT_EQ(plan.mapping, kBlockedReuseMapping);
  EXPECT_EQ(plan.coresUsed, 64);
  EXPECT_EQ(plan.maxTilesPerCore, 4);
  EXPECT_EQ(plan.dramReadBytes, (1024 + 512) * 2048);
  EXPECT_EQ(plan.dramWriteBytes, 256 * 2048);
  EXPECT_EQ(plan.fusedStageTiles, 32);
  EXPECT_TRUE(plan.fusedFeasible);
}

TEST(EmissionEstimatorTest, MatmulExampleFallsBackToOutputTileRanges) {
  // The same geometry with only 81 budget tiles: the resident blocks need 82,
  // so the selection keeps the output-tile-ranges mapping and its unicast
  // formulas: every output tile reads its K activation and K weight tiles.
  EmissionInputs in = inputs(/*mTiles=*/8, /*nTiles=*/32, /*kTiles=*/16);
  in.cbBudgetBytes = 81 * 2048;
  const EmissionPlan plan = estimateEmission(in);
  EXPECT_EQ(plan.mapping, kOutputTileRangesMapping);
  EXPECT_EQ(plan.coresUsed, 64);
  EXPECT_EQ(plan.maxTilesPerCore, 4);
  EXPECT_EQ(plan.dramReadBytes, 256 * 16 * 2 * 2048);
  EXPECT_EQ(plan.dramWriteBytes, 256 * 2048);
  EXPECT_EQ(plan.fusedStageTiles, 32);
  EXPECT_FALSE(plan.fusedFeasible);
}

TEST(EmissionEstimatorTest, SelectorBoundaryIsExact) {
  // Six budget tiles exactly hold the blocked-reuse allocation of the 2 x 2
  // output-tile geometry (A block 2 + B block 2 + output 2), and five do not.
  EmissionInputs in = inputs(/*mTiles=*/2, /*nTiles=*/2, /*kTiles=*/2);
  in.cbBudgetBytes = 6 * 2048;
  EXPECT_EQ(estimateEmission(in).mapping, kBlockedReuseMapping);
  in.cbBudgetBytes = 5 * 2048;
  EXPECT_EQ(estimateEmission(in).mapping, kOutputTileRangesMapping);
}

TEST(EmissionEstimatorTest, MatmulSmallExampleValues) {
  // examples/matmul_small.mlir: M=128, N=128, K=64, so 4 x 4 output tiles and
  // 2 K tiles. The grid has 64 cores but the tiling uses 4 row blocks and 4
  // column blocks of one tile each:
  //   A reads = 2 * 4 * (1 * 4) = 32 tiles
  //   B reads = 2 * 1 * (1 * 4) = 8 tiles
  //   writes  = 4 * 4 = 16 tiles
  const EmissionPlan plan = estimateEmission(
      inputs(/*mTiles=*/4, /*nTiles=*/4, /*kTiles=*/2));
  EXPECT_EQ(plan.mapping, kBlockedReuseMapping);
  EXPECT_EQ(plan.coresUsed, 16);
  EXPECT_EQ(plan.maxTilesPerCore, 1);
  EXPECT_EQ(plan.dramReadBytes, (32 + 8) * 2048);
  EXPECT_EQ(plan.dramWriteBytes, 16 * 2048);
  EXPECT_EQ(plan.fusedStageTiles, 4);
  EXPECT_TRUE(plan.fusedFeasible);
}

TEST(EmissionEstimatorTest, PaddedExampleMatchesTileGeometry) {
  // examples/matmul_padded.mlir: M=100, N=100, K=50 rounds up to 4 x 4 output
  // tiles and 2 K tiles, the same tile geometry as matmul_small.
  const EmissionPlan plan = estimateEmission(
      inputs(/*mTiles=*/4, /*nTiles=*/4, /*kTiles=*/2));
  EXPECT_EQ(plan.mapping, kBlockedReuseMapping);
  EXPECT_EQ(plan.coresUsed, 16);
  EXPECT_EQ(plan.maxTilesPerCore, 1);
  EXPECT_EQ(plan.dramReadBytes, (32 + 8) * 2048);
  EXPECT_EQ(plan.dramWriteBytes, 16 * 2048);
}

TEST(EmissionEstimatorTest, BlockAssignmentMatchesTheExamples) {
  // matmul.mlir: 8 rows over 8 row blocks of 1, 32 columns over 8 column
  // blocks of 4; the resident buffers are 1 * 16 and 4 * 16 tiles.
  const BlockAssignment matmul =
      planBlockAssignment(inputs(/*mTiles=*/8, /*nTiles=*/32, /*kTiles=*/16));
  EXPECT_EQ(matmul.rowsPerCore, 1);
  EXPECT_EQ(matmul.colsPerCore, 4);
  EXPECT_EQ(matmul.usedRows, 8);
  EXPECT_EQ(matmul.usedCols, 8);
  EXPECT_EQ(matmul.maxTilesPerCore, 4);
  EXPECT_EQ(matmul.in0Tiles, 16);
  EXPECT_EQ(matmul.in1Tiles, 64);
  EXPECT_EQ(matmul.cbTiles, 82);
  EXPECT_TRUE(matmul.weightMulticast);
  EXPECT_TRUE(matmul.feasible);

  // matmul_small.mlir: 4 rows and 4 columns over 8 grid rows and 8 grid
  // columns, so only four row blocks and four column blocks are used.
  const BlockAssignment small =
      planBlockAssignment(inputs(/*mTiles=*/4, /*nTiles=*/4, /*kTiles=*/2));
  EXPECT_EQ(small.rowsPerCore, 1);
  EXPECT_EQ(small.colsPerCore, 1);
  EXPECT_EQ(small.usedRows, 4);
  EXPECT_EQ(small.usedCols, 4);
  EXPECT_EQ(small.maxTilesPerCore, 1);
  EXPECT_EQ(small.in0Tiles, 2);
  EXPECT_EQ(small.in1Tiles, 2);
  EXPECT_EQ(small.cbTiles, 6);
  EXPECT_TRUE(small.weightMulticast);
  EXPECT_TRUE(small.feasible);
}

TEST(EmissionEstimatorTest, FusedPlanKeepsOutputTileRangesMapping) {
  // The fused emitter stages activation rows itself over contiguous
  // output-tile ranges, so a fused plan records the range mapping even when the
  // blocked-reuse blocks would fit the budget.
  EmissionInputs in = inputs(/*mTiles=*/4, /*nTiles=*/4, /*kTiles=*/2);
  in.fused = true;
  const EmissionPlan plan = estimateEmission(in);
  EXPECT_EQ(plan.mapping, kOutputTileRangesMapping);
  EXPECT_EQ(plan.coresUsed, 16);
  EXPECT_EQ(plan.maxTilesPerCore, 1);
  EXPECT_EQ(plan.dramReadBytes, 16 * 2 * 2 * 2048);
  EXPECT_EQ(plan.dramWriteBytes, 16 * 2048);
  EXPECT_EQ(plan.fusedStageTiles, 4);
  EXPECT_TRUE(plan.fusedFeasible);
}

TEST(EmissionEstimatorTest, FusedMlpSelectsBlockedMappingAndSizesFootprint) {
  // The small MLP: 4 x 4 output tiles, 4 K tiles, 4 hidden tiles on the 8 x 8
  // grid, so blocks of 1 x 1 over 4 used row blocks and 4 used column blocks.
  // Reads: activation 4 * 4 * 4 = 64 tiles (replicated across column blocks),
  // first weight block 4 * 4 * 16 = 256 (on every used core), second weight
  // block 4 * 4 * 4 = 64 (once per used row block), bias 4 * 16 = 64, gamma
  // 4 * 16 = 64, residual 16. The intermediate staging is 1 * 4 = 4 tiles.
  EmissionInputs in = inputs(/*mTiles=*/4, /*nTiles=*/4, /*kTiles=*/4);
  in.fused = true;
  in.fusedMlp = true;
  in.hTiles = 4;
  const EmissionPlan plan = estimateEmission(in);
  EXPECT_EQ(plan.mapping, kBlockedReuseMapping);
  EXPECT_EQ(plan.coresUsed, 16);
  EXPECT_EQ(plan.maxTilesPerCore, 1);
  EXPECT_EQ(plan.fusedStageTiles, 4);
  EXPECT_TRUE(plan.fusedFeasible);
  const int64_t readTiles = 64 + 256 + 64 + 64 + 64 + 16;
  EXPECT_EQ(plan.dramReadBytes, readTiles * 2048);
  EXPECT_EQ(plan.dramWriteBytes, 16 * 2048);
}

TEST(EmissionEstimatorTest, FusedMlpFootprintInfeasibleForLargeHiddenWidth) {
  // The large MLP: 64 x 128 output tiles over the 8 x 8 grid gives 8 row block
  // tiles and 16 column block tiles per core; the 128 x 384 first weight block
  // is 49152 tiles on its own. The intermediate staging is 8 * 384 = 3072
  // tiles, so the footprint is far past the 700-tile budget.
  EmissionInputs in = inputs(/*mTiles=*/64, /*nTiles=*/128, /*kTiles=*/128);
  in.fused = true;
  in.fusedMlp = true;
  in.hTiles = 384;
  const EmissionPlan plan = estimateEmission(in);
  EXPECT_EQ(plan.mapping, kBlockedReuseMapping);
  EXPECT_EQ(plan.fusedStageTiles, 3072);
  EXPECT_FALSE(plan.fusedFeasible);
}

TEST(EmissionEstimatorTest, CoresAreCappedByOutputTiles) {
  // One output tile cannot use more than one core even on a 64-core grid, and
  // the blocked assignment puts it on one used row block and one used column
  // block: one activation tile plus one weight tile.
  const EmissionPlan plan =
      estimateEmission(inputs(/*mTiles=*/1, /*nTiles=*/1, /*kTiles=*/1));
  EXPECT_EQ(plan.mapping, kBlockedReuseMapping);
  EXPECT_EQ(plan.coresUsed, 1);
  EXPECT_EQ(plan.maxTilesPerCore, 1);
  EXPECT_EQ(plan.dramReadBytes, 2 * 2048);
  EXPECT_EQ(plan.dramWriteBytes, 2048);
  EXPECT_EQ(plan.fusedStageTiles, 1);
  EXPECT_TRUE(plan.fusedFeasible);
}

TEST(EmissionEstimatorTest, UnevenSplitBlocksAndStagesOneContiguousRange) {
  // 3 x 5 output tiles over a 2 x 2 grid: row blocks of 2 and 1, column blocks
  // of 3 and 2. The reads are A = 4 * 2 * (2 + 1) = 24 tiles and the
  // multicast B = 4 * 1 * (3 + 2) = 20 tiles; the deepest block is 2 * 3.
  EmissionInputs in = inputs(/*mTiles=*/3, /*nTiles=*/5, /*kTiles=*/4,
                             /*gridY=*/2, /*gridX=*/2);
  const EmissionPlan plan = estimateEmission(in);
  EXPECT_EQ(plan.mapping, kBlockedReuseMapping);
  EXPECT_EQ(plan.coresUsed, 4);
  EXPECT_EQ(plan.maxTilesPerCore, 6);
  EXPECT_EQ(plan.dramReadBytes, (24 + 20) * 2048);
  EXPECT_EQ(plan.dramWriteBytes, 15 * 2048);
  EXPECT_EQ(plan.fusedStageTiles, 2 * 4);
  EXPECT_TRUE(plan.fusedFeasible);
}

TEST(EmissionEstimatorTest, AsymmetricGridShowsTheMulticastSaving) {
  // 2 x 8 output tiles over a 2 x 4 grid: 2 row blocks of 1 and 4 column
  // blocks of 2. A reads = 2 * 4 * (1 * 2) = 16 tiles. Without multicast the
  // weight block would be read once per used row block, 2 * 2 * 8 = 32 tiles;
  // with the multicast it is read once per used column block, 2 * 1 * 8 = 16,
  // so the read total is 32 tiles instead of 48.
  const EmissionPlan plan = estimateEmission(
      inputs(/*mTiles=*/2, /*nTiles=*/8, /*kTiles=*/2, /*gridY=*/2,
             /*gridX=*/4));
  EXPECT_EQ(plan.mapping, kBlockedReuseMapping);
  EXPECT_EQ(plan.coresUsed, 8);
  EXPECT_EQ(plan.maxTilesPerCore, 2);
  EXPECT_EQ(plan.dramReadBytes, (16 + 16) * 2048);
  EXPECT_EQ(plan.dramWriteBytes, 16 * 2048);
}

TEST(EmissionEstimatorTest, SingleRowBlockReadsTheWeightDirectly) {
  // One row block of tiles has no weight replication, so the blocked-reuse
  // reader stays unicast: no multicast is emitted and the formulas count the
  // weight once per used column block either way.
  const BlockAssignment block =
      planBlockAssignment(inputs(/*mTiles=*/1, /*nTiles=*/4, /*kTiles=*/2));
  EXPECT_EQ(block.usedRows, 1);
  EXPECT_FALSE(block.weightMulticast);
  const EmissionPlan plan = estimateEmission(
      inputs(/*mTiles=*/1, /*nTiles=*/4, /*kTiles=*/2));
  EXPECT_EQ(plan.mapping, kBlockedReuseMapping);
  EXPECT_EQ(plan.dramReadBytes, (8 + 8) * 2048);
  EXPECT_EQ(plan.dramWriteBytes, 4 * 2048);
}

TEST(EmissionEstimatorTest, LargeExampleIsInfeasibleForFusion) {
  // examples/rmsnorm_matmul.mlir: 64 x 384 output tiles and 128 K tiles, a
  // fused plan. The blocked-reuse buffers need (8 + 48) * 128 + 2 = 7170 tiles,
  // far above the 700-tile n150 budget, so the mapping stays output-tile-ranges
  // and the fused allocation needs 2 * 256 staged tiles + 256 weight and gamma
  // tiles + 3 fixed tiles = 771, also above the budget.
  EmissionInputs in = inputs(/*mTiles=*/64, /*nTiles=*/384, /*kTiles=*/128);
  in.fused = true;
  const EmissionPlan plan = estimateEmission(in);
  EXPECT_EQ(plan.mapping, kOutputTileRangesMapping);
  EXPECT_EQ(plan.coresUsed, 64);
  EXPECT_EQ(plan.maxTilesPerCore, 384);
  EXPECT_EQ(plan.dramReadBytes, 12884901888LL);
  EXPECT_EQ(plan.dramWriteBytes, 50331648);
  EXPECT_EQ(plan.fusedStageTiles, 256);
  EXPECT_FALSE(plan.fusedFeasible);
}

TEST(EmissionEstimatorTest, LargeUnfusedPlanFallsBackToOutputTileRanges) {
  // The same geometry planned unfused: the blocked-reuse blocks do not fit the
  // budget, so the selection keeps the unicast range mapping.
  const EmissionPlan plan = estimateEmission(
      inputs(/*mTiles=*/64, /*nTiles=*/384, /*kTiles=*/128));
  EXPECT_EQ(plan.mapping, kOutputTileRangesMapping);
  EXPECT_EQ(plan.coresUsed, 64);
  EXPECT_EQ(plan.maxTilesPerCore, 384);
  EXPECT_EQ(plan.dramReadBytes, 12884901888LL);
  EXPECT_EQ(plan.dramWriteBytes, 50331648);
}

TEST(EmissionEstimatorTest, FusedVariantCanBeInfeasible) {
  // A tiny circular-buffer budget (10 tiles) cannot hold a fused two-row stage.
  EmissionInputs in = inputs(/*mTiles=*/3, /*nTiles=*/5, /*kTiles=*/4,
                             /*gridY=*/2, /*gridX=*/2);
  in.cbBudgetBytes = 10 * 2048;
  const EmissionPlan plan = estimateEmission(in);
  EXPECT_EQ(plan.mapping, kOutputTileRangesMapping);
  EXPECT_FALSE(plan.fusedFeasible);
}

TEST(EmissionEstimatorTest, DegenerateInputsProduceZeroes) {
  // No output tiles and no grid are both guarded, not divided by.
  EmissionInputs noTiles = inputs(/*mTiles=*/0, /*nTiles=*/0, /*kTiles=*/0);
  const EmissionPlan emptyPlan = estimateEmission(noTiles);
  EXPECT_EQ(emptyPlan.mapping, kOutputTileRangesMapping);
  EXPECT_EQ(emptyPlan.coresUsed, 0);
  EXPECT_EQ(emptyPlan.maxTilesPerCore, 0);
  EXPECT_EQ(emptyPlan.dramReadBytes, 0);
  EXPECT_FALSE(emptyPlan.fusedFeasible);

  EmissionInputs noGrid = inputs(/*mTiles=*/2, /*nTiles=*/2, /*kTiles=*/2,
                                 /*gridY=*/8, /*gridX=*/8);
  noGrid.gridY = 0;
  const EmissionPlan noGridPlan = estimateEmission(noGrid);
  EXPECT_EQ(noGridPlan.mapping, kOutputTileRangesMapping);
  EXPECT_EQ(noGridPlan.coresUsed, 0);
  EXPECT_FALSE(noGridPlan.fusedFeasible);
}
