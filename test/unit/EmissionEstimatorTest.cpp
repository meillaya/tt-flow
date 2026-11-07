//===- EmissionEstimatorTest.cpp --------------------------------*- C++ -*-===//
//
// Unit tests for the emitted-path accounting: the work split and traffic the
// emitted TT-Metalium program will produce. The expected values are computed by
// hand from the documented formulas, not read back from the implementation.
//
//===----------------------------------------------------------------------===//

#include "EmissionEstimator.h"

#include "gtest/gtest.h"

using mlir::tensixflow::scheduler::EmissionInputs;
using mlir::tensixflow::scheduler::EmissionPlan;
using mlir::tensixflow::scheduler::estimateEmission;
using mlir::tensixflow::scheduler::kOutputTileRangesMapping;

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

TEST(EmissionEstimatorTest, MatmulExampleValues) {
  // examples/matmul.mlir: M=256, N=1024, K=512 on the n150 8x8 grid, so
  // 8 x 32 output tiles and 16 K tiles.
  const EmissionPlan plan = estimateEmission(inputs(/*mTiles=*/8, /*nTiles=*/32,
                                                    /*kTiles=*/16));
  EXPECT_EQ(plan.mapping, kOutputTileRangesMapping);
  EXPECT_EQ(plan.coresUsed, 64);
  EXPECT_EQ(plan.maxTilesPerCore, 4);
  EXPECT_EQ(plan.dramReadBytes, 256 * 16 * 2 * 2048);
  EXPECT_EQ(plan.dramWriteBytes, 256 * 2048);
  EXPECT_EQ(plan.fusedStageTiles, 32);
  EXPECT_TRUE(plan.fusedFeasible);
}

TEST(EmissionEstimatorTest, MatmulSmallExampleValues) {
  // examples/matmul_small.mlir: M=128, N=128, K=64, so 4 x 4 output tiles and
  // 2 K tiles. The grid has 64 cores but only 16 output tiles exist.
  const EmissionPlan plan = estimateEmission(inputs(/*mTiles=*/4, /*nTiles=*/4,
                                                    /*kTiles=*/2));
  EXPECT_EQ(plan.coresUsed, 16);
  EXPECT_EQ(plan.maxTilesPerCore, 1);
  EXPECT_EQ(plan.dramReadBytes, 16 * 2 * 2 * 2048);
  EXPECT_EQ(plan.dramWriteBytes, 16 * 2048);
  EXPECT_EQ(plan.fusedStageTiles, 4);
  EXPECT_TRUE(plan.fusedFeasible);
}

TEST(EmissionEstimatorTest, CoresAreCappedByOutputTiles) {
  // One output tile cannot use more than one core even on a 64-core grid.
  const EmissionPlan plan =
      estimateEmission(inputs(/*mTiles=*/1, /*nTiles=*/1, /*kTiles=*/1));
  EXPECT_EQ(plan.coresUsed, 1);
  EXPECT_EQ(plan.maxTilesPerCore, 1);
  EXPECT_EQ(plan.dramReadBytes, 2 * 2048);
  EXPECT_EQ(plan.dramWriteBytes, 2048);
  EXPECT_EQ(plan.fusedStageTiles, 1);
  EXPECT_TRUE(plan.fusedFeasible);
}

TEST(EmissionEstimatorTest, UnevenSplitFillsTheCoresAndStagesTwoRows) {
  // 15 output tiles over 4 cores: max tiles per core is ceil(15 / 4) = 4, which
  // can straddle a row boundary, so two rows are staged per core.
  EmissionInputs in = inputs(/*mTiles=*/3, /*nTiles=*/5, /*kTiles=*/4,
                             /*gridY=*/2, /*gridX=*/2);
  const EmissionPlan plan = estimateEmission(in);
  EXPECT_EQ(plan.coresUsed, 4);
  EXPECT_EQ(plan.maxTilesPerCore, 4);
  EXPECT_EQ(plan.dramReadBytes, 15 * 4 * 2 * 2048);
  EXPECT_EQ(plan.dramWriteBytes, 15 * 2048);
  EXPECT_EQ(plan.fusedStageTiles, 2 * 4);
  EXPECT_TRUE(plan.fusedFeasible);
}

TEST(EmissionEstimatorTest, LargeExampleIsInfeasibleForFusion) {
  // examples/rmsnorm_matmul.mlir: 64 x 384 output tiles and 128 K tiles. The
  // fused allocation needs 2 * 256 staged tiles + 256 weight and gamma tiles +
  // 3 fixed tiles = 771, above the 700-tile n150 budget.
  const EmissionPlan plan = estimateEmission(
      inputs(/*mTiles=*/64, /*nTiles=*/384, /*kTiles=*/128));
  EXPECT_EQ(plan.coresUsed, 64);
  EXPECT_EQ(plan.maxTilesPerCore, 384);
  EXPECT_EQ(plan.fusedStageTiles, 256);
  EXPECT_FALSE(plan.fusedFeasible);
}

TEST(EmissionEstimatorTest, FusedVariantCanBeInfeasible) {
  // A tiny circular-buffer budget (10 tiles) cannot hold a fused two-row stage.
  EmissionInputs in = inputs(/*mTiles=*/3, /*nTiles=*/5, /*kTiles=*/4,
                             /*gridY=*/2, /*gridX=*/2);
  in.cbBudgetBytes = 10 * 2048;
  const EmissionPlan plan = estimateEmission(in);
  EXPECT_FALSE(plan.fusedFeasible);
}

TEST(EmissionEstimatorTest, DegenerateInputsProduceZeroes) {
  // No output tiles and no grid are both guarded, not divided by.
  EmissionInputs noTiles = inputs(/*mTiles=*/0, /*nTiles=*/0, /*kTiles=*/0);
  const EmissionPlan emptyPlan = estimateEmission(noTiles);
  EXPECT_EQ(emptyPlan.coresUsed, 0);
  EXPECT_EQ(emptyPlan.maxTilesPerCore, 0);
  EXPECT_EQ(emptyPlan.dramReadBytes, 0);
  EXPECT_FALSE(emptyPlan.fusedFeasible);

  EmissionInputs noGrid = inputs(/*mTiles=*/2, /*nTiles=*/2, /*kTiles=*/2,
                                 /*gridY=*/8, /*gridX=*/8);
  noGrid.gridY = 0;
  const EmissionPlan noGridPlan = estimateEmission(noGrid);
  EXPECT_EQ(noGridPlan.coresUsed, 0);
  EXPECT_FALSE(noGridPlan.fusedFeasible);
}
