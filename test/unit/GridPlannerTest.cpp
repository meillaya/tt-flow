//===- GridPlannerTest.cpp --------------------------------------*- C++ -*-===//
//
// Unit tests for the TensixFlow core-grid planner (todo 26):
//   1. L1Constants        — the shared L1-budget constants (single source of
//                           truth in L1Model.h): 1 MiB total, 64 KiB reserve,
//                           960 KiB CB budget, 2048 bytes/tile, and consistency
//                           with the scheduler model's TileShape.
//   2. RowMajorGridPlan   — the user's [1,2048,4096] activation over an 8x8 grid
//                           -> row-major, per-core [1,256,4096], CB budget
//                           feasible with depth >= 2.
//   3. OversizedNoOverflow — a huge activation still gets a reduced feasible
//                           depth (never an overflow); a degenerate numCBs that
//                           cannot fit one tile per CB is flagged infeasible.
//   4. MaxCBDepthFormula  — maxCBDepth matches floor(budget / (n * tileBytes))
//                           and never exceeds the budget.
//   5. DraftConversion    — toPlanDraft populates the grid + CB specs via the
//                           tt-metal c_0/c_1/c_16 convention (the todo-30 seam).
//
//===----------------------------------------------------------------------===//

#include "GridPlanner.h"

#include "L1Model.h"
#include "Model.h"

#include "gtest/gtest.h"

#include <vector>

using namespace mlir::tensixflow::scheduler;

namespace {

TEST(GridPlannerTest, L1Constants) {
  EXPECT_EQ(kL1TotalBytes, 1024 * 1024);
  EXPECT_EQ(kL1ReserveBytes, 64 * 1024);
  EXPECT_EQ(kCBShardBudget, 960 * 1024);
  EXPECT_EQ(kCBShardBudget, 983040);
  EXPECT_EQ(kCBShardBudget, kL1TotalBytes - kL1ReserveBytes);
  EXPECT_EQ(kTileBytes, 2048);

  // The shared tile constant must agree with the scheduler model's TileShape.
  TileShape tile; // defaults to 32x32
  EXPECT_EQ(kTileBytes, tile.bytes(bytesPerElem("bf16")));
}

TEST(GridPlannerTest, RowMajorGridPlan) {
  TensorShape activation{"activation", {1, 2048, 4096}, "bf16"};
  plan::ShardGrid grid{8, 8};

  GridPlan plan = planGrid(activation, grid, bytesPerElem("bf16"));

  EXPECT_EQ(plan.grid.gridY, 8);
  EXPECT_EQ(plan.grid.gridX, 8);
  EXPECT_EQ(plan.sharding, "row-major");

  // 2048 rows / 8 Y-cores = 256 rows per core; batch + K untouched.
  const std::vector<int64_t> expectedDims = {1, 256, 4096};
  EXPECT_EQ(plan.perCoreShard.dims, expectedDims);
  EXPECT_EQ(plan.perCoreShard.name, "activation");
  EXPECT_EQ(plan.perCoreShard.dtype, "bf16");

  // CB budget is feasible and never exceeds the per-core shard budget.
  EXPECT_TRUE(plan.budgetFeasible);
  EXPECT_LE(plan.cbBytes, kCBShardBudget);
  EXPECT_GE(plan.cbDepth, 2);
  EXPECT_EQ(plan.numCBs, 3);
  EXPECT_EQ(plan.cbBytes, plan.numCBs * plan.cbDepth * kTileBytes);
}

TEST(GridPlannerTest, OversizedNoOverflow) {
  plan::ShardGrid grid{8, 8};

  // A huge activation still yields a feasible reduced depth, never an overflow:
  // the planner answers an oversized request with "reduce depth", documented as
  // the split-K / reduce-depth fallback.
  TensorShape huge{"huge", {64, 1 << 20, 1 << 20}, "bf16"};
  GridPlan plan = planGrid(huge, grid, bytesPerElem("bf16"));
  EXPECT_LE(plan.cbBytes, kCBShardBudget);
  EXPECT_EQ(plan.cbBytes, plan.numCBs * plan.cbDepth * kTileBytes);
  // Row-major sharding still applies (ceil(2^20 / 8) = 2^17 rows per core).
  const std::vector<int64_t> expectedDims = {64, 1 << 17, 1 << 20};
  EXPECT_EQ(plan.perCoreShard.dims, expectedDims);

  // A degenerate CB count that cannot fit even one tile per CB is flagged
  // infeasible with depth 0 — not a crash, not a silent overflow.
  GridPlan starved = planGrid(huge, grid, bytesPerElem("bf16"),
                              /*numCBs=*/1000);
  EXPECT_EQ(starved.cbDepth, 0);
  EXPECT_EQ(starved.cbBytes, 0);
  EXPECT_FALSE(starved.budgetFeasible);

  // numCBs == 0 must not divide by zero.
  GridPlan none = planGrid(huge, grid, bytesPerElem("bf16"), /*numCBs=*/0);
  EXPECT_EQ(none.cbDepth, 0);
  EXPECT_FALSE(none.budgetFeasible);
}

TEST(GridPlannerTest, MaxCBDepthFormula) {
  EXPECT_EQ(maxCBDepth(3), kCBShardBudget / (3 * kTileBytes));
  EXPECT_LE(3 * maxCBDepth(3) * kTileBytes, kCBShardBudget);

  // Floor semantics: 983040 / 6144 = 160 exactly for the default 3 CBs.
  EXPECT_EQ(maxCBDepth(3), 160);
  // Degenerate inputs never divide by zero nor return a negative depth.
  EXPECT_EQ(maxCBDepth(0), 0);
  EXPECT_EQ(maxCBDepth(-1), 0);
}

TEST(GridPlannerTest, DraftConversion) {
  TensorShape activation{"activation", {1, 2048, 4096}, "bf16"};
  GridPlan plan = planGrid(activation, plan::ShardGrid{8, 8},
                           bytesPerElem("bf16"));

  PlanDraft draft = toPlanDraft(plan, bytesPerElem("bf16"));

  EXPECT_EQ(draft.grid.gridY, 8);
  EXPECT_EQ(draft.grid.gridX, 8);
  EXPECT_EQ(draft.activationSharding, "row-major");

  // The default 3-CB pipeline maps to the tt-metal in0/in1/out convention.
  ASSERT_EQ(draft.cbSpecs.size(), 3u);
  EXPECT_EQ(draft.cbSpecs[0].index, 0);
  EXPECT_EQ(draft.cbSpecs[1].index, 1);
  EXPECT_EQ(draft.cbSpecs[2].index, 16);
  for (const plan::CBSpec &cb : draft.cbSpecs) {
    EXPECT_EQ(cb.depth, plan.cbDepth);
    EXPECT_EQ(cb.tileBytes, kTileBytes);
  }

  // The draft assembles into a schema Plan (the todo-30 seam).
  plan::Plan report = toPlan(draft);
  EXPECT_EQ(report.layout.activationSharding, "row-major");
  EXPECT_EQ(report.shardGrid.gridY, 8);
  EXPECT_EQ(report.cbSpecs.size(), 3u);
}

} // namespace
