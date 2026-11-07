//===- GridPlannerTest.cpp --------------------------------------*- C++ -*-===//
//
// Unit tests for the core-grid planner. The grid comes from the generated
// device table, so these tests cover per-device grids, row-major sharding, the
// default CB depth, the CB-budget invariant, and the draft conversion seam.
//
//   1. DeviceGrids         - n150 plans 8x8, n300 plans 8x7, p100a plans 12x10.
//   2. RowMajorGridPlan    - [1,2048,4096] over n150 gives per-core
//                            [1,256,4096], default depth 2, inside the budget.
//   3. MaxCBDepthFormula   - floor of budget / (numCBs * tileBytes), clamped.
//   4. ClampedDepth        - a large CB count clamps the depth below two tiles.
//   5. OversizedNoOverflow - a huge activation still gets a feasible reduced
//                            depth, never an overflow.
//   6. DraftConversion     - the draft carries the device tile size and the
//                            tt-metal c_0/c_1/c_16 convention.
//
//===----------------------------------------------------------------------===//

#include "GridPlanner.h"

#include "Model.h"

#include "tensixflow/Generated/DeviceSpecs.h"

#include "gtest/gtest.h"

#include <vector>

using namespace mlir::tensixflow::scheduler;
using mlir::tensixflow::device::DeviceSpec;

namespace {

const DeviceSpec &n150() {
  static const DeviceSpec *spec = mlir::tensixflow::device::lookup("n150");
  return *spec;
}
const DeviceSpec &n300() {
  static const DeviceSpec *spec = mlir::tensixflow::device::lookup("n300");
  return *spec;
}
const DeviceSpec &p100a() {
  static const DeviceSpec *spec = mlir::tensixflow::device::lookup("p100a");
  return *spec;
}

TensorShape canonicalActivation() {
  return TensorShape{"activation", {1, 2048, 4096}, "bf16"};
}

TEST(GridPlannerTest, DeviceGrids) {
  // The planner takes the compute grid from the device, not from a constant.
  GridPlan onN150 = planGrid(canonicalActivation(), n150());
  EXPECT_EQ(onN150.grid.gridY, 8);
  EXPECT_EQ(onN150.grid.gridX, 8);

  GridPlan onN300 = planGrid(canonicalActivation(), n300());
  EXPECT_EQ(onN300.grid.gridY, 7);
  EXPECT_EQ(onN300.grid.gridX, 8);

  GridPlan onP100A = planGrid(canonicalActivation(), p100a());
  EXPECT_EQ(onP100A.grid.gridY, 10);
  EXPECT_EQ(onP100A.grid.gridX, 12);
}

TEST(GridPlannerTest, RowMajorGridPlan) {
  const DeviceSpec &device = n150();
  GridPlan plan = planGrid(canonicalActivation(), device);

  EXPECT_EQ(plan.sharding, "row-major");
  // 2048 rows / 8 Y-cores = 256 rows per core; batch and K untouched.
  const std::vector<int64_t> expectedDims = {1, 256, 4096};
  EXPECT_EQ(plan.perCoreShard.dims, expectedDims);
  EXPECT_EQ(plan.perCoreShard.name, "activation");
  EXPECT_EQ(plan.perCoreShard.dtype, "bf16");

  // Default depth is the tt-metal example depth of two tiles per buffer.
  EXPECT_EQ(plan.numCBs, kDefaultNumCBs);
  EXPECT_EQ(plan.cbDepth, mlir::tensixflow::device::kDefaultCBDepthTiles);
  EXPECT_EQ(plan.cbDepth, 2);
  EXPECT_EQ(plan.cbBytes, kDefaultNumCBs * plan.cbDepth * device.tileBytes);
  EXPECT_EQ(plan.cbBytes, 12288);

  // The footprint never exceeds the per-core CB budget.
  const int64_t budget =
      device.cbBudget(mlir::tensixflow::device::kDefaultL1ReserveBytes);
  EXPECT_TRUE(plan.budgetFeasible);
  EXPECT_LE(plan.cbBytes, budget);
}

TEST(GridPlannerTest, MaxCBDepthFormula) {
  const DeviceSpec &device = n150();
  const int64_t budget =
      device.cbBudget(mlir::tensixflow::device::kDefaultL1ReserveBytes);
  EXPECT_EQ(budget, 1433600);

  EXPECT_EQ(maxCBDepth(device, 3), budget / (3 * device.tileBytes));
  // Floor semantics: 1433600 / 6144 = 233 for the default three CBs.
  EXPECT_EQ(maxCBDepth(device, 3), 233);
  EXPECT_LE(3 * maxCBDepth(device, 3) * device.tileBytes, budget);

  // The feasible maximum stays inside the budget for every CB count.
  for (int64_t numCBs = 1; numCBs <= 16; ++numCBs) {
    const int64_t depth = maxCBDepth(device, numCBs);
    EXPECT_GE(depth, 1);
    EXPECT_LE(numCBs * depth * device.tileBytes, budget);
  }

  // Degenerate inputs never divide by zero nor return a negative depth.
  EXPECT_EQ(maxCBDepth(device, 0), 0);
  EXPECT_EQ(maxCBDepth(device, -1), 0);
  EXPECT_EQ(maxCBDepth(device, 1000), 0);
}

TEST(GridPlannerTest, ClampedDepth) {
  const DeviceSpec &device = n150();
  const int64_t budget =
      device.cbBudget(mlir::tensixflow::device::kDefaultL1ReserveBytes);

  // 700 CBs * 2048 bytes = 1433600 bytes = exactly the budget, so only one tile
  // per buffer fits and the two-tile default clamps down to one.
  GridPlan plan = planGrid(canonicalActivation(), device, /*numCBs=*/700);
  EXPECT_EQ(plan.cbDepth, 1);
  EXPECT_EQ(plan.cbBytes, 700 * 2048);
  EXPECT_EQ(plan.cbBytes, budget);
  EXPECT_TRUE(plan.budgetFeasible);

  // p150a has a larger usable L1 than n150, so its feasible maximum depth is
  // higher for the same CB count.
  const DeviceSpec *p150a = mlir::tensixflow::device::lookup("p150a");
  ASSERT_NE(p150a, nullptr);
  EXPECT_GT(maxCBDepth(*p150a, kDefaultNumCBs),
            maxCBDepth(device, kDefaultNumCBs));
}

TEST(GridPlannerTest, OversizedNoOverflow) {
  const DeviceSpec &device = n150();

  // A huge activation still yields a feasible default depth; the sharding
  // handles the shape and the CB budget is shape independent.
  TensorShape huge{"huge", {64, 1 << 20, 1 << 20}, "bf16"};
  GridPlan plan = planGrid(huge, device);
  EXPECT_TRUE(plan.budgetFeasible);
  EXPECT_LE(plan.cbBytes,
            device.cbBudget(mlir::tensixflow::device::kDefaultL1ReserveBytes));
  // Row-major sharding still applies (ceil(2^20 / 8) = 2^17 rows per core).
  const std::vector<int64_t> expectedDims = {64, 1 << 17, 1 << 20};
  EXPECT_EQ(plan.perCoreShard.dims, expectedDims);

  // A CB count that cannot fit even one tile per buffer is flagged infeasible
  // with depth 0: not a crash, not a silent overflow.
  GridPlan starved = planGrid(huge, device, /*numCBs=*/1000);
  EXPECT_EQ(starved.cbDepth, 0);
  EXPECT_EQ(starved.cbBytes, 0);
  EXPECT_FALSE(starved.budgetFeasible);

  // numCBs == 0 must not divide by zero.
  GridPlan none = planGrid(huge, device, /*numCBs=*/0);
  EXPECT_EQ(none.cbDepth, 0);
  EXPECT_FALSE(none.budgetFeasible);
}

TEST(GridPlannerTest, DraftConversion) {
  const DeviceSpec &device = n150();
  GridPlan plan = planGrid(canonicalActivation(), device);

  PlanDraft draft = toPlanDraft(plan);
  EXPECT_EQ(draft.grid.gridY, 8);
  EXPECT_EQ(draft.grid.gridX, 8);
  EXPECT_EQ(draft.activationSharding, "row-major");

  // The default three-CB pipeline maps to the tt-metal in0/in1/out convention
  // and uses the device tile size.
  ASSERT_EQ(draft.cbSpecs.size(), 3u);
  EXPECT_EQ(draft.cbSpecs[0].index, kCBIn0);
  EXPECT_EQ(draft.cbSpecs[1].index, kCBIn1);
  EXPECT_EQ(draft.cbSpecs[2].index, kCBOut);
  for (const plan::CBSpec &cb : draft.cbSpecs) {
    EXPECT_EQ(cb.depth, plan.cbDepth);
    EXPECT_EQ(cb.tileBytes, device.tileBytes);
  }

  // The draft assembles into a schema Plan.
  plan::Plan report = toPlan(draft);
  EXPECT_EQ(report.layout.activationSharding, "row-major");
  EXPECT_EQ(report.shardGrid.gridY, 8);
  EXPECT_EQ(report.cbSpecs.size(), 3u);
}

} // namespace
