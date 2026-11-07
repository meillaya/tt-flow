//===- WeightPlannerTest.cpp ------------------------------------*- C++ -*-===//
//
// Unit tests for the weight-strategy planner. The per-core CB budget and the
// grid come from the generated device table.
//
//   1. LargeWeightColumnShardMulticast - [4096,12288] bf16 exceeds the n150
//                           budget, so it multicasts with per-core N =
//                           ceilDiv(12288, 8) = 1536.
//   2. SmallWeightReplicates           - [256,256] bf16 fits, so it replicates.
//   3. BoundaryReplicateThreshold      - a weight exactly at the budget
//                           replicates; one element row over multicasts.
//   4. PerDeviceGrid                   - p100a's 12 X-cores give per-core N =
//                           ceilDiv(12288, 12) = 1024.
//   5. DegenerateShapeGuarded          - a rank-1 weight leaves dims unchanged.
//   6. EnumStrings                     - replicate/shard/multicast strings.
//
//===----------------------------------------------------------------------===//

#include "WeightPlanner.h"

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
const DeviceSpec &p100a() {
  static const DeviceSpec *spec = mlir::tensixflow::device::lookup("p100a");
  return *spec;
}

int64_t budgetOf(const DeviceSpec &device) {
  return device.cbBudget(mlir::tensixflow::device::kDefaultL1ReserveBytes);
}

TEST(WeightPlannerTest, LargeWeightColumnShardMulticast) {
  TensorShape weight{"weight", {4096, 12288}, "bf16"};

  WeightPlan plan = planWeight(weight, n150(), bytesPerElem("bf16"));

  // 4096*12288*2 = 100663296 bytes, far above the 1433600 byte budget.
  EXPECT_EQ(plan.strategy, WeightStrategy::Multicast);
  EXPECT_EQ(plan.multicastBytes, 4096 * 12288 * 2);
  EXPECT_EQ(plan.multicastBytes, 100663296);
  EXPECT_GT(plan.multicastBytes, 0);
  EXPECT_GT(plan.multicastBytes, budgetOf(n150()));

  // Column-shard N across gridX: ceilDiv(12288, 8) = 1536 per-core columns;
  // K is untouched.
  const std::vector<int64_t> expectedDims = {4096, 1536};
  EXPECT_EQ(plan.perCoreWeight.dims, expectedDims);
  EXPECT_EQ(plan.perCoreWeight.name, "weight");
  EXPECT_EQ(plan.perCoreWeight.dtype, "bf16");

  // The rationale names the device and the budget it exceeded.
  EXPECT_NE(plan.rationale.find("n150"), std::string::npos);
  EXPECT_NE(plan.rationale.find("1433600"), std::string::npos);

  // Report seam: the combined shard+multicast decision reports as "multicast".
  EXPECT_EQ(toString(plan.strategy), "multicast");
}

TEST(WeightPlannerTest, SmallWeightReplicates) {
  TensorShape weight{"weight", {256, 256}, "bf16"};

  WeightPlan plan = planWeight(weight, n150(), bytesPerElem("bf16"));

  // 256*256*2 = 131072 bytes fits the 1433600 byte budget.
  EXPECT_EQ(256 * 256 * 2, 131072);
  EXPECT_LE(131072, budgetOf(n150()));
  EXPECT_EQ(plan.strategy, WeightStrategy::Replicate);
  EXPECT_EQ(plan.multicastBytes, 0);

  // Every core holds a full copy: the per-core weight is the whole weight.
  const std::vector<int64_t> expectedDims = {256, 256};
  EXPECT_EQ(plan.perCoreWeight.dims, expectedDims);
  EXPECT_EQ(toString(plan.strategy), "replicate");
}

TEST(WeightPlannerTest, BoundaryReplicateThreshold) {
  const int bpe = bytesPerElem("bf16");
  const int64_t budget = budgetOf(n150());
  EXPECT_EQ(budget, 1433600);

  // Exactly at the budget replicates: 700*1024*2 = 1433600 == budget.
  TensorShape exact{"exact", {700, 1024}, "bf16"};
  ASSERT_EQ(exact.bytes(bpe), budget);
  WeightPlan atBudget = planWeight(exact, n150(), bpe);
  EXPECT_EQ(atBudget.strategy, WeightStrategy::Replicate);
  EXPECT_EQ(atBudget.multicastBytes, 0);

  // One element row over multicasts: 701*1024*2 = 1435648 > budget.
  TensorShape over{"over", {701, 1024}, "bf16"};
  ASSERT_GT(over.bytes(bpe), budget);
  WeightPlan overBudget = planWeight(over, n150(), bpe);
  EXPECT_EQ(overBudget.strategy, WeightStrategy::Multicast);
  EXPECT_EQ(overBudget.multicastBytes, over.bytes(bpe));
  // per-core N = ceilDiv(1024, 8) = 128.
  EXPECT_EQ(overBudget.perCoreWeight.dims.back(), 128);
}

TEST(WeightPlannerTest, PerDeviceGrid) {
  const int bpe = bytesPerElem("bf16");
  TensorShape weight{"weight", {4096, 12288}, "bf16"};

  // p100a has 12 X-cores on its compute grid: ceilDiv(12288, 12) = 1024.
  WeightPlan onP100A = planWeight(weight, p100a(), bpe);
  EXPECT_EQ(onP100A.strategy, WeightStrategy::Multicast);
  EXPECT_EQ(onP100A.perCoreWeight.dims.back(), 1024);
  EXPECT_EQ(onP100A.perCoreWeight.dims.front(), 4096);
  EXPECT_NE(onP100A.rationale.find("p100a"), std::string::npos);
}

TEST(WeightPlannerTest, DegenerateShapeGuarded) {
  const int bpe = bytesPerElem("bf16");

  // A rank-1 tensor cannot be column-sharded: the shape is left unchanged and
  // the planner does not crash on the missing N dimension.
  TensorShape vector{"vector", {1 << 20}, "bf16"};
  WeightPlan plan = planWeight(vector, n150(), bpe);
  EXPECT_EQ(plan.strategy, WeightStrategy::Multicast);
  const std::vector<int64_t> expectedDims = {1 << 20};
  EXPECT_EQ(plan.perCoreWeight.dims, expectedDims);
}

TEST(WeightPlannerTest, EnumStrings) {
  EXPECT_EQ(toString(WeightStrategy::Replicate), "replicate");
  EXPECT_EQ(toString(WeightStrategy::Shard), "shard");
  EXPECT_EQ(toString(WeightStrategy::Multicast), "multicast");
}

} // namespace
