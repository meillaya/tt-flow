//===- WeightPlannerTest.cpp ------------------------------------*- C++ -*-===//
//
// Unit tests for the TensixFlow weight-strategy planner (todo 27):
//   1. LargeWeightColumnShardMulticast — the user's [4096,12288] bf16 weight over
//                           an 8x8 grid -> Multicast, multicast_bytes =
//                           4096*12288*2 = 100663296, per-core N = ceilDiv(12288,8)
//                           = 1536, reported as "multicast".
//   2. SmallWeightReplicates — a [256,256] bf16 weight (131072 bytes <= 983040)
//                           -> Replicate, multicast_bytes = 0, per-core weight ==
//                           full weight, reported as "replicate".
//   3. BoundaryReplicateThreshold — a weight exactly == kCBShardBudget replicates;
//                           one element-row over multicasts.
//   4. DegenerateGridNoCrash — gridX == 1 leaves per-core N == full N (no crash).
//   5. EnumStrings — toString covers replicate/shard/multicast (report seam).
//
//===----------------------------------------------------------------------===//

#include "WeightPlanner.h"

#include "L1Model.h"
#include "Model.h"

#include "gtest/gtest.h"

#include <vector>

using namespace mlir::tensixflow::scheduler;

namespace {

TEST(WeightPlannerTest, LargeWeightColumnShardMulticast) {
  TensorShape weight{"weight", {4096, 12288}, "bf16"};
  plan::ShardGrid grid{8, 8};

  WeightPlan plan = planWeight(weight, grid, bytesPerElem("bf16"));

  // 4096*12288*2 = 100663296 bytes >> kCBShardBudget (983040) -> multicast.
  EXPECT_EQ(plan.strategy, WeightStrategy::Multicast);
  EXPECT_EQ(plan.multicastBytes, 4096 * 12288 * 2);
  EXPECT_EQ(plan.multicastBytes, 100663296);
  EXPECT_GT(plan.multicastBytes, 0);

  // Column-shard N across gridX: ceilDiv(12288, 8) = 1536 per-core columns;
  // K is untouched.
  const std::vector<int64_t> expectedDims = {4096, 1536};
  EXPECT_EQ(plan.perCoreWeight.dims, expectedDims);
  EXPECT_EQ(plan.perCoreWeight.dims.back(), 1536);
  EXPECT_EQ(plan.perCoreWeight.name, "weight");
  EXPECT_EQ(plan.perCoreWeight.dtype, "bf16");

  // Report seam: the combined shard+multicast decision reports as "multicast".
  EXPECT_EQ(toString(plan.strategy), "multicast");
}

TEST(WeightPlannerTest, SmallWeightReplicates) {
  TensorShape weight{"weight", {256, 256}, "bf16"};
  plan::ShardGrid grid{8, 8};

  WeightPlan plan = planWeight(weight, grid, bytesPerElem("bf16"));

  // 256*256*2 = 131072 bytes <= kCBShardBudget (983040) -> replicate.
  EXPECT_EQ(256 * 256 * 2, 131072);
  EXPECT_LE(131072, kCBShardBudget);
  EXPECT_EQ(plan.strategy, WeightStrategy::Replicate);
  EXPECT_EQ(plan.multicastBytes, 0);

  // Every core holds a full copy: the per-core weight is the whole weight.
  const std::vector<int64_t> expectedDims = {256, 256};
  EXPECT_EQ(plan.perCoreWeight.dims, expectedDims);

  EXPECT_EQ(toString(plan.strategy), "replicate");
}

TEST(WeightPlannerTest, BoundaryReplicateThreshold) {
  plan::ShardGrid grid{8, 8};
  const int bpe = bytesPerElem("bf16");

  // Exactly == kCBShardBudget replicates: 480*1024*2 = 983040 == budget.
  TensorShape exact{"exact", {480, 1024}, "bf16"};
  ASSERT_EQ(exact.bytes(bpe), kCBShardBudget);
  WeightPlan atBudget = planWeight(exact, grid, bpe);
  EXPECT_EQ(atBudget.strategy, WeightStrategy::Replicate);
  EXPECT_EQ(atBudget.multicastBytes, 0);

  // One element-row over multicasts: 481*1024*2 = 985088 > budget.
  TensorShape over{"over", {481, 1024}, "bf16"};
  ASSERT_GT(over.bytes(bpe), kCBShardBudget);
  WeightPlan overBudget = planWeight(over, grid, bpe);
  EXPECT_EQ(overBudget.strategy, WeightStrategy::Multicast);
  EXPECT_EQ(overBudget.multicastBytes, over.bytes(bpe));
  // per-core N = ceilDiv(1024, 8) = 128.
  EXPECT_EQ(overBudget.perCoreWeight.dims.back(), 128);
}

TEST(WeightPlannerTest, DegenerateGridNoCrash) {
  const int bpe = bytesPerElem("bf16");

  // gridX == 1: column-sharding leaves per-core N == full N (no division issue).
  TensorShape weight{"weight", {4096, 12288}, "bf16"};
  plan::ShardGrid singleCol{8, 1};
  WeightPlan plan = planWeight(weight, singleCol, bpe);
  EXPECT_EQ(plan.strategy, WeightStrategy::Multicast);
  EXPECT_EQ(plan.perCoreWeight.dims.back(), 12288);

  // gridX == 0: guarded — the weight shape is left unchanged, no crash.
  plan::ShardGrid zeroX{8, 0};
  WeightPlan guarded = planWeight(weight, zeroX, bpe);
  EXPECT_EQ(guarded.strategy, WeightStrategy::Multicast);
  EXPECT_EQ(guarded.perCoreWeight.dims.back(), 12288);
}

TEST(WeightPlannerTest, EnumStrings) {
  EXPECT_EQ(toString(WeightStrategy::Replicate), "replicate");
  EXPECT_EQ(toString(WeightStrategy::Shard), "shard");
  EXPECT_EQ(toString(WeightStrategy::Multicast), "multicast");
}

} // namespace
