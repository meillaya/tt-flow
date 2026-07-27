//===- CBAllocatorTest.cpp --------------------------------------*- C++ -*-===//
//
// Unit tests for the TensixFlow circular-buffer allocator (todo 28):
//   1. DefaultPipelineConvention — numCBs == 3 -> CB indices {0, 1, 16} (the
//                           tt-metal in0/in1/out convention), depth =
//                           maxCBDepth(3) = 160 (>= 2), totalBytes =
//                           3*160*2048 = 983040 <= kCBShardBudget, feasible,
//                           every CBSpec.tileBytes == 2048.
//   2. IndicesAlwaysValid — every emitted index satisfies isValidCBIndex (0..31)
//                           across a range of feasible counts.
//   3. OverRangeInfeasible — a request that would run past c_31 is flagged
//                           infeasible and emits no index > 31.
//   4. BudgetNeverExceeded — for numCBs 1..8 the total CB bytes never exceed
//                           kCBShardBudget; depth decreases as numCBs grows but
//                           stays >= 1 while feasible.
//   5. DegenerateGuarded — numCBs <= 0 is guarded: feasible == false, empty
//                           allocation, no crash, no division by zero.
//   6. ReportSeam — toCBSpecs yields the allocation's plan::CBSpec vector, ready
//                           for PlanDraft.cbSpecs (the todo-30 seam).
//
//===----------------------------------------------------------------------===//

#include "CBAllocator.h"

#include "GridPlanner.h"
#include "L1Model.h"
#include "Model.h"

#include "gtest/gtest.h"

#include <vector>

using namespace mlir::tensixflow::scheduler;

namespace {

TEST(CBAllocatorTest, DefaultPipelineConvention) {
  CBAllocation allocation = allocateCBs(3, bytesPerElem("bf16"));

  // The default 3-CB pipeline maps to the tt-metal in0/in1/out convention.
  ASSERT_EQ(allocation.cbs.size(), 3u);
  EXPECT_EQ(allocation.cbs[0].index, kCBIn0);
  EXPECT_EQ(allocation.cbs[1].index, kCBIn1);
  EXPECT_EQ(allocation.cbs[2].index, kCBOut);
  EXPECT_EQ(allocation.cbs[0].index, 0);
  EXPECT_EQ(allocation.cbs[1].index, 1);
  EXPECT_EQ(allocation.cbs[2].index, 16);

  // depth = maxCBDepth(3) = floor(983040 / (3*2048)) = 160 (>= 2).
  EXPECT_EQ(allocation.cbs[0].depth, maxCBDepth(3));
  EXPECT_EQ(allocation.cbs[0].depth, 160);
  EXPECT_GE(allocation.cbs[0].depth, 2);

  // Every CB shares the depth and the canonical bf16 tile size.
  for (const plan::CBSpec &cb : allocation.cbs) {
    EXPECT_EQ(cb.depth, 160);
    EXPECT_EQ(cb.tileBytes, kTileBytes);
    EXPECT_EQ(cb.tileBytes, 2048);
  }

  // totalBytes = 3*160*2048 = 983040 <= kCBShardBudget, feasible.
  EXPECT_EQ(allocation.totalBytes, 3 * 160 * 2048);
  EXPECT_EQ(allocation.totalBytes, 983040);
  EXPECT_LE(allocation.totalBytes, kCBShardBudget);
  EXPECT_TRUE(allocation.feasible);
}

TEST(CBAllocatorTest, IndicesAlwaysValid) {
  // Every emitted index is a valid tt::CBIndex (0..31) across a range of counts.
  for (int64_t numCBs = 1; numCBs <= 8; ++numCBs) {
    CBAllocation allocation = allocateCBs(numCBs, bytesPerElem("bf16"));
    ASSERT_EQ(allocation.cbs.size(), static_cast<size_t>(numCBs));
    for (const plan::CBSpec &cb : allocation.cbs) {
      EXPECT_TRUE(isValidCBIndex(cb.index));
      EXPECT_GE(cb.index, 0);
      EXPECT_LE(cb.index, kMaxCBIndex);
    }
    EXPECT_TRUE(allocation.feasible);
  }
}

TEST(CBAllocatorTest, OverRangeInfeasible) {
  // A request that would run past c_31 (sequential indices 0..numCBs-1) is
  // flagged infeasible and never emits an index > 31, even though the budget
  // alone would still allow a positive depth.
  CBAllocation allocation = allocateCBs(kMaxCBIndex + 2, bytesPerElem("bf16"));
  EXPECT_FALSE(allocation.feasible);
  for (const plan::CBSpec &cb : allocation.cbs) {
    EXPECT_TRUE(isValidCBIndex(cb.index));
    EXPECT_LE(cb.index, kMaxCBIndex);
  }
}

TEST(CBAllocatorTest, BudgetNeverExceeded) {
  // For numCBs 1..8 the total footprint never exceeds the per-core CB budget;
  // depth decreases as numCBs grows but stays >= 1 while feasible.
  int64_t prevDepth = maxCBDepth(1);
  for (int64_t numCBs = 1; numCBs <= 8; ++numCBs) {
    CBAllocation allocation = allocateCBs(numCBs, bytesPerElem("bf16"));
    ASSERT_TRUE(allocation.feasible);
    ASSERT_FALSE(allocation.cbs.empty());
    EXPECT_LE(allocation.totalBytes, kCBShardBudget);
    EXPECT_EQ(allocation.totalBytes,
              numCBs * allocation.cbs[0].depth * kTileBytes);
    EXPECT_GE(allocation.cbs[0].depth, 1);
    // Depth is monotonically non-increasing as more CBs share the budget.
    EXPECT_LE(allocation.cbs[0].depth, prevDepth);
    prevDepth = allocation.cbs[0].depth;
  }
}

TEST(CBAllocatorTest, DegenerateGuarded) {
  // numCBs == 0: guarded — infeasible, empty, no crash, no division by zero.
  CBAllocation none = allocateCBs(0, bytesPerElem("bf16"));
  EXPECT_FALSE(none.feasible);
  EXPECT_TRUE(none.cbs.empty());
  EXPECT_EQ(none.totalBytes, 0);

  // numCBs < 0: likewise guarded.
  CBAllocation negative = allocateCBs(-1, bytesPerElem("bf16"));
  EXPECT_FALSE(negative.feasible);
  EXPECT_TRUE(negative.cbs.empty());
  EXPECT_EQ(negative.totalBytes, 0);
}

TEST(CBAllocatorTest, ReportSeam) {
  CBAllocation allocation = allocateCBs(3, bytesPerElem("bf16"));

  // toCBSpecs yields the allocation's plan::CBSpec vector (PlanDraft.cbSpecs).
  std::vector<plan::CBSpec> specs = toCBSpecs(allocation);
  ASSERT_EQ(specs.size(), 3u);
  EXPECT_EQ(specs[0].index, 0);
  EXPECT_EQ(specs[1].index, 1);
  EXPECT_EQ(specs[2].index, 16);
  for (const plan::CBSpec &cb : specs) {
    EXPECT_EQ(cb.depth, 160);
    EXPECT_EQ(cb.tileBytes, kTileBytes);
  }
}

} // namespace
