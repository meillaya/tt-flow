//===- CBAllocatorTest.cpp --------------------------------------*- C++ -*-===//
//
// Unit tests for the circular-buffer allocator. The depth starts at the
// tt-metal example depth of two tiles per buffer and clamps to the per-core L1
// budget from the generated device table.
//
//   1. DefaultPipelineConvention - numCBs 3 gives indices {0,1,16}, depth 2,
//                                  total 12288 bytes, feasible.
//   2. BudgetInvariant           - for every CB count the footprint stays
//                                  inside the device CB budget and the depth
//                                  matches maxCBDepth.
//   3. RequestedDepthClamped     - an oversized depth request clamps to the
//                                  feasible maximum; a zero request is
//                                  infeasible.
//   4. PerDeviceBudget           - the allocator uses the device usable L1, so
//                                  n150 and p100a get different capacities.
//   5. IndicesAlwaysValid        - every emitted index is c_0..c_31.
//   6. OverRangeInfeasible       - a request past c_31 is infeasible and emits
//                                  no out-of-range index.
//   7. DegenerateGuarded         - numCBs <= 0 is guarded.
//   8. ReportSeam                - toCBSpecs yields the plan::CBSpec vector.
//   9. Fused*                    - the fused six-buffer layout: deterministic
//                                  order, budget invariant, guarded request.
//  10. FusedMlp*                - the fused MLP twelve-buffer layout:
//                                  deterministic order, budget invariant,
//                                  guarded request.
//
//===----------------------------------------------------------------------===//

#include "CBAllocator.h"

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
const DeviceSpec &p100a() {
  static const DeviceSpec *spec = mlir::tensixflow::device::lookup("p100a");
  return *spec;
}

int64_t budgetOf(const DeviceSpec &device) {
  return device.cbBudget(mlir::tensixflow::device::kDefaultL1ReserveBytes);
}

TEST(CBAllocatorTest, DefaultPipelineConvention) {
  CBAllocation allocation = allocateCBs(n150(), kDefaultNumCBs);

  // The default three-CB pipeline maps to the tt-metal in0/in1/out convention.
  ASSERT_EQ(allocation.cbs.size(), 3u);
  EXPECT_EQ(allocation.cbs[0].index, kCBIn0);
  EXPECT_EQ(allocation.cbs[1].index, kCBIn1);
  EXPECT_EQ(allocation.cbs[2].index, kCBOut);
  EXPECT_EQ(allocation.cbs[0].index, 0);
  EXPECT_EQ(allocation.cbs[1].index, 1);
  EXPECT_EQ(allocation.cbs[2].index, 16);

  // Default depth is two tiles per buffer, as in the tt-metal examples.
  EXPECT_EQ(allocation.depth, 2);
  for (const plan::CBSpec &cb : allocation.cbs) {
    EXPECT_EQ(cb.depth, 2);
    EXPECT_EQ(cb.tileBytes, 2048);
  }

  // 3 buffers * 2 tiles * 2048 bytes = 12288 bytes.
  EXPECT_EQ(allocation.totalBytes, 12288);
  EXPECT_LE(allocation.totalBytes, budgetOf(n150()));
  EXPECT_TRUE(allocation.feasible);
}

TEST(CBAllocatorTest, BudgetInvariant) {
  const DeviceSpec &device = n150();
  const int64_t budget = budgetOf(device);

  // For a range of CB counts the allocation stays inside the budget, and the
  // depth is the minimum of the requested depth and the feasible maximum.
  for (int64_t numCBs = 1; numCBs <= 16; ++numCBs) {
    CBAllocation allocation = allocateCBs(device, numCBs);
    ASSERT_TRUE(allocation.feasible) << "numCBs " << numCBs;
    EXPECT_EQ(allocation.cbs.size(), static_cast<size_t>(numCBs));
    // The depth starts at the example depth and only clamps down when the
    // feasible maximum is smaller.
    EXPECT_EQ(allocation.depth, 2);
    EXPECT_LE(allocation.depth, maxCBDepth(device, numCBs));
    EXPECT_LE(allocation.totalBytes, budget);
    EXPECT_EQ(allocation.totalBytes,
              numCBs * allocation.depth * device.tileBytes);
  }
}

TEST(CBAllocatorTest, RequestedDepthClamped) {
  const DeviceSpec &device = n150();

  // An oversized depth request clamps to the feasible maximum.
  CBAllocation huge =
      allocateCBs(device, kDefaultNumCBs, /*requestedDepth=*/1000);
  EXPECT_TRUE(huge.feasible);
  EXPECT_EQ(huge.depth, maxCBDepth(device, kDefaultNumCBs));
  EXPECT_EQ(huge.depth, 233);
  EXPECT_LE(huge.totalBytes, budgetOf(device));

  // A request of one tile is respected; the allocator does not force two.
  CBAllocation shallow =
      allocateCBs(device, kDefaultNumCBs, /*requestedDepth=*/1);
  EXPECT_EQ(shallow.depth, 1);
  EXPECT_TRUE(shallow.feasible);

  // A non-positive depth request is infeasible, not silently defaulted.
  CBAllocation zero =
      allocateCBs(device, kDefaultNumCBs, /*requestedDepth=*/0);
  EXPECT_EQ(zero.depth, 0);
  EXPECT_FALSE(zero.feasible);
}

TEST(CBAllocatorTest, PerDeviceBudget) {
  // p100a has 1572864 usable L1 bytes, so its CB budget is larger than n150's.
  EXPECT_GT(budgetOf(p100a()), budgetOf(n150()));
  EXPECT_EQ(budgetOf(p100a()), 1572864 - 65536);
  EXPECT_EQ(budgetOf(n150()), 1499136 - 65536);

  CBAllocation onP100A = allocateCBs(p100a(), kDefaultNumCBs);
  EXPECT_TRUE(onP100A.feasible);
  EXPECT_EQ(onP100A.depth, 2);

  // Both devices accept the default allocation, but the maximum depth differs
  // with the published L1 size.
  EXPECT_GT(maxCBDepth(p100a(), kDefaultNumCBs),
            maxCBDepth(n150(), kDefaultNumCBs));
}

TEST(CBAllocatorTest, IndicesAlwaysValid) {
  // Every emitted index is a valid tt::CBIndex (0..31) across a range of counts.
  for (int64_t numCBs = 1; numCBs <= 8; ++numCBs) {
    CBAllocation allocation = allocateCBs(n150(), numCBs);
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
  // flagged infeasible and never emits an index above 31.
  CBAllocation allocation =
      allocateCBs(n150(), kMaxCBIndex + 2, /*requestedDepth=*/1);
  EXPECT_FALSE(allocation.feasible);
  for (const plan::CBSpec &cb : allocation.cbs) {
    EXPECT_TRUE(isValidCBIndex(cb.index));
    EXPECT_LE(cb.index, kMaxCBIndex);
  }
}

TEST(CBAllocatorTest, DegenerateGuarded) {
  // numCBs == 0: guarded, infeasible, empty, no crash, no division by zero.
  CBAllocation none = allocateCBs(n150(), 0);
  EXPECT_FALSE(none.feasible);
  EXPECT_TRUE(none.cbs.empty());
  EXPECT_EQ(none.totalBytes, 0);

  // numCBs < 0: likewise guarded.
  CBAllocation negative = allocateCBs(n150(), -1);
  EXPECT_FALSE(negative.feasible);
  EXPECT_TRUE(negative.cbs.empty());
  EXPECT_EQ(negative.totalBytes, 0);
}

TEST(CBAllocatorTest, FusedLayoutInDeterministicOrder) {
  // The fused plan allocates six buffers in the documented order: activation
  // stage c_0, weight c_1, gamma c_2, scaler c_3, normalized c_4, output c_16.
  CBAllocation allocation = allocateFusedCBs(n150(), /*kTiles=*/4,
                                             /*stageTiles=*/8);
  EXPECT_TRUE(allocation.feasible);
  ASSERT_EQ(allocation.cbs.size(), kFusedCbCount);
  const int64_t expectedIndices[kFusedCbCount] = {0, 1, 2, 3, 4, 16};
  const int64_t expectedDepths[kFusedCbCount] = {8, 4, 4, 1, 8, 2};
  for (std::size_t i = 0; i < kFusedCbCount; ++i) {
    EXPECT_EQ(allocation.cbs[i].index, expectedIndices[i]);
    EXPECT_EQ(allocation.cbs[i].depth, expectedDepths[i]);
    EXPECT_EQ(allocation.cbs[i].tileBytes, 2048);
  }
  EXPECT_EQ(allocation.totalBytes, 27 * 2048);
  EXPECT_EQ(fusedCbTileCount(4, 8), 27);
}

TEST(CBAllocatorTest, FusedLayoutFitsBudgetInvariant) {
  // Staging for the emittable example fits the n150 budget; the oversized
  // staging of the large running example does not.
  EXPECT_TRUE(allocateFusedCBs(n150(), 4, 8).feasible);
  CBAllocation oversized = allocateFusedCBs(n150(), 128, 256);
  EXPECT_FALSE(oversized.feasible);
  EXPECT_EQ(fusedCbTileCount(128, 256), 771);
}

TEST(CBAllocatorTest, FusedLayoutDegenerateGuarded) {
  // Non-positive tile counts are guarded: infeasible, no buffers, no crash.
  EXPECT_FALSE(allocateFusedCBs(n150(), 0, 8).feasible);
  EXPECT_FALSE(allocateFusedCBs(n150(), 4, 0).feasible);
  EXPECT_EQ(fusedCbTileCount(0, 8), 0);
}

TEST(CBAllocatorTest, FusedMlpLayoutInDeterministicOrder) {
  // The fused MLP plan allocates twelve buffers in the documented order:
  // activation row c_0, squares c_1, inverse RMS c_2, normalized row c_3, first
  // weight block c_4, gamma c_5, scaler c_6, bias c_7, second weight block c_8,
  // residual row c_9, intermediate rows c_10, output c_16. The 4 x 4 tile
  // geometry with one row and one column per core is 49 tiles.
  CBAllocation allocation = allocateFusedMlpCBs(n150(), /*kTiles=*/4,
                                                /*hTiles=*/4,
                                                /*rowsPerCore=*/1,
                                                /*colsPerCore=*/1);
  EXPECT_TRUE(allocation.feasible);
  ASSERT_EQ(allocation.cbs.size(), kFusedMlpCbCount);
  const int64_t expectedIndices[kFusedMlpCbCount] = {0, 1, 2, 3, 4,
                                                     5, 6, 7, 8, 9,
                                                     10, 16};
  const int64_t expectedDepths[kFusedMlpCbCount] = {4, 4, 1, 4, 16, 4,
                                                    1, 4, 4, 1, 4,  2};
  for (std::size_t i = 0; i < kFusedMlpCbCount; ++i) {
    EXPECT_EQ(allocation.cbs[i].index, expectedIndices[i]);
    EXPECT_EQ(allocation.cbs[i].depth, expectedDepths[i]);
    EXPECT_EQ(allocation.cbs[i].tileBytes, 2048);
  }
  EXPECT_EQ(allocation.totalBytes, 49 * 2048);
  EXPECT_EQ(fusedMlpCbTileCount(4, 4, 1, 1), 49);
}

TEST(CBAllocatorTest, FusedMlpLayoutFitsBudgetInvariant) {
  // The emittable small MLP fits the n150 budget. The large running example
  // stages a 128 x 384 first weight block, which cannot fit: 59284 tiles.
  EXPECT_TRUE(allocateFusedMlpCBs(n150(), 4, 4, 1, 1).feasible);
  CBAllocation oversized = allocateFusedMlpCBs(n150(), 128, 384, 8, 16);
  EXPECT_FALSE(oversized.feasible);
  EXPECT_EQ(fusedMlpCbTileCount(128, 384, 8, 16), 59284);
}

TEST(CBAllocatorTest, FusedMlpLayoutDegenerateGuarded) {
  // Non-positive tile or block counts are guarded: infeasible, no buffers.
  EXPECT_FALSE(allocateFusedMlpCBs(n150(), 0, 4, 1, 1).feasible);
  EXPECT_FALSE(allocateFusedMlpCBs(n150(), 4, 0, 1, 1).feasible);
  EXPECT_FALSE(allocateFusedMlpCBs(n150(), 4, 4, 0, 1).feasible);
  EXPECT_FALSE(allocateFusedMlpCBs(n150(), 4, 4, 1, 0).feasible);
  EXPECT_EQ(fusedMlpCbTileCount(0, 4, 1, 1), 0);
}

TEST(CBAllocatorTest, ReportSeam) {
  CBAllocation allocation = allocateCBs(n150(), kDefaultNumCBs);

  // toCBSpecs yields the allocation's plan::CBSpec vector for PlanDraft.
  std::vector<plan::CBSpec> specs = toCBSpecs(allocation);
  ASSERT_EQ(specs.size(), 3u);
  EXPECT_EQ(specs[0].index, kCBIn0);
  EXPECT_EQ(specs[1].index, kCBIn1);
  EXPECT_EQ(specs[2].index, kCBOut);
  for (const plan::CBSpec &cb : specs) {
    EXPECT_EQ(cb.depth, 2);
    EXPECT_EQ(cb.tileBytes, 2048);
  }
}

} // namespace
