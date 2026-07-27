//===- NoCEstimatorTest.cpp -------------------------------------*- C++ -*-===//
//
// Unit tests for the TensixFlow NoC traffic estimator (todo 29). Every figure is
// a STATIC ANALYTIC ESTIMATE computed from tensor shapes (bf16 = 2 bytes/elem) —
// NOT a measured hardware number.
//
//   1. FusedSavesDramReads — canonical shapes; fused.dramReadBytes < unfused.
//   2. SavedIntermediateEqualsRmsnormOutput — saved intermediate bytes == the
//                           rmsnorm-output tensor bytes (== activation bytes).
//   3. MulticastPositive — a multicast weight plan yields multicastBytes > 0
//                           (consumed from WeightPlan.multicastBytes).
//   4. NoFusionZeroSavings — a non-fused plan reports zero fusion savings (the
//                           required failure-path assertion).
//   5. UserExampleMagnitudes — a GB-scale case; unfused reads several GB, fused
//                           strictly less, both > 1 GiB, gap == saved intermediate.
//   6. DramWritesReduced — fused.dramWriteBytes < unfused (the intermediate
//                           write is elided).
//
//===----------------------------------------------------------------------===//

#include "NoCEstimator.h"

#include "Model.h"
#include "WeightPlanner.h"

#include "gtest/gtest.h"

using namespace mlir::tensixflow::scheduler;

namespace {

constexpr int64_t kGiB = int64_t{1} << 30; // 1073741824

// Canonical rmsnorm+matmul shapes (user spec): activation [1,2048,4096] bf16
// (16 MiB), weight [4096,12288] bf16 (96 MiB). rmsnorm output == activation
// shape (16 MiB); matmul output [1,2048,12288] (48 MiB).
TensorShape canonicalActivation() {
  return TensorShape{"activation", {1, 2048, 4096}, "bf16"};
}
TensorShape canonicalWeight() {
  return TensorShape{"weight", {4096, 12288}, "bf16"};
}

TEST(NoCEstimatorTest, FusedSavesDramReads) {
  const int bpe = bytesPerElem("bf16");
  TensorShape activation = canonicalActivation();
  TensorShape weight = canonicalWeight();
  plan::ShardGrid grid{8, 8};
  WeightPlan wp = planWeight(weight, grid, bpe);

  NocTraffic unfused = estimateNoC(NocInputs{activation, weight, wp, false}, bpe);
  NocTraffic fused = estimateNoC(NocInputs{activation, weight, wp, true}, bpe);

  // Unfused re-reads the 16 MiB intermediate from DRAM; fused does not.
  // unfused reads = 16MiB (act) + 16MiB (intermediate) + 96MiB (weight).
  EXPECT_EQ(unfused.estimate.dramReadBytes, 16777216 + 16777216 + 100663296);
  // fused reads = 16MiB (act) + 96MiB (weight).
  EXPECT_EQ(fused.estimate.dramReadBytes, 16777216 + 100663296);
  EXPECT_LT(fused.estimate.dramReadBytes, unfused.estimate.dramReadBytes);
}

TEST(NoCEstimatorTest, SavedIntermediateEqualsRmsnormOutput) {
  const int bpe = bytesPerElem("bf16");
  TensorShape activation = canonicalActivation();
  TensorShape weight = canonicalWeight();
  plan::ShardGrid grid{8, 8};
  WeightPlan wp = planWeight(weight, grid, bpe);

  NocTraffic fused = estimateNoC(NocInputs{activation, weight, wp, true}, bpe);

  // rmsnorm output shape == activation shape -> 1*2048*4096*2 = 16777216.
  const int64_t rmsnormOutBytes = activation.bytes(bpe);
  EXPECT_EQ(rmsnormOutBytes, 16777216);
  EXPECT_EQ(fused.savedIntermediateBytes, rmsnormOutBytes);
  // The breakdown is internally consistent: the saved DRAM read and the saved
  // DRAM write each equal the intermediate that no longer round-trips DRAM.
  EXPECT_EQ(fused.savedDramReadBytes, rmsnormOutBytes);
  EXPECT_EQ(fused.savedDramWriteBytes, rmsnormOutBytes);
}

TEST(NoCEstimatorTest, MulticastPositive) {
  const int bpe = bytesPerElem("bf16");
  TensorShape activation = canonicalActivation();
  TensorShape weight = canonicalWeight();
  plan::ShardGrid grid{8, 8};
  WeightPlan wp = planWeight(weight, grid, bpe);

  // The canonical weight is too big for L1 -> multicast (todo 27).
  ASSERT_EQ(wp.strategy, WeightStrategy::Multicast);
  ASSERT_GT(wp.multicastBytes, 0);

  NocTraffic fused = estimateNoC(NocInputs{activation, weight, wp, true}, bpe);
  NocTraffic unfused = estimateNoC(NocInputs{activation, weight, wp, false}, bpe);

  // The estimator CONSUMES the weight plan's multicast figure verbatim (it does
  // not recompute it), for both the fused and the unfused model.
  EXPECT_EQ(fused.estimate.multicastBytes, wp.multicastBytes);
  EXPECT_EQ(unfused.estimate.multicastBytes, wp.multicastBytes);
  EXPECT_GT(fused.estimate.multicastBytes, 0);

  // A replicated (small) weight multicasts nothing.
  TensorShape smallWeight{"weight", {256, 256}, "bf16"};
  WeightPlan replicated = planWeight(smallWeight, grid, bpe);
  ASSERT_EQ(replicated.strategy, WeightStrategy::Replicate);
  NocTraffic rep =
      estimateNoC(NocInputs{activation, smallWeight, replicated, true}, bpe);
  EXPECT_EQ(rep.estimate.multicastBytes, 0);
}

TEST(NoCEstimatorTest, NoFusionZeroSavings) {
  const int bpe = bytesPerElem("bf16");
  TensorShape activation = canonicalActivation();
  TensorShape weight = canonicalWeight();
  plan::ShardGrid grid{8, 8};
  WeightPlan wp = planWeight(weight, grid, bpe);

  NocTraffic unfused = estimateNoC(NocInputs{activation, weight, wp, false}, bpe);

  // The required failure-path assertion: no fusion -> nothing saved.
  EXPECT_EQ(unfused.savedDramReadBytes, 0);
  EXPECT_EQ(unfused.savedDramWriteBytes, 0);
  EXPECT_EQ(unfused.savedIntermediateBytes, 0);
  // The unfused estimate itself is still populated (traffic happens), and
  // nothing is held in L1 across the two separate ops.
  EXPECT_GT(unfused.estimate.dramReadBytes, 0);
  EXPECT_GT(unfused.estimate.dramWriteBytes, 0);
  EXPECT_EQ(unfused.estimate.l1Bytes, 0);
}

TEST(NoCEstimatorTest, UserExampleMagnitudes) {
  const int bpe = bytesPerElem("bf16");
  // GB-scale: batch-scaled activation [128,2048,4096] bf16 = 2 GiB exactly,
  // canonical weight [4096,12288] bf16 = 96 MiB. Reproduces the user spec's
  // unfused-vs-fused STYLE (unfused reads several GB, fused strictly less) —
  // computed from shapes, NOT hardcoded 3.1/2.1 literals.
  TensorShape activation{"activation", {128, 2048, 4096}, "bf16"};
  TensorShape weight = canonicalWeight();
  plan::ShardGrid grid{8, 8};
  WeightPlan wp = planWeight(weight, grid, bpe);

  const int64_t actBytes = activation.bytes(bpe);
  const int64_t twoGiB = 2 * kGiB;
  ASSERT_EQ(actBytes, twoGiB); // 128*2048*4096*2 = 2 GiB

  NocTraffic unfused = estimateNoC(NocInputs{activation, weight, wp, false}, bpe);
  NocTraffic fused = estimateNoC(NocInputs{activation, weight, wp, true}, bpe);

  // Both exceed 1 GiB; unfused is several GB and strictly greater than fused.
  EXPECT_GT(fused.estimate.dramReadBytes, kGiB);
  EXPECT_GT(unfused.estimate.dramReadBytes, kGiB);
  EXPECT_GT(unfused.estimate.dramReadBytes, fused.estimate.dramReadBytes);

  // The gap equals the saved intermediate traffic (the 2 GiB rmsnorm output
  // that fused keeps in L1 instead of round-tripping DRAM).
  const int64_t readGap =
      unfused.estimate.dramReadBytes - fused.estimate.dramReadBytes;
  EXPECT_EQ(readGap, fused.savedIntermediateBytes);
  EXPECT_EQ(readGap, actBytes);
}

TEST(NoCEstimatorTest, DramWritesReduced) {
  const int bpe = bytesPerElem("bf16");
  TensorShape activation = canonicalActivation();
  TensorShape weight = canonicalWeight();
  plan::ShardGrid grid{8, 8};
  WeightPlan wp = planWeight(weight, grid, bpe);

  NocTraffic unfused = estimateNoC(NocInputs{activation, weight, wp, false}, bpe);
  NocTraffic fused = estimateNoC(NocInputs{activation, weight, wp, true}, bpe);

  // matmul output [1,2048,12288] bf16 = 1*2048*12288*2 = 50331648.
  const int64_t matmulOutBytes = 50331648;
  // Unfused writes the intermediate (16 MiB) AND the matmul output (48 MiB);
  // fused writes only the matmul output.
  EXPECT_EQ(unfused.estimate.dramWriteBytes, 16777216 + matmulOutBytes);
  EXPECT_EQ(fused.estimate.dramWriteBytes, matmulOutBytes);
  EXPECT_LT(fused.estimate.dramWriteBytes, unfused.estimate.dramWriteBytes);
}

} // namespace
