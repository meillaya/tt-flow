//===- PerfModelTest.cpp ----------------------------------------*- C++ -*-===//
//
// Unit tests for the TensixFlow static performance model (todo 42). Every figure
// is a STATIC ANALYTIC ESTIMATE (bf16 = 2 bytes/elem) — NOT a measured hardware
// number. The plans under test come from buildPlanReport (todo 30), so the perf
// model is exercised on real scheduler output.
//
//   1. FusedReportFieldsValid — the user's report shape: unfused>fused DRAM
//      reads+writes, non-empty bottleneck, non-empty suggestions, finite numbers,
//      kernel launches unfused 2 -> fused 1.
//   2. PlanPerfSectionFilled — toPerfEstimate fills the todo-24 perf section
//      (est_cycles, compute_util, bottleneck, suggestions); the plan JSON carries
//      them and round-trips through parsePlan.
//   3. NoMulticastDifferentBottleneck — the required failure path: a replicated
//      (small) weight yields multicast_bytes == 0 and a DIFFERENT bottleneck.
//   4. CBOccupancyWithinBudget — CB occupancy in (0,1], footprint <= the imported
//      kCBShardBudget.
//   5. SuggestionsDerived — suggestions are derived from the plan (split-K for a
//      large K; multicast for a large replicated weight).
//
//===----------------------------------------------------------------------===//

#include "PerfModel.h"

#include "L1Model.h"
#include "Model.h"
#include "PlanReport.h"

#include "llvm/Support/JSON.h"

#include "gtest/gtest.h"

#include <cmath>
#include <string>

// Bring in the scheduler planners/types wholesale; pull in the analysis symbols
// by name so the `plan` namespace alias (defined in BOTH namespaces) is never
// ambiguous.
using namespace mlir::tensixflow::scheduler;
using mlir::tensixflow::analysis::estimatePerf;
using mlir::tensixflow::analysis::formatPerfReport;
using mlir::tensixflow::analysis::PerfReport;
using mlir::tensixflow::analysis::toPerfEstimate;

namespace {

PlanRequest canonicalRequest() {
  PlanRequest req;
  req.graphId = "rmsnorm_matmul";
  req.activation = TensorShape{"activation", {1, 2048, 4096}, "bf16"};
  req.weight = TensorShape{"weight", {4096, 12288}, "bf16"};
  req.fusion = "fused_rmsnorm_matmul";
  req.fused = true;
  req.bytesPerElem = bytesPerElem("bf16");
  return req;
}

plan::Plan canonicalPlan() { return buildPlanReport(canonicalRequest()).plan; }

bool hasSuggestion(const PerfReport &report, const std::string &substr) {
  for (const std::string &suggestion : report.suggestions)
    if (suggestion.find(substr) != std::string::npos)
      return true;
  return false;
}

TEST(PerfModelTest, FusedReportFieldsValid) {
  PerfReport perf = estimatePerf(canonicalPlan());

  // The fusion-reduces-DRAM invariant: the fused path keeps the intermediate in
  // L1 instead of round-tripping DRAM.
  EXPECT_GT(perf.unfused.dramReadBytes, perf.fused.dramReadBytes);
  EXPECT_GT(perf.unfused.dramWriteBytes, perf.fused.dramWriteBytes);
  EXPECT_GT(perf.savedDramReadBytes, 0);
  EXPECT_GT(perf.savedDramWriteBytes, 0);

  // Non-empty bottleneck enum + non-empty suggestions.
  EXPECT_FALSE(perf.bottleneck.empty());
  EXPECT_FALSE(perf.suggestions.empty());

  // Kernel launches: the unfused op pair vs the single fused kernel.
  EXPECT_EQ(perf.unfusedKernelLaunches, 2);
  EXPECT_EQ(perf.fusedKernelLaunches, 1);

  // All numeric fields finite + sane (ints are finite by construction; guard the
  // doubles + ranges explicitly).
  EXPECT_GT(perf.estCycles, 0);
  EXPECT_GT(perf.computeCycles, 0);
  EXPECT_GT(perf.memoryCycles, 0);
  EXPECT_TRUE(std::isfinite(perf.computeUtil));
  EXPECT_GE(perf.computeUtil, 0.0);
  EXPECT_LE(perf.computeUtil, 1.0);
  EXPECT_TRUE(std::isfinite(perf.cbOccupancy));
  EXPECT_GE(perf.fused.dramReadBytes, 0);
  EXPECT_GE(perf.fused.l1Bytes, 0);
  EXPECT_GT(perf.fused.multicastBytes, 0); // canonical weight is multicast

  // The human-readable report carries the user's fields.
  const std::string text = formatPerfReport(perf);
  EXPECT_NE(text.find("unfused DRAM reads:"), std::string::npos);
  EXPECT_NE(text.find("fused DRAM reads:"), std::string::npos);
  EXPECT_NE(text.find("kernel launches:"), std::string::npos);
  EXPECT_NE(text.find("predicted bottleneck:"), std::string::npos);
  EXPECT_NE(text.find("suggested optimization:"), std::string::npos);
}

TEST(PerfModelTest, PlanPerfSectionFilled) {
  PerfReport perf = estimatePerf(canonicalPlan());
  plan::PerfEstimate est = toPerfEstimate(perf);

  // The todo-24 perf section fields are present + numeric.
  EXPECT_GT(est.estCycles, 0);
  EXPECT_TRUE(std::isfinite(est.computeUtil));
  EXPECT_FALSE(est.bottleneck.empty());
  EXPECT_FALSE(est.suggestions.empty());

  // Fill the plan's perf section, serialize, and round-trip through parsePlan.
  plan::Plan p = canonicalPlan();
  p.perf = est;
  const std::string json = plan::toJson(p);
  EXPECT_NE(json.find("\"est_cycles\": "), std::string::npos);
  EXPECT_NE(json.find("\"compute_util\": "), std::string::npos);
  EXPECT_NE(json.find("\"bottleneck\": "), std::string::npos);
  EXPECT_NE(json.find("\"suggestions\": "), std::string::npos);

  llvm::Expected<llvm::json::Value> parsed = llvm::json::parse(json);
  ASSERT_TRUE(bool(parsed));
  std::optional<plan::Plan> roundTrip = plan::parsePlan(*parsed);
  ASSERT_TRUE(roundTrip.has_value());
  EXPECT_EQ(roundTrip->perf.estCycles, est.estCycles);
  EXPECT_EQ(roundTrip->perf.bottleneck, est.bottleneck);
  EXPECT_FALSE(roundTrip->perf.suggestions.empty());
}

TEST(PerfModelTest, NoMulticastDifferentBottleneck) {
  PerfReport canonical = estimatePerf(canonicalPlan());
  ASSERT_GT(canonical.fused.multicastBytes, 0); // canonical multicasts the weight

  // A small weight that fits L1 -> replicated -> no multicast. K stays 4096 so it
  // still matches the activation's contracting dim.
  PlanRequest req = canonicalRequest();
  req.weight = TensorShape{"weight", {4096, 64}, "bf16"}; // 512 KiB < 960 KiB
  PerfReport small = estimatePerf(buildPlanReport(req).plan);

  // The required failure-path assertions: no multicast + a different bottleneck.
  EXPECT_EQ(small.fused.multicastBytes, 0);
  EXPECT_FALSE(small.bottleneck.empty());
  EXPECT_NE(small.bottleneck, canonical.bottleneck);

  // Pinned: the large multicast matmul is compute-bound; the tiny replicated one
  // reads a big activation for little compute -> memory-bound.
  EXPECT_EQ(canonical.bottleneck, "compute");
  EXPECT_EQ(small.bottleneck, "memory");
}

TEST(PerfModelTest, CBOccupancyWithinBudget) {
  PerfReport perf = estimatePerf(canonicalPlan());

  // The canonical 3-CB allocation fills the imported per-core CB budget exactly
  // (3 * depth 160 * 2048 = 983040 = kCBShardBudget).
  EXPECT_GT(perf.cbBytes, 0);
  EXPECT_LE(perf.cbBytes, kCBShardBudget);
  EXPECT_TRUE(std::isfinite(perf.cbOccupancy));
  EXPECT_GT(perf.cbOccupancy, 0.0);
  EXPECT_LE(perf.cbOccupancy, 1.0);
}

TEST(PerfModelTest, SuggestionsDerived) {
  PerfReport canonical = estimatePerf(canonicalPlan());
  // Large K (4096) -> split-K suggestion.
  EXPECT_TRUE(hasSuggestion(canonical, "split K"));
  // The canonical weight is already multicast -> no "use column multicast" hint.
  EXPECT_FALSE(hasSuggestion(canonical, "use column multicast"));

  // A large replicated weight (fits L1 but > half the budget) -> multicast hint.
  PlanRequest req = canonicalRequest();
  req.weight = TensorShape{"weight", {4096, 64}, "bf16"};
  PerfReport small = estimatePerf(buildPlanReport(req).plan);
  EXPECT_TRUE(hasSuggestion(small, "multicast"));
  EXPECT_FALSE(small.suggestions.empty());
}

} // namespace
