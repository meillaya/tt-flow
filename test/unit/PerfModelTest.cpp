//===- PerfModelTest.cpp ----------------------------------------*- C++ -*-===//
//
// Unit tests for the static performance model. Every figure is a projected
// analytic estimate (bf16 = 2 bytes per element); none of these numbers is
// measured on hardware. The plans under test come from buildPlanReport, so the
// model is exercised on real scheduler output.
//
//   1. ProjectedFieldsValid  - canonical plan on n150: fusion cuts DRAM traffic,
//                              launches 2 -> 1, and the cycle terms match the
//                              documented formulas for these shapes.
//   2. ProjectedTextLabeled  - every rendered line says "projected" and names
//                              the device.
//   3. DeviceDependence      - the same plan projects differently on n150 and
//                              p100a; the binding constraint reflects the device.
//   4. SmallReplicatedWeight - a weight that fits L1 replicates, multicasts
//                              nothing, and becomes DRAM-bound.
//   5. CBOccupancy           - occupancy is in (0,1] against the device budget.
//   6. SuggestionsDerived    - CB depth and multicast suggestions follow from
//                              the plan and the device budget.
//   7. DegenerateGuard       - a plan without a matmul yields a valid zero
//                              report, not a crash.
//   8. PerfSectionRoundTrip  - toPerfEstimate fills the plan perf block and the
//                              JSON round-trips through parsePlan.
//
//===----------------------------------------------------------------------===//

#include "PerfModel.h"

#include "Model.h"
#include "PlanReport.h"

#include "tensixflow/Generated/DeviceSpecs.h"
#include "tensixflow/PlanSchema.h"

#include "llvm/Support/JSON.h"

#include "gtest/gtest.h"

#include <cmath>
#include <string>

using namespace mlir::tensixflow::scheduler;
using mlir::tensixflow::analysis::estimatePerf;
using mlir::tensixflow::analysis::formatPerfReport;
using mlir::tensixflow::analysis::PerfReport;
using mlir::tensixflow::analysis::toPerfEstimate;
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

constexpr int64_t kActivationBytes = 16777216;     // [1,2048,4096] bf16
constexpr int64_t kWeightBytes = 100663296;        // [4096,12288] bf16
constexpr int64_t kMatmulOutputBytes = 50331648;   // [1,2048,12288] bf16

PlanRequest canonicalRequest(const DeviceSpec &device) {
  PlanRequest request;
  request.graphId = "rmsnorm_matmul";
  request.activation = TensorShape{"activation", {1, 2048, 4096}, "bf16"};
  request.weight = TensorShape{"weight", {4096, 12288}, "bf16"};
  request.fusion = "fused_rmsnorm_matmul";
  request.fused = true;
  request.bytesPerElem = bytesPerElem("bf16");
  request.device = &device;
  return request;
}

plan::Plan canonicalPlan(const DeviceSpec &device) {
  return buildPlanReport(canonicalRequest(device)).plan;
}

bool hasSuggestion(const PerfReport &report, const std::string &substring) {
  for (const std::string &suggestion : report.suggestions) {
    if (suggestion.find(substring) != std::string::npos) {
      return true;
    }
  }
  return false;
}

TEST(PerfModelTest, ProjectedFieldsValid) {
  PerfReport report = estimatePerf(canonicalPlan(n150()), n150());

  // Fusion keeps the intermediate in L1, so unfused DRAM traffic is larger.
  EXPECT_EQ(report.unfused.dramReadBytes,
            kActivationBytes + kActivationBytes + kWeightBytes);
  EXPECT_EQ(report.fused.dramReadBytes, kActivationBytes + kWeightBytes);
  EXPECT_EQ(report.unfused.dramWriteBytes,
            kActivationBytes + kMatmulOutputBytes);
  EXPECT_EQ(report.fused.dramWriteBytes, kMatmulOutputBytes);
  EXPECT_GT(report.unfused.dramReadBytes, report.fused.dramReadBytes);
  EXPECT_EQ(report.savedDramReadBytes, kActivationBytes);
  EXPECT_EQ(report.savedDramWriteBytes, kActivationBytes);
  EXPECT_EQ(report.fused.multicastBytes, kWeightBytes);

  // Kernel launches: the unfused pair vs the single fused kernel.
  EXPECT_EQ(report.unfusedKernelLaunches, 2);
  EXPECT_EQ(report.fusedKernelLaunches, 1);

  // Projected compute term: 2*M*N*K / (cores * flopsPerCyclePerCore) with
  // M=2048, K=4096, N=12288 and 64 Tensix cores on n150.
  EXPECT_EQ(report.cores, 64);
  EXPECT_EQ(report.computeCycles, 786432);

  // Projected DRAM term: traffic / (DRAM bytes per second / clock).
  // 167772160 bytes / 288 bytes per cycle = 582542 cycles (truncated).
  EXPECT_EQ(report.dramCycles, 582542);

  // Projected NoC term: busiest per-core direction / 32 bytes per cycle.
  EXPECT_EQ(report.nocCycles, 106496);

  // Roofline and binding constraint.
  EXPECT_EQ(report.estCycles, 786432);
  EXPECT_DOUBLE_EQ(report.computeUtil, 1.0);
  EXPECT_EQ(report.bound, "compute");
  EXPECT_EQ(report.deviceName, "n150");
  EXPECT_EQ(report.deviceArch, "wormhole_b0");
  EXPECT_FALSE(report.suggestions.empty());
}

TEST(PerfModelTest, ProjectedTextLabeled) {
  PerfReport report = estimatePerf(canonicalPlan(n150()), n150());
  const std::string text = formatPerfReport(report);

  // Every rendered line names the device and is labelled projected.
  size_t lineStart = 0;
  while (lineStart < text.size()) {
    size_t lineEnd = text.find('\n', lineStart);
    if (lineEnd == std::string::npos) {
      lineEnd = text.size();
    }
    const std::string line = text.substr(lineStart, lineEnd - lineStart);
    if (!line.empty()) {
      EXPECT_NE(line.find("projected"), std::string::npos) << line;
      EXPECT_NE(line.find("n150"), std::string::npos) << line;
    }
    lineStart = lineEnd + 1;
  }

  EXPECT_NE(text.find("fused DRAM reads:"), std::string::npos);
  EXPECT_NE(text.find("compute cycles: 786432"), std::string::npos);
  EXPECT_NE(text.find("binding constraint: compute"), std::string::npos);
}

TEST(PerfModelTest, DeviceDependence) {
  const PerfReport onN150 = estimatePerf(canonicalPlan(n150()), n150());
  const PerfReport onP100A = estimatePerf(canonicalPlan(p100a()), p100a());

  // 120 Tensix cores and a faster clock project differently.
  EXPECT_EQ(onN150.cores, 64);
  EXPECT_EQ(onP100A.cores, 120);
  EXPECT_EQ(onN150.computeCycles, 786432);
  EXPECT_EQ(onP100A.computeCycles, 419430);
  EXPECT_GT(onN150.dramCycles, onP100A.dramCycles);

  // The binding constraint follows the device: n150 is compute-bound for these
  // shapes, while p100a's higher FLOP rate and DRAM bandwidth leave DRAM
  // binding.
  EXPECT_EQ(onN150.bound, "compute");
  EXPECT_EQ(onP100A.bound, "dram");
  EXPECT_NE(onN150.bound, onP100A.bound);
}

TEST(PerfModelTest, SmallReplicatedWeight) {
  PlanRequest request = canonicalRequest(n150());
  request.weight = TensorShape{"weight", {4096, 64}, "bf16"}; // 512 KiB
  PerfReport report = estimatePerf(buildPlanReport(request).plan, n150());

  // 524288 bytes fits the 1433600 byte CB budget, so the weight replicates and
  // the NoC moves no multicast traffic.
  EXPECT_EQ(report.fused.multicastBytes, 0);
  EXPECT_EQ(report.fused.dramReadBytes, kActivationBytes + 4096 * 64 * 2);
  EXPECT_TRUE(std::isfinite(report.computeUtil));
  EXPECT_GE(report.computeUtil, 0.0);
  EXPECT_LE(report.computeUtil, 1.0);

  // Little compute over a large activation read: the projection is DRAM-bound.
  EXPECT_EQ(report.bound, "dram");
}

TEST(PerfModelTest, CBOccupancy) {
  PerfReport report = estimatePerf(canonicalPlan(n150()), n150());
  const int64_t budget = n150().cbBudget(
      mlir::tensixflow::device::kDefaultL1ReserveBytes);

  // The canonical three-CB allocation at depth 2 is 12288 bytes.
  EXPECT_EQ(report.cbBytes, 12288);
  EXPECT_LE(report.cbBytes, budget);
  EXPECT_DOUBLE_EQ(report.cbOccupancy, double(12288) / double(budget));
  EXPECT_GT(report.cbOccupancy, 0.0);
  EXPECT_LE(report.cbOccupancy, 1.0);
}

TEST(PerfModelTest, SuggestionsDerived) {
  PerfReport canonical = estimatePerf(canonicalPlan(n150()), n150());
  // Large K (4096) suggests split-K.
  EXPECT_TRUE(hasSuggestion(canonical, "split K"));
  // The canonical allocation is below the feasible device maximum (233 tiles).
  EXPECT_TRUE(hasSuggestion(canonical, "increase CB depth"));
  // The canonical weight already multicasts, so no multicast hint.
  EXPECT_FALSE(hasSuggestion(canonical, "use column multicast"));

  // A replicated weight larger than half the budget suggests multicast.
  PlanRequest request = canonicalRequest(n150());
  request.weight = TensorShape{"weight", {4096, 128}, "bf16"}; // 1 MiB
  PerfReport small = estimatePerf(buildPlanReport(request).plan, n150());
  EXPECT_EQ(small.fused.multicastBytes, 0);
  EXPECT_TRUE(hasSuggestion(small, "use column multicast"));
}

TEST(PerfModelTest, DegenerateGuard) {
  mlir::tensixflow::Plan plan;
  plan.graphId = "no_ops";
  const PerfReport report = estimatePerf(plan, n150());

  EXPECT_EQ(report.estCycles, 0);
  EXPECT_EQ(report.bound, "balanced");
  EXPECT_FALSE(report.suggestions.empty());
  EXPECT_EQ(report.deviceName, "n150");
  EXPECT_EQ(report.cores, 64);
}

TEST(PerfModelTest, PerfSectionRoundTrip) {
  plan::Plan plan = canonicalPlan(n150());
  plan.perf = toPerfEstimate(estimatePerf(plan, n150()));

  EXPECT_GT(plan.perf.estCycles, 0);
  EXPECT_FALSE(plan.perf.bottleneck.empty());
  EXPECT_FALSE(plan.perf.suggestions.empty());

  const std::string json = plan::toJson(plan);
  llvm::Expected<llvm::json::Value> parsed = llvm::json::parse(json);
  ASSERT_TRUE(bool(parsed));
  std::optional<plan::Plan> roundTrip = plan::parsePlan(*parsed);
  ASSERT_TRUE(roundTrip.has_value());
  EXPECT_EQ(roundTrip->device.name, "n150");
  EXPECT_EQ(roundTrip->perf.estCycles, plan.perf.estCycles);
  EXPECT_EQ(roundTrip->perf.bottleneck, plan.perf.bottleneck);
}

} // namespace
