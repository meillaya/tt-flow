//===- PlanReportTest.cpp ---------------------------------------*- C++ -*-===//
//
// Unit tests for the plan-report assembler. buildPlanReport runs the four
// planners over the canonical rmsnorm+matmul shapes on a selected device and
// packages the decisions; formatPlanReport renders the report text. Every
// figure is a projected analytic estimate, not a measured hardware number.
//
//   1. FusedReportHasChosenLayoutFields - the report text carries every report
//                              field, the device name, and the 16 MiB savings.
//   2. PlanJsonHasDeviceAndSavings      - the schema Plan serializes the device
//                              block, weight_strategy, and savings, and
//                              re-parses through parsePlan.
//   3. PlanHasThreeCBsAtConventionIndices - CB indices {0,1,16}, depth 2.
//   4. UnfusedReportHasZeroSavings      - the unfused path reports zero savings.
//   5. DeviceSelectionChangesGrid       - n300 plans 8x7, p100a plans 12x10.
//   6. DefaultDeviceIsN150              - a null request device resolves to the
//                              generated default.
//
//===----------------------------------------------------------------------===//

#include "PlanReport.h"

#include "Model.h"

#include "tensixflow/Generated/DeviceSpecs.h"
#include "tensixflow/PlanSchema.h"

#include "llvm/Support/JSON.h"

#include "gtest/gtest.h"

#include <optional>
#include <string>

using namespace mlir::tensixflow::scheduler;
using mlir::tensixflow::device::DeviceSpec;

namespace {

TensorShape canonicalActivation() {
  return TensorShape{"activation", {1, 2048, 4096}, "bf16"};
}
TensorShape canonicalWeight() {
  return TensorShape{"weight", {4096, 12288}, "bf16"};
}

PlanRequest fusedRequest(const DeviceSpec &device) {
  PlanRequest request;
  request.graphId = "rmsnorm_matmul";
  request.activation = canonicalActivation();
  request.weight = canonicalWeight();
  request.fusion = "fused_rmsnorm_matmul";
  request.fused = true;
  request.bytesPerElem = bytesPerElem("bf16");
  request.device = &device;
  return request;
}

TEST(PlanReportTest, FusedReportHasChosenLayoutFields) {
  const DeviceSpec *n150 = mlir::tensixflow::device::lookup("n150");
  ASSERT_NE(n150, nullptr);
  PlanReport report = buildPlanReport(fusedRequest(*n150));
  const std::string text = formatPlanReport(report);

  EXPECT_NE(text.find("Chosen layout: rmsnorm_matmul"), std::string::npos);
  EXPECT_NE(text.find("device: n150 (wormhole_b0)"), std::string::npos);
  EXPECT_NE(text.find("activation_sharding: row-major"), std::string::npos);
  EXPECT_NE(text.find("weight_strategy: multicast"), std::string::npos);
  EXPECT_NE(text.find("intermediate_storage: L1"), std::string::npos);
  EXPECT_NE(text.find("fusion: fused_rmsnorm_matmul"), std::string::npos);
  EXPECT_NE(text.find("estimated_DRAM_reads_saved: 16777216"),
            std::string::npos);
  EXPECT_NE(text.find("estimated_intermediate_writes_saved: 16777216"),
            std::string::npos);
  // The default three-CB pipeline at depth 2 is 12288 bytes per core.
  EXPECT_NE(text.find("planned_cb_bytes_per_core: 12288"), std::string::npos);
}

TEST(PlanReportTest, PlanJsonHasDeviceAndSavings) {
  const DeviceSpec *n150 = mlir::tensixflow::device::lookup("n150");
  ASSERT_NE(n150, nullptr);
  PlanReport report = buildPlanReport(fusedRequest(*n150));
  const std::string json = plan::toJson(report.plan);

  EXPECT_NE(json.find("\"device\""), std::string::npos);
  EXPECT_NE(json.find("\"name\": \"n150\""), std::string::npos);
  EXPECT_NE(json.find("\"arch\": \"wormhole_b0\""), std::string::npos);
  EXPECT_NE(json.find("\"grid_x\": 8"), std::string::npos);
  EXPECT_NE(json.find("\"grid_y\": 8"), std::string::npos);
  EXPECT_NE(json.find("\"l1_usable_bytes\": 1499136"), std::string::npos);
  EXPECT_NE(json.find("\"flops_per_cycle_per_core\": 4096"),
            std::string::npos);
  EXPECT_NE(json.find("\"dram_bandwidth_bytes_per_second\": 288000000000"),
            std::string::npos);

  EXPECT_NE(json.find("\"weight_strategy\": \"multicast\""), std::string::npos);
  EXPECT_GT(report.noc.savedDramReadBytes, 0);
  EXPECT_EQ(report.noc.savedDramReadBytes, 16777216);
  EXPECT_EQ(report.plan.noc.multicastBytes, 100663296);

  // The plan JSON round-trips through the schema parse seam.
  llvm::Expected<llvm::json::Value> parsed = llvm::json::parse(json);
  ASSERT_TRUE(bool(parsed));
  std::optional<plan::Plan> roundTrip = plan::parsePlan(*parsed);
  ASSERT_TRUE(roundTrip.has_value());
  EXPECT_EQ(roundTrip->device.name, "n150");
  EXPECT_EQ(roundTrip->device.gridX, 8);
  EXPECT_EQ(roundTrip->device.gridY, 8);
  EXPECT_EQ(roundTrip->layout.weightStrategy, "multicast");
}

TEST(PlanReportTest, PlanHasThreeCBsAtConventionIndices) {
  const DeviceSpec *n150 = mlir::tensixflow::device::lookup("n150");
  ASSERT_NE(n150, nullptr);
  PlanReport report = buildPlanReport(fusedRequest(*n150));

  ASSERT_EQ(report.plan.cbSpecs.size(), 3u);
  EXPECT_EQ(report.plan.cbSpecs[0].index, 0);
  EXPECT_EQ(report.plan.cbSpecs[1].index, 1);
  EXPECT_EQ(report.plan.cbSpecs[2].index, 16);
  EXPECT_EQ(report.plan.cbSpecs[0].depth, 2);
  EXPECT_EQ(report.plan.cbSpecs[0].tileBytes, 2048);
  EXPECT_TRUE(report.cbs.feasible);
}

TEST(PlanReportTest, UnfusedReportHasZeroSavings) {
  const DeviceSpec *n150 = mlir::tensixflow::device::lookup("n150");
  ASSERT_NE(n150, nullptr);
  PlanRequest request = fusedRequest(*n150);
  request.fusion = "none";
  request.fused = false;

  PlanReport report = buildPlanReport(request);
  const std::string text = formatPlanReport(report);
  EXPECT_NE(text.find("fusion: none"), std::string::npos);
  EXPECT_NE(text.find("intermediate_storage: DRAM"), std::string::npos);
  EXPECT_NE(text.find("estimated_DRAM_reads_saved: 0"), std::string::npos);
  EXPECT_EQ(report.noc.savedDramReadBytes, 0);
  EXPECT_EQ(report.noc.savedDramWriteBytes, 0);
  EXPECT_EQ(report.noc.savedIntermediateBytes, 0);
}

TEST(PlanReportTest, DeviceSelectionChangesGrid) {
  const DeviceSpec *n300 = mlir::tensixflow::device::lookup("n300");
  ASSERT_NE(n300, nullptr);
  PlanReport onN300 = buildPlanReport(fusedRequest(*n300));
  EXPECT_EQ(onN300.plan.shardGrid.gridX, 8);
  EXPECT_EQ(onN300.plan.shardGrid.gridY, 7);
  EXPECT_EQ(onN300.plan.device.name, "n300");
  EXPECT_NE(formatPlanReport(onN300).find("device: n300 (wormhole_b0)"),
            std::string::npos);

  const DeviceSpec *p100a = mlir::tensixflow::device::lookup("p100a");
  ASSERT_NE(p100a, nullptr);
  PlanReport onP100A = buildPlanReport(fusedRequest(*p100a));
  EXPECT_EQ(onP100A.plan.shardGrid.gridX, 12);
  EXPECT_EQ(onP100A.plan.shardGrid.gridY, 10);
  EXPECT_EQ(onP100A.plan.device.arch, "blackhole");
  EXPECT_EQ(onP100A.plan.device.l1UsableBytes, 1572864);

  // The weight budget differs per device, but this weight multicasts on both.
  EXPECT_EQ(onN300.weight.strategy, WeightStrategy::Multicast);
  EXPECT_EQ(onP100A.weight.strategy, WeightStrategy::Multicast);
}

TEST(PlanReportTest, DefaultDeviceIsN150) {
  PlanRequest request = fusedRequest(*mlir::tensixflow::device::defaultDevice());
  request.device = nullptr;
  PlanReport report = buildPlanReport(request);

  EXPECT_EQ(report.plan.device.name, "n150");
  EXPECT_EQ(report.plan.device.arch, "wormhole_b0");
  EXPECT_EQ(report.plan.shardGrid.gridX, 8);
  EXPECT_EQ(report.plan.shardGrid.gridY, 8);
}

} // namespace
