//===- PlanReportTest.cpp ---------------------------------------*- C++ -*-===//
//
// Unit tests for the TensixFlow plan-report assembler (todo 30): buildPlanReport
// runs the four planners over the canonical rmsnorm+matmul shapes and packages
// the decisions; formatPlanReport renders the user-facing "Chosen layout: ..."
// text. Every figure is a STATIC ANALYTIC ESTIMATE (bf16 = 2 bytes/elem), NOT a
// measured hardware number.
//
//   1. FusedReportHasChosenLayoutFields — the report text carries every user-spec
//      field with the canonical fused values (savings 16777216).
//   2. FusedPlanJsonHasWeightStrategyAndSavings — the schema Plan serializes with
//      weight_strategy "multicast" and a positive saved-DRAM-reads figure.
//   3. FusedPlanHasThreeCBsAtConventionIndices — the CB specs are {0,1,16}, depth
//      160, 2048 bytes/tile.
//   4. UnfusedReportHasZeroSavings — the required failure path: an unfused plan
//      reports fusion "none", DRAM intermediate storage, and zero savings.
//
//===----------------------------------------------------------------------===//

#include "PlanReport.h"

#include "Model.h"

#include "gtest/gtest.h"

#include <string>

using namespace mlir::tensixflow::scheduler;

namespace {

TensorShape canonicalActivation() {
  return TensorShape{"activation", {1, 2048, 4096}, "bf16"};
}
TensorShape canonicalWeight() {
  return TensorShape{"weight", {4096, 12288}, "bf16"};
}

PlanRequest fusedRequest() {
  PlanRequest req;
  req.graphId = "rmsnorm_matmul";
  req.activation = canonicalActivation();
  req.weight = canonicalWeight();
  req.fusion = "fused_rmsnorm_matmul";
  req.fused = true;
  req.bytesPerElem = bytesPerElem("bf16");
  return req;
}

TEST(PlanReportTest, FusedReportHasChosenLayoutFields) {
  PlanReport report = buildPlanReport(fusedRequest());
  std::string text = formatPlanReport(report);
  EXPECT_NE(text.find("Chosen layout: rmsnorm_matmul"), std::string::npos);
  EXPECT_NE(text.find("activation_sharding: row-major"), std::string::npos);
  EXPECT_NE(text.find("weight_strategy: multicast"), std::string::npos);
  EXPECT_NE(text.find("intermediate_storage: L1"), std::string::npos);
  EXPECT_NE(text.find("fusion: fused_rmsnorm_matmul"), std::string::npos);
  EXPECT_NE(text.find("estimated_DRAM_reads_saved: 16777216"),
            std::string::npos);
  EXPECT_NE(text.find("estimated_intermediate_writes_saved: 16777216"),
            std::string::npos);
}

TEST(PlanReportTest, FusedPlanJsonHasWeightStrategyAndSavings) {
  PlanReport report = buildPlanReport(fusedRequest());
  std::string json = plan::toJson(report.plan);
  EXPECT_NE(json.find("\"weight_strategy\": \"multicast\""), std::string::npos);
  EXPECT_GT(report.noc.savedDramReadBytes, 0);
  EXPECT_EQ(report.noc.savedDramReadBytes, 16777216);
  EXPECT_EQ(report.plan.noc.multicastBytes, 100663296);
}

TEST(PlanReportTest, FusedPlanHasThreeCBsAtConventionIndices) {
  PlanReport report = buildPlanReport(fusedRequest());
  ASSERT_EQ(report.plan.cbSpecs.size(), 3u);
  EXPECT_EQ(report.plan.cbSpecs[0].index, 0);
  EXPECT_EQ(report.plan.cbSpecs[1].index, 1);
  EXPECT_EQ(report.plan.cbSpecs[2].index, 16);
  EXPECT_EQ(report.plan.cbSpecs[0].depth, 160);
  EXPECT_EQ(report.plan.cbSpecs[0].tileBytes, 2048);
}

TEST(PlanReportTest, UnfusedReportHasZeroSavings) {
  PlanRequest req = fusedRequest();
  req.fusion = "none";
  req.fused = false;
  PlanReport report = buildPlanReport(req);
  std::string text = formatPlanReport(report);
  EXPECT_NE(text.find("fusion: none"), std::string::npos);
  EXPECT_NE(text.find("intermediate_storage: DRAM"), std::string::npos);
  EXPECT_NE(text.find("estimated_DRAM_reads_saved: 0"), std::string::npos);
  EXPECT_EQ(report.noc.savedDramReadBytes, 0);
  EXPECT_EQ(report.noc.savedDramWriteBytes, 0);
  EXPECT_EQ(report.noc.savedIntermediateBytes, 0);
}

} // namespace
