//===- PlanSchemaTest.cpp ---------------------------------------*- C++ -*-===//
//
// Unit tests for the TensixFlow plan/perf/viz JSON schema:
//   1. RoundTrip          - serialize a sample plan, re-parse field-for-field,
//                           including the required device and emission blocks.
//   2. ConformsToSchema   - validate the sample against schema/plan.schema.json.
//   3. RejectsMissingField - a missing required field, including the device and
//                           emission blocks, is rejected by parsePlan and by
//                           the schema validator.
//
//===----------------------------------------------------------------------===//

#include "tensixflow/PlanSchema.h"

#include "llvm/Support/JSON.h"

#include "gtest/gtest.h"

#include <fstream>
#include <iterator>
#include <optional>
#include <string>
#include <vector>

namespace json = llvm::json;
using mlir::tensixflow::CBSpec;
using mlir::tensixflow::Plan;
using mlir::tensixflow::TensorDesc;
using mlir::tensixflow::parsePlan;
using mlir::tensixflow::toJson;

#ifndef TENSIXFLOW_SCHEMA_DIR
#error "TENSIXFLOW_SCHEMA_DIR must be defined to the source schema/ directory"
#endif

namespace {

Plan makeSamplePlan() {
  Plan plan;
  plan.graphId = "rmsnorm_matmul";

  TensorDesc activation;
  activation.name = "activation";
  activation.shape = {1, 2048, 4096};
  activation.dtype = "bf16";
  activation.placement = "DRAM";

  TensorDesc weight;
  weight.name = "weight";
  weight.shape = {4096, 12288};
  weight.dtype = "bf16";
  weight.placement = "DRAM";

  plan.tensors = {activation, weight};
  plan.device.name = "n150";
  plan.device.arch = "wormhole_b0";
  plan.device.gridX = 8;
  plan.device.gridY = 8;
  plan.device.l1UsableBytes = 1499136;
  plan.device.flopsPerCyclePerCore = 4096;
  plan.device.dramBandwidthBytesPerSecond = 288000000000;
  plan.layout.activationSharding = "row-major";
  plan.layout.weightStrategy = "multicast";
  plan.shardGrid.gridY = 8;
  plan.shardGrid.gridX = 8;
  plan.cbSpecs.push_back(CBSpec{/*index=*/0, /*depth=*/4, /*tileBytes=*/2048});
  plan.noc.dramReadBytes = 2147483648LL;
  plan.noc.dramWriteBytes = 1073741824LL;
  plan.noc.l1Bytes = 536870912LL;
  plan.noc.multicastBytes = 268435456LL;
  plan.emission.mapping = "output-tile-ranges";
  plan.emission.coresUsed = 64;
  plan.emission.maxTilesPerCore = 4;
  plan.emission.dramReadBytes = 2147483648LL;
  plan.emission.dramWriteBytes = 1073741824LL;
  plan.emission.fusedStageTiles = 32;
  plan.emission.fusedFeasible = true;
  plan.perf.estCycles = 1234567LL;
  plan.perf.computeUtil = 0.75;
  plan.perf.bottleneck = "memory";
  plan.perf.suggestions = {"increase CB depth", "use column multicast"};
  plan.viz.coreGrid = "8x8 grid";
  plan.viz.shards = "row-major shards";
  plan.viz.nocRoutes = "multicast routes";
  plan.viz.cbUsage = "CB0 depth 4";
  plan.viz.timeline = "reader/compute/writer";
  return plan;
}

std::optional<json::Value> parseText(const std::string &text) {
  llvm::Expected<json::Value> value = json::parse(text);
  if (!value) {
    llvm::consumeError(value.takeError());
    return std::nullopt;
  }
  return std::move(*value);
}

std::string readSchemaFile() {
  std::ifstream in(std::string(TENSIXFLOW_SCHEMA_DIR) + "/plan.schema.json");
  EXPECT_TRUE(in.good()) << "cannot open schema/plan.schema.json";
  return std::string((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());
}

// Schema-driven structural validator: walks schema/plan.schema.json and checks
// the data has every "required" field (recursing through "properties" and array
// "items"). It records human-readable errors rather than asserting, so a single
// run reports every missing field.
void validateAgainstSchema(const json::Object &schema, const json::Value &data,
                           const std::string &path,
                           std::vector<std::string> &errors) {
  if (const json::Array *required = schema.getArray("required")) {
    const json::Object *dataObj = data.getAsObject();
    if (dataObj == nullptr) {
      errors.push_back(path + ": expected object");
      return;
    }
    const json::Object *props = schema.getObject("properties");
    for (const json::Value &req : *required) {
      std::optional<llvm::StringRef> name = req.getAsString();
      if (!name)
        continue;
      const json::Value *subData = dataObj->get(*name);
      if (subData == nullptr) {
        errors.push_back(path + ": missing required field '" +
                         std::string(*name) + "'");
        continue;
      }
      if (props != nullptr) {
        const json::Object *subSchema = props->getObject(*name);
        if (subSchema != nullptr)
          validateAgainstSchema(*subSchema, *subData,
                                path + "." + std::string(*name), errors);
      }
    }
  }
  if (const json::Object *items = schema.getObject("items")) {
    const json::Array *dataArr = data.getAsArray();
    if (dataArr == nullptr) {
      errors.push_back(path + ": expected array");
      return;
    }
    size_t index = 0;
    for (const json::Value &element : *dataArr)
      validateAgainstSchema(*items, element,
                            path + "[" + std::to_string(index++) + "]", errors);
  }
}

TEST(PlanSchemaTest, RoundTrip) {
  Plan original = makeSamplePlan();
  std::string text = toJson(original);

  std::optional<json::Value> parsed = parseText(text);
  ASSERT_TRUE(parsed.has_value()) << "toJson output must be valid JSON";

  std::optional<Plan> reparsed = parsePlan(*parsed);
  ASSERT_TRUE(reparsed.has_value()) << "sample plan must re-parse";

  EXPECT_EQ(reparsed->graphId, original.graphId);

  ASSERT_EQ(reparsed->tensors.size(), original.tensors.size());
  for (size_t i = 0; i < original.tensors.size(); ++i) {
    EXPECT_EQ(reparsed->tensors[i].name, original.tensors[i].name);
    EXPECT_EQ(reparsed->tensors[i].shape, original.tensors[i].shape);
    EXPECT_EQ(reparsed->tensors[i].dtype, original.tensors[i].dtype);
    EXPECT_EQ(reparsed->tensors[i].placement, original.tensors[i].placement);
  }

  EXPECT_EQ(reparsed->device.name, original.device.name);
  EXPECT_EQ(reparsed->device.arch, original.device.arch);
  EXPECT_EQ(reparsed->device.gridX, original.device.gridX);
  EXPECT_EQ(reparsed->device.gridY, original.device.gridY);
  EXPECT_EQ(reparsed->device.l1UsableBytes, original.device.l1UsableBytes);
  EXPECT_EQ(reparsed->device.flopsPerCyclePerCore,
            original.device.flopsPerCyclePerCore);
  EXPECT_EQ(reparsed->device.dramBandwidthBytesPerSecond,
            original.device.dramBandwidthBytesPerSecond);

  EXPECT_EQ(reparsed->layout.activationSharding,
            original.layout.activationSharding);
  EXPECT_EQ(reparsed->layout.weightStrategy, original.layout.weightStrategy);

  EXPECT_EQ(reparsed->shardGrid.gridY, original.shardGrid.gridY);
  EXPECT_EQ(reparsed->shardGrid.gridX, original.shardGrid.gridX);

  ASSERT_EQ(reparsed->cbSpecs.size(), original.cbSpecs.size());
  for (size_t i = 0; i < original.cbSpecs.size(); ++i) {
    EXPECT_EQ(reparsed->cbSpecs[i].index, original.cbSpecs[i].index);
    EXPECT_EQ(reparsed->cbSpecs[i].depth, original.cbSpecs[i].depth);
    EXPECT_EQ(reparsed->cbSpecs[i].tileBytes, original.cbSpecs[i].tileBytes);
  }

  EXPECT_EQ(reparsed->noc.dramReadBytes, original.noc.dramReadBytes);
  EXPECT_EQ(reparsed->noc.dramWriteBytes, original.noc.dramWriteBytes);
  EXPECT_EQ(reparsed->noc.l1Bytes, original.noc.l1Bytes);
  EXPECT_EQ(reparsed->noc.multicastBytes, original.noc.multicastBytes);

  EXPECT_EQ(reparsed->emission.mapping, original.emission.mapping);
  EXPECT_EQ(reparsed->emission.coresUsed, original.emission.coresUsed);
  EXPECT_EQ(reparsed->emission.maxTilesPerCore,
            original.emission.maxTilesPerCore);
  EXPECT_EQ(reparsed->emission.dramReadBytes,
            original.emission.dramReadBytes);
  EXPECT_EQ(reparsed->emission.dramWriteBytes,
            original.emission.dramWriteBytes);
  EXPECT_EQ(reparsed->emission.fusedStageTiles,
            original.emission.fusedStageTiles);
  EXPECT_EQ(reparsed->emission.fusedFeasible,
            original.emission.fusedFeasible);

  EXPECT_EQ(reparsed->perf.estCycles, original.perf.estCycles);
  EXPECT_DOUBLE_EQ(reparsed->perf.computeUtil, original.perf.computeUtil);
  EXPECT_EQ(reparsed->perf.bottleneck, original.perf.bottleneck);
  EXPECT_EQ(reparsed->perf.suggestions, original.perf.suggestions);

  EXPECT_EQ(reparsed->viz.coreGrid, original.viz.coreGrid);
  EXPECT_EQ(reparsed->viz.shards, original.viz.shards);
  EXPECT_EQ(reparsed->viz.nocRoutes, original.viz.nocRoutes);
  EXPECT_EQ(reparsed->viz.cbUsage, original.viz.cbUsage);
  EXPECT_EQ(reparsed->viz.timeline, original.viz.timeline);
}

TEST(PlanSchemaTest, ConformsToSchema) {
  std::optional<json::Value> schemaValue = parseText(readSchemaFile());
  ASSERT_TRUE(schemaValue.has_value()) << "schema must be valid JSON";
  const json::Object *schema = schemaValue->getAsObject();
  ASSERT_NE(schema, nullptr);

  std::optional<llvm::StringRef> draft = schema->getString("$schema");
  ASSERT_TRUE(draft.has_value());
  EXPECT_TRUE(draft->contains("draft-07"));

  std::optional<json::Value> planValue = parseText(toJson(makeSamplePlan()));
  ASSERT_TRUE(planValue.has_value());

  std::vector<std::string> errors;
  validateAgainstSchema(*schema, *planValue, "$", errors);
  EXPECT_TRUE(errors.empty()) << "sample plan must conform to the schema";
  for (const std::string &error : errors)
    ADD_FAILURE() << error;
}

TEST(PlanSchemaTest, RejectsMissingField) {
  // C++ parse seam: dropping a required top-level field is rejected.
  std::optional<json::Value> planValue = parseText(toJson(makeSamplePlan()));
  ASSERT_TRUE(planValue.has_value());
  json::Object *planObj = planValue->getAsObject();
  ASSERT_NE(planObj, nullptr);
  ASSERT_TRUE(planObj->erase("graph_id"));
  EXPECT_FALSE(parsePlan(*planValue).has_value())
      << "parsePlan must reject a plan missing graph_id";

  // Schema seam: the same omission is flagged by the schema validator.
  std::optional<json::Value> schemaValue = parseText(readSchemaFile());
  ASSERT_TRUE(schemaValue.has_value());
  const json::Object *schema = schemaValue->getAsObject();
  ASSERT_NE(schema, nullptr);
  std::vector<std::string> errors;
  validateAgainstSchema(*schema, *planValue, "$", errors);
  EXPECT_FALSE(errors.empty())
      << "schema validator must flag the missing graph_id";

  // The device block is required too: a plan without one is rejected.
  std::optional<json::Value> withoutDevice =
      parseText(toJson(makeSamplePlan()));
  ASSERT_TRUE(withoutDevice.has_value());
  json::Object *withoutDeviceObj = withoutDevice->getAsObject();
  ASSERT_NE(withoutDeviceObj, nullptr);
  ASSERT_TRUE(withoutDeviceObj->erase("device"));
  EXPECT_FALSE(parsePlan(*withoutDevice).has_value())
      << "parsePlan must reject a plan missing the device block";
  std::vector<std::string> deviceErrors;
  validateAgainstSchema(*schema, *withoutDevice, "$", deviceErrors);
  EXPECT_FALSE(deviceErrors.empty())
      << "schema validator must flag the missing device block";

  // The emission block is required too: the plan JSON records the projected
  // emitted-path accounting, so a plan without it is rejected at both seams.
  std::optional<json::Value> withoutEmission =
      parseText(toJson(makeSamplePlan()));
  ASSERT_TRUE(withoutEmission.has_value());
  json::Object *withoutEmissionObj = withoutEmission->getAsObject();
  ASSERT_NE(withoutEmissionObj, nullptr);
  ASSERT_TRUE(withoutEmissionObj->erase("emission"));
  EXPECT_FALSE(parsePlan(*withoutEmission).has_value())
      << "parsePlan must reject a plan missing the emission block";
  std::vector<std::string> emissionErrors;
  validateAgainstSchema(*schema, *withoutEmission, "$", emissionErrors);
  EXPECT_FALSE(emissionErrors.empty())
      << "schema validator must flag the missing emission block";

  // A required field inside the emission block is rejected field-for-field.
  std::optional<json::Value> badEmission =
      parseText(toJson(makeSamplePlan()));
  ASSERT_TRUE(badEmission.has_value());
  json::Object *badEmissionObj = badEmission->getAsObject();
  ASSERT_NE(badEmissionObj, nullptr);
  json::Value *emissionValue = badEmissionObj->get("emission");
  ASSERT_NE(emissionValue, nullptr);
  json::Object *emissionObj = emissionValue->getAsObject();
  ASSERT_NE(emissionObj, nullptr);
  ASSERT_TRUE(emissionObj->erase("fused_feasible"));
  EXPECT_FALSE(parsePlan(*badEmission).has_value())
      << "parsePlan must reject an emission block missing fused_feasible";
}

} // namespace
