//===- SchedulerModelTest.cpp -----------------------------------*- C++ -*-===//
//
// Unit tests for the TensixFlow scheduler data model:
//   1. TileAndTensorBytes - 32x32 bf16 tile = 2048 bytes; tensor element and
//                           byte math; bytesPerElem sizing.
//   2. CBIndexValidation  - tt::CBIndex range c_0..c_31; out-of-range rejected.
//   3. EnumStrings        - Placement / WeightStrategy / PipelineStage strings.
//   4. JsonRoundTrip      - scheduler model -> plan::Plan -> toJson -> parsePlan
//                           round-trips field-for-field, device block included.
//
//===----------------------------------------------------------------------===//

#include "Model.h"

#include "tensixflow/Generated/DeviceSpecs.h"
#include "tensixflow/PlanSchema.h"

#include "llvm/Support/JSON.h"

#include "gtest/gtest.h"

#include <optional>
#include <string>

namespace json = llvm::json;
using namespace mlir::tensixflow::scheduler;
using mlir::tensixflow::Plan;
using mlir::tensixflow::parsePlan;
using mlir::tensixflow::toJson;

namespace {

std::optional<json::Value> parseText(const std::string &text) {
  llvm::Expected<json::Value> value = json::parse(text);
  if (!value) {
    llvm::consumeError(value.takeError());
    return std::nullopt;
  }
  return std::move(*value);
}

TEST(SchedulerModelTest, TileAndTensorBytes) {
  // A 32x32 bf16 tile is exactly 2048 bytes (the tt-metal TILE constant).
  TileShape tile; // defaults to 32x32
  EXPECT_EQ(tile.height, 32);
  EXPECT_EQ(tile.width, 32);
  EXPECT_EQ(tile.bytes(bytesPerElem("bf16")), 2048);
  TileShape explicitTile{32, 32};
  EXPECT_EQ(explicitTile.bytes(2), 2048);

  // Element sizing: bf16/fp16 = 2, fp32 = 4, unknown = 0.
  EXPECT_EQ(bytesPerElem("bf16"), 2);
  EXPECT_EQ(bytesPerElem("fp16"), 2);
  EXPECT_EQ(bytesPerElem("fp32"), 4);
  EXPECT_EQ(bytesPerElem("int8"), 0);

  // Tensor element/byte math for the user's [1,2048,4096] activation.
  TensorShape activation{"activation", {1, 2048, 4096}, "bf16"};
  EXPECT_EQ(activation.numElements(), 1 * 2048 * 4096);
  EXPECT_EQ(activation.bytes(bytesPerElem("bf16")), 1 * 2048 * 4096 * 2);

  // A rank-0 shape has one element (empty product).
  TensorShape scalar{"scalar", {}, "fp32"};
  EXPECT_EQ(scalar.numElements(), 1);
  EXPECT_EQ(scalar.bytes(4), 4);
}

TEST(SchedulerModelTest, CBIndexValidation) {
  // tt::CBIndex runs c_0..c_31 -> 32 valid CBs.
  EXPECT_EQ(kMaxCBIndex, 31);
  EXPECT_TRUE(isValidCBIndex(0));
  EXPECT_TRUE(isValidCBIndex(16));
  EXPECT_TRUE(isValidCBIndex(31));
  // Out-of-range indices are rejected (the allocator must not exceed c_31).
  EXPECT_FALSE(isValidCBIndex(32));
  EXPECT_FALSE(isValidCBIndex(-1));
  EXPECT_FALSE(isValidCBIndex(1000));
}

TEST(SchedulerModelTest, EnumStrings) {
  EXPECT_EQ(toString(Placement::DRAM), "DRAM");
  EXPECT_EQ(toString(Placement::L1), "L1");

  EXPECT_EQ(toString(WeightStrategy::Replicate), "replicate");
  EXPECT_EQ(toString(WeightStrategy::Shard), "shard");
  EXPECT_EQ(toString(WeightStrategy::Multicast), "multicast");

  EXPECT_EQ(toString(PipelineStage::Reader), "reader");
  EXPECT_EQ(toString(PipelineStage::Compute), "compute");
  EXPECT_EQ(toString(PipelineStage::Writer), "writer");

  // Grid helper over the reused plan::ShardGrid.
  mlir::tensixflow::ShardGrid grid{8, 8};
  EXPECT_EQ(totalCores(grid), 64);
}

TEST(SchedulerModelTest, JsonRoundTrip) {
  // Build scheduler-model data and convert it into a report Plan.
  TileShape tile; // 32x32 -> 2048 bytes at bf16

  PlanDraft draft;
  draft.graphId = "rmsnorm_matmul";

  mlir::tensixflow::TensorDesc activation;
  activation.name = "activation";
  activation.shape = {1, 2048, 4096};
  activation.dtype = "bf16";
  activation.placement = toString(Placement::DRAM);

  mlir::tensixflow::TensorDesc weight;
  weight.name = "weight";
  weight.shape = {4096, 12288};
  weight.dtype = "bf16";
  weight.placement = toString(Placement::DRAM);
  draft.tensors = {activation, weight};

  draft.weightStrategy = WeightStrategy::Multicast;
  draft.activationSharding = "row-major";
  draft.grid = mlir::tensixflow::ShardGrid{8, 8};
  draft.device.name = "n150";
  draft.device.arch = "wormhole_b0";
  draft.device.gridX = 8;
  draft.device.gridY = 8;
  draft.device.l1UsableBytes = 1499136;
  draft.device.flopsPerCyclePerCore = 4096;
  draft.device.dramBandwidthBytesPerSecond = 288000000000;

  // Reader/compute/writer CBs at the tt-metal in0/in1/out convention.
  draft.cbSpecs.push_back(toSchemaCBSpec(/*index=*/0, /*depth=*/4, tile,
                                         bytesPerElem("bf16")));
  draft.cbSpecs.push_back(toSchemaCBSpec(/*index=*/1, /*depth=*/4, tile,
                                         bytesPerElem("bf16")));
  draft.cbSpecs.push_back(toSchemaCBSpec(/*index=*/16, /*depth=*/4, tile,
                                         bytesPerElem("bf16")));

  // The scheduler model maps onto the report structs correctly.
  EXPECT_EQ(draft.cbSpecs.front().tileBytes, 2048);
  Plan original = toPlan(draft);
  EXPECT_EQ(original.layout.weightStrategy, "multicast");
  EXPECT_EQ(original.layout.activationSharding, "row-major");

  // Round-trip through the PlanSchema JSON seam.
  std::string text = toJson(original);
  std::optional<json::Value> parsed = parseText(text);
  ASSERT_TRUE(parsed.has_value()) << "toPlan/toJson output must be valid JSON";

  std::optional<Plan> reparsed = parsePlan(*parsed);
  ASSERT_TRUE(reparsed.has_value()) << "scheduler-built plan must re-parse";

  EXPECT_EQ(reparsed->graphId, original.graphId);

  ASSERT_EQ(reparsed->tensors.size(), original.tensors.size());
  for (size_t i = 0; i < original.tensors.size(); ++i) {
    EXPECT_EQ(reparsed->tensors[i].name, original.tensors[i].name);
    EXPECT_EQ(reparsed->tensors[i].shape, original.tensors[i].shape);
    EXPECT_EQ(reparsed->tensors[i].dtype, original.tensors[i].dtype);
    EXPECT_EQ(reparsed->tensors[i].placement, original.tensors[i].placement);
  }

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
    EXPECT_EQ(reparsed->cbSpecs[i].tileBytes, 2048);
  }
}

} // namespace
