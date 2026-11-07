//===- Model.cpp ------------------------------------------------*- C++ -*-===//
//
// Implementation of the TensixFlow scheduler data model. See Model.h: pure data
// and helpers with no MLIR dependency. The enum toString helpers use exhaustive
// switches without a default, so adding a variant is a compile-time warning.
//
//===----------------------------------------------------------------------===//

#include "Model.h"

#include <numeric>

namespace mlir {
namespace tensixflow {
namespace scheduler {

int bytesPerElem(const std::string &dtype) {
  if (dtype == "bf16" || dtype == "fp16")
    return 2;
  if (dtype == "fp32")
    return 4;
  return 0;
}

int64_t TensorShape::numElements() const {
  return std::accumulate(dims.begin(), dims.end(), int64_t{1},
                         std::multiplies<int64_t>());
}

int64_t TensorShape::bytes(int bytesPerElem) const {
  return numElements() * bytesPerElem;
}

std::string toString(Placement placement) {
  switch (placement) {
  case Placement::DRAM:
    return "DRAM";
  case Placement::L1:
    return "L1";
  }
  return "unknown";
}

std::string toString(WeightStrategy strategy) {
  switch (strategy) {
  case WeightStrategy::Replicate:
    return "replicate";
  case WeightStrategy::Shard:
    return "shard";
  case WeightStrategy::Multicast:
    return "multicast";
  }
  return "unknown";
}

std::string toString(PipelineStage stage) {
  switch (stage) {
  case PipelineStage::Reader:
    return "reader";
  case PipelineStage::Compute:
    return "compute";
  case PipelineStage::Writer:
    return "writer";
  }
  return "unknown";
}

plan::CBSpec toSchemaCBSpec(int64_t index, int64_t depth, const TileShape &tile,
                            int bytesPerElem) {
  return plan::CBSpec{index, depth, tile.bytes(bytesPerElem)};
}

plan::Layout toSchemaLayout(WeightStrategy strategy,
                            const std::string &activationSharding) {
  plan::Layout layout;
  layout.activationSharding = activationSharding;
  layout.weightStrategy = toString(strategy);
  return layout;
}

plan::Plan toPlan(const PlanDraft &draft) {
  plan::Plan plan;
  plan.graphId = draft.graphId;
  plan.tensors = draft.tensors;
  plan.layout = toSchemaLayout(draft.weightStrategy, draft.activationSharding);
  plan.shardGrid = draft.grid;
  plan.cbSpecs = draft.cbSpecs;
  plan.device = draft.device;
  return plan;
}

} // namespace scheduler
} // namespace tensixflow
} // namespace mlir
