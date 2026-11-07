//===- PlanSchema.cpp -------------------------------------------*- C++ -*-===//
//
// Serialization (Plan -> llvm::json) for the TensixFlow plan/perf/viz data
// model. See PlanSchema.h: the single source of truth consumed by the
// scheduler, the plan/perf report, and the HTML visualizer. Parsing lives in
// PlanSchemaParse.cpp.
//
//===----------------------------------------------------------------------===//

#include "tensixflow/PlanSchema.h"

#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace json = llvm::json;

namespace mlir {
namespace tensixflow {

static json::Value deviceToJson(const DeviceInfo &device) {
  json::Object obj;
  obj["name"] = device.name;
  obj["arch"] = device.arch;
  obj["grid_x"] = device.gridX;
  obj["grid_y"] = device.gridY;
  obj["l1_usable_bytes"] = device.l1UsableBytes;
  obj["flops_per_cycle_per_core"] = device.flopsPerCyclePerCore;
  obj["dram_bandwidth_bytes_per_second"] =
      device.dramBandwidthBytesPerSecond;
  return json::Value(std::move(obj));
}

static json::Value tensorToJson(const TensorDesc &tensor) {
  json::Array shape;
  for (int64_t dim : tensor.shape) {
    shape.push_back(json::Value(dim));
  }
  json::Object obj;
  obj["name"] = tensor.name;
  obj["shape"] = json::Value(std::move(shape));
  obj["dtype"] = tensor.dtype;
  obj["placement"] = tensor.placement;
  return json::Value(std::move(obj));
}

static json::Value cbToJson(const CBSpec &cb) {
  json::Object obj;
  obj["index"] = cb.index;
  obj["depth"] = cb.depth;
  obj["tile_bytes"] = cb.tileBytes;
  return json::Value(std::move(obj));
}

static json::Value planToValue(const Plan &plan) {
  json::Array tensors;
  for (const TensorDesc &tensor : plan.tensors) {
    tensors.push_back(tensorToJson(tensor));
  }

  json::Array cbSpecs;
  for (const CBSpec &cb : plan.cbSpecs) {
    cbSpecs.push_back(cbToJson(cb));
  }

  json::Array suggestions;
  for (const std::string &suggestion : plan.perf.suggestions) {
    suggestions.push_back(json::Value(suggestion));
  }

  json::Object layout;
  layout["activation_sharding"] = plan.layout.activationSharding;
  layout["weight_strategy"] = plan.layout.weightStrategy;

  json::Object shardGrid;
  shardGrid["grid_y"] = plan.shardGrid.gridY;
  shardGrid["grid_x"] = plan.shardGrid.gridX;

  json::Object noc;
  noc["dram_read_bytes"] = plan.noc.dramReadBytes;
  noc["dram_write_bytes"] = plan.noc.dramWriteBytes;
  noc["l1_bytes"] = plan.noc.l1Bytes;
  noc["multicast_bytes"] = plan.noc.multicastBytes;

  json::Object perf;
  perf["est_cycles"] = plan.perf.estCycles;
  perf["compute_util"] = plan.perf.computeUtil;
  perf["bottleneck"] = plan.perf.bottleneck;
  perf["suggestions"] = json::Value(std::move(suggestions));

  json::Object viz;
  viz["core_grid"] = plan.viz.coreGrid;
  viz["shards"] = plan.viz.shards;
  viz["noc_routes"] = plan.viz.nocRoutes;
  viz["cb_usage"] = plan.viz.cbUsage;
  viz["timeline"] = plan.viz.timeline;

  json::Object root;
  root["graph_id"] = plan.graphId;
  root["device"] = deviceToJson(plan.device);
  root["tensors"] = json::Value(std::move(tensors));
  root["layout"] = json::Value(std::move(layout));
  root["shard_grid"] = json::Value(std::move(shardGrid));
  root["cb_specs"] = json::Value(std::move(cbSpecs));
  root["noc"] = json::Value(std::move(noc));
  root["perf"] = json::Value(std::move(perf));
  root["viz"] = json::Value(std::move(viz));
  return json::Value(std::move(root));
}

std::string toJson(const Plan &plan) {
  std::string out;
  llvm::raw_string_ostream os(out);
  os << llvm::formatv("{0:2}", planToValue(plan)) << "\n";
  return out;
}

} // namespace tensixflow
} // namespace mlir
