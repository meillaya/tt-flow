//===- PlanSchemaParse.cpp --------------------------------------*- C++ -*-===//
//
// Parsing (llvm::json -> Plan) for the TensixFlow plan/perf/viz data model,
// field-for-field with required-field rejection. See PlanSchema.h. The
// serialization half lives in PlanSchema.cpp.
//
// Each accessor returns nullopt when a key is absent or wrongly typed, so a
// malformed plan, including one without a device block, is rejected instead of
// silently defaulted.
//
//===----------------------------------------------------------------------===//

#include "tensixflow/PlanSchema.h"

#include "llvm/Support/JSON.h"

#include <cstdint>
#include <optional>
#include <string>

namespace json = llvm::json;

namespace mlir {
namespace tensixflow {

namespace {

std::optional<std::string> reqString(const json::Object &obj, const char *key) {
  std::optional<llvm::StringRef> value = obj.getString(key);
  if (!value) {
    return std::nullopt;
  }
  return std::string(*value);
}

std::optional<int64_t> reqInt(const json::Object &obj, const char *key) {
  return obj.getInteger(key);
}

std::optional<double> reqDouble(const json::Object &obj, const char *key) {
  return obj.getNumber(key);
}

std::optional<bool> reqBool(const json::Object &obj, const char *key) {
  return obj.getBoolean(key);
}

const json::Object *reqObject(const json::Object &obj, const char *key) {
  return obj.getObject(key);
}

const json::Array *reqArray(const json::Object &obj, const char *key) {
  return obj.getArray(key);
}

std::optional<DeviceInfo> parseDevice(const json::Object &obj) {
  DeviceInfo device;
  if (auto name = reqString(obj, "name")) {
    device.name = *name;
  } else {
    return std::nullopt;
  }
  if (auto arch = reqString(obj, "arch")) {
    device.arch = *arch;
  } else {
    return std::nullopt;
  }
  if (auto gridX = reqInt(obj, "grid_x")) {
    device.gridX = *gridX;
  } else {
    return std::nullopt;
  }
  if (auto gridY = reqInt(obj, "grid_y")) {
    device.gridY = *gridY;
  } else {
    return std::nullopt;
  }
  if (auto l1 = reqInt(obj, "l1_usable_bytes")) {
    device.l1UsableBytes = *l1;
  } else {
    return std::nullopt;
  }
  if (auto flops = reqInt(obj, "flops_per_cycle_per_core")) {
    device.flopsPerCyclePerCore = *flops;
  } else {
    return std::nullopt;
  }
  if (auto bandwidth = reqInt(obj, "dram_bandwidth_bytes_per_second")) {
    device.dramBandwidthBytesPerSecond = *bandwidth;
  } else {
    return std::nullopt;
  }
  return device;
}

std::optional<TensorDesc> parseTensor(const json::Object &obj) {
  TensorDesc tensor;
  if (auto name = reqString(obj, "name")) {
    tensor.name = *name;
  } else {
    return std::nullopt;
  }
  if (auto dtype = reqString(obj, "dtype")) {
    tensor.dtype = *dtype;
  } else {
    return std::nullopt;
  }
  if (auto placement = reqString(obj, "placement")) {
    tensor.placement = *placement;
  } else {
    return std::nullopt;
  }
  const json::Array *shape = reqArray(obj, "shape");
  if (!shape) {
    return std::nullopt;
  }
  for (const json::Value &dim : *shape) {
    std::optional<int64_t> extent = dim.getAsInteger();
    if (!extent) {
      return std::nullopt;
    }
    tensor.shape.push_back(*extent);
  }
  return tensor;
}

std::optional<CBSpec> parseCB(const json::Object &obj) {
  CBSpec cb;
  if (auto index = reqInt(obj, "index")) {
    cb.index = *index;
  } else {
    return std::nullopt;
  }
  if (auto depth = reqInt(obj, "depth")) {
    cb.depth = *depth;
  } else {
    return std::nullopt;
  }
  if (auto tileBytes = reqInt(obj, "tile_bytes")) {
    cb.tileBytes = *tileBytes;
  } else {
    return std::nullopt;
  }
  return cb;
}

std::optional<Layout> parseLayout(const json::Object &obj) {
  Layout layout;
  if (auto sharding = reqString(obj, "activation_sharding")) {
    layout.activationSharding = *sharding;
  } else {
    return std::nullopt;
  }
  if (auto strategy = reqString(obj, "weight_strategy")) {
    layout.weightStrategy = *strategy;
  } else {
    return std::nullopt;
  }
  return layout;
}

std::optional<ShardGrid> parseShardGrid(const json::Object &obj) {
  ShardGrid grid;
  if (auto gridY = reqInt(obj, "grid_y")) {
    grid.gridY = *gridY;
  } else {
    return std::nullopt;
  }
  if (auto gridX = reqInt(obj, "grid_x")) {
    grid.gridX = *gridX;
  } else {
    return std::nullopt;
  }
  return grid;
}

std::optional<NocEstimate> parseNoc(const json::Object &obj) {
  NocEstimate noc;
  if (auto value = reqInt(obj, "dram_read_bytes")) {
    noc.dramReadBytes = *value;
  } else {
    return std::nullopt;
  }
  if (auto value = reqInt(obj, "dram_write_bytes")) {
    noc.dramWriteBytes = *value;
  } else {
    return std::nullopt;
  }
  if (auto value = reqInt(obj, "l1_bytes")) {
    noc.l1Bytes = *value;
  } else {
    return std::nullopt;
  }
  if (auto value = reqInt(obj, "multicast_bytes")) {
    noc.multicastBytes = *value;
  } else {
    return std::nullopt;
  }
  return noc;
}

std::optional<EmissionEstimate> parseEmission(const json::Object &obj) {
  EmissionEstimate emission;
  if (auto mapping = reqString(obj, "mapping")) {
    emission.mapping = *mapping;
  } else {
    return std::nullopt;
  }
  if (auto value = reqInt(obj, "cores_used")) {
    emission.coresUsed = *value;
  } else {
    return std::nullopt;
  }
  if (auto value = reqInt(obj, "max_tiles_per_core")) {
    emission.maxTilesPerCore = *value;
  } else {
    return std::nullopt;
  }
  if (auto value = reqInt(obj, "dram_read_bytes")) {
    emission.dramReadBytes = *value;
  } else {
    return std::nullopt;
  }
  if (auto value = reqInt(obj, "dram_write_bytes")) {
    emission.dramWriteBytes = *value;
  } else {
    return std::nullopt;
  }
  if (auto value = reqInt(obj, "fused_stage_tiles")) {
    emission.fusedStageTiles = *value;
  } else {
    return std::nullopt;
  }
  if (auto value = reqBool(obj, "fused_feasible")) {
    emission.fusedFeasible = *value;
  } else {
    return std::nullopt;
  }
  return emission;
}

std::optional<PerfEstimate> parsePerf(const json::Object &obj) {
  PerfEstimate perf;
  if (auto value = reqInt(obj, "est_cycles")) {
    perf.estCycles = *value;
  } else {
    return std::nullopt;
  }
  if (auto value = reqDouble(obj, "compute_util")) {
    perf.computeUtil = *value;
  } else {
    return std::nullopt;
  }
  if (auto value = reqString(obj, "bottleneck")) {
    perf.bottleneck = *value;
  } else {
    return std::nullopt;
  }
  const json::Array *suggestions = reqArray(obj, "suggestions");
  if (!suggestions) {
    return std::nullopt;
  }
  for (const json::Value &suggestion : *suggestions) {
    std::optional<llvm::StringRef> text = suggestion.getAsString();
    if (!text) {
      return std::nullopt;
    }
    perf.suggestions.emplace_back(*text);
  }
  return perf;
}

std::optional<VizSections> parseViz(const json::Object &obj) {
  VizSections viz;
  if (auto value = reqString(obj, "core_grid")) {
    viz.coreGrid = *value;
  } else {
    return std::nullopt;
  }
  if (auto value = reqString(obj, "shards")) {
    viz.shards = *value;
  } else {
    return std::nullopt;
  }
  if (auto value = reqString(obj, "noc_routes")) {
    viz.nocRoutes = *value;
  } else {
    return std::nullopt;
  }
  if (auto value = reqString(obj, "cb_usage")) {
    viz.cbUsage = *value;
  } else {
    return std::nullopt;
  }
  if (auto value = reqString(obj, "timeline")) {
    viz.timeline = *value;
  } else {
    return std::nullopt;
  }
  return viz;
}

} // namespace

std::optional<Plan> parsePlan(const json::Value &value) {
  const json::Object *root = value.getAsObject();
  if (!root) {
    return std::nullopt;
  }

  Plan plan;
  if (auto graphId = reqString(*root, "graph_id")) {
    plan.graphId = *graphId;
  } else {
    return std::nullopt;
  }

  const json::Object *device = reqObject(*root, "device");
  if (!device) {
    return std::nullopt;
  }
  if (auto parsed = parseDevice(*device)) {
    plan.device = *parsed;
  } else {
    return std::nullopt;
  }

  const json::Array *tensors = reqArray(*root, "tensors");
  if (!tensors) {
    return std::nullopt;
  }
  for (const json::Value &tensorValue : *tensors) {
    const json::Object *tensorObject = tensorValue.getAsObject();
    if (!tensorObject) {
      return std::nullopt;
    }
    if (auto tensor = parseTensor(*tensorObject)) {
      plan.tensors.push_back(*tensor);
    } else {
      return std::nullopt;
    }
  }

  const json::Object *layout = reqObject(*root, "layout");
  if (!layout) {
    return std::nullopt;
  }
  if (auto parsed = parseLayout(*layout)) {
    plan.layout = *parsed;
  } else {
    return std::nullopt;
  }

  const json::Object *shardGrid = reqObject(*root, "shard_grid");
  if (!shardGrid) {
    return std::nullopt;
  }
  if (auto parsed = parseShardGrid(*shardGrid)) {
    plan.shardGrid = *parsed;
  } else {
    return std::nullopt;
  }

  const json::Array *cbSpecs = reqArray(*root, "cb_specs");
  if (!cbSpecs) {
    return std::nullopt;
  }
  for (const json::Value &cbValue : *cbSpecs) {
    const json::Object *cbObject = cbValue.getAsObject();
    if (!cbObject) {
      return std::nullopt;
    }
    if (auto spec = parseCB(*cbObject)) {
      plan.cbSpecs.push_back(*spec);
    } else {
      return std::nullopt;
    }
  }

  const json::Object *noc = reqObject(*root, "noc");
  if (!noc) {
    return std::nullopt;
  }
  if (auto parsed = parseNoc(*noc)) {
    plan.noc = *parsed;
  } else {
    return std::nullopt;
  }

  const json::Object *emission = reqObject(*root, "emission");
  if (!emission) {
    return std::nullopt;
  }
  if (auto parsed = parseEmission(*emission)) {
    plan.emission = *parsed;
  } else {
    return std::nullopt;
  }

  const json::Object *perf = reqObject(*root, "perf");
  if (!perf) {
    return std::nullopt;
  }
  if (auto parsed = parsePerf(*perf)) {
    plan.perf = *parsed;
  } else {
    return std::nullopt;
  }

  const json::Object *viz = reqObject(*root, "viz");
  if (!viz) {
    return std::nullopt;
  }
  if (auto parsed = parseViz(*viz)) {
    plan.viz = *parsed;
  } else {
    return std::nullopt;
  }

  return plan;
}

} // namespace tensixflow
} // namespace mlir
