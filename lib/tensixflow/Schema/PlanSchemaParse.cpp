//===- PlanSchemaParse.cpp --------------------------------------*- C++ -*-===//
//
// Parsing (llvm::json -> Plan) for the TensixFlow plan/perf/viz data model,
// field-for-field with required-field rejection. See PlanSchema.h (D15). The
// serialization half lives in PlanSchema.cpp.
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

// Required-field accessors: each returns nullopt when the key is absent or the
// type is wrong, so a malformed plan is rejected rather than silently defaulted.
std::optional<std::string> reqString(const json::Object &obj, const char *key) {
  std::optional<llvm::StringRef> v = obj.getString(key);
  if (!v)
    return std::nullopt;
  return std::string(*v);
}

std::optional<int64_t> reqInt(const json::Object &obj, const char *key) {
  return obj.getInteger(key);
}

std::optional<double> reqDouble(const json::Object &obj, const char *key) {
  return obj.getNumber(key);
}

const json::Object *reqObject(const json::Object &obj, const char *key) {
  return obj.getObject(key);
}

const json::Array *reqArray(const json::Object &obj, const char *key) {
  return obj.getArray(key);
}

std::optional<TensorDesc> parseTensor(const json::Object &obj) {
  TensorDesc t;
  if (auto name = reqString(obj, "name"))
    t.name = *name;
  else
    return std::nullopt;
  if (auto dtype = reqString(obj, "dtype"))
    t.dtype = *dtype;
  else
    return std::nullopt;
  if (auto placement = reqString(obj, "placement"))
    t.placement = *placement;
  else
    return std::nullopt;
  const json::Array *shape = reqArray(obj, "shape");
  if (!shape)
    return std::nullopt;
  for (const json::Value &dim : *shape) {
    std::optional<int64_t> d = dim.getAsInteger();
    if (!d)
      return std::nullopt;
    t.shape.push_back(*d);
  }
  return t;
}

std::optional<CBSpec> parseCB(const json::Object &obj) {
  CBSpec cb;
  if (auto index = reqInt(obj, "index"))
    cb.index = *index;
  else
    return std::nullopt;
  if (auto depth = reqInt(obj, "depth"))
    cb.depth = *depth;
  else
    return std::nullopt;
  if (auto tileBytes = reqInt(obj, "tile_bytes"))
    cb.tileBytes = *tileBytes;
  else
    return std::nullopt;
  return cb;
}

std::optional<Layout> parseLayout(const json::Object &obj) {
  Layout layout;
  if (auto a = reqString(obj, "activation_sharding"))
    layout.activationSharding = *a;
  else
    return std::nullopt;
  if (auto w = reqString(obj, "weight_strategy"))
    layout.weightStrategy = *w;
  else
    return std::nullopt;
  return layout;
}

std::optional<ShardGrid> parseShardGrid(const json::Object &obj) {
  ShardGrid grid;
  if (auto y = reqInt(obj, "grid_y"))
    grid.gridY = *y;
  else
    return std::nullopt;
  if (auto x = reqInt(obj, "grid_x"))
    grid.gridX = *x;
  else
    return std::nullopt;
  return grid;
}

std::optional<NocEstimate> parseNoc(const json::Object &obj) {
  NocEstimate noc;
  if (auto v = reqInt(obj, "dram_read_bytes"))
    noc.dramReadBytes = *v;
  else
    return std::nullopt;
  if (auto v = reqInt(obj, "dram_write_bytes"))
    noc.dramWriteBytes = *v;
  else
    return std::nullopt;
  if (auto v = reqInt(obj, "l1_bytes"))
    noc.l1Bytes = *v;
  else
    return std::nullopt;
  if (auto v = reqInt(obj, "multicast_bytes"))
    noc.multicastBytes = *v;
  else
    return std::nullopt;
  return noc;
}

std::optional<PerfEstimate> parsePerf(const json::Object &obj) {
  PerfEstimate perf;
  if (auto v = reqInt(obj, "est_cycles"))
    perf.estCycles = *v;
  else
    return std::nullopt;
  if (auto v = reqDouble(obj, "compute_util"))
    perf.computeUtil = *v;
  else
    return std::nullopt;
  if (auto v = reqString(obj, "bottleneck"))
    perf.bottleneck = *v;
  else
    return std::nullopt;
  const json::Array *suggestions = reqArray(obj, "suggestions");
  if (!suggestions)
    return std::nullopt;
  for (const json::Value &s : *suggestions) {
    std::optional<llvm::StringRef> str = s.getAsString();
    if (!str)
      return std::nullopt;
    perf.suggestions.emplace_back(*str);
  }
  return perf;
}

std::optional<VizSections> parseViz(const json::Object &obj) {
  VizSections viz;
  if (auto v = reqString(obj, "core_grid"))
    viz.coreGrid = *v;
  else
    return std::nullopt;
  if (auto v = reqString(obj, "shards"))
    viz.shards = *v;
  else
    return std::nullopt;
  if (auto v = reqString(obj, "noc_routes"))
    viz.nocRoutes = *v;
  else
    return std::nullopt;
  if (auto v = reqString(obj, "cb_usage"))
    viz.cbUsage = *v;
  else
    return std::nullopt;
  if (auto v = reqString(obj, "timeline"))
    viz.timeline = *v;
  else
    return std::nullopt;
  return viz;
}

} // namespace

std::optional<Plan> parsePlan(const json::Value &value) {
  const json::Object *root = value.getAsObject();
  if (!root)
    return std::nullopt;

  Plan plan;
  if (auto graphId = reqString(*root, "graph_id"))
    plan.graphId = *graphId;
  else
    return std::nullopt;

  const json::Array *tensors = reqArray(*root, "tensors");
  if (!tensors)
    return std::nullopt;
  for (const json::Value &t : *tensors) {
    const json::Object *tobj = t.getAsObject();
    if (!tobj)
      return std::nullopt;
    if (auto tensor = parseTensor(*tobj))
      plan.tensors.push_back(*tensor);
    else
      return std::nullopt;
  }

  const json::Object *layout = reqObject(*root, "layout");
  if (!layout)
    return std::nullopt;
  if (auto l = parseLayout(*layout))
    plan.layout = *l;
  else
    return std::nullopt;

  const json::Object *shardGrid = reqObject(*root, "shard_grid");
  if (!shardGrid)
    return std::nullopt;
  if (auto g = parseShardGrid(*shardGrid))
    plan.shardGrid = *g;
  else
    return std::nullopt;

  const json::Array *cbSpecs = reqArray(*root, "cb_specs");
  if (!cbSpecs)
    return std::nullopt;
  for (const json::Value &cb : *cbSpecs) {
    const json::Object *cbobj = cb.getAsObject();
    if (!cbobj)
      return std::nullopt;
    if (auto spec = parseCB(*cbobj))
      plan.cbSpecs.push_back(*spec);
    else
      return std::nullopt;
  }

  const json::Object *noc = reqObject(*root, "noc");
  if (!noc)
    return std::nullopt;
  if (auto n = parseNoc(*noc))
    plan.noc = *n;
  else
    return std::nullopt;

  const json::Object *perf = reqObject(*root, "perf");
  if (!perf)
    return std::nullopt;
  if (auto p = parsePerf(*perf))
    plan.perf = *p;
  else
    return std::nullopt;

  const json::Object *viz = reqObject(*root, "viz");
  if (!viz)
    return std::nullopt;
  if (auto v = parseViz(*viz))
    plan.viz = *v;
  else
    return std::nullopt;

  return plan;
}

} // namespace tensixflow
} // namespace mlir
