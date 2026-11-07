//===- tensixflow-translate.cpp - TensixFlow translate driver ---*- C++ -*-===//
//
// TensixFlow: An MLIR Compiler for Tile-Native LLM Kernels on
// Tenstorrent-Style Spatial Accelerators.
//
//===----------------------------------------------------------------------===//
//
// `tensixflow-translate` driver modeled on llvm-project's
// mlir/tools/mlir-translate. Registers a single translation,
// `--tensixflow-to-json`, that serializes a tensixflow module to the plan JSON
// contract in schema/plan.schema.json. A planned module serializes the full
// plan, including the required device block; an unplanned module falls back to
// the function/argument/op skeleton.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include "tensixflow/Generated/DeviceSpecs.h"
#include "tensixflow/TensixflowAttrs.h"
#include "tensixflow/TensixflowDialect.h"
#include "tensixflow/TensixflowOps.h"

#include "DeviceSelection.h"

using namespace mlir;

namespace {

/// Register the dialects the input module may reference so it parses.
void registerTensixflowDialects(DialectRegistry &registry) {
  registry.insert<mlir::tensixflow::TensixflowDialect,
                  mlir::func::FuncDialect, mlir::arith::ArithDialect>();
}

/// Render an MLIR type to its canonical textual form.
std::string typeToString(Type type) {
  std::string result;
  llvm::raw_string_ostream os(result);
  os << type;
  return result;
}

/// Element-dtype name for the plan JSON ("bf16" / "fp16" / "fp32").
std::string elemDtype(Type elem) {
  if (elem.isBF16())
    return "bf16";
  if (elem.isF16())
    return "fp16";
  if (elem.isF32())
    return "fp32";
  return "bf16";
}

/// Serialize a planned module (one carrying a #tensixflow.schedule attribute) to
/// the plan JSON. The document carries the report fields at the top level
/// (graph_id, activation_sharding, weight_strategy, intermediate_storage,
/// fusion, estimated_dram_reads_saved, estimated_intermediate_writes_saved)
/// plus the schema/plan.schema.json nested objects (device, tensors, layout,
/// shard_grid, cb_specs, noc, perf, viz). Field names mirror PlanSchema.cpp.
/// The perf and viz sections are placeholders that the perf model and the
/// visualizer fill later.
LogicalResult planToJson(Operation *op,
                         mlir::tensixflow::TensixflowScheduleAttr schedule,
                         raw_ostream &os) {
  namespace json = llvm::json;

  // Device: the module attribute stamped by the planning passes, then the
  // driver selection, then the generated default. The plan schema requires the
  // block, so an unknown stamped name falls back instead of emitting a plan the
  // report tool must reject.
  const mlir::tensixflow::device::DeviceSpec *deviceSpec =
      mlir::tensixflow::device::defaultDevice();
  if (auto deviceAttr =
          op->getAttrOfType<StringAttr>(mlir::tensixflow::device::kDeviceAttrName)) {
    if (const auto *looked =
            mlir::tensixflow::device::lookup(deviceAttr.getValue())) {
      deviceSpec = looked;
    }
  }

  std::string graphId = "graph";
  op->walk([&](mlir::tensixflow::TensixflowFuncOp func) {
    if (graphId == "graph")
      graphId = func.getName().str();
  });

  StringRef activationSharding = schedule.getActivationSharding().getValue();
  StringRef weightStrategy = schedule.getWeightStrategy().getValue();

  json::Object device;
  device["name"] = deviceSpec->name;
  device["arch"] = deviceSpec->arch;
  device["grid_x"] = deviceSpec->computeGrid.gridX;
  device["grid_y"] = deviceSpec->computeGrid.gridY;
  device["l1_usable_bytes"] = deviceSpec->l1UsableBytes;
  device["flops_per_cycle_per_core"] = deviceSpec->flopsPerCyclePerCore;
  device["dram_bandwidth_bytes_per_second"] =
      deviceSpec->dramBandwidthBytesPerSecond;

  json::Object root;
  root["graph_id"] = graphId;
  root["device"] = json::Value(std::move(device));
  root["activation_sharding"] = activationSharding.str();
  root["weight_strategy"] = weightStrategy.str();
  root["intermediate_storage"] =
      schedule.getIntermediateStorage().getValue().str();
  root["fusion"] = schedule.getFusion().getValue().str();
  root["estimated_dram_reads_saved"] = schedule.getSavedDramReadBytes();
  root["estimated_intermediate_writes_saved"] =
      schedule.getSavedDramWriteBytes();

  // Tensors referenced by the plan: the function's ranked-tensor arguments, all
  // placed in DRAM (the planner reads activations + weights from DRAM).
  json::Array tensors;
  op->walk([&](mlir::tensixflow::TensixflowFuncOp func) {
    ArrayRef<Type> argTypes = func.getArgumentTypes();
    for (size_t i = 0, e = argTypes.size(); i < e; ++i) {
      auto ranked = dyn_cast<RankedTensorType>(argTypes[i]);
      if (!ranked)
        continue;
      json::Array shape;
      for (int64_t dim : ranked.getShape())
        shape.push_back(json::Value(dim));
      json::Object tensor;
      tensor["name"] = (llvm::Twine("arg") + llvm::Twine(i)).str();
      tensor["shape"] = json::Value(std::move(shape));
      tensor["dtype"] = elemDtype(ranked.getElementType());
      tensor["placement"] = "DRAM";
      tensors.push_back(json::Value(std::move(tensor)));
    }
  });
  root["tensors"] = json::Value(std::move(tensors));

  json::Object layout;
  layout["activation_sharding"] = activationSharding.str();
  layout["weight_strategy"] = weightStrategy.str();
  root["layout"] = json::Value(std::move(layout));

  json::Object shardGrid;
  shardGrid["grid_y"] = static_cast<int64_t>(schedule.getGridY());
  shardGrid["grid_x"] = static_cast<int64_t>(schedule.getGridX());
  root["shard_grid"] = json::Value(std::move(shardGrid));

  ArrayAttr cbIndices = schedule.getCbIndices();
  ArrayAttr cbDepths = schedule.getCbDepths();
  ArrayAttr cbTileBytes = schedule.getCbTileBytes();
  json::Array cbSpecs;
  for (size_t i = 0, e = cbIndices.size(); i < e; ++i) {
    json::Object cb;
    cb["index"] = cast<IntegerAttr>(cbIndices[i]).getInt();
    cb["depth"] = cast<IntegerAttr>(cbDepths[i]).getInt();
    cb["tile_bytes"] = cast<IntegerAttr>(cbTileBytes[i]).getInt();
    cbSpecs.push_back(json::Value(std::move(cb)));
  }
  root["cb_specs"] = json::Value(std::move(cbSpecs));

  json::Object noc;
  noc["dram_read_bytes"] = schedule.getDramReadBytes();
  noc["dram_write_bytes"] = schedule.getDramWriteBytes();
  noc["l1_bytes"] = schedule.getL1Bytes();
  noc["multicast_bytes"] = schedule.getMulticastBytes();
  root["noc"] = json::Value(std::move(noc));

  // Placeholder perf block. The translate tool reports the plan it was given;
  // tensixflow-report recomputes this section from the device data. The neutral
  // classification is valid on its own so a raw plan JSON stays schema-valid.
  json::Object perf;
  perf["est_cycles"] = static_cast<int64_t>(0);
  perf["compute_util"] = 0.0;
  perf["bottleneck"] = "balanced";
  perf["suggestions"] = json::Value(json::Array{});
  root["perf"] = json::Value(std::move(perf));

  json::Object viz;
  viz["core_grid"] = "";
  viz["shards"] = "";
  viz["noc_routes"] = "";
  viz["cb_usage"] = "";
  viz["timeline"] = "";
  root["viz"] = json::Value(std::move(viz));

  os << llvm::formatv("{0:2}", json::Value(std::move(root))) << "\n";
  return success();
}

/// Serialize a tensixflow module that has no plan attribute to a JSON skeleton.
///
/// Shape (skeleton, not the full plan schema):
///   {
///     "dialect": "tensixflow",
///     "functions": [
///       {
///         "name": "...",
///         "arguments": ["tensor<...>", ...],
///         "results":   ["tensor<...>", ...],
///         "ops": [ { "name": "tensixflow.gelu",
///                    "num_operands": 1, "num_results": 1 }, ... ]
///       }
///     ]
///   }
LogicalResult tensixflowToJson(Operation *op, raw_ostream &os) {
  namespace json = llvm::json;

  // A planned module carries a #tensixflow.schedule attribute (attached by
  // --tensixflow-plan-noc-sharding + --tensixflow-allocate-circular-buffers);
  // serialize it as the plan JSON. Otherwise fall back to the IR skeleton.
  if (auto schedule =
          op->getAttrOfType<mlir::tensixflow::TensixflowScheduleAttr>(
              "tensixflow.schedule"))
    return planToJson(op, schedule, os);

  json::Array functions;
  op->walk([&](mlir::tensixflow::TensixflowFuncOp func) {
    json::Object function;
    function["name"] = func.getName().str();

    json::Array arguments;
    for (Type type : func.getArgumentTypes())
      arguments.push_back(json::Value(typeToString(type)));
    function["arguments"] = std::move(arguments);

    json::Array results;
    for (Type type : func.getResultTypes())
      results.push_back(json::Value(typeToString(type)));
    function["results"] = std::move(results);

    json::Array ops;
    for (Operation &nested : func.getBody().front()) {
      json::Object entry;
      entry["name"] = nested.getName().getStringRef().str();
      entry["num_operands"] = static_cast<int64_t>(nested.getNumOperands());
      entry["num_results"] = static_cast<int64_t>(nested.getNumResults());
      ops.push_back(json::Value(std::move(entry)));
    }
    function["ops"] = std::move(ops);

    functions.push_back(json::Value(std::move(function)));
  });

  json::Object root;
  root["dialect"] = "tensixflow";
  root["functions"] = std::move(functions);

  // Pretty-print with a 2-space indent; output is valid JSON (json.load-safe).
  os << llvm::formatv("{0:2}", json::Value(std::move(root))) << "\n";
  return success();
}

} // namespace

// Self-registers before main(): the `--tensixflow-to-json` translation plus the
// dialects needed to parse its input.
static TranslateFromMLIRRegistration reg(
    "tensixflow-to-json", "Serialize a tensixflow module to JSON",
    tensixflowToJson, registerTensixflowDialects);

int main(int argc, char **argv) {
  return failed(
      mlirTranslateMain(argc, argv, "TensixFlow MLIR translation tool\n"));
}
