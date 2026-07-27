//===- PlanSchema.h ---------------------------------------------*- C++ -*-===//
//
// TensixFlow plan/perf/viz data model — the SINGLE SOURCE OF TRUTH (D15) for
// the JSON contract shared by the scheduler (Wave 5), the plan/perf report
// (todo 30), and the HTML visualizer (todo 43). Plain data + llvm::json only;
// NO MLIR dependency, so any stage can link the small MLIRTensixflowSchema lib
// without dragging in the dialect.
//
// JSON field names are snake_case to match the user spec's report format
// (graph_id, dram_read_bytes, weight_strategy, ...). See schema/plan.schema.json
// for the validation contract and docs/schema.md for field documentation.
//
//===----------------------------------------------------------------------===//

#ifndef TENSIXFLOW_PLANSCHEMA_H
#define TENSIXFLOW_PLANSCHEMA_H

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace llvm {
namespace json {
class Value;
} // namespace json
} // namespace llvm

namespace mlir {
namespace tensixflow {

/// A tensor referenced by the plan: its logical name, static shape, element
/// dtype (e.g. "bf16"), and memory placement ("DRAM" or "L1").
struct TensorDesc {
  std::string name;
  std::vector<int64_t> shape;
  std::string dtype;
  std::string placement;
};

/// The core grid the plan maps onto (e.g. an 8x8 Wormhole n150 grid).
struct ShardGrid {
  int64_t gridY = 0;
  int64_t gridX = 0;
};

/// The chosen data layout: how activations are sharded and how weights are
/// distributed ("replicate" / "shard" / "multicast").
struct Layout {
  std::string activationSharding;
  std::string weightStrategy;
};

/// A circular-buffer allocation: CB index, depth (in tiles), and per-tile bytes.
struct CBSpec {
  int64_t index = 0;
  int64_t depth = 0;
  int64_t tileBytes = 0;
};

/// Static NoC traffic estimate, in bytes.
struct NocEstimate {
  int64_t dramReadBytes = 0;
  int64_t dramWriteBytes = 0;
  int64_t l1Bytes = 0;
  int64_t multicastBytes = 0;
};

/// Static roofline-style performance estimate.
struct PerfEstimate {
  int64_t estCycles = 0;
  double computeUtil = 0.0;
  std::string bottleneck;
  std::vector<std::string> suggestions;
};

/// Lightweight placeholders for the HTML visualizer sections (todo 43 fills the
/// real detail). Kept as opaque strings so the schema is stable before the viz.
struct VizSections {
  std::string coreGrid;
  std::string shards;
  std::string nocRoutes;
  std::string cbUsage;
  std::string timeline;
};

/// The full plan: everything the scheduler decides and every downstream stage
/// consumes.
struct Plan {
  std::string graphId;
  std::vector<TensorDesc> tensors;
  Layout layout;
  ShardGrid shardGrid;
  std::vector<CBSpec> cbSpecs;
  NocEstimate noc;
  PerfEstimate perf;
  VizSections viz;
};

/// Serialize a plan to pretty-printed JSON (indent 2, trailing newline). The
/// output is guaranteed parseable by llvm::json::parse and by python json.load.
std::string toJson(const Plan &plan);

/// Parse a plan field-for-field from a JSON value. Returns std::nullopt if the
/// value is not an object or any required field is missing / wrongly typed —
/// this is the validation seam: a malformed plan is rejected, not defaulted.
std::optional<Plan> parsePlan(const llvm::json::Value &value);

} // namespace tensixflow
} // namespace mlir

#endif // TENSIXFLOW_PLANSCHEMA_H
