//===- PlanSchema.h ---------------------------------------------*- C++ -*-===//
//
// TensixFlow plan/perf/viz data model: the single source of truth for the JSON
// contract shared by the scheduler, the plan/perf report, and the HTML
// visualizer. Plain data plus llvm::json only; no MLIR dependency, so any stage
// can link the small MLIRTensixflowSchema library without the dialect.
//
// JSON field names are snake_case to match the project report format
// (graph_id, dram_read_bytes, weight_strategy, ...). schema/plan.schema.json is
// the validation contract. Every numeric estimate in a Plan is projected from a
// static model, never measured on hardware.
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

/// The device the plan was scheduled for. The values are copied from
/// spec/devices.json through the generated DeviceSpecs.h table; clock and NoC
/// rates stay out of the JSON because the perf model resolves the full spec by
/// name. The name must exist in the generated table.
struct DeviceInfo {
  std::string name;
  std::string arch;
  int64_t gridX = 0;
  int64_t gridY = 0;
  int64_t l1UsableBytes = 0;
  int64_t flopsPerCyclePerCore = 0;
  int64_t dramBandwidthBytesPerSecond = 0;
};

/// The core grid the plan maps onto (the device's compute grid).
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

/// Projected emitted-path accounting: what the emitted TT-Metalium program will
/// do with the plan's tile geometry. Every value is projected from static
/// shapes, never measured. The mapping names how output tiles are assigned to
/// cores: output-tile-ranges reads the operands of every output tile, and
/// blocked-reuse stages a core's activation row block and weight column block
/// once. The blocked-reuse path multicasts the weight column block down the used
/// row blocks, but the activation row block stays unicast, so it is still read
/// once per used column block.
struct EmissionEstimate {
  std::string mapping;
  int64_t coresUsed = 0;
  int64_t maxTilesPerCore = 0;
  int64_t dramReadBytes = 0;
  int64_t dramWriteBytes = 0;
  int64_t fusedStageTiles = 0;
  bool fusedFeasible = false;
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

/// Placeholders for the HTML visualizer sections. Kept as opaque strings so the
/// schema is stable before the visualizer fills the detail.
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
  DeviceInfo device;
  std::vector<TensorDesc> tensors;
  Layout layout;
  ShardGrid shardGrid;
  std::vector<CBSpec> cbSpecs;
  NocEstimate noc;
  EmissionEstimate emission;
  PerfEstimate perf;
  VizSections viz;
};

/// Serialize a plan to pretty-printed JSON (indent 2, trailing newline). The
/// output parses with llvm::json::parse and with python json.load.
std::string toJson(const Plan &plan);

/// Parse a plan field-for-field from a JSON value. Returns std::nullopt if the
/// value is not an object or any required field is missing or wrongly typed.
/// A malformed plan, including one without a device block or emission block,
/// is rejected rather than defaulted.
std::optional<Plan> parsePlan(const llvm::json::Value &value);

} // namespace tensixflow
} // namespace mlir

#endif // TENSIXFLOW_PLANSCHEMA_H
