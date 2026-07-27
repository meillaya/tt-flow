//===- Model.h --------------------------------------------------*- C++ -*-===//
//
// TensixFlow scheduler in-memory data model (todo 25). Pure data + helpers with
// NO MLIR dependency, so the planners (todos 26-29) and the CB allocator can
// link the small MLIRTensixflowScheduler lib without dragging in the dialect.
//
// The plan/perf/viz serialization structs (ShardGrid, CBSpec, Layout, Plan, ...)
// are OWNED by PlanSchema.h (D15) and REUSED here by reference — this model does
// NOT redefine them. The toSchema* / toPlan helpers below are the seam that lets
// todo 30 build a PlanSchema::Plan from scheduler decisions for the report.
//
// Device constants mirror tt-metal: tt::CBIndex is c_0..c_31 (32 CBs), a tile is
// 32x32, and bf16 is 2 bytes -> 2048 bytes/tile.
//
//===----------------------------------------------------------------------===//

#ifndef TENSIXFLOW_SCHEDULER_MODEL_H
#define TENSIXFLOW_SCHEDULER_MODEL_H

#include "tensixflow/PlanSchema.h"

#include <cstdint>
#include <string>
#include <vector>

namespace mlir {
namespace tensixflow {
namespace scheduler {

/// The plan/perf/viz serialization structs live in the enclosing
/// mlir::tensixflow namespace (PlanSchema.h). Alias them so the reuse is
/// explicit at every call site without redefining anything.
namespace plan = ::mlir::tensixflow;

//===----------------------------------------------------------------------===//
// Element dtype sizing
//===----------------------------------------------------------------------===//

/// Bytes per element for a dtype name: bf16/fp16 = 2, fp32 = 4. Returns 0 for
/// an unrecognized dtype (callers validate before use).
int bytesPerElem(const std::string &dtype);

//===----------------------------------------------------------------------===//
// Tensor + tile shapes
//===----------------------------------------------------------------------===//

/// A logical tensor the scheduler reasons about: name, static dims, element
/// dtype. Distinct from plan::TensorDesc (which adds a memory placement string
/// for the report) — this is the pre-placement shape the planners start from.
struct TensorShape {
  std::string name;
  std::vector<int64_t> dims;
  std::string dtype;

  /// Product of dims (1 for an empty/rank-0 shape).
  int64_t numElements() const;

  /// numElements() * bytesPerElem.
  int64_t bytes(int bytesPerElem) const;
};

/// A hardware tile. Defaults to the Tenstorrent 32x32 tile; bf16 -> 2048 bytes.
struct TileShape {
  int64_t height = 32;
  int64_t width = 32;

  /// height * width * bytesPerElem (32x32 bf16 = 2048).
  int64_t bytes(int bytesPerElem) const { return height * width * bytesPerElem; }
};

//===----------------------------------------------------------------------===//
// Core grid (reuses plan::ShardGrid)
//===----------------------------------------------------------------------===//

/// Total cores in a shard grid (gridY * gridX). Free function over the reused
/// plan::ShardGrid so the struct itself is not redefined.
inline int64_t totalCores(const plan::ShardGrid &grid) {
  return grid.gridY * grid.gridX;
}

//===----------------------------------------------------------------------===//
// Placement + weight strategy
//===----------------------------------------------------------------------===//

/// Where a tensor lives. Mirrors the dialect's MemorySpace but is plain data.
enum class Placement { DRAM, L1 };

/// "DRAM" / "L1".
std::string toString(Placement placement);

/// How weights are distributed across the grid.
enum class WeightStrategy { Replicate, Shard, Multicast };

/// "replicate" / "shard" / "multicast" (matches plan::Layout::weightStrategy).
std::string toString(WeightStrategy strategy);

//===----------------------------------------------------------------------===//
// Pipeline schedule
//===----------------------------------------------------------------------===//

/// A reader/compute/writer pipeline stage.
enum class PipelineStage { Reader, Compute, Writer };

/// "reader" / "compute" / "writer".
std::string toString(PipelineStage stage);

/// A lightweight description of the kernel pipeline: the ordered stages a fused
/// op runs (typically {Reader, Compute, Writer}).
struct PipelineSchedule {
  std::vector<PipelineStage> stages;
};

//===----------------------------------------------------------------------===//
// Circular-buffer index range (tt::CBIndex c_0..c_31)
//===----------------------------------------------------------------------===//

/// Highest valid CB index (tt::CBIndex runs c_0..c_31 -> 32 CBs).
constexpr int64_t kMaxCBIndex = 31;

/// True iff index is a valid CB index (0..kMaxCBIndex inclusive).
inline bool isValidCBIndex(int64_t index) {
  return index >= 0 && index <= kMaxCBIndex;
}

//===----------------------------------------------------------------------===//
// Conversion to the PlanSchema (report) structs — the todo-30 seam
//===----------------------------------------------------------------------===//

/// Build a plan::CBSpec from a CB index, depth (in tiles), and the tile's byte
/// size (tile.bytes(bytesPerElem)).
plan::CBSpec toSchemaCBSpec(int64_t index, int64_t depth, const TileShape &tile,
                            int bytesPerElem);

/// Build a plan::Layout from a weight strategy + activation sharding label.
plan::Layout toSchemaLayout(WeightStrategy strategy,
                            const std::string &activationSharding);

/// Everything needed to assemble a report Plan from scheduler decisions. noc /
/// perf / viz are left at their defaults for the planners (todos 26-29, 42) to
/// fill; toPlan only maps the fields the scheduler model owns.
struct PlanDraft {
  std::string graphId;
  std::vector<plan::TensorDesc> tensors;
  WeightStrategy weightStrategy = WeightStrategy::Replicate;
  std::string activationSharding;
  plan::ShardGrid grid;
  std::vector<plan::CBSpec> cbSpecs;
};

/// Assemble a plan::Plan from a PlanDraft (the single-arg conversion seam).
plan::Plan toPlan(const PlanDraft &draft);

} // namespace scheduler
} // namespace tensixflow
} // namespace mlir

#endif // TENSIXFLOW_SCHEDULER_MODEL_H
