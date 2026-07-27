//===- CBAllocator.h --------------------------------------------*- C++ -*-===//
//
// Circular-buffer allocator (todo 28): assigns CB indices + a feasible depth
// (from the shared per-core L1 budget, L1Model.h) to the reader/compute/writer
// circular buffers of a fused op. Indices follow the tt-metal convention
// (tt::CBIndex runs c_0..c_31): the default 3-CB pipeline maps in0 = c_0,
// in1 = c_1, out = c_16; any other CB count gets sequential indices 0..numCBs-1.
// The depth is the max feasible depth shared across all CBs (GridPlanner's
// maxCBDepth), so the total CB footprint never exceeds kCBShardBudget — an
// oversized request is answered with a reduced depth (or flagged infeasible),
// never an overflow, and no index past c_31 is ever emitted.
//
// Pure data + math over the scheduler model (Model.h) — NO MLIR dependency. The
// L1-budget constants are IMPORTED from L1Model.h (single source of truth) and
// the CB index range (kMaxCBIndex / isValidCBIndex) + plan::CBSpec are REUSED
// from Model.h / PlanSchema.h — nothing is redefined here. The report seam is the
// allocation's std::vector<plan::CBSpec> (already plan::CBSpec), suitable for
// PlanDraft.cbSpecs (todo 30).
//
//===----------------------------------------------------------------------===//

#ifndef TENSIXFLOW_SCHEDULER_CBALLOCATOR_H
#define TENSIXFLOW_SCHEDULER_CBALLOCATOR_H

#include "GridPlanner.h"
#include "L1Model.h"
#include "Model.h"

#include <cstdint>
#include <vector>

namespace mlir {
namespace tensixflow {
namespace scheduler {

//===----------------------------------------------------------------------===//
// CB index convention (tt-metal tt::CBIndex c_0..c_31)
//===----------------------------------------------------------------------===//

/// The reader/compute/writer convention for the default 3-CB fused-op pipeline:
/// the two reader inputs are c_0 / c_1 and the writer output is c_16.
constexpr int64_t kCBIn0 = 0;  // c_0: reader input 0 (e.g. activation)
constexpr int64_t kCBIn1 = 1;  // c_1: reader input 1 (e.g. weight)
constexpr int64_t kCBOut = 16; // c_16: writer output

//===----------------------------------------------------------------------===//
// Allocation result
//===----------------------------------------------------------------------===//

/// The allocator's decision: one plan::CBSpec per circular buffer (index, depth
/// in tiles, per-tile bytes), the total CB footprint in bytes, and whether the
/// allocation is feasible — a positive depth that fits the per-core CB budget
/// with every index in the valid c_0..c_31 range.
struct CBAllocation {
  std::vector<plan::CBSpec> cbs;
  int64_t totalBytes = 0;
  bool feasible = false;
};

/// Allocate `numCBs` circular buffers from the shared per-core CB budget.
///   - Indices: the default numCBs == 3 follows the tt-metal reader/compute/
///     writer convention (in0 = kCBIn0, in1 = kCBIn1, out = kCBOut); any other
///     count gets sequential indices 0..numCBs-1. Every emitted index satisfies
///     isValidCBIndex; a request that would run past c_31 is flagged infeasible
///     and emits no index > 31.
///   - Depth: maxCBDepth(numCBs) = floor(kCBShardBudget / (numCBs * kTileBytes)),
///     shared across all CBs, so totalBytes = numCBs * depth * kTileBytes never
///     exceeds kCBShardBudget. The budget math is dtype-agnostic (kTileBytes is
///     the canonical bf16 tile constant), so `bytesPerElem` exists for signature
///     parity with the other planners and is unused here.
///   - Feasibility: depth > 0 && totalBytes <= kCBShardBudget && all indices
///     valid. A degenerate request (numCBs <= 0) is guarded: feasible == false,
///     empty allocation, no crash, no division by zero.
CBAllocation allocateCBs(int64_t numCBs, int bytesPerElem);

/// The report seam (todo 30): the allocation's CB specs, already plan::CBSpec,
/// ready to drop into PlanDraft.cbSpecs.
std::vector<plan::CBSpec> toCBSpecs(const CBAllocation &allocation);

} // namespace scheduler
} // namespace tensixflow
} // namespace mlir

#endif // TENSIXFLOW_SCHEDULER_CBALLOCATOR_H
