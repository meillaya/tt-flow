//===- PerfModel.h ----------------------------------------------*- C++ -*-===//
//
// Static performance model (todo 42): a FIRST-ORDER STATIC ANALYTIC roofline-style
// model of the cycles, DRAM/L1/NoC traffic, CB occupancy, compute utilization,
// memory-vs-compute-bound classification, and concrete optimization suggestions
// for a plan. Every figure is a STATIC ESTIMATE computed from the plan's tensor
// shapes + the shared L1-budget constants (L1Model.h, todo 26 / D17) + the NoC
// traffic estimator (NoCEstimator.h, todo 29) — these are NOT measured hardware
// numbers and must never be reported as such (per the plan's Must-NOT). There is
// no device here; the model is purely analytical.
//
// Cycle model (the modeled Wormhole n150, bf16; documented estimates):
//   - Compute cycles: the matmul dominates compute (rmsnorm/gelu are O(M*K)
//     elementwise, negligible next to the O(M*N*K) matmul). computeFlops =
//     2*M*N*K; the grid runs in parallel, so per-core work = computeFlops /
//     totalCores; computeCycles = perCoreFlops / kFlopsPerCyclePerCore.
//   - Memory cycles: the bandwidth-bound path is off-chip DRAM + NoC multicast
//     (L1 traffic is on-chip and not on the critical path). perCoreTraffic =
//     (fused DRAM read + write + multicast) / totalCores; memoryCycles =
//     perCoreTraffic / kBytesPerCyclePerCore.
//   - estCycles = max(computeCycles, memoryCycles): the reader/compute/writer
//     pipeline overlaps the two stages, so the slower dominates (roofline).
//   - computeUtil = computeCycles / estCycles (fraction of runtime the compute
//     units are busy; 1.0 when compute-bound, < 1.0 when memory-bound).
//   - bottleneck: ratio = computeCycles / memoryCycles; ratio > kComputeBoundRatio
//     -> "compute", ratio < kMemoryBoundRatio -> "memory", else "balanced".
//
// The central invariant the model reproduces: fusion REDUCES DRAM traffic
// (unfused.dramReadBytes > fused.dramReadBytes) because the fused path keeps the
// rmsnorm-output intermediate in L1 instead of round-tripping DRAM (the same
// insight as todo 29/30). The unfused-vs-fused figures come straight from
// estimateNoC run for both paths; nothing is redefined here.
//
// Pure data + math over the scheduler model — NO MLIR dependency. The L1-budget
// constants are IMPORTED from L1Model.h (single source of truth) and the traffic
// figures REUSE plan::NocEstimate + estimateNoC. The report seam is toPerfEstimate
// (fills plan::Plan.perf: est_cycles, compute_util, bottleneck, suggestions).
//
//===----------------------------------------------------------------------===//

#ifndef TENSIXFLOW_ANALYSIS_PERFMODEL_H
#define TENSIXFLOW_ANALYSIS_PERFMODEL_H

#include "tensixflow/PlanSchema.h"

#include <cstdint>
#include <string>
#include <vector>

namespace mlir {
namespace tensixflow {
namespace analysis {

/// The plan/perf/viz serialization structs live in the enclosing
/// mlir::tensixflow namespace (PlanSchema.h). Alias them so the reuse is explicit
/// at every call site without redefining anything (mirrors the scheduler model).
namespace plan = ::mlir::tensixflow;

//===----------------------------------------------------------------------===//
// Modeled static cost constants (NOT measured hardware — documented estimates)
//===----------------------------------------------------------------------===//

/// One 32x32 tile outer-product accumulate per cycle per core = 32*32 MACs =
/// 32*32*2 = 2048 FLOPs/cycle/core.
constexpr double kFlopsPerCyclePerCore = 2048.0;
/// Modeled aggregate off-chip (DRAM + NoC multicast) bandwidth per core, bytes per
/// cycle. Conservative documented value (tunable; not a measured hardware rate).
constexpr double kBytesPerCyclePerCore = 16.0;
/// Bound-classification band on the compute/memory cycle ratio: above
/// kComputeBoundRatio -> compute-bound, below kMemoryBoundRatio -> memory-bound,
/// in between -> balanced.
constexpr double kComputeBoundRatio = 1.25;
constexpr double kMemoryBoundRatio = 0.8;
/// Split-K suggestion threshold on the contracting dimension K.
constexpr int64_t kSplitKThreshold = 2048;

//===----------------------------------------------------------------------===//
// Perf report
//===----------------------------------------------------------------------===//

/// The static performance report for a plan: the unfused-vs-fused DRAM/L1/NoC
/// traffic (from estimateNoC, todo 29), kernel launches, the cycle model, CB
/// occupancy, the bound classification, and derived optimization suggestions.
/// Every field is a static estimate, never a measured hardware number.
struct PerfReport {
  // Traffic (bytes), unfused vs fused — reused plan::NocEstimate figures (each
  // carries dramReadBytes, dramWriteBytes, l1Bytes, multicastBytes).
  plan::NocEstimate unfused;
  plan::NocEstimate fused;
  int64_t savedDramReadBytes = 0;  // unfused - fused DRAM reads (the intermediate)
  int64_t savedDramWriteBytes = 0; // unfused - fused DRAM writes

  // Kernel launches (user's report field): the unfused op pair vs the fused op.
  int64_t unfusedKernelLaunches = 0;
  int64_t fusedKernelLaunches = 0;

  // Cycle model (static roofline).
  int64_t computeCycles = 0;
  int64_t memoryCycles = 0;
  int64_t estCycles = 0;    // max(computeCycles, memoryCycles)
  double computeUtil = 0.0; // computeCycles / estCycles, in [0,1]

  // CB occupancy vs the per-core CB budget (kCBShardBudget, L1Model.h).
  int64_t cbBytes = 0;
  double cbOccupancy = 0.0; // cbBytes / kCBShardBudget

  // Classification + suggestions (both always non-empty for a valid plan).
  std::string bottleneck; // "memory" / "compute" / "balanced"
  std::vector<std::string> suggestions;
};

/// Estimate the static performance of a plan. Reads the plan's tensor shapes
/// (tensors[0] = activation, tensors[1] = weight — the PlanReport convention),
/// shard grid, and CB specs; recomputes the weight plan (todo 27) and runs the
/// NoC estimator (todo 29) for BOTH the unfused and fused paths; then applies the
/// documented cycle model. The L1-budget constants are imported from L1Model.h.
/// A degenerate plan (fewer than 2 tensors, no matmul work, or an empty grid)
/// yields a valid report: zero cycles/traffic, bottleneck "balanced", and a single
/// shape-check suggestion — never a crash, NaN, or empty bottleneck/suggestions.
PerfReport estimatePerf(const plan::Plan &plan);

/// The report seam (fills plan::Plan.perf, todo 24 schema): est_cycles,
/// compute_util, bottleneck, suggestions.
plan::PerfEstimate toPerfEstimate(const PerfReport &report);

/// The human-readable performance report (the user-spec shape: unfused vs fused
/// DRAM reads/writes, kernel launches, predicted bottleneck, suggested
/// optimization).
std::string formatPerfReport(const PerfReport &report);

} // namespace analysis
} // namespace tensixflow
} // namespace mlir

#endif // TENSIXFLOW_ANALYSIS_PERFMODEL_H
