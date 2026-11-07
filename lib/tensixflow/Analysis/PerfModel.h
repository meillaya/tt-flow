//===- PerfModel.h ----------------------------------------------*- C++ -*-===//
//
// Static performance model: a first-order analytic roofline-style projection of
// the cycles, DRAM and NoC traffic, CB occupancy, compute utilization, binding
// constraint, and concrete optimization suggestions for a plan. Every figure is
// a projected static estimate computed from the plan's tensor shapes, the
// scheduler's NoC traffic estimate, and the device values in
// spec/devices.json. There is no device in the loop and no measured number in
// this model.
//
// Cycle model (all three terms are per-plan and projected):
//   - computeCycles = 2*M*N*K / (cores * flopsPerCyclePerCore), where cores is
//     the product's Tensix count from the generated device table (chips times
//     the per-die compute grid).
//   - dramCycles = (fused DRAM read + write bytes) / bytesPerDramCycle, where
//     bytesPerDramCycle = dram bandwidth in bytes per second / clockHz.
//   - nocCycles = max(per-core read bytes, per-core write bytes) /
//     nocBytesPerCyclePerDirection. Read bytes are the fused DRAM reads plus
//     the multicast bytes; write bytes are the fused DRAM writes. The divisor
//     is one NoC's per-direction L1 rate, the conservative choice even though
//     the devices have two NoCs.
//   - estCycles = max of the three terms; the reader/compute/writer pipeline
//     overlaps the stages, so the slowest term bounds the projection.
//   - computeUtil = computeCycles / estCycles.
//   - bound = the binding term ("compute", "dram", or "noc"), or "balanced"
//     when the second largest term is within kBalancedBandRatio of the largest.
//
// The fusion invariant is preserved: the fused path keeps the rmsnorm-output
// intermediate in L1, so fused DRAM traffic is lower than unfused traffic.
//
// Pure data and math over the scheduler model with no MLIR dependency. The
// report seam is toPerfEstimate, which fills plan::Plan.perf.
//
//===----------------------------------------------------------------------===//

#ifndef TENSIXFLOW_ANALYSIS_PERFMODEL_H
#define TENSIXFLOW_ANALYSIS_PERFMODEL_H

#include "tensixflow/Generated/DeviceSpecs.h"
#include "tensixflow/PlanSchema.h"

#include <cstdint>
#include <string>
#include <vector>

namespace mlir {
namespace tensixflow {
namespace analysis {

/// The plan/perf/viz serialization structs live in the enclosing
/// mlir::tensixflow namespace (PlanSchema.h). Alias them so the reuse is
/// explicit without redefining anything.
namespace plan = ::mlir::tensixflow;

//===----------------------------------------------------------------------===//
// Projection constants (modeling choices, not hardware measurements)
//===----------------------------------------------------------------------===//

/// Two leading terms within this ratio count as balanced instead of naming one
/// binding constraint. A modeling band, not a hardware fact.
constexpr double kBalancedBandRatio = 0.9;

/// Split-K suggestion threshold on the contracting dimension K.
constexpr int64_t kSplitKThreshold = 2048;

//===----------------------------------------------------------------------===//
// Projected performance report
//===----------------------------------------------------------------------===//

/// The projected performance report for a plan: the unfused-vs-fused DRAM/L1/
/// NoC traffic (from the NoC estimator), kernel launch counts, the three cycle
/// terms, CB occupancy, the binding constraint, and derived optimization
/// suggestions. Every field is a static estimate, never measured.
struct PerfReport {
  // Device the projection comes from. Copied so rendered text always names it.
  std::string deviceName;
  std::string deviceArch;
  int64_t cores = 0;

  // Traffic in bytes, unfused vs fused (reused plan::NocEstimate figures).
  plan::NocEstimate unfused;
  plan::NocEstimate fused;
  int64_t savedDramReadBytes = 0;
  int64_t savedDramWriteBytes = 0;

  // Kernel launches: the unfused op pair vs the fused op.
  int64_t unfusedKernelLaunches = 0;
  int64_t fusedKernelLaunches = 0;

  // Projected cycle terms and the roofline maximum.
  int64_t computeCycles = 0;
  int64_t dramCycles = 0;
  int64_t nocCycles = 0;
  int64_t estCycles = 0;
  double computeUtil = 0.0;

  // CB occupancy against the per-core CB budget (usable L1 minus the planner
  // reserve from the generated device table).
  int64_t cbBytes = 0;
  double cbOccupancy = 0.0;

  // Binding constraint plus suggestions; both always non-empty for a valid
  // plan: "compute", "dram", "noc", or "balanced".
  std::string bound;
  std::vector<std::string> suggestions;
};

/// Project the static performance of a plan on a device. Reads the plan's
/// tensor shapes (tensors[0] is the activation, tensors[1] the weight), the
/// shard grid, and the CB specs; recomputes the weight plan for the device and
/// runs the NoC estimator for both the unfused and fused paths; then applies the
/// documented cycle model. A degenerate plan (fewer than two tensors, no matmul
/// work, or an empty grid) yields a valid report with zero cycles, bound
/// "balanced", and one shape-check suggestion; never a crash, NaN, or empty
/// bound list.
PerfReport estimatePerf(const plan::Plan &plan,
                        const device::DeviceSpec &deviceSpec);

/// The report seam: fills plan::Plan.perf (est_cycles, compute_util,
/// bottleneck, suggestions).
plan::PerfEstimate toPerfEstimate(const PerfReport &report);

/// Human-readable projection. Every rendered line names the device and is
/// labelled projected.
std::string formatPerfReport(const PerfReport &report);

} // namespace analysis
} // namespace tensixflow
} // namespace mlir

#endif // TENSIXFLOW_ANALYSIS_PERFMODEL_H
