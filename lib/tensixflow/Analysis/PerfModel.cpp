//===- PerfModel.cpp --------------------------------------------*- C++ -*-===//
//
// Implementation of the TensixFlow static performance model (todo 42). See
// PerfModel.h — a first-order static roofline-style estimate (NOT measured
// hardware). Reuses estimateNoC (todo 29) for the unfused-vs-fused traffic and
// imports the L1-budget constants from L1Model.h (todo 26 / D17); NO MLIR dep.
//
//===----------------------------------------------------------------------===//

#include "PerfModel.h"

#include "GridPlanner.h"
#include "L1Model.h"
#include "Model.h"
#include "NoCEstimator.h"
#include "WeightPlanner.h"

#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <string>

namespace mlir {
namespace tensixflow {
namespace analysis {

namespace {

// The rmsnorm+matmul graph runs as TWO unfused kernel launches (rmsnorm, matmul)
// or ONE fused launch (the user's "kernel launches" report field).
constexpr int64_t kUnfusedLaunches = 2;
constexpr int64_t kFusedLaunches = 1;

/// The memory-vs-compute-bound classification. Stringified into the report's
/// bottleneck field (one of "memory"/"compute"/"balanced" — the schema enum).
enum class Bottleneck { Memory, Compute, Balanced };

/// Exhaustive stringify (no default -> a new variant is a compile-time warning).
std::string toString(Bottleneck bottleneck) {
  switch (bottleneck) {
  case Bottleneck::Memory:
    return "memory";
  case Bottleneck::Compute:
    return "compute";
  case Bottleneck::Balanced:
    return "balanced";
  }
  return "balanced";
}

/// A bound-derived optimization suggestion. Exhaustive, so this alone guarantees
/// the suggestion list is non-empty for any plan with matmul work.
std::string boundSuggestion(Bottleneck bottleneck) {
  switch (bottleneck) {
  case Bottleneck::Compute:
    return "compute-bound: spread N across more X-cores to raise parallelism";
  case Bottleneck::Memory:
    return "memory-bound: deepen CBs / prefetch activations to hide DRAM "
           "latency";
  case Bottleneck::Balanced:
    return "balanced: pipeline reader/compute/writer stages to overlap DRAM "
           "and compute";
  }
  return "balanced: pipeline reader/compute/writer stages to overlap DRAM and "
         "compute";
}

/// Product of all dims but the last (the matmul M = activation leading dims).
int64_t leadingProduct(const scheduler::TensorShape &tensor) {
  int64_t product = 1;
  for (size_t i = 0; i + 1 < tensor.dims.size(); ++i)
    product *= tensor.dims[i];
  return product;
}

/// The matmul dimensions (M, K, N) derived from the activation [.., M, K] +
/// weight [K, N]. `valid` is false when the shapes are too rank-poor /
/// non-positive to form a matmul (the degenerate guard).
struct MatmulDims {
  int64_t m = 0;
  int64_t k = 0;
  int64_t n = 0;
  bool valid = false;
};

MatmulDims matmulDims(const scheduler::TensorShape &activation,
                      const scheduler::TensorShape &weight) {
  MatmulDims dims;
  if (activation.dims.empty() || weight.dims.empty())
    return dims;
  dims.m = leadingProduct(activation);
  dims.k = activation.dims.back();
  dims.n = weight.dims.back();
  dims.valid = dims.m > 0 && dims.k > 0 && dims.n > 0;
  return dims;
}

/// Convert a plan::TensorDesc to the scheduler's TensorShape.
scheduler::TensorShape toTensorShape(const plan::TensorDesc &desc) {
  return scheduler::TensorShape{desc.name, desc.shape, desc.dtype};
}

/// A valid-but-degenerate report: zero cycles/traffic, "balanced", one concrete
/// shape-check suggestion. Keeps bottleneck + suggestions non-empty + finite even
/// for a malformed plan (no crash, no NaN).
PerfReport degenerateReport() {
  PerfReport report;
  report.bottleneck = toString(Bottleneck::Balanced);
  report.suggestions = {
      "no matmul work detected; verify activation/weight tensor shapes"};
  return report;
}

} // namespace

PerfReport estimatePerf(const plan::Plan &plan) {
  // Degenerate guard: a plan needs an activation + a weight to model a matmul.
  if (plan.tensors.size() < 2)
    return degenerateReport();

  const scheduler::TensorShape activation = toTensorShape(plan.tensors[0]);
  const scheduler::TensorShape weight = toTensorShape(plan.tensors[1]);
  int bpe = scheduler::bytesPerElem(activation.dtype);
  if (bpe <= 0)
    bpe = 2; // default to bf16 for an unrecognized dtype

  const int64_t cores = scheduler::totalCores(plan.shardGrid);
  const MatmulDims dims = matmulDims(activation, weight);
  if (!dims.valid || cores <= 0)
    return degenerateReport();

  PerfReport report;

  // Traffic: recompute the weight plan (todo 27) and run the NoC estimator
  // (todo 29) for BOTH paths. The fused path keeps the intermediate in L1, so
  // unfused DRAM > fused DRAM (the invariant this model reproduces).
  const scheduler::WeightPlan weightPlan =
      scheduler::planWeight(weight, plan.shardGrid, bpe);
  const scheduler::NocTraffic unfused = scheduler::estimateNoC(
      scheduler::NocInputs{activation, weight, weightPlan, false}, bpe);
  const scheduler::NocTraffic fused = scheduler::estimateNoC(
      scheduler::NocInputs{activation, weight, weightPlan, true}, bpe);
  report.unfused = unfused.estimate;
  report.fused = fused.estimate;
  report.savedDramReadBytes =
      unfused.estimate.dramReadBytes - fused.estimate.dramReadBytes;
  report.savedDramWriteBytes =
      unfused.estimate.dramWriteBytes - fused.estimate.dramWriteBytes;
  report.unfusedKernelLaunches = kUnfusedLaunches;
  report.fusedKernelLaunches = kFusedLaunches;

  // Compute cycles: the matmul dominates (2*M*N*K FLOPs); the grid is parallel,
  // so per-core work = total / cores.
  const double computeFlops =
      2.0 * double(dims.m) * double(dims.n) * double(dims.k);
  const double perCoreFlops = computeFlops / double(cores);
  report.computeCycles = int64_t(perCoreFlops / kFlopsPerCyclePerCore);

  // Memory cycles: off-chip DRAM + NoC multicast on the fused (actual) path, per
  // core. L1 traffic is on-chip and not on the critical path.
  const int64_t fusedTraffic = fused.estimate.dramReadBytes +
                               fused.estimate.dramWriteBytes +
                               fused.estimate.multicastBytes;
  const double perCoreBytes = double(fusedTraffic) / double(cores);
  report.memoryCycles = int64_t(perCoreBytes / kBytesPerCyclePerCore);

  // Roofline: the pipelined reader/compute/writer overlaps the stages, so the
  // slower dominates.
  report.estCycles = std::max(report.computeCycles, report.memoryCycles);
  report.computeUtil = report.estCycles > 0
                           ? double(report.computeCycles) / double(report.estCycles)
                           : 0.0;

  // CB occupancy vs the imported per-core CB budget (L1Model.h).
  for (const plan::CBSpec &cb : plan.cbSpecs)
    report.cbBytes += cb.depth * cb.tileBytes;
  report.cbOccupancy = double(report.cbBytes) / double(scheduler::kCBShardBudget);

  // Bound classification from the compute/memory cycle ratio.
  Bottleneck bottleneck;
  if (report.memoryCycles > 0) {
    const double ratio =
        double(report.computeCycles) / double(report.memoryCycles);
    if (ratio > kComputeBoundRatio)
      bottleneck = Bottleneck::Compute;
    else if (ratio < kMemoryBoundRatio)
      bottleneck = Bottleneck::Memory;
    else
      bottleneck = Bottleneck::Balanced;
  } else {
    bottleneck = report.computeCycles > 0 ? Bottleneck::Compute
                                          : Bottleneck::Balanced;
  }
  report.bottleneck = toString(bottleneck);

  // Optimization suggestions, each derived from the plan's actual parameters.
  // CB depth: suggest deepening when the plan is below the feasible max.
  const int64_t numCBs = int64_t(plan.cbSpecs.size());
  if (numCBs > 0) {
    int64_t minDepth = plan.cbSpecs[0].depth;
    for (const plan::CBSpec &cb : plan.cbSpecs)
      minDepth = std::min(minDepth, cb.depth);
    const int64_t maxDepth = scheduler::maxCBDepth(numCBs);
    if (minDepth > 0 && minDepth < maxDepth) {
      const int64_t target = std::min(2 * minDepth, maxDepth);
      report.suggestions.push_back("increase CB depth " +
                                   std::to_string(minDepth) + "->" +
                                   std::to_string(target) +
                                   " to deepen the pipeline");
    }
  }

  // Multicast: a large replicated weight should move to column multicast.
  const int64_t weightBytes = weight.bytes(bpe);
  if (plan.layout.weightStrategy != "multicast" &&
      weightBytes > scheduler::kCBShardBudget / 2)
    report.suggestions.push_back(
        "use column multicast for W tiles to free per-core L1");

  // Split-K: a large contracting dimension should be parallelized.
  if (dims.k >= kSplitKThreshold && cores >= 2)
    report.suggestions.push_back(
        "split K across 2 core groups to parallelize the reduction");

  // Bound-derived suggestion (guarantees the list is non-empty).
  report.suggestions.push_back(boundSuggestion(bottleneck));

  return report;
}

plan::PerfEstimate toPerfEstimate(const PerfReport &report) {
  plan::PerfEstimate perf;
  perf.estCycles = report.estCycles;
  perf.computeUtil = report.computeUtil;
  perf.bottleneck = report.bottleneck;
  perf.suggestions = report.suggestions;
  return perf;
}

std::string formatPerfReport(const PerfReport &report) {
  std::string out;
  llvm::raw_string_ostream os(out);
  os << "Performance report\n"
     << "  unfused DRAM reads: " << report.unfused.dramReadBytes << " bytes\n"
     << "  fused DRAM reads: " << report.fused.dramReadBytes << " bytes\n"
     << "  unfused DRAM writes: " << report.unfused.dramWriteBytes << " bytes\n"
     << "  fused DRAM writes: " << report.fused.dramWriteBytes << " bytes\n"
     << "  kernel launches: unfused " << report.unfusedKernelLaunches
     << " -> fused " << report.fusedKernelLaunches << "\n"
     << "  predicted bottleneck: " << report.bottleneck << "\n";
  for (const std::string &suggestion : report.suggestions)
    os << "  suggested optimization: " << suggestion << "\n";
  return out;
}

} // namespace analysis
} // namespace tensixflow
} // namespace mlir
