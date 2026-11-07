//===- PerfModel.cpp --------------------------------------------*- C++ -*-===//
//
// Implementation of the TensixFlow static performance model. See PerfModel.h:
// a first-order roofline projection, never a measured hardware number. Reuses
// the NoC estimator for unfused-vs-fused traffic and the generated device table
// for the projection constants.
//
//===----------------------------------------------------------------------===//

#include "PerfModel.h"

#include "GridPlanner.h"
#include "Model.h"
#include "NoCEstimator.h"
#include "WeightPlanner.h"

#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <string>

namespace mlir {
namespace tensixflow {
namespace analysis {

namespace {

// The rmsnorm+matmul graph runs as two unfused kernel launches or one fused
// launch; those counts feed the kernel-launch report field.
constexpr int64_t kUnfusedLaunches = 2;
constexpr int64_t kFusedLaunches = 1;

/// Element width used when a payload dtype is unrecognized. bf16 is the project
/// working dtype.
constexpr int kDefaultBytesPerElem = 2;

/// The binding constraint of the projected roofline.
enum class Bound { Compute, Dram, Noc, Balanced };

/// Exhaustive stringify (no default, so adding a variant is a compile-time
/// warning).
std::string toString(Bound bound) {
  switch (bound) {
  case Bound::Compute:
    return "compute";
  case Bound::Dram:
    return "dram";
  case Bound::Noc:
    return "noc";
  case Bound::Balanced:
    return "balanced";
  }
  return "balanced";
}

/// A bound-derived optimization suggestion, one per binding constraint.
std::string boundSuggestion(Bound bound) {
  switch (bound) {
  case Bound::Compute:
    return "compute-bound: spread N across more X-cores to raise parallelism";
  case Bound::Dram:
    return "dram-bound: deepen CBs or prefetch activations to hide DRAM "
           "latency";
  case Bound::Noc:
    return "noc-bound: multicast weight tiles instead of re-reading them per "
           "core";
  case Bound::Balanced:
    return "balanced: pipeline reader/compute/writer stages to overlap DRAM "
           "and compute";
  }
  return "balanced: pipeline reader/compute/writer stages to overlap DRAM and "
         "compute";
}

/// Product of all dims but the last (the matmul M = activation leading dims).
int64_t leadingProduct(const scheduler::TensorShape &tensor) {
  int64_t product = 1;
  for (size_t i = 0; i + 1 < tensor.dims.size(); ++i) {
    product *= tensor.dims[i];
  }
  return product;
}

/// The matmul dimensions (M, K, N) derived from the activation [.., M, K] and
/// the weight [K, N]. `valid` is false when the shapes are too rank-poor or
/// non-positive to form a matmul.
struct MatmulDims {
  int64_t m = 0;
  int64_t k = 0;
  int64_t n = 0;
  bool valid = false;
};

MatmulDims matmulDims(const scheduler::TensorShape &activation,
                      const scheduler::TensorShape &weight) {
  MatmulDims dims;
  if (activation.dims.empty() || weight.dims.empty()) {
    return dims;
  }
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

/// A valid but degenerate report: zero cycles and traffic, bound "balanced",
/// one concrete shape-check suggestion. Keeps every rendered field finite.
PerfReport degenerateReport(const device::DeviceSpec &deviceSpec) {
  PerfReport report;
  report.deviceName = deviceSpec.name;
  report.deviceArch = deviceSpec.arch;
  report.cores = deviceSpec.computeGrid.totalCores();
  report.bound = toString(Bound::Balanced);
  report.suggestions = {
      "no matmul work detected; verify activation/weight tensor shapes"};
  return report;
}

/// The binding term for three projected cycle counts. Returns Balanced when the
/// second largest term is within kBalancedBandRatio of the largest.
Bound classifyBound(int64_t computeCycles, int64_t dramCycles,
                    int64_t nocCycles) {
  const int64_t largest = std::max({computeCycles, dramCycles, nocCycles});
  if (largest <= 0) {
    return Bound::Balanced;
  }

  const int64_t smallest = std::min({computeCycles, dramCycles, nocCycles});
  const int64_t second = computeCycles + dramCycles + nocCycles - largest -
                         smallest;
  if (double(second) >= kBalancedBandRatio * double(largest)) {
    return Bound::Balanced;
  }
  if (largest == computeCycles) {
    return Bound::Compute;
  }
  if (largest == dramCycles) {
    return Bound::Dram;
  }
  return Bound::Noc;
}

/// Bytes per clock of DRAM bandwidth for one device. The device table stores
/// bytes per second and a clock, so the conversion lives here once.
double dramBytesPerClock(const device::DeviceSpec &deviceSpec) {
  if (deviceSpec.clockHz <= 0) {
    return 0.0;
  }
  return double(deviceSpec.dramBandwidthBytesPerSecond) /
         double(deviceSpec.clockHz);
}

} // namespace

PerfReport estimatePerf(const plan::Plan &plan,
                        const device::DeviceSpec &deviceSpec) {
  // Degenerate guard: a plan needs an activation and a weight to model matmul.
  if (plan.tensors.size() < 2) {
    return degenerateReport(deviceSpec);
  }

  const scheduler::TensorShape activation = toTensorShape(plan.tensors[0]);
  const scheduler::TensorShape weight = toTensorShape(plan.tensors[1]);
  int bytesPerElement = scheduler::bytesPerElem(activation.dtype);
  if (bytesPerElement <= 0) {
    bytesPerElement = kDefaultBytesPerElem;
  }

  const int64_t cores = deviceSpec.computeGrid.totalCores();
  const MatmulDims dims = matmulDims(activation, weight);
  if (!dims.valid || cores <= 0) {
    return degenerateReport(deviceSpec);
  }

  PerfReport report;
  report.deviceName = deviceSpec.name;
  report.deviceArch = deviceSpec.arch;
  report.cores = cores;

  // Traffic: recompute the weight plan for this device and run the NoC
  // estimator for both paths. The fused path keeps the intermediate in L1, so
  // unfused DRAM traffic exceeds fused DRAM traffic.
  const scheduler::WeightPlan weightPlan =
      scheduler::planWeight(weight, deviceSpec, bytesPerElement);
  const scheduler::NocTraffic unfused = scheduler::estimateNoC(
      scheduler::NocInputs{activation, weight, weightPlan, false},
      bytesPerElement);
  const scheduler::NocTraffic fused = scheduler::estimateNoC(
      scheduler::NocInputs{activation, weight, weightPlan, true},
      bytesPerElement);
  report.unfused = unfused.estimate;
  report.fused = fused.estimate;
  report.savedDramReadBytes =
      unfused.estimate.dramReadBytes - fused.estimate.dramReadBytes;
  report.savedDramWriteBytes =
      unfused.estimate.dramWriteBytes - fused.estimate.dramWriteBytes;
  report.unfusedKernelLaunches = kUnfusedLaunches;
  report.fusedKernelLaunches = kFusedLaunches;

  // Compute term: the matmul dominates 2*M*N*K FLOPs, spread over the product's
  // Tensix cores.
  const double computeFlops =
      2.0 * double(dims.m) * double(dims.n) * double(dims.k);
  report.computeCycles =
      int64_t(computeFlops /
              (double(cores) * double(deviceSpec.flopsPerCyclePerCore)));

  // DRAM term: fused off-chip read plus write bytes over bytes per clock.
  const int64_t dramTraffic =
      fused.estimate.dramReadBytes + fused.estimate.dramWriteBytes;
  const double bytesPerClock = dramBytesPerClock(deviceSpec);
  report.dramCycles = bytesPerClock > 0.0
                          ? int64_t(double(dramTraffic) / bytesPerClock)
                          : 0;

  // NoC term: the busiest direction per core at one NoC's L1 rate. Read traffic
  // includes multicast because the weight fans out over the NoC.
  const double perCoreReadBytes =
      double(fused.estimate.dramReadBytes + fused.estimate.multicastBytes) /
      double(cores);
  const double perCoreWriteBytes = double(fused.estimate.dramWriteBytes) /
                                   double(cores);
  const double busiestDirection = std::max(perCoreReadBytes, perCoreWriteBytes);
  report.nocCycles =
      deviceSpec.nocBytesPerCyclePerDirection > 0
          ? int64_t(busiestDirection /
                    double(deviceSpec.nocBytesPerCyclePerDirection))
          : 0;

  // Roofline: the pipelined reader/compute/writer overlaps the stages, so the
  // slowest projected term bounds the estimate.
  report.estCycles =
      std::max({report.computeCycles, report.dramCycles, report.nocCycles});
  report.computeUtil = report.estCycles > 0
                           ? double(report.computeCycles) /
                                 double(report.estCycles)
                           : 0.0;

  // CB occupancy against the per-core budget from the generated device table.
  for (const plan::CBSpec &cb : plan.cbSpecs) {
    report.cbBytes += cb.depth * cb.tileBytes;
  }
  report.cbOccupancy =
      double(report.cbBytes) /
      double(deviceSpec.cbBudget(device::kDefaultL1ReserveBytes));

  const Bound bound =
      classifyBound(report.computeCycles, report.dramCycles, report.nocCycles);
  report.bound = toString(bound);

  // CB depth: suggest deepening when the plan sits below the device's feasible
  // maximum for this CB count.
  const int64_t numCBs = int64_t(plan.cbSpecs.size());
  if (numCBs > 0) {
    int64_t minDepth = plan.cbSpecs[0].depth;
    for (const plan::CBSpec &cb : plan.cbSpecs) {
      minDepth = std::min(minDepth, cb.depth);
    }
    const int64_t maxDepth = scheduler::maxCBDepth(deviceSpec, numCBs);
    if (minDepth > 0 && minDepth < maxDepth) {
      const int64_t target = std::min(2 * minDepth, maxDepth);
      report.suggestions.push_back("increase CB depth " +
                                   std::to_string(minDepth) + "->" +
                                   std::to_string(target) +
                                   " to deepen the pipeline");
    }
  }

  // Multicast: a large replicated weight should move to column multicast.
  const int64_t weightBytes = weight.bytes(bytesPerElement);
  const int64_t cbBudget =
      deviceSpec.cbBudget(device::kDefaultL1ReserveBytes);
  if (plan.layout.weightStrategy != "multicast" &&
      weightBytes > cbBudget / 2) {
    report.suggestions.push_back(
        "use column multicast for W tiles to free per-core L1");
  }

  // Split-K: a large contracting dimension can be parallelized across cores.
  if (dims.k >= kSplitKThreshold && cores >= 2) {
    report.suggestions.push_back(
        "split K across 2 core groups to parallelize the reduction");
  }

  // Bound-derived suggestion: guarantees the list is non-empty.
  report.suggestions.push_back(boundSuggestion(bound));

  return report;
}

plan::PerfEstimate toPerfEstimate(const PerfReport &report) {
  plan::PerfEstimate perf;
  perf.estCycles = report.estCycles;
  perf.computeUtil = report.computeUtil;
  perf.bottleneck = report.bound;
  perf.suggestions = report.suggestions;
  return perf;
}

std::string formatPerfReport(const PerfReport &report) {
  std::string out;
  llvm::raw_string_ostream os(out);
  const std::string prefix = "projected " + report.deviceName + " ";
  os << "projected " << report.deviceName << " roofline estimate ("
     << report.deviceArch << "): static model, not measured hardware\n";
  os << prefix << "cores: " << report.cores << "\n";
  os << prefix << "unfused DRAM reads: " << report.unfused.dramReadBytes
     << " bytes\n";
  os << prefix << "fused DRAM reads: " << report.fused.dramReadBytes
     << " bytes\n";
  os << prefix << "unfused DRAM writes: " << report.unfused.dramWriteBytes
     << " bytes\n";
  os << prefix << "fused DRAM writes: " << report.fused.dramWriteBytes
     << " bytes\n";
  os << prefix << "kernel launches: " << report.unfusedKernelLaunches
     << " unfused -> " << report.fusedKernelLaunches << " fused\n";
  os << prefix << "compute cycles: " << report.computeCycles << "\n";
  os << prefix << "dram cycles: " << report.dramCycles << "\n";
  os << prefix << "noc cycles: " << report.nocCycles << "\n";
  os << prefix << "estimated cycles: " << report.estCycles << "\n";
  os << prefix << "compute utilisation: "
     << llvm::formatv("{0:P1}", report.computeUtil) << "\n";
  os << prefix << "binding constraint: " << report.bound << "\n";
  for (const std::string &suggestion : report.suggestions) {
    os << prefix << "suggested optimization: " << suggestion << "\n";
  }
  return out;
}

} // namespace analysis
} // namespace tensixflow
} // namespace mlir
