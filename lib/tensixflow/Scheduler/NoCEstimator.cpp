//===- NoCEstimator.cpp -----------------------------------------*- C++ -*-===//
//
// NoC traffic estimator. First-order static analytic traffic model, not a
// measured hardware number. Fusion keeps the rmsnorm-output intermediate in L1,
// eliding one intermediate DRAM write and one DRAM re-read. multicastBytes is
// consumed from the weight plan, not recomputed.
//
//===----------------------------------------------------------------------===//

#include "NoCEstimator.h"

#include <cstddef>

namespace mlir {
namespace tensixflow {
namespace scheduler {

namespace {

/// matmul output bytes: the output is [activation leading dims (all but the
/// contracting K = activation last dim)] by [weight N = weight last dim].
/// Degenerate shapes (empty activation or weight) yield 0.
int64_t matmulOutputBytes(const TensorShape &activation,
                          const TensorShape &weight, int bytesPerElem) {
  if (activation.dims.empty() || weight.dims.empty()) {
    return 0;
  }
  int64_t leading = 1;
  for (size_t i = 0; i + 1 < activation.dims.size(); ++i) {
    leading *= activation.dims[i];
  }
  return leading * weight.dims.back() * bytesPerElem;
}

} // namespace

NocTraffic estimateNoC(const NocInputs &inputs, int bytesPerElem) {
  const int64_t actBytes = inputs.activation.bytes(bytesPerElem);
  const int64_t weightBytes = inputs.weight.bytes(bytesPerElem);
  // rmsnorm preserves shape, so the intermediate is the activation size.
  const int64_t intermediateBytes = actBytes;
  const int64_t outBytes =
      matmulOutputBytes(inputs.activation, inputs.weight, bytesPerElem);

  NocTraffic traffic;
  // Multicast traffic is decided by the weight plan and is independent of
  // fusion; consume it verbatim.
  traffic.estimate.multicastBytes = inputs.weightPlan.multicastBytes;

  if (inputs.fused) {
    // Fused: the intermediate stays in L1 and never round-trips DRAM.
    traffic.estimate.dramReadBytes = actBytes + weightBytes;
    traffic.estimate.dramWriteBytes = outBytes;
    traffic.estimate.l1Bytes = intermediateBytes;
    traffic.savedDramReadBytes = intermediateBytes;
    traffic.savedDramWriteBytes = intermediateBytes;
    traffic.savedIntermediateBytes = intermediateBytes;
    return traffic;
  }

  // Unfused: rmsnorm reads the activation and writes the intermediate, then
  // matmul re-reads the intermediate, reads the weight, and writes the output.
  // No fusion means no savings, so the saved fields stay at their zero defaults.
  traffic.estimate.dramReadBytes = actBytes + intermediateBytes + weightBytes;
  traffic.estimate.dramWriteBytes = intermediateBytes + outBytes;
  traffic.estimate.l1Bytes = 0;
  return traffic;
}

plan::NocEstimate toNocEstimate(const NocTraffic &traffic) {
  return traffic.estimate;
}

} // namespace scheduler
} // namespace tensixflow
} // namespace mlir
