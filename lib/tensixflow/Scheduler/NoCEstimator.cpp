//===- NoCEstimator.cpp -----------------------------------------*- C++ -*-===//
//
// NoC traffic estimator (todo 29). First-order static analytic traffic model —
// NOT measured hardware numbers. Fusion keeps the rmsnorm-output intermediate
// in L1, eliding one intermediate DRAM write and one DRAM re-read
// (arXiv:2606.09879). multicastBytes is consumed from the weight plan (todo 27),
// not recomputed.
//
//===----------------------------------------------------------------------===//

#include "NoCEstimator.h"

#include <cstddef>

namespace mlir {
namespace tensixflow {
namespace scheduler {

namespace {

/// matmul output bytes: the output is [activation leading dims (all but the
/// contracting K = activation last dim)] x [weight N = weight last dim].
/// Degenerate shapes (empty activation or weight) yield 0 (guarded, no crash).
int64_t matmulOutputBytes(const TensorShape &activation, const TensorShape &weight,
                          int bytesPerElem) {
  if (activation.dims.empty() || weight.dims.empty())
    return 0;
  int64_t leading = 1;
  for (size_t i = 0; i + 1 < activation.dims.size(); ++i)
    leading *= activation.dims[i];
  return leading * weight.dims.back() * bytesPerElem;
}

} // namespace

NocTraffic estimateNoC(const NocInputs &inputs, int bytesPerElem) {
  const int64_t actBytes = inputs.activation.bytes(bytesPerElem);
  const int64_t weightBytes = inputs.weight.bytes(bytesPerElem);
  // rmsnorm output shape == activation shape, so the intermediate is actBytes.
  const int64_t intermediateBytes = actBytes;
  const int64_t outBytes =
      matmulOutputBytes(inputs.activation, inputs.weight, bytesPerElem);

  NocTraffic traffic;
  // Multicast traffic is decided by the weight plan (todo 27), independent of
  // fusion — consume it verbatim, do not recompute.
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

  // Unfused: rmsnorm reads the activation + writes the intermediate; matmul
  // re-reads the intermediate + reads the weight + writes the output. No fusion
  // -> no savings (the saved fields stay at their zero defaults).
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
