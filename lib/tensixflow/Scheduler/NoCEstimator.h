//===- NoCEstimator.h -------------------------------------------*- C++ -*-===//
//
// NoC traffic estimator: a first-order static analytic model of the DRAM, L1,
// and NoC traffic a fused rmsnorm+matmul plan incurs, and the savings fusion
// delivers over the unfused op pair. Every figure is a static estimate computed
// from tensor shapes and the weight plan; none is a measured hardware number.
//
// Traffic model (bf16 element sizes, device independent):
//   - Unfused: every op round-trips DRAM. rmsnorm reads the activation from
//     DRAM and writes its output (the intermediate) back to DRAM; matmul
//     re-reads that intermediate, reads the weight, and writes the matmul
//     output.
//       dramReads  = activation + intermediate + weight
//       dramWrites = intermediate + matmulOutput
//   - Fused: the rmsnorm-output intermediate stays in L1 and never round-trips
//     DRAM, which is the traffic saving fusion buys.
//       dramReads  = activation + weight
//       dramWrites = matmulOutput
//       l1Bytes    = intermediate (resident in L1 across the two ops)
//   Fusion saves exactly one intermediate DRAM write and one intermediate DRAM
//   re-read, so savedIntermediateBytes is the rmsnorm-output tensor size (equal
//   to the activation size because rmsnorm preserves shape).
//
// multicastBytes comes from the weight plan verbatim, not from a second
// calculation: it is positive for the multicast strategy and zero when the
// weight is replicated.
//
// Pure data and math over the scheduler model (Model.h) with no MLIR
// dependency. The traffic figures reuse plan::NocEstimate from PlanSchema.h.
//
//===----------------------------------------------------------------------===//

#ifndef TENSIXFLOW_SCHEDULER_NOCESTIMATOR_H
#define TENSIXFLOW_SCHEDULER_NOCESTIMATOR_H

#include "Model.h"
#include "WeightPlanner.h"

#include <cstdint>

namespace mlir {
namespace tensixflow {
namespace scheduler {

//===----------------------------------------------------------------------===//
// Estimator inputs
//===----------------------------------------------------------------------===//

/// The plan inputs the estimator reasons about: the activation and weight
/// tensor shapes, the weight plan (whose multicastBytes is consumed verbatim),
/// and whether the rmsnorm+matmul pair is fused or runs as two unfused ops.
struct NocInputs {
  TensorShape activation;
  TensorShape weight;
  WeightPlan weightPlan;
  bool fused = false;
};

//===----------------------------------------------------------------------===//
// Estimator result
//===----------------------------------------------------------------------===//

/// The static traffic estimate for a plan (reusing plan::NocEstimate for the
/// four traffic figures) plus the unfused-vs-fused savings breakdown. The saved
/// figures are populated only for a fused plan; a non-fused plan reports zero
/// savings because there is nothing to save without fusion.
struct NocTraffic {
  plan::NocEstimate estimate;         // dramRead/Write, l1, multicast bytes
  int64_t savedDramReadBytes = 0;     // intermediate DRAM re-read elided
  int64_t savedDramWriteBytes = 0;    // intermediate DRAM write elided
  int64_t savedIntermediateBytes = 0; // intermediate never round-tripping DRAM
};

/// Estimate the NoC/DRAM/L1 traffic for a plan. See the traffic model in the
/// header banner. `bytesPerElem` sizes every tensor (bf16 gives 2); the matmul
/// output shape is the activation leading dims by the weight last dim.
NocTraffic estimateNoC(const NocInputs &inputs, int bytesPerElem);

/// The report seam: the traffic figures, already plan::NocEstimate, ready to
/// drop into plan::Plan.noc.
plan::NocEstimate toNocEstimate(const NocTraffic &traffic);

} // namespace scheduler
} // namespace tensixflow
} // namespace mlir

#endif // TENSIXFLOW_SCHEDULER_NOCESTIMATOR_H
