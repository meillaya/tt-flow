//===- NoCEstimator.h -------------------------------------------*- C++ -*-===//
//
// NoC traffic estimator (todo 29): a FIRST-ORDER STATIC ANALYTIC model of the
// DRAM / L1 / NoC traffic a fused-rmsnorm+matmul plan incurs, and the savings
// fusion delivers over the unfused op pair. These are STATIC ESTIMATES computed
// from tensor shapes + the weight plan — they are NOT measured hardware numbers
// and must never be reported as such (per the plan's Must-NOT).
//
// Traffic model (the modeled Wormhole n150, bf16):
//   - UNFUSED: every op round-trips DRAM. rmsnorm reads the activation from DRAM
//     and writes its output (the intermediate) back to DRAM; matmul re-reads
//     that intermediate from DRAM, reads the weight from DRAM, and writes the
//     matmul output to DRAM.
//       dramReads  = activation + intermediate + weight
//       dramWrites = intermediate + matmulOutput
//   - FUSED: the rmsnorm-output intermediate is kept in L1 and never round-trips
//     DRAM (arXiv:2606.09879 — fusion reduces DRAM r/w of intermediates; NoC
//     multicast alleviates DRAM contention).
//       dramReads  = activation + weight
//       dramWrites = matmulOutput
//       l1Bytes    = intermediate (held resident in L1 across the two ops)
//   So fusion saves exactly one intermediate DRAM write AND one intermediate
//   DRAM re-read; savedIntermediateBytes = the rmsnorm-output tensor bytes
//   (== activation bytes, since rmsnorm output shape == input shape).
//
// multicastBytes is CONSUMED from the weight plan (WeightPlan.multicastBytes,
// todo 27) — it is NOT recomputed here; it is > 0 for the multicast strategy
// and 0 when the weight is replicated.
//
// Pure data + math over the scheduler model (Model.h) — NO MLIR dependency. The
// traffic figures reuse plan::NocEstimate (PlanSchema.h, D15); nothing is
// redefined. The report seam is toNocEstimate (todo 30 drops it into Plan.noc).
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

/// The plan inputs the estimator reasons about: the activation + weight tensor
/// shapes, the weight plan (whose multicastBytes is consumed verbatim), and
/// whether the rmsnorm+matmul pair is fused (true) or run as two unfused ops.
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
/// savings (there is nothing to save without fusion).
struct NocTraffic {
  plan::NocEstimate estimate;         // dramRead/Write, l1, multicast bytes
  int64_t savedDramReadBytes = 0;     // intermediate DRAM re-read elided by fusion
  int64_t savedDramWriteBytes = 0;    // intermediate DRAM write elided by fusion
  int64_t savedIntermediateBytes = 0; // intermediate bytes never round-tripping DRAM
};

/// Estimate the NoC/DRAM/L1 traffic for a plan. See the traffic model in the
/// header banner. `bytesPerElem` sizes every tensor (bf16 -> 2); the matmul
/// output shape is derived as [activation leading dims] x [weight last dim].
NocTraffic estimateNoC(const NocInputs &inputs, int bytesPerElem);

/// The report seam (todo 30): the traffic figures, already plan::NocEstimate,
/// ready to drop into Plan.noc.
plan::NocEstimate toNocEstimate(const NocTraffic &traffic);

} // namespace scheduler
} // namespace tensixflow
} // namespace mlir

#endif // TENSIXFLOW_SCHEDULER_NOCESTIMATOR_H
