//===- PlanReport.cpp -------------------------------------------*- C++ -*-===//
//
// Implementation of the TensixFlow plan-report assembler (todo 30). See
// PlanReport.h — composes the four planners into a schema Plan + report text;
// NO MLIR dependency.
//
//===----------------------------------------------------------------------===//

#include "PlanReport.h"

#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace tensixflow {
namespace scheduler {

namespace {
// The modeled Wormhole n150 core grid and the default reader/compute/writer
// circular-buffer count (matches the CB allocator's 3-CB convention).
constexpr int64_t kGridY = 8;
constexpr int64_t kGridX = 8;
constexpr int64_t kNumCBs = 3;

plan::TensorDesc toTensorDesc(const TensorShape &shape,
                              const std::string &placement) {
  plan::TensorDesc desc;
  desc.name = shape.name;
  desc.shape = shape.dims;
  desc.dtype = shape.dtype;
  desc.placement = placement;
  return desc;
}
} // namespace

PlanReport buildPlanReport(const PlanRequest &req) {
  PlanReport report;
  report.graphId = req.graphId;
  report.fusion = req.fusion;
  report.fused = req.fused;
  report.intermediateStorage = req.fused ? "L1" : "DRAM";

  plan::ShardGrid grid{kGridY, kGridX};
  report.grid = planGrid(req.activation, grid, req.bytesPerElem, kNumCBs);
  report.weight = planWeight(req.weight, grid, req.bytesPerElem);
  report.cbs = allocateCBs(kNumCBs, req.bytesPerElem);
  report.noc = estimateNoC(
      NocInputs{req.activation, req.weight, report.weight, req.fused},
      req.bytesPerElem);

  PlanDraft draft = toPlanDraft(report.grid, req.bytesPerElem);
  draft.graphId = req.graphId;
  draft.weightStrategy = report.weight.strategy;
  draft.activationSharding = report.grid.sharding;
  draft.cbSpecs = toCBSpecs(report.cbs);
  draft.tensors = {toTensorDesc(req.activation, "DRAM"),
                   toTensorDesc(req.weight, "DRAM")};
  report.plan = toPlan(draft);
  report.plan.noc = toNocEstimate(report.noc);
  return report;
}

std::string formatPlanReport(const PlanReport &report) {
  std::string out;
  llvm::raw_string_ostream os(out);
  os << "Chosen layout: " << report.graphId << "\n"
     << "  activation_sharding: " << report.plan.layout.activationSharding
     << "\n"
     << "  weight_strategy: " << report.plan.layout.weightStrategy << "\n"
     << "  intermediate_storage: " << report.intermediateStorage << "\n"
     << "  fusion: " << report.fusion << "\n"
     << "  estimated_DRAM_reads_saved: " << report.noc.savedDramReadBytes
     << "\n"
     << "  estimated_intermediate_writes_saved: "
     << report.noc.savedDramWriteBytes << "\n";
  return out;
}

} // namespace scheduler
} // namespace tensixflow
} // namespace mlir
