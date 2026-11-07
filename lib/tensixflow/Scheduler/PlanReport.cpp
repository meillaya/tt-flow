//===- PlanReport.cpp -------------------------------------------*- C++ -*-===//
//
// Implementation of the TensixFlow plan-report assembler. See PlanReport.h:
// composes the four planners into a schema Plan plus report text, with no MLIR
// dependency.
//
//===----------------------------------------------------------------------===//

#include "PlanReport.h"

#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace tensixflow {
namespace scheduler {

namespace {

plan::TensorDesc toTensorDesc(const TensorShape &shape,
                              const std::string &placement) {
  plan::TensorDesc desc;
  desc.name = shape.name;
  desc.shape = shape.dims;
  desc.dtype = shape.dtype;
  desc.placement = placement;
  return desc;
}

/// The device block of the plan JSON. Only values the schema carries are copied
/// here; the perf model resolves the full spec by name.
plan::DeviceInfo toDeviceInfo(const device::DeviceSpec &deviceSpec) {
  plan::DeviceInfo info;
  info.name = deviceSpec.name;
  info.arch = deviceSpec.arch;
  info.gridX = deviceSpec.computeGrid.gridX;
  info.gridY = deviceSpec.computeGrid.gridY;
  info.l1UsableBytes = deviceSpec.l1UsableBytes;
  info.flopsPerCyclePerCore = deviceSpec.flopsPerCyclePerCore;
  info.dramBandwidthBytesPerSecond = deviceSpec.dramBandwidthBytesPerSecond;
  return info;
}

} // namespace

PlanReport buildPlanReport(const PlanRequest &request) {
  PlanReport report;
  report.graphId = request.graphId;
  report.fusion = request.fusion;
  report.fused = request.fused;
  report.intermediateStorage = request.fused ? "L1" : "DRAM";

  // Resolve the device once so every planner decision uses the same spec.
  report.device = request.device;
  if (report.device == nullptr) {
    report.device = device::defaultDevice();
  }
  const device::DeviceSpec &deviceSpec = *report.device;

  report.grid = planGrid(request.activation, deviceSpec, kDefaultNumCBs);
  report.weight = planWeight(request.weight, deviceSpec, request.bytesPerElem);
  report.cbs = allocateCBs(deviceSpec, kDefaultNumCBs);
  report.noc = estimateNoC(
      NocInputs{request.activation, request.weight, report.weight,
                request.fused},
      request.bytesPerElem);

  PlanDraft draft = toPlanDraft(report.grid);
  draft.graphId = request.graphId;
  draft.weightStrategy = report.weight.strategy;
  draft.activationSharding = report.grid.sharding;
  draft.cbSpecs = toCBSpecs(report.cbs);
  draft.tensors = {toTensorDesc(request.activation, "DRAM"),
                   toTensorDesc(request.weight, "DRAM")};
  draft.device = toDeviceInfo(deviceSpec);
  report.plan = toPlan(draft);
  report.plan.noc = toNocEstimate(report.noc);
  return report;
}

std::string formatPlanReport(const PlanReport &report) {
  std::string out;
  llvm::raw_string_ostream os(out);
  os << "Chosen layout: " << report.graphId << "\n"
     << "  device: " << report.plan.device.name << " ("
     << report.plan.device.arch << ")\n"
     << "  activation_sharding: " << report.plan.layout.activationSharding
     << "\n"
     << "  weight_strategy: " << report.plan.layout.weightStrategy << "\n"
     << "  intermediate_storage: " << report.intermediateStorage << "\n"
     << "  fusion: " << report.fusion << "\n"
     << "  estimated_DRAM_reads_saved: " << report.noc.savedDramReadBytes
     << " bytes\n"
     << "  estimated_intermediate_writes_saved: "
     << report.noc.savedDramWriteBytes << " bytes\n"
     << "  planned_cb_bytes_per_core: " << report.cbs.totalBytes
     << " bytes (depth " << report.cbs.depth << " tiles per buffer, budget "
     << report.device->cbBudget(device::kDefaultL1ReserveBytes) << " bytes)\n";
  return out;
}

} // namespace scheduler
} // namespace tensixflow
} // namespace mlir
