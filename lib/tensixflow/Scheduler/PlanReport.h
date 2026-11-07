//===- PlanReport.h ---------------------------------------------*- C++ -*-===//
//
// TensixFlow plan-report assembler: the single seam that runs the four planners
// (grid, weight, CB, NoC) over a graph's operand shapes on a chosen device and
// packages their decisions into (a) the schema Plan consumed by
// tensixflow-translate and (b) the human-readable "Chosen layout:" report text.
// Pure data and math over the scheduler model with no MLIR dependency, so the
// planning passes and the unit tests share one implementation.
//
//===----------------------------------------------------------------------===//

#ifndef TENSIXFLOW_SCHEDULER_PLANREPORT_H
#define TENSIXFLOW_SCHEDULER_PLANREPORT_H

#include "CBAllocator.h"
#include "GridPlanner.h"
#include "Model.h"
#include "NoCEstimator.h"
#include "WeightPlanner.h"

#include "tensixflow/Generated/DeviceSpecs.h"

#include <string>

namespace mlir {
namespace tensixflow {
namespace scheduler {

/// The four planner decisions for one graph plus the assembled schema Plan. The
/// planning passes attach this to the IR as a schedule attribute and print it.
struct PlanReport {
  std::string graphId;
  std::string fusion;
  bool fused = false;
  std::string intermediateStorage;
  const device::DeviceSpec *device = nullptr;
  GridPlan grid;
  WeightPlan weight;
  CBAllocation cbs;
  NocTraffic noc;
  plan::Plan plan;
};

/// The plan inputs the passes derive from the IR: the graph id, the activation
/// and weight operand shapes, the fusion decision, the element byte width, and
/// the selected device. A null device means the generated default device; the
/// planner resolves the pointer once so every decision uses the same spec.
struct PlanRequest {
  std::string graphId;
  TensorShape activation;
  TensorShape weight;
  std::string fusion;
  bool fused = false;
  int bytesPerElem = 2;
  const device::DeviceSpec *device = nullptr;
};

/// Run the grid/weight/CB/NoC planners over the operand shapes on the selected
/// device's compute grid and assemble the full report (planner decisions plus
/// schema Plan). The Plan's device block records the selected spec; the noc
/// field carries the traffic estimate and the savings breakdown lives on `noc`.
PlanReport buildPlanReport(const PlanRequest &request);

/// The human-readable "Chosen layout: ..." report text (activation_sharding,
/// weight_strategy, intermediate_storage, fusion, device, and the estimated
/// DRAM and intermediate-write savings). Every figure is a static estimate.
std::string formatPlanReport(const PlanReport &report);

} // namespace scheduler
} // namespace tensixflow
} // namespace mlir

#endif // TENSIXFLOW_SCHEDULER_PLANREPORT_H
