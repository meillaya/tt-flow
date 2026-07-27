//===- PlanReport.h ---------------------------------------------*- C++ -*-===//
//
// TensixFlow plan-report assembler (todo 30): the single seam that runs the four
// planners (grid / weight / CB / NoC) over a graph's operand shapes and packages
// their decisions into (a) the schema Plan consumed by tensixflow-translate and
// (b) the human-readable "Chosen layout: ..." report text. Pure data + math over
// the scheduler model — NO MLIR dependency — so the planning passes and the unit
// test share one implementation. The planners and the plan:: structs are reused
// from GridPlanner/WeightPlanner/CBAllocator/NoCEstimator and PlanSchema.h; this
// file only composes them.
//
//===----------------------------------------------------------------------===//

#ifndef TENSIXFLOW_SCHEDULER_PLANREPORT_H
#define TENSIXFLOW_SCHEDULER_PLANREPORT_H

#include "CBAllocator.h"
#include "GridPlanner.h"
#include "Model.h"
#include "NoCEstimator.h"
#include "WeightPlanner.h"

#include <string>

namespace mlir {
namespace tensixflow {
namespace scheduler {

/// The four planner decisions for one graph plus the assembled schema Plan. The
/// planning passes attach this to the IR (as a schedule attribute) and print it.
struct PlanReport {
  std::string graphId;
  std::string fusion;            // "fused_rmsnorm_matmul" / "fused_mlp" / "none"
  bool fused = false;
  std::string intermediateStorage; // "L1" when fused, else "DRAM"
  GridPlan grid;
  WeightPlan weight;
  CBAllocation cbs;
  NocTraffic noc;
  plan::Plan plan;
};

/// The plan inputs the passes derive from the IR: the graph id, the activation +
/// weight operand shapes, the fusion decision, and the element byte width.
struct PlanRequest {
  std::string graphId;
  TensorShape activation;
  TensorShape weight;
  std::string fusion;
  bool fused = false;
  int bytesPerElem = 2;
};

/// Run the grid/weight/CB/NoC planners over the operand shapes on the modeled
/// 8x8 grid and assemble the full report (planner decisions + schema Plan). The
/// Plan's noc field carries the traffic estimate; the savings breakdown lives on
/// `noc` (savedDramReadBytes etc.) for the report.
PlanReport buildPlanReport(const PlanRequest &req);

/// The human-readable "Chosen layout: ..." report text (the user-spec format:
/// activation_sharding, weight_strategy, intermediate_storage, fusion,
/// estimated_DRAM_reads_saved, estimated_intermediate_writes_saved).
std::string formatPlanReport(const PlanReport &report);

} // namespace scheduler
} // namespace tensixflow
} // namespace mlir

#endif // TENSIXFLOW_SCHEDULER_PLANREPORT_H
