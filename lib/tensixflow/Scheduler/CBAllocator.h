//===- CBAllocator.h --------------------------------------------*- C++ -*-===//
//
// Circular-buffer allocator: assigns CB indices and a feasible depth to the
// reader/compute/writer circular buffers of a fused op. The per-core budget
// comes from the generated device table; the depth starts at the tt-metal
// example depth of two tiles per buffer and clamps to the feasible maximum, so
// the total footprint never exceeds the budget.
//
// Indices follow the tt-metal convention (tt::CBIndex runs c_0..c_31): the
// default three-CB pipeline maps in0 = c_0, in1 = c_1, out = c_16; any other CB
// count gets sequential indices 0..numCBs-1. A request that would run past c_31
// is flagged infeasible and never emits an out-of-range index.
//
// Pure data and math over the scheduler model (Model.h) with no MLIR
// dependency. The report seam is the allocation's std::vector<plan::CBSpec>,
// suitable for PlanDraft.cbSpecs.
//
//===----------------------------------------------------------------------===//

#ifndef TENSIXFLOW_SCHEDULER_CBALLOCATOR_H
#define TENSIXFLOW_SCHEDULER_CBALLOCATOR_H

#include "GridPlanner.h"
#include "Model.h"

#include "tensixflow/Generated/DeviceSpecs.h"

#include <cstdint>
#include <vector>

namespace mlir {
namespace tensixflow {
namespace scheduler {

//===----------------------------------------------------------------------===//
// Allocation result
//===----------------------------------------------------------------------===//

/// The allocator's decision: one plan::CBSpec per circular buffer (index, depth
/// in tiles, per-tile bytes), the total CB footprint in bytes, the depth the
/// allocator settled on, and whether the allocation fits the per-core CB budget
/// with every index in the valid c_0..c_31 range.
struct CBAllocation {
  std::vector<plan::CBSpec> cbs;
  int64_t totalBytes = 0;
  int64_t depth = 0;
  bool feasible = false;
};

/// Allocate `numCBs` circular buffers from the device's per-core CB budget.
///   - Indices: the default numCBs == 3 follows the tt-metal reader/compute/
///     writer convention (in0 = kCBIn0, in1 = kCBIn1, out = kCBOut); any other
///     count gets sequential indices 0..numCBs-1. Every emitted index satisfies
///     isValidCBIndex; a request that would run past c_31 is flagged infeasible
///     and emits no index above 31.
///   - Depth: requestedDepth (default two tiles, the tt-metal example depth)
///     clamped to maxCBDepth(deviceSpec, numCBs, reserveBytes), so totalBytes =
///     numCBs * depth * deviceSpec.tileBytes never exceeds the CB budget.
///   - Feasibility: depth > 0, totalBytes within the budget, and all indices
///     valid. A degenerate request (numCBs <= 0 or a non-positive requested
///     depth) is guarded: feasible == false, no crash, no division by zero.
CBAllocation allocateCBs(
    const device::DeviceSpec &deviceSpec, int64_t numCBs,
    int64_t requestedDepth = device::kDefaultCBDepthTiles,
    int64_t reserveBytes = device::kDefaultL1ReserveBytes);

/// The report seam: the allocation's CB specs, already plan::CBSpec, ready to
/// drop into PlanDraft.cbSpecs.
std::vector<plan::CBSpec> toCBSpecs(const CBAllocation &allocation);

//===----------------------------------------------------------------------===//
// Fused RMSNorm + MatMul allocation
//===----------------------------------------------------------------------===//

/// The fused plan's circular buffers, in the deterministic allocation order the
/// compute kernel reads them by: activation stage c_0, weight c_1, gamma c_2,
/// reduce scaler c_3, normalized activation c_4, output c_16.
enum class FusedCbSlot : std::size_t {
  ActivationStage = 0,
  Weight = 1,
  Gamma = 2,
  Scaler = 3,
  NormalizedActivation = 4,
  Output = 5,
};

inline constexpr std::size_t kFusedCbCount = 6;
inline constexpr int64_t kFusedCBIndices[kFusedCbCount] = {0, 1, 2, 3, 4, 16};
inline constexpr int64_t kFusedScalerCbDepth = 1;
inline constexpr int64_t kFusedOutputCbDepth = 2;

/// Circular-buffer tiles the fused plan needs: the staged activation rows and
/// the normalized activation rows (stageTiles each), the weight tiles and the
/// gamma tiles (kTiles each), one reduce scaler tile, and two output tiles.
int64_t fusedCbTileCount(int64_t kTiles, int64_t stageTiles);

/// Allocate the fused plan's six circular buffers from the device's per-core CB
/// budget, in FusedCbSlot order:
///   c_0 activation stage  depth stageTiles
///   c_1 weight            depth kTiles
///   c_2 gamma             depth kTiles
///   c_3 scaler            depth 1
///   c_4 normalized        depth stageTiles
///   c_16 output           depth 2
/// Feasibility is true only when every index is valid and the total footprint
/// fits the budget; a degenerate request (non-positive kTiles or stageTiles)
/// returns feasible == false with no buffers instead of dividing by zero.
CBAllocation allocateFusedCBs(
    const device::DeviceSpec &deviceSpec, int64_t kTiles, int64_t stageTiles,
    int64_t reserveBytes = device::kDefaultL1ReserveBytes);

} // namespace scheduler
} // namespace tensixflow
} // namespace mlir

#endif // TENSIXFLOW_SCHEDULER_CBALLOCATOR_H
