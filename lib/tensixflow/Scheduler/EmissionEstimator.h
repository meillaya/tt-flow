//===- EmissionEstimator.h - Emitted-path accounting -----------*- C++ -*-===//
//
// TensixFlow emitted-path accounting: the traffic and work-distribution numbers
// the emitted TT-Metalium program will produce, computed from the plan's tile
// geometry. Pure data and math with no MLIR dependency, so the planning pass
// and the unit tests share one implementation.
//
// The emitted program assigns whole output tiles to cores from a row-major
// range of linear tile ids ("output-tile-ranges"). Each core reads the A and B
// tiles of every output tile it owns, so a tile in K is re-read once per output
// row that needs it: no tile reuse is modeled. The DRAM read figure is
// therefore an upper bound, not a measured count.
//
// Every number here is projected from static shapes. None of it is measured on
// hardware.
//
//===----------------------------------------------------------------------===//

#ifndef TENSIXFLOW_SCHEDULER_EMISSIONESTIMATOR_H
#define TENSIXFLOW_SCHEDULER_EMISSIONESTIMATOR_H

#include <cstdint>
#include <string>

namespace mlir {
namespace tensixflow {
namespace scheduler {

/// The module attribute name the planning pass stamps with the emitted-path
/// block. The emitter and the JSON translate read the attribute by this name.
inline constexpr const char *kEmissionAttrName = "tensixflow.emission";

/// The emitted mapping this estimator describes: one core owns a contiguous
/// range of linear output-tile ids. The emitter creates the kernels on the
/// plan grid and splits the output tiles with split_work_to_cores, which
/// produces exactly this mapping.
inline constexpr const char *kOutputTileRangesMapping =
    "output-tile-ranges";

/// The tt-metal tile is 32x32 elements and bf16 is 2 bytes, so the default tile
/// is 2048 bytes. Callers pass the device table's tile size instead of relying
/// on this fallback.
inline constexpr int64_t kEmissionTileBytes = 2048;

/// Inputs to the emitted-path accounting: the matmul tile geometry, the plan
/// grid, and the per-core L1 budget available to circular buffers.
struct EmissionInputs {
  int64_t mTiles = 0;
  int64_t nTiles = 0;
  int64_t kTiles = 0;
  int64_t gridX = 0;
  int64_t gridY = 0;
  int64_t tileBytes = kEmissionTileBytes;
  int64_t cbBudgetBytes = 0;
};

/// One projected emitted-path block. The values are the contract the plan JSON
/// records and the generated project README quotes.
struct EmissionPlan {
  std::string mapping = kOutputTileRangesMapping;
  int64_t coresUsed = 0;
  int64_t maxTilesPerCore = 0;
  int64_t dramReadBytes = 0;
  int64_t dramWriteBytes = 0;
  int64_t fusedStageTiles = 0;
  bool fusedFeasible = false;
};

/// Project the emitted-path accounting for a plan. The formulas are:
///   cores_used = min(gridX * gridY, mTiles * nTiles)
///   max_tiles_per_core = ceil(mTiles * nTiles / cores_used)
///   dram_read_bytes = mTiles * nTiles * kTiles * 2 * tileBytes
///   dram_write_bytes = mTiles * nTiles * tileBytes
///   max_rows_per_core = min(mTiles, (max_tiles_per_core - 1) / nTiles + 2)
///   fused_stage_tiles = max_rows_per_core * kTiles
///   fused_feasible = 2 * fused_stage_tiles + 2 * kTiles + 3 <= cbBudgetTiles
/// with cbBudgetTiles = cbBudgetBytes / tileBytes. The integer divisions are
/// floor divisions. Degenerate inputs (no tiles, no cores, no budget) produce
/// zeroed fields with fusedFeasible false instead of dividing by zero.
EmissionPlan estimateEmission(const EmissionInputs &inputs);

} // namespace scheduler
} // namespace tensixflow
} // namespace mlir

#endif // TENSIXFLOW_SCHEDULER_EMISSIONESTIMATOR_H
