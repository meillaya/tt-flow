//===- EmissionEstimator.h - Emitted-path accounting -----------*- C++ -*-===//
//
// TensixFlow emitted-path accounting: the traffic and work-distribution numbers
// the emitted TT-Metalium program will produce, computed from the plan's tile
// geometry. Pure data and math with no MLIR dependency, so the planning pass
// and the unit tests share one implementation.
//
// The emitted matmul has two mappings:
//   "output-tile-ranges": each core owns a contiguous range of linear
//   output-tile ids and reads the A and B tiles of every output tile it owns,
//   so a tile in K is re-read once per output tile that needs it.
//   "blocked-reuse": each core owns a 2D block of output tiles and stages its
//   full activation row block and weight column block in L1 once. The
//   activation row block is read once per used column block (it is replicated
//   across columns); the weight column block is read once per used column block
//   and multicast down the used row blocks when more than one row block is
//   used. The remaining replication is A across the used column blocks.
//
// The blocked-reuse mapping is chosen when its two resident input buffers plus
// the output buffer fit the per-core circular-buffer budget. Otherwise the
// program keeps the output-tile-ranges mapping unchanged.
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

/// The mapping where one core owns a contiguous range of linear output-tile ids.
/// The emitter splits the output tiles with split_work_to_cores, which produces
/// exactly this mapping.
inline constexpr const char *kOutputTileRangesMapping =
    "output-tile-ranges";

/// The mapping where one core owns a 2D block of output tiles and stages its
/// activation row block and weight column block in L1 once.
inline constexpr const char *kBlockedReuseMapping = "blocked-reuse";

/// Output circular-buffer depth of the emitted matmul: the compute kernel packs
/// one result tile at a time and the writer drains it, so two tiles cover the
/// reader/compute handoff without changing the projected traffic.
inline constexpr int64_t kEmissionOutputCbTiles = 2;

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
  /// True when the plan emits the fused RMSNorm + MatMul kernel. That emitter
  /// stages activation rows itself with contiguous output-tile ranges, so the
  /// blocked-reuse selector applies to the plain matmul path only.
  bool fused = false;
};

/// The 2D block assignment of the blocked-reuse mapping for one tile geometry:
/// the block extent per core, the used grid extents, the deepest block of
/// output tiles, and the resident input-buffer depths.
struct BlockAssignment {
  int64_t rowsPerCore = 0;
  int64_t colsPerCore = 0;
  int64_t usedRows = 0;
  int64_t usedCols = 0;
  int64_t maxTilesPerCore = 0;
  int64_t in0Tiles = 0;
  int64_t in1Tiles = 0;
  int64_t cbTiles = 0;
  /// True when the emitted blocked-reuse reader multicasts each weight column
  /// block down the used row blocks from the top row block. A single used row
  /// block has no replication to remove, so it reads the weight directly.
  bool weightMulticast = false;
  bool feasible = false;
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

/// Plan the blocked-reuse 2D block assignment. The grid is split evenly:
/// rowsPerCore = ceil(mTiles / gridY), colsPerCore = ceil(nTiles / gridX); the
/// last used block in each dimension is smaller when the dimension does not
/// divide evenly. The assignment is feasible when the resident activation block
/// (rowsPerCore * kTiles tiles), the resident weight block (colsPerCore *
/// kTiles tiles), and the output buffer (kEmissionOutputCbTiles tiles) fit the
/// per-core circular-buffer budget.
BlockAssignment planBlockAssignment(const EmissionInputs &inputs);

/// Project the emitted-path accounting for a plan. For the blocked-reuse
/// mapping (selected when planBlockAssignment is feasible and the plan is not
/// fused) the formulas are:
///   cores_used = usedRows * usedCols
///   max_tiles_per_core = largest row block * largest column block
///   dram_read_bytes = (kTiles * usedCols * sum(rows per used row block) +
///       kTiles * weightReplication * sum(cols per used column block)) *
///       tileBytes
///   dram_write_bytes = mTiles * nTiles * tileBytes
/// with weightReplication = usedRows when the weight column block is read on
/// every used row block, and 1 when the blocked-reuse reader multicasts it down
/// the used row blocks. The activation row block is always replicated across
/// the used column blocks; that is the read the reuse mapping cannot remove
/// without an activation multicast.
///
/// For the output-tile-ranges mapping the formulas are:
///   cores_used = min(gridX * gridY, mTiles * nTiles)
///   max_tiles_per_core = ceil(mTiles * nTiles / cores_used)
///   dram_read_bytes = mTiles * nTiles * kTiles * 2 * tileBytes
///   dram_write_bytes = mTiles * nTiles * tileBytes
///
/// The fused fields are mapping independent: a fused variant of the same
/// matmul stages the rows of a contiguous output-tile range:
///   max_rows_per_core = min(mTiles, (ceil(mTiles * nTiles /
///       min(grid cores, mTiles * nTiles)) - 1) / nTiles + 2)
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
