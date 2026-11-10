//===- Codegen/EmitCommon.h - Shared helpers for the Metalium emitters -*- C++ -*-===//
//
// TensixFlow: An MLIR Compiler for Tile-Native LLM Kernels on
// Tenstorrent-Style Spatial Accelerators.
//
//===----------------------------------------------------------------------===//
//
// Helpers shared by the four Metalium emitters: the plan geometry read off the
// lowered schedule IR, the circular-buffer allocation read off the
// #tensixflow.schedule attribute, and the tile-loop trip counts.
//
// The emitters target the pinned tt-metal release (deps/tt-metal.lock). The
// signatures they emit were checked against the headers extracted from the
// pinned tt-metalium-dev deb, and the emitted project was compiled, linked,
// and run against that release. Nothing here is a stub.
//
//===----------------------------------------------------------------------===//

#ifndef TENSIXFLOW_CODEGEN_EMITCOMMON_H
#define TENSIXFLOW_CODEGEN_EMITCOMMON_H

#include "tensixflow/TensixflowAttrs.h"
#include "tensixflow/TensixflowSchedule.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <algorithm>
#include <cstdint>
#include <optional>

namespace mlir {
namespace tensixflow {

// The modeled Tensix tile. The emitter needs the value before the host program
// runs, so it is a compile-time constant rather than a device-spec lookup.
inline constexpr int64_t kEmitTileHeight = 32;
inline constexpr int64_t kEmitTileWidth = 32;
inline constexpr int64_t kEmitTileBytes = 2048;

// Output circular-buffer depth of the emitted matmul: the compute kernel packs
// one result tile at a time and the writer drains it, so two tiles cover the
// compute/writer handoff without changing the projected traffic. The same depth
// is part of the blocked-reuse feasibility selector (EmissionEstimator.h).
inline constexpr int64_t kEmitOutputCbTiles = 2;

// The module attribute --tensixflow-plan-noc-sharding stamps with the
// emitted-path accounting block. The Metalium emitter requires it.
inline constexpr llvm::StringLiteral kEmissionAttrName =
    "tensixflow.emission";

// The output-tile mapping where each core owns a contiguous range of linear
// output-tile ids, which is what split_work_to_cores produces.
inline constexpr llvm::StringLiteral kOutputTileRangesMapping =
    "output-tile-ranges";

// The output-tile mapping where each core owns a 2D block of output tiles and
// stages its activation row block and weight column block in L1 once.
inline constexpr llvm::StringLiteral kBlockedReuseMapping = "blocked-reuse";

// The output-tile mapping the emitted matmul realizes.
enum class MatmulMapping {
  OutputTileRanges,
  BlockedReuse,
};

// The per-core geometry of the selected mapping. The output-tile-ranges fields
// are zero; the blocked-reuse fields size the resident circular buffers and the
// 2D block the host assigns to each core.
struct MatmulEmission {
  MatmulMapping mapping = MatmulMapping::OutputTileRanges;
  int64_t rowsPerCore = 0;
  int64_t colsPerCore = 0;
  int64_t usedRows = 0;
  int64_t usedCols = 0;
  int64_t in0DepthTiles = 0;
  int64_t in1DepthTiles = 0;
  // True when the blocked-reuse reader multicasts each weight column block
  // down the used row blocks from the top row block instead of reading it on
  // every core. The activation row block stays unicast in both cases.
  bool multicastB = false;

  bool blockedReuse() const { return mapping == MatmulMapping::BlockedReuse; }
};

// The attribute --tensixflow-lower-to-ttkernel stamps on the sched.kernel. It
// carries the matmul shape in elements; trip counts alone cannot distinguish
// the dimensions, and the host program needs M/N/K to size the DRAM buffers and
// run the reference matmul.
inline constexpr llvm::StringLiteral kMatmulShapeAttrName =
    "tensixflow.matmul_shape";

// The plan geometry the host emitter consumes. Elements are rounded up to whole
// tiles because the device operates on tiles.
struct MatmulShape {
  int64_t m = 0;
  int64_t n = 0;
  int64_t k = 0;

  int64_t mTiles() const { return ceilTiles(m, kEmitTileHeight); }
  int64_t nTiles() const { return ceilTiles(n, kEmitTileWidth); }
  int64_t kTiles() const { return ceilTiles(k, kEmitTileWidth); }
  int64_t mPadded() const { return mTiles() * kEmitTileHeight; }
  int64_t nPadded() const { return nTiles() * kEmitTileWidth; }
  int64_t kPadded() const { return kTiles() * kEmitTileWidth; }

private:
  static int64_t ceilTiles(int64_t elements, int64_t tileSize) {
    if (elements <= 0) {
      return 0;
    }
    return (elements + tileSize - 1) / tileSize;
  }
};

// Read the matmul shape the lowering pass recorded. Returns std::nullopt when
// the attribute is absent or malformed; the pass reports that as a diagnostic
// instead of emitting a project with invented dimensions.
inline std::optional<MatmulShape> readMatmulShape(
    TensixflowSchedKernelOp kernel) {
  auto attr =
      kernel->getAttrOfType<DenseI64ArrayAttr>(kMatmulShapeAttrName);
  if (!attr) {
    return std::nullopt;
  }
  llvm::ArrayRef<int64_t> values = attr.asArrayRef();
  if (values.size() != 3) {
    return std::nullopt;
  }
  if (values[0] <= 0 || values[1] <= 0 || values[2] <= 0) {
    return std::nullopt;
  }
  return MatmulShape{values[0], values[1], values[2]};
}

// One circular buffer of the plan's allocation: the tt::CBIndex value, the
// depth in tiles, and the per-tile byte size.
struct CbSlot {
  int64_t index = 0;
  int64_t depth = 0;
  int64_t tileBytes = 0;

  int64_t bytes() const { return depth * tileBytes; }
};

// Read the schedule's parallel CB arrays. The plan pipeline always allocates
// three buffers (in0, in1, out); the documented fallback keeps a hand-written
// schedule attribute from crashing the emitter.
inline llvm::SmallVector<CbSlot> readCbSlots(TensixflowScheduleAttr schedule) {
  llvm::SmallVector<CbSlot> slots;
  ArrayAttr indices = schedule.getCbIndices();
  ArrayAttr depths = schedule.getCbDepths();
  ArrayAttr tileBytes = schedule.getCbTileBytes();
  const unsigned count =
      std::min({indices.size(), depths.size(), tileBytes.size()});
  for (unsigned i = 0; i < count; ++i) {
    slots.push_back({cast<IntegerAttr>(indices[i]).getInt(),
                     cast<IntegerAttr>(depths[i]).getInt(),
                     cast<IntegerAttr>(tileBytes[i]).getInt()});
  }
  if (slots.size() < 3) {
    slots.clear();
    slots.push_back({0, 2, kEmitTileBytes});
    slots.push_back({1, 2, kEmitTileBytes});
    slots.push_back({16, 2, kEmitTileBytes});
  }
  return slots;
}

// The trip count of the first sched.tile_loop inside a tile program. The
// lowering pass makes the reader/compute programs stream the K dimension and
// the writer drain every output tile; the count is 1 for an absent program.
inline int64_t tripCount(Operation *op) {
  int64_t trip = 1;
  if (!op) {
    return trip;
  }
  op->walk([&](TensixflowSchedTileLoopOp loop) {
    trip = loop.getTripCount();
    return WalkResult::interrupt();
  });
  return trip;
}

// The plan's fusion decision. The Metalium emitter supports the pure matmul
// path only; a fused graph is rejected with an actionable diagnostic before any
// output is written.
inline llvm::StringRef fusionKind(TensixflowScheduleAttr schedule) {
  return schedule.getFusion().getValue();
}

inline bool isFusedPlan(TensixflowScheduleAttr schedule) {
  return fusionKind(schedule) != "none";
}

// The one fused pattern the Metalium emitter supports.
inline constexpr llvm::StringLiteral kFusedRmsnormMatmulKind =
    "fused_rmsnorm_matmul";

inline bool isFusedRmsnormMatmul(TensixflowScheduleAttr schedule) {
  return fusionKind(schedule) == kFusedRmsnormMatmulKind;
}

// The fused plan's circular-buffer positions in the schedule allocation. The
// allocation is deterministic (see CBAllocator.h): index order, not role order.
inline constexpr std::size_t kFusedActivationSlot = 0;
inline constexpr std::size_t kFusedWeightSlot = 1;
inline constexpr std::size_t kFusedGammaSlot = 2;
inline constexpr std::size_t kFusedScalerSlot = 3;
inline constexpr std::size_t kFusedNormalizedSlot = 4;
inline constexpr std::size_t kFusedOutputSlot = 5;
inline constexpr std::size_t kFusedCbSlotCount = 6;

// The fused compute kernel holds one output tile per output column block in the
// destination registers while it reduces over K. FP32 destination accumulation
// leaves eight destination tiles, so that is the limit on output column tiles.
inline constexpr int64_t kFusedOutputColumnTileLimit = 8;

// The RMSNorm epsilon recorded on the fused graph's prologue op. The compute
// kernel bakes the epsilon into the SFPU scalar add, so the emitter needs the
// value. Returns a null attribute when the kernel carries no rmsnorm op.
inline FloatAttr fusedEpsilon(TensixflowSchedKernelOp kernel) {
  FloatAttr epsilon;
  kernel.walk([&](TensixflowSchedRmsnormOp op) {
    if (!epsilon) {
      epsilon = op.getEpsilonAttr();
    }
  });
  return epsilon;
}

} // namespace tensixflow
} // namespace mlir

#endif // TENSIXFLOW_CODEGEN_EMITCOMMON_H
