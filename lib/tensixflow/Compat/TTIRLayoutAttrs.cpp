//===- Compat/TTIRLayoutAttrs.cpp - TTIR layout attr vocabulary --*- C++ -*-===//
//
// TensixFlow: An MLIR Compiler for Tile-Native LLM Kernels on
// Tenstorrent-Style Spatial Accelerators.
//
//===----------------------------------------------------------------------===//

#include "tensixflow/Compat/TTIRLayoutAttrs.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>

namespace mlir {
namespace tensixflow {

namespace {

/// Tile edge the pinned layout builder aligns the last two logical dims to.
constexpr int64_t kTileAlignment = 32;

/// Fill value for out-of-bounds elements. `undef` is the enum spelling in
/// every attribute of the pinned fixture
/// test/ttmlir/Dialect/TTCore/metal_layout_attr_verify.mlir.
constexpr llvm::StringLiteral kOobVal = "undef";

/// Memory layout of the emitted tensors. The emitter runs before sharding, so
/// every tensor is interleaved; `sharded` appears only after PlanNocSharding.
constexpr llvm::StringLiteral kMemoryLayout = "interleaved";

/// Identity index map sentinel the pinned parser accepts through its
/// custom<IdentityAffineMap> field; the fixture spells it `map(0)`.
constexpr llvm::StringLiteral kIdentityIndexMap = "map(0)";

/// Text of one MemSpace enum value. The spellings are the pinned
/// TTCore_MemorySpace cases (`dram`, `l1`) from TTCoreOpsEnums.td.
llvm::StringRef memorySpaceText(MemSpace space) {
  if (space == MemSpace::L1) {
    return "l1";
  }
  return "dram";
}

} // namespace

// The pinned attribute form, quoted from
// test/ttmlir/Dialect/TTCore/metal_layout_attr_verify.mlir:
//
//   #ttcore.metal_layout<logical_shape = 32x32, dim_alignments = 32x32,
//       collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>,
//       undef, l1, sharded, index_map = map(0)>
//
// Field order and spellings come from the assemblyFormat in
// include/ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.td and the enum cases in
// TTCoreOpsEnums.td. Two values are derived the way the pinned builder derives
// them (lib/Dialect/TTCore/IR/TTCoreOpsTypes.cpp):
// * dim_alignments: 1 for the leading dims and 32 for the last two (the
//   pinned computeTileAlignments aligns the last two dims to the 32x32 tile).
// * collapsed_intervals: the leading rank-1 dims collapse into one interval and
//   the innermost dim stays alone (computeDefaultFlattenedIntervals), so rank 2
//   gives [[0, 1], [1, 2]] and rank 3 gives [[0, 2], [2, 3]].
//
// The verifier rejects a rank-1 logical_shape ("logical_shape must have at
// least 2 dimensions, got 1" in the fixture), so callers skip the encoding for
// rank-1 tensors.
void printMetalLayout(llvm::raw_ostream &os, RankedTensorType type,
                      MemSpace space) {
  ArrayRef<int64_t> shape = type.getShape();
  const int64_t rank = type.getRank();
  assert(rank >= kMinLayoutRank &&
         "the pinned metal_layout verifier rejects a rank-1 logical shape");

  // Align the two innermost dims to the tile; the leading dims stay at 1.
  SmallVector<int64_t> alignments(rank, 1);
  alignments[rank - 2] = kTileAlignment;
  alignments[rank - 1] = kTileAlignment;

  // One interval covers the leading dims, the last interval covers dim rank-1.
  const int64_t leadingIntervalEnd = rank - 1;

  os << "#ttcore.metal_layout<logical_shape = ";
  llvm::interleave(shape, os, "x");
  os << ", dim_alignments = ";
  llvm::interleave(alignments, os, "x");
  os << ", collapsed_intervals = dense<[[0, " << leadingIntervalEnd << "], ["
     << leadingIntervalEnd << ", " << rank << "]]> : tensor<2x2xi64>";
  os << ", " << kOobVal << ", " << memorySpaceText(space) << ", "
     << kMemoryLayout << ", index_map = " << kIdentityIndexMap << ">";
}

} // namespace tensixflow
} // namespace mlir
