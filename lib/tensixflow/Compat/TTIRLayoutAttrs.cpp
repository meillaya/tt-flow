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
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace tensixflow {

// The study-grounded attribute form (the test/ttmlir/Dialect/D2M/
// lower_to_layout.mlir syntax quoted in docs/ttmlir_study.md):
//
//   #ttcore.metal_layout<logical_shape = ..., dim_alignments = ...,
//       collapsed_intervals = ..., l1|dram, interleaved>
//
// Conventions, each grounded in a real tt-mlir @9f06802f test file:
// * dim_alignments: 32 for dims >= 32, else 1 (the interleaved-layout
//   convention of TTIRToD2M/ttnn_dram_interleaved_translation.mlir:
//   32x32 for 1024x1024, 1x32x32 for rank 3, 1x1x32x32 for rank 4).
// * collapsed_intervals: the uncollapsed identity. Rank 2 spells it
//   (dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, the file's own rank-2 form);
//   rank >= 3 uses the empty form (dense<> : tensor<0x2xi64>, the file's
//   rank-3/4 form); rank 1 uses the documented rank-1 analogue.
// * `interleaved` (not `sharded`): this emitter translates pre-sharding
//   source ops; sharding is PlanNocSharding's concern. The flag itself is
//   the study's metal_layout vocabulary.
void printMetalLayout(llvm::raw_ostream &os, RankedTensorType type,
                      MemSpace space) {
  ArrayRef<int64_t> dims = type.getShape();
  SmallVector<int64_t> alignments;
  for (int64_t dim : dims)
    alignments.push_back(dim >= 32 ? 32 : 1);
  os << "#ttcore.metal_layout<logical_shape = ";
  llvm::interleave(dims, os, "x");
  os << ", dim_alignments = ";
  llvm::interleave(alignments, os, "x");
  os << ", collapsed_intervals = ";
  if (type.getRank() == 1)
    os << "dense<[[0, 1]]> : tensor<1x2xi64>";
  else if (type.getRank() == 2)
    os << "dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>";
  else
    os << "dense<> : tensor<0x2xi64>";
  os << ", " << (space == MemSpace::L1 ? "l1" : "dram") << ", interleaved>";
}

} // namespace tensixflow
} // namespace mlir
