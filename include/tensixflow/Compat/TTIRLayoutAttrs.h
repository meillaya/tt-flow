//===- Compat/TTIRLayoutAttrs.h - TTIR layout attr vocabulary ---*- C++ -*-===//
//
// TensixFlow: An MLIR Compiler for Tile-Native LLM Kernels on
// Tenstorrent-Style Spatial Accelerators.
//
//===----------------------------------------------------------------------===//
//
// The `#ttcore.metal_layout` text vocabulary the TTIR emitter attaches to
// tensor types. The emitted form follows tt-mlir 0.9.0.dev20260221 (the
// released wheel pinned in deps/tt-mlir.lock); the quoted fixture and the
// reasons for each field live on printMetalLayout in TTIRLayoutAttrs.cpp.
//
//===----------------------------------------------------------------------===//

#ifndef TENSIXFLOW_COMPAT_TTIRLAYOUTATTRS_H
#define TENSIXFLOW_COMPAT_TTIRLAYOUTATTRS_H

#include <cstdint>

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace mlir {
class RankedTensorType;

namespace tensixflow {

/// Smallest logical rank the pinned `#ttcore.metal_layout` verifier accepts.
/// A rank-1 logical shape is rejected at parse time, so rank-1 tensors keep a
/// plain tensor type.
constexpr int64_t kMinLayoutRank = 2;

/// Memory space of an emitted layout: function inputs stream from DRAM, op
/// results live in L1 (the InferTileLayoutPass convention).
enum class MemSpace { L1, DRAM };

/// Print the pinned `#ttcore.metal_layout<...>` attribute text for a tensor
/// type in a memory space. The caller must pass a type whose rank is at least
/// kMinLayoutRank.
void printMetalLayout(llvm::raw_ostream &os, RankedTensorType type,
                      MemSpace space);

} // namespace tensixflow
} // namespace mlir

#endif // TENSIXFLOW_COMPAT_TTIRLAYOUTATTRS_H
