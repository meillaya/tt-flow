//===- Compat/TTIRLayoutAttrs.h - TTIR layout attr vocabulary ---*- C++ -*-===//
//
// TensixFlow: An MLIR Compiler for Tile-Native LLM Kernels on
// Tenstorrent-Style Spatial Accelerators.
//
//===----------------------------------------------------------------------===//
//
// The `#ttcore.metal_layout` text vocabulary the TTIR emitter (todo 51)
// attaches to tensor types, grounded in docs/ttmlir_study.md (tt-mlir
// @9f06802f): logical_shape, tile-sized dim_alignments, uncollapsed
// collapsed_intervals, the l1/dram memory space, and `interleaved`. The
// exact emitted form and its upstream sources are documented on
// printMetalLayout's definition in TTIRLayoutAttrs.cpp.
//
//===----------------------------------------------------------------------===//

#ifndef TENSIXFLOW_COMPAT_TTIRLAYOUTATTRS_H
#define TENSIXFLOW_COMPAT_TTIRLAYOUTATTRS_H

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace mlir {
class RankedTensorType;

namespace tensixflow {

/// Memory space of an emitted layout: function inputs stream from DRAM, op
/// results live in L1 (the InferTileLayoutPass convention).
enum class MemSpace { L1, DRAM };

/// Print the study-grounded `#ttcore.metal_layout<...>` attribute text for a
/// tensor type in a memory space.
void printMetalLayout(llvm::raw_ostream &os, RankedTensorType type,
                      MemSpace space);

} // namespace tensixflow
} // namespace mlir

#endif // TENSIXFLOW_COMPAT_TTIRLAYOUTATTRS_H
