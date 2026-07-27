//===- TensixflowPasses.h - Tensixflow passes -------------------*- C++ -*-===//
//
// TensixFlow: An MLIR Compiler for Tile-Native LLM Kernels on
// Tenstorrent-Style Spatial Accelerators.
//
//===----------------------------------------------------------------------===//
//
// Public pass interface for the tensixflow dialect: the generated per-pass
// `createXxx()` declarations plus `mlir::tensixflow::registerPasses()`, which
// registers all seven tensixflow passes with the global MLIR pass registry.
// Mirrors mlir/examples/standalone's StandalonePasses.h (GEN_PASS_DECL /
// GEN_PASS_REGISTRATION pattern).
//
//===----------------------------------------------------------------------===//

#ifndef TENSIXFLOW_PASSES_H
#define TENSIXFLOW_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace tensixflow {

#define GEN_PASS_DECL
#include "tensixflow/TensixflowPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "tensixflow/TensixflowPasses.h.inc"

} // namespace tensixflow
} // namespace mlir

#endif // TENSIXFLOW_PASSES_H
