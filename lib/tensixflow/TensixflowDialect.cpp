//===- TensixflowDialect.cpp - Tensixflow dialect ---------------*- C++ -*-===//
//
// TensixFlow: An MLIR Compiler for Tile-Native LLM Kernels on
// Tenstorrent-Style Spatial Accelerators.
//
//===----------------------------------------------------------------------===//

#include "tensixflow/TensixflowDialect.h"
#include "tensixflow/TensixflowOps.h"

using namespace mlir;
using namespace mlir::tensixflow;

#include "tensixflow/TensixflowOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Tensixflow dialect.
//===----------------------------------------------------------------------===//

void TensixflowDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "tensixflow/TensixflowOps.cpp.inc"
      >();
}
