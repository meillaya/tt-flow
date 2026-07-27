//===- TensixflowScheduleOps.cpp - Tile-level schedule IR -------*- C++ -*-===//
//
// TensixFlow: An MLIR Compiler for Tile-Native LLM Kernels on
// Tenstorrent-Style Spatial Accelerators.
//
//===----------------------------------------------------------------------===//

#include "tensixflow/TensixflowSchedule.h"
#include "tensixflow/TensixflowDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::tensixflow;

//===----------------------------------------------------------------------===//
// TensixflowSchedTileLoopOp
//===----------------------------------------------------------------------===//

LogicalResult TensixflowSchedTileLoopOp::verify() {
  int64_t tripCount = getTripCount();
  if (tripCount < 0)
    return emitOpError("trip count must be non-negative, got ") << tripCount;
  return success();
}

//===----------------------------------------------------------------------===//
// Generated op definitions (must share a TU with the complete dialect/op
// storage; the dialect registers these via addOperations<> in
// TensixflowDialect.cpp).
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "tensixflow/TensixflowSchedule.cpp.inc"
