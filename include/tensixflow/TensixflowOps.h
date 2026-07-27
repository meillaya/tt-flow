//===- TensixflowOps.h - Tensixflow dialect ops -----------------*- C++ -*-===//
//
// TensixFlow: An MLIR Compiler for Tile-Native LLM Kernels on
// Tenstorrent-Style Spatial Accelerators.
//
//===----------------------------------------------------------------------===//

#ifndef TENSIXFLOW_TENSIXFLOWOPS_H
#define TENSIXFLOW_TENSIXFLOWOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "tensixflow/TensixflowOps.h.inc"

#endif // TENSIXFLOW_TENSIXFLOWOPS_H
