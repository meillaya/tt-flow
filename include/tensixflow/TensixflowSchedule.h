//===- TensixflowSchedule.h - Tile-level schedule IR ------------*- C++ -*-===//
//
// TensixFlow: An MLIR Compiler for Tile-Native LLM Kernels on
// Tenstorrent-Style Spatial Accelerators.
//
//===----------------------------------------------------------------------===//

#ifndef TENSIXFLOW_TENSIXFLOWSCHEDULE_H
#define TENSIXFLOW_TENSIXFLOWSCHEDULE_H

#include "tensixflow/TensixflowDialect.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"

#define GET_OP_CLASSES
#include "tensixflow/TensixflowSchedule.h.inc"

#endif // TENSIXFLOW_TENSIXFLOWSCHEDULE_H
