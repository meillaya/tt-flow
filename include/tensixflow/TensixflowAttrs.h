//===- TensixflowAttrs.h - Tensixflow attributes ----------------*- C++ -*-===//
//
// TensixFlow: An MLIR Compiler for Tile-Native LLM Kernels on
// Tenstorrent-Style Spatial Accelerators.
//
//===----------------------------------------------------------------------===//

#ifndef TENSIXFLOW_TENSIXFLOWATTRS_H
#define TENSIXFLOW_TENSIXFLOWATTRS_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h"

#include "tensixflow/TensixflowEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "tensixflow/TensixflowAttrs.h.inc"

#endif // TENSIXFLOW_TENSIXFLOWATTRS_H
