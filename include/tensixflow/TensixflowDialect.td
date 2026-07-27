//===- TensixflowDialect.td - Tensixflow dialect -----------*- tablegen -*-===//
//
// TensixFlow: An MLIR Compiler for Tile-Native LLM Kernels on
// Tenstorrent-Style Spatial Accelerators.
//
//===----------------------------------------------------------------------===//

#ifndef TENSIXFLOW_DIALECT
#define TENSIXFLOW_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Tensixflow dialect definition.
//===----------------------------------------------------------------------===//

def Tensixflow_Dialect : Dialect {
  let name = "tensixflow";
  let summary = "A tile-native dialect for LLM kernels on spatial accelerators.";
  let description = [{
    The `tensixflow` dialect models tile-level operations for Tenstorrent-style
    spatial accelerators, lowering fused transformer operators into NoC-aware,
    circular-buffered reader/compute/writer kernels.
  }];
  let cppNamespace = "::mlir::tensixflow";
}

#endif // TENSIXFLOW_DIALECT
