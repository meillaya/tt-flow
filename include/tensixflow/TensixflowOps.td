//===- TensixflowOps.td - Tensixflow dialect ops -----------*- tablegen -*-===//
//
// TensixFlow: An MLIR Compiler for Tile-Native LLM Kernels on
// Tenstorrent-Style Spatial Accelerators.
//
//===----------------------------------------------------------------------===//

#ifndef TENSIXFLOW_OPS
#define TENSIXFLOW_OPS

include "tensixflow/TensixflowDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Base tensixflow operation definition.
//===----------------------------------------------------------------------===//

class Tensixflow_Op<string mnemonic, list<Trait> traits = []> :
    Op<Tensixflow_Dialect, mnemonic, traits>;

// Operations are defined in todos 9-11 (func/return/constant/add/gelu/matmul/
// rmsnorm). The GET_OP_LIST machinery generates correctly with an empty op set.

#endif // TENSIXFLOW_OPS
