//===- TensixflowOps.td - Tensixflow dialect ops -----------*- tablegen -*-===//
//
// TensixFlow: An MLIR Compiler for Tile-Native LLM Kernels on
// Tenstorrent-Style Spatial Accelerators.
//
//===----------------------------------------------------------------------===//

#ifndef TENSIXFLOW_OPS
#define TENSIXFLOW_OPS

include "tensixflow/TensixflowDialect.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Base tensixflow operation definition.
//===----------------------------------------------------------------------===//

class Tensixflow_Op<string mnemonic, list<Trait> traits = []> :
    Op<Tensixflow_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// TensixflowFuncOp
//===----------------------------------------------------------------------===//

def TensixflowFuncOp : Tensixflow_Op<"func", [
    IsolatedFromAbove, FunctionOpInterface, Symbol,
    SingleBlockImplicitTerminator<"TensixflowReturnOp">]> {
  let summary = "A tensixflow function operation";
  let description = [{
    The `tensixflow.func` operation is a container for tensixflow computations.
    It is isolated from above and has a single-block body terminated by
    `tensixflow.return`.

    Example:

    ```mlir
    tensixflow.func @add(%x: tensor<2048x4096xbf16>, %y: tensor<2048x4096xbf16>)
        -> tensor<2048x4096xbf16> {
      %0 = tensixflow.add %x, %y : tensor<2048x4096xbf16>
      tensixflow.return %0 : tensor<2048x4096xbf16>
    }
    ```
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<FunctionType>:$function_type,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs);
  let regions = (region AnyRegion:$body);

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the region on the current operation that is callable.
    ::mlir::Region *getCallableRegion() {
      return isExternal() ? nullptr : &getBody();
    }

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }
  }];
}

//===----------------------------------------------------------------------===//
// TensixflowReturnOp
//===----------------------------------------------------------------------===//

def TensixflowReturnOp : Tensixflow_Op<"return", [Pure, ReturnLike, Terminator]> {
  let summary = "Function return operation";
  let description = [{
    The `tensixflow.return` operation terminates a `tensixflow.func` body.

    Example:

    ```mlir
    tensixflow.return %0 : tensor<2048x4096xbf16>
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

//===----------------------------------------------------------------------===//
// TensixflowConstantOp
//===----------------------------------------------------------------------===//

def TensixflowConstantOp : Tensixflow_Op<"constant", [Pure, ConstantLike]> {
  let summary = "Constant operation";
  let description = [{
    The `tensixflow.constant` operation produces an SSA value from a typed
    attribute.

    Example:

    ```mlir
    %0 = tensixflow.constant dense<1.0> : tensor<2048x4096xbf16>
    ```
  }];

  let arguments = (ins TypedAttrInterface:$value);
  let results = (outs AnyType:$result);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 0;

  let builders = [
    OpBuilder<(ins "Attribute":$value), [{
      build($_builder, $_state, cast<TypedAttr>(value).getType(),
            cast<TypedAttr>(value));
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// TensixflowAddOp
//===----------------------------------------------------------------------===//

def TensixflowAddOp : Tensixflow_Op<"add", [
    Pure, SameOperandsAndResultShape, SameOperandsAndResultType]> {
  let summary = "Elementwise add operation";
  let description = [{
    The `tensixflow.add` operation computes the elementwise sum of two tensors
    with the same shape.

    Example:

    ```mlir
    %0 = tensixflow.add %x, %y : tensor<2048x4096xbf16>
    ```
  }];

  let arguments = (ins AnyRankedTensor:$lhs, AnyRankedTensor:$rhs);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs)";
}

//===----------------------------------------------------------------------===//
// TensixflowGeluOp
//===----------------------------------------------------------------------===//

def TensixflowGeluOp : Tensixflow_Op<"gelu", [
    Pure, SameOperandsAndResultShape, SameOperandsAndResultType]> {
  let summary = "Elementwise GELU/SiLU activation";
  let description = [{
    The `tensixflow.gelu` operation applies the GELU (or SiLU) activation
    function elementwise to a tensor.

    Example:

    ```mlir
    %0 = tensixflow.gelu %x : tensor<2048x4096xbf16>
    ```
  }];

  let arguments = (ins AnyRankedTensor:$input);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$input attr-dict `:` type($input)";
}

//===----------------------------------------------------------------------===//
// TensixflowMatmulOp
//===----------------------------------------------------------------------===//

def TensixflowMatmulOp : Tensixflow_Op<"matmul", [
    Pure,
    DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
                              ["inferReturnTypeComponents"]>]> {
  let summary = "Matrix multiplication operation";
  let description = [{
    The `tensixflow.matmul` operation computes a matrix product. For
    `lhs` of shape `[..., M, K]` and `rhs` of shape `[K, N]`, the result has
    shape `[..., M, N]` (batch dims taken from `lhs`).

    Example:

    ```mlir
    %0 = tensixflow.matmul %x, %w
        : tensor<1x2048x4096xbf16> x tensor<4096x12288xbf16>
        -> tensor<1x2048x12288xbf16>
    ```
  }];

  let arguments = (ins AnyRankedTensor:$lhs, AnyRankedTensor:$rhs);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `x` type($rhs) `->` type($result)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// TensixflowRmsnormOp
//===----------------------------------------------------------------------===//

def TensixflowRmsnormOp : Tensixflow_Op<"rmsnorm", [
    Pure,
    DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
                              ["inferReturnTypeComponents"]>]> {
  let summary = "RMS normalization operation";
  let description = [{
    The `tensixflow.rmsnorm` operation normalizes `input` over its last
    dimension (RMSNorm): `y = input * rsqrt(mean(input^2, last_dim) + epsilon) * gamma`.
    `gamma` is a 1-D tensor whose size equals the last dimension of `input`.
    The result has the SAME shape as `input`.
  }];
  let arguments = (ins AnyRankedTensor:$input,
                       AnyRankedTensor:$gamma,
                       DefaultValuedAttr<F32Attr, "1.0e-6">:$epsilon);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$input `,` $gamma attr-dict `:` type($input) `,` type($gamma) `->` type($result)";
  let hasVerifier = 1;
}

#endif // TENSIXFLOW_OPS
