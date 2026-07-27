//===- TensixflowOps.cpp - Tensixflow dialect ops ---------------*- C++ -*-===//
//
// TensixFlow: An MLIR Compiler for Tile-Native LLM Kernels on
// Tenstorrent-Style Spatial Accelerators.
//
//===----------------------------------------------------------------------===//

#include "tensixflow/TensixflowOps.h"
#include "tensixflow/TensixflowDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

using namespace mlir;
using namespace mlir::tensixflow;

//===----------------------------------------------------------------------===//
// TensixflowFuncOp
//===----------------------------------------------------------------------===//

ParseResult TensixflowFuncOp::parse(OpAsmParser &parser,
                                    OperationState &result) {
  auto buildFuncType =
      [](Builder &builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
         function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };

  return function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false,
      getFunctionTypeAttrName(result.name), buildFuncType,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void TensixflowFuncOp::print(OpAsmPrinter &p) {
  function_interface_impl::printFunctionOp(
      p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
      getArgAttrsAttrName(), getResAttrsAttrName());
}

//===----------------------------------------------------------------------===//
// TensixflowConstantOp
//===----------------------------------------------------------------------===//

ParseResult TensixflowConstantOp::parse(OpAsmParser &parser,
                                        OperationState &result) {
  TypedAttr value;
  if (parser.parseAttribute(value, "value", result.attributes))
    return failure();
  result.addTypes(value.getType());
  return success();
}

void TensixflowConstantOp::print(OpAsmPrinter &p) {
  p << ' ' << getValue();
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"value"});
}

//===----------------------------------------------------------------------===//
// TensixflowMatmulOp
//===----------------------------------------------------------------------===//

LogicalResult TensixflowMatmulOp::inferReturnTypeComponents(
    MLIRContext *context, std::optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes,
    OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  ShapeAdaptor lhsShape = operands.getShape(0);
  ShapeAdaptor rhsShape = operands.getShape(1);

  if (!lhsShape.hasRank() || !rhsShape.hasRank())
    return emitOptionalError(location,
                             "expected ranked tensors for matmul operands");

  int64_t lhsRank = lhsShape.getRank();
  int64_t rhsRank = rhsShape.getRank();
  if (lhsRank < 2)
    return emitOptionalError(location, "expected lhs to have rank >= 2, got ",
                             lhsRank);
  if (rhsRank < 2)
    return emitOptionalError(location, "expected rhs to have rank >= 2, got ",
                             rhsRank);

  if (lhsShape.getElementType() != rhsShape.getElementType())
    return emitOptionalError(
        location, "expected lhs and rhs element types to match, got ",
        lhsShape.getElementType(), " and ", rhsShape.getElementType());

  // Result shape = lhs batch/M dims (all but the contracting K) followed by
  // rhs's trailing N dim: [lhs.dim(0)..lhs.dim(lhsRank-2), rhs.dim(rhsRank-1)].
  SmallVector<int64_t> resultShape;
  resultShape.reserve(lhsRank);
  for (int64_t i = 0; i < lhsRank - 1; ++i)
    resultShape.push_back(lhsShape.getDimSize(i));
  resultShape.push_back(rhsShape.getDimSize(rhsRank - 1));

  inferredReturnShapes.emplace_back(resultShape, lhsShape.getElementType());
  return success();
}

LogicalResult TensixflowMatmulOp::verify() {
  auto lhsType = cast<RankedTensorType>(getLhs().getType());
  auto rhsType = cast<RankedTensorType>(getRhs().getType());

  if (lhsType.getRank() < 2)
    return emitOpError("expected lhs to have rank >= 2, got ")
           << lhsType.getRank();
  if (rhsType.getRank() < 2)
    return emitOpError("expected rhs to have rank >= 2, got ")
           << rhsType.getRank();

  if (lhsType.getElementType() != rhsType.getElementType())
    return emitOpError("expected lhs and rhs element types to match, got ")
           << lhsType.getElementType() << " and " << rhsType.getElementType();

  // Contracting dimension: lhs's last dim (K) must equal rhs's second-to-last
  // dim (K). Only check when both are static.
  int64_t lhsK = lhsType.getShape()[lhsType.getRank() - 1];
  int64_t rhsK = rhsType.getShape()[rhsType.getRank() - 2];
  if (!ShapedType::isDynamic(lhsK) && !ShapedType::isDynamic(rhsK) &&
      lhsK != rhsK)
    return emitOpError("mismatched contracting dimensions: lhs K = ")
           << lhsK << " but rhs K = " << rhsK;

  return success();
}

//===----------------------------------------------------------------------===//
// TensixflowRmsnormOp
//===----------------------------------------------------------------------===//

LogicalResult TensixflowRmsnormOp::inferReturnTypeComponents(
    MLIRContext *context, std::optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes,
    OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  ShapeAdaptor inputShape = operands.getShape(0);

  if (!inputShape.hasRank())
    return emitOptionalError(location,
                             "expected a ranked tensor for rmsnorm input");

  if (inputShape.getRank() < 1)
    return emitOptionalError(location,
                             "expected rmsnorm input to have rank >= 1, got ",
                             inputShape.getRank());

  // RMSNorm normalizes over the last dimension; the result has the same shape
  // and element type as the input.
  SmallVector<int64_t> resultShape;
  inputShape.getDims(resultShape);
  inferredReturnShapes.emplace_back(resultShape, inputShape.getElementType());
  return success();
}

LogicalResult TensixflowRmsnormOp::verify() {
  auto inputType = cast<RankedTensorType>(getInput().getType());
  auto gammaType = cast<RankedTensorType>(getGamma().getType());

  if (inputType.getRank() < 1)
    return emitOpError("expected input to have rank >= 1, got ")
           << inputType.getRank();

  if (gammaType.getRank() != 1)
    return emitOpError("gamma must be a 1-D tensor, got rank ")
           << gammaType.getRank();

  if (inputType.getElementType() != gammaType.getElementType())
    return emitOpError("expected input and gamma element types to match, got ")
           << inputType.getElementType() << " and "
           << gammaType.getElementType();

  // Gamma size must equal the input's last dimension. Only check when both are
  // static.
  int64_t inputLastDim = inputType.getShape()[inputType.getRank() - 1];
  int64_t gammaSize = gammaType.getShape()[0];
  if (!ShapedType::isDynamic(inputLastDim) &&
      !ShapedType::isDynamic(gammaSize) && gammaSize != inputLastDim)
    return emitOpError("gamma size (")
           << gammaSize << ") must match the input's last dimension ("
           << inputLastDim << ")";

  return success();
}

#define GET_OP_CLASSES
#include "tensixflow/TensixflowOps.cpp.inc"
