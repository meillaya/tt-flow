//===- TensixflowOps.cpp - Tensixflow dialect ops ---------------*- C++ -*-===//
//
// TensixFlow: An MLIR Compiler for Tile-Native LLM Kernels on
// Tenstorrent-Style Spatial Accelerators.
//
//===----------------------------------------------------------------------===//

#include "tensixflow/TensixflowOps.h"
#include "tensixflow/TensixflowDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
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

OpFoldResult TensixflowConstantOp::fold(FoldAdaptor) { return getValue(); }

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

//===----------------------------------------------------------------------===//
// TensixflowAddOp canonicalization
//===----------------------------------------------------------------------===//

namespace {

/// Returns true if `value` is a splat-zero constant tensor (the additive
/// identity). Matches any ConstantLike op (e.g. tensixflow.constant) via
/// m_Constant, then requires a splat DenseElementsAttr of float element type
/// (bf16/fp32) whose splat APFloat isZero(). Non-float element types are
/// rejected so getSplatValue<APFloat> is never called on an integer attr.
static bool isZeroConstant(Value value) {
  Attribute attr;
  if (!matchPattern(value, m_Constant(&attr)))
    return false;
  auto dense = dyn_cast_or_null<DenseElementsAttr>(attr);
  if (!dense || !dense.isSplat())
    return false;
  if (!isa<FloatType>(dense.getElementType()))
    return false;
  return dense.getSplatValue<APFloat>().isZero();
}

/// add %x, %zero -> %x  and the symmetric  add %zero, %x -> %x, where the zero
/// operand is a splat-zero constant tensor. If neither operand is a zero
/// constant the pattern does nothing (returns failure). Deliberately the ONLY
/// add canonicalization: more aggressive folds (e.g. add %x,%x -> scale) are
/// withheld so they cannot defeat the Wave-4 fusion matcher.
struct AddIdentityZero final : OpRewritePattern<TensixflowAddOp> {
  using OpRewritePattern<TensixflowAddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TensixflowAddOp op,
                                PatternRewriter &rewriter) const final {
    if (isZeroConstant(op.getRhs())) {
      rewriter.replaceOp(op, op.getLhs());
      return success();
    }
    if (isZeroConstant(op.getLhs())) {
      rewriter.replaceOp(op, op.getRhs());
      return success();
    }
    return failure();
  }
};

} // namespace

void TensixflowAddOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                  MLIRContext *ctx) {
  patterns.add<AddIdentityZero>(ctx);
}

#define GET_OP_CLASSES
#include "tensixflow/TensixflowOps.cpp.inc"
