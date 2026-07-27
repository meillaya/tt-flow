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
// TensixflowFusedRmsnormMatmulOp
//===----------------------------------------------------------------------===//

LogicalResult TensixflowFusedRmsnormMatmulOp::inferReturnTypeComponents(
    MLIRContext *context, std::optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes,
    OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  ShapeAdaptor inputShape = operands.getShape(0);
  ShapeAdaptor weightShape = operands.getShape(2);

  if (!inputShape.hasRank() || !weightShape.hasRank())
    return emitOptionalError(
        location, "expected ranked tensors for fused_rmsnorm_matmul operands");

  int64_t inputRank = inputShape.getRank();
  int64_t weightRank = weightShape.getRank();
  if (inputRank < 2)
    return emitOptionalError(location,
                             "expected input to have rank >= 2, got ",
                             inputRank);
  if (weightRank < 2)
    return emitOptionalError(location,
                             "expected weight to have rank >= 2, got ",
                             weightRank);

  if (inputShape.getElementType() != weightShape.getElementType())
    return emitOptionalError(
        location, "expected input and weight element types to match, got ",
        inputShape.getElementType(), " and ", weightShape.getElementType());

  // Result shape = matmul(rmsnorm(input), weight): the input's batch/M dims
  // (all but the normalized/contracting last dim) followed by weight's
  // trailing N dim: [input.dim(0)..input.dim(inputRank-2), weight.dim(N-1)].
  SmallVector<int64_t> resultShape;
  resultShape.reserve(inputRank);
  for (int64_t i = 0; i < inputRank - 1; ++i)
    resultShape.push_back(inputShape.getDimSize(i));
  resultShape.push_back(weightShape.getDimSize(weightRank - 1));

  inferredReturnShapes.emplace_back(resultShape, inputShape.getElementType());
  return success();
}

LogicalResult TensixflowFusedRmsnormMatmulOp::verify() {
  auto inputType = cast<RankedTensorType>(getInput().getType());
  auto gammaType = cast<RankedTensorType>(getGamma().getType());
  auto weightType = cast<RankedTensorType>(getWeight().getType());

  if (inputType.getRank() < 2)
    return emitOpError("expected input to have rank >= 2, got ")
           << inputType.getRank();
  if (weightType.getRank() < 2)
    return emitOpError("expected weight to have rank >= 2, got ")
           << weightType.getRank();
  if (gammaType.getRank() != 1)
    return emitOpError("gamma must be a 1-D tensor, got rank ")
           << gammaType.getRank();

  if (inputType.getElementType() != gammaType.getElementType() ||
      inputType.getElementType() != weightType.getElementType())
    return emitOpError(
               "expected input, gamma, and weight element types to match, got ")
           << inputType.getElementType() << ", " << gammaType.getElementType()
           << ", and " << weightType.getElementType();

  // Gamma normalizes the input's last dimension. Only check when both static.
  int64_t inputLastDim = inputType.getShape()[inputType.getRank() - 1];
  int64_t gammaSize = gammaType.getShape()[0];
  if (!ShapedType::isDynamic(inputLastDim) &&
      !ShapedType::isDynamic(gammaSize) && gammaSize != inputLastDim)
    return emitOpError("gamma size (")
           << gammaSize << ") must match the input's last dimension ("
           << inputLastDim << ")";

  // Contracting dimension: input's last dim (K) must equal weight's
  // second-to-last dim (K). Only check when both are static.
  int64_t weightK = weightType.getShape()[weightType.getRank() - 2];
  if (!ShapedType::isDynamic(inputLastDim) &&
      !ShapedType::isDynamic(weightK) && inputLastDim != weightK)
    return emitOpError("mismatched contracting dimensions: input K = ")
           << inputLastDim << " but weight K = " << weightK;

  return success();
}

//===----------------------------------------------------------------------===//
// TensixflowFusedMlpOp
//===----------------------------------------------------------------------===//

LogicalResult TensixflowFusedMlpOp::inferReturnTypeComponents(
    MLIRContext *context, std::optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes,
    OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  ShapeAdaptor inputShape = operands.getShape(0);
  ShapeAdaptor residualShape = operands.getShape(5);

  if (!inputShape.hasRank() || !residualShape.hasRank())
    return emitOptionalError(
        location, "expected ranked tensors for fused_mlp operands");

  if (inputShape.getRank() < 2)
    return emitOptionalError(location,
                             "expected input to have rank >= 2, got ",
                             inputShape.getRank());

  // The fused MLP output is the residual-added result, so it has the SAME
  // shape as the residual operand; element type follows the input.
  SmallVector<int64_t> resultShape;
  residualShape.getDims(resultShape);
  inferredReturnShapes.emplace_back(resultShape, inputShape.getElementType());
  return success();
}

LogicalResult TensixflowFusedMlpOp::verify() {
  auto inputType = cast<RankedTensorType>(getInput().getType());
  auto gammaType = cast<RankedTensorType>(getGamma().getType());
  auto w1Type = cast<RankedTensorType>(getW1().getType());
  auto b1Type = cast<RankedTensorType>(getB1().getType());
  auto w2Type = cast<RankedTensorType>(getW2().getType());
  auto residualType = cast<RankedTensorType>(getResidual().getType());

  if (inputType.getRank() < 2)
    return emitOpError("expected input to have rank >= 2, got ")
           << inputType.getRank();
  if (gammaType.getRank() != 1)
    return emitOpError("gamma must be a 1-D tensor, got rank ")
           << gammaType.getRank();
  if (w1Type.getRank() < 2)
    return emitOpError("expected w1 to have rank >= 2, got ")
           << w1Type.getRank();
  if (w2Type.getRank() < 2)
    return emitOpError("expected w2 to have rank >= 2, got ")
           << w2Type.getRank();
  if (b1Type.getRank() != 1)
    return emitOpError("b1 must be a 1-D tensor, got rank ")
           << b1Type.getRank();

  Type inputElt = inputType.getElementType();
  if (gammaType.getElementType() != inputElt ||
      w1Type.getElementType() != inputElt ||
      b1Type.getElementType() != inputElt ||
      w2Type.getElementType() != inputElt ||
      residualType.getElementType() != inputElt)
    return emitOpError(
        "expected input, gamma, w1, b1, w2, and residual element types to "
        "match, got ")
           << inputElt << ", " << gammaType.getElementType() << ", "
           << w1Type.getElementType() << ", " << b1Type.getElementType()
           << ", " << w2Type.getElementType() << ", and "
           << residualType.getElementType();

  int64_t inputLastDim = inputType.getShape()[inputType.getRank() - 1];

  // Gamma normalizes the input's last dimension.
  int64_t gammaSize = gammaType.getShape()[0];
  if (!ShapedType::isDynamic(inputLastDim) &&
      !ShapedType::isDynamic(gammaSize) && gammaSize != inputLastDim)
    return emitOpError("gamma size (")
           << gammaSize << ") must match the input's last dimension ("
           << inputLastDim << ")";

  // First matmul contracting dim: w1's second-to-last dim (K1) == input K.
  int64_t w1K = w1Type.getShape()[w1Type.getRank() - 2];
  if (!ShapedType::isDynamic(inputLastDim) && !ShapedType::isDynamic(w1K) &&
      w1K != inputLastDim)
    return emitOpError("mismatched contracting dimensions for w1: input K = ")
           << inputLastDim << " but w1 K = " << w1K;

  // Second matmul contracting dim: w2's second-to-last dim (K2) == w1's
  // output/hidden dim H (w1's last dim).
  int64_t w1H = w1Type.getShape()[w1Type.getRank() - 1];
  int64_t w2K = w2Type.getShape()[w2Type.getRank() - 2];
  if (!ShapedType::isDynamic(w1H) && !ShapedType::isDynamic(w2K) &&
      w2K != w1H)
    return emitOpError("mismatched contracting dimensions for w2: w1 hidden = ")
           << w1H << " but w2 K = " << w2K;

  // Bias broadcasts over the hidden dim H.
  int64_t b1Size = b1Type.getShape()[0];
  if (!ShapedType::isDynamic(w1H) && !ShapedType::isDynamic(b1Size) &&
      b1Size != w1H)
    return emitOpError("b1 size (")
           << b1Size << ") must match w1's output dimension (" << w1H << ")";

  // The residual add requires the residual's last dim (N) to match w2's
  // output dim, and its leading/batch dims to match the input's.
  int64_t w2N = w2Type.getShape()[w2Type.getRank() - 1];
  int64_t residualLastDim =
      residualType.getShape()[residualType.getRank() - 1];
  if (!ShapedType::isDynamic(w2N) && !ShapedType::isDynamic(residualLastDim) &&
      w2N != residualLastDim)
    return emitOpError("w2 output dimension (")
           << w2N << ") must match the residual's last dimension ("
           << residualLastDim << ")";

  if (residualType.getRank() != inputType.getRank())
    return emitOpError("expected residual rank (")
           << residualType.getRank() << ") to match input rank ("
           << inputType.getRank() << ")";

  for (int64_t i = 0, e = inputType.getRank() - 1; i < e; ++i) {
    int64_t residualDim = residualType.getShape()[i];
    int64_t inputDim = inputType.getShape()[i];
    if (!ShapedType::isDynamic(residualDim) &&
        !ShapedType::isDynamic(inputDim) && residualDim != inputDim)
      return emitOpError("residual batch dimension ")
             << i << " (" << residualDim
             << ") must match the input's batch dimension (" << inputDim
             << ")";
  }

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
