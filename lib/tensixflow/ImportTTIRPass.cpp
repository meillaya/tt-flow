//===- ImportTTIRPass.cpp - Import bare-tensor TTIR into tensixflow ------===//
//
// TensixFlow: An MLIR Compiler for Tile-Native LLM Kernels on
// Tenstorrent-Style Spatial Accelerators.
//
//===----------------------------------------------------------------------===//
//
// The --tensixflow-import-ttir pass: rebuild a module written in the documented
// bare-tensor TTIR subset (docs/ttir.md) as tensixflow IR, so the planning
// pipeline runs unchanged.
//
// This build does not link the ttir or ttcore dialects, so the input is parsed
// as unregistered generic operations under --allow-unregistered-dialect. The
// pass therefore matches operations by name and reads their attributes by
// string key. Layout-encoded tensor types and ttcore or ttnn attributes are
// rejected rather than stripped: the build cannot interpret them, and guessing
// at a layout would change the program.
//
// Accepted subset: exactly one func.func with one block and one result, bare
// static ranked tensor types (bf16 or f32), generic-form ttir.matmul,
// ttir.add, ttir.gelu, and ttir.rms_norm ops, and func.return. Every source op
// becomes the matching tensixflow op. The ttir.rms_norm normalized_shape must
// name the last dimension, and epsilon maps onto the tensixflow epsilon
// attribute. Everything else fails with a diagnostic that names the construct
// and points at docs/ttir.md.
//
//===----------------------------------------------------------------------===//

#include "tensixflow/TensixflowPasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "tensixflow/TensixflowDialect.h"
#include "tensixflow/TensixflowOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

#include <cassert>

namespace mlir {
namespace tensixflow {

// Emit impl::ImportTTIRPassBase<DerivedT> (in mlir::tensixflow::impl) and the
// public createImportTTIRPass() definition for this one pass only.
#define GEN_PASS_DEF_IMPORTTTIRPASS
#include "tensixflow/TensixflowPasses.h.inc"

namespace {

// Pinned tt-mlir names. The source ops are unregistered in this build, so the
// importer matches them by name instead of by C++ class.
constexpr StringLiteral kFuncFuncOp = "func.func";
constexpr StringLiteral kFuncReturnOp = "func.return";
constexpr StringLiteral kTTIRMatmulOp = "ttir.matmul";
constexpr StringLiteral kTTIRRmsNormOp = "ttir.rms_norm";
constexpr StringLiteral kTTIRGeluOp = "ttir.gelu";
constexpr StringLiteral kTTIRAddOp = "ttir.add";

// Attribute keys of the pinned generic forms.
constexpr StringLiteral kTransposeAAttr = "transpose_a";
constexpr StringLiteral kTransposeBAttr = "transpose_b";
constexpr StringLiteral kNormalizedShapeAttr = "normalized_shape";
constexpr StringLiteral kEpsilonAttr = "epsilon";
constexpr StringLiteral kOperandSegmentSizesAttr = "operandSegmentSizes";

// Builtin attributes the function interface stores on a func.func.
constexpr StringLiteral kSymNameAttr = "sym_name";
constexpr StringLiteral kSymVisibilityAttr = "sym_visibility";
constexpr StringLiteral kFunctionTypeAttr = "function_type";
constexpr StringLiteral kArgAttrsAttr = "arg_attrs";
constexpr StringLiteral kResAttrsAttr = "res_attrs";

// Tensor ranks and counts the subset fixes. The planning pipeline works on
// 2-D tiles, and the only rank-1 tensor it consumes is the rms_norm weight,
// which tensixflow.rmsnorm requires. The matmul weight is a rank-2 matrix.
constexpr int64_t kMinTensorRank = 2;
constexpr int64_t kWeightRank = 1;
constexpr int64_t kMatmulLhsRank = 2;
constexpr int64_t kMatmulRhsRank = 2;
constexpr int64_t kUnaryOperandCount = 1;
constexpr int64_t kBinaryOperandCount = 2;
constexpr int64_t kSingleResultCount = 1;
constexpr int64_t kSingleFunctionCount = 1;

// Default epsilon of TTIR_RMSNormOp at the pinned release. The importer writes
// the epsilon attribute explicitly so the tensixflow default (1e-6) never
// applies silently.
constexpr float kTTIRDefaultEpsilon = 1.0e-5f;

/// True when the subset accepts the element type. bf16 is the pipeline's
/// working type; f32 is accepted because the pinned TTIR uses it for math and
/// for some constants.
bool isSupportedElementType(Type type) {
  if (type.isBF16()) {
    return true;
  }
  return type.isF32();
}

/// Validate the parts of a tensor type that every imported value must satisfy:
/// ranked, bare (no layout encoding), static, and bf16 or f32. Returns null
/// after emitting a diagnostic that names the construct.
RankedTensorType checkedBareTensor(Type type, Location loc, StringRef role) {
  auto ranked = dyn_cast<RankedTensorType>(type);
  if (!ranked) {
    emitError(loc) << role << " is not a ranked tensor: " << type
                   << "; the import subset accepts ranked tensors only "
                      "(docs/ttir.md)";
    return nullptr;
  }
  // A layout encoding is a ttcore or ttnn attribute. This build does not link
  // those dialects, so the importer cannot read the encoding and refuses to
  // pretend it does.
  if (ranked.getEncoding()) {
    emitError(loc) << role << " is layout-encoded: " << type
                   << "; this build does not link the ttcore or ttnn dialects, "
                      "so the importer accepts bare tensors and does not strip "
                      "layouts (docs/ttir.md)";
    return nullptr;
  }
  if (!ranked.hasStaticShape()) {
    emitError(loc) << role << " has a dynamic shape: " << type
                   << "; the import subset requires static shapes "
                      "(docs/ttir.md)";
    return nullptr;
  }
  if (!isSupportedElementType(ranked.getElementType())) {
    emitError(loc) << role << " has element type " << ranked.getElementType()
                   << "; the import subset accepts bf16 and f32 only "
                      "(docs/ttir.md)";
    return nullptr;
  }
  return ranked;
}

/// Require the rank the planning pipeline needs for a value that is not the
/// rms_norm weight. Rank-1 values reach this check when they appear anywhere
/// else, which is the documented limit.
bool requirePlanRank(Location loc, StringRef role, RankedTensorType type) {
  if (type.getRank() == 0) {
    emitError(loc) << role << " is a rank-0 tensor: " << type
                   << "; the import subset requires rank 2 or higher here "
                      "(docs/ttir.md)";
    return false;
  }
  if (type.getRank() < kMinTensorRank) {
    emitError(loc) << role << " has rank " << type.getRank() << ": " << type
                   << "; rank-1 tensors are outside the import subset except "
                      "the ttir.rms_norm weight (docs/ttir.md)";
    return false;
  }
  return true;
}

/// The rms_norm weight is the one rank-1 tensor the subset accepts, because
/// tensixflow.rmsnorm takes a per-column gamma vector.
bool requireWeightRank(Location loc, RankedTensorType type) {
  if (type.getRank() == kWeightRank) {
    return true;
  }
  emitError(loc) << "ttir.rms_norm weight has rank " << type.getRank() << ": "
                 << type
                 << "; the import subset requires the rank-1 weight operand "
                    "(docs/ttir.md)";
  return false;
}

/// Reject attributes on an imported op that the subset does not define. The
/// source ops are unregistered, so nothing else checks their attributes.
LogicalResult checkAllowedAttributes(Operation &op,
                                     ArrayRef<StringRef> allowed) {
  for (const NamedAttribute &attr : op.getAttrs()) {
    if (llvm::is_contained(allowed, attr.getName().strref())) {
      continue;
    }
    op.emitError() << "attribute '" << attr.getName().strref() << "' on '"
                   << op.getName()
                   << "' is outside the import subset (docs/ttir.md)";
    return failure();
  }
  return success();
}

/// The pinned ttir.matmul has default-false transpose attributes. The importer
/// accepts the absent or false form; a true transpose needs an op the
/// tensixflow dialect does not have.
LogicalResult checkDisabledTranspose(Operation &op, StringRef attrName) {
  Attribute attr = op.getAttr(attrName);
  if (!attr) {
    return success();
  }
  auto boolean = dyn_cast<BoolAttr>(attr);
  if (!boolean) {
    op.emitError() << "attribute '" << attrName << "' on '" << op.getName()
                   << "' must be a boolean (docs/ttir.md)";
    return failure();
  }
  if (!boolean.getValue()) {
    return success();
  }
  op.emitError() << "'" << op.getName() << "' attribute '" << attrName
                 << " = true' is outside the import subset; only untransposed "
                    "operands are imported (docs/ttir.md)";
  return failure();
}

/// Look up the tensixflow value an imported operand was mapped to. The walk is
/// in definition order, so every operand is mapped before its use.
Value mappedOperand(const DenseMap<Value, Value> &values, Operation &op,
                    unsigned index) {
  Value source = op.getOperand(index);
  assert(values.contains(source) && "operand must be mapped before its use");
  return values.lookup(source);
}

LogicalResult importReturn(Operation &op, OpBuilder &builder,
                           const DenseMap<Value, Value> &values,
                           Type functionResultType) {
  if (op.getNumOperands() != kSingleResultCount) {
    op.emitError() << "'func.return' must return exactly one tensor in the "
                      "import subset (docs/ttir.md)";
    return failure();
  }
  Type operandType = op.getOperand(0).getType();
  if (operandType != functionResultType) {
    op.emitError() << "'func.return' operand type " << operandType
                   << " does not match the function result type "
                   << functionResultType << " (docs/ttir.md)";
    return failure();
  }
  builder.create<TensixflowReturnOp>(op.getLoc(),
                                     ValueRange{mappedOperand(values, op, 0)});
  return success();
}

LogicalResult importMatmul(Operation &op, OpBuilder &builder,
                           DenseMap<Value, Value> &values) {
  if (failed(checkAllowedAttributes(op, {kTransposeAAttr, kTransposeBAttr}))) {
    return failure();
  }
  if (failed(checkDisabledTranspose(op, kTransposeAAttr)) ||
      failed(checkDisabledTranspose(op, kTransposeBAttr))) {
    return failure();
  }
  if (op.getNumOperands() != kBinaryOperandCount ||
      op.getNumResults() != kSingleResultCount) {
    op.emitError() << "'" << op.getName() << "' must have two operands and "
                      "one result (docs/ttir.md)";
    return failure();
  }

  RankedTensorType lhsType = checkedBareTensor(
      op.getOperand(0).getType(), op.getLoc(), "'ttir.matmul' lhs");
  if (!lhsType) {
    return failure();
  }
  if (!requirePlanRank(op.getLoc(), "'ttir.matmul' lhs", lhsType)) {
    return failure();
  }
  RankedTensorType rhsType = checkedBareTensor(
      op.getOperand(1).getType(), op.getLoc(), "'ttir.matmul' rhs");
  if (!rhsType) {
    return failure();
  }
  if (rhsType.getRank() != kMatmulRhsRank) {
    op.emitError() << "'ttir.matmul' rhs has rank " << rhsType.getRank() << ": "
                   << rhsType
                   << "; the import subset accepts a rank-2 weight operand "
                      "only (docs/ttir.md)";
    return failure();
  }
  if (lhsType.getElementType() != rhsType.getElementType()) {
    op.emitError() << "'ttir.matmul' operands have different element types: "
                   << lhsType.getElementType() << " and "
                   << rhsType.getElementType() << " (docs/ttir.md)";
    return failure();
  }
  const int64_t lhsContraction = lhsType.getShape().back();
  const int64_t rhsContraction = rhsType.getShape()[0];
  if (lhsContraction != rhsContraction) {
    op.emitError() << "'ttir.matmul' contracting dimensions do not match: lhs "
                   << "K = " << lhsContraction << " but rhs K = "
                   << rhsContraction << " (docs/ttir.md)";
    return failure();
  }

  RankedTensorType resultType = checkedBareTensor(
      op.getResult(0).getType(), op.getLoc(), "'ttir.matmul' result");
  if (!resultType) {
    return failure();
  }
  // The tensixflow matmul takes the batch and M dims from lhs and N from rhs,
  // so the declared TTIR result type must be exactly that.
  SmallVector<int64_t> expectedShape(lhsType.getShape().begin(),
                                     lhsType.getShape().end() - 1);
  expectedShape.push_back(rhsType.getShape().back());
  auto expectedType =
      RankedTensorType::get(expectedShape, lhsType.getElementType());
  if (resultType != expectedType) {
    op.emitError() << "'ttir.matmul' result type " << resultType
                   << " does not match the imported result type "
                   << expectedType << " (docs/ttir.md)";
    return failure();
  }

  auto matmul = builder.create<TensixflowMatmulOp>(
      op.getLoc(), resultType, mappedOperand(values, op, 0),
      mappedOperand(values, op, 1));
  values[op.getResult(0)] = matmul.getResult();
  return success();
}

LogicalResult importAdd(Operation &op, OpBuilder &builder,
                        DenseMap<Value, Value> &values) {
  if (failed(checkAllowedAttributes(op, {}))) {
    return failure();
  }
  if (op.getNumOperands() != kBinaryOperandCount ||
      op.getNumResults() != kSingleResultCount) {
    op.emitError() << "'" << op.getName() << "' must have two operands and "
                      "one result (docs/ttir.md)";
    return failure();
  }
  RankedTensorType lhsType = checkedBareTensor(
      op.getOperand(0).getType(), op.getLoc(), "'ttir.add' lhs");
  if (!lhsType) {
    return failure();
  }
  RankedTensorType rhsType = checkedBareTensor(
      op.getOperand(1).getType(), op.getLoc(), "'ttir.add' rhs");
  if (!rhsType) {
    return failure();
  }
  RankedTensorType resultType =
      checkedBareTensor(op.getResult(0).getType(), op.getLoc(), "'ttir.add' "
                                                               "result");
  if (!resultType) {
    return failure();
  }
  if (!requirePlanRank(op.getLoc(), "'ttir.add' operands", lhsType) ||
      !requirePlanRank(op.getLoc(), "'ttir.add' operands", rhsType) ||
      !requirePlanRank(op.getLoc(), "'ttir.add' result", resultType)) {
    return failure();
  }
  // tensixflow.add is elementwise on identical types; the pinned ttir.add is
  // broadcastable, so a broadcast form would need an explicit broadcast op.
  if (lhsType != rhsType || lhsType != resultType) {
    op.emitError() << "'ttir.add' operands and result must have the same "
                      "type; broadcast operands are outside the import subset "
                      "(docs/ttir.md)";
    return failure();
  }
  auto sum = builder.create<TensixflowAddOp>(
      op.getLoc(), resultType, mappedOperand(values, op, 0),
      mappedOperand(values, op, 1));
  values[op.getResult(0)] = sum.getResult();
  return success();
}

LogicalResult importGelu(Operation &op, OpBuilder &builder,
                         DenseMap<Value, Value> &values) {
  if (failed(checkAllowedAttributes(op, {}))) {
    return failure();
  }
  if (op.getNumOperands() != kUnaryOperandCount ||
      op.getNumResults() != kSingleResultCount) {
    op.emitError() << "'" << op.getName() << "' must have one operand and one "
                      "result (docs/ttir.md)";
    return failure();
  }
  RankedTensorType inputType = checkedBareTensor(
      op.getOperand(0).getType(), op.getLoc(), "'ttir.gelu' input");
  if (!inputType) {
    return failure();
  }
  RankedTensorType resultType = checkedBareTensor(
      op.getResult(0).getType(), op.getLoc(), "'ttir.gelu' result");
  if (!resultType) {
    return failure();
  }
  if (!requirePlanRank(op.getLoc(), "'ttir.gelu' input", inputType) ||
      !requirePlanRank(op.getLoc(), "'ttir.gelu' result", resultType)) {
    return failure();
  }
  if (inputType != resultType) {
    op.emitError() << "'ttir.gelu' operand and result must have the same type "
                      "(docs/ttir.md)";
    return failure();
  }
  auto activation = builder.create<TensixflowGeluOp>(
      op.getLoc(), resultType, mappedOperand(values, op, 0));
  values[op.getResult(0)] = activation.getResult();
  return success();
}

LogicalResult importRmsNorm(Operation &op, OpBuilder &builder,
                            DenseMap<Value, Value> &values,
                            DenseSet<Value> &weightValues) {
  if (failed(checkAllowedAttributes(
          op, {kNormalizedShapeAttr, kEpsilonAttr, kOperandSegmentSizesAttr}))) {
    return failure();
  }
  // tensixflow.rmsnorm has no bias operand, and the subset imports the weight
  // form only because the pipeline needs the gamma vector.
  if (op.getNumOperands() == kBinaryOperandCount + 1) {
    op.emitError() << "'" << op.getName()
                   << "' with a bias operand is outside the import subset; "
                      "only the input and weight operands are imported "
                      "(docs/ttir.md)";
    return failure();
  }
  if (op.getNumOperands() != kBinaryOperandCount ||
      op.getNumResults() != kSingleResultCount) {
    op.emitError() << "'" << op.getName() << "' must have an input operand, a "
                      "weight operand, and one result (docs/ttir.md)";
    return failure();
  }

  RankedTensorType inputType = checkedBareTensor(
      op.getOperand(0).getType(), op.getLoc(), "'ttir.rms_norm' input");
  if (!inputType) {
    return failure();
  }
  if (!requirePlanRank(op.getLoc(), "'ttir.rms_norm' input", inputType)) {
    return failure();
  }
  RankedTensorType weightType = checkedBareTensor(
      op.getOperand(1).getType(), op.getLoc(), "'ttir.rms_norm' weight");
  if (!weightType) {
    return failure();
  }
  if (!requireWeightRank(op.getLoc(), weightType)) {
    return failure();
  }
  RankedTensorType resultType = checkedBareTensor(
      op.getResult(0).getType(), op.getLoc(), "'ttir.rms_norm' result");
  if (!resultType) {
    return failure();
  }
  if (resultType != inputType) {
    op.emitError() << "'ttir.rms_norm' result type " << resultType
                   << " must equal the input type " << inputType
                   << " (docs/ttir.md)";
    return failure();
  }
  if (weightType.getElementType() != inputType.getElementType()) {
    op.emitError() << "'ttir.rms_norm' input and weight element types differ: "
                   << inputType.getElementType() << " and "
                   << weightType.getElementType() << " (docs/ttir.md)";
    return failure();
  }
  const int64_t normalizedSize = inputType.getShape().back();
  if (weightType.getShape()[0] != normalizedSize) {
    op.emitError() << "'ttir.rms_norm' weight size ("
                   << weightType.getShape()[0]
                   << ") must match the input's last dimension ("
                   << normalizedSize << ") (docs/ttir.md)";
    return failure();
  }

  // tensixflow.rmsnorm normalizes the last dimension, so the pinned
  // normalized_shape can only be the single last-dimension entry.
  Attribute normalizedShapeAttr = op.getAttr(kNormalizedShapeAttr);
  if (!normalizedShapeAttr) {
    op.emitError() << "'" << op.getName()
                   << "' is missing the required normalized_shape attribute "
                      "(docs/ttir.md)";
    return failure();
  }
  auto normalizedShape = dyn_cast<DenseI64ArrayAttr>(normalizedShapeAttr);
  if (!normalizedShape) {
    op.emitError() << "'ttir.rms_norm' normalized_shape must be an i64 array: "
                   << normalizedShapeAttr << " (docs/ttir.md)";
    return failure();
  }
  ArrayRef<int64_t> normalizedValues = normalizedShape.asArrayRef();
  if (normalizedValues.size() != 1 ||
      normalizedValues.front() != normalizedSize) {
    op.emitError() << "'ttir.rms_norm' normalized_shape "
                   << normalizedShapeAttr
                   << " is outside the import subset; tensixflow normalizes "
                      "the last dimension ("
                   << normalizedSize << ") only (docs/ttir.md)";
    return failure();
  }

  // epsilon is optional in TTIR with a 1e-5 default. The importer always sets
  // the tensixflow attribute so the source value survives.
  FloatAttr epsilon = builder.getF32FloatAttr(kTTIRDefaultEpsilon);
  if (Attribute epsilonAttr = op.getAttr(kEpsilonAttr)) {
    epsilon = dyn_cast<FloatAttr>(epsilonAttr);
    if (!epsilon) {
      op.emitError() << "'ttir.rms_norm' epsilon must be a float attribute: "
                     << epsilonAttr << " (docs/ttir.md)";
      return failure();
    }
    if (!epsilon.getType().isF32()) {
      op.emitError() << "'ttir.rms_norm' epsilon must be f32, got "
                     << epsilon.getType() << " (docs/ttir.md)";
      return failure();
    }
  }

  weightValues.insert(op.getOperand(1));
  auto rmsnorm = builder.create<TensixflowRmsnormOp>(
      op.getLoc(), resultType, mappedOperand(values, op, 0),
      mappedOperand(values, op, 1), epsilon);
  values[op.getResult(0)] = rmsnorm.getResult();
  return success();
}

LogicalResult importOperation(Operation &op, OpBuilder &builder,
                              DenseMap<Value, Value> &values,
                              DenseSet<Value> &weightValues,
                              Type functionResultType) {
  StringRef name = op.getName().getStringRef();
  if (name == kFuncReturnOp) {
    return importReturn(op, builder, values, functionResultType);
  }
  if (name == kTTIRMatmulOp) {
    return importMatmul(op, builder, values);
  }
  if (name == kTTIRAddOp) {
    return importAdd(op, builder, values);
  }
  if (name == kTTIRGeluOp) {
    return importGelu(op, builder, values);
  }
  if (name == kTTIRRmsNormOp) {
    return importRmsNorm(op, builder, values, weightValues);
  }
  // A ttir or ttnn op outside the four mapped ops names itself. Any other
  // operation does the same with the generic wording.
  if (name.starts_with("ttir.") || name.starts_with("ttnn.")) {
    op.emitError() << "operation '" << name
                   << "' is outside the import subset; the importer accepts "
                      "ttir.matmul, ttir.add, ttir.gelu, and ttir.rms_norm "
                      "(docs/ttir.md)";
    return failure();
  }
  op.emitError() << "operation '" << name
                 << "' is outside the import subset; only func.func, "
                    "func.return, and the four ttir compute ops are imported "
                    "(docs/ttir.md)";
  return failure();
}

/// Reject module attributes from unlinked dialects before any rewrite. A
/// ttcore or ttnn device attribute would otherwise ride into the output
/// unread.
LogicalResult checkModuleAttributes(ModuleOp module) {
  for (const NamedAttribute &attr : module->getAttrs()) {
    StringRef name = attr.getName().strref();
    if (name == kSymNameAttr || name == kSymVisibilityAttr) {
      continue;
    }
    module.emitError() << "module attribute '" << attr.getName().strref()
                       << "' is outside the import subset; device and layout "
                          "attributes from the ttcore or ttnn dialects are "
                          "rejected because this build does not link them "
                          "(docs/ttir.md)";
    return failure();
  }
  return success();
}

/// The subset is one function per module. The pass returns a diagnostic that
/// names the first offending top-level operation or the function count.
FailureOr<func::FuncOp> findOnlyFunction(ModuleOp module) {
  SmallVector<func::FuncOp> functions;
  for (Operation &op : module.getBody()->getOperations()) {
    if (auto candidate = dyn_cast<func::FuncOp>(op)) {
      functions.push_back(candidate);
      continue;
    }
    op.emitError() << "operation '" << op.getName()
                   << "' is outside the import subset; the module may contain "
                      "exactly one func.func (docs/ttir.md)";
    return failure();
  }
  if (functions.size() != static_cast<size_t>(kSingleFunctionCount)) {
    module.emitError() << "expected exactly one func.func in the module, found "
                       << functions.size()
                       << "; multiple functions are outside the import subset "
                          "(docs/ttir.md)";
    return failure();
  }
  return functions.front();
}

/// Validate the source signature and function-level attributes. The source
/// arguments are allowed to be rank 1 here because the rms_norm weight is a
/// rank-1 argument; the body walk checks that any rank-1 argument is used as
/// that weight.
LogicalResult checkFunctionSignature(func::FuncOp function) {
  for (const NamedAttribute &attr : function->getAttrs()) {
    StringRef name = attr.getName().strref();
    if (name == kSymNameAttr || name == kSymVisibilityAttr ||
        name == kFunctionTypeAttr) {
      continue;
    }
    // Per-argument and per-result attributes would carry shard or layout
    // metadata from a dialect this build does not link.
    if (name == kArgAttrsAttr || name == kResAttrsAttr) {
      auto dictionaries = dyn_cast<ArrayAttr>(attr.getValue());
      if (dictionaries && dictionaries.empty()) {
        continue;
      }
      function.emitError() << "function argument or result attributes are "
                              "outside the import subset; the importer accepts "
                              "bare tensor arguments (docs/ttir.md)";
      return failure();
    }
    function.emitError() << "function attribute '" << attr.getName().strref()
                         << "' is outside the import subset (docs/ttir.md)";
    return failure();
  }

  FunctionType type = function.getFunctionType();
  for (auto [index, input] : llvm::enumerate(type.getInputs())) {
    std::string role =
        ("function argument " + llvm::Twine(index)).str();
    if (!checkedBareTensor(input, function.getLoc(), role)) {
      return failure();
    }
  }
  if (type.getNumResults() != kSingleResultCount) {
    function.emitError() << "'func.func' must have exactly one result in the "
                            "import subset, got "
                         << type.getNumResults() << " (docs/ttir.md)";
    return failure();
  }
  RankedTensorType resultType = checkedBareTensor(
      type.getResult(0), function.getLoc(), "function result");
  if (!resultType) {
    return failure();
  }
  if (!requirePlanRank(function.getLoc(), "function result", resultType)) {
    return failure();
  }
  if (function.isExternal()) {
    function.emitError() << "external function declarations are outside the "
                            "import subset (docs/ttir.md)";
    return failure();
  }
  if (!llvm::hasSingleElement(function.getBody())) {
    function.emitError() << "'func.func' must have a single block in the "
                            "import subset (docs/ttir.md)";
    return failure();
  }
  return success();
}

/// Validate the module and rebuild its single function as tensixflow IR.
LogicalResult importFunction(ModuleOp module, func::FuncOp source) {
  if (failed(checkFunctionSignature(source))) {
    return failure();
  }

  OpBuilder builder(module.getContext());
  builder.setInsertionPoint(source);
  auto imported = builder.create<TensixflowFuncOp>(
      source.getLoc(), source.getName(), source.getFunctionType(), ArrayAttr(),
      ArrayAttr());
  Block &entry = imported.getBody().emplaceBlock();
  for (Type input : source.getFunctionType().getInputs()) {
    entry.addArgument(input, source.getLoc());
  }

  // The body builds from the entry point in definition order, so every operand
  // is mapped before its use.
  OpBuilder bodyBuilder = OpBuilder::atBlockBegin(&entry);
  DenseMap<Value, Value> values;
  for (auto [sourceArg, importedArg] :
       llvm::zip(source.getArguments(), imported.getArguments())) {
    values[sourceArg] = importedArg;
  }
  DenseSet<Value> weightValues;
  Type resultType = source.getFunctionType().getResult(0);
  for (Operation &op : source.getBody().front().getOperations()) {
    if (failed(
            importOperation(op, bodyBuilder, values, weightValues, resultType))) {
      imported.erase();
      return failure();
    }
  }

  // A rank-1 argument is accepted only when the body uses it as the rms_norm
  // weight. Anything else would put a rank-1 tensor into the planning pipeline.
  for (auto [index, argument] : llvm::enumerate(source.getArguments())) {
    auto argumentType = dyn_cast<RankedTensorType>(argument.getType());
    if (!argumentType || argumentType.getRank() != kWeightRank) {
      continue;
    }
    if (weightValues.contains(argument)) {
      continue;
    }
    emitError(source.getLoc())
        << "function argument " << index << " has rank 1: " << argumentType
        << "; rank-1 tensors are outside the import subset except the "
           "ttir.rms_norm weight (docs/ttir.md)";
    imported.erase();
    return failure();
  }

  source.erase();
  return success();
}

class ImportTTIRPass final
    : public impl::ImportTTIRPassBase<ImportTTIRPass> {
public:
  using Base = impl::ImportTTIRPassBase<ImportTTIRPass>;
  using Base::Base;

  void runOnOperation() final {
    ModuleOp module = getOperation();
    // The source module contains no tensixflow ops, so the dialect has not been
    // loaded when this pass starts. Load it before the pass builds its ops.
    module.getContext()->getOrLoadDialect<TensixflowDialect>();
    if (failed(checkModuleAttributes(module))) {
      signalPassFailure();
      return;
    }
    FailureOr<func::FuncOp> source = findOnlyFunction(module);
    if (failed(source)) {
      signalPassFailure();
      return;
    }
    if (failed(importFunction(module, *source))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace
} // namespace tensixflow
} // namespace mlir
