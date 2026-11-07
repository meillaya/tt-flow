//===- Compat/TTIRFuncTranslator.cpp - Per-func op -> TTIR ------*- C++ -*-===//
//
// TensixFlow: An MLIR Compiler for Tile-Native LLM Kernels on
// Tenstorrent-Style Spatial Accelerators.
//
//===----------------------------------------------------------------------===//
//
// The per-function op -> TTIR mapping of the tensixflow -> TTIR text emitter.
// Every emitted construct follows tt-mlir 0.9.0.dev20260221; the mapping
// contract and the pinned sources are documented on the
// --tensixflow-emit-ttir pass in EmitTTIR.cpp and in docs/ttir.md.
//
//===----------------------------------------------------------------------===//

#include "tensixflow/Compat/TTIRTextEmitter.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "tensixflow/TensixflowOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace tensixflow {

namespace {

// TTIR op names used by the mapping. They are pinned strings, not labels.
constexpr llvm::StringLiteral kTTIRAddOp = "ttir.add";
constexpr llvm::StringLiteral kTTIRMatmulOp = "ttir.matmul";
constexpr llvm::StringLiteral kTTIRRmsNormOp = "ttir.rms_norm";
constexpr llvm::StringLiteral kTTIRGeluOp = "ttir.gelu";
constexpr llvm::StringLiteral kTTIRConstantOp = "ttir.constant";

// A value-returning function has exactly one result in this dialect. The
// signature fallback needs that one result's type; zero results take the void
// path and emit no arrow.
constexpr size_t kSingleResultReturn = 1;

} // namespace

std::string TTIRFuncTranslator::translate(TensixflowFuncOp func,
                                          llvm::raw_ostream &bos) {
  values.clear();
  ssaCounter = 0;

  // Function arguments: %argN. Tensor arguments live in DRAM; a rank-1 operand
  // keeps a plain type (see renderTensorType). Non-tensor arguments keep their
  // plain type because they cannot feed a tensor op.
  SmallVector<std::string> argTexts;
  for (auto [i, arg] : llvm::enumerate(func.getArguments())) {
    std::string ssa = (llvm::Twine("%arg") + llvm::Twine(i)).str();
    std::string typeText;
    if (auto tensorType = dyn_cast<RankedTensorType>(arg.getType())) {
      typeText = emitter.renderTensorType(tensorType, MemSpace::DRAM);
    } else {
      llvm::raw_string_ostream typeStream(typeText);
      typeStream << arg.getType();
    }
    values[arg] = RenderedValue{ssa, typeText};
    argTexts.push_back(ssa + ": " + typeText);
  }

  std::string returnTypeText;
  for (Operation &op : func.getBody().front()) {
    dispatch(op, bos, returnTypeText);
  }

  std::string signature = (llvm::Twine("func.func @") + func.getName() + "(" +
                           llvm::join(argTexts, ", ") + ")")
                              .str();
  if (!returnTypeText.empty()) {
    signature += " -> " + returnTypeText;
  }
  signature += " {\n";
  return signature;
}

const RenderedValue *TTIRFuncTranslator::lookup(Value v) const {
  auto it = values.find(v);
  if (it == values.end()) {
    return nullptr;
  }
  return &it->second;
}

void TTIRFuncTranslator::markUnsupported(llvm::raw_ostream &bos,
                                         Operation &op, StringRef reason) {
  bos << "  // UNSUPPORTED: " << op.getName() << " " << reason << "\n";
}

RenderedValue
TTIRFuncTranslator::emitBinaryLine(llvm::raw_ostream &bos, StringRef ttirOp,
                                   const RenderedValue &lhs,
                                   const RenderedValue &rhs, StringRef attrs,
                                   StringRef resultType) {
  const RenderedValue *operands[2] = {&lhs, &rhs};
  const unsigned n = ssaCounter++;
  bos << "  %" << n << " = \"" << ttirOp << "\"(";
  llvm::interleaveComma(
      operands, bos, [&](const RenderedValue *value) { bos << value->ssa; });
  bos << ")";
  if (!attrs.empty()) {
    bos << " <{" << attrs << "}>";
  }
  bos << " : (";
  llvm::interleaveComma(
      operands, bos, [&](const RenderedValue *value) { bos << value->type; });
  bos << ") -> " << resultType << "\n";
  return RenderedValue{"%" + std::to_string(n), std::string(resultType)};
}

RenderedValue TTIRFuncTranslator::emitUnaryLine(
    llvm::raw_ostream &bos, StringRef ttirOp, const RenderedValue &operand,
    StringRef resultType) {
  const unsigned n = ssaCounter++;
  bos << "  %" << n << " = \"" << ttirOp << "\"(" << operand.ssa << ") : ("
      << operand.type << ") -> " << resultType << "\n";
  return RenderedValue{"%" + std::to_string(n), std::string(resultType)};
}

void TTIRFuncTranslator::emitBinary(llvm::raw_ostream &bos, StringRef ttirOp,
                                    Operation &owner, Value lhs, Value rhs,
                                    Value result) {
  const RenderedValue *left = lookup(lhs);
  const RenderedValue *right = lookup(rhs);
  auto resultType = dyn_cast<RankedTensorType>(result.getType());
  if (left == nullptr || right == nullptr) {
    markUnsupported(bos, owner,
                    "has no TTIR mapping (operand is unsupported)");
    return;
  }
  if (!resultType || !resultType.hasStaticShape()) {
    markUnsupported(bos, owner,
                    "has no TTIR mapping (dynamic result shape)");
    return;
  }
  values[result] =
      emitBinaryLine(bos, ttirOp, *left, *right, /*attrs=*/"",
                     emitter.renderTensorType(resultType, MemSpace::L1));
}

void TTIRFuncTranslator::emitUnary(llvm::raw_ostream &bos, StringRef ttirOp,
                                   Operation &owner, Value input, Value result) {
  const RenderedValue *operand = lookup(input);
  auto resultType = dyn_cast<RankedTensorType>(result.getType());
  if (operand == nullptr) {
    markUnsupported(bos, owner,
                    "has no TTIR mapping (operand is unsupported)");
    return;
  }
  if (!resultType || !resultType.hasStaticShape()) {
    markUnsupported(bos, owner,
                    "has no TTIR mapping (dynamic result shape)");
    return;
  }
  values[result] = emitUnaryLine(
      bos, ttirOp, *operand, emitter.renderTensorType(resultType, MemSpace::L1));
}

std::optional<RenderedValue>
TTIRFuncTranslator::emitRmsNorm(llvm::raw_ostream &bos, Operation &owner,
                                Value input, Value gamma, FloatAttr epsilon) {
  const RenderedValue *in = lookup(input);
  const RenderedValue *g = lookup(gamma);
  auto inType = dyn_cast<RankedTensorType>(input.getType());
  auto gammaType = dyn_cast<RankedTensorType>(gamma.getType());
  if (in == nullptr || g == nullptr) {
    markUnsupported(bos, owner,
                    "has no TTIR mapping (operand is unsupported)");
    return std::nullopt;
  }
  if (!inType || !gammaType || !inType.hasStaticShape() ||
      !gammaType.hasStaticShape()) {
    markUnsupported(bos, owner, "has no TTIR mapping (dynamic shape)");
    return std::nullopt;
  }
  // Pinned generic form, quoted from
  // test/ttmlir/Dialect/TTIR/ttir_rms_norm_negative.mlir:
  //
  //   "ttir.rms_norm"(%input, %weight) <{normalized_shape = array<i64: 8>,
  //       epsilon = 1.000000e-05 : f32,
  //       operandSegmentSizes = array<i32: 1, 1, 0>}>
  //       : (tensor<2x4x8xf32>, tensor<8xf32>) -> tensor<2x4x8xf32>
  //
  // The op carries AttrSizedOperandSegments at the pin, so operandSegmentSizes
  // is part of the text. The tensixflow operand list is input and gamma, with
  // no bias, which is the 1, 1, 0 segment split.
  std::string attrs;
  llvm::raw_string_ostream attrStream(attrs);
  attrStream << "epsilon = ";
  epsilon.print(attrStream);
  attrStream << ", normalized_shape = array<i64: " << inType.getShape().back()
             << ">, operandSegmentSizes = array<i32: 1, 1, 0>";
  // The rmsnorm result has the same shape as the input (dialect contract).
  return emitBinaryLine(bos, kTTIRRmsNormOp, *in, *g, attrs,
                        emitter.renderTensorType(inType, MemSpace::L1));
}

void TTIRFuncTranslator::emitConstant(llvm::raw_ostream &bos,
                                      TensixflowConstantOp op) {
  auto type = dyn_cast<RankedTensorType>(op.getResult().getType());
  if (!type || !type.hasStaticShape()) {
    markUnsupported(bos, *op,
                    "has no TTIR mapping (dynamic or non-tensor constant)");
    return;
  }
  // The value attr prints verbatim with its own (plain) type and the result
  // keeps that same type: ttir.constant's contract is value-attr-type ==
  // result-type, and layout aliases attach to func boundaries + compute-op
  // results. The pinned negative fixture (creation/constant) fixes the value
  // attribute to dense elements, which the source constant op already holds.
  std::string typeText;
  llvm::raw_string_ostream typeStream(typeText);
  typeStream << type;
  const unsigned n = ssaCounter++;
  bos << "  %" << n << " = \"" << kTTIRConstantOp << "\"() <{value = ";
  op.getValue().print(bos);
  bos << "}> : () -> " << typeText << "\n";
  values[op.getResult()] = RenderedValue{"%" + std::to_string(n), typeText};
}

void TTIRFuncTranslator::emitFusedRmsnormMatmul(
    llvm::raw_ostream &bos, TensixflowFusedRmsnormMatmulOp op) {
  std::optional<RenderedValue> normed = emitRmsNorm(
      bos, *op, op.getInput(), op.getGamma(), op.getEpsilonAttr());
  if (!normed) {
    return; // The marker is already emitted.
  }
  const RenderedValue *weight = lookup(op.getWeight());
  auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
  if (weight == nullptr || !resultType || !resultType.hasStaticShape()) {
    markUnsupported(bos, *op,
                    "has no TTIR mapping (operand is unsupported)");
    return;
  }
  values[op.getResult()] =
      emitBinaryLine(bos, kTTIRMatmulOp, *normed, *weight, /*attrs=*/"",
                     emitter.renderTensorType(resultType, MemSpace::L1));
}

void TTIRFuncTranslator::emitFusedMlp(llvm::raw_ostream &bos,
                                      TensixflowFusedMlpOp op) {
  // The mapped prefix is rms_norm -> matmul w1 -> broadcast bias add -> gelu.
  // The tail (matmul w2, residual add) has no TTIR mapping yet, so the fused
  // op's result stays unbound and the return marks transitively.
  std::optional<RenderedValue> normed = emitRmsNorm(
      bos, *op, op.getInput(), op.getGamma(), op.getEpsilonAttr());
  if (!normed) {
    return;
  }
  const RenderedValue *w1 = lookup(op.getW1());
  const RenderedValue *b1 = lookup(op.getB1());
  auto w1Type = dyn_cast<RankedTensorType>(op.getW1().getType());
  auto inType = dyn_cast<RankedTensorType>(op.getInput().getType());
  if (!w1 || !b1 || !w1Type || !inType || !w1Type.hasStaticShape() ||
      !inType.hasStaticShape()) {
    markUnsupported(bos, *op,
                    "has no TTIR mapping (operand is unsupported)");
    return;
  }
  // matmul(w1) result shape: input.shape[:-1] + [w1.shape.back()].
  SmallVector<int64_t> hiddenShape(inType.getShape().drop_back());
  hiddenShape.push_back(w1Type.getShape().back());
  auto hiddenType = RankedTensorType::get(hiddenShape, inType.getElementType());
  RenderedValue hidden =
      emitBinaryLine(bos, kTTIRMatmulOp, *normed, *w1, /*attrs=*/"",
                     emitter.renderTensorType(hiddenType, MemSpace::L1));
  // A rank-1 bias is legal here: TTIR elementwise ops are broadcastable at the
  // pin (fixture test/ttmlir/Dialect/TTIR/fusing/matmul_with_bias_positive.mlir
  // adds tensor<68x1024xf32> and tensor<1024xf32> into tensor<68x1024xf32>).
  RenderedValue biased =
      emitBinaryLine(bos, kTTIRAddOp, hidden, *b1, /*attrs=*/"", hidden.type);
  // The fused gelu stage maps to the same unary op as tensixflow.gelu.
  emitUnaryLine(bos, kTTIRGeluOp, biased, biased.type);
  // The tail consumes the gelu result, but the projection and residual add have
  // no TTIR mapping here.
  bos << "  // UNSUPPORTED: tensixflow.fused_mlp tail (matmul w2, residual "
         "add) has no TTIR mapping\n";
}

void TTIRFuncTranslator::emitReturn(TensixflowReturnOp op,
                                    llvm::raw_ostream &bos,
                                    std::string &returnTypeText) {
  const RenderedValue *rendered =
      op.getOperands().empty() ? nullptr : lookup(op.getOperands().front());
  if (rendered == nullptr) {
    if (!op.getOperands().empty()) {
      markUnsupported(bos, *op,
                      "has no TTIR mapping (operand is unsupported)");
    }
    // Signature fallback: the source func's result type, written back to DRAM
    // (the writer-kernel convention) when it is a tensor. A void func leaves
    // returnTypeText empty, and translate then omits the arrow.
    FunctionType fnType =
        op->getParentOfType<TensixflowFuncOp>().getFunctionType();
    if (fnType.getNumResults() != kSingleResultReturn) {
      return;
    }
    Type sourceResult = fnType.getResult(0);
    if (auto tensorType = dyn_cast<RankedTensorType>(sourceResult)) {
      returnTypeText = emitter.renderTensorType(tensorType, MemSpace::DRAM);
    } else {
      llvm::raw_string_ostream typeStream(returnTypeText);
      typeStream << sourceResult;
    }
    return;
  }
  bos << "  return " << rendered->ssa << " : " << rendered->type << "\n";
  returnTypeText = rendered->type;
}

void TTIRFuncTranslator::dispatch(Operation &op, llvm::raw_ostream &bos,
                                  std::string &returnTypeText) {
  if (auto ret = dyn_cast<TensixflowReturnOp>(op)) {
    return emitReturn(ret, bos, returnTypeText);
  }
  if (auto add = dyn_cast<TensixflowAddOp>(op)) {
    return emitBinary(bos, kTTIRAddOp, *add, add.getLhs(), add.getRhs(),
                      add.getResult());
  }
  if (auto matmul = dyn_cast<TensixflowMatmulOp>(op)) {
    return emitBinary(bos, kTTIRMatmulOp, *matmul, matmul.getLhs(),
                      matmul.getRhs(), matmul.getResult());
  }
  if (auto rmsnorm = dyn_cast<TensixflowRmsnormOp>(op)) {
    if (auto rendered = emitRmsNorm(bos, *rmsnorm, rmsnorm.getInput(),
                                    rmsnorm.getGamma(),
                                    rmsnorm.getEpsilonAttr())) {
      values[rmsnorm.getResult()] = *rendered;
    }
    return;
  }
  if (auto gelu = dyn_cast<TensixflowGeluOp>(op)) {
    return emitUnary(bos, kTTIRGeluOp, *gelu, gelu.getInput(),
                     gelu.getResult());
  }
  if (auto constant = dyn_cast<TensixflowConstantOp>(op)) {
    return emitConstant(bos, constant);
  }
  if (auto fused = dyn_cast<TensixflowFusedRmsnormMatmulOp>(op)) {
    return emitFusedRmsnormMatmul(bos, fused);
  }
  if (auto mlp = dyn_cast<TensixflowFusedMlpOp>(op)) {
    return emitFusedMlp(bos, mlp);
  }
  // Any op without a pinned TTIR mapping (for example a schedule op) emits the
  // documented marker instead of crashing.
  markUnsupported(bos, op, "has no TTIR mapping");
}

} // namespace tensixflow
} // namespace mlir
