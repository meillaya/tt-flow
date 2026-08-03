//===- Compat/TTIRFuncTranslator.cpp - Per-func op -> TTIR ------*- C++ -*-===//
//
// TensixFlow: An MLIR Compiler for Tile-Native LLM Kernels on
// Tenstorrent-Style Spatial Accelerators.
//
//===----------------------------------------------------------------------===//
//
// The per-function op -> TTIR mapping of the tensixflow -> TTIR text
// emitter. The mapping contract (which op maps to what, the rms_norm and
// gelu choices, the UNSUPPORTED convention, D21's TTIR-only scope) is
// documented on the --tensixflow-emit-ttir pass in EmitTTIR.cpp.
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

std::string TTIRFuncTranslator::translate(TensixflowFuncOp func,
                                          llvm::raw_ostream &bos) {
  values.clear();
  ssaCounter = 0;

  // Function arguments: %argN, inputs live in DRAM. Non-tensor arguments
  // keep their plain type (they cannot feed tensor ops; a consumer would
  // hit the UNSUPPORTED path via its operand lookup).
  SmallVector<std::string> argTexts;
  for (auto [i, arg] : llvm::enumerate(func.getArguments())) {
    std::string ssa = (llvm::Twine("%arg") + llvm::Twine(i)).str();
    std::string typeText;
    if (auto t = dyn_cast<RankedTensorType>(arg.getType()))
      typeText = emitter.renderTensorType(t, MemSpace::DRAM);
    else {
      llvm::raw_string_ostream ts(typeText);
      ts << arg.getType();
    }
    values[arg] = RenderedValue{ssa, typeText};
    argTexts.push_back(ssa + ": " + typeText);
  }

  std::string returnTypeText;
  for (Operation &op : func.getBody().front())
    dispatch(op, bos, returnTypeText);

  std::string signature = (llvm::Twine("func.func @") + func.getName() +
                           "(" + llvm::join(argTexts, ", ") + ")")
                              .str();
  if (!returnTypeText.empty())
    signature += " -> " + returnTypeText;
  signature += " {\n";
  return signature;
}

const RenderedValue *TTIRFuncTranslator::lookup(Value v) const {
  auto it = values.find(v);
  return it == values.end() ? nullptr : &it->second;
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
  unsigned n = ssaCounter++;
  bos << "  %" << n << " = \"" << ttirOp << "\"(";
  llvm::interleaveComma(
      operands, bos, [&](const RenderedValue *v) { bos << v->ssa; });
  bos << ")";
  if (!attrs.empty())
    bos << " <{" << attrs << "}>";
  bos << " : (";
  llvm::interleaveComma(
      operands, bos, [&](const RenderedValue *v) { bos << v->type; });
  bos << ") -> " << resultType << "\n";
  return RenderedValue{"%" + std::to_string(n), std::string(resultType)};
}

void TTIRFuncTranslator::emitBinary(llvm::raw_ostream &bos, StringRef ttirOp,
                                    Operation &owner, Value lhs, Value rhs,
                                    Value result) {
  const RenderedValue *l = lookup(lhs);
  const RenderedValue *r = lookup(rhs);
  auto resultType = dyn_cast<RankedTensorType>(result.getType());
  if (!l || !r) {
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
      emitBinaryLine(bos, ttirOp, *l, *r, /*attrs=*/"",
                     emitter.renderTensorType(resultType, MemSpace::L1));
}

std::optional<RenderedValue>
TTIRFuncTranslator::emitRmsNorm(llvm::raw_ostream &bos, Operation &owner,
                                Value input, Value gamma, FloatAttr epsilon) {
  const RenderedValue *in = lookup(input);
  const RenderedValue *g = lookup(gamma);
  auto inType = dyn_cast<RankedTensorType>(input.getType());
  auto gammaType = dyn_cast<RankedTensorType>(gamma.getType());
  if (!in || !g) {
    markUnsupported(bos, owner,
                    "has no TTIR mapping (operand is unsupported)");
    return std::nullopt;
  }
  if (!inType || !gammaType || !inType.hasStaticShape() ||
      !gammaType.hasStaticShape()) {
    markUnsupported(bos, owner, "has no TTIR mapping (dynamic shape)");
    return std::nullopt;
  }
  std::string attrs;
  llvm::raw_string_ostream as(attrs);
  as << "epsilon = ";
  epsilon.print(as);
  as << ", normalized_shape = array<i64: " << inType.getShape().back()
     << ">, operandSegmentSizes = array<i32: 1, 1, 0>";
  // The rmsnorm result has the SAME shape as the input (dialect contract).
  return emitBinaryLine(bos, "ttir.rms_norm", *in, *g, attrs,
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
  // results.
  std::string typeText;
  llvm::raw_string_ostream ts(typeText);
  ts << type;
  unsigned n = ssaCounter++;
  bos << "  %" << n << " = \"ttir.constant\"() <{value = ";
  op.getValue().print(bos);
  bos << "}> : () -> " << typeText << "\n";
  values[op.getResult()] = RenderedValue{"%" + std::to_string(n), typeText};
}

void TTIRFuncTranslator::emitFusedRmsnormMatmul(
    llvm::raw_ostream &bos, TensixflowFusedRmsnormMatmulOp op) {
  std::optional<RenderedValue> normed = emitRmsNorm(
      bos, *op, op.getInput(), op.getGamma(), op.getEpsilonAttr());
  if (!normed)
    return; // Marker already emitted.
  const RenderedValue *w = lookup(op.getWeight());
  auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
  if (!w || !resultType || !resultType.hasStaticShape()) {
    markUnsupported(bos, *op,
                    "has no TTIR mapping (operand is unsupported)");
    return;
  }
  values[op.getResult()] =
      emitBinaryLine(bos, "ttir.matmul", *normed, *w, /*attrs=*/"",
                     emitter.renderTensorType(resultType, MemSpace::L1));
}

void TTIRFuncTranslator::emitFusedMlp(llvm::raw_ostream &bos,
                                      TensixflowFusedMlpOp op) {
  // The study-grounded PREFIX decomposes: rms_norm -> matmul w1 -> bias add
  // (the bias add uses the study's segformer broadcast-add form). The gelu
  // stage has no study-grounded mapping, so it and the tail consuming it
  // (matmul w2, residual add) are marked UNSUPPORTED.
  std::optional<RenderedValue> normed = emitRmsNorm(
      bos, *op, op.getInput(), op.getGamma(), op.getEpsilonAttr());
  if (!normed)
    return;
  const RenderedValue *w1 = lookup(op.getW1());
  auto w1Type = dyn_cast<RankedTensorType>(op.getW1().getType());
  auto inType = dyn_cast<RankedTensorType>(op.getInput().getType());
  if (!w1 || !w1Type || !inType || !w1Type.hasStaticShape() ||
      !inType.hasStaticShape()) {
    markUnsupported(bos, *op,
                    "has no TTIR mapping (operand is unsupported)");
    return;
  }
  // matmul(w1) result shape: input.shape[:-1] + [w1.shape.back()].
  SmallVector<int64_t> hiddenShape(inType.getShape().drop_back());
  hiddenShape.push_back(w1Type.getShape().back());
  auto hiddenType =
      RankedTensorType::get(hiddenShape, inType.getElementType());
  RenderedValue hidden =
      emitBinaryLine(bos, "ttir.matmul", *normed, *w1, /*attrs=*/"",
                     emitter.renderTensorType(hiddenType, MemSpace::L1));
  if (const RenderedValue *b1 = lookup(op.getB1()))
    emitBinaryLine(bos, "ttir.add", hidden, *b1, /*attrs=*/"", hidden.type);
  else
    markUnsupported(bos, *op,
                    "bias-add stage has no TTIR mapping (operand is "
                    "unsupported)");
  // The gelu stage is the unmapped cut; everything after it is poisoned.
  bos << "  // UNSUPPORTED: tensixflow.gelu has no TTIR mapping\n"
      << "  // UNSUPPORTED: tensixflow.fused_mlp tail (matmul w2, residual "
         "add) has no TTIR mapping (operand is unsupported)\n";
  // The fused op's result stays unbound, so the return marks too.
}

void TTIRFuncTranslator::emitReturn(TensixflowReturnOp op,
                                    llvm::raw_ostream &bos,
                                    std::string &returnTypeText) {
  const RenderedValue *rv =
      op.getOperands().empty() ? nullptr : lookup(op.getOperands().front());
  if (!rv) {
    if (!op.getOperands().empty())
      markUnsupported(bos, *op,
                      "has no TTIR mapping (operand is unsupported)");
    // Signature fallback: the source func's result type, written back to
    // DRAM (the writer-kernel convention) when it is a tensor. A void func
    // leaves returnTypeText empty (translate then omits the arrow).
    FunctionType fnType =
        op->getParentOfType<TensixflowFuncOp>().getFunctionType();
    if (fnType.getNumResults() != 1)
      return;
    Type srcResult = fnType.getResult(0);
    if (auto t = dyn_cast<RankedTensorType>(srcResult))
      returnTypeText = emitter.renderTensorType(t, MemSpace::DRAM);
    else {
      llvm::raw_string_ostream ts(returnTypeText);
      ts << srcResult;
    }
    return;
  }
  bos << "  return " << rv->ssa << " : " << rv->type << "\n";
  returnTypeText = rv->type;
}

void TTIRFuncTranslator::dispatch(Operation &op, llvm::raw_ostream &bos,
                                  std::string &returnTypeText) {
  if (auto ret = dyn_cast<TensixflowReturnOp>(op))
    return emitReturn(ret, bos, returnTypeText);
  if (auto add = dyn_cast<TensixflowAddOp>(op))
    return emitBinary(bos, "ttir.add", *add, add.getLhs(), add.getRhs(),
                      add.getResult());
  if (auto matmul = dyn_cast<TensixflowMatmulOp>(op))
    return emitBinary(bos, "ttir.matmul", *matmul, matmul.getLhs(),
                      matmul.getRhs(), matmul.getResult());
  if (auto rmsnorm = dyn_cast<TensixflowRmsnormOp>(op)) {
    if (auto rendered = emitRmsNorm(bos, *rmsnorm, rmsnorm.getInput(),
                                    rmsnorm.getGamma(),
                                    rmsnorm.getEpsilonAttr()))
      values[rmsnorm.getResult()] = *rendered;
    return;
  }
  if (auto constant = dyn_cast<TensixflowConstantOp>(op))
    return emitConstant(bos, constant);
  if (auto fused = dyn_cast<TensixflowFusedRmsnormMatmulOp>(op))
    return emitFusedRmsnormMatmul(bos, fused);
  if (auto mlp = dyn_cast<TensixflowFusedMlpOp>(op))
    return emitFusedMlp(bos, mlp);
  // tensixflow.gelu today, any future op: the study-grounded vocabulary
  // (docs/ttmlir_study.md) decides, everything else is a documented marker.
  markUnsupported(bos, op, "has no TTIR mapping");
}

} // namespace tensixflow
} // namespace mlir
