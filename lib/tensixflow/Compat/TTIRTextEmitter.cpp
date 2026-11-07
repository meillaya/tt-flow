//===- Compat/TTIRTextEmitter.cpp - Module-level TTIR emission ---*- C++ -*-===//
//
// TensixFlow: An MLIR Compiler for Tile-Native LLM Kernels on
// Tenstorrent-Style Spatial Accelerators.
//
//===----------------------------------------------------------------------===//
//
// The module-level scaffolding of the tensixflow -> TTIR text emitter: the
// banner, the module-global #layoutN alias interning, and per-function
// assembly (the op mapping itself is TTIRFuncTranslator; the contract is
// documented on the pass in EmitTTIR.cpp).
//
//===----------------------------------------------------------------------===//

#include "tensixflow/Compat/TTIRTextEmitter.h"

#include "mlir/IR/BuiltinOps.h"
#include "tensixflow/TensixflowOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace tensixflow {

void TTIRTextEmitter::emitModule(ModuleOp module) {
  os << "// ===== tensixflow -> TTIR (--tensixflow-emit-ttir) =====\n"
     << "// Syntax contract: tt-mlir 0.9.0.dev20260221 (deps/tt-mlir.lock),\n"
     << "// checked by the released tt-mlir parser in\n"
     << "// python/tensixflow/tests/test_ttir_parse.py. Tensor placement\n"
     << "// rides in #ttcore.metal_layout aliases with the pinned field order:\n"
     << "// logical shape, dim alignments, collapse intervals, OOB value,\n"
     << "// memory space, memory layout, index map. Rank-1 tensors keep a\n"
     << "// plain type because the pinned verifier rejects a rank-1 logical\n"
     << "// shape. Ops with no TTIR mapping emit an UNSUPPORTED comment marker\n"
     << "// naming the op, never a crash.\n";
  module.walk([&](TensixflowFuncOp func) { emitFunc(func); });
}

unsigned TTIRTextEmitter::internLayout(RankedTensorType type, MemSpace space) {
  auto key = std::make_pair(type, space);
  auto it = layouts.find(key);
  if (it != layouts.end()) {
    return it->second;
  }
  unsigned idx = layouts.size();
  layouts.insert({key, idx});
  return idx;
}

std::string TTIRTextEmitter::renderTensorType(RankedTensorType type,
                                              MemSpace space) {
  // The pinned metal_layout verifier rejects a rank-1 logical shape, so a
  // rank-1 weight or result keeps a plain tensor type.
  if (type.getRank() < kMinLayoutRank) {
    std::string plain;
    llvm::raw_string_ostream ss(plain);
    ss << type;
    return plain;
  }
  unsigned idx = internLayout(type, space);
  std::string text;
  llvm::raw_string_ostream ss(text);
  ss << "tensor<";
  llvm::interleave(type.getShape(), ss, "x");
  ss << "x";
  type.getElementType().print(ss);
  ss << ", #layout" << idx << ">";
  return text;
}

void TTIRTextEmitter::printNewLayoutAliases() {
  for (const auto &[key, idx] : layouts) {
    if (idx < nextPrintedLayout) {
      continue;
    }
    os << "#layout" << idx << " = ";
    printMetalLayout(os, key.first, key.second);
    os << "\n";
  }
  nextPrintedLayout = layouts.size();
}

void TTIRTextEmitter::emitFunc(TensixflowFuncOp func) {
  // The body lands in a scratch buffer first: the signature's result type
  // is only known once the return operand has been translated.
  std::string body;
  llvm::raw_string_ostream bos(body);
  TTIRFuncTranslator translator(*this);
  std::string signature = translator.translate(func, bos);
  printNewLayoutAliases();
  os << signature << body << "}\n";
}

} // namespace tensixflow
} // namespace mlir
