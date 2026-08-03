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
  os << "// ===== tensixflow -> TTIR (--tensixflow-emit-ttir, todo 51) =====\n"
     << "// TTIR only, per draft D21: TTNN/D2M/TTKernel/TTMetal stay prose\n"
     << "// mappings in docs/tenstorrent_mapping.md, never emitted as code.\n"
     << "// Syntax contract: docs/ttmlir_study.md (tt-mlir @9f06802f).\n"
     << "// Memory-space prose mapping: intermediates live in\n"
     << "// #ttcore.memory_space<l1> and function inputs in\n"
     << "// #ttcore.memory_space<dram>, spelled `l1`/`dram` inside each\n"
     << "// #ttcore.metal_layout below; the #ttcore.memory_space<l1> attr\n"
     << "// itself is the memref-level (D2M) spelling this emitter does not\n"
     << "// produce. Unmapped ops emit `// UNSUPPORTED: <op> has no TTIR\n"
     << "// mapping` markers (never a crash).\n";
  module.walk([&](TensixflowFuncOp func) { emitFunc(func); });
}

unsigned TTIRTextEmitter::internLayout(RankedTensorType type, MemSpace space) {
  auto key = std::make_pair(type, space);
  auto it = layouts.find(key);
  if (it != layouts.end())
    return it->second;
  unsigned idx = layouts.size();
  layouts.insert({key, idx});
  return idx;
}

std::string TTIRTextEmitter::renderTensorType(RankedTensorType type,
                                              MemSpace space) {
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
    if (idx < nextPrintedLayout)
      continue;
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
