//===- Compat/TTIRTextEmitter.h - Emit TTIR-shaped MLIR text -----*- C++ -*-===//
//
// TensixFlow: An MLIR Compiler for Tile-Native LLM Kernels on
// Tenstorrent-Style Spatial Accelerators.
//
//===----------------------------------------------------------------------===//
//
// The tensixflow -> TTIR text emitter engine, in two cooperating classes:
//
// * TTIRTextEmitter: the module-level scaffolding. Prints the banner, walks
//   the tensixflow.funcs, interns the module-global `#layoutN` =
//   `#ttcore.metal_layout<...>` aliases (TTIRLayoutAttrs), and assembles
//   each function's aliases + signature + body.
// * TTIRFuncTranslator: the per-function op -> TTIR mapping (add/matmul/
//   rmsnorm/gelu/constant/fused ops, the UNSUPPORTED convention).
//
// The mapping contract, including the pinned syntax sources, is documented on
// the --tensixflow-emit-ttir pass in lib/tensixflow/Compat/EmitTTIR.cpp and in
// docs/ttir.md.
//
//===----------------------------------------------------------------------===//

#ifndef TENSIXFLOW_COMPAT_TTIRTEXTEMITTER_H
#define TENSIXFLOW_COMPAT_TTIRTEXTEMITTER_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/StringRef.h"
#include "tensixflow/Compat/TTIRLayoutAttrs.h"

#include <optional>
#include <string>

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace mlir {
class FloatAttr;
class ModuleOp;
class Operation;

namespace tensixflow {
class TensixflowConstantOp;
class TensixflowFuncOp;
class TensixflowFusedMlpOp;
class TensixflowFusedRmsnormMatmulOp;
class TensixflowReturnOp;

/// A rendered SSA value: its name plus its already-rendered type text.
struct RenderedValue {
  std::string ssa;
  std::string type;
};

class TTIRFuncTranslator;

/// Module-level TTIR text emission: banner, layout aliases, func assembly.
/// Layout aliases are module-global so `#layoutN` never collides across
/// functions.
class TTIRTextEmitter {
public:
  explicit TTIRTextEmitter(llvm::raw_ostream &os) : os(os) {}

  /// Print the banner plus one TTIR func per tensixflow.func in the module.
  void emitModule(ModuleOp module);

  /// Render `tensor<shape x dtype, #layoutN>` for a rank of at least
  /// kMinLayoutRank, interning the alias. A lower rank keeps its plain tensor
  /// type because the pinned metal_layout verifier rejects rank-1 shapes.
  std::string renderTensorType(RankedTensorType type, MemSpace space);

private:
  friend class TTIRFuncTranslator;

  llvm::raw_ostream &os;
  /// Interned layout aliases, in first-use order (module-global).
  llvm::MapVector<std::pair<RankedTensorType, MemSpace>, unsigned> layouts;
  /// Number of aliases already printed to `os`.
  unsigned nextPrintedLayout = 0;

  unsigned internLayout(RankedTensorType type, MemSpace space);
  void printNewLayoutAliases();
  void emitFunc(TensixflowFuncOp func);
};

/// Per-function op -> TTIR translation. SSA names are per-function (%argN
/// for arguments, %0, %1, ... for emitted results).
class TTIRFuncTranslator {
public:
  explicit TTIRFuncTranslator(TTIRTextEmitter &emitter) : emitter(emitter) {}

  /// Translate the function's arguments + body to `bos`; returns the
  /// `func.func @name(args) -> ret {` signature line (the return operand's
  /// rendered type is only known once the body has been translated).
  std::string translate(TensixflowFuncOp func, llvm::raw_ostream &bos);

private:
  TTIRTextEmitter &emitter;
  /// Rendered values of the current function (args + emitted results).
  DenseMap<Value, RenderedValue> values;
  /// Numeric SSA-name counter for emitted op results.
  unsigned ssaCounter = 0;

  const RenderedValue *lookup(Value v) const;
  void markUnsupported(llvm::raw_ostream &bos, Operation &op,
                       StringRef reason);
  RenderedValue emitBinaryLine(llvm::raw_ostream &bos, StringRef ttirOp,
                               const RenderedValue &lhs,
                               const RenderedValue &rhs, StringRef attrs,
                               StringRef resultType);
  void emitBinary(llvm::raw_ostream &bos, StringRef ttirOp, Operation &owner,
                  Value lhs, Value rhs, Value result);
  RenderedValue emitUnaryLine(llvm::raw_ostream &bos, StringRef ttirOp,
                              const RenderedValue &operand,
                              StringRef resultType);
  void emitUnary(llvm::raw_ostream &bos, StringRef ttirOp, Operation &owner,
                 Value input, Value result);
  std::optional<RenderedValue> emitRmsNorm(llvm::raw_ostream &bos,
                                           Operation &owner, Value input,
                                           Value gamma, FloatAttr epsilon);
  void emitConstant(llvm::raw_ostream &bos, TensixflowConstantOp op);
  void emitFusedRmsnormMatmul(llvm::raw_ostream &bos,
                              TensixflowFusedRmsnormMatmulOp op);
  void emitFusedMlp(llvm::raw_ostream &bos, TensixflowFusedMlpOp op);
  void emitReturn(TensixflowReturnOp op, llvm::raw_ostream &bos,
                  std::string &returnTypeText);
  void dispatch(Operation &op, llvm::raw_ostream &bos,
                std::string &returnTypeText);
};

} // namespace tensixflow
} // namespace mlir

#endif // TENSIXFLOW_COMPAT_TTIRTEXTEMITTER_H
