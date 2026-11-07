//===- Compat/EmitTTIR.cpp - The --tensixflow-emit-ttir pass ----*- C++ -*-===//
//
// TensixFlow: An MLIR Compiler for Tile-Native LLM Kernels on
// Tenstorrent-Style Spatial Accelerators.
//
//===----------------------------------------------------------------------===//
//
// The --tensixflow-emit-ttir pass: translates tensixflow ops into TTIR-shaped
// MLIR text printed to stdout. The engine lives in Compat/TTIRTextEmitter and
// the layout vocabulary in Compat/TTIRLayoutAttrs. The released tt-mlir parser
// validates the text in python/tensixflow/tests/test_ttir_parse.py.
//
// Syntax contract: tt-mlir 0.9.0.dev20260221, the wheel pinned in
// deps/tt-mlir.lock. scripts/fetch_ttmlir_fixtures.sh caches the quoted
// fixtures under .deps/ttmlir/fixtures, and docs/ttir.md lists the supported
// subset.
//
// * Ops use the generic MLIR form with operand and result types spelled out:
//   `"ttir.matmul"(%a, %b) : (tensor<...>, tensor<...>) -> tensor<...>`, with
//   attributes in the `<{...}>` slot.
// * Placement rides in `#ttcore.metal_layout` tensor encodings with the pinned
//   field order quoted in TTIRLayoutAttrs.cpp. Function inputs are `dram`, op
//   results are `l1` (the InferTileLayoutPass convention). A rank-1 tensor
//   keeps a plain type because the pinned verifier rejects a rank-1
//   logical_shape.
//
// Op mappings:
// * add -> "ttir.add"; matmul -> "ttir.matmul". TTIR elementwise binary ops are
//   broadcastable at the pin, so a rank-1 bias operand in the fused MLP is
//   legal.
// * rmsnorm -> "ttir.rms_norm" with the optional weight operand and the
//   operandSegmentSizes attribute the pin requires (see TTIRFuncTranslator.cpp).
// * gelu -> "ttir.gelu".
// * constant -> "ttir.constant" with a dense value attr.
// * fused_rmsnorm_matmul -> rms_norm + matmul. fused_mlp -> rms_norm + matmul +
//   broadcast bias add + gelu, with the remaining tail marked unsupported.
//
// UNSUPPORTED convention: an op with no mapping emits
// `// UNSUPPORTED: <op> has no TTIR mapping` (never a crash), and its results
// stay unbound so dependent ops mark transitively.
//
// This is a terminal emission pass (the EmitMetaliumPass convention): the
// stdout text is the artifact, so the pass flushes and exits 0 before
// mlir-opt would append the module IR.
//
//===----------------------------------------------------------------------===//

#include "tensixflow/TensixflowPasses.h"

#include "mlir/IR/BuiltinOps.h"
#include "tensixflow/Compat/TTIRTextEmitter.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>

namespace mlir {
namespace tensixflow {

// Emit impl::EmitTTIRPassBase<DerivedT> (in mlir::tensixflow::impl) and the
// public createEmitTTIRPass() definition for this one pass only.
#define GEN_PASS_DEF_EMITTTIRPASS
#include "tensixflow/TensixflowPasses.h.inc"

namespace {

class EmitTTIRPass final : public impl::EmitTTIRPassBase<EmitTTIRPass> {
public:
  using Base = impl::EmitTTIRPassBase<EmitTTIRPass>;
  using Base::Base;

  void runOnOperation() final {
    TTIRTextEmitter emitter(llvm::outs());
    emitter.emitModule(getOperation());
    llvm::outs().flush();

    // Terminal emission (the EmitMetaliumPass convention): stdout is the
    // artifact, so exit 0 before mlir-opt appends the module IR.
    std::exit(0);
  }
};

} // namespace
} // namespace tensixflow
} // namespace mlir
