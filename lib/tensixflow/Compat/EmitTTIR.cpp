//===- Compat/EmitTTIR.cpp - The --tensixflow-emit-ttir pass ----*- C++ -*-===//
//
// TensixFlow: An MLIR Compiler for Tile-Native LLM Kernels on
// Tenstorrent-Style Spatial Accelerators.
//
//===----------------------------------------------------------------------===//
//
// The --tensixflow-emit-ttir pass (todo 51, Wave 9): translates tensixflow
// ops into TTIR-shaped MLIR TEXT printed to stdout. This is the first code
// deliverable of the TT-MLIR compatibility wave. The engine lives in
// Compat/TTIRTextEmitter (layout-attr vocabulary in Compat/TTIRLayoutAttrs).
//
// TTIR ONLY per draft D21: TTNN/D2M/TTKernel/TTMetal stay prose mappings in
// docs/tenstorrent_mapping.md and are never emitted as code.
//
// Syntax contract = docs/ttmlir_study.md (tt-mlir @9f06802f test snippets):
//
// * Ops use the generic MLIR form with the shape in the function type:
//   `"ttir.matmul"(%a, %b) : (tensor<MxKxbf16>, tensor<KxNxbf16>) ->
//   tensor<MxNxbf16>`, attributes in the `<{...}>` slot.
// * Placement rides in `#ttcore.metal_layout` tensor encodings (see
//   TTIRLayoutAttrs.cpp for the exact grounded form). Function inputs are
//   `dram`, op results are `l1` (the InferTileLayoutPass memory
//   convention); the memref-level `#ttcore.memory_space<l1>` spelling is
//   the post-bufferization D2M form and stays prose in the banner.
//
// Op mappings:
// * add -> "ttir.add"; matmul -> "ttir.matmul" (generic form, no transpose
//   attrs: the positive-test form the study quotes).
// * rmsnorm -> "ttir.rms_norm" (CHOICE, documented): the fused canonical op
//   tt-mlir's own --ttir-fusing produces from the pow/sum/multiply/
//   add/rsqrt/broadcast/multiply decomposition chain the study quotes
//   (rms_norm_fusion.mlir's CHECK lines). Fused-to-fused is the 1:1
//   semantic match and carries the epsilon attribute across
//   (`<{epsilon = ..., normalized_shape = array<i64: K>,
//   operandSegmentSizes = array<i32: 1, 1, 0>}>`).
// * fused_rmsnorm_matmul -> rms_norm + matmul composition; fused_mlp ->
//   the study-grounded prefix (rms_norm + matmul + bias add) plus markers
//   for the gelu stage and its poisoned tail.
// * tensixflow.gelu emits a `// UNSUPPORTED` marker (CHOICE, documented):
//   ttir.gelu exists upstream, but it is outside the study doc's grounded
//   inventory, and the study doc is this wave's emitter contract; the
//   emitter maps only study-grounded ops rather than emit syntax the
//   contract does not cover. Todo 52 (mapping-doc extension) is the place
//   to admit it.
//
// UNSUPPORTED convention: an op with no study-grounded mapping emits
// `// UNSUPPORTED: <op> has no TTIR mapping` (never a crash); its results
// stay unbound, so dependent ops are marked transitively.
//
// This is a TERMINAL emission pass (the EmitMetaliumPass convention): the
// stdout text IS the artifact, so the pass flushes and exits 0 before
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

    // Terminal emission (the EmitMetaliumPass convention): stdout IS the
    // artifact, so exit 0 before mlir-opt appends the module IR. The text is
    // standalone TTIR for FileCheck today and for todo 53's gated
    // ttmlir-opt parse later.
    std::exit(0);
  }
};

} // namespace
} // namespace tensixflow
} // namespace mlir
