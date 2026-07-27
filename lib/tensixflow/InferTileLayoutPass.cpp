//===- InferTileLayoutPass.cpp - Attach tile layouts ----------*- C++ -*-===//
//
// TensixFlow: An MLIR Compiler for Tile-Native LLM Kernels on
// Tenstorrent-Style Spatial Accelerators.
//
//===----------------------------------------------------------------------===//
//
// The --infer-tile-layout pass. Walks the module and attaches a
// #tensixflow.tile_layout<tile 32x32, grid 8x8, mem …> encoding as pure
// metadata (a "tensixflow.layout" discardable attr): function inputs (block
// arguments) live in DRAM, intermediate op results live in L1. Tensor shapes
// are never modified — the layout is metadata only. Sharding across the grid
// is planned later (Wave 5); this pass only records the memory space.
//
// This is the real body for the InferTileLayoutPass stub that previously
// lived in TensixflowPasses.cpp (that stub is removed so there is exactly
// one definition of the pass).
//
//===----------------------------------------------------------------------===//

#include "tensixflow/TensixflowPasses.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "tensixflow/TensixflowAttrs.h"
#include "tensixflow/TensixflowOps.h"

namespace mlir {
namespace tensixflow {

// Emit impl::InferTileLayoutPassBase<DerivedT> (in mlir::tensixflow::impl) and
// the public createInferTileLayoutPass() definition for this one pass only —
// the other six passes keep their GEN_PASS_DEF_* in TensixflowPasses.cpp, so
// no createXxx() is defined twice.
#define GEN_PASS_DEF_INFERTILELAYOUTPASS
#include "tensixflow/TensixflowPasses.h.inc"

namespace {

/// Modeled device: a single Wormhole n150-class part — an 8x8 core grid with
/// 32x32 tiles. The 4-arg builder defaults the tile to 32x32, so the attr
/// prints `tile 32x32, grid 8x8`.
constexpr unsigned kGridY = 8;
constexpr unsigned kGridX = 8;

/// The discardable attribute name the layout is recorded under. Chosen as a
/// dialect-prefixed discardable attr (not a tensor-type encoding) so shapes
/// and op verifiers are untouched.
constexpr StringLiteral kLayoutAttrName = "tensixflow.layout";

class InferTileLayoutPass final
    : public impl::InferTileLayoutPassBase<InferTileLayoutPass> {
public:
  using Base = impl::InferTileLayoutPassBase<InferTileLayoutPass>;
  using Base::Base;

  void runOnOperation() final {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();

    auto l1 =
        TensixflowTileLayoutAttr::get(ctx, kGridY, kGridX, MemorySpace::L1);
    auto dram =
        TensixflowTileLayoutAttr::get(ctx, kGridY, kGridX, MemorySpace::DRAM);

    // Inputs (function block arguments of ranked-tensor type) live in DRAM.
    // Non-tensor arguments (e.g. scalars) are skipped and get no layout.
    module.walk([&](TensixflowFuncOp func) {
      for (unsigned i = 0, e = func.getNumArguments(); i < e; ++i)
        if (isa<RankedTensorType>(func.getArgument(i).getType()))
          func.setArgAttr(i, kLayoutAttrName, dram);
    });

    // Intermediate op results of ranked-tensor type live in L1. Function ops
    // are skipped (their block arguments were handled above; a func op itself
    // has no SSA results).
    module.walk([&](Operation *op) {
      if (isa<TensixflowFuncOp>(op))
        return;
      for (OpResult result : op->getResults())
        if (isa<RankedTensorType>(result.getType())) {
          op->setAttr(kLayoutAttrName, l1);
          return;
        }
    });
  }
};

} // namespace
} // namespace tensixflow
} // namespace mlir
