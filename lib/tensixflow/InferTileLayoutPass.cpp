//===- InferTileLayoutPass.cpp - Attach tile layouts ----------*- C++ -*-===//
//
// TensixFlow: An MLIR Compiler for Tile-Native LLM Kernels on
// Tenstorrent-Style Spatial Accelerators.
//
//===----------------------------------------------------------------------===//
//
// The --tensixflow-infer-tile-layout pass. Walks the module and attaches a
// #tensixflow.tile_layout<tile 32x32, grid <device grid>, mem ...> encoding as
// pure metadata (a "tensixflow.layout" discardable attr): function block
// arguments live in DRAM, and intermediate op results live in L1. Tensor shapes
// are never modified; the layout is metadata only. The grid comes from the
// selected device in the generated device table, not from a hard-coded 8x8.
//
// The pass also stamps the module with the "tensixflow.device" attribute so
// later passes and the JSON emitter agree on one device.
//
//===----------------------------------------------------------------------===//

#include "tensixflow/TensixflowPasses.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "tensixflow/Generated/DeviceSpecs.h"
#include "tensixflow/TensixflowAttrs.h"
#include "tensixflow/TensixflowOps.h"

#include "DeviceSelection.h"

namespace mlir {
namespace tensixflow {

// Emit impl::InferTileLayoutPassBase<DerivedT> (in mlir::tensixflow::impl) and
// the public createInferTileLayoutPass() definition for this one pass only.
// The other passes keep their GEN_PASS_DEF in TensixflowPasses.cpp, so no
// createXxx() is defined twice.
#define GEN_PASS_DEF_INFERTILELAYOUTPASS
#include "tensixflow/TensixflowPasses.h.inc"

namespace {

/// The discardable attribute name the layout is recorded under. A
/// dialect-prefixed discardable attr (not a tensor-type encoding) keeps shapes
/// and op verifiers untouched.
constexpr StringLiteral kLayoutAttrName = "tensixflow.layout";

/// Resolve the device for this module: the stamped attribute wins, then the
/// driver selection (default n150), then the generated default. A malformed
/// attribute naming an unknown device falls through instead of crashing.
const device::DeviceSpec *resolveDevice(ModuleOp module) {
  if (auto attr = module->getAttrOfType<StringAttr>(device::kDeviceAttrName)) {
    if (const device::DeviceSpec *spec = device::lookup(attr.getValue())) {
      return spec;
    }
  }
  if (const device::DeviceSpec *spec = device::selectedDevice()) {
    return spec;
  }
  return device::defaultDevice();
}

class InferTileLayoutPass final
    : public impl::InferTileLayoutPassBase<InferTileLayoutPass> {
public:
  using Base = impl::InferTileLayoutPassBase<InferTileLayoutPass>;
  using Base::Base;

  void runOnOperation() final {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();
    const device::DeviceSpec *deviceSpec = resolveDevice(module);
    if (deviceSpec == nullptr) {
      module.emitError("no device spec available; pass --tensixflow-device");
      signalPassFailure();
      return;
    }

    // Record the device once so later passes and the JSON emitter agree.
    if (!module->hasAttr(device::kDeviceAttrName)) {
      module->setAttr(device::kDeviceAttrName,
                      StringAttr::get(ctx, deviceSpec->name));
    }

    const unsigned gridY = static_cast<unsigned>(deviceSpec->computeGrid.gridY);
    const unsigned gridX = static_cast<unsigned>(deviceSpec->computeGrid.gridX);
    auto l1 = TensixflowTileLayoutAttr::get(ctx, gridY, gridX, MemorySpace::L1);
    auto dram =
        TensixflowTileLayoutAttr::get(ctx, gridY, gridX, MemorySpace::DRAM);

    // Inputs (function block arguments of ranked-tensor type) live in DRAM.
    // Non-tensor arguments (e.g. scalars) are skipped and get no layout.
    module.walk([&](TensixflowFuncOp func) {
      for (unsigned i = 0, e = func.getNumArguments(); i < e; ++i) {
        if (isa<RankedTensorType>(func.getArgument(i).getType())) {
          func.setArgAttr(i, kLayoutAttrName, dram);
        }
      }
    });

    // Intermediate op results of ranked-tensor type live in L1. Function ops are
    // skipped: their block arguments were handled above and a func op itself has
    // no SSA results.
    module.walk([&](Operation *op) {
      if (isa<TensixflowFuncOp>(op)) {
        return;
      }
      for (OpResult result : op->getResults()) {
        if (isa<RankedTensorType>(result.getType())) {
          op->setAttr(kLayoutAttrName, l1);
          return;
        }
      }
    });
  }
};

} // namespace
} // namespace tensixflow
} // namespace mlir
