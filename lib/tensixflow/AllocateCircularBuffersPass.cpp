//===- AllocateCircularBuffersPass.cpp - Allocate CBs ----------*- C++ -*-===//
//
// TensixFlow: An MLIR Compiler for Tile-Native LLM Kernels on
// Tenstorrent-Style Spatial Accelerators.
//
//===----------------------------------------------------------------------===//
//
// The --tensixflow-allocate-circular-buffers pass. Reads the #tensixflow.schedule
// attribute attached by --tensixflow-plan-noc-sharding, runs the circular-buffer
// allocator for the default three-CB reader/compute/writer pipeline on the
// device named by the module's "tensixflow.device" attribute, and fills the
// schedule's CB arrays (indices {0,1,16}, depth, tile-bytes) in place. The depth
// starts at the tt-metal example depth of two tiles per buffer and clamps to the
// per-core L1 budget, so the footprint never exceeds the budget. This pass only
// refines the plan; it emits no kernels.
//
//===----------------------------------------------------------------------===//

#include "tensixflow/TensixflowPasses.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "tensixflow/Generated/DeviceSpecs.h"
#include "tensixflow/TensixflowAttrs.h"

#include "CBAllocator.h"
#include "DeviceSelection.h"
#include "Model.h"

namespace mlir {
namespace tensixflow {

// Emit impl::AllocateCircularBuffersPassBase<DerivedT> and the public
// createAllocateCircularBuffersPass() definition for this one pass only.
#define GEN_PASS_DEF_ALLOCATECIRCULARBUFFERSPASS
#include "tensixflow/TensixflowPasses.h.inc"

namespace {

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

class AllocateCircularBuffersPass final
    : public impl::AllocateCircularBuffersPassBase<
          AllocateCircularBuffersPass> {
public:
  using Base =
      impl::AllocateCircularBuffersPassBase<AllocateCircularBuffersPass>;
  using Base::Base;

  void runOnOperation() final {
    ModuleOp module = getOperation();
    auto schedule =
        module->getAttrOfType<TensixflowScheduleAttr>("tensixflow.schedule");
    if (!schedule) {
      return; // no plan to refine: the planning pass did not run or found nothing.
    }

    const device::DeviceSpec *deviceSpec = resolveDevice(module);
    if (deviceSpec == nullptr) {
      module.emitError("no device spec available; pass --tensixflow-device");
      signalPassFailure();
      return;
    }

    MLIRContext *ctx = &getContext();
    scheduler::CBAllocation allocation =
        scheduler::allocateCBs(*deviceSpec, scheduler::kDefaultNumCBs);

    IntegerType i64 = IntegerType::get(ctx, 64);
    SmallVector<Attribute> indices;
    SmallVector<Attribute> depths;
    SmallVector<Attribute> tileBytes;
    for (const CBSpec &cb : allocation.cbs) {
      indices.push_back(IntegerAttr::get(i64, cb.index));
      depths.push_back(IntegerAttr::get(i64, cb.depth));
      tileBytes.push_back(IntegerAttr::get(i64, cb.tileBytes));
    }

    module->setAttr(
        "tensixflow.schedule",
        TensixflowScheduleAttr::get(
            ctx, schedule.getGridY(), schedule.getGridX(),
            schedule.getActivationSharding(), schedule.getWeightStrategy(),
            schedule.getIntermediateStorage(), schedule.getFusion(),
            ArrayAttr::get(ctx, indices), ArrayAttr::get(ctx, depths),
            ArrayAttr::get(ctx, tileBytes), schedule.getDramReadBytes(),
            schedule.getDramWriteBytes(), schedule.getL1Bytes(),
            schedule.getMulticastBytes(), schedule.getSavedDramReadBytes(),
            schedule.getSavedDramWriteBytes(),
            schedule.getSavedIntermediateBytes()));
  }
};

} // namespace
} // namespace tensixflow
} // namespace mlir
