//===- AllocateCircularBuffersPass.cpp - Allocate CBs ----------*- C++ -*-===//
//
// TensixFlow: An MLIR Compiler for Tile-Native LLM Kernels on
// Tenstorrent-Style Spatial Accelerators.
//
//===----------------------------------------------------------------------===//
//
// The --tensixflow-allocate-circular-buffers pass. Reads the #tensixflow.schedule
// attribute attached by --tensixflow-plan-noc-sharding, runs the circular-buffer
// allocator (todo 28) for the default 3-CB reader/compute/writer pipeline, and
// fills the schedule's CB arrays (indices {0,1,16}, depth, tile-bytes) in place
// on the module. The allocator's depth is the max feasible under the per-core L1
// budget, so the footprint never exceeds it. This pass only refines the plan; it
// emits NO kernels (Wave 6).
//
// This is the real body for the AllocateCircularBuffersPass stub that previously
// lived in TensixflowPasses.cpp (that stub + its GEN_PASS_DEF are removed so
// there is exactly one definition of the pass and its createXxx()).
//
//===----------------------------------------------------------------------===//

#include "tensixflow/TensixflowPasses.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "tensixflow/TensixflowAttrs.h"

#include "CBAllocator.h"
#include "Model.h"

namespace mlir {
namespace tensixflow {

// Emit impl::AllocateCircularBuffersPassBase<DerivedT> and the public
// createAllocateCircularBuffersPass() definition for this one pass only.
#define GEN_PASS_DEF_ALLOCATECIRCULARBUFFERSPASS
#include "tensixflow/TensixflowPasses.h.inc"

namespace {

// The default reader/compute/writer pipeline uses three circular buffers; the
// allocator's budget math is dtype-agnostic, so bf16 sizes the report tile.
constexpr int64_t kNumCBs = 3;

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
    if (!schedule)
      return; // no plan to refine (the planning pass did not run / found nothing)

    MLIRContext *ctx = &getContext();
    scheduler::CBAllocation alloc =
        scheduler::allocateCBs(kNumCBs, scheduler::bytesPerElem("bf16"));

    IntegerType i64 = IntegerType::get(ctx, 64);
    SmallVector<Attribute> indices;
    SmallVector<Attribute> depths;
    SmallVector<Attribute> tileBytes;
    for (const CBSpec &cb : alloc.cbs) {
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
