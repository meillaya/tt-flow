//===- PlanNocShardingPass.cpp - Plan NoC sharding -------------*- C++ -*-===//
//
// TensixFlow: An MLIR Compiler for Tile-Native LLM Kernels on
// Tenstorrent-Style Spatial Accelerators.
//
//===----------------------------------------------------------------------===//
//
// The --tensixflow-plan-noc-sharding pass. Walks the module, finds the graph to
// plan (an already-fused op, an unfused rmsnorm -> matmul pair planned as fused
// because the planner decides fusion, or a lone matmul planned unfused), derives
// the activation and weight operand shapes, runs the grid/weight/CB/NoC planners
// via buildPlanReport on the selected device, attaches a #tensixflow.schedule
// attribute to the module (grid/sharding/weight/fusion/NoC populated; the CB
// arrays are left empty for --tensixflow-allocate-circular-buffers to fill), and
// prints the human-readable "Chosen layout: ..." report to stderr. The pass
// stamps the module with "tensixflow.device" so later passes use one device.
// This pass only plans and reports; it emits no kernels.
//
//===----------------------------------------------------------------------===//

#include "tensixflow/TensixflowPasses.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "tensixflow/Generated/DeviceSpecs.h"
#include "tensixflow/TensixflowAttrs.h"
#include "tensixflow/TensixflowOps.h"

#include "DeviceSelection.h"
#include "EmissionEstimator.h"
#include "Model.h"
#include "PlanReport.h"

#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace tensixflow {

// Emit impl::PlanNocShardingPassBase<DerivedT> (in mlir::tensixflow::impl) and
// the public createPlanNocShardingPass() definition for this one pass only. The
// other passes keep their GEN_PASS_DEF elsewhere, so no createXxx() is defined
// twice. (Double-N macro: PlanNoc -> PLANNOC.)
#define GEN_PASS_DEF_PLANNOCSHARDINGPASS
#include "tensixflow/TensixflowPasses.h.inc"

namespace {

/// Element-dtype name for the scheduler model ("bf16" / "fp16" / "fp32").
std::string dtypeString(Type elem) {
  if (elem.isBF16())
    return "bf16";
  if (elem.isF16())
    return "fp16";
  if (elem.isF32())
    return "fp32";
  return "bf16";
}

/// The scheduler shape of a value's ranked static tensor type (empty dims for a
/// non-tensor / dynamic value, which the planners guard against).
scheduler::TensorShape tensorShapeOf(Value value, const std::string &name) {
  scheduler::TensorShape shape;
  shape.name = name;
  auto ranked = dyn_cast<RankedTensorType>(value.getType());
  if (ranked && ranked.hasStaticShape()) {
    shape.dims.assign(ranked.getShape().begin(), ranked.getShape().end());
    shape.dtype = dtypeString(ranked.getElementType());
  }
  return shape;
}

/// Ceiling division for the tile decomposition, guarded for a non-positive
/// divisor.
int64_t ceilDiv(int64_t numerator, int64_t divisor) {
  if (numerator <= 0 || divisor <= 0) {
    return 0;
  }
  return (numerator + divisor - 1) / divisor;
}

/// The matmul tile geometry the emitted-path accounting needs. The matmul treats
/// the activation as [..., M, K] and the weight as [K, N], the same convention
/// --tensixflow-lower-to-ttkernel uses when it builds the schedule IR.
struct MatmulTiles {
  int64_t m = 0;
  int64_t n = 0;
  int64_t k = 0;
};

MatmulTiles matmulTiles(const scheduler::TensorShape &activation,
                        const scheduler::TensorShape &weight,
                        const device::DeviceSpec &deviceSpec) {
  MatmulTiles tiles;
  int64_t m = 0;
  int64_t k = 0;
  int64_t n = 0;
  if (activation.dims.size() >= 2) {
    k = activation.dims.back();
    m = 1;
    for (size_t i = 0; i + 1 < activation.dims.size(); ++i) {
      m *= activation.dims[i];
    }
  }
  if (!weight.dims.empty()) {
    n = weight.dims.back();
  }
  tiles.m = ceilDiv(m, deviceSpec.tileHeight);
  tiles.n = ceilDiv(n, deviceSpec.tileWidth);
  tiles.k = ceilDiv(k, deviceSpec.tileWidth);
  return tiles;
}

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

class PlanNocShardingPass final
    : public impl::PlanNocShardingPassBase<PlanNocShardingPass> {
public:
  using Base = impl::PlanNocShardingPassBase<PlanNocShardingPass>;
  using Base::Base;

  void runOnOperation() final {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();

    std::string graphId = "graph";
    module.walk([&](TensixflowFuncOp func) {
      if (graphId == "graph") {
        graphId = func.getName().str();
      }
    });

    const device::DeviceSpec *deviceSpec = resolveDevice(module);
    if (deviceSpec == nullptr) {
      module.emitError("no device spec available; pass --tensixflow-device");
      signalPassFailure();
      return;
    }
    // Record the device so later passes and the JSON emitter agree on one spec.
    if (!module->hasAttr(device::kDeviceAttrName)) {
      module->setAttr(device::kDeviceAttrName,
                      StringAttr::get(ctx, deviceSpec->name));
    }

    scheduler::TensorShape activation;
    scheduler::TensorShape weight;
    std::string fusion = "none";
    bool fused = false;
    bool found = false;
    // The fused MLP chain needs the hidden width (the first matmul's output) and
    // the final output width in addition to the first matmul's M and K.
    int64_t mlpHiddenElements = 0;
    int64_t mlpOutputElements = 0;

    // An already-fused op carries its operand shapes directly.
    module.walk([&](TensixflowFusedRmsnormMatmulOp op) {
      if (found) {
        return;
      }
      activation = tensorShapeOf(op.getInput(), "activation");
      weight = tensorShapeOf(op.getWeight(), "weight");
      fusion = "fused_rmsnorm_matmul";
      fused = true;
      found = true;
    });
    if (!found) {
      module.walk([&](TensixflowFusedMlpOp op) {
        if (found) {
          return;
        }
        activation = tensorShapeOf(op.getInput(), "activation");
        weight = tensorShapeOf(op.getW1(), "weight");
        const scheduler::TensorShape w2Shape =
            tensorShapeOf(op.getW2(), "second weight");
        if (!weight.dims.empty()) {
          mlpHiddenElements = weight.dims.back();
        }
        if (!w2Shape.dims.empty()) {
          mlpOutputElements = w2Shape.dims.back();
        }
        fusion = "fused_mlp";
        fused = true;
        found = true;
      });
    }

    // An unfused rmsnorm -> matmul pair. The rmsnorm result feeds the
    // matmul LHS with a single use; the planner decides fusion, so this pair
    // is planned as fused.
    if (!found) {
      module.walk([&](TensixflowRmsnormOp rmsnorm) {
        if (found) {
          return;
        }
        Value result = rmsnorm.getResult();
        if (!result.hasOneUse()) {
          return;
        }
        auto matmul = dyn_cast<TensixflowMatmulOp>(*result.user_begin());
        if (!matmul || matmul.getLhs() != result) {
          return;
        }
        activation = tensorShapeOf(rmsnorm.getInput(), "activation");
        weight = tensorShapeOf(matmul.getRhs(), "weight");
        fusion = "fused_rmsnorm_matmul";
        fused = true;
        found = true;
      });
    }

    // A lone matmul is planned unfused, so the fusion savings are zero.
    if (!found) {
      module.walk([&](TensixflowMatmulOp matmul) {
        if (found) {
          return;
        }
        activation = tensorShapeOf(matmul.getLhs(), "activation");
        weight = tensorShapeOf(matmul.getRhs(), "weight");
        fusion = "none";
        fused = false;
        found = true;
      });
    }

    if (!found) {
      return;
    }

    int bpe = scheduler::bytesPerElem(activation.dtype);
    if (bpe == 0) {
      bpe = 2;
    }

    scheduler::PlanRequest request{graphId,   activation, weight, fusion,
                                   fused,     bpe,        deviceSpec};
    scheduler::PlanReport report = scheduler::buildPlanReport(request);

    // Attach the schedule attribute. The CB arrays are left empty here;
    // --tensixflow-allocate-circular-buffers fills them from the CB allocator.
    ArrayAttr emptyCbs = ArrayAttr::get(ctx, SmallVector<Attribute>{});
    auto schedule = TensixflowScheduleAttr::get(
        ctx, static_cast<unsigned>(report.plan.shardGrid.gridY),
        static_cast<unsigned>(report.plan.shardGrid.gridX),
        StringAttr::get(ctx, report.plan.layout.activationSharding),
        StringAttr::get(ctx, report.plan.layout.weightStrategy),
        StringAttr::get(ctx, report.intermediateStorage),
        StringAttr::get(ctx, report.fusion), emptyCbs, emptyCbs, emptyCbs,
        report.noc.estimate.dramReadBytes, report.noc.estimate.dramWriteBytes,
        report.noc.estimate.l1Bytes, report.noc.estimate.multicastBytes,
        report.noc.savedDramReadBytes, report.noc.savedDramWriteBytes,
        report.noc.savedIntermediateBytes);
    module->setAttr("tensixflow.schedule", schedule);

    // Record the emitted-path accounting. The Metalium emitter requires this
    // block: it is the plan-side record of the traffic and work split the
    // emitted program will produce, computed from the same tile geometry the
    // lowering pass uses.
    const MatmulTiles tiles =
        matmulTiles(request.activation, request.weight, *report.device);
    scheduler::EmissionInputs emissionInputs;
    emissionInputs.mTiles = tiles.m;
    emissionInputs.nTiles = tiles.n;
    emissionInputs.kTiles = tiles.k;
    emissionInputs.gridY = report.plan.shardGrid.gridY;
    emissionInputs.gridX = report.plan.shardGrid.gridX;
    emissionInputs.tileBytes = report.device->tileBytes;
    emissionInputs.cbBudgetBytes =
        report.device->cbBudget(device::kDefaultL1ReserveBytes);
    // The fused emitters stage their own intermediates, so a fused plan keeps
    // the mapping its emitter realizes: the MLP emitter uses a 2D block per
    // core, the fused RMSNorm emitter uses contiguous output-tile ranges.
    emissionInputs.fused = fused;
    if (fusion == "fused_mlp") {
      emissionInputs.fusedMlp = true;
      emissionInputs.hTiles =
          ceilDiv(mlpHiddenElements, report.device->tileWidth);
      emissionInputs.nTiles =
          ceilDiv(mlpOutputElements, report.device->tileWidth);
    }
    const scheduler::EmissionPlan emission =
        scheduler::estimateEmission(emissionInputs);
    module->setAttr(
        scheduler::kEmissionAttrName,
        TensixflowEmissionAttr::get(
            ctx, StringAttr::get(ctx, emission.mapping), emission.coresUsed,
            emission.maxTilesPerCore, emission.dramReadBytes,
            emission.dramWriteBytes, emission.fusedStageTiles,
            emission.fusedFeasible));

    llvm::errs() << scheduler::formatPlanReport(report);
  }
};

} // namespace
} // namespace tensixflow
} // namespace mlir
