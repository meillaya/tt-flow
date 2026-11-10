//===- EmitMetaliumPass.cpp - Emit a TT-Metalium project --------*- C++ -*-===//
//
// TensixFlow: An MLIR Compiler for Tile-Native LLM Kernels on
// Tenstorrent-Style Spatial Accelerators.
//
//===----------------------------------------------------------------------===//
//
// The --tensixflow-emit-metalium pass: the code generation entry point. It reads
// the tile-level schedule IR (the `tensixflow.sched.kernel` built by
// --tensixflow-lower-to-ttkernel), the module-level #tensixflow.schedule plan
// attribute, the module's tensixflow.emission emitted-path accounting, and the
// module's tensixflow.device attribute, then emits the four generated
// translation units of the pinned-tt-metal project:
//
//   host.cpp            (emitHost),
//   kernels/compute.cpp (emitCompute),
//   kernels/reader.cpp  (emitReader),
//   kernels/writer.cpp  (emitWriter).
//
// With the `output-dir` option set (--tensixflow-emit-metalium=output-dir=<dir>)
// the pass writes a complete standalone project tree (the four sources plus a
// TT-Metalium CMakeLists.txt and a README.md, see lib/tensixflow/Codegen/
// EmitProject.cpp). With the option unset it prints the four artifacts to stdout
// separated by file-delimiter banners.
//
// Validation runs before any output in both modes: the schedule attribute, the
// sched.kernel, the matmul shape, the emission block, the recorded output-tile
// mapping against the selector's choice for the tile geometry, and the
// unsupported fused path are all checked first, so a failure never leaves a
// half-written tree and never reports a compile that did not happen.
//
// The emitted project targets the pinned tt-metal release recorded in
// deps/tt-metal.lock. scripts/build_generated.sh compiles and links it, and
// scripts/run_generated.sh runs it under the official simulator.
//
//===----------------------------------------------------------------------===//

#include "tensixflow/TensixflowPasses.h"

#include "mlir/IR/BuiltinOps.h"
#include "tensixflow/Codegen/EmitCompute.h"
#include "tensixflow/Codegen/EmitCommon.h"
#include "tensixflow/Codegen/EmitHost.h"
#include "tensixflow/Codegen/EmitProject.h"
#include "tensixflow/Codegen/EmitReader.h"
#include "tensixflow/Codegen/EmitWriter.h"
#include "tensixflow/Generated/DeviceSpecs.h"
#include "tensixflow/TensixflowAttrs.h"
#include "tensixflow/TensixflowSchedule.h"

#include "CBAllocator.h"
#include "EmissionEstimator.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <string>

namespace mlir {
namespace tensixflow {

// Emit impl::EmitMetaliumPassBase<DerivedT> (in mlir::tensixflow::impl) and the
// public createEmitMetaliumPass() definition for this one pass only.
#define GEN_PASS_DEF_EMITMETALIUMPASS
#include "tensixflow/TensixflowPasses.h.inc"

namespace {

// The schedule attribute with the fused plan's six-buffer allocation in place.
// The emitters read the circular-buffer roles by position from this copy; the
// module keeps the three-buffer planning allocation.
TensixflowScheduleAttr withFusedCbAllocation(
    MLIRContext *ctx, TensixflowScheduleAttr schedule,
    const scheduler::CBAllocation &allocation) {
  IntegerType i64 = IntegerType::get(ctx, 64);
  llvm::SmallVector<Attribute> indices;
  llvm::SmallVector<Attribute> depths;
  llvm::SmallVector<Attribute> tileBytes;
  for (const CBSpec &cb : allocation.cbs) {
    indices.push_back(IntegerAttr::get(i64, cb.index));
    depths.push_back(IntegerAttr::get(i64, cb.depth));
    tileBytes.push_back(IntegerAttr::get(i64, cb.tileBytes));
  }
  return TensixflowScheduleAttr::get(
      ctx, schedule.getGridY(), schedule.getGridX(),
      schedule.getActivationSharding(), schedule.getWeightStrategy(),
      schedule.getIntermediateStorage(), schedule.getFusion(),
      ArrayAttr::get(ctx, indices), ArrayAttr::get(ctx, depths),
      ArrayAttr::get(ctx, tileBytes), schedule.getDramReadBytes(),
      schedule.getDramWriteBytes(), schedule.getL1Bytes(),
      schedule.getMulticastBytes(), schedule.getSavedDramReadBytes(),
      schedule.getSavedDramWriteBytes(),
      schedule.getSavedIntermediateBytes());
}

// The device name carried by the module's tensixflow.device attribute; the
// planner stamps it, and a default keeps a hand-written module emittable.
llvm::StringRef resolveDeviceName(ModuleOp module) {
  if (auto attr = module->getAttrOfType<StringAttr>("tensixflow.device")) {
    if (device::lookup(attr.getValue())) {
      return attr.getValue();
    }
  }
  return device::defaultDeviceName();
}

class EmitMetaliumPass final
    : public impl::EmitMetaliumPassBase<EmitMetaliumPass> {
public:
  using Base = impl::EmitMetaliumPassBase<EmitMetaliumPass>;
  using Base::Base;

  void runOnOperation() final {
    ModuleOp module = getOperation();

    // The plan attribute is required input.
    auto schedule =
        module->getAttrOfType<TensixflowScheduleAttr>("tensixflow.schedule");
    if (!schedule) {
      module.emitError(
          "requires a tensixflow.schedule attribute; run "
          "--tensixflow-plan-noc-sharding "
          "--tensixflow-allocate-circular-buffers first");
      signalPassFailure();
      return;
    }

    // The schedule-IR kernel is required input.
    TensixflowSchedKernelOp kernel;
    module.walk([&](TensixflowSchedKernelOp op) { kernel = op; });
    if (!kernel) {
      module.emitError(
          "requires a tensixflow.sched.kernel; run "
          "--tensixflow-lower-to-ttkernel first");
      signalPassFailure();
      return;
    }

    // The matmul shape is what sizes the DRAM buffers and the reference matmul
    // in the emitted host program. Without it the emitter would have to invent
    // the dimensions, so refuse.
    if (!readMatmulShape(kernel)) {
      kernel.emitError()
          << "requires the " << kMatmulShapeAttrName
          << " attribute on " << kernel.getSymName()
          << "; run --tensixflow-lower-to-ttkernel to stamp it";
      signalPassFailure();
      return;
    }

    // The emitted-path accounting is required: it records the work split and
    // traffic the emitted program will produce, and the generated README quotes
    // it. The attribute is stamped by the planning pass.
    auto emission =
        module->getAttrOfType<TensixflowEmissionAttr>(kEmissionAttrName);
    if (!emission) {
      kernel.emitError()
          << "requires a " << kEmissionAttrName
          << " attribute; run --tensixflow-plan-noc-sharding first";
      signalPassFailure();
      return;
    }
    const llvm::StringRef recordedMapping = emission.getMapping().getValue();
    if (recordedMapping != kOutputTileRangesMapping &&
        recordedMapping != kBlockedReuseMapping) {
      kernel.emitError()
          << "unknown emission mapping '" << recordedMapping
          << "'; the emitter supports '" << kOutputTileRangesMapping << "' and '"
          << kBlockedReuseMapping
          << "', which --tensixflow-plan-noc-sharding stamps";
      signalPassFailure();
      return;
    }

    llvm::StringRef deviceName = resolveDeviceName(module);
    std::optional<MatmulShape> shape = readMatmulShape(kernel);
    std::optional<MlpShape> mlpShape;
    if (isFusedMlp(schedule)) {
      mlpShape = readMlpShape(kernel);
      if (!mlpShape) {
        kernel.emitError()
            << "requires the " << kFusedMlpShapeAttrName << " attribute on "
            << kernel.getSymName()
            << " for a fused MLP plan; run --tensixflow-lower-to-ttkernel to "
               "stamp it";
        signalPassFailure();
        return;
      }
    }
    TensixflowScheduleAttr effectiveSchedule = schedule;

    // Recompute the mapping selector from the module's own geometry and record
    // it as the codegen decision. A stale or hand-edited emission block cannot
    // force a mapping its tile geometry does not support.
    const device::DeviceSpec *deviceSpec = device::lookup(deviceName);
    if (deviceSpec == nullptr) {
      deviceSpec = device::defaultDevice();
    }
    scheduler::EmissionInputs projectionInputs;
    projectionInputs.mTiles = shape->mTiles();
    projectionInputs.nTiles = shape->nTiles();
    projectionInputs.kTiles = shape->kTiles();
    projectionInputs.gridY = schedule.getGridY();
    projectionInputs.gridX = schedule.getGridX();
    projectionInputs.tileBytes = deviceSpec->tileBytes;
    projectionInputs.cbBudgetBytes =
        deviceSpec->cbBudget(device::kDefaultL1ReserveBytes);
    projectionInputs.fused = isFusedPlan(schedule);
    if (mlpShape) {
      projectionInputs.fusedMlp = true;
      projectionInputs.hTiles = mlpShape->hTiles();
      projectionInputs.nTiles = mlpShape->nTiles();
    }
    const scheduler::EmissionPlan projection =
        scheduler::estimateEmission(projectionInputs);
    if (recordedMapping != projection.mapping) {
      kernel.emitError()
          << "the emission mapping '" << recordedMapping << "' does not match '"
          << projection.mapping
          << "', the mapping the selector chooses for this shape and grid; "
             "re-run --tensixflow-plan-noc-sharding to re-stamp the plan";
      signalPassFailure();
      return;
    }

    // Blocked reuse sizes its resident buffers from the 2D block geometry; the
    // CB indices still come from the plan allocation the emitters read. The
    // fused MLP emitter uses the same 2D block geometry with its own buffer set.
    MatmulEmission mapping;
    if (recordedMapping == kBlockedReuseMapping) {
      const scheduler::BlockAssignment block =
          scheduler::planBlockAssignment(projectionInputs);
      mapping.mapping = MatmulMapping::BlockedReuse;
      mapping.rowsPerCore = block.rowsPerCore;
      mapping.colsPerCore = block.colsPerCore;
      mapping.usedRows = block.usedRows;
      mapping.usedCols = block.usedCols;
      mapping.in0DepthTiles = block.in0Tiles;
      mapping.in1DepthTiles = block.in1Tiles;
      // With a single used row block the weight column block is read once per
      // block already, so there is nothing for the multicast to remove.
      mapping.multicastB = block.weightMulticast;
    }

    // Fused RMSNorm + MatMul: the plan's estimate gates emission, then the
    // emitter allocates the fused circular-buffer set against the device budget
    // and re-stamps the schedule copy the emitters read. The module keeps the
    // planning allocation so the plan JSON stays the planning artifact.
    if (isFusedPlan(schedule)) {
      if (isFusedMlp(schedule)) {
        const device::DeviceSpec *spec = deviceSpec;
        const int64_t budgetTiles =
            spec->cbBudget(device::kDefaultL1ReserveBytes) / spec->tileBytes;
        const scheduler::BlockAssignment block =
            scheduler::planBlockAssignment(projectionInputs);
        const int64_t stageTiles = emission.getFusedStageTiles();
        const int64_t neededTiles = scheduler::fusedMlpCbTileCount(
            mlpShape->kTiles(), mlpShape->hTiles(), block.rowsPerCore,
            block.colsPerCore);

        // The plan's emission block is the first gate: the planning pass
        // records whether the MLP footprint fits the per-core budget. The
        // diagnostic reports the recorded intermediate staging and the
        // recomputed footprint against the budget it was compared with.
        if (!emission.getFusedFeasible()) {
          kernel.emitError()
              << "the fused plan '" << fusionKind(schedule)
              << "' is not emittable: it keeps " << stageTiles
              << " intermediate (GELU output) circular-buffer tiles in L1, "
                 "and the whole chain needs "
              << neededTiles
              << " circular-buffer tiles per core (activation row, squares, "
                 "inverse RMS, normalized row, both weight blocks, gamma, "
                 "scaler, bias, residual row, intermediate rows, output) "
                 "against "
              << budgetTiles << " available tiles per core on '" << spec->name
              << "', so the emitted-path estimate marks the fused variant "
                 "infeasible. Emit a smaller MLP from "
                 "examples/mlp_block_small.mlir, or keep the fused plan as a "
                 "planning result: run tensixflow-opt through "
                 "--tensixflow-plan-noc-sharding, pipe the module into "
                 "tensixflow-translate --tensixflow-to-json, then render it "
                 "with tensixflow-report.";
          signalPassFailure();
          return;
        }

        scheduler::CBAllocation fused = scheduler::allocateFusedMlpCBs(
            *spec, mlpShape->kTiles(), mlpShape->hTiles(), block.rowsPerCore,
            block.colsPerCore);
        if (!fused.feasible) {
          kernel.emitError()
              << "the fused plan '" << fusionKind(schedule) << "' needs "
              << neededTiles
              << " circular-buffer tiles per core against " << budgetTiles
              << " available tiles per core on '" << spec->name
              << "', so the fused MLP program cannot be emitted. Emit a "
                 "smaller MLP from examples/mlp_block_small.mlir, or use the "
                 "plan and report path.";
          signalPassFailure();
          return;
        }

        mapping.mapping = MatmulMapping::BlockedReuse;
        mapping.rowsPerCore = block.rowsPerCore;
        mapping.colsPerCore = block.colsPerCore;
        mapping.usedRows = block.usedRows;
        mapping.usedCols = block.usedCols;
        mapping.in0DepthTiles = block.rowsPerCore * mlpShape->kTiles();
        mapping.in1DepthTiles = block.colsPerCore * mlpShape->hTiles();
        // The fused MLP reader stages its own blocks unicast; the blocked-reuse
        // selector above sized the resident blocks only.
        mapping.multicastB = false;
        effectiveSchedule =
            withFusedCbAllocation(&getContext(), schedule, fused);
      } else if (!isFusedRmsnormMatmul(schedule)) {
        kernel.emitError()
            << "cannot emit a TT-Metalium program for the fused plan '"
            << fusionKind(schedule) << "'. The emitter supports the '"
            << kFusedRmsnormMatmulKind << "' and '" << kFusedMlpKind
            << "' patterns (an RMSNorm prologue followed by a matmul, and the "
               "full MLP chain; see examples/rmsnorm_matmul_small.mlir and "
               "examples/mlp_block_small.mlir). For another fused graph, emit "
               "the matmul-only path from examples/matmul.mlir, or use the "
               "plan and report path: run the planning passes, pipe the module "
               "into tensixflow-translate --tensixflow-to-json, then render it "
               "with tensixflow-report.";
        signalPassFailure();
        return;
      } else {
        const device::DeviceSpec *spec = deviceSpec;
        const int64_t budgetTiles =
            spec->cbBudget(device::kDefaultL1ReserveBytes) / spec->tileBytes;
        const int64_t kTiles = shape->kTiles();
        const int64_t stageTiles = emission.getFusedStageTiles();

        // The plan's emission block is the first gate: the planning pass records
        // whether the fused staging fits the per-core budget, including the
        // gamma tiles and the scaler. The emitter reports the recorded staging
        // and the budget it was compared against rather than recomputing the
        // estimate, so the diagnostic cannot disagree with the plan.
        if (!emission.getFusedFeasible()) {
          kernel.emitError()
              << "the fused plan '" << fusionKind(schedule)
              << "' is not emittable: it stages " << stageTiles
              << " circular-buffer tiles of activation rows plus " << kTiles
              << " weight tiles and " << kTiles << " gamma tiles, against "
              << budgetTiles << " available tiles per core on '" << spec->name
              << "', and the emitted-path estimate marks the fused variant "
                 "infeasible. Emit the matmul-only path from examples/matmul.mlir, "
                 "or keep the fused plan as a planning result: run tensixflow-opt "
                 "through --tensixflow-plan-noc-sharding, pipe the module into "
                 "tensixflow-translate --tensixflow-to-json, then render it with "
                 "tensixflow-report.";
          signalPassFailure();
          return;
        }

        scheduler::CBAllocation fused =
            scheduler::allocateFusedCBs(*spec, kTiles, stageTiles);
        const int64_t neededTiles = scheduler::fusedCbTileCount(kTiles, stageTiles);
        if (!fused.feasible) {
          kernel.emitError()
              << "the fused plan '" << fusionKind(schedule) << "' needs "
              << neededTiles << " circular-buffer tiles (2 * " << stageTiles
              << " staged + " << kTiles << " weight + " << kTiles
              << " gamma + 1 scaler + 2 output) against " << budgetTiles
              << " available tiles per core on '" << spec->name
              << "', so the fused program cannot be emitted. Emit the "
                 "matmul-only path from examples/matmul.mlir, or keep the fused "
                 "plan as a planning result: run tensixflow-opt through "
                 "--tensixflow-plan-noc-sharding, pipe the module into "
                 "tensixflow-translate --tensixflow-to-json, then render it with "
                 "tensixflow-report.";
          signalPassFailure();
          return;
        }

        // The compute kernel accumulates one output tile per output column block
        // in the destination register file. FP32 destination accumulation leaves
        // eight destination tiles.
        if (shape->nTiles() > kFusedOutputColumnTileLimit) {
          kernel.emitError()
              << "the fused plan '" << fusionKind(schedule) << "' has "
              << shape->nTiles() << " output column tiles, but the emitted fused "
                 "compute kernel holds one output tile per column block in the "
                 "destination registers and FP32 accumulation leaves "
              << kFusedOutputColumnTileLimit
              << ". Emit the matmul-only path from examples/matmul.mlir for this "
                 "shape, or use the plan and report path.";
          signalPassFailure();
          return;
        }

        effectiveSchedule = withFusedCbAllocation(&getContext(), schedule, fused);
      }
    }

    // File mode: assemble the complete standalone project tree.
    if (!outputDir.empty()) {
      std::string error;
      if (failed(emitProject(kernel, effectiveSchedule, deviceName, emission,
                             mapping, outputDir, error))) {
        module.emitError("failed to write the Metalium project: ") << error;
        signalPassFailure();
        return;
      }
      std::exit(0);
    }

    // Stdout mode: the four artifacts with file-delimiter banners. host.cpp is
    // the prefix before the first banner; compute.cpp stays between host.cpp and
    // the two data-movement kernels so the FileCheck gates on reader/writer
    // output stay separate from the compute kernel.
    emitHost(kernel, effectiveSchedule, mapping, llvm::outs());
    llvm::outs() << "\n// ===== file: compute.cpp =====\n";
    emitCompute(kernel, effectiveSchedule, mapping, llvm::outs());
    if (mapping.multicastB) {
      llvm::outs() << "\n// ===== file: reader_sender.cpp =====\n";
      emitReaderMulticastSender(effectiveSchedule, llvm::outs());
      llvm::outs() << "\n// ===== file: reader_receiver.cpp =====\n";
      emitReaderMulticastReceiver(effectiveSchedule, llvm::outs());
    } else {
      llvm::outs() << "\n// ===== file: reader.cpp =====\n";
      emitReader(kernel, effectiveSchedule, mapping, llvm::outs());
    }
    llvm::outs() << "\n// ===== file: writer.cpp =====\n";
    emitWriter(kernel, effectiveSchedule, mapping, llvm::outs());
    llvm::outs().flush();

    // mlir-opt prints the transformed module to stdout after all passes run, and
    // this MLIR build has no -disable-output flag. This is a terminal emission
    // pass whose stdout is the artifact set, so exit here with success; the
    // failure paths above signal failure and let mlir-opt report it.
    std::exit(0);
  }
};

} // namespace
} // namespace tensixflow
} // namespace mlir
