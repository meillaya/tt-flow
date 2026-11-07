//===- LowerToTTKernelPass.cpp - Lower a plan to schedule IR ----*- C++ -*-===//
//
// TensixFlow: An MLIR Compiler for Tile-Native LLM Kernels on
// Tenstorrent-Style Spatial Accelerators.
//
//===----------------------------------------------------------------------===//
//
// The --tensixflow-lower-to-ttkernel pass: the codegen heart. It reads the
// module-level #tensixflow.schedule attribute (attached by
// --tensixflow-plan-noc-sharding + --tensixflow-allocate-circular-buffers) and the
// graph it plans, then builds the tile-level schedule IR: a
// `tensixflow.sched.kernel @<graphId>` holding three tile programs --
//   reader  : a tile_loop staging activation tiles into in0 and weight tiles into
//             in1 (noc_multicast when the weight strategy is "multicast", else a
//             plain noc_read), mirroring tt-metal's cb_reserve_back / noc_read /
//             cb_push_back reader kernel_main;
//   compute : a tile_loop that waits on the input CBs, runs the RMSNorm prologue
//             (for a fused_rmsnorm_matmul / fused_mlp), the matmul_tiles (in0 x
//             in1 -> out), an optional GELU epilogue (fused_mlp), and pops the
//             consumed input tiles;
//   writer  : a tile_loop that waits on the out CB and drains its tiles to the
//             destination over the NoC.
//
// This is a STRUCTURE-PRODUCING lowering: it emits schedule IR ONLY. It emits NO
// host C++ and changes no numerics; the simulator owns execution. CB indices
// are read from the schedule's allocation (default
// 3-CB pipeline {0,1,16}); trip counts are a documented first-order tile
// decomposition of the operand shapes over the modeled 32x32 tile.
//
// The pass registration lives with the other six in TensixflowPasses.cpp;
// this file owns the lowering body and the pass definition.
//
//===----------------------------------------------------------------------===//

#include "tensixflow/TensixflowPasses.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "tensixflow/Codegen/EmitCommon.h"
#include "tensixflow/TensixflowAttrs.h"
#include "tensixflow/TensixflowOps.h"
#include "tensixflow/TensixflowSchedule.h"

#include "llvm/Support/MathExtras.h"

namespace mlir {
namespace tensixflow {

// Emit impl::LowerToTTKernelPassBase<DerivedT> (in mlir::tensixflow::impl) and
// the public createLowerToTTKernelPass() definition for this one pass only.
#define GEN_PASS_DEF_LOWERTOTTKERNELPASS
#include "tensixflow/TensixflowPasses.h.inc"

namespace {

// The modeled Tensix tile is 32x32; trip counts decompose operand shapes into it.
constexpr int64_t kTile = 32;

// The RMSNorm prologue reads its per-column scale (gamma) from c_2, the first free
// CB index after the in0/in1 reader CBs (the 3-CB allocation reserves in0/in1/out
// only). A first-order structural choice the simulator/emitter reconciles; mirrors
// the schedule_ir.mlir convention (rmsnorm 0, 2, 0).
constexpr int64_t kCBGamma = 2;

// First-order NoC base addresses (structure only; the emitter/simulator resolves
// real get_noc_addr / get_noc_multicast_addr values).
constexpr int64_t kSrcAddr = 0;
constexpr int64_t kDstAddr = 0;

// ceil(dim / kTile), guarded to a positive minimum so a degenerate / dynamic shape
// still yields a valid (non-negative, non-zero) trip count.
int64_t tileCount(int64_t dim) {
  if (dim <= 0)
    return 1;
  return std::max<int64_t>(1, llvm::divideCeil(dim, kTile));
}

// The shared lowering context: the CB indices read from the schedule allocation,
// the per-region trip counts, the prologue/epilogue + weight-distribution flags
// derived from the schedule's fusion / weight-strategy decisions, and the rmsnorm
// epsilon carried off the source op.
struct KernelCtx {
  Location loc;
  int64_t in0, in1, outCb;
  int64_t readerTrip, computeTrip, writerTrip;
  int64_t numDests;
  bool hasRmsnorm, hasGelu, multicast;
  FloatAttr epsilon;
};

class LowerToTTKernelPass final
    : public impl::LowerToTTKernelPassBase<LowerToTTKernelPass> {
public:
  using Base = impl::LowerToTTKernelPassBase<LowerToTTKernelPass>;
  using Base::Base;

  void runOnOperation() final {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();

    // The schedule attribute is the lowering's input. Without it there is nothing
    // to lower from -- the required failure path.
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

    // Find the graph to lower (mirror PlanNocShardingPass's case analysis): an
    // already-fused op, an unfused rmsnorm -> matmul pair, or a lone matmul. Only
    // the activation + weight operand shapes (trip counts), the graph id (kernel
    // symbol), and the rmsnorm epsilon (prologue) are needed.
    std::string graphId = "graph";
    TensixflowFuncOp funcOp;
    Value activation, weight;
    FloatAttr epsilonAttr;
    bool found = false;

    module.walk([&](TensixflowFuncOp func) {
      if (graphId == "graph")
        graphId = func.getName().str();
      if (!funcOp)
        funcOp = func;
    });
    module.walk([&](TensixflowFusedRmsnormMatmulOp op) {
      if (found)
        return;
      activation = op.getInput();
      weight = op.getWeight();
      epsilonAttr = op.getEpsilonAttr();
      found = true;
    });
    if (!found)
      module.walk([&](TensixflowFusedMlpOp op) {
        if (found)
          return;
        activation = op.getInput();
        weight = op.getW1();
        epsilonAttr = op.getEpsilonAttr();
        found = true;
      });
    if (!found)
      module.walk([&](TensixflowRmsnormOp rmsnorm) {
        if (found)
          return;
        Value result = rmsnorm.getResult();
        if (!result.hasOneUse())
          return;
        auto matmul = dyn_cast<TensixflowMatmulOp>(*result.user_begin());
        if (!matmul || matmul.getLhs() != result)
          return;
        activation = rmsnorm.getInput();
        weight = matmul.getRhs();
        epsilonAttr = rmsnorm.getEpsilonAttr();
        found = true;
      });
    if (!found)
      module.walk([&](TensixflowMatmulOp matmul) {
        if (found)
          return;
        activation = matmul.getLhs();
        weight = matmul.getRhs();
        found = true;
      });

    if (!found)
      return; // schedule present but no recognizable graph (already lowered)

    // Derive M/K/N: the matmul treats the activation as [..., M, K] (M = product
    // of the leading/batch dims, K the contracting last dim) and the weight as
    // [K, N] (N the last dim).
    int64_t m = 1, k = 1, n = 1;
    if (auto actTy = dyn_cast<RankedTensorType>(activation.getType());
        actTy && actTy.getRank() >= 2 && actTy.hasStaticShape()) {
      ArrayRef<int64_t> dims = actTy.getShape();
      k = dims.back();
      m = 1;
      for (int64_t d : dims.drop_back())
        m *= d;
    }
    if (auto wTy = dyn_cast<RankedTensorType>(weight.getType());
        wTy && wTy.getRank() >= 1 && wTy.hasStaticShape())
      n = wTy.getShape().back();

    // Trip counts (documented first-order tile decomposition): the reader and
    // compute programs stream the reduction dimension (K tiles -- the matmul inner
    // K-loop of matmul_single_core); the writer drains every output tile.
    int64_t mTiles = tileCount(m);
    int64_t nTiles = tileCount(n);
    int64_t kTiles = tileCount(k);

    // CB indices come from the schedule's allocation (default 3-CB pipeline
    // {0,1,16}); they are read here, never hardcoded.
    ArrayAttr cbIdx = schedule.getCbIndices();
    if (cbIdx.size() < 3) {
      module.emitError(
          "tensixflow.schedule has no circular-buffer allocation; run "
          "--tensixflow-allocate-circular-buffers first");
      signalPassFailure();
      return;
    }
    int64_t in0 = cast<IntegerAttr>(cbIdx[0]).getInt();
    int64_t in1 = cast<IntegerAttr>(cbIdx[1]).getInt();
    int64_t outCb = cast<IntegerAttr>(cbIdx[2]).getInt();

    StringRef fusion = schedule.getFusion().getValue();
    if (!epsilonAttr)
      epsilonAttr = FloatAttr::get(Float32Type::get(ctx), 1.0e-6);

    KernelCtx c{module.getLoc(),
                in0,
                in1,
                outCb,
                /*readerTrip=*/kTiles,
                /*computeTrip=*/kTiles,
                /*writerTrip=*/mTiles * nTiles,
                /*numDests=*/static_cast<int64_t>(schedule.getGridY()),
                /*hasRmsnorm=*/fusion == "fused_rmsnorm_matmul" ||
                    fusion == "fused_mlp",
                /*hasGelu=*/fusion == "fused_mlp",
                /*multicast=*/schedule.getWeightStrategy().getValue() ==
                    "multicast",
                epsilonAttr};

    // Build the schedule IR: a sched.kernel holding reader / compute / writer.
    OpBuilder b(module.getBodyRegion());
    b.setInsertionPointToEnd(module.getBody());
    auto kernel = b.create<TensixflowSchedKernelOp>(c.loc, StringRef(graphId));
    Block *kernelBlock = b.createBlock(&kernel.getBody());

    buildReader(b, kernelBlock, c);
    buildCompute(b, kernelBlock, c);
    buildWriter(b, kernelBlock, c);

    // Trip counts alone cannot recover the element dimensions, and the host
    // emitter sizes the DRAM buffers and the CPU reference matmul from them, so
    // record M, N, K on the kernel.
    kernel->setAttr(kMatmulShapeAttrName,
                    b.getDenseI64ArrayAttr({m, n, k}));

    // The graph function has been lowered into the kernel; erase it so the kernel
    // is the sole symbol named graphId (the module is a SymbolTable and would
    // otherwise reject the duplicate).
    if (funcOp)
      funcOp->erase();
  }

private:
  // Reader: per K-tile, stage one activation tile into in0 and one weight tile
  // into in1 (multicast across the core row when the weight strategy is
  // "multicast", else a plain NoC read).
  void buildReader(OpBuilder &b, Block *kernelBlock, const KernelCtx &c) {
    auto reader = b.create<TensixflowSchedReaderOp>(c.loc);
    b.createBlock(&reader.getBody());
    auto loop = b.create<TensixflowSchedTileLoopOp>(c.loc, c.readerTrip);
    b.createBlock(&loop.getBody());
    b.create<TensixflowSchedCbReserveBackOp>(c.loc, c.in0, 1);
    b.create<TensixflowSchedNocReadOp>(c.loc, kSrcAddr, c.in0, 1);
    b.create<TensixflowSchedCbPushBackOp>(c.loc, c.in0, 1);
    b.create<TensixflowSchedCbReserveBackOp>(c.loc, c.in1, 1);
    if (c.multicast)
      b.create<TensixflowSchedNocMulticastOp>(c.loc, c.in1, kDstAddr, 1,
                                              c.numDests);
    else
      b.create<TensixflowSchedNocReadOp>(c.loc, kSrcAddr, c.in1, 1);
    b.create<TensixflowSchedCbPushBackOp>(c.loc, c.in1, 1);
    b.setInsertionPointToEnd(kernelBlock);
  }

  // Compute: per K-tile, wait on the input CBs, run the RMSNorm prologue (fused
  // ops), the tile matmul (in0 x in1 -> out), an optional GELU epilogue (fused
  // mlp), then release the consumed input tiles.
  void buildCompute(OpBuilder &b, Block *kernelBlock, const KernelCtx &c) {
    auto compute = b.create<TensixflowSchedComputeOp>(c.loc);
    b.createBlock(&compute.getBody());
    auto loop = b.create<TensixflowSchedTileLoopOp>(c.loc, c.computeTrip);
    b.createBlock(&loop.getBody());
    b.create<TensixflowSchedCbWaitFrontOp>(c.loc, c.in0, 1);
    b.create<TensixflowSchedCbWaitFrontOp>(c.loc, c.in1, 1);
    if (c.hasRmsnorm)
      b.create<TensixflowSchedRmsnormOp>(
          c.loc, b.getI64IntegerAttr(c.in0), b.getI64IntegerAttr(kCBGamma),
          b.getI64IntegerAttr(c.in0), c.epsilon);
    b.create<TensixflowSchedMatmulTilesOp>(c.loc, c.in0, c.in1, c.outCb);
    if (c.hasGelu)
      b.create<TensixflowSchedGeluOp>(c.loc, c.outCb, c.outCb);
    b.create<TensixflowSchedCbPopFrontOp>(c.loc, c.in0, 1);
    b.create<TensixflowSchedCbPopFrontOp>(c.loc, c.in1, 1);
    b.setInsertionPointToEnd(kernelBlock);
  }

  // Writer: per output tile, wait on the out CB, drain it to the destination over
  // the NoC, and release it.
  void buildWriter(OpBuilder &b, Block *kernelBlock, const KernelCtx &c) {
    auto writer = b.create<TensixflowSchedWriterOp>(c.loc);
    b.createBlock(&writer.getBody());
    auto loop = b.create<TensixflowSchedTileLoopOp>(c.loc, c.writerTrip);
    b.createBlock(&loop.getBody());
    b.create<TensixflowSchedCbWaitFrontOp>(c.loc, c.outCb, 1);
    b.create<TensixflowSchedNocWriteOp>(c.loc, c.outCb, kDstAddr, 1);
    b.create<TensixflowSchedCbPopFrontOp>(c.loc, c.outCb, 1);
    b.setInsertionPointToEnd(kernelBlock);
  }
};

} // namespace
} // namespace tensixflow
} // namespace mlir
