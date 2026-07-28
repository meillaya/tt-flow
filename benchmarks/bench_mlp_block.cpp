//===- bench_mlp_block.cpp - MLP block bench --------------------*- C++ -*-===//
//
// TensixFlow benchmark (todo 44, D21): a SIMPLE timing loop for the fused MLP
// block (rmsnorm -> matmul(w1) -> add(b1) -> gelu -> matmul(w2) -> add(residual))
// -- NOT a benchmarking framework (no Google Benchmark, no statistics engine).
// Two parts:
//
//   1. HEADLINE FIGURES -- the static perf model (todo 42). buildPlanReport
//      (todo 30) assembles the plan for the example shapes
//      (activation [1,2048,4096] @ w1 [4096,12288]); estimatePerf runs the NoC
//      estimator for BOTH the unfused and fused paths; formatPerfReport prints
//      the user's report (unfused vs fused DRAM reads/writes, kernel launches,
//      predicted bottleneck, suggested optimizations). The model characterizes
//      the DOMINANT matmul (w1) of the block. These figures are STATIC ANALYTIC
//      ESTIMATES -- NOT measured wall-clock hardware performance.
//
//   2. SIMULATOR TIMING LOOP -- a functional-execution measure. The full
//      schedule-producing pipeline fuses the reduced chain into a
//      `tensixflow.fused_mlp` and lowers it to a sched.kernel; runTileSimulator
//      (todo 39) executes it in a plain std::chrono loop. DOCUMENTED SCOPE: the
//      simulator executes ONLY the compute prefix rmsnorm -> matmul(w1) -> gelu
//      (the lowering emits no schedule ops for bias(b1)/matmul(w2)/residual --
//      the documented MLP scope reduction, equivalent to identity trailing
//      operands). The microseconds measure the functional simulator interpreting
//      the tile program on the host CPU -- NOT wall-clock hardware performance.
//
//===----------------------------------------------------------------------===//

#include "PerfModel.h"
#include "PlanReport.h"
#include "TileSimulator.h"

#include "tensixflow/TensixflowDialect.h"
#include "tensixflow/TensixflowPasses.h"
#include "tensixflow/TensixflowSchedule.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace tf = mlir::tensixflow;
namespace sim = mlir::tensixflow::sim;

namespace {

// The example graph's operand shapes (examples/mlp_block.mlir): the static perf
// model is analytic, so it runs on the FULL shapes; the weight is w1 (the first,
// dominant matmul of the block).
const std::vector<int64_t> kActivationShape = {1, 2048, 4096};
const std::vector<int64_t> kW1Shape = {4096, 12288};

// Reduced feasible shapes for the functional tile simulator (a non-BLAS tiled
// interpreter): the schedule IR is built from this func, and the simulator
// derives its per-tile grid from these input shapes.
constexpr int64_t kSimM = 64;
constexpr int64_t kSimK = 64;
constexpr int64_t kSimN = 64;
constexpr int kSimIterations = 20;

// A fixed-seed 32-bit LCG (Numerical Recipes constants) so the simulator inputs
// are deterministic across runs.
struct Lcg {
  uint32_t state;
  explicit Lcg(uint32_t seed) : state(seed) {}
  float next(float lo, float hi) {
    state = state * 1664525u + 1013904223u; // mod 2^32 (unsigned wrap)
    float u = (float)(state >> 8) * (1.0f / 16777216.0f); // [0,1), 24-bit mantissa
    return lo + (hi - lo) * u;
  }
};

sim::Tensor makeTensor(std::vector<int64_t> shape, float lo, float hi, Lcg &lcg) {
  sim::Tensor t;
  t.shape = std::move(shape);
  int64_t n = 1;
  for (int64_t d : t.shape)
    n *= d;
  t.data.reserve((size_t)n);
  for (int64_t i = 0; i < n; ++i)
    t.data.push_back(lcg.next(lo, hi));
  return t;
}

int fail(const std::string &msg) {
  llvm::errs() << "bench_mlp_block: error: " << msg << "\n";
  return 1;
}

// Part 1: the headline static perf-model report (unfused vs fused).
void runPerfModel() {
  tf::scheduler::PlanRequest req;
  req.graphId = "mlp_block";
  req.activation = tf::scheduler::TensorShape{"x", kActivationShape, "bf16"};
  req.weight = tf::scheduler::TensorShape{"w1", kW1Shape, "bf16"};
  req.fusion = "fused_mlp";
  req.fused = true;
  req.bytesPerElem = 2;

  const tf::scheduler::PlanReport planReport = tf::scheduler::buildPlanReport(req);
  const tf::analysis::PerfReport perf =
      tf::analysis::estimatePerf(planReport.plan);

  // All benchmark output goes through llvm::outs() (the same single stream the
  // planning passes print on) so the ordering is deterministic.
  llvm::outs() << "=== bench_mlp_block ===\n"
               << "graph: mlp_block (activation [1,2048,4096] @ w1 "
                  "[4096,12288], bf16; dominant matmul of the fused MLP)\n"
               << "fusion: fused_mlp\n\n"
               << tf::analysis::formatPerfReport(perf)
               << "\n"
               << "NOTE: the figures above are STATIC perf-model estimates "
                  "(analytic roofline over the plan's tensor shapes), NOT "
                  "measured wall-clock hardware performance.\n";
}

// Part 2: a simple timing loop over the functional tile simulator (fused path).
int runSimulatorLoop() {
  // The reduced full MLP chain (mirrors examples/mlp_block.mlir at
  // simulator-feasible shapes); the fuse pass folds it into fused_mlp.
  constexpr llvm::StringRef kMlir = R"mlir(
    tensixflow.func @mlp_block(%x: tensor<1x64x64xbf16>, %gamma: tensor<64xbf16>, %w1: tensor<64x64xbf16>, %b1: tensor<1x64x64xbf16>, %w2: tensor<64x64xbf16>, %residual: tensor<1x64x64xbf16>) -> tensor<1x64x64xbf16> {
      %0 = tensixflow.rmsnorm %x, %gamma : tensor<1x64x64xbf16>, tensor<64xbf16> -> tensor<1x64x64xbf16>
      %1 = tensixflow.matmul %0, %w1 : tensor<1x64x64xbf16> x tensor<64x64xbf16> -> tensor<1x64x64xbf16>
      %2 = tensixflow.add %1, %b1 : tensor<1x64x64xbf16>
      %3 = tensixflow.gelu %2 : tensor<1x64x64xbf16>
      %4 = tensixflow.matmul %3, %w2 : tensor<1x64x64xbf16> x tensor<64x64xbf16> -> tensor<1x64x64xbf16>
      %5 = tensixflow.add %4, %residual : tensor<1x64x64xbf16>
      tensixflow.return %5 : tensor<1x64x64xbf16>
    }
  )mlir";

  mlir::DialectRegistry registry;
  registry.insert<tf::TensixflowDialect, mlir::func::FuncDialect,
                  mlir::arith::ArithDialect>();
  mlir::MLIRContext ctx(registry);
  ctx.loadAllAvailableDialects();

  auto module = mlir::parseSourceString<mlir::ModuleOp>(kMlir, &ctx);
  if (!module)
    return fail("failed to parse the reduced mlp_block func");

  // The full schedule-producing pipeline (the same passes tensixflow-simulate
  // runs): infer-layout -> fuse -> plan -> allocate -> lower.
  mlir::PassManager pm(&ctx);
  pm.addPass(tf::createInferTileLayoutPass());
  pm.addPass(tf::createFuseRmsnormMatmulPass());
  pm.addPass(tf::createPlanNocShardingPass());
  pm.addPass(tf::createAllocateCircularBuffersPass());
  pm.addPass(tf::createLowerToTTKernelPass());
  if (mlir::failed(pm.run(*module)))
    return fail("schedule pipeline failed on the reduced func");

  tf::TensixflowSchedKernelOp kernel;
  module->walk([&](tf::TensixflowSchedKernelOp op) { kernel = op; });
  if (!kernel)
    return fail("pipeline produced no tensixflow.sched.kernel");

  // The simulator reads the activation as [M, K] (the func's [1,64,64] flattens
  // to [M=64, K=64] -- the same convention the correctness harness uses).
  Lcg lcg(0x20260729u);
  sim::SimInputs inputs;
  inputs.activation = makeTensor({kSimM, kSimK}, -0.5f, 0.5f, lcg);
  inputs.weight = makeTensor({kSimK, kSimN}, -0.5f, 0.5f, lcg); // w1
  inputs.gamma = makeTensor({kSimK}, 0.5f, 1.5f, lcg);
  inputs.allocatedCBs = sim::collectReferencedCBs(kernel);

  // The simple timing loop: run the functional simulator kSimIterations times.
  // The simulator executes the fused compute PREFIX (rmsnorm -> matmul(w1) ->
  // gelu); b1/w2/residual are the documented identity trailing operands.
  int64_t outElements = 0;
  double totalUs = 0.0;
  for (int i = 0; i < kSimIterations; ++i) {
    const auto start = std::chrono::steady_clock::now();
    auto result = sim::runTileSimulator(kernel, inputs);
    const auto end = std::chrono::steady_clock::now();
    if (!result)
      return fail("simulator failed: " + llvm::toString(result.takeError()));
    outElements = result->numElements();
    totalUs += std::chrono::duration<double, std::micro>(end - start).count();
  }

  llvm::outs() << "\nfunctional tile simulator (fused path, compute prefix "
                  "rmsnorm->matmul(w1)->gelu), reduced shapes [M=64, K=64] @ "
                  "[K=64, N=64], "
               << kSimIterations << " runs\n"
               << "  output elements: " << outElements << "\n"
               << "  mean functional-execution time: "
               << (totalUs / kSimIterations) << " us/run\n"
               << "NOTE: this is a functional-execution measure (the tile "
                  "simulator interpreting the schedule IR on the host CPU), NOT "
                  "wall-clock hardware performance.\n";
  return 0;
}

} // namespace

int main() {
  runPerfModel();
  return runSimulatorLoop();
}
