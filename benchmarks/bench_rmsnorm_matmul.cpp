//===- bench_rmsnorm_matmul.cpp - RMSNorm+MatMul bench ----------*- C++ -*-===//
//
// TensixFlow benchmark (todo 44, D21): a SIMPLE timing loop for the fused
// RMSNorm+MatMul graph -- NOT a benchmarking framework (no Google Benchmark, no
// statistics engine, no registry). Two parts:
//
//   1. HEADLINE FIGURES -- the static perf model (todo 42). buildPlanReport
//      (todo 30) assembles the plan for the example shapes
//      (activation [1,2048,4096] @ weight [4096,12288]); estimatePerf runs the
//      NoC estimator for BOTH the unfused and fused paths; formatPerfReport
//      prints the user's report (unfused vs fused DRAM reads/writes, kernel
//      launches, predicted bottleneck, suggested optimizations). These figures
//      are STATIC ANALYTIC ESTIMATES from the documented roofline-style model --
//      they are NOT measured wall-clock hardware performance and must never be
//      reported as such.
//
//   2. SIMULATOR TIMING LOOP -- a functional-execution measure. The full
//      schedule-producing pipeline (infer-layout -> fuse -> plan -> allocate ->
//      lower) builds the fused `tensixflow.sched.kernel` from a REDUCED func
//      ([1,64,64] @ [64,64] -- the example shapes are far too large for the
//      non-BLAS tiled simulator), and runTileSimulator (todo 39) executes it
//      over modeled bf16 circular buffers in a plain std::chrono loop. The
//      resulting microseconds measure how long the FUNCTIONAL SIMULATOR takes to
//      interpret the tile program on this host CPU -- they say NOTHING about
//      hardware latency/throughput and are NOT wall-clock hardware performance.
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

// The example graph's operand shapes (examples/rmsnorm_matmul.mlir): the static
// perf model is analytic, so it runs on the FULL shapes.
const std::vector<int64_t> kActivationShape = {1, 2048, 4096};
const std::vector<int64_t> kWeightShape = {4096, 12288};

// Reduced feasible shapes for the functional tile simulator (a non-BLAS tiled
// interpreter): the schedule IR is built from this func, and the simulator
// derives its per-tile grid from these input shapes (the IR's trip counts are
// first-order only; the simulator reconciles them with the shapes it is given).
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
  llvm::errs() << "bench_rmsnorm_matmul: error: " << msg << "\n";
  return 1;
}

// Part 1: the headline static perf-model report (unfused vs fused).
void runPerfModel() {
  tf::scheduler::PlanRequest req;
  req.graphId = "rmsnorm_matmul";
  req.activation = tf::scheduler::TensorShape{"x", kActivationShape, "bf16"};
  req.weight = tf::scheduler::TensorShape{"w", kWeightShape, "bf16"};
  req.fusion = "fused_rmsnorm_matmul";
  req.fused = true;
  req.bytesPerElem = 2;

  const tf::scheduler::PlanReport planReport = tf::scheduler::buildPlanReport(req);
  const tf::analysis::PerfReport perf =
      tf::analysis::estimatePerf(planReport.plan);

  // All benchmark output goes through llvm::outs() (the same single stream the
  // planning passes print on) so the ordering is deterministic.
  llvm::outs() << "=== bench_rmsnorm_matmul ===\n"
               << "graph: rmsnorm_matmul (activation [1,2048,4096] @ weight "
                  "[4096,12288], bf16)\n"
               << "fusion: fused_rmsnorm_matmul\n\n"
               << tf::analysis::formatPerfReport(perf)
               << "\n"
               << "NOTE: the figures above are STATIC perf-model estimates "
                  "(analytic roofline over the plan's tensor shapes), NOT "
                  "measured wall-clock hardware performance.\n";
}

// Part 2: a simple timing loop over the functional tile simulator (fused path).
int runSimulatorLoop() {
  // The reduced fused rmsnorm+matmul func (mirrors examples/rmsnorm_matmul.mlir
  // at simulator-feasible shapes).
  constexpr llvm::StringRef kMlir = R"mlir(
    tensixflow.func @rmsnorm_matmul(%x: tensor<1x64x64xbf16>, %gamma: tensor<64xbf16>, %w: tensor<64x64xbf16>) -> tensor<1x64x64xbf16> {
      %0 = tensixflow.rmsnorm %x, %gamma : tensor<1x64x64xbf16>, tensor<64xbf16> -> tensor<1x64x64xbf16>
      %1 = tensixflow.matmul %0, %w : tensor<1x64x64xbf16> x tensor<64x64xbf16> -> tensor<1x64x64xbf16>
      tensixflow.return %1 : tensor<1x64x64xbf16>
    }
  )mlir";

  mlir::DialectRegistry registry;
  registry.insert<tf::TensixflowDialect, mlir::func::FuncDialect,
                  mlir::arith::ArithDialect>();
  mlir::MLIRContext ctx(registry);
  ctx.loadAllAvailableDialects();

  auto module = mlir::parseSourceString<mlir::ModuleOp>(kMlir, &ctx);
  if (!module)
    return fail("failed to parse the reduced rmsnorm_matmul func");

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
  Lcg lcg(0x20260728u);
  sim::SimInputs inputs;
  inputs.activation = makeTensor({kSimM, kSimK}, -0.5f, 0.5f, lcg);
  inputs.weight = makeTensor({kSimK, kSimN}, -0.5f, 0.5f, lcg);
  inputs.gamma = makeTensor({kSimK}, 0.5f, 1.5f, lcg);
  inputs.allocatedCBs = sim::collectReferencedCBs(kernel);

  // The simple timing loop: run the functional simulator kSimIterations times.
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

  llvm::outs() << "\nfunctional tile simulator (fused path), reduced shapes "
                  "[M=64, K=64] @ [K=64, N=64], "
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
