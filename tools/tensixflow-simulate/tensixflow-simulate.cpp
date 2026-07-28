//===- tensixflow-simulate.cpp - Python<->C++ simulator bridge ---*- C++ -*-===//
//
// TensixFlow: An MLIR Compiler for Tile-Native LLM Kernels on
// Tenstorrent-Style Spatial Accelerators.
//
//===----------------------------------------------------------------------===//
//
// The correctness-harness bridge (todo 41, draft D14). The Python harness
// (python/tensixflow/harness.py) cannot call the C++ functional tile simulator
// (todo 39) directly, so this small tool wraps it end-to-end:
//
//   1. parse an example func (examples/rmsnorm_matmul.mlir / mlp_block.mlir);
//   2. run the full schedule-producing pipeline -- infer-tile-layout ->
//      fuse-rmsnorm-matmul -> plan-noc-sharding -> allocate-circular-buffers ->
//      lower-to-ttkernel -- to build the `tensixflow.sched.kernel` (the structure
//      under test: fusion decision + CB allocation);
//   3. read the input tensors (activation, weight, gamma) the harness wrote to a
//      self-describing binary file (the .tft format below);
//   4. run `sim::runTileSimulator` (the REAL C++ simulator, todo 39) over modeled
//      bf16 circular buffers, allocating exactly the CBs the kernel references
//      (collectReferencedCBs -- includes the rmsnorm gamma c_2);
//   5. write the output tensor (shape + fp32 data) back to a .tft file the harness
//      reads and compares against the fp32-exact golden (todo 40).
//
// This validates the REAL C++ simulator -- it does NOT re-implement it in Python
// and does NOT touch the emitted C++ kernels (those are compile-checked elsewhere).
//
// The simulator derives its per-tile grid from the INPUT tensor shapes it is given
// (the schedule IR carries only first-order trip counts, which the simulator
// reconciles with the shapes -- see TileSimulator.h). The harness therefore runs
// the pipeline on the full example funcs (to get the fusion/CB STRUCTURE) but feeds
// FEASIBLE reduced input shapes (the example [1,2048,4096]@[4096,12288] are far too
// large for the non-BLAS tiled simulator in a test). The trip counts built from the
// example shapes are ignored by the simulator; only the IR structure matters.
//
// .tft ("TensixFlow Tensor") binary format, little-endian, self-describing -- one
// tensor per record, records concatenated:
//     uint32  magic == 0x54465401 ("TFT\x01")
//     uint32  rank
//     int64   shape[rank]
//     float32 data[product(shape)]   (row-major)
// The input file holds THREE records (activation, weight, gamma); the output file
// holds ONE (the simulator output).
//
//===----------------------------------------------------------------------===//

#include "TileSimulator.h"

#include "tensixflow/TensixflowAttrs.h"
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

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace tf = mlir::tensixflow;
namespace sim = mlir::tensixflow::sim;

namespace {

constexpr uint32_t kTensorMagic = 0x54465401u; // "TFT\x01"

// Read one .tft record from `in` into `t`. Returns false on a truncated file or a
// magic mismatch (the harness writes well-formed records; this is a loud guard).
bool readTensor(std::ifstream &in, sim::Tensor &t) {
  uint32_t magic = 0, rank = 0;
  in.read(reinterpret_cast<char *>(&magic), sizeof(magic));
  if (!in || magic != kTensorMagic)
    return false;
  in.read(reinterpret_cast<char *>(&rank), sizeof(rank));
  if (!in)
    return false;
  t.shape.resize(rank);
  int64_t numel = 1;
  for (uint32_t i = 0; i < rank; ++i) {
    in.read(reinterpret_cast<char *>(&t.shape[i]), sizeof(int64_t));
    if (!in)
      return false;
    numel *= t.shape[i];
  }
  t.data.resize((size_t)numel);
  in.read(reinterpret_cast<char *>(t.data.data()),
          (std::streamsize)(numel * sizeof(float)));
  return (bool)in;
}

// Write one .tft record for `t` to `out`.
bool writeTensor(std::ofstream &out, const sim::Tensor &t) {
  uint32_t magic = kTensorMagic, rank = (uint32_t)t.shape.size();
  out.write(reinterpret_cast<const char *>(&magic), sizeof(magic));
  out.write(reinterpret_cast<const char *>(&rank), sizeof(rank));
  for (int64_t d : t.shape)
    out.write(reinterpret_cast<const char *>(&d), sizeof(int64_t));
  out.write(reinterpret_cast<const char *>(t.data.data()),
            (std::streamsize)(t.data.size() * sizeof(float)));
  return (bool)out;
}

int fail(const std::string &msg) {
  std::cerr << "tensixflow-simulate: error: " << msg << "\n";
  return 1;
}

} // namespace

int main(int argc, char **argv) {
  std::string mlirPath, inputPath, outputPath;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto next = [&](std::string &dst) {
      if (i + 1 < argc)
        dst = argv[++i];
    };
    if (arg == "--mlir")
      next(mlirPath);
    else if (arg == "--input")
      next(inputPath);
    else if (arg == "--output")
      next(outputPath);
    else
      return fail("unknown argument '" + arg +
                  "' (expected --mlir <file> --input <in.tft> --output <out.tft>)");
  }
  if (mlirPath.empty() || inputPath.empty() || outputPath.empty())
    return fail("usage: tensixflow-simulate --mlir <file> --input <in.tft> "
                "--output <out.tft>");

  // (1) Parse the example func.
  mlir::DialectRegistry registry;
  registry.insert<tf::TensixflowDialect, mlir::func::FuncDialect,
                  mlir::arith::ArithDialect>();
  mlir::MLIRContext ctx(registry);
  ctx.loadAllAvailableDialects();

  auto module = mlir::parseSourceFile<mlir::ModuleOp>(mlirPath, &ctx);
  if (!module)
    return fail("failed to parse MLIR file '" + mlirPath + "'");

  // (2) Run the full schedule-producing pipeline (the same passes the example
  // headers document, plus plan/allocate/lower that build the sched.kernel).
  mlir::PassManager pm(&ctx);
  pm.addPass(tf::createInferTileLayoutPass());
  pm.addPass(tf::createFuseRmsnormMatmulPass());
  pm.addPass(tf::createPlanNocShardingPass());
  pm.addPass(tf::createAllocateCircularBuffersPass());
  pm.addPass(tf::createLowerToTTKernelPass());
  if (mlir::failed(pm.run(*module)))
    return fail("schedule pipeline failed on '" + mlirPath + "'");

  tf::TensixflowSchedKernelOp kernel;
  module->walk([&](tf::TensixflowSchedKernelOp op) { kernel = op; });
  if (!kernel)
    return fail("pipeline produced no tensixflow.sched.kernel");

  std::string fusion = "unknown";
  if (auto schedule =
          (*module)->getAttrOfType<tf::TensixflowScheduleAttr>("tensixflow.schedule"))
    fusion = schedule.getFusion().getValue().str();

  // (3) Read the three input tensors (activation, weight, gamma).
  std::ifstream in(inputPath, std::ios::binary);
  if (!in)
    return fail("cannot open input file '" + inputPath + "'");
  sim::SimInputs inputs;
  if (!readTensor(in, inputs.activation))
    return fail("bad/missing activation tensor in '" + inputPath + "'");
  if (!readTensor(in, inputs.weight))
    return fail("bad/missing weight tensor in '" + inputPath + "'");
  if (!readTensor(in, inputs.gamma))
    return fail("bad/missing gamma tensor in '" + inputPath + "'");
  in.close();

  // (4) Run the REAL C++ simulator, allocating exactly the referenced CBs.
  inputs.allocatedCBs = sim::collectReferencedCBs(kernel);
  auto result = sim::runTileSimulator(kernel, inputs);
  if (!result)
    return fail("simulator failed: " + llvm::toString(result.takeError()));
  sim::Tensor out = std::move(*result);

  // (5) Write the output tensor.
  std::ofstream ofs(outputPath, std::ios::binary);
  if (!ofs || !writeTensor(ofs, out))
    return fail("cannot write output file '" + outputPath + "'");
  ofs.close();

  // A one-line summary the harness logs (fusion decision + output shape).
  std::cout << "fusion=" << fusion << " kernel=@" << kernel.getSymName().str()
            << " out_shape=[";
  for (size_t i = 0; i < out.shape.size(); ++i)
    std::cout << (i ? "," : "") << out.shape[i];
  std::cout << "] out_elements=" << out.numElements() << "\n";
  return 0;
}
