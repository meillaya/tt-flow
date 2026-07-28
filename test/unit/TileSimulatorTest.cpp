//===- TileSimulatorTest.cpp -------------------------------------*- C++ -*-===//
//
// Unit tests for the TensixFlow functional tile simulator (todo 39, draft D14) in
// lib/tensixflow/Sim/TileSimulator.cpp. The simulator EXECUTES the tile-level
// schedule IR (sched.kernel) over modeled bf16 circular buffers -- it does NOT
// parse/execute the emitted C++. These tests prove:
//
//   * a 64x64 matmul (lowered from a lone-matmul func via the full
//     plan -> allocate -> lower pipeline) matches an inline fp32 reference within
//     bf16 tolerance;
//   * a 1x64 rmsnorm (hand-built schedule IR -- the lowering has no lone-rmsnorm
//     case) matches an inline fp32 reference within bf16 tolerance;
//   * a tiny fused rmsnorm_matmul runs END-TO-END from schedule IR (full pipeline)
//     and matches the composed fp32 reference within bf16 tolerance;
//   * NON-VACUITY: the simulator output is NOT bit-identical to the fp32 reference
//     (maxAbsDiff > 0 -- bf16 rounding was applied) AND every output value is
//     bf16-representable (CB storage rounding actually happened);
//   * feeding schedule IR that references an unallocated CB yields a clear error
//     (not silent garbage), as does an out-of-range CB index.
//
// The bf16 tolerance (kAbsTol + kRelTol*|ref|) absorbs the deliberate divergence
// between the simulator's bf16-spatial path (round-on-push CB storage, fp32
// accumulate within a tile, bf16 round per output tile / fused intermediate) and
// the fp32-exact reference. bf16 has 7 mantissa bits (RNE max relative error
// 2^-8 ~= 3.9e-3); the bounds below are a small multiple of that, scaled by the
// K=64 accumulation depth. The observed maxAbsDiff/maxRelDiff are printed so the
// bounds can be calibrated; they are documented in .omo/notepads/tensixflow.
//
//===----------------------------------------------------------------------===//

#include "TileSimulator.h"
#include "Rounding.h"

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

#include "gtest/gtest.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace tf = mlir::tensixflow;
namespace sim = mlir::tensixflow::sim;

namespace {

// Combined bf16 tolerance: |sim - ref| <= kAbsTol + kRelTol * |ref|. kRelTol is a
// small multiple of bf16 epsilon (2^-8 ~= 3.9e-3) scaled by the K=64 accumulation;
// kAbsTol covers near-zero outputs where cancellation blows up relative error.
constexpr float kAbsTol = 0.1f;
constexpr float kRelTol = 0.05f;

// A fixed-seed 32-bit LCG (Numerical Recipes constants). The top 24 bits fill a full
// fp32 mantissa, so the generated values are NOT bf16-representable -- guaranteeing
// the simulator's round-on-push actually changes them (the non-vacuity invariant).
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

//===----------------------------------------------------------------------===//
// fp32-EXACT references (the golden semantics, todo 40: no intermediate rounding).
//===----------------------------------------------------------------------===//

// C[M,N] = A[M,K] @ B[K,N], fp32 accumulate, no rounding.
sim::Tensor refMatmul(const sim::Tensor &a, const sim::Tensor &b) {
  int64_t m = a.dim(0), k = a.dim(1), n = b.dim(1);
  sim::Tensor c{{m, n}, std::vector<float>((size_t)(m * n), 0.0f)};
  for (int64_t i = 0; i < m; ++i)
    for (int64_t j = 0; j < n; ++j) {
      float s = 0.0f;
      for (int64_t p = 0; p < k; ++p)
        s += a.data[(size_t)(i * k + p)] * b.data[(size_t)(p * n + j)];
      c.data[(size_t)(i * n + j)] = s;
    }
  return c;
}

// y[M,K] = x * rsqrt(mean_k(x^2) + eps) * gamma, fp32, no rounding.
sim::Tensor refRmsnorm(const sim::Tensor &x, const sim::Tensor &gamma, float eps) {
  int64_t m = x.dim(0), k = x.dim(1);
  sim::Tensor y{{m, k}, std::vector<float>((size_t)(m * k))};
  for (int64_t i = 0; i < m; ++i) {
    float ss = 0.0f;
    for (int64_t p = 0; p < k; ++p)
      ss += x.data[(size_t)(i * k + p)] * x.data[(size_t)(i * k + p)];
    float rms = 1.0f / std::sqrt(ss / (float)k + eps);
    for (int64_t p = 0; p < k; ++p)
      y.data[(size_t)(i * k + p)] = x.data[(size_t)(i * k + p)] * rms * gamma.data[(size_t)p];
  }
  return y;
}

//===----------------------------------------------------------------------===//
// Comparison + assertions.
//===----------------------------------------------------------------------===//

struct DiffStats {
  float maxAbs = 0.0f;
  float maxRel = 0.0f;
  float worstViol = -std::numeric_limits<float>::infinity(); // max(|sim-ref| - bound)
  bool allBf16 = true;
};

DiffStats compare(const sim::Tensor &got, const sim::Tensor &ref) {
  DiffStats s;
  for (size_t i = 0; i < got.data.size(); ++i) {
    float a = std::abs(got.data[i] - ref.data[i]);
    s.maxAbs = std::max(s.maxAbs, a);
    s.maxRel = std::max(s.maxRel, a / std::max(std::abs(ref.data[i]), 1e-6f));
    s.worstViol = std::max(s.worstViol, a - (kAbsTol + kRelTol * std::abs(ref.data[i])));
    if (sim::to_bf16(got.data[i]) != got.data[i])
      s.allBf16 = false;
  }
  return s;
}

void expectWithinTol(const DiffStats &s, const char *name) {
  std::cout << "[" << name << "] maxAbsDiff=" << s.maxAbs << " maxRelDiff=" << s.maxRel
            << " allBf16=" << s.allBf16 << "\n";
  EXPECT_LT(s.worstViol, 0.0f) << name << ": an element exceeds the combined bf16 tolerance";
  // NON-VACUITY: the bf16-spatial simulator must DIFFER from the fp32-exact ref
  // (bit-identical output would mean the bf16 rounding was skipped -- PCC -> 1.0).
  EXPECT_GT(s.maxAbs, 0.0f) << name << ": simulator output is bit-identical to the fp32 ref (vacuous)";
  EXPECT_TRUE(s.allBf16) << name << ": simulator output has non-bf16 values (CB rounding skipped)";
}

//===----------------------------------------------------------------------===//
// Fixture: an MLIRContext with the tensixflow + func + arith dialects loaded.
//===----------------------------------------------------------------------===//

class TileSimulatorTest : public ::testing::Test {
protected:
  void SetUp() override {
    registry.insert<tf::TensixflowDialect, mlir::func::FuncDialect,
                    mlir::arith::ArithDialect>();
    ctx.appendDialectRegistry(registry);
    ctx.loadAllAvailableDialects();
  }

  mlir::OwningOpRef<mlir::ModuleOp> parseModule(llvm::StringRef text) {
    return mlir::parseSourceString<mlir::ModuleOp>(text, &ctx);
  }

  // Run the full plan -> allocate -> lower pipeline and return the sched.kernel.
  tf::TensixflowSchedKernelOp lowerToKernel(mlir::ModuleOp module) {
    mlir::PassManager pm(module.getContext());
    pm.addPass(tf::createPlanNocShardingPass());
    pm.addPass(tf::createAllocateCircularBuffersPass());
    pm.addPass(tf::createLowerToTTKernelPass());
    if (mlir::failed(pm.run(module)))
      return {};
    tf::TensixflowSchedKernelOp kernel;
    module.walk([&](tf::TensixflowSchedKernelOp op) { kernel = op; });
    return kernel;
  }

  mlir::MLIRContext ctx;
  mlir::DialectRegistry registry;
};

//===----------------------------------------------------------------------===//
// Happy path: 64x64 matmul via the full pipeline.
//===----------------------------------------------------------------------===//

TEST_F(TileSimulatorTest, Matmul64x64) {
  auto module = parseModule(R"mlir(
    tensixflow.func @matmul(%a: tensor<64x64xbf16>, %b: tensor<64x64xbf16>) -> tensor<64x64xbf16> {
      %0 = tensixflow.matmul %a, %b : tensor<64x64xbf16> x tensor<64x64xbf16> -> tensor<64x64xbf16>
      tensixflow.return %0 : tensor<64x64xbf16>
    }
  )mlir");
  ASSERT_TRUE(module) << "failed to parse the lone-matmul func";
  auto kernel = lowerToKernel(*module);
  ASSERT_TRUE(kernel) << "pipeline failed to produce a sched.kernel";

  Lcg lcg(0x20260727u);
  sim::Tensor a = makeTensor({64, 64}, -0.5f, 0.5f, lcg);
  sim::Tensor b = makeTensor({64, 64}, -0.5f, 0.5f, lcg);

  sim::SimInputs inputs;
  inputs.activation = a;
  inputs.weight = b;
  inputs.allocatedCBs = sim::collectReferencedCBs(kernel);

  auto result = sim::runTileSimulator(kernel, inputs);
  if (!result)
    FAIL() << "simulator failed: " << llvm::toString(result.takeError());
  sim::Tensor out = std::move(*result);

  sim::Tensor ref = refMatmul(a, b);
  ASSERT_EQ(out.numElements(), ref.numElements());
  expectWithinTol(compare(out, ref), "Matmul64x64");
}

//===----------------------------------------------------------------------===//
// Happy path: 1x64 rmsnorm via hand-built schedule IR (no lone-rmsnorm lowering).
//===----------------------------------------------------------------------===//

TEST_F(TileSimulatorTest, Rmsnorm1x64) {
  // reader stages activation -> c_0 and gamma -> c_2; compute runs rmsnorm 0,2,16;
  // writer drains c_16. epsilon defaults to 1e-6 (elided in the pretty form).
  auto module = parseModule(R"mlir(
    tensixflow.sched.kernel @rmsnorm {
      tensixflow.sched.reader {
        tensixflow.sched.tile_loop 2 {
          tensixflow.sched.cb_reserve_back 0, 1
          tensixflow.sched.noc_read 0, 0, 1
          tensixflow.sched.cb_push_back 0, 1
          tensixflow.sched.cb_reserve_back 2, 1
          tensixflow.sched.noc_read 0, 2, 1
          tensixflow.sched.cb_push_back 2, 1
        }
      }
      tensixflow.sched.compute {
        tensixflow.sched.tile_loop 2 {
          tensixflow.sched.cb_wait_front 0, 1
          tensixflow.sched.cb_wait_front 2, 1
          tensixflow.sched.rmsnorm 0, 2, 16
          tensixflow.sched.cb_pop_front 0, 1
          tensixflow.sched.cb_pop_front 2, 1
        }
      }
      tensixflow.sched.writer {
        tensixflow.sched.tile_loop 2 {
          tensixflow.sched.cb_wait_front 16, 1
          tensixflow.sched.noc_write 16, 0, 1
          tensixflow.sched.cb_pop_front 16, 1
        }
      }
    }
  )mlir");
  ASSERT_TRUE(module) << "failed to parse the hand-built rmsnorm schedule IR";
  tf::TensixflowSchedKernelOp kernel;
  module->walk([&](tf::TensixflowSchedKernelOp op) { kernel = op; });
  ASSERT_TRUE(kernel);

  Lcg lcg(0x20260728u);
  sim::Tensor x = makeTensor({1, 64}, -0.5f, 0.5f, lcg);
  sim::Tensor gamma = makeTensor({64}, 0.5f, 1.5f, lcg);

  sim::SimInputs inputs;
  inputs.activation = x;
  inputs.gamma = gamma;
  inputs.allocatedCBs = {0, 2, 16};

  auto result = sim::runTileSimulator(kernel, inputs);
  if (!result)
    FAIL() << "simulator failed: " << llvm::toString(result.takeError());
  sim::Tensor out = std::move(*result);

  sim::Tensor ref = refRmsnorm(x, gamma, 1.0e-6f);
  ASSERT_EQ(out.numElements(), ref.numElements());
  expectWithinTol(compare(out, ref), "Rmsnorm1x64");
}

//===----------------------------------------------------------------------===//
// Happy path: tiny fused rmsnorm_matmul, end-to-end from schedule IR (pipeline).
//===----------------------------------------------------------------------===//

TEST_F(TileSimulatorTest, FusedRmsnormMatmul) {
  // The unfused rmsnorm -> matmul pair is planned AS fused_rmsnorm_matmul (the
  // planner's case 2). Activation [1,64,64] flattens to [M=64, K=64].
  auto module = parseModule(R"mlir(
    tensixflow.func @fused(%x: tensor<1x64x64xbf16>, %gamma: tensor<64xbf16>, %w: tensor<64x64xbf16>) -> tensor<1x64x64xbf16> {
      %0 = tensixflow.rmsnorm %x, %gamma : tensor<1x64x64xbf16>, tensor<64xbf16> -> tensor<1x64x64xbf16>
      %1 = tensixflow.matmul %0, %w : tensor<1x64x64xbf16> x tensor<64x64xbf16> -> tensor<1x64x64xbf16>
      tensixflow.return %1 : tensor<1x64x64xbf16>
    }
  )mlir");
  ASSERT_TRUE(module) << "failed to parse the fused func";
  auto kernel = lowerToKernel(*module);
  ASSERT_TRUE(kernel) << "pipeline failed to produce a sched.kernel";

  Lcg lcg(0x20260729u);
  sim::Tensor x = makeTensor({64, 64}, -0.5f, 0.5f, lcg);
  sim::Tensor gamma = makeTensor({64}, 0.5f, 1.5f, lcg);
  sim::Tensor w = makeTensor({64, 64}, -0.5f, 0.5f, lcg);

  sim::SimInputs inputs;
  inputs.activation = x;
  inputs.weight = w;
  inputs.gamma = gamma;
  inputs.allocatedCBs = sim::collectReferencedCBs(kernel); // includes the gamma c_2

  auto result = sim::runTileSimulator(kernel, inputs);
  if (!result)
    FAIL() << "simulator failed: " << llvm::toString(result.takeError());
  sim::Tensor out = std::move(*result);

  sim::Tensor norm = refRmsnorm(x, gamma, 1.0e-6f);
  sim::Tensor ref = refMatmul(norm, w);
  ASSERT_EQ(out.numElements(), ref.numElements());
  expectWithinTol(compare(out, ref), "FusedRmsnormMatmul");
}

//===----------------------------------------------------------------------===//
// Failure path: schedule IR referencing an unallocated CB -> clear error.
//===----------------------------------------------------------------------===//

TEST_F(TileSimulatorTest, UnallocatedCBError) {
  // A structurally-valid matmul kernel referencing c_0, c_1, c_16.
  auto module = parseModule(R"mlir(
    tensixflow.sched.kernel @matmul {
      tensixflow.sched.reader {
        tensixflow.sched.tile_loop 1 {
          tensixflow.sched.cb_reserve_back 0, 1
          tensixflow.sched.noc_read 0, 0, 1
          tensixflow.sched.cb_push_back 0, 1
          tensixflow.sched.cb_reserve_back 1, 1
          tensixflow.sched.noc_read 0, 1, 1
          tensixflow.sched.cb_push_back 1, 1
        }
      }
      tensixflow.sched.compute {
        tensixflow.sched.tile_loop 1 {
          tensixflow.sched.cb_wait_front 0, 1
          tensixflow.sched.cb_wait_front 1, 1
          tensixflow.sched.matmul_tiles 0, 1, 16
          tensixflow.sched.cb_pop_front 0, 1
          tensixflow.sched.cb_pop_front 1, 1
        }
      }
      tensixflow.sched.writer {
        tensixflow.sched.tile_loop 1 {
          tensixflow.sched.cb_wait_front 16, 1
          tensixflow.sched.noc_write 16, 0, 1
          tensixflow.sched.cb_pop_front 16, 1
        }
      }
    }
  )mlir");
  ASSERT_TRUE(module);
  tf::TensixflowSchedKernelOp kernel;
  module->walk([&](tf::TensixflowSchedKernelOp op) { kernel = op; });
  ASSERT_TRUE(kernel);

  Lcg lcg(1u);
  sim::SimInputs inputs;
  inputs.activation = makeTensor({32, 32}, -0.5f, 0.5f, lcg);
  inputs.weight = makeTensor({32, 32}, -0.5f, 0.5f, lcg);
  inputs.allocatedCBs = {0, 1}; // MISSING c_16 -> unallocated

  auto result = sim::runTileSimulator(kernel, inputs);
  ASSERT_FALSE(result) << "expected an unallocated-CB error, got a tensor";
  std::string msg = llvm::toString(result.takeError());
  EXPECT_NE(msg.find("unallocated circular buffer c_16"), std::string::npos) << "msg=" << msg;
}

//===----------------------------------------------------------------------===//
// Failure path: an out-of-range CB index (outside c_0..c_31) -> clear error.
//===----------------------------------------------------------------------===//

TEST_F(TileSimulatorTest, OutOfRangeCBError) {
  auto module = parseModule(R"mlir(
    tensixflow.sched.kernel @bad {
      tensixflow.sched.compute {
        tensixflow.sched.tile_loop 1 {
          tensixflow.sched.cb_wait_front 99, 1
          tensixflow.sched.matmul_tiles 0, 1, 16
          tensixflow.sched.cb_pop_front 99, 1
        }
      }
      tensixflow.sched.writer {
        tensixflow.sched.tile_loop 1 {
          tensixflow.sched.noc_write 16, 0, 1
        }
      }
    }
  )mlir");
  ASSERT_TRUE(module);
  tf::TensixflowSchedKernelOp kernel;
  module->walk([&](tf::TensixflowSchedKernelOp op) { kernel = op; });
  ASSERT_TRUE(kernel);

  Lcg lcg(2u);
  sim::SimInputs inputs;
  inputs.activation = makeTensor({32, 32}, -0.5f, 0.5f, lcg);
  inputs.weight = makeTensor({32, 32}, -0.5f, 0.5f, lcg);
  inputs.allocatedCBs = {0, 1, 16, 99}; // even "allocated", 99 is out of range

  auto result = sim::runTileSimulator(kernel, inputs);
  ASSERT_FALSE(result) << "expected an out-of-range CB error, got a tensor";
  std::string msg = llvm::toString(result.takeError());
  EXPECT_NE(msg.find("outside the valid tt-metal range"), std::string::npos) << "msg=" << msg;
}

} // namespace
