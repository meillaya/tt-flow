//===- Sim/TileSimulator.cpp - Functional tile simulator ---------*- C++ -*-===//
//
// TensixFlow: An MLIR Compiler for Tile-Native LLM Kernels on
// Tenstorrent-Style Spatial Accelerators.
//
//===----------------------------------------------------------------------===//
//
// See TileSimulator.h for the D14 contract. This file interprets the schedule IR
// (the sched.kernel's reader/compute/writer regions) and executes the denoted tile
// computation over modeled bf16 circular buffers: round-on-push CB storage, fp32
// accumulation within a tile across K-tiles (tiled accumulation order), and a bf16
// round of each output tile / fused intermediate.
//
//===----------------------------------------------------------------------===//

#include "TileSimulator.h"
#include "Rounding.h"

#include "llvm/ADT/SmallVector.h"

#include <cmath>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <system_error>
#include <vector>

namespace mlir::tensixflow::sim {

namespace {

int64_t ceilDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

llvm::Error simError(const Twine &msg) {
  return llvm::createStringError(std::make_error_code(std::errc::invalid_argument),
                                 "tensixflow tile simulator: " + msg.str());
}

// The compute dataflow extracted from the schedule IR: which ops are present and
// which CB indices each uses. CB indices are READ from the IR (never hardcoded).
struct KernelConfig {
  bool hasRmsnorm = false;
  bool hasMatmul = false;
  bool hasGelu = false;
  float epsilon = 1.0e-6f;
  int64_t rmsnormInCB = -1, rmsnormGammaCB = -1, rmsnormOutCB = -1;
  int64_t matmulIn0CB = -1, matmulIn1CB = -1, matmulDstCB = -1;
  int64_t geluInCB = -1, geluOutCB = -1;
  int64_t writerOutCB = -1; // the out CB the writer drains
  std::set<int64_t> referencedCBs;
};

void addCB(std::set<int64_t> &s, int64_t cb) { s.insert(cb); }

// Walk the kernel, collecting every referenced CB index and the compute config.
KernelConfig extractConfig(TensixflowSchedKernelOp kernel) {
  KernelConfig cfg;

  kernel.walk([&](TensixflowSchedCbReserveBackOp op) { addCB(cfg.referencedCBs, (int64_t)op.getCb()); });
  kernel.walk([&](TensixflowSchedCbPushBackOp op) { addCB(cfg.referencedCBs, (int64_t)op.getCb()); });
  kernel.walk([&](TensixflowSchedCbWaitFrontOp op) { addCB(cfg.referencedCBs, (int64_t)op.getCb()); });
  kernel.walk([&](TensixflowSchedCbPopFrontOp op) { addCB(cfg.referencedCBs, (int64_t)op.getCb()); });
  kernel.walk([&](TensixflowSchedNocReadOp op) { addCB(cfg.referencedCBs, (int64_t)op.getCb()); });
  kernel.walk([&](TensixflowSchedNocWriteOp op) {
    addCB(cfg.referencedCBs, (int64_t)op.getCb());
    if (cfg.writerOutCB < 0)
      cfg.writerOutCB = (int64_t)op.getCb();
  });
  kernel.walk([&](TensixflowSchedNocMulticastOp op) { addCB(cfg.referencedCBs, (int64_t)op.getCb()); });

  kernel.walk([&](TensixflowSchedRmsnormOp op) {
    cfg.hasRmsnorm = true;
    cfg.rmsnormInCB = (int64_t)op.getInput();
    cfg.rmsnormGammaCB = (int64_t)op.getGamma();
    cfg.rmsnormOutCB = (int64_t)op.getOut();
    cfg.epsilon = op.getEpsilon().convertToFloat();
    addCB(cfg.referencedCBs, cfg.rmsnormInCB);
    addCB(cfg.referencedCBs, cfg.rmsnormGammaCB);
    addCB(cfg.referencedCBs, cfg.rmsnormOutCB);
  });
  kernel.walk([&](TensixflowSchedMatmulTilesOp op) {
    cfg.hasMatmul = true;
    cfg.matmulIn0CB = (int64_t)op.getIn0();
    cfg.matmulIn1CB = (int64_t)op.getIn1();
    cfg.matmulDstCB = (int64_t)op.getDst();
    addCB(cfg.referencedCBs, cfg.matmulIn0CB);
    addCB(cfg.referencedCBs, cfg.matmulIn1CB);
    addCB(cfg.referencedCBs, cfg.matmulDstCB);
  });
  kernel.walk([&](TensixflowSchedGeluOp op) {
    cfg.hasGelu = true;
    cfg.geluInCB = (int64_t)op.getInput();
    cfg.geluOutCB = (int64_t)op.getOut();
    addCB(cfg.referencedCBs, cfg.geluInCB);
    addCB(cfg.referencedCBs, cfg.geluOutCB);
  });

  // Fallback: a writer that only waits/pops (no noc_write) still names its drain CB.
  if (cfg.writerOutCB < 0)
    kernel.walk([&](TensixflowSchedCbWaitFrontOp op) {
      if (cfg.writerOutCB < 0)
        cfg.writerOutCB = (int64_t)op.getCb();
    });

  return cfg;
}

// Round every element to bf16 -- models a NoC read staging tiles into a CB with
// round-on-push (the CB stores bf16).
Tensor roundToBf16(const Tensor &t) {
  Tensor r = t;
  for (float &v : r.data)
    v = to_bf16(v);
  return r;
}

// Tiled matmul C[M,N] = A[M,K] @ B[K,N] in TILED accumulation order: for each 32x32
// output tile, accumulate over the K-tiles in fp32, then round the output tile to
// bf16 (push to the out CB). A and B are bf16-valued (CB storage); the multiply/add
// is fp32. Out-of-tile elements are zero (padding for non-tile-aligned shapes).
Tensor tiledMatmul(const Tensor &a, const Tensor &b) {
  int64_t m = a.dim(0), k = a.dim(1), n = b.dim(1);
  Tensor c{{m, n}, std::vector<float>((size_t)(m * n), 0.0f)};
  int64_t mTiles = ceilDiv(m, kTileDim), nTiles = ceilDiv(n, kTileDim);
  int64_t kTiles = ceilDiv(k, kTileDim);
  for (int64_t mt = 0; mt < mTiles; ++mt)
    for (int64_t nt = 0; nt < nTiles; ++nt) {
      float acc[kTileDim][kTileDim] = {}; // fp32 accumulator, zero-init
      for (int64_t kt = 0; kt < kTiles; ++kt)
        for (int i = 0; i < kTileDim; ++i) {
          int64_t row = mt * kTileDim + i;
          if (row >= m)
            continue;
          for (int j = 0; j < kTileDim; ++j) {
            int64_t col = nt * kTileDim + j;
            if (col >= n)
              continue;
            float s = 0.0f;
            for (int kk = 0; kk < kTileDim; ++kk) {
              int64_t kkIdx = kt * kTileDim + kk;
              if (kkIdx >= k)
                continue;
              s += a.data[(size_t)(row * k + kkIdx)] *
                   b.data[(size_t)(kkIdx * n + col)];
            }
            acc[i][j] += s; // accumulate across K-tiles in fp32
          }
        }
      for (int i = 0; i < kTileDim; ++i)
        for (int j = 0; j < kTileDim; ++j) {
          int64_t row = mt * kTileDim + i, col = nt * kTileDim + j;
          if (row < m && col < n)
            c.data[(size_t)(row * n + col)] = to_bf16(acc[i][j]); // bf16 per tile
        }
    }
  return c;
}

// Tile RMSNorm y[M,K] = x[M,K] * rsqrt(mean_k(x^2) + eps) * gamma[K]. x and gamma
// are bf16-valued (CB storage); the square-sum reduce + rsqrt + scale is fp32; the
// normalized output is rounded to bf16 (the fused CB intermediate feeding the
// matmul -- rounded between fused stages per the contract).
Tensor rmsnormTile(const Tensor &x, const Tensor &gamma, float eps) {
  int64_t m = x.dim(0), k = x.dim(1);
  Tensor y{{m, k}, std::vector<float>((size_t)(m * k))};
  for (int64_t row = 0; row < m; ++row) {
    float ss = 0.0f;
    for (int64_t c = 0; c < k; ++c) {
      float v = x.data[(size_t)(row * k + c)];
      ss += v * v; // fp32 reduce-square-sum
    }
    float rms = 1.0f / std::sqrt(ss / (float)k + eps);
    for (int64_t c = 0; c < k; ++c)
      y.data[(size_t)(row * k + c)] =
          to_bf16(x.data[(size_t)(row * k + c)] * rms * gamma.data[(size_t)c]);
  }
  return y;
}

// Tile GELU (exact erf form), fp32 compute, bf16 round of the result.
Tensor geluTile(const Tensor &x) {
  Tensor y = x;
  for (float &v : y.data)
    v = to_bf16(0.5f * v * (1.0f + std::erf(v / std::sqrt(2.0f))));
  return y;
}

} // namespace

std::vector<int64_t> collectReferencedCBs(TensixflowSchedKernelOp kernel) {
  KernelConfig cfg = extractConfig(kernel);
  return {cfg.referencedCBs.begin(), cfg.referencedCBs.end()};
}

llvm::Expected<Tensor> runTileSimulator(TensixflowSchedKernelOp kernel,
                                        const SimInputs &inputs) {
  KernelConfig cfg = extractConfig(kernel);

  // Validate the allocation: every referenced CB must be in range AND allocated.
  // This is the required clear-error path (not silent garbage) for an unknown CB.
  std::set<int64_t> allocated(inputs.allocatedCBs.begin(), inputs.allocatedCBs.end());
  for (int64_t cb : cfg.referencedCBs) {
    if (cb < 0 || cb > 31)
      return simError(Twine("references circular buffer c_") + Twine(cb) +
                      " outside the valid tt-metal range c_0..c_31");
    if (!allocated.count(cb))
      return simError(Twine("references unallocated circular buffer c_") + Twine(cb));
  }

  if (!cfg.hasMatmul && !cfg.hasRmsnorm && !cfg.hasGelu)
    return simError("schedule IR has no recognizable compute op "
                    "(matmul_tiles / rmsnorm / gelu)");

  // Model the circular buffers as bf16 buffers keyed by CB index. The reader stages
  // the input tiles (NoC read = memory copy with bf16 round-on-push).
  int64_t actCB = cfg.hasRmsnorm ? cfg.rmsnormInCB : cfg.matmulIn0CB;
  std::map<int64_t, Tensor> cbs;
  cbs[actCB] = roundToBf16(inputs.activation);
  if (cfg.hasMatmul)
    cbs[cfg.matmulIn1CB] = roundToBf16(inputs.weight);
  if (cfg.hasRmsnorm)
    cbs[cfg.rmsnormGammaCB] = roundToBf16(inputs.gamma);

  // Model the compute (fp32 accumulate, bf16 round per tile / fused intermediate).
  if (cfg.hasRmsnorm)
    cbs[cfg.rmsnormOutCB] =
        rmsnormTile(cbs[cfg.rmsnormInCB], cbs[cfg.rmsnormGammaCB], cfg.epsilon);
  if (cfg.hasMatmul)
    cbs[cfg.matmulDstCB] = tiledMatmul(cbs[cfg.matmulIn0CB], cbs[cfg.matmulIn1CB]);
  if (cfg.hasGelu)
    cbs[cfg.geluOutCB] = geluTile(cbs[cfg.geluInCB]);

  // Model the writer: drain the out CB to the output tensor (already bf16; a memory
  // copy, no further rounding).
  if (cfg.writerOutCB < 0)
    return simError("schedule IR writer drains no circular buffer");
  auto it = cbs.find(cfg.writerOutCB);
  if (it == cbs.end())
    return simError(Twine("writer drains circular buffer c_") + Twine(cfg.writerOutCB) +
                    " which the compute never produced");
  return it->second;
}

} // namespace mlir::tensixflow::sim
