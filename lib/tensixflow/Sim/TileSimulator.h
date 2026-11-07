//===- Sim/TileSimulator.h - Functional tile simulator -----------*- C++ -*-===//
//
// TensixFlow: An MLIR Compiler for Tile-Native LLM Kernels on
// Tenstorrent-Style Spatial Accelerators.
//
//===----------------------------------------------------------------------===//
//
// The functional tile simulator. It EXECUTES the tile-level
// schedule IR (the `tensixflow.sched.kernel` built by --tensixflow-lower-to-ttkernel,
// lowering) over modeled circular buffers -- it does NOT parse or execute the
// emitted C++ (that is compile-checked elsewhere, todos 33/37).
//
// THE ROUNDING CONTRACT it enforces (shared with Rounding.h):
//   * each circular buffer is a bf16 buffer -- tiles pushed into a CB are rounded
//     to bf16 (round-on-push via sim::to_bf16); CB storage is bf16, NOT fp32;
//   * NoC read/write are modeled as memory copies (the reader rounds input tiles
//     into the in0/in1/gamma CBs; the writer drains the out CB to the output);
//   * compute accumulates in fp32 WITHIN a tile across K-tiles (TILED accumulation
//     order), and rounds each output TILE back to bf16 before pushing it to the
//     out CB; fused intermediates (e.g. the rmsnorm output feeding the matmul) are
//     stored as bf16 (rounded between fused stages).
//
// The deliberate bf16 rounding (against the golden's fp32-exact path) is what
// makes the PCC bar meaningful: PCC(simulator_bf16_tiled, golden_fp32)
// measures the real bf16-spatial-vs-exact cost of fusion + tiled execution.
//
// The simulator interprets the schedule IR STRUCTURE: it walks the kernel's
// reader/compute/writer regions to learn WHICH compute ops are present (rmsnorm
// prologue / matmul_tiles / gelu epilogue), WHICH CB indices they use, the rmsnorm
// epsilon, and the tile-loop trip counts. It then performs the denoted tile
// computation over the input tensors. The per-tile grid is derived from the input
// tensor shapes (the ground truth the simulator is given -- the schedule IR carries
// no per-tile coordinates, only the first-order trip counts, which the simulator
// reconciles with the shapes); the IR supplies the dataflow + fusion config + CBs.
//
//===----------------------------------------------------------------------===//

#ifndef TENSIXFLOW_SIM_TILESIMULATOR_H
#define TENSIXFLOW_SIM_TILESIMULATOR_H

#include "tensixflow/TensixflowSchedule.h"

#include "llvm/Support/Error.h"

#include <cstdint>
#include <vector>

namespace mlir::tensixflow::sim {

// The modeled Tensix tile is 32x32 (matches the lowering's kTile).
inline constexpr int64_t kTileDim = 32;

//===----------------------------------------------------------------------===//
// Tensor: a row-major shaped buffer of fp32 values. bf16 CB-storage semantics are
// ENFORCED BY THE SIMULATOR (it rounds on push), not by this type -- the values are
// held as fp32 whose low 16 mantissa bits are zero whenever they model CB contents.
//===----------------------------------------------------------------------===//
struct Tensor {
  std::vector<int64_t> shape;
  std::vector<float> data; // row-major, size == product(shape)

  int64_t rank() const { return static_cast<int64_t>(shape.size()); }
  int64_t dim(int64_t i) const { return shape[i]; }
  int64_t numElements() const {
    int64_t n = 1;
    for (int64_t d : shape)
      n *= d;
    return n;
  }
  bool empty() const { return data.empty(); }
};

//===----------------------------------------------------------------------===//
// SimInputs: the simulator's inputs (a genuine domain aggregate, not a throwaway
// wrapper). The activation [M, K] and (for a matmul) the weight [K, N] are the
// operand tensors; gamma [K] is the per-column RMSNorm scale (empty when the kernel
// has no rmsnorm). allocatedCBs is the set of circular-buffer indices the caller
// (the CB allocator) has allocated -- the simulator rejects any schedule
// IR that references a CB outside this set (the required clear-error path).
//===----------------------------------------------------------------------===//
struct SimInputs {
  Tensor activation;
  Tensor weight;
  Tensor gamma;
  std::vector<int64_t> allocatedCBs;
};

//===----------------------------------------------------------------------===//
// Run the functional tile simulator: interpret the schedule IR `kernel` over modeled
// bf16 circular buffers and produce the output tensor. Returns an llvm::Error (not
// silent garbage) if the IR references an unallocated or out-of-range CB, has no
// recognizable compute, or drains a CB the compute never produced.
//===----------------------------------------------------------------------===//
llvm::Expected<Tensor> runTileSimulator(TensixflowSchedKernelOp kernel,
                                        const SimInputs &inputs);

//===----------------------------------------------------------------------===//
// Collect every CB index the schedule IR references (across the CB / NoC / compute
// ops). A convenience for callers that want to allocate exactly the CBs a kernel
// uses (the happy-path allocation); the simulator's validation is the authoritative
// check that referenced CBs are a subset of the caller's allocation.
//===----------------------------------------------------------------------===//
std::vector<int64_t> collectReferencedCBs(TensixflowSchedKernelOp kernel);

} // namespace mlir::tensixflow::sim

#endif // TENSIXFLOW_SIM_TILESIMULATOR_H
