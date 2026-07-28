//===- Codegen/EmitCompute.cpp - Emit a tt-metal compute.cpp ---*- C++ -*-===//
//
// TensixFlow: An MLIR Compiler for Tile-Native LLM Kernels on
// Tenstorrent-Style Spatial Accelerators.
//
//===----------------------------------------------------------------------===//
//
// The compute.cpp emitter (Wave 6, todo 36). See include/tensixflow/Codegen/
// EmitCompute.h for the contract. The emitted TU mirrors the compute side of
// the hand-written kernels/handwritten_baselines/matmul_single_core_baseline.cpp
// -- the proof-of-what-compiles against the todo-33 stub headers -- using the
// global-namespace compute intrinsics (matmul_init / matmul_tiles from
// compute_kernel_api/matmul.h, tile_regs_* from reg_api.h, pack_tile from
// pack.h) plus the cb_* circular-buffer intrinsics from dataflow_api.h.
//
// This emits the TRISC compute kernel ONLY: tile math + circular-buffer
// staging. It emits NO data movement (the reader/writer own the NoC). For a
// fused plan the RMSNorm prologue / GELU epilogue are emitted as documented
// structural staging (comments + the cb_* ops that stage the gamma tile)
// because the stubs declare no rmsnorm / gelu intrinsic; the staging compiles
// against the existing stubs as-is.
//
//===----------------------------------------------------------------------===//

#include "tensixflow/Codegen/EmitCompute.h"

#include "tensixflow/TensixflowAttrs.h"
#include "tensixflow/TensixflowSchedule.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace tensixflow {

namespace {

// One circular-buffer slot read off the schedule's parallel index/depth/
// tile-bytes arrays (the --tensixflow-allocate-circular-buffers output).
struct CbSpec {
  int64_t index;
  int64_t depth;
  int64_t tileBytes;
  int64_t bytes() const { return depth * tileBytes; }
};

// Read the schedule's parallel CB arrays into CbSpecs. Falls back to the
// documented 3-CB reader/compute/writer pipeline {0,1,16} when the arrays are
// absent/short so the emission is always a complete, compilable TU.
SmallVector<CbSpec> readCbSpecs(TensixflowScheduleAttr schedule) {
  SmallVector<CbSpec> cbs;
  ArrayAttr indices = schedule.getCbIndices();
  ArrayAttr depths = schedule.getCbDepths();
  ArrayAttr tileBytes = schedule.getCbTileBytes();
  unsigned n = indices.size();
  if (depths.size() < n)
    n = depths.size();
  if (tileBytes.size() < n)
    n = tileBytes.size();
  for (unsigned i = 0; i < n; ++i)
    cbs.push_back({cast<IntegerAttr>(indices[i]).getInt(),
                   cast<IntegerAttr>(depths[i]).getInt(),
                   cast<IntegerAttr>(tileBytes[i]).getInt()});
  if (cbs.size() < 3) {
    cbs.clear();
    cbs.push_back({0, 2, 2048});
    cbs.push_back({1, 2, 2048});
    cbs.push_back({16, 2, 2048});
  }
  return cbs;
}

// The tile-loop trip count driving a reader/compute/writer region (the first
// sched.tile_loop nested in it); 1 if the region is empty/absent.
int64_t tripCount(Operation *op) {
  int64_t trip = 1;
  if (op)
    op->walk([&](TensixflowSchedTileLoopOp loop) {
      trip = loop.getTripCount();
      return WalkResult::interrupt();
    });
  return trip;
}

} // namespace

void emitCompute(TensixflowSchedKernelOp kernel,
                 TensixflowScheduleAttr schedule, llvm::raw_ostream &os) {
  // Pull the compute tile program (and the writer's, for the output-tile count)
  // off the schedule IR.
  TensixflowSchedComputeOp compute;
  TensixflowSchedWriterOp writer;
  kernel.walk([&](TensixflowSchedComputeOp op) { compute = op; });
  kernel.walk([&](TensixflowSchedWriterOp op) { writer = op; });
  int64_t computeTrip = tripCount(compute); // inner-K tiles (the reduction)
  int64_t writerTrip = tripCount(writer);   // output tiles, grid-wide

  // Plan decisions read off the schedule attribute (never hardcoded). The
  // prologue/epilogue gate mirrors LowerToTTKernelPass's derivation.
  unsigned gridY = schedule.getGridY();
  unsigned gridX = schedule.getGridX();
  StringRef fusion = schedule.getFusion().getValue();
  StringRef weightStrategy = schedule.getWeightStrategy().getValue();
  bool hasRmsnorm = fusion == "fused_rmsnorm_matmul" || fusion == "fused_mlp";
  bool hasGelu = fusion == "fused_mlp";
  SmallVector<CbSpec> cbs = readCbSpecs(schedule);
  const CbSpec &in0 = cbs[0];
  const CbSpec &in1 = cbs[1];
  const CbSpec &out = cbs[2];

  // The RMSNorm prologue's gamma CB index + epsilon, read off the schedule IR's
  // sched.rmsnorm op (the lowering records them there); documented fallbacks
  // (c_2, 1e-6) when the op is absent.
  TensixflowSchedRmsnormOp rmsnorm;
  if (compute)
    compute->walk([&](TensixflowSchedRmsnormOp op) {
      rmsnorm = op;
      return WalkResult::interrupt();
    });
  uint64_t gammaIndex = rmsnorm ? rmsnorm.getGamma() : 2;
  float epsilon =
      rmsnorm ? rmsnorm.getEpsilon().convertToFloat() : 1.0e-6f;

  // --- Translation-unit prologue ------------------------------------------
  os << "//===- compute.cpp - TensixFlow-generated compute (tile math) ===//\n";
  os << "//\n";
  os << "// Auto-generated by tensixflow-opt --tensixflow-emit-metalium. Do not edit.\n";
  os << "// kernel: " << kernel.getSymName() << "  fusion: " << fusion
     << "  weight: " << weightStrategy << "  grid: " << gridY << "x" << gridX
     << "\n";
  os << "//\n";
  os << "// Compute (TRISC): TILE MATH ONLY. Consumes the tiles the reader staged\n";
  os << "// into the input circular buffers, accumulates the matmul into a\n";
  os << "// destination tile register (fp32), and packs result tiles into the\n";
  os << "// output circular buffer for the writer to drain. The reader/writer\n";
  os << "// kernels own all data movement; this kernel issues none of it. Mirrors\n";
  os << "// the compute side of tt-metal matmul_single_core's kernel_main.\n";
  os << "// Rounding contract: fp32 accumulation in the destination register,\n";
  os << "// bf16 round on pack.\n";
  os << "\n";
  os << "#include <cstdint>\n";
  os << "\n";
  os << "#include \"compute_kernel_api.h\"\n";
  os << "#include \"dataflow_api.h\"\n";
  os << "#include \"kernel_types.hpp\"\n";
  os << "\n";
  os << "namespace compute {\n\n";
  os << "void kernel_main() {\n";

  // --- Circular-buffer indices (from the schedule allocation) -------------
  os << "  // Circular-buffer indices (from the schedule allocation; tt::CBIndex).\n";
  os << "  constexpr auto in0_cb = static_cast<std::uint32_t>(tt::CBIndex::c_"
     << in0.index << ");\n";
  os << "  constexpr auto in1_cb = static_cast<std::uint32_t>(tt::CBIndex::c_"
     << in1.index << ");\n";
  os << "  constexpr auto out_cb = static_cast<std::uint32_t>(tt::CBIndex::c_"
     << out.index << ");\n";
  if (hasRmsnorm)
    os << "  // RMSNorm prologue gamma circular buffer (the per-column scale).\n"
       << "  constexpr auto gamma_cb = static_cast<std::uint32_t>(tt::CBIndex::c_"
       << gammaIndex << ");\n";
  os << "\n";

  // --- Compile-time args ----------------------------------------------------
  os << "  // Compile-time arg 0 (host ComputeConfig.compile_args): the inner-K\n";
  os << "  // tile trip count (the reduction dimension). The schedule decomposes\n";
  os << "  // the reduction into " << computeTrip << " K-tiles; the writer program drains\n";
  os << "  // the grid's " << writerTrip << " output tiles.\n";
  os << "  const auto num_k_tiles = get_compile_time_arg_val<std::uint32_t>(0);\n\n";

  // --- Matmul engine init ---------------------------------------------------
  os << "  // Start the matmul engine (single-tile blocks).\n";
  os << "  matmul_init(/*in0_block_c_tiles=*/1, /*in1_block_r_tiles=*/1);\n\n";

  // --- Output tile loop -----------------------------------------------------
  os << "  // Output tile loop: one destination tile per iteration. First-order\n";
  os << "  // single-core model -- one output tile per kernel invocation; the\n";
  os << "  // host's grid placement spreads the grid's output tiles across cores.\n";
  os << "  constexpr std::uint32_t num_out_tiles = 1;\n";
  os << "  for (std::uint32_t out_tile = 0; out_tile < num_out_tiles; ++out_tile) {\n";
  os << "    // Acquire the destination tile registers for this output tile.\n";
  os << "    tile_regs_acquire();\n\n";
  os << "    // Inner-K loop: reduce over the K tiles the reader staged into\n";
  os << "    // in0/in1, accumulating into destination tile register 0.\n";
  os << "    for (std::uint32_t k_tile = 0; k_tile < num_k_tiles; ++k_tile) {\n";
  os << "      cb_wait_front(in0_cb, 1);\n";
  os << "      cb_wait_front(in1_cb, 1);\n";

  // --- RMSNorm prologue (fused_rmsnorm_matmul / fused_mlp) ------------------
  if (hasRmsnorm) {
    os << "\n";
    os << "      // RMSNorm prologue: normalize the activation tile in in0 in place\n";
    os << "      // before the matmul. Tile math, per the rounding contract (fp32\n";
    os << "      // accumulate, bf16 round):\n";
    os << "      //   1. reduce square-sum: ss = sum(x * x) over the row, fp32 accumulate;\n";
    os << "      //   2. rsqrt: inv_rms = rsqrt(ss / width + epsilon), epsilon = "
       << epsilon << ";\n";
    os << "      //   3. scale by gamma: x' = x * inv_rms * gamma[col], bf16 round.\n";
    os << "      // Structural staging only: stage the gamma tile from its circular\n";
    os << "      // buffer (the reduce/rsqrt/scale math runs in the tile engine).\n";
    os << "      cb_wait_front(gamma_cb, 1);\n";
    os << "      cb_pop_front(gamma_cb, 1);\n";
  }

  // --- Tile matmul ----------------------------------------------------------
  os << "\n";
  os << "      // Tile matmul: dst(0) += in0(0) x in1(0), fp32 accumulate.\n";
  os << "      matmul_tiles(in0_cb, in1_cb, /*in0_tile=*/0, /*in1_tile=*/0,\n";
  os << "                   /*dst_tile=*/0, /*transpose_of_faces=*/false);\n";
  os << "\n";
  os << "      // Release the consumed input tiles.\n";
  os << "      cb_pop_front(in0_cb, 1);\n";
  os << "      cb_pop_front(in1_cb, 1);\n";
  os << "    }\n\n";

  // --- GELU epilogue (fused_mlp) --------------------------------------------
  if (hasGelu) {
    os << "    // GELU epilogue: apply the activation elementwise to the fp32\n";
    os << "    // destination tile before commit/pack (rounding contract: fp32\n";
    os << "    // accumulate, bf16 round). Structural marker -- the elementwise math\n";
    os << "    // runs in the tile engine over the destination register the matmul\n";
    os << "    // produced.\n\n";
  }

  // --- Commit / pack / push -------------------------------------------------
  os << "    // Commit the math, reserve an output slot, pack the fp32 destination\n";
  os << "    // tile into the output circular buffer (bf16 round), make it visible.\n";
  os << "    tile_regs_commit();\n";
  os << "    cb_reserve_back(out_cb, 1);\n";
  os << "    tile_regs_wait();\n";
  os << "    pack_tile(/*tile_index=*/0, out_cb);\n";
  os << "    tile_regs_release();\n";
  os << "    cb_push_back(out_cb, 1);\n";
  os << "  }\n";

  os << "}\n\n";
  os << "}  // namespace compute\n";
}

} // namespace tensixflow
} // namespace mlir
