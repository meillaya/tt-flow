//===- TensixflowPasses.td - Tensixflow passes ---------------*- tablegen -*-===//
//
// TensixFlow: An MLIR Compiler for Tile-Native LLM Kernels on
// Tenstorrent-Style Spatial Accelerators.
//
//===----------------------------------------------------------------------===//
//
// Pass declarations for the tensixflow transform pipeline. The seven passes
// form the lowering spine (tile-layout inference -> canonicalize -> fusion ->
// NoC sharding -> circular-buffer allocation -> kernel lowering -> Metalium
// emission). Bodies land in later todos; registration + composition are proven
// here with no-op stubs.
//
//===----------------------------------------------------------------------===//

#ifndef TENSIXFLOW_PASSES_TD
#define TENSIXFLOW_PASSES_TD

include "mlir/Pass/PassBase.td"

// Shared shape for every tensixflow pass: a module-level anchor (fine for the
// stub stage). LLVM 20's Pass has no cppNamespace field — the generated
// decls/defs/registration land in whatever namespace the .h.inc is included
// in (TensixflowPasses.h/.cpp include it inside mlir::tensixflow), exactly as
// mlir/examples/standalone does.
class TensixflowPass<string name, string s> : Pass<name, "::mlir::ModuleOp"> {
  let summary = s;
}

def InferTileLayoutPass : TensixflowPass<"tensixflow-infer-tile-layout",
    "Infer 32x32 tile layouts for tensor results"> {
  let description = [{
    Attaches a `#tensixflow.tile_layout` encoding to each tensor result based
    on its shape (dims rounded to 32x32 tiles, default 8x8 Wormhole grid).
    Stub: no-op until todo 15.
  }];
}

def CanonicalizePass : TensixflowPass<"tensixflow-canonicalize",
    "Canonicalize tensixflow ops"> {
  let description = [{
    Applies tensixflow canonicalization patterns (identity-add removal,
    trivial constant folding, structural dedup). Stub: no-op until todo 16.
  }];
}

def FuseRmsnormMatmulPass : TensixflowPass<"tensixflow-fuse-rmsnorm-matmul",
    "Fuse rmsnorm + matmul into a single fused op"> {
  let description = [{
    Matches an adjacent rmsnorm -> matmul pair and rewrites it into one fused
    op to cut intermediate DRAM traffic. Stub: no-op until a later todo.
  }];
}

def PlanNocShardingPass : TensixflowPass<"tensixflow-plan-noc-sharding",
    "Plan NoC-aware sharding under the L1 budget"> {
  let description = [{
    Assigns tensor shards to cores of the 8x8 grid under the ~1 MB L1 budget
    and plans NoC movement. Stub: no-op until a later todo.
  }];
}

def AllocateCircularBuffersPass :
    TensixflowPass<"tensixflow-allocate-circular-buffers",
    "Allocate circular buffers for tile streams"> {
  let description = [{
    Allocates L1 circular buffers for the tile streams between
    reader/compute/writer kernels. Stub: no-op until a later todo.
  }];
}

def LowerToTTKernelPass : TensixflowPass<"tensixflow-lower-to-ttkernel",
    "Lower to the TT-Metalium kernel schedule"> {
  let description = [{
    Lowers fused/sharded ops into the tile-level schedule IR
    (tile-loop nests + CB + NoC + compute ops). Stub: no-op until a later
    todo.
  }];
}

def EmitMetaliumPass : TensixflowPass<"tensixflow-emit-metalium",
    "Emit TT-Metalium-style reader/compute/writer kernels"> {
  let description = [{
    Emits TT-Metalium-style host/reader/compute/writer kernel sources from
    the schedule IR. Stub: no-op until a later todo.
  }];
}

#endif // TENSIXFLOW_PASSES_TD
