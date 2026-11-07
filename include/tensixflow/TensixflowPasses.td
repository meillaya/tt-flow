//===- TensixflowPasses.td - Tensixflow passes ---------------*- tablegen -*-===//
//
// TensixFlow: An MLIR Compiler for Tile-Native LLM Kernels on
// Tenstorrent Hardware.
//
//===----------------------------------------------------------------------===//
//
// Pass declarations for the tensixflow transform pipeline. The seven passes form
// the lowering spine: tile-layout inference, canonicalize, fusion, NoC
// sharding, circular-buffer allocation, kernel lowering, and Metalium emission.
// The device each pass plans against comes from the generated table in
// Devices.h and the `tensixflow.device` attribute the driver stamps.
//
//===----------------------------------------------------------------------===//

#ifndef TENSIXFLOW_PASSES_TD
#define TENSIXFLOW_PASSES_TD

include "mlir/Pass/PassBase.td"

// Shared shape for every tensixflow pass: a module-level anchor. LLVM 20's Pass
// has no cppNamespace field, so the generated declarations and definitions land
// in whatever namespace includes the .h.inc, which is mlir::tensixflow in both
// TensixflowPasses.h and TensixflowPasses.cpp.
class TensixflowPass<string name, string s> : Pass<name, "::mlir::ModuleOp"> {
  let summary = s;
}

def InferTileLayoutPass : TensixflowPass<"tensixflow-infer-tile-layout",
    "Infer 32x32 tile layouts for tensor results"> {
  let description = [{
    Attaches a `#tensixflow.tile_layout` encoding to each tensor result. The
    encoding records the 32x32 tile shape, the compute grid of the selected
    device, and whether the tensor lives in DRAM or L1.
  }];
}

def CanonicalizePass : TensixflowPass<"tensixflow-canonicalize",
    "Canonicalize tensixflow ops"> {
  let description = [{
    Applies tensixflow canonicalization patterns: identity-add removal, trivial
    constant folding, and structural dedup.
  }];
}

def FuseRmsnormMatmulPass : TensixflowPass<"tensixflow-fuse-rmsnorm-matmul",
    "Fuse rmsnorm + matmul into a single fused op"> {
  let description = [{
    Matches an rmsnorm -> matmul pair, or a full rmsnorm -> matmul -> bias ->
    gelu -> matmul -> residual chain, and rewrites the match into one fused op
    to cut intermediate DRAM traffic.
  }];
}

def PlanNocShardingPass : TensixflowPass<"tensixflow-plan-noc-sharding",
    "Plan NoC-aware sharding under the L1 budget"> {
  let description = [{
    Assigns tensor shards to cores of the selected device's compute grid under
    that device's L1 budget, picks a weight strategy, plans NoC movement, and
    records the plan on the module.
  }];
}

def AllocateCircularBuffersPass :
    TensixflowPass<"tensixflow-allocate-circular-buffers",
    "Allocate circular buffers for tile streams"> {
  let description = [{
    Allocates L1 circular buffers for the tile streams between reader, compute,
    and writer kernels. Buffer depth starts at two tiles and is clamped to the
    per-core budget of the selected device.
  }];
}

def LowerToTTKernelPass : TensixflowPass<"tensixflow-lower-to-ttkernel",
    "Lower to the TT-Metalium kernel schedule"> {
  let description = [{
    Lowers fused and sharded ops into the tile-level schedule IR: tile-loop
    nests over circular-buffer, NoC, and compute ops.
  }];
}

def EmitMetaliumPass : TensixflowPass<"tensixflow-emit-metalium",
    "Emit TT-Metalium host and kernel sources"> {
  let description = [{
    Emits a TT-Metalium host program and reader, compute, and writer kernels
    from the schedule IR. By default the artifacts print to stdout separated by
    `// ===== file: <name>.cpp =====` banners. With `output-dir` set, the pass
    writes a standalone project tree to that directory: host.cpp, the three
    kernel sources, a CMakeLists.txt that links against the pinned TT-Metalium
    release, and a README.md.

    The emitted program is the single-core matmul path. A schedule built from a
    fused RMSNorm graph is rejected with a message that names the supported path
    and the plan and report alternative.

    Pass options take the form `--tensixflow-emit-metalium=output-dir=<dir>` or
    the pipeline syntax `builtin.module(tensixflow-emit-metalium{output-dir=
    <dir>})`. A bare `-o <dir>` is the global output-file flag, not this option.
  }];
  let options = [
    Option<"outputDir", "output-dir", "std::string", /*default=*/"\"\"",
           "Write a complete standalone project tree (host, kernels, CMakeLists.txt, README.md) to this directory instead of stdout">
  ];
}

def EmitTTIRPass : TensixflowPass<"tensixflow-emit-ttir",
    "Emit TTIR-shaped MLIR text from tensixflow ops"> {
  let description = [{
    Translates tensixflow ops to TTIR-shaped MLIR text and prints the result to
    stdout as a terminal emission. The syntax contract is tt-mlir
    0.9.0.dev20260221, pinned in deps/tt-mlir.lock; the released wheel bindings
    are the parser gate in python/tensixflow/tests/test_ttir_parse.py.

    Supported subset: add, matmul, rmsnorm, gelu, constant, and the mapped
    prefixes of fused_rmsnorm_matmul and fused_mlp. The full op table and the
    layout encoding rules are in docs/ttir.md. An op with no TTIR mapping emits
    a `// UNSUPPORTED: <op>` marker instead of crashing, and dependent ops of an
    unsupported result are marked transitively.

    The pass does not run any upstream tt-mlir pipeline. Lowering the emitted
    text through tt-mlir is outside this repository's checks.
  }];
}

#endif // TENSIXFLOW_PASSES_TD
