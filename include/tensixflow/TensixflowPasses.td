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
    the schedule IR. By default the four artifacts print to stdout separated
    by `// ===== file: <name>.cpp =====` delimiter banners. With `output-dir`
    set (todo 37), the pass instead writes a complete standalone project
    tree to that directory: host.cpp, reader.cpp, compute.cpp, writer.cpp,
    a compile-only CMakeLists.txt (syntax-checked against the signature-only
    tt-metal stubs; no link, no device) and a README.md.

    MLIR 20.1.8 pass-option CLI form (verified): the option is attached to
    the pass flag as `--tensixflow-emit-metalium=output-dir=<dir>` (a bare
    `-o <dir>` is NOT possible — mlir-opt owns the global `-o` output-file
    flag, and pass options parse only as `--<pass>=<option>=<value>` or via
    the pipeline syntax `builtin.module(tensixflow-emit-metalium{output-dir=
    <dir>})`).
  }];
  let options = [
    Option<"outputDir", "output-dir", "std::string", /*default=*/"\"\"",
           "Write a complete standalone project tree (host/reader/compute/writer.cpp + CMakeLists.txt + README.md) to this directory instead of stdout">
  ];
}

def EmitTTIRPass : TensixflowPass<"tensixflow-emit-ttir",
    "Emit TTIR-shaped MLIR text from tensixflow ops"> {
  let description = [{
    Translates tensixflow ops to TTIR-shaped MLIR text (todo 51, Wave 9),
    printed to stdout as a TERMINAL emission (flush + exit 0 once printed, the
    EmitMetaliumPass convention, so mlir-opt does not append the module IR and
    the stdout artifact is standalone TTIR text).

    TTIR ONLY per draft D21: TTNN/D2M/TTKernel/TTMetal stay prose mappings in
    docs/tenstorrent_mapping.md and are never emitted as code. The syntax
    contract is docs/ttmlir_study.md (tt-mlir @9f06802f test snippets): the
    generic op form with the shape in the function type, attributes in
    `<{...}>`, and placement in `#ttcore.metal_layout` tensor encodings.

    Mappings: add -> "ttir.add", matmul -> "ttir.matmul", rmsnorm ->
    "ttir.rms_norm" (the fused canonical op tt-mlir's own --ttir-fusing
    produces from the pow/sum/multiply/add/rsqrt/broadcast/multiply
    decomposition), fused_rmsnorm_matmul -> rms_norm + matmul, fused_mlp ->
    the study-grounded prefix (rms_norm + matmul + bias add) plus UNSUPPORTED
    markers for the gelu stage and its poisoned tail. An op with no
    study-grounded TTIR mapping (tensixflow.gelu today) emits a documented
    `// UNSUPPORTED: <op> has no TTIR mapping` marker instead of crashing;
    dependent ops of an unsupported result are marked transitively.
  }];
}

#endif // TENSIXFLOW_PASSES_TD
