//===- TensixflowSchedule.td - Tile-level schedule IR -------*- tablegen -*-===//
//
// TensixFlow: An MLIR Compiler for Tile-Native LLM Kernels on
// Tenstorrent-Style Spatial Accelerators.
//
//===----------------------------------------------------------------------===//
//
// The tile-level schedule IR is consumed by two components: the lowering pass
// (--tensixflow-lower-to-ttkernel, which builds these ops) and the functional
// simulator (tensixflow-simulate, which executes them).
//
// A `tensixflow.sched.kernel` holds the three tile programs of a TT-Metalium
// style kernel, reader / compute / writer, mirroring the kernel_main split in
// tt-metal's matmul example. Each program is a region of structured
// `sched.tile_loop` nests that stage tiles through circular buffers, move them
// over the NoC, and compute on them. CB indices follow the tt-metal
// tt::CBIndex c_0..c_31 convention; the default pipeline uses in0=c_0, in1=c_1,
// out=c_16.
//
// This file defines structure only. Emission and execution live in the emitter
// passes and in the simulator tool.
//
//===----------------------------------------------------------------------===//

#ifndef TENSIXFLOW_SCHEDULE
#define TENSIXFLOW_SCHEDULE

include "tensixflow/TensixflowDialect.td"
include "mlir/IR/SymbolInterfaces.td"

//===----------------------------------------------------------------------===//
// Base tensixflow.sched operation definition.
//===----------------------------------------------------------------------===//
//
// The mnemonic carries the `sched.` prefix so the full op name is
// `tensixflow.sched.<name>` (MLIR splits the dialect name off the first dot;
// the dotted remainder is the op mnemonic).

class Tensixflow_SchedOp<string mnemonic, list<Trait> traits = []> :
    Op<Tensixflow_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Container ops: kernel -> { reader, compute, writer }
//===----------------------------------------------------------------------===//

def TensixflowSchedKernelOp : Tensixflow_SchedOp<"sched.kernel", [
    Symbol, IsolatedFromAbove, NoTerminator]> {
  let summary = "Tile-program kernel container (reader/compute/writer)";
  let description = [{
    The `tensixflow.sched.kernel` operation is the top-level container for one
    kernel's tile program. It is isolated from above and holds the three tile
    programs of a TT-Metalium-style kernel as nested `tensixflow.sched.reader`,
    `tensixflow.sched.compute`, and `tensixflow.sched.writer` ops.

    Example:

    ```mlir
    tensixflow.sched.kernel @matmul {
      tensixflow.sched.reader { ... }
      tensixflow.sched.compute { ... }
      tensixflow.sched.writer { ... }
    }
    ```
  }];

  let arguments = (ins SymbolNameAttr:$sym_name);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$sym_name $body attr-dict";
}

def TensixflowSchedReaderOp : Tensixflow_SchedOp<"sched.reader", [NoTerminator]> {
  let summary = "Reader (data-movement) tile program";
  let description = [{
    The `tensixflow.sched.reader` operation holds the reader tile program: it
    stages input tiles into circular buffers via NoC reads, mirroring the
    tt-metal reader kernel_main pattern (cb_reserve_back / noc_async_read /
    cb_push_back). Body is a structured nest of `sched.tile_loop` + CB/NoC ops.
  }];

  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$body attr-dict";
}

def TensixflowSchedComputeOp : Tensixflow_SchedOp<"sched.compute", [NoTerminator]> {
  let summary = "Compute tile program";
  let description = [{
    The `tensixflow.sched.compute` operation holds the compute tile program: it
    pops tiles from input CBs, runs tile math (matmul_tiles / rmsnorm / gelu),
    and pushes result tiles to the output CB, mirroring the tt-metal compute
    kernel_main pattern (cb_pop_front / matmul_tiles / cb_push_back).
  }];

  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$body attr-dict";
}

def TensixflowSchedWriterOp : Tensixflow_SchedOp<"sched.writer", [NoTerminator]> {
  let summary = "Writer (data-movement) tile program";
  let description = [{
    The `tensixflow.sched.writer` operation holds the writer tile program: it
    drains result tiles from the output CB to their destination over the NoC,
    mirroring the tt-metal writer kernel_main pattern (cb_wait_front /
    noc_async_write / cb_pop_front).
  }];

  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$body attr-dict";
}

//===----------------------------------------------------------------------===//
// Structured tile-loop nest.
//===----------------------------------------------------------------------===//

def TensixflowSchedTileLoopOp : Tensixflow_SchedOp<"sched.tile_loop", [NoTerminator]> {
  let summary = "Structured tile loop (repeat a tile program `trip_count` times)";
  let description = [{
    The `tensixflow.sched.tile_loop` operation is a structured loop that repeats
    its body `trip_count` times -- one iteration per tile (or per tile block).
    It is the unit the reader/compute/writer programs are built from and may be
    nested. `trip_count` is a static integer attribute; the verifier rejects a
    negative trip count.

    Example:

    ```mlir
    tensixflow.sched.tile_loop 64 {
      tensixflow.sched.cb_reserve_back 0, 1
      tensixflow.sched.noc_read 12345, 0, 1
      tensixflow.sched.cb_push_back 0, 1
    }
    ```
  }];

  let arguments = (ins I64Attr:$trip_count);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$trip_count $body attr-dict";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Circular-buffer ops (mirror tt-metal cb_*; keyed by a CB index c_0..c_31).
//===----------------------------------------------------------------------===//
//
// Writer side stages tiles IN: cb_reserve_back (claim slots) then cb_push_back
// (make them visible). Reader/compute side takes tiles OUT: cb_wait_front (wait
// for tiles) then cb_pop_front (release them). `cb` is the CB index; `num_tiles`
// is the tile count.

def TensixflowSchedCbReserveBackOp : Tensixflow_SchedOp<"sched.cb_reserve_back"> {
  let summary = "Reserve `num_tiles` slots in CB `cb` (writer side)";
  let description = [{
    Mirrors tt-metal `cb_reserve_back(cb, num_tiles)`: blocks until `num_tiles`
    slots are free in circular buffer `cb`, reserving them for a subsequent push.
  }];
  let arguments = (ins I64Attr:$cb, I64Attr:$num_tiles);
  let assemblyFormat = "$cb `,` $num_tiles attr-dict";
}

def TensixflowSchedCbPushBackOp : Tensixflow_SchedOp<"sched.cb_push_back"> {
  let summary = "Push `num_tiles` tiles into CB `cb` (writer side)";
  let description = [{
    Mirrors tt-metal `cb_push_back(cb, num_tiles)`: makes `num_tiles` reserved
    tiles in circular buffer `cb` visible to the consumer.
  }];
  let arguments = (ins I64Attr:$cb, I64Attr:$num_tiles);
  let assemblyFormat = "$cb `,` $num_tiles attr-dict";
}

def TensixflowSchedCbWaitFrontOp : Tensixflow_SchedOp<"sched.cb_wait_front"> {
  let summary = "Wait for `num_tiles` tiles at the front of CB `cb` (reader side)";
  let description = [{
    Mirrors tt-metal `cb_wait_front(cb, num_tiles)`: blocks until `num_tiles`
    tiles are available at the front of circular buffer `cb`.
  }];
  let arguments = (ins I64Attr:$cb, I64Attr:$num_tiles);
  let assemblyFormat = "$cb `,` $num_tiles attr-dict";
}

def TensixflowSchedCbPopFrontOp : Tensixflow_SchedOp<"sched.cb_pop_front"> {
  let summary = "Pop `num_tiles` tiles from the front of CB `cb` (reader side)";
  let description = [{
    Mirrors tt-metal `cb_pop_front(cb, num_tiles)`: releases `num_tiles` tiles
    from the front of circular buffer `cb` after they have been consumed.
  }];
  let arguments = (ins I64Attr:$cb, I64Attr:$num_tiles);
  let assemblyFormat = "$cb `,` $num_tiles attr-dict";
}

//===----------------------------------------------------------------------===//
// NoC ops (mirror tt-metal noc_async_*; src/dst addressing + a CB index).
//===----------------------------------------------------------------------===//
//
// Addresses are carried as resolved integer NoC addresses (the lowering pass
// computes them from get_noc_addr / get_noc_multicast_addr). `num_tiles` is the
// tile count moved; multicast additionally carries `num_dests` (fan-out).

def TensixflowSchedNocReadOp : Tensixflow_SchedOp<"sched.noc_read"> {
  let summary = "Async NoC read of `num_tiles` tiles from `src_addr` into CB `cb`";
  let description = [{
    Mirrors tt-metal `noc_async_read(src_addr, dst_l1_addr, size)`: asynchronously
    reads `num_tiles` tiles from the NoC source address `src_addr` into circular
    buffer `cb` (the destination L1 address is implied by the CB).
  }];
  let arguments = (ins I64Attr:$src_addr, I64Attr:$cb, I64Attr:$num_tiles);
  let assemblyFormat = "$src_addr `,` $cb `,` $num_tiles attr-dict";
}

def TensixflowSchedNocWriteOp : Tensixflow_SchedOp<"sched.noc_write"> {
  let summary = "Async NoC write of `num_tiles` tiles from CB `cb` to `dst_addr`";
  let description = [{
    Mirrors tt-metal `noc_async_write(src_l1_addr, dst_addr, size)`: asynchronously
    writes `num_tiles` tiles from circular buffer `cb` (the source L1 address is
    implied by the CB) to the NoC destination address `dst_addr`.
  }];
  let arguments = (ins I64Attr:$cb, I64Attr:$dst_addr, I64Attr:$num_tiles);
  let assemblyFormat = "$cb `,` $dst_addr `,` $num_tiles attr-dict";
}

def TensixflowSchedNocMulticastOp : Tensixflow_SchedOp<"sched.noc_multicast"> {
  let summary = "Async NoC multicast write of `num_tiles` tiles from CB `cb`";
  let description = [{
    Mirrors tt-metal `noc_async_write_multicast` (address from
    `get_noc_multicast_addr(x0,y0,x1,y1,addr)`): asynchronously multicasts
    `num_tiles` tiles from circular buffer `cb` to the multicast destination
    address `dst_addr`, fanning out to `num_dests` cores (the NoC row/column
    master-node distribution that alleviates DRAM bandwidth contention).
  }];
  let arguments = (ins I64Attr:$cb, I64Attr:$dst_addr, I64Attr:$num_tiles,
                       I64Attr:$num_dests);
  let assemblyFormat = "$cb `,` $dst_addr `,` $num_tiles `,` $num_dests attr-dict";
}

//===----------------------------------------------------------------------===//
// Compute ops (tile math over CB indices; mirror matmul_tiles + fused epi/prologue).
//===----------------------------------------------------------------------===//

def TensixflowSchedMatmulTilesOp : Tensixflow_SchedOp<"sched.matmul_tiles"> {
  let summary = "Tile matmul: accumulate tiles from CBs `in0` x `in1` into CB `dst`";
  let description = [{
    Mirrors tt-metal `matmul_tiles(in0_cb, in1_cb, dst_cb)`: multiplies the tiles
    staged in circular buffers `in0` and `in1` and accumulates the result into the
    destination tile register / circular buffer `dst` (fp32 accumulate within a
    tile in the simulator's bf16 spatial model).
  }];
  let arguments = (ins I64Attr:$in0, I64Attr:$in1, I64Attr:$dst);
  let assemblyFormat = "$in0 `,` $in1 `,` $dst attr-dict";
}

def TensixflowSchedRmsnormOp : Tensixflow_SchedOp<"sched.rmsnorm"> {
  let summary = "Tile RMSNorm: normalize tiles from CB `input` with CB `gamma` into CB `out`";
  let description = [{
    Tile-level RMSNorm (the fused op's prologue): normalizes the tiles staged in
    circular buffer `input` using the per-column scale in circular buffer `gamma`
    and writes the result to circular buffer `out`. `epsilon` defaults to 1e-6.
  }];
  let arguments = (ins I64Attr:$input, I64Attr:$gamma, I64Attr:$out,
                       DefaultValuedAttr<F32Attr, "1.0e-6">:$epsilon);
  let assemblyFormat = "$input `,` $gamma `,` $out attr-dict";
}

def TensixflowSchedGeluOp : Tensixflow_SchedOp<"sched.gelu"> {
  let summary = "Tile GELU: apply GELU to tiles from CB `input` into CB `out`";
  let description = [{
    Tile-level GELU/SiLU (the fused op's epilogue): applies the activation to the
    tiles staged in circular buffer `input` and writes the result to circular
    buffer `out` (in-place when `input` == `out`).
  }];
  let arguments = (ins I64Attr:$input, I64Attr:$out);
  let assemblyFormat = "$input `,` $out attr-dict";
}

#endif // TENSIXFLOW_SCHEDULE
