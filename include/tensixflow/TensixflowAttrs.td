//===- TensixflowAttrs.td - Tensixflow attributes ----------*- tablegen -*-===//
//
// TensixFlow: An MLIR Compiler for Tile-Native LLM Kernels on
// Tenstorrent-Style Spatial Accelerators.
//
//===----------------------------------------------------------------------===//

#ifndef TENSIXFLOW_ATTRS
#define TENSIXFLOW_ATTRS

include "tensixflow/TensixflowDialect.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// Tensixflow memory space enum.
//===----------------------------------------------------------------------===//

def Tensixflow_MemorySpaceDRAM : I32EnumAttrCase<"DRAM", 0>;
def Tensixflow_MemorySpaceL1 : I32EnumAttrCase<"L1", 1>;

def Tensixflow_MemorySpace : I32EnumAttr<"MemorySpace",
    "Tensixflow memory space (DRAM or on-core L1)", [
  Tensixflow_MemorySpaceDRAM,
  Tensixflow_MemorySpaceL1
]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tensixflow";
}

//===----------------------------------------------------------------------===//
// Tensixflow tile-layout encoding attribute.
//===----------------------------------------------------------------------===//
//
// Assembly format (custom, round-trips):
//
//   #tensixflow.tile_layout<tile 32x32, grid 8x8, mem L1>
//
// `tile HxW` is the tile shape (defaults to 32x32, the modeled Tensix tile),
// `grid YxX` is the core grid the value is placed on, and `mem` is the memory
// space (DRAM or L1).

def Tensixflow_TileLayoutAttr : AttrDef<Tensixflow_Dialect, "TensixflowTileLayout"> {
  let summary = "Tile layout encoding (32x32 tile, core grid, memory space)";
  let mnemonic = "tile_layout";
  let description = [{
    A discardable encoding attribute describing how a tensor value is laid out
    on the modeled device: the tile shape (default 32x32), the core grid it is
    placed on (`gridY` x `gridX`), and the memory space (DRAM or L1) it lives
    in. Attaching it does not change tensor shapes; it is pure metadata consumed
    by later layout/sharding passes.
  }];
  let parameters = (ins
    "unsigned":$tileHeight,
    "unsigned":$tileWidth,
    "unsigned":$gridY,
    "unsigned":$gridX,
    "::mlir::tensixflow::MemorySpace":$memorySpace
  );
  let hasCustomAssemblyFormat = 1;
  let extraClassDeclaration = [{
    /// Convenience builder defaulting the tile shape to the modeled 32x32 tile.
    static ::mlir::tensixflow::TensixflowTileLayoutAttr
    get(::mlir::MLIRContext *ctx, unsigned gridY, unsigned gridX,
        ::mlir::tensixflow::MemorySpace memorySpace) {
      return get(ctx, /*tileHeight=*/32, /*tileWidth=*/32, gridY, gridX,
                 memorySpace);
    }
  }];
}

//===----------------------------------------------------------------------===//
// Tensixflow scheduler-plan (schedule) attribute.
//===----------------------------------------------------------------------===//
//
// Assembly format (custom, round-trips):
//
//   #tensixflow.schedule<grid 8x8, act "row-major", weight "multicast",
//     storage "L1", fusion "fused_rmsnorm_matmul",
//     cbs [0:160:2048, 1:160:2048, 16:160:2048],
//     dram_read 117440512, dram_write 50331648, l1 16777216,
//     multicast 100663296, saved_read 16777216, saved_write 16777216,
//     saved_intermediate 16777216>
//
// `grid YxX` is the core grid; `act`/`weight`/`storage`/`fusion` are the layout
// decisions (quoted strings); `cbs [idx:depth:tileBytes, ...]` are the allocated
// circular buffers (parallel index/depth/tile-bytes triples, possibly empty
// between the two planning passes); the trailing integers are the static NoC
// traffic estimate (DRAM/L1/multicast bytes) plus the fusion savings breakdown.

def Tensixflow_ScheduleAttr : AttrDef<Tensixflow_Dialect, "TensixflowSchedule"> {
  let summary = "Scheduler plan: grid, sharding, weight strategy, CB specs, NoC";
  let mnemonic = "schedule";
  let description = [{
    A structured attribute recording the scheduler's plan for a graph: the core
    grid (gridY x gridX), the activation-sharding scheme, the weight-distribution
    strategy (replicate/shard/multicast), where the fused intermediate is stored
    (L1 when fused, else DRAM), the fusion decision (fused_rmsnorm_matmul /
    fused_mlp / none), the allocated circular buffers (parallel index/depth/
    tile-bytes arrays), and the static NoC traffic estimate (DRAM read/write, L1,
    multicast bytes) plus the fusion savings (saved DRAM reads/writes and the
    intermediate bytes that never round-trip DRAM). Attached to the module by
    --tensixflow-plan-noc-sharding (grid/sharding/weight/fusion/NoC, CB arrays
    empty) and completed by --tensixflow-allocate-circular-buffers (CB arrays
    filled). Consumed by tensixflow-translate --tensixflow-to-json. Pure metadata.
  }];
  // Variable-length fields are stored as attribute HANDLES (StringAttr /
  // ArrayAttr-of-IntegerAttr), NOT as StringRef / ArrayRef<int64_t>: mixing the
  // latter two in one generated parametric storage corrupts the string-blob
  // offsets (the field adjacent to the arrays reads a dangling pointer). Handles
  // keep the storage struct purely fixed-size (scalars + attribute pointers).
  let parameters = (ins
    "unsigned":$gridY,
    "unsigned":$gridX,
    "::mlir::StringAttr":$activationSharding,
    "::mlir::StringAttr":$weightStrategy,
    "::mlir::StringAttr":$intermediateStorage,
    "::mlir::StringAttr":$fusion,
    "::mlir::ArrayAttr":$cbIndices,
    "::mlir::ArrayAttr":$cbDepths,
    "::mlir::ArrayAttr":$cbTileBytes,
    "int64_t":$dramReadBytes,
    "int64_t":$dramWriteBytes,
    "int64_t":$l1Bytes,
    "int64_t":$multicastBytes,
    "int64_t":$savedDramReadBytes,
    "int64_t":$savedDramWriteBytes,
    "int64_t":$savedIntermediateBytes
  );
  let hasCustomAssemblyFormat = 1;
}

#endif // TENSIXFLOW_ATTRS
