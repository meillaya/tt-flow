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

#endif // TENSIXFLOW_ATTRS
