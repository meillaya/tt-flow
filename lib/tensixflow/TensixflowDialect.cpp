//===- TensixflowDialect.cpp - Tensixflow dialect ---------------*- C++ -*-===//
//
// TensixFlow: An MLIR Compiler for Tile-Native LLM Kernels on
// Tenstorrent-Style Spatial Accelerators.
//
//===----------------------------------------------------------------------===//

#include "tensixflow/TensixflowDialect.h"
#include "tensixflow/TensixflowAttrs.h"
#include "tensixflow/TensixflowOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::tensixflow;

#include "tensixflow/TensixflowEnums.cpp.inc"
#include "tensixflow/TensixflowOpsDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "tensixflow/TensixflowAttrs.cpp.inc"

//===----------------------------------------------------------------------===//
// TensixflowTileLayoutAttr custom assembly format.
//===----------------------------------------------------------------------===//
//
//   tile-layout-attr ::= `#tensixflow.tile_layout` `<`
//                          `tile` static-dimension-list `,`
//                          `grid` static-dimension-list `,`
//                          `mem` (`DRAM` | `L1`)
//                        `>`
//
// The `#tensixflow.tile_layout` mnemonic is consumed/emitted by the dialect
// dispatch (useDefaultAttributePrinterParser); parse/print below handle the
// `< ... >` body. Example:
//
//   #tensixflow.tile_layout<tile 32x32, grid 8x8, mem L1>

namespace {
/// Parse a mandatory `HxW` static dimension list (exactly two dimensions).
ParseResult parseHW(AsmParser &parser, unsigned &height, unsigned &width,
                    const Twine &what) {
  SmallVector<int64_t, 2> dims;
  if (parser.parseDimensionList(dims, /*allowDynamic=*/false,
                                /*withTrailingX=*/false))
    return failure();
  if (dims.size() != 2)
    return parser.emitError(parser.getNameLoc())
           << "expected " << what << " dimensions in the form HxW, got "
           << dims.size() << " dimension(s)";
  height = static_cast<unsigned>(dims[0]);
  width = static_cast<unsigned>(dims[1]);
  return success();
}
} // namespace

Attribute TensixflowTileLayoutAttr::parse(AsmParser &parser, Type type) {
  if (type) {
    parser.emitError(parser.getNameLoc(),
                     "tile_layout attribute does not take a type");
    return Attribute();
  }

  unsigned tileHeight = 0, tileWidth = 0, gridY = 0, gridX = 0;
  StringRef memKeyword;
  if (parser.parseLess() || parser.parseKeyword("tile") ||
      parseHW(parser, tileHeight, tileWidth, "tile") || parser.parseComma() ||
      parser.parseKeyword("grid") || parseHW(parser, gridY, gridX,
                                             "core grid") ||
      parser.parseComma() || parser.parseKeyword("mem") ||
      parser.parseKeyword(&memKeyword) || parser.parseGreater())
    return Attribute();

  std::optional<MemorySpace> memorySpace = symbolizeMemorySpace(memKeyword);
  if (!memorySpace) {
    parser.emitError(parser.getNameLoc(),
                     "invalid memory space keyword '" + memKeyword +
                         "', expected one of: DRAM, L1");
    return Attribute();
  }

  return TensixflowTileLayoutAttr::get(parser.getContext(), tileHeight,
                                       tileWidth, gridY, gridX, *memorySpace);
}

void TensixflowTileLayoutAttr::print(AsmPrinter &printer) const {
  printer << "<tile ";
  printer.printDimensionList({static_cast<int64_t>(getTileHeight()),
                              static_cast<int64_t>(getTileWidth())});
  printer << ", grid ";
  printer.printDimensionList(
      {static_cast<int64_t>(getGridY()), static_cast<int64_t>(getGridX())});
  printer << ", mem " << stringifyMemorySpace(getMemorySpace()) << ">";
}

//===----------------------------------------------------------------------===//
// Tensixflow dialect.
//===----------------------------------------------------------------------===//

void TensixflowDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "tensixflow/TensixflowOps.cpp.inc"
      >();
  addAttributes<TensixflowTileLayoutAttr>();
}
