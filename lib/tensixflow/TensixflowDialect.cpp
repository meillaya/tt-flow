//===- TensixflowDialect.cpp - Tensixflow dialect ---------------*- C++ -*-===//
//
// TensixFlow: An MLIR Compiler for Tile-Native LLM Kernels on
// Tenstorrent-Style Spatial Accelerators.
//
//===----------------------------------------------------------------------===//

#include "tensixflow/TensixflowDialect.h"
#include "tensixflow/TensixflowAttrs.h"
#include "tensixflow/TensixflowOps.h"
#include "tensixflow/TensixflowSchedule.h"

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
// TensixflowScheduleAttr custom assembly format.
//===----------------------------------------------------------------------===//
//
//   schedule-attr ::= `#tensixflow.schedule` `<`
//                       `grid` static-dimension-list `,`
//                       `act` string `,` `weight` string `,`
//                       `storage` string `,` `fusion` string `,`
//                       `cbs` `[` (int `:` int `:` int (`,` ...)?)? `]` `,`
//                       `dram_read` int `,` `dram_write` int `,` `l1` int `,`
//                       `multicast` int `,` `saved_read` int `,`
//                       `saved_write` int `,` `saved_intermediate` int
//                     `>`
//
// The `#tensixflow.schedule` mnemonic is consumed/emitted by the dialect
// dispatch (useDefaultAttributePrinterParser); parse/print below handle the
// `< ... >` body. The CB list may be empty between the two planning passes.

namespace {
/// Pack int64s into an ArrayAttr of i64 IntegerAttrs (the CB-array storage form).
ArrayAttr i64ArrayAttr(MLIRContext *ctx, ArrayRef<int64_t> values) {
  SmallVector<Attribute> attrs;
  attrs.reserve(values.size());
  IntegerType i64 = IntegerType::get(ctx, 64);
  for (int64_t v : values)
    attrs.push_back(IntegerAttr::get(i64, v));
  return ArrayAttr::get(ctx, attrs);
}

/// Read the i-th element of an ArrayAttr of i64 IntegerAttrs.
int64_t i64At(ArrayAttr array, size_t i) {
  return cast<IntegerAttr>(array[i]).getInt();
}
} // namespace

Attribute TensixflowScheduleAttr::parse(AsmParser &parser, Type type) {
  if (type) {
    parser.emitError(parser.getNameLoc(),
                     "schedule attribute does not take a type");
    return Attribute();
  }

  unsigned gridY = 0, gridX = 0;
  std::string activationSharding, weightStrategy, intermediateStorage, fusion;
  SmallVector<int64_t> cbIndices, cbDepths, cbTileBytes;
  int64_t dramRead = 0, dramWrite = 0, l1 = 0, multicast = 0;
  int64_t savedRead = 0, savedWrite = 0, savedIntermediate = 0;

  if (parser.parseLess() || parser.parseKeyword("grid") ||
      parseHW(parser, gridY, gridX, "core grid") || parser.parseComma() ||
      parser.parseKeyword("act") || parser.parseString(&activationSharding) ||
      parser.parseComma() || parser.parseKeyword("weight") ||
      parser.parseString(&weightStrategy) || parser.parseComma() ||
      parser.parseKeyword("storage") ||
      parser.parseString(&intermediateStorage) || parser.parseComma() ||
      parser.parseKeyword("fusion") || parser.parseString(&fusion) ||
      parser.parseComma() || parser.parseKeyword("cbs") ||
      parser.parseLSquare())
    return Attribute();

  // Circular-buffer triples idx:depth:tileBytes, comma-separated; an empty list
  // (parseOptionalRSquare succeeds at once) is valid between the two passes.
  if (failed(parser.parseOptionalRSquare())) {
    do {
      int64_t index = 0, depth = 0, tileBytes = 0;
      if (parser.parseInteger(index) || parser.parseColon() ||
          parser.parseInteger(depth) || parser.parseColon() ||
          parser.parseInteger(tileBytes))
        return Attribute();
      cbIndices.push_back(index);
      cbDepths.push_back(depth);
      cbTileBytes.push_back(tileBytes);
    } while (succeeded(parser.parseOptionalComma()));
    if (parser.parseRSquare())
      return Attribute();
  }

  if (parser.parseComma() || parser.parseKeyword("dram_read") ||
      parser.parseInteger(dramRead) || parser.parseComma() ||
      parser.parseKeyword("dram_write") || parser.parseInteger(dramWrite) ||
      parser.parseComma() || parser.parseKeyword("l1") ||
      parser.parseInteger(l1) || parser.parseComma() ||
      parser.parseKeyword("multicast") || parser.parseInteger(multicast) ||
      parser.parseComma() || parser.parseKeyword("saved_read") ||
      parser.parseInteger(savedRead) || parser.parseComma() ||
      parser.parseKeyword("saved_write") || parser.parseInteger(savedWrite) ||
      parser.parseComma() || parser.parseKeyword("saved_intermediate") ||
      parser.parseInteger(savedIntermediate) || parser.parseGreater())
    return Attribute();

  MLIRContext *ctx = parser.getContext();
  return TensixflowScheduleAttr::get(
      ctx, gridY, gridX, StringAttr::get(ctx, activationSharding),
      StringAttr::get(ctx, weightStrategy),
      StringAttr::get(ctx, intermediateStorage), StringAttr::get(ctx, fusion),
      i64ArrayAttr(ctx, cbIndices), i64ArrayAttr(ctx, cbDepths),
      i64ArrayAttr(ctx, cbTileBytes), dramRead, dramWrite, l1, multicast,
      savedRead, savedWrite, savedIntermediate);
}

void TensixflowScheduleAttr::print(AsmPrinter &printer) const {
  printer << "<grid ";
  printer.printDimensionList(
      {static_cast<int64_t>(getGridY()), static_cast<int64_t>(getGridX())});
  printer << ", act ";
  printer.printString(getActivationSharding().getValue());
  printer << ", weight ";
  printer.printString(getWeightStrategy().getValue());
  printer << ", storage ";
  printer.printString(getIntermediateStorage().getValue());
  printer << ", fusion ";
  printer.printString(getFusion().getValue());
  printer << ", cbs [";
  ArrayAttr indices = getCbIndices();
  ArrayAttr depths = getCbDepths();
  ArrayAttr tileBytes = getCbTileBytes();
  for (size_t i = 0, e = indices.size(); i < e; ++i) {
    if (i != 0)
      printer << ", ";
    printer << i64At(indices, i) << ":" << i64At(depths, i) << ":"
            << i64At(tileBytes, i);
  }
  printer << "], dram_read " << getDramReadBytes()
          << ", dram_write " << getDramWriteBytes() << ", l1 " << getL1Bytes()
          << ", multicast " << getMulticastBytes()
          << ", saved_read " << getSavedDramReadBytes()
          << ", saved_write " << getSavedDramWriteBytes()
          << ", saved_intermediate " << getSavedIntermediateBytes() << ">";
}

//===----------------------------------------------------------------------===//
// TensixflowEmissionAttr custom assembly format.
//===----------------------------------------------------------------------===//
//
//   emission-attr ::= `#tensixflow.emission` `<`
//                       `mapping` string `,`
//                       `cores_used` int `,`
//                       `max_tiles_per_core` int `,`
//                       `dram_read` int `,` `dram_write` int `,`
//                       `fused_stage_tiles` int `,`
//                       `fused_feasible` (`true` | `false`)
//                     `>`
//
// The `#tensixflow.emission` mnemonic is consumed/emitted by the dialect
// dispatch (useDefaultAttributePrinterParser); parse/print below handle the
// `< ... >` body. The mapping field carries one of the two emitted matmul
// mappings. Example:
//
//   #tensixflow.emission<mapping "output-tile-ranges", cores_used 64,
//     max_tiles_per_core 4, dram_read 16777216, dram_write 524288,
//     fused_stage_tiles 32, fused_feasible true>

Attribute TensixflowEmissionAttr::parse(AsmParser &parser, Type type) {
  if (type) {
    parser.emitError(parser.getNameLoc(),
                     "emission attribute does not take a type");
    return Attribute();
  }

  std::string mapping;
  int64_t coresUsed = 0, maxTilesPerCore = 0;
  int64_t dramRead = 0, dramWrite = 0, fusedStageTiles = 0;
  bool fusedFeasible = false;
  StringRef feasibleKeyword;
  if (parser.parseLess() || parser.parseKeyword("mapping") ||
      parser.parseString(&mapping) || parser.parseComma() ||
      parser.parseKeyword("cores_used") || parser.parseInteger(coresUsed) ||
      parser.parseComma() || parser.parseKeyword("max_tiles_per_core") ||
      parser.parseInteger(maxTilesPerCore) || parser.parseComma() ||
      parser.parseKeyword("dram_read") || parser.parseInteger(dramRead) ||
      parser.parseComma() || parser.parseKeyword("dram_write") ||
      parser.parseInteger(dramWrite) || parser.parseComma() ||
      parser.parseKeyword("fused_stage_tiles") ||
      parser.parseInteger(fusedStageTiles) || parser.parseComma() ||
      parser.parseKeyword("fused_feasible") ||
      parser.parseKeyword(&feasibleKeyword) || parser.parseGreater())
    return Attribute();

  if (feasibleKeyword == "true") {
    fusedFeasible = true;
  } else if (feasibleKeyword == "false") {
    fusedFeasible = false;
  } else {
    parser.emitError(parser.getNameLoc(),
                     "invalid fused_feasible keyword '" + feasibleKeyword +
                         "', expected true or false");
    return Attribute();
  }

  MLIRContext *ctx = parser.getContext();
  return TensixflowEmissionAttr::get(
      ctx, StringAttr::get(ctx, mapping), coresUsed, maxTilesPerCore, dramRead,
      dramWrite, fusedStageTiles, fusedFeasible);
}

void TensixflowEmissionAttr::print(AsmPrinter &printer) const {
  printer << "<mapping ";
  printer.printString(getMapping().getValue());
  printer << ", cores_used " << getCoresUsed() << ", max_tiles_per_core "
          << getMaxTilesPerCore() << ", dram_read " << getDramReadBytes()
          << ", dram_write " << getDramWriteBytes() << ", fused_stage_tiles "
          << getFusedStageTiles() << ", fused_feasible "
          << (getFusedFeasible() ? "true" : "false") << ">";
}

//===----------------------------------------------------------------------===//
// Tensixflow dialect.
//===----------------------------------------------------------------------===//

void TensixflowDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "tensixflow/TensixflowOps.cpp.inc"
      >();
#undef GET_OP_LIST
  // Tile-level schedule IR ops: the tensixflow.sched.* set, the
  // codegen and simulator contract. Same dialect, registered alongside
  // the tensor ops; their GET_OP_CLASSES definitions live in
  // TensixflowScheduleOps.cpp (the single TU with the complete op storage).
  addOperations<
#define GET_OP_LIST
#include "tensixflow/TensixflowSchedule.cpp.inc"
      >();
  addAttributes<TensixflowTileLayoutAttr, TensixflowScheduleAttr,
                TensixflowEmissionAttr>();
}
