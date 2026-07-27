//===- tensixflow-opt.cpp - TensixFlow optimizer driver ---------*- C++ -*-===//
//
// TensixFlow: An MLIR Compiler for Tile-Native LLM Kernels on
// Tenstorrent-Style Spatial Accelerators.
//
//===----------------------------------------------------------------------===//
//
// Skeleton `tensixflow-opt` driver modeled on llvm-project's
// mlir/examples/standalone/standalone-opt. Registers only the upstream Func +
// Arith dialects for now; the TensixflowDialect is registered here in a later
// todo.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "tensixflow/TensixflowDialect.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::DialectRegistry registry;
  registry.insert<mlir::tensixflow::TensixflowDialect,
                  mlir::func::FuncDialect, mlir::arith::ArithDialect>();
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "tensixflow optimizer\n", registry));
}
