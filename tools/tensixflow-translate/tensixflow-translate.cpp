//===- tensixflow-translate.cpp - TensixFlow translate driver ---*- C++ -*-===//
//
// TensixFlow: An MLIR Compiler for Tile-Native LLM Kernels on
// Tenstorrent-Style Spatial Accelerators.
//
//===----------------------------------------------------------------------===//
//
// `tensixflow-translate` driver modeled on llvm-project's
// mlir/tools/mlir-translate. Registers a single translation,
// `--tensixflow-to-json`, that serializes a tensixflow module to a
// simple-but-valid JSON document. This is a SKELETON: it emits a meaningful
// structure (functions / arguments / ops) but deliberately does NOT define the
// full plan/perf/viz schema — that lands in todo 24.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include "tensixflow/TensixflowDialect.h"
#include "tensixflow/TensixflowOps.h"

using namespace mlir;

namespace {

/// Register the dialects the input module may reference so it parses.
void registerTensixflowDialects(DialectRegistry &registry) {
  registry.insert<mlir::tensixflow::TensixflowDialect,
                  mlir::func::FuncDialect, mlir::arith::ArithDialect>();
}

/// Render an MLIR type to its canonical textual form.
std::string typeToString(Type type) {
  std::string result;
  llvm::raw_string_ostream os(result);
  os << type;
  return result;
}

/// Serialize a tensixflow module to a JSON skeleton.
///
/// Shape (skeleton, NOT the full todo-24 schema):
///   {
///     "dialect": "tensixflow",
///     "functions": [
///       {
///         "name": "...",
///         "arguments": ["tensor<...>", ...],
///         "results":   ["tensor<...>", ...],
///         "ops": [ { "name": "tensixflow.gelu",
///                    "num_operands": 1, "num_results": 1 }, ... ]
///       }
///     ]
///   }
LogicalResult tensixflowToJson(Operation *op, raw_ostream &os) {
  namespace json = llvm::json;

  json::Array functions;
  op->walk([&](mlir::tensixflow::TensixflowFuncOp func) {
    json::Object function;
    function["name"] = func.getName().str();

    json::Array arguments;
    for (Type type : func.getArgumentTypes())
      arguments.push_back(json::Value(typeToString(type)));
    function["arguments"] = std::move(arguments);

    json::Array results;
    for (Type type : func.getResultTypes())
      results.push_back(json::Value(typeToString(type)));
    function["results"] = std::move(results);

    json::Array ops;
    for (Operation &nested : func.getBody().front()) {
      json::Object entry;
      entry["name"] = nested.getName().getStringRef().str();
      entry["num_operands"] = static_cast<int64_t>(nested.getNumOperands());
      entry["num_results"] = static_cast<int64_t>(nested.getNumResults());
      ops.push_back(json::Value(std::move(entry)));
    }
    function["ops"] = std::move(ops);

    functions.push_back(json::Value(std::move(function)));
  });

  json::Object root;
  root["dialect"] = "tensixflow";
  root["functions"] = std::move(functions);

  // Pretty-print with a 2-space indent; output is valid JSON (json.load-safe).
  os << llvm::formatv("{0:2}", json::Value(std::move(root))) << "\n";
  return success();
}

} // namespace

// Self-registers before main(): the `--tensixflow-to-json` translation plus the
// dialects needed to parse its input.
static TranslateFromMLIRRegistration reg(
    "tensixflow-to-json", "Serialize a tensixflow module to JSON",
    tensixflowToJson, registerTensixflowDialects);

int main(int argc, char **argv) {
  return failed(
      mlirTranslateMain(argc, argv, "TensixFlow MLIR translation tool\n"));
}
