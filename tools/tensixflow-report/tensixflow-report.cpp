//===- tensixflow-report.cpp - Self-contained HTML visualizer ---*- C++ -*-===//
//
// TensixFlow: An MLIR Compiler for Tile-Native LLM Kernels on
// Tenstorrent-Style Spatial Accelerators.
//
//===----------------------------------------------------------------------===//
//
// tensixflow-report reads a plan JSON (schema/plan.schema.json), fills the perf
// section with the static projection model, and emits a single self-contained
// HTML file: inline CSS, the plan JSON embedded as `const PLAN = {...}`, and
// vanilla inline JS that statically renders the six sections (#core-grid,
// #shards, #noc-routes, #cb-usage, #timeline, #bottlenecks).
//
// The output is fully offline: no https?:// URLs, no <script src>, no CDN, and
// no build-time JS framework. The render is static. The device block from the
// plan is resolved against the generated device table; an unknown device name
// is an error, not a guess.
//
// No MLIR dependency: the tool links only the schema, the analysis perf model,
// and LLVMSupport.
//
//===----------------------------------------------------------------------===//

#include "ReportTemplate.h"

#include "tensixflow/Generated/DeviceSpecs.h"
#include "tensixflow/PlanSchema.h"

#include "PerfModel.h"

#include "DeviceSelection.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <string>
#include <system_error>

using namespace mlir::tensixflow;
namespace cl = llvm::cl;
namespace json = llvm::json;

namespace {

/// Build the perf-model input from a plan. The perf model reads tensors[0] as
/// the activation and tensors[1] as the weight (the PlanReport convention). The
/// translate tool emits all function arguments as tensors (for example
/// activation, gamma, weight), so select the activation (first tensor) and the
/// weight (the rank>=2 tensor other than the activation with the most
/// elements, skipping rank-1 operands like gamma). The rest of the plan
/// (device, shard grid, cb specs, noc, layout) passes through unchanged.
Plan selectPerfInput(const Plan &plan) {
  Plan perfPlan = plan;
  perfPlan.tensors.clear();
  if (plan.tensors.empty()) {
    return perfPlan;
  }
  perfPlan.tensors.push_back(plan.tensors[0]); // activation

  const TensorDesc *weight = nullptr;
  int64_t bestElements = -1;
  for (size_t i = 1; i < plan.tensors.size(); ++i) {
    const TensorDesc &tensor = plan.tensors[i];
    if (tensor.shape.size() < 2) {
      continue; // skip rank-1 operands (e.g. the rmsnorm gamma)
    }
    int64_t elements = 1;
    for (int64_t dim : tensor.shape) {
      elements *= dim;
    }
    if (elements > bestElements) {
      bestElements = elements;
      weight = &tensor;
    }
  }
  if (weight != nullptr) {
    perfPlan.tensors.push_back(*weight);
  } else if (plan.tensors.size() >= 2) {
    perfPlan.tensors.push_back(plan.tensors[1]);
  }
  return perfPlan;
}

/// Escape `</` as `<\/` so the embedded JSON literal can never close the
/// surrounding <script> tag (standard safe-inline JSON; `\/` is a valid JSON
/// escape for `/`, so the literal stays parseable by the inline JS).
std::string escapeForScript(const std::string &jsonText) {
  std::string safe;
  safe.reserve(jsonText.size());
  for (size_t i = 0; i < jsonText.size(); ++i) {
    if (jsonText[i] == '<' && i + 1 < jsonText.size() &&
        jsonText[i + 1] == '/') {
      safe += "<\\/";
      ++i;
    } else {
      safe += jsonText[i];
    }
  }
  return safe;
}

/// Emit the single self-contained HTML document for the plan.
void emitHtml(const Plan &plan, llvm::raw_ostream &out) {
  const std::string safeJson = escapeForScript(toJson(plan));
  out << "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
      << "<meta charset=\"utf-8\">\n"
      << "<meta name=\"viewport\" content=\"width=device-width, "
         "initial-scale=1\">\n"
      << "<title>TensixFlow Plan Report</title>\n<style>\n"
      << report::kCss << "\n</style>\n</head>\n<body>\n"
      << report::kSections << "\n"
      << "<script>const PLAN = " << safeJson << ";</script>\n"
      << "<script>\n" << report::kJs << "\n</script>\n"
      << "</body>\n</html>\n";
}

} // namespace

int main(int argc, char **argv) {
  cl::OptionCategory category("tensixflow-report options");
  cl::opt<std::string> in("in", cl::desc("<input plan.json>"), cl::Required,
                          cl::cat(category));
  cl::opt<std::string> out("out", cl::desc("<output report.html>"),
                           cl::Required, cl::cat(category));
  cl::ParseCommandLineOptions(argc, argv,
                              "TensixFlow self-contained HTML visualizer\n");

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(in);
  if (!buffer) {
    llvm::errs() << "error: cannot read " << in << ": "
                 << buffer.getError().message() << "\n";
    return 1;
  }

  llvm::Expected<json::Value> parsed = json::parse((*buffer)->getBuffer());
  if (!parsed) {
    llvm::errs() << "error: invalid JSON in " << in << ": "
                 << llvm::toString(parsed.takeError()) << "\n";
    return 1;
  }

  std::optional<Plan> planOpt = parsePlan(*parsed);
  if (!planOpt) {
    llvm::errs() << "error: " << in
                 << " is not a valid TensixFlow plan (missing or mistyped "
                    "required field, including the device block)\n";
    return 1;
  }
  Plan plan = std::move(*planOpt);

  // Resolve the plan's device against the generated table. An unknown name is a
  // hard error: rendering with guessed device numbers would invent facts.
  const device::DeviceSpec *deviceSpec = device::lookup(plan.device.name);
  if (deviceSpec == nullptr) {
    llvm::errs() << "error: plan names unknown device '" << plan.device.name
                 << "'; valid devices: " << device::validDeviceNames() << "\n";
    return 1;
  }

  // Fill the perf section from the static projection model so the rendered
  // #bottlenecks is meaningful regardless of the input's placeholder perf.
  plan.perf = analysis::toPerfEstimate(
      analysis::estimatePerf(selectPerfInput(plan), *deviceSpec));

  std::error_code errorCode;
  llvm::raw_fd_ostream file(out, errorCode);
  if (errorCode) {
    llvm::errs() << "error: cannot open " << out << ": "
                 << errorCode.message() << "\n";
    return 1;
  }
  emitHtml(plan, file);
  file.close();
  if (file.has_error()) {
    llvm::errs() << "error: failed writing " << out << "\n";
    return 1;
  }

  llvm::outs() << "wrote " << out << "\n";
  return 0;
}
