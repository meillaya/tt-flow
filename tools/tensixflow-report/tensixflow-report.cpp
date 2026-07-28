//===- tensixflow-report.cpp - Self-contained HTML visualizer ---*- C++ -*-===//
//
// TensixFlow: An MLIR Compiler for Tile-Native LLM Kernels on
// Tenstorrent-Style Spatial Accelerators.
//
//===----------------------------------------------------------------------===//
//
// `tensixflow-report` (todo 43, D21): reads a plan/perf JSON (todo 24 schema) via
// plan::parsePlan, fills the perf section with the static perf model (todo 42 —
// the report seam: estimatePerf -> toPerfEstimate), and emits a SINGLE
// self-contained HTML file: inline CSS + the plan JSON embedded as
// `const PLAN = {...}` + vanilla inline JS that statically renders the six
// sections (#core-grid #shards #noc-routes #cb-usage #timeline #bottlenecks).
//
// The output is fully OFFLINE: no https?:// URLs, no <script src>, no CDN, no
// build-time JS framework — validated by a grep + a playwright network-blocked
// load. The render is STATIC (no animation, no interactivity beyond layout).
//
// NO MLIR dependency: the tool links only the schema (D15) + the analysis perf
// model (todo 42) + LLVMSupport.
//
//===----------------------------------------------------------------------===//

#include "ReportTemplate.h"
#include "tensixflow/PlanSchema.h"

#include "PerfModel.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <system_error>

using namespace mlir::tensixflow;
namespace cl = llvm::cl;
namespace json = llvm::json;

namespace {

/// Build the perf-model input from a plan. The perf model (todo 42) reads
/// tensors[0] = activation and tensors[1] = weight (the PlanReport.cpp
/// convention). The translate tool (todo 30) emits ALL function args as tensors
/// (e.g. activation, gamma, weight), so select the activation (first tensor) and
/// the weight (the rank>=2 tensor, other than the activation, with the most
/// elements — skipping rank-1 operands like gamma). The rest of the plan
/// (shard_grid, cb_specs, noc, layout) passes through unchanged.
Plan selectPerfInput(const Plan &plan) {
  Plan perfPlan = plan;
  perfPlan.tensors.clear();
  if (plan.tensors.empty())
    return perfPlan;
  perfPlan.tensors.push_back(plan.tensors[0]); // activation
  const TensorDesc *weight = nullptr;
  int64_t bestElems = -1;
  for (size_t i = 1; i < plan.tensors.size(); ++i) {
    const TensorDesc &t = plan.tensors[i];
    if (t.shape.size() < 2)
      continue; // skip rank-1 operands (e.g. the rmsnorm gamma)
    int64_t elems = 1;
    for (int64_t dim : t.shape)
      elems *= dim;
    if (elems > bestElems) {
      bestElems = elems;
      weight = &t;
    }
  }
  if (weight != nullptr)
    perfPlan.tensors.push_back(*weight);
  else if (plan.tensors.size() >= 2)
    perfPlan.tensors.push_back(plan.tensors[1]);
  return perfPlan;
}

/// Escape `</` as `<\/` so the embedded JSON literal can never close the
/// surrounding <script> tag (standard safe-inline-JSON; `\/` is a valid JSON
/// escape for `/`, so the literal stays parseable by the inline JS).
std::string escapeForScript(const std::string &jsonStr) {
  std::string safe;
  safe.reserve(jsonStr.size());
  for (size_t i = 0; i < jsonStr.size(); ++i) {
    if (jsonStr[i] == '<' && i + 1 < jsonStr.size() && jsonStr[i + 1] == '/') {
      safe += "<\\/";
      ++i;
    } else {
      safe += jsonStr[i];
    }
  }
  return safe;
}

/// Emit the single self-contained HTML document for the plan.
void emitHtml(const Plan &plan, llvm::raw_ostream &os) {
  const std::string safeJson = escapeForScript(toJson(plan));
  os << "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
     << "<meta charset=\"utf-8\">\n"
     << "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
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

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buf =
      llvm::MemoryBuffer::getFile(in);
  if (!buf) {
    llvm::errs() << "error: cannot read " << in << ": "
                 << buf.getError().message() << "\n";
    return 1;
  }

  llvm::Expected<json::Value> parsed = json::parse((*buf)->getBuffer());
  if (!parsed) {
    llvm::errs() << "error: invalid JSON in " << in << ": "
                 << llvm::toString(parsed.takeError()) << "\n";
    return 1;
  }

  std::optional<Plan> planOpt = parsePlan(*parsed);
  if (!planOpt) {
    llvm::errs() << "error: " << in
                 << " is not a valid TensixFlow plan (missing or mistyped "
                    "required field)\n";
    return 1;
  }
  Plan plan = std::move(*planOpt);

  // Fill the perf section from the static perf model (todo 42 report seam) so the
  // rendered #bottlenecks is meaningful regardless of the input's placeholder
  // perf. The render itself stays a static render of the (now perf-filled) plan.
  plan.perf = analysis::toPerfEstimate(analysis::estimatePerf(
      selectPerfInput(plan)));

  std::error_code ec;
  llvm::raw_fd_ostream os(out, ec);
  if (ec) {
    llvm::errs() << "error: cannot open " << out << ": " << ec.message()
                 << "\n";
    return 1;
  }
  emitHtml(plan, os);
  os.close();
  if (os.has_error()) {
    llvm::errs() << "error: failed writing " << out << "\n";
    return 1;
  }

  llvm::outs() << "wrote " << out << "\n";
  return 0;
}
