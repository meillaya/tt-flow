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
#include "EmissionEstimator.h"

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

/// Derive the matmul tile geometry from a plan: the activation tensor is
/// tensors[0] (as in selectPerfInput), [..., M, K], and the weight is the
/// rank>=2 tensor with the most elements, [K, N]. Returns false when the plan
/// does not carry both shapes, which makes the emission block unverifiable.
bool matmulTilesFromPlan(const Plan &perfPlan, const device::DeviceSpec &spec,
                         scheduler::EmissionInputs &inputs) {
  if (perfPlan.tensors.size() < 2) {
    return false;
  }
  const TensorDesc &activation = perfPlan.tensors[0];
  const TensorDesc &weight = perfPlan.tensors[1];
  if (activation.shape.size() < 2 || weight.shape.size() < 2) {
    return false;
  }
  int64_t m = 1;
  int64_t k = activation.shape.back();
  for (size_t i = 0; i + 1 < activation.shape.size(); ++i) {
    m *= activation.shape[i];
  }
  const int64_t n = weight.shape.back();
  const int64_t tileHeight = spec.tileHeight;
  const int64_t tileWidth = spec.tileWidth;
  inputs.mTiles = (m + tileHeight - 1) / tileHeight;
  inputs.nTiles = (n + tileWidth - 1) / tileWidth;
  inputs.kTiles = (k + tileWidth - 1) / tileWidth;
  inputs.tileBytes = spec.tileBytes;
  inputs.cbBudgetBytes = spec.cbBudget(device::kDefaultL1ReserveBytes);
  return true;
}

/// Derive the fused MLP chain geometry from a plan: the activation is
/// tensors[0] ([..., M, K]), the first weight is the rank>=2 tensor with the
/// most elements ([K, H]), and the second weight is the rank-2 tensor whose
/// first dimension equals the hidden width ([H, N]). Returns false when the
/// plan does not carry those shapes, which makes the emission block
/// unverifiable.
bool mlpTilesFromPlan(const Plan &plan, const Plan &perfPlan,
                      const device::DeviceSpec &spec,
                      scheduler::EmissionInputs &inputs, std::string &error) {
  if (perfPlan.tensors.size() < 2 || plan.tensors.size() < 2) {
    error = "the plan does not carry the activation and weight shapes needed "
            "to validate the fused MLP emission block";
    return false;
  }
  const TensorDesc &activation = perfPlan.tensors[0];
  const TensorDesc &weight = perfPlan.tensors[1];
  if (activation.shape.size() < 2 || weight.shape.size() < 2) {
    error = "the fused MLP plan's activation and weight shapes are not rank 2 "
            "or higher";
    return false;
  }
  int64_t m = 1;
  for (size_t i = 0; i + 1 < activation.shape.size(); ++i) {
    m *= activation.shape[i];
  }
  const int64_t k = activation.shape.back();
  const int64_t h = weight.shape.back();

  // Find the index of the first weight in the plan's full tensor list, matching
  // selectPerfInput's choice: the rank>=2 tensor after the activation with the
  // most elements.
  size_t weightIndex = 0;
  int64_t bestElements = -1;
  for (size_t i = 1; i < plan.tensors.size(); ++i) {
    const TensorDesc &tensor = plan.tensors[i];
    if (tensor.shape.size() < 2) {
      continue;
    }
    int64_t elements = 1;
    for (int64_t dim : tensor.shape) {
      elements *= dim;
    }
    if (elements > bestElements) {
      bestElements = elements;
      weightIndex = i;
    }
  }

  // The second weight is the other rank-2 operand whose first dimension is the
  // hidden width.
  int64_t n = 0;
  for (size_t i = 1; i < plan.tensors.size(); ++i) {
    if (i == weightIndex) {
      continue;
    }
    const TensorDesc &tensor = plan.tensors[i];
    if (tensor.shape.size() != 2 || tensor.shape[0] != h) {
      continue;
    }
    n = tensor.shape[1];
    break;
  }
  if (n <= 0) {
    error = "the fused MLP plan does not carry the second weight shape "
            "[H, N] needed to validate the emission block";
    return false;
  }
  const int64_t tileHeight = spec.tileHeight;
  const int64_t tileWidth = spec.tileWidth;
  inputs.mTiles = (m + tileHeight - 1) / tileHeight;
  inputs.kTiles = (k + tileWidth - 1) / tileWidth;
  inputs.hTiles = (h + tileWidth - 1) / tileWidth;
  inputs.nTiles = (n + tileWidth - 1) / tileWidth;
  inputs.tileBytes = spec.tileBytes;
  inputs.cbBudgetBytes = spec.cbBudget(device::kDefaultL1ReserveBytes);
  return true;
}

/// Validate the emission block: the mapping must be one the emitter supports,
/// the work split must fit the plan grid, and every byte count must match the
/// projection from the plan's own tile geometry. The projection models the
/// emitted blocked-reuse weight multicast. A stale or hand-edited block is an
/// error rather than a rendered guess.
bool validateEmission(const Plan &plan, const Plan &perfPlan,
                      const device::DeviceSpec &spec, llvm::StringRef fusion,
                      std::string &error) {
  const EmissionEstimate &emission = plan.emission;
  if (emission.mapping != scheduler::kOutputTileRangesMapping &&
      emission.mapping != scheduler::kBlockedReuseMapping) {
    error = "emission mapping '" + emission.mapping +
            "' is not one the emitter supports; expected '" +
            scheduler::kOutputTileRangesMapping + "' or '" +
            scheduler::kBlockedReuseMapping + "'";
    return false;
  }
  const int64_t gridCores = plan.shardGrid.gridY * plan.shardGrid.gridX;
  if (emission.coresUsed < 1 || emission.coresUsed > gridCores) {
    error = "emission.cores_used " + std::to_string(emission.coresUsed) +
            " is outside the plan grid's 1.." + std::to_string(gridCores) +
            " cores";
    return false;
  }
  if (emission.maxTilesPerCore < 1) {
    error = "emission.max_tiles_per_core must be positive";
    return false;
  }
  if (emission.dramReadBytes <= 0 || emission.dramWriteBytes <= 0) {
    error = "emission DRAM byte counts must be positive";
    return false;
  }

  scheduler::EmissionInputs inputs;
  const bool fused = fusion != "none";
  const bool fusedMlp = fusion == "fused_mlp";
  if (fusedMlp) {
    if (!mlpTilesFromPlan(plan, perfPlan, spec, inputs, error)) {
      return false;
    }
  } else if (!matmulTilesFromPlan(perfPlan, spec, inputs)) {
    error = "the plan does not carry the activation and weight shapes needed "
            "to validate the emission block";
    return false;
  }
  inputs.gridY = plan.shardGrid.gridY;
  inputs.gridX = plan.shardGrid.gridX;
  // The fused emitters stage their own intermediates, so the plan's optional
  // fusion field decides which projection applies: the fused MLP chain uses the
  // 2D block mapping and the hidden width, the fused RMSNorm + MatMul path
  // keeps the contiguous range mapping, and a plan that omits the field is
  // validated as the plain matmul path.
  inputs.fused = fused;
  inputs.fusedMlp = fusedMlp;
  const scheduler::EmissionPlan expected = scheduler::estimateEmission(inputs);
  if (emission.mapping != expected.mapping) {
    error = "emission mapping '" + emission.mapping + "' is not the '" +
            expected.mapping + "' the selector projects for this geometry";
    return false;
  }
  if (emission.coresUsed != expected.coresUsed) {
    error = "emission.cores_used is " + std::to_string(emission.coresUsed) +
            ", projected " + std::to_string(expected.coresUsed);
    return false;
  }
  if (emission.maxTilesPerCore != expected.maxTilesPerCore) {
    error = "emission.max_tiles_per_core is " +
            std::to_string(emission.maxTilesPerCore) + ", projected " +
            std::to_string(expected.maxTilesPerCore);
    return false;
  }
  if (emission.dramReadBytes != expected.dramReadBytes) {
    error = "emission.dram_read_bytes is " +
            std::to_string(emission.dramReadBytes) + ", projected " +
            std::to_string(expected.dramReadBytes);
    return false;
  }
  if (emission.dramWriteBytes != expected.dramWriteBytes) {
    error = "emission.dram_write_bytes is " +
            std::to_string(emission.dramWriteBytes) + ", projected " +
            std::to_string(expected.dramWriteBytes);
    return false;
  }
  if (emission.fusedStageTiles != expected.fusedStageTiles) {
    error = "emission.fused_stage_tiles is " +
            std::to_string(emission.fusedStageTiles) + ", projected " +
            std::to_string(expected.fusedStageTiles);
    return false;
  }
  if (emission.fusedFeasible != expected.fusedFeasible) {
    error = "emission.fused_feasible does not match the projection";
    return false;
  }
  return true;
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

  // The perf model reads the activation and weight operands; selecting them
  // once gives both the perf fill and the emission validation the same view.
  Plan perfPlan = selectPerfInput(plan);

  // The emitted-path projection depends on the fusion decision: the fused
  // emitters stage their own intermediates. The plan document carries it as the
  // optional report field "fusion"; a plain matmul plan omits or zeroes it.
  llvm::StringRef fusion = "none";
  if (const json::Object *root = parsed->getAsObject()) {
    if (std::optional<llvm::StringRef> name = root->getString("fusion")) {
      fusion = *name;
    }
  }

  // Validate the required emitted-path block against the plan's own geometry
  // before rendering it.
  std::string emissionError;
  if (!validateEmission(plan, perfPlan, *deviceSpec, fusion, emissionError)) {
    llvm::errs() << "error: " << in << " has an invalid emission block: "
                 << emissionError << "\n";
    return 1;
  }

  // Fill the perf section from the static projection model so the rendered
  // #bottlenecks is meaningful regardless of the input's placeholder perf.
  plan.perf = analysis::toPerfEstimate(
      analysis::estimatePerf(perfPlan, *deviceSpec));

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
