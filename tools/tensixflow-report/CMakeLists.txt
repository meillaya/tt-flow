# tensixflow-report: self-contained offline HTML visualizer (todo 43, D21). Reads a
# plan/perf JSON (todo 24 schema) via plan::parsePlan, fills the perf section with
# the static perf model (todo 42, the report seam), and emits a SINGLE self-contained
# HTML file (inline CSS + embedded JSON + vanilla inline JS; NO external URLs, NO CDN,
# NO build-time JS framework). STATIC render only — no animation, no interactivity.
#
# NO MLIR dependency: the tool is pure data + rendering over the schema (D15) + the
# analysis perf model. MLIRTensixflowAnalysis PUBLIC-links MLIRTensixflowScheduler +
# MLIRTensixflowSchema + LLVMSupport, so linking it transitively brings parsePlan /
# toJson / estimatePerf / toPerfEstimate + the llvm::cl + MemoryBuffer + json runtime.

# Land the binary in <build>/bin/ (as mlir/examples/standalone does): installed
# LLVMConfig.cmake does not preset LLVM_RUNTIME_OUTPUT_INTDIR for external projects,
# so add_llvm_executable would otherwise put it under build/tools/.
set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)

add_llvm_executable(tensixflow-report tensixflow-report.cpp)
llvm_update_compile_flags(tensixflow-report)
target_link_libraries(tensixflow-report PRIVATE
  MLIRTensixflowAnalysis
  LLVMSupport
  )
