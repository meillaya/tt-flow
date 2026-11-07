# tensixflow-report: self-contained offline HTML visualizer. Reads a plan JSON
# (schema/plan.schema.json) via plan::parsePlan, resolves the required device
# block against the generated device table, fills the perf section with the
# static projection model, and emits a single self-contained HTML file with
# inline CSS, embedded JSON, and vanilla inline JS. No external URLs, no CDN, no
# build-time JS framework, and a static render only.
#
# No MLIR dependency: the tool is pure data and rendering over the schema and
# the analysis perf model. MLIRTensixflowAnalysis PUBLIC-links
# MLIRTensixflowScheduler plus MLIRTensixflowSchema plus LLVMSupport, so linking
# it transitively brings the device table, parsePlan, toJson, estimatePerf,
# toPerfEstimate, llvm::cl, MemoryBuffer, and the JSON runtime.

# Land the binary in <build>/bin/ (as mlir/examples/standalone does): installed
# LLVMConfig.cmake does not preset LLVM_RUNTIME_OUTPUT_INTDIR for external
# projects, so add_llvm_executable would otherwise put it under build/tools/.
set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)

add_llvm_executable(tensixflow-report tensixflow-report.cpp)
llvm_update_compile_flags(tensixflow-report)
target_link_libraries(tensixflow-report PRIVATE
  MLIRTensixflowAnalysis
  LLVMSupport
  )
