# TensixFlow benchmarks (todo 44, D21): SIMPLE timing loops -- NOT a benchmarking
# framework (no Google Benchmark dependency). Each bench drives the static perf
# model (todo 42: buildPlanReport -> estimatePerf -> formatPerfReport, unfused vs
# fused) for the HEADLINE figures and runs a plain std::chrono loop over the
# functional tile simulator (todo 39: runTileSimulator) for the fused path. The
# perf figures are STATIC ESTIMATES and the simulator timing is a
# functional-execution measure -- NEITHER is wall-clock hardware performance.
#
# Land the binaries in <build>/benchmarks/ (the acceptance runs
# ./build/benchmarks/bench_rmsnorm_matmul). Plain add_executable (not
# add_llvm_executable) mirrors the test/unit pattern; the MLIR libs come from the
# linked targets' PUBLIC interface + the top-level include_directories.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/benchmarks)

# Shared link set: the Analysis lib (perf model; PUBLIC-links the Scheduler +
# Schema libs), the Sim lib (tile simulator; PUBLIC-links MLIRTensixflow), and the
# MLIR libs needed to parse the reduced func and run the infer-layout -> fuse ->
# plan -> allocate -> lower pipeline. MLIRTransforms is required by the fuse
# pass's greedy rewrite driver (same set as tools/tensixflow-simulate).
set(TENSIXFLOW_BENCH_LIBS
  MLIRTensixflowAnalysis
  MLIRTensixflowSim
  MLIRTensixflowScheduler
  MLIRTensixflow
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRFuncDialect
  MLIRArithDialect
  MLIRTransforms
  MLIRSupport
  LLVMSupport
)

add_executable(bench_rmsnorm_matmul bench_rmsnorm_matmul.cpp)
target_link_libraries(bench_rmsnorm_matmul PRIVATE ${TENSIXFLOW_BENCH_LIBS})

add_executable(bench_mlp_block bench_mlp_block.cpp)
target_link_libraries(bench_mlp_block PRIVATE ${TENSIXFLOW_BENCH_LIBS})

# Report-regeneration target (todo 44): (re)generates
# reports/rmsnorm_matmul_report.html from reports/rmsnorm_matmul_plan.json via the
# todo-43 tensixflow-report tool (self-contained offline HTML; the tool fills the
# perf section from the static perf model). The output is a staged deliverable
# artifact in the source tree. A missing/invalid plan.json makes tensixflow-report
# exit 1 with a clear diagnostic BEFORE writing -- never a corrupt HTML.
add_custom_target(regenerate-rmsnorm-report
  COMMAND tensixflow-report
          --in ${PROJECT_SOURCE_DIR}/reports/rmsnorm_matmul_plan.json
          --out ${PROJECT_SOURCE_DIR}/reports/rmsnorm_matmul_report.html
  DEPENDS tensixflow-report
  COMMENT "Regenerating reports/rmsnorm_matmul_report.html via tensixflow-report"
  VERBATIM
)
