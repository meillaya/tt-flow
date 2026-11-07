# TensixFlow benchmarks: simple timing loops, not a benchmarking
# framework (no Google Benchmark dependency). Each bench drives the static perf
# model (buildPlanReport -> estimatePerf -> formatPerfReport, unfused vs
# fused) for the HEADLINE figures and runs a plain std::chrono loop over the
# functional tile simulator (runTileSimulator) for the fused path. The
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

# Sample report target: plan examples/matmul.mlir, serialize the plan, and
# render the offline HTML report into the build tree. The translate tool rejects
# a plan without a device block, so this target also exercises the JSON
# contract. No artifact from this target belongs in the source tree.
add_custom_target(render-sample-report
  COMMAND ${CMAKE_COMMAND} -E make_directory "${PROJECT_BINARY_DIR}/reports"
  COMMAND bash -c "$<TARGET_FILE:tensixflow-opt> --tensixflow-device=n150 --tensixflow-infer-tile-layout --tensixflow-plan-noc-sharding --tensixflow-allocate-circular-buffers ${PROJECT_SOURCE_DIR}/examples/matmul.mlir | $<TARGET_FILE:tensixflow-translate> --tensixflow-to-json > ${PROJECT_BINARY_DIR}/reports/matmul_plan.json"
  COMMAND $<TARGET_FILE:tensixflow-report> --in "${PROJECT_BINARY_DIR}/reports/matmul_plan.json" --out "${PROJECT_BINARY_DIR}/reports/matmul_report.html"
  DEPENDS tensixflow-opt tensixflow-translate tensixflow-report
  COMMENT "Rendering the sample plan report into the build tree"
  VERBATIM
)
