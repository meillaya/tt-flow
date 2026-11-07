# Static performance-model library. Own subdirectory because LLVM enforces one
# build target per directory (the same rule as Schema/, Scheduler/, and Sim/).
# A plain add_library rather than add_mlir_library keeps the target free of MLIR
# build machinery: the perf model is pure data and math over the scheduler
# model, with no MLIR dependency.
#
# Links MLIRTensixflowScheduler PUBLIC: that target carries the NoC estimator,
# weight planner, grid planner, CB allocator, and the generated device table,
# and reuses the plan/perf/viz structs from MLIRTensixflowSchema transitively.
# The PUBLIC include directory lets consumers include "PerfModel.h".
add_library(MLIRTensixflowAnalysis STATIC PerfModel.cpp)
target_include_directories(MLIRTensixflowAnalysis PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(MLIRTensixflowAnalysis PUBLIC
  MLIRTensixflowScheduler
  MLIRTensixflowSchema
  LLVMSupport
)
