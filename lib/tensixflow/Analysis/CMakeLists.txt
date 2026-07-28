# Static performance-model library (todo 42). Own subdirectory because LLVM
# enforces one build target per directory (the same rule as Schema/ + Scheduler/
# + Sim/). A PLAIN add_library (NOT add_mlir_library) keeps it free of MLIR build
# machinery / tablegen — the perf model is pure data + math over the scheduler
# model, with NO MLIR dependency.
#
# Links MLIRTensixflowScheduler PUBLIC: that target carries the NoC estimator
# (todo 29), weight planner (todo 27), grid planner (todo 26), the shared
# L1-budget constants (L1Model.h, todo 26 / D17 — imported, NOT redefined), and
# reuses the plan/perf/viz structs from MLIRTensixflowSchema (D15) transitively.
# The PUBLIC include dir lets consumers `#include "PerfModel.h"`.
add_library(MLIRTensixflowAnalysis STATIC PerfModel.cpp)
target_include_directories(MLIRTensixflowAnalysis PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(MLIRTensixflowAnalysis PUBLIC
  MLIRTensixflowScheduler
  MLIRTensixflowSchema
  LLVMSupport
)
