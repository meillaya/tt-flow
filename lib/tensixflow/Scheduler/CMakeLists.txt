# Scheduler data-model library (todo 25): pure data + helpers, NO MLIR
# dependency, so the planners (todos 26-29) and CB allocator can link it without
# dragging in the dialect. Reuses the plan/perf/viz structs from
# MLIRTensixflowSchema (D15) — links it, does NOT redefine them.
#
# Own subdirectory because LLVM enforces exactly one build target per directory
# (llvm_check_source_file_list); the parent lib/tensixflow/ owns the MLIRTensixflow
# dialect target. A plain add_library (not add_mlir_library) keeps this free of
# MLIR build machinery. The global include_directories() in the top CMakeLists
# already expose include/ + LLVM; the scheduler headers live next to this target.
add_library(MLIRTensixflowScheduler STATIC
  Model.cpp
  GridPlanner.cpp
)
# Expose this directory so consumers can #include "Model.h" / "L1Model.h" etc.
target_include_directories(MLIRTensixflowScheduler PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(MLIRTensixflowScheduler PUBLIC
  MLIRTensixflowSchema
  LLVMSupport
)
