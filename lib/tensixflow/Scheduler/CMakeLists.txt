# Scheduler data-model library: pure data and helpers with no MLIR dependency,
# so the planners and the circular-buffer allocator can link it without the
# dialect. Reuses the plan schema structs from MLIRTensixflowSchema (links it,
# does not redefine them).
#
# Own subdirectory because LLVM enforces one build target per directory; the
# parent lib/tensixflow owns the dialect target. A plain add_library keeps the
# target free of MLIR build machinery. The global include_directories() in the
# top CMakeLists already expose include/ and LLVM, and the scheduler headers
# live next to this target.
add_library(MLIRTensixflowScheduler STATIC
  DeviceSelection.cpp
  Model.cpp
  GridPlanner.cpp
  WeightPlanner.cpp
  CBAllocator.cpp
  NoCEstimator.cpp
  PlanReport.cpp
)

# Expose this directory so consumers can include "Model.h" and the other
# scheduler headers. The generated device table is header-only and lives under
# include/tensixflow/Generated.
target_include_directories(MLIRTensixflowScheduler PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(MLIRTensixflowScheduler PUBLIC
  MLIRTensixflowSchema
  LLVMSupport
)
