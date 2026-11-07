# Functional tile simulator library. Own subdirectory because
# LLVM enforces one build target per directory (the same rule as Schema/ +
# Scheduler/). Rounding.h is header-only and lives alongside; this target
# lists the single .cpp so the one-target/dir rule holds. A PLAIN add_library (NOT
# add_mlir_library) keeps it free of MLIR build machinery / tablegen.
#
# The PUBLIC include dir lets consumers `#include "TileSimulator.h"` AND
# `#include "Rounding.h"` (both in this dir). MLIRTensixflow is linked PUBLIC so the
# schedule-op classes (TensixflowSched*Op) the simulator walks resolve for consumers
# (the gtest links MLIRTensixflowSim and gets the dialect transitively).
add_library(MLIRTensixflowSim STATIC TileSimulator.cpp)
target_include_directories(MLIRTensixflowSim PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(MLIRTensixflowSim PUBLIC
  MLIRTensixflow
  MLIRIR
  LLVMSupport
)
