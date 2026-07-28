add_mlir_dialect_library(MLIRTensixflow
  TensixflowDialect.cpp
  TensixflowOps.cpp
  TensixflowScheduleOps.cpp
  TensixflowPasses.cpp
  InferTileLayoutPass.cpp
  FuseRmsnormMatmulPass.cpp
  PlanNocShardingPass.cpp
  AllocateCircularBuffersPass.cpp
  LowerToTTKernelPass.cpp
  EmitMetaliumPass.cpp
  Codegen/EmitHost.cpp
  Codegen/EmitReader.cpp
  Codegen/EmitWriter.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/tensixflow

  DEPENDS
  MLIRTensixflowOpsIncGen
  MLIRTensixflowScheduleIncGen
  MLIRTensixflowAttrsIncGen
  MLIRTensixflowPassesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRFuncDialect
  MLIRFunctionInterfaces
  MLIRPass
  MLIRTransforms
  MLIRTensixflowScheduler
  )

# Plan/perf/viz schema library (D15). Own subdirectory because LLVM enforces one
# build target per directory; see Schema/CMakeLists.txt.
add_subdirectory(Schema)

# Scheduler data-model library (todo 25). Own subdirectory for the same reason;
# see Scheduler/CMakeLists.txt.
add_subdirectory(Scheduler)
