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
  Compat/EmitTTIR.cpp
  Compat/TTIRTextEmitter.cpp
  Compat/TTIRFuncTranslator.cpp
  Compat/TTIRLayoutAttrs.cpp
  Codegen/EmitHost.cpp
  Codegen/EmitCompute.cpp
  Codegen/EmitReader.cpp
  Codegen/EmitWriter.cpp
  Codegen/EmitProject.cpp

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

# Static performance-model library (todo 42). Own subdirectory for the same
# reason; see Analysis/CMakeLists.txt. Links the scheduler lib PUBLIC (NoC
# estimator + L1-budget constants + plan structs), so it is added after Scheduler.
add_subdirectory(Analysis)

# Functional tile simulator library (todo 39, D14). Own subdirectory for the same
# reason; see Sim/CMakeLists.txt. Links MLIRTensixflow (it walks the schedule IR).
add_subdirectory(Sim)
