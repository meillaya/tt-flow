add_mlir_dialect_library(MLIRTensixflow
  TensixflowDialect.cpp
  TensixflowOps.cpp
  TensixflowPasses.cpp
  InferTileLayoutPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/tensixflow

  DEPENDS
  MLIRTensixflowOpsIncGen
  MLIRTensixflowAttrsIncGen
  MLIRTensixflowPassesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRFuncDialect
  MLIRFunctionInterfaces
  MLIRPass
  )
