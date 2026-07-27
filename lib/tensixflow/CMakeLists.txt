add_mlir_dialect_library(MLIRTensixflow
  TensixflowDialect.cpp
  TensixflowOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/tensixflow

  DEPENDS
  MLIRTensixflowOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRFuncDialect
  )
