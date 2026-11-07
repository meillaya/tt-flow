# tensixflow-simulate: the Python and C++ simulator bridge. Wraps the
# functional tile simulator (lib/tensixflow/Sim) end-to-end so the Python
# correctness harness can drive the REAL C++ simulator over a schedule IR built by
# the full pipeline. Links the Sim library (which PUBLIC-links MLIRTensixflow, so
# the schedule-op classes resolve) plus the MLIR libs needed to parse the example
# func and run the infer-layout -> fuse -> plan -> allocate -> lower pipeline.
# MLIRTransforms is required by the fuse pass's greedy rewrite driver.

# Land the binary in <build>/bin/ (as mlir/examples/standalone does): installed
# LLVMConfig.cmake does not preset LLVM_RUNTIME_OUTPUT_INTDIR for external
# projects, so add_llvm_executable would otherwise put it under build/tools/.
set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)

add_llvm_executable(tensixflow-simulate tensixflow-simulate.cpp)
llvm_update_compile_flags(tensixflow-simulate)
target_link_libraries(tensixflow-simulate PRIVATE
  MLIRTensixflowSim
  MLIRTensixflow
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRFuncDialect
  MLIRArithDialect
  MLIRTransforms
  MLIRSupport
  )
