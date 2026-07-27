# tensixflow-opt skeleton driver.
#
# Link set mirrors upstream mlir/tools/mlir-opt: `registerAllPasses()` is an
# inline function (mlir/InitAllPasses.h, LLVM 20) that references the pass
# registration symbols of every dialect, so the executable must link the full
# dialect/extension library set. The installed MLIRConfig.cmake exports both
# global properties (MLIR_DIALECT_LIBS / MLIR_EXTENSION_LIBS), so this works
# against the nix-provided prebuilt MLIR. On top of that, LLVM 20 has no
# monolithic MLIRConversion target: registerConversionPasses() references the
# createConvert*Pass creators spread over per-conversion libraries that are
# NOT in MLIR_DIALECT_LIBS and not propagated transitively, so they are listed
# explicitly below (discovered via `nm` on the installed store libs).
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

# Land the binary in <build>/bin/ (as mlir/examples/standalone does): installed
# LLVMConfig.cmake does not preset LLVM_RUNTIME_OUTPUT_INTDIR for external
# projects, so add_llvm_executable would otherwise put it under build/tools/.
set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)

add_llvm_executable(tensixflow-opt tensixflow-opt.cpp)
llvm_update_compile_flags(tensixflow-opt)
target_link_libraries(tensixflow-opt PRIVATE
  ${dialect_libs}
  ${extension_libs}
  MLIRTensixflow
  MLIRIR
  MLIRPass
  MLIRParser
  MLIRSupport
  MLIROptLib
  MLIRFuncDialect
  MLIRArithDialect
  MLIRTransforms
  MLIRBytecodeWriter
  # Per-conversion libraries required by registerConversionPasses() creators:
  MLIRAMDGPUToROCDL
  MLIRArithToAMDGPU
  MLIRArithToArmSME
  MLIRArithToEmitC
  MLIRArithToSPIRV
  MLIRArmNeon2dToIntr
  MLIRArmSMEToLLVM
  MLIRArmSMEToSCF
  MLIRBufferizationToMemRef
  MLIRComplexToSPIRV
  MLIRControlFlowToSCF
  MLIRControlFlowToSPIRV
  MLIRConvertToSPIRVPass
  MLIRFuncToEmitC
  MLIRFuncToSPIRV
  MLIRGPUToLLVMSPV
  MLIRGPUToROCDLTransforms
  MLIRGPUToSPIRV
  MLIRIndexToSPIRV
  MLIRLinalgToStandard
  MLIRMathToEmitC
  MLIRMathToFuncs
  MLIRMathToROCDL
  MLIRMathToSPIRV
  MLIRMemRefToEmitC
  MLIRMemRefToSPIRV
  MLIRMeshToMPI
  MLIROpenACCToSCF
  MLIROpenMPToLLVM
  MLIRSCFToEmitC
  MLIRSCFToGPU
  MLIRSCFToOpenMP
  MLIRSCFToSPIRV
  MLIRShapeToStandard
  MLIRSPIRVToLLVM
  MLIRTensorToLinalg
  MLIRTensorToSPIRV
  MLIRTosaToArith
  MLIRTosaToLinalg
  MLIRTosaToMLProgram
  MLIRTosaToSCF
  MLIRTosaToTensor
  MLIRUBToLLVM
  MLIRUBToSPIRV
  MLIRVectorToArmSME
  MLIRVectorToGPU
  MLIRVectorToSPIRV
  MLIRVectorToXeGPU
  )
