# tensixflow-translate driver (MLIR <-> JSON skeleton).
#
# Much lighter than tensixflow-opt: it does NOT register all passes/dialects, so
# it only needs the translate driver library, the parser (mlirTranslateMain
# reads the input module textually), core IR/support, and the dialects the
# translation registers (tensixflow + func + arith). MLIRTranslateLib is an
# imported STATIC target exported by the installed MLIRTargets.cmake; its
# INTERFACE_LINK_LIBRARIES transitively pull in the `LLVM` shared lib, so LLVM
# symbols resolve without listing them here.

# Land the binary in <build>/bin/ (as mlir/examples/standalone does): installed
# LLVMConfig.cmake does not preset LLVM_RUNTIME_OUTPUT_INTDIR for external
# projects, so add_llvm_executable would otherwise put it under build/tools/.
set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)

add_llvm_executable(tensixflow-translate tensixflow-translate.cpp)
llvm_update_compile_flags(tensixflow-translate)
target_link_libraries(tensixflow-translate PRIVATE
  MLIRTranslateLib
  MLIRTensixflow
  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRFuncDialect
  MLIRArithDialect
  )
