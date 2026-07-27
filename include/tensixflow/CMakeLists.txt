add_mlir_dialect(TensixflowOps tensixflow)

set(LLVM_TARGET_DEFINITIONS TensixflowAttrs.td)
mlir_tablegen(TensixflowEnums.h.inc -gen-enum-decls)
mlir_tablegen(TensixflowEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(TensixflowAttrs.h.inc -gen-attrdef-decls)
mlir_tablegen(TensixflowAttrs.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRTensixflowAttrsIncGen)

# No -name: the group registration function is registerPasses() (in
# mlir::tensixflow via each pass's cppNamespace), matching the standalone
# example's GEN_PASS_REGISTRATION pattern.
set(LLVM_TARGET_DEFINITIONS TensixflowPasses.td)
mlir_tablegen(TensixflowPasses.h.inc -gen-pass-decls)
add_public_tablegen_target(MLIRTensixflowPassesIncGen)
