add_mlir_dialect(TensixflowOps tensixflow)

set(LLVM_TARGET_DEFINITIONS TensixflowAttrs.td)
mlir_tablegen(TensixflowEnums.h.inc -gen-enum-decls)
mlir_tablegen(TensixflowEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(TensixflowAttrs.h.inc -gen-attrdef-decls)
mlir_tablegen(TensixflowAttrs.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRTensixflowAttrsIncGen)
