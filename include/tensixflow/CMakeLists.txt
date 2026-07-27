add_mlir_dialect(TensixflowOps tensixflow)

# Tile-level schedule IR (todo 31): the tensixflow.sched.* op set that is the
# codegen<->simulator contract (draft D14). Own tablegen target, folded into the
# dialect TU via addOperations<> (see lib/tensixflow/TensixflowDialect.cpp).
set(LLVM_TARGET_DEFINITIONS TensixflowSchedule.td)
mlir_tablegen(TensixflowSchedule.h.inc -gen-op-decls)
mlir_tablegen(TensixflowSchedule.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRTensixflowScheduleIncGen)

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
