# Plan/perf/viz schema library (D15, todo 24): plain data + llvm::json, NO MLIR
# dependency, so the scheduler (Wave 5), plan/perf report (todo 30), and HTML
# visualizer (todo 43) can link it without dragging in the dialect.
#
# Lives in its own directory because LLVM's build system enforces exactly one
# build target per directory (llvm_check_source_file_list); the parent
# lib/tensixflow/ owns the MLIRTensixflow dialect target. A plain add_library
# (not add_mlir_library) keeps this free of MLIR build machinery. The global
# include_directories() in the top CMakeLists already expose include/ + LLVM.
add_library(MLIRTensixflowSchema STATIC
  PlanSchema.cpp
  PlanSchemaParse.cpp
)
target_link_libraries(MLIRTensixflowSchema PUBLIC LLVMSupport)
