# Plan/perf/viz schema library: plain data plus llvm::json, with no MLIR
# dependency, so the scheduler, the plan report, and the HTML visualizer can
# link it without dragging in the dialect.
#
# Lives in its own directory because LLVM's build system enforces one build
# target per directory; the parent lib/tensixflow owns the dialect target. A
# plain add_library keeps this free of MLIR build machinery. The global
# include_directories() in the top CMakeLists already expose include/ and LLVM.
add_library(MLIRTensixflowSchema STATIC
  PlanSchema.cpp
  PlanSchemaParse.cpp
)
target_link_libraries(MLIRTensixflowSchema PUBLIC LLVMSupport)
