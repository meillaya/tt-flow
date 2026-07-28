# Compile-only structural check for TT-Metalium kernels (todo 33 / D16).
#
# Generated/baseline kernels are NOT run on hardware (there is no device and no
# tt-metal library). They are validated STRUCTURALLY by compiling them
# `-fsyntax-only` against the signature-only stub headers in
# kernels/handwritten_baselines/tt_metal_stubs/ (pinned to tt-metal @df36cd6).
#
# The target `check-generated-kernels-compile` asserts ALL THREE halves of the
# contract (see check_kernels_compile.sh):
#   1. the current-API baseline compiles, AND
#   2. a kernel referencing the removed legacy symbol EnqueueProgram FAILS
#      (1+2 make the check non-vacuous), AND
#   3. every generated/<op>/*.cpp project written by
#      --tensixflow-emit-metalium=output-dir= compiles (todo 37).
# Any half failing fails the build.

set(TT_STUBS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/handwritten_baselines/tt_metal_stubs)
set(TT_BASELINE ${CMAKE_CURRENT_SOURCE_DIR}/handwritten_baselines/matmul_single_core_baseline.cpp)
set(TT_NEGATIVE ${CMAKE_CURRENT_SOURCE_DIR}/handwritten_baselines/legacy_negative.cpp)
set(TT_CHECK_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/handwritten_baselines/check_kernels_compile.sh)
set(TT_GENERATED_DIR ${CMAKE_CURRENT_SOURCE_DIR}/generated)

add_custom_target(check-generated-kernels-compile
  COMMAND bash ${TT_CHECK_SCRIPT}
          ${CMAKE_CXX_COMPILER}
          ${TT_STUBS_DIR}
          ${TT_BASELINE}
          ${TT_NEGATIVE}
          ${TT_GENERATED_DIR}
  COMMENT "Compile-checking kernels against signature-only tt-metal stubs (@df36cd6)"
  VERBATIM
)
set_target_properties(check-generated-kernels-compile PROPERTIES FOLDER "Tests")
