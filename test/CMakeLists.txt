# lit/FileCheck regression suite for TensixFlow.
#
# Modeled on the upstream out-of-tree example llvm-project
# mlir/examples/standalone/test (release/20.x @ llvmorg-20.1.8), adapted
# standalone -> tensixflow. The site-config template is substituted by
# configure_lit_site_cfg (AddLLVM.cmake) via configure_file(@ONLY).

# add_lit_testsuite -> add_lit_target spawns lit as "${Python3_EXECUTABLE} <lit>"
# (AddLLVM.cmake). The installed LLVMConfig.cmake does NOT find Python3 for
# external projects, so find it here or LIT_COMMAND would have an empty
# interpreter and the check target would fail to run lit.
find_package(Python3 COMPONENTS Interpreter REQUIRED)

# Variables consumed by lit.site.cfg.py.in. LLVM_TOOLS_BINARY_DIR is exported by
# the installed LLVMConfig.cmake and is the LLVM bin/ holding FileCheck/count/not
# (under nix this is the llvm runtime store path, NOT the -dev cmake path).
set(TENSIXFLOW_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(TENSIXFLOW_BINARY_DIR ${PROJECT_BINARY_DIR})
set(TENSIXFLOW_TOOLS_DIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_TOOLS_DIR ${LLVM_TOOLS_BINARY_DIR})

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

# FileCheck/count/not are prebuilt in the installed LLVM (nix store), not CMake
# targets, so DEPENDS lists only the tool we build here.
add_lit_testsuite(check-tensixflow "Running the tensixflow regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS tensixflow-opt
)
set_target_properties(check-tensixflow PROPERTIES FOLDER "Tests")
