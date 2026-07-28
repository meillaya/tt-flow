cmake_minimum_required(VERSION 3.20)
project(tensixflow LANGUAGES CXX C)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)

# REQUIRED shim (see .omo/notepads/tensixflow/learnings.md): nixpkgs MLIRConfig ships
# MLIR_TABLEGEN_EXE as a bare name and shadows -DMLIR_TABLEGEN_EXE; LLVM TableGen.cmake
# needs it as a CMake TARGET. unset() first: MLIRConfig.cmake:13 sets it as a NORMAL
# variable, which would short-circuit find_program and leave IMPORTED_LOCATION bare;
# unset forces a real PATH search so the imported target gets an absolute location.
unset(MLIR_TABLEGEN_EXE)
find_program(MLIR_TABLEGEN_EXE mlir-tblgen REQUIRED)
add_executable(mlir-tblgen IMPORTED GLOBAL)
set_target_properties(mlir-tblgen PROPERTIES IMPORTED_LOCATION "${MLIR_TABLEGEN_EXE}")

option(TENSIXFLOW_ENABLE_RUNTIME_STUBS "Generate compile-only stub tt-metal headers for codegen checks" ON)

# Mirror mlir/examples/standalone: LLVM/MLIR include dirs (tablegen reads the
# directory INCLUDE_DIRECTORIES property to build its -I flags, so these are
# required for mlir-tblgen to find mlir/IR/OpBase.td), plus the project include
# dir so #include "tensixflow/..." resolves in both C++ and tablegen.
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)

add_subdirectory(include/tensixflow)
add_subdirectory(lib/tensixflow)

add_subdirectory(tools/tensixflow-opt)
add_subdirectory(tools/tensixflow-translate)
add_subdirectory(tools/tensixflow-simulate)
add_subdirectory(tools/tensixflow-report)

# Simple benchmark timing loops + the report-regeneration target (todo 44, D21).
# Must come after tools/ so the tensixflow-report target the regenerate target
# DEPENDS on already exists.
add_subdirectory(benchmarks)

# lit/FileCheck regression suite (defines the check-tensixflow target). Must come
# after tools/ so the tensixflow-opt target it DEPENDS on already exists.
add_subdirectory(test)

# C++ unit-test suite (GoogleTest + ctest). Separate from the lit suite above;
# requires the MLIRTensixflowSchema lib from lib/tensixflow. enable_testing()
# registers the add_test() entries with ctest (e.g. `ctest -R plan_schema`).
enable_testing()
add_subdirectory(test/unit)

# Compile-only structural check of TT-Metalium kernels against signature-only
# stub headers (defines the check-generated-kernels-compile target; todo 33/D16).
add_subdirectory(kernels)
