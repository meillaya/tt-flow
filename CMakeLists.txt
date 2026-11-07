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

# nixpkgs MLIRConfig ships MLIR_TABLEGEN_EXE as a bare executable name, which
# shadows -DMLIR_TABLEGEN_EXE and leaves LLVM's TableGen.cmake with an imported
# target whose location is not absolute. Clearing the variable first forces a
# real PATH search so the imported target points at the pinned mlir-tblgen.
unset(MLIR_TABLEGEN_EXE)
find_program(MLIR_TABLEGEN_EXE mlir-tblgen REQUIRED)
add_executable(mlir-tblgen IMPORTED GLOBAL)
set_target_properties(mlir-tblgen PROPERTIES IMPORTED_LOCATION "${MLIR_TABLEGEN_EXE}")

# nixpkgs LLVMConfig.cmake points LLVM_EXTERNAL_LIT at a binary that the package
# does not install. The devenv shell exports the working Python entry point, so
# prefer the environment value over the packaged default.
if(DEFINED ENV{LLVM_EXTERNAL_LIT})
  set(LLVM_EXTERNAL_LIT "$ENV{LLVM_EXTERNAL_LIT}" CACHE FILEPATH "Path to the lit entry point" FORCE)
endif()

# Mirror mlir/examples/standalone. Tablegen reads the directory
# INCLUDE_DIRECTORIES property to build its include flags, so the LLVM and MLIR
# include dirs are required in addition to the project's own headers.
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

# Benchmarks and the report-regeneration target. These come after tools/ so the
# targets they depend on exist.
add_subdirectory(benchmarks)

# lit/FileCheck suite. Defines check-tensixflow; after tools/ for the same reason.
add_subdirectory(test)

# GoogleTest suites registered with ctest.
enable_testing()
add_subdirectory(test/unit)

# Emitted-project checks against the pinned tt-metal release. The target prints
# the fetch command and skips explicitly when .deps is absent; it never passes
# without compiling.
add_subdirectory(kernels)
