# -*- Python -*-

import os
import shutil

import lit.formats

from lit.llvm import llvm_config

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = "TENSIXFLOW"

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".mlir"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.tensixflow_obj_root, "test")

config.substitutions.append(("%PATH%", config.environment["PATH"]))
config.substitutions.append(("%shlibext", config.llvm_shlib_ext))
# The C++ compiler for syntax-checking generated kernels against the stubs
# (todo 37); absolute path from CMAKE_CXX_COMPILER, so no PATH dependence.
config.substitutions.append(("%cxx", config.tensixflow_cxx))

llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])

llvm_config.use_default_substitutions()

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = ["Inputs", "Examples", "CMakeLists.txt", "README.txt", "LICENSE.txt"]

# Tweak the PATH to include the tools dir.
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)

tool_dirs = [config.tensixflow_tools_dir, config.llvm_tools_dir]
llvm_config.add_tool_substitutions(["tensixflow-opt", "FileCheck"], tool_dirs)

# Gated ttmlir-opt feature detection (todo 53). If ttmlir-opt is on PATH, add
# the "ttmlir-opt" feature so tests with `// REQUIRES: ttmlir-opt` run. If it
# is absent (the usual case; todo 50 took the source-study fallback and did
# NOT build it), the feature is simply not added, and lit reports those tests
# as SKIP (UNSUPPORTED), never FAIL. The test still exits 0 either way.
if shutil.which("ttmlir-opt") is not None:
    config.available_features.add("ttmlir-opt")
