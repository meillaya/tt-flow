#!/usr/bin/env bash
# Configure and build the compiler. Run through devenv:
#
#   devenv shell -- bash scripts/build.sh
#
# The optional first argument is the build directory (default: build).
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
build_dir="${1:-${repo_root}/build}"

if [ -z "${MLIR_DIR:-}" ] || [ -z "${LLVM_DIR:-}" ]; then
  echo "build: MLIR_DIR and LLVM_DIR are unset; enter the devenv shell first" >&2
  exit 2
fi

cmake -G Ninja -B "${build_dir}" -S "${repo_root}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${build_dir}"
