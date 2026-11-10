#!/usr/bin/env bash
# Emit a TensixFlow project, then configure, compile, and link it against the
# pinned TT-Metalium release. This is the real compile gate for the Metalium
# emitter: there is no stub header anywhere in the path.
#
# Run through devenv:
#
#   devenv shell -- bash scripts/build_generated.sh
#   devenv shell -- bash scripts/build_generated.sh <input.mlir>
#   devenv shell -- bash scripts/build_generated.sh <existing-project-dir>
#
# With no argument the script emits a fresh project into
# build-generated/project/ from examples/matmul.mlir. A .mlir argument emits a
# fresh project into build-generated/<input-stem>/ and builds that; a directory
# argument builds an already-emitted project.
#
# The pinned artifacts must be present in .deps/ (scripts/fetch_tt_metal.sh).
# When they are absent the script prints the fetch command and exits 0 with an
# explicit skip message; the skip is never reported as a pass.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
deps_env="$repo_root/.deps/env.sh"
default_project="$repo_root/build-generated/project"

fail() {
    echo "error: $*" >&2
    exit 1
}

# The pinned TT-Metalium package and simulator environment. Everything the
# build and run need is exported by the fetch script's env file.
if [ ! -f "$deps_env" ]; then
    echo "skipping: the pinned TT-Metalium artifacts are not fetched."
    echo "Run: devenv shell -- bash scripts/fetch_tt_metal.sh"
    exit 0
fi
# shellcheck source=/dev/null
source "$deps_env"

if [ -z "${TT_METAL_SYSTEM_LINK_FLAGS:-}" ]; then
    fail "TT_METAL_SYSTEM_LINK_FLAGS is unset; run inside 'devenv shell' so devenv.nix provides the TT-Metalium link dependencies"
fi

# Argument forms: no argument emits from examples/matmul.mlir into
# build-generated/project; a .mlir argument emits that file into
# build-generated/<stem>; anything else is treated as an emitted project
# directory.
project_dir="$default_project"
emit_fresh=1
input_mlir=""
if [ "$#" -ge 1 ]; then
    if [ -f "$1" ] && [[ "$1" == *.mlir ]]; then
        input_mlir="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
        input_stem="$(basename "$input_mlir" .mlir)"
        project_dir="$repo_root/build-generated/$input_stem"
    else
        project_dir="$1"
        emit_fresh=0
    fi
fi

# The host program is built by the compiler's own emitter, so the emitter has to
# exist. TENSIXFLOW_BUILD_DIR is set by the kernels/CMakeLists.txt target; the
# fallback covers a direct shell invocation after scripts/build.sh.
find_opt() {
    if [ -n "${TENSIXFLOW_OPT:-}" ] && [ -x "$TENSIXFLOW_OPT" ]; then
        echo "$TENSIXFLOW_OPT"
        return 0
    fi
    local candidate
    for candidate in \
        "${TENSIXFLOW_BUILD_DIR:-/nonexistent}/bin/tensixflow-opt" \
        "$repo_root/build/bin/tensixflow-opt" \
        "$repo_root/build-codegen/bin/tensixflow-opt"; do
        if [ -x "$candidate" ]; then
            echo "$candidate"
            return 0
        fi
    done
    if command -v tensixflow-opt >/dev/null 2>&1; then
        command -v tensixflow-opt
        return 0
    fi
    return 1
}

# The source graph for the default fresh emission. examples/matmul.mlir is the
# documented pure matmul path; the in-tree fixture keeps the target usable while
# the example is being added.
find_input() {
    if [ -n "${TENSIXFLOW_MATMUL_MLIR:-}" ]; then
        if [ ! -f "$TENSIXFLOW_MATMUL_MLIR" ]; then
            fail "TENSIXFLOW_MATMUL_MLIR points at $TENSIXFLOW_MATMUL_MLIR, which does not exist"
        fi
        echo "$TENSIXFLOW_MATMUL_MLIR"
        return 0
    fi
    local candidate
    for candidate in \
        "$repo_root/examples/matmul.mlir" \
        "$repo_root/kernels/fixtures/matmul.mlir"; do
        if [ -f "$candidate" ]; then
            echo "$candidate"
            return 0
        fi
    done
    fail "no matmul input found; add examples/matmul.mlir, keep kernels/fixtures/matmul.mlir, or set TENSIXFLOW_MATMUL_MLIR"
}

if [ "$emit_fresh" -eq 1 ]; then
    opt="$(find_opt)" || fail "tensixflow-opt not found; build the compiler first with: devenv shell -- bash scripts/build.sh"
    if [ -z "$input_mlir" ]; then
        input_mlir="$(find_input)"
    fi
    echo "== emit =="
    echo "  $opt --tensixflow-device=n150 --tensixflow-fuse-rmsnorm-matmul \\"
    echo "      --tensixflow-plan-noc-sharding \\"
    echo "      --tensixflow-allocate-circular-buffers --tensixflow-lower-to-ttkernel \\"
    echo "      --tensixflow-emit-metalium=output-dir=$project_dir $input_mlir"
    rm -rf "$project_dir"
    mkdir -p "$project_dir"
    "$opt" \
        --tensixflow-device=n150 \
        --tensixflow-fuse-rmsnorm-matmul \
        --tensixflow-plan-noc-sharding \
        --tensixflow-allocate-circular-buffers \
        --tensixflow-lower-to-ttkernel \
        --tensixflow-emit-metalium=output-dir="$project_dir" \
        "$input_mlir"
else
    if [ ! -f "$project_dir/CMakeLists.txt" ]; then
        fail "$project_dir does not hold an emitted project (CMakeLists.txt missing)"
    fi
fi

build_dir="$project_dir/build"
echo "== configure =="
cmake -G Ninja -S "$project_dir" -B "$build_dir" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_EXE_LINKER_FLAGS="$TT_METAL_SYSTEM_LINK_FLAGS"

echo "== build =="
cmake --build "$build_dir"

binary="$build_dir/tensixflow_generated"
if [ ! -x "$binary" ]; then
    fail "the build finished but $binary is not executable"
fi
echo "built $binary"
echo "next: devenv shell -- bash scripts/run_generated.sh $project_dir"
