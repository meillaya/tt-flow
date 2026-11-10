#!/usr/bin/env bash
# Run tools/normalize_ttir.py with the pinned tt-mlir wheel environment.
#
# The pinned wheel is CPython 3.11 and its manylinux shared objects expect the
# nix store libraries, so the interpreter, PYTHONPATH, and LD_LIBRARY_PATH have
# to agree. This wrapper does that assembly; the caller only names the input:
#
#   devenv shell -- bash scripts/normalize_ttir.sh layout.mlir -o bare.mlir
#   tensixflow-opt --tensixflow-emit-ttir examples/rmsnorm_matmul.mlir \
#     | devenv shell -- bash scripts/normalize_ttir.sh
#
# The wheel lives in .deps/ttmlir/site after scripts/fetch_ttmlir_fixtures.sh.
# --site points at another extraction, --python at another CPython 3.11.
#
# The library resolution here mirrors python/tensixflow/tests/test_ttir_parse.py:
# the wheel targets a system glibc layout while the devenv shell exposes nix
# store paths, so a linked name on the "not found" list is looked up in the
# closures of the interpreter and the C++ toolchain.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"

site_dir="$repo_root/.deps/ttmlir/site"
python_bin="python3.11"
tool_path="$repo_root/tools/normalize_ttir.py"

wheel_package="ttmlir/ir.py"
library_glob="ttmlir/_mlir_libs/*.so"
library_missing_marker="not found"
library_dir_names=("lib" "lib64")
library_toolchains=("g++" "cc")
fetch_hint="run 'devenv shell -- bash scripts/fetch_ttmlir_fixtures.sh'"

fail() {
    echo "error: $*" >&2
    exit 1
}

usage() {
    cat <<'USAGE'
usage: normalize_ttir.sh [--site DIR] [--python BIN] [tool options] [input]

Normalize layout-encoded TTIR into the bare-tensor TTIR the importer accepts.

  --site DIR    extracted tt-mlir wheel directory (default .deps/ttmlir/site)
  --python BIN  CPython 3.11 interpreter (default python3.11)

Tool options are -o/--output PATH and an input path; '-' reads stdin.
USAGE
}

# Print the library directories the wheel needs beyond the current search path.
# Nothing is printed when ldd resolves every name already.
resolve_library_path() {
    local linked name root dir found joined=""
    local -a missing=() roots=() closure_roots=() dirs=()

    if ! command -v ldd >/dev/null 2>&1; then
        return 0
    fi
    while IFS= read -r name; do
        if [[ -n "$name" ]]; then
            missing+=("$name")
        fi
    done < <(
        for linked in "$site_dir"/ttmlir/_mlir_libs/*.so; do
            ldd "$linked" 2>/dev/null || true
        done | awk -v marker="$library_missing_marker" \
            'index($0, marker) {print $1}' | sort -u
    )
    if [[ ${#missing[@]} -eq 0 ]]; then
        return 0
    fi
    if ! command -v nix-store >/dev/null 2>&1; then
        fail "cannot resolve wheel shared libraries without nix-store; $fetch_hint and run this wrapper inside the devenv shell"
    fi

    # The interpreter and the C++ toolchain provide the system libraries the
    # manylinux wheel links against; search both closures.
    roots+=("$(command -v "$python_bin")")
    for linked in "${library_toolchains[@]}"; do
        if command -v "$linked" >/dev/null 2>&1; then
            roots+=("$(command -v "$linked")")
        fi
    done
    for linked in "${roots[@]}"; do
        while IFS= read -r root; do
            if [[ -n "$root" ]]; then
                closure_roots+=("$root")
            fi
        done < <(nix-store -qR "$linked" 2>/dev/null || true)
    done

    for name in "${missing[@]}"; do
        found=""
        for root in "${closure_roots[@]}"; do
            for dir in "${library_dir_names[@]}"; do
                if [[ -e "$root/$dir/$name" ]]; then
                    found="$root/$dir"
                    break 2
                fi
            done
        done
        if [[ -z "$found" ]]; then
            fail "cannot resolve shared library '$name' for the ttmlir wheel; devenv.nix must provide it"
        fi
        # Keep the search path short when several names share a directory.
        if [[ ":${dirs[*]:-}:" != *":$found:"* ]]; then
            dirs+=("$found")
        fi
    done

    for dir in "${dirs[@]}"; do
        joined="${joined:+$joined:}$dir"
    done
    printf '%s' "$joined"
}

tool_args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --site)
            [[ $# -ge 2 ]] || fail "--site needs a directory"
            site_dir="$2"
            shift 2
            ;;
        --site=*)
            site_dir="${1#*=}"
            shift
            ;;
        --python)
            [[ $# -ge 2 ]] || fail "--python needs an interpreter"
            python_bin="$2"
            shift 2
            ;;
        --python=*)
            python_bin="${1#*=}"
            shift
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        --)
            shift
            tool_args+=("$@")
            break
            ;;
        *)
            tool_args+=("$1")
            shift
            ;;
    esac
done

if [[ ! -f "$site_dir/$wheel_package" ]]; then
    fail "the pinned tt-mlir wheel is not extracted at $site_dir; $fetch_hint"
fi
if ! command -v "$python_bin" >/dev/null 2>&1; then
    fail "interpreter '$python_bin' is not on PATH; run inside the devenv shell or pass --python"
fi
if [[ ! -f "$tool_path" ]]; then
    fail "normalizer tool is missing at $tool_path"
fi

export PYTHONPATH="$site_dir${PYTHONPATH:+:$PYTHONPATH}"
library_path="$(resolve_library_path)" || exit 1
if [[ -n "$library_path" ]]; then
    export LD_LIBRARY_PATH="$library_path${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

exec "$python_bin" "$tool_path" "${tool_args[@]}"
