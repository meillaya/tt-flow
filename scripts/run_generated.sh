#!/usr/bin/env bash
# Run an emitted TensixFlow project and report the result as JSON.
#
# Run through devenv:
#
#   devenv shell -- bash scripts/run_generated.sh                  # ttsim
#   devenv shell -- bash scripts/run_generated.sh <project-dir>    # ttsim
#   devenv shell -- bash scripts/run_generated.sh --device         # card
#
# The default backend is the official ttsim simulator when .deps/ttsim holds
# libttsim_wh.so: the script sets TT_METAL_SIMULATOR and
# TT_METAL_SLOW_DISPATCH_MODE=1, so no Tenstorrent card is needed. With
# --device the script requires scripts/device_probe.py to confirm a card and
# runs without the simulator variables. There is no silent fallback between the
# two backends.
#
# The emitted program prints a TENSIXFLOW_RESULT line carrying the PCC. The
# script parses that line and prints:
#
#   {"status":"pass|fail","backend":"ttsim|device","pcc":<number or null>}
#
# The script exits 0 only when the status is pass, and 2 when no backend is
# available.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
deps_env="$repo_root/.deps/env.sh"
default_project="$repo_root/build-generated/project"

fail() {
    echo "error: $*" >&2
    exit 2
}

project_dir="$default_project"
backend="ttsim"
while [ "$#" -gt 0 ]; do
    case "$1" in
        --device)
            backend="device"
            shift
            ;;
        -h|--help)
            sed -n '2,20p' "${BASH_SOURCE[0]}"
            exit 0
            ;;
        -*)
            fail "unknown option '$1'"
            ;;
        *)
            project_dir="$1"
            shift
            ;;
    esac
done

# Resolve the project directory now: the program is run from inside it so that
# the host program's relative kernel source paths resolve, and a relative
# project argument would otherwise break the binary path.
if [ -d "$project_dir" ]; then
    project_dir="$(cd "$project_dir" && pwd)"
fi

binary="$project_dir/build/tensixflow_generated"
if [ ! -x "$binary" ]; then
    echo "error: $binary is missing" >&2
    echo "run: devenv shell -- bash scripts/build_generated.sh $project_dir" >&2
    exit 2
fi

if [ ! -f "$deps_env" ]; then
    fail "the pinned TT-Metalium artifacts are not fetched; run: devenv shell -- bash scripts/fetch_tt_metal.sh"
fi
# shellcheck source=/dev/null
source "$deps_env"

# The device probe is the only authority on whether a card is visible. A missing
# probe script is treated as no card rather than assumed to work.
probe_device() {
    if [ ! -f "$repo_root/scripts/device_probe.py" ]; then
        return 1
    fi
    if python3 "$repo_root/scripts/device_probe.py" >/dev/null 2>&1; then
        return 0
    fi
    return 1
}

run_dir="$project_dir"
run_log="$(mktemp)"
trap 'rm -f "$run_log"' EXIT

case "$backend" in
    ttsim)
        if [ ! -f "${TT_METAL_SIMULATOR:-}" ]; then
            # No simulator library: try a card, but do not pretend ttsim ran.
            if ! probe_device; then
                fail "no simulator at ${TT_METAL_SIMULATOR:-.deps/ttsim/libttsim_wh.so} and no card visible; run: devenv shell -- bash scripts/fetch_tt_metal.sh"
            fi
            backend="device"
        fi
        ;;
    device)
        if ! probe_device; then
            fail "no Tenstorrent card visible; scripts/device_probe.py exited nonzero. Use the default ttsim backend for CPU execution."
        fi
        ;;
esac

echo "== run ==" >&2
echo "  backend: $backend" >&2
echo "  project: $project_dir" >&2
echo "  binary:  $binary" >&2

set +e
if [ "$backend" = "ttsim" ]; then
    (
        cd "$run_dir" && \
        TT_METAL_SIMULATOR="$TT_METAL_SIMULATOR" \
        TT_METAL_SLOW_DISPATCH_MODE=1 \
        "$binary"
    ) >"$run_log" 2>&1
else
    (
        cd "$run_dir" && \
        env -u TT_METAL_SIMULATOR -u TT_METAL_SLOW_DISPATCH_MODE "$binary"
    ) >"$run_log" 2>&1
fi
program_status=$?
set -e

cat "$run_log"

# The emitted program prints exactly one machine-readable result line. The PCC
# is only accepted when the program itself reported a pass status.
reported_status="$(sed -n 's/.*TENSIXFLOW_RESULT {"status":"\([a-z]*\)".*/\1/p' "$run_log" | tail -n 1)"
pcc_value="$(sed -n 's/.*TENSIXFLOW_RESULT {"status":"[a-z]*","pcc":\([-0-9.eE+]*\),.*/\1/p' "$run_log" | tail -n 1)"

if [ -z "$pcc_value" ]; then
    pcc_json="null"
else
    pcc_json="$pcc_value"
fi

if [ "$program_status" -eq 0 ] && [ "$reported_status" = "pass" ]; then
    status="pass"
    exit_code=0
else
    status="fail"
    exit_code=1
fi

printf '{"status":"%s","backend":"%s","pcc":%s}\n' \
    "$status" "$backend" "$pcc_json"
exit "$exit_code"
