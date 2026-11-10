#!/usr/bin/env bash
# Run all five emittable examples on a Tenstorrent card and print the
# acceptance table for docs/device.md.
#
# Run through devenv:
#
#   devenv shell -- bash scripts/hardware_acceptance.sh
#
# The script is the card-side counterpart of scripts/test.sh. It builds the
# compiler when tensixflow-opt is missing, emits and builds each example
# against the pinned TT-Metalium release, runs each with --device, and writes
# one Markdown table to stdout. Progress and failure logs go to stderr, so
#
#   devenv shell -- bash scripts/hardware_acceptance.sh > hardware-acceptance.md
#
# keeps only the table. The pinned artifacts must be in .deps/
# (scripts/fetch_tt_metal.sh). Emission is fixed to n150:
# scripts/build_generated.sh passes --tensixflow-device=n150, so the probe
# must match the n150 spec entry and another matched device exits 2.
#
# For an acceptance run, the script exits 0 only when all five runs pass, 1
# when an example fails to build or run, and 2 before any build when the
# probe does not confirm an n150 card, the pinned artifacts are missing,
# the link flags are unset, or the compiler build fails.
#
# The card path is untested on the development machine.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
deps_env="$repo_root/.deps/env.sh"
probe_script="$repo_root/scripts/device_probe.py"

# Named outcomes separate a failed run from a run that could not start at all.
EXIT_ALL_PASS=0
EXIT_RUN_FAILED=1
EXIT_NOT_STARTED=2

# The five examples that emit, build, and run. The large rmsnorm_matmul and
# mlp_block inputs are analysis only; the emitter rejects them.
EXAMPLE_PATHS=(
    examples/matmul.mlir
    examples/matmul_padded.mlir
    examples/matmul_small.mlir
    examples/rmsnorm_matmul_small.mlir
    examples/mlp_block_small.mlir
)

fail() {
    echo "error: $*" >&2
    exit "$EXIT_NOT_STARTED"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        -h|--help)
            sed -n '2,26p' "${BASH_SOURCE[0]}"
            exit "$EXIT_ALL_PASS"
            ;;
        *)
            fail "unknown argument '$1'; this script takes no arguments"
            ;;
    esac
done

# The card probe is the only authority on whether a card is visible, so it
# runs before any build work. A failed probe ends the run here; there is no
# fallback to the simulator, because a simulator result is not a device result.
echo "== probe the card ==" >&2
probe_exit=0
probe_report="$(python3 "$probe_script" 2>&1)" || probe_exit=$?
if [ "$probe_exit" -ne 0 ]; then
    printf '%s\n' "$probe_report" >&2
    if [ "$probe_exit" -eq 1 ]; then
        fail "scripts/device_probe.py found a card that does not match spec/devices.json; fix the board, firmware, or tt-kmd first"
    fi
    fail "scripts/device_probe.py did not confirm a card; device acceptance needs an n150 card"
fi
printf '%s\n' "$probe_report" >&2

# The probe names the spec entry that matched each card. The table records that
# name; when no line matches, the label stays unknown rather than guessed.
device_label="$(sed -n "s/.*-> spec entry '\([^']*\)'.*/\1/p" <<<"$probe_report" | head -n 1)"
if [ -z "$device_label" ]; then
    device_label="unknown"
fi
run_date="$(date -u +%Y-%m-%d)"

# The emitted projects are fixed to n150. A table that recorded another card
# would name hardware the plan did not target, so a mismatch stops here.
emit_device="n150"
if [ "$device_label" = "unknown" ]; then
    fail "the probe report does not name a spec entry; the n150 emit target cannot be matched to the card"
fi
if [ "$device_label" != "$emit_device" ]; then
    fail "the probe matched '$device_label', but the emitted projects target $emit_device; extend the emitter before running acceptance on another device"
fi

# The emitted projects link against the pinned release, which the fetch script
# unpacks into .deps/ together with the environment it exports.
if [ ! -f "$deps_env" ]; then
    fail "the pinned TT-Metalium artifacts are not fetched; run: devenv shell -- bash scripts/fetch_tt_metal.sh"
fi
# shellcheck source=/dev/null
if ! source "$deps_env"; then
    fail "cannot load the pinned environment from $deps_env; re-run: devenv shell -- bash scripts/fetch_tt_metal.sh"
fi

# The link flags come from devenv.nix. Without the environment the emitted
# projects cannot link, so the script stops here instead of recording five
# identical build failures.
if [ -z "${TT_METAL_SYSTEM_LINK_FLAGS:-}" ]; then
    fail "TT_METAL_SYSTEM_LINK_FLAGS is unset; run this script inside 'devenv shell'"
fi

# Emission needs tensixflow-opt. Building it on first use keeps the card host
# to one command after the fetch step; the resolved path is exported so the
# emitted-project builds use the same binary.
find_opt() {
    if [ -n "${TENSIXFLOW_OPT:-}" ] && [ -x "$TENSIXFLOW_OPT" ]; then
        echo "$TENSIXFLOW_OPT"
        return 0
    fi
    local candidate
    for candidate in \
        "${TENSIXFLOW_BUILD_DIR:-$repo_root/build}/bin/tensixflow-opt" \
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

echo "== find the compiler ==" >&2
if ! opt_path="$(find_opt)"; then
    echo "tensixflow-opt is missing; building the compiler first" >&2
    if ! bash "$repo_root/scripts/build.sh" 1>&2; then
        fail "the compiler build failed; run: devenv shell -- bash scripts/build.sh"
    fi
    opt_path="$(find_opt)" || fail "the compiler build finished, but tensixflow-opt is still missing"
fi
export TENSIXFLOW_OPT="$opt_path"
echo "  compiler: $opt_path" >&2

# Each build and run writes to its own temporary log. A log is printed only
# when that step fails, which keeps stderr short and the table the only stdout.
build_log="$(mktemp)" || fail "cannot create a temporary log under ${TMPDIR:-/tmp}"
if ! run_log="$(mktemp)"; then
    rm -f "$build_log"
    fail "cannot create a temporary log under ${TMPDIR:-/tmp}"
fi
trap 'rm -f "$build_log" "$run_log"' EXIT

rows=()
failures=0
total="${#EXAMPLE_PATHS[@]}"
for example in "${EXAMPLE_PATHS[@]}"; do
    name="$(basename "$example" .mlir)"
    project_dir="$repo_root/build-generated/$name"
    echo "== $example ==" >&2

    # A build failure is a row of its own: the program did not run, so there is
    # no status or PCC line to record.
    if ! bash "$repo_root/scripts/build_generated.sh" "$repo_root/$example" >"$build_log" 2>&1; then
        cat "$build_log" >&2
        rows+=("| $run_date | $device_label | \`$example\` | device | build-failed | - | build failed |")
        failures=$((failures + 1))
        continue
    fi
    echo "  built $project_dir" >&2

    # run_generated.sh prints one JSON object on its last line. The exit status
    # and the JSON status must agree before a row records a pass.
    run_exit=0
    bash "$repo_root/scripts/run_generated.sh" --device "$project_dir" >"$run_log" 2>&1 || run_exit=$?
    result_line="$(tail -n 1 "$run_log")"
    status="$(sed -n 's/.*"status":"\([a-z]*\)".*/\1/p' <<<"$result_line")"
    pcc="$(sed -n 's/.*"pcc":\([0-9.eE+-]*\).*/\1/p' <<<"$result_line")"
    if [ -z "$pcc" ]; then
        pcc="-"
    fi

    if [ "$run_exit" -eq 0 ] && [ "$status" = "pass" ]; then
        rows+=("| $run_date | $device_label | \`$example\` | device | pass | $pcc | - |")
        echo "  pass, PCC $pcc" >&2
        continue
    fi

    # A failing run keeps the PCC when the program printed one, so the table
    # shows how far the numerics landed below the program's own bar.
    cat "$run_log" >&2
    if [ -z "$status" ]; then
        status="no-result"
        note="run_generated.sh exited $run_exit without a JSON result"
    else
        status="fail"
        note="run failed"
    fi
    rows+=("| $run_date | $device_label | \`$example\` | device | $status | $pcc | $note |")
    failures=$((failures + 1))
done

echo "== acceptance table ==" >&2
printf '| Date | Device | Example | Backend | Status | PCC | Notes |\n'
printf '|---|---|---|---|---|---|---|\n'
for row in "${rows[@]}"; do
    printf '%s\n' "$row"
done

passed=$((total - failures))
echo "== result: $passed of $total examples passed ==" >&2
if [ "$failures" -ne 0 ]; then
    echo "error: $failures of $total examples did not pass; the table records the status of every example" >&2
    exit "$EXIT_RUN_FAILED"
fi
echo "paste the table above into docs/device.md and commit the docs update" >&2
exit "$EXIT_ALL_PASS"
