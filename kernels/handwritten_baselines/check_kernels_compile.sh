#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Compile-only structural check for generated/baseline TT-Metalium kernels.
#
# Two assertions, BOTH required for success (this is what makes the check
# non-vacuous):
#
#   1. POSITIVE: the current-API baseline kernel compiles `-fsyntax-only`
#      against the signature-only stubs. If it does NOT compile, the stubs have
#      dropped a current symbol -> FAIL.
#
#   2. NEGATIVE: a kernel referencing a REMOVED legacy symbol (EnqueueProgram)
#      MUST FAIL to compile, and the diagnostic MUST name EnqueueProgram. If it
#      compiles, the stubs have drifted back to a legacy API -> FAIL.
#
# Usage: check_kernels_compile.sh <CXX> <STUBS_DIR> <BASELINE.cpp> <NEGATIVE.cpp>
set -u

cxx="$1"
stubs="$2"
baseline="$3"
negative="$4"

flags=(-std=c++20 -fsyntax-only -Wall -I "$stubs")

echo "[check-generated-kernels-compile] compiler: $cxx"
echo "[check-generated-kernels-compile] stubs:    $stubs"

# --- 1. POSITIVE: baseline must compile -------------------------------------
echo "[positive] compiling $(basename "$baseline") (must SUCCEED)..."
if ! "$cxx" "${flags[@]}" "$baseline"; then
    echo "FAIL: current-API baseline did NOT compile against the stubs." >&2
    echo "      The stubs are missing a current tt-metal symbol the kernel uses." >&2
    exit 1
fi
echo "[positive] OK: baseline compiled -fsyntax-only."

# --- 2. NEGATIVE: legacy kernel must FAIL, naming EnqueueProgram ------------
echo "[negative] compiling $(basename "$negative") (must FAIL with 'EnqueueProgram' undeclared)..."
neg_err="$("$cxx" "${flags[@]}" "$negative" 2>&1)"
neg_status=$?

if [ "$neg_status" -eq 0 ]; then
    echo "FAIL: legacy kernel COMPILED — the stubs declare a removed legacy symbol." >&2
    echo "      The compile check would be VACUOUS. EnqueueProgram must stay undeclared." >&2
    exit 1
fi

if ! printf '%s' "$neg_err" | grep -q "EnqueueProgram"; then
    echo "FAIL: legacy kernel failed, but NOT because EnqueueProgram is undeclared." >&2
    echo "      Compiler output was:" >&2
    printf '%s\n' "$neg_err" >&2
    exit 1
fi

echo "[negative] OK: legacy kernel rejected (EnqueueProgram undeclared), exit=$neg_status."
echo "[check-generated-kernels-compile] PASS: current API compiles AND legacy API is rejected."
