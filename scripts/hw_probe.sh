#!/usr/bin/env bash
#
# hw_probe.sh - device detection + tt-metal-runtime build as a SKIPPABLE probe.
#
# A GATE, NEVER a blocker (.omo/plans/tensixflow.md todo 54, draft D19).
# Detects /dev/tenstorrent (tt-metal INSTALLING.md hardware prereq #1) and
# branches:
#
#   * device ABSENT (this box): prints "no device — using no-device branch"
#     and exits 0. The no-device fallback branch is fully INDEPENDENT of this
#     probe (D19: it uses the D16 stub headers + compile-only checks, never a
#     tt-metal-with-runtime build), so the probe is never a prerequisite.
#   * device PRESENT: attempts a tt-metal build WITH the runtime enabled
#     (TT_METAL_RUNTIME=ON-equivalent) and REPORTS the outcome. A heavy or
#     failing build is reported only.
#
# This script NEVER exits non-zero solely because the device or the build is
# absent or failing: the probe reports, the caller decides. There is therefore
# no die() here (unlike build-tt-mlir.sh): every failure is logged and the
# script still finishes with exit 0. Only `set -u` + pipefail are used (-e is
# deliberately absent: a probe must report failures, not die on them).
#
# Usage:
#   bash scripts/hw_probe.sh
#
# Configuration (override via environment):
#   TT_DEVICE_PATH         path to probe                  (default: /dev/tenstorrent)
#   TT_METAL_REPO          git URL to clone               (default: upstream)
#   TT_METAL_COMMIT        commit to pin                  (default: df36cd6)
#   TT_METAL_DIR           checkout directory             (default: $HOME/tt-metal)
#   BUILD_PARALLEL_LEVEL   parallel compile jobs          (default: 4, OOM-safe)
#   HW_PROBE_BUILD_TIMEOUT seconds before killing build   (default: 0 = none)

set -uo pipefail

# --- configuration -----------------------------------------------------------

TT_DEVICE_PATH="${TT_DEVICE_PATH:-/dev/tenstorrent}"
TT_METAL_REPO="${TT_METAL_REPO:-https://github.com/tenstorrent/tt-metal.git}"
# tt-metal @df36cd6, the commit the TensixFlow stub headers (D16) are pinned to.
TT_METAL_COMMIT="${TT_METAL_COMMIT:-df36cd6}"
TT_METAL_DIR="${TT_METAL_DIR:-$HOME/tt-metal}"
BUILD_PARALLEL_LEVEL="${BUILD_PARALLEL_LEVEL:-4}"
HW_PROBE_BUILD_TIMEOUT="${HW_PROBE_BUILD_TIMEOUT:-0}"
case "${HW_PROBE_BUILD_TIMEOUT}" in
  ''|*[!0-9]*) HW_PROBE_BUILD_TIMEOUT=0 ;;
esac

log() { printf '[hw-probe] %s\n' "$*" >&2; }

finish() {  # <RESULT label> — always exit 0: gate, never blocker
  log "PROBE RESULT: $1 (report only; never blocks the wave — D19)"
  exit 0
}

# --- 1. device detection (tt-metal INSTALLING.md hardware prereq #1) ----------

if [ ! -e "${TT_DEVICE_PATH}" ]; then
  log "no device — using no-device branch"
  log "(probed ${TT_DEVICE_PATH}: tt-metal INSTALLING.md hardware prereq #1 absent)"
  finish "NO_DEVICE"
fi

log "device path present: ${TT_DEVICE_PATH}"
ls -l "${TT_DEVICE_PATH}" >&2 2>/dev/null || true
if [ ! -c "${TT_DEVICE_PATH}" ]; then
  log "WARNING: ${TT_DEVICE_PATH} is present but NOT a character device — likely"
  log "broken, a placeholder, or a stale node. The runtime build attempt below"
  log "will report what happens; the probe still exits 0 (gate, never blocker)."
elif [ ! -r "${TT_DEVICE_PATH}" ] || [ ! -w "${TT_DEVICE_PATH}" ]; then
  log "WARNING: ${TT_DEVICE_PATH} is not user read/writable — check udev rules /"
  log "group membership per tt-metal INSTALLING.md before expecting a runtime run."
fi

# --- 2. locate (or clone) the pinned tt-metal checkout (idempotent) -----------

if [ -f "${TT_METAL_DIR}/CMakeLists.txt" ] || [ -d "${TT_METAL_DIR}/.git" ]; then
  log "tt-metal checkout present: ${TT_METAL_DIR}"
else
  log "no tt-metal checkout at ${TT_METAL_DIR} — cloning ${TT_METAL_REPO} @ ${TT_METAL_COMMIT}"
  clone_ok=1
  if command -v git >/dev/null 2>&1; then
    if git clone "${TT_METAL_REPO}" "${TT_METAL_DIR}" \
        && git -C "${TT_METAL_DIR}" checkout --detach "${TT_METAL_COMMIT}"; then
      clone_ok=0
    fi
  else
    log "git not on PATH — cannot clone tt-metal"
  fi
  if [ "${clone_ok}" -ne 0 ]; then
    log "FAILED: could not obtain tt-metal @ ${TT_METAL_COMMIT} into ${TT_METAL_DIR}"
    finish "BUILD_FAILED (checkout unavailable)"
  fi
fi

# --- 3. attempt the runtime build (TT_METAL_RUNTIME=ON-equivalent) -------------
#
# Upstream entry point is ./build_metal.sh (it honours the TT_METAL_RUNTIME env
# var); fall back to a plain cmake configure+build with -DTT_METAL_RUNTIME=ON.
# Either way the attempt may take 30+ minutes on real hardware boxes; cap it
# with HW_PROBE_BUILD_TIMEOUT if the caller wants a bound.

export TT_METAL_RUNTIME=ON
export CMAKE_BUILD_PARALLEL_LEVEL="${BUILD_PARALLEL_LEVEL}"

maybe_timeout() {
  if [ "${HW_PROBE_BUILD_TIMEOUT}" -gt 0 ] && command -v timeout >/dev/null 2>&1; then
    timeout "${HW_PROBE_BUILD_TIMEOUT}" "$@"
  else
    "$@"
  fi
}

build_ok=1
if [ -x "${TT_METAL_DIR}/build_metal.sh" ]; then
  log "building via ${TT_METAL_DIR}/build_metal.sh (TT_METAL_RUNTIME=ON)"
  if ( cd "${TT_METAL_DIR}" && maybe_timeout ./build_metal.sh ); then
    build_ok=0
  fi
elif [ -f "${TT_METAL_DIR}/CMakeLists.txt" ]; then
  log "building via cmake -DTT_METAL_RUNTIME=ON (build dir: ${TT_METAL_DIR}/build_probe)"
  if ( cd "${TT_METAL_DIR}" \
      && maybe_timeout cmake -G Ninja -B build_probe \
           -DCMAKE_BUILD_TYPE=Release -DTT_METAL_RUNTIME=ON \
      && maybe_timeout cmake --build build_probe ); then
    build_ok=0
  fi
else
  log "FAILED: ${TT_METAL_DIR} is not a tt-metal checkout (no build_metal.sh or"
  log "CMakeLists.txt) — nothing to build against the device."
  finish "BUILD_FAILED (not a tt-metal checkout)"
fi

# --- 4. report the outcome (gate: either result exits 0) -----------------------

if [ "${build_ok}" -eq 0 ]; then
  log "tt-metal runtime build SUCCEEDED with device ${TT_DEVICE_PATH}"
  finish "BUILD_OK"
else
  log "tt-metal runtime build FAILED (see output above). Reported only — a heavy"
  log "or failing build never blocks the wave (gate, not blocker; D19)."
  finish "BUILD_FAILED"
fi
