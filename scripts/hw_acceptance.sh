#!/usr/bin/env bash
#
# hw_acceptance.sh — DISJUNCTIVE hardware acceptance harness (todo 57, D19/D25).
#
# EXPLICIT if/else — NEVER `&&...||`: an `on-device ... || no-device` form would
# let a PRESENT-but-FAILING device run fall through to the passing no-device
# branch and be masked. With if/else exactly ONE branch runs; the exit status
# of this script is that branch's status, unmasked:
#
#   * device PRESENT (if arm, todo 55): the ON-DEVICE branch — compile the
#     generated project against the REAL tt-metal, run it, and validate
#     PCC >= 0.9875 vs the NumPy golden via scripts/hw_run_on_device.sh.
#     That entry point does NOT exist yet (todo 55 is SKIPPED: no Tenstorrent
#     device is owned). If this arm is ever reached without it — or the run
#     fails — the script FAILS HONESTLY with a non-zero exit. It NEVER falls
#     through to the no-device branch (D19/D25 no-masking).
#   * device ABSENT (else arm, todo 56): the NO-DEVICE branch — bring-up
#     runbook dry-run + the compile-only check of every generated kernel
#     against the D16 signature-only stubs. Must PASS on a no-device box,
#     never be skipped; it does REAL work (non-vacuous), so a broken kernel
#     fails this branch honestly too.
#
# Exit status: 0 iff the ACTIVE branch succeeds; non-zero if it fails. The
# inactive branch never runs and can never mask the active one.
#
# Usage:
#   bash scripts/hw_acceptance.sh              (or: ctest -R hw-acceptance)
#
# Configuration (override via environment, mirroring hw_probe.sh todo 54):
#   TT_DEVICE_PATH  device path to probe  (default: /dev/tenstorrent)
#                   Overridable so the device-present path is simulatable for
#                   QA without touching /dev or owning hardware.

set -euo pipefail

# --- configuration ------------------------------------------------------------

TT_DEVICE_PATH="${TT_DEVICE_PATH:-/dev/tenstorrent}"

log() { printf '[hw-acceptance] %s\n' "$*" >&2; }

# Anchor at the repository root: the no-device arm runs the literal D19 command
# pair with repo-relative paths (scripts/... , build/), which must resolve from
# the root regardless of the caller CWD (ctest defaults to the build dir).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

# --- the disjunction (D19/D25): EXPLICIT if/else, never `&&...||` --------------

if [ -e "${TT_DEVICE_PATH}" ]; then
  # ON-DEVICE branch (todo 55): compile + run + PCC >= 0.9875 vs the golden.
  # On a device box ONLY this branch runs and it may fail honestly.
  log "device present: ${TT_DEVICE_PATH} — running the ON-DEVICE branch (todo 55)"
  log "only the on-device branch runs on a device box; a failure here is never masked"
  if [ ! -x scripts/hw_run_on_device.sh ]; then
    log "FAIL: scripts/hw_run_on_device.sh is not present or not executable."
    log "      The on-device entry point (todo 55) has not landed on this box;"
    log "      refusing to fall through to the no-device branch (D19: no masking)."
    exit 1
  fi
  bash scripts/hw_run_on_device.sh
else
  # NO-DEVICE branch (todo 56): runbook dry-run + compile-only kernel check.
  # Must PASS on a no-device box; never a no-op (the compile check is real).
  log "no device at ${TT_DEVICE_PATH} — running the NO-DEVICE branch (todo 56)"
  bash scripts/hw_bringup_dryrun.sh && cmake --build build --target check-generated-kernels-compile
fi
