#!/usr/bin/env bash
#
# hw_bringup_dryrun.sh — NO-DEVICE hardware bring-up dry-run (todo 56 / D19).
#
# TensixFlow targets a Wormhole n150, but this repository is developed on a
# CPU-only box with NO Tenstorrent device. This script is the always-available
# half of the disjunctive hardware acceptance (D12/D19): it PRINTS the
# validated bring-up steps (grounded in tt-metal INSTALLING.md @df36cd6) so a
# future owner of a Wormhole n150 can go from zero to a running generated
# kernel project, then DETECTS the (absent) device and exits 0.
#
# This script NEVER depends on a tt-metal build. The no-device validation of
# the generated kernels is the compile-only check against the signature-only
# stubs (todo 33 / D16), run separately as:
#
#   cmake --build build --target check-generated-kernels-compile
#
# Together, `bash scripts/hw_bringup_dryrun.sh && cmake --build build --target
# check-generated-kernels-compile` is the no-device branch of the disjunctive
# acceptance harness (todo 57). It must PASS on a no-device box and never be
# skipped; on a device box only the on-device branch (todo 55) runs.
#
# Exit status: ALWAYS 0 (this is a runbook dry-run, not a probe gate). Device
# presence only changes the closing message (which branch to continue with).

set -euo pipefail

log() { printf '[hw-bringup-dryrun] %s\n' "$*"; }

# --- device detection (the D19 probe target: /dev/tenstorrent) --------------

DEVICE_PRESENT=0
if [ -e /dev/tenstorrent ]; then
  DEVICE_PRESENT=1
fi

log "device probe: /dev/tenstorrent"
if [ "$DEVICE_PRESENT" -eq 1 ]; then
  log "device DETECTED: a Tenstorrent device node exists."
  log "This dry-run still exits 0, but you want the ON-DEVICE branch (todo 55),"
  log "NOT the no-device fallback: compile+run the generated project against"
  log "the real tt-metal and validate PCC vs the golden."
else
  log "no device detected - using the no-device branch (this is expected:"
  log "no Tenstorrent hardware is owned). The steps below print what WOULD"
  log "be done on a Wormhole n150 box; nothing here requires hardware."
fi

# --- the validated bring-up steps (tt-metal INSTALLING.md @df36cd6) ----------

log ""
log "Validated bring-up steps for a Wormhole n150 (five steps):"
log ""
log "  Step 1 - obtain/access a Wormhole n150"
log "    Get shell access to a box with a Tenstorrent Wormhole n150 card"
log "    seated (PCIe, per the hardware setup guide at docs.tenstorrent.com)."
log "    The model this repo targets: 8x8 Tensix grid, 32x32 tiles, bf16."
log ""
log "  Step 2 - install the KMD (kernel-mode driver) + host tools"
log "    Use the TT-Installer script (tt-metal INSTALLING.md prereq #2):"
log "      curl -fsSL https://github.com/tenstorrent/tt-installer/releases/download/v2.1.0/install.sh -O"
log "      chmod +x install.sh"
log "      ./install.sh --install-container-runtime=no"
log "    This installs TT-KMD (driver >= v2.8.0), tt-smi, and tt-flash."
log "    When the KMD has loaded, the device node /dev/tenstorrent exists."
log ""
log "  Step 3 - build tt-metal WITH the runtime (from source)"
log "    The runtime (host API + device dispatch) is required to actually"
log "    enqueue kernels; the stub headers this repo compiles against are"
log "    only for the no-device structural check. At the pinned revision:"
log "      git clone https://github.com/tenstorrent/tt-metal.git --recurse-submodules"
log "      cd tt-metal && git checkout df36cd6"
log "      ./build_metal.sh   # or cmake -G Ninja -DCMAKE_BUILD_TYPE=Release + ninja"
log ""
log "  Step 4 - flash the firmware"
log "    Flash the device firmware bundle before first use (tt-installer"
log "    passes --fw-version to tt-flash; Wormhole boards in the INSTALLING"
log "    table use fw_pack-19.8.1.fwbundle / v19.8.1). Verify with tt-smi:"
log "      tt-smi -s   # shows the card, FW version, and health"
log ""
log "  Step 5 - run the generated project on device"
log "    Build the generated standalone project kernels/generated/fused_rmsnorm_matmul/"
log "    (host.cpp + reader.cpp + compute.cpp + writer.cpp) against the REAL"
log "    tt-metal (NOT the stubs), enqueue it with distributed::EnqueueMeshWorkload,"
log "    read back the output, and validate PCC >= 0.9875 vs the NumPy golden"
log "    (the same bar as the simulator harness). That is the on-device branch"
log "    (todo 55); results are recorded in docs/hardware_path.md."
log ""

# --- the no-device validation pointer ----------------------------------------

log "No-device validation (what THIS box runs instead of the five steps):"
log "  bash scripts/hw_bringup_dryrun.sh                 (this script, exits 0)"
log "  cmake --build build --target check-generated-kernels-compile"
log "    -> compiles every generated kernel -fsyntax-only against the"
log "       signature-only stubs pinned to tt-metal @df36cd6 (non-vacuous:"
log "       baseline compiles AND the removed legacy API is rejected)."
log "  No tt-metal build, no device, no network required."

log ""
log "done - dry-run finished without hardware."
exit 0
