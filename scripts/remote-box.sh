#!/usr/bin/env bash
#
# remote-box.sh — OPTIONAL, opt-in setup for a remote Linux build box.
#
# TensixFlow's MVP is CPU-only and local-first: you do NOT need a GPU or a
# remote machine to build or test. This script exists only for contributors
# who want a separate, always-on Linux build machine (e.g. a rented sandbox)
# so their laptop stays free. See docs/remote-box.md for provider guidance.
#
# The script is IDEMPOTENT: every step is guarded, so re-running it on an
# already-provisioned box is a safe no-op for the completed steps.
#
#   1. install Nix            (only if `nix` is not already on PATH)
#   2. clone the repository   (only if the target dir is not already a clone)
#   3. enter the dev shell    (devenv if present, else `nix develop`)
#   4. configure + build      (cmake + ninja, using the shell's MLIR/LLVM)
#
# This script does NOT contact or provision any cloud provider; it only
# prepares a machine you already have shell access to. It is NOT run in CI.
#
# Usage (on the remote box, as a non-root user with sudo):
#   curl -fsSL <this-script> | bash
#   # or, after cloning:
#   bash scripts/remote-box.sh
#
# Configuration (override via environment):
#   TENSIXFLOW_REPO    git URL to clone        (default: the public repo)
#   TENSIXFLOW_DIR     checkout directory      (default: $HOME/tensixflow)
#   TENSIXFLOW_BUILD   build subdirectory      (default: build)

set -euo pipefail

# --- configuration ---------------------------------------------------------

TENSIXFLOW_REPO="${TENSIXFLOW_REPO:-https://github.com/tensixflow/tensixflow.git}"
TENSIXFLOW_DIR="${TENSIXFLOW_DIR:-$HOME/tensixflow}"
TENSIXFLOW_BUILD="${TENSIXFLOW_BUILD:-build}"

log() { printf '[remote-box] %s\n' "$*" >&2; }

# --- 1. install Nix (guarded) ----------------------------------------------
#
# Nix provides the pinned LLVM/MLIR 20.1.8 toolchain via the dev shell, so it
# is the only hard prerequisite. We install the single-user daemonless variant
# only when `nix` is not already resolvable on PATH.

install_nix() {
  if command -v nix >/dev/null 2>&1; then
    log "nix already installed: $(command -v nix) — skipping install"
    return 0
  fi

  log "nix not found — installing (single-user, daemonless)"
  # Official installer. NIX_INSTALLER_NO_MODIFY_PROFILE avoids touching shell
  # rc files unexpectedly on a shared box; we source the profile manually below.
  curl -fsSL https://install.determinate.systems/nix | sh -s -- install linux \
    --init none --confirm --no-modify-profile

  # Make `nix` available in THIS shell for the rest of the run.
  if [ -e /nix/var/nix/profiles/default/etc/profile.d/nix-daemon.sh ]; then
    # shellcheck disable=SC1091
    . /nix/var/nix/profiles/default/etc/profile.d/nix-daemon.sh
  elif [ -e "$HOME/.nix-profile/etc/profile.d/nix.sh" ]; then
    # shellcheck disable=SC1091
    . "$HOME/.nix-profile/etc/profile.d/nix.sh"
  fi

  command -v nix >/dev/null 2>&1 || {
    log "ERROR: nix install finished but 'nix' is still not on PATH"
    return 1
  }
  log "nix installed: $(command -v nix)"
}

# --- 2. clone the repository (guarded) -------------------------------------

clone_repo() {
  if [ -d "$TENSIXFLOW_DIR/.git" ]; then
    log "repo already cloned at $TENSIXFLOW_DIR — skipping clone"
    return 0
  fi

  if [ -e "$TENSIXFLOW_DIR" ]; then
    log "ERROR: $TENSIXFLOW_DIR exists but is not a git checkout; refusing to clobber"
    return 1
  fi

  log "cloning $TENSIXFLOW_REPO -> $TENSIXFLOW_DIR"
  git clone "$TENSIXFLOW_REPO" "$TENSIXFLOW_DIR"
}

# --- 3+4. enter the dev shell, then configure + build ----------------------
#
# The dev shell (flake) exports MLIR_DIR and LLVM_DIR and puts mlir-tblgen on
# PATH. We run the real build commands inside that shell. Prefer `devenv shell`
# (the verified non-interactive form) when available, else `nix develop`.
#
# Build commands (must match docs/learnings — the verified working set):
#   cmake -G Ninja -B build -DMLIR_DIR=$MLIR_DIR -DLLVM_DIR=$LLVM_DIR
#   cmake --build build

BUILD_CMD="cmake -G Ninja -B '$TENSIXFLOW_BUILD' -DMLIR_DIR=\"\$MLIR_DIR\" -DLLVM_DIR=\"\$LLVM_DIR\" && cmake --build '$TENSIXFLOW_BUILD'"

run_build() {
  cd "$TENSIXFLOW_DIR"

  if command -v devenv >/dev/null 2>&1; then
    log "entering dev shell via 'devenv shell' to configure + build"
    devenv shell bash -c "$BUILD_CMD"
  else
    log "entering dev shell via 'nix develop' to configure + build"
    nix develop --command bash -c "$BUILD_CMD"
  fi

  log "build complete (artifacts under $TENSIXFLOW_DIR/$TENSIXFLOW_BUILD/bin)"
}

# --- main ------------------------------------------------------------------

main() {
  log "starting (repo=$TENSIXFLOW_REPO dir=$TENSIXFLOW_DIR build=$TENSIXFLOW_BUILD)"
  install_nix
  clone_repo
  run_build
  log "done — remote build box is ready."
}

main "$@"
