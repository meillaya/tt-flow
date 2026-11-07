#!/usr/bin/env bash
# Launch Chromium for the excalidraw CLI inside a restricted sandbox.
#
# The development container has no writable home directory, so Chromium's
# crash reporter cannot create its database and the browser aborts before
# rendering. This wrapper gives the browser a scratch HOME and the flags that
# are safe for a disposable, non-network rendering job. It is only used by
# scripts/render_figures.sh; it is not part of the compiled project.
set -euo pipefail

if [ -z "${CHROMIUM_BIN:-}" ]; then
  echo "chromium_sandbox_wrapper: set CHROMIUM_BIN to the chromium executable" >&2
  exit 2
fi

scratch_root="${TENSIXFLOW_BROWSER_SCRATCH:-/tmp/tt-flow-browser}"
mkdir -p "${scratch_root}/home" "${scratch_root}/tmp" "${scratch_root}/config"
export HOME="${scratch_root}/home"
export TMPDIR="${scratch_root}/tmp"
export XDG_CONFIG_HOME="${scratch_root}/config"

exec "${CHROMIUM_BIN}" \
  --no-sandbox \
  --disable-gpu \
  --disable-dev-shm-usage \
  --disable-crash-reporter \
  --user-data-dir="${scratch_root}/profile" \
  "$@"
