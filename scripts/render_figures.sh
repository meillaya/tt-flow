#!/usr/bin/env bash
# Regenerate the figure scenes and render them to PNG with the excalidraw CLI.
#
# The CLI is the packaged one on the configured hosts. Chromium is provisioned
# by devenv.nix through the render-figures script, so run this through
# `devenv shell -- render-figures` after editing the scenes.
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)

python3 "$repo_root/scripts/gen_figures.py"

cli="${EXCALIDRAW_CLI:-excalidraw}"
if ! command -v "$cli" >/dev/null 2>&1; then
  if [ -x "${HOME}/.nix-profile/bin/excalidraw" ]; then
    cli="${HOME}/.nix-profile/bin/excalidraw"
  else
    echo "render_figures: excalidraw CLI not found on PATH" >&2
    exit 1
  fi
fi

render_args=(--scale 2 --max-width 3200)
if [ -n "${EXCALIDRAW_BROWSER_PATH:-}" ]; then
  render_args+=(--browser-path "${EXCALIDRAW_BROWSER_PATH}")
fi

for scene in "$repo_root"/docs/figures/*.excalidraw; do
  out="${scene%.excalidraw}.png"
  rm -f "$out"
  echo "rendering ${scene#"$repo_root"/}"
  "$cli" render "$scene" --out "$out" "${render_args[@]}"
  if [ ! -s "$out" ]; then
    echo "render_figures: ${out#"$repo_root"/} was not written" >&2
    exit 1
  fi
done

echo "figures rendered"
