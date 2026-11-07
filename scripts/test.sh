#!/usr/bin/env bash
# Run every check that works on a CPU-only machine. Run through devenv:
#
#   devenv shell -- bash scripts/test.sh
#
# The emitted-project checks need the pinned tt-metal release artifacts and are
# skipped with an explicit message when .deps is absent. They never pass
# without compiling.
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
build_dir="${TENSIXFLOW_BUILD_DIR:-${repo_root}/build}"

echo "== build =="
bash "${repo_root}/scripts/build.sh" "${build_dir}"

echo "== lit/FileCheck =="
cmake --build "${build_dir}" --target check-tensixflow

echo "== ctest =="
ctest --test-dir "${build_dir}" --output-on-failure

echo "== python =="
if [ ! -d "${repo_root}/.deps/ttmlir/site" ]; then
  echo "note: run scripts/fetch_ttmlir_fixtures.sh to activate the TTIR parser gate"
fi
python3 -m pytest "${repo_root}/python" -q

echo "== prose =="
python3 "${repo_root}/tools/check-prose.py"

echo "== device spec =="
if [ -f "${repo_root}/spec/devices.json" ]; then
  python3 "${repo_root}/tools/gen_device_spec.py" --check
else
  echo "skipping: spec/devices.json is not present"
fi

echo "== emitted project against pinned tt-metal =="
if [ -f "${repo_root}/.deps/env.sh" ]; then
  bash "${repo_root}/scripts/build_generated.sh"
  bash "${repo_root}/scripts/run_generated.sh"
else
  echo "skipping: run scripts/fetch_tt_metal.sh first to compile and run an emitted project"
fi

echo "all CPU-side checks passed"
