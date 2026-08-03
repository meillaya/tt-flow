#!/usr/bin/env bash
#
# build-tt-mlir.sh - OPTIONAL tt-mlir build + ttmlir-opt study helper.
#
# OFF THE MVP CRITICAL PATH. Nothing in the TensixFlow CMake build invokes
# this script; check-tensixflow and check-generated-kernels-compile never
# depend on it. It exists for the OPTIONAL tt-mlir compatibility study
# (.omo/plans/tensixflow.md todo 50, draft D10): clone tt-mlir at the pinned
# commit, build ONLY the offline compiler (ttmlir-opt) with the runtime
# disabled, and run a sample StableHLO -> TTIR -> TTNN pipeline so the
# captured op shapes can be folded into docs/ttmlir_study.md.
#
# WARNINGS, READ FIRST:
#   * HEAVY BUILD. tt-mlir builds its OWN LLVM toolchain (env/CMakeLists.txt,
#     llvm-project @4efe170d, MinSizeRel) plus flatbuffers and a venv before
#     the project itself compiles. Expect roughly 30-60 minutes on a modern
#     box and several GB of disk. This is why the default TensixFlow workflow
#     takes the source-study fallback instead of running this script.
#   * RUNTIME STAYS OFF. The script configures -DTTMLIR_ENABLE_RUNTIME=OFF
#     (the tt-mlir default, CMakeLists.txt option line). The runtime needs a
#     Tenstorrent device and the tt-metal host dependencies; this study box
#     has neither, and the offline compiler is all the study needs.
#   * SUPPORTED HOST. tt-mlir @this pin documents "Offline Compiler Only" for
#     Ubuntu 24.04 (also Ubuntu 20.04 and macOS); see docs/src/getting-started.md
#     in the tt-mlir repo for the build matrix and system dependencies
#     (clang 14-18, ninja, cmake >= 3.24, python3.12 + python3.12-venv).
#
# The script is IDEMPOTENT: every step is guarded, so re-running it on an
# already-provisioned machine is a safe no-op for the completed steps.
#
#   1. clone tt-mlir at the pinned commit   (only if the dir is missing)
#   2. check the toolchain dir exists       (prints the sudo mkdir/chown)
#   3. build the env toolchain (LLVM etc.)  (only if the venv is missing)
#   4. source env/activate + configure      (-DTTMLIR_ENABLE_RUNTIME=OFF)
#   5. build the ttmlir-opt target only     (parallel level 4 by default)
#   6. run the sample pipeline + capture    (StableHLO -> TTIR -> TTNN, +D2M)
#
# Usage:
#   bash scripts/build-tt-mlir.sh
#
# Configuration (override via environment):
#   TT_MLIR_REPO           git URL to clone           (default: upstream)
#   TT_MLIR_COMMIT         commit to pin              (default: 9f06802f-era)
#   TT_MLIR_DIR            checkout directory         (default: $HOME/tt-mlir-study)
#   TTMLIR_TOOLCHAIN_DIR   toolchain install prefix   (default: /opt/ttmlir-toolchain)
#   BUILD_PARALLEL_LEVEL   parallel compile jobs      (default: 4, OOM-safe)

set -euo pipefail

# --- configuration ----------------------------------------------------------

TT_MLIR_REPO="${TT_MLIR_REPO:-https://github.com/tenstorrent/tt-mlir.git}"
# tt-mlir @9f06802f (full SHA), the commit the TensixFlow mapping was studied at.
TT_MLIR_COMMIT="${TT_MLIR_COMMIT:-9f06802f5494f41ccdce4d3f725462f7ce13cdc1}"
TT_MLIR_DIR="${TT_MLIR_DIR:-$HOME/tt-mlir-study}"
TTMLIR_TOOLCHAIN_DIR="${TTMLIR_TOOLCHAIN_DIR:-/opt/ttmlir-toolchain}"
# getting-started.md: "to workaround OOM issues it can be useful to decrease
# the number of parallel jobs with -DCMAKE_BUILD_PARALLEL_LEVEL=4".
BUILD_PARALLEL_LEVEL="${BUILD_PARALLEL_LEVEL:-4}"

log()  { printf '[build-tt-mlir] %s\n' "$*" >&2; }
die()  { printf '[build-tt-mlir] ERROR: %s\n' "$*" >&2; exit 1; }

log "WARNING: heavy build ahead (~30-60 min). tt-mlir compiles its own LLVM"
log "toolchain (env/CMakeLists.txt, llvm-project @4efe170d) before the project."
log "The runtime stays OFF (-DTTMLIR_ENABLE_RUNTIME=OFF): it needs a Tenstorrent"
log "device, and the offline compiler (ttmlir-opt) is all this study needs."
log "Captured op shapes land under ${TT_MLIR_DIR}/study/ for docs/ttmlir_study.md."

# --- 0. prerequisites ---------------------------------------------------------
#
# Per tt-mlir docs/src/getting-started.md: git, clang (14-18), ninja,
# cmake >= 3.24, python3.12 with venv. We only check presence, not versions:
# the tt-mlir configure step fails loudly with actionable messages otherwise.

for tool in git cmake ninja clang clang++ python3; do
  command -v "$tool" >/dev/null 2>&1 || die "missing prerequisite: $tool (see tt-mlir docs/src/getting-started.md)"
done
log "prerequisites present: git, cmake, ninja, clang, clang++, python3"

# --- 1. clone tt-mlir at the pinned commit (guarded) --------------------------

if [ -d "${TT_MLIR_DIR}/.git" ]; then
  log "clone already present: ${TT_MLIR_DIR} (fetching + checking out the pin)"
  git -C "${TT_MLIR_DIR}" fetch origin
else
  log "cloning ${TT_MLIR_REPO} -> ${TT_MLIR_DIR}"
  git clone "${TT_MLIR_REPO}" "${TT_MLIR_DIR}"
fi
git -C "${TT_MLIR_DIR}" checkout --detach "${TT_MLIR_COMMIT}"
# Submodules are not required for the offline compiler at this pin (third_party
# tt-metal is only pulled when the runtime is enabled), so no --recurse here.
log "pinned: $(git -C "${TT_MLIR_DIR}" rev-parse HEAD)"

# --- 2. toolchain directory (manual sudo step, printed not run) ---------------
#
# env/CMakeLists.txt FATAL_ERRORs with exactly this instruction when the
# toolchain prefix does not exist. We never run sudo from a script; the
# operator runs the two printed lines once.

if [ ! -d "${TTMLIR_TOOLCHAIN_DIR}" ]; then
  die "toolchain dir missing. Run once, then re-run this script:
  sudo mkdir -p ${TTMLIR_TOOLCHAIN_DIR}
  sudo chown -R $(id -un) ${TTMLIR_TOOLCHAIN_DIR}"
fi
export TTMLIR_TOOLCHAIN_DIR

# --- 3. build the env toolchain: LLVM + flatbuffers + venv (guarded) ----------
#
# This is the heavy ~30-60 min step. The marker for a finished toolchain is
# the venv activate script env/activate sources. The exact two commands come
# from tt-mlir .github/Dockerfile.ci (ANCHOR environment_build), which the
# getting-started page includes verbatim.

if [ -f "${TTMLIR_TOOLCHAIN_DIR}/venv/bin/activate" ]; then
  log "toolchain venv present: ${TTMLIR_TOOLCHAIN_DIR}/venv (skipping env build)"
else
  log "building the env toolchain (LLVM @4efe170d, MinSizeRel): ~30-60 min"
  export CMAKE_BUILD_PARALLEL_LEVEL="${BUILD_PARALLEL_LEVEL}"
  cmake -B env/build env -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
  cmake --build env/build
fi

# --- 4. activate + configure the project (runtime OFF) ------------------------
#
# env/activate exports TTMLIR_ENV_ACTIVATED, TTMLIR_TOOLCHAIN_DIR,
# TTMLIR_VENV_DIR and prepends build/bin to PATH; the top-level CMakeLists.txt
# FATAL_ERRORs without it. It must be sourced from the tt-mlir root.

cd "${TT_MLIR_DIR}"
# shellcheck disable=SC1091
source env/activate

# TTMLIR_ENABLE_RUNTIME=OFF is the upstream default (option() in the top
# CMakeLists.txt) and is passed again explicitly: this study must never drag
# in the device runtime. TTMLIR_ENABLE_STABLEHLO=ON is needed for the sample
# StableHLO -> TTIR pipeline (OFF by default). Python bindings are OFF because
# the study only needs ttmlir-opt and the bindings slow the build down.
cmake -G Ninja -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DTTMLIR_ENABLE_RUNTIME=OFF \
  -DTTMLIR_ENABLE_STABLEHLO=ON \
  -DTTMLIR_ENABLE_BINDINGS_PYTHON=OFF \
  -DCMAKE_BUILD_PARALLEL_LEVEL="${BUILD_PARALLEL_LEVEL}"

# --- 5. build ttmlir-opt only --------------------------------------------------

cmake --build build --target ttmlir-opt
TTMLIR_OPT="${TT_MLIR_DIR}/build/bin/ttmlir-opt"
[ -x "${TTMLIR_OPT}" ] || die "ttmlir-opt not found at ${TTMLIR_OPT}"
"${TTMLIR_OPT}" --help >/dev/null
log "ttmlir-opt builds and runs: ${TTMLIR_OPT}"

# --- 6. sample pipeline: StableHLO -> TTIR -> TTNN (+ TTIR -> D2M) -------------
#
# A matmul + add micro-graph, the same op pair TensixFlow lowers. Capture the
# TTIR and TTNN forms (op names + tensor shapes) for docs/ttmlir_study.md.

STUDY_DIR="${TT_MLIR_DIR}/study"
mkdir -p "${STUDY_DIR}"
cat > "${STUDY_DIR}/sample_stablehlo.mlir" <<'EOF'
module {
  func.func @sample(%a: tensor<64x128xbf16>, %b: tensor<128x64xbf16>,
                    %c: tensor<64x64xbf16>) -> tensor<64x64xbf16> {
    %0 = stablehlo.dot_general(%a, %b),
        contracting_dims = [1] x [0]
        : (tensor<64x128xbf16>, tensor<128x64xbf16>) -> tensor<64x64xbf16>
    %1 = stablehlo.add(%0, %c) : tensor<64x64xbf16>
    return %1 : tensor<64x64xbf16>
  }
}
EOF

log "running: --stablehlo-to-ttir-pipeline"
"${TTMLIR_OPT}" --stablehlo-to-ttir-pipeline \
  "${STUDY_DIR}/sample_stablehlo.mlir" -o "${STUDY_DIR}/sample_ttir.mlir"

log "running: --ttir-to-ttnn-backend-pipeline"
"${TTMLIR_OPT}" --ttir-to-ttnn-backend-pipeline \
  "${STUDY_DIR}/sample_ttir.mlir" -o "${STUDY_DIR}/sample_ttnn.mlir"

log "running: --ttcore-register-device --ttir-to-d2m (direct-to-metal path)"
"${TTMLIR_OPT}" --ttcore-register-device --ttir-to-d2m \
  "${STUDY_DIR}/sample_ttir.mlir" -o "${STUDY_DIR}/sample_d2m.mlir"

log "captured shapes for docs/ttmlir_study.md:"
log "  TTIR : ${STUDY_DIR}/sample_ttir.mlir  (ttir.matmul / ttir.add + shapes)"
log "  TTNN : ${STUDY_DIR}/sample_ttnn.mlir  (ttnn.* runtime-dialect form)"
log "  D2M  : ${STUDY_DIR}/sample_d2m.mlir   (direct-to-metal tile form)"
log "done. Paste the captured op lines into docs/ttmlir_study.md if the"
log "source-study fallback section should be upgraded with live output."
