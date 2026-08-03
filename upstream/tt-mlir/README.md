# Upstream contribution (todo 59): ttir.softmax verifier negative lit test

This directory holds the IMPLEMENTED version of the contribution scouted in
todo 58 ([docs/upstream_contribution.md](../../docs/upstream_contribution.md)):
a single new lit test for the `tt-mlir` TTIR dialect. It is a permanent copy
of the file implemented in a local scratch clone of `tenstorrent/tt-mlir`.
**Nothing here was submitted upstream: no fork, no push, no PR, no issue
(explicit user instruction; T60 is skipped). This is local work only.**

## The file

- `softmax_tests_negative.mlir`: the contribution verbatim
  (byte-identical to the scratch-clone file; `diff` empty).
- Intended upstream path:
  `test/ttmlir/Dialect/TTIR/softmax/softmax_tests_negative.mlir`
  (a new directory; lit auto-discovers `.mlir` files, no CMake change).

## What it tests

The two error branches of `ttir.softmax`'s verifier,
`SoftmaxOp::verify()` in `lib/Dialect/TTIR/IR/TTIROps.cpp` (line 5488 at the
base commit):

1. Shape mismatch between input and output, diagnostic
   `Input and output shapes must be the same`. This branch had ZERO test
   coverage anywhere in tt-mlir at the base commit (the diagnostic string
   appears only in C++ sources).
2. `dimension` outside `[-rank, rank-1]`, diagnostic
   `Dimension attribute must be within the bounds of the input tensor`.
   Previously only tripped indirectly by
   `test/ttmlir/Dialect/TTNN/softmax/softmax_negative_{1,2}.mlir`, which run
   the full `--convert-ttir-to-ttnn` conversion pipeline from the TTNN
   directory. Cases 2 and 3 (positive and negative out-of-bounds) add the
   direct dialect-level coverage where the verifier actually lives.

Convention follows the upstream template
`test/ttmlir/Dialect/TTIR/ttir_rms_norm_negative.mlir`:
`RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s`, bare
`func.func` per case (no module wrapper), `// -----` split-file separators,
CHECK lines quoting the diagnostics verbatim, no SPDX header (directory
practice).

## Provenance

- Scratch clone: `/tmp/opencode/tt-mlir-scratch` (blob-filtered clone of
  `https://github.com/tenstorrent/tt-mlir`, NOT a fork).
- Local branch: `softmax-negative-tests`, based on `main` @
  `da6f1ce081af95630f5d679392e50a4bafa7c6ea` (2026-08-03, "[d2m-jit]
  Pure-Python/torch simulator - v1 (#9120)"). Same commit as the todo 58
  scout base; the gap was re-verified still open at this HEAD before
  implementing.
- The branch exists ONLY in the scratch clone. Nothing was pushed anywhere.

## Verification evidence

- **Re-verification of the gap (2026-08-03, current main = base commit):**
  `test/ttmlir/Dialect/TTIR/softmax/` still absent from the full git tree;
  `SoftmaxOp::verify()` unchanged; no open PR addresses softmax negatives.
- **CHECK strings verbatim:** both diagnostics extracted from the re-fetched
  `lib/Dialect/TTIR/IR/TTIROps.cpp` at current main and matched against the
  CHECK lines byte-for-byte (diagnostic 1: 1 CHECK line; diagnostic 2: 2
  CHECK lines).
- **Upstream lint (the real hook suite):** `uvx pre-commit run --files
  test/ttmlir/Dialect/TTIR/softmax/softmax_tests_negative.mlir` in the
  scratch clone with the repo's own `.pre-commit-config.yaml`:
  Trim Trailing Whitespace Passed, Fix End of Files Passed, Check for added
  large files Passed; black/clang-format/check-copyright/fix-include-style
  all Skipped (no files to check for `.mlir`), exactly the effective-hook
  set predicted by the scout.
- **Syntax:** every op line matches the generic-form shape of the upstream
  precedent (softmax_negative_1.mlir's own
  `"ttir.softmax"(%arg0) <{dimension = 2 : si32}> : (tensor<...>) -> tensor<...>`);
  whole-file balance check passes (quotes, parens, braces, angle brackets
  with MLIR `->` arrows accounted for); 3 funcs, 3 returns, 3 split-file
  sections.
- **Scoped diff:** `git diff --cached` in the scratch clone shows exactly
  ONE new file, 22 insertions:
  `test/ttmlir/Dialect/TTIR/softmax/softmax_tests_negative.mlir`. Nothing
  else changed.
- Evidence log: `.omo/evidence/task-59-tensixflow.log`.

## What is NOT verified (and what it would take)

The test was NOT executed against a real `ttmlir-opt`. Building it requires
tt-mlir's private toolchain: `/opt/ttmlir-toolchain` is absent on this box
(its creation needs sudo) and the toolchain builds its own LLVM first, a
30-60 minute cost that todo 50 already documented and deferred. To run the
test for real:

```sh
# 1. one-time: toolchain dir + env toolchain (see scripts/build-tt-mlir.sh)
sudo mkdir -p /opt/ttmlir-toolchain && sudo chown -R $USER /opt/ttmlir-toolchain
cmake -B env/build env -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
cmake --build env/build
source env/activate
# 2. offline compiler build (no device needed)
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release \
  -DTTMLIR_ENABLE_RUNTIME=OFF -DCMAKE_BUILD_PARALLEL_LEVEL=4
cmake --build build --target ttmlir-opt
# 3. the targeted test
llvm-lit -v test/ttmlir/Dialect/TTIR/softmax/softmax_tests_negative.mlir
```

The CHECK strings are quoted verbatim from the verifier source, so the
expected pass/fail behavior is pinned even without the local binary.

## Future submission

Submission is deliberately NOT part of this project's scope. If the
contribution is ever submitted, the process (fork, branch, pre-commit,
targeted lit run, PR template, CI expectations) is documented in
[docs/upstream_contribution.md, "Submission process" section](../../docs/upstream_contribution.md#submission-process-documented-only-never-executed-in-this-plan).
Before any real submission, re-verify the gap against upstream `main` of
that day (the repo moves daily).
