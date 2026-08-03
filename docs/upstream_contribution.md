# Upstream Contribution Scout: a concrete small tt-mlir contribution (Todo 58)

Status: **SCOUTED + DOCUMENTED ONLY. Nothing is submitted, forked, pushed, or
merged.** Todo 58 scouts and records; todo 59 implements the change LOCALLY in
a scratch clone; any real submission is explicitly out of scope for this plan
(the submission process below is documentation, never executed here).

Scout date: 2026-08-03, against `tenstorrent/tt-mlir` `main` @
`da6f1ce081af95630f5d679392e50a4bafa7c6ea` (2026-08-03, "[d2m-jit]
Pure-Python/torch simulator - v1 (#9120)"). The tt-mlir study doc
([ttmlir_study.md](ttmlir_study.md), todo 50) used pin `9f06802f`
(2026-07-22); the gap documented below exists at BOTH pins, but any real
contribution targets CURRENT `main`, so every citation here is from
`da6f1ce0`.

## Target repo: tenstorrent/tt-mlir

Chosen over `tenstorrent/tt-metal` and `tenstorrent/tt-forge` because this
project has the deepest familiarity with it (todos 50-53: op shapes, verifier
diagnostics, lit test paths, and the exact `RUN: ttmlir-opt ... | FileCheck`
convention are all grounded in `docs/ttmlir_study.md`). A lit-test
contribution needs exactly that familiarity and nothing else: no device, no
30-60 minute toolchain build to scout, no runtime.

## The specific contribution

**Add ONE new lit test file that covers the `ttir.softmax` verifier's two
error branches at the TTIR dialect level, one of which has ZERO test coverage
anywhere in the repo.**

- New file: `test/ttmlir/Dialect/TTIR/softmax/softmax_tests_negative.mlir`
- No other file changes. No C++ change, no CMake change (lit auto-discovers
  every `.mlir` under `test/`: `test/lit.cfg.py` sets
  `config.suffixes = [".mlir"]`).

### Why this is a real gap (cited from the actual repo @ main)

1. The verifier exists with two error branches, in
   `lib/Dialect/TTIR/IR/TTIROps.cpp`, `SoftmaxOp::verify()`:
   - `return emitOpError("Input and output shapes must be the same");` (input
     and output ranks/shapes must match)
   - `return emitOpError("Dimension attribute must be within the bounds of the input tensor");`
     (`dimension >= rank` or `dimension < -rank`)
2. The op definition is `def TTIR_SoftmaxOp : TTIR_NamedOp<"softmax">` in
   `include/ttmlir/Dialect/TTIR/IR/TTIROps.td` (arguments: `AnyRankedTensor`
   input, `SI3Attr:$dimension`, `DefaultValuedAttr<BoolAttr, "false">:$numericStable`;
   `let hasVerifier = 1;`).
3. **Shape-mismatch branch: zero coverage.** A GitHub code search for the
   diagnostic string `Input and output shapes must be the same` in
   `tenstorrent/tt-mlir` returns only C++ sources
   (`lib/Dialect/TTIR/IR/TTIROps.cpp`, `lib/Dialect/TTNN/IR/TTNNOps.cpp`) and
   clamp tests (which exercise `ttir.clamp`/`ttnn.clamp`, not softmax). No
   test file anywhere asserts this diagnostic for `ttir.softmax`.
4. **Dimension-bounds branch: covered only indirectly, in the wrong layer.**
   The only tests that trip it are
   `test/ttmlir/Dialect/TTNN/softmax/softmax_negative_1.mlir` and
   `softmax_negative_2.mlir`, which run the full
   `--ttcore-register-device --ttnn-layout --convert-ttir-to-ttnn` conversion
   pipeline and live under the TTNN dialect directory even though the op and
   the diagnostic both belong to TTIR. A dialect-level negative test (the
   layer where the verifier lives) does not exist.
5. There is no `test/ttmlir/Dialect/TTIR/softmax/` directory at all (verified
   by listing the full git tree of `main` @ `da6f1ce0`), while nearly every
   other verifier-bearing TTIR op has a `*_negative.mlir` sibling
   (`matmul/`, `select/`, `sort/`, `reverse/`, `rand/`, `clamp/`, `linear/`,
   `embedding/`, ...).
6. No open or merged PR addresses this (PR search for softmax negative tests:
   only the 2024 op introduction #138 and the 2025 TTIR-to-Linalg lowering
   #2582, both merged, neither adding negative tests).

### What the test file contains (the exact shape for todo 59)

Three cases, each in its own `// -----` split-file section, following the
current convention verbatim from
`test/ttmlir/Dialect/TTIR/ttir_rms_norm_negative.mlir` (the newest negative
test of a fused-form op, last touched by "Remove dps from TTIR (#6052)",
2025-11-27):

```mlir
// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for the TTIR softmax operation.

// CHECK: error: 'ttir.softmax' op Input and output shapes must be the same
func.func @softmax_shape_mismatch(%arg0: tensor<2x4x8xf32>) -> tensor<2x4x16xf32> {
  %0 = "ttir.softmax"(%arg0) <{dimension = 1 : si32}> : (tensor<2x4x8xf32>) -> tensor<2x4x16xf32>
  return %0 : tensor<2x4x16xf32>
}

// -----
// CHECK: error: 'ttir.softmax' op Dimension attribute must be within the bounds of the input tensor
func.func @softmax_dimension_too_large(%arg0: tensor<2x4x8xf32>) -> tensor<2x4x8xf32> {
  %0 = "ttir.softmax"(%arg0) <{dimension = 3 : si32}> : (tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
  return %0 : tensor<2x4x8xf32>
}

// -----
// CHECK: error: 'ttir.softmax' op Dimension attribute must be within the bounds of the input tensor
func.func @softmax_dimension_too_negative(%arg0: tensor<2x4x8xf32>) -> tensor<2x4x8xf32> {
  %0 = "ttir.softmax"(%arg0) <{dimension = -4 : si32}> : (tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
  return %0 : tensor<2x4x8xf32>
}
```

Notes on the sketch:

- Case 1 is the genuinely NEW coverage (branch 3 above); cases 2-3 replicate
  the two out-of-bounds directions at the dialect level (positive
  `dimension = 3` equals rank; negative `dimension = -4` is below `-rank`),
  replacing the indirect conversion-pipeline trip with a direct verifier
  test. The verifier checks shape first, so the dimension cases keep
  input/output shapes equal.
- Generic op form `"ttir.softmax"(%arg0) <{dimension = ... : si32}>` is the
  exact spelling the existing TTNN softmax negatives use, so it is known to
  parse. `numericStable` defaults to false and is elided (same as the
  existing tests).
- Bare `func.func` at top level, no `module {}` wrapper: the current
  convention (`ttir_rms_norm_negative.mlir`, post-DPS-removal).
- No SPDX header: every existing `Dialect/TTIR/*_negative.mlir` file starts
  straight with `// RUN:` (matmul, rms_norm, and the TTNN softmax negatives
  all lack headers), so the new file matches its directory's practice.
- Single file, ~25 lines. This is the whole change.

## CONTRIBUTING guidelines + code style + CI (the required links)

- Contributing guidelines:
  <https://github.com/tenstorrent/tt-mlir/blob/main/CONTRIBUTING.md>
  Key points: PRs get an initial review for CI/CD readiness, after which a
  Tenstorrent developer kicks off the CI pipeline on the contributor's
  behalf; landing needs at least 1 reviewer sign-off, component-owner
  sign-off (CODEOWNERS), green CI, and a 24-hour minimum window.
- Coding guidelines and standards (linked from CONTRIBUTING.md):
  <https://github.com/tenstorrent/tt-mlir/blob/main/docs/src/coding-guidelines.md>
  (naming, include style, denesting; C++ is LLVM style per `.clang-format`
  and `.clang-tidy`; this contribution is .mlir-only, so no C++ formatting
  applies).
- Agent/build-lint-test command sheet:
  <https://github.com/tenstorrent/tt-mlir/blob/main/AGENTS.md>
  Lint = `pre-commit run --all-files`; compiler tests =
  `cmake --build build --target check-ttmlir`; a single MLIR test =
  `llvm-lit test/ttmlir/path/to/test.mlir`.
- Pre-commit config:
  <https://github.com/tenstorrent/tt-mlir/blob/main/.pre-commit-config.yaml>
  Hooks: black (python3), clang-format v18.1.7 (c/c++ only), check-copyright
  (SPDX, config `.github/check-spdx.yaml`), trailing-whitespace,
  end-of-file-fixer, check-added-large-files, and a local
  `fix-include-style` hook (C++ only). For a single new `.mlir` file the
  effective hooks are trailing-whitespace and end-of-file-fixer.
- PR template:
  <https://github.com/tenstorrent/tt-mlir/blob/main/.github/pull_request_template.md>
  Sections: Ticket (link to GitHub issue), Problem description, What's
  changed, Checklist (`New/Existing tests provide coverage for changes`).
- CI entry point:
  <https://github.com/tenstorrent/tt-mlir/blob/main/.github/workflows/on-pr.yml>
  Runs on PR open/synchronize/reopen: pre-commit job, docker build image,
  prepare-run, release/debug builds, lint, then the test matrix.
- Lit mechanics: `test/lit.cfg.py` (suite name `TTMLIR`,
  `config.suffixes = [".mlir"]`, ShTest format) is why the new file needs no
  CMake registration.

## Rationale: why this is small, newcomer-suitable, and not controversial

- **Small:** one new ~25-line file; zero C++, zero CMake, zero behavioral
  change. It only pins diagnostics the verifier already emits.
- **Newcomer-suitable:** the pattern is copy-from-neighbor
  (`ttir_rms_norm_negative.mlir` is a live template), the op semantics are
  exactly what `docs/ttmlir_study.md` grounded (TTIR op shapes + verifier
  diagnostic format), and verification is one lit invocation. No device, no
  runtime build.
- **Real value:** it turns an untested verifier branch (shape mismatch) into
  covered behavior and moves the dimension-bounds coverage from an indirect
  conversion-pipeline trip to the dialect layer where the verifier lives.
  The PR template's checklist item ("tests provide coverage for changes") is
  precisely about this kind of gap.
- **Not large/controversial:** it changes nothing observable, touches no
  pass pipeline, no op definition, no lowering. Worst case a maintainer asks
  for a rename or directory move. There is no open PR racing for it.

## Implementation sketch for todo 59 (local implementation only)

Todo 59 implements this in a SCRATCH clone; nothing is pushed. Steps:

1. Fresh clone of `tenstorrent/tt-mlir`, `git checkout -b <branch>` from
   current `main` (re-verify the gap at implementation time: the repo moves
   daily; if a softmax negative test lands upstream first, fall back to a
   candidate from the list below).
2. Create `test/ttmlir/Dialect/TTIR/softmax/softmax_tests_negative.mlir`
   with the three cases above.
3. Verify locally without the heavy toolchain build if ttmlir-opt is absent:
   the test's CHECK text is quoted VERBATIM from the verifier source, so the
   expected strings are known without running. If a ttmlir-opt binary is
   available (todo 50's `scripts/build-tt-mlir.sh`), run
   `llvm-lit test/ttmlir/Dialect/TTIR/softmax/softmax_tests_negative.mlir`
   (expect PASS) and optionally
   `cmake --build build --target check-ttmlir` for the full suite.
4. Upstream test convention followed: lit + FileCheck with
   `RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s`,
   `// CHECK: error: 'ttir.softmax' op <diagnostic>`, and `// -----`
   section separators. Diagnostics are quoted verbatim from
   `SoftmaxOp::verify()` so the CHECK lines cannot drift.
5. Run `pre-commit run --all-files` on the changed file (trailing-whitespace,
   end-of-file-fixer are the effective hooks for .mlir).

## Submission process (DOCUMENTED ONLY; never executed in this plan)

If a future change ever turns this into a real PR, the process per the linked
guidelines is:

1. **Fork + branch:** fork `tenstorrent/tt-mlir` (e.g. `gh repo fork`),
   branch from current `main` with a descriptive name (e.g.
   `softmax-negative-tests`); keep the change to the single test file.
2. **Style/lint:** install `pre-commit` and run
   `pre-commit run --all-files` locally; the config pulls black,
   clang-format v18, the SPDX check-copyright hook, trailing-whitespace,
   end-of-file-fixer, and fix-include-style. For .mlir-only, only the
   whitespace/EOF hooks fire.
3. **Tests:** `llvm-lit test/ttmlir/Dialect/TTIR/softmax/softmax_tests_negative.mlir`
   must pass; optionally the full `check-ttmlir` target if a local build
   exists.
4. **PR:** open against `main`, fill the PR template (link a ticket/issue if
   one exists, problem description quoting the untested verifier branch,
   what's changed = the new test file + the coverage it adds, tick the tests
   checklist box).
5. **CI:** the `on-pr.yml` pipeline (pre-commit, builds, lint, tests) runs
   after a Tenstorrent developer approves CI/CD readiness and triggers it
   (CONTRIBUTING.md's community-contributions policy); landing additionally
   needs reviewer + component-owner sign-off and the 24-hour window.

This plan explicitly stops before step 1: no fork, no branch on the upstream
repo, no PR, no issue (user instruction for this wave; the plan's D13
deliverable is adjusted accordingly: scouting + local implementation +
submission documentation, with submission itself skipped).

## Candidates considered (backups)

Scouted and rejected in favor of the softmax test, listed per the QA failure
scenario in case the primary target is superseded upstream before todo 59:

1. **`ttir.sampling` negative tests**: `SamplingOp::verify()`
   (`lib/Dialect/TTIR/IR/TTIROps.cpp`) has five distinct error branches and
   only `test/ttmlir/Dialect/TTIR/reduction/sampling_tests_positive.mlir`
   exists. Real gap, but ~5 cases = more than twice the scope; backup only.
2. **`ttir.get_dimension_size` silent failure**: its `verify()` returns a
   bare `failure()` with no diagnostic message. Real defect, but fixing it is
   a C++ change (small, yet higher review weight than a pure test add).
3. **Documentation pass over `docs/src/`**: no concrete, cited doc error was
   found fast enough to beat the verified test gap above; a doc fix remains
   the fallback-of-last-resort if the test candidates are all superseded.

## Honesty

- NOTHING here is submitted, merged, or even forked. This is scouting.
- The gap is verified against the real repo at main @ `da6f1ce0` via GitHub
  API tree listings, verbatim source fetches of `TTIROps.cpp` / `TTIROps.td`
  / the lit config / the existing negative tests, and a code search over the
  diagnostic strings; it is not inferred or fabricated.
- tt-mlir is a fast-moving repo (pushed the same day as this scout). Todo 59
  MUST re-verify the gap at implementation time before relying on it.
