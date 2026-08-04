# TensixFlow report — build guide & content-worker contract

This document is the single source of truth for (1) the LaTeX/TikZ toolchain,
(2) the exact build commands, and (3) the labeling / package contract the three
content workers (lanes WA, WB, WC) and the integrator must follow.

Everything below is **nix-managed and reproducible**. No `flake.nix` /
`devenv.nix` mutation is required or permitted; the report builds in an
ephemeral `nix shell`.

---

## 1. Toolchain (exact, resolved)

The project dev shell (`nix develop`) ships **no LaTeX at all** (`pdflatex`,
`lualatex`, `xelatex`, `kpsewhich` all absent). Therefore the report is built
inside an ephemeral TeX Live environment provided by `nix shell`.

### 1.1 Invocation

```sh
nix shell nixpkgs#texlive.combined.scheme-full --command bash -c '<build commands>'
```

`nixpkgs` here resolves through the flake registry to the same pinned input the
dev environment uses (`flake.lock` → `NixOS/nixpkgs` rev `bd3c09ca2a2858daa15f26eb557faa4709635a41`,
branch `nixos-unstable`). Because the flake pins the registry input, the
ephemeral `nix shell nixpkgs#...` resolves against that pinned rev — this is
intentional and reproducible. (Documented fallback, **not** needed here:
`nix shell github:NixOS/nixpkgs/nixos-unstable#texlive.combined.scheme-full`.)

### 1.2 Scheme selection (scheme-medium → scheme-full)

`scheme-medium` was tried first and found **insufficient**. Package probe inside
`scheme-medium`:

| Package      | scheme-medium | scheme-full |
|--------------|:-------------:|:-----------:|
| tikz.sty     | ok            | ok          |
| pgfplots.sty | **MISSING**   | ok          |
| booktabs.sty | ok            | ok          |
| listings.sty | ok            | ok          |
| enumitem.sty | **MISSING**   | ok          |
| tcolorbox.sty| **MISSING**   | ok          |
| longtable.sty| ok            | ok          |
| fancyhdr.sty | ok            | ok          |
| hyperref.sty | ok            | ok          |
| xcolor.sty   | ok            | ok          |
| caption.sty  | ok            | ok          |

Because `pgfplots`, `enumitem`, and `tcolorbox` are missing from
`scheme-medium`, the toolchain was **escalated to `scheme-full`**, which
provides every package the preamble uses. **Use `scheme-full` for all builds.**

### 1.3 Resolved store paths (recorded verbatim)

- `pdflatex` binary:
  `/nix/store/h8pcc9fhs0vvm4g3n0gf3j5fv04glilm-texlive-combined-full-2025-final/bin/pdflatex`
- Texlive distribution tree (texmf, where `.sty` files live):
  `/nix/store/arhp5gpazidag3p1hpnvhbabis0bd7bd-texlive-combined-full-2025-final-texmfdist`
- Engine: `pdfTeX 3.141592653-2.6-1.40.27 (TeX Live 2025/nixos.org)`
- Spot check: `kpsewhich tikz.sty` →
  `.../texlive-combined-full-2025-final-texmfdist/tex/latex/pgf/frontendlayer/tikz.sty` (non-empty ✔)

---

## 2. Build commands (exact sequence)

Build is run **from the `report/` directory**, writing artifacts into `report/`
(`-output-directory=.`). The stub uses jobname `tensixflow_report_stub`. Run
`pdflatex` **twice** so the table of contents and cross-references settle.

```sh
cd report

# Clean any stale LaTeX state first (idempotent, guarded by report/.gitignore)
rm -f *.aux *.log *.out *.toc *.lof *.lot *.fls *.fdb_latexmk

# Pass 1 and Pass 2 (jobname=tensixflow_report_stub, output stays in report/)
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=. \
    -jobname=tensixflow_report_stub main.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=. \
    -jobname=tensixflow_report_stub main.tex
```

Or, without entering a shell first:

```sh
nix shell nixpkgs#texlive.combined.scheme-full --command bash -c '
  cd report
  rm -f *.aux *.log *.out *.toc *.lof *.lot *.fls *.fdb_latexmk
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory=. -jobname=tensixflow_report_stub main.tex
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory=. -jobname=tensixflow_report_stub main.tex
'
```

**Success criteria (all must hold):**
- `pdflatex` exit code `0` on both passes.
- `grep -E '^!' report/main.log` (and the stub `.log`) prints nothing.
- `grep -E 'Emergency stop|Fatal error' <log>` prints nothing.
- Stub PDF `report/tensixflow_report_stub.pdf` exists with **>= 3 pages**
  (current: 10 pages, verified with `pdfinfo`).

### 2.1 Artifact hygiene / gitignore

Build artifacts (`*.aux`, `*.log`, `*.out`, `*.toc`, `*.lof`, `*.lot`, `*.fls`,
`*.fdb_latexmk`) are gitignored via `report/.gitignore`. The stub PDF
`tensixflow_report_stub.pdf` is also gitignored. **The final
`tensixflow_report.pdf` is deliberately NOT gitignored** — it is a committed
deliverable.

---

## 3. Labeling contract (content workers)

All `\label` / `\ref` identifiers must follow these exact namespaces. No
exceptions; the integrator will reject non-conforming labels.

- **Sections:** `sec:<key>` — one per section file. Existing keys:
  `sec:theory`, `sec:decisions`, `sec:policy`, `sec:orchestration`,
  `sec:hardware`, `sec:software`, `sec:deployment`, `sec:appendix`.
  (The abstract is intentionally unlabeled.)
- **Figures:** `fig:<key>-<desc>` — `<key>` is the owning section key,
  `<desc>` is a short kebab-case description,
  e.g. `fig:orchestration-datapath`, `fig:hardware-tensix-core`.
- **Tables:** `tab:<key>-<desc>` — same pattern,
  e.g. `tab:deployment-envmatrix`.
- **Citations:** **citation-free by default.** No BibTeX / `.bib`. Use inline
  `\footnote{...}` or hyperlinked URLs (`\url{...}` / `\href{url}{text}`) for
  references instead.

---

## 4. TikZ / package contract

- All `\usepackage` and `\usetikzlibrary` calls live **only in `main.tex`**.
- **Content workers must NOT add `\usetikzlibrary{...}` inside their section
  files.** Doing so risks duplicate-load or conflict errors. If a new library
  is needed, note it here (below) and the integrator will add it to `main.tex`.
- TikZ figures should use only the already-loaded libraries.

### 4.1 Libraries the preamble already loads

`shapes.geometric`, `arrows.meta`, `positioning`, `calc`, `fit`, `backgrounds`,
`decorations.pathmorphing`, `decorations.pathreplacing`, `matrix`.

> Note: a `decorations.pathrepeating` library does **not** exist in PGF/TikZ
> (the build fails fatally if requested). The intended path-decoration
> libraries are `decorations.pathmorphing` and `decorations.pathreplacing`;
> both are loaded. If you need a "repeat along path" effect, use the
> `decorations.markings` library — request it via the table below.

### 4.2 Requested TikZ library additions (integrator adds to `main.tex`)

| Requested library | Requested by (lane) | Status |
|-------------------|---------------------|--------|
| _(none yet)_      | —                   | —      |

---

## 5. File map (current skeleton)

| File | Role | Owner |
|------|------|-------|
| `report/main.tex` | master doc, preamble, `\input` order | W0 / integrator |
| `report/sections/00_abstract.tex` | abstract (unlabeled) | lane WA |
| `report/sections/01_theory.tex` | `sec:theory` | lane WA |
| `report/sections/02_decisions.tex` | `sec:decisions` | lane WA |
| `report/sections/03_policy.tex` | `sec:policy` | lane WB |
| `report/sections/04_orchestration.tex` | `sec:orchestration` | lane WB |
| `report/sections/05_hardware.tex` | `sec:hardware` | lane WB |
| `report/sections/06_software.tex` | `sec:software` | lane WC |
| `report/sections/07_deployment.tex` | `sec:deployment` | lane WC |
| `report/sections/08_appendix.tex` | `sec:appendix` | lane WC |
| `report/.gitignore` | artifact hygiene | W0 |
| `report/BUILDING.md` | this file | W0 / integrator |

Stub sections currently contain `\section{...}\label{...}` + one placeholder
sentence `\textit{Content pending --- lane W?.}` and are expected to be
replaced by the content workers.
