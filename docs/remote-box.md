# Optional remote build box

> **You do not need this.** TensixFlow's MVP is **CPU-only and local-first**.
> The compiler, the lit/`ctest`/`pytest` suites, and the whole build run on an
> ordinary laptop with no GPU and no Tenstorrent device. A remote box is purely
> **opt-in** — use it only if you want a separate, always-on Linux build machine
> so your laptop stays free. Nothing in CI or the test suite provisions or
> contacts a cloud provider, and nothing here incurs spend automatically.

If you *do* want a remote Linux box, [`scripts/remote-box.sh`](../scripts/remote-box.sh)
is an idempotent setup script you can run on a machine you already have shell
access to. It installs Nix (if absent), clones the repo, enters the dev shell
(`devenv shell` or `nix develop`), and runs the standard configure + build:

```bash
cmake -G Ninja -B build -DMLIR_DIR=$MLIR_DIR -DLLVM_DIR=$LLVM_DIR && cmake --build build
```

Re-running it is safe: each step is guarded, so completed steps are skipped.

## Choosing a provider

For a cheap, opt-in Linux sandbox, **PrimeIntellect** is currently the lowest-cost
option we found, with **RunPod** a near-tie. Indicative pricing (GPU not required —
a plain CPU instance is enough for the MVP build):

- **PrimeIntellect** — RTX A4000 GPU at roughly **$0.15/hr**, or a **CPU Sandbox**
  at about **$0.08/hr** (the CPU Sandbox is the sensible pick for a build-only box).
- **RunPod** — RTX A5000 at roughly **$0.16/hr** on the Community tier, a close
  alternative if PrimeIntellect capacity is unavailable.

> **Prices as-of 2026-07-21, verify current in-console.** Cloud GPU/CPU pricing and
> availability change frequently; treat the numbers above as a rough ordering only
> and confirm the live rate in each provider's console before you commit spend.

Because the MVP build is CPU-only, prefer the cheapest **CPU** instance a provider
offers; renting a GPU buys you nothing for the current compiler work.

## Workflow

1. Rent a Linux sandbox (Ubuntu recommended) from your chosen provider.
2. SSH in as a non-root user with `sudo`.
3. Run the setup script:

   ```bash
   git clone https://github.com/tensixflow/tensixflow.git
   cd tensixflow
   bash scripts/remote-box.sh
   ```

   (Or pipe the script straight into `bash` on a fresh box.)
4. Build artifacts land under `build/bin/`.

Override the defaults with environment variables if you need a different checkout
location or repo URL:

```bash
TENSIXFLOW_DIR=$HOME/work/tensixflow bash scripts/remote-box.sh
```
