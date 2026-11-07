# tt-flow

A compiler and device-cost planner for fused transformer kernels on Tenstorrent hardware.
It reads MLIR, fuses RMSNorm+MatMul and MLP chains, plans the core grid and circular buffers
against published device limits, and emits either TTIR text or a TT-Metalium project that
builds and runs against a pinned upstream release.

The point is to answer two questions before you write a kernel: does the fused op fit on the
card, and what does the schedule cost in bytes and projected cycles. With no card in the
machine, the emitted program still runs on Tenstorrent's official simulator.

![tt-flow pipeline](docs/figures/pipeline.png)

## What it does

The compiler takes MLIR written in a small tensor dialect, runs fusion and tile-layout
inference, plans a core grid and circular-buffer allocation for a named device, and reports a
projected roofline. It can stop at a plan, emit TTIR text for tt-mlir, or emit a TT-Metalium
project with a host program and three kernels.

Build it first:

```sh
devenv shell -- bash scripts/build.sh
```

Plan the fused RMSNorm+MatMul example on a Wormhole n150:

```sh
devenv shell -- build/bin/tensixflow-opt \
  --tensixflow-device=n150 \
  --tensixflow-infer-tile-layout \
  --tensixflow-fuse-rmsnorm-matmul \
  --tensixflow-plan-noc-sharding \
  examples/rmsnorm_matmul.mlir
```

The pass prints the device it planned against, the grid, weight strategy, circular-buffer
footprint, and projected traffic for the unfused and fused paths. The default device is n150;
n300, p100a, and p150a are accepted too. Every figure in the output comes from the published
device data in `spec/devices.json`.

## Devices

| Device | Architecture | Compute grid | L1 per core | DRAM |
|---|---|---|---|---|
| n150 | Wormhole B0 | 8x8 | 1,499,136 B | 12 GiB at 288 GB/s |
| n300 | 2 x Wormhole B0 | 8x7 per chip | 1,499,136 B per chip | 24 GiB at 576 GB/s |
| p100a | Blackhole | 12x10 | 1,572,864 B | 28 GiB at 448 GB/s |
| p150a | Blackhole | 12x10 | 1,572,864 B | 32 GiB at 512 GB/s |

Tenstorrent's product pages list 120 Tensix cores for the Blackhole cards while the die has 140
workers and the unharvested descriptor has a 13x10 grid. The repository records all three
numbers in the device data and models the 120-core product figure. Cost figures are projected,
never measured.

![per-core budget](docs/figures/device_budget.png)

## Emit and run a project

Emission targets a pure matmul schedule. A fused graph is planned and reported, but the
emitter exits with a message rather than shipping a kernel it cannot check. The emitted
project builds against the official tt-metal release recorded in `deps/tt-metal.lock`:

```sh
devenv shell -- bash scripts/fetch_tt_metal.sh      # pinned headers, runtime, and ttsim
devenv shell -- bash scripts/build_generated.sh     # compile and link against real headers
devenv shell -- bash scripts/run_generated.sh       # ttsim by default
devenv shell -- bash scripts/run_generated.sh --device
```

The build script is a real gate: it configures with `find_package(TT-Metalium)` from the
extracted release package, then compiles and links the emitted host program against
`libtt_metal.so`. The run script reports the program's own PCC against its CPU reference. On
this machine, with no card, the emitted 256x512x1024 bf16 matmul ran under ttsim and returned:

```json
{"status":"pass","backend":"ttsim","pcc":0.9890072}
```

The last command needs a card. The device path is written against the same pinned release and
has not been executed on silicon.

## What runs where

![verification map](docs/figures/verification.png)

`devenv shell -- bash scripts/test.sh` checks the device data against its sources, runs the lit
and GoogleTest suites, runs the Python reference checks and the real TTIR parser gate, enforces
the prose rules, and compiles an emitted project against the pinned release when `.deps` is
present. The simulator run is the one execution check available without a card; device
execution and measured performance stay untested.

## Status

The emitted matmul program is compiled, linked, and executed on the pinned simulator on this
machine. The upstream matmul example from the same release also ran under the simulator, at
PCC 0.9791022. No kernel from this project has run on Tenstorrent hardware.

Current limits, stated plainly:

- Emission is one core. The plan's grid is analysis and report only; the generated program does
  not yet realize it.
- Fused RMSNorm and MLP kernels are planned and reported, not emitted.
- Cost figures are a roofline projection. Nothing in this repository measures device time.
- The reference model checks the compiler's schedule semantics against a NumPy golden. It is not
  a model of the device.
