# Performance

The cost model projects cycles, DRAM traffic, and NoC traffic from a plan and a device. It is a static roofline over published device limits and the plan's tensor shapes. It times nothing, and every figure it renders carries the device name and the word projected.

## Inputs

The model reads the plan's device block, the activation and weight shapes, the circular-buffer specs, and the weight strategy. It resolves the full device spec by name: clock, NoC rate, and per-core budget come from `spec/devices.json` through the generated table. The activation is `[..., M, K]` and the weight is `[K, N]`, so `M` is the product of the activation leading dimensions and `N` is the weight's last dimension.

## Cycle model

```
cores         = chips * compute_grid_x * compute_grid_y
computeCycles = 2 * M * N * K / (cores * flops_per_cycle_per_core)
dramCycles    = (dram_read_bytes + dram_write_bytes) / (dram_bandwidth / clock_hz)
nocReadBytes  = (dram_read_bytes + multicast_bytes) / cores
nocWriteBytes = dram_write_bytes / cores
nocCycles     = max(nocReadBytes, nocWriteBytes) / noc_bytes_per_cycle_per_direction
estCycles     = max(computeCycles, dramCycles, nocCycles)
computeUtil   = computeCycles / estCycles
```

`dram_read_bytes`, `dram_write_bytes`, and `multicast_bytes` are the plan's NoC estimate for the fused path. `cores` is 64 for n150, 112 for n300, and 120 for p100a and p150a.

The model uses one NoC's per-direction rate for `nocCycles` even though each device has two NoCs. That gives a conservative bound. The two conditions that mark a plan balanced are in [binding constraint](#binding-constraint).

## Constants

| Constant | Value | Basis |
|---|---|---|
| Tile bytes | 2,048 | 32x32 bf16 tile |
| FLOPs per cycle per core | 4,096 | Matrix unit figure for all four devices, one fidelity phase |
| NoC bytes per cycle per direction | 32 | One NoC, 256-bit flit per cycle into L1 |
| L1 planner reserve | 65,536 B | Project planning choice for code, stack, and runtime bookkeeping, not a vendor figure |
| Default circular-buffer depth | 2 tiles per buffer | Clamped to `floor(cb_budget / (num_buffers * 2,048))` |
| Balanced band | 0.9 | A model band, not a hardware fact |
| Split-K threshold | K >= 2,048 | Suggestion trigger |
| Default element width | 2 bytes | Used only for an unrecognized dtype |

The per-device clock and DRAM bandwidth are in the [hardware table](hardware.md#memory). The circular-buffer budget is `l1_usable_bytes - 65,536`. For n150 the budget is 1,433,600 B, the deepest three-buffer allocation is 233 tiles per buffer, and the default allocation is three buffers at two tiles each, or 12,288 B.

## Binding constraint

The model names the largest projected term as the binding constraint: `compute`, `dram`, or `noc`. When the second largest term is at least 0.9 of the largest, the model reports `balanced` instead. `compute_util` is `computeCycles / estCycles`, so it is 1.0 for a compute-bound plan and below 1.0 otherwise.

The model adds suggestions from the plan's own parameters:

- Deepen circular buffers when the planned depth is below the device's feasible maximum for that buffer count.
- Use column multicast for the weight when the weight is larger than half the CB budget and the plan does not already multicast.
- Split K across two core groups when K is at least 2,048 and the grid has at least two cores.
- The binding constraint's own suggestion: spread N for compute, deepen buffers or prefetch for DRAM, multicast weight tiles for NoC, or pipeline the three stages for balanced.

## Fusion traffic

The NoC estimator counts the unfused and fused DRAM paths from the tensor shapes.

```
unfused read  = activation + intermediate + weight
unfused write = intermediate + output
fused read    = activation + weight
fused write   = output
fused L1      = intermediate
```

For the running `rmsnorm_matmul` example, with a bf16 activation `[1, 2048, 4096]`, weight `[4096, 12288]`, and output `[1, 2048, 12288]`:

| Figure | Bytes | Derivation |
|---|---|---|
| Activation | 16,777,216 | 16 MiB |
| Intermediate | 16,777,216 | Same shape as the activation |
| Weight | 100,663,296 | 96 MiB |
| Output | 50,331,648 | 48 MiB |
| Unfused DRAM reads | 134,217,728 | Activation + intermediate + weight |
| Unfused DRAM writes | 67,108,864 | Intermediate + output |
| Fused DRAM reads | 117,440,512 | Activation + weight |
| Fused DRAM writes | 50,331,648 | Output |
| DRAM reads saved | 16,777,216 | The intermediate never re-read |
| DRAM writes saved | 16,777,216 | The intermediate never written |

Those figures are byte arithmetic from the shapes and the bf16 element width. Fusion moves the intermediate from DRAM to L1 and leaves the arithmetic checkable.

## Projected example

The model projects `examples/rmsnorm_matmul.mlir` on n150 at the following terms. The plan is the fused RMSNorm plus matmul schedule, the activation is `[1, 2048, 4096]`, and the weight is `[4096, 12288]`, both bf16.

| Projected term | Cycles |
|---|---|
| Compute | 786,432 |
| DRAM | 582,542 |
| NoC | 106,496 |
| Estimate, the maximum of the three | 786,432 |
| Compute utilization | 1.0 |

The binding constraint is compute: the DRAM term is 74% of the compute term, below the 0.9 balanced band. The three-buffer allocation of 12,288 B occupies 0.86% of the 1,433,600 B n150 CB budget, so the model suggests deepening the buffers from 2 to 4 tiles. It also suggests splitting K, which is 4,096, and spreading N across more X-cores because the plan is compute-bound.

## What the model leaves out

- RMSNorm and GELU arithmetic. The compute term counts the matmul FLOPs only; the elementwise stages are O(M x K) against O(M x N x K).
- L1 bandwidth and the time a tile spends in a circular buffer.
- DRAM row activation, bus efficiency, and refresh.
- NoC latency, arbitration, and congestion between the two NoCs.
- Dispatch, kernel launch, host-side work, and PCIe transfer.
- Fidelity phases beyond the single modeled phase.
- Any calibration against a card or against ttsim.

The consequence: the projected terms rank two plans against each other in the model's own units. They do not predict a measured runtime.

## Where the figures appear

`tensixflow-report` runs the model, fills the plan's `perf` block (`est_cycles`, `compute_util`, `bottleneck`, `suggestions`), and renders the HTML report. `formatPerfReport` prints one line per term, each prefixed with `projected <device>`. The plan's `device` block names the device the projection used; [schema.md](schema.md) defines the JSON field names.
