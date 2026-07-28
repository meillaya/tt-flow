# TensixFlow Performance Model

The performance model is a first-order **static analytic** roofline. It computes
cycles, DRAM/L1/NoC traffic, circular-buffer occupancy, compute utilization, and a
bound classification from the plan's tensor shapes and a shared set of documented
constants. There is no measurement here. Every figure is an estimate, and none is
ever reported as hardware-observed. The model shares its L1 budget verbatim with the
scheduler, so the circular-buffer occupancy it reports is the same budget the
allocator actually planned against.

The model lives in `lib/tensixflow/Analysis/PerfModel.h`; the L1 constants live in
`lib/tensixflow/Scheduler/L1Model.h`.

## L1-budget model

The scheduler and the performance model share one header
(`lib/tensixflow/Scheduler/L1Model.h`) as the single source of truth for the modeled
per-core L1 budget:

| Constant | Value | Meaning |
|---|---|---|
| `kL1TotalBytes` | 1048576 (1 MiB) | modeled per-core L1 |
| `kL1ReserveBytes` | 65536 (64 KiB) | code / stack reserve |
| `kCBShardBudget` | 983040 (960 KiB) | `kL1TotalBytes - kL1ReserveBytes`, the circular-buffer + shard budget |
| `kTileBytes` | 2048 | one 32x32 bf16 tile (32 x 32 x 2 bytes) |

**This 1 MiB figure is a documented, conservative, modeled budget, and it is a tunable
constant.** The literal Wormhole n150 L1 per core is larger, about 1.5 MB. TensixFlow
models 1 MiB so the planner stays on the safe side of the real budget; nothing else in
the stack depends on the exact silicon number. Changing the constant (and re-checking
the planner and model) is the supported way to target a different budget.

### Circular-buffer arithmetic

Circular-buffer bytes follow `CB_bytes = n_CB x depth x 2048`. The maximum feasible
depth for a given buffer count is

```
maxCBDepth(n_CB) = floor(kCBShardBudget / (n_CB x kTileBytes))
```

with guards: a non-positive count yields 0, and a request whose single-tile footprint
exceeds the budget yields 0 (cannot fit one tile per buffer). The allocator picks that
maximum depth, so `CB_bytes <= kCBShardBudget` always holds; an oversized request gets
a reduced depth, never an overflow. For the default three-buffer pipeline:

```
depth     = floor(983040 / (3 x 2048)) = 160
CB_bytes  = 3 x 160 x 2048            = 983040   (== kCBShardBudget)
occupancy = CB_bytes / kCBShardBudget = 1.0
```

The three buffers take the tt-metal convention indices `c_0` (in0), `c_1` (in1), and
`c_16` (out).

## Static performance formulas

The cycle model is a roofline over two parallel paths. The reader/compute/writer
pipeline overlaps them, so the slower path dominates.

```
computeCycles = (2 * M * N * K) / totalCores / kFlopsPerCyclePerCore
memoryCycles  = (fused DRAM read + write + multicast bytes) / totalCores / kBytesPerCyclePerCore
estCycles     = max(computeCycles, memoryCycles)
computeUtil   = computeCycles / estCycles          # in [0, 1]
```

- `M` is the product of the activation's leading (batch) dims, `K` the activation's
  last dim, `N` the weight's last dim. The matmul dominates compute; rmsnorm and gelu
  are O(M x K) elementwise and documented as negligible next to the O(M x N x K)
  matmul.
- `totalCores = gridY x gridX` (64 for the 8x8 grid).
- Memory traffic counts the off-chip path only: DRAM reads, DRAM writes, and NoC
  multicast bytes. L1 traffic is on-chip and not on the critical path.

The cost constants are documented static estimates, not measured hardware rates, and
are tunable:

| Constant | Value | Meaning |
|---|---|---|
| `kFlopsPerCyclePerCore` | 2048 | one 32x32 tile outer-product accumulate per cycle (32 x 32 x 2 FLOPs) |
| `kBytesPerCyclePerCore` | 16 | modeled aggregate DRAM + NoC bandwidth per core |
| `kComputeBoundRatio` | 1.25 | above this compute/memory ratio: compute-bound |
| `kMemoryBoundRatio` | 0.8 | below this ratio: memory-bound |
| `kSplitKThreshold` | 2048 | contracting dim at/above which split-K is suggested |

### Bound classification and suggestions

The bottleneck comes from the ratio `r = computeCycles / memoryCycles`:

```
r > 1.25   ->  "compute"
r < 0.8    ->  "memory"
otherwise  ->  "balanced"
```

`computeUtil` is 1.0 when compute-bound and below 1.0 when memory-bound. Suggestions
are derived from the plan's actual parameters: a bound-derived suggestion always fires
(spread N across more X-cores when compute-bound; deepen circular buffers or prefetch
activations when memory-bound; pipeline the three stages when balanced), plus
conditional ones (increase circular-buffer depth when below the feasible maximum, use
column multicast for a large replicated weight, split K across core groups when
`K >= 2048`). A degenerate plan yields a valid report: zero cycles, bottleneck
"balanced", and a single shape-check suggestion. Never a crash, NaN, or empty
classification.

### The fusion invariant

The central invariant the model reproduces is that **fusion reduces DRAM traffic**:

```
unfused.dramReadBytes  >  fused.dramReadBytes
```

because the fused path keeps the rmsnorm-output intermediate resident in L1 instead of
round-tripping it through DRAM. For the running `rmsnorm_matmul` example (activation
`[1, 2048, 4096]` bf16, weight `[4096, 12288]` bf16, multicast):

```
unfused DRAM reads  = 134217728      fused DRAM reads = 117440512
unfused DRAM writes =  67108864      fused DRAM writes = 50331648
saved intermediate  =  16777216      (= unfused reads - fused reads)
```

The same figures drive the report's "kernel launches: unfused 2 -> fused 1" line. For
this example the cycle model gives `computeCycles = 1572864`, `memoryCycles = 262144`,
ratio 6.0, so the classification is compute-bound with `computeUtil = 1.0`. A small
replicated weight (no multicast) flips the classification to memory-bound, which the
model's unit test asserts as a distinct bottleneck.

## Rounding contract

The correctness bar is meaningful only because the simulator and the golden run at
deliberately different precision. This is the D14 contract, shared by the C++
simulator (`lib/tensixflow/Sim/Rounding.h`), the NumPy golden
(`python/tensixflow/golden.py`), and a bit-for-bit identical Python rounding twin
(`python/tensixflow/rounding.py`).

**Simulator path (bf16 spatial hardware model):**

- inputs are rounded to bf16 (`to_bf16`, round-to-nearest-even);
- matmul / rmsnorm / gelu accumulate in fp32 **within** a tile;
- each output tile is rounded back to bf16;
- fused circular-buffer intermediates between stages are stored as bf16, i.e. rounded
  between fused stages.

**Golden path (fp32-exact reference):**

- fp32 accumulate, fp32 intermediates, and **no** bf16 rounding of intermediates;
- round only the final compared output, if at all.

The bf16 rounding itself is the high 16 bits of an IEEE-754 fp32 with the low 16 bits
rounded to nearest even (the `+lsb` bias makes exact ties land on the even candidate),
matching tt-metal `DataType::BFloat16`. NaN inputs are canonicalized to a
sign-preserving quiet NaN before the bias add.

The bar is

```
PCC(simulator_bf16_tiled, golden_fp32_exact) >= 0.9875
```

measuring the real bf16-spatial-versus-exact cost of fusion plus tiled execution. The
0.9875 floor comes from the fused-LLM-on-Wormhole context the project models.

### Why the bar is not vacuous

If the golden also rounded its intermediates to bf16, the two paths would be identical
and PCC would collapse to 1.0, making any floor pass for nothing. A layered defense
keeps the bar honest:

1. **Primary, structural:** an assertion that the golden holds its fused intermediates
   in fp32 (not bf16-representable via `to_bf16`) while the simulator output is
   bf16-representable, plus the golden's fp32 dtype.
2. **Primary, bit-difference:** `max_abs_diff(simulator, golden) > 0`, i.e. PCC is
   strictly below 1.0. This deterministically catches the both-bf16-identical case.
3. **Secondary, empirical trip-wire:** an upper band set strictly above the measured
   good-case PCC (across the documented seeds) and strictly below 1.0, recorded in the
   metrics JSON. It is calibrated from the running implementation, never hard-coded,
   so a correct single-op case whose good-case PCC exceeds 0.9999 still passes.

The lower floor 0.9875 is fixed. It is never lowered, and divergence is never injected
to land inside a band.
