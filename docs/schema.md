# Plan schema

Every stage shares one plan document. The C++ structs live in `include/tensixflow/PlanSchema.h`, serialization in `lib/tensixflow/Schema/PlanSchema.cpp`, parsing in `PlanSchemaParse.cpp`, and the machine validation contract in `schema/plan.schema.json` (JSON Schema draft-07). `tensixflow-translate --tensixflow-to-json` writes the document, and `tensixflow-report` reads it. Field names are snake_case.

The translate step must run before `--tensixflow-lower-to-ttkernel`, because the lowering erases the source function that carries `graph_id` and the tensor list.

## Top level

| Field | Type | Required | Meaning |
|---|---|---|---|
| `graph_id` | string | yes | Identifier of the fused graph, for example `rmsnorm_matmul`. |
| `device` | object | yes | Device the plan was scheduled for. See [device block](#device-block). |
| `tensors` | array of objects | yes | Tensors the plan references. |
| `layout` | object | yes | Activation sharding and weight distribution. |
| `shard_grid` | object | yes | Core grid the plan maps onto. |
| `cb_specs` | array of objects | yes | Circular-buffer allocations. |
| `emission` | object | yes | Projected cost of the emitted program. See [emission block](#emission-block). |
| `noc` | object | yes | Projected NoC traffic in bytes. |
| `perf` | object | yes | Projected roofline result. |
| `viz` | object | yes | Visualizer section placeholders. |

Six user-facing report fields are optional at the top level: `activation_sharding`, `weight_strategy`, `intermediate_storage`, `fusion`, `estimated_dram_reads_saved`, and `estimated_intermediate_writes_saved`. `tensixflow-translate` flattens these from the schedule so a script can read the chosen layout without walking the nested objects. They mirror `layout` and the fusion decision, and the `plan::toJson` output that nests them stays valid.

## Device block

| Field | Type | Meaning |
|---|---|---|
| `name` | string | Device key, for example `n150`. |
| `arch` | string | Architecture name, for example `wormhole_b0`. |
| `grid_x` | integer | Compute-grid width. |
| `grid_y` | integer | Compute-grid height. |
| `l1_usable_bytes` | integer | Usable L1 per Tensix core. |
| `flops_per_cycle_per_core` | integer | Matrix-unit FLOPs per cycle per core. |
| `dram_bandwidth_bytes_per_second` | integer | Board DRAM bandwidth. |

The block carries the values the projection needs and that the plan cannot inherit from the file system. Clock, NoC rate, tile size, and the circular-buffer budget stay out of the JSON: `tensixflow-report` resolves the full spec by name from the generated device table. A plan whose `name` is not in that table is rejected with the valid names.

## Emission block

The emission block describes the program the Metalium emitter writes. It is separate from `noc` and `perf` because the emitted mapping is simpler than the planner's strategy: output tiles are split into contiguous ranges over the plan grid, and every core reads the operand tiles its output tiles need directly from DRAM with no multicast and no reuse.

| Field | Type | Meaning |
|---|---|---|
| `mapping` | string | `blocked-reuse` or `output-tile-ranges`. The emitter selects `blocked-reuse` when the core's resident activation and weight blocks fit the per-core circular-buffer budget, and keeps `output-tile-ranges` otherwise. The fused RMSNorm+MatMul path uses `output-tile-ranges`. The fused MLP chain reuses `blocked-reuse` with MLP semantics. |
| `cores_used` | integer | `min(grid cores, mTiles * nTiles)`. |
| `max_tiles_per_core` | integer | `ceil(mTiles * nTiles / cores_used)`. |
| `dram_read_bytes` | integer | Projected reads for the selected mapping. `blocked-reuse`: `kTiles * usedCols * sum(rows per used row block) + kTiles * weightReplication * sum(cols per used column block)` tiles. `output-tile-ranges`: `mTiles * nTiles * kTiles * 2` tiles. |
| `dram_write_bytes` | integer | `mTiles * nTiles * 2048`, one output tile each. |
| `fused_stage_tiles` | integer | `max_rows_per_core * kTiles` for the fused RMSNorm path, the activation tiles a core stages in L1. For the fused MLP chain it is `rowsPerCore * hTiles`, the GELU output rows a core keeps in L1 between the matmuls. |
| `fused_feasible` | boolean | True when the fused variant's buffers fit the per-core circular-buffer budget. The fused RMSNorm path needs `2 * fused_stage_tiles + 2 * kTiles + 3` tiles: staged activation, normalized activation, weight, gamma, one scaler, and two output. The fused MLP chain needs its twelve buffers, which the emitter prints when the chain does not fit. |

The values are projections of the emitted access pattern. Under `blocked-reuse` the activation row block is read once per used column block and the weight column block is read once and multicast down the used row blocks when the plan uses more than one row block. The activation replication is the one read the mapping cannot remove without an activation multicast, so the emitted reads stay above the planner's NoC projection. The report labels both, and `docs/performance.md` explains where each is used.

## Circular-buffer roles

The allocation is an ordered array. The Metalium emitter and the lowering pass read roles by position, not by index value.

- Unfused matmul plan: `[activation, weight, output]`, default indices `c_0`, `c_1`, `c_16`.
- Fused RMSNorm plan: `[activation stage, weight, gamma, scaler, normalized activation, output]`, default indices `c_0`, `c_1`, `c_2`, `c_3`, `c_4`, `c_16`.

A fused plan whose staging does not fit the budget is still a valid plan. Its emission block carries `fused_feasible: false`, and the emitter refuses to write a project with a diagnostic that prints the required and available tile counts. `examples/rmsnorm_matmul.mlir` and `examples/mlp_block.mlir` are the recorded infeasible cases.

## Nested objects

### `tensors` entry

| Field | Type | Meaning |
|---|---|---|
| `name` | string | Logical tensor name. |
| `shape` | array of integers | Static extent of each dimension. |
| `dtype` | string | Element type, for example `bf16` or `fp32`. |
| `placement` | string | `DRAM` or `L1`. |

### `layout`

| Field | Type | Meaning |
|---|---|---|
| `activation_sharding` | string | Activation sharding scheme, for example `row-major`. |
| `weight_strategy` | string | `replicate`, `shard`, or `multicast`. |

### `shard_grid`

| Field | Type | Meaning |
|---|---|---|
| `grid_y` | integer | Core-grid height. |
| `grid_x` | integer | Core-grid width. |

### `cb_specs` entry

| Field | Type | Meaning |
|---|---|---|
| `index` | integer | Circular-buffer index in the `tt::CBIndex` range, 0 to 31. |
| `depth` | integer | Depth in tiles. |
| `tile_bytes` | integer | Bytes per tile, 2,048 for a 32x32 bf16 tile. |

### `noc`

All four values are projected bytes from the static NoC estimator for the fused path.

| Field | Type | Meaning |
|---|---|---|
| `dram_read_bytes` | integer | Activation and weight reads. |
| `dram_write_bytes` | integer | Output writes. |
| `l1_bytes` | integer | Intermediate held in L1. |
| `multicast_bytes` | integer | Weight bytes moved by multicast. |

### `perf`

| Field | Type | Meaning |
|---|---|---|
| `est_cycles` | integer | Projected cycles, the largest of the compute, DRAM, and NoC terms. |
| `compute_util` | number | Compute cycles divided by projected cycles, 0.0 to 1.0. |
| `bottleneck` | string | `compute`, `dram`, `noc`, or `balanced` after the report runs the model. |
| `suggestions` | array of strings | Suggestions derived from the plan's parameters. |

[performance.md](performance.md) gives the formulas and constants behind these fields.

`tensixflow-translate` writes a placeholder `perf` block (zero cycles, zero utilization, `balanced`, no suggestions). `tensixflow-report` replaces it with the model's projection before rendering, so only the report output carries the projected values. A plan file produced by the translate step alone is a valid plan with an unfilled projection, not a measurement.

### `viz`

| Field | Type | Meaning |
|---|---|---|
| `core_grid` | string | Core-grid section placeholder. |
| `shards` | string | Shards section placeholder. |
| `noc_routes` | string | NoC-routes section placeholder. |
| `cb_usage` | string | Circular-buffer section placeholder. |
| `timeline` | string | Pipeline timeline placeholder. |

## Validation

`schema/plan.schema.json` sets `additionalProperties: false` at the top level and in every nested object, so an undeclared field or a wrong type is rejected. The ten top-level fields are required, and so are the seven fields of the device block and the seven fields of the emission block. `parsePlan` enforces the same contract in C++ and returns `std::nullopt` when a required field is absent or wrongly typed. A plan without a device block or an emission block is malformed, not defaulted.

`tensixflow-report` is the practical validator: it parses the plan before writing anything, so a malformed plan exits non-zero and leaves the output path untouched.

```sh
build/bin/tensixflow-report --in plan.json --out report.html
```

An external checker can validate the same document against `schema/plan.schema.json` with any JSON Schema draft-07 validator.

The numbers in `noc` and `perf` are projections. The schema has no field for measured performance, and the rendered report labels every cost figure as projected.
