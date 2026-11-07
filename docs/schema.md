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

`tensixflow-translate` writes a placeholder `perf` block (zero cycles, zero utilization, `memory`, no suggestions). `tensixflow-report` replaces it with the model's projection before rendering, so only the report output carries the projected values. A plan file produced by the translate step alone is a valid plan with an unfilled projection, not a measurement.

### `viz`

| Field | Type | Meaning |
|---|---|---|
| `core_grid` | string | Core-grid section placeholder. |
| `shards` | string | Shards section placeholder. |
| `noc_routes` | string | NoC-routes section placeholder. |
| `cb_usage` | string | Circular-buffer section placeholder. |
| `timeline` | string | Pipeline timeline placeholder. |

## Validation

`schema/plan.schema.json` sets `additionalProperties: false` at the top level and in every nested object, so an undeclared field or a wrong type is rejected. The nine top-level fields are required, and so are the seven fields of the device block. `parsePlan` enforces the same contract in C++ and returns `std::nullopt` when a required field is absent or wrongly typed. A plan without a device block is malformed, not defaulted.

`tensixflow-report` is the practical validator: it parses the plan before writing anything, so a malformed plan exits non-zero and leaves the output path untouched.

```sh
build/bin/tensixflow-report --in plan.json --out report.html
```

An external checker can validate the same document against `schema/plan.schema.json` with any JSON Schema draft-07 validator.

The numbers in `noc` and `perf` are projections. The schema has no field for measured performance, and the rendered report labels every cost figure as projected.
