# TensixFlow Plan / Perf / Viz JSON Schema

The plan schema is the **single source of truth** (decision D15) for the JSON
contract shared by every downstream stage:

- the **scheduler / planner** (Wave 5, todos 25–29) produces a `Plan`,
- the **plan + perf report** (todo 30) renders it,
- the **HTML visualizer** (todo 43) draws its `viz` sections.

No stage invents ad-hoc JSON. The C++ structs live in
[`include/tensixflow/PlanSchema.h`](../include/tensixflow/PlanSchema.h)
(serialized by `lib/tensixflow/PlanSchema.cpp`, library target
`MLIRTensixflowSchema`); the machine validation contract lives in
[`schema/plan.schema.json`](../schema/plan.schema.json) (JSON Schema draft-07).
Field names are `snake_case` to match the user-spec report format.

## Top level (`Plan`)

| Field | Type | Meaning |
| --- | --- | --- |
| `graph_id` | string | Identifier of the fused graph this plan schedules (e.g. `rmsnorm_matmul`). |
| `tensors` | array of `TensorDesc` | Tensors referenced by the plan. |
| `layout` | `Layout` | Chosen activation sharding + weight distribution. |
| `shard_grid` | `ShardGrid` | Core grid the plan maps onto. |
| `cb_specs` | array of `CBSpec` | Circular-buffer allocations. |
| `noc` | `NocEstimate` | Static NoC traffic estimate (bytes). |
| `perf` | `PerfEstimate` | Static roofline-style performance estimate. |
| `viz` | `VizSections` | Visualizer section placeholders (todo 43 fills detail). |

All eight fields are required.

## `TensorDesc`

| Field | Type | Meaning |
| --- | --- | --- |
| `name` | string | Logical tensor name. |
| `shape` | array of int64 | Static extent of each dimension. |
| `dtype` | string | Element dtype (e.g. `bf16`, `fp32`). |
| `placement` | string | Memory placement: `DRAM` or `L1`. |

## `Layout`

| Field | Type | Meaning |
| --- | --- | --- |
| `activation_sharding` | string | Activation sharding scheme (e.g. `row-major`). |
| `weight_strategy` | string | Weight distribution: `replicate`, `shard`, or `multicast`. |

## `ShardGrid`

| Field | Type | Meaning |
| --- | --- | --- |
| `grid_y` | int64 | Core-grid height (e.g. 8). |
| `grid_x` | int64 | Core-grid width (e.g. 8). |

## `CBSpec`

| Field | Type | Meaning |
| --- | --- | --- |
| `index` | int64 | Circular-buffer index (`tt::CBIndex`, 0..31). |
| `depth` | int64 | Depth in tiles. |
| `tile_bytes` | int64 | Bytes per tile (32×32 bf16 = 2048). |

## `NocEstimate`

| Field | Type | Meaning |
| --- | --- | --- |
| `dram_read_bytes` | int64 | Estimated DRAM read traffic (bytes). |
| `dram_write_bytes` | int64 | Estimated DRAM write traffic (bytes). |
| `l1_bytes` | int64 | Estimated L1 traffic (bytes). |
| `multicast_bytes` | int64 | Estimated NoC multicast traffic (bytes). |

## `PerfEstimate`

| Field | Type | Meaning |
| --- | --- | --- |
| `est_cycles` | int64 | Estimated total cycles. |
| `compute_util` | number | Compute utilization fraction (0.0..1.0). |
| `bottleneck` | string | Classification (e.g. `memory`, `compute`). |
| `suggestions` | array of string | Suggested optimizations. |

## `VizSections`

| Field | Type | Meaning |
| --- | --- | --- |
| `core_grid` | string | Core-grid section placeholder. |
| `shards` | string | Shards section placeholder. |
| `noc_routes` | string | NoC-routes section placeholder. |
| `cb_usage` | string | CB-usage section placeholder. |
| `timeline` | string | Timeline section placeholder. |

## Validation

`schema/plan.schema.json` is a draft-07 JSON Schema with `additionalProperties:
false` and every field above listed as required, so a plan with a missing or
wrongly-typed field is rejected. `parsePlan` (C++) enforces the same contract:
it returns `std::nullopt` when any required field is absent or mistyped.
