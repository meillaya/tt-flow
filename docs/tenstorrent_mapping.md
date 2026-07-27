# TensixFlow to Tenstorrent Mapping

Stub. Full content lands in a later documentation pass. Each section below names what it will cover.

## tensixflow to TT-Metalium current API mapping

How emitted kernels align to the current tt-metal API: a single `CreateKernel` using the `DataMovementConfig`/`ComputeConfig` variant, `tt::CBIndex::c_N` circular-buffer indices, the `distributed::Enqueue*` mesh API (`EnqueueWriteMeshBuffer`, `EnqueueMeshWorkload`, `EnqueueReadMeshBuffer`), and `MeshDevice` (including `create_unit_mesh`).

## Legacy API note

The removed upstream APIs (`EnqueueProgram` and per-role `CreateKernel`) are documented here as a mapping reference only. They are not the primary codegen path and are not emitted.

## Dialect taxonomy

Where TensixFlow sits relative to the TT-MLIR dialect family: TTIR, TTNN, D2M, TTKernel, and TTMetal, and which of those each TensixFlow stage corresponds to.
