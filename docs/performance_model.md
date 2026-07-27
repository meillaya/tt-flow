# TensixFlow Performance Model

Stub. Full content lands in a later documentation pass. Each section below names what it will cover.

## L1-budget model

The on-chip budget the scheduler and performance model share: a conservative 1 MiB modeled L1 (versus roughly 1.5 MB on a real Wormhole n150), minus a 64 KiB code/stack reserve, leaving the circular-buffer and shard budget. Circular-buffer bytes follow `CB_bytes = n_CB x depth x 2048` (one 32x32 bf16 tile is 2048 bytes).

## Static performance formulas

The closed-form estimates behind the report: estimated cycles, DRAM and L1 bytes read/written, NoC multicast bytes, circular-buffer occupancy, compute utilization, and the memory-bound versus compute-bound classification that drives the suggested optimizations.

## Rounding contract

The deliberate precision split that makes PCC meaningful: an fp32-exact NumPy golden (no intermediate bf16 rounding) versus a bf16-tiled simulator (bf16 inputs and tile outputs, fp32 accumulation within a tile). The PCC pass bar is 0.9875, defended against vacuity by a structural fp32/bf16 dtype assertion plus a bit-difference sanity check (`max_abs_diff(sim, golden) > 0`) and an empirically calibrated upper trip-wire.
