//===- L1Model.h ------------------------------------------------*- C++ -*-===//
//
// Shared L1-budget constants for the TensixFlow scheduler (todo 26). This is the
// SINGLE SOURCE OF TRUTH for the modeled per-core L1 budget: the CB allocator
// (todo 28) and the perf model (todo 42) import these constants — they must NOT
// be duplicated anywhere else.
//
// Modeled device: a single Wormhole n150-class part, 8x8 core grid, 32x32 tiles,
// bf16 data. tt-metal addresses memory by (x, y, local L1 address); each core's
// L1 is carved into code/stack plus the circular-buffer shard budget the planner
// reasons about.
//
//===----------------------------------------------------------------------===//

#ifndef TENSIXFLOW_SCHEDULER_L1MODEL_H
#define TENSIXFLOW_SCHEDULER_L1MODEL_H

#include <cstdint>

namespace mlir {
namespace tensixflow {
namespace scheduler {

// Modeled Wormhole n150 L1 budget. CONSERVATIVE modeled value: the literal n150
// L1 per core is larger (~1.5 MB); we model 1 MiB conservatively. Single source
// of truth — imported by the CB allocator (todo 28) and perf model (todo 42).
constexpr int64_t kL1TotalBytes = 1024 * 1024; // 1 MiB
constexpr int64_t kL1ReserveBytes = 64 * 1024; // 64 KiB (code/stack)
constexpr int64_t kCBShardBudget = kL1TotalBytes - kL1ReserveBytes; // 960 KiB
constexpr int64_t kTileBytes = 32 * 32 * 2; // 32x32 bf16 = 2048

} // namespace scheduler
} // namespace tensixflow
} // namespace mlir

#endif // TENSIXFLOW_SCHEDULER_L1MODEL_H
