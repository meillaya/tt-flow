// SPDX-License-Identifier: Apache-2.0
//
// SIGNATURE-ONLY stub of tt-metal kernel/host TYPES, pinned to tt-metal @df36cd6.
//
// This header declares the type/enum/config surface the TensixFlow emitter
// (todos 34-36) references when it writes host.cpp. It is a COMPILE CONTRACT,
// not an implementation: every body is empty (`{}` / `= default` / `return {};`).
// There is NO behavior here and there must never be any.
//
// Structural-validation philosophy (README "Status / honesty"): generated
// TT-Metalium kernels are validated by compiling them `-fsyntax-only` against
// these stubs. The stubs declare the CURRENT tt-metal symbols (so a correct
// kernel compiles) and deliberately OMIT removed legacy symbols (so a stale
// kernel fails to compile). See host_api.hpp for the omitted-legacy note.
//
// Reference: tt-metal @df36cd6 — kernel_types.hpp L64-129, kernel_structs.h
// L12-44, plus the DataMovementConfig/ComputeConfig/CircularBufferConfig/
// InterleavedBufferConfig/ShardedBufferConfig aggregates used by host_api.hpp.

#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

// tt::CBIndex — the circular-buffer index enum (c_0..c_31). Lives in namespace
// `tt` (matching `tt::CBIndex::c_0` usage in current tt-metal host code).
namespace tt {
enum class CBIndex : std::uint8_t {
    c_0 = 0, c_1, c_2, c_3, c_4, c_5, c_6, c_7,
    c_8, c_9, c_10, c_11, c_12, c_13, c_14, c_15,
    c_16, c_17, c_18, c_19, c_20, c_21, c_22, c_23,
    c_24, c_25, c_26, c_27, c_28, c_29, c_30, c_31,
};
}  // namespace tt

namespace tt::tt_metal {

// --- Core grid addressing (kernel_structs.h) --------------------------------
struct CoreCoord {
    std::size_t x = 0;
    std::size_t y = 0;
};

struct CoreRange {
    CoreCoord start_coord{};
    CoreCoord end_coord{};
};

struct CoreRangeSet {
    CoreRangeSet() = default;
    CoreRangeSet(const CoreRange&) {}
    CoreRangeSet(const std::vector<CoreRange>&) {}
};

// A core spec is either a single coord, a range, or a set of ranges. The stub
// models this as a small variant-like aggregate accepting any of the three.
using CoreSpec = CoreRangeSet;

// --- Enumerations (kernel_types.hpp) ----------------------------------------
enum class DataMovementProcessor { RISCV_0, RISCV_1 };
enum class NOC { RISCV_0_default, RISCV_1_default, UNCONFIGURED };
enum class MathFidelity { LoFi, HiFi2, HiFi3, HiFi4 };
enum class DataType { BFloat16, Float32, BFloat8_b, UInt16, Int32 };
enum class BufferType { DRAM, L1, SYSTEM_MEMORY };

// --- Kernel config aggregates (host_api.hpp CreateKernel variant) -----------
// Designated-initializer aggregates (`.processor = ...`, `.noc = ...`,
// `.math_fidelity = ...`, `.compile_args = {...}`) — the exact fields the
// emitter sets. C++20 designated initializers require declaration order.
struct DataMovementConfig {
    DataMovementProcessor processor = DataMovementProcessor::RISCV_0;
    NOC noc = NOC::RISCV_0_default;
    std::vector<std::uint32_t> compile_args{};
};

struct ComputeConfig {
    MathFidelity math_fidelity = MathFidelity::HiFi4;
    bool fp32_dest_acc_en = false;
    bool dst_full_sync_en = false;
    bool bfp8_pack_precise = false;
    bool math_approx_mode = false;
    std::vector<std::uint32_t> compile_args{};
};

// --- Opaque handle types returned by the host surface -----------------------
class Program {};
using KernelHandle = std::uint32_t;
using CBHandle = std::uint32_t;
class Buffer {};

// --- Circular-buffer config (CreateCircularBuffer) --------------------------
struct CircularBufferConfig {
    CircularBufferConfig() = default;
    CircularBufferConfig(std::uint32_t, const std::map<CBIndex, std::uint32_t>&) {}
    CircularBufferConfig& set_page_size(CBIndex, std::uint32_t) { return *this; }
};

// --- Buffer configs (CreateBuffer Interleaved | Sharded) --------------------
struct InterleavedBufferConfig {
    std::uint32_t size = 0;
    std::uint32_t page_size = 0;
    BufferType buffer_type = BufferType::DRAM;
};

struct ShardSpec {
    CoreRange grid{CoreCoord{}, CoreCoord{}};
    std::vector<std::uint32_t> shape{};
};

struct ShardedBufferConfig {
    std::uint32_t size = 0;
    std::uint32_t page_size = 0;
    BufferType buffer_type = BufferType::L1;
    ShardSpec shard_spec{};
};

}  // namespace tt::tt_metal
