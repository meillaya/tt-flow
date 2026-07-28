// SPDX-License-Identifier: Apache-2.0
//
// SIGNATURE-ONLY stub of the tt-metal KERNEL-SIDE dataflow intrinsics, pinned
// to tt-metal @df36cd6 (dataflow_api.h).
//
// COMPILE CONTRACT ONLY — every body is empty (`{}` / `return {};`). NO
// behavior. These live in the GLOBAL namespace exactly as the real kernel
// intrinsics do (kernels call `cb_reserve_back(...)`, not `tt::cb_reserve_back`).
//
// Declared: cb_reserve_back / cb_push_back / cb_wait_front / cb_pop_front /
//   get_write_ptr / get_read_ptr, noc_async_read / noc_async_write (+ barriers),
//   get_noc_addr / get_noc_multicast_addr, noc_semaphore_*,
//   get_arg_val / get_compile_time_arg_val.

#pragma once

#include <cstdint>

// --- Circular-buffer intrinsics ---------------------------------------------
inline void cb_reserve_back(std::uint32_t cb_id, std::uint32_t ntiles) {}
inline void cb_push_back(std::uint32_t cb_id, std::uint32_t ntiles) {}
inline void cb_wait_front(std::uint32_t cb_id, std::uint32_t ntiles) {}
inline void cb_pop_front(std::uint32_t cb_id, std::uint32_t ntiles) {}
inline std::uint32_t get_write_ptr(std::uint32_t cb_id) { return {}; }
inline std::uint32_t get_read_ptr(std::uint32_t cb_id) { return {}; }

// --- NoC async data movement ------------------------------------------------
inline void noc_async_read(std::uint32_t src_addr, std::uint32_t dest_addr, std::uint32_t size) {}
inline void noc_async_write(std::uint32_t src_addr, std::uint32_t dest_addr, std::uint32_t size) {}
inline void noc_async_read_barrier() {}
inline void noc_async_write_barrier() {}

// --- NoC addressing ---------------------------------------------------------
inline std::uint64_t get_noc_addr(std::uint32_t x, std::uint32_t y, std::uint32_t addr) { return {}; }
inline std::uint64_t get_noc_multicast_addr(std::uint32_t x_start,
                                            std::uint32_t y_start,
                                            std::uint32_t x_end,
                                            std::uint32_t y_end,
                                            std::uint32_t addr) { return {}; }

// --- NoC semaphores ---------------------------------------------------------
inline void noc_semaphore_inc(std::uint64_t addr, std::uint32_t inc) {}
inline void noc_semaphore_set(std::uint64_t addr, std::uint32_t value) {}
inline std::uint32_t noc_semaphore_read(std::uint64_t addr) { return {}; }

// --- Runtime / compile-time kernel arguments --------------------------------
template <typename T>
inline T get_arg_val(std::uint32_t arg_idx) { return T{}; }

template <typename T>
inline T get_compile_time_arg_val(std::uint32_t arg_idx) { return T{}; }
