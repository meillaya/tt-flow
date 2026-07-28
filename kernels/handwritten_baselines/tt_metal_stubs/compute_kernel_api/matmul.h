// SPDX-License-Identifier: Apache-2.0
//
// SIGNATURE-ONLY stub of the tt-metal matmul compute intrinsic, pinned to
// tt-metal @df36cd6 (matmul.h). COMPILE CONTRACT ONLY — empty bodies, NO
// behavior. Global namespace, matching real kernel usage.

#pragma once

#include <cstdint>

// Initialize the matmul engine (block dims default to a single tile).
inline void matmul_init(std::uint32_t in0_block_c_tiles = 1,
                        std::uint32_t in1_block_r_tiles = 1) {}

// Multiply one tile from in0_cb by one tile from in1_cb, accumulating into a
// destination tile register.
inline void matmul_tiles(std::uint32_t in0_cb_id,
                         std::uint32_t in1_cb_id,
                         std::uint32_t in0_tile_index,
                         std::uint32_t in1_tile_index,
                         std::uint32_t dst_tile_index,
                         bool transpose_of_faces) {}
