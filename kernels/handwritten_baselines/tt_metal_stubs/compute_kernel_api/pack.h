// SPDX-License-Identifier: Apache-2.0
//
// SIGNATURE-ONLY stub of the tt-metal pack intrinsic, pinned to tt-metal
// @df36cd6 (pack.h). COMPILE CONTRACT ONLY — empty bodies, NO behavior.
// Global namespace, matching real kernel usage.

#pragma once

#include <cstdint>

// Pack a destination tile register into a circular buffer.
inline void pack_tile(std::uint32_t tile_index,
                      std::uint32_t cb_id,
                      bool transpose_of_faces = false) {}
