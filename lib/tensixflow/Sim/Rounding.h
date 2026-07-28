//===- Sim/Rounding.h -------------------------------------------*- C++ -*-===//
//
// TensixFlow rounding contract (draft D14) — shared by the functional tile
// simulator (todo 39) AND the NumPy golden (todo 40), with a Python twin in
// python/tensixflow/rounding.py that is BIT-FOR-BIT identical (proven by the
// C++/Python consistency test in test/unit/RoundingTest.cpp +
// python/tensixflow/tests/test_rounding.py).
//
// THE D14 CONTRACT — the simulator and the golden are DELIBERATELY at different
// precision so the PCC bar (todo 41, >= 0.9875) measures REAL bf16-spatial-vs-
// exact divergence rather than collapsing to ~1.0:
//
//   * SIMULATOR path (namespace `sim`) = bf16 SPATIAL hardware model:
//       - inputs are rounded to bf16 (to_bf16, round-to-nearest-even);
//       - matmul/rmsnorm/gelu ACCUMULATE in fp32 WITHIN a tile;
//       - each output TILE is rounded back to bf16;
//       - fused SRAM / circular-buffer INTERMEDIATES between stages are stored
//         as bf16 (i.e. rounded between fused stages).
//     The simulator must NOT silently keep fp32 storage for fused intermediates
//     — that would under-model the bf16 spatial cost.
//
//   * GOLDEN path (namespace `golden`) = fp32-EXACT reference:
//       - fp32 accumulate, fp32 intermediates, NO bf16 rounding of intermediates
//         (golden::passthrough is the identity on fp32). Round only the final
//         compared output, if at all.
//     The golden must NOT round intermediates to bf16 — doing so makes the two
//     paths identical and the PCC bar VACUOUS (PCC -> 1.0).
//
// PCC(simulator_bf16_tiled, golden_fp32_exact) therefore measures the real
// cost of fusion + tiled bf16 execution (arXiv:2606.09879 context: hardware-bf16
// vs fp32 reference; the 0.9875 floor comes from there).
//
// bf16 = sign(1) + exponent(8) + mantissa(7): the HIGH 16 bits of an IEEE-754
// fp32, with the LOW 16 bits rounded. to_bf16 below is round-to-nearest-EVEN
// (the +lsb bias makes exact ties land on the even candidate), matching
// tt-metal@df36cd6 base_types.hpp DataType::BFloat16.
//
//===----------------------------------------------------------------------===//

#ifndef TENSIXFLOW_SIM_ROUNDING_H
#define TENSIXFLOW_SIM_ROUNDING_H

#include <cstdint>
#include <cstring>
#include <vector>

namespace mlir::tensixflow::sim {

//===----------------------------------------------------------------------===//
// Bit reinterpretation (IEEE-754 fp32 <-> uint32, little-endian host).
//===----------------------------------------------------------------------===//

inline uint32_t floatToBits(float x) {
  uint32_t b;
  std::memcpy(&b, &x, sizeof(b));
  return b;
}

inline float bitsToFloat(uint32_t b) {
  float x;
  std::memcpy(&x, &b, sizeof(x));
  return x;
}

/// True iff `b` encodes a NaN (exponent all-ones AND non-zero mantissa).
inline bool isNaNBits(uint32_t b) {
  return (b & 0x7F800000u) == 0x7F800000u && (b & 0x007FFFFFu) != 0u;
}

//===----------------------------------------------------------------------===//
// SIMULATOR bf16 spatial rounding (round-to-nearest-even).
//===----------------------------------------------------------------------===//

/// Round an fp32 to bf16 precision (round-to-nearest-even), returned as an fp32
/// whose low 16 mantissa bits are zero. This is the simulator's spatial model:
/// hardware stores activations/weights/intermediates as bf16.
///
/// Bit manipulation: add the rounding bias `0x7FFF + lsb` (lsb = the bf16
/// mantissa LSB) so exact ties round to EVEN, then mask off the low 16 bits.
/// NaN inputs are canonicalized to a sign-preserving quiet NaN (the unguarded
/// bias-add can overflow a NaN payload into the sign bit and produce a zero —
/// the guard makes "NaN stays NaN" hold for every payload). inf and negative
/// zero pass through unchanged.
inline float to_bf16(float x) {
  uint32_t b = floatToBits(x);
  if (isNaNBits(b))
    return bitsToFloat((b & 0x80000000u) | 0x7FC00000u); // quiet NaN, sign kept
  b += 0x7FFFu + ((b >> 16) & 1u);
  b &= 0xFFFF0000u;
  return bitsToFloat(b);
}

/// The bf16 encoding of `x` (the high 16 bits of to_bf16(x)) as a uint16.
inline uint16_t to_bf16_bits(float x) {
  return static_cast<uint16_t>(floatToBits(to_bf16(x)) >> 16);
}

//===----------------------------------------------------------------------===//
// Independent reference (textbook conditional rounding) — used ONLY by the unit
// test to cross-check to_bf16. Deliberately a DIFFERENT algorithm (explicit
// round-up decision) so it is a genuine oracle, not a restatement of the bias
// trick. The NaN canonicalization is the shared contract point.
//===----------------------------------------------------------------------===//

inline float to_bf16_reference(float x) {
  uint32_t b = floatToBits(x);
  if (isNaNBits(b))
    return bitsToFloat((b & 0x80000000u) | 0x7FC00000u);
  uint32_t top = b & 0xFFFF0000u; // truncated (round-down) bf16
  uint32_t low = b & 0x0000FFFFu; // discarded low 16 bits
  uint32_t lsb = (b >> 16) & 1u;  // retained bf16 mantissa LSB
  // Round up when the discarded part exceeds half, OR is exactly half and the
  // retained value is odd (increment makes it even) — round-to-nearest-even.
  bool roundUp = (low > 0x8000u) || (low == 0x8000u && lsb == 1u);
  if (roundUp)
    top += 0x00010000u; // carries mantissa -> exponent naturally
  return bitsToFloat(top);
}

//===----------------------------------------------------------------------===//
// Consistency-test input generator (shared with Python). A fixed-seed 32-bit LCG
// (Numerical Recipes constants) whose uint32 sequence is bit-identical in C++
// (unsigned wrap) and Python (& 0xFFFFFFFF). Each uint32 is reinterpreted as an
// fp32, giving a spread of normals, subnormals, inf and NaN — so the edge-case
// handling is exercised identically across both languages.
//===----------------------------------------------------------------------===//

inline constexpr uint32_t kConsistencySeed = 0x5EED2026u;
inline constexpr std::size_t kConsistencyCount = 1000;

inline std::vector<uint32_t>
consistencyInputs(std::size_t n = kConsistencyCount,
                  uint32_t seed = kConsistencySeed) {
  std::vector<uint32_t> v;
  v.reserve(n);
  uint32_t state = seed;
  for (std::size_t i = 0; i < n; ++i) {
    state = state * 1664525u + 1013904223u; // mod 2^32 (unsigned wrap)
    v.push_back(state);
  }
  return v;
}

} // namespace mlir::tensixflow::sim

namespace mlir::tensixflow::golden {

//===----------------------------------------------------------------------===//
// GOLDEN fp32-exact path (draft D14). The golden reference accumulates and
// stores intermediates in fp32 with NO bf16 rounding — passthrough is the
// identity on fp32. The simulator (todo 39) calls sim::to_bf16; the golden
// (todo 40) calls golden::passthrough. Keeping these two paths distinct is what
// makes the PCC bar (todo 41) non-vacuous.
//===----------------------------------------------------------------------===//

inline float passthrough(float x) { return x; }

} // namespace mlir::tensixflow::golden

#endif // TENSIXFLOW_SIM_ROUNDING_H
