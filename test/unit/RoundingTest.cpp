//===- RoundingTest.cpp ------------------------------------------*- C++ -*-===//
//
// Unit tests for the TensixFlow bf16/fp32 rounding contract (draft D14, todo 38)
// defined in lib/tensixflow/Sim/Rounding.h. Covers:
//
//   * to_bf16 known values (round-trip of representable bf16, rounding of
//     1.0001);
//   * round-to-nearest-EVEN tie rule (a value exactly halfway rounds to even);
//   * inf / NaN / negative-zero handling;
//   * to_bf16 matches an independent reference on the fixed 1000-value vector;
//   * the GOLDEN path holds a fused intermediate in fp32 (NOT bf16) — the
//     non-vacuity invariant the PCC bar (todo 41) relies on;
//   * writes build/rounding_consistency.txt so the Python test can assert the
//     C++ and Python to_bf16 agree bit-for-bit on 1000 random fp32 values.
//
//===----------------------------------------------------------------------===//

#include "Rounding.h"

#include "gtest/gtest.h"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>

using namespace mlir::tensixflow;

namespace {

uint32_t bits(float x) { return sim::floatToBits(x); }
float fromBits(uint32_t b) { return sim::bitsToFloat(b); }

TEST(RoundingTest, RepresentableBf16RoundTrips) {
  // A bf16-representable value (low 16 bits already zero) is unchanged.
  for (float x : {1.0f, 0.5f, -2.0f, 1.5f, 2.0f, 0.0f})
    EXPECT_EQ(bits(sim::to_bf16(x)), bits(x)) << "x=" << x;
}

TEST(RoundingTest, RoundingOf1_0001) {
  // 1.0001 is just above 1.0; the nearest bf16 is 1.0 (next is 1.0078125).
  EXPECT_EQ(bits(sim::to_bf16(1.0001f)), 0x3F800000u);
}

TEST(RoundingTest, TieRoundsToEvenDown) {
  // 0x3F808000 = 1.00390625, halfway between bf16 1.0 (0x3F80, EVEN) and
  // 1.0078125 (0x3F81, odd). Round-to-even picks 1.0 (rounds DOWN).
  EXPECT_EQ(bits(sim::to_bf16(fromBits(0x3F808000u))), 0x3F800000u);
}

TEST(RoundingTest, TieRoundsToEvenUp) {
  // 0x3F818000 = 1.01171875, halfway between bf16 1.0078125 (0x3F81, odd) and
  // 1.015625 (0x3F82, EVEN). Round-to-even picks 1.015625 (rounds UP).
  EXPECT_EQ(bits(sim::to_bf16(fromBits(0x3F818000u))), 0x3F820000u);
}

TEST(RoundingTest, WrongRoundingModeDetected) {
  // Truncation (round-down) differs from RNE on a tie that rounds UP, so a
  // deliberately-wrong mode would fail the assertions above.
  float halfUp = fromBits(0x3F818000u);
  uint32_t truncated = bits(halfUp) & 0xFFFF0000u;
  EXPECT_EQ(truncated, 0x3F810000u); // round-down keeps the odd candidate
  EXPECT_NE(bits(sim::to_bf16(halfUp)), truncated); // RNE rounds to even
}

TEST(RoundingTest, InfPreserved) {
  float inf = std::numeric_limits<float>::infinity();
  EXPECT_EQ(bits(sim::to_bf16(inf)), 0x7F800000u);
  EXPECT_EQ(bits(sim::to_bf16(-inf)), 0xFF800000u);
}

TEST(RoundingTest, NanStaysNanCanonical) {
  float out = sim::to_bf16(std::numeric_limits<float>::quiet_NaN());
  EXPECT_TRUE(std::isnan(out));
  EXPECT_EQ(bits(out), 0x7FC00000u); // sign-preserving quiet NaN, payload cleared
  // A negative NaN keeps its sign bit.
  float negNan = fromBits(0xFFC00001u);
  EXPECT_TRUE(std::isnan(sim::to_bf16(negNan)));
  EXPECT_EQ(bits(sim::to_bf16(negNan)), 0xFFC00000u);
}

TEST(RoundingTest, NegativeZeroPreserved) {
  EXPECT_EQ(bits(sim::to_bf16(-0.0f)), 0x80000000u);
}

TEST(RoundingTest, ToBf16BitsMatches) {
  EXPECT_EQ(sim::to_bf16_bits(1.0f), 0x3F80);
  EXPECT_EQ(sim::to_bf16_bits(fromBits(0x3F818000u)), 0x3F82);
}

TEST(RoundingTest, MatchesReferenceOnFixedVector) {
  // to_bf16 (bias trick) agrees with the independent reference (conditional
  // rounding) on every one of the 1000 fixed inputs, incl. NaN/inf/subnormals.
  for (uint32_t in : sim::consistencyInputs()) {
    float x = fromBits(in);
    EXPECT_EQ(bits(sim::to_bf16(x)), bits(sim::to_bf16_reference(x)))
        << "input bits " << in;
  }
}

TEST(RoundingTest, GoldenHoldsFp32NotBf16) {
  float x = 1.0001f; // bf16 cannot represent exactly
  uint32_t xBits = bits(x);
  // GOLDEN passthrough preserves every fp32 bit (fp32-exact intermediate).
  EXPECT_EQ(bits(golden::passthrough(x)), xBits);
  // SIMULATOR rounds it -> different bits.
  EXPECT_NE(bits(sim::to_bf16(x)), xBits);
  // The two paths DIVERGE (non-vacuity, D14): if the golden also rounded to bf16
  // these would be equal and PCC(sim, golden) would collapse to 1.0.
  EXPECT_NE(bits(golden::passthrough(x)), bits(sim::to_bf16(x)));
}

TEST(RoundingTest, ConsistencyInputsLocked) {
  auto v = sim::consistencyInputs();
  ASSERT_EQ(v.size(), sim::kConsistencyCount);
  EXPECT_EQ(v[0], 0xACFEB94Du); // first LCG output is fixed
}

// Writes the C++ bf16 bit patterns for the consistency vector so the pytest
// (python/tensixflow/tests/test_rounding.py) can assert bit-for-bit agreement —
// the cross-language handshake. Format per line: "<input_u32_hex> <bf16_u16_hex>".
TEST(RoundingTest, WriteConsistencyFile) {
  std::ofstream out(TENSIXFLOW_ROUNDING_OUT_DIR "/rounding_consistency.txt");
  ASSERT_TRUE(out.good()) << "cannot open consistency output file";
  for (uint32_t in : sim::consistencyInputs()) {
    out << std::hex << std::setfill('0') << std::setw(8) << in << " "
        << std::setw(4) << static_cast<unsigned>(sim::to_bf16_bits(fromBits(in)))
        << "\n";
  }
  out.flush();
  EXPECT_TRUE(out.good());
}

} // namespace
