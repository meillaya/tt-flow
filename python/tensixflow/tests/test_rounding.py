"""Pytest for the TensixFlow bf16/fp32 rounding contract (draft D14, todo 38).

Covers:
  * to_bf16 known values (round-trip of representable bf16, rounding of 1.0001);
  * round-to-nearest-EVEN tie rule (a value exactly halfway rounds to even);
  * inf / NaN / negative-zero handling;
  * to_bf16 matches an independent reference on the fixed 1000-value vector;
  * the GOLDEN path holds a fused intermediate in fp32 (NOT bf16) — the
    non-vacuity invariant the PCC bar (todo 41) relies on;
  * C++/Python bit-for-bit consistency on 1000 random fp32 values (handshake
    against the file the gtest writes — run `ctest -R rounding` first);
  * a deliberately-wrong rounding mode (truncation) is detected.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from tensixflow import rounding as R


def bits1(x) -> int:
    """fp32 scalar -> uint32 bits."""
    return int(R.float_to_bits(np.float32(x)))


def f32_from_bits(b: int) -> float:
    """uint32 bits -> python float holding that fp32 value."""
    return float(np.array([b], dtype=np.uint32).view(np.float32)[0])


# --- to_bf16: known values -------------------------------------------------


@pytest.mark.parametrize("x", [1.0, 0.5, -2.0, 1.5, 2.0, 0.0])
def test_representable_bf16_round_trips_exactly(x):
    # A bf16-representable value (low 16 bits already zero) is unchanged.
    assert bits1(R.to_bf16(x)) == bits1(x)


def test_rounding_of_1_0001():
    # 1.0001 is just above 1.0; the nearest bf16 is 1.0 (next is 1.0078125).
    assert bits1(R.to_bf16(1.0001)) == 0x3F800000


# --- to_bf16: round-to-nearest-EVEN tie rule -------------------------------


def test_tie_rounds_to_even_down():
    # 0x3F808000 = 1.00390625, exactly halfway between bf16 1.0 (0x3F80, EVEN)
    # and 1.0078125 (0x3F81, odd). Round-to-even picks 1.0 (rounds DOWN).
    assert bits1(R.to_bf16(f32_from_bits(0x3F808000))) == 0x3F800000


def test_tie_rounds_to_even_up():
    # 0x3F818000 = 1.01171875, halfway between bf16 1.0078125 (0x3F81, odd) and
    # 1.015625 (0x3F82, EVEN). Round-to-even picks 1.015625 (rounds UP).
    assert bits1(R.to_bf16(f32_from_bits(0x3F818000))) == 0x3F820000


def test_wrong_rounding_mode_is_detected():
    # Truncation (round-down) differs from RNE on a tie that rounds UP, so a
    # deliberately-wrong mode would fail the consistency assertions above.
    half_up = f32_from_bits(0x3F818000)
    truncated = int(R.float_to_bits(np.float32(half_up))) & 0xFFFF0000
    assert truncated == 0x3F810000  # round-down keeps the odd candidate
    assert bits1(R.to_bf16(half_up)) != truncated  # RNE rounds to even instead


# --- to_bf16: special values -----------------------------------------------


def test_inf_preserved():
    assert bits1(R.to_bf16(float("inf"))) == 0x7F800000
    assert bits1(R.to_bf16(float("-inf"))) == 0xFF800000


def test_nan_stays_nan_canonical():
    out = R.to_bf16(float("nan"))
    assert np.isnan(out)
    assert bits1(out) == 0x7FC00000  # sign-preserving quiet NaN, payload cleared
    # A negative NaN keeps its sign bit.
    neg_nan = f32_from_bits(0xFFC00001)
    assert np.isnan(R.to_bf16(neg_nan))
    assert bits1(R.to_bf16(neg_nan)) == 0xFFC00000


def test_negative_zero_preserved():
    assert bits1(R.to_bf16(-0.0)) == 0x80000000


# --- to_bf16 vs independent reference on the fixed vector ------------------


def test_to_bf16_matches_reference_on_fixed_vector():
    inputs = R.bits_to_float(R.consistency_inputs())  # 1000 fp32 values
    got = R.float_to_bits(R.to_bf16(inputs))
    want = R.float_to_bits(R.to_bf16_reference(inputs))
    assert np.array_equal(got, want)


# --- GOLDEN path holds fp32 (NOT bf16) — the D14 non-vacuity invariant -----


def test_golden_holds_fused_intermediate_in_fp32_not_bf16():
    # A value bf16 CANNOT represent exactly (the simulator rounds it).
    x = np.float32(1.0001)
    x_bits = int(R.float_to_bits(x))
    # GOLDEN: passthrough preserves EVERY fp32 bit (fp32-exact intermediate).
    golden = R.golden_passthrough(x)
    golden_bits = int(R.float_to_bits(np.float32(golden)))
    assert golden_bits == x_bits
    # SIMULATOR: to_bf16 rounds it -> DIFFERENT bits.
    sim_bits = int(R.float_to_bits(np.float32(R.to_bf16(x))))
    assert sim_bits != x_bits
    # The two paths DIVERGE on this intermediate. If the golden also rounded to
    # bf16 these would be equal and PCC(sim, golden) would collapse to 1.0.
    assert golden_bits != sim_bits


def test_golden_preserves_fp32_array_exactly():
    x = R.bits_to_float(R.consistency_inputs(64))  # 64 arbitrary fp32 values
    golden = R.golden_passthrough(x)
    assert np.array_equal(R.float_to_bits(golden), R.float_to_bits(x))  # exact
    sim = R.to_bf16(x)
    # The simulator rounds at least one value -> the paths are NOT identical.
    assert not np.array_equal(R.float_to_bits(sim), R.float_to_bits(x))


# --- C++/Python bit-for-bit consistency (file handshake) -------------------


def _consistency_file() -> Path:
    env = os.environ.get("TENSIXFLOW_ROUNDING_FILE")
    if env:
        return Path(env)
    repo_root = Path(__file__).resolve().parents[3]  # tests->tensixflow->python->repo
    return repo_root / "build" / "rounding_consistency.txt"


def test_cpp_python_to_bf16_bit_for_bit_on_1000_random():
    path = _consistency_file()
    assert path.exists(), (
        f"{path} missing: run `ctest -R rounding` first so the C++ gtest writes "
        "its bf16 bit patterns (the cross-language consistency handshake)."
    )
    rows = [ln.split() for ln in path.read_text().splitlines() if ln.strip()]
    assert len(rows) == R.CONSISTENCY_COUNT
    cpp_in = np.array([int(a, 16) for a, _ in rows], dtype=np.uint32)
    cpp_bf16 = np.array([int(b, 16) for _, b in rows], dtype=np.uint16)
    # (1) The input generators agree bit-for-bit (same seed -> same uint32 seq).
    py_in = R.consistency_inputs()
    assert np.array_equal(py_in, cpp_in), "C++/Python LCG generators diverged"
    # (2) to_bf16 agrees bit-for-bit on all 1000 values.
    py_bf16 = R.to_bf16_bits(R.bits_to_float(py_in))
    assert np.array_equal(py_bf16, cpp_bf16), "C++/Python to_bf16 diverged"


# --- consistency-input generator lock --------------------------------------


def test_consistency_inputs_deterministic_and_locked():
    a = R.consistency_inputs()
    b = R.consistency_inputs()
    assert a.dtype == np.uint32
    assert len(a) == R.CONSISTENCY_COUNT
    assert np.array_equal(a, b)
    # First LCG output is fixed (locks the generator against drift).
    assert int(a[0]) == 0xACFEB94D
