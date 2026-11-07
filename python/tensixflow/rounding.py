"""TensixFlow rounding contract: the Python twin of
``lib/tensixflow/Sim/Rounding.h``, bit-for-bit identical.

The simulator path rounds inputs to bf16 (round-to-nearest-even), accumulates in
fp32 within a tile, rounds each output tile to bf16, and stores fused
intermediates as bf16. The golden path keeps every intermediate in fp32. The two
paths differ on purpose so the PCC bar of 0.9875 measures real bf16 divergence.

bf16 is the high 16 bits of an IEEE-754 fp32 with the low 16 bits rounded.
``to_bf16`` rounds to nearest even. NaN inputs canonicalize to a
sign-preserving quiet NaN; inf and negative zero pass through unchanged.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "CONSISTENCY_SEED",
    "CONSISTENCY_COUNT",
    "float_to_bits",
    "bits_to_float",
    "to_bf16",
    "to_bf16_bits",
    "to_bf16_reference",
    "golden_passthrough",
    "consistency_inputs",
]

# Fixed seed + count for the cross-language consistency vector (shared with C++).
CONSISTENCY_SEED: int = 0x5EED2026
CONSISTENCY_COUNT: int = 1000

_U32 = np.uint32


def float_to_bits(x: object) -> np.ndarray:
    """IEEE-754 fp32 -> uint32 bit pattern (elementwise, little-endian host)."""
    return np.asarray(x, dtype=np.float32).view(_U32)


def bits_to_float(b: object) -> np.ndarray:
    """uint32 bit pattern -> fp32 (elementwise)."""
    return np.asarray(b, dtype=_U32).view(np.float32)


def _is_nan_bits(bits: np.ndarray) -> np.ndarray:
    """Exponent all-ones AND non-zero mantissa == NaN (elementwise)."""
    return ((bits & _U32(0x7F800000)) == _U32(0x7F800000)) & (
        (bits & _U32(0x007FFFFF)) != _U32(0)
    )


def _round_u32(bits: np.ndarray) -> np.ndarray:
    """SIMULATOR bf16 round-to-nearest-even on uint32 fp32 bits.

    Mirrors the C++ bit manipulation exactly: add the rounding bias
    ``0x7FFF + lsb`` (lsb = the bf16 mantissa LSB) so exact ties round to EVEN,
    mask off the low 16 bits; NaN canonicalizes to a sign-preserving quiet NaN.
    uint32 arithmetic wraps mod 2^32, matching C++ unsigned wrap.
    """
    bits = np.asarray(bits, dtype=_U32)
    nan = _is_nan_bits(bits)
    lsb = (bits >> _U32(16)) & _U32(1)
    rounded = (bits + _U32(0x7FFF) + lsb) & _U32(0xFFFF0000)
    canonical = (bits & _U32(0x80000000)) | _U32(0x7FC00000)
    return np.where(nan, canonical, rounded).astype(_U32)


def _reference_u32(bits: np.ndarray) -> np.ndarray:
    """Independent textbook reference (conditional rounding) for cross-checking.

    A DIFFERENT algorithm from the bias trick (explicit round-up decision) so it
    is a genuine oracle. The NaN canonicalization is the shared contract point.
    """
    bits = np.asarray(bits, dtype=_U32)
    nan = _is_nan_bits(bits)
    top = bits & _U32(0xFFFF0000)
    low = bits & _U32(0x0000FFFF)
    lsb = (bits >> _U32(16)) & _U32(1)
    round_up = (low > _U32(0x8000)) | ((low == _U32(0x8000)) & (lsb == _U32(1)))
    top = np.where(round_up, top + _U32(0x00010000), top).astype(_U32)
    canonical = (bits & _U32(0x80000000)) | _U32(0x7FC00000)
    return np.where(nan, canonical, top).astype(_U32)


def _scalar_in(x: object) -> bool:
    return np.asarray(x).ndim == 0


def to_bf16(x: object) -> object:
    """Round fp32 -> bf16 precision (round-to-nearest-even), as fp32.

    Scalar in -> Python ``float`` out; array in -> ``np.float32`` ndarray out.
    The result's low 16 mantissa bits are zero (it is exactly bf16-representable).
    """
    arr = np.asarray(x, dtype=np.float32)
    out = bits_to_float(_round_u32(arr.view(_U32)))
    return float(out) if arr.ndim == 0 else out


def to_bf16_bits(x: object) -> object:
    """The bf16 encoding (high 16 bits of to_bf16) as uint16, elementwise."""
    arr = np.asarray(x, dtype=np.float32)
    out = (_round_u32(arr.view(_U32)) >> _U32(16)).astype(np.uint16)
    return np.uint16(out) if arr.ndim == 0 else out


def to_bf16_reference(x: object) -> object:
    """Independent reference rounding (see ``_reference_u32``); same I/O shape
    as ``to_bf16``. Used only by the tests to cross-check ``to_bf16``."""
    arr = np.asarray(x, dtype=np.float32)
    out = bits_to_float(_reference_u32(arr.view(_U32)))
    return float(out) if arr.ndim == 0 else out


def golden_passthrough(x: object) -> object:
    """GOLDEN fp32-exact path: the identity on fp32, with NO bf16 rounding of
    intermediates. The golden calls this; the simulator calls ``to_bf16``.
    Preserves every fp32 bit exactly."""
    arr = np.asarray(x, dtype=np.float32)
    return float(arr) if arr.ndim == 0 else arr


def consistency_inputs(n: int = CONSISTENCY_COUNT, seed: int = CONSISTENCY_SEED) -> np.ndarray:
    """The fixed-seed 32-bit LCG (Numerical Recipes constants) shared with C++.

    Returns ``n`` uint32 values bit-identical to the C++ generator (Python masks
    with ``& 0xFFFFFFFF`` to match C++ unsigned wrap). Reinterpret with
    ``bits_to_float`` to get the fp32 consistency vector.
    """
    state = seed
    out = np.empty(n, dtype=_U32)
    for i in range(n):
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        out[i] = state
    return out
