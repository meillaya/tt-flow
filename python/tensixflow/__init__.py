"""TensixFlow Python package: NumPy golden + correctness harness (Wave 7).

Currently exposes the bf16/fp32 rounding contract (draft D14) shared with the
C++ simulator (``lib/tensixflow/Sim/Rounding.h``); the golden (todo 40) and the
PCC harness (todo 41) build on it.
"""

from tensixflow.rounding import (
    CONSISTENCY_COUNT,
    CONSISTENCY_SEED,
    bits_to_float,
    consistency_inputs,
    float_to_bits,
    golden_passthrough,
    to_bf16,
    to_bf16_bits,
    to_bf16_reference,
)

__all__ = [
    "CONSISTENCY_COUNT",
    "CONSISTENCY_SEED",
    "bits_to_float",
    "consistency_inputs",
    "float_to_bits",
    "golden_passthrough",
    "to_bf16",
    "to_bf16_bits",
    "to_bf16_reference",
]
