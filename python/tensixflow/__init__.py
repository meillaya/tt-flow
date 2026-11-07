"""TensixFlow Python package: DSL frontend, NumPy golden, and correctness harness.

The package exposes the rounding contract shared with the C++ simulator, the
fp32-exact golden reference, the PCC harness, the explicit DSL builders, and the
``@tensixflow.jit`` decorator. The lowering driver lives in
``tensixflow.lowering`` and imports on demand to avoid an import cycle.

Build and test commands are in AGENTS.md.
"""

from tensixflow.builders import (
    BuilderError,
    FuncBuilder,
    TensorType,
    Value,
    build_func,
)
from tensixflow.jit import JitFunction, JitTypeError, jit
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
    "BuilderError",
    "CONSISTENCY_COUNT",
    "CONSISTENCY_SEED",
    "FuncBuilder",
    "JitFunction",
    "JitTypeError",
    "TensorType",
    "Value",
    "bits_to_float",
    "build_func",
    "consistency_inputs",
    "float_to_bits",
    "golden_passthrough",
    "jit",
    "to_bf16",
    "to_bf16_bits",
    "to_bf16_reference",
]
