"""TensixFlow Python package: DSL frontend + NumPy golden + correctness harness.

Wave 7 exposes the bf16/fp32 rounding contract (draft D14) shared with the C++
simulator (``lib/tensixflow/Sim/Rounding.h``), the fp32-exact golden (todo 40),
and the PCC harness (todo 41). Stretch A (Wave 8) adds the ``@tensixflow.jit``
decorator scaffold (todo 46, draft D21 thin builder): it captures a kernel
function for later MLIR emission and is NOT a tracing JIT; and the explicit DSL
builders (todo 47) that construct ``tensixflow`` MLIR TEXT (``FuncBuilder`` /
``build_func`` + the ``rmsnorm``/``matmul``/``gelu``/``add``/``residual`` op
builders) — explicit builder calls, no AST/tracing.
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
