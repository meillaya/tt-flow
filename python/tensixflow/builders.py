"""TensixFlow DSL builders — explicit construction of ``tensixflow`` MLIR TEXT
(Stretch A, todo 47; draft D21 thin builder).

D21 GUARDRAIL: this is a THIN TEXT BUILDER, NOT a tracing JIT and NOT an AST
compiler. There is no ``ast`` / ``inspect.getsource`` / function tracing anywhere:
the user makes EXPLICIT builder calls (``b.rmsnorm(x, gamma)``, ``b.matmul(a, b)``,
...) and each call appends the corresponding ``tensixflow.*`` op TEXT to a function
body, threading SSA names + tensor types. The ``@tensixflow.jit`` decorator (todo 46)
captures a kernel; the lowering driver that invokes the ``tensixflow-opt`` pipeline
on the built text lands in todo 48.

The emitted op syntax matches the dialect EXACTLY (read off
``include/tensixflow/TensixflowOps.td`` + ``examples/rmsnorm_matmul.mlir`` +
``test/Tensixflow/{ops,matmul,rmsnorm}.mlir``):

* ``tensixflow.func @name(%x: tensor<...>, ...) -> tensor<...> { ...body...
  tensixflow.return %r : tensor<...> }``
* ``%N = tensixflow.rmsnorm %x, %g : tensor<...>, tensor<...> -> tensor<...>``
  (input type, gamma type, result type; epsilon is a defaulted attr, elided)
* ``%N = tensixflow.matmul %a, %b : tensor<...> x tensor<...> -> tensor<...>``
  (lhs type ``x`` rhs type, then the result type)
* ``%N = tensixflow.gelu %x : tensor<...>``            (single shared type)
* ``%N = tensixflow.add %a, %b : tensor<...>``         (single shared type; the
  residual connection is spelled ``tensixflow.add`` too — there is no residual op)

No MLIR Python bindings are required: the IR is emitted as plain text and is
validated by piping it through the built ``build/bin/tensixflow-opt`` (todo 8).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import NoReturn

__all__ = [
    "BuilderError",
    "TensorType",
    "Value",
    "FuncBuilder",
    "build_func",
]

# Element types the DSL accepts. The dialect's AnyRankedTensor admits any builtin
# float; these are the LLM-kernel-relevant ones (bf16 in / fp32 accum per the README
# and the examples). An unknown dtype is a builder error AT THE BOUNDARY, not a
# confusing MLIR parse failure downstream.
_FLOAT_DTYPES: frozenset[str] = frozenset({"bf16", "f16", "f32", "f64"})


class BuilderError(Exception):
    """Malformed DSL builder input (bad dtype / shape / SSA name).

    The dialect verifiers remain the authority on SEMANTIC errors (e.g. a matmul
    contracting-dim mismatch): those surface when the emitted text is parsed by
    ``tensixflow-opt``, which the builder deliberately does NOT pre-check.
    """


@dataclass(frozen=True, slots=True)
class TensorType:
    """A ranked tensor type: a static shape + an element type, rendered as MLIR text
    (``tensor<1x2048x4096xbf16>``). Immutable; the boundary parse validates the dtype
    and dims ONCE, so builders thread it without re-checking."""

    shape: tuple[int, ...]
    dtype: str = "bf16"

    def __post_init__(self) -> None:
        # Normalize any sequence to a tuple (frozen dataclass -> object.__setattr__).
        object.__setattr__(self, "shape", tuple(self.shape))
        if self.dtype not in _FLOAT_DTYPES:
            raise BuilderError(
                f"unknown dtype {self.dtype!r}; expected one of {sorted(_FLOAT_DTYPES)}"
            )
        if not self.shape:
            raise BuilderError("tensor type must have rank >= 1 (got an empty shape)")
        for dim in self.shape:
            if not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0:
                raise BuilderError(
                    f"tensor dims must be positive ints, got shape {self.shape!r}"
                )

    def __str__(self) -> str:
        dims = "x".join(str(d) for d in self.shape)
        return f"tensor<{dims}x{self.dtype}>"


@dataclass(frozen=True, slots=True)
class Value:
    """A symbolic SSA value: an SSA name (without the ``%``) + its tensor type.
    Builders consume and return these to thread names + types through the IR."""

    name: str
    type: TensorType

    @property
    def ssa(self) -> str:
        """The SSA name as it appears in IR (``%x`` / ``%0``)."""
        return f"%{self.name}"


class FuncBuilder:
    """Accumulates ``tensixflow`` op text inside one ``tensixflow.func``.

    Manages a small SSA-name counter for op RESULTS (``%0``, ``%1``, ...); function
    ARGUMENTS carry user-chosen names (``%x``, ``%gamma``). Call the op builders to
    append ops, then ``render(result)`` to produce the complete, parseable func text.
    """

    def __init__(self, name: str) -> None:
        if not name or not (name[0].isalpha() or name[0] == "_"):
            raise BuilderError(f"func name must be a valid MLIR symbol, got {name!r}")
        self._name = name
        self._counter = 0
        self._args: list[Value] = []
        self._body: list[str] = []

    @property
    def name(self) -> str:
        """The function's symbol name (``@<name>``)."""
        return self._name

    def __getattr__(self, name: str) -> NoReturn:
        """Clear "no builder" error for an unsupported op (todo 48 failure path).

        Called ONLY when normal lookup fails (real builders + attrs are found the
        usual way), so a jitted kernel that calls a non-existent builder method
        (``b.softmax(x)``) gets a precise :class:`BuilderError` naming the op and
        the supported set, instead of a bare ``AttributeError``. Raising the DSL's
        typed error (not ``AttributeError``) makes the failure path catchable as
        ``BuilderError``; the dialect verifiers remain the authority on SEMANTIC
        errors (e.g. a matmul K mismatch) at parse/verify time.
        """
        raise BuilderError(
            f"no builder for op {name!r}; supported ops: "
            "rmsnorm, matmul, gelu, add, residual"
        )

    # --- arguments ---------------------------------------------------------

    def arg(self, name: str, shape: Sequence[int], dtype: str = "bf16") -> Value:
        """Register a function argument ``%name: tensor<shape x dtype>`` and return
        its symbolic value. Argument names must be non-numeric MLIR identifiers so
        they can never collide with the numeric result counter."""
        if not name or name[0].isdigit():
            raise BuilderError(
                f"argument name must be a non-numeric MLIR identifier, got {name!r}"
            )
        value = Value(name, TensorType(tuple(shape), dtype))
        self._args.append(value)
        return value

    # --- SSA counter -------------------------------------------------------

    def _fresh(self) -> str:
        name = str(self._counter)
        self._counter += 1
        return name

    def _emit(self, line: str) -> None:
        self._body.append(line)

    # --- op builders (exact dialect syntax) --------------------------------

    def rmsnorm(self, x: Value, gamma: Value) -> Value:
        """``tensixflow.rmsnorm`` — the result has the SAME shape as ``x`` (gamma is
        the 1-D scale over the last dim; epsilon defaults to 1e-6 and is elided)."""
        result = Value(self._fresh(), x.type)
        self._emit(
            f"  {result.ssa} = tensixflow.rmsnorm {x.ssa}, {gamma.ssa} : "
            f"{x.type}, {gamma.type} -> {result.type}"
        )
        return result

    def matmul(self, a: Value, b: Value) -> Value:
        """``tensixflow.matmul`` — for ``a: [..., M, K]`` and ``b: [K, N]`` the result
        is ``[..., M, N]`` (batch dims from ``a``). The contracting-dim check is the
        dialect verifier's job: a K mismatch is rejected at parse/verify time, so the
        builder does NOT pre-validate it (it just threads the result shape)."""
        result_type = TensorType(a.type.shape[:-1] + (b.type.shape[-1],), a.type.dtype)
        result = Value(self._fresh(), result_type)
        self._emit(
            f"  {result.ssa} = tensixflow.matmul {a.ssa}, {b.ssa} : "
            f"{a.type} x {b.type} -> {result.type}"
        )
        return result

    def gelu(self, x: Value) -> Value:
        """``tensixflow.gelu`` — elementwise activation; result shape == ``x`` shape."""
        result = Value(self._fresh(), x.type)
        self._emit(f"  {result.ssa} = tensixflow.gelu {x.ssa} : {x.type}")
        return result

    def add(self, a: Value, b: Value) -> Value:
        """``tensixflow.add`` — elementwise sum (SameOperandsAndResultType: one shared
        type for both operands + the result); result shape == ``a`` shape."""
        result = Value(self._fresh(), a.type)
        self._emit(f"  {result.ssa} = tensixflow.add {a.ssa}, {b.ssa} : {a.type}")
        return result

    def residual(self, a: Value, b: Value) -> Value:
        """Residual connection — semantically an elementwise add, so it emits
        ``tensixflow.add`` (there is no ``tensixflow.residual`` op; ``examples/
        mlp_block.mlir`` spells the trailing ``+ residual`` as ``tensixflow.add``)."""
        result = Value(self._fresh(), a.type)
        self._emit(f"  {result.ssa} = tensixflow.add {a.ssa}, {b.ssa} : {a.type}")
        return result

    # --- rendering ---------------------------------------------------------

    def render(self, result: Value) -> str:
        """Render the complete ``tensixflow.func`` text, terminated by
        ``tensixflow.return %result``. The text parses under ``tensixflow-opt``."""
        args = ", ".join(f"{v.ssa}: {v.type}" for v in self._args)
        lines = [f"tensixflow.func @{self._name}({args}) -> {result.type} {{"]
        lines.extend(self._body)
        lines.append(f"  tensixflow.return {result.ssa} : {result.type}")
        lines.append("}")
        return "\n".join(lines) + "\n"


def build_func(
    name: str,
    args: Sequence[tuple[str, Sequence[int]] | tuple[str, Sequence[int], str]],
    body_fn: Callable[..., Value],
) -> str:
    """Build a complete ``tensixflow.func`` in one call.

    ``args`` is an ordered list of ``(name, shape)`` or ``(name, shape, dtype)``
    specs; ``body_fn(builder, *arg_values)`` wires the op builders and RETURNS the
    result value. Returns the parseable func text.

    Example::

        ir = build_func(
            "fused_rmsnorm_matmul",
            [("x", (1, 2048, 4096)), ("gamma", (4096,)), ("w", (4096, 12288))],
            lambda b, x, gamma, w: b.matmul(b.rmsnorm(x, gamma), w),
        )
    """
    builder = FuncBuilder(name)
    values = [builder.arg(spec[0], spec[1], *spec[2:]) for spec in args]
    result = body_fn(builder, *values)
    return builder.render(result)
