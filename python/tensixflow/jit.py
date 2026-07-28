"""TensixFlow ``@jit`` decorator (Stretch A; draft D21 thin builder).

D21 GUARDRAIL: this is a THIN BUILDER, NOT a tracing JIT and NOT an AST
compiler. ``jit`` CAPTURES the user's kernel function and nothing more: it does
not inspect, parse, trace, or execute the function body at decoration time. The
kernel makes EXPLICIT builder calls (todo 47); on a call with real NumPy arrays
the lowering driver (todo 48, :mod:`tensixflow.lowering`) builds the
``tensixflow`` MLIR text from those calls, runs the ``tensixflow-opt`` pipeline +
the C++ simulator, and returns a :class:`tensixflow.lowering.LoweredResult`.

A call that is NOT positional NumPy arrays (scalars, mixed args, kwargs) keeps
the EAGER fallback: it runs the captured Python function directly, so decorated
kernels stay runnable during bring-up.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import TYPE_CHECKING, Generic, ParamSpec, TypeVar

if TYPE_CHECKING:
    # Typing-only (never imported at runtime, so no __init__ -> jit -> lowering
    # cycle): names the lowering driver's result for the union return type below.
    from tensixflow.lowering import LoweredResult

_P = ParamSpec("_P")
_R = TypeVar("_R")
_T = TypeVar("_T")


class JitTypeError(TypeError):
    """Raised when ``@tensixflow.jit`` is applied to a non-callable.

    A ``TypeError`` subclass (so ``pytest.raises(TypeError)`` and ordinary
    ``except TypeError`` catch it) that carries the offending type as a typed
    field instead of a bare message string.
    """

    def __init__(self, received: _T) -> None:
        self.received_type: str = type(received).__name__
        super().__init__(
            "@tensixflow.jit expects a callable kernel function, got "
            f"{self.received_type!r}"
        )


class JitFunction(Generic[_P, _R]):
    """A kernel captured by ``@tensixflow.jit``.

    Holds the user function for MLIR emission (todo 47/48). The instance is
    immutable after construction; ``functools.update_wrapper`` copies the wrapped
    function's ``__name__``/``__doc__``/``__wrapped__`` metadata onto it so the
    decorated object still reads like the original kernel.

    Calling it with positional NumPy arrays LOWERS the kernel (todo 48): the driver
    builds the MLIR text from the function's explicit builder calls, runs the
    ``tensixflow-opt`` pipeline + the C++ simulator, and returns a
    :class:`tensixflow.lowering.LoweredResult` (plan JSON + result array). Any other
    call runs the captured function eagerly (the bring-up fallback).
    """

    def __init__(self, fn: Callable[_P, _R]) -> None:
        if not callable(fn):
            raise JitTypeError(fn)
        self._fn: Callable[_P, _R] = fn
        functools.update_wrapper(self, fn)

    @property
    def fn(self) -> Callable[_P, _R]:
        """The captured kernel function."""
        return self._fn

    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _R | LoweredResult:
        # Dispatch (todo 48): positional NumPy arrays LOWER the kernel (build MLIR
        # from its builder calls, drive the tensixflow-opt pipeline + simulator,
        # return a LoweredResult); any other call keeps the eager fallback (returns
        # the captured function's own result _R) so decorated kernels stay runnable
        # during bring-up. lowering is imported LAZILY (it pulls in the harness/golden
        # stack): a module-level import here would create an
        # __init__ -> jit -> lowering -> harness -> __init__ cycle, so it loads on
        # first call, when the package is fully initialized.
        from tensixflow import lowering

        if lowering.should_lower(args, kwargs):
            return lowering.lower_and_run(self._fn, args)
        return self._fn(*args, **kwargs)


def jit(fn: Callable[_P, _R]) -> JitFunction[_P, _R]:
    """Capture a kernel function for later MLIR emission (thin scaffold, D21).

    Usage::

        @tensixflow.jit
        def rmsnorm_kernel(x, gamma):
            ...

    Returns a :class:`JitFunction` wrapping ``fn``. Raises
    :class:`JitTypeError` (a ``TypeError``) if ``fn`` is not callable. Does NOT
    trace or parse the function body.
    """
    return JitFunction(fn)
