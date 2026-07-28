"""TensixFlow ``@jit`` decorator scaffold (Stretch A; draft D21 thin builder).

D21 GUARDRAIL: this is a THIN BUILDER scaffold, NOT a tracing JIT and NOT an
AST compiler. ``jit`` CAPTURES the user's kernel function and nothing more: it
does not inspect, parse, trace, or execute the function body at decoration
time. The explicit builder calls that emit ``tensixflow`` MLIR text land in
todo 47; the lowering driver that invokes the ``tensixflow-opt`` pipeline
lands in todo 48.

Until lowering exists, a decorated kernel falls back to EAGER execution:
calling it runs the captured Python function directly, so decorated kernels
stay runnable during bring-up.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Generic, ParamSpec, TypeVar

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

    Holds the user function for later MLIR emission (todo 47/48). The instance
    is immutable after construction; ``functools.update_wrapper`` copies the
    wrapped function's ``__name__``/``__doc__``/``__wrapped__`` metadata onto it
    so the decorated object still reads like the original kernel. Calling it
    runs the captured function eagerly (the pre-lowering fallback).
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

    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        # Scaffold only: no lowering yet (todo 47 emits MLIR, todo 48 drives the
        # pipeline). The eager fallback keeps the decorated kernel runnable.
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
