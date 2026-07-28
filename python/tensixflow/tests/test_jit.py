"""Pytest for the ``@tensixflow.jit`` scaffold (todo 46, draft D21 thin builder).

Covers:
  * ``jit`` is exposed at the package top level (``tensixflow.jit``);
  * decorating CAPTURES the function (``.fn`` / ``__wrapped__`` / metadata);
  * decoration does NOT trace or execute the body (D21: thin builder only);
  * the decorated kernel stays callable via the eager fallback;
  * decorating a non-callable raises a clear ``TypeError`` (``JitTypeError``).
"""

import pytest

import tensixflow
from tensixflow.jit import JitFunction, JitTypeError


def test_jit_exposed_at_package_top_level():
    # Given the installed package
    # When reading the top-level export
    # Then it IS the jit decorator (and the error/wrapper types come with it)
    assert callable(tensixflow.jit)
    assert tensixflow.JitFunction is JitFunction
    assert tensixflow.JitTypeError is JitTypeError
    assert "jit" in tensixflow.__all__


def test_decorate_captures_the_function():
    # Given a kernel function
    def kernel(x):
        return x + 1

    # When decorating it
    decorated = tensixflow.jit(kernel)

    # Then the wrapper captures the original (identity, not a re-parse)
    assert isinstance(decorated, JitFunction)
    assert decorated.fn is kernel
    assert decorated.__wrapped__ is kernel
    assert decorated.__name__ == "kernel"


def test_decorator_syntax_captures():
    # Given/When the decorator is applied with @ syntax
    @tensixflow.jit
    def f():
        return 7

    # Then the name now binds a capturing wrapper around the original
    assert isinstance(f, JitFunction)
    assert f.__wrapped__ is not None
    assert f.__name__ == "f"


def test_decoration_does_not_trace_or_call_the_body():
    # Given a kernel that records execution
    called = []

    def kernel():
        called.append(1)

    # When decorating (and ONLY decorating)
    tensixflow.jit(kernel)

    # Then the body was never run: capture only, no tracing (D21)
    assert called == []


def test_decorated_kernel_runs_via_eager_fallback():
    # Given a decorated kernel
    @tensixflow.jit
    def add(a, b):
        return a + b

    # When calling it (pre-lowering eager fallback)
    # Then it runs the captured function and forwards args + result
    assert add(2, 3) == 5
    assert add(a=10, b=-4) == 6


@pytest.mark.parametrize(
    "bad",
    [42, 3.14, "kernel", None, ["fn"], {"fn": 1}],
    ids=["int", "float", "str", "none", "list", "dict"],
)
def test_non_callable_raises_type_error(bad):
    # Given a non-callable value
    # When decorating it
    # Then a clear TypeError (JitTypeError) names the offending type
    with pytest.raises(TypeError, match="callable kernel function"):
        tensixflow.jit(bad)


def test_non_callable_error_is_typed_and_catchable_as_typeerror():
    # Given/When a non-callable is decorated
    with pytest.raises(JitTypeError) as excinfo:
        tensixflow.jit(42)

    # Then the error carries the offending type as a typed field
    assert isinstance(excinfo.value, TypeError)
    assert excinfo.value.received_type == "int"
    assert "'int'" in str(excinfo.value)
