"""Pytest for the TensixFlow DSL builders (todo 47, draft D21 thin builder).

The builders construct ``tensixflow`` MLIR TEXT via explicit calls (no AST/tracing).
These tests assert:

* HAPPY — a built ``fused_rmsnorm_matmul`` (rmsnorm -> matmul) emits text containing
  ``tensixflow.rmsnorm`` + ``tensixflow.matmul``, and that text PARSES (round-trips,
  exit 0) under the built ``build/bin/tensixflow-opt``; gelu/add/residual parse too.
* FAILURE — a matmul built with an incompatible contracting dim is rejected at
  parse/verify time (``tensixflow-opt`` exits non-zero with the verifier diagnostic).

The parse check drives the REAL C++ tool (todo 8) via subprocess — the same bridge
pattern as the correctness harness — NOT a Python re-parser.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import pytest

import tensixflow
from tensixflow.builders import (
    BuilderError,
    FuncBuilder,
    TensorType,
    Value,
    build_func,
)


def _repo_root() -> Path:
    # python/tensixflow/tests/test_builders.py -> parents[3] == repo root.
    return Path(__file__).resolve().parents[3]


def _opt_bin() -> Path:
    """Locate the ``tensixflow-opt`` parser/round-trip tool (env override honored).

    Raises loudly (NOT a skip) if the binary is missing: a missing handshake is a
    build problem, not a reason to silently pass (same policy as the harness).
    """
    env = os.environ.get("TENSIXFLOW_OPT_BIN")
    path = Path(env) if env else _repo_root() / "build" / "bin" / "tensixflow-opt"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing: build it with `nix develop --command bash -c "
            "'cmake --build build'` (the MLIR parser/round-trip tool)."
        )
    return path


def _roundtrip(ir_text: str) -> subprocess.CompletedProcess[str]:
    """Write ``ir_text`` to a temp ``.mlir`` and run ``tensixflow-opt`` over it.

    A bare ``tensixflow-opt <file>`` parses + verifies + reprints the module; exit 0
    means the IR is well-formed, non-zero means a parse/verify rejection.
    """
    with tempfile.TemporaryDirectory(prefix="tensixflow_dsl_") as tmp:
        path = Path(tmp) / "built.mlir"
        path.write_text(ir_text)
        return subprocess.run([str(_opt_bin()), str(path)], capture_output=True, text=True)


def _fused_rmsnorm_matmul_ir() -> str:
    """The running example: rmsnorm -> matmul, built via the explicit DSL calls."""
    return build_func(
        "rmsnorm_matmul",
        [("x", (1, 2048, 4096)), ("gamma", (4096,)), ("w", (4096, 12288))],
        lambda b, x, gamma, w: b.matmul(b.rmsnorm(x, gamma), w),
    )


# --- package surface + D21 guard ---------------------------------------------


def test_builders_exposed_at_package_top_level():
    # Given the installed package
    # When reading the top-level exports
    # Then the builder surface IS present (and listed in __all__)
    assert tensixflow.FuncBuilder is FuncBuilder
    assert tensixflow.build_func is build_func
    assert tensixflow.TensorType is TensorType
    assert tensixflow.Value is Value
    assert tensixflow.BuilderError is BuilderError
    for name in ("FuncBuilder", "build_func", "TensorType", "Value", "BuilderError"):
        assert name in tensixflow.__all__


def test_d21_builders_do_not_parse_ast_or_trace():
    # Given the imported builders module
    import tensixflow.builders as builders_mod

    # Then it is a THIN TEXT builder: it imports NEITHER `ast` NOR `inspect` (no AST
    # parsing, no source inspection / tracing — D21). This is a structural fact about
    # the module's imports (an `import ast` would bind a `ast` attribute), not prose.
    assert not hasattr(builders_mod, "ast")
    assert not hasattr(builders_mod, "inspect")


# --- type + value rendering --------------------------------------------------


def test_tensor_type_renders_mlir_text():
    # Given a ranked tensor type
    # When rendering it
    # Then it matches the dialect's tensor syntax exactly
    assert str(TensorType((1, 2048, 4096))) == "tensor<1x2048x4096xbf16>"
    assert str(TensorType((4096,))) == "tensor<4096xbf16>"
    assert str(TensorType((64, 64), "f32")) == "tensor<64x64xf32>"


@pytest.mark.parametrize("bad_dtype", ["bfloat16", "fp32", "BF16", "", "i32"])
def test_tensor_type_rejects_unknown_dtype(bad_dtype):
    # Given an element type the DSL does not admit
    # When constructing the type
    # Then a clear BuilderError names it (not a confusing MLIR parse error later)
    with pytest.raises(BuilderError, match="unknown dtype"):
        TensorType((64, 64), bad_dtype)


@pytest.mark.parametrize("bad_shape", [(), (64, 0), (64, -1), (64, True)])
def test_tensor_type_rejects_bad_shape(bad_shape):
    # Given a non-positive / empty / non-int shape
    # When constructing the type
    # Then a clear BuilderError rejects it
    with pytest.raises(BuilderError):
        TensorType(bad_shape)


def test_arg_rejects_numeric_ssa_name():
    # Given a builder
    b = FuncBuilder("f")
    # When registering an argument whose name is numeric
    # Then it is rejected (a numeric arg would collide with the result counter)
    with pytest.raises(BuilderError, match="non-numeric"):
        b.arg("0", (64, 64))


# --- shape threading + SSA counter -------------------------------------------


def test_matmul_result_shape_is_threaded_from_operands():
    # Given a: [..., M, K] and b: [K, N]
    b = FuncBuilder("f")
    a = b.arg("a", (1, 2048, 4096))   # [1, 2048, K=4096]
    w = b.arg("w", (4096, 12288))      # [K=4096, N=12288]
    # When building the matmul
    out = b.matmul(a, w)
    # Then the result shape is [..., M, N] (batch dims from a, N from b)
    assert out.type.shape == (1, 2048, 12288)
    assert out.type.dtype == "bf16"


def test_ssa_result_counter_increments():
    # Given a builder
    b = FuncBuilder("f")
    x = b.arg("x", (2048, 4096))
    # When emitting several ops
    r0 = b.gelu(x)
    r1 = b.gelu(r0)
    # Then results are named %0, %1, ... while arguments keep their given names
    assert r0.ssa == "%0"
    assert r1.ssa == "%1"
    assert x.ssa == "%x"


def test_residual_builder_emits_add_op():
    # Given two same-shape values
    b = FuncBuilder("f")
    a = b.arg("a", (1, 2048, 4096))
    r = b.arg("r", (1, 2048, 4096))
    # When building a residual connection
    out = b.residual(a, r)
    ir = b.render(out)
    # Then it emits tensixflow.add (there is no tensixflow.residual op; the dialect
    # spells the residual '+' as add — see examples/mlp_block.mlir)
    assert "tensixflow.add %a, %r : tensor<1x2048x4096xbf16>" in ir
    assert "tensixflow.residual" not in ir


# --- happy path: emitted text + parse/round-trip -----------------------------


def test_fused_rmsnorm_matmul_emits_the_ops():
    # Given/When building the running example
    ir = _fused_rmsnorm_matmul_ir()
    # Then the emitted text contains both ops (the acceptance assertion)
    assert "tensixflow.rmsnorm" in ir
    assert "tensixflow.matmul" in ir
    assert "tensixflow.func @rmsnorm_matmul" in ir
    assert "tensixflow.return" in ir


def test_fused_rmsnorm_matmul_exact_op_syntax():
    # Given/When building the running example
    ir = _fused_rmsnorm_matmul_ir()
    # Then each op line matches the dialect assembly format EXACTLY
    assert (
        "%0 = tensixflow.rmsnorm %x, %gamma : "
        "tensor<1x2048x4096xbf16>, tensor<4096xbf16> -> tensor<1x2048x4096xbf16>"
    ) in ir
    assert (
        "%1 = tensixflow.matmul %0, %w : "
        "tensor<1x2048x4096xbf16> x tensor<4096x12288xbf16> -> tensor<1x2048x12288xbf16>"
    ) in ir
    assert "tensixflow.return %1 : tensor<1x2048x12288xbf16>" in ir


def test_fused_rmsnorm_matmul_parses_under_tensixflow_opt():
    # Given the built IR
    ir = _fused_rmsnorm_matmul_ir()
    # When parsing + round-tripping it under the real tool
    proc = _roundtrip(ir)
    # Then it is well-formed (exit 0) and the ops survive the round-trip
    assert proc.returncode == 0, proc.stderr
    assert "tensixflow.rmsnorm" in proc.stdout
    assert "tensixflow.matmul" in proc.stdout


def test_built_ir_roundtrips_identical_to_handwritten_example():
    # Given the DSL-built example AND the hand-written examples/rmsnorm_matmul.mlir
    built = _fused_rmsnorm_matmul_ir()
    example = (_repo_root() / "examples" / "rmsnorm_matmul.mlir").read_text()
    # When round-tripping both under tensixflow-opt
    built_proc = _roundtrip(built)
    example_proc = _roundtrip(example)
    # Then the builder reproduces the hand-written IR exactly (byte-identical
    # round-trip): the strongest proof the emitted syntax matches the dialect
    assert built_proc.returncode == 0, built_proc.stderr
    assert example_proc.returncode == 0, example_proc.stderr
    assert built_proc.stdout == example_proc.stdout


def test_gelu_add_residual_parse_under_tensixflow_opt():
    # Given an MLP tail built from gelu + add + residual
    ir = build_func(
        "mlp_tail",
        [("h", (1, 2048, 12288)), ("bias", (1, 2048, 12288)), ("res", (1, 2048, 12288))],
        lambda b, h, bias, res: b.residual(b.gelu(b.add(h, bias)), res),
    )
    # Then the text carries gelu + add and parses (exit 0)
    assert "tensixflow.gelu" in ir
    assert "tensixflow.add" in ir
    proc = _roundtrip(ir)
    assert proc.returncode == 0, proc.stderr


# --- failure path: incompatible shapes rejected at parse/verify --------------


def test_matmul_incompatible_shapes_rejected_by_verifier():
    # Given a matmul whose contracting dims mismatch (a K=4096, b K=2048)
    ir = build_func(
        "bad_matmul",
        [("a", (2048, 4096)), ("b", (2048, 12288))],
        lambda b, a, bb: b.matmul(a, bb),
    )
    # When parsing it under tensixflow-opt
    proc = _roundtrip(ir)
    # Then the dialect verifier rejects it (non-zero exit + the contracting-dim
    # diagnostic) — the builder threads shapes but does NOT pre-validate them
    assert proc.returncode != 0
    assert "mismatched contracting dimensions" in proc.stderr
    assert "lhs K = 4096 but rhs K = 2048" in proc.stderr
