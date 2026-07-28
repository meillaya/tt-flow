"""Pytest for DSL-vs-handwritten canonicalized-IR equivalence (todo 49, draft D21).

The Stretch-A acceptance (D21): the DSL builders (todo 47) produce IR that is
STRUCTURALLY IDENTICAL to the handwritten ``examples/rmsnorm_matmul.mlir``, after
BOTH are canonicalized by the BUILT ``tensixflow-opt`` (the todo-16
``--tensixflow-canonicalize`` pass) and SSA value names are NORMALIZED: every
``%<name>`` is renamed to a canonical ``%v0``, ``%v1``, ... sequence in order of
FIRST APPEARANCE, and the func symbol is replaced by ``@func``. Raw text is NEVER
compared: the DSL and the example choose different names (``@fused_rmsnorm_matmul``
vs ``@rmsnorm_matmul``; the pre-canonical DSL args ``%x``/``%gamma``/``%w`` vs the
lowering driver's ``%arg0``/...), so raw equality would be an accident and raw
inequality a false alarm. The normalized compare checks STRUCTURE (ops, operand
wiring, tensor types, order), which is the actual claim.

Non-vacuity (the QA failure scenario): a deliberately-different DSL build (an extra
``tensixflow.gelu`` the example does not have) must NOT match, proving the test
reads real structure rather than passing vacuously. CPU only: the gate invokes the
built ``tensixflow-opt`` (via ``tensixflow.lowering.find_opt_bin``, honoring
``TENSIXFLOW_OPT_BIN``), never a device.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Callable
from pathlib import Path

from tensixflow.builders import FuncBuilder, Value, build_func
from tensixflow.lowering import find_opt_bin

# tests/ -> tensixflow/ -> python/ -> repo root (the test_builders.py convention).
_REPO_ROOT = Path(__file__).resolve().parents[3]
_HANDWRITTEN_EXAMPLE = _REPO_ROOT / "examples" / "rmsnorm_matmul.mlir"

# The running example's argument specs (examples/rmsnorm_matmul.mlir verbatim):
# x [1,2048,4096], gamma [4096], w [4096,12288], all bf16 (the builder default).
_EXAMPLE_ARGS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("x", (1, 2048, 4096)),
    ("gamma", (4096,)),
    ("w", (4096, 12288)),
)


def _fused_body(b: FuncBuilder, x: Value, gamma: Value, w: Value) -> Value:
    """The fused_rmsnorm_matmul wiring: matmul(rmsnorm(x, gamma), w)."""
    return b.matmul(b.rmsnorm(x, gamma), w)


# An SSA value reference (%<identifier>): the only %-prefixed token in this IR
# (tensor types, attrs, and keywords carry no %).
_SSA_REF = re.compile(r"%[A-Za-z0-9_]+")
# A func symbol name (@<identifier>): the DSL and the example pick different names.
_FUNC_SYMBOL = re.compile(r"@[A-Za-z0-9_]+")


def _canonicalize(ir_text: str) -> str:
    """Run the BUILT ``tensixflow-opt --tensixflow-canonicalize`` (todo 16) over IR.

    Missing binary raises FileNotFoundError LOUDLY (the lowering.py convention,
    never a silent skip); a nonzero exit raises RuntimeError with both streams.
    """
    opt_bin = find_opt_bin()
    if not opt_bin.exists():
        raise FileNotFoundError(
            f"{opt_bin} missing: build it with `nix develop --command bash -c "
            "'cmake --build build'` (the DSL-vs-handwritten equivalence gate)."
        )
    proc = subprocess.run(
        [str(opt_bin), "--tensixflow-canonicalize"],
        input=ir_text,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"tensixflow-opt --tensixflow-canonicalize failed (exit {proc.returncode}):\n"
            f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
        )
    return proc.stdout


def _normalize(ir_text: str) -> str:
    """Normalize away everything that is NOT structure, so the compare checks structure.

    Renames every SSA value ``%<name>`` to ``%v0``, ``%v1``, ... in order of FIRST
    APPEARANCE (a deterministic, name-independent canonical numbering) and replaces
    the func symbol with ``@func``. Two IRs that differ only in SSA/symbol naming
    normalize to the SAME string; IRs that differ in ops, operand wiring, types, or
    order do NOT.
    """
    names: dict[str, str] = {}

    def _renumber(match: re.Match[str]) -> str:
        name = match.group(0)
        if name not in names:
            names[name] = f"%v{len(names)}"
        return names[name]

    return _FUNC_SYMBOL.sub("@func", _SSA_REF.sub(_renumber, ir_text))


def _normalized_canonical(ir_text: str) -> str:
    """Canonicalize (todo 16) then normalize SSA/symbol names: the compare key."""
    return _normalize(_canonicalize(ir_text))


def test_dsl_build_matches_handwritten_example_after_canonicalization():
    # Given the DSL-built fused_rmsnorm_matmul (todo 47 builders, the example's
    # shapes) and the handwritten example (the equivalence target)
    dsl_ir = build_func("fused_rmsnorm_matmul", _EXAMPLE_ARGS, _fused_body)
    handwritten_ir = _HANDWRITTEN_EXAMPLE.read_text()
    # When both are canonicalized by the SAME pass and SSA names normalized
    dsl_norm = _normalized_canonical(dsl_ir)
    handwritten_norm = _normalized_canonical(handwritten_ir)
    # Then the normalized canonical IRs are EQUAL (structure, not raw text)
    assert dsl_norm == handwritten_norm


def test_raw_canonical_texts_differ_so_name_normalization_is_required():
    # Given the same two IRs (the DSL build + the handwritten example)
    dsl_ir = build_func("fused_rmsnorm_matmul", _EXAMPLE_ARGS, _fused_body)
    handwritten_ir = _HANDWRITTEN_EXAMPLE.read_text()
    # When canonicalized WITHOUT the SSA/symbol normalization
    dsl_raw = _canonicalize(dsl_ir)
    handwritten_raw = _canonicalize(handwritten_ir)
    # Then the raw texts DIFFER (the func symbols @fused_rmsnorm_matmul vs
    # @rmsnorm_matmul) — a raw-text compare would be a false alarm, which is why
    # the equivalence gate normalizes names before comparing
    assert dsl_raw != handwritten_raw
    assert "@fused_rmsnorm_matmul" in dsl_raw
    assert "@rmsnorm_matmul" in handwritten_raw


def test_normalization_ignores_names_but_not_structure():
    # Given two IRs that differ ONLY in SSA value + func symbol names
    gelu_a = (
        "tensixflow.func @one(%a: tensor<2x2xbf16>) -> tensor<2x2xbf16> {\n"
        "  %r = tensixflow.gelu %a : tensor<2x2xbf16>\n"
        "  tensixflow.return %r : tensor<2x2xbf16>\n"
        "}\n"
    )
    gelu_b = (
        "tensixflow.func @two(%zzz: tensor<2x2xbf16>) -> tensor<2x2xbf16> {\n"
        "  %99 = tensixflow.gelu %zzz : tensor<2x2xbf16>\n"
        "  tensixflow.return %99 : tensor<2x2xbf16>\n"
        "}\n"
    )
    # Then they normalize to the SAME string (naming is ignored)
    assert _normalize(gelu_a) == _normalize(gelu_b)
    # But a STRUCTURAL change (a different op with a different operand arity)
    # normalizes DIFFERENTLY — the compare is structure-sensitive, not vacuous
    add_ir = (
        "tensixflow.func @three(%a: tensor<2x2xbf16>, %q: tensor<2x2xbf16>) -> tensor<2x2xbf16> {\n"
        "  %r = tensixflow.add %a, %q : tensor<2x2xbf16>\n"
        "  tensixflow.return %r : tensor<2x2xbf16>\n"
        "}\n"
    )
    assert _normalize(gelu_a) != _normalize(add_ir)


def test_deliberately_different_build_breaks_equivalence():
    # Given a DSL build with an EXTRA gelu the handwritten example does not have
    different_ir = build_func(
        "fused_rmsnorm_matmul_gelu",
        _EXAMPLE_ARGS,
        lambda b, x, gamma, w: b.gelu(b.matmul(b.rmsnorm(x, gamma), w)),
    )
    handwritten_norm = _normalized_canonical(_HANDWRITTEN_EXAMPLE.read_text())
    # When canonicalized + normalized the exact same way
    different_norm = _normalized_canonical(different_ir)
    # Then it does NOT match — the gate reads real structure (QA failure scenario)
    assert different_norm != handwritten_norm
    assert "tensixflow.gelu" in different_norm
