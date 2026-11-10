#!/usr/bin/env python3.11
"""Normalize layout-encoded TTIR so the TensixFlow importer can read it.

The TTIR emitter writes tensor placement as `#ttcore.metal_layout` encodings and
real tt-mlir output carries device attributes such as `ttcore.meshes`. The
importer in lib/tensixflow/ImportTTIRPass.cpp accepts bare-tensor TTIR only and
rejects that plumbing by design. This tool closes the loop: it parses the input
with the pinned tt-mlir Python bindings, rebuilds every layout-encoded tensor
type as its bare ranked type, drops the ttcore and ttnn module attributes, and
prints the result through the pinned printer. Function signatures, operations,
operand and result order, and semantic attributes are kept.

The rebuild runs through the bindings, not through a textual search. Parsing the
input with the pinned parser is also the syntax gate: a malformed module fails
here with the pinned diagnostic instead of at import time.

Run the tool through scripts/normalize_ttir.sh, which puts the extracted wheel
on PYTHONPATH and resolves the shared libraries the manylinux wheel needs:

    scripts/normalize_ttir.sh layout.mlir -o bare.mlir
    tensixflow-opt --tensixflow-emit-ttir examples/rmsnorm_matmul.mlir \\
        | scripts/normalize_ttir.sh

What the output still has to satisfy: the documented bare-tensor import subset
in docs/ttir.md. Normalization removes the layout encoding and the device
attributes; it does not widen the subset. An operation, a shape, or a module
attribute outside the subset still fails at import with a diagnostic.

Exit codes: 0 success, 1 unreadable input, parse failure, or a normalization
failure, 2 usage error.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NoReturn

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SITE = REPO_ROOT / ".deps" / "ttmlir" / "site"
SITE_PACKAGE = DEFAULT_SITE / "ttmlir" / "ir.py"
FETCH_HINT = "run 'devenv shell -- bash scripts/fetch_ttmlir_fixtures.sh'"
WRAPPER_HINT = (
    "run the tool through 'scripts/normalize_ttir.sh' inside the devenv shell"
)

# The released wheel registers these dialects at import. Registering all of
# them lets the pinned parser read every construct the emitter and tt-mlir
# write, including the ttcore layout attribute.
TTMLIR_DIALECTS = ("ttcore", "ttir", "ttnn", "d2m", "ttkernel")

# Attribute name prefixes of the dialects that own device placement. The
# importer rejects these dialects wholesale, so their module attributes are the
# second thing normalization drops after the tensor encodings.
DEVICE_DIALECT_PREFIXES = ("ttcore.", "ttnn.")

# Names of the operations the rebuild treats specially. A func.func carries the
# signature types in its function_type attribute, and a builtin.module carries
# the device attributes; every other operation only needs its value types.
FUNC_OP_NAME = "func.func"
MODULE_OP_NAME = "builtin.module"
FUNCTION_TYPE_ATTR = "function_type"

# Standard input and output spelling, shared by the positional input and the
# --output option.
STDIO_PATH = "-"

EXIT_OK = 0
EXIT_ERROR = 1

ARGS_INPUT_HELP = "TTIR file to normalize; '-' or omitted reads stdin"
ARGS_OUTPUT_HELP = "write the normalized TTIR here; '-' or omitted writes stdout"


def fail(message: str) -> NoReturn:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(EXIT_ERROR)


def load_pinned_bindings():
    """Import the pinned tt-mlir bindings and their dialect modules.

    The wrapper puts the extracted wheel on PYTHONPATH. A direct invocation
    without that path fails here with the fetch or wrapper hint rather than a
    traceback.
    """
    try:
        from ttmlir import ir

        for dialect in TTMLIR_DIALECTS:
            __import__(f"ttmlir.dialects.{dialect}")
    except ImportError as error:
        if not SITE_PACKAGE.is_file():
            fail(
                f"cannot import the pinned tt-mlir bindings: {error}; "
                f"the wheel is not extracted at {DEFAULT_SITE}, {FETCH_HINT}"
            )
        fail(f"cannot import the pinned tt-mlir bindings: {error}; {WRAPPER_HINT}")
    return ir


def read_input(path: str) -> tuple[str, str]:
    """Return the input text and a name for diagnostics."""
    if path == STDIO_PATH:
        return sys.stdin.read(), "stdin"
    try:
        return Path(path).read_text(encoding="utf-8"), path
    except OSError as error:
        fail(f"cannot read {path}: {error}")


def bare_type(ir, type_):
    """Return the type with any ttcore or ttnn encoding removed.

    Only ranked tensors can carry an encoding. Every other type is returned
    unchanged, so the rebuild does not disturb shapes or element types.
    """
    if not isinstance(type_, ir.RankedTensorType):
        return type_
    if type_.encoding is None:
        return type_
    return ir.RankedTensorType.get(type_.shape, type_.element_type)


def rewrite_function_type(ir, operation) -> None:
    """Rebuild a func.func signature from its now-bare types."""
    for attribute in operation.attributes:
        if str(attribute.name) != FUNCTION_TYPE_ATTR:
            continue
        old_type = attribute.attr.value
        new_type = ir.FunctionType.get(
            [bare_type(ir, input_type) for input_type in old_type.inputs],
            [bare_type(ir, result_type) for result_type in old_type.results],
        )
        operation.attributes[FUNCTION_TYPE_ATTR] = ir.TypeAttr.get(new_type)
        return


def drop_device_attributes(operation) -> None:
    """Delete module attributes that belong to the device dialects."""
    names = [str(attribute.name) for attribute in operation.attributes]
    for name in names:
        if name.startswith(DEVICE_DIALECT_PREFIXES):
            del operation.attributes[name]


def rewrite_operation(ir, operation) -> None:
    """Strip the encoding and device plumbing from one operation tree."""
    for result in operation.results:
        result.set_type(bare_type(ir, result.type))
    for region in operation.regions:
        for block in region.blocks:
            for argument in block.arguments:
                argument.set_type(bare_type(ir, argument.type))
            for nested in block.operations:
                rewrite_operation(ir, nested)
    if operation.operation.name == FUNC_OP_NAME:
        rewrite_function_type(ir, operation)
    if operation.operation.name == MODULE_OP_NAME:
        drop_device_attributes(operation)


def encoded_tensor_type(ir, type_):
    """Return the type when it still carries an encoding, else None."""
    if not isinstance(type_, ir.RankedTensorType):
        return None
    if type_.encoding is None:
        return None
    return type_


def check_no_encodings(ir, operation) -> None:
    """Fail when an encoded type survives, so the printer cannot emit one."""
    for result in operation.results:
        remaining = encoded_tensor_type(ir, result.type)
        if remaining is not None:
            fail(
                f"normalization left an encoded type on a "
                f"'{operation.operation.name}' result: {remaining}"
            )
    for region in operation.regions:
        for block in region.blocks:
            for argument in block.arguments:
                remaining = encoded_tensor_type(ir, argument.type)
                if remaining is not None:
                    fail(f"normalization left an encoded block argument: {remaining}")
            for nested in block.operations:
                check_no_encodings(ir, nested)


def normalize(ir, text: str, source: str) -> str:
    """Parse one TTIR module, rebuild it bare, and return the printed text."""
    context = ir.Context()
    # Unregistered dialects are legal input: the importer itself reads generic
    # operations under --allow-unregistered-dialect, and a tt-mlir module can
    # carry attributes from dialects the wheel does not register.
    context.allow_unregistered_dialects = True
    try:
        module = ir.Module.parse(text, context)
    except ir.MLIRError as error:
        fail(f"cannot parse {source} with the pinned tt-mlir bindings: {error}")
    # Attribute and type construction need a default location in the context.
    with ir.Location.unknown(context):
        rewrite_operation(ir, module.operation)
        check_no_encodings(ir, module.operation)
        return str(module)


def write_output(path: str, text: str) -> None:
    """Write the normalized module, with one trailing newline."""
    if not text.endswith("\n"):
        text = f"{text}\n"
    if path == STDIO_PATH:
        sys.stdout.write(text)
        return
    try:
        Path(path).write_text(text, encoding="utf-8")
    except OSError as error:
        fail(f"cannot write {path}: {error}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Strip layout encodings and device attributes from TTIR."
    )
    parser.add_argument("input", nargs="?", default=STDIO_PATH, help=ARGS_INPUT_HELP)
    parser.add_argument("-o", "--output", default=STDIO_PATH, help=ARGS_OUTPUT_HELP)
    args = parser.parse_args()

    text, source = read_input(args.input)
    ir = load_pinned_bindings()
    write_output(args.output, normalize(ir, text, source))
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
