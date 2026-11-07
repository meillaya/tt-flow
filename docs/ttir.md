# TTIR emission and import

TensixFlow moves TTIR text in both directions. `tensixflow-opt
--tensixflow-emit-ttir` prints a TTIR module for a tt-mlir frontend, and
`tensixflow-opt --tensixflow-import-ttir` rebuilds a documented subset of
bare-tensor TTIR as tensixflow IR so the planning pipeline can run on it.

Both directions use the tt-mlir release pinned in
[deps/tt-mlir.lock](../deps/tt-mlir.lock). Emitted text is parsed by the
released tt-mlir Python bindings in
[python/tensixflow/tests/test_ttir_parse.py](../python/tensixflow/tests/test_ttir_parse.py).
The emission pass itself is a terminal emission: it prints the text and exits 0,
and the lit suite checks its structure in
[test/Tensixflow/emit_ttir.mlir](../test/Tensixflow/emit_ttir.mlir).

## Pinned release

| Item | Value |
| --- | --- |
| Release tag | `0.9.0.dev20260221` |
| Annotated tag object | `f305713d0997b64f41420d3b2a948db0566e20a9` |
| Commit the tag points to | `5f396fd6ef85780bf1b6b85bef9c7d2f84a85f3c` |
| Wheel | `ttmlir-0.9.0.dev20260221-cp311-cp311-manylinux_2_34_x86_64.whl` |
| Wheel sha256 | `37b8fd1e7ee1d12b1ba531df9e63038db3cbea97acab2f21547083b4a5bb962e` |

`scripts/fetch_ttmlir_fixtures.sh` downloads the wheel and 14 syntax files at
the tagged commit into `.deps/ttmlir/`. Every file digest is recorded in the
lock file and checked again on each fetch.

The release ships a CPython 3.11 wheel and no `ttmlir-opt` executable. The wheel
contains `ttmlir.ir`, `ttmlir.passmanager`, and the ttcore, ttir, ttnn, d2m,
ttkernel, and ttmetal dialects. Because no other binary is released, the wheel
is the parser gate.

Evidence files live under `.deps/ttmlir/fixtures/` after the fetch. This page
quotes paths relative to that directory.

## Emitted ops

Each mapped op uses the generic MLIR form with the operand and result types
spelled out. Attributes sit in the `<{...}>` slot.

| tensixflow op | Emitted TTIR | Evidence |
| --- | --- | --- |
| `tensixflow.add` | `"ttir.add"` | `test/ttmlir/Dialect/TTIR/hoist/simple_add.mlir`; broadcast operands are legal per `test/ttmlir/Dialect/TTIR/fusing/matmul_with_bias_positive.mlir` |
| `tensixflow.matmul` | `"ttir.matmul"` | `test/ttmlir/Dialect/TTIR/matmul/matmul_tests_negative.mlir`; the parse test accepts the emitted shapes |
| `tensixflow.rmsnorm` | `"ttir.rms_norm"` with `epsilon`, `normalized_shape`, and `operandSegmentSizes = array<i32: 1, 1, 0>` | `test/ttmlir/Dialect/TTIR/fusing/rms_norm_fusion.mlir`, `test/ttmlir/Dialect/TTIR/ttir_rms_norm_negative.mlir` |
| `tensixflow.gelu` | `"ttir.gelu"` | `test/ttmlir/Dialect/TTIR/fusing/gelu_fusing.mlir` |
| `tensixflow.constant` | `"ttir.constant"` with a dense value attr | `test/ttmlir/Dialect/TTIR/creation/constant/simple_constant_negative.mlir` |
| `tensixflow.fused_rmsnorm_matmul` | rms_norm + matmul | the two rows above |
| `tensixflow.fused_mlp` | rms_norm + matmul + broadcast add + gelu, then an UNSUPPORTED marker for the tail | the rows above; the tail has no mapping |

The rmsnorm mapping uses the weight operand and no bias, so `operandSegmentSizes`
is `1, 1, 0`. A bias operand would make the split `1, 1, 1`, but no tensixflow op
carries one. The op declares `AttrSizedOperandSegments` at the pin, which is why
the attribute is part of the text.

An op with no mapping, such as `tensixflow.sched.gelu`, emits
`// UNSUPPORTED: <op> has no TTIR mapping` and leaves its results unbound.
Dependent ops are marked transitively. The marker is a comment, so it does not
break the parser.

## Layout attributes

Every tensor of rank 2 or more carries a `#ttcore.metal_layout` alias. The
dialect mnemonic at the pin is still `#ttcore.metal_layout`; the pinned fixture
gives the field order, which comes from the `assemblyFormat` of
`TTCore_MetalLayoutAttr`. This is the form quoted from
`test/ttmlir/Dialect/TTCore/metal_layout_attr_verify.mlir`:

```mlir
#ttcore.metal_layout<logical_shape = 32x32, dim_alignments = 32x32,
    collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>,
    undef, l1, sharded, index_map = map(0)>
```

The emitter fills the fields like this:

| Field | Emitted value | Reason |
| --- | --- | --- |
| `logical_shape` | the tensor shape | the shape the source op sees |
| `dim_alignments` | 1 for leading dims, 32 for the last two | mirrors `MetalLayoutAttr::computeTileAlignments` in `lib/Dialect/TTCore/IR/TTCoreOpsTypes.cpp` |
| `collapsed_intervals` | `[[0, rank-1], [rank-1, rank]]` | mirrors `computeDefaultFlattenedIntervals` in the same file |
| OOB value | `undef` | the enum case used by every attribute in the fixture above |
| memory space | `dram` for arguments, `l1` for op results | the emitter convention; spellings are the `TTCore_MemorySpace` cases in `include/ttmlir/Dialect/TTCore/IR/TTCoreOpsEnums.td` |
| memory layout | `interleaved` | the emitter runs before sharding, which owns the `sharded` form |
| `index_map` | `map(0)` | the identity sentinel the fixture uses |

A tensor of rank 1 keeps a plain type with no alias. The pinned verifier rejects
a rank-1 logical shape, and `test/ttmlir/Dialect/TTCore/metal_layout_attr_verify.mlir`
states the error text: `logical_shape must have at least 2 dimensions, got 1`.

## Running the parse gate

Fetch the wheel once, then run the test:

```sh
devenv shell -- bash scripts/fetch_ttmlir_fixtures.sh
devenv shell -- python3 -m pytest python/tensixflow/tests/test_ttir_parse.py -q
```

`scripts/test.sh` runs the same test inside the full CPU-only check.

The test does five things:

1. Reads `deps/tt-mlir.lock` and checks the sha256 of the cached wheel.
2. Unpacks the wheel into `.deps/ttmlir/site` when the cached extraction is
   missing or stale.
3. Emits TTIR from every lit input that runs `--tensixflow-emit-ttir`, which is
   `test/Tensixflow/emit_ttir.mlir` and
   `test/Tensixflow/emit_ttir_ttmlir_parse.mlir`. The `TENSIXFLOW_OPT` variable
   overrides the binary path; otherwise the test uses the newest
   `tensixflow-opt` under `build/bin` or `build-*/bin`.
4. Resolves the shared libraries the manylinux wheel expects (`libz`,
   `libstdc++`) from the nix store closures and starts a `python3.11` driver
   with `PYTHONPATH` and `LD_LIBRARY_PATH` set.
5. Imports `ttmlir.ir` and parses each emitted text. A parse or verifier
   diagnostic fails the test with the diagnostic text.

The test skips with a stated reason only when the pinned wheel or `python3.11`
is absent. When the wheel is present, an import or parse failure is a failure,
not a skip. The test also fails if the wheel digest does not match the lock, or
if `emit_ttir_ttmlir_parse.mlir` emits an UNSUPPORTED marker, because a marker
would hide a mapping gap behind a parse pass.

`test/Tensixflow/emit_ttir_unsupported.mlir` is not a parser-gate input. It
covers the marker path for the fused MLP tail, whose emitted function has no
return terminator. Its lit test checks the marker text only.

## Importing upstream TTIR

`tensixflow-opt --tensixflow-import-ttir` accepts a module written in a small
bare-tensor TTIR subset and rebuilds it as tensixflow IR. The import is a
rewrite, not a lowering: every accepted source op becomes the matching
tensixflow op, and the function becomes `tensixflow.func`, so the planning
pipeline runs unchanged.

The driver must be started with `--allow-unregistered-dialect`:

```sh
tensixflow-opt --allow-unregistered-dialect --tensixflow-import-ttir \
  --tensixflow-infer-tile-layout --tensixflow-plan-noc-sharding \
  --tensixflow-allocate-circular-buffers examples/ttir/rmsnorm_matmul.mlir
```

This build links neither the ttir dialect nor the ttcore dialect. The
`--allow-unregistered-dialect` flag is a driver flag, not a pass option: it lets
the parser read `"ttir.matmul"`-style generic operations and opaque `#ttcore.*`
attributes as unregistered constructs. The importer matches operations by name,
reads their attributes by string key, and rejects everything outside the
documented subset.

### Accepted subset

| Construct | Accepted form |
| --- | --- |
| container | exactly one `func.func` with one block and one result |
| types | bare static `tensor<...>` of bf16 or f32, rank 2 or higher, except the rms_norm weight |
| `ttir.matmul` | two operands: lhs rank 2 or higher, rhs rank 2; `transpose_a` and `transpose_b` absent or false |
| `ttir.add` | two operands and a result of the same bare type |
| `ttir.gelu` | one operand and a result of the same bare type |
| `ttir.rms_norm` | input plus rank-1 weight; `normalized_shape` names the input's last dimension; f32 `epsilon`; no bias |
| terminator | `func.return` with one operand of the function result type |

`ttir.rms_norm` keeps its pinned generic form, but only the input and weight
operands are imported. The importer maps `normalized_shape` by checking it
equals the last dimension, which is the only dimension `tensixflow.rmsnorm`
normalizes, and copies `epsilon` verbatim. An absent `epsilon` takes the pinned
TTIR default of 1e-5, not the tensixflow default.

The examples [examples/ttir/matmul.mlir](../examples/ttir/matmul.mlir) and
[examples/ttir/rmsnorm_matmul.mlir](../examples/ttir/rmsnorm_matmul.mlir) are
written in this subset. The lit tests
[test/Tensixflow/import_ttir.mlir](../test/Tensixflow/import_ttir.mlir) and
[test/Tensixflow/import_ttir_negative.mlir](../test/Tensixflow/import_ttir_negative.mlir)
run the import and the planning pipeline and check each rejection.

### Rejected constructs

Every rejection is a diagnostic that names the construct and points at this
page. The importer does not repair the input.

| Construct | Diagnostic names |
| --- | --- |
| another `ttir.*` or `ttnn.*` op | the operation, for example `ttir.constant` |
| any other operation, including nested modules and `scf` regions | the operation |
| layout-encoded tensor type | the encoded type, including `#ttcore.metal_layout` |
| module attribute from an unlinked dialect | the attribute, for example `ttcore.meshes` |
| dynamic shape | the tensor type |
| element type other than bf16 or f32 | the element type |
| `transpose_a = true` or `transpose_b = true` | the attribute and value |
| rank-1 tensor outside the rms_norm weight | the argument or operand type |
| more than one function | the function count |
| `ttir.rms_norm` bias operand | the bias operand form |
| `normalized_shape` other than the last dimension | the attribute value |

Function argument and result attributes are rejected as well: a
`ttcore.shard_status` or similar attribute would carry metadata from a dialect
this build does not link, and the subset accepts bare tensor arguments only.

### Why layout-encoded TTIR is rejected

Upstream tt-mlir attaches placements as `#ttcore.metal_layout` tensor encodings
and device metadata as `#ttcore.*` module attributes. This project does not link
the ttcore or ttnn dialects, so it cannot interpret those attributes, and
guessing at a layout would change the program. The importer therefore rejects
them with a diagnostic that names the attribute instead of stripping it from the
text. A strip-layout helper is out of scope: the supported way to hand TTIR to
TensixFlow is to request bare-tensor TTIR from the producer.

One consequence is that the two passes do not round-trip each other.
`--tensixflow-emit-ttir` writes `#ttcore.metal_layout` encodings, and
`--tensixflow-import-ttir` rejects them.

## What remains unverified

- Parse and local verification only. No upstream tt-mlir pipeline
  (TTIR-to-TTNN or TTIR-to-D2M) runs on the emitted module, so nothing proves
  that a lowering pipeline accepts these encodings.
- The assignment of `dram` to function arguments and `l1` to op results is a
  TensixFlow convention. The pinned builders derive alignments and collapse
  intervals, and the emitter mirrors them, but no pinned fixture validates this
  exact placement choice.
- The `fused_mlp` tail (matmul w2, residual add) has no mapping. Its emitted
  function carries markers and is excluded from the parser gate.
- There is no `ttmlir-opt` executable in the release, so the round trip through
  a real tool pipeline and its printer is not exercised.
- The pin is a nightly development wheel, not a stable release. A stable 0.9.0
  can change the syntax.
- Nothing executes the emitted TTIR. The parser gate is a syntax check, not a
  numeric or performance result.
- The importer accepts a documented subset, not arbitrary upstream TTIR. A
  real tt-mlir pipeline emits more operations, layout encodings, and module
  attributes than the subset covers, and those inputs are rejected by design.
- No upstream tool run validates the import. The examples are written in the
  shape of the pinned fixtures, but the release has no ttmlir-opt executable
  that could produce a real module for the importer.

## Updating the pin

Edit `deps/tt-mlir.lock`: the tag, the tag object, the commit, the wheel name,
the wheel digest, and the file digests. Re-run
`scripts/fetch_ttmlir_fixtures.sh`, then update this page, the two
`test/Tensixflow/emit_ttir*.mlir` fixtures, the import fixtures and examples,
and the comments that quote the pinned forms. Run the parser gate and the lit
suite before reporting the result.
