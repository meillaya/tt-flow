"""TensixFlow DSL lowering driver.

A call to a jitted kernel with positional NumPy arrays runs three steps:

1. Build the ``tensixflow`` MLIR text from the kernel's explicit builder calls.
   The jitted function is the body function: it receives a ``FuncBuilder`` and
   one symbolic ``Value`` per input array, and returns the result ``Value``.
2. Run the schedule-producing passes through the built ``tensixflow-opt`` and
   serialize the module schedule attribute to plan JSON through
   ``tensixflow-translate --tensixflow-to-json``.
3. Run the built ``tensixflow-simulate`` over the same pipeline output. Inputs
   and outputs use the self-describing ``.tft`` binary. The simulator reads the
   activation as rank 2 ``[M, K]``, so the driver flattens leading batch dims
   into ``M``.

The driver returns a ``LoweredResult`` with the plan JSON and the simulator
result. It invokes the built binaries and does not reimplement a pass in Python.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tensixflow import harness
from tensixflow.builders import BuilderError, FuncBuilder, Value

__all__ = [
    "PIPELINE_FLAGS",
    "LoweredResult",
    "build_mlir",
    "find_opt_bin",
    "find_translate_bin",
    "lower_and_run",
    "should_lower",
]

# The full schedule-producing pipeline, FULL ``--tensixflow-*`` flag names (confirmed
# via ``build/bin/tensixflow-opt --help``; the plan-prose shorthand ``--infer-tile-layout``
# etc. is NOT registered). Order matters: infer layouts -> fuse -> plan NoC sharding
# -> allocate circular buffers -> lower to the TT-Metalium kernel schedule.
PIPELINE_FLAGS: tuple[str, ...] = (
    "--tensixflow-infer-tile-layout",
    "--tensixflow-fuse-rmsnorm-matmul",
    "--tensixflow-plan-noc-sharding",
    "--tensixflow-allocate-circular-buffers",
    "--tensixflow-lower-to-ttkernel",
)


def _repo_root() -> Path:
    # python/tensixflow/lowering.py -> parents[2] == repo root.
    return Path(__file__).resolve().parents[2]


def find_opt_bin() -> Path:
    """Locate the ``tensixflow-opt`` driver (env override ``TENSIXFLOW_OPT_BIN``)."""
    env = os.environ.get("TENSIXFLOW_OPT_BIN")
    if env:
        return Path(env)
    return _repo_root() / "build" / "bin" / "tensixflow-opt"


def find_translate_bin() -> Path:
    """Locate ``tensixflow-translate`` (env override ``TENSIXFLOW_TRANSLATE_BIN``)."""
    env = os.environ.get("TENSIXFLOW_TRANSLATE_BIN")
    if env:
        return Path(env)
    return _repo_root() / "build" / "bin" / "tensixflow-translate"


@dataclass(frozen=True, slots=True)
class LoweredResult:
    """The outcome of lowering a jitted kernel + running the simulator.

    * ``plan``: the plan JSON (a dict) from ``tensixflow-translate``. It carries the
      user-facing report fields (``weight_strategy``, ``fusion``, ``shard_grid``,
      ``cb_specs``, ``noc``, ...). ``weight_strategy`` is the acceptance field.
    * ``output``: the simulator's output tensor (fp32 ndarray, shape ``[M, N]``).
    * ``mlir``: the built ``tensixflow.func`` text, for debugging and for the
      DSL-versus-handwritten equivalence check.
    """

    plan: dict[str, object]
    output: np.ndarray
    mlir: str


def should_lower(args: tuple[object, ...], kwargs: dict[str, object]) -> bool:
    """The lowering trigger: a call of POSITIONAL NumPy arrays (and no kwargs).

    Anything else (scalars, mixed args, kwargs) keeps the eager fallback, so the
    scaffold's bring-up behavior (``add(2, 3) == 5``) is untouched.
    """
    return bool(args) and not kwargs and all(isinstance(a, np.ndarray) for a in args)


def build_mlir(fn: Callable[..., object], args: Sequence[np.ndarray], *, name: str | None = None) -> str:
    """Build the ``tensixflow.func`` text from a kernel's explicit builder calls.

    ``fn`` is the jitted body: it must call the builders and return the result
    ``Value`` (``fn(builder, *arg_values) -> Value``); the return is validated at
    this boundary (the isinstance check below), so ``fn`` is typed ``-> object``.
    Each input array becomes a symbolic arg ``%arg<i>: tensor<shape x bf16>`` (shape
    from the array). Returns the parseable func text. Raises :class:`BuilderError` if
    ``fn`` calls an unsupported op (no builder) or does not return a builder ``Value``.
    """
    if name is None:
        attr = getattr(fn, "__name__", None)
        name = attr if isinstance(attr, str) else "kernel"
    builder = FuncBuilder(name)
    values = [builder.arg(f"arg{i}", tuple(a.shape), "bf16") for i, a in enumerate(args)]
    result = fn(builder, *values)
    if not isinstance(result, Value):
        raise BuilderError(
            f"jitted kernel {name!r} must return a builder Value, got {type(result).__name__}"
        )
    return builder.render(result)


def _order_sim_inputs(args: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map the kernel's input arrays onto the simulator's (activation, weight, gamma).

    Convention (documented, no tracing): gamma is the unique rank-1 tensor; the
    activation is the FIRST rank>=2 tensor (flattened to rank-2 ``[M, K]`` for the
    simulator) and the weight is the SECOND. This matches both the rmsnorm_matmul
    ``(x, gamma, w)`` and the mlp ``(x, gamma, w1, ...)`` signatures deterministically
    (square matrices included, since the split is positional, not shape-based).
    """
    gamma = [a for a in args if a.ndim == 1]
    matrices = [a for a in args if a.ndim >= 2]
    if len(gamma) != 1:
        raise BuilderError(
            f"the simulator expects exactly one rank-1 gamma tensor, got {len(gamma)}"
        )
    if len(matrices) < 2:
        raise BuilderError(
            "the simulator expects an activation and a weight (rank >= 2), got "
            f"{len(matrices)} matrix tensor(s)"
        )
    activation = matrices[0].reshape(-1, matrices[0].shape[-1])
    return activation, matrices[1], gamma[0]


def _run(cmd: Sequence[str], *, stdin: str | None = None) -> str:
    """Run a built tool, returning stdout or raising loudly with both streams."""
    proc = subprocess.run(list(cmd), input=stdin, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{cmd[0]} failed (exit {proc.returncode}):\n"
            f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
        )
    return proc.stdout


def lower_and_run(
    fn: Callable[..., object],
    args: Sequence[np.ndarray],
    *,
    name: str | None = None,
) -> LoweredResult:
    """Lower a jitted kernel and run it end-to-end on the CPU simulator.

    Builds the MLIR text from ``fn``'s builder calls, runs the full
    ``tensixflow-opt`` pipeline -> ``tensixflow-translate`` for the plan JSON, and
    runs ``tensixflow-simulate`` over the ``.tft`` inputs for the result array.
    Raises ``FileNotFoundError`` (loudly) if a built binary is missing.
    """
    mlir_text = build_mlir(fn, args, name=name)

    opt_bin = find_opt_bin()
    translate_bin = find_translate_bin()
    sim_bin = harness.find_sim_bin()
    for binary in (opt_bin, translate_bin, sim_bin):
        if not binary.exists():
            raise FileNotFoundError(
                f"{binary} missing: build it with `nix develop --command bash -c "
                "'cmake --build build'` (the Python->C++ lowering bridge)."
            )

    with tempfile.TemporaryDirectory(prefix="tensixflow_lower_") as tmp:
        workdir = Path(tmp)
        mlir_path = workdir / "func.mlir"
        mlir_path.write_text(mlir_text)

        # Plan JSON: full pipeline -> schedule IR (module-level schedule attr) -> JSON.
        schedule_ir = _run([str(opt_bin), *PIPELINE_FLAGS, str(mlir_path)])
        plan = json.loads(_run([str(translate_bin), "--tensixflow-to-json"], stdin=schedule_ir))

        # Simulator result: tensixflow-simulate runs the same pipeline internally and
        # executes the REAL C++ tile simulator over the .tft inputs (harness I/O).
        activation, weight, gamma = _order_sim_inputs(args)
        in_path = workdir / "input.tft"
        out_path = workdir / "output.tft"
        harness.write_tensors(in_path, [activation, weight, gamma])
        # The simulator's one-line stdout summary is discarded; the result is read
        # back from the .tft output (the harness contract).
        _ = _run([str(sim_bin), "--mlir", str(mlir_path), "--input", str(in_path), "--output", str(out_path)])
        output = harness.read_tensor(out_path)

    return LoweredResult(plan=plan, output=output, mlir=mlir_text)
