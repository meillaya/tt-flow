"""Pytest for the TensixFlow DSL lowering driver.

The gate that proves the ``@tensixflow.jit`` body works end-to-end on a CPU-only
box: a jitted ``fused_rmsnorm_matmul(x, gamma, w)`` called with small NumPy arrays
builds the ``tensixflow`` MLIR text from its EXPLICIT builder calls, drives the
BUILT ``tensixflow-opt`` pipeline (the five ``--tensixflow-*`` passes) +
``tensixflow-translate`` for the plan JSON (with ``weight_strategy``), and triggers
the ``tensixflow-simulate`` C++ simulator (over the reused ``.tft`` I/O) for
a result array. It returns a :class:`LoweredResult` carrying BOTH. No device, and no
pass reimplemented in Python (the built binaries do the work).

Covers:
  * happy path: plan JSON has ``weight_strategy`` (+ fusion / shard_grid / cb_specs)
    AND a simulator result array of the right shape (bf16-representable, finite);
  * the built MLIR text is the real fused func (rmsnorm + matmul);
  * a rank-3 batched activation is flattened for the rank-2 simulator contract;
  * failure path: an unsupported op (no builder) raises a clear ``BuilderError``;
  * the eager fallback is preserved for non-NumPy calls (bring-up behavior).
"""

import numpy as np
import pytest

import tensixflow
from tensixflow import rounding as R
from tensixflow.builders import BuilderError
from tensixflow.lowering import LoweredResult

# Small tile-aligned shapes so the non-BLAS tiled simulator runs fast (the acceptance
# "small shapes" directive; M=K=N=64 = a 2x2 tile grid). Seed = the todo-39 base seed.
SEED = 0x20260727
M, K, N = 64, 64, 64


def _inputs(m: int = M, k: int = K, n: int = N) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Uniform fp32 with a full mantissa (NOT bf16-representable), mirroring the
    # harness/todo-39 ranges: activation/weight in [-0.5, 0.5], gamma in [0.5, 1.5].
    rng = np.random.default_rng(SEED)
    x = rng.uniform(-0.5, 0.5, (m, k)).astype(np.float32)
    gamma = rng.uniform(0.5, 1.5, (k,)).astype(np.float32)
    w = rng.uniform(-0.5, 0.5, (k, n)).astype(np.float32)
    return x, gamma, w


@tensixflow.jit
def fused_rmsnorm_matmul(b, x, gamma, w):
    # The jitted kernel IS the builder body: explicit builder calls, no tracing.
    return b.matmul(b.rmsnorm(x, gamma), w)


@pytest.fixture(scope="module")
def lowered() -> LoweredResult:
    """Lower + run the jitted kernel once (loud if a built binary is missing)."""
    x, gamma, w = _inputs()
    return fused_rmsnorm_matmul(x, gamma, w)


def test_lowering_driver_public_api_is_importable():
    # Given the lowering driver (a submodule, like harness/golden -- kept out of the
    # package __init__ to avoid an import cycle)
    # When importing its public API
    # Then the result type + driver functions ARE exposed (and the jit entrypoint too)
    from tensixflow import lowering as L

    assert L.LoweredResult is LoweredResult
    assert callable(L.lower_and_run)
    assert callable(L.build_mlir)
    assert callable(L.should_lower)
    assert len(L.PIPELINE_FLAGS) == 5
    assert callable(tensixflow.jit)


def test_jit_call_with_numpy_returns_a_lowered_result(lowered):
    # Given a jitted kernel called with NumPy arrays (the fixture)
    # Then the call LOWERS (does not run the Python body) and returns a LoweredResult
    assert isinstance(lowered, LoweredResult)


def test_plan_json_has_weight_strategy(lowered):
    # The acceptance field: the plan JSON carries weight_strategy, a valid strategy.
    assert "weight_strategy" in lowered.plan
    assert lowered.plan["weight_strategy"] in {"replicate", "shard", "multicast"}


def test_plan_json_has_fusion_grid_and_cbs(lowered):
    # The schedule attr (module-level, persists through the pipeline) drives the rest
    # of the plan: the fusion decision, the 8x8 shard grid, and the 3 circular buffers.
    assert lowered.plan["fusion"] == "fused_rmsnorm_matmul"
    assert lowered.plan["shard_grid"] == {"grid_x": 8, "grid_y": 8}
    assert len(lowered.plan["cb_specs"]) == 3


def test_simulator_result_has_expected_shape(lowered):
    # The simulator result array: fp32, shape [M, N] = matmul([M,K] @ [K,N]), finite.
    assert isinstance(lowered.output, np.ndarray)
    assert lowered.output.shape == (M, N)
    assert lowered.output.dtype == np.float32
    assert np.isfinite(lowered.output).all()


def test_simulator_output_is_bf16_representable(lowered):
    # Non-vacuity (mirrors the harness): the REAL C++ simulator rounds CB storage to
    # bf16, so every output value equals its own bf16 round (the run actually happened).
    assert np.array_equal(R.to_bf16(lowered.output), lowered.output)


def test_built_mlir_is_the_fused_func(lowered):
    # The driver built the tensixflow.func text from the kernel's builder calls.
    assert "tensixflow.func @fused_rmsnorm_matmul" in lowered.mlir
    assert "tensixflow.rmsnorm" in lowered.mlir
    assert "tensixflow.matmul" in lowered.mlir


def test_rank3_batch_activation_is_flattened_for_the_simulator():
    # Given a batched [1, M, K] activation (the examples' rank-3 convention)
    x, gamma, w = _inputs()
    x3 = x.reshape(1, M, K)
    # When lowered (the driver flattens to rank-2 [M, K] for the simulator)
    res = fused_rmsnorm_matmul(x3, gamma, w)
    # Then the result still carries M*N elements (the batch dim folds into M)
    assert res.output.size == M * N


def test_unsupported_op_raises_clear_no_builder_error():
    # Given a jitted kernel that calls a non-existent builder op
    @tensixflow.jit
    def bad_kernel(b, x, gamma, w):
        return b.softmax(x)

    x, gamma, w = _inputs()
    # When called with NumPy arrays (the lowering trigger)
    # Then a clear BuilderError names the missing op (not a confusing crash)
    with pytest.raises(BuilderError, match=r"no builder for op 'softmax'"):
        bad_kernel(x, gamma, w)


def test_eager_fallback_preserved_for_non_numpy_calls():
    # Given a jitted kernel
    @tensixflow.jit
    def add(a, c):
        return a + c

    # When called with non-NumPy args (scalars / kwargs)
    # Then it runs eagerly (bring-up behavior untouched by the lowering driver)
    assert add(2, 3) == 5
    assert add(a=10, c=-4) == 6
