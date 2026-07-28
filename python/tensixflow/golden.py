"""TensixFlow NumPy golden reference (op-semantics) — the FP32-EXACT reference
the simulator's bf16 tiled path (todo 39) is measured against (draft D14).

THE D14 CONTRACT — the simulator and the golden are DELIBERATELY at different
precision so the PCC bar (todo 41, >= 0.9875) measures REAL bf16-spatial-vs-exact
divergence rather than collapsing to ~1.0:

* GOLDEN (this module) = fp32-EXACT: fp32 accumulate, fp32 intermediates, NO bf16
  rounding of intermediates (round only the final compared output, if at all).
  Every op below computes in float32 and keeps EVERY intermediate in float32.
* SIMULATOR (``lib/tensixflow/Sim/TileSimulator.cpp``) = bf16 SPATIAL: rounds CB
  storage and fused intermediates to bf16 (round-to-nearest-even, todo 38), fp32
  accumulate within a tile, tiled accumulation order.

The deliberate precision gap is what makes
``PCC(simulator_bf16_tiled, golden_fp32_exact)`` a meaningful bar: if the golden
ALSO rounded intermediates to bf16, the two paths would be identical and PCC would
collapse to ~1.0 (VACUOUS). The fused helpers therefore HOLD their intermediates in
fp32 — the rmsnorm output feeding the matmul is NEVER bf16-rounded here. That is
the structural non-vacuity invariant the harness (todo 41) asserts (it shows the
golden's intermediate is not bf16-representable via ``rounding.to_bf16``).

Op semantics mirror the dialect (``include/tensixflow/TensixflowOps.td``) and the
simulator's op math — the SAME formulas, minus tiling / circular buffers / bf16:

* ``rmsnorm(x, gamma, eps) = x * rsqrt(mean_k(x^2) + eps) * gamma``  (per-row over
  the last dim K; fp32 reduce-square-sum, fp32 rsqrt, fp32 scale).
* ``matmul(a, b) = a @ b``                                            (fp32 matmul;
  matches numpy ``@`` exactly).
* ``gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))``                      (exact erf
  form, matching the simulator's geluTile; libm erf — the same C erf as std::erf).
* ``silu(x) = x / (1 + exp(-x))``                                     (SiLU/swish;
  the dialect's gelu op models GELU/SiLU — provided for completeness).
* ``add(a, b)`` / ``residual(x, r)`` = elementwise fp32 sum.

Fused composition (verified against the dialect):

* ``fused_rmsnorm_matmul(x, gamma, w, eps) = matmul(rmsnorm(x, gamma, eps), w)``
  (``TensixflowFusedRmsnormMatmulOp``: ``result = matmul(rmsnorm(input, gamma),
  weight)``; the rmsnorm intermediate is held in fp32, NOT bf16-rounded.)
* ``fused_mlp(x, gamma, w1, b1, w2, residual, eps)
      = (gelu(matmul(rmsnorm(x, gamma, eps), w1) + b1) @ w2) + residual``
  (``TensixflowFusedMlpOp``: ``result = (gelu(matmul(rmsnorm(input, gamma), w1) +
  b1) @ w2) + residual``; operands ``input, gamma, w1, b1, w2, residual``. ALL five
  intermediates — rmsnorm out, matmul1 out, bias-add out, gelu out, matmul2 out —
  are held in fp32, NOT bf16-rounded.)

CPU / numpy only. No GPU, no tiling, no circular buffers (the simulator models
those). Dependency-free: numpy + stdlib (``math.erf``) — numpy has no vectorized
erf and the verification env is numpy+pytest only (no scipy).
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "rmsnorm",
    "matmul",
    "gelu",
    "silu",
    "add",
    "residual",
    "fused_rmsnorm_matmul",
    "fused_mlp",
]

_F32 = np.float32


def _f32(x: object) -> np.ndarray:
    """Boundary parse: coerce to a float32 ndarray. Every golden input enters as
    fp32 and EVERY intermediate stays fp32 — there is NO bf16 rounding anywhere in
    this module (that is the simulator's job, todo 39)."""
    return np.asarray(x, dtype=_F32)


def rmsnorm(x: object, gamma: object, eps: float = 1e-6) -> np.ndarray:
    """RMSNorm over the last dim K: ``x * rsqrt(mean_k(x^2) + eps) * gamma``.

    fp32 reduce-square-sum, fp32 rsqrt, fp32 scale (the simulator's rmsnormTile
    minus the bf16 round of its output). ``gamma`` is 1-D of size K; the result has
    the same shape as ``x``.
    """
    x = _f32(x)
    gamma = _f32(gamma)
    k = x.shape[-1]
    mean_sq = np.sum(x * x, axis=-1, keepdims=True) / _F32(k)  # fp32 reduce-square-sum
    rsqrt = _F32(1.0) / np.sqrt(mean_sq + _F32(eps))           # fp32 rsqrt(mean + eps)
    return x * rsqrt * gamma                                    # fp32, NO bf16 round


def matmul(a: object, b: object) -> np.ndarray:
    """fp32 matmul ``a @ b``; matches numpy ``@`` exactly (it IS ``@``). For ``a``
    of shape ``[..., M, K]`` and ``b`` of shape ``[K, N]`` the result is
    ``[..., M, N]`` (the simulator's tiledMatmul minus tiling / bf16)."""
    return _f32(a) @ _f32(b)


def _erf(x: np.ndarray) -> np.ndarray:
    """Elementwise erf -> float32 via libm ``math.erf`` — the SAME C erf the
    simulator's ``std::erf`` calls. numpy exposes no vectorized erf; this keeps the
    golden dependency-free (numpy + stdlib, CPU only)."""
    flat = x.ravel()
    return np.array([math.erf(float(v)) for v in flat], dtype=_F32).reshape(x.shape)


def gelu(x: object) -> np.ndarray:
    """Exact-erf GELU ``0.5 * x * (1 + erf(x / sqrt(2)))`` (the simulator's
    geluTile minus the bf16 round). Equivalent to ``x * Phi(x)`` (standard-normal
    CDF); this is the exact form, NOT the tanh approximation."""
    x = _f32(x)
    arg = x / _F32(math.sqrt(2.0))
    return (_F32(0.5) * x * (_F32(1.0) + _erf(arg))).astype(_F32)


def silu(x: object) -> np.ndarray:
    """SiLU / swish ``x * sigmoid(x) = x / (1 + exp(-x))`` (fp32)."""
    x = _f32(x)
    return (x / (_F32(1.0) + np.exp(-x))).astype(_F32)


def add(a: object, b: object) -> np.ndarray:
    """Elementwise fp32 sum (the dialect's ``tensixflow.add`` is same-shape; numpy
    broadcasting is a harmless superset for a broadcastable bias)."""
    return _f32(a) + _f32(b)


def residual(x: object, r: object) -> np.ndarray:
    """Residual connection: elementwise fp32 ``x + r`` (semantically an add)."""
    return _f32(x) + _f32(r)


def fused_rmsnorm_matmul(x: object, gamma: object, w: object, eps: float = 1e-6) -> np.ndarray:
    """``matmul(rmsnorm(x, gamma, eps), w)`` — the dialect's fused_rmsnorm_matmul,
    with the rmsnorm intermediate HELD IN FP32 (NOT bf16-rounded): the D14
    non-vacuity invariant. The simulator bf16-rounds this same intermediate."""
    normed = rmsnorm(x, gamma, eps)  # fp32 intermediate (the simulator rounds it to bf16)
    return matmul(normed, w)


def fused_mlp(
    x: object,
    gamma: object,
    w1: object,
    b1: object,
    w2: object,
    residual: object,
    eps: float = 1e-6,
) -> np.ndarray:
    """``(gelu(matmul(rmsnorm(x, gamma, eps), w1) + b1) @ w2) + residual`` — the
    dialect's fused_mlp (operands ``input, gamma, w1, b1, w2, residual``). ALL five
    intermediates are HELD IN FP32 (NOT bf16-rounded); the simulator rounds each."""
    normed = rmsnorm(x, gamma, eps)    # fp32 intermediate
    hidden = matmul(normed, w1)        # fp32 intermediate
    biased = add(hidden, b1)           # bias add, fp32 intermediate
    activated = gelu(biased)           # fp32 intermediate
    projected = matmul(activated, w2)  # fp32 intermediate
    return add(projected, residual)    # residual connection, fp32
