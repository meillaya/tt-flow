"""TensixFlow NumPy golden reference (op semantics).

The golden is the fp32-exact reference the simulator's bf16 tiled path is
measured against. Every op below computes in float32 and keeps every
intermediate in float32. The simulator rounds to bf16, so the two paths differ
on purpose and PCC measures the divergence. The fused helpers therefore hold
their intermediates in fp32: the rmsnorm output feeding the matmul is never
bf16-rounded here.

Op semantics mirror the dialect and the simulator's op math, without tiling,
circular buffers, or bf16 rounding.
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
    fp32 and every intermediate stays fp32; there is no bf16 rounding in this
    module (that is the simulator's job)."""
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
    """Elementwise erf -> float32 via libm ``math.erf``, the same C erf the
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
    """``matmul(rmsnorm(x, gamma, eps), w)``: the dialect's fused_rmsnorm_matmul,
    with the rmsnorm intermediate HELD IN FP32 (NOT bf16-rounded): the
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
    """``(gelu(matmul(rmsnorm(x, gamma, eps), w1) + b1) @ w2) + residual``: the
    dialect's fused_mlp (operands ``input, gamma, w1, b1, w2, residual``). ALL five
    intermediates are HELD IN FP32 (NOT bf16-rounded); the simulator rounds each."""
    normed = rmsnorm(x, gamma, eps)    # fp32 intermediate
    hidden = matmul(normed, w1)        # fp32 intermediate
    biased = add(hidden, b1)           # bias add, fp32 intermediate
    activated = gelu(biased)           # fp32 intermediate
    projected = matmul(activated, w2)  # fp32 intermediate
    return add(projected, residual)    # residual connection, fp32
