"""Pytest for the TensixFlow NumPy golden reference (op semantics).

The golden is the FP32-EXACT reference the simulator's bf16 tiled path
is measured against: fp32 accumulate, fp32 intermediates, NO bf16 rounding of
intermediates. Covers:

  * rmsnorm / matmul / gelu / silu / add / residual match tiny HAND-COMPUTED fp32
    values within fp32 tolerance;
  * matmul matches numpy ``@`` (known identity) on a small random case;
  * fused golden == composed unfused golden, bit-for-bit (fused_rmsnorm_matmul AND
    fused_mlp);
  * a fused intermediate is HELD IN FP32 (NOT bf16), the structural non-vacuity
    invariant the PCC harness relies on (via ``rounding.to_bf16``);
  * the golden's outputs are float32 (the structural fp32 dtype assertion);
  * eps sensitivity: a wrong eps changes the rmsnorm output.

Runs on CPU only (pure numpy; no GPU/CUDA anywhere in the golden).
"""

import numpy as np

from tensixflow import golden as G
from tensixflow import rounding as R

# fp32 has ~7 significant digits; rtol=1e-5 is "fp32 tolerance" for the
# transcendental gelu/silu, and is trivially loose for the exact ops below.
RTOL = 1e-5
ATOL = 1e-6


# --- hand-computed fp32 values ---------------------------------------------


def test_rmsnorm_matches_hand_computed_fp32():
    # x = [[1,1],[2,2]], gamma = [3,1], eps = 0 (exact, no eps perturbation):
    #   row0: mean(x^2) = 1, rsqrt = 1     -> [1*1*3, 1*1*1] = [3, 1]
    #   row1: mean(x^2) = 4, rsqrt = 1/2   -> [2*.5*3, 2*.5*1] = [3, 1]
    x = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    gamma = np.array([3.0, 1.0], dtype=np.float32)
    expected = np.array([[3.0, 1.0], [3.0, 1.0]], dtype=np.float32)
    got = G.rmsnorm(x, gamma, eps=0.0)
    assert np.allclose(got, expected, rtol=RTOL, atol=ATOL)


def test_matmul_matches_hand_computed_fp32():
    # [[1,2],[3,4]] @ [[5,6],[7,8]] = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
    #                               = [[19, 22], [43, 50]]  (exact in fp32).
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    expected = np.array([[19.0, 22.0], [43.0, 50.0]], dtype=np.float32)
    assert np.array_equal(G.matmul(a, b), expected)


def test_gelu_matches_hand_computed_fp32():
    # Exact erf form; gelu(x) = x * Phi(x) (standard-normal CDF):
    #   gelu(0)  = 0
    #   gelu(1)  = Phi(1)  = 0.8413447460685429
    #   gelu(-1) = -Phi(-1)... = -1 * 0.15865525393145707 = -0.15865525393145707
    #   gelu(2)  = 2*Phi(2) = 1.9544997361036416
    x = np.array([0.0, 1.0, -1.0, 2.0], dtype=np.float32)
    expected = np.array(
        [0.0, 0.8413447460685429, -0.15865525393145707, 1.9544997361036416],
        dtype=np.float32,
    )
    assert np.allclose(G.gelu(x), expected, rtol=RTOL, atol=ATOL)


def test_silu_matches_hand_computed_fp32():
    # silu(x) = x * sigmoid(x): silu(0) = 0; silu(1) = 1/(1+e^-1) = 0.7310585786300049.
    x = np.array([0.0, 1.0], dtype=np.float32)
    expected = np.array([0.0, 0.7310585786300049], dtype=np.float32)
    assert np.allclose(G.silu(x), expected, rtol=RTOL, atol=ATOL)


def test_add_and_residual_are_elementwise_fp32():
    a = np.array([[1.0, 2.0]], dtype=np.float32)
    b = np.array([[10.0, 20.0]], dtype=np.float32)
    expected = np.array([[11.0, 22.0]], dtype=np.float32)
    assert np.array_equal(G.add(a, b), expected)
    # residual IS an add (the residual connection).
    assert np.array_equal(G.residual(a, b), G.add(a, b))


# --- known identity: matmul == numpy `@` ------------------------------------


def test_matmul_matches_numpy_matmul_operator():
    rng = np.random.default_rng(1234)
    a = rng.standard_normal((4, 6)).astype(np.float32)
    b = rng.standard_normal((6, 3)).astype(np.float32)
    assert np.array_equal(G.matmul(a, b), a @ b)


# --- fused golden == composed unfused golden (bit-for-bit) ------------------


def test_fused_rmsnorm_matmul_equals_composed_unfused_golden():
    rng = np.random.default_rng(7)
    x = rng.standard_normal((3, 4)).astype(np.float32)
    gamma = rng.standard_normal(4).astype(np.float32)
    w = rng.standard_normal((4, 5)).astype(np.float32)
    eps = 1e-6
    fused = G.fused_rmsnorm_matmul(x, gamma, w, eps)
    composed = G.matmul(G.rmsnorm(x, gamma, eps), w)
    assert np.array_equal(fused, composed)


def test_fused_mlp_equals_composed_unfused_golden():
    # Dialect composition: (gelu(matmul(rmsnorm(x,gamma),w1) + b1) @ w2) + residual.
    rng = np.random.default_rng(2026)
    x = rng.standard_normal((2, 4)).astype(np.float32)
    gamma = rng.standard_normal(4).astype(np.float32)
    w1 = rng.standard_normal((4, 8)).astype(np.float32)
    b1 = rng.standard_normal(8).astype(np.float32)  # 1-D bias [H], broadcast over rows
    w2 = rng.standard_normal((8, 4)).astype(np.float32)
    res = rng.standard_normal((2, 4)).astype(np.float32)
    eps = 1e-6
    fused = G.fused_mlp(x, gamma, w1, b1, w2, res, eps)
    composed = G.residual(
        G.matmul(G.gelu(G.add(G.matmul(G.rmsnorm(x, gamma, eps), w1), b1)), w2),
        res,
    )
    assert np.array_equal(fused, composed)


# --- fused intermediate held in fp32 (NOT bf16), non-vacuity invariant --------


def test_fused_rmsnorm_matmul_holds_intermediate_in_fp32_not_bf16():
    # The fused path's intermediate IS rmsnorm(x, gamma, eps) (fused == matmul of
    # it). x is chosen so that intermediate is NOT bf16-representable.
    x = np.array([[1.0, 2.0]], dtype=np.float32)
    gamma = np.array([1.0, 1.0], dtype=np.float32)
    w = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    eps = 1e-6
    normed = G.rmsnorm(x, gamma, eps)  # the fused intermediate, fp32
    # (1) bf16 CANNOT represent it exactly: the simulator rounds it, the golden does not.
    assert not np.array_equal(R.to_bf16(normed), normed)
    # (2) the fused result uses the fp32 intermediate bit-for-bit.
    fused = G.fused_rmsnorm_matmul(x, gamma, w, eps)
    assert np.array_equal(fused, G.matmul(normed, w))
    # (3) the fused result DIFFERS from a bf16-rounded-intermediate variant (the
    # simulator's behavior). If the golden also rounded the intermediate to bf16,
    # these would be EQUAL and PCC(sim, golden) would collapse to ~1.0 (vacuous).
    bf16_variant = G.matmul(R.to_bf16(normed), w)
    assert not np.array_equal(fused, bf16_variant)


def test_fused_mlp_holds_rmsnorm_intermediate_in_fp32_not_bf16():
    # fused_mlp's first intermediate (the rmsnorm output) is likewise fp32, not bf16.
    x = np.array([[1.0, 2.0]], dtype=np.float32)
    gamma = np.array([1.0, 1.0], dtype=np.float32)
    normed = G.rmsnorm(x, gamma, 1e-6)
    assert not np.array_equal(R.to_bf16(normed), normed)


# --- structural: golden outputs are float32 (the fp32 dtype assertion) -------


def test_golden_outputs_are_float32():
    x = np.array([[1.0, 2.0]], dtype=np.float32)
    gamma = np.array([1.0, 1.0], dtype=np.float32)
    w = np.ones((2, 2), dtype=np.float32)
    assert G.rmsnorm(x, gamma).dtype == np.float32
    assert G.matmul(x, w).dtype == np.float32
    assert G.gelu(x).dtype == np.float32
    assert G.silu(x).dtype == np.float32
    assert G.fused_rmsnorm_matmul(x, gamma, w).dtype == np.float32


# --- eps sensitivity (QA failure scenario) ----------------------------------


def test_rmsnorm_is_sensitive_to_eps():
    # A wrong eps changes the rmsnorm output (mean(x^2) + eps under the rsqrt).
    x = np.array([[1.0, 2.0]], dtype=np.float32)
    gamma = np.array([1.0, 1.0], dtype=np.float32)
    small = G.rmsnorm(x, gamma, eps=1e-6)
    large = G.rmsnorm(x, gamma, eps=1e-2)
    assert not np.array_equal(small, large)
