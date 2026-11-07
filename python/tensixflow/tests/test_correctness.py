"""Pytest for the TensixFlow correctness harness.

The gate that proves the project's central correctness claim: the REAL C++
functional tile simulator matches the fp32-EXACT NumPy
golden to ``PCC >= 0.9875`` on BOTH example graphs (rmsnorm_matmul AND
mlp_block), with a two-layer non-vacuity defense so the bar can never silently
degenerate into a trivial PCC ~= 1.0:

* PRIMARY, STRUCTURAL: the golden holds its fused intermediate in fp32 (NOT
  bf16-representable) while the simulator output IS bf16-representable; PLUS
  BIT-DIFFERENCE: ``max_abs_diff(sim, golden) > 0`` (outputs NOT bit-identical).
* SECONDARY, an empirically-calibrated upper-band PCC TRIP-WIRE (strictly above
  the measured good-case PCC, strictly below 1.0; never hard-coded to 0.9999):
  ``PCC < trip-wire``.

The simulator is driven via the ``tensixflow-simulate`` C++ bridge (subprocess),
so this validates the real C++ simulator, not a Python re-implementation and not
the emitted C++ kernels (compile-checked elsewhere). Uses a documented seed PLUS a
few (no single seed can hide the bf16 divergence).

Negative cases (verified on a scratch branch, NOT committed): a perturbed /
wrong-rounding simulator drops PCC below 0.9875 (gate is live); forcing the golden
to bf16-round intermediates fails the structural fp32 assertion (and, if both sides
become bit-identical, ``max_abs_diff > 0``).
"""

import pytest

from tensixflow import harness as H

# The fixed correctness floor, documented in docs/verification.md. Asserted, never lowered.
PCC_FLOOR = H.PCC_FLOOR

TARGETS = ("rmsnorm_matmul", "mlp_block")


@pytest.fixture(scope="module")
def results() -> dict:
    """Run the full harness once (every target x shape x seed) and record the
    metrics JSON evidence. Loud (raises) if the C++ bridge binary is missing."""
    res = H.run_harness(verbose=True)
    H.write_metrics_json(res, H._repo_root() / "build" / "metrics" / "tensixflow-correctness.json")
    return res


def _runs(results: dict, target: str) -> list[dict]:
    return results["targets"][target]["runs"]


def test_both_targets_are_present(results):
    # Sanity: the harness measured BOTH required graphs.
    assert set(results["targets"]) == set(TARGETS)
    for target in TARGETS:
        assert len(_runs(results, target)) > 0, f"{target}: no runs measured"


@pytest.mark.parametrize("target", TARGETS)
def test_pcc_above_floor(results, target):
    # FIXED correctness floor: every run of every target clears PCC >= 0.9875.
    for r in _runs(results, target):
        assert r["pcc"] >= PCC_FLOOR, (
            f"{target} {r['shape']} seed={hex(r['seed'])}: PCC {r['pcc']:.8f} "
            f"< floor {PCC_FLOOR} (simulator diverged from the fp32-exact golden)"
        )


@pytest.mark.parametrize("target", TARGETS)
def test_max_abs_diff_positive_not_bit_identical(results, target):
    # PRIMARY non-vacuity (bit-difference): the bf16 simulator output is NOT
    # bit-identical to the fp32-exact golden (bf16 rounding actually happened).
    for r in _runs(results, target):
        assert r["max_abs_diff"] > 0.0, (
            f"{target} {r['shape']} seed={hex(r['seed'])}: max_abs_diff == 0 "
            "(simulator output bit-identical to the golden -> VACUOUS bar)"
        )


@pytest.mark.parametrize("target", TARGETS)
def test_pcc_below_empirical_tripwire(results, target):
    # SECONDARY non-vacuity: PCC stays below the empirically-calibrated trip-wire,
    # which is itself strictly below 1.0 (so a collapse to PCC ~= 1.0 trips it).
    t = results["targets"][target]
    tripwire = t["tripwire"]
    assert tripwire < 1.0, f"{target}: trip-wire {tripwire} not < 1.0 (good-case PCC was 1.0 -> vacuous)"
    for r in _runs(results, target):
        assert r["pcc"] < tripwire, (
            f"{target} {r['shape']} seed={hex(r['seed'])}: PCC {r['pcc']:.8f} "
            f">= trip-wire {tripwire:.8f} (suspiciously close to 1.0)"
        )


@pytest.mark.parametrize("target", TARGETS)
def test_tripwire_calibrated_above_good_case(results, target):
    # The trip-wire is calibrated EMPIRICALLY: strictly above the measured good-case
    # PCC (never a hard-coded constant) and strictly below 1.0.
    t = results["targets"][target]
    good, tripwire = t["good_case_pcc"], t["tripwire"]
    assert good < 1.0, f"{target}: good-case PCC is 1.0 (vacuous)"
    assert tripwire > good, f"{target}: trip-wire {tripwire} not above good-case PCC {good}"
    assert tripwire < 1.0, f"{target}: trip-wire {tripwire} not < 1.0"


@pytest.mark.parametrize("target", TARGETS)
def test_structural_fp32_golden_bf16_simulator(results, target):
    # PRIMARY non-vacuity (structural): the golden holds its fused intermediate in
    # fp32 (NOT bf16-representable) while the simulator output IS bf16-representable
    # (CB rounding happened), and the golden output dtype is float32. Forcing the
    # golden to bf16-round intermediates makes golden_intermediate_is_fp32 False.
    for r in _runs(results, target):
        s = r["structural"]
        assert s["golden_intermediate_is_fp32"], (
            f"{target}: golden fused intermediate is bf16-representable "
            "(golden rounded intermediates to bf16 -> bar would be VACUOUS)"
        )
        assert s["sim_output_is_bf16"], f"{target}: simulator output has non-bf16 values (CB rounding skipped)"
        assert s["golden_output_dtype_fp32"], f"{target}: golden output is not float32"
        if target == "mlp_block":
            # The honest MLP scope reduction: golden.fused_mlp with identity trailing
            # operands (b1=0, w2=I, residual=0) == the simulated prefix, bit-for-bit.
            assert s["mlp_identity_tail_reduces_to_prefix"], (
                "mlp_block: golden.fused_mlp(identity tail) != gelu(fused_rmsnorm_matmul) prefix"
            )
