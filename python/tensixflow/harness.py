"""TensixFlow correctness harness (todo 41, draft D14) — drives the REAL C++
functional tile simulator (todo 39) over the two example graphs and scores its
bf16-tiled output against the fp32-EXACT NumPy golden (todo 40).

THE D14 CONTRACT under test (why the bar is meaningful, not vacuous):

* SIMULATOR (``lib/tensixflow/Sim/TileSimulator.cpp``, C++) = bf16 SPATIAL: rounds
  CB storage + fused intermediates to bf16 (round-to-nearest-even, todo 38), fp32
  accumulate within a tile, tiled accumulation order.
* GOLDEN (``golden.py``) = fp32-EXACT: fp32 accumulate, fp32 intermediates, NO bf16
  rounding of intermediates.

``PCC(simulator_bf16_tiled, golden_fp32_exact)`` therefore measures the REAL
bf16-spatial-vs-exact cost of fusion + tiled execution (arXiv:2606.09879). The
deliberate precision gap is what keeps the bar honest: if both sides were identical
PCC would collapse to ~1.0 (VACUOUS).

PASS iff, on BOTH targets (rmsnorm_matmul AND mlp_block):

* ``PCC >= 0.9875`` (FIXED correctness floor), AND
* the two-layer non-vacuity defense holds:
  - PRIMARY: (a) STRUCTURAL — the golden holds its fused intermediate in fp32 (not
    bf16-representable via ``rounding.to_bf16``) while the simulator output IS
    bf16-representable (CB rounding happened); PLUS (b) BIT-DIFFERENCE —
    ``max_abs_diff(sim, golden) > 0`` (the outputs are NOT bit-identical; catches
    the both-identical vacuous case deterministically, no fragile threshold);
  - SECONDARY: an upper-band PCC TRIP-WIRE calibrated EMPIRICALLY — measure the
    good-case PCC over the documented seeds, set the trip-wire strictly ABOVE it but
    below 1.0 (midpoint to 1.0), record both, and assert ``PCC < trip-wire``. Never
    hard-coded to 0.9999 (a clean software bf16 model can exceed that and would
    false-fail a correct implementation).

PYTHON -> C++ BRIDGE: the simulator is C++; this harness cannot call it directly.
It writes the input tensors to a self-describing binary file (.tft), invokes the
``tensixflow-simulate`` tool (``tools/tensixflow-simulate``) via subprocess — which
runs the full infer-layout -> fuse -> plan -> allocate -> lower pipeline on the
example func and then ``runTileSimulator`` — and reads the output tensor back. This
validates the REAL C++ simulator, NOT a Python re-implementation, and does NOT touch
the emitted C++ kernels (compile-checked elsewhere).

INPUT SIZES (pragmatic, documented): the example shapes ([1,2048,4096]@[4096,12288])
are far too large for the non-BLAS tiled simulator in a test. The pipeline runs on
the example funcs to get the schedule-IR STRUCTURE (fusion + CB allocation); the
simulator is fed FEASIBLE reduced shapes (M,K,N/H in 32..64, tile 32x32). The PCC
property is size-independent (it is the bf16-vs-fp32 divergence). The simulator
derives its per-tile grid from the input shapes it is given and ignores the IR trip
counts (built from the example shapes), so the two are reconciled by design.

MLP SCOPE (documented honestly): the fused_mlp schedule IR (todo 32 lowering)
denotes the compute PREFIX rmsnorm -> matmul(w1) -> gelu; the simulator (todo 39,
``SimInputs`` = activation/weight/gamma only) executes exactly that prefix and does
NOT model the trailing bias(b1)/matmul(w2)/residual stages. The mlp_block golden is
therefore ``golden.fused_mlp`` evaluated with those trailing operands as IDENTITY
operations (b1=0, w2=I, residual=0), which reduces it bit-for-bit to the prefix the
simulator computes — asserted explicitly in ``structural_check``. This is forced by
the simulator's documented scope (todo 39), which this harness must NOT modify.
"""

from __future__ import annotations

import json
import os
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from tensixflow import golden as G
from tensixflow import rounding as R

__all__ = [
    "PCC_FLOOR",
    "SEEDS",
    "SHAPES",
    "EXAMPLES",
    "find_sim_bin",
    "write_tensors",
    "read_tensor",
    "make_inputs",
    "compute_golden",
    "compute_metrics",
    "structural_check",
    "calibrate_tripwire",
    "run_harness",
    "write_metrics_json",
]

# FIXED correctness floor (arXiv:2606.09879 PCC > 98.75%). Never lowered.
PCC_FLOOR: float = 0.9875

# RMSNorm epsilon (the dialect default; the examples elide it -> 1e-6).
EPS: float = 1e-6

# Documented seeds: the first two are project constants (todo-39 simulator-test base
# seed; todo-38 rounding-consistency seed), the rest are arbitrary — a documented
# seed PLUS a few, so no single seed can hide the bf16 divergence.
SEEDS: tuple[int, ...] = (0x20260727, 0x5EED2026, 0xC0FFEE, 20260727)

# Feasible reduced shapes (documented). rmsnorm_matmul: (M, K, N); mlp_block:
# (M, K, H). Each target runs a tile-aligned shape (64, 2x2 tiles) AND a
# non-tile-aligned shape (exercises the simulator's tile padding).
SHAPES: dict[str, list[tuple[int, int, int]]] = {
    "rmsnorm_matmul": [(64, 64, 64), (48, 64, 40)],
    "mlp_block": [(64, 64, 64), (48, 64, 48)],
}

_TENSOR_MAGIC = 0x54465401  # "TFT\x01" — the .tft record magic (shared with C++).


def _repo_root() -> Path:
    # python/tensixflow/harness.py -> parents[2] == repo root.
    return Path(__file__).resolve().parents[2]


# Example graphs under test (the pipeline runs on THESE to get the schedule IR).
EXAMPLES: dict[str, Path] = {
    "rmsnorm_matmul": _repo_root() / "examples" / "rmsnorm_matmul.mlir",
    "mlp_block": _repo_root() / "examples" / "mlp_block.mlir",
}


def find_sim_bin() -> Path:
    """Locate the ``tensixflow-simulate`` bridge binary (env override honored)."""
    env = os.environ.get("TENSIXFLOW_SIMULATE_BIN")
    if env:
        return Path(env)
    return _repo_root() / "build" / "bin" / "tensixflow-simulate"


# --- .tft tensor IO (self-describing binary; mirrors the C++ driver) ----------


def write_tensors(path: Path, arrays: list[np.ndarray]) -> None:
    """Write ``arrays`` as concatenated .tft records (magic, rank, shape, fp32)."""
    with open(path, "wb") as f:
        for a in arrays:
            a = np.ascontiguousarray(a, dtype="<f4")
            f.write(struct.pack("<II", _TENSOR_MAGIC, a.ndim))
            f.write(struct.pack(f"<{a.ndim}q", *a.shape))
            f.write(a.tobytes())


def read_tensor(path: Path) -> np.ndarray:
    """Read the first .tft record from ``path`` as a float32 ndarray."""
    with open(path, "rb") as f:
        magic, rank = struct.unpack("<II", f.read(8))
        if magic != _TENSOR_MAGIC:
            raise ValueError(f"{path}: bad .tft magic {magic:#x}")
        shape = struct.unpack(f"<{rank}q", f.read(8 * rank))
        numel = int(np.prod(shape)) if rank else 1
        return np.frombuffer(f.read(4 * numel), dtype="<f4").reshape(shape).astype(np.float32)


# --- input generation + golden ------------------------------------------------


def make_inputs(target: str, shape: tuple[int, int, int], seed: int) -> dict[str, np.ndarray]:
    """Generate the simulator inputs (activation, weight, gamma) for one run.

    Uniform random fp32 with a full mantissa — NOT bf16-representable, so the
    simulator's round-on-push actually changes them (the non-vacuity invariant).
    Ranges mirror todo-39's verified simulator test (activation/weight in
    [-0.5, 0.5], gamma in [0.5, 1.5]).
    """
    rng = np.random.default_rng(seed)
    if target == "rmsnorm_matmul":
        m, k, n = shape
        return {
            "activation": rng.uniform(-0.5, 0.5, (m, k)).astype(np.float32),
            "weight": rng.uniform(-0.5, 0.5, (k, n)).astype(np.float32),
            "gamma": rng.uniform(0.5, 1.5, (k,)).astype(np.float32),
        }
    if target == "mlp_block":
        m, k, h = shape
        return {
            "activation": rng.uniform(-0.5, 0.5, (m, k)).astype(np.float32),
            "weight": rng.uniform(-0.5, 0.5, (k, h)).astype(np.float32),  # w1
            "gamma": rng.uniform(0.5, 1.5, (k,)).astype(np.float32),
        }
    raise ValueError(f"unknown target {target!r}")


def compute_golden(target: str, inputs: dict[str, np.ndarray], shape: tuple[int, int, int]) -> np.ndarray:
    """The fp32-EXACT golden (todo 40) for the dataflow the simulator executes."""
    x, w, gamma = inputs["activation"], inputs["weight"], inputs["gamma"]
    if target == "rmsnorm_matmul":
        return G.fused_rmsnorm_matmul(x, gamma, w, EPS)
    # mlp_block: golden.fused_mlp with the trailing bias/w2/residual as IDENTITY
    # ops (b1=0, w2=I, residual=0) — reduces to the rmsnorm->matmul(w1)->gelu
    # prefix the fused_mlp schedule IR denotes and the simulator executes.
    _, _, h = shape
    b1 = np.zeros(h, dtype=np.float32)
    w2 = np.eye(h, dtype=np.float32)
    residual = np.zeros((shape[0], h), dtype=np.float32)
    return G.fused_mlp(x, gamma, w, b1, w2, residual, EPS)


# --- metrics ------------------------------------------------------------------


def compute_metrics(sim: np.ndarray, golden: np.ndarray) -> dict[str, float]:
    """PCC (Pearson), cosine similarity, absolute/relative error, max_abs_diff.

    Accumulates in float64 for metric precision (the comparison itself is the
    raw fp32 simulator output vs the fp32-exact golden).
    """
    s = sim.ravel().astype(np.float64)
    g = golden.ravel().astype(np.float64)
    abs_diff = np.abs(s - g)
    rel = abs_diff / np.maximum(np.abs(g), 1e-6)
    return {
        "pcc": float(np.corrcoef(s, g)[0, 1]),
        "cosine": float(np.dot(s, g) / (np.linalg.norm(s) * np.linalg.norm(g))),
        "max_abs_diff": float(abs_diff.max()),
        "mean_abs_err": float(abs_diff.mean()),
        "mean_rel_err": float(rel.mean()),
        "max_rel_err": float(rel.max()),
    }


def structural_check(target: str, inputs: dict[str, np.ndarray], sim: np.ndarray) -> dict[str, bool]:
    """PRIMARY non-vacuity (structural): the golden holds its fused intermediate in
    fp32 (NOT bf16-representable) while the simulator output IS bf16-representable.

    For mlp_block this ALSO asserts the honest scope reduction: golden.fused_mlp
    with identity trailing operands == the gelu(fused_rmsnorm_matmul) prefix.
    """
    x, w, gamma = inputs["activation"], inputs["weight"], inputs["gamma"]
    normed = G.rmsnorm(x, gamma, EPS)  # the golden's fused intermediate, fp32
    out = {
        "golden_intermediate_is_fp32": bool(not np.array_equal(R.to_bf16(normed), normed)),
        "sim_output_is_bf16": bool(np.array_equal(R.to_bf16(sim), sim)),
        "golden_output_dtype_fp32": bool(G.fused_rmsnorm_matmul(x, gamma, w, EPS).dtype == np.float32),
    }
    if target == "mlp_block":
        h = w.shape[1]
        prefix = G.gelu(G.fused_rmsnorm_matmul(x, gamma, w, EPS))
        full = G.fused_mlp(
            x, gamma, w, np.zeros(h, dtype=np.float32), np.eye(h, dtype=np.float32),
            np.zeros((x.shape[0], h), dtype=np.float32), EPS,
        )
        out["mlp_identity_tail_reduces_to_prefix"] = bool(np.array_equal(full, prefix))
    return out


def calibrate_tripwire(pccs: list[float]) -> tuple[float, float]:
    """SECONDARY non-vacuity: the empirical upper-band trip-wire.

    Returns (good_case_pcc, tripwire): the good-case PCC is the MAX measured across
    the documented seeds (the value closest to 1.0); the trip-wire is the midpoint
    between it and 1.0 — strictly ABOVE the measured good-case PCC, strictly BELOW
    1.0. Never a hard-coded constant.
    """
    good = max(pccs)
    tripwire = good + (1.0 - good) / 2.0
    return good, tripwire


# --- driver -------------------------------------------------------------------


def run_harness(sim_bin: Path | None = None, seeds: tuple[int, ...] = SEEDS,
                shapes: dict[str, list[tuple[int, int, int]]] | None = None,
                verbose: bool = True) -> dict:
    """Run every (target, shape, seed), calibrate the trip-wires, return results."""
    sim_bin = sim_bin or find_sim_bin()
    shapes = shapes or SHAPES
    if not sim_bin.exists():
        raise FileNotFoundError(
            f"{sim_bin} missing: build it with `nix develop --command bash -c "
            "'cmake --build build'` (the Python->C++ simulator bridge)."
        )

    # Nested so it closes over the resolved sim_bin + the per-run workdir, keeping
    # the call signature to the genuine per-run spec (target, shape, seed).
    def run_one(target: str, shape: tuple[int, int, int], seed: int) -> dict:
        inputs = make_inputs(target, shape, seed)
        tag = f"{target}_{'x'.join(map(str, shape))}_{seed:#x}"
        in_path = workdir / f"{tag}.in.tft"
        out_path = workdir / f"{tag}.out.tft"
        write_tensors(in_path, [inputs["activation"], inputs["weight"], inputs["gamma"]])
        proc = subprocess.run(
            [str(sim_bin), "--mlir", str(EXAMPLES[target]),
             "--input", str(in_path), "--output", str(out_path)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"tensixflow-simulate failed ({target}, {shape}, {seed:#x}):\n"
                f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
            )
        sim = read_tensor(out_path)
        golden = compute_golden(target, inputs, shape)
        if sim.shape != golden.shape:
            raise RuntimeError(f"{tag}: shape mismatch sim={sim.shape} golden={golden.shape}")
        return {
            "shape": list(shape),
            "seed": seed,
            "driver": proc.stdout.strip(),
            "structural": structural_check(target, inputs, sim),
            **compute_metrics(sim, golden),
        }

    results: dict = {"pcc_floor": PCC_FLOOR, "seeds": [hex(s) for s in seeds], "targets": {}}
    with tempfile.TemporaryDirectory(prefix="tensixflow_harness_") as tmp:
        workdir = Path(tmp)
        for target, tshapes in shapes.items():
            runs = [run_one(target, sh, seed) for sh in tshapes for seed in seeds]
            good, tripwire = calibrate_tripwire([r["pcc"] for r in runs])
            results["targets"][target] = {
                "example": str(EXAMPLES[target].relative_to(_repo_root())),
                "shapes": [list(s) for s in tshapes],
                "runs": runs,
                "good_case_pcc": good,
                "tripwire": tripwire,
            }
    if verbose:
        print_table(results)
    return results


# --- reporting ----------------------------------------------------------------


def print_table(results: dict) -> None:
    """Print a per-run metrics table + the per-target trip-wire calibration."""
    hdr = f"{'target':17} {'shape':13} {'seed':10} {'PCC':12} {'cosine':12} {'max_abs_diff':12} {'mean_rel':10}"
    print("=" * len(hdr))
    print("TensixFlow correctness harness (D14): simulator bf16-tiled vs golden fp32-exact")
    print(f"PCC floor = {results['pcc_floor']}  (FIXED; arXiv:2606.09879)")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for target, t in results["targets"].items():
        for r in t["runs"]:
            print(f"{target:17} {str(r['shape']):13} {hex(r['seed']):10} "
                  f"{r['pcc']:<12.8f} {r['cosine']:<12.8f} {r['max_abs_diff']:<12.6f} {r['mean_rel_err']:<10.6f}")
        print(f"  -> {target}: good-case PCC = {t['good_case_pcc']:.8f}, "
              f"trip-wire = {t['tripwire']:.8f} (strictly above good-case, < 1.0)")
    print("=" * len(hdr))


def write_metrics_json(results: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2) + "\n")


def main() -> int:
    results = run_harness()
    out = _repo_root() / ".omo" / "evidence" / "task-41-tensixflow.json"
    write_metrics_json(results, out)
    print(f"metrics JSON written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
