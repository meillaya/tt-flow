"""Normalize layout-encoded TTIR and prove the import loop closes.

The TTIR emitter writes `#ttcore.metal_layout` tensor encodings and a real
tt-mlir module can carry device attributes such as `ttcore.meshes`. The importer
rejects both by design, so the loop from emit to import needs a normalization
step first. This test drives the shipped normalizer end to end against the built
binaries:

1. `tensixflow-opt --tensixflow-emit-ttir` writes layout-encoded TTIR from the
   native RMSNorm to matmul example.
2. `scripts/normalize_ttir.sh` parses that text with the pinned tt-mlir Python
   bindings and prints bare-tensor TTIR.
3. `tensixflow-opt --allow-unregistered-dialect --tensixflow-import-ttir`
   imports the normalized text and the planning passes run on it.
4. `tensixflow-translate --tensixflow-to-json` reports the fused graph, which is
   the acceptance evidence that the round trip preserved the program.

A malformed input fails with a pinned parse diagnostic and no traceback. The
checked-in layout sample examples/ttir/rmsnorm_matmul_layout.mlir covers the same
positive path plus the negative case that motivates the tool: the importer
rejects the raw sample.

The normalize tests skip with a stated reason when the pinned wheel or
python3.11 is absent, because the tool cannot run without them. A present wheel
is never a skip: an import, parse, or normalization failure is a test failure.
The built binaries are required; they are reported as a failure when absent,
never as a skip.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
LOCK_FILE = REPO_ROOT / "deps" / "tt-mlir.lock"
DEPS_DIR = REPO_ROOT / ".deps" / "ttmlir"
SITE_DIR = DEPS_DIR / "site"
WHEEL_DOWNLOAD_DIR = DEPS_DIR / "downloads"
WHEEL_STAMP = SITE_DIR / ".ttmlir-wheel"
SITE_PACKAGE = SITE_DIR / "ttmlir" / "ir.py"

NORMALIZER = REPO_ROOT / "scripts" / "normalize_ttir.sh"
NATIVE_EXAMPLE = REPO_ROOT / "examples" / "rmsnorm_matmul.mlir"
LAYOUT_SAMPLE = REPO_ROOT / "examples" / "ttir" / "rmsnorm_matmul_layout.mlir"
BARE_SAMPLE = REPO_ROOT / "examples" / "ttir" / "rmsnorm_matmul.mlir"

INTERPRETER_NAME = "python3.11"
FETCH_HINT = "run 'devenv shell -- bash scripts/fetch_ttmlir_fixtures.sh'"
BUILD_HINT = "build the compiler first (devenv shell -- bash scripts/build.sh)"

# Binary lookup. An explicit override wins; otherwise the newest build tree is
# used so a stale build-baseline cannot shadow the current binary.
OPT_ENV = "TENSIXFLOW_OPT"
TRANSLATE_ENV = "TENSIXFLOW_TRANSLATE"
BUILD_BIN_GLOB = "build*/bin"

# The planning pipeline the import test runs. Fusion and circular-buffer
# allocation are the passes that turn the imported rmsnorm to matmul pair into
# one scheduled graph, which is what the plan JSON is asserted to report.
IMPORT_PLAN_FLAGS = (
    "--allow-unregistered-dialect",
    "--tensixflow-import-ttir",
    "--tensixflow-infer-tile-layout",
    "--tensixflow-fuse-rmsnorm-matmul",
    "--tensixflow-plan-noc-sharding",
    "--tensixflow-allocate-circular-buffers",
)

EXPECTED_GRAPH_ID = "rmsnorm_matmul"
EXPECTED_FUSION = "fused_rmsnorm_matmul"
EXPECTED_INTERMEDIATE_STORAGE = "L1"

# Text that must not survive normalization. The importer rejects both spellings,
# so their absence is the normalized-output contract.
LAYOUT_ALIAS = "#ttcore.metal_layout"
DEVICE_DIALECTS = ("ttcore", "ttnn")

# The rms_norm attributes define the program and must survive the rebuild.
SEMANTIC_MARKERS = (
    '"ttir.rms_norm"',
    '"ttir.matmul"',
    "normalized_shape = array<i64: 4096>",
    "epsilon",
)

# One intentionally malformed module: the generic operation call never closes,
# so the pinned parser rejects the text.
MALFORMED_TTIR = 'module {\n  "ttir.matmul"(%arg0, %arg1 : (tensor<64x64xbf16>\n}\n'


def _read_lock() -> dict[str, str]:
    """Return the NAME=value pairs from deps/tt-mlir.lock."""
    assert LOCK_FILE.is_file(), f"missing pinned tt-mlir lock file {LOCK_FILE}"
    values: dict[str, str] = {}
    for line in LOCK_FILE.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, value = stripped.split("=", 1)
        values[name.strip()] = value.strip()
    return values


def _wheel_path() -> Path | None:
    """Return the cached pinned wheel, or None when it was never fetched."""
    lock = _read_lock()
    wheel_name = lock.get("TT_MLIR_WHEEL")
    assert wheel_name, "deps/tt-mlir.lock does not define TT_MLIR_WHEEL"
    wheel = WHEEL_DOWNLOAD_DIR / wheel_name
    if not wheel.is_file():
        return None
    return wheel


def _ensure_site(wheel: Path) -> Path:
    """Extract the wheel for the wrapper, reusing a matching extraction."""
    stamp_matches = WHEEL_STAMP.is_file() and WHEEL_STAMP.read_text() == wheel.name
    if stamp_matches and SITE_PACKAGE.is_file():
        return SITE_DIR
    shutil.rmtree(SITE_DIR, ignore_errors=True)
    SITE_DIR.mkdir(parents=True)
    try:
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(SITE_DIR)
    except (OSError, zipfile.BadZipFile) as error:
        pytest.fail(f"cannot extract the pinned wheel {wheel}: {error}; {FETCH_HINT}")
    if not SITE_PACKAGE.is_file():
        pytest.fail(f"extracted wheel {wheel} has no ttmlir/ir.py; {FETCH_HINT}")
    WHEEL_STAMP.write_text(wheel.name)
    return SITE_DIR


def _normalizer_environment() -> tuple[Path, str]:
    """Return the wheel site directory and the python3.11 interpreter.

    The two absences the tool cannot work around are the only skips. When both
    are present the test runs for real.
    """
    interpreter = shutil.which(INTERPRETER_NAME)
    if interpreter is None:
        pytest.skip(
            "python3.11 is absent; the pinned ttmlir wheel is built for CPython 3.11"
        )
    wheel = _wheel_path()
    if wheel is None:
        pytest.skip(f"pinned tt-mlir wheel is absent; {FETCH_HINT}")
    return _ensure_site(wheel), interpreter


def _find_bin(name: str, env_name: str) -> Path:
    """Return a built tool, preferring an explicit override."""
    override = os.environ.get(env_name)
    if override:
        path = Path(override)
        assert path.is_file(), f"{env_name} points at {path}, which is not a file"
        return path
    candidates = [REPO_ROOT / "build" / "bin" / name]
    candidates.extend(REPO_ROOT.glob(f"{BUILD_BIN_GLOB}/{name}"))
    built = [candidate for candidate in candidates if candidate.is_file()]
    if not built:
        pytest.fail(f"{name} was not found; {BUILD_HINT} or set {env_name}")
    return max(built, key=lambda candidate: candidate.stat().st_mtime)


def _run_normalizer(
    site: Path,
    interpreter: str,
    arguments: list[str],
    input_text: str | None = None,
    use_overrides: bool = False,
) -> subprocess.CompletedProcess:
    """Run the wrapper, optionally with the explicit environment overrides."""
    command = ["bash", str(NORMALIZER)]
    if use_overrides:
        command.extend(["--site", str(site), "--python", interpreter])
    command.extend(arguments)
    # A run with a file input must not inherit the pytest stdin: the tool would
    # block on a pipe that never closes.
    stdin = subprocess.DEVNULL if input_text is None else None
    return subprocess.run(
        command,
        input=input_text,
        stdin=stdin,
        capture_output=True,
        text=True,
    )


def _emit_ttir(opt: Path, module: Path) -> str:
    result = subprocess.run(
        [str(opt), "--tensixflow-emit-ttir", str(module)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"{opt} --tensixflow-emit-ttir {module} failed with exit code "
        f"{result.returncode}:\n{result.stderr}"
    )
    return result.stdout


def _plan_module(opt: Path, translate: Path, module: Path) -> dict:
    """Import, plan, and translate one bare-tensor module to plan JSON."""
    planned = subprocess.run(
        [str(opt), *IMPORT_PLAN_FLAGS, str(module)],
        capture_output=True,
        text=True,
    )
    assert planned.returncode == 0, (
        f"the import pipeline rejected {module} with exit code "
        f"{planned.returncode}:\n{planned.stderr}"
    )
    assert "tensixflow.fused_rmsnorm_matmul" in planned.stdout, (
        f"the plan did not fuse {module} into tensixflow.fused_rmsnorm_matmul:\n"
        f"{planned.stdout}"
    )
    translated = subprocess.run(
        [str(translate), "--tensixflow-to-json"],
        input=planned.stdout,
        capture_output=True,
        text=True,
    )
    assert translated.returncode == 0, (
        f"tensixflow-translate failed with exit code {translated.returncode}:\n"
        f"{translated.stderr}"
    )
    return json.loads(translated.stdout)


def _assert_plan_reports_fusion(plan: dict) -> None:
    """Check the plan JSON fields that name the fused graph."""
    assert plan["graph_id"] == EXPECTED_GRAPH_ID, plan["graph_id"]
    assert plan["fusion"] == EXPECTED_FUSION, plan["fusion"]
    assert plan["intermediate_storage"] == EXPECTED_INTERMEDIATE_STORAGE, plan[
        "intermediate_storage"
    ]
    assert plan["estimated_dram_reads_saved"] > 0, plan["estimated_dram_reads_saved"]
    # The emitted-path block is required by the plan schema; a fusion that
    # dropped it would not be a schedulable graph.
    assert "emission" in plan, sorted(plan)


def _assert_bare_ttir(text: str) -> None:
    """Check the normalized text carries no layout or device plumbing."""
    assert LAYOUT_ALIAS not in text, text
    for dialect in DEVICE_DIALECTS:
        assert dialect not in text, f"normalized TTIR still mentions {dialect}:\n{text}"


def test_emitted_ttir_normalizes_and_plans_as_fused(tmp_path: Path) -> None:
    # Given the native RMSNorm to matmul example and the built emitter
    site, interpreter = _normalizer_environment()
    opt = _find_bin("tensixflow-opt", OPT_ENV)
    translate = _find_bin("tensixflow-translate", TRANSLATE_ENV)
    emitted = _emit_ttir(opt, NATIVE_EXAMPLE)
    assert LAYOUT_ALIAS in emitted, f"{NATIVE_EXAMPLE} emitted no layout encoding"
    assert '"ttir.' in emitted, f"{NATIVE_EXAMPLE} emitted no ttir op"

    # When the emitted text is normalized through stdin
    normalized = _run_normalizer(site, interpreter, [], input_text=emitted)
    assert normalized.returncode == 0, (
        f"normalization failed with exit code {normalized.returncode}:\n"
        f"{normalized.stderr}"
    )

    # Then the text is bare tensor TTIR with the rms_norm attributes intact
    _assert_bare_ttir(normalized.stdout)
    assert "func.func @rmsnorm_matmul" in normalized.stdout, normalized.stdout
    for marker in SEMANTIC_MARKERS:
        assert marker in normalized.stdout, f"{marker} missing from:\n{normalized.stdout}"

    # And the importer accepts the normalized text and the plan reports fusion
    module_path = tmp_path / "emitted.mlir"
    module_path.write_text(normalized.stdout)
    plan = _plan_module(opt, translate, module_path)
    _assert_plan_reports_fusion(plan)


def test_layout_sample_normalizes_with_explicit_overrides(tmp_path: Path) -> None:
    # Given the checked-in layout-encoded sample and an explicit wheel override
    site, interpreter = _normalizer_environment()
    opt = _find_bin("tensixflow-opt", OPT_ENV)
    translate = _find_bin("tensixflow-translate", TRANSLATE_ENV)

    # When the wrapper is told where the wheel and the interpreter are
    output_path = tmp_path / "sample.mlir"
    normalized = _run_normalizer(
        site,
        interpreter,
        [str(LAYOUT_SAMPLE), "-o", str(output_path)],
        use_overrides=True,
    )

    # Then the encoding and the module attribute are gone
    assert normalized.returncode == 0, (
        f"normalization failed with exit code {normalized.returncode}:\n"
        f"{normalized.stderr}"
    )
    normalized_text = output_path.read_text()
    _assert_bare_ttir(normalized_text)
    assert "func.func @rmsnorm_matmul" in normalized_text, normalized_text

    # And the plan reports the fused graph for the sample shape
    plan = _plan_module(opt, translate, output_path)
    _assert_plan_reports_fusion(plan)


def test_importer_rejects_the_layout_sample_without_normalization() -> None:
    # Given the raw checked-in sample
    opt = _find_bin("tensixflow-opt", OPT_ENV)

    # When the importer reads it directly
    imported = subprocess.run(
        [str(opt), *IMPORT_PLAN_FLAGS, str(LAYOUT_SAMPLE)],
        capture_output=True,
        text=True,
    )

    # Then the diagnostic names the device attribute and the subset docs
    assert imported.returncode != 0, "the importer accepted layout-encoded TTIR"
    assert "ttcore.meshes" in imported.stderr, imported.stderr
    assert "docs/ttir.md" in imported.stderr, imported.stderr


def test_malformed_input_fails_without_a_traceback(tmp_path: Path) -> None:
    # Given the normalizer environment and a malformed module
    site, interpreter = _normalizer_environment()
    module_path = tmp_path / "malformed.mlir"
    module_path.write_text(MALFORMED_TTIR)
    output_path = tmp_path / "normalized.mlir"

    # When the tool reads the malformed module
    normalized = _run_normalizer(
        site,
        interpreter,
        [str(module_path), "-o", str(output_path)],
    )

    # Then it fails and writes no output
    assert normalized.returncode != 0, "the malformed module was normalized"
    assert normalized.stdout == "", normalized.stdout
    assert not output_path.exists(), "the malformed module produced an output file"

    # And the failure is the pinned parse diagnostic, not a Python traceback
    assert "error: cannot parse" in normalized.stderr, normalized.stderr
    assert "Unable to parse module assembly" in normalized.stderr, normalized.stderr
    assert "Traceback" not in normalized.stderr, normalized.stderr


def test_bare_ttir_is_stable_through_normalization() -> None:
    # Given a module already written in the bare-tensor subset
    site, interpreter = _normalizer_environment()

    # When it is normalized twice
    first = _run_normalizer(site, interpreter, [str(BARE_SAMPLE)])
    assert first.returncode == 0, first.stderr
    second = _run_normalizer(site, interpreter, [], input_text=first.stdout)
    assert second.returncode == 0, second.stderr

    # Then the normalized text is importable and the second pass changes nothing
    _assert_bare_ttir(second.stdout)
    assert second.stdout == first.stdout, "normalization is not idempotent"
    assert '"ttir.rms_norm"' in second.stdout, second.stdout
