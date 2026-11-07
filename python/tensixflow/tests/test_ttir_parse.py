"""Parse emitted TTIR with the pinned tt-mlir Python bindings.

The pinned tt-mlir release ships a CPython 3.11 wheel and no ttmlir-opt
executable, so the released bindings are the only real parser. This test emits
TTIR text with tensixflow-opt and parses that text in a separate python3.11
process that imports ttmlir.ir. A parse or verifier diagnostic fails the test.

The test skips with an explicit reason when the pinned wheel or python3.11 is
absent. A present wheel is never treated as a skip: if the bindings import but
the emitted TTIR does not parse, the test fails.
"""

from __future__ import annotations

import hashlib
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

# Every lit input that runs --tensixflow-emit-ttir. The emitter runs as a
# terminal pass, so one input module is one emitted TTIR text, and every such
# text is parsed here. emit_ttir.mlir carries one intentional UNSUPPORTED marker
# in a function that still returns, so it parses but is not fully mapped.
PARSE_INPUT_MODULES = (
    REPO_ROOT / "test" / "Tensixflow" / "emit_ttir.mlir",
    REPO_ROOT / "test" / "Tensixflow" / "emit_ttir_ttmlir_parse.mlir",
)

# Inputs whose emitted text must contain no UNSUPPORTED marker at all. The
# parser cannot see markers because they are comments, so this separate check
# keeps a mapping gap from hiding behind a parse pass.
FULLY_MAPPED_MODULES = (REPO_ROOT / "test" / "Tensixflow" / "emit_ttir_ttmlir_parse.mlir",)

# Environment override for a build tree outside the default search.
EMIT_OPT_ENV = "TENSIXFLOW_OPT"
EMIT_OPT_BUILD_GLOB = "build*/bin/tensixflow-opt"
CANONICAL_EMIT_OPT = REPO_ROOT / "build" / "bin" / "tensixflow-opt"

WHEEL_DOWNLOAD_DIR = DEPS_DIR / "downloads"
WHEEL_STAMP = SITE_DIR / ".ttmlir-wheel"
SITE_PACKAGE = SITE_DIR / "ttmlir" / "ir.py"

PYTHON311 = "python3.11"
FETCH_HINT = "run 'devenv shell -- bash scripts/fetch_ttmlir_fixtures.sh'"

# Driver executed by python3.11. The bindings register the dialects when the
# package is imported, so importing ttmlir.ir is what makes ttcore and ttir
# attributes parse. The text arrives on stdin and the parse result is the exit
# status; the raised diagnostic text is printed so pytest can report it.
_PARSE_DRIVER = r"""
import sys

from ttmlir import ir

for dialect in ("ttcore", "ttir", "ttnn", "d2m", "ttkernel"):
    __import__("ttmlir.dialects." + dialect)

text = sys.stdin.read()
if not text.strip():
    print("no emitted TTIR on stdin", file=sys.stderr)
    sys.exit(1)
try:
    ir.Module.parse(text, ir.Context())
except Exception as error:
    print(str(error), file=sys.stderr)
    sys.exit(1)
print("parsed emitted TTIR")
"""


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


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_wheel(lock: dict[str, str]) -> Path | None:
    """Return the cached pinned wheel, or None when it was never fetched."""
    wheel_name = lock.get("TT_MLIR_WHEEL")
    assert wheel_name, "deps/tt-mlir.lock does not define TT_MLIR_WHEEL"
    expected = lock.get("TT_MLIR_WHEEL_SHA256")
    assert expected, "deps/tt-mlir.lock does not define TT_MLIR_WHEEL_SHA256"
    wheel = WHEEL_DOWNLOAD_DIR / wheel_name
    if not wheel.is_file():
        return None
    # A stale or corrupt cache must fail here rather than parse with the wrong
    # compiler, so the lock digest is checked again instead of trusted.
    actual = _sha256_of(wheel)
    assert actual == expected, (
        f"cached wheel {wheel} has sha256 {actual}, the lock pins {expected}; "
        f"delete the download and {FETCH_HINT}"
    )
    return wheel


def _extract_wheel(wheel: Path) -> Path:
    """Unpack the wheel for PYTHONPATH use, reusing a matching extraction."""
    stamp_matches = WHEEL_STAMP.is_file() and WHEEL_STAMP.read_text() == wheel.name
    if stamp_matches and SITE_PACKAGE.is_file():
        return SITE_DIR
    shutil.rmtree(SITE_DIR, ignore_errors=True)
    SITE_DIR.mkdir(parents=True)
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(SITE_DIR)
    if not SITE_PACKAGE.is_file():
        pytest.fail(f"extracted wheel {wheel} has no ttmlir/ir.py")
    WHEEL_STAMP.write_text(wheel.name)
    return SITE_DIR


def _find_emit_opt() -> Path | None:
    """Return the built emitter, preferring TENSIXFLOW_OPT.

    When several build trees exist, the newest binary wins. A stale tree such
    as build-baseline must not be parsed as if it were the current emitter.
    """
    override = os.environ.get(EMIT_OPT_ENV)
    if override:
        path = Path(override)
        assert path.is_file(), f"{EMIT_OPT_ENV} points at {path}, which is not a file"
        return path
    candidates = [CANONICAL_EMIT_OPT]
    candidates.extend(REPO_ROOT.glob(EMIT_OPT_BUILD_GLOB))
    built = [candidate for candidate in candidates if candidate.is_file()]
    if not built:
        return None
    return max(built, key=lambda candidate: candidate.stat().st_mtime)


def _linked_but_missing(shared_object: Path) -> list[str]:
    """Return the library names ldd cannot resolve for one shared object."""
    ldd = shutil.which("ldd")
    if ldd is None:
        return []
    result = subprocess.run(
        [ldd, str(shared_object)], capture_output=True, text=True
    )
    missing: list[str] = []
    for line in result.stdout.splitlines():
        if "not found" not in line:
            continue
        name = line.split("=>", 1)[0].strip()
        if name:
            missing.append(name)
    return missing


def _nix_closure(executable: str) -> list[Path]:
    """Return the store paths an executable depends on, when nix-store exists."""
    nix_store = shutil.which("nix-store")
    if nix_store is None:
        return []
    result = subprocess.run(
        [nix_store, "-qR", executable], capture_output=True, text=True
    )
    if result.returncode != 0:
        return []
    return [Path(line) for line in result.stdout.splitlines() if line]


def _resolve_library_dirs(interpreter: str) -> list[str]:
    """Resolve the non-nix libraries the manylinux wheel links against.

    The wheel targets a system glibc layout while the devenv shell exposes nix
    store paths, so libz and libstdc++ are not on the default search path. The
    directory of each unresolvable name is found in the closures of the
    python3.11 interpreter and the C++ toolchain, then exported as
    LD_LIBRARY_PATH for the driver process. A name that stays unresolved fails
    the test instead of producing a confusing ImportError.
    """
    shared_objects = sorted((SITE_DIR / "ttmlir" / "_mlir_libs").glob("*.so"))
    missing: list[str] = []
    for shared_object in shared_objects:
        for name in _linked_but_missing(shared_object):
            if name not in missing:
                missing.append(name)
    if not missing:
        return []
    roots = _nix_closure(interpreter)
    for tool in ("g++", "cc"):
        executable = shutil.which(tool)
        if executable is not None:
            roots.extend(_nix_closure(executable))
    library_dirs: list[str] = []
    unresolved: list[str] = []
    for name in missing:
        resolved = None
        for root in roots:
            for directory in (root / "lib", root / "lib64"):
                candidate = directory / name
                if candidate.is_file():
                    resolved = candidate
                    break
            if resolved is not None:
                break
        if resolved is None:
            unresolved.append(name)
        elif str(resolved.parent) not in library_dirs:
            library_dirs.append(str(resolved.parent))
    assert not unresolved, (
        "cannot resolve shared libraries for the pinned ttmlir wheel: "
        + ", ".join(unresolved)
        + "; devenv.nix must provide them"
    )
    return sorted(library_dirs)


def _driver_environment(site: Path, library_dirs: list[str]) -> dict[str, str]:
    """Build the python3.11 environment for the parser driver."""
    env = dict(os.environ)
    python_path = [str(site)]
    if env.get("PYTHONPATH"):
        python_path.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_path)
    library_path = list(library_dirs)
    if env.get("LD_LIBRARY_PATH"):
        library_path.append(env["LD_LIBRARY_PATH"])
    if library_path:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(library_path)
    return env


def _emit_ttir(emit_opt: Path, module: Path) -> str:
    result = subprocess.run(
        [str(emit_opt), "--tensixflow-emit-ttir", str(module)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"{emit_opt} --tensixflow-emit-ttir {module} failed with exit code "
        f"{result.returncode}:\n{result.stderr}"
    )
    return result.stdout


def _assert_has_ttir_ops(module: Path, emitted: str) -> None:
    """Fail on an emission with no TTIR op, so a parse cannot pass vacuously."""
    assert '"ttir.' in emitted, f"{module} emitted no ttir op; the parse would prove nothing"


def _assert_fully_mapped(module: Path, emitted: str) -> None:
    """Reject UNSUPPORTED markers in a fixture that must stay parse-clean.

    The markers are comments, so the parser cannot see them. A marker in this
    fixture would hide a mapping gap behind a parse pass.
    """
    markers = [line for line in emitted.splitlines() if "// UNSUPPORTED:" in line]
    assert not markers, (
        f"{module} emitted UNSUPPORTED markers, so it is not a parse-clean fixture:\n"
        + "\n".join(markers)
    )


def _parse_emitted_ttir(
    interpreter: str, env: dict[str, str], emitted: str, module: Path
) -> None:
    result = subprocess.run(
        [interpreter, "-c", _PARSE_DRIVER],
        input=emitted,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, (
        f"the pinned ttmlir bindings rejected TTIR emitted from {module}:\n"
        f"{result.stderr.strip()}\n"
        f"the emitted text was:\n{emitted}"
    )


def test_emitted_ttir_parses_with_pinned_ttmlir() -> None:
    lock = _read_lock()
    wheel = _find_wheel(lock)
    if wheel is None:
        pytest.skip(f"pinned tt-mlir wheel is absent; {FETCH_HINT}")
    interpreter = shutil.which(PYTHON311)
    if interpreter is None:
        pytest.skip("python3.11 is absent; the pinned ttmlir wheel is built for CPython 3.11")
    emit_opt = _find_emit_opt()
    if emit_opt is None:
        pytest.fail(
            "tensixflow-opt was not found; build the compiler first "
            "(devenv shell -- bash scripts/build.sh) or set " + EMIT_OPT_ENV
        )
    site = _extract_wheel(wheel)
    library_dirs = _resolve_library_dirs(interpreter)
    env = _driver_environment(site, library_dirs)
    for module in PARSE_INPUT_MODULES:
        emitted = _emit_ttir(emit_opt, module)
        _assert_has_ttir_ops(module, emitted)
        _parse_emitted_ttir(interpreter, env, emitted, module)
        if module in FULLY_MAPPED_MODULES:
            _assert_fully_mapped(module, emitted)
