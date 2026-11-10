"""Pytest for the plan report tool.

The report is the practical validator for a plan document: it parses the plan,
recomputes the projected emitted-path block from the plan's own tile geometry,
and refuses to render a plan whose emission numbers or mapping disagree with
that projection. This test drives the built binaries, so it runs the same
validation a user hits.

The test covers the accept path and the two rejection paths the report owns:
a stale byte count and an unknown mapping. A plan without the required
emission block is rejected by the parser before rendering.
"""

import json
import os
import subprocess
from pathlib import Path

import pytest

from tensixflow.lowering import find_opt_bin, find_translate_bin

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MATMUL_EXAMPLE = _REPO_ROOT / "examples" / "matmul.mlir"


def _find_report_bin() -> Path:
    env = os.environ.get("TENSIXFLOW_REPORT_BIN")
    if env:
        return Path(env)
    return _REPO_ROOT / "build" / "bin" / "tensixflow-report"


def _require_bins() -> tuple[Path, Path, Path]:
    opt_bin = find_opt_bin()
    translate_bin = find_translate_bin()
    report_bin = _find_report_bin()
    missing = [str(b) for b in (opt_bin, translate_bin, report_bin) if not b.is_file()]
    if missing:
        pytest.skip(f"built tools are absent: {', '.join(missing)}; run scripts/build.sh")
    return opt_bin, translate_bin, report_bin


def _build_plan(opt_bin: Path, translate_bin: Path) -> dict:
    planned = subprocess.run(
        [
            str(opt_bin),
            "--tensixflow-device=n150",
            "--tensixflow-infer-tile-layout",
            "--tensixflow-plan-noc-sharding",
            "--tensixflow-allocate-circular-buffers",
            str(_MATMUL_EXAMPLE),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    translated = subprocess.run(
        [str(translate_bin), "--tensixflow-to-json"],
        input=planned.stdout,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(translated.stdout)


def _write_plan(path: Path, plan: dict) -> Path:
    path.write_text(json.dumps(plan))
    return path


def _run_report(report_bin: Path, plan_path: Path, html_path: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(report_bin), "--in", str(plan_path), "--out", str(html_path)],
        capture_output=True,
        text=True,
    )


@pytest.fixture(scope="module")
def plan(tmp_path_factory) -> tuple[Path, Path, Path, dict]:
    opt_bin, translate_bin, report_bin = _require_bins()
    work = tmp_path_factory.mktemp("report")
    document = _build_plan(opt_bin, translate_bin)
    plan_path = _write_plan(work / "plan.json", document)
    return report_bin, work, plan_path, document


def test_report_renders_the_emitted_path_section(plan):
    report_bin, work, plan_path, _ = plan
    html_path = work / "report.html"

    result = _run_report(report_bin, plan_path, html_path)

    assert result.returncode == 0, result.stderr
    rendered = html_path.read_text()
    assert "Projected Emitted Path" in rendered
    assert "emitted-path item" in rendered


def test_report_rejects_a_stale_emitted_byte_count(plan):
    report_bin, work, plan_path, document = plan
    stale = json.loads(json.dumps(document))
    stale["emission"]["dram_read_bytes"] += 1
    stale_path = _write_plan(work / "stale.json", stale)

    result = _run_report(report_bin, stale_path, work / "stale.html")

    assert result.returncode != 0
    assert "emission.dram_read_bytes" in result.stderr
    assert not (work / "stale.html").exists()


def test_report_rejects_an_unknown_emission_mapping(plan):
    report_bin, work, plan_path, document = plan
    unknown = json.loads(json.dumps(document))
    unknown["emission"]["mapping"] = "mystery"
    unknown_path = _write_plan(work / "unknown.json", unknown)

    result = _run_report(report_bin, unknown_path, work / "unknown.html")

    assert result.returncode != 0
    assert "mapping" in result.stderr
    assert "mystery" in result.stderr


def test_report_rejects_a_plan_without_an_emission_block(plan):
    report_bin, work, plan_path, document = plan
    # The parser requires the block, so this fails before any rendering.
    stripped = json.loads(json.dumps(document))
    del stripped["emission"]
    stripped_path = _write_plan(work / "stripped.json", stripped)

    result = _run_report(report_bin, stripped_path, work / "stripped.html")

    assert result.returncode != 0
    assert not (work / "stripped.html").exists()


_MLP_EXAMPLE = _REPO_ROOT / "examples" / "mlp_block_small.mlir"


def test_report_accepts_a_fused_mlp_plan(tmp_path):
    # A fused MLP plan records the blocked-reuse mapping and the hidden width is
    # only recoverable from the plan's tensor list, so the report has to
    # reproject the MLP footprint from those shapes to accept the block.
    opt_bin, translate_bin, report_bin = _require_bins()
    if not _MLP_EXAMPLE.is_file():
        pytest.skip("the small MLP example is absent")
    planned = subprocess.run(
        [
            str(opt_bin),
            "--tensixflow-device=n150",
            "--tensixflow-infer-tile-layout",
            "--tensixflow-fuse-rmsnorm-matmul",
            "--tensixflow-plan-noc-sharding",
            "--tensixflow-allocate-circular-buffers",
            str(_MLP_EXAMPLE),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    translated = subprocess.run(
        [str(translate_bin), "--tensixflow-to-json"],
        input=planned.stdout,
        capture_output=True,
        text=True,
        check=True,
    )
    document = json.loads(translated.stdout)
    assert document["fusion"] == "fused_mlp"
    assert document["emission"]["mapping"] == "blocked-reuse"
    plan_path = _write_plan(tmp_path / "mlp_plan.json", document)

    result = _run_report(report_bin, plan_path, tmp_path / "mlp_plan.html")

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "mlp_plan.html").is_file()
    assert "blocked-reuse" in (tmp_path / "mlp_plan.html").read_text()
