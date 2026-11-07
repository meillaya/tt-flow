#!/usr/bin/env python3
"""Generate the Excalidraw scenes used in the README.

The scenes are drawn rather than shipped as opaque binaries so a later edit is a
reproducible run of this file plus scripts/render_figures.sh. The drawing rules
are the house ones: white background, hachure fills, roughness 2, font family 1,
a labelled arrow for every transition, and one claim per figure.

    python3 scripts/gen_figures.py
    bash scripts/render_figures.sh
"""

from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "docs" / "figures"

# House palette. Strokes carry the meaning and fills stay pale.
INK = "#1e1e1e"
MUTED = "#5c5f66"
COMPILER = ("#1971c2", "#d0ebff")
RUNTIME = ("#9c36b5", "#f3d9fa")
ANALYSIS = ("#f08c00", "#fff3bf")
MODEL = ("#0c8599", "#c5f6fa")
RISK = ("#e03131", "#ffe3e3")
NEUTRAL = ("#495057", "#f1f3f5")

FONT_BODY = 22
FONT_TITLE = 32
FONT_NOTE = 18

SCENE_W = 1520
SCENE_H = 940
MARGIN = 40


def _seed(rng: random.Random) -> int:
    return rng.randint(1, 2**31 - 1)


def _base(
    rng: random.Random,
    kind: str,
    x: float,
    y: float,
    width: float,
    height: float,
    stroke: str,
    fill: str,
) -> dict:
    return {
        "id": f"{kind}-{rng.randrange(16**8):08x}",
        "type": kind,
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "angle": 0,
        "strokeColor": stroke,
        "backgroundColor": fill,
        "fillStyle": "hachure",
        "strokeWidth": 2,
        "strokeStyle": "solid",
        "roughness": 2,
        "opacity": 100,
        "groupIds": [],
        "frameId": None,
        "roundness": {"type": 3},
        "seed": _seed(rng),
        "version": 1,
        "versionNonce": _seed(rng),
        "isDeleted": False,
        "boundElements": None,
        "updated": 1,
        "link": None,
        "locked": False,
    }


def _text_element(
    rng: random.Random,
    text: str,
    x: float,
    y: float,
    width: float,
    height: float,
    stroke: str,
    font: int,
    align: str = "center",
    container_id: str | None = None,
) -> dict:
    element = _base(rng, "text", x, y, width, height, stroke, "transparent")
    element.update(
        {
            "text": text,
            "originalText": text,
            "fontSize": font,
            "fontFamily": 1,
            "textAlign": align,
            "verticalAlign": "middle",
            "baseline": int(font * 0.9),
            "containerId": container_id,
            "lineHeight": 1.25,
            "backgroundColor": "transparent",
            "strokeWidth": 2,
            "roughness": 2,
            "roundness": None,
        }
    )
    return element


def label_width(text: str, font: int) -> float:
    """Approximate the rendered width of a line so boxes are not cramped."""
    longest = max(len(line) for line in text.split("\n"))
    return max(48.0, longest * font * 0.58)


def text(rng: random.Random, content: str, x: float, y: float, font: int, stroke: str = INK,
         align: str = "left", width: float | None = None) -> dict:
    lines = content.split("\n")
    width = width if width is not None else label_width(content, font)
    height = len(lines) * font * 1.25
    return _text_element(rng, content, x, y, width, height, stroke, font, align=align)


def box(
    rng: random.Random,
    content: str,
    x: float,
    y: float,
    width: float | None = None,
    height: float | None = None,
    role: tuple[str, str] = COMPILER,
    font: int = FONT_BODY,
) -> tuple[dict, dict]:
    """A rectangle with a centred label bound to it.

    The width never drops below the width the longest line needs, because a
    bound text element does not wrap and would otherwise spill over the edge.
    """
    stroke, fill = role
    lines = content.split("\n")
    height = height if height is not None else max(72.0, len(lines) * font * 1.35 + 24)
    needed = label_width(content, font) + 44
    width = max(width if width is not None else 0.0, needed)
    shape = _base(rng, "rectangle", x, y, width, height, stroke, fill)
    bound = _text_element(
        rng,
        content,
        x + 6,
        y + (height - len(lines) * font * 1.25) / 2,
        max(width - 12, 16.0),
        len(lines) * font * 1.25,
        INK,
        font,
        container_id=shape["id"],
    )
    bound["boundElements"] = None
    shape["boundElements"] = [{"type": "text", "id": bound["id"]}]
    return shape, bound


def arrow(
    rng: random.Random,
    start: tuple[float, float],
    end: tuple[float, float],
    label: str,
    label_box: tuple[float, float] | None = None,
) -> tuple[dict, dict]:
    """A straight arrow plus a separate text label near it."""
    sx, sy = start
    ex, ey = end
    dx, dy = ex - sx, ey - sy
    element = _base(rng, "arrow", sx, sy, abs(dx), abs(dy), MUTED, "transparent")
    element.update(
        {
            "points": [[0, 0], [dx, dy]],
            "lastCommittedPoint": None,
            "startBinding": None,
            "endBinding": None,
            "startArrowhead": None,
            "endArrowhead": "arrow",
            "roundness": {"type": 2},
        }
    )
    if label_box is None:
        label_x = (sx + ex) / 2 + 10
        label_y = (sy + ey) / 2 - 30
    else:
        label_x, label_y = label_box
    label_element = text(rng, label, label_x, label_y, FONT_NOTE, MUTED)
    return element, label_element


def scene(elements: list[dict], width: int = SCENE_W, height: int = SCENE_H) -> dict:
    return {
        "type": "excalidraw",
        "version": 2,
        "source": "https://github.com/meillaya/tt-flow",
        "elements": elements,
        "appState": {"viewBackgroundColor": "#ffffff"},
        "files": {},
        "_scene": {"width": width, "height": height},
    }


def _geometry(shape_and_text: tuple[dict, dict]) -> tuple[float, float, float, float]:
    """Return left, top, right, bottom of a box and its bound text."""
    shape = shape_and_text[0]
    return (
        shape["x"],
        shape["y"],
        shape["x"] + shape["width"],
        shape["y"] + shape["height"],
    )


def _mid_y(shape_and_text: tuple[dict, dict]) -> float:
    _, top, _, bottom = _geometry(shape_and_text)
    return (top + bottom) / 2


def _gap_label_x(left_geometry, right_geometry) -> float:
    return (left_geometry[2] + right_geometry[0]) / 2 - 24


def pipeline_scene(rng: random.Random) -> dict:
    """One claim: every result is checked at the layer that produced it."""
    els: list[dict] = []
    els.append(text(rng, "tt-flow: plan, emit, verify", MARGIN, 24, FONT_TITLE, INK))

    # Column 1: the compiler chain.
    els.append(text(rng, "compiler", 50, 92, FONT_NOTE, MUTED))
    a1 = box(rng, "MLIR input\ntensixflow dialect", 50, 130, role=COMPILER)
    a2 = box(rng, "fusion, tile layout, plan\nagainst the device spec", 50, 290, role=ANALYSIS)
    a3 = box(rng, "schedule IR\nreader, compute, writer", 50, 470, role=COMPILER)
    els.extend(a1 + a2 + a3)
    g1, g2, g3 = _geometry(a1), _geometry(a2), _geometry(a3)
    center_x = (g1[0] + g1[2]) / 2
    els.extend(arrow(rng, (center_x, g1[3]), (center_x, g2[1] - 2), "passes", (center_x + 14, g1[3] + 2)))
    els.extend(arrow(rng, (center_x, g2[3]), (center_x, g3[1] - 2), "lower", (center_x + 14, g2[3] + 2)))

    # Column 2: the emitters.
    els.append(text(rng, "emitters", 660, 92, FONT_NOTE, MUTED))
    b1 = box(rng, "TTIR text\nfor tt-mlir", 660, 150, role=COMPILER)
    b2 = box(rng, "TT-Metalium project\nhost and three kernels", 660, 440, role=RUNTIME)
    els.extend(b1 + b2)
    h1, h2 = _geometry(b1), _geometry(b2)
    els.extend(
        arrow(
            rng,
            (g3[2] + 4, _mid_y(a3) - 8),
            (h1[0] - 6, _mid_y(b1) + 12),
            "emit IR",
            (_gap_label_x(g3, h1) + 4, _mid_y(a3) - 110),
        )
    )
    els.extend(
        arrow(
            rng,
            (g3[2] + 4, _mid_y(a3) + 16),
            (h2[0] - 6, _mid_y(b2)),
            "emit C++",
            (_gap_label_x(g3, h2) + 4, _mid_y(a3) + 44),
        )
    )

    # Column 3: the checks, one per artifact.
    els.append(text(rng, "checks", 1200, 92, FONT_NOTE, MUTED))
    c1 = box(rng, "ttmlir parses\nthe emitted IR", 1200, 160, role=MODEL)
    c2 = box(rng, "links against the\npinned tt-metal release", 1200, 330, role=MODEL)
    c3 = box(rng, "ttsim or a card\nruns it and reports PCC", 1200, 500, role=RISK)
    els.extend(c1 + c2 + c3)
    k1, k2, k3 = _geometry(c1), _geometry(c2), _geometry(c3)
    els.extend(
        arrow(
            rng,
            (h1[2] + 4, _mid_y(b1)),
            (k1[0] - 6, _mid_y(c1)),
            "parse",
            (_gap_label_x(h1, k1), _mid_y(b1) - 62),
        )
    )
    els.extend(
        arrow(
            rng,
            (h2[2] + 4, _mid_y(b2) - 14),
            (k2[0] - 6, _mid_y(c2) + 10),
            "build",
            (_gap_label_x(h2, k2), _mid_y(b2) - 92),
        )
    )
    els.extend(
        arrow(
            rng,
            (h2[2] + 4, _mid_y(b2) + 18),
            (k3[0] - 6, _mid_y(c3) - 20),
            "run",
            (_gap_label_x(h2, k3), _mid_y(b2) + 48),
        )
    )

    els.append(
        text(
            rng,
            "solid boxes run on this machine; the red box needs a card or the pinned simulator",
            50,
            700,
            FONT_NOTE,
            MUTED,
        )
    )
    return scene(els, 1520, 770)


def device_budget_scene(rng: random.Random) -> dict:
    """One claim: a 32x32 bf16 tile costs 2 KiB and the n150 L1 budget is 1,464 KiB."""
    els: list[dict] = []
    els.append(text(rng, "Wormhole n150 per-core budget", MARGIN, 24, FONT_TITLE, INK))

    # The L1 bar: segment widths are readable rather than exactly to scale,
    # because 12 KiB of default circular buffers would otherwise be two pixels.
    x0 = 60.0
    y0 = 160.0
    segments = [
        ("kernels and reserved\n64 KiB", 240.0, NEUTRAL),
        ("CBs\n12 KiB", 200.0, ANALYSIS),
        ("free for shards\n1,388 KiB", 640.0, COMPILER),
    ]
    cursor = x0
    for label, segment_width, role in segments:
        shape, bound = box(rng, label, cursor, y0, width=segment_width, height=96, role=role, font=FONT_NOTE)
        shape["roundness"] = None
        els.extend([shape, bound])
        cursor += segment_width
    els.append(
        text(
            rng,
            "1,464 KiB of usable L1 per Tensix core (1464 x 1024 = 1,499,136 B); bar not to scale",
            x0,
            120,
            FONT_NOTE,
            MUTED,
        )
    )

    # The tile.
    tile, tile_text = box(rng, "32 x 32 bf16 tile\n2 KiB", 120, 340, role=ANALYSIS, width=260)
    els.extend([tile, tile_text])

    # Grid and NoC facts.
    grid, grid_text = box(rng, "8 x 8 = 64 compute cores\n80 Tensix on the die", 460, 340, role=MODEL, width=320)
    noc, noc_text = box(rng, "2 NoCs\n32 B/cycle per direction", 840, 340, role=RUNTIME, width=300)
    dram, dram_text = box(rng, "12 GiB GDDR6\n288 GB/s", 1200, 340, role=NEUTRAL, width=250)
    els.extend([grid, grid_text, noc, noc_text, dram, dram_text])

    # The planner rule.
    rule, rule_text = box(
        rng,
        "planner check: 3 CBs x depth x 2 KiB <= L1 - 64 KiB\n"
        "default depth is 2 tiles, the upstream double-buffer depth",
        120,
        560,
        role=ANALYSIS,
        width=820,
        font=FONT_BODY,
    )
    els.extend([rule, rule_text])
    # One arrow, from the CB slice of the budget to the rule it constrains.
    els.extend(arrow(rng, (400, 258), (400, 558), "budget", (414, 300)))
    els.append(text(rng, "device facts", 60, 310, FONT_NOTE, MUTED))

    els.append(
        text(
            rng,
            "sources: tt-isa-documentation L1.md and NoC README, tt-metal core and SoC descriptors, tenstorrent.com AIB pages",
            MARGIN,
            SCENE_H - MARGIN - 30,
            FONT_NOTE,
            MUTED,
        )
    )
    return scene(els, 1520, 800)


def verification_scene(rng: random.Random) -> dict:
    """One claim: what this machine checks, what only a card can check, what is projected."""
    els: list[dict] = []
    els.append(text(rng, "What runs where", MARGIN, 24, FONT_TITLE, INK))

    columns = [
        (
            "checked on this machine",
            60,
            MODEL,
            [
                "spec values against\nunit tests",
                "IR against\nlit and FileCheck",
                "TTIR in and out with\nthe pinned ttmlir parser",
                "emitted host links\nagainst release headers",
                "matmul on the planned\ngrid under ttsim",
                "fused RMSNorm matmul\nunder ttsim",
                "reference model vs\nNumPy golden",
            ],
        ),
        (
            "needs a card",
            580,
            RISK,
            [
                "device execution",
                "measured cycles and\nbandwidth",
                "silicon numerics",
            ],
        ),
        (
            "projected only",
            1080,
            ANALYSIS,
            [
                "roofline cycles",
                "DRAM and NoC traffic",
                "bound classification",
                "optimisation hints",
            ],
        ),
    ]
    for title, x, role, items in columns:
        els.append(text(rng, title, x, 96, FONT_NOTE, MUTED))
        y = 124
        for item in items:
            shape, bound = box(rng, item, x, y, role=role, width=380, font=FONT_NOTE)
            els.extend([shape, bound])
            y += 110

    els.append(
        text(
            rng,
            "the red column is not a gap in the report: it is the part no CPU-only run can claim",
            MARGIN,
            900,
            FONT_NOTE,
            MUTED,
        )
    )
    return scene(els, 1520, 980)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    rng = random.Random(7)
    scenes = {
        "pipeline": pipeline_scene(rng),
        "device_budget": device_budget_scene(rng),
        "verification": verification_scene(rng),
    }
    for name, content in scenes.items():
        path = FIGURES / f"{name}.excalidraw"
        path.write_text(json.dumps(content, indent=2) + "\n")
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
