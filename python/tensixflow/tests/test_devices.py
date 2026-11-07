"""Tests for the generated device table (python/tensixflow/devices.py).

The values are the published figures recorded in spec/devices.json. The parity
test compares the generated module against the JSON source so the two cannot
drift; the value tests pin the numbers the scheduler and perf model use.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tensixflow import devices

REPO_ROOT = Path(__file__).resolve().parents[3]
SPEC_PATH = REPO_ROOT / "spec" / "devices.json"


def test_generated_module_matches_json_source() -> None:
    """Every field the generator copies appears in both files with one value."""
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    assert set(devices.DEVICES) == set(spec["devices"])
    for name, entry in spec["devices"].items():
        generated = devices.DEVICES[name]
        assert generated.arch == entry["arch"]
        assert generated.chips == entry["chips"]
        assert generated.compute_grid_x == entry["compute"]["grid_x"]
        assert generated.compute_grid_y == entry["compute"]["grid_y"]
        assert generated.l1_usable_bytes == entry["l1"]["usable_bytes"]
        assert generated.l1_nominal_bytes == entry["l1"]["nominal_bytes"]
        assert generated.tile_bytes == entry["tile"]["bytes"]
        assert generated.clock_hz == entry["clock_hz"]
        assert (
            generated.flops_per_cycle_per_core
            == entry["matrix_unit"]["flops_per_cycle_per_core"]
        )
        assert generated.dram_bytes == entry["dram"]["bytes"]
        assert (
            generated.dram_bandwidth_bytes_per_second
            == entry["dram"]["bandwidth_bytes_per_second"]
        )
        assert generated.noc_count == entry["noc"]["count"]
        assert (
            generated.noc_bytes_per_cycle_per_direction
            == entry["noc"]["l1_bytes_per_cycle_per_direction"]
        )
        assert generated.power_watts == entry["power_watts"]


def test_default_device_is_n150() -> None:
    assert devices.DEFAULT_DEVICE == "n150"
    assert devices.lookup(devices.DEFAULT_DEVICE).name == "n150"
    assert ", ".join(devices.DEVICES) == "n150, n300, p100a, p150a"


def test_wormhole_values() -> None:
    n150 = devices.lookup("n150")
    assert n150.arch == "wormhole_b0"
    assert (n150.compute_grid_x, n150.compute_grid_y) == (8, 8)
    assert n150.chips == 1
    assert n150.total_cores == 64
    assert n150.l1_usable_bytes == 1499136
    assert n150.clock_hz == 1000000000
    assert n150.flops_per_cycle_per_core == 4096
    assert n150.dram_bytes == 12 * 1024**3
    assert n150.dram_bandwidth_bytes_per_second == 288_000_000_000
    assert n150.power_watts == 160

    n300 = devices.lookup("n300")
    assert (n300.compute_grid_x, n300.compute_grid_y) == (8, 7)
    assert n300.chips == 2
    assert n300.total_cores == 112
    assert n300.l1_usable_bytes == n150.l1_usable_bytes
    assert n300.dram_bandwidth_bytes_per_second == 576_000_000_000


def test_blackhole_values() -> None:
    p100a = devices.lookup("p100a")
    assert p100a.arch == "blackhole"
    assert (p100a.compute_grid_x, p100a.compute_grid_y) == (12, 10)
    assert p100a.total_cores == 120
    assert p100a.l1_usable_bytes == 1572864
    assert p100a.clock_hz == 1_350_000_000
    assert p100a.dram_bandwidth_bytes_per_second == 448_000_000_000
    assert p100a.dram_bytes == 28 * 1024**3

    p150a = devices.lookup("p150a")
    assert p150a.dram_bandwidth_bytes_per_second == 512_000_000_000
    assert p150a.dram_bytes == 32 * 1024**3


def test_aliases_and_unknown_names() -> None:
    assert devices.lookup("n150d") is devices.lookup("n150")
    assert devices.lookup("bh_p100a") is devices.lookup("p100a")
    with pytest.raises(KeyError, match="valid devices"):
        devices.lookup("bogus")


def test_cb_budget_and_depth_invariant() -> None:
    """The default three-buffer allocation must fit every published device."""
    for device in devices.all_devices():
        budget = device.cb_budget()
        assert budget == device.l1_usable_bytes - devices.DEFAULT_L1_RESERVE_BYTES
        assert budget > 0
        default_bytes = (
            3 * devices.DEFAULT_CB_DEPTH_TILES * device.tile_bytes
        )
        assert default_bytes <= budget
        assert budget // (3 * device.tile_bytes) >= devices.DEFAULT_CB_DEPTH_TILES
