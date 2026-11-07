#!/usr/bin/env python3
"""Generate the C++ and Python device specification tables.

The values live in `spec/devices.json`, which records one source URL and one
retrieval date per numeric group. This script validates that file, then writes:

  include/tensixflow/Generated/DeviceSpecs.h
  python/tensixflow/devices.py

The generated files are committed so the C++ build needs no Python. Run
`--check` in CI and in the `check-device-spec` CMake target to prove the
generated files match the JSON source.

Exit codes: 0 success, 1 invalid or stale spec, 2 usage error (for example an
unknown device name passed to `--device`).
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
import tempfile
from datetime import date
from pathlib import Path
from typing import NoReturn

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = REPO_ROOT / "spec" / "devices.json"
HEADER_PATH = REPO_ROOT / "include" / "tensixflow" / "Generated" / "DeviceSpecs.h"
PYTHON_PATH = REPO_ROOT / "python" / "tensixflow" / "devices.py"

SCHEMA_VERSION = 1
BYTES_PER_BF16 = 2
TILE_HEIGHT = 32
TILE_WIDTH = 32
TILE_DTYPE = "bf16"
FLOP_PER_MAC = 2
DISPATCH_AXES = ("row", "col")

# The source id of every numeric group. A device entry must map each group to a
# source id from the top-level table, so every published number has one origin.
SOURCED_GROUPS = (
    "arch",
    "die",
    "sram_bytes",
    "compute",
    "l1",
    "clock_hz",
    "matrix_unit",
    "dram",
    "noc",
    "pcie",
    "power_watts",
    "ethernet",
    "toolchain",
)


def fail(message: str, exit_code: int = 1) -> NoReturn:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(exit_code)


def require_int(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        fail(f"{path}: expected an integer, got {value!r}")
    return value


def require_positive_int(value: object, path: str) -> int:
    number = require_int(value, path)
    if number <= 0:
        fail(f"{path}: expected a positive integer, got {number}")
    return number


def require_str(value: object, path: str) -> str:
    if not isinstance(value, str) or not value:
        fail(f"{path}: expected a non-empty string, got {value!r}")
    return value


def require_list_of_str(value: object, path: str) -> list[str]:
    if not isinstance(value, list):
        fail(f"{path}: expected a list of strings, got {value!r}")
    for index, item in enumerate(value):
        require_str(item, f"{path}[{index}]")
    return list(value)


def require_object(value: object, path: str) -> dict:
    if not isinstance(value, dict):
        fail(f"{path}: expected an object, got {value!r}")
    return value


def require_keys(obj: dict, keys: tuple[str, ...], path: str) -> None:
    for key in keys:
        if key not in obj:
            fail(f"{path}: missing required key '{key}'")


def validate_source(source: dict, index: int) -> dict:
    path = f"sources[{index}]"
    require_keys(source, ("id", "url", "retrieved", "supports"), path)
    source_id = require_str(source["id"], f"{path}.id")
    url = require_str(source["url"], f"{path}.url")
    if not url.startswith("https://"):
        fail(f"{path}.url: expected an https URL, got '{url}'")
    retrieved = require_str(source["retrieved"], f"{path}.retrieved")
    try:
        date.fromisoformat(retrieved)
    except ValueError:
        fail(f"{path}.retrieved: expected YYYY-MM-DD, got '{retrieved}'")
    require_str(source["supports"], f"{path}.supports")
    return source


def validate_model(model: dict) -> dict:
    require_keys(model, ("cb_depth_tiles", "l1_reserve_bytes"), "model")
    require_positive_int(model["cb_depth_tiles"], "model.cb_depth_tiles")
    require_int(model["l1_reserve_bytes"], "model.l1_reserve_bytes")
    return model


def validate_tile(tile: dict, path: str) -> dict:
    require_keys(tile, ("height", "width", "dtype", "bytes"), path)
    height = require_positive_int(tile["height"], f"{path}.height")
    width = require_positive_int(tile["width"], f"{path}.width")
    dtype = require_str(tile["dtype"], f"{path}.dtype")
    tile_bytes = require_positive_int(tile["bytes"], f"{path}.bytes")
    # The scheduler and the generated header hard-code the 32x32 bf16 tile, so
    # require the JSON to agree instead of emitting a wider model.
    if (height, width, dtype) != (TILE_HEIGHT, TILE_WIDTH, TILE_DTYPE):
        fail(
            f"{path}: expected a {TILE_HEIGHT}x{TILE_WIDTH} {TILE_DTYPE} tile, "
            f"got {height}x{width} {dtype}"
        )
    expected = height * width * BYTES_PER_BF16
    if tile_bytes != expected:
        fail(f"{path}.bytes: expected {expected}, got {tile_bytes}")
    return tile


def validate_device_entry(name: str, device: dict, source_ids: set[str]) -> dict:
    path = f"devices.{name}"
    require_keys(
        device,
        (
            "aliases",
            "arch",
            "status",
            "chips",
            "die",
            "sram_bytes",
            "compute",
            "l1",
            "tile",
            "clock_hz",
            "matrix_unit",
            "dram",
            "noc",
            "pcie",
            "power_watts",
            "ethernet",
            "toolchain",
            "sources",
            "notes",
        ),
        path,
    )
    require_list_of_str(device["aliases"], f"{path}.aliases")
    require_str(device["arch"], f"{path}.arch")
    require_str(device["status"], f"{path}.status")
    require_positive_int(device["chips"], f"{path}.chips")

    die = require_object(device["die"], f"{path}.die")
    require_keys(
        die,
        (
            "grid_x",
            "grid_y",
            "tensix_workers_per_die",
            "harvested_tensix_workers_per_die",
        ),
        f"{path}.die",
    )
    require_positive_int(die["grid_x"], f"{path}.die.grid_x")
    require_positive_int(die["grid_y"], f"{path}.die.grid_y")
    require_positive_int(
        die["tensix_workers_per_die"], f"{path}.die.tensix_workers_per_die"
    )
    require_positive_int(
        die["harvested_tensix_workers_per_die"],
        f"{path}.die.harvested_tensix_workers_per_die",
    )
    if die["harvested_tensix_workers_per_die"] > die["tensix_workers_per_die"]:
        fail(f"{path}.die: harvested workers exceed the die worker count")

    compute = require_object(device["compute"], f"{path}.compute")
    require_keys(
        compute, ("grid_x", "grid_y", "dispatch_axis", "core_descriptor"), f"{path}.compute"
    )
    require_positive_int(compute["grid_x"], f"{path}.compute.grid_x")
    require_positive_int(compute["grid_y"], f"{path}.compute.grid_y")
    dispatch_axis = require_str(compute["dispatch_axis"], f"{path}.compute.dispatch_axis")
    if dispatch_axis not in DISPATCH_AXES:
        fail(f"{path}.compute.dispatch_axis: expected one of {DISPATCH_AXES}, got '{dispatch_axis}'")
    require_str(compute["core_descriptor"], f"{path}.compute.core_descriptor")
    compute_cores = compute["grid_x"] * compute["grid_y"]
    if compute_cores > die["harvested_tensix_workers_per_die"]:
        fail(f"{path}.compute: compute grid {compute_cores} cores exceeds the harvested worker count")

    l1 = require_object(device["l1"], f"{path}.l1")
    require_keys(l1, ("usable_bytes", "nominal_bytes"), f"{path}.l1")
    usable = require_positive_int(l1["usable_bytes"], f"{path}.l1.usable_bytes")
    nominal = require_positive_int(l1["nominal_bytes"], f"{path}.l1.nominal_bytes")
    if usable > nominal:
        fail(f"{path}.l1: usable bytes exceed nominal bytes")

    tile = validate_tile(require_object(device["tile"], f"{path}.tile"), f"{path}.tile")

    require_positive_int(device["sram_bytes"], f"{path}.sram_bytes")
    require_positive_int(device["clock_hz"], f"{path}.clock_hz")
    require_positive_int(device["power_watts"], f"{path}.power_watts")
    require_list_of_str(device["ethernet"], f"{path}.ethernet")

    matrix_unit = require_object(device["matrix_unit"], f"{path}.matrix_unit")
    require_keys(
        matrix_unit,
        ("macs_per_instruction", "flops_per_cycle_per_core", "fidelity_phases"),
        f"{path}.matrix_unit",
    )
    macs = require_positive_int(
        matrix_unit["macs_per_instruction"], f"{path}.matrix_unit.macs_per_instruction"
    )
    flops = require_positive_int(
        matrix_unit["flops_per_cycle_per_core"],
        f"{path}.matrix_unit.flops_per_cycle_per_core",
    )
    require_positive_int(
        matrix_unit["fidelity_phases"], f"{path}.matrix_unit.fidelity_phases"
    )
    if flops != FLOP_PER_MAC * macs:
        fail(f"{path}.matrix_unit: flops_per_cycle_per_core must be {FLOP_PER_MAC} x macs")

    dram = require_object(device["dram"], f"{path}.dram")
    require_keys(
        dram,
        ("bytes", "bandwidth_bytes_per_second", "type"),
        f"{path}.dram",
    )
    require_positive_int(dram["bytes"], f"{path}.dram.bytes")
    require_positive_int(
        dram["bandwidth_bytes_per_second"], f"{path}.dram.bandwidth_bytes_per_second"
    )
    require_str(dram["type"], f"{path}.dram.type")

    noc = require_object(device["noc"], f"{path}.noc")
    require_keys(
        noc,
        ("count", "flit_bytes", "max_burst_bytes", "l1_bytes_per_cycle_per_direction"),
        f"{path}.noc",
    )
    require_positive_int(noc["count"], f"{path}.noc.count")
    require_positive_int(noc["flit_bytes"], f"{path}.noc.flit_bytes")
    require_positive_int(noc["max_burst_bytes"], f"{path}.noc.max_burst_bytes")
    require_positive_int(
        noc["l1_bytes_per_cycle_per_direction"],
        f"{path}.noc.l1_bytes_per_cycle_per_direction",
    )

    pcie = require_object(device["pcie"], f"{path}.pcie")
    require_keys(pcie, ("generation", "lanes"), f"{path}.pcie")
    require_positive_int(pcie["generation"], f"{path}.pcie.generation")
    require_positive_int(pcie["lanes"], f"{path}.pcie.lanes")

    toolchain = require_object(device["toolchain"], f"{path}.toolchain")
    require_keys(
        toolchain, ("os", "kmd_min", "firmware_min", "tt_smi_min", "iommu"), f"{path}.toolchain"
    )
    for key in ("os", "kmd_min", "firmware_min", "tt_smi_min", "iommu"):
        require_str(toolchain[key], f"{path}.toolchain.{key}")

    notes = require_list_of_str(device["notes"], f"{path}.notes")
    if not notes:
        fail(f"{path}.notes: expected at least one note")

    sources = require_object(device["sources"], f"{path}.sources")
    for group in SOURCED_GROUPS:
        if group not in sources:
            fail(f"{path}.sources: missing source id for group '{group}'")
        source_id = require_str(sources[group], f"{path}.sources.{group}")
        if source_id not in source_ids:
            fail(f"{path}.sources.{group}: unknown source id '{source_id}'")

    return device


def validate_spec(spec: dict) -> dict:
    version = require_int(spec.get("schema_version"), "schema_version")
    if version != SCHEMA_VERSION:
        fail(f"schema_version: expected {SCHEMA_VERSION}, got {version}")
    retrieved = require_str(spec.get("retrieved"), "retrieved")
    try:
        date.fromisoformat(retrieved)
    except ValueError:
        fail(f"retrieved: expected YYYY-MM-DD, got '{retrieved}'")

    sources = spec.get("sources")
    if not isinstance(sources, list) or not sources:
        fail("sources: expected a non-empty list")
    source_ids: set[str] = set()
    for index, source in enumerate(sources):
        validated = validate_source(require_object(source, f"sources[{index}]"), index)
        source_id = validated["id"]
        if source_id in source_ids:
            fail(f"sources[{index}].id: duplicate source id '{source_id}'")
        source_ids.add(source_id)

    devices = spec.get("devices")
    if not isinstance(devices, dict) or not devices:
        fail("devices: expected a non-empty object")
    names = set()
    aliases: dict[str, str] = {}
    for name, device in devices.items():
        if not isinstance(name, str) or not name:
            fail(f"devices: invalid device name {name!r}")
        validate_device_entry(name, require_object(device, f"devices.{name}"), source_ids)
        names.add(name)
        for alias in device["aliases"]:
            if alias in aliases:
                fail(f"devices.{name}.aliases: alias '{alias}' also names '{aliases[alias]}'")
            aliases[alias] = name

    default_device = require_str(spec.get("default_device"), "default_device")
    if default_device not in names:
        fail(f"default_device: '{default_device}' is not in devices")
    validate_model(require_object(spec.get("model"), "model"))
    return spec


def load_spec(spec_path: Path) -> dict:
    try:
        text = spec_path.read_text(encoding="utf-8")
    except OSError as error:
        fail(f"cannot read {spec_path}: {error}")
    try:
        spec = json.loads(text)
    except json.JSONDecodeError as error:
        fail(f"{spec_path} is not valid JSON: {error}")
    return validate_spec(require_object(spec, "spec"))


def cpp_int(value: int) -> str:
    return f"{value}"


def emit_header(spec: dict) -> str:
    model = spec["model"]
    default_device = spec["default_device"]
    names = list(spec["devices"].keys())
    lines: list[str] = []
    lines.append("//===- DeviceSpecs.h - Generated Tenstorrent device table --------*- C++ -*-===//")
    lines.append("//")
    lines.append("// Generated by tools/gen_device_spec.py from spec/devices.json. Do not edit.")
    lines.append("// Edit spec/devices.json, then run")
    lines.append("//   python3 tools/gen_device_spec.py")
    lines.append("// and prove the result matches with")
    lines.append("//   python3 tools/gen_device_spec.py --check")
    lines.append("//")
    lines.append(f"// Retrieved {spec['retrieved']}. Every numeric group in spec/devices.json")
    lines.append("// cites one source URL and the retrieval date.")
    lines.append("//")
    lines.append("//===----------------------------------------------------------------------===//")
    lines.append("")
    lines.append("#ifndef TENSIXFLOW_GENERATED_DEVICESPECS_H")
    lines.append("#define TENSIXFLOW_GENERATED_DEVICESPECS_H")
    lines.append("")
    lines.append('#include "llvm/ADT/ArrayRef.h"')
    lines.append('#include "llvm/ADT/StringRef.h"')
    lines.append("")
    lines.append("#include <cstddef>")
    lines.append("#include <cstdint>")
    lines.append("")
    lines.append("namespace mlir {")
    lines.append("namespace tensixflow {")
    lines.append("namespace device {")
    lines.append("")
    lines.append("/// Per-die Tensix compute grid plus the board's die count. The scheduler")
    lines.append("/// plans one die at a time; totalCores reports the whole product.")
    lines.append("struct CoreGrid {")
    lines.append("  int64_t gridX = 0;")
    lines.append("  int64_t gridY = 0;")
    lines.append("  int64_t chips = 1;")
    lines.append("")
    lines.append("  int64_t totalCores() const { return chips * gridX * gridY; }")
    lines.append("};")
    lines.append("")
    lines.append("/// One published device specification. The values are static planning")
    lines.append("/// inputs, never measured performance.")
    lines.append("struct DeviceSpec {")
    lines.append("  const char *name;")
    lines.append("  const char *arch;")
    lines.append("  CoreGrid computeGrid;")
    lines.append("  int64_t l1UsableBytes;")
    lines.append("  int64_t l1NominalBytes;")
    lines.append("  int64_t tileHeight;")
    lines.append("  int64_t tileWidth;")
    lines.append("  int64_t tileBytes;")
    lines.append("  int64_t clockHz;")
    lines.append("  int64_t flopsPerCyclePerCore;")
    lines.append("  int64_t dramBytes;")
    lines.append("  int64_t dramBandwidthBytesPerSecond;")
    lines.append("  int64_t nocCount;")
    lines.append("  int64_t nocFlitBytes;")
    lines.append("  int64_t nocMaxBurstBytes;")
    lines.append("  int64_t nocBytesPerCyclePerDirection;")
    lines.append("  int64_t powerWatts;")
    lines.append("")
    lines.append("  /// Bytes of per-core L1 left for circular buffers after reserveBytes of")
    lines.append("  /// code, stack, and runtime bookkeeping. Never negative.")
    lines.append("  int64_t cbBudget(int64_t reserveBytes) const;")
    lines.append("};")
    lines.append("")
    lines.append("inline int64_t DeviceSpec::cbBudget(int64_t reserveBytes) const {")
    lines.append("  const int64_t budget = l1UsableBytes - reserveBytes;")
    lines.append("  if (budget <= 0) {")
    lines.append("    return 0;")
    lines.append("  }")
    lines.append("  return budget;")
    lines.append("}")
    lines.append("")
    lines.append("/// Device selected when the driver gets no --tensixflow-device option.")
    lines.append(f"constexpr const char kDefaultDeviceName[] = \"{default_device}\";")
    lines.append("")
    lines.append("/// Planning reserve taken out of usable L1 before circular buffers are sized.")
    lines.append(
        f"constexpr int64_t kDefaultL1ReserveBytes = {cpp_int(model['l1_reserve_bytes'])};"
    )
    lines.append("")
    lines.append("/// Tiles per circular buffer that tt-metal examples start from.")
    lines.append(
        f"constexpr int64_t kDefaultCBDepthTiles = {cpp_int(model['cb_depth_tiles'])};"
    )
    lines.append("")
    lines.append(f"constexpr std::size_t kNumDevices = {len(names)};")
    lines.append("")
    lines.append('constexpr const char kValidDeviceNames[] = "' + ", ".join(names) + '";')
    lines.append("")
    lines.append("inline constexpr DeviceSpec kDevices[] = {")
    for name in names:
        device = spec["devices"][name]
        die = device["die"]
        compute = device["compute"]
        l1 = device["l1"]
        tile = device["tile"]
        matrix_unit = device["matrix_unit"]
        dram = device["dram"]
        noc = device["noc"]
        lines.append("    DeviceSpec{")
        lines.append(f'        .name = "{name}",')
        lines.append(f'        .arch = "{device["arch"]}",')
        lines.append("        .computeGrid = CoreGrid{")
        lines.append(f"            .gridX = {cpp_int(compute['grid_x'])},")
        lines.append(f"            .gridY = {cpp_int(compute['grid_y'])},")
        lines.append(f"            .chips = {cpp_int(device['chips'])},")
        lines.append("        },")
        lines.append(f"        .l1UsableBytes = {cpp_int(l1['usable_bytes'])},")
        lines.append(f"        .l1NominalBytes = {cpp_int(l1['nominal_bytes'])},")
        lines.append(f"        .tileHeight = {cpp_int(tile['height'])},")
        lines.append(f"        .tileWidth = {cpp_int(tile['width'])},")
        lines.append(f"        .tileBytes = {cpp_int(tile['bytes'])},")
        lines.append(f"        .clockHz = {cpp_int(device['clock_hz'])},")
        lines.append(
            f"        .flopsPerCyclePerCore = {cpp_int(matrix_unit['flops_per_cycle_per_core'])},")
        lines.append(f"        .dramBytes = {cpp_int(dram['bytes'])},")
        lines.append(
            f"        .dramBandwidthBytesPerSecond = {cpp_int(dram['bandwidth_bytes_per_second'])},")
        lines.append(f"        .nocCount = {cpp_int(noc['count'])},")
        lines.append(f"        .nocFlitBytes = {cpp_int(noc['flit_bytes'])},")
        lines.append(f"        .nocMaxBurstBytes = {cpp_int(noc['max_burst_bytes'])},")
        lines.append(
            f"        .nocBytesPerCyclePerDirection = "
            f"{cpp_int(noc['l1_bytes_per_cycle_per_direction'])},")
        lines.append(f"        .powerWatts = {cpp_int(device['power_watts'])},")
        lines.append("    },")
    lines.append("};")
    lines.append("")
    lines.append("inline llvm::ArrayRef<DeviceSpec> allDevices() {")
    lines.append("  return llvm::ArrayRef<DeviceSpec>(kDevices, kNumDevices);")
    lines.append("}")
    lines.append("")
    lines.append("/// One alternate product name accepted by lookup().")
    lines.append("struct DeviceAlias {")
    lines.append("  const char *alias;")
    lines.append("  const char *device;")
    lines.append("};")
    lines.append("")
    lines.append(f"constexpr std::size_t kNumAliases = {sum(len(spec['devices'][name]['aliases']) for name in names)};")
    lines.append("")
    lines.append("inline constexpr DeviceAlias kAliases[] = {")
    for name in names:
        for alias in spec["devices"][name]["aliases"]:
            lines.append(f'    DeviceAlias{{"{alias}", "{name}"}},')
    lines.append("};")
    lines.append("")
    lines.append("/// Look up a device by name or alias. Returns nullptr for an unknown name.")
    lines.append("inline const DeviceSpec *lookup(llvm::StringRef name) {")
    lines.append("  for (const DeviceSpec &spec : allDevices()) {")
    lines.append("    if (name == spec.name) {")
    lines.append("      return &spec;")
    lines.append("    }")
    lines.append("  }")
    lines.append("  for (const DeviceAlias &alias : kAliases) {")
    lines.append("    if (name == alias.alias) {")
    lines.append("      return lookup(alias.device);")
    lines.append("    }")
    lines.append("  }")
    lines.append("  return nullptr;")
    lines.append("}")
    lines.append("")
    lines.append("inline llvm::StringRef defaultDeviceName() { return kDefaultDeviceName; }")
    lines.append("")
    lines.append("inline const DeviceSpec *defaultDevice() { return lookup(defaultDeviceName()); }")
    lines.append("")
    lines.append("/// Comma-separated list for diagnostics about an unknown device name.")
    lines.append("inline llvm::StringRef validDeviceNames() { return kValidDeviceNames; }")
    lines.append("")
    lines.append("} // namespace device")
    lines.append("} // namespace tensixflow")
    lines.append("} // namespace mlir")
    lines.append("")
    lines.append("#endif // TENSIXFLOW_GENERATED_DEVICESPECS_H")
    lines.append("")
    return "\n".join(lines)


def emit_python(spec: dict) -> str:
    model = spec["model"]
    default_device = spec["default_device"]
    names = list(spec["devices"].keys())
    lines: list[str] = []
    lines.append('"""Generated Tenstorrent device specifications. Do not edit.')
    lines.append("")
    lines.append("Generated by tools/gen_device_spec.py from spec/devices.json.")
    lines.append(f"Retrieved {spec['retrieved']}. Edit spec/devices.json, then run")
    lines.append("`python3 tools/gen_device_spec.py` and prove the result with")
    lines.append("`python3 tools/gen_device_spec.py --check`.")
    lines.append('"""')
    lines.append("")
    lines.append("from __future__ import annotations")
    lines.append("")
    lines.append("from dataclasses import dataclass")
    lines.append("")
    lines.append(f'DEFAULT_DEVICE = "{default_device}"')
    lines.append(f"DEFAULT_L1_RESERVE_BYTES = {model['l1_reserve_bytes']}")
    lines.append(f"DEFAULT_CB_DEPTH_TILES = {model['cb_depth_tiles']}")
    lines.append("")
    lines.append("")
    lines.append("@dataclass(frozen=True)")
    lines.append("class DeviceSpec:")
    lines.append('    """One published device specification, a static planning input."""')
    lines.append("")
    lines.append("    name: str")
    lines.append("    arch: str")
    lines.append("    compute_grid_x: int")
    lines.append("    compute_grid_y: int")
    lines.append("    chips: int")
    lines.append("    l1_usable_bytes: int")
    lines.append("    l1_nominal_bytes: int")
    lines.append("    tile_height: int")
    lines.append("    tile_width: int")
    lines.append("    tile_bytes: int")
    lines.append("    clock_hz: int")
    lines.append("    flops_per_cycle_per_core: int")
    lines.append("    dram_bytes: int")
    lines.append("    dram_bandwidth_bytes_per_second: int")
    lines.append("    noc_count: int")
    lines.append("    noc_flit_bytes: int")
    lines.append("    noc_max_burst_bytes: int")
    lines.append("    noc_bytes_per_cycle_per_direction: int")
    lines.append("    power_watts: int")
    lines.append("")
    lines.append("    @property")
    lines.append("    def total_cores(self) -> int:")
    lines.append('        """Tensix cores across the board\'s dies."""')
    lines.append("        return self.chips * self.compute_grid_x * self.compute_grid_y")
    lines.append("")
    lines.append(
        "    def cb_budget(self, reserve_bytes: int = DEFAULT_L1_RESERVE_BYTES) -> int:"
    )
    lines.append('        """Per-core L1 bytes left for circular buffers after the reserve."""')
    lines.append("        return max(0, self.l1_usable_bytes - reserve_bytes)")
    lines.append("")
    lines.append("")
    lines.append("DEVICES: dict[str, DeviceSpec] = {")
    for name in names:
        device = spec["devices"][name]
        compute = device["compute"]
        l1 = device["l1"]
        tile = device["tile"]
        matrix_unit = device["matrix_unit"]
        dram = device["dram"]
        noc = device["noc"]
        lines.append(f'    "{name}": DeviceSpec(')
        lines.append(f'        name="{name}",')
        lines.append(f'        arch="{device["arch"]}",')
        lines.append(f"        compute_grid_x={compute['grid_x']},")
        lines.append(f"        compute_grid_y={compute['grid_y']},")
        lines.append(f"        chips={device['chips']},")
        lines.append(f"        l1_usable_bytes={l1['usable_bytes']},")
        lines.append(f"        l1_nominal_bytes={l1['nominal_bytes']},")
        lines.append(f"        tile_height={tile['height']},")
        lines.append(f"        tile_width={tile['width']},")
        lines.append(f"        tile_bytes={tile['bytes']},")
        lines.append(f"        clock_hz={device['clock_hz']},")
        lines.append(
            f"        flops_per_cycle_per_core={matrix_unit['flops_per_cycle_per_core']},"
        )
        lines.append(f"        dram_bytes={dram['bytes']},")
        lines.append(
            f"        dram_bandwidth_bytes_per_second={dram['bandwidth_bytes_per_second']},"
        )
        lines.append(f"        noc_count={noc['count']},")
        lines.append(f"        noc_flit_bytes={noc['flit_bytes']},")
        lines.append(f"        noc_max_burst_bytes={noc['max_burst_bytes']},")
        lines.append(
            f"        noc_bytes_per_cycle_per_direction={noc['l1_bytes_per_cycle_per_direction']},"
        )
        lines.append(f"        power_watts={device['power_watts']},")
        lines.append("    ),")
    lines.append("}")
    lines.append("")
    lines.append("ALIASES: dict[str, str] = {")
    for name in names:
        for alias in spec["devices"][name]["aliases"]:
            lines.append(f'    "{alias}": "{name}",')
    lines.append("}")
    lines.append("")
    lines.append("")
    lines.append("def valid_device_names() -> str:")
    lines.append('    """Comma-separated device names for error messages."""')
    lines.append('    return ", ".join(DEVICES)')
    lines.append("")
    lines.append("")
    lines.append("def lookup(name: str) -> DeviceSpec:")
    lines.append('    """Return the spec for a name or alias; raise KeyError if unknown."""')
    lines.append("    if name in DEVICES:")
    lines.append("        return DEVICES[name]")
    lines.append("    resolved = ALIASES.get(name)")
    lines.append("    if resolved is None:")
    lines.append(
        "        raise KeyError(f\"unknown device {name!r}; valid devices: {valid_device_names()}\")"
    )
    lines.append("    return DEVICES[resolved]")
    lines.append("")
    lines.append("")
    lines.append("def all_devices() -> tuple[DeviceSpec, ...]:")
    lines.append('    """Every device entry in source order."""')
    lines.append("    return tuple(DEVICES.values())")
    lines.append("")
    return "\n".join(lines)


def generated_files(spec: dict) -> dict[Path, str]:
    return {
        HEADER_PATH: emit_header(spec),
        PYTHON_PATH: emit_python(spec),
    }


def write_generated(files: dict[Path, str]) -> None:
    for path, text in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"wrote {path}")


def check_generated(spec: dict) -> int:
    files = generated_files(spec)
    with tempfile.TemporaryDirectory(prefix="tensixflow-device-spec-") as temp_dir:
        stale = 0
        for path, text in files.items():
            temp_path = Path(temp_dir) / path.name
            temp_path.write_text(text, encoding="utf-8")
            expected = path.read_text(encoding="utf-8") if path.exists() else ""
            actual = temp_path.read_text(encoding="utf-8")
            if expected == actual:
                print(f"up to date: {path}")
                continue
            stale += 1
            print(f"stale: {path}", file=sys.stderr)
            for line in difflib.unified_diff(
                expected.splitlines(),
                actual.splitlines(),
                fromfile=f"{path} (on disk)",
                tofile=f"{path} (from spec/devices.json)",
                lineterm="",
            ):
                print(line, file=sys.stderr)
        if stale:
            print(
                "error: generated device artifacts are stale; run "
                "python3 tools/gen_device_spec.py",
                file=sys.stderr,
            )
            return 1
    return 0


def find_device(spec: dict, name: str) -> dict:
    devices = spec["devices"]
    if name in devices:
        return devices[name]
    for device_name, device in devices.items():
        if name in device["aliases"]:
            return devices[device_name]
    fail(
        f"unknown device '{name}'; valid devices: {', '.join(devices)}",
        exit_code=2,
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the device specification tables from spec/devices.json."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="regenerate into a temporary directory and fail if the committed files differ",
    )
    parser.add_argument(
        "--device",
        metavar="NAME",
        help="print one device entry from spec/devices.json as JSON",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="print the device names and exit",
    )
    parser.add_argument(
        "--spec",
        type=Path,
        default=SPEC_PATH,
        help=f"device spec JSON (default: {SPEC_PATH})",
    )
    parser.add_argument("--header", type=Path, default=HEADER_PATH, help="C++ header output path")
    parser.add_argument("--python", type=Path, default=PYTHON_PATH, help="Python module output path")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    spec = load_spec(args.spec)
    if args.list:
        for name in spec["devices"]:
            print(name)
        return 0
    if args.device:
        print(json.dumps(find_device(spec, args.device), indent=2))
        return 0
    if args.check:
        return check_generated(spec)
    if args.header != HEADER_PATH or args.python != PYTHON_PATH:
        files = {
            args.header: emit_header(spec),
            args.python: emit_python(spec),
        }
        write_generated(files)
        return 0
    write_generated(generated_files(spec))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
