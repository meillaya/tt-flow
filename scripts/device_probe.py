#!/usr/bin/env python3
"""Probe a Tenstorrent device and compare it with the published device spec.

The project has no card in CI, so this script reports honestly instead of
pretending. It exits 2 when the probe cannot run at all (tt-smi missing or no
/dev/tenstorrent/N node), 1 when a device is present but its board, firmware,
or KMD does not match spec/devices.json, and 0 when the probe matches.

The board name comes from `tt-smi -s` JSON. An unrecognized board is reported
verbatim and treated as a mismatch; the script never invents a device entry.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = REPO_ROOT / "spec" / "devices.json"

EXIT_OK = 0
EXIT_MISMATCH = 1
EXIT_NO_DEVICE = 2

DEVICE_NODE_GLOB = "/dev/tenstorrent/*"
# tt-smi snapshot JSON uses these keys across tt-smi 5.x releases. Accept the
# documented spellings and ignore unknown extra keys.
BOARD_KEYS = ("board_type", "board_name", "board")
FIRMWARE_KEYS = ("fw_version", "firmware_version", "fw")
KMD_KEYS = ("tt_kmd_version", "kmd_version", "tt_kmd")


def fail(message: str, exit_code: int) -> NoReturn:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(exit_code)


def load_spec(spec_path: Path) -> dict:
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
    except OSError as error:
        fail(f"cannot read {spec_path}: {error}", EXIT_MISMATCH)
    except json.JSONDecodeError as error:
        fail(f"{spec_path} is not valid JSON: {error}", EXIT_MISMATCH)
    if "devices" not in spec or not spec["devices"]:
        fail(f"{spec_path} has no devices", EXIT_MISMATCH)
    return spec


def first_value(record: dict, keys: tuple[str, ...]) -> str | None:
    """First present, non-empty string value among the accepted key spellings."""
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def find_device_nodes() -> list[str]:
    """Device nodes under /dev/tenstorrent that look like per-device numbers."""
    root = Path("/dev/tenstorrent")
    if not root.is_dir():
        return []
    nodes = []
    for entry in sorted(root.iterdir()):
        if entry.name.isdigit():
            nodes.append(str(entry))
    return nodes


def run_snapshot(tt_smi: str) -> dict:
    """Run `tt-smi -s` and parse its JSON. tt-smi prints the snapshot to stdout."""
    try:
        completed = subprocess.run(
            [tt_smi, "-s"],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except OSError as error:
        fail(f"cannot run {tt_smi}: {error}", EXIT_NO_DEVICE)
    except subprocess.TimeoutExpired:
        fail(f"{tt_smi} -s timed out after 60 seconds", EXIT_NO_DEVICE)

    output = completed.stdout.strip()
    if completed.returncode != 0 or not output:
        detail = completed.stderr.strip() or "no output"
        fail(f"{tt_smi} -s failed (exit {completed.returncode}): {detail}",
             EXIT_NO_DEVICE)

    # tt-smi prints only the JSON document, but tolerate a leading or trailing
    # banner line by slicing the outermost object.
    start = output.find("{")
    end = output.rfind("}")
    if start < 0 or end <= start:
        fail(f"{tt_smi} -s did not print a JSON object", EXIT_NO_DEVICE)
    try:
        snapshot = json.loads(output[start : end + 1])
    except json.JSONDecodeError as error:
        fail(f"cannot parse {tt_smi} -s JSON: {error}", EXIT_NO_DEVICE)
    if not isinstance(snapshot, dict):
        fail(f"{tt_smi} -s JSON is not an object", EXIT_NO_DEVICE)
    return snapshot


def match_device(board: str, devices: dict[str, dict]) -> str | None:
    """Resolve a reported board string to a device name or alias.

    Match is case-insensitive and substring-based because tt-smi reports names
    such as "tenstorrent_n150". Longest match wins so n150 does not shadow a
    longer alias.
    """
    lowered = board.lower()
    best_name = None
    best_length = 0
    for name, entry in devices.items():
        candidates = [name, *entry.get("aliases", [])]
        for candidate in candidates:
            if candidate.lower() in lowered and len(candidate) > best_length:
                best_name = name
                best_length = len(candidate)
    return best_name


def version_tuple(text: str) -> tuple[int, ...] | None:
    """Leading dotted numeric segments of a version string, else None."""
    match = re.match(r"\s*(\d+(?:\.\d+)*)", text)
    if not match:
        return None
    try:
        return tuple(int(part) for part in match.group(1).split("."))
    except ValueError:
        return None


def version_at_least(reported: str, minimum: str) -> bool:
    left = version_tuple(reported)
    right = version_tuple(minimum)
    if left is None:
        return False
    # Pad the shorter tuple so "2.8" compares equal to "2.8.0".
    width = max(len(left), len(right))
    left_padded = left + (0,) * (width - len(left))
    right_padded = right + (0,) * (width - len(right))
    return left_padded >= right_padded


def toolchain_minimums(entry: dict) -> dict:
    return entry.get("toolchain", {})


def probe(spec_path: Path) -> int:
    spec = load_spec(spec_path)
    devices = spec["devices"]

    tt_smi = shutil.which("tt-smi")
    nodes = find_device_nodes()
    if tt_smi is None or not nodes:
        missing = []
        if tt_smi is None:
            missing.append("tt-smi is not on PATH")
        if not nodes:
            missing.append(f"no device node matches {DEVICE_NODE_GLOB}")
        fail(
            "no usable Tenstorrent device for the probe: "
            + "; ".join(missing)
            + ". This machine has no card, so nothing was measured. "
            "Install the toolchain (Ubuntu 22.04, IOMMU enabled, tt-kmd 2.8.0+, "
            "firmware 19.8.1, tt-smi 5.0.0+) and re-run, or treat all run "
            "results as unverified without hardware.",
            EXIT_NO_DEVICE,
        )

    snapshot = run_snapshot(tt_smi)
    records = snapshot.get("device_info")
    if not isinstance(records, list) or not records:
        fail(f"{tt_smi} -s reported no device_info entries", EXIT_MISMATCH)

    print(f"probe: tt-smi {tt_smi}")
    print(f"probe: {len(nodes)} device node(s): {', '.join(nodes)}")
    print(f"probe: host OS {platform.platform()}")

    failures = 0
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            print(f"device {index}: snapshot entry is not an object", file=sys.stderr)
            failures += 1
            continue
        board = first_value(record, BOARD_KEYS)
        if board is None:
            print(f"device {index}: snapshot has no board name; not guessing",
                  file=sys.stderr)
            failures += 1
            continue
        device_name = match_device(board, devices)
        if device_name is None:
            print(
                f"device {index}: unknown board '{board}'; add a sourced entry to "
                f"spec/devices.json before modelling it (known devices: "
                f"{', '.join(devices)})",
                file=sys.stderr,
            )
            failures += 1
            continue

        entry = devices[device_name]
        toolchain = toolchain_minimums(entry)
        print(f"device {index}: board '{board}' -> spec entry '{device_name}' "
              f"({entry['arch']}, {entry['chips']} chip(s), compute grid "
              f"{entry['compute']['grid_x']}x{entry['compute']['grid_y']})")

        firmware = first_value(record, FIRMWARE_KEYS)
        if firmware is None:
            print("  firmware: not reported; cannot verify", file=sys.stderr)
            failures += 1
        else:
            minimum = toolchain.get("firmware_min", "")
            if version_at_least(firmware, minimum):
                print(f"  firmware: {firmware} (minimum {minimum})")
            else:
                print(f"  firmware: {firmware} is below the minimum {minimum}",
                      file=sys.stderr)
                failures += 1

        kmd = first_value(record, KMD_KEYS)
        if kmd is None:
            print("  tt-kmd: not reported; cannot verify", file=sys.stderr)
            failures += 1
        else:
            minimum = toolchain.get("kmd_min", "")
            if version_at_least(kmd, minimum):
                print(f"  tt-kmd: {kmd} (minimum {minimum})")
            else:
                print(f"  tt-kmd: {kmd} is below the minimum {minimum}",
                      file=sys.stderr)
                failures += 1

    if failures:
        print(
            f"probe: {failures} check(s) failed; device does not match the "
            "published spec",
            file=sys.stderr,
        )
        return EXIT_MISMATCH
    print("probe: device matches spec/devices.json")
    return EXIT_OK


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe a Tenstorrent card and compare it with spec/devices.json."
    )
    parser.add_argument(
        "--spec",
        type=Path,
        default=SPEC_PATH,
        help=f"device spec JSON (default: {SPEC_PATH})",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    return probe(args.spec)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
