#!/usr/bin/env python3
"""Fail on the house style's banned constructions.

Prose in this repository, including code comments, is held to one rule set so
that nobody has to remember it and no review has to argue about it. The checks
here are the mechanical ones. Everything else is a matter of taste and stays
that way.

    tools/check-prose.py            # the whole tree
    tools/check-prose.py README.md
"""

import argparse
import re
import sys
from pathlib import Path

# Extensions whose prose is held to the rules. Generated data is excluded:
# a JSON plan full of store path names is not prose and would only make noise.
CHECKED_SUFFIXES = {
    ".md",
    ".py",
    ".sh",
    ".nix",
    ".yml",
    ".yaml",
    ".cpp",
    ".h",
    ".hpp",
    ".td",
    ".mlir",
}

# Directory names whose contents are build output, vendored dependencies, or
# tool caches rather than project prose.
SKIP_DIRS = {
    ".git",
    ".jj",
    ".omo",
    ".direnv",
    ".devenv",
    ".deps",
    ".lit",
    "_spike",
    ".venv",
    "__pycache__",
    ".mypy_cache",
    "node_modules",
    "result",
    "target",
    "data",
}

# Build trees are named build, build-baseline, build-generated, and so on.
BUILD_DIR_PREFIX = "build"


def is_skipped(rel: Path) -> bool:
    """True when any path component names a directory this checker ignores."""
    for part in rel.parts[:-1]:
        if part in SKIP_DIRS or part.startswith(BUILD_DIR_PREFIX):
            return True
    return False


# The loudest signal of machine written text. Matched case insensitively on a
# word boundary.
BANNED_WORDS = [
    "delve",
    "tapestry",
    "meticulous",
    "pivotal",
    "intricate",
    "interplay",
    "underscore",
    "garner",
    "bolster",
    "vibrant",
    "bustling",
    "multifaceted",
    "seamless",
    "commendable",
    "ever-evolving",
    "realm",
    "testament",
    "showcase",
    "foster",
    "unlock",
    "elevate",
    "embark",
    "robust",
    "crucial",
    "essential",
    "profound",
    "nuanced",
    "holistic",
    "myriad",
    "plethora",
    "leverage",
]

# Connective tics and stock phrases.
BANNED_PHRASES = [
    "it is important to note",
    "it is worth noting",
    "it should be noted",
    "keep in mind that",
    "plays a crucial role",
    "stands as a testament",
    "in the heart of",
    "nestled in",
    "hidden gem",
    "it just works",
    "batteries included",
    "zero config",
    "from the ground up",
    "first-class citizen",
    "game changer",
    "despite these challenges",
    "challenges remain",
    "only time will tell",
    "to be honest",
    "let me be clear",
    "that is the whole point",
    "not just",
    "not only",
]

# An em dash. The style allows them rarely; this repository does without.
EM_DASH = "—"


def offenders(path):
    """Every (line number, rule, line) this file breaks."""
    found = []
    for n, line in enumerate(path.read_text(errors="replace").splitlines(), start=1):
        if EM_DASH in line:
            found.append((n, "em dash", line.strip()))

        lowered = line.lower()
        for phrase in BANNED_PHRASES:
            if phrase in lowered:
                found.append((n, f"phrase {phrase!r}", line.strip()))

        for word in BANNED_WORDS:
            if re.search(rf"\b{re.escape(word)}\b", lowered):
                found.append((n, f"word {word!r}", line.strip()))

    return found


def walk(root):
    """Every checkable file under root, minus build and tool directories."""
    for path in sorted(root.rglob("*")):
        if is_skipped(path.relative_to(root)):
            continue
        if path.is_file() and path.suffix in CHECKED_SUFFIXES:
            yield path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "paths", nargs="*", help="files to check; the whole tree when omitted"
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    if args.paths:
        paths: list[Path] = []
        for raw in args.paths:
            candidate = Path(raw)
            if candidate.is_dir():
                paths.extend(walk(candidate.resolve()))
            elif candidate.is_file():
                paths.append(candidate)
            else:
                sys.exit(f"check-prose: no such file or directory: {raw}")
    else:
        paths = list(walk(root))

    # This file names every banned word, so the rules cannot be applied to it.
    exempt = {Path(__file__).resolve()}
    paths = [p for p in paths if p.resolve() not in exempt]

    total = 0
    for path in paths:
        for n, rule, line in offenders(path):
            rel = (
                path.resolve().relative_to(root)
                if path.resolve().is_relative_to(root)
                else path
            )
            print(f"{rel}:{n}: {rule}\n    {line}")
            total += 1

    if total:
        print(f"\n{total} violation(s)", file=sys.stderr)
        sys.exit(1)

    # A check that read nothing is not a check that passed.
    if not paths:
        sys.exit(f"check-prose: no files to check under {root}")

    print(f"{len(paths)} files checked, no violations")


if __name__ == "__main__":
    main()
