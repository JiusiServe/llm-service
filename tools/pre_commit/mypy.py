#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the LM-Service project
"""Run mypy on the staged files.

Usage: python tools/pre_commit/mypy.py <ci flag> <python version> <paths...>
`<python version>` may be a specific major.minor (e.g. `3.10`) or `auto` to
match the interpreter used to execute the hook.
The hook mirrors the behaviour used in the vLLM project but trimmed for this
repository.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def group_targets(paths: list[str]) -> list[str]:
    targets: set[Path] = set()
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            continue
        if path.is_dir():
            targets.add(path)
        elif path.suffix == ".py":
            targets.add(path)
    if not targets:
        project_root = Path("lm_service")
        if project_root.exists():
            targets.add(project_root)
    return sorted(str(path) for path in targets)


def resolve_python_version(raw: str) -> str:
    if raw.lower() in {"auto", "native"}:
        # Keep the mypy target in sync with the interpreter running the hook.
        return f"{sys.version_info.major}.{sys.version_info.minor}"
    return raw


def run_mypy(ci_mode: bool, python_version: str, targets: list[str]) -> int:
    if not targets:
        return 0

    resolved_version = resolve_python_version(python_version)
    # Use explicit package bases to avoid duplicate module discovery when
    # passing directory targets (e.g. "lm_service").
    args = [
        "mypy",
        "--python-version",
        resolved_version,
        "--explicit-package-bases",
    ]
    if ci_mode:
        args += ["--follow-imports", "silent"]
    else:
        args += ["--follow-imports", "skip"]

    config_path = Path(__file__).with_name("mypy_local.ini")
    if config_path.exists():
        args += ["--config-file", str(config_path)]

    print("$", " ".join(args + targets))
    return subprocess.run(args + targets, check=False).returncode


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        return 0
    ci_mode = argv[0] == "1"
    python_version = argv[1]
    targets = group_targets(argv[2:])
    return run_mypy(ci_mode, python_version, targets)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
