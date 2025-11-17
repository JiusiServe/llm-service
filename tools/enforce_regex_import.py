#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the LM-Service project
"""Disallow importing the standard library ``re`` module directly.

We rely on the third-party ``regex`` package for extended regular
expression support. This hook blocks new ``import re`` statements unless
``regex`` is imported instead.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import regex as re

FORBIDDEN_PATTERNS = re.compile(
    r"^\s*(?:import\s+re(?:$|\s|,)|from\s+re\s+import)"
)
ALLOWED_PATTERNS = [
    re.compile(r"^\s*import\s+regex\s+as\s+re\s*$"),
    re.compile(r"^\s*import\s+regex\s*$"),
]


def get_staged_python_files() -> list[str]:
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=AM"],
            capture_output=True,
            text=True,
            check=True,
        )
        files = (
            result.stdout.strip().split("\n") if result.stdout.strip() else []
        )
        return [f for f in files if f.endswith(".py")]
    except subprocess.CalledProcessError:
        return []


def is_forbidden_import(line: str) -> bool:
    line = line.strip()
    return bool(
        FORBIDDEN_PATTERNS.match(line)
        and not any(pattern.match(line) for pattern in ALLOWED_PATTERNS)
    )


def check_file(filepath: str) -> list[tuple[int, str]]:
    violations: list[tuple[int, str]] = []
    try:
        with open(filepath, encoding="utf-8") as handle:
            for line_num, line in enumerate(handle, 1):
                if is_forbidden_import(line):
                    violations.append((line_num, line.strip()))
    except (OSError, UnicodeDecodeError):
        pass
    return violations


def main() -> int:
    files = get_staged_python_files()
    if not files:
        return 0

    total_violations = 0
    for filepath in files:
        if not Path(filepath).exists():
            continue

        violations = check_file(filepath)
        if violations:
            print(f"\n❌ {filepath}:")
            for line_num, line in violations:
                print(f"  Line {line_num}: {line}")
                total_violations += 1

    if total_violations > 0:
        print(f"\n💡 Found {total_violations} violation(s).")
        print("❌ Please replace 'import re' with 'import regex as re'")
        print(
            "   Also replace 'from re import ...' with 'from regex import ...'"
        )
        print("✅ Allowed imports:")
        print("   - import regex as re")
        print("   - import regex")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
