#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the llm-service project
"""Ensure Python sources carry the expected SPDX header.

The hook verifies that tracked files contain both the license and copyright
lines. When a header is missing, we insert the required lines automatically,
mirroring the behaviour of the upstream vLLM project.
"""

from __future__ import annotations

import sys
from enum import Enum
from pathlib import Path

LICENSE_LINE = "# SPDX-License-Identifier: Apache-2.0"
COPYRIGHT_LINE = "# SPDX-FileCopyrightText: Copyright contributors to the llm-service project"
FULL_HEADER = f"{LICENSE_LINE}\n{COPYRIGHT_LINE}"


class SPDXStatus(Enum):
    EMPTY = "empty"
    COMPLETE = "complete"
    MISSING_LICENSE = "missing_license"
    MISSING_COPYRIGHT = "missing_copyright"
    MISSING_BOTH = "missing_both"


def evaluate_status(path: Path) -> SPDXStatus:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return SPDXStatus.MISSING_BOTH

    if not lines:
        return SPDXStatus.EMPTY

    idx = 1 if lines[0].startswith("#!") else 0

    has_license = False
    has_copyright = False
    for line in lines[idx:]:
        stripped = line.strip()
        if stripped == LICENSE_LINE:
            has_license = True
        elif stripped == COPYRIGHT_LINE:
            has_copyright = True

    if has_license and has_copyright:
        return SPDXStatus.COMPLETE
    if has_license:
        return SPDXStatus.MISSING_COPYRIGHT
    if has_copyright:
        return SPDXStatus.MISSING_LICENSE
    return SPDXStatus.MISSING_BOTH


def insert_header(path: Path, status: SPDXStatus) -> None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return

    shebang = ""
    body = lines
    if lines and lines[0].startswith("#!"):
        shebang, body = lines[0], lines[1:]

    new_lines: list[str] = []
    if shebang:
        new_lines.append(shebang)

    if status == SPDXStatus.MISSING_BOTH:
        new_lines.append(FULL_HEADER)
        new_lines.extend(body)
    else:
        inserted_license = False
        inserted_copyright = False
        for line in body:
            stripped = line.strip()
            if stripped == LICENSE_LINE:
                inserted_license = True
                new_lines.append(line)
                if status == SPDXStatus.MISSING_COPYRIGHT:
                    new_lines.append(COPYRIGHT_LINE)
                continue
            if stripped == COPYRIGHT_LINE:
                inserted_copyright = True
                if (
                    status == SPDXStatus.MISSING_LICENSE
                    and not inserted_license
                ):
                    new_lines.append(LICENSE_LINE)
                    inserted_license = True
                new_lines.append(line)
                continue
            new_lines.append(line)

        if status == SPDXStatus.MISSING_LICENSE and not inserted_license:
            new_lines.insert(1 if shebang else 0, LICENSE_LINE)
        if status == SPDXStatus.MISSING_COPYRIGHT and not inserted_copyright:
            new_lines.insert(2 if shebang else 1, COPYRIGHT_LINE)

    path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def main(argv: list[str]) -> int:
    missing: list[Path] = []
    for filepath in argv:
        path = Path(filepath)
        status = evaluate_status(path)
        if status in {SPDXStatus.COMPLETE, SPDXStatus.EMPTY}:
            continue
        missing.append(path)
        insert_header(path, status)

    if missing:
        print(
            "The following files were missing the SPDX header and were updated:"
        )
        for path in missing:
            print(f"  {path}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
