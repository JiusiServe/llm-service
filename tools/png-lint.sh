#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the LM-Service project

# Ensure that *.excalidraw.png files keep their embedded scene metadata.
# This allows round-tripping diagrams back into Excalidraw for future edits.

set -euo pipefail

find . -iname '*.excalidraw.png' | while read -r file; do
    if git check-ignore -q "$file"; then
        continue
    fi
    if ! grep -q "excalidraw+json" "$file"; then
        echo "$file was not exported from Excalidraw with 'Embed Scene' enabled."
        exit 1
    fi
done
