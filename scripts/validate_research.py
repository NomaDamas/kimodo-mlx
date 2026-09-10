#!/usr/bin/env python3
"""Validate the machine-consumed structure of the Kimodo research report."""

from __future__ import annotations

import sys
from pathlib import Path

REQUIRED_HEADINGS = (
    "## Executive decision",
    "## Existing implementations",
    "## Apple Silicon backend decision",
    "## Parity hazards",
    "## Licensing and reuse boundary",
    "## Sources",
)


def validate_report(path: Path) -> list[str]:
    """Return missing report requirements without interpreting prose claims."""
    if not path.is_file():
        return [f"missing report: {path}"]

    text = path.read_text(encoding="utf-8")
    return [
        f"missing required heading: {heading}"
        for heading in REQUIRED_HEADINGS
        if heading not in text
    ]


def main() -> int:
    """Validate one report path and print machine-readable failures."""
    path = Path(sys.argv[1]) if len(sys.argv) == 2 else Path("docs/research.md")
    errors = validate_report(path)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"valid research report: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
