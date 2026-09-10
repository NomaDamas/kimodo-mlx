"""Tests for the machine-consumed research report contract."""

from pathlib import Path

from scripts.validate_research import validate_report


def test_research_report_contains_all_decision_sections() -> None:
    """Given the checked-in report, all required decision sections exist."""
    report = Path("docs/research.md")

    errors = validate_report(report)

    assert errors == []
