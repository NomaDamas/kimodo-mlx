"""Tests for the paired benchmark evidence contract."""

import json
from pathlib import Path

from kimodo_mlx.benchmark import main


def test_benchmark_records_blocked_assets_without_claiming_speed(tmp_path: Path) -> None:
    """Given unavailable weights, benchmark emits an explicit blocked artifact."""
    output = tmp_path / "benchmark.json"

    status = main(["--motion", str(tmp_path / "missing.gguf"), "--output", str(output)])

    assert status == 2
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "blocked"
    assert "diagnostics" in payload
