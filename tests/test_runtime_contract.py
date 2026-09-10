"""Failing-first contract tests for the Apple Silicon runtime boundary."""

from pathlib import Path

import pytest

from kimodo_mlx.runtime import AssetManifest, RuntimeConfig, generate


def test_missing_motion_asset_is_rejected_before_inference(tmp_path: Path) -> None:
    """Given a missing motion asset, generation fails with a typed error."""
    manifest = AssetManifest(motion=tmp_path / "missing.gguf", text=None)

    with pytest.raises(FileNotFoundError):
        generate(
            prompt="walk forward",
            manifest=manifest,
            config=RuntimeConfig(seed=42, steps=1, backend="fixture"),
        )


def test_generation_refuses_placeholder_inference(tmp_path: Path) -> None:
    """Given an unconverted fixture, generation refuses a false result."""
    motion = tmp_path / "motion.gguf"
    motion.write_bytes(b"GGUFfixture")
    manifest = AssetManifest(motion=motion, text=None)

    with pytest.raises(RuntimeError, match="refusing placeholder inference"):
        generate(
            prompt="walk forward",
            manifest=manifest,
            config=RuntimeConfig(seed=42, steps=1, backend="fixture"),
        )
