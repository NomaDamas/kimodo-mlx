"""Kimodo asset validation and deterministic Apple backend dispatch."""

from __future__ import annotations

import importlib.util
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

GGUF_MAGIC: Final[bytes] = b"GGUF"


@dataclass(frozen=True, slots=True)
class AssetManifest:
    """Paths to the motion and optional text assets."""

    motion: Path
    text: Path | None


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Deterministic runtime parameters."""

    seed: int
    steps: int
    backend: str = "auto"


@dataclass(frozen=True, slots=True)
class Generation:
    """Portable result used by tests and benchmark tooling."""

    output: bytes
    elapsed_ms: float
    backend: str


def _backend_name(requested: str) -> str:
    if requested != "auto":
        return requested
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        if importlib.util.find_spec("mlx.core") is not None:
            return "mlx-metal"
        if importlib.util.find_spec("torch") is not None:
            return "torch-mps"
    return "unavailable"


def _validate_gguf(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as handle:
        magic = handle.read(4)
    if magic != GGUF_MAGIC:
        raise ValueError(f"unsupported motion asset (expected GGUF): {path}")


def diagnose(requested: str = "auto") -> dict[str, str | bool]:
    """Return observable device and backend capability diagnostics."""
    selected = _backend_name(requested)
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "backend_requested": requested,
        "backend_selected": selected,
        "mlx_installed": importlib.util.find_spec("mlx.core") is not None,
        "torch_installed": importlib.util.find_spec("torch") is not None,
        "neural_engine_used": False,
    }


def generate(
    *,
    prompt: str,
    manifest: AssetManifest,
    config: RuntimeConfig,
) -> Generation:
    """Validate assets and refuse to claim inference before the adapter exists."""
    _validate_gguf(manifest.motion)
    backend = _backend_name(config.backend)
    if backend == "unavailable":
        raise RuntimeError(
            "no Apple backend installed; install MLX and provide Kimodo GGUF assets"
        )
    raise RuntimeError(
        f"Kimodo tensor adapter is not implemented for backend {backend}; "
        "refusing placeholder inference"
    )
