"""Kimodo asset validation and Apple Silicon generation."""

from __future__ import annotations

import importlib.util
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

from kimodo_mlx.motion import MotionModel
from kimodo_mlx.text_encoder import TextEncoder

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
    frames: int = 30


@dataclass(frozen=True, slots=True)
class Generation:
    """Portable result used by tests and benchmark tooling."""

    output: bytes
    elapsed_ms: float
    backend: str
    local_rotations_xyzw: object
    root_positions: object


def _backend_name(requested: str) -> str:
    if requested != "auto":
        return requested
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        if importlib.util.find_spec("mlx.core") is not None:
            return "mlx-metal"
        if importlib.util.find_spec("torch") is not None:
            return "torch-mps"
    return "unavailable"


def _validate_motion(path: Path) -> None:
    if path.is_dir():
        if not (path / "model.safetensors").is_file():
            raise FileNotFoundError(path / "model.safetensors")
        return
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
    """Load assets, encode the prompt, and sample motion on the Apple GPU."""
    _validate_motion(manifest.motion)
    backend = _backend_name(config.backend)
    if backend == "unavailable":
        raise RuntimeError("no Apple backend installed; install MLX and provide Kimodo assets")
    if manifest.text is None:
        raise ValueError("text bundle is required for prompt generation")
    started = time.perf_counter()
    encoder = TextEncoder(manifest.text)
    embedding = encoder.encode(prompt)
    model = MotionModel.load(manifest.motion)
    result = model.sample(embedding, frames=config.frames, steps=config.steps, seed=config.seed)
    elapsed_ms = (time.perf_counter() - started) * 1000
    packed = np.concatenate([result.local_rotations_xyzw.reshape(-1), result.root_positions.reshape(-1)])
    return Generation(
        output=np.asarray(packed, dtype=np.float32).tobytes(),
        elapsed_ms=elapsed_ms,
        backend=backend,
        local_rotations_xyzw=result.local_rotations_xyzw,
        root_positions=result.root_positions,
    )
