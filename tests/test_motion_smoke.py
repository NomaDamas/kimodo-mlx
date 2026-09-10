"""Real-checkpoint smoke tests for the MLX motion denoiser."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kimodo_mlx.motion import MotionModel

CHECKPOINT = Path("models/nvidia-soma-rp-v1.1")


@pytest.mark.skipif(not (CHECKPOINT / "model.safetensors").is_file(), reason="SOMA checkpoint is not downloaded")
def test_one_denoise_step_is_finite_and_correct_shape() -> None:
    """Given the SOMA checkpoint, one denoiser step returns finite [1, T, 369]."""
    model = MotionModel.load(CHECKPOINT)
    frames = 8
    motion = np.zeros((1, frames, model.motion_dim), dtype=np.float32)
    embedding = np.zeros(4096, dtype=np.float32)

    output = model.denoise(motion, embedding, timestep=500)

    assert output.shape == (1, frames, 369)
    assert np.isfinite(output).all()
