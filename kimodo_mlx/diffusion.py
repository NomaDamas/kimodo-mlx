"""Cosine diffusion schedule, DDIM, and separated CFG matching kimodo.cpp."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class DiffusionSchedule:
    """Selected DDIM timesteps and cumulative alphas."""

    use_timesteps: NDArray[np.int32]
    alpha_cumprod: NDArray[np.float32]
    alpha_cumprod_prev: NDArray[np.float32]


def make_cosine_schedule(base_steps: int, sample_steps: int) -> DiffusionSchedule:
    """Build the native cosine schedule used by kimodo.cpp."""
    if base_steps < 2 or sample_steps < 1 or sample_steps > base_steps:
        raise ValueError("invalid diffusion schedule step count")

    def alpha_bar(step: float) -> float:
        return math.cos((step + 0.008) / 1.008 * math.pi / 2) ** 2

    base_alpha = np.empty(base_steps, dtype=np.float32)
    cumulative = 1.0
    for index in range(base_steps):
        beta = min(
            1.0 - alpha_bar((index + 1) / base_steps) / alpha_bar(index / base_steps),
            0.999,
        )
        cumulative *= 1.0 - beta
        base_alpha[index] = cumulative
    stride = (base_steps - 1) / max(1, sample_steps - 1)
    use_timesteps = np.empty(sample_steps, dtype=np.int32)
    alpha_cumprod = np.empty(sample_steps, dtype=np.float32)
    alpha_cumprod_prev = np.empty(sample_steps, dtype=np.float32)
    previous = 1.0
    for index in range(sample_steps):
        timestep = min(int(math.floor(index * stride + 0.5)), base_steps - 1)
        alpha = max(float(base_alpha[timestep]), 1e-9)
        use_timesteps[index] = timestep
        alpha_cumprod[index] = alpha
        alpha_cumprod_prev[index] = previous
        previous = alpha
    return DiffusionSchedule(use_timesteps, alpha_cumprod, alpha_cumprod_prev)


def ddim_step(
    schedule: DiffusionSchedule,
    index: int,
    x_t: NDArray[np.float32],
    pred: NDArray[np.float32],
    out: NDArray[np.float32],
) -> None:
    """Apply one deterministic eta=0 DDIM update into out."""
    if index < 0 or index >= schedule.alpha_cumprod.shape[0]:
        raise ValueError("invalid DDIM input")
    alpha = float(schedule.alpha_cumprod[index])
    previous = float(schedule.alpha_cumprod_prev[index])
    reciprocal = 1.0 / math.sqrt(alpha)
    reciprocal_m1 = math.sqrt((1.0 - alpha) / alpha)
    epsilon = (reciprocal * x_t - pred) / reciprocal_m1
    out[:] = pred * math.sqrt(previous) + math.sqrt(1.0 - previous) * epsilon


def separated_cfg(
    text: NDArray[np.float32],
    constraint: NDArray[np.float32],
    uncond: NDArray[np.float32],
    text_weight: float,
    constraint_weight: float,
) -> NDArray[np.float32]:
    """Combine [text, constraint, uncond] branches in native order."""
    return np.asarray(
        uncond + text_weight * (text - uncond) + constraint_weight * (constraint - uncond),
        dtype=np.float32,
    )
