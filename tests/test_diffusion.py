"""Failing-first diffusion/DDIM/CFG parity against the native C++ schedule."""

from __future__ import annotations

import math

import numpy as np

from kimodo_mlx.diffusion import ddim_step, make_cosine_schedule, separated_cfg


def _cpp_cosine_schedule(base_steps: int, sample_steps: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    return use_timesteps, alpha_cumprod, alpha_cumprod_prev


def test_cosine_schedule_selects_native_timesteps() -> None:
    """Given 1000/100 steps, selected timesteps match native round-stride sampling."""
    expected_t, expected_alpha, expected_prev = _cpp_cosine_schedule(1000, 100)

    schedule = make_cosine_schedule(1000, 100)

    np.testing.assert_array_equal(schedule.use_timesteps, expected_t)
    np.testing.assert_allclose(schedule.alpha_cumprod, expected_alpha, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(schedule.alpha_cumprod_prev, expected_prev, rtol=1e-6, atol=1e-6)


def test_ddim_step_matches_native_eta0_arithmetic() -> None:
    """Given one DDIM index, the update matches the native eta=0 formula."""
    schedule = make_cosine_schedule(1000, 100)
    x_t = np.linspace(-1.5, 1.5, 24, dtype=np.float32)
    pred = np.linspace(-0.5, 0.75, 24, dtype=np.float32)
    index = 37
    alpha = float(schedule.alpha_cumprod[index])
    previous = float(schedule.alpha_cumprod_prev[index])
    reciprocal = 1.0 / math.sqrt(alpha)
    reciprocal_m1 = math.sqrt((1.0 - alpha) / alpha)
    expected = pred * math.sqrt(previous) + math.sqrt(1.0 - previous) * (
        (reciprocal * x_t - pred) / reciprocal_m1
    )

    actual = np.empty_like(expected)
    ddim_step(schedule, index, x_t, pred, actual)

    np.testing.assert_allclose(actual, expected.astype(np.float32), rtol=1e-5, atol=1e-5)


def test_separated_cfg_matches_text_constraint_uncond_order() -> None:
    """Given three CFG branches, combination follows native text/constraint/uncond order."""
    text = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    constraint = np.array([0.5, 0.0, -1.0], dtype=np.float32)
    uncond = np.array([0.25, 0.5, 0.75], dtype=np.float32)
    expected = uncond + 2.0 * (text - uncond) + 1.5 * (constraint - uncond)

    actual = separated_cfg(text, constraint, uncond, 2.0, 1.5)

    np.testing.assert_allclose(actual, expected, rtol=0, atol=0)
