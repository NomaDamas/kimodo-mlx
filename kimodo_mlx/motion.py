"""MLX two-stage Kimodo motion denoiser, DDIM sampler, and decoder."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
from numpy.typing import NDArray

from kimodo_mlx.diffusion import ddim_step, make_cosine_schedule, separated_cfg
from kimodo_mlx.layers import as_mx, encoder_layer, linear, sinusoidal_encoding
from kimodo_mlx.motion_io import load_motion_tensors

TEXT_TOKENS = 50
HIDDEN = 1024
HEADS = 8
LAYERS = 16
SOMA_PARENTS = (
    -1, 0, 1, 2, 3, 4, 5, 6, 6, 6, 3, 10, 11, 12, 13, 13, 3, 16, 17, 18, 19, 19, 0, 22, 23, 24, 0, 26, 27, 28,
)


@dataclass(frozen=True, slots=True)
class MotionResult:
    """Decoded kinematic output for one generated clip."""

    motion: NDArray[np.float32]
    local_rotations_xyzw: NDArray[np.float32]
    root_positions: NDArray[np.float32]


def _scale(std: NDArray[np.float32]) -> NDArray[np.float32]:
    return np.sqrt(std * std + 1e-5).astype(np.float32)


def silu(x: mx.array) -> mx.array:
    """SiLU used by the timestep MLP."""
    return x * mx.sigmoid(x)


def global_root_to_local_root(
    root: NDArray[np.float32],
    frames: int,
    global_mean: NDArray[np.float32],
    global_std: NDArray[np.float32],
    local_mean: NDArray[np.float32],
    local_std: NDArray[np.float32],
    fps: float = 30.0,
) -> NDArray[np.float32]:
    """Convert normalized global-root features to local-root features."""
    batch = root.shape[0] // (frames * 5)
    global_scale = _scale(global_std)
    local_scale = _scale(local_std)
    result = np.empty((batch * frames, 4), dtype=np.float32)
    for sample in range(batch):
        block = root[sample * frames * 5 : (sample + 1) * frames * 5].reshape(frames, 5)
        x = block[:, 0] * global_scale[0] + global_mean[0]
        y = block[:, 1] * global_scale[1] + global_mean[1]
        z = block[:, 2] * global_scale[2] + global_mean[2]
        angle = np.arctan2(
            block[:, 4] * global_scale[4] + global_mean[4],
            block[:, 3] * global_scale[3] + global_mean[3],
        )
        for time in range(frames):
            nxt = time + 1 if time + 1 < frames else frames - 1
            prev = time if time + 1 < frames else frames - 2
            cos_diff = np.cos(angle[nxt]) * np.cos(angle[prev]) + np.sin(angle[nxt]) * np.sin(angle[prev])
            sin_diff = np.sin(angle[nxt]) * np.cos(angle[prev]) - np.cos(angle[nxt]) * np.sin(angle[prev])
            raw = np.array(
                [fps * np.arctan2(sin_diff, cos_diff), fps * (x[nxt] - x[prev]), fps * (z[nxt] - z[prev]), y[time]],
                dtype=np.float32,
            )
            result[sample * frames + time] = (raw - local_mean) / local_scale
    return result.reshape(-1)


def _sixd_to_matrix(values: NDArray[np.float32]) -> NDArray[np.float32]:
    a = values[:3] / np.linalg.norm(values[:3])
    z = np.cross(a, values[3:6])
    z = z / np.linalg.norm(z)
    b = np.cross(z, a)
    return np.stack((a, b, z), axis=1).astype(np.float32)


def _matrix_to_xyzw(matrix: NDArray[np.float32]) -> NDArray[np.float32]:
    trace = float(matrix[0, 0] + matrix[1, 1] + matrix[2, 2])
    if trace > 0:
        scale = 2.0 * math.sqrt(trace + 1.0)
        return np.array(
            [(matrix[2, 1] - matrix[1, 2]) / scale, (matrix[0, 2] - matrix[2, 0]) / scale, (matrix[1, 0] - matrix[0, 1]) / scale, 0.25 * scale],
            dtype=np.float32,
        )
    if matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = 2.0 * math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
        return np.array(
            [0.25 * scale, (matrix[0, 1] + matrix[1, 0]) / scale, (matrix[0, 2] + matrix[2, 0]) / scale, (matrix[2, 1] - matrix[1, 2]) / scale],
            dtype=np.float32,
        )
    if matrix[1, 1] > matrix[2, 2]:
        scale = 2.0 * math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
        return np.array(
            [(matrix[0, 1] + matrix[1, 0]) / scale, 0.25 * scale, (matrix[1, 2] + matrix[2, 1]) / scale, (matrix[0, 2] - matrix[2, 0]) / scale],
            dtype=np.float32,
        )
    scale = 2.0 * math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
    return np.array(
        [(matrix[0, 2] + matrix[2, 0]) / scale, (matrix[1, 2] + matrix[2, 1]) / scale, 0.25 * scale, (matrix[1, 0] - matrix[0, 1]) / scale],
        dtype=np.float32,
    )


def decode_motion(
    motion: NDArray[np.float32],
    frames: int,
    joints: int,
    global_mean: NDArray[np.float32],
    global_std: NDArray[np.float32],
    body_mean: NDArray[np.float32],
    body_std: NDArray[np.float32],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Unnormalize motion features into local XYZW rotations and root translations."""
    dim = 9 + 12 * joints
    g_scale = _scale(global_std)
    b_scale = _scale(body_std)
    rotations = np.empty((frames, joints, 4), dtype=np.float32)
    roots = np.empty((frames, 3), dtype=np.float32)
    rotation_offset = 5 + 3 * joints
    for time in range(frames):
        features = np.empty(dim, dtype=np.float32)
        features[:5] = motion[time, :5] * g_scale + global_mean
        features[5:] = motion[time, 5:] * b_scale + body_mean
        roots[time, 0] = features[0] + features[5]
        roots[time, 1] = features[6]
        roots[time, 2] = features[2] + features[7]
        global_mats = [
            _sixd_to_matrix(features[rotation_offset + joint * 6 : rotation_offset + (joint + 1) * 6])
            for joint in range(joints)
        ]
        for joint, parent in enumerate(SOMA_PARENTS[:joints]):
            matrix = global_mats[joint] if parent < 0 else global_mats[parent].T @ global_mats[joint]
            rotations[time, joint] = _matrix_to_xyzw(matrix)
    return rotations, roots


class MotionModel:
    """Two-stage Kimodo denoiser running on Apple GPU via MLX."""

    def __init__(self, tensors: dict[str, np.ndarray], meta: dict[str, object]):
        self.tensors = {key: as_mx(np.asarray(value, dtype=np.float32)) for key, value in tensors.items()}
        self.numpy_tensors = {key: np.asarray(value, dtype=np.float32) for key, value in tensors.items()}
        motion_dim = meta.get("kimodo.motion_dim", 369)
        body_dim = meta.get("kimodo.body_dim", 364)
        fps = meta.get("kimodo.fps", 30)
        self.motion_dim = int(str(motion_dim))
        self.body_dim = int(str(body_dim))
        self.joints = (self.motion_dim - 9) // 12
        self.fps = float(str(fps))

    @classmethod
    def load(cls, path: str | Path) -> "MotionModel":
        tensors, meta = load_motion_tensors(path)
        return cls(tensors, meta)

    def _run_transformer(self, prefix: str, motion: mx.array, embedding: mx.array, timesteps: mx.array) -> mx.array:
        batch, frames, _ = motion.shape
        text = mx.zeros((batch, TEXT_TOKENS, 4096))
        text[:, 0, :] = embedding
        encoded_text = linear(text, self.tensors[prefix + "embed_text.weight"], self.tensors[prefix + "embed_text.bias"])
        time = mx.take(sinusoidal_encoding(1000, HIDDEN), timesteps, axis=0)
        time = mx.reshape(time, (batch, 1, HIDDEN))
        time = linear(time, self.tensors[prefix + "embed_timestep.time_embed.0.weight"], self.tensors[prefix + "embed_timestep.time_embed.0.bias"])
        time = linear(silu(time), self.tensors[prefix + "embed_timestep.time_embed.2.weight"], self.tensors[prefix + "embed_timestep.time_embed.2.bias"])
        headings = mx.zeros((batch, 1, 2))
        headings[:, 0, 0] = 1.0
        heading = linear(
            headings,
            self.tensors[prefix + "linear_first_heading_angle.weight"],
            self.tensors[prefix + "linear_first_heading_angle.bias"],
        )
        motion_h = linear(motion, self.tensors[prefix + "input_linear.weight"], self.tensors[prefix + "input_linear.bias"])
        tokens = mx.concatenate([encoded_text, time, heading, motion_h], axis=1)
        tokens = tokens + sinusoidal_encoding(int(tokens.shape[1]), HIDDEN)
        for layer in range(LAYERS):
            tokens = encoder_layer(tokens, self.tensors, prefix + f"seqTransEncoder.layers.{layer}.", HEADS)
        motion_tokens = tokens[:, TEXT_TOKENS + 2 :, :]
        return linear(motion_tokens, self.tensors[prefix + "output_linear.weight"], self.tensors[prefix + "output_linear.bias"])

    def denoise(self, motion: NDArray[np.float32], embedding: NDArray[np.float32], timestep: int) -> NDArray[np.float32]:
        """Run one two-stage denoiser call on a [B, T, D] clip."""
        batch, frames, dim = motion.shape
        if dim != self.motion_dim:
            raise ValueError("motion feature width does not match the checkpoint")
        if embedding.ndim == 1:
            embedding = np.broadcast_to(embedding.astype(np.float32), (batch, 4096))
        x = as_mx(motion)
        text = as_mx(np.ascontiguousarray(embedding, dtype=np.float32))
        times = mx.array(np.full((batch,), timestep, dtype=np.int32))
        concat = mx.concatenate([x, mx.zeros_like(x)], axis=-1)
        root = self._run_transformer("root_model.", concat, text, times)
        mx.eval(root)
        root_np = np.array(root, dtype=np.float32).reshape(batch, frames, 5)
        local = np.empty((batch, frames, 4), dtype=np.float32)
        for sample in range(batch):
            local[sample] = global_root_to_local_root(
                root_np[sample].reshape(-1),
                frames,
                self.numpy_tensors["stats.global_root.mean"],
                self.numpy_tensors["stats.global_root.std"],
                self.numpy_tensors["stats.local_root.mean"],
                self.numpy_tensors["stats.local_root.std"],
                self.fps,
            ).reshape(frames, 4)
        zeros = np.zeros((batch, frames, self.motion_dim), dtype=np.float32)
        body_in = np.concatenate([local, motion[:, :, 5:], zeros], axis=-1)
        body = self._run_transformer("body_model.", as_mx(body_in), text, times)
        mx.eval(body)
        body_np = np.array(body, dtype=np.float32).reshape(batch, frames, self.body_dim)
        return np.concatenate([root_np, body_np], axis=-1).astype(np.float32)

    def sample(
        self,
        embedding: NDArray[np.float32],
        frames: int,
        steps: int,
        seed: int,
        initial_noise: NDArray[np.float32] | None = None,
        text_cfg: float = 2.0,
        constraint_cfg: float = 2.0,
    ) -> MotionResult:
        """Run DDIM sampling and decode local rotations plus root translations."""
        if frames < 1 or frames > 300:
            raise ValueError("frames must be in 1..300")
        schedule = make_cosine_schedule(1000, steps)
        rng = np.random.default_rng(seed)
        motion = initial_noise if initial_noise is not None else rng.standard_normal((frames, self.motion_dim), dtype=np.float32)
        motion = np.ascontiguousarray(motion, dtype=np.float32)
        zero = np.zeros(4096, dtype=np.float32)
        embedding = np.ascontiguousarray(embedding, dtype=np.float32)
        pred = np.empty((frames, self.motion_dim), dtype=np.float32)
        for index in range(steps - 1, -1, -1):
            timestep = int(schedule.use_timesteps[index])
            stacked = np.stack([motion, motion, motion], axis=0)
            texts = np.stack([embedding, zero, zero], axis=0)
            preds = self.denoise(stacked, texts, timestep)
            pred[:] = separated_cfg(preds[0], preds[1], preds[2], text_cfg, constraint_cfg)
            flat = motion.reshape(-1)
            ddim_step(schedule, index, flat, pred.reshape(-1), flat)
        rotations, roots = decode_motion(
            motion,
            frames,
            self.joints,
            self.numpy_tensors["stats.global_root.mean"],
            self.numpy_tensors["stats.global_root.std"],
            self.numpy_tensors["stats.body.mean"],
            self.numpy_tensors["stats.body.std"],
        )
        return MotionResult(motion=motion, local_rotations_xyzw=rotations, root_positions=roots)
