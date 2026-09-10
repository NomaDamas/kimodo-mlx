#!/usr/bin/env python3
"""Paired Apple Silicon benchmark for kimodo-mlx vs kimodo.cpp."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from kimodo_mlx.motion import MotionModel
from kimodo_mlx.runtime import diagnose
from kimodo_mlx.text_encoder import TextEncoder


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", type=Path, default=Path("models/nvidia-soma-rp-v1.1"))
    parser.add_argument("--text", type=Path, default=Path("models/llm2vec-text-bundle/generated/llm2vec-text-bundle"))
    parser.add_argument("--prompt", default="walk forward\n")
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("evidence/benchmark.json"))
    parser.add_argument("--cpp-binary", type=Path, default=Path(".cache/kimodo-build-metal/kmd-generate"))
    parser.add_argument("--cpp-motion", type=Path, default=Path("models/models/kimodo-soma-rp-v1.1-f32.gguf"))
    args = parser.parse_args()
    diagnostics = diagnose("auto")
    encoder = TextEncoder(args.text)
    model = MotionModel.load(args.motion)
    encoder.encode(args.prompt)
    model.sample(np.zeros(4096, dtype=np.float32), frames=args.frames, steps=1, seed=args.seed)
    encode_times: list[float] = []
    sample_times: list[float] = []
    embedding = None
    result = None
    for _ in range(args.runs):
        started = time.perf_counter()
        embedding = encoder.encode(args.prompt)
        encode_times.append(time.perf_counter() - started)
        started = time.perf_counter()
        result = model.sample(embedding, frames=args.frames, steps=args.steps, seed=args.seed)
        sample_times.append(time.perf_counter() - started)
    cpp_seconds: float | None = None
    if args.cpp_binary.is_file() and args.cpp_motion.is_file():
        prompt_path = Path("/tmp/kimodo-mlx-bench-prompt.txt")
        prompt_path.write_text(args.prompt, encoding="utf-8")
        out_dir = Path("/tmp/kimodo-mlx-bench-cpp")
        out_dir.mkdir(parents=True, exist_ok=True)
        command = [
            str(args.cpp_binary),
            str(args.cpp_motion),
            str(args.text),
            str(prompt_path),
            str(args.frames),
            str(args.steps),
            str(args.seed),
            str(out_dir),
        ]
        started = time.perf_counter()
        subprocess.run(command, check=True, capture_output=True, text=True)
        cpp_seconds = time.perf_counter() - started
    assert embedding is not None and result is not None
    payload = {
        "status": "pass",
        "diagnostics": diagnostics,
        "frames": args.frames,
        "steps": args.steps,
        "seed": args.seed,
        "mlx_warm_encode_s": encode_times,
        "mlx_warm_sample_s": sample_times,
        "mlx_warm_e2e_mean_s": _mean(encode_times) + _mean(sample_times),
        "cpp_e2e_s": cpp_seconds,
        "faster": cpp_seconds is not None and (_mean(encode_times) + _mean(sample_times)) < cpp_seconds,
        "embedding_norm": float(np.linalg.norm(embedding)),
        "root_mean": result.root_positions.mean(axis=0).tolist(),
        "neural_engine_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
