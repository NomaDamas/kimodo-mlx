"""Reproducible backend and paired-output benchmark utilities."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from kimodo_mlx.runtime import AssetManifest, RuntimeConfig, diagnose, generate


def main(argv: list[str] | None = None) -> int:
    """Run a measured fixture benchmark and emit JSON evidence."""
    parser = argparse.ArgumentParser(prog="kimodo-mlx-benchmark")
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--prompt", default="walk forward")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    diagnostics = diagnose(args.backend)
    started = time.perf_counter()
    try:
        result = generate(
            prompt=args.prompt,
            manifest=AssetManifest(motion=args.motion, text=None),
            config=RuntimeConfig(
                seed=args.seed,
                steps=args.steps,
                backend=args.backend,
            ),
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        payload = {
            "status": "blocked",
            "reason": str(error),
            "diagnostics": diagnostics,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, sort_keys=True))
        return 2
    payload = {
        "status": "pass",
        "diagnostics": diagnostics,
        "backend": result.backend,
        "elapsed_ms": (time.perf_counter() - started) * 1000,
        "output_sha256": result.output.hex(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
