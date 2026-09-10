"""Command-line diagnostics and benchmark entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from kimodo_mlx.runtime import AssetManifest, RuntimeConfig, diagnose, generate


def main(argv: list[str] | None = None) -> int:
    """Run diagnostics or a validated deterministic generation request."""
    parser = argparse.ArgumentParser(prog="kimodo-mlx")
    subparsers = parser.add_subparsers(dest="command", required=True)
    diagnostics = subparsers.add_parser("diagnose")
    diagnostics.add_argument("--backend", default="auto")
    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--prompt", required=True)
    generate_parser.add_argument("--motion", type=Path, required=True)
    generate_parser.add_argument("--text", type=Path)
    generate_parser.add_argument("--seed", type=int, default=42)
    generate_parser.add_argument("--steps", type=int, default=100)
    generate_parser.add_argument("--backend", default="auto")
    args = parser.parse_args(argv)
    try:
        if args.command == "diagnose":
            print(json.dumps(diagnose(args.backend), sort_keys=True))
            return 0
        result = generate(
            prompt=args.prompt,
            manifest=AssetManifest(motion=args.motion, text=args.text),
            config=RuntimeConfig(seed=args.seed, steps=args.steps, backend=args.backend),
        )
        print(
            json.dumps(
                {
                    "backend": result.backend,
                    "elapsed_ms": result.elapsed_ms,
                    "output_sha256": result.output.hex(),
                },
                sort_keys=True,
            )
        )
        return 0
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"kimodo-mlx: {error}", file=sys.stderr)
        return 2
