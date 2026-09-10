"""Motion checkpoint IO for native safetensors and F32 GGUF."""
from __future__ import annotations
from pathlib import Path
import numpy as np


def load_motion_tensors(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    p = Path(path)
    if p.is_dir():
        from safetensors.numpy import load_file
        raw = load_file(str(p / "model.safetensors"))
        tensors = {k.removeprefix("denoiser.backbone."): np.asarray(v, dtype=np.float32) for k, v in raw.items()}
        stats = p / "stats" / "motion"
        for group in ("global_root", "local_root", "body"):
            for stat in ("mean", "std"):
                tensors[f"stats.{group}.{stat}"] = np.asarray(np.load(stats / group / f"{stat}.npy"), dtype=np.float32)
        return tensors, {"kimodo.skeleton": "soma30", "kimodo.motion_dim": 369, "kimodo.body_dim": 364,
                        "kimodo.text_embedding_width": 4096, "kimodo.fps": 30, "kimodo.layers": 16,
                        "kimodo.hidden_size": 1024, "kimodo.heads": 8, "kimodo.num_text_tokens": 50,
                        "kimodo.base_diffusion_steps": 1000}
    from gguf import GGUFReader
    reader = GGUFReader(str(p))
    tensors = {field.name: np.asarray(field.data, dtype=np.float32).reshape(tuple(reversed(field.shape)))
               for field in reader.tensors}
    meta: dict[str, object] = {}
    for key, field in reader.fields.items():
        value = field.parts[-1]
        if isinstance(value, np.generic): value = value.item()
        meta[key] = value
    return tensors, meta
