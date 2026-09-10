# kimodo-mlx

Apple Silicon port of NVIDIA Kimodo. The selected runtime is MLX on Metal
for the motion denoiser and the bidirectional LLM2Vec text encoder. GGML
Metal is retained as a native C++ comparison path. Apple Neural Engine is
not claimed.

## Status

- Motion denoiser: MLX two-stage transformer + DDIM, loads NVIDIA
  `model.safetensors` or LocalAI F32 GGUF.
- Text encoder: MLX bidirectional Llama-3-8B LLM2Vec using the published
  LocalAI GGUF bundle. A causal llama.cpp embedding path is not compatible.
- Benchmarks compare quality (cosine / max-abs on embeddings and motion)
  and wall-clock generation time against `kimodo.cpp` on the same Mac.

## Quick start

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
hf download nvidia/Kimodo-SOMA-RP-v1.1 --local-dir models/nvidia-soma-rp-v1.1
hf download LocalAI-io/Llama-3-Kimodo-GGML --local-dir models/llm2vec-text-bundle
.venv/bin/python -m kimodo_mlx diagnose
.venv/bin/python -m kimodo_mlx generate \
  --prompt "walk forward" \
  --motion models/nvidia-soma-rp-v1.1 \
  --text models/llm2vec-text-bundle/generated/llm2vec-text-bundle \
  --frames 30 --steps 20 --seed 42
```

Weights are not vendored. Review NVIDIA Open Model License and Meta Llama 3
terms before downloading.

## Tests

```sh
.venv/bin/python -m pytest -q
```

## Native Metal comparison

```sh
sh scripts/build_kimodo_metal.sh
```

See `docs/research.md`, `docs/benchmark-protocol.md`, and the measured comparison page `docs/compare.html`.
