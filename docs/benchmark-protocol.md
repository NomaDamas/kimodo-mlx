# Benchmark protocol

The benchmark compares the unmodified `kimodo.cpp` runtime with the patched
Metal runtime only after both use the same NVIDIA checkpoint revision, GGUF
hash, prompt, seed, frame count, diffusion steps, CFG values, precision, and
warm/cold procedure.

The primary execution command is:

```sh
KIMODO_BACKEND=metal .cache/kimodo-build-metal/kmd-generate \
  --model models/kimodo-soma-rp-v1.1-f32.gguf \
  --text-bundle generated/llm2vec-text-bundle \
  --prompt "walk forward" --frames 150 --steps 100 --seed 42
```

The exact flags may be confirmed with `kmd-generate --help` for the pinned
upstream revision. Record stdout, stderr, elapsed time, selected backend,
model hashes, and output tensor hashes in `evidence/`.

The benchmark is not considered a pass without the real F32 motion GGUF and
the separate LLM2Vec bundle. This repository never vendors either asset.
Quality parity is measured from local rotations and root positions using
maximum absolute error, mean absolute error, and fixed-seed output hashes.
Performance is reported for at least five warm runs after one cold run. A
missing asset produces a blocked artifact, not a fabricated speedup.
