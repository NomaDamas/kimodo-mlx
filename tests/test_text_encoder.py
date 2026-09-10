"""LLM2Vec tokenizer and embedding parity against the native C++ encoder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kimodo_mlx.text_encoder import TextEncoder

BUNDLE = Path("models/llm2vec-text-bundle/generated/llm2vec-text-bundle")
REFERENCE = Path("/tmp/kimodo-mlx-qa/ref_embed.f32")


@pytest.mark.skipif(not (BUNDLE / "tokenizer.gguf").is_file(), reason="LLM2Vec bundle is not downloaded")
def test_tokenizer_prefixes_bos_and_is_stable() -> None:
    """Given a short prompt, tokenization is deterministic and starts with BOS."""
    encoder = TextEncoder(BUNDLE)

    first = encoder.tokenize("walk forward")
    second = encoder.tokenize("walk forward")

    assert first[0] == 128000
    assert first == second
    assert 2 <= len(first) <= 512


@pytest.mark.skipif(not REFERENCE.is_file(), reason="C++ embedding fixture is missing")
def test_embedding_matches_native_cpp_vector() -> None:
    """Given the same prompt, MLX pooled embedding matches kmd-encode within tolerance."""
    encoder = TextEncoder(BUNDLE)
    reference = np.fromfile(REFERENCE, dtype=np.float32)
    assert reference.shape == (4096,)

    actual = encoder.encode("walk forward\n")

    cosine = float(np.dot(actual, reference) / (np.linalg.norm(actual) * np.linalg.norm(reference)))
    max_abs = float(np.max(np.abs(actual - reference)))
    rmse = float(np.sqrt(np.mean((actual - reference) ** 2)))
    assert cosine > 0.99
    assert rmse < 0.02
    assert max_abs < 0.1
