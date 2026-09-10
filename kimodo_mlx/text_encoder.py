"""MLX LLM2Vec encoder for the Kimodo Meta-Llama-3 text bundle."""
from __future__ import annotations

from pathlib import Path
import math
import re
from typing import Iterable

import mlx.core as mx
import numpy as np
from gguf import GGUFReader


_HIDDEN = 4096
_HEADS = 32
_KV_HEADS = 8
_HEAD_DIM = 128
_FF = 14336
_BOS = 128000


def _bf16(data: np.ndarray, shape: tuple[int, ...]) -> mx.array:
    """Decode a GGUF BF16 payload (the gguf reader exposes it as bytes)."""
    raw = np.asarray(data, dtype=np.uint8).view(np.uint16)
    values = (raw.astype(np.uint32) << 16).view(np.float32).reshape(shape)
    return mx.array(values, dtype=mx.bfloat16)


def _tensor(path: Path, name: str) -> mx.array:
    reader = GGUFReader(str(path), "r")
    for tensor in reader.tensors:
        if tensor.name == name:
            shape = tuple(reversed(tuple(int(x) for x in tensor.shape)))
            if tensor.tensor_type == 30:  # GGML_TYPE_BF16
                return _bf16(tensor.data, shape)
            if tensor.tensor_type == 0:  # GGML_TYPE_F32
                return mx.array(np.asarray(tensor.data, dtype=np.float32).reshape(shape))
            raise ValueError(f"unsupported tensor type {tensor.tensor_type} for {name}")
    raise ValueError(f"{path.name} is missing tensor {name}")


class _Tokenizer:
    def __init__(self, path: Path):
        reader = GGUFReader(str(path), "r")
        fields = reader.fields
        if fields["general.architecture"].contents() != "kimodo-llm2vec-tokenizer":
            raise ValueError("not a Kimodo LLM2Vec tokenizer GGUF")
        self.tokens = fields["kimodo.tokenizer.tokens"].contents()
        merges = fields["kimodo.tokenizer.merges"].contents()
        self.ids = {token: i for i, token in enumerate(self.tokens)}
        self.ranks = {tuple(merge.split(" ", 1)): i for i, merge in enumerate(merges)}
        direct = set(range(33, 127)) | set(range(161, 173)) | set(range(174, 256))
        extra = 256
        self.byte_tokens = []
        for i in range(256):
            # The vocabulary stores byte tokens as UTF-8 strings: printable
            # bytes are literal, while the remaining bytes map to U+0100+.
            if i in direct:
                self.byte_tokens.append(chr(i))
            else:
                self.byte_tokens.append(chr(extra))
                extra += 1

    @staticmethod
    def _is_alpha(byte: int) -> bool:
        # The C++ reference uses std::isalpha on the byte OR high-bit set.
        return (0x41 <= byte <= 0x5A) or (0x61 <= byte <= 0x7A) or byte >= 0x80

    @staticmethod
    def _is_digit(byte: int) -> bool:
        return 0x30 <= byte <= 0x39

    @staticmethod
    def _is_space(byte: int) -> bool:
        return byte in (0x20, 0x09, 0x0D, 0x0A)

    @staticmethod
    def _contraction(text: bytes, pos: int) -> int:
        if pos >= len(text) or text[pos] != 0x27:
            return 0
        lower = bytes(text[pos:pos + 4]).lower()
        for opt in (b"'re", b"'ve", b"'ll", b"'s", b"'t", b"'m", b"'d"):
            if lower[:len(opt)] == opt:
                return len(opt)
        return 0

    @staticmethod
    def _pieces(text: bytes) -> Iterable[bytes]:
        # Byte-exact port of the C++ llm_tokenizer::encode pre-tokenizer.
        pos = 0
        n = len(text)
        while pos < n:
            clen = _Tokenizer._contraction(text, pos)
            if clen:
                yield text[pos:pos + clen]
                pos += clen
                continue
            first = text[pos]
            prefixed_letter = (not _Tokenizer._is_digit(first) and first != 0x0D and first != 0x0A
                and not _Tokenizer._is_alpha(first) and pos + 1 < n and _Tokenizer._is_alpha(text[pos + 1]))
            if _Tokenizer._is_alpha(first) or prefixed_letter:
                size = 1 if prefixed_letter else 0
                while pos + size < n and _Tokenizer._is_alpha(text[pos + size]): size += 1
                yield text[pos:pos + size]
                pos += size
                continue
            if _Tokenizer._is_digit(first):
                size = 0
                while size < 3 and pos + size < n and _Tokenizer._is_digit(text[pos + size]): size += 1
                yield text[pos:pos + size]
                pos += size
                continue
            if (not _Tokenizer._is_space(first) or (pos + 1 < n and not _Tokenizer._is_space(text[pos + 1])
                    and not _Tokenizer._is_alpha(text[pos + 1]) and not _Tokenizer._is_digit(text[pos + 1]))):
                size = 1 if first == 0x20 else 0
                while (pos + size < n and not _Tokenizer._is_space(text[pos + size])
                        and not _Tokenizer._is_alpha(text[pos + size]) and not _Tokenizer._is_digit(text[pos + size])):
                    size += 1
                yield text[pos:pos + size]
                pos += size
                continue
            size = 0
            while pos + size < n and _Tokenizer._is_space(text[pos + size]): size += 1
            yield text[pos:pos + size]
            pos += size

    def encode(self, text: str) -> list[int]:
        result = [_BOS]
        for word in self._pieces(text.encode("utf-8")):
            symbols = [self.byte_tokens[b] for b in word]
            while len(symbols) > 1:
                candidates = [(self.ranks.get((symbols[i], symbols[i + 1]), 2**63 - 1), i) for i in range(len(symbols) - 1)]
                rank, at = min(candidates)
                if rank == 2**63 - 1: break
                symbols[at : at + 2] = [symbols[at] + symbols[at + 1]]
            try:
                result.extend(self.ids[symbol] for symbol in symbols)
            except KeyError as exc:
                raise ValueError("BPE symbol is absent from vocabulary") from exc
        return result


def _rms(x: mx.array, weight: mx.array, *, bf16: bool) -> mx.array:
    value = mx.astype(x, mx.float32)
    value = value * (1.0 / mx.sqrt(mx.mean(value * value, axis=-1, keepdims=True) + 1e-5))
    if bf16:
        value = mx.astype(value, mx.bfloat16)
        return mx.astype(mx.astype(value, mx.float32) * mx.astype(weight, mx.float32), mx.bfloat16)
    return value * mx.astype(weight, mx.float32)


def _silu(x: mx.array) -> mx.array:
    return x / (1.0 + mx.exp(-x))


def _linear(x: mx.array, tensors: dict[str, mx.array], stem: str) -> mx.array:
    # GGML dispatches BF16 weights with vec_dot_type=BF16 (converts the input to
    # BF16, then accumulates in double).  MLX cannot accumulate in double, so we
    # keep the input in F32 to preserve precision — this gives closer parity
    # than rounding to BF16 and accumulating in F32.
    xf = mx.astype(x, mx.float32)
    base = mx.matmul(xf, mx.astype(tensors[stem + "_base.weight"], mx.float32).T)
    lora = mx.matmul(mx.matmul(xf, tensors[stem + "_lora_a.weight"].T), tensors[stem + "_lora_b.weight"].T)
    return base + 2.0 * lora


def _rope(x: mx.array, positions: mx.array) -> mx.array:
    # NeoX / Transformers rotate_half: first half and second half are paired.
    inv = 1.0 / (500000.0 ** (mx.arange(0, _HEAD_DIM, 2, dtype=mx.float32) / _HEAD_DIM))
    angles = mx.expand_dims(mx.astype(positions, mx.float32), -1) * inv
    cos = mx.cos(angles)[:, None, :]
    sin = mx.sin(angles)[:, None, :]
    first, second = x[..., : _HEAD_DIM // 2], x[..., _HEAD_DIM // 2 :]
    return mx.concatenate([first * cos - second * sin, first * sin + second * cos], axis=-1)


def _layer(x: mx.array, t: dict[str, mx.array], positions: mx.array) -> mx.array:
    residual = mx.astype(x, mx.bfloat16)
    n = _rms(residual, t["attn_norm.weight"], bf16=True)
    q = _linear(n, t, "attn_q_proj").reshape((-1, _HEADS, _HEAD_DIM))
    k = _linear(n, t, "attn_k_proj").reshape((-1, _KV_HEADS, _HEAD_DIM))
    v = _linear(n, t, "attn_v_proj").reshape((-1, _KV_HEADS, _HEAD_DIM))
    q, k = _rope(q, positions), _rope(k, positions)
    k = mx.repeat(k, _HEADS // _KV_HEADS, axis=1)
    v = mx.repeat(v, _HEADS // _KV_HEADS, axis=1)
    scores = mx.einsum("thd,shd->hts", q, k) / math.sqrt(_HEAD_DIM)
    probs = mx.softmax(scores, axis=-1)
    attended = mx.einsum("hts,shd->thd", probs, v).reshape((-1, _HIDDEN))
    x = mx.astype(residual, mx.float32) + _linear(attended, t, "attn_o_proj")
    residual = x
    n = _rms(x, t["ffn_norm.weight"], bf16=False)
    gate = _silu(_linear(n, t, "ffn_gate_proj"))
    up = _linear(n, t, "ffn_up_proj")
    down = _linear(gate * up, t, "ffn_down_proj")
    return residual + down


class TextEncoder:
    """Bidirectional Meta-Llama-3-8B LLM2Vec encoder backed by MLX."""
    def __init__(self, bundle_dir: str | Path):
        self.bundle_dir = Path(bundle_dir)
        required = ["tokenizer.gguf", "embedding.gguf", "final-norm.gguf"] + [f"layer-{i:02d}.gguf" for i in range(32)]
        missing = [name for name in required if not (self.bundle_dir / name).is_file()]
        if missing: raise FileNotFoundError(f"text bundle missing {missing[0]}")
        self._tokenizer = _Tokenizer(self.bundle_dir / "tokenizer.gguf")
        self._embedding = _tensor(self.bundle_dir / "embedding.gguf", "token_embedding.weight")
        self._final_norm = _tensor(self.bundle_dir / "final-norm.gguf", "final_norm.weight")
        self._layers: list[dict[str, mx.array] | None] = [None] * 32

    def tokenize(self, prompt: str) -> list[int]:
        return self._tokenizer.encode(prompt)

    def encode(self, prompt: str) -> np.ndarray:
        ids = self.tokenize(prompt)
        if len(ids) < 2 or len(ids) > 512: raise ValueError("prompt token count must be in 1..511 excluding BOS")
        x = self._embedding[mx.array(ids, dtype=mx.int32)]
        positions = mx.arange(len(ids), dtype=mx.int32)
        for i in range(32):
            tensors = self._layers[i]
            if tensors is None:
                path = self.bundle_dir / f"layer-{i:02d}.gguf"
                names = ["attn_norm.weight", "ffn_norm.weight"]
                for stem in ("attn_q_proj", "attn_k_proj", "attn_v_proj", "attn_o_proj", "ffn_gate_proj", "ffn_up_proj", "ffn_down_proj"):
                    names += [stem + "_base.weight", stem + "_lora_a.weight", stem + "_lora_b.weight"]
                tensors = {name: _tensor(path, name) for name in names}
                self._layers[i] = tensors
            x = _layer(x, tensors, positions)
            mx.eval(x)
        x = _rms(x, self._final_norm, bf16=False)
        x = mx.mean(x[1:], axis=0)
        mx.eval(x)
        return np.asarray(x, dtype=np.float32)
