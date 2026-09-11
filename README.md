# kimodo-mlx

Apple Silicon에서 NVIDIA Kimodo를 돌리기 위한 MLX/Metal 포트입니다. 뉴럴 엔진은 쓰지 않습니다.

같은 문장(`walk forward`), 30프레임, 10스텝, seed 42를 Mac Studio M4 Max 64GB에서 실측했습니다. kimodo.cpp Metal은 레이어를 스트리밍하고, kimodo-mlx는 15GB LLM2Vec을 통합 메모리에 올립니다.

[모션 비교 페이지](docs/compare.html) · [실측 JSON](evidence/compare/timings.json)

## 실측

![End-to-end generation time](docs/charts/e2e.svg)

![MLX encode vs sample](docs/charts/mlx-breakdown.svg)

| 런타임 | 가중치 상주 | 인코드 | 샘플 | E2E |
| --- | --- | --- | --- | --- |
| kimodo-mlx Metal, cold | 아니오 | 35.05s | 325ms | 35.38s |
| **kimodo-mlx Metal, resident** | **예** | **1.74s** | **277ms** | **2.01s** |
| kimodo.cpp Metal | 아니오 (레이어 스트리밍) | — | — | 25.04s |
| kimodo.cpp CPU | 아니오 | — | — | 35.50s |
| ONNX Runtime / CoreML | — | — | — | 미실행 (그래프 없음) |
| NVIDIA Kimodo PyTorch MPS | — | — | — | 미실행 (게이트된 Llama 3 인코더) |

보조 클립(16프레임 / 8스텝)에서는 상주 MLX 평균 E2E **0.93s**, cpp Metal **37.0s**. LLM2Vec vs `kmd-encode` cosine > 0.99. 관절 위치 MAE 0.107m (시드는 같고 초기 노이즈 RNG는 다름).

**효과:** 한 번만 켜는 콜드는 cpp Metal이 이깁니다. 같은 프로세스에서 반복 생성하면 MLX가 약 **12배** 빠릅니다. 이득의 핵심은 GPU 커널이 아니라 8B 인코더를 통합 메모리에 올려 두는 정책입니다. 16GB대 맥에서는 상주가 부담이고, cpp 스트리밍이 맞습니다.

## 이 포트가 하는 일

- 양방향 LLM2Vec (Llama-3-8B, causal llama.cpp 임베딩과 호환되지 않음)
- 2단 트랜스포머 모션 디노이저 + DDIM
- NVIDIA SOMA safetensors 또는 LocalAI F32 GGUF
- Metal GPU. ANE/Core ML/ONNX는 이 릴리스에 없습니다.

## Quick start

가중치는 저장소에 없습니다. NVIDIA Open Model License와 Meta Llama 3 조건을 읽고 받으십시오. Built with Meta Llama 3.

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
  --frames 30 --steps 10 --seed 42
```

`diagnose`는 `mlx-metal`과 `neural_engine_used: false`를 보고해야 합니다.

## 벤치와 비교

```sh
PYTHONPATH=. .venv/bin/python scripts/capture_comparison.py
PYTHONPATH=. .venv/bin/python scripts/build_compare_html.py
open docs/compare.html
```

kimodo.cpp Metal 대조군:

```sh
sh scripts/build_kimodo_metal.sh
```

## Tests

```sh
PYTHONPATH=. .venv/bin/python -m pytest -q
```

## License

코드는 MIT (`LICENSE`). 모델 가중치는 NVIDIA / Meta Llama 3 / LocalAI 조건을 따릅니다. SMPL-X RP 파생물은 재배포하지 마십시오. 출처는 `ATTRIBUTION.md`.

하드웨어 스냅샷: macOS 26.5.1, Apple M4 Max, 64GB, Python 3.12.9, MLX 0.32, kimodo.cpp `568b0253`.
