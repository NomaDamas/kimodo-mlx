# Kimodo on Apple Silicon: porting decision

## Executive decision

The first implementation target is an MLX/Metal runtime for Apple Silicon
that keeps the Kimodo motion denoiser numerically faithful and treats the
LLM2Vec text encoder as a separate compatibility boundary. The runtime will
reuse the existing Kimodo GGUF motion weights and the Llama-3-8B-derived
LLM2Vec weights only after tensor-level parity fixtures pass.

MLX is the selected primary backend because it is designed for Apple Silicon
unified memory and exposes Metal-backed arrays. PyTorch MPS remains the
reference backend for parity capture, not the optimized product path. GGML
Metal is a useful fallback and comparison baseline, but `kimodo.cpp` currently
documents CPU/Vulkan rather than a tested Metal implementation. Core ML/ANE is
not selected for the first port: a Core ML conversion may use
`CPU_AND_NEURAL_ENGINE`, but Kimodo's dynamic diffusion loop, bidirectional
LLM2Vec attention, custom tensor shapes, and operator coverage make an
unverified ANE speed claim invalid.

The first shippable milestone is therefore a truthful Apple GPU port and a
benchmark harness. ANE offload is a later experiment only if a converted
subgraph can be proven equivalent and Instruments/Core ML diagnostics show
that the Neural Engine actually ran.

## Existing implementations

### NVIDIA reference

NVIDIA's Kimodo project is a text-to-motion model with separate text
conditioning and motion generation components. The reference graph uses
LLM2Vec based on Meta-Llama-3-8B with modified bidirectional attention,
produces a 4096-wide text embedding, and runs a diffusion/DDIM denoiser.
Motion checkpoints differ by skeleton and representation, including SOMA and
Unitree G1 variants.

The investigated reference revision was
`1aece8c124d73d255ceff5086d983b844c9f4e94`. Its default denoiser
configuration is 16 layers per stage, width 1024, feed-forward width 2048,
8 attention heads, GELU, and zero dropout. The public motion feature layout
is `9 + 12*J`: 273 features for 22-joint SMPL-X, 369 for 30-joint SOMA, and
417 for 34-joint G1. One prompt is limited to 300 frames / 10 seconds.

### localai-org/kimodo.cpp

`kimodo.cpp` is an Apache-2.0 C++/GGML implementation. Its porting notes
identify the following parity surfaces: non-causal attention, PEFT adapter
merge, exact pooling, CFG batching/order, diffusion schedule, DDIM arithmetic,
global/local root conversion, normalization statistics, and motion
representation inversion.

Its README documents:

- motion and text components in GGUF/native GGML-compatible bundles;
- safetensors conversion and checked GGUF loading;
- CPU and Vulkan execution;
- a reusable Llama-3-Kimodo-GGML text bundle;
- no quantised-model support in the current status;
- separate model repositories under the `LocalAI-io` Hugging Face
  organization.

The existing motion-only path is reusable as a reference and asset source.
The text encoder must not be replaced by an ordinary causal llama.cpp
embedding call.

### ONNX

ONNX Runtime's CoreML execution provider is a possible future backend for
static subgraphs. It is not the selected first path because the complete
Kimodo graph includes dynamic diffusion control and a non-causal adapted
LLM2Vec encoder; an ONNX export would need operator and numerical parity
fixtures before it could be called equivalent.

## Apple Silicon backend decision

| Backend | Decision | Reason |
| --- | --- | --- |
| MLX + Metal | Primary | Apple-Silicon-first arrays, unified-memory behavior, direct GPU execution |
| PyTorch MPS | Reference | Useful for parity against the Python implementation, but not the optimized product runtime |
| GGML Metal | Fallback/comparison | Reuses C++ GGUF ecosystem; current Kimodo port documents CPU/Vulkan, so Metal needs its own verification |
| ONNX Runtime CoreML | Deferred | Potential static-subgraph acceleration, but export and dynamic-graph parity are unproven |
| Core ML / ANE | Deferred experiment | `CPU_AND_NEURAL_ENGINE` is an execution option, not proof that this graph will run on ANE |

Apple Silicon's unified memory means model weights and intermediate arrays can
be shared between CPU and GPU without a discrete VRAM copy. The benchmark
must still report peak memory and warm/cold timings, because unified memory
does not remove allocation or synchronization costs.

## Parity hazards

The benchmark protocol must fix all of the following:

1. model revision and hashes;
2. tokenizer and attention mask;
3. bidirectional attention and adapter merge;
4. pooled 4096-vector;
5. initial noise and random generator;
6. diffusion steps, schedule, guidance scale, and DDIM arithmetic;
7. motion representation, normalization, skeleton, and post-processing;
8. precision and warm/cold execution protocol.

Quality comparison must use tensor metrics and motion-specific invariants, not
only wall-clock time. A faster result with materially different motion is a
failed port.

## Licensing and reuse boundary

The `kimodo.cpp` source is Apache-2.0. Its README states that model weights
retain their own licenses. SOMA and G1 checkpoints are associated with the
NVIDIA Open Model License; SMPL-X RP is identified as internal R&D-only and
not redistributable. The repository will not vendor model weights. It will
record exact model identifiers, revisions, hashes, and license acknowledgments
in benchmark artifacts and require users to obtain gated assets themselves.

The LocalAI port was investigated at revision
`568b0253f346fbe369587c7dae73d58594a14c90a`, with GGML submodule revision
`8c63e70982c95ceb862e3a1073a2c1beef75d60a`. The public SOMA/G1 motion GGUFs
are native F32 artifacts; the native text bundle is separate and uses
bidirectional LLM2Vec semantics, so neither a stock causal embedding path nor
an assumed quantized motion model is a valid parity baseline.

## Sources

- NVIDIA Kimodo repository: https://github.com/nv-tlabs/kimodo
- NVIDIA model concepts: https://research.nvidia.com/labs/sil/projects/kimodo/docs/key_concepts/model.html
- NVIDIA model API: https://research.nvidia.com/labs/sil/projects/kimodo/docs/api_reference/model.html
- NVIDIA motion representation API: https://research.nvidia.com/labs/sil/projects/kimodo/docs/api_reference/motion_rep.html
- NVIDIA Kimodo model card: https://huggingface.co/nvidia/Kimodo-G1-SEED-v1
- LocalAI Kimodo README: https://github.com/localai-org/kimodo.cpp/blob/main/README.md
- LocalAI Kimodo porting notes: https://github.com/localai-org/kimodo.cpp/blob/main/PORTING.md
- MLX documentation: https://ml-explore.github.io/mlx/
- MLX unified memory: https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html
- Apple Metal capabilities: https://developer.apple.com/metal/capabilities/
- PyTorch MPS backend: https://docs.pytorch.org/docs/stable/notes/mps.html
- Apple PyTorch Metal guidance: https://developer.apple.com/metal/pytorch/
- Core ML compute units: https://developer.apple.com/documentation/coreml/mlcomputeunits/cpuandneuralengine
- ONNX Runtime CoreML provider: https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html
