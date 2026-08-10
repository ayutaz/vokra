# Vokra

**English** | [日本語](README.ja.md)

**Vokra** is a Rust inference runtime specialized for speech AI — TTS, ASR,
speech-to-speech, voice conversion, speaker identification, and VAD — built
as an alternative to ONNX / ONNX Runtime for speech workloads.

General-purpose runtimes chronically underserve speech models: STFT/iSTFT and
streaming state, vocoder numerics, neural codec (RVQ/FSQ) decoding,
flow-matching samplers, beam search / CTC / RNN-T decoding, VAD, and speaker
embeddings all end up as fragile graph exports or host-side glue. Vokra makes
them first-class native operators instead.

- **Pronunciation**: "vo-krah" (English) / 「ヴォクラ」 (Japanese)
- **License**: [Apache-2.0](LICENSE) — no GPL/LGPL anywhere in the dependency
  closure.
- **Repository**: <https://github.com/ayutaz/vokra>
- **Model hub**: <https://huggingface.co/vokra>

> APIs, file formats, and the model roster are pre-1.0 and may change in
> breaking ways. A stable C ABI will accompany the first stable release.

## Table of contents

- [Key features](#key-features)
- [Supported models](#supported-models)
- [Supported platforms and backends](#supported-platforms-and-backends)
- [Getting started](#getting-started)
- [Using the C ABI](#using-the-c-abi)
- [Architecture overview](#architecture-overview)
- [Bindings and integrations](#bindings-and-integrations)
- [Model publications](#model-publications)
- [piper-plus integration](#piper-plus-integration)
- [Documentation](#documentation)
- [Related projects](#related-projects)
- [Contributing](#contributing)
- [Legal and compliance](#legal-and-compliance)
- [License](#license)

## Key features

- **Native re-implementation of speech models** (whisper.cpp-style): model
  code lives in Rust and consumes upstream `safetensors` / `GGUF`
  checkpoints directly. The runtime never loads ONNX graphs, so it carries
  no `onnx` / `protobuf` / `abseil` transitive dependencies.
- **Zero external dependency invariant**: the root `Cargo.lock` contains
  only first-party `vokra-*` crates. GPU/NPU backends use hand-written FFI
  (no `metal-rs`, no `cudarc`, no `ash`, no `wgpu`) and are strictly opt-in
  via Cargo features, so the default build stays first-party-only.
  Enforced in CI by [`scripts/check-zero-deps.sh`](scripts/check-zero-deps.sh).
- **Speech-first operator set**: STFT / iSTFT with explicit window / hop /
  normalization / RFFT attributes, mel filterbank, polyphase resampling,
  vocoder chains (HiFi-GAN, BigVGAN, HiFTNet, Vocos-style iSTFT heads),
  flow-matching samplers with configurable CFG modes and schedules, neural
  codec decoders (DAC, Mimi, WavTokenizer, X-Codec 2), beam search / CTC /
  RNN-T decoding, streaming KV cache (paged, 3D `[time, stream, codebook]`),
  VAD, speech enhancement (AEC / AGC / HPF / loudness-norm / DeepFilterNet3),
  speaker embedding, F0 extraction, and objective quality metrics.
- **CPU as a first-class backend** with runtime ISA dispatch — x86-64 SSE2
  baseline through AVX2, AVX-512F/DQ/BW/VL, AVX-512 VNNI/BF16, AVX-VNNI
  256-bit, and AMX; ARM64 NEON through fp16 arithmetic, dotprod (SDOT/UDOT),
  i8mm, and bf16; RVV 1.0 baseline. Includes K-quants (Q4_K / Q5_K / Q6_K)
  and a per-layer, config-driven quantization policy with a minimum-dtype
  registry that refuses INT8 for numerically fragile vocoders.
- **No silent fallback**: an operator a backend does not implement is an
  explicit, loud error. GPU backends never silently drop to CPU. Vokra
  prefers a hard error over a wrong answer.
- **GGUF with speech metadata**: model files are GGUF augmented with
  `vokra.*` chunks (`vokra.frontend.*`, `vokra.whisper.*`, `vokra.piper.*`,
  `vokra.provenance.*`, `vokra.quant.*`, `vokra.schema.*`) so the front-end
  spec, quantization policy, and licensing provenance travel with the
  weights and are bit-exactly reproducible.
- **Cross-platform distribution**: single library, single C ABI header,
  static or dynamic linking; iOS XCFramework and Swift Package; Unity UPM
  package; Godot GDExtension; Python `ctypes` wheels; HTTP compatibility
  server exposing OpenAI Whisper, vLLM, piper-plus, and Wyoming Protocol
  endpoints.
- **Safe-by-default Rust**: `unsafe_code = "deny"` is set workspace-wide;
  `unsafe` is allowed only in backend and FFI crates and requires
  `// SAFETY:` justifications enforced by
  `clippy::undocumented_unsafe_blocks = "deny"`.
- **License hygiene by construction**: a compliance gate refuses
  non-commercially licensed weights (F5-TTS, Fish-Speech, EnCodec,
  X-Codec 2) on the default path unless the caller opts in via an explicit
  research flag / `--allow-noncommercial`.

## Supported models

Vokra ships native implementations of the models below. Weight loading and
tokenization are built into the runtime; converters producing the GGUF
files live in the `vokra-convert` crate.

**ASR**
- Whisper — `base`, `small`, `medium`, `large-v3`, `turbo`
- Voxtral — `Mini-3B`, `Small-24B` (streaming loader for large variants)
- Canary-Qwen-2.5B (FastConformer + Qwen decoder)
- omniASR-CTC — 300M and 7B variants
- Charsiu (wav2vec2 CTC)
- Kyutai STT
- Zipformer / E-Branchformer / Hybrid CTC-Attention decoders

**TTS**
- piper-plus (native MB-iSTFT-VITS2, 8-language G2P: JA / EN / ZH / ES / FR
  / PT / SV / KO)
- Kokoro-82M
- CosyVoice2 (FSQ tokens + Qwen2.5-0.5B AR + chunk-aware CFM → mel → HiFTNet)
- Style-Bert-VITS2 v2 — multilingual (JA / EN / ZH) with per-language
  conditioning encoders: DeBERTa v2 (JA), DeBERTa v3 (EN), and
  Chinese-RoBERTa-wwm-ext (ZH)
- VoxCPM-0.5B and VoxCPM2-2B
- Qwen3-TTS 1.7B
- Fun-CosyVoice3-0.5B

**Speech-to-speech (full-duplex)**
- Sesame CSM-1B
- Moshi (Helium + Mimi codec)

**VAD**
- Silero VAD v5 (default) and v6.2.1
- FSMN-VAD

**Speaker embedding / verification**
- CAM++ (192-d, zero-shot voice cloning input)
- TitaNet-L
- ECAPA-TDNN

**F0 / pitch**
- RMVPE (front-end + decoder; internal U-Net + GRU forward is a
  loud-partial, awaiting real-checkpoint verification)
- FCPE (real Conformer forward)
- CREPE (real 6-block CNN, 5 model sizes)

**Neural codecs**
- DAC (24 kHz), Mimi, WavTokenizer, X-Codec 2 (research-only, CC-BY-NC-4.0)

**Speech enhancement**
- DeepFilterNet3, AEC, AGC, HPF, loudness normalization

**Objective quality**
- UTMOS22-strong

See [`docs/license-audit.md`](docs/license-audit.md) for the licensing
audit and [`docs/legal-compliance.md`](docs/legal-compliance.md) for the
distribution rules that follow from it.

## Supported platforms and backends

Every platform below is in scope for the single library and single C ABI.
Backend acceleration is enabled with Cargo features so the default build
stays zero-dependency.

| Backend | Cargo feature | Notes |
|---|---|---|
| CPU (default) | — | x86-64 SSE2 → AVX2 → AVX-512F/DQ/BW/VL → AVX-512 VNNI/BF16 → AVX-VNNI 256 → AMX; ARM64 NEON → fp16 → dotprod → i8mm → bf16; RVV 1.0 |
| Metal (macOS / iOS) | `metal` | Hand-written raw `objc` + Metal FFI, MSL compute kernels |
| CUDA (Windows / Linux) | `cuda` | Driver API + NVRTC loaded via `dlopen` / `LoadLibrary` — no CUDA library is bundled, per NVIDIA EULA |
| Vulkan (Android / Linux / Windows) | `vulkan` | dlopen + pre-compiled SPIR-V, subgroup and cooperative matrix path with a fallback |
| WebGPU / WASM (browsers) | `webgpu`, target `wasm32-unknown-unknown` | wasm extern-import shim, no `wgpu` / `wasm-bindgen` dependency |
| CoreML (Apple ANE) | `coreml` | Opt-in delegate scaffold |
| QNN (Qualcomm Hexagon) | `qnn` | Opt-in delegate scaffold |

**Operating systems**: Windows, macOS, Linux, Android, iOS, and modern web
browsers (via WebGPU / WASM SIMD128 + threads).

**Explicitly not supported**: NNAPI (deprecated by Google in Android 15,
October 2024); Piper `OHF-Voice/piper1-gpl` (GPL-3.0 with an eSpeak-NG
GPL-3.0 transitive dependency — the only Piper-family integration Vokra
supports is the owner's MIT-licensed [`piper-plus`](https://github.com/ayutaz/piper-plus)
fork).

## Getting started

### Prerequisites

- Rust toolchain (edition 2024). MSRV is `1.85`; some AVX-512 intrinsics in
  `vokra-backend-cpu` require `1.89`. Install from <https://rustup.rs>.
- A C compiler (only if you plan to link against the C ABI).

### Build the CLI

```sh
git clone https://github.com/ayutaz/vokra.git
cd vokra
cargo build --release -p vokra-cli
# The binary is at target/release/vokra-cli
```

### Download a model

Every published model card on <https://huggingface.co/vokra> lists the exact
`.gguf` file to fetch. For example, to grab a Whisper base checkpoint:

```sh
# Any of: curl / wget / huggingface-cli — the file is a plain GGUF blob.
huggingface-cli download vokra/whisper-base whisper-base.gguf --local-dir .
```

### Transcribe audio

```sh
target/release/vokra-cli run whisper-base.gguf --input audio.wav
```

### Run inference on a GPU backend

Enable the relevant Cargo feature at build time:

```sh
# macOS / iOS
cargo build --release -p vokra-cli --features metal
target/release/vokra-cli run whisper-base.gguf --input audio.wav --backend metal

# Linux / Windows with a discrete NVIDIA GPU (requires a developer-installed CUDA)
cargo build --release -p vokra-cli --features cuda
target/release/vokra-cli run whisper-base.gguf --input audio.wav --backend cuda
```

### Convert an upstream checkpoint

```sh
target/release/vokra-cli convert --model whisper \
  --input path/to/upstream/checkpoint --output whisper-base.gguf
```

More detail — including a per-backend guide, a per-tutorial walkthrough
(CLI, Android, iOS, Godot, Unity, Python, server, web), and a migration
guide from onnxruntime / whisper.cpp — is in [`docs/`](docs).

## Using the C ABI

Vokra exposes a single C header, [`include/vokra.h`](include/vokra.h)
(generated with cbindgen; regenerate with
[`scripts/gen-c-abi.sh`](scripts/gen-c-abi.sh)). Building the `vokra-capi`
crate produces the shared and static libraries:

```sh
cargo build -p vokra-capi --release
# -> target/release/libvokra.dylib | libvokra.so | vokra.dll  (+ libvokra.a)
```

A session is created from a GGUF model; the architecture is detected from
the file's `vokra.model.arch` metadata and the matching task is wired
automatically (Whisper → ASR, Silero VAD → VAD stream, piper-plus → TTS).
All functions return a `vokra_status_t` (`VOKRA_OK` is `0`); on error a
per-thread message is available from `vokra_last_error()`. Vokra-allocated
outputs are released with their matching `vokra_*_free` / `vokra_*_destroy`
function.

```c
#include "vokra.h"

vokra_session_t *session = NULL;
if (vokra_session_create_from_file("whisper-base.gguf", &session) != VOKRA_OK) {
    fprintf(stderr, "load failed: %s\n", vokra_last_error());
    return 1;
}

char *text = NULL;
if (vokra_asr_transcribe(session, pcm, num_samples, 16000, &text) == VOKRA_OK) {
    printf("%s\n", text);
    vokra_string_free(text);
}
vokra_session_destroy(session);
```

Compile against the header and link the shared library:

```sh
cc app.c -I include -L target/release -lvokra -Wl,-rpath,target/release -o app
```

Runnable end-to-end examples (ASR / TTS / VAD) live in
[`tests/capi/`](tests/capi); `scripts/run-capi-smoke.sh` builds and runs
them.

## Architecture overview

**Native re-implementation.** Every supported model is a Rust module —
tokenizer, tensor layout, forward pass, decoder loop, and streaming state
included — that consumes an upstream `safetensors` or Vokra-flavoured
`GGUF` file. This side-steps the chronic fragility of `torch.onnx.export`
for speech models, keeps the runtime free of `onnx` / `protobuf` /
`abseil`, and makes bug reports actionable (a single Rust file, not a graph
export).

**Zero external dependency.** The root workspace `Cargo.lock` resolves to
first-party `vokra-*` crates only. This invariant is checked by
`scripts/check-zero-deps.sh` on every CI run. Anything that would break it
— an 8-language G2P port, a Godot GDExtension, an HTTP server, an ONNX
converter — lives in an isolated sub-workspace under `integrations/` with
its own `Cargo.lock`.

**Hand-written FFI, EULA-compliant CUDA.** GPU and NPU backends do not
depend on binding crates. Metal uses raw `objc` runtime calls plus MSL
compute kernels; CUDA loads the NVIDIA Driver API and NVRTC through
`dlopen` / `LoadLibrary` at runtime against a developer-installed CUDA
(nothing is bundled, which keeps the distribution compatible with the
NVIDIA CUDA / cuDNN EULA — see [`NOTICE`](NOTICE)); Vulkan pre-compiles
SPIR-V and loads the loader through `dlopen`; WebGPU speaks to the browser
through a small wasm extern-import shim.

**GGUF plus `vokra.*` metadata.** Weight files are standard GGUF with a set
of Vokra-owned chunks (`vokra.frontend.*` for STFT / mel spec,
`vokra.<arch>.*` for per-model hyperparameters, `vokra.quant.*` for
quantization policy, `vokra.provenance.*` for licensing provenance,
`vokra.schema.version` / `vokra.schema.producer` for producer identity).
Because the front-end spec travels with the weights, the runtime rejects a
checkpoint whose spec does not bit-exactly match what the model was trained
against — no silent librosa-vs-torchaudio Mel-filter drift.

**Compute seams and a graph executor.** Each model reaches the backends
through a small `Compute` seam (per-backend `Cpu` / `Metal` / `Cuda` /
`Vulkan` arms for the GEMM hot path). Longer-lived pipelines — the pre-norm
encoder stack, autoregressive decoder steps with a device-resident KV cache,
codec chains — are represented as a data-carrying graph so intermediates
stay device-resident and host↔device readback stays at a small constant per
step.

**Loud errors, never wrong answers.** If a backend does not implement an
operator, the call fails with an explicit `VokraError::UnsupportedOp`.
Vokra never silently reroutes to a different backend or dtype.

## Bindings and integrations

- **iOS** — XCFramework + Swift Package
  ([`Package.swift`](Package.swift), built by
  [`scripts/build-ios.sh`](scripts/build-ios.sh)): arm64 device and
  Simulator slices, static-linked, `DllImport("__Internal")`-compatible.
- **Unity** — UPM package `com.vokra.unity` under
  [`bindings/unity/`](bindings/unity), built by
  [`scripts/build-unity-plugin.sh`](scripts/build-unity-plugin.sh):
  IL2CPP-safe callback marshalling, Android `persistentDataPath` helper,
  and a non-NVIDIA-bundle scanner (`check-unity-package-no-nvidia.sh`) so
  distributions cannot accidentally ship CUDA libraries.
- **Godot** — GDExtension under
  [`integrations/vokra-godot/`](integrations/vokra-godot), built by
  [`scripts/build-godot-gdextension.sh`](scripts/build-godot-gdextension.sh):
  five-target cross-build matrix (macOS Intel + Apple Silicon, Linux x64,
  Windows MSVC, Android arm64) with an AssetLib-shaped release layout.
- **Python** — pure `ctypes` (no `pyo3`) under
  [`bindings/python/`](bindings/python), published as PyPI wheels via
  `cibuildwheel`.
- **HTTP server** — [`integrations/vokra-server`](integrations/vokra-server):
  an isolated workspace exposing four compatibility APIs so existing
  clients drop in unchanged: **OpenAI Whisper**
  (`/v1/audio/transcriptions`), **vLLM** (`/v1/completions`,
  `/v1/chat/completions`), **piper-plus HTTP** (`/api/tts`), and
  **Wyoming Protocol** for Home Assistant Voice backends.

## Model publications

Ready-to-run GGUF conversions are published under the
[`vokra`](https://huggingface.co/vokra) organization on Hugging Face. Every
published artifact carries:

- A model card generated from its own metadata.
- A `LICENSE` containing the upstream license text (fetched at publish
  time by [`scripts/publish/fetch_license.sh`](scripts/publish/fetch_license.sh)).
- A `NOTICE` when the upstream requires attribution (for example, Mimi is
  CC-BY-4.0 and requires crediting Kyutai).
- A `SOURCE.md` with the upstream URL and the re-conversion recipe.

Every publish goes through
[`scripts/publish/publish-one.sh`](scripts/publish/publish-one.sh), which
is a five-tier gate designed to fail closed:

1. **Catalog reality** — refuses artifacts that are not on the tracked
   catalog (no accidental publishes of unreviewed models).
2. **Redistributability** — refuses corpora with contractual non-
   redistribution clauses (VOICEVOX, CSJ, JSUT, JVS).
3. **Provenance stamp presence** — requires `vokra.schema.version` and
   `vokra.schema.producer` chunks so consumers can identify the producer.
4. **Owner sign-off** — requires the corresponding row in
   [`docs/license-audit.md`](docs/license-audit.md) §3.1 to be signed off,
   with a source-of-truth link.
5. **Non-commercial opt-in** — requires an explicit `--allow-noncommercial`
   flag for the research-only tier (for example, X-Codec 2 under
   CC-BY-NC-4.0).

Combined with a low-memory `restamp_provenance` rewrite path, this makes
publishing multi-gigabyte checkpoints from modest hardware routine.

## piper-plus integration

[piper-plus](https://github.com/ayutaz/piper-plus) is an MIT-licensed
Piper fork by the project owner (8-language G2P without eSpeak-NG,
MB-iSTFT-VITS2 decoder, CUDA / CoreML / DirectML support, Unity binding).
Vokra integrates it as the standard TTS layer and as its first natively
implemented TTS model:

- The MB-iSTFT-VITS2 inference stack — text encoder, duration predictor,
  flow, MB-iSTFT decoder — is re-implemented natively in Rust. There is no
  wrap of the upstream ONNX-based implementation and there is no
  `onnxruntime` on Vokra's end-to-end inference path.
- Voice models are converted offline to GGUF; the runtime loads only GGUF.
- The 8-language G2P (JA / EN / ZH / ES / FR / PT / SV / KO) is reused
  from piper-plus for the time being; a Rust port is a follow-up item.

## Documentation

All user-facing documentation lives under [`docs/`](docs). Every top-level
document has both an English (`.md`) and a Japanese (`.ja.md`) version.

| Document | What it covers |
|---|---|
| [`docs/getting-started.md`](docs/getting-started.md) | Five-minute quickstart |
| [`docs/architecture.md`](docs/architecture.md) | Internal architecture, crate layout, graph executor |
| [`docs/api-reference.md`](docs/api-reference.md) | C ABI + CLI reference |
| [`docs/backend-guide.md`](docs/backend-guide.md) | CPU / Metal / CUDA / Vulkan / WebGPU / CoreML / QNN guide |
| [`docs/tutorials/`](docs/tutorials) | Per-platform tutorials: CLI, Android, iOS, Godot, Unity, Python, server, web |
| [`docs/migration-guide.md`](docs/migration-guide.md) | Migrating from onnxruntime / whisper.cpp / piper |
| [`docs/license-audit.md`](docs/license-audit.md) | Model and dependency license audit |
| [`docs/legal-compliance.md`](docs/legal-compliance.md) | EU AI Act Article 50, SB 942, ELVIS Act, C2PA |
| [`docs/good-first-tasks.md`](docs/good-first-tasks.md) | Contributor entry points |
| [`docs/abi-changelog.md`](docs/abi-changelog.md) | C ABI change log |
| [`NOTICE`](NOTICE) | Attribution requirements and bundling policies |

## Related projects

- **[piper-plus](https://github.com/ayutaz/piper-plus)** — the MIT Piper
  fork by the project owner that Vokra integrates as its standard TTS
  layer (see above).

## Contributing

Contributions are welcome. Please open an issue before opening a large
pull request so scope and approach can be aligned early.

- **Where to start**:
  [`docs/good-first-tasks.md`](docs/good-first-tasks.md) — self-contained
  tasks with file:line anchors or reproduction commands, acceptance
  criteria you can check yourself, and a rough size.
- **Questions and discussion**: open an
  [issue on GitHub](https://github.com/ayutaz/vokra/issues).
- **Pull requests**: read [`CONTRIBUTING.md`](CONTRIBUTING.md). All
  changes go through pull requests with CI quality gates covering build,
  tests, formatting, clippy `-D warnings`, the zero-dependency invariant,
  the C ABI change log, and license auditing.

## Legal and compliance

- **EU AI Act Article 50** and **California SB 942**: TTS and voice-
  conversion outputs are considered synthetic audio and require
  disclosure. Vokra provides AudioSeal watermarking and C2PA manifest
  support (via `c2pa-rs`) as building blocks; disclosure obligations of
  the deployer are documented in
  [`docs/legal-compliance.md`](docs/legal-compliance.md).
- **Voice-cloning separation**: RVC v2, GPT-SoVITS, and other voice-
  conversion "trigger" models are deliberately **not** in this
  repository. They are split into a separate
  `vokra-voiceclone-experimental` project because of Tennessee's ELVIS Act
  (2024-07-01) and the federal NO FAKES Act. Speaker embedding for
  zero-shot TTS (feature extraction only, no conversion) stays in core.
- **NVIDIA CUDA / cuDNN EULA**: Vokra does not bundle any NVIDIA library.
  The CUDA backend `dlopen`s the developer-installed system CUDA at
  runtime. Recorded in [`NOTICE`](NOTICE) and
  [`docs/license-audit.md`](docs/license-audit.md).
- **Non-commercial weights**: F5-TTS (CC-BY-NC-4.0), Fish-Speech
  (CC-BY-NC-SA-4.0), EnCodec (CC-BY-NC-4.0), and X-Codec 2 (CC-BY-NC-4.0)
  are not included in the default model zoo. The engine can run them
  behind an explicit research flag / `--allow-noncommercial`.
- **Piper (`OHF-Voice/piper1-gpl`) is not supported** (GPL-3.0 with an
  eSpeak-NG GPL-3.0 transitive dependency). The only Piper-family
  integration Vokra supports is [piper-plus](https://github.com/ayutaz/piper-plus).

## License

Vokra is licensed under the [Apache License, Version 2.0](LICENSE).

Additional licensing and distribution notices — including per-model
attribution obligations (for example, Mimi under CC-BY-4.0), the BigVGAN
attribution, and the NVIDIA runtime non-bundling policy — are recorded in
[`NOTICE`](NOTICE).
