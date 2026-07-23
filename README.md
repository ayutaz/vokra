# Vokra

**English** | [日本語](README.ja.md)

**Vokra** is an inference runtime specialized for speech AI — TTS, ASR,
speech-to-speech, voice conversion, speaker identification, and VAD — built
in Rust as an alternative to ONNX / ONNX Runtime for speech workloads.

- **Pronunciation**: "vo-krah" (English) / 「ヴォクラ」 (Japanese)
- **License**: [Apache-2.0](LICENSE)
- **Status**: **pre-release, under active development** — v0.5 (M2) and v0.9
  (M3) are merged to `main`; v1.0-rc (M4) feature work is complete on the
  development branch (owner verification pending). The only tagged release is
  `v0.1.0`; APIs, file formats, and model coverage are unstable and incomplete.

## What Vokra is

General-purpose runtimes chronically underserve speech models: STFT/iSTFT
and streaming state, vocoder numerics, neural codec (RVQ/FSQ) decoding,
flow-matching samplers, beam search / CTC / RNN-T decoding, VAD, and
speaker embeddings all end up as fragile graph exports or host-side glue.
Vokra makes them first-class native operators instead.

Key design points:

- **Rust core, C ABI** (generated with cbindgen) for Unity / Godot / other
  engine and language bindings; Apache-2.0 with no GPL/LGPL dependencies.
- **Direct weight loading** from GGUF (with `vokra.*` audio metadata
  chunks) and safetensors. **The runtime never loads ONNX graphs** — ONNX
  models are handled by an offline conversion tool only, so the runtime
  carries no onnx/protobuf dependency.
- **Speech-first operator set**: STFT/iSTFT (explicit window/hop/norm/RFFT
  attributes), mel filterbank, resampling, vocoder chains, flow-matching
  samplers, codec decode, beam search / CTC / RNN-T, streaming KV cache,
  VAD, speech enhancement (AEC/denoise), speaker embedding, and F0
  extraction. A weight-license compliance gate keeps CC-BY-NC weights out of
  the default path unless a research flag is set (audio watermarking is
  designed but not yet enabled).
- **CPU as a first-class backend** (x86-64 SSE2 baseline through
  AVX2/AVX-512/AMX, ARM64 NEON through SVE/SME, with runtime dispatch),
  then staged GPU/NPU acceleration: Metal, CUDA, Vulkan, WebGPU, CoreML,
  QNN. **Metal and CUDA backends are already implemented and validated on
  real hardware** (see Status); GPU support uses hand-written zero-dependency
  FFI (no `metal-rs` / `cudarc` / binding crates) and never silently falls
  back to CPU — an op a backend does not cover is an explicit error.
- **All platforms are in scope**: Windows / macOS / Linux / Android / iOS /
  Web, plus x86-64 and ARM64 servers. The roadmap staggers *when* each
  backend gets official acceleration, not *whether* a platform is
  supported.

## Status and design documents

The v0.1 spike and v0.1 MVP are complete; **v0.5** (Metal / CUDA GPU backends)
and **v0.9** (CUDA-complete, Vulkan, CosyVoice2, Voxtral, RVV 1.0) are merged to
`main`, and **v1.0-rc** (M4: WebGPU/WASM, Sesame CSM-1B, Moshi, all-platform
support) feature work is complete on the development branch. Nothing is ready
for production use yet, but the following are implemented and validated:

- **CPU speech stack**: Silero VAD, Whisper (base…large-v3, with an embedded
  detokenizer for correct transcription), and piper-plus native TTS with the
  real 8-language G2P and a native CAM++ speaker encoder for zero-shot voice
  cloning. All are numerically parity-checked against reference runtimes
  (onnxruntime / PyTorch) to FP32 `atol = 0.01`. **Real-checkpoint
  validation** (Apple M1, vs onnxruntime 1.19.2 CPU, same downloaded
  weights): Whisper base/small/medium/turbo transcripts are
  **byte-identical to ONNX Runtime** (identical WER); piper output is
  near-bit-exact (mel-L1 ≈ 0.003); Mimi/DAC/WavTokenizer codec parity all
  pass; DeepFilterNet3 denoising matches upstream to an SI-SNR gap of
  2.0e-7 dB (see
  [`docs/bench-baselines/m1-real-weight-eval-2026-07-16/`](docs/bench-baselines/m1-real-weight-eval-2026-07-16/)).
- **CPU speed** (rig-scoped: Apple M1, 8 threads, vs onnxruntime 1.19.2
  CPU on the same machine and weights; methodology + raw logs in
  [`docs/bench-baselines/m5-14-final-2026-07-18/`](docs/bench-baselines/m5-14-final-2026-07-18/)):
  after the packed-GEMM/vectorization wave, **Whisper base runs ~2.5×
  faster than ONNX Runtime, whisper-turbo ~2.7× faster, and Silero VAD
  ~2.3× faster**; whisper-medium/small land within 1.17–1.24× of ORT and
  piper within ~2.2×. Every optimization is bit-identical by construction
  (no parity tolerance was changed).
- **GPU backends** (`vokra-backend-metal`, `vokra-backend-cuda`): a
  data-carrying graph evaluator plus a per-model dispatch seam. **Whisper
  runs end-to-end on both Metal (validated on Apple M1) and CUDA (validated
  on an RTX 4090)** with greedy output matching the CPU path exactly. GPU
  intermediates stay device-resident across the whole Whisper encoder (all
  pre-norm blocks fused into one submission) and across each autoregressive
  decoder step (fused causal attention with an on-device KV cache), cutting
  host↔device readback to a small constant. With that, **Whisper large-v3
  reaches RTF < 0.15 on an RTX 4090** (measured 0.081-0.115 on 30 s of audio
  depending on individual GPU / runtime conditions; several× the CPU path).
  Both backends are hand-written FFI with zero external crates and are
  exercised in CI.
- **Tooling**: `vokra-cli` (`run` / `convert` / `bench`, with
  `bench --backend cpu|metal|cuda` for GPU RTF), an offline `vokra-convert`,
  a `vokra-eval` metrics crate, and true zero-copy `mmap` GGUF loading.
- **Distribution**: an iOS **XCFramework + Swift Package** (arm64 device +
  Simulator slices, static-linked, `DllImport("__Internal")` compatible), a
  **Unity UPM package** (`com.vokra.unity`, IL2CPP-safe callbacks + Android
  `persistentDataPath` helper), and **Python bindings** (pure `ctypes`, no
  `pyo3`, published as PyPI wheels via `cibuildwheel`). See
  [`bindings/`](bindings) and [`Package.swift`](Package.swift).
- **Server**: [`integrations/vokra-server`](integrations/vokra-server) is an
  isolated workspace (own `Cargo.lock`) exposing four HTTP compatibility
  layers — **OpenAI Whisper** (`/v1/audio/transcriptions`, faster-whisper
  drop-in), **vLLM** (`/v1/completions`, `/v1/chat/completions`), **piper-plus
  HTTP** (`/api/tts`), and **Wyoming Protocol** (Home Assistant Voice
  backend). Kept out of the root workspace so the core's zero-dependency
  invariant stays intact.
- **Graph fusion**: a log-mel front-end fusion (STFT + magnitude + mel + log
  collapsed into a single kernel) with AVX2 / NEON specializations, wired
  through the `mel-frontend` `vokra-cli bench` task and gated by a 5%
  regression check in CI.
- **Quantization policy**: per-layer, config-driven quantization
  (`W4A16Q4K` / `W8A8Int8` / `FP16` / `FP32`) with a minimum-dtype registry
  that refuses INT8 for the ops that need FP16 (Vocos / BigVGAN), applied
  during `vokra-convert` and baked into a `vokra.quant.*` GGUF chunk.
- **Compliance gate**: a research-flag enforcement layer that refuses
  CC-BY-NC / CC-BY-NC-SA weights (F5-TTS / Fish-Speech / EnCodec) unless the
  caller opts in — from the same `vokra.provenance.*` chunk the compliance
  API surfaces.
- **Model hub**: [`huggingface.co/vokra`](https://huggingface.co/vokra) —
  **16 converted GGUFs live** as of 2026-07-23. Every artifact carries a
  matching model card generated from its own metadata, a `LICENSE`, a
  `NOTICE` (attribution-required cases), and a `SOURCE.md` with the
  upstream URL and re-conversion recipe. The publication path
  (`scripts/publish/*.sh`) is a five-tier gate that fails closed on
  contractual bans (VOICEVOX / CSJ / JSUT-JVS), on artifacts that cannot
  state their own licence, and on blank owner sign-off; a
  `restamp_provenance` low-memory rewrite lets an 8.7 GB checkpoint
  publish on a 16 GB host without vast.ai (peak footprint measured at
  6.4 MB). Currently live: **whisper-{base,small,medium,turbo}** (matched
  to each source repo's licence, apache-2.0 / mit), **kokoro-82m** and
  **kokoro-82m-stacked** (54 voices + 178 phoneme symbols),
  **piper-plus-css10-ja-6lang** and **piper-plus-mera-multilingual**,
  **silero-vad-v5**, **campplus-speaker-encoder**, **dac-24khz**, **mimi**
  (CC-BY-4.0 with Kyutai attribution in-card), **deepfilternet3**,
  **utmos22-strong**, **moshiko-7b-bf16** (15 GB, with an unmissable
  "not real-time on this runtime" warning), and **voxtral-mini-3b-2507**
  (8.7 GB).

Everything above holds Vokra's **zero-external-dependency** invariant: the
resolved dependency graph contains only first-party `vokra-*` crates,
enforced in CI.

Public reference documents (Japanese):

- [docs/license-audit.md](docs/license-audit.md) — model / dependency
  license audit
- [docs/legal-compliance.md](docs/legal-compliance.md) — EU AI Act, SB 942,
  ELVIS Act, C2PA compliance

Detailed requirement / deliverable / milestone planning is maintained
privately by the maintainer; the roadmap summary below reflects it.

## Roadmap

Durations are engineering estimates under a Claude Code-driven
implementation model; any calendar dates derived from them are **rough
indications only** ("目安"), not commitments.

| Phase | Estimated duration | Focus |
|---|---|---|
| v0.1 spike | 1.5-2 months | Rust scaffold, GGUF loader + `vokra.*` metadata, STFT/iSTFT/mel ops, Silero VAD, Whisper base, piper-plus native TTS, CPU backend (AVX2/NEON), C ABI, Unity demo, public repo + CI gates — **done** |
| v0.1 MVP | 1.5-2.5 months | K-quant loader, engine, streaming, resample, `vokra-cli` / `vokra-eval`, real 8-language G2P wiring, native CAM++ zero-shot cloning, `vokra-mmap` — **done** |
| v0.5 | 2.5-4 months | Metal + CUDA backends (graph evaluator + per-model GPU dispatch; Whisper end-to-end on both, validated on M1 / RTX 4090), Whisper large-v3 conversion + tokenizer, whole-encoder and per-decoder-step device residency (large-v3 RTF < 0.15 on RTX 4090, measured 0.081-0.115), Kokoro-82M, `vokra-server` (4 HTTP compatibility APIs), `bench --backend` — **done** (merged to `main`) |
| v0.9 | 4-5 months | CUDA complete, Vulkan, CosyVoice2, Voxtral, RVV 1.0 baseline — **done** (merged to `main`) |
| **v1.0-rc** (current) | 4-5 months | WebGPU/WASM (**landed**: browser Whisper base over a raw WebGPU import shim + WASM SIMD128 2-artifact CPU path, npm package CD — see [docs/tutorials/web.md](docs/tutorials/web.md)), Sesame CSM-1B, Moshi (full-duplex + AEC), all-platform official support — **feature-complete on the development branch** (owner verification pending) |
| v1.0 GA | 8+ months | CoreML (ANE) / QNN delegates, MCU tier re-evaluation, commercial GA, C ABI freeze (semver compliance from v1.0) |

Cumulative estimate to v1.0 GA: **20-25 months**. Version labels were
re-assigned on 2026-07-14: the scope formerly planned through v2.0 now
ships as v1.0 (the former v1.0 / v1.5 phases are now v0.9 / v1.0-rc).
v1.0-rc is a semver prerelease; the C ABI freezes at the v1.0 GA tag.
The v0.1 spike was extended from 1-1.5 months to 1.5-2 months when the
piper-plus native TTS implementation was added to its scope (decision of
2026-07-02).

## piper-plus integration (native TTS)

[piper-plus](https://github.com/ayutaz/piper-plus) is an MIT-licensed Piper
fork by the project owner (8-language G2P without eSpeak-NG, MB-iSTFT-VITS2
decoder). Vokra integrates it as the standard TTS layer and as **Vokra's
first natively implemented TTS model** (decided 2026-07-02):

- The MB-iSTFT-VITS2 inference stack (text encoder / duration predictor /
  flow / MB-iSTFT decoder) is reimplemented natively in Rust. The earlier
  plan to wrap the existing ONNX-based implementation was dropped.
- **The end-to-end inference path contains no onnxruntime.** Voice models
  are converted offline to GGUF; the runtime loads only GGUF.
- G2P (text preprocessing, 8 languages: JA/EN/ZH/ES/FR/PT/SV/KO) is reused
  from piper-plus for the time being; a Rust port will be re-evaluated
  later.

## Using the C ABI

Vokra exposes a single C header, [`include/vokra.h`](include/vokra.h)
(cbindgen-generated; regenerate with `scripts/gen-c-abi.sh`). Building the
`vokra-capi` crate produces the shared and static libraries:

```sh
cargo build -p vokra-capi --release
# -> target/release/libvokra.dylib | libvokra.so | vokra.dll  (+ libvokra.a)
```

A session is created from a GGUF model; the architecture is detected from the
file's `vokra.model.arch` metadata and the matching task is wired
automatically (Whisper → ASR, Silero VAD → VAD stream, piper-plus → TTS). All
functions return a `vokra_status_t` (`VOKRA_OK` is 0); on error, a per-thread
message is available from `vokra_last_error()`. Vokra-allocated outputs are
released with their matching `vokra_*_free` / `vokra_*_destroy` function.

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
cc app.c -Iinclude -Ltarget/release -lvokra -Wl,-rpath,target/release -o app
```

Runnable end-to-end examples (ASR / TTS / VAD) live in
[`tests/capi/`](tests/capi); `scripts/run-capi-smoke.sh` builds and runs them.
The M0 (v0.1 spike) ABI is **not** stable — it may change in breaking ways
until the v1.0 semver commitment.

## Planned model support

The official model zoo distributes **Apache-2.0 / MIT weights only**. See
[docs/license-audit.md](docs/license-audit.md) for the full audit.

| Model | Task | License (code / weights) | Commercial use | Planned |
|---|---|---|---|---|
| Silero VAD v5 | VAD | MIT / MIT | Yes | v0.1 MVP |
| Whisper base/small/medium/large-v3/turbo | ASR | MIT / MIT | Yes | v0.1 MVP (base), v0.5 (large-v3), v1.0-rc (small/medium/turbo) |
| piper-plus | TTS | MIT / MIT | Yes | v0.1 spike (native implementation) |
| Kokoro-82M | TTS | Apache-2.0 / Apache-2.0 | Yes | v0.5 |
| CosyVoice2 | TTS / S2S | Apache-2.0 / Apache-2.0 | Yes | v0.9 |
| Voxtral (Mistral) | ASR / S2S | Apache-2.0 / Apache-2.0 | Yes | v0.9 |
| Sesame CSM-1B | S2S | Apache-2.0 / Apache-2.0 | Yes | v1.0-rc |
| Moshi (Helium + Mimi) | S2S | Apache-2.0 / CC-BY 4.0 (attribution required) | Yes, with credit | v1.0-rc |
| F5-TTS | TTS | MIT / **CC-BY-NC 4.0** | **No (non-commercial weights)** | Engine support only; weights excluded from the official zoo, behind a research flag |
| Fish-Speech v1.4/v1.5 | TTS | Apache-2.0 / **CC-BY-NC-SA 4.0** | **No (non-commercial weights)** | Engine support only; weights excluded, research flag |
| RVC v2 / GPT-SoVITS | VC | MIT / unclear | Restricted (training-data concerns) | Separate repository `vokra-voiceclone-experimental` |
| Bark (Suno) | TTS | MIT / MIT (voice-cloning retraining prohibited by Suno policy) | Restricted | post-v1.0 GA (under consideration, research flag) |
| StyleTTS 2 | TTS | MIT / unclear (audit pending) | Restricted | post-v1.0 GA (after audit) |
| Matcha-TTS | TTS | MIT / MIT | Yes | post-v1.0 GA |

Notes:

- **F5-TTS and Fish-Speech weights are CC-BY-NC(-SA) licensed and are not
  included in any official Vokra distribution.** The engine can run them
  for research via an explicit research flag.
- Voice cloning (RVC v2, GPT-SoVITS, speaker cloning) is fully separated
  into the `vokra-voiceclone-experimental` repository for legal reasons
  (ELVIS Act / NO FAKES Act); speaker embedding for zero-shot TTS stays in
  core.
- Piper (OHF-Voice/piper1-gpl) is **not** supported (GPL-3.0 + eSpeak-NG
  GPL-3.0); piper-plus is the only Piper-family integration.

## Community

- **First step**: [docs/good-first-tasks.md](docs/good-first-tasks.md) —
  self-contained starting points, each with a file:line anchor or a
  reproduction command, acceptance criteria you can check yourself, and a
  rough size.
- **Questions & discussion**: open a
  [GitHub issue](https://github.com/ayutaz/vokra/issues).
- **Issues / pull requests**: see [CONTRIBUTING.md](CONTRIBUTING.md). All
  changes go through PRs with CI quality gates.

## License

Licensed under the [Apache License, Version 2.0](LICENSE).

Additional licensing and distribution notices — the BigVGAN scratch
reimplementation policy and the NVIDIA runtime non-bundling policy — are
recorded in [NOTICE](NOTICE).
