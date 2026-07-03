# Vokra

**English** | [日本語](README.ja.md)

**Vokra** is an inference runtime specialized for speech AI — TTS, ASR,
speech-to-speech, voice conversion, speaker identification, and VAD — built
in Rust as an alternative to ONNX / ONNX Runtime for speech workloads.

- **Pronunciation**: "vo-krah" (English) / 「ヴォクラ」 (Japanese)
- **License**: [Apache-2.0](LICENSE)
- **Status**: **v0.1 spike, under active development** — pre-release; APIs,
  file formats, and model coverage are unstable and incomplete.

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
  VAD, speech enhancement (AEC/denoise), speaker embedding, F0 extraction,
  and audio watermarking (EU AI Act Article 50 readiness).
- **CPU as a first-class backend** (x86-64 SSE2 baseline through
  AVX2/AVX-512/AMX, ARM64 NEON through SVE/SME, with runtime dispatch),
  then staged GPU/NPU acceleration: Metal, CUDA, Vulkan, WebGPU, CoreML,
  QNN.
- **All platforms are in scope**: Windows / macOS / Linux / Android / iOS /
  Web, plus x86-64 and ARM64 servers. The roadmap staggers *when* each
  backend gets official acceleration, not *whether* a platform is
  supported.

## Status and design documents

The project is in the **v0.1 spike** phase (Rust scaffold, GGUF loader,
STFT/iSTFT/mel ops, Silero VAD, Whisper base, piper-plus native TTS, CPU
backend, C ABI, Unity demo). Nothing is ready for production use yet.

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
| **v0.1 spike** (current) | 1.5-2 months | Rust scaffold, GGUF loader + `vokra.*` metadata, STFT/iSTFT/mel ops, Silero VAD, Whisper base, piper-plus native TTS, CPU backend (AVX2/NEON), C ABI, Unity demo, public repo + CI gates |
| v0.1 MVP | 1.5-2.5 months | Silero VAD v5 + Whisper base official support; model-parity checkpoint (Kill switch I) right after release |
| v0.5 | 2.5-4 months | Metal backend, CUDA backend start, Kokoro-82M, Whisper large-v3/turbo, OpenAI-compatible server API |
| v1.0 | 4-5 months | CUDA complete, Vulkan, CosyVoice2, Voxtral, RVV 1.0 baseline |
| v1.5 | 4-5 months | WebGPU/WASM, Sesame CSM-1B, Moshi (full-duplex + AEC), all-platform official support complete |
| v2.0 | 8+ months | CoreML (ANE) / QNN delegates, MCU tier re-evaluation |

Cumulative estimate to v2.0: **20-25 months**. The v0.1 spike was extended
from 1-1.5 months to 1.5-2 months when the piper-plus native TTS
implementation was added to its scope (decision of 2026-07-02).

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
| Whisper base/small/medium/large-v3/turbo | ASR | MIT / MIT | Yes | v0.1 MVP (base), v0.5 (large-v3/turbo) |
| piper-plus | TTS | MIT / MIT | Yes | v0.1 spike (native implementation) |
| Kokoro-82M | TTS | Apache-2.0 / Apache-2.0 | Yes | v0.5 |
| CosyVoice2 | TTS / S2S | Apache-2.0 / Apache-2.0 | Yes | v1.0 |
| Voxtral (Mistral) | ASR / S2S | Apache-2.0 / Apache-2.0 | Yes | v1.0 |
| Sesame CSM-1B | S2S | Apache-2.0 / Apache-2.0 | Yes | v1.5 |
| Moshi (Helium + Mimi) | S2S | Apache-2.0 / CC-BY 4.0 (attribution required) | Yes, with credit | v1.5 |
| F5-TTS | TTS | MIT / **CC-BY-NC 4.0** | **No (non-commercial weights)** | Engine support only; weights excluded from the official zoo, behind a research flag |
| Fish-Speech v1.4/v1.5 | TTS | Apache-2.0 / **CC-BY-NC-SA 4.0** | **No (non-commercial weights)** | Engine support only; weights excluded, research flag |
| RVC v2 / GPT-SoVITS | VC | MIT / unclear | Restricted (training-data concerns) | Separate repository `vokra-voiceclone-experimental` |
| Bark (Suno) | TTS | MIT / MIT (voice-cloning retraining prohibited by Suno policy) | Restricted | v2.0+ (under consideration, research flag) |
| StyleTTS 2 | TTS | MIT / unclear (audit pending) | Restricted | v2.0+ (after audit) |
| Matcha-TTS | TTS | MIT / MIT | Yes | v2.0+ |

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

- **Discord**: TBD — the invite link will be published here once the server
  is up (tracked as M0-01-T01).
- **Issues / pull requests**: see [CONTRIBUTING.md](CONTRIBUTING.md). All
  changes go through PRs with CI quality gates.

## License

Licensed under the [Apache License, Version 2.0](LICENSE).

Additional licensing and distribution notices — the BigVGAN scratch
reimplementation policy and the NVIDIA runtime non-bundling policy — are
recorded in [NOTICE](NOTICE).
