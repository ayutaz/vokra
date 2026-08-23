# Vokra

**English** | [日本語](README.ja.md)

[![CI](https://github.com/ayutaz/vokra/actions/workflows/ci.yml/badge.svg)](https://github.com/ayutaz/vokra/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Vokra is a speech-first inference runtime written in Rust. It implements the
audio pieces that general-purpose graph runtimes often leave to application
code: streaming state, STFT/iSTFT and mel frontends, vocoders, neural codecs,
CTC/RNN-T decoding, VAD, speaker features, pitch extraction, and audio
enhancement.

Vokra loads provenance-aware GGUF files and does not load ONNX graphs at
runtime. The default runtime has no third-party Cargo dependencies: the root
`Cargo.lock` contains only first-party `vokra-*` crates.

> **Release status:** `0.1.0` is prepared as the first tagged release. Rust
> APIs, the C ABI, GGUF metadata, and model coverage remain pre-1.0 and may
> change. Pin an exact release when evaluating Vokra in another project.

## Why Vokra

- **Audio-native execution:** speech frontends, streaming caches, decoders,
  vocoders, codecs, VAD, and enhancement are native operators rather than ONNX
  graph glue.
- **Small dependency surface:** runtime crates depend only on first-party
  `vokra-*` crates. Offline conversion remains separate from runtime loading.
- **Explicit failures:** unsupported operations and unavailable devices return
  errors; GPU work never silently falls back to CPU.
- **Reproducible model files:** Vokra GGUF metadata records frontend settings,
  topology, quantization policy, source provenance, and licence information.
- **Portable integration:** CPU is the default; Metal, CUDA, Vulkan, and WebGPU
  are opt-in. A generated C header supports native and language bindings.

## Quick start

You need Git and Rust 1.89 or newer. Build the CLI from source:

```sh
git clone https://github.com/ayutaz/vokra.git
cd vokra
cargo build --release -p vokra-cli
```

Download the published Whisper base GGUF and run the included public-domain
audio fixture:

```sh
curl -L https://huggingface.co/vokra/whisper-base/resolve/main/whisper-base.gguf \
  -o whisper-base.gguf
target/release/vokra-cli run \
  --model whisper-base.gguf \
  --input tests/fixtures/audio/jfk-30s.wav
```

Use the built-in help before converting or running another architecture:

```sh
target/release/vokra-cli --help
target/release/vokra-cli convert --help
target/release/vokra-cli run --help
```

The [getting-started guide](docs/getting-started.md) covers conversion, VAD,
TTS, benchmarking, and the C ABI.

## Model and backend status

Vokra covers ASR, TTS, speech-to-speech, VAD and turn-taking, keyword
spotting, speaker processing, pitch, codecs and vocoders, enhancement,
separation, and audio understanding. Maturity is tracked per architecture:
a converter, a GGUF loader, a native forward pass, numerical parity, and a
published artifact are separate milestones. The existence of one does not
imply the others.

Use these sources instead of a copied model list:

- `vokra-cli convert --help` — accepted converter identifiers;
- `vokra-cli run --help` — CLI-routed inputs, outputs, and backend options;
- the [Vokra model hub](https://huggingface.co/vokra) — published artifacts and
  model-specific licence cards;
- [`crates/vokra-cli/src/engine.rs`](crates/vokra-cli/src/engine.rs) — explicit
  runtime routing and deferred-operation registry for developers.

CPU is the default backend. Metal, CUDA, Vulkan, and WebGPU are opt-in and have
operation-specific coverage; CoreML and QNN are experimental delegates. See
the [backend guide](docs/backend-guide.md) before selecting an accelerator.

## Library integration

Build the C library with:

```sh
cargo build --release -p vokra-capi
```

[`include/vokra.h`](include/vokra.h) is the generated C reference. The
[API index](docs/api-reference.md) links to the Rust and binding surfaces,
including Python, Swift/iOS, Unity, Godot, Android, web, and server examples.
The C ABI remains pre-1.0 and is not frozen.

## Documentation

- [Documentation map](docs/README.md)
- [Getting started](docs/getting-started.md)
- [CLI tutorial](docs/tutorials/cli.md)
- [Architecture](docs/architecture.md)
- [Backend guide](docs/backend-guide.md)
- [Migration guide](docs/migration-guide.md)
- [Licence audit](docs/license-audit.md) and
  [legal/compliance notes](docs/legal-compliance.md)

## Contributing

Contributions are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) before a
large change, and use the
[good first tasks](docs/good-first-tasks.md) for scoped entry points. Bugs and
proposals can be filed in [GitHub Issues](https://github.com/ayutaz/vokra/issues).
Participation is governed by the [Code of Conduct](CODE_OF_CONDUCT.md).
Report vulnerabilities privately as described in the
[Security Policy](SECURITY.md), not in a public issue.

## Licence

Vokra source code is licensed under [Apache-2.0](LICENSE). Model weights and
reference assets may use different licences; review each model card,
[`docs/license-audit.md`](docs/license-audit.md), and [NOTICE](NOTICE) before
redistribution or commercial use. Non-commercial weights are excluded from
the default publication path unless an explicit research-only gate is used.
