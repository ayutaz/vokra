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

> **CPU preview:** [v0.4.0-rc.1](https://github.com/ayutaz/vokra/releases/tag/v0.4.0-rc.1)
> is available for evaluation. The published Linux CLI has functional checks for
> fixed Whisper base ASR and Silero VAD v5 models. See the
> [installation guide](docs/release-0.4.0-rc1.md) for exact files and limitations.
> This does not establish numerical parity, performance, other-OS model inference
> or full-catalog Apple Metal support. [Current project status](docs/current-status.md)
> tracks releases, source, PRs and remaining work.

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

Choose the guide for your task:

| Task | Guide |
|---|---|
| Try the verified prebuilt CPU preview | [Preview installation](docs/release-0.4.0-rc1.md) |
| Build from source and convert checkpoints | [Getting started](docs/getting-started.md) |
| Use model-specific CLI options and formats | [CLI tutorial](docs/tutorials/cli.md) |

## Model and backend status

A converter, a GGUF binder, native inference, independent numerical parity,
publication and real-device validation are separate milestones. A listed model
or successful build does not establish all of them. Use
[current status](docs/current-status.md) for the dated inventory and outstanding
work, and [model cards](https://huggingface.co/vokra) for artifact-specific terms.

CPU is the default. Accelerator support is optional and operation-specific;
see the [backend guide](docs/backend-guide.md) and
[architecture](docs/architecture.md). Unsupported operations and unavailable
devices return errors instead of silently falling back to CPU.

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
