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

> **Published release candidate (2026-10-18 JST):** [`v0.4.0-rc.1`](https://github.com/ayutaz/vokra/releases/tag/v0.4.0-rc.1)
> is published as GitHub release `416018534` from exact source/tag
> `d84d311e98d0abb2987b5c4e96157ae8f2068e1a` (`draft=false`, `prerelease=true`).
> Latest remains `v0.3.0`; the release has 22 uploaded assets. Final hosted run
> [`39062234813`](https://github.com/ayutaz/vokra/actions/runs/39062234813) passed
> all 10 jobs: three-OS producers, fresh transferred native CLI `--help` and
> C ABI exact-version checks, Linux assembled-CLI Whisper/Silero functional
> smoke, and publication byte round-trip. The recovered receipt SHA-256 is
> `d4b4656fbfa67c8f10b7dd5f6642fc896780ab11da4bc4db80d7922bb2348697`, binding
> the Linux CLI `2a5d4ba5ca543b749f6738bd92d59da0e884a6693ab7e222b8b38ce81a91d8c4`
> (13,344,272 bytes) and manifest `32b03f537f48ff6f2bb7b12fda1a9426090debbd59da361c0ffa59e8f7a60e92`.
> This remains a CPU-only preview: it does not claim macOS/Windows model
> inference, Apple Metal/no-fallback, full-catalog completion, performance or
> numerical parity, and it does not update model/Space/HF/registry/X/Scaleway
> artifacts.
>
> **Historical prepublication note (2026-10-17):** `v0.4.0-rc.1` packaging was
> in progress; the published Latest remains `v0.3.0`. At source `9ee5834d`,
> Whisper base ASR and Silero VAD v5 speech/silence passed hosted Linux CPU
> functional smoke. This is not numerical parity or Apple/Metal validation.
> See the [actual CPU results](docs/handoff/release-0.4.0-preview-cpu-results-2026-10-17.md)
> and [candidate acceptance plan](docs/handoff/release-0.4.0-rc1-packaging-2026-10-17.md).
> The [candidate installation guide](docs/release-0.4.0-rc1.md) is staged for
> publication, not a claim that its download assets already exist.
>
> **Historical campaign snapshot (2026-10-17; superseded by the published RC):** packaging baseline main is
> `fab4638a`. PR #255's model-free CPU/API and normal acceptance are complete.
> PR #259 is closed without merge; its additional observer patch has no
> accepted upstream compilation/API result. Six other PRs remain open at
> this snapshot. The latest retained inventory is the
> 2026-10-16 snapshot: 194 repositories, 136 code/artifact-full and 58 non-full;
> it is not a fresh inventory or an Apple hardware pass count. Per-model
> non-Apple prerequisites, owner/legal decisions, independent real-weight CPU,
> final packets and Apple CPU/Metal/no-fallback remain incomplete. See the
> [current status and remaining tasks](docs/handoff/current-status-and-remaining-tasks-2026-10-17.md).
>
> **Historical audit snapshot (2026-10-13):** the workspace is `0.3.0` development;
> the latest VibeVoice Realtime-0.5B closure audit was a static-only VAST
> audit on disposable worker `55648877`. It authenticated the selected source,
> lock, collector, and provenance inputs, then verified recovery and destruction
> of the worker. It did not download or execute model weights, run a model,
> invoke a package/native API, or upload an artifact.
> The collector verdict is `OWNER_REVIEW_REQUIRED_NO_UPLOAD`; the controller
> completed separately. Owner/legal approval, license review,
> provenance review, native runtime status, and runtime parity are unresolved.
> The audit found missing bundled license files for `safetensors` 0.5.3 and
> `tokenizers` 0.22.2, plus ten native packages requiring review. See the
> [VibeVoice static-closure audit](docs/handoff/vibevoice-realtime-static-closure-audit-2026-10-13.md)
> and [documentation refresh record](docs/handoff/documentation-refresh-2026-10-13.md).
> At exact `main` HEAD `7d38061c3dcb56f6a16bcf4b8085d966b3cdab09`, the fresh
> metadata/card-only audit reports 194 repositories, 193 GGUF-bearing
> repositories, 198 GGUF files, CPU `136/43/14/1`, and Metal `136/57/1`,
> leaving 58 unresolved public rows. This audit does not claim Apple
> CPU/Metal/no-fallback verification.
>
> **Historical catalog snapshot (2026-10-08):** the workspace is `0.3.0` development;
> this documentation review and the read-only public audit use exact GitHub
> `main` checkout `d100d93778191ccab77bd1c37fe3552e3d889758`. The
> metadata/card-only public audit on 2026-10-08 reports 194 repositories, 193
> GGUF-bearing repositories and 198
> GGUF files: CPU code status is 136 full, 43 partial, 14 without a runtime binder and
> 1 non-artifact. That dated audit recorded these counts; it did not download
> or execute model weights.
> Apple Metal is 136 full, 57 blocked by CPU and 1
> non-artifact. VibeVoice Realtime-0.5B now has a strict structural
> `vibevoice_streaming` binder and CLI inspection route, so it is partial;
> its complete 605-tensor checkpoint has been converted and a narrow
> independent Microsoft-source reference probe is accepted, but full
> streaming, native CPU parity, and Apple CPU/Metal verification remain
> pending. Qwen3-TTS PR #109 subsequently completed a clean,
> no-upload VAST run for all four public variants plus the shared 12 Hz
> decoder: strict official-weight reload and independent CPU parity passed
> 4/4, and the recovered batch closed 38/38 checksums. Apple CPU/Metal and
> no-fallback verification, and any public-artifact replacement, remain
> pending. The bounded Apple Silicon batch recorded
> in the [Apple results](docs/handoff/mac-cpu-metal-scaleway-results-2026-09-11.md)
> passed only for its named scopes, and four explicitly approved artifacts
> were subsequently published. That snapshot recorded 58 unresolved public rows;
> refresh live metadata before treating these counts as current.
> On 2026-10-04, a corrected VAST source-only audit authenticated its fixed
> Kyutai source inputs and independently verified worker cleanup. It did not
> acquire or execute model weights, and it does not advance CPU parity, Apple
> hardware, or publication status.
> Version 0.3.0 is the first GitHub source release; external package registries
> remain disabled. Vokra remains pre-1.0, so Rust APIs, the C ABI, GGUF metadata
> and model coverage may change. Pin the exact `v0.3.0` tag when evaluating
> this release in another project.

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

You need Git and Rust 1.89 or newer (the effective MSRV; the CPU backend
requires 1.89). Build the CLI from source:

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
operation-specific coverage. CoreML has an experimental whole-submodel
delegate path for the Whisper encoder; QNN remains an SDK-gated experimental
delegate scaffold. See the [backend guide](docs/backend-guide.md) before
selecting an accelerator.

The current coverage snapshot separates source/runtime readiness from numerical
evidence and publication. Incomplete routes fail closed; unsupported operations
never fall back silently to CPU. The UTMOS legacy Lightning checkpoint is also
intentionally refused by the restricted `weights_only=True` loader, so no
UTMOS numeric parity result is claimed.

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
