# Changelog

All notable changes to Vokra are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning
follows [SemVer](https://semver.org/spec/v2.0.0.html).

**Pre-1.0**: the C ABI is **not frozen** before v1.0 (see ADR-0003,
requirement IF-01). Every pre-1.0 release may change the C ABI without
deprecation windows. Here "v1.0" means the **v1.0 GA** tag (M5 close, WP
M5-13); the **v1.0-rc** prerelease (`1.0.0-rc.N`, M4) is still pre-1.0 and
therefore not frozen — see the `[1.0.0-rc.1]` ABI-policy notes below.

## [Unreleased]

Milestone: v0.5 (M2). CC-side implementation is complete for the WPs
listed below; owner-side verification (real-device RTF measurement,
license sign-off, PyPI / Unity secret provisioning) is tracked in
`docs/m2-owner-verification-checklist.md`.

### Added

- **Metal backend** (M2-01): hand-written `objc_msgSend` FFI + MSL
  compute kernels, no `metal-rs` or `MPS`/`MPSGraph`/`MLX` dependency
  (preserves the zero-external-dependency invariant). Full Whisper runs
  end-to-end on Metal on Apple M1 with encoder + decoder-step device
  residency (`MetalDecodeSession`), bit-identical to the CPU greedy path.
- **CUDA backend** (M2-03): hand-written Driver API + NVRTC FFI, both
  `dlopen`'d at runtime (satisfies the NVIDIA EULA
  "installed-only-in-a-private-directory" clause). `libcuda` /
  `libnvrtc` are the only shared libraries touched; no `cudart` /
  `cudnn` / `cublas` bundling. Full Whisper runs end-to-end with
  encoder + decoder-step device residency (`CudaDecodeSession`,
  `CudaDecodeSessionPool` opt-in). **Whisper large-v3 RTF < 0.15 on
  RTX 4090** (measured 0.081–0.115 over 30 s of audio).
- **CUDA Flash-Attention v2** (`vokra_flash_attn_v2_causal_f32`): fused
  causal-attention kernel with online-softmax rescale, `Br=16`, `Bc=64`,
  `grid.z = n_head` for inter-head parallelism. FA v3 (Hopper WGMMA /
  TMA) is deferred to v1.5+.
- **iOS build scaffold** (M2-02): 2-slice XCFramework (device arm64 +
  Simulator arm64/x86_64) built by `scripts/build-ios.sh`; Swift Package
  (`Package.swift`) exposes the Clang module. `#[cfg(target_os = "ios")]
  compile_error!(...)` in `vokra-backend-cuda` blocks accidental CUDA
  linkage on iOS.
- **Graph fusion** (M2-04): IR fusion pass infrastructure
  (`vokra-core/src/ir/fusion/`), log-mel fusion pattern (STFT + mag +
  Mel + log collapsed into a single kernel) with AVX2 + NEON
  specializations, wired through the `mel-frontend` `vokra-cli bench`
  task; NFR-PF-13 5% regression gate.
- **`istft_streaming` op** (M2-05): tail-buffer length as an explicit
  attribute; per-layer state carry-over.
- **Whisper large-v3 / turbo support** (M2-06): converter now derives
  `dims` and `n_mels` from checkpoint shape; F16 passthrough for fp16
  checkpoints; the multilingual text vocab is embedded in the GGUF
  (`vokra.tokenizer.model`) so large-v3 emits correct transcripts;
  special-token ids are `n_vocab`-driven.
- **Kokoro-82M skeleton** (M2-07): `vokra-models/src/kokoro/` (config /
  weights / nn / text_encoder / prosody / decoder), `vokra-convert`
  Kokoro adapter, iSTFTNet vocoder using `vokra_ops::istft` — **not**
  `vocos_head` (reviewer A correction).
- **Quantization policy** (M2-08): `QuantScheme` (`W4A16Q4K`,
  `W8A8Int8`, `FP16`, `FP32`), minimum-dtype registry, `vokra.quant.*`
  metadata chunk, `--policy-preset` CLI switch, `vokra-eval` degradation
  detection.
- **`vokra-server`** (M2-09): isolated workspace
  (`integrations/vokra-server`, own `Cargo.lock`, own `deny.toml`)
  exposing four HTTP compatibility APIs — **OpenAI Whisper**
  (`/v1/audio/transcriptions`, faster-whisper drop-in), **vLLM**
  (`/v1/completions`, `/v1/chat/completions`), **piper-plus HTTP**
  (`/api/tts`), and **Wyoming Protocol** (HA Voice backend).
- **Unity UPM package** (M2-11): `bindings/unity/com.vokra.unity`
  skeleton with IL2CPP-safe callbacks (`[MonoPInvokeCallback]` + static
  readonly delegate root + `GCHandle`), Android `persistentDataPath`
  helper, iOS `DllImport("__Internal")` switch. Publisher: `release.yml`
  ships a UPM tarball per tag; `nightly-il2cpp.yml` runs a game-ci
  IL2CPP smoke test (requires `secrets.UNITY_LICENSE`).
- **Python bindings** (M2-12): `bindings/python/` — pure `ctypes`
  wrapper (no `pyo3`, no Rust code added), `pyproject.toml` with
  `hatchling` build backend, `name = "vokra"`. `_native.py` auto-locates
  the platform's `libvokra.*`. `cibuildwheel` matrix (`linux/x86_64`,
  `linux/aarch64`, `macosx_11_0_universal2`, `win_amd64`), Python 3.9–3.12.
  Publisher: `release.yml` runs `python-pypi-publish` (OIDC trusted
  publisher preferred, `PYPI_API_TOKEN` fallback).
- **Compliance gate** (M2-13): `vokra-core/src/compliance/` runtime
  gate + `vokra.provenance.*` chunk. CC-BY-NC / CC-BY-NC-SA weights
  (F5-TTS, Fish-Speech, EnCodec) are refused without a research flag
  from the compliance API.
- **Kokoro parity dumper + gated test harness**:
  `tools/parity/dump_kokoro_reference.py` regenerates the fixtures under
  `tests/parity/kokoro/`; `crates/vokra-models/tests/parity_kokoro.rs`
  runs fixture-only shape / length checks on every PR, byte-level parity
  when `VOKRA_KOKORO_GGUF` is set and the manifest declares `mode = full`.
- **Docs**: `docs/getting-started.md` (English) + `docs/getting-started.ja.md`
  (日本語) — 5-minute quick start; `docs/migration-guide.md` (+
  `.ja.md`) — from ONNX Runtime / whisper.cpp / sherpa-onnx / faster-whisper
  / piper; `docs/tutorials/{unity,ios,python}.{md,ja.md}` —
  per-platform tutorials.
- **CPU + Vulkan-only build target** (M4-15): `--no-default-features
  --features vulkan` builds of `vokra-capi` (the shipped cdylib) and
  `vokra-cli`; the `metal` / `cuda` backends stay out of the dependency
  graph (enforced by a `cargo tree` audit and the
  `scripts/compliance/check-cpu-vulkan-only-no-nvidia.sh` scanner), the
  `vokra-models/vulkan` feature now forwards to the backend's raw-FFI
  feature, a `build-target-vulkan-only` CI job builds/tests the
  combination, a build-time NOTICE variant omits the NVIDIA runtime
  section (`scripts/gen-notice-cpu-vulkan-only.sh`), and a deterministic
  SPDX 2.3 SBOM is generated per build (`scripts/sbom/generate_spdx.py`).
  No new C ABI symbol: the build flavor is identified by build-time
  metadata only (SBOM / NOTICE / artifact name).

### Changed

- **Metal / CUDA are first-party optional features, default OFF** —
  the root `Cargo.lock` is always `vokra-*`-only. Enabling the feature
  compiles the raw-FFI path.
- **README**: adds distribution artefacts (iOS XCFramework + Swift
  Package, Unity UPM, Python wheel), `vokra-server`, graph fusion,
  quantization policy, compliance gate sections.

### Removed / Descoped

- **Discord entirely** (2026-07-04 and 2026-07-06 owner decisions):
  community Discord was descoped on 2026-07-04 in favour of GitHub
  Issues / Discussions; the M2-10 product-demo Discord bot was
  descoped on 2026-07-06 with the same rationale. `integrations/vokra-discord-bot`
  is not to be added.
- **`.github/workflows/kill-switch-check.yml`** (FR-TL-05) — replaced
  by manual quarterly Go/No-go review under NFR-MT-05 (M2-15 is the
  concrete WP).
- **M0-11 trademark investigation** — dropped; informal clearance
  (documented in CLAUDE.md rebrand history) continues to apply.
- **watermark / C2PA runtime embedding** (FR-CP-01/02, M1-07) — the
  `WatermarkConfig` surface stays as a config-only forward-compat
  hook; runtime embedding is deferred. NFR-LG-01/02 remains a residual
  open item.
- **M1-02 pickle path** — `safetensors` only; `pickle` weights are not
  loaded (`FR-LD-05` already forbids ONNX at runtime, and pickle is
  arbitrary-code-execution-prone).

### Fixed

- **Whisper decoder tokenizer**: large-v3 previously emitted the base
  vocab's text through the special-token id space. The converter now
  embeds the multilingual byte-level BPE vocab in the GGUF
  (`vokra.tokenizer.model`) and the decoder reads it directly, so
  large-v3 / turbo produce correct transcripts.

## [1.0.0-rc.1]

First v1.0 release candidate (semver prerelease `1.0.0-rc.1`). The feature
deltas for v0.5 (M2) / v0.9 (M3) / v1.0-rc (M4) accrue in `[Unreleased]`
above and are rolled into this section by the M4 tag-preparation step (the
v1.0-rc tag is an owner milestone decision, not WP M4-12). This section was
added by **M4-12** to record the **ABI policy for the rc window**.

### ABI policy (v1.0-rc)

- **Not frozen.** The C ABI (`include/vokra.h`) and the `vokra.*` GGUF
  metadata schema are a semver **prerelease** surface at `1.0.0-rc.N`; they
  are **not frozen** at the rc tag.
- **The pre-1.0 policy stays in force through the whole rc series.** Any add /
  rename / remove of an exported C symbol, cbindgen-reflected Rust `pub` item,
  or `vokra.*` GGUF chunk remains legal and still requires a dated entry in
  `docs/abi-changelog.md`. Only the freeze point moved — the policy text is
  unchanged (`docs/abi-changelog.md` "Pre-1.0 policy (prerelease semver)").
- **The freeze fires at v1.0 GA (M5-13), not at the rc tag.** The IF-01 freeze,
  the start of semver ABI-stability compliance, and the promotion of the ABI
  gate (`scripts/check-abi-changelog.sh` / `scripts/abi-diff.sh`) from advisory
  to a required CI check all happen at the v1.0 GA tag (= M5 close, WP M5-13 —
  2026-07-14 v-label reassignment #2). See `docs/handoff/m4-12.md` §(b)(d)(f).
- **A recorded, diffable baseline — advisory, not frozen.** The rc baseline
  (`docs/abi/vokra.h.v1.0-rc-baseline.symbols`, paired Rust surface
  `docs/abi/vokra-rust-public-api.v1.0-rc.list`) gives Unity / Godot / Swift /
  Kotlin / Python / JS integrators (the IF-01 consumers) a recorded, diffable
  baseline to track — it is **advisory, not a frozen one**. The stable-ABI
  commitment to integrators arrives at v1.0 GA (2027-07〜2028-01 estimate)
  rather than at the rc tag; the owner accepted this trade-off
  (`docs/handoff/m4-12.md` §(f)-6).
- **Non-C-ABI surfaces track their own semver.** `vokra-server`'s HTTP
  compatibility APIs (OpenAI-Whisper / vLLM / piper-plus) and the Wyoming
  Protocol endpoint are a separate **protocol-tracking / experimental** tier:
  they follow their upstream protocol's semver, not Vokra's C ABI semver, and
  are outside the IF-01 freeze surface. The npm `@vokra/web` JS/TS API is
  versioned with its own package tag, and the CPU + Vulkan-only build flavor
  (M4-15) is identified by build metadata only — it adds no C ABI symbol and
  carries no market claim. The `## Non-C-ABI surface areas` STABILITY-block
  section that names these tiers in the header is added at the M5-13 freeze
  (`docs/handoff/m4-12.md` §(e)-1).

## [0.1.0] — 2026-07-04

Initial public release. **v0.1 spike + v0.1 MVP** implementations
complete; repository made public on 2026-07-04 with CI quality gates
enforced (`.github/workflows/ci.yml`).

### Added

- **Rust Cargo workspace** with 12 crates (`vokra-core` / `-ops` /
  `-backend-cpu` / `-backend-metal` / `-backend-cuda` / `-models` /
  `-piper-plus` / `-capi` / `-convert` / `-cli` / `-eval` / `-mmap`).
- **GGUF loader + `vokra.*` metadata chunks** (`vokra.frontend.*`,
  `vokra.whisper.*`, `vokra.piper.*`, `vokra.campplus.*`,
  `vokra.tokenizer.model`, `vokra.provenance.*`).
- **Speech-first native operators**: STFT / iSTFT / mel filterbank with
  explicit window / hop / norm / RFFT attributes, resampling
  (`speexdsp`-style polyphase sinc, no `soxr` GPL dependency).
- **Silero VAD v5** (1:1 native subgraph reimplementation).
- **Whisper base + large-v3** native implementation (encoder + decoder
  + beam search + embedded detokenizer).
- **piper-plus native TTS**: MB-iSTFT-VITS2 inference (text encoder /
  duration predictor / flow / MB-iSTFT decoder) reimplemented in Rust.
  End-to-end path contains **no `onnxruntime`**. G2P (8 languages,
  JA/EN/ZH/ES/FR/PT/SV/KO) reused from piper-plus.
- **Native CAM++ speaker encoder** (`speaker_encode` op) for zero-shot
  voice cloning (7e-6 parity vs `onnxruntime`).
- **K-quants (Q4_K / Q5_K / Q6_K)** weight loading + offline quantizer;
  native `safetensors` loader.
- **CPU backend** with runtime dispatch (x86-64 SSE2 baseline through
  AVX2 + FMA + F16C, ARM64 NEON).
- **Streaming** (SPSC ring buffer), engine (KV cache, sampler, CFG),
  `frontend_spec` bit-exact enforcement.
- **`vokra-cli`** (`run` / `convert` / `bench` with relative-regression
  gate), **`vokra-eval`** (mel-loss / WER / CER), **`vokra-mmap`**
  (true `mmap` GGUF loading — no `libc` / `memmap2` dependency).
- **C ABI + cbindgen** (`include/vokra.h`) + Unity C# demo.
- **CI quality gates**: build, test, `rustfmt`, `clippy`, numerical
  parity, license (`cargo deny`), zero-dependency invariant, hot-path
  audit, iOS build, Python wheel build, license audit, GPU backends.

[Unreleased]: https://github.com/ayutaz/vokra/compare/v0.1.0...HEAD
[1.0.0-rc.1]: https://github.com/ayutaz/vokra/releases/tag/v1.0.0-rc.1
[0.1.0]: https://github.com/ayutaz/vokra/releases/tag/v0.1.0
