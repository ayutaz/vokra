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

This section accumulates **all** untagged feature deltas — **v0.5 (M2) +
v0.9 (M3) + v1.0-rc (M4)** — that roll into `[1.0.0-rc.1]`. CC-side
implementation is complete for the WPs listed below; owner-side
verification (real-device RTF / GPU / NPU measurement, real-weight
parity flip-the-switch, license sign-off, PyPI / Unity / npm / CDN
provisioning) is tracked in the per-milestone owner checklists
(`docs/m2-owner-verification-checklist.md`,
`docs/m3-owner-verification-checklist.md`,
`docs/m4-owner-verification-checklist.md`). Added entries are grouped by
milestone below.

### Added

#### v0.5 (M2)

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
  TMA) shipped in v1.0-rc (M4-07), confined to `vokra-backend-cuda`.
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

#### v0.9 (M3)

- **CUDA backend complete** (M3-01): graph-executor extended with the
  decoder primitives (Gemv / Softmax / SoftmaxCausal / LayerNorm / Gelu /
  Conv1D), FA v2 `compute_89` pin, an op-coverage test, a long-form
  decoder sanity dumper, and the `gpu-cuda-rtf.yml` scaffold. The formal
  always-on RTF < 0.10 gate is deferred to a self-hosted runner (M2-14 +
  M3-01 5% regression gate); the vast.ai N=10 reference (2026-07-10)
  recorded the decomposed leg at mean 0.766 / CV 0.024 and the gated FA v2
  leg at mean 0.782 / CV 0.024 (honest negative: the `FA_V2_MIN_TQ=16`
  gate does not fire at decoder-step `t_q=1`).
- **Vulkan backend** (M3-02): new `vokra-backend-vulkan` crate, opt-in
  `vulkan` feature, raw `dlopen` FFI (no binding crate), a runtime object
  stack (device / command pool / descriptor set / pipeline / memory /
  buffer), a SHA-256-pinned SPIR-V manifest, and end-to-end
  `OpKind::Copy` / `Add` dispatch. Honest partial: the only real `.spv`
  blobs are the hand-crafted `copy_f32` / `add_f32`; further kernels await
  owner-side `glslc` compilation (completed in M4-13).
  `gpu-vulkan-parity.yml` scaffolded (workflow_dispatch + weekly cron, not
  a required check yet).
- **paged KV cache** (M3-03): 3D `[time, stream, codebook]` logical
  addressing, `block_size` 4 or 2 (audio-rate aligned), time-dimension
  paging with contiguous codebooks, a `KvElement` trait +
  `GpuPagedKvCacheOps` seam, no hot-path malloc / free.
- **KV-cache quantization Q4_0 / Q5_0 / Q8_0** (M3-04): plus a
  `KvQuantVerifyReport` hook and a `KvQuantDequantGemvOps` trait with CUDA
  NVRTC PTX + Metal MSL fused dequant-GEMV kernels (Apple M1
  max |Δ| = 5.245e-6 vs host over 8 shapes × 3 formats).
- **`flow_sampler` + ODE solvers** (M3-05): a runtime function (not a
  graph node) with 5 solvers (DDIM / DPM++ / Euler / Heun / Flow ODE), 3
  cfg modes (none / split_batch / dual_forward), and 3 schedules (linear /
  sway / epss); step-count / CFG changes need no model re-conversion
  (FR-EX-10).
- **`mimi_rvq` codec** (M3-06): RVQ decode subgraph (Mimi, for CosyVoice2)
  with a Mimi NOTICE (CC-BY 4.0 attribution) and an EnCodec model-zoo
  exclusion gate (`scripts/compliance/check-encodec-exclusion.sh`). A
  Metal / CUDA GPU seam was added later (silent CPU fallback still
  forbidden, FR-EX-08).
- **`hifigan_generator` op** (M3-07): FP32 / FP16 parity; the INT8 path is
  opt-in and cannot be enabled without per-channel calibration + a
  spectral check.
- **`length_conditioning` op** (M3-08): F5-TTS / CosyVoice2-style
  target-duration conditioning, kept distinct from `duration_expander` in
  the IR.
- **CosyVoice2** (M3-09): native reimplementation module tree — text
  encoder, Flow Matching + chunk-aware CFM, Mimi bridge, and a
  Mistral-style LLM backbone (RoPE / GQA / SwiGLU / RMSNorm) with
  `forward` / `step` / `greedy_decode`, a GGUF converter, and a
  `parity::assert_vs_hf_reference` flip-the-switch harness. Real HF
  checkpoint parity + streaming-latency / MEL-loss / UTMOS validation are
  owner-gated.
- **Voxtral** (M3-10): native Mistral-flavored ASR / S2S — audio encoder,
  a text decoder with KV cache, a SentencePiece BPE tokenizer, a 30-s
  streaming driver, a pluggable audio adapter (`AdapterKind`
  None / Linear / Mlp / DownsampleLinear) via a `--adapter-config` JSON
  side-car, beam search + n-best (GNMT length penalty, no-repeat-ngram),
  and Metal / CUDA decode sessions. The Whisper-compatible server endpoint
  returns 200. The real adapter side-car + WER measurement are owner-gated.
- **Godot GDExtension** (M3-11): a `vokra-godot` excluded workspace, raw
  `gdextension_interface.h` FFI (no `godot-cpp`), Godot 4.3 ClassDB
  registration for `VokraSession` / `VokraStream` with `catch_unwind`
  panic → Godot Error trampolines, Variant unpack + real dispatch, a
  5-target cross-build script + `godot-crossbuild.yml` + a release
  packaging job + a `check-godot-package-no-nvidia.sh` compliance scanner,
  and asr_demo / tts_demo scaffolds. Real in-editor verification (T19) +
  the WP-close PR (T20) are owner-gated.
- **piper-plus GPU backends** (M3-12): the M0 native MB-iSTFT-VITS2
  implementation now runs on the Metal / CUDA Compute seam
  (`synthesize_with_intermediates` with explicit deterministic backend
  selection); no silent CPU fallback (FR-EX-08).
- **RISC-V RVV 1.0** (M3-13): a runtime dispatch path + `vec_add_f32`
  intrinsics (SpacemiT K1 / Banana Pi BPI-F3 class) + a CI cross-build
  assembly check. RVV 0.7.1 fallback is M4-08.
- **barge-in** (M3-14): `Stream::interrupt()` + the
  `vokra_stream_interrupt` C ABI symbol; in-flight audio output is flushed
  immediately.
- **`vokra-server` multi-session** (M3-15): a concurrent-stream scheduler
  over the paged KV cache, Voxtral / Whisper beam + n-best surfaced over
  HTTP (`no_repeat_ngram` honored core-side), and TTS-latency bench hooks.
  The in-process FakeSynth floor is measured (http 87 µs / TTFA 34 µs);
  the real-network 75 ms budget is owner-measured.
- **ABI changelog infrastructure** (M3-16): `docs/abi-changelog.md` + the
  `scripts/check-abi-changelog.sh` gate, an m0 anchor
  (`docs/abi/vokra.h.m0-anchor.symbols`), a Rust public-api snapshot
  (`docs/abi/vokra-rust-public-api.v0.9.list`), and diff tools. The C ABI
  freeze does **not** fire here (it fires at v1.0 GA / M5-13).
- **`prosody_control` unified API** (M3-17): a pitch / speed / pause /
  emotion surface + trait; the v0.9 scope is CosyVoice2 instruction
  control.

#### v1.0-rc (M4)

All 20 M4 work-packages reached CC-side terminal on 2026-07-15
(`docs/tickets/m4/README.md`); real-weight parity, real-hardware runs,
license sign-off, and owner ADR decisions are pending
(`docs/m4-owner-verification-checklist.md`).

- **WebGPU + WASM backend** (M4-01): browser Whisper base over a raw
  WebGPU extern-import shim (`wgpu` deliberately **not** adopted, to
  preserve the zero-external-dependency invariant), a 2-artifact WASM
  SIMD128 CPU path, and an `@vokra/web` npm package with CD.
- **Unity WebGL link** (M4-02): a staticlib link path + the
  `vokra_session_create_from_bytes` C ABI symbol (load a model from a
  memory buffer, for WebGL). `secrets.UNITY_LICENSE` provisioning is
  owner-side.
- **`aec` op** (M4-03): a SpeexDSP `mdf.c` (MDF / AUMDF) Rust port with a
  time-tagged far-end reference queue, NOTICE section 7, and the
  `vokra_aec_*` C ABI. Completed first, since CSM / Moshi depend on it.
- **RVQ codec completion** (M4-04): `dac_rvq` (greenfield), the
  `encodec_rvq` code path (weights excluded, research flag), and Mimi
  multi-stream.
- **Sesame CSM-1B** (M4-05): a Llama-3.2-flavored backbone + depth
  transformer + the Mimi neural chain (encoder + neural decoder in
  `crates/vokra-models/src/mimi/`, shared with Moshi). Apache-2.0;
  real-weight parity is an owner flip-the-switch harness.
- **Moshi full-duplex** (M4-06): a Helium temporal transformer + per-step
  depformer + inner-monologue text stream + a full-duplex session runtime,
  plus the `vokra_s2s_duplex_*` and `vokra_model_attribution` C ABI (Moshi
  / Mimi weights are Kyutai CC-BY 4.0, attribution required; NOTICE
  section 5, `vokra.provenance.attribution` GGUF chunk). Apache-2.0 code /
  CC-BY 4.0 weights; real-weight parity is an owner flip-the-switch
  harness.
- **FlashAttention v3** (M4-07): a Hopper WGMMA / TMA path, confined to
  `vokra-backend-cuda` (enforced by
  `scripts/check-fa-v3-confinement.sh`). The real Hopper H100 bakeoff is
  owner-side.
- **RISC-V RVV 0.7.1 fallback dispatch** (M4-08): LicheePi 4A (C910) /
  Milk-V Duo (C906) class; real-hardware runs are owner-side.
- **G2P policy ADR** (M4-09): the decision framework is drafted; the owner
  decision section is intentionally left open.
- **MLIR / StableHLO re-evaluation ADR** (M4-10): the evaluation is
  drafted; the owner judgment section is intentionally left open.
- **All-platform support matrix** (M4-11):
  `docs/platform-support/v1.0-rc-support-matrix.md` (50 anchors, the
  `scripts/check-platform-support.sh` gate) confirming Windows / macOS /
  Linux / Android / iOS / Web official support.
- **v1.0-rc ABI baseline snapshot** (M4-12): a recorded, diffable baseline
  of the C ABI (33 exported functions + 11 typedefs,
  `docs/abi/vokra.h.v1.0-rc-baseline.symbols` + the paired Rust surface)
  with updated diff-tool anchors. **Advisory, not a freeze** — the IF-01
  freeze fires at v1.0 GA (M5-13); see the `[1.0.0-rc.1]` ABI-policy notes
  below.
- **Vulkan complete** (M4-13, M3-02 carry-over): runtime + dispatch
  completion. Honest partial — the only real `.spv` blobs are still the
  hand-crafted `copy_f32` / `add_f32`; MatMul / Mul / Softmax etc. await
  owner-side `glslc` compilation.
- **Whisper family small / medium / turbo** (M4-14, M2-06 carry-over): a
  shape-driven converter, a per-size atol lookup, and a 3-size parity CI
  matrix. Real checkpoints + weight sign-off are owner-side.
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
  metadata only (SBOM / NOTICE / artifact name). The critical-safe
  (medical / automotive / defense) market claim stays in M5.
- **FSQ codec** (M4-16): `wavtokenizer_vq` + `xcodec2_fsq` ops
  (synthetic-weight parity), with metadata marked EXPERIMENTAL.
  `wfst_decode` + SynthID stay in M5.
- **CPU ISA server tier** (M4-17): AVX-512 / VNNI / BF16 + AVX-VNNI
  256-bit + ARM64 dotprod / i8mm / bf16 kernels, `IsaPath`
  `#[non_exhaustive]`. AMX is split to M5; perf is cloud-VM advisory (no
  standing runner).
- **UTMOS eval harness** (M4-18): a skeleton + an `AudioMosMetric` trait +
  a `parity_utmos` harness. The UTMOS weight + license are an owner
  kickoff gate (currently **NO-GO-defer**, so the WP may defer to a
  v1.0.x patch); DNSMOS is license-fail-closed.
- **Wyoming server completion** (M4-19, M2-09 completion): accept-loop /
  synthesize-dispatch / multi-session / barge-in wiring + a protocol e2e.
  The public endpoint is a protocol-tracking / experimental tier (outside
  the v1.0 semver / IF-01 surface).
- **audio op subset** (M4-20): `beam_search` word-level timestamps (DTW),
  `speaker_verify`, and `denoise` (DeepFilterNet topology, NOTICE section
  8) + `agc` / `hpf` / `loudness_norm`. BigVGAN / CTC / RNN-T / diarize
  stay in M5 (mechanism anchors recorded).

#### v1.0-rc (M4) — post-terminal CC-gap completions (2026-07-16)

A post-terminal pass (on the same branch, no new C ABI symbol, no new
external dependency) completed several within-WP gaps and fixed one build
regression; the default test suite went 2340 → 2418, all-features
2364 → 2443.

- **Build fix**: the M4-16 FSQ compute methods (`wavtokenizer_vq_f32`,
  `xcodec2_fsq_f32`) had landed without the `Be::WebGpu` match arm their
  RVQ siblings carry, breaking the `wasm32-unknown-unknown --features
  webgpu` build (the M4-01 WASM deliverable). Both arms added (explicit
  `UnsupportedOp`, FR-EX-08).
- **Whisper word timestamps now reach the model + server** (M4-20 / M4-14):
  the converter emits `vokra.whisper.alignment_heads` (built-in openai
  table for base/small/medium/large-v3/turbo, decode-verified against the
  published constants, plus safetensors passthrough); the tokenizer merges
  subword timings into word-level via the (previously unwired)
  `words_from_alignment`; `vokra-server` surfaces a `word_timestamps`
  request field → response DTO. A model without alignment heads reports word
  timestamps unavailable via an explicit error (never a fabricated table).
- **`vokra-server` production startup** (M4-19 / M2-09): model-path
  Config (CLI / env / TOML), `InferenceService` build, OpenAI + vLLM router
  attach, Wyoming + scheduler wiring; a missing / broken GGUF is a hard
  startup error (no silent skip).
- **Sesame CSM / Mimi `from_gguf`** (M4-05 / M4-06): real named-tensor
  binding (synthesized-weight round-trip tested; real-checkpoint numeric
  parity stays owner).
- **Metal decode routing + real-GPU parity**: `MoshiEngine::with_backend`
  now also routes the decode chain (`CsmAudioDecodeChain::with_backend`
  added); new Metal MSL kernels (gamma-only RMSNorm, adjacent-pair RoPE,
  SiLU, fused SwiGLU) and CPU-vs-Metal parity tests for the Mimi decode /
  Moshi / CosyVoice2 forward paths ran on real Apple-M1 hardware
  (max |Δ| ~5–7e-7). The fused device-resident decode-step driver remains a
  follow-up.
- **Streaming enhancement + memory bounds**: `agc` / `hpf` gained a
  stateful streaming API (chunked == whole-buffer); Moshi gained a bounded
  `RingKVCache` for long full-duplex sessions.
- **Test coverage + a bug fix**: WGSL formula-transcription tests for 8
  WebGPU kernels + a fixed `Add` coverage-gate shader-name drift; 12
  Vulkan GLSL arithmetic mirror tests (host-side, M1-native); C-ABI smoke
  tests for the AEC and S2S surfaces; SIMD128 f32x4 activation kernels.

#### v1.0-rc (M4) — real-weight validation campaigns + fixes (2026-07-16/17)

Two evaluation campaigns converted and ran REAL upstream checkpoints
end-to-end on Apple M1 against onnxruntime 1.19.2 CPU
(`docs/bench-baselines/m1-real-weight-eval-2026-07-16/`): Whisper
base/small/medium/turbo transcripts are byte-identical to ORT (same WER),
Mimi/DAC/WavTokenizer real-weight parity all pass, piper is
near-bit-exact. The campaigns surfaced nine defect families only
reachable with real weights + real audio; all were fixed on-branch and
re-verified against the same real checkpoints:

- **Silero VAD**: the official v5 64-sample rolling context
  (`[1,576]`/`[1,288]` input) was missing, so real speech was never
  detected (max prob 0.0037 → 0 segments). `ContextMode::Official` is now
  the default across Session / CLI / C ABI; jfk segments match ORT
  span-for-span; the raw 1:1 parity anchor (7.9e-8) is unchanged. The
  converter now emits both-rate (8 k/16 k) GGUFs byte-identical to the
  committed fixture.
- **Kokoro-82M upstream fidelity**: the native implementation and its
  same-author reference dumper were mutually consistent but diverged from
  the real `kokoro` package (round-trip WER 1.0 = unintelligible). Eleven
  source-verified fixes (LeakyReLU slopes, ALBERT 12-layer + `gelu_new`,
  decoder input wiring, real F0/N contours + full NSF source path,
  duration formula, AdaIN conventions, iSTFT semantics, …) bring
  round-trip WER to 0.0 and mel-L1 vs ORT to the pipeline noise floor
  (0.53 → 0.0835). The reference dumper now imports the REAL upstream
  package (loud abort otherwise) so a self-consistent mirror cannot recur;
  the `PROSODY_F0_ATOL=0.05` special case was removed as an artifact of
  the flawed reference.
- **CosyVoice2**: the real Qwen2 checkpoint's q/k/v attention biases were
  unrepresentable (argmax 0/10 vs upstream) — optional biases now flow
  through weights/forward/converter/`from_gguf` (argmax 10/10,
  max |Δ| 3.4e-5); converter hparams are shape-derived (+`--config` for
  the head split); the flip-the-switch parity harness is de-stubbed.
- **Voxtral**: `TextDecoder` now loads real GQA shapes (decoupled
  `head_dim`, untied `lm_head`); the converter accepts raw BF16 shards as
  verbatim BF16 passthrough (spot-check Δ 0.0) and **refuses to emit a
  weightless GGUF** (was: exit 0 with 1,696 bytes — a success-shaped
  silent failure); multi-shard `*.index.json` input; 762/762 tensors
  convert byte-identically.
- **Mimi PCM roundtrip** first-ever pass: converter neural-chain adapter
  (`vokra.mimi.*` config chunk + 284 structural tensors as exact
  re-layouts) + runtime replicate-pad/SplitRVQ fixes → encode code ids
  4384/4384 = 100 % agreement with upstream, decode PCM max |Δ| 3.67e-6.
- **DeepFilterNet3 real topology**: the denoise op was a synthetic-weight
  scaffold with zero name overlap with the real 133-tensor checkpoint;
  it is now a full libDF transcription (STFT/ERB frontend, conv+GRU
  encoder/decoders, lookahead deep filtering) with sample-level parity —
  enhanced waveform max |Δ| 4.17e-7, SI-SNR 14.768399 dB vs upstream
  14.768398 dB. License cleared (dual MIT/Apache-2.0); a converter
  (`--model denoise`) and an env-gated 21-tap parity suite landed.
- **Whisper word timestamps**: an emission-row off-by-one (the slice
  included `<|notimestamps|>`, unlike openai `timing.py`) shifted every
  word ~one emission late (mean 212–443 ms) and leaked the final word to
  the 30 s padded-window end; fixed together with valid-frame restriction
  and a faithful `merge_punctuations` port — per-word deltas now
  5–50 ms mean vs openai-whisper, sanity 6/6, word counts match.
- **`vokra-server` TTS**: real 8-language G2P is injectable
  (`--piper-g2p`, derived entirely from the voice GGUF metadata) and the
  `/api/tts` router is actually mounted (was 404) — plain-text Japanese
  synthesis returns deterministic audio at ~89 ms median; without the
  flag the explicit-error passthrough contract is unchanged.
- **CLI**: `run --backend cpu|metal` (whisper small/turbo verified Metal
  == CPU byte-identical on real weights), a `speaker` task for CAM++
  GGUFs (embedding + `--compare` cosine), and `convert` help now lists
  all 11 model kinds.

#### CPU performance (M5-14 early wave, 2026-07-18)

An M5 pre-wave rebuilt the CPU hot path with **bit-identical**
guarantees (every output element keeps its exact legacy accumulation
chain; no parity tolerance changed anywhere; Kokoro stays 8/8):

- Packed/blocked GEMM driver (pack kills a 16 KiB power-of-2-stride L1
  aliasing pathology: 19 → 88 GF/s single-thread on the whisper-medium
  fc1 shape), chunked work-queue threading (8T now 4.4× over 1T; the
  old 8T-slower-than-4T inversion is gone), automatic m=1 → GEMV
  routing, TLS conv scratch, vectorized Silero/DFN3 (previously private
  scalar), Mimi batched transformer + codebook-outer RVQ, and a
  per-beam incremental KV cache for whisper beam search
  (beam-1 remains bit-identical to greedy).
- Final quiet-window results vs onnxruntime CPU on the same M1
  (`docs/bench-baselines/m5-14-final-2026-07-18/`): whisper base
  **0.41×** and turbo **0.37×** (i.e. ~2.5–2.7× FASTER than ORT),
  Silero **0.43×**, medium 7.8× slower → **1.17×**, small → 1.24×,
  piper → 2.17×, DFN3 → 1.77×; 9 of 11 explicit targets met (two
  documented near-misses: CAM++ +9 %, beam-5 2.71× vs the 1.6× goal).

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

### Security

- **`vokra-server` Wyoming: unbounded network-controlled allocation**
  (fixed 2026-07-17 together with the data-continuation framing): the
  event reader allocated `vec![0u8; header.data_length]` directly from a
  network-supplied length with no cap — a remote peer could trigger an
  arbitrary-size allocation (latent DoS; the payload path had a cap, the
  data path did not). Both paths are now capped (1 MiB data) and checked
  BEFORE allocation, with explicit protocol errors. The same change makes
  the server speak the wyoming >= 1.2.0 data-continuation framing, which
  every modern (Home Assistant-generation) client requires — previously
  all inference operations failed against real clients; a genuine
  `wyoming` 1.10.0 client now round-trips the canonical JFK transcript
  byte-identically to the HTTP route.

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
