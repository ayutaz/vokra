//! # vokra-ops
//!
//! Speech-specialized operators for the Vokra runtime (SRS §1.3:
//! "音声オペレータ" — the audio operators crate).
//!
//! Operator implementations land with their owning work packages:
//!
//! - **M0-04** (this WP, landed): `stft` / `istft` / `mel_filterbank` /
//!   `mfcc` / `dct` with explicit attributes (window / hop / n_fft / pad /
//!   normalization / causal / `real_input` RFFT — FR-OP-01/03) and the CPU FFT
//!   lowering (a from-scratch Rust reimplementation of the pocketfft algorithm,
//!   BSD-3 — FR-OP-05). See [`fft`], [`window`], [`stft()`], [`istft()`], [`mel`],
//!   [`dct()`], [`mfcc()`] and the [`dispatch()`] bridge to the IR;
//! - **M0-05**: LSTM family needed by the Silero VAD subgraph;
//! - **M0-06**: attention / decoder family needed by Whisper;
//! - **M1-06** (landed): front-end preprocessing — [`resample()`] (a native
//!   Kaiser-windowed-sinc converter, GPL-free by construction) and the
//!   `frontend_spec`-driven [`dc_offset_remove`] / [`pre_emphasis`] chain
//!   ([`apply_frontend`]);
//! - **M1-03** (landed): the [`frontend`] `frontend_spec` → `StftAttrs` /
//!   `MelAttrs` translation ([`stft_attrs_from_spec`] / [`mel_attrs_from_spec`])
//!   — the librosa/torchaudio/TF compat layer that makes the log-mel front-end
//!   data-driven; the bit-exact *inspection* of the chunk lives in `vokra-core`;
//! - **M0-08** (landed): the Kaldi fbank front-end the CAM++ speaker encoder
//!   needs — the [`window::povey`] window, the [`mel`] Kaldi mel-domain ramp
//!   (`MelInterp::Mel`), and [`kaldi_fbank()`] (snip-edges framing, per-frame
//!   DC/pre-emphasis, power spectrum, log, CMN);
//! - later WPs: vocoder chains, flow-matching samplers, codec decode, and
//!   the rest of the audio dialect (CLAUDE.md "音声特化オペレータ").
//!
//! The corresponding [`vokra_core::OpKind`] variants for the M0-04 ops are
//! defined in `vokra-core` (the attribute types embedded in those variants
//! live there because the crate dependency edge runs `vokra-ops → vokra-core`);
//! remaining families are added by their own WPs.
//!
//! # Unsafe policy (NFR-RL-07, SRS §5-(1))
//!
//! `unsafe` + SIMD intrinsics are *permitted inside operator
//! implementations* for RTF, which is why this crate opts out of the
//! workspace-wide `unsafe_code = "deny"` below. Public APIs must stay safe,
//! and every `unsafe` block requires a `// SAFETY:` comment (enforced by
//! `clippy::undocumented_unsafe_blocks` at the workspace level).

// Local opt-out from the workspace `unsafe_code = "deny"` lint — see the
// crate-level "Unsafe policy" docs above (M0-02-T03). The M0-04 ops are
// written in safe Rust; the opt-out is kept for the SIMD kernels of later WPs.
#![allow(unsafe_code)]

// ---- M4-03 aec (FR-OP-60, runtime function — not an OpKind variant) -----
// SpeexDSP MDF/AUMDF float-build port; the time-tagged far-end queue lives
// in vokra-core::stream::aec_ref (crate edge runs ops → core). New module +
// re-export kept as one localized patch block (M3-05/M3-06 pattern) so
// parallel M4 waves rebase cleanly.
pub mod aec;
// -------------------------------------------------------------------------
// ---- M4-20 (c) speech-enhancement subset (FR-OP-61/62/63) ---------------
// agc / hpf / loudness_norm are RUNTIME FUNCTIONS (per-frame state or
// whole-signal transforms), NOT `OpKind` variants (ADR M4-20 §D-5, the
// runtime-function posture of `flow_sampler` / FR-EX-10): first-class in the
// public API, absent from `dispatch.rs` (a graph-side call falls into the
// existing `UnsupportedOp` default, FR-EX-08). `denoise` (FR-OP-61,
// DeepFilterNet MIT) is a network — its forward + GGUF binding live in the
// `denoise` module. Localized patch block for clean parallel-wave rebases.
pub mod agc;
pub mod denoise;
pub mod hpf;
pub mod loudness_norm;
// -------------------------------------------------------------------------
pub mod attrs;
// ---- Vocoder Metal wave — polyphase anti-aliased upsample primitive ------
// The multiply-add core of BigVGAN's `UpSample1d` (`alias_free_activation
// .torch.act`) and every HiFTNet-family alias-free activation chain. Consumes
// a caller-supplied Kaiser-window filter kernel (the design step lives on
// the host — see module docs), so the runtime op signature is narrow (three
// tensor inputs + one scalar ratio), a good fit for a GPU dispatch. Runtime
// function, NOT an OpKind variant (same posture as `snake_activation_f32` /
// `snake_beta_f32` / `sinegen_deterministic_f32` — see the localized re-
// export block below).
pub mod anti_aliased_upsample;
// ---- SoTA plan Phase 3 BigVGAN vocoder (TTS bigvgan_generator primitive) ---
// Anti-aliased periodic-activation vocoder — verbatim port of upstream
// NVIDIA/BigVGAN (MIT, Copyright (c) 2024 NVIDIA CORPORATION). AMPBlock1 +
// Snake/SnakeBeta + tanh terminal — see module docstring for the exact
// upstream line references, including the alias-free activation wrapper.
// Snake activation is reused from [`crate::hiftnet`];
// SnakeBeta lives here.
pub mod bigvgan_generator;
// ---------------------------------------------------------------------------
// ---- pyannote Wave 4 agglomerative hierarchical clustering (runtime
// function, NOT an OpKind variant — same posture as `ctc_decode` /
// `flow_sampler`). Aggregates segment-level speaker embeddings into
// speaker clusters (pyannote MIT, `docs/license-audit.md` §3.1 row 263).
// The full `diarize` op remains M5-residual (HF-gated weight license +
// trigger model blocker); this primitive is the *clustering step alone*
// and lands independently.
pub mod clustering;
// ---------------------------------------------------------------------------
// ---- SoTA plan Phase 2 Conformer / FastConformer ASR encoder ------------
// One implementation covers both — FastConformer differs only in the
// subsampling stem (`ConvSubsampleKind::Stacking { factor }`). Consumed by
// the parakeet family, canary, granite-speech, Qwen3-ASR, and
// reazonspeech-nemo-v2. Verbatim port of the upstream NeMo modules
// (`nemo/collections/asr/modules/conformer_encoder.py` +
// `.../parts/submodules/conformer_modules.py`, MIT).
pub mod conformer;
// -------------------------------------------------------------------------
// ---- SoTA plan Phase 2 ctc_decode (ASR primitive, FR-OP-41) -------------
// CTC greedy blank-fold + prefix beam search with n-gram LM shallow fusion
// and hotword boost. Runtime function, NOT an `OpKind` variant (FR-OP-40 /
// FR-EX-10 posture — same as `beam_search` and `flow_sample`). The reserved
// graph-op identifier lives in `vokra_core::m5_residual_ops::CTC_DECODE_OP`
// (unregistered in the min-dtype registry). Consumed by omniASR-CTC
// (1600 languages, Apache-2.0), parakeet-ctc-1.1b, and the
// jonatasgrosman/wav2vec2 family. Localized re-export block for clean
// parallel-wave rebases.
pub mod ctc_decode;
// -------------------------------------------------------------------------
// ---- M4-04 dac_rvq codec decode (RVQ family, FR-OP-30) ------------------
// DAC's factorized (low-dim codebook + per-quantizer out_proj) residual VQ
// decode. Shapes verified from the upstream descript-audio-codec (MIT)
// implementation + the 24 kHz checkpoint metadata (ADR M4-04 §T02). Paged
// variant primary block size = 4 (75-86 Hz released variants).
pub mod dac_rvq;
// -------------------------------------------------------------------------
pub mod dct;
// ---- SoTA plan Phase 4 ddpm_sampler (TTS primitive, new class) ---------
// DDPM sampler with `v-prediction` support (Salimans & Ho 2022) and a
// cosine β schedule (Nichol & Dhariwal 2021) — the two axes VibeVoice
// (Microsoft, MIT, `huggingface.co/microsoft/VibeVoice-1.5B`) needs and
// the existing `flow_sampler` cannot express (its DDIM/DPM++ branches
// carry `ε`-prediction with a linear α schedule pinned inside the
// solver per ADR M3-05 §D4). Runtime function, NOT an OpKind variant
// (same posture as `flow_sampler` / `mimi_rvq` / `dac_rvq` /
// `qwen3_tts_codec` / `vae_continuous` — FR-OP-30 / FR-EX-10 / ADR
// M3-06 §D-b). Localized re-export block for clean parallel-wave rebases.
pub mod ddpm_sampler;
// -------------------------------------------------------------------------
// ---- M4-04 encodec_rvq (engine op only — FR-OP-32 permanent weight
// exclusion; parity uses synthetic codebooks, never pretrained weights) ----
pub mod encodec_rvq;
// -------------------------------------------------------------------------
pub mod dispatch;
pub mod fft;
// ---- M4-16 FSQ codec family (FR-OP-31, runtime functions — not OpKind
// variants). Single-stage subgraph, deliberately separate from the RVQ
// family (FR-OP-30: mimi_rvq / dac_rvq / encodec_rvq): no cross-codebook
// residual sum, no paged variant, no cross-family adapter. Localized patch
// block (M3-05/M3-06 pattern) for clean parallel-wave rebases.
pub mod fsq_codec;
// -------------------------------------------------------------------------
// ---- M3-05 flow_sampler / ODE solvers (runtime function, FR-EX-10) -----
// New module + re-export block, kept as a single localized patch so Wave 3
// (M3-06 / M3-07) has a clean rebase target. The op-only re-export follows
// the M3-08 length_conditioning and M3-17 prosody pattern.
pub mod flow_sampler;
// -----------------------------------------------------------------------
// ---- FireRedVAD native streaming DFSMN ---------------------------------
// Exact causal DFSMN core for the official FireRedTeam Stream-VAD model.
pub mod firered_vad;
// -----------------------------------------------------------------------
pub mod frontend;
// ---- SoTA plan Phase 5 VAD-2 fsmn_vad primitive ---------------------------
// FSMN-VAD (funasr/fsmn-vad, MIT) — Feed-forward Sequential Memory Network
// for voice activity detection. First-class audio-dialect op (distinct
// posture from Silero VAD v5, which is a 1:1-preserved subgraph per
// FR-LD-06). FSMN's stateless feed-forward + memory blocks lower cleanly to
// graph-level ops. Upstream: iic/speech_fsmn_vad_zh-cn-16k-common-pytorch
// (docs/license-audit.md §3.1 row landed 2026-07-30).
pub mod fsmn_vad;
// ---------------------------------------------------------------------------
pub mod fused_logmel;
// ---- M3-07 hifigan_generator (vocoder chain, FR-OP-10) ------------------
// New module + re-export block. INT8 is an opt-in path (per-channel
// calibration + NFR-QL-02 5% spectral check required); FR-EX-08 is preserved
// at the runtime function (`VokraError::HifiganInt8VerifyMissing` when the
// gate is un-satisfied, `VokraError::UnsupportedOp` while the INT8 kernel
// stays deferred to the M3-09 consumer WP). ADR-equivalent rationale lives in
// the module-level docstring.
pub mod hifigan;
// -------------------------------------------------------------------------
// ---- SoTA plan Phase 1-2 HiFTNet vocoder --------------------------------
// HiFTNet = "Neural Source Filter + ISTFTNet" (upstream CosyVoice2/3 +
// Chatterbox family). This module hosts the F0 predictor (Wave 2) and,
// once Wave 3 lands, the HiFTGenerator chain. The NSF core lives in
// [`crate::nsf`] rather than here so a caller can drive it without pulling
// the full generator.
pub mod hiftnet;
// -------------------------------------------------------------------------
pub mod istft;
pub mod istft_streaming;
pub mod kaldi_fbank;
pub mod length_conditioning;
// Wave 1 2026-08-14 audit follow-up: shallow-fusion n-gram LM for
// CTC / RNN-T / attention decoders (FR-OP-41 / FR-OP-42). Consumed via
// `vokra_core::decode::LmScorer` trait; the fusion arithmetic lives one
// indirection above in `vokra_core::decode::BeamSearchConfig::lm_fusion`.
pub mod lm_fusion;
pub mod mel;
pub mod mfcc;
// Wave 4 2026-08-14 audit follow-up: unified VoiceRef API for TTS
// engines (Kokoro voice_id / piper-plus fixed voice / CosyVoice2
// reference audio) — API surface only; adapter wiring per-engine is
// follow-up.
pub mod voice_ref;
// ---- Vocos Fourier vocoder ------------------------------------------------
// Native ConvNeXt-1D backbone + magnitude/phase iSTFT head. Runtime
// function, not an OpKind variant, matching the hifigan/bigvgan generator
// posture.
pub mod vocos;
// ---------------------------------------------------------------------------
// ---- SoTA plan KWS binder (openwakeword classifier MLP, 2026-08-05) -----
// Per-wake-word `Linear → ReLU → Linear → Sigmoid` classifier over a shared
// 96-d speech embedding. First consumer is `vokra-models::kws::openwakeword`
// (dscripka/openWakeWord, Apache-2.0 code). The embedding extractor itself
// (frozen Google `speech_embedding` TFLite) is a **loud-partial** follow-up
// gated on the owner-provisioned bundle (RMVPE precedent — see
// `crate::denoise` / `vokra_models::f0::rmvpe`). Runtime function, NOT an
// `OpKind` variant (same posture as `flow_sampler` / RVQ / FSQ — ADR M3-06
// §D-b): the per-wake-word bundle shape does not fit the `OpValue`
// dispatch surface. Localised re-export block for clean parallel-wave
// rebases.
pub mod openwakeword;
// -------------------------------------------------------------------------
// ---- M3-06 mimi_rvq codec decode (RVQ family, FR-OP-30) -----------------
// New module + re-export block. Wave 3 (M3-07) will touch the same file, so
// this block is kept localised for a clean rebase target. Mimi is CC-BY 4.0
// (attribution recorded in NOTICE / docs/license-audit.md — ADR M3-06 §D3);
// EnCodec weights (CC-BY-NC 4.0) are permanently excluded from the official
// model zoo (FR-OP-32 — enforced by the M2-13 compliance gate and the
// `scripts/compliance/check-encodec-exclusion.sh` release-side script).
pub mod mimi_rvq;
// -------------------------------------------------------------------------
// ---- SoTA plan Wave C MoE dispatch primitive (2026-08-13) ---------------
// Top-k expert routing + capacity-factor gating for Mixture-of-Experts
// audio-LLMs (`qwen3-omni-30b-a3b-moe`, `zonos2-8b-moe`, plus the
// future Kimi-Audio-A28B family — see
// docs/tickets/coverage-audit-2026-08-03/IMPL-PLAN.md §2.3). Runtime
// function, NOT an `OpKind` variant (same posture as `flow_sampler` /
// `mimi_rvq` — ADR M3-06 §D-b): the heterogeneous inputs (router
// logits, per-expert weight bundles, dispatch plan) do not fit the
// `OpValue` dispatch surface. Localised re-export block for clean
// parallel-wave rebases.
pub mod moe_dispatch;
// -------------------------------------------------------------------------
// ---- SoTA plan Wave C MoE expert GEMM primitive (2026-08-13) ------------
// Per-expert weight reduction that consumes a plan from `moe_dispatch`
// and folds each expert's contribution back into the per-token output.
// Split from `moe_dispatch` so the routing decision is independently
// testable and so a later SIMD / GPU / Metal kernel can replace this
// inner loop without redoing the softmax + top-k math. Runtime
// function, NOT an `OpKind` variant (same posture as `moe_dispatch`).
pub mod moe_expert_gemm;
// -------------------------------------------------------------------------
// ---- SoTA plan Phase 1-2 NSF (HiFTNet source-filter core) ---------------
// Neural Source Filter (SineGen + SourceModuleHnNSF) — verbatim port of the
// upstream CosyVoice implementation (`cosyvoice/hifigan/generator.py` L163-
// 368). Consumed by the HiFTNet vocoder; multiple published models feed the
// same layer (CosyVoice2 / CosyVoice3 / Chatterbox family), so this lives in
// `vokra-ops` rather than a per-model module.
pub mod nsf;
// -------------------------------------------------------------------------
pub mod preprocess;
pub mod prosody;
// ---- SoTA plan Phase 3 qwen3_tts_codec (TTS codec primitive, FR-OP-30) --
// 16-quantizer RVQ codec at 12.5 Hz / 24 kHz — the code → summed codec-
// feature decode step consumed by every released Qwen3-TTS-12Hz voice
// (Qwen/Qwen3-TTS-12Hz-{0.6B,1.7B}-{Base,CustomVoice,VoiceDesign}, Apache-2.0).
// Distinct from Mimi / DAC / EnCodec because quantizer 0 is *semantic*
// (larger vocab: 4096) while quantizers 1..N are acoustic (2048); a single
// `codebook_size` axis à la `MimiRvqAttrs` cannot express the split without
// silently clamping the semantic index (FR-EX-08). Runtime function, not an
// OpKind variant (same posture as mimi_rvq / dac_rvq / encodec_rvq — ADR
// M3-06 §D-b). Localized re-export block for clean parallel-wave rebases.
pub mod qwen3_tts_codec;
// -------------------------------------------------------------------------
pub mod resample;
// ---- SoTA plan denoise Wave A rnnoise primitives (2026-08-05) -----------
// Xiph RNNoise v0.2 primitives (Vorbis window, Bark filterbank, 3-gate GRU
// forward, feature packer, DCT-II) plus the loud-partial pitch_analysis
// stub. No RNNoise runtime binder exists yet: outside this crate the full
// primitive set is exercised only by the env-gated harness
// `crates/vokra-models/tests/parity_rnnoise_v02.rs`, and the sole
// production consumer is `vokra_models::nsnet2`, which reuses
// `rnnoise_gru_forward` alone. Runtime function set,
// NOT `OpKind` variants (same posture as `openwakeword_classifier_forward`
// / `denoise` / `fsmn_vad_forward` — ADR M3-06 §D-b).
pub mod rnnoise;
// -------------------------------------------------------------------------
// ---- SoTA plan Phase 2 rnnt_decode (ASR primitive, FR-OP-42) ------------
// RNN-T / TDT decoding: greedy + beam + TDT (duration head). Consumed by
// parakeet-rnnt-1.1b and parakeet-tdt v2/v3/1.1b (CC-BY-4.0). Ported /
// cross-referenced against NeMo's classical greedy and TDT beam decoders
// (see the module docstring for exact line refs); label-looping (~1500x
// RTFx) is a deferred follow-up. Localized re-export block for clean
// parallel-wave rebases.
pub mod rnnt_decode;
// -------------------------------------------------------------------------
// ---- SoTA plan Phase 3 snac_decode (TTS primitive, RVQ family) ----------
// SNAC (Multi-Scale Neural Audio Codec) 3-stage hierarchical RVQ decode
// (~12 / 23 / 47 Hz per stage for the 24 kHz variant). Reuses the factorized
// `DacOutProj` + `CodebookTable` shapes since SNAC's per-quantizer
// `WNConv1d(codebook_dim, input_dim)` folds identically to DAC's. Consumed
// by Orpheus and Maya1 (upstream `hubertsiuzdak/snac`, MIT / Apache-2.0).
pub mod snac_decode;
// -------------------------------------------------------------------------
// ---- Vocoder Metal wave WF2 snake activation primitive (2026-08-13) -----
// Snake activation (Ziyin et al. 2020; upstream `activations.py`) — the
// per-channel periodic activation `y = x + (1/(α+ε))·sin(α·x)²` used by the
// BigVGAN / HiFTNet / Kokoro-82M vocoder lineage. Exposes a stateless
// out-of-place free function `snake_activation_f32` that the
// `vokra_models::compute::Compute` seam dispatches through (CPU / Metal /
// deferred CUDA), mirroring the silu / gelu / softmax family. The stateful
// [`hiftnet::Snake`] (with optional `alpha_logscale`) and
// [`bigvgan_generator::SnakeBeta`] (two-vector variant) stay as-is and are
// unrelated to this module — this is a narrower, lower-level entry for a
// GPU dispatch.
pub mod snake;
// -------------------------------------------------------------------------
pub mod stft;
// ---- SoTA plan Phase JA JA-ASR-1 waveform_frontend (raw-waveform 7-layer
// strided conv stem, FR-OP-40) — the mel-free ASR input path wav2vec 2.0 /
// HuBERT / k2SSL consume. Runtime function, NOT an `OpKind` variant (same
// posture as [`fsq_codec`] / [`mimi_rvq`] / [`dac_rvq`] — ADR M4-04 §D-b /
// ADR M4-16 §D-b): the heterogeneous inputs (`&[f32]` raw PCM + per-layer
// weight bundles) do not fit the `OpValue` dispatch surface, and the
// planned consumers (jonatasgrosman wav2vec 2.0, reazonspeech k2SSL,
// omniASR-CTC) are imperative models that want the tight function API.
// Localized re-export block for clean parallel-wave rebases.
pub mod waveform_frontend;
// -------------------------------------------------------------------------
// ---- SoTA plan Phase 4 vae_continuous (TTS primitive, new class) --------
// Continuous VAE encoder / decoder scaffold — the first consumer is
// VoxCPM-0.5B (openbmb/VoxCPM, apache-2.0) which pairs a continuous latent
// stream with a diffusion / flow-matching feature generator (unlike every
// existing codec op in this crate, which is discrete RVQ or FSQ). Shared
// across VoxCPM2 and the planned VibeVoice consumer. Runtime function, NOT
// an OpKind variant (same posture as `flow_sampler` / `mimi_rvq` /
// `dac_rvq` / `qwen3_tts_codec` — FR-OP-30 / FR-EX-10 / ADR M3-06 §D-b).
// Localized re-export block for clean parallel-wave rebases.
pub mod vae_continuous;
// -------------------------------------------------------------------------
pub mod window;
// ---- SoTA plan Phase JA JA-ASR-5 Zipformer encoder (multi-resolution) --
// Zipformer = down/up-sample pyramid + attention weight sharing (single QK
// per stack, per-layer V + output projection). k2-fsa/icefall reference
// (Apache-2.0), consumed by the reazonspeech-k2 CTC family. Localised
// re-export block for clean parallel-wave rebases.
pub mod zipformer;
// -------------------------------------------------------------------------
// ---- SoTA plan Phase JA JA-ASR-4 E-Branchformer encoder ----------------
// E-Branchformer = parallel MHA branch + gated cgMLP branch merged via a
// DepthwiseConv + Linear "Merge" module (Kim et al. 2023). ESPnet OWSM
// family reference (CC-BY-4.0). Reuses the Conformer primitive's FF /
// MHA / stem layouts. Localised re-export block for clean parallel-wave
// rebases.
pub mod ebranchformer;
// -------------------------------------------------------------------------
// ---- SoTA plan Phase JA JA-ASR-3 hybrid CTC/attention decoder ----------
// ESPnet-style hybrid rescoring: attention beam extends the prefix, CTC
// gives a prefix score per candidate, LSTM LM optionally shallow-fuses.
// Runtime function (NOT an OpKind variant, same posture as ctc_decode /
// beam_search — FR-OP-40 / FR-EX-10). Localised re-export block for clean
// parallel-wave rebases.
pub mod hybrid_ctc_attention;
// -------------------------------------------------------------------------
// Wave 7 2026-08-14 audit follow-up: classical DSP F0 extractors (YIN —
// de Cheveigné & Kawahara 2002 JASA; PyIN — Mauch & Dixon 2014 ICASSP).
// Weight-free algorithm implementations sitting one level below the
// neural F0 subgraphs in `vokra-models::f0` (RMVPE / FCPE / CREPE);
// together they form the FR-OP-83 unified `f0_extract` API (landing WP
// M5-16 — dormant per owner ADR 2026-07-22 until the M3-17 prosody /
// voice-clone consumer arrives). Runtime functions, NOT `OpKind`
// variants (same posture as `resample` / `agc` / `hpf` — ADR M4-20 §D-5).
// Implemented from the paper spec — no code lifted from aubio (GPL-3.0)
// or librosa (mirroring resample.rs's soxr/rubberband disclaimer,
// NFR-LC-03/04).
pub mod f0;
// -------------------------------------------------------------------------

// ---- Wave A (2026-08-15) WPE dereverberation ----------------------------
// Weighted Prediction Error blind dereverberation — the first *executable*
// dereverberation path in the tree.
//
// Placement vs the existing enhancement arm: DeepFilterNet3 / GTCRN /
// RNNoise / NSNet2 all target additive noise, which is a different
// degradation from late reverberation. `vokra-models::storm` (StoRM,
// arXiv:2312.09386) does cover dereverberation, but it is a neural model
// whose `enhance()` is currently a loud-partial (pending an NCSN++ v2 U-Net
// score network + OUVE-SDE sampler) and whose §3.1 sign-off row is blank
// and fail-closed, so it cannot run today. WPE is complementary rather than
// redundant.
//
// Transcribed from the algorithm in `github.com/fgnt/nara_wpe` (MIT). Being
// pure DSP it carries no weights, so unlike every neural entry on the
// enhancement arm there is no checkpoint, no license class and no §3.1 row
// to gate on — it is available unconditionally, like `resample`. Runtime
// functions, NOT `OpKind` variants (same posture as `resample` / `agc` /
// `hpf` — ADR M4-20 §D-5).
pub mod wpe;
// -------------------------------------------------------------------------

// ---- Wave E (2026-08-15) ViT audio encoder ------------------------------
// 2-D patch embedding over a mel plane + plain pre-norm Transformer encoder.
//
// This is the ONE primitive the whole SSL audio-embedding fleet was missing.
// The `atst` / `eat` / `m2d` / `maest` binders (and the `dasheng` / `beats` /
// `ast` converters behind them) each loud-partialled for the same reason:
// `conformer` / `ebranchformer` / `zipformer` are conv-augmented ASR encoders
// over a 1-D frame sequence, which is a different architecture, not a
// substitute for a patch-grid ViT.
//
// Pre-norm, not post-norm — getting that backwards is silently wrong rather
// than loud, so it is stated here as well as in the module docs. Every axis
// is caller-supplied: five models use this with five different axis sets, so
// there is deliberately no "upstream default" to invent.
//
// Runtime functions, NOT `OpKind` variants (same posture as `resample` /
// `agc` / `hpf` / `wpe` — ADR M4-20 §D-5).
pub mod vit;
// -------------------------------------------------------------------------

// ---- Wave D (2026-08-15) inverse text normalization / text normalization -
// `itn` — the WeTextProcessing (Apache-2.0) two-stage tagger -> verbalizer
// pipeline over compiled OpenFST grammars.
//
// A brand-new category: every ASR model here emits normalized, unpunctuated
// text ("one hundred fourteen thousand five"); production transcripts need
// "114005". This is the missing back half.
//
// It is mostly REUSE, not new machinery: `vokra_core::decode::wfst` (M5-06)
// already ships the tropical semiring, the `Fst` type, structural validation
// and a byte-verified `read_openfst_vector` for exactly the
// `VectorFst<StdArc>` binary the upstream C++ runtime reads with
// `StdVectorFst::Read`. The only new algorithm is composing a LINEAR input
// string with a transducer and taking the best path — which does not need
// the general `compose` that ADR M5-06 §1 deliberately omits.
//
// Feature-gated in HALVES: `vokra-ops/vokra-wfst` forwards to
// `vokra-core/vokra-wfst` and gates only the two FST compositions. The
// tagged-token parser, the per-language field-order tables, the `Reorder`
// rewrite, the grammar container and the OpenFST header probe are all in the
// DEFAULT build, so `cargo test --workspace` actually exercises them.
//
// Runtime functions, NOT `OpKind` variants (same posture as `resample` /
// `agc` / `hpf` / `wpe` / `vit` — ADR M4-20 §D-5).
pub mod itn;
// -------------------------------------------------------------------------

// ---- M4-03 aec re-exports ------------------------------------------------
pub use aec::{Aec, AecAttrs, AecStatus};
// ---------------------------------------------------------------------------
// ---- SoTA plan Phase 3 bigvgan_generator re-exports ---------------------
pub use bigvgan_generator::{
    AmpBlock1, AmpBlock1Weights, BigVGanConfig, BigVGanGenerator, BigVGanWeights, SnakeBeta,
    SnakeKind,
};
// -------------------------------------------------------------------------
// ---- pyannote Wave 4 clustering re-exports ------------------------------
pub use clustering::{AgglomerativeClustering, DistanceMetric, LinkageMethod};
// -------------------------------------------------------------------------
// ---- SoTA plan Phase 2 ctc_decode re-exports ----------------------------
pub use ctc_decode::{CtcBeamAttrs, CtcHypothesis, ctc_decode_beam, ctc_decode_greedy};
// -------------------------------------------------------------------------
// ---- M4-04 dac_rvq re-exports --------------------------------------------
pub use dac_rvq::{
    DacOutProj, DacRvqAttrs, dac_paged_dims, dac_rvq_decode, dac_rvq_decode_paged,
    dac_rvq_read_summed, dac_rvq_read_summed_range,
};
// ---------------------------------------------------------------------------
// ---- M4-20 (c) speech-enhancement re-exports ----------------------------
pub use agc::{AgcAttrs, AgcState, agc};
pub use denoise::{
    DeepFilterNetConfig, DenoiseModel, DenoiseTaps, TensorSpec, denoise, denoise_apply_mask_f32,
    denoise_skipped_checkpoint_tensors, denoise_synthesized_tensors, denoise_tensor_manifest,
};
pub use hpf::{HpfAttrs, HpfState, hpf};
pub use loudness_norm::{LoudnessNormAttrs, integrated_lufs, loudness_norm};
// -------------------------------------------------------------------------
pub use dct::dct;
// ---- M4-04 encodec_rvq re-exports -----------------------------------------
pub use encodec_rvq::{EncodecRvqAttrs, encodec_rvq_decode};
// ---------------------------------------------------------------------------
pub use dispatch::{OpValue, dispatch};
// ---- M3-05 flow_sampler re-exports --------------------------------------
pub use flow_sampler::{
    CfgMode, CfgScaleProfile, FlowSamplerConfig, FlowSamplerState, ForwardPass, OdeSolver,
    Schedule, flow_sample,
};
// -------------------------------------------------------------------------
// ---- SoTA plan Phase 4 ddpm_sampler re-exports --------------------------
pub use ddpm_sampler::{
    BetaSchedule, DdpmSamplerConfig, PredictionType, build_alphas_cumprod, ddpm_sample,
    pick_inference_timesteps,
};
// -------------------------------------------------------------------------
// ---- M4-16 fsq_codec re-exports ------------------------------------------
pub use fsq_codec::{
    FsqOutProj, WavTokenizerVqAttrs, Xcodec2FsqAttrs, fsq_index_to_grid_codes,
    wavtokenizer_vq_decode, xcodec2_fsq_decode,
};
// ---------------------------------------------------------------------------
pub use firered_vad::{
    FireredVadDfsmnBlockWeights, FireredVadDfsmnConfig, FireredVadDfsmnState,
    FireredVadDfsmnWeights, firered_vad_dfsmn_forward,
};
pub use frontend::{mel_attrs_from_spec, stft_attrs_from_spec};
// ---- SoTA plan Phase 5 VAD-2 fsmn_vad re-exports --------------------------
pub use fsmn_vad::{
    FsmnBlockWeights, FsmnEncoderConfig, FsmnStreamState, FsmnVadWeights, fsmn_vad_forward,
    softmax_last_axis,
};
// ---------------------------------------------------------------------------
pub use fused_logmel::fused_log_mel_scalar;
// ---- M3-07 hifigan_generator re-exports ---------------------------------
pub use hifigan::{
    CalibrationStrategy, CalibrationTable, GinCondition, HifiGanCalibrator, HifiGanConfig,
    HifiGanPrecision, HifiGanSpectralChecker, HifiGanWeights, MrfBranchWeights, ResBlockLayer,
    SPECTRAL_CHECK_THRESHOLD, SpectralCheckResult, UpsampleStageWeights, hifigan_generator,
    hifigan_generator_conditioned,
};
// -------------------------------------------------------------------------
pub use istft::istft;
pub use istft_streaming::{IstftStreamingState, istft_streaming_oneshot};
pub use kaldi_fbank::{KaldiFbankOpts, kaldi_fbank};
pub use length_conditioning::length_conditioning;
pub use mel::mel_filterbank;
pub use mfcc::mfcc;
pub use vocos::{
    VocosAttrs, VocosBlockWeights, VocosIstftPadding, VocosNormWeights, VocosWeights, vocos_decode,
};
// ---- SoTA plan KWS binder openwakeword re-exports (2026-08-05) ----------
pub use openwakeword::{
    OpenwakewordClassifierWeights, OpenwakewordConv2dWeights, OpenwakewordDenseWeights,
    OpenwakewordDnnClassifierWeights, OpenwakewordEmbeddingWeights, OpenwakewordMelspecWeights,
    openwakeword_classifier_forward, openwakeword_dnn_classifier_forward,
    openwakeword_embedding_forward, openwakeword_melspectrogram,
};
// -------------------------------------------------------------------------
// ---- M3-06 mimi_rvq re-exports ------------------------------------------
pub use mimi_rvq::{
    CodebookTable, MimiDecoder, MimiRvqAttrs, codebook_lookup, mimi_paged_dims, mimi_rvq_decode,
    mimi_rvq_decode_paged, mimi_rvq_read_summed, mimi_rvq_read_summed_range,
};
// -------------------------------------------------------------------------
// ---- SoTA plan Wave C MoE dispatch primitive (2026-08-13) ---------------
pub use moe_dispatch::{MoeAssignment, MoeDispatchAttrs, MoeDispatchPlan, moe_dispatch};
// -------------------------------------------------------------------------
// ---- SoTA plan Wave C MoE expert GEMM primitive (2026-08-13) ------------
pub use moe_expert_gemm::{MoeExpertWeights, moe_expert_gemm};
// -------------------------------------------------------------------------
pub use preprocess::{apply_frontend, dc_offset_remove, pre_emphasis};
pub use prosody::{ApplyProsody, ProsodyControl};
// ---- SoTA plan Phase 3 qwen3_tts_codec re-exports -----------------------
pub use qwen3_tts_codec::{Qwen3TtsCodec, Qwen3TtsCodecConfig, qwen3_tts_codec_decode};
// -------------------------------------------------------------------------
pub use resample::resample;
// ---- SoTA plan denoise Wave A rnnoise re-exports (2026-08-05) -----------
pub use rnnoise::{
    Activation as RnnoiseActivation, BARK_BAND_EDGES, FRAME_HOP, FRAME_SIZE, N_BARK_BANDS,
    N_FEATURES, N_PITCH_BANDS, N_STFT_BINS, PITCH_BUF_SIZE, PitchState, bark_dct, bark_filterbank,
    dense_forward as rnnoise_dense_forward, gru_forward as rnnoise_gru_forward, interp_bark_gains,
    pack_features as rnnoise_pack_features, pitch_analysis, vorbis_window as rnnoise_vorbis_window,
    zero_pitch_features as rnnoise_zero_pitch_features,
};
// -------------------------------------------------------------------------
// ---- SoTA plan Phase 2 rnnt_decode re-exports ---------------------------
pub use rnnt_decode::{RnntAttrs, RnntDecoderKind, RnntHypothesis, rnnt_decode};
// -------------------------------------------------------------------------
// ---- SoTA plan Phase 3 snac_decode re-exports ---------------------------
pub use snac_decode::{SnacConfig, SnacDecoder, SnacWeights};
// -------------------------------------------------------------------------
// ---- Vocoder Metal wave WF2 snake activation re-exports (2026-08-13) ----
pub use snake::snake_activation_f32;
// ---- Vocoder Metal wave — common vocoder primitives re-exports ----------
// Stateless out-of-place free functions that mirror the shape the
// `vokra_models::compute::Compute` seam dispatches through (mirroring the
// silu / gelu / softmax family — read inputs, write out). Consumed by the
// BigVGAN / HiFTNet / Kokoro-82M vocoder lineage. Each has its own module
// with rationale + upstream refs.
pub use anti_aliased_upsample::anti_aliased_upsample_f32;
pub use nsf::sinegen_deterministic_f32;
pub use snake::snake_beta_f32;
// -------------------------------------------------------------------------
pub use stft::{Spectrogram, stft};
// ---- SoTA plan Phase JA JA-ASR-1 waveform_frontend re-exports -----------
pub use waveform_frontend::{
    ConvLayerAttrs, ConvLayerWeights, Norm, WaveformFrontendAttrs, WaveformFrontendWeights,
    waveform_frontend,
};
// -------------------------------------------------------------------------
// ---- SoTA plan Phase 4 vae_continuous re-exports ------------------------
pub use vae_continuous::{
    ContinuousVaeConfig, ContinuousVaeDecoder, ContinuousVaeDecoderWeights, ContinuousVaeEncoder,
    ContinuousVaeEncoderWeights, continuous_vae_decode, continuous_vae_encode,
};
// -------------------------------------------------------------------------
// ---- Wave 4 2026-08-14 audit follow-up voice_ref re-exports -------------
pub use voice_ref::{VoiceRef, VoiceRefSource};
// -------------------------------------------------------------------------
// ---- Wave 7 2026-08-14 audit follow-up classical DSP F0 re-exports ------
// YIN / PyIN weight-free extractors. See the `pub mod f0` block above for
// placement / red-line rationale.
pub use f0::{PyinFrame, pyin, pyin_detailed, yin};
// -------------------------------------------------------------------------
// ---- Wave A (2026-08-15) WPE dereverberation re-exports -----------------
// See the `pub mod wpe` block above for placement / licence rationale.
// Localised re-export block for clean parallel-wave rebases.
//
// The upstream-default constants (DEFAULT_TAPS / DEFAULT_DELAY / ...) are
// deliberately NOT hoisted to the crate root: those names are generic enough
// to collide with a future op's defaults. Reach them as `wpe::DEFAULT_TAPS`.
pub use wpe::{StatisticsMode, WpeAttrs, wpe, wpe_dereverb_pcm, wpe_dereverb_pcm_multi, wpe_mono};
// -------------------------------------------------------------------------
pub use vokra_core::Complex32;

#[cfg(test)]
mod tests {
    #[test]
    fn links_against_vokra_core() {
        // Smoke test for the crate wiring (M0-02-T02): vokra-ops builds on
        // the vokra-core IR types.
        let dtype = vokra_core::DType::F32;
        assert_eq!(dtype.size_in_bytes(), 4);
    }
}
