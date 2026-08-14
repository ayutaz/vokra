//! Kyutai **STT-2.6B-EN** — English streaming ASR (SoTA plan Phase 2,
//! 2026-07-24).
//!
//! # What Kyutai STT is (primary source)
//!
//! Kyutai STT is a **decoder-only transformer** that consumes audio tokenized
//! by the **Mimi** codec and emits text tokens. It is the "delayed streams
//! modeling" family from `kyutai-labs/delayed-streams-modeling` — the same
//! Helium-style backbone Moshi ships (M4-06), specialised for one-way ASR:
//! the model itself generates only text, so the depformer is present in the
//! upstream config for structural symmetry but the "own audio" prediction
//! count (`dep_q`) is `0`.
//!
//! Every hparam below is transcribed **verbatim** from the upstream
//! `huggingface.co/kyutai/stt-2.6b-en/raw/main/config.json` (CLAUDE.md
//! ハルシネーション厳禁; primary source recorded per fetch on 2026-07-24):
//!
//! - **Backbone** (`model_type: "stt"`): `dim=2048`, `num_layers=48`,
//!   `num_heads=32`, `hidden_scale=4.125`, `positional_embedding="rope"`,
//!   `max_period=100000`, `norm="rms_norm_f32"`, `gating="silu"`,
//!   `causal=true`, `context=375`, `layer_scale=null`.
//! - **Depformer** (structurally present, unused for audio when `dep_q=0`):
//!   `depformer_dim=1024`, `depformer_num_layers=6`,
//!   `depformer_num_heads=16`, `depformer_dim_feedforward=null`,
//!   `depformer_multi_linear=true`, `depformer_pos_emb="none"`,
//!   `depformer_weights_per_step=true`.
//! - **Audio input** (Mimi RVQ): `n_q=32` quantizers, `card=2048` codebook
//!   size, `delays=[0]*33` (text + 32 audio channels, all synchronous —
//!   the 2.5 s "audio_delay_seconds" is a *streaming* delay applied at
//!   session level, not a per-channel token shift).
//! - **Text**: `text_card=4000`, `existing_text_padding_id=3`.
//! - **Streaming**: `stt_config.audio_delay_seconds=2.5`,
//!   `stt_config.audio_silence_prefix_seconds=1.0`.
//! - **Codec side-car**: `mimi_name="mimi-pytorch-e351c8d8@125.safetensors"`
//!   (**24 kHz / 12.5 Hz** — the Mimi sample-rate / frame-rate live in
//!   `vokra.mimi.*`, ADR M4-06 §D3; the STT chunk group deliberately does
//!   *not* duplicate them).
//! - **Tokenizer side-car**: `tokenizer_name="tokenizer_en_audio_4000.model"`
//!   (raw SentencePiece; the T29-equivalent owner hand-off embeds it into
//!   `vokra.tokenizer.model` — the Moshi / CSM pattern).
//! - **Weight license**: **CC-BY 4.0** (`AttributionRequired`) — the
//!   converter stamps the FR-MD-09 attribution text; the compliance
//!   registry maps `kyutai-stt` / `kyutai-stt-2.6b-en` to
//!   [`vokra_core::LicenseClass::AttributionRequired`] so the M2-13 gate
//!   passes commercially *and* the FR-MD-09 attribution surface activates.
//!
//! # Boundary — Mimi consumed, never re-implemented
//!
//! Kyutai STT consumes Mimi audio tokens directly (`n_q=32` codes per
//! 12.5 Hz frame). Vokra's shared Mimi op lives in
//! [`vokra_ops::mimi_rvq`] (M3-06 / M4-04) — this module never duplicates
//! it. The two boundaries stay independent Apache 2.0 (Moshi code) + CC-BY
//! 4.0 (Mimi weights) provenance chains and the caller pairs the STT GGUF
//! with any 24 kHz Mimi codec GGUF.
//!
//! # What lands in this Phase 2 slice
//!
//! - [`KyutaiSttConfig`] — every hparam transcribed from the primary
//!   source (no hardcoded fabrication; sample-rate is inherited from Mimi
//!   24 kHz per upstream `mimi_name`, documented on the field).
//! - [`KyutaiSttWeights`] — a backbone weight store with a deterministic
//!   [`KyutaiSttWeights::synthesized`] fixture (SplitMix64 + Xavier) so
//!   shape / dtype / size flow can be exercised without the real HF
//!   checkpoint.
//! - [`KyutaiSttAsr`] — engine handle carrying config + weights.
//!   [`KyutaiSttAsr::transcribe`] returns [`VokraError::NotImplemented`]
//!   until real weights are bound (the real forward — audio-token embedding
//!   sum → per-layer prenorm MHA + gating FFN → sliding-window causal
//!   attention → text logits → sampling → SentencePiece detokenize — is a
//!   follow-up wave gated on the real-checkpoint tensor manifest).
//!
//! Real-checkpoint parity is deferred exactly like CosyVoice2 T02 / CSM T29
//! / Moshi T29: this scaffold sets the seam so the follow-up lands drop-in.

use vokra_core::gguf::{GgufFile, GgufMetadataValue, chunks};
use vokra_core::rng::SplitMix64;
use vokra_core::{CompliancePolicy, Result, VokraError, check_weight_license};

/// `vokra.model.arch` a Kyutai STT GGUF must carry. Written by
/// `vokra-convert::models::kyutai_stt::ARCH`; the compliance registry
/// (`vokra_core::compliance`) knows `kyutai-stt` / `kyutai-stt-2.6b-en` as
/// [`vokra_core::LicenseClass::AttributionRequired`] (CC-BY 4.0 — the M2-13
/// gate passes commercially *and* the FR-MD-09 attribution surface
/// activates).
pub const EXPECTED_ARCH: &str = "kyutai-stt";

/// PCM sample rate Kyutai STT expects at the Mimi boundary. Not written in
/// the upstream `config.json`; inherited from Mimi (the codec the config's
/// `mimi_name` names — `mimi-pytorch-e351c8d8@125`, 24 kHz / 12.5 Hz per
/// the shared Mimi module docs, ADR M4-06 §D3).
pub const KYUTAI_STT_SAMPLE_RATE: u32 = 24_000;

/// Deterministic seed [`KyutaiSttAsr::from_gguf_with_policy`] threads into
/// [`KyutaiSttWeights::synthesized`] until the real-checkpoint tensor-name
/// manifest lands (T29-equivalent — the CSM
/// [`CSM_FROM_GGUF_DEFAULT_SEED`](super::csm::CSM_FROM_GGUF_DEFAULT_SEED)
/// pattern). Fixed so every `from_gguf` build against the same shape
/// config produces bit-identical weight bytes → reproducible bug reports.
pub const KYUTAI_STT_FROM_GGUF_DEFAULT_SEED: u64 = 0x0C57_0C57_0C57_0C57;

// ---------------------------------------------------------------------------
// `vokra.kyutai_stt.*` metadata keys
// ---------------------------------------------------------------------------
//
// These strings mirror the offline converter (`vokra-convert::models::kyutai_stt`)
// verbatim; the two crates only share `vokra-core`, so the string
// constants are the sole handshake (the cross-crate pattern established
// by CSM / CosyVoice2 / Kokoro / Dia / Zonos — see this module docstring
// and the CSM `config.rs` for the same layout).

const KEY_SAMPLE_RATE: &str = "vokra.kyutai_stt.sample_rate";

// Backbone
const KEY_BB_N_LAYER: &str = "vokra.kyutai_stt.arch.backbone.n_layer";
const KEY_BB_D_MODEL: &str = "vokra.kyutai_stt.arch.backbone.d_model";
const KEY_BB_N_HEAD: &str = "vokra.kyutai_stt.arch.backbone.n_head";
const KEY_BB_HIDDEN_SCALE: &str = "vokra.kyutai_stt.arch.backbone.hidden_scale";
const KEY_BB_FFN_HIDDEN: &str = "vokra.kyutai_stt.arch.backbone.ffn_hidden";
const KEY_BB_CONTEXT: &str = "vokra.kyutai_stt.arch.backbone.context";
const KEY_BB_ROPE_MAX_PERIOD: &str = "vokra.kyutai_stt.arch.backbone.rope_max_period";
const KEY_BB_CAUSAL: &str = "vokra.kyutai_stt.arch.backbone.causal";
const KEY_BB_RMS_NORM_EPS: &str = "vokra.kyutai_stt.arch.backbone.rms_norm_eps";

// Depformer (structurally present, unused for audio when dep_q=0)
const KEY_DEP_N_LAYER: &str = "vokra.kyutai_stt.arch.depformer.n_layer";
const KEY_DEP_D_MODEL: &str = "vokra.kyutai_stt.arch.depformer.d_model";
const KEY_DEP_N_HEAD: &str = "vokra.kyutai_stt.arch.depformer.n_head";
const KEY_DEP_MULTI_LINEAR: &str = "vokra.kyutai_stt.arch.depformer.multi_linear";
const KEY_DEP_WEIGHTS_PER_STEP: &str = "vokra.kyutai_stt.arch.depformer.weights_per_step";

// Audio / text / streaming
const KEY_N_Q: &str = "vokra.kyutai_stt.audio.n_q";
const KEY_DEP_Q: &str = "vokra.kyutai_stt.audio.dep_q";
const KEY_AUDIO_CARD: &str = "vokra.kyutai_stt.audio.card";
const KEY_TEXT_CARD: &str = "vokra.kyutai_stt.text.card";
const KEY_TEXT_PAD_ID: &str = "vokra.kyutai_stt.text.pad_id";
const KEY_AUDIO_DELAY_SECS: &str = "vokra.kyutai_stt.stream.audio_delay_seconds";
const KEY_AUDIO_SILENCE_PREFIX_SECS: &str = "vokra.kyutai_stt.stream.audio_silence_prefix_seconds";

// Delays (indexed keys — the CSM / Moshi / Dia pattern for array metadata)
const KEY_N_DELAYS: &str = "vokra.kyutai_stt.n_delays";
const PREFIX_DELAY: &str = "vokra.kyutai_stt.delay.";

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Backbone hparams (primary source: `config.json` — every field is a
/// verbatim transcription).
///
/// The backbone is a stack of pre-norm MHA + SiLU-gating FFN blocks with
/// interleaved-pair RoPE and sliding-window causal attention. `d_model`
/// is the residual width; the per-head width is `d_model / num_heads`.
#[derive(Debug, Clone, PartialEq)]
pub struct KyutaiSttBackboneConfig {
    /// `num_layers` — 48 for STT-2.6B-EN.
    pub n_layer: usize,
    /// `dim` — hidden width, 2048.
    pub d_model: usize,
    /// `num_heads` — MHA (query = key = value heads), 32.
    pub n_head: usize,
    /// `hidden_scale` — the gating FFN inner-width multiplier (4.125).
    /// The runtime derives `ffn_hidden` from this + `d_model`; the
    /// converter mirrors the derivation so the GGUF carries the resolved
    /// value directly.
    pub hidden_scale: f32,
    /// `context` — sliding attention window in frame positions (375).
    pub context: usize,
    /// `max_period` — RoPE max period (100000).
    pub rope_max_period: f32,
}

impl KyutaiSttBackboneConfig {
    /// Per-head width (`d_model / n_head`); `0` when `n_head == 0`
    /// (shape-only converter sentinel) so shape checks never panic.
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.d_model.checked_div(self.n_head).unwrap_or(0)
    }

    /// MHA algebraic constraint: heads divide the width, all non-zero.
    #[must_use]
    pub fn is_well_formed(&self) -> bool {
        self.n_head != 0 && self.d_model != 0 && self.d_model % self.n_head == 0
    }

    /// Gating FFN hidden width — `round(hidden_scale * d_model)`.
    ///
    /// The upstream Kyutai `config.json` records the multiplier
    /// (`hidden_scale`), not the resolved width. For STT-2.6B-EN this is
    /// `round(4.125 * 2048) = 8448`. Real-weight binding cross-checks the
    /// resolved value against the checkpoint's `linear_in` / `linear_out`
    /// tensor shapes and fails loudly on a mismatch (FR-EX-08).
    #[must_use]
    pub fn ffn_hidden(&self) -> usize {
        // `.round()` matches Python's default rounding for the STT-2.6B
        // case; a checkpoint whose shapes disagree with the derivation
        // surfaces at the `KyutaiSttAsr::new` shape gate.
        let scaled = self.hidden_scale * self.d_model as f32;
        if scaled.is_finite() && scaled >= 0.0 {
            scaled.round() as usize
        } else {
            0
        }
    }
}

/// Depformer hparams (structurally present per the upstream
/// `config.json` — the same Helium-style depth transformer Moshi ships
/// M4-06). STT sets `dep_q=0` so the depformer's per-step weights are
/// unused for audio prediction; the fields are captured verbatim from the
/// primary source for the audit trail (a future variant that predicts
/// audio would consume them). No depformer weights ride in the scaffold's
/// [`KyutaiSttWeights`] until a `dep_q > 0` variant lands.
#[derive(Debug, Clone, PartialEq)]
pub struct KyutaiSttDepformerConfig {
    /// `depformer_num_layers` — 6.
    pub n_layer: usize,
    /// `depformer_dim` — 1024.
    pub d_model: usize,
    /// `depformer_num_heads` — 16.
    pub n_head: usize,
    /// `depformer_multi_linear` — one linear-in per codebook step (true).
    pub multi_linear: bool,
    /// `depformer_weights_per_step` — one weight set per codebook step
    /// (true; combined with `dep_q=0` means the resolved set count is 0).
    pub weights_per_step: bool,
}

/// Resolved Kyutai STT hparam snapshot — every field is transcribed from
/// the upstream `config.json` (module docstring) or from the Mimi codec
/// STT depends on (`sample_rate`).
#[derive(Debug, Clone, PartialEq)]
pub struct KyutaiSttConfig {
    /// Backbone hparams.
    pub backbone: KyutaiSttBackboneConfig,
    /// Depformer hparams (structurally present, unused when `dep_q=0`).
    pub depformer: KyutaiSttDepformerConfig,
    /// `n_q` — audio codebooks per Mimi frame (32).
    pub n_q: usize,
    /// `dep_q` — codebooks the depformer would generate (0 for STT —
    /// text-only prediction).
    pub dep_q: usize,
    /// `card` — per-codebook audio vocab (2048; the Mimi codebook size).
    pub audio_card: usize,
    /// `text_card` — text vocab (4000; the SentencePiece side-car has 4000
    /// tokens).
    pub text_card: usize,
    /// `existing_text_padding_id` — 3.
    pub text_pad_id: u32,
    /// `causal` — attention causality (true — STT is left-to-right).
    pub causal: bool,
    /// `norm == "rms_norm_f32"` → RMSNorm ε (1e-8, the upstream default —
    /// mirrors Moshi `create_norm_fn`).
    pub rms_norm_eps: f32,
    /// Per-channel delays (`len == n_q + 1`), index 0 = text, 1..=n_q =
    /// audio (`delays: [0, 0, …]` for STT — all synchronous). The 2.5 s
    /// streaming delay applies at session level, not per-channel.
    pub delays: Vec<u32>,
    /// `stt_config.audio_delay_seconds` (2.5) — how far the text stream
    /// lags the audio stream at inference time.
    pub audio_delay_seconds: f32,
    /// `stt_config.audio_silence_prefix_seconds` (1.0) — the silence
    /// prefix the session prepends before decoding the first token.
    pub audio_silence_prefix_seconds: f32,
    /// PCM sample rate Kyutai STT expects at the Mimi boundary — 24_000
    /// (inherited from Mimi; **not** written in the upstream
    /// `config.json`).
    pub sample_rate: u32,
}

impl KyutaiSttConfig {
    /// Primary-source Kyutai STT-2.6B-EN config (every value transcribed
    /// from `huggingface.co/kyutai/stt-2.6b-en/raw/main/config.json`).
    #[must_use]
    pub fn stt_2_6b_en() -> Self {
        Self {
            backbone: KyutaiSttBackboneConfig {
                n_layer: 48,
                d_model: 2048,
                n_head: 32,
                hidden_scale: 4.125,
                context: 375,
                rope_max_period: 100_000.0,
            },
            depformer: KyutaiSttDepformerConfig {
                n_layer: 6,
                d_model: 1024,
                n_head: 16,
                multi_linear: true,
                weights_per_step: true,
            },
            n_q: 32,
            dep_q: 0,
            audio_card: 2048,
            text_card: 4000,
            text_pad_id: 3,
            causal: true,
            // `norm: "rms_norm_f32"` upstream — ε = 1e-8 (Moshi
            // `create_norm_fn`; see this module docstring for the
            // primary-source reference).
            rms_norm_eps: 1e-8,
            // 33 channels (text + 32 audio), all synchronous per the
            // upstream config.
            delays: vec![0; 33],
            audio_delay_seconds: 2.5,
            audio_silence_prefix_seconds: 1.0,
            sample_rate: KYUTAI_STT_SAMPLE_RATE,
        }
    }

    /// Miniature well-formed config for shape / stability tests. Dims are
    /// tiny so synthesized-weight builds fit in KB; the *shape
    /// relationships* (n_q + 1 delays, MHA well-formed head split, even
    /// head_dim for RoPE pairs) mirror the real model.
    #[must_use]
    pub fn tiny_for_tests() -> Self {
        Self {
            backbone: KyutaiSttBackboneConfig {
                n_layer: 2,
                d_model: 16,
                n_head: 4,
                hidden_scale: 4.0,
                context: 32,
                rope_max_period: 100_000.0,
            },
            depformer: KyutaiSttDepformerConfig {
                n_layer: 2,
                d_model: 8,
                n_head: 2,
                multi_linear: true,
                weights_per_step: true,
            },
            n_q: 4,
            dep_q: 0,
            audio_card: 8,
            text_card: 12,
            text_pad_id: 3,
            causal: true,
            rms_norm_eps: 1e-8,
            // 5 channels (text + 4 audio), all zero delays.
            delays: vec![0; 5],
            audio_delay_seconds: 0.5,
            audio_silence_prefix_seconds: 0.25,
            sample_rate: KYUTAI_STT_SAMPLE_RATE,
        }
    }

    /// Total token channels the backbone sees per step (`text +
    /// n_q_audio`).
    #[must_use]
    pub fn n_channels(&self) -> usize {
        self.n_q + 1
    }

    /// The largest per-channel delay (STT is all-zero — kept for parity
    /// with the Moshi arithmetic).
    #[must_use]
    pub fn max_delay(&self) -> u32 {
        self.delays.iter().copied().max().unwrap_or(0)
    }

    /// Rejects `0`-placeholder / ill-formed configs before any forward
    /// runs (FR-EX-08 — a shape-only converter path fails loudly here,
    /// not deep inside a GEMM).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] naming the offending field.
    pub fn validate_for_forward(&self) -> Result<()> {
        if !self.backbone.is_well_formed() {
            return Err(VokraError::InvalidArgument(format!(
                "kyutai-stt config: backbone ill-formed (n_layer={}, d_model={}, \
                 n_head={}) — expected d_model % n_head == 0, all fields > 0",
                self.backbone.n_layer, self.backbone.d_model, self.backbone.n_head,
            )));
        }
        if self.backbone.n_layer == 0 {
            return Err(VokraError::InvalidArgument(
                "kyutai-stt config: backbone.n_layer must be > 0".to_owned(),
            ));
        }
        if self.backbone.head_dim() % 2 != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "kyutai-stt config: backbone head_dim {} must be even (RoPE pairs)",
                self.backbone.head_dim(),
            )));
        }
        if self.backbone.ffn_hidden() == 0 {
            return Err(VokraError::InvalidArgument(format!(
                "kyutai-stt config: ffn_hidden derived to 0 (hidden_scale={} × \
                 d_model={}) — non-finite or non-positive scale",
                self.backbone.hidden_scale, self.backbone.d_model,
            )));
        }
        if self.backbone.context == 0 {
            return Err(VokraError::InvalidArgument(
                "kyutai-stt config: backbone.context must be > 0 (no forward \
                 can bound its sliding-window attention)"
                    .to_owned(),
            ));
        }
        if self.n_q == 0 {
            return Err(VokraError::InvalidArgument(
                "kyutai-stt config: n_q must be > 0 (no audio input channels)".to_owned(),
            ));
        }
        if self.dep_q > self.n_q {
            return Err(VokraError::InvalidArgument(format!(
                "kyutai-stt config: dep_q={} exceeds n_q={} — own streams are \
                 a subset of the audio channels (STT sets dep_q=0)",
                self.dep_q, self.n_q,
            )));
        }
        if self.audio_card == 0 || self.text_card == 0 {
            return Err(VokraError::InvalidArgument(format!(
                "kyutai-stt config: zero-size vocab (audio_card={}, text_card={})",
                self.audio_card, self.text_card,
            )));
        }
        if self.delays.len() != self.n_channels() {
            return Err(VokraError::InvalidArgument(format!(
                "kyutai-stt config: {} delays for {} channels (text + n_q — \
                 `_lm_kwargs[\"delays\"]` is per-channel)",
                self.delays.len(),
                self.n_channels(),
            )));
        }
        if (self.text_pad_id as usize) >= self.text_card {
            return Err(VokraError::InvalidArgument(format!(
                "kyutai-stt config: text_pad_id={} must be < text_card={}",
                self.text_pad_id, self.text_card,
            )));
        }
        Ok(())
    }

    /// Reads the Kyutai STT hparams from a Kyutai STT GGUF.
    ///
    /// Missing numeric keys read as `0` placeholders (the CSM
    /// `read_u32_or_zero` / `read_f32_or` pattern) so a shape-only
    /// converter path decays gracefully to [`Self::validate_for_forward`]'s
    /// loud gate; wrong-typed keys are loud
    /// [`VokraError::InvalidArgument`] here (FR-EX-08 — never a silent
    /// type coercion). Booleans ride as u32 0/1 per the converter contract
    /// (`u32::from(bool)`), so `causal` / `multi_linear` / `weights_per_step`
    /// read back through the same `read_u32_or_zero` helper.
    ///
    /// The `delays` vector is reconstructed from the `n_delays` count and
    /// `delay.{i}` indexed keys the converter emits — the same array-
    /// metadata pattern Moshi / mimi use. When `n_delays == 0` (a
    /// metadata-only test fixture) the returned vector is empty and the
    /// downstream [`Self::validate_for_forward`] gate refuses the config
    /// because `delays.len() != n_channels()`; a `n_delays > 0` reads
    /// every indexed entry back verbatim.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] if any present key has the wrong
    /// metadata type.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let backbone = KyutaiSttBackboneConfig {
            n_layer: read_u32_or_zero(file, KEY_BB_N_LAYER)? as usize,
            d_model: read_u32_or_zero(file, KEY_BB_D_MODEL)? as usize,
            n_head: read_u32_or_zero(file, KEY_BB_N_HEAD)? as usize,
            hidden_scale: read_f32_or(file, KEY_BB_HIDDEN_SCALE, 0.0)?,
            context: read_u32_or_zero(file, KEY_BB_CONTEXT)? as usize,
            rope_max_period: read_f32_or(file, KEY_BB_ROPE_MAX_PERIOD, 0.0)?,
        };
        let depformer = KyutaiSttDepformerConfig {
            n_layer: read_u32_or_zero(file, KEY_DEP_N_LAYER)? as usize,
            d_model: read_u32_or_zero(file, KEY_DEP_D_MODEL)? as usize,
            n_head: read_u32_or_zero(file, KEY_DEP_N_HEAD)? as usize,
            multi_linear: read_u32_or_zero(file, KEY_DEP_MULTI_LINEAR)? != 0,
            weights_per_step: read_u32_or_zero(file, KEY_DEP_WEIGHTS_PER_STEP)? != 0,
        };
        let n_delays = read_u32_or_zero(file, KEY_N_DELAYS)? as usize;
        let mut delays = Vec::with_capacity(n_delays);
        for i in 0..n_delays {
            let key = format!("{PREFIX_DELAY}{i}");
            delays.push(read_u32_or_zero(file, &key)?);
        }
        Ok(Self {
            backbone,
            depformer,
            n_q: read_u32_or_zero(file, KEY_N_Q)? as usize,
            dep_q: read_u32_or_zero(file, KEY_DEP_Q)? as usize,
            audio_card: read_u32_or_zero(file, KEY_AUDIO_CARD)? as usize,
            text_card: read_u32_or_zero(file, KEY_TEXT_CARD)? as usize,
            text_pad_id: read_u32_or_zero(file, KEY_TEXT_PAD_ID)?,
            causal: read_u32_or_zero(file, KEY_BB_CAUSAL)? != 0,
            rms_norm_eps: read_f32_or(file, KEY_BB_RMS_NORM_EPS, 1e-8)?,
            delays,
            audio_delay_seconds: read_f32_or(file, KEY_AUDIO_DELAY_SECS, 0.0)?,
            audio_silence_prefix_seconds: read_f32_or(file, KEY_AUDIO_SILENCE_PREFIX_SECS, 0.0)?,
            sample_rate: read_u32_or_zero(file, KEY_SAMPLE_RATE)?,
        })
    }
}

// Missing numeric keys read as `0` placeholders (a shape-only converter
// path decays gracefully to `validate_for_forward`'s loud gate); wrong-
// typed keys are loud `VokraError::InvalidArgument` (FR-EX-08 — never a
// silent type coercion). Mirrors the CSM helper of the same name.
fn read_u32_or_zero(file: &GgufFile, key: &str) -> Result<u32> {
    match file.get(key) {
        Some(GgufMetadataValue::U32(v)) => Ok(*v),
        None => Ok(0),
        Some(other) => Err(VokraError::InvalidArgument(format!(
            "kyutai-stt config: `{key}` is not a UINT32 (got {:?})",
            other.value_type()
        ))),
    }
}

fn read_f32_or(file: &GgufFile, key: &str, default: f32) -> Result<f32> {
    match file.get(key) {
        Some(GgufMetadataValue::F32(v)) => Ok(*v),
        None => Ok(default),
        Some(other) => Err(VokraError::InvalidArgument(format!(
            "kyutai-stt config: `{key}` is not a FLOAT32 (got {:?})",
            other.value_type()
        ))),
    }
}

// ---------------------------------------------------------------------------
// Weights
// ---------------------------------------------------------------------------

/// Per-block backbone weights (pre-norm MHA + SiLU-gating FFN).
///
/// Field names mirror the Moshi backbone (`kyutai-labs/moshi`
/// `transformer.py`, ADR M4-06 §D2): fused Q/K/V projection with
/// `[3·d_model, d_model]` transposed shape, an output projection, and a
/// gating FFN with the `linear_in = [2·ffn_hidden, d_model]` +
/// `linear_out = [d_model, ffn_hidden]` shape upstream `ActivationGating`
/// exposes.
#[derive(Debug, Clone)]
pub struct KyutaiSttBlockWeights {
    /// Pre-attention RMSNorm γ, shape `[d_model]`.
    pub attn_norm: Vec<f32>,
    /// Fused Q/K/V projection (transposed), shape `[d_model, 3*d_model]`.
    pub qkv_proj: Vec<f32>,
    /// Output projection (transposed), shape `[d_model, d_model]`.
    pub out_proj: Vec<f32>,
    /// Pre-FFN RMSNorm γ, shape `[d_model]`.
    pub ffn_norm: Vec<f32>,
    /// Gating linear-in (fused gate + up), shape
    /// `[d_model, 2 * ffn_hidden]`.
    pub linear_in: Vec<f32>,
    /// Gating linear-out, shape `[ffn_hidden, d_model]`.
    pub linear_out: Vec<f32>,
}

/// Kyutai STT weight store: text/audio embeddings + backbone blocks +
/// final norm + text head.
///
/// [`Self::synthesized`] builds a deterministic fixture (SplitMix64 +
/// Xavier) against `config` so shape / dtype / size can be exercised
/// without the real HF checkpoint. Real-checkpoint binding is a follow-up
/// (T29-equivalent — tensor-name manifest fetch from the upstream release).
///
/// Depformer weights are **absent** from the scaffold: with `dep_q=0` the
/// depformer per-step count is zero and no audio-prediction weights ride
/// the checkpoint. A hypothetical future `dep_q > 0` variant would extend
/// this store with a per-step depformer weight vector.
#[derive(Debug, Clone)]
pub struct KyutaiSttWeights {
    /// Text-token input embedding, shape `[text_card + 1, d_model]`.
    /// The extra row is the initial text token (`text_initial_token_id =
    /// text_card`, the Moshi convention Kyutai inherits).
    pub text_embedding: Vec<f32>,
    /// Per-audio-channel input embeddings — `n_q` tables each of shape
    /// `[audio_card + 1, d_model]` (extra row = initial audio token).
    pub audio_embeddings: Vec<Vec<f32>>,
    /// Backbone blocks in order.
    pub blocks: Vec<KyutaiSttBlockWeights>,
    /// Final backbone RMSNorm γ, shape `[d_model]`.
    pub final_norm: Vec<f32>,
    /// Text output head (transposed), shape `[d_model, text_card]`.
    pub text_head: Vec<f32>,
    /// `true` when built by [`Self::synthesized`] — never a real upstream
    /// checkpoint. Real-checkpoint bindings set this to `false`.
    pub is_synthesized: bool,
}

impl KyutaiSttWeights {
    /// Builds a deterministic synthesized fixture from `config` and
    /// `seed`.
    ///
    /// Draws are Xavier-uniform ± `sqrt(6 / (fan_in + fan_out))` via a
    /// [`SplitMix64`] stream — reproducible, allocation-only, zero-dep.
    /// Every RMSNorm γ starts at `1.0`.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] if `config.validate_for_forward`
    /// fails.
    pub fn synthesized(config: &KyutaiSttConfig, seed: u64) -> Result<Self> {
        config.validate_for_forward()?;
        let mut rng = SplitMix64::new(seed);
        let bb = &config.backbone;
        let d = bb.d_model;
        let ffn = bb.ffn_hidden();
        let text_rows = config.text_card + 1;
        let audio_rows = config.audio_card + 1;

        let text_embedding = xavier(&mut rng, text_rows * d, text_rows, d);
        let mut audio_embeddings = Vec::with_capacity(config.n_q);
        for _ in 0..config.n_q {
            audio_embeddings.push(xavier(&mut rng, audio_rows * d, audio_rows, d));
        }

        let mut blocks = Vec::with_capacity(bb.n_layer);
        for _ in 0..bb.n_layer {
            blocks.push(KyutaiSttBlockWeights {
                attn_norm: vec![1.0; d],
                qkv_proj: xavier(&mut rng, d * 3 * d, d, 3 * d),
                out_proj: xavier(&mut rng, d * d, d, d),
                ffn_norm: vec![1.0; d],
                linear_in: xavier(&mut rng, d * 2 * ffn, d, 2 * ffn),
                linear_out: xavier(&mut rng, ffn * d, ffn, d),
            });
        }
        let final_norm = vec![1.0; d];
        let text_head = xavier(&mut rng, d * config.text_card, d, config.text_card);

        Ok(Self {
            text_embedding,
            audio_embeddings,
            blocks,
            final_norm,
            text_head,
            is_synthesized: true,
        })
    }
}

/// Xavier-uniform draw of `count` `f32`s in `[-a, +a]` where
/// `a = sqrt(6 / (fan_in + fan_out))`. Deterministic under a fixed `rng`.
fn xavier(rng: &mut SplitMix64, count: usize, fan_in: usize, fan_out: usize) -> Vec<f32> {
    let a = (6.0 / (fan_in + fan_out) as f32).sqrt();
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        // Map the top 24 bits of the u64 stream to a f32 in [0, 1).
        let raw = (rng.next_u64() >> 40) as u32;
        let u01 = (raw as f32) / ((1u32 << 24) as f32);
        out.push((u01 * 2.0 - 1.0) * a);
    }
    out
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

/// Kyutai STT ASR engine handle.
///
/// Carries the resolved config and weight store. [`Self::transcribe`] is
/// the primary Mimi-tokens → text entry point; until real weights are
/// bound (see the module docstring) it returns
/// [`VokraError::NotImplemented`] with a message naming the blocker
/// (FR-EX-08 — never a silent zero-fill or empty transcript).
#[derive(Debug, Clone)]
pub struct KyutaiSttAsr {
    cfg: KyutaiSttConfig,
    weights: KyutaiSttWeights,
}

impl KyutaiSttAsr {
    /// Assembles an engine from `cfg` and `weights`. Cross-checks the
    /// weight-store shapes against `cfg` (block count, audio-embedding
    /// table count, per-tensor sizes) so a mismatched pair fails loudly
    /// here rather than deep inside a forward.
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] from `cfg.validate_for_forward`.
    /// - [`VokraError::InvalidArgument`] naming the first shape
    ///   mismatch.
    pub fn new(cfg: KyutaiSttConfig, weights: KyutaiSttWeights) -> Result<Self> {
        cfg.validate_for_forward()?;
        let bb = &cfg.backbone;
        let d = bb.d_model;
        let ffn = bb.ffn_hidden();
        let text_rows = cfg.text_card + 1;
        let audio_rows = cfg.audio_card + 1;

        if weights.text_embedding.len() != text_rows * d {
            return Err(VokraError::InvalidArgument(format!(
                "kyutai-stt weights: text_embedding.len()={} != (text_card+1)*d_model={}",
                weights.text_embedding.len(),
                text_rows * d,
            )));
        }
        if weights.audio_embeddings.len() != cfg.n_q {
            return Err(VokraError::InvalidArgument(format!(
                "kyutai-stt weights: audio_embeddings.len()={} != n_q={}",
                weights.audio_embeddings.len(),
                cfg.n_q,
            )));
        }
        for (i, tbl) in weights.audio_embeddings.iter().enumerate() {
            let expected = audio_rows * d;
            if tbl.len() != expected {
                return Err(VokraError::InvalidArgument(format!(
                    "kyutai-stt weights: audio_embeddings[{i}].len()={} != {expected}",
                    tbl.len(),
                )));
            }
        }
        if weights.blocks.len() != bb.n_layer {
            return Err(VokraError::InvalidArgument(format!(
                "kyutai-stt weights: blocks.len()={} != backbone.n_layer={}",
                weights.blocks.len(),
                bb.n_layer,
            )));
        }
        for (i, blk) in weights.blocks.iter().enumerate() {
            for (name, len, expected) in [
                ("attn_norm", blk.attn_norm.len(), d),
                ("qkv_proj", blk.qkv_proj.len(), d * 3 * d),
                ("out_proj", blk.out_proj.len(), d * d),
                ("ffn_norm", blk.ffn_norm.len(), d),
                ("linear_in", blk.linear_in.len(), d * 2 * ffn),
                ("linear_out", blk.linear_out.len(), ffn * d),
            ] {
                if len != expected {
                    return Err(VokraError::InvalidArgument(format!(
                        "kyutai-stt weights: block {i} `{name}` len={len} != {expected}",
                    )));
                }
            }
        }
        if weights.final_norm.len() != d {
            return Err(VokraError::InvalidArgument(format!(
                "kyutai-stt weights: final_norm.len()={} != d_model={}",
                weights.final_norm.len(),
                d,
            )));
        }
        if weights.text_head.len() != d * cfg.text_card {
            return Err(VokraError::InvalidArgument(format!(
                "kyutai-stt weights: text_head.len()={} != d_model * text_card = {}",
                weights.text_head.len(),
                d * cfg.text_card,
            )));
        }
        Ok(Self { cfg, weights })
    }

    /// The resolved configuration.
    #[must_use]
    pub fn config(&self) -> &KyutaiSttConfig {
        &self.cfg
    }

    /// True iff the weight store was built by
    /// [`KyutaiSttWeights::synthesized`] (never a real upstream
    /// checkpoint).
    #[must_use]
    pub fn is_synthesized(&self) -> bool {
        self.weights.is_synthesized
    }

    /// Transcribes a sequence of Mimi codes into text tokens.
    ///
    /// `mimi_codes` is a **row-major `[T, n_q]`** matrix of audio codes:
    /// `T` = number of 12.5 Hz Mimi frames, `n_q` = `config().n_q`
    /// (32 for STT-2.6B-EN). Each code is in `[0, audio_card)`; the
    /// caller Mimi-encodes PCM first (see [`vokra_ops::mimi_rvq`]).
    ///
    /// This is the primary Mimi tokens → text entry point. **Real
    /// weights required**: synthesized-weight builds cannot produce
    /// meaningful text (they would be noise or a hallucinated fixed
    /// sequence), so this returns [`VokraError::NotImplemented`] naming
    /// the blocker. Callers verify the shape flow through
    /// [`KyutaiSttAsr::new`] + [`KyutaiSttWeights::synthesized`] today;
    /// a follow-up wave binds the real HF checkpoint tensor names and
    /// wires the forward.
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] if `mimi_codes.len()` is not a
    ///   multiple of `n_q`, is empty, or contains an id outside
    ///   `[0, audio_card)`.
    /// - [`VokraError::NotImplemented`] otherwise (real forward not yet
    ///   bound — FR-EX-08).
    pub fn transcribe(&self, mimi_codes: &[u32]) -> Result<Vec<u32>> {
        if mimi_codes.is_empty() {
            return Err(VokraError::InvalidArgument(
                "kyutai-stt transcribe: mimi_codes is empty".to_owned(),
            ));
        }
        if mimi_codes.len() % self.cfg.n_q != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "kyutai-stt transcribe: mimi_codes.len()={} is not a multiple \
                 of n_q={} — expected a row-major [T, n_q] frame matrix",
                mimi_codes.len(),
                self.cfg.n_q,
            )));
        }
        let audio_vocab = self.cfg.audio_card as u32;
        for (i, code) in mimi_codes.iter().enumerate() {
            if *code >= audio_vocab {
                return Err(VokraError::InvalidArgument(format!(
                    "kyutai-stt transcribe: mimi_codes[{i}]={code} out of [0, {audio_vocab})",
                )));
            }
        }
        if self.weights.is_synthesized {
            return Err(VokraError::NotImplemented(
                "kyutai-stt transcribe: this engine holds synthesized weights \
                 (deterministic fixture from KyutaiSttWeights::synthesized) — \
                 synthesized-weight text would be a hallucinated sequence, not \
                 a real transcript. Bind real Kyutai STT-2.6B-EN weights \
                 (CC-BY 4.0, kyutai/stt-2.6b-en) before invoking transcribe. \
                 The shape flow (config validation, weight-store construction, \
                 code-frame shape check) is exercised through KyutaiSttAsr::new; \
                 the real-checkpoint tensor-name manifest lands in a follow-up \
                 wave (T29-equivalent — the Moshi / CSM pattern). \
                 Primary source: https://huggingface.co/kyutai/stt-2.6b-en / \
                 https://github.com/kyutai-labs/delayed-streams-modeling",
            ));
        }
        Err(VokraError::NotImplemented(
            "kyutai-stt transcribe: real weights are bound but the \
             audio-embedding sum + prenorm MHA + gating FFN + text-head \
             sampling + SentencePiece detokenize forward path has not landed \
             yet. Follow-up wave: transcribe the upstream tensor manifest and \
             wire the sliding-window causal attention (context=375) forward \
             through the `Compute` seam (Moshi T29 pattern). \
             Primary source: https://huggingface.co/kyutai/stt-2.6b-en / \
             https://github.com/kyutai-labs/delayed-streams-modeling",
        ))
    }

    /// Loads a Kyutai STT GGUF from raw bytes under `policy` (M2-13 gate —
    /// a non-commercial provenance without a research flag is refused).
    ///
    /// Weight posture: **synthesized bridge** until the real-checkpoint
    /// tensor-name manifest lands (T29-equivalent — the CSM
    /// [`from_gguf_with_policy`](super::csm::CsmEngine::from_gguf_with_policy)
    /// precedent). The engine binds
    /// [`KyutaiSttWeights::synthesized`] against the GGUF's shape
    /// config using [`KYUTAI_STT_FROM_GGUF_DEFAULT_SEED`] so shape /
    /// dtype / size flow can be exercised without the real HF
    /// checkpoint; a `transcribe` call fires the synthesized-weight
    /// loud-partial arm and names the primary source URL.
    ///
    /// The Kyutai STT weight license is **CC-BY 4.0** (`AttributionRequired`) —
    /// the converter's registry mapping and provenance stamps make the
    /// M2-13 gate pass commercially, and the FR-MD-09 attribution
    /// surface activates. `docs/license-audit.md` row 272 records the
    /// commercial sign-off (2026-07-28 yousan).
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] on parse failure / wrong or missing
    ///   `vokra.model.arch` — the message names the expected arch tag
    ///   (`kyutai-stt`), sibling arch tags (`csm` / `moshi` / `kyutai-tts`)
    ///   so a mis-routed GGUF fails specifically here, and the primary
    ///   source URL.
    /// - [`VokraError::ResearchLicenseRequired`] (from the M2-13 gate)
    ///   when the weight class is gated and `policy` grants no research
    ///   opt-in (never a silent skip / substitution).
    /// - [`VokraError::InvalidArgument`] on a `0`-placeholder shape
    ///   config (a scaffold converter path that never wrote the real
    ///   hparams) from the downstream
    ///   [`KyutaiSttConfig::validate_for_forward`] gate.
    pub fn from_gguf_with_policy(bytes: &[u8], policy: &CompliancePolicy) -> Result<Self> {
        let file = GgufFile::parse(bytes.to_vec())
            .map_err(|e| VokraError::ModelLoad(format!("kyutai-stt GGUF: {e}")))?;
        match file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()) {
            Some(a) if a == EXPECTED_ARCH => {}
            Some(other) => {
                return Err(VokraError::ModelLoad(format!(
                    "kyutai-stt: GGUF arch is `{other}`, expected `{EXPECTED_ARCH}` \
                     (was this GGUF produced by `vokra-cli convert --model kyutai-stt`? \
                     Sibling Kyutai / Moshi-family arches — `csm` (Sesame CSM-1B \
                     S2S), `moshi` (Kyutai Helium + Mimi full-duplex), `kyutai-tts` \
                     (Kyutai text-to-speech) — are different topologies). \
                     Primary source: https://huggingface.co/kyutai/stt-2.6b-en / \
                     https://github.com/kyutai-labs/delayed-streams-modeling"
                )));
            }
            None => {
                return Err(VokraError::ModelLoad(format!(
                    "kyutai-stt: GGUF is missing `vokra.model.arch` (converter did \
                     not stamp it — this is not a Vokra-native `{EXPECTED_ARCH}` \
                     GGUF). Primary source: \
                     https://huggingface.co/kyutai/stt-2.6b-en / \
                     https://github.com/kyutai-labs/delayed-streams-modeling"
                )));
            }
        }
        check_weight_license(&file, policy)?;
        let cfg = KyutaiSttConfig::from_gguf(&file)?;
        // `synthesized` runs `validate_for_forward` internally; keep the
        // explicit call here so a validate failure surfaces with the config
        // context intact (same posture as CSM `from_gguf_with_policy`).
        cfg.validate_for_forward()?;
        let weights = KyutaiSttWeights::synthesized(&cfg, KYUTAI_STT_FROM_GGUF_DEFAULT_SEED)?;
        Self::new(cfg, weights)
    }

    /// Loads a Kyutai STT GGUF from a file path with the fail-closed
    /// strict policy ([`CompliancePolicy::strict`]).
    ///
    /// The Kyutai STT weight license is **CC-BY 4.0**
    /// (`AttributionRequired`), which is commercially permitted — the
    /// M2-13 gate passes under `strict` without a research opt-in.
    ///
    /// # Errors
    ///
    /// - [`VokraError::Io`] on read failure.
    /// - See [`Self::from_gguf_with_policy`].
    pub fn from_path(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let bytes = std::fs::read(path.as_ref()).map_err(VokraError::Io)?;
        Self::from_gguf_with_policy(&bytes, &CompliancePolicy::strict())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::LicenseClass;
    use vokra_core::gguf::GgufBuilder;

    /// Every hparam matches the primary source
    /// (`huggingface.co/kyutai/stt-2.6b-en/raw/main/config.json`) verbatim.
    #[test]
    fn stt_2_6b_en_matches_primary_source_config_json() {
        let c = KyutaiSttConfig::stt_2_6b_en();
        // Backbone
        assert_eq!(c.backbone.n_layer, 48);
        assert_eq!(c.backbone.d_model, 2048);
        assert_eq!(c.backbone.n_head, 32);
        assert_eq!(c.backbone.hidden_scale, 4.125);
        assert_eq!(c.backbone.context, 375);
        assert_eq!(c.backbone.rope_max_period, 100_000.0);
        // Depformer (structurally present, unused when dep_q=0)
        assert_eq!(c.depformer.n_layer, 6);
        assert_eq!(c.depformer.d_model, 1024);
        assert_eq!(c.depformer.n_head, 16);
        assert!(c.depformer.multi_linear);
        assert!(c.depformer.weights_per_step);
        // Audio / text / streaming
        assert_eq!(c.n_q, 32);
        assert_eq!(c.dep_q, 0);
        assert_eq!(c.audio_card, 2048);
        assert_eq!(c.text_card, 4000);
        assert_eq!(c.text_pad_id, 3);
        assert!(c.causal);
        assert_eq!(c.rms_norm_eps, 1e-8);
        assert_eq!(c.delays.len(), 33);
        assert!(c.delays.iter().all(|d| *d == 0));
        assert_eq!(c.audio_delay_seconds, 2.5);
        assert_eq!(c.audio_silence_prefix_seconds, 1.0);
        // Mimi 24 kHz inheritance.
        assert_eq!(c.sample_rate, 24_000);
        // Derived values.
        assert_eq!(c.backbone.head_dim(), 64);
        assert_eq!(c.backbone.ffn_hidden(), 8448);
        assert_eq!(c.n_channels(), 33);
        assert_eq!(c.max_delay(), 0);
        // Everything above adds up to a well-formed config.
        c.validate_for_forward()
            .expect("stt-2.6b-en is well-formed");
    }

    #[test]
    fn tiny_config_is_well_formed() {
        KyutaiSttConfig::tiny_for_tests()
            .validate_for_forward()
            .expect("tiny config is well-formed");
    }

    #[test]
    fn config_head_split_ill_formed_is_rejected() {
        let mut c = KyutaiSttConfig::tiny_for_tests();
        c.backbone.n_head = 3; // 16 % 3 != 0
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_odd_head_dim_is_rejected() {
        let mut c = KyutaiSttConfig::tiny_for_tests();
        // Make head_dim odd: d_model=12, n_head=4 → head_dim=3, odd.
        c.backbone.d_model = 12;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_delays_length_must_equal_channels() {
        let mut c = KyutaiSttConfig::tiny_for_tests();
        c.delays.push(0);
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
        c.delays.pop();
        c.delays.pop();
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_pad_id_out_of_range_is_rejected() {
        let mut c = KyutaiSttConfig::tiny_for_tests();
        c.text_pad_id = c.text_card as u32;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_dep_q_exceeds_n_q_is_rejected() {
        let mut c = KyutaiSttConfig::tiny_for_tests();
        c.dep_q = c.n_q + 1;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_zero_n_q_is_rejected() {
        let mut c = KyutaiSttConfig::tiny_for_tests();
        c.n_q = 0;
        c.delays = vec![0];
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_zero_vocab_is_rejected() {
        let mut c = KyutaiSttConfig::tiny_for_tests();
        c.audio_card = 0;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
        let mut c = KyutaiSttConfig::tiny_for_tests();
        c.text_card = 0;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_zero_context_is_rejected() {
        let mut c = KyutaiSttConfig::tiny_for_tests();
        c.backbone.context = 0;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_zero_layer_is_rejected() {
        let mut c = KyutaiSttConfig::tiny_for_tests();
        c.backbone.n_layer = 0;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_non_finite_hidden_scale_is_rejected() {
        let mut c = KyutaiSttConfig::tiny_for_tests();
        c.backbone.hidden_scale = f32::NAN;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
        let mut c = KyutaiSttConfig::tiny_for_tests();
        c.backbone.hidden_scale = 0.0;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn synthesized_weights_are_deterministic_and_shape_correct() {
        let c = KyutaiSttConfig::tiny_for_tests();
        let w1 = KyutaiSttWeights::synthesized(&c, 0x42).expect("build 1");
        let w2 = KyutaiSttWeights::synthesized(&c, 0x42).expect("build 2");
        // Determinism.
        assert_eq!(w1.text_embedding, w2.text_embedding);
        assert_eq!(
            w1.blocks[0].qkv_proj, w2.blocks[0].qkv_proj,
            "same seed → same weights"
        );
        assert!(w1.is_synthesized);
        // Shape flow.
        let d = c.backbone.d_model;
        let ffn = c.backbone.ffn_hidden();
        assert_eq!(w1.text_embedding.len(), (c.text_card + 1) * d);
        assert_eq!(w1.audio_embeddings.len(), c.n_q);
        for tbl in &w1.audio_embeddings {
            assert_eq!(tbl.len(), (c.audio_card + 1) * d);
        }
        assert_eq!(w1.blocks.len(), c.backbone.n_layer);
        for blk in &w1.blocks {
            assert_eq!(blk.attn_norm.len(), d);
            assert_eq!(blk.qkv_proj.len(), d * 3 * d);
            assert_eq!(blk.out_proj.len(), d * d);
            assert_eq!(blk.ffn_norm.len(), d);
            assert_eq!(blk.linear_in.len(), d * 2 * ffn);
            assert_eq!(blk.linear_out.len(), ffn * d);
        }
        assert_eq!(w1.final_norm.len(), d);
        assert_eq!(w1.text_head.len(), d * c.text_card);
    }

    #[test]
    fn synthesized_weights_different_seeds_diverge() {
        let c = KyutaiSttConfig::tiny_for_tests();
        let w_a = KyutaiSttWeights::synthesized(&c, 1).expect("build a");
        let w_b = KyutaiSttWeights::synthesized(&c, 2).expect("build b");
        // Two distinct seeds must produce different Xavier draws (probability
        // of collision on the first row is vanishing).
        assert_ne!(w_a.text_embedding, w_b.text_embedding);
    }

    #[test]
    fn synthesized_rejects_ill_formed_config() {
        let mut c = KyutaiSttConfig::tiny_for_tests();
        c.backbone.n_head = 3; // 16 % 3 != 0
        assert!(matches!(
            KyutaiSttWeights::synthesized(&c, 7),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn asr_new_accepts_matching_config_and_weights() {
        let c = KyutaiSttConfig::tiny_for_tests();
        let w = KyutaiSttWeights::synthesized(&c, 7).expect("weights");
        let asr = KyutaiSttAsr::new(c.clone(), w).expect("kyutai-stt asr");
        assert_eq!(asr.config().backbone.d_model, c.backbone.d_model);
        assert!(asr.is_synthesized());
    }

    #[test]
    fn asr_new_rejects_layer_count_mismatch() {
        let c = KyutaiSttConfig::tiny_for_tests();
        let mut w = KyutaiSttWeights::synthesized(&c, 7).expect("weights");
        w.blocks.pop();
        assert!(matches!(
            KyutaiSttAsr::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn asr_new_rejects_tensor_size_mismatch() {
        let c = KyutaiSttConfig::tiny_for_tests();
        let mut w = KyutaiSttWeights::synthesized(&c, 7).expect("weights");
        w.blocks[0].qkv_proj.pop();
        assert!(matches!(
            KyutaiSttAsr::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn asr_new_rejects_audio_embedding_count_mismatch() {
        let c = KyutaiSttConfig::tiny_for_tests();
        let mut w = KyutaiSttWeights::synthesized(&c, 7).expect("weights");
        w.audio_embeddings.pop();
        assert!(matches!(
            KyutaiSttAsr::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn asr_new_rejects_per_audio_embedding_size_mismatch() {
        let c = KyutaiSttConfig::tiny_for_tests();
        let mut w = KyutaiSttWeights::synthesized(&c, 7).expect("weights");
        w.audio_embeddings[0].pop();
        assert!(matches!(
            KyutaiSttAsr::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn asr_new_rejects_text_embedding_size_mismatch() {
        let c = KyutaiSttConfig::tiny_for_tests();
        let mut w = KyutaiSttWeights::synthesized(&c, 7).expect("weights");
        w.text_embedding.pop();
        assert!(matches!(
            KyutaiSttAsr::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn asr_new_rejects_final_norm_size_mismatch() {
        let c = KyutaiSttConfig::tiny_for_tests();
        let mut w = KyutaiSttWeights::synthesized(&c, 7).expect("weights");
        w.final_norm.pop();
        assert!(matches!(
            KyutaiSttAsr::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn asr_new_rejects_text_head_size_mismatch() {
        let c = KyutaiSttConfig::tiny_for_tests();
        let mut w = KyutaiSttWeights::synthesized(&c, 7).expect("weights");
        w.text_head.pop();
        assert!(matches!(
            KyutaiSttAsr::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn transcribe_rejects_empty_codes() {
        let c = KyutaiSttConfig::tiny_for_tests();
        let w = KyutaiSttWeights::synthesized(&c, 7).expect("weights");
        let asr = KyutaiSttAsr::new(c, w).expect("kyutai-stt asr");
        assert!(matches!(
            asr.transcribe(&[]),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn transcribe_rejects_non_multiple_of_n_q_length() {
        let c = KyutaiSttConfig::tiny_for_tests();
        let n_q = c.n_q;
        let w = KyutaiSttWeights::synthesized(&c, 7).expect("weights");
        let asr = KyutaiSttAsr::new(c, w).expect("kyutai-stt asr");
        // A slice one code short of a full frame.
        let codes = vec![0u32; n_q - 1];
        assert!(matches!(
            asr.transcribe(&codes),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn transcribe_rejects_out_of_range_code() {
        let c = KyutaiSttConfig::tiny_for_tests();
        let n_q = c.n_q;
        let vocab = c.audio_card as u32;
        let w = KyutaiSttWeights::synthesized(&c, 7).expect("weights");
        let asr = KyutaiSttAsr::new(c, w).expect("kyutai-stt asr");
        let mut codes = vec![0u32; n_q];
        codes[n_q - 1] = vocab;
        assert!(matches!(
            asr.transcribe(&codes),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    /// The primary NotImplemented path names the synthesized-weight
    /// blocker (FR-EX-08 — never a silent zero-fill / hallucinated
    /// transcript).
    #[test]
    fn transcribe_on_synthesized_weights_is_loud_not_implemented() {
        let c = KyutaiSttConfig::tiny_for_tests();
        let n_q = c.n_q;
        let w = KyutaiSttWeights::synthesized(&c, 7).expect("weights");
        let asr = KyutaiSttAsr::new(c, w).expect("kyutai-stt asr");
        let codes = vec![0u32; n_q * 2];
        let err = asr.transcribe(&codes).unwrap_err();
        match err {
            VokraError::NotImplemented(msg) => {
                assert!(
                    msg.contains("synthesized"),
                    "message must name synthesized-weight blocker: {msg}"
                );
            }
            other => panic!("expected NotImplemented, got {other:?}"),
        }
    }

    #[test]
    fn expected_arch_is_kyutai_stt() {
        assert_eq!(EXPECTED_ARCH, "kyutai-stt");
    }

    #[test]
    fn sample_rate_matches_mimi_boundary() {
        // 24 kHz — inherited from Mimi (the codec `mimi_name` names in the
        // upstream config). Kyutai STT does NOT operate on PCM directly;
        // this constant documents the Mimi-side sample rate the caller
        // must use before encoding.
        assert_eq!(KYUTAI_STT_SAMPLE_RATE, 24_000);
    }

    // -----------------------------------------------------------------------
    // GGUF-loader (`from_gguf` / `from_gguf_with_policy` / `from_path`) tests
    //
    // These pin the loud-partial scaffold the M2-13 gate + arch check +
    // config round-trip + license read + synthesized-weight `transcribe`
    // arm depend on. Every path fails loudly (FR-EX-08) — never a silent
    // zero-fill / substitution / mis-typed cast.
    // -----------------------------------------------------------------------

    /// Builds a metadata-only GGUF whose `vokra.model.arch` is `arch`
    /// (unless `set_arch` is false). Adds every well-formed
    /// `vokra.kyutai_stt.*` chunk group `KyutaiSttConfig::from_gguf`
    /// reads, mirroring the offline converter's `write_hparams` so a
    /// round-trip yields the same `KyutaiSttConfig::stt_2_6b_en()`
    /// snapshot without dragging the converter crate into the models
    /// test tree.
    fn build_gguf_with_hparams(arch: Option<&str>) -> Vec<u8> {
        let mut b = GgufBuilder::new();
        if let Some(a) = arch {
            b.add_string(chunks::KEY_MODEL_ARCH, a);
        }
        let cfg = KyutaiSttConfig::stt_2_6b_en();
        b.add_u32(KEY_SAMPLE_RATE, cfg.sample_rate);
        // Backbone
        b.add_u32(KEY_BB_N_LAYER, cfg.backbone.n_layer as u32);
        b.add_u32(KEY_BB_D_MODEL, cfg.backbone.d_model as u32);
        b.add_u32(KEY_BB_N_HEAD, cfg.backbone.n_head as u32);
        b.add_f32(KEY_BB_HIDDEN_SCALE, cfg.backbone.hidden_scale);
        b.add_u32(KEY_BB_FFN_HIDDEN, cfg.backbone.ffn_hidden() as u32);
        b.add_u32(KEY_BB_CONTEXT, cfg.backbone.context as u32);
        b.add_f32(KEY_BB_ROPE_MAX_PERIOD, cfg.backbone.rope_max_period);
        b.add_u32(KEY_BB_CAUSAL, u32::from(cfg.causal));
        b.add_f32(KEY_BB_RMS_NORM_EPS, cfg.rms_norm_eps);
        // Depformer
        b.add_u32(KEY_DEP_N_LAYER, cfg.depformer.n_layer as u32);
        b.add_u32(KEY_DEP_D_MODEL, cfg.depformer.d_model as u32);
        b.add_u32(KEY_DEP_N_HEAD, cfg.depformer.n_head as u32);
        b.add_u32(KEY_DEP_MULTI_LINEAR, u32::from(cfg.depformer.multi_linear));
        b.add_u32(
            KEY_DEP_WEIGHTS_PER_STEP,
            u32::from(cfg.depformer.weights_per_step),
        );
        // Audio / text / streaming
        b.add_u32(KEY_N_Q, cfg.n_q as u32);
        b.add_u32(KEY_DEP_Q, cfg.dep_q as u32);
        b.add_u32(KEY_AUDIO_CARD, cfg.audio_card as u32);
        b.add_u32(KEY_TEXT_CARD, cfg.text_card as u32);
        b.add_u32(KEY_TEXT_PAD_ID, cfg.text_pad_id);
        b.add_f32(KEY_AUDIO_DELAY_SECS, cfg.audio_delay_seconds);
        b.add_f32(
            KEY_AUDIO_SILENCE_PREFIX_SECS,
            cfg.audio_silence_prefix_seconds,
        );
        // Delays
        b.add_u32(KEY_N_DELAYS, cfg.delays.len() as u32);
        for (i, d) in cfg.delays.iter().enumerate() {
            b.add_u32(&format!("{PREFIX_DELAY}{i}"), *d);
        }
        // Provenance — AttributionRequired (CC-BY 4.0) so the M2-13 gate
        // passes under `CompliancePolicy::strict()`.
        b.add_string(
            chunks::KEY_PROVENANCE_WEIGHT_LICENSE,
            LicenseClass::AttributionRequired.as_str(),
        );
        b.add_string(chunks::KEY_PROVENANCE_LICENSE, "CC-BY-4.0");
        b.add_string(chunks::KEY_PROVENANCE_MODEL_ID, "kyutai/stt-2.6b-en");
        b.to_bytes().expect("serialize kyutai-stt fixture GGUF")
    }

    /// A GGUF with no `vokra.model.arch` fails
    /// [`KyutaiSttAsr::from_gguf_with_policy`] with a message that names
    /// the expected arch tag + the primary source URL. Never a silent
    /// substitution (FR-EX-08).
    #[test]
    fn from_gguf_rejects_missing_arch() {
        let bytes = build_gguf_with_hparams(None);
        let err = KyutaiSttAsr::from_gguf_with_policy(&bytes, &CompliancePolicy::strict())
            .expect_err("missing arch must be rejected");
        match err {
            VokraError::ModelLoad(msg) => {
                assert!(
                    msg.contains(EXPECTED_ARCH),
                    "message must name expected arch `{EXPECTED_ARCH}`: {msg}"
                );
                assert!(
                    msg.contains("huggingface.co/kyutai/stt-2.6b-en"),
                    "message must name the primary source URL: {msg}"
                );
            }
            other => panic!("expected ModelLoad, got {other:?}"),
        }
    }

    /// A GGUF whose arch is a sibling (`csm`) fails with a message that
    /// names both `kyutai-stt` and the offending tag so the caller can
    /// diagnose the mis-routed conversion.
    #[test]
    fn from_gguf_rejects_wrong_arch() {
        let bytes = build_gguf_with_hparams(Some("csm"));
        let err = KyutaiSttAsr::from_gguf_with_policy(&bytes, &CompliancePolicy::strict())
            .expect_err("wrong arch must be rejected");
        match err {
            VokraError::ModelLoad(msg) => {
                assert!(
                    msg.contains(EXPECTED_ARCH),
                    "message must name expected arch `{EXPECTED_ARCH}`: {msg}"
                );
                assert!(
                    msg.contains("csm"),
                    "message must name the offending arch tag `csm`: {msg}"
                );
                assert!(
                    msg.contains("huggingface.co/kyutai/stt-2.6b-en"),
                    "message must name the primary source URL: {msg}"
                );
            }
            other => panic!("expected ModelLoad, got {other:?}"),
        }
    }

    /// The `vokra.kyutai_stt.*` chunk group round-trips through the
    /// offline-converter format: every field of
    /// [`KyutaiSttConfig::stt_2_6b_en`] survives write → parse → read.
    /// This pins the cross-crate handshake with `vokra-convert`
    /// (`vokra-convert::models::kyutai_stt::write_hparams`) verbatim —
    /// the two crates only share `vokra-core`, so a converter-side
    /// key-string change surfaces as a runtime `from_gguf` regression.
    #[test]
    fn config_round_trips_from_converter_written_gguf() {
        let bytes = build_gguf_with_hparams(Some(EXPECTED_ARCH));
        let file = GgufFile::parse(bytes).expect("parse fixture");
        let cfg = KyutaiSttConfig::from_gguf(&file).expect("from_gguf");
        let want = KyutaiSttConfig::stt_2_6b_en();
        assert_eq!(cfg, want);
    }

    /// A GGUF whose provenance advertises `AttributionRequired` (CC-BY
    /// 4.0) passes the M2-13 gate under [`CompliancePolicy::strict`]
    /// (no research opt-in needed — the license is commercially
    /// permitted) and the resolution surfaces the attribution-required
    /// class + `is_research_only == false`. This is what makes Kyutai
    /// STT loadable in the default posture.
    #[test]
    fn from_gguf_reads_attribution_required_license() {
        let bytes = build_gguf_with_hparams(Some(EXPECTED_ARCH));
        let file = GgufFile::parse(bytes).expect("parse fixture");
        let resolution =
            check_weight_license(&file, &CompliancePolicy::strict()).expect("strict must pass");
        assert_eq!(resolution.class, LicenseClass::AttributionRequired);
        assert!(
            !resolution.is_research_only(),
            "CC-BY 4.0 is commercial-permitted; must NOT be marked research-only"
        );
        // The M2-13 gate + arch check + config load all pass together.
        let asr = KyutaiSttAsr::from_gguf_with_policy(
            &build_gguf_with_hparams(Some(EXPECTED_ARCH)),
            &CompliancePolicy::strict(),
        )
        .expect("kyutai-stt from_gguf under strict policy");
        assert!(asr.is_synthesized(), "from_gguf binds synthesized bridge");
        assert_eq!(asr.config(), &KyutaiSttConfig::stt_2_6b_en());
    }

    /// The loud-partial transcribe gate names the primary source URL so a
    /// downstream caller / user can look up the real forward's status
    /// (Wave 4 loud-partial contract — never a silent noise transcript).
    #[test]
    fn transcribe_loud_partial_names_primary_source_url() {
        let bytes = build_gguf_with_hparams(Some(EXPECTED_ARCH));
        let asr = KyutaiSttAsr::from_gguf_with_policy(&bytes, &CompliancePolicy::strict())
            .expect("kyutai-stt from_gguf");
        // Build a legal one-frame code slice against the resolved config.
        let n_q = asr.config().n_q;
        let codes = vec![0u32; n_q];
        let err = asr.transcribe(&codes).unwrap_err();
        match err {
            VokraError::NotImplemented(msg) => {
                assert!(
                    msg.contains("https://huggingface.co/kyutai/stt-2.6b-en"),
                    "message must name the HF primary source URL: {msg}"
                );
                assert!(
                    msg.contains("github.com/kyutai-labs/delayed-streams-modeling"),
                    "message must name the GitHub primary source URL: {msg}"
                );
                assert!(
                    msg.contains("synthesized"),
                    "message must name the synthesized-weight blocker: {msg}"
                );
            }
            other => panic!("expected NotImplemented, got {other:?}"),
        }
    }

    /// A GGUF with `n_layer = 0` (a scaffold converter path that never
    /// wrote the real hparams) fails at the downstream
    /// [`KyutaiSttConfig::validate_for_forward`] gate — the loud FR-EX-08
    /// surface, not deep inside a GEMM.
    #[test]
    fn from_gguf_rejects_zero_placeholder_config() {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, EXPECTED_ARCH);
        b.add_string(
            chunks::KEY_PROVENANCE_WEIGHT_LICENSE,
            LicenseClass::AttributionRequired.as_str(),
        );
        // Deliberately omit every `vokra.kyutai_stt.*` chunk — every
        // read decays to the `0` placeholder branch.
        let bytes = b.to_bytes().expect("serialize");
        let err = KyutaiSttAsr::from_gguf_with_policy(&bytes, &CompliancePolicy::strict())
            .expect_err("0-placeholder config must be rejected");
        assert!(
            matches!(err, VokraError::InvalidArgument(_)),
            "expected InvalidArgument, got {err:?}"
        );
    }

    /// A GGUF that mis-types `sample_rate` (F32 instead of U32 — a
    /// hypothetical bad converter path) fails with a loud
    /// [`VokraError::InvalidArgument`] naming the offending key
    /// (FR-EX-08 — never a silent type coercion). This pins the
    /// [`read_u32_or_zero`] helper's type check.
    #[test]
    fn from_gguf_rejects_wrong_typed_key() {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, EXPECTED_ARCH);
        b.add_string(
            chunks::KEY_PROVENANCE_WEIGHT_LICENSE,
            LicenseClass::AttributionRequired.as_str(),
        );
        // sample_rate riding as F32 instead of U32.
        b.add_f32(KEY_SAMPLE_RATE, 24_000.0);
        let bytes = b.to_bytes().expect("serialize");
        let err = KyutaiSttAsr::from_gguf_with_policy(&bytes, &CompliancePolicy::strict())
            .expect_err("wrong-typed key must be rejected");
        match err {
            VokraError::InvalidArgument(msg) => {
                assert!(
                    msg.contains(KEY_SAMPLE_RATE),
                    "message must name the offending key `{KEY_SAMPLE_RATE}`: {msg}"
                );
                assert!(
                    msg.contains("UINT32"),
                    "message must name the expected type UINT32: {msg}"
                );
            }
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    /// `KyutaiSttAsr::from_path` reads the file bytes and threads them
    /// through [`KyutaiSttAsr::from_gguf_with_policy`] with
    /// [`CompliancePolicy::strict`] — the resulting engines are
    /// equivalent (same config, same synthesized-weight bridge, same
    /// loud-partial arm).
    #[test]
    fn from_path_round_trip() {
        let bytes = build_gguf_with_hparams(Some(EXPECTED_ARCH));
        let path = std::env::temp_dir().join(format!(
            "vokra-kyutai-stt-scout-{}.gguf",
            std::process::id()
        ));
        std::fs::write(&path, &bytes).expect("write fixture");
        let via_path = KyutaiSttAsr::from_path(&path).expect("from_path");
        let via_bytes = KyutaiSttAsr::from_gguf_with_policy(&bytes, &CompliancePolicy::strict())
            .expect("from_gguf_with_policy");
        // Best-effort cleanup — never a panic on cleanup failure (test
        // determinism must not depend on tmp cleanup).
        let _ = std::fs::remove_file(&path);
        assert_eq!(via_path.config(), via_bytes.config());
        assert_eq!(via_path.is_synthesized(), via_bytes.is_synthesized());
        // Both engines refuse to synthesise real text (synthesized-weight
        // loud-partial arm) — pin the message parity so downstream
        // callers see identical behaviour whichever loader they use.
        let n_q = via_path.config().n_q;
        let codes = vec![0u32; n_q];
        let e1 = via_path.transcribe(&codes).unwrap_err();
        let e2 = via_bytes.transcribe(&codes).unwrap_err();
        match (e1, e2) {
            (VokraError::NotImplemented(m1), VokraError::NotImplemented(m2)) => {
                assert_eq!(m1, m2, "from_path and from_gguf must yield identical arms");
            }
            (a, b) => panic!("expected two NotImplemented, got {a:?} / {b:?}"),
        }
    }

    /// `from_path` on a non-existent file surfaces
    /// [`VokraError::Io`] loudly (never a silent empty-string fabricated
    /// success).
    #[test]
    fn from_path_missing_file_returns_io_error() {
        let path = std::env::temp_dir().join(format!(
            "vokra-kyutai-stt-scout-does-not-exist-{}.gguf",
            std::process::id()
        ));
        // Ensure the path really is missing before the assertion runs.
        let _ = std::fs::remove_file(&path);
        let err = KyutaiSttAsr::from_path(&path).expect_err("missing file must be rejected");
        assert!(matches!(err, VokraError::Io(_)), "expected Io, got {err:?}");
    }
}
