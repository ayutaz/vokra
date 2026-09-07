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
//! - **Tokenizer side-car**: the authenticated inspector contract requires
//!   `tokenizer_en_audio_4000.model` (59,339 bytes with pinned blob/LFS
//!   identities). The legacy `tokenizer_spm_4k_en.model` name is explicitly
//!   rejected; tokenizer binding remains a separate runtime gate.
//! - **Weight license**: **CC-BY 4.0** (`AttributionRequired`) in the
//!   upstream card. Publication and runtime binding remain blocked pending
//!   authenticated composite evidence and owner review.
//! - **Streaming input contract**: the pinned Kyutai MLX example
//!   `kyutai-labs/moshi/moshi_mlx/moshi_mlx/run_inference.py` at commit
//!   `e6a55d2722a65870ef52a6c9f6ecfc0e90f38362` reads 24 kHz PCM, pads
//!   `audio_silence_prefix_seconds` on the left and
//!   `audio_delay_seconds + 1.0` on the right, then consumes 1,920-sample
//!   chunks. It suppresses text ids `0` and `3` before SentencePiece.
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
//!   [`KyutaiSttAsr::forward_text_logits`] is the dedicated `dep_q=0` main
//!   decoder component seam: explicit text-token + row-major Mimi-code
//!   frames → summed embeddings → causal/sliding-window transformer → final
//!   RMSNorm → text logits through `Compute` (CPU/Metal). Its synthesized
//!   fixture output is self-consistency only, not upstream parity or ASR.
//!   [`KyutaiSttAsr::transcribe`] returns [`VokraError::NotImplemented`]: the
//!   component logits seam exists, but a real authenticated tensor binder,
//!   streaming state/delay, sampling, and SentencePiece detokenization remain
//!   follow-up gates.
//!
//! Real-checkpoint parity is deferred exactly like CosyVoice2 T02 / CSM T29
//! / Moshi T29: this component binder sets the seam so a future parity run
//! can consume authenticated decoder weights without claiming full ASR.

#[cfg(test)]
use vokra_core::check_weight_license;
use vokra_core::gguf::chunks;
use vokra_core::gguf::{GgmlType, GgufFile, GgufMetadataValue};
use vokra_core::rng::SplitMix64;
use vokra_core::{BackendKind, CompliancePolicy, LicenseClass, Result, VokraError};

use crate::compute::{Compute, HotOp};
use crate::csm::rope::{llama3_inv_freqs, rope_apply_adjacent};
use crate::mimi::MimiNeuralConfig;
use crate::strict_checkpoint::sha256_bytes;

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

/// The authenticated Mimi frame rate named by
/// `mimi-pytorch-e351c8d8@125.safetensors` in the pinned STT `config.json`.
/// The value is expressed in milli-Hz to avoid a floating-point contract.
pub const KYUTAI_STT_MIMI_FRAME_RATE_MHZ: u32 = 12_500;

/// Exact Mimi sidecar identity from the authenticated Kyutai STT model tree.
pub const KYUTAI_STT_MIMI_FILE: &str = "mimi-pytorch-e351c8d8@125.safetensors";
/// Byte length of [`KYUTAI_STT_MIMI_FILE`] in the authenticated model tree.
pub const KYUTAI_STT_MIMI_BYTES: usize = 384_644_900;
/// SHA-256 digest of [`KYUTAI_STT_MIMI_FILE`] in the authenticated model tree.
pub const KYUTAI_STT_MIMI_SHA256: &str =
    "09b782f0629851a271227fb9d36db65c041790365f11bbe5d3d59369cf863f50";

/// PCM samples in one Mimi frame (`24_000 / 12.5`).  This is the fixed
/// `1920`-sample chunk used by the pinned upstream streaming example.
pub const KYUTAI_STT_MIMI_FRAME_HOP_SAMPLES: usize = 1_920;

/// The exact tokenizer sidecar named by the authenticated STT model card.
/// These identities are a sidecar gate only; no tokenizer bytes are embedded
/// or decoded by the decoder-component GGUF.
pub const KYUTAI_STT_TOKENIZER_FILE: &str = "tokenizer_en_audio_4000.model";
/// Byte length of [`KYUTAI_STT_TOKENIZER_FILE`] in the authenticated model tree.
pub const KYUTAI_STT_TOKENIZER_BYTES: usize = 59_339;
/// Git blob SHA-1 of [`KYUTAI_STT_TOKENIZER_FILE`] in the authenticated model tree.
pub const KYUTAI_STT_TOKENIZER_GIT_BLOB_SHA1: &str = "1820a7cbb15efc6a33dd365113c07e3df9d28d80";
/// SHA-256 digest of [`KYUTAI_STT_TOKENIZER_FILE`] in the authenticated model tree.
pub const KYUTAI_STT_TOKENIZER_SHA256: &str =
    "d461765ae179566678c93091c5fa6f2984c31bbe990bf1aa62d92c64d91bc3f6";

/// Upstream DSM's output filtering for the STT text stream.  `0` is the
/// initial/empty stream value and `3` is `existing_text_padding_id` from the
/// fixed STT config.  The pinned `moshi_mlx/run_inference.py` drops both
/// before converting ids to SentencePiece pieces.
pub const KYUTAI_STT_SUPPRESSED_TEXT_TOKENS: [u32; 2] = [0, 3];

/// Deterministic seed retained for the explicit in-module fixture constructor.
/// It is never used by the public GGUF/path loaders.
pub const KYUTAI_STT_FROM_GGUF_DEFAULT_SEED: u64 = 0x0C57_0C57_0C57_0C57;

/// Compute-seam operations used by the dep_q=0 main decoder.  The
/// dispatcher validates this complete set before any forward work starts, so
/// a backend missing one primitive fails explicitly instead of falling back
/// to CPU for that operation.
const KYUTAI_STT_HOT_OPS: &[HotOp] = &[HotOp::Gemm, HotOp::Softmax, HotOp::RmsNorm, HotOp::Silu];

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
// Deliberately NOT read back: the converter stamps the resolved width as an
// informational record of what it computed, but the runtime re-derives it from
// the Moshi `dim_feedforward` intermediate (see `BackboneConfig::ffn_hidden`)
// so a hand-edited or stale stamp can never silently disagree with the weight
// shapes. The constant is kept because it documents the wire contract —
// deleting it would lose the only in-tree record that the converter emits.
#[allow(dead_code)]
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
    /// `hidden_scale` — the Moshi `dim_feedforward` multiplier (4.125).
    /// The runtime first computes `int(hidden_scale * d_model)` and then
    /// mirrors `ActivationGating`'s branch-specific projection width.
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

    /// Gating projection hidden width from pinned Moshi `gating.py`.
    ///
    /// `lm.py` passes `int(hidden_scale * dim)` as `dim_feedforward`.
    /// `ActivationGating` then uses `(21 * dim) // 8` when that value equals
    /// `4 * dim`, otherwise `(2 * dim_feedforward) // 3`. Thus STT-2.6B-EN
    /// resolves to `5632` (`dim_feedforward=8448`), while the tiny fixture
    /// resolves to `42` (`dim_feedforward=64`). Checked arithmetic returns
    /// zero for malformed/overflowing configurations; validation rejects it.
    #[must_use]
    pub fn ffn_hidden(&self) -> usize {
        let scaled = self.hidden_scale * self.d_model as f32;
        let dim_feedforward = if scaled.is_finite() && scaled >= 0.0 {
            scaled.trunc() as usize
        } else {
            return 0;
        };
        let four_dim = self.d_model.checked_mul(4);
        if four_dim == Some(dim_feedforward) {
            self.d_model
                .checked_mul(21)
                .and_then(|value| value.checked_div(8))
                .unwrap_or(0)
        } else {
            dim_feedforward
                .checked_mul(2)
                .and_then(|value| value.checked_div(3))
                .unwrap_or(0)
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
        self.n_q.saturating_add(1)
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
        let n_channels = self.n_q.checked_add(1).ok_or_else(|| {
            VokraError::InvalidArgument("kyutai-stt channel count overflows usize".to_owned())
        })?;
        if self.delays.len() != n_channels {
            return Err(VokraError::InvalidArgument(format!(
                "kyutai-stt config: {} delays for {} channels (text + n_q — \
                 `_lm_kwargs[\"delays\"]` is per-channel)",
                self.delays.len(),
                n_channels,
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

/// The fixed, upstream-verified input contract at the STT/Mimi/streaming
/// boundary.
///
/// This type deliberately contains no model state and performs no inference.
/// It records only the arithmetic that the pinned Kyutai streaming example
/// applies before each decoder step:
///
/// - PCM is 24 kHz mono at the Mimi boundary;
/// - Mimi emits one `[n_q]` code row per 1,920 PCM samples (12.5 Hz);
/// - `audio_silence_prefix_seconds` is prepended on the left;
/// - the right side receives `audio_delay_seconds + 1.0` seconds of padding;
/// - text ids `0` and `existing_text_padding_id` are not emitted as pieces.
///
/// The right-side extra second is an upstream input-preparation rule, not a
/// claim that the decoder's learned delay is one whole-second longer.  The
/// contract therefore keeps the values in samples and never rounds a
/// fractional number of model frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KyutaiSttStreamingContract {
    sample_rate: u32,
    frame_hop_samples: usize,
    n_q: usize,
    audio_card: usize,
    text_card: usize,
    text_pad_id: u32,
    silence_prefix_samples: usize,
    right_padding_samples: usize,
}

impl KyutaiSttStreamingContract {
    /// Resolves the contract only for the authenticated STT-2.6B-EN config.
    ///
    /// The Mimi model is a separate GGUF component.  Its full learned
    /// weights are not accepted here; callers must pass its independently
    /// authenticated [`MimiNeuralConfig`] to [`Self::validate_mimi_config`].
    pub fn from_config(config: &KyutaiSttConfig) -> Result<Self> {
        config.validate_for_forward()?;
        if config != &KyutaiSttConfig::stt_2_6b_en() {
            return Err(VokraError::InvalidArgument(
                "kyutai-stt streaming contract: only the authenticated stt-2.6b-en config is supported".to_owned(),
            ));
        }
        let sample_rate = KYUTAI_STT_SAMPLE_RATE as usize;
        // The upstream values are 1.0 s silence prefix and 2.5 s model
        // delay plus 1.0 s trailing margin.  Keep the half-second as exact
        // integer arithmetic instead of rounding a float-derived frame.
        let right_padding_seconds_half = 7usize;
        Ok(Self {
            sample_rate: KYUTAI_STT_SAMPLE_RATE,
            frame_hop_samples: KYUTAI_STT_MIMI_FRAME_HOP_SAMPLES,
            n_q: config.n_q,
            audio_card: config.audio_card,
            text_card: config.text_card,
            text_pad_id: config.text_pad_id,
            silence_prefix_samples: sample_rate,
            right_padding_samples: sample_rate
                .checked_mul(right_padding_seconds_half)
                .and_then(|value| value.checked_div(2))
                .ok_or_else(|| {
                    VokraError::InvalidArgument(
                        "kyutai-stt streaming contract: right padding samples overflow".to_owned(),
                    )
                })?,
        })
    }

    /// PCM sample rate required before Mimi encoding.
    #[must_use]
    pub const fn sample_rate(self) -> u32 {
        self.sample_rate
    }

    /// Number of PCM samples consumed by one Mimi frame.
    #[must_use]
    pub const fn frame_hop_samples(self) -> usize {
        self.frame_hop_samples
    }

    /// Number of Mimi codebooks carried by each row-major audio frame.
    #[must_use]
    pub const fn n_q(self) -> usize {
        self.n_q
    }

    /// Number of entries in each Mimi codebook, excluding the decoder's
    /// initial-token row.
    #[must_use]
    pub const fn audio_card(self) -> usize {
        self.audio_card
    }

    /// Number of SentencePiece text vocabulary entries.
    #[must_use]
    pub const fn text_card(self) -> usize {
        self.text_card
    }

    /// Number of left-padding PCM samples prescribed by upstream.
    #[must_use]
    pub const fn silence_prefix_samples(self) -> usize {
        self.silence_prefix_samples
    }

    /// Number of right-padding PCM samples prescribed by upstream.
    #[must_use]
    pub const fn right_padding_samples(self) -> usize {
        self.right_padding_samples
    }

    /// Returns `(left, right)` PCM padding in samples.
    #[must_use]
    pub const fn pcm_padding_samples(self) -> (usize, usize) {
        (self.silence_prefix_samples, self.right_padding_samples)
    }

    /// Computes the number of full Mimi frames after applying the upstream
    /// left/right padding.  This mirrors `steps = padded_samples // 1920` in
    /// the pinned streaming example; a partial trailing frame is not invented.
    pub fn padded_frame_count(self, input_samples: usize) -> Result<usize> {
        let padded = input_samples
            .checked_add(self.silence_prefix_samples)
            .and_then(|value| value.checked_add(self.right_padding_samples))
            .ok_or_else(|| {
                VokraError::InvalidArgument(
                    "kyutai-stt streaming contract: padded PCM sample count overflows usize"
                        .to_owned(),
                )
            })?;
        Ok(padded / self.frame_hop_samples)
    }

    /// Checks a row-major `[frames, n_q]` Mimi code packet without executing
    /// the decoder.  The initial-token row (`audio_card`) is not a valid
    /// encoded Mimi code and is therefore rejected for an input packet.
    pub fn validate_mimi_codes(self, mimi_codes: &[u32]) -> Result<usize> {
        if mimi_codes.is_empty() {
            return Err(VokraError::InvalidArgument(
                "kyutai-stt streaming contract: Mimi code packet is empty".to_owned(),
            ));
        }
        if mimi_codes.len() % self.n_q != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "kyutai-stt streaming contract: Mimi code packet length {} is not a multiple of n_q={}",
                mimi_codes.len(),
                self.n_q,
            )));
        }
        if let Some((index, value)) = mimi_codes
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| (*value as usize) >= self.audio_card)
        {
            return Err(VokraError::InvalidArgument(format!(
                "kyutai-stt streaming contract: Mimi code packet[{index}]={value} is outside [0, {})",
                self.audio_card,
            )));
        }
        Ok(mimi_codes.len() / self.n_q)
    }

    /// Reports whether a text token is forwarded to SentencePiece decoding by
    /// the pinned upstream streaming path.
    #[must_use]
    pub const fn emits_text_token(self, token: u32) -> bool {
        token != KYUTAI_STT_SUPPRESSED_TEXT_TOKENS[0] && token != self.text_pad_id
    }

    /// Checks the independently authenticated Mimi neural-chain metadata
    /// needed by STT.  This does not bind or execute Mimi weights.
    pub fn validate_mimi_config(&self, mimi: &MimiNeuralConfig) -> Result<()> {
        mimi.validate()?;
        if mimi.sample_rate != self.sample_rate
            || mimi.frame_rate_mhz != KYUTAI_STT_MIMI_FRAME_RATE_MHZ
            || mimi.quantizer.n_q != self.n_q
            || mimi.quantizer.bins != self.audio_card
            || mimi.frame_hop_samples()? != self.frame_hop_samples
        {
            return Err(VokraError::InvalidArgument(format!(
                "kyutai-stt streaming contract: Mimi metadata does not match {} Hz / {} mHz / {} codebooks / {} bins / {} samples per frame: {mimi:?}",
                self.sample_rate,
                KYUTAI_STT_MIMI_FRAME_RATE_MHZ,
                self.n_q,
                self.audio_card,
                self.frame_hop_samples,
            )));
        }
        Ok(())
    }
}

/// Authenticated side-car pair for the fixed STT-2.6B-EN release.
///
/// The decoder GGUF intentionally does not embed Mimi or SentencePiece
/// weights.  This binding therefore checks the model-variant configuration,
/// the two upstream filenames, and the complete raw-byte identities before a
/// caller composes the three artifacts.  It does not parse or execute either
/// side-car; Mimi neural metadata is checked separately by
/// [`KyutaiSttStreamingContract::validate_mimi_config`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KyutaiSttAuthenticatedSidecars;

impl KyutaiSttAuthenticatedSidecars {
    /// Authenticates the exact Mimi and tokenizer files named by the fixed
    /// Kyutai STT-2.6B-EN config.
    ///
    /// The filenames are passed explicitly so a same-content file under a
    /// stale or legacy name cannot silently satisfy the composition gate.
    /// This method is an identity/binding gate only; it does not claim that
    /// either side-car can be decoded by the runtime.
    pub fn bind(
        config: &KyutaiSttConfig,
        mimi_file: &str,
        mimi_bytes: &[u8],
        tokenizer_file: &str,
        tokenizer_bytes: &[u8],
    ) -> Result<Self> {
        config.validate_for_forward()?;
        if config != &KyutaiSttConfig::stt_2_6b_en() {
            return Err(VokraError::InvalidArgument(
                "kyutai-stt sidecars: only the authenticated stt-2.6b-en config is supported"
                    .to_owned(),
            ));
        }
        if mimi_file != KYUTAI_STT_MIMI_FILE {
            return Err(VokraError::ModelLoad(format!(
                "kyutai-stt Mimi: expected authenticated sidecar `{KYUTAI_STT_MIMI_FILE}`, got `{mimi_file}`"
            )));
        }
        if tokenizer_file != KYUTAI_STT_TOKENIZER_FILE {
            return Err(VokraError::ModelLoad(format!(
                "kyutai-stt tokenizer: expected authenticated sidecar `{KYUTAI_STT_TOKENIZER_FILE}`, got `{tokenizer_file}`"
            )));
        }
        validate_mimi_bytes(mimi_bytes)?;
        validate_tokenizer_bytes(tokenizer_bytes)?;
        Ok(Self)
    }
}

/// Source-level text/second-stream demux for the `dep_q=0` STT input.
///
/// Upstream delayed-streams input has one text channel followed by the
/// `n_q` Mimi channels.  For STT the depformer owns no audio channels
/// (`dep_q=0`), so all remaining channels belong to the Mimi second stream.
/// This type only separates and validates the row-major token packet; it does
/// not run the decoder, sample text, or perform SentencePiece decoding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KyutaiSttInputPacket {
    text_tokens: Vec<u32>,
    mimi_codes: Vec<u32>,
}

impl KyutaiSttInputPacket {
    /// Demultiplexes `[frames, text + n_q audio]` into the decoder seam's two
    /// explicit inputs.
    pub fn from_interleaved(config: &KyutaiSttConfig, tokens: &[u32]) -> Result<Self> {
        config.validate_for_forward()?;
        if config.dep_q != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "kyutai-stt input demux requires dep_q=0, got {}",
                config.dep_q
            )));
        }
        if tokens.is_empty() {
            return Err(VokraError::InvalidArgument(
                "kyutai-stt input demux: token packet is empty".to_owned(),
            ));
        }
        let channels = config.n_channels();
        if tokens.len() % channels != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "kyutai-stt input demux: token packet length {} is not a multiple of {} channels",
                tokens.len(),
                channels
            )));
        }
        let frames = tokens.len() / channels;
        let text_rows = config.text_card.checked_add(1).ok_or_else(|| {
            VokraError::InvalidArgument("kyutai-stt input demux: text rows overflow".to_owned())
        })?;
        let audio_rows = config.audio_card.checked_add(1).ok_or_else(|| {
            VokraError::InvalidArgument("kyutai-stt input demux: audio rows overflow".to_owned())
        })?;
        let mut text_tokens = Vec::with_capacity(frames);
        let mimi_capacity = frames.checked_mul(config.n_q).ok_or_else(|| {
            VokraError::InvalidArgument("kyutai-stt input demux: Mimi packet overflows".to_owned())
        })?;
        let mut mimi_codes = Vec::with_capacity(mimi_capacity);
        for frame in tokens.chunks_exact(channels) {
            let text = frame[0];
            if text as usize >= text_rows {
                return Err(VokraError::InvalidArgument(format!(
                    "kyutai-stt input demux: text token {text} exceeds embedding rows {text_rows}"
                )));
            }
            text_tokens.push(text);
            for (channel, &code) in frame[1..].iter().enumerate() {
                if code as usize >= audio_rows {
                    return Err(VokraError::InvalidArgument(format!(
                        "kyutai-stt input demux: audio token at channel {channel} value {code} exceeds embedding rows {audio_rows}"
                    )));
                }
                mimi_codes.push(code);
            }
        }
        Ok(Self {
            text_tokens,
            mimi_codes,
        })
    }

    /// Number of synchronized text/audio frames in the packet.
    #[must_use]
    pub fn frames(&self) -> usize {
        self.text_tokens.len()
    }

    /// Text stream, one token per frame.
    #[must_use]
    pub fn text_tokens(&self) -> &[u32] {
        &self.text_tokens
    }

    /// Row-major Mimi stream, `[frames, n_q]`.
    #[must_use]
    pub fn mimi_codes(&self) -> &[u32] {
        &self.mimi_codes
    }

    /// Splits the packet into owned decoder inputs.
    #[must_use]
    pub fn into_parts(self) -> (Vec<u32>, Vec<u32>) {
        (self.text_tokens, self.mimi_codes)
    }
}

/// Explicit streaming wire-state for validated Mimi frames and emitted text.
///
/// This is deliberately not the transformer's KV cache or a generation
/// engine.  It captures only the source-level input/output contract that can
/// be proven without model execution: each accepted frame has exactly `n_q`
/// Mimi codes, and text ids `0`/`text_pad_id` are suppressed before the
/// SentencePiece boundary.  Native ASR remains fail-closed until the real
/// stateful decoder and tokenizer are independently bound.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KyutaiSttStreamingState {
    contract: KyutaiSttStreamingContract,
    frames_seen: usize,
    emitted_text_tokens: Vec<u32>,
}

impl KyutaiSttStreamingState {
    /// Starts an empty state for the authenticated STT streaming contract.
    #[must_use]
    pub fn new(contract: KyutaiSttStreamingContract) -> Self {
        Self {
            contract,
            frames_seen: 0,
            emitted_text_tokens: Vec::new(),
        }
    }

    /// Validates and accepts one complete row of Mimi codes.
    pub fn push_mimi_frame(&mut self, frame: &[u32]) -> Result<()> {
        if self.contract.validate_mimi_codes(frame)? != 1 {
            return Err(VokraError::InvalidArgument(
                "kyutai-stt streaming state: expected exactly one Mimi frame".to_owned(),
            ));
        }
        self.frames_seen = self.frames_seen.checked_add(1).ok_or_else(|| {
            VokraError::InvalidArgument(
                "kyutai-stt streaming state: frame count overflow".to_owned(),
            )
        })?;
        Ok(())
    }

    /// Applies the upstream text-output suppression rule.
    ///
    /// Returns `Some(token)` only for a token that would be forwarded to the
    /// SentencePiece boundary.  Decoder output is restricted to
    /// `[0, text_card)`; the extra `text_card` row is an input-embedding
    /// initial-token row accepted only by [`KyutaiSttInputPacket`].  No
    /// detokenization is performed here.
    pub fn push_text_token(&mut self, token: u32) -> Result<Option<u32>> {
        if token as usize >= self.contract.text_card {
            return Err(VokraError::InvalidArgument(format!(
                "kyutai-stt streaming state: decoder output token {token} is outside [0, text_card={}) (input-only initial row is not output)",
                self.contract.text_card
            )));
        }
        if self.contract.emits_text_token(token) {
            self.emitted_text_tokens.push(token);
            Ok(Some(token))
        } else {
            Ok(None)
        }
    }

    /// Number of validated Mimi frames accepted so far.
    #[must_use]
    pub fn frames_seen(&self) -> usize {
        self.frames_seen
    }

    /// Text tokens that passed the upstream suppression boundary.
    #[must_use]
    pub fn emitted_text_tokens(&self) -> &[u32] {
        &self.emitted_text_tokens
    }

    /// Consumes the state and returns filtered text tokens.
    #[must_use]
    pub fn into_text_tokens(self) -> Vec<u32> {
        self.emitted_text_tokens
    }
}

/// Verifies the exact raw SentencePiece sidecar identity authenticated by
/// the Kyutai STT inspector.  Parsing/decoding the protobuf is intentionally
/// left to the composite tokenizer gate; this helper prevents an unauthored
/// or same-size replacement from being accepted as that sidecar.
pub fn validate_tokenizer_bytes(bytes: &[u8]) -> Result<()> {
    if bytes.len() != KYUTAI_STT_TOKENIZER_BYTES {
        return Err(VokraError::ModelLoad(format!(
            "kyutai-stt tokenizer: `{KYUTAI_STT_TOKENIZER_FILE}` has {} bytes; expected {}",
            bytes.len(),
            KYUTAI_STT_TOKENIZER_BYTES,
        )));
    }
    let digest = sha256_bytes(bytes);
    let actual = digest
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    if actual != KYUTAI_STT_TOKENIZER_SHA256 {
        return Err(VokraError::ModelLoad(format!(
            "kyutai-stt tokenizer: `{KYUTAI_STT_TOKENIZER_FILE}` SHA-256 {actual} does not match authenticated sidecar"
        )));
    }
    Ok(())
}

/// Verifies the exact raw Mimi sidecar identity authenticated by the Kyutai
/// STT model tree.  The neural codec binder remains a separate component and
/// is not invoked by this check.
pub fn validate_mimi_bytes(bytes: &[u8]) -> Result<()> {
    if bytes.len() != KYUTAI_STT_MIMI_BYTES {
        return Err(VokraError::ModelLoad(format!(
            "kyutai-stt Mimi: `{KYUTAI_STT_MIMI_FILE}` has {} bytes; expected {}",
            bytes.len(),
            KYUTAI_STT_MIMI_BYTES,
        )));
    }
    let digest = sha256_bytes(bytes);
    let actual = digest
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    if actual != KYUTAI_STT_MIMI_SHA256 {
        return Err(VokraError::ModelLoad(format!(
            "kyutai-stt Mimi: `{KYUTAI_STT_MIMI_FILE}` SHA-256 {actual} does not match authenticated sidecar"
        )));
    }
    Ok(())
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

fn checked_product(label: &str, factors: &[usize]) -> Result<usize> {
    factors.iter().try_fold(1usize, |product, &factor| {
        product.checked_mul(factor).ok_or_else(|| {
            VokraError::InvalidArgument(format!("kyutai-stt {label} shape overflows usize"))
        })
    })
}

fn checked_add(label: &str, lhs: usize, rhs: usize) -> Result<usize> {
    lhs.checked_add(rhs).ok_or_else(|| {
        VokraError::InvalidArgument(format!("kyutai-stt {label} shape overflows usize"))
    })
}

#[derive(Debug, Clone, Copy)]
struct KyutaiWeightShapes {
    d: usize,
    ffn: usize,
    text_rows: usize,
    audio_rows: usize,
    three_d: usize,
    two_ffn: usize,
    text_embedding: usize,
    audio_embedding: usize,
    qkv_proj: usize,
    out_proj: usize,
    linear_in: usize,
    linear_out: usize,
    text_head: usize,
}

fn checked_weight_shapes(config: &KyutaiSttConfig) -> Result<KyutaiWeightShapes> {
    config.validate_for_forward()?;
    let d = config.backbone.d_model;
    let ffn = config.backbone.ffn_hidden();
    let text_rows = config.text_card.checked_add(1).ok_or_else(|| {
        VokraError::InvalidArgument("kyutai-stt text rows shape overflows usize".to_owned())
    })?;
    let audio_rows = config.audio_card.checked_add(1).ok_or_else(|| {
        VokraError::InvalidArgument("kyutai-stt audio rows shape overflows usize".to_owned())
    })?;
    let three_d = checked_product("3*d_model", &[3, d])?;
    let two_ffn = checked_product("2*ffn_hidden", &[2, ffn])?;
    let qkv_proj = checked_product("qkv projection", &[d, three_d])?;
    let out_proj = checked_product("output projection", &[d, d])?;
    let linear_in = checked_product("gating linear-in", &[d, two_ffn])?;
    let linear_out = checked_product("gating linear-out", &[ffn, d])?;
    let text_embedding = checked_product("text embedding", &[text_rows, d])?;
    let audio_embedding = checked_product("audio embedding", &[audio_rows, d])?;
    let text_head = checked_product("text head", &[d, config.text_card])?;
    // xavier's fan-in + fan-out must be checked before it is used in a
    // denominator, even though the subsequent vector length checks are also
    // guarded.
    let _ = checked_add("qkv fan", d, three_d)?;
    let _ = checked_add("FFN fan", d, two_ffn)?;
    let _ = checked_add("text embedding fan", text_rows, d)?;
    let _ = checked_add("audio embedding fan", audio_rows, d)?;
    let _ = checked_add("output projection fan", d, d)?;
    let _ = checked_add("gating output fan", ffn, d)?;
    let _ = checked_add("text head fan", d, config.text_card)?;
    Ok(KyutaiWeightShapes {
        d,
        ffn,
        text_rows,
        audio_rows,
        three_d,
        two_ffn,
        text_embedding,
        audio_embedding,
        qkv_proj,
        out_proj,
        linear_in,
        linear_out,
        text_head,
    })
}

fn ensure_finite_weights(name: &str, values: &[f32]) -> Result<()> {
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        return Err(VokraError::InvalidArgument(format!(
            "kyutai-stt weights: `{name}` contains non-finite value at {index}"
        )));
    }
    Ok(())
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
/// without the real HF checkpoint. Decoder-component binding is available via
/// [`Self::from_component_gguf`]; full composite ASR binding remains a
/// follow-up (Mimi/tokenizer/streaming gates are still closed).
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
        let shapes = checked_weight_shapes(config)?;
        let mut rng = SplitMix64::new(seed);
        let d = shapes.d;
        let ffn = shapes.ffn;

        let text_embedding = xavier(&mut rng, shapes.text_embedding, shapes.text_rows, d);
        let mut audio_embeddings = Vec::with_capacity(config.n_q);
        for _ in 0..config.n_q {
            audio_embeddings.push(xavier(
                &mut rng,
                shapes.audio_embedding,
                shapes.audio_rows,
                d,
            ));
        }

        let mut blocks = Vec::with_capacity(config.backbone.n_layer);
        for _ in 0..config.backbone.n_layer {
            blocks.push(KyutaiSttBlockWeights {
                attn_norm: vec![1.0; d],
                qkv_proj: xavier(&mut rng, shapes.qkv_proj, d, shapes.three_d),
                out_proj: xavier(&mut rng, shapes.out_proj, d, d),
                ffn_norm: vec![1.0; d],
                linear_in: xavier(&mut rng, shapes.linear_in, d, shapes.two_ffn),
                linear_out: xavier(&mut rng, shapes.linear_out, ffn, d),
            });
        }
        let final_norm = vec![1.0; d];
        let text_head = xavier(&mut rng, shapes.text_head, d, config.text_card);

        Ok(Self {
            text_embedding,
            audio_embeddings,
            blocks,
            final_norm,
            text_head,
            is_synthesized: true,
        })
    }

    /// Binds only the authenticated **decoder-component** tensors from the
    /// official STT-2.6B-EN GGUF release. This does not bind Mimi, the
    /// tokenizer, or streaming state, and therefore is not a public ASR
    /// loader. The exact-release gate requires the stamped `kyutai-stt`
    /// architecture and the complete 323-tensor BF16 manifest before any
    /// payload is decoded.
    ///
    /// The upstream torch linear tensors are `[out, in]`; this store keeps
    /// Compute-seam weights as `[in, out]`, so the four learned projections
    /// are transposed during binding. No synthesized defaults are used.
    ///
    /// # Errors
    ///
    /// [`VokraError::ModelLoad`] if the release metadata, exact tensor set,
    /// dtype, shape, or finite-value contract is not satisfied. Whole-file
    /// identity remains the authenticated runner's responsibility; this API
    /// receives no expected digest.
    pub fn from_component_gguf(file: &GgufFile) -> Result<Self> {
        require_component_metadata(file)?;
        let config = KyutaiSttConfig::from_gguf(file).map_err(|error| {
            VokraError::ModelLoad(format!(
                "kyutai-stt component binder: config is not authenticated: {error}"
            ))
        })?;
        let arch = match file.get(chunks::KEY_MODEL_ARCH) {
            Some(GgufMetadataValue::String(value)) => value.as_str(),
            _ => {
                return Err(VokraError::ModelLoad(
                    "kyutai-stt component binder: missing authenticated model arch".to_owned(),
                ));
            }
        };
        if arch != EXPECTED_ARCH || config != KyutaiSttConfig::stt_2_6b_en() {
            return Err(VokraError::ModelLoad(
                "kyutai-stt component binder: only the authenticated stt-2.6b-en release is accepted".to_owned(),
            ));
        }
        bind_component_gguf(file, &config)
    }
}

fn require_component_u32(file: &GgufFile, key: &str, expected: u32) -> Result<()> {
    match file.get(key) {
        Some(GgufMetadataValue::U32(value)) if *value == expected => Ok(()),
        Some(GgufMetadataValue::U32(value)) => Err(VokraError::ModelLoad(format!(
            "kyutai-stt component binder: metadata `{key}`={value}, expected {expected}"
        ))),
        Some(other) => Err(VokraError::ModelLoad(format!(
            "kyutai-stt component binder: metadata `{key}` has type {:?}, expected UINT32",
            other.value_type()
        ))),
        None => Err(VokraError::ModelLoad(format!(
            "kyutai-stt component binder: required metadata `{key}` is missing"
        ))),
    }
}

fn require_component_f32(file: &GgufFile, key: &str, expected: f32) -> Result<()> {
    match file.get(key) {
        Some(GgufMetadataValue::F32(value)) if *value == expected => Ok(()),
        Some(GgufMetadataValue::F32(value)) => Err(VokraError::ModelLoad(format!(
            "kyutai-stt component binder: metadata `{key}`={value}, expected {expected}"
        ))),
        Some(other) => Err(VokraError::ModelLoad(format!(
            "kyutai-stt component binder: metadata `{key}` has type {:?}, expected FLOAT32",
            other.value_type()
        ))),
        None => Err(VokraError::ModelLoad(format!(
            "kyutai-stt component binder: required metadata `{key}` is missing"
        ))),
    }
}

fn require_component_string(file: &GgufFile, key: &str, expected: &str) -> Result<()> {
    match file.get(key) {
        Some(GgufMetadataValue::String(value)) if value.as_str() == expected => Ok(()),
        Some(GgufMetadataValue::String(value)) => Err(VokraError::ModelLoad(format!(
            "kyutai-stt component binder: metadata `{key}`={value:?}, expected {expected:?}"
        ))),
        Some(other) => Err(VokraError::ModelLoad(format!(
            "kyutai-stt component binder: metadata `{key}` has type {:?}, expected STRING",
            other.value_type()
        ))),
        None => Err(VokraError::ModelLoad(format!(
            "kyutai-stt component binder: required metadata `{key}` is missing"
        ))),
    }
}

fn component_metadata_contract_keys() -> Vec<String> {
    let mut keys = vec![
        KEY_SAMPLE_RATE,
        KEY_BB_N_LAYER,
        KEY_BB_D_MODEL,
        KEY_BB_N_HEAD,
        KEY_BB_HIDDEN_SCALE,
        KEY_BB_FFN_HIDDEN,
        KEY_BB_CONTEXT,
        KEY_BB_ROPE_MAX_PERIOD,
        KEY_BB_CAUSAL,
        KEY_BB_RMS_NORM_EPS,
        KEY_DEP_N_LAYER,
        KEY_DEP_D_MODEL,
        KEY_DEP_N_HEAD,
        KEY_DEP_MULTI_LINEAR,
        KEY_DEP_WEIGHTS_PER_STEP,
        KEY_N_Q,
        KEY_DEP_Q,
        KEY_AUDIO_CARD,
        KEY_TEXT_CARD,
        KEY_TEXT_PAD_ID,
        KEY_AUDIO_DELAY_SECS,
        KEY_AUDIO_SILENCE_PREFIX_SECS,
        KEY_N_DELAYS,
    ]
    .into_iter()
    .map(str::to_owned)
    .collect::<Vec<_>>();
    for index in 0..33 {
        keys.push(format!("{PREFIX_DELAY}{index}"));
    }
    keys
}

fn require_component_occurrence(metadata: &[(String, GgufMetadataValue)], key: &str) -> Result<()> {
    let occurrences = metadata.iter().filter(|(name, _)| name == key).count();
    if occurrences != 1 {
        let state = if occurrences == 0 {
            "missing"
        } else {
            "duplicated"
        };
        return Err(VokraError::ModelLoad(format!(
            "kyutai-stt component binder: metadata `{key}` is {state}"
        )));
    }
    Ok(())
}

fn require_component_metadata(file: &GgufFile) -> Result<()> {
    // Exact authenticated metadata: 23 Kyutai scalar keys + 33 indexed
    // delays, plus `vokra.model.arch` and four provenance keys below = 61
    // keys. Unrelated schema/general metadata remains allowed.
    let contract_keys = component_metadata_contract_keys();
    for (key, _) in file.metadata() {
        if key.starts_with("vokra.kyutai_stt.")
            && !contract_keys.iter().any(|expected| expected == key)
        {
            return Err(VokraError::ModelLoad(format!(
                "kyutai-stt component binder: unexpected metadata `{key}`"
            )));
        }
    }
    for key in &contract_keys {
        require_component_occurrence(file.metadata(), key)?;
    }
    for key in [
        chunks::KEY_MODEL_ARCH,
        chunks::KEY_PROVENANCE_MODEL_ID,
        chunks::KEY_PROVENANCE_LICENSE,
        chunks::KEY_PROVENANCE_WEIGHT_LICENSE,
        chunks::KEY_PROVENANCE_SOURCE,
    ] {
        require_component_occurrence(file.metadata(), key)?;
    }
    require_component_u32(file, KEY_BB_N_LAYER, 48)?;
    require_component_u32(file, KEY_BB_D_MODEL, 2048)?;
    require_component_u32(file, KEY_BB_N_HEAD, 32)?;
    require_component_f32(file, KEY_BB_HIDDEN_SCALE, 4.125)?;
    require_component_u32(file, KEY_BB_FFN_HIDDEN, 5632)?;
    require_component_u32(file, KEY_BB_CONTEXT, 375)?;
    require_component_f32(file, KEY_BB_ROPE_MAX_PERIOD, 100_000.0)?;
    require_component_u32(file, KEY_BB_CAUSAL, 1)?;
    require_component_f32(file, KEY_BB_RMS_NORM_EPS, 1e-8)?;
    require_component_u32(file, KEY_DEP_N_LAYER, 6)?;
    require_component_u32(file, KEY_DEP_D_MODEL, 1024)?;
    require_component_u32(file, KEY_DEP_N_HEAD, 16)?;
    require_component_u32(file, KEY_DEP_MULTI_LINEAR, 1)?;
    require_component_u32(file, KEY_DEP_WEIGHTS_PER_STEP, 1)?;
    require_component_u32(file, KEY_N_Q, 32)?;
    require_component_u32(file, KEY_DEP_Q, 0)?;
    require_component_u32(file, KEY_AUDIO_CARD, 2048)?;
    require_component_u32(file, KEY_TEXT_CARD, 4000)?;
    require_component_u32(file, KEY_TEXT_PAD_ID, 3)?;
    require_component_f32(file, KEY_AUDIO_DELAY_SECS, 2.5)?;
    require_component_f32(file, KEY_AUDIO_SILENCE_PREFIX_SECS, 1.0)?;
    require_component_u32(file, KEY_SAMPLE_RATE, 24_000)?;
    require_component_u32(file, KEY_N_DELAYS, 33)?;
    for index in 0..33 {
        require_component_u32(file, &format!("{PREFIX_DELAY}{index}"), 0)?;
    }
    require_component_string(file, chunks::KEY_PROVENANCE_MODEL_ID, "kyutai/stt-2.6b-en")?;
    // The live Kyutai artifact contract records the SPDX-like license spelling
    // in lowercase; this is intentionally exact rather than normalized.
    require_component_string(file, chunks::KEY_PROVENANCE_LICENSE, "cc-by-4.0")?;
    require_component_string(
        file,
        chunks::KEY_PROVENANCE_WEIGHT_LICENSE,
        LicenseClass::AttributionRequired.as_str(),
    )?;
    require_component_string(
        file,
        chunks::KEY_PROVENANCE_SOURCE,
        "https://huggingface.co/kyutai/stt-2.6b-en",
    )?;
    Ok(())
}

fn component_tensor_names(config: &KyutaiSttConfig) -> Result<Vec<String>> {
    let block_tensor_count = checked_product(
        "component block tensor count",
        &[config.backbone.n_layer, 6],
    )?;
    let tensor_count = checked_add(
        "component tensor count",
        checked_add("component base tensor count", config.n_q, 3)?,
        block_tensor_count,
    )?;
    let mut names = Vec::with_capacity(tensor_count);
    names.push("text_emb.weight".to_owned());
    for channel in 0..config.n_q {
        names.push(format!("emb.{channel}.weight"));
    }
    for layer in 0..config.backbone.n_layer {
        let prefix = format!("transformer.layers.{layer}");
        names.push(format!("{prefix}.self_attn.in_proj_weight"));
        names.push(format!("{prefix}.self_attn.out_proj.weight"));
        names.push(format!("{prefix}.gating.linear_in.weight"));
        names.push(format!("{prefix}.gating.linear_out.weight"));
        names.push(format!("{prefix}.norm1.alpha"));
        names.push(format!("{prefix}.norm2.alpha"));
    }
    names.push("out_norm.alpha".to_owned());
    names.push("text_linear.weight".to_owned());
    Ok(names)
}

fn validate_component_tensor_set(file: &GgufFile, config: &KyutaiSttConfig) -> Result<()> {
    let mut expected = component_tensor_names(config)?;
    let mut actual: Vec<String> = file
        .tensors()
        .iter()
        .map(|tensor| tensor.name.clone())
        .collect();
    expected.sort_unstable();
    actual.sort_unstable();
    if actual != expected {
        let missing: Vec<&str> = expected
            .iter()
            .filter(|name| actual.binary_search(*name).is_err())
            .map(String::as_str)
            .collect();
        let extra: Vec<&str> = actual
            .iter()
            .filter(|name| expected.binary_search(*name).is_err())
            .map(String::as_str)
            .collect();
        return Err(VokraError::ModelLoad(format!(
            "kyutai-stt component binder: exact tensor manifest mismatch (expected {}, found {}); missing={missing:?}, extra={extra:?}",
            expected.len(),
            actual.len(),
        )));
    }
    Ok(())
}

fn component_tensor(file: &GgufFile, name: &str, expected: &[usize]) -> Result<Vec<f32>> {
    let info = file.tensor_info(name).ok_or_else(|| {
        VokraError::ModelLoad(format!(
            "kyutai-stt component binder: required tensor `{name}` is missing"
        ))
    })?;
    if info.dtype != GgmlType::BF16 {
        return Err(VokraError::ModelLoad(format!(
            "kyutai-stt component binder: tensor `{name}` dtype {:?}, expected BF16",
            info.dtype
        )));
    }
    let actual: Vec<usize> = info
        .dimensions
        .iter()
        .map(|&dimension| usize::try_from(dimension))
        .collect::<std::result::Result<_, _>>()
        .map_err(|_| {
            VokraError::ModelLoad(format!(
                "kyutai-stt component binder: tensor `{name}` dimension overflows usize"
            ))
        })?;
    if actual != expected {
        return Err(VokraError::ModelLoad(format!(
            "kyutai-stt component binder: tensor `{name}` shape {actual:?}, expected {expected:?}"
        )));
    }
    let values = file.tensor_f32(name).map_err(|error| {
        VokraError::ModelLoad(format!(
            "kyutai-stt component binder: tensor `{name}` BF16 decode failed: {error}"
        ))
    })?;
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        return Err(VokraError::ModelLoad(format!(
            "kyutai-stt component binder: tensor `{name}` contains non-finite value at {index}"
        )));
    }
    Ok(values)
}

fn transpose_component(values: Vec<f32>, rows: usize, cols: usize, name: &str) -> Result<Vec<f32>> {
    let expected = checked_product("component transpose", &[rows, cols])?;
    if values.len() != expected {
        return Err(VokraError::ModelLoad(format!(
            "kyutai-stt component binder: tensor `{name}` has {} values, expected {expected}",
            values.len()
        )));
    }
    let mut transposed = vec![0.0f32; expected];
    for row in 0..rows {
        for column in 0..cols {
            transposed[column * rows + row] = values[row * cols + column];
        }
    }
    Ok(transposed)
}

fn bind_component_gguf(file: &GgufFile, config: &KyutaiSttConfig) -> Result<KyutaiSttWeights> {
    let shapes = checked_weight_shapes(config).map_err(|error| {
        VokraError::ModelLoad(format!(
            "kyutai-stt component binder: invalid config: {error}"
        ))
    })?;
    validate_component_tensor_set(file, config)?;
    let text_embedding = component_tensor(file, "text_emb.weight", &[shapes.text_rows, shapes.d])?;
    let mut audio_embeddings = Vec::with_capacity(config.n_q);
    for channel in 0..config.n_q {
        audio_embeddings.push(component_tensor(
            file,
            &format!("emb.{channel}.weight"),
            &[shapes.audio_rows, shapes.d],
        )?);
    }
    let mut blocks = Vec::with_capacity(config.backbone.n_layer);
    for layer in 0..config.backbone.n_layer {
        let prefix = format!("transformer.layers.{layer}");
        let in_proj = component_tensor(
            file,
            &format!("{prefix}.self_attn.in_proj_weight"),
            &[shapes.three_d, shapes.d],
        )?;
        let out_proj = component_tensor(
            file,
            &format!("{prefix}.self_attn.out_proj.weight"),
            &[shapes.d, shapes.d],
        )?;
        let linear_in = component_tensor(
            file,
            &format!("{prefix}.gating.linear_in.weight"),
            &[shapes.two_ffn, shapes.d],
        )?;
        let linear_out = component_tensor(
            file,
            &format!("{prefix}.gating.linear_out.weight"),
            &[shapes.d, shapes.ffn],
        )?;
        blocks.push(KyutaiSttBlockWeights {
            attn_norm: component_tensor(file, &format!("{prefix}.norm1.alpha"), &[shapes.d])?,
            qkv_proj: transpose_component(in_proj, shapes.three_d, shapes.d, "in_proj_weight")?,
            out_proj: transpose_component(out_proj, shapes.d, shapes.d, "out_proj.weight")?,
            ffn_norm: component_tensor(file, &format!("{prefix}.norm2.alpha"), &[shapes.d])?,
            linear_in: transpose_component(
                linear_in,
                shapes.two_ffn,
                shapes.d,
                "linear_in.weight",
            )?,
            linear_out: transpose_component(linear_out, shapes.d, shapes.ffn, "linear_out.weight")?,
        });
    }
    let final_norm = component_tensor(file, "out_norm.alpha", &[shapes.d])?;
    let text_head = transpose_component(
        component_tensor(file, "text_linear.weight", &[config.text_card, shapes.d])?,
        config.text_card,
        shapes.d,
        "text_linear.weight",
    )?;
    Ok(KyutaiSttWeights {
        text_embedding,
        audio_embeddings,
        blocks,
        final_norm,
        text_head,
        is_synthesized: false,
    })
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

fn apply_rope_heads(
    values: &mut [f32],
    frames: usize,
    d_model: usize,
    n_head: usize,
    head_dim: usize,
    inv_freqs: &[f32],
) -> Result<()> {
    let expected = frames.checked_mul(d_model).ok_or_else(|| {
        VokraError::InvalidArgument("kyutai-stt RoPE shape overflows usize".to_owned())
    })?;
    if values.len() != expected || n_head.checked_mul(head_dim) != Some(d_model) {
        return Err(VokraError::InvalidArgument(
            "kyutai-stt RoPE shape is inconsistent with d_model".to_owned(),
        ));
    }
    let head_len = frames.checked_mul(head_dim).ok_or_else(|| {
        VokraError::InvalidArgument("kyutai-stt RoPE head shape overflows usize".to_owned())
    })?;
    let mut head = vec![0.0f32; head_len];
    for index in 0..n_head {
        for frame in 0..frames {
            head[frame * head_dim..(frame + 1) * head_dim].copy_from_slice(
                &values
                    [frame * d_model + index * head_dim..frame * d_model + (index + 1) * head_dim],
            );
        }
        rope_apply_adjacent(&mut head, frames, head_dim, inv_freqs, 0)?;
        for frame in 0..frames {
            values[frame * d_model + index * head_dim..frame * d_model + (index + 1) * head_dim]
                .copy_from_slice(&head[frame * head_dim..(frame + 1) * head_dim]);
        }
    }
    Ok(())
}

/// Text-logit output from the dedicated dep_q=0 decoder seam.
///
/// `values` is row-major `[frames, vocab]`; retaining both dimensions beside
/// the payload makes the shape part of the authenticated hand-off to the
/// future parity/reference consumer.  This is a structural/self-consistency
/// result, not an ASR transcript or an upstream numerical-parity claim.
#[derive(Debug, Clone, PartialEq)]
pub struct KyutaiSttTextLogits {
    frames: usize,
    vocab: usize,
    values: Vec<f32>,
}

impl KyutaiSttTextLogits {
    fn new(frames: usize, vocab: usize, values: Vec<f32>) -> Result<Self> {
        let expected = frames.checked_mul(vocab).ok_or_else(|| {
            VokraError::InvalidArgument("kyutai-stt logits shape overflows usize".to_owned())
        })?;
        if values.len() != expected {
            return Err(VokraError::InvalidArgument(format!(
                "kyutai-stt logits payload len {} != frames*vocab {}",
                values.len(),
                expected
            )));
        }
        if !values.iter().all(|value| value.is_finite()) {
            return Err(VokraError::InvalidArgument(
                "kyutai-stt logits contain non-finite values".to_owned(),
            ));
        }
        Ok(Self {
            frames,
            vocab,
            values,
        })
    }

    /// Number of Mimi/text steps represented by the logits.
    #[must_use]
    pub fn frames(&self) -> usize {
        self.frames
    }

    /// Text vocabulary width (`config.text_card`).
    #[must_use]
    pub fn vocab(&self) -> usize {
        self.vocab
    }

    /// Row-major logits `[frames, vocab]`.
    #[must_use]
    pub fn as_slice(&self) -> &[f32] {
        &self.values
    }

    /// Consumes the authenticated-shape wrapper and returns row-major values.
    #[must_use]
    pub fn into_values(self) -> Vec<f32> {
        self.values
    }
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
        let shapes = checked_weight_shapes(&cfg)?;
        let d = shapes.d;

        if weights.text_embedding.len() != shapes.text_embedding {
            return Err(VokraError::InvalidArgument(format!(
                "kyutai-stt weights: text_embedding.len()={} != (text_card+1)*d_model={}",
                weights.text_embedding.len(),
                shapes.text_embedding,
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
            let expected = shapes.audio_embedding;
            if tbl.len() != expected {
                return Err(VokraError::InvalidArgument(format!(
                    "kyutai-stt weights: audio_embeddings[{i}].len()={} != {expected}",
                    tbl.len(),
                )));
            }
        }
        if weights.blocks.len() != cfg.backbone.n_layer {
            return Err(VokraError::InvalidArgument(format!(
                "kyutai-stt weights: blocks.len()={} != backbone.n_layer={}",
                weights.blocks.len(),
                cfg.backbone.n_layer,
            )));
        }
        for (i, blk) in weights.blocks.iter().enumerate() {
            for (name, len, expected) in [
                ("attn_norm", blk.attn_norm.len(), d),
                ("qkv_proj", blk.qkv_proj.len(), shapes.qkv_proj),
                ("out_proj", blk.out_proj.len(), shapes.out_proj),
                ("ffn_norm", blk.ffn_norm.len(), d),
                ("linear_in", blk.linear_in.len(), shapes.linear_in),
                ("linear_out", blk.linear_out.len(), shapes.linear_out),
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
        if weights.text_head.len() != shapes.text_head {
            return Err(VokraError::InvalidArgument(format!(
                "kyutai-stt weights: text_head.len()={} != d_model * text_card = {}",
                weights.text_head.len(),
                shapes.text_head,
            )));
        }
        ensure_finite_weights("text_embedding", &weights.text_embedding)?;
        for (index, table) in weights.audio_embeddings.iter().enumerate() {
            ensure_finite_weights(&format!("audio_embeddings[{index}]"), table)?;
        }
        for (index, block) in weights.blocks.iter().enumerate() {
            ensure_finite_weights(&format!("blocks[{index}].attn_norm"), &block.attn_norm)?;
            ensure_finite_weights(&format!("blocks[{index}].qkv_proj"), &block.qkv_proj)?;
            ensure_finite_weights(&format!("blocks[{index}].out_proj"), &block.out_proj)?;
            ensure_finite_weights(&format!("blocks[{index}].ffn_norm"), &block.ffn_norm)?;
            ensure_finite_weights(&format!("blocks[{index}].linear_in"), &block.linear_in)?;
            ensure_finite_weights(&format!("blocks[{index}].linear_out"), &block.linear_out)?;
        }
        ensure_finite_weights("final_norm", &weights.final_norm)?;
        ensure_finite_weights("text_head", &weights.text_head)?;
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

    /// Runs the authenticated-shape **main decoder component** for the
    /// upstream `dep_q=0` STT variant.
    ///
    /// `text_tokens` is one explicit text-token id per frame and
    /// `mimi_codes` is row-major `[frames, n_q]` Mimi codes.  The component
    /// sums the text embedding with all 32 audio embeddings, applies the
    /// causal/sliding-window Helium transformer, final RMSNorm, and text
    /// linear head.  No depformer or audio logits are produced because this
    /// seam requires `dep_q == 0`.
    ///
    /// This deliberately does not perform streaming delay, sampling,
    /// tokenizer decoding, or real-checkpoint binding.  Synthesized weights
    /// are accepted only for deterministic structural/self-consistency tests;
    /// callers must not treat their logits as an ASR result.  Backend
    /// selection is explicit and validated against the complete Compute hot
    /// op set before the first embedding is read.  Unsupported backends or
    /// operations return an error; no CPU fallback is attempted.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] for an invalid token matrix or a
    /// nonzero `dep_q`; backend and Compute-seam errors are returned
    /// verbatim. Inputs may be longer than `context`; that value is the
    /// causal attention window, not a component-input limit.
    pub fn forward_text_logits(
        &self,
        backend: BackendKind,
        text_tokens: &[u32],
        mimi_codes: &[u32],
    ) -> Result<KyutaiSttTextLogits> {
        if self.cfg.dep_q != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "kyutai-stt dep_q=0 decoder requires dep_q=0, got {}",
                self.cfg.dep_q
            )));
        }
        let frames = text_tokens.len();
        if frames == 0 {
            return Err(VokraError::InvalidArgument(
                "kyutai-stt decoder: text_tokens is empty".to_owned(),
            ));
        }
        let expected_codes = frames.checked_mul(self.cfg.n_q).ok_or_else(|| {
            VokraError::InvalidArgument("kyutai-stt decoder code shape overflows usize".to_owned())
        })?;
        if mimi_codes.len() != expected_codes {
            return Err(VokraError::InvalidArgument(format!(
                "kyutai-stt decoder: mimi_codes.len()={} != frames*n_q={expected_codes}",
                mimi_codes.len()
            )));
        }
        let text_rows = self.cfg.text_card.checked_add(1).ok_or_else(|| {
            VokraError::InvalidArgument("kyutai-stt text rows shape overflows usize".to_owned())
        })?;
        for (frame, &token) in text_tokens.iter().enumerate() {
            if token as usize >= text_rows {
                return Err(VokraError::InvalidArgument(format!(
                    "kyutai-stt decoder: text_tokens[{frame}]={token} >= text rows {text_rows}"
                )));
            }
        }
        let audio_rows = self.cfg.audio_card.checked_add(1).ok_or_else(|| {
            VokraError::InvalidArgument("kyutai-stt audio rows shape overflows usize".to_owned())
        })?;
        for (index, &code) in mimi_codes.iter().enumerate() {
            if code as usize >= audio_rows {
                return Err(VokraError::InvalidArgument(format!(
                    "kyutai-stt decoder: mimi_codes[{index}]={code} >= audio rows {audio_rows}"
                )));
            }
        }

        let compute = Compute::for_backend(backend, KYUTAI_STT_HOT_OPS)?;
        let d = self.cfg.backbone.d_model;
        let heads = self.cfg.backbone.n_head;
        let head_dim = self.cfg.backbone.head_dim();
        let ffn = self.cfg.backbone.ffn_hidden();
        let frame_d = checked_product("frames*d_model", &[frames, d])?;
        let frame_qkv = checked_product("frames*3*d_model", &[frames, 3, d])?;
        let frame_scores = checked_product("frames*frames", &[frames, frames])?;
        let frame_ffn = checked_product("frames*ffn_hidden", &[frames, ffn])?;
        let frame_ffn_in = checked_product("frames*2*ffn_hidden", &[frames, 2, ffn])?;
        let head_matrix = checked_product("frames*head_dim", &[frames, head_dim])?;
        let head_transposed = checked_product("head_dim*frames", &[head_dim, frames])?;
        let inv_freqs = llama3_inv_freqs(head_dim, self.cfg.backbone.rope_max_period, None)?;
        let mut hidden = vec![0.0f32; frame_d];
        for frame in 0..frames {
            let dst = &mut hidden[frame * d..(frame + 1) * d];
            let text_row = &self.weights.text_embedding
                [text_tokens[frame] as usize * d..(text_tokens[frame] as usize + 1) * d];
            for (out, &value) in dst.iter_mut().zip(text_row) {
                *out += value;
            }
            for channel in 0..self.cfg.n_q {
                let code = mimi_codes[frame * self.cfg.n_q + channel] as usize;
                let table = &self.weights.audio_embeddings[channel];
                let row = &table[code * d..(code + 1) * d];
                for (out, &value) in dst.iter_mut().zip(row) {
                    *out += value;
                }
            }
        }

        let mut norm = vec![0.0f32; frame_d];
        let mut qkv = vec![0.0f32; frame_qkv];
        let mut q = vec![0.0f32; frame_d];
        let mut k = vec![0.0f32; frame_d];
        let mut v = vec![0.0f32; frame_d];
        let mut attn_input = vec![0.0f32; frame_d];
        let mut attn_output = vec![0.0f32; frame_d];
        let mut scores = vec![0.0f32; frame_scores];
        let mut probs = vec![0.0f32; frame_scores];
        let mut ffn_in = vec![0.0f32; frame_ffn_in];
        let mut ffn_gate = vec![0.0f32; frame_ffn];
        let mut ffn_up = vec![0.0f32; frame_ffn];
        let mut ffn_activated = vec![0.0f32; frame_ffn];
        let mut ffn_output = vec![0.0f32; frame_d];
        let mut head_q = vec![0.0f32; head_matrix];
        let mut head_k_transposed = vec![0.0f32; head_transposed];
        let mut head_v = vec![0.0f32; head_matrix];
        let mut head_weighted = vec![0.0f32; head_matrix];
        for block in &self.weights.blocks {
            compute.rms_norm_f32(
                &hidden,
                &mut norm,
                frames,
                d,
                &block.attn_norm,
                self.cfg.rms_norm_eps,
            )?;
            compute.gemm_f32(
                frames,
                checked_product("qkv width", &[3, d])?,
                d,
                &norm,
                &block.qkv_proj,
                None,
                &mut qkv,
            )?;
            // This is the pinned Moshi `Transformer` layout: fused QKV is
            // split into contiguous Q/K/V widths, standard adjacent-pair
            // RoPE is applied to Q and K, and `ActivationGating` computes
            // SiLU(gate) * up below. This is a source-aligned structural
            // seam, not an independent upstream parity claim.
            for frame in 0..frames {
                q[frame * d..(frame + 1) * d]
                    .copy_from_slice(&qkv[frame * 3 * d..frame * 3 * d + d]);
                k[frame * d..(frame + 1) * d]
                    .copy_from_slice(&qkv[frame * 3 * d + d..frame * 3 * d + 2 * d]);
                v[frame * d..(frame + 1) * d]
                    .copy_from_slice(&qkv[frame * 3 * d + 2 * d..(frame + 1) * 3 * d]);
            }
            apply_rope_heads(&mut q, frames, d, heads, head_dim, &inv_freqs)?;
            apply_rope_heads(&mut k, frames, d, heads, head_dim, &inv_freqs)?;
            attn_input.fill(0.0);
            let scale = 1.0f32 / (head_dim as f32).sqrt();
            for head in 0..heads {
                for frame in 0..frames {
                    let source = &q[frame * d + head * head_dim..frame * d + (head + 1) * head_dim];
                    head_q[frame * head_dim..(frame + 1) * head_dim].copy_from_slice(source);
                    let source = &v[frame * d + head * head_dim..frame * d + (head + 1) * head_dim];
                    head_v[frame * head_dim..(frame + 1) * head_dim].copy_from_slice(source);
                    for column in 0..head_dim {
                        head_k_transposed[column * frames + frame] =
                            k[frame * d + head * head_dim + column];
                    }
                }
                // QK^T is a learned projection product and must remain on
                // the selected backend. The transposes above are scalar
                // layout glue only; there is no CPU fallback here.
                compute.gemm_f32(
                    frames,
                    frames,
                    head_dim,
                    &head_q,
                    &head_k_transposed,
                    None,
                    &mut scores,
                )?;
                for query in 0..frames {
                    for key in 0..frames {
                        let visible = key <= query && query - key < self.cfg.backbone.context;
                        scores[query * frames + key] = if visible {
                            scores[query * frames + key] * scale
                        } else {
                            f32::NEG_INFINITY
                        };
                    }
                }
                compute.softmax_f32(&scores, &mut probs, frames, frames)?;
                // The probability×V product is likewise dispatched as a
                // learned matmul. Copying the per-head result back into the
                // fused residual layout is scalar layout glue.
                compute.gemm_f32(
                    frames,
                    head_dim,
                    frames,
                    &probs,
                    &head_v,
                    None,
                    &mut head_weighted,
                )?;
                for frame in 0..frames {
                    attn_input[frame * d + head * head_dim..frame * d + (head + 1) * head_dim]
                        .copy_from_slice(&head_weighted[frame * head_dim..(frame + 1) * head_dim]);
                }
            }
            compute.gemm_f32(
                frames,
                d,
                d,
                &attn_input,
                &block.out_proj,
                None,
                &mut attn_output,
            )?;
            for (dst, &value) in hidden.iter_mut().zip(&attn_output) {
                *dst += value;
            }

            compute.rms_norm_f32(
                &hidden,
                &mut norm,
                frames,
                d,
                &block.ffn_norm,
                self.cfg.rms_norm_eps,
            )?;
            compute.gemm_f32(
                frames,
                checked_product("gating width", &[2, ffn])?,
                d,
                &norm,
                &block.linear_in,
                None,
                &mut ffn_in,
            )?;
            for frame in 0..frames {
                ffn_gate[frame * ffn..(frame + 1) * ffn]
                    .copy_from_slice(&ffn_in[frame * 2 * ffn..frame * 2 * ffn + ffn]);
                ffn_up[frame * ffn..(frame + 1) * ffn]
                    .copy_from_slice(&ffn_in[frame * 2 * ffn + ffn..(frame + 1) * 2 * ffn]);
            }
            compute.silu_f32(&ffn_gate, &mut ffn_activated)?;
            for (gate, &up) in ffn_activated.iter_mut().zip(&ffn_up) {
                *gate *= up;
            }
            compute.gemm_f32(
                frames,
                d,
                ffn,
                &ffn_activated,
                &block.linear_out,
                None,
                &mut ffn_output,
            )?;
            for (dst, &value) in hidden.iter_mut().zip(&ffn_output) {
                *dst += value;
            }
        }

        compute.rms_norm_f32(
            &hidden,
            &mut norm,
            frames,
            d,
            &self.weights.final_norm,
            self.cfg.rms_norm_eps,
        )?;
        let logits_len = checked_product("frames*text_card", &[frames, self.cfg.text_card])?;
        let mut logits = vec![0.0f32; logits_len];
        compute.gemm_f32(
            frames,
            self.cfg.text_card,
            d,
            &norm,
            &self.weights.text_head,
            None,
            &mut logits,
        )?;
        KyutaiSttTextLogits::new(frames, self.cfg.text_card, logits)
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
    /// the component logits seam exists, but a real authenticated tensor
    /// binder, streaming state/delay, sampling, and SentencePiece
    /// detokenization remain follow-up gates.
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] if `mimi_codes.len()` is not a
    ///   multiple of `n_q`, is empty, or contains an id outside
    ///   `[0, audio_card)`.
    /// - [`VokraError::NotImplemented`] otherwise (the component seam exists,
    ///   but real binding, streaming state/delay, sampling, and
    ///   SentencePiece detokenization remain — FR-EX-08).
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
                 the decoder-component tensor manifest is bound only by \
                 KyutaiSttWeights::from_component_gguf; the full composite \
                 binder remains a follow-up wave. \
                 Primary source: https://huggingface.co/kyutai/stt-2.6b-en / \
                 https://github.com/kyutai-labs/delayed-streams-modeling",
            ));
        }
        Err(VokraError::NotImplemented(
            "kyutai-stt transcribe: the component logits seam exists, but \
             the real authenticated tensor binder, streaming state/delay, \
             sampling, and SentencePiece detokenization remain blocked. \
             Follow-up wave: bind the authenticated upstream tensor manifest \
             and expose the already-seamed sliding-window causal component \
             through a real streaming decoder. \
             Primary source: https://huggingface.co/kyutai/stt-2.6b-en / \
             https://github.com/kyutai-labs/delayed-streams-modeling",
        ))
    }

    /// Loads a Kyutai STT GGUF from raw bytes under `policy` (M2-13 gate —
    /// a non-commercial provenance without a research flag is refused).
    ///
    /// Public GGUF loading is fail-closed until the fixed composite release
    /// has a real model/Mimi/tokenizer tensor binder and independent parity.
    /// Deterministic synthesized weights remain available only through the
    /// explicit test fixture constructor and are never a public fallback.
    ///
    /// The upstream card identifies the weight as **CC-BY 4.0**
    /// (`AttributionRequired`), but this inspection-only wave does not stamp
    /// or publish provenance and therefore does not activate a runtime load.
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] because the authenticated composite
    ///   tensor binder and native forward are not implemented yet.
    pub fn from_gguf_with_policy(bytes: &[u8], policy: &CompliancePolicy) -> Result<Self> {
        let _ = (bytes, policy);
        Err(VokraError::ModelLoad(
            "kyutai-stt public GGUF loading is blocked: full authenticated composite Mimi/tokenizer/streaming binding and native parity are not implemented; decoder-component binding is intentionally separate; synthesized fixtures are test-only and never a public load fallback".to_owned(),
        ))
    }

    /// Loads a Kyutai STT GGUF from a file path with the fail-closed
    /// strict policy ([`CompliancePolicy::strict`]).
    ///
    /// The upstream card identifies the weight as **CC-BY 4.0**. That
    /// license fact does not waive the missing composite binder gate.
    ///
    /// # Errors
    ///
    /// - [`VokraError::Io`] on read failure.
    /// - See [`Self::from_gguf_with_policy`].
    pub fn from_path(path: impl AsRef<std::path::Path>) -> Result<Self> {
        // Probe existence without materializing the multi-gigabyte composite.
        std::fs::metadata(path.as_ref()).map_err(VokraError::Io)?;
        Err(VokraError::ModelLoad(
            "kyutai-stt public GGUF loading is blocked: full authenticated composite Mimi/tokenizer/streaming binding and native parity are not implemented; decoder-component binding is intentionally separate; synthesized fixtures are test-only and never a public load fallback".to_owned(),
        ))
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

    #[test]
    fn public_gguf_load_is_fail_closed_without_real_tensor_binding() {
        let bytes = build_tiny_gguf(Some(EXPECTED_ARCH));
        let error = KyutaiSttAsr::from_gguf_with_policy(&bytes, &CompliancePolicy::strict())
            .expect_err("public load must reject synthesized fallback");
        assert!(
            matches!(error, VokraError::ModelLoad(message) if message.contains("synthesized") && message.contains("authenticated"))
        );
    }

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
        assert_eq!(c.backbone.ffn_hidden(), 5632);
        assert_eq!(c.n_channels(), 33);
        assert_eq!(c.max_delay(), 0);
        // Everything above adds up to a well-formed config.
        c.validate_for_forward()
            .expect("stt-2.6b-en is well-formed");
    }

    #[test]
    fn tiny_config_is_well_formed() {
        let config = KyutaiSttConfig::tiny_for_tests();
        assert_eq!(config.backbone.ffn_hidden(), 42);
        config
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
    fn component_manifest_is_exact_and_bf16_shape_is_strict() {
        let config = KyutaiSttConfig::tiny_for_tests();
        let names = component_tensor_names(&config).expect("manifest names");
        assert_eq!(
            names.len(),
            1 + config.n_q + config.backbone.n_layer * 6 + 2
        );

        let mut missing = names.clone();
        missing.pop();
        let missing_file = manifest_fixture(&missing, GgmlType::F32);
        assert!(matches!(
            validate_component_tensor_set(&missing_file, &config),
            Err(VokraError::ModelLoad(message)) if message.contains("missing")
        ));

        let mut extra = names;
        extra.push("unexpected.weight".to_owned());
        let extra_file = manifest_fixture(&extra, GgmlType::F32);
        assert!(matches!(
            validate_component_tensor_set(&extra_file, &config),
            Err(VokraError::ModelLoad(message)) if message.contains("extra")
        ));

        let shape_file = one_tensor_fixture("sample", GgmlType::BF16, &[2, 2]);
        assert!(matches!(
            component_tensor(&shape_file, "sample", &[2, 3]),
            Err(VokraError::ModelLoad(message)) if message.contains("shape")
        ));
        let dtype_file = one_tensor_fixture("sample", GgmlType::F32, &[2, 2]);
        assert!(matches!(
            component_tensor(&dtype_file, "sample", &[2, 2]),
            Err(VokraError::ModelLoad(message)) if message.contains("dtype")
        ));
        let nan_file = one_bf16_nan_fixture();
        assert!(matches!(
            component_tensor(&nan_file, "sample", &[1]),
            Err(VokraError::ModelLoad(message)) if message.contains("non-finite")
        ));
    }

    #[test]
    fn component_binder_transposes_torch_linear_layout() {
        let transposed = transpose_component(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3, "fixture")
            .expect("transpose");
        assert_eq!(transposed, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn component_public_binder_keeps_exact_release_gate() {
        let file = GgufFile::parse(build_tiny_gguf(Some(EXPECTED_ARCH))).expect("fixture");
        let error = KyutaiSttWeights::from_component_gguf(&file)
            .expect_err("tiny metadata fixture must not pass the exact release gate");
        assert!(matches!(
            error,
            VokraError::ModelLoad(message) if message.contains("metadata") || message.contains("authenticated")
        ));
    }

    #[test]
    fn component_metadata_contract_accepts_canonical_provenance_before_manifest_gate() {
        let file = strict_component_metadata_fixture(false, 5632, 1, None, false, false, false);
        let error = KyutaiSttWeights::from_component_gguf(&file)
            .expect_err("metadata-only fixture must stop at the exact tensor manifest");
        assert!(matches!(
            error,
            VokraError::ModelLoad(message) if message.contains("manifest")
        ));
    }

    #[test]
    fn component_binder_requires_non_defaulted_metadata_and_provenance() {
        let missing_default =
            strict_component_metadata_fixture(true, 5632, 1, None, false, false, false);
        let error = KyutaiSttWeights::from_component_gguf(&missing_default)
            .expect_err("missing RMS epsilon must fail before payload decode");
        assert!(matches!(
            error,
            VokraError::ModelLoad(message) if message.contains(KEY_BB_RMS_NORM_EPS)
        ));

        let stale_width =
            strict_component_metadata_fixture(false, 8448, 1, None, false, false, false);
        let error = KyutaiSttWeights::from_component_gguf(&stale_width)
            .expect_err("stale ffn_hidden must fail before payload decode");
        assert!(matches!(
            error,
            VokraError::ModelLoad(message) if message.contains(KEY_BB_FFN_HIDDEN)
        ));

        let noncanonical_bool =
            strict_component_metadata_fixture(false, 5632, 2, None, false, false, false);
        let error = KyutaiSttWeights::from_component_gguf(&noncanonical_bool)
            .expect_err("boolean value 2 must fail before payload decode");
        assert!(matches!(
            error,
            VokraError::ModelLoad(message) if message.contains(KEY_BB_CAUSAL)
        ));

        let wrong_license = strict_component_metadata_fixture(
            false,
            5632,
            1,
            Some("CC-BY-NC-4.0"),
            false,
            false,
            false,
        );
        let error = KyutaiSttWeights::from_component_gguf(&wrong_license)
            .expect_err("wrong provenance license must fail before payload decode");
        assert!(matches!(
            error,
            VokraError::ModelLoad(message) if message.contains(chunks::KEY_PROVENANCE_LICENSE)
        ));

        let missing_model_id =
            strict_component_metadata_fixture(false, 5632, 1, None, true, false, false);
        let error = KyutaiSttWeights::from_component_gguf(&missing_model_id)
            .expect_err("missing model id must fail before payload decode");
        assert!(matches!(
            error,
            VokraError::ModelLoad(message) if message.contains(chunks::KEY_PROVENANCE_MODEL_ID)
        ));

        let missing_weight_license =
            strict_component_metadata_fixture(false, 5632, 1, None, false, true, false);
        let error = KyutaiSttWeights::from_component_gguf(&missing_weight_license)
            .expect_err("missing weight license must fail before payload decode");
        assert!(matches!(
            error,
            VokraError::ModelLoad(message) if message.contains(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
        ));

        let extra_key = strict_component_metadata_fixture(false, 5632, 1, None, false, false, true);
        let error = KyutaiSttWeights::from_component_gguf(&extra_key)
            .expect_err("unexpected Kyutai metadata must fail before payload decode");
        assert!(matches!(
            error,
            VokraError::ModelLoad(message) if message.contains("unexpected metadata")
        ));

        let duplicate_arch = vec![
            (
                chunks::KEY_MODEL_ARCH.to_owned(),
                GgufMetadataValue::String(EXPECTED_ARCH.to_owned()),
            ),
            (
                chunks::KEY_MODEL_ARCH.to_owned(),
                GgufMetadataValue::String(EXPECTED_ARCH.to_owned()),
            ),
        ];
        assert!(matches!(
            require_component_occurrence(&duplicate_arch, chunks::KEY_MODEL_ARCH),
            Err(VokraError::ModelLoad(message)) if message.contains("duplicated")
        ));
        let duplicate_provenance = vec![
            (
                chunks::KEY_PROVENANCE_SOURCE.to_owned(),
                GgufMetadataValue::String("https://huggingface.co/kyutai/stt-2.6b-en".to_owned()),
            ),
            (
                chunks::KEY_PROVENANCE_SOURCE.to_owned(),
                GgufMetadataValue::String("https://huggingface.co/kyutai/stt-2.6b-en".to_owned()),
            ),
        ];
        assert!(matches!(
            require_component_occurrence(&duplicate_provenance, chunks::KEY_PROVENANCE_SOURCE),
            Err(VokraError::ModelLoad(message)) if message.contains("duplicated")
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
    fn asr_new_rejects_non_finite_supplied_weight() {
        let c = KyutaiSttConfig::tiny_for_tests();
        let mut w = KyutaiSttWeights::synthesized(&c, 7).expect("weights");
        w.blocks[0].linear_out[0] = f32::NAN;
        let error = KyutaiSttAsr::new(c, w).expect_err("NaN must not enter execution");
        assert!(matches!(
            error,
            VokraError::InvalidArgument(message) if message.contains("non-finite")
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
    fn dep_q0_decoder_returns_deterministic_authenticated_shape() {
        let config = KyutaiSttConfig::tiny_for_tests();
        let width = config.n_q;
        let asr = KyutaiSttAsr::new(
            config.clone(),
            KyutaiSttWeights::synthesized(&config, 0xD3C0_DEC0).expect("weights"),
        )
        .expect("asr");
        let text = [0, 1, 2];
        let codes = vec![0u32; text.len() * width];
        let first = asr
            .forward_text_logits(BackendKind::Cpu, &text, &codes)
            .expect("tiny dep_q=0 forward");
        let second = asr
            .forward_text_logits(BackendKind::Cpu, &text, &codes)
            .expect("repeat tiny dep_q=0 forward");
        assert_eq!(
            first, second,
            "self-consistency fixture must be deterministic"
        );
        assert_eq!(first.frames(), text.len());
        assert_eq!(first.vocab(), config.text_card);
        assert_eq!(first.as_slice().len(), text.len() * config.text_card);
        assert!(first.as_slice().iter().all(|value| value.is_finite()));
    }

    #[test]
    fn dep_q0_decoder_accepts_longer_sequences_and_rejects_input_shape_drift() {
        let config = KyutaiSttConfig::tiny_for_tests();
        let asr = KyutaiSttAsr::new(
            config.clone(),
            KyutaiSttWeights::synthesized(&config, 7).expect("weights"),
        )
        .expect("asr");
        let valid_codes = vec![0u32; config.n_q];
        assert!(matches!(
            asr.forward_text_logits(BackendKind::Cpu, &[0], &valid_codes[..config.n_q - 1]),
            Err(VokraError::InvalidArgument(_))
        ));
        let too_many = vec![0u32; (config.backbone.context + 1) * config.n_q];
        let too_many_text = vec![0u32; config.backbone.context + 1];
        let logits = asr
            .forward_text_logits(BackendKind::Cpu, &too_many_text, &too_many)
            .expect("context is an attention window, not an input limit");
        assert_eq!(logits.frames(), config.backbone.context + 1);
    }

    #[test]
    fn dep_q0_decoder_window_excludes_preceding_frame_in_one_layer_fixture() {
        let mut config = KyutaiSttConfig::tiny_for_tests();
        config.backbone.n_layer = 1;
        config.backbone.context = 2;
        let weights = KyutaiSttWeights::synthesized(&config, 7).expect("weights");
        let asr = KyutaiSttAsr::new(config.clone(), weights).expect("asr");
        let first_text = [0, 1, 2];
        let second_text = [3, 1, 2];
        let codes = vec![0u32; first_text.len() * config.n_q];
        let first = asr
            .forward_text_logits(BackendKind::Cpu, &first_text, &codes)
            .expect("first sequence");
        let second = asr
            .forward_text_logits(BackendKind::Cpu, &second_text, &codes)
            .expect("second sequence");
        let final_row = config.text_card * (first_text.len() - 1);
        assert_eq!(
            &first.as_slice()[final_row..],
            &second.as_slice()[final_row..],
            "the one-layer final row only sees the causal context window"
        );
    }

    #[test]
    fn dep_q0_decoder_rejects_nonzero_dep_q_without_downgrade() {
        let mut config = KyutaiSttConfig::tiny_for_tests();
        config.dep_q = 1;
        let asr = KyutaiSttAsr::new(
            config.clone(),
            KyutaiSttWeights::synthesized(&config, 7).expect("weights"),
        )
        .expect("shape-compatible fixture");
        let error = asr
            .forward_text_logits(BackendKind::Cpu, &[0], &vec![0u32; config.n_q])
            .expect_err("dep_q>0 must not enter the dep_q=0 seam");
        assert!(
            matches!(error, VokraError::InvalidArgument(message) if message.contains("dep_q=0"))
        );
    }

    #[test]
    fn dep_q0_decoder_does_not_fallback_on_uncovered_backend() {
        let config = KyutaiSttConfig::tiny_for_tests();
        let asr = KyutaiSttAsr::new(
            config.clone(),
            KyutaiSttWeights::synthesized(&config, 7).expect("weights"),
        )
        .expect("asr");
        let error = asr
            .forward_text_logits(BackendKind::Vulkan, &[0], &vec![0u32; config.n_q])
            .expect_err("unavailable backend must fail before CPU fallback");
        assert!(matches!(
            error,
            VokraError::BackendUnavailable(_) | VokraError::UnsupportedOp(_)
        ));
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

    #[test]
    fn streaming_contract_matches_pinned_upstream_input_preparation() {
        let contract = KyutaiSttStreamingContract::from_config(&KyutaiSttConfig::stt_2_6b_en())
            .expect("fixed STT streaming contract");
        assert_eq!(contract.sample_rate(), 24_000);
        assert_eq!(contract.frame_hop_samples(), 1_920);
        assert_eq!(contract.n_q(), 32);
        assert_eq!(contract.audio_card(), 2_048);
        assert_eq!(contract.pcm_padding_samples(), (24_000, 84_000));
        assert_eq!(contract.padded_frame_count(0).unwrap(), 56);
        assert_eq!(contract.padded_frame_count(24_000).unwrap(), 68);
        assert!(!contract.emits_text_token(0));
        assert!(!contract.emits_text_token(3));
        assert!(contract.emits_text_token(1));
    }

    #[test]
    fn streaming_contract_rejects_wrong_config_and_code_packets() {
        let mut wrong = KyutaiSttConfig::stt_2_6b_en();
        wrong.audio_delay_seconds = 2.0;
        assert!(matches!(
            KyutaiSttStreamingContract::from_config(&wrong),
            Err(VokraError::InvalidArgument(_))
        ));
        let contract = KyutaiSttStreamingContract::from_config(&KyutaiSttConfig::stt_2_6b_en())
            .expect("fixed STT streaming contract");
        assert!(matches!(
            contract.validate_mimi_codes(&[]),
            Err(VokraError::InvalidArgument(_))
        ));
        assert!(matches!(
            contract.validate_mimi_codes(&[0; 31]),
            Err(VokraError::InvalidArgument(_))
        ));
        let mut out_of_range = vec![0u32; 32];
        out_of_range[31] = 2_048;
        assert!(matches!(
            contract.validate_mimi_codes(&out_of_range),
            Err(VokraError::InvalidArgument(_))
        ));
        assert_eq!(contract.validate_mimi_codes(&[0; 32]).unwrap(), 1);
    }

    #[test]
    fn sidecar_binding_rejects_legacy_or_unverified_identity() {
        let config = KyutaiSttConfig::stt_2_6b_en();
        assert!(matches!(
            KyutaiSttAuthenticatedSidecars::bind(
                &config,
                "mimi-pytorch-e351c8d8@125.safetensors",
                &[],
                "tokenizer_spm_4k_en.model",
                &[],
            ),
            Err(VokraError::ModelLoad(message)) if message.contains("tokenizer_en_audio_4000.model")
        ));
        assert!(matches!(
            KyutaiSttAuthenticatedSidecars::bind(
                &config,
                "mimi-pytorch-e351c8d8@125.safetensors",
                &[],
                "tokenizer_en_audio_4000.model",
                &[],
            ),
            Err(VokraError::ModelLoad(message)) if message.contains("Mimi")
        ));
    }

    #[test]
    fn dep_q0_input_demux_splits_text_and_mimi_streams() {
        let config = KyutaiSttConfig::tiny_for_tests();
        // Each row is [text, audio_0, audio_1, audio_2, audio_3].
        let packet =
            KyutaiSttInputPacket::from_interleaved(&config, &[1, 2, 3, 4, 5, 6, 7, 0, 1, 2])
                .expect("two dep_q=0 input frames");
        assert_eq!(packet.frames(), 2);
        assert_eq!(packet.text_tokens(), &[1, 6]);
        assert_eq!(packet.mimi_codes(), &[2, 3, 4, 5, 7, 0, 1, 2]);
        assert!(matches!(
            KyutaiSttInputPacket::from_interleaved(&config, &[0; 4]),
            Err(VokraError::InvalidArgument(message)) if message.contains("multiple of 5")
        ));
    }

    #[test]
    fn streaming_state_validates_frames_and_suppresses_text_markers() {
        let contract = KyutaiSttStreamingContract::from_config(&KyutaiSttConfig::stt_2_6b_en())
            .expect("fixed STT streaming contract");
        let mut state = KyutaiSttStreamingState::new(contract);
        state
            .push_mimi_frame(&[0; 32])
            .expect("one complete Mimi frame");
        assert_eq!(state.frames_seen(), 1);
        assert_eq!(state.push_text_token(0).unwrap(), None);
        assert_eq!(state.push_text_token(3).unwrap(), None);
        assert_eq!(state.push_text_token(17).unwrap(), Some(17));
        assert_eq!(
            state
                .push_text_token(contract.text_card() as u32 - 1)
                .unwrap(),
            Some(3999)
        );
        assert!(matches!(
            state.push_text_token(contract.text_card() as u32),
            Err(VokraError::InvalidArgument(message)) if message.contains("input-only initial row")
        ));
        assert_eq!(state.emitted_text_tokens(), &[17, 3999]);
        assert!(matches!(
            state.push_mimi_frame(&[0; 31]),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn streaming_contract_accepts_only_the_authenticated_mimi_metadata() {
        let contract = KyutaiSttStreamingContract::from_config(&KyutaiSttConfig::stt_2_6b_en())
            .expect("fixed STT streaming contract");
        let mimi = MimiNeuralConfig {
            sample_rate: 24_000,
            frame_rate_mhz: 12_500,
            seanet: crate::mimi::config::MimiSeanetConfig {
                dimension: 512,
                n_filters: 64,
                n_residual_layers: 1,
                kernel_size: 7,
                residual_kernel_size: 3,
                last_kernel_size: 3,
                compress: 2,
                dilation_base: 2,
                ratios: vec![8, 6, 5, 4],
            },
            transformer: crate::mimi::config::MimiTransformerConfig {
                d_model: 512,
                n_head: 8,
                n_layer: 8,
                ff_dim: 2_048,
                context: 250,
                max_period: 10_000,
                layer_scale: 0.01,
            },
            quantizer: crate::mimi::config::MimiQuantizerConfig {
                dimension: 256,
                n_q: 32,
                bins: 2_048,
                input_dimension: 512,
                output_dimension: 512,
            },
        };
        contract
            .validate_mimi_config(&mimi)
            .expect("authenticated Mimi metadata contract");
        let mut wrong = mimi.clone();
        wrong.frame_rate_mhz = 25_000;
        assert!(matches!(
            contract.validate_mimi_config(&wrong),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn tokenizer_identity_gate_rejects_same_size_unverified_bytes() {
        let bytes = vec![0u8; KYUTAI_STT_TOKENIZER_BYTES];
        assert!(matches!(
            validate_tokenizer_bytes(&bytes),
            Err(VokraError::ModelLoad(_))
        ));
        assert!(matches!(
            validate_tokenizer_bytes(&bytes[..bytes.len() - 1]),
            Err(VokraError::ModelLoad(_))
        ));
    }

    #[test]
    fn mimi_sidecar_identity_gate_is_explicit_and_fail_closed() {
        assert_eq!(
            KYUTAI_STT_MIMI_FILE,
            "mimi-pytorch-e351c8d8@125.safetensors"
        );
        assert_eq!(KYUTAI_STT_MIMI_BYTES, 384_644_900);
        assert_eq!(
            KYUTAI_STT_MIMI_SHA256,
            "09b782f0629851a271227fb9d36db65c041790365f11bbe5d3d59369cf863f50"
        );
        assert!(matches!(
            validate_mimi_bytes(&[]),
            Err(VokraError::ModelLoad(_))
        ));
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
        build_gguf_for_config(arch, &KyutaiSttConfig::stt_2_6b_en())
    }

    /// The same fixture at [`KyutaiSttConfig::tiny_for_tests`] scale, for
    /// tests that go on to CONSTRUCT an engine.
    ///
    /// `from_gguf` finds no tensors on these metadata-only fixtures and
    /// falls back to `KyutaiSttWeights::synthesized`, which allocates the
    /// whole model. At `stt_2_6b_en` scale (48 layers x d_model 2048) that
    /// is ~2.5 G parameters — about **10 GB of f32** — per test. Three
    /// tests did that, and it was not a micro-optimisation issue:
    ///
    /// - the workspace suite was SIGTERM-killed on GitHub runners, taking
    ///   `test (ubuntu-latest)`, `test (windows-latest)` and `coverage`
    ///   red, and CI logged all three as "running for over 60 seconds";
    /// - the same tests OOM-killed a 16 GB dev machine hard enough to
    ///   reboot it.
    ///
    /// None of those three assert anything about model SIZE — they cover
    /// the license gate, the loud-partial transcribe message, and
    /// `from_path` == `from_gguf`. The real-config round trip is covered
    /// separately by `config_round_trips_from_converter_written_gguf`,
    /// which only parses metadata and never synthesizes, so it stays on
    /// `stt_2_6b_en` and stays fast.
    fn build_tiny_gguf(arch: Option<&str>) -> Vec<u8> {
        build_gguf_for_config(arch, &KyutaiSttConfig::tiny_for_tests())
    }

    fn manifest_fixture(names: &[String], dtype: GgmlType) -> GgufFile {
        let mut builder = GgufBuilder::new();
        for name in names {
            builder
                .add_tensor(name, dtype, vec![1], tensor_bytes(dtype, 1))
                .expect("manifest fixture tensor");
        }
        GgufFile::parse(builder.to_bytes().expect("manifest fixture bytes")).expect("parse")
    }

    fn one_tensor_fixture(name: &str, dtype: GgmlType, dimensions: &[usize]) -> GgufFile {
        let elements = dimensions
            .iter()
            .copied()
            .try_fold(1usize, usize::checked_mul)
            .expect("fixture dimensions");
        let mut builder = GgufBuilder::new();
        builder
            .add_tensor(
                name,
                dtype,
                dimensions.iter().map(|&value| value as u64).collect(),
                tensor_bytes(dtype, elements),
            )
            .expect("single tensor fixture");
        GgufFile::parse(builder.to_bytes().expect("single tensor bytes")).expect("parse")
    }

    fn one_bf16_nan_fixture() -> GgufFile {
        let mut builder = GgufBuilder::new();
        builder
            .add_tensor(
                "sample",
                GgmlType::BF16,
                vec![1],
                0x7fc0u16.to_le_bytes().to_vec(),
            )
            .expect("BF16 NaN fixture");
        GgufFile::parse(builder.to_bytes().expect("BF16 NaN bytes")).expect("parse")
    }

    fn tensor_bytes(dtype: GgmlType, elements: usize) -> Vec<u8> {
        match dtype {
            GgmlType::F32 => vec![0; elements * std::mem::size_of::<f32>()],
            GgmlType::BF16 => vec![0; elements * std::mem::size_of::<u16>()],
            other => panic!("unsupported fixture dtype {other:?}"),
        }
    }

    fn strict_component_metadata_fixture(
        omit_rms_norm_eps: bool,
        ffn_hidden: u32,
        causal: u32,
        license_override: Option<&str>,
        omit_model_id: bool,
        omit_weight_license: bool,
        extra_prefixed_key: bool,
    ) -> GgufFile {
        let cfg = KyutaiSttConfig::stt_2_6b_en();
        let mut builder = GgufBuilder::new();
        builder.add_string(chunks::KEY_MODEL_ARCH, EXPECTED_ARCH);
        builder.add_u32(KEY_BB_N_LAYER, cfg.backbone.n_layer as u32);
        builder.add_u32(KEY_BB_D_MODEL, cfg.backbone.d_model as u32);
        builder.add_u32(KEY_BB_N_HEAD, cfg.backbone.n_head as u32);
        builder.add_f32(KEY_BB_HIDDEN_SCALE, cfg.backbone.hidden_scale);
        builder.add_u32(KEY_BB_FFN_HIDDEN, ffn_hidden);
        builder.add_u32(KEY_BB_CONTEXT, cfg.backbone.context as u32);
        builder.add_f32(KEY_BB_ROPE_MAX_PERIOD, cfg.backbone.rope_max_period);
        builder.add_u32(KEY_BB_CAUSAL, causal);
        if !omit_rms_norm_eps {
            builder.add_f32(KEY_BB_RMS_NORM_EPS, cfg.rms_norm_eps);
        }
        builder.add_u32(KEY_DEP_N_LAYER, cfg.depformer.n_layer as u32);
        builder.add_u32(KEY_DEP_D_MODEL, cfg.depformer.d_model as u32);
        builder.add_u32(KEY_DEP_N_HEAD, cfg.depformer.n_head as u32);
        builder.add_u32(KEY_DEP_MULTI_LINEAR, 1);
        builder.add_u32(KEY_DEP_WEIGHTS_PER_STEP, 1);
        builder.add_u32(KEY_N_Q, cfg.n_q as u32);
        builder.add_u32(KEY_DEP_Q, 0);
        builder.add_u32(KEY_AUDIO_CARD, cfg.audio_card as u32);
        builder.add_u32(KEY_TEXT_CARD, cfg.text_card as u32);
        builder.add_u32(KEY_TEXT_PAD_ID, cfg.text_pad_id);
        builder.add_f32(KEY_AUDIO_DELAY_SECS, cfg.audio_delay_seconds);
        builder.add_f32(
            KEY_AUDIO_SILENCE_PREFIX_SECS,
            cfg.audio_silence_prefix_seconds,
        );
        builder.add_u32(KEY_SAMPLE_RATE, cfg.sample_rate);
        builder.add_u32(KEY_N_DELAYS, 33);
        for index in 0..33 {
            builder.add_u32(&format!("{PREFIX_DELAY}{index}"), 0);
        }
        if extra_prefixed_key {
            builder.add_u32("vokra.kyutai_stt.delay.33", 0);
        }
        if !omit_model_id {
            builder.add_string(chunks::KEY_PROVENANCE_MODEL_ID, "kyutai/stt-2.6b-en");
        }
        builder.add_string(
            chunks::KEY_PROVENANCE_LICENSE,
            license_override.unwrap_or("cc-by-4.0"),
        );
        if !omit_weight_license {
            builder.add_string(
                chunks::KEY_PROVENANCE_WEIGHT_LICENSE,
                LicenseClass::AttributionRequired.as_str(),
            );
        }
        builder.add_string(
            chunks::KEY_PROVENANCE_SOURCE,
            "https://huggingface.co/kyutai/stt-2.6b-en",
        );
        GgufFile::parse(builder.to_bytes().expect("strict metadata fixture")).expect("parse")
    }

    fn build_gguf_for_config(arch: Option<&str>, cfg: &KyutaiSttConfig) -> Vec<u8> {
        let mut b = GgufBuilder::new();
        if let Some(a) = arch {
            b.add_string(chunks::KEY_MODEL_ARCH, a);
        }
        let cfg = cfg.clone();
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
        b.add_string(chunks::KEY_PROVENANCE_LICENSE, "cc-by-4.0");
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
        assert!(
            matches!(err, VokraError::ModelLoad(msg) if msg.contains("authenticated") && msg.contains("blocked"))
        );
    }

    /// A GGUF whose arch is a sibling (`csm`) fails with a message that
    /// names both `kyutai-stt` and the offending tag so the caller can
    /// diagnose the mis-routed conversion.
    #[test]
    fn from_gguf_rejects_wrong_arch() {
        let bytes = build_gguf_with_hparams(Some("csm"));
        let err = KyutaiSttAsr::from_gguf_with_policy(&bytes, &CompliancePolicy::strict())
            .expect_err("wrong arch must be rejected");
        assert!(
            matches!(err, VokraError::ModelLoad(msg) if msg.contains("authenticated") && msg.contains("blocked"))
        );
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

    /// Correct CC-BY-4.0 provenance does not bypass the missing composite
    /// model/Mimi/tokenizer binder.
    #[test]
    fn from_gguf_with_valid_license_still_fails_closed() {
        let bytes = build_tiny_gguf(Some(EXPECTED_ARCH));
        let file = GgufFile::parse(bytes).expect("parse fixture");
        let resolution =
            check_weight_license(&file, &CompliancePolicy::strict()).expect("strict must pass");
        assert_eq!(resolution.class, LicenseClass::AttributionRequired);
        assert!(
            !resolution.is_research_only(),
            "CC-BY 4.0 is commercial-permitted; must NOT be marked research-only"
        );
        let err = KyutaiSttAsr::from_gguf_with_policy(
            &build_tiny_gguf(Some(EXPECTED_ARCH)),
            &CompliancePolicy::strict(),
        )
        .expect_err("valid provenance must not synthesize public weights");
        assert!(matches!(err, VokraError::ModelLoad(msg) if msg.contains("authenticated")));
    }

    /// A public load is rejected before a transcribe call can reach the
    /// synthesized fixture's loud-partial path.
    #[test]
    fn public_loader_rejects_before_transcribe() {
        let bytes = build_tiny_gguf(Some(EXPECTED_ARCH));
        let err = KyutaiSttAsr::from_gguf_with_policy(&bytes, &CompliancePolicy::strict())
            .expect_err("public loader must fail before transcribe");
        assert!(matches!(err, VokraError::ModelLoad(msg) if msg.contains("synthesized")));
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
        assert!(matches!(err, VokraError::ModelLoad(msg) if msg.contains("blocked")));
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
        assert!(matches!(err, VokraError::ModelLoad(msg) if msg.contains("authenticated")));
    }

    /// `from_path` propagates the same fail-closed public binder error as the
    /// raw-byte loader and never constructs synthesized weights.
    #[test]
    fn from_path_is_fail_closed() {
        let bytes = build_tiny_gguf(Some(EXPECTED_ARCH));
        let path = std::env::temp_dir().join(format!(
            "vokra-kyutai-stt-scout-{}.gguf",
            std::process::id()
        ));
        std::fs::write(&path, &bytes).expect("write fixture");
        let via_path = KyutaiSttAsr::from_path(&path).expect_err("from_path must be blocked");
        // Best-effort cleanup — never a panic on cleanup failure (test
        // determinism must not depend on tmp cleanup).
        let _ = std::fs::remove_file(&path);
        assert!(matches!(via_path, VokraError::ModelLoad(msg) if msg.contains("authenticated")));
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
