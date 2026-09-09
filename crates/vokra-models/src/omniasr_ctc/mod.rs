//! Meta **omniASR-CTC-1B** — the Omnilingual ASR family's 1B-parameter
//! wav2vec 2.0 CTC checkpoint (SoTA plan Phase 2, 2026-07-24).
//!
//! # What omniASR-CTC-1B is (primary source)
//!
//! omniASR is Meta AI / Omnilingual ASR Team's open-source multilingual
//! ASR system covering **1600+ languages** (`facebook/omniASR-CTC-1B`,
//! `README.md` model card; the paired 1B-parameter W2V and LLM
//! checkpoints ship in the same family). The CTC variant is a
//! wav2vec 2.0 **waveform-in** encoder + a single-Linear CTC head; unlike
//! the LLM variant it has no autoregressive decoder, and decoding is a
//! host-side greedy blank-fold or prefix beam search
//! ([`vokra_ops::ctc_decode`]).
//!
//! Every hparam below is transcribed **verbatim** from the upstream
//! omnilingual-asr registry (fetched 2026-07-24 — CLAUDE.md「ハルシネー
//! ション厳禁」):
//!
//! - **Family + arch**: `omniasr-ctc` / `1b` — the CTC 1B is one of four
//!   sizes (300M / 1B / 3B / 7B) that share the same wav2vec 2.0 stack.
//!   Registry source:
//!   `github.com/facebookresearch/omnilingual-asr/blob/main/src/omnilingual_asr/models/wav2vec2_asr/config.py`
//!   (arch `1b`) walks
//!   `Wav2Vec2AsrConfig` "base_10h" (fairseq2)
//!   → replaces `encoder_config` with wav2vec 2.0 arch "1b"
//!   (`src/omnilingual_asr/models/wav2vec2_ssl/config.py`) →
//!   which walks fairseq2 "large_lv60k" → sets
//!   `model_dim=1280`, `num_encoder_layers=48`, `ffn_inner_dim=5120`.
//! - **Encoder** (wav2vec 2.0):
//!   - `model_dim` = 1280 (transformer residual width),
//!   - `num_encoder_layers` = 48,
//!   - `num_encoder_attn_heads` = 16 (inherits fairseq2 "large" —
//!     "1b" does not override; MHA, no GQA broadcast),
//!   - `ffn_inner_dim` = 5120 (~4× `model_dim`),
//!   - `feature_dim` = 512 (waveform-feature-extractor output width),
//!   - `feature_extractor_layer_descs` =
//!     `[(512, 10, 5), (512, 3, 2), (512, 3, 2), (512, 3, 2),
//!       (512, 3, 2), (512, 2, 2), (512, 2, 2)]`
//!     (7 Conv1D layers; total stride product = 5·2·2·2·2·2·2 =
//!     **320× downsampling** — one CTC frame per 20 ms at 16 kHz),
//!   - `feature_extractor_bias` = **true** (large_lv60k override),
//!   - `feature_extractor_layer_norm_convs` = **true** (large_lv60k),
//!   - `layer_norm_features` = **false** (large_lv60k — this controls the
//!     separate post-pos/model-dimension norm; the frontend's
//!     post-extraction LayerNorm remains unconditional),
//!   - `pos_encoder_type` = `"conv"` (grouped Conv1D positional
//!     encoder — GELU tail),
//!   - `pos_conv_kernel_size` = 128,
//!   - `num_pos_conv_groups` = 16,
//!   - `pos_encoder_depth` = 1,
//!   - `use_conformer` = **false** (plain Transformer encoder),
//!   - `use_fbank` = **false** (**waveform input** — the front-end
//!     produces the features, unlike Whisper / Parakeet which take
//!     log-mel bins in),
//!   - `norm_order` = `PRE` (large_lv60k — pre-LayerNorm blocks),
//!   - `max_seq_len` = 4096 (upper bound on the post-extraction frame
//!     count).
//! - **CTC head + vocab**:
//!   - `target_vocab_size` = 9812 (the initial release tokenizer's char
//!     vocab; card `omniASR_CTC_1B` uses tokenizer ref
//!     `omniASR_tokenizer` in
//!     `src/omnilingual_asr/cards/models/rc_models.yaml`),
//!   - **`blank_id` = 0** — the fairseq2 wav2vec 2.0 CTC convention
//!     (`torch.nn.functional.ctc_loss` is called without an explicit
//!     `blank=` argument in
//!     `fairseq2/models/wav2vec2/asr/model.py::Wav2Vec2AsrModel.forward`,
//!     which defaults to `blank=0`), passed through into
//!     [`vokra_ops::ctc_decode_greedy`]'s `blank_id` parameter.
//! - **Audio boundary**: `sample_rate` = 16 000 (16 kHz mono per the
//!   model card and the wav2vec 2.0 convention; **not** written in
//!   `config.json` because the upstream release does not ship one — the
//!   HF repo only carries the `.pt` checkpoint + SentencePiece tokenizer).
//! - **Weight license**: **Apache-2.0** (per `facebook/omniASR-CTC-1B`
//!   HF card `license: apache-2.0`; the corpus dataset ships CC-BY-4.0
//!   separately, but the model weights are Apache-2.0) — resolves to
//!   [`vokra_core::LicenseClass::Permissive`] via the
//!   `omniasr-ctc-` family prefix walk, so the M2-13 gate passes
//!   commercially without any attribution obligation on the runtime side.
//!
//! # Boundary — CTC decoding op consumed, never re-implemented
//!
//! omniASR-CTC reuses one shared Vokra primitive instead of duplicating:
//!
//! - **CTC decoding**: [`vokra_ops::ctc_decode`] — greedy blank-fold or
//!   prefix beam search with LM shallow fusion + hotword boost. Uses
//!   `blank_id = 0` (the fairseq2 wav2vec 2.0 CTC default).
//!
//! **Encoder implementation.** Unlike Parakeet's FastConformer encoder,
//! wav2vec 2.0 uses a 7-layer waveform Conv1D feature extractor, grouped
//! Conv1D positional encoder, and plain pre-norm Transformer encoder. The
//! stem routes through [`vokra_ops::waveform_frontend`], while the positional
//! and Transformer stages use the existing backend-dispatched Charsiu
//! primitives and an OmniASR-specific fused-QKV block.
//!
//! # What lands in this Phase 2 slice
//!
//! - [`OmniasrCtcConfig`] — every hparam transcribed from the primary
//!   source (no hardcoded fabrication; sample-rate is inherited from the
//!   wav2vec 2.0 waveform convention, documented on the field).
//! - [`OmniasrCtcWeights`] — a strict weight store plus a deterministic
//!   [`OmniasrCtcWeights::synthesized`] test fixture. `from_gguf` binds only
//!   the complete audited VAST tensor-name/shape manifest and exact prepared
//!   SHA-256 `cda8d7dd7cad2a0361b6946c42342b85ef7b0a8d672b99631dc75b4c3123dbc5`.
//! - [`OmniasrCtcAsr`] — backend-selectable native-forward handle carrying
//!   config and explicitly bound weights. `transcribe_tokens` runs the
//!   complete encoder and CTC path; SentencePiece detokenization remains
//!   outside this module and the GGUF binder authenticates the pinned release
//!   before constructing the native engine.
//!
//! # No ONNX (permanent)
//!
//! omniASR-CTC ships as a fairseq2 `.pt` checkpoint (plus a SentencePiece
//! tokenizer); the pipeline is re-implemented natively in
//! `vokra-models/src/omniasr_ctc/` (whisper.cpp 型, CLAUDE.md 設計判断 4).
//! This module never touches ONNX.

use vokra_core::backend::BackendKind;
use vokra_core::gguf::{GgmlType, GgufFile, chunks};
use vokra_core::rng::SplitMix64;
use vokra_core::{LicenseClass, Result, VokraError};
use vokra_ops::{
    ConvLayerAttrs, ConvLayerWeights, Norm, WaveformFrontendAttrs, WaveformFrontendWeights,
    ctc_decode_greedy,
};

use crate::align::charsiu::{
    CharsiuConfig, CharsiuFeatureProjection, CharsiuPosConv,
    feature_projection_forward_with_compute, layer_norm_with_compute_inplace,
    linear_forward_with_compute, positional_conv_forward_with_compute,
};
use crate::compute::{Compute, HotOp};
use std::collections::{BTreeMap, BTreeSet};

use crate::strict_checkpoint::load_tensor;

/// `vokra.model.arch` an omniASR-CTC GGUF must carry. Written by
/// `vokra-convert::models::omniasr_ctc::ARCH`; the compliance registry
/// (`vokra_core::compliance`) knows `omniasr-ctc-1b` / `omniasr-ctc` /
/// `omniasr-ctc-1_1b` (and the family variants) as
/// [`vokra_core::LicenseClass::Permissive`] via the `omniasr-ctc-`
/// family prefix walk (Apache-2.0 — the M2-13 gate passes commercially).
pub const EXPECTED_ARCH: &str = "omniasr-ctc";

/// PCM sample rate omniASR-CTC expects. **Not** written in an upstream
/// `config.json` (the HF repo carries no config; only the `.pt`
/// checkpoint + SentencePiece tokenizer); taken from the model card
/// ("16 kHz mono waveform") and the wav2vec 2.0 convention.
pub const OMNIASR_CTC_SAMPLE_RATE: u32 = 16_000;

/// The official Omnilingual ASR preprocessing boundary is
/// `torch.nn.functional.layer_norm(waveform, waveform.shape)` with the
/// PyTorch default `eps=1e-5` (the initial upstream commit's
/// `datasets/utils/audio.py::apply_audio_normalization`).  This is
/// intentionally Omni-specific: the shared wav2vec2 CTC helper uses a
/// different `1e-7` epsilon and must not be changed for its other models.
const OMNIASR_CTC_NORMALIZATION_EPS: f32 = 1e-5;

/// Fixed count of Conv1D layers in the wav2vec 2.0 waveform feature
/// extractor — 7, matching the fairseq2
/// `Wav2Vec2EncoderConfig.feature_extractor_layer_descs` default which
/// is `[(512, 10, 5), (512, 3, 2), (512, 3, 2), (512, 3, 2),
/// (512, 3, 2), (512, 2, 2), (512, 2, 2)]`. The "1b" arch does not
/// override this, so all four omniASR sizes share the same 7-layer
/// stem. Pinned as a constant so the weight-store shape gate cannot
/// silently accept a mismatched checkpoint (FR-EX-08).
pub const OMNIASR_CTC_NUM_FEATURE_EXTRACTOR_LAYERS: usize = 7;

/// The pinned fairseq2 payload size for `facebook/omniASR-CTC-1B`.
///
/// The loader additionally checks the complete audited name/shape/F32 map;
/// this count is only the first cheap rejection before tensor interpretation.
pub const OMNIASR_CTC_EXPECTED_TENSOR_COUNT: usize = 807;

/// Immutable identity of the audited VAST extraction.  The source and
/// prepared digests are metadata bindings, not parity claims: they identify
/// the exact fairseq2 checkpoint and the safetensors prepared from it.
const OMNIASR_CTC_MODEL_ID: &str = "facebook/omniASR-CTC-1B";
const OMNIASR_CTC_HF_REVISION: &str = "8c22e3ffdaa4aab6431b128b84b991a7d9c2515c";
const OMNIASR_CTC_SOURCE_SHA256: &str =
    "e8564fa59dab7caedbcdb54ab7fb9bd6c96989f4d19add2ad81ddd969716952c";
const OMNIASR_CTC_PREPARED_SHA256: &str =
    "cda8d7dd7cad2a0361b6946c42342b85ef7b0a8d672b99631dc75b4c3123dbc5";
const KEY_PROVENANCE_UPSTREAM_REVISION: &str = "vokra.provenance.upstream_revision";
const KEY_OMNIASR_SOURCE_SHA256: &str = "vokra.omniasr_ctc.source_sha256";
const KEY_OMNIASR_PREPARED_SHA256: &str = "vokra.omniasr_ctc.prepared_sha256";

/// Learned operations used by the native forward path.
pub const OMNIASR_CTC_HOT_OPS: &[HotOp] = &[
    HotOp::Gemm,
    HotOp::Softmax,
    HotOp::LayerNorm,
    HotOp::Gelu,
    HotOp::Conv1d,
    HotOp::GroupedConv1d,
];

// ---------------------------------------------------------------------------
// `vokra.omniasr_ctc.*` chunk-key mirrors — duplicated verbatim from the
// converter (`crates/vokra-convert/src/models/omniasr_ctc.rs`) so
// `vokra-models` does not gain a dependency edge onto `vokra-convert`.
// This is the same layered-convention rule the sibling BF16 pass-through
// binders (`pyannote` / `snac` / `parakeet_ctc`) use.
//
// Booleans (`feature_extractor_bias`, `feature_extractor_layer_norm_convs`,
// `layer_norm_features`, `use_conformer`) are stamped by the converter as
// `u32` via `u32::from(bool)` (0 / 1); the read side inverts with `!= 0`.
// ---------------------------------------------------------------------------

const KEY_SAMPLE_RATE: &str = "vokra.omniasr_ctc.sample_rate";

// Encoder (wav2vec 2.0)
const KEY_ENC_MODEL_DIM: &str = "vokra.omniasr_ctc.arch.encoder.model_dim";
const KEY_ENC_N_LAYER: &str = "vokra.omniasr_ctc.arch.encoder.num_encoder_layers";
const KEY_ENC_N_HEAD: &str = "vokra.omniasr_ctc.arch.encoder.num_encoder_attn_heads";
const KEY_ENC_FFN_INNER: &str = "vokra.omniasr_ctc.arch.encoder.ffn_inner_dim";
const KEY_ENC_FEATURE_DIM: &str = "vokra.omniasr_ctc.arch.encoder.feature_dim";
const KEY_ENC_MAX_SEQ_LEN: &str = "vokra.omniasr_ctc.arch.encoder.max_seq_len";
const KEY_ENC_FEATURE_BIAS: &str = "vokra.omniasr_ctc.arch.encoder.feature_extractor_bias";
const KEY_ENC_FEATURE_LN_CONVS: &str =
    "vokra.omniasr_ctc.arch.encoder.feature_extractor_layer_norm_convs";
const KEY_ENC_LN_FEATURES: &str = "vokra.omniasr_ctc.arch.encoder.layer_norm_features";
const KEY_ENC_POS_KERNEL: &str = "vokra.omniasr_ctc.arch.encoder.pos_conv_kernel_size";
const KEY_ENC_POS_GROUPS: &str = "vokra.omniasr_ctc.arch.encoder.num_pos_conv_groups";
const KEY_ENC_POS_DEPTH: &str = "vokra.omniasr_ctc.arch.encoder.pos_encoder_depth";
const KEY_ENC_USE_CONFORMER: &str = "vokra.omniasr_ctc.arch.encoder.use_conformer";

// Feature extractor stem (7 layers — the fairseq2 wav2vec 2.0 default,
// pinned as a fixed count). Rides as `count + N × (out_dim, kernel,
// stride)` — the CSM / Dia array pattern for GGUF portability.
const KEY_ENC_FEATURE_LAYERS: &str = "vokra.omniasr_ctc.arch.encoder.feature_extractor_layer_count";
const KEY_ENC_FEATURE_OUT_PREFIX: &str =
    "vokra.omniasr_ctc.arch.encoder.feature_extractor_out_dim.";
const KEY_ENC_FEATURE_KERNEL_PREFIX: &str =
    "vokra.omniasr_ctc.arch.encoder.feature_extractor_kernel.";
const KEY_ENC_FEATURE_STRIDE_PREFIX: &str =
    "vokra.omniasr_ctc.arch.encoder.feature_extractor_stride.";

// CTC head
const KEY_HEAD_VOCAB_SIZE: &str = "vokra.omniasr_ctc.head.target_vocab_size";
const KEY_HEAD_BLANK_ID: &str = "vokra.omniasr_ctc.head.blank_id";

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Per-layer descriptor for the waveform feature extractor
/// (`(out_dim, kernel, stride)` — the fairseq2 tuple used verbatim in
/// `Wav2Vec2EncoderConfig.feature_extractor_layer_descs`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OmniasrCtcConvLayerDesc {
    /// Conv1D output channels.
    pub out_dim: usize,
    /// Conv1D kernel length.
    pub kernel: usize,
    /// Conv1D stride length.
    pub stride: usize,
}

/// wav2vec 2.0 encoder hparams (primary source: the fairseq2 registry
/// walk `large` → `large_lv60k` → `1b` documented in the module
/// docstring).
///
/// The encoder is (a) a 7-layer Conv1D waveform feature extractor + (b) a
/// Linear feature projection + (c) a grouped-Conv1D positional encoder +
/// (d) a stack of pre-norm Transformer blocks. `model_dim` is the
/// residual width; the per-head width is `model_dim / num_encoder_attn_heads`.
#[derive(Debug, Clone, PartialEq)]
pub struct OmniasrCtcEncoderConfig {
    /// `model_dim` — transformer hidden width, 1280 for the 1B.
    pub model_dim: usize,
    /// `num_encoder_layers` — 48 for the 1B.
    pub num_encoder_layers: usize,
    /// `num_encoder_attn_heads` — 16 for the 1B (from fairseq2 "large";
    /// the "1b" arch does not override).
    pub num_encoder_attn_heads: usize,
    /// `ffn_inner_dim` — FFN inner width, 5120 for the 1B.
    pub ffn_inner_dim: usize,
    /// `feature_dim` — waveform-feature-extractor output width, 512.
    pub feature_dim: usize,
    /// Fixed 7-layer stem `[(512,10,5), (512,3,2), (512,3,2), (512,3,2),
    /// (512,3,2), (512,2,2), (512,2,2)]`. The runtime cross-checks
    /// against [`OMNIASR_CTC_NUM_FEATURE_EXTRACTOR_LAYERS`].
    pub feature_extractor_layer_descs:
        [OmniasrCtcConvLayerDesc; OMNIASR_CTC_NUM_FEATURE_EXTRACTOR_LAYERS],
    /// `feature_extractor_bias` — **true** for large_lv60k (each Conv1D
    /// in the stem learns an additive bias; the base wav2vec2 arch has
    /// this false, but omniASR walks large_lv60k which overrides).
    pub feature_extractor_bias: bool,
    /// `feature_extractor_layer_norm_convs` — **true** for large_lv60k
    /// (each conv output is layer-normalised; a Group Norm variant
    /// exists for the base arch but omniASR uses per-layer LayerNorm).
    pub feature_extractor_layer_norm_convs: bool,
    /// `layer_norm_features` — **false** for large_lv60k. This is a
    /// separate post-pos/model-dimension normalization flag; it does not
    /// disable the frontend's unconditional post-extraction LayerNorm.
    pub layer_norm_features: bool,
    /// `pos_conv_kernel_size` — 128.
    pub pos_conv_kernel_size: usize,
    /// `num_pos_conv_groups` — 16 (grouped Conv1D — the group count
    /// divides `model_dim`).
    pub num_pos_conv_groups: usize,
    /// `pos_encoder_depth` — 1 (single positional Conv1D block).
    pub pos_encoder_depth: usize,
    /// `use_conformer` — **false** for wav2vec 2.0 (plain Transformer
    /// blocks). Kept as a field so a hypothetical future Conformer
    /// variant is representable without a new type.
    pub use_conformer: bool,
    /// `max_seq_len` — 4096 (upper bound on the post-extraction frame
    /// count; a real forward asserts the incoming sequence does not
    /// exceed it).
    pub max_seq_len: usize,
}

impl OmniasrCtcEncoderConfig {
    /// Per-head width (`model_dim / num_encoder_attn_heads`); `0` when
    /// `num_encoder_attn_heads == 0` (shape-only converter sentinel) so
    /// shape checks never panic.
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.model_dim
            .checked_div(self.num_encoder_attn_heads)
            .unwrap_or(0)
    }

    /// MHA algebraic constraint: attention heads divide the width. All
    /// non-zero.
    #[must_use]
    pub fn is_well_formed(&self) -> bool {
        self.num_encoder_attn_heads != 0
            && self.model_dim != 0
            && self.model_dim % self.num_encoder_attn_heads == 0
    }

    /// Overall temporal stride of the waveform feature extractor
    /// (product of all Conv1D strides — 320× for the 1B stem, so one
    /// CTC frame per 20 ms at 16 kHz).
    #[must_use]
    pub fn feature_extractor_total_stride(&self) -> usize {
        let mut s: usize = 1;
        for d in &self.feature_extractor_layer_descs {
            s = s.saturating_mul(d.stride);
        }
        s
    }
}

/// CTC head hparams (primary source: `target_vocab_size` from
/// `wav2vec2_asr/config.py::_1b_asr`; `blank_id` from the fairseq2
/// wav2vec 2.0 CTC convention — `ctc_loss` called without an explicit
/// `blank=` uses `blank=0`).
///
/// The head is a single Linear from encoder `model_dim` to
/// `target_vocab_size`; the CTC blank is a distinguished token at
/// `blank_id = 0` (not at the vocab tail — that is the NeMo /
/// Parakeet convention, not the fairseq2 wav2vec 2.0 one).
#[derive(Debug, Clone, PartialEq)]
pub struct OmniasrCtcHeadConfig {
    /// `target_vocab_size` — 9812 for the CTC 1B v1 tokenizer.
    pub target_vocab_size: usize,
    /// **`blank_id` = 0** — the fairseq2 wav2vec 2.0 CTC convention.
    /// Passed straight into [`vokra_ops::ctc_decode_greedy`]'s
    /// `blank_id` parameter.
    pub blank_id: u32,
    /// `final_dropout_p` — 0.0 for inference (train-time only; the
    /// forward path is dropout-free).
    pub final_dropout_p: f32,
}

impl OmniasrCtcHeadConfig {
    /// The CTC blank id — an alias for [`Self::blank_id`] that reads as
    /// the term of art in the decoding call site (symmetric with the
    /// Parakeet-CTC accessor).
    #[must_use]
    pub fn blank_id(&self) -> u32 {
        self.blank_id
    }
}

/// The omniASR-CTC size variant the loader targets.
///
/// The Meta Omnilingual ASR family ships four sizes (300M / 1B / 3B /
/// 7B) that share the same wav2vec 2.0 topology and Apache-2.0 license
/// but scale the LM depth / hidden dim differently. This enum
/// discriminates the loader path so a caller can request the sibling
/// checkpoint sizes reusing the shared 1B loader — the encoder body,
/// CTC head, tokenizer contract, and blank-id convention are identical
/// across sizes, only the fairseq2 arch preset the encoder walks
/// changes (`base` / `large_lv60k` / scaled derivatives).
///
/// SoTA plan reuse bundle (2026-07-30): capacity-factor branch of the
/// existing 1B loader — new variants are additive against the same
/// primitives (`vokra_ops::ctc_decode` + the shared wav2vec 2.0
/// scaffold), so a downstream picks a variant without duplicating
/// arch code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OmniasrCtcVariant {
    /// `facebook/omniASR-CTC-300M` — the smallest size (~300M params).
    /// Encoder walks the fairseq2 wav2vec 2.0 **base** arch preset:
    /// `model_dim=768`, `num_encoder_layers=12`,
    /// `num_encoder_attn_heads=12`, `ffn_inner_dim=3072`. Uses the
    /// **base** feature-extractor axes (`feature_extractor_bias=false`,
    /// `feature_extractor_layer_norm_convs=false`,
    /// `layer_norm_features=true` — the separate post-pos/model-dimension
    /// normalization is enabled; the frontend post-extraction LayerNorm is
    /// unconditional). Same 7-layer Conv1D stem, same
    /// waveform-in front-end, same target vocab / blank-id convention.
    M300,
    /// `facebook/omniASR-CTC-1B` — the anchor size (~1B params).
    /// Encoder walks the fairseq2 wav2vec 2.0 **large_lv60k** arch:
    /// `model_dim=1280`, `num_encoder_layers=48`,
    /// `num_encoder_attn_heads=16`, `ffn_inner_dim=5120`. Uses the
    /// large_lv60k feature-extractor axes
    /// (`feature_extractor_bias=true`,
    /// `feature_extractor_layer_norm_convs=true`,
    /// `layer_norm_features=false`).
    B1,
    /// `facebook/omniASR-CTC-7B` — the largest size (~7B params).
    /// Encoder walks a scaled derivative of the fairseq2 wav2vec 2.0
    /// large_lv60k arch preset (Meta's Omnilingual ASR release does
    /// not publish an authoritative `config.json` for the 7B model on
    /// HF; the `.pt` checkpoint's tensor shapes are the ultimate
    /// source of truth). The runtime loader carries **`0`-placeholder
    /// dims** for the transformer axes (`model_dim`,
    /// `num_encoder_layers`, `num_encoder_attn_heads`,
    /// `ffn_inner_dim`) — the shape-validation gate rejects the `0`
    /// sentinels loudly (FR-EX-08), so a caller cannot silently run a
    /// hallucinated forward. Real 7B binding fills the placeholder
    /// dims from the `.pt` checkpoint's tensor shapes (T29-equivalent
    /// — the same posture as the Canary-Qwen decoder-dim path).
    B7,
}

impl OmniasrCtcVariant {
    /// Canonical model-card slug for this variant (`omniasr-ctc-300m` /
    /// `omniasr-ctc-1b` / `omniasr-ctc-7b`).
    #[must_use]
    pub fn model_id(self) -> &'static str {
        match self {
            Self::M300 => "omniasr-ctc-300m",
            Self::B1 => "omniasr-ctc-1b",
            Self::B7 => "omniasr-ctc-7b",
        }
    }
}

/// Resolved omniASR-CTC hparam snapshot — every field is transcribed
/// from the upstream fairseq2 registry (module docstring) or from the
/// wav2vec 2.0 convention (`sample_rate` — the HF repo carries no
/// `config.json`).
#[derive(Debug, Clone, PartialEq)]
pub struct OmniasrCtcConfig {
    /// wav2vec 2.0 encoder hparams.
    pub encoder: OmniasrCtcEncoderConfig,
    /// CTC head hparams (vocab + blank id + final dropout).
    pub head: OmniasrCtcHeadConfig,
    /// PCM sample rate omniASR-CTC expects — 16 000 (from the model
    /// card + wav2vec 2.0 convention; the HF repo carries no
    /// `config.json`).
    pub sample_rate: u32,
}

impl OmniasrCtcConfig {
    /// Primary-source omniASR-CTC-1B config (every value transcribed
    /// from the fairseq2 registry — see module docstring).
    #[must_use]
    pub fn omniasr_ctc_1b() -> Self {
        Self {
            encoder: OmniasrCtcEncoderConfig {
                model_dim: 1280,
                num_encoder_layers: 48,
                num_encoder_attn_heads: 16,
                ffn_inner_dim: 5120,
                feature_dim: 512,
                feature_extractor_layer_descs: [
                    OmniasrCtcConvLayerDesc {
                        out_dim: 512,
                        kernel: 10,
                        stride: 5,
                    },
                    OmniasrCtcConvLayerDesc {
                        out_dim: 512,
                        kernel: 3,
                        stride: 2,
                    },
                    OmniasrCtcConvLayerDesc {
                        out_dim: 512,
                        kernel: 3,
                        stride: 2,
                    },
                    OmniasrCtcConvLayerDesc {
                        out_dim: 512,
                        kernel: 3,
                        stride: 2,
                    },
                    OmniasrCtcConvLayerDesc {
                        out_dim: 512,
                        kernel: 3,
                        stride: 2,
                    },
                    OmniasrCtcConvLayerDesc {
                        out_dim: 512,
                        kernel: 2,
                        stride: 2,
                    },
                    OmniasrCtcConvLayerDesc {
                        out_dim: 512,
                        kernel: 2,
                        stride: 2,
                    },
                ],
                feature_extractor_bias: true,
                feature_extractor_layer_norm_convs: true,
                layer_norm_features: false,
                pos_conv_kernel_size: 128,
                num_pos_conv_groups: 16,
                pos_encoder_depth: 1,
                use_conformer: false,
                max_seq_len: 4096,
            },
            head: OmniasrCtcHeadConfig {
                target_vocab_size: 9812,
                blank_id: 0,
                final_dropout_p: 0.0,
            },
            sample_rate: OMNIASR_CTC_SAMPLE_RATE,
        }
    }

    /// `facebook/omniASR-CTC-300M` — the smallest omniASR-CTC size
    /// (~300M params). Encoder walks the fairseq2 wav2vec 2.0 **base**
    /// arch preset (SoTA plan reuse bundle, 2026-07-30):
    ///
    /// - `model_dim = 768` (base arch — Parakeet's Conformer widths are
    ///   different; this is the plain wav2vec 2.0 base hidden size).
    /// - `num_encoder_layers = 12`.
    /// - `num_encoder_attn_heads = 12` (base — head_dim = 64).
    /// - `ffn_inner_dim = 3072` (base — ~4× model_dim).
    /// - `feature_dim = 512` (same waveform-extractor output width).
    /// - Same 7-layer Conv1D stem `[(512,10,5), (512,3,2)×4,
    ///   (512,2,2)×2]` — 320× total stride (all wav2vec 2.0 variants
    ///   share this stem).
    /// - **Base-arch feature-extractor axes**:
    ///   `feature_extractor_bias = false` (the base wav2vec 2.0 conv
    ///   stem carries no additive bias), `feature_extractor_layer_norm_convs
    ///   = false` (the base uses GroupNorm on the stem, not per-layer
    ///   LayerNorm), `layer_norm_features = true` (the separate
    ///   post-pos/model-dimension normalization is enabled; the frontend
    ///   post-extraction LayerNorm remains present).
    /// - Same positional Conv1D encoder (`pos_conv_kernel_size = 128`,
    ///   `num_pos_conv_groups = 16`, `pos_encoder_depth = 1`).
    /// - Same CTC head (`target_vocab_size = 9812`, `blank_id = 0` —
    ///   fairseq2 wav2vec 2.0 convention).
    ///
    /// Reuses the shared 1B loader's shape / weight-store machinery
    /// (`OmniasrCtcWeights::synthesized` + `OmniasrCtcAsr::new`) —
    /// only the config differs.
    #[must_use]
    pub fn omniasr_ctc_300m() -> Self {
        Self {
            encoder: OmniasrCtcEncoderConfig {
                model_dim: 768,
                num_encoder_layers: 12,
                num_encoder_attn_heads: 12,
                ffn_inner_dim: 3072,
                feature_dim: 512,
                feature_extractor_layer_descs: [
                    OmniasrCtcConvLayerDesc {
                        out_dim: 512,
                        kernel: 10,
                        stride: 5,
                    },
                    OmniasrCtcConvLayerDesc {
                        out_dim: 512,
                        kernel: 3,
                        stride: 2,
                    },
                    OmniasrCtcConvLayerDesc {
                        out_dim: 512,
                        kernel: 3,
                        stride: 2,
                    },
                    OmniasrCtcConvLayerDesc {
                        out_dim: 512,
                        kernel: 3,
                        stride: 2,
                    },
                    OmniasrCtcConvLayerDesc {
                        out_dim: 512,
                        kernel: 3,
                        stride: 2,
                    },
                    OmniasrCtcConvLayerDesc {
                        out_dim: 512,
                        kernel: 2,
                        stride: 2,
                    },
                    OmniasrCtcConvLayerDesc {
                        out_dim: 512,
                        kernel: 2,
                        stride: 2,
                    },
                ],
                feature_extractor_bias: false,
                feature_extractor_layer_norm_convs: false,
                layer_norm_features: true,
                pos_conv_kernel_size: 128,
                num_pos_conv_groups: 16,
                pos_encoder_depth: 1,
                use_conformer: false,
                max_seq_len: 4096,
            },
            head: OmniasrCtcHeadConfig {
                target_vocab_size: 9812,
                blank_id: 0,
                final_dropout_p: 0.0,
            },
            sample_rate: OMNIASR_CTC_SAMPLE_RATE,
        }
    }

    /// `facebook/omniASR-CTC-7B` — the largest omniASR-CTC size
    /// (~7B params, SoTA plan reuse bundle 2026-07-30).
    ///
    /// **Placeholder-dim posture**: Meta's Omnilingual ASR release does
    /// **not** publish an authoritative `config.json` for the 7B model
    /// on HF; the `.pt` checkpoint's tensor shapes are the ultimate
    /// source of truth. The runtime carries `0`-placeholder transformer
    /// axes (`model_dim`, `num_encoder_layers`,
    /// `num_encoder_attn_heads`, `ffn_inner_dim`) — the shape
    /// validation gate rejects the `0` sentinels loudly (FR-EX-08 —
    /// same posture as the Canary-Qwen decoder-dim path). Real 7B
    /// binding fills the placeholder dims from the `.pt` checkpoint's
    /// tensor shapes (T29-equivalent follow-up wave).
    ///
    /// Fields that are **not** placeholder:
    ///
    /// - `feature_dim = 512` (all wav2vec 2.0 sizes share the same
    ///   waveform-extractor output width — the 7-layer Conv1D stem is
    ///   size-invariant).
    /// - Feature extractor axes = large_lv60k
    ///   (`feature_extractor_bias = true`,
    ///   `feature_extractor_layer_norm_convs = true`,
    ///   `layer_norm_features = false` — the 7B is a scaled derivative
    ///   of the large_lv60k arch preset, not the base).
    /// - `pos_conv_kernel_size = 128`, `num_pos_conv_groups = 16`,
    ///   `pos_encoder_depth = 1` (positional encoder is arch-invariant
    ///   in wav2vec 2.0).
    /// - `use_conformer = false` (all omniASR variants use plain
    ///   Transformer encoders, not Conformer).
    /// - `max_seq_len = 4096`, `target_vocab_size = 9812`, `blank_id =
    ///   0`, `final_dropout_p = 0.0` — same head axes as every
    ///   sibling size (v1 SentencePiece char tokenizer + fairseq2
    ///   wav2vec 2.0 blank convention are family-wide, not scale-
    ///   dependent).
    #[must_use]
    pub fn omniasr_ctc_7b() -> Self {
        Self {
            encoder: OmniasrCtcEncoderConfig {
                // Placeholder — validate_for_forward rejects 0.
                model_dim: 0,
                num_encoder_layers: 0,
                num_encoder_attn_heads: 0,
                ffn_inner_dim: 0,
                feature_dim: 512,
                feature_extractor_layer_descs: [
                    OmniasrCtcConvLayerDesc {
                        out_dim: 512,
                        kernel: 10,
                        stride: 5,
                    },
                    OmniasrCtcConvLayerDesc {
                        out_dim: 512,
                        kernel: 3,
                        stride: 2,
                    },
                    OmniasrCtcConvLayerDesc {
                        out_dim: 512,
                        kernel: 3,
                        stride: 2,
                    },
                    OmniasrCtcConvLayerDesc {
                        out_dim: 512,
                        kernel: 3,
                        stride: 2,
                    },
                    OmniasrCtcConvLayerDesc {
                        out_dim: 512,
                        kernel: 3,
                        stride: 2,
                    },
                    OmniasrCtcConvLayerDesc {
                        out_dim: 512,
                        kernel: 2,
                        stride: 2,
                    },
                    OmniasrCtcConvLayerDesc {
                        out_dim: 512,
                        kernel: 2,
                        stride: 2,
                    },
                ],
                feature_extractor_bias: true,
                feature_extractor_layer_norm_convs: true,
                layer_norm_features: false,
                pos_conv_kernel_size: 128,
                num_pos_conv_groups: 16,
                pos_encoder_depth: 1,
                use_conformer: false,
                max_seq_len: 4096,
            },
            head: OmniasrCtcHeadConfig {
                target_vocab_size: 9812,
                blank_id: 0,
                final_dropout_p: 0.0,
            },
            sample_rate: OMNIASR_CTC_SAMPLE_RATE,
        }
    }

    /// Variant-aware constructor — dispatches to `omniasr_ctc_300m()` /
    /// `omniasr_ctc_1b()` / `omniasr_ctc_7b()` based on the passed
    /// [`OmniasrCtcVariant`]. Convenience for callers that already
    /// carry the variant tag (a converter side-car, a CLI arg).
    #[must_use]
    pub fn for_variant(variant: OmniasrCtcVariant) -> Self {
        match variant {
            OmniasrCtcVariant::M300 => Self::omniasr_ctc_300m(),
            OmniasrCtcVariant::B1 => Self::omniasr_ctc_1b(),
            OmniasrCtcVariant::B7 => Self::omniasr_ctc_7b(),
        }
    }

    /// Miniature well-formed config for shape / stability tests. Dims
    /// are tiny so synthesized-weight builds fit in KB; the *shape
    /// relationships* (MHA head split, even head_dim, blank at 0,
    /// feature_extractor 7-layer stem with distinct kernel/stride axes,
    /// pos_conv group divides model_dim) mirror the real model.
    #[must_use]
    pub fn tiny_for_tests() -> Self {
        Self {
            encoder: OmniasrCtcEncoderConfig {
                model_dim: 16,
                num_encoder_layers: 2,
                num_encoder_attn_heads: 4,
                ffn_inner_dim: 32,
                feature_dim: 8,
                // Mirror the real 7-layer stride pattern but with tiny
                // channel widths, so the total stride still = 5*2^6 = 320.
                feature_extractor_layer_descs: [
                    OmniasrCtcConvLayerDesc {
                        out_dim: 8,
                        kernel: 10,
                        stride: 5,
                    },
                    OmniasrCtcConvLayerDesc {
                        out_dim: 8,
                        kernel: 3,
                        stride: 2,
                    },
                    OmniasrCtcConvLayerDesc {
                        out_dim: 8,
                        kernel: 3,
                        stride: 2,
                    },
                    OmniasrCtcConvLayerDesc {
                        out_dim: 8,
                        kernel: 3,
                        stride: 2,
                    },
                    OmniasrCtcConvLayerDesc {
                        out_dim: 8,
                        kernel: 3,
                        stride: 2,
                    },
                    OmniasrCtcConvLayerDesc {
                        out_dim: 8,
                        kernel: 2,
                        stride: 2,
                    },
                    OmniasrCtcConvLayerDesc {
                        out_dim: 8,
                        kernel: 2,
                        stride: 2,
                    },
                ],
                feature_extractor_bias: true,
                feature_extractor_layer_norm_convs: true,
                layer_norm_features: false,
                pos_conv_kernel_size: 4,
                num_pos_conv_groups: 4,
                pos_encoder_depth: 1,
                use_conformer: false,
                max_seq_len: 64,
            },
            head: OmniasrCtcHeadConfig {
                target_vocab_size: 5,
                blank_id: 0,
                final_dropout_p: 0.0,
            },
            sample_rate: OMNIASR_CTC_SAMPLE_RATE,
        }
    }

    /// Rejects `0`-placeholder / ill-formed configs before any forward
    /// runs (FR-EX-08 — a shape-only converter path fails loudly here,
    /// not deep inside a GEMM).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] naming the offending field.
    pub fn validate_for_forward(&self) -> Result<()> {
        // ---- Encoder ------------------------------------------------------
        if !self.encoder.is_well_formed() {
            return Err(VokraError::InvalidArgument(format!(
                "omniasr-ctc config: encoder ill-formed \
                 (num_encoder_layers={}, model_dim={}, num_encoder_attn_heads={}) — \
                 expected model_dim % num_encoder_attn_heads == 0, all fields > 0",
                self.encoder.num_encoder_layers,
                self.encoder.model_dim,
                self.encoder.num_encoder_attn_heads,
            )));
        }
        if self.encoder.num_encoder_layers == 0 {
            return Err(VokraError::InvalidArgument(
                "omniasr-ctc config: encoder.num_encoder_layers must be > 0".to_owned(),
            ));
        }
        if self.encoder.head_dim() % 2 != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "omniasr-ctc config: encoder head_dim {} must be even \
                 (attention K/V pair layout)",
                self.encoder.head_dim(),
            )));
        }
        if self.encoder.ffn_inner_dim == 0 {
            return Err(VokraError::InvalidArgument(
                "omniasr-ctc config: encoder.ffn_inner_dim must be > 0".to_owned(),
            ));
        }
        if self.encoder.feature_dim == 0 {
            return Err(VokraError::InvalidArgument(
                "omniasr-ctc config: encoder.feature_dim must be > 0".to_owned(),
            ));
        }
        if self.encoder.max_seq_len == 0 {
            return Err(VokraError::InvalidArgument(
                "omniasr-ctc config: encoder.max_seq_len must be > 0".to_owned(),
            ));
        }
        if self.encoder.pos_conv_kernel_size == 0 {
            return Err(VokraError::InvalidArgument(
                "omniasr-ctc config: encoder.pos_conv_kernel_size must be > 0".to_owned(),
            ));
        }
        if self.encoder.num_pos_conv_groups == 0
            || self.encoder.model_dim % self.encoder.num_pos_conv_groups != 0
        {
            return Err(VokraError::InvalidArgument(format!(
                "omniasr-ctc config: encoder.num_pos_conv_groups={} must be > 0 and \
                 divide encoder.model_dim={}",
                self.encoder.num_pos_conv_groups, self.encoder.model_dim,
            )));
        }
        if self.encoder.pos_encoder_depth == 0 {
            return Err(VokraError::InvalidArgument(
                "omniasr-ctc config: encoder.pos_encoder_depth must be > 0".to_owned(),
            ));
        }
        // Every feature extractor layer must have positive kernel / stride
        // and a non-zero output channel. The 7-layer count is pinned by the
        // array type; each descriptor is inspected loudly.
        for (i, d) in self
            .encoder
            .feature_extractor_layer_descs
            .iter()
            .enumerate()
        {
            if d.out_dim == 0 || d.kernel == 0 || d.stride == 0 {
                return Err(VokraError::InvalidArgument(format!(
                    "omniasr-ctc config: encoder.feature_extractor_layer_descs[{i}] = \
                     (out_dim={}, kernel={}, stride={}) — every field must be > 0",
                    d.out_dim, d.kernel, d.stride,
                )));
            }
        }
        // The last conv layer's output channels must match `feature_dim`
        // (the projection input; the fairseq2 default has every layer at
        // 512 = feature_dim).
        let last_out = self.encoder.feature_extractor_layer_descs
            [OMNIASR_CTC_NUM_FEATURE_EXTRACTOR_LAYERS - 1]
            .out_dim;
        if last_out != self.encoder.feature_dim {
            return Err(VokraError::InvalidArgument(format!(
                "omniasr-ctc config: last feature_extractor layer out_dim={} must equal \
                 encoder.feature_dim={} (feature projection input width)",
                last_out, self.encoder.feature_dim,
            )));
        }

        // ---- CTC head -----------------------------------------------------
        if self.head.target_vocab_size == 0 {
            return Err(VokraError::InvalidArgument(
                "omniasr-ctc config: head.target_vocab_size must be > 0".to_owned(),
            ));
        }
        // The CTC blank id lives inside the vocab head width
        // `[0, target_vocab_size)`; the fairseq2 wav2vec 2.0 convention
        // puts the blank at index 0 (not at the tail — that is the NeMo
        // convention that Parakeet-CTC follows).
        if (self.head.blank_id as usize) >= self.head.target_vocab_size {
            return Err(VokraError::InvalidArgument(format!(
                "omniasr-ctc config: head.blank_id={} must be < target_vocab_size={}",
                self.head.blank_id, self.head.target_vocab_size,
            )));
        }
        if !self.head.final_dropout_p.is_finite() {
            return Err(VokraError::InvalidArgument(format!(
                "omniasr-ctc config: head.final_dropout_p={} must be finite",
                self.head.final_dropout_p,
            )));
        }
        Ok(())
    }

    /// Reads the strict `vokra.omniasr_ctc.*` chunk group from `gguf`
    /// and assembles the resolved config.
    ///
    /// Every mandatory u32 hparam is required — a converter that fails
    /// to stamp any one is a converter bug, not a runtime silent-default
    /// (FR-EX-08). The 7-layer feature-extractor stem descriptor is
    /// walked via `count + N × (out_dim, kernel, stride)` and
    /// cross-checked against [`OMNIASR_CTC_NUM_FEATURE_EXTRACTOR_LAYERS`]
    /// (loud reject on mismatch — a shape-only silent 8-layer variant
    /// cannot slip in).
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when any mandatory `vokra.omniasr_ctc.*`
    ///   u32 chunk is absent.
    /// - [`VokraError::ModelLoad`] when
    ///   `vokra.omniasr_ctc.arch.encoder.feature_extractor_layer_count`
    ///   does not equal [`OMNIASR_CTC_NUM_FEATURE_EXTRACTOR_LAYERS`].
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        fn req_u32(gguf: &GgufFile, key: &str) -> Result<u32> {
            gguf.get(key)
                .and_then(vokra_core::gguf::GgufMetadataValue::as_u64)
                .map(|v| v as u32)
                .ok_or_else(|| {
                    VokraError::ModelLoad(format!(
                        "omniasr-ctc: GGUF is missing required u32 chunk `{key}` — the \
                         upstream `facebook/omniASR-CTC-1B` fairseq2 registry walk \
                         (fetched 2026-07-24) transcribes every wav2vec 2.0 axis; a \
                         converter that fails to stamp one is a converter bug, not a \
                         runtime silent-default (FR-EX-08). Re-run `vokra-cli convert \
                         --model omniasr-ctc` against a `facebook/omniASR-CTC-1B` \
                         safetensors — primary source: \
                         https://huggingface.co/facebook/omniASR-CTC-1B."
                    ))
                })
        }

        // Read the fixed-count feature-extractor stem descriptor first —
        // a `count` mismatch is a loud reject before we even try to walk
        // the per-layer prefix keys.
        let feature_layer_count = req_u32(gguf, KEY_ENC_FEATURE_LAYERS)? as usize;
        if feature_layer_count != OMNIASR_CTC_NUM_FEATURE_EXTRACTOR_LAYERS {
            return Err(VokraError::ModelLoad(format!(
                "omniasr-ctc: GGUF chunk `{KEY_ENC_FEATURE_LAYERS}`={feature_layer_count} \
                 does not equal the fairseq2 wav2vec 2.0 fixed 7-layer stem \
                 (OMNIASR_CTC_NUM_FEATURE_EXTRACTOR_LAYERS={OMNIASR_CTC_NUM_FEATURE_EXTRACTOR_LAYERS}) — \
                 the \"1b\" arch does not override the default stem, so every \
                 omniASR size shares the same 7-layer count. A GGUF with a \
                 different count was produced by a converter that broke the \
                 pinned-count invariant (FR-EX-08)."
            )));
        }
        let mut feature_extractor_layer_descs = [OmniasrCtcConvLayerDesc {
            out_dim: 0,
            kernel: 0,
            stride: 0,
        };
            OMNIASR_CTC_NUM_FEATURE_EXTRACTOR_LAYERS];
        for (i, desc) in feature_extractor_layer_descs.iter_mut().enumerate() {
            let out_key = format!("{KEY_ENC_FEATURE_OUT_PREFIX}{i}");
            let kernel_key = format!("{KEY_ENC_FEATURE_KERNEL_PREFIX}{i}");
            let stride_key = format!("{KEY_ENC_FEATURE_STRIDE_PREFIX}{i}");
            *desc = OmniasrCtcConvLayerDesc {
                out_dim: req_u32(gguf, &out_key)? as usize,
                kernel: req_u32(gguf, &kernel_key)? as usize,
                stride: req_u32(gguf, &stride_key)? as usize,
            };
        }

        Ok(Self {
            encoder: OmniasrCtcEncoderConfig {
                model_dim: req_u32(gguf, KEY_ENC_MODEL_DIM)? as usize,
                num_encoder_layers: req_u32(gguf, KEY_ENC_N_LAYER)? as usize,
                num_encoder_attn_heads: req_u32(gguf, KEY_ENC_N_HEAD)? as usize,
                ffn_inner_dim: req_u32(gguf, KEY_ENC_FFN_INNER)? as usize,
                feature_dim: req_u32(gguf, KEY_ENC_FEATURE_DIM)? as usize,
                feature_extractor_layer_descs,
                feature_extractor_bias: req_u32(gguf, KEY_ENC_FEATURE_BIAS)? != 0,
                feature_extractor_layer_norm_convs: req_u32(gguf, KEY_ENC_FEATURE_LN_CONVS)? != 0,
                layer_norm_features: req_u32(gguf, KEY_ENC_LN_FEATURES)? != 0,
                pos_conv_kernel_size: req_u32(gguf, KEY_ENC_POS_KERNEL)? as usize,
                num_pos_conv_groups: req_u32(gguf, KEY_ENC_POS_GROUPS)? as usize,
                pos_encoder_depth: req_u32(gguf, KEY_ENC_POS_DEPTH)? as usize,
                use_conformer: req_u32(gguf, KEY_ENC_USE_CONFORMER)? != 0,
                max_seq_len: req_u32(gguf, KEY_ENC_MAX_SEQ_LEN)? as usize,
            },
            head: OmniasrCtcHeadConfig {
                target_vocab_size: req_u32(gguf, KEY_HEAD_VOCAB_SIZE)? as usize,
                blank_id: req_u32(gguf, KEY_HEAD_BLANK_ID)?,
                // `final_dropout_p` is not stamped by the converter
                // (train-time only; inference forward is dropout-free).
                // Default to 0.0 — the validator accepts any finite
                // value and this is the inference-path value.
                final_dropout_p: 0.0,
            },
            sample_rate: req_u32(gguf, KEY_SAMPLE_RATE)?,
        })
    }
}

// ---------------------------------------------------------------------------
// Weights
// ---------------------------------------------------------------------------

/// Per-Conv1D layer in the waveform feature extractor. Weight shape is
/// `[out_dim, in_dim, kernel]`; bias is `[out_dim]` when
/// `feature_extractor_bias=true`; LayerNorm gamma/beta are
/// `[out_dim]` each when `feature_extractor_layer_norm_convs=true`.
///
/// `in_dim` for layer 0 is 1 (raw mono waveform); for layer i>0 it is
/// the previous layer's `out_dim`.
#[derive(Debug, Clone)]
pub struct OmniasrCtcFeatureExtractorLayerWeights {
    /// `[out_dim, in_dim, kernel]`.
    pub conv_w: Vec<f32>,
    /// `[out_dim]` — Some iff `feature_extractor_bias=true`.
    pub conv_b: Option<Vec<f32>>,
    /// `[out_dim]` — LayerNorm gamma, Some iff
    /// `feature_extractor_layer_norm_convs=true`.
    pub norm_gamma: Option<Vec<f32>>,
    /// `[out_dim]` — LayerNorm beta, Some iff
    /// `feature_extractor_layer_norm_convs=true`.
    pub norm_beta: Option<Vec<f32>>,
}

/// The feature projection: fairseq2's post-extraction LayerNorm followed by
/// Linear from `feature_dim` to `model_dim`.  The upstream `large_lv60k`
/// metadata calls `layer_norm_features=false` for the *feature extractor*
/// branch, but its frontend still owns this post-extraction LayerNorm; the
/// pinned 807-tensor payload includes these two tensors.
#[derive(Debug, Clone)]
pub struct OmniasrCtcFeatureProjectionWeights {
    /// `[feature_dim]` — post-extraction LayerNorm gamma.
    pub norm_gamma: Option<Vec<f32>>,
    /// `[feature_dim]` — post-extraction LayerNorm beta.
    pub norm_beta: Option<Vec<f32>>,
    /// `[model_dim, feature_dim]` in output-major Linear layout.
    pub linear_w: Vec<f32>,
    /// `[model_dim]`.
    pub linear_b: Vec<f32>,
}

/// Grouped-Conv1D positional encoder. The wav2vec 2.0 conv positional
/// encoder is a single 1D grouped conv over the sequence dim.
///
/// Weight shape is `[model_dim, model_dim / num_pos_conv_groups,
/// pos_conv_kernel_size]`; bias is `[model_dim]`.
#[derive(Debug, Clone)]
pub struct OmniasrCtcPosEncoderLayerWeights {
    /// `[model_dim, model_dim / num_pos_conv_groups, pos_conv_kernel_size]`.
    pub conv_w: Vec<f32>,
    /// `[model_dim]`.
    pub conv_b: Vec<f32>,
}

/// Per-encoder-block scaffold weights (pre-norm Transformer MHA + FFN
/// branches). `attn_norm` / `ffn_norm` are the pre-LayerNorm gamma+beta;
/// `qkv_proj` is the fused Q/K/V projection (MHA — for a hypothetical
/// future GQA flavor the shape would be
/// `[model_dim, model_dim + 2 * kv_hidden]`); every projection carries
/// a bias (fairseq2 wav2vec 2.0 default).
#[derive(Debug, Clone)]
pub struct OmniasrCtcEncoderBlockWeights {
    /// Attention pre-norm γ, shape `[model_dim]`.
    pub attn_norm_gamma: Vec<f32>,
    /// Attention pre-norm β, shape `[model_dim]`.
    pub attn_norm_beta: Vec<f32>,
    /// Fused Q/K/V projection, output-major shape `[3*model_dim, model_dim]` (MHA).
    pub qkv_proj: Vec<f32>,
    /// Fused Q/K/V bias, shape `[3*model_dim]`.
    pub qkv_bias: Vec<f32>,
    /// Attention output projection, output-major shape `[model_dim, model_dim]`.
    pub attn_out: Vec<f32>,
    /// Attention output bias, shape `[model_dim]`.
    pub attn_out_bias: Vec<f32>,
    /// FFN pre-norm γ, shape `[model_dim]`.
    pub ffn_norm_gamma: Vec<f32>,
    /// FFN pre-norm β, shape `[model_dim]`.
    pub ffn_norm_beta: Vec<f32>,
    /// FFN hidden projection, output-major shape `[ffn_inner_dim, model_dim]`.
    pub ffn_fc1: Vec<f32>,
    /// FFN hidden bias, shape `[ffn_inner_dim]`.
    pub ffn_fc1_bias: Vec<f32>,
    /// FFN output projection, output-major shape `[model_dim, ffn_inner_dim]`.
    pub ffn_fc2: Vec<f32>,
    /// FFN output bias, shape `[model_dim]`.
    pub ffn_fc2_bias: Vec<f32>,
}

/// CTC head scaffold: a single Linear from encoder `model_dim` to
/// `target_vocab_size`, plus a bias (fairseq2 wav2vec 2.0 default has
/// `final_proj_bias=True`).
#[derive(Debug, Clone)]
pub struct OmniasrCtcHeadWeights {
    /// `[target_vocab_size, model_dim]` in output-major Linear layout — CTC
    /// vocab projection (blank inclusive at index `blank_id=0`).
    pub vocab_head: Vec<f32>,
    /// `[target_vocab_size]` — CTC vocab bias.
    pub vocab_bias: Vec<f32>,
}

/// omniASR-CTC weight store: feature extractor (7 layers) + feature
/// projection + positional encoder + Transformer blocks + final norm +
/// CTC head.
///
/// [`Self::synthesized`] builds a deterministic fixture (SplitMix64 +
/// Xavier) against `config` so shape / dtype / size can be exercised
/// without the real HF checkpoint. Real-checkpoint binding consumes the
/// complete audited VAST manifest; this fixture is never accepted by
/// `from_gguf`.
#[derive(Debug, Clone)]
pub struct OmniasrCtcWeights {
    /// 7 waveform-feature-extractor Conv1D layers, in order.
    pub feature_extractor: Vec<OmniasrCtcFeatureExtractorLayerWeights>,
    /// Feature projection (post-extraction LayerNorm then Linear).
    pub feature_projection: OmniasrCtcFeatureProjectionWeights,
    /// Optional fairseq2 frontend model-dimension LayerNorm, applied after
    /// positional encoding and before the Transformer stack. Present iff
    /// `encoder.layer_norm_features=true`.
    pub frontend_model_dim_norm_gamma: Option<Vec<f32>>,
    /// Optional fairseq2 frontend model-dimension LayerNorm beta.
    pub frontend_model_dim_norm_beta: Option<Vec<f32>>,
    /// Positional Conv1D encoder blocks in order (depth = 1 for the 1B).
    pub pos_encoder_layers: Vec<OmniasrCtcPosEncoderLayerWeights>,
    /// Transformer encoder blocks in order.
    pub encoder_blocks: Vec<OmniasrCtcEncoderBlockWeights>,
    /// Encoder-out LayerNorm γ, shape `[model_dim]` (pre-norm
    /// architecture — the final norm sits outside the last block).
    pub encoder_final_norm_gamma: Vec<f32>,
    /// Encoder-out LayerNorm β, shape `[model_dim]`.
    pub encoder_final_norm_beta: Vec<f32>,
    /// CTC head.
    pub head: OmniasrCtcHeadWeights,
    /// `true` when built by [`Self::synthesized`] — never a real
    /// upstream checkpoint. Real-checkpoint bindings set this to
    /// `false`.
    pub is_synthesized: bool,
}

impl OmniasrCtcWeights {
    /// Builds a deterministic synthesized fixture from `config` and
    /// `seed`.
    ///
    /// Draws are Xavier-uniform ± `sqrt(6 / (fan_in + fan_out))` via a
    /// [`SplitMix64`] stream — reproducible, allocation-only, zero-dep.
    /// Every LayerNorm γ starts at `1.0`; every LayerNorm β and every
    /// bias starts at `0.0`.
    ///
    /// Feature-extractor per-layer bias / LayerNorm are `Some` iff the
    /// corresponding config flag is on
    /// (`feature_extractor_bias=true` / `feature_extractor_layer_norm_convs=true`
    /// — the omniASR / large_lv60k case). The fairseq2 frontend's
    /// post-extraction LayerNorm is always present, including the
    /// `large_lv60k` `layer_norm_features=false` preset. Its separate
    /// model-dimension LayerNorm is synthesized iff `layer_norm_features=true`.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] if `config.validate_for_forward`
    /// fails.
    pub fn synthesized(config: &OmniasrCtcConfig, seed: u64) -> Result<Self> {
        config.validate_for_forward()?;
        let mut rng = SplitMix64::new(seed);
        let enc = &config.encoder;
        let head = &config.head;
        let d_enc = enc.model_dim;
        let ffn = enc.ffn_inner_dim;
        let vocab = head.target_vocab_size;
        let feat_dim = enc.feature_dim;

        // -- Feature extractor: 7 Conv1D layers. Layer 0's in_dim is 1
        //    (raw mono waveform); every subsequent layer's in_dim is
        //    the previous layer's out_dim.
        let mut feature_extractor = Vec::with_capacity(OMNIASR_CTC_NUM_FEATURE_EXTRACTOR_LAYERS);
        let mut in_dim: usize = 1;
        for d in &enc.feature_extractor_layer_descs {
            let count = d.out_dim * in_dim * d.kernel;
            let conv_w = xavier(&mut rng, count, in_dim * d.kernel, d.out_dim);
            let conv_b = enc.feature_extractor_bias.then(|| vec![0.0; d.out_dim]);
            let (norm_gamma, norm_beta) = if enc.feature_extractor_layer_norm_convs {
                (Some(vec![1.0; d.out_dim]), Some(vec![0.0; d.out_dim]))
            } else {
                (None, None)
            };
            feature_extractor.push(OmniasrCtcFeatureExtractorLayerWeights {
                conv_w,
                conv_b,
                norm_gamma,
                norm_beta,
            });
            in_dim = d.out_dim;
        }

        // -- Feature projection: fairseq2's post-extraction LayerNorm is
        //    followed by Linear from feature_dim to model_dim.  The
        //    `layer_norm_features` preset controls the convolutional
        //    extractor mode and does not remove this frontend norm.
        let (norm_gamma, norm_beta) = (Some(vec![1.0; feat_dim]), Some(vec![0.0; feat_dim]));
        let feature_projection = OmniasrCtcFeatureProjectionWeights {
            norm_gamma,
            norm_beta,
            linear_w: xavier(&mut rng, feat_dim * d_enc, feat_dim, d_enc),
            linear_b: vec![0.0; d_enc],
        };
        let (frontend_model_dim_norm_gamma, frontend_model_dim_norm_beta) =
            if enc.layer_norm_features {
                (Some(vec![1.0; d_enc]), Some(vec![0.0; d_enc]))
            } else {
                (None, None)
            };

        // -- Positional encoder: grouped Conv1D, `pos_encoder_depth`
        //    blocks (depth = 1 for the 1B). Kernel over the sequence
        //    dim; input channels per group = model_dim / n_groups.
        let per_group = d_enc / enc.num_pos_conv_groups;
        let k = enc.pos_conv_kernel_size;
        let mut pos_encoder_layers = Vec::with_capacity(enc.pos_encoder_depth);
        for _ in 0..enc.pos_encoder_depth {
            pos_encoder_layers.push(OmniasrCtcPosEncoderLayerWeights {
                conv_w: xavier(&mut rng, d_enc * per_group * k, per_group * k, d_enc),
                conv_b: vec![0.0; d_enc],
            });
        }

        // -- Transformer encoder blocks.
        let mut encoder_blocks = Vec::with_capacity(enc.num_encoder_layers);
        for _ in 0..enc.num_encoder_layers {
            encoder_blocks.push(OmniasrCtcEncoderBlockWeights {
                attn_norm_gamma: vec![1.0; d_enc],
                attn_norm_beta: vec![0.0; d_enc],
                qkv_proj: xavier(&mut rng, d_enc * 3 * d_enc, d_enc, 3 * d_enc),
                qkv_bias: vec![0.0; 3 * d_enc],
                attn_out: xavier(&mut rng, d_enc * d_enc, d_enc, d_enc),
                attn_out_bias: vec![0.0; d_enc],
                ffn_norm_gamma: vec![1.0; d_enc],
                ffn_norm_beta: vec![0.0; d_enc],
                ffn_fc1: xavier(&mut rng, d_enc * ffn, d_enc, ffn),
                ffn_fc1_bias: vec![0.0; ffn],
                ffn_fc2: xavier(&mut rng, ffn * d_enc, ffn, d_enc),
                ffn_fc2_bias: vec![0.0; d_enc],
            });
        }
        let encoder_final_norm_gamma = vec![1.0; d_enc];
        let encoder_final_norm_beta = vec![0.0; d_enc];

        // -- CTC head — single Linear from model_dim to target_vocab_size
        //    with bias (fairseq2 wav2vec 2.0 default: final_proj_bias=True).
        let head_w = OmniasrCtcHeadWeights {
            vocab_head: xavier(&mut rng, d_enc * vocab, d_enc, vocab),
            vocab_bias: vec![0.0; vocab],
        };

        Ok(Self {
            feature_extractor,
            feature_projection,
            frontend_model_dim_norm_gamma,
            frontend_model_dim_norm_beta,
            pos_encoder_layers,
            encoder_blocks,
            encoder_final_norm_gamma,
            encoder_final_norm_beta,
            head: head_w,
            is_synthesized: true,
        })
    }
}

/// Xavier-uniform draw of `count` `f32`s in `[-a, +a]` where
/// `a = sqrt(6 / (fan_in + fan_out))`. Deterministic under a fixed
/// `rng`.
fn xavier(rng: &mut SplitMix64, count: usize, fan_in: usize, fan_out: usize) -> Vec<f32> {
    let a = (6.0 / (fan_in + fan_out).max(1) as f32).sqrt();
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        // Map the top 24 bits of the u64 stream to a f32 in [0, 1).
        let raw = (rng.next_u64() >> 40) as u32;
        let u01 = (raw as f32) / ((1u32 << 24) as f32);
        out.push((u01 * 2.0 - 1.0) * a);
    }
    out
}

fn normalize_omniasr_pcm(pcm: &[f32]) -> Vec<f32> {
    let mean = pcm.iter().copied().sum::<f32>() / pcm.len() as f32;
    let variance = pcm
        .iter()
        .map(|&value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f32>()
        / pcm.len() as f32;
    let scale = 1.0 / (variance + OMNIASR_CTC_NORMALIZATION_EPS).sqrt();
    pcm.iter().map(|&value| (value - mean) * scale).collect()
}

/// The complete 807-entry contract extracted from the authoritative VAST
/// manifest.  Keeping this as an explicit name/shape map means a matching
/// tensor count can never authenticate a different fairseq2 state dict.
fn omniasr_manifest() -> BTreeMap<String, Vec<u64>> {
    let mut expected = BTreeMap::new();
    let mut add = |name: String, shape: &[u64]| {
        expected.insert(name, shape.to_vec());
    };
    add("encoder.layer_norm.bias".to_owned(), &[1280]);
    add("encoder.layer_norm.weight".to_owned(), &[1280]);
    add("final_proj.bias".to_owned(), &[9812]);
    add("final_proj.weight".to_owned(), &[9812, 1280]);

    for i in 0..48 {
        let p = format!("encoder.layers.{i}");
        add(format!("{p}.ffn.inner_proj.bias"), &[5120]);
        add(format!("{p}.ffn.inner_proj.weight"), &[5120, 1280]);
        add(format!("{p}.ffn.output_proj.bias"), &[1280]);
        add(format!("{p}.ffn.output_proj.weight"), &[1280, 5120]);
        add(format!("{p}.ffn_layer_norm.bias"), &[1280]);
        add(format!("{p}.ffn_layer_norm.weight"), &[1280]);
        for projection in ["k", "output", "q", "v"] {
            add(format!("{p}.self_attn.{projection}_proj.bias"), &[1280]);
            add(
                format!("{p}.self_attn.{projection}_proj.weight"),
                &[1280, 1280],
            );
        }
        add(format!("{p}.self_attn_layer_norm.bias"), &[1280]);
        add(format!("{p}.self_attn_layer_norm.weight"), &[1280]);
    }

    let kernels = [10u64, 3, 3, 3, 3, 2, 2];
    for (i, &kernel) in kernels.iter().enumerate() {
        let input = if i == 0 { 1 } else { 512 };
        let p = format!("encoder_frontend.feature_extractor.layers.{i}");
        add(format!("{p}.conv.bias"), &[512]);
        add(format!("{p}.conv.weight"), &[512, input, kernel]);
        add(format!("{p}.layer_norm.bias"), &[512]);
        add(format!("{p}.layer_norm.weight"), &[512]);
    }
    add("encoder_frontend.model_dim_proj.bias".to_owned(), &[1280]);
    add(
        "encoder_frontend.model_dim_proj.weight".to_owned(),
        &[1280, 512],
    );
    add("encoder_frontend.pos_encoder.conv.bias".to_owned(), &[1280]);
    add(
        "encoder_frontend.pos_encoder.conv.weight_g".to_owned(),
        &[1, 1, 128],
    );
    add(
        "encoder_frontend.pos_encoder.conv.weight_v".to_owned(),
        &[1280, 80, 128],
    );
    add(
        "encoder_frontend.post_extract_layer_norm.bias".to_owned(),
        &[512],
    );
    add(
        "encoder_frontend.post_extract_layer_norm.weight".to_owned(),
        &[512],
    );
    expected
}

fn require_omniasr_provenance(file: &GgufFile) -> Result<()> {
    let required = [
        (chunks::KEY_MODEL_NAME, "omniasr-ctc-1b"),
        (chunks::KEY_PROVENANCE_MODEL_ID, OMNIASR_CTC_MODEL_ID),
        (
            chunks::KEY_PROVENANCE_SOURCE,
            "https://huggingface.co/facebook/omniASR-CTC-1B",
        ),
        (KEY_PROVENANCE_UPSTREAM_REVISION, OMNIASR_CTC_HF_REVISION),
        (KEY_OMNIASR_SOURCE_SHA256, OMNIASR_CTC_SOURCE_SHA256),
        (KEY_OMNIASR_PREPARED_SHA256, OMNIASR_CTC_PREPARED_SHA256),
        (chunks::KEY_PROVENANCE_LICENSE, "Apache-2.0"),
        (
            chunks::KEY_PROVENANCE_WEIGHT_LICENSE,
            LicenseClass::Permissive.as_str(),
        ),
    ];
    for (key, expected) in required {
        let actual = file.get(key).and_then(|value| value.as_str());
        if actual != Some(expected) {
            return Err(VokraError::ModelLoad(format!(
                "omniasr-ctc: provenance `{key}` is {actual:?}, expected exact {expected:?}"
            )));
        }
    }
    Ok(())
}

fn validate_omniasr_manifest(file: &GgufFile) -> Result<()> {
    let expected = omniasr_manifest();
    if expected.len() != OMNIASR_CTC_EXPECTED_TENSOR_COUNT {
        return Err(VokraError::ModelLoad(format!(
            "omniasr-ctc: internal manifest has {} entries, expected {}",
            expected.len(),
            OMNIASR_CTC_EXPECTED_TENSOR_COUNT
        )));
    }
    validate_manifest_contract(file, &expected)
}

fn validate_manifest_contract(
    file: &GgufFile,
    expected: &BTreeMap<String, Vec<u64>>,
) -> Result<()> {
    if file.tensors().len() != expected.len() {
        return Err(VokraError::ModelLoad(format!(
            "omniasr-ctc: GGUF contains {} tensors, expected exact audited manifest size {}",
            file.tensors().len(),
            expected.len()
        )));
    }
    let mut seen = BTreeSet::new();
    for info in file.tensors() {
        if !seen.insert(info.name.as_str()) {
            return Err(VokraError::ModelLoad(format!(
                "omniasr-ctc: duplicate tensor `{}`",
                info.name
            )));
        }
        let Some(shape) = expected.get(&info.name) else {
            return Err(VokraError::ModelLoad(format!(
                "omniasr-ctc: unexpected tensor `{}`",
                info.name
            )));
        };
        if &info.dimensions != shape {
            return Err(VokraError::ModelLoad(format!(
                "omniasr-ctc: tensor `{}` has shape {:?}, expected {:?}",
                info.name, info.dimensions, shape
            )));
        }
        if info.dtype != GgmlType::F32 {
            return Err(VokraError::ModelLoad(format!(
                "omniasr-ctc: tensor `{}` has dtype {:?}, expected F32 from the audited manifest",
                info.name, info.dtype
            )));
        }
    }
    if seen.len() != expected.len() {
        return Err(VokraError::ModelLoad(
            "omniasr-ctc: audited manifest contains missing tensor names".to_owned(),
        ));
    }
    Ok(())
}

fn load_omniasr_tensor(file: &GgufFile, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
    let values = load_tensor(file, "omniasr-ctc", name, shape)?;
    if values.iter().any(|value| !value.is_finite()) {
        return Err(VokraError::ModelLoad(format!(
            "omniasr-ctc: tensor `{name}` contains non-finite values"
        )));
    }
    Ok(values)
}

/// Materializes the legacy PyTorch/fairseq2 `weight_norm(..., dim=2)` pair.
/// With `weight_v [out, in, kernel]` and `weight_g [1, 1, kernel]`, each
/// kernel position is one normalization vector over the first two axes. This
/// is the same axis convention recorded by the official fairseq2 Wav2Vec2
/// positional Conv1D, not an inferred per-output-channel normalization.
fn materialize_positional_weight(file: &GgufFile) -> Result<Vec<f32>> {
    let g = load_omniasr_tensor(
        file,
        "encoder_frontend.pos_encoder.conv.weight_g",
        &[1, 1, 128],
    )?;
    let v = load_omniasr_tensor(
        file,
        "encoder_frontend.pos_encoder.conv.weight_v",
        &[1280, 80, 128],
    )?;
    materialize_positional_weight_values(&g, &v, 1280, 80, 128)
}

fn materialize_positional_weight_values(
    g: &[f32],
    v: &[f32],
    out_channels: usize,
    in_channels: usize,
    kernel_size: usize,
) -> Result<Vec<f32>> {
    if g.len() != kernel_size || v.len() != out_channels * in_channels * kernel_size {
        return Err(VokraError::ModelLoad(
            "omniasr-ctc: positional Conv1D weight_norm g/v shape mismatch".to_owned(),
        ));
    }
    let mut output = vec![0.0; v.len()];
    for (kernel, &g_value) in g.iter().enumerate() {
        let mut norm_sq = 0.0f32;
        for out in 0..out_channels {
            for input in 0..in_channels {
                let index = (out * in_channels + input) * kernel_size + kernel;
                norm_sq += v[index] * v[index];
            }
        }
        let norm = norm_sq.sqrt();
        if !norm.is_finite() || norm == 0.0 || !g_value.is_finite() {
            return Err(VokraError::ModelLoad(format!(
                "omniasr-ctc: positional Conv1D weight_norm kernel {kernel} has invalid norm/g"
            )));
        }
        let scale = g_value / norm;
        for out in 0..out_channels {
            for input in 0..in_channels {
                let index = (out * in_channels + input) * kernel_size + kernel;
                output[index] = v[index] * scale;
            }
        }
    }
    if output.iter().any(|value| !value.is_finite()) {
        return Err(VokraError::ModelLoad(
            "omniasr-ctc: materialized positional Conv1D contains non-finite values".to_owned(),
        ));
    }
    Ok(output)
}

fn fuse_qkv(q: Vec<f32>, k: Vec<f32>, v: Vec<f32>, model_dim: usize) -> Result<Vec<f32>> {
    let expected = model_dim * model_dim;
    if q.len() != expected || k.len() != expected || v.len() != expected {
        return Err(VokraError::ModelLoad(
            "omniasr-ctc: q/k/v projection shape mismatch".to_owned(),
        ));
    }
    let mut fused = Vec::with_capacity(expected * 3);
    fused.extend(q);
    fused.extend(k);
    fused.extend(v);
    Ok(fused)
}

fn fuse_qkv_bias(q: Vec<f32>, k: Vec<f32>, v: Vec<f32>, model_dim: usize) -> Result<Vec<f32>> {
    if q.len() != model_dim || k.len() != model_dim || v.len() != model_dim {
        return Err(VokraError::ModelLoad(
            "omniasr-ctc: q/k/v bias shape mismatch".to_owned(),
        ));
    }
    let mut fused = Vec::with_capacity(model_dim * 3);
    fused.extend(q);
    fused.extend(k);
    fused.extend(v);
    Ok(fused)
}

fn bind_omniasr_weights(file: &GgufFile, cfg: &OmniasrCtcConfig) -> Result<OmniasrCtcWeights> {
    let enc = &cfg.encoder;
    let mut feature_extractor = Vec::with_capacity(7);
    for (i, desc) in enc.feature_extractor_layer_descs.iter().enumerate() {
        let input = if i == 0 { 1 } else { 512 };
        let p = format!("encoder_frontend.feature_extractor.layers.{i}");
        feature_extractor.push(OmniasrCtcFeatureExtractorLayerWeights {
            conv_w: load_omniasr_tensor(
                file,
                &format!("{p}.conv.weight"),
                &[desc.out_dim, input, desc.kernel],
            )?,
            conv_b: Some(load_omniasr_tensor(
                file,
                &format!("{p}.conv.bias"),
                &[desc.out_dim],
            )?),
            norm_gamma: Some(load_omniasr_tensor(
                file,
                &format!("{p}.layer_norm.weight"),
                &[desc.out_dim],
            )?),
            norm_beta: Some(load_omniasr_tensor(
                file,
                &format!("{p}.layer_norm.bias"),
                &[desc.out_dim],
            )?),
        });
    }
    let feature_projection = OmniasrCtcFeatureProjectionWeights {
        norm_gamma: Some(load_omniasr_tensor(
            file,
            "encoder_frontend.post_extract_layer_norm.weight",
            &[512],
        )?),
        norm_beta: Some(load_omniasr_tensor(
            file,
            "encoder_frontend.post_extract_layer_norm.bias",
            &[512],
        )?),
        linear_w: load_omniasr_tensor(
            file,
            "encoder_frontend.model_dim_proj.weight",
            &[1280, 512],
        )?,
        linear_b: load_omniasr_tensor(file, "encoder_frontend.model_dim_proj.bias", &[1280])?,
    };
    let pos_encoder_layers = vec![OmniasrCtcPosEncoderLayerWeights {
        conv_w: materialize_positional_weight(file)?,
        conv_b: load_omniasr_tensor(file, "encoder_frontend.pos_encoder.conv.bias", &[1280])?,
    }];

    let mut encoder_blocks = Vec::with_capacity(48);
    for i in 0..48 {
        let p = format!("encoder.layers.{i}");
        let mut q_proj = Vec::with_capacity(1280 * 1280);
        let mut k_proj = Vec::with_capacity(1280 * 1280);
        let mut v_proj = Vec::with_capacity(1280 * 1280);
        let mut q_bias = Vec::with_capacity(1280);
        let mut k_bias = Vec::with_capacity(1280);
        let mut v_bias = Vec::with_capacity(1280);
        // The fairseq2 module exposes separate q_proj, k_proj, v_proj rows;
        // the native runtime's fused layout is explicitly [Q rows, K rows,
        // V rows], preserving the official source module order.
        q_proj.extend(load_omniasr_tensor(
            file,
            &format!("{p}.self_attn.q_proj.weight"),
            &[1280, 1280],
        )?);
        k_proj.extend(load_omniasr_tensor(
            file,
            &format!("{p}.self_attn.k_proj.weight"),
            &[1280, 1280],
        )?);
        v_proj.extend(load_omniasr_tensor(
            file,
            &format!("{p}.self_attn.v_proj.weight"),
            &[1280, 1280],
        )?);
        q_bias.extend(load_omniasr_tensor(
            file,
            &format!("{p}.self_attn.q_proj.bias"),
            &[1280],
        )?);
        k_bias.extend(load_omniasr_tensor(
            file,
            &format!("{p}.self_attn.k_proj.bias"),
            &[1280],
        )?);
        v_bias.extend(load_omniasr_tensor(
            file,
            &format!("{p}.self_attn.v_proj.bias"),
            &[1280],
        )?);
        let qkv_proj = fuse_qkv(q_proj, k_proj, v_proj, 1280)?;
        let qkv_bias = fuse_qkv_bias(q_bias, k_bias, v_bias, 1280)?;
        encoder_blocks.push(OmniasrCtcEncoderBlockWeights {
            attn_norm_gamma: load_omniasr_tensor(
                file,
                &format!("{p}.self_attn_layer_norm.weight"),
                &[1280],
            )?,
            attn_norm_beta: load_omniasr_tensor(
                file,
                &format!("{p}.self_attn_layer_norm.bias"),
                &[1280],
            )?,
            qkv_proj,
            qkv_bias,
            attn_out: load_omniasr_tensor(
                file,
                &format!("{p}.self_attn.output_proj.weight"),
                &[1280, 1280],
            )?,
            attn_out_bias: load_omniasr_tensor(
                file,
                &format!("{p}.self_attn.output_proj.bias"),
                &[1280],
            )?,
            ffn_norm_gamma: load_omniasr_tensor(
                file,
                &format!("{p}.ffn_layer_norm.weight"),
                &[1280],
            )?,
            ffn_norm_beta: load_omniasr_tensor(file, &format!("{p}.ffn_layer_norm.bias"), &[1280])?,
            ffn_fc1: load_omniasr_tensor(
                file,
                &format!("{p}.ffn.inner_proj.weight"),
                &[5120, 1280],
            )?,
            ffn_fc1_bias: load_omniasr_tensor(file, &format!("{p}.ffn.inner_proj.bias"), &[5120])?,
            ffn_fc2: load_omniasr_tensor(
                file,
                &format!("{p}.ffn.output_proj.weight"),
                &[1280, 5120],
            )?,
            ffn_fc2_bias: load_omniasr_tensor(file, &format!("{p}.ffn.output_proj.bias"), &[1280])?,
        });
    }
    Ok(OmniasrCtcWeights {
        feature_extractor,
        feature_projection,
        frontend_model_dim_norm_gamma: None,
        frontend_model_dim_norm_beta: None,
        pos_encoder_layers,
        encoder_blocks,
        encoder_final_norm_gamma: load_omniasr_tensor(file, "encoder.layer_norm.weight", &[1280])?,
        encoder_final_norm_beta: load_omniasr_tensor(file, "encoder.layer_norm.bias", &[1280])?,
        head: OmniasrCtcHeadWeights {
            vocab_head: load_omniasr_tensor(file, "final_proj.weight", &[9812, 1280])?,
            vocab_bias: load_omniasr_tensor(file, "final_proj.bias", &[9812])?,
        },
        is_synthesized: false,
    })
}

fn omni_encoder_config(cfg: &OmniasrCtcConfig) -> CharsiuConfig {
    CharsiuConfig {
        hidden_size: cfg.encoder.model_dim,
        n_layer: cfg.encoder.num_encoder_layers,
        n_head: cfg.encoder.num_encoder_attn_heads,
        ffn_dim: cfg.encoder.ffn_inner_dim,
        vocab_size: cfg.head.target_vocab_size,
        silence_id: 0,
        pad_id: 0,
        sample_rate: cfg.sample_rate,
        frame_shift_sec: cfg.encoder.feature_extractor_total_stride() as f32
            / cfg.sample_rate as f32,
        layer_norm_eps: 1e-5,
        pos_conv_kernel: cfg.encoder.pos_conv_kernel_size,
        pos_conv_groups: cfg.encoder.num_pos_conv_groups,
        silence_threshold: 1,
        feature_projection_has_layer_norm: true,
        stem_conv_bias: cfg.encoder.feature_extractor_bias,
    }
}

fn omni_stem_attrs(cfg: &OmniasrCtcConfig) -> WaveformFrontendAttrs {
    WaveformFrontendAttrs {
        in_channels: 1,
        layers: cfg
            .encoder
            .feature_extractor_layer_descs
            .iter()
            .map(|d| ConvLayerAttrs {
                out_channels: d.out_dim,
                kernel: d.kernel,
                stride: d.stride,
            })
            .collect(),
        norm: if cfg.encoder.feature_extractor_layer_norm_convs {
            Norm::LayerAll
        } else {
            Norm::GroupFirstOnly
        },
        conv_bias: cfg.encoder.feature_extractor_bias,
    }
}

fn omni_stem_weights(weights: &OmniasrCtcWeights) -> WaveformFrontendWeights {
    WaveformFrontendWeights {
        layers: weights
            .feature_extractor
            .iter()
            .map(|layer| ConvLayerWeights {
                conv_w: layer.conv_w.clone(),
                conv_b: layer.conv_b.clone().unwrap_or_default(),
                norm_gamma: layer.norm_gamma.clone(),
                norm_beta: layer.norm_beta.clone(),
            })
            .collect(),
    }
}

fn omni_mha(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    frames: usize,
    heads: usize,
    head_dim: usize,
    compute: &Compute,
) -> Result<Vec<f32>> {
    let hidden = heads * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut output = vec![0.0; frames * hidden];
    let mut q_head = vec![0.0; frames * head_dim];
    let mut k_head_t = vec![0.0; head_dim * frames];
    let mut v_head = vec![0.0; frames * head_dim];
    let mut scores = vec![0.0; frames * frames];
    let mut probabilities = vec![0.0; frames * frames];
    let mut head_output = vec![0.0; frames * head_dim];
    for head in 0..heads {
        for frame in 0..frames {
            let source = frame * hidden + head * head_dim;
            let destination = frame * head_dim;
            q_head[destination..destination + head_dim]
                .copy_from_slice(&q[source..source + head_dim]);
            v_head[destination..destination + head_dim]
                .copy_from_slice(&v[source..source + head_dim]);
            for dim in 0..head_dim {
                k_head_t[dim * frames + frame] = k[source + dim];
            }
        }
        compute.gemm_f32(
            frames,
            frames,
            head_dim,
            &q_head,
            &k_head_t,
            None,
            &mut scores,
        )?;
        for score in &mut scores {
            *score *= scale;
        }
        compute.softmax_f32(&scores, &mut probabilities, frames, frames)?;
        compute.gemm_f32(
            frames,
            head_dim,
            frames,
            &probabilities,
            &v_head,
            None,
            &mut head_output,
        )?;
        for frame in 0..frames {
            let source = frame * head_dim;
            let destination = frame * hidden + head * head_dim;
            output[destination..destination + head_dim]
                .copy_from_slice(&head_output[source..source + head_dim]);
        }
    }
    Ok(output)
}

/// Splits the output-major fused projection result `[frames, 3 * hidden]`
/// into three frame-major `[frames, hidden]` buffers.  The projection result
/// is row-major, so taking three contiguous thirds would be incorrect when
/// `frames > 1`.
fn unpack_fused_qkv(
    qkv: &[f32],
    frames: usize,
    hidden: usize,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let row = 3 * hidden;
    if qkv.len() != frames * row {
        return Err(VokraError::InvalidArgument(format!(
            "omniasr-ctc: fused QKV output len {} != frames * 3 * hidden = {}",
            qkv.len(),
            frames * row
        )));
    }
    let mut q = vec![0.0; frames * hidden];
    let mut k = vec![0.0; frames * hidden];
    let mut v = vec![0.0; frames * hidden];
    for frame in 0..frames {
        let source = frame * row;
        let destination = frame * hidden;
        q[destination..destination + hidden].copy_from_slice(&qkv[source..source + hidden]);
        k[destination..destination + hidden]
            .copy_from_slice(&qkv[source + hidden..source + 2 * hidden]);
        v[destination..destination + hidden]
            .copy_from_slice(&qkv[source + 2 * hidden..source + 3 * hidden]);
    }
    Ok((q, k, v))
}

fn omni_stable_block(
    hidden: &mut [f32],
    frames: usize,
    cfg: &CharsiuConfig,
    block: &OmniasrCtcEncoderBlockWeights,
    compute: &Compute,
) -> Result<()> {
    let h = cfg.hidden_size;
    let mut normalized = hidden.to_vec();
    layer_norm_with_compute_inplace(
        &mut normalized,
        frames,
        h,
        &block.attn_norm_gamma,
        &block.attn_norm_beta,
        cfg.layer_norm_eps,
        compute,
    )?;
    let qkv = linear_forward_with_compute(
        &normalized,
        frames,
        h,
        &block.qkv_proj,
        &block.qkv_bias,
        3 * h,
        compute,
    )?;
    let (q, k, v) = unpack_fused_qkv(&qkv, frames, h)?;
    let attention = omni_mha(&q, &k, &v, frames, cfg.n_head, h / cfg.n_head, compute)?;
    let projected = linear_forward_with_compute(
        &attention,
        frames,
        h,
        &block.attn_out,
        &block.attn_out_bias,
        h,
        compute,
    )?;
    for (value, residual) in hidden.iter_mut().zip(projected) {
        *value += residual;
    }

    normalized.copy_from_slice(hidden);
    layer_norm_with_compute_inplace(
        &mut normalized,
        frames,
        h,
        &block.ffn_norm_gamma,
        &block.ffn_norm_beta,
        cfg.layer_norm_eps,
        compute,
    )?;
    let intermediate = linear_forward_with_compute(
        &normalized,
        frames,
        h,
        &block.ffn_fc1,
        &block.ffn_fc1_bias,
        cfg.ffn_dim,
        compute,
    )?;
    let mut activated = vec![0.0; intermediate.len()];
    compute.gelu_f32(&intermediate, &mut activated)?;
    let output = linear_forward_with_compute(
        &activated,
        frames,
        cfg.ffn_dim,
        &block.ffn_fc2,
        &block.ffn_fc2_bias,
        h,
        compute,
    )?;
    for (value, residual) in hidden.iter_mut().zip(output) {
        *value += residual;
    }
    Ok(())
}

fn omni_reject_non_finite(label: &str, values: &[f32]) -> Result<()> {
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        return Err(VokraError::ModelLoad(format!(
            "omniasr-ctc: non-finite value in {label} at flat index {index}"
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

/// omniASR-CTC ASR engine handle.
///
/// Carries the resolved config, explicit weight store, and selected backend.
/// [`Self::transcribe_tokens`] is the waveform → CTC token entry point;
/// synthesized fixtures and unbound GGUF payloads fail closed rather than
/// producing a fabricated transcript.
///
/// # Weight license surfacing
///
/// The `weight_license` field defaults to [`LicenseClass::Permissive`]
/// under [`Self::new`] (the Apache-2.0 class that is the only
/// legitimate class for real omniASR-CTC weights per the compliance
/// registry — `vokra_core::compliance::license_class` maps
/// `omniasr-ctc` / `omniasr-ctc-1b` / `omniasr-ctc-300m` /
/// `omniasr-ctc-3b` / `omniasr-ctc-7b` to
/// [`LicenseClass::Permissive`] via the `omniasr-ctc-` family prefix
/// walk). Differs from Parakeet-CTC (CC-BY 4.0 =
/// [`LicenseClass::AttributionRequired`]) — omniASR carries no
/// runtime-side attribution obligation. The M2-13 outer compliance
/// gate does the strict enforcement; this handle simply surfaces the
/// class so callers can cross-check.
#[derive(Debug, Clone)]
pub struct OmniasrCtcAsr {
    cfg: OmniasrCtcConfig,
    weights: OmniasrCtcWeights,
    weight_license: LicenseClass,
    backend: BackendKind,
}

impl OmniasrCtcAsr {
    /// Assembles an engine from `cfg` and `weights`. Cross-checks the
    /// weight-store shapes against `cfg` (feature extractor 7-layer
    /// count + per-layer sizes, feature projection dims, positional
    /// encoder depth, encoder block count + per-tensor sizes, final
    /// norm width, head vocab width) so a mismatched pair fails loudly
    /// here rather than deep inside a forward.
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] from `cfg.validate_for_forward`.
    /// - [`VokraError::InvalidArgument`] naming the first shape
    ///   mismatch.
    pub fn new(cfg: OmniasrCtcConfig, weights: OmniasrCtcWeights) -> Result<Self> {
        cfg.validate_for_forward()?;
        let enc = &cfg.encoder;
        let head = &cfg.head;
        let d_enc = enc.model_dim;
        let ffn = enc.ffn_inner_dim;
        let vocab = head.target_vocab_size;
        let feat_dim = enc.feature_dim;
        let bias_on = enc.feature_extractor_bias;
        let ln_on = enc.feature_extractor_layer_norm_convs;

        // -- Feature extractor -------------------------------------------------
        if weights.feature_extractor.len() != OMNIASR_CTC_NUM_FEATURE_EXTRACTOR_LAYERS {
            return Err(VokraError::InvalidArgument(format!(
                "omniasr-ctc weights: feature_extractor.len()={} != {} \
                 (fairseq2 wav2vec 2.0 fixed 7-layer stem)",
                weights.feature_extractor.len(),
                OMNIASR_CTC_NUM_FEATURE_EXTRACTOR_LAYERS,
            )));
        }
        let mut in_dim: usize = 1;
        for (i, lw) in weights.feature_extractor.iter().enumerate() {
            let d = &enc.feature_extractor_layer_descs[i];
            let expected_w_len = d.out_dim * in_dim * d.kernel;
            if lw.conv_w.len() != expected_w_len {
                return Err(VokraError::InvalidArgument(format!(
                    "omniasr-ctc weights: feature_extractor[{i}].conv_w.len()={} != \
                     out_dim*in_dim*kernel = {}*{}*{} = {}",
                    lw.conv_w.len(),
                    d.out_dim,
                    in_dim,
                    d.kernel,
                    expected_w_len,
                )));
            }
            match (bias_on, &lw.conv_b) {
                (true, Some(v)) if v.len() == d.out_dim => {}
                (true, Some(v)) => {
                    return Err(VokraError::InvalidArgument(format!(
                        "omniasr-ctc weights: feature_extractor[{i}].conv_b.len()={} \
                         != out_dim={}",
                        v.len(),
                        d.out_dim,
                    )));
                }
                (true, None) => {
                    return Err(VokraError::InvalidArgument(format!(
                        "omniasr-ctc weights: feature_extractor[{i}].conv_b is None but \
                         feature_extractor_bias=true — a bias-free variant must set \
                         feature_extractor_bias=false",
                    )));
                }
                (false, Some(_)) => {
                    return Err(VokraError::InvalidArgument(format!(
                        "omniasr-ctc weights: feature_extractor[{i}].conv_b is Some but \
                         feature_extractor_bias=false — a bias-carrying variant must set \
                         feature_extractor_bias=true",
                    )));
                }
                (false, None) => {}
            }
            for (name, opt, expected) in [
                ("norm_gamma", lw.norm_gamma.as_ref(), d.out_dim),
                ("norm_beta", lw.norm_beta.as_ref(), d.out_dim),
            ] {
                match (ln_on, opt) {
                    (true, Some(v)) if v.len() == expected => {}
                    (true, Some(v)) => {
                        return Err(VokraError::InvalidArgument(format!(
                            "omniasr-ctc weights: feature_extractor[{i}].{name}.len()={} \
                             != out_dim={}",
                            v.len(),
                            expected,
                        )));
                    }
                    (true, None) => {
                        return Err(VokraError::InvalidArgument(format!(
                            "omniasr-ctc weights: feature_extractor[{i}].{name} is None but \
                             feature_extractor_layer_norm_convs=true — a norm-free variant \
                             must set feature_extractor_layer_norm_convs=false",
                        )));
                    }
                    (false, Some(_)) => {
                        return Err(VokraError::InvalidArgument(format!(
                            "omniasr-ctc weights: feature_extractor[{i}].{name} is Some but \
                             feature_extractor_layer_norm_convs=false — a norm-carrying \
                             variant must set feature_extractor_layer_norm_convs=true",
                        )));
                    }
                    (false, None) => {}
                }
            }
            in_dim = d.out_dim;
        }

        // -- Feature projection ------------------------------------------------
        // fairseq2's Wav2Vec2Frontend always applies post-extraction
        // LayerNorm before the model-dimension projection.  This is distinct
        // from the `layer_norm_features` extractor preset, which is false for
        // large_lv60k/omniASR but does not remove this pair of tensors.
        for (name, opt) in [
            ("norm_gamma", weights.feature_projection.norm_gamma.as_ref()),
            ("norm_beta", weights.feature_projection.norm_beta.as_ref()),
        ] {
            match opt {
                Some(v) if v.len() == feat_dim => {}
                Some(v) => {
                    return Err(VokraError::InvalidArgument(format!(
                        "omniasr-ctc weights: feature_projection.{name}.len()={} != feature_dim={}",
                        v.len(),
                        feat_dim,
                    )));
                }
                None => {
                    return Err(VokraError::InvalidArgument(format!(
                        "omniasr-ctc weights: feature_projection.{name} is missing; fairseq2 frontend post-extraction LayerNorm is required",
                    )));
                }
            }
        }
        if weights.feature_projection.linear_w.len() != feat_dim * d_enc {
            return Err(VokraError::InvalidArgument(format!(
                "omniasr-ctc weights: feature_projection.linear_w.len()={} != \
                 feature_dim * model_dim = {} * {} = {}",
                weights.feature_projection.linear_w.len(),
                feat_dim,
                d_enc,
                feat_dim * d_enc,
            )));
        }
        if weights.feature_projection.linear_b.len() != d_enc {
            return Err(VokraError::InvalidArgument(format!(
                "omniasr-ctc weights: feature_projection.linear_b.len()={} != model_dim={}",
                weights.feature_projection.linear_b.len(),
                d_enc,
            )));
        }

        // fairseq2 applies this separate model-dimension LayerNorm after
        // model_dim_proj + positional encoding when layer_norm_features is
        // enabled. The 1B large_lv60k path is false and therefore carries no
        // such tensors.
        for (name, opt) in [
            (
                "frontend_model_dim_norm_gamma",
                weights.frontend_model_dim_norm_gamma.as_ref(),
            ),
            (
                "frontend_model_dim_norm_beta",
                weights.frontend_model_dim_norm_beta.as_ref(),
            ),
        ] {
            match (enc.layer_norm_features, opt) {
                (true, Some(values)) if values.len() == d_enc => {}
                (true, Some(values)) => {
                    return Err(VokraError::InvalidArgument(format!(
                        "omniasr-ctc weights: {name}.len()={} != model_dim={}",
                        values.len(),
                        d_enc,
                    )));
                }
                (true, None) => {
                    return Err(VokraError::InvalidArgument(format!(
                        "omniasr-ctc weights: {name} is missing but layer_norm_features=true"
                    )));
                }
                (false, Some(_)) => {
                    return Err(VokraError::InvalidArgument(format!(
                        "omniasr-ctc weights: {name} is present but layer_norm_features=false"
                    )));
                }
                (false, None) => {}
            }
        }

        // -- Positional encoder ------------------------------------------------
        if weights.pos_encoder_layers.len() != enc.pos_encoder_depth {
            return Err(VokraError::InvalidArgument(format!(
                "omniasr-ctc weights: pos_encoder_layers.len()={} != pos_encoder_depth={}",
                weights.pos_encoder_layers.len(),
                enc.pos_encoder_depth,
            )));
        }
        let per_group = d_enc / enc.num_pos_conv_groups;
        let expected_pos_w = d_enc * per_group * enc.pos_conv_kernel_size;
        for (i, pl) in weights.pos_encoder_layers.iter().enumerate() {
            if pl.conv_w.len() != expected_pos_w {
                return Err(VokraError::InvalidArgument(format!(
                    "omniasr-ctc weights: pos_encoder_layers[{i}].conv_w.len()={} != \
                     model_dim * (model_dim/num_pos_conv_groups) * pos_conv_kernel_size \
                     = {} * {} * {} = {}",
                    pl.conv_w.len(),
                    d_enc,
                    per_group,
                    enc.pos_conv_kernel_size,
                    expected_pos_w,
                )));
            }
            if pl.conv_b.len() != d_enc {
                return Err(VokraError::InvalidArgument(format!(
                    "omniasr-ctc weights: pos_encoder_layers[{i}].conv_b.len()={} != \
                     model_dim={}",
                    pl.conv_b.len(),
                    d_enc,
                )));
            }
        }

        // -- Transformer blocks ------------------------------------------------
        if weights.encoder_blocks.len() != enc.num_encoder_layers {
            return Err(VokraError::InvalidArgument(format!(
                "omniasr-ctc weights: encoder_blocks.len()={} != num_encoder_layers={}",
                weights.encoder_blocks.len(),
                enc.num_encoder_layers,
            )));
        }
        for (i, blk) in weights.encoder_blocks.iter().enumerate() {
            for (name, len, expected) in [
                ("attn_norm_gamma", blk.attn_norm_gamma.len(), d_enc),
                ("attn_norm_beta", blk.attn_norm_beta.len(), d_enc),
                ("qkv_proj", blk.qkv_proj.len(), d_enc * 3 * d_enc),
                ("qkv_bias", blk.qkv_bias.len(), 3 * d_enc),
                ("attn_out", blk.attn_out.len(), d_enc * d_enc),
                ("attn_out_bias", blk.attn_out_bias.len(), d_enc),
                ("ffn_norm_gamma", blk.ffn_norm_gamma.len(), d_enc),
                ("ffn_norm_beta", blk.ffn_norm_beta.len(), d_enc),
                ("ffn_fc1", blk.ffn_fc1.len(), d_enc * ffn),
                ("ffn_fc1_bias", blk.ffn_fc1_bias.len(), ffn),
                ("ffn_fc2", blk.ffn_fc2.len(), ffn * d_enc),
                ("ffn_fc2_bias", blk.ffn_fc2_bias.len(), d_enc),
            ] {
                if len != expected {
                    return Err(VokraError::InvalidArgument(format!(
                        "omniasr-ctc weights: encoder block {i} `{name}` \
                         len={len} != {expected}",
                    )));
                }
            }
        }
        if weights.encoder_final_norm_gamma.len() != d_enc {
            return Err(VokraError::InvalidArgument(format!(
                "omniasr-ctc weights: encoder_final_norm_gamma.len()={} != model_dim={}",
                weights.encoder_final_norm_gamma.len(),
                d_enc,
            )));
        }
        if weights.encoder_final_norm_beta.len() != d_enc {
            return Err(VokraError::InvalidArgument(format!(
                "omniasr-ctc weights: encoder_final_norm_beta.len()={} != model_dim={}",
                weights.encoder_final_norm_beta.len(),
                d_enc,
            )));
        }

        // -- CTC head ----------------------------------------------------------
        if weights.head.vocab_head.len() != d_enc * vocab {
            return Err(VokraError::InvalidArgument(format!(
                "omniasr-ctc weights: head.vocab_head.len()={} != model_dim * vocab = \
                 {} * {} = {}",
                weights.head.vocab_head.len(),
                d_enc,
                vocab,
                d_enc * vocab,
            )));
        }
        if weights.head.vocab_bias.len() != vocab {
            return Err(VokraError::InvalidArgument(format!(
                "omniasr-ctc weights: head.vocab_bias.len()={} != target_vocab_size={}",
                weights.head.vocab_bias.len(),
                vocab,
            )));
        }

        Ok(Self {
            cfg,
            weights,
            // Default weight-license class under `new()` mirrors the
            // compliance registry (`vokra_core::compliance::license_class`
            // maps every `omniasr-ctc-*` slug to Apache-2.0 =
            // Permissive via the `omniasr-ctc-` family prefix walk).
            // Distinct from Parakeet-CTC's default (AttributionRequired
            // = CC-BY 4.0) — omniASR has no runtime-side attribution
            // obligation. The manifest-backed `from_gguf` path re-checks the
            // same class and all pinned provenance before constructing an
            // engine.
            weight_license: LicenseClass::Permissive,
            backend: BackendKind::Cpu,
        })
    }

    /// The resolved configuration.
    #[must_use]
    pub fn config(&self) -> &OmniasrCtcConfig {
        &self.cfg
    }

    /// True iff the weight store was built by
    /// [`OmniasrCtcWeights::synthesized`] (never a real upstream
    /// checkpoint).
    #[must_use]
    pub fn is_synthesized(&self) -> bool {
        self.weights.is_synthesized
    }

    /// The currently selected weight-license class. `new()` defaults to the
    /// compliance registry's Apache-2.0/Permissive class. The strict
    /// `from_gguf` path replaces this with the authenticated provenance class.
    #[inline]
    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    /// Selects the execution backend for a bound engine. The default from
    /// [`Self::new`] is CPU. Backend capability and device availability are
    /// checked when [`Self::encode_features`] / [`Self::transcribe_tokens`]
    /// dispatch through [`Compute::for_backend`], not at selection time.
    pub fn with_backend(mut self, backend: BackendKind) -> Self {
        self.backend = backend;
        self
    }

    /// Runs the waveform frontend, feature projection, positional
    /// convolution, pre-norm Transformer encoder, and final encoder norm.
    pub fn encode_features(&self, pcm: &[f32]) -> Result<(Vec<f32>, usize)> {
        let (_, encoder, frames) = self.forward_trace(pcm)?;
        Ok((encoder, frames))
    }

    /// Returns the post-frontend representation and final encoder output from
    /// one native forward.  The VAST reference packet uses this diagnostic
    /// boundary for `frontend.f32le`; it is not a second implementation.
    pub fn diagnostic_trace(&self, pcm: &[f32]) -> Result<(Vec<f32>, Vec<f32>, usize)> {
        self.forward_trace(pcm)
    }

    fn forward_trace(&self, pcm: &[f32]) -> Result<(Vec<f32>, Vec<f32>, usize)> {
        if pcm.is_empty() {
            return Err(VokraError::InvalidArgument(
                "omniasr-ctc encode_features: pcm slice is empty".to_owned(),
            ));
        }
        if self.weights.is_synthesized {
            return Err(VokraError::NotImplemented(
                "omniasr-ctc encode_features: synthesized weights are not a real checkpoint",
            ));
        }
        let compute = Compute::for_backend(self.backend, OMNIASR_CTC_HOT_OPS)?;
        let attrs = omni_stem_attrs(&self.cfg);
        let stem = omni_stem_weights(&self.weights);
        let normalized = normalize_omniasr_pcm(pcm);
        let features = crate::wav2vec2_ctc::waveform_frontend_with_compute(
            &normalized,
            &attrs,
            &stem,
            &compute,
        )?;
        let feature_dim = self.cfg.encoder.feature_dim;
        if features.len() % feature_dim != 0 {
            return Err(VokraError::ModelLoad(format!(
                "omniasr-ctc: waveform frontend output {} is not divisible by feature_dim {}",
                features.len(),
                feature_dim
            )));
        }
        let frames = features.len() / feature_dim;
        if frames == 0 || frames > self.cfg.encoder.max_seq_len {
            return Err(VokraError::InvalidArgument(format!(
                "omniasr-ctc: frontend produced {frames} frame(s), outside 1..={} max_seq_len",
                self.cfg.encoder.max_seq_len
            )));
        }
        let encoder_cfg = omni_encoder_config(&self.cfg);
        let projection = CharsiuFeatureProjection {
            norm_gamma: self.weights.feature_projection.norm_gamma.clone(),
            norm_beta: self.weights.feature_projection.norm_beta.clone(),
            linear_w: self.weights.feature_projection.linear_w.clone(),
            linear_b: self.weights.feature_projection.linear_b.clone(),
        };
        let mut hidden = feature_projection_forward_with_compute(
            &features,
            frames,
            feature_dim,
            &projection,
            self.cfg.encoder.model_dim,
            true,
            encoder_cfg.layer_norm_eps,
            &compute,
        )?;
        for pos in &self.weights.pos_encoder_layers {
            let positional = positional_conv_forward_with_compute(
                &hidden,
                frames,
                &encoder_cfg,
                &CharsiuPosConv {
                    weight: pos.conv_w.clone(),
                    bias: pos.conv_b.clone(),
                },
                &compute,
            )?;
            for (value, offset) in hidden.iter_mut().zip(positional) {
                *value += offset;
            }
        }
        if let (Some(gamma), Some(beta)) = (
            self.weights.frontend_model_dim_norm_gamma.as_ref(),
            self.weights.frontend_model_dim_norm_beta.as_ref(),
        ) {
            layer_norm_with_compute_inplace(
                &mut hidden,
                frames,
                self.cfg.encoder.model_dim,
                gamma,
                beta,
                encoder_cfg.layer_norm_eps,
                &compute,
            )?;
        }
        let frontend = hidden.clone();
        for block in &self.weights.encoder_blocks {
            omni_stable_block(&mut hidden, frames, &encoder_cfg, block, &compute)?;
        }
        layer_norm_with_compute_inplace(
            &mut hidden,
            frames,
            self.cfg.encoder.model_dim,
            &self.weights.encoder_final_norm_gamma,
            &self.weights.encoder_final_norm_beta,
            encoder_cfg.layer_norm_eps,
            &compute,
        )?;
        omni_reject_non_finite("encoder output", &hidden)?;
        Ok((frontend, hidden, frames))
    }

    /// Runs the CTC projection and returns frame-major logits plus frame count.
    ///
    /// This is the diagnostic boundary used by the independent VAST reference
    /// packet.  It shares the exact encoder path with [`Self::transcribe_tokens`]
    /// so parity cannot accidentally compare a second implementation.
    pub fn ctc_logits(&self, pcm: &[f32]) -> Result<(Vec<f32>, usize)> {
        let (hidden, frames) = self.encode_features(pcm)?;
        let logits = linear_forward_with_compute(
            &hidden,
            frames,
            self.cfg.encoder.model_dim,
            &self.weights.head.vocab_head,
            &self.weights.head.vocab_bias,
            self.cfg.head.target_vocab_size,
            &Compute::for_backend(self.backend, OMNIASR_CTC_HOT_OPS)?,
        )?;
        omni_reject_non_finite("CTC logits", &logits)?;
        Ok((logits, frames))
    }

    /// Runs the CTC projection and greedy blank/repeat folding.
    pub fn transcribe_tokens(&self, pcm: &[f32]) -> Result<Vec<u32>> {
        let (logits, frames) = self.ctc_logits(pcm)?;
        ctc_decode_greedy(
            &logits,
            frames,
            self.cfg.head.target_vocab_size,
            self.cfg.head.blank_id as usize,
        )
    }

    /// Binds an omniASR-CTC GGUF: validates arch, reads the strict
    /// `vokra.omniasr_ctc.*` topology chunk group, and then requires the
    /// complete audited fairseq2 tensor-name manifest before decoding any
    /// tensor. Synthesized fixtures are never substituted for a checkpoint.
    ///
    /// This binder is a *loud* validation step. Every failure is a
    /// distinct [`VokraError::ModelLoad`] naming the missing / wrong
    /// key so a reader diagnosing a mis-produced GGUF has exactly one
    /// place to walk (FR-EX-08 — never a silent partial bind).
    ///
    /// # Loud-partial contract
    ///
    /// The exact 807-entry manifest is checked before decoding any payload.
    /// The binder also resolves the fairseq2 positional-convolution
    /// weight-normalization representation and fuses separate source q/k/v
    /// rows in the native [Q,K,V] layout before constructing weights.
    /// Two primitives already exist and must not be rewritten: the
    /// 7-layer strided conv stem ([`vokra_ops::waveform_frontend`],
    /// whose `wav2vec2_base` preset targets exactly these checkpoints)
    /// and the CTC decoding primitive ([`vokra_ops::ctc_decode`]) with
    /// `blank_id = 0`.
    /// SentencePiece detokenize is model-specific (not a shared op).
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when `vokra.model.arch` is absent or
    ///   not `"omniasr-ctc"` (a `parakeet-ctc` GGUF handed to us by
    ///   mistake fails with a hint pointing at the FastConformer
    ///   sibling `parakeet_ctc::ParakeetCtcAsr::from_gguf`, matching
    ///   the sibling-arch disambiguation pattern used by
    ///   `Mt3::from_gguf` / `Snac::from_gguf` /
    ///   `ParakeetCtcAsr::from_gguf`).
    /// - [`VokraError::ModelLoad`] when any `vokra.omniasr_ctc.*` chunk
    ///   is absent ([`OmniasrCtcConfig::from_gguf`] is strict).
    /// - [`VokraError::InvalidArgument`] from
    ///   [`OmniasrCtcConfig::validate_for_forward`] +
    ///   [`OmniasrCtcAsr::new`] shape gates.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        // 1. Arch check — always first so a mis-typed model handed here
        //    fails with a specific message instead of a downstream
        //    "vokra.omniasr_ctc.arch.encoder.model_dim missing".
        match file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()) {
            Some(a) if a == EXPECTED_ARCH => {}
            Some("parakeet-ctc") | Some("parakeet-ctc-1.1b") | Some("parakeet-ctc-1_1b") => {
                return Err(VokraError::ModelLoad(format!(
                    "omniasr-ctc: GGUF arch is a FastConformer or Parakeet-CTC \
                     variant (log-mel input + 8× FastConformer subsampling + \
                     CTC head with blank at vocab tail), expected \
                     `{EXPECTED_ARCH}` (wav2vec 2.0 waveform input + 7-layer \
                     Conv1D feature extractor + plain Transformer encoder + \
                     CTC head with blank at index 0). These are different \
                     topologies — Parakeet-CTC has num_mel_bins=80 on the \
                     input and a FastConformer body that the wav2vec 2.0 \
                     binder cannot dispatch. Route the GGUF through the \
                     sibling `parakeet_ctc::ParakeetCtcAsr::from_gguf` binder \
                     (`crates/vokra-models/src/parakeet_ctc/mod.rs`) instead."
                )));
            }
            Some(other) => {
                return Err(VokraError::ModelLoad(format!(
                    "omniasr-ctc: GGUF arch is `{other}`, expected `{EXPECTED_ARCH}` \
                     (was this GGUF produced by `vokra-cli convert --model omniasr-ctc`? \
                     Sibling ASR arches — `whisper`, `voxtral`, `canary`, \
                     `canary-qwen`, `parakeet-ctc`, `parakeet-tdt` — are \
                     completely different topologies)"
                )));
            }
            None => {
                return Err(VokraError::ModelLoad(
                    "omniasr-ctc: GGUF is missing `vokra.model.arch` (converter did \
                     not stamp it — this is not a Vokra-native omniasr-ctc GGUF)"
                        .to_owned(),
                ));
            }
        }

        // 2. Strict topology axes from the `vokra.omniasr_ctc.*` chunk
        //    group.
        let cfg = OmniasrCtcConfig::from_gguf(file)?;
        let expected_cfg = OmniasrCtcConfig::omniasr_ctc_1b();
        if cfg != expected_cfg {
            return Err(VokraError::ModelLoad(
                "omniasr-ctc: GGUF topology is not the audited facebook/omniASR-CTC-1B registry configuration".to_owned(),
            ));
        }
        require_omniasr_provenance(file)?;
        validate_omniasr_manifest(file)?;
        let weights = bind_omniasr_weights(file, &cfg)?;
        let mut asr = Self::new(cfg, weights)?;
        asr.weight_license = LicenseClass::Permissive;
        Ok(asr)
    }

    /// Transcribes a mono `f32` PCM slice using the native wav2vec2 CTC path.
    /// The returned values are CTC token ids; SentencePiece string assembly
    /// remains a caller concern because the tokenizer is not part of GGUF.
    pub fn transcribe(&self, pcm: &[f32]) -> Result<Vec<u32>> {
        self.transcribe_tokens(pcm)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Every hparam matches the fairseq2 registry walk documented in the
    /// module docstring
    /// (`omnilingual_asr/models/wav2vec2_asr/config.py::_1b_asr` →
    /// `omnilingual_asr/models/wav2vec2_ssl/config.py::_1b_ssl` →
    /// `fairseq2/models/wav2vec2/config.py::large_lv60k` /
    /// `fairseq2/models/wav2vec2/asr/config.py::base_10h`).
    #[test]
    fn omniasr_ctc_1b_matches_primary_source_registry_walk() {
        let c = OmniasrCtcConfig::omniasr_ctc_1b();
        // Encoder.
        assert_eq!(c.encoder.model_dim, 1280);
        assert_eq!(c.encoder.num_encoder_layers, 48);
        // NOTE: 16 (inherits from fairseq2 "large"; "1b" does not override).
        assert_eq!(c.encoder.num_encoder_attn_heads, 16);
        assert_eq!(c.encoder.ffn_inner_dim, 5120);
        assert_eq!(c.encoder.feature_dim, 512);
        // Feature extractor stem — 7 layers, total stride = 320.
        assert_eq!(
            c.encoder.feature_extractor_layer_descs.len(),
            OMNIASR_CTC_NUM_FEATURE_EXTRACTOR_LAYERS
        );
        for d in &c.encoder.feature_extractor_layer_descs {
            assert_eq!(d.out_dim, 512, "every stem layer is 512-channel");
        }
        assert_eq!(c.encoder.feature_extractor_layer_descs[0].kernel, 10);
        assert_eq!(c.encoder.feature_extractor_layer_descs[0].stride, 5);
        assert_eq!(c.encoder.feature_extractor_layer_descs[6].kernel, 2);
        assert_eq!(c.encoder.feature_extractor_layer_descs[6].stride, 2);
        assert_eq!(c.encoder.feature_extractor_total_stride(), 320);
        // large_lv60k axes.
        assert!(c.encoder.feature_extractor_bias);
        assert!(c.encoder.feature_extractor_layer_norm_convs);
        assert!(!c.encoder.layer_norm_features);
        // Positional encoder.
        assert_eq!(c.encoder.pos_conv_kernel_size, 128);
        assert_eq!(c.encoder.num_pos_conv_groups, 16);
        assert_eq!(c.encoder.pos_encoder_depth, 1);
        assert!(!c.encoder.use_conformer);
        assert_eq!(c.encoder.max_seq_len, 4096);
        // CTC head.
        assert_eq!(c.head.target_vocab_size, 9812);
        // Blank at index 0 — fairseq2 wav2vec 2.0 convention.
        assert_eq!(c.head.blank_id, 0);
        assert_eq!(c.head.blank_id(), 0);
        assert_eq!(c.head.final_dropout_p, 0.0);
        // Audio boundary (model card + wav2vec 2.0 convention).
        assert_eq!(c.sample_rate, 16_000);
        // Derived.
        assert_eq!(c.encoder.head_dim(), 80); // 1280 / 16
        // Everything above adds up to a well-formed config.
        c.validate_for_forward()
            .expect("omniasr-ctc-1b is well-formed");
    }

    /// Guards the axes that distinguish omniASR-CTC from Parakeet-CTC:
    /// waveform input (feature_dim=512, no log-mel channel), blank at
    /// index 0 (not vocab tail), 7-layer waveform feature extractor.
    /// Getting these wrong at conversion time would silently mis-slot
    /// the feature extractor tensors or misplace the CTC blank — a
    /// class of regression the CI should catch on sight.
    #[test]
    fn omniasr_ctc_1b_differs_from_parakeet_ctc_on_key_axes() {
        let c = OmniasrCtcConfig::omniasr_ctc_1b();
        // Waveform input — feature extractor produces features, not
        // log-mel bins (Parakeet-CTC has num_mel_bins=80 on the input).
        assert_eq!(
            c.encoder.feature_dim, 512,
            "omniASR-CTC: waveform-features 512 (Parakeet-CTC uses 80 log-mel bins)"
        );
        // Blank at 0 — fairseq2 default (Parakeet-CTC has blank at
        // vocab tail = pad_token_id = 1024).
        assert_eq!(
            c.head.blank_id, 0,
            "omniASR-CTC: blank at 0 (Parakeet-CTC has blank at vocab tail)"
        );
        // 48 transformer layers (Parakeet-CTC has 42 FastConformer
        // layers — same family, different topology).
        assert_eq!(
            c.encoder.num_encoder_layers, 48,
            "omniASR-CTC-1B: 48 transformer layers (Parakeet-CTC-1.1B: 42 FastConformer)"
        );
        // Plain Transformer, not FastConformer — no depthwise conv,
        // no macaron FFN.
        assert!(
            !c.encoder.use_conformer,
            "omniASR-CTC: plain Transformer encoder (Parakeet-CTC: FastConformer)"
        );
    }

    #[test]
    fn tiny_config_is_well_formed() {
        OmniasrCtcConfig::tiny_for_tests()
            .validate_for_forward()
            .expect("tiny config is well-formed");
    }

    #[test]
    fn official_pcm_normalization_uses_layer_norm_default_epsilon() {
        let pcm = [0.0_f32, 1.0, 2.0, 3.0];
        let normalized = normalize_omniasr_pcm(&pcm);
        let mean = 1.5_f32;
        let variance = 1.25_f32;
        let scale = 1.0 / (variance + 1e-5_f32).sqrt();
        for (actual, value) in normalized.iter().zip(pcm) {
            assert_eq!(*actual, (value - mean) * scale);
        }
        assert_ne!(
            normalized[1],
            (pcm[1] - mean) / (variance + 1e-7_f32).sqrt(),
            "the shared wav2vec2 CTC epsilon must not leak into OmniASR"
        );
    }

    #[test]
    fn config_head_split_ill_formed_is_rejected() {
        let mut c = OmniasrCtcConfig::tiny_for_tests();
        c.encoder.num_encoder_attn_heads = 3; // 16 % 3 != 0
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_odd_head_dim_is_rejected() {
        let mut c = OmniasrCtcConfig::tiny_for_tests();
        // 12 / 4 = 3 (odd).
        c.encoder.model_dim = 12;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_zero_layer_is_rejected() {
        let mut c = OmniasrCtcConfig::tiny_for_tests();
        c.encoder.num_encoder_layers = 0;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_zero_ffn_is_rejected() {
        let mut c = OmniasrCtcConfig::tiny_for_tests();
        c.encoder.ffn_inner_dim = 0;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_zero_feature_dim_is_rejected() {
        let mut c = OmniasrCtcConfig::tiny_for_tests();
        c.encoder.feature_dim = 0;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_zero_max_seq_len_is_rejected() {
        let mut c = OmniasrCtcConfig::tiny_for_tests();
        c.encoder.max_seq_len = 0;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_zero_pos_kernel_is_rejected() {
        let mut c = OmniasrCtcConfig::tiny_for_tests();
        c.encoder.pos_conv_kernel_size = 0;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_pos_groups_not_dividing_is_rejected() {
        let mut c = OmniasrCtcConfig::tiny_for_tests();
        c.encoder.num_pos_conv_groups = 3; // 16 % 3 != 0
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_zero_pos_depth_is_rejected() {
        let mut c = OmniasrCtcConfig::tiny_for_tests();
        c.encoder.pos_encoder_depth = 0;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_feature_extractor_last_out_dim_mismatch_is_rejected() {
        let mut c = OmniasrCtcConfig::tiny_for_tests();
        // Change the last layer's out_dim so it no longer matches
        // feature_dim; the validator must catch this before any forward.
        c.encoder.feature_extractor_layer_descs[6].out_dim = 7;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_feature_extractor_zero_stride_is_rejected() {
        let mut c = OmniasrCtcConfig::tiny_for_tests();
        c.encoder.feature_extractor_layer_descs[0].stride = 0;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_zero_vocab_is_rejected() {
        let mut c = OmniasrCtcConfig::tiny_for_tests();
        c.head.target_vocab_size = 0;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_blank_out_of_range_is_rejected() {
        let mut c = OmniasrCtcConfig::tiny_for_tests();
        c.head.blank_id = c.head.target_vocab_size as u32;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_non_finite_dropout_is_rejected() {
        let mut c = OmniasrCtcConfig::tiny_for_tests();
        c.head.final_dropout_p = f32::NAN;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn synthesized_weights_are_deterministic_and_shape_correct() {
        let c = OmniasrCtcConfig::tiny_for_tests();
        let w1 = OmniasrCtcWeights::synthesized(&c, 0x42).expect("build 1");
        let w2 = OmniasrCtcWeights::synthesized(&c, 0x42).expect("build 2");
        // Determinism: identical seeds ⇒ identical draws across every
        // separately-drawn tensor.
        assert_eq!(
            w1.feature_extractor[0].conv_w,
            w2.feature_extractor[0].conv_w
        );
        assert_eq!(
            w1.feature_projection.linear_w,
            w2.feature_projection.linear_w
        );
        assert_eq!(
            w1.pos_encoder_layers[0].conv_w,
            w2.pos_encoder_layers[0].conv_w
        );
        assert_eq!(w1.encoder_blocks[0].qkv_proj, w2.encoder_blocks[0].qkv_proj);
        assert_eq!(w1.head.vocab_head, w2.head.vocab_head);
        assert!(w1.is_synthesized);

        // Shape flow.
        let enc = &c.encoder;
        let head = &c.head;
        let d_enc = enc.model_dim;
        let ffn = enc.ffn_inner_dim;
        let vocab = head.target_vocab_size;
        let feat_dim = enc.feature_dim;
        // Feature extractor: 7 layers, in_dim walks 1, out_dim, out_dim, ...
        assert_eq!(
            w1.feature_extractor.len(),
            OMNIASR_CTC_NUM_FEATURE_EXTRACTOR_LAYERS
        );
        let mut in_dim: usize = 1;
        for (i, lw) in w1.feature_extractor.iter().enumerate() {
            let d = &enc.feature_extractor_layer_descs[i];
            assert_eq!(lw.conv_w.len(), d.out_dim * in_dim * d.kernel);
            // Feature extractor bias / norm are Some for the tiny
            // config (matches large_lv60k / omniASR real config).
            assert!(lw.conv_b.is_some());
            assert_eq!(lw.conv_b.as_ref().unwrap().len(), d.out_dim);
            assert!(lw.norm_gamma.is_some());
            assert_eq!(lw.norm_gamma.as_ref().unwrap().len(), d.out_dim);
            assert!(lw.norm_beta.is_some());
            assert_eq!(lw.norm_beta.as_ref().unwrap().len(), d.out_dim);
            in_dim = d.out_dim;
        }
        // Feature projection: fairseq2 post-extraction LayerNorm is always
        // present, then Linear feat_dim → d_enc.
        assert!(w1.feature_projection.norm_gamma.is_some());
        assert!(w1.feature_projection.norm_beta.is_some());
        assert_eq!(w1.feature_projection.linear_w.len(), feat_dim * d_enc);
        assert_eq!(w1.feature_projection.linear_b.len(), d_enc);
        assert!(
            w1.frontend_model_dim_norm_gamma.is_none(),
            "1B layer_norm_features=false must omit the separate model-dim norm"
        );
        assert!(w1.frontend_model_dim_norm_beta.is_none());
        // Positional encoder: 1 layer, [d_enc, d_enc/n_groups, k].
        let per_group = d_enc / enc.num_pos_conv_groups;
        assert_eq!(w1.pos_encoder_layers.len(), enc.pos_encoder_depth);
        assert_eq!(
            w1.pos_encoder_layers[0].conv_w.len(),
            d_enc * per_group * enc.pos_conv_kernel_size
        );
        assert_eq!(w1.pos_encoder_layers[0].conv_b.len(), d_enc);
        // Encoder blocks.
        assert_eq!(w1.encoder_blocks.len(), enc.num_encoder_layers);
        for blk in &w1.encoder_blocks {
            assert_eq!(blk.attn_norm_gamma.len(), d_enc);
            assert_eq!(blk.attn_norm_beta.len(), d_enc);
            assert_eq!(blk.qkv_proj.len(), d_enc * 3 * d_enc);
            assert_eq!(blk.qkv_bias.len(), 3 * d_enc);
            assert_eq!(blk.attn_out.len(), d_enc * d_enc);
            assert_eq!(blk.attn_out_bias.len(), d_enc);
            assert_eq!(blk.ffn_norm_gamma.len(), d_enc);
            assert_eq!(blk.ffn_norm_beta.len(), d_enc);
            assert_eq!(blk.ffn_fc1.len(), d_enc * ffn);
            assert_eq!(blk.ffn_fc1_bias.len(), ffn);
            assert_eq!(blk.ffn_fc2.len(), ffn * d_enc);
            assert_eq!(blk.ffn_fc2_bias.len(), d_enc);
        }
        assert_eq!(w1.encoder_final_norm_gamma.len(), d_enc);
        assert_eq!(w1.encoder_final_norm_beta.len(), d_enc);
        // CTC head.
        assert_eq!(w1.head.vocab_head.len(), d_enc * vocab);
        assert_eq!(w1.head.vocab_bias.len(), vocab);
    }

    /// A no-norm / no-bias variant (a hypothetical future config with
    /// `feature_extractor_bias=false` /
    /// `feature_extractor_layer_norm_convs=false`) drops the
    /// corresponding tensors to None; the synthesized builder must
    /// respect the flags, and the runtime must accept the resulting
    /// None triples.
    #[test]
    fn synthesized_weights_respect_feature_extractor_bias_and_norm_off() {
        let mut c = OmniasrCtcConfig::tiny_for_tests();
        c.encoder.feature_extractor_bias = false;
        c.encoder.feature_extractor_layer_norm_convs = false;
        let w = OmniasrCtcWeights::synthesized(&c, 7).expect("weights");
        for lw in &w.feature_extractor {
            assert!(
                lw.conv_b.is_none(),
                "conv_b must be None when feature_extractor_bias=false"
            );
            assert!(
                lw.norm_gamma.is_none(),
                "norm_gamma must be None when feature_extractor_layer_norm_convs=false"
            );
            assert!(
                lw.norm_beta.is_none(),
                "norm_beta must be None when feature_extractor_layer_norm_convs=false"
            );
        }
        OmniasrCtcAsr::new(c, w).expect("bias/norm-free variant is loadable");
    }

    /// The feature projection always carries the fairseq2 frontend's
    /// post-extraction LayerNorm. The separate model-dimension norm is
    /// present only when `layer_norm_features=true`.
    #[test]
    fn synthesized_weights_respect_layer_norm_features_on() {
        let mut c = OmniasrCtcConfig::tiny_for_tests();
        c.encoder.layer_norm_features = true;
        let w = OmniasrCtcWeights::synthesized(&c, 7).expect("weights");
        assert!(w.feature_projection.norm_gamma.is_some());
        assert!(w.feature_projection.norm_beta.is_some());
        assert_eq!(
            w.feature_projection.norm_gamma.as_ref().unwrap().len(),
            c.encoder.feature_dim
        );
        assert_eq!(
            w.feature_projection.norm_beta.as_ref().unwrap().len(),
            c.encoder.feature_dim
        );
        assert_eq!(
            w.frontend_model_dim_norm_gamma.as_ref().unwrap().len(),
            c.encoder.model_dim
        );
        assert_eq!(
            w.frontend_model_dim_norm_beta.as_ref().unwrap().len(),
            c.encoder.model_dim
        );
        OmniasrCtcAsr::new(c, w).expect("layer_norm_features=true variant is loadable");
    }

    #[test]
    fn asr_new_rejects_missing_frontend_model_dim_norm() {
        let mut cfg = OmniasrCtcConfig::tiny_for_tests();
        cfg.encoder.layer_norm_features = true;
        let mut weights = OmniasrCtcWeights::synthesized(&cfg, 7).expect("weights");
        weights.frontend_model_dim_norm_beta = None;
        let error = OmniasrCtcAsr::new(cfg, weights).expect_err("required norm must be present");
        assert!(
            matches!(error, VokraError::InvalidArgument(message) if message.contains("frontend_model_dim_norm_beta"))
        );
    }

    #[test]
    fn native_forward_accepts_model_dim_norm_before_transformer() {
        let mut cfg = OmniasrCtcConfig::tiny_for_tests();
        cfg.encoder.layer_norm_features = true;
        let mut weights = OmniasrCtcWeights::synthesized(&cfg, 13).expect("weights");
        weights.is_synthesized = false;
        let asr = OmniasrCtcAsr::new(cfg, weights).expect("valid bound shapes");
        asr.encode_features(&vec![0.0; 16_000])
            .expect("model-dim frontend norm must run after positional encoding");
    }

    #[test]
    fn synthesized_weights_different_seeds_diverge() {
        let c = OmniasrCtcConfig::tiny_for_tests();
        let w_a = OmniasrCtcWeights::synthesized(&c, 1).expect("build a");
        let w_b = OmniasrCtcWeights::synthesized(&c, 2).expect("build b");
        assert_ne!(
            w_a.feature_extractor[0].conv_w,
            w_b.feature_extractor[0].conv_w
        );
    }

    #[test]
    fn synthesized_rejects_ill_formed_config() {
        let mut c = OmniasrCtcConfig::tiny_for_tests();
        c.encoder.num_encoder_attn_heads = 3; // 16 % 3 != 0
        assert!(matches!(
            OmniasrCtcWeights::synthesized(&c, 7),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn asr_new_accepts_matching_config_and_weights() {
        let c = OmniasrCtcConfig::tiny_for_tests();
        let w = OmniasrCtcWeights::synthesized(&c, 7).expect("weights");
        let asr = OmniasrCtcAsr::new(c.clone(), w).expect("omniasr-ctc asr");
        assert_eq!(asr.config().encoder.model_dim, c.encoder.model_dim);
        assert!(asr.is_synthesized());
    }

    #[test]
    fn asr_new_rejects_encoder_layer_count_mismatch() {
        let c = OmniasrCtcConfig::tiny_for_tests();
        let mut w = OmniasrCtcWeights::synthesized(&c, 7).expect("weights");
        w.encoder_blocks.pop();
        assert!(matches!(
            OmniasrCtcAsr::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn asr_new_rejects_feature_extractor_layer_count_mismatch() {
        let c = OmniasrCtcConfig::tiny_for_tests();
        let mut w = OmniasrCtcWeights::synthesized(&c, 7).expect("weights");
        w.feature_extractor.pop();
        assert!(matches!(
            OmniasrCtcAsr::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn asr_new_rejects_feature_extractor_conv_w_size_mismatch() {
        let c = OmniasrCtcConfig::tiny_for_tests();
        let mut w = OmniasrCtcWeights::synthesized(&c, 7).expect("weights");
        w.feature_extractor[0].conv_w.pop();
        assert!(matches!(
            OmniasrCtcAsr::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn asr_new_rejects_missing_conv_bias_when_flag_on() {
        let c = OmniasrCtcConfig::tiny_for_tests();
        let mut w = OmniasrCtcWeights::synthesized(&c, 7).expect("weights");
        w.feature_extractor[0].conv_b = None;
        assert!(matches!(
            OmniasrCtcAsr::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn asr_new_rejects_present_conv_bias_when_flag_off() {
        // Build a bias-free variant, then splice in a stray bias
        // vector — the runtime must refuse it (a bias-carrying variant
        // must set feature_extractor_bias=true so the runtime knows to
        // use them).
        let mut c = OmniasrCtcConfig::tiny_for_tests();
        c.encoder.feature_extractor_bias = false;
        let mut w = OmniasrCtcWeights::synthesized(&c, 7).expect("weights");
        w.feature_extractor[0].conv_b = Some(vec![
            0.0;
            c.encoder.feature_extractor_layer_descs[0]
                .out_dim
        ]);
        assert!(matches!(
            OmniasrCtcAsr::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn asr_new_rejects_feature_projection_linear_size_mismatch() {
        let c = OmniasrCtcConfig::tiny_for_tests();
        let mut w = OmniasrCtcWeights::synthesized(&c, 7).expect("weights");
        w.feature_projection.linear_w.pop();
        assert!(matches!(
            OmniasrCtcAsr::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn asr_new_rejects_pos_encoder_depth_mismatch() {
        let c = OmniasrCtcConfig::tiny_for_tests();
        let mut w = OmniasrCtcWeights::synthesized(&c, 7).expect("weights");
        w.pos_encoder_layers.pop();
        assert!(matches!(
            OmniasrCtcAsr::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn asr_new_rejects_pos_encoder_conv_w_size_mismatch() {
        let c = OmniasrCtcConfig::tiny_for_tests();
        let mut w = OmniasrCtcWeights::synthesized(&c, 7).expect("weights");
        w.pos_encoder_layers[0].conv_w.pop();
        assert!(matches!(
            OmniasrCtcAsr::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn asr_new_rejects_encoder_final_norm_mismatch() {
        let c = OmniasrCtcConfig::tiny_for_tests();
        let mut w = OmniasrCtcWeights::synthesized(&c, 7).expect("weights");
        w.encoder_final_norm_gamma.pop();
        assert!(matches!(
            OmniasrCtcAsr::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn asr_new_rejects_head_vocab_head_mismatch() {
        let c = OmniasrCtcConfig::tiny_for_tests();
        let mut w = OmniasrCtcWeights::synthesized(&c, 7).expect("weights");
        w.head.vocab_head.pop();
        assert!(matches!(
            OmniasrCtcAsr::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn asr_new_rejects_head_vocab_bias_mismatch() {
        let c = OmniasrCtcConfig::tiny_for_tests();
        let mut w = OmniasrCtcWeights::synthesized(&c, 7).expect("weights");
        w.head.vocab_bias.pop();
        assert!(matches!(
            OmniasrCtcAsr::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn asr_new_rejects_encoder_qkv_proj_size_mismatch() {
        let c = OmniasrCtcConfig::tiny_for_tests();
        let mut w = OmniasrCtcWeights::synthesized(&c, 7).expect("weights");
        w.encoder_blocks[0].qkv_proj.pop();
        assert!(matches!(
            OmniasrCtcAsr::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn transcribe_rejects_empty_pcm() {
        let c = OmniasrCtcConfig::tiny_for_tests();
        let w = OmniasrCtcWeights::synthesized(&c, 7).expect("weights");
        let asr = OmniasrCtcAsr::new(c, w).expect("omniasr-ctc asr");
        assert!(matches!(
            asr.transcribe(&[]),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    /// The primary NotImplemented path names the synthesized-weight
    /// blocker (FR-EX-08 — never a silent zero-fill / hallucinated
    /// transcript).
    #[test]
    fn transcribe_on_synthesized_weights_is_loud_not_implemented() {
        let c = OmniasrCtcConfig::tiny_for_tests();
        let w = OmniasrCtcWeights::synthesized(&c, 7).expect("weights");
        let asr = OmniasrCtcAsr::new(c, w).expect("omniasr-ctc asr");
        let pcm = vec![0.0f32; 1024];
        let err = asr.transcribe(&pcm).unwrap_err();
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
    fn expected_arch_is_omniasr_ctc() {
        assert_eq!(EXPECTED_ARCH, "omniasr-ctc");
    }

    #[test]
    fn sample_rate_matches_model_card_boundary() {
        // 16 kHz — per the model card + wav2vec 2.0 convention.
        assert_eq!(OMNIASR_CTC_SAMPLE_RATE, 16_000);
    }

    /// The 7-layer stem count is a pinned constant, not a field —
    /// mirrors the fairseq2 `feature_extractor_layer_descs` default
    /// used by every wav2vec 2.0 variant omniASR ships. Guarantees no
    /// silent-8-layer variant slips into a checkpoint (FR-EX-08).
    #[test]
    fn feature_extractor_layer_count_is_pinned() {
        assert_eq!(OMNIASR_CTC_NUM_FEATURE_EXTRACTOR_LAYERS, 7);
    }

    /// The authenticated 1B topology reaches only these learned operators.
    /// Keep this registry in lock-step with the forward below: every learned
    /// tensor application must be represented here so a non-CPU backend is
    /// rejected before execution if one of the required kernels is absent.
    /// GroupNorm is deliberately absent: the authenticated `large_lv60k`
    /// configuration is `LayerAll`, and `from_gguf` rejects any other config.
    #[test]
    fn authenticated_1b_declares_only_compute_dispatched_learned_ops() {
        assert_eq!(
            OMNIASR_CTC_HOT_OPS,
            &[
                HotOp::Gemm,
                HotOp::Softmax,
                HotOp::LayerNorm,
                HotOp::Gelu,
                HotOp::Conv1d,
                HotOp::GroupedConv1d,
            ]
        );
        let cfg = OmniasrCtcConfig::omniasr_ctc_1b();
        assert!(cfg.encoder.feature_extractor_layer_norm_convs);
        assert!(!OMNIASR_CTC_HOT_OPS.contains(&HotOp::GroupNorm));
    }

    // ---- SoTA reuse bundle (2026-07-30): variant enum + 300M / 7B ----

    #[test]
    fn variant_model_id_slugs_are_stable() {
        assert_eq!(OmniasrCtcVariant::M300.model_id(), "omniasr-ctc-300m");
        assert_eq!(OmniasrCtcVariant::B1.model_id(), "omniasr-ctc-1b");
        assert_eq!(OmniasrCtcVariant::B7.model_id(), "omniasr-ctc-7b");
    }

    /// omniASR-CTC-300M carries the fairseq2 wav2vec 2.0 **base** arch
    /// axes (distinct from large_lv60k = 1B).
    #[test]
    fn omniasr_ctc_300m_carries_base_arch_axes() {
        let c = OmniasrCtcConfig::omniasr_ctc_300m();
        // Transformer axes = base.
        assert_eq!(c.encoder.model_dim, 768, "base arch: hidden 768");
        assert_eq!(c.encoder.num_encoder_layers, 12, "base arch: 12 layers");
        assert_eq!(
            c.encoder.num_encoder_attn_heads, 12,
            "base arch: 12 heads (head_dim=64)"
        );
        assert_eq!(
            c.encoder.ffn_inner_dim, 3072,
            "base arch: FFN 3072 (~4× hidden)"
        );
        // Waveform extractor shares the 7-layer stem (size-invariant),
        // but the base arch axes differ from large_lv60k.
        assert_eq!(c.encoder.feature_dim, 512);
        assert_eq!(
            c.encoder.feature_extractor_layer_descs.len(),
            OMNIASR_CTC_NUM_FEATURE_EXTRACTOR_LAYERS
        );
        assert_eq!(c.encoder.feature_extractor_total_stride(), 320);
        assert!(
            !c.encoder.feature_extractor_bias,
            "base arch: no per-layer conv bias (distinct from large_lv60k = true)"
        );
        assert!(
            !c.encoder.feature_extractor_layer_norm_convs,
            "base arch: GroupNorm on stem, not per-layer LayerNorm (distinct from large_lv60k = true)"
        );
        assert!(
            c.encoder.layer_norm_features,
            "base arch: separate post-pos/model-dimension normalization enabled"
        );
        // Head axes are family-wide (not scale-dependent).
        assert_eq!(c.head.target_vocab_size, 9812);
        assert_eq!(c.head.blank_id, 0);
        assert_eq!(c.sample_rate, 16_000);
        // Derived axes are well-formed.
        assert_eq!(c.encoder.head_dim(), 64, "768 / 12 = 64");
        c.validate_for_forward()
            .expect("omniasr-ctc-300m is well-formed");
    }

    /// omniASR-CTC-7B carries `0`-placeholder transformer axes (Meta's
    /// release does not publish a `config.json`) — the runtime rejects
    /// the placeholder loudly (FR-EX-08). Non-placeholder axes match
    /// the large_lv60k arch preset.
    #[test]
    fn omniasr_ctc_7b_carries_zero_placeholder_transformer_axes() {
        let c = OmniasrCtcConfig::omniasr_ctc_7b();
        // Placeholder — must be 0 to force the validator to reject.
        assert_eq!(c.encoder.model_dim, 0, "placeholder pending .pt inspect");
        assert_eq!(
            c.encoder.num_encoder_layers, 0,
            "placeholder pending .pt inspect"
        );
        assert_eq!(
            c.encoder.num_encoder_attn_heads, 0,
            "placeholder pending .pt inspect"
        );
        assert_eq!(
            c.encoder.ffn_inner_dim, 0,
            "placeholder pending .pt inspect"
        );
        // Non-placeholder — large_lv60k axes + family-wide constants.
        assert_eq!(c.encoder.feature_dim, 512);
        assert!(c.encoder.feature_extractor_bias);
        assert!(c.encoder.feature_extractor_layer_norm_convs);
        assert!(!c.encoder.layer_norm_features);
        assert_eq!(c.encoder.pos_conv_kernel_size, 128);
        assert_eq!(c.encoder.max_seq_len, 4096);
        assert_eq!(c.head.target_vocab_size, 9812);
        assert_eq!(c.head.blank_id, 0);
        // Placeholder axes = 0 → validate_for_forward rejects loudly.
        let err = c
            .validate_for_forward()
            .expect_err("0-placeholder transformer axes must reject");
        match err {
            VokraError::InvalidArgument(msg) => {
                assert!(
                    msg.contains("encoder"),
                    "message must name encoder blocker: {msg}"
                );
            }
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    /// `for_variant()` dispatches correctly to the three config
    /// methods.
    #[test]
    fn for_variant_dispatches_to_matching_config() {
        assert_eq!(
            OmniasrCtcConfig::for_variant(OmniasrCtcVariant::M300),
            OmniasrCtcConfig::omniasr_ctc_300m()
        );
        assert_eq!(
            OmniasrCtcConfig::for_variant(OmniasrCtcVariant::B1),
            OmniasrCtcConfig::omniasr_ctc_1b()
        );
        assert_eq!(
            OmniasrCtcConfig::for_variant(OmniasrCtcVariant::B7),
            OmniasrCtcConfig::omniasr_ctc_7b()
        );
    }

    /// Synthesized-weight round-trip works for both real variants that
    /// have non-placeholder transformer axes (300M + 1B). The 7B
    /// placeholder config cannot synthesize (validator rejects the
    /// `0`-axes upstream), which is honest partial by design.
    #[test]
    fn synthesized_round_trip_covers_300m_and_1b_variants() {
        for cfg in [
            OmniasrCtcConfig::omniasr_ctc_300m(),
            OmniasrCtcConfig::omniasr_ctc_1b(),
        ] {
            let w = OmniasrCtcWeights::synthesized(&cfg, 42)
                .expect("synth must succeed for well-formed variant");
            let asr = OmniasrCtcAsr::new(cfg.clone(), w).expect("asr must accept matching pair");
            assert!(asr.is_synthesized());
        }
    }

    /// The 7B variant rejects synthesized-weight construction because
    /// its transformer dims are `0`-placeholders (the validator refuses
    /// them). This is the FR-EX-08-compliant fail-loud path — a caller
    /// cannot silently run a hallucinated 7B forward.
    #[test]
    fn synthesized_rejects_7b_placeholder_dims() {
        let cfg = OmniasrCtcConfig::omniasr_ctc_7b();
        assert!(matches!(
            OmniasrCtcWeights::synthesized(&cfg, 42),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    /// Variants use distinct transformer widths — a converter that
    /// picks the wrong variant would silently mis-slot the encoder
    /// weights.
    #[test]
    fn variants_have_distinct_transformer_widths() {
        let m300 = OmniasrCtcConfig::omniasr_ctc_300m();
        let b1 = OmniasrCtcConfig::omniasr_ctc_1b();
        let b7 = OmniasrCtcConfig::omniasr_ctc_7b();
        assert_ne!(m300.encoder.model_dim, b1.encoder.model_dim);
        assert_ne!(
            m300.encoder.num_encoder_layers,
            b1.encoder.num_encoder_layers
        );
        // 7B's dims are `0`-placeholders — distinct from both real
        // variants by definition.
        assert_eq!(b7.encoder.model_dim, 0);
        assert_eq!(b7.encoder.num_encoder_layers, 0);
    }

    // -----------------------------------------------------------------------
    // Manifest-gated `from_gguf` contract: topology metadata is parsed, but
    // payloads without the audited tensor-name/shape manifest never construct
    // an engine or surface provenance as runtime state.
    // -----------------------------------------------------------------------

    /// Builds a minimal omniASR-CTC GGUF carrying the arch tag + full
    /// `vokra.omniasr_ctc.*` chunk group matching the passed config
    /// (feature extractor stem walked as `count + 7 × (out_dim,
    /// kernel, stride)`). `weight_license_class` is written under
    /// `vokra.provenance.weight_license` (or omitted if `None`).
    fn omniasr_ctc_gguf(
        cfg: &OmniasrCtcConfig,
        weight_license_class: Option<LicenseClass>,
    ) -> vokra_core::gguf::GgufFile {
        use vokra_core::gguf::{GgufBuilder, GgufFile};

        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, EXPECTED_ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, "omniasr-ctc-1b");
        // Chunk group — mirrors the converter (`write_hparams`).
        b.add_u32(KEY_SAMPLE_RATE, cfg.sample_rate);
        // Encoder
        b.add_u32(KEY_ENC_MODEL_DIM, cfg.encoder.model_dim as u32);
        b.add_u32(KEY_ENC_N_LAYER, cfg.encoder.num_encoder_layers as u32);
        b.add_u32(KEY_ENC_N_HEAD, cfg.encoder.num_encoder_attn_heads as u32);
        b.add_u32(KEY_ENC_FFN_INNER, cfg.encoder.ffn_inner_dim as u32);
        b.add_u32(KEY_ENC_FEATURE_DIM, cfg.encoder.feature_dim as u32);
        b.add_u32(KEY_ENC_MAX_SEQ_LEN, cfg.encoder.max_seq_len as u32);
        b.add_u32(
            KEY_ENC_FEATURE_BIAS,
            u32::from(cfg.encoder.feature_extractor_bias),
        );
        b.add_u32(
            KEY_ENC_FEATURE_LN_CONVS,
            u32::from(cfg.encoder.feature_extractor_layer_norm_convs),
        );
        b.add_u32(
            KEY_ENC_LN_FEATURES,
            u32::from(cfg.encoder.layer_norm_features),
        );
        b.add_u32(KEY_ENC_POS_KERNEL, cfg.encoder.pos_conv_kernel_size as u32);
        b.add_u32(KEY_ENC_POS_GROUPS, cfg.encoder.num_pos_conv_groups as u32);
        b.add_u32(KEY_ENC_POS_DEPTH, cfg.encoder.pos_encoder_depth as u32);
        b.add_u32(KEY_ENC_USE_CONFORMER, u32::from(cfg.encoder.use_conformer));
        // Feature extractor stem — count + 7 × (out_dim, kernel, stride).
        b.add_u32(
            KEY_ENC_FEATURE_LAYERS,
            OMNIASR_CTC_NUM_FEATURE_EXTRACTOR_LAYERS as u32,
        );
        for i in 0..OMNIASR_CTC_NUM_FEATURE_EXTRACTOR_LAYERS {
            b.add_u32(
                &format!("{KEY_ENC_FEATURE_OUT_PREFIX}{i}"),
                cfg.encoder.feature_extractor_layer_descs[i].out_dim as u32,
            );
            b.add_u32(
                &format!("{KEY_ENC_FEATURE_KERNEL_PREFIX}{i}"),
                cfg.encoder.feature_extractor_layer_descs[i].kernel as u32,
            );
            b.add_u32(
                &format!("{KEY_ENC_FEATURE_STRIDE_PREFIX}{i}"),
                cfg.encoder.feature_extractor_layer_descs[i].stride as u32,
            );
        }
        // Head
        b.add_u32(KEY_HEAD_VOCAB_SIZE, cfg.head.target_vocab_size as u32);
        b.add_u32(KEY_HEAD_BLANK_ID, cfg.head.blank_id);
        if let Some(cls) = weight_license_class {
            b.add_string(chunks::KEY_PROVENANCE_WEIGHT_LICENSE, cls.as_str());
        }
        // Keep the helper payload-bearing. The production binder rejects this
        // one-tensor fixture because it is not the pinned VAST payload; the
        // helper is only for topology/provenance metadata tests.
        b.add_tensor(
            "test.payload",
            vokra_core::gguf::GgmlType::F32,
            vec![1],
            vec![0; 4],
        )
        .expect("test payload tensor");
        GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse")
    }

    #[test]
    fn from_gguf_rejects_metadata_only_artifact() {
        use vokra_core::gguf::{GgufBuilder, GgufFile};

        let cfg = OmniasrCtcConfig::tiny_for_tests();
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, EXPECTED_ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, "omniasr-ctc-1b");
        b.add_u32(KEY_SAMPLE_RATE, cfg.sample_rate);
        b.add_u32(KEY_ENC_MODEL_DIM, cfg.encoder.model_dim as u32);
        b.add_u32(KEY_ENC_N_LAYER, cfg.encoder.num_encoder_layers as u32);
        b.add_u32(KEY_ENC_N_HEAD, cfg.encoder.num_encoder_attn_heads as u32);
        b.add_u32(KEY_ENC_FFN_INNER, cfg.encoder.ffn_inner_dim as u32);
        b.add_u32(KEY_ENC_FEATURE_DIM, cfg.encoder.feature_dim as u32);
        b.add_u32(KEY_ENC_MAX_SEQ_LEN, cfg.encoder.max_seq_len as u32);
        b.add_u32(
            KEY_ENC_FEATURE_BIAS,
            u32::from(cfg.encoder.feature_extractor_bias),
        );
        b.add_u32(
            KEY_ENC_FEATURE_LN_CONVS,
            u32::from(cfg.encoder.feature_extractor_layer_norm_convs),
        );
        b.add_u32(
            KEY_ENC_LN_FEATURES,
            u32::from(cfg.encoder.layer_norm_features),
        );
        b.add_u32(KEY_ENC_POS_KERNEL, cfg.encoder.pos_conv_kernel_size as u32);
        b.add_u32(KEY_ENC_POS_GROUPS, cfg.encoder.num_pos_conv_groups as u32);
        b.add_u32(KEY_ENC_POS_DEPTH, cfg.encoder.pos_encoder_depth as u32);
        b.add_u32(KEY_ENC_USE_CONFORMER, u32::from(cfg.encoder.use_conformer));
        b.add_u32(
            KEY_ENC_FEATURE_LAYERS,
            OMNIASR_CTC_NUM_FEATURE_EXTRACTOR_LAYERS as u32,
        );
        for (i, desc) in cfg.encoder.feature_extractor_layer_descs.iter().enumerate() {
            b.add_u32(
                &format!("{KEY_ENC_FEATURE_OUT_PREFIX}{i}"),
                desc.out_dim as u32,
            );
            b.add_u32(
                &format!("{KEY_ENC_FEATURE_KERNEL_PREFIX}{i}"),
                desc.kernel as u32,
            );
            b.add_u32(
                &format!("{KEY_ENC_FEATURE_STRIDE_PREFIX}{i}"),
                desc.stride as u32,
            );
        }
        b.add_u32(KEY_HEAD_VOCAB_SIZE, cfg.head.target_vocab_size as u32);
        b.add_u32(KEY_HEAD_BLANK_ID, cfg.head.blank_id);
        let file = GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse");
        let error = OmniasrCtcAsr::from_gguf(&file).expect_err("metadata-only GGUF must fail");
        match error {
            VokraError::ModelLoad(message) => {
                assert!(!message.is_empty(), "error must explain the rejection");
            }
            other => panic!("expected ModelLoad, got {other:?}"),
        }
    }

    /// A `whisper` / `voxtral` / `parakeet-ctc` GGUF handed to the
    /// omniASR-CTC binder by mistake must fail loud with a specific
    /// message rather than silently mis-binding (FR-EX-08). The
    /// Parakeet-CTC case gets a dedicated hint pointing at the sibling
    /// binder (very close ASR sibling — same CTC head shape but
    /// different encoder body: FastConformer vs wav2vec 2.0).
    #[test]
    fn from_gguf_rejects_wrong_arch() {
        use vokra_core::gguf::{GgufBuilder, GgufFile};

        // Generic wrong arch — names both got + expected + sibling
        // ASR arches.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, "whisper");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = OmniasrCtcAsr::from_gguf(&file) else {
            panic!("expected ModelLoad on wrong arch");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("`whisper`") && m.contains("`omniasr-ctc`"),
                    "message must name both got + expected arch tags, got `{m}`"
                );
                // Sibling ASR arches are enumerated in the hint so a
                // reader has one place to walk.
                assert!(
                    m.contains("parakeet-ctc") || m.contains("voxtral"),
                    "message must name sibling ASR arches for disambiguation, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }

        // Parakeet-CTC sibling — dedicated hint pointing at the
        // FastConformer binder so a reader diagnosing this mis-route
        // has exactly one place to walk.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, "parakeet-ctc");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = OmniasrCtcAsr::from_gguf(&file) else {
            panic!("expected ModelLoad on parakeet-ctc");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    (m.contains("FastConformer") || m.contains("Parakeet-CTC"))
                        && m.contains("parakeet_ctc::ParakeetCtcAsr"),
                    "message must name the FastConformer / Parakeet-CTC sibling + \
                     point at the sibling binder, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    /// A GGUF that omits `vokra.model.arch` entirely fails loud
    /// (converter did not stamp it — the GGUF is not Vokra-native).
    #[test]
    fn from_gguf_rejects_missing_arch() {
        use vokra_core::gguf::{GgufBuilder, GgufFile};

        // No arch chunk at all — but we need at least one metadata key
        // to build a valid GGUF, so include a benign name.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_NAME, "unknown");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = OmniasrCtcAsr::from_gguf(&file) else {
            panic!("expected ModelLoad on missing arch");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("`vokra.model.arch`") && m.contains("did not stamp"),
                    "message must name the missing arch key + fingerprint the converter, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    /// Every mandatory `vokra.omniasr_ctc.*` chunk is required — a
    /// converter that fails to stamp any one is a converter bug, not a
    /// runtime silent-default (FR-EX-08). The loud error names the
    /// exact absent chunk key.
    #[test]
    fn from_gguf_rejects_missing_encoder_axis() {
        use vokra_core::gguf::{GgufBuilder, GgufFile};

        let cfg = OmniasrCtcConfig::omniasr_ctc_1b();
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, EXPECTED_ARCH);
        b.add_u32(KEY_SAMPLE_RATE, cfg.sample_rate);
        // Deliberately omit KEY_ENC_MODEL_DIM — every other encoder axis
        // is stamped so the loud error must fire on `model_dim`.
        b.add_u32(KEY_ENC_N_LAYER, cfg.encoder.num_encoder_layers as u32);
        b.add_u32(KEY_ENC_N_HEAD, cfg.encoder.num_encoder_attn_heads as u32);
        b.add_u32(KEY_ENC_FFN_INNER, cfg.encoder.ffn_inner_dim as u32);
        b.add_u32(KEY_ENC_FEATURE_DIM, cfg.encoder.feature_dim as u32);
        b.add_u32(KEY_ENC_MAX_SEQ_LEN, cfg.encoder.max_seq_len as u32);
        b.add_u32(
            KEY_ENC_FEATURE_BIAS,
            u32::from(cfg.encoder.feature_extractor_bias),
        );
        b.add_u32(
            KEY_ENC_FEATURE_LN_CONVS,
            u32::from(cfg.encoder.feature_extractor_layer_norm_convs),
        );
        b.add_u32(
            KEY_ENC_LN_FEATURES,
            u32::from(cfg.encoder.layer_norm_features),
        );
        b.add_u32(KEY_ENC_POS_KERNEL, cfg.encoder.pos_conv_kernel_size as u32);
        b.add_u32(KEY_ENC_POS_GROUPS, cfg.encoder.num_pos_conv_groups as u32);
        b.add_u32(KEY_ENC_POS_DEPTH, cfg.encoder.pos_encoder_depth as u32);
        b.add_u32(KEY_ENC_USE_CONFORMER, u32::from(cfg.encoder.use_conformer));
        b.add_u32(
            KEY_ENC_FEATURE_LAYERS,
            OMNIASR_CTC_NUM_FEATURE_EXTRACTOR_LAYERS as u32,
        );
        for i in 0..OMNIASR_CTC_NUM_FEATURE_EXTRACTOR_LAYERS {
            b.add_u32(
                &format!("{KEY_ENC_FEATURE_OUT_PREFIX}{i}"),
                cfg.encoder.feature_extractor_layer_descs[i].out_dim as u32,
            );
            b.add_u32(
                &format!("{KEY_ENC_FEATURE_KERNEL_PREFIX}{i}"),
                cfg.encoder.feature_extractor_layer_descs[i].kernel as u32,
            );
            b.add_u32(
                &format!("{KEY_ENC_FEATURE_STRIDE_PREFIX}{i}"),
                cfg.encoder.feature_extractor_layer_descs[i].stride as u32,
            );
        }
        b.add_u32(KEY_HEAD_VOCAB_SIZE, cfg.head.target_vocab_size as u32);
        b.add_u32(KEY_HEAD_BLANK_ID, cfg.head.blank_id);
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();

        let Err(err) = OmniasrCtcAsr::from_gguf(&file) else {
            panic!("expected ModelLoad on missing encoder axis");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains(KEY_ENC_MODEL_DIM),
                    "message must name the exact missing chunk key `{KEY_ENC_MODEL_DIM}`, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    /// The full omniASR-CTC-1B primary-source config round-trips: stamp
    /// every chunk with the transcribed values, read them back with
    /// `from_gguf`, assert every field of the resulting
    /// `OmniasrCtcConfig` equals `omniasr_ctc_1b()`.
    #[test]
    fn from_gguf_reads_full_omniasr_ctc_1b_config() {
        let cfg = OmniasrCtcConfig::omniasr_ctc_1b();
        let file = omniasr_ctc_gguf(&cfg, None);
        let round_trip =
            OmniasrCtcConfig::from_gguf(&file).expect("valid GGUF metadata must parse");
        assert_eq!(
            round_trip, cfg,
            "every field of the resolved config must round-trip verbatim"
        );
    }

    /// A GGUF carrying provenance still fails closed when its payload is not
    /// the pinned real checkpoint; metadata never substitutes for weights.
    #[test]
    fn from_gguf_rejects_unbound_payload_with_provenance() {
        // The one-tensor helper is intentionally rejected before any weight
        // allocation. The 1B config round trip is covered separately.
        let cfg = OmniasrCtcConfig::tiny_for_tests();
        let file = omniasr_ctc_gguf(&cfg, Some(LicenseClass::Permissive));
        let error = OmniasrCtcAsr::from_gguf(&file).expect_err("unbound payload must fail closed");
        assert!(matches!(error, VokraError::ModelLoad(_)));
    }

    /// A GGUF that omits provenance is also rejected when its payload is not
    /// the pinned real checkpoint; the binder never assumes a license class.
    #[test]
    fn from_gguf_rejects_unbound_payload_without_provenance() {
        // The one-tensor helper is rejected before any synthesized fixture is
        // allocated.
        let cfg = OmniasrCtcConfig::tiny_for_tests();
        let file = omniasr_ctc_gguf(&cfg, None);
        let error = OmniasrCtcAsr::from_gguf(&file).expect_err("unbound payload must fail closed");
        assert!(matches!(error, VokraError::ModelLoad(_)));
    }

    /// A tiny/noncanonical GGUF cannot construct an engine even when its
    /// metadata looks plausible; the production contract is fixed to the
    /// audited 1B artifact.
    #[test]
    fn from_gguf_rejects_noncanonical_payload() {
        // Tiny scale — production binding never substitutes synthesized
        // weights for the audited 807-tensor release.
        let cfg = OmniasrCtcConfig::tiny_for_tests();
        let file = omniasr_ctc_gguf(&cfg, Some(LicenseClass::Permissive));
        let error = OmniasrCtcAsr::from_gguf(&file).expect_err("unbound payload must fail closed");
        assert!(matches!(error, VokraError::ModelLoad(_)));
    }

    #[test]
    fn native_forward_contract_runs_only_for_explicitly_bound_weights() {
        let cfg = OmniasrCtcConfig::tiny_for_tests();
        let mut weights = OmniasrCtcWeights::synthesized(&cfg, 11).expect("weights");
        // Exercise tensor-layout and backend plumbing only. Production GGUF
        // loading never does this and remains strictly manifest-bound.
        weights.is_synthesized = false;
        let asr = OmniasrCtcAsr::new(cfg, weights).expect("valid bound shapes");
        let tokens = asr
            .transcribe_tokens(&vec![0.0; 16_000])
            .expect("tiny native path");
        assert!(tokens.len() <= 16_000 / 320 + 1);
    }

    #[test]
    fn gguf_payload_gate_is_pinned_to_the_vast_manifest_count() {
        assert_eq!(OMNIASR_CTC_EXPECTED_TENSOR_COUNT, 807);
    }

    #[test]
    fn fused_qkv_unpack_preserves_per_frame_triplets() {
        let qkv = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // frame 0: q=[1,2], k=[3,4], v=[5,6]
            11.0, 12.0, 13.0, 14.0, 15.0, 16.0, // frame 1
        ];
        let (q, k, v) = unpack_fused_qkv(&qkv, 2, 2).expect("valid fused output");
        assert_eq!(q, [1.0, 2.0, 11.0, 12.0]);
        assert_eq!(k, [3.0, 4.0, 13.0, 14.0]);
        assert_eq!(v, [5.0, 6.0, 15.0, 16.0]);
    }

    #[test]
    fn audited_manifest_has_all_807_names_and_pinned_shapes() {
        let manifest = omniasr_manifest();
        assert_eq!(manifest.len(), 807);
        assert_eq!(
            manifest["encoder_frontend.pos_encoder.conv.weight_g"],
            vec![1, 1, 128]
        );
        assert_eq!(
            manifest["encoder_frontend.pos_encoder.conv.weight_v"],
            vec![1280, 80, 128]
        );
        assert_eq!(
            manifest["encoder.layers.47.self_attn.v_proj.weight"],
            vec![1280, 1280]
        );
    }

    #[test]
    fn tiny_synthetic_manifest_accepts_and_tampering_fails_closed() {
        use vokra_core::gguf::{GgmlType, GgufBuilder, GgufFile};

        let mut expected = BTreeMap::new();
        expected.insert("q".to_owned(), vec![2, 2]);
        expected.insert("k".to_owned(), vec![2, 2]);
        let mut builder = GgufBuilder::new();
        builder
            .add_tensor("q", GgmlType::F32, vec![2, 2], vec![0; 16])
            .unwrap();
        builder
            .add_tensor("k", GgmlType::F32, vec![2, 2], vec![0; 16])
            .unwrap();
        let file = GgufFile::parse(builder.to_bytes().unwrap()).unwrap();
        assert!(validate_manifest_contract(&file, &expected).is_ok());

        let mut tampered = GgufBuilder::new();
        tampered
            .add_tensor("q", GgmlType::F32, vec![4], vec![0; 16])
            .unwrap();
        tampered
            .add_tensor("k", GgmlType::F32, vec![2, 2], vec![0; 16])
            .unwrap();
        let file = GgufFile::parse(tampered.to_bytes().unwrap()).unwrap();
        assert!(validate_manifest_contract(&file, &expected).is_err());

        let mut extra = GgufBuilder::new();
        extra
            .add_tensor("q", GgmlType::F32, vec![2, 2], vec![0; 16])
            .unwrap();
        extra
            .add_tensor("k", GgmlType::F32, vec![2, 2], vec![0; 16])
            .unwrap();
        extra
            .add_tensor("evil", GgmlType::F32, vec![2, 2], vec![0; 16])
            .unwrap();
        let file = GgufFile::parse(extra.to_bytes().unwrap()).unwrap();
        assert!(validate_manifest_contract(&file, &expected).is_err());

        let mut wrong_name = GgufBuilder::new();
        wrong_name
            .add_tensor("q", GgmlType::F32, vec![2, 2], vec![0; 16])
            .unwrap();
        wrong_name
            .add_tensor("evil", GgmlType::F32, vec![2, 2], vec![0; 16])
            .unwrap();
        let file = GgufFile::parse(wrong_name.to_bytes().unwrap()).unwrap();
        assert!(validate_manifest_contract(&file, &expected).is_err());

        let mut wrong_dtype = GgufBuilder::new();
        wrong_dtype
            .add_tensor("q", GgmlType::F16, vec![2, 2], vec![0; 8])
            .unwrap();
        wrong_dtype
            .add_tensor("k", GgmlType::F32, vec![2, 2], vec![0; 16])
            .unwrap();
        let file = GgufFile::parse(wrong_dtype.to_bytes().unwrap()).unwrap();
        assert!(validate_manifest_contract(&file, &expected).is_err());
    }

    #[test]
    fn qkv_fusion_is_source_ordered_and_weight_norm_uses_kernel_axis() {
        assert_eq!(
            fuse_qkv(
                vec![1.0, 2.0, 3.0, 4.0],
                vec![5.0, 6.0, 7.0, 8.0],
                vec![9.0, 10.0, 11.0, 12.0],
                2,
            )
            .unwrap(),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0
            ]
        );
        assert_eq!(
            fuse_qkv_bias(vec![1.0], vec![2.0], vec![3.0], 1).unwrap(),
            vec![1.0, 2.0, 3.0]
        );
        let actual = materialize_positional_weight_values(
            &[2.0, 4.0],
            &[3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0],
            2,
            2,
            2,
        )
        .unwrap();
        assert!((actual[0] - 3.0 * 2.0 / 9.0f32.sqrt()).abs() < 1e-6);
        assert!((actual[1] - 4.0 * 4.0 / 41.0f32.sqrt()).abs() < 1e-6);
        assert!(materialize_positional_weight_values(&[1.0], &[0.0], 1, 1, 1).is_err());
    }
}
