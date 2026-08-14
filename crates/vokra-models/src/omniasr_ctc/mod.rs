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
//!   - `layer_norm_features` = **false** (large_lv60k — the outer
//!     post-extraction LayerNorm is skipped since every conv layer is
//!     itself layer-normalised),
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
//!   - `target_vocab_size` = 9812 (the v1 tokenizer's char vocab;
//!     `omniASR_tokenizer_v1`, `char_tokenizer` family per
//!     `src/omnilingual_asr/cards/models/rc_models_v1.yaml`),
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
//! **Encoder body — no shared op yet.** Unlike Parakeet's FastConformer
//! encoder (which routes through [`vokra_ops::conformer`]), the wav2vec 2.0
//! encoder is a distinct topology (7-layer waveform Conv1D feature
//! extractor + Conv1D positional encoder + plain Transformer encoder,
//! not a Conformer). No shared "wav2vec 2.0 encoder" op exists in
//! `vokra_ops` today; the task note explicitly calls this out ("may need
//! new op"). This scaffold stops at shape / weight-store flow — a
//! follow-up wave decides whether to (a) extract a shared
//! `vokra_ops::wav2vec2_encoder` op (also usable for the paired W2V /
//! LLM omniASR variants and the jonatasgrosman/wav2vec2 family) or
//! (b) keep the encoder in this module and route only `ctc_decode`
//! through the shared primitive.
//!
//! # What lands in this Phase 2 slice
//!
//! - [`OmniasrCtcConfig`] — every hparam transcribed from the primary
//!   source (no hardcoded fabrication; sample-rate is inherited from the
//!   wav2vec 2.0 waveform convention, documented on the field).
//! - [`OmniasrCtcWeights`] — a scaffold weight store with a deterministic
//!   [`OmniasrCtcWeights::synthesized`] fixture (SplitMix64 + Xavier) so
//!   shape / dtype / size flow can be exercised without the real HF
//!   checkpoint. Real-checkpoint parity is a follow-up wave gated on the
//!   real-checkpoint tensor-name manifest (T29-equivalent — the Moshi /
//!   CSM / Zonos / Kyutai STT / Parakeet-CTC pattern).
//! - [`OmniasrCtcAsr`] — engine handle carrying config + weights.
//!   [`OmniasrCtcAsr::transcribe`] returns [`VokraError::NotImplemented`]
//!   until real weights are bound (the real forward — 16 kHz waveform →
//!   7-layer Conv1D feature extractor → feature projection → Conv1D
//!   positional encoder → 48-layer Transformer encoder → CTC vocab head
//!   → `ctc_decode_greedy(blank_id = 0)` → SentencePiece detokenize —
//!   is a follow-up wave gated on the real-checkpoint tensor manifest).
//!
//! # No ONNX (permanent)
//!
//! omniASR-CTC ships as a fairseq2 `.pt` checkpoint (plus a SentencePiece
//! tokenizer); the pipeline is re-implemented natively in
//! `vokra-models/src/omniasr_ctc/` (whisper.cpp 型, CLAUDE.md 設計判断 4).
//! This module never touches ONNX.

use vokra_core::gguf::{GgufFile, chunks};
use vokra_core::rng::SplitMix64;
use vokra_core::{LicenseClass, Result, VokraError};

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

/// Fixed count of Conv1D layers in the wav2vec 2.0 waveform feature
/// extractor — 7, matching the fairseq2
/// `Wav2Vec2EncoderConfig.feature_extractor_layer_descs` default which
/// is `[(512, 10, 5), (512, 3, 2), (512, 3, 2), (512, 3, 2),
/// (512, 3, 2), (512, 2, 2), (512, 2, 2)]`. The "1b" arch does not
/// override this, so all four omniASR sizes share the same 7-layer
/// stem. Pinned as a constant so the weight-store shape gate cannot
/// silently accept a mismatched checkpoint (FR-EX-08).
pub const OMNIASR_CTC_NUM_FEATURE_EXTRACTOR_LAYERS: usize = 7;

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
    /// `layer_norm_features` — **false** for large_lv60k (the outer
    /// post-extraction LayerNorm is skipped since every conv layer is
    /// itself layer-normed; the base arch has this true).
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
    /// `layer_norm_features=true` — the outer post-extraction LayerNorm
    /// is applied since conv layers are not layer-normalised
    /// individually in the base arch). Same 7-layer Conv1D stem, same
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
    ///   LayerNorm), `layer_norm_features = true` (the outer
    ///   post-extraction LayerNorm is present in base — distinct from
    ///   `large_lv60k` which omits it).
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

/// The feature projection: LayerNorm (optional — when
/// `layer_norm_features=true`) then Linear from `feature_dim` to
/// `model_dim`. large_lv60k omits the LayerNorm.
#[derive(Debug, Clone)]
pub struct OmniasrCtcFeatureProjectionWeights {
    /// `[feature_dim]` — LayerNorm gamma, Some iff
    /// `layer_norm_features=true` (omitted for large_lv60k / omniASR).
    pub norm_gamma: Option<Vec<f32>>,
    /// `[feature_dim]` — LayerNorm beta, Some iff
    /// `layer_norm_features=true`.
    pub norm_beta: Option<Vec<f32>>,
    /// `[feature_dim, model_dim]`.
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
    /// Fused Q/K/V projection, shape `[model_dim, 3*model_dim]` (MHA).
    pub qkv_proj: Vec<f32>,
    /// Fused Q/K/V bias, shape `[3*model_dim]`.
    pub qkv_bias: Vec<f32>,
    /// Attention output projection, shape `[model_dim, model_dim]`.
    pub attn_out: Vec<f32>,
    /// Attention output bias, shape `[model_dim]`.
    pub attn_out_bias: Vec<f32>,
    /// FFN pre-norm γ, shape `[model_dim]`.
    pub ffn_norm_gamma: Vec<f32>,
    /// FFN pre-norm β, shape `[model_dim]`.
    pub ffn_norm_beta: Vec<f32>,
    /// FFN hidden projection, shape `[model_dim, ffn_inner_dim]`.
    pub ffn_fc1: Vec<f32>,
    /// FFN hidden bias, shape `[ffn_inner_dim]`.
    pub ffn_fc1_bias: Vec<f32>,
    /// FFN output projection, shape `[ffn_inner_dim, model_dim]`.
    pub ffn_fc2: Vec<f32>,
    /// FFN output bias, shape `[model_dim]`.
    pub ffn_fc2_bias: Vec<f32>,
}

/// CTC head scaffold: a single Linear from encoder `model_dim` to
/// `target_vocab_size`, plus a bias (fairseq2 wav2vec 2.0 default has
/// `final_proj_bias=True`).
#[derive(Debug, Clone)]
pub struct OmniasrCtcHeadWeights {
    /// `[model_dim, target_vocab_size]` — CTC vocab projection (blank
    /// inclusive at index `blank_id=0`).
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
/// without the real HF checkpoint. Real-checkpoint binding is a
/// follow-up (T29-equivalent — tensor-name manifest fetch from the
/// upstream release).
#[derive(Debug, Clone)]
pub struct OmniasrCtcWeights {
    /// 7 waveform-feature-extractor Conv1D layers, in order.
    pub feature_extractor: Vec<OmniasrCtcFeatureExtractorLayerWeights>,
    /// Feature projection (optional LayerNorm then Linear).
    pub feature_projection: OmniasrCtcFeatureProjectionWeights,
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
    /// — the omniASR / large_lv60k case). Feature-projection LayerNorm
    /// is `Some` iff `layer_norm_features=true` (which is `false` for
    /// large_lv60k, so it stays `None` for omniASR).
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

        // -- Feature projection: optional LayerNorm (skipped for
        //    large_lv60k) + Linear from feature_dim to model_dim.
        let (norm_gamma, norm_beta) = if enc.layer_norm_features {
            (Some(vec![1.0; feat_dim]), Some(vec![0.0; feat_dim]))
        } else {
            (None, None)
        };
        let feature_projection = OmniasrCtcFeatureProjectionWeights {
            norm_gamma,
            norm_beta,
            linear_w: xavier(&mut rng, feat_dim * d_enc, feat_dim, d_enc),
            linear_b: vec![0.0; d_enc],
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

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

/// omniASR-CTC ASR engine handle.
///
/// Carries the resolved config and weight store. [`Self::transcribe`]
/// is the primary waveform → text entry point; until real weights are
/// bound (see the module docstring) it returns
/// [`VokraError::NotImplemented`] with a message naming the blocker
/// (FR-EX-08 — never a silent zero-fill or empty transcript).
///
/// # Weight license surfacing
///
/// The `weight_license` field carries the compliance class surfaced
/// from the GGUF's `vokra.provenance.weight_license` chunk (populated
/// by [`Self::from_gguf`]) or defaults to [`LicenseClass::Permissive`]
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
        let ln_feat_on = enc.layer_norm_features;

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
        for (name, opt) in [
            ("norm_gamma", weights.feature_projection.norm_gamma.as_ref()),
            ("norm_beta", weights.feature_projection.norm_beta.as_ref()),
        ] {
            match (ln_feat_on, opt) {
                (true, Some(v)) if v.len() == feat_dim => {}
                (true, Some(v)) => {
                    return Err(VokraError::InvalidArgument(format!(
                        "omniasr-ctc weights: feature_projection.{name}.len()={} != \
                         feature_dim={}",
                        v.len(),
                        feat_dim,
                    )));
                }
                (true, None) => {
                    return Err(VokraError::InvalidArgument(format!(
                        "omniasr-ctc weights: feature_projection.{name} is None but \
                         layer_norm_features=true — a norm-free variant must set \
                         layer_norm_features=false",
                    )));
                }
                (false, Some(_)) => {
                    return Err(VokraError::InvalidArgument(format!(
                        "omniasr-ctc weights: feature_projection.{name} is Some but \
                         layer_norm_features=false — a norm-carrying variant must set \
                         layer_norm_features=true",
                    )));
                }
                (false, None) => {}
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
            // obligation. `from_gguf` overrides with whatever the
            // provenance chunk carries (or `Unknown` if absent).
            weight_license: LicenseClass::Permissive,
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

    /// The stamped weight-license class surfaced from the GGUF's
    /// `vokra.provenance.weight_license` chunk. For real omniASR-CTC
    /// checkpoints the compliance registry
    /// (`vokra_core::compliance::license_class`) maps every
    /// `omniasr-ctc-*` slug to [`LicenseClass::Permissive`]
    /// (Apache-2.0) via the `omniasr-ctc-` family prefix walk. A GGUF
    /// missing the stamp reads back as [`LicenseClass::Unknown`]
    /// (fail-closed at the outer M2-13 gate); a GGUF stamped with any
    /// non-Permissive class surfaces that class here so the outer
    /// gate can enforce (M2-13 refuses commercial dispatch on
    /// mismatches).
    #[inline]
    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    /// Binds an omniASR-CTC GGUF: validates arch, reads the strict
    /// `vokra.omniasr_ctc.*` topology chunk group, builds a
    /// deterministic synthesized weight fixture matching the resolved
    /// config, and surfaces the stamped weight-license class for
    /// compliance gate cross-checks.
    ///
    /// This binder is a *loud* validation step. Every failure is a
    /// distinct [`VokraError::ModelLoad`] naming the missing / wrong
    /// key so a reader diagnosing a mis-produced GGUF has exactly one
    /// place to walk (FR-EX-08 — never a silent partial bind).
    ///
    /// # Loud-partial contract
    ///
    /// After this returns `Ok(_)`, the resulting engine is a
    /// **synthesized-weight** handle — the shape / dtype / size flow is
    /// exercised end-to-end (config chunk validation, weight-store
    /// construction, PCM boundary check), but calling
    /// [`Self::transcribe`] still returns [`VokraError::NotImplemented`]
    /// naming the real-checkpoint tensor-name manifest binding
    /// (T29-equivalent — the Moshi / CSM / Zonos / Kyutai STT /
    /// Parakeet-CTC pattern) as the follow-up wave's anchor. The
    /// missing pieces are (a) the HF safetensors → fairseq2 state-dict
    /// → [`OmniasrCtcWeights`] tensor-name mapping, and (b) the
    /// wav2vec 2.0 encoder body (no shared op yet — a follow-up wave
    /// decides between extracting `vokra_ops::wav2vec2_encoder` or
    /// keeping the encoder inline). The CTC decoding primitive
    /// ([`vokra_ops::ctc_decode`]) with `blank_id = 0` already exists;
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

        // 3. Provenance surfacing — read the stamped weight-license class
        //    for compliance gate cross-checks (defaults to `Unknown` if
        //    absent, which is fail-closed at the outer M2-13 gate).
        //    Matches the MT3 / SNAC / Parakeet-CTC precedent — surface
        //    the class here, let the outer gate do the strict
        //    enforcement.
        let weight_license = file
            .get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
            .and_then(|v| v.as_str())
            .and_then(LicenseClass::from_class_str)
            .unwrap_or(LicenseClass::Unknown);

        // 4. Build synthesized weights against the freshly-read config
        //    so the engine is constructible. `transcribe` still loud-
        //    partials with the synthesized-weight blocker message —
        //    binding real HF checkpoint tensor names is the follow-up
        //    wave (T29-equivalent).
        let weights = OmniasrCtcWeights::synthesized(&cfg, /* seed */ 0)?;
        let mut asr = Self::new(cfg, weights)?;
        asr.weight_license = weight_license;
        Ok(asr)
    }

    /// Transcribes a mono `f32` PCM slice at [`Self::config`]'s sample
    /// rate.
    ///
    /// This is the primary waveform → text entry point. **Real weights
    /// required**: synthesized-weight builds cannot produce meaningful
    /// text (they would be noise or a hallucinated fixed sequence), so
    /// this returns [`VokraError::NotImplemented`] naming the blocker.
    /// Callers verify the shape flow through [`OmniasrCtcAsr::new`] +
    /// [`OmniasrCtcWeights::synthesized`] today; a follow-up wave binds
    /// the real fairseq2 `.pt` checkpoint tensor names and wires the
    /// forward.
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] if `pcm` is empty.
    /// - [`VokraError::NotImplemented`] otherwise (real forward not yet
    ///   bound — FR-EX-08).
    pub fn transcribe(&self, pcm: &[f32]) -> Result<Vec<u32>> {
        if pcm.is_empty() {
            return Err(VokraError::InvalidArgument(
                "omniasr-ctc transcribe: pcm slice is empty".to_owned(),
            ));
        }
        if self.weights.is_synthesized {
            return Err(VokraError::NotImplemented(
                "omniasr-ctc transcribe: this engine holds synthesized weights \
                 (deterministic fixture from OmniasrCtcWeights::synthesized) — \
                 synthesized-weight text would be a hallucinated sequence, \
                 not a real transcript. Bind real omniASR-CTC-1B weights \
                 (Apache-2.0, facebook/omniASR-CTC-1B) before invoking \
                 transcribe. The shape flow (config validation, weight-store \
                 construction, PCM boundary check) is exercised through \
                 OmniasrCtcAsr::new; the real-checkpoint tensor-name manifest \
                 lands in a follow-up wave (T29-equivalent — the Moshi / CSM / \
                 Zonos / Kyutai STT / Parakeet-CTC pattern).",
            ));
        }
        Err(VokraError::NotImplemented(
            "omniasr-ctc transcribe: real weights are bound but the 16 kHz \
             waveform → wav2vec 2.0 waveform-Conv1D feature extractor → \
             feature projection → grouped-Conv1D positional encoder → \
             48-layer pre-norm Transformer encoder → CTC vocab head → \
             ctc_decode_greedy(blank_id = 0) → SentencePiece detokenize \
             forward path has not landed yet. Follow-up wave: extract a \
             shared vokra_ops::wav2vec2_encoder op (also usable for the \
             paired omniASR-W2V / omniASR-LLM sizes and the \
             jonatasgrosman/wav2vec2 family) or keep the encoder inline and \
             route only vokra_ops::ctc_decode_greedy with blank_id = \
             head.blank_id() (= head.blank_id, fairseq2 default 0).",
        ))
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
        // Feature projection: LayerNorm skipped (large_lv60k;
        // layer_norm_features=false); Linear feat_dim → d_enc.
        assert!(w1.feature_projection.norm_gamma.is_none());
        assert!(w1.feature_projection.norm_beta.is_none());
        assert_eq!(w1.feature_projection.linear_w.len(), feat_dim * d_enc);
        assert_eq!(w1.feature_projection.linear_b.len(), d_enc);
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

    /// Flipping `layer_norm_features` to true (the base wav2vec 2.0
    /// arch — not large_lv60k / omniASR) must produce a Some(feat_dim)
    /// pair on the feature projection LayerNorm and the runtime must
    /// accept it.
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
        OmniasrCtcAsr::new(c, w).expect("layer_norm_features=true variant is loadable");
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
            "base arch: outer post-extraction LayerNorm present (distinct from large_lv60k = false)"
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
    // Wave 5: `from_gguf` loud-partial contract (real config validation,
    // arch + provenance surface, license class round-trip, engine
    // constructibility from GGUF, `transcribe` still loud-partials on the
    // synthesized-weight blocker so a follow-up wave has exactly one place
    // to walk — mirror of MT3 / SNAC / vocos / bigvgan / parakeet_ctc
    // Wave 4 precedent).
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
        GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse")
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
        let round_trip = OmniasrCtcConfig::from_gguf(&file).expect("valid GGUF must bind");
        assert_eq!(
            round_trip, cfg,
            "every field of the resolved config must round-trip verbatim"
        );
    }

    /// A GGUF carrying `vokra.provenance.weight_license = "permissive"`
    /// (the Apache-2.0 class the omniASR-CTC converter stamps) surfaces
    /// back through `Self::weight_license()` — the outer M2-13 gate can
    /// then enforce.
    #[test]
    fn from_gguf_surfaces_stamped_permissive() {
        let cfg = OmniasrCtcConfig::omniasr_ctc_1b();
        let file = omniasr_ctc_gguf(&cfg, Some(LicenseClass::Permissive));
        let asr = OmniasrCtcAsr::from_gguf(&file).expect("valid GGUF must bind");
        assert_eq!(
            asr.weight_license(),
            LicenseClass::Permissive,
            "Apache-2.0 = Permissive must surface (distinct from Parakeet-CTC's \
             CC-BY 4.0 = AttributionRequired default)"
        );
    }

    /// A GGUF that omits `vokra.provenance.weight_license` reads back
    /// as `LicenseClass::Unknown` (fail-closed at the outer M2-13 gate,
    /// matching MT3 / SNAC / Parakeet-CTC precedent). Distinct from the
    /// `Self::new` default of `Permissive` — `from_gguf` never assumes
    /// the class, it only surfaces what the GGUF stamps.
    #[test]
    fn from_gguf_defaults_missing_provenance_to_unknown() {
        let cfg = OmniasrCtcConfig::omniasr_ctc_1b();
        let file = omniasr_ctc_gguf(&cfg, None);
        let asr = OmniasrCtcAsr::from_gguf(&file).expect("valid GGUF must bind");
        assert_eq!(
            asr.weight_license(),
            LicenseClass::Unknown,
            "missing provenance must default to Unknown (fail-closed at outer gate)"
        );
    }

    /// After a full omniASR-CTC-1B GGUF round-trip, `transcribe` still
    /// returns `NotImplemented` naming the synthesized-weight blocker
    /// (loud-partial contract preserved — the follow-up wave binds real
    /// HF checkpoint tensor names via `tools/parity/
    /// omniasr_ctc_prepare_checkpoint.py`, T29-equivalent).
    #[test]
    fn from_gguf_engine_transcribe_is_loud_not_implemented() {
        let cfg = OmniasrCtcConfig::omniasr_ctc_1b();
        let file = omniasr_ctc_gguf(&cfg, Some(LicenseClass::Permissive));
        let asr = OmniasrCtcAsr::from_gguf(&file).expect("valid GGUF must bind");
        // Round-tripped engine is still synthesized-weight; the primary
        // NotImplemented path fires (not the "real weights bound but
        // forward path not landed" path — because the follow-up wave
        // will replace the synthesized weights with real ones and flip
        // the message).
        assert!(asr.is_synthesized(), "from_gguf builds synthesized weights");

        // 1 second of 16 kHz mono silence — legitimate input shape, so
        // the loud-partial gate fires (not the empty-pcm gate).
        let pcm = vec![0.0f32; 16_000];
        let err = asr.transcribe(&pcm).unwrap_err();
        match err {
            VokraError::NotImplemented(msg) => {
                assert!(
                    msg.contains("synthesized")
                        && (msg.contains("facebook/omniASR-CTC-1B")
                            || msg.contains("real omniASR-CTC-1B")),
                    "message must name the synthesized-weight blocker + primary-source anchor: {msg}"
                );
            }
            other => panic!("expected NotImplemented, got {other:?}"),
        }
    }
}
