//! NVIDIA **Canary-1B-v2** — FastConformer + Transformer AED multi-task
//! multilingual ASR / AST (SoTA plan Phase 2, 2026-07-24).
//!
//! # What Canary-1B-v2 is (primary source)
//!
//! Canary-1B-v2 is NVIDIA NeMo's 978 M-parameter FastConformer encoder plus a
//! Transformer decoder for **multi-task, multilingual** speech recognition +
//! translation across 25 primarily European languages. Unlike Parakeet-CTC /
//! Parakeet-TDT (which are English CTC / RNN-T ASR only), Canary uses an
//! **attention encoder-decoder** (AED): the decoder is a stack of pre-norm
//! self-attn + cross-attn (to encoder) + FFN blocks driven by a **prompt
//! prefix** that carries task-specific tokens (`<source_lang>`,
//! `<target_lang>`, `<taskname>`, `<pnc>`, `<itn>`, `<timestamp>`,
//! `<diarize>`, `<emotion>`), and decoding uses a standard beam search over
//! the vocabulary head (the shared `vokra_core::decode::beam_search` primitive —
//! same op class Whisper and Voxtral consume). No new op is introduced by
//! Canary; the encoder body reuses [`vokra_ops::conformer`] via the shared
//! `Stacking { factor: 8 }` variant, exactly like Parakeet.
//!
//! Every hparam below is transcribed **verbatim** from the primary sources
//! (CLAUDE.md「ハルシネーション厳禁」):
//!
//! ## Primary sources
//!
//! - `huggingface.co/nvidia/canary-1b-v2` model card (fetched 2026-07-24):
//!   - **Encoder**: FastConformer, **32 layers**.
//!   - **Decoder**: Transformer, **8 layers**.
//!   - **Total params**: **978 million**.
//!   - **Tokenizer**: unified SentencePiece, **vocab_size = 16 384**.
//!   - **Sample rate**: **16 kHz**, monochannel .wav / .flac.
//!   - **License**: **CC-BY-4.0**.
//!   - **Task tokens**: `<source_lang>` + `<target_lang>` + task tokens
//!     (transcribe / translate).
//! - The Canary-1B-v2 checkpoint is distributed as a `.nemo` tarball
//!   (`canary-1b-v2.nemo`, 6.36 GB) that carries an internal
//!   `model_config.yaml`; the tarball is not extracted at conversion time in
//!   this scaffold, so the encoder / decoder / preprocessor hparams that are
//!   **not** stated on the model card are transcribed from the **published
//!   FastConformer-Transformer AED reference config**, which the whole Canary
//!   family shares:
//!   `github.com/NVIDIA-NeMo/Speech/blob/main/examples/asr/conf/speech_multitask/fast-conformer_aed.yaml`
//!   (fetched 2026-07-24). The header's "Canary model family" table
//!   distinguishes only the four axes the model card also names — encoder
//!   `n_layers`, decoder `num_layers`, `max_sequence_length`, and the
//!   `model_defaults.asr_enc_hidden` / `.lm_dec_hidden` widths — and the
//!   width column is `1024` for every Canary variant recorded there
//!   (`canary-1b`, `canary-1b-flash`, `canary-180m-flash`). Every other axis
//!   the reference sets (`d_model=1024`, `n_heads=8`, `ff_expansion_factor=4`
//!   → `ffn_dim=4096`, `conv_kernel_size=9`, `subsampling_factor=8`,
//!   `subsampling_conv_channels=256`, `num_mel_bins (feat_in) =128`,
//!   `pos_emb_max_len=5000`, `hidden_act="relu"`, `pre_ln=true`,
//!   decoder `inner_size = 4 * lm_dec_hidden = 4096`,
//!   `hidden_size = 1024`, decoder `num_attention_heads = 8`,
//!   preprocessor `n_fft=512`, `features=128`) is taken from that reference,
//!   marked as **family default** on the field. The `.nemo` tarball's own
//!   `model_config.yaml` is the ultimate authority; a follow-up wave
//!   (T29-equivalent) inspects it and updates the transcribed constants if
//!   any diverge — the runtime shape gate ([`CanaryConfig::validate_for_forward`])
//!   catches a divergence loudly on load (FR-EX-08 — never a silent widen).
//!
//! # Decoder max_sequence_length
//!
//! The FastConformer-Transformer AED reference table lists variant-specific
//! `max_sequence_length`: `canary-1b` = 512, `canary-1b-flash` = 1024,
//! `canary-180m-flash` = 1024. Canary-1B-v2 is a long-form multitask model
//! and the two **flash** variants (the direct architectural predecessors of
//! v2 for decoder depth) both use 1024; this scaffold adopts **1024** as the
//! anchor. The runtime accepts this until the `.nemo` extraction confirms
//! the exact value — a divergent `.nemo` config surfaces through the shape
//! validator, not through silent extrapolation.
//!
//! # Boundary — Conformer + beam-search ops consumed, never re-implemented
//!
//! Canary reuses two shared Vokra primitives instead of duplicating them:
//!
//! - **Encoder body**: [`vokra_ops::conformer`] — the Conformer /
//!   FastConformer encoder covers Canary via
//!   `ConvSubsampleKind::Stacking { factor: 8 }` (matches
//!   `subsampling_factor=8`). Same primitive Parakeet uses.
//! - **Decoder search (OP-3)**: `vokra_core::decode::beam_search` — the beam
//!   search / length-normalisation / early-stopping / n-best surface
//!   Whisper and Voxtral already consume. Canary reuses it; no per-model
//!   decoder primitive is introduced.
//!
//! # What lands in this Phase 2 slice
//!
//! - [`CanaryConfig`] — every hparam transcribed from the primary sources,
//!   with per-field docstrings naming which source the value comes from
//!   (`model card` vs `family default (fast-conformer_aed.yaml)` vs
//!   `family convention (flash max_sequence_length)`).
//! - [`CanaryWeights`] — a scaffold weight store with a deterministic
//!   [`CanaryWeights::synthesized`] fixture (SplitMix64 + Xavier) so
//!   shape / dtype / size flow can be exercised without the real
//!   `.nemo` checkpoint. Real-checkpoint parity is a follow-up wave gated
//!   on the `.nemo` extraction (T29-equivalent — Moshi / CSM / Zonos /
//!   Kyutai STT / Parakeet-TDT pattern).
//! - [`CanaryAsr`] — engine handle carrying config + weights.
//!   [`CanaryAsr::transcribe`] returns [`VokraError::NotImplemented`] until
//!   real weights are bound (the real forward — 128-bin log-mel →
//!   FastConformer encoder → task-prompt-prefixed decoder →
//!   `beam_search` → SentencePiece detokenize — is a follow-up wave gated
//!   on the `.nemo` tokenizer + tensor-name manifest).
//!
//! # No ONNX (permanent)
//!
//! Canary ships as a `.nemo` tarball / PyTorch pipeline; the pipeline is
//! re-implemented natively in `vokra-models/src/canary/` (whisper.cpp 型,
//! CLAUDE.md 設計判断 4). This module never touches ONNX.

use vokra_core::gguf::{GgufFile, chunks};
use vokra_core::rng::SplitMix64;
use vokra_core::{CompliancePolicy, LicenseClass, Result, VokraError, check_weight_license};

/// `vokra.model.arch` a Canary GGUF must carry. Written by
/// `vokra-convert::models::canary::ARCH`; the compliance registry
/// (`vokra_core::compliance`) knows `canary` / `canary-1b-v2` / `canary-1b`
/// / `canary-1b-flash` / `canary-180m-flash` as
/// [`vokra_core::LicenseClass::AttributionRequired`] (CC-BY 4.0 — the M2-13
/// gate passes commercially *and* the FR-MD-09 attribution surface
/// activates).
pub const EXPECTED_ARCH: &str = "canary";

/// PCM sample rate Canary expects — **16 000 Hz**. Model card:
/// "16kHz Audio, .wav and .flac audio formats, Monochannel audio".
pub const CANARY_SAMPLE_RATE: u32 = 16_000;

/// Deterministic seed [`CanaryAsr::from_gguf_with_policy`] threads into
/// [`CanaryWeights::synthesized`] until the real-checkpoint tensor-name
/// manifest lands (T29-equivalent — the CSM
/// [`CSM_FROM_GGUF_DEFAULT_SEED`](super::csm::CSM_FROM_GGUF_DEFAULT_SEED) /
/// [`KYUTAI_STT_FROM_GGUF_DEFAULT_SEED`](super::kyutai_stt::KYUTAI_STT_FROM_GGUF_DEFAULT_SEED)
/// / Parakeet-CTC precedent). Fixed so every `from_gguf` build against the
/// same shape config produces bit-identical weight bytes → reproducible bug
/// reports.
pub const CANARY_FROM_GGUF_DEFAULT_SEED: u64 = 0xCA5A_1B72_CA5A_1B72;

// ---------------------------------------------------------------------------
// `vokra.canary.*` chunk-key mirrors — duplicated verbatim from the
// converter (`crates/vokra-convert/src/models/canary.rs`) so
// `vokra-models` does not gain a dependency edge onto `vokra-convert`.
// This is the same layered-convention rule sibling ASR binders
// (`parakeet-ctc` / `kyutai-stt` / `mt3` / `snac` / `vocos` / `bigvgan`)
// use.
//
// Booleans (`attention_bias`, `convolution_bias`, `scale_input`, `pre_ln`)
// are stamped by the converter as `u32` via `u32::from(bool)` (0 / 1); the
// read side inverts with `!= 0`. `hidden_act` rides as a string.
// ---------------------------------------------------------------------------

const KEY_SAMPLE_RATE: &str = "vokra.canary.sample_rate";

// Encoder (FastConformer)
const KEY_ENC_N_LAYER: &str = "vokra.canary.arch.encoder.n_layer";
const KEY_ENC_D_MODEL: &str = "vokra.canary.arch.encoder.d_model";
const KEY_ENC_N_HEAD: &str = "vokra.canary.arch.encoder.n_head";
const KEY_ENC_N_HEAD_KV: &str = "vokra.canary.arch.encoder.n_head_kv";
const KEY_ENC_FFN_DIM: &str = "vokra.canary.arch.encoder.ffn_dim";
const KEY_ENC_CONV_KERNEL: &str = "vokra.canary.arch.encoder.conv_kernel_size";
const KEY_ENC_IN_DIM: &str = "vokra.canary.arch.encoder.in_dim";
const KEY_ENC_SUBSAMPLING_FACTOR: &str = "vokra.canary.arch.encoder.subsampling_factor";
const KEY_ENC_SUB_CONV_KERNEL: &str = "vokra.canary.arch.encoder.subsampling_conv_kernel_size";
const KEY_ENC_SUB_CONV_STRIDE: &str = "vokra.canary.arch.encoder.subsampling_conv_stride";
const KEY_ENC_SUB_CONV_CHANNELS: &str = "vokra.canary.arch.encoder.subsampling_conv_channels";
const KEY_ENC_MAX_POS: &str = "vokra.canary.arch.encoder.max_position_embeddings";
const KEY_ENC_ATTN_BIAS: &str = "vokra.canary.arch.encoder.attention_bias";
const KEY_ENC_CONV_BIAS: &str = "vokra.canary.arch.encoder.convolution_bias";
const KEY_ENC_SCALE_INPUT: &str = "vokra.canary.arch.encoder.scale_input";

// Decoder (Transformer AED)
const KEY_DEC_N_LAYER: &str = "vokra.canary.arch.decoder.n_layer";
const KEY_DEC_D_MODEL: &str = "vokra.canary.arch.decoder.d_model";
const KEY_DEC_N_HEAD: &str = "vokra.canary.arch.decoder.n_head";
const KEY_DEC_FFN_DIM: &str = "vokra.canary.arch.decoder.ffn_dim";
const KEY_DEC_MAX_SEQ: &str = "vokra.canary.arch.decoder.max_sequence_length";
const KEY_DEC_PRE_LN: &str = "vokra.canary.arch.decoder.pre_ln";
const KEY_DEC_HIDDEN_ACT: &str = "vokra.canary.arch.decoder.hidden_act";

// Head + vocab
const KEY_HEAD_VOCAB_SIZE: &str = "vokra.canary.head.vocab_size";
const KEY_HEAD_PAD_ID: &str = "vokra.canary.head.pad_token_id";
const KEY_HEAD_BOS_ID: &str = "vokra.canary.head.bos_token_id";
const KEY_HEAD_EOS_ID: &str = "vokra.canary.head.eos_token_id";

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// FastConformer encoder hparams for Canary-1B-v2.
///
/// The encoder is a stack of pre-norm Conformer blocks with 8× subsampling
/// on the input (the "Fast" in FastConformer). `d_model` is the residual
/// width; the per-head width is `d_model / num_attention_heads`.
#[derive(Debug, Clone, PartialEq)]
pub struct CanaryEncoderConfig {
    /// `num_hidden_layers` — **32 for Canary-1B-v2** (primary source:
    /// model card — "FastConformer Encoder: 32 encoder layers").
    pub n_layer: usize,
    /// `hidden_size` — hidden width, **1024** (family default per the
    /// `fast-conformer_aed.yaml` `model_defaults.asr_enc_hidden` column
    /// for every recorded Canary variant).
    pub d_model: usize,
    /// `num_attention_heads` — Q-heads, **8** (family default —
    /// `encoder.n_heads: 8`).
    pub n_head: usize,
    /// `num_key_value_heads` — KV-heads; **8 for MHA** (family default —
    /// `self_attention_model: rel_pos` uses MHA, no GQA broadcast).
    /// Kept as a field so a hypothetical future GQA flavor is
    /// representable without a new type.
    pub n_head_kv: usize,
    /// `intermediate_size` — FFN inner width, **4096** (family default:
    /// `ff_expansion_factor: 4` × `d_model=1024`).
    pub ffn_dim: usize,
    /// `conv_kernel_size` — FastConformer depthwise convolution kernel
    /// size, **9** (family default — `conv_kernel_size: 9`). Must be odd
    /// for symmetric same-padding.
    pub conv_kernel_size: usize,
    /// `num_mel_bins` — log-mel channels on the input, **128** (family
    /// default — `preprocessor.features: 128` and `encoder.feat_in:
    /// ${model.preprocessor.features}` in the reference yaml).
    pub in_dim: usize,
    /// `subsampling_factor` — **8** (family default — `subsampling_factor:
    /// 8`).
    pub subsampling_factor: usize,
    /// `subsampling_conv_kernel_size` — **3** (family default — reference
    /// yaml specifies `subsampling: dw_striding` whose canonical NeMo
    /// implementation uses stride-2 kernel-3 depth-wise conv stages).
    pub subsampling_conv_kernel_size: usize,
    /// `subsampling_conv_stride` — **2** (family default — the same
    /// stride-2 stages from `subsampling: dw_striding`; three stride-2
    /// stages compose the total `subsampling_factor: 8`).
    pub subsampling_conv_stride: usize,
    /// `subsampling_conv_channels` — **256** (family default —
    /// `subsampling_conv_channels: 256`).
    pub subsampling_conv_channels: usize,
    /// `max_position_embeddings` — **5000** (family default —
    /// `pos_emb_max_len: 5000`). Upper bound on the RoPE / rel-pos index;
    /// a real forward asserts the incoming subsampled sequence length
    /// does not exceed it.
    pub max_position_embeddings: usize,
    /// `attention_bias` — **true** (family default — the reference yaml
    /// uses `untie_biases: true` for the rel-pos MHA, so Q/K/V/out
    /// projections carry biases).
    pub attention_bias: bool,
    /// `convolution_bias` — **false** (family default — the FastConformer
    /// depthwise / point-wise convolutions are bias-free; the reference
    /// yaml records `conv_norm_type: batch_norm` and does not add a
    /// convolution bias flag).
    pub convolution_bias: bool,
    /// `xscaling` — **false** (family default — the reference yaml
    /// records `xscaling: false`, i.e. the subsample stem does *not*
    /// scale by `sqrt(d_model)`).
    pub scale_input: bool,
}

impl CanaryEncoderConfig {
    /// Per-head width (`d_model / n_head`); `0` when `n_head == 0`
    /// (shape-only converter sentinel) so shape checks never panic.
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.d_model.checked_div(self.n_head).unwrap_or(0)
    }

    /// MHA / GQA algebraic constraint: Q-heads divide the width, and
    /// KV-heads divide Q-heads (broadcast). All non-zero.
    #[must_use]
    pub fn is_well_formed(&self) -> bool {
        self.n_head != 0
            && self.n_head_kv != 0
            && self.d_model != 0
            && self.d_model % self.n_head == 0
            && self.n_head % self.n_head_kv == 0
    }

    /// KV hidden width, `n_head_kv * head_dim`. Equals `d_model` when
    /// `n_head_kv == n_head` (MHA — the Canary-1B-v2 case).
    #[must_use]
    pub fn kv_hidden(&self) -> usize {
        self.n_head_kv * self.head_dim()
    }
}

/// Transformer decoder hparams (AED = attention encoder-decoder).
///
/// The decoder is a stack of pre-norm blocks: **self-attention** (masked
/// causal) over the emitted / prompt tokens, **cross-attention** to the
/// FastConformer encoder output, and a **feed-forward** module. Cross-attn
/// K/V comes from the encoder-out sequence at every step, so the decoder
/// carries a separate K/V projection for cross-attention (Q comes from the
/// decoder-side stream).
#[derive(Debug, Clone, PartialEq)]
pub struct CanaryDecoderConfig {
    /// `num_layers` — **8 for Canary-1B-v2** (primary source: model
    /// card — "Transformer Decoder: 8 decoder layers").
    pub n_layer: usize,
    /// `hidden_size` — decoder width, **1024** (family default —
    /// `model_defaults.lm_dec_hidden: 1024` for every recorded Canary
    /// variant).
    pub d_model: usize,
    /// `num_attention_heads` — Q-heads, **8** (family default —
    /// `transf_decoder.config_dict.num_attention_heads: 8`).
    pub n_head: usize,
    /// `inner_size` — FFN inner width, **4096** (family default —
    /// `transf_decoder.config_dict.inner_size: ${multiply:${model.model_defaults.lm_dec_hidden}, 4}` = 4 × 1024).
    pub ffn_dim: usize,
    /// `max_sequence_length` — **1024** (family convention — the two
    /// flash Canary variants both use 1024 in the reference table; the
    /// non-flash v1 uses 512. Canary-1B-v2 is a long-form multitask
    /// model; the `.nemo` tarball's `model_config.yaml` is ultimately
    /// authoritative, and a follow-up wave verifies against the extracted
    /// config. The runtime accepts the anchored 1024 until then).
    pub max_sequence_length: usize,
    /// `pre_ln` — **true** (family default — `pre_ln: true`).
    pub pre_ln: bool,
    /// `hidden_act` — **"relu"** (family default —
    /// `transf_decoder.config_dict.hidden_act: relu`; recorded verbatim as
    /// descriptive metadata — the runtime FFN implementation reads this
    /// string to pick the elementwise activation).
    pub hidden_act: String,
}

/// Vocabulary / prompt / head hparams (primary source: model card + the
/// Canary prompt-format contract that the tokenizer prefaces every decoder
/// input with).
#[derive(Debug, Clone, PartialEq)]
pub struct CanaryHeadConfig {
    /// `vocab_size` — **16 384** (primary source: model card —
    /// "unified SentencePiece Tokenizer with a vocabulary of 16,384
    /// tokens, optimized across all 25 supported languages"). The head
    /// therefore has output width `vocab_size` (task tokens inclusive —
    /// the special tokens `<source_lang>`, `<target_lang>`, `<taskname>`,
    /// `<pnc>`, `<itn>`, `<timestamp>`, `<diarize>`, `<emotion>` all live
    /// in this vocabulary alongside the SentencePiece BPE pieces).
    pub vocab_size: usize,
    /// `pad_token_id` — placeholder-sentinel index reserved for the
    /// tokenizer's pad token. The `.nemo` extraction sets the real value
    /// (upstream NeMo `EncDecMultiTaskModel` defaults to the tokenizer's
    /// own pad; the model card does not enumerate a specific integer).
    /// `0` here means "the tokenizer will bind this at load"; the runtime
    /// validator rejects a beam search whose blank / pad exceeds
    /// `vocab_size`.
    pub pad_token_id: u32,
    /// `bos_token_id` — placeholder-sentinel for the decoder's beginning
    /// of sequence token (Canary emits its prompt prefix through this
    /// token; the `.nemo` extraction sets the real value).
    pub bos_token_id: u32,
    /// `eos_token_id` — placeholder-sentinel for the decoder's end of
    /// sequence token (beam search halts on this; the `.nemo` extraction
    /// sets the real value).
    pub eos_token_id: u32,
}

/// Resolved Canary hparam snapshot — every field is transcribed from
/// primary sources (model card + family reference config).
#[derive(Debug, Clone, PartialEq)]
pub struct CanaryConfig {
    /// FastConformer encoder hparams.
    pub encoder: CanaryEncoderConfig,
    /// Transformer decoder hparams (AED).
    pub decoder: CanaryDecoderConfig,
    /// Vocabulary + special-token / head hparams.
    pub head: CanaryHeadConfig,
    /// PCM sample rate Canary expects — **16 000 Hz** (from the model
    /// card).
    pub sample_rate: u32,
}

impl CanaryConfig {
    /// Primary-source Canary-1B-v2 config. Every value is transcribed
    /// from the model card (`n_layer`, `vocab_size`, `sample_rate`,
    /// `license`) or the shared FastConformer-Transformer AED reference
    /// (all other axes). The `.nemo` tarball's `model_config.yaml` is the
    /// ultimate authority; a follow-up wave (T29-equivalent) inspects it
    /// and updates any transcribed constants that diverge.
    #[must_use]
    pub fn canary_1b_v2() -> Self {
        Self {
            encoder: CanaryEncoderConfig {
                n_layer: 32,
                d_model: 1024,
                n_head: 8,
                n_head_kv: 8,
                ffn_dim: 4096,
                conv_kernel_size: 9,
                in_dim: 128,
                subsampling_factor: 8,
                subsampling_conv_kernel_size: 3,
                subsampling_conv_stride: 2,
                subsampling_conv_channels: 256,
                max_position_embeddings: 5000,
                attention_bias: true,
                convolution_bias: false,
                scale_input: false,
            },
            decoder: CanaryDecoderConfig {
                n_layer: 8,
                d_model: 1024,
                n_head: 8,
                ffn_dim: 4096,
                max_sequence_length: 1024,
                pre_ln: true,
                hidden_act: "relu".to_owned(),
            },
            head: CanaryHeadConfig {
                vocab_size: 16_384,
                pad_token_id: 0,
                bos_token_id: 0,
                eos_token_id: 0,
            },
            sample_rate: CANARY_SAMPLE_RATE,
        }
    }

    /// Miniature well-formed config for shape / stability tests. Dims are
    /// tiny so synthesized-weight builds fit in KB; the *shape
    /// relationships* (MHA head split, even head_dim, encoder / decoder
    /// widths, cross-attn Q from decoder + K/V from encoder) mirror the
    /// real model.
    #[must_use]
    pub fn tiny_for_tests() -> Self {
        Self {
            encoder: CanaryEncoderConfig {
                n_layer: 2,
                d_model: 16,
                n_head: 4,
                n_head_kv: 4,
                ffn_dim: 32,
                conv_kernel_size: 3,
                in_dim: 8,
                subsampling_factor: 2,
                subsampling_conv_kernel_size: 3,
                subsampling_conv_stride: 2,
                subsampling_conv_channels: 16,
                max_position_embeddings: 128,
                attention_bias: true,
                convolution_bias: false,
                scale_input: false,
            },
            decoder: CanaryDecoderConfig {
                n_layer: 2,
                d_model: 16,
                n_head: 4,
                ffn_dim: 32,
                max_sequence_length: 64,
                pre_ln: true,
                hidden_act: "relu".to_owned(),
            },
            head: CanaryHeadConfig {
                vocab_size: 32,
                pad_token_id: 0,
                bos_token_id: 1,
                eos_token_id: 2,
            },
            sample_rate: CANARY_SAMPLE_RATE,
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
                "canary config: encoder ill-formed \
                 (n_layer={}, d_model={}, n_head={}, n_head_kv={}) — \
                 expected d_model % n_head == 0, n_head % n_head_kv == 0, \
                 all fields > 0",
                self.encoder.n_layer,
                self.encoder.d_model,
                self.encoder.n_head,
                self.encoder.n_head_kv,
            )));
        }
        if self.encoder.n_layer == 0 {
            return Err(VokraError::InvalidArgument(
                "canary config: encoder.n_layer must be > 0".to_owned(),
            ));
        }
        if self.encoder.head_dim() % 2 != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "canary config: encoder head_dim {} must be even \
                 (RoPE / rel-pos pairs)",
                self.encoder.head_dim(),
            )));
        }
        if self.encoder.ffn_dim == 0 {
            return Err(VokraError::InvalidArgument(
                "canary config: encoder.ffn_dim must be > 0".to_owned(),
            ));
        }
        if self.encoder.in_dim == 0 {
            return Err(VokraError::InvalidArgument(
                "canary config: encoder.in_dim (num_mel_bins) must be > 0".to_owned(),
            ));
        }
        if self.encoder.conv_kernel_size == 0 || self.encoder.conv_kernel_size % 2 == 0 {
            return Err(VokraError::InvalidArgument(format!(
                "canary config: encoder.conv_kernel_size {} must be odd and > 0 \
                 (Conformer symmetric same-padding)",
                self.encoder.conv_kernel_size,
            )));
        }
        if self.encoder.subsampling_factor == 0 {
            return Err(VokraError::InvalidArgument(
                "canary config: encoder.subsampling_factor must be > 0 \
                 (FastConformer subsampling)"
                    .to_owned(),
            ));
        }
        if self.encoder.max_position_embeddings == 0 {
            return Err(VokraError::InvalidArgument(
                "canary config: encoder.max_position_embeddings must be > 0".to_owned(),
            ));
        }

        // ---- Decoder ------------------------------------------------------
        if self.decoder.n_layer == 0 || self.decoder.d_model == 0 || self.decoder.n_head == 0 {
            return Err(VokraError::InvalidArgument(format!(
                "canary config: decoder ill-formed \
                 (n_layer={}, d_model={}, n_head={}) — all fields > 0",
                self.decoder.n_layer, self.decoder.d_model, self.decoder.n_head,
            )));
        }
        if self.decoder.d_model % self.decoder.n_head != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "canary config: decoder d_model={} not divisible by n_head={}",
                self.decoder.d_model, self.decoder.n_head,
            )));
        }
        // Cross-attention keys come from the encoder-out sequence, which is
        // `encoder.d_model`-wide; the decoder's cross-attn K/V projections
        // must therefore project from `encoder.d_model` (not `decoder.d_model`).
        // We do not require the two widths to match — the runtime forward
        // handles the linear cross-projection. But we do require the decoder
        // head split to divide the decoder width evenly.
        let dec_head_dim = self.decoder.d_model / self.decoder.n_head;
        if dec_head_dim == 0 || dec_head_dim % 2 != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "canary config: decoder head_dim {} must be even and > 0 \
                 (positional embedding pairs)",
                dec_head_dim,
            )));
        }
        if self.decoder.ffn_dim == 0 {
            return Err(VokraError::InvalidArgument(
                "canary config: decoder.ffn_dim must be > 0".to_owned(),
            ));
        }
        if self.decoder.max_sequence_length == 0 {
            return Err(VokraError::InvalidArgument(
                "canary config: decoder.max_sequence_length must be > 0".to_owned(),
            ));
        }
        if self.decoder.hidden_act.is_empty() {
            return Err(VokraError::InvalidArgument(
                "canary config: decoder.hidden_act must be non-empty \
                 (e.g. \"relu\")"
                    .to_owned(),
            ));
        }

        // ---- Head / vocab -------------------------------------------------
        if self.head.vocab_size == 0 {
            return Err(VokraError::InvalidArgument(
                "canary config: head.vocab_size must be > 0".to_owned(),
            ));
        }
        // pad / bos / eos ids must live inside the vocab head width. `0` is a
        // legal id on the model card / .nemo default (the tokenizer's pad is
        // typically 0). The validator only rejects an id that exceeds the
        // head width — a real forward would index out of bounds.
        if (self.head.pad_token_id as usize) >= self.head.vocab_size {
            return Err(VokraError::InvalidArgument(format!(
                "canary config: pad_token_id={} must be < vocab_size={}",
                self.head.pad_token_id, self.head.vocab_size,
            )));
        }
        if (self.head.bos_token_id as usize) >= self.head.vocab_size {
            return Err(VokraError::InvalidArgument(format!(
                "canary config: bos_token_id={} must be < vocab_size={}",
                self.head.bos_token_id, self.head.vocab_size,
            )));
        }
        if (self.head.eos_token_id as usize) >= self.head.vocab_size {
            return Err(VokraError::InvalidArgument(format!(
                "canary config: eos_token_id={} must be < vocab_size={}",
                self.head.eos_token_id, self.head.vocab_size,
            )));
        }
        Ok(())
    }

    /// Reads every `vokra.canary.*` chunk from `gguf` (strict).
    ///
    /// Missing axis = loud [`VokraError::ModelLoad`] naming the absent
    /// key (FR-EX-08 — no primary-source constant fallback because a
    /// converter that fails to stamp an axis is a converter bug, not a
    /// runtime silent-default).
    ///
    /// Primary source for the axis table: `huggingface.co/nvidia/canary-1b-v2`
    /// (fetched 2026-07-24 by the converter, transcribed verbatim into
    /// [`Self::canary_1b_v2`]).
    ///
    /// Booleans (`attention_bias`, `convolution_bias`, `scale_input`,
    /// `pre_ln`) are stamped by the converter as u32 (0 / 1); this
    /// reader inverts back to `bool` with `!= 0`, mirroring the
    /// Parakeet-CTC / Zonos / CSM / Kyutai STT convention. `hidden_act`
    /// rides as a string.
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when any mandatory `vokra.canary.*`
    ///   chunk is absent (numeric axes ride as `u32`; `hidden_act` rides
    ///   as a string).
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        fn req_u32(gguf: &GgufFile, key: &str) -> Result<u32> {
            gguf.get(key)
                .and_then(vokra_core::gguf::GgufMetadataValue::as_u64)
                .map(|v| v as u32)
                .ok_or_else(|| {
                    VokraError::ModelLoad(format!(
                        "canary: GGUF is missing required u32 chunk `{key}` — the \
                         upstream `nvidia/canary-1b-v2` model card + family reference \
                         yaml (fetched 2026-07-24) supply every FastConformer + \
                         Transformer AED axis; a converter that fails to stamp one \
                         is a converter bug, not a runtime silent-default \
                         (FR-EX-08). Re-run `vokra-cli convert --model canary` \
                         against `nvidia/canary-1b-v2` safetensors. Primary source: \
                         https://huggingface.co/nvidia/canary-1b-v2"
                    ))
                })
        }
        fn req_str<'a>(gguf: &'a GgufFile, key: &str) -> Result<&'a str> {
            gguf.get(key).and_then(|v| v.as_str()).ok_or_else(|| {
                VokraError::ModelLoad(format!(
                    "canary: GGUF is missing required string chunk `{key}` — the \
                     upstream reference yaml specifies `hidden_act` (\"relu\" for \
                     Canary-1B-v2); a converter that fails to stamp it is a \
                     converter bug, not a runtime silent-default (FR-EX-08). \
                     Re-run `vokra-cli convert --model canary`. Primary source: \
                     https://huggingface.co/nvidia/canary-1b-v2"
                ))
            })
        }
        Ok(Self {
            encoder: CanaryEncoderConfig {
                n_layer: req_u32(gguf, KEY_ENC_N_LAYER)? as usize,
                d_model: req_u32(gguf, KEY_ENC_D_MODEL)? as usize,
                n_head: req_u32(gguf, KEY_ENC_N_HEAD)? as usize,
                n_head_kv: req_u32(gguf, KEY_ENC_N_HEAD_KV)? as usize,
                ffn_dim: req_u32(gguf, KEY_ENC_FFN_DIM)? as usize,
                conv_kernel_size: req_u32(gguf, KEY_ENC_CONV_KERNEL)? as usize,
                in_dim: req_u32(gguf, KEY_ENC_IN_DIM)? as usize,
                subsampling_factor: req_u32(gguf, KEY_ENC_SUBSAMPLING_FACTOR)? as usize,
                subsampling_conv_kernel_size: req_u32(gguf, KEY_ENC_SUB_CONV_KERNEL)? as usize,
                subsampling_conv_stride: req_u32(gguf, KEY_ENC_SUB_CONV_STRIDE)? as usize,
                subsampling_conv_channels: req_u32(gguf, KEY_ENC_SUB_CONV_CHANNELS)? as usize,
                max_position_embeddings: req_u32(gguf, KEY_ENC_MAX_POS)? as usize,
                attention_bias: req_u32(gguf, KEY_ENC_ATTN_BIAS)? != 0,
                convolution_bias: req_u32(gguf, KEY_ENC_CONV_BIAS)? != 0,
                scale_input: req_u32(gguf, KEY_ENC_SCALE_INPUT)? != 0,
            },
            decoder: CanaryDecoderConfig {
                n_layer: req_u32(gguf, KEY_DEC_N_LAYER)? as usize,
                d_model: req_u32(gguf, KEY_DEC_D_MODEL)? as usize,
                n_head: req_u32(gguf, KEY_DEC_N_HEAD)? as usize,
                ffn_dim: req_u32(gguf, KEY_DEC_FFN_DIM)? as usize,
                max_sequence_length: req_u32(gguf, KEY_DEC_MAX_SEQ)? as usize,
                pre_ln: req_u32(gguf, KEY_DEC_PRE_LN)? != 0,
                hidden_act: req_str(gguf, KEY_DEC_HIDDEN_ACT)?.to_owned(),
            },
            head: CanaryHeadConfig {
                vocab_size: req_u32(gguf, KEY_HEAD_VOCAB_SIZE)? as usize,
                pad_token_id: req_u32(gguf, KEY_HEAD_PAD_ID)?,
                bos_token_id: req_u32(gguf, KEY_HEAD_BOS_ID)?,
                eos_token_id: req_u32(gguf, KEY_HEAD_EOS_ID)?,
            },
            sample_rate: req_u32(gguf, KEY_SAMPLE_RATE)?,
        })
    }
}

// ---------------------------------------------------------------------------
// Weights
// ---------------------------------------------------------------------------

/// Per-encoder-block scaffold weights (pre-norm Conformer FF1 / MHA / Conv
/// / FF2 branches). Same shape as the Parakeet encoder block; every Canary
/// / Parakeet FastConformer variant reuses this identical body.
///
/// Field names mirror the upstream NeMo `ConformerLayer` module names.
///
/// # Attention biases (Canary-1B-v2 specific)
///
/// Canary-1B-v2 has `attention_bias = true` (the reference yaml uses
/// `untie_biases: true` for the rel-pos MHA — every projection carries a
/// bias). The four projection biases ride as separate optional vectors so
/// a hypothetical future bias-free variant is representable without a new
/// type; `Some` on all four means the biases participate, `None` means
/// they do not. A mismatch between the config flag and the presence of
/// the vectors is a loud error at load, not a silent zero-fill.
#[derive(Debug, Clone)]
pub struct CanaryEncoderBlockWeights {
    /// FF1 pre-norm γ, shape `[d_model]`.
    pub ff1_norm: Vec<f32>,
    /// FF1 hidden projection, shape `[d_model, ffn_dim]`.
    pub ff1_fc1: Vec<f32>,
    /// FF1 output projection, shape `[ffn_dim, d_model]`.
    pub ff1_fc2: Vec<f32>,
    /// Attention pre-norm γ, shape `[d_model]`.
    pub attn_norm: Vec<f32>,
    /// Fused Q/K/V projection, shape `[d_model, 3*d_model]` (MHA).
    pub qkv_proj: Vec<f32>,
    /// Optional fused Q/K/V bias, shape `[3*d_model]`. Present iff
    /// `encoder.attention_bias == true` (the Canary-1B-v2 case).
    pub qkv_bias: Option<Vec<f32>>,
    /// Attention output projection, shape `[d_model, d_model]`.
    pub attn_out: Vec<f32>,
    /// Optional attention output bias, shape `[d_model]`. Present iff
    /// `encoder.attention_bias == true`.
    pub attn_out_bias: Option<Vec<f32>>,
    /// Conv module pre-norm γ, shape `[d_model]`.
    pub conv_norm: Vec<f32>,
    /// Conv module point-wise 1: `[d_model, 2*d_model]` (GLU pre-split).
    pub conv_pw1: Vec<f32>,
    /// Depthwise conv kernel, shape `[d_model, 1, conv_kernel_size]`.
    pub conv_dw: Vec<f32>,
    /// Depthwise LayerNorm γ, shape `[d_model]`.
    pub conv_dw_norm: Vec<f32>,
    /// Conv module point-wise 2: `[d_model, d_model]`.
    pub conv_pw2: Vec<f32>,
    /// FF2 pre-norm γ, shape `[d_model]`.
    pub ff2_norm: Vec<f32>,
    /// FF2 hidden projection, shape `[d_model, ffn_dim]`.
    pub ff2_fc1: Vec<f32>,
    /// FF2 output projection, shape `[ffn_dim, d_model]`.
    pub ff2_fc2: Vec<f32>,
    /// Final block LayerNorm γ, shape `[d_model]`.
    pub final_norm: Vec<f32>,
}

/// Subsample stem scaffold weights (a Linear + optional norm with
/// `factor = subsampling_factor` — the [`vokra_ops::conformer`]
/// `Stacking` variant). Kept flat so the sizes are trivially checkable.
#[derive(Debug, Clone)]
pub struct CanarySubsampleWeights {
    /// `[d_model, factor * in_dim]`.
    pub linear_w: Vec<f32>,
    /// `[d_model]`.
    pub linear_b: Vec<f32>,
}

/// Per-decoder-block scaffold weights (pre-norm self-attn + cross-attn +
/// FFN, AED style). Cross-attention K/V comes from the encoder-out, so
/// the K/V projection matrix is `[enc_d_model, 2 * dec_d_model]` and the
/// Q projection is `[dec_d_model, dec_d_model]`.
#[derive(Debug, Clone)]
pub struct CanaryDecoderBlockWeights {
    /// Self-attn pre-norm γ, shape `[dec_d_model]`.
    pub self_attn_norm: Vec<f32>,
    /// Fused self-attn Q/K/V projection, shape `[dec_d_model, 3*dec_d_model]`.
    pub self_attn_qkv: Vec<f32>,
    /// Self-attn Q/K/V bias, shape `[3*dec_d_model]`.
    pub self_attn_qkv_bias: Vec<f32>,
    /// Self-attn output projection, shape `[dec_d_model, dec_d_model]`.
    pub self_attn_out: Vec<f32>,
    /// Self-attn output bias, shape `[dec_d_model]`.
    pub self_attn_out_bias: Vec<f32>,
    /// Cross-attn pre-norm γ, shape `[dec_d_model]`.
    pub cross_attn_norm: Vec<f32>,
    /// Cross-attn Q projection (from decoder-side), shape
    /// `[dec_d_model, dec_d_model]`.
    pub cross_attn_q: Vec<f32>,
    /// Cross-attn Q bias, shape `[dec_d_model]`.
    pub cross_attn_q_bias: Vec<f32>,
    /// Cross-attn K/V projection (from encoder-out width), fused, shape
    /// `[enc_d_model, 2 * dec_d_model]`.
    pub cross_attn_kv: Vec<f32>,
    /// Cross-attn K/V bias, shape `[2 * dec_d_model]`.
    pub cross_attn_kv_bias: Vec<f32>,
    /// Cross-attn output projection, shape `[dec_d_model, dec_d_model]`.
    pub cross_attn_out: Vec<f32>,
    /// Cross-attn output bias, shape `[dec_d_model]`.
    pub cross_attn_out_bias: Vec<f32>,
    /// FFN pre-norm γ, shape `[dec_d_model]`.
    pub ffn_norm: Vec<f32>,
    /// FFN hidden projection, shape `[dec_d_model, ffn_dim]`.
    pub ffn_fc1: Vec<f32>,
    /// FFN hidden bias, shape `[ffn_dim]`.
    pub ffn_fc1_bias: Vec<f32>,
    /// FFN output projection, shape `[ffn_dim, dec_d_model]`.
    pub ffn_fc2: Vec<f32>,
    /// FFN output bias, shape `[dec_d_model]`.
    pub ffn_fc2_bias: Vec<f32>,
}

/// Canary weight store.
///
/// The layout mirrors the AED forward:
/// 1. Subsample stem → encoder blocks → encoder final norm (FastConformer
///    encoder produces the `[T', enc_d_model]` context sequence).
/// 2. Optional projection from `enc_d_model` to `dec_d_model` (the
///    reference yaml notes "One extra (linear projection) layer is added
///    between FastConformer encoder and Transformer decoder if they have
///    different hidden sizes" — for the standard Canary widths the two
///    are equal and the projection is the identity; we still carry the
///    scaffold tensor so a future asymmetric variant is representable).
/// 3. Decoder token embedding + positional embedding.
/// 4. Decoder blocks (self-attn + cross-attn + FFN) → decoder final norm.
/// 5. Vocab head (linear from `dec_d_model` to `vocab_size`, plus bias).
///
/// [`Self::synthesized`] builds a deterministic fixture (SplitMix64 +
/// Xavier) against `config` so shape / dtype / size can be exercised
/// without the real `.nemo` checkpoint. Real-checkpoint binding is a
/// follow-up (T29-equivalent — tensor-name manifest fetch from the
/// upstream `.nemo` extraction).
#[derive(Debug, Clone)]
pub struct CanaryWeights {
    /// Subsample stem.
    pub subsample: CanarySubsampleWeights,
    /// Encoder blocks in order.
    pub encoder_blocks: Vec<CanaryEncoderBlockWeights>,
    /// Encoder-out LayerNorm γ, shape `[enc_d_model]`.
    pub encoder_final_norm: Vec<f32>,
    /// Optional encoder→decoder width projection, shape
    /// `[enc_d_model, dec_d_model]`. For the standard Canary widths the
    /// two are equal (1024 == 1024) and this holds the identity /
    /// synthesized weight; a future asymmetric variant would carry the
    /// real projection.
    pub enc_to_dec_proj: Vec<f32>,
    /// Optional encoder→decoder projection bias, shape `[dec_d_model]`.
    pub enc_to_dec_proj_bias: Vec<f32>,
    /// Decoder token embedding, shape `[vocab_size, dec_d_model]`.
    pub dec_token_embedding: Vec<f32>,
    /// Decoder positional embedding, shape
    /// `[max_sequence_length, dec_d_model]`. The reference yaml records
    /// `learn_positional_encodings: false`, so upstream this is fixed
    /// sinusoidal; we still carry it as a scaffold vector so a future
    /// learned-position variant is representable — a fixed variant leaves
    /// this untouched at the sinusoidal values the runtime installs.
    pub dec_position_embedding: Vec<f32>,
    /// Decoder blocks in order.
    pub decoder_blocks: Vec<CanaryDecoderBlockWeights>,
    /// Decoder-out LayerNorm γ, shape `[dec_d_model]`
    /// (pre_ln_final_layer_norm = true — the reference yaml explicitly
    /// asks for a final LN after the last decoder block).
    pub decoder_final_norm: Vec<f32>,
    /// Vocab head, shape `[dec_d_model, vocab_size]`.
    pub vocab_head: Vec<f32>,
    /// Vocab head bias, shape `[vocab_size]`.
    pub vocab_bias: Vec<f32>,
    /// `true` when built by [`Self::synthesized`] — never a real upstream
    /// checkpoint. Real-checkpoint bindings set this to `false`.
    pub is_synthesized: bool,
}

impl CanaryWeights {
    /// Builds a deterministic synthesized fixture from `config` and
    /// `seed`.
    ///
    /// Draws are Xavier-uniform ± `sqrt(6 / (fan_in + fan_out))` via a
    /// [`SplitMix64`] stream — reproducible, allocation-only, zero-dep.
    /// Every LayerNorm γ starts at `1.0`; every bias starts at `0.0`.
    ///
    /// Attention biases (`qkv_bias`, `attn_out_bias`) on the encoder side
    /// are `Some` iff `config.encoder.attention_bias == true` (the
    /// Canary-1B-v2 case). The decoder always carries biases on every
    /// projection (upstream NeMo `TransformerDecoderLayer` defaults them
    /// on and the reference yaml does not turn them off).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] if `config.validate_for_forward`
    /// fails.
    pub fn synthesized(config: &CanaryConfig, seed: u64) -> Result<Self> {
        config.validate_for_forward()?;
        let mut rng = SplitMix64::new(seed);
        let enc = &config.encoder;
        let dec = &config.decoder;
        let head = &config.head;
        let d_enc = enc.d_model;
        let d_dec = dec.d_model;
        let enc_ffn = enc.ffn_dim;
        let dec_ffn = dec.ffn_dim;
        let vocab = head.vocab_size;
        let k = enc.conv_kernel_size;
        let bias_on = enc.attention_bias;

        // Subsample stem — flat Linear (Stacking variant).
        let projection_in = enc.subsampling_factor * enc.in_dim;
        let subsample = CanarySubsampleWeights {
            linear_w: xavier(&mut rng, d_enc * projection_in, projection_in, d_enc),
            linear_b: vec![0.0; d_enc],
        };

        // Encoder blocks.
        let mut encoder_blocks = Vec::with_capacity(enc.n_layer);
        for _ in 0..enc.n_layer {
            encoder_blocks.push(CanaryEncoderBlockWeights {
                ff1_norm: vec![1.0; d_enc],
                ff1_fc1: xavier(&mut rng, d_enc * enc_ffn, d_enc, enc_ffn),
                ff1_fc2: xavier(&mut rng, enc_ffn * d_enc, enc_ffn, d_enc),
                attn_norm: vec![1.0; d_enc],
                qkv_proj: xavier(&mut rng, d_enc * 3 * d_enc, d_enc, 3 * d_enc),
                qkv_bias: bias_on.then(|| vec![0.0; 3 * d_enc]),
                attn_out: xavier(&mut rng, d_enc * d_enc, d_enc, d_enc),
                attn_out_bias: bias_on.then(|| vec![0.0; d_enc]),
                conv_norm: vec![1.0; d_enc],
                conv_pw1: xavier(&mut rng, d_enc * 2 * d_enc, d_enc, 2 * d_enc),
                conv_dw: xavier(&mut rng, d_enc * k, k, 1),
                conv_dw_norm: vec![1.0; d_enc],
                conv_pw2: xavier(&mut rng, d_enc * d_enc, d_enc, d_enc),
                ff2_norm: vec![1.0; d_enc],
                ff2_fc1: xavier(&mut rng, d_enc * enc_ffn, d_enc, enc_ffn),
                ff2_fc2: xavier(&mut rng, enc_ffn * d_enc, enc_ffn, d_enc),
                final_norm: vec![1.0; d_enc],
            });
        }
        let encoder_final_norm = vec![1.0; d_enc];

        // Encoder→decoder width projection. For d_enc == d_dec the real
        // Canary release skips the extra Linear; we still carry the
        // scaffold vector (zero-filled — the runtime treats a zero-filled
        // proj as "identity, use encoder-out directly" until the real
        // asymmetric weights land).
        let enc_to_dec_proj = vec![0.0; d_enc * d_dec];
        let enc_to_dec_proj_bias = vec![0.0; d_dec];

        // Decoder token + positional embeddings.
        let dec_token_embedding = xavier(&mut rng, vocab * d_dec, vocab, d_dec);
        let dec_position_embedding = vec![0.0; dec.max_sequence_length * d_dec];

        // Decoder blocks.
        let mut decoder_blocks = Vec::with_capacity(dec.n_layer);
        for _ in 0..dec.n_layer {
            decoder_blocks.push(CanaryDecoderBlockWeights {
                self_attn_norm: vec![1.0; d_dec],
                self_attn_qkv: xavier(&mut rng, d_dec * 3 * d_dec, d_dec, 3 * d_dec),
                self_attn_qkv_bias: vec![0.0; 3 * d_dec],
                self_attn_out: xavier(&mut rng, d_dec * d_dec, d_dec, d_dec),
                self_attn_out_bias: vec![0.0; d_dec],
                cross_attn_norm: vec![1.0; d_dec],
                cross_attn_q: xavier(&mut rng, d_dec * d_dec, d_dec, d_dec),
                cross_attn_q_bias: vec![0.0; d_dec],
                cross_attn_kv: xavier(&mut rng, d_enc * 2 * d_dec, d_enc, 2 * d_dec),
                cross_attn_kv_bias: vec![0.0; 2 * d_dec],
                cross_attn_out: xavier(&mut rng, d_dec * d_dec, d_dec, d_dec),
                cross_attn_out_bias: vec![0.0; d_dec],
                ffn_norm: vec![1.0; d_dec],
                ffn_fc1: xavier(&mut rng, d_dec * dec_ffn, d_dec, dec_ffn),
                ffn_fc1_bias: vec![0.0; dec_ffn],
                ffn_fc2: xavier(&mut rng, dec_ffn * d_dec, dec_ffn, d_dec),
                ffn_fc2_bias: vec![0.0; d_dec],
            });
        }
        let decoder_final_norm = vec![1.0; d_dec];

        // Vocab head — single Linear from d_dec to vocab_size with bias.
        let vocab_head = xavier(&mut rng, d_dec * vocab, d_dec, vocab);
        let vocab_bias = vec![0.0; vocab];

        Ok(Self {
            subsample,
            encoder_blocks,
            encoder_final_norm,
            enc_to_dec_proj,
            enc_to_dec_proj_bias,
            dec_token_embedding,
            dec_position_embedding,
            decoder_blocks,
            decoder_final_norm,
            vocab_head,
            vocab_bias,
            is_synthesized: true,
        })
    }
}

/// Xavier-uniform draw of `count` `f32`s in `[-a, +a]` where
/// `a = sqrt(6 / (fan_in + fan_out))`. Deterministic under a fixed `rng`.
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

/// Canary ASR engine handle.
///
/// Carries the resolved config and weight store. [`Self::transcribe`] is
/// the primary PCM → text entry point; until real weights are bound (see
/// the module docstring) it returns [`VokraError::NotImplemented`] with a
/// message naming the blocker (FR-EX-08 — never a silent zero-fill or
/// empty transcript).
///
/// # Weight license surfacing
///
/// The `weight_license` field carries the compliance class surfaced from
/// the GGUF's `vokra.provenance.weight_license` chunk (populated by
/// [`Self::from_gguf`] / [`Self::from_gguf_with_policy`]) or defaults to
/// [`LicenseClass::AttributionRequired`] under [`Self::new`] (the CC-BY
/// 4.0 class that is the only legitimate class for real Canary weights
/// per the compliance registry — `vokra_core::compliance::license_class`
/// maps `canary` / `canary-1b-v2` / the whole `canary-*` family to
/// [`LicenseClass::AttributionRequired`]). The M2-13 outer compliance
/// gate does the strict enforcement (see
/// [`Self::from_gguf_with_policy`]); this handle simply surfaces the
/// class so callers can cross-check.
#[derive(Debug, Clone)]
pub struct CanaryAsr {
    cfg: CanaryConfig,
    weights: CanaryWeights,
    weight_license: LicenseClass,
}

impl CanaryAsr {
    /// Assembles an engine from `cfg` and `weights`. Cross-checks the
    /// weight-store shapes against `cfg` (encoder / decoder block counts,
    /// per-tensor sizes, encoder attention-bias presence) so a mismatched
    /// pair fails loudly here rather than deep inside a forward.
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] from `cfg.validate_for_forward`.
    /// - [`VokraError::InvalidArgument`] naming the first shape
    ///   mismatch.
    pub fn new(cfg: CanaryConfig, weights: CanaryWeights) -> Result<Self> {
        cfg.validate_for_forward()?;
        let enc = &cfg.encoder;
        let dec = &cfg.decoder;
        let head = &cfg.head;
        let d_enc = enc.d_model;
        let d_dec = dec.d_model;
        let enc_ffn = enc.ffn_dim;
        let dec_ffn = dec.ffn_dim;
        let vocab = head.vocab_size;
        let k = enc.conv_kernel_size;
        let projection_in = enc.subsampling_factor * enc.in_dim;
        let bias_on = enc.attention_bias;

        // Subsample stem.
        if weights.subsample.linear_w.len() != d_enc * projection_in {
            return Err(VokraError::InvalidArgument(format!(
                "canary weights: subsample.linear_w.len()={} != d_enc * \
                 (subsampling_factor * in_dim) = {} * {} = {}",
                weights.subsample.linear_w.len(),
                d_enc,
                projection_in,
                d_enc * projection_in,
            )));
        }
        if weights.subsample.linear_b.len() != d_enc {
            return Err(VokraError::InvalidArgument(format!(
                "canary weights: subsample.linear_b.len()={} != d_enc={}",
                weights.subsample.linear_b.len(),
                d_enc,
            )));
        }

        // Encoder blocks.
        if weights.encoder_blocks.len() != enc.n_layer {
            return Err(VokraError::InvalidArgument(format!(
                "canary weights: encoder_blocks.len()={} != encoder.n_layer={}",
                weights.encoder_blocks.len(),
                enc.n_layer,
            )));
        }
        for (i, blk) in weights.encoder_blocks.iter().enumerate() {
            for (name, len, expected) in [
                ("ff1_norm", blk.ff1_norm.len(), d_enc),
                ("ff1_fc1", blk.ff1_fc1.len(), d_enc * enc_ffn),
                ("ff1_fc2", blk.ff1_fc2.len(), enc_ffn * d_enc),
                ("attn_norm", blk.attn_norm.len(), d_enc),
                ("qkv_proj", blk.qkv_proj.len(), d_enc * 3 * d_enc),
                ("attn_out", blk.attn_out.len(), d_enc * d_enc),
                ("conv_norm", blk.conv_norm.len(), d_enc),
                ("conv_pw1", blk.conv_pw1.len(), d_enc * 2 * d_enc),
                ("conv_dw", blk.conv_dw.len(), d_enc * k),
                ("conv_dw_norm", blk.conv_dw_norm.len(), d_enc),
                ("conv_pw2", blk.conv_pw2.len(), d_enc * d_enc),
                ("ff2_norm", blk.ff2_norm.len(), d_enc),
                ("ff2_fc1", blk.ff2_fc1.len(), d_enc * enc_ffn),
                ("ff2_fc2", blk.ff2_fc2.len(), enc_ffn * d_enc),
                ("final_norm", blk.final_norm.len(), d_enc),
            ] {
                if len != expected {
                    return Err(VokraError::InvalidArgument(format!(
                        "canary weights: encoder block {i} `{name}` \
                         len={len} != {expected}",
                    )));
                }
            }
            // Encoder attention bias presence + shape cross-check
            // (attention_bias=true means every projection carries a bias;
            // false means neither of the two vectors is present). A
            // mismatch is a loud error — no silent zero-fill, no silent
            // drop (FR-EX-08).
            match (bias_on, &blk.qkv_bias) {
                (true, Some(v)) if v.len() == 3 * d_enc => {}
                (true, Some(v)) => {
                    return Err(VokraError::InvalidArgument(format!(
                        "canary weights: encoder block {i} qkv_bias.len()={} \
                         != 3*d_enc={}",
                        v.len(),
                        3 * d_enc,
                    )));
                }
                (true, None) => {
                    return Err(VokraError::InvalidArgument(format!(
                        "canary weights: encoder block {i} qkv_bias is None but \
                         encoder.attention_bias=true — a bias-free variant must set \
                         attention_bias=false",
                    )));
                }
                (false, Some(_)) => {
                    return Err(VokraError::InvalidArgument(format!(
                        "canary weights: encoder block {i} qkv_bias is Some but \
                         encoder.attention_bias=false — a bias-carrying variant must \
                         set attention_bias=true",
                    )));
                }
                (false, None) => {}
            }
            match (bias_on, &blk.attn_out_bias) {
                (true, Some(v)) if v.len() == d_enc => {}
                (true, Some(v)) => {
                    return Err(VokraError::InvalidArgument(format!(
                        "canary weights: encoder block {i} attn_out_bias.len()={} \
                         != d_enc={}",
                        v.len(),
                        d_enc,
                    )));
                }
                (true, None) => {
                    return Err(VokraError::InvalidArgument(format!(
                        "canary weights: encoder block {i} attn_out_bias is None but \
                         encoder.attention_bias=true — a bias-free variant must set \
                         attention_bias=false",
                    )));
                }
                (false, Some(_)) => {
                    return Err(VokraError::InvalidArgument(format!(
                        "canary weights: encoder block {i} attn_out_bias is Some but \
                         encoder.attention_bias=false — a bias-carrying variant must \
                         set attention_bias=true",
                    )));
                }
                (false, None) => {}
            }
        }
        if weights.encoder_final_norm.len() != d_enc {
            return Err(VokraError::InvalidArgument(format!(
                "canary weights: encoder_final_norm.len()={} != d_enc={}",
                weights.encoder_final_norm.len(),
                d_enc,
            )));
        }

        // Encoder→decoder width projection scaffold.
        if weights.enc_to_dec_proj.len() != d_enc * d_dec {
            return Err(VokraError::InvalidArgument(format!(
                "canary weights: enc_to_dec_proj.len()={} != d_enc * d_dec = {} * {} = {}",
                weights.enc_to_dec_proj.len(),
                d_enc,
                d_dec,
                d_enc * d_dec,
            )));
        }
        if weights.enc_to_dec_proj_bias.len() != d_dec {
            return Err(VokraError::InvalidArgument(format!(
                "canary weights: enc_to_dec_proj_bias.len()={} != d_dec={}",
                weights.enc_to_dec_proj_bias.len(),
                d_dec,
            )));
        }

        // Decoder embeddings.
        if weights.dec_token_embedding.len() != vocab * d_dec {
            return Err(VokraError::InvalidArgument(format!(
                "canary weights: dec_token_embedding.len()={} != vocab * d_dec = {} * {} = {}",
                weights.dec_token_embedding.len(),
                vocab,
                d_dec,
                vocab * d_dec,
            )));
        }
        if weights.dec_position_embedding.len() != dec.max_sequence_length * d_dec {
            return Err(VokraError::InvalidArgument(format!(
                "canary weights: dec_position_embedding.len()={} != \
                 max_sequence_length * d_dec = {} * {} = {}",
                weights.dec_position_embedding.len(),
                dec.max_sequence_length,
                d_dec,
                dec.max_sequence_length * d_dec,
            )));
        }

        // Decoder blocks.
        if weights.decoder_blocks.len() != dec.n_layer {
            return Err(VokraError::InvalidArgument(format!(
                "canary weights: decoder_blocks.len()={} != decoder.n_layer={}",
                weights.decoder_blocks.len(),
                dec.n_layer,
            )));
        }
        for (i, blk) in weights.decoder_blocks.iter().enumerate() {
            for (name, len, expected) in [
                ("self_attn_norm", blk.self_attn_norm.len(), d_dec),
                ("self_attn_qkv", blk.self_attn_qkv.len(), d_dec * 3 * d_dec),
                (
                    "self_attn_qkv_bias",
                    blk.self_attn_qkv_bias.len(),
                    3 * d_dec,
                ),
                ("self_attn_out", blk.self_attn_out.len(), d_dec * d_dec),
                ("self_attn_out_bias", blk.self_attn_out_bias.len(), d_dec),
                ("cross_attn_norm", blk.cross_attn_norm.len(), d_dec),
                ("cross_attn_q", blk.cross_attn_q.len(), d_dec * d_dec),
                ("cross_attn_q_bias", blk.cross_attn_q_bias.len(), d_dec),
                ("cross_attn_kv", blk.cross_attn_kv.len(), d_enc * 2 * d_dec),
                (
                    "cross_attn_kv_bias",
                    blk.cross_attn_kv_bias.len(),
                    2 * d_dec,
                ),
                ("cross_attn_out", blk.cross_attn_out.len(), d_dec * d_dec),
                ("cross_attn_out_bias", blk.cross_attn_out_bias.len(), d_dec),
                ("ffn_norm", blk.ffn_norm.len(), d_dec),
                ("ffn_fc1", blk.ffn_fc1.len(), d_dec * dec_ffn),
                ("ffn_fc1_bias", blk.ffn_fc1_bias.len(), dec_ffn),
                ("ffn_fc2", blk.ffn_fc2.len(), dec_ffn * d_dec),
                ("ffn_fc2_bias", blk.ffn_fc2_bias.len(), d_dec),
            ] {
                if len != expected {
                    return Err(VokraError::InvalidArgument(format!(
                        "canary weights: decoder block {i} `{name}` \
                         len={len} != {expected}",
                    )));
                }
            }
        }
        if weights.decoder_final_norm.len() != d_dec {
            return Err(VokraError::InvalidArgument(format!(
                "canary weights: decoder_final_norm.len()={} != d_dec={}",
                weights.decoder_final_norm.len(),
                d_dec,
            )));
        }

        // Vocab head.
        if weights.vocab_head.len() != d_dec * vocab {
            return Err(VokraError::InvalidArgument(format!(
                "canary weights: vocab_head.len()={} != d_dec * vocab = {} * {} = {}",
                weights.vocab_head.len(),
                d_dec,
                vocab,
                d_dec * vocab,
            )));
        }
        if weights.vocab_bias.len() != vocab {
            return Err(VokraError::InvalidArgument(format!(
                "canary weights: vocab_bias.len()={} != vocab_size={}",
                weights.vocab_bias.len(),
                vocab,
            )));
        }

        Ok(Self {
            cfg,
            weights,
            // Default weight-license class under `new()` mirrors the
            // compliance registry (`vokra_core::compliance::license_class`
            // maps `canary` / `canary-1b-v2` / the whole `canary-*` family
            // to CC-BY 4.0 = AttributionRequired). `from_gguf` overrides
            // with whatever the provenance chunk carries (or `Unknown` if
            // absent — fail-closed at the outer M2-13 gate).
            weight_license: LicenseClass::AttributionRequired,
        })
    }

    /// The resolved configuration.
    #[must_use]
    pub fn config(&self) -> &CanaryConfig {
        &self.cfg
    }

    /// True iff the weight store was built by
    /// [`CanaryWeights::synthesized`] (never a real upstream checkpoint).
    #[must_use]
    pub fn is_synthesized(&self) -> bool {
        self.weights.is_synthesized
    }

    /// Transcribes a mono `f32` PCM slice at [`Self::config`]'s sample
    /// rate.
    ///
    /// This is the primary PCM → text entry point. **Real weights
    /// required**: synthesized-weight builds cannot produce meaningful
    /// text (they would be a hallucinated sequence), so this returns
    /// [`VokraError::NotImplemented`] naming the blocker. Callers verify
    /// the shape flow through [`CanaryAsr::new`] +
    /// [`CanaryWeights::synthesized`] today; a follow-up wave binds the
    /// real `.nemo` checkpoint tensor names and wires the forward.
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] if `pcm` is empty.
    /// - [`VokraError::NotImplemented`] otherwise (real forward not yet
    ///   bound — FR-EX-08).
    pub fn transcribe(&self, pcm: &[f32]) -> Result<Vec<u32>> {
        if pcm.is_empty() {
            return Err(VokraError::InvalidArgument(
                "canary transcribe: pcm slice is empty".to_owned(),
            ));
        }
        if self.weights.is_synthesized {
            return Err(VokraError::NotImplemented(
                "canary transcribe: this engine holds synthesized weights \
                 (deterministic fixture from CanaryWeights::synthesized) — \
                 synthesized-weight text would be a hallucinated sequence, \
                 not a real transcript. Bind real Canary-1B-v2 weights \
                 (CC-BY 4.0, nvidia/canary-1b-v2 — distributed as a .nemo \
                 tarball) before invoking transcribe. The shape flow (config \
                 validation, weight-store construction, PCM boundary check) \
                 is exercised through CanaryAsr::new; the real-checkpoint \
                 tensor-name manifest lands in a follow-up wave \
                 (T29-equivalent — the Moshi / CSM / Zonos / Kyutai STT / \
                 Parakeet-TDT / Parakeet-CTC pattern).",
            ));
        }
        Err(VokraError::NotImplemented(
            "canary transcribe: real weights are bound but the log-mel \
             front-end → FastConformer encoder (vokra_ops::conformer) → \
             encoder→decoder projection → task-prompt-prefixed Transformer \
             decoder (self-attn + cross-attn + FFN) → vocab head → \
             beam_search (vokra_core::decode::beam_search) → SentencePiece \
             detokenize forward path has not landed yet. Follow-up wave: \
             wire CanaryWeights to vokra_ops::conformer::ConformerEncoder + \
             the decoder step + vokra_core::decode::beam_search with \
             blank_id / bos / eos taken from head.{pad,bos,eos}_token_id \
             once the .nemo extraction supplies them. Primary source: \
             https://huggingface.co/nvidia/canary-1b-v2",
        ))
    }

    /// The stamped weight-license class surfaced from the GGUF's
    /// `vokra.provenance.weight_license` chunk. For real Canary
    /// checkpoints the compliance registry
    /// (`vokra_core::compliance::license_class`) maps `canary` /
    /// `canary-1b-v2` / the whole `canary-*` family to
    /// [`LicenseClass::AttributionRequired`] (CC-BY 4.0). A GGUF missing
    /// the stamp reads back as [`LicenseClass::Unknown`] (fail-closed at
    /// the outer M2-13 gate); [`Self::new`] defaults to
    /// [`LicenseClass::AttributionRequired`] (the only legitimate class
    /// for real weights).
    #[inline]
    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    /// Binds a Canary GGUF: validates arch, reads the strict
    /// `vokra.canary.*` topology chunk group, builds a deterministic
    /// synthesized weight fixture matching the resolved config, and
    /// surfaces the stamped weight-license class for compliance-gate
    /// cross-checks.
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
    /// Parakeet-TDT / Parakeet-CTC pattern) as the follow-up wave's
    /// anchor. The primitives named in that message
    /// ([`vokra_ops::conformer`] + `vokra_core::decode::beam_search` —
    /// the search primitive lives in `vokra-core`, not `vokra-ops`,
    /// because per FR-OP-40 it is a host-side search rather than a
    /// graph `OpKind`) already
    /// exist; the missing piece is the HF `.nemo` extraction →
    /// [`CanaryWeights`] tensor-name manifest plus SentencePiece
    /// detokenize (model-specific, not a shared op).
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when `vokra.model.arch` is absent or
    ///   not `"canary"` (a sibling ASR GGUF handed to us by mistake
    ///   fails with a hint naming the sibling arches — the
    ///   Parakeet-CTC / Kyutai STT precedent).
    /// - [`VokraError::ModelLoad`] when any `vokra.canary.*` chunk is
    ///   absent ([`CanaryConfig::from_gguf`] is strict).
    /// - [`VokraError::InvalidArgument`] from
    ///   [`CanaryConfig::validate_for_forward`] +
    ///   [`CanaryAsr::new`] shape gates.
    ///
    /// # See also
    ///
    /// - [`Self::from_gguf_with_policy`] — the M2-13 compliance-gated
    ///   primary path (parses raw bytes, enforces
    ///   [`CompliancePolicy`]).
    /// - [`Self::from_path`] — fail-closed convenience wrapper around
    ///   `from_gguf_with_policy` with [`CompliancePolicy::strict`].
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        // 1. Arch check — always first so a mis-typed model handed here
        //    fails with a specific message instead of a downstream
        //    "vokra.canary.arch.encoder.n_layer missing".
        match file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()) {
            Some(a) if a == EXPECTED_ARCH => {}
            Some(other) => {
                return Err(VokraError::ModelLoad(format!(
                    "canary: GGUF arch is `{other}`, expected `{EXPECTED_ARCH}` \
                     (was this GGUF produced by `vokra-cli convert --model canary`? \
                     Sibling ASR arches — `whisper`, `voxtral`, `parakeet-ctc`, \
                     `parakeet-tdt`, `kyutai-stt` — are completely different \
                     topologies). Primary source: \
                     https://huggingface.co/nvidia/canary-1b-v2"
                )));
            }
            None => {
                return Err(VokraError::ModelLoad(
                    "canary: GGUF is missing `vokra.model.arch` (converter did \
                     not stamp it — this is not a Vokra-native canary GGUF). \
                     Primary source: https://huggingface.co/nvidia/canary-1b-v2"
                        .to_owned(),
                ));
            }
        }

        // 2. Strict topology axes from the `vokra.canary.*` chunk group.
        let cfg = CanaryConfig::from_gguf(file)?;

        // 3. Provenance surfacing — read the stamped weight-license class
        //    for compliance-gate cross-checks (defaults to `Unknown` if
        //    absent, which is fail-closed at the outer M2-13 gate).
        //    Matches the Parakeet-CTC / MT3 / SNAC precedent — surface
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
        //    binding real `.nemo` checkpoint tensor names is the
        //    follow-up wave (T29-equivalent).
        let weights = CanaryWeights::synthesized(&cfg, CANARY_FROM_GGUF_DEFAULT_SEED)?;
        let mut asr = Self::new(cfg, weights)?;
        asr.weight_license = weight_license;
        Ok(asr)
    }

    /// Loads a Canary GGUF from raw bytes under `policy` (M2-13 gate —
    /// a non-commercial provenance without a research flag is refused).
    ///
    /// Weight posture: **synthesized bridge** until the real-checkpoint
    /// tensor-name manifest lands (T29-equivalent — the CSM /
    /// Parakeet-CTC / Kyutai STT precedent). The engine binds
    /// [`CanaryWeights::synthesized`] against the GGUF's shape config
    /// using [`CANARY_FROM_GGUF_DEFAULT_SEED`] so shape / dtype / size
    /// flow can be exercised without the real `.nemo` extraction; a
    /// [`Self::transcribe`] call fires the synthesized-weight
    /// loud-partial arm and names the primary source URL.
    ///
    /// The Canary weight license is **CC-BY 4.0** (`AttributionRequired`) —
    /// the converter's registry mapping and provenance stamps make the
    /// M2-13 gate pass commercially, and the FR-MD-09 attribution
    /// surface activates.
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] on parse failure / wrong or missing
    ///   `vokra.model.arch` — the message names the expected arch tag
    ///   (`canary`), sibling ASR arch tags (`whisper` / `voxtral` /
    ///   `parakeet-ctc` / `parakeet-tdt` / `kyutai-stt`) so a mis-routed
    ///   GGUF fails specifically here, and the primary source URL.
    /// - [`VokraError::ResearchLicenseRequired`] (from the M2-13 gate)
    ///   when the weight class is gated and `policy` grants no research
    ///   opt-in (never a silent skip / substitution).
    /// - [`VokraError::ModelLoad`] when any `vokra.canary.*` chunk is
    ///   absent ([`CanaryConfig::from_gguf`] is strict).
    /// - [`VokraError::InvalidArgument`] on a `0`-placeholder shape
    ///   config (a scaffold converter path that never wrote the real
    ///   hparams) from the downstream
    ///   [`CanaryConfig::validate_for_forward`] gate.
    pub fn from_gguf_with_policy(bytes: &[u8], policy: &CompliancePolicy) -> Result<Self> {
        let file = GgufFile::parse(bytes.to_vec())
            .map_err(|e| VokraError::ModelLoad(format!("canary GGUF: {e}")))?;
        match file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()) {
            Some(a) if a == EXPECTED_ARCH => {}
            Some(other) => {
                return Err(VokraError::ModelLoad(format!(
                    "canary: GGUF arch is `{other}`, expected `{EXPECTED_ARCH}` \
                     (was this GGUF produced by `vokra-cli convert --model canary`? \
                     Sibling ASR arches — `whisper`, `voxtral`, `parakeet-ctc`, \
                     `parakeet-tdt`, `kyutai-stt` — are completely different \
                     topologies). Primary source: \
                     https://huggingface.co/nvidia/canary-1b-v2"
                )));
            }
            None => {
                return Err(VokraError::ModelLoad(
                    "canary: GGUF is missing `vokra.model.arch` (converter did \
                     not stamp it — this is not a Vokra-native canary GGUF). \
                     Primary source: https://huggingface.co/nvidia/canary-1b-v2"
                        .to_owned(),
                ));
            }
        }
        check_weight_license(&file, policy)?;
        let cfg = CanaryConfig::from_gguf(&file)?;
        // `synthesized` runs `validate_for_forward` internally; keep the
        // explicit call here so a validate failure surfaces with the config
        // context intact (same posture as CSM / Kyutai STT
        // `from_gguf_with_policy`).
        cfg.validate_for_forward()?;
        let weight_license = file
            .get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
            .and_then(|v| v.as_str())
            .and_then(LicenseClass::from_class_str)
            .unwrap_or(LicenseClass::Unknown);
        let weights = CanaryWeights::synthesized(&cfg, CANARY_FROM_GGUF_DEFAULT_SEED)?;
        let mut asr = Self::new(cfg, weights)?;
        asr.weight_license = weight_license;
        Ok(asr)
    }

    /// Loads a Canary GGUF from a file path with the fail-closed strict
    /// policy ([`CompliancePolicy::strict`]).
    ///
    /// The Canary weight license is **CC-BY 4.0**
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

    /// Every hparam matches the primary sources (model card + family
    /// reference yaml) verbatim.
    #[test]
    fn canary_1b_v2_matches_primary_sources() {
        let c = CanaryConfig::canary_1b_v2();
        // Encoder (model card + family reference).
        assert_eq!(c.encoder.n_layer, 32, "model card: 32 encoder layers");
        assert_eq!(c.encoder.d_model, 1024, "family default: d_model=1024");
        assert_eq!(c.encoder.n_head, 8, "family default: n_heads=8");
        assert_eq!(c.encoder.n_head_kv, 8, "MHA (no GQA)");
        assert_eq!(
            c.encoder.ffn_dim, 4096,
            "family default: ff_expansion_factor=4 x 1024"
        );
        assert_eq!(c.encoder.conv_kernel_size, 9, "family default");
        assert_eq!(c.encoder.in_dim, 128, "family default: features=128");
        assert_eq!(c.encoder.subsampling_factor, 8, "family default");
        assert_eq!(c.encoder.subsampling_conv_kernel_size, 3);
        assert_eq!(c.encoder.subsampling_conv_stride, 2);
        assert_eq!(c.encoder.subsampling_conv_channels, 256);
        assert_eq!(c.encoder.max_position_embeddings, 5000);
        assert!(
            c.encoder.attention_bias,
            "family default: untie_biases=true"
        );
        assert!(!c.encoder.convolution_bias);
        assert!(!c.encoder.scale_input, "family default: xscaling=false");
        // Decoder (model card + family reference).
        assert_eq!(c.decoder.n_layer, 8, "model card: 8 decoder layers");
        assert_eq!(
            c.decoder.d_model, 1024,
            "family default: lm_dec_hidden=1024"
        );
        assert_eq!(c.decoder.n_head, 8);
        assert_eq!(c.decoder.ffn_dim, 4096, "family default: 4 x lm_dec_hidden");
        assert_eq!(
            c.decoder.max_sequence_length, 1024,
            "family convention (flash variants)"
        );
        assert!(c.decoder.pre_ln);
        assert_eq!(c.decoder.hidden_act, "relu");
        // Head (model card).
        assert_eq!(
            c.head.vocab_size, 16_384,
            "model card: unified SentencePiece 16,384 tokens"
        );
        // Placeholder-sentinel token ids (the .nemo extraction will set real values).
        assert_eq!(c.head.pad_token_id, 0);
        assert_eq!(c.head.bos_token_id, 0);
        assert_eq!(c.head.eos_token_id, 0);
        // Audio boundary (model card).
        assert_eq!(c.sample_rate, 16_000);
        // Derived.
        assert_eq!(c.encoder.head_dim(), 128);
        assert_eq!(c.encoder.kv_hidden(), 1024); // MHA
        // The `0` token id sentinel is still a legal id (< vocab_size), so
        // the config validates today; the .nemo extraction can bump bos/eos
        // without changing the validator.
        c.validate_for_forward()
            .expect("canary-1b-v2 is well-formed at the head sentinels");
    }

    #[test]
    fn tiny_config_is_well_formed() {
        CanaryConfig::tiny_for_tests()
            .validate_for_forward()
            .expect("tiny config is well-formed");
    }

    #[test]
    fn config_encoder_head_split_ill_formed_is_rejected() {
        let mut c = CanaryConfig::tiny_for_tests();
        c.encoder.n_head = 3; // 16 % 3 != 0
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_encoder_odd_head_dim_is_rejected() {
        let mut c = CanaryConfig::tiny_for_tests();
        // 12 / 4 = 3 (odd)
        c.encoder.d_model = 12;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_encoder_gqa_broadcast_not_dividing_is_rejected() {
        let mut c = CanaryConfig::tiny_for_tests();
        c.encoder.n_head = 6;
        c.encoder.d_model = 24;
        c.encoder.n_head_kv = 4; // 6 % 4 != 0
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_encoder_zero_layer_is_rejected() {
        let mut c = CanaryConfig::tiny_for_tests();
        c.encoder.n_layer = 0;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_encoder_zero_ffn_is_rejected() {
        let mut c = CanaryConfig::tiny_for_tests();
        c.encoder.ffn_dim = 0;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_encoder_zero_in_dim_is_rejected() {
        let mut c = CanaryConfig::tiny_for_tests();
        c.encoder.in_dim = 0;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_encoder_even_conv_kernel_is_rejected() {
        let mut c = CanaryConfig::tiny_for_tests();
        c.encoder.conv_kernel_size = 4;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_encoder_zero_subsampling_is_rejected() {
        let mut c = CanaryConfig::tiny_for_tests();
        c.encoder.subsampling_factor = 0;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_encoder_zero_max_positions_is_rejected() {
        let mut c = CanaryConfig::tiny_for_tests();
        c.encoder.max_position_embeddings = 0;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_decoder_zero_layer_is_rejected() {
        let mut c = CanaryConfig::tiny_for_tests();
        c.decoder.n_layer = 0;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_decoder_head_split_ill_formed_is_rejected() {
        let mut c = CanaryConfig::tiny_for_tests();
        c.decoder.n_head = 3; // 16 % 3 != 0
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_decoder_zero_max_seq_is_rejected() {
        let mut c = CanaryConfig::tiny_for_tests();
        c.decoder.max_sequence_length = 0;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_decoder_empty_hidden_act_is_rejected() {
        let mut c = CanaryConfig::tiny_for_tests();
        c.decoder.hidden_act.clear();
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_zero_vocab_is_rejected() {
        let mut c = CanaryConfig::tiny_for_tests();
        c.head.vocab_size = 0;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_bos_out_of_range_is_rejected() {
        let mut c = CanaryConfig::tiny_for_tests();
        c.head.bos_token_id = c.head.vocab_size as u32;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_eos_out_of_range_is_rejected() {
        let mut c = CanaryConfig::tiny_for_tests();
        c.head.eos_token_id = c.head.vocab_size as u32;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_pad_out_of_range_is_rejected() {
        let mut c = CanaryConfig::tiny_for_tests();
        c.head.pad_token_id = c.head.vocab_size as u32;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn synthesized_weights_are_deterministic_and_shape_correct() {
        let c = CanaryConfig::tiny_for_tests();
        let w1 = CanaryWeights::synthesized(&c, 0x42).expect("build 1");
        let w2 = CanaryWeights::synthesized(&c, 0x42).expect("build 2");
        // Determinism.
        assert_eq!(w1.subsample.linear_w, w2.subsample.linear_w);
        assert_eq!(w1.encoder_blocks[0].qkv_proj, w2.encoder_blocks[0].qkv_proj);
        assert_eq!(w1.dec_token_embedding, w2.dec_token_embedding);
        assert_eq!(w1.vocab_head, w2.vocab_head);
        assert!(w1.is_synthesized);

        // Shape flow (encoder).
        let enc = &c.encoder;
        let dec = &c.decoder;
        let head = &c.head;
        let d_enc = enc.d_model;
        let d_dec = dec.d_model;
        let enc_ffn = enc.ffn_dim;
        let dec_ffn = dec.ffn_dim;
        let vocab = head.vocab_size;
        let k = enc.conv_kernel_size;
        let projection_in = enc.subsampling_factor * enc.in_dim;
        assert_eq!(w1.subsample.linear_w.len(), d_enc * projection_in);
        assert_eq!(w1.subsample.linear_b.len(), d_enc);
        assert_eq!(w1.encoder_blocks.len(), enc.n_layer);
        for blk in &w1.encoder_blocks {
            assert_eq!(blk.ff1_norm.len(), d_enc);
            assert_eq!(blk.ff1_fc1.len(), d_enc * enc_ffn);
            assert_eq!(blk.ff1_fc2.len(), enc_ffn * d_enc);
            assert_eq!(blk.attn_norm.len(), d_enc);
            assert_eq!(blk.qkv_proj.len(), d_enc * 3 * d_enc);
            assert_eq!(blk.attn_out.len(), d_enc * d_enc);
            assert_eq!(blk.conv_norm.len(), d_enc);
            assert_eq!(blk.conv_pw1.len(), d_enc * 2 * d_enc);
            assert_eq!(blk.conv_dw.len(), d_enc * k);
            assert_eq!(blk.conv_dw_norm.len(), d_enc);
            assert_eq!(blk.conv_pw2.len(), d_enc * d_enc);
            assert_eq!(blk.ff2_norm.len(), d_enc);
            assert_eq!(blk.ff2_fc1.len(), d_enc * enc_ffn);
            assert_eq!(blk.ff2_fc2.len(), enc_ffn * d_enc);
            assert_eq!(blk.final_norm.len(), d_enc);
            // Encoder biases present iff attention_bias == true.
            assert!(blk.qkv_bias.is_some());
            assert_eq!(blk.qkv_bias.as_ref().unwrap().len(), 3 * d_enc);
            assert!(blk.attn_out_bias.is_some());
            assert_eq!(blk.attn_out_bias.as_ref().unwrap().len(), d_enc);
        }
        assert_eq!(w1.encoder_final_norm.len(), d_enc);

        // Encoder→decoder projection scaffold + embeddings.
        assert_eq!(w1.enc_to_dec_proj.len(), d_enc * d_dec);
        assert_eq!(w1.enc_to_dec_proj_bias.len(), d_dec);
        assert_eq!(w1.dec_token_embedding.len(), vocab * d_dec);
        assert_eq!(
            w1.dec_position_embedding.len(),
            dec.max_sequence_length * d_dec
        );

        // Decoder blocks.
        assert_eq!(w1.decoder_blocks.len(), dec.n_layer);
        for blk in &w1.decoder_blocks {
            assert_eq!(blk.self_attn_norm.len(), d_dec);
            assert_eq!(blk.self_attn_qkv.len(), d_dec * 3 * d_dec);
            assert_eq!(blk.self_attn_qkv_bias.len(), 3 * d_dec);
            assert_eq!(blk.self_attn_out.len(), d_dec * d_dec);
            assert_eq!(blk.self_attn_out_bias.len(), d_dec);
            assert_eq!(blk.cross_attn_norm.len(), d_dec);
            assert_eq!(blk.cross_attn_q.len(), d_dec * d_dec);
            assert_eq!(blk.cross_attn_q_bias.len(), d_dec);
            assert_eq!(blk.cross_attn_kv.len(), d_enc * 2 * d_dec);
            assert_eq!(blk.cross_attn_kv_bias.len(), 2 * d_dec);
            assert_eq!(blk.cross_attn_out.len(), d_dec * d_dec);
            assert_eq!(blk.cross_attn_out_bias.len(), d_dec);
            assert_eq!(blk.ffn_norm.len(), d_dec);
            assert_eq!(blk.ffn_fc1.len(), d_dec * dec_ffn);
            assert_eq!(blk.ffn_fc1_bias.len(), dec_ffn);
            assert_eq!(blk.ffn_fc2.len(), dec_ffn * d_dec);
            assert_eq!(blk.ffn_fc2_bias.len(), d_dec);
        }
        assert_eq!(w1.decoder_final_norm.len(), d_dec);
        assert_eq!(w1.vocab_head.len(), d_dec * vocab);
        assert_eq!(w1.vocab_bias.len(), vocab);
    }

    /// A bias-free encoder variant (a hypothetical future Canary config
    /// with `attention_bias=false`) drops both encoder biases to None —
    /// the synthesized builder must respect the flag, and the runtime
    /// must accept the resulting None pair.
    #[test]
    fn synthesized_weights_respect_encoder_attention_bias_off() {
        let mut c = CanaryConfig::tiny_for_tests();
        c.encoder.attention_bias = false;
        let w = CanaryWeights::synthesized(&c, 7).expect("weights");
        for blk in &w.encoder_blocks {
            assert!(blk.qkv_bias.is_none());
            assert!(blk.attn_out_bias.is_none());
        }
        // The runtime accepts the bias-free pair.
        CanaryAsr::new(c, w).expect("bias-free variant loadable");
    }

    #[test]
    fn synthesized_weights_different_seeds_diverge() {
        let c = CanaryConfig::tiny_for_tests();
        let w_a = CanaryWeights::synthesized(&c, 1).expect("build a");
        let w_b = CanaryWeights::synthesized(&c, 2).expect("build b");
        assert_ne!(w_a.subsample.linear_w, w_b.subsample.linear_w);
        assert_ne!(w_a.dec_token_embedding, w_b.dec_token_embedding);
    }

    #[test]
    fn synthesized_rejects_ill_formed_config() {
        let mut c = CanaryConfig::tiny_for_tests();
        c.encoder.n_head = 3;
        assert!(matches!(
            CanaryWeights::synthesized(&c, 7),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn asr_new_accepts_matching_config_and_weights() {
        let c = CanaryConfig::tiny_for_tests();
        let w = CanaryWeights::synthesized(&c, 7).expect("weights");
        let asr = CanaryAsr::new(c.clone(), w).expect("canary asr");
        assert_eq!(asr.config().encoder.d_model, c.encoder.d_model);
        assert_eq!(asr.config().decoder.d_model, c.decoder.d_model);
        assert!(asr.is_synthesized());
    }

    #[test]
    fn asr_new_rejects_encoder_layer_count_mismatch() {
        let c = CanaryConfig::tiny_for_tests();
        let mut w = CanaryWeights::synthesized(&c, 7).expect("weights");
        w.encoder_blocks.pop();
        assert!(matches!(
            CanaryAsr::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn asr_new_rejects_encoder_tensor_size_mismatch() {
        let c = CanaryConfig::tiny_for_tests();
        let mut w = CanaryWeights::synthesized(&c, 7).expect("weights");
        w.encoder_blocks[0].qkv_proj.pop();
        assert!(matches!(
            CanaryAsr::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn asr_new_rejects_subsample_size_mismatch() {
        let c = CanaryConfig::tiny_for_tests();
        let mut w = CanaryWeights::synthesized(&c, 7).expect("weights");
        w.subsample.linear_w.pop();
        assert!(matches!(
            CanaryAsr::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn asr_new_rejects_encoder_final_norm_mismatch() {
        let c = CanaryConfig::tiny_for_tests();
        let mut w = CanaryWeights::synthesized(&c, 7).expect("weights");
        w.encoder_final_norm.pop();
        assert!(matches!(
            CanaryAsr::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn asr_new_rejects_enc_to_dec_proj_mismatch() {
        let c = CanaryConfig::tiny_for_tests();
        let mut w = CanaryWeights::synthesized(&c, 7).expect("weights");
        w.enc_to_dec_proj.pop();
        assert!(matches!(
            CanaryAsr::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn asr_new_rejects_dec_token_embedding_mismatch() {
        let c = CanaryConfig::tiny_for_tests();
        let mut w = CanaryWeights::synthesized(&c, 7).expect("weights");
        w.dec_token_embedding.pop();
        assert!(matches!(
            CanaryAsr::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn asr_new_rejects_dec_position_embedding_mismatch() {
        let c = CanaryConfig::tiny_for_tests();
        let mut w = CanaryWeights::synthesized(&c, 7).expect("weights");
        w.dec_position_embedding.pop();
        assert!(matches!(
            CanaryAsr::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn asr_new_rejects_decoder_layer_count_mismatch() {
        let c = CanaryConfig::tiny_for_tests();
        let mut w = CanaryWeights::synthesized(&c, 7).expect("weights");
        w.decoder_blocks.pop();
        assert!(matches!(
            CanaryAsr::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn asr_new_rejects_decoder_self_attn_qkv_size_mismatch() {
        let c = CanaryConfig::tiny_for_tests();
        let mut w = CanaryWeights::synthesized(&c, 7).expect("weights");
        w.decoder_blocks[0].self_attn_qkv.pop();
        assert!(matches!(
            CanaryAsr::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn asr_new_rejects_decoder_cross_attn_kv_size_mismatch() {
        let c = CanaryConfig::tiny_for_tests();
        let mut w = CanaryWeights::synthesized(&c, 7).expect("weights");
        w.decoder_blocks[0].cross_attn_kv.pop();
        assert!(matches!(
            CanaryAsr::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn asr_new_rejects_decoder_final_norm_mismatch() {
        let c = CanaryConfig::tiny_for_tests();
        let mut w = CanaryWeights::synthesized(&c, 7).expect("weights");
        w.decoder_final_norm.pop();
        assert!(matches!(
            CanaryAsr::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn asr_new_rejects_vocab_head_mismatch() {
        let c = CanaryConfig::tiny_for_tests();
        let mut w = CanaryWeights::synthesized(&c, 7).expect("weights");
        w.vocab_head.pop();
        assert!(matches!(
            CanaryAsr::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn asr_new_rejects_vocab_bias_mismatch() {
        let c = CanaryConfig::tiny_for_tests();
        let mut w = CanaryWeights::synthesized(&c, 7).expect("weights");
        w.vocab_bias.pop();
        assert!(matches!(
            CanaryAsr::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    /// `attention_bias=true` (Canary-1B-v2) requires the bias vectors to
    /// be present and correctly shaped — dropping either raises a loud
    /// `InvalidArgument`, not a silent zero-fill (FR-EX-08).
    #[test]
    fn asr_new_rejects_missing_qkv_bias_when_encoder_attention_bias_on() {
        let c = CanaryConfig::tiny_for_tests();
        let mut w = CanaryWeights::synthesized(&c, 7).expect("weights");
        w.encoder_blocks[0].qkv_bias = None;
        assert!(matches!(
            CanaryAsr::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn asr_new_rejects_wrong_size_qkv_bias() {
        let c = CanaryConfig::tiny_for_tests();
        let mut w = CanaryWeights::synthesized(&c, 7).expect("weights");
        if let Some(v) = w.encoder_blocks[0].qkv_bias.as_mut() {
            v.pop();
        }
        assert!(matches!(
            CanaryAsr::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn asr_new_rejects_missing_attn_out_bias_when_encoder_attention_bias_on() {
        let c = CanaryConfig::tiny_for_tests();
        let mut w = CanaryWeights::synthesized(&c, 7).expect("weights");
        w.encoder_blocks[0].attn_out_bias = None;
        assert!(matches!(
            CanaryAsr::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn asr_new_rejects_present_qkv_bias_when_encoder_attention_bias_off() {
        // Build a bias-free variant, then splice in a stray bias — the
        // runtime must refuse it (a bias-carrying variant must set
        // encoder.attention_bias=true so the runtime uses them).
        let mut c = CanaryConfig::tiny_for_tests();
        c.encoder.attention_bias = false;
        let mut w = CanaryWeights::synthesized(&c, 7).expect("weights");
        w.encoder_blocks[0].qkv_bias = Some(vec![0.0; 3 * c.encoder.d_model]);
        assert!(matches!(
            CanaryAsr::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn transcribe_rejects_empty_pcm() {
        let c = CanaryConfig::tiny_for_tests();
        let w = CanaryWeights::synthesized(&c, 7).expect("weights");
        let asr = CanaryAsr::new(c, w).expect("canary asr");
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
        let c = CanaryConfig::tiny_for_tests();
        let w = CanaryWeights::synthesized(&c, 7).expect("weights");
        let asr = CanaryAsr::new(c, w).expect("canary asr");
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
    fn expected_arch_is_canary() {
        assert_eq!(EXPECTED_ARCH, "canary");
    }

    #[test]
    fn sample_rate_matches_model_card_boundary() {
        // 16 kHz — per the model card (.wav / .flac mono @ 16 kHz).
        assert_eq!(CANARY_SAMPLE_RATE, 16_000);
    }

    // -----------------------------------------------------------------------
    // Wave 5: `from_gguf` / `from_gguf_with_policy` / `from_path` loud-partial
    // contract (real config validation, arch + provenance surface, license
    // class round-trip, engine constructibility from GGUF, `transcribe` still
    // loud-partials on the synthesized-weight blocker so a follow-up wave has
    // exactly one place to walk — mirror of Wave 4 kyutai_stt / parakeet_ctc
    // precedent).
    // -----------------------------------------------------------------------

    /// Builds a metadata-only Canary GGUF matching the offline converter
    /// (`vokra-convert::models::canary::write_hparams`) so a round trip
    /// yields `cfg` back through [`CanaryConfig::from_gguf`]. If
    /// `weight_license_class` is `Some`, the `vokra.provenance.weight_license`
    /// chunk is stamped; otherwise the reader must fall back to
    /// [`LicenseClass::Unknown`] (fail-closed at the outer M2-13 gate).
    fn build_canary_gguf(
        cfg: &CanaryConfig,
        arch: Option<&str>,
        weight_license_class: Option<LicenseClass>,
    ) -> Vec<u8> {
        use vokra_core::gguf::GgufBuilder;

        let mut b = GgufBuilder::new();
        if let Some(a) = arch {
            b.add_string(chunks::KEY_MODEL_ARCH, a);
        }
        b.add_u32(KEY_SAMPLE_RATE, cfg.sample_rate);
        // Encoder
        b.add_u32(KEY_ENC_N_LAYER, cfg.encoder.n_layer as u32);
        b.add_u32(KEY_ENC_D_MODEL, cfg.encoder.d_model as u32);
        b.add_u32(KEY_ENC_N_HEAD, cfg.encoder.n_head as u32);
        b.add_u32(KEY_ENC_N_HEAD_KV, cfg.encoder.n_head_kv as u32);
        b.add_u32(KEY_ENC_FFN_DIM, cfg.encoder.ffn_dim as u32);
        b.add_u32(KEY_ENC_CONV_KERNEL, cfg.encoder.conv_kernel_size as u32);
        b.add_u32(KEY_ENC_IN_DIM, cfg.encoder.in_dim as u32);
        b.add_u32(
            KEY_ENC_SUBSAMPLING_FACTOR,
            cfg.encoder.subsampling_factor as u32,
        );
        b.add_u32(
            KEY_ENC_SUB_CONV_KERNEL,
            cfg.encoder.subsampling_conv_kernel_size as u32,
        );
        b.add_u32(
            KEY_ENC_SUB_CONV_STRIDE,
            cfg.encoder.subsampling_conv_stride as u32,
        );
        b.add_u32(
            KEY_ENC_SUB_CONV_CHANNELS,
            cfg.encoder.subsampling_conv_channels as u32,
        );
        b.add_u32(KEY_ENC_MAX_POS, cfg.encoder.max_position_embeddings as u32);
        b.add_u32(KEY_ENC_ATTN_BIAS, u32::from(cfg.encoder.attention_bias));
        b.add_u32(KEY_ENC_CONV_BIAS, u32::from(cfg.encoder.convolution_bias));
        b.add_u32(KEY_ENC_SCALE_INPUT, u32::from(cfg.encoder.scale_input));
        // Decoder
        b.add_u32(KEY_DEC_N_LAYER, cfg.decoder.n_layer as u32);
        b.add_u32(KEY_DEC_D_MODEL, cfg.decoder.d_model as u32);
        b.add_u32(KEY_DEC_N_HEAD, cfg.decoder.n_head as u32);
        b.add_u32(KEY_DEC_FFN_DIM, cfg.decoder.ffn_dim as u32);
        b.add_u32(KEY_DEC_MAX_SEQ, cfg.decoder.max_sequence_length as u32);
        b.add_u32(KEY_DEC_PRE_LN, u32::from(cfg.decoder.pre_ln));
        b.add_string(KEY_DEC_HIDDEN_ACT, &cfg.decoder.hidden_act);
        // Head + vocab
        b.add_u32(KEY_HEAD_VOCAB_SIZE, cfg.head.vocab_size as u32);
        b.add_u32(KEY_HEAD_PAD_ID, cfg.head.pad_token_id);
        b.add_u32(KEY_HEAD_BOS_ID, cfg.head.bos_token_id);
        b.add_u32(KEY_HEAD_EOS_ID, cfg.head.eos_token_id);
        if let Some(cls) = weight_license_class {
            b.add_string(chunks::KEY_PROVENANCE_WEIGHT_LICENSE, cls.as_str());
        }
        b.to_bytes().expect("serialize canary fixture GGUF")
    }

    /// A well-formed tiny GGUF binds: arch matches, every
    /// `vokra.canary.*` chunk round-trips, `AttributionRequired`
    /// provenance surfaces, synthesized weights are constructed.
    #[test]
    fn from_gguf_binds_synthesized_from_tiny_config() {
        use vokra_core::gguf::GgufFile;

        let cfg = CanaryConfig::tiny_for_tests();
        let bytes = build_canary_gguf(
            &cfg,
            Some(EXPECTED_ARCH),
            Some(LicenseClass::AttributionRequired),
        );
        let file = GgufFile::parse(bytes).expect("parse fixture");
        let asr = CanaryAsr::from_gguf(&file).expect("valid GGUF must bind");
        assert!(asr.is_synthesized(), "from_gguf binds synthesized bridge");
        assert_eq!(
            asr.weight_license(),
            LicenseClass::AttributionRequired,
            "CC-BY 4.0 = AttributionRequired must surface"
        );
        assert_eq!(asr.config(), &cfg, "config must round-trip verbatim");
    }

    /// A GGUF that omits `vokra.model.arch` entirely fails loud
    /// (converter did not stamp it — the GGUF is not Vokra-native).
    /// Message names the missing arch key + primary source URL.
    #[test]
    fn from_gguf_rejects_missing_arch() {
        use vokra_core::gguf::GgufFile;

        let cfg = CanaryConfig::tiny_for_tests();
        // Build fixture WITHOUT an arch chunk (arch = None).
        let bytes = build_canary_gguf(&cfg, None, Some(LicenseClass::AttributionRequired));
        let file = GgufFile::parse(bytes).expect("parse fixture");
        let Err(err) = CanaryAsr::from_gguf(&file) else {
            panic!("expected ModelLoad on missing arch");
        };
        match err {
            VokraError::ModelLoad(msg) => {
                assert!(
                    msg.contains("vokra.model.arch"),
                    "message must name the missing arch key: {msg}"
                );
                assert!(
                    msg.contains("huggingface.co/nvidia/canary-1b-v2"),
                    "message must name the primary source URL: {msg}"
                );
            }
            other => panic!("expected ModelLoad, got {other:?}"),
        }
    }

    /// A GGUF whose arch is a sibling ASR (`whisper`) fails loud with a
    /// message naming both `canary` and the offending tag + sibling
    /// arches so a reader diagnosing the mis-routed conversion has
    /// exactly one place to walk.
    #[test]
    fn from_gguf_rejects_wrong_arch() {
        use vokra_core::gguf::GgufFile;

        let cfg = CanaryConfig::tiny_for_tests();
        let bytes = build_canary_gguf(
            &cfg,
            Some("whisper"),
            Some(LicenseClass::AttributionRequired),
        );
        let file = GgufFile::parse(bytes).expect("parse fixture");
        let Err(err) = CanaryAsr::from_gguf(&file) else {
            panic!("expected ModelLoad on wrong arch");
        };
        match err {
            VokraError::ModelLoad(msg) => {
                assert!(
                    msg.contains("`whisper`"),
                    "message must name the offending arch tag `whisper`: {msg}"
                );
                assert!(
                    msg.contains(EXPECTED_ARCH),
                    "message must name the expected arch `{EXPECTED_ARCH}`: {msg}"
                );
                assert!(
                    msg.contains("parakeet-ctc") && msg.contains("kyutai-stt"),
                    "message must name sibling ASR arches for disambiguation: {msg}"
                );
                assert!(
                    msg.contains("huggingface.co/nvidia/canary-1b-v2"),
                    "message must name the primary source URL: {msg}"
                );
            }
            other => panic!("expected ModelLoad, got {other:?}"),
        }
    }

    /// Every mandatory `vokra.canary.*` chunk is required — a converter
    /// that fails to stamp any one is a converter bug, not a runtime
    /// silent-default (FR-EX-08). The loud error names the exact absent
    /// chunk key.
    #[test]
    fn from_gguf_rejects_missing_axis() {
        use vokra_core::gguf::{GgufBuilder, GgufFile};

        let cfg = CanaryConfig::tiny_for_tests();
        // Build a hand-crafted GGUF that stamps every axis EXCEPT
        // `KEY_ENC_N_LAYER` — the loud error must fire on `n_layer`.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, EXPECTED_ARCH);
        b.add_u32(KEY_SAMPLE_RATE, cfg.sample_rate);
        // Deliberately omit KEY_ENC_N_LAYER.
        b.add_u32(KEY_ENC_D_MODEL, cfg.encoder.d_model as u32);
        b.add_u32(KEY_ENC_N_HEAD, cfg.encoder.n_head as u32);
        b.add_u32(KEY_ENC_N_HEAD_KV, cfg.encoder.n_head_kv as u32);
        b.add_u32(KEY_ENC_FFN_DIM, cfg.encoder.ffn_dim as u32);
        b.add_u32(KEY_ENC_CONV_KERNEL, cfg.encoder.conv_kernel_size as u32);
        b.add_u32(KEY_ENC_IN_DIM, cfg.encoder.in_dim as u32);
        b.add_u32(
            KEY_ENC_SUBSAMPLING_FACTOR,
            cfg.encoder.subsampling_factor as u32,
        );
        b.add_u32(
            KEY_ENC_SUB_CONV_KERNEL,
            cfg.encoder.subsampling_conv_kernel_size as u32,
        );
        b.add_u32(
            KEY_ENC_SUB_CONV_STRIDE,
            cfg.encoder.subsampling_conv_stride as u32,
        );
        b.add_u32(
            KEY_ENC_SUB_CONV_CHANNELS,
            cfg.encoder.subsampling_conv_channels as u32,
        );
        b.add_u32(KEY_ENC_MAX_POS, cfg.encoder.max_position_embeddings as u32);
        b.add_u32(KEY_ENC_ATTN_BIAS, u32::from(cfg.encoder.attention_bias));
        b.add_u32(KEY_ENC_CONV_BIAS, u32::from(cfg.encoder.convolution_bias));
        b.add_u32(KEY_ENC_SCALE_INPUT, u32::from(cfg.encoder.scale_input));
        b.add_u32(KEY_DEC_N_LAYER, cfg.decoder.n_layer as u32);
        b.add_u32(KEY_DEC_D_MODEL, cfg.decoder.d_model as u32);
        b.add_u32(KEY_DEC_N_HEAD, cfg.decoder.n_head as u32);
        b.add_u32(KEY_DEC_FFN_DIM, cfg.decoder.ffn_dim as u32);
        b.add_u32(KEY_DEC_MAX_SEQ, cfg.decoder.max_sequence_length as u32);
        b.add_u32(KEY_DEC_PRE_LN, u32::from(cfg.decoder.pre_ln));
        b.add_string(KEY_DEC_HIDDEN_ACT, &cfg.decoder.hidden_act);
        b.add_u32(KEY_HEAD_VOCAB_SIZE, cfg.head.vocab_size as u32);
        b.add_u32(KEY_HEAD_PAD_ID, cfg.head.pad_token_id);
        b.add_u32(KEY_HEAD_BOS_ID, cfg.head.bos_token_id);
        b.add_u32(KEY_HEAD_EOS_ID, cfg.head.eos_token_id);
        b.add_string(
            chunks::KEY_PROVENANCE_WEIGHT_LICENSE,
            LicenseClass::AttributionRequired.as_str(),
        );
        let bytes = b.to_bytes().expect("serialize");
        let file = GgufFile::parse(bytes).expect("parse");
        let Err(err) = CanaryAsr::from_gguf(&file) else {
            panic!("expected ModelLoad on missing axis");
        };
        match err {
            VokraError::ModelLoad(msg) => {
                assert!(
                    msg.contains(KEY_ENC_N_LAYER),
                    "message must name the exact missing chunk key `{KEY_ENC_N_LAYER}`: {msg}"
                );
            }
            other => panic!("expected ModelLoad, got {other:?}"),
        }
    }

    /// After a full round-trip via `from_gguf`, `transcribe` still
    /// returns `NotImplemented` naming the synthesized-weight blocker +
    /// the primary source URL (loud-partial contract preserved — the
    /// follow-up wave binds real HF `.nemo` checkpoint tensor names,
    /// T29-equivalent).
    #[test]
    fn from_gguf_transcribe_returns_not_implemented() {
        use vokra_core::gguf::GgufFile;

        let cfg = CanaryConfig::tiny_for_tests();
        let bytes = build_canary_gguf(
            &cfg,
            Some(EXPECTED_ARCH),
            Some(LicenseClass::AttributionRequired),
        );
        let file = GgufFile::parse(bytes).expect("parse fixture");
        let asr = CanaryAsr::from_gguf(&file).expect("valid GGUF must bind");
        assert!(asr.is_synthesized(), "from_gguf builds synthesized weights");
        // 1 second of 16 kHz mono content — legitimate input shape, so
        // the loud-partial gate fires (not the empty-pcm gate).
        let pcm = [0.1_f32; 16_000];
        let err = asr.transcribe(&pcm).unwrap_err();
        match err {
            VokraError::NotImplemented(msg) => {
                assert!(
                    msg.contains("synthesized"),
                    "message must name the synthesized-weight blocker: {msg}"
                );
                assert!(
                    msg.contains("canary-1b-v2"),
                    "message must name the primary source anchor `canary-1b-v2`: {msg}"
                );
            }
            other => panic!("expected NotImplemented, got {other:?}"),
        }
    }

    /// A GGUF that omits `vokra.provenance.weight_license` reads back
    /// as `LicenseClass::Unknown` (fail-closed at the outer M2-13 gate,
    /// matching Parakeet-CTC / MT3 / SNAC precedent).
    #[test]
    fn from_gguf_defaults_weight_license_when_provenance_missing() {
        use vokra_core::gguf::GgufFile;

        let cfg = CanaryConfig::tiny_for_tests();
        // weight_license_class = None → no provenance chunk written.
        let bytes = build_canary_gguf(&cfg, Some(EXPECTED_ARCH), None);
        let file = GgufFile::parse(bytes).expect("parse fixture");
        let asr = CanaryAsr::from_gguf(&file).expect("valid GGUF must bind");
        assert_eq!(
            asr.weight_license(),
            LicenseClass::Unknown,
            "missing provenance must default to Unknown (fail-closed at outer gate)"
        );
    }

    /// [`CanaryAsr::from_path`] uses [`CompliancePolicy::strict`] — a
    /// GGUF advertising `NonCommercial` weight license without a
    /// research opt-in is refused by the M2-13 gate. Guards against a
    /// future silent skip / substitution regression.
    #[test]
    fn from_path_fail_closed_strict_policy() {
        let cfg = CanaryConfig::tiny_for_tests();
        // Stamp NonCommercial weight license — the M2-13 gate under
        // strict must refuse (never a silent skip / substitution).
        let bytes = build_canary_gguf(&cfg, Some(EXPECTED_ARCH), Some(LicenseClass::NonCommercial));
        let path =
            std::env::temp_dir().join(format!("vokra-canary-scout-nc-{}.gguf", std::process::id()));
        std::fs::write(&path, &bytes).expect("write fixture");
        let result = CanaryAsr::from_path(&path);
        // Best-effort cleanup — never a panic on cleanup failure (test
        // determinism must not depend on tmp cleanup).
        let _ = std::fs::remove_file(&path);
        let Err(err) = result else {
            panic!("expected ResearchLicenseRequired on NonCommercial under strict");
        };
        assert!(
            matches!(err, VokraError::ResearchLicenseRequired { .. }),
            "expected ResearchLicenseRequired, got {err:?}"
        );
    }
}
