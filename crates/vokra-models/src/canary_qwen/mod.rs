//! NVIDIA **Canary-Qwen-2.5B** — FastConformer encoder + Qwen LLM
//! decoder (SoTA plan reuse bundle, 2026-07-30).
//!
//! # What Canary-Qwen-2.5B is (primary source)
//!
//! `nvidia/canary-qwen-2.5b` is a 2.5B-parameter multimodal ASR + LLM
//! head-swap on top of NVIDIA's Canary FastConformer encoder. Unlike
//! Canary-1B-v2 (whose decoder is an 8-layer Transformer AED with
//! cross-attention to the encoder), Canary-Qwen replaces the decoder
//! with a **Qwen LLM** (Qwen family — GQA + RoPE + SwiGLU + RMSNorm)
//! that consumes the encoder output as a **soft-prompt prefix** the way
//! the Voxtral text decoder does. The two halves reuse existing Vokra
//! primitives verbatim:
//!
//! - **Encoder** = [`crate::canary::CanaryEncoderConfig`] — the shared
//!   FastConformer body Canary-1B-v2 already ships (`vokra_ops::conformer`
//!   via `Stacking { factor: 8 }`). No new op is introduced.
//! - **Decoder** = [`crate::voxtral::config::TextDecoderConfig`] — the
//!   Voxtral-style Qwen-flavour decoder (GQA / RoPE / SwiGLU / RMSNorm).
//!   Reuses [`crate::voxtral::text_decoder`] end-to-end; the difference
//!   from Voxtral is only the config values (a Qwen 1.7B-ish LM in place
//!   of a Mistral 3B-ish LM).
//!
//! Because the two halves are shared, this module is a **thin config +
//! weight-store shim**. The runtime forward path (Phase-3 real-weight
//! binding) reuses the Canary encoder forward and the Voxtral decoder
//! session; nothing about the topology diverges from what Vokra already
//! implements.
//!
//! # Primary source axes (transcribed verbatim)
//!
//! - `huggingface.co/nvidia/canary-qwen-2.5b` model card (fetched
//!   2026-07-30):
//!   - **Total params**: **2.5 B** (≈ 800M FastConformer encoder + 1.7B
//!     Qwen LM).
//!   - **License**: **CC-BY 4.0** (attribution required — the whole
//!     Canary family ships under CC-BY 4.0; the `canary-` family prefix
//!     walk in `vokra_core::compliance::license_class` already
//!     resolves `canary-qwen*` to `LicenseClass::AttributionRequired`).
//!   - **Encoder** = FastConformer, `32 layers` (same as Canary-1B-v2).
//!   - **Decoder** = Qwen LLM (the release notes name Qwen-2.5-1.7B-
//!     equivalent transformer for the LM head; the Qwen family axes ride
//!     the same primitive as Voxtral / CosyVoice2 / Qwen3-TTS).
//!   - **Sample rate**: **16 kHz** (16 kHz mono, .wav / .flac — inherits
//!     from Canary's FastConformer front-end).
//! - Every hparam **not** stated on the model card is transcribed from
//!   the shared FastConformer-Transformer AED reference config
//!   (`github.com/NVIDIA-NeMo/Speech/blob/main/examples/asr/conf/
//!   speech_multitask/fast-conformer_aed.yaml`, fetched 2026-07-24 —
//!   used by every Canary variant) for the encoder side, and from the
//!   Qwen-family conventions (GQA head split, RoPE θ = 1_000_000,
//!   RMSNorm ε = 1e-6, SwiGLU FFN) for the decoder side. The `.nemo`
//!   tarball's `model_config.yaml` is the ultimate authority; a
//!   follow-up wave (T29-equivalent) inspects it and updates any
//!   transcribed constants that diverge — the runtime shape gate
//!   ([`CanaryQwenConfig::validate_for_forward`]) catches a divergence
//!   loudly on load (FR-EX-08 — never a silent widen).
//!
//! # Decoder hparam provenance — honest placeholder values pending .nemo
//!
//! The Qwen 1.7B-ish LM's exact hparams (`hidden_dim`, `n_layer`,
//! `n_head_q`, `n_head_kv`, `head_dim`, `ffn_dim`, `vocab_size`,
//! `n_ctx`) are **not** enumerated on the HF model card front page and
//! the `.nemo` tarball's `model_config.yaml` is the authoritative source
//! (same posture as Canary-1B-v2's `pad/bos/eos_token_id`).
//! [`CanaryQwenConfig::canary_qwen_2_5b`] carries the **canonical
//! Qwen-family axes** (GQA 16 Q ÷ 8 KV, `head_dim = 128`, `rope_base =
//! 1_000_000`, `rms_norm_eps = 1e-6`, SwiGLU) with `0`-placeholder
//! dims — the runtime validator rejects the `0` sentinels loudly, so a
//! caller *cannot* silently run a hallucinated forward. Bind real
//! Canary-Qwen-2.5B weights (T29-equivalent follow-up) to fill the
//! placeholder dims from the `.nemo` config.
//!
//! # What lands in this reuse-bundle slice
//!
//! - [`CanaryQwenConfig`] — top-level config bundling the shared
//!   [`crate::canary::CanaryEncoderConfig`] (FastConformer encoder,
//!   real values) and a Qwen-flavour decoder config (the Voxtral
//!   [`crate::voxtral::config::TextDecoderConfig`] alias — same shape / same ops).
//! - [`CanaryQwenWeights`] — a scaffold weight store that reuses the
//!   [`crate::canary::CanaryWeights`] encoder layout and adds a Qwen-
//!   flavour decoder scaffold aligned with the Voxtral text-decoder
//!   naming (`text_embed`, per-block `q_proj` / `k_proj` / `v_proj` /
//!   `o_proj` + `gate_proj` / `up_proj` / `down_proj`, RMSNorm γ, plus
//!   a `lm_head`). A deterministic
//!   [`CanaryQwenWeights::synthesized`] fixture (SplitMix64 + Xavier)
//!   exercises shape / dtype / size flow without the real `.nemo`
//!   checkpoint.
//! - [`CanaryQwenAsr`] — engine handle carrying config + weights.
//!   [`CanaryQwenAsr::transcribe`] returns [`VokraError::NotImplemented`]
//!   until real weights are bound and the forward path is wired to the
//!   shared FastConformer encoder + Voxtral text-decoder session
//!   (T29-equivalent follow-up).
//!
//! # No ONNX (permanent)
//!
//! Canary-Qwen ships as a `.nemo` tarball / PyTorch pipeline; the
//! pipeline is re-implemented natively in `vokra-models/src/canary_qwen/`
//! (whisper.cpp 型, CLAUDE.md 設計判断 4). This module never touches
//! ONNX.

use vokra_core::gguf::{GgufFile, chunks};
use vokra_core::rng::SplitMix64;
use vokra_core::{LicenseClass, Result, VokraError};

pub use crate::canary::{CanaryEncoderBlockWeights, CanaryEncoderConfig, CanarySubsampleWeights};
pub use crate::voxtral::config::TextDecoderConfig as CanaryQwenDecoderConfig;

/// `vokra.model.arch` a Canary-Qwen GGUF must carry. Written by
/// `vokra-convert::models::canary_qwen::ARCH`; the compliance registry
/// (`vokra_core::compliance`) resolves `canary-qwen*` to
/// [`vokra_core::LicenseClass::AttributionRequired`] via the `canary-`
/// family prefix walk (CC-BY 4.0 — the M2-13 gate passes commercially
/// *and* the FR-MD-09 attribution surface activates).
///
/// **Intentionally distinct** from `crate::canary::EXPECTED_ARCH`
/// (`"canary"`): silently sharing the base Canary arch tag would
/// mis-route the runtime dispatch to the Transformer AED decoder path
/// instead of the Qwen LLM decoder path.
pub const EXPECTED_ARCH: &str = "canary-qwen";

/// PCM sample rate Canary-Qwen expects — **16 000 Hz**. Inherits from
/// Canary's FastConformer front-end (the model card documents 16 kHz
/// mono WAV / FLAC input).
pub const CANARY_QWEN_SAMPLE_RATE: u32 = 16_000;

// ---------------------------------------------------------------------------
// vokra.canary_qwen.* GGUF metadata keys — mirror of
// `crates/vokra-convert/src/models/canary_qwen.rs` (the cross-crate string
// handshake documented in that converter). Duplicated here rather than
// depending on `vokra-convert` because `vokra-models` sits below the
// converter in the crate graph; the two constants must move together and
// the [`crate::canary_qwen::tests::gguf_keys_match_converter_wire_names`]
// pin catches a rename in either half at test time.
// ---------------------------------------------------------------------------

/// `vokra.canary_qwen.sample_rate` — PCM sample rate stamped by the
/// converter (16 000 Hz for the canonical Canary-Qwen-2.5B release).
pub const GGUF_KEY_SAMPLE_RATE: &str = "vokra.canary_qwen.sample_rate";

// Encoder (FastConformer — Canary-1B-v2 family axes).
/// `vokra.canary_qwen.arch.encoder.n_layer` — FastConformer layer count.
pub const GGUF_KEY_ENC_N_LAYER: &str = "vokra.canary_qwen.arch.encoder.n_layer";
/// `vokra.canary_qwen.arch.encoder.d_model` — encoder hidden width.
pub const GGUF_KEY_ENC_D_MODEL: &str = "vokra.canary_qwen.arch.encoder.d_model";
/// `vokra.canary_qwen.arch.encoder.n_head` — encoder attention Q-head count.
pub const GGUF_KEY_ENC_N_HEAD: &str = "vokra.canary_qwen.arch.encoder.n_head";
/// `vokra.canary_qwen.arch.encoder.n_head_kv` — encoder KV-head count
/// (MHA on Canary encoder, no GQA broadcast).
pub const GGUF_KEY_ENC_N_HEAD_KV: &str = "vokra.canary_qwen.arch.encoder.n_head_kv";
/// `vokra.canary_qwen.arch.encoder.ffn_dim` — encoder FFN inner width.
pub const GGUF_KEY_ENC_FFN_DIM: &str = "vokra.canary_qwen.arch.encoder.ffn_dim";
/// `vokra.canary_qwen.arch.encoder.conv_kernel_size` — FastConformer
/// depthwise conv kernel size (odd, symmetric same-padding).
pub const GGUF_KEY_ENC_CONV_KERNEL: &str = "vokra.canary_qwen.arch.encoder.conv_kernel_size";
/// `vokra.canary_qwen.arch.encoder.in_dim` — log-mel bin count on input.
pub const GGUF_KEY_ENC_IN_DIM: &str = "vokra.canary_qwen.arch.encoder.in_dim";
/// `vokra.canary_qwen.arch.encoder.subsampling_factor` — subsample stem
/// stride product (8× for FastConformer).
pub const GGUF_KEY_ENC_SUBSAMPLING_FACTOR: &str =
    "vokra.canary_qwen.arch.encoder.subsampling_factor";
/// `vokra.canary_qwen.arch.encoder.max_position_embeddings` — upper
/// bound on the subsampled sequence length (rel-pos index).
pub const GGUF_KEY_ENC_MAX_POS: &str = "vokra.canary_qwen.arch.encoder.max_position_embeddings";
/// `vokra.canary_qwen.arch.encoder.attention_bias` — Q/K/V/out
/// projections carry biases (u32 0/1 for GGUF portability, decoded to
/// bool at binder time).
pub const GGUF_KEY_ENC_ATTN_BIAS: &str = "vokra.canary_qwen.arch.encoder.attention_bias";

// Decoder (Qwen LLM — canonical Qwen-family axes).
/// `vokra.canary_qwen.arch.decoder.n_layer` — Qwen LM layer count
/// (0-placeholder in the canonical converter output pending .nemo
/// config extraction).
pub const GGUF_KEY_DEC_N_LAYER: &str = "vokra.canary_qwen.arch.decoder.n_layer";
/// `vokra.canary_qwen.arch.decoder.hidden_dim` — Qwen LM residual width
/// (0-placeholder pending .nemo config extraction).
pub const GGUF_KEY_DEC_HIDDEN_DIM: &str = "vokra.canary_qwen.arch.decoder.hidden_dim";
/// `vokra.canary_qwen.arch.decoder.n_head_q` — Qwen LM Q-head count
/// (Qwen family default 16).
pub const GGUF_KEY_DEC_N_HEAD_Q: &str = "vokra.canary_qwen.arch.decoder.n_head_q";
/// `vokra.canary_qwen.arch.decoder.n_head_kv` — Qwen LM KV-head count
/// (Qwen family default 8 — GQA group ratio 2).
pub const GGUF_KEY_DEC_N_HEAD_KV: &str = "vokra.canary_qwen.arch.decoder.n_head_kv";
/// `vokra.canary_qwen.arch.decoder.head_dim` — Qwen LM per-head width
/// (Qwen family default 128).
pub const GGUF_KEY_DEC_HEAD_DIM: &str = "vokra.canary_qwen.arch.decoder.head_dim";
/// `vokra.canary_qwen.arch.decoder.ffn_dim` — Qwen LM SwiGLU inner width
/// (0-placeholder pending .nemo config extraction).
pub const GGUF_KEY_DEC_FFN_DIM: &str = "vokra.canary_qwen.arch.decoder.ffn_dim";
/// `vokra.canary_qwen.arch.decoder.vocab_size` — Qwen LM SentencePiece
/// vocabulary size (0-placeholder pending .nemo config extraction).
pub const GGUF_KEY_DEC_VOCAB_SIZE: &str = "vokra.canary_qwen.arch.decoder.vocab_size";
/// `vokra.canary_qwen.arch.decoder.n_ctx` — Qwen LM context window
/// (0-placeholder pending .nemo config extraction).
pub const GGUF_KEY_DEC_N_CTX: &str = "vokra.canary_qwen.arch.decoder.n_ctx";
/// `vokra.canary_qwen.arch.decoder.rope_base` — RoPE θ base (Qwen
/// family default 1_000_000.0, `FLOAT32`).
pub const GGUF_KEY_DEC_ROPE_BASE: &str = "vokra.canary_qwen.arch.decoder.rope_base";
/// `vokra.canary_qwen.arch.decoder.rms_norm_eps` — RMSNorm ε (Qwen
/// family default 1e-6, `FLOAT32`).
pub const GGUF_KEY_DEC_RMS_NORM_EPS: &str = "vokra.canary_qwen.arch.decoder.rms_norm_eps";

// Cross-attention / soft-prompt bridge.
/// `vokra.canary_qwen.arch.cross_attn.hidden_dim` — encoder-out width
/// the LM soft-prompt bridge projects from (equals `encoder.d_model` on
/// the canonical release).
pub const GGUF_KEY_CROSS_ATTN_HIDDEN_DIM: &str = "vokra.canary_qwen.arch.cross_attn.hidden_dim";

// ---------------------------------------------------------------------------
// Primary source anchors — cited in the loud-partial transcribe error so
// a reader diagnosing the gap has explicit anchors to walk (redimnet /
// sortformer / RMVPE / pyannote loud-partial-message precedent —
// CLAUDE.md 教訓 (a): "loud-partial は fake-complete より honest").
// ---------------------------------------------------------------------------

/// HuggingFace model-card anchor — the ultimate authority on model id,
/// license, and .nemo distribution format.
const PRIMARY_SOURCE_HF_CARD: &str = "https://huggingface.co/nvidia/canary-qwen-2.5b";
/// Canary family reference config — the shared FastConformer-Transformer
/// AED YAML the runtime encoder axes are transcribed from. Used by every
/// Canary variant (Canary-1B-v2, Canary-Qwen-2.5B, ...).
const PRIMARY_SOURCE_FAMILY_YAML: &str = "github.com/NVIDIA-NeMo/Speech/blob/main/examples/asr/conf/\
     speech_multitask/fast-conformer_aed.yaml";
/// arXiv paper anchor — the FastConformer paper (Rekesh et al., 2023),
/// authoritative for the encoder topology every Canary variant reuses.
const PRIMARY_SOURCE_PAPER: &str = "arxiv.org/abs/2305.05084";

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Full Canary-Qwen config: [`CanaryEncoderConfig`] (real FastConformer
/// axes, transcribed from the Canary family reference) plus a
/// Qwen-flavour [`CanaryQwenDecoderConfig`] (canonical Qwen-family axes
/// with `0`-placeholder dims pending .nemo config extraction).
#[derive(Debug, Clone, PartialEq)]
pub struct CanaryQwenConfig {
    /// FastConformer encoder sub-config — reuses the primary-source
    /// Canary-1B-v2 axes (32 layers × 1024 dim × 8 heads × 128 mel bins).
    pub encoder: CanaryEncoderConfig,
    /// Qwen-flavour decoder sub-config — GQA + RoPE + SwiGLU +
    /// RMSNorm axes (canonical Qwen-family constants; dim / layer / vocab
    /// axes are `0`-placeholders pending .nemo extraction, the validator
    /// rejects `0`).
    pub decoder: CanaryQwenDecoderConfig,
    /// Cross-attention hidden width — for Canary-Qwen the LM consumes
    /// the encoder output as a **soft-prompt prefix** (like Voxtral, not
    /// the Canary-1B-v2 cross-attention decoder), so this field carries
    /// the encoder-out width the LM projects from. Equals
    /// `encoder.d_model` for the canonical release.
    pub cross_attn_hidden_dim: u32,
    /// PCM sample rate — **16 000 Hz** (from the Canary FastConformer
    /// front-end).
    pub sample_rate: u32,
}

impl CanaryQwenConfig {
    /// Primary-source Canary-Qwen-2.5B config. Encoder axes are the real
    /// Canary-1B-v2 constants (model card + family reference); decoder
    /// axes are the canonical Qwen-family constants with `0`-placeholder
    /// dims pending .nemo config extraction. [`Self::validate_for_forward`]
    /// rejects the `0` sentinels so a caller cannot silently run a
    /// hallucinated forward (FR-EX-08).
    #[must_use]
    pub fn canary_qwen_2_5b() -> Self {
        Self {
            encoder: crate::canary::CanaryConfig::canary_1b_v2().encoder,
            decoder: CanaryQwenDecoderConfig {
                // GQA + RoPE + SwiGLU + RMSNorm axes — canonical Qwen-family
                // constants. The Q head split (16), KV head split (8), head
                // dim (128), RoPE base (1_000_000), RMSNorm eps (1e-6) all
                // ride the Qwen family; the `hidden_dim`, `n_layer`,
                // `ffn_dim`, `vocab_size`, `n_ctx` axes are `0`-placeholder
                // sentinels that `validate_for_forward` rejects. Real
                // Canary-Qwen weights land through T29-equivalent .nemo
                // extraction.
                n_layer: 0,
                n_head_q: 16,
                n_head_kv: 8,
                head_dim: 128,
                hidden_dim: 0,
                ffn_dim: 0,
                vocab_size: 0,
                n_ctx: 0,
                rope_base: 1_000_000.0,
                rms_norm_eps: 1e-6,
            },
            cross_attn_hidden_dim: 1024, // = encoder.d_model
            sample_rate: CANARY_QWEN_SAMPLE_RATE,
        }
    }

    /// Miniature well-formed config for shape / stability tests. Dims
    /// are tiny so synthesized-weight builds fit in KB; the shape
    /// relationships (encoder MHA head split, decoder GQA head split,
    /// RoPE even head_dim, encoder → decoder soft-prompt width match)
    /// mirror the real model.
    #[must_use]
    pub fn tiny_for_tests() -> Self {
        Self {
            encoder: crate::canary::CanaryConfig::tiny_for_tests().encoder,
            decoder: CanaryQwenDecoderConfig {
                n_layer: 2,
                n_head_q: 4,
                n_head_kv: 2,
                head_dim: 8,
                hidden_dim: 16,
                ffn_dim: 32,
                vocab_size: 32,
                n_ctx: 64,
                rope_base: 1_000_000.0,
                rms_norm_eps: 1e-6,
            },
            cross_attn_hidden_dim: 16, // = tiny encoder.d_model
            sample_rate: CANARY_QWEN_SAMPLE_RATE,
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
        // Encoder — reuse the well-established Canary encoder validation
        // by wrapping in a full config with a matching-shape decoder
        // sentinel. The Canary validator's decoder branch does not apply
        // here (we use a Qwen decoder), so we do the encoder half
        // inline: reproduce the essential encoder gates directly.
        let e = &self.encoder;
        if !e.is_well_formed() {
            return Err(VokraError::InvalidArgument(format!(
                "canary_qwen config: encoder ill-formed \
                 (n_layer={}, d_model={}, n_head={}, n_head_kv={}) — \
                 expected d_model % n_head == 0, n_head % n_head_kv == 0, \
                 all fields > 0",
                e.n_layer, e.d_model, e.n_head, e.n_head_kv,
            )));
        }
        if e.n_layer == 0 || e.ffn_dim == 0 || e.in_dim == 0 {
            return Err(VokraError::InvalidArgument(
                "canary_qwen config: encoder.n_layer / ffn_dim / in_dim must be > 0".to_owned(),
            ));
        }
        if e.head_dim() % 2 != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "canary_qwen config: encoder head_dim {} must be even",
                e.head_dim(),
            )));
        }
        if e.conv_kernel_size == 0 || e.conv_kernel_size % 2 == 0 {
            return Err(VokraError::InvalidArgument(format!(
                "canary_qwen config: encoder.conv_kernel_size {} must be odd and > 0",
                e.conv_kernel_size,
            )));
        }
        if e.subsampling_factor == 0 {
            return Err(VokraError::InvalidArgument(
                "canary_qwen config: encoder.subsampling_factor must be > 0".to_owned(),
            ));
        }
        if e.max_position_embeddings == 0 {
            return Err(VokraError::InvalidArgument(
                "canary_qwen config: encoder.max_position_embeddings must be > 0".to_owned(),
            ));
        }

        // Decoder — Qwen-family GQA axes.
        let d = &self.decoder;
        if d.n_layer == 0
            || d.hidden_dim == 0
            || d.ffn_dim == 0
            || d.vocab_size == 0
            || d.n_head_q == 0
            || d.n_head_kv == 0
            || d.head_dim == 0
            || d.n_ctx == 0
        {
            return Err(VokraError::InvalidArgument(
                "canary_qwen config: decoder axes must be > 0 (n_layer, hidden_dim, ffn_dim, \
                 vocab_size, n_head_q, n_head_kv, head_dim, n_ctx). Bind real \
                 Canary-Qwen-2.5B weights (T29-equivalent .nemo extraction) to fill the \
                 placeholder dims."
                    .to_owned(),
            ));
        }
        if d.n_head_q % d.n_head_kv != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "canary_qwen config: decoder.n_head_kv ({}) must divide decoder.n_head_q ({})",
                d.n_head_kv, d.n_head_q,
            )));
        }
        if d.head_dim % 2 != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "canary_qwen config: decoder.head_dim ({}) must be even (RoPE pairs)",
                d.head_dim,
            )));
        }
        if !(d.rope_base.is_finite() && d.rope_base > 0.0) {
            return Err(VokraError::InvalidArgument(format!(
                "canary_qwen config: decoder.rope_base ({}) must be positive and finite",
                d.rope_base,
            )));
        }
        if !(d.rms_norm_eps.is_finite() && d.rms_norm_eps > 0.0) {
            return Err(VokraError::InvalidArgument(format!(
                "canary_qwen config: decoder.rms_norm_eps ({}) must be positive and finite",
                d.rms_norm_eps,
            )));
        }

        // Cross-hop consistency: the LM projects from encoder-out width.
        if self.cross_attn_hidden_dim == 0 {
            return Err(VokraError::InvalidArgument(
                "canary_qwen config: cross_attn_hidden_dim must be > 0".to_owned(),
            ));
        }
        if self.cross_attn_hidden_dim as usize != e.d_model {
            return Err(VokraError::InvalidArgument(format!(
                "canary_qwen config: cross_attn_hidden_dim ({}) must equal encoder.d_model ({}) — \
                 the LM soft-prompt bridge projects from the encoder-out width",
                self.cross_attn_hidden_dim, e.d_model,
            )));
        }
        if self.sample_rate == 0 {
            return Err(VokraError::InvalidArgument(
                "canary_qwen config: sample_rate must be > 0".to_owned(),
            ));
        }
        Ok(())
    }

    /// Reads the full 20-axis `vokra.canary_qwen.*` chunk group from
    /// `file`. Encoder axes not stamped by the converter
    /// (`subsampling_conv_kernel_size` / `subsampling_conv_stride` /
    /// `subsampling_conv_channels` / `convolution_bias` / `scale_input`)
    /// inherit the shared Canary family constants — same primary-source
    /// posture as the transcribed encoder axes in
    /// [`Self::canary_qwen_2_5b`] (family constants come from
    /// `fast-conformer_aed.yaml`, the primary-source YAML every Canary
    /// variant reuses; they are stable across the family and the
    /// canary_qwen converter does not stamp them because they never
    /// diverge in-family).
    ///
    /// # Loud-partial posture
    ///
    /// This reader **does not** call [`Self::validate_for_forward`]. The
    /// canonical converter output carries `0`-placeholder decoder dims
    /// (`n_layer` / `hidden_dim` / `ffn_dim` / `vocab_size` / `n_ctx`)
    /// pending `.nemo` config extraction; a strict validate would reject
    /// them and prevent the loud-partial [`CanaryQwenAsr::transcribe`]
    /// arm from firing with a specific error message. The validator is
    /// still callable by consumers that bind real dims later.
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] naming the missing key when any of
    ///   the 20 mandatory `vokra.canary_qwen.*` chunks is absent or of
    ///   the wrong dtype.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let family_encoder = crate::canary::CanaryConfig::canary_1b_v2().encoder;
        let encoder = CanaryEncoderConfig {
            n_layer: req_u32(file, GGUF_KEY_ENC_N_LAYER)? as usize,
            d_model: req_u32(file, GGUF_KEY_ENC_D_MODEL)? as usize,
            n_head: req_u32(file, GGUF_KEY_ENC_N_HEAD)? as usize,
            n_head_kv: req_u32(file, GGUF_KEY_ENC_N_HEAD_KV)? as usize,
            ffn_dim: req_u32(file, GGUF_KEY_ENC_FFN_DIM)? as usize,
            conv_kernel_size: req_u32(file, GGUF_KEY_ENC_CONV_KERNEL)? as usize,
            in_dim: req_u32(file, GGUF_KEY_ENC_IN_DIM)? as usize,
            subsampling_factor: req_u32(file, GGUF_KEY_ENC_SUBSAMPLING_FACTOR)? as usize,
            max_position_embeddings: req_u32(file, GGUF_KEY_ENC_MAX_POS)? as usize,
            attention_bias: req_u32(file, GGUF_KEY_ENC_ATTN_BIAS)? != 0,
            // Family constants — the converter does not stamp these
            // because they never diverge in-family. Primary-source anchor
            // is `fast-conformer_aed.yaml` (same as the transcribed axes
            // in `canary_qwen_2_5b`).
            ..family_encoder
        };

        let decoder = CanaryQwenDecoderConfig {
            n_layer: req_u32(file, GGUF_KEY_DEC_N_LAYER)? as usize,
            hidden_dim: req_u32(file, GGUF_KEY_DEC_HIDDEN_DIM)? as usize,
            n_head_q: req_u32(file, GGUF_KEY_DEC_N_HEAD_Q)? as usize,
            n_head_kv: req_u32(file, GGUF_KEY_DEC_N_HEAD_KV)? as usize,
            head_dim: req_u32(file, GGUF_KEY_DEC_HEAD_DIM)? as usize,
            ffn_dim: req_u32(file, GGUF_KEY_DEC_FFN_DIM)? as usize,
            vocab_size: req_u32(file, GGUF_KEY_DEC_VOCAB_SIZE)? as usize,
            n_ctx: req_u32(file, GGUF_KEY_DEC_N_CTX)? as usize,
            rope_base: req_f32(file, GGUF_KEY_DEC_ROPE_BASE)?,
            rms_norm_eps: req_f32(file, GGUF_KEY_DEC_RMS_NORM_EPS)?,
        };

        Ok(Self {
            encoder,
            decoder,
            cross_attn_hidden_dim: req_u32(file, GGUF_KEY_CROSS_ATTN_HIDDEN_DIM)?,
            sample_rate: req_u32(file, GGUF_KEY_SAMPLE_RATE)?,
        })
    }
}

/// Reads a mandatory `u32`-range integer from `file`. Widens any
/// `U8`/`U16`/`U32`/`U64` payload as long as it fits in `u32`; a
/// signed / float / string / missing value fails loud.
fn req_u32(file: &GgufFile, key: &str) -> Result<u32> {
    let value = file.get(key).ok_or_else(|| {
        VokraError::ModelLoad(format!(
            "canary_qwen: GGUF is missing required u32 chunk `{key}` — the upstream \
             Canary-Qwen-2.5B `.nemo` release is converted through `vokra-cli convert \
             --model canary_qwen`, which stamps the full `vokra.canary_qwen.*` chunk \
             group. This binder refuses to fabricate axes from primary-source constants \
             (FR-EX-08 — no silent partial bind). Re-run the converter against a \
             prepared safetensors checkpoint. Primary source: {source}.",
            source = PRIMARY_SOURCE_HF_CARD
        ))
    })?;
    value
        .as_u64()
        .and_then(|n| u32::try_from(n).ok())
        .ok_or_else(|| {
            VokraError::ModelLoad(format!(
                "canary_qwen: metadata key `{key}` is not a u32-range unsigned integer \
                 (got {value:?}) — the converter always stamps u32 for encoder / \
                 decoder axis counts; a divergent dtype indicates a corrupted or \
                 hand-assembled GGUF (FR-EX-08)."
            ))
        })
}

/// Reads a mandatory `f32` from `file`. Accepts `F32`/`F64` payloads
/// (`F64` narrows to `f32`); anything else fails loud.
#[allow(clippy::cast_possible_truncation)]
fn req_f32(file: &GgufFile, key: &str) -> Result<f32> {
    let value = file.get(key).ok_or_else(|| {
        VokraError::ModelLoad(format!(
            "canary_qwen: GGUF is missing required f32 chunk `{key}` — the upstream \
             Canary-Qwen-2.5B `.nemo` release is converted through `vokra-cli convert \
             --model canary_qwen`, which stamps `rope_base` / `rms_norm_eps` from the \
             canonical Qwen-family constants. This binder refuses to fabricate them \
             (FR-EX-08). Re-run the converter against a prepared safetensors \
             checkpoint. Primary source: {source}.",
            source = PRIMARY_SOURCE_HF_CARD
        ))
    })?;
    value.as_f64().map(|f| f as f32).ok_or_else(|| {
        VokraError::ModelLoad(format!(
            "canary_qwen: metadata key `{key}` is not a float (got {value:?}) — the \
             converter always stamps F32 for `rope_base` / `rms_norm_eps`; a divergent \
             dtype indicates a corrupted or hand-assembled GGUF (FR-EX-08)."
        ))
    })
}

// ---------------------------------------------------------------------------
// Weights
// ---------------------------------------------------------------------------

/// Per-Qwen-decoder-block scaffold weights (pre-norm GQA self-attn +
/// SwiGLU FFN — the Qwen block topology). Mirrors the Voxtral
/// text-decoder block naming so a real binding walks the same slot
/// names.
#[derive(Debug, Clone)]
pub struct CanaryQwenDecoderBlockWeights {
    /// Self-attention pre-norm γ, shape `[hidden_dim]`.
    pub attn_norm: Vec<f32>,
    /// Q projection, shape `[n_head_q * head_dim, hidden_dim]`.
    pub q_proj: Vec<f32>,
    /// K projection, shape `[n_head_kv * head_dim, hidden_dim]` (GQA).
    pub k_proj: Vec<f32>,
    /// V projection, shape `[n_head_kv * head_dim, hidden_dim]` (GQA).
    pub v_proj: Vec<f32>,
    /// O projection, shape `[hidden_dim, n_head_q * head_dim]`.
    pub o_proj: Vec<f32>,
    /// FFN pre-norm γ, shape `[hidden_dim]`.
    pub ffn_norm: Vec<f32>,
    /// SwiGLU gate projection, shape `[ffn_dim, hidden_dim]`.
    pub ffn_gate: Vec<f32>,
    /// SwiGLU up projection, shape `[ffn_dim, hidden_dim]`.
    pub ffn_up: Vec<f32>,
    /// SwiGLU down projection, shape `[hidden_dim, ffn_dim]`.
    pub ffn_down: Vec<f32>,
}

/// Canary-Qwen weight store: shared FastConformer encoder + Qwen LM
/// decoder + soft-prompt bridge from encoder-out to LM hidden width.
#[derive(Debug, Clone)]
pub struct CanaryQwenWeights {
    /// Subsample stem (Canary FastConformer front-end).
    pub subsample: CanarySubsampleWeights,
    /// FastConformer encoder blocks in order.
    pub encoder_blocks: Vec<CanaryEncoderBlockWeights>,
    /// Encoder-out LayerNorm γ, shape `[encoder.d_model]`.
    pub encoder_final_norm: Vec<f32>,
    /// Soft-prompt projection from encoder-out (`encoder.d_model`) to LM
    /// hidden width (`decoder.hidden_dim`). Shape
    /// `[encoder.d_model, decoder.hidden_dim]`.
    pub enc_to_lm_proj: Vec<f32>,
    /// Soft-prompt projection bias, shape `[decoder.hidden_dim]`.
    pub enc_to_lm_proj_bias: Vec<f32>,
    /// Qwen LM token embedding, shape `[decoder.vocab_size, decoder.hidden_dim]`.
    pub lm_token_embed: Vec<f32>,
    /// Qwen LM decoder blocks in order.
    pub decoder_blocks: Vec<CanaryQwenDecoderBlockWeights>,
    /// Final RMSNorm γ, shape `[decoder.hidden_dim]`.
    pub decoder_final_norm: Vec<f32>,
    /// LM head, shape `[decoder.hidden_dim, decoder.vocab_size]`.
    /// Qwen typically ties `lm_head` to `token_embed`; the scaffold
    /// carries a distinct tensor so a real binding can walk either the
    /// tied-weight or the untied-weight path.
    pub lm_head: Vec<f32>,
    /// `true` when built by [`Self::synthesized`] — never a real
    /// upstream checkpoint. Real-checkpoint bindings set this to
    /// `false`.
    pub is_synthesized: bool,
    /// Diagnostic list of tensors discovered on disk by
    /// [`Self::from_gguf`], indexed by upstream `state_dict` name with
    /// their GGUF-side dims. Empty when the weights come from
    /// [`Self::synthesized`] (the synthesized path populates the typed
    /// slots above directly). Used by the load-time non-emptiness gate
    /// and by the future full real-weight binding wave (T29-equivalent
    /// `.nemo` extraction), which will walk this list to fill the typed
    /// slot vectors. Mirrors the `ReDimNetWeights` / `Mt3Weights` /
    /// `SortformerWeights` loud-partial-diagnostic posture.
    tensors: Vec<(String, Vec<usize>)>,
}

impl CanaryQwenWeights {
    /// Builds a deterministic synthesized fixture from `config` and
    /// `seed`.
    ///
    /// Draws are Xavier-uniform ± `sqrt(6 / (fan_in + fan_out))` via a
    /// [`SplitMix64`] stream — reproducible, allocation-only, zero-dep.
    /// Every LayerNorm / RMSNorm γ starts at `1.0`; every bias starts
    /// at `0.0`.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] if `config.validate_for_forward`
    /// fails.
    pub fn synthesized(config: &CanaryQwenConfig, seed: u64) -> Result<Self> {
        config.validate_for_forward()?;
        let mut rng = SplitMix64::new(seed);
        let e = &config.encoder;
        let d = &config.decoder;
        let d_enc = e.d_model;
        let enc_ffn = e.ffn_dim;
        let d_lm = d.hidden_dim;
        let vocab = d.vocab_size;
        let lm_ffn = d.ffn_dim;
        let k = e.conv_kernel_size;
        let bias_on = e.attention_bias;
        let q_out = d.n_head_q * d.head_dim;
        let kv_out = d.n_head_kv * d.head_dim;

        // Subsample stem — flat Linear (Stacking variant).
        let projection_in = e.subsampling_factor * e.in_dim;
        let subsample = CanarySubsampleWeights {
            linear_w: xavier(&mut rng, d_enc * projection_in, projection_in, d_enc),
            linear_b: vec![0.0; d_enc],
        };

        // Encoder blocks — reuse the Canary block layout verbatim.
        let mut encoder_blocks = Vec::with_capacity(e.n_layer);
        for _ in 0..e.n_layer {
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

        // Soft-prompt bridge (encoder-out -> LM hidden).
        let enc_to_lm_proj = xavier(&mut rng, d_enc * d_lm, d_enc, d_lm);
        let enc_to_lm_proj_bias = vec![0.0; d_lm];

        // Qwen LM token embedding.
        let lm_token_embed = xavier(&mut rng, vocab * d_lm, vocab, d_lm);

        // Qwen decoder blocks.
        let mut decoder_blocks = Vec::with_capacity(d.n_layer);
        for _ in 0..d.n_layer {
            decoder_blocks.push(CanaryQwenDecoderBlockWeights {
                attn_norm: vec![1.0; d_lm],
                q_proj: xavier(&mut rng, q_out * d_lm, d_lm, q_out),
                k_proj: xavier(&mut rng, kv_out * d_lm, d_lm, kv_out),
                v_proj: xavier(&mut rng, kv_out * d_lm, d_lm, kv_out),
                o_proj: xavier(&mut rng, d_lm * q_out, q_out, d_lm),
                ffn_norm: vec![1.0; d_lm],
                ffn_gate: xavier(&mut rng, lm_ffn * d_lm, d_lm, lm_ffn),
                ffn_up: xavier(&mut rng, lm_ffn * d_lm, d_lm, lm_ffn),
                ffn_down: xavier(&mut rng, d_lm * lm_ffn, lm_ffn, d_lm),
            });
        }
        let decoder_final_norm = vec![1.0; d_lm];

        // Qwen LM head — untied scaffold slot; a real binding can also
        // resolve this to the token-embedding tensor when the checkpoint
        // uses tied weights.
        let lm_head = xavier(&mut rng, d_lm * vocab, d_lm, vocab);

        Ok(Self {
            subsample,
            encoder_blocks,
            encoder_final_norm,
            enc_to_lm_proj,
            enc_to_lm_proj_bias,
            lm_token_embed,
            decoder_blocks,
            decoder_final_norm,
            lm_head,
            is_synthesized: true,
            tensors: Vec::new(),
        })
    }

    /// Scans `file` for Canary-Qwen state_dict tensors and returns a
    /// weight store that carries only the diagnostic tensor manifest
    /// (typed weight slots are left empty). Refuses to bind if `file`
    /// carries zero tensors (FR-EX-08 — an all-zero forward is never
    /// what the caller wants).
    ///
    /// # Loud-partial posture
    ///
    /// Under the current landing the typed weight slots
    /// (`subsample` / `encoder_blocks` / `decoder_blocks` / ...) stay
    /// empty because the canonical converter output carries
    /// `0`-placeholder decoder dims (`n_layer` / `hidden_dim` /
    /// `ffn_dim` / `vocab_size` / `n_ctx`) pending `.nemo` extraction
    /// — walking upstream tensor names into typed slots against
    /// placeholder shapes would fabricate the layout. The follow-up
    /// wave (T29-equivalent) fills the placeholder dims from the
    /// `.nemo` config and then walks the tensor manifest into the
    /// typed slots. Callers observe the diagnostic tensor count via
    /// [`Self::tensor_count`].
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when `file` carries zero tensors.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let mut tensors: Vec<(String, Vec<usize>)> = Vec::new();
        for info in file.tensors() {
            let dims: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).collect();
            tensors.push((info.name.clone(), dims));
        }
        if tensors.is_empty() {
            return Err(VokraError::ModelLoad(
                "canary_qwen: GGUF carries zero tensors — refusing to bind an all-zero \
                 forward (FR-EX-08). Re-run `vokra-cli convert --model canary_qwen` \
                 against a prepared safetensors checkpoint (the `.nemo` tarball's \
                 PyTorch checkpoint is typically BF16 and passes through the converter \
                 verbatim). Primary source: https://huggingface.co/nvidia/canary-qwen-2.5b."
                    .to_owned(),
            ));
        }

        // Empty typed slots — the real forward path is deferred pending
        // `.nemo` extraction (T29-equivalent). See the rustdoc above +
        // the module doc for the loud-partial posture.
        Ok(Self {
            subsample: CanarySubsampleWeights {
                linear_w: Vec::new(),
                linear_b: Vec::new(),
            },
            encoder_blocks: Vec::new(),
            encoder_final_norm: Vec::new(),
            enc_to_lm_proj: Vec::new(),
            enc_to_lm_proj_bias: Vec::new(),
            lm_token_embed: Vec::new(),
            decoder_blocks: Vec::new(),
            decoder_final_norm: Vec::new(),
            lm_head: Vec::new(),
            is_synthesized: false,
            tensors,
        })
    }

    /// Number of tensors discovered on disk by [`Self::from_gguf`]. `0`
    /// for a [`Self::synthesized`] weight store. Purely a diagnostic
    /// accessor — the real-weight binding wave uses it to size its
    /// expectations.
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }
}

/// Xavier-uniform draw of `count` `f32`s in `[-a, +a]` where
/// `a = sqrt(6 / (fan_in + fan_out))`. Deterministic under a fixed
/// `rng`.
fn xavier(rng: &mut SplitMix64, count: usize, fan_in: usize, fan_out: usize) -> Vec<f32> {
    let a = (6.0 / (fan_in + fan_out).max(1) as f32).sqrt();
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let raw = (rng.next_u64() >> 40) as u32;
        let u01 = (raw as f32) / ((1u32 << 24) as f32);
        out.push((u01 * 2.0 - 1.0) * a);
    }
    out
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

/// Canary-Qwen ASR engine handle.
///
/// Carries the resolved config and weight store. [`Self::transcribe`]
/// is the primary PCM → text entry point; until real weights are bound
/// and the forward path is wired to the shared FastConformer encoder +
/// Voxtral text-decoder session (T29-equivalent), it returns
/// [`VokraError::NotImplemented`] with a message naming the blocker
/// (FR-EX-08 — never a silent zero-fill or empty transcript).
#[derive(Debug, Clone)]
pub struct CanaryQwenAsr {
    cfg: CanaryQwenConfig,
    weights: CanaryQwenWeights,
    weight_license: LicenseClass,
}

impl CanaryQwenAsr {
    /// Assembles an engine from `cfg` and `weights`. Cross-checks the
    /// weight-store shapes against `cfg`.
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] from `cfg.validate_for_forward`.
    /// - [`VokraError::InvalidArgument`] naming the first shape
    ///   mismatch.
    pub fn new(cfg: CanaryQwenConfig, weights: CanaryQwenWeights) -> Result<Self> {
        cfg.validate_for_forward()?;
        let e = &cfg.encoder;
        let d = &cfg.decoder;
        let d_enc = e.d_model;
        let d_lm = d.hidden_dim;
        let vocab = d.vocab_size;
        let projection_in = e.subsampling_factor * e.in_dim;
        let q_out = d.n_head_q * d.head_dim;
        let kv_out = d.n_head_kv * d.head_dim;

        // Subsample stem.
        if weights.subsample.linear_w.len() != d_enc * projection_in {
            return Err(VokraError::InvalidArgument(format!(
                "canary_qwen weights: subsample.linear_w.len()={} != d_enc * (subsampling_factor \
                 * in_dim) = {} * {} = {}",
                weights.subsample.linear_w.len(),
                d_enc,
                projection_in,
                d_enc * projection_in,
            )));
        }
        if weights.subsample.linear_b.len() != d_enc {
            return Err(VokraError::InvalidArgument(format!(
                "canary_qwen weights: subsample.linear_b.len()={} != d_enc={}",
                weights.subsample.linear_b.len(),
                d_enc,
            )));
        }

        // Encoder blocks.
        if weights.encoder_blocks.len() != e.n_layer {
            return Err(VokraError::InvalidArgument(format!(
                "canary_qwen weights: encoder_blocks.len()={} != encoder.n_layer={}",
                weights.encoder_blocks.len(),
                e.n_layer,
            )));
        }
        if weights.encoder_final_norm.len() != d_enc {
            return Err(VokraError::InvalidArgument(format!(
                "canary_qwen weights: encoder_final_norm.len()={} != d_enc={}",
                weights.encoder_final_norm.len(),
                d_enc,
            )));
        }

        // Soft-prompt bridge.
        if weights.enc_to_lm_proj.len() != d_enc * d_lm {
            return Err(VokraError::InvalidArgument(format!(
                "canary_qwen weights: enc_to_lm_proj.len()={} != d_enc * d_lm = {} * {} = {}",
                weights.enc_to_lm_proj.len(),
                d_enc,
                d_lm,
                d_enc * d_lm,
            )));
        }
        if weights.enc_to_lm_proj_bias.len() != d_lm {
            return Err(VokraError::InvalidArgument(format!(
                "canary_qwen weights: enc_to_lm_proj_bias.len()={} != d_lm={}",
                weights.enc_to_lm_proj_bias.len(),
                d_lm,
            )));
        }

        // LM embedding.
        if weights.lm_token_embed.len() != vocab * d_lm {
            return Err(VokraError::InvalidArgument(format!(
                "canary_qwen weights: lm_token_embed.len()={} != vocab * d_lm = {} * {} = {}",
                weights.lm_token_embed.len(),
                vocab,
                d_lm,
                vocab * d_lm,
            )));
        }

        // Decoder blocks.
        if weights.decoder_blocks.len() != d.n_layer {
            return Err(VokraError::InvalidArgument(format!(
                "canary_qwen weights: decoder_blocks.len()={} != decoder.n_layer={}",
                weights.decoder_blocks.len(),
                d.n_layer,
            )));
        }
        for (i, blk) in weights.decoder_blocks.iter().enumerate() {
            for (name, len, expected) in [
                ("attn_norm", blk.attn_norm.len(), d_lm),
                ("q_proj", blk.q_proj.len(), q_out * d_lm),
                ("k_proj", blk.k_proj.len(), kv_out * d_lm),
                ("v_proj", blk.v_proj.len(), kv_out * d_lm),
                ("o_proj", blk.o_proj.len(), d_lm * q_out),
                ("ffn_norm", blk.ffn_norm.len(), d_lm),
                ("ffn_gate", blk.ffn_gate.len(), d.ffn_dim * d_lm),
                ("ffn_up", blk.ffn_up.len(), d.ffn_dim * d_lm),
                ("ffn_down", blk.ffn_down.len(), d_lm * d.ffn_dim),
            ] {
                if len != expected {
                    return Err(VokraError::InvalidArgument(format!(
                        "canary_qwen weights: decoder block {i} `{name}` len={len} != {expected}",
                    )));
                }
            }
        }
        if weights.decoder_final_norm.len() != d_lm {
            return Err(VokraError::InvalidArgument(format!(
                "canary_qwen weights: decoder_final_norm.len()={} != d_lm={}",
                weights.decoder_final_norm.len(),
                d_lm,
            )));
        }
        if weights.lm_head.len() != d_lm * vocab {
            return Err(VokraError::InvalidArgument(format!(
                "canary_qwen weights: lm_head.len()={} != d_lm * vocab = {} * {} = {}",
                weights.lm_head.len(),
                d_lm,
                vocab,
                d_lm * vocab,
            )));
        }

        Ok(Self {
            cfg,
            weights,
            // `new()` does not read a stamped weight-license; the
            // upstream compliance is caller-responsibility here (the
            // caller synthesized weights). `from_gguf` populates this
            // from the GGUF provenance chunk.
            weight_license: LicenseClass::Unknown,
        })
    }

    /// Binds a Canary-Qwen GGUF: validates arch, reads the strict 20-axis
    /// `vokra.canary_qwen.*` chunk group, discovers the tensor manifest,
    /// and surfaces the stamped weight-license class for compliance
    /// gate cross-checks.
    ///
    /// # Loud-partial posture
    ///
    /// This binder deliberately **does not** call
    /// [`CanaryQwenConfig::validate_for_forward`] and **does not**
    /// route through [`Self::new`]'s shape cross-check. The canonical
    /// converter output carries `0`-placeholder decoder dims pending
    /// `.nemo` config extraction, and the typed weight slots (subsample /
    /// encoder_blocks / ...) stay empty on this path. The runtime fires
    /// [`VokraError::NotImplemented`] on [`Self::transcribe`] naming
    /// the primary source, the `.nemo` extraction blocker, and the four
    /// Voxtral-style soft-prompt wiring pieces still owed (log-mel front
    /// end → FastConformer encoder → `enc_to_lm_proj` soft-prompt bridge
    /// → Voxtral-style Qwen decoder session). Same posture as
    /// `ReDimNet::from_gguf` / `SortformerDiar::from_gguf` / `Mt3::from_gguf`
    /// (CLAUDE.md 教訓 (a) — "loud-partial は fake-complete より
    /// honest").
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when `vokra.model.arch` is absent
    ///   or not `"canary-qwen"` (a base `"canary"` GGUF handed here by
    ///   mistake would silently mis-route Canary-1B-v2's Transformer
    ///   AED decoder loader against the Qwen LLM decoder tensor
    ///   manifest — the two topologies are distinct).
    /// - [`VokraError::ModelLoad`] when any of the 20 mandatory
    ///   `vokra.canary_qwen.*` chunks is absent (see
    ///   [`CanaryQwenConfig::from_gguf`]).
    /// - [`VokraError::ModelLoad`] when the GGUF carries zero tensors
    ///   (see [`CanaryQwenWeights::from_gguf`]).
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        // 1. Arch check first so a mis-typed model surfaces a specific
        //    message rather than a downstream "missing key" trail.
        match file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()) {
            Some(a) if a == EXPECTED_ARCH => {}
            Some(other) => {
                return Err(VokraError::ModelLoad(format!(
                    "canary_qwen: GGUF arch is `{other}`, expected `{EXPECTED_ARCH}` \
                     (was this GGUF produced by `vokra-cli convert --model canary_qwen`? \
                     Note that base `canary` GGUFs carry the Canary-1B-v2 **Transformer \
                     AED** decoder tensor manifest — an 8-layer decoder with cross-attn \
                     to the encoder — whereas `canary-qwen` carries the **Qwen LLM** \
                     decoder tensor manifest — GQA + RoPE + SwiGLU + RMSNorm consuming \
                     the encoder-out as a soft-prompt prefix like Voxtral. Silently \
                     sharing the arch tag would mis-route runtime dispatch — FR-EX-08)"
                )));
            }
            None => {
                return Err(VokraError::ModelLoad(
                    "canary_qwen: GGUF is missing `vokra.model.arch` (converter did not \
                     stamp it — this is not a Vokra-native canary_qwen GGUF). Primary \
                     source: https://huggingface.co/nvidia/canary-qwen-2.5b."
                        .to_owned(),
                ));
            }
        }

        // 2. Strict topology axes from the `vokra.canary_qwen.*` chunk
        //    group. Deliberately NOT validate_for_forward'd — the
        //    canonical converter output carries 0-placeholder decoder
        //    dims.
        let cfg = CanaryQwenConfig::from_gguf(file)?;

        // 3. Load the tensor manifest with the non-emptiness gate. Typed
        //    weight slots stay empty on this loud-partial path.
        let weights = CanaryQwenWeights::from_gguf(file)?;

        // 4. Provenance surfacing — read the stamped weight-license
        //    class. The canary_qwen converter stamps
        //    `AttributionRequired` (CC-BY 4.0). A GGUF missing the
        //    stamp reads back as [`LicenseClass::Unknown`] (fail-closed
        //    at the compliance gate).
        let weight_license = file
            .get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
            .and_then(|v| v.as_str())
            .and_then(LicenseClass::from_class_str)
            .unwrap_or(LicenseClass::Unknown);

        Ok(Self {
            cfg,
            weights,
            weight_license,
        })
    }

    /// The resolved configuration.
    #[must_use]
    pub fn config(&self) -> &CanaryQwenConfig {
        &self.cfg
    }

    /// True iff the weight store was built by
    /// [`CanaryQwenWeights::synthesized`] (never a real upstream
    /// checkpoint).
    #[must_use]
    pub fn is_synthesized(&self) -> bool {
        self.weights.is_synthesized
    }

    /// The stamped weight-license class surfaced from the GGUF's
    /// `vokra.provenance.weight_license` chunk. The canary_qwen
    /// converter stamps [`LicenseClass::AttributionRequired`] (CC-BY
    /// 4.0). A GGUF missing the stamp reads back as
    /// [`LicenseClass::Unknown`] (fail-closed).
    #[inline]
    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    /// Number of tensors discovered on disk by [`Self::from_gguf`]. `0`
    /// for a [`Self::new`] / [`CanaryQwenWeights::synthesized`] handle.
    /// Purely a diagnostic accessor — the real-weight binding wave uses
    /// it to size its expectations.
    #[inline]
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.weights.tensor_count()
    }

    /// Transcribes a mono `f32` PCM slice at [`Self::config`]'s sample
    /// rate.
    ///
    /// **Real weights required**: synthesized-weight builds cannot
    /// produce meaningful text (they would be a hallucinated sequence),
    /// so this returns [`VokraError::NotImplemented`] naming the
    /// blocker. Callers verify the shape flow through
    /// [`CanaryQwenAsr::new`] + [`CanaryQwenWeights::synthesized`]
    /// today; a follow-up wave binds the real `.nemo` checkpoint tensor
    /// names and wires the forward (shared FastConformer encoder →
    /// soft-prompt bridge → Voxtral-style Qwen decoder session).
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] if `pcm` is empty.
    /// - [`VokraError::NotImplemented`] otherwise (real forward not yet
    ///   bound — FR-EX-08).
    pub fn transcribe(&self, pcm: &[f32]) -> Result<Vec<u32>> {
        if pcm.is_empty() {
            return Err(VokraError::InvalidArgument(
                "canary_qwen transcribe: pcm slice is empty".to_owned(),
            ));
        }
        if self.weights.is_synthesized {
            return Err(VokraError::NotImplemented(
                "canary_qwen transcribe: this engine holds synthesized weights \
                 (deterministic fixture from CanaryQwenWeights::synthesized) — \
                 synthesized-weight text would be a hallucinated sequence, not a \
                 real transcript. Bind real Canary-Qwen-2.5B weights (CC-BY 4.0, \
                 nvidia/canary-qwen-2.5b — distributed as a .nemo tarball) before \
                 invoking transcribe. The shape flow (config validation, weight-store \
                 construction, PCM boundary check) is exercised through \
                 CanaryQwenAsr::new; the real-checkpoint tensor-name manifest lands \
                 in a follow-up wave (T29-equivalent — Canary-1B-v2 / Voxtral / \
                 CSM / Moshi pattern).",
            ));
        }
        // Loud-partial arm: real weights are bound from a GGUF (via
        // `from_gguf`) but the decoder placeholder dims mark that the
        // `.nemo` extraction has not landed yet, so the typed weight
        // slots stay empty. Fire a distinct NotImplemented naming the
        // primary source + the four wiring pieces still owed (FR-EX-08
        // — no silent fabricated forward on placeholder dims).
        let d = &self.cfg.decoder;
        if d.n_layer == 0
            || d.hidden_dim == 0
            || d.ffn_dim == 0
            || d.vocab_size == 0
            || d.n_ctx == 0
        {
            return Err(VokraError::UnsupportedOp(format!(
                "canary_qwen transcribe: real weights are bound but the decoder \
                 placeholder dims are still 0 pending .nemo config extraction \
                 (decoder.n_layer={n_layer}, hidden_dim={hidden_dim}, \
                 ffn_dim={ffn_dim}, vocab_size={vocab_size}, n_ctx={n_ctx}); the \
                 typed weight slots (subsample / encoder_blocks / decoder_blocks / \
                 lm_head) are empty on the loud-partial from_gguf path. Four \
                 Voxtral-style soft-prompt-prefix wiring pieces are owed: \
                 (i) 128-bin log-mel front-end (STFT + mel filterbank) — reuse \
                 `vokra_ops::waveform_frontend`; \
                 (ii) FastConformer encoder — reuse `vokra_ops::conformer` (shared \
                 with Canary-1B-v2); \
                 (iii) `enc_to_lm_proj` soft-prompt bridge from `encoder.d_model` \
                 to `decoder.hidden_dim`; \
                 (iv) Voxtral-style Qwen decoder session — reuse \
                 `crate::voxtral::text_decoder` with GQA + RoPE + SwiGLU + RMSNorm, \
                 fed the encoder-out as a soft-prompt prefix, plus \
                 `vokra_core::decode::beam_search` with blank_id / bos / eos taken from \
                 the .nemo tokenizer manifest. Bind real Canary-Qwen-2.5B weights \
                 (T29-equivalent .nemo extraction) — primary source: {hf}. \
                 Reference: FastConformer paper — {paper}; family YAML — {yaml}. \
                 Loud-partial (CLAUDE.md 教訓 (a) — 'loud-partial は fake-complete \
                 より honest') — no silent fabricated transcript on placeholder \
                 dims (FR-EX-08). Tensors discovered on disk: {tensor_count}.",
                n_layer = d.n_layer,
                hidden_dim = d.hidden_dim,
                ffn_dim = d.ffn_dim,
                vocab_size = d.vocab_size,
                n_ctx = d.n_ctx,
                hf = PRIMARY_SOURCE_HF_CARD,
                paper = PRIMARY_SOURCE_PAPER,
                yaml = PRIMARY_SOURCE_FAMILY_YAML,
                tensor_count = self.weights.tensor_count(),
            )));
        }
        // Bind unused `pcm` argument so a `#[warn(unused_variables)]`
        // change does not silently mask the loud-partial fire path;
        // the future real implementation consumes it.
        let _ = pcm;
        Err(VokraError::NotImplemented(
            "canary_qwen transcribe: real weights are bound but the log-mel front-end \
             (STFT + mel filterbank) -> FastConformer encoder (vokra_ops::conformer, \
             shared with Canary-1B-v2) -> soft-prompt projection -> Qwen LLM decoder \
             (shared with Voxtral) -> greedy/beam search -> SentencePiece detokenize \
             forward path has not landed yet. Follow-up wave: wire CanaryQwenWeights \
             to the shared ConformerEncoder + Voxtral TextDecoderSession, feeding the \
             encoder-out through enc_to_lm_proj as a soft-prompt prefix and reusing \
             vokra_core::decode::beam_search with blank_id / bos / eos taken from the .nemo \
             tokenizer manifest.",
        ))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expected_arch_is_distinct_from_base_canary() {
        // Sharing the base `"canary"` arch tag would mis-route the runtime
        // dispatch to the Canary-1B-v2 Transformer AED decoder path instead
        // of the Qwen LLM decoder path.
        assert_eq!(EXPECTED_ARCH, "canary-qwen");
        assert_ne!(EXPECTED_ARCH, crate::canary::EXPECTED_ARCH);
    }

    #[test]
    fn sample_rate_matches_canary_frontend() {
        // 16 kHz — inherited from Canary's FastConformer front-end.
        assert_eq!(CANARY_QWEN_SAMPLE_RATE, 16_000);
        assert_eq!(CANARY_QWEN_SAMPLE_RATE, crate::canary::CANARY_SAMPLE_RATE);
    }

    #[test]
    fn canary_qwen_2_5b_carries_real_encoder_axes() {
        // Encoder axes are transcribed verbatim from the Canary-1B-v2 model
        // card + shared FastConformer reference.
        let c = CanaryQwenConfig::canary_qwen_2_5b();
        assert_eq!(c.encoder.n_layer, 32, "model card: 32 encoder layers");
        assert_eq!(c.encoder.d_model, 1024, "family default d_model");
        assert_eq!(c.encoder.n_head, 8, "family default n_heads");
        assert_eq!(c.encoder.n_head_kv, 8, "MHA — no GQA on encoder side");
        assert_eq!(c.encoder.ffn_dim, 4096);
        assert_eq!(c.encoder.in_dim, 128);
        assert_eq!(c.encoder.subsampling_factor, 8);
        assert!(c.encoder.attention_bias);
        assert!(!c.encoder.scale_input);
        assert_eq!(c.cross_attn_hidden_dim, 1024);
        assert_eq!(c.sample_rate, 16_000);
    }

    #[test]
    fn canary_qwen_2_5b_carries_canonical_qwen_family_constants() {
        // The Q head split, KV head split, head_dim, rope_base, rms_norm_eps
        // are canonical Qwen-family constants — bit-identical whether the
        // real LM is Qwen-2.5-1.7B or a future sibling.
        let c = CanaryQwenConfig::canary_qwen_2_5b();
        assert_eq!(c.decoder.n_head_q, 16);
        assert_eq!(c.decoder.n_head_kv, 8, "GQA 16 Q / 8 KV — group ratio 2");
        assert_eq!(c.decoder.head_dim, 128);
        assert!((c.decoder.rope_base - 1_000_000.0).abs() < 1.0);
        assert!((c.decoder.rms_norm_eps - 1e-6).abs() < 1e-12);
    }

    #[test]
    fn canary_qwen_2_5b_rejects_zero_placeholder_dims() {
        // The canonical config carries `0`-placeholder dims (n_layer,
        // hidden_dim, ffn_dim, vocab_size, n_ctx) pending .nemo extraction —
        // validate_for_forward must reject them loudly (FR-EX-08).
        let c = CanaryQwenConfig::canary_qwen_2_5b();
        let err = c
            .validate_for_forward()
            .expect_err("0-placeholder decoder dims must reject");
        match err {
            VokraError::InvalidArgument(msg) => {
                assert!(
                    msg.contains("decoder"),
                    "message must name decoder blocker: {msg}"
                );
            }
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    #[test]
    fn tiny_config_is_well_formed_and_validates() {
        let c = CanaryQwenConfig::tiny_for_tests();
        c.validate_for_forward()
            .expect("tiny config must be well-formed for shape tests");
    }

    #[test]
    fn config_rejects_gqa_non_divisor() {
        let mut c = CanaryQwenConfig::tiny_for_tests();
        c.decoder.n_head_kv = 3; // 4 % 3 != 0
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_rejects_odd_decoder_head_dim() {
        let mut c = CanaryQwenConfig::tiny_for_tests();
        c.decoder.head_dim = 7;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_rejects_non_finite_rope_base() {
        let mut c = CanaryQwenConfig::tiny_for_tests();
        c.decoder.rope_base = f32::NAN;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_rejects_cross_attn_width_mismatch() {
        // The soft-prompt bridge projects from encoder-out width; a
        // mismatch here would silently mis-slot the projection at load
        // time — reject loudly at validate.
        let mut c = CanaryQwenConfig::tiny_for_tests();
        c.cross_attn_hidden_dim = c.encoder.d_model as u32 + 1;
        let err = c
            .validate_for_forward()
            .expect_err("cross-attn width mismatch must reject");
        match err {
            VokraError::InvalidArgument(msg) => {
                assert!(msg.contains("cross_attn_hidden_dim"));
                assert!(msg.contains("encoder.d_model"));
            }
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    #[test]
    fn synthesized_weights_are_deterministic_and_shape_correct() {
        let c = CanaryQwenConfig::tiny_for_tests();
        let w1 = CanaryQwenWeights::synthesized(&c, 0x42).expect("build 1");
        let w2 = CanaryQwenWeights::synthesized(&c, 0x42).expect("build 2");
        // Determinism on both encoder + decoder sides.
        assert_eq!(w1.subsample.linear_w, w2.subsample.linear_w);
        assert_eq!(w1.encoder_blocks[0].qkv_proj, w2.encoder_blocks[0].qkv_proj);
        assert_eq!(w1.enc_to_lm_proj, w2.enc_to_lm_proj);
        assert_eq!(w1.lm_token_embed, w2.lm_token_embed);
        assert_eq!(w1.decoder_blocks[0].q_proj, w2.decoder_blocks[0].q_proj);
        assert_eq!(w1.lm_head, w2.lm_head);
        assert!(w1.is_synthesized);

        // Shape flow — encoder side (mirrors the Canary encoder layout).
        let e = &c.encoder;
        let d = &c.decoder;
        let projection_in = e.subsampling_factor * e.in_dim;
        assert_eq!(w1.subsample.linear_w.len(), e.d_model * projection_in);
        assert_eq!(w1.subsample.linear_b.len(), e.d_model);
        assert_eq!(w1.encoder_blocks.len(), e.n_layer);
        assert_eq!(w1.encoder_final_norm.len(), e.d_model);
        for blk in &w1.encoder_blocks {
            assert_eq!(blk.qkv_proj.len(), e.d_model * 3 * e.d_model);
            assert_eq!(blk.attn_out.len(), e.d_model * e.d_model);
            assert_eq!(blk.ff1_fc1.len(), e.d_model * e.ffn_dim);
            assert_eq!(blk.final_norm.len(), e.d_model);
        }

        // Shape flow — soft-prompt bridge + decoder side.
        assert_eq!(w1.enc_to_lm_proj.len(), e.d_model * d.hidden_dim);
        assert_eq!(w1.enc_to_lm_proj_bias.len(), d.hidden_dim);
        assert_eq!(w1.lm_token_embed.len(), d.vocab_size * d.hidden_dim);
        assert_eq!(w1.decoder_blocks.len(), d.n_layer);
        let q_out = d.n_head_q * d.head_dim;
        let kv_out = d.n_head_kv * d.head_dim;
        for blk in &w1.decoder_blocks {
            assert_eq!(blk.q_proj.len(), q_out * d.hidden_dim);
            assert_eq!(blk.k_proj.len(), kv_out * d.hidden_dim);
            assert_eq!(blk.v_proj.len(), kv_out * d.hidden_dim);
            assert_eq!(blk.o_proj.len(), d.hidden_dim * q_out);
            assert_eq!(blk.ffn_gate.len(), d.ffn_dim * d.hidden_dim);
            assert_eq!(blk.ffn_up.len(), d.ffn_dim * d.hidden_dim);
            assert_eq!(blk.ffn_down.len(), d.hidden_dim * d.ffn_dim);
        }
        assert_eq!(w1.decoder_final_norm.len(), d.hidden_dim);
        assert_eq!(w1.lm_head.len(), d.hidden_dim * d.vocab_size);
    }

    #[test]
    fn synthesized_weights_different_seeds_diverge() {
        let c = CanaryQwenConfig::tiny_for_tests();
        let w_a = CanaryQwenWeights::synthesized(&c, 1).expect("a");
        let w_b = CanaryQwenWeights::synthesized(&c, 2).expect("b");
        assert_ne!(w_a.lm_token_embed, w_b.lm_token_embed);
        assert_ne!(
            w_a.encoder_blocks[0].qkv_proj,
            w_b.encoder_blocks[0].qkv_proj
        );
    }

    #[test]
    fn synthesized_rejects_ill_formed_config() {
        let mut c = CanaryQwenConfig::tiny_for_tests();
        c.decoder.n_head_kv = 3; // 4 % 3 != 0
        assert!(matches!(
            CanaryQwenWeights::synthesized(&c, 7),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn asr_new_accepts_matching_config_and_weights() {
        let c = CanaryQwenConfig::tiny_for_tests();
        let w = CanaryQwenWeights::synthesized(&c, 7).expect("weights");
        let asr = CanaryQwenAsr::new(c.clone(), w).expect("asr");
        assert_eq!(asr.config().encoder.d_model, c.encoder.d_model);
        assert_eq!(asr.config().decoder.hidden_dim, c.decoder.hidden_dim);
        assert!(asr.is_synthesized());
    }

    #[test]
    fn asr_new_rejects_encoder_layer_count_mismatch() {
        let c = CanaryQwenConfig::tiny_for_tests();
        let mut w = CanaryQwenWeights::synthesized(&c, 7).expect("weights");
        w.encoder_blocks.pop();
        assert!(matches!(
            CanaryQwenAsr::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn asr_new_rejects_decoder_layer_count_mismatch() {
        let c = CanaryQwenConfig::tiny_for_tests();
        let mut w = CanaryQwenWeights::synthesized(&c, 7).expect("weights");
        w.decoder_blocks.pop();
        assert!(matches!(
            CanaryQwenAsr::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn asr_new_rejects_enc_to_lm_proj_size_mismatch() {
        let c = CanaryQwenConfig::tiny_for_tests();
        let mut w = CanaryQwenWeights::synthesized(&c, 7).expect("weights");
        w.enc_to_lm_proj.pop();
        assert!(matches!(
            CanaryQwenAsr::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn asr_new_rejects_lm_token_embed_size_mismatch() {
        let c = CanaryQwenConfig::tiny_for_tests();
        let mut w = CanaryQwenWeights::synthesized(&c, 7).expect("weights");
        w.lm_token_embed.pop();
        assert!(matches!(
            CanaryQwenAsr::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn asr_new_rejects_lm_head_size_mismatch() {
        let c = CanaryQwenConfig::tiny_for_tests();
        let mut w = CanaryQwenWeights::synthesized(&c, 7).expect("weights");
        w.lm_head.pop();
        assert!(matches!(
            CanaryQwenAsr::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn asr_new_rejects_decoder_q_proj_size_mismatch() {
        let c = CanaryQwenConfig::tiny_for_tests();
        let mut w = CanaryQwenWeights::synthesized(&c, 7).expect("weights");
        w.decoder_blocks[0].q_proj.pop();
        assert!(matches!(
            CanaryQwenAsr::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn transcribe_rejects_empty_pcm() {
        let c = CanaryQwenConfig::tiny_for_tests();
        let w = CanaryQwenWeights::synthesized(&c, 7).expect("weights");
        let asr = CanaryQwenAsr::new(c, w).expect("asr");
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
        let c = CanaryQwenConfig::tiny_for_tests();
        let w = CanaryQwenWeights::synthesized(&c, 7).expect("weights");
        let asr = CanaryQwenAsr::new(c, w).expect("asr");
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

    // -----------------------------------------------------------------------
    // Loud-partial from_gguf tests (redimnet / sortformer / mt3 precedent —
    // CLAUDE.md 教訓 (a): "loud-partial は fake-complete より honest").
    // -----------------------------------------------------------------------

    use vokra_core::gguf::{GgmlType, GgufBuilder, GgufFile};

    /// Builds a minimal Canary-Qwen GGUF carrying the arch tag + the
    /// full 20-axis `vokra.canary_qwen.*` chunk group (mirroring
    /// `crates/vokra-convert/src/models/canary_qwen.rs`'s stamped
    /// constants) + optionally one representative tensor + optional
    /// weight-license class stamp.
    fn canary_qwen_gguf(add_tensor: bool, license_class: Option<LicenseClass>) -> GgufFile {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, EXPECTED_ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, "canary-qwen-2.5b");
        // Encoder axes — mirror of the canary_qwen converter's transcribed
        // Canary-1B-v2 FastConformer defaults.
        b.add_u32(GGUF_KEY_ENC_N_LAYER, 32);
        b.add_u32(GGUF_KEY_ENC_D_MODEL, 1024);
        b.add_u32(GGUF_KEY_ENC_N_HEAD, 8);
        b.add_u32(GGUF_KEY_ENC_N_HEAD_KV, 8);
        b.add_u32(GGUF_KEY_ENC_FFN_DIM, 4096);
        b.add_u32(GGUF_KEY_ENC_CONV_KERNEL, 9);
        b.add_u32(GGUF_KEY_ENC_IN_DIM, 128);
        b.add_u32(GGUF_KEY_ENC_SUBSAMPLING_FACTOR, 8);
        b.add_u32(GGUF_KEY_ENC_MAX_POS, 5000);
        b.add_u32(GGUF_KEY_ENC_ATTN_BIAS, 1);
        // Decoder axes — canonical Qwen-family constants + 0-placeholder
        // dims pending `.nemo` extraction (mirror of the canary_qwen
        // converter's transcribed constants).
        b.add_u32(GGUF_KEY_DEC_N_LAYER, 0);
        b.add_u32(GGUF_KEY_DEC_HIDDEN_DIM, 0);
        b.add_u32(GGUF_KEY_DEC_N_HEAD_Q, 16);
        b.add_u32(GGUF_KEY_DEC_N_HEAD_KV, 8);
        b.add_u32(GGUF_KEY_DEC_HEAD_DIM, 128);
        b.add_u32(GGUF_KEY_DEC_FFN_DIM, 0);
        b.add_u32(GGUF_KEY_DEC_VOCAB_SIZE, 0);
        b.add_u32(GGUF_KEY_DEC_N_CTX, 0);
        b.add_f32(GGUF_KEY_DEC_ROPE_BASE, 1_000_000.0);
        b.add_f32(GGUF_KEY_DEC_RMS_NORM_EPS, 1e-6);
        // Cross-attention hidden dim = encoder d_model.
        b.add_u32(GGUF_KEY_CROSS_ATTN_HIDDEN_DIM, 1024);
        // Sample rate.
        b.add_u32(GGUF_KEY_SAMPLE_RATE, 16_000);
        // Provenance stamp — canary_qwen converter stamps
        // AttributionRequired (CC-BY 4.0).
        if let Some(cls) = license_class {
            b.add_string(chunks::KEY_PROVENANCE_WEIGHT_LICENSE, cls.as_str());
        }
        if add_tensor {
            // One representative BF16 tensor so the non-emptiness gate
            // passes. Name mirrors the Qwen LM decoder q_proj slot the
            // future real binding would walk.
            b.add_tensor(
                "decoder.model.layers.0.self_attn.q_proj.weight",
                GgmlType::BF16,
                vec![4, 4],
                vec![0u8; 4 * 4 * 2],
            )
            .expect("add_tensor");
        }
        GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse")
    }

    /// Pins the arch-tag distinctness. A `canary` GGUF handed to the
    /// `canary_qwen` binder by mistake must fail loud with a specific
    /// message rather than silently mis-routing Canary-1B-v2's
    /// Transformer AED decoder loader against the Qwen LLM decoder
    /// tensor manifest (FR-EX-08).
    #[test]
    fn from_gguf_rejects_wrong_arch() {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, "canary");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = CanaryQwenAsr::from_gguf(&file) else {
            panic!("expected ModelLoad on wrong arch");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("`canary`") && m.contains("`canary-qwen`"),
                    "message must name both the got + expected arch tags, got `{m}`"
                );
                assert!(
                    m.contains("Transformer AED") && m.contains("Qwen LLM"),
                    "message must disambiguate the two decoder topologies so a \
                     reader diagnosing the mis-route knows why the arch tags are \
                     distinct, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    /// A GGUF that never stamped `vokra.model.arch` at all must fail
    /// loud too (not silently pass through to the chunk-group reader).
    #[test]
    fn from_gguf_rejects_missing_arch() {
        let b = GgufBuilder::new();
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = CanaryQwenAsr::from_gguf(&file) else {
            panic!("expected ModelLoad on missing arch");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("missing `vokra.model.arch`"),
                    "message must name the missing arch chunk: {m}"
                );
                assert!(
                    m.contains("canary-qwen-2.5b"),
                    "message must cite the primary source anchor: {m}"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    /// The strict encoder-side chunk reader — dropping any of the 10
    /// stamped encoder axes fails with a `ModelLoad` naming the exact
    /// key (no primary-source fallback that would fabricate axes,
    /// FR-EX-08). Uses `n_layer` as a representative case.
    #[test]
    fn from_gguf_rejects_missing_encoder_chunk() {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, EXPECTED_ARCH);
        // deliberately omit GGUF_KEY_ENC_N_LAYER
        b.add_u32(GGUF_KEY_ENC_D_MODEL, 1024);
        b.add_u32(GGUF_KEY_ENC_N_HEAD, 8);
        b.add_u32(GGUF_KEY_ENC_N_HEAD_KV, 8);
        b.add_u32(GGUF_KEY_ENC_FFN_DIM, 4096);
        b.add_u32(GGUF_KEY_ENC_CONV_KERNEL, 9);
        b.add_u32(GGUF_KEY_ENC_IN_DIM, 128);
        b.add_u32(GGUF_KEY_ENC_SUBSAMPLING_FACTOR, 8);
        b.add_u32(GGUF_KEY_ENC_MAX_POS, 5000);
        b.add_u32(GGUF_KEY_ENC_ATTN_BIAS, 1);
        b.add_u32(GGUF_KEY_DEC_N_LAYER, 0);
        b.add_u32(GGUF_KEY_DEC_HIDDEN_DIM, 0);
        b.add_u32(GGUF_KEY_DEC_N_HEAD_Q, 16);
        b.add_u32(GGUF_KEY_DEC_N_HEAD_KV, 8);
        b.add_u32(GGUF_KEY_DEC_HEAD_DIM, 128);
        b.add_u32(GGUF_KEY_DEC_FFN_DIM, 0);
        b.add_u32(GGUF_KEY_DEC_VOCAB_SIZE, 0);
        b.add_u32(GGUF_KEY_DEC_N_CTX, 0);
        b.add_f32(GGUF_KEY_DEC_ROPE_BASE, 1_000_000.0);
        b.add_f32(GGUF_KEY_DEC_RMS_NORM_EPS, 1e-6);
        b.add_u32(GGUF_KEY_CROSS_ATTN_HIDDEN_DIM, 1024);
        b.add_u32(GGUF_KEY_SAMPLE_RATE, 16_000);
        b.add_tensor(
            "decoder.model.layers.0.self_attn.q_proj.weight",
            GgmlType::F32,
            vec![4, 4],
            vec![0u8; 16 * 4],
        )
        .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = CanaryQwenAsr::from_gguf(&file) else {
            panic!("expected ModelLoad on missing encoder chunk");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains(GGUF_KEY_ENC_N_LAYER),
                    "message must name the missing key exactly, got `{m}`"
                );
                assert!(
                    m.contains("FR-EX-08") || m.contains("silent partial bind"),
                    "message should cite the FR-EX-08 no-silent-fabrication clause: `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    /// The strict decoder-side f32 reader — dropping `rope_base` (a
    /// stamped f32 axis) must fail loud with a `ModelLoad` naming the
    /// exact key (verifies the f32 code path, complementing the u32
    /// coverage above).
    #[test]
    fn from_gguf_rejects_missing_decoder_rope_base() {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, EXPECTED_ARCH);
        // Full encoder chunk group.
        b.add_u32(GGUF_KEY_ENC_N_LAYER, 32);
        b.add_u32(GGUF_KEY_ENC_D_MODEL, 1024);
        b.add_u32(GGUF_KEY_ENC_N_HEAD, 8);
        b.add_u32(GGUF_KEY_ENC_N_HEAD_KV, 8);
        b.add_u32(GGUF_KEY_ENC_FFN_DIM, 4096);
        b.add_u32(GGUF_KEY_ENC_CONV_KERNEL, 9);
        b.add_u32(GGUF_KEY_ENC_IN_DIM, 128);
        b.add_u32(GGUF_KEY_ENC_SUBSAMPLING_FACTOR, 8);
        b.add_u32(GGUF_KEY_ENC_MAX_POS, 5000);
        b.add_u32(GGUF_KEY_ENC_ATTN_BIAS, 1);
        b.add_u32(GGUF_KEY_DEC_N_LAYER, 0);
        b.add_u32(GGUF_KEY_DEC_HIDDEN_DIM, 0);
        b.add_u32(GGUF_KEY_DEC_N_HEAD_Q, 16);
        b.add_u32(GGUF_KEY_DEC_N_HEAD_KV, 8);
        b.add_u32(GGUF_KEY_DEC_HEAD_DIM, 128);
        b.add_u32(GGUF_KEY_DEC_FFN_DIM, 0);
        b.add_u32(GGUF_KEY_DEC_VOCAB_SIZE, 0);
        b.add_u32(GGUF_KEY_DEC_N_CTX, 0);
        // deliberately omit GGUF_KEY_DEC_ROPE_BASE
        b.add_f32(GGUF_KEY_DEC_RMS_NORM_EPS, 1e-6);
        b.add_u32(GGUF_KEY_CROSS_ATTN_HIDDEN_DIM, 1024);
        b.add_u32(GGUF_KEY_SAMPLE_RATE, 16_000);
        b.add_tensor(
            "decoder.model.layers.0.self_attn.q_proj.weight",
            GgmlType::F32,
            vec![4, 4],
            vec![0u8; 16 * 4],
        )
        .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = CanaryQwenAsr::from_gguf(&file) else {
            panic!("expected ModelLoad on missing decoder rope_base");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains(GGUF_KEY_DEC_ROPE_BASE),
                    "message must name the missing f32 key exactly, got `{m}`"
                );
                assert!(
                    m.contains("rope_base") || m.contains("rms_norm_eps"),
                    "message should reference the Qwen-family f32 constants context: `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    /// The non-emptiness gate on the tensor manifest — a chunk-group
    /// GGUF that carries zero tensors must fail loud rather than
    /// silently binding an all-zero forward (FR-EX-08).
    #[test]
    fn from_gguf_rejects_zero_tensors() {
        let file = canary_qwen_gguf(false, Some(LicenseClass::AttributionRequired));
        let Err(err) = CanaryQwenAsr::from_gguf(&file) else {
            panic!("expected ModelLoad on empty tensor manifest");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("zero tensors"),
                    "message must name the empty-manifest blocker: `{m}`"
                );
                assert!(
                    m.contains("FR-EX-08"),
                    "message must cite the FR-EX-08 clause: `{m}`"
                );
                assert!(
                    m.contains("canary-qwen-2.5b"),
                    "message must cite the primary source anchor: `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    /// End-to-end round-trip: a canonical canary_qwen GGUF binds through
    /// `CanaryQwenAsr::from_gguf`, echoes the stamped config axes
    /// verbatim (encoder = Canary-1B-v2 real values; decoder = Qwen
    /// family constants + 0-placeholders), surfaces the stamped
    /// weight-license class, and reports the tensor count. The
    /// downstream `transcribe` call fires the loud-partial arm because
    /// the decoder placeholder dims are still 0.
    #[test]
    fn from_gguf_binds_converter_output_verbatim() {
        let file = canary_qwen_gguf(true, Some(LicenseClass::AttributionRequired));
        let asr = CanaryQwenAsr::from_gguf(&file).expect("valid GGUF must bind");
        // Encoder axes round-trip.
        assert_eq!(asr.config().encoder.n_layer, 32);
        assert_eq!(asr.config().encoder.d_model, 1024);
        assert_eq!(asr.config().encoder.n_head, 8);
        assert_eq!(asr.config().encoder.n_head_kv, 8);
        assert_eq!(asr.config().encoder.ffn_dim, 4096);
        assert_eq!(asr.config().encoder.conv_kernel_size, 9);
        assert_eq!(asr.config().encoder.in_dim, 128);
        assert_eq!(asr.config().encoder.subsampling_factor, 8);
        assert_eq!(asr.config().encoder.max_position_embeddings, 5000);
        assert!(
            asr.config().encoder.attention_bias,
            "encoder.attention_bias u32 flag must decode to true"
        );
        // Decoder axes round-trip — 0-placeholders preserved because
        // the loud-partial from_gguf skips validate_for_forward.
        assert_eq!(asr.config().decoder.n_layer, 0);
        assert_eq!(asr.config().decoder.hidden_dim, 0);
        assert_eq!(asr.config().decoder.n_head_q, 16);
        assert_eq!(asr.config().decoder.n_head_kv, 8);
        assert_eq!(asr.config().decoder.head_dim, 128);
        assert_eq!(asr.config().decoder.ffn_dim, 0);
        assert_eq!(asr.config().decoder.vocab_size, 0);
        assert_eq!(asr.config().decoder.n_ctx, 0);
        assert!((asr.config().decoder.rope_base - 1_000_000.0).abs() < 1.0);
        assert!((asr.config().decoder.rms_norm_eps - 1e-6).abs() < 1e-9);
        // Cross-attention hidden dim = encoder d_model.
        assert_eq!(asr.config().cross_attn_hidden_dim, 1024);
        assert_eq!(asr.config().sample_rate, 16_000);
        // Weight-license surface (canary_qwen converter stamps
        // AttributionRequired per CC-BY 4.0).
        assert_eq!(asr.weight_license(), LicenseClass::AttributionRequired);
        // Non-empty tensor manifest.
        assert!(asr.tensor_count() >= 1);
        // Real weights posture: from_gguf is not the synthesized path.
        assert!(!asr.is_synthesized());
    }

    /// Verify the loud-partial transcribe arm fires with the correct
    /// primary-source URLs and names all four Voxtral-style wiring
    /// pieces still owed. The message must be actionable enough that
    /// the follow-up wave has exactly four things to walk.
    #[test]
    fn transcribe_loud_partial_names_primary_source_and_forward_wiring() {
        let file = canary_qwen_gguf(true, Some(LicenseClass::AttributionRequired));
        let asr = CanaryQwenAsr::from_gguf(&file).expect("valid GGUF must bind");
        let pcm = vec![0.0f32; 16_000]; // 1 second at 16 kHz — legitimate PCM shape
        let Err(err) = asr.transcribe(&pcm) else {
            panic!("transcribe must fire loud-partial arm on placeholder decoder dims");
        };
        match err {
            // UnsupportedOp is used for the placeholder-dims arm because it
            // carries `String`; `NotImplemented` is `&'static str` and cannot
            // format the per-call decoder axes. Both variants are honest
            // loud-partials per CLAUDE.md 教訓 (a).
            VokraError::UnsupportedOp(msg) => {
                // Primary source URL.
                assert!(
                    msg.contains("huggingface.co/nvidia/canary-qwen-2.5b"),
                    "message must cite the HF primary source URL: `{msg}`"
                );
                // .nemo extraction as concrete blocker.
                assert!(
                    msg.contains(".nemo"),
                    "message must name the .nemo extraction blocker: `{msg}`"
                );
                // FastConformer paper anchor (arXiv reference).
                assert!(
                    msg.contains("arxiv.org"),
                    "message must cite an arXiv reference: `{msg}`"
                );
                // All four Voxtral-style soft-prompt-prefix wiring pieces.
                assert!(
                    msg.contains("log-mel"),
                    "wiring piece (i): 128-bin log-mel front-end: `{msg}`"
                );
                assert!(
                    msg.contains("FastConformer") && msg.contains("conformer"),
                    "wiring piece (ii): FastConformer encoder: `{msg}`"
                );
                assert!(
                    msg.contains("soft-prompt") && msg.contains("enc_to_lm_proj"),
                    "wiring piece (iii): enc_to_lm_proj soft-prompt bridge: `{msg}`"
                );
                assert!(
                    msg.contains("Qwen") && msg.contains("Voxtral"),
                    "wiring piece (iv): Voxtral-style Qwen decoder session: `{msg}`"
                );
                // FR-EX-08 clause + honest-partial rationale.
                assert!(
                    msg.contains("FR-EX-08"),
                    "message must cite the FR-EX-08 no-silent-fabricated-forward clause: `{msg}`"
                );
                assert!(
                    msg.contains("loud-partial") || msg.contains("教訓"),
                    "message should cite the CLAUDE.md 教訓 (a) honesty rationale: `{msg}`"
                );
                // Every decoder placeholder axis echoed so the reader
                // sees exactly why the arm fired.
                assert!(
                    msg.contains("decoder.n_layer=0"),
                    "placeholder axis must be echoed: `{msg}`"
                );
                assert!(
                    msg.contains("hidden_dim=0"),
                    "placeholder axis must be echoed: `{msg}`"
                );
                assert!(
                    msg.contains("vocab_size=0"),
                    "placeholder axis must be echoed: `{msg}`"
                );
            }
            other => panic!("expected VokraError::UnsupportedOp, got {other:?}"),
        }
    }

    /// Missing provenance stamp defaults to `LicenseClass::Unknown`
    /// (fail-closed at the compliance gate) rather than silently
    /// assuming a permissive class. Complements the round-trip test
    /// above which stamps `AttributionRequired` explicitly.
    #[test]
    fn from_gguf_missing_provenance_defaults_to_unknown() {
        let file = canary_qwen_gguf(true, None);
        let asr = CanaryQwenAsr::from_gguf(&file).expect("valid GGUF must bind");
        assert_eq!(
            asr.weight_license(),
            LicenseClass::Unknown,
            "missing provenance stamp must fail-closed to Unknown at the compliance gate"
        );
    }

    /// Cross-crate string handshake pin: the runtime GGUF key constants
    /// must match the wire names the converter stamps. A rename in
    /// either half would land here in the same commit or fail this
    /// test (mirror of `arch_string_matches_runtime_constant` in the
    /// converter tests).
    #[test]
    fn gguf_keys_match_converter_wire_names() {
        assert_eq!(GGUF_KEY_SAMPLE_RATE, "vokra.canary_qwen.sample_rate");
        assert_eq!(
            GGUF_KEY_ENC_N_LAYER,
            "vokra.canary_qwen.arch.encoder.n_layer"
        );
        assert_eq!(
            GGUF_KEY_ENC_D_MODEL,
            "vokra.canary_qwen.arch.encoder.d_model"
        );
        assert_eq!(GGUF_KEY_ENC_N_HEAD, "vokra.canary_qwen.arch.encoder.n_head");
        assert_eq!(
            GGUF_KEY_ENC_N_HEAD_KV,
            "vokra.canary_qwen.arch.encoder.n_head_kv"
        );
        assert_eq!(
            GGUF_KEY_ENC_FFN_DIM,
            "vokra.canary_qwen.arch.encoder.ffn_dim"
        );
        assert_eq!(
            GGUF_KEY_ENC_CONV_KERNEL,
            "vokra.canary_qwen.arch.encoder.conv_kernel_size"
        );
        assert_eq!(GGUF_KEY_ENC_IN_DIM, "vokra.canary_qwen.arch.encoder.in_dim");
        assert_eq!(
            GGUF_KEY_ENC_SUBSAMPLING_FACTOR,
            "vokra.canary_qwen.arch.encoder.subsampling_factor"
        );
        assert_eq!(
            GGUF_KEY_ENC_MAX_POS,
            "vokra.canary_qwen.arch.encoder.max_position_embeddings"
        );
        assert_eq!(
            GGUF_KEY_ENC_ATTN_BIAS,
            "vokra.canary_qwen.arch.encoder.attention_bias"
        );
        assert_eq!(
            GGUF_KEY_DEC_N_LAYER,
            "vokra.canary_qwen.arch.decoder.n_layer"
        );
        assert_eq!(
            GGUF_KEY_DEC_HIDDEN_DIM,
            "vokra.canary_qwen.arch.decoder.hidden_dim"
        );
        assert_eq!(
            GGUF_KEY_DEC_N_HEAD_Q,
            "vokra.canary_qwen.arch.decoder.n_head_q"
        );
        assert_eq!(
            GGUF_KEY_DEC_N_HEAD_KV,
            "vokra.canary_qwen.arch.decoder.n_head_kv"
        );
        assert_eq!(
            GGUF_KEY_DEC_HEAD_DIM,
            "vokra.canary_qwen.arch.decoder.head_dim"
        );
        assert_eq!(
            GGUF_KEY_DEC_FFN_DIM,
            "vokra.canary_qwen.arch.decoder.ffn_dim"
        );
        assert_eq!(
            GGUF_KEY_DEC_VOCAB_SIZE,
            "vokra.canary_qwen.arch.decoder.vocab_size"
        );
        assert_eq!(GGUF_KEY_DEC_N_CTX, "vokra.canary_qwen.arch.decoder.n_ctx");
        assert_eq!(
            GGUF_KEY_DEC_ROPE_BASE,
            "vokra.canary_qwen.arch.decoder.rope_base"
        );
        assert_eq!(
            GGUF_KEY_DEC_RMS_NORM_EPS,
            "vokra.canary_qwen.arch.decoder.rms_norm_eps"
        );
        assert_eq!(
            GGUF_KEY_CROSS_ATTN_HIDDEN_DIM,
            "vokra.canary_qwen.arch.cross_attn.hidden_dim"
        );
    }
}
