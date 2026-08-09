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

use vokra_core::rng::SplitMix64;
use vokra_core::{Result, VokraError};

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

        Ok(Self { cfg, weights })
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
        Err(VokraError::NotImplemented(
            "canary_qwen transcribe: real weights are bound but the log-mel front-end \
             (STFT + mel filterbank) -> FastConformer encoder (vokra_ops::conformer, \
             shared with Canary-1B-v2) -> soft-prompt projection -> Qwen LLM decoder \
             (shared with Voxtral) -> greedy/beam search -> SentencePiece detokenize \
             forward path has not landed yet. Follow-up wave: wire CanaryQwenWeights \
             to the shared ConformerEncoder + Voxtral TextDecoderSession, feeding the \
             encoder-out through enc_to_lm_proj as a soft-prompt prefix and reusing \
             vokra_ops::beam_search with blank_id / bos / eos taken from the .nemo \
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
}
