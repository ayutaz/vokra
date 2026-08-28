//! HuggingFace **distil-whisper / distil-large-v3.5** — Whisper large-v3
//! encoder + a 2-layer decoder (SoTA plan Phase 2, 2026-07-24).
//!
//! # What distil-large-v3.5 is (primary source)
//!
//! `distil-whisper/distil-large-v3.5` is a distilled Whisper checkpoint:
//! the **large-v3 encoder is kept intact** (32 layers, d_model=1280,
//! n_mels=128, encoder_attention_heads=20) and the **decoder is shrunk to
//! 2 layers** (same width / head count as large-v3). The tokenizer is the
//! large-v3 multilingual byte-level BPE (`vocab_size=51866`,
//! `eos_token_id=50257`, `decoder_start_token_id=50258`).
//!
//! Every hparam below is transcribed **verbatim** from
//! `huggingface.co/distil-whisper/distil-large-v3.5/raw/main/config.json`
//! (fetched 2026-07-24 — CLAUDE.md「ハルシネーション厳禁」):
//!
//! - **Model type** (`model_type`): `"whisper"`
//!   (`architectures = ["WhisperForConditionalGeneration"]`). The
//!   underlying architecture is Whisper, only the decoder depth changes,
//!   which is why this module is a thin scaffold over the existing
//!   [`crate::whisper`] plumbing — every op / kernel / attention detail
//!   is shared verbatim.
//! - **Encoder** (identical to Whisper `large-v3`):
//!   - `d_model` = 1280,
//!   - `encoder_layers` = 32,
//!   - `encoder_attention_heads` = 20 (`head_dim = 1280 / 20 = 64`,
//!     the Whisper family invariant),
//!   - `encoder_ffn_dim` = 5120,
//!   - `num_mel_bins` = 128 (large-v3 log-mel front-end),
//!   - `max_source_positions` = 1500 (encoder positional length —
//!     `n_audio_ctx`).
//! - **Decoder** (the distil axis — this is where it differs from
//!   large-v3):
//!   - `decoder_layers` = **2** (large-v3 has 32),
//!   - `decoder_attention_heads` = 20 (unchanged),
//!   - `decoder_ffn_dim` = 5120 (unchanged),
//!   - `max_target_positions` = 448 (decoder positional length —
//!     `n_text_ctx`).
//! - **Tokenizer**:
//!   - `vocab_size` = 51866 (large-v3 multilingual +1 for `<|yue|>` vs
//!     base/small/medium's 51865),
//!   - `bos_token_id` = 50257 (`<|endoftext|>` — same as `eos_token_id`
//!     by Whisper convention; the EOT special token),
//!   - `eos_token_id` = 50257,
//!   - `decoder_start_token_id` = 50258 (`<|startoftranscript|>`),
//!   - `pad_token_id` = 50256.
//! - **Audio boundary**: `sample_rate` = 16 000 (Whisper convention).
//! - **Weight license**: **MIT** (per the `distil-whisper` repo — the
//!   distil-whisper family is MIT code + MIT weights, mirroring
//!   `openai/whisper`'s MIT posture) — resolves to
//!   [`vokra_core::LicenseClass::Permissive`] via the
//!   `distil-whisper-` / `distil-large-` family walks, so the M2-13 gate
//!   passes commercially without any attribution obligation on the
//!   runtime side.
//!
//! # Very-cheap follow-on — reuses Whisper verbatim
//!
//! Because the topology is a Whisper checkpoint whose only difference is
//! `n_text_layer = 2`, distil-large-v3.5 **does not add any new op**
//! (`vokra-ops`) or backend kernel: the same STFT / mel filterbank / GEMM /
//! GEMV / softmax / layer-norm / GELU / conv1d inventory Whisper base
//! consumes (see [`crate::whisper`] docstring §Operator inventory) is
//! also what distil-large-v3.5 uses. The runtime forward is a follow-up
//! wave (T29-equivalent — the Moshi / CSM / Zonos / Kyutai STT /
//! Parakeet-CTC pattern): when it lands it will delegate to
//! [`crate::whisper::WhisperModel`] with an appropriately-shrunk
//! `WhisperConfig`, since the checkpoint's tensor names follow the
//! upstream HF Whisper convention verbatim
//! (`model.encoder.layers.*` / `model.decoder.layers.*`) and the
//! converter (`vokra-convert::models::distil_whisper`) writes them
//! through unchanged.
//!
//! # What lands in this Phase 2 slice
//!
//! - [`DistilWhisperConfig`] — every hparam transcribed from the primary
//!   source, plus a `distil_invariant` sanity check
//!   (`n_text_layer < n_audio_layer`) that catches a checkpoint whose
//!   decoder depth was left at the source (large-v3 = 32) instead of the
//!   shrunk distil count.
//! - [`DistilWhisperWeights`] — a scaffold weight store with a
//!   deterministic [`DistilWhisperWeights::synthesized`] fixture
//!   (SplitMix64 + Xavier) so shape / dtype / size flow can be exercised
//!   without the real HF checkpoint.
//! - [`DistilWhisperAsr`] — engine handle with two construction paths.
//!   [`DistilWhisperAsr::from_gguf`] binds a converted GGUF through
//!   [`crate::whisper::WhisperAsr`] and [`DistilWhisperAsr::transcribe`]
//!   then runs the **real** forward (log-mel front-end → 32-layer
//!   encoder → 2-layer decoder → BPE detokenize), shared verbatim with
//!   vanilla Whisper; the [`AsrEngine`] impl below exposes the same
//!   forward behind the session facade. The scaffold path
//!   [`DistilWhisperAsr::new`] (config + a standalone
//!   [`DistilWhisperWeights`] store) is the only one that hard-errors
//!   with [`VokraError::NotImplemented`] — that store is deliberately not
//!   wired to the shared engine, so it exercises shape / invariant flow
//!   only.
//!
//! # No ONNX (permanent)
//!
//! `distil-whisper/distil-large-v3.5` ships PyTorch safetensors; the
//! pipeline is re-implemented natively via [`crate::whisper`]
//! (whisper.cpp 型, CLAUDE.md 設計判断 4). This module never touches ONNX.

use vokra_core::engines::AsrEngine;
use vokra_core::gguf::GgufFile;
use vokra_core::rng::SplitMix64;
use vokra_core::tasks::Transcription;
use vokra_core::{BackendKind, Result, VokraError};

use crate::whisper::{WhisperAsr, WhisperTokenizer};

#[cfg(feature = "coreml")]
use crate::whisper::CoreMlArtifact;

/// `vokra.model.arch` a distil-whisper GGUF must carry. Written by
/// `vokra-convert::models::distil_whisper::ARCH`; the compliance registry
/// (`vokra_core::compliance`) knows `distil-whisper` / `distil-large-v3` /
/// `distil-large-v3.5` (and every family variant that lands later) as
/// [`vokra_core::LicenseClass::Permissive`] via the `distil-whisper-` /
/// `distil-large-` family prefix walks (MIT — the M2-13 gate passes
/// commercially).
///
/// This arch string is intentionally **distinct** from Whisper's
/// (`"whisper"`) so the runtime can label the loaded model correctly in
/// telemetry / logs / model cards while still consuming the same
/// `vokra.whisper.*` hparam chunk schema and Whisper decoder plumbing —
/// the "very cheap follow-on" contract in the task.
pub const EXPECTED_ARCH: &str = "distil-whisper";

/// PCM sample rate distil-whisper expects. Same as vanilla Whisper —
/// 16 kHz mono, per the openai/whisper convention (not written directly in
/// `config.json` but inherited from the Whisper feature extractor
/// preprocessor).
pub const DISTIL_WHISPER_SAMPLE_RATE: u32 = 16_000;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// distil-whisper architectural hyperparameters.
///
/// A deliberate subset of the Whisper `config.json` schema — every field
/// maps 1-to-1 to the corresponding Whisper axis (see
/// [`crate::whisper::WhisperConfig`]). The distinguishing invariant is
/// **`n_text_layer < n_audio_layer`** ("the decoder is smaller than the
/// encoder"), enforced by [`Self::validate_for_forward`] — a real
/// non-distil checkpoint would have `n_text_layer == n_audio_layer`
/// (Whisper base…large-v3) or `n_text_layer < n_audio_layer` only for
/// the turbo variant (which is a separate arch, `"whisper"`, at
/// `whisper-turbo`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DistilWhisperConfig {
    /// Mel input channels (encoder conv1 in-channels). **128** for
    /// distil-large-v3.5 (matching large-v3's 128-bin front-end); 80 for
    /// distil-small.en / distil-medium.en (the smaller distil variants).
    pub n_mels: usize,
    /// Hidden width `d_model` shared by encoder and decoder — 1280 for
    /// distil-large-v3.5.
    pub d_model: usize,
    /// Encoder positional length (`max_source_positions`), 1500.
    pub n_audio_ctx: usize,
    /// Encoder attention heads — 20 for distil-large-v3.5
    /// (`head_dim = 1280 / 20 = 64`).
    pub n_audio_head: usize,
    /// Encoder block count. **32** for distil-large-v3.5 — the distil
    /// family keeps the large-v3 encoder intact.
    pub n_audio_layer: usize,
    /// Decoder positional length (`max_target_positions`), 448.
    pub n_text_ctx: usize,
    /// Decoder attention heads — same as `n_audio_head` for the
    /// distil-large family.
    pub n_text_head: usize,
    /// Decoder block count. **2** for distil-large-v3.5 — the distil
    /// axis (large-v3 has 32).
    pub n_text_layer: usize,
    /// Token vocabulary size — **51 866** for distil-large-v3.5 (the
    /// large-v3 multilingual vocab including `<|yue|>`).
    pub n_vocab: usize,
    /// Feed-forward inner width — 5120 for distil-large-v3.5.
    pub ffn_dim: usize,
    /// End-of-transcript token id (decode stop condition) — 50257 for
    /// the Whisper multilingual tokenizer.
    pub eot: u32,
    /// Start-of-transcript token id (decoder prompt seed) — 50258 for
    /// the Whisper multilingual tokenizer.
    pub sot: u32,
    /// PCM sample rate — 16 000 (Whisper convention).
    pub sample_rate: u32,
}

impl DistilWhisperConfig {
    /// Per-head width. Whisper fixes this at 64 across every family
    /// size, so it is simply `d_model / n_audio_head` (validated
    /// non-zero and exact in [`Self::validate_for_forward`]).
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.d_model.checked_div(self.n_audio_head).unwrap_or(0)
    }

    /// Primary-source distil-large-v3.5 config (every value transcribed
    /// verbatim from the upstream `config.json` — see module docstring).
    #[must_use]
    pub fn distil_large_v3_5() -> Self {
        Self {
            n_mels: 128,
            d_model: 1280,
            n_audio_ctx: 1500,
            n_audio_head: 20,
            n_audio_layer: 32,
            n_text_ctx: 448,
            n_text_head: 20,
            n_text_layer: 2,
            n_vocab: 51_866,
            ffn_dim: 5120,
            eot: 50_257,
            sot: 50_258,
            sample_rate: DISTIL_WHISPER_SAMPLE_RATE,
        }
    }

    /// Miniature well-formed config for shape / stability tests. Dims
    /// are tiny so synthesized-weight builds fit in KB; the *shape
    /// relationships* (encoder deeper than decoder, MHA head split,
    /// even head_dim, vocab non-zero) mirror the real model.
    #[must_use]
    pub fn tiny_for_tests() -> Self {
        Self {
            n_mels: 16,
            d_model: 16,
            n_audio_ctx: 24,
            n_audio_head: 4,
            n_audio_layer: 4,
            n_text_ctx: 16,
            n_text_head: 4,
            n_text_layer: 2,
            n_vocab: 32,
            ffn_dim: 32,
            eot: 30,
            sot: 31,
            sample_rate: DISTIL_WHISPER_SAMPLE_RATE,
        }
    }

    /// Rejects `0`-placeholder / ill-formed configs before any forward
    /// runs (FR-EX-08 — a shape-only converter path fails loudly here,
    /// not deep inside a GEMM).
    ///
    /// Enforces the Whisper cross-checks (`d_model % n_head == 0` for
    /// both encoder and decoder, non-zero axes, EOT/SOT inside the
    /// vocab) plus the distil invariant
    /// (`n_text_layer < n_audio_layer`).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] naming the offending field.
    pub fn validate_for_forward(&self) -> Result<()> {
        if self.d_model == 0
            || self.n_mels == 0
            || self.n_audio_ctx == 0
            || self.n_audio_layer == 0
            || self.n_text_ctx == 0
            || self.n_text_layer == 0
            || self.n_vocab == 0
            || self.ffn_dim == 0
            || self.n_audio_head == 0
            || self.n_text_head == 0
            || self.sample_rate == 0
        {
            return Err(VokraError::InvalidArgument(
                "distil-whisper config: every architectural axis must be > 0".to_owned(),
            ));
        }
        if self.d_model % self.n_audio_head != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "distil-whisper config: n_audio_head ({}) must divide d_model ({})",
                self.n_audio_head, self.d_model,
            )));
        }
        if self.d_model % self.n_text_head != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "distil-whisper config: n_text_head ({}) must divide d_model ({})",
                self.n_text_head, self.d_model,
            )));
        }
        if self.head_dim() % 2 != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "distil-whisper config: head_dim {} must be even (attention K/V pair layout)",
                self.head_dim(),
            )));
        }
        // The distil axis: a distil-whisper checkpoint has fewer decoder
        // layers than encoder layers. A checkpoint where the two are equal
        // is (a) a real Whisper (large-v3, medium, etc.) that landed on
        // the distil path by mistake, or (b) a mis-flattened distil
        // checkpoint where the decoder tensors were duplicated to the
        // encoder count. Either way this must fail loudly (FR-EX-08).
        if self.n_text_layer >= self.n_audio_layer {
            return Err(VokraError::InvalidArgument(format!(
                "distil-whisper config: n_text_layer ({}) must be < n_audio_layer ({}); \
                 a distil checkpoint shrinks the decoder, so equal or larger decoder \
                 depth means this is not a distil-whisper (use --model whisper for \
                 vanilla Whisper sizes)",
                self.n_text_layer, self.n_audio_layer,
            )));
        }
        if (self.eot as usize) >= self.n_vocab {
            return Err(VokraError::InvalidArgument(format!(
                "distil-whisper config: eot ({}) must be < n_vocab ({})",
                self.eot, self.n_vocab,
            )));
        }
        if (self.sot as usize) >= self.n_vocab {
            return Err(VokraError::InvalidArgument(format!(
                "distil-whisper config: sot ({}) must be < n_vocab ({})",
                self.sot, self.n_vocab,
            )));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Weights
// ---------------------------------------------------------------------------

/// Per-attention-projection weights for one encoder / decoder block.
///
/// Q / K / V / out projections are stored separately (matching the
/// upstream HF `whisper.model.encoder.layers.i.self_attn.{q,k,v,out}_proj`
/// convention that the converter passes through). Whisper convention has
/// **`k_proj` bias-free** and every other projection carrying a bias —
/// the same detail [`crate::whisper::weights`] handles.
#[derive(Debug, Clone)]
pub struct DistilWhisperAttnProjWeights {
    /// `[d_model, d_model]` — Q projection.
    pub q_proj: Vec<f32>,
    /// `[d_model]` — Q bias.
    pub q_bias: Vec<f32>,
    /// `[d_model, d_model]` — K projection.
    pub k_proj: Vec<f32>,
    // K carries NO bias per Whisper convention — a stray Some here would
    // silently shift attention scores.
    /// `[d_model, d_model]` — V projection.
    pub v_proj: Vec<f32>,
    /// `[d_model]` — V bias.
    pub v_bias: Vec<f32>,
    /// `[d_model, d_model]` — attention output projection.
    pub out_proj: Vec<f32>,
    /// `[d_model]` — attention output bias.
    pub out_bias: Vec<f32>,
}

/// Per-encoder-block scaffold weights (pre-norm self-attention + FFN,
/// GELU activation — the Whisper block topology).
#[derive(Debug, Clone)]
pub struct DistilWhisperEncoderBlockWeights {
    /// Self-attention pre-norm γ, shape `[d_model]`.
    pub self_attn_norm_gamma: Vec<f32>,
    /// Self-attention pre-norm β, shape `[d_model]`.
    pub self_attn_norm_beta: Vec<f32>,
    /// Self-attention Q/K/V/out projections.
    pub self_attn: DistilWhisperAttnProjWeights,
    /// FFN pre-norm γ, shape `[d_model]`.
    pub ffn_norm_gamma: Vec<f32>,
    /// FFN pre-norm β, shape `[d_model]`.
    pub ffn_norm_beta: Vec<f32>,
    /// FFN hidden projection, shape `[d_model, ffn_dim]`.
    pub ffn_fc1: Vec<f32>,
    /// FFN hidden bias, shape `[ffn_dim]`.
    pub ffn_fc1_bias: Vec<f32>,
    /// FFN output projection, shape `[ffn_dim, d_model]`.
    pub ffn_fc2: Vec<f32>,
    /// FFN output bias, shape `[d_model]`.
    pub ffn_fc2_bias: Vec<f32>,
}

/// Per-decoder-block scaffold weights (pre-norm self-attention + pre-norm
/// cross-attention + FFN — the Whisper decoder block topology).
#[derive(Debug, Clone)]
pub struct DistilWhisperDecoderBlockWeights {
    /// Self-attention pre-norm γ, shape `[d_model]`.
    pub self_attn_norm_gamma: Vec<f32>,
    /// Self-attention pre-norm β, shape `[d_model]`.
    pub self_attn_norm_beta: Vec<f32>,
    /// Causal self-attention Q/K/V/out projections.
    pub self_attn: DistilWhisperAttnProjWeights,
    /// Cross-attention pre-norm γ, shape `[d_model]`.
    pub cross_attn_norm_gamma: Vec<f32>,
    /// Cross-attention pre-norm β, shape `[d_model]`.
    pub cross_attn_norm_beta: Vec<f32>,
    /// Cross-attention Q/K/V/out projections (K/V read the encoder
    /// output, so `k_proj` / `v_proj` have shape `[d_model, d_model]`
    /// as well).
    pub cross_attn: DistilWhisperAttnProjWeights,
    /// FFN pre-norm γ, shape `[d_model]`.
    pub ffn_norm_gamma: Vec<f32>,
    /// FFN pre-norm β, shape `[d_model]`.
    pub ffn_norm_beta: Vec<f32>,
    /// FFN hidden projection, shape `[d_model, ffn_dim]`.
    pub ffn_fc1: Vec<f32>,
    /// FFN hidden bias, shape `[ffn_dim]`.
    pub ffn_fc1_bias: Vec<f32>,
    /// FFN output projection, shape `[ffn_dim, d_model]`.
    pub ffn_fc2: Vec<f32>,
    /// FFN output bias, shape `[d_model]`.
    pub ffn_fc2_bias: Vec<f32>,
}

/// distil-whisper weight store: conv1 / conv2 stem (log-mel → d_model) +
/// encoder positional embedding + encoder blocks + encoder-out LayerNorm +
/// token embedding + decoder positional embedding + decoder blocks +
/// decoder-out LayerNorm.
///
/// [`Self::synthesized`] builds a deterministic fixture (SplitMix64 +
/// Xavier) against `config` so shape / dtype / size can be exercised
/// without the real HF checkpoint. Real-checkpoint binding is a
/// follow-up (T29-equivalent — the Moshi / CSM / Kyutai STT /
/// Parakeet-CTC pattern).
#[derive(Debug, Clone)]
pub struct DistilWhisperWeights {
    /// Encoder conv1: `[d_model, n_mels, 3]` (Whisper convention —
    /// stride 1, kernel 3, in-channels = n_mels).
    pub conv1_weight: Vec<f32>,
    /// Encoder conv1 bias: `[d_model]`.
    pub conv1_bias: Vec<f32>,
    /// Encoder conv2: `[d_model, d_model, 3]` (stride 2, kernel 3).
    pub conv2_weight: Vec<f32>,
    /// Encoder conv2 bias: `[d_model]`.
    pub conv2_bias: Vec<f32>,
    /// Encoder positional embedding: `[n_audio_ctx, d_model]`.
    pub encoder_pos_embed: Vec<f32>,
    /// Encoder blocks in order.
    pub encoder_blocks: Vec<DistilWhisperEncoderBlockWeights>,
    /// Encoder-out LayerNorm γ, shape `[d_model]`.
    pub encoder_final_norm_gamma: Vec<f32>,
    /// Encoder-out LayerNorm β, shape `[d_model]`.
    pub encoder_final_norm_beta: Vec<f32>,
    /// Token embedding: `[n_vocab, d_model]` (tied to the logits head —
    /// same tensor is used both places, Whisper convention).
    pub token_embed: Vec<f32>,
    /// Decoder positional embedding: `[n_text_ctx, d_model]`.
    pub decoder_pos_embed: Vec<f32>,
    /// Decoder blocks in order.
    pub decoder_blocks: Vec<DistilWhisperDecoderBlockWeights>,
    /// Decoder-out LayerNorm γ, shape `[d_model]`.
    pub decoder_final_norm_gamma: Vec<f32>,
    /// Decoder-out LayerNorm β, shape `[d_model]`.
    pub decoder_final_norm_beta: Vec<f32>,
    /// `true` when built by [`Self::synthesized`] — never a real
    /// upstream checkpoint.
    pub is_synthesized: bool,
}

impl DistilWhisperWeights {
    /// Builds a deterministic synthesized fixture from `config` and
    /// `seed`.
    ///
    /// Draws are Xavier-uniform ± `sqrt(6 / (fan_in + fan_out))` via a
    /// [`SplitMix64`] stream — reproducible, allocation-only, zero-dep.
    /// Every LayerNorm γ starts at `1.0`; every LayerNorm β and every
    /// bias starts at `0.0`.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] if `config.validate_for_forward`
    /// fails.
    pub fn synthesized(config: &DistilWhisperConfig, seed: u64) -> Result<Self> {
        config.validate_for_forward()?;
        let mut rng = SplitMix64::new(seed);
        let d = config.d_model;
        let ffn = config.ffn_dim;

        let conv1_weight = xavier(&mut rng, d * config.n_mels * 3, config.n_mels * 3, d);
        let conv1_bias = vec![0.0; d];
        let conv2_weight = xavier(&mut rng, d * d * 3, d * 3, d);
        let conv2_bias = vec![0.0; d];
        let encoder_pos_embed = xavier(&mut rng, config.n_audio_ctx * d, d, d);

        let mut encoder_blocks = Vec::with_capacity(config.n_audio_layer);
        for _ in 0..config.n_audio_layer {
            encoder_blocks.push(DistilWhisperEncoderBlockWeights {
                self_attn_norm_gamma: vec![1.0; d],
                self_attn_norm_beta: vec![0.0; d],
                self_attn: build_attn_projs(&mut rng, d),
                ffn_norm_gamma: vec![1.0; d],
                ffn_norm_beta: vec![0.0; d],
                ffn_fc1: xavier(&mut rng, d * ffn, d, ffn),
                ffn_fc1_bias: vec![0.0; ffn],
                ffn_fc2: xavier(&mut rng, ffn * d, ffn, d),
                ffn_fc2_bias: vec![0.0; d],
            });
        }
        let encoder_final_norm_gamma = vec![1.0; d];
        let encoder_final_norm_beta = vec![0.0; d];

        let token_embed = xavier(&mut rng, config.n_vocab * d, d, d);
        let decoder_pos_embed = xavier(&mut rng, config.n_text_ctx * d, d, d);

        let mut decoder_blocks = Vec::with_capacity(config.n_text_layer);
        for _ in 0..config.n_text_layer {
            decoder_blocks.push(DistilWhisperDecoderBlockWeights {
                self_attn_norm_gamma: vec![1.0; d],
                self_attn_norm_beta: vec![0.0; d],
                self_attn: build_attn_projs(&mut rng, d),
                cross_attn_norm_gamma: vec![1.0; d],
                cross_attn_norm_beta: vec![0.0; d],
                cross_attn: build_attn_projs(&mut rng, d),
                ffn_norm_gamma: vec![1.0; d],
                ffn_norm_beta: vec![0.0; d],
                ffn_fc1: xavier(&mut rng, d * ffn, d, ffn),
                ffn_fc1_bias: vec![0.0; ffn],
                ffn_fc2: xavier(&mut rng, ffn * d, ffn, d),
                ffn_fc2_bias: vec![0.0; d],
            });
        }
        let decoder_final_norm_gamma = vec![1.0; d];
        let decoder_final_norm_beta = vec![0.0; d];

        Ok(Self {
            conv1_weight,
            conv1_bias,
            conv2_weight,
            conv2_bias,
            encoder_pos_embed,
            encoder_blocks,
            encoder_final_norm_gamma,
            encoder_final_norm_beta,
            token_embed,
            decoder_pos_embed,
            decoder_blocks,
            decoder_final_norm_gamma,
            decoder_final_norm_beta,
            is_synthesized: true,
        })
    }
}

/// Builds the four attention projections (Q / K / V / out) for one block.
///
/// Q / V / out carry biases; K does not (Whisper convention — a
/// `k_proj.bias` in the checkpoint is a converter red flag).
fn build_attn_projs(rng: &mut SplitMix64, d: usize) -> DistilWhisperAttnProjWeights {
    DistilWhisperAttnProjWeights {
        q_proj: xavier(rng, d * d, d, d),
        q_bias: vec![0.0; d],
        k_proj: xavier(rng, d * d, d, d),
        v_proj: xavier(rng, d * d, d, d),
        v_bias: vec![0.0; d],
        out_proj: xavier(rng, d * d, d, d),
        out_bias: vec![0.0; d],
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

/// distil-whisper ASR engine handle.
///
/// Carries the resolved config and weight store. [`Self::transcribe`]
/// is the primary waveform → text entry point; until real weights are
/// bound (see the module docstring) it returns
/// [`VokraError::NotImplemented`] with a message naming the blocker
/// (FR-EX-08 — never a silent zero-fill or empty transcript).
/// distil-whisper ASR engine handle.
///
/// # Two construction paths
///
/// - [`Self::new`] — the scaffold path (cfg + owned `DistilWhisperWeights`),
///   for shape-flow / invariant / synthesized-weight tests. `transcribe`
///   hard-errors with [`VokraError::NotImplemented`] on this path (real
///   forward not bound against the module's own weight store).
/// - [`Self::from_gguf`] — the real path: delegates to the shared
///   [`crate::whisper::WhisperAsr`] engine (distil-whisper is
///   architecturally a Whisper checkpoint whose only axis of difference is
///   `n_text_layer < n_audio_layer`; the upstream converter writes the
///   standard `vokra.whisper.*` chunk and keeps HF Whisper tensor names
///   verbatim, so the runtime forward is a delegation, not a
///   re-implementation).
///
/// `Debug` / `Clone` are intentionally NOT derived: [`WhisperAsr`] carries
/// non-trivially-cloneable engine state (Arc + kv caches). Introspection is
/// exposed via [`Self::has_weights_bound`], [`Self::is_synthesized`], and
/// [`Self::config`].
pub struct DistilWhisperAsr {
    cfg: DistilWhisperConfig,
    kind: DistilWhisperAsrKind,
}

/// Internal — how this handle was built.
///
/// - `Scaffold(w)` = old `new(cfg, w)` path. `transcribe` returns a loud
///   NotImplemented naming the follow-up wave (real weights required).
/// - `Delegate(asr)` = new `from_gguf` path. `transcribe` delegates to the
///   shared Whisper engine (real forward).
enum DistilWhisperAsrKind {
    Scaffold(Box<DistilWhisperWeights>),
    Delegate(WhisperAsr),
}

impl DistilWhisperAsr {
    /// Assembles a scaffold engine from `cfg` and `weights` (shape-flow
    /// path). Cross-checks the weight-store shapes against `cfg` (encoder
    /// / decoder block counts + per-tensor sizes, positional embedding
    /// shapes, token embedding shape, conv stem shape) so a mismatched
    /// pair fails loudly here rather than deep inside a forward.
    ///
    /// **This scaffold does not wire a real forward** — the resulting
    /// handle exercises shape flow / invariant checks / the loud
    /// [`Self::transcribe`] refusal path. For real transcription, use
    /// [`Self::from_gguf`] which delegates to the shared Whisper engine.
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] from `cfg.validate_for_forward`.
    /// - [`VokraError::InvalidArgument`] naming the first shape
    ///   mismatch.
    pub fn new(cfg: DistilWhisperConfig, weights: DistilWhisperWeights) -> Result<Self> {
        cfg.validate_for_forward()?;
        let d = cfg.d_model;
        let ffn = cfg.ffn_dim;

        check_len(
            "conv1_weight",
            weights.conv1_weight.len(),
            d * cfg.n_mels * 3,
        )?;
        check_len("conv1_bias", weights.conv1_bias.len(), d)?;
        check_len("conv2_weight", weights.conv2_weight.len(), d * d * 3)?;
        check_len("conv2_bias", weights.conv2_bias.len(), d)?;
        check_len(
            "encoder_pos_embed",
            weights.encoder_pos_embed.len(),
            cfg.n_audio_ctx * d,
        )?;

        if weights.encoder_blocks.len() != cfg.n_audio_layer {
            return Err(VokraError::InvalidArgument(format!(
                "distil-whisper weights: encoder_blocks.len()={} != n_audio_layer={}",
                weights.encoder_blocks.len(),
                cfg.n_audio_layer,
            )));
        }
        for (i, blk) in weights.encoder_blocks.iter().enumerate() {
            check_encoder_block_shapes(i, blk, d, ffn)?;
        }
        check_len(
            "encoder_final_norm_gamma",
            weights.encoder_final_norm_gamma.len(),
            d,
        )?;
        check_len(
            "encoder_final_norm_beta",
            weights.encoder_final_norm_beta.len(),
            d,
        )?;

        check_len("token_embed", weights.token_embed.len(), cfg.n_vocab * d)?;
        check_len(
            "decoder_pos_embed",
            weights.decoder_pos_embed.len(),
            cfg.n_text_ctx * d,
        )?;

        if weights.decoder_blocks.len() != cfg.n_text_layer {
            return Err(VokraError::InvalidArgument(format!(
                "distil-whisper weights: decoder_blocks.len()={} != n_text_layer={} \
                 (the distil axis — this must equal the small decoder depth from the \
                 checkpoint's `decoder_layers` field, not the encoder count)",
                weights.decoder_blocks.len(),
                cfg.n_text_layer,
            )));
        }
        for (i, blk) in weights.decoder_blocks.iter().enumerate() {
            check_decoder_block_shapes(i, blk, d, ffn)?;
        }
        check_len(
            "decoder_final_norm_gamma",
            weights.decoder_final_norm_gamma.len(),
            d,
        )?;
        check_len(
            "decoder_final_norm_beta",
            weights.decoder_final_norm_beta.len(),
            d,
        )?;

        Ok(Self {
            cfg,
            kind: DistilWhisperAsrKind::Scaffold(Box::new(weights)),
        })
    }

    /// Loads a real distil-whisper GGUF and binds the full weight set by
    /// delegating to the shared [`crate::whisper::WhisperAsr`] engine.
    ///
    /// **distil-whisper is architecturally a Whisper checkpoint whose only
    /// difference is `n_text_layer < n_audio_layer`** (see module docs).
    /// The upstream converter (`vokra-convert::models::distil_whisper`)
    /// therefore writes the standard `vokra.whisper.*` hparam chunk and
    /// keeps HF Whisper tensor names verbatim, so this delegates the
    /// forward to the shared Whisper plumbing — same op set (STFT / mel
    /// filterbank / GEMM / GEMV / softmax / layer-norm / GELU / conv1d),
    /// same kernels, same greedy / beam-search paths.
    ///
    /// The **distil invariant** (`n_text_layer < n_audio_layer`) is
    /// enforced on the loaded config: a checkpoint whose decoder-layer
    /// count equals or exceeds the encoder count is either vanilla
    /// Whisper (large-v3 = 32/32) or a mis-flattened distil, and this
    /// fails loudly (FR-EX-08) rather than mis-labeling a Whisper GGUF
    /// as distil-whisper.
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] via the delegate load path (missing
    ///   `vokra.whisper.*` metadata, missing / mis-shaped weight tensors,
    ///   or the front-end chunk check).
    /// - [`VokraError::ModelLoad`] if the loaded config violates the
    ///   distil-whisper distil invariant (`n_text_layer >= n_audio_layer`).
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let inner = WhisperAsr::from_gguf(file)?;
        let wc = inner.model().config();
        if wc.n_text_layer >= wc.n_audio_layer {
            return Err(VokraError::ModelLoad(format!(
                "distil-whisper: loaded GGUF has n_text_layer ({}) >= n_audio_layer ({}); \
                 distil-whisper is a Whisper checkpoint whose decoder is strictly smaller \
                 than the encoder — equal or larger decoder depth means this GGUF is \
                 vanilla Whisper (use --model whisper) or a mis-flattened distil \
                 (decoder tensors duplicated to the encoder count). This is a loud-fail \
                 contract (FR-EX-08), never a silent mis-label.",
                wc.n_text_layer, wc.n_audio_layer,
            )));
        }
        // Build a `DistilWhisperConfig` snapshot from the loaded Whisper config
        // so [`Self::config`] stays stable across construction paths.
        let cfg = DistilWhisperConfig {
            n_mels: wc.n_mels,
            d_model: wc.d_model,
            n_audio_ctx: wc.n_audio_ctx,
            n_audio_head: wc.n_audio_head,
            n_audio_layer: wc.n_audio_layer,
            n_text_ctx: wc.n_text_ctx,
            n_text_head: wc.n_text_head,
            n_text_layer: wc.n_text_layer,
            n_vocab: wc.n_vocab,
            ffn_dim: wc.ffn_dim,
            eot: wc.eot,
            sot: wc.decoder_start_ids.first().copied().unwrap_or(50_258),
            sample_rate: DISTIL_WHISPER_SAMPLE_RATE,
        };
        cfg.validate_for_forward()?;
        Ok(Self {
            cfg,
            kind: DistilWhisperAsrKind::Delegate(inner),
        })
    }

    /// Attaches a detokenizer for [`Self::transcribe`]. No-op on the
    /// scaffold path ([`Self::new`]) — the scaffold has no inner engine.
    #[must_use]
    pub fn with_tokenizer(mut self, tokenizer: WhisperTokenizer) -> Self {
        if let DistilWhisperAsrKind::Delegate(asr) = self.kind {
            self.kind = DistilWhisperAsrKind::Delegate(asr.with_tokenizer(tokenizer));
        }
        self
    }

    /// Selects the backend the transcription forward runs on (default
    /// [`BackendKind::Cpu`]).
    ///
    /// No-op on the scaffold path. On the delegate path an unsupported
    /// backend surfaces as an explicit [`VokraError::UnsupportedOp`] at
    /// [`Self::transcribe`] time (never a silent CPU fall back — FR-EX-08).
    #[must_use]
    pub fn with_backend(mut self, backend: BackendKind) -> Self {
        if let DistilWhisperAsrKind::Delegate(asr) = self.kind {
            self.kind = DistilWhisperAsrKind::Delegate(asr.with_backend(backend));
        }
        self
    }

    /// Binds the verified whole-encoder CoreML sidecar to the shared Whisper
    /// delegate path. The config-only scaffold has no executable model and
    /// therefore rejects the artifact instead of dropping it silently.
    #[cfg(feature = "coreml")]
    pub fn with_coreml_artifact(mut self, artifact: CoreMlArtifact) -> Result<Self> {
        self.kind = match self.kind {
            DistilWhisperAsrKind::Delegate(asr) => {
                DistilWhisperAsrKind::Delegate(asr.with_coreml_artifact(artifact)?)
            }
            DistilWhisperAsrKind::Scaffold(_) => {
                return Err(VokraError::UnsupportedOp(
                    "distil-whisper CoreML artifact requires from_gguf; the config-only scaffold has no executable weights"
                        .to_owned(),
                ));
            }
        };
        Ok(self)
    }

    /// The resolved configuration.
    #[must_use]
    pub fn config(&self) -> &DistilWhisperConfig {
        &self.cfg
    }

    /// True iff this handle was built via [`Self::from_gguf`] (real
    /// Whisper delegation is bound). Scaffold path ([`Self::new`]) is
    /// `false`.
    #[must_use]
    pub fn has_weights_bound(&self) -> bool {
        matches!(self.kind, DistilWhisperAsrKind::Delegate(_))
    }

    /// True iff the *scaffold* weight store was built by
    /// [`DistilWhisperWeights::synthesized`]. Returns `false` on the
    /// delegate path ([`Self::from_gguf`] loads real Whisper weights,
    /// which are by definition not synthesized).
    #[must_use]
    pub fn is_synthesized(&self) -> bool {
        match &self.kind {
            DistilWhisperAsrKind::Scaffold(w) => w.is_synthesized,
            DistilWhisperAsrKind::Delegate(_) => false,
        }
    }

    /// Transcribes a mono `f32` PCM slice at [`Self::config`]'s sample
    /// rate (16 kHz — [`DISTIL_WHISPER_SAMPLE_RATE`]).
    ///
    /// # Path-dependent behavior
    ///
    /// - **Delegate path** ([`Self::from_gguf`]): delegates to the shared
    ///   Whisper greedy decode (log-mel front-end → 32-layer encoder →
    ///   distil-shrunk decoder → byte-level BPE ids). Real forward.
    /// - **Scaffold path** ([`Self::new`]): hard-errors with
    ///   [`VokraError::NotImplemented`] naming `from_gguf` (or real weight
    ///   binding on the scaffold) as the fix. Both synthesized and
    ///   real-weight scaffolds error — the scaffold surface never invokes
    ///   the forward (its weight store is not wired to the shared Whisper
    ///   engine; use `from_gguf` for real ASR).
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] if `pcm` is empty.
    /// - [`VokraError::NotImplemented`] on the scaffold path (real forward
    ///   is only bound via [`Self::from_gguf`]).
    /// - Any error from [`WhisperAsr::transcribe_tokens`] on the delegate
    ///   path (backend unsupported, decoder failure, etc.).
    pub fn transcribe(&self, pcm: &[f32]) -> Result<Vec<u32>> {
        if pcm.is_empty() {
            return Err(VokraError::InvalidArgument(
                "distil-whisper transcribe: pcm slice is empty".to_owned(),
            ));
        }
        match &self.kind {
            DistilWhisperAsrKind::Delegate(asr) => asr.transcribe_tokens(pcm),
            DistilWhisperAsrKind::Scaffold(w) if w.is_synthesized => {
                Err(VokraError::NotImplemented(
                    "distil-whisper transcribe: this engine holds synthesized weights \
                     (deterministic fixture from DistilWhisperWeights::synthesized) — \
                     synthesized-weight text would be a hallucinated sequence, not a \
                     real transcript. Bind real distil-whisper weights (MIT, \
                     huggingface.co/distil-whisper/distil-large-v3.5) via \
                     DistilWhisperAsr::from_gguf(&GgufFile) instead of ::new(cfg, w). \
                     The shape flow (config validation with the distil invariant, \
                     weight-store construction, PCM boundary check) stays exercised \
                     through DistilWhisperAsr::new; real transcription delegates to \
                     the shared crate::whisper::WhisperAsr plumbing.",
                ))
            }
            DistilWhisperAsrKind::Scaffold(_) => Err(VokraError::NotImplemented(
                "distil-whisper transcribe: this handle was built from a \
                 shape-flow scaffold via DistilWhisperAsr::new (weights are not \
                 wired to the shared Whisper engine). Real transcription requires \
                 DistilWhisperAsr::from_gguf(&GgufFile) — distil-whisper is \
                 architecturally a Whisper checkpoint (only n_text_layer < \
                 n_audio_layer differs), so the forward delegates to the shared \
                 crate::whisper::WhisperAsr plumbing (op set STFT / mel filterbank / \
                 GEMM / GEMV / softmax / layer-norm / GELU / conv1d shared verbatim \
                 with vanilla Whisper). FR-EX-08: never a silent zero-fill.",
            )),
        }
    }

    /// Detokenizes `ids`. Delegates to [`WhisperAsr::render_ids`] on the
    /// delegate path; falls back to the bracketed id form on the scaffold
    /// path (matching the Whisper convention).
    pub fn render_ids(&self, ids: &[u32]) -> Result<String> {
        match &self.kind {
            DistilWhisperAsrKind::Delegate(asr) => asr.render_ids(ids),
            DistilWhisperAsrKind::Scaffold(_) => Ok(format!(
                "[no tokenizer; token ids: {}]",
                ids.iter().map(u32::to_string).collect::<Vec<_>>().join(" ")
            )),
        }
    }

    /// Test-only wrapper: build a Delegate-kind handle around an already-loaded
    /// [`WhisperAsr`] **without** enforcing the [`Self::from_gguf`] distil
    /// invariant (`n_text_layer < n_audio_layer`) or
    /// [`DistilWhisperConfig::validate_for_forward`]. Tests that exercise the
    /// [`AsrEngine`] trait dispatch (composition, empty-PCM early return) only
    /// need a Delegate-kind handle whose `transcribe` funnels through the
    /// shared Whisper engine — they do not exercise the mislabel-refusal path,
    /// which has its own dedicated `from_gguf_rejects_non_distil_shape_via_delegate_chain`
    /// coverage below.
    ///
    /// The config surfaced through [`Self::config`] mirrors the inner
    /// Whisper config verbatim (same shape as [`Self::from_gguf`]); this keeps
    /// [`Self::has_weights_bound`] `true` and [`Self::is_synthesized`] `false`
    /// so the handle behaves indistinguishably from a real GGUF load to
    /// downstream code that only reads the introspection surface.
    ///
    /// Not part of the public API (compiled only under `cfg(test)`).
    #[cfg(test)]
    pub(crate) fn from_whisper_asr_for_test(inner: WhisperAsr) -> Self {
        let wc = inner.model().config();
        let cfg = DistilWhisperConfig {
            n_mels: wc.n_mels,
            d_model: wc.d_model,
            n_audio_ctx: wc.n_audio_ctx,
            n_audio_head: wc.n_audio_head,
            n_audio_layer: wc.n_audio_layer,
            n_text_ctx: wc.n_text_ctx,
            n_text_head: wc.n_text_head,
            n_text_layer: wc.n_text_layer,
            n_vocab: wc.n_vocab,
            ffn_dim: wc.ffn_dim,
            eot: wc.eot,
            sot: wc.decoder_start_ids.first().copied().unwrap_or(50_258),
            sample_rate: DISTIL_WHISPER_SAMPLE_RATE,
        };
        Self {
            cfg,
            kind: DistilWhisperAsrKind::Delegate(inner),
        }
    }
}

/// [`AsrEngine`] blanket wiring so a distil-whisper handle can be injected via
/// [`vokra_core::Session::with_asr_engine`] and drive
/// `session.asr().transcribe()` end-to-end.
///
/// Composition — verbatim the [`WhisperAsr`] pattern
/// (`crates/vokra-models/src/whisper/asr.rs`):
/// 1. call the inherent [`DistilWhisperAsr::transcribe`] to get raw token ids
///    (delegate path → [`WhisperAsr::transcribe_tokens`] greedy;
///    scaffold path → loud [`VokraError::NotImplemented`]),
/// 2. render them through [`DistilWhisperAsr::render_ids`] (delegate path →
///    [`WhisperAsr::render_ids`]; scaffold path → the bracketed-id fallback),
/// 3. wrap the resulting `String` in a [`Transcription`].
///
/// Because the inherent method and this trait method share the receiver +
/// argument shape, method resolution inside the trait body picks the inherent
/// method first (return `Result<Vec<u32>>`), which is exactly the composition
/// leg we want — no explicit qualification needed and no accidental recursion.
///
/// The empty-PCM guard fires inside the inherent [`DistilWhisperAsr::transcribe`]
/// before either arm runs, so the trait method inherits the same
/// [`VokraError::InvalidArgument`] early return on `pcm.is_empty()` (FR-EX-08 —
/// never a silent empty transcription).
impl AsrEngine for DistilWhisperAsr {
    fn transcribe(&self, pcm: &[f32]) -> Result<Transcription> {
        let ids = self.transcribe(pcm)?;
        Ok(Transcription::new(self.render_ids(&ids)?))
    }

    /// Asks the delegate rather than storing a second copy: the backend is
    /// set through [`DistilWhisperAsr::with_backend`], which forwards to the
    /// inner [`WhisperAsr`], so a duplicate field here could disagree with
    /// the engine that actually runs.
    ///
    /// The scaffold arm reports `Cpu`, which cannot mislead in the way the
    /// trait warns about: it wires no forward at all, so there is no
    /// execution anywhere else for the answer to contradict.
    fn backend(&self) -> BackendKind {
        match &self.kind {
            DistilWhisperAsrKind::Delegate(asr) => asr.backend(),
            DistilWhisperAsrKind::Scaffold(_) => BackendKind::Cpu,
        }
    }
}

fn check_len(name: &str, got: usize, expected: usize) -> Result<()> {
    if got != expected {
        return Err(VokraError::InvalidArgument(format!(
            "distil-whisper weights: {name}.len()={got} != {expected}"
        )));
    }
    Ok(())
}

fn check_attn_shapes(
    context: &str,
    i: usize,
    attn: &DistilWhisperAttnProjWeights,
    d: usize,
) -> Result<()> {
    let m = d * d;
    check_len(&format!("{context}[{i}].q_proj"), attn.q_proj.len(), m)?;
    check_len(&format!("{context}[{i}].q_bias"), attn.q_bias.len(), d)?;
    check_len(&format!("{context}[{i}].k_proj"), attn.k_proj.len(), m)?;
    check_len(&format!("{context}[{i}].v_proj"), attn.v_proj.len(), m)?;
    check_len(&format!("{context}[{i}].v_bias"), attn.v_bias.len(), d)?;
    check_len(&format!("{context}[{i}].out_proj"), attn.out_proj.len(), m)?;
    check_len(&format!("{context}[{i}].out_bias"), attn.out_bias.len(), d)?;
    Ok(())
}

fn check_encoder_block_shapes(
    i: usize,
    blk: &DistilWhisperEncoderBlockWeights,
    d: usize,
    ffn: usize,
) -> Result<()> {
    check_len(
        &format!("encoder[{i}].self_attn_norm_gamma"),
        blk.self_attn_norm_gamma.len(),
        d,
    )?;
    check_len(
        &format!("encoder[{i}].self_attn_norm_beta"),
        blk.self_attn_norm_beta.len(),
        d,
    )?;
    check_attn_shapes("encoder.self_attn", i, &blk.self_attn, d)?;
    check_len(
        &format!("encoder[{i}].ffn_norm_gamma"),
        blk.ffn_norm_gamma.len(),
        d,
    )?;
    check_len(
        &format!("encoder[{i}].ffn_norm_beta"),
        blk.ffn_norm_beta.len(),
        d,
    )?;
    check_len(&format!("encoder[{i}].ffn_fc1"), blk.ffn_fc1.len(), d * ffn)?;
    check_len(
        &format!("encoder[{i}].ffn_fc1_bias"),
        blk.ffn_fc1_bias.len(),
        ffn,
    )?;
    check_len(&format!("encoder[{i}].ffn_fc2"), blk.ffn_fc2.len(), ffn * d)?;
    check_len(
        &format!("encoder[{i}].ffn_fc2_bias"),
        blk.ffn_fc2_bias.len(),
        d,
    )?;
    Ok(())
}

fn check_decoder_block_shapes(
    i: usize,
    blk: &DistilWhisperDecoderBlockWeights,
    d: usize,
    ffn: usize,
) -> Result<()> {
    check_len(
        &format!("decoder[{i}].self_attn_norm_gamma"),
        blk.self_attn_norm_gamma.len(),
        d,
    )?;
    check_len(
        &format!("decoder[{i}].self_attn_norm_beta"),
        blk.self_attn_norm_beta.len(),
        d,
    )?;
    check_attn_shapes("decoder.self_attn", i, &blk.self_attn, d)?;
    check_len(
        &format!("decoder[{i}].cross_attn_norm_gamma"),
        blk.cross_attn_norm_gamma.len(),
        d,
    )?;
    check_len(
        &format!("decoder[{i}].cross_attn_norm_beta"),
        blk.cross_attn_norm_beta.len(),
        d,
    )?;
    check_attn_shapes("decoder.cross_attn", i, &blk.cross_attn, d)?;
    check_len(
        &format!("decoder[{i}].ffn_norm_gamma"),
        blk.ffn_norm_gamma.len(),
        d,
    )?;
    check_len(
        &format!("decoder[{i}].ffn_norm_beta"),
        blk.ffn_norm_beta.len(),
        d,
    )?;
    check_len(&format!("decoder[{i}].ffn_fc1"), blk.ffn_fc1.len(), d * ffn)?;
    check_len(
        &format!("decoder[{i}].ffn_fc1_bias"),
        blk.ffn_fc1_bias.len(),
        ffn,
    )?;
    check_len(&format!("decoder[{i}].ffn_fc2"), blk.ffn_fc2.len(), ffn * d)?;
    check_len(
        &format!("decoder[{i}].ffn_fc2_bias"),
        blk.ffn_fc2_bias.len(),
        d,
    )?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Every hparam matches `huggingface.co/distil-whisper/distil-large-v3.5/
    /// raw/main/config.json` (fetched 2026-07-24).
    #[test]
    fn distil_large_v3_5_matches_primary_source_config_json() {
        let c = DistilWhisperConfig::distil_large_v3_5();
        // Encoder — identical to whisper-large-v3.
        assert_eq!(c.d_model, 1280);
        assert_eq!(c.n_audio_layer, 32);
        assert_eq!(c.n_audio_head, 20);
        assert_eq!(c.ffn_dim, 5120);
        assert_eq!(c.n_mels, 128);
        assert_eq!(c.n_audio_ctx, 1500);
        // Decoder — the distil axis.
        assert_eq!(
            c.n_text_layer, 2,
            "distil-large-v3.5 shrinks decoder to 2 layers"
        );
        assert_eq!(c.n_text_head, 20);
        assert_eq!(c.n_text_ctx, 448);
        // Tokenizer — large-v3 multilingual (+1 vocab for <|yue|>).
        assert_eq!(c.n_vocab, 51_866);
        assert_eq!(c.eot, 50_257);
        assert_eq!(c.sot, 50_258);
        // Audio boundary.
        assert_eq!(c.sample_rate, 16_000);
        // Derived — head_dim is the Whisper invariant.
        assert_eq!(
            c.head_dim(),
            64,
            "distil-large-v3.5 head_dim = 1280/20 = 64"
        );
        // Distil invariant holds.
        assert!(
            c.n_text_layer < c.n_audio_layer,
            "distil-large-v3.5 must have decoder < encoder depth"
        );
        c.validate_for_forward()
            .expect("distil-large-v3.5 is well-formed");
    }

    #[test]
    fn tiny_config_is_well_formed() {
        DistilWhisperConfig::tiny_for_tests()
            .validate_for_forward()
            .expect("tiny config is well-formed");
    }

    /// A checkpoint whose decoder depth equals the encoder depth is
    /// **not** distil-whisper — it is vanilla Whisper. The validator
    /// must catch this so a mis-flattened checkpoint (decoder tensors
    /// duplicated to the encoder count) fails loudly at
    /// `DistilWhisperAsr::new`, not silently deep in a forward.
    #[test]
    fn config_rejects_equal_encoder_decoder_depth() {
        let mut c = DistilWhisperConfig::tiny_for_tests();
        c.n_text_layer = c.n_audio_layer;
        let err = c
            .validate_for_forward()
            .expect_err("equal depth is not distil");
        assert!(
            matches!(err, VokraError::InvalidArgument(_)),
            "expected InvalidArgument, got {err:?}"
        );
    }

    #[test]
    fn config_rejects_decoder_larger_than_encoder() {
        let mut c = DistilWhisperConfig::tiny_for_tests();
        c.n_text_layer = c.n_audio_layer + 1;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_rejects_zero_axis() {
        for mutate in [
            |c: &mut DistilWhisperConfig| c.d_model = 0,
            |c: &mut DistilWhisperConfig| c.n_mels = 0,
            |c: &mut DistilWhisperConfig| c.n_audio_ctx = 0,
            |c: &mut DistilWhisperConfig| c.n_audio_layer = 0,
            |c: &mut DistilWhisperConfig| c.n_text_ctx = 0,
            |c: &mut DistilWhisperConfig| c.n_text_layer = 0,
            |c: &mut DistilWhisperConfig| c.n_vocab = 0,
            |c: &mut DistilWhisperConfig| c.ffn_dim = 0,
            |c: &mut DistilWhisperConfig| c.n_audio_head = 0,
            |c: &mut DistilWhisperConfig| c.n_text_head = 0,
            |c: &mut DistilWhisperConfig| c.sample_rate = 0,
        ] {
            let mut c = DistilWhisperConfig::tiny_for_tests();
            mutate(&mut c);
            assert!(matches!(
                c.validate_for_forward(),
                Err(VokraError::InvalidArgument(_))
            ));
        }
    }

    #[test]
    fn config_rejects_head_not_dividing_d_model() {
        let mut c = DistilWhisperConfig::tiny_for_tests();
        c.n_audio_head = 3; // 16 % 3 != 0
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_rejects_odd_head_dim() {
        let mut c = DistilWhisperConfig::tiny_for_tests();
        // 12 / 4 = 3 (odd).
        c.d_model = 12;
        c.n_audio_head = 4;
        c.n_text_head = 4;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_rejects_eot_or_sot_outside_vocab() {
        let mut c = DistilWhisperConfig::tiny_for_tests();
        c.eot = c.n_vocab as u32;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));

        let mut c = DistilWhisperConfig::tiny_for_tests();
        c.sot = c.n_vocab as u32 + 10;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn synthesized_weights_are_deterministic_and_shape_correct() {
        let c = DistilWhisperConfig::tiny_for_tests();
        let w1 = DistilWhisperWeights::synthesized(&c, 0x42).expect("build 1");
        let w2 = DistilWhisperWeights::synthesized(&c, 0x42).expect("build 2");
        // Determinism.
        assert_eq!(w1.conv1_weight, w2.conv1_weight);
        assert_eq!(
            w1.encoder_blocks[0].self_attn.q_proj,
            w2.encoder_blocks[0].self_attn.q_proj,
        );
        assert_eq!(
            w1.decoder_blocks[0].cross_attn.k_proj,
            w2.decoder_blocks[0].cross_attn.k_proj,
        );
        assert_eq!(w1.token_embed, w2.token_embed);
        assert!(w1.is_synthesized);

        // Shape flow.
        let d = c.d_model;
        let ffn = c.ffn_dim;
        assert_eq!(w1.conv1_weight.len(), d * c.n_mels * 3);
        assert_eq!(w1.conv1_bias.len(), d);
        assert_eq!(w1.conv2_weight.len(), d * d * 3);
        assert_eq!(w1.conv2_bias.len(), d);
        assert_eq!(w1.encoder_pos_embed.len(), c.n_audio_ctx * d);
        assert_eq!(w1.encoder_blocks.len(), c.n_audio_layer);
        for blk in &w1.encoder_blocks {
            assert_eq!(blk.self_attn.q_proj.len(), d * d);
            assert_eq!(blk.self_attn.k_proj.len(), d * d);
            assert_eq!(blk.self_attn.v_proj.len(), d * d);
            assert_eq!(blk.self_attn.out_proj.len(), d * d);
            assert_eq!(blk.self_attn.q_bias.len(), d);
            assert_eq!(blk.ffn_fc1.len(), d * ffn);
            assert_eq!(blk.ffn_fc2.len(), ffn * d);
        }
        assert_eq!(w1.token_embed.len(), c.n_vocab * d);
        assert_eq!(w1.decoder_pos_embed.len(), c.n_text_ctx * d);
        assert_eq!(w1.decoder_blocks.len(), c.n_text_layer);
        for blk in &w1.decoder_blocks {
            assert_eq!(blk.self_attn.q_proj.len(), d * d);
            assert_eq!(blk.cross_attn.q_proj.len(), d * d);
            assert_eq!(blk.ffn_fc1.len(), d * ffn);
        }
    }

    #[test]
    fn synthesized_weights_different_seeds_diverge() {
        let c = DistilWhisperConfig::tiny_for_tests();
        let a = DistilWhisperWeights::synthesized(&c, 1).expect("a");
        let b = DistilWhisperWeights::synthesized(&c, 2).expect("b");
        assert_ne!(a.conv1_weight, b.conv1_weight);
        assert_ne!(
            a.encoder_blocks[0].self_attn.q_proj,
            b.encoder_blocks[0].self_attn.q_proj,
        );
    }

    #[test]
    fn synthesized_rejects_ill_formed_config() {
        let mut c = DistilWhisperConfig::tiny_for_tests();
        c.n_text_layer = c.n_audio_layer;
        assert!(matches!(
            DistilWhisperWeights::synthesized(&c, 7),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn asr_new_accepts_matching_config_and_weights() {
        let c = DistilWhisperConfig::tiny_for_tests();
        let w = DistilWhisperWeights::synthesized(&c, 7).expect("weights");
        let asr = DistilWhisperAsr::new(c.clone(), w).expect("distil-whisper asr");
        assert_eq!(asr.config().d_model, c.d_model);
        assert_eq!(asr.config().n_text_layer, c.n_text_layer);
        assert!(asr.is_synthesized());
    }

    #[test]
    fn asr_new_rejects_encoder_layer_count_mismatch() {
        let c = DistilWhisperConfig::tiny_for_tests();
        let mut w = DistilWhisperWeights::synthesized(&c, 7).expect("weights");
        w.encoder_blocks.pop();
        assert!(matches!(
            DistilWhisperAsr::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn asr_new_rejects_decoder_layer_count_mismatch() {
        let c = DistilWhisperConfig::tiny_for_tests();
        let mut w = DistilWhisperWeights::synthesized(&c, 7).expect("weights");
        w.decoder_blocks.pop();
        assert!(matches!(
            DistilWhisperAsr::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn asr_new_rejects_conv1_shape_mismatch() {
        let c = DistilWhisperConfig::tiny_for_tests();
        let mut w = DistilWhisperWeights::synthesized(&c, 7).expect("weights");
        w.conv1_weight.pop();
        assert!(matches!(
            DistilWhisperAsr::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn asr_new_rejects_token_embed_shape_mismatch() {
        let c = DistilWhisperConfig::tiny_for_tests();
        let mut w = DistilWhisperWeights::synthesized(&c, 7).expect("weights");
        w.token_embed.pop();
        assert!(matches!(
            DistilWhisperAsr::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn asr_new_rejects_encoder_block_qkv_size_mismatch() {
        let c = DistilWhisperConfig::tiny_for_tests();
        let mut w = DistilWhisperWeights::synthesized(&c, 7).expect("weights");
        w.encoder_blocks[0].self_attn.q_proj.pop();
        assert!(matches!(
            DistilWhisperAsr::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn asr_new_rejects_decoder_cross_attn_size_mismatch() {
        let c = DistilWhisperConfig::tiny_for_tests();
        let mut w = DistilWhisperWeights::synthesized(&c, 7).expect("weights");
        w.decoder_blocks[0].cross_attn.v_proj.pop();
        assert!(matches!(
            DistilWhisperAsr::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn transcribe_rejects_empty_pcm() {
        let c = DistilWhisperConfig::tiny_for_tests();
        let w = DistilWhisperWeights::synthesized(&c, 7).expect("weights");
        let asr = DistilWhisperAsr::new(c, w).expect("distil-whisper asr");
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
        let c = DistilWhisperConfig::tiny_for_tests();
        let w = DistilWhisperWeights::synthesized(&c, 7).expect("weights");
        let asr = DistilWhisperAsr::new(c, w).expect("distil-whisper asr");
        let pcm = vec![0.0f32; 1024];
        let err = asr.transcribe(&pcm).unwrap_err();
        match err {
            VokraError::NotImplemented(msg) => {
                assert!(
                    msg.contains("synthesized"),
                    "message must name synthesized-weight blocker: {msg}"
                );
                assert!(msg.contains("distil"), "message must name the model: {msg}");
            }
            other => panic!("expected NotImplemented, got {other:?}"),
        }
    }

    #[test]
    fn expected_arch_is_distil_whisper() {
        assert_eq!(EXPECTED_ARCH, "distil-whisper");
    }

    #[test]
    fn sample_rate_matches_whisper_convention() {
        assert_eq!(DISTIL_WHISPER_SAMPLE_RATE, 16_000);
    }

    // ---------- from_gguf delegation tests (Wave 7 Part A RUNTIME-NOTIMPL) ----------

    use vokra_core::gguf::{GgufArray, GgufBuilder, GgufFile, GgufMetadataValue, GgufValueType};

    /// Builds a GGUF carrying a `vokra.whisper.*` chunk with the distil-shape
    /// (n_text_layer < n_audio_layer). No weight tensors — the delegate load
    /// path then fails on the front-end check (Whisper requires a
    /// `vokra.frontend.*` chunk), which is exactly the loud error we want to
    /// observe: the delegate is live and config parsing works.
    fn write_distil_shape_config(b: &mut GgufBuilder, n_audio_layer: u32, n_text_layer: u32) {
        b.add_u32("vokra.whisper.n_mels", 128);
        b.add_u32("vokra.whisper.n_audio_ctx", 1500);
        b.add_u32("vokra.whisper.n_audio_state", 1280);
        b.add_u32("vokra.whisper.n_audio_head", 20);
        b.add_u32("vokra.whisper.n_audio_layer", n_audio_layer);
        b.add_u32("vokra.whisper.n_text_ctx", 448);
        b.add_u32("vokra.whisper.n_text_state", 1280);
        b.add_u32("vokra.whisper.n_text_head", 20);
        b.add_u32("vokra.whisper.n_text_layer", n_text_layer);
        b.add_u32("vokra.whisper.n_vocab", 51_866);
        b.add_u32("vokra.whisper.ffn_dim", 5120);
        b.add_u32("vokra.whisper.eot", 50_257);
        b.add_metadata(
            "vokra.whisper.decoder_start_ids",
            GgufMetadataValue::Array(GgufArray {
                element_type: GgufValueType::U32,
                values: [50_258u32, 50_259, 50_359, 50_363]
                    .iter()
                    .map(|&id| GgufMetadataValue::U32(id))
                    .collect(),
            }),
        );
    }

    /// Scaffold path (`new`) reports `has_weights_bound() = false` and
    /// `transcribe` returns a NotImplemented naming `from_gguf` as the fix.
    #[test]
    fn scaffold_path_reports_no_delegate_bound() {
        let c = DistilWhisperConfig::tiny_for_tests();
        let w = DistilWhisperWeights::synthesized(&c, 42).expect("weights");
        let asr = DistilWhisperAsr::new(c, w).expect("distil-whisper asr");
        assert!(
            !asr.has_weights_bound(),
            "scaffold path must not have a delegate bound"
        );
        assert!(asr.is_synthesized());

        let err = asr.transcribe(&[0.0f32; 512]).unwrap_err();
        match err {
            VokraError::NotImplemented(msg) => {
                assert!(msg.contains("from_gguf"), "hint must name the fix: {msg}");
                assert!(
                    msg.contains("synthesized")
                        || msg.contains("distil")
                        || msg.contains("scaffold"),
                    "message must name the blocker: {msg}"
                );
            }
            other => panic!("expected NotImplemented, got {other:?}"),
        }
    }

    /// `from_gguf` delegates the load to `WhisperAsr::from_gguf`, which
    /// requires the `vokra.frontend.*` chunk. A shape-only GGUF fails as
    /// `ModelLoad` before any weight bind — this observes that the
    /// delegation wiring is live.
    #[test]
    fn from_gguf_delegates_and_reports_missing_frontend_chunk() {
        let mut b = GgufBuilder::new();
        write_distil_shape_config(&mut b, 32, 2);
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();

        match DistilWhisperAsr::from_gguf(&file) {
            Err(VokraError::ModelLoad(msg)) => assert!(!msg.is_empty()),
            Err(other) => {
                panic!("expected ModelLoad from the delegate Whisper load path, got {other:?}")
            }
            Ok(_) => panic!(
                "expected ModelLoad from the delegate Whisper load path, got Ok(_) \
                 — but this GGUF carries no weights (front-end check should fire)"
            ),
        }
    }

    /// A GGUF whose decoder is NOT smaller than the encoder is **not** a
    /// distil-whisper (vanilla Whisper large-v3 has 32/32). The distil
    /// invariant must fire — FR-EX-08, loud mislabel refusal. Whichever
    /// check (front-end / distil-invariant / weight-bind) fires first,
    /// the GGUF must not load as distil-whisper.
    #[test]
    fn from_gguf_rejects_non_distil_shape_via_delegate_chain() {
        let mut b = GgufBuilder::new();
        // Vanilla-shape: 6/6 (matches whisper base). NOT distil.
        write_distil_shape_config(&mut b, 6, 6);
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();

        assert!(
            DistilWhisperAsr::from_gguf(&file).is_err(),
            "matched-depth GGUF must not load as distil-whisper (FR-EX-08 mislabel refusal)"
        );
    }

    /// The M2-13 compliance registry must resolve every canonical
    /// distil-whisper id to Permissive (MIT). Cross-crate test to keep
    /// this module's registry-side contract honest.
    #[test]
    fn registry_lookup_maps_distil_whisper_to_permissive_mit() {
        use vokra_core::compliance::{LicenseClass, registry_lookup};
        for id in [
            "distil-whisper",
            "distil-whisper-large-v3",
            "distil-whisper-large-v3.5",
            "distil-large-v3",
            "distil-large-v3.5",
        ] {
            assert_eq!(
                registry_lookup(id),
                Some(LicenseClass::Permissive),
                "registry must map `{id}` to Permissive (MIT)"
            );
        }
    }

    // -------- AsrEngine trait dispatch tests (this task) --------
    //
    // These three tests prove the newly-added `impl AsrEngine for
    // DistilWhisperAsr` (a) actually reaches the shared Whisper delegate on
    // the delegate arm (not the scaffold NotImplemented arm), (b) honors the
    // empty-PCM early return through the trait method, and (c) composes to
    // the same text as the inherent `.transcribe(...)` → `.render_ids(...)`
    // pipeline — i.e. the trait method is a straight greedy composition of
    // the two inherent helpers, with no separate beam / sampling branch
    // introduced.

    // `WhisperAsr`, `AsrEngine`, `Transcription`, `VokraError` are all
    // already in scope via `use super::*;` above (they are top-level
    // `use`-imports in the parent module, which glob-import from the child
    // brings in transitively — the same way every earlier test in this
    // module references `VokraError` without re-importing). Only the
    // crate-private test-support helper needs an explicit import here.
    use crate::whisper::decoder::test_support::tiny_model_distil;

    /// Builds a delegate-kind `DistilWhisperAsr` wrapping a whisper-shape
    /// synthetic model (`n_audio_ctx = 1500` so the encoder passes its
    /// output-length check; 2 encoder layers, 1 decoder layer to keep the
    /// distil axis honest even though the test-only ctor bypasses the
    /// invariant check).
    fn delegate_asr() -> DistilWhisperAsr {
        let model = tiny_model_distil(2, 1);
        let inner = WhisperAsr::from_model_for_test(model);
        DistilWhisperAsr::from_whisper_asr_for_test(inner)
    }

    /// (a) The `AsrEngine::transcribe` trait method reaches the shared
    /// Whisper delegate (never the scaffold `NotImplemented` arm) and
    /// returns a bounded `Transcription` — no panic / hang, text length
    /// bounded by the greedy `DEFAULT_MAX_NEW_TOKENS = 224` cap × the
    /// bracketed-fallback per-id width.
    #[test]
    fn asr_engine_transcribe_delegate_returns_finite_transcription() {
        let asr = delegate_asr();
        // 1024 mono samples: the WhisperAsr log-mel front-end zero-pads to
        // its fixed 30 s window (N_FRAMES = 3000 frames) regardless, so any
        // non-empty PCM exercises the full PCM → mel → encoder → decoder
        // path.
        let pcm = vec![0.0f32; 1024];
        let out: Transcription = <DistilWhisperAsr as AsrEngine>::transcribe(&asr, &pcm)
            .expect("delegate AsrEngine::transcribe must return Ok(Transcription)");
        // Bounded (never NaN / infinite / DoS): greedy stops on eot within
        // DEFAULT_MAX_NEW_TOKENS = 224 iterations, so the bracketed-ids
        // render is at most a few KB even in the worst case.
        assert!(
            out.text.len() < 16 * 1024,
            "transcription text must stay bounded; got {} bytes",
            out.text.len()
        );
        // Not the loud NotImplemented scaffold shape (which returns Err,
        // not Ok) — Ok here on the delegate arm proves the trait dispatch
        // funnelled through the shared Whisper engine, not the scaffold's
        // hard-refusal arm.
    }

    /// (b) The `AsrEngine::transcribe` trait method honors the empty-PCM
    /// early return that the inherent method enforces, so a caller behind
    /// `dyn AsrEngine` (e.g. `session.asr().transcribe(&[])`) sees the same
    /// loud `InvalidArgument` — never a silent empty transcript (FR-EX-08).
    #[test]
    fn asr_engine_transcribe_rejects_empty_pcm() {
        let asr = delegate_asr();
        let err = <DistilWhisperAsr as AsrEngine>::transcribe(&asr, &[])
            .expect_err("trait method must reject empty PCM via the inherent early return");
        match err {
            VokraError::InvalidArgument(msg) => {
                assert!(
                    msg.contains("distil-whisper"),
                    "error must name the model: {msg}"
                );
                assert!(msg.contains("empty"), "error must name the blocker: {msg}");
            }
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    /// (c) The `AsrEngine::transcribe` trait method is exactly the
    /// composition `Transcription::new(self.render_ids(&self.transcribe(pcm)?)?)`
    /// — the transcript text is byte-identical to the manual pipeline, which
    /// proves the trait method introduced no separate beam / sampling
    /// branch and no post-processing beyond `render_ids`.
    #[test]
    fn asr_engine_transcribe_composes_with_inherent_transcribe() {
        let asr = delegate_asr();
        let pcm = vec![0.0f32; 1024];

        // Trait method: single-call, returns Transcription.
        let via_trait = <DistilWhisperAsr as AsrEngine>::transcribe(&asr, &pcm)
            .expect("trait transcribe must succeed on the delegate path");

        // Manual composition: inherent transcribe (Vec<u32>) → render_ids
        // (String) → Transcription::new. WhisperAsr::transcribe_tokens is
        // idempotent per-call (fresh KV cache, no RNG on greedy), so a
        // second call over the same PCM reproduces the same ids
        // deterministically.
        let ids = asr
            .transcribe(&pcm)
            .expect("inherent transcribe must succeed on the delegate path");
        let text = asr
            .render_ids(&ids)
            .expect("render_ids must succeed on the delegate path");
        let via_inherent = Transcription::new(text);

        assert_eq!(
            via_trait.text, via_inherent.text,
            "trait method must be a straight composition of inherent transcribe + render_ids",
        );
    }
}
