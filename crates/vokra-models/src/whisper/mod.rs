//! Whisper base — native encoder / decoder / beam search (M0-06).
//!
//! whisper.cpp-style native implementation: the model *definition* lives here
//! and only the upstream **checkpoint** is consumed, converted offline to
//! GGUF by `vokra-convert` (M0-03). No ONNX graph is loaded at runtime
//! (FR-LD-05, permanent). Hyperparameters come from the `vokra.*` GGUF
//! metadata, never hard-coded (FR-LD-02 / FR-MD-02).
//!
//! # Layout (M0-06)
//!
//! - [`config`] — [`WhisperConfig`], read from `vokra.whisper.*` metadata;
//! - [`weights`] — GGUF tensors bound to typed weight structs (owned f32; the
//!   `unsafe`-free reason is documented there);
//! - [`mel`] — the PCM → log-mel front-end (reuses the `vokra-ops` STFT + mel
//!   filter bank);
//! - [`nn`] — small forward helpers (linear / layer-norm / attention) built on
//!   the M0-08 `vokra-backend-cpu` kernels;
//! - [`encoder`] — conv stem + positional embedding + self-attention stack;
//! - [`decoder`] — token/positional embedding + causal self-attention (KV
//!   cache) + cross-attention + tied logits head;
//! - [`tokenizer`] — id ↔ text (byte-level BPE) for detokenization;
//! - [`greedy`] — greedy decode loop (special-token prefix, stop condition);
//! - [`asr`] — the [`vokra_core::engines::AsrEngine`] wired to
//!   `session.asr().transcribe()`.
//!
//! Search (`beam_search`) itself is model-independent and lives in
//! [`vokra_core::decode`]; this module supplies a `BeamScorer` from the
//! decoder (see [`decoder`]).
//!
//! # Operator inventory and gap analysis (M0-06-T02/T03)
//!
//! Every operator Whisper base needs was already available, so **no new
//! `vokra-ops` op or M0-08 kernel had to be added** — the gap list is empty:
//!
//! | need | provided by |
//! |------|-------------|
//! | STFT, mel filter bank | `vokra-ops` (M0-04): [`vokra_ops::stft`], [`vokra_ops::mel_filterbank`] |
//! | matmul / linear (bias) | `vokra-backend-cpu` (M0-08) `gemm_f32` |
//! | softmax, layer-norm | `vokra-backend-cpu` `softmax_f32`, `layer_norm_f32` |
//! | exact (erf) GELU | `vokra-backend-cpu` `gelu_f32` |
//! | conv1d (stem) | `vokra-backend-cpu` `conv1d_f32` (im2col + GEMM) |
//! | residual add | `vokra-backend-cpu` `add_f32` |
//! | embedding lookup, transpose, head split | plain indexing in [`nn`] / [`decoder`] (memory-bound, intentionally not kernels — M0-08 boundary note) |
//! | log-mel post-processing (log10 / clamp / range) | [`mel`] (Whisper-specific, not a general op) |
//! | causal / cross attention, KV cache, logits head | assembled here from the above |
//! | beam search | [`vokra_core::decode::beam_search`] (host-side, FR-OP-40) |
//!
//! The Whisper-specific `k_proj`-has-no-bias detail and the tied logits head
//! are handled in [`weights`] / [`decoder`], not as new ops.
//!
//! # Scope boundary
//!
//! - whisper.cpp-style native reimplementation: only the upstream safetensors
//!   checkpoint is consumed (FR-MD-02 / IF-06); no ONNX at runtime (FR-LD-05);
//! - the KV cache is a **model-internal** detail here; promoting it to a
//!   first-class session state (FR-EX-02) is M1-04;
//! - `frontend_spec` bit-exact **checking** (FR-LD-03) and `resample`
//!   (FR-OP-04) are M1 — this WP only *reads* `vokra.frontend.*` values and
//!   requires the input to already be at the model sample rate;
//! - word-level timestamps are a `beam_search` attribute (FR-OP-40) but not
//!   implemented in M0 (WP completion = demo + parity).

pub mod asr;
pub mod beam_glue;
pub mod config;
pub mod decoder;
pub mod encoder;
pub mod greedy;
pub mod mel;
pub mod nn;
pub mod tokenizer;
pub mod weights;

pub use asr::WhisperAsr;
pub use config::WhisperConfig;
pub use tokenizer::WhisperTokenizer;
pub use weights::WhisperWeights;

use vokra_core::Result;
use vokra_core::gguf::GgufFile;

use encoder::EncoderOutput;

/// A loaded Whisper model: validated config plus bound weights.
///
/// Construct with [`WhisperModel::from_gguf`]. The high-level transcription
/// entry point is [`WhisperAsr`] (the [`AsrEngine`](vokra_core::AsrEngine)
/// implementation); this type exposes the encoder / decoder forwards used by
/// the parity tests and by the search integration.
pub struct WhisperModel {
    config: WhisperConfig,
    weights: WhisperWeights,
}

impl WhisperModel {
    /// Loads config (`vokra.whisper.*`) and every weight tensor from `file`.
    ///
    /// # Errors
    ///
    /// [`VokraError::ModelLoad`] if a hyperparameter key or a weight tensor is
    /// missing, mistyped or mis-shaped.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let config = WhisperConfig::from_gguf(file)?;
        let weights = WhisperWeights::load(file, &config)?;
        Ok(Self { config, weights })
    }

    /// The model hyperparameters.
    pub fn config(&self) -> &WhisperConfig {
        &self.config
    }

    /// Runs the log-mel front-end on mono `pcm` at the model sample rate.
    ///
    /// Returns the `[n_mels, n_frames]` log-mel features (row-major). See
    /// [`mel::log_mel`] for the algorithm and its parity guarantees.
    pub fn log_mel(&self, pcm: &[f32]) -> Vec<f32> {
        mel::log_mel(pcm, self.config.n_mels)
    }

    /// Encodes `[n_mels, n_frames]` log-mel features into the encoder hidden
    /// states `[n_audio_ctx, d_model]`.
    pub fn encode(&self, log_mel: &[f32], n_frames: usize) -> Result<EncoderOutput> {
        encoder::encode(&self.config, &self.weights.encoder, log_mel, n_frames)
    }

    /// Convenience: PCM → log-mel → encoder hidden states.
    pub fn encode_pcm(&self, pcm: &[f32]) -> Result<EncoderOutput> {
        let n_frames = mel::N_FRAMES;
        let feats = self.log_mel(pcm);
        self.encode(&feats, n_frames)
    }

    /// Creates a decoder run bound to `encoder`, with fresh KV caches. Used by
    /// the greedy / beam drivers and by the decoder parity tests.
    pub fn decoder<'a>(&'a self, encoder: &EncoderOutput) -> Result<decoder::DecoderState<'a>> {
        decoder::DecoderState::new(&self.config, &self.weights.decoder, encoder)
    }

    /// Borrows the decoder weights / config for the [`decoder`] forward and the
    /// [`greedy`] / search drivers.
    pub(crate) fn decoder_state(&self) -> (&WhisperConfig, &weights::DecoderWeights) {
        (&self.config, &self.weights.decoder)
    }
}
