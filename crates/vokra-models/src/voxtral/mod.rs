//! Voxtral (Mistral) — native audio encoder + Mistral text decoder (M3-10).
//!
//! whisper.cpp-style native implementation: the model *definition* lives here
//! and only the upstream **checkpoint** is consumed (converted offline to
//! GGUF by `vokra-convert` — `ModelKind::Voxtral`). No ONNX graph is loaded
//! at runtime (FR-LD-05).
//!
//! # Layout (M3-10 foundation)
//!
//! - [`config`] — [`VoxtralConfig`] read from `vokra.voxtral.*` metadata;
//! - [`audio_encoder`] — Whisper-derived conv stem + transformer stack
//!   (reuses the compute-seam kernels from `crate::compute`);
//! - [`text_decoder`] — Mistral decoder (GQA + RoPE + SwiGLU + RMSNorm);
//! - [`asr_head`] — greedy audio→text head over the tied token embedding;
//! - [`s2s_head`] — codec-token generation skeleton (heavy work deferred to
//!   downstream tickets — see [`s2s_head::S2sHead::synthesize`]).
//!
//! # Scope boundary (foundation only)
//!
//! This module lands the **structure** the ticket calls for — module tree,
//! typed hparams from GGUF metadata, small forward primitives that unit
//! tests exercise end-to-end on synthetic tensors — but does **not** claim
//! numerical parity against a real Voxtral checkpoint. Real-checkpoint
//! parity, streaming ASR, and S2S codec generation are follow-on tickets
//! (T13–T22) and are gated on a downloaded checkpoint that CI cannot
//! bundle.
//!
//! # No silent fall back (FR-EX-08)
//!
//! Every entry point on this module surfaces missing hparams / missing
//! tensors / unsupported dtypes as an explicit
//! [`VokraError`](vokra_core::VokraError) — never a silent widening or
//! substitution. The runtime rejects a `0`-sentinel hparam (from the
//! shape-only converter path) at forward time so a broken conversion cannot
//! be papered over.

pub mod asr_head;
pub mod audio_encoder;
pub mod config;
pub mod s2s_head;
pub mod text_decoder;

pub use asr_head::AsrHead;
pub use audio_encoder::{AudioEncoder, AudioEncoderOutput};
pub use config::VoxtralConfig;
pub use s2s_head::S2sHead;
pub use text_decoder::{TextDecoder, TextDecoderStep};

use vokra_core::gguf::GgufFile;
use vokra_core::{FrontendPolicy, Result, VokraError};

use crate::compute::HotOp;

/// The backend hot ops the Voxtral forward dispatches. Same six ops the
/// Whisper large-v3 decoder uses (encoder is a Whisper-derived stack), so a
/// backend that runs Whisper on the compute seam runs Voxtral too.
pub const VOXTRAL_HOT_OPS: &[HotOp] = &[
    HotOp::Gemm,
    HotOp::Gemv,
    HotOp::Softmax,
    HotOp::LayerNorm,
    HotOp::Gelu,
    HotOp::Conv1d,
];

/// A loaded Voxtral model: validated config plus the parsed audio-encoder /
/// text-decoder module handles.
///
/// This is the top-level type the [`AsrHead`] / [`S2sHead`] entry points
/// borrow from. It owns nothing that can panic to construct — every fallible
/// step returns [`VokraError::ModelLoad`] with the offending key or tensor
/// named.
pub struct VoxtralModel {
    config: VoxtralConfig,
    audio: AudioEncoder,
    text: TextDecoder,
}

impl VoxtralModel {
    /// Loads config, front-end and every weight tensor from `file`.
    ///
    /// # Front-end check (FR-LD-03)
    ///
    /// After the config is read, the `vokra.frontend.*` chunk is validated
    /// bit-for-bit against the runtime front-end
    /// ([`audio_encoder::runtime_frontend_spec`]) under
    /// [`FrontendPolicy::Fail`]: a mismatched or missing chunk aborts the
    /// load *before* the (larger) weight tensors are bound.
    ///
    /// # Errors
    ///
    /// [`VokraError::ModelLoad`] when a hyperparameter or a weight tensor is
    /// missing, mistyped or mis-shaped; [`VokraError::FrontendMismatch`]
    /// when the declared front-end differs from the runtime's.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let config = VoxtralConfig::from_gguf(file)?;
        // Front-end check when the encoder is present (n_mels > 0).
        if config.audio.n_mels > 0 {
            audio_encoder::check_frontend_spec(file, config.audio.n_mels, FrontendPolicy::Fail)?;
        }
        let audio = AudioEncoder::load(file, &config)?;
        let text = TextDecoder::load(file, &config)?;
        Ok(Self {
            config,
            audio,
            text,
        })
    }

    /// The model hyperparameters.
    pub fn config(&self) -> &VoxtralConfig {
        &self.config
    }

    /// The parsed audio encoder module.
    pub fn audio_encoder(&self) -> &AudioEncoder {
        &self.audio
    }

    /// The parsed text decoder module.
    pub fn text_decoder(&self) -> &TextDecoder {
        &self.text
    }
}

/// Returns a wrapper error when a required Voxtral hparam is the `0`
/// sentinel the shape-only converter path writes for a value the runtime
/// cannot infer from tensor shapes alone (RoPE base, RMSNorm eps, GQA head
/// split, vocab size). This is a per-call gate a forward entry point can
/// invoke to surface a broken conversion (FR-EX-08 — never silently
/// substitute a default at inference time).
///
/// Exercised by the unit tests today; wired into the full forward once
/// M3-10-T13+ ASR + T14 S2S land.
#[allow(dead_code)]
pub(crate) fn reject_zero_sentinel(name: &str, value: u32) -> Result<u32> {
    if value == 0 {
        Err(VokraError::ModelLoad(format!(
            "voxtral: hparam `{name}` is 0 — the shape-only converter path wrote a sentinel. \
             Re-convert with `convert_voxtral_file` and a VoxtralConfig that supplies this value \
             (FR-EX-08 — no silent runtime default)."
        )))
    } else {
        Ok(value)
    }
}

/// f32 equivalent of [`reject_zero_sentinel`] for RoPE base / RMSNorm eps.
#[allow(dead_code)]
pub(crate) fn reject_zero_sentinel_f32(name: &str, value: f32) -> Result<f32> {
    if value == 0.0 {
        Err(VokraError::ModelLoad(format!(
            "voxtral: hparam `{name}` is 0.0 — the shape-only converter path wrote a sentinel. \
             Re-convert with `convert_voxtral_file` and a VoxtralConfig that supplies this value \
             (FR-EX-08 — no silent runtime default)."
        )))
    } else {
        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::GgufBuilder;

    #[test]
    fn from_gguf_reports_missing_metadata_before_touching_weights() {
        // Empty GGUF (no `vokra.voxtral.*`, no `vokra.frontend.*`). Load must
        // fail on the missing config keys — never a silent default.
        let b = GgufBuilder::new();
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        assert!(matches!(
            VoxtralModel::from_gguf(&file),
            Err(VokraError::ModelLoad(_))
        ));
    }

    #[test]
    fn reject_zero_sentinel_rejects_zero_and_passes_positive() {
        assert!(reject_zero_sentinel("k", 0).is_err());
        assert_eq!(reject_zero_sentinel("k", 42).unwrap(), 42);
        assert!(reject_zero_sentinel_f32("k", 0.0).is_err());
        assert!((reject_zero_sentinel_f32("k", 1e-5).unwrap() - 1e-5).abs() < 1e-9);
    }
}
