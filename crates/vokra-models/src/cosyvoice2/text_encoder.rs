//! CosyVoice2 text encoder + LLM backbone — stub (M3-09-T07 / T08).
//!
//! The real text encoder + LLM backbone (embedding lookup → transformer
//! blocks → hidden states consumed by the Flow Matching CFM) is implemented
//! against the upstream safetensors manifest in the follow-on session
//! (T07 embedding / positional / stem; T08 transformer blocks + GEMM hot
//! path). This scaffold intentionally lands the **type + trait surface**
//! only, so a caller who wires an engine against this module receives an
//! explicit [`VokraError::NotImplemented`] on any forward attempt rather
//! than a silent zero-fill fallback (FR-EX-08).
//!
//! # Numeric parity strategy (follow-on)
//!
//! The follow-on sessions will:
//!
//! 1. Read the upstream CosyVoice2 safetensors on a build machine (T02
//!    upstream inspection, still open — this scaffold does not invent
//!    tensor names) and record each `tensor_name → shape/dtype` in
//!    `docs/adr/M3-09-cosyvoice2.md` §T02.
//! 2. Bind those tensors verbatim through
//!    [`vokra_core::gguf::GgufFile::get_tensor`] in a new
//!    `weights::TensorStore` (mirrors `piper_plus::weights::TensorStore`).
//! 3. Route the GEMM hot path through [`crate::compute::Compute::gemm_f32`]
//!    so the Metal / CUDA seams (T19/T20) offload without a second
//!    kernel path.

use vokra_core::{Result, VokraError};

use super::config::CosyVoice2Config;

/// Text encoder + LLM backbone — scaffold handle.
///
/// The struct itself holds no numeric state yet; the follow-on session adds
/// the tensor store + block cache. The public shape (`encode`) is stable
/// so callers can compile against the final surface today.
pub struct TextEncoderStub {
    /// Copy of the caller-provided config so shape validation can proceed
    /// without a live GGUF handle — the follow-on session replaces this
    /// with a `TensorStore` reference.
    #[allow(dead_code)] // consumed by T07/T08 numeric implementation
    config: CosyVoice2Config,
}

impl TextEncoderStub {
    /// Builds a stub bound to `config`. Never fails today; the follow-on
    /// implementation may return [`VokraError::InvalidArgument`] on a
    /// shape mismatch between `config` and the loaded weight tensors.
    #[must_use]
    pub fn new(config: CosyVoice2Config) -> Self {
        Self { config }
    }

    /// Encodes `token_ids` through the text embedding + LLM backbone and
    /// returns the per-token hidden features (`[t, hidden_dim]` row-major).
    ///
    /// # Errors
    ///
    /// This scaffold returns [`VokraError::NotImplemented`] unconditionally
    /// — the real forward path lands with T07/T08. The `token_ids` param
    /// is documented so callers can build the plumbing (tokenizer +
    /// batching) against the final signature today.
    pub fn encode(&self, token_ids: &[u32]) -> Result<Vec<f32>> {
        // Reference the arg so the intent is documented in-source.
        let _ = token_ids.len();
        Err(VokraError::NotImplemented(
            "CosyVoice2 text encoder + LLM backbone forward is not implemented in this \
             scaffold; T07 embedding / T08 transformer blocks / T09 unit test land the \
             numeric path against the upstream safetensors manifest",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::chunks::KEY_MODEL_ARCH;
    use vokra_core::gguf::{GgufBuilder, GgufFile};

    fn stub_config() -> CosyVoice2Config {
        let mut b = GgufBuilder::new();
        b.add_string(KEY_MODEL_ARCH, "cosyvoice2");
        b.add_u32(super::super::config::KEY_SAMPLE_RATE, 24_000);
        b.add_u32(super::super::config::KEY_VOCAB_SIZE, 32);
        b.add_u32(super::super::config::KEY_HIDDEN_DIM, 16);
        b.add_u32(super::super::config::KEY_N_LAYER, 2);
        b.add_u32(super::super::config::KEY_N_HEAD, 2);
        b.add_u32(super::super::config::KEY_FFN_DIM, 32);
        b.add_u32(super::super::config::KEY_FLOW_NFE, 4);
        b.add_string(super::super::config::KEY_FLOW_SCHEDULE, "linear");
        b.add_u32(super::super::config::KEY_MIMI_N_CODEBOOKS, 4);
        b.add_u32(super::super::config::KEY_MIMI_CODEBOOK_SIZE, 16);
        b.add_u32(super::super::config::KEY_MIMI_D_MODEL, 8);
        b.add_u32(super::super::config::KEY_STREAMING_CHUNK_SIZE, 4);
        b.add_u32(super::super::config::KEY_STREAMING_CHUNK_HOP, 4);
        let bytes = b.to_bytes().expect("serialize");
        let file = GgufFile::parse(bytes).expect("parse");
        CosyVoice2Config::from_gguf(&file).expect("read")
    }

    #[test]
    fn encode_returns_not_implemented_never_silent() {
        // No silent zero-fill fallback (FR-EX-08). The stub returns an
        // explicit NotImplemented on any call, so a caller who wires
        // against this scaffold today learns immediately that the
        // numeric path is not yet available.
        let enc = TextEncoderStub::new(stub_config());
        let err = enc
            .encode(&[1, 2, 3])
            .expect_err("scaffold must not produce features");
        assert!(matches!(err, VokraError::NotImplemented(_)));
    }

    #[test]
    fn encode_stub_accepts_empty_token_sequence() {
        // Degenerate but well-defined: an empty token sequence must still
        // produce the same NotImplemented error today (never an empty
        // Vec that could be misread as "encoded successfully").
        let enc = TextEncoderStub::new(stub_config());
        let err = enc
            .encode(&[])
            .expect_err("scaffold must not produce features");
        assert!(matches!(err, VokraError::NotImplemented(_)));
    }
}
