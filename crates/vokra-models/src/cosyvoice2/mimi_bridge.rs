//! CosyVoice2 → Mimi codec bridge — stub (M3-09-T13).
//!
//! CosyVoice2 emits residual vector-quantized codes (RVQ, `[time,
//! n_codebooks]` `u32` indices) that the Mimi codec decodes to a
//! `[time, d_model]` feature buffer via [`vokra_ops::mimi_rvq_decode`]
//! (M3-06 landed). The Mimi decoder chain then upsamples that feature
//! buffer to a 24 kHz PCM waveform.
//!
//! # M3-06 attribution — CC-BY 4.0
//!
//! Mimi is Kyutai Apache 2.0 **code** + CC-BY 4.0 attribution-required
//! **weight**. The attribution requirement is recorded in the top-level
//! `NOTICE` file (M3-06-T22 owner ticket) and in
//! `docs/license-audit.md` §3 (T26 owner ticket). This bridge does not
//! re-record the attribution — it consumes the M3-06 op family, which
//! already carries the entry.
//!
//! # Scope of this scaffold (T13)
//!
//! The concrete decode path lands with T13: (a) bind the `MimiRvqAttrs`
//! from the CosyVoice2 config's `vokra.cosyvoice2.mimi.*` chunk group,
//! (b) load the codebook tables from the GGUF tensor slice, (c) call
//! [`vokra_ops::MimiDecoder::decode`] on each chunk. This scaffold owns
//! (a) — the attribute assembly from config — and returns
//! [`VokraError::NotImplemented`] on the code-→-features step until the
//! codebook tensor binding lands.

use vokra_core::{Result, VokraError};
use vokra_ops::MimiRvqAttrs;

use super::config::CosyVoice2Config;

/// CosyVoice2 → Mimi RVQ bridge — scaffold handle.
///
/// Owns the `MimiRvqAttrs` derived from the CosyVoice2 config; the real
/// decoder (with codebook tensors) lands in the T13 follow-on.
#[derive(Debug)]
pub struct MimiBridge {
    /// Shape attributes derived from the CosyVoice2 config's
    /// `vokra.cosyvoice2.mimi.*` chunk group (T04).
    attrs: MimiRvqAttrs,
}

impl MimiBridge {
    /// Builds a bridge from the CosyVoice2 config.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] if any of `n_codebooks` /
    /// `codebook_size` / `d_model` is `0` (FR-EX-08 — the runtime never
    /// silently accepts a degenerate codec shape; the converter is
    /// allowed to emit `0` placeholders during T02 upstream inspection,
    /// but the runtime rejects them on load).
    pub fn from_config(config: &CosyVoice2Config) -> Result<Self> {
        let attrs = MimiRvqAttrs {
            n_codebooks: config.mimi_n_codebooks as usize,
            codebook_size: config.mimi_codebook_size as usize,
            d_model: config.mimi_d_model as usize,
        };
        if attrs.n_codebooks == 0 || attrs.codebook_size == 0 || attrs.d_model == 0 {
            return Err(VokraError::InvalidArgument(format!(
                "cosyvoice2 mimi bridge: degenerate MimiRvqAttrs {attrs:?} — the converter \
                 emitted `0` placeholders; T02 upstream inspection has to fill them before \
                 a runtime load succeeds"
            )));
        }
        Ok(Self { attrs })
    }

    /// The Mimi RVQ shape attributes derived from the config.
    #[must_use]
    pub fn attrs(&self) -> &MimiRvqAttrs {
        &self.attrs
    }

    /// Decodes a chunk of RVQ codes `[time, n_codebooks]` (row-major) into
    /// a `[time, d_model]` feature buffer.
    ///
    /// This scaffold returns [`VokraError::NotImplemented`] because the
    /// codebook tables are not yet bound (T13 follow-on will read them
    /// from the CosyVoice2 GGUF and hand a
    /// [`vokra_ops::MimiDecoder`] to this method). The `codes` shape is
    /// checked up front so a caller with a wrong-length buffer gets a
    /// loud error today.
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] if `codes.len() != time *
    ///   attrs.n_codebooks`;
    /// - [`VokraError::NotImplemented`] until T13 binds the codebook
    ///   tables.
    pub fn decode_chunk(&self, codes: &[u32], time: usize) -> Result<Vec<f32>> {
        let expected = time.checked_mul(self.attrs.n_codebooks).ok_or_else(|| {
            VokraError::InvalidArgument(format!(
                "cosyvoice2 mimi bridge: time·n_codebooks overflow (time={time}, \
                 n_codebooks={})",
                self.attrs.n_codebooks
            ))
        })?;
        if codes.len() != expected {
            return Err(VokraError::InvalidArgument(format!(
                "cosyvoice2 mimi bridge: codes length {} != time·n_codebooks {expected}",
                codes.len()
            )));
        }
        Err(VokraError::NotImplemented(
            "CosyVoice2 → Mimi codebook tables are not bound in this scaffold; T13 \
             follow-on binds them from the CosyVoice2 GGUF tensor slice and delegates \
             to vokra_ops::MimiDecoder::decode",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::chunks::KEY_MODEL_ARCH;
    use vokra_core::gguf::{GgufBuilder, GgufFile};

    fn stub_config_with_mimi_shape(n_cb: u32, cb_size: u32, d_model: u32) -> CosyVoice2Config {
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
        b.add_u32(super::super::config::KEY_MIMI_N_CODEBOOKS, n_cb);
        b.add_u32(super::super::config::KEY_MIMI_CODEBOOK_SIZE, cb_size);
        b.add_u32(super::super::config::KEY_MIMI_D_MODEL, d_model);
        b.add_u32(super::super::config::KEY_STREAMING_CHUNK_SIZE, 4);
        b.add_u32(super::super::config::KEY_STREAMING_CHUNK_HOP, 4);
        let bytes = b.to_bytes().expect("serialize");
        let file = GgufFile::parse(bytes).expect("parse");
        CosyVoice2Config::from_gguf(&file).expect("read")
    }

    #[test]
    fn config_zero_mimi_shape_fails_loudly() {
        // FR-EX-08: converter may emit `0` placeholders while T02 is
        // open, but the runtime rejects them at load time.
        let cfg = stub_config_with_mimi_shape(0, 2048, 512);
        let err = MimiBridge::from_config(&cfg).expect_err("n_codebooks=0 must fail");
        assert!(matches!(err, VokraError::InvalidArgument(_)));

        let cfg = stub_config_with_mimi_shape(8, 0, 512);
        let err = MimiBridge::from_config(&cfg).expect_err("codebook_size=0 must fail");
        assert!(matches!(err, VokraError::InvalidArgument(_)));

        let cfg = stub_config_with_mimi_shape(8, 2048, 0);
        let err = MimiBridge::from_config(&cfg).expect_err("d_model=0 must fail");
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn mimi_shape_matches_canonical_kyutai_defaults() {
        // The canonical Mimi shape (8 × 2048 × 512) round-trips through
        // the config; sanity check that the constants agree with
        // vokra_ops::MimiRvqAttrs::mimi().
        let cfg = stub_config_with_mimi_shape(8, 2048, 512);
        let bridge = MimiBridge::from_config(&cfg).expect("build");
        assert_eq!(*bridge.attrs(), MimiRvqAttrs::mimi());
    }

    #[test]
    fn decode_chunk_rejects_wrong_length() {
        // The runtime enforces `codes.len() == time · n_codebooks`
        // before any NotImplemented — the shape-mismatch error surface
        // is testable today.
        let cfg = stub_config_with_mimi_shape(8, 2048, 512);
        let bridge = MimiBridge::from_config(&cfg).expect("build");
        // 3 timesteps · 8 codebooks = 24; we pass 10 to force a mismatch.
        let bogus = vec![0u32; 10];
        let err = bridge
            .decode_chunk(&bogus, 3)
            .expect_err("wrong length must fail");
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn decode_chunk_correct_length_returns_not_implemented() {
        // With the right length the scaffold returns the honest
        // NotImplemented — never a silent zero-fill feature buffer.
        let cfg = stub_config_with_mimi_shape(8, 2048, 512);
        let bridge = MimiBridge::from_config(&cfg).expect("build");
        let codes = vec![0u32; 3 * 8];
        let err = bridge
            .decode_chunk(&codes, 3)
            .expect_err("scaffold must not produce features");
        assert!(matches!(err, VokraError::NotImplemented(_)));
    }
}
