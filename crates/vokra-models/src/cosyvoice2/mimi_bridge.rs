//! # DEPRECATED (2026-07-24, SoTA plan §1(a) 訂正)
//!
//! **CosyVoice2 uses HiFTNet (Neural Source Filter + ISTFTNet), NOT Mimi.**
//! This module was built on a wrong premise. WebFetch to upstream
//! `cosyvoice/hifigan/generator.py:378` confirmed `class HiFTGenerator`
//! (docstring: `"HiFTNet Generator: Neural Source Filter + ISTFTNet"`) —
//! no Mimi consumer exists in the CosyVoice ツリー at all. Mimi is the
//! codec for Moshi and Sesame CSM-1B (arXiv:2410.00037 / arXiv:2410.14567),
//! never CosyVoice2 (arXiv:2412.10117). The full 訂正 is in the SoTA plan
//! `docs/tickets/sota-coverage-plan-2026-07-22.md` §1(a).
//!
//! Kept temporarily to avoid breaking existing test imports; use
//! [`crate::cosyvoice2::hift_chain::HiFTChain`] instead.
//!
//! # Original (wrong-premise) rationale, retained for historical context
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
//! already carries the entry. Note that the attribution requirement is
//! still binding on the Moshi / CSM consumers (which really do use Mimi);
//! this module's deprecation does not remove the attribution obligation
//! from those callers.
//!
//! # Scope of this (deprecated) scaffold (originally M3-09-T13)
//!
//! The concrete decode path was to land at T13: (a) bind the
//! `MimiRvqAttrs` from the CosyVoice2 config's `vokra.cosyvoice2.mimi.*`
//! chunk group, (b) load the codebook tables from the GGUF tensor slice,
//! (c) call [`vokra_ops::MimiDecoder::decode`] on each chunk. This
//! scaffold owns (a) — the attribute assembly from config — and returns
//! [`VokraError::NotImplemented`] on the code-→-features step until the
//! codebook tensor binding lands.
//!
//! Because the premise itself was wrong, the T13 real-codebook binding
//! will NOT be pursued for CosyVoice2 — the terminal vocoder migration
//! is `HiFTChain` (see [`crate::cosyvoice2::hift_chain`]).

// The deprecated APIs in this module are still consumed by the
// `chunk_pipeline` scaffold + the internal-oracle tests + the
// `parity_cosyvoice2` integration test. Silencing the deprecation
// warning module-wide (rather than at every call site) keeps the
// `-D warnings` gate green until those consumers migrate onto
// `HiFTChain`. New callers must not import from this module.
#![allow(deprecated)]

use vokra_core::{Result, VokraError};
use vokra_ops::{MimiDecoder, MimiRvqAttrs};

use super::config::CosyVoice2Config;

/// CosyVoice2 → Mimi RVQ bridge.
///
/// **DEPRECATED (2026-07-24, SoTA plan §1(a) 訂正).** CosyVoice2 does not
/// use the Mimi codec — the terminal vocoder is HiFTNet. Use
/// [`crate::cosyvoice2::hift_chain::HiFTChain`] instead. See the module
/// docstring for the full 訂正 rationale.
///
/// Owns the `MimiRvqAttrs` derived from the CosyVoice2 config, plus an
/// optional `MimiDecoder` handle (bound by the runtime once the codebook
/// tensors are read from the GGUF). Two construction paths are supported
/// today:
///
/// - [`MimiBridge::from_config`] — attrs-only, decoder deferred (the real
///   codebook binding is T13 follow-on).
/// - [`MimiBridge::with_identity_decoder`] — attrs + an identity fixture
///   decoder built by [`MimiDecoder::identity`], which lets the internal
///   oracle tests exercise the code-→-features seam end-to-end today
///   without a real Kyutai checkpoint (M3-06 identity smoke pattern).
#[deprecated(
    since = "0.1.0",
    note = "CosyVoice2 uses HiFTNet (Neural Source Filter + ISTFTNet), not Mimi. \
            Use crate::cosyvoice2::hift_chain::HiFTChain instead. See mimi_bridge \
            module docstring for the SoTA plan §1(a) 訂正 rationale."
)]
#[derive(Debug)]
pub struct MimiBridge {
    /// Shape attributes derived from the CosyVoice2 config's
    /// `vokra.cosyvoice2.mimi.*` chunk group (T04).
    attrs: MimiRvqAttrs,
    /// Codebook-owning decoder. `None` when the runtime has not bound the
    /// codebook tensors yet; `Some` after [`Self::with_identity_decoder`]
    /// or the T13 follow-on real-checkpoint binding.
    decoder: Option<MimiDecoder>,
}

impl MimiBridge {
    /// Builds a bridge from the CosyVoice2 config.
    ///
    /// **DEPRECATED (2026-07-24).** Use [`crate::cosyvoice2::hift_chain::HiFTChain::new`]
    /// instead — see the module docstring.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] if any of `n_codebooks` /
    /// `codebook_size` / `d_model` is `0` (FR-EX-08 — the runtime never
    /// silently accepts a degenerate codec shape; the converter is
    /// allowed to emit `0` placeholders during T02 upstream inspection,
    /// but the runtime rejects them on load).
    #[deprecated(
        since = "0.1.0",
        note = "Use HiFTChain::new instead. CosyVoice2 uses HiFTNet, not Mimi."
    )]
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
        Ok(Self {
            attrs,
            decoder: None,
        })
    }

    /// Builds a bridge whose decoder is the M3-06 identity fixture (row
    /// `i` puts a `1.0` at column `i mod d_model`, zero elsewhere;
    /// every codebook is the same identity). Useful for internal-oracle
    /// tests that verify the code-→-features seam without a real Kyutai
    /// checkpoint.
    ///
    /// The T13 follow-on replaces the identity decoder with one whose
    /// codebook tables are loaded from the CosyVoice2 GGUF tensor slice.
    ///
    /// **DEPRECATED (2026-07-24).** Use [`crate::cosyvoice2::hift_chain::HiFTChain`]
    /// instead.
    ///
    /// # Errors
    ///
    /// Propagates [`Self::from_config`] validation errors and
    /// [`MimiDecoder::identity`] shape-validation errors.
    #[deprecated(
        since = "0.1.0",
        note = "Use HiFTChain instead. CosyVoice2 uses HiFTNet, not Mimi."
    )]
    pub fn with_identity_decoder(config: &CosyVoice2Config) -> Result<Self> {
        let mut bridge = Self::from_config(config)?;
        let decoder = MimiDecoder::identity(bridge.attrs)?;
        bridge.decoder = Some(decoder);
        Ok(bridge)
    }

    /// Attaches an already-built [`MimiDecoder`] to a bridge; used by the
    /// T13 follow-on that will read codebook tensors directly from the
    /// GGUF.
    ///
    /// **DEPRECATED (2026-07-24).** Use [`crate::cosyvoice2::hift_chain::HiFTChain`]
    /// instead.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] if the decoder's attrs disagree
    /// with the bridge's — this is a shape sanity check that prevents a
    /// silently-mismatched decoder from producing corrupted features
    /// (FR-EX-08).
    #[deprecated(
        since = "0.1.0",
        note = "Use HiFTChain instead. CosyVoice2 uses HiFTNet, not Mimi."
    )]
    pub fn with_decoder(mut self, decoder: MimiDecoder) -> Result<Self> {
        if *decoder.attrs() != self.attrs {
            return Err(VokraError::InvalidArgument(format!(
                "cosyvoice2 mimi bridge: decoder attrs {:?} disagree with bridge \
                 attrs {:?}",
                decoder.attrs(),
                self.attrs
            )));
        }
        self.decoder = Some(decoder);
        Ok(self)
    }

    /// The Mimi RVQ shape attributes derived from the config.
    ///
    /// **DEPRECATED (2026-07-24).** Use [`crate::cosyvoice2::hift_chain::HiFTChain::config`]
    /// on the HiFTNet chain instead.
    #[deprecated(
        since = "0.1.0",
        note = "Use HiFTChain::config instead. CosyVoice2 uses HiFTNet, not Mimi."
    )]
    #[must_use]
    pub fn attrs(&self) -> &MimiRvqAttrs {
        &self.attrs
    }

    /// True iff a decoder has been bound (via
    /// [`Self::with_identity_decoder`] or [`Self::with_decoder`]).
    ///
    /// **DEPRECATED (2026-07-24).** Use
    /// [`crate::cosyvoice2::CosyVoice2Tts::has_hift_chain`] instead.
    #[deprecated(
        since = "0.1.0",
        note = "Use CosyVoice2Tts::has_hift_chain instead. CosyVoice2 uses HiFTNet, not Mimi."
    )]
    #[must_use]
    pub fn has_decoder(&self) -> bool {
        self.decoder.is_some()
    }

    /// Decodes a chunk of RVQ codes `[time, n_codebooks]` (row-major) into
    /// a `[time, d_model]` feature buffer.
    ///
    /// - If a decoder is bound (via [`Self::with_identity_decoder`] or
    ///   [`Self::with_decoder`]), delegates to
    ///   [`vokra_ops::MimiDecoder::decode`].
    /// - Otherwise returns [`VokraError::NotImplemented`] with a clear
    ///   next-step message (T13 follow-on binds the real codebook tables
    ///   from the CosyVoice2 GGUF tensor slice).
    ///
    /// The `codes` shape is checked up front so a caller with a
    /// wrong-length buffer gets a loud error today.
    ///
    /// **DEPRECATED (2026-07-24).** Use [`crate::cosyvoice2::hift_chain::HiFTChain::forward`]
    /// instead — the CosyVoice2 CFM emits a mel spectrogram, and HiFTNet
    /// (not the Mimi codec) is the terminal vocoder.
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] if `codes.len() != time *
    ///   attrs.n_codebooks` or if a bound decoder rejects an index;
    /// - [`VokraError::NotImplemented`] when no decoder is bound.
    #[deprecated(
        since = "0.1.0",
        note = "Use HiFTChain::forward instead. CosyVoice2 uses HiFTNet, not Mimi."
    )]
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
        match &self.decoder {
            Some(decoder) => decoder.decode(codes, time),
            None => Err(VokraError::NotImplemented(
                "CosyVoice2 → Mimi codebook tables are not bound in this bridge \
                 instance; build with `with_identity_decoder` for tests or \
                 `with_decoder` for the T13 follow-on real-checkpoint binding",
            )),
        }
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

    // ---- Identity decoder path (M3-09-T13 partial) ----------------------

    #[test]
    fn with_identity_decoder_binds_a_decoder_and_reports_it() {
        let cfg = stub_config_with_mimi_shape(4, 16, 8);
        let bridge = MimiBridge::with_identity_decoder(&cfg).expect("build");
        assert!(bridge.has_decoder(), "identity decoder must be bound");
    }

    #[test]
    fn identity_decoder_decode_matches_hand_folded_sum() {
        // With the M3-06 identity fixture (every codebook row `i` = one-hot
        // at column `i mod d_model`), decoding codes = [c; n_codebooks] at
        // one timestep sums `n_codebooks` one-hots — all landing at
        // `col = c mod d_model`. So the summed feature is `n_codebooks` at
        // that column and zero elsewhere. This is the same internal-oracle
        // invariant M3-06 asserts on its own.
        let cfg = stub_config_with_mimi_shape(3, 5, 4);
        let bridge = MimiBridge::with_identity_decoder(&cfg).expect("build");
        // At t=0, every codebook gets code=2 → summed row = 3.0 at col
        // (2 mod 4 = 2), 0 elsewhere.
        let codes = vec![2u32; 3];
        let out = bridge.decode_chunk(&codes, 1).expect("decode");
        assert_eq!(out.len(), 4, "one d_model-long row");
        let mut want = vec![0.0_f32; 4];
        want[2] = 3.0;
        assert_eq!(out, want);
    }

    #[test]
    fn identity_decoder_still_rejects_out_of_range_index() {
        // Even with a decoder bound, an out-of-range codebook index
        // remains an explicit error (FR-EX-08). The M3-06 CodebookTable
        // raises the error; the bridge's job is to propagate it, not
        // clamp.
        let cfg = stub_config_with_mimi_shape(3, 5, 4);
        let bridge = MimiBridge::with_identity_decoder(&cfg).expect("build");
        // codebook_size = 5, so index 5 is out of range.
        let mut codes = vec![0u32; 3];
        codes[1] = 5;
        let err = bridge
            .decode_chunk(&codes, 1)
            .expect_err("out-of-range must fail");
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn with_decoder_rejects_shape_mismatched_handoff() {
        // A caller who accidentally hands a decoder built with a
        // different `MimiRvqAttrs` must get a loud error rather than a
        // silently-wrong output (FR-EX-08).
        let cfg = stub_config_with_mimi_shape(3, 5, 4);
        let bridge = MimiBridge::from_config(&cfg).expect("build");
        // Different d_model on the decoder.
        let bad_attrs = MimiRvqAttrs {
            n_codebooks: 3,
            codebook_size: 5,
            d_model: 8,
        };
        let bad_decoder = MimiDecoder::identity(bad_attrs).expect("identity");
        let err = bridge
            .with_decoder(bad_decoder)
            .expect_err("shape mismatch must fail");
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }
}
