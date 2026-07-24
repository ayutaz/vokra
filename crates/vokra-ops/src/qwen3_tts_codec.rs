//! Qwen3-TTS codec (`qwen3_tts_codec`) RVQ decode primitive
//! (SoTA plan Phase 3 TTS codec primitive; FR-OP-30 posture — same family as
//! [`crate::mimi_rvq`] / [`crate::dac_rvq`] / [`crate::encodec_rvq`]).
//!
//! # What this module ships
//!
//! The **RVQ code → summed codec latent** decode step used by the Qwen3-TTS
//! codec that ships as a submodule of every released Qwen3-TTS-12Hz voice
//! ("Qwen/Qwen3-TTS-12Hz-{0.6B,1.7B}-{Base,CustomVoice,VoiceDesign}",
//! Apache-2.0). Given a set of per-quantizer `u32` code streams (one stream
//! per quantizer) the primitive gathers the corresponding codebook rows and
//! sums them in FP32:
//!
//! ```text
//!   decoded[t, :] = sum_{q=0..num_quantizers} tables[q].row(codes[q][t])
//! ```
//!
//! This is the same shape-generic FP32 residual fold as the RVQ family
//! (`mimi_rvq` / `encodec_rvq`); the difference from `dac_rvq` is that Qwen3-
//! TTS-Codec is **non-factorized** — each codebook entry is already the full
//! per-quantizer contribution, no per-quantizer output projection is applied
//! at decode time (that projection is the neural decoder's first layer, and
//! belongs to the consumer WP — same op boundary rule as ADR M4-04 §D-g
//! "features, not PCM").
//!
//! The output is `[time, codebook_dim]` row-major **codec features** — the
//! same latent stream the upstream neural decoder chain (transformer +
//! ConvTranspose1d upsampler → PCM) starts from. The transformer / upsample
//! stack is deliberately **out of scope** for this primitive (SoTA plan Phase
//! 3 description: "16 codebook RVQ at 12.5 Hz output rate; decoder chain
//! similar to DAC / Mimi"). The consumer WP wires the feature → PCM chain,
//! mirroring the split ADR M4-04 §D-g established for DAC and ADR M3-06 §D-b
//! for Mimi.
//!
//! # Structural distinction vs the rest of the RVQ family — the semantic
//! quantizer
//!
//! Qwen3-TTS-Codec differs from Mimi / EnCodec / DAC in one important way: it
//! is a **hybrid semantic + acoustic** RVQ. The first
//! [`Qwen3TtsCodecConfig::num_semantic_quantizers`] quantizers (=1 for every
//! released variant) use a **larger** vocabulary
//! ([`Qwen3TtsCodecConfig::semantic_codebook_size`] = 4096) than the
//! remaining "acoustic" quantizers ([`Qwen3TtsCodecConfig::codebook_size`] =
//! 2048). Every codebook still emits the same
//! [`Qwen3TtsCodecConfig::codebook_dim`]-wide row (=512), so the FP32
//! residual sum is well-defined; the asymmetry lives entirely in the per-
//! quantizer vocab size. This is *why* the op is not an alias of
//! [`crate::mimi_rvq::mimi_rvq_decode`] (whose [`crate::mimi_rvq::MimiRvqAttrs`]
//! carries a single `codebook_size` axis and cannot express the semantic /
//! acoustic split without silently clamping the semantic index).
//!
//! # Primary source (verbatim — nothing invented)
//!
//! Every shape / axis below comes from the Apache-2.0-licensed released
//! Qwen3-TTS-12Hz-0.6B-Base checkpoint's
//! `speech_tokenizer/config.json` (upstream repo
//! `Qwen/Qwen3-TTS-12Hz-0.6B-Base`, model_type
//! `qwen3_tts_tokenizer_12hz`; ships as a submodule of every released
//! Qwen3-TTS-12Hz voice, upstream config verified 2026-07-24). The relevant
//! subset:
//!
//! ```text
//!   {
//!     "architectures": ["Qwen3TTSTokenizerV2Model"],
//!     "model_type": "qwen3_tts_tokenizer_12hz",
//!     "encoder_valid_num_quantizers": 16,
//!     "input_sample_rate":            24000,
//!     "output_sample_rate":           24000,
//!     "decode_upsample_rate":         1920,
//!     "decoder_config": {
//!       "num_quantizers":          16,
//!       "num_semantic_quantizers": 1,
//!       "codebook_size":           2048,
//!       "semantic_codebook_size":  4096,
//!       "codebook_dim":            512,
//!       "vector_quantization_hidden_dimension": 512,
//!       ...
//!     },
//!     "encoder_config": {
//!       "_frame_rate": 12.5,
//!       ...
//!     }
//!   }
//! ```
//!
//! The 12.5 Hz frame rate is recoverable from `output_sample_rate /
//! decode_upsample_rate = 24000 / 1920 = 12.5`, and matches the encoder's
//! declared `_frame_rate`. [`Qwen3TtsCodecConfig::qwen3_tts_12hz`] returns
//! this canonical config verbatim so callers do not have to type the numbers
//! out again.
//!
//! # FP32 accumulator (audio-dialect rule)
//!
//! The residual sum is FP32-accumulated even if a future variant stores
//! codebook tables in FP16 / BF16 — the mixing precision follows the audio-
//! dialect rule of thumb ("BF16 mantissa loss is the real problem",
//! CLAUDE.md; same rule as [`crate::mimi_rvq`] / [`crate::dac_rvq`]).
//!
//! # No silent fallback (FR-EX-08)
//!
//! Any out-of-range index (`codes[q][t] >= per_quantizer_vocab(q)`), shape
//! mismatch, or config-vs-weights inconsistency is an explicit
//! [`VokraError::InvalidArgument`] — never a silent clamp. A wrong RVQ index
//! decodes into plausible-looking wrong audio downstream, so surfacing the
//! error at decode time is safer than producing silently-wrong output.
//!
//! # Runtime function — not an `OpKind` variant
//!
//! The primitive is a runtime function, not a [`vokra_core::OpKind`] variant.
//! Same two reasons as [`crate::mimi_rvq`] (module docs) / ADR M3-06 §D-b /
//! ADR M4-04 §D-b:
//!
//! 1. **Heterogeneous signature**: `&[Vec<u32>]` per-quantizer streams +
//!    borrowed codebook tables → `Vec<f32>` do not fit the
//!    [`crate::dispatch::OpValue`] `Real` / `Complex` dispatch surface, and
//!    threading a per-quantizer code slice through `dispatch` just to serve
//!    one op would tax every other op.
//! 2. **Consumer shape**: the planned consumer is an imperative Qwen3-TTS
//!    model WP (a future SoTA Phase 3 model integration) that already threads
//!    its own compute seam and wants the tight function API, not a graph-node
//!    round-trip (FR-EX-10 精神).
//!
//! # Reserved `OpKind` sentinel
//!
//! The reserved graph-op identifier will be registered as
//! `vokra_core::m5_residual_ops::QWEN3_TTS_CODEC_OP` in the M5-13 minimum-
//! dtype registry (same "reserve but do not register" posture as `flow_sample`
//! / `beam_search` / `ctc_decode`). No entry is added here — the primitive
//! ships as a pure runtime function until the model WP wires the seam.
//!
//! # GPU seam — kernel deferred, silent fallback forbidden
//!
//! `vokra-models/src/compute.rs` will expose a `HotOp::Qwen3TtsCodec` /
//! `Compute::qwen3_tts_codec_f32` seam alongside the existing `HotOp::MimiRvq`
//! / `HotOp::WavTokenizerVq` seams when the consumer WP lands. The CPU arm
//! delegates here; Metal / CUDA / Vulkan arms return an explicit
//! [`VokraError::UnsupportedOp`] until real kernels ship (FR-EX-08 — never a
//! silent CPU fall back). Because the fold is embedding-lookup + FP32 sum
//! per-timestep, a naive `blockDim.x = codebook_dim, gridDim.x = time` layout
//! is enough — same shape-generic kernel shape as [`crate::mimi_rvq`] L104-
//! 106. This module deliberately does not depend on `vokra-models` (the crate
//! edge runs `vokra-models → vokra-ops`), so no cross-crate intra-doc link is
//! written; the seam wiring is the consumer WP's concern.
//!
//! # GGUF metadata contract (documented — M5-13 EXPERIMENTAL target)
//!
//! A future Qwen3-TTS model WP's converter will bake this config into the
//! GGUF metadata chunks below (1:1 with [`Qwen3TtsCodecConfig`] fields). The
//! namespace is `vokra.qwen3_tts_codec.*` (single underscore in each dot
//! segment — matches the existing RVQ family convention: `vokra.mimi.*`,
//! `vokra.dac.*`, `vokra.encodec.*`):
//!
//! - `vokra.qwen3_tts_codec.num_quantizers` (`u32`)
//! - `vokra.qwen3_tts_codec.num_semantic_quantizers` (`u32`)
//! - `vokra.qwen3_tts_codec.codebook_size` (`u32`)
//! - `vokra.qwen3_tts_codec.semantic_codebook_size` (`u32`)
//! - `vokra.qwen3_tts_codec.codebook_dim` (`u32`)
//! - `vokra.qwen3_tts_codec.sample_rate` (`u32`)
//! - `vokra.qwen3_tts_codec.downsample_rate` (`u32`)
//!
//! The convention lines up with the `docs/abi-changelog.md` "GGUF Metadata
//! additions" pattern (fsq_codec.rs L110-119 rule) and is intended to land as
//! EXPERIMENTAL at the M5-13 C ABI / GGUF schema freeze so schema evolution
//! stays legal at minor bumps until the codec API stabilises.

use vokra_core::{Result, VokraError};

use crate::mimi_rvq::CodebookTable;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Static shape / rate attributes for a Qwen3-TTS-Codec decode.
///
/// Every field maps 1:1 to a name in the upstream
/// `speech_tokenizer/config.json` (module docs — verbatim primary source):
///
/// - [`Self::num_quantizers`]           ↔ `decoder_config.num_quantizers`
/// - [`Self::num_semantic_quantizers`]  ↔ `decoder_config.num_semantic_quantizers`
/// - [`Self::codebook_size`]            ↔ `decoder_config.codebook_size` (acoustic)
/// - [`Self::semantic_codebook_size`]   ↔ `decoder_config.semantic_codebook_size`
/// - [`Self::codebook_dim`]             ↔ `decoder_config.codebook_dim`
/// - [`Self::sample_rate`]              ↔ `output_sample_rate`
/// - [`Self::downsample_rate`]          ↔ `decode_upsample_rate`
///
/// The frame rate (12.5 Hz for every released variant) is recoverable from
/// `sample_rate / downsample_rate` via [`Self::frame_rate_hz`], and is
/// deliberately *not* stored on the struct — deriving it removes any risk of
/// the two rates drifting out of sync in a hand-built config.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Qwen3TtsCodecConfig {
    /// Total number of quantizers (semantic + acoustic). Released variant = 16.
    pub num_quantizers: usize,
    /// Number of "semantic" quantizers at the head of the RVQ stack. Released
    /// variant = 1; those quantizers use [`Self::semantic_codebook_size`] as
    /// their vocab (4096) instead of [`Self::codebook_size`] (2048).
    pub num_semantic_quantizers: usize,
    /// Acoustic per-codebook vocab size. Released variant = 2048.
    pub codebook_size: usize,
    /// Semantic per-codebook vocab size. Released variant = 4096.
    pub semantic_codebook_size: usize,
    /// Feature width per codebook entry (= the codec latent width). Released
    /// variant = 512.
    pub codebook_dim: usize,
    /// Output sample rate (Hz). Released variant = 24000.
    pub sample_rate: u32,
    /// Encoder / decoder time-downsample factor
    /// (samples-per-latent-frame). Released variant = 1920 (⇒ 12.5 Hz frame
    /// rate).
    pub downsample_rate: u32,
}

impl Qwen3TtsCodecConfig {
    /// Builds the released Qwen3-TTS-12Hz canonical config verbatim from the
    /// upstream `speech_tokenizer/config.json` (module-level primary source).
    ///
    /// Every released Qwen3-TTS-12Hz voice
    /// (`Qwen/Qwen3-TTS-12Hz-{0.6B,1.7B}-{Base,CustomVoice,VoiceDesign}`,
    /// Apache-2.0) ships the same codec submodule, so the same numbers apply
    /// to every 0.6B / 1.7B variant.
    ///
    /// Callers with a hypothetical future variant build the struct
    /// field-by-field from that variant's `speech_tokenizer/config.json`.
    #[inline]
    #[must_use]
    pub const fn qwen3_tts_12hz() -> Self {
        Self {
            num_quantizers: 16,
            num_semantic_quantizers: 1,
            codebook_size: 2048,
            semantic_codebook_size: 4096,
            codebook_dim: 512,
            sample_rate: 24_000,
            downsample_rate: 1_920,
        }
    }

    /// The codec latent frame rate in Hz — `sample_rate / downsample_rate`.
    /// Released variant = 12.5.
    #[inline]
    #[must_use]
    pub fn frame_rate_hz(&self) -> f32 {
        // Both fields are `u32` and the released variant does not overflow
        // `f32` precision; kept as an f32 divide (not a rational) because the
        // upstream config surfaces `_frame_rate` as a plain float.
        (self.sample_rate as f32) / (self.downsample_rate as f32)
    }

    /// Returns the per-quantizer vocab size (semantic vs acoustic) for
    /// quantizer index `q` (`0`-based, must be `< num_quantizers`). Semantic
    /// quantizers `[0, num_semantic_quantizers)` get
    /// [`Self::semantic_codebook_size`]; the rest get [`Self::codebook_size`].
    ///
    /// Returns `None` if `q >= num_quantizers` (a call-site out-of-range is
    /// programmer error; the pubic decode entry points surface it as an
    /// explicit [`VokraError::InvalidArgument`] instead — FR-EX-08).
    #[inline]
    #[must_use]
    pub fn quantizer_vocab_size(&self, q: usize) -> Option<usize> {
        if q >= self.num_quantizers {
            None
        } else if q < self.num_semantic_quantizers {
            Some(self.semantic_codebook_size)
        } else {
            Some(self.codebook_size)
        }
    }

    /// Validates every axis is non-zero and
    /// `num_semantic_quantizers <= num_quantizers`. Called by
    /// [`Qwen3TtsCodec::new`] and by the free-function decode entry points.
    fn validate(&self) -> Result<()> {
        if self.num_quantizers == 0
            || self.codebook_size == 0
            || self.semantic_codebook_size == 0
            || self.codebook_dim == 0
            || self.sample_rate == 0
            || self.downsample_rate == 0
        {
            return Err(VokraError::InvalidArgument(format!(
                "qwen3_tts_codec: config must have every axis > 0, got \
                 num_quantizers={} codebook_size={} semantic_codebook_size={} \
                 codebook_dim={} sample_rate={} downsample_rate={}",
                self.num_quantizers,
                self.codebook_size,
                self.semantic_codebook_size,
                self.codebook_dim,
                self.sample_rate,
                self.downsample_rate,
            )));
        }
        if self.num_semantic_quantizers > self.num_quantizers {
            return Err(VokraError::InvalidArgument(format!(
                "qwen3_tts_codec: num_semantic_quantizers {} > num_quantizers {}",
                self.num_semantic_quantizers, self.num_quantizers,
            )));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Consumer stub — Qwen3TtsCodec (session-like helper for the future model WP)
// ---------------------------------------------------------------------------

/// Host-side helper that owns a Qwen3-TTS-Codec's per-quantizer codebook
/// tables.
///
/// Keeping the helper here in `vokra-ops` avoids a reverse crate dependency
/// (`vokra-core` must not depend on `vokra-ops`; see [`crate::mimi_rvq`] L620-
/// 627 rationale). A future Qwen3-TTS model WP will forward its
/// [`vokra_core::Session`] entry points to a [`Qwen3TtsCodec`] loaded from the
/// GGUF's `vokra.qwen3_tts_codec.*` metadata and tensor chunks (module-level
/// GGUF metadata contract).
#[derive(Debug, Clone)]
pub struct Qwen3TtsCodec {
    config: Qwen3TtsCodecConfig,
    tables: Vec<CodebookTable>,
}

impl Qwen3TtsCodec {
    /// Builds a codec from an already-loaded config + per-quantizer codebook
    /// tables (semantic first, then acoustic).
    ///
    /// `weights.len()` must equal `config.num_quantizers`. The `i`-th entry
    /// must have `codebook_size == config.quantizer_vocab_size(i).unwrap()`
    /// and `d_model == config.codebook_dim`. Both axes are validated at
    /// construction so per-decode calls do not repay the cost.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on:
    /// - config axis validation (`config.validate`);
    /// - `weights.len() != num_quantizers`;
    /// - any per-quantizer shape mismatch (semantic entries must use
    ///   `semantic_codebook_size`; acoustic entries must use `codebook_size`;
    ///   both must use `codebook_dim`).
    pub fn new(config: Qwen3TtsCodecConfig, weights: Vec<CodebookTable>) -> Result<Self> {
        config.validate()?;
        check_weights_shape(&weights, &config)?;
        Ok(Self {
            config,
            tables: weights,
        })
    }

    /// Attribute snapshot.
    #[inline]
    #[must_use]
    pub const fn config(&self) -> &Qwen3TtsCodecConfig {
        &self.config
    }

    /// Read-only view of the codebook tables (used by tests and by the future
    /// Qwen3-TTS converter for round-trip audits).
    #[inline]
    #[must_use]
    pub fn tables(&self) -> &[CodebookTable] {
        &self.tables
    }

    /// Decodes a full block of per-quantizer code streams into a
    /// `[time, codebook_dim]` row-major codec-feature buffer.
    ///
    /// `codes.len()` must equal `config.num_quantizers`. Every inner
    /// `Vec<u32>` must have the same length (that length is `time`) — an
    /// asymmetric input is an explicit [`VokraError::InvalidArgument`]
    /// (FR-EX-08 — never a silent shortest-stream truncation). An empty input
    /// (`time == 0`) returns an empty `Vec` (not an error) — matches
    /// [`crate::mimi_rvq::mimi_rvq_read_summed_range`] gap semantics for
    /// zero-length windows.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on any of:
    /// - `codes.len() != config.num_quantizers`;
    /// - inner-length mismatch (see above);
    /// - `codes[q][t] >= config.quantizer_vocab_size(q)` (no silent clamp).
    pub fn decode(&self, codes: &[Vec<u32>]) -> Result<Vec<f32>> {
        qwen3_tts_codec_decode(codes, &self.tables, &self.config)
    }
}

// ---------------------------------------------------------------------------
// Core op — free function
// ---------------------------------------------------------------------------

/// Free-function form of [`Qwen3TtsCodec::decode`] — decodes per-quantizer
/// code streams into a `[time, codebook_dim]` row-major codec-feature buffer.
///
/// This is the shape-generic worker; [`Qwen3TtsCodec::decode`] is a thin
/// wrapper that reuses the codec's already-validated `config` + `tables`. The
/// symmetric split (free function + owning struct) mirrors
/// [`crate::mimi_rvq::mimi_rvq_decode`] / [`crate::mimi_rvq::MimiDecoder`] and
/// exists so a converter can call the fold directly without allocating a
/// [`Qwen3TtsCodec`] just to burn its tables afterwards.
///
/// # Errors
///
/// [`VokraError::InvalidArgument`] on any of:
/// - config axis validation (`config.validate`);
/// - `codebook_tables.len() != config.num_quantizers`;
/// - per-quantizer shape mismatch (see [`Qwen3TtsCodec::new`]);
/// - `codes.len() != config.num_quantizers`;
/// - inner-length mismatch across quantizer streams;
/// - `codes[q][t] >= config.quantizer_vocab_size(q)` (no silent clamp —
///   FR-EX-08).
pub fn qwen3_tts_codec_decode(
    codes: &[Vec<u32>],
    codebook_tables: &[CodebookTable],
    config: &Qwen3TtsCodecConfig,
) -> Result<Vec<f32>> {
    config.validate()?;
    check_weights_shape(codebook_tables, config)?;
    let time = check_codes_shape(codes, config)?;

    let d = config.codebook_dim;
    let mut out = vec![0.0_f32; time * d];
    for (q, table) in codebook_tables.iter().enumerate() {
        let stream = &codes[q];
        for (t, &idx) in stream.iter().enumerate() {
            // `CodebookTable::row` performs the per-index out-of-range check
            // (FR-EX-08). The check_weights_shape / check_codes_shape passes
            // above guarantee the outer axes match; only the per-code range is
            // dynamic.
            let row = table.row(idx)?;
            let base = t * d;
            // FP32 fold (see module docs — no FP16 / BF16 accumulator here).
            for (dst, src) in out[base..base + d].iter_mut().zip(row.iter()) {
                *dst += *src;
            }
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Shared shape checks
// ---------------------------------------------------------------------------

/// Validates that `codebook_tables` matches `config` — one table per
/// quantizer, semantic entries carry the semantic vocab, acoustic entries
/// carry the acoustic vocab, every table emits `codebook_dim`-wide rows.
fn check_weights_shape(
    codebook_tables: &[CodebookTable],
    config: &Qwen3TtsCodecConfig,
) -> Result<()> {
    if codebook_tables.len() != config.num_quantizers {
        return Err(VokraError::InvalidArgument(format!(
            "qwen3_tts_codec: codebook_tables.len() {} != config.num_quantizers {}",
            codebook_tables.len(),
            config.num_quantizers
        )));
    }
    for (q, table) in codebook_tables.iter().enumerate() {
        // Unwrap is safe: `q < codebook_tables.len() == num_quantizers`, so
        // `quantizer_vocab_size` is `Some(_)` by construction. The
        // `expect_msg` documents the invariant for future readers.
        let expected_vocab = config
            .quantizer_vocab_size(q)
            .expect("q < num_quantizers is enforced by the check above");
        let role = if q < config.num_semantic_quantizers {
            "semantic"
        } else {
            "acoustic"
        };
        if table.codebook_size != expected_vocab {
            return Err(VokraError::InvalidArgument(format!(
                "qwen3_tts_codec: codebook_tables[{q}] ({role}) codebook_size {} != \
                 expected {expected_vocab}",
                table.codebook_size,
            )));
        }
        if table.d_model != config.codebook_dim {
            return Err(VokraError::InvalidArgument(format!(
                "qwen3_tts_codec: codebook_tables[{q}] ({role}) d_model {} != \
                 config.codebook_dim {}",
                table.d_model, config.codebook_dim,
            )));
        }
    }
    Ok(())
}

/// Validates that `codes` is `[num_quantizers, time]` with a single `time`
/// value shared across every inner stream, and returns that `time`.
fn check_codes_shape(codes: &[Vec<u32>], config: &Qwen3TtsCodecConfig) -> Result<usize> {
    if codes.len() != config.num_quantizers {
        return Err(VokraError::InvalidArgument(format!(
            "qwen3_tts_codec: codes.len() {} != config.num_quantizers {}",
            codes.len(),
            config.num_quantizers
        )));
    }
    let time = codes[0].len();
    for (q, stream) in codes.iter().enumerate().skip(1) {
        if stream.len() != time {
            return Err(VokraError::InvalidArgument(format!(
                "qwen3_tts_codec: codes[{q}].len() {} != codes[0].len() {time} \
                 (per-quantizer streams must share the same time axis)",
                stream.len(),
            )));
        }
    }
    Ok(time)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Config primary-source pin ---------------------------------------

    #[test]
    fn config_canonical_matches_primary_source_speech_tokenizer_config() {
        // Verbatim from `Qwen/Qwen3-TTS-12Hz-0.6B-Base/speech_tokenizer/config.json`
        // (module-level primary source; upstream verified 2026-07-24).
        let c = Qwen3TtsCodecConfig::qwen3_tts_12hz();
        assert_eq!(c.num_quantizers, 16);
        assert_eq!(c.num_semantic_quantizers, 1);
        assert_eq!(c.codebook_size, 2048);
        assert_eq!(c.semantic_codebook_size, 4096);
        assert_eq!(c.codebook_dim, 512);
        assert_eq!(c.sample_rate, 24_000);
        assert_eq!(c.downsample_rate, 1_920);
        // 24000 / 1920 = 12.5 exactly.
        assert!((c.frame_rate_hz() - 12.5).abs() < 1e-6);
    }

    #[test]
    fn quantizer_vocab_size_semantic_vs_acoustic_split() {
        let c = Qwen3TtsCodecConfig::qwen3_tts_12hz();
        // First quantizer is semantic (num_semantic_quantizers = 1).
        assert_eq!(c.quantizer_vocab_size(0), Some(4096));
        // Rest are acoustic.
        for q in 1..16 {
            assert_eq!(c.quantizer_vocab_size(q), Some(2048), "quantizer {q}");
        }
        // Out-of-range is None (call-site programmer error).
        assert_eq!(c.quantizer_vocab_size(16), None);
    }

    #[test]
    fn config_validate_rejects_zero_axes() {
        let mut c = Qwen3TtsCodecConfig::qwen3_tts_12hz();
        c.num_quantizers = 0;
        assert!(matches!(c.validate(), Err(VokraError::InvalidArgument(_))));

        let mut c = Qwen3TtsCodecConfig::qwen3_tts_12hz();
        c.codebook_dim = 0;
        assert!(matches!(c.validate(), Err(VokraError::InvalidArgument(_))));

        let mut c = Qwen3TtsCodecConfig::qwen3_tts_12hz();
        c.sample_rate = 0;
        assert!(matches!(c.validate(), Err(VokraError::InvalidArgument(_))));

        let mut c = Qwen3TtsCodecConfig::qwen3_tts_12hz();
        c.downsample_rate = 0;
        assert!(matches!(c.validate(), Err(VokraError::InvalidArgument(_))));
    }

    #[test]
    fn config_validate_rejects_semantic_gt_total() {
        let mut c = Qwen3TtsCodecConfig::qwen3_tts_12hz();
        c.num_semantic_quantizers = c.num_quantizers + 1;
        assert!(matches!(c.validate(), Err(VokraError::InvalidArgument(_))));
    }

    // ---- Tiny fixture --------------------------------------------------

    /// Tiny attrs: 3 quantizers (1 semantic + 2 acoustic), 5-entry semantic
    /// vocab, 4-entry acoustic vocab, feature width 3. Small enough to
    /// hand-fold, distinct enough to catch a silent swap of semantic vs
    /// acoustic.
    fn tiny_config() -> Qwen3TtsCodecConfig {
        Qwen3TtsCodecConfig {
            num_quantizers: 3,
            num_semantic_quantizers: 1,
            codebook_size: 4,
            semantic_codebook_size: 5,
            codebook_dim: 3,
            sample_rate: 24_000,
            downsample_rate: 1_920,
        }
    }

    /// Deterministic codebooks: quantizer `q`, row `i` is
    /// `[q*100 + i*10, q*100 + i*10 + 1, q*100 + i*10 + 2]`. The semantic
    /// quantizer (q=0) has 5 rows; the two acoustic quantizers have 4 rows.
    fn tiny_tables(c: &Qwen3TtsCodecConfig) -> Vec<CodebookTable> {
        let mut tables = Vec::with_capacity(c.num_quantizers);
        for q in 0..c.num_quantizers {
            let vocab = c.quantizer_vocab_size(q).unwrap();
            let mut data = vec![0.0_f32; vocab * c.codebook_dim];
            for i in 0..vocab {
                for d in 0..c.codebook_dim {
                    data[i * c.codebook_dim + d] =
                        (q as f32) * 100.0 + (i as f32) * 10.0 + (d as f32);
                }
            }
            tables.push(CodebookTable::new(vocab, c.codebook_dim, data).unwrap());
        }
        tables
    }

    // ---- Happy path -----------------------------------------------------

    #[test]
    #[allow(clippy::needless_range_loop)] // index-form hand fold mirrors the op's math 1:1
    fn decode_matches_hand_fold_across_semantic_and_acoustic() {
        let c = tiny_config();
        let tables = tiny_tables(&c);
        // 4 timesteps of per-quantizer codes; every index is in-range for
        // its quantizer's vocab.
        let codes = vec![
            vec![0, 1, 4, 2], // semantic quantizer, vocab 5 (max 4)
            vec![3, 0, 2, 1], // acoustic quantizer #0, vocab 4 (max 3)
            vec![1, 2, 3, 0], // acoustic quantizer #1, vocab 4 (max 3)
        ];

        let got = qwen3_tts_codec_decode(&codes, &tables, &c).unwrap();
        assert_eq!(got.len(), 4 * c.codebook_dim);

        // Hand fold: same scalar loop, written independently.
        let mut want = vec![0.0_f32; 4 * c.codebook_dim];
        for q in 0..c.num_quantizers {
            for t in 0..4 {
                let idx = codes[q][t] as usize;
                let row_base = idx * c.codebook_dim;
                let out_base = t * c.codebook_dim;
                for d in 0..c.codebook_dim {
                    want[out_base + d] += tables[q].data[row_base + d];
                }
            }
        }
        assert_eq!(
            got, want,
            "qwen3_tts_codec_decode must be a bit-identical FP32 fold"
        );
    }

    #[test]
    fn decode_single_frame_matches_manual_arithmetic() {
        // 2 quantizers (1 semantic + 1 acoustic), 2 semantic entries, 2
        // acoustic entries, feature width 2. Feature values are chosen so the
        // hand-computed sum is exact in f32 (small integers).
        let c = Qwen3TtsCodecConfig {
            num_quantizers: 2,
            num_semantic_quantizers: 1,
            codebook_size: 2,
            semantic_codebook_size: 2,
            codebook_dim: 2,
            sample_rate: 24_000,
            downsample_rate: 1_920,
        };
        // Semantic table rows: [1, 2], [3, 4].
        let semantic = CodebookTable::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        // Acoustic table rows: [10, 20], [30, 40].
        let acoustic = CodebookTable::new(2, 2, vec![10.0, 20.0, 30.0, 40.0]).unwrap();

        // codes: t=0 -> semantic row 1 ([3,4]) + acoustic row 0 ([10,20]) = [13, 24]
        let codes = vec![vec![1_u32], vec![0_u32]];
        let out = qwen3_tts_codec_decode(&codes, &[semantic, acoustic], &c).unwrap();
        assert_eq!(out, vec![13.0, 24.0]);
    }

    #[test]
    fn decode_via_codec_struct_matches_free_function() {
        let c = tiny_config();
        let tables = tiny_tables(&c);
        let codec = Qwen3TtsCodec::new(c, tables.clone()).unwrap();
        assert_eq!(codec.config().num_quantizers, c.num_quantizers);
        assert_eq!(codec.tables().len(), tables.len());

        let codes = vec![vec![4, 0], vec![1, 3], vec![2, 0]];
        let via_struct = codec.decode(&codes).unwrap();
        let via_free = qwen3_tts_codec_decode(&codes, &tables, &c).unwrap();
        assert_eq!(via_struct, via_free);
    }

    // ---- Weights / config shape validation -----------------------------

    #[test]
    fn new_rejects_wrong_number_of_weight_tables() {
        let c = tiny_config();
        let mut tables = tiny_tables(&c);
        tables.pop(); // now 2, not 3
        assert!(matches!(
            Qwen3TtsCodec::new(c, tables),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn new_rejects_semantic_vocab_size_swapped_for_acoustic() {
        let c = tiny_config();
        // Build tables with the semantic quantizer sized as acoustic — the
        // shape check must catch it.
        let mut tables: Vec<CodebookTable> = Vec::with_capacity(c.num_quantizers);
        // Semantic slot (q=0) built with acoustic vocab (4) — WRONG.
        tables.push(
            CodebookTable::new(
                c.codebook_size,
                c.codebook_dim,
                vec![0.0; c.codebook_size * c.codebook_dim],
            )
            .unwrap(),
        );
        // The rest are correct.
        for _ in 1..c.num_quantizers {
            tables.push(
                CodebookTable::new(
                    c.codebook_size,
                    c.codebook_dim,
                    vec![0.0; c.codebook_size * c.codebook_dim],
                )
                .unwrap(),
            );
        }
        assert!(matches!(
            Qwen3TtsCodec::new(c, tables),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn new_rejects_wrong_feature_width() {
        let c = tiny_config();
        // Build one table with a wrong codebook_dim (2 instead of 3).
        let bad_dim = c.codebook_dim + 1;
        let bad_table = CodebookTable::new(
            c.semantic_codebook_size,
            bad_dim,
            vec![0.0; c.semantic_codebook_size * bad_dim],
        )
        .unwrap();
        let mut tables = tiny_tables(&c);
        tables[0] = bad_table;
        assert!(matches!(
            Qwen3TtsCodec::new(c, tables),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    // ---- Codes shape validation ----------------------------------------

    #[test]
    fn decode_rejects_wrong_number_of_code_streams() {
        let c = tiny_config();
        let tables = tiny_tables(&c);
        let codec = Qwen3TtsCodec::new(c, tables).unwrap();
        // Only 2 streams provided; config expects 3.
        let codes = vec![vec![0_u32], vec![1_u32]];
        assert!(matches!(
            codec.decode(&codes),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn decode_rejects_mismatched_inner_stream_lengths() {
        let c = tiny_config();
        let tables = tiny_tables(&c);
        let codec = Qwen3TtsCodec::new(c, tables).unwrap();
        // Streams 0 and 1 are length 3; stream 2 is length 2 → mismatch.
        let codes = vec![vec![0, 1, 2], vec![1, 2, 0], vec![0, 1]];
        assert!(matches!(
            codec.decode(&codes),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn decode_rejects_out_of_range_semantic_index() {
        let c = tiny_config();
        let tables = tiny_tables(&c);
        let codec = Qwen3TtsCodec::new(c, tables).unwrap();
        // Semantic vocab is 5 (indices 0..=4); index 5 must fail with no
        // silent clamp (FR-EX-08).
        let codes = vec![vec![5_u32], vec![0_u32], vec![0_u32]];
        assert!(matches!(
            codec.decode(&codes),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn decode_rejects_out_of_range_acoustic_index() {
        let c = tiny_config();
        let tables = tiny_tables(&c);
        let codec = Qwen3TtsCodec::new(c, tables).unwrap();
        // Acoustic vocab is 4 (indices 0..=3); index 4 must fail — and the
        // semantic index 4 is *legal* for the semantic vocab of 5, so this
        // test pins that the per-quantizer vocab is enforced, not a global
        // clamp.
        let codes = vec![vec![4_u32], vec![4_u32], vec![0_u32]];
        assert!(matches!(
            codec.decode(&codes),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn decode_accepts_semantic_index_that_would_overflow_acoustic_vocab() {
        // Positive pin of the asymmetry: semantic index 4 is in-range for the
        // semantic vocab of 5, but out-of-range if the primitive silently
        // clamped everything at the acoustic vocab of 4. Must succeed.
        let c = tiny_config();
        let tables = tiny_tables(&c);
        let codec = Qwen3TtsCodec::new(c, tables).unwrap();
        let codes = vec![vec![4_u32], vec![0_u32], vec![0_u32]];
        assert!(codec.decode(&codes).is_ok());
    }

    // ---- Edge cases ----------------------------------------------------

    #[test]
    fn decode_empty_input_returns_empty_vec() {
        let c = tiny_config();
        let tables = tiny_tables(&c);
        // Every quantizer has a zero-length stream (time == 0). This should
        // return an empty vec (not an error) — matches the RVQ family's
        // empty-window semantics (mimi_rvq.rs L599-600).
        let codes: Vec<Vec<u32>> = (0..c.num_quantizers).map(|_| Vec::new()).collect();
        let out = qwen3_tts_codec_decode(&codes, &tables, &c).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    #[allow(clippy::needless_range_loop)] // index-form hand fold mirrors the op's math 1:1
    fn decode_zero_only_streams_produces_row_sum_at_every_timestep() {
        // Every quantizer emits row 0 at every t; the decoded feature must be
        // the FP32 sum of every quantizer's row 0. Time > 0 to distinguish
        // from the empty case above.
        let c = tiny_config();
        let tables = tiny_tables(&c);
        let time = 3;
        let codes: Vec<Vec<u32>> = (0..c.num_quantizers).map(|_| vec![0_u32; time]).collect();

        let got = qwen3_tts_codec_decode(&codes, &tables, &c).unwrap();

        // Sum of row 0 of every quantizer's table.
        let mut want_row = vec![0.0_f32; c.codebook_dim];
        for table in &tables {
            for d in 0..c.codebook_dim {
                want_row[d] += table.data[d];
            }
        }
        // Repeat that row `time` times.
        let mut want = Vec::with_capacity(time * c.codebook_dim);
        for _ in 0..time {
            want.extend_from_slice(&want_row);
        }
        assert_eq!(got, want);
    }

    #[test]
    fn decode_rejects_config_with_zero_axis() {
        // The free-function entry point re-validates config even if the
        // caller skips the Qwen3TtsCodec wrapper. This guards direct
        // qwen3_tts_codec_decode users from a silently-malformed config.
        let mut c = tiny_config();
        c.codebook_dim = 0;
        let tables: Vec<CodebookTable> = Vec::new();
        let codes: Vec<Vec<u32>> = Vec::new();
        assert!(matches!(
            qwen3_tts_codec_decode(&codes, &tables, &c),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn decode_all_semantic_or_all_acoustic_configs_are_still_valid() {
        // If every quantizer is semantic (num_semantic_quantizers == num_quantizers)
        // the fold should just be an N-way sum over the semantic vocab; if none
        // are semantic, over the acoustic vocab. Pin both extremes.
        let mut c_all_sem = tiny_config();
        c_all_sem.num_semantic_quantizers = c_all_sem.num_quantizers;
        let tables: Vec<CodebookTable> = (0..c_all_sem.num_quantizers)
            .map(|_| {
                CodebookTable::new(
                    c_all_sem.semantic_codebook_size,
                    c_all_sem.codebook_dim,
                    vec![1.0_f32; c_all_sem.semantic_codebook_size * c_all_sem.codebook_dim],
                )
                .unwrap()
            })
            .collect();
        let codec = Qwen3TtsCodec::new(c_all_sem, tables).unwrap();
        let codes: Vec<Vec<u32>> = (0..c_all_sem.num_quantizers).map(|_| vec![0_u32]).collect();
        let out = codec.decode(&codes).unwrap();
        // Every codebook row is all-ones → summed feature = num_quantizers.
        assert_eq!(
            out,
            vec![c_all_sem.num_quantizers as f32; c_all_sem.codebook_dim]
        );

        let mut c_all_ac = tiny_config();
        c_all_ac.num_semantic_quantizers = 0;
        let tables: Vec<CodebookTable> = (0..c_all_ac.num_quantizers)
            .map(|_| {
                CodebookTable::new(
                    c_all_ac.codebook_size,
                    c_all_ac.codebook_dim,
                    vec![1.0_f32; c_all_ac.codebook_size * c_all_ac.codebook_dim],
                )
                .unwrap()
            })
            .collect();
        let codec = Qwen3TtsCodec::new(c_all_ac, tables).unwrap();
        let codes: Vec<Vec<u32>> = (0..c_all_ac.num_quantizers).map(|_| vec![0_u32]).collect();
        let out = codec.decode(&codes).unwrap();
        assert_eq!(
            out,
            vec![c_all_ac.num_quantizers as f32; c_all_ac.codebook_dim]
        );
    }

    #[test]
    #[allow(clippy::needless_range_loop)] // index-form hand fold mirrors the op's math 1:1
    fn decode_time_axis_is_row_major() {
        // Pin the output layout: `out[t*codebook_dim + d]` is the (t, d) cell,
        // never `out[d*time + t]`. A layout swap would break the neural-
        // decoder chain's consumer.
        let c = tiny_config();
        let tables = tiny_tables(&c);
        // Two distinct timesteps so the two rows differ.
        let codes = vec![vec![0_u32, 4_u32], vec![0_u32, 3_u32], vec![0_u32, 3_u32]];
        let out = qwen3_tts_codec_decode(&codes, &tables, &c).unwrap();

        // Row 0 = sum of every quantizer's row 0.
        let mut row0 = vec![0.0_f32; c.codebook_dim];
        for q in 0..c.num_quantizers {
            for d in 0..c.codebook_dim {
                row0[d] += tables[q].data[d];
            }
        }
        // Row 1 = sum of semantic row 4 + acoustic row 3 (both acoustic
        // quantizers).
        let mut row1 = vec![0.0_f32; c.codebook_dim];
        let idxs = [4_usize, 3, 3];
        for q in 0..c.num_quantizers {
            let base = idxs[q] * c.codebook_dim;
            for d in 0..c.codebook_dim {
                row1[d] += tables[q].data[base + d];
            }
        }

        assert_eq!(&out[0..c.codebook_dim], row0.as_slice());
        assert_eq!(&out[c.codebook_dim..2 * c.codebook_dim], row1.as_slice());
    }

    // ---- Time == 0 with wrong number of streams: shape check wins -----

    #[test]
    fn decode_rejects_wrong_stream_count_even_when_streams_are_empty() {
        // Even if every stream is empty, mismatched *count* is still an
        // error — this pins that the shape check runs before the length
        // check.
        let c = tiny_config();
        let tables = tiny_tables(&c);
        // Only two streams instead of three.
        let codes: Vec<Vec<u32>> = vec![Vec::new(), Vec::new()];
        assert!(matches!(
            qwen3_tts_codec_decode(&codes, &tables, &c),
            Err(VokraError::InvalidArgument(_))
        ));
    }
}
