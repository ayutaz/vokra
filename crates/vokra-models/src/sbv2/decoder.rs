//! SBV2 HiFi-GAN decoder wrapper: turns the VITS2 flow's acoustic latent
//! (`mel_hidden`) into a raw PCM waveform via the shared
//! [`vokra_ops::hifigan::hifigan_generator`] op (HiFi-GAN family neural
//! vocoder — Kong et al. 2020, arXiv:2010.05646; jik876/hifi-gan, MIT).
//! (Clean-room comment: see `mod.rs` — this module only calls the existing
//! shared `vokra_ops::hifigan` op; no SBV2/BV2 decoder source referenced.
//! `vits_ja`'s `VitsJaConfig::to_hifigan_attrs` is the only sibling model
//! that bridges into the same shared op — see that module's doc for the
//! identical "plain VITS decodes through a HiFi-GAN generator directly"
//! precedent this wrapper follows for SBV2/VITS2.)
//!
//! # No bundled "generator object" in `vokra_ops::hifigan`
//!
//! `hifigan_generator` is a free function over three separately-built
//! bundles — a weight bundle ([`HifiGanWeights`]), a shape/upsample-ladder
//! bundle ([`HifiGanAttrs`]), and a precision policy ([`HifiGanConfig`]) —
//! not a method on some `HiFiGanGenerator` struct. [`SbV2Decoder`] bundles
//! exactly those three plus a `sample_rate` metadata field.
//!
//! # SBV2 JP-Extra base config (target shape, not hard-coded)
//!
//! Style-Bert-VITS2 JP-Extra targets 44.1 kHz output through the
//! `upsample_rates = [8, 8, 2, 2]` ladder (total 256x time-domain
//! upsampling). Pairing that with `upsample_kernel_sizes = [16, 16, 4,
//! 4]` (the `kernel = 2 * stride` convention jik876/hifi-gan's V1/V2/V3
//! presets all follow) makes every upsample stage's output length land on
//! `(in_len - 1) * stride + kernel - 2 * ((kernel - stride) / 2) ==
//! in_len * stride` **exactly** (no rounding slack — `hifigan_generator`'s
//! transposed-conv formula, see that function's doc), so a full 4-stage
//! JP-Extra forward produces exactly `mel_seq_len * 256` samples.
//! [`SbV2Decoder`] does not hard-code this ladder: [`SbV2Decoder::new`]
//! takes whatever [`HifiGanAttrs`] the caller supplies, matching
//! `vits_ja`'s identical config-is-caller-supplied precedent for the same
//! shared op. The Task 24-27 converter is what will read the real ladder
//! out of an actual SBV2 JP-Extra checkpoint.

use vokra_ops::attrs::HifiGanAttrs;
use vokra_ops::hifigan::{HifiGanConfig, HifiGanWeights, hifigan_generator};

/// Thin wrapper over [`hifigan_generator`]: owns a pre-trained HiFi-GAN
/// weight / shape / precision bundle and exposes one
/// [`generate`](Self::generate) entry point that turns a VITS2 flow
/// output into a raw PCM waveform. See the module doc for why this struct
/// bundles three separate types rather than wrapping a single "generator
/// object" (`vokra_ops::hifigan` has none).
pub struct SbV2Decoder {
    /// Pre-trained HiFi-GAN weight bundle (conv_pre / upsample stack /
    /// MRF branches / conv_post — see [`HifiGanWeights`]'s field docs).
    weights: HifiGanWeights,
    /// Shape + upsample-ladder metadata (`n_mels`, `upsample_rates`, ...).
    /// [`generate`](Self::generate)'s `mel_hidden.len()` must equal
    /// `mel_seq_len * attrs.n_mels` (checked via `debug_assert!` —
    /// FR-EX-08).
    attrs: HifiGanAttrs,
    /// Precision policy (FP32 / FP16 mixed-precision; INT8 stays opt-in
    /// and gated — see [`HifiGanConfig`]'s doc). [`new`](Self::new) does
    /// not pick a default: the caller decides.
    config: HifiGanConfig,
    /// Output sample rate in Hz — informational metadata only, mirroring
    /// [`HifiGanAttrs::sample_rate`]'s own doc: neither field feeds the
    /// forward math, both exist so a caller can cross-check against the
    /// frontend_spec `sample_rate` (FR-LD-03). SBV2 JP-Extra targets
    /// 44,100 Hz.
    sample_rate: u32,
}

impl SbV2Decoder {
    /// Builds a decoder from a pre-trained HiFi-GAN weight bundle, its
    /// shape/upsample-ladder attributes, a precision policy, and the
    /// intended output sample rate.
    ///
    /// # Panics
    ///
    /// Panics (via `debug_assert!`, so only in debug builds — see
    /// [`StyleVectorInjector::from_projections`](super::style::StyleVectorInjector::from_projections)'s
    /// panic docs for why this crate uses `debug_assert!` rather than
    /// `Result` for constructor shape checks) if `attrs.validate_shape()`
    /// or `config.validate()` fails, or if `sample_rate !=
    /// attrs.sample_rate` (the two are expected to always agree — see the
    /// `sample_rate` field doc).
    pub fn new(
        weights: HifiGanWeights,
        attrs: HifiGanAttrs,
        config: HifiGanConfig,
        sample_rate: u32,
    ) -> Self {
        debug_assert!(
            attrs.validate_shape().is_ok(),
            "HifiGanAttrs must be internally consistent"
        );
        debug_assert!(
            config.validate().is_ok(),
            "HifiGanConfig must satisfy its INT8 opt-in invariant"
        );
        debug_assert_eq!(
            sample_rate, attrs.sample_rate,
            "SbV2Decoder::sample_rate must match attrs.sample_rate"
        );
        Self {
            weights,
            attrs,
            config,
            sample_rate,
        }
    }

    /// Output sample rate in Hz (see the `sample_rate` field doc).
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Generates a raw PCM waveform from a VITS2 flow output.
    ///
    /// `mel_hidden` is a `[n_mels, mel_seq_len]` row-major
    /// (**channel-major**) buffer — the exact layout
    /// [`hifigan_generator`]'s `mel` parameter documents (see that
    /// function's doc). Note this is channel-major, unlike the
    /// position-major `[T, C]` convention this crate's other `sbv2`
    /// modules use for their own row-major buffers (e.g.
    /// [`SbV2Flow::inverse`](super::flow::SbV2Flow::inverse)'s
    /// `[mel_seq_len, d_z]` output) — bridging one layout to the other is
    /// Task 23's integration concern, not this thin wrapper's.
    ///
    /// Returns a `[n_samples]` waveform bounded to `(−1, 1)` by
    /// `hifigan_generator`'s terminal `tanh`. For the SBV2 JP-Extra base
    /// config (`upsample_rates = [8, 8, 2, 2]`, `upsample_kernel_sizes =
    /// [16, 16, 4, 4]`), `n_samples == mel_seq_len * 256` exactly — see
    /// the module doc.
    ///
    /// # Panics
    ///
    /// Panics (via `debug_assert!` — FR-EX-08's "no silent fallback"
    /// reads as "no silently-wrong shape" here too, so a debug build
    /// catches a caller's `mel_seq_len` mismatch loudly) if
    /// `mel_hidden.len() != mel_seq_len * self.attrs.n_mels`. In a
    /// release build an inconsistent `mel_hidden` instead reaches
    /// [`hifigan_generator`]'s own `InvalidArgument` shape check, and
    /// this function turns that `Err` into a panic via `.expect(..)` —
    /// once `self`'s `attrs`/`config` were validated at construction time
    /// (see [`new`](Self::new)'s doc), `hifigan_generator` has no other
    /// caller-reachable error path here, so a `Result`-returning
    /// signature would only ever surface a programmer error, not a
    /// runtime condition — matching every other `sbv2` module's plain
    /// `Vec<f32>`-return convention (e.g.
    /// [`SbV2Flow::inverse`](super::flow::SbV2Flow::inverse)).
    pub fn generate(&self, mel_hidden: &[f32], mel_seq_len: usize) -> Vec<f32> {
        debug_assert_eq!(
            mel_hidden.len(),
            mel_seq_len * self.attrs.n_mels,
            "mel_hidden must be [n_mels, mel_seq_len] ({} * {}), got {}",
            self.attrs.n_mels,
            mel_seq_len,
            mel_hidden.len()
        );
        hifigan_generator(
            mel_hidden,
            mel_seq_len,
            &self.weights,
            &self.attrs,
            &self.config,
        )
        .expect(
            "SbV2Decoder::generate: hifigan_generator failed on a validated attrs/config bundle",
        )
    }
}
