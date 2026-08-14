//! **SNAC** (`hubertsiuzdak/snac_{24khz,44khz}`, MIT) — Multi-Scale Neural
//! Audio Codec (Siuzdak et al. 2024, arXiv:2410.14411) — runtime binder for
//! the `snac` converter arch.
//!
//! # Runtime layout (loud-partial, RMVPE + DNSMOS + openwakeword precedent)
//!
//! ```text
//! PCM (24 kHz or 44.1 kHz mono f32)
//!   -> Encoder Conv1D stack           ← **loud-partial**
//!        (encoder_rates=[2,4,8,8] Hz24 / [2,3,8,8] Hz44,
//!         512x / 384x downsampling, noise-conditioned residual,
//!         Snake activation)
//!   -> VectorQuantize.forward          ← **loud-partial**
//!        (avg_pool1d(stride) per stage, nearest-neighbour argmin
//!         over factorized codebook rows,
//!         local attention (attn_window_size=32) on Hz44 only)
//!   -> hierarchical codes (3 stages Hz24 / 4 stages Hz44)
//!   -> ResidualVectorQuantize.from_codes   ← REAL (vokra_ops::SnacDecoder)
//!        (codes → [T, d_model] intermediate features)
//!   -> Decoder Conv1D upsample stack   ← **loud-partial**
//!        (decoder_rates=[8,8,4,2] Hz24 / [8,8,3,2] Hz44,
//!         Snake activation, noise module, Hz44 local attention)
//!   -> PCM output
//! ```
//!
//! # Loud-partial classification (design § — CLAUDE.md 教訓 (a))
//!
//! - **Real (this WP)**: `Snac::from_gguf` (`vokra.model.arch=="snac"` +
//!   `vokra.snac.variant` validation + variant-driven [`SnacConfig`]
//!   exposure), variant accessors, license-class surfacing.
//! - **Loud-partial (this WP)**: [`Snac::encode`] / [`Snac::decode`] both
//!   return [`VokraError::UnsupportedOp`] naming the exact missing primitive
//!   (encoder Conv1D stack / decoder feature→PCM synthesis chain — neither
//!   exists in `vokra-ops` today, and both are follow-up WPs sized similarly
//!   to M4-04/M4-05 codec waves).
//! - **Deferred (converter-side)**: [`Snac::decode_codes_to_features`] shim
//!   surfaces the derived-tensor gap: the existing
//!   `vokra_ops::SnacDecoder::decode` primitive covers the intermediate
//!   RVQ codes → `[T, d_model]` features step, but the converter has not
//!   yet extended to emit derived
//!   `vokra.snac.codebook_tables` + `vokra.snac.quantizer.{i}.out_proj_*`
//!   tensors (offline weight-norm fold + factorization extraction — mirror
//!   of the M4-04 DAC converter T10/T11 pattern).
//!
//! Rationale (RMVPE precedent, CLAUDE.md 教訓 (a)): the surrounding scaffold
//! + `from_gguf` variant-round-trip + FR-EX-08 loud-fails land today so a
//!   follow-up wave can flip the switch by (i) extending
//!   `crates/vokra-convert/src/models/snac.rs` to emit the derived
//!   per-quantizer tensors (weight-norm folding of the upstream
//!   `weight_g` + `weight_v` parametrization), and (ii) writing the encoder /
//!   decoder body primitives against those tensors. The RMVPE
//!   `VokraError::UnsupportedOp` messages cite the primary source
//!   (`hubertsiuzdak/snac/blob/main/snac/snac.py` for the Encoder / Decoder
//!   and `snac/vq.py` for `VectorQuantize.forward`) so a reader diagnosing
//!   this gap has exactly one place to walk.
//!
//! # `vokra.snac.*` chunk group (read here)
//!
//! Written by `vokra-convert::models::snac::convert_snac_file`:
//!
//! - `vokra.model.arch` (`String`): must equal [`ARCH`] (`"snac"`).
//! - `vokra.model.name` (`String`): `"snac-24khz"` / `"snac-44khz"` per
//!   variant — auxiliary check.
//! - `vokra.snac.variant` (`String`): `"24khz"` / `"44khz"` — the
//!   discriminator the runtime dispatches on (mirrors
//!   `vokra.focalcodec.variant` + `vokra.bigvgan.variant`).
//! - `vokra.provenance.*`: license class + raw license string, so the
//!   runtime compliance gate (FR-CP-03) can classify the artifact without
//!   re-inspecting the safetensors provenance.
//!
//! # Cross-crate constant duplication (mirror of the converter's
//! [`ARCH`] / [`KEY_SNAC_VARIANT`] / variant-tag surface — same rule the
//! sibling BF16 pass-through binders (`fsmn_vad`, `openwakeword`, `dnsmos`)
//! use so `vokra-models` does not gain a dependency edge onto
//! `vokra-convert`, preserving the layered convention `vokra-ops → nothing
//! GGUF-aware`, `vokra-core → GGUF reader`, `vokra-models → GGUF binder`,
//! `vokra-convert → GGUF writer`).

use vokra_core::gguf::{GgufFile, chunks};
use vokra_core::{LicenseClass, Result, VokraError};

#[cfg(test)]
mod tests;

// ---------------------------------------------------------------------------
// Arch / variant / metadata-key constants — mirror of
// crates/vokra-convert/src/models/snac.rs (see the module docstring).
// ---------------------------------------------------------------------------

/// Expected `vokra.model.arch` value written by
/// `vokra-cli convert --model snac-{24khz,44khz}`.
///
/// Distinct from every sibling codec arch tag (`mimi`, `dac`,
/// `wavtokenizer`, `neucodec`, `funcodec`, `xcodec2`, `speechtokenizer`,
/// `bicodec`, `xy_tokenizer`, `focalcodec`, `step_audio2_mini`) because
/// SNAC is a multi-scale hierarchical RVQ family member — flat-RVQ / FSQ /
/// SoundStream siblings share none of SNAC's per-stage `vq_strides` axes
/// (mirror of the converter's `vokra_convert::models::snac::ARCH` docstring).
pub const ARCH: &str = "snac";

/// `vokra.snac.variant` metadata key: `"24khz"` / `"44khz"`. Consumers
/// dispatch on this without parsing free-text `vokra.model.name`
/// (mirrors `vokra.focalcodec.variant` + `vokra.bigvgan.variant`).
pub const KEY_SNAC_VARIANT: &str = "vokra.snac.variant";

/// Variant tag written for the Hz24 release.
pub const VARIANT_TAG_HZ24: &str = "24khz";

/// Variant tag written for the Hz44 release.
pub const VARIANT_TAG_HZ44: &str = "44khz";

// ---------------------------------------------------------------------------
// SnacVariant — mirror of crates/vokra-convert/src/models/snac.rs::SnacVariant
// ---------------------------------------------------------------------------

/// Which SNAC release the loaded GGUF carries. Selected via the
/// `vokra.snac.variant` chunk written by the converter.
///
/// Mirror of `vokra_convert::models::snac::SnacVariant` — the
/// two enums are kept structurally identical (same order, same
/// `#[derive]`s, same variant docstrings) so a reader that inspects one
/// side has no drift risk on the other. The cross-crate constant
/// duplication rule (see module doc) applies: adding a dependency edge
/// `vokra-models → vokra-convert` would reverse the layer stack.
///
/// # Per-variant config axes
///
/// Primary source: HF `config.json` for each release
/// (`hubertsiuzdak/snac_24khz` / `hubertsiuzdak/snac_44khz`), verified
/// 2026-08-01 in the converter's rustdoc — the axes are transcribed
/// verbatim there.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SnacVariant {
    /// `hubertsiuzdak/snac_24khz`: 24 kHz sample rate, 3 hierarchical
    /// RVQ levels @ ~12/23/47 Hz, no attention (canonical /
    /// higher-download release, primary consumer = Orpheus-TTS + MOSS
    /// voice + CSM-1B-adjacent TTS stacks). `vokra.snac.variant =
    /// "24khz"`.
    Hz24,
    /// `hubertsiuzdak/snac_44khz`: 44.1 kHz sample rate, 4 hierarchical
    /// RVQ levels, `attn_window_size=32` for local attention
    /// (music-quality variant, lower download volume).
    /// `vokra.snac.variant = "44khz"`.
    Hz44,
}

impl SnacVariant {
    /// The `vokra.snac.variant` tag written under [`KEY_SNAC_VARIANT`].
    pub const fn tag(self) -> &'static str {
        match self {
            Self::Hz24 => VARIANT_TAG_HZ24,
            Self::Hz44 => VARIANT_TAG_HZ44,
        }
    }

    /// Parses a `vokra.snac.variant` chunk value into a variant, or
    /// returns `None` for an unrecognized string. Unlike a `TryFrom`
    /// impl this preserves the caller's ability to add a per-key context
    /// prefix to the loud error message (`Snac::from_gguf` does exactly
    /// that below — a `TryFrom` would force a fixed message shape).
    pub fn from_tag(tag: &str) -> Option<Self> {
        match tag {
            VARIANT_TAG_HZ24 => Some(Self::Hz24),
            VARIANT_TAG_HZ44 => Some(Self::Hz44),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// SnacConfig — derived per-variant axes surfaced by from_gguf
// ---------------------------------------------------------------------------

/// Per-variant SNAC config axes surfaced by [`Snac::from_gguf`] so a
/// consumer can pick a specific frame-rate / RVQ-depth head without
/// having to parse the converter's rustdoc table.
///
/// The axes are transcribed verbatim from the upstream HF `config.json`
/// (see [`SnacVariant`] docstring for the source). Kept as a plain
/// `pub` struct — every field is a primitive with a fixed value per
/// variant, so pinning the shape in the type is a stability win.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SnacConfig {
    /// Which upstream SNAC release this config describes.
    pub variant: SnacVariant,
    /// Output PCM sample rate the underlying SNAC model was trained
    /// for. Hz24 = 24_000, Hz44 = 44_100.
    pub sample_rate: u32,
    /// Per-stage temporal strides in the hierarchical RVQ. Every stage
    /// `i` runs at `base_frame_rate / vq_strides[i]`. Hz24 = `[4, 2, 1]`
    /// (3 stages @ ~12/23/47 Hz); Hz44 = `[8, 4, 2, 1]` (4 stages).
    ///
    /// Sized to the deepest supported variant (Hz44 = 4 stages); Hz24
    /// uses the first 3 entries and pads with `0` in slots 3-∞ so a
    /// caller iterating the full slice sees the honest zero for absent
    /// stages (never a fabricated stride).
    pub vq_strides: [u32; 4],
    /// Number of active RVQ stages for this variant (3 for Hz24, 4 for
    /// Hz44). Consumers walk `vq_strides[..n_stages]` to iterate only
    /// the active stages — a stage index `>= n_stages` is invalid.
    pub n_stages: usize,
}

impl SnacConfig {
    /// Builds the config for a given variant from the primary-source
    /// upstream `config.json` axes (see the [`SnacVariant`] docstring
    /// table).
    #[inline]
    #[must_use]
    pub const fn for_variant(variant: SnacVariant) -> Self {
        match variant {
            SnacVariant::Hz24 => Self {
                variant: SnacVariant::Hz24,
                sample_rate: 24_000,
                vq_strides: [4, 2, 1, 0],
                n_stages: 3,
            },
            SnacVariant::Hz44 => Self {
                variant: SnacVariant::Hz44,
                sample_rate: 44_100,
                vq_strides: [8, 4, 2, 1],
                n_stages: 4,
            },
        }
    }

    /// Slice of the active per-stage strides (length =
    /// [`Self::n_stages`]). Consumers iterating the strides SHOULD use
    /// this accessor rather than reading the full `[u32; 4]` — the
    /// trailing `0` slot on Hz24 is deliberately unpopulated and would
    /// divide by zero if fed into the frame-rate formula.
    #[inline]
    #[must_use]
    pub fn active_vq_strides(&self) -> &[u32] {
        &self.vq_strides[..self.n_stages]
    }
}

// ---------------------------------------------------------------------------
// Snac — the runtime binder
// ---------------------------------------------------------------------------

/// A standalone SNAC codec GGUF bound to its variant-driven config and
/// license class. Turns a `vokra.model.arch == "snac"` GGUF into an
/// addressable `Snac` object.
///
/// This binder does **not** run the M2-13 weight-license gate itself —
/// callers loading untrusted GGUFs go through the usual
/// `vokra_core::check_weight_license` path first (both SNAC variants
/// are `Permissive` / MIT per the converter's `DEFAULT_LICENSE_SPDX`).
#[derive(Debug, Clone)]
pub struct Snac {
    config: SnacConfig,
    weight_license: LicenseClass,
}

impl Snac {
    /// Binds a SNAC GGUF: validates arch, reads variant, derives
    /// per-variant config, and surfaces the stamped weight-license
    /// class for compliance gate cross-checks.
    ///
    /// This binder is a *loud* validation step. Every failure is a
    /// distinct [`VokraError::ModelLoad`] naming the missing / wrong
    /// key so a reader diagnosing a mis-produced GGUF has exactly one
    /// place to walk (FR-EX-08 — never a silent partial bind).
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when `vokra.model.arch` is absent or
    ///   not `"snac"` (a DAC / Mimi / WavTokenizer GGUF handed to us by
    ///   mistake fails with a clear message instead of a downstream
    ///   "missing tensor" — same pattern as `Dnsmos::from_gguf`).
    /// - [`VokraError::ModelLoad`] when `vokra.snac.variant` is absent.
    /// - [`VokraError::ModelLoad`] when `vokra.snac.variant` carries an
    ///   unrecognized tag (a rogue converter or a future 3rd variant
    ///   the runtime does not know how to dispatch — refuse loud rather
    ///   than silently defaulting to Hz24).
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        // 1. Arch check — always first so a mis-typed model handed here
        //    fails with a specific message instead of a downstream
        //    "vokra.snac.variant missing".
        match file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()) {
            Some(a) if a == ARCH => {}
            Some(other) => {
                return Err(VokraError::ModelLoad(format!(
                    "snac: GGUF arch is `{other}`, expected `{ARCH}` (was this GGUF \
                     produced by `vokra-cli convert --model snac-{{24khz,44khz}}`?)"
                )));
            }
            None => {
                return Err(VokraError::ModelLoad(
                    "snac: GGUF is missing `vokra.model.arch` (converter did not \
                     stamp it — this is not a Vokra-native SNAC GGUF)"
                        .to_owned(),
                ));
            }
        }

        // 2. Variant discrimination — `vokra.snac.variant` is required
        //    (no silent default: a Hz44 GGUF loaded as Hz24 would
        //    silently corrupt every downstream code-rate calculation).
        let variant_tag = match file.get(KEY_SNAC_VARIANT).and_then(|v| v.as_str()) {
            Some(t) => t,
            None => {
                return Err(VokraError::ModelLoad(format!(
                    "snac: GGUF is missing `{KEY_SNAC_VARIANT}` (converter did \
                     not stamp it — every SNAC GGUF must declare its variant \
                     so the runtime can pick the correct per-variant config \
                     axes; expected `\"{VARIANT_TAG_HZ24}\"` or `\"{VARIANT_TAG_HZ44}\"`)"
                )));
            }
        };
        let variant = SnacVariant::from_tag(variant_tag).ok_or_else(|| {
            VokraError::ModelLoad(format!(
                "snac: `{KEY_SNAC_VARIANT}` = `{variant_tag}` is not a recognized \
                 variant tag (expected `\"{VARIANT_TAG_HZ24}\"` or \
                 `\"{VARIANT_TAG_HZ44}\"`; a rogue converter or a future \
                 3rd variant this runtime does not dispatch on — refusing loud \
                 rather than silently defaulting to Hz24, FR-EX-08)"
            ))
        })?;

        let config = SnacConfig::for_variant(variant);

        // 3. Provenance surfacing — read the stamped weight-license class
        //    for compliance gate cross-checks (defaults to `Unknown` if
        //    absent, which is fail-closed at the gate). Not raising a
        //    `ModelLoad` on missing provenance keeps the binder able to
        //    load hand-assembled GGUFs the test harness uses without
        //    forcing every fixture to stamp the full provenance chunk.
        let weight_license = file
            .get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
            .and_then(|v| v.as_str())
            .and_then(LicenseClass::from_class_str)
            .unwrap_or(LicenseClass::Unknown);

        Ok(Self {
            config,
            weight_license,
        })
    }

    /// Read-only view of the bound [`SnacConfig`]. Consumers pick the
    /// sample rate / stride axes / active stage count from here.
    #[inline]
    #[must_use]
    pub const fn config(&self) -> &SnacConfig {
        &self.config
    }

    /// The variant this binder loaded (equivalent to
    /// `self.config().variant` but exposed directly so a consumer
    /// dispatching on the variant does not have to reach through the
    /// config).
    #[inline]
    #[must_use]
    pub const fn variant(&self) -> SnacVariant {
        self.config.variant
    }

    /// The stamped weight-license class surfaced from the GGUF's
    /// `vokra.provenance.weight_license` chunk. Both upstream SNAC
    /// variants are `Permissive` (MIT); a GGUF missing the stamp reads
    /// back as `Unknown` (fail-closed at the compliance gate).
    #[inline]
    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    /// Encodes a PCM waveform into hierarchical SNAC codes.
    ///
    /// Return shape: one code vector per hierarchical RVQ stage
    /// (`self.config().n_stages` vectors) — the outer length matches
    /// `n_stages`, each inner length is `T / vq_strides[stage]` where
    /// `T` is the shared base frame count.
    ///
    /// # Loud-partial (this WP)
    ///
    /// Returns [`VokraError::UnsupportedOp`] — the SNAC encoder path
    /// (encoder Conv1D stack + `VectorQuantize.forward`) is a
    /// follow-up WP sized similarly to M4-04 / M4-05 codec waves. See
    /// [`encoder_forward_loud_partial`] for the specific missing
    /// primitives and the upstream source anchors.
    pub fn encode(&self, pcm: &[f32], sample_rate: u32) -> Result<Vec<Vec<u32>>> {
        // Bind unused args so a `#[warn(unused_variables)]` change does
        // not silently mask the loud-partial fire path; the future real
        // implementation will consume both.
        let _ = pcm;
        if sample_rate != self.config.sample_rate {
            return Err(VokraError::InvalidArgument(format!(
                "snac encode: input sample_rate {sample_rate} != model \
                 sample_rate {} for variant {:?}. Resample the PCM before \
                 calling encode (FR-EX-08 — never a silent resample)",
                self.config.sample_rate, self.config.variant
            )));
        }
        Err(encoder_forward_loud_partial(self.config.variant))
    }

    /// Decodes hierarchical SNAC codes to a PCM waveform.
    ///
    /// Input shape: one code vector per hierarchical RVQ stage
    /// (outer length must equal `self.config().n_stages`); inner length
    /// per stage is caller-supplied and must satisfy the SNAC
    /// stage-length co-alignment invariant
    /// (`codes[i].len() * vq_strides[i]` equal across every stage).
    ///
    /// # Loud-partial (this WP)
    ///
    /// Returns [`VokraError::UnsupportedOp`] — the SNAC decoder feature
    /// → PCM synthesis chain (decoder Conv1D upsample + Snake + noise
    /// + Hz44 local attention) is a follow-up WP. The intermediate
    ///   **codes → `[T, d_model]` features** step is *already* served by
    ///   [`vokra_ops::SnacDecoder::decode`], but this binder cannot wire
    ///   that primitive today because the derived per-quantizer tensors
    ///   have not been extracted into the GGUF yet — see
    ///   [`Snac::decode_codes_to_features`] for the specific gap. Errors
    ///   from [`decoder_pcm_forward_loud_partial`] name the exact
    ///   upstream source anchors so the follow-up wave has a single
    ///   place to walk.
    pub fn decode(&self, codes: &[Vec<u32>]) -> Result<Vec<f32>> {
        // Validate the outer shape up front so a caller passing the
        // wrong number of stages sees an `InvalidArgument` (not the
        // loud-partial gate — that would misdirect the fix).
        if codes.len() != self.config.n_stages {
            return Err(VokraError::InvalidArgument(format!(
                "snac decode: got {} code stages, expected {} for variant \
                 {:?} (FR-EX-08 — never a silent shape truncation)",
                codes.len(),
                self.config.n_stages,
                self.config.variant
            )));
        }
        Err(decoder_pcm_forward_loud_partial(self.config.variant))
    }

    /// Decodes hierarchical SNAC codes to intermediate `[T, d_model]`
    /// features (the output of `ResidualVectorQuantize.from_codes`,
    /// *before* the terminal PCM decoder).
    ///
    /// The forward math is fully served by
    /// [`vokra_ops::SnacDecoder::decode`], but this shim requires the
    /// GGUF to carry derived per-quantizer tensors
    /// (`vokra.snac.codebook_tables` + `vokra.snac.quantizer.{i}.out_proj_{weight,bias}`)
    /// that the current converter does **not** emit — the upstream
    /// `snac.SNAC` state-dict is passed through verbatim under
    /// weight-norm parametrization (`weight_g` + `weight_v`), and
    /// factorization + weight-norm fold is a converter-side offline
    /// step that has not landed yet.
    ///
    /// # Loud-partial (this WP)
    ///
    /// Returns [`VokraError::ModelLoad`] naming the specific derived
    /// tensors the converter needs to emit. This shim exists as a seam
    /// so consumers can differentiate the *intermediate* primitive gap
    /// (converter extension) from the *terminal* PCM decoder gap
    /// (encoder / decoder Conv1D + Snake — separate follow-up WP).
    pub fn decode_codes_to_features(&self, codes: &[Vec<u32>]) -> Result<Vec<f32>> {
        // Same outer-shape validation as `decode` so a caller passing
        // the wrong number of stages sees the shape error, not the
        // deferred converter-side gap.
        if codes.len() != self.config.n_stages {
            return Err(VokraError::InvalidArgument(format!(
                "snac decode_codes_to_features: got {} code stages, \
                 expected {} for variant {:?} (FR-EX-08 — never a silent \
                 shape truncation)",
                codes.len(),
                self.config.n_stages,
                self.config.variant
            )));
        }
        Err(codes_to_features_loud_partial(self.config.variant))
    }
}

// ---------------------------------------------------------------------------
// Loud-partial constructors — one per surface point, so an owner (or a
// follow-up CC wave) reading the error message knows exactly where to flip
// the switch. Every message cites the primary upstream source so no
// searching is required.
// ---------------------------------------------------------------------------

/// Constructs the loud-partial [`VokraError::UnsupportedOp`] returned by
/// [`Snac::encode`] until the encoder body lands.
///
/// Names the specific missing primitives from `vokra-ops` — the encoder
/// Conv1D stack, the `VectorQuantize.forward` op (per-stage `avg_pool1d`
/// + nearest-neighbour argmin), the noise-conditioned residual, and the
///   Hz44-only sliding-window local attention. RMVPE / DNSMOS
///   loud-partial-message precedent — one place to walk when the switch
///   gets flipped (CLAUDE.md 教訓 (a)).
fn encoder_forward_loud_partial(variant: SnacVariant) -> VokraError {
    let downsample_rates = match variant {
        SnacVariant::Hz24 => "[2, 4, 8, 8]",
        SnacVariant::Hz44 => "[2, 3, 8, 8]",
    };
    let downsample_factor = match variant {
        SnacVariant::Hz24 => "512x",
        SnacVariant::Hz44 => "384x",
    };
    let attn_note = match variant {
        SnacVariant::Hz24 => "no local attention (Hz24 has attn_window_size=null)",
        SnacVariant::Hz44 => "sliding-window local attention (attn_window_size=32)",
    };
    VokraError::UnsupportedOp(format!(
        "snac ({variant:?}) encode: SNAC encoder Conv1D stack + \
         `VectorQuantize.forward` are follow-up WPs — none of the required \
         primitives are in `vokra-ops` today. Missing: (a) encoder Conv1D \
         chain (`encoder_rates={downsample_rates}`, {downsample_factor} \
         downsampling — upstream `hubertsiuzdak/snac/blob/main/snac/snac.py` \
         `class Encoder`); (b) `VectorQuantize.forward` (per-stage \
         `avg_pool1d(stride)` on the encoder features + nearest-neighbour \
         argmin over the factorized codebook rows — upstream \
         `hubertsiuzdak/snac/blob/main/snac/vq.py` `VectorQuantize.forward` \
         L27-40); (c) noise-conditioned residual (`noise=true` for both \
         variants — upstream `snac/layers.py`); (d) {attn_note}. Loud pending \
         (CLAUDE.md 教訓 (a) — 'loud-partial は fake-complete より honest') — \
         no silent fabricated codes ever emitted (FR-EX-08)."
    ))
}

/// Constructs the loud-partial [`VokraError::UnsupportedOp`] returned by
/// [`Snac::decode`] until the terminal PCM decoder body lands.
///
/// Distinct from the intermediate-features gap in
/// [`codes_to_features_loud_partial`]: this covers the *decoder
/// Conv1D upsample chain*, which turns the RVQ-decoded features into
/// the PCM output — the missing part that even a fully-derived-tensor
/// GGUF could not run today.
fn decoder_pcm_forward_loud_partial(variant: SnacVariant) -> VokraError {
    let upsample_rates = match variant {
        SnacVariant::Hz24 => "[8, 8, 4, 2]",
        SnacVariant::Hz44 => "[8, 8, 3, 2]",
    };
    let attn_note = match variant {
        SnacVariant::Hz24 => "no local attention (Hz24 has attn_window_size=null)",
        SnacVariant::Hz44 => "sliding-window local attention (attn_window_size=32)",
    };
    VokraError::UnsupportedOp(format!(
        "snac ({variant:?}) decode: SNAC decoder feature→PCM synthesis \
         chain is a follow-up WP. The existing `vokra_ops::SnacDecoder::decode` \
         primitive only covers the intermediate RVQ codes → `[T, d_model]` \
         features step (upstream `snac/vq.py` `ResidualVectorQuantize.from_codes` \
         L61-71), NOT the terminal PCM synthesis. Missing: (a) decoder Conv1D \
         upsample stack (`decoder_rates={upsample_rates}` — upstream \
         `hubertsiuzdak/snac/blob/main/snac/snac.py` `class Decoder`); \
         (b) Snake activation on every decoder block (upstream \
         `snac/layers.py`); (c) noise module (`noise=true` for both variants); \
         (d) {attn_note}. See `Snac::decode_codes_to_features` for the \
         intermediate-features gap (converter-side, separate follow-up). \
         Loud pending — no silent fabricated PCM ever emitted (FR-EX-08)."
    ))
}

/// Constructs the loud-partial [`VokraError::ModelLoad`] returned by
/// [`Snac::decode_codes_to_features`] until the converter emits the
/// derived per-quantizer tensors.
///
/// Named as `ModelLoad` (not `UnsupportedOp`) because the primitive
/// itself (`vokra_ops::SnacDecoder::decode`) *exists* — the block is a
/// converter-side missing tensor extraction, not a missing op. This
/// distinction lets a follow-up wave know to touch
/// `crates/vokra-convert/src/models/snac.rs` rather than adding a new
/// op to `vokra-ops`.
fn codes_to_features_loud_partial(variant: SnacVariant) -> VokraError {
    VokraError::ModelLoad(format!(
        "snac ({variant:?}) decode_codes_to_features: the intermediate \
         RVQ codes → `[T, d_model]` features primitive \
         (`vokra_ops::SnacDecoder::decode`) exists, but the current \
         `vokra-convert::models::snac::convert_snac_file` passes the \
         upstream `snac.SNAC` state-dict tensors through verbatim under \
         their weight-norm parametrization (`weight_g` + `weight_v`) — \
         it does not yet emit the derived per-quantizer tensors this \
         binder needs (`vokra.snac.codebook_tables` + \
         `vokra.snac.quantizer.{{i}}.out_proj_weight` + \
         `vokra.snac.quantizer.{{i}}.out_proj_bias`, mirror of the M4-04 \
         DAC converter T10/T11 derived-tensor pattern). Extend the \
         converter with offline weight-norm folding + factorization \
         extraction, then this seam flips to a real bind (FR-EX-08 — \
         no silent zero-features until then)."
    ))
}
