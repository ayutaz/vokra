//! Native **SNAC** (`hubertsiuzdak/snac_{24khz,44khz}`, MIT) codec runtime.
//!
//! Both public GGUFs are bound against an exact tensor manifest: 269 tensors
//! for 24 kHz and 286 for 44.1 kHz. CPU executes the full waveform encoder,
//! hierarchical factorized RVQ, stochastic decoder, and the 44.1 kHz local
//! attention path. Metal executes the complete token-to-waveform route; an
//! encode request on Metal returns an explicit unsupported-operation error
//! because nearest-codebook search has no GPU kernel. No backend silently
//! falls back to CPU.
//!
//! Weight-normalized upstream tensors remain under PyTorch's
//! `parametrizations.weight.original{0,1}` names in the public artifacts and
//! are folded exactly once at bind time. No derived tensors or republished
//! checkpoints are required.
mod runtime;

pub use runtime::{SNAC_HOT_OPS, Snac};

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
/// (mirror of the `ARCH` docstring in
/// `crates/vokra-convert/src/models/snac.rs`).
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
/// Mirror of `SnacVariant` in
/// `crates/vokra-convert/src/models/snac.rs` — the
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
