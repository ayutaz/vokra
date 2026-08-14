//! **Moonshine** (`UsefulSensors/moonshine-{tiny,base}`, MIT) —
//! Useful Sensors' edge-optimized real-time ASR family (Jeffries et al.
//! 2024, arXiv:2410.15608 "Moonshine: Speech Recognition for Live
//! Transcription and Voice Commands") — runtime binder for the
//! `moonshine` converter arch.
//!
//! # What Moonshine is (primary source)
//!
//! Moonshine is a transformer encoder-decoder ASR family from Useful
//! Sensors, ~5× faster than Whisper-tiny at 16 kHz per the paper.
//! **Distinct from sibling Whisper** in two significant ways:
//!
//! 1. **No mel front-end** — the model consumes raw 16 kHz PCM directly
//!    via a learned strided **Conv1D stem** (bypassing STFT + Mel
//!    filterbank). This is the *defining* trait that separates
//!    Moonshine from every Whisper-family sibling in this crate
//!    (`whisper` / `distil_whisper` / `kotoba_whisper`) and forces a
//!    distinct runtime arch tag — a Moonshine checkpoint fed to a
//!    Whisper loader would misroute at the audio-input boundary
//!    (raw-audio Conv1D vs mel filterbank), which FR-EX-08 (no silent
//!    op-shape misroute) forbids.
//! 2. **Rotary position encoding + SwiGLU** activations rather than
//!    Whisper's sinusoidal position embeddings + GELU.
//!
//! The family ships two sibling sizes — see [`MoonshineVariant`].
//!
//! # Runtime layout (loud-partial, RMVPE + DNSMOS + snac precedent)
//!
//! ```text
//! raw waveform (mono f32, [T] 16 kHz PCM)
//!   -> Conv1D stem                                   ← **loud-partial**
//!        (strides [64, 3, 2] = 384x downsampling per
//!         primary source `moonshine/model.py` —
//!         NO mel front-end; distinguishing Moonshine trait)
//!   -> Transformer encoder                            ← **loud-partial**
//!        (RoPE positional encoding, SwiGLU activations,
//!         `n_encoder_layers` × `encoder_num_heads`;
//!         distinct from Whisper's sinusoidal + GELU)
//!   -> Transformer decoder                            ← **loud-partial**
//!        (RoPE positional encoding, self-attention + cross-attention
//!         to encoder outputs, `n_decoder_layers` ×
//!         `decoder_num_heads`)
//!   -> greedy / beam token search + SentencePiece detokenize ← **loud-partial**
//!   -> transcribed text (String)
//! ```
//!
//! # Loud-partial classification (design § — CLAUDE.md 教訓 (a))
//!
//! - **Real (this WP)**: [`Moonshine::from_gguf`] with strict
//!   `vokra.model.arch == "moonshine"` validation + variant discrimination
//!   via `vokra.model.name` (`"moonshine-tiny"` / `"moonshine-base"`) +
//!   per-variant [`MoonshineConfig`] surfacing; [`MoonshineVariant`]
//!   round-trip; sibling-arch hinting on wrong-arch loud errors.
//! - **Loud-partial (this WP)**: [`Moonshine::transcribe`] returns
//!   [`VokraError::UnsupportedOp`] naming the three exact missing pieces:
//!   (i) raw-audio Conv1D stem walk (strides [64, 3, 2] — NO mel
//!       front-end, distinguishing Moonshine trait vs Whisper family),
//!   (ii) RoPE + SwiGLU transformer encoder-decoder forward,
//!   (iii) greedy / beam decoding + SentencePiece detokenize.
//!   The message cites all three primary source URLs
//!   (github.com/usefulsensors/moonshine, arXiv:2410.15608,
//!   huggingface.co/UsefulSensors/moonshine-{tiny,base}) so a reader
//!   diagnosing this gap has exactly three anchors to walk.
//!
//! Rationale (RMVPE / snac / wavlm / llama_omni2 Wave 1-7 precedent,
//! CLAUDE.md 教訓 (a)): the surrounding scaffold + `from_gguf`
//! validation + FR-EX-08 loud-fails land today so a follow-up wave can
//! flip the switch by (i) implementing the Conv1D stem primitive
//! against upstream `moonshine/model.py`, (ii) implementing the RoPE +
//! SwiGLU transformer encoder-decoder forward, and (iii) implementing
//! greedy decode + SentencePiece detokenize. The
//! [`VokraError::UnsupportedOp`] message cites all three anchors so no
//! searching is required.
//!
//! # Per-variant hparams (primary source: `moonshine/model.py`)
//!
//! The per-variant axes (`hidden_size` / `n_*_layers` / `*_num_heads` /
//! `ffn_multiplier` / `encoder_conv_strides` / `vocab_size`) are
//! transcribed from `github.com/usefulsensors/moonshine`
//! `moonshine/model.py` per the audit ticket. Owner should verify at
//! bind time against the exact commit shipping in a given checkpoint
//! — the primary source URL lives on [`MoonshineConfig::tiny`] /
//! [`MoonshineConfig::base`]. See CLAUDE.md 「ハルシネーション厳禁」:
//! the per-variant constants can drift as upstream releases new
//! variants, and owner primary-source verification is the safety net.
//!
//! # `vokra.moonshine.*` chunk group (read here)
//!
//! Written by `vokra-convert::models::moonshine_{tiny,base}`:
//!
//! - `vokra.model.arch` (`String`): must equal [`ARCH`] (`"moonshine"`).
//! - `vokra.model.name` (`String`): `"moonshine-tiny"` /
//!   `"moonshine-base"` — the variant discriminator (mirrors the
//!   converter's `NAME` constant).
//! - `vokra.model.category` (`String`): `"asr"` — shared with the
//!   Whisper family; kept auxiliary rather than a hard gate.
//! - `vokra.provenance.*`: license class + raw license string, so the
//!   runtime compliance gate (FR-CP-03) can classify the artifact
//!   without re-inspecting the safetensors provenance. Both Moonshine
//!   variants are `Permissive` / MIT per the converter's
//!   `DEFAULT_LICENSE_SPDX`.
//!
//! # Cross-crate constant duplication
//!
//! Mirror of the converter's [`ARCH`] constant — same rule the sibling
//! BF16 pass-through binders (`snac` / `wavlm` / `fsmn_vad` /
//! `openwakeword` / `dnsmos_p808_p835`) use so `vokra-models` does not
//! gain a dependency edge onto `vokra-convert`, preserving the layered
//! convention `vokra-ops → nothing GGUF-aware`, `vokra-core → GGUF
//! reader`, `vokra-models → GGUF binder`, `vokra-convert → GGUF
//! writer`.

use vokra_core::gguf::{GgufFile, chunks};
use vokra_core::{LicenseClass, Result, VokraError};

// ---------------------------------------------------------------------------
// Arch / variant / metadata-key constants — mirror of
// crates/vokra-convert/src/models/moonshine_{tiny,base}.rs (see the
// module docstring for the cross-crate duplication rationale).
// ---------------------------------------------------------------------------

/// Expected `vokra.model.arch` value written by
/// `vokra-cli convert --model moonshine-{tiny,base}`.
///
/// Shared across Tiny and Base (they differ only in depth / width — same
/// arch family). Distinct from every sibling ASR arch (`whisper` /
/// `distil_whisper` / `kotoba_whisper` / `parakeet` / `parakeet_ctc` /
/// `canary` / `canary_qwen` / `omniasr_ctc` / `kyutai_stt`) — silently
/// sharing arch would misroute runtime dispatch at the audio-input
/// boundary (raw-audio Conv1D vs mel filterbank), which FR-EX-08
/// (no silent op-shape misroute) forbids.
pub const ARCH: &str = "moonshine";

/// Raw PCM sample rate the Moonshine Conv1D stem consumes at the
/// front-end (per upstream release manifest). Distinct from
/// Whisper-family siblings (which all also key on 16 kHz) in that
/// Moonshine consumes the raw waveform directly with **no mel
/// filterbank** in between.
pub const MOONSHINE_SAMPLE_RATE: u32 = 16_000;

/// `vokra.model.name` tag written for the Tiny release (~27M params,
/// ~110 MB).
pub const NAME_TAG_TINY: &str = "moonshine-tiny";

/// `vokra.model.name` tag written for the Base release (~61.5M params,
/// ~250 MB).
pub const NAME_TAG_BASE: &str = "moonshine-base";

// ---------------------------------------------------------------------------
// MoonshineVariant — mirror of the converter's per-variant NAME constants
// ---------------------------------------------------------------------------

/// Which Moonshine release the loaded GGUF carries. Selected via the
/// `vokra.model.name` chunk written by the sibling converter modules
/// (`moonshine_tiny.rs` writes `"moonshine-tiny"` /
/// `moonshine_base.rs` writes `"moonshine-base"`).
///
/// Kept parallel to the converter-side sibling
/// [`crate::ModelKind::MoonshineTiny`] / [`crate::ModelKind::MoonshineBase`]
/// arms — the two crates only share `vokra-core`, so the wire tag
/// string is the handshake (mirror of the sibling `snac::SnacVariant`
/// pattern).
///
/// # Per-variant hparams (primary source)
///
/// The Tiny and Base variants share the Moonshine arch family (raw-
/// audio Conv1D stem + RoPE + SwiGLU transformer encoder-decoder) but
/// differ in depth and width — see [`MoonshineConfig::tiny`] /
/// [`MoonshineConfig::base`] for the transcribed hparams and the
/// primary source URL.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoonshineVariant {
    /// `UsefulSensors/moonshine-tiny`: 27M parameters, ~110 MB. Local
    /// convert-safe on M1 iMac (well below the ≥8 GB vast.ai cutoff per
    /// memory `[[feedback-large-models-on-vast-ai]]`).
    /// `vokra.model.name = "moonshine-tiny"`.
    Tiny,
    /// `UsefulSensors/moonshine-base`: 61.5M parameters, ~250 MB. Local
    /// convert-safe on M1 iMac. `vokra.model.name = "moonshine-base"`.
    Base,
}

impl MoonshineVariant {
    /// The `vokra.model.name` tag written for this variant. Kept as the
    /// wire-format handshake with the converter (`moonshine_tiny::NAME`
    /// / `moonshine_base::NAME`).
    #[inline]
    #[must_use]
    pub const fn as_name(self) -> &'static str {
        match self {
            Self::Tiny => NAME_TAG_TINY,
            Self::Base => NAME_TAG_BASE,
        }
    }

    /// Parses a `vokra.model.name` chunk value into a variant, or
    /// returns `None` for an unrecognized string. Unlike a `TryFrom`
    /// impl this preserves the caller's ability to add a per-key
    /// context prefix to the loud error message
    /// ([`Moonshine::from_gguf`] does exactly that below — a `TryFrom`
    /// would force a fixed message shape).
    #[inline]
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            NAME_TAG_TINY => Some(Self::Tiny),
            NAME_TAG_BASE => Some(Self::Base),
            _ => None,
        }
    }

    /// Canonical HF repo id for this variant (kept in sync with the
    /// converter-side `UPSTREAM_HF` constants).
    #[inline]
    #[must_use]
    pub const fn upstream_hf(self) -> &'static str {
        match self {
            Self::Tiny => "UsefulSensors/moonshine-tiny",
            Self::Base => "UsefulSensors/moonshine-base",
        }
    }
}

// ---------------------------------------------------------------------------
// MoonshineConfig — per-variant hparams surfaced by from_gguf
// ---------------------------------------------------------------------------

/// Per-variant Moonshine hparams surfaced by [`Moonshine::from_gguf`]
/// so a consumer can pick a specific encoder / decoder head configuration
/// without having to parse the converter's rustdoc table.
///
/// The axes are transcribed from `github.com/usefulsensors/moonshine`
/// `moonshine/model.py` per the audit ticket — see [`Self::tiny`] /
/// [`Self::base`] for the primary source URL. Owner should verify at
/// bind time against the exact commit shipping in a given checkpoint
/// (CLAUDE.md 「ハルシネーション厳禁」).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MoonshineConfig {
    /// Which upstream Moonshine release this config describes.
    pub variant: MoonshineVariant,
    /// Transformer residual width (`dim` in the upstream `model.py`).
    /// Tiny = 288, Base = 416.
    pub hidden_size: u32,
    /// Number of Transformer encoder layers. Tiny = 6, Base = 8.
    pub n_encoder_layers: u32,
    /// Number of Transformer decoder layers. Tiny = 6, Base = 8.
    pub n_decoder_layers: u32,
    /// Number of encoder self-attention heads. Tiny = 6, Base = 8.
    pub encoder_num_heads: u32,
    /// Number of decoder self- + cross-attention heads. Tiny = 6,
    /// Base = 8.
    pub decoder_num_heads: u32,
    /// SwiGLU FFN inner-width multiplier (`ff_mult` in the upstream
    /// `model.py`). Both variants = 4.
    pub ffn_multiplier: u32,
    /// Per-layer strides in the raw-audio Conv1D stem
    /// (`conv_strides` in the upstream `model.py`). Both variants =
    /// `[64, 3, 2]` (product = 384x downsampling). Stored as a fixed
    /// 3-slot array because the upstream stem has exactly three
    /// Conv1D layers — a stride axis diverging from 3 slots would
    /// indicate a topology change requiring a variant enum extension.
    pub encoder_conv_strides: [u32; 3],
    /// SentencePiece vocabulary size shared across encoder and
    /// decoder token spaces. Both variants = 32768.
    pub vocab_size: u32,
    /// Raw PCM sample rate the Conv1D stem consumes at the front-end
    /// (16 kHz per upstream release manifest — shared with every
    /// Whisper-family sibling ASR).
    pub sample_rate: u32,
}

impl MoonshineConfig {
    /// Builds the config for the Tiny variant.
    ///
    /// Hparams transcribed from `github.com/usefulsensors/moonshine`
    /// `moonshine/model.py` (`Moonshine.tiny` factory) per the audit
    /// ticket. Owner should verify at bind time against the exact
    /// commit shipping in a given checkpoint.
    #[inline]
    #[must_use]
    pub const fn tiny() -> Self {
        Self {
            variant: MoonshineVariant::Tiny,
            hidden_size: 288,
            n_encoder_layers: 6,
            n_decoder_layers: 6,
            encoder_num_heads: 6,
            decoder_num_heads: 6,
            ffn_multiplier: 4,
            encoder_conv_strides: [64, 3, 2],
            vocab_size: 32_768,
            sample_rate: MOONSHINE_SAMPLE_RATE,
        }
    }

    /// Builds the config for the Base variant.
    ///
    /// Hparams transcribed from `github.com/usefulsensors/moonshine`
    /// `moonshine/model.py` (`Moonshine.base` factory) per the audit
    /// ticket. Owner should verify at bind time against the exact
    /// commit shipping in a given checkpoint.
    #[inline]
    #[must_use]
    pub const fn base() -> Self {
        Self {
            variant: MoonshineVariant::Base,
            hidden_size: 416,
            n_encoder_layers: 8,
            n_decoder_layers: 8,
            encoder_num_heads: 8,
            decoder_num_heads: 8,
            ffn_multiplier: 4,
            encoder_conv_strides: [64, 3, 2],
            vocab_size: 32_768,
            sample_rate: MOONSHINE_SAMPLE_RATE,
        }
    }

    /// Dispatches to [`Self::tiny`] / [`Self::base`] based on `variant`.
    #[inline]
    #[must_use]
    pub const fn for_variant(variant: MoonshineVariant) -> Self {
        match variant {
            MoonshineVariant::Tiny => Self::tiny(),
            MoonshineVariant::Base => Self::base(),
        }
    }

    /// Well-formedness gate: all counts non-zero and per-head width
    /// divides evenly. Fires **before** any forward runs so a shape-
    /// corrupt fixture fails loudly here rather than deep inside a
    /// GEMM (FR-EX-08).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] naming the offending field.
    pub fn validate(&self) -> Result<()> {
        if self.hidden_size == 0 {
            return Err(VokraError::InvalidArgument(
                "moonshine config: hidden_size must be > 0".to_owned(),
            ));
        }
        if self.n_encoder_layers == 0 {
            return Err(VokraError::InvalidArgument(
                "moonshine config: n_encoder_layers must be > 0".to_owned(),
            ));
        }
        if self.n_decoder_layers == 0 {
            return Err(VokraError::InvalidArgument(
                "moonshine config: n_decoder_layers must be > 0".to_owned(),
            ));
        }
        if self.encoder_num_heads == 0 {
            return Err(VokraError::InvalidArgument(
                "moonshine config: encoder_num_heads must be > 0".to_owned(),
            ));
        }
        if self.decoder_num_heads == 0 {
            return Err(VokraError::InvalidArgument(
                "moonshine config: decoder_num_heads must be > 0".to_owned(),
            ));
        }
        if self.hidden_size % self.encoder_num_heads != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "moonshine config: hidden_size {} must be divisible by \
                 encoder_num_heads {}",
                self.hidden_size, self.encoder_num_heads
            )));
        }
        if self.hidden_size % self.decoder_num_heads != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "moonshine config: hidden_size {} must be divisible by \
                 decoder_num_heads {}",
                self.hidden_size, self.decoder_num_heads
            )));
        }
        if self.ffn_multiplier == 0 {
            return Err(VokraError::InvalidArgument(
                "moonshine config: ffn_multiplier must be > 0 (SwiGLU FFN)".to_owned(),
            ));
        }
        if self.vocab_size == 0 {
            return Err(VokraError::InvalidArgument(
                "moonshine config: vocab_size must be > 0 (SentencePiece vocab)".to_owned(),
            ));
        }
        if self.sample_rate == 0 {
            return Err(VokraError::InvalidArgument(
                "moonshine config: sample_rate must be > 0".to_owned(),
            ));
        }
        for (i, &s) in self.encoder_conv_strides.iter().enumerate() {
            if s == 0 {
                return Err(VokraError::InvalidArgument(format!(
                    "moonshine config: encoder_conv_strides[{i}] = 0 (would \
                     divide the frame rate by zero in the Conv1D stem)"
                )));
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Moonshine — the runtime binder
// ---------------------------------------------------------------------------

/// A standalone Moonshine ASR GGUF bound to its variant-driven config
/// and license class. Turns a `vokra.model.arch == "moonshine"` GGUF
/// into an addressable `Moonshine` object.
///
/// This binder does **not** run the M2-13 weight-license gate itself —
/// callers loading untrusted GGUFs go through the usual
/// `vokra_core::check_weight_license` path first (both Moonshine
/// variants are `Permissive` / MIT per the converter's
/// `DEFAULT_LICENSE_SPDX`).
#[derive(Debug, Clone)]
pub struct Moonshine {
    config: MoonshineConfig,
    weight_license: LicenseClass,
}

impl Moonshine {
    /// Binds a Moonshine GGUF: validates arch, reads variant, derives
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
    ///   not `"moonshine"` (a Whisper / distil-Whisper / kotoba-Whisper
    ///   / Parakeet / Canary GGUF handed here by mistake fails with a
    ///   clear sibling-arch hint instead of a downstream "missing
    ///   tensor" — same pattern as `Snac::from_gguf`).
    /// - [`VokraError::ModelLoad`] when `vokra.model.name` is absent.
    /// - [`VokraError::ModelLoad`] when `vokra.model.name` carries an
    ///   unrecognized tag (a rogue converter or a future 3rd variant
    ///   the runtime does not know how to dispatch — refuse loud
    ///   rather than silently defaulting to Tiny).
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        // 1. Arch check — always first so a mis-typed model handed here
        //    fails with a specific message + sibling ASR-family hint,
        //    not a downstream "vokra.model.name missing".
        match file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()) {
            Some(a) if a == ARCH => {}
            Some(other) => {
                return Err(VokraError::ModelLoad(format!(
                    "moonshine: GGUF arch is `{other}`, expected `{ARCH}` \
                     (was this GGUF produced by `vokra-cli convert --model \
                     moonshine-{{tiny,base}}`? Sibling ASR arches this runtime \
                     dispatches on: `whisper` / `distil_whisper` / \
                     `kotoba_whisper` / `parakeet` / `parakeet_ctc` / \
                     `canary` / `canary_qwen` / `omniasr_ctc` / `kyutai_stt` — \
                     each has its own from_gguf. Primary source: \
                     https://github.com/usefulsensors/moonshine)"
                )));
            }
            None => {
                return Err(VokraError::ModelLoad(
                    "moonshine: GGUF is missing `vokra.model.arch` \
                     (converter did not stamp it — this is not a \
                     Vokra-native Moonshine GGUF). Primary source: \
                     https://github.com/usefulsensors/moonshine"
                        .to_owned(),
                ));
            }
        }

        // 2. Variant discrimination — `vokra.model.name` is required
        //    (no silent default: a Base GGUF loaded as Tiny would
        //    silently corrupt every downstream shape check).
        let name_tag = match file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()) {
            Some(t) => t,
            None => {
                return Err(VokraError::ModelLoad(format!(
                    "moonshine: GGUF is missing `vokra.model.name` \
                     (converter did not stamp it — every Moonshine GGUF must \
                     declare its variant so the runtime can pick the correct \
                     per-variant config axes; expected `\"{NAME_TAG_TINY}\"` \
                     or `\"{NAME_TAG_BASE}\"`). Primary source: \
                     https://github.com/usefulsensors/moonshine"
                )));
            }
        };
        let variant = MoonshineVariant::from_name(name_tag).ok_or_else(|| {
            VokraError::ModelLoad(format!(
                "moonshine: `vokra.model.name` = `{name_tag}` is not a \
                 recognized variant tag (expected `\"{NAME_TAG_TINY}\"` or \
                 `\"{NAME_TAG_BASE}\"`; a rogue converter or a future 3rd \
                 variant this runtime does not dispatch on — refusing loud \
                 rather than silently defaulting to Tiny, FR-EX-08). \
                 Primary source: https://github.com/usefulsensors/moonshine"
            ))
        })?;

        let config = MoonshineConfig::for_variant(variant);

        // 3. Provenance surfacing — read the stamped weight-license
        //    class for compliance gate cross-checks (defaults to
        //    `Unknown` if absent, which is fail-closed at the gate).
        //    Not raising a `ModelLoad` on missing provenance keeps the
        //    binder able to load hand-assembled GGUFs the test harness
        //    uses without forcing every fixture to stamp the full
        //    provenance chunk.
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

    /// Convenience wrapper for [`Self::from_gguf`] that opens the GGUF
    /// from a filesystem path first.
    ///
    /// # Errors
    ///
    /// Any error surfaced by `GgufFile::open` (IO / parse), or any
    /// error surfaced by [`Self::from_gguf`].
    pub fn from_path(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let file = GgufFile::open(path)?;
        Self::from_gguf(&file)
    }

    /// Read-only view of the bound [`MoonshineConfig`]. Consumers pick
    /// the hidden size / layer counts / attention heads / conv strides
    /// / vocab / sample rate from here.
    #[inline]
    #[must_use]
    pub const fn config(&self) -> &MoonshineConfig {
        &self.config
    }

    /// The variant this binder loaded (equivalent to
    /// `self.config().variant` but exposed directly so a consumer
    /// dispatching on the variant does not have to reach through the
    /// config).
    #[inline]
    #[must_use]
    pub const fn variant(&self) -> MoonshineVariant {
        self.config.variant
    }

    /// The stamped weight-license class surfaced from the GGUF's
    /// `vokra.provenance.weight_license` chunk. Both upstream
    /// Moonshine variants are `Permissive` (MIT); a GGUF missing the
    /// stamp reads back as `Unknown` (fail-closed at the compliance
    /// gate).
    #[inline]
    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    /// Transcribes raw 16 kHz PCM into text.
    ///
    /// Input: mono `f32` PCM at [`MOONSHINE_SAMPLE_RATE`] (16 kHz).
    ///
    /// # Loud-partial (this WP)
    ///
    /// Returns [`VokraError::UnsupportedOp`] — the Moonshine forward
    /// path is a follow-up WP sized similarly to sibling ASR waves.
    /// The message names the three exact missing pieces so a follow-up
    /// wave can flip the switch without cross-referencing rustdoc:
    /// (i) raw-audio Conv1D stem walk (strides = `[64, 3, 2]`, NO mel
    ///     front-end — this is what makes Moonshine *distinct* from
    ///     every Whisper-family sibling),
    /// (ii) RoPE + SwiGLU transformer encoder-decoder forward,
    /// (iii) greedy / beam decoding + SentencePiece detokenize.
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] when `pcm` is empty (an empty
    ///   input cannot produce a transcript; caller passed the wrong
    ///   buffer). Fires **before** the loud-partial gate so the caller
    ///   sees the actionable error (fix the input), not the deeper
    ///   "primitive missing" error.
    /// - [`VokraError::UnsupportedOp`] on non-empty input — the
    ///   loud-partial gate documented above.
    pub fn transcribe(&self, pcm: &[f32]) -> Result<String> {
        if pcm.is_empty() {
            return Err(VokraError::InvalidArgument(format!(
                "moonshine ({:?}) transcribe: input PCM buffer is empty \
                 (expected non-empty {} Hz mono f32 raw waveform; an empty \
                 buffer cannot produce a transcript — FR-EX-08, never a \
                 silent empty-string return)",
                self.config.variant, self.config.sample_rate
            )));
        }
        Err(transcribe_forward_loud_partial(&self.config))
    }
}

// ---------------------------------------------------------------------------
// Loud-partial constructor — one per surface point, so an owner (or a
// follow-up CC wave) reading the error message knows exactly where to flip
// the switch. Every message cites all three primary source URLs so no
// searching is required.
// ---------------------------------------------------------------------------

/// Constructs the loud-partial [`VokraError::UnsupportedOp`] returned by
/// [`Moonshine::transcribe`] until the real forward body lands.
///
/// Names the three specific missing pieces (raw-audio Conv1D stem
/// walk, RoPE + SwiGLU transformer encoder-decoder forward, greedy
/// decode + SentencePiece detokenize) plus every primary source URL a
/// reader would need. RMVPE / DNSMOS / snac / wavlm loud-partial-
/// message precedent — one place to walk when the switch gets flipped
/// (CLAUDE.md 教訓 (a)).
fn transcribe_forward_loud_partial(config: &MoonshineConfig) -> VokraError {
    let [s0, s1, s2] = config.encoder_conv_strides;
    let downsample = s0 * s1 * s2;
    VokraError::UnsupportedOp(format!(
        "moonshine ({:?}) transcribe: real Moonshine forward is a follow-up \
         WP — none of the required primitives are wired in `vokra-models` \
         today. Missing: (i) raw-audio Conv1D stem walk (strides = \
         [{s0}, {s1}, {s2}], {downsample}x downsampling, NO mel front-end — \
         this is what distinguishes Moonshine from every Whisper-family \
         sibling ASR (whisper / distil_whisper / kotoba_whisper), which all \
         key on STFT + Mel filterbank input; upstream reference \
         `github.com/usefulsensors/moonshine/blob/main/moonshine/model.py`); \
         (ii) RoPE + SwiGLU transformer encoder ({} layers × {} heads) + \
         decoder ({} layers × {} heads) forward (distinct from Whisper's \
         sinusoidal position embeddings + GELU activations — upstream \
         reference same file); (iii) greedy / beam decoding + SentencePiece \
         detokenize ({}-piece vocab). Primary sources: \
         https://github.com/usefulsensors/moonshine + \
         https://arxiv.org/abs/2410.15608 + \
         https://huggingface.co/{}. Loud pending (CLAUDE.md 教訓 (a) — \
         'loud-partial は fake-complete より honest') — no silent \
         fabricated transcript ever emitted (FR-EX-08).",
        config.variant,
        config.n_encoder_layers,
        config.encoder_num_heads,
        config.n_decoder_layers,
        config.decoder_num_heads,
        config.vocab_size,
        config.variant.upstream_hf(),
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::{GgmlType, GgufBuilder};

    // -----------------------------------------------------------------------
    // Fixture helpers — hand-assembled GGUFs (bypass the converter for
    // isolation; the converter e2e lives in
    // `crates/vokra-convert/src/models/moonshine_{tiny,base}.rs::tests`).
    // -----------------------------------------------------------------------

    /// Builds a minimal Moonshine GGUF carrying the arch tag + variant
    /// name + provenance stamp — the same three chunks every real
    /// converter output carries. `weight_license_class` is written
    /// under `vokra.provenance.weight_license` (or omitted if `None`).
    fn moonshine_gguf_for(name_tag: &str, weight_license_class: Option<LicenseClass>) -> GgufFile {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, name_tag);
        if let Some(cls) = weight_license_class {
            b.add_string(chunks::KEY_PROVENANCE_WEIGHT_LICENSE, cls.as_str());
        }
        // Stamp a defensive extra tensor so a downstream reader that
        // walks tensors on a real-weight GGUF does not accidentally
        // short-circuit on an empty-file heuristic. Value arbitrary
        // — no primitive today consumes it.
        b.add_tensor(
            "encoder.audio_conv.weight",
            GgmlType::F32,
            vec![2, 3],
            vec![0u8; 2 * 3 * 4],
        )
        .expect("add_tensor");
        GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse")
    }

    // -----------------------------------------------------------------------
    // Variant enum round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn variant_name_roundtrip() {
        // Every variant round-trips through as_name / from_name.
        assert_eq!(
            MoonshineVariant::from_name(MoonshineVariant::Tiny.as_name()),
            Some(MoonshineVariant::Tiny)
        );
        assert_eq!(
            MoonshineVariant::from_name(MoonshineVariant::Base.as_name()),
            Some(MoonshineVariant::Base)
        );
        // Silently sharing the tag across variants would break the runtime
        // dispatch — pin the tags are distinct.
        assert_ne!(
            MoonshineVariant::Tiny.as_name(),
            MoonshineVariant::Base.as_name()
        );
        // Unknown tag → None (fail-closed; from_gguf turns this into a
        // loud ModelLoad rather than silently defaulting).
        assert_eq!(MoonshineVariant::from_name("moonshine-large"), None);
        assert_eq!(MoonshineVariant::from_name(""), None);
        // upstream_hf is variant-distinct too.
        assert_ne!(
            MoonshineVariant::Tiny.upstream_hf(),
            MoonshineVariant::Base.upstream_hf()
        );
    }

    // -----------------------------------------------------------------------
    // Config axes — Tiny vs Base differ where they must, share sample_rate
    // -----------------------------------------------------------------------

    #[test]
    fn tiny_and_base_configs_differ_but_share_sample_rate() {
        let tiny = MoonshineConfig::tiny();
        let base = MoonshineConfig::base();

        // Same arch family — sample rate + FFN multiplier + conv strides
        // + vocab shared.
        assert_eq!(tiny.sample_rate, MOONSHINE_SAMPLE_RATE);
        assert_eq!(base.sample_rate, MOONSHINE_SAMPLE_RATE);
        assert_eq!(tiny.sample_rate, base.sample_rate);
        assert_eq!(tiny.ffn_multiplier, 4);
        assert_eq!(base.ffn_multiplier, 4);
        assert_eq!(tiny.encoder_conv_strides, [64, 3, 2]);
        assert_eq!(base.encoder_conv_strides, [64, 3, 2]);
        assert_eq!(tiny.vocab_size, 32_768);
        assert_eq!(base.vocab_size, 32_768);

        // Depth / width differ — Base is the wider / deeper sibling.
        assert!(
            base.hidden_size > tiny.hidden_size,
            "Base hidden_size must be > Tiny (got Tiny={}, Base={})",
            tiny.hidden_size,
            base.hidden_size
        );
        assert!(
            base.n_encoder_layers > tiny.n_encoder_layers,
            "Base must have more encoder layers than Tiny"
        );
        assert!(
            base.n_decoder_layers > tiny.n_decoder_layers,
            "Base must have more decoder layers than Tiny"
        );
        assert!(
            base.encoder_num_heads > tiny.encoder_num_heads,
            "Base must have more encoder heads than Tiny"
        );

        // Variant round-trip via for_variant.
        assert_eq!(MoonshineConfig::for_variant(MoonshineVariant::Tiny), tiny);
        assert_eq!(MoonshineConfig::for_variant(MoonshineVariant::Base), base);

        // Both configs are well-formed (heads divide hidden_size).
        tiny.validate().expect("tiny config must validate");
        base.validate().expect("base config must validate");
    }

    // -----------------------------------------------------------------------
    // Loud-error round-trip — arch / variant validation (FR-EX-08)
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_wrong_arch_is_loud() {
        // A Whisper GGUF handed to the Moonshine binder by mistake must
        // fail loud with a specific message rather than silently
        // mis-binding — and the sibling-arch hint list must guide the
        // reader.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, "whisper");
        b.add_string(chunks::KEY_MODEL_NAME, NAME_TAG_TINY);
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = Moonshine::from_gguf(&file) else {
            panic!("wrong arch must be rejected");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("`whisper`") && m.contains("`moonshine`"),
                    "message must name both the got and expected arch tags, got `{m}`"
                );
                // The message must forward the reader to the primary source
                // so they can walk the arch family without cross-referencing
                // rustdoc.
                assert!(
                    m.contains("github.com/usefulsensors/moonshine"),
                    "message must cite the primary source URL, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    #[test]
    fn from_gguf_missing_arch_is_loud() {
        // A GGUF with no `vokra.model.arch` at all — a converter that
        // forgot to stamp it must be caught here, not surface as a
        // downstream "missing tensor".
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_NAME, NAME_TAG_TINY);
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = Moonshine::from_gguf(&file) else {
            panic!("missing arch must be rejected");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("vokra.model.arch"),
                    "message must name the missing arch key, got `{m}`"
                );
                assert!(
                    m.contains("github.com/usefulsensors/moonshine"),
                    "message must cite the primary source URL, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    #[test]
    fn from_gguf_unknown_variant_is_loud() {
        // A rogue converter or a future 3rd variant this runtime does
        // not dispatch on — never a silent default to Tiny.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, "moonshine-large"); // not a real variant
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = Moonshine::from_gguf(&file) else {
            panic!("unknown variant must be rejected");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("`moonshine-large`"),
                    "message must echo the bad tag, got `{m}`"
                );
                // Both accepted tags MUST appear in the hint so the
                // reader can pick the correct one without cross-
                // referencing rustdoc.
                assert!(
                    m.contains(NAME_TAG_TINY),
                    "message must list moonshine-tiny as an accepted tag, got `{m}`"
                );
                assert!(
                    m.contains(NAME_TAG_BASE),
                    "message must list moonshine-base as an accepted tag, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    #[test]
    fn from_gguf_tiny_roundtrip_binds_tiny_config() {
        let file = moonshine_gguf_for(NAME_TAG_TINY, Some(LicenseClass::Permissive));
        let m = Moonshine::from_gguf(&file).expect("Tiny GGUF must bind");
        assert_eq!(m.variant(), MoonshineVariant::Tiny);
        let cfg = m.config();
        assert_eq!(cfg.variant, MoonshineVariant::Tiny);
        // Pin the Tiny hparams — every one is transcribed from the
        // primary source; a silent drift in either the converter or
        // this binder would fail here.
        assert_eq!(cfg.hidden_size, 288);
        assert_eq!(cfg.n_encoder_layers, 6);
        assert_eq!(cfg.n_decoder_layers, 6);
        assert_eq!(cfg.encoder_num_heads, 6);
        assert_eq!(cfg.decoder_num_heads, 6);
        assert_eq!(cfg.ffn_multiplier, 4);
        assert_eq!(cfg.encoder_conv_strides, [64, 3, 2]);
        assert_eq!(cfg.vocab_size, 32_768);
        assert_eq!(cfg.sample_rate, MOONSHINE_SAMPLE_RATE);
        assert_eq!(m.weight_license(), LicenseClass::Permissive);
    }

    #[test]
    fn from_gguf_base_roundtrip_binds_base_config() {
        let file = moonshine_gguf_for(NAME_TAG_BASE, Some(LicenseClass::Permissive));
        let m = Moonshine::from_gguf(&file).expect("Base GGUF must bind");
        assert_eq!(m.variant(), MoonshineVariant::Base);
        let cfg = m.config();
        assert_eq!(cfg.variant, MoonshineVariant::Base);
        // Pin the Base hparams — same primary-source-transcription rule
        // as the Tiny test above; drift = loud test failure.
        assert_eq!(cfg.hidden_size, 416);
        assert_eq!(cfg.n_encoder_layers, 8);
        assert_eq!(cfg.n_decoder_layers, 8);
        assert_eq!(cfg.encoder_num_heads, 8);
        assert_eq!(cfg.decoder_num_heads, 8);
        assert_eq!(cfg.ffn_multiplier, 4);
        assert_eq!(cfg.encoder_conv_strides, [64, 3, 2]);
        assert_eq!(cfg.vocab_size, 32_768);
        assert_eq!(cfg.sample_rate, MOONSHINE_SAMPLE_RATE);
        assert_eq!(m.weight_license(), LicenseClass::Permissive);
    }

    #[test]
    fn from_gguf_defaults_weight_license_to_unknown_when_missing() {
        // A GGUF missing `vokra.provenance.weight_license` reads back as
        // `Unknown` (fail-closed at the compliance gate). Never a silent
        // Permissive default.
        let file = moonshine_gguf_for(NAME_TAG_TINY, None);
        let m = Moonshine::from_gguf(&file).expect("missing provenance must still bind");
        assert_eq!(m.weight_license(), LicenseClass::Unknown);
    }

    // -----------------------------------------------------------------------
    // Loud-partial round-trip — transcribe fires at its documented surface
    // point with the documented content. A silent stub swap (replacing the
    // loud gate with an `Ok(String::new())` return) would break these
    // tests immediately.
    // -----------------------------------------------------------------------

    #[test]
    fn transcribe_empty_pcm_is_invalid_argument() {
        // Empty PCM cannot produce a transcript; the caller sees the
        // actionable InvalidArgument (fix the input), not the deeper
        // loud-partial gate (which they can't fix at all).
        let file = moonshine_gguf_for(NAME_TAG_TINY, Some(LicenseClass::Permissive));
        let m = Moonshine::from_gguf(&file).unwrap();
        let Err(err) = m.transcribe(&[]) else {
            panic!("empty pcm must be rejected");
        };
        match err {
            VokraError::InvalidArgument(msg) => {
                assert!(
                    msg.contains("empty"),
                    "message must call out the empty PCM, got `{msg}`"
                );
                assert!(
                    msg.contains("16000"),
                    "message must name the expected sample rate, got `{msg}`"
                );
            }
            other => panic!("expected VokraError::InvalidArgument, got {other:?}"),
        }
    }

    #[test]
    fn transcribe_nonempty_returns_unsupported_op_with_primary_source_urls() {
        // A well-shaped non-empty input fires the loud-partial gate.
        // The gate message must cite ALL three primary source URLs and
        // call out the Moonshine-distinguishing "no mel front-end"
        // trait so the follow-up wave knows exactly where to look.
        let file = moonshine_gguf_for(NAME_TAG_TINY, Some(LicenseClass::Permissive));
        let m = Moonshine::from_gguf(&file).unwrap();
        // 1 s of silence at 16 kHz — legitimate input shape, so the
        // loud-partial gate is what fires (not the empty-buffer guard).
        let pcm = vec![0.0f32; 16_000];
        let Err(err) = m.transcribe(&pcm) else {
            panic!("non-empty pcm must loud-partial");
        };
        match err {
            VokraError::UnsupportedOp(msg) => {
                // The three primary source URLs must all appear so a
                // follow-up wave has one place to walk.
                assert!(
                    msg.contains("github.com/usefulsensors/moonshine"),
                    "message must cite the upstream GitHub source, got `{msg}`"
                );
                assert!(
                    msg.contains("2410.15608"),
                    "message must cite the arXiv paper (2410.15608), got `{msg}`"
                );
                assert!(
                    msg.contains("huggingface.co/UsefulSensors/moonshine-tiny"),
                    "message must cite the HF repo card, got `{msg}`"
                );
                // The Moonshine-distinguishing trait (no mel front-end)
                // must be called out — a follow-up wave landing a mel
                // front-end here would silently misroute against every
                // Whisper-family sibling.
                assert!(
                    msg.contains("NO mel front-end"),
                    "message must call out the 'no mel front-end' Moonshine \
                     distinguishing trait, got `{msg}`"
                );
                // The conv-stem strides must be echoed (Moonshine's
                // 384x downsampling anchor).
                assert!(
                    msg.contains("[64, 3, 2]"),
                    "message must cite the conv-stem strides, got `{msg}`"
                );
                assert!(
                    msg.contains("384x"),
                    "message must cite the derived 384x downsampling factor, got `{msg}`"
                );
                // The three missing pieces must all be named (i, ii, iii).
                assert!(
                    msg.contains("(i)") && msg.contains("(ii)") && msg.contains("(iii)"),
                    "message must enumerate the three missing pieces, got `{msg}`"
                );
                // The three distinct primitives must all be called out.
                assert!(
                    msg.contains("Conv1D"),
                    "message must name the Conv1D stem gap, got `{msg}`"
                );
                assert!(
                    msg.contains("RoPE") && msg.contains("SwiGLU"),
                    "message must name RoPE + SwiGLU transformer gap, got `{msg}`"
                );
                assert!(
                    msg.contains("SentencePiece"),
                    "message must name the SentencePiece detokenize gap, got `{msg}`"
                );
            }
            other => panic!("expected VokraError::UnsupportedOp, got {other:?}"),
        }
    }

    #[test]
    fn transcribe_base_variant_message_names_base_upstream() {
        // The loud-partial message must cite the Base HF repo when the
        // binder loaded the Base variant — otherwise a Base checkpoint
        // owner following the message would land on the Tiny repo page
        // and get confused.
        let file = moonshine_gguf_for(NAME_TAG_BASE, Some(LicenseClass::Permissive));
        let m = Moonshine::from_gguf(&file).unwrap();
        let pcm = vec![0.0f32; 16_000];
        let Err(err) = m.transcribe(&pcm) else {
            panic!("non-empty pcm must loud-partial");
        };
        match err {
            VokraError::UnsupportedOp(msg) => {
                assert!(
                    msg.contains("huggingface.co/UsefulSensors/moonshine-base"),
                    "message must cite the Base HF repo card, got `{msg}`"
                );
                // Base per-variant axes echo through: 8 encoder + 8
                // decoder layers, 8 heads each.
                assert!(
                    msg.contains("8 layers"),
                    "message must echo the Base layer count (8), got `{msg}`"
                );
                assert!(
                    msg.contains("8 heads"),
                    "message must echo the Base head count (8), got `{msg}`"
                );
            }
            other => panic!("expected VokraError::UnsupportedOp, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Config validation — well-formedness gate
    // -----------------------------------------------------------------------

    #[test]
    fn config_validate_rejects_zero_axes() {
        // Zero-fill any single axis and validate must catch it. Iterate
        // through every field so a future axis addition can't silently
        // slip past the gate.
        let bad_hidden = MoonshineConfig {
            hidden_size: 0,
            ..MoonshineConfig::tiny()
        };
        assert!(bad_hidden.validate().is_err());

        let bad_enc_layers = MoonshineConfig {
            n_encoder_layers: 0,
            ..MoonshineConfig::tiny()
        };
        assert!(bad_enc_layers.validate().is_err());

        let bad_dec_layers = MoonshineConfig {
            n_decoder_layers: 0,
            ..MoonshineConfig::tiny()
        };
        assert!(bad_dec_layers.validate().is_err());

        let bad_enc_heads = MoonshineConfig {
            encoder_num_heads: 0,
            ..MoonshineConfig::tiny()
        };
        assert!(bad_enc_heads.validate().is_err());

        let bad_dec_heads = MoonshineConfig {
            decoder_num_heads: 0,
            ..MoonshineConfig::tiny()
        };
        assert!(bad_dec_heads.validate().is_err());

        let bad_ffn_mult = MoonshineConfig {
            ffn_multiplier: 0,
            ..MoonshineConfig::tiny()
        };
        assert!(bad_ffn_mult.validate().is_err());

        let bad_vocab = MoonshineConfig {
            vocab_size: 0,
            ..MoonshineConfig::tiny()
        };
        assert!(bad_vocab.validate().is_err());

        let bad_sample_rate = MoonshineConfig {
            sample_rate: 0,
            ..MoonshineConfig::tiny()
        };
        assert!(bad_sample_rate.validate().is_err());

        let bad_stride = MoonshineConfig {
            encoder_conv_strides: [64, 0, 2],
            ..MoonshineConfig::tiny()
        };
        assert!(bad_stride.validate().is_err());
    }

    #[test]
    fn config_validate_rejects_head_split_mismatch() {
        // MHA algebraic constraint: hidden_size must divide evenly by
        // the head count. A poorly-shaped fixture would silently corrupt
        // the per-head slice arithmetic; the validate gate catches this
        // before any forward runs.
        let bad_enc = MoonshineConfig {
            hidden_size: 288, // divisible by 6 but NOT by 5
            encoder_num_heads: 5,
            ..MoonshineConfig::tiny()
        };
        assert!(bad_enc.validate().is_err());

        let bad_dec = MoonshineConfig {
            hidden_size: 288,
            decoder_num_heads: 7, // 288 % 7 != 0
            ..MoonshineConfig::tiny()
        };
        assert!(bad_dec.validate().is_err());
    }
}
